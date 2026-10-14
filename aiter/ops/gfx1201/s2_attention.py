# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx1201 two-stage INT8-QK / FP8-PV attention for MiniMax-H3 (lossy, opt-in): ASM cores in hsa/gfx1201/s2_attention.

Preparation uses either ``norm_rope_prepare_sage`` (fused QK RMSNorm + 3D-RoPE + INT8/FP8 quantization) or
``prepare_sage`` (when QK RMSNorm + RoPE were already applied).

Cores (``hsa/gfx1201/s2_attention/attention_s2_<core>.hsaco``, kernel ``sage_hip_attn_bm128_bn32``, BM=256, BN=32):
- ``s2_4_4``: two-stage QK screening (Stage 1 computes the 6 3D-RoPE WMMA k-steps = 96 dims and skips the remaining
  2 content k-steps, P generation, V LDS loads, O rescale and PV when the tile max is >= 4.0 log2 units below the
  96-dim running max; Stage 2 completes the 128-dim QK and skips P/V/PV at threshold 4.0 log2 units, with subgroup
  PV skip threshold 4.0).
- ``s2_4_7``: same Stage-1 6-step QK screening at 4.0 log2 units, with a tighter Stage-2 whole-tile skip threshold
  of 7.0 log2 units for higher accuracy.
- ``th4f``: full 8-step QK with subgroup PV skip threshold 4.0 and fast int->float score conversion when scale < 0.4.
- ``t2f``: full 8-step QK with subgroup PV skip threshold 0.25 (2^-2) for outlier-sensitive layers (e.g. block 40).
"""

import ctypes
import os

import torch

from aiter.jit.core import AITER_ASM_DIR

CORES = ("s2_4_4", "s2_4_7", "th4f", "t2f")
_runtime = None
_modules = {}


def _check(status):
    if status:
        raise RuntimeError(f"HIP module API failed: {status}")


def _function(core):
    global _runtime
    if _runtime is None:
        _runtime = ctypes.CDLL("libamdhip64.so")
        _runtime.hipModuleLoad.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]
        _runtime.hipModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p]
        _runtime.hipModuleLaunchKernel.argtypes = [ctypes.c_void_p] + [ctypes.c_uint] * 7 + [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
        ]
    code_object = os.path.join(AITER_ASM_DIR, "gfx1201", "s2_attention", f"attention_s2_{core}.hsaco")
    key = (torch.cuda.current_device(), code_object)
    if key not in _modules:
        module = ctypes.c_void_p()
        function = ctypes.c_void_p()
        _check(_runtime.hipModuleLoad(ctypes.byref(module), os.fsencode(key[1])))
        _check(_runtime.hipModuleGetFunction(ctypes.byref(function), module, b"sage_hip_attn_bm128_bn32"))
        _modules[key] = (module, function)
    return _modules[key][1]


def launch_s2_core(core, q_int8, k_int8, v_fp8, q_scale, k_scale, v_scale, out, batch_size, padded_seq_len,
                   valid_seq_len, num_heads):
    """Run core ``core`` (one of ``CORES``) on :func:`norm_rope_prepare_sage` / :func:`prepare_sage` operands."""
    if core not in CORES:
        raise ValueError(f"Unknown S2 core {core!r}, expected one of {CORES}")
    if batch_size != 1 or min(valid_seq_len, num_heads) <= 0 or padded_seq_len != (valid_seq_len + 31) // 32 * 32:
        raise ValueError("Expected batch 1, positive shape and sequence padded to a multiple of 32")
    if out.dtype != torch.bfloat16 or out.device != q_int8.device or out.shape != (1, padded_seq_len, num_heads, 128):
        raise ValueError("Expected BF16 output [1, Sp, H, 128] on the input device")
    with torch.cuda.device(q_int8.device):
        function = _function(core)
        arguments = [ctypes.c_void_p(t.data_ptr()) for t in (q_int8, k_int8, v_fp8, q_scale, k_scale, v_scale, out)]
        arguments += [ctypes.c_int(x) for x in (batch_size, padded_seq_len, valid_seq_len, num_heads)]
        parameters = (ctypes.c_void_p * len(arguments))(*[ctypes.addressof(a) for a in arguments])
        grid = (valid_seq_len + 255) // 256 * num_heads
        _check(
            _runtime.hipModuleLaunchKernel(
                function, grid, 1, 1, 256, 1, 1, 0,
                ctypes.c_void_p(torch.cuda.current_stream().cuda_stream), parameters, None,
            )
        )


def s2_attention(query, key, value, core="s2_4_4"):
    """softmax(Q K^T / sqrt(128)) V for BF16 ``[S, H, 128]`` Q/K/V (after QK norm and RoPE) -> BF16 ``[S, H, 128]``."""
    from .prepare import prepare_sage

    if query.ndim != 3 or query.shape[2] != 128 or key.shape != query.shape or value.shape != query.shape:
        raise ValueError(f"Expected [S, H, 128] Q/K/V, got {tuple(query.shape)}")
    rows, heads = query.shape[0], query.shape[1]
    q8, qs, k8, ks, v8, vs = prepare_sage(
        query.contiguous().unsqueeze(0),
        key.contiguous().unsqueeze(0),
        value.contiguous().unsqueeze(0),
    )
    out = torch.empty((1, q8.shape[1], heads, 128), dtype=torch.bfloat16, device=query.device)
    launch_s2_core(core, q8, k8, v8, qs, ks, vs, out, 1, q8.shape[1], rows, heads)
    return out[0, :rows]
