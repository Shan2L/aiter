# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx1201 INT4 attention for MiniMax-H3 (lossy, opt-in): INT4 Q.K^T and FP8 P.V with ASM cores.

Preparation (HIP, ``module_gfx1201_int4_attention``): Q and K are rotated by the normalized 128x128
Walsh-Hadamard matrix (K minus its per-head mean first; neither changes softmax(Q K^T)) and quantized to INT4 per
(head, 32 rows) with scale ``0.8 * amax / 7``; V is FP8 e4m3, transposed, with the per-channel scale
``amax|v| / 448`` (the ``prepare_sage`` V layout).

Cores (``hsa/gfx1201/int4_attention/attention_int4_<core>.hsaco``, kernel ``sage_hip_attn_bm128_bn32``): the
P.V product of a group of 8 keys is skipped when the group's P sum (P scaled to a maximum of 448) is below a
threshold. ``th4``: threshold 4; ``t2``: threshold 2^-2 (more accurate, slower); the ``f`` variants (``th4f``,
``t2f``) convert the integer scores with a fast path folded into the FMAs where the tile scale is below 0.4.
"""

import ctypes
import os

import torch
from torch import Tensor

from aiter.jit.core import AITER_ASM_DIR, compile_ops

CORES = ("th4", "t2", "th4f", "t2f")
KV_ROWS = 128
_runtime = None
_modules = {}


@compile_ops("module_gfx1201_int4_attention", fc_name="gfx1201_int4_qk_norm_rope_hip", develop=True)
def gfx1201_int4_qk_norm_rope_hip(
    q: Tensor,
    k: Tensor,
    q_weight: Tensor,
    k_weight: Tensor,
    cosine: Tensor,
    sine: Tensor,
    q_out: Tensor,
    k_out: Tensor,
) -> None:
    """QK RMSNorm + RoPE (BF16 -> BF16); q_out/k_out may alias q/k."""


@compile_ops("module_gfx1201_int4_attention", fc_name="gfx1201_int4_prepare_hip", develop=True)
def gfx1201_int4_prepare_hip(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    part: Tensor,
    kmean: Tensor,
    q4: Tensor,
    k4: Tensor,
    qs: Tensor,
    ks: Tensor,
    v8: Tensor,
    vs: Tensor,
    qmul: float,
) -> None:
    """Write the INT4 Q/K and FP8 V^T operands of the cores."""


def _check_qkv(*tensors):
    rows, heads = tensors[0].shape[0], tensors[0].shape[1]
    for t in tensors:
        if t.shape != (rows, heads, 128) or t.dtype != torch.bfloat16 or not t.is_contiguous():
            raise ValueError(f"Expected contiguous BF16 [S,H,128] tensors, got {tuple(t.shape)} {t.dtype}")
        if t.device != tensors[0].device:
            raise ValueError("Expected tensors on one device")
    if rows <= 0 or heads <= 0 or tensors[0].numel() >= 2**31:
        raise ValueError("Expected positive shape within the 32-bit index range")
    return rows, heads


def qk_norm_rope(query, key, query_weight, key_weight, cosine, sine, query_out=None, key_out=None):
    """QK RMSNorm (eps 1e-5) + RoPE on channels 0..95 (rotate-half, pairs c / c+48) of BF16 ``[S, H, 128]``
    query/key; norm weights BF16 ``[128]``, cosine/sine FP32 ``[S, 96]``. Rows are processed independently, so
    ``query_out=query, key_out=key`` runs in place. Returns ``(query_out, key_out)``."""
    rows, _ = _check_qkv(query, key)
    if any(t.shape != (128,) or t.dtype != torch.bfloat16 or not t.is_contiguous() for t in (query_weight, key_weight)):
        raise ValueError("Expected contiguous BF16 norm weights [128]")
    if any(t.shape != (rows, 96) or t.dtype != torch.float32 or not t.is_contiguous() for t in (cosine, sine)):
        raise ValueError("Expected contiguous FP32 cosine/sine [S,96]")
    query_out = torch.empty_like(query) if query_out is None else query_out
    key_out = torch.empty_like(key) if key_out is None else key_out
    _check_qkv(query, query_out, key_out)
    gfx1201_int4_qk_norm_rope_hip(query, key, query_weight, key_weight, cosine, sine, query_out, key_out)
    return query_out, key_out


def int4_prepare(query, key, value):
    """BF16 ``[S, H, 128]`` Q/K/V (after QK norm and RoPE) -> ``(q_int4 [1,Sp,H,128] int8, q_scale [1,H,Sp/32],
    k_int4, k_scale, v_fp8 [1,H,128,Sp], v_scale [1,H,128])`` for :func:`launch_int4_core`, Sp = S rounded up to
    32. INT4 values are packed in bytes 0..63 of each row; q_scale includes softmax scale * log2(e)."""
    rows, heads = _check_qkv(query, key, value)
    padded, device = (rows + 31) // 32 * 32, query.device
    part = torch.empty((2, (rows + KV_ROWS - 1) // KV_ROWS, heads, 128), dtype=torch.float32, device=device)
    kmean = torch.empty((heads, 128), dtype=torch.float32, device=device)
    q4 = torch.empty((1, padded, heads, 128), dtype=torch.int8, device=device)
    k4 = torch.empty_like(q4)
    qs = torch.empty((1, heads, padded // 32), dtype=torch.float32, device=device)
    ks = torch.empty_like(qs)
    v8 = torch.empty((1, heads, 128, padded), dtype=torch.float8_e4m3fn, device=device)
    vs = torch.empty((1, heads, 128), dtype=torch.float32, device=device)
    gfx1201_int4_prepare_hip(query, key, value, part, kmean, q4, k4, qs, ks, v8, vs, 128**-0.5 * 1.4426950408889634)
    return q4, qs, k4, ks, v8, vs


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
    code_object = os.path.join(AITER_ASM_DIR, "gfx1201", "int4_attention", f"attention_int4_{core}.hsaco")
    key = (torch.cuda.current_device(), code_object)
    if key not in _modules:
        module = ctypes.c_void_p()
        function = ctypes.c_void_p()
        _check(_runtime.hipModuleLoad(ctypes.byref(module), os.fsencode(key[1])))
        _check(_runtime.hipModuleGetFunction(ctypes.byref(function), module, b"sage_hip_attn_bm128_bn32"))
        _modules[key] = (module, function)
    return _modules[key][1]


def launch_int4_core(core, q_int4, k_int4, v_fp8, q_scale, k_scale, v_scale, out, batch_size, padded_seq_len,
                     valid_seq_len, num_heads):
    """Run core ``core`` (one of ``CORES``) on :func:`int4_prepare` operands; out: BF16 [1, Sp, H, 128]."""
    if core not in CORES:
        raise ValueError(f"Unknown INT4 core {core!r}, expected one of {CORES}")
    if batch_size != 1 or min(valid_seq_len, num_heads) <= 0 or padded_seq_len != (valid_seq_len + 31) // 32 * 32:
        raise ValueError("Expected batch 1, positive shape and sequence padded to a multiple of 32")
    if out.dtype != torch.bfloat16 or out.device != q_int4.device or out.shape != (1, padded_seq_len, num_heads, 128):
        raise ValueError("Expected BF16 output [1, Sp, H, 128] on the input device")
    with torch.cuda.device(q_int4.device):
        function = _function(core)
        arguments = [ctypes.c_void_p(t.data_ptr()) for t in (q_int4, k_int4, v_fp8, q_scale, k_scale, v_scale, out)]
        arguments += [ctypes.c_int(x) for x in (batch_size, padded_seq_len, valid_seq_len, num_heads)]
        parameters = (ctypes.c_void_p * len(arguments))(*[ctypes.addressof(a) for a in arguments])
        grid = (valid_seq_len + 255) // 256 * num_heads
        _check(
            _runtime.hipModuleLaunchKernel(
                function, grid, 1, 1, 256, 1, 1, 0,
                ctypes.c_void_p(torch.cuda.current_stream().cuda_stream), parameters, None,
            )
        )


def int4_attention(query, key, value, core="th4f"):
    """softmax(Q K^T / sqrt(128)) V for BF16 ``[S, H, 128]`` Q/K/V (after QK norm and RoPE) -> BF16 ``[S, H, 128]``."""
    rows, heads = query.shape[0], query.shape[1]
    q4, qs, k4, ks, v8, vs = int4_prepare(query, key, value)
    out = torch.empty((1, q4.shape[1], heads, 128), dtype=torch.bfloat16, device=query.device)
    launch_int4_core(core, q4, k4, v8, qs, ks, vs, out, 1, q4.shape[1], rows, heads)
    return out[0, :rows]
