# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx1201 Sol sparse attention: Sol-Attn (arXiv 2607.24027) block routing on top of gfx1201_sage_attention.

Per 64-row query block, key block ``j`` (64 rows) is computed exactly if
``qc . kc_j > qc . mu + tau * sqrt(sum(qc^2 * var) + 1e-6)`` (``qc``/``kc`` block-mean query/key in
log2 softmax units, ``mu``/``var`` per-dimension mean/variance of ``kc`` over all key blocks; Sol-Attn
``thresh_type="diag"``), if ``|qb - j| <= 1``, if ``j`` overlaps the sink tokens
``[sink_start, sink_start + sink_tokens)`` (Sol-Attn ``sink_tokens`` / ``sink_start``; ``sink_start=None``
places them at the end of the sequence), or if ``j`` is the last block. Each 512-row workgroup computes the union of its eight query
blocks' selections exactly; the remaining blocks enter through the Sol-Attn block-summary correction
(logit ``q . kc_j + log2(len_j)``, value ``vc_j / len_j``) on INT8 pooled keys and FP8 value sums.
Larger ``tau`` is sparser; with every block selected the result is bitwise equal to the dense
``gfx1201_sage_attention_fwd`` with the same ``backend``.
"""

import torch
from torch import Tensor

from csrc.cpp_itfs.torch_utils import direct_register_custom_op

from ..jit.core import compile_ops
from .gfx1201_sage_attention import _BACKENDS, _check
from .gfx1201_sage_prepare import gfx1201_sage_prepare

_KEY_BLOCK = 64
_WORKGROUP_ROWS = 512
_MAX_KEY_BLOCKS = 4096
_STATE_SHAPE = (16, 33, 32, 4)  # per workgroup: waves x (32 O chunks + l/m) x lanes x 4


@compile_ops("module_gfx1201_sage_attention", develop=True)
def gfx1201_sol_route_hip(
    q_int8: Tensor,
    q_scale: Tensor,
    k_int8: Tensor,
    k_scale: Tensor,
    v_fp8: Tensor,
    v_scale: Tensor,
    kbar: Tensor,
    vsum: Tensor,
    kb: Tensor,
    kb_scale: Tensor,
    vb: Tensor,
    vb_scale: Tensor,
    mu: Tensor,
    var: Tensor,
    tile_list: Tensor,
    tile_count: Tensor,
    mask: Tensor,
    seq_len: int,
    sink_start: int,
    sink_tokens: int,
    tau: float,
) -> None: ...


@compile_ops("module_gfx1201_sage_attention", develop=True)
def gfx1201_sol_phase_a_hip(
    q_int8: Tensor,
    q_scale: Tensor,
    v_scale: Tensor,
    kb: Tensor,
    vb: Tensor,
    kb_scale: Tensor,
    vb_scale: Tensor,
    block_len: Tensor,
    mask: Tensor,
    state: Tensor,
    seq_len: int,
) -> None: ...


def _sink_start(seq_len: int, sink_tokens: int, sink_start: int | None) -> int:
    if not 0 <= sink_tokens <= seq_len:
        raise ValueError(f"sink_tokens must be in [0, {seq_len}], got {sink_tokens}")
    if sink_start is None:
        return seq_len - sink_tokens
    if not 0 <= sink_start <= seq_len - sink_tokens:
        raise ValueError(
            f"sink_start + sink_tokens must lie in [0, {seq_len}], got {sink_start} + {sink_tokens}"
        )
    return sink_start


def gfx1201_sol_route(
    q_int8: Tensor,
    q_scale: Tensor,
    k_int8: Tensor,
    k_scale: Tensor,
    v_fp8: Tensor,
    v_scale: Tensor,
    seq_len: int,
    tau: float = 1.0,
    sink_tokens: int = 0,
    sink_start: int | None = None,
) -> dict:
    """Route one batch-1 ``gfx1201_sage_prepare`` output.

    Returns the plan: ``tile_count`` ``[W]`` (exact 32-key tiles per workgroup, W = H * ceil(S / 512)
    numbered ``head * q_tiles + q_tile``), ``tile_list`` ``[W, 2 * nkb + 2]``, ``mask``
    ``[W, nkbp / 32]`` (bit set = exact 64-key block) and the pooled operands for Phase A.
    """
    batch, padded, heads, _ = q_int8.shape
    if batch != 1 or not 0 < seq_len <= padded or padded % 32:
        raise ValueError("Expected a batch-1 prepared input padded to a multiple of 32")
    sink_start = _sink_start(seq_len, sink_tokens, sink_start)
    nkb = (seq_len + _KEY_BLOCK - 1) // _KEY_BLOCK
    nkbp = (nkb + 31) // 32 * 32
    if nkbp > _MAX_KEY_BLOCKS:
        raise ValueError(
            f"gfx1201_sol_attention supports up to {_MAX_KEY_BLOCKS * _KEY_BLOCK} tokens"
        )
    n_wg = heads * ((seq_len + _WORKGROUP_ROWS - 1) // _WORKGROUP_ROWS)
    device = q_int8.device

    def f32(*shape):
        return torch.empty(shape, device=device, dtype=torch.float32)

    plan = {
        "nkb": nkb,
        "n_wg": n_wg,
        "kbar": f32(heads, nkbp, 128),
        "vsum": f32(heads, 128, nkbp),
        "kb": torch.empty(nkbp, heads, 128, device=device, dtype=torch.int8),
        "kb_scale": f32(heads, nkbp // 32),
        "vb": torch.empty(heads, 128, nkbp, device=device, dtype=torch.uint8),
        "vb_scale": f32(heads, 128),
        "mu": f32(heads, 128),
        "var": f32(heads, 128),
        "tile_list": torch.empty(n_wg, 2 * nkb + 2, device=device, dtype=torch.int32),
        "tile_count": torch.empty(n_wg, device=device, dtype=torch.int32),
        "mask": torch.empty(n_wg, nkbp // 32, device=device, dtype=torch.int32),
    }
    gfx1201_sol_route_hip(
        q_int8,
        q_scale,
        k_int8,
        k_scale,
        v_fp8,
        v_scale,
        *(
            plan[name]
            for name in (
                "kbar",
                "vsum",
                "kb",
                "kb_scale",
                "vb",
                "vb_scale",
                "mu",
                "var",
                "tile_list",
                "tile_count",
                "mask",
            )
        ),
        seq_len,
        sink_start,
        sink_tokens,
        float(tau),
    )
    return plan


def gfx1201_sol_attention_fwd(
    q_int8: Tensor,
    q_scale: Tensor,
    k_int8: Tensor,
    k_scale: Tensor,
    v_fp8: Tensor,
    v_scale: Tensor,
    seq_len: int,
    tau: float = 1.0,
    sink_tokens: int = 0,
    sink_start: int | None = None,
    backend: str = "asm",
) -> Tensor:
    """Sol attention over the tensors returned by ``gfx1201_sage_prepare``.

    ``backend`` selects the kernel that visits the listed tiles: ``"asm"`` the list variant of the
    SageAttention code object, ``"hip"`` the HIP kernel (same numerics difference as in
    ``gfx1201_sage_attention_fwd``; routing and phase A are shared).

    Returns BF16 ``[B, S_pad, H, 128]``; rows at or beyond ``seq_len`` are left uninitialized.
    """
    if backend not in _BACKENDS:
        raise ValueError(f"backend must be one of {sorted(_BACKENDS)}, got {backend!r}")
    batch, padded, heads, dim = q_int8.shape
    out = torch.empty(
        (batch, padded, heads, dim), device=q_int8.device, dtype=torch.bfloat16
    )
    for b in range(batch):
        q8, qs, k8, ks, v8, vs = (
            x[b : b + 1] for x in (q_int8, q_scale, k_int8, k_scale, v_fp8, v_scale)
        )
        plan = gfx1201_sol_route(
            q8, qs, k8, ks, v8, vs, seq_len, tau, sink_tokens, sink_start
        )
        nkb = plan["nkb"]
        block_len = torch.zeros(plan["kb"].shape[0], device=q_int8.device)
        block_len[:nkb] = _KEY_BLOCK
        block_len[nkb - 1] = seq_len - (nkb - 1) * _KEY_BLOCK
        state = torch.empty(
            (plan["n_wg"], *_STATE_SHAPE), device=q_int8.device, dtype=torch.float32
        )
        gfx1201_sol_phase_a_hip(
            q8,
            qs,
            vs,
            plan["kb"],
            plan["vb"],
            plan["kb_scale"],
            plan["vb_scale"],
            block_len,
            plan["mask"],
            state,
            seq_len,
        )
        _BACKENDS[backend](
            q8,
            qs,
            k8,
            ks,
            v8,
            vs,
            out[b : b + 1],
            seq_len,
            plan["tile_list"],
            plan["tile_count"],
            state,
        )
    return out


def gfx1201_sol_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    tau: float = 1.0,
    sink_tokens: int = 0,
    sink_start: int | None = None,
    softmax_scale: float | None = None,
    query_weight: Tensor | None = None,
    key_weight: Tensor | None = None,
    cosine: Tensor | None = None,
    sine: Tensor | None = None,
    eps: float = 1e-6,
    backend: str = "asm",
) -> Tensor:
    """Non-causal Sol sparse attention for gfx1201 on contiguous BF16 ``[B, S, H, 128]`` Q/K/V.

    Lossy: SageAttention quantization plus the Sol-Attn approximation of the unselected key blocks
    (see the module docstring). ``tau``: routing threshold in standard deviations (larger is
    sparser); ``sink_tokens`` / ``sink_start``: key tokens that every query attends to exactly
    (``sink_start=None``: the last ``sink_tokens``). The norm/RoPE arguments are applied to Q and K
    first, see ``gfx1201_sage_prepare``. ``backend`` selects the kernel, see
    ``gfx1201_sol_attention_fwd``.
    """
    _check(query, key, value)
    with torch.cuda.device(query.device):
        prepared = gfx1201_sage_prepare(
            query,
            key,
            value,
            softmax_scale,
            query_weight,
            key_weight,
            cosine,
            sine,
            eps,
        )
        out = gfx1201_sol_attention_fwd(
            *prepared, query.shape[1], tau, sink_tokens, sink_start, backend
        )
    return out[:, : query.shape[1]].contiguous()


def _gfx1201_sol_attention_fake(
    query,
    key,
    value,
    tau=1.0,
    sink_tokens=0,
    sink_start=None,
    softmax_scale=None,
    query_weight=None,
    key_weight=None,
    cosine=None,
    sine=None,
    eps=1e-6,
    backend="asm",
):
    return torch.empty_like(query)


direct_register_custom_op(
    op_name="gfx1201_sol_attention",
    op_func=gfx1201_sol_attention,
    mutates_args=[],
    fake_impl=_gfx1201_sol_attention_fake,
)
