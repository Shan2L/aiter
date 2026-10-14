# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import math
import sys

import torch
import torch.nn.functional as F

from aiter.jit.utils.chip_info import get_gfx

LOG2E = 1.4426950408889634


def report(name, stats, ok):
    text = " ".join(f"{k}={v:.4g}" for k, v in stats.items())
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {text}")
    return ok


def inputs(batch, rows, heads, seed=11, query_scale=1.0):
    # A few shared directions give the attention structure the router can find.
    g = torch.Generator(device="cuda").manual_seed(seed)
    basis = torch.randn((8, 128), device="cuda", generator=g)

    def tensor(scale):
        mix = torch.randn((batch, rows, heads, 8), device="cuda", generator=g)
        noise = torch.randn((batch, rows, heads, 128), device="cuda", generator=g)
        return ((mix @ basis + 0.5 * noise) * scale).bfloat16()

    return tensor(query_scale), tensor(1.0), tensor(1.0)


def prepared(q, k, v):
    import aiter

    return aiter.gfx1201_sage_prepare(q, k, v)


BACKENDS = ("asm", "hip")


def dense(p, rows, backend="asm"):
    import aiter

    return aiter.gfx1201_sage_attention_fwd(*p, rows, backend=backend)


def sol(p, rows, tau, sink_tokens=0, sink_start=None, backend="asm"):
    import aiter

    return aiter.gfx1201_sol_attention_fwd(
        *p, rows, tau, sink_tokens, sink_start, backend
    )


def plan_mask(plan):
    words = plan["mask"].to(torch.int64) & 0xFFFFFFFF
    bits = (words[:, :, None] >> torch.arange(32, device="cuda")) & 1
    return bits.reshape(words.shape[0], -1)[:, : plan["nkb"]].bool()


def reference_mask(p, rows, tau, sink_tokens=0, sink_start=None):
    """Sol-Attn diag routing + band / sink / last-block policy on the dequantized INT8 Q/K, unioned
    over the eight 64-row query blocks of each 512-row workgroup -> [H * q_tiles, nkb].
    """
    q, qs, k, ks = p[:4]
    heads = q.shape[2]

    def deq(x, s):
        return (
            x[0, :rows].float() * s[0].repeat_interleave(32, 1)[:, :rows].T[:, :, None]
        )

    qf, kf = deq(q, qs), deq(k, ks)
    nb = math.ceil(rows / 64)
    lengths = torch.full((nb,), 64.0, device="cuda")
    lengths[-1] = rows - (nb - 1) * 64

    def pool(x):
        x = F.pad(x, (0, 0, 0, 0, 0, nb * 64 - rows)).view(nb, 64, heads, 128)
        return x.sum(1) / lengths[:, None, None]

    qc, kc = pool(qf), pool(kf)
    mu, var = kc.mean(0), (kc * kc).mean(0) - kc.mean(0) ** 2
    thr = (qc * mu).sum(-1) + tau * torch.sqrt(
        (qc * qc * var).sum(-1).clamp_min(0) + 1e-6
    )
    exact = torch.einsum("qhd,khd->hqk", qc, kc) > thr.T[:, :, None]
    blocks = torch.arange(nb, device="cuda")
    start = rows - sink_tokens if sink_start is None else sink_start
    sink = (blocks >= start // 64) & (blocks < (start + sink_tokens + 63) // 64)
    exact |= ((blocks[:, None] - blocks[None, :]).abs() <= 1)[None]
    if sink_tokens:
        exact |= sink[None, None]
    exact[..., -1] = True
    exact = F.pad(exact, (0, 0, 0, (-nb) % 8)).view(heads, -1, 8, nb).any(2)
    return exact.reshape(-1, nb)


def sol_reference(q, k, v, rows, exact_wg, sample):
    """Sol-Attn output (exact keys + block-summary correction) in FP32 for query rows
    `sample`, with the workgroup-union selection exact_wg [H * q_tiles, nkb]."""
    heads = q.shape[2]
    nb = exact_wg.shape[1]
    lengths = torch.full((nb,), 64.0, device="cuda")
    lengths[-1] = rows - (nb - 1) * 64

    def pool(x, mean):
        x = (
            F.pad(x[0, :rows].float(), (0, 0, 0, 0, 0, nb * 64 - rows))
            .view(nb, 64, heads, 128)
            .sum(1)
        )
        return x / lengths[:, None, None] if mean else x

    kc, vc = pool(k, True), pool(v, False)
    key_block = torch.arange(rows, device="cuda") // 64
    q_tiles = exact_wg.shape[0] // heads
    out = torch.empty(sample.numel(), heads, 128, device="cuda")
    for h in range(heads):
        sel = exact_wg[h * q_tiles + sample // 512]
        qr = q[0, sample, h].float() * 128**-0.5
        logit = (qr @ k[0, :rows, h].float().T).masked_fill(
            ~sel[:, key_block], float("-inf")
        )
        approx = (qr @ kc[:, h].T + torch.log(lengths)[None]).masked_fill(
            sel, float("-inf")
        )
        prob = torch.softmax(torch.cat([logit, approx], 1), 1)
        out[:, h] = prob[:, :rows] @ v[0, :rows, h].float() + prob[:, rows:] @ (
            vc[:, h] / lengths[:, None]
        )
    return out


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def test_all_exact_matches_dense():
    # Every block exact (sink covering the sequence, or a threshold below every score): the list
    # core with the neutral Phase-A state must reproduce the dense kernel of the same backend
    # bitwise.
    ok = True
    for rows, heads in ((33, 7), (513, 14), (4097, 28), (20000, 7)):
        p = prepared(*inputs(1, rows, heads))
        for backend in BACKENDS:
            ref = dense(p, rows, backend)[:, :rows]
            for tau, sink_tokens in ((1.0, rows), (-1.0e30, 0)):
                out = sol(p, rows, tau, sink_tokens, backend=backend)[:, :rows]
                same = torch.equal(out, ref)
                ok &= report(
                    f"all exact == dense {backend} S{rows} H{heads} tau={tau:g}",
                    {"bitwise": float(same)},
                    same,
                )
    return ok


def test_routing():
    import aiter

    ok = True
    for rows, heads, sink_tokens, sink_start in (
        (4097, 7, 0, None),
        (20000, 7, 1000, 0),
        (30001, 4, 6804, None),
        (30001, 4, 700, 12345),
    ):
        p = prepared(*inputs(1, rows, heads, seed=rows))
        for tau in (0.0, 1.0, 2.5):
            plan = aiter.gfx1201_sol_route(*p, rows, tau, sink_tokens, sink_start)
            got = plan_mask(plan)
            ref = reference_mask(p, rows, tau, sink_tokens, sink_start)
            mismatch = (got != ref).float().mean().item()
            # FP32 summation order may flip blocks whose score sits on the threshold.
            ok &= report(
                f"routing S{rows} tau={tau} sink={sink_tokens}@{sink_start}",
                {"mismatch": mismatch},
                mismatch < 1e-4,
            )
    return ok


def segmented(rows, heads, segments, strength, seed=3):
    # Tokens attend mostly within their segment, so most tiles take the approximate path.
    g = torch.Generator(device="cuda").manual_seed(seed)
    basis = torch.randn((segments, heads, 128), device="cuda", generator=g)
    segment = torch.arange(rows, device="cuda") * segments // rows

    def tensor(scale):
        noise = torch.randn((rows, heads, 128), device="cuda", generator=g)
        return (basis[segment] * scale + noise)[None].bfloat16()

    return tensor(strength), tensor(strength), tensor(1.0)


def test_against_sol_reference():
    import aiter

    ok = True
    rows, heads = 20000, 7
    sample = torch.arange(0, rows, 97, device="cuda")
    for segments, strength in ((8, 1.0), (32, 2.0)):
        q, k, v = segmented(rows, heads, segments, strength)
        p = prepared(q, k, v)
        all_exact = torch.ones(
            heads * math.ceil(rows / 512),
            math.ceil(rows / 64),
            dtype=torch.bool,
            device="cuda",
        )
        exact_ref = sol_reference(q, k, v, rows, all_exact, sample)
        for tau in (0.0, 1.0, 2.5):
            plan = aiter.gfx1201_sol_route(*p, rows, tau)
            density = plan["tile_count"].sum().item() / (
                plan["n_wg"] * math.ceil(rows / 32)
            )
            ref = sol_reference(q, k, v, rows, plan_mask(plan), sample)
            for backend in BACKENDS:
                dense_err = rel(dense(p, rows, backend)[0, sample], exact_ref)
                err = rel(sol(p, rows, tau, backend=backend)[0, sample], ref)
                # Same selection as Sol-Attn: what remains is the quantization error of the
                # dense kernel.
                stats = {"density": density, "rel_l2": err, "dense_rel_l2": dense_err}
                ok &= report(
                    f"vs Sol-Attn reference {backend} segments={segments} tau={tau}",
                    stats,
                    density < 0.6 and err <= 1.1 * dense_err + 1e-3,
                )
    return ok


def test_tau_monotone():
    import aiter

    ok = True
    rows, heads = 20000, 7
    q, k, v = inputs(1, rows, heads, query_scale=0.5)
    p = prepared(q, k, v)
    sample = torch.arange(0, rows, 97, device="cuda")
    exact = torch.stack(
        [
            torch.softmax(
                q[0, sample, h].float() @ k[0, :, h].float().T * 128**-0.5, -1
            )
            @ v[0, :, h].float()
            for h in range(heads)
        ],
        1,
    )
    base = rel(dense(p, rows)[0, sample], exact)
    densities, errors = [], []
    for tau in (3.0, 1.0, 0.0, -3.0):
        plan = aiter.gfx1201_sol_route(*p, rows, tau, 1024)
        out = sol(p, rows, tau, 1024)[0, :rows]
        ok &= report(f"finite tau={tau}", {}, bool(torch.isfinite(out).all()))
        densities.append(
            plan["tile_count"].sum().item() / (plan["n_wg"] * math.ceil(rows / 32))
        )
        errors.append(rel(out[sample], exact))
    ok &= report(
        "density grows as tau falls",
        dict(zip(("d3", "d1", "d0", "d-3"), densities)),
        densities == sorted(densities) and densities[0] < densities[-1],
    )
    ok &= report(
        "dense-like error at tau=-3",
        {"rel_l2": errors[-1], "dense": base},
        errors[-1] < 2 * base + 0.01,
    )
    return ok


def test_shapes_and_streams():
    ok = True
    op = torch.ops.aiter.gfx1201_sol_attention
    for batch, rows in ((1, 1), (1, 64), (1, 65), (2, 511), (2, 513), (1, 1025)):
        q, k, v = inputs(batch, rows, 3, seed=rows)
        for backend in BACKENDS:
            out = op(q, k, v, 1.0, 0, backend=backend)
            ok &= report(
                f"shape {backend} B{batch} S{rows}",
                {},
                out.shape == q.shape and bool(torch.isfinite(out).all()),
            )
    q, k, v = inputs(2, 3000, 5)
    out = op(q, k, v, 1.0, 256)
    per_batch = torch.cat(
        [op(q[i : i + 1], k[i : i + 1], v[i : i + 1], 1.0, 256) for i in range(2)]
    )
    same = torch.equal(out, per_batch)
    ok &= report("batch 2 == per-batch", {"bitwise": float(same)}, same)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        other = op(q, k, v, 1.0, 256)
    torch.cuda.current_stream().wait_stream(stream)
    same = torch.equal(other, out)
    ok &= report("non-default stream", {"bitwise": float(same)}, same)
    results = torch.library.opcheck(
        torch.ops.aiter.gfx1201_sol_attention.default, inputs(1, 97, 3)
    )
    passed = all(r == "SUCCESS" for r in results.values())
    ok &= report("opcheck", {"passed": float(passed)}, passed)
    return ok


def test_asm_vs_hip():
    # The list code object differs from the HIP list kernel only by the score conversion of
    # gfx1201_sage_attention_fwd: close with it, bitwise without it (Q/K x8 puts every tile
    # above the fast-path guard).
    ok = True
    for rows, heads in ((33, 7), (4097, 28), (20000, 7)):
        q, k, v = inputs(1, rows, heads, seed=rows)
        for scale in (1.0, 8.0):
            p = prepared(
                (q.float() * scale).bfloat16(), (k.float() * scale).bfloat16(), v
            )
            for tau, sink_tokens in ((1.0, 0), (2.5, rows // 5)):
                asm = sol(p, rows, tau, sink_tokens, backend="asm")[:, :rows]
                hip = sol(p, rows, tau, sink_tokens, backend="hip")[:, :rows]
                again = sol(p, rows, tau, sink_tokens, backend="asm")[:, :rows]
                err = rel(asm, hip)
                stats = {"rel_l2": err, "bitwise": float(torch.equal(asm, hip))}
                good = torch.equal(asm, again) and bool(torch.isfinite(asm).all())
                good &= torch.equal(asm, hip) if scale == 8.0 else err <= 5e-3
                ok &= report(
                    f"asm vs hip S{rows} H{heads} x{scale:g} tau={tau:g} sink={sink_tokens}",
                    stats,
                    good,
                )
    return ok


def test_invalid_inputs():
    import aiter

    ok = True
    q, k, v = inputs(1, 64, 2)
    for name, args in {
        "fp32": (q.float(), k.float(), v.float()),
        "head dim 64": (
            q[..., :64].contiguous(),
            k[..., :64].contiguous(),
            v[..., :64].contiguous(),
        ),
        "requires_grad": (q.clone().requires_grad_(), k, v),
    }.items():
        try:
            torch.ops.aiter.gfx1201_sol_attention(*args)
            ok &= report(f"rejects {name}", {}, False)
        except (ValueError, RuntimeError):
            ok &= report(f"rejects {name}", {}, True)
    try:
        aiter.gfx1201_sol_route(*prepared(*inputs(2, 64, 2)), 64)
        ok &= report("route rejects batch 2", {}, False)
    except (ValueError, RuntimeError):
        ok &= report("route rejects batch 2", {}, True)
    for sink_tokens, sink_start in ((65, None), (-1, None), (32, 40), (8, -1)):
        try:
            torch.ops.aiter.gfx1201_sol_attention(q, k, v, 1.0, sink_tokens, sink_start)
            ok &= report(f"rejects sink {sink_tokens}@{sink_start}", {}, False)
        except (ValueError, RuntimeError):
            ok &= report(f"rejects sink {sink_tokens}@{sink_start}", {}, True)
    return ok


def main():
    if get_gfx() != "gfx1201":
        print(f"skip: gfx1201_sol_attention requires gfx1201, got {get_gfx()}")
        return 0
    import aiter  # noqa: F401  registers torch.ops.aiter.gfx1201_sol_attention

    ok = True
    for test in (
        test_all_exact_matches_dense,
        test_routing,
        test_against_sol_reference,
        test_tau_monotone,
        test_shapes_and_streams,
        test_asm_vs_hip,
        test_invalid_inputs,
    ):
        ok &= test()
    print("ALL PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
