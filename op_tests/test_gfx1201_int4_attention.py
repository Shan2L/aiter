# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import math
import unittest

import torch
from aiter.ops.gfx1201.int4_attention import CORES, int4_attention, int4_prepare, qk_norm_rope
from aiter.ops.gfx1201.prepare import prepare_sage

SHAPES = ((4097, 28), (1000, 14), (333, 7))


def make_qkv(rows, heads, seed=7):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    basis = torch.randn((8, 128), device="cuda", generator=generator)

    def tensor(scale):
        mix = torch.randn((rows, heads, 8), device="cuda", generator=generator)
        return (scale * (mix @ basis + 0.5 * torch.randn((rows, heads, 128), device="cuda", generator=generator))).bfloat16()

    return tensor(1.0), tensor(1.0) + 2.0, tensor(1.0)


def hadamard(device):
    h = torch.ones(1, 1)
    while h.shape[0] < 128:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return (h / math.sqrt(128)).to(device)


def reference_int4(x, padded):
    """[S, H, 128] fp32 (rotated) -> INT4 values [Sp, H, 128] and scales [H, Sp/32]."""
    rows, heads, _ = x.shape
    xp = torch.nn.functional.pad(x, (0, 0, 0, 0, 0, padded - rows)).view(padded // 32, 32, heads, 128)
    scale = 0.8 * xp.abs().amax((1, 3)).clamp_min(1e-12) / 7.0
    values = torch.clamp(torch.round(xp / scale[:, None, :, None]), -8, 7).view(padded, heads, 128)
    return values, scale.T


def unpack(packed):
    b = packed[..., :64].view(torch.uint8).to(torch.int16)
    lo, hi = b & 0xF, b >> 4
    return torch.stack([lo - 16 * (lo > 7), hi - 16 * (hi > 7)], -1).flatten(-2)


class TestInt4Attention(unittest.TestCase):
    def test_prepare(self):
        for rows, heads in SHAPES:
            q, k, v = make_qkv(rows, heads)
            q4, qs, k4, ks, v8, vs = int4_prepare(q, k, v)
            padded = (rows + 31) // 32 * 32
            self.assertEqual(q4.shape, (1, padded, heads, 128))
            ref_v = prepare_sage(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0))
            self.assertTrue(torch.equal(v8.view(torch.uint8), ref_v[4].view(torch.uint8)))
            self.assertTrue(torch.equal(vs, ref_v[5]))
            had = hadamard(q.device)
            kf = k.float()
            for name, packed, scale, x, mul in (
                ("q", q4, qs, q.float() @ had, 128**-0.5 * 1.4426950408889634),
                ("k", k4, ks, (kf - kf.mean(0, keepdim=True)) @ had, 1.0),
            ):
                values, ref_scale = reference_int4(x, padded)
                mismatch = (unpack(packed[0]) != values).float().mean().item()
                self.assertLess(mismatch, 2e-3, (name, rows, heads))
                self.assertTrue(bool((packed[0, :, :, 64:] == 0).all()))
                rel = ((scale[0] - ref_scale * mul).abs() / (ref_scale * mul)).max().item()
                self.assertLess(rel, 1e-5, (name, rows, heads))

    def test_attention(self):
        for rows, heads in SHAPES:
            q, k, v = make_qkv(rows, heads)
            q4, qs, k4, ks, v8, vs = int4_prepare(q, k, v)
            torch.manual_seed(0)
            sample = torch.randperm(rows, device="cuda")[:256]
            # reference on the dequantized operands: isolates the core arithmetic from the quantization
            qd = unpack(q4[0, sample]).float() * qs[0].T.repeat_interleave(32, 0)[sample, :, None]
            kd = unpack(k4[0, :rows]).float() * ks[0].T.repeat_interleave(32, 0)[:rows, :, None]
            vd = v8[0, :, :, :rows].float() * vs[0][:, :, None]
            ref = torch.stack([torch.softmax((qd[:, h] @ kd[:, h].T) * math.log(2), -1) @ vd[h].T for h in range(heads)], 1)
            errors = {}
            for core in CORES:
                out = int4_attention(q, k, v, core)
                self.assertEqual(out.shape, (rows, heads, 128))
                self.assertTrue(bool(out.isfinite().all()), core)
                errors[core] = ((out[sample].float() - ref).norm() / ref.norm()).item()
            print(f"S={rows} H={heads} rel err vs dequantized fp32: " + " ".join(f"{c} {e:.4f}" for c, e in errors.items()))
            self.assertLess(max(errors["th4"], errors["t2"]), 0.035, errors)
            self.assertLess(max(errors["th4f"], errors["t2f"]), 0.06, errors)
            self.assertLessEqual(errors["t2"], errors["th4"] * 1.001, errors)

    def test_qk_norm_rope(self):
        generator = torch.Generator(device="cuda").manual_seed(3)
        for rows, heads in SHAPES:
            q, k = ((torch.randn(rows, heads, 128, device="cuda", generator=generator) * 3).bfloat16() for _ in range(2))
            qw, kw = ((1 + 0.3 * torch.randn(128, device="cuda", generator=generator)).bfloat16() for _ in range(2))
            angle = torch.rand(rows, 48, device="cuda", generator=generator) * 6.28
            cos = torch.cat([angle.cos(), angle.cos()], 1).contiguous()
            sin = torch.cat([angle.sin(), angle.sin()], 1).contiguous()
            qo, ko = qk_norm_rope(q, k, qw, kw, cos, sin)
            for x, w, out in ((q, qw, qo), (k, kw, ko)):
                xf = x.float()
                n = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-5) * w.float()).bfloat16().float()
                rot = torch.cat([-n[..., 48:96], n[..., :48]], -1)
                ref = torch.cat([n[..., :96] * cos[:, None] + rot * sin[:, None], n[..., 96:]], -1).bfloat16()
                self.assertGreater((out == ref).float().mean().item(), 0.999)
                self.assertLess((out.float() - ref.float()).abs().max().item(), 0.05)
            q2, k2 = q.clone(), k.clone()
            qk_norm_rope(q2, k2, qw, kw, cos, sin, q2, k2)
            self.assertTrue(torch.equal(q2, qo) and torch.equal(k2, ko))


if __name__ == "__main__":
    unittest.main()
