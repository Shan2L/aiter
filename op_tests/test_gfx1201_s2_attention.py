# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import unittest

import torch
from aiter.ops.gfx1201.prepare import prepare_sage
from aiter.ops.gfx1201.s2_attention import CORES, s2_attention

SHAPES = ((4097, 28), (1000, 14), (333, 7))


def make_qkv(rows, heads, seed=7):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn((rows, heads, 128), device="cuda", generator=generator)
    k = q * 0.6 + torch.randn((rows, heads, 128), device="cuda", generator=generator) * 0.8 + 2.0
    v = torch.randn((rows, heads, 128), device="cuda", generator=generator)
    return q.bfloat16(), k.bfloat16(), v.bfloat16()


def fp32_reference(prepared, sequence, sample):
    q8, qs, k8, ks, v8, vs = prepared
    heads = q8.shape[2]
    keys = torch.arange(sequence, device="cuda")
    out = torch.empty((len(sample), heads, 128), device="cuda", dtype=torch.float32)
    for h in range(heads):
        q = q8[0, sample, h].float() * qs[0, h, sample // 32].unsqueeze(-1)
        k = k8[0, :sequence, h].float() * ks[0, h, keys // 32].unsqueeze(-1)
        v = v8[0, h, :, :sequence].float().t() * vs[0, h].unsqueeze(0)
        s = (q @ k.t()).double()
        p = torch.exp2(s - s.max(dim=-1, keepdim=True).values)
        out[:, h] = ((p @ v.double()) / p.sum(dim=-1, keepdim=True)).float()
    return out


class TestS2Attention(unittest.TestCase):
    def test_attention(self):
        for rows, heads in SHAPES:
            q, k, v = make_qkv(rows, heads)
            prepared = prepare_sage(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0))
            torch.manual_seed(0)
            sample = torch.randperm(rows, device="cuda")[:256].sort().values
            ref = fp32_reference(prepared, rows, sample)
            errors = {}
            for core in CORES:
                out = s2_attention(q, k, v, core)
                self.assertEqual(out.shape, (rows, heads, 128))
                self.assertTrue(bool(out.isfinite().all()), core)
                errors[core] = ((out[sample].float() - ref).norm() / ref.norm()).item()
            print(f"S={rows} H={heads} rel err vs dequantized fp32: " + " ".join(f"{c} {e:.4f}" for c, e in errors.items()))
            self.assertLess(errors["t2f"], 0.015, errors)
            self.assertLess(errors["th4f"], 0.015, errors)
            self.assertLess(errors["s2_4_7"], 0.02, errors)
            self.assertLess(errors["s2_4_4"], 0.20, errors)
            self.assertLessEqual(errors["t2f"], errors["th4f"] * 1.01, errors)
            self.assertLessEqual(errors["s2_4_7"], errors["s2_4_4"] * 1.001, errors)


if __name__ == "__main__":
    unittest.main()
