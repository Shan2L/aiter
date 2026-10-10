// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// gfx1201 INT4 attention, preparation for the INT4-QK / FP8-PV ASM cores (hsa/gfx1201/int4_attention):
// - qk_norm_rope: QK RMSNorm (eps 1e-5, BF16 weight) + RoPE on the first 96 channels, BF16 in -> BF16 out
//   (row-local, so it may run in place)
// - prepare: Q/K are rotated by the normalized 128x128 Walsh-Hadamard matrix (K minus its per-head mean first)
//   and quantized to INT4 per (head, 32 rows) with scale 0.8 * amax / 7 (pairs packed in bytes 0..63 of each
//   128-byte row, bytes 64..127 zero); V is FP8 e4m3 V^T with the per-channel scale amax|v| / 448.
// q, k, v: BF16 [S, H, 128]. q4/k4: INT8 [Sp, H, 128]; qs/ks: FP32 [H, Sp/32] (qs includes softmax scale * log2 e);
// v8: FP8 [H, 128, Sp]; vs: FP32 [H, 128]. Sp = S rounded up to 32.
#include <hip/hip_runtime.h>
#include <stdint.h>

#define D 128
#define KV_ROWS 128

namespace {

__device__ __forceinline__ float bf_lo(uint32_t u) { return __uint_as_float(u << 16); }
__device__ __forceinline__ float bf_hi(uint32_t u) { return __uint_as_float(u & 0xffff0000u); }

__device__ __forceinline__ uint32_t f2bf(float f) {  // round to nearest even (finite inputs)
  uint32_t u = __float_as_uint(f);
  return (u + 0x7fffu + ((u >> 16) & 1u)) >> 16;
}

// grid (ceil(S/32), H, 2) z: 0 = Q, 1 = K; block 32, lane = row
__global__ void __launch_bounds__(32) qk_norm_rope(const uint4* q, const uint4* k, const uint16_t* __restrict__ qw,
                                                   const uint16_t* __restrict__ kw, const float* __restrict__ cosine,
                                                   const float* __restrict__ sine, uint4* q_out, uint4* k_out, int S,
                                                   int H) {
  int h = blockIdx.y, isk = blockIdx.z, n = blockIdx.x * 32 + threadIdx.x;
  if (n >= S) return;
  const uint4* p = (isk ? k : q) + ((size_t)n * H + h) * (D / 8);
  const uint16_t* w = isk ? kw : qw;
  float x[D];
#pragma unroll
  for (int i = 0; i < D / 8; ++i) {
    uint4 u = p[i];
    x[8 * i + 0] = bf_lo(u.x); x[8 * i + 1] = bf_hi(u.x); x[8 * i + 2] = bf_lo(u.y); x[8 * i + 3] = bf_hi(u.y);
    x[8 * i + 4] = bf_lo(u.z); x[8 * i + 5] = bf_hi(u.z); x[8 * i + 6] = bf_lo(u.w); x[8 * i + 7] = bf_hi(u.w);
  }
  float ss = 0.f;
#pragma unroll
  for (int i = 0; i < D; ++i) ss += x[i] * x[i];
  const float r = rsqrtf(ss * (1.0f / D) + 1e-5f);
#pragma unroll
  for (int i = 0; i < D; ++i) x[i] = __uint_as_float(f2bf(x[i] * r * __uint_as_float((uint32_t)w[i] << 16)) << 16);
  const float* c = cosine + (size_t)n * 96;
  const float* s = sine + (size_t)n * 96;
  uint32_t o[D];
#pragma unroll
  for (int i = 0; i < 48; ++i) {
    o[i] = f2bf(x[i] * c[i] - x[i + 48] * s[i]);
    o[i + 48] = f2bf(x[i + 48] * c[i + 48] + x[i] * s[i + 48]);
  }
#pragma unroll
  for (int i = 96; i < D; ++i) o[i] = f2bf(x[i]);
  uint4* dst = (isk ? k_out : q_out) + ((size_t)n * H + h) * (D / 8);
#pragma unroll
  for (int i = 0; i < D / 8; ++i)
    dst[i] = make_uint4(o[8 * i] | (o[8 * i + 1] << 16), o[8 * i + 2] | (o[8 * i + 3] << 16),
                        o[8 * i + 4] | (o[8 * i + 5] << 16), o[8 * i + 6] | (o[8 * i + 7] << 16));
}

// grid (ceil(S/KV_ROWS), H), block 64: thread t owns channels 2t, 2t+1
__global__ void __launch_bounds__(64) kv_part(const uint32_t* __restrict__ k, const uint32_t* __restrict__ v,
                                              float* __restrict__ ksum, float* __restrict__ vmax, int S, int H) {
  int c = blockIdx.x, h = blockIdx.y, t = threadIdx.x;
  int n0 = c * KV_ROWS, n1 = min(n0 + KV_ROWS, S);
  float s0 = 0.f, s1 = 0.f, m0 = 0.f, m1 = 0.f;
  for (int n = n0; n < n1; ++n) {
    size_t o = ((size_t)n * H + h) * (D / 2) + t;
    uint32_t a = k[o], b = v[o];
    s0 += bf_lo(a); s1 += bf_hi(a);
    m0 = fmaxf(m0, fabsf(bf_lo(b))); m1 = fmaxf(m1, fabsf(bf_hi(b)));
  }
  size_t p = ((size_t)c * H + h) * D + 2 * t;
  ksum[p] = s0; ksum[p + 1] = s1; vmax[p] = m0; vmax[p + 1] = m1;
}

// grid H, block 128
__global__ void __launch_bounds__(128) kv_reduce(const float* __restrict__ ksum, const float* __restrict__ vmax,
                                                 float* __restrict__ kmean, float* __restrict__ vs, int S, int H,
                                                 int chunks) {
  int h = blockIdx.x, d = threadIdx.x;
  float s = 0.f, m = 0.f;
  for (int c = 0; c < chunks; ++c) {
    size_t p = ((size_t)c * H + h) * D + d;
    s += ksum[p]; m = fmaxf(m, vmax[p]);
  }
  kmean[h * D + d] = s / (float)S;
  vs[h * D + d] = m * (1.0f / 448.0f);
}

// grid (Sp/32, H, 2) z: 0 = Q, 1 = K; block 32 (one wave), lane = row
__global__ void __launch_bounds__(32) qk_quant(const uint4* __restrict__ q, const uint4* __restrict__ k,
                                               const float* __restrict__ kmean, int8_t* __restrict__ q4,
                                               int8_t* __restrict__ k4, float* __restrict__ qs, float* __restrict__ ks,
                                               int S, int Sp, int H, float qmul) {
  int blk = blockIdx.x, h = blockIdx.y, isk = blockIdx.z, lane = threadIdx.x;
  int n = blk * 32 + lane;
  const uint4* src = isk ? k : q;
  float x[D];
  if (n < S) {
    const uint4* p = src + ((size_t)n * H + h) * (D / 8);
#pragma unroll
    for (int i = 0; i < D / 8; ++i) {
      uint4 u = p[i];
      x[8 * i + 0] = bf_lo(u.x); x[8 * i + 1] = bf_hi(u.x); x[8 * i + 2] = bf_lo(u.y); x[8 * i + 3] = bf_hi(u.y);
      x[8 * i + 4] = bf_lo(u.z); x[8 * i + 5] = bf_hi(u.z); x[8 * i + 6] = bf_lo(u.w); x[8 * i + 7] = bf_hi(u.w);
    }
    if (isk) {
      const float* m = kmean + h * D;
#pragma unroll
      for (int i = 0; i < D; ++i) x[i] -= m[i];
    }
  } else {
#pragma unroll
    for (int i = 0; i < D; ++i) x[i] = 0.f;
  }
  // Sylvester Walsh-Hadamard (== x @ H128), normalized by 1/sqrt(128)
#pragma unroll
  for (int hlen = 1; hlen < D; hlen <<= 1) {
#pragma unroll
    for (int i = 0; i < D; ++i) {
      if ((i & hlen) == 0) {
        float a = x[i], b = x[i + hlen];
        x[i] = a + b; x[i + hlen] = a - b;
      }
    }
  }
  const float norm = 0.08838834764831845f;
  float am = 0.f;
#pragma unroll
  for (int i = 0; i < D; ++i) { x[i] *= norm; am = fmaxf(am, fabsf(x[i])); }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) am = fmaxf(am, __shfl_xor(am, off, 32));
  float s = 0.8f * fmaxf(am, 1e-12f) / 7.0f;
  uint32_t w[16];
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    uint32_t r = 0;
#pragma unroll
    for (int b = 0; b < 8; ++b) {
      int qv = (int)fminf(fmaxf(rintf(x[8 * j + b] / s), -8.f), 7.f);
      r |= (uint32_t)(qv & 0xF) << (4 * b);
    }
    w[j] = r;
  }
  uint4* dst = (uint4*)((isk ? k4 : q4) + ((size_t)n * H + h) * D);
  dst[0] = make_uint4(w[0], w[1], w[2], w[3]);
  dst[1] = make_uint4(w[4], w[5], w[6], w[7]);
  dst[2] = make_uint4(w[8], w[9], w[10], w[11]);
  dst[3] = make_uint4(w[12], w[13], w[14], w[15]);
  uint4 z = make_uint4(0, 0, 0, 0);
  dst[4] = z; dst[5] = z; dst[6] = z; dst[7] = z;
  if (lane == 0) (isk ? ks : qs)[(size_t)h * (Sp / 32) + blk] = isk ? s : s * qmul;
}

// grid (Sp/32, H), block 256: 32 rows x 128 channels -> v8[h, d, n0:n0+32]
__global__ void __launch_bounds__(256) v_trans(const uint4* __restrict__ v, const float* __restrict__ vs,
                                               uint8_t* __restrict__ v8, int S, int Sp, int H) {
  __shared__ uint8_t tile[D][32 + 4];
  int blk = blockIdx.x, h = blockIdx.y, t = threadIdx.x;
  int r = t >> 3, c0 = (t & 7) * 16, n = blk * 32 + r;
  float x[16];
  if (n < S) {
    const uint4* p = v + (((size_t)n * H + h) * D + c0) / 8;
    uint4 a = p[0], b = p[1];
    uint32_t u[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      x[2 * i] = bf_lo(u[i]) / vs[h * D + c0 + 2 * i];
      x[2 * i + 1] = bf_hi(u[i]) / vs[h * D + c0 + 2 * i + 1];
    }
  } else {
#pragma unroll
    for (int i = 0; i < 16; ++i) x[i] = 0.f;
  }
#pragma unroll
  for (int i = 0; i < 16; i += 2) {
    int pk = __builtin_amdgcn_cvt_pk_fp8_f32(x[i], x[i + 1], 0, false);
    tile[c0 + i][r] = (uint8_t)(pk & 0xff);
    tile[c0 + i + 1][r] = (uint8_t)((pk >> 8) & 0xff);
  }
  __syncthreads();
  int d = t >> 1, half = t & 1;
  const uint8_t* s = &tile[d][half * 16];
  uint32_t w[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) w[i] = s[4 * i] | (s[4 * i + 1] << 8) | (s[4 * i + 2] << 16) | ((uint32_t)s[4 * i + 3] << 24);
  *(uint4*)(v8 + ((size_t)h * D + d) * Sp + blk * 32 + half * 16) = make_uint4(w[0], w[1], w[2], w[3]);
}

}  // namespace

void launch_gfx1201_int4_qk_norm_rope(int64_t q, int64_t k, int64_t qw, int64_t kw, int64_t cosine, int64_t sine,
                                      int64_t q_out, int64_t k_out, int64_t S, int64_t H, int64_t stream) {
  hipLaunchKernelGGL(qk_norm_rope, dim3((S + 31) / 32, H, 2), dim3(32), 0, (hipStream_t)stream, (const uint4*)q,
                     (const uint4*)k, (const uint16_t*)qw, (const uint16_t*)kw, (const float*)cosine,
                     (const float*)sine, (uint4*)q_out, (uint4*)k_out, (int)S, (int)H);
}

void launch_gfx1201_int4_prepare(int64_t q, int64_t k, int64_t v, int64_t part, int64_t kmean, int64_t q4, int64_t k4,
                                 int64_t qs, int64_t ks, int64_t v8, int64_t vs, int64_t S, int64_t Sp, int64_t H,
                                 double qmul, int64_t stream) {
  hipStream_t st = (hipStream_t)stream;
  const int s = (int)S, sp = (int)Sp, h = (int)H, chunks = (s + KV_ROWS - 1) / KV_ROWS;
  float* ksum = (float*)part;
  float* vmax = ksum + (size_t)chunks * h * D;
  hipLaunchKernelGGL(kv_part, dim3(chunks, h), dim3(64), 0, st, (const uint32_t*)k, (const uint32_t*)v, ksum, vmax, s,
                     h);
  hipLaunchKernelGGL(kv_reduce, dim3(h), dim3(128), 0, st, ksum, vmax, (float*)kmean, (float*)vs, s, h, chunks);
  hipLaunchKernelGGL(qk_quant, dim3(sp / 32, h, 2), dim3(32), 0, st, (const uint4*)q, (const uint4*)k,
                     (const float*)kmean, (int8_t*)q4, (int8_t*)k4, (float*)qs, (float*)ks, s, sp, h, (float)qmul);
  hipLaunchKernelGGL(v_trans, dim3(sp / 32, h), dim3(256), 0, st, (const uint4*)v, (const float*)vs, (uint8_t*)v8, s,
                     sp, h);
}
