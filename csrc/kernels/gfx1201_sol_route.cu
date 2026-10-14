// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

// gfx1201 Sol sparse attention routing (Sol-Attn "diag" rule, arXiv 2607.24027): 64-row key-block
// summaries, pooled INT8 keys / FP8 value sums for Phase A, the exact key blocks of every 64-row
// query block, and their union per 512-row workgroup as a tile list for the LIST core plus a
// Phase-A mask. Workgroup w = head * q_tiles + q_tile.
#include <hip/hip_runtime.h>
#include <math.h>
#include <stdint.h>

#include <stdexcept>

#include "aiter_stream.h"
#include "gfx1201_sol_attention.h"

static constexpr int D = 128, KBR = 64, QBR = 64, WGR = 512, QB_PER_WG = WGR / QBR;
static constexpr int MAX_NKB = 4096, KC_CHUNK = 32, KC_STRIDE = D + 1;

__device__ __forceinline__ float fp8_to_f32(uint32_t w, int sel)
{
    switch(sel)
    {
    case 0: return __builtin_amdgcn_cvt_f32_fp8((int)w, 0);
    case 1: return __builtin_amdgcn_cvt_f32_fp8((int)w, 1);
    case 2: return __builtin_amdgcn_cvt_f32_fp8((int)w, 2);
    default: return __builtin_amdgcn_cvt_f32_fp8((int)w, 3);
    }
}

// grid (nkbp, H), block 128: kbar [H, nkbp, 128] block-mean dequantized keys (Sol kc),
// vsum [H, 128, nkbp] block-sum dequantized values (Sol vc).
static __global__ void gfx1201_sol_pool_kv(const int8_t* K,
                                           const float* KS,
                                           const uint8_t* V,
                                           const float* VS,
                                           float* kbar,
                                           float* vsum,
                                           int Sp,
                                           int valid,
                                           int H,
                                           int nkbp)
{
    const int b = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    const int r0 = b * KBR;
    const int n  = max(0, min(KBR, valid - r0));
    float ks     = 0.f;
    for(int g = 0; g < n; g += 32)
    {
        int acc     = 0;
        const int m = min(32, n - g);
        for(int r = 0; r < m; ++r)
            acc += K[((long long)(r0 + g + r) * H + h) * D + d];
        ks += (float)acc * KS[(long long)h * (Sp / 32) + (r0 + g) / 32];
    }
    kbar[((long long)h * nkbp + b) * D + d] = ks / (float)max(n, 1);
    const uint8_t* vp                       = V + ((long long)h * D + d) * Sp + r0;
    float vsm                               = 0.f;
    for(int r = 0; r < n; r += 4)
    {
        const uint32_t w = *reinterpret_cast<const uint32_t*>(vp + r);
        const int m      = min(4, n - r);
        for(int i = 0; i < m; ++i)
            vsm += fp8_to_f32(w, i);
    }
    vsum[((long long)h * D + d) * nkbp + b] = vsm * VS[h * D + d];
}

// grid (nkbp / 32, H), block 256: INT8 kb [nkbp, H, 128] with one scale per 32 blocks
// (kb_scale [H, nkbp / 32]).
static __global__ void
gfx1201_sol_quant_k(const float* kbar, int8_t* kb, float* kb_scale, int H, int nkbp)
{
    __shared__ float red[256];
    const int t = blockIdx.x, h = blockIdx.y, tid = threadIdx.x;
    const float* src = kbar + ((long long)h * nkbp + t * 32) * D;
    float am         = 0.f;
    for(int i = tid; i < 32 * D; i += 256)
        am = fmaxf(am, fabsf(src[i]));
    red[tid] = am;
    __syncthreads();
    for(int s = 128; s > 0; s >>= 1)
    {
        if(tid < s)
            red[tid] = fmaxf(red[tid], red[tid + s]);
        __syncthreads();
    }
    const float sc = fmaxf(red[0] / 127.f, 1e-12f);
    if(tid == 0)
        kb_scale[h * (nkbp / 32) + t] = sc;
    for(int i = tid; i < 32 * D; i += 256)
    {
        const float x = fminf(fmaxf(rintf(src[i] / sc), -127.f), 127.f);
        kb[((long long)(t * 32 + i / D) * H + h) * D + i % D] = (int8_t)x;
    }
}

// grid H, block 128: Sol kc_mean / kc_var (over all key blocks), FP8 vb [H, 128, nkbp] with a
// per-channel scale vb_scale [H, 128].
static __global__ void gfx1201_sol_stats(const float* kbar,
                                         const float* vsum,
                                         uint8_t* vb,
                                         float* vb_scale,
                                         float* mu,
                                         float* var,
                                         int H,
                                         int nkb,
                                         int nkbp)
{
    const int h = blockIdx.x, d = threadIdx.x;
    float s = 0.f, s2 = 0.f;
    for(int b = 0; b < nkb; ++b)
    {
        const float x = kbar[((long long)h * nkbp + b) * D + d];
        s += x;
        s2 += x * x;
    }
    const float m   = s / nkb;
    mu[h * D + d]   = m;
    var[h * D + d]  = s2 / nkb - m * m;
    const float* vs = vsum + ((long long)h * D + d) * nkbp;
    float am        = 0.f;
    for(int b = 0; b < nkbp; ++b)
        am = fmaxf(am, fabsf(vs[b]));
    const float sc      = fmaxf(am / 448.f, 1e-20f);
    vb_scale[h * D + d] = sc;
    uint8_t* dst        = vb + ((long long)h * D + d) * nkbp;
    for(int b = 0; b < nkbp; b += 4)
    {
        int w = __builtin_amdgcn_cvt_pk_fp8_f32(vs[b] / sc, vs[b + 1] / sc, 0, false);
        w     = __builtin_amdgcn_cvt_pk_fp8_f32(vs[b + 2] / sc, vs[b + 3] / sc, w, true);
        *reinterpret_cast<int*>(dst + b) = w;
    }
}

// Ascending 32-key tile starts of the selected blocks (full tiles only; the partial tail tile is
// always run by the core), followed by the tail start and a dummy; the Phase-A mask has bit j of
// word t set for block 32 * t + j when it is exact (or padding).
__device__ void gfx1201_sol_emit_list(uint8_t* uni,
                                      int* scan,
                                      int32_t* list,
                                      int32_t* count,
                                      uint32_t* mask,
                                      int wg,
                                      int valid,
                                      int nkb,
                                      int nkbp,
                                      int stride)
{
    const int tid = threadIdx.x;
    const int per = (nkb + 255) / 256;
    const int b0 = tid * per, b1 = min(nkb, b0 + per);
    const int full_end = valid / 32 * 32;
    for(int b = nkb + tid; b < nkbp; b += 256)
        uni[b] = 1;
    int ntile = 0;
    for(int b = b0; b < b1; ++b)
        if(uni[b])
            ntile += (b * KBR < full_end) + (b * KBR + 32 < full_end);
    scan[tid] = ntile;
    __syncthreads();
    for(int off = 1; off < 256; off <<= 1)
    {
        const int v = tid >= off ? scan[tid - off] : 0;
        __syncthreads();
        scan[tid] += v;
        __syncthreads();
    }
    int pos      = scan[tid] - ntile;
    int32_t* lst = list + (long long)wg * stride;
    for(int b = b0; b < b1; ++b)
    {
        if(!uni[b])
            continue;
        if(b * KBR < full_end)
            lst[pos++] = b * KBR;
        if(b * KBR + 32 < full_end)
            lst[pos++] = b * KBR + 32;
    }
    const int total = scan[255];
    if(tid == 0)
    {
        count[wg]      = total;
        lst[total]     = full_end;
        lst[total + 1] = 0;
    }
    for(int t = tid; t < nkbp / 32; t += 256)
    {
        uint32_t w = 0;
        for(int j = 0; j < 32; ++j)
            w |= (uint32_t)uni[t * 32 + j] << j;
        mask[(long long)wg * (nkbp / 32) + t] = w;
    }
}

// grid W, block 256: wave w routes query block q_tile * 8 + w (64 rows). qc = block-mean query (the
// Q scales carry softmax_scale * log2 e); key block j is exact if qc . kc_j > qc . mu + tau *
// sqrt(sum qc^2 var + 1e-6), |qb - j| <= 1, j overlaps the sink tokens [sink_start, sink_start +
// sink_tokens), or j is the last block.
static __global__ __launch_bounds__(256) void gfx1201_sol_select(const int8_t* Q,
                                                                 const float* QS,
                                                                 const float* kbar,
                                                                 const float* mu,
                                                                 const float* var,
                                                                 int32_t* list,
                                                                 int32_t* count,
                                                                 uint32_t* mask,
                                                                 int Sp,
                                                                 int valid,
                                                                 int H,
                                                                 int nkb,
                                                                 int nkbp,
                                                                 int nqt,
                                                                 int stride,
                                                                 int sink_start,
                                                                 int sink_tokens,
                                                                 float tau)
{
    __shared__ uint8_t uni[MAX_NKB + 32];
    __shared__ int scan[256];
    __shared__ float sq[QB_PER_WG][D];
    __shared__ float skc[KC_CHUNK * KC_STRIDE];
    const int wg = blockIdx.x, tid = threadIdx.x;
    const int h = wg / nqt, qt = wg % nqt;
    const int wave = tid / 32, lane = tid % 32;
    for(int i = tid; i < nkbp; i += 256)
        uni[i] = 0;
    const int qb = qt * QB_PER_WG + wave;
    const int q0 = qb * QBR, n = max(0, min(QBR, valid - q0));
    const int sink_lo = sink_start / KBR;
    const int sink_hi = sink_tokens > 0 ? (sink_start + sink_tokens + KBR - 1) / KBR : 0;
    // qc: lane owns dims 4 * lane .. 4 * lane + 3
    float qc[4] = {0.f, 0.f, 0.f, 0.f};
    for(int g = 0; g < n; g += 32)
    {
        int acc[4]  = {0, 0, 0, 0};
        const int m = min(32, n - g);
        for(int r = 0; r < m; ++r)
        {
            const int w =
                *reinterpret_cast<const int*>(Q + ((long long)(q0 + g + r) * H + h) * D + 4 * lane);
#pragma unroll
            for(int i = 0; i < 4; ++i)
                acc[i] += (int)(int8_t)(w >> (8 * i));
        }
        const float sc = QS[(long long)h * (Sp / 32) + (q0 + g) / 32];
#pragma unroll
        for(int i = 0; i < 4; ++i)
            qc[i] += (float)acc[i] * sc;
    }
    float mean = 0.f, vv = 0.f;
#pragma unroll
    for(int i = 0; i < 4; ++i)
    {
        qc[i] /= (float)max(n, 1);
        sq[wave][4 * lane + i] = qc[i];
        mean += qc[i] * mu[h * D + 4 * lane + i];
        vv += qc[i] * qc[i] * var[h * D + 4 * lane + i];
    }
    for(int o = 16; o > 0; o >>= 1)
    {
        mean += __shfl_xor(mean, o, 32);
        vv += __shfl_xor(vv, o, 32);
    }
    const float thr   = mean + tau * sqrtf(fmaxf(vv, 0.f) + 1e-6f);
    const bool active = n > 0;
    for(int j0 = 0; j0 < nkb; j0 += KC_CHUNK)
    {
        __syncthreads();
        for(int i = tid; i < KC_CHUNK * D; i += 256)
        {
            const int jb = j0 + i / D;
            skc[(i / D) * KC_STRIDE + i % D] =
                jb < nkb ? kbar[((long long)h * nkbp + jb) * D + i % D] : 0.f;
        }
        __syncthreads();
        const int j = j0 + lane;
        if(active && j < nkb)
        {
            float s = 0.f;
#pragma unroll 8
            for(int d = 0; d < D; ++d)
                s += sq[wave][d] * skc[lane * KC_STRIDE + d];
            if(s > thr || abs(qb - j) <= 1 || (j >= sink_lo && j < sink_hi) || j == nkb - 1)
                uni[j] = 1;
        }
    }
    __syncthreads();
    gfx1201_sol_emit_list(uni, scan, list, count, mask, wg, valid, nkb, nkbp, stride);
}

void gfx1201_sol_route_hip(const aiter_tensor_t& q_int8,
                           const aiter_tensor_t& q_scale,
                           const aiter_tensor_t& k_int8,
                           const aiter_tensor_t& k_scale,
                           const aiter_tensor_t& v_fp8,
                           const aiter_tensor_t& v_scale,
                           aiter_tensor_t& kbar,
                           aiter_tensor_t& vsum,
                           aiter_tensor_t& kb,
                           aiter_tensor_t& kb_scale,
                           aiter_tensor_t& vb,
                           aiter_tensor_t& vb_scale,
                           aiter_tensor_t& mu,
                           aiter_tensor_t& var,
                           aiter_tensor_t& tile_list,
                           aiter_tensor_t& tile_count,
                           aiter_tensor_t& mask,
                           int64_t seq_len,
                           int64_t sink_start,
                           int64_t sink_tokens,
                           double tau)
{
    const int Sp = (int)q_int8.size(1), H = (int)q_int8.size(2), valid = (int)seq_len;
    const int nkb = (valid + KBR - 1) / KBR, nkbp = (int)kb.size(0);
    const int nqt = (valid + WGR - 1) / WGR;
    if(q_int8.size(0) != 1 || valid <= 0 || valid > Sp || nkbp % 32 || nkbp < nkb ||
       nkbp > MAX_NKB || tile_list.size(0) != (int64_t)nqt * H || tile_list.size(1) < 2 * nkb + 2)
        throw std::invalid_argument("gfx1201_sol_route: invalid shape");
    if(sink_tokens < 0 || sink_start < 0 || sink_start + sink_tokens > seq_len)
        throw std::invalid_argument("gfx1201_sol_route: invalid sink range");
    const hipStream_t st = aiter::getCurrentHIPStream();
    hipLaunchKernelGGL(gfx1201_sol_pool_kv,
                       dim3(nkbp, H),
                       dim3(D),
                       0,
                       st,
                       (const int8_t*)k_int8.ptr,
                       (const float*)k_scale.ptr,
                       (const uint8_t*)v_fp8.ptr,
                       (const float*)v_scale.ptr,
                       (float*)kbar.ptr,
                       (float*)vsum.ptr,
                       Sp,
                       valid,
                       H,
                       nkbp);
    hipLaunchKernelGGL(gfx1201_sol_quant_k,
                       dim3(nkbp / 32, H),
                       dim3(256),
                       0,
                       st,
                       (const float*)kbar.ptr,
                       (int8_t*)kb.ptr,
                       (float*)kb_scale.ptr,
                       H,
                       nkbp);
    hipLaunchKernelGGL(gfx1201_sol_stats,
                       dim3(H),
                       dim3(D),
                       0,
                       st,
                       (const float*)kbar.ptr,
                       (const float*)vsum.ptr,
                       (uint8_t*)vb.ptr,
                       (float*)vb_scale.ptr,
                       (float*)mu.ptr,
                       (float*)var.ptr,
                       H,
                       nkb,
                       nkbp);
    hipLaunchKernelGGL(gfx1201_sol_select,
                       dim3(nqt * H),
                       dim3(256),
                       0,
                       st,
                       (const int8_t*)q_int8.ptr,
                       (const float*)q_scale.ptr,
                       (const float*)kbar.ptr,
                       (const float*)mu.ptr,
                       (const float*)var.ptr,
                       (int32_t*)tile_list.ptr,
                       (int32_t*)tile_count.ptr,
                       (uint32_t*)mask.ptr,
                       Sp,
                       valid,
                       H,
                       nkb,
                       nkbp,
                       nqt,
                       (int)tile_list.size(1),
                       (int)sink_start,
                       (int)sink_tokens,
                       (float)tau);
}
