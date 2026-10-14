// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

// gfx1201 SageAttention core: INT8 QK^T and FP8 PV on WMMA with FP32 accumulation and online
// softmax, over the tensors written by gfx1201_sage_prepare. Non-causal, head dim 128, BF16 output.
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>
#include <math.h>
#include <stdint.h>

#include <stdexcept>
#include <type_traits>

#include "aiter_stream.h"
#include "gfx1201_sage_attention.h"
#include "gfx1201_sol_attention.h"

using int32x2_t   = int __attribute__((ext_vector_type(2)));
using int32x8_t   = int __attribute__((ext_vector_type(8)));
using float32x8_t = float __attribute__((ext_vector_type(8)));

static constexpr int WARP              = 32;
static constexpr int WMMA_M            = 16;
static constexpr int WMMA_N            = 16;
static constexpr int WMMA_K            = 16;
static constexpr int WMMA_LANE_K       = 8;
static constexpr int BLOCK_M           = 512;
static constexpr int BLOCK_N           = 32;
static constexpr int HEAD_DIM          = 128;
static constexpr int ROWS_PER_WAVE     = 32;
static constexpr int ROW_GROUPS        = ROWS_PER_WAVE / WMMA_M;                       // 2
static constexpr int NUM_WAVES         = BLOCK_M / ROWS_PER_WAVE;                      // 16
static constexpr int BLOCK_SIZE        = NUM_WAVES * WARP;                             // 512
static constexpr int K_THREADS_PER_ROW = HEAD_DIM / 16;                                // 8
static constexpr int K_LOAD_ITEMS      = BLOCK_N * K_THREADS_PER_ROW;                  // 256
static constexpr int K_LOAD_BATCHES    = (K_LOAD_ITEMS + BLOCK_SIZE - 1) / BLOCK_SIZE; // 1
static constexpr int V_CHUNKS_PER_ROW  = BLOCK_N / 16;                                 // 2
static constexpr int V_LOAD_ITEMS      = HEAD_DIM * V_CHUNKS_PER_ROW;                  // 256
static constexpr int V_LOAD_BATCHES    = (V_LOAD_ITEMS + BLOCK_SIZE - 1) / BLOCK_SIZE; // 1
static constexpr int K_SUB_N           = 32;
static constexpr int N_SUB_TILES       = BLOCK_N / K_SUB_N; // 1
static constexpr int NUM_S_ACCS        = N_SUB_TILES * 2;   // 2
static constexpr int NUM_S_VALS        = NUM_S_ACCS * 8;    // 16
static constexpr int K_STEPS_QK        = HEAD_DIM / WMMA_K; // 8
static constexpr int D_CHUNKS          = HEAD_DIM / WMMA_N; // 8
static constexpr int PV_K_STEPS        = K_SUB_N / WMMA_K;  // 2
static constexpr int Q_SCALE_ROWS      = 32;
static constexpr int VEC_WIDTH         = 16;
static constexpr int LDS_PAD           = 8;
static constexpr int K_STRIDE          = HEAD_DIM + LDS_PAD;  // 136
static constexpr int V_STRIDE          = BLOCK_N + LDS_PAD;   // 40
static constexpr int LDS_K_TILE        = BLOCK_N * K_STRIDE;  // 4352
static constexpr int LDS_V_TILE        = HEAD_DIM * V_STRIDE; // 5120
static constexpr int LDS_BUFFERS       = 2;
static constexpr int LDS_V_BASE        = LDS_BUFFERS * LDS_K_TILE;
// P = exp2(s - m + log2(448)) so the row maximum lands at the top of the FP8 e4m3 range.
static constexpr float FP8_P_OFFSET = 8.807f;

__device__ __forceinline__ int32x2_t pack_i8x8(const int8_t* p)
{
    int32x2_t out;
    out[0] = *reinterpret_cast<const int32_t*>(p);
    out[1] = *reinterpret_cast<const int32_t*>(p + 4);
    return out;
}

__device__ __forceinline__ int32x8_t wmma_qk_i8(int32x2_t a, int32x2_t b, int32x8_t c)
{
#if defined(__gfx1201__) || defined(__gfx12__)
    // LLVM: (A_signed, A, B_signed, B, C, clamp) — signed to match FlyDSL.
    return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a, true, b, c, false);
#else
    (void)a;
    (void)b;
    return c;
#endif
}

__device__ __forceinline__ float32x8_t wmma_pv_fp8(int32x2_t a, int32x2_t b, float32x8_t c)
{
#if defined(__gfx1201__) || defined(__gfx12__)
    return __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, c);
#else
    (void)a;
    (void)b;
    return c;
#endif
}

__device__ __forceinline__ int32x2_t pack_fp8_prob(const float p[8])
{
    // Pack 8 f32 probabilities into 8 e4m3 bytes (two i32).
    int32_t lo = 0, hi = 0;
#if defined(__gfx1201__) || defined(__gfx12__)
    lo = __builtin_amdgcn_cvt_pk_fp8_f32(p[0], p[1], lo, false);
    lo = __builtin_amdgcn_cvt_pk_fp8_f32(p[2], p[3], lo, true);
    hi = __builtin_amdgcn_cvt_pk_fp8_f32(p[4], p[5], hi, false);
    hi = __builtin_amdgcn_cvt_pk_fp8_f32(p[6], p[7], hi, true);
#else
    (void)p;
#endif
    int32x2_t out;
    out[0] = lo;
    out[1] = hi;
    return out;
}

__device__ __forceinline__ float shfl_xor_f(float v, int mask) { return __shfl_xor(v, mask, WARP); }

__device__ __forceinline__ float exp2f_fast(float x) { return __builtin_amdgcn_exp2f(x); }

// Grid: batch * q_tiles * num_heads. Block: 512 threads = 16 waves x 32 query rows.
// Two LDS buffers: compute reads buf[i%2] while the next tile sits in registers
// and is published to buf[(i+1)%2] behind a single barrier.
// LIST (Sol sparse attention, batch 1): workgroup w = head * q_tiles + q_tile visits only the key
// tiles in tile_list[w] and starts from the online-softmax state written by gfx1201_sol_phase_a.
template <bool LIST>
__global__
__launch_bounds__(BLOCK_SIZE,
                  2) void gfx1201_sage_attention_kernel(const int8_t* __restrict__ Q,
                                                        const int8_t* __restrict__ K,
                                                        const int8_t* __restrict__ V,
                                                        const float* __restrict__ QScale,
                                                        const float* __restrict__ KScale,
                                                        const float* __restrict__ VScale,
                                                        __hip_bfloat16* __restrict__ O,
                                                        int batch_size,
                                                        int padded_seq_len,
                                                        int valid_seq_len,
                                                        int num_heads,
                                                        const int32_t* __restrict__ tile_list,
                                                        const int32_t* __restrict__ tile_count,
                                                        const float* __restrict__ state,
                                                        int list_stride)
{
    __shared__ int8_t lds[LDS_BUFFERS * (LDS_K_TILE + LDS_V_TILE)];

    const int tid     = (int)threadIdx.x;
    const int wave_id = tid / WARP;
    const int lane    = tid % WARP;
    const int lane16  = lane % 16;
    const int klane   = lane / 16;

    const int q_tiles    = (valid_seq_len + BLOCK_M - 1) / BLOCK_M;
    const int block_id   = (int)blockIdx.x;
    const int head_idx   = LIST ? block_id / q_tiles : block_id % num_heads;
    const int q_tile_idx = LIST ? block_id % q_tiles : block_id / num_heads % q_tiles;
    const int batch_idx  = LIST ? 0 : block_id / num_heads / q_tiles;
    const int q_start    = q_tile_idx * BLOCK_M;
    const int seq        = padded_seq_len;

    int q_row[ROW_GROUPS];
    bool q_ok[ROW_GROUPS];
#pragma unroll
    for(int rg = 0; rg < ROW_GROUPS; ++rg)
    {
        q_row[rg] = q_start + wave_id * ROWS_PER_WAVE + rg * WMMA_M + lane16;
        q_ok[rg]  = q_row[rg] < valid_seq_len;
    }

    auto qk_idx = [&](int token, int col) -> long long {
        return ((((long long)batch_idx * seq + token) * num_heads + head_idx) * HEAD_DIM + col);
    };
    auto v_idx = [&](int d, int token) -> long long {
        return ((((long long)batch_idx * num_heads + head_idx) * HEAD_DIM + d) * seq + token);
    };

    int32x2_t q_frags[ROW_GROUPS][K_STEPS_QK];
#pragma unroll
    for(int rg = 0; rg < ROW_GROUPS; ++rg)
    {
        const int q_safe = q_ok[rg] ? q_row[rg] : 0;
#pragma unroll
        for(int ks = 0; ks < K_STEPS_QK; ++ks)
        {
            const int q_col = ks * WMMA_K + klane * WMMA_LANE_K;
            int8_t tmp[8]   = {0, 0, 0, 0, 0, 0, 0, 0};
            if(q_ok[rg])
            {
                *reinterpret_cast<int64_t*>(tmp) =
                    *reinterpret_cast<const int64_t*>(Q + qk_idx(q_safe, q_col));
            }
            q_frags[rg][ks] = pack_i8x8(tmp);
        }
    }

    const int q_scale_blocks = seq / Q_SCALE_ROWS;
    const int q_safe0        = q_ok[0] ? q_row[0] : 0;
    const float q_scale =
        QScale[((batch_idx * num_heads + head_idx) * q_scale_blocks) + (q_safe0 / Q_SCALE_ROWS)];

    float m_running[ROW_GROUPS], l_running[ROW_GROUPS];
    float32x8_t o_acc[ROW_GROUPS][D_CHUNKS];
#pragma unroll
    for(int rg = 0; rg < ROW_GROUPS; ++rg)
    {
        m_running[rg] = -INFINITY;
        l_running[rg] = 0.f;
#pragma unroll
        for(int dc = 0; dc < D_CHUNKS; ++dc)
            o_acc[rg][dc] = (float32x8_t){0, 0, 0, 0, 0, 0, 0, 0};
    }
    if constexpr(LIST)
    {
        // State layout [workgroup][wave][33 chunks][lane][4 floats]: o element (rg, dc, i) is
        // float r = (1 - rg) * 64 + (7 - dc) * 8 + i at chunk r / 4; chunk 32 = (l1, l0, m1, m0).
        const float* st =
            state + (((long long)block_id * NUM_WAVES + wave_id) * 33 * WARP + lane) * 4;
#pragma unroll
        for(int rg = 0; rg < ROW_GROUPS; ++rg)
#pragma unroll
            for(int dc = 0; dc < D_CHUNKS; ++dc)
#pragma unroll
                for(int i = 0; i < 8; ++i)
                {
                    const int r      = (1 - rg) * 64 + (7 - dc) * 8 + i;
                    o_acc[rg][dc][i] = st[r / 4 * WARP * 4 + r % 4];
                }
        const float4 lm = *reinterpret_cast<const float4*>(st + 32 * WARP * 4);
        l_running[1]    = lm.x;
        l_running[0]    = lm.y;
        m_running[1]    = lm.z;
        m_running[0]    = lm.w;
    }

    int4 k_pref[K_LOAD_BATCHES];
    int4 v_pref[V_LOAD_BATCHES];
    auto prefetch_k = [&](int tile) {
#pragma unroll
        for(int b = 0; b < K_LOAD_BATCHES; ++b)
        {
            const int item = tid + b * BLOCK_SIZE;
            if(item >= K_LOAD_ITEMS)
                continue;
            const int k_row = item / K_THREADS_PER_ROW;
            const int k_col = (item % K_THREADS_PER_ROW) * VEC_WIDTH;
            k_pref[b]       = *reinterpret_cast<const int4*>(K + qk_idx(tile + k_row, k_col));
        }
    };
    auto prefetch_v = [&](int tile) {
#pragma unroll
        for(int b = 0; b < V_LOAD_BATCHES; ++b)
        {
            const int item = tid + b * BLOCK_SIZE;
            if(item >= V_LOAD_ITEMS)
                continue;
            const int d_row     = item / V_CHUNKS_PER_ROW;
            const int token_off = (item % V_CHUNKS_PER_ROW) * VEC_WIDTH;
            v_pref[b] = *reinterpret_cast<const int4*>(V + v_idx(d_row, tile + token_off));
        }
    };
    auto commit_k = [&](int buf) {
#pragma unroll
        for(int b = 0; b < K_LOAD_BATCHES; ++b)
        {
            const int item = tid + b * BLOCK_SIZE;
            if(item >= K_LOAD_ITEMS)
                continue;
            const int k_row = item / K_THREADS_PER_ROW;
            const int k_col = (item % K_THREADS_PER_ROW) * VEC_WIDTH;
            *reinterpret_cast<int4*>(lds + buf * LDS_K_TILE + k_row * K_STRIDE + k_col) = k_pref[b];
        }
    };
    auto commit_v = [&](int buf) {
#pragma unroll
        for(int b = 0; b < V_LOAD_BATCHES; ++b)
        {
            const int item = tid + b * BLOCK_SIZE;
            if(item >= V_LOAD_ITEMS)
                continue;
            const int d_row                     = item / V_CHUNKS_PER_ROW;
            const int token_off                 = (item % V_CHUNKS_PER_ROW) * VEC_WIDTH;
            *reinterpret_cast<int4*>(lds + LDS_V_BASE + buf * LDS_V_TILE + d_row * V_STRIDE +
                                     token_off) = v_pref[b];
        }
    };
    auto safe_tile = [&](int tile) -> int { return tile < seq ? tile : 0; };

    // Full key tiles in processing order, then the partial tail tile (if any) at index n_full.
    // A Sol tile list stores the tail start and a dummy entry after its n_full tiles.
    const int32_t* list = LIST ? tile_list + (long long)block_id * list_stride : nullptr;
    const int n_full    = LIST ? tile_count[block_id] : valid_seq_len / BLOCK_N;
    auto tile_at        = [&](int idx) -> int {
        if constexpr(LIST)
            return idx < n_full + 2 ? list[idx] : 0;
        else
            return idx * BLOCK_N;
    };

    // Tile 0 into buf 0; tile 1 stays in regs until the first loop barrier.
    prefetch_k(safe_tile(tile_at(0)));
    prefetch_v(safe_tile(tile_at(0)));
    commit_k(0);
    commit_v(0);
    prefetch_k(safe_tile(tile_at(1)));
    prefetch_v(safe_tile(tile_at(1)));

    auto process_tile = [&](int idx, auto tail_tag) {
        constexpr bool MASK_TAIL = decltype(tail_tag)::value;
        const int kv_start       = tile_at(idx);
        const int cur            = idx & 1;
        const int nxt            = cur ^ 1;
        __syncthreads();
        // Publish the prefetched tile into the idle buffer, then issue the
        // tile two ahead so those global loads overlap this iteration's WMMA.
        commit_k(nxt);
        commit_v(nxt);
        prefetch_k(safe_tile(tile_at(idx + 2)));
        prefetch_v(safe_tile(tile_at(idx + 2)));
        const int8_t* k_lds = lds + cur * LDS_K_TILE;
        const int8_t* v_lds = lds + LDS_V_BASE + cur * LDS_V_TILE;

        int32x8_t s_acc[ROW_GROUPS][NUM_S_ACCS];
#pragma unroll
        for(int rg = 0; rg < ROW_GROUPS; ++rg)
#pragma unroll
            for(int i = 0; i < NUM_S_ACCS; ++i)
                s_acc[rg][i] = (int32x8_t){0, 0, 0, 0, 0, 0, 0, 0};

#pragma unroll
        for(int ks = 0; ks < K_STEPS_QK; ++ks)
        {
            const int k_col = ks * WMMA_K + klane * WMMA_LANE_K;
            int8_t ka[8], kb[8];
            *reinterpret_cast<int64_t*>(ka) =
                *reinterpret_cast<const int64_t*>(k_lds + lane16 * K_STRIDE + k_col);
            *reinterpret_cast<int64_t*>(kb) =
                *reinterpret_cast<const int64_t*>(k_lds + (lane16 + 16) * K_STRIDE + k_col);
            int32x2_t pk_a = pack_i8x8(ka);
            int32x2_t pk_b = pack_i8x8(kb);
#pragma unroll
            for(int rg = 0; rg < ROW_GROUPS; ++rg)
            {
                s_acc[rg][0] = wmma_qk_i8(pk_a, q_frags[rg][ks], s_acc[rg][0]);
                s_acc[rg][1] = wmma_qk_i8(pk_b, q_frags[rg][ks], s_acc[rg][1]);
            }
        }

        const int k_scale_blocks = seq / BLOCK_N;
        const float k_scale =
            KScale[((batch_idx * num_heads + head_idx) * k_scale_blocks) + (kv_start / BLOCK_N)];
        const float score_scale = q_scale * k_scale;

        int32x2_t p_frags[ROW_GROUPS][PV_K_STEPS];
#pragma unroll
        for(int rg = 0; rg < ROW_GROUPS; ++rg)
        {
            float raw[NUM_S_VALS];
#pragma unroll
            for(int st = 0; st < NUM_S_ACCS; ++st)
#pragma unroll
                for(int item = 0; item < 8; ++item)
                    raw[st * 8 + item] = (float)s_acc[rg][st][item];
            if constexpr(MASK_TAIL)
            {
#pragma unroll
                for(int item = 0; item < NUM_S_VALS; ++item)
                    if(kv_start + (item / 8) * 16 + klane * 8 + item % 8 >= valid_seq_len)
                        raw[item] = -3.402823466e+38F;
            }

            float local_max = raw[0];
#pragma unroll
            for(int i = 1; i < NUM_S_VALS; ++i)
                local_max = fmaxf(local_max, raw[i]);
            local_max           = fmaxf(local_max, shfl_xor_f(local_max, 16));
            const float row_max = fmaf(local_max, score_scale, -FP8_P_OFFSET);
            const float m_new   = fmaxf(m_running[rg], row_max);
            const float corr    = exp2f_fast(m_running[rg] - m_new);

            float p_vals[NUM_S_VALS];
            float local_sum   = 0.f;
            const float neg_m = -m_new;
            if constexpr(!MASK_TAIL)
            {
#pragma unroll
                for(int i = 0; i < NUM_S_VALS; ++i)
                {
                    const float p = exp2f_fast(fmaf(raw[i], score_scale, neg_m));
                    p_vals[i]     = p;
                    local_sum += p;
                }
            }
            else
            {
#pragma unroll
                for(int item = 0; item < NUM_S_VALS; ++item)
                {
                    const bool key_valid =
                        kv_start + (item / 8) * 16 + klane * 8 + item % 8 < valid_seq_len;
                    const float probability =
                        key_valid ? exp2f_fast(fmaf(raw[item], score_scale, neg_m)) : 0.f;
                    p_vals[item] = probability;
                    local_sum += probability;
                }
            }
            const float tile_sum = local_sum + shfl_xor_f(local_sum, 16);
            l_running[rg]        = corr * l_running[rg] + tile_sum;
            m_running[rg]        = m_new;
            if(corr != 1.f)
            {
#pragma unroll
                for(int dc = 0; dc < D_CHUNKS; ++dc)
#pragma unroll
                    for(int i = 0; i < 8; ++i)
                        o_acc[rg][dc][i] *= corr;
            }

#pragma unroll
            for(int pks = 0; pks < PV_K_STEPS; ++pks)
                p_frags[rg][pks] = pack_fp8_prob(&p_vals[pks * 8]);
        }

#pragma unroll
        for(int pks = 0; pks < PV_K_STEPS; ++pks)
        {
#pragma unroll
            for(int dc = 0; dc < D_CHUNKS; ++dc)
            {
                const int d_pos     = dc * WMMA_N + lane16;
                const int token_pos = pks * WMMA_K + klane * WMMA_LANE_K;
                int8_t vb[8];
                *reinterpret_cast<int64_t*>(vb) =
                    *reinterpret_cast<const int64_t*>(v_lds + d_pos * V_STRIDE + token_pos);
                int32x2_t pv = pack_i8x8(vb);
#pragma unroll
                for(int rg = 0; rg < ROW_GROUPS; ++rg)
                    o_acc[rg][dc] = wmma_pv_fp8(pv, p_frags[rg][pks], o_acc[rg][dc]);
            }
        }
    };
    for(int idx = 0; idx < n_full; ++idx)
        process_tile(idx, std::false_type{});
    if(valid_seq_len % BLOCK_N)
        process_tile(n_full, std::true_type{});

#pragma unroll
    for(int rg = 0; rg < ROW_GROUPS; ++rg)
    {
        if(!q_ok[rg])
            continue;
        const float inv_l = 1.f / l_running[rg];
#pragma unroll
        for(int dc = 0; dc < D_CHUNKS; ++dc)
        {
            const int d_col = dc * WMMA_N + klane * 8;
            const long long scale_index =
                ((long long)batch_idx * num_heads + head_idx) * HEAD_DIM + d_col;
            float vs[8];
            *reinterpret_cast<float4*>(vs) = *reinterpret_cast<const float4*>(VScale + scale_index);
            *reinterpret_cast<float4*>(vs + 4) =
                *reinterpret_cast<const float4*>(VScale + scale_index + 4);
            __hip_bfloat16 outv[8];
#pragma unroll
            for(int i = 0; i < 8; ++i)
                outv[i] = __float2bfloat16(o_acc[rg][dc][i] * inv_l * vs[i]);
            *reinterpret_cast<int4*>(O + qk_idx(q_row[rg], d_col)) = *reinterpret_cast<int4*>(outv);
        }
    }
}

// Sol Phase A: Sol-Attn block-summary correction for the key blocks outside each workgroup's exact
// union. Pooled keys kb (INT8, per-32-block scale) and value sums vb (FP8, per-channel scale) go
// through the core's QK/softmax/PV pipeline with each block's logit raised by log2 of its length;
// the result is written as the initial state of the LIST core. Workgroup w = head * q_tiles +
// q_tile.
static constexpr float SOL_MASKED = -3.0e38f;

__global__ __launch_bounds__(BLOCK_SIZE,
                             2) void gfx1201_sol_phase_a_kernel(const int8_t* __restrict__ Q,
                                                                const float* __restrict__ QScale,
                                                                const float* __restrict__ VScale,
                                                                const int8_t* __restrict__ KB,
                                                                const uint8_t* __restrict__ VB,
                                                                const float* __restrict__ KBScale,
                                                                const float* __restrict__ VBScale,
                                                                const float* __restrict__ Cnt,
                                                                const uint32_t* __restrict__ Mask,
                                                                float* __restrict__ State,
                                                                int padded_seq_len,
                                                                int valid_seq_len,
                                                                int num_heads,
                                                                int nkb,
                                                                int nkbp)
{
    constexpr int LDS_BUF = LDS_K_TILE + LDS_V_TILE;
    __shared__ int8_t lds_all[2 * LDS_BUF];
    const int tid      = (int)threadIdx.x;
    const int wave_id  = tid / WARP;
    const int lane     = tid % WARP;
    const int lane16   = lane % 16;
    const int klane    = lane / 16;
    const int H        = num_heads;
    const int nqt      = (valid_seq_len + BLOCK_M - 1) / BLOCK_M;
    const int head     = (int)blockIdx.x / nqt;
    const int qt       = (int)blockIdx.x % nqt;
    const long long wg = (long long)blockIdx.x;

    int q_row[ROW_GROUPS];
    bool q_ok[ROW_GROUPS];
#pragma unroll
    for(int rg = 0; rg < ROW_GROUPS; ++rg)
    {
        q_row[rg] = qt * BLOCK_M + wave_id * ROWS_PER_WAVE + rg * WMMA_M + lane16;
        q_ok[rg]  = q_row[rg] < valid_seq_len;
    }
    int32x2_t q_frags[ROW_GROUPS][K_STEPS_QK];
#pragma unroll
    for(int rg = 0; rg < ROW_GROUPS; ++rg)
#pragma unroll
        for(int ks = 0; ks < K_STEPS_QK; ++ks)
        {
            int8_t tmp[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            if(q_ok[rg])
                *reinterpret_cast<int64_t*>(tmp) = *reinterpret_cast<const int64_t*>(
                    Q + (((long long)q_row[rg] * H + head) * HEAD_DIM + ks * WMMA_K +
                         klane * WMMA_LANE_K));
            q_frags[rg][ks] = pack_i8x8(tmp);
        }
    const int row0      = qt * BLOCK_M + wave_id * ROWS_PER_WAVE; // one Q scale per wave
    const float q_scale = QScale[(long long)head * (padded_seq_len / Q_SCALE_ROWS) +
                                 (row0 < valid_seq_len ? row0 / Q_SCALE_ROWS : 0)];

    float m_run[ROW_GROUPS], l_run[ROW_GROUPS];
    float32x8_t o_acc[ROW_GROUPS][D_CHUNKS];
#pragma unroll
    for(int rg = 0; rg < ROW_GROUPS; ++rg)
    {
        m_run[rg] = -1.0e30f;
        l_run[rg] = 0.f;
#pragma unroll
        for(int dc = 0; dc < D_CHUNKS; ++dc)
            o_acc[rg][dc] = (float32x8_t){0, 0, 0, 0, 0, 0, 0, 0};
    }

    // Threads < 256 move one int4 of the pooled-K tile, the others one int4 of the pooled-V^T tile.
    const bool is_k = tid < 256;
    const int k_r = (tid & 255) / 8, k_c = ((tid & 255) % 8) * 16;
    const int v_d = (tid & 255) / 2, v_off = ((tid & 255) % 2) * 16;
    auto fetch = [&](int blk) -> int4 {
        if(blk >= nkbp)
            return make_int4(0, 0, 0, 0);
        return is_k ? *reinterpret_cast<const int4*>(
                          KB + (((long long)(blk + k_r) * H + head) * HEAD_DIM + k_c))
                    : *reinterpret_cast<const int4*>(
                          VB + (((long long)head * HEAD_DIM + v_d) * nkbp + blk + v_off));
    };
    auto commit = [&](int buf, int4 val) {
        int8_t* b = lds_all + buf * LDS_BUF;
        if(is_k)
            *reinterpret_cast<int4*>(b + k_r * K_STRIDE + k_c) = val;
        else
            *reinterpret_cast<int4*>(b + LDS_K_TILE + v_d * V_STRIDE + v_off) = val;
    };

    const uint32_t* mask_row = Mask + wg * (nkbp / 32);
    const int n_pool         = nkbp / 32;
    bool any_approx          = false;
    for(int t = 0; t < n_pool && !any_approx; ++t)
        any_approx = mask_row[t] != 0xffffffffu;

    if(any_approx)
    {
        int4 reg = fetch(0);
        for(int t = 0; t < n_pool; ++t)
        {
            __syncthreads();
            commit(t & 1, reg);
            if(t + 1 < n_pool)
                reg = fetch((t + 1) * 32);
            __syncthreads();
            const uint32_t bits = mask_row[t];
            if(bits == 0xffffffffu)
                continue;
            const int8_t* lds = lds_all + (t & 1) * LDS_BUF;

            int32x8_t sacc[ROW_GROUPS][2];
#pragma unroll
            for(int rg = 0; rg < ROW_GROUPS; ++rg)
                sacc[rg][0] = sacc[rg][1] = (int32x8_t){0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
            for(int ks = 0; ks < K_STEPS_QK; ++ks)
            {
                const int k_col = ks * WMMA_K + klane * WMMA_LANE_K;
                int32x2_t a     = pack_i8x8(lds + lane16 * K_STRIDE + k_col);
                int32x2_t b     = pack_i8x8(lds + (lane16 + 16) * K_STRIDE + k_col);
#pragma unroll
                for(int rg = 0; rg < ROW_GROUPS; ++rg)
                {
                    sacc[rg][0] = wmma_qk_i8(a, q_frags[rg][ks], sacc[rg][0]);
                    sacc[rg][1] = wmma_qk_i8(b, q_frags[rg][ks], sacc[rg][1]);
                }
            }
            float raw[ROW_GROUPS][NUM_S_VALS], weight[NUM_S_VALS];
#pragma unroll
            for(int rg = 0; rg < ROW_GROUPS; ++rg)
#pragma unroll
                for(int i = 0; i < 8; ++i)
                {
                    raw[rg][i]     = (float)sacc[rg][0][i];
                    raw[rg][8 + i] = (float)sacc[rg][1][i];
                }
#pragma unroll
            for(int i = 0; i < NUM_S_VALS; ++i)
            {
                const int j    = (i / 8) * 16 + klane * 8 + (i % 8);
                const int blk  = t * 32 + j;
                const bool use = blk < nkb && !((bits >> j) & 1u);
                weight[i]      = use ? Cnt[blk] : 0.f;
                if(!use)
#pragma unroll
                    for(int rg = 0; rg < ROW_GROUPS; ++rg)
                        raw[rg][i] = SOL_MASKED;
            }
            const float score_scale = q_scale * KBScale[(long long)head * n_pool + t];

            int32x2_t pf[ROW_GROUPS][2];
#pragma unroll
            for(int rg = 0; rg < ROW_GROUPS; ++rg)
            {
                float local_max = raw[rg][0];
#pragma unroll
                for(int i = 1; i < NUM_S_VALS; ++i)
                    local_max = fmaxf(local_max, raw[rg][i]);
                local_max           = fmaxf(local_max, shfl_xor_f(local_max, 16));
                const float row_max = local_max <= SOL_MASKED
                                          ? -1.0e30f
                                          : fmaf(local_max, score_scale, -FP8_P_OFFSET);
                const float m_new   = fmaxf(m_run[rg], row_max);
                const float corr    = exp2f_fast(m_run[rg] - m_new);
                float pv[NUM_S_VALS];
                float local_sum = 0.f;
#pragma unroll
                for(int i = 0; i < NUM_S_VALS; ++i)
                {
                    const float e = raw[rg][i] <= SOL_MASKED
                                        ? 0.f
                                        : exp2f_fast(fmaf(raw[rg][i], score_scale, -m_new));
                    pv[i]         = e;
                    local_sum += e * weight[i];
                }
                l_run[rg] = corr * l_run[rg] + local_sum + shfl_xor_f(local_sum, 16);
                m_run[rg] = m_new;
                if(corr != 1.f)
#pragma unroll
                    for(int dc = 0; dc < D_CHUNKS; ++dc)
#pragma unroll
                        for(int i = 0; i < 8; ++i)
                            o_acc[rg][dc][i] *= corr;
                pf[rg][0] = pack_fp8_prob(&pv[0]);
                pf[rg][1] = pack_fp8_prob(&pv[8]);
            }
#pragma unroll
            for(int dc = 0; dc < D_CHUNKS; ++dc)
            {
                const int d_pos = dc * WMMA_N + lane16;
                int32x2_t v0 = pack_i8x8(lds + LDS_K_TILE + d_pos * V_STRIDE + klane * WMMA_LANE_K);
                int32x2_t v1 =
                    pack_i8x8(lds + LDS_K_TILE + d_pos * V_STRIDE + 16 + klane * WMMA_LANE_K);
#pragma unroll
                for(int rg = 0; rg < ROW_GROUPS; ++rg)
                {
                    o_acc[rg][dc] = wmma_pv_fp8(v0, pf[rg][0], o_acc[rg][dc]);
                    o_acc[rg][dc] = wmma_pv_fp8(v1, pf[rg][1], o_acc[rg][dc]);
                }
            }
        }
        // The core scales o by the per-channel V scale at the end; rescale the pooled-V part to it.
#pragma unroll
        for(int dc = 0; dc < D_CHUNKS; ++dc)
#pragma unroll
            for(int i = 0; i < 8; ++i)
            {
                const int c       = dc * WMMA_N + klane * 8 + i;
                const float vs    = VScale[(long long)head * HEAD_DIM + c];
                const float ratio = vs > 0.f ? VBScale[(long long)head * HEAD_DIM + c] / vs : 0.f;
#pragma unroll
                for(int rg = 0; rg < ROW_GROUPS; ++rg)
                    o_acc[rg][dc][i] *= ratio;
            }
    }

    // State layout read by the LIST core: see gfx1201_sage_attention_kernel.
    float* st = State + ((wg * NUM_WAVES + wave_id) * 33 * WARP + lane) * 4;
#pragma unroll
    for(int rg = 0; rg < ROW_GROUPS; ++rg)
#pragma unroll
        for(int dc = 0; dc < D_CHUNKS; ++dc)
#pragma unroll
            for(int i4 = 0; i4 < 2; ++i4)
            {
                const int r = (1 - rg) * 64 + (7 - dc) * 8 + i4 * 4;
                *reinterpret_cast<float4*>(st + r / 4 * WARP * 4) =
                    make_float4(o_acc[rg][dc][i4 * 4],
                                o_acc[rg][dc][i4 * 4 + 1],
                                o_acc[rg][dc][i4 * 4 + 2],
                                o_acc[rg][dc][i4 * 4 + 3]);
            }
    *reinterpret_cast<float4*>(st + 32 * WARP * 4) =
        any_approx ? make_float4(l_run[1], l_run[0], m_run[1], m_run[0])
                   : make_float4(0.f, 0.f, -INFINITY, -INFINITY);
}

void gfx1201_sage_attention_fwd_hip(const aiter_tensor_t& q_int8,
                                    const aiter_tensor_t& q_scale,
                                    const aiter_tensor_t& k_int8,
                                    const aiter_tensor_t& k_scale,
                                    const aiter_tensor_t& v_fp8,
                                    const aiter_tensor_t& v_scale,
                                    aiter_tensor_t& out,
                                    int64_t seq_len,
                                    std::optional<aiter_tensor_t> tile_list,
                                    std::optional<aiter_tensor_t> tile_count,
                                    std::optional<aiter_tensor_t> state)
{
    const int batch = (int)q_int8.size(0), padded = (int)q_int8.size(1),
              heads   = (int)q_int8.size(2);
    const int q_tiles = ((int)seq_len + BLOCK_M - 1) / BLOCK_M;
    const bool list   = tile_list.has_value();
    if(list != tile_count.has_value() || list != state.has_value())
        throw std::invalid_argument(
            "gfx1201_sage_attention: tile_list, tile_count and state must be given together");
    if(list && batch != 1)
        throw std::invalid_argument("gfx1201_sage_attention: a tile list requires batch 1");
    auto launch =
        [&](auto kernel, const void* tiles, const void* counts, const void* init, int stride) {
            hipLaunchKernelGGL(kernel,
                               dim3(batch * q_tiles * heads),
                               dim3(BLOCK_SIZE),
                               0,
                               aiter::getCurrentHIPStream(),
                               (const int8_t*)q_int8.ptr,
                               (const int8_t*)k_int8.ptr,
                               (const int8_t*)v_fp8.ptr,
                               (const float*)q_scale.ptr,
                               (const float*)k_scale.ptr,
                               (const float*)v_scale.ptr,
                               (__hip_bfloat16*)out.ptr,
                               batch,
                               padded,
                               (int)seq_len,
                               heads,
                               (const int32_t*)tiles,
                               (const int32_t*)counts,
                               (const float*)init,
                               stride);
        };
    if(list)
        launch(gfx1201_sage_attention_kernel<true>,
               tile_list->ptr,
               tile_count->ptr,
               state->ptr,
               (int)tile_list->size(1));
    else
        launch(gfx1201_sage_attention_kernel<false>, nullptr, nullptr, nullptr, 0);
}

void gfx1201_sol_phase_a_hip(const aiter_tensor_t& q_int8,
                             const aiter_tensor_t& q_scale,
                             const aiter_tensor_t& v_scale,
                             const aiter_tensor_t& kb,
                             const aiter_tensor_t& vb,
                             const aiter_tensor_t& kb_scale,
                             const aiter_tensor_t& vb_scale,
                             const aiter_tensor_t& block_len,
                             const aiter_tensor_t& mask,
                             aiter_tensor_t& state,
                             int64_t seq_len)
{
    const int padded = (int)q_int8.size(1), heads = (int)q_int8.size(2);
    const int nkbp = (int)kb.size(0);
    const int nkb  = ((int)seq_len + 2 * BLOCK_N - 1) / (2 * BLOCK_N);
    if(q_int8.size(0) != 1 || nkbp % 32 || nkbp < nkb)
        throw std::invalid_argument("gfx1201_sol_phase_a: invalid shape");
    hipLaunchKernelGGL(gfx1201_sol_phase_a_kernel,
                       dim3((int)mask.size(0)),
                       dim3(BLOCK_SIZE),
                       0,
                       aiter::getCurrentHIPStream(),
                       (const int8_t*)q_int8.ptr,
                       (const float*)q_scale.ptr,
                       (const float*)v_scale.ptr,
                       (const int8_t*)kb.ptr,
                       (const uint8_t*)vb.ptr,
                       (const float*)kb_scale.ptr,
                       (const float*)vb_scale.ptr,
                       (const float*)block_len.ptr,
                       (const uint32_t*)mask.ptr,
                       (float*)state.ptr,
                       padded,
                       (int)seq_len,
                       heads,
                       nkb,
                       nkbp);
}
