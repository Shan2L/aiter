// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// gfx1201 SageAttention forward from the hand-scheduled code objects in hsa/gfx1201/sage_attn.
// Same inputs, launch shape and kernel arguments as csrc/kernels/gfx1201_sage_attention.cu,
// including its LIST mode (Sol sparse attention: tile list + initial state, batch 1).
#include "aiter_hip_common.h"
#include "aiter_tensor.h"

#include "aiter_ctypes_error.h"
#include "asm_sage_attn_configs.hpp"

struct __attribute__((packed)) KernelArgs
{
    const void* q_int8;
    const void* k_int8;
    const void* v_fp8;
    const void* q_scale;
    const void* k_scale;
    const void* v_scale;
    void* out;
    int batch;
    int padded_seq_len;
    int seq_len;
    int num_heads;
};
static_assert(sizeof(KernelArgs) == 72, "kernarg layout of the code object");

struct __attribute__((packed)) ListKernelArgs
{
    KernelArgs base;
    const void* tile_list;
    const void* tile_count;
    const void* state;
    int list_stride;
    int pad;
};
static_assert(sizeof(ListKernelArgs) == 104, "kernarg layout of the list code object");

static constexpr int kBlockM    = 512; // query rows per workgroup
static constexpr int kBlockSize = 512; // 16 waves x 32 query rows

AITER_CTYPES_ERROR_DEF

AITER_CTYPES_DEFINE_ENTRYPOINT_VOID(gfx1201_sage_attention_fwd_asm,
                                    (aiter_tensor_t * q_int8,
                                     aiter_tensor_t* q_scale,
                                     aiter_tensor_t* k_int8,
                                     aiter_tensor_t* k_scale,
                                     aiter_tensor_t* v_fp8,
                                     aiter_tensor_t* v_scale,
                                     aiter_tensor_t* out,
                                     int64_t seq_len,
                                     aiter_tensor_t* tile_list,
                                     aiter_tensor_t* tile_count,
                                     aiter_tensor_t* state,
                                     hipStream_t stream),
                                    (q_int8,
                                     q_scale,
                                     k_int8,
                                     k_scale,
                                     v_fp8,
                                     v_scale,
                                     out,
                                     seq_len,
                                     tile_list,
                                     tile_count,
                                     state,
                                     stream))
{
    const HipDeviceGuard device_guard(q_int8->device_id);
    const bool list = tile_list != nullptr;
    AITER_CHECK(list == (tile_count != nullptr) && list == (state != nullptr),
                __func__,
                ": tile_list, tile_count and state must be given together");
    AITER_CHECK(!list || q_int8->size(0) == 1, __func__, ": a tile list requires batch 1");
    const int head_dim         = (int)q_int8->size(3);
    const std::string arch_id  = get_gpu_arch();
    const sage_attnConfig* cfg = nullptr;
    for(const auto& el : cfg_sage_attn)
        if(el.first.find(arch_id) == 0 && el.second.hdim == head_dim && el.second.list == list)
            cfg = &el.second;
    AITER_CHECK(cfg != nullptr,
                __func__,
                ": no code object for arch ",
                arch_id,
                " head dim ",
                head_dim,
                list ? " (list)" : "");

    static SynchronizedCache<std::string_view, AiterAsmKernel> kernels;
    AiterAsmKernel& kernel = kernels.get_or_create(cfg->knl_name, [&]() {
        return AiterAsmKernel(cfg->knl_name.c_str(), cfg->co_name.c_str());
    });

    ListKernelArgs list_args;
    KernelArgs& args    = list_args.base;
    args.q_int8         = q_int8->ptr;
    args.k_int8         = k_int8->ptr;
    args.v_fp8          = v_fp8->ptr;
    args.q_scale        = q_scale->ptr;
    args.k_scale        = k_scale->ptr;
    args.v_scale        = v_scale->ptr;
    args.out            = out->ptr;
    args.batch          = (int)q_int8->size(0);
    args.padded_seq_len = (int)q_int8->size(1);
    args.seq_len        = (int)seq_len;
    args.num_heads      = (int)q_int8->size(2);
    size_t arg_size     = sizeof(args);
    if(list)
    {
        list_args.tile_list   = tile_list->ptr;
        list_args.tile_count  = tile_count->ptr;
        list_args.state       = state->ptr;
        list_args.list_stride = (int)tile_list->size(1);
        list_args.pad         = 0;
        arg_size              = sizeof(list_args);
    }

    const int64_t grid = (int64_t)args.batch * ((seq_len + kBlockM - 1) / kBlockM) * args.num_heads;
    AITER_CHECK(grid < (int64_t(1) << 31), __func__, ": grid too large");
    kernel.launch_kernel({&list_args, &arg_size, (int)grid, 1, 1, kBlockSize, 1, 1, stream});
}
