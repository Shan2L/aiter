// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

#include "aiter_stream.h"
#include "aiter_tensor.h"
#include "gfx1201_int4_attention.h"
#include "rocm_ops.hpp"

namespace {

int64_t address(const aiter_tensor_t& tensor) { return reinterpret_cast<int64_t>(tensor.ptr); }

int64_t current_stream() { return reinterpret_cast<int64_t>(aiter::getCurrentHIPStream()); }

} // namespace

// q/k (and q_out/k_out, which may alias them): BF16 [S, H, 128]; weights BF16 [128]; cosine/sine FP32 [S, 96].
void gfx1201_int4_qk_norm_rope_hip(const aiter_tensor_t& q,
                                   const aiter_tensor_t& k,
                                   const aiter_tensor_t& q_weight,
                                   const aiter_tensor_t& k_weight,
                                   const aiter_tensor_t& cosine,
                                   const aiter_tensor_t& sine,
                                   aiter_tensor_t& q_out,
                                   aiter_tensor_t& k_out)
{
    launch_gfx1201_int4_qk_norm_rope(address(q), address(k), address(q_weight), address(k_weight), address(cosine),
                                     address(sine), address(q_out), address(k_out), q.size(0), q.size(1),
                                     current_stream());
}

// q/k/v: BF16 [S, H, 128]; part: FP32 scratch [2, ceil(S/128), H, 128]; kmean: FP32 [H, 128];
// q4/k4: INT8 [1, Sp, H, 128]; qs/ks: FP32 [1, H, Sp/32]; v8: FP8 [1, H, 128, Sp]; vs: FP32 [1, H, 128].
void gfx1201_int4_prepare_hip(const aiter_tensor_t& q,
                              const aiter_tensor_t& k,
                              const aiter_tensor_t& v,
                              aiter_tensor_t& part,
                              aiter_tensor_t& kmean,
                              aiter_tensor_t& q4,
                              aiter_tensor_t& k4,
                              aiter_tensor_t& qs,
                              aiter_tensor_t& ks,
                              aiter_tensor_t& v8,
                              aiter_tensor_t& vs,
                              double qmul)
{
    launch_gfx1201_int4_prepare(address(q), address(k), address(v), address(part), address(kmean), address(q4),
                                address(k4), address(qs), address(ks), address(v8), address(vs), q.size(0),
                                q4.size(1), q.size(1), qmul, current_stream());
}

PYBIND11_MODULE(AITER_EXTENSION_NAME, m)
{
    AITER_SET_STREAM_PYBIND;
    m.def("gfx1201_int4_qk_norm_rope_hip", &gfx1201_int4_qk_norm_rope_hip);
    m.def("gfx1201_int4_prepare_hip", &gfx1201_int4_prepare_hip);
}
