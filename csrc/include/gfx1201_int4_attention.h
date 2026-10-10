// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include <cstdint>

// gfx1201 INT4 attention preparation for the ASM cores in hsa/gfx1201/int4_attention.
// Pointers are device addresses, `stream` a hipStream_t.
void launch_gfx1201_int4_qk_norm_rope(int64_t q, int64_t k, int64_t qw, int64_t kw, int64_t cosine, int64_t sine,
                                      int64_t q_out, int64_t k_out, int64_t S, int64_t H, int64_t stream);
void launch_gfx1201_int4_prepare(int64_t q, int64_t k, int64_t v, int64_t part, int64_t kmean, int64_t q4, int64_t k4,
                                 int64_t qs, int64_t ks, int64_t v8, int64_t vs, int64_t S, int64_t Sp, int64_t H,
                                 double qmul, int64_t stream);
