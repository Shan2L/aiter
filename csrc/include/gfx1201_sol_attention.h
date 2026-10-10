// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "aiter_tensor.h"
#include <cstdint>

// gfx1201 Sol sparse attention (batch 1) over gfx1201_sage_prepare outputs; see
// aiter/ops/gfx1201_sol_attention.py for the buffer shapes. W = H * ceil(seq_len / 512) workgroups,
// numbered head * q_tiles + q_tile.

// Key-block summaries, pooled operands for Phase A, per-workgroup tile list / count and Phase-A
// mask.
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
                           double tau);

// Block-summary correction of the unselected key blocks -> initial online-softmax state per
// workgroup.
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
                             int64_t seq_len);
