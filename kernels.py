# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0
# Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
"""Fused integer translation kernel; no persistent input-derived buffers."""

import torch
import triton
import triton.language as tl


@triton.jit
def _translate(slots, mapping, output, N: tl.constexpr, B: tl.constexpr,
               S: tl.constexpr, SHIFT: tl.constexpr, RECIPROCAL: tl.constexpr,
               TILE: tl.constexpr, TABLE: tl.constexpr) -> None:
    index = tl.program_id(0) * TILE + tl.arange(0, TILE)
    layer = tl.program_id(1)
    slot = tl.load(slots + index, index < N, other=-1)
    safe = tl.maximum(slot, 0)
    if SHIFT >= 0:
        logical = safe >> SHIFT
        remainder = safe & (S - 1)
    else:
        # The ceiling reciprocal overestimates the quotient by at most one.
        wide = safe.to(tl.int64)
        estimate = (wide * RECIPROCAL) >> 32
        logical = (estimate - (estimate * S > wide).to(tl.int64)).to(tl.int32)
        remainder = safe - logical * S
    if B <= 8192:
        table_index = tl.arange(0, TABLE)
        table = tl.load(mapping + layer * B + table_index, table_index < B, other=0)
        # Ascend gather accepts fp32 lanes; bitcasts preserve every int32 bit.
        page = tl.gather(table.to(tl.float32, bitcast=True), logical, axis=0).to(tl.int32, bitcast=True)
    else:
        page = tl.load(mapping + layer * B + logical,
                       (index < N) & (slot >= 0), other=0)
    physical = page * S + remainder
    tl.store(output + layer * N + index, tl.where(slot < 0, -1, physical), index < N)


def launch(slots: torch.Tensor, mapping: torch.Tensor, output: torch.Tensor,
           block_size: int) -> None:
    """Launch with live tensor pointers and the runtime's current stream."""
    layers = 1 if mapping.ndim == 1 else mapping.shape[0]
    shift = block_size.bit_length() - 1 if block_size & (block_size - 1) == 0 else -1
    reciprocal = ((1 << 32) + block_size - 1) // block_size
    tile = max(256, min(2048, triton.next_power_of_2(slots.numel())))
    _translate[(triton.cdiv(slots.numel(), tile), layers)](
        slots, mapping, output, slots.numel(), mapping.shape[-1],
        block_size, shift, reciprocal, tile,
        min(8192, triton.next_power_of_2(mapping.shape[-1])))
