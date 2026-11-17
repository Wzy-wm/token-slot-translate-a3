# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0
# Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
"""Prepared token-slot address translation on Triton Ascend."""

from collections.abc import Sequence
from dataclasses import dataclass

import torch

_I32_MAX = 2147483647


@dataclass(frozen=True)
class Prepared:
    """Retain tensors themselves so legal set_ updates remain observable."""

    _slots: torch.Tensor
    _mapping: torch.Tensor
    _block_size: int
    _input_shapes: tuple[tuple[int, ...], tuple[int, ...]]
    _device: torch.device

    def run(self) -> torch.Tensor:
        """Allocate independent output and enqueue on the current NPU stream."""
        for tensor, shape in zip((self._slots, self._mapping), self._input_shapes):
            if (tuple(tensor.shape) != shape or tensor.dtype != torch.int32
                    or tensor.device != self._device or not tensor.is_contiguous()):
                raise ValueError("input metadata changed; call prepare again")
        slots, mapping = self._slots, self._mapping
        shape = (slots.numel(),) if mapping.ndim == 1 else (mapping.shape[0], slots.numel())
        with torch.npu.device(self._device):
            output = torch.empty(shape, dtype=torch.int32, device=self._device)
            if slots.numel():
                from kernels import launch

                launch(slots, mapping, output, self._block_size)
        return output


def prepare(inputs: Sequence[torch.Tensor], attributes: dict[str, int]) -> Prepared:
    """Validate once, including the value readback permitted by the contract."""
    if not isinstance(inputs, Sequence) or isinstance(inputs, (str, bytes)):
        raise TypeError("inputs must be a sequence of tensors")
    if len(inputs) != 2:
        raise ValueError("exactly two inputs are required")
    if not isinstance(attributes, dict):
        raise TypeError("attributes must be a dictionary")
    if set(attributes) != {"block_size"}:
        raise ValueError("attributes must contain exactly block_size")
    block_size = attributes["block_size"]
    if type(block_size) is not int or not 1 <= block_size <= _I32_MAX:
        raise ValueError("block_size must be an integer in 1..INT32_MAX")
    for tensor in inputs:
        if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.int32:
            raise TypeError("all inputs must be int32 tensors")
        if tensor.device.type != "npu":
            raise ValueError("NPU tensors are required")
        if tensor.device != inputs[0].device or not tensor.is_contiguous():
            raise ValueError("inputs must be contiguous and on the same NPU")
        if tensor.numel() > _I32_MAX:
            raise ValueError("input element count exceeds INT32_MAX")
    slots, mapping = inputs
    if slots.ndim != 1:
        raise ValueError("slots must have shape [N]")
    if mapping.ndim not in (1, 2) or any(dim < 1 for dim in mapping.shape):
        raise ValueError("mapping must have shape [B] or [L,B], with positive dimensions")
    layers = 1 if mapping.ndim == 1 else mapping.shape[0]
    if layers * slots.numel() > _I32_MAX:
        raise ValueError("output element count exceeds INT32_MAX")
    slot_values = slots.detach().cpu().tolist()
    mapping_values = mapping.detach().cpu().reshape(-1).tolist()
    if any(value >= 0 and value // block_size >= mapping.shape[-1] for value in slot_values):
        raise ValueError("logical slot is outside mapping")
    maximum_page = (_I32_MAX - (block_size - 1)) // block_size
    if any(value < 0 or value > maximum_page for value in mapping_values):
        raise ValueError("mapping must be nonnegative and its entire page must fit int32")
    return Prepared(slots, mapping, block_size,
                    (tuple(slots.shape), tuple(mapping.shape)), slots.device)


def run(inputs: Sequence[torch.Tensor], attributes: dict[str, int]) -> torch.Tensor:
    """Convenience entry; includes validation and preparation."""
    return prepare(inputs, attributes).run()
