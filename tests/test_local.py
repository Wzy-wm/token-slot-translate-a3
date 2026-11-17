# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0
# Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
"""CPU-only evidence: oracle fixtures, integer model, API rejection and statistics."""

import ast
import hashlib
import json
from pathlib import Path
import random
import sys
import unittest

_SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SOURCE))
sys.path.insert(0, str(_SOURCE / "fixtures"))

import generate_data
import golden
from paired_statistics import summarize
import solution
import torch


def _integer_model(value: int, size: int) -> tuple[int, int]:
    """Scalar model of the proposed kernel arithmetic, independent of divmod."""
    reciprocal = ((1 << 32) + size - 1) // size
    if size & (size - 1) == 0:
        return value >> (size.bit_length() - 1), value & (size - 1)
    quotient = (value * reciprocal) >> 32
    quotient -= int(quotient * size > value)
    return quotient, value - quotient * size


class IntegerModelTests(unittest.TestCase):
    def test_exhaustive_small_range(self) -> None:
        for size in range(1, 130):
            for value in range(2048):
                self.assertEqual(_integer_model(value, size), divmod(value, size))

    def test_random_i32_and_wide_divisors(self) -> None:
        rng = random.Random(20261117)
        sizes = [1, 3, 7, 17, 31, 64, 96, 128, 1073741823, 1073741824, 2147483647]
        sizes += [rng.randrange(1, 2147483648) for _ in range(1000)]
        for size in sizes:
            values = [0, 1, size - 1, size, 2147483646, 2147483647]
            values += [rng.randrange(2147483648) for _ in range(32)]
            for value in values:
                self.assertEqual(_integer_model(value, size), divmod(value, size))

    def test_all_fourteen_fixture_oracles(self) -> None:
        cases = json.loads((_SOURCE / "fixtures/case.json").read_text())["input_cases"]
        self.assertEqual(len(cases), 14)
        for case in cases:
            with self.subTest(case=case["id"]):
                values, attrs = generate_data.generate(case)
                slots, mapping = values
                single = bool(mapping) and isinstance(mapping[0], int)
                rows = [mapping] if single else mapping
                quotient_remainder = [_integer_model(max(slot, 0), attrs["block_size"]) for slot in slots]
                outputs = [[-1 if slot < 0 else row[quotient] * attrs["block_size"] + remainder
                            for slot, (quotient, remainder) in zip(slots, quotient_remainder)] for row in rows]
                actual = outputs[0] if single else outputs
                expected = golden.calculate(values, attrs, single)
                self.assertEqual(actual, expected)
                if "expected" in case:
                    self.assertEqual(expected, case["expected"])
                if "generalization" in case:
                    digest = hashlib.sha256(json.dumps(expected, separators=(",", ":")).encode()).hexdigest()
                    self.assertEqual(digest, case["generalization"]["expected_sha256"])

    def test_reference_source_integrity(self) -> None:
        contract = json.loads((_SOURCE / "fixtures/case.json").read_text())
        for name, expected in contract["source_sha256"]["reference"].items():
            source = golden.REFERENCE_SOURCES[name]
            self.assertEqual(hashlib.sha256(source.encode()).hexdigest(), expected)


class StatisticsTests(unittest.TestCase):
    def test_constant_paired_saving(self) -> None:
        result = summarize([100, 101, 102, 103, 104, 105], [50, 51, 52, 53, 54, 55], 85)
        self.assertEqual(result["saved_mean_ci95_us"], [50, 50])
        self.assertEqual(result["status"], "PASS")

    def test_fixed_target_failure(self) -> None:
        result = summarize([200] * 6, [90] * 6, 85)
        self.assertFalse(result["checks"]["fixed_target"])
        self.assertEqual(result["status"], "FAIL")

    def test_joint_sampling_captures_correlated_rounds(self) -> None:
        result = summarize([10, 100, 1000, 20, 200, 2000], [9, 99, 999, 19, 199, 1999])
        self.assertEqual(result["saved_mean_ci95_us"], [1, 1])

    def test_nonpositive_interval_fails(self) -> None:
        result = summarize([100] * 6, [100] * 6)
        self.assertEqual(result["status"], "FAIL")

    def test_invalid_samples_rejected(self) -> None:
        for samples in ([1] * 5, [1, 1, 1, 1, 1, float("nan")], [0] * 6):
            with self.assertRaises(ValueError):
                summarize([100] * 6, samples)


class ApiBoundaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.inputs = [torch.tensor([0], dtype=torch.int32), torch.tensor([0], dtype=torch.int32)]

    def test_cpu_input_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "NPU"):
            solution.prepare(self.inputs, {"block_size": 128})

    def test_attributes_rejected_before_device_use(self) -> None:
        for value in (True, False, 0, -1, 2147483648, 1.5, "128"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                solution.prepare(self.inputs, {"block_size": value})
        for attrs in ({}, {"block_size": 128, "unknown": 1}):
            with self.assertRaises(ValueError):
                solution.prepare(self.inputs, attrs)

    def test_bad_input_count_and_type(self) -> None:
        with self.assertRaises(ValueError):
            solution.prepare(self.inputs[:1], {"block_size": 128})
        with self.assertRaises(TypeError):
            solution.prepare(None, {"block_size": 128})
        with self.assertRaises(TypeError):
            solution.prepare([1, 2], {"block_size": 128})

    def test_syntax_of_all_component_files(self) -> None:
        for path in _SOURCE.rglob("*.py"):
            ast.parse(path.read_text(), filename=str(path))


if __name__ == "__main__":
    unittest.main()
