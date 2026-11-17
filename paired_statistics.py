# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0
# Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
"""Paired bootstrap of complete-entry per-round latencies, in microseconds."""

import math
import random
import statistics
from collections.abc import Sequence


def _quantile(values: list[float], probability: float) -> float:
    position = (len(values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (position - lower) * (values[upper] - values[lower])


def summarize(baseline: Sequence[float], candidate: Sequence[float],
              max_mean_us: float | None = None) -> dict:
    """Use the same sampled round indices for both sides of every pair."""
    if len(baseline) != 6 or len(candidate) != 6:
        raise ValueError("six paired rounds are required")
    if any(not math.isfinite(value) or value <= 0 for value in (*baseline, *candidate)):
        raise ValueError("latencies must be positive and finite")
    differences = [base - cand for base, cand in zip(baseline, candidate)]
    rng = random.Random(20261031)
    samples = sorted(sum(differences[rng.randrange(6)] for _ in range(6)) / 6
                     for _ in range(10000))
    ci = [_quantile(samples, 0.025), _quantile(samples, 0.975)]
    mean = statistics.mean(candidate)
    mean_saving = statistics.mean(differences)
    median_saving = statistics.median(baseline) - statistics.median(candidate)
    gates = {"mean_improves": mean_saving > 0,
             "median_improves": median_saving > 0, "ci_lower_positive": ci[0] > 0}
    if max_mean_us is not None:
        gates["fixed_target"] = mean <= max_mean_us
    return {"baseline_rounds_us": list(baseline), "candidate_rounds_us": list(candidate),
            "baseline_mean_us": statistics.mean(baseline), "candidate_mean_us": mean,
            "saved_mean_us": mean_saving, "saved_median_us": median_saving,
            "saved_mean_ci95_us": ci, "max_mean_us": max_mean_us,
            "bootstrap": {"resamples": 10000, "seed": 20261031,
                          "rng": "Python random.Random", "percentile": "linear"},
            "checks": gates, "status": "PASS" if all(gates.values()) else "FAIL"}
