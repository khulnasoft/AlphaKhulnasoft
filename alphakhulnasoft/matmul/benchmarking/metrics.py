"""Metric aggregation for benchmark runs.

Aggregates timing samples into a stable, reproducible schema. Pure-python and
CPU-runnable; no CUDA required.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class MetricSummary:
    """Aggregated statistics over a set of timing samples (milliseconds)."""

    count: int
    median_ms: float
    mean_ms: float
    variance_ms: float
    min_ms: float
    max_ms: float

    def as_dict(self) -> dict[str, float]:
        return {
            "count": self.count,
            "median_ms": self.median_ms,
            "mean_ms": self.mean_ms,
            "variance_ms": self.variance_ms,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
        }


def aggregate(samples_ms: Sequence[float]) -> MetricSummary:
    """Aggregate timing samples (in milliseconds) into a summary."""
    if not samples_ms:
        raise ValueError("Cannot aggregate an empty sample list.")
    values = [float(s) for s in samples_ms]
    variance = statistics.pvariance(values) if len(values) > 1 else 0.0
    return MetricSummary(
        count=len(values),
        median_ms=statistics.median(values),
        mean_ms=statistics.fmean(values),
        variance_ms=variance,
        min_ms=min(values),
        max_ms=max(values),
    )


@dataclass(frozen=True)
class BenchmarkResult:
    """Full reproducible record of a single benchmark run."""

    algorithm: str
    dimensions: tuple[int, int, int]
    rank: int
    dtype: str
    device: str
    device_name: str
    cuda_version: str | None
    batch_size: int
    warmups: int
    repetitions: int
    metric: MetricSummary
    baseline_median_ms: float | None = None
    baseline_speedup: float | None = None
    verified_correct: bool = False
    error: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "algorithm": self.algorithm,
            "dimensions": list(self.dimensions),
            "rank": self.rank,
            "dtype": self.dtype,
            "device": self.device,
            "device_name": self.device_name,
            "cuda_version": self.cuda_version,
            "batch_size": self.batch_size,
            "warmups": self.warmups,
            "repetitions": self.repetitions,
            "metric": self.metric.as_dict(),
            "baseline_median_ms": self.baseline_median_ms,
            "baseline_speedup": self.baseline_speedup,
            "verified_correct": self.verified_correct,
            "error": self.error,
        }
