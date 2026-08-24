"""Tests for benchmarking metrics and the CPU/GPU runner.

GPU tests are gated behind RUN_GPU_TESTS so CPU-only CI stays green.
"""

import os

import pytest

import alphakhulnasoft.matmul as mm
from alphakhulnasoft.matmul.benchmarking import metrics, runner


def test_metric_aggregation():
    samples = [1.0, 2.0, 3.0, 4.0, 5.0]
    summary = metrics.aggregate(samples)
    assert summary.count == 5
    assert summary.median_ms == 3.0
    assert summary.min_ms == 1.0
    assert summary.max_ms == 5.0
    assert summary.variance_ms > 0


def test_metric_aggregation_empty_raises():
    with pytest.raises(ValueError):
        metrics.aggregate([])


def test_runner_cpu_stable_schema_and_correct():
    result = runner.run_benchmark(
        mm.registry.get("strassen_2x2"),
        dims=(2, 2, 2),
        device="cpu",
        batch_size=4,
        warmups=2,
        repetitions=5,
    )
    d = result.as_dict()
    assert set(d) >= {
        "algorithm",
        "dimensions",
        "rank",
        "device",
        "device_name",
        "metric",
        "verified_correct",
    }
    assert d["device"] == "cpu"
    assert d["verified_correct"] is True
    assert d["error"] is None
    assert d["metric"]["count"] == 5


def test_runner_naive_baseline():
    result = runner.run_benchmark(
        None, dims=(3, 3, 3), device="cpu", batch_size=2, warmups=1, repetitions=3
    )
    assert result.algorithm == "naive"
    assert result.baseline_median_ms is None  # no second baseline when algo is baseline


def test_runner_never_reports_cpu_as_cuda():
    # Requesting CUDA without torch available must fail loudly, with CPU
    # timings never presented as GPU results.
    result = runner.run_benchmark(
        mm.registry.get("strassen_2x2"),
        dims=(2, 2, 2),
        device="cuda",
        batch_size=2,
        warmups=1,
        repetitions=3,
    )
    assert result.device == "cuda"
    assert result.error is not None
    assert result.device_name == "cpu"
    assert result.metric.count == 0


def test_runner_failure_keeps_stable_schema():
    result = runner.run_benchmark(
        mm.registry.get("strassen_2x2"),
        dims=(2, 2, 2),
        device="cuda",
    )
    d = result.as_dict()
    assert d["error"] is not None
    assert d["metric"]["count"] == 0
    assert "baseline_speedup" in d


@pytest.mark.skipif(os.environ.get("RUN_GPU_TESTS") != "1", reason="requires CUDA GPU")
def test_gpu_smoke_benchmark():
    result = runner.run_benchmark(
        mm.registry.get("strassen_2x2"),
        dims=(2, 2, 2),
        device="cuda",
        batch_size=8,
        warmups=3,
        repetitions=10,
    )
    assert result.error is None
    assert result.device_name != "cpu"
    assert result.metric.count == 10
