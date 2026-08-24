"""Benchmark runner: CPU (numpy) and optional CUDA (PyTorch) execution.

Design rules enforced here:
* The default install and CI are CPU-only; CUDA is an optional dependency.
* If ``--device cuda`` is requested but CUDA is unavailable, the runner fails
  loudly and NEVER falls back to reporting CPU timings as GPU results.
* ``--verify-device v100`` additionally fails unless the device name reports a
  V100, so a non-V100 GPU cannot masquerade as one.
* The reported schema is stable even when the run fails (an ``error`` field is
  set instead of dropping fields).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from typing import Any

from ..algorithms.registry import get
from ..tensor import Factorization, MatmulSpec
from .metrics import BenchmarkResult, MetricSummary, aggregate


def _numeric_reconstruct(factors: Factorization, a: Any, b: Any) -> Any:
    """Float (numpy) reconstruction of a factorization for performance runs."""
    import numpy as np

    m, k, n = factors.spec.dims
    u = np.array([np.array(f.u, dtype=np.float64) for f in factors.factors])  # (R,m,k)
    v = np.array([np.array(f.v, dtype=np.float64) for f in factors.factors])  # (R,k,n)
    w = np.array([np.array(f.w, dtype=np.float64) for f in factors.factors])  # (R,m,n)
    su = np.einsum("bik,tik->bt", a, u)
    sv = np.einsum("bkn,tkn->bt", b, v)
    prod = su * sv
    return np.einsum("bt,tij->bij", prod, w)


def _naive(a: Any, b: Any) -> Any:
    import numpy as np

    return np.einsum("bik,bkn->bin", a, b)


def _random_batch(spec: MatmulSpec, batch: int, device: str) -> tuple[Any, Any]:
    import numpy as np

    rng = np.random.default_rng(0)
    a = rng.standard_normal((batch, spec.m, spec.k)).astype(np.float64)
    b = rng.standard_normal((batch, spec.k, spec.n)).astype(np.float64)
    if device == "cuda":
        import torch

        return torch.from_numpy(a).cuda(), torch.from_numpy(b).cuda()
    return a, b


def run_benchmark(
    algorithm: Factorization | None,
    dims: tuple[int, int, int],
    device: str = "cpu",
    batch_size: int = 8,
    warmups: int = 3,
    repetitions: int = 10,
    verify_device: str | None = None,
    dtype: str = "float64",
) -> BenchmarkResult:
    """Run a benchmark, returning a stable :class:`BenchmarkResult`.

    If ``algorithm`` is ``None`` a naive matmul baseline is measured.
    """
    spec = MatmulSpec(*dims)
    algo_name = algorithm.algorithm_name if algorithm is not None else "naive"
    rank = algorithm.rank if algorithm is not None else spec.m * spec.k * spec.n

    device_name = "cpu"
    cuda_version = None
    error: str | None = None

    try:
        if device == "cuda":
            device_name, cuda_version = _require_cuda(verify_device)
            samples, baseline_samples, verified = _run_cuda(
                algorithm, spec, batch_size, warmups, repetitions, dtype
            )
        else:
            samples, baseline_samples, verified = _run_cpu(
                algorithm, spec, batch_size, warmups, repetitions
            )
            device_name = "cpu"
    except Exception as exc:  # capture failure in stable schema
        error = str(exc)
        empty = MetricSummary(0, 0.0, 0.0, 0.0, 0.0, 0.0)
        return BenchmarkResult(
            algorithm=algo_name,
            dimensions=dims,
            rank=rank,
            dtype=dtype,
            device=device,
            device_name=device_name,
            cuda_version=cuda_version,
            batch_size=batch_size,
            warmups=warmups,
            repetitions=repetitions,
            metric=empty,
            verified_correct=False,
            error=error,
        )

    metric = aggregate(samples)
    baseline_median = aggregate(baseline_samples).median_ms if baseline_samples else None
    speedup = (baseline_median / metric.median_ms) if baseline_median else None
    return BenchmarkResult(
        algorithm=algo_name,
        dimensions=dims,
        rank=rank,
        dtype=dtype,
        device=device,
        device_name=device_name,
        cuda_version=cuda_version,
        batch_size=batch_size,
        warmups=warmups,
        repetitions=repetitions,
        metric=metric,
        baseline_median_ms=baseline_median,
        baseline_speedup=speedup,
        verified_correct=verified,
    )


def _require_cuda(verify_device: str | None) -> tuple[str, str]:
    """Ensure CUDA is usable; return (device_name, cuda_version)."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "CUDA benchmark requested but PyTorch is not installed. "
            "Install the optional 'gpu' dependency group; CPU results are "
            "never reported as V100 results."
        ) from exc
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA benchmark requested but no CUDA device is available. "
            "Refusing to report CPU timings as GPU results."
        )
    name = torch.cuda.get_device_name(0)
    version = torch.version.cuda or "unknown"
    if verify_device is not None and verify_device.lower() not in name.lower():
        raise RuntimeError(
            f"Device verification failed: expected '{verify_device}' but "
            f"CUDA device is '{name}'. Refusing to benchmark."
        )
    return name, version


def _run_cpu(
    algorithm: Factorization | None,
    spec: MatmulSpec,
    batch: int,
    warmups: int,
    reps: int,
) -> tuple[list[float], list[float], bool]:
    import numpy as np

    a, b = _random_batch(spec, batch, "cpu")
    algo_fn = _numeric_reconstruct if algorithm is not None else _naive
    algo_arg = algorithm if algorithm is not None else None

    def run_once(fn: Any, arg: Any) -> None:
        if arg is not None:
            fn(arg, a, b)
        else:
            fn(a, b)

    for _ in range(warmups):
        run_once(algo_fn, algo_arg)
    algo_samples: list[float] = []
    for _ in range(reps):
        t0 = time.perf_counter()
        run_once(algo_fn, algo_arg)
        algo_samples.append((time.perf_counter() - t0) * 1000.0)

    base_a, base_b = a, b
    base_samples: list[float] = []
    if algorithm is not None:
        for _ in range(warmups):
            _naive(base_a, base_b)
        for _ in range(reps):
            t0 = time.perf_counter()
            _naive(base_a, base_b)
            base_samples.append((time.perf_counter() - t0) * 1000.0)

    verified = True
    if algorithm is not None:
        c = _numeric_reconstruct(algorithm, a, b)
        ref = _naive(a, b)
        verified = bool(np.allclose(c, ref, atol=1e-8))
    return algo_samples, base_samples, verified


def _run_cuda(
    algorithm: Factorization | None,
    spec: MatmulSpec,
    batch: int,
    warmups: int,
    reps: int,
    dtype: str,
) -> tuple[list[float], list[float], bool]:
    import torch

    a, b = _random_batch(spec, batch, "cuda")
    algo_fn = _numeric_reconstruct if algorithm is not None else _naive
    algo_arg = algorithm if algorithm is not None else None

    def run_once(fn: Any, arg: Any) -> None:
        if arg is not None:
            fn(arg, a, b)
        else:
            fn(a, b)

    for _ in range(warmups):
        run_once(algo_fn, algo_arg)
    torch.cuda.synchronize()

    algo_samples: list[float] = []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        run_once(algo_fn, algo_arg)
        end.record()
        torch.cuda.synchronize()
        algo_samples.append(start.elapsed_time(end))

    base_samples: list[float] = []
    if algorithm is not None:
        for _ in range(warmups):
            _naive(a, b)
        torch.cuda.synchronize()
        for _ in range(reps):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _naive(a, b)
            end.record()
            torch.cuda.synchronize()
            base_samples.append(start.elapsed_time(end))

    verified = True
    if algorithm is not None:
        c = _numeric_reconstruct(algorithm, a, b)
        ref = _naive(a, b)
        verified = bool(torch.allclose(c, ref, atol=1e-4))
    return algo_samples, base_samples, verified


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m alphakhulnasoft.matmul.benchmarking.runner",
        description="Benchmark matrix-multiplication algorithms (CPU or V100 CUDA).",
    )
    parser.add_argument(
        "--algorithm", default=None, help="Registered algorithm name (default: naive baseline)."
    )
    parser.add_argument("--dims", type=int, nargs=3, default=(2, 2, 2), metavar=("M", "K", "N"))
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument(
        "--verify-device", default=None, help="e.g. 'v100' to restrict to that GPU."
    )
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--dtype", default="float64")
    parser.add_argument("--format", choices=["json", "csv"], default="json")
    parser.add_argument("--output", default=None, help="Write result to this path.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_cli().parse_args(argv)
    algorithm = get(args.algorithm) if args.algorithm else None
    try:
        result = run_benchmark(
            algorithm=algorithm,
            dims=tuple(args.dims),
            device=args.device,
            batch_size=args.batch,
            warmups=args.warmups,
            repetitions=args.reps,
            verify_device=args.verify_device,
            dtype=args.dtype,
        )
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, indent=2), file=sys.stderr)
        return 2

    payload = result.as_dict()
    if args.format == "csv":
        import csv
        import io

        buf = io.StringIO()
        writer = csv.writer(buf)
        flat = {k: (v if not isinstance(v, dict) else json.dumps(v)) for k, v in payload.items()}
        writer.writerow(flat.keys())
        writer.writerow(flat.values())
        text = buf.getvalue()
    else:
        text = json.dumps(payload, indent=2)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(text)
    else:
        print(text)

    if result.error is not None:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
