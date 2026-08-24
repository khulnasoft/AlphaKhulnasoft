# Tensor Matrix Multiplication Implementation Plan

## 1. Project Analysis

### Current state

This repository is currently an LLM-based code repair and formal-proof system:

- Core orchestration lives in `alphakhulnasoft/alpha_repair.py`.
- Existing benchmark code in `alphakhulnasoft/benchmark.py` measures code-repair flows, not matrix multiplication kernels.
- Proof-specific modules already use a sandbox, evaluator, benchmark, and visualizer pattern.
- `notebooks/AlphaKhulnasoft_Demo.ipynb` is the existing Colab precedent.
- CI runs on CPU-only Ubuntu through `uv`, Ruff, Mypy, and Pytest.
- No tensor representation, matrix-multiplication factorization format, equivalence checker, recombination implementation, CUDA backend, or V100 benchmark exists yet.
- NumPy is only transitive today; PyTorch, CuPy, JAX, and CUDA tooling are not declared dependencies.

### Architectural decision

Implement the tensor work as an isolated package rather than coupling it to the LLM repair pipeline:

```text
alphakhulnasoft/matmul/
    __init__.py
    tensor.py
    reference.py
    algorithms/
        __init__.py
        formats.py
        loader.py
        registry.py
    benchmarking/
        __init__.py
        runner.py
        metrics.py
    nonequivalence/
        __init__.py
        invariants.py
        verifier.py
        four_by_four.py
    recombination/
        __init__.py
        compose.py
        decomposition.py
```

Use `tests/test_matmul_*.py` for CPU-testable behavior and keep GPU tests opt-in. Do not extend the existing LLM-oriented `benchmark.py`, `DataLoader`, or `Evaluator` contracts.

## 2. Mathematical Contract

Before implementing algorithms, document and test one canonical convention for the matrix multiplication tensor:

- Inputs are $A \in F^{m \times k}$ and $B \in F^{k \times n}$.
- The output is $C = AB \in F^{m \times n}$.
- A factorization is a collection of rank-one triples that describe the bilinear map.
- Every factorization records dimensions, scalar field or numeric dtype, rank, factor tensors, algorithm name, and provenance.
- Index ordering must be explicit and stable across serialization, reconstruction, equivalence checks, and GPU execution.
- Exact validation uses integer or rational coefficients where possible; floating-point execution is for performance measurement only.
- “Equivalent” must be defined as an allowed transformation, not inferred from approximate output equality. Initially specify supported basis changes, factor permutation, scaling, and input/output permutations.

Acceptance criteria:

- Invalid dimensions, ranks, coefficient shapes, and fields fail with actionable errors.
- A factorization can be serialized and loaded without changing its canonical representation.
- Reconstruction agrees with ordinary matrix multiplication on basis cases and randomized CPU inputs.

## 3. Algorithms Package

### Implementation

1. Add immutable data structures in `tensor.py` for matrix-multiplication tensors, rank-one factors, and algorithm metadata.
2. Implement exact reconstruction of the represented bilinear map and a conventional reference multiplication in `reference.py`.
3. Define a documented JSON format in `algorithms/formats.py` with versioning, dimensions, field/dtype, rank, factors, and provenance.
4. Implement `algorithms/loader.py` for local paths, package resources, and notebook-friendly URLs or Google Drive downloads. Validate input before returning an algorithm.
5. Implement `algorithms/registry.py` for deterministic discovery and named lookup.
6. Add at least one baseline algorithm and the required small factorizations as checked-in fixtures. Keep fixtures separate from loader logic.
7. Add a notebook section or dedicated Colab notebook showing download/load, metadata inspection, reconstruction, and correctness verification.

### Tests

- JSON round trips preserve values and metadata.
- Loading rejects malformed or incompatible files.
- Registered algorithms have deterministic names and ranks.
- Every fixture passes exact basis tests and randomized CPU correctness tests.
- The notebook can load a fixture from a clean Colab runtime.

## 4. V100 Benchmarking

### Implementation

1. Add an optional GPU dependency group, likely PyTorch with a CUDA-compatible installation path. Keep the default install and CI CPU-only.
2. Implement a dedicated runner in `benchmarking/runner.py` that separates factor construction, execution, and measurement.
3. Provide a naive matrix multiplication baseline and the factorization execution path using the same dimensions, dtype, and batch size.
4. Use CUDA events and explicit synchronization around timed regions. Include warmup iterations before recording samples.
5. Record reproducibility metadata: algorithm, dimensions, rank, dtype, device name, CUDA version, batch size, warmups, repetitions, median, variance, min/max, and baseline speedup.
6. Add a CLI that emits JSON or CSV and fails clearly when CUDA or a V100 is unavailable. Never report CPU timings as V100 results.
7. Add a Colab/V100 usage section with installation, device detection, benchmark invocation, and result interpretation.

### Tests

- CPU unit tests cover metric aggregation, validation, serialization, and unavailable-device behavior without requiring CUDA.
- GPU tests are marked or gated with `RUN_GPU_TESTS=1` and verify one small smoke benchmark on a real CUDA device.
- The runner synchronizes correctly and returns stable schema even when an algorithm fails.
- Benchmark output contains enough environment metadata to reproduce a run.

## 5. Nonequivalence Package

### Implementation

1. Encode the allowed equivalence transformations explicitly in `nonequivalence/invariants.py`.
2. Implement cheap invariants first, such as rank-related data, support patterns, coefficient structure, and dimensions.
3. Implement `verifier.py` to compare candidates under the supported transformation group. Separate “proven nonequivalent,” “equivalent,” and “inconclusive” results; do not turn a bounded search failure into a proof.
4. Add the requested nonequivalent algorithms for the $4 \times 4$ multiplication problem in `four_by_four.py` as immutable, named fixtures.
5. Add a Colab demonstration that checks correctness first and then reports the invariant and transformation evidence for nonequivalence.

### Tests

- Known equivalent algorithms are accepted under each supported transformation.
- Known nonequivalent $4 \times 4$ pairs are rejected with the invariant that separates them.
- Invalid dimensions and malformed algorithms are rejected before comparison.
- Numerical tolerance is never used as the sole proof of algebraic equivalence or nonequivalence.
- Inconclusive bounded searches are reported explicitly.

## 6. Recombination Package

### Implementation

1. Define compatible composition rules for combining smaller matrix multiplication factorizations.
2. Implement `compose.py` to recombine factors while preserving dimension mappings, coefficient field, rank, and provenance.
3. Implement `decomposition.py` for the reverse metadata representation needed to inspect how a larger tensor was assembled.
4. Validate compatibility before composition: shared dimensions, index ordering, field/dtype, and factor format version.
5. Expose operation-count and rank metadata so recombined algorithms can be compared with their component algorithms.
6. Add an example of decomposing a larger multiplication tensor from the checked-in smaller factorizations.

### Tests

- Compatible factorizations compose into the expected dimensions.
- Composed tensors pass exact basis and randomized CPU correctness checks.
- Incompatible dimensions, fields, or index conventions fail early.
- Provenance survives repeated composition and serialization.
- Identity and smallest supported cases behave predictably.

## 7. Integration and Documentation

1. Export only stable tensor APIs from `alphakhulnasoft/matmul/__init__.py`; avoid root-package exports until the API is settled.
2. Add direct CPU dependencies to `pyproject.toml` and keep CUDA dependencies optional.
3. Update `uv.lock` through the project’s normal `uv` workflow.
4. Add the algorithm data directory and document its schema and licensing/provenance.
5. Add a dedicated notebook, or extend the existing notebook only if the resulting Colab remains coherent. It should cover loading, correctness, nonequivalence, recombination, and optional V100 benchmarking.
6. Update `README.md` with CPU setup, optional GPU setup, CLI examples, and the limitations of the equivalence verifier.
7. Keep tensor functionality independent from API keys, LLM providers, and the proof sandbox.

## 8. Recommended Implementation Order

1. Write the tensor contract and fixtures.
2. Implement tensor data structures, exact reconstruction, and the reference multiplier.
3. Implement algorithm serialization, loading, and registry.
4. Add correctness tests and baseline algorithms.
5. Implement recombination and validate composed tensors.
6. Implement nonequivalence invariants and the $4 \times 4$ fixtures.
7. Add the Colab loading and verification workflow.
8. Implement the optional CUDA runner and V100 CLI.
9. Integrate documentation, dependency groups, and CI-safe test markers.

## 9. Validation Gates

Run these after each relevant phase:

```bash
uv run pytest tests/test_matmul_tensor.py tests/test_matmul_algorithms.py tests/test_matmul_recombination.py tests/test_matmul_nonequivalence.py
uv run ruff check .
uv run ruff format --check .
uv run mypy alphakhulnasoft/matmul
uv run pytest tests/
```

On a machine with an NVIDIA V100:

```bash
RUN_GPU_TESTS=1 uv run pytest tests/test_matmul_benchmarking.py
uv run python -m alphakhulnasoft.matmul.benchmarking.runner --device cuda --verify-device v100
```

Before considering the feature complete, run the Colab notebook from a clean runtime and verify that loading, exact correctness, nonequivalence reporting, recombination, and benchmark output all work independently. CPU CI must remain green without CUDA installed.

## 10. Risks and Open Decisions

- The exact Nexus-style factorization format is not yet present; the tensor JSON schema must be agreed before fixtures are added.
- The allowed equivalence transformation group may be computationally expensive. The verifier must preserve an inconclusive result instead of overstating certainty.
- V100 performance depends heavily on dtype, batching, layout, and kernel launch overhead. Reports must preserve these parameters and compare like with like.
- The preferred GPU framework and CUDA wheel source need to be selected for the supported Python version.
- Large factorization fixtures may make the repository heavy; use compressed assets or release-hosted data only after reproducible loading and integrity checks are defined.
- External algorithm data must have clear provenance and compatible licensing before inclusion.