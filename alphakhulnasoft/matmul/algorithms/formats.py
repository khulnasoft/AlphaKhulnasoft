"""JSON serialization format for matrix-multiplication factorizations.

The on-disk schema is versioned and explicitly records dimensions, field,
rank, factor tensors, algorithm name, and provenance. Values are stored as
plain nested lists so the format is stable across serialization, loading,
equivalence checks, and GPU execution.
"""

from __future__ import annotations

import json
from typing import Any

from ..tensor import FORMAT_VERSION, Factorization, Field, MatmulSpec, RankOneFactor


def to_dict(factorization: Factorization) -> dict[str, Any]:
    """Serialize a :class:`Factorization` to a plain JSON-compatible dict."""
    factorization.validate()
    return {
        "format_version": factorization.format_version,
        "algorithm": factorization.algorithm_name,
        "field": factorization.field.value,
        "dimensions": {
            "m": factorization.spec.m,
            "k": factorization.spec.k,
            "n": factorization.spec.n,
        },
        "rank": factorization.rank,
        "provenance": dict(factorization.provenance),
        "factors": [
            {"u": list(map(list, f.u)), "v": list(map(list, f.v)), "w": list(map(list, f.w))}
            for f in factorization.factors
        ],
    }


def from_dict(data: dict[str, Any]) -> Factorization:
    """Deserialize a dict produced by :func:`to_dict` (validates strictly)."""
    if not isinstance(data, dict):
        raise ValueError("Factorization payload must be a JSON object.")

    format_version = data.get("format_version")
    if format_version != FORMAT_VERSION:
        raise ValueError(
            f"Unsupported format_version {format_version!r}; expected {FORMAT_VERSION!r}"
        )

    algorithm = data.get("algorithm")
    if not isinstance(algorithm, str) or not algorithm:
        raise ValueError("Missing or invalid 'algorithm' name.")

    field_value = data.get("field")
    try:
        field = Field(field_value)
    except ValueError as exc:
        raise ValueError(f"Unknown field {field_value!r}") from exc

    dims = data.get("dimensions")
    if not isinstance(dims, dict):
        raise ValueError("Missing 'dimensions' object.")
    try:
        spec = MatmulSpec(int(dims["m"]), int(dims["k"]), int(dims["n"]))
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError(f"Invalid dimensions {dims!r}") from exc

    raw_factors = data.get("factors")
    if not isinstance(raw_factors, list) or not raw_factors:
        raise ValueError("Missing or empty 'factors' list.")

    factors: list[RankOneFactor] = []
    for idx, raw in enumerate(raw_factors):
        if not isinstance(raw, dict):
            raise ValueError(f"Factor {idx} must be an object with u/v/w.")
        try:
            factor = RankOneFactor(
                u=tuple(tuple(row) for row in raw["u"]),
                v=tuple(tuple(row) for row in raw["v"]),
                w=tuple(tuple(row) for row in raw["w"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Factor {idx} has invalid u/v/w structure") from exc
        factors.append(factor)

    provenance = data.get("provenance", {})
    if not isinstance(provenance, dict):
        raise ValueError("'provenance' must be an object.")

    factorization = Factorization(
        spec=spec,
        field=field,
        algorithm_name=algorithm,
        factors=tuple(factors),
        provenance=dict(provenance),
        format_version=format_version,
    )
    factorization.validate()
    return factorization


def to_json(factorization: Factorization, **kwargs: Any) -> str:
    """Serialize a factorization to a JSON string."""
    return json.dumps(to_dict(factorization), **kwargs)


def from_json(text: str) -> Factorization:
    """Deserialize a factorization from a JSON string."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {exc}") from exc
    return from_dict(data)
