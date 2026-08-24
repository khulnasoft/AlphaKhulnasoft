"""Loading factorizations from local files, package resources, and URLs."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

from ..tensor import Factorization
from .formats import from_json


def load_local(path: str | Path) -> Factorization:
    """Load and validate a factorization from a local JSON file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Algorithm file not found: {path}")
    text = path.read_text(encoding="utf-8")
    factorization = from_json(text)
    factorization.validate()
    return factorization


def load_resource(package_path: str, resource_name: str) -> Factorization:
    """Load a factorization bundled as a package resource (Colab-friendly)."""
    try:
        import importlib.resources as resources

        ref = resources.files(package_path).joinpath(resource_name)
        text = ref.read_text(encoding="utf-8")
    except Exception as exc:  # pragma: no cover - depends on env
        raise ValueError(
            f"Could not load package resource {package_path}/{resource_name}: {exc}"
        ) from exc
    factorization = from_json(text)
    factorization.validate()
    return factorization


def load_url(url: str, timeout: float = 30.0) -> Factorization:
    """Download and validate a factorization JSON from a URL (notebook use)."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
            text = resp.read().decode("utf-8")
    except Exception as exc:
        raise ValueError(f"Failed to download algorithm from {url}: {exc}") from exc
    factorization = from_json(text)
    factorization.validate()
    return factorization


def save_local(factorization: Factorization, path: str | Path) -> None:
    """Serialize a factorization to a local JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_json_pretty(factorization), encoding="utf-8")


def to_json_pretty(factorization: Factorization) -> str:
    """Serialize with stable, human-readable indentation."""
    from .formats import to_dict

    return json.dumps(to_dict(factorization), indent=2, sort_keys=False)
