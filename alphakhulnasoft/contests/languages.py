"""Language registry for the multi-language execution harness.

Each language declares how to compile and run a solution, plus default
resource caps. Interpreted languages have no compile step.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LanguageSpec:
    """How to build and execute a solution in a given language."""

    name: str
    ext: str
    run_cmd: list[str]  # contains "{executable}" placeholder
    compile_cmd: list[str] | None = None  # contains {source}, {output}
    default_timeout_s: int = 5
    default_memory_mb: int = 512
    # For Java the compiled entry point differs from the file name.
    java_main_class: str | None = None
    source_filename: str | None = None  # override (e.g. "Main.java" for Java)


_REGISTRY: dict[str, LanguageSpec] = {
    "py": LanguageSpec(
        name="py",
        ext=".py",
        run_cmd=["python3", "{executable}"],
        default_timeout_s=5,
        default_memory_mb=512,
    ),
    "cpp": LanguageSpec(
        name="cpp",
        ext=".cpp",
        compile_cmd=["g++", "-O2", "-std=c++17", "-o", "{output}", "{source}"],
        run_cmd=["{executable}"],
        default_timeout_s=5,
        default_memory_mb=512,
    ),
    "java": LanguageSpec(
        name="java",
        ext=".java",
        compile_cmd=["javac", "{source}"],
        run_cmd=["java", "{executable}"],
        default_timeout_s=8,
        default_memory_mb=768,
        java_main_class="Main",
        source_filename="Main.java",
    ),
}


def register_language(spec: LanguageSpec) -> None:
    """Register or override a language specification."""
    _REGISTRY[spec.name] = spec


def get_language_spec(name: str) -> LanguageSpec:
    """Return the spec for a language, raising ``KeyError`` if unsupported."""
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"Unsupported language {name!r}; known: {sorted(_REGISTRY)}") from None


def known_languages() -> list[str]:
    return sorted(_REGISTRY)
