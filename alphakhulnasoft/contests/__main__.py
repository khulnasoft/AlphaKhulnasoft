"""Module entry point so the CLI can run as ``python -m alphakhulnasoft.contests``."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
