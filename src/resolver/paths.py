"""Project-wide filesystem anchor, shared across every subpackage."""

from pathlib import Path


def _find_project_root(start: Path) -> Path:
    """Walk upward from `start` until a directory containing pyproject.toml is
    found, so this stays correct regardless of how deep the caller is nested."""
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError("could not locate project root (no pyproject.toml found)")


PROJECT_ROOT = _find_project_root(Path(__file__).resolve())
