"""Project-wide filesystem anchor"""

from pathlib import Path


def _find_project_root(start: Path) -> Path:
    """Walk upward from `start` until a directory containing pyproject.toml is
    found, this will always find the project root regardless of where it starts."""
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError("could not locate project root (no pyproject.toml found)")


PROJECT_ROOT = _find_project_root(Path(__file__).resolve())
