"""Path helpers shared by dataset packaging scripts."""

from pathlib import Path


def is_within(path: Path, parent: Path) -> bool:
    """Return whether *path* is contained by *parent* without filesystem I/O."""
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True
