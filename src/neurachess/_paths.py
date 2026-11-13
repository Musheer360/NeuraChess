"""Shared filesystem path resolution helpers for both editable/source checkouts and installed packages."""

from __future__ import annotations

import os
from pathlib import Path


def find_repo_root() -> Path:
    """Resolve the NeuraChess repository/runtime root directory containing `assets/` and `weights/`."""
    env_root = os.environ.get("NEURACHESS_ROOT", "").strip()
    if env_root:
        cand = Path(env_root).expanduser().resolve()
        if cand.exists():
            return cand

    pkg_dir = Path(__file__).resolve().parent
    for parent in (pkg_dir, *pkg_dir.parents, Path.cwd(), *Path.cwd().parents):
        if (parent / "pyproject.toml").is_file() or (parent / "assets" / "opening_book.json").is_file():
            return parent

    return pkg_dir.parents[2] if len(pkg_dir.parents) >= 3 else pkg_dir.parent
