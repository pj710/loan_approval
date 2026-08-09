from __future__ import annotations

from pathlib import Path


def find_project_root(start_path=None):
    """Find the project root by walking upward from a file or directory."""
    if start_path is None:
        start_path = Path.cwd()

    path = Path(start_path).expanduser().resolve()
    if path.is_file():
        path = path.parent

    for candidate in [path, *path.parents]:
        if (candidate / "config.yaml").exists() and (candidate / "notebooks").exists():
            return candidate

        for child in candidate.iterdir():
            if child.is_dir() and (child / "config.yaml").exists() and (child / "notebooks").exists():
                return child

    return path


def resolve_path(path_str, project_root=None):
    """Resolve a path relative to the project root when possible."""
    project_root = Path(project_root or find_project_root())
    candidate = Path(path_str)
    if candidate.is_absolute():
        return candidate
    return (project_root / candidate).resolve()