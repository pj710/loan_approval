from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency guard
    yaml = None

from .paths import find_project_root


def load_config(config_path: Optional[str | Path] = None) -> Dict[str, Any]:
    """Load the YAML project configuration from disk."""
    path = Path(config_path or find_project_root() / "config.yaml")
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    if yaml is None:
        raise ImportError(
            "PyYAML is required to load configuration files. Install it with `pip install pyyaml`."
        )

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}