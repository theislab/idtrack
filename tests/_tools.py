"""Import a maintainer's tool of ``tools/`` (a script, not a module of a package) for its tests."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

TOOLS = Path(__file__).resolve().parents[1] / "tools"


def load_tool(name: str) -> ModuleType:
    """Import ``tools/<name>.py`` once and return it."""
    module_name = f"idtrack_tool_{name}"
    if module_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(module_name, TOOLS / f"{name}.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return sys.modules[module_name]
