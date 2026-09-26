"""Select the external red-libero checkout and isolate its resource configuration."""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import os
from pathlib import Path
import sys
import tempfile

import yaml


def resolve_red_libero_root():
    selected = os.environ.get("RED_LIBERO_ROOT")
    if selected:
        root = Path(selected).expanduser().resolve()
    else:
        spec = importlib.util.find_spec("red_libero") or importlib.util.find_spec("libero")
        root = Path(spec.origin).resolve().parents[1] if spec and spec.origin else None
        if root is None or not any((root / marker).is_file() for marker in ("editor_gui.py", "editor_v2_gui.py")):
            raise FileNotFoundError("Set RED_LIBERO_ROOT to your red-libero checkout and install it in the evaluation environment")
    resource_root = root / "libero/libero"
    for relative in ("envs", "assets", "bddl_files", "init_files"):
        if not (resource_root / relative).is_dir():
            raise FileNotFoundError(f"Invalid red-libero checkout: missing {resource_root / relative}")
    return root


def configure_red_libero():
    """Bind the selected checkout before importing its simulation modules."""
    root = resolve_red_libero_root()
    for name in ("red_libero", "libero", "libero.libero"):
        loaded = sys.modules.get(name)
        filename = getattr(loaded, "__file__", None)
        if filename and not Path(filename).resolve().is_relative_to(root):
            raise ImportError("A different LIBERO checkout is already loaded. Run RedVLA in a separate evaluation process")
    # Put the selected repository first even when it was already on PYTHONPATH.
    if str(root) in sys.path:
        sys.path.remove(str(root))
    sys.path.insert(0, str(root))
    importlib.invalidate_caches()

    resources = root / "libero/libero"
    cache_base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "redvla"
    cache_dir = cache_base / ("red-libero-" + hashlib.sha256(str(root).encode()).hexdigest()[:12])
    config_dir = Path(os.environ.get("RED_LIBERO_CONFIG_PATH") or os.environ.get("LIBERO_CONFIG_PATH") or str(cache_dir)).expanduser().resolve()
    config_file = config_dir / "config.yaml"
    loaded_config = getattr(sys.modules.get("red_libero.paths"), "CONFIG_FILE", None)
    loaded_config = loaded_config or getattr(sys.modules.get("libero.libero"), "config_file", None)
    if loaded_config and Path(loaded_config).resolve() != config_file:
        raise ImportError("red-libero was imported with a different resource configuration. Start a new evaluation process")
    expected = {"benchmark_root": str(resources), "bddl_files": str(resources / "bddl_files"),
                "init_states": str(resources / "init_files"), "assets": str(resources / "assets"),
                "datasets": str(cache_dir / "datasets")}
    if config_file.exists():
        existing = yaml.safe_load(config_file.read_text())
        if not isinstance(existing, dict) or any(key not in existing for key in expected):
            raise ValueError(f"Incomplete red-libero path configuration: {config_file}")
        if Path(existing["benchmark_root"]).resolve() != resources:
            raise ValueError(f"{config_file} points to another LIBERO. Select a separate LIBERO_CONFIG_PATH directory")
    else:
        config_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / "datasets").mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", dir=config_dir, delete=False, encoding="utf-8") as handle:
            yaml.safe_dump(expected, handle, sort_keys=False)
            temporary = handle.name
        os.replace(temporary, config_file)
    os.environ["RED_LIBERO_CONFIG_PATH"] = str(config_dir)
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
    return {"name": "red-libero", "source_root": str(root), "config_file": str(config_file)}


def backend_report():
    try:
        return {"name": "red-libero", "source_root": str(resolve_red_libero_root())}
    except (ValueError, FileNotFoundError, ImportError) as error:
        return {"name": "red-libero", "error": str(error)}
