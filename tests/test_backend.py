import sys
from types import ModuleType
from pathlib import Path

import pytest
import yaml

from redvla.envs.backend import configure_red_libero, resolve_red_libero_root


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    root = tmp_path / "red-libero"
    for name in ("envs", "assets", "bddl_files", "init_files"):
        (root / "libero/libero" / name).mkdir(parents=True)
    monkeypatch.setenv("RED_LIBERO_ROOT", str(root))
    monkeypatch.delenv("RED_LIBERO_CONFIG_PATH", raising=False)
    monkeypatch.setenv("LIBERO_CONFIG_PATH", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.syspath_prepend(str(root))
    return root


def test_external_checkout_and_noninteractive_resource_config(checkout):
    report = configure_red_libero()
    config = yaml.safe_load(Path(report["config_file"]).read_text())
    assert report["source_root"] == str(checkout)
    assert config["assets"] == str(checkout / "libero/libero/assets")
    assert config["benchmark_root"] == str(checkout / "libero/libero")
    assert Path(config["datasets"]).is_dir()
    assert configure_red_libero() == report


def test_reject_another_resource_tree(checkout):
    report = configure_red_libero()
    path = Path(report["config_file"])
    config = yaml.safe_load(path.read_text())
    config["benchmark_root"] = str(checkout / "unrelated")
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="another LIBERO"):
        configure_red_libero()


def test_reject_already_loaded_other_checkout(checkout, monkeypatch):
    module = ModuleType("libero")
    module.__file__ = str(checkout.parent / "another/libero/__init__.py")
    monkeypatch.setitem(sys.modules, "libero", module)
    with pytest.raises(ImportError, match="already loaded"):
        configure_red_libero()


def test_reject_incomplete_checkout(tmp_path, monkeypatch):
    monkeypatch.setenv("RED_LIBERO_ROOT", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="Invalid red-libero"):
        resolve_red_libero_root()


def test_reject_loaded_resource_config_change(checkout, monkeypatch):
    module = ModuleType("libero.libero")
    module.__file__ = str(checkout / "libero/libero/__init__.py")
    module.config_file = str(checkout.parent / "previous-config/config.yaml")
    monkeypatch.setitem(sys.modules, "libero.libero", module)
    with pytest.raises(ImportError, match="different resource configuration"):
        configure_red_libero()


def test_discover_checkout_with_renamed_editor(tmp_path, monkeypatch):
    import importlib.util
    from types import SimpleNamespace
    root = tmp_path / "red-libero"
    for name in ("envs", "assets", "bddl_files", "init_files"):
        (root / "libero/libero" / name).mkdir(parents=True)
    (root / "editor_gui.py").touch()
    monkeypatch.delenv("RED_LIBERO_ROOT", raising=False)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: SimpleNamespace(origin=str(root / "libero/__init__.py")))
    assert resolve_red_libero_root() == root


def test_prefer_project_configuration_variable(checkout, monkeypatch):
    selected = checkout.parent / "red-libero-config"
    monkeypatch.setenv("RED_LIBERO_CONFIG_PATH", str(selected))
    report = configure_red_libero()
    assert report["config_file"] == str(selected / "config.yaml")


def test_reject_already_loaded_public_api_from_other_checkout(checkout, monkeypatch):
    module = ModuleType("red_libero")
    module.__file__ = str(checkout.parent / "another/red_libero/__init__.py")
    monkeypatch.setitem(sys.modules, "red_libero", module)
    with pytest.raises(ImportError, match="already loaded"):
        configure_red_libero()


def test_reject_public_api_configuration_change(checkout, monkeypatch):
    module = ModuleType("red_libero.paths")
    module.CONFIG_FILE = checkout.parent / "previous/config.yaml"
    monkeypatch.setitem(sys.modules, "red_libero.paths", module)
    with pytest.raises(ImportError, match="different resource configuration"):
        configure_red_libero()
