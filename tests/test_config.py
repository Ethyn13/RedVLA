from pathlib import Path
import yaml
import pytest
from redvla.config import load_config


@pytest.fixture
def config_path(tmp_path):
    (tmp_path / "scene").mkdir()
    (tmp_path / "scene/scene.bddl").write_text("(define (problem fixture))")
    (tmp_path / "scene/scene.pruned_init").write_bytes(b"fixture")
    (tmp_path / "rule.bddl").write_text("(define (:safety_rules (grasping knife_1)))")
    raw = {"models": {"p": {"type": "remote", "path": "http://localhost:8001", "options": {"timeout": 3}}},
           "safety_configs": {"test": "rule.bddl"}, "scene_base": ".",
           "defaults": {"seeds": [7, 42], "episodes": [0, 2], "constraint_threshold": 0.025,
                        "max_constraint_retries": 2, "max_steps": 20},
           "tasks": [{"name": "test", "scene": "scene", "threat_objects": ["knife_1"]}]}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def test_config_preserves_options_and_cross_product(config_path, monkeypatch):
    monkeypatch.chdir(config_path.parent.parent)
    output, jobs = load_config(config_path)
    assert len(jobs) == 4
    assert {(j.episode.episode_idx, j.episode.seed) for j in jobs} == {(0,7), (0,42), (2,7), (2,42)}
    assert jobs[0].model["path"] == "http://localhost:8001"
    assert jobs[0].model["options"] == {"timeout": 3}
    assert jobs[0].episode.constraint_threshold == 0.025
    assert jobs[0].episode.max_constraint_retries == 2
    assert jobs[0].episode.max_iterations == 1


@pytest.mark.parametrize("change", ["unknown_model", "unknown_key", "missing_rule", "empty_tasks", "short_horizon", "missing_env", "bad_strategy", "bad_strategy_options"])
def test_fail_instead_of_silently_skipping(config_path, change):
    raw = yaml.safe_load(config_path.read_text())
    if change == "unknown_model": raw["tasks"][0]["model"] = "missing"
    if change == "unknown_key": raw["defaults"]["max_stepz"] = 50
    if change == "missing_rule": raw["safety_configs"] = {}
    if change == "empty_tasks": raw["tasks"] = []
    if change == "short_horizon": raw["defaults"]["max_steps"] = 10
    if change == "missing_env": raw["scene_base"] = "${REDVLA_TEST_UNDEFINED}"
    if change == "bad_strategy": raw["defaults"]["placement_optimizer"] = ""
    if change == "bad_strategy_options": raw["defaults"]["placement_options"] = [1, 2]
    config_path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError):
        load_config(config_path)


def test_benign_has_no_safety_requirement(config_path):
    raw = yaml.safe_load(config_path.read_text())
    raw["safety_configs"] = {}
    raw["tasks"][0].pop("threat_objects")
    config_path.write_text(yaml.safe_dump(raw))
    _, jobs = load_config(config_path, "benign")
    assert jobs[0].episode.evaluation_only


def test_missing_rules_never_become_safe_results(tmp_path):
    pytest.importorskip("bddl")
    from redvla.safety.parser import get_safety_rules_from_config
    with pytest.raises(FileNotFoundError):
        get_safety_rules_from_config(str(tmp_path / "missing.bddl"))
