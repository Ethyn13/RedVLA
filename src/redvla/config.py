"""Portable YAML configuration with strict validation and explicit episode/seed products."""
from __future__ import annotations
from dataclasses import asdict, dataclass, fields
import os
from pathlib import Path
import re
import yaml
from redvla.core.types import EpisodeConfig, TaskConfig


@dataclass
class Job:
    task: TaskConfig
    episode: EpisodeConfig
    model: dict


def _expand(value):
    if isinstance(value, str):
        def replace(match):
            name = match.group(1)
            if name not in os.environ:
                raise ValueError(f"Required environment variable {name} is not set")
            return os.environ[name]
        return re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", replace, value)
    if isinstance(value, list):
        return [_expand(x) for x in value]
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    return value


def _path(value, base):
    value = str(value)
    if "://" in value:
        return value
    path = Path(value).expanduser()
    return str((base / path).resolve())


def load_config(filename, mode="evaluate", task_filter=None, limit=None, max_steps=None):
    filename = Path(filename).resolve()
    base = filename.parent
    raw = _expand(yaml.safe_load(filename.read_text(encoding="utf-8")))
    if not isinstance(raw, dict):
        raise ValueError("Configuration must be a YAML mapping")
    allowed = {"output_dir", "models", "models_by_suite", "scene_base", "safety_configs", "defaults", "tasks", "tasks_file"}
    if set(raw) - allowed:
        raise ValueError(f"Unknown top-level config keys: {sorted(set(raw) - allowed)}")
    models = raw.get("models", {})
    if not isinstance(models, dict) or not models:
        raise ValueError("At least one model is required")
    for name, model in models.items():
        if not isinstance(model, dict) or not model.get("type"):
            raise ValueError(f"models.{name} requires type")
        if set(model) - {"type", "path", "options", "source_root"}:
            raise ValueError(f"Unknown model fields in {name}")
        if not isinstance(model.get("options", {}), dict):
            raise ValueError(f"models.{name}.options must be a mapping")
        path = str(model.get("path", ""))
        if path.startswith("hf://"):
            model["path"] = path[5:]
        else:
            model["path"] = _path(path, base) if path else ""
        if model.get("source_root"):
            model["source_root"] = _path(model["source_root"], base)
    tasks = raw.get("tasks", [])
    if "tasks_file" in raw:
        if tasks:
            raise ValueError("Use tasks or tasks_file, not both")
        tasks = _expand(yaml.safe_load(Path(_path(raw["tasks_file"], base)).read_text(encoding="utf-8")))
    if not isinstance(tasks, list):
        raise ValueError("tasks must be a list")
    defaults = raw.get("defaults", {})
    ep_fields = {f.name for f in fields(EpisodeConfig)} - {"seed", "episode_idx", "evaluation_only"}
    allowed_task = ep_fields | {"name", "scene", "model", "threat_objects", "safety_config", "seeds",
                                "episodes", "enabled", "max_steps", "attention", "description"}
    if set(defaults) - allowed_task:
        raise ValueError(f"Unknown defaults keys: {sorted(set(defaults) - allowed_task)}")
    jobs = []
    for item in tasks:
        if not isinstance(item, dict) or set(item) - allowed_task:
            raise ValueError(f"Invalid task fields: {item}")
        entry = {**defaults, **item}
        if not entry.get("enabled", True):
            continue
        name, scene = entry.get("name"), entry.get("scene")
        if not name or not scene:
            raise ValueError("Every task needs name and scene")
        if task_filter and task_filter not in name and task_filter not in scene:
            continue
        suite = Path(scene).parts[-2] if len(Path(scene).parts) >= 2 else ""
        model_key = item.get("model", raw.get("models_by_suite", {}).get(suite,
                             defaults.get("model", next(iter(models)))))
        if model_key not in models:
            raise ValueError(f"Unknown model {model_key}")
        model = models[model_key]
        scene_path = Path(_path(scene, Path(_path(raw.get("scene_base", "."), base))))
        bddl = list(scene_path.glob("*.bddl"))
        if len(bddl) != 1 or not any(scene_path.rglob("*.pruned_init")):
            raise ValueError(f"Scene needs exactly one BDDL and at least one .pruned_init: {scene_path}")
        rule_key = entry.get("safety_config", name)
        rule_path = raw.get("safety_configs", {}).get(rule_key)
        if mode != "benign" and not rule_path:
            raise ValueError(f"No safety config for {rule_key}")
        rule_path = _path(rule_path, base) if rule_path else ""
        if rule_path and not Path(rule_path).is_file():
            raise FileNotFoundError(rule_path)
        threats = entry.get("threat_objects", [])
        if isinstance(threats, str):
            threats = [threats]
        if mode != "benign" and not threats:
            raise ValueError(f"Task {name} requires threat_objects")
        seeds, episodes = entry.get("seeds", [0]), entry.get("episodes", [0])
        for label, values in (("seeds", seeds), ("episodes", episodes)):
            if not isinstance(values, list) or not values or any(type(v) is not int or v < 0 for v in values):
                raise ValueError(f"{label} must be a nonempty list of nonnegative integers")
            if len(values) != len(set(values)):
                raise ValueError(f"Duplicate {label}")
        kwargs = {k: entry[k] for k in ep_fields if k in entry}
        if "max_steps" in entry:
            kwargs["max_steps_per_iteration"] = entry["max_steps"]
        if max_steps is not None:
            kwargs["max_steps_per_iteration"] = max_steps
        if mode != "attack":
            kwargs["max_iterations"] = 1
        if "attention" in entry:
            kwargs["attention_config"] = entry["attention"]
        for episode_idx in episodes:
            for seed in seeds:
                ep = EpisodeConfig(**kwargs, seed=seed, episode_idx=episode_idx, evaluation_only=mode == "benign")
                if not isinstance(ep.placement_optimizer, str) or not ep.placement_optimizer.strip():
                    raise ValueError("placement_optimizer must be directional or an importable module:Class")
                if not isinstance(ep.placement_options, dict):
                    raise ValueError("placement_options must be a mapping")
                if ep.max_steps_per_iteration <= 10 or ep.max_iterations < 1:
                    raise ValueError("max_steps must exceed the 10 warmup steps; max_iterations must be positive")
                if ep.direction not in ("toward", "away") or ep.mode not in (1, 2):
                    raise ValueError("direction must be toward/away and mode must be 1/2")
                if ep.eps <= 0 or ep.constraint_threshold <= 0 or ep.max_constraint_retries < 0:
                    raise ValueError("Invalid placement constraint parameters")
                task = TaskConfig(name, entry.get("description", ""), str(scene_path), list(threats),
                                  rule_path, model_key, model["path"], model["type"])
                jobs.append(Job(task, ep, model))
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        jobs = jobs[:limit]
    if not jobs:
        raise ValueError("No jobs matched the configuration / task filter")
    return _path(raw.get("output_dir", "./outputs"), base), jobs


def serialize_jobs(jobs):
    return [asdict(job) for job in jobs]
