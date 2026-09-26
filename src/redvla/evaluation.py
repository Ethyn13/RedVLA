"""Run manifest, deterministic job identities, resource cleanup and aggregate results."""
from __future__ import annotations
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import sys

from redvla import __version__
from redvla.config import serialize_jobs


def run_jobs(output_dir, jobs, mode):
    # Must be set before importing LIBERO / MuJoCo. Users may select EGL explicitly.
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
    from redvla.envs.libero import BACKEND_INFO, LiberoEnv
    from redvla.core.runner import AdversarialEpisodeRunner
    from redvla.core.recorder import Recorder
    from redvla.models.factory import create_model
    import csv

    run_dir = Path(output_dir) / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest = {"version": __version__, "mode": mode, "python": sys.version,
                "platform": platform.platform(), "simulator": BACKEND_INFO,
                "jobs": serialize_jobs(jobs), "status": "running"}
    paths = set()
    for job in jobs:
        paths.update(Path(job.task.scene_dir).rglob("*.bddl"))
        paths.update(Path(job.task.scene_dir).rglob("*.pruned_init"))
        if job.task.safety_config_path:
            paths.add(Path(job.task.safety_config_path))
    manifest["input_sha256"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    results = []
    current_key = None
    model = None
    print(f"Run directory: {run_dir}", flush=True)
    try:
        for index, job in enumerate(jobs):
            key = json.dumps(job.model, sort_keys=True)
            if key != current_key:
                if model is not None:
                    model.close()
                    model = None
                model = create_model(job.task.model_type, job.task.model_path,
                                     options=job.model.get("options"), source_root=job.model.get("source_root"))
                manifest.setdefault("loaded_models", []).append(model.metadata)
                current_key = key
            slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", f"{job.task.model_key}-{job.task.name}-{Path(job.task.scene_dir).name}")
            episode_dir = run_dir / f"{index:05d}-{slug}-ep{job.episode.episode_idx}-seed{job.episode.seed}"
            env = LiberoEnv(seed=job.episode.seed, horizon=job.episode.max_steps_per_iteration + 10)
            try:
                recorder = Recorder(str(episode_dir), save_videos=job.episode.save_videos)
                runner = AdversarialEpisodeRunner(env, model, recorder, job.task, job.episode,
                                                 safety_config_path=job.task.safety_config_path or None)
                result = runner.run()
            finally:
                env.close()
            results.append(result)
            with (episode_dir / "results.csv").open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                rows, fieldnames = list(reader), reader.fieldnames
            with (run_dir / "total_results.csv").open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                if index == 0:
                    writer.writeheader()
                writer.writerows(rows)
        manifest["status"] = "complete"
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        manifest["completed_jobs"] = len(results)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        if model is not None:
            model.close()
    count = len(results)
    summary = {"episodes": count, "mode": mode,
               "task_success_rate": sum(any(it.success for it in r.iterations) for r in results) / count,
               "unsafe_episode_rate": None if mode == "benign" else sum(r.any_triggered for r in results) / count,
               "iterations": sum(len(r.iterations) for r in results), "output_dir": str(run_dir)}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary
