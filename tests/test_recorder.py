import csv
import json
import numpy as np
from redvla.core.recorder import Recorder
from redvla.core.types import EpisodeConfig, EpisodeResult, TaskConfig, IterationResult


def test_benign_results_are_not_labeled_safe(tmp_path):
    recorder = Recorder(str(tmp_path), save_videos=False)
    task = TaskConfig("test", "", "scene", [], "", "fixture", "", "fixture")
    recorder.on_episode_start(task, EpisodeConfig(evaluation_only=True, save_videos=False))
    iteration = IterationResult(0, False, [], False, np.zeros(3), np.zeros(3), 12)
    result = EpisodeResult("test", 0, 0, "fixture", [iteration], False, str(tmp_path))
    recorder.on_episode_end(result)
    events = json.loads((tmp_path / "events.json").read_text())
    assert events["any_triggered"] is None
    assert events["safety_evaluated"] is False
    with (tmp_path / "results.csv").open() as handle:
        row = next(csv.DictReader(handle))
    assert row["safety_status"] == "NotEvaluated"
    assert row["any_triggered"] == ""
