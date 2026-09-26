"""
Recorder：统一的输出记录器。

将之前散落在 adversarial_placement.py 各处的 CSV 写入、
JSON 保存、视频录制逻辑集中到此模块。

输出目录结构：
  {output_dir}/
  ├── rollouts/
  │   ├── agentview_{iteration:03d}.mp4
  │   └── wrist_{iteration:03d}.mp4
  ├── iterations.csv
  ├── events.json
  └── results.csv
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from redvla.core.types import (
    EpisodeConfig,
    EpisodeResult,
    IterationResult,
    TaskConfig,
)


class Recorder:
    """负责 episode 期间所有持久化输出。

    生命周期：
        recorder = Recorder(output_dir, save_videos=True)
        recorder.on_episode_start(task, config)
        for iteration in ...:
            recorder.on_iteration_start(iteration)
            for step in ...:
                recorder.on_step(step, agentview_frame, wrist_frame)
            recorder.on_iteration_end(iteration_result)
        recorder.on_episode_end(episode_result)
    """

    def __init__(self, output_dir: str, save_videos: bool = True):
        self.output_dir = Path(output_dir)
        self.save_videos = save_videos

        self._rollouts_dir = self.output_dir / "rollouts"
        self._iterations_csv_path = self.output_dir / "iterations.csv"
        self._events_json_path = self.output_dir / "events.json"
        self._results_csv_path = self.output_dir / "results.csv"

        # 当前迭代的帧缓冲
        self._agentview_frames: List[np.ndarray] = []
        self._wrist_frames: List[np.ndarray] = []
        self._current_iteration: int = 0

        # 全量事件收集（events.json）
        self._all_step_events: List[Dict[str, Any]] = []

        # 任务 / episode 元信息（on_episode_start 时填入）
        self._task: Optional[TaskConfig] = None
        self._config: Optional[EpisodeConfig] = None

    # ------------------------------------------------------------------
    # 生命周期回调
    # ------------------------------------------------------------------

    def on_episode_start(self, task: TaskConfig, config: EpisodeConfig) -> None:
        """创建目录、初始化 CSV 表头。"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.save_videos:
            self._rollouts_dir.mkdir(parents=True, exist_ok=True)

        self._task = task
        self._config = config

        # iterations.csv 表头
        with open(self._iterations_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._iteration_csv_fields())
            writer.writeheader()

    def on_iteration_start(self, iteration: int) -> None:
        """清空帧缓冲，准备本次迭代的记录。"""
        self._current_iteration = iteration
        self._agentview_frames = []
        self._wrist_frames = []

    def on_step(
        self,
        step: int,
        agentview_frame: Optional[np.ndarray],
        wrist_frame: Optional[np.ndarray],
    ) -> None:
        """收集单步视频帧（在 runner 的步循环中调用）。"""
        if self.save_videos:
            if agentview_frame is not None:
                self._agentview_frames.append(agentview_frame)
            if wrist_frame is not None:
                self._wrist_frames.append(wrist_frame)

    def on_iteration_end(self, result: IterationResult) -> None:
        """追加 iterations.csv 一行，保存本次迭代视频。"""
        # 写 iterations.csv
        row = self._iteration_result_to_row(result)
        with open(self._iterations_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._iteration_csv_fields())
            writer.writerow(row)

        # 保存视频
        if self.save_videos:
            iter_str = f"{result.iteration:03d}"
            if self._agentview_frames:
                self._save_video(
                    self._agentview_frames,
                    self._rollouts_dir / f"agentview_{iter_str}.mp4",
                )
            if self._wrist_frames:
                self._save_video(
                    self._wrist_frames,
                    self._rollouts_dir / f"wrist_{iter_str}.mp4",
                )

        # 累积事件（用于 events.json）
        self._all_step_events.extend(result.step_events)

    def on_episode_end(self, result: EpisodeResult) -> None:
        """写 events.json 和 results.csv（追加一行）。"""
        self._write_events_json(result)
        self._append_results_csv(result)

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    @staticmethod
    def _iteration_csv_fields() -> List[str]:
        return [
            "iteration",
            "triggered",
            "triggered_rules",
            "success",
            "had_interaction",
            "breakdown_mode",
            "constraint_rejections",
            "position_drift",
            "steps",
            "threat_x",
            "threat_y",
            "threat_z",
            "gripper_x",
            "gripper_y",
            "gripper_z",
            "distance_threat_to_gripper",
            "total_events",
        ]

    def _iteration_result_to_row(self, r: IterationResult) -> Dict[str, Any]:
        dist = float(np.linalg.norm(r.threat_pos - r.gripper_pos))
        total_events = sum(
            len(se.get("events", [])) for se in r.step_events
        )
        return {
            "iteration": r.iteration,
            "triggered": r.triggered,
            "triggered_rules": "; ".join(r.triggered_rules) if r.triggered_rules else "None",
            "success": r.success,
            "had_interaction": r.had_interaction,
            "breakdown_mode": r.breakdown_mode,
            "constraint_rejections": r.constraint_rejections,
            "position_drift": round(r.position_drift, 6),
            "steps": r.steps,
            "threat_x": float(r.threat_pos[0]),
            "threat_y": float(r.threat_pos[1]),
            "threat_z": float(r.threat_pos[2]),
            "gripper_x": float(r.gripper_pos[0]),
            "gripper_y": float(r.gripper_pos[1]),
            "gripper_z": float(r.gripper_pos[2]),
            "distance_threat_to_gripper": dist,
            "total_events": total_events,
        }

    def _write_events_json(self, result: EpisodeResult) -> None:
        iterations_data = []
        for ir in result.iterations:
            iterations_data.append(
                {
                    "iteration": ir.iteration,
                    "triggered": ir.triggered,
                    "triggered_rules": ir.triggered_rules,
                    "success": ir.success,
                    "had_interaction": ir.had_interaction,
                    "breakdown_mode": ir.breakdown_mode,
                    "constraint_rejections": ir.constraint_rejections,
                    "position_drift": round(ir.position_drift, 6),
                    "threat_pos": ir.threat_pos.tolist(),
                    "gripper_pos": ir.gripper_pos.tolist(),
                    "steps": ir.steps,
                    "step_events": ir.step_events,
                }
            )
        payload = {
            "task_name": result.task_name,
            "episode_idx": result.episode_idx,
            "seed": result.seed,
            "model_name": result.model_name,
            "safety_evaluated": not self._config.evaluation_only,
            "any_triggered": None if self._config.evaluation_only else result.any_triggered,
            "breakdown_mode": result.breakdown_mode,
            "original_instruction": result.original_instruction,
            "total_iterations": len(result.iterations),
            "iterations": iterations_data,
        }
        with open(self._events_json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def _append_results_csv(self, result: EpisodeResult) -> None:
        """在 results.csv 中追加一行（共享 CSV，由入口脚本维护表头）。"""
        # 收集所有触发的规则名
        all_rules: set[str] = set()
        total_steps = 0
        for ir in result.iterations:
            all_rules.update(ir.triggered_rules)
            total_steps += ir.steps

        row = {
            "task_name": result.task_name,
            "model_name": result.model_name,
            "episode_idx": result.episode_idx,
            "seed": result.seed,
            "any_triggered": None if self._config.evaluation_only else result.any_triggered,
            "safety_status": "NotEvaluated" if self._config.evaluation_only else ("Unsafe" if result.any_triggered else "Safe"),
            "breakdown_mode": result.breakdown_mode,
            "triggered_rules": "; ".join(sorted(all_rules)) if all_rules else "None",
            "original_instruction": result.original_instruction,
            "total_iterations": len(result.iterations),
            "total_steps": total_steps,
            "total_constraint_rejections": sum(
                ir.constraint_rejections for ir in result.iterations
            ),
            "max_position_drift": round(
                max((ir.position_drift for ir in result.iterations), default=0.0), 6
            ),
            "output_dir": str(self.output_dir),
        }
        fields = list(row.keys())
        write_header = not self._results_csv_path.exists()
        with open(self._results_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    @staticmethod
    def _save_video(frames: List[np.ndarray], path: Path) -> None:
        try:
            import imageio
            writer = imageio.get_writer(str(path), fps=30)
            for frame in frames:
                writer.append_data(frame)
            writer.close()
        except Exception as exc:
            print(f"[Recorder] 视频保存失败 {path}: {exc}")
