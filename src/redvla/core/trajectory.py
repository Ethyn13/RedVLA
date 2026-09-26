"""
adversarial/core/trajectory.py — 轨迹信息提取模块。

职责：
  从机器人执行轨迹的原始序列中提取有语义的锚点，
  供 optimizer.py 计算下一次威胁物体的移动方向。

支持的锚点策略（mode）：
  mode 1 — 夹爪从开到关的所有切换点中，距威胁物体 B 最近的那个
  mode 2 — 整条轨迹中距威胁物体 B 最近的 EEF 位置
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class StepRecord:
    """单步轨迹记录。"""
    position: np.ndarray   # EEF 位置 (3,)
    openness: float        # 夹爪开合度：sum(|gripper_qpos|)，越小表示越闭合


# ---------------------------------------------------------------------------
# 切换点检测
# ---------------------------------------------------------------------------

def find_gripper_close_transitions(
    records: List[StepRecord],
    noise_thresh_ratio: float = 0.05,
) -> List[int]:
    """找出轨迹中所有「夹爪从开到关」的切换帧索引。

    Args:
        records:             轨迹记录列表。
        noise_thresh_ratio:  噪声阈值比例，防止静止时浮点抖动被误判。

    Returns:
        切换帧索引列表（关闭动作发生后的那一帧），可能为空。
    """
    if len(records) < 2:
        return []

    noise_thresh = max(abs(records[0].openness) * noise_thresh_ratio, 1e-4)
    diffs = [records[i + 1].openness - records[i].openness
             for i in range(len(records) - 1)]

    return [i + 1 for i, d in enumerate(diffs) if d < -noise_thresh]


# ---------------------------------------------------------------------------
# 锚点选取
# ---------------------------------------------------------------------------

def select_anchor(
    mode: int,
    records: List[StepRecord],
    current_pos: np.ndarray,
    debug: bool = False,
) -> Optional[np.ndarray]:
    """从轨迹中选取锚点 A。

    Args:
        mode:        锚点策略（1 / 2）。
        records:     本次迭代的轨迹记录列表。
        current_pos: 威胁物体当前位置 B (3,)。
        debug:       是否输出调试信息（仅 mode=1 有效）。

    Returns:
        锚点位置 (3,)；轨迹为空时返回 None。
    """
    if not records:
        return None

    if mode == 1:
        return _anchor_mode1(records, current_pos, debug)
    elif mode == 2:
        return _anchor_nearest(records, current_pos)

    return None


# ---------------------------------------------------------------------------
# 内部实现
# ---------------------------------------------------------------------------

def _anchor_mode1(
    records: List[StepRecord],
    current_pos: np.ndarray,
    debug: bool,
) -> np.ndarray:
    """mode 1：找所有夹爪切换点，取距 B 最近的那个。"""
    transitions = find_gripper_close_transitions(records)

    if transitions:
        ranked = sorted(
            transitions,
            key=lambda i: float(np.linalg.norm(records[i].position - current_pos)),
        )
        best_idx = ranked[0]

        if debug:
            print(f"\n  [Traj/mode1] 夹爪切换点共 {len(transitions)} 个：")
            for rank, idx in enumerate(ranked):
                dist = float(np.linalg.norm(records[idx].position - current_pos))
                marker = "★" if rank == 0 else " "
                print(
                    f"    {marker} #{rank + 1:2d}  "
                    f"step={idx:4d}  "
                    f"pos=[{records[idx].position[0]:6.3f} "
                    f"{records[idx].position[1]:6.3f} "
                    f"{records[idx].position[2]:6.3f}]  "
                    f"dist_to_B={dist:.4f}m"
                )

        return records[best_idx].position

    if debug:
        print("  [Traj/mode1] 轨迹中未检测到夹爪关闭动作，退化为最近轨迹点")
    return _anchor_nearest(records, current_pos)


def _anchor_nearest(
    records: List[StepRecord],
    current_pos: np.ndarray,
) -> np.ndarray:
    """找轨迹中距 B 最近的 EEF 位置（mode 2 / 退化路径共用）。"""
    dists = [np.linalg.norm(r.position - current_pos) for r in records]
    return records[int(np.argmin(dists))].position
