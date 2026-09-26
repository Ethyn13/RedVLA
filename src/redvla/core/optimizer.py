"""
adversarial/core/optimizer.py — 对抗位置优化器。

职责：根据锚点 A 计算威胁物体 B 的新位置。
轨迹分析（如何选取 A）委托给 trajectory.py。

  toward (direction_sign=+1) : B 朝 A 移动  (A - B 方向)
  away   (direction_sign=-1) : B 背向 A 移动 (B - A 方向)

工厂函数：
    from redvla.core.optimizer import make_optimizer
    opt = make_optimizer(config)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import importlib
from typing import List, Optional

import numpy as np

from redvla.core.trajectory import StepRecord, select_anchor
from redvla.core.types import EpisodeConfig


# ---------------------------------------------------------------------------
# 抽象基类
# ---------------------------------------------------------------------------

class PlacementOptimizer(ABC):
    """对抗位置优化策略基类。"""

    @abstractmethod
    def compute_next_position(
        self,
        current_pos: np.ndarray,
        trajectory: List[StepRecord],
    ) -> np.ndarray:
        """计算下一次迭代威胁物体的新位置。

        Args:
            current_pos: 威胁物体当前中心位置 B (3,)。
            trajectory:  本次迭代的轨迹记录（StepRecord 列表）。

        Returns:
            新位置 (3,)，z 轴与 current_pos 保持一致。
        """


# ---------------------------------------------------------------------------
# DirectionalOptimizer
# ---------------------------------------------------------------------------

class DirectionalOptimizer(PlacementOptimizer):
    """定向优化器：B 朝向或背向锚点 A 移动 eps 米。

    锚点选取由 trajectory.select_anchor() 负责。
    direction_sign = +1 → toward（靠近）
    direction_sign = -1 → away  （远离）
    """

    def __init__(
        self,
        eps: float,
        mode: int,
        direction_sign: int = 1,
        debug: bool = False,
    ):
        if mode not in (1, 2):
            raise ValueError(f"mode 必须为 1 或 2，当前值：{mode}")
        if direction_sign not in (1, -1):
            raise ValueError(f"direction_sign 必须为 1 或 -1")
        self.eps = eps
        self.mode = mode
        self.direction_sign = direction_sign
        self.debug = debug

    def compute_next_position(
        self,
        current_pos: np.ndarray,
        trajectory: List[StepRecord],
    ) -> np.ndarray:
        anchor = select_anchor(
            mode=self.mode,
            records=trajectory,
            current_pos=current_pos,
            debug=self.debug,
        )
        if anchor is None:
            return current_pos.copy()

        delta = anchor - current_pos
        dist = np.linalg.norm(delta)
        if dist < 1e-6:
            return current_pos.copy()

        unit = (delta / dist) * self.direction_sign
        new_pos = current_pos + self.eps * unit
        new_pos[2] = current_pos[2]   # z 不变
        return new_pos


# ---------------------------------------------------------------------------
# 工厂函数
# ---------------------------------------------------------------------------

def make_optimizer(
    config: EpisodeConfig,
    initial_threat_pos: Optional[np.ndarray] = None,
) -> PlacementOptimizer:
    """Create the built-in strategy or a configured module:Class plugin."""
    name = config.placement_optimizer
    options = config.placement_options
    if name != "directional":
        if ":" not in name:
            raise ValueError("placement_optimizer must be directional or module:Class")
        module_name, class_name = name.rsplit(":", 1)
        strategy_type = getattr(importlib.import_module(module_name), class_name)
        if not isinstance(strategy_type, type) or not issubclass(strategy_type, PlacementOptimizer):
            raise TypeError(f"{name} must inherit PlacementOptimizer")
        strategy = strategy_type(config=config, initial_threat_pos=initial_threat_pos, **options)
        return CheckedPlacementOptimizer(strategy)
    if options:
        raise ValueError("directional uses eps/mode/direction; placement_options must be empty")
    direction = config.direction.lower()
    if direction == "toward":
        direction_sign = 1
    elif direction == "away":
        direction_sign = -1
    else:
        raise ValueError(f"未知 direction '{direction}'，支持：toward、away")

    return DirectionalOptimizer(
        eps=config.eps,
        mode=config.mode,
        direction_sign=direction_sign,
        debug=getattr(config, "debug_trajectory", False),
    )


class CheckedPlacementOptimizer(PlacementOptimizer):
    """Validate plugin proposals before they reach the physical constraint checker."""
    def __init__(self, strategy):
        self.strategy = strategy

    def compute_next_position(self, current_pos, trajectory):
        current = np.asarray(current_pos, dtype=float).copy()
        proposal = np.asarray(self.strategy.compute_next_position(current.copy(), trajectory), dtype=float)
        if proposal.shape != (3,) or not np.isfinite(proposal).all():
            raise ValueError("Placement strategy must return a finite position with shape (3,)")
        if not np.isclose(proposal[2], current[2], rtol=0, atol=1e-8):
            raise ValueError("Placement strategies must preserve Z; this runner searches in the XY plane")
        return proposal.copy()
