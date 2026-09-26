"""
ConstraintValidator：威胁物体位置的物理约束验证器。

每次迭代末尾，对优化器提出的候选位置做"试探"（不计入迭代数）：

  探测流程：
  1. 将候选位置写入 qpos → reset 环境（含 sim.forward()）
  2. 【穿模检测】读取 sim.data.contact，若威胁物体的碰撞箱与任意环境物体
     发生深度穿插（contact.dist < -PENETRATION_DIST_THRESH）→ 拒绝
  3. 执行 warmup_steps 步空动作，等待物理稳定
  4. 【位移检测】读取实际落点，与上一迭代稳定位置的 xy 距离
     > displacement_threshold → 拒绝（步长过大或被弹飞）
  5. 拒绝时取 (prev_pos, candidate) 的 xy 中点，重试（最多 max_constraint_retries 次）

Every proposed placement is checked, including the first update after the initial rollout.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from redvla.core.types import EpisodeConfig, ObjectInfo
from redvla.envs.base import RobotEnv


class ConstraintValidator:

    # 碰撞箱穿插深度阈值（米）：超过此值视为穿模
    _PENETRATION_DIST_THRESH: float = 0.005   # 5 mm

    def __init__(
        self,
        env: RobotEnv,
        threat_infos: List[ObjectInfo],
        config: EpisodeConfig,
        warmup_steps: int = 10,
    ):
        self._env = env
        self._threat_infos = threat_infos
        self._displacement_threshold = config.constraint_threshold
        self._max_retries = config.max_constraint_retries
        self._warmup_steps = warmup_steps
        # 威胁物体的 geom ID 集合（延迟初始化，首次探测时计算并缓存）
        self._threat_geom_ids: Optional[Set[int]] = None

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def validate(
        self,
        base_state: np.ndarray,
        current_threat_pos: np.ndarray,
        candidate_center: np.ndarray,
        initial_relative_positions: Optional[Dict[str, np.ndarray]] = None,
        iteration: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray, int, float]:
        """验证候选位置，穿模或位移过大时缩步重试。

        Returns:
            (accepted_state, accepted_center, probe_count, max_displacement)
        """
        probe_count = 0
        max_displacement = 0.0
        center = candidate_center.copy()

        for attempt in range(self._max_retries + 1):
            cand_state, has_penetration, displacement = self._probe(
                base_state, center, current_threat_pos, initial_relative_positions
            )
            max_displacement = max(max_displacement, displacement)

            ok_penetration = not has_penetration
            ok_displacement = displacement <= self._displacement_threshold

            if ok_penetration and ok_displacement:
                if probe_count > 0:
                    print(
                        f"  [约束] 接受（第{attempt + 1}次探测）"
                        f"  位移={displacement:.4f}m  共缩步{probe_count}次"
                    )
                return cand_state, center, probe_count, max_displacement

            # 拒绝：打印原因，步长减半
            probe_count += 1
            reasons = []
            if not ok_penetration:
                reasons.append("穿模（碰撞箱有穿插）")
            if not ok_displacement:
                reasons.append(
                    f"位移={displacement:.4f}m>阈值={self._displacement_threshold:.4f}m"
                )
            print(f"  [约束] 缩步（第{probe_count}次）：{', '.join(reasons)}")
            center = center.copy()
            center[:2] = (current_threat_pos[:2] + center[:2]) / 2.0

        print(
            f"  [约束] 已达最大缩步次数({self._max_retries})，强制接受"
            f"（最大位移={max_displacement:.4f}m）"
        )
        final_state = self._env.set_object_position_in_state(
            base_state, self._threat_infos, center, initial_relative_positions
        )
        return final_state, center, probe_count, max_displacement

    # ------------------------------------------------------------------
    # 探测
    # ------------------------------------------------------------------

    def _probe(
        self,
        base_state: np.ndarray,
        center: np.ndarray,
        prev_threat_pos: np.ndarray,
        initial_relative_positions: Optional[Dict[str, np.ndarray]],
    ) -> Tuple[np.ndarray, bool, float]:
        """单次探测：放置 → 穿模检测 → warmup → 位移检测。

        Returns:
            (cand_state, has_penetration, displacement)
            - has_penetration: 放置后碰撞箱是否穿插进环境物体
            - displacement:    warmup 稳定后实际位置与上一迭代稳定位置的 xy 距离
        """
        # 1. 写入候选位置并 reset（reset_with_state 内含 sim.forward()）
        cand_state = self._env.set_object_position_in_state(
            base_state, self._threat_infos, center, initial_relative_positions
        )
        self._env.reset_with_state(cand_state)

        # 2. 穿模检测：检查放置后 contact.dist
        has_penetration = self._check_contact_penetration()

        # 3. Warmup：等待物理稳定
        for _ in range(self._warmup_steps):
            self._env.step_noop()

        # 4. 位移检测：稳定后实际位置 vs 上一迭代稳定位置
        actual_pos = self._env.get_object_position(self._threat_infos[0])
        displacement = float(np.linalg.norm(actual_pos[:2] - prev_threat_pos[:2]))

        return cand_state, has_penetration, displacement

    # ------------------------------------------------------------------
    # 穿模检测
    # ------------------------------------------------------------------

    def _check_contact_penetration(self) -> bool:
        """检查威胁物体碰撞箱是否与任意物体发生穿插（contact.dist < 阈值）。

        使用 sim.data.contact 判断，不依赖坐标距离。
        仅当 get_native_sim_env() 可用时生效；否则返回 False（降级跳过）。
        """
        try:
            native_env = self._env.get_native_sim_env()
            sim = native_env.sim
        except Exception:
            return False

        threat_geom_ids = self._get_threat_geom_ids(sim)
        if not threat_geom_ids:
            return False

        ncon = int(sim.data.ncon)
        for i in range(ncon):
            c = sim.data.contact[i]
            if c.geom1 in threat_geom_ids or c.geom2 in threat_geom_ids:
                if c.dist < -self._PENETRATION_DIST_THRESH:
                    return True
        return False

    def _get_threat_geom_ids(self, sim) -> Set[int]:
        """返回威胁物体所有碰撞 geom 的 ID 集合（结果缓存）。"""
        if self._threat_geom_ids is not None:
            return self._threat_geom_ids

        ids: Set[int] = set()
        for info in self._threat_infos:
            for gid in range(sim.model.ngeom):
                if sim.model.geom_bodyid[gid] == info.body_id:
                    ids.add(gid)
        self._threat_geom_ids = ids
        return ids
