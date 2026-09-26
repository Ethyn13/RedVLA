"""
RobotEnv 抽象基类。

定义测试流程所需的最小环境接口。
具体实现（如 LiberoEnv）负责 MuJoCo / LIBERO 的细节。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np

from redvla.core.types import ObjectInfo, Observation


class RobotEnv(ABC):
    """机器人仿真环境抽象接口。

    职责：
    - 场景加载（解析 BDDL、初始状态）
    - 环境重置与步进
    - 物体位置的读取与写入（用于对抗位置修改）
    - 观测提取（图像 + 本体感受觉）
    """

    # ------------------------------------------------------------------
    # 场景管理
    # ------------------------------------------------------------------

    @abstractmethod
    def load_scene(self, scene_dir: str) -> str:
        """加载场景目录，解析 BDDL 和 pruned_init 状态文件。

        Args:
            scene_dir: 包含 *.bddl 和 *.pruned_init 文件的目录路径。

        Returns:
            任务描述字符串（供 PolicyModel 使用）。
        """

    @abstractmethod
    def get_initial_state(self, episode_idx: int) -> np.ndarray:
        """获取指定 episode 的初始 qpos 状态向量。

        Args:
            episode_idx: pruned_init 列表中的索引。

        Returns:
            初始状态向量 (state_dim,)。
        """

    @abstractmethod
    def num_episodes(self) -> int:
        """返回当前场景可用的初始状态数量（即 pruned_init 的个数）。"""

    # ------------------------------------------------------------------
    # 运行控制
    # ------------------------------------------------------------------

    @abstractmethod
    def reset_with_state(self, state: np.ndarray) -> Observation:
        """用给定 qpos 状态重置环境，返回初始观测。

        Args:
            state: 完整的 qpos 状态向量。

        Returns:
            初始时刻的 Observation。
        """

    @abstractmethod
    def step(self, action: np.ndarray) -> Tuple[Observation, float, bool, dict]:
        """执行一步动作。

        Returns:
            (obs, reward, done, info)
        """

    @abstractmethod
    def step_noop(self) -> Tuple[Observation, float, bool, dict]:
        """执行一步空动作（用于物理稳定）。"""

    # ------------------------------------------------------------------
    # 状态查询
    # ------------------------------------------------------------------

    @abstractmethod
    def get_gripper_position(self, obs: Observation) -> np.ndarray:
        """返回夹爪末端的 xyz 世界坐标。

        Returns:
            形如 (3,) 的 ndarray。
        """

    # ------------------------------------------------------------------
    # 物体位置操作（用于对抗位置搜索）
    # ------------------------------------------------------------------

    @abstractmethod
    def find_object(self, name: str) -> ObjectInfo:
        """根据名称查找物体，返回其 MuJoCo 信息。

        支持：
        - 精确名称匹配
        - 模糊匹配（去掉数字后缀等）
        - 部分引用语法：knife_1[0] 表示查找刀体 body_id=0 作为参考点

        Args:
            name: 物体名称，可含索引后缀。

        Returns:
            ObjectInfo 实例。

        Raises:
            ValueError: 找不到对应物体时。
        """

    @abstractmethod
    def get_object_position(self, info: ObjectInfo) -> np.ndarray:
        """获取物体当前的世界坐标 xyz。

        对于多物体联动，返回所有物体的几何中心。

        Args:
            info: find_object() 返回的 ObjectInfo。

        Returns:
            形如 (3,) 的 ndarray。
        """

    @abstractmethod
    def set_object_position_in_state(
        self,
        state: np.ndarray,
        infos: List[ObjectInfo],
        center_pos: np.ndarray,
        initial_relative_positions: Optional[dict] = None,
    ) -> np.ndarray:
        """在 state 向量中更新物体位置，返回修改后的新 state。

        对于多物体联动，以 center_pos 为新中心，
        按 initial_relative_positions 保持各物体的相对偏移不变。

        Args:
            state: 当前完整状态向量。
            infos: 需要移动的物体列表（ObjectInfo）。
            center_pos: 新的中心位置 xyz。
            initial_relative_positions: {name: offset_xyz} 字典，
                如果为 None 则所有物体都移动到 center_pos。

        Returns:
            修改后的状态向量（不修改原 state）。
        """

    @abstractmethod
    def get_current_full_state(self) -> np.ndarray:
        """获取环境当前完整 qpos 状态（用于迭代间传递）。"""

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    @abstractmethod
    def close(self) -> None:
        """释放仿真器资源。"""

    def get_native_sim_env(self):
        """返回底层仿真环境实例（供 SafetyMonitor 等需要直接访问仿真器的组件使用）。

        子类在 load_scene() 之后覆盖此方法或直接返回内部 env 实例。
        默认返回 None。
        """
        return None

    def get_obj_of_interest(self) -> List[str]:
        """返回当前场景 BDDL 中 :obj_of_interest 字段的物体名称列表。

        子类在 load_scene() 解析 BDDL 后填充；默认返回空列表。
        """
        return []
