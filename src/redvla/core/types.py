"""
共享数据类型定义。

这里的类型在整个 adversarial 框架中流转，
不依赖任何具体的模型或环境实现。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# 观测 / 动作
# ---------------------------------------------------------------------------

@dataclass
class Observation:
    """单步观测，由 RobotEnv 产生，传给 PolicyModel。"""
    image: np.ndarray          # 主视角 RGB 图像 (H, W, 3)
    wrist_image: np.ndarray    # 腕部摄像头图像 (H, W, 3)
    state: np.ndarray          # 本体感受觉向量（关节角 + 夹爪状态等）


# ---------------------------------------------------------------------------
# 物体信息（由环境后端填充）
# ---------------------------------------------------------------------------

@dataclass
class ObjectInfo:
    """单个物体在仿真器中的 ID 信息，由 RobotEnv.find_object() 返回。"""
    name: str                   # 原始请求名称（可能带索引后缀，如 knife_1[0]）
    mujoco_name: str            # 仿真器中的实际名称
    body_id: int
    joint_id: int
    qpos_adr: int               # qpos 数组起始地址
    qpos_num: int               # 该物体占用的 qpos 数量
    # 可选：用于方向计算的"参考部位"（如刀刃而非刀柄）
    reference_body_id: Optional[int] = None
    reference_mujoco_name: Optional[str] = None


# ---------------------------------------------------------------------------
# 任务配置
# ---------------------------------------------------------------------------

@dataclass
class TaskConfig:
    """描述一个对抗性测试任务（模型无关、环境无关）。"""
    name: str                   # 规则类型标识（如 state-eh、cond-dim）
    description: str            # 给模型看的任务文本（如 "pick up the bowl"）
    scene_dir: str              # 绝对路径，包含 scene.bddl + *.pruned_init
    threat_objects: List[str]   # 威胁物体名称列表（支持多物体联动）
    safety_config_path: str     # BDDL 安全规则文件绝对路径
    model_key: str              # 模型名称键（对应 RunConfig.models）
    model_path: str             # 模型权重绝对路径
    model_type: str             # 模型类型（对应 MODEL_REGISTRY 中的键）


# ---------------------------------------------------------------------------
# Episode 运行参数
# ---------------------------------------------------------------------------

@dataclass
class EpisodeConfig:
    """单个 episode 的运行超参数，可被 task 级别覆盖。"""
    max_iterations: int = 10          # 对抗优化最大迭代次数
    max_steps_per_iteration: int = 400  # 每次迭代最多执行步数
    eps: float = 0.01                 # toward 模式的移动步长（米）
    mode: int = 1                     # 重要点选取模式（1/2，见 optimizer.py）
    direction: str = "toward"         # 移动方向：toward | away
    save_videos: bool = True
    evaluation_only: bool = False       # 原始任务评测：不运行安全监控和对抗位置优化
    seed: int = 42
    episode_idx: int = 0              # 在 pruned_init 列表中的索引
    # 物理约束验证参数
    constraint_threshold: float = 0.05  # 位置漂移容忍阈值（米），超过则拒绝并减半步长
    max_constraint_retries: int = 5     # 最大重试次数（每次重试步长减半）
    # 调试开关
    debug_trajectory: bool = False      # 输出轨迹锚点调试信息（mode=1 时有效）
    # openpi attention 提取配置（None 表示不提取）
    attention_config: Optional[Dict[str, Any]] = field(default=None)
    placement_optimizer: str = "directional"
    placement_options: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 迭代 / Episode 结果
# ---------------------------------------------------------------------------

@dataclass
class IterationResult:
    """单次对抗迭代的结果。"""
    iteration: int
    triggered: bool                   # 安全规则是否触发
    triggered_rules: List[str]        # 触发的规则名称列表
    success: bool                     # 任务是否完成（done=True）
    threat_pos: np.ndarray            # 本次迭代威胁物体的中心位置 (3,)
    gripper_pos: np.ndarray           # 最终夹爪位置 (3,)
    steps: int                        # 本次迭代实际执行步数
    step_events: List[Dict[str, Any]] = field(default_factory=list)
    # 每步的事件记录：[{'step': t, 'triggered': bool, 'events': [...]}]
    had_interaction: bool = False     # 是否与 obj_of_interest 发生过交互
    breakdown_mode: str = ""          # "succ" / "attempt" / "failure"
    constraint_rejections: int = 0    # 本次迭代位置更新时被约束拒绝的次数
    position_drift: float = 0.0       # 约束验证中观测到的最大位置漂移（米）
    guard_stopped: bool = False       # guard 是否在安全规则触发前提前终止本次迭代
    guard_detection_step: int = -1    # guard 首次触发时的 infer 步号（-1=未触发）


@dataclass
class EpisodeResult:
    """完整 episode（多次迭代）的汇总结果。"""
    task_name: str
    episode_idx: int
    seed: int
    model_name: str                   # os.path.basename(model_path)
    iterations: List[IterationResult]
    any_triggered: bool               # 任意迭代中安全规则被触发
    output_dir: str                   # 该 episode 的输出目录
    breakdown_mode: str = ""          # "succ" / "attempt" / "failure"
    original_instruction: str = ""   # 原始任务指令
