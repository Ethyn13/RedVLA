"""
OpenVLA-OFT 模型适配器。

实现 PolicyModel 接口，封装 OpenVLA-OFT 项目的模型加载和推理细节。
测试流程（runner.py）通过 PolicyModel.predict() 调用，不感知内部实现。

与 OpenVLA 原版的关键差异：
  - 使用 action_head（L1RegressionActionHead）预测连续动作块
  - 使用 proprio_projector 将机器人本体感受觉融入推理
  - 支持双图像输入（main image + wrist image）
  - 每次预测返回长度为 num_open_loop_steps 的动作块

【sys.path 隔离策略】
  1. load() 中临时将 OpenVLA-OFT 根目录插入 sys.path
  2. 导入所有函数并保存为实例属性
  3. 移除根目录并清理 sys.modules 中被污染的 experiments.* 条目
  4. predict() 使用保存的函数引用，不再重新 import
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

from redvla.core.types import Observation
from redvla.models.base import PolicyModel

# OpenVLA-OFT 项目根目录（可通过环境变量覆盖）
_OPENVLA_OFT_ROOT = Path(os.environ["OPENVLA_OFT_ROOT"])


@dataclass
class _OpenVLAOFTConfig:
    """OpenVLA-OFT 所需的配置字段（内部使用）。"""
    model_family: str = "openvla"
    pretrained_checkpoint: str = ""
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    center_crop: bool = True
    unnorm_key: str = "libero_spatial"
    # OFT 专有字段
    use_film: bool = False
    use_proprio: bool = True             # OFT 默认开启本体感受觉
    use_l1_regression: bool = True       # OFT 默认使用 L1 回归动作头
    use_diffusion: bool = False
    num_diffusion_steps_train: int = 50
    num_diffusion_steps_inference: int = 50
    num_images_in_input: int = 2         # OFT 默认使用主视角 + 腕部视角双图像
    use_minivlm: bool = False
    lora_rank: int = 32                  # OFT LoRA rank
    save_version: str = ""               # 兼容 vla-adapter openvla_utils.py（该文件检查 cfg.save_version）


def _cleanup_modules(pre_snapshot: set, project_root: str) -> None:
    """清理导入过程污染的 sys.path 和 sys.modules。

    只清理 experiments.* 条目（与 LiberoEnv 冲突的来源）。
    保留 prismatic.* 条目——已加载的 PyTorch 模型内部依赖这些类。
    """
    while project_root in sys.path:
        sys.path.remove(project_root)

    polluted = {
        k for k in sys.modules
        if k == "experiments" or k.startswith("experiments.")
    } - pre_snapshot
    for k in polluted:
        sys.modules.pop(k, None)


class OpenVLAOFTModel(PolicyModel):
    """OpenVLA-OFT 模型（连续动作块预测，带 action_head 和 proprio_projector）。

    内部使用 openvla-oft 项目的 experiments/robot/ 下的工具函数，
    对外只暴露 load() / predict() / prepare_for_new_episode() 接口。
    """

    def __init__(self, **config_overrides):
        """
        Args:
            **config_overrides: 覆盖默认 _OpenVLAOFTConfig 中的字段，例如：
                OpenVLAOFTModel(use_proprio=False, num_images_in_input=1)
        """
        unknown = set(config_overrides) - set(_OpenVLAOFTConfig.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown model options: {sorted(unknown)}")
        self._cfg = _OpenVLAOFTConfig(**{
            k: v for k, v in config_overrides.items()
            if hasattr(_OpenVLAOFTConfig, k)
        })
        self._model = None
        self._processor = None
        self._action_head = None
        self._proprio_projector = None
        # 函数引用（load 时保存，predict 时使用，避免重新 import）
        self._fn_get_action = None
        self._fn_normalize_gripper = None
        self._fn_invert_gripper = None

    # ------------------------------------------------------------------
    # PolicyModel 接口实现
    # ------------------------------------------------------------------

    def load(self, model_path: str) -> None:
        """加载 OFT 模型权重，包括 action_head 和 proprio_projector。

        Args:
            model_path: 本地模型目录路径。
        """
        project_root = str(_OPENVLA_OFT_ROOT.resolve())

        # 快照：记录清理前已存在的 experiments.* 模块
        pre_snapshot = {
            k for k in sys.modules
            if k == "experiments" or k.startswith("experiments.")
        }

        # 临时将 OpenVLA-OFT 根目录插入 sys.path
        sys.path.insert(0, project_root)

        try:
            from experiments.robot.openvla_utils import (
                get_action_head,
                get_noisy_action_projector,
                get_processor,
                get_proprio_projector,
            )
            from experiments.robot.robot_utils import (
                get_action,
                get_model,
                invert_gripper_action,
                normalize_gripper_action,
            )

            self._cfg.pretrained_checkpoint = model_path

            # 1. 加载 VLA 主模型
            model = get_model(self._cfg)

            # 2. 修正 unnorm_key（自动选取可用的键）
            if hasattr(model, "norm_stats"):
                if self._cfg.unnorm_key not in model.norm_stats:
                    available = list(model.norm_stats.keys())
                    if available:
                        libero_keys = [k for k in available if "libero" in k.lower()]
                        self._cfg.unnorm_key = libero_keys[0] if libero_keys else available[0]
                        print(f"[OpenVLAOFTModel] 使用 unnorm_key: {self._cfg.unnorm_key}")

            # 3. Processor
            self._processor = get_processor(self._cfg)

            # 4. 加载 action_head（L1 回归头 / 扩散头）
            llm_dim = model.llm_dim
            action_head = get_action_head(self._cfg, llm_dim)
            self._action_head = action_head
            print(f"[OpenVLAOFTModel] action_head 加载完成: {type(action_head).__name__}")

            # 5. 加载 proprio_projector
            if self._cfg.use_proprio:
                # proprio_dim: eef_pos(3) + eef_quat_as_axisangle(3) + gripper_qpos(2) = 8
                proprio_projector = get_proprio_projector(self._cfg, llm_dim, proprio_dim=8)
                self._proprio_projector = proprio_projector
                print("[OpenVLAOFTModel] proprio_projector 加载完成")

            # 6. 加载 noisy_action_projector（仅扩散模式）
            self._noisy_action_projector = None
            if self._cfg.use_diffusion:
                noisy_action_projector = get_noisy_action_projector(self._cfg, llm_dim)
                self._noisy_action_projector = noisy_action_projector
                print("[OpenVLAOFTModel] noisy_action_projector 加载完成")

            self._model = model

            # 7. 保存函数引用
            self._fn_get_action = get_action
            self._fn_normalize_gripper = normalize_gripper_action
            self._fn_invert_gripper = invert_gripper_action

        finally:
            _cleanup_modules(pre_snapshot, project_root)

        print(f"[OpenVLAOFTModel] 加载完成: {model_path}")

    def predict(self, obs: Observation, task_description: str) -> List[np.ndarray]:
        """给定观测，返回动作块（open-loop action chunk）。

        OFT 每次预测 (num_open_loop_steps, 7) 动作块，返回为列表方便
        AdversarialEpisodeRunner 的 action_queue 直接消费。

        Returns:
            长度为 num_open_loop_steps 的动作列表，每个元素形如 (7,)。
        """
        if self._model is None:
            raise RuntimeError("请先调用 load() 加载模型")

        get_action = self._fn_get_action

        # 构建 OFT 期望的 obs dict
        observation = {
            "full_image": obs.image,
        }
        if self._cfg.num_images_in_input > 1 and obs.wrist_image is not None:
            observation["wrist_image"] = obs.wrist_image
        if self._cfg.use_proprio and obs.state is not None:
            observation["state"] = obs.state

        # OFT 的 get_action 返回长度为 num_open_loop_steps 的 list，每个 (7,)
        actions = get_action(
            self._cfg,
            self._model,
            observation,
            task_description,
            processor=self._processor,
            action_head=self._action_head,
            proprio_projector=self._proprio_projector,
            noisy_action_projector=self._noisy_action_projector,
            use_film=self._cfg.use_film,
        )

        # normalize + invert gripper（与 run_libero_eval.py 保持一致）
        normalize_gripper_action = self._fn_normalize_gripper
        invert_gripper_action = self._fn_invert_gripper

        processed = []
        for action in actions:
            action = normalize_gripper_action(action, binarize=True)
            action = invert_gripper_action(action)
            processed.append(action)

        return processed

    def prepare_for_new_episode(self) -> None:
        """OFT 无需 per-episode 清理（action_queue 由 runner 管理）。"""

    def close(self) -> None:
        """释放模型占用的 GPU 显存。"""
        import torch

        for attr in ("_model", "_action_head", "_proprio_projector", "_noisy_action_projector"):
            obj = getattr(self, attr, None)
            if obj is not None and hasattr(obj, "cpu"):
                obj.cpu()
            setattr(self, attr, None)

        self._processor = None
        self._fn_get_action = None
        self._fn_normalize_gripper = None
        self._fn_invert_gripper = None

        torch.cuda.empty_cache()
        print("[OpenVLAOFTModel] 模型已卸载")

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def config(self) -> _OpenVLAOFTConfig:
        return self._cfg


# 显式声明，让 registry 能找到
MODEL_CLASS = OpenVLAOFTModel
