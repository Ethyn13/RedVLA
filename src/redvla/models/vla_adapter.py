"""
VLA-Adapter / OpenVLA 模型适配器。

实现 PolicyModel 接口，封装 VLA-Adapter 项目的模型加载和推理细节。
测试流程（runner.py）通过 PolicyModel.predict() 调用，不感知内部实现。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from redvla.core.types import Observation
from redvla.models.base import PolicyModel




@dataclass
class _VLAConfig:
    """VLA-Adapter 所需的配置字段（内部使用）。"""
    model_family: str = "openvla"
    pretrained_checkpoint: str = ""
    use_l1_regression: bool = True
    use_minivlm: bool = True
    use_film: bool = False
    num_images_in_input: int = 2
    use_proprio: bool = True
    center_crop: bool = True
    num_open_loop_steps: int = 8
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    save_version: str = "vla-adapter"
    use_pro_version: bool = False
    unnorm_key: str = "libero_spatial"


class VLAAdapterModel(PolicyModel):
    """VLA-Adapter 模型（OpenVLA 骨干 + L1 回归动作头 + 本体感受觉投影器）。

    内部使用 experiments/robot/ 下的工具函数，
    对外只暴露 load() / predict() / prepare_for_new_episode() 接口。
    """

    def __init__(self, **config_overrides):
        """
        Args:
            **config_overrides: 覆盖默认 _VLAConfig 中的字段，例如：
                VLAAdapterModel(num_open_loop_steps=4, use_proprio=False)
        """
        unknown = set(config_overrides) - set(_VLAConfig.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown model options: {sorted(unknown)}")
        self._cfg = _VLAConfig(**{
            k: v for k, v in config_overrides.items()
            if hasattr(_VLAConfig, k)
        })
        self._model = None
        self._action_head = None
        self._proprio_projector = None
        self._processor = None

    # ------------------------------------------------------------------
    # PolicyModel 接口实现
    # ------------------------------------------------------------------

    def load(self, model_path: str) -> None:
        """加载模型权重并初始化所有组件。

        Args:
            model_path: 本地模型目录路径。
        """
        from experiments.robot.openvla_utils import (
            get_action_head,
            get_processor,
            get_proprio_projector,
        )
        from experiments.robot.robot_utils import get_model

        self._cfg.pretrained_checkpoint = model_path

        # 1. 加载主干 VLA 模型
        model = get_model(self._cfg)
        model.set_version(self._cfg.save_version)

        # 2. 修正 unnorm_key（自动选取可用的键）
        if hasattr(model, "norm_stats"):
            if self._cfg.unnorm_key not in model.norm_stats:
                available = list(model.norm_stats.keys())
                if available:
                    libero_keys = [k for k in available if "libero" in k.lower()]
                    self._cfg.unnorm_key = libero_keys[0] if libero_keys else available[0]
                    print(f"[VLAAdapterModel] 使用 unnorm_key: {self._cfg.unnorm_key}")

        # 3. 本体感受觉投影器
        self._proprio_projector = None
        if self._cfg.use_proprio:
            self._proprio_projector = get_proprio_projector(
                self._cfg, model.llm_dim, proprio_dim=8
            )

        # 4. L1 回归动作头
        self._action_head = None
        if self._cfg.use_l1_regression:
            self._action_head = get_action_head(self._cfg, model.llm_dim)

        # 5. Processor
        self._processor = None
        if self._cfg.model_family == "openvla":
            self._processor = get_processor(self._cfg)

        self._model = model
        print(f"[VLAAdapterModel] 加载完成: {model_path}")

    def predict(self, obs: Observation, task_description: str) -> List[np.ndarray]:
        """给定观测，返回开环动作序列。

        Returns:
            动作列表，每个元素形如 (7,)，已归一化且夹爪方向已反转。
        """
        if self._model is None:
            raise RuntimeError("请先调用 load() 加载模型")

        from experiments.robot.robot_utils import (
            get_action,
            invert_gripper_action,
            normalize_gripper_action,
        )

        # 将 Observation 转为 get_action 期望的 dict
        observation = {
            "full_image": obs.image,
            "wrist_image": obs.wrist_image,
            "state": obs.state,
        }

        raw_actions = get_action(
            self._cfg,
            self._model,
            observation,
            task_description,
            processor=self._processor,
            action_head=self._action_head,
            proprio_projector=self._proprio_projector,
            noisy_action_projector=None,
            use_film=self._cfg.use_film,
            use_minivlm=self._cfg.use_minivlm,
        )

        # 处理每个动作：归一化夹爪 + 反转夹爪符号
        processed = []
        for action in raw_actions:
            action = normalize_gripper_action(action, binarize=True)
            if self._cfg.model_family == "openvla":
                action = invert_gripper_action(action)
            processed.append(action)

        return processed

    def prepare_for_new_episode(self) -> None:
        """清空任何可能存在的 per-episode 缓存状态。

        当前 VLA-Adapter 无需特殊处理，保留作为扩展点。
        """

    def close(self) -> None:
        """释放模型占用的 GPU 显存。"""
        import torch

        for obj in [self._model, self._action_head, self._proprio_projector]:
            if obj is not None and hasattr(obj, "cpu"):
                obj.cpu()

        self._model = None
        self._action_head = None
        self._proprio_projector = None
        self._processor = None

        torch.cuda.empty_cache()
        print("[VLAAdapterModel] 模型已卸载")

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def config(self) -> _VLAConfig:
        return self._cfg
