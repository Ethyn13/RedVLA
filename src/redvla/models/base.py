"""
PolicyModel 抽象基类。

测试流程（runner.py）只依赖此接口，与具体 VLA 实现完全解耦。
添加新模型：继承 PolicyModel，在 registry.py 中注册即可。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np

from redvla.core.types import Observation


class PolicyModel(ABC):
    """策略模型抽象接口。

    子类需实现 load() 和 predict()，其余方法有默认的空实现。
    """

    @abstractmethod
    def load(self, model_path: str) -> None:
        """加载模型权重，初始化所有推理组件（processor、action_head 等）。

        Args:
            model_path: 本地模型目录或 HuggingFace Hub ID。
        """

    @abstractmethod
    def predict(self, obs: Observation, task_description: str) -> List[np.ndarray]:
        """给定当前观测和任务描述，返回开环动作序列。

        Args:
            obs: 当前时刻的观测（图像 + 本体感受觉）。
            task_description: 自然语言任务描述。

        Returns:
            动作序列，每个元素为形如 (action_dim,) 的 ndarray。
            序列长度由模型的 num_open_loop_steps 决定。
        """

    def prepare_for_new_episode(self) -> None:
        """每个 episode 开始前调用，可用于清空 KV 缓存等状态。

        默认为空操作，子类按需覆盖。
        """

    def set_inference_context(self, **context) -> None:
        """设置下一次 predict 使用的可选推理上下文。

        默认不做任何事；需要透传服务端扩展参数的适配器可覆盖此方法。
        """

    def close(self) -> None:
        """释放 GPU 显存和其他资源。默认为空操作。"""
