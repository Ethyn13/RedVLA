"""Original OpenVLA adapter with optional semantic attention rendering."""

from __future__ import annotations

import gc
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

import numpy as np

from redvla.core.types import Observation
from redvla.models.base import PolicyModel


_OPENVLA_ROOT = Path(
    os.environ["OPENVLA_ROOT"]
)


@dataclass
class _OpenVLAConfig:
    model_family: str = "openvla"
    pretrained_checkpoint: str = ""
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    center_crop: bool = True
    unnorm_key: str = "libero_spatial"


def _cleanup_modules(pre_snapshot: set, project_root: str) -> None:
    while project_root in sys.path:
        sys.path.remove(project_root)
    polluted = {
        key
        for key in sys.modules
        if key == "experiments" or key.startswith("experiments.")
    } - pre_snapshot
    for key in polluted:
        sys.modules.pop(key, None)


def _register_openvla_auto_classes() -> None:
    from transformers import (
        AutoConfig,
        AutoImageProcessor,
        AutoModelForVision2Seq,
        AutoProcessor,
    )
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import (
        OpenVLAForActionPrediction,
    )
    from prismatic.extern.hf.processing_prismatic import (
        PrismaticImageProcessor,
        PrismaticProcessor,
    )

    registrations = (
        lambda: AutoConfig.register("openvla", OpenVLAConfig),
        lambda: AutoImageProcessor.register(
            OpenVLAConfig, PrismaticImageProcessor
        ),
        lambda: AutoProcessor.register(OpenVLAConfig, PrismaticProcessor),
        lambda: AutoModelForVision2Seq.register(
            OpenVLAConfig, OpenVLAForActionPrediction
        ),
    )
    for register in registrations:
        try:
            register()
        except ValueError:
            pass


def _load_eager_openvla(cfg: _OpenVLAConfig):
    import torch
    from transformers import AutoModelForVision2Seq

    _register_openvla_auto_classes()
    model = AutoModelForVision2Seq.from_pretrained(
        cfg.pretrained_checkpoint,
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        load_in_8bit=cfg.load_in_8bit,
        load_in_4bit=cfg.load_in_4bit,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    if not cfg.load_in_8bit and not cfg.load_in_4bit:
        device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        model = model.to(device)

    statistics_path = (
        Path(cfg.pretrained_checkpoint) / "dataset_statistics.json"
    )
    if statistics_path.is_file():
        model.norm_stats = json.loads(
            statistics_path.read_text(encoding="utf-8")
        )
    return model


class OpenVLAModel(PolicyModel):
    """Discrete-token OpenVLA; OFT/diffusion paths are intentionally absent."""

    def __init__(self, **config_overrides):
        unknown = set(config_overrides) - set(_OpenVLAConfig.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown model options: {sorted(unknown)}")
        known = {
            key: value
            for key, value in config_overrides.items()
            if hasattr(_OpenVLAConfig, key)
        }
        self._cfg = _OpenVLAConfig(**known)
        self._model = None
        self._processor = None
        self._fn_get_action = None
        self._fn_normalize_gripper = None
        self._fn_invert_gripper = None
        self._fn_crop_and_resize = None
        self._attention_eager = False
        self._attention_request: dict[str, Any] | None = None
        self._attention_strict = True

    def load(self, model_path: str) -> None:
        project_root = str(_OPENVLA_ROOT.resolve())
        pre_snapshot = {
            key
            for key in sys.modules
            if key == "experiments" or key.startswith("experiments.")
        }
        sys.path.insert(0, project_root)
        try:
            from experiments.robot.openvla_utils import (
                crop_and_resize,
                get_processor,
            )
            from experiments.robot.robot_utils import (
                get_action,
                get_model,
                invert_gripper_action,
                normalize_gripper_action,
            )

            self._cfg.pretrained_checkpoint = model_path
            self._model = get_model(self._cfg)
            self._select_unnorm_key()
            self._processor = get_processor(self._cfg)
            self._fn_get_action = get_action
            self._fn_normalize_gripper = normalize_gripper_action
            self._fn_invert_gripper = invert_gripper_action
            self._fn_crop_and_resize = crop_and_resize
        finally:
            _cleanup_modules(pre_snapshot, project_root)
        self._attention_eager = False
        print(f"[OpenVLAModel] 加载完成: {model_path}")

    def _select_unnorm_key(self) -> None:
        if not hasattr(self._model, "norm_stats"):
            return
        if self._cfg.unnorm_key in self._model.norm_stats:
            return
        available = list(self._model.norm_stats)
        if not available:
            return
        libero_keys = [
            key for key in available if "libero" in key.lower()
        ]
        self._cfg.unnorm_key = (
            libero_keys[0] if libero_keys else available[0]
        )
        print(
            f"[OpenVLAModel] 使用 unnorm_key: {self._cfg.unnorm_key}"
        )

    def set_inference_context(self, **context) -> None:
        request = context.get("task_attention_request")
        self._attention_request = (
            dict(request)
            if isinstance(request, dict) and request.get("enabled", False)
            else None
        )
        self._attention_strict = bool(
            context.get("attention_strict", True)
        )

    def predict(
        self,
        obs: Observation,
        task_description: str,
    ) -> List[np.ndarray]:
        if self._model is None:
            raise RuntimeError("请先调用 load() 加载模型")

        request = self._attention_request
        self._attention_request = None
        if request is None:
            action = self._fn_get_action(
                self._cfg,
                self._model,
                {"full_image": obs.image},
                task_description,
                processor=self._processor,
            )
        else:
            self._ensure_attention_eager_model()
            action = self._predict_with_attention(
                obs, task_description, request
            )

        action = self._fn_normalize_gripper(action, binarize=True)
        action = self._fn_invert_gripper(action)
        return [action]

    def _ensure_attention_eager_model(self) -> None:
        if self._attention_eager:
            return
        import torch

        old_model = self._model
        self._model = None
        if old_model is not None and hasattr(old_model, "cpu"):
            old_model.cpu()
        del old_model
        gc.collect()
        torch.cuda.empty_cache()

        project_root = str(_OPENVLA_ROOT.resolve())
        pre_snapshot = {
            key
            for key in sys.modules
            if key == "experiments" or key.startswith("experiments.")
        }
        sys.path.insert(0, project_root)
        try:
            self._model = _load_eager_openvla(self._cfg)
        finally:
            _cleanup_modules(pre_snapshot, project_root)
        self._select_unnorm_key()
        self._attention_eager = True
        print("[OpenVLAModel] Attention 请求已启用 eager attention")

    def _predict_with_attention(
        self,
        obs: Observation,
        task_description: str,
        request: dict[str, Any],
    ) -> np.ndarray:
        import tensorflow as tf
        import torch
        from PIL import Image

        image = Image.fromarray(obs.image).convert("RGB")
        if self._cfg.center_crop:
            tensor = tf.convert_to_tensor(np.asarray(image))
            original_dtype = tensor.dtype
            tensor = tf.image.convert_image_dtype(tensor, tf.float32)
            tensor = self._fn_crop_and_resize(tensor, 0.9, 1)
            tensor = tf.clip_by_value(tensor, 0, 1)
            tensor = tf.image.convert_image_dtype(
                tensor, original_dtype, saturate=True
            )
            image = Image.fromarray(tensor.numpy()).convert("RGB")

        task = str(task_description).strip().lower()
        if "openvla-v01" in self._cfg.pretrained_checkpoint:
            system = (
                "A chat between a curious user and an artificial "
                "intelligence assistant. The assistant gives helpful, "
                "detailed, and polite answers to the user's questions."
            )
            prompt = (
                f"{system} USER: What action should the robot take to "
                f"{task}? ASSISTANT:"
            )
        else:
            prompt = (
                f"In: What action should the robot take to {task}?\nOut:"
            )

        device = next(self._model.parameters()).device
        inputs = self._processor(prompt, image).to(
            device, dtype=torch.bfloat16
        )
        from redvla.attention.openvla import capture_openvla_action_attention
        return capture_openvla_action_attention(
            model=self._model,
            tokenizer=self._processor.tokenizer,
            model_inputs=inputs,
            image_rgb=np.asarray(image, dtype=np.uint8),
            object_tokens=list(request.get("object_tokens", [])),
            output_root=request["output_root"],
            timestep=int(request["timestep"]),
            query_index=int(request["query_index"]),
            unnorm_key=self._cfg.unnorm_key,
            layer_scope=str(request.get("layer_scope", "all")),
            head_agg=str(request.get("head_agg", "average")),
            topk=int(request.get("topk", 8)),
            overlay_alpha=float(request.get("overlay_alpha", 0.45)),
            blur_radius=float(request.get("blur_radius", 2.0)),
            save_npy=bool(request.get("save_npy", True)),
            strict=self._attention_strict,
        )

    def prepare_for_new_episode(self) -> None:
        self._attention_request = None

    def close(self) -> None:
        import torch

        if self._model is not None and hasattr(self._model, "cpu"):
            self._model.cpu()
        self._model = None
        self._processor = None
        self._fn_get_action = None
        self._fn_normalize_gripper = None
        self._fn_invert_gripper = None
        self._fn_crop_and_resize = None
        self._attention_request = None
        torch.cuda.empty_cache()
        print("[OpenVLAModel] 模型已卸载")

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def config(self) -> _OpenVLAConfig:
        return self._cfg


MODEL_CLASS = OpenVLAModel
