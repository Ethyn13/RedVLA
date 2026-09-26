"""Lazy model loading with explicit upstream checkout and environment boundaries."""
from __future__ import annotations
import importlib
import os
from pathlib import Path
import sys

from redvla.models.base import PolicyModel
from redvla.core.types import Observation
from redvla.protocol import validate_actions, validate_observation

REGISTRY = {
    "remote": "redvla.models.remote:RemoteModel",
    "openpi": "redvla.models.openpi:OpenPIModel",
    "pi0": "redvla.models.openpi:OpenPIModel",
    "pi05": "redvla.models.openpi:OpenPIModel",
    "openvla": "redvla.models.openvla:OpenVLAModel",
    "openvla-oft": "redvla.models.openvla_oft:OpenVLAOFTModel",
    "vla-adapter": "redvla.models.vla_adapter:VLAAdapterModel",
    "vla-adapter-pro": "redvla.models.vla_adapter_pro:VLAAdapterProModel",
}
SOURCE_ENV = {"openvla": "OPENVLA_ROOT", "openvla-oft": "OPENVLA_OFT_ROOT",
              "vla-adapter": "VLA_ADAPTER_ROOT", "vla-adapter-pro": "VLA_ADAPTER_ROOT"}


def list_models():
    return sorted(REGISTRY)


def get_model_class(model_type):
    target = REGISTRY.get(model_type, model_type)
    if ":" not in target:
        raise ValueError(f"Unknown model {model_type!r}; choose {list_models()} or module:Class")
    module, name = target.split(":", 1)
    cls = getattr(importlib.import_module(module), name)
    if not isinstance(cls, type) or not issubclass(cls, PolicyModel):
        raise TypeError(f"{target} must inherit PolicyModel")
    return cls


class CheckedPolicy(PolicyModel):
    def __init__(self, inner, preprocessing=False):
        self.inner = inner
        self.preprocessing = preprocessing

    def load(self, model_path):
        self.inner.load(model_path)

    def prepare_for_new_episode(self):
        self.inner.prepare_for_new_episode()

    def set_inference_context(self, **context):
        self.inner.set_inference_context(**context)

    def predict(self, obs, task_description):
        validate_observation(obs)
        if self.preprocessing:
            # Original OpenVLA / VLA-Adapter training preprocessing, inside the model environment.
            import tensorflow as tf
            def resize(image):
                image = tf.io.decode_image(tf.image.encode_jpeg(image), expand_animations=False, dtype=tf.uint8)
                image = tf.image.resize(image, (224, 224), method="lanczos3", antialias=True)
                return tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8).numpy()
            obs = Observation(resize(obs.image), resize(obs.wrist_image), obs.state)
        return list(validate_actions(self.inner.predict(obs, task_description)))

    def close(self):
        self.inner.close()


def create_model(model_type, path="", options=None, source_root=None):
    source_env = SOURCE_ENV.get(model_type)
    root = source_root or (os.environ.get(source_env) if source_env else None)
    if source_env:
        if not root:
            raise ValueError(f"Set source_root or {source_env} to the upstream model checkout")
        root = str(Path(root).expanduser().resolve())
        if not (Path(root) / "experiments/robot/robot_utils.py").is_file():
            raise FileNotFoundError(f"No experiments/robot/robot_utils.py under {root}")
        for name in ("experiments.robot.robot_utils", "prismatic"):
            loaded = sys.modules.get(name)
            location = getattr(loaded, "__file__", None)
            if location and not Path(location).resolve().is_relative_to(root):
                raise RuntimeError("Model repositories conflict in this interpreter. Use separate redvla serve processes.")
        os.environ[source_env] = root
        sys.path.insert(0, root)
    policy = CheckedPolicy(get_model_class(model_type)(**(options or {})), preprocessing=bool(source_env))
    try:
        policy.load(path)
        policy.metadata = {"type": model_type, "path": path, "source_root": root,
                           "options": options or {}, "server": getattr(policy.inner, "metadata", None)}
        config = getattr(policy.inner, "config", None)
        if config is not None:
            policy.metadata["unnorm_key"] = getattr(config, "unnorm_key", None)
    except BaseException:
        try:
            policy.close()
        except Exception:
            pass
        raise
    return policy
