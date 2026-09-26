"""Native OpenPI WebSocket protocol with bounded connect/inference timeouts.

NumPy MessagePack field names follow openpi-client's public wire format.
No JAX, torch or TensorFlow is imported by this client.
"""
from __future__ import annotations
import inspect
import os
import time
import msgpack
import numpy as np
from PIL import Image
from redvla.models.base import PolicyModel
from redvla.protocol import MAX_BYTES, validate_actions, validate_observation


def _pack(value):
    if isinstance(value, np.ndarray) and value.dtype.kind in "buif":
        return {b"__ndarray__": True, b"data": value.tobytes(),
                b"dtype": value.dtype.str, b"shape": value.shape}
    raise TypeError(f"Unsupported OpenPI message value: {type(value)}")


def _unpack(value):
    if b"__ndarray__" in value:
        dtype = np.dtype(value[b"dtype"])
        if dtype.kind not in "buif":
            raise ValueError("Unsupported OpenPI response dtype")
        return np.frombuffer(value[b"data"], dtype=dtype).reshape(value[b"shape"])
    if b"__npgeneric__" in value:
        dtype = np.dtype(value[b"dtype"])
        if dtype.kind not in "buif":
            raise ValueError("Unsupported OpenPI scalar dtype")
        return dtype.type(value[b"data"])
    return value


def resize_with_pad(image, size=224):
    height, width = image.shape[:2]
    ratio = max(width / size, height / size)
    out_width, out_height = max(1, int(width / ratio)), max(1, int(height / ratio))
    resized = Image.fromarray(image).resize((out_width, out_height), Image.Resampling.BILINEAR)
    result = Image.new("RGB", (size, size))
    result.paste(resized, ((size - out_width) // 2, (size - out_height) // 2))
    return np.asarray(result)


class OpenPIModel(PolicyModel):
    def __init__(self, replan_steps=8, connect_timeout=30.0, inference_timeout=120.0,
                 image_size=224, api_key_env="OPENPI_API_KEY"):
        if int(replan_steps) < 1 or connect_timeout <= 0 or inference_timeout <= 0:
            raise ValueError("replan_steps and timeouts must be positive")
        self.replan_steps = int(replan_steps)
        self.connect_timeout = connect_timeout
        self.inference_timeout = inference_timeout
        self.image_size = image_size
        self.api_key_env = api_key_env
        self.ws = None
        self.context = {}

    def load(self, model_path):
        from websockets.sync.client import connect
        url = model_path if "://" in model_path else "ws://" + model_path
        if not url.startswith(("ws://", "wss://")):
            raise ValueError("OpenPI requires a ws:// or wss:// URL")
        deadline = time.monotonic() + self.connect_timeout
        options = {"compression": None, "max_size": MAX_BYTES, "close_timeout": 2}
        if "proxy" in inspect.signature(connect).parameters:
            options["proxy"] = None
        key = os.environ.get(self.api_key_env)
        if key:
            options["additional_headers"] = {"Authorization": f"Api-Key {key}"}
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"OpenPI connection timed out: {url}")
            try:
                self.ws = connect(url, open_timeout=remaining, **options)
                self.metadata = msgpack.unpackb(self.ws.recv(timeout=max(0.01, deadline - time.monotonic())),
                                               object_hook=_unpack)
                break
            except (OSError, TimeoutError):
                self.close()
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"OpenPI connection timed out: {url}")
                time.sleep(min(0.25, max(0, deadline - time.monotonic())))

    def prepare_for_new_episode(self):
        # Standard OpenPI policies are stateless and have no wire-level reset method.
        self.context = {}

    def set_inference_context(self, **context):
        self.context = context

    def predict(self, obs, task_description):
        validate_observation(obs)
        if self.ws is None:
            raise RuntimeError("OpenPI is not connected")
        context, self.context = self.context, {}
        request = {"observation/image": resize_with_pad(obs.image, self.image_size),
                   "observation/wrist_image": resize_with_pad(obs.wrist_image, self.image_size),
                   "observation/state": np.asarray(obs.state, dtype=np.float32), "prompt": task_description}
        attention = context.get("task_attention_request")
        if attention:
            request["task_attention_request"] = attention
        self.ws.send(msgpack.packb(request, default=_pack))
        response = self.ws.recv(timeout=self.inference_timeout)
        if isinstance(response, str):
            raise RuntimeError(f"OpenPI inference error: {response}")
        result = msgpack.unpackb(response, object_hook=_unpack)
        if attention and context.get("attention_strict") and not result.get("task_attention_status", {}).get("ok"):
            raise RuntimeError("OpenPI server did not confirm attention extraction")
        return list(validate_actions(result["actions"])[:self.replan_steps])

    def close(self):
        if self.ws is not None:
            self.ws.close()
            self.ws = None
