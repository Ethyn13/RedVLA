"""Versioned, non-pickle transport for canonical LIBERO observations and actions."""
from __future__ import annotations

import base64
import math
import numpy as np
from redvla.core.types import Observation

VERSION = 1
CONVENTION = "libero-rgb-rot180-eef-axisangle-osc7-v1"
MAX_BYTES = 16 * 1024 * 1024


def validate_observation(obs: Observation) -> Observation:
    for name in ("image", "wrist_image"):
        value = np.asarray(getattr(obs, name))
        if value.dtype != np.uint8 or value.ndim != 3 or value.shape[2] != 3:
            raise ValueError(f"{name} must be uint8 RGB [H,W,3]")
        if not all(1 <= n <= 2048 for n in value.shape[:2]):
            raise ValueError(f"Invalid {name} dimensions")
    if np.shape(obs.state) != (8,) or not np.isfinite(obs.state).all():
        raise ValueError("state must contain eight finite values: xyz, axis-angle, gripper qpos")
    return obs


def validate_actions(actions):
    array = np.asarray(actions, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 7 or not 1 <= len(array) <= 1024:
        raise ValueError("Policy must return a nonempty [T,7] action chunk, T <= 1024")
    if not np.isfinite(array).all():
        raise ValueError("Policy returned non-finite actions")
    return array


def encode_array(value):
    array = np.ascontiguousarray(value)
    if array.dtype.kind not in "buif" or array.nbytes > MAX_BYTES:
        raise ValueError("Unsupported array dtype or size")
    return {"dtype": array.dtype.str, "shape": list(array.shape),
            "data": base64.b64encode(array.tobytes()).decode("ascii")}


def decode_array(value):
    dtype = np.dtype(value["dtype"])
    shape = value["shape"]
    if dtype.kind not in "buif" or not isinstance(shape, list) or len(shape) > 4:
        raise ValueError("Unsupported array schema")
    if any(type(n) is not int or n < 0 for n in shape):
        raise ValueError("Invalid array shape")
    expected = math.prod(shape) * dtype.itemsize
    if expected > MAX_BYTES or len(value["data"]) > MAX_BYTES * 4 // 3 + 4:
        raise ValueError("Array exceeds transport limit")
    data = base64.b64decode(value["data"], validate=True)
    if len(data) != expected:
        raise ValueError("Array byte count does not match shape")
    return np.frombuffer(data, dtype=dtype).reshape(shape).copy()


def encode_observation(obs):
    validate_observation(obs)
    return {name: encode_array(getattr(obs, name)) for name in ("image", "wrist_image", "state")}


def decode_observation(data):
    return validate_observation(Observation(**{
        name: decode_array(data[name]) for name in ("image", "wrist_image", "state")
    }))
