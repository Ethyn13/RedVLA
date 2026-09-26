"""LIBERO raw observation conversion without any model repository imports."""
import numpy as np
from redvla.core.types import Observation
from redvla.protocol import validate_observation


def from_libero(raw):
    quat = np.asarray(raw["robot0_eef_quat"], dtype=np.float64).copy()
    w = np.clip(quat[3], -1, 1)
    denominator = np.sqrt(max(0, 1 - w * w))
    axis_angle = np.zeros(3) if denominator < 1e-8 else quat[:3] * (2 * np.arccos(w) / denominator)
    state = np.concatenate([raw["robot0_eef_pos"], axis_angle, raw["robot0_gripper_qpos"]]).astype(np.float32)
    return validate_observation(Observation(
        image=np.ascontiguousarray(raw["agentview_image"][::-1, ::-1]),
        wrist_image=np.ascontiguousarray(raw["robot0_eye_in_hand_image"][::-1, ::-1]), state=state))
