import numpy as np
import pytest
from redvla.core.types import Observation
from redvla.protocol import decode_array, encode_array, decode_observation, encode_observation, validate_actions
from redvla.envs.observation import from_libero


def test_lossless_observation_transport():
    obs = Observation(np.arange(72, dtype=np.uint8).reshape(4, 6, 3)[:, ::-1],
                      np.zeros((8, 9, 3), np.uint8), np.linspace(-1, 1, 8, dtype=np.float32))
    restored = decode_observation(encode_observation(obs))
    for key in ("image", "wrist_image", "state"):
        np.testing.assert_array_equal(getattr(obs, key), getattr(restored, key))


@pytest.mark.parametrize("bad", [[], [1] * 7, [[0] * 6], [[float("nan")] * 7], [[float("inf")] * 7]])
def test_reject_invalid_policy_actions(bad):
    with pytest.raises(ValueError):
        validate_actions(bad)


def test_reject_object_array_and_forged_shape():
    with pytest.raises(ValueError):
        encode_array(np.array([object()], dtype=object))
    wire = encode_array(np.zeros(3, np.float32))
    wire["shape"] = [999999999]
    with pytest.raises(ValueError):
        decode_array(wire)
    wire["shape"] = [4]
    with pytest.raises(ValueError):
        decode_array(wire)


def test_observation_orientation_and_state_do_not_mutate_input():
    img = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    quat = np.array([0, 0, 0, 1.00001])
    obs = from_libero({"agentview_image": img, "robot0_eye_in_hand_image": img,
                       "robot0_eef_pos": [1, 2, 3], "robot0_eef_quat": quat, "robot0_gripper_qpos": [0.02, -0.02]})
    np.testing.assert_array_equal(obs.image, img[::-1, ::-1])
    np.testing.assert_array_equal(obs.state[:6], [1, 2, 3, 0, 0, 0])
    assert quat[3] == 1.00001
