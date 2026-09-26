import numpy as np
import pytest

from redvla.core.optimizer import CheckedPlacementOptimizer, DirectionalOptimizer, make_optimizer
from redvla.core.types import EpisodeConfig


def test_default_optimizer_unchanged():
    optimizer = make_optimizer(EpisodeConfig())
    assert isinstance(optimizer, DirectionalOptimizer)
    pos = np.array([0.0, 0.1, 0.2])
    np.testing.assert_array_equal(optimizer.compute_next_position(pos, []), pos)


def test_strategy_seed_radius_and_input_preserved():
    config = EpisodeConfig(
        placement_optimizer="examples.strategies.lateral_search:LateralSearch",
        placement_options={"step_m": 0.02, "max_radius_m": 0.03}, seed=17,
    )
    origin = np.array([0.3, -0.2, 0.8])
    a, b = (make_optimizer(config, origin) for _ in range(2))
    pa, pb = origin.copy(), origin.copy()
    for _ in range(50):
        before = pa.copy()
        na, nb = a.compute_next_position(pa, []), b.compute_next_position(pb, [])
        np.testing.assert_array_equal(pa, before)
        np.testing.assert_array_equal(na, nb)
        assert np.linalg.norm(na[:2] - origin[:2]) <= 0.03 + 1e-12
        assert na[2] == origin[2]
        pa, pb = na, nb


@pytest.mark.parametrize("proposal", [[0, 1], [0, 1, float("nan")], [0, 1, 2]])
def test_bad_strategy_proposal_rejected(proposal):
    class BadStrategy:
        def compute_next_position(self, pos, trajectory):
            pos[:] = 99
            return proposal
    original = np.array([0., 0., 0.8])
    with pytest.raises(ValueError):
        CheckedPlacementOptimizer(BadStrategy()).compute_next_position(original, [])
    np.testing.assert_array_equal(original, [0, 0, 0.8])


def test_strategy_requires_base_class():
    with pytest.raises(TypeError, match="inherit PlacementOptimizer"):
        make_optimizer(EpisodeConfig(placement_optimizer="builtins:dict"))
