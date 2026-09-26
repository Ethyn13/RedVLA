import numpy as np
import pytest

from redvla.core.constraint import ConstraintValidator
from redvla.core.types import EpisodeConfig


@pytest.mark.parametrize("iteration", [0, 1])
@pytest.mark.parametrize("violation", ["penetration", "drift"])
def test_every_search_candidate_is_checked_and_reduced(monkeypatch, iteration, violation):
    class Environment:
        def set_object_position_in_state(self, state, objects, center, relative=None):
            return center.copy()

    validator = ConstraintValidator(
        Environment(), [], EpisodeConfig(constraint_threshold=0.03, max_constraint_retries=2)
    )
    probed = []

    def probe(state, center, previous, relative):
        probed.append(center.copy())
        invalid = center[0] > 0.02
        return center.copy(), invalid and violation == "penetration", 0.1 if invalid and violation == "drift" else 0.0

    monkeypatch.setattr(validator, "_probe", probe)
    initial = np.array([0.0, 0.0, 0.9])
    proposed = np.array([0.04, 0.0, 0.9])
    state, accepted, rejected, drift = validator.validate(
        initial, initial, proposed, iteration=iteration
    )

    assert len(probed) == 2
    assert rejected == 1
    np.testing.assert_allclose(accepted, [0.02, 0.0, 0.9])
    np.testing.assert_array_equal(state, accepted)
    np.testing.assert_array_equal(proposed, [0.04, 0.0, 0.9])
    assert drift == (0.1 if violation == "drift" else 0.0)
