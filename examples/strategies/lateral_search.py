"""A bounded, seed-controlled XY random walk that preserves the scene's object height."""
import numpy as np
from redvla.core.optimizer import PlacementOptimizer


class LateralSearch(PlacementOptimizer):
    def __init__(self, config, initial_threat_pos=None, step_m=0.01, max_radius_m=0.05):
        if not np.isfinite([step_m, max_radius_m]).all() or step_m <= 0 or max_radius_m <= 0:
            raise ValueError("step_m and max_radius_m must be finite positive numbers")
        self.step = step_m
        self.radius = max_radius_m
        self.origin = None if initial_threat_pos is None else np.asarray(initial_threat_pos, dtype=float).copy()
        self.rng = np.random.default_rng(config.seed)

    def compute_next_position(self, current_pos, trajectory):
        if self.origin is None:
            self.origin = np.asarray(current_pos, dtype=float).copy()
        angle = self.rng.uniform(0, 2 * np.pi)
        proposal = np.asarray(current_pos, dtype=float).copy()
        proposal[:2] += self.step * np.array([np.cos(angle), np.sin(angle)])
        displacement = proposal[:2] - self.origin[:2]
        length = np.linalg.norm(displacement)
        if length > self.radius:
            proposal[:2] = self.origin[:2] + displacement * self.radius / length
        return proposal
