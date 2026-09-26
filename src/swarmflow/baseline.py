import itertools
import numpy as np

from .geometry import limit_norm


class PredictiveController:
    """Sampled local velocity planning; not ORCA and not a safety guarantee."""

    def __init__(self, config):
        self.config = config
        directions = np.array([v for v in itertools.product((-1, 0, 1), repeat=3) if any(v)], dtype=float)
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        self.candidates = np.concatenate([directions * speed for speed in (0.8, 1.6, config.max_speed)])
        self.times = np.arange(0.1, 2.01, 0.1)

    def action(self, observation):
        c = self.config
        own = observation["self"]
        goal, velocity, target_velocity, previous = own[:3], own[3:6], own[6:9], own[9:12]
        nominal = limit_norm(target_velocity + 1.8 * goal, c.max_speed)
        tracking = [target_velocity + gain * goal for gain in (0.35, 0.6, 0.9, 1.2)]
        candidates = np.vstack((nominal, *tracking, target_velocity, velocity, np.zeros(3), self.candidates))
        candidates = limit_norm(candidates, c.max_speed)
        positions = np.zeros((len(candidates), len(self.times), 3))
        v = np.broadcast_to(velocity, candidates.shape).copy()
        p = np.zeros_like(v)
        for k in range(len(self.times)):
            v += limit_norm((candidates - v) / c.response_time, c.max_accel) * 0.1
            p += v * 0.1
            positions[:, k] = p
        desired = goal + self.times[:, None] * target_velocity
        errors = np.sum((positions - desired) ** 2, axis=-1)
        cost = 1.5 * errors.mean(axis=1) + 2 * errors[:, -1]
        cost += 0.12 * np.sum((candidates - nominal) ** 2, axis=1)
        cost += 0.06 * np.sum((candidates - previous) ** 2, axis=1)
        for kind in ("neighbors", "obstacles"):
            rows = observation[kind]
            rows = rows[rows[:, -1] > 0.5]
            if not len(rows):
                continue
            other = rows[:, None, :3] + (rows[:, None, 3:6] + velocity) * self.times[None, :, None]
            gaps = np.linalg.norm(positions[:, None] - other[None], axis=-1) - rows[None, :, 6:7] - c.radius
            cost += 180 * np.maximum(0, 0.35 - gaps).max(axis=(1, 2)) ** 2
            cost += 20000 * np.maximum(0, 0.08 - gaps).max(axis=(1, 2)) ** 2
        altitude = own[15] + positions[:, :, 2]
        cost += 5000 * (np.maximum(0, 0.6 - altitude).max(axis=1) ** 2 +
                        np.maximum(0, altitude - 9.4).max(axis=1) ** 2)
        return (candidates[int(np.argmin(cost))] / c.max_speed).astype(np.float32)

    def __call__(self, observations):
        return {agent: self.action(obs) for agent, obs in observations.items()}


class SlotController:
    """No-avoidance ablation for checking whether obstacle tests are meaningful."""

    def __init__(self, config):
        self.config = config

    def __call__(self, observations):
        return {agent: (limit_norm(obs["self"][6:9] + 1.8 * obs["self"][:3], self.config.max_speed)
                        / self.config.max_speed).astype(np.float32) for agent, obs in observations.items()}
