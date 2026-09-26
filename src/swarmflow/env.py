from functools import lru_cache
import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from .config import Config
from .geometry import limit_norm, swept_distance
from .scenarios import SCENARIOS, crossing, formation, make_obstacles, target_state


class SwarmEnv(ParallelEnv):
    metadata = {"name": "swarmflow_v0", "render_modes": []}

    def __init__(self, config=None, scenario="crossing", render_mode=None):
        self.config = config or Config()
        if scenario not in SCENARIOS:
            raise ValueError(f"Unknown scenario {scenario!r}. Choose from {SCENARIOS}")
        self.scenario = scenario
        self.render_mode = render_mode
        self.possible_agents = [f"drone_{i}" for i in range(self.config.drones)]
        self.agents = []
        self.offsets = formation(self.config.drones, self.config.spacing)
        self.rng = np.random.default_rng()

    @lru_cache(maxsize=None)
    def observation_space(self, agent):
        c = self.config
        return spaces.Dict({"self": spaces.Box(-np.inf, np.inf, (17,), np.float32),
                            "neighbors": spaces.Box(-np.inf, np.inf, (c.neighbors, 8), np.float32),
                            "obstacles": spaces.Box(-np.inf, np.inf, (c.obstacles, 8), np.float32)})

    @lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.Box(-1., 1., (3,), np.float32)

    def target(self, time=None):
        return target_state(self.time if time is None else time, self.scenario, self.config)

    def slots(self, time=None):
        target, _ = self.target(time)
        return target - np.array([self.config.stand_off, 0., 0.]) + self.offsets

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.seed_value = seed
        self.agents = self.possible_agents[:]
        self.time = 0.0
        self.steps = 0
        self.positions = self.slots() + self.rng.uniform(-0.07, 0.07, (self.config.drones, 3))
        self.velocities = np.tile(self.target()[1], (self.config.drones, 1))
        self.commands = self.velocities.copy()
        self.accelerations = np.zeros_like(self.positions)
        self.obstacle_list = make_obstacles(self.scenario, self.config, self.rng)
        self.events = {}
        self.collision = False
        self.failure = None
        self.min_clearance = float("inf")
        self.error_integral = 0.
        self.centroid_integral = 0.
        self.smoothness_integral = 0.
        self.returns = np.zeros(self.config.drones)
        self.last_observations = self._observe()
        return self.last_observations, {a: {} for a in self.agents}

    def active_obstacles(self, time=None):
        t = self.time if time is None else time
        return [o for o in self.obstacle_list if o.start <= t < o.end]

    def inject(self, speed=1.6, radius=0.7, angle=np.pi / 2):
        if not self.agents:
            raise ValueError("Reset the environment before injecting an object")
        if not np.isfinite([speed, radius, angle]).all() or not 0.3 <= speed <= 5 or not 0.2 <= radius <= 1.5:
            raise ValueError("Object speed must be 0.3–5 m/s and radius 0.2–1.5 m")
        if len(self.active_obstacles()) >= self.config.obstacles:
            raise ValueError("Maximum simultaneous objects reached")
        obj = crossing(self.config, self.time, f"injected-{len(self.obstacle_list)}", speed,
                       radius, angle, self.scenario)
        self.obstacle_list.append(obj)
        self.last_observations = self._observe()
        return obj.name

    def _object_rows(self, positions, velocities, radii, own, count):
        rows = np.zeros((count, 8), dtype=np.float32)
        if len(positions) == 0:
            return rows
        rel = np.asarray(positions) - self.positions[own]
        distances = np.linalg.norm(rel, axis=1)
        indices = np.argsort(distances, kind="stable")
        indices = indices[distances[indices] <= self.config.sensor_range][:count]
        for k, j in enumerate(indices):
            if self.rng.random() < self.config.dropout:
                continue
            rows[k, :3] = rel[j] + self.rng.normal(0, self.config.noise, 3)
            rows[k, 3:6] = velocities[j] - self.velocities[own]
            rows[k, 6:] = [radii[j], 1.]
        return rows

    def _observe(self):
        result = {}
        active = self.active_obstacles()
        positions = np.array([o.position(self.time) for o in active]).reshape(-1, 3)
        velocities = np.array([o.velocity for o in active]).reshape(-1, 3)
        radii = np.array([o.radius for o in active])
        slots = self.slots()
        for i, agent in enumerate(self.possible_agents):
            others = np.arange(self.config.drones) != i
            own = np.concatenate((slots[i] - self.positions[i], self.velocities[i], self.target()[1],
                                  self.commands[i], self.offsets[i], [self.positions[i, 2],
                                  max(0., 1 - self.time / self.config.duration)]))
            result[agent] = {"self": own.astype(np.float32),
                "neighbors": self._object_rows(self.positions[others], self.velocities[others],
                    np.full(np.sum(others), self.config.radius), i, self.config.neighbors),
                "obstacles": self._object_rows(positions, velocities, radii, i, self.config.obstacles)}
        return result

    def _clearance(self, before, after, start, end):
        c = self.config
        a, b = np.triu_indices(c.drones, 1)
        distances = swept_distance(before[a] - before[b], after[a] - after[b]) - 2 * c.radius
        clearance = float(np.min(distances))
        for obstacle in self.obstacle_list:
            lo, hi = max(start, obstacle.start), min(end, obstacle.end)
            if lo >= hi:
                continue
            alpha, beta = (lo - start) / (end - start), (hi - start) / (end - start)
            p0 = before + alpha * (after - before)
            p1 = before + beta * (after - before)
            dist = swept_distance(p0 - obstacle.position(lo), p1 - obstacle.position(hi))
            clearance = min(clearance, float(np.min(dist) - c.radius - obstacle.radius))
        return clearance

    def _update_recovery(self):
        c = self.config
        settled = (np.max(np.linalg.norm(self.positions - self.slots(), axis=1)) <= c.recovery_position and
                   np.max(np.linalg.norm(self.velocities - self.target()[1], axis=1)) <= c.recovery_velocity)
        for obj in self.obstacle_list:
            if not obj.recovery or self.time < obj.start:
                continue
            event = self.events.setdefault(obj.name, {"start_s": obj.start, "clear_s": obj.end,
                     "recovery_s": None, "settled_since_s": None})
            if self.time + 1e-9 < obj.end or event["recovery_s"] is not None:
                continue
            if settled and not self.failure:
                if event["settled_since_s"] is None:
                    event["settled_since_s"] = self.time
            else:
                event["settled_since_s"] = None
            if event["settled_since_s"] is not None and self.time - event["settled_since_s"] + 1e-9 >= c.recovery_hold:
                event["recovery_s"] = max(0., self.time - obj.end)

    def step(self, actions):
        if not self.agents:
            return {}, {}, {}, {}, {}
        if set(actions) != set(self.agents):
            raise ValueError("Provide one action for each active agent")
        commands = np.asarray([actions[a] for a in self.agents], dtype=float)
        if commands.shape != (self.config.drones, 3) or not np.isfinite(commands).all():
            raise ValueError("Actions must be finite 3D vectors")
        if np.any(np.abs(commands) > 1 + 1e-6):
            raise ValueError("Actions must lie in [-1, 1]")
        c = self.config
        previous = self.commands.copy()
        self.commands = limit_norm(commands * c.max_speed, c.max_speed)
        h = c.dt / c.substeps
        for sub in range(c.substeps):
            t0 = self.time + sub * h
            before = self.positions.copy()
            self.accelerations = limit_norm((self.commands - self.velocities) / c.response_time, c.max_accel)
            wind = np.array([0., 1.2, 0.2]) if self.scenario == "gust" and 7 <= t0 < 9 else 0.
            self.velocities += (self.accelerations + wind) * h
            self.positions += self.velocities * h
            clearance = self._clearance(before, self.positions, t0, t0 + h)
            self.min_clearance = min(self.min_clearance, clearance)
            if clearance <= 0:
                self.collision = True
                self.failure = "collision"
            if (np.any(self.positions[:, 2] <= c.radius) or np.any(self.positions[:, 2] >= 10 - c.radius)
                    or np.any(np.abs(self.positions[:, :2]) >= 60 - c.radius)):
                self.failure = self.failure or "world_boundary"
        self.steps += 1
        self.time = self.steps * c.dt
        errors = np.linalg.norm(self.positions - self.slots(), axis=1)
        velocity_error = np.linalg.norm(self.velocities - self.target()[1], axis=1)
        differences = np.linalg.norm(self.commands - previous, axis=1) / c.dt
        self.error_integral += float(np.mean(errors ** 2)) * c.dt
        self.centroid_integral += float(np.linalg.norm(self.positions.mean(axis=0) - self.slots().mean(axis=0)) ** 2) * c.dt
        self.smoothness_integral += float(np.mean(differences ** 2)) * c.dt
        pair = np.linalg.norm(self.positions[:, None] - self.positions[None, :], axis=-1)
        np.fill_diagonal(pair, np.inf)
        proximity = np.maximum(0, 0.45 - (pair.min(axis=1) - 2 * c.radius)) ** 2
        for obj in self.active_obstacles():
            gap = np.linalg.norm(self.positions - obj.position(self.time), axis=1) - obj.radius - c.radius
            proximity += np.maximum(0, 0.45 - gap) ** 2
        reward = -c.dt * (errors ** 2 + 0.12 * velocity_error ** 2 + 6 * proximity +
                         0.005 * np.sum(self.commands ** 2, axis=1) + 0.002 * differences ** 2)
        if self.failure:
            reward -= 200.
        self.returns += reward
        self._update_recovery()
        self.last_observations = self._observe()
        terminated = bool(self.failure)
        truncated = not terminated and self.time >= c.duration - 1e-9
        agents = self.agents[:]
        infos = {a: {"slot_error_m": float(errors[i]), "failure": self.failure} for i, a in enumerate(agents)}
        if terminated or truncated:
            summary = self.summary()
            for info in infos.values():
                info["episode"] = summary
            self.agents = []
        return self.last_observations, dict(zip(agents, reward.tolist())), {a: terminated for a in agents}, {a: truncated for a in agents}, infos

    def summary(self):
        duration = max(self.time, 1e-9)
        return {"scenario": self.scenario, "seed": self.seed_value, "drones": self.config.drones,
                "duration_s": self.time, "collision": self.collision, "failure": self.failure,
                "min_surface_clearance_m": self.min_clearance if np.isfinite(self.min_clearance) else None,
                "slot_rmse_m": float(np.sqrt(self.error_integral / duration)),
                "centroid_rmse_m": float(np.sqrt(self.centroid_integral / duration)),
                "command_derivative_rms_m_s2": float(np.sqrt(self.smoothness_integral / duration)),
                "mean_return": float(np.mean(self.returns)),
                "events": [{"name": k, **{p: v for p, v in e.items() if p != "settled_since_s"}}
                           for k, e in self.events.items()]}

    def snapshot(self):
        return {"time": self.time, "scenario": self.scenario, "running": bool(self.agents),
                "positions": self.positions.tolist(), "velocities": self.velocities.tolist(),
                "accelerations": self.accelerations.tolist(), "slots": self.slots().tolist(),
                "target": self.target()[0].tolist(), "radius": self.config.radius,
                "obstacles": [{"position": o.position(self.time).tolist(), "radius": o.radius,
                               "velocity": o.velocity.tolist(), "name": o.name} for o in self.active_obstacles()],
                "metrics": self.summary()}

    def close(self):
        pass
