from functools import lru_cache
import copy
import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from .config import Config
from .geometry import limit_norm, swept_distance
from .scenarios import (SCENARIOS, Obstacle, crossing, formation, ground_height, make_canyon_segment,
                        make_obstacles, make_peaks, make_terrain, target_state)

REWARD_ERROR_CAP_M = 5.0
FAILURE_PENALTY = 100.0
OBSTACLE_MARGIN_M = 1.0
OBSTACLE_WEIGHT = 3.0
RECOVERY_WINDOW_S = 5.0


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
        self.offsets = formation(self.config.drones, self.config.spacing, self.config.geometry)
        self.guide_position = np.array([2., 0., self.config.altitude])
        self.guide_velocity = np.zeros(3)
        self.peaks = []
        self.continuous = False
        self._last_stream = 0.0
        self._canyon_built_to = 22.0
        self._peaks_built_to = 22.0
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

    def _terrain_world(self):
        return "hills" if self.scenario == "terrain_follow" else self.config.world

    def _min_fly_z(self, x, y):
        floor = ground_height(np.asarray(x), np.asarray(y), self._terrain_world(), getattr(self, "peaks", ()))
        return float(np.asarray(floor).reshape(-1)[0] + self.config.radius + 0.22)

    def target(self, time=None):
        """Reference stays on its path; it does not climb over hills."""
        if self.scenario == "pilot":
            return self.guide_position.copy(), self.guide_velocity.copy()
        return target_state(self.time if time is None else time, self.scenario, self.config)

    def slots(self, time=None):
        """Drones climb only as needed to clear local ground under their slots."""
        target, velocity = self.target(time)
        slots = target - np.array([self.config.stand_off, 0., 0.]) + self.offsets
        if self._terrain_world() == "hills":
            # Short look-ahead: climb late and stay close to the ridge, not high early.
            speed = float(np.linalg.norm(velocity[:2]))
            if speed >= 0.3:
                vx, vy = float(velocity[0]), float(velocity[1])
            else:
                vx, vy = float(max(self.config.target_speed, 1.2)), 0.0
            look = (0.0, 1.0, 2.0, 3.2)
            for i in range(len(slots)):
                need = max(self._min_fly_z(slots[i, 0] + t * vx,
                                           slots[i, 1] + t * vy) for t in look)
                slots[i, 2] = max(slots[i, 2], need)
        return slots

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.seed_value = seed
        self.agents = self.possible_agents[:]
        self.time = 0.0
        self.steps = 0
        self.guide_position = np.array([2., 0., self.config.altitude])
        self.guide_velocity = np.zeros(3)
        world = "hills" if self.scenario == "terrain_follow" else self.config.world
        self.peaks = make_peaks(world, self.config, self.rng, 2.0, 22.0)
        self._peaks_built_to = 22.0
        self.positions = self.slots() + self.rng.uniform(-0.07, 0.07, (self.config.drones, 3))
        self.velocities = np.tile(self.target()[1], (self.config.drones, 1))
        self.commands = self.velocities.copy()
        self.accelerations = np.zeros_like(self.positions)
        self.obstacle_list = make_obstacles(self.scenario, self.config, self.rng)
        self.obstacle_list.extend(make_terrain(world, self.config, self.rng, self.peaks))
        self._last_stream = 0.0
        self._canyon_built_to = 22.0
        self.events = {}
        self.collision = False
        self.failure = None
        self.culprits = np.zeros(self.config.drones, dtype=bool)
        self.min_clearance = float("inf")
        self.error_integral = 0.
        self.centroid_integral = 0.
        self.smoothness_integral = 0.
        self.returns = np.zeros(self.config.drones)
        self.last_observations = self._observe()
        return self.last_observations, {a: {} for a in self.agents}

    def _maybe_stream_world(self):
        if not self.continuous or self.failure or not self.agents:
            return
        if self.time - self._last_stream < 3.0:
            return
        self._last_stream = self.time
        # Drop finished movers so the list does not grow forever.
        self.obstacle_list = [o for o in self.obstacle_list
                              if o.end > self.time - 0.5 or not o.recovery]
        target, _ = self.target()
        # Only traffic scenarios auto-spawn movers; follow/hold/etc. stay empty unless Inject is used.
        if self.scenario in ("crossing", "double_crossing", "head_on"):
            angle = float(self.rng.choice([np.pi / 2, -np.pi / 2, np.pi]))
            try:
                self.inject(speed=float(self.rng.uniform(1.3, 1.9)),
                            radius=float(self.rng.uniform(0.55, 0.85)), angle=angle)
            except ValueError:
                pass
        if self._terrain_world() == "canyon" and target[0] + 12.0 > self._canyon_built_to:
            x0, x1 = self._canyon_built_to, self._canyon_built_to + 12.0
            self.obstacle_list.extend(make_canyon_segment(x0, x1, self.config, count=12,
                                                          tag=f"{self._canyon_built_to:.0f}"))
            self._canyon_built_to = x1
        if self._terrain_world() == "hills" and target[0] + 14.0 > self._peaks_built_to:
            x0, x1 = self._peaks_built_to, self._peaks_built_to + 14.0
            self.peaks.extend(make_peaks("hills", self.config, self.rng, x0, x1))
            # Keep nearby peaks only so the mesh stays bounded.
            self.peaks = [p for p in self.peaks if p[0] > target[0] - 18.0]
            self._peaks_built_to = x1
        traffic = sum(1 for o in self.active_obstacles() if o.recovery)
        if self._terrain_world() == "pillars" and traffic < self.config.obstacles:
            self.obstacle_list.append(Obstacle(
                f"pillar-stream-{self.steps}",
                np.array([float(target[0] + self.rng.uniform(6, 10)),
                          float(self.rng.uniform(-2.5, 2.5)), self.config.altitude]),
                np.zeros(3), float(self.rng.uniform(0.35, 0.55))))

    def active_obstacles(self, time=None):
        t = self.time if time is None else time
        return [o for o in self.obstacle_list if o.start <= t < o.end]

    def inject(self, speed=1.6, radius=0.7, angle=np.pi / 2):
        if not self.agents:
            raise ValueError("Reset the environment before injecting an object")
        if not np.isfinite([speed, radius, angle]).all() or not 0.3 <= speed <= 5 or not 0.2 <= radius <= 1.5:
            raise ValueError("Object speed must be 0.3–5 m/s and radius 0.2–1.5 m")
        traffic = [o for o in self.active_obstacles() if o.recovery]
        if len(traffic) >= self.config.obstacles:
            raise ValueError("Maximum simultaneous objects reached")
        obj = crossing(self.config, self.time, f"injected-{len(self.obstacle_list)}", speed,
                       radius, angle, self.scenario, target=self.target())
        self.obstacle_list.append(obj)
        self.last_observations = self._observe()
        return obj.name

    def steer(self, velocity=None, position=None):
        if self.scenario != "pilot":
            raise ValueError("Reference steering is only available in the pilot scenario")
        if not self.agents:
            raise ValueError("Reset the environment before steering the reference")
        if position is None and velocity is None:
            raise ValueError("Provide velocity and/or position")
        if position is not None:
            point = np.asarray(position, dtype=float).reshape(3)
            if not np.isfinite(point).all():
                raise ValueError("position must be finite")
            self.guide_position = point
            self.guide_velocity = np.zeros(3)
        if velocity is not None:
            command = np.asarray(velocity, dtype=float).reshape(3)
            if not np.isfinite(command).all():
                raise ValueError("velocity must be finite")
            self.guide_velocity = limit_norm(command, self.config.max_speed)
        self.last_observations = self._observe()

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

    def _ray_hit(self, origin, direction, positions, radii):
        if len(positions) == 0:
            return None
        offset = origin - positions
        b = 2.0 * offset @ direction
        c = np.einsum("ij,ij->i", offset, offset) - radii ** 2
        disc = b * b - 4.0 * c
        root = np.sqrt(np.maximum(disc, 0.0))
        entry = (-b - root) / 2.0
        exit_t = (-b + root) / 2.0
        hit = np.where(entry > 1e-6, entry, exit_t)
        hit = np.where((disc >= 0) & (hit > 1e-6) & (hit <= self.config.ray_range), hit, np.inf)
        best = float(np.min(hit))
        return None if not np.isfinite(best) else best

    def _fill_ray_rows(self, rows, own, positions, radii):
        empty = np.flatnonzero(rows[:, -1] <= 0.5)
        if len(empty) == 0 or self.config.rays <= 0:
            return rows
        origin = self.positions[own]
        hits = []
        for k in range(self.config.rays):
            angle = 2.0 * np.pi * k / self.config.rays
            direction = np.array([np.cos(angle), np.sin(angle), 0.0])
            distance = self._ray_hit(origin, direction, positions, radii)
            if distance is not None:
                hits.append((distance, direction * distance))
        hits.sort(key=lambda item: item[0])
        for slot, (_, relative) in zip(empty, hits):
            rows[slot, :3] = relative
            rows[slot, 3:6] = 0.0
            rows[slot, 6] = 0.05
            rows[slot, 7] = 1.0
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
            obstacles = self._object_rows(positions, velocities, radii, i, self.config.obstacles)
            if self.config.rays > 0:
                obstacles = self._fill_ray_rows(obstacles, i, positions, radii)
            result[agent] = {"self": own.astype(np.float32),
                "neighbors": self._object_rows(self.positions[others], self.velocities[others],
                    np.full(np.sum(others), self.config.radius), i, self.config.neighbors),
                "obstacles": obstacles}
        return result

    def _clearance(self, before, after, start, end):
        """Minimum surface clearance of each drone over one straight integration segment."""
        c = self.config
        clearance = np.full(c.drones, np.inf)
        a, b = np.triu_indices(c.drones, 1)
        distances = swept_distance(before[a] - before[b], after[a] - after[b]) - 2 * c.radius
        np.minimum.at(clearance, a, distances)
        np.minimum.at(clearance, b, distances)
        for obstacle in self.obstacle_list:
            lo, hi = max(start, obstacle.start), min(end, obstacle.end)
            if lo >= hi:
                continue
            alpha, beta = (lo - start) / (end - start), (hi - start) / (end - start)
            p0 = before + alpha * (after - before)
            p1 = before + beta * (after - before)
            dist = swept_distance(p0 - obstacle.position(lo), p1 - obstacle.position(hi))
            clearance = np.minimum(clearance, dist - c.radius - obstacle.radius)
        return clearance

    def _update_recovery(self):
        c = self.config
        settled = (np.max(np.linalg.norm(self.positions - self.slots(), axis=1)) <= c.recovery_position and
                   np.max(np.linalg.norm(self.velocities - self.target()[1], axis=1)) <= c.recovery_velocity)
        for obj in self.obstacle_list:
            if not obj.recovery or self.time < obj.start:
                continue
            event = self.events.setdefault(obj.name, {"start_s": obj.start, "clear_s": obj.end,
                     "measured_from_s": obj.end, "recovery_s": None, "settled_since_s": None})
            if self.time + 1e-9 < obj.end or event["recovery_s"] is not None:
                continue
            # A second object that arrives inside this event's recovery window disturbs the swarm again;
            # the event is then measured from the moment that object has left as well.
            overlapping = [o.end for o in self.obstacle_list if o is not obj and o.recovery
                           and o.start <= self.time < o.end and o.start < event["measured_from_s"] + RECOVERY_WINDOW_S]
            if overlapping:
                event["measured_from_s"] = max(event["measured_from_s"], max(overlapping))
                event["settled_since_s"] = None
                continue
            if settled and not self.failure:
                if event["settled_since_s"] is None:
                    event["settled_since_s"] = self.time
            else:
                event["settled_since_s"] = None
            if event["settled_since_s"] is not None and self.time - event["settled_since_s"] + 1e-9 >= c.recovery_hold:
                event["recovery_s"] = max(0., self.time - event["measured_from_s"])

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
            if self.scenario == "pilot":
                self.guide_position = self.guide_position + self.guide_velocity * h
                self.guide_position[0] = float(np.clip(self.guide_position[0], -55, 55))
                self.guide_position[1] = float(np.clip(self.guide_position[1], -55, 55))
                low = c.radius + 0.5
                if self._terrain_world() == "hills":
                    gx, gy = self.guide_position[0], self.guide_position[1]
                    vx, vy = self.guide_velocity[0], self.guide_velocity[1]
                    low = max(self._min_fly_z(gx + t * vx, gy + t * vy) for t in (0.0, 1.0, 2.0, 3.2))
                    if self.guide_position[2] <= low + 0.2 and self.guide_velocity[2] < 0:
                        self.guide_velocity[2] = 0.0
                self.guide_position[2] = float(np.clip(self.guide_position[2], low, 10 - c.radius))
            before = self.positions.copy()
            self.accelerations = limit_norm((self.commands - self.velocities) / c.response_time, c.max_accel)
            wind = np.array([0., 1.2, 0.2]) if self.scenario == "gust" and 7 <= t0 < 9 else 0.
            self.velocities += (self.accelerations + wind) * h
            self.positions += self.velocities * h
            clearance = self._clearance(before, self.positions, t0, t0 + h)
            self.min_clearance = min(self.min_clearance, float(clearance.min()))
            if np.any(clearance <= 0):
                self.collision = True
                self.failure = "collision"
                self.culprits |= clearance <= 0
            world = "hills" if self.scenario == "terrain_follow" else c.world
            floor = c.radius + ground_height(self.positions[:, 0], self.positions[:, 1], world, self.peaks)
            grounded = self.positions[:, 2] <= floor
            outside = (self.positions[:, 2] >= 10 - c.radius) | np.any(np.abs(self.positions[:, :2]) >= 60 - c.radius,
                                                                      axis=1)
            if np.any(grounded):
                self.collision = True
                self.failure = self.failure or "terrain"
                self.culprits |= grounded
            elif np.any(outside):
                self.failure = self.failure or "world_boundary"
                self.culprits |= outside
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
        danger = np.zeros(c.drones)
        for obj in self.active_obstacles():
            gap = np.linalg.norm(self.positions - obj.position(self.time), axis=1) - obj.radius - c.radius
            danger += np.maximum(0, 1 - gap / OBSTACLE_MARGIN_M) ** 2
        reward = self._reward(errors, velocity_error, proximity, danger, differences)
        self.returns += reward
        self._update_recovery()
        if self.continuous and not self.failure:
            self._maybe_stream_world()
            self.last_observations = self._observe()
        else:
            self.last_observations = self._observe()
        terminated = bool(self.failure)
        truncated = (not self.continuous) and (not terminated) and self.time >= c.duration - 1e-9
        agents = self.agents[:]
        infos = {a: {"slot_error_m": float(errors[i]), "failure": self.failure} for i, a in enumerate(agents)}
        if terminated or truncated:
            summary = self.summary()
            for info in infos.values():
                info["episode"] = summary
            self.agents = []
        return self.last_observations, dict(zip(agents, reward.tolist())), {a: terminated for a in agents}, {a: truncated for a in agents}, infos

    def _reward(self, errors, velocity_error, proximity, danger, differences):
        """Per-drone reward: +1 per second alive minus bounded costs, and a failure penalty.

        Every cost is capped, so even a badly drifted drone earns more (discounted) by staying
        alive than by ending the episode: ending early can never be the easiest way out.
        The failure penalty goes to the drones that collided, grounded or left the world; the
        episode still ends for the whole team.
        """
        c = self.config
        capped = np.minimum(errors, REWARD_ERROR_CAP_M)
        tracking = np.where(capped <= 1.0, capped ** 2, 2.0 * capped - 1.0)
        cost = (tracking + 0.12 * np.minimum(velocity_error, 3.0) ** 2 + 6 * proximity
                + OBSTACLE_WEIGHT * np.minimum(danger, 2.0)
                + 0.005 * np.sum(self.commands ** 2, axis=1) + 0.001 * np.minimum(differences, 20.0) ** 2)
        reward = c.dt * (1.0 - cost)
        if self.failure:
            reward -= np.where(self.culprits, FAILURE_PENALTY, 0.0)
        return reward

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
                "target": self.target()[0].tolist(), "target_velocity": self.target()[1].tolist(),
                "geometry": self.config.geometry,
                "world": "hills" if self.scenario == "terrain_follow" else self.config.world,
                "radius": self.config.radius,
                "peaks": [{"x": p[0], "y": p[1], "height": p[2], "width": p[3]} for p in self.peaks],
                "obstacles": [{"position": o.position(self.time).tolist(), "radius": o.radius,
                               "velocity": o.velocity.tolist(), "name": o.name} for o in self.active_obstacles()],
                "metrics": self.summary()}

    def capture_state(self):
        state = {key: getattr(self, key) for key in
                 ("time", "steps", "collision", "failure", "min_clearance",
                  "error_integral", "centroid_integral", "smoothness_integral", "seed_value")}
        for key in ("positions", "velocities", "commands", "accelerations", "returns",
                    "guide_position", "guide_velocity", "culprits"):
            state[key] = getattr(self, key).copy()
        for key in ("obstacle_list", "events", "last_observations", "peaks"):
            state[key] = copy.deepcopy(getattr(self, key))
        state["agents"] = self.agents[:]
        return state

    def restore_state(self, state):
        for key in ("time", "steps", "collision", "failure", "min_clearance",
                    "error_integral", "centroid_integral", "smoothness_integral", "seed_value"):
            setattr(self, key, state[key])
        for key in ("positions", "velocities", "commands", "accelerations", "returns",
                    "guide_position", "guide_velocity", "culprits"):
            setattr(self, key, np.array(state[key], copy=True))
        for key in ("obstacle_list", "events", "last_observations", "peaks"):
            setattr(self, key, copy.deepcopy(state[key]))
        self.agents = list(state["agents"])

    def close(self):
        pass
