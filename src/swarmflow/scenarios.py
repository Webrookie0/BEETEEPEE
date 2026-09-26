from dataclasses import dataclass
import numpy as np

SCENARIOS = ("hold", "follow", "crossing", "head_on", "double_crossing", "slalom", "turn", "gust")


@dataclass
class Obstacle:
    name: str
    origin: np.ndarray
    velocity: np.ndarray
    radius: float
    start: float = 0.0
    end: float = 1e6
    recovery: bool = False

    def position(self, time):
        return self.origin + self.velocity * (time - self.start)


def formation(count, spacing):
    columns = int(np.ceil(np.sqrt(count)))
    offsets = np.array([[i // columns, i % columns, 0] for i in range(count)], dtype=float)
    return (offsets - offsets.mean(axis=0)) * spacing


def target_state(time, scenario, config):
    speed = 0 if scenario == "hold" else config.target_speed
    position = np.array([2 + speed * time, 0., config.altitude])
    velocity = np.array([speed, 0., 0.])
    if scenario == "turn":
        position[1] = 1.8 * np.sin(time * 0.22)
        velocity[1] = 1.8 * 0.22 * np.cos(time * 0.22)
        position[2] += 0.35 * np.sin(time * 0.3)
        velocity[2] = 0.105 * np.cos(time * 0.3)
    return position, velocity


def crossing(config, time, name, speed=1.6, radius=0.7, angle=np.pi / 2, scenario="crossing"):
    target, target_velocity = target_state(time, scenario, config)
    center = target - np.array([config.stand_off, 0, 0])
    direction = np.array([np.cos(angle), np.sin(angle), 0.])
    extent = float(np.max(np.linalg.norm(formation(config.drones, config.spacing), axis=1)))
    distance = max(5.0, extent + radius + 2.0)
    return Obstacle(name, center - direction * distance, target_velocity + direction * speed,
                    radius, time, time + 2 * distance / speed, True)


def make_obstacles(scenario, config, rng):
    if scenario in ("crossing", "double_crossing"):
        first = crossing(config, 4.0, "crossing-1", float(rng.uniform(1.4, 1.8)),
                         float(rng.uniform(0.6, 0.8)))
        result = [first]
        if scenario == "double_crossing":
            result.append(crossing(config, 12.0, "crossing-2", 1.7, 0.65, -np.pi / 2))
        return result
    if scenario == "head_on":
        return [crossing(config, 4.0, "head-on-1", 2.0, 0.8, np.pi)]
    if scenario == "slalom":
        return [Obstacle(f"static-{i}", np.array([x, y, config.altitude]), np.zeros(3), 0.7)
                for i, (x, y) in enumerate(((4, 0.3), (7, -0.8), (10, 0.6)))]
    return []
