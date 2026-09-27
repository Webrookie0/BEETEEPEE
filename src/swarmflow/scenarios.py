from dataclasses import dataclass
import numpy as np

from .config import GEOMETRIES

SCENARIOS = ("hold", "follow", "crossing", "head_on", "double_crossing", "slalom", "dense",
             "turn", "gust", "pilot", "terrain_follow")


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


def _enforce_min_spacing(offsets, spacing):
    if len(offsets) < 2:
        return offsets
    delta = offsets[:, None, :2] - offsets[None, :, :2]
    distances = np.linalg.norm(delta, axis=-1)
    np.fill_diagonal(distances, np.inf)
    nearest = float(np.min(distances))
    if nearest < spacing and nearest > 1e-9:
        offsets = offsets * (spacing / nearest)
    return offsets - offsets.mean(axis=0)


def _polygon_ring(count, spacing, sides):
    """Place drones inset on each edge so corners keep Euclidean spacing."""
    base, rem = divmod(count, sides)
    per_side = [base + (1 if i < rem else 0) for i in range(sides)]
    side = (max(per_side) + 1) * spacing
    angles = 2 * np.pi * np.arange(sides) / sides - np.pi / 2
    verts = np.stack((np.cos(angles), np.sin(angles)), axis=1)
    verts *= side / np.linalg.norm(verts[1] - verts[0])
    offsets = np.zeros((count, 3))
    index = 0
    for s, n in enumerate(per_side):
        if n == 0:
            continue
        a, b = verts[s], verts[(s + 1) % sides]
        for k in range(n):
            t = (k + 1) / (n + 1)
            offsets[index, :2] = a + t * (b - a)
            index += 1
    return _enforce_min_spacing(offsets, spacing)


def formation(count, spacing, geometry="grid"):
    if geometry not in GEOMETRIES:
        raise ValueError(f"Unknown geometry {geometry!r}")
    # Keep a small floor so rotors do not overlap; compaction UI sets spacing.
    gap = max(float(spacing), 0.75)
    if geometry == "grid":
        columns = int(np.ceil(np.sqrt(count)))
        offsets = np.array([[i // columns, i % columns, 0] for i in range(count)], dtype=float)
        return _enforce_min_spacing((offsets - offsets.mean(axis=0)) * gap, gap)
    if geometry == "circle":
        radius = 0.0 if count < 2 else gap / (2 * np.sin(np.pi / count))
        angles = 2 * np.pi * np.arange(count) / count
        offsets = np.stack((radius * np.cos(angles), radius * np.sin(angles), np.zeros(count)), axis=1)
        return _enforce_min_spacing(offsets, gap)
    return _polygon_ring(count, gap, 4 if geometry == "square" else 3)


def ground_height(x, y, world, peaks=()):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if world != "hills":
        return 0.0 * x
    # Steep continuous ridges (mountain-like), not a shallow swell.
    height = (1.15 * np.sin(0.55 * x) * (0.7 + 0.3 * np.cos(0.3 * y))
              + 0.75 * np.sin(0.33 * x + 1.0)
              + 0.5 * np.sin(0.95 * x) * np.cos(0.45 * y)
              + 0.28 * np.sin(0.2 * x - 0.28 * y))
    for px, py, amplitude, width in peaks:
        radius = ((x - px) ** 2 + (y - py) ** 2) / max(2.0 * width * width, 1e-6)
        height = height + amplitude * np.exp(-radius)
    return np.maximum(height, 0.0)


def make_peaks(world, config, rng, x0=2.0, x1=20.0):
    """Peak centers that sculpt the ground mesh only (never drawn as objects)."""
    if world != "hills":
        return []
    peaks = []
    x = float(x0) + 1.0
    while x < float(x1):
        # Paired ridge knobs: taller + narrower so flanks read as hills/mountains.
        for side in (-1.0, 1.0):
            peaks.append((
                float(x + rng.uniform(-0.3, 0.3)),
                float(side * rng.uniform(0.6, 2.4) + rng.uniform(-0.3, 0.3)),
                float(rng.uniform(1.4, 2.5)),
                float(rng.uniform(1.05, 1.7)),
            ))
        x += float(rng.uniform(2.4, 3.6))
    return peaks


def canyon_center_y(x):
    x = np.asarray(x, dtype=float)
    return 1.6 * np.sin(0.22 * x) + 0.55 * np.sin(0.48 * x + 0.7)


def make_canyon_segment(x0, x1, config, count=10, tag="0"):
    xs = np.linspace(x0, x1, count)
    half = 3.2
    walls = []
    for i, x in enumerate(xs):
        cy = float(canyon_center_y(x))
        walls.append(Obstacle(f"canyon-l-{tag}-{i}", np.array([float(x), cy + half, config.altitude]),
                              np.zeros(3), 0.55))
        walls.append(Obstacle(f"canyon-r-{tag}-{i}", np.array([float(x), cy - half, config.altitude]),
                              np.zeros(3), 0.55))
    return walls


def make_terrain(world, config, rng, peaks=()):
    if world == "empty":
        return []
    if world == "canyon":
        return make_canyon_segment(2.0, 22.0, config, count=24, tag="base")
    if world == "pillars":
        return [Obstacle(f"pillar-{i}", np.array([
            float(rng.uniform(3.5, 13.0)), float(rng.uniform(-2.8, 2.8)), config.altitude
        ]), np.zeros(3), float(rng.uniform(0.35, 0.6))) for i in range(7)]
    if world == "hills":
        return []
    raise ValueError(f"Unknown world {world!r}")


def target_state(time, scenario, config):
    speed = 0 if scenario in ("hold", "pilot") else config.target_speed
    position = np.array([2 + speed * time, 0., config.altitude])
    velocity = np.array([speed, 0., 0.])
    if scenario == "turn":
        position[1] = 1.8 * np.sin(time * 0.22)
        velocity[1] = 1.8 * 0.22 * np.cos(time * 0.22)
        position[2] += 0.35 * np.sin(time * 0.3)
        velocity[2] = 0.105 * np.cos(time * 0.3)
    if scenario == "terrain_follow":
        position[1] = 1.2 * np.sin(time * 0.18)
        velocity[1] = 1.2 * 0.18 * np.cos(time * 0.18)
    return position, velocity


def crossing(config, time, name, speed=1.6, radius=0.7, angle=np.pi / 2, scenario="crossing",
             target=None):
    if target is None:
        target, target_velocity = target_state(time, scenario, config)
    else:
        target, target_velocity = target
    center = target - np.array([config.stand_off, 0, 0])
    direction = np.array([np.cos(angle), np.sin(angle), 0.])
    extent = float(np.max(np.linalg.norm(
        formation(config.drones, config.spacing, config.geometry), axis=1)))
    distance = max(5.0, extent + radius + 2.0)
    return Obstacle(name, center - direction * distance, target_velocity + direction * speed,
                    radius, time, time + 2 * distance / speed, True)


def make_obstacles(scenario, config, rng):
    if scenario in ("crossing", "double_crossing"):
        first_time = float(rng.uniform(3.0, 5.5))
        first_angle = float(rng.uniform(np.pi / 2 - 0.35, np.pi / 2 + 0.35))
        first = crossing(config, first_time, "crossing-1", float(rng.uniform(1.4, 1.8)),
                         float(rng.uniform(0.6, 0.8)), first_angle)
        result = [first]
        if scenario == "double_crossing":
            second_time = first_time + float(rng.uniform(6.0, 10.0))
            second_angle = float(rng.uniform(-np.pi / 2 - 0.35, -np.pi / 2 + 0.35))
            result.append(crossing(config, second_time, "crossing-2", float(rng.uniform(1.4, 1.8)),
                                   float(rng.uniform(0.6, 0.8)), second_angle))
        return result
    if scenario == "head_on":
        return [crossing(config, float(rng.uniform(3.0, 5.5)), "head-on-1",
                         float(rng.uniform(1.8, 2.2)), float(rng.uniform(0.7, 0.9)),
                         np.pi + float(rng.uniform(-0.2, 0.2)))]
    if scenario == "slalom":
        return [Obstacle(f"static-{i}", np.array([
            x + float(rng.uniform(-0.3, 0.3)), y + float(rng.uniform(-0.4, 0.4)), config.altitude
        ]), np.zeros(3), 0.7) for i, (x, y) in enumerate(((4, 0.3), (7, -0.8), (10, 0.6)))]
    if scenario == "dense":
        count = int(rng.integers(8, 13))
        span = 10.0 / max(count - 1, 1)
        return [Obstacle(f"dense-{i}", np.array([
            4.0 + i * span + float(rng.uniform(-0.25, 0.25)),
            (1 if i % 2 == 0 else -1) * (0.9 + 0.35 * (i % 3)) + float(rng.uniform(-0.2, 0.2)),
            config.altitude
        ]), np.zeros(3), float(rng.uniform(0.45, 0.7))) for i in range(count)]
    return []
