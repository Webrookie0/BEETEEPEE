from dataclasses import asdict, dataclass
import math

GEOMETRIES = ("grid", "circle", "square", "triangle")
WORLDS = ("empty", "hills", "canyon", "pillars")


@dataclass(frozen=True)
class Config:
    drones: int = 8
    dt: float = 0.05
    substeps: int = 5
    duration: float = 24.0
    radius: float = 0.18
    spacing: float = 1.25
    altitude: float = 3.0
    stand_off: float = 3.0
    max_speed: float = 3.0
    max_accel: float = 5.0
    response_time: float = 0.28
    target_speed: float = 0.55
    sensor_range: float = 7.0
    neighbors: int = 6
    obstacles: int = 6
    noise: float = 0.0
    dropout: float = 0.0
    recovery_position: float = 0.35
    recovery_velocity: float = 0.35
    recovery_hold: float = 1.0
    rays: int = 0
    ray_range: float = 5.0
    geometry: str = "grid"
    world: str = "empty"

    def __post_init__(self):
        if self.geometry not in GEOMETRIES:
            raise ValueError(f"geometry must be one of {GEOMETRIES}")
        if self.world not in WORLDS:
            raise ValueError(f"world must be one of {WORLDS}")
        for key, value in asdict(self).items():
            if isinstance(value, str):
                continue
            if not math.isfinite(value):
                raise ValueError(f"{key} must be finite")
        for key in ("drones", "substeps", "neighbors", "obstacles"):
            value = getattr(self, key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        if not 2 <= self.drones <= 32:
            raise ValueError("drones must be between 2 and 32")
        if not isinstance(self.rays, int) or isinstance(self.rays, bool) or self.rays < 0:
            raise ValueError("rays must be a nonnegative integer")
        for key in ("dt", "duration", "radius", "spacing", "altitude", "stand_off",
                    "max_speed", "max_accel", "response_time", "sensor_range",
                    "recovery_position", "recovery_velocity", "recovery_hold", "ray_range"):
            if getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive")
        if self.spacing <= 2 * self.radius + 0.2:
            raise ValueError("spacing must leave at least 0.2 m of surface clearance")
        if not 0 <= self.dropout < 1 or self.noise < 0:
            raise ValueError("dropout must be in [0, 1); noise must be nonnegative")
        if not 0 <= self.target_speed < self.max_speed:
            raise ValueError("target_speed must be nonnegative and below max_speed")
        if self.dt > 0.1 or self.dt / self.substeps > self.response_time / 2:
            raise ValueError("time step is too large for the flight response")

    def to_dict(self):
        return asdict(self)
