from dataclasses import asdict, dataclass
import math


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

    def __post_init__(self):
        for key, value in asdict(self).items():
            if not math.isfinite(value):
                raise ValueError(f"{key} must be finite")
        for key in ("drones", "substeps", "neighbors", "obstacles"):
            value = getattr(self, key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        if not 2 <= self.drones <= 32:
            raise ValueError("drones must be between 2 and 32")
        for key in ("dt", "duration", "radius", "spacing", "altitude", "stand_off",
                    "max_speed", "max_accel", "response_time", "sensor_range",
                    "recovery_position", "recovery_velocity", "recovery_hold"):
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
