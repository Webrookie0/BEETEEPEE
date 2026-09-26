import math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F

from .config import Config


def stack_observations(observations):
    return {key: np.stack([o[key] for o in observations.values()]) for key in ("self", "neighbors", "obstacles")}


def tensors(observations, device="cpu"):
    return {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in observations.items()}


class SetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(7, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh())

    def forward(self, rows):
        mask = rows[..., 7:8]
        features = rows[..., :7] / rows.new_tensor([7, 7, 7, 6, 6, 6, 1.5])
        encoded = self.net(features)
        mean = (encoded * mask).sum(dim=-2) / mask.sum(dim=-2).clamp(min=1)
        maximum = encoded.masked_fill(mask == 0, -1e9).max(dim=-2).values
        maximum = torch.where(mask.sum(dim=-2) > 0, maximum, torch.zeros_like(maximum))
        return torch.cat((mean, maximum), dim=-1)


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.own = nn.Sequential(nn.Linear(17, 64), nn.Tanh())
        self.neighbors = SetEncoder()
        self.obstacles = SetEncoder()
        self.trunk = nn.Sequential(nn.Linear(192, 128), nn.Tanh(), nn.Linear(128, 64), nn.Tanh())
        self.actor = nn.Linear(64, 3)
        self.critic = nn.Linear(64, 1)
        self.log_std = nn.Parameter(torch.full((3,), -1.0))
        nn.init.orthogonal_(self.actor.weight, 0.01)
        nn.init.zeros_(self.actor.bias)

    def forward(self, observation):
        own = observation["self"]
        scale = own.new_tensor([5, 5, 5, 3, 3, 3, 3, 3, 3, 3, 3, 3, 5, 5, 5, 10, 1])
        encoded = torch.cat((self.own(own / scale), self.neighbors(observation["neighbors"]),
                             self.obstacles(observation["obstacles"])), dim=-1)
        hidden = self.trunk(encoded)
        return self.actor(hidden), self.critic(hidden).squeeze(-1)

    def distribution(self, observation):
        mean, value = self(observation)
        return Normal(mean, self.log_std.clamp(-5, 1).exp()), value

    @staticmethod
    def log_probability(distribution, raw):
        correction = 2 * (math.log(2) - raw - F.softplus(-2 * raw))
        return (distribution.log_prob(raw) - correction).sum(-1)

    def sample(self, observation):
        distribution, value = self.distribution(observation)
        raw = distribution.sample()
        return raw.tanh(), raw, self.log_probability(distribution, raw), value

    def evaluate(self, observation, raw):
        distribution, value = self.distribution(observation)
        return self.log_probability(distribution, raw), distribution.entropy().sum(-1), value


class LearnedController:
    def __init__(self, path):
        data = torch.load(Path(path), map_location="cpu", weights_only=True)
        if data.get("format") != "swarmflow-v1":
            raise ValueError("Unsupported checkpoint format")
        self.config = Config(**data["config"])
        self.model = ActorCritic()
        self.model.load_state_dict(data["model"])
        self.model.eval()
        self.metadata = {k: v for k, v in data.items() if k != "model" and k != "optimizer"}

    def __call__(self, observations):
        with torch.no_grad():
            mean, _ = self.model(tensors(stack_observations(observations)))
        return dict(zip(observations, mean.tanh().numpy()))
