import math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F

from .config import Config

CHECKPOINT_FORMATS = ("swarmflow-v1", "swarmflow-v2")
CHECKPOINT_FORMAT = "swarmflow-v2"


def stack_observations(observations):
    return {key: np.stack([o[key] for o in observations.values()]) for key in ("self", "neighbors", "obstacles")}


def tensors(observations, device="cpu"):
    return {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in observations.items()}


class SetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(7, 32), nn.Tanh(), nn.Linear(32, 64), nn.Tanh())
        self.query = nn.Parameter(torch.zeros(1, 1, 64))
        self.attn = nn.MultiheadAttention(64, 2, batch_first=True)

    def forward(self, rows):
        mask = rows[..., 7]
        features = rows[..., :7] / rows.new_tensor([7, 7, 7, 6, 6, 6, 1.5])
        encoded = self.net(features)
        padding = mask == 0
        empty = padding.all(-1)
        if empty.any():
            padding = padding.clone()
            padding[empty] = False
        pooled, _ = self.attn(self.query.expand(encoded.size(0), -1, -1), encoded, encoded,
                              key_padding_mask=padding, need_weights=False)
        out = pooled.squeeze(1)
        return torch.where(empty.unsqueeze(-1), out.new_zeros(out.shape), out)


class ActorCritic(nn.Module):
    def __init__(self, central_critic=False):
        super().__init__()
        self.central_critic = central_critic
        self.own = nn.Sequential(nn.Linear(17, 64), nn.Tanh())
        self.neighbors = SetEncoder()
        self.obstacles = SetEncoder()
        self.trunk = nn.Sequential(nn.Linear(192, 128), nn.Tanh(), nn.Linear(128, 64), nn.Tanh())
        self.actor = nn.Linear(64, 3)
        self.critic = nn.Linear(64, 1)
        if central_critic:
            self.central = nn.Sequential(nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))
        self.log_std = nn.Parameter(torch.full((3,), -1.0))
        nn.init.orthogonal_(self.actor.weight, 0.01)
        nn.init.zeros_(self.actor.bias)

    def encode(self, observation):
        own = observation["self"]
        scale = own.new_tensor([5, 5, 5, 3, 3, 3, 3, 3, 3, 3, 3, 3, 5, 5, 5, 10, 1])
        encoded = torch.cat((self.own(own / scale), self.neighbors(observation["neighbors"]),
                             self.obstacles(observation["obstacles"])), dim=-1)
        return self.trunk(encoded)

    def act_from_hidden(self, hidden):
        return self.actor(hidden)

    def value_local(self, hidden):
        return self.critic(hidden).squeeze(-1)

    def value_central(self, hidden, n_agents):
        pooled = hidden.reshape(-1, n_agents, hidden.size(-1)).mean(1)
        return self.central(pooled).expand(-1, n_agents).reshape(-1)

    def _value(self, hidden, n_agents=None):
        if self.central_critic:
            return self.value_central(hidden, n_agents or hidden.size(0))
        return self.value_local(hidden)

    def forward(self, observation):
        hidden = self.encode(observation)
        return self.act_from_hidden(hidden), self._value(hidden)

    def distribution(self, observation, n_agents=None):
        hidden = self.encode(observation)
        return Normal(self.act_from_hidden(hidden), self.log_std.clamp(-5, 1).exp()), self._value(hidden, n_agents)

    @staticmethod
    def log_probability(distribution, raw):
        correction = 2 * (math.log(2) - raw - F.softplus(-2 * raw))
        return (distribution.log_prob(raw) - correction).sum(-1)

    def sample(self, observation):
        distribution, value = self.distribution(observation)
        raw = distribution.sample()
        return raw.tanh(), raw, self.log_probability(distribution, raw), value

    def evaluate(self, observation, raw, n_agents=None):
        distribution, value = self.distribution(observation, n_agents)
        return self.log_probability(distribution, raw), distribution.entropy().sum(-1), value


class LearnedController:
    def __init__(self, path):
        data = torch.load(Path(path), map_location="cpu", weights_only=True)
        if data.get("format") not in CHECKPOINT_FORMATS:
            raise ValueError("Unsupported checkpoint format")
        self.config = Config(**data["config"])
        training = data.get("training") or {}
        central = bool(data.get("central_critic", training.get("central_critic", False)))
        self.model = ActorCritic(central_critic=central)
        try:
            self.model.load_state_dict(data["model"])
        except RuntimeError as error:
            raise ValueError("Checkpoint weights do not match this policy architecture") from error
        self.model.eval()
        self.metadata = {k: v for k, v in data.items() if k != "model" and k != "optimizer"}

    def __call__(self, observations):
        with torch.no_grad():
            mean, _ = self.model(tensors(stack_observations(observations)))
        return dict(zip(observations, mean.tanh().numpy()))
