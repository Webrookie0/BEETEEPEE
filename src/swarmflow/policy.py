import math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F

from .config import Config

CHECKPOINT_FORMATS = ("swarmflow-v1", "swarmflow-v2", "swarmflow-v3", "swarmflow-v4")
CHECKPOINT_FORMAT = "swarmflow-v4"
OBSERVATION_KEYS = ("self", "neighbors", "obstacles")
DRONE_RADIUS_M = Config().radius


def stack_observations(observations):
    return {key: np.stack([o[key] for o in observations.values()]) for key in OBSERVATION_KEYS}


def stack_teams(teams):
    """Stack several environments' observation dicts; rows stay grouped by environment, then drone."""
    return {key: np.stack([o[key] for team in teams for o in team.values()]) for key in OBSERVATION_KEYS}


def tensors(observations, device="cpu"):
    return {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in observations.items()}


class SetEncoder(nn.Module):
    """Masked attention pooling over neighbour or object rows.

    With gap_features each row also gets its surface gap, a proximity term and its closing speed,
    so decisions that switch at a few tens of centimetres of clearance do not have to be derived
    from centre positions scaled by the 7 m sensor range.
    """

    def __init__(self, width=64, gap_features=False):
        super().__init__()
        self.gap_features = gap_features
        self.net = nn.Sequential(nn.Linear(10 if gap_features else 7, width // 2), nn.Tanh(),
                                 nn.Linear(width // 2, width), nn.Tanh())
        self.query = nn.Parameter(torch.zeros(1, 1, width))
        self.attn = nn.MultiheadAttention(width, max(2, width // 32), batch_first=True)

    def forward(self, rows):
        mask = rows[..., 7]
        features = rows[..., :7] / rows.new_tensor([7, 7, 7, 6, 6, 6, 1.5])
        if self.gap_features:
            relative, velocity = rows[..., :3], rows[..., 3:6]
            distance = relative.norm(dim=-1).clamp(min=1e-6)
            gap = distance - rows[..., 6] - DRONE_RADIUS_M
            closing = -(relative * velocity).sum(-1) / distance
            features = torch.cat((features, torch.stack(
                (gap / 2.0, torch.exp(-gap.clamp(min=0.0) / 0.3), closing / 3.0), dim=-1)), dim=-1)
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


class Encoder(nn.Module):
    """Self MLP plus masked attention pooling over neighbors and obstacles, reduced to `width` features."""

    def __init__(self, width=64, gap_features=False):
        super().__init__()
        self.own = nn.Sequential(nn.Linear(17, width), nn.Tanh())
        self.neighbors = SetEncoder(width, gap_features)
        self.obstacles = SetEncoder(width, gap_features)
        self.trunk = nn.Sequential(nn.Linear(3 * width, 2 * width), nn.Tanh(), nn.Linear(2 * width, width), nn.Tanh())

    def encode(self, observation):
        own = observation["self"]
        scale = own.new_tensor([5, 5, 5, 3, 3, 3, 3, 3, 3, 3, 3, 3, 5, 5, 5, 10, 1])
        encoded = torch.cat((self.own(own / scale), self.neighbors(observation["neighbors"]),
                             self.obstacles(observation["obstacles"])), dim=-1)
        return self.trunk(encoded)


class ActorCritic(Encoder):
    """Shared per-drone policy. With separate_critic the value function has its own encoder,
    so value-loss gradients cannot overwrite the features the actor relies on."""

    def __init__(self, central_critic=False, separate_critic=False, init_log_std=-1.0, width=64,
                 min_log_std=-5.0, gap_features=False):
        super().__init__(width, gap_features)
        self.central_critic = central_critic
        self.separate_critic = separate_critic
        self.width = width
        self.min_log_std = min_log_std
        self.gap_features = gap_features
        self.actor = nn.Linear(width, 3)
        self.critic = nn.Linear(width, 1)
        if central_critic:
            self.central = nn.Sequential(nn.Linear(width, width), nn.Tanh(), nn.Linear(width, 1))
        self.log_std = nn.Parameter(torch.full((3,), float(init_log_std)))
        self.critic_encoder = Encoder(width, gap_features) if separate_critic else None
        nn.init.orthogonal_(self.actor.weight, 0.01)
        nn.init.zeros_(self.actor.bias)

    def network_spec(self):
        return {"width": self.width, "min_log_std": float(self.min_log_std), "gap_features": self.gap_features}

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

    def _critic_hidden(self, observation, actor_hidden):
        return self.critic_encoder.encode(observation) if self.separate_critic else actor_hidden

    def forward(self, observation, n_agents=None):
        hidden = self.encode(observation)
        return self.act_from_hidden(hidden), self._value(self._critic_hidden(observation, hidden), n_agents)

    def act(self, observation):
        """Deterministic action in [-1, 1]: the squashed mean, as used for evaluation and viewing."""
        return self.act_from_hidden(self.encode(observation)).tanh()

    def value(self, observation, n_agents=None):
        hidden = self.critic_encoder.encode(observation) if self.separate_critic else self.encode(observation)
        return self._value(hidden, n_agents)

    def critic_parameters(self):
        modules = [self.critic]
        if self.separate_critic:
            modules.append(self.critic_encoder)
        if self.central_critic:
            modules.append(self.central)
        return [p for module in modules for p in module.parameters()]

    def actor_parameters(self):
        critic = {id(p) for p in self.critic_parameters()}
        return [p for p in self.parameters() if id(p) not in critic]

    def distribution(self, observation, n_agents=None):
        hidden = self.encode(observation)
        value = self._value(self._critic_hidden(observation, hidden), n_agents)
        return Normal(self.act_from_hidden(hidden), self.log_std.clamp(self.min_log_std, 1).exp()), value

    @staticmethod
    def log_probability(distribution, raw):
        correction = 2 * (math.log(2) - raw - F.softplus(-2 * raw))
        return (distribution.log_prob(raw) - correction).sum(-1)

    def sample(self, observation, n_agents=None):
        distribution, value = self.distribution(observation, n_agents)
        raw = distribution.sample()
        return raw.tanh(), raw, self.log_probability(distribution, raw), value

    def evaluate(self, observation, raw, n_agents=None):
        distribution, value = self.distribution(observation, n_agents)
        return self.log_probability(distribution, raw), distribution.entropy().sum(-1), value


class ModelController:
    """Drives a team with the deterministic policy of an in-memory model."""

    def __init__(self, model, device="cpu"):
        self.model = model
        self.device = device

    def __call__(self, observations):
        with torch.no_grad():
            action = self.model.act(tensors(stack_observations(observations), self.device))
        return dict(zip(observations, action.cpu().numpy()))


class LearnedController:
    def __init__(self, path):
        data = torch.load(Path(path), map_location="cpu", weights_only=True)
        if data.get("format") not in CHECKPOINT_FORMATS:
            raise ValueError("Unsupported checkpoint format")
        self.config = Config(**data["config"])
        training = data.get("training") or {}
        central = bool(data.get("central_critic", training.get("central_critic", False)))
        separate = bool(data.get("separate_critic", False))
        self.model = ActorCritic(central_critic=central, separate_critic=separate, **(data.get("network") or {}))
        try:
            self.model.load_state_dict(data["model"])
        except RuntimeError as error:
            raise ValueError("Checkpoint weights do not match this policy architecture") from error
        self.model.eval()
        self.metadata = {k: v for k, v in data.items() if k != "model" and k != "optimizer"}
        self.controller = ModelController(self.model)

    def __call__(self, observations):
        return self.controller(observations)
