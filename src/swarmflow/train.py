import json
import time
from collections import deque
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

from .baseline import PredictiveController
from .env import SwarmEnv
from .policy import CHECKPOINT_FORMAT, CHECKPOINT_FORMATS, ActorCritic, stack_observations, tensors


def advantages(rewards, values, next_values, terminated, ended, gamma=0.99, lam=0.95):
    """Bootstrap time limits, but never propagate advantage across an episode reset."""
    result = np.zeros_like(rewards)
    carry = np.zeros_like(rewards[0])
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_values[t] * (1 - terminated[t]) - values[t]
        carry = delta + gamma * lam * (1 - ended[t]) * carry
        result[t] = carry
    return result, result + values


def clone(model, config, scenarios, steps, epochs, rng, device):
    teacher = PredictiveController(config)
    data = {k: [] for k in ("self", "neighbors", "obstacles")}
    labels = []
    env = None
    for step in range(steps):
        if env is None or not env.agents:
            env = SwarmEnv(config, str(rng.choice(scenarios)))
            observations, _ = env.reset(seed=int(rng.integers(0, 1000)))
        packed = stack_observations(observations)
        actions = teacher(observations)
        for key in data:
            data[key].append(packed[key])
        labels.append(np.stack(list(actions.values())))
        observations, _, _, _, _ = env.step(actions)
    batch = tensors({k: np.concatenate(v) for k, v in data.items()}, device)
    labels = torch.as_tensor(np.concatenate(labels), dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    loss = None
    for _ in range(epochs):
        order = rng.permutation(len(labels))
        for start in range(0, len(labels), 256):
            ids = torch.as_tensor(order[start:start + 256], device=device)
            mean, _ = model({k: v[ids] for k, v in batch.items()})
            loss = F.mse_loss(mean.tanh(), labels[ids])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
    env.close()
    return {"type": "behavior_cloning", "environment_steps": steps, "agent_samples": len(labels),
            "epochs": epochs, "final_minibatch_mse": float(loss.detach().cpu())}


def train(config, scenarios, total_steps=10000, horizon=128, epochs=4, seed=0,
          output="runs/train", clone_steps=0, clone_epochs=15, resume=None, device="cpu",
          critic="local", failure_reset=False):
    if total_steps < 1 or horizon < 2 or epochs < 1 or clone_steps < 0 or clone_epochs < 1:
        raise ValueError("Invalid training budget")
    if critic not in ("local", "central"):
        raise ValueError("critic must be 'local' or 'central'")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    target = Path(output)
    if (target / "latest.pt").exists() and resume is None:
        raise ValueError(f"{target}/latest.pt already exists; choose a new --output or use --resume")
    target.mkdir(parents=True, exist_ok=True)
    central_critic = critic == "central"
    prior_steps = 0
    checkpoint = None
    if resume:
        checkpoint = torch.load(resume, map_location=device, weights_only=True)
        if checkpoint.get("format") not in CHECKPOINT_FORMATS or checkpoint["config"] != config.to_dict():
            raise ValueError("Resume checkpoint has incompatible environment configuration")
        training = checkpoint.get("training") or {}
        central_critic = bool(checkpoint.get("central_critic", training.get("central_critic", False)))
        prior_steps = checkpoint["environment_steps"]
        if clone_steps:
            raise ValueError("Do not repeat behavior cloning when resuming PPO")
    model = ActorCritic(central_critic=central_critic).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, eps=1e-5)
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    log = target / "training.jsonl"
    algorithm = ("parameter-sharing PPO; centralized critic (MAPPO-style value)"
                 if central_critic else
                 "parameter-sharing IPPO; local critic; tanh Gaussian velocity actions")
    settings = {"seed": seed, "scenarios": list(scenarios), "config": config.to_dict(),
                "total_steps_this_run": total_steps, "horizon": horizon, "epochs": epochs,
                "clone_steps": clone_steps, "clone_epochs": clone_epochs, "resume": str(resume) if resume else None,
                "algorithm": algorithm, "central_critic": central_critic, "failure_reset": failure_reset}
    settings["training_scenario_seeds"] = "0..999"
    settings["resume_semantics"] = "weights and optimizer; fresh seeded episode"
    (target / "config.json").write_text(json.dumps(settings, indent=2) + "\n")

    def record(row):
        with log.open("a") as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
        print(json.dumps(row), flush=True)

    if clone_steps:
        record(clone(model, config, scenarios, clone_steps, clone_epochs, rng, device))
    started = time.monotonic()
    env = None
    done_steps = 0
    episode_count = 0
    collision_count = 0
    failure_resets_this_run = 0
    n_agents = config.drones
    reset_interval = max(1, int(2.0 / config.dt))
    ring = deque(maxlen=4)
    restores_this_episode = 0
    while done_steps < total_steps:
        observations_store = {k: [] for k in ("self", "neighbors", "obstacles")}
        raw_store, log_store, rewards_store, values_store, next_store, term_store, end_store = ([] for _ in range(7))
        for _ in range(min(horizon, total_steps - done_steps)):
            if env is None or not env.agents:
                env = SwarmEnv(config, str(rng.choice(scenarios)))
                observations, _ = env.reset(seed=int(rng.integers(0, 1000)))
                restores_this_episode = 0
                ring.clear()
            packed = stack_observations(observations)
            with torch.no_grad():
                action, raw, logp, value = model.sample(tensors(packed, device))
            next_obs, reward, term, trunc, _ = env.step(dict(zip(observations, action.cpu().numpy())))
            with torch.no_grad():
                _, next_value = model(tensors(stack_observations(next_obs), device))
            for key in packed:
                observations_store[key].append(packed[key])
            raw_store.append(raw.cpu().numpy())
            log_store.append(logp.cpu().numpy())
            rewards_store.append(list(reward.values()))
            values_store.append(value.cpu().numpy())
            next_store.append(next_value.cpu().numpy())
            term_store.append(list(term.values()))
            end_store.append([term[a] or trunc[a] for a in reward])
            observations = next_obs
            done_steps += 1
            if failure_reset and env.agents and env.steps % reset_interval == 0:
                ring.append(env.capture_state())
            if not env.agents:
                collision_count += int(env.collision)
                if (failure_reset and env.collision and ring and restores_this_episode < 2
                        and hasattr(env, "restore_state")):
                    env.restore_state(ring[0])
                    observations = env.last_observations
                    restores_this_episode += 1
                    failure_resets_this_run += 1
                    ring.clear()
                else:
                    episode_count += 1
                    restores_this_episode = 0
                    ring.clear()
        values = np.asarray(values_store, dtype=np.float32)
        adv, returns = advantages(np.asarray(rewards_store, dtype=np.float32), values,
                                 np.asarray(next_store, dtype=np.float32), np.asarray(term_store, dtype=np.float32),
                                 np.asarray(end_store, dtype=np.float32))
        batch = tensors({k: np.concatenate(v) for k, v in observations_store.items()}, device)
        raw = torch.as_tensor(np.concatenate(raw_store), device=device)
        old_logp = torch.as_tensor(np.concatenate(log_store), device=device)
        advantage = torch.as_tensor(adv.reshape(-1), device=device)
        advantage = (advantage - advantage.mean()) / advantage.std(unbiased=False).clamp(min=1e-8)
        returns = torch.as_tensor(returns.reshape(-1), device=device)
        loss_rows = []
        mb_items = max(1, 512 // n_agents) if central_critic else 512
        unit = len(advantage) // n_agents if central_critic else len(advantage)
        for epoch in range(epochs):
            order = rng.permutation(unit)
            kl_values = []
            for start in range(0, unit, mb_items):
                chunk = order[start:start + mb_items]
                if central_critic:
                    index = (chunk[:, None] * n_agents + np.arange(n_agents)).reshape(-1)
                else:
                    index = chunk
                index = torch.as_tensor(index, device=device)
                logp, entropy, value = model.evaluate({k: v[index] for k, v in batch.items()}, raw[index],
                                                      n_agents if central_critic else None)
                log_ratio = logp - old_logp[index]
                ratio = log_ratio.exp()
                policy_loss = -torch.minimum(ratio * advantage[index], ratio.clamp(0.8, 1.2) * advantage[index]).mean()
                value_loss = 0.5 * (value - returns[index]).square().mean()
                loss = policy_loss + 0.5 * value_loss - 0.001 * entropy.mean()
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite PPO loss; training stopped without replacing checkpoint")
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                kl = float(((ratio - 1) - log_ratio).mean().detach().cpu())
                kl_values.append(kl)
                loss_rows.append((float(policy_loss.detach().cpu()), float(value_loss.detach().cpu()), kl))
            if np.mean(kl_values) > 0.03:
                break
        elapsed = time.monotonic() - started
        record({"type": "ppo", "environment_steps": prior_steps + done_steps,
                "agent_transitions": (prior_steps + done_steps) * config.drones,
                "episodes_this_run": episode_count, "collisions_this_run": collision_count,
                "failure_resets_this_run": failure_resets_this_run,
                "mean_step_reward": float(np.mean(rewards_store)), "policy_loss": float(np.mean(loss_rows, axis=0)[0]),
                "value_loss": float(np.mean(loss_rows, axis=0)[1]), "approx_kl": float(np.mean(loss_rows, axis=0)[2]),
                "environment_steps_per_second": done_steps / max(elapsed, 1e-9)})
        status = "swarmflow-v2; attention set encoder"
        if central_critic:
            status += "; centralized critic"
        status += "; evaluate before claiming success"
        checkpoint = {"format": CHECKPOINT_FORMAT, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                      "config": config.to_dict(), "environment_steps": prior_steps + done_steps,
                      "training": settings, "central_critic": central_critic, "status": status}
        temp = target / "latest.tmp"
        torch.save(checkpoint, temp)
        temp.replace(target / "latest.pt")
    env.close()
    return target / "latest.pt"
