import json
import math
import os
import time
from collections import deque
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

from .baseline import PredictiveController
from .config import Config
from .env import SwarmEnv
from .evaluate import aggregate, run_episode, selection_score
from .policy import (CHECKPOINT_FORMAT, CHECKPOINT_FORMATS, OBSERVATION_KEYS, ActorCritic, ModelController,
                     stack_observations, stack_teams, tensors)

# Episode seeds: training rollouts and cloning draw from 0..999, checkpoint selection uses 5000+,
# development validation uses 1000+ and the final held-out test uses 100000+.
TRAINING_SEEDS = (0, 1000)
SELECTION_SEED_START = 5000
HARD_OBJECT_GAP_M = 2.0
HARD_NEIGHBOR_GAP_M = 0.6


def advantages(rewards, values, next_values, terminated, ended, gamma=0.99, lam=0.95):
    """Bootstrap time limits, but never propagate advantage across an episode reset."""
    result = np.zeros_like(rewards)
    carry = np.zeros_like(rewards[0])
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_values[t] * (1 - terminated[t]) - values[t]
        carry = delta + gamma * lam * (1 - ended[t]) * carry
        result[t] = carry
    return result, result + values


class RewardScaler:
    """Divides rewards by a running standard deviation of the discounted return."""

    def __init__(self, gamma, count=1e-4, mean=0.0, var=1.0):
        self.gamma = gamma
        self.count, self.mean, self.var = float(count), float(mean), float(var)
        self.returns = None

    def __call__(self, rewards, ended):
        if self.returns is None or self.returns.shape != rewards.shape:
            self.returns = np.zeros_like(rewards)
        self.returns = self.gamma * self.returns + rewards
        self._update(self.returns)
        self.returns = self.returns * (1.0 - ended)
        return rewards / math.sqrt(self.var + 1e-8)

    def _update(self, values):
        mean, var, n = float(values.mean()), float(values.var()), values.size
        delta, total = mean - self.mean, self.count + n
        self.var = (self.var * self.count + var * n + delta ** 2 * self.count * n / total) / total
        self.mean += delta * n / total
        self.count = total

    def state(self):
        return {"count": self.count, "mean": self.mean, "var": self.var}


class Worker:
    """One environment that starts a new random training episode whenever the current one ends."""

    def __init__(self, config, scenarios, rng, failure_reset):
        self.config, self.scenarios, self.rng, self.failure_reset = config, scenarios, rng, failure_reset
        self.interval = max(1, int(2.0 / config.dt))
        self.ring = deque(maxlen=4)
        self.restores = 0
        self.start_episode()

    def start_episode(self):
        self.env = SwarmEnv(self.config, str(self.rng.choice(self.scenarios)))
        self.observations, _ = self.env.reset(seed=int(self.rng.integers(*TRAINING_SEEDS)))
        self.restores = 0
        self.ring.clear()

    def step(self, actions):
        env = self.env
        final, reward, term, trunc, _ = env.step(dict(zip(self.observations, actions)))
        agents = list(reward)
        result = {"final": final, "done": not env.agents, "collision": bool(env.collision),
                  "restored": False, "summary": None,
                  "reward": np.array([reward[a] for a in agents], dtype=np.float32),
                  "terminated": np.array([term[a] for a in agents], dtype=np.float32),
                  "ended": np.array([term[a] or trunc[a] for a in agents], dtype=np.float32)}
        if env.agents:
            if self.failure_reset and env.steps % self.interval == 0:
                self.ring.append(env.capture_state())
            self.observations = final
        elif self.failure_reset and env.collision and self.ring and self.restores < 2:
            env.restore_state(self.ring[0])
            self.observations = env.last_observations
            self.restores += 1
            self.ring.clear()
            result["restored"] = True
        else:
            result["summary"] = env.summary()
            # A replayed crash still counts: the episode failed before it was given another try.
            result["failed"] = env.failure is not None or self.restores > 0
            self.start_episode()
        return result


def _hard_share(observation, radius):
    """Per labelled state: True when an object is within 2 m or a neighbour within 0.6 m (surface gap)."""
    near = torch.zeros(len(observation["self"]), dtype=torch.bool, device=observation["self"].device)
    for key, limit in (("obstacles", HARD_OBJECT_GAP_M), ("neighbors", HARD_NEIGHBOR_GAP_M)):
        rows = observation[key]
        gap = rows[..., :3].norm(dim=-1) - rows[..., 6] - radius
        near |= ((rows[..., 7] > 0.5) & (gap < limit)).any(-1)
    return near.cpu().numpy()


def _demo_tensors(data, labels, device, radius, hard_weight):
    """Teacher dataset (observations, actions, sampling CDF or None, share of hard states)."""
    observation = tensors({k: np.concatenate(v) for k, v in data.items()}, device)
    target = torch.as_tensor(np.concatenate(labels), dtype=torch.float32, device=device)
    near = _hard_share(observation, radius)
    cdf = None
    if hard_weight != 1.0:
        weights = 1.0 + (hard_weight - 1.0) * near
        cdf = np.cumsum(weights) / weights.sum()
    return observation, target, cdf, float(near.mean())


def _demo_ids(demo, count, rng):
    """Sample demonstration indices; near-obstacle states are drawn hard_weight times more often."""
    cdf = demo[2]
    if cdf is None:
        return rng.integers(0, len(demo[1]), count)
    return np.minimum(np.searchsorted(cdf, rng.random(count)), len(cdf) - 1)


def _fit_clone(model, optimizer, demo, epochs, rng, device, batch_size=1024):
    batch, target = demo[0], demo[1]
    parameters = model.actor_parameters()
    losses = []
    for _ in range(epochs):
        losses = []
        order = rng.permutation(len(target)) if demo[2] is None else _demo_ids(demo, len(target), rng)
        for start in range(0, len(target), batch_size):
            ids = torch.as_tensor(order[start:start + batch_size], device=device)
            loss = F.mse_loss(model.act({k: v[ids] for k, v in batch.items()}), target[ids])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 0.5)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def _collect(model, teacher, config, scenarios, steps, rng, device, data, labels, noise=None):
    """Fly `steps` team steps and record every state with the teacher's action as its label.

    With `noise` the teacher flies with that much action noise (DART); otherwise the student
    flies (DAgger). Returns (episodes finished, failed episodes).
    """
    student = ModelController(model, device)
    env = None
    episodes = failures = 0
    for _ in range(steps):
        if env is None or not env.agents:
            env = SwarmEnv(config, str(rng.choice(scenarios)))
            observations, _ = env.reset(seed=int(rng.integers(*TRAINING_SEEDS)))
        targets = teacher(observations)
        packed = stack_observations(observations)
        for key in data:
            data[key].append(packed[key])
        labels.append(np.stack(list(targets.values())))
        if noise is not None:
            actions = {a: np.clip(v + rng.normal(0, noise, 3), -1, 1) for a, v in targets.items()}
        else:
            actions = student(observations)
        observations, _, _, _, _ = env.step(actions)
        if not env.agents:
            episodes += 1
            failures += int(env.failure is not None)
    return episodes, failures


def clone(model, config, scenarios, steps, epochs, rng, device, rounds=0, noise=0.3, report=None,
          teacher_margins=(0.35, 0.08), hard_weight=1.0):
    """Behavior cloning from the predictive controller, followed by DAgger rounds.

    Round 0 flies the teacher with injected action noise while recording its clean actions, so the
    data already contains states away from the slots. Every later round flies the current student,
    labels each state it visits with the teacher's action and refits on all data collected so far.
    Returns the log rows and the aggregated teacher dataset.
    """
    teacher = PredictiveController(config, *teacher_margins)
    optimizer = torch.optim.Adam(model.actor_parameters(), lr=5e-4)
    data = {key: [] for key in OBSERVATION_KEYS}
    labels = []
    rows = []
    demo = None
    for round_index in range(rounds + 1):
        episodes, failures = _collect(model, teacher, config, scenarios, steps, rng, device, data, labels,
                                      noise if round_index == 0 else None)
        demo = _demo_tensors(data, labels, device, config.radius, hard_weight)
        mse = _fit_clone(model, optimizer, demo, epochs, rng, device)
        rows.append({"type": "behavior_cloning", "round": round_index,
                     "driver": "teacher with action noise" if round_index == 0 else "student (DAgger)",
                     "environment_steps": steps, "agent_samples": len(demo[1]), "epochs": epochs,
                     "driver_episodes_finished": episodes, "driver_failures": failures,
                     "hard_state_share": demo[3], "final_epoch_mse": mse})
        if report:
            report(rows[-1])
    return rows, demo


def demonstrations(model, config, scenarios, steps, rng, device, teacher_margins=(0.35, 0.08), hard_weight=1.0):
    """Teacher-labelled states visited by the current student, without refitting (used by --continue)."""
    data = {key: [] for key in OBSERVATION_KEYS}
    labels = []
    _collect(model, PredictiveController(config, *teacher_margins), config, scenarios, steps, rng, device,
             data, labels)
    return _demo_tensors(data, labels, device, config.radius, hard_weight)


def _clock(seconds):
    seconds = int(max(0, seconds))
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def train(config, scenarios, total_steps=10000, horizon=128, epochs=4, seed=0,
          output="runs/train", clone_steps=0, clone_epochs=15, resume=None, device="cpu",
          critic="local", failure_reset=False, envs=8, separate_critic=True, lr=3e-4, minibatch=1024,
          gamma=0.99, lam=0.95, clip=0.2, entropy=0.0, target_kl=0.03, dagger_rounds=0, clone_noise=0.3,
          init_log_std=None, critic_warmup=None, normalize_rewards=True, eval_every=25000, eval_seeds=5,
          eval_scenarios=None, snapshot_every=0, threads=None, continue_run=False, stage=None,
          width=64, min_log_std=-3.0, bc_coef=1.0, bc_coef_end=0.5, bc_batch=1024, pipeline_version=None,
          teacher_margins=(0.7, 0.3), hard_weight=1.0, gap_features=False):
    """Train the shared drone policy and return the path of latest.pt.

    The output folder receives latest.pt (every update, for resuming), best.pt (highest selection
    score so far), best.json, evaluations.jsonl, training.jsonl, config.json and finished.json.
    With clone_steps the run first clones the predictive controller (DART + DAgger) on its own
    scenarios, also when resuming from an earlier stage, and keeps that teacher data as an
    imitation loss inside PPO whose weight decays from bc_coef to bc_coef_end over the run.
    """
    if (total_steps < 1 or horizon < 2 or epochs < 1 or clone_steps < 0 or clone_epochs < 1 or envs < 1
            or minibatch < 1 or dagger_rounds < 0 or eval_seeds < 1):
        raise ValueError("Invalid training budget")
    if critic not in ("local", "central"):
        raise ValueError("critic must be 'local' or 'central'")
    target = Path(output)
    if continue_run:
        resume = target / "latest.pt"
        if not resume.exists():
            raise ValueError(f"Nothing to continue: {resume} does not exist")
        clone_steps = 0
    same_folder = bool(resume) and Path(resume).resolve().parent == target.resolve()
    if not same_folder and any((target / name).exists() for name in ("latest.pt", "best.pt")):
        raise ValueError(f"{target} already holds a trained model; use --continue to finish it, "
                         "or choose a new --output")
    hyper = {"scenarios": list(scenarios), "horizon": horizon, "epochs": epochs, "envs": envs, "lr": lr,
             "minibatch": minibatch, "gamma": gamma, "lam": lam, "clip": clip, "entropy": entropy,
             "target_kl": target_kl, "normalize_rewards": normalize_rewards, "failure_reset": failure_reset,
             "eval_every": eval_every, "eval_seeds": eval_seeds,
             "eval_scenarios": list(eval_scenarios or scenarios), "snapshot_every": snapshot_every,
             "seed": seed, "stage": stage, "bc_coef": bc_coef, "bc_coef_end": bc_coef_end, "bc_batch": bc_batch,
             "pipeline_version": pipeline_version, "teacher_margins": [float(m) for m in teacher_margins],
             "hard_weight": float(hard_weight)}
    central_critic = critic == "central"
    prior_steps = 0
    checkpoint = None
    previous = {}
    if resume:
        checkpoint = torch.load(resume, map_location=device, weights_only=True)
        if checkpoint.get("format") not in CHECKPOINT_FORMATS:
            raise ValueError("Unsupported checkpoint format")
        if checkpoint["format"] != CHECKPOINT_FORMAT:
            raise ValueError(f"{resume} is a {checkpoint['format']} checkpoint trained with the old reward and "
                             "network; start a fresh run")
        if Config(**checkpoint["config"]).to_dict() != config.to_dict():
            raise ValueError("Resume checkpoint has incompatible environment configuration")
        previous = checkpoint.get("training") or {}
        central_critic = bool(checkpoint.get("central_critic", False))
        separate_critic = bool(checkpoint.get("separate_critic", False))
        prior_steps = int(checkpoint["environment_steps"])
    run_start, target_steps = prior_steps, prior_steps + total_steps
    initialized_from = str(resume) if resume else None
    if continue_run:
        hyper.update({key: previous[key] for key in hyper if key in previous})
        run_start = int(previous.get("stage_start_environment_steps", prior_steps))
        target_steps = int(previous.get("target_environment_steps", prior_steps + total_steps))
        initialized_from = previous.get("initialized_from")
        seed = hyper["seed"]
        total_steps = target_steps - prior_steps
        if total_steps <= 0:
            print(f"{target} already reached {target_steps:,} environment steps; nothing to continue.", flush=True)
            return target / "latest.pt"
    torch.set_num_threads(threads or min(4, os.cpu_count() or 1))
    torch.manual_seed(seed + prior_steps)
    rng = np.random.default_rng([seed, prior_steps])
    target.mkdir(parents=True, exist_ok=True)
    if init_log_std is None:
        init_log_std = -1.6 if clone_steps else -1.0
    if critic_warmup is None:
        critic_warmup = 10 if clone_steps else 5 if resume and not same_folder else 0
    network = {"width": width, "min_log_std": min_log_std, "gap_features": gap_features}
    if checkpoint:
        network = checkpoint.get("network") or {"width": 64, "min_log_std": -5.0}
    model = ActorCritic(central_critic=central_critic, separate_critic=separate_critic,
                        init_log_std=init_log_std, **network).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hyper["lr"], eps=1e-5)
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    scaler = None
    if hyper["normalize_rewards"]:
        scaler = RewardScaler(hyper["gamma"], **((checkpoint or {}).get("reward_scaler") or {}))
    actor_params, critic_params = model.actor_parameters(), model.critic_parameters()
    n_agents, n_envs = config.drones, hyper["envs"]
    eval_scenarios = hyper["eval_scenarios"]
    selection_seeds = range(SELECTION_SEED_START, SELECTION_SEED_START + hyper["eval_seeds"])
    algorithm = ("parameter-sharing PPO; "
                 + ("centralized critic (MAPPO-style value)" if central_critic else "local critic (IPPO)")
                 + ("; separate critic network" if separate_critic else "; critic shares the actor encoder")
                 + "; tanh Gaussian velocity actions"
                 + ("; decaying imitation loss on teacher data" if hyper["bc_coef"] > 0 else ""))
    settings = {**hyper, "config": config.to_dict(), "network": model.network_spec(), "total_steps_this_run": total_steps,
                "stage_start_environment_steps": run_start, "target_environment_steps": target_steps,
                "clone_steps": clone_steps, "clone_epochs": clone_epochs, "dagger_rounds": dagger_rounds,
                "clone_noise": clone_noise, "init_log_std": float(init_log_std),
                "critic_warmup_updates": critic_warmup, "initialized_from": initialized_from,
                "resume": str(resume) if resume else None, "continued": bool(continue_run),
                "algorithm": algorithm, "central_critic": central_critic, "separate_critic": separate_critic,
                "training_scenario_seeds": f"{TRAINING_SEEDS[0]}..{TRAINING_SEEDS[1] - 1}",
                "selection_seeds": f"{selection_seeds.start}..{selection_seeds.stop - 1}",
                "resume_semantics": "weights, optimizer and reward scale; fresh seeded episodes"}
    if continue_run:
        for key in ("clone_steps", "clone_epochs", "dagger_rounds", "clone_noise", "init_log_std",
                    "critic_warmup_updates"):
            settings[key] = previous.get(key, settings[key])
    (target / "config.json").write_text(json.dumps(settings, indent=2) + "\n")
    status = "swarmflow-v4; attention set encoder"
    status += "; separate critic" if separate_critic else ""
    status += "; centralized critic" if central_critic else ""
    status += "; evaluate before claiming success"
    log, evaluations = target / "training.jsonl", target / "evaluations.jsonl"
    best_file = target / "best.json"
    best_score = json.loads(best_file.read_text())["score"] if same_folder and best_file.exists() else -math.inf
    done_steps = 0

    def write(path, row):
        with path.open("a") as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")

    def save(path, extra=None):
        data = {"format": CHECKPOINT_FORMAT, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "config": config.to_dict(), "environment_steps": prior_steps + done_steps, "training": settings,
                "central_critic": central_critic, "separate_critic": separate_critic,
                "network": model.network_spec(),
                "reward_scaler": scaler.state() if scaler else None, "status": status, **(extra or {})}
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(path.name + ".tmp")
        torch.save(data, temp)
        temp.replace(path)

    controller = ModelController(model, device)

    def validate(reason):
        nonlocal best_score
        began = time.monotonic()
        episodes = [run_episode(config, scenario, s, controller)[0] for scenario in eval_scenarios
                    for s in selection_seeds]
        summary = aggregate(episodes)
        score = selection_score(summary)
        rows = list(summary.values())
        row = {"type": "evaluation", "reason": reason, "environment_steps": prior_steps + done_steps,
               "score": score, "best": score > best_score,
               "success_rate": float(np.mean([r["success_rate"] for r in rows])),
               "mean_slot_rmse_m": float(np.mean([r["mean_slot_rmse_m"] for r in rows])),
               "events": sum(r["events"] for r in rows),
               "recovered_events": sum(r["recovered_events"] for r in rows),
               "scenarios": eval_scenarios, "seeds": [selection_seeds.start, selection_seeds.stop - 1],
               "seconds": round(time.monotonic() - began, 1), "summary": summary}
        write(evaluations, row)
        note = ""
        if row["best"]:
            best_score = score
            save(target / "best.pt", {"selection": {key: row[key] for key in
                 ("environment_steps", "score", "success_rate", "mean_slot_rmse_m", "scenarios", "seeds")}})
            best_file.write_text(json.dumps({**row, "checkpoint": "best.pt"}, indent=2, allow_nan=False) + "\n")
            note = "  -> new best, saved best.pt"
        recovered = f"  recovered {row['recovered_events']}/{row['events']}" if row["events"] else ""
        print(f"[eval:{reason}] steps {row['environment_steps']:,}  score {score:.3f} (best {best_score:.3f})  "
              f"success {row['success_rate']:.0%}  slot RMSE {row['mean_slot_rmse_m']:.3f} m{recovered}{note}",
              flush=True)
        return time.monotonic() - began

    workers = []
    demo = None
    try:
        if clone_steps:
            print(f"Behavior cloning on {', '.join(hyper['scenarios'])}: {dagger_rounds + 1} round(s) x "
                  f"{clone_steps:,} steps", flush=True)

            def report(row):
                write(log, row)
                print(f"[clone] round {row['round']} ({row['driver']}): {row['agent_samples']:,} samples "
                      f"({row['hard_state_share']:.0%} near objects), mse {row['final_epoch_mse']:.5f}, "
                      f"driver failures {row['driver_failures']}/{row['driver_episodes_finished']}", flush=True)

            _, demo = clone(model, config, hyper["scenarios"], clone_steps, clone_epochs, rng, device,
                            dagger_rounds, clone_noise, report, hyper["teacher_margins"], hyper["hard_weight"])
            if resume:
                with torch.no_grad():
                    model.log_std.clamp_(min=init_log_std)
        elif continue_run and previous.get("clone_steps") and hyper["bc_coef"] > 0:
            print("Rebuilding the teacher dataset for the imitation loss", flush=True)
            demo = demonstrations(model, config, hyper["scenarios"], int(previous["clone_steps"]), rng, device,
                                  hyper["teacher_margins"], hyper["hard_weight"])
        validate("start")
        last_validated = 0
        workers = [Worker(config, hyper["scenarios"], rng, hyper["failure_reset"]) for _ in range(n_envs)]
        next_eval = hyper["eval_every"] if hyper["eval_every"] > 0 else None
        snapshot_every = hyper["snapshot_every"]
        next_snapshot = (prior_steps // snapshot_every + 1) * snapshot_every if snapshot_every > 0 else None
        started = time.monotonic()
        paused = 0.0
        updates = episodes = collisions = restores = 0
        recent = deque(maxlen=100)
        print(f"PPO: {total_steps:,} environment steps over {n_envs} parallel environments "
              f"({algorithm})", flush=True)
        while done_steps < total_steps:
            steps_now = min(hyper["horizon"], max(1, math.ceil((total_steps - done_steps) / n_envs)))
            obs_store = {key: [] for key in OBSERVATION_KEYS}
            raw_store, logp_store, value_store, reward_store, scaled_store, term_store, end_store = (
                [] for _ in range(7))
            bootstrap = []
            for t in range(steps_now):
                packed = stack_teams([w.observations for w in workers])
                with torch.no_grad():
                    action, raw, logp, value = model.sample(tensors(packed, device), n_agents)
                actions = action.cpu().numpy().reshape(n_envs, n_agents, 3)
                results = [worker.step(actions[e]) for e, worker in enumerate(workers)]
                rewards = np.concatenate([r["reward"] for r in results])
                ended = np.concatenate([r["ended"] for r in results])
                for key in OBSERVATION_KEYS:
                    obs_store[key].append(packed[key])
                raw_store.append(raw.cpu().numpy())
                logp_store.append(logp.cpu().numpy())
                value_store.append(value.cpu().numpy())
                reward_store.append(rewards)
                scaled_store.append(scaler(rewards, ended) if scaler else rewards)
                term_store.append(np.concatenate([r["terminated"] for r in results]))
                end_store.append(ended)
                finals = [(e, r["final"]) for e, r in enumerate(results) if r["done"]]
                if finals:
                    with torch.no_grad():
                        final_values = model.value(tensors(stack_teams([f for _, f in finals]), device),
                                                   n_agents).cpu().numpy()
                    for k, (e, _) in enumerate(finals):
                        bootstrap.append((t, e, final_values[k * n_agents:(k + 1) * n_agents]))
                for r in results:
                    if r["done"]:
                        collisions += int(r["collision"])
                        restores += int(r["restored"])
                    if r["summary"] is not None:
                        episodes += 1
                        recent.append((r["failed"], r["summary"]["slot_rmse_m"]))
                done_steps += n_envs
            with torch.no_grad():
                last_value = model.value(tensors(stack_teams([w.observations for w in workers]), device),
                                         n_agents).cpu().numpy()
            values = np.asarray(value_store, dtype=np.float32)
            next_values = np.empty_like(values)
            next_values[:-1] = values[1:]
            next_values[-1] = last_value
            for t, e, final_value in bootstrap:
                next_values[t, e * n_agents:(e + 1) * n_agents] = final_value
            adv, returns = advantages(np.asarray(scaled_store, dtype=np.float32), values, next_values,
                                      np.asarray(term_store, dtype=np.float32), np.asarray(end_store, dtype=np.float32),
                                      hyper["gamma"], hyper["lam"])
            batch = tensors({k: np.concatenate(v) for k, v in obs_store.items()}, device)
            raw = torch.as_tensor(np.concatenate(raw_store), device=device)
            old_logp = torch.as_tensor(np.concatenate(logp_store), device=device)
            advantage = torch.as_tensor(adv.reshape(-1), device=device)
            advantage = (advantage - advantage.mean()) / advantage.std(unbiased=False).clamp(min=1e-8)
            returns_t = torch.as_tensor(returns.reshape(-1), device=device)
            warmup = updates < critic_warmup
            progress = min(1.0, (prior_steps + done_steps - run_start) / max(1, target_steps - run_start))
            lr_now = hyper["lr"] * (1.0 - 0.9 * progress)
            bc_now = hyper["bc_coef"] + (hyper["bc_coef_end"] - hyper["bc_coef"]) * progress
            for group in optimizer.param_groups:
                group["lr"] = lr_now
            loss_rows = []
            unit = len(advantage) // n_agents if central_critic else len(advantage)
            mb_items = max(1, hyper["minibatch"] // n_agents) if central_critic else hyper["minibatch"]
            epochs_done = 0
            for _ in range(hyper["epochs"]):
                order = rng.permutation(unit)
                kl_values = []
                for start in range(0, unit, mb_items):
                    chunk = order[start:start + mb_items]
                    if central_critic:
                        index = (chunk[:, None] * n_agents + np.arange(n_agents)).reshape(-1)
                    else:
                        index = chunk
                    index = torch.as_tensor(index, device=device)
                    logp, entropy_values, value = model.evaluate({k: v[index] for k, v in batch.items()}, raw[index],
                                                                 n_agents if central_critic else None)
                    log_ratio = logp - old_logp[index]
                    ratio = log_ratio.exp()
                    adv_mb = advantage[index]
                    policy_loss = -torch.minimum(ratio * adv_mb, ratio.clamp(1 - hyper["clip"], 1 + hyper["clip"])
                                                 * adv_mb).mean()
                    value_loss = 0.5 * (value - returns_t[index]).square().mean()
                    bc_loss = torch.zeros((), device=device)
                    if warmup:
                        loss = 0.5 * value_loss
                    else:
                        loss = policy_loss + 0.5 * value_loss - hyper["entropy"] * entropy_values.mean()
                        if demo is not None and bc_now > 0:
                            ids = torch.as_tensor(_demo_ids(demo, hyper["bc_batch"], rng), device=device)
                            bc_loss = F.mse_loss(model.act({k: v[ids] for k, v in demo[0].items()}), demo[1][ids])
                            loss = loss + bc_now * bc_loss
                    if not torch.isfinite(loss):
                        raise RuntimeError("Non-finite PPO loss; training stopped without replacing checkpoint")
                    optimizer.zero_grad()
                    loss.backward()
                    if warmup:
                        for parameter in actor_params:
                            parameter.grad = None
                    if separate_critic:
                        if not warmup:
                            torch.nn.utils.clip_grad_norm_(actor_params, 0.5)
                        torch.nn.utils.clip_grad_norm_(critic_params, 0.5)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    optimizer.step()
                    kl = float(((ratio - 1) - log_ratio).mean().detach().cpu())
                    kl_values.append(kl)
                    loss_rows.append((float(policy_loss.detach().cpu()), float(value_loss.detach().cpu()), kl,
                                      float(entropy_values.mean().detach().cpu()), float(bc_loss.detach().cpu())))
                epochs_done += 1
                if not warmup and np.mean(kl_values) > hyper["target_kl"]:
                    break
            updates += 1
            flat_returns, flat_values = returns.reshape(-1), values.reshape(-1)
            spread = float(np.var(flat_returns))
            explained = 1.0 - float(np.var(flat_returns - flat_values)) / spread if spread > 1e-12 else None
            losses = np.mean(loss_rows, axis=0)
            elapsed = max(time.monotonic() - started - paused, 1e-9)
            rate = done_steps / elapsed
            row = {"type": "ppo", "phase": "critic_warmup" if warmup else "ppo", "update": updates,
                   "environment_steps": prior_steps + done_steps,
                   "agent_transitions": (prior_steps + done_steps) * n_agents,
                   "episodes_this_run": episodes, "collisions_this_run": collisions,
                   "failure_resets_this_run": restores,
                   "recent_failure_rate": float(np.mean([f for f, _ in recent])) if recent else None,
                   "recent_slot_rmse_m": float(np.mean([s for _, s in recent])) if recent else None,
                   "mean_step_reward": float(np.mean(reward_store)), "policy_loss": float(losses[0]),
                   "value_loss": float(losses[1]), "approx_kl": float(losses[2]), "entropy": float(losses[3]),
                   "bc_loss": float(losses[4]), "bc_coef": bc_now if demo is not None else 0.0,
                   "explained_variance": explained, "epochs_done": epochs_done, "lr": lr_now,
                   "action_std": float(model.log_std.detach().clamp(model.min_log_std, 1).exp().mean()),
                   "environment_steps_per_second": rate}
            write(log, row)
            failure = "-" if row["recent_failure_rate"] is None else f"{row['recent_failure_rate']:.0%}"
            rmse = "-" if row["recent_slot_rmse_m"] is None else f"{row['recent_slot_rmse_m']:.2f} m"
            imitation = f"  imitation {row['bc_loss']:.4f}" if demo is not None and not warmup else ""
            print(f"[{row['phase']}] {done_steps:,}/{total_steps:,} ({done_steps / total_steps:.1%})  "
                  f"reward/step {row['mean_step_reward']:+.4f}  failures(last 100) {failure}  "
                  f"slot RMSE(last 100) {rmse}  kl {row['approx_kl']:.4f}  value loss {row['value_loss']:.3f}"
                  f"{imitation}  std {row['action_std']:.3f}  {rate:,.0f} steps/s  "
                  f"ETA {_clock((total_steps - done_steps) / rate)}", flush=True)
            save(target / "latest.pt")
            if next_snapshot is not None and prior_steps + done_steps >= next_snapshot:
                save(target / "checkpoints" / f"step_{prior_steps + done_steps:09d}.pt")
                next_snapshot += snapshot_every
            if next_eval is not None and done_steps >= next_eval:
                paused += validate("periodic")
                last_validated = done_steps
                next_eval += hyper["eval_every"]
        if last_validated != done_steps:
            validate("final")
        (target / "finished.json").write_text(json.dumps({
            "environment_steps": prior_steps + done_steps, "target_environment_steps": target_steps,
            "best_score": best_score,
            "best_environment_steps": json.loads(best_file.read_text())["environment_steps"],
            "updates_this_run": updates, "training_seconds_this_run": round(time.monotonic() - started, 1)},
            indent=2) + "\n")
    except KeyboardInterrupt:
        print(f"\nInterrupted. {target / 'latest.pt'} holds the last completed update and best.pt the best "
              "validated one; finish the run with --continue.", flush=True)
        raise
    finally:
        for worker in workers:
            worker.env.close()
    return target / "latest.pt"
