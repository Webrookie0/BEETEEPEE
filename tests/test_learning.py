import json
import numpy as np
import torch

from swarmflow import Config, SwarmEnv
from swarmflow.env import FAILURE_PENALTY
from swarmflow.scenarios import Obstacle
from swarmflow.policy import ActorCritic, LearnedController, SetEncoder, stack_observations, tensors
from swarmflow.train import advantages, train


def test_gae_bootstraps_truncation_without_crossing_reset():
    rewards = np.array([[1.], [100.]], dtype=np.float32)
    values = np.zeros_like(rewards)
    next_values = np.array([[2.], [0.]], dtype=np.float32)
    ended = np.ones_like(rewards)
    adv, _ = advantages(rewards, values, next_values, np.zeros_like(rewards), ended, gamma=.9)
    np.testing.assert_allclose(adv[0], 2.8)
    terminal, _ = advantages(rewards, values, next_values, np.ones_like(rewards), ended, gamma=.9)
    np.testing.assert_allclose(terminal[0], 1)


def test_policy_is_invariant_to_object_order_and_ignores_padding():
    torch.set_num_threads(1)
    torch.manual_seed(1)
    env = SwarmEnv()
    observations, _ = env.reset(seed=0)
    env.inject()
    batch = tensors(stack_observations(env.last_observations))
    model = ActorCritic().eval()
    with torch.no_grad():
        reference, _ = model(batch)
        permuted = {key: value.clone() for key, value in batch.items()}
        for key in ("neighbors", "obstacles"):
            permuted[key] = permuted[key].flip(1)
            masked = permuted[key][..., -1] == 0
            permuted[key][..., :7][masked] = 1000
        actual, _ = model(permuted)
        torch.testing.assert_close(reference, actual)
        action, raw, logp, _ = model.sample(batch)
        repeated, _, _ = model.evaluate(batch, raw)
        torch.testing.assert_close(logp, repeated)
        assert torch.isfinite(logp).all() and torch.all(action.abs() <= 1)


def test_training_checkpoint_reload_and_resume(tmp_path):
    config = Config(drones=4, duration=.2)
    path = train(config, ["hold", "follow"], total_steps=16, horizon=8, epochs=1,
                 output=tmp_path, clone_steps=4, clone_epochs=1)
    controller = LearnedController(path)
    env = SwarmEnv(config)
    observations, _ = env.reset(seed=333)
    actions = controller(observations)
    env.step(actions)
    assert set(actions) == set(observations)
    before = {key: value.clone() for key, value in controller.model.state_dict().items()}
    train(config, ["hold"], total_steps=8, horizon=8, epochs=1, output=tmp_path, resume=path)
    resumed = LearnedController(path)
    assert resumed.metadata["environment_steps"] == 24
    assert any(not torch.equal(before[k], v) for k, v in resumed.model.state_dict().items())
    rows = [json.loads(row) for row in (tmp_path / "training.jsonl").read_text().splitlines()]
    assert rows[0]["type"] == "behavior_cloning"
    assert all(np.isfinite(row["value_loss"]) for row in rows if row["type"] == "ppo")
    assert all(row["failure_resets_this_run"] == 0 for row in rows if row["type"] == "ppo")
    assert resumed.metadata["format"] == "swarmflow-v4"
    assert resumed.metadata["central_critic"] is False
    assert resumed.metadata["separate_critic"] is True


def test_training_keeps_best_checkpoint_and_validation_log(tmp_path):
    config = Config(drones=4, duration=.2)
    train(config, ["hold"], total_steps=16, horizon=8, epochs=1, output=tmp_path, clone_steps=4, clone_epochs=1,
          dagger_rounds=1, eval_seeds=2, envs=2)
    evaluations = [json.loads(row) for row in (tmp_path / "evaluations.jsonl").read_text().splitlines()]
    assert [row["reason"] for row in evaluations] == ["start", "final"]
    best = json.loads((tmp_path / "best.json").read_text())
    assert best["score"] == max(row["score"] for row in evaluations)
    assert best["seeds"] == [5000, 5001]
    selected = LearnedController(tmp_path / "best.pt")
    assert selected.metadata["selection"]["score"] == best["score"]
    assert json.loads((tmp_path / "finished.json").read_text())["environment_steps"] == 16
    rows = [json.loads(row) for row in (tmp_path / "training.jsonl").read_text().splitlines()]
    assert [row["driver"] for row in rows if row["type"] == "behavior_cloning"] == [
        "teacher with action noise", "student (DAgger)"]


def test_continue_finishes_the_original_budget(tmp_path):
    config = Config(drones=4, duration=.2)
    latest = train(config, ["hold"], total_steps=8, horizon=8, epochs=1, output=tmp_path, envs=1)
    data = torch.load(latest, weights_only=True)
    data["training"]["target_environment_steps"] = 24
    torch.save(data, latest)
    train(config, ["follow"], total_steps=1, output=tmp_path, continue_run=True)
    resumed = LearnedController(latest)
    assert resumed.metadata["environment_steps"] == 24
    assert resumed.metadata["training"]["scenarios"] == ["hold"]


def test_resumed_stage_clones_its_scenarios_and_keeps_an_imitation_loss(tmp_path):
    config = Config(drones=4, duration=.2)
    first = train(config, ["hold"], total_steps=8, horizon=8, epochs=1, output=tmp_path / "a", envs=1,
                  clone_steps=4, clone_epochs=1, eval_seeds=1, width=32, init_log_std=-3.0)
    train(config, ["hold", "crossing"], total_steps=16, horizon=8, epochs=1, output=tmp_path / "b", envs=1,
          resume=first.with_name("best.pt"), clone_steps=4, clone_epochs=1, dagger_rounds=1, eval_seeds=1,
          critic_warmup=0, init_log_std=-1.6)
    rows = [json.loads(row) for row in (tmp_path / "b" / "training.jsonl").read_text().splitlines()]
    assert [row["driver"] for row in rows if row["type"] == "behavior_cloning"] == [
        "teacher with action noise", "student (DAgger)"]
    ppo = [row for row in rows if row["type"] == "ppo"]
    assert all(row["bc_coef"] > 0 and row["bc_loss"] >= 0 for row in ppo)
    assert ppo[0]["action_std"] > 0.19
    resumed = LearnedController(tmp_path / "b" / "latest.pt")
    assert resumed.metadata["network"] == {"width": 32, "min_log_std": -3.0}
    assert resumed.model.width == 32


def test_continue_rebuilds_teacher_data_for_the_imitation_loss(tmp_path):
    config = Config(drones=4, duration=.2)
    latest = train(config, ["hold"], total_steps=8, horizon=8, epochs=1, output=tmp_path, envs=1,
                   clone_steps=4, clone_epochs=1, eval_seeds=1, critic_warmup=0)
    data = torch.load(latest, weights_only=True)
    data["training"]["target_environment_steps"] = 16
    torch.save(data, latest)
    train(config, ["hold"], total_steps=1, output=tmp_path, continue_run=True)
    rows = [json.loads(row) for row in (tmp_path / "training.jsonl").read_text().splitlines()]
    assert rows[-1]["type"] == "ppo" and rows[-1]["environment_steps"] == 16 and rows[-1]["bc_coef"] > 0


def test_crash_penalty_goes_to_the_drones_involved():
    env = SwarmEnv(Config(drones=3), "hold")
    env.reset(seed=1)
    env.positions[1] = env.positions[0] + np.array([0.2, 0, 0])
    _, reward, terminated, *_ = env.step({a: np.zeros(3) for a in env.agents})
    assert all(terminated.values()) and env.failure == "collision"
    assert reward["drone_0"] < -FAILURE_PENALTY + 1 and reward["drone_1"] < -FAILURE_PENALTY + 1
    assert reward["drone_2"] > -1


def test_obstacle_warning_starts_one_metre_away():
    env = SwarmEnv(Config(drones=2), "hold")
    env.reset(seed=3)
    near = env.positions[0] + np.array([0.5 + 0.3 + env.config.radius, 0, 0])
    env.obstacle_list = [Obstacle("near", near, np.zeros(3), 0.3)]
    _, reward, terminated, *_ = env.step({a: np.zeros(3) for a in env.agents})
    assert not any(terminated.values())
    assert reward["drone_0"] < reward["drone_1"] - 0.02


def test_separate_critic_does_not_touch_actor_weights():
    env = SwarmEnv(Config(drones=4, duration=.2), "hold")
    observations, _ = env.reset(seed=2)
    model = ActorCritic(separate_critic=True)
    model.value(tensors(stack_observations(observations))).sum().backward()
    assert all(p.grad is None for p in model.actor_parameters())
    assert any(p.grad is not None for p in model.critic_parameters())
    ids = {id(p) for p in model.actor_parameters()}
    assert not ids & {id(p) for p in model.critic_parameters()}


def test_reward_is_bounded_and_failure_is_always_worse():
    env = SwarmEnv(Config(drones=4), "hold")
    env.reset(seed=3)
    env.positions = env.slots() + np.array([30., 0, 0])
    _, reward, *_ = env.step({a: np.zeros(3) for a in env.agents})
    worst_alive = min(reward.values())
    assert worst_alive > -0.05 * 15
    discounted_alive = worst_alive / (1 - 0.99)
    assert discounted_alive > -FAILURE_PENALTY


def test_attention_encoder_permutation_invariance():
    torch.manual_seed(0)
    encoder = SetEncoder().eval()
    rows = torch.zeros(3, 5, 8)
    rows[0, 0, :7] = torch.tensor([1., 0, 0, 0.2, 0, 0, 0.18])
    rows[0, 0, 7] = 1
    rows[0, 3, :7] = torch.tensor([0., 1.5, 0, 0, -0.1, 0, 0.18])
    rows[0, 3, 7] = 1
    rows[1, 2, :7] = torch.tensor([2., 0, 0.5, 0, 0, 0, 0.18])
    rows[1, 2, 7] = 1
    shuffled = rows[:, torch.randperm(5)]
    shuffled = shuffled.clone()
    padded = shuffled[..., 7] == 0
    shuffled[..., :7][padded] = 50
    with torch.no_grad():
        torch.testing.assert_close(encoder(rows), encoder(shuffled), atol=1e-5, rtol=1e-5)


def test_actor_critic_forward_and_central_value_shapes():
    torch.manual_seed(0)
    env = SwarmEnv(Config(drones=4, duration=.2), "hold")
    observations, _ = env.reset(seed=1)
    batch = tensors(stack_observations(observations))
    local = ActorCritic().eval()
    central = ActorCritic(central_critic=True).eval()
    with torch.no_grad():
        mean, value = local(batch)
        assert mean.shape == (4, 3) and value.shape == (4,)
        hidden = central.encode(batch)
        assert hidden.shape == (4, 64)
        mean_c, value_c = central(batch)
        assert mean_c.shape == (4, 3) and value_c.shape == (4,)
        torch.testing.assert_close(central.act_from_hidden(hidden), mean_c)
        torch.testing.assert_close(central.value_central(hidden, 4), value_c)
        torch.testing.assert_close(value_c, value_c[:1].expand_as(value_c))
        stacked = {k: torch.cat((v, v), 0) for k, v in batch.items()}
        _, _, two_step = central.evaluate(stacked, torch.zeros(8, 3), n_agents=4)
        assert two_step.shape == (8,)
        torch.testing.assert_close(two_step[:4], two_step[4:])


def test_restore_state_returns_to_prior_time_and_positions():
    env = SwarmEnv(Config(drones=4, duration=1), "hold")
    if not hasattr(env, "restore_state") or not hasattr(env, "capture_state"):
        return
    env.reset(seed=4)
    env.step({a: np.zeros(3) for a in env.agents})
    state = env.capture_state()
    t0, p0 = env.time, env.positions.copy()
    env.step({a: np.zeros(3) for a in env.agents})
    assert env.time != t0
    env.restore_state(state)
    assert env.time == t0
    np.testing.assert_array_equal(env.positions, p0)
    assert env.agents and not env.collision and env.failure is None
