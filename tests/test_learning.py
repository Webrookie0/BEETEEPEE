import json
import numpy as np
import torch

from swarmflow import Config, SwarmEnv
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
    assert resumed.metadata["format"] == "swarmflow-v2"
    assert resumed.metadata["central_critic"] is False


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
