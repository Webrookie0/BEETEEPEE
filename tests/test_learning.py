import json
import numpy as np
import torch

from swarmflow import Config, SwarmEnv
from swarmflow.policy import ActorCritic, LearnedController, stack_observations, tensors
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
