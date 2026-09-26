from dataclasses import replace
import numpy as np
import pytest
from pettingzoo.test import parallel_api_test

from swarmflow import Config, SwarmEnv
from swarmflow.baseline import PredictiveController, SlotController
from swarmflow.evaluate import run_episode
from swarmflow.geometry import swept_distance
from swarmflow.scenarios import Obstacle


def test_parallel_api():
    parallel_api_test(SwarmEnv(Config(drones=4, duration=1)), num_cycles=30)


def test_seed_and_slot_identity():
    a, b = SwarmEnv(), SwarmEnv()
    obs_a, _ = a.reset(seed=22)
    obs_b, _ = b.reset(seed=22)
    slots = a.offsets.copy()
    for agent in obs_a:
        for key in obs_a[agent]:
            np.testing.assert_array_equal(obs_a[agent][key], obs_b[agent][key])
    controller = PredictiveController(a.config)
    for _ in range(20):
        actions = controller(obs_a)
        obs_a, *_ = a.step(actions)
        obs_b, *_ = b.step(actions)
    np.testing.assert_allclose(a.positions, b.positions)
    np.testing.assert_array_equal(a.offsets, slots)
    relative = a.slots() - a.target()[0]
    np.testing.assert_allclose(relative, slots - [a.config.stand_off, 0, 0])


def test_swept_collision_catches_object_between_frames():
    np.testing.assert_allclose(swept_distance(np.array([[-2., 0, 0]]), np.array([[2., 0, 0]])), 0)
    env = SwarmEnv(Config(drones=2), "hold")
    env.reset(seed=3)
    p = env.positions[0].copy()
    env.obstacle_list = [Obstacle("fast", p + [-2, 0, 0], np.array([400., 0, 0]), .2)]
    _, _, term, trunc, _ = env.step({a: np.zeros(3) for a in env.agents})
    assert all(term.values()) and not any(trunc.values())
    assert env.collision and not env.agents
    assert env.min_clearance < 0


def test_acceleration_limit_and_action_validation():
    env = SwarmEnv(Config(drones=2), "hold")
    env.reset(seed=3)
    before = env.positions.copy()
    with pytest.raises(ValueError):
        env.step({a: np.full(3, np.nan) for a in env.agents})
    np.testing.assert_array_equal(env.positions, before)
    env.step({a: np.array([1., 0, 0]) for a in env.agents})
    assert np.linalg.norm(env.velocities, axis=1).max() <= env.config.max_accel * env.config.dt + 1e-9
    assert np.linalg.norm(env.accelerations, axis=1).max() <= env.config.max_accel + 1e-9


def test_time_limit_is_truncation():
    env = SwarmEnv(Config(duration=.1), "hold")
    observations, _ = env.reset(seed=7)
    controller = SlotController(env.config)
    env.step(controller(observations))
    _, _, terminated, truncated, _ = env.step(controller(env.last_observations))
    assert not any(terminated.values()) and all(truncated.values())
    assert env.step({}) == ({}, {}, {}, {}, {})


def test_range_mask_and_dropout():
    env = SwarmEnv(Config(sensor_range=.1))
    observations, _ = env.reset(seed=1)
    assert all(not obs["neighbors"][:, -1].any() for obs in observations.values())
    env.obstacle_list = [Obstacle("far", np.array([50., 0, 3]), np.zeros(3), .5)]
    assert all(not obs["obstacles"][:, -1].any() for obs in env._observe().values())


def test_recovery_requires_each_original_slot_and_full_hold():
    env = SwarmEnv(Config(drones=2), "hold")
    env.reset(seed=5)
    env.obstacle_list = [Obstacle("event", np.array([30., 0, 3]), np.zeros(3), .5, 0, 1, True)]
    env.time = 1.0
    env.positions = env.slots()[::-1].copy()
    env.velocities[:] = 0
    env._update_recovery()
    assert env.events["event"]["settled_since_s"] is None
    env.positions = env.slots().copy()
    for t in np.arange(1, 2, .05):
        env.time = float(t)
        env._update_recovery()
        assert env.events["event"]["recovery_s"] is None
    env.time = 2.0
    env._update_recovery()
    assert env.events["event"]["recovery_s"] == pytest.approx(1.)


def test_injection_validation_and_persistent_slots():
    env = SwarmEnv()
    env.reset(seed=7)
    before = env.slots().copy()
    with pytest.raises(ValueError):
        env.inject(speed=float("nan"))
    name = env.inject(speed=2, radius=.8)
    assert name in [o.name for o in env.active_obstacles()]
    np.testing.assert_array_equal(before, env.slots())


def test_predictive_controller_recovers_but_blind_controller_collides():
    config = Config()
    result, _ = run_episode(config, "crossing", 1000, PredictiveController(config))
    blind, _ = run_episode(config, "crossing", 1000, SlotController(config))
    assert result["failure"] is None
    assert 1 <= result["events"][0]["recovery_s"] < 5
    assert blind["collision"]


@pytest.mark.parametrize("change", [{"drones": 1}, {"spacing": .1}, {"dt": float("nan")},
                                   {"dropout": 1}, {"substeps": 1.5}, {"max_speed": 0}])
def test_invalid_config(change):
    with pytest.raises(ValueError):
        replace(Config(), **change)
