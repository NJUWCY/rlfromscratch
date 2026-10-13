"""Small deterministic PWIL mass, boundary, replay and SAC integration checks."""

import copy
import tempfile
import unittest
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from omegaconf import OmegaConf
from stable_baselines3.common.vec_env import DummyVecEnv

from agent import SACAgent
from algorithm import ALGORITHM_DICT, PWIL
from algorithm.pwil import PWILRewarder, PWILRewardWrapper
from env.basic_envs import Monitor, TruncatedMonitor
from memory import ExpertDataset, ReplayBuffer
from utils.networks import DoubleQFunction, MLPNetwork, TanhGaussianActor


class ShortEnv(gym.Env):
    observation_space = gym.spaces.Box(-np.inf, np.inf, (2,), dtype=np.float32)
    action_space = gym.spaces.Box(-0.4, 0.4, (1,), dtype=np.float32)

    def __init__(self, length=3, timeout=False):
        self.length, self.timeout = length, timeout
        self.t = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        return np.array([0, 10], np.float32), {}

    def step(self, action):
        self.t += 1
        end = self.t == self.length
        return np.array([self.t, 10], np.float32), 99.0, end and not self.timeout, end and self.timeout, {}


class PWILTest(unittest.TestCase):
    def test_greedy_fractional_mass_and_reference_epsilon(self):
        # D=3, T=2. Reference consumes 1/2-1e-6 per policy transition.
        s = np.array([[0.0], [2.0], [4.0]])
        r = PWILRewarder(s, np.zeros((3, 1)), time_horizon=2)
        scale = np.std(s)
        expected_cost = (1 / 6 - 1e-6) * 2 / scale
        reward = r.compute_reward([0], [0])
        self.assertAlmostEqual(r.last_cost[0], expected_cost)
        self.assertAlmostEqual(reward, 5 * np.exp(-10 / np.sqrt(2) * expected_cost), places=6)
        np.testing.assert_allclose(r.expert_weights[0], [1 / 6 + 1e-6, 1 / 3])
        r.compute_reward([4], [0])
        self.assertAlmostEqual(r.expert_weights[0].sum(), 2e-6)
        self.assertEqual(r.steps[0], 2)
        r.reset()
        np.testing.assert_array_equal(r.expert_weights[0], np.ones(3) / 3)

    def test_depletion_changes_reward_and_observation_only_ignores_action(self):
        s = np.array([[0.0], [2.0]])
        r = PWILRewarder(s, np.array([[0.0], [1.0]]), time_horizon=2)
        self.assertEqual(r.compute_reward([0], [0]), 5)
        self.assertLess(r.compute_reward([0], [0]), 5)
        r = PWILRewarder(s, np.zeros((2, 1)), time_horizon=2, observation_only=True)
        first = r.compute_reward([0], [100])
        r.reset()
        self.assertEqual(first, r.compute_reward([0], [-100]))
        self.assertTrue(np.isfinite(r.expert_atoms).all())

    def test_vector_env_resets_only_finished_stream_and_keeps_task_return(self):
        envs = DummyVecEnv([lambda: Monitor(ShortEnv(2)), lambda: Monitor(ShortEnv(3, True))])
        self.addCleanup(envs.close)
        r = PWILRewarder(np.array([[0, 10], [1, 10], [2, 10]]), np.zeros((3, 1)), 3, num_envs=2)
        wrapped = PWILRewardWrapper(envs, r)
        wrapped.reset()
        _, rew, done, _ = wrapped.step(np.zeros((2, 1)))
        self.assertTrue(np.all(rew <= 5))
        np.testing.assert_array_equal(r.steps, [1, 1])
        _, _, done, infos = wrapped.step(np.zeros((2, 1)))
        np.testing.assert_array_equal(done, [True, False])
        np.testing.assert_array_equal(r.steps, [0, 2])
        self.assertEqual(infos[0]["episode"]["r"], 198)
        _, _, done, _ = wrapped.step(np.zeros((2, 1)))
        np.testing.assert_array_equal(done, [False, True])
        np.testing.assert_array_equal(r.steps, [1, 0])

    def make_algorithm(self, normalized=False, frequency=1):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        envs = DummyVecEnv([lambda: TruncatedMonitor(Monitor(ShortEnv(3, True)))])
        self.addCleanup(envs.close)
        obs, act = envs.observation_space, envs.action_space
        path = Path(tmp.name) / "expert.npz"
        np.savez(path, observations=np.array([[0, 10], [1, 10], [2, 10], [3, 10]], np.float32),
                 next_observations=np.array([[1, 10], [2, 10], [3, 10], [4, 10]], np.float32),
                 actions=np.zeros((4, 1), np.float32), rewards=np.full(4, 99),
                 terminals=np.array([0, 1, 0, 0]), timeouts=np.array([0, 0, 0, 1]),
                 episode_ends=np.array([0, 1, 0, 1]))
        expert = ExpertDataset(str(path), obs, act, 2, frequency)
        replay = ReplayBuffer(obs, act, 32, 1)
        rl = OmegaConf.load(Path(__file__).resolve().parents[1] / "config/algorithm/sac.yaml")
        rl.batch_size = 8
        rl.learn_temp = True
        rl.update_log_temp = True
        args = OmegaConf.create(dict(
            seed=0, total_epoch=2, test_interval=1, interact_per_epoch=4,
            update_step_per_epoch=1, test_episodes=1, train_action_deterministic=False,
            save_interval=1, train_log_interval=1, reward_buffer_size=10,
            gamma=0.99, test_epsilon=0, log_dir=tmp.name, start_train_step=0,
            expert_dataset=dict(subsample_frequency=frequency),
            algorithm=dict(time_horizon=4, alpha=5., beta=5., observation_only=False, num_transitions_rb=8),
            env=dict(name="Hopper-v5", num_training_envs=1, num_testing_envs=1,
                     obs_norm=normalized, absorbing=False, max_episode_length=4)))
        actor = TanhGaussianActor(obs, act, MLPNetwork, hidden_sizes=[8, 8], state_dependent_std=True)
        critic = DoubleQFunction(obs, act, MLPNetwork, hidden_sizes=[8, 8])
        agent = SACAgent(obs, act, "cpu", actor, critic, target_critic=copy.deepcopy(critic), use_target=True)
        algo = PWIL(envs, None, replay, expert, agent, None, "cpu", args, rl)
        return algo, expert

    def test_prefill_true_transitions_reward_and_no_expert_mutation(self):
        algo, expert = self.make_algorithm()
        snapshot = {k: v.copy() for k, v in expert.buffer.items()}
        algo.initialize()
        self.assertIs(ALGORITHM_DICT["PWIL"], PWIL)
        self.assertEqual(algo.buffer.pos, 8)
        self.assertEqual(algo.interaction_step, 0)
        for k in ("states", "next_states", "actions", "dones", "truncateds"):
            np.testing.assert_array_equal(algo.buffer.buffer[k][0, :8], np.tile(snapshot[k][0], (2,) + (1,) * (snapshot[k].ndim - 2)))
        np.testing.assert_array_equal(algo.buffer.buffer["rewards"][0, :8], 5)
        for k in snapshot:
            np.testing.assert_array_equal(expert.buffer[k], snapshot[k])

    def test_normalized_collection_timeout_and_actual_sac_update(self):
        algo, _ = self.make_algorithm(normalized=True)
        algo.initialize()
        rms = algo.training_envs.get_obs_rms()
        frozen_count = rms.count
        before = [p.detach().clone() for p in algo.agent.actor.parameters()]
        batch, _ = algo.interact_with_envs()
        self.assertTrue(batch["truncateds"].any())
        self.assertFalse(batch["dones"].any())
        np.testing.assert_allclose(batch["states"][0, 0], rms.norm([0, 10]))
        np.testing.assert_allclose(batch["next_states"][0, 2], rms.norm([3, 10]))
        self.assertTrue(np.all(batch["rewards"] <= 5))
        result = algo.update(batch, start_train=True)
        self.assertEqual(rms.count, frozen_count)
        self.assertEqual(algo.gradient_step, 1)
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, algo.agent.actor.parameters())))
        self.assertTrue(all(np.isfinite(v) for v in result.metrics.values()))
        np.testing.assert_array_equal(algo.rewarder.steps, [1])

    def test_random_offset_subsampling_and_invalid_mass_parameters(self):
        algo, expert = self.make_algorithm(frequency=2)
        self.assertEqual(expert.buffer_size, 2)
        self.assertEqual(algo.rewarder.num_atoms, expert.buffer_size)
        np.testing.assert_array_equal(expert.episode_ends, [0, 1])
        states = expert.buffer["states"][0]
        self.assertEqual(sum(states[:, 0] < 2), 1)
        self.assertEqual(sum(states[:, 0] >= 2), 1)
        np.testing.assert_allclose(algo.rewarder.expert_atoms[0],
                                   (np.concatenate([states, expert.buffer["actions"][0]], axis=1)
                                    - algo.rewarder.scaler.mean_) / algo.rewarder.scaler.scale_)
        algo.initialize()
        np.testing.assert_array_equal(algo.buffer.buffer["next_states"][0, :2],
                                      expert.buffer["next_states"][0])
        for kwargs in [dict(time_horizon=0), dict(alpha=-1), dict(beta=float("nan"))]:
            with self.assertRaises(ValueError):
                PWILRewarder([[0]], [[0]], **kwargs)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
