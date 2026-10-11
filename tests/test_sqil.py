"""SQIL reward, replay and Bellman-update checks; CPU, no MuJoCo rollout."""

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from gymnasium.spaces import Box
from omegaconf import OmegaConf

from agent import SACAgent
from algorithm import ALGORITHM_DICT, SQIL
from memory import ExpertDataset, ReplayBuffer
from utils.networks import DoubleQFunction, MLPNetwork, QFunction, TanhGaussianActor
from utils.result import Result


class SQILTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(13)
        torch.manual_seed(13)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.obs = Box(-np.inf, np.inf, (3,), dtype=np.float32)
        self.act = Box(-0.4, 0.4, (2,), dtype=np.float32)
        self.env = SimpleNamespace(observation_space=self.obs, action_space=self.act,
                                   _norm_obs=lambda values: values + 100)
        self.online = dict(
            states=np.arange(12, dtype=np.float32).reshape(1, 4, 3),
            next_states=np.arange(12, dtype=np.float32).reshape(1, 4, 3) + 1,
            actions=np.zeros((1, 4, 2), dtype=np.float32),
            rewards=np.array([[99, -23, 18, 42]], dtype=np.float32),
            dones=np.array([[0, 1, 0, 0]], dtype=np.float32),
            truncateds=np.array([[0, 0, 1, 0]], dtype=np.float32),
        )
        # Include identical online/expert state-action pairs with different
        # source labels, plus both genuine terminations and timeouts.
        self.path = Path(self.directory.name) / "expert.npz"
        np.savez(self.path,
                 observations=self.online["states"][0],
                 next_observations=self.online["next_states"][0],
                 actions=self.online["actions"][0],
                 rewards=self.online["rewards"][0],
                 terminals=self.online["dones"][0],
                 timeouts=self.online["truncateds"][0],
                 episode_ends=np.array([False, True, False, True]))
        self.expert = ExpertDataset(str(self.path), self.obs, self.act, 2, 1, gamma=0.98)
        self.buffer = ReplayBuffer(self.obs, self.act, 4, 1, gamma=0.98)
        config_root = Path(__file__).resolve().parents[1] / "config"
        self.rl_args = OmegaConf.load(config_root / "algorithm/sac.yaml")
        self.rl_args.batch_size = 16
        self.rl_args.learn_temp = True
        self.rl_args.update_log_temp = True
        self.args = OmegaConf.create(dict(
            seed=0, total_epoch=2, test_interval=1, interact_per_epoch=4,
            update_step_per_epoch=1, test_episodes=1, train_action_deterministic=False,
            save_interval=1, train_log_interval=1, reward_buffer_size=10,
            gamma=0.98, test_epsilon=0, log_dir=self.directory.name,
            start_train_step=0,
            env=dict(name="Humanoid-v5", num_training_envs=1, num_testing_envs=1,
                     obs_norm=False, absorbing=False),
        ))
        actor = TanhGaussianActor(self.obs, self.act, MLPNetwork, hidden_sizes=[8, 8],
                                  state_dependent_std=True, activation=torch.nn.ReLU)
        critic = DoubleQFunction(self.obs, self.act, MLPNetwork,
                                 hidden_sizes=[8, 8], activation=torch.nn.ReLU)
        agent = SACAgent(self.obs, self.act, "cpu", actor, critic,
                         target_critic=copy.deepcopy(critic), use_target=True)
        self.kwargs = dict(training_envs=self.env, testing_envs=None, buffer=self.buffer,
                           expert_buffer=self.expert, agent=agent, logger=None,
                           device=torch.device("cpu"), args=self.args, rl_args=self.rl_args)
        self.algorithm = SQIL(**self.kwargs)

    def sampled_batch(self):
        with patch.object(self.algorithm, "_sac_update", return_value=Result("train")) as update:
            self.algorithm._update_policy()
            return update.call_args.args[0]

    def test_registration(self):
        self.assertIs(ALGORITHM_DICT["SQIL"], SQIL)

    def test_labels_are_balanced_and_task_rewards_never_reach_sac(self):
        original_rewards = self.online["rewards"].copy()
        self.algorithm._update_buffer(self.online)
        np.testing.assert_array_equal(self.online["rewards"], original_rewards)
        np.testing.assert_array_equal(self.buffer.buffer["rewards"], 0)
        # Source labels must be applied even if the replay's reward field changes.
        self.buffer.buffer["rewards"].fill(-999)
        batch = self.sampled_batch()
        np.testing.assert_array_equal(batch["rewards"], [0] * 8 + [1] * 8)
        self.assertTrue(np.all(batch["nstep_gamma"] == 0.98))

    def test_sampling_schedule_controls_source_counts(self):
        self.args.algorithm = dict(sampling_schedule=[[1.0, 0.5, 10], [0.5, 0.0, 20]])
        self.rl_args.batch_size = 15
        algo = SQIL(**self.kwargs)
        algo._update_buffer(self.online)
        for step, ratio, expert_count in ((0, 1.0, 15), (5, 0.75, 11),
                                           (10, 0.5, 7), (15, 0.25, 3),
                                           (20, 0.0, 0), (30, 0.0, 0)):
            with self.subTest(step=step), patch.object(
                    algo, "_sac_update", return_value=Result("train")) as update:
                algo.gradient_step = step
                result = algo._update_policy()
                rewards = update.call_args.args[0]["rewards"]
                np.testing.assert_array_equal(rewards, [0] * (15-expert_count) + [1] * expert_count)
                self.assertAlmostEqual(result.metrics["sqil/expert_ratio"], ratio)
                self.assertAlmostEqual(result.metrics["sqil/expert_fraction"], expert_count/15)

    def test_online_overwrites_do_not_evict_or_mutate_demonstrations(self):
        snapshot = {key: value.copy() for key, value in self.expert.buffer.items()}
        file_bytes = self.path.read_bytes()
        for _ in range(4):
            self.algorithm._update_buffer(self.online)
            self.sampled_batch()
        self.assertTrue(self.buffer.full)
        for key, value in snapshot.items():
            np.testing.assert_array_equal(self.expert.buffer[key], value)
        self.assertEqual(self.path.read_bytes(), file_bytes)

    def test_normalization_applies_only_to_sampled_expert_observations(self):
        self.algorithm._update_buffer(self.online)
        np.random.seed(7)
        raw = self.sampled_batch()
        self.args.env.obs_norm = True
        np.random.seed(7)
        normalized = self.sampled_batch()
        for key in ("states", "next_states"):
            np.testing.assert_array_equal(normalized[key][:8], raw[key][:8])
            np.testing.assert_array_equal(normalized[key][8:], raw[key][8:] + 100)
        np.testing.assert_array_equal(normalized["actions"], raw["actions"])
        np.testing.assert_array_equal(self.buffer.buffer["states"], self.online["states"])

    def test_real_update_matches_terminal_and_timeout_bellman_targets(self):
        class RecordingAdam(torch.optim.Adam):
            def step(self, closure=None):
                self.recorded = [p.grad.detach().clone() for group in self.param_groups
                                 for p in group["params"]]
                return super().step(closure)

        algo = self.algorithm
        algo._update_buffer(self.online)
        optimizer = RecordingAdam(algo.agent.critic.parameters(), lr=3e-4)
        algo.critic_optimizer = optimizer
        actor_before = [p.detach().clone() for p in algo.agent.actor.parameters()]
        target_before = [p.detach().clone() for p in algo.agent.target_critic.parameters()]
        alpha_before = algo.log_temp.detach().clone()
        # Deterministically include every transition in each half of the batch.
        def ordered_sample(buffer, size):
            return buffer._sample_from_indices(np.arange(size) % buffer.buffer_size)
        with patch.object(self.buffer, "sample", side_effect=lambda n: ordered_sample(self.buffer, n)), \
             patch.object(self.expert, "sample", side_effect=lambda n: ordered_sample(self.expert, n)):
            batch = self.sampled_batch()
            target = torch.tensor(batch["rewards"] +
                                  (1 - batch["dones"]) * batch["nstep_gamma"] * 2,
                                  dtype=torch.float32).unsqueeze(1)
            qs = algo.agent.get_double_q_function(batch["states"], batch["actions"])
            expected_loss = sum(torch.nn.functional.mse_loss(q, target) for q in qs)
            expected_gradients = torch.autograd.grad(expected_loss, algo.agent.critic.parameters())
            with patch.object(algo.agent, "get_value", return_value=torch.full((16, 1), 2.0)):
                result = algo._update_policy()
        for actual, expected in zip(optimizer.recorded, expected_gradients):
            torch.testing.assert_close(actual, expected)
        self.assertAlmostEqual(result.metrics["critic/q_loss"], expected_loss.item(), places=5)
        self.assertEqual(result.metrics["sqil/expert_fraction"], 0.5)
        self.assertEqual(result.metrics["sqil/reward_mean"], 0.5)
        self.assertEqual(algo.gradient_step, 1)
        self.assertTrue(all(np.isfinite(v) for v in result.metrics.values()))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(actor_before, algo.agent.actor.parameters())))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(target_before, algo.agent.target_critic.parameters())))
        self.assertFalse(torch.equal(alpha_before, algo.log_temp))
        terminal = batch["dones"].astype(bool)
        timeout = batch["truncateds"].astype(bool)
        self.assertTrue(terminal.any() and timeout.any())
        np.testing.assert_allclose(target[:, 0].numpy()[terminal], batch["rewards"][terminal])
        np.testing.assert_allclose(target[:, 0].numpy()[timeout], batch["rewards"][timeout] + 1.96)

    def test_single_critic_lsiq_defaults_update(self):
        rl_args = copy.deepcopy(self.rl_args)
        rl_args.double_critic = False
        rl_args.critic_loss_coef = 0.5
        rl_args.learn_temp = False
        rl_args.init_temp = 0.001
        rl_args.update_log_temp = False
        critic = QFunction(self.obs, self.act, MLPNetwork,
                           hidden_sizes=[8, 8], activation=torch.nn.ReLU)
        agent = SACAgent(self.obs, self.act, "cpu", self.algorithm.agent.actor,
                         critic, target_critic=copy.deepcopy(critic),
                         use_target=True, double_critic=False)
        algo = SQIL(**dict(self.kwargs, agent=agent, rl_args=rl_args))
        algo._update_buffer(self.online)
        before = [p.detach().clone() for p in critic.parameters()]
        alpha_before = algo.log_temp.detach().clone()
        ordered = lambda buffer, size: buffer._sample_from_indices(
            np.arange(size) % buffer.buffer_size)
        with patch.object(self.buffer, "sample", side_effect=lambda n: ordered(self.buffer, n)), \
             patch.object(self.expert, "sample", side_effect=lambda n: ordered(self.expert, n)):
            batch = self.sampled_batch()
            target = torch.tensor(batch["rewards"] +
                                  (1 - batch["dones"]) * batch["nstep_gamma"] * 2,
                                  dtype=torch.float32).unsqueeze(1)
            errors = (agent.get_q_function(batch["states"], batch["actions"]) - target).square()
            expected_loss = 0.25 * (errors[:8].mean() + errors[8:].mean())
            with patch.object(agent, "get_value", return_value=torch.full((16, 1), 2.0)):
                result = algo._update_policy()
        self.assertAlmostEqual(result.metrics["critic/q_loss"], expected_loss.item(), places=5)
        self.assertEqual(algo.gradient_step, 1)
        self.assertIn("critic1/td_error_abs", result.metrics)
        self.assertNotIn("critic2/td_error_abs", result.metrics)
        self.assertTrue(all(np.isfinite(v) for v in result.metrics.values()))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, critic.parameters())))
        torch.testing.assert_close(algo.log_temp, alpha_before)
        self.assertAlmostEqual(algo.log_temp.exp().item(), 0.001, places=7)

    def test_rejects_incompatible_configurations(self):
        for key, value in (("batch_size", 0), ("batch_size", -1), ("nstep", 2)):
            with self.subTest(key=key, value=value):
                rl_args = copy.deepcopy(self.rl_args)
                rl_args[key] = value
                with self.assertRaises(ValueError):
                    SQIL(**dict(self.kwargs, rl_args=rl_args))
        self.args.env.absorbing = True
        with self.assertRaisesRegex(ValueError, "absorbing"):
            SQIL(**self.kwargs)


if __name__ == "__main__":
    unittest.main()
