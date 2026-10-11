"""Small CPU checks of BC likelihoods, action APIs and the offline loop."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch
from gymnasium.spaces import Box, Discrete
from omegaconf import OmegaConf

from agent import PolicyAgent
from algorithm import BehaviorCloning, OfflineAlgorithm, BaseAlgorithm
from logger.logger import Logger
from memory import ReplayBuffer
from run_bc import build_actor
from utils.networks import DeterministicActor, MLPNetwork
from utils.result import Result


class BCTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        np.random.seed(7)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.obs = Box(-np.inf, np.inf, (3,), dtype=np.float32)
        self.actor_args = OmegaConf.create(dict(
            name="BC", actor_type="auto", actor_lr=0.003, actor_optimizer="Adam",
            batch_size=16, hidden_sizes=[8, 8], activation="tanh", initialize=False,
            state_dependent_std=True, initial_log_sigma=0.0, common_net=False,
            embedding_dim=16,
        ))

    def make_bc(self, kind, obs_norm=False):
        action_space = Discrete(3) if kind == "DiscreteProbabilityActor" else Box(-0.4, 0.4, (2,), dtype=np.float32)
        self.actor_args.actor_type = kind
        actor = build_actor(self.obs, action_space, self.actor_args, "cpu")
        agent = PolicyAgent(self.obs, action_space, "cpu", actor)
        buffer = ReplayBuffer(self.obs, action_space, 8, 1, nstep=1)
        buffer.buffer["states"][0] = np.arange(24, dtype=np.float32).reshape(8, 3) / 24
        buffer.buffer["actions"].fill(1 if isinstance(action_space, Discrete) else 0.15)
        buffer.full = True
        args = SimpleNamespace(
            seed=0, total_epoch=3, env=SimpleNamespace(
                name="CartPole-v1", num_training_envs=1, num_testing_envs=1,
                obs_norm=obs_norm, absorbing=False),
            test_interval=0, test_episodes=0, interact_per_epoch=0,
            update_step_per_epoch=1, train_action_deterministic=False,
            save_interval=0, train_log_interval=1, reward_buffer_size=2,
            gamma=0.99, test_epsilon=0, log_dir=self.tmp.name,
        )
        logger = Mock()
        algorithm = BehaviorCloning(
            None, None, buffer, agent, logger, "cpu", args, self.actor_args)
        return algorithm

    def test_all_probabilistic_actors_optimize_exact_likelihood_only(self):
        for kind in ("DiagGaussianActor", "TanhGaussianActor", "DiscreteProbabilityActor"):
            with self.subTest(actor=kind):
                algorithm = self.make_bc(kind)
                states = algorithm.buffer.buffer["states"][0]
                actions = algorithm.buffer.buffer["actions"][0]
                extra = torch.nn.Linear(3, 1)
                algorithm.agent.extra_critic = extra
                untouched = {k: v.clone() for k, v in extra.state_dict().items()}
                with torch.no_grad():
                    before = -algorithm.agent.log_prob(states, actions).mean().item()
                    if kind == "DiscreteProbabilityActor":
                        expected = torch.nn.functional.cross_entropy(
                            algorithm.agent.actor(torch.as_tensor(states)),
                            torch.as_tensor(actions[:, 0]).long()).item()
                    else:
                        expected = -algorithm.agent.actor.get_dist(
                            torch.as_tensor(states)).log_prob(torch.as_tensor(actions)).mean().item()
                result = algorithm._update_policy(dict(states=states, actions=actions))
                self.assertAlmostEqual(result.metrics["actor/loss"], expected, places=5)
                for _ in range(7):
                    algorithm._update_policy(dict(states=states, actions=actions))
                after = -algorithm.agent.log_prob(states, actions).mean().item()
                self.assertLess(after, before)
                for key, value in extra.state_dict().items():
                    torch.testing.assert_close(value, untouched[key], rtol=0, atol=0)
                self.assertTrue(all(p.grad is None for p in extra.parameters()))
                for deterministic in (False, True):
                    action, info = algorithm.agent.select_action(states, deterministic)
                    self.assertIsInstance(action, np.ndarray)
                    self.assertEqual(info["log_probs"].shape, (8,))
                if kind == "DiscreteProbabilityActor":
                    torch.testing.assert_close(
                        algorithm.agent.log_prob(states, actions[:, 0]),
                        algorithm.agent.log_prob(states, actions))

    def test_offline_loop_never_collects_or_writes_dataset(self):
        algorithm = self.make_bc("DiagGaussianActor", obs_norm=True)
        self.assertIsInstance(algorithm, OfflineAlgorithm)
        self.assertIsInstance(algorithm, BaseAlgorithm)
        algorithm.update_step_per_epoch = 2
        original = {k: v.copy() for k, v in algorithm.buffer.buffer.items()}
        poison_env = Mock()
        poison_env.reset.side_effect = AssertionError("reset during offline training")
        poison_env.step.side_effect = AssertionError("step during offline training")
        algorithm.training_envs = poison_env
        algorithm.buffer.add = Mock(side_effect=AssertionError("online buffer write"))
        with patch("algorithm.offline.evaluate", side_effect=AssertionError("evaluation disabled")):
            algorithm.run()
        self.assertEqual(algorithm.interaction_step, 0)
        self.assertEqual(algorithm.gradient_step, 6)
        self.assertEqual(algorithm.logger.step_axis, "gradient")
        self.assertEqual([c.args[2] for c in algorithm.logger.log_train.call_args_list], [2, 4, 6])
        np.testing.assert_allclose(algorithm.obs_rms.mean, original["states"][0].mean(axis=0))
        self.assertEqual(algorithm.obs_rms.count, 8)
        for key, value in algorithm.buffer.buffer.items():
            np.testing.assert_array_equal(value, original[key])
        poison_env.reset.assert_not_called()
        poison_env.step.assert_not_called()
        algorithm.buffer.add.assert_not_called()
        self.assertTrue(Path(self.tmp.name, "newest_model.pth").is_file())
        self.assertTrue(Path(self.tmp.name, "newest_obs_rms.pth").is_file())
        algorithm.logger.close.assert_called_once()

    def test_optional_evaluation_uses_updates_without_collecting_training_data(self):
        algorithm = self.make_bc("DiscreteProbabilityActor")
        algorithm.test_interval = 2
        algorithm.test_episodes = 1
        with patch("algorithm.offline.evaluate", return_value=(Result("test"), 10.0)) as evaluate:
            algorithm.run()
        self.assertEqual(evaluate.call_count, 3)  # before updates 0, 2, and after 3
        self.assertEqual(algorithm.interaction_step, 0)
        self.assertEqual(algorithm.gradient_step, 3)
        self.assertTrue(Path(self.tmp.name, "best_model.pth").is_file())

    def test_logger_uses_gradient_axis_only_when_requested(self):
        logger = Logger("bc", "test", {}, use_wandb=False, use_tensorboard=False,
                        use_swanlab=False, step_axis="gradient")
        logger._log = Mock()
        result = Result("train")
        result.add_metric("actor/loss", 1.0)
        with patch("logger.logger._print"):
            logger.log_train(1, 0, 9, result)
        logger._log.assert_called_once_with("train/actor/loss", 1.0, 9)

    def test_reject_deterministic_policy_and_keep_tanh_boundary_finite(self):
        action_space = Box(-0.4, 0.4, (2,), dtype=np.float32)
        deterministic = DeterministicActor(self.obs, action_space, MLPNetwork)
        with self.assertRaisesRegex(TypeError, "probabilistic"):
            PolicyAgent(self.obs, action_space, "cpu", deterministic)
        algorithm = self.make_bc("TanhGaussianActor")
        states = np.ones((2, 3), dtype=np.float32)
        actions = np.array([[-0.4, 0.4], [0.4, -0.4]], dtype=np.float32)
        result = algorithm._update_policy(dict(states=states, actions=actions))
        self.assertTrue(np.isfinite(result.metrics["actor/loss"]))
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all()
                            for p in algorithm.agent.actor.parameters()))


if __name__ == "__main__":
    unittest.main()
