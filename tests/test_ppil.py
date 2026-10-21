"""Deep PPIL tests: reference parity, gradient flow, update counts and saving."""
import ast
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from gymnasium.spaces import Box
from hydra import compose, initialize_config_dir

from agent import SACAgent
from algorithm import PPIL
from memory import ReplayBuffer
from utils.networks import MLPNetwork, QFunction, DoubleQFunction, TanhGaussianActor

ROOT = Path(__file__).resolve().parents[1]


class PPILTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(4)
        np.random.seed(4)

    def loss_model(self, source="expert", loss_type="value_expert"):
        algo = PPIL.__new__(PPIL)
        algo.gamma, algo.tau, algo.reg_alpha = .99, .08, .5
        algo.weight_source, algo.loss_type = source, loss_type
        algo.regularize, algo.chi = True, False
        return algo

    def test_reference_loss_and_gradients(self):
        # Execute the actual reference function with one inner evaluation.
        tree = ast.parse((ROOT / "outputs/P2IL/deep_PPIL/iq.py").read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "ppil_loss")
        assignment = next(n for n in fn.body if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == "steps" for t in n.targets))
        assignment.value = ast.Constant(value=1)
        scope = {"torch": torch, "np": np}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), "reference", "exec"), scope)
        for source in ("expert", "online"):
            for loss_type in ("value_expert", "value", "v0"):
                for target in (False, True):
                    model = self.loss_model(source, loss_type)
                    model.chi = True
                    values = [torch.randn(6, 1, dtype=torch.float64, requires_grad=True) for _ in range(4)]
                    q, r, v, nv = values
                    mask = torch.tensor([False] * 3 + [True] * 3)
                    dones = torch.tensor([[0.], [1.], [0.], [0.], [0.], [1.]], dtype=torch.float64)
                    obs = torch.arange(6).reshape(-1, 1)
                    nxt = obs + 6
                    getv = lambda s: torch.cat([v, nv])[s[:, 0].long()]
                    agent = SimpleNamespace(
                        gamma=.99, critic=lambda *args: q, get_reward=lambda *args: r,
                        getV=getv, get_targetV=getv, critic_target_update_frequency=1,
                        args=SimpleNamespace(
                            train=SimpleNamespace(use_target=target),
                            agent=SimpleNamespace(online=source == "online", tau=.08),
                            method=SimpleNamespace(loss=loss_type, grad_pen=False,
                                                   div=None, chi=True, regularize=True, alpha=.5)))
                    batch = obs, nxt, obs, torch.zeros_like(r), dones, mask[:, None]
                    expected, _ = scope["ppil_loss"](agent, r, q, v, nv, batch)
                    actual, _ = model._ppil_loss(q, r, v, nv.detach() if target else nv, dones, mask)
                    torch.testing.assert_close(actual, expected)
                    ga = torch.autograd.grad(actual, values, retain_graph=True, allow_unused=True)
                    ge = torch.autograd.grad(expected, values, allow_unused=True)
                    for a, e, leaf in zip(ga, ge, values):
                        # The fake getV concatenates v/nv; an unused slice has
                        # an explicit zero gradient instead of a disconnected None.
                        torch.testing.assert_close(torch.zeros_like(leaf) if a is None else a,
                                                   torch.zeros_like(leaf) if e is None else e)

    def test_logmeanexp_gradient_and_extreme_residuals(self):
        model = self.loss_model()
        q, r, v, nv = [torch.randn(8, 1, requires_grad=True) for _ in range(4)]
        expert = torch.tensor([False] * 4 + [True] * 4)
        done = torch.zeros_like(q)
        loss, metrics = model._ppil_loss(q, r, v, nv, done, expert)
        delta = (r + .99 * nv - q)[expert]
        exact = (-r[expert].mean()
                 + (torch.logsumexp(.08 * delta.flatten(), 0) - np.log(4)) / .08
                 + (v - .99 * nv)[expert].mean()
                 + (q.square().mean() + r.square().mean()) / 2)
        for a, b in zip(torch.autograd.grad(loss, [q, r, v, nv], retain_graph=True),
                        torch.autograd.grad(exact, [q, r, v, nv])):
            torch.testing.assert_close(a, b)
        torch.testing.assert_close(metrics["objective"], exact.detach())
        model.tau = 100
        huge_loss, huge_metrics = model._ppil_loss(q * 1e4, r, v, nv, done, expert)
        self.assertTrue(torch.isfinite(huge_loss))
        self.assertTrue(all(torch.isfinite(t) for t in huge_metrics.values()))

    def make_algorithm(self, target=False, double=False):
        obs = Box(-np.inf, np.inf, (3,), dtype=np.float32)
        act = Box(-.4, .4, (2,), dtype=np.float32)
        with initialize_config_dir(config_dir=str(ROOT / "config"), version_base="1.3"):
            args = compose(config_name="config", overrides=[
                "algorithm=ppil", "env.name=Humanoid-v5", "env.num_training_envs=1",
                "env.num_testing_envs=1", "env.obs_norm=false", "env.absorbing=false",
                "algorithm.rl.batch_size=4", f"algorithm.rl.use_target={str(target).lower()}",
                f"algorithm.rl.double_critic={str(double).lower()}",
                "algorithm.rl.learn_temp=true", "interact_per_epoch=1"])
        actor = TanhGaussianActor(obs, act, MLPNetwork, hidden_sizes=[8, 8],
                                  state_dependent_std=True, common_net=True, log_std_bounds=[-5, 2])
        critic = (DoubleQFunction if double else QFunction)(obs, act, MLPNetwork, hidden_sizes=[8, 8])
        agent = SACAgent(obs, act, "cpu", actor, critic,
                         target_critic=copy.deepcopy(critic) if target else None,
                         use_target=target, double_critic=double)
        agent.reward = QFunction(obs, act, MLPNetwork, hidden_sizes=[8, 8])
        buffer, expert = [ReplayBuffer(obs, act, 16, 1, gamma=.99) for _ in range(2)]
        batch = dict(states=np.random.randn(1, 8, 3).astype("float32"),
                     next_states=np.random.randn(1, 8, 3).astype("float32"),
                     actions=np.random.uniform(-.4, .4, (1, 8, 2)).astype("float32"),
                     rewards=np.full((1, 8), 1e6, dtype="float32"),
                     dones=np.array([[0, 0, 1, 0, 0, 0, 0, 1]], dtype="float32"),
                     truncateds=np.zeros((1, 8), dtype="float32"))
        buffer.add(batch); expert.add(batch)
        env = SimpleNamespace(observation_space=obs, action_space=act)
        return PPIL(env, None, buffer, expert, agent, None, "cpu", args, args.algorithm.rl)

    def test_inner_steps_update_all_networks_and_checkpoint(self):
        for target, double in ((False, False), (True, True)):
            algo = self.make_algorithm(target, double)
            before = {k: v.clone() for k, v in algo.agent.state_dict().items()}
            with patch.object(algo.buffer, "sample_mixed", wraps=algo.buffer.sample_mixed) as sample:
                result = algo._update_policy()
                self.assertEqual(sample.call_count, 1)
            self.assertEqual(algo.critic_updates, 20)
            for optimizer, count in ((algo.critic_optimizer, 20),
                                     (algo.reward_optimizer, 20), (algo.actor_optimizer, 1)):
                self.assertEqual({int(v["step"]) for v in optimizer.state.values()}, {count})
            for prefix in ("actor.", "critic.", "reward."):
                self.assertTrue(any(not torch.equal(v, before[k])
                                    for k, v in algo.agent.state_dict().items() if k.startswith(prefix)))
            self.assertTrue(all(np.isfinite(v) for v in result.metrics.values()))
            with tempfile.TemporaryDirectory() as tmp:
                path = str(Path(tmp) / "agent.pth")
                algo.agent.save(path)
                expected = {k: v.clone() for k, v in algo.agent.state_dict().items()}
                for param in algo.agent.reward.parameters():
                    param.data.zero_()
                algo.agent.load(path)
                for k, v in expected.items():
                    torch.testing.assert_close(algo.agent.state_dict()[k], v)

    def test_task_rewards_do_not_change_update(self):
        a = self.make_algorithm()
        b = self.make_algorithm()
        b.agent.load_state_dict(a.agent.state_dict())
        # Same sampled data; only the task reward field differs.
        batch = a.buffer.sample_mixed(8, a.expert_buffer, .5)
        other = copy.deepcopy(batch)
        other["rewards"].fill(-1e9)
        for algo, data in ((a, batch), (b, other)):
            torch.manual_seed(8)
            with patch.object(algo.buffer, "sample_mixed", return_value=copy.deepcopy(data)):
                algo._update_policy()
        for k, value in a.agent.state_dict().items():
            torch.testing.assert_close(value, b.agent.state_dict()[k])

    def test_actor_bounds_and_default_compatibility(self):
        obs = Box(-np.inf, np.inf, (3,), dtype=np.float32)
        act = Box(-.4, .4, (2,), dtype=np.float32)
        actor = TanhGaussianActor(obs, act, MLPNetwork, hidden_sizes=[8],
                                  state_dependent_std=True, common_net=True)
        states = torch.randn(4, 3)
        raw = actor.log_sigma(actor.trunk(states))
        _, std = actor(states)
        torch.testing.assert_close(std, raw.clamp(-20, 2).exp())
        actor.log_std_bounds = (-5., 2.)
        _, std = actor(states)
        torch.testing.assert_close(std, (-5 + 3.5 * (raw.tanh() + 1)).exp())
        actions, logp, u = actor.get_action(states, False)
        self.assertTrue((actions.abs() <= .4).all())
        torch.testing.assert_close(logp, actor.get_log_prob(states, actions, u))


if __name__ == "__main__":
    unittest.main()
