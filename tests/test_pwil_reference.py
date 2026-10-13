"""Compare against the actual reference source without requiring Acme/POT."""

import ast
import contextlib
import copy
import io
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from gymnasium.spaces import Box
from sklearn import preprocessing

from algorithm.pwil import PWILRewarder
from memory import ExpertDataset


REFERENCE = Path(__file__).resolve().parents[1] / "outputs/pwil/rewarder.py"


def load_reference_class():
    # Compile the unmodified class from the reference file. Its training reward
    # path only needs these modules and StepType labels, not Acme or POT.
    tree = ast.parse(REFERENCE.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                and n.name == "PWILRewarder")
    namespace = dict(np=np, copy=copy, random=random, preprocessing=preprocessing,
                     dm_env=SimpleNamespace(StepType=SimpleNamespace(FIRST=0, MID=1, LAST=2)))
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(REFERENCE), "exec"), namespace)
    return namespace["PWILRewarder"]


@unittest.skipUnless(REFERENCE.exists(), "outputs/pwil reference source is unavailable")
class ReferenceRewardTest(unittest.TestCase):
    def test_float32_dataset_rewards_match_reference(self):
        reference_class = load_reference_class()
        cases = [
            (np.float32, np.float32, False, 4, 1000),
            (np.float64, np.float64, False, 4, 37),
            (np.float64, np.float32, False, 4, 37),
            (np.float32, np.float32, True, 4, 37),
            (np.float64, np.float32, True, 4, 37),
            (np.float32, np.float32, False, 1, 37),
        ]
        for obs_dtype, act_dtype, state_only, frequency, horizon in cases:
            with self.subTest(obs=obs_dtype, action=act_dtype, state_only=state_only,
                              frequency=frequency, horizon=horizon), tempfile.TemporaryDirectory() as tmp:
                rng = np.random.default_rng(91)
                states = rng.normal(size=(200, 3)).astype(obs_dtype)
                states[:, 1] = 1  # Cover StandardScaler's constant-feature path.
                actions = (0.3 * np.tanh(rng.normal(size=(200, 2)))).astype(act_dtype)
                ends = np.zeros(200, dtype=bool)
                ends[39::40] = True
                path = Path(tmp) / "expert.npz"
                np.savez(path, observations=states, next_observations=states + 0.01,
                         actions=actions, rewards=np.ones(200), terminals=np.zeros(200),
                         timeouts=ends, episode_ends=ends)
                obs_space = Box(-np.inf, np.inf, (3,), dtype=np.float32)
                action_space = Box(-0.4, 0.4, (2,), dtype=np.float32)
                np.random.seed(17)
                python_rng_state = random.getstate()
                with contextlib.redirect_stdout(io.StringIO()):
                    expert = ExpertDataset(str(path), obs_space, action_space, 3, frequency,
                                           random_start=True)
                self.assertEqual(random.getstate(), python_rng_state)
                for key in ("states", "next_states", "actions"):
                    self.assertEqual(expert.buffer[key].dtype, np.dtype(np.float32))
                # Compare reward arithmetic on the same selected float32 data.
                # ExpertDataset keeps the library's NumPy selection, not the
                # reference's Python RNG selection or original source dtype.
                demos = [[dict(observation=obs.copy(), action=action.copy())
                          for obs, action in zip(expert.buffer["states"][0],
                                                 expert.buffer["actions"][0])]]
                reference = reference_class(
                    demos, 1,
                    SimpleNamespace(observations=obs_space, actions=action_space),
                    num_demonstrations=1, time_horizon=horizon, observation_only=state_only)
                current = PWILRewarder(expert.buffer["states"][0], expert.buffer["actions"][0],
                                       time_horizon=horizon, observation_only=state_only, num_envs=2)
                np.testing.assert_array_equal(current.vectorized_demonstrations,
                                              reference.vectorized_demonstrations)
                np.testing.assert_array_equal(current.scaler.mean_, reference.scaler.mean_)
                np.testing.assert_array_equal(current.scaler.scale_, reference.scaler.scale_)
                references = [copy.deepcopy(reference), copy.deepcopy(reference)]
                for ref in references:
                    ref.reset()
                # One stream resets halfway; the other completes a full horizon.
                for step in range(horizon):
                    if step == horizon // 2:
                        current.reset(0)
                        references[0].reset()
                    for stream, ref in enumerate(references):
                        obs = rng.normal(size=3).astype(np.float32)
                        obs[1] = 1
                        action = (0.3 * np.tanh(rng.normal(size=2))).astype(np.float32)
                        expected = ref.compute_reward(dict(observation=obs, action=action))
                        actual = current.compute_reward(obs, action, stream)
                        self.assertEqual(actual.dtype, expected.dtype)
                        self.assertEqual(actual.tobytes(), expected.tobytes())
                        np.testing.assert_array_equal(current.expert_weights[stream],
                                                      ref.expert_weights)
                        np.testing.assert_array_equal(current.expert_atoms[stream], ref.expert_atoms)
                self.assertAlmostEqual(current.expert_weights[1].sum(), horizon * 1e-6, places=12)
                current.reset()
                for stream, ref in enumerate(references):
                    ref.reset()
                    np.testing.assert_array_equal(current.expert_weights[stream], ref.expert_weights)
                    np.testing.assert_array_equal(current.expert_atoms[stream], ref.expert_atoms)
                print("Reward parity on shared float32 data:", np.dtype(obs_dtype), np.dtype(act_dtype),
                      "state_only=", state_only, "frequency=", frequency, "horizon=", horizon)


if __name__ == "__main__":
    unittest.main()
