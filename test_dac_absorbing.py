"""Checks on DAC's absorbing-state plumbing, on both the policy and expert side."""
import tempfile
from pathlib import Path

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf

from env.make_envs import make_vec_envs
from env.absorbing import absorbing_state
from memory import AbsorbingReplayBuffer, ExpertDataset
from utils.utils import RunningMeanStd


def test_env_wrapper_extends_observations():
    env_args = OmegaConf.create(dict(name="Hopper-v5", max_episode_length=1000,
                                     num_training_envs=1, num_testing_envs=1,
                                     obs_norm=False, absorbing=True))
    envs = make_vec_envs(env_args, True, seed=0)
    assert envs.observation_space.shape == (12,), envs.observation_space.shape
    obs = envs.reset()
    assert obs.shape == (1, 12)
    assert obs[0, -1] == 0.0
    envs.close()
    print("env wrapper: obs 11 -> 12, indicator bit 0 on live states  OK")


def test_policy_side_absorbing():
    """Terminations get redirected to the absorbing state plus a self-loop."""
    from algorithm.dac import DAC

    obs_dim, action_dim, num_envs, horizon = 12, 3, 2, 5
    batch = dict(
        states=np.random.randn(num_envs, horizon, obs_dim).astype(np.float32),
        next_states=np.random.randn(num_envs, horizon, obs_dim).astype(np.float32),
        actions=np.random.randn(num_envs, horizon, action_dim).astype(np.float32),
        rewards=np.random.randn(num_envs, horizon).astype(np.float32),
        dones=np.zeros((num_envs, horizon), dtype=np.float32),
        truncateds=np.zeros((num_envs, horizon), dtype=np.float32),
    )
    # interact_with_envs already reports a time limit as done=0, truncated=1
    batch['dones'][0, 2] = 1.0        # real termination
    batch['truncateds'][1, 3] = 1.0   # time limit

    dac = DAC.__new__(DAC)
    dac.observation_space = type("S", (), {"shape": (obs_dim,)})()
    dac.action_dim = action_dim
    dac.absorbing = True
    dac.absorbing_state = absorbing_state(obs_dim)
    dac.absorbing_action = np.zeros(action_dim, dtype=np.float32)

    out = dac._flatten_transitions(batch)

    assert len(out['states']) == num_envs * horizon + 1, "one absorbing self-loop expected"
    assert out['absorbing'].sum() == 1
    assert out['next_absorbing'].sum() == 2, "terminal transition + self-loop point at absorbing"

    terminal = 2
    np.testing.assert_allclose(out['next_states'][terminal], dac.absorbing_state)
    assert out['dones'][terminal] == 0.0, "termination must not be bootstrapped away"

    loop = terminal + 1
    np.testing.assert_allclose(out['states'][loop], dac.absorbing_state)
    np.testing.assert_allclose(out['next_states'][loop], dac.absorbing_state)
    np.testing.assert_allclose(out['actions'][loop], np.zeros(action_dim))
    assert out['rewards'][loop] == 0.0 and out['dones'][loop] == 0.0
    assert out['absorbing'][loop] == 1.0
    np.testing.assert_allclose(out['states'][loop + 1], batch['states'][0, 3])

    # env 0 grew by one, so env 1 starts at horizon + 1
    time_limit = horizon + 1 + 3
    assert out['dones'][time_limit] == 0.0 and out['truncateds'][time_limit] == 1.0
    np.testing.assert_allclose(out['next_states'][time_limit], batch['next_states'][1, 3])
    print("policy side: absorbing self-loop sits immediately after the terminal  OK")


def test_buffer_roundtrip():
    import gymnasium as gym
    obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)
    act_space = gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
    buffer = AbsorbingReplayBuffer(obs_space, act_space, buffer_size=64, gamma=0.99)

    n = 40
    transitions = dict(
        states=np.random.randn(n, 12).astype(np.float32),
        next_states=np.random.randn(n, 12).astype(np.float32),
        actions=np.random.randn(n, 3).astype(np.float32),
        rewards=np.arange(n, dtype=np.float32),
        dones=np.zeros(n, dtype=np.float32),
        truncateds=np.zeros(n, dtype=np.float32),
        absorbing=np.zeros(n, dtype=np.float32),
        next_absorbing=np.zeros(n, dtype=np.float32),
    )
    transitions['absorbing'][7] = 1.0
    buffer.add_transitions(transitions)
    assert buffer.pos == n and not buffer.full

    sample = buffer.sample(256)
    for key in ('absorbing', 'next_absorbing', 'nstep_gamma'):
        assert key in sample, key
    assert set(np.unique(sample['rewards'])).issubset(set(np.arange(n, dtype=np.float32)))
    np.testing.assert_allclose(sample['nstep_gamma'], 0.99)

    buffer.add_transitions({k: v[:30] for k, v in transitions.items()})
    assert buffer.full and buffer.pos == 6, (buffer.full, buffer.pos)
    print("buffer: flat append, wraparound, absorbing fields sampled  OK")


def _dac_for_wrap(obs_dim, action_dim):
    from algorithm.dac import DAC
    dac = DAC.__new__(DAC)
    dac.observation_space = type("S", (), {"shape": (obs_dim,)})()
    dac.action_dim = action_dim
    dac.absorbing = True
    dac.absorbing_state = absorbing_state(obs_dim)
    dac.absorbing_action = np.zeros(action_dim, dtype=np.float32)
    return dac


def test_nstep_stops_at_absorbing_and_not_across_envs():
    """n-step follows one env, includes the self-loop, then stops before reset."""
    import gymnasium as gym

    obs_dim, action_dim = 4, 2
    gamma = 0.5
    nstep = 5
    obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
    act_space = gym.spaces.Box(low=-1, high=1, shape=(action_dim,), dtype=np.float32)
    buffer = AbsorbingReplayBuffer(obs_space, act_space, buffer_size=32,
                                   gamma=gamma, nstep=nstep, num_envs=2)
    dac = _dac_for_wrap(obs_dim, action_dim)

    horizon = 4
    batch = dict(
        states=np.arange(2 * horizon * obs_dim, dtype=np.float32).reshape(2, horizon, obs_dim),
        next_states=np.arange(100, 100 + 2 * horizon * obs_dim, dtype=np.float32).reshape(2, horizon, obs_dim),
        actions=np.arange(2 * horizon * action_dim, dtype=np.float32).reshape(2, horizon, action_dim),
        rewards=np.array([[1.0, 2.0, 3.0, 4.0],
                          [10.0, 20.0, 30.0, 40.0]], dtype=np.float32),
        dones=np.zeros((2, horizon), dtype=np.float32),
        truncateds=np.zeros((2, horizon), dtype=np.float32),
    )
    batch['dones'][0, 2] = 1.0          # env0 terminates at t=2, t=3 is the reset step
    batch['truncateds'][1, 2] = 1.0     # env1 hits the time limit at t=2

    for env_i in range(2):
        buffer.add_transitions(
            dac._wrap_absorbing_sequence(
                batch['states'][env_i], batch['next_states'][env_i],
                batch['actions'][env_i], batch['rewards'][env_i],
                batch['dones'][env_i], batch['truncateds'][env_i],
            ),
            env_index=env_i,
        )

    assert buffer.env_pos[0] == 5 and buffer.env_pos[1] == 4

    env0 = np.array([0])
    from_start = buffer._sample_from_indices(env0, np.array([0]))
    np.testing.assert_allclose(from_start['nstep_mask'][0], [1, 1, 1, 1, 0])
    np.testing.assert_allclose(from_start['nstep_gamma'][0], gamma ** 4)
    np.testing.assert_allclose(from_start['next_states'][0], dac.absorbing_state)
    assert from_start['next_absorbing'][0] == 1.0
    assert from_start['dones'][0] == 0.0
    assert from_start['truncateds'][0] == 0.0
    np.testing.assert_allclose(from_start['rewards'][0], 1 + gamma * 2 + gamma ** 2 * 3 + gamma ** 3 * 0)

    from_terminal = buffer._sample_from_indices(env0, np.array([2]))
    np.testing.assert_allclose(from_terminal['nstep_mask'][0], [1, 1, 0, 0, 0])
    np.testing.assert_allclose(from_terminal['next_states'][0], dac.absorbing_state)
    np.testing.assert_allclose(from_terminal['nstep_gamma'][0], gamma ** 2)

    from_loop = buffer._sample_from_indices(env0, np.array([3]))
    np.testing.assert_allclose(from_loop['nstep_mask'][0], [1, 0, 0, 0, 0])
    np.testing.assert_allclose(from_loop['nstep_gamma'][0], gamma)
    assert from_loop['absorbing'][0] == 1.0

    from_reset = buffer._sample_from_indices(env0, np.array([4]))
    np.testing.assert_allclose(from_reset['states'][0], batch['states'][0, 3])
    np.testing.assert_allclose(from_reset['nstep_mask'][0], [1, 0, 0, 0, 0])
    np.testing.assert_allclose(from_reset['rewards'][0], 4.0)

    env1 = np.array([1])
    from_env1 = buffer._sample_from_indices(env1, np.array([0]))
    np.testing.assert_allclose(from_env1['nstep_mask'][0], [1, 1, 1, 0, 0])
    np.testing.assert_allclose(from_env1['rewards'][0], 10 + gamma * 20 + gamma ** 2 * 30)
    np.testing.assert_allclose(from_env1['next_states'][0], batch['next_states'][1, 2])
    assert from_env1['dones'][0] == 0.0
    assert from_env1['truncateds'][0] == 1.0
    assert from_env1['nstep_states'][0, 1, 0] != from_start['nstep_states'][0, 1, 0]
    print("n-step: includes absorbing self-loop, stops before reset, stays in-env  OK")


def test_expert_side_absorbing():
    """An expert episode ending in a real terminal also visits the absorbing state."""
    obs_dim, action_dim = 11, 3
    lengths = [4, 3]
    n = sum(lengths)
    terminals = np.zeros(n, dtype=np.float32)
    timeouts = np.zeros(n, dtype=np.float32)
    terminals[3] = 1.0    # first episode really terminated
    timeouts[6] = 1.0     # second episode hit the time limit
    data = dict(
        observations=np.random.randn(n, obs_dim).astype(np.float32),
        next_observations=np.random.randn(n, obs_dim).astype(np.float32),
        actions=np.random.randn(n, action_dim).astype(np.float32),
        rewards=np.ones(n, dtype=np.float32),
        terminals=terminals,
        timeouts=timeouts,
    )

    import gymnasium as gym
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "expert.hdf5"
        with h5py.File(path, "w") as f:
            for key, value in data.items():
                f.create_dataset(key, data=value)

        obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim + 1,), dtype=np.float32)
        act_space = gym.spaces.Box(low=-1, high=1, shape=(action_dim,), dtype=np.float32)
        dataset = ExpertDataset(str(path), obs_space, act_space, trajectory_num=2,
                                subsample_frequency=1, gamma=0.99, nstep=1, absorbing=True)

    assert dataset.buffer_size == n + 1, "exactly one absorbing transition added"
    states = dataset.buffer['states'][0]
    flags = dataset.buffer['absorbing'][0]
    next_flags = dataset.buffer['next_absorbing'][0]
    assert states.shape[1] == obs_dim + 1
    assert flags.sum() == 1 and next_flags.sum() == 2
    np.testing.assert_allclose(states[flags > 0][0], absorbing_state(obs_dim + 1))
    # the terminal transition is redirected and no longer flagged as done
    terminal_row = int(np.flatnonzero(next_flags > 0)[0])
    np.testing.assert_allclose(dataset.buffer['next_states'][0, terminal_row], absorbing_state(obs_dim + 1))
    assert dataset.buffer['dones'][0, terminal_row] == 0.0
    # the time-limited episode is left alone
    assert np.all(states[:, -1][flags == 0] == 0.0)
    print("expert side: terminal episode wrapped, timeout episode untouched  OK")


def test_expert_wrap_scans_every_step():
    """A terminate that is not the last step of the split trajectory still gets a self-loop."""
    obs_dim, action_dim = 11, 3
    n = 5
    terminals = np.zeros(n, dtype=np.float32)
    timeouts = np.zeros(n, dtype=np.float32)
    terminals[1] = 1.0
    terminals[4] = 1.0
    episode_ends = np.zeros(n, dtype=np.float32)
    episode_ends[-1] = 1.0
    data = dict(
        observations=np.random.randn(n, obs_dim).astype(np.float32),
        next_observations=np.random.randn(n, obs_dim).astype(np.float32),
        actions=np.random.randn(n, action_dim).astype(np.float32),
        rewards=np.ones(n, dtype=np.float32),
        terminals=terminals,
        timeouts=timeouts,
        episode_ends=episode_ends,
    )

    import gymnasium as gym
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "expert.hdf5"
        with h5py.File(path, "w") as f:
            for key, value in data.items():
                f.create_dataset(key, data=value)

        obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim + 1,), dtype=np.float32)
        act_space = gym.spaces.Box(low=-1, high=1, shape=(action_dim,), dtype=np.float32)
        dataset = ExpertDataset(str(path), obs_space, act_space, trajectory_num=1,
                                subsample_frequency=1, gamma=0.99, nstep=1, absorbing=True)

    assert dataset.buffer_size == n + 2
    flags = dataset.buffer['absorbing'][0]
    next_flags = dataset.buffer['next_absorbing'][0]
    assert flags.sum() == 2 and next_flags.sum() == 4
    np.testing.assert_allclose(dataset.buffer['states'][0, 2], absorbing_state(obs_dim + 1))
    np.testing.assert_allclose(dataset.buffer['states'][0, 3, :-1], data['observations'][2])
    np.testing.assert_allclose(dataset.buffer['next_states'][0, 1], absorbing_state(obs_dim + 1))
    assert dataset.buffer['dones'][0, 1] == 0.0
    print("expert side: every real termination is wrapped, not just the last step  OK")


def test_obs_norm_leaves_indicator_alone():
    """Normalising the indicator bit would rescale the absorbing state's 1."""
    from env.make_envs import VecObsNorm

    obs_dim = 12
    live = np.concatenate([np.random.randn(500, obs_dim - 1) * 3 + 7,
                           np.zeros((500, 1))], axis=1).astype(np.float32)

    norm = VecObsNorm.__new__(VecObsNorm)
    norm.obs_rms = RunningMeanStd()
    norm.update_obs_rms = True
    norm.skip_last_dim = True
    norm._update_obs_rms(live)

    assert np.shape(norm.obs_rms.mean) == (obs_dim - 1,), "indicator must stay out of the statistics"

    normed_live = norm._norm_obs(live)
    assert np.all(normed_live[:, -1] == 0.0)
    assert abs(normed_live[:, :-1].mean()) < 0.1, "features are actually being normalised"

    absorbing = absorbing_state(obs_dim)[None]
    assert norm._norm_obs(absorbing)[0, -1] == 1.0, "absorbing indicator must survive untouched"

    # Without the guard the constant-zero column has zero variance, so the
    # absorbing 1 is divided by sqrt(eps) and then clipped to clip_max.
    naive = VecObsNorm.__new__(VecObsNorm)
    naive.obs_rms = RunningMeanStd()
    naive.update_obs_rms = True
    naive.skip_last_dim = False
    naive._update_obs_rms(live)
    assert naive._norm_obs(absorbing)[0, -1] >= 10.0, "this is the failure mode being prevented"
    print("obs_norm: indicator bit excluded from RMS and never rescaled  OK")


def test_expert_absorbing_matches_policy_under_obs_norm():
    """Expert and policy must agree on the absorbing state after normalisation."""
    from algorithm.dac import DAC
    from env.make_envs import VecObsNorm

    obs_dim = 12
    live = np.concatenate([np.random.randn(500, obs_dim - 1) * 3 + 7,
                           np.zeros((500, 1))], axis=1).astype(np.float32)
    envs = VecObsNorm.__new__(VecObsNorm)
    envs.obs_rms = RunningMeanStd()
    envs.update_obs_rms = True
    envs.skip_last_dim = True
    envs._update_obs_rms(live)

    dac = DAC.__new__(DAC)
    dac.device = torch.device("cpu")
    dac.training_envs = envs
    dac.absorbing_state = absorbing_state(obs_dim)
    dac.args = OmegaConf.create(dict(env=dict(obs_norm=True)))

    expert_states = np.concatenate([live[:4], absorbing_state(obs_dim)[None]], axis=0)
    flags = np.array([0, 0, 0, 0, 1], dtype=np.float32)
    out = dac._expert_states_to_tensor(expert_states, flags).numpy()

    # The policy side writes this exact constant into the buffer after normalisation.
    np.testing.assert_allclose(out[-1], dac.absorbing_state)
    np.testing.assert_allclose(out[:4], envs._norm_obs(live[:4]), rtol=1e-5)
    print("expert vs policy: identical absorbing constant under obs_norm  OK")


def test_sac_masks_absorbing():
    """The critic target must not add an entropy bonus at the absorbing state."""
    import gymnasium as gym
    from agent import SACAgent
    from utils.networks import MLPNetwork, DoubleQFunction, TanhGaussianActor

    obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)
    act_space = gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
    device = torch.device("cpu")
    actor = TanhGaussianActor(obs_space, act_space, MLPNetwork, device=device, state_dependent_std=True, hidden_sizes=[32, 32])
    critic = DoubleQFunction(obs_space, act_space, MLPNetwork, hidden_sizes=[32, 32])
    target = DoubleQFunction(obs_space, act_space, MLPNetwork, hidden_sizes=[32, 32])
    agent = SACAgent(obs_space, act_space, device, actor, critic, target_critic=target, use_target=True)

    states = np.zeros((4, 12), dtype=np.float32)
    states[:, -1] = 1.0
    temperature = torch.tensor(10.0)
    torch.manual_seed(0)
    masked = agent.get_value(states, temperature, absorbing=torch.ones(4, 1))
    zero_q = agent.get_q_function_from_target(states, torch.zeros(4, 3))

    torch.testing.assert_close(masked, zero_q)
    print("SAC: absorbing target = Q(s_absorbing, 0) with no entropy term  OK")


if __name__ == "__main__":
    test_env_wrapper_extends_observations()
    test_policy_side_absorbing()
    test_buffer_roundtrip()
    test_nstep_stops_at_absorbing_and_not_across_envs()
    test_expert_side_absorbing()
    test_expert_wrap_scans_every_step()
    test_obs_norm_leaves_indicator_alone()
    test_expert_absorbing_matches_policy_under_obs_norm()
    test_sac_masks_absorbing()
    print("\nall absorbing-state checks passed")
