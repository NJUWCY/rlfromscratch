import argparse
from pathlib import Path
from typing import Sequence
import h5py

import numpy as np
import torch
from omegaconf import OmegaConf
from agent.expert_loader import build_expert_agent, load_policy_weights
from env.make_envs import make_vec_envs

from utils.utils import (
    RunningMeanStd,
    get_action_dim,
    get_best_device,
    set_seed,
    to_useful_action,
)


# Maps the ``--model`` choice to the checkpoint filename written during training
# (see the ``save_pth``/``best_pth`` arguments in the ``run_*.py`` entry points).
MODEL_FILENAMES = {"best": "best_model.pth", "newest": "newest_model.pth"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect complete expert trajectories from a trained policy."
    )
    # A run directory is the Hydra output folder of a single training run, e.g.
    # outputs/Hopper-v5-PPO/2026-07-12_10-21-28. It holds best_model.pth,
    # newest_model.pth, obs_rms.pth (only when obs_norm was used) and .hydra/config.yaml.
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--model", choices=sorted(MODEL_FILENAMES), default="best"
    )
    parser.add_argument("--num-testing-envs", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--collect-episodes", type=int, default=10)
    parser.add_argument("--min-reward", type=float, default=-np.inf)
    parser.add_argument("--min-length", type=int, default=0)
    parser.add_argument("--max-attempt-episodes", type=int, default=None)
    parser.add_argument("--config-path", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--test-epsilon", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.collect_episodes <= 0:
        parser.error("--collect-episodes must be positive")
    if args.min_length < 0:
        parser.error("--min-length must be non-negative")
    if args.num_testing_envs is not None and args.num_testing_envs <= 0:
        parser.error("--num-testing-envs must be positive")
    if args.max_attempt_episodes is None:
        args.max_attempt_episodes = 10 * args.collect_episodes
    if args.max_attempt_episodes < args.collect_episodes:
        parser.error("--max-attempt-episodes must be at least --collect-episodes")
    if args.output.suffix.lower() in (".npz", ".h5", ".hdf5"):
        args.output = args.output.with_suffix("")
    return args


def resolve_run_dir(run_dir: str | Path) -> Path:
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise NotADirectoryError(f"Run directory not found: {run_dir}")
    return run_dir


def resolve_checkpoint_path(run_dir: Path, model: str) -> Path:
    checkpoint = run_dir / MODEL_FILENAMES[model]
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    return checkpoint


def resolve_config_path(
    run_dir: Path,
    config_path: str | Path | None = None,
) -> Path:
    resolved = (
        Path(config_path)
        if config_path is not None
        else run_dir / ".hydra" / "config.yaml"
    )
    if not resolved.is_file():
        raise FileNotFoundError(f"Hydra config not found: {resolved}")
    return resolved


def load_obs_rms(run_dir: Path) -> RunningMeanStd:
    """Load the frozen training observation statistics saved next to the model."""
    obs_rms_path = run_dir / "obs_rms.pth"
    if not obs_rms_path.is_file():
        raise FileNotFoundError(
            f"obs_norm is enabled but obs_rms was not found: {obs_rms_path}"
        )
    state = torch.load(obs_rms_path, map_location="cpu", weights_only=False)
    obs_rms = RunningMeanStd()
    obs_rms.load_state_dict(state)
    return obs_rms


def _validate_config(cfg, config_path: Path) -> None:
    if OmegaConf.select(cfg, "algorithm.name") is None:
        raise ValueError(f"Config {config_path} is missing algorithm.name.")
    if OmegaConf.select(cfg, "env.name") is None:
        raise ValueError(f"Config {config_path} is missing env.name.")


def prepare_collection(args: argparse.Namespace):
    """Resolve paths, load config, build the eval env, and load the expert agent.

    Returns ``(cfg, env, agent, obs_rms, seed)``. Keeping this separate from ``main``
    lets the environment/agent setup be exercised on its own.
    """
    run_dir = resolve_run_dir(args.run_dir)
    checkpoint_path = resolve_checkpoint_path(run_dir, args.model)
    config_path = resolve_config_path(run_dir, args.config_path)

    cfg = OmegaConf.load(config_path)
    _validate_config(cfg, config_path)

    seed = (
        args.seed
        if args.seed is not None
        else int(OmegaConf.select(cfg, "seed", default=0))
    )
    set_seed(seed)
    device = get_best_device()

    # Override the number of parallel eval envs with this collection run's setting
    # before the vectorized env is created (make_vec_envs reads num_testing_envs).
    if args.num_testing_envs is not None:
        cfg.env.num_testing_envs = args.num_testing_envs

    # Load obs_rms if the training run used obs_norm. The env itself is created
    # WITHOUT VecObsNorm so that env.reset()/env.step() return raw observations.
    # Normalization for the agent is applied manually via to_agent_obs in the
    # collection loop.
    obs_norm = bool(OmegaConf.select(cfg, "env.obs_norm", default=False))
    obs_rms = load_obs_rms(run_dir) if obs_norm else None

    env_cfg = OmegaConf.merge(cfg.env, {"obs_norm": False})
    env = make_vec_envs(env_cfg, is_training=False, seed=seed)

    # build_expert_agent refuses obs_norm=true because it assumes the statistics
    # are unavailable. Since we already disabled VecObsNorm above, pass the config
    # with obs_norm=False so the check passes.
    agent_cfg = OmegaConf.merge(cfg, {"env": {"obs_norm": False}})
    agent = build_expert_agent(
        agent_cfg,
        env.observation_space,
        env.action_space,
        device,
    )
    load_policy_weights(agent, checkpoint_path)
    agent.eval()
    return cfg, env, agent, obs_rms, seed


def collect_expert_trajectories(
    env,
    agent,
    *,
    obs_rms: RunningMeanStd | None = None,
    collect_episodes: int = 10,
    min_reward: float = -np.inf,
    min_length: int = 0,
    max_attempt_episodes: int | None = None,
    test_epsilon: float = 0.0,
) -> dict:
    """Roll out complete episodes and keep those passing the reward/length filters.

    Observations are stored **un-normalized** (raw env output). When ``obs_rms`` is
    provided the agent still receives normalized observations for correct inference,
    but raw observations are what get recorded into the dataset.

    Returns a dict of per-episode lists matching the D4RL convention
    (observations, next_observations, actions, rewards, terminals, timeouts, infos).
    """
    if max_attempt_episodes is None:
        max_attempt_episodes = 10 * collect_episodes

    num_envs = env.num_envs
    action_dim = get_action_dim(env.action_space)

    def to_agent_obs(raw_obs: np.ndarray) -> np.ndarray:
        """Normalize raw observations for the agent (trained with obs_norm)."""
        if obs_rms is not None:
            return obs_rms.norm(raw_obs).astype(np.float32)
        return raw_obs

    observations = [[] for _ in range(num_envs)]
    next_observations = [[] for _ in range(num_envs)]
    actions = [[] for _ in range(num_envs)]
    rewards = [[] for _ in range(num_envs)]
    terminals = [[] for _ in range(num_envs)]
    timeouts = [[] for _ in range(num_envs)]

    episodes: dict[str, list] = {
        "observations": [],
        "next_observations": [],
        "actions": [],
        "rewards": [],
        "terminals": [],
        "timeouts": []
    }
    attempts = 0

    observation = env.reset()
    while len(episodes["rewards"]) < collect_episodes and attempts < max_attempt_episodes:
        agent_obs = to_agent_obs(observation)
        if np.random.rand() < test_epsilon:
            action = np.array(
                [[env.action_space.sample()] for _ in range(num_envs)],
                dtype=env.action_space.dtype,
            )
        else:
            with torch.no_grad():
                action, _ = agent.select_action(agent_obs, deterministic=True)
        action = to_useful_action(env.action_space, action_dim, action)
        next_observation, step_rewards, dones, infos = env.step(action)

        for i in range(num_envs):
            is_truncated = infos[i].get("truncated", False)
            is_terminated = bool(dones[i]) and not is_truncated

            observations[i].append(np.asarray(observation[i]))
            if is_truncated and "truncated_observation" in infos[i]:
                next_obs = np.asarray(infos[i]["truncated_observation"])
            else:
                next_obs = np.asarray(next_observation[i])
            next_observations[i].append(next_obs)
            actions[i].append(np.asarray(action[i]))
            rewards[i].append(float(step_rewards[i]))
            terminals[i].append(is_terminated)
            timeouts[i].append(is_truncated)

            if "episode" not in infos[i]:
                continue

            attempts += 1
            ep_obs = np.asarray(observations[i])
            ep_next_obs = np.asarray(next_observations[i])
            ep_actions = np.asarray(actions[i])
            ep_rewards = np.asarray(rewards[i], dtype=np.float32)
            ep_terminals = np.asarray(terminals[i], dtype=np.bool_)
            ep_timeouts = np.asarray(timeouts[i], dtype=np.bool_)
            observations[i], actions[i], rewards[i] = [], [], []
            next_observations[i] = []
            terminals[i], timeouts[i] = [], []

            episode_return = float(infos[i]["episode"].get("r", ep_rewards.sum()))
            ep_len = len(ep_rewards)
            accepted = (
                len(episodes["rewards"]) < collect_episodes
                and episode_return >= min_reward
                and ep_len >= min_length
            )
            status = "accepted" if accepted else "rejected"
            print(
                f"[Episode {attempts}] return={episode_return:.2f}, "
                f"length={ep_len}, {status} "
                f"({len(episodes['rewards']) + int(accepted)}/{collect_episodes})"
            )
            if accepted:
                episodes["observations"].append(ep_obs)
                episodes["next_observations"].append(ep_next_obs)
                episodes["actions"].append(ep_actions)
                episodes["rewards"].append(ep_rewards)
                episodes["terminals"].append(ep_terminals)
                episodes["timeouts"].append(ep_timeouts)

        observation = next_observation

    if len(episodes["rewards"]) < collect_episodes:
        raise RuntimeError(
            f"Collected only {len(episodes['rewards'])}/{collect_episodes} episodes "
            f"after {attempts} attempts. Loosen --min-reward/--min-length or raise "
            f"--max-attempt-episodes."
        )
    return episodes


def save_trajectories(
    data: dict,
    output: str | Path,
    *,
    obs_rms: RunningMeanStd | None = None,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Flatten per-episode data into D4RL-style flat arrays and save as .npz and .h5.

    ``output`` should be a stem (no extension); both ``<stem>.npz`` and
    ``<stem>.h5`` are written.  obs_rms is saved as ``<stem>_obs_rms.pth``.
    """
    base = Path(output).with_suffix("")
    npz_path = base.with_suffix(".npz")
    h5_path = base.with_suffix(".hdf5")

    for p in (npz_path, h5_path):
        if p.exists() and not overwrite:
            raise FileExistsError(f"Output file already exists: {p}")

    base.parent.mkdir(parents=True, exist_ok=True)

    flat = {
        "observations": np.concatenate(data["observations"]),
        "next_observations": np.concatenate(data["next_observations"]),
        "actions": np.concatenate(data["actions"]),
        "rewards": np.concatenate(data["rewards"]),
        "terminals": np.concatenate(data["terminals"]),
        "timeouts": np.concatenate(data["timeouts"]),
    }

    np.savez_compressed(npz_path, **flat)

    with h5py.File(h5_path, "w") as f:
        for key, arr in flat.items():
            f.create_dataset(key, data=arr, compression="gzip")

    if obs_rms is not None:
        obs_rms_path = base.with_name(f"{base.stem}_obs_rms.pth")
        torch.save(obs_rms.state_dict(), obs_rms_path)

    return npz_path, h5_path


def main(argv: Sequence[str] | None = None) -> Path:
    args = parse_args(argv)
    base = args.output.with_suffix("")
    for p in (base.with_suffix(".npz"), base.with_suffix(".h5")):
        if p.exists() and not args.overwrite:
            raise FileExistsError(f"Output file already exists: {p}")

    cfg, env, agent, obs_rms, _ = prepare_collection(args)

    try:
        data = collect_expert_trajectories(
            env,
            agent,
            obs_rms=obs_rms,
            collect_episodes=args.collect_episodes,
            min_reward=args.min_reward,
            min_length=args.min_length,
            max_attempt_episodes=args.max_attempt_episodes,
            test_epsilon=args.test_epsilon,
        )
    finally:
        env.close()

    npz_path, h5_path = save_trajectories(
        data,
        args.output,
        obs_rms=obs_rms,
        overwrite=args.overwrite,
    )

    episode_rewards = [float(rewards.sum()) for rewards in data["rewards"]]
    episode_lengths = [len(rewards) for rewards in data["rewards"]]
    print(
        f"Saved {len(episode_rewards)} expert trajectories → "
        f"{npz_path} and {h5_path}. "
        f"Mean reward: {np.mean(episode_rewards):.3f}; "
        f"mean length: {np.mean(episode_lengths):.1f}."
    )
    return npz_path


if __name__ == "__main__":
    main()
