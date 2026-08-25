import argparse
from pathlib import Path
from typing import Callable, Sequence
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


# Maps the ``--model`` choice to checkpoint / obs_rms filenames written by
# ``BaseAlgorithm.save(pre_fix=...)`` (e.g. best_model.pth, best_obs_rms.pth).
MODEL_FILENAMES = {"best": "best_model.pth", "newest": "newest_model.pth"}
OBS_RMS_FILENAMES = {"best": "best_obs_rms.pth", "newest": "newest_obs_rms.pth"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect complete expert trajectories from a trained policy."
    )
    # A run directory is the Hydra output folder of a single training run, e.g.
    # outputs/Hopper-v5-PPO/2026-07-12_10-21-28. It holds best_model.pth /
    # newest_model.pth, matching *_obs_rms.pth when obs_norm was used, and
    # .hydra/config.yaml.
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--model", choices=sorted(MODEL_FILENAMES), default="best"
    )
    parser.add_argument("--num-testing-envs", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--collect-episodes", type=int, default=10)
    parser.add_argument(
        "--save-every",
        type=int,
        default=0,
        help=(
            "Append accepted trajectories to one HDF5 file every N episodes. "
            "Use 0 to keep the original one-shot NPZ and HDF5 saving behavior."
        ),
    )
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
    if args.save_every < 0:
        parser.error("--save-every must be non-negative")
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


def resolve_obs_rms_path(run_dir: Path, model: str) -> Path:
    """Resolve obs_rms path for the selected checkpoint (best/newest)."""
    return run_dir / OBS_RMS_FILENAMES[model]


def load_obs_rms(run_dir: Path, model: str) -> RunningMeanStd:
    """Load the frozen training observation statistics matching ``--model``."""
    obs_rms_path = resolve_obs_rms_path(run_dir, model)
    if not obs_rms_path.is_file():
        raise FileNotFoundError(
            f"obs_norm is enabled but obs_rms was not found for model={model!r}: "
            f"{obs_rms_path}"
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
    obs_rms = load_obs_rms(run_dir, args.model) if obs_norm else None

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


REQUIRED_DATA_KEYS = (
    "observations",
    "next_observations",
    "actions",
    "rewards",
    "terminals",
    "timeouts",
    "episode_ends",
)
OPTIONAL_DATA_KEYS = (
    "infos/action_log_probs",
    "infos/qpos",
    "infos/qvel",
)
HDF5_DTYPES = {
    "observations": np.float32,
    "next_observations": np.float32,
    "actions": np.float32,
    "rewards": np.float32,
    "terminals": np.bool_,
    "timeouts": np.bool_,
    "episode_ends": np.bool_,
    "infos/action_log_probs": np.float32,
    "infos/qpos": np.float64,
    "infos/qvel": np.float64,
}


def make_episode_buffer() -> dict[str, list]:
    return {key: [] for key in REQUIRED_DATA_KEYS + OPTIONAL_DATA_KEYS}


def drop_empty_optional_keys(data: dict[str, list]) -> dict[str, list]:
    return {
        key: value
        for key, value in data.items()
        if key not in OPTIONAL_DATA_KEYS or value
    }


def validate_episode_ends(
    episode_ends: np.ndarray,
    transition_count: int,
    expected_episodes: int,
) -> None:
    """Validate complete-episode boundary metadata before marking data saved."""
    episode_end_mask = np.asarray(episode_ends, dtype=np.bool_).reshape(-1)

    if episode_end_mask.shape[0] != transition_count:
        raise ValueError(
            "episode_ends length does not match the number of transitions."
        )

    actual_episodes = int(np.count_nonzero(episode_end_mask))
    if actual_episodes != expected_episodes:
        raise ValueError(
            f"Expected {expected_episodes} complete episodes, "
            f"but episode_ends contains {actual_episodes}."
        )

    if episode_end_mask.size == 0 or not episode_end_mask[-1]:
        raise ValueError(
            "The last transition must be marked as an episode end."
        )


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
    save_every: int = 0,
    on_chunk: Callable[[dict, int, int], None] | None = None,
) -> dict:
    """Roll out complete episodes and keep those passing the reward/length filters.

    Observations are stored **un-normalized** (raw env output). When ``obs_rms`` is
    provided the agent still receives normalized observations for correct inference,
    but raw observations are what get recorded into the dataset.

    Without ``on_chunk``, returns the original dict of per-episode D4RL fields.
    With ``on_chunk``, every ``save_every`` accepted episodes are passed to the
    callback and released; the return value then only contains reward/length
    summaries, so the complete trajectories are never retained in memory.
    """
    if max_attempt_episodes is None:
        max_attempt_episodes = 10 * collect_episodes
    if save_every < 0:
        raise ValueError("save_every must be non-negative")
    if (save_every > 0) != (on_chunk is not None):
        raise ValueError("save_every and on_chunk must be enabled together")

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
    action_log_probs = [[] for _ in range(num_envs)]
    qpos_list = [[] for _ in range(num_envs)]
    qvel_list = [[] for _ in range(num_envs)]
    # Track whether optional fields appeared at least once (D4RL infos/*).
    saw_log_probs = False
    saw_qpos = False
    saw_qvel = False

    episodes = make_episode_buffer()
    attempts = 0
    accepted_count = 0
    accepted_rewards: list[float] = []
    accepted_lengths: list[int] = []

    observation = env.reset()
    while accepted_count < collect_episodes and attempts < max_attempt_episodes:
        agent_obs = to_agent_obs(observation)
        action_info: dict = {}
        if np.random.rand() < test_epsilon:
            action = np.array(
                [[env.action_space.sample()] for _ in range(num_envs)],
                dtype=env.action_space.dtype,
            )
        else:
            with torch.no_grad():
                action, action_info = agent.select_action(agent_obs, deterministic=True)
        action = to_useful_action(env.action_space, action_dim, action)
        next_observation, step_rewards, dones, infos = env.step(action)

        log_probs = action_info.get("log_probs") if action_info else None
        if log_probs is not None:
            saw_log_probs = True
            log_probs = np.asarray(log_probs).reshape(num_envs)

        for i in range(num_envs):
            is_truncated = infos[i].get("truncated", False)
            is_terminated = bool(dones[i]) and not is_truncated

            ##if is_truncated and "truncated_observation" in infos[i]:
              ##  next_obs = np.asarray(infos[i]["truncated_observation"])
            ##else:
              ##  next_obs = np.asarray(next_observation[i])

            observations[i].append(np.asarray(observation[i]))

            if bool(dones[i]) and "terminal_observation" in infos[i]:
                next_obs = np.asarray(
                    infos[i]["terminal_observation"]
                )
            elif is_truncated and "truncated_observation" in infos[i]:
                next_obs = np.asarray(
                    infos[i]["truncated_observation"]
                )
            else:
                next_obs = np.asarray(next_observation[i])

            next_observations[i].append(next_obs)

            actions[i].append(np.asarray(action[i]))
            rewards[i].append(float(step_rewards[i]))
            terminals[i].append(is_terminated)
            timeouts[i].append(is_truncated)

            if log_probs is not None:
                action_log_probs[i].append(float(log_probs[i]))
            elif saw_log_probs:
                action_log_probs[i].append(np.nan)

            if "qpos" in infos[i]:
                saw_qpos = True
                qpos_list[i].append(np.asarray(infos[i]["qpos"], dtype=np.float64))
            elif saw_qpos:
                raise RuntimeError("qpos missing from infos after it was previously present")

            if "qvel" in infos[i]:
                saw_qvel = True
                qvel_list[i].append(np.asarray(infos[i]["qvel"], dtype=np.float64))
            elif saw_qvel:
                raise RuntimeError("qvel missing from infos after it was previously present")

            if "episode" not in infos[i]:
                continue

            attempts += 1
            ep_obs = np.asarray(observations[i])
            ep_next_obs = np.asarray(next_observations[i])
            ep_actions = np.asarray(actions[i])
            ep_rewards = np.asarray(rewards[i], dtype=np.float32)
            ep_terminals = np.asarray(terminals[i], dtype=np.bool_)
            ep_timeouts = np.asarray(timeouts[i], dtype=np.bool_)
            ep_log_probs = (
                np.asarray(action_log_probs[i], dtype=np.float32)
                if action_log_probs[i]
                else None
            )
            ep_qpos = np.asarray(qpos_list[i], dtype=np.float64) if qpos_list[i] else None
            ep_qvel = np.asarray(qvel_list[i], dtype=np.float64) if qvel_list[i] else None

            observations[i], actions[i], rewards[i] = [], [], []
            next_observations[i] = []
            terminals[i], timeouts[i] = [], []
            action_log_probs[i], qpos_list[i], qvel_list[i] = [], [], []

            episode_return = float(infos[i]["episode"].get("r", ep_rewards.sum()))
            ep_len = len(ep_rewards)
            accepted = (
                accepted_count < collect_episodes
                and episode_return >= min_reward
                and ep_len >= min_length
            )
            if accepted:
                # Only the last transition of a complete game is an episode boundary.
                ep_episode_ends = np.zeros(ep_len, dtype=np.bool_)
                ep_episode_ends[-1] = True

                episodes["observations"].append(ep_obs)
                episodes["next_observations"].append(ep_next_obs)
                episodes["actions"].append(ep_actions)
                episodes["rewards"].append(ep_rewards)
                episodes["terminals"].append(ep_terminals)
                episodes["timeouts"].append(ep_timeouts)
                episodes["episode_ends"].append(ep_episode_ends)
                if ep_log_probs is not None:
                    episodes["infos/action_log_probs"].append(ep_log_probs)
                if ep_qpos is not None:
                    episodes["infos/qpos"].append(ep_qpos)
                if ep_qvel is not None:
                    episodes["infos/qvel"].append(ep_qvel)

                accepted_count += 1
                accepted_rewards.append(float(ep_rewards.sum()))
                accepted_lengths.append(ep_len)

            status = "accepted" if accepted else "rejected"
            print(
                f"[Episode {attempts}] return={episode_return:.2f}, "
                f"length={ep_len}, {status} "
                f"({accepted_count}/{collect_episodes})"
            )

            should_save = accepted and on_chunk is not None and (
                len(episodes["rewards"]) >= save_every
                or accepted_count == collect_episodes
            )
            if should_save:
                chunk_start = accepted_count - len(episodes["rewards"]) + 1
                chunk_end = accepted_count
                on_chunk(
                    drop_empty_optional_keys(episodes),
                    chunk_start,
                    chunk_end,
                )
                episodes = make_episode_buffer()

        observation = next_observation

    if accepted_count < collect_episodes:
        raise RuntimeError(
            f"Collected only {accepted_count}/{collect_episodes} episodes "
            f"after {attempts} attempts. Loosen --min-reward/--min-length or raise "
            f"--max-attempt-episodes."
        )
    if on_chunk is not None:
        return {
            "episode_rewards": accepted_rewards,
            "episode_lengths": accepted_lengths,
        }
    # Drop empty optional infos keys so save only writes fields that were collected.
    return drop_empty_optional_keys(episodes)


def append_trajectories_to_hdf5(
    data: dict[str, list],
    output: str | Path,
    *,
    first_chunk: bool,
    overwrite: bool,
    saved_episodes: int,
    target_episodes: int,
) -> Path:
    """Append one trajectory chunk to a resizable D4RL-style HDF5 file."""
    base = Path(output).with_suffix("")
    h5_path = base.with_suffix(".hdf5")
    base.parent.mkdir(parents=True, exist_ok=True)

    if first_chunk and h5_path.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists: {h5_path}")
    if not first_chunk and not h5_path.is_file():
        raise FileNotFoundError(f"Streaming HDF5 file disappeared: {h5_path}")

    mode = "w" if first_chunk else "a"
    with h5py.File(h5_path, mode) as f:
        for key, episode_arrays in data.items():
            if not episode_arrays:
                continue

            array = np.concatenate(episode_arrays).astype(
                HDF5_DTYPES[key],
                copy=False,
            )

            if "/" in key:
                group_name, dataset_name = key.rsplit("/", 1)
                group = f.require_group(group_name)
            else:
                group = f
                dataset_name = key

            if dataset_name not in group:
                dataset = group.create_dataset(
                    dataset_name,
                    shape=(0, *array.shape[1:]),
                    maxshape=(None, *array.shape[1:]),
                    dtype=array.dtype,
                    chunks=True,
                    compression="gzip",
                )
            else:
                dataset = group[dataset_name]
                if dataset.shape[1:] != array.shape[1:]:
                    raise ValueError(
                        f"Shape mismatch for {key}: existing={dataset.shape[1:]}, "
                        f"new={array.shape[1:]}"
                    )
                if dataset.dtype != array.dtype:
                    raise ValueError(
                        f"Dtype mismatch for {key}: existing={dataset.dtype}, "
                        f"new={array.dtype}"
                    )

            old_size = dataset.shape[0]
            dataset.resize(old_size + array.shape[0], axis=0)
            dataset[old_size:] = array

        f.attrs["saved_episodes"] = saved_episodes
        f.attrs["target_episodes"] = target_episodes
        f.attrs["complete"] = False
        f.flush()

    return h5_path


def save_trajectories(
    data: dict,
    output: str | Path,
    *,
    obs_rms: RunningMeanStd | None = None,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Flatten per-episode data into D4RL-style flat arrays and save as .npz and .hdf5.

    ``output`` should be a stem (no extension); both ``<stem>.npz`` and
    ``<stem>.hdf5`` are written.  obs_rms is saved as ``<stem>_obs_rms.pth``.

    HDF5 layout matches D4RL (e.g. ``hopper_medium-v2.hdf5``):
    top-level ``observations`` / ``actions`` / ``rewards`` / ``terminals`` /
    ``timeouts`` / ``episode_ends`` / ``next_observations``, plus optional
    ``infos/action_log_probs``, ``infos/qpos``, ``infos/qvel``.
    """
    base = Path(output).with_suffix("")
    npz_path = base.with_suffix(".npz")
    h5_path = base.with_suffix(".hdf5")

    for p in (npz_path, h5_path):
        if p.exists() and not overwrite:
            raise FileExistsError(f"Output file already exists: {p}")

    base.parent.mkdir(parents=True, exist_ok=True)

    flat = {
        "observations": np.concatenate(data["observations"]).astype(np.float32),
        "next_observations": np.concatenate(data["next_observations"]).astype(np.float32),
        "actions": np.concatenate(data["actions"]).astype(np.float32),
        "rewards": np.concatenate(data["rewards"]).astype(np.float32),
        "terminals": np.concatenate(data["terminals"]).astype(np.bool_),
        "timeouts": np.concatenate(data["timeouts"]).astype(np.bool_),
        "episode_ends": np.concatenate(data["episode_ends"]).astype(np.bool_),
    }

    validate_episode_ends(
        flat["episode_ends"],
        transition_count=flat["rewards"].shape[0],
        expected_episodes=len(data["rewards"]),
    )

    if "infos/action_log_probs" in data:
        flat["infos/action_log_probs"] = np.concatenate(
            data["infos/action_log_probs"]
        ).astype(np.float32)
    if "infos/qpos" in data:
        flat["infos/qpos"] = np.concatenate(data["infos/qpos"]).astype(np.float64)
    if "infos/qvel" in data:
        flat["infos/qvel"] = np.concatenate(data["infos/qvel"]).astype(np.float64)

    # np.savez does not allow '/' in keys; use '__' for nested infos fields.
    npz_flat = {k.replace("/", "__"): v for k, v in flat.items()}
    np.savez_compressed(npz_path, **npz_flat)

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
    for p in (base.with_suffix(".npz"), base.with_suffix(".hdf5"), base.with_suffix(".h5")):
        if p.exists() and not args.overwrite:
            raise FileExistsError(f"Output file already exists: {p}")

    cfg, env, agent, obs_rms, _ = prepare_collection(args)

    streaming = args.save_every > 0
    stream_h5_path = base.with_suffix(".hdf5")
    first_chunk = True

    def save_chunk(chunk: dict, chunk_start: int, chunk_end: int) -> None:
        nonlocal first_chunk
        append_trajectories_to_hdf5(
            chunk,
            args.output,
            first_chunk=first_chunk,
            overwrite=args.overwrite,
            saved_episodes=chunk_end,
            target_episodes=args.collect_episodes,
        )
        if first_chunk and obs_rms is not None:
            obs_rms_path = base.with_name(f"{base.stem}_obs_rms.pth")
            torch.save(obs_rms.state_dict(), obs_rms_path)
        first_chunk = False
        print(
            f"Saved expert trajectories {chunk_start}-{chunk_end} "
            f"to {stream_h5_path}."
        )

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
            save_every=args.save_every,
            on_chunk=save_chunk if streaming else None,
        )
    finally:
        env.close()

    if streaming:
        with h5py.File(stream_h5_path, "a") as f:
            validate_episode_ends(
                f["episode_ends"][:],
                transition_count=f["rewards"].shape[0],
                expected_episodes=args.collect_episodes,
            )

            f.attrs["saved_episodes"] = args.collect_episodes
            f.attrs["target_episodes"] = args.collect_episodes
            f.attrs["complete"] = True
            f.flush()

        episode_rewards = data["episode_rewards"]
        episode_lengths = data["episode_lengths"]
        print(
            f"Saved {len(episode_rewards)} expert trajectories to "
            f"{stream_h5_path}. Mean reward: {np.mean(episode_rewards):.3f}; "
            f"mean length: {np.mean(episode_lengths):.1f}. "
            "Streaming mode writes HDF5 only."
        )
        return stream_h5_path

    npz_path, h5_path = save_trajectories(
        data,
        args.output,
        obs_rms=obs_rms,
        overwrite=args.overwrite,
    )

    episode_rewards = [float(rewards.sum()) for rewards in data["rewards"]]
    episode_lengths = [len(rewards) for rewards in data["rewards"]]
    info_keys = [k for k in data if k.startswith("infos/")]
    print(
        f"Saved {len(episode_rewards)} expert trajectories → "
        f"{npz_path} and {h5_path}. "
        f"Mean reward: {np.mean(episode_rewards):.3f}; "
        f"mean length: {np.mean(episode_lengths):.1f}. "
        f"infos fields: {info_keys or 'none'}."
    )
    return npz_path


if __name__ == "__main__":
    main()
