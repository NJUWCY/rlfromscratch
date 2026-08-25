"""Parallel parameter sweep launcher for PPO on Humanoid-v5.

Runs the same grid as the original ``ppo_humanoid_param_search.sh`` but
executes several ``run_ppo.py`` processes concurrently (default 16).

Typical usage
-------------
Run the full grid with 16 parallel jobs::

    python parallel_search.py

Run only 64 random configurations (very common in practice because the
full grid is huge)::

    python parallel_search.py --shuffle --limit 64

Assign one GPU per concurrent worker via ``CUDA_VISIBLE_DEVICES``::

    python parallel_search.py --workers 8 --gpus 0,1,2,3,4,5,6,7

Forward extra Hydra overrides to every run (e.g. seed)::

    python parallel_search.py seed=1 total_epoch=1000

Notes
-----
* Each run gets its own ``hydra.run.dir`` / ``log_dir`` under
  ``outputs/Humanoid-v5-PPO-search/<sweep-timestamp>/run_<idx>`` so that
  parallel jobs do not write into the same directory even if they start
  in the same second.
* stdout / stderr of every child process is captured into
  ``stdout.log`` inside that run directory.
* When ``--gpus`` is omitted the child processes inherit the parent's
  ``CUDA_VISIBLE_DEVICES``; if you have multiple GPUs and want to spread
  the load across them, pass ``--gpus 0,1,...``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import re
import signal
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from queue import Queue


# ---------------------------------------------------------------------------
# Built-in default sweep (PPO / Humanoid-v5).
#
# These act as the fallback when no ``--config`` file is supplied, so the old
# ``ppo_humanoid_param_search.sh`` keeps working unchanged. A search command
# can override any of them by passing ``--config <spec.json>`` (see
# ``load_spec`` below and ``scripts/sac_humanoid_param_search.sh`` for an
# example).
# ---------------------------------------------------------------------------

DEFAULT_RUN_SCRIPT = "run_ppo.py"
DEFAULT_SWEEP_NAME = "Humanoid-v5-PPO-search"

DEFAULT_SEARCH_SPACE: dict[str, list] = {
    "algorithm.update_epochs": [4, 6, 8],
    "algorithm.minibatch_size": [64, 128, 256],
    "algorithm.advan_norm": ["true", "false"],
    "algorithm.use_grad_clip": ["true", "false"],
    "algorithm.actor_lr": ["1e-4", "3e-4"],
    "algorithm.value_coef": [0.25, 0.5],
    "algorithm.entropy_coef": [0.0],
    "algorithm.eps_clip": [0.2],
    "algorithm.lambda_": [0.95],
    "algorithm.optimizer": ["Adam"],
    "algorithm.hidden_sizes": ["[256,256]"],
    "interact_per_epoch": [1024, 2048],
}


DEFAULT_FIXED_OVERRIDES: list[str] = [
    "algorithm=ppo",
    "total_epoch=2500",
    "train_log_interval=10",
    "save_interval=1000",
    "test_interval=50",
    "train_action_deterministic=false",
    "algorithm.rescale=true",
    "algorithm.collect_traj=false",
    "env.name=Humanoid-v5",
    "algorithm.buffer_name=ReplayBuffer",
]


class SearchSpec:
    """A fully-resolved description of one parameter sweep.

    Attributes
    ----------
    run_script:
        The training entry-point to invoke, e.g. ``run_sac.py``.
    search_space:
        Mapping of Hydra override key -> list of candidate values. The grid is
        the Cartesian product of all value lists.
    fixed_overrides:
        Hydra overrides applied verbatim to every run in the sweep.
    sweep_name:
        Human-readable tag used to build the default output directory
        ``outputs/<sweep_name>/<timestamp>``.
    """

    def __init__(
        self,
        run_script: str,
        search_space: dict[str, list],
        fixed_overrides: list[str],
        sweep_name: str,
    ) -> None:
        self.run_script = run_script
        self.search_space = search_space
        self.fixed_overrides = fixed_overrides
        self.sweep_name = sweep_name


def default_spec() -> SearchSpec:
    return SearchSpec(
        run_script=DEFAULT_RUN_SCRIPT,
        search_space=DEFAULT_SEARCH_SPACE,
        fixed_overrides=DEFAULT_FIXED_OVERRIDES,
        sweep_name=DEFAULT_SWEEP_NAME,
    )


def load_spec(config_path: str | None) -> SearchSpec:
    """Load a :class:`SearchSpec` from a JSON file, or fall back to defaults.

    The JSON schema is::

        {
          "run_script":   "run_sac.py",           # optional
          "sweep_name":   "Humanoid-v5-SAC-search", # optional
          "search_space": { "<key>": [<values>], ... },
          "fixed_overrides": ["algorithm=sac", ...]
        }

    ``search_space`` is required; everything else falls back to the PPO
    defaults so partial configs still work.
    """
    if not config_path:
        return default_spec()

    with open(config_path, "r") as f:
        data = json.load(f)

    if "search_space" not in data or not isinstance(data["search_space"], dict):
        raise ValueError(
            f"config {config_path!r} must contain a 'search_space' object"
        )

    search_space = {k: list(v) for k, v in data["search_space"].items()}
    return SearchSpec(
        run_script=data.get("run_script", DEFAULT_RUN_SCRIPT),
        search_space=search_space,
        fixed_overrides=list(data.get("fixed_overrides", DEFAULT_FIXED_OVERRIDES)),
        sweep_name=data.get("sweep_name", DEFAULT_SWEEP_NAME),
    )


_print_lock = threading.Lock()


def _print(*args, **kwargs) -> None:
    with _print_lock:
        print(*args, **kwargs, flush=True)


def total_grid_size(spec: SearchSpec) -> int:
    n = 1
    for values in spec.search_space.values():
        n *= len(values)
    return n


def iter_configs(spec: SearchSpec):
    keys = list(spec.search_space.keys())
    value_lists = [spec.search_space[k] for k in keys]
    for values in itertools.product(*value_lists):
        yield dict(zip(keys, values))


def _sanitize(text: str) -> str:
    """Make a string safe to embed in a directory name.

    Keeps letters, digits, dot, plus and minus; every other character
    (brackets, commas, spaces, slashes, ...) collapses to a single dash.
    """
    text = str(text)
    text = re.sub(r"[^0-9A-Za-z.+-]+", "-", text)
    return text.strip("-")


def config_dirname(cfg: dict[str, object]) -> str:
    """Encode a config as ``param1-value1_param2-value2`` for a run dir.

    Only the last dotted segment of each key is used (e.g.
    ``algorithm.actor_lr`` -> ``actor_lr``) to keep the name short while
    staying readable. Values are sanitized so brackets / commas in things
    like ``[256,256]`` don't break the path.
    """
    parts = []
    for key, value in cfg.items():
        short_key = key.rsplit(".", 1)[-1]
        parts.append(f"{_sanitize(short_key)}-{_sanitize(value)}")
    return "_".join(parts)


def build_command(
    spec: SearchSpec,
    cfg: dict[str, object],
    run_dir: str,
    extra_overrides: list[str],
) -> list[str]:
    cmd: list[str] = ["python", spec.run_script]
    cmd.extend(spec.fixed_overrides)
    for k, v in cfg.items():
        cmd.append(f"{k}={v}")
    cmd.append(f"hydra.run.dir={run_dir}")
    cmd.append(f"log_dir={run_dir}")
    cmd.extend(extra_overrides)
    return cmd


def _kill_process_group(proc: subprocess.Popen) -> None:
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    os.killpg(pgid, signal.SIGTERM)
    try:
        proc.wait(timeout=60)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def run_one(
    idx: int,
    total: int,
    cmd: list[str],
    run_dir: str,
    gpu_queue: Queue | None,
    started_at: float,
    counter: dict,
    workers: int,
    timeout: float | None = None,
) -> tuple[int, int, float]:
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    log_path = os.path.join(run_dir, "stdout.log")

    env = os.environ.copy()
    gpu = None
    if gpu_queue is not None:
        gpu = gpu_queue.get()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    _print(f"[start {idx:06d}/{total}] gpu={gpu} -> {run_dir}")
    started = time.time()
    try:
        with open(log_path, "w") as f:
            f.write(f"[{datetime.now().isoformat()}] CMD: {' '.join(cmd)}\n")
            f.write(f"CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', '')}\n\n")
            f.flush()
            # start_new_session so a stuck run can be killed together with the
            # env / dataloader subprocesses it spawned.
            proc = subprocess.Popen(
                cmd, stdout=f, stderr=subprocess.STDOUT, env=env, start_new_session=True
            )
            try:
                rc = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                _print(
                    f"[timeout {idx:06d}] exceeded {timeout / 3600.0:.2f}h, killing -> {run_dir}"
                )
                _kill_process_group(proc)
                rc = proc.wait()
    finally:
        if gpu_queue is not None and gpu is not None:
            gpu_queue.put(gpu)

    elapsed = time.time() - started
    with counter["lock"]:
        counter["done"] += 1
        done = counter["done"]

    wall = time.time() - started_at
    avg_wall = wall / done
    eta_h = (total - done) * avg_wall / max(1, workers) / 3600.0
    status = "OK" if rc == 0 else f"FAIL({rc})"
    _print(
        f"[{done}/{total}] run={idx:06d} {status} "
        f"elapsed={elapsed / 60.0:.1f}min "
        f"wall/run={avg_wall / 60.0:.1f}min "
        f"eta={eta_h:.1f}h log={log_path}"
    )
    return idx, rc, elapsed


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", type=str, default=None,
                   help="path to a JSON search spec (run_script / sweep_name / "
                        "search_space / fixed_overrides). When omitted the "
                        "built-in PPO/Humanoid-v5 grid is used.")
    p.add_argument("--workers", type=int, default=16,
                   help="number of concurrent training processes (default: 16)")
    p.add_argument("--gpus", type=str, default=None,
                   help="comma-separated GPU ids (e.g. '0,1,2,3'); "
                        "each running job is masked to exactly one of them "
                        "via CUDA_VISIBLE_DEVICES.")
    p.add_argument("--limit", type=int, default=None,
                   help="only run the first N configs (after --shuffle if set)")
    p.add_argument("--shuffle", action="store_true",
                   help="shuffle the configuration order; combine with "
                        "--limit for random search.")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed used for --shuffle (default: 0)")
    p.add_argument("--start", type=int, default=0,
                   help="skip the first N configs (useful for resuming)")
    p.add_argument("--log-dir", type=str, default=None,
                   help="output directory; defaults to "
                        "outputs/Humanoid-v5-PPO-search/<timestamp>")
    p.add_argument("--timeout-hours", type=float, default=None,
                   help="kill a run (and its children) after this many hours; "
                        "guards against jobs hanging in shutdown, e.g. when the "
                        "logger cannot reach its cloud backend.")
    p.add_argument("--dry-run", action="store_true",
                   help="print the first few commands without running them")
    return p.parse_known_args()


def main() -> None:
    args, extra = parse_args()
    extra = list(extra)
    if extra and extra[0] == "--":
        extra = extra[1:]

    spec = load_spec(args.config)
    _print(f"Run script: {spec.run_script}  sweep: {spec.sweep_name}")

    total = total_grid_size(spec)
    _print(f"Full grid size: {total}")

    configs = list(iter_configs(spec))
    if args.shuffle:
        random.Random(args.seed).shuffle(configs)
    if args.start:
        configs = configs[args.start:]
    if args.limit is not None:
        configs = configs[: args.limit]
    n = len(configs)
    _print(f"Will run: {n}  workers: {args.workers}")

    sweep_tag = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_log_dir = args.log_dir or f"outputs/{spec.sweep_name}/{sweep_tag}"
    Path(base_log_dir).mkdir(parents=True, exist_ok=True)
    _print(f"Sweep dir: {base_log_dir}")

    tasks: list[tuple[int, list[str], str]] = []
    seen_names: dict[str, int] = {}
    for offset, cfg in enumerate(configs):
        idx = args.start + offset
        name = config_dirname(cfg)
        # Guard against two configs sanitizing to the same string.
        dup = seen_names.get(name, 0)
        seen_names[name] = dup + 1
        if dup:
            name = f"{name}__{dup}"
        run_dir = f"{base_log_dir}/{name}"
        cmd = build_command(spec, cfg, run_dir, extra)
        tasks.append((idx, cmd, run_dir))

    if args.dry_run:
        for idx, cmd, _ in tasks[:5]:
            _print(f"[{idx:06d}] {' '.join(cmd)}")
        if len(tasks) > 5:
            _print(f"... (+{len(tasks) - 5} more)")
        return

    gpu_queue: Queue | None = None
    if args.gpus:
        gpu_queue = Queue()
        for g in args.gpus.split(","):
            g = g.strip()
            if g:
                gpu_queue.put(g)
        if gpu_queue.qsize() < args.workers:
            _print(
                f"WARNING: --gpus has {gpu_queue.qsize()} ids but --workers "
                f"is {args.workers}; effective concurrency = {gpu_queue.qsize()}"
            )

    counter = {"done": 0, "lock": threading.Lock()}
    started_at = time.time()

    failures: list[tuple[int, int]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(
                run_one,
                idx,
                n,
                cmd,
                run_dir,
                gpu_queue,
                started_at,
                counter,
                args.workers,
                args.timeout_hours * 3600.0 if args.timeout_hours else None,
            )
            for idx, cmd, run_dir in tasks
        ]
        try:
            for fut in as_completed(futures):
                idx, rc, _ = fut.result()
                if rc != 0:
                    failures.append((idx, rc))
        except KeyboardInterrupt:
            _print("KeyboardInterrupt received; cancelling pending tasks...")
            for fut in futures:
                fut.cancel()
            raise

    total_h = (time.time() - started_at) / 3600.0
    _print(f"All done. {n} runs in {total_h:.2f}h, sweep dir: {base_log_dir}")
    if failures:
        failure_text = ", ".join(
            f"run {idx:06d}: exit {rc}" for idx, rc in failures
        )
        _print(f"ERROR: {len(failures)} run(s) failed: {failure_text}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
