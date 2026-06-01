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
import os
import random
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from queue import Queue


SEARCH_SPACE: dict[str, list] = {
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


FIXED_OVERRIDES: list[str] = [
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


_print_lock = threading.Lock()


def _print(*args, **kwargs) -> None:
    with _print_lock:
        print(*args, **kwargs, flush=True)


def total_grid_size() -> int:
    n = 1
    for values in SEARCH_SPACE.values():
        n *= len(values)
    return n


def iter_configs():
    keys = list(SEARCH_SPACE.keys())
    value_lists = [SEARCH_SPACE[k] for k in keys]
    for values in itertools.product(*value_lists):
        yield dict(zip(keys, values))


def build_command(
    cfg: dict[str, object],
    run_dir: str,
    extra_overrides: list[str],
) -> list[str]:
    cmd: list[str] = ["python", "run_ppo.py"]
    cmd.extend(FIXED_OVERRIDES)
    for k, v in cfg.items():
        cmd.append(f"{k}={v}")
    cmd.append(f"hydra.run.dir={run_dir}")
    cmd.append(f"log_dir={run_dir}")
    cmd.extend(extra_overrides)
    return cmd


def run_one(
    idx: int,
    total: int,
    cmd: list[str],
    run_dir: str,
    gpu_queue: Queue | None,
    started_at: float,
    counter: dict,
    workers: int,
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
            proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
        rc = proc.returncode
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
    p.add_argument("--workers", type=int, default=16,
                   help="number of concurrent run_ppo.py processes (default: 16)")
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
    p.add_argument("--dry-run", action="store_true",
                   help="print the first few commands without running them")
    return p.parse_known_args()


def main() -> None:
    args, extra = parse_args()
    extra = list(extra)
    if extra and extra[0] == "--":
        extra = extra[1:]

    total = total_grid_size()
    _print(f"Full grid size: {total}")

    configs = list(iter_configs())
    if args.shuffle:
        random.Random(args.seed).shuffle(configs)
    if args.start:
        configs = configs[args.start:]
    if args.limit is not None:
        configs = configs[: args.limit]
    n = len(configs)
    _print(f"Will run: {n}  workers: {args.workers}")

    sweep_tag = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_log_dir = args.log_dir or f"outputs/Humanoid-v5-PPO-search/{sweep_tag}"
    Path(base_log_dir).mkdir(parents=True, exist_ok=True)
    _print(f"Sweep dir: {base_log_dir}")

    tasks: list[tuple[int, list[str], str]] = []
    for offset, cfg in enumerate(configs):
        idx = args.start + offset
        run_dir = f"{base_log_dir}/run_{idx:06d}"
        cmd = build_command(cfg, run_dir, extra)
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
            )
            for idx, cmd, run_dir in tasks
        ]
        try:
            for fut in as_completed(futures):
                fut.result()
        except KeyboardInterrupt:
            _print("KeyboardInterrupt received; cancelling pending tasks...")
            for fut in futures:
                fut.cancel()
            raise

    total_h = (time.time() - started_at) / 3600.0
    _print(f"All done. {n} runs in {total_h:.2f}h, sweep dir: {base_log_dir}")


if __name__ == "__main__":
    main()
