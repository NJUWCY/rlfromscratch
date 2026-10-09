"""Inherit candidate parameters and summarize local HyPE HalfCheetah sweeps."""

import argparse
import csv
import json
import math
from pathlib import Path

from omegaconf import OmegaConf


def inherit(args):
    source = Path(args.from_run).expanduser().resolve()
    cfg = OmegaConf.load(source / ".hydra/config.yaml")
    if cfg.env.name != "HalfCheetah-v5" or cfg.algorithm.name != "HyPESAC":
        raise ValueError("--from-run must be a HalfCheetah-v5 HyPESAC run")
    if cfg.expert_dataset.trajectory_num != 5 or cfg.expert_dataset.subsample_frequency != 1:
        raise ValueError("The source must use five complete expert trajectories")
    spec = json.loads(Path(args.spec).read_text())
    fixed = dict(item.split("=", 1) for item in spec["fixed_overrides"])
    # Carry model/optimizer and sampling settings, preserving the destination
    # budget, logging, seed grid and experiment identity. Resolve interpolations
    # before copying so a selected candidate has its actual numerical settings.
    values = OmegaConf.to_container(cfg.algorithm, resolve=True)

    def flatten(mapping, prefix):
        for key, value in mapping.items():
            name = f"{prefix}.{key}"
            if isinstance(value, dict):
                yield from flatten(value, name)
            else:
                yield name, value

    copied = dict(flatten(values, "algorithm"))
    for name in ("gamma", "update_step_per_epoch", "update_discriminator_step_per_epoch",
                 "discriminator_train_interval", "start_train_step", "env.obs_norm",
                 "env.scale", "env.absorbing", "expert_dataset.data_path"):
        copied[name] = OmegaConf.select(cfg, name)
    # Do not overwrite an axis being searched, or inherit the source's budget.
    for key, value in copied.items():
        if key not in spec["search_space"]:
            fixed[key] = json.dumps(value, separators=(",", ":"))
    # Existing fixed entries for a search axis are harmless: grid overrides come
    # last in parallel_search.build_command().
    spec["fixed_overrides"] = [f"{key}={value}" for key, value in fixed.items()]
    spec["source_run"] = str(source)
    Path(args.output).write_text(json.dumps(spec, indent=2) + "\n")
    print(f"Inherited candidate parameters from {source}")


def summarize(args):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    root = Path(args.directory).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Sweep directory not found: {root}")
    rows = []
    errors = []
    for config_file in sorted(root.rglob(".hydra/config.yaml")):
        run = config_file.parent.parent
        try:
            cfg = OmegaConf.load(config_file)
            if cfg.env.name != "HalfCheetah-v5" or cfg.algorithm.name != "HyPESAC":
                continue
            acc = EventAccumulator(str(run), size_guidance={"scalars": 0}).Reload()
            tags = set(acc.Tags().get("scalars", []))

            def series(tag):
                return acc.Scalars(tag) if tag in tags else []

            evaluations = series("test/episode_rewards_mean")
            # The epoch-zero evaluation is before any learning.
            trained_evals = [item for item in evaluations if item.step > 0]
            finite_evals = [item for item in trained_evals if math.isfinite(item.value)]
            tail = trained_evals[-args.tail:]
            tail_score = (sum(item.value for item in tail) / len(tail)
                          if tail and all(math.isfinite(item.value) for item in tail) else None)
            policy_steps = series("train/hybrid/policy_interaction_step")
            last_policy = int(policy_steps[-1].value) if policy_steps else 0
            budget = cfg.total_epoch * cfg.interact_per_epoch * cfg.env.num_training_envs
            metrics = {
                "max_q_loss": "train/critic/q_loss",
                "max_abs_q": "train/critic/q_value",
                "max_alpha": "train/temp/value",
                "last_expert_ratio": "train/hybrid/expert_ratio",
            }
            row = {
                "run_dir": str(run), "seed": cfg.seed,
                "budget_reached": last_policy >= budget,
                "policy_steps": last_policy, "policy_budget": budget,
                "evaluation_count": len(trained_evals), "tail_count": len(tail),
                "tail_mean_return": tail_score,
                "last_return": trained_evals[-1].value if trained_evals else None,
                "best_return": max((item.value for item in finite_evals), default=None),
                "last_evaluation_total_steps": trained_evals[-1].step if trained_evals else 0,
                "nonfinite_metrics": False,
            }
            for key, tag in metrics.items():
                events = series(tag)
                finite = [item.value for item in events if math.isfinite(item.value)]
                row["nonfinite_metrics"] |= any(not math.isfinite(item.value) for item in events)
                row[key] = (finite[-1] if key.startswith("last_") else max(abs(v) for v in finite)) if finite else None
            row["nonfinite_metrics"] |= any(not math.isfinite(item.value) for item in trained_evals)
            for key in ("algorithm.rl.critic_optimizer", "algorithm.rl.critic_lr",
                        "algorithm.rl.actor_optimizer", "algorithm.rl.actor_lr",
                        "algorithm.rl.temp_lr", "algorithm.rl.target_update_tau",
                        "update_step_per_epoch", "discriminator_train_interval",
                        "algorithm.discriminator.discriminator_train_steps",
                        "algorithm.sampling_schedule"):
                value = OmegaConf.select(cfg, key)
                row[key] = (json.dumps(OmegaConf.to_container(value), separators=(",", ":"))
                            if OmegaConf.is_config(value) else value)
            row["rankable"] = (row["budget_reached"] and len(tail) == args.tail
                               and tail_score is not None and not row["nonfinite_metrics"])
            rows.append(row)
        except Exception as exc:
            errors.append(f"{run}: {exc}")
    if not rows:
        raise RuntimeError("No readable HyPESAC runs found. " + "; ".join(errors))
    rows.sort(key=lambda row: (row["rankable"], row["tail_mean_return"]
              if row["tail_mean_return"] is not None else -math.inf), reverse=True)
    output = Path(args.output).expanduser().resolve() if args.output else root / "summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    eligible = [row for row in rows if row["rankable"]]
    print(f"Wrote {output}; {len(rows)} runs, {len(eligible)} completed finite candidates.")
    print(f"Ranking uses the last {args.tail} post-training evaluations; it does not use best_return.")
    for index, row in enumerate(eligible[:args.top], 1):
        print(f"{index:2d}. tail={row['tail_mean_return']:.2f}, last={row['last_return']:.2f}, "
              f"max_Q_loss={row['max_q_loss']}, seed={row['seed']}\n    {row['run_dir']}")
    if not eligible:
        print("No run is rankable yet; partial runs remain in the CSV for inspection.")
    for error in errors:
        print(f"Could not read {error}")
    if args.best_output:
        if not eligible:
            raise RuntimeError("No completed finite candidate has enough evaluations; stopping stage selection")
        best = Path(args.best_output).expanduser().resolve()
        best.parent.mkdir(parents=True, exist_ok=True)
        best.write_text(json.dumps(eligible[0], indent=2, allow_nan=False) + "\n")
        print(f"Selected candidate saved to {best}")
    if args.require_complete:
        if errors or len(eligible) != len(rows):
            raise RuntimeError("Some runs are incomplete, nonfinite or unreadable; confirmation did not pass")
        if len({row["seed"] for row in eligible}) != len(eligible):
            raise RuntimeError("Confirmation runs must use distinct seeds; a seed override may have collapsed the grid")
        scores = [row["tail_mean_return"] for row in eligible]
        mean = sum(scores) / len(scores)
        std = math.sqrt(sum((score - mean) ** 2 for score in scores) / len(scores))
        print(f"Confirmation tail return across {len(scores)} runs: {mean:.2f} +/- {std:.2f} (population std)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inherited = sub.add_parser("inherit")
    inherited.add_argument("--spec", required=True)
    inherited.add_argument("--from-run", required=True)
    inherited.add_argument("--output", required=True)
    report = sub.add_parser("summarize")
    report.add_argument("directory")
    report.add_argument("--tail", type=int, default=3)
    report.add_argument("--top", type=int, default=5)
    report.add_argument("--output")
    report.add_argument("--best-output", help="write the best eligible candidate to JSON; fail if none exists")
    report.add_argument("--require-complete", action="store_true",
                        help="fail unless every discovered run is eligible; summarize all validation seeds")
    args = parser.parse_args()
    if args.command == "inherit":
        inherit(args)
    else:
        if args.tail < 1 or args.top < 1:
            parser.error("--tail and --top must be positive")
        summarize(args)


if __name__ == "__main__":
    main()
