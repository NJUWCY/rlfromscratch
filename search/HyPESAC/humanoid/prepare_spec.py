"""Prepare a Humanoid sweep by inheriting a selected run's hyperparameters."""
import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf


def flatten(mapping, prefix=""):
    for key, value in mapping.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            yield from flatten(value, name)
        else:
            yield name, value


def prepare(spec_path, output, source_run=None):
    spec = json.loads(Path(spec_path).read_text())
    if spec.get("requires_source") and not source_run:
        raise ValueError(
            "This stage requires --from-run <selected-run-directory>. "
            "Select a configuration using all three seeds, not one peak return."
        )
    if source_run:
        source = Path(source_run).expanduser().resolve()
        cfg = OmegaConf.load(source / ".hydra/config.yaml")
        if cfg.env.name != "Humanoid-v5" or cfg.algorithm.name != "HyPESAC":
            raise ValueError("--from-run must be a Humanoid-v5 HyPESAC run")
        if cfg.expert_dataset.trajectory_num != 5 or cfg.expert_dataset.subsample_frequency != 1:
            raise ValueError("The selected run must use five complete expert trajectories")
        # Saved standalone runs can contain a Hydra 'now' interpolation here.
        cfg.log_dir = str(source)
        values = OmegaConf.to_container(cfg, resolve=True)
        # Inherit hyperparameters, not run identity or random seed. Each new
        # training starts from scratch; this does not restore model weights.
        for key in ("log_dir", "hydra_base_dir", "experiment_name", "seed", "hydra"):
            values.pop(key, None)
        fixed = {
            key: json.dumps(value, separators=(",", ":"))
            for key, value in flatten(values)
            if key not in spec["search_space"]
        }
        # Stage-specific fixed overrides take precedence over inherited ones;
        # grid values and user CLI overrides are appended later by the launcher.
        fixed.update(item.split("=", 1) for item in spec.get("fixed_overrides", []))
        spec["fixed_overrides"] = [f"{key}={value}" for key, value in fixed.items()]
        spec["source_run"] = str(source)
        print(f"Inherited hyperparameters from {source}; starting fresh models.")
    Path(output).write_text(json.dumps(spec, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--from-run")
    args = parser.parse_args()
    try:
        prepare(args.spec, args.output, args.from_run)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
