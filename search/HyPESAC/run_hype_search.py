"""Adapter for parallel_search.py: use the real hype.sh without copied defaults."""

import json
import subprocess
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[2]
    overrides = sys.argv[1:]
    log_dirs = [arg.split("=", 1)[1] for arg in overrides if arg.startswith("log_dir=")]
    status_file = Path(log_dirs[-1]) / "search_exit.json" if log_dirs else None
    if status_file is not None and not status_file.is_absolute():
        status_file = root / status_file
    if status_file is not None:
        status_file.parent.mkdir(parents=True, exist_ok=True)
        status_file.write_text(json.dumps({"returncode": None, "overrides": overrides}) + "\n")
    result = subprocess.run(["bash", str(root / "scripts/hype.sh"), *overrides], cwd=root)
    if status_file is not None:
        status_file.write_text(json.dumps({"returncode": result.returncode,
                                          "overrides": overrides}) + "\n")
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
