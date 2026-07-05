from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


BASELINE_ROOT = Path(__file__).resolve().parent
DEFAULT_TRAIN_SCRIPT = BASELINE_ROOT / "run_experiments.py"
PHYSICAL_CLASSES = ["chair", "desk", "sofa", "bed", "toilet"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Thin wrapper for five-class physical-parameter ablations. "
            "It calls run_experiments.py with chair/desk/sofa/bed/toilet only, "
            "leaving image2d for the separate six-class abnormal-input study."
        )
    )
    parser.add_argument("--runner", type=Path, default=DEFAULT_TRAIN_SCRIPT)
    parser.add_argument("runner_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.runner_args[:1] == ["--"]:
        args.runner_args = args.runner_args[1:]
    return args


def main() -> int:
    args = parse_args()
    command = [
        sys.executable,
        str(args.runner),
        "--classes",
        *PHYSICAL_CLASSES,
        *args.runner_args,
    ]
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=str(args.runner.parent), check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
