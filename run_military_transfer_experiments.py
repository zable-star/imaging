from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


BASELINE_ROOT = Path(__file__).resolve().parent
DEFAULT_RUNNER = BASELINE_ROOT / "run_experiments.py"
DEFAULT_MILITARY_CLASSES = ["tank", "aircraft", "helicopter", "military_truck", "missile_vehicle"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Thin wrapper for small-sample military transfer experiments. "
            "It forwards a military class list and all remaining arguments to run_experiments.py."
        )
    )
    parser.add_argument("--runner", type=Path, default=DEFAULT_RUNNER)
    parser.add_argument(
        "--classes",
        nargs="+",
        default=DEFAULT_MILITARY_CLASSES,
        help="Military class folder names under the selected dataset root.",
    )
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
        *args.classes,
        *args.runner_args,
    ]
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=str(args.runner.parent), check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
