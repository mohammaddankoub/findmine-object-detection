from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from findmine.grouped_split import assign_groups, collect_samples, copy_split, split_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild a YOLO dataset while keeping near-duplicate scenes together."
    )
    parser.add_argument("source", type=Path, help="Existing YOLO dataset root")
    parser.add_argument("destination", type=Path, help="New dataset root")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    samples = collect_samples(args.source)
    assignment = assign_groups(list(samples), seed=args.seed)
    copy_split(samples, assignment, args.destination)

    print(f"Created {args.destination}")
    for row in split_report(samples, assignment):
        print(
            "{split:5s}: images={images:3d}, backgrounds={backgrounds:2d}, "
            "Wallet={wallet_boxes:3d}, Travel_Mug={travel_mug_boxes:3d}".format(**row)
        )


if __name__ == "__main__":
    main()

