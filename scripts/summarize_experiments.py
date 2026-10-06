from __future__ import annotations

import argparse
import csv
from pathlib import Path


FIELDS = {
    "precision": "metrics/precision(B)",
    "recall": "metrics/recall(B)",
    "map50": "metrics/mAP50(B)",
    "map50_95": "metrics/mAP50-95(B)",
}


def best_row(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"No epochs found in {path}")
    return max(rows, key=lambda row: float(row[FIELDS["map50"]]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Ultralytics training CSV files.")
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output = []
    for path in args.inputs:
        row = best_row(path)
        output.append(
            {
                "run": path.parent.name,
                "best_epoch": int(float(row["epoch"])),
                "precision": round(float(row[FIELDS["precision"]]), 4),
                "recall": round(float(row[FIELDS["recall"]]), 4),
                "mAP50": round(float(row[FIELDS["map50"]]), 4),
                "mAP50-95": round(float(row[FIELDS["map50_95"]]), 4),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=output[0].keys())
        writer.writeheader()
        writer.writerows(output)
    print(f"Wrote {len(output)} runs to {args.output}")


if __name__ == "__main__":
    main()

