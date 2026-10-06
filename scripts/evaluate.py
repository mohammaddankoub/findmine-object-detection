from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a FindMine checkpoint on a YOLO split.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    for path in (args.weights, args.data):
        if not path.is_file():
            parser.error(f"File not found: {path}")

    metrics = YOLO(str(args.weights)).val(
        data=str(args.data), split=args.split, imgsz=args.imgsz, device=args.device
    )
    print(f"precision: {metrics.box.mp:.4f}")
    print(f"recall:    {metrics.box.mr:.4f}")
    print(f"mAP50:     {metrics.box.map50:.4f}")
    print(f"mAP50-95:  {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()

