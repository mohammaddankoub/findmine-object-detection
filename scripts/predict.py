from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FindMine inference on images or video.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--source", required=True, help="Image, directory, video, or camera index")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", type=Path, default=Path("runs/predict"))
    parser.add_argument("--name", default="findmine")
    args = parser.parse_args()

    if not args.weights.is_file():
        parser.error(f"Weights not found: {args.weights}")
    if not 0 <= args.conf <= 1:
        parser.error("--conf must be between 0 and 1")

    YOLO(str(args.weights)).predict(
        source=args.source,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        project=str(args.project),
        name=args.name,
        save=True,
        save_txt=True,
        save_conf=True,
    )


if __name__ == "__main__":
    main()

