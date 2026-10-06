from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a FindMine YOLO detector.")
    parser.add_argument("--data", type=Path, required=True, help="Path to data.yaml")
    parser.add_argument("--model", default="yolo11s.pt", help="Base model or checkpoint")
    parser.add_argument("--name", required=True, help="Run name")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0", help="CUDA device, 'cpu', or 'mps'")
    parser.add_argument("--project", type=Path, default=Path("runs"))
    parser.add_argument("--patience", type=int, default=15)
    args = parser.parse_args()

    if not args.data.is_file():
        parser.error(f"Dataset configuration not found: {args.data}")

    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(args.project),
        name=args.name,
        patience=args.patience,
        seed=42,
        deterministic=True,
        cache=True,
        close_mosaic=10,
    )


if __name__ == "__main__":
    main()

