from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

from run_findmine_pipeline import classify, load_classifier, padded_crop


CLASS_NAMES = {0: "Wallet", 1: "Travel_Mug"}


def yolo_xyxy(line: str, width: int, height: int):
    class_id, xc, yc, box_width, box_height = map(float, line.split())
    x1 = (xc - box_width / 2) * width
    y1 = (yc - box_height / 2) * height
    x2 = (xc + box_width / 2) * width
    y2 = (yc + box_height / 2) * height
    return int(class_id), [x1, y1, x2, y2]


def iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return intersection / max(area_a + area_b - intersection, 1e-9)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate FindMine detection and owner recognition end to end."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--wallet-model", type=Path, required=True)
    parser.add_argument("--mug-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--det-conf", type=float, default=0.88)
    parser.add_argument("--owner-conf", type=float, default=0.50)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector = YOLO(args.detector)
    wallet_model, wallet_classes, _ = load_classifier(args.wallet_model, device)
    mug_model, mug_classes, _ = load_classifier(args.mug_model, device)
    classifiers = {
        "Wallet": (wallet_model, wallet_classes),
        "Travel_Mug": (mug_model, mug_classes),
    }

    owner_lookup = {}
    with args.manifest.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row["split"] == args.split:
                key = (row["source_image"], row["object_class"], int(row["box_index"]))
                owner_lookup[key] = row["owner"]

    totals = Counter()
    per_class = defaultdict(Counter)
    rows = []
    image_dir = args.dataset / "images" / args.split
    label_dir = args.dataset / "labels" / args.split
    image_count = 0

    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        image_count += 1
        height, width = image.shape[:2]
        label_path = label_dir / f"{image_path.stem}.txt"
        ground_truth = []
        for box_index, line in enumerate(label_path.read_text().splitlines()):
            if not line.strip():
                continue
            class_id, box = yolo_xyxy(line, width, height)
            object_class = CLASS_NAMES[class_id]
            owner = owner_lookup.get((image_path.name, object_class, box_index))
            if owner is None:
                raise KeyError(
                    f"Missing owner label: {image_path.name}, {object_class}, "
                    f"box {box_index}"
                )
            ground_truth.append(
                {
                    "class_id": class_id,
                    "object_class": object_class,
                    "box": box,
                    "owner": owner,
                    "matched": False,
                }
            )
            totals["gt"] += 1
            per_class[object_class]["gt"] += 1

        result = detector.predict(
            image,
            imgsz=args.imgsz,
            conf=args.det_conf,
            device=0 if device.type == "cuda" else "cpu",
            verbose=False,
        )[0]
        detections = sorted(
            result.boxes, key=lambda box: float(box.conf.item()), reverse=True
        )
        for detection in detections:
            class_id = int(detection.cls.item())
            object_class = str(result.names[class_id])
            if object_class not in classifiers:
                continue
            predicted_box = detection.xyxy[0].cpu().tolist()
            candidates = [
                (iou(predicted_box, gt["box"]), index)
                for index, gt in enumerate(ground_truth)
                if not gt["matched"] and gt["class_id"] == class_id
            ]
            best_iou, best_index = max(candidates, default=(0.0, -1))
            if best_iou < args.iou:
                totals["false_positive"] += 1
                per_class[object_class]["false_positive"] += 1
                rows.append(
                    {
                        "image": image_path.name,
                        "object_class": object_class,
                        "gt_owner": "",
                        "predicted_owner": "",
                        "detection_confidence": float(detection.conf.item()),
                        "owner_confidence": "",
                        "iou": best_iou,
                        "status": "false_positive",
                    }
                )
                continue

            ground_truth[best_index]["matched"] = True
            gt = ground_truth[best_index]
            crop, _ = padded_crop(image, predicted_box)
            classifier, classes = classifiers[object_class]
            predicted_owner, owner_confidence = classify(
                crop, classifier, classes, device, args.owner_conf
            )
            correct = predicted_owner == gt["owner"]
            totals["matched"] += 1
            totals["owner_correct"] += int(correct)
            per_class[object_class]["matched"] += 1
            per_class[object_class]["owner_correct"] += int(correct)
            rows.append(
                {
                    "image": image_path.name,
                    "object_class": object_class,
                    "gt_owner": gt["owner"],
                    "predicted_owner": predicted_owner,
                    "detection_confidence": float(detection.conf.item()),
                    "owner_confidence": owner_confidence,
                    "iou": best_iou,
                    "status": "correct" if correct else "wrong_owner",
                }
            )

        for gt in ground_truth:
            if not gt["matched"]:
                totals["missed"] += 1
                per_class[gt["object_class"]]["missed"] += 1
                rows.append(
                    {
                        "image": image_path.name,
                        "object_class": gt["object_class"],
                        "gt_owner": gt["owner"],
                        "predicted_owner": "",
                        "detection_confidence": "",
                        "owner_confidence": "",
                        "iou": "",
                        "status": "missed",
                    }
                )

    detection_precision = totals["matched"] / max(
        totals["matched"] + totals["false_positive"], 1
    )
    detection_recall = totals["matched"] / max(totals["gt"], 1)
    detection_f1 = 2 * detection_precision * detection_recall / max(
        detection_precision + detection_recall, 1e-9
    )
    exact_precision = totals["owner_correct"] / max(
        totals["matched"] + totals["false_positive"], 1
    )
    exact_recall = totals["owner_correct"] / max(totals["gt"], 1)
    exact_f1 = 2 * exact_precision * exact_recall / max(
        exact_precision + exact_recall, 1e-9
    )
    summary = {
        "split": args.split,
        "images": image_count,
        "gt_objects": totals["gt"],
        "matched_detections": totals["matched"],
        "missed_objects": totals["missed"],
        "false_positives": totals["false_positive"],
        "detection_precision_at_iou": detection_precision,
        "detection_recall_at_iou": detection_recall,
        "detection_f1_at_iou": detection_f1,
        "conditional_owner_accuracy": totals["owner_correct"]
        / max(totals["matched"], 1),
        "end_to_end_precision": exact_precision,
        "end_to_end_recall": exact_recall,
        "end_to_end_f1": exact_f1,
        "per_class": {},
        "settings": {
            "det_conf": args.det_conf,
            "owner_conf": args.owner_conf,
            "iou": args.iou,
            "imgsz": args.imgsz,
        },
    }
    for object_class, values in sorted(per_class.items()):
        summary["per_class"][object_class] = {
            **values,
            "detection_recall_at_iou": values["matched"] / max(values["gt"], 1),
            "conditional_owner_accuracy": values["owner_correct"]
            / max(values["matched"], 1),
            "end_to_end_recall": values["owner_correct"] / max(values["gt"], 1),
        }

    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    fields = [
        "image",
        "object_class",
        "gt_owner",
        "predicted_owner",
        "detection_confidence",
        "owner_confidence",
        "iou",
        "status",
    ]
    with (args.output / "details.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
