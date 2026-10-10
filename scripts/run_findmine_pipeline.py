from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms
from ultralytics import YOLO


EVAL_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def load_classifier(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    classes = checkpoint["classes"]
    architecture = checkpoint["arch"]
    if architecture == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(classes))
    elif architecture == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(classes))
    else:
        raise ValueError(f"Unsupported classifier: {architecture}")
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    return model, classes, architecture


def classify(crop_bgr, model, classes, device, minimum_confidence: float):
    image = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
    tensor = EVAL_TRANSFORM(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probability = torch.softmax(model(tensor), dim=1)[0]
    confidence, index = probability.max(dim=0)
    owner = classes[int(index)]
    confidence_value = float(confidence)
    if confidence_value < minimum_confidence:
        owner = "Unknown"
    return owner, confidence_value


def image_paths(source: Path):
    if source.is_file():
        return [source]
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(path for path in source.iterdir() if path.suffix.lower() in extensions)


def padded_crop(image, xyxy, padding_fraction: float = 0.08):
    x1, y1, x2, y2 = map(float, xyxy)
    padding = padding_fraction * max(x2 - x1, y2 - y1)
    height, width = image.shape[:2]
    left = max(0, int(x1 - padding))
    top = max(0, int(y1 - padding))
    right = min(width, int(x2 + padding + 0.999))
    bottom = min(height, int(y2 + padding + 0.999))
    return image[top:bottom, left:right], (left, top, right, bottom)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run FindMine object detection and owner recognition."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--wallet-model", type=Path, required=True)
    parser.add_argument("--mug-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--det-conf", type=float, default=0.88)
    parser.add_argument("--owner-conf", type=float, default=0.50)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector = YOLO(args.detector)
    wallet_model, wallet_classes, wallet_arch = load_classifier(
        args.wallet_model, device
    )
    mug_model, mug_classes, mug_arch = load_classifier(args.mug_model, device)
    classifiers = {
        "Wallet": (wallet_model, wallet_classes, wallet_arch),
        "Travel_Mug": (mug_model, mug_classes, mug_arch),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    records = []

    for path in image_paths(args.source):
        image = cv2.imread(str(path))
        if image is None:
            print(f"Skipping unreadable image: {path}")
            continue
        result = detector.predict(
            image,
            imgsz=args.imgsz,
            conf=args.det_conf,
            device=0 if device.type == "cuda" else "cpu",
            verbose=False,
        )[0]
        for box in result.boxes:
            class_id = int(box.cls.item())
            object_class = str(result.names[class_id])
            if object_class not in classifiers:
                continue
            detection_confidence = float(box.conf.item())
            crop, (x1, y1, x2, y2) = padded_crop(
                image, box.xyxy[0].cpu().tolist()
            )
            if crop.size == 0:
                continue
            classifier, classes, architecture = classifiers[object_class]
            owner, owner_confidence = classify(
                crop, classifier, classes, device, args.owner_conf
            )
            label = (
                f"{object_class} | {owner} | det {detection_confidence:.2f} | "
                f"owner {owner_confidence:.2f}"
            )
            colour = (40, 190, 40) if owner != "Unknown" else (0, 165, 255)
            cv2.rectangle(image, (x1, y1), (x2, y2), colour, 3)
            text_y = max(24, y1 - 8)
            cv2.putText(
                image,
                label,
                (x1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.56,
                (0, 0, 0),
                4,
                cv2.LINE_AA,
            )
            cv2.putText(
                image,
                label,
                (x1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.56,
                colour,
                2,
                cv2.LINE_AA,
            )
            records.append(
                {
                    "image": path.name,
                    "object_class": object_class,
                    "owner": owner,
                    "detection_confidence": f"{detection_confidence:.6f}",
                    "owner_confidence": f"{owner_confidence:.6f}",
                    "classifier": architecture,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                }
            )
        cv2.imwrite(str(args.output / path.name), image)

    fields = [
        "image",
        "object_class",
        "owner",
        "detection_confidence",
        "owner_confidence",
        "classifier",
        "x1",
        "y1",
        "x2",
        "y2",
    ]
    with (args.output / "predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    print(f"Saved {len(records)} predictions to {args.output}")


if __name__ == "__main__":
    main()
