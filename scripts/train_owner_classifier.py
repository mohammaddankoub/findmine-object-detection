from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def macro_f1(targets: list[int], predictions: list[int], classes: int) -> float:
    scores = []
    for label in range(classes):
        tp = sum(t == label and p == label for t, p in zip(targets, predictions))
        fp = sum(t != label and p == label for t, p in zip(targets, predictions))
        fn = sum(t == label and p != label for t, p in zip(targets, predictions))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
    return float(np.mean(scores))


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
):
    model.eval()
    total_loss = 0.0
    targets: list[int] = []
    predictions: list[int] = []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            total_loss += criterion(logits, labels).item() * labels.size(0)
            targets.extend(labels.cpu().tolist())
            predictions.extend(logits.argmax(1).cpu().tolist())
    accuracy = sum(t == p for t, p in zip(targets, predictions)) / max(
        len(targets), 1
    )
    score = macro_f1(targets, predictions, len(loader.dataset.classes))
    return total_loss / max(len(targets), 1), accuracy, score, targets, predictions


def make_model(architecture: str, classes: int) -> nn.Module:
    if architecture == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, classes)
    elif architecture == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT
        )
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, classes)
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")
    return model


def set_backbone_trainable(
    model: nn.Module, architecture: str, trainable: bool
) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = trainable
    head = model.fc if architecture == "resnet18" else model.classifier
    for parameter in head.parameters():
        parameter.requires_grad = True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a FindMine owner classifier from ImageFolder datasets."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--arch", choices=("resnet18", "mobilenet_v3_small"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--freeze-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            ),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            ),
        ]
    )
    train_set = datasets.ImageFolder(args.data / "train", train_transform)
    val_set = datasets.ImageFolder(args.data / "val", eval_transform)
    test_set = datasets.ImageFolder(args.data / "test", eval_transform)
    if not (train_set.class_to_idx == val_set.class_to_idx == test_set.class_to_idx):
        raise RuntimeError("Class folders differ between train, validation, and test")

    generator = torch.Generator().manual_seed(args.seed)
    loaders = {
        "train": DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            generator=generator,
        ),
        "val": DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        ),
        "test": DataLoader(
            test_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        ),
    }

    class_counts = np.bincount(train_set.targets, minlength=len(train_set.classes))
    class_weights = len(train_set) / (
        len(train_set.classes) * np.maximum(class_counts, 1)
    )
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )
    model = make_model(args.arch, len(train_set.classes)).to(device)
    set_backbone_trainable(model, args.arch, False)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=1e-3,
        weight_decay=1e-4,
    )
    scheduler = None
    best_f1 = -1.0
    history = []
    started = time.time()

    for epoch in range(args.epochs):
        if epoch == args.freeze_epochs:
            set_backbone_trainable(model, args.arch, True)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=1e-4, weight_decay=1e-4
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(args.epochs - epoch, 1)
            )

        model.train()
        total_loss = 0.0
        correct = 0
        seen = 0
        for images, labels in loaders["train"]:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            seen += labels.size(0)
        if scheduler is not None:
            scheduler.step()

        val_loss, val_accuracy, val_f1, _, _ = evaluate(
            model, loaders["val"], criterion, device
        )
        record = {
            "epoch": epoch + 1,
            "train_loss": total_loss / seen,
            "train_accuracy": correct / seen,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "val_macro_f1": val_f1,
        }
        history.append(record)
        print(json.dumps(record))
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(
                {
                    "model": model.state_dict(),
                    "classes": train_set.classes,
                    "arch": args.arch,
                },
                args.output / "best.pt",
            )

    checkpoint = torch.load(
        args.output / "best.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model"])
    test_loss, test_accuracy, test_f1, targets, predictions = evaluate(
        model, loaders["test"], criterion, device
    )
    confusion = np.zeros((len(train_set.classes), len(train_set.classes)), dtype=int)
    for target, prediction in zip(targets, predictions):
        confusion[target, prediction] += 1
    result = {
        "architecture": args.arch,
        "classes": train_set.classes,
        "train_samples": len(train_set),
        "val_samples": len(val_set),
        "test_samples": len(test_set),
        "best_val_macro_f1": best_f1,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "test_macro_f1": test_f1,
        "confusion_matrix": confusion.tolist(),
        "elapsed_seconds": time.time() - started,
        "history": history,
    }
    (args.output / "metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "history"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
