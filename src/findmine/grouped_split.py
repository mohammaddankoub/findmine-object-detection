from __future__ import annotations

import random
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class Sample:
    image: Path
    label: Path


def scene_group(filename: str) -> tuple[str, str] | None:
    """Extract the capture batch and scene number from a FindMine filename."""
    match = re.search(r"(b\d+).*?photo_(\d+)_", filename, flags=re.IGNORECASE)
    if not match:
        return None
    return match.group(1).lower(), match.group(2)


def collect_samples(source: Path) -> dict[str, Sample]:
    samples: dict[str, Sample] = {}
    for split in ("train", "val", "test"):
        image_dir = source / "images" / split
        label_dir = source / "labels" / split
        if not image_dir.exists() or not label_dir.exists():
            raise FileNotFoundError(f"Missing images or labels directory for split: {split}")
        for image in image_dir.iterdir():
            if image.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            label = label_dir / f"{image.stem}.txt"
            if not label.exists():
                raise FileNotFoundError(f"Missing label for {image.name}")
            if image.name in samples:
                raise RuntimeError(f"Duplicate image name: {image.name}")
            samples[image.name] = Sample(image=image, label=label)
    return samples


def assign_groups(filenames: list[str], seed: int = 42) -> dict[str, str]:
    """Create a 70/15/15 split without separating views of the same scene."""
    grouped: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    assignment: dict[str, str] = {}

    for name in filenames:
        group = scene_group(name)
        if group is None:
            # Supplemental and targeted hard examples support training only.
            assignment[name] = "train"
            continue
        batch, scene = group
        grouped[batch][scene].append(name)

    for batch_index, batch in enumerate(sorted(grouped)):
        scenes = sorted(grouped[batch])
        random.Random(f"{seed}-{batch}").shuffle(scenes)
        if len(scenes) < 3:
            raise RuntimeError(f"{batch} has only {len(scenes)} scene groups")

        train_count = max(1, round(len(scenes) * 0.70))
        remaining = len(scenes) - train_count
        val_count = (remaining + (batch_index % 2 == 0)) // 2
        split_scenes = {
            "train": scenes[:train_count],
            "val": scenes[train_count : train_count + val_count],
            "test": scenes[train_count + val_count :],
        }
        for split, selected in split_scenes.items():
            for scene in selected:
                for name in grouped[batch][scene]:
                    assignment[name] = split
    return assignment


def copy_split(samples: dict[str, Sample], assignment: dict[str, str], destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")
    if set(samples) != set(assignment):
        raise RuntimeError("Not every image received a split")

    for split in ("train", "val", "test"):
        (destination / "images" / split).mkdir(parents=True)
        (destination / "labels" / split).mkdir(parents=True)

    for name, sample in samples.items():
        split = assignment[name]
        shutil.copy2(sample.image, destination / "images" / split / name)
        shutil.copy2(sample.label, destination / "labels" / split / sample.label.name)

    (destination / "data.yaml").write_text(
        "\n".join(
            [
                f"path: {destination.resolve().as_posix()}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "",
                "names:",
                "  0: Wallet",
                "  1: Travel_Mug",
                "",
            ]
        ),
        encoding="utf-8",
    )


def split_report(samples: dict[str, Sample], assignment: dict[str, str]) -> list[dict[str, object]]:
    rows = []
    for split in ("train", "val", "test"):
        classes: Counter[int] = Counter()
        backgrounds = 0
        selected = [name for name, value in assignment.items() if value == split]
        for name in selected:
            lines = [line for line in samples[name].label.read_text().splitlines() if line.strip()]
            if not lines:
                backgrounds += 1
            for line in lines:
                classes[int(line.split()[0])] += 1
        rows.append(
            {
                "split": split,
                "images": len(selected),
                "backgrounds": backgrounds,
                "wallet_boxes": classes[0],
                "travel_mug_boxes": classes[1],
            }
        )
    return rows

