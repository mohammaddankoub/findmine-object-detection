import tempfile
import unittest
from pathlib import Path

from src.findmine.grouped_split import Sample, assign_groups, scene_group, split_report


class SceneGroupingTests(unittest.TestCase):
    def test_scene_group_parses_batch_and_photo_number(self):
        self.assertEqual(scene_group("b02_photo_5_2026-10-04_10-22-38.jpg"), ("b02", "5"))

    def test_supplemental_filename_has_no_scene_group(self):
        self.assertIsNone(scene_group("public_wallet_014.jpg"))

    def test_near_duplicate_views_stay_in_one_split(self):
        names = [
            f"b01_photo_{scene}_2026-10-02_15-{minute:02d}-00.jpg"
            for scene in range(1, 11)
            for minute in (20, 21, 22)
        ]
        assignment = assign_groups(names)
        for scene in range(1, 11):
            splits = {assignment[name] for name in names if f"photo_{scene}_" in name}
            self.assertEqual(len(splits), 1)

    def test_support_images_are_training_only(self):
        names = [
            "public_wallet_01.jpg",
            *[f"b01_photo_{scene}_capture.jpg" for scene in range(1, 8)],
        ]
        self.assertEqual(assign_groups(names)["public_wallet_01.jpg"], "train")


class ReportTests(unittest.TestCase):
    def test_report_counts_classes_and_backgrounds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            samples = {}
            for index, content in enumerate(("0 0.5 0.5 0.2 0.2\n", "1 0.5 0.5 0.2 0.2\n", "")):
                label = root / f"sample_{index}.txt"
                label.write_text(content, encoding="utf-8")
                samples[f"sample_{index}.jpg"] = Sample(root / f"sample_{index}.jpg", label)
            assignment = {name: "train" for name in samples}
            report = split_report(samples, assignment)[0]
            self.assertEqual(report["images"], 3)
            self.assertEqual(report["backgrounds"], 1)
            self.assertEqual(report["wallet_boxes"], 1)
            self.assertEqual(report["travel_mug_boxes"], 1)


if __name__ == "__main__":
    unittest.main()

