import csv
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_experiments import best_row


class ExperimentSummaryTests(unittest.TestCase):
    def test_best_row_uses_map50(self):
        fields = [
            "epoch",
            "metrics/precision(B)",
            "metrics/recall(B)",
            "metrics/mAP50(B)",
            "metrics/mAP50-95(B)",
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerow(dict(zip(fields, (1, 0.7, 0.6, 0.65, 0.4))))
                writer.writerow(dict(zip(fields, (2, 0.8, 0.7, 0.75, 0.5))))
            self.assertEqual(best_row(path)["epoch"], "2")


if __name__ == "__main__":
    unittest.main()

