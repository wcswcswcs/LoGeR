from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from stream4d_native.object_field import ObjectField
from tools.run_v42_native_support_bridge import _labels_from_tube_assignments, load_v42_object_fields


class V42NativeSupportBridgeTests(unittest.TestCase):
    def test_loads_v42_object_fields_from_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with (root / "object_field_rows.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "scene",
                        "variant",
                        "source",
                        "object_id",
                        "primary_field_id",
                        "semantic_masklet_ids",
                        "attached_tube_ids",
                        "confidence",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "scene": "scene0081_01",
                        "variant": "Q5",
                        "source": "dinov2_maskcut",
                        "object_id": "7",
                        "primary_field_id": "2",
                        "semantic_masklet_ids": "[1, 2]",
                        "attached_tube_ids": "[10, 11]",
                        "confidence": "0.75",
                    }
                )
            fields = load_v42_object_fields(
                root,
                scene="scene0081_01",
                variant="Q5",
                source="dinov2_maskcut",
            )
        self.assertEqual(len(fields), 1)
        self.assertEqual(fields[0].object_id, 7)
        self.assertEqual(fields[0].semantic_masklet_ids, [1, 2])
        self.assertEqual(fields[0].attached_tube_ids, [10, 11])

    def test_labels_from_tube_assignments_marks_unattached_tubes_unknown(self) -> None:
        fields = [ObjectField(object_id=3, primary_field_id=0, semantic_masklet_ids=[1], attached_tube_ids=[10])]
        pred, unknown_ratio, info = _labels_from_tube_assignments(fields, {10: 5, 11: 5, 12: 0})
        self.assertEqual(pred[10], 3)
        self.assertGreaterEqual(pred[11], 1_000_000)
        self.assertNotIn(12, pred)
        self.assertEqual(unknown_ratio, 0.5)
        self.assertEqual(info["assigned_labeled_tube_count"], 1)
        self.assertEqual(info["unknown_labeled_tube_count"], 1)


if __name__ == "__main__":
    unittest.main()
