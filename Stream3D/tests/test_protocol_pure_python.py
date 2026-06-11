from __future__ import annotations

import unittest

import numpy as np

from stream4d.export_scannet import score_export_record
from stream4d.reliable_densifier import apply_wta_to_records


class ProtocolPurePythonTests(unittest.TestCase):
    def test_score_export_record_uses_requested_field(self) -> None:
        record = {
            "point_ids": {1, 2, 3},
            "area_score": 3.0,
            "observations": 2.0,
            "reliability": 5.0,
        }

        self.assertEqual(score_export_record(record, "area"), 3.0)
        self.assertEqual(score_export_record(record, "observations"), 2.0)
        self.assertEqual(score_export_record(record, "reliability"), 5.0)

    def test_wta_recomputes_area_after_conflict_resolution(self) -> None:
        records = [
            {"object_id": 0, "point_ids": {1, 2}, "area_score": 2.0, "score": 2.0, "reliability": 1.0},
            {"object_id": 1, "point_ids": {2, 3}, "area_score": 2.0, "score": 2.0, "reliability": 3.0},
        ]

        reassigned, diag = apply_wta_to_records(records)

        self.assertEqual(reassigned[0]["point_ids"], {1})
        self.assertEqual(reassigned[0]["area_score"], 1.0)
        self.assertEqual(reassigned[1]["point_ids"], {2, 3})
        self.assertEqual(diag["densify_wta_conflict_points"], 1.0)

    def test_numpy_sanity_fixture_is_deterministic(self) -> None:
        rng = np.random.default_rng(13)
        self.assertEqual(rng.permutation(4).tolist(), [3, 2, 1, 0])


if __name__ == "__main__":
    unittest.main()
