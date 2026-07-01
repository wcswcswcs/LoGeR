from __future__ import annotations

import unittest

from stream4d_native.v65_geometry_contract import build_v65_geometry_contract


class V65GeometryContractTest(unittest.TestCase):
    def test_geometry_contract_has_g0_to_g5(self) -> None:
        payload = build_v65_geometry_contract()
        summary = payload["summary"]
        self.assertEqual(summary["missing_levels"], [])
        self.assertEqual(set(summary["levels_present"]), {"G0", "G1", "G2", "G3", "G4", "G5"})
        self.assertTrue(summary["gate"]["chunk_level_not_promoted_to_scene_claim"])

    def test_scene_metrics_are_diagnostic_when_using_scannet_pose_mesh(self) -> None:
        payload = build_v65_geometry_contract()
        g3_rows = [row for row in payload["geometry_metric_rows"] if row["metric_level"] == "G3"]
        self.assertTrue(g3_rows)
        self.assertTrue(all(row["is_diagnostic_metric"] for row in g3_rows))
        self.assertTrue(all(not row["is_method_safe_metric"] for row in g3_rows))


if __name__ == "__main__":
    unittest.main()
