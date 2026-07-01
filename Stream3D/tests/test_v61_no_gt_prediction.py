from __future__ import annotations

import json
import unittest
from pathlib import Path


class V61NoGtPredictionTest(unittest.TestCase):
    def test_v61_summaries_do_not_use_gt_for_prediction(self) -> None:
        root = Path(__file__).resolve().parents[1]
        paths = [
            root / "outputs/audit/v61_phase0_failure_lock/failure_lock.json",
            root / "outputs/audit/v61_graph_v3/graph_v3_summary.json",
            root / "outputs/audit/v61_global_embedding/embedding_summary.json",
            root / "outputs/audit/v61_refinement/refinement_summary.json",
            root / "outputs/audit/v61_manifold_query/query_summary.json",
            root / "outputs/audit/v61_stress/stress_summary.json",
            root / "outputs/audit/v61_native_field/native_field_summary.json",
        ]
        missing = [str(path) for path in paths if not path.exists()]
        if missing:
            self.skipTest(f"v61 summaries missing: {missing}")
        for path in paths:
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertFalse(payload.get("uses_gt_for_prediction"), str(path))
        native = json.loads((root / "outputs/audit/v61_native_field/native_field_summary.json").read_text(encoding="utf-8"))
        self.assertFalse(native.get("uses_rgbd_pose_mesh_for_export"))


if __name__ == "__main__":
    unittest.main()
