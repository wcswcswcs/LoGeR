from __future__ import annotations

import unittest

from stream4d_native.v49_mosaic_stage1 import build_semantic_set_compatibility


class TestV49SemanticSetCompatibility(unittest.TestCase):
    def test_dense_backends_not_claimed_without_component_artifacts(self) -> None:
        payload = build_semantic_set_compatibility()
        dense_rows = [row for row in payload["backend_rows"] if row["backend_id"] in {"DINOv2", "RADIO/RADSeg"}]
        self.assertTrue(dense_rows)
        self.assertTrue(all(not row["component_level_available"] for row in dense_rows))
        self.assertFalse(payload["uses_gt_for_prediction"])


if __name__ == "__main__":
    unittest.main()
