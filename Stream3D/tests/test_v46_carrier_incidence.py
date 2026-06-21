from __future__ import annotations

import unittest
from pathlib import Path

from stream4d_native.v44_typed_mask_assembly import SceneArtifact
from stream4d_native.v46_signed_mask_graph import incidence_audit


def _artifact() -> SceneArtifact:
    return SceneArtifact(
        scene="scene_test",
        root=Path("."),
        token_rows=[
            {"token_id": "1", "frame_id": "0", "mask_id": "1", "area": "100", "boundary_contrast": "0.1", "diagnostic_gt_instance": "1", "diagnostic_gt_purity": "0.9"},
            {"token_id": "2", "frame_id": "1", "mask_id": "2", "area": "100", "boundary_contrast": "0.1", "diagnostic_gt_instance": "1", "diagnostic_gt_purity": "0.9"},
        ],
        edge_rows=[],
        source_rows=[],
        alignment_rows=[
            {
                "token_i": "1",
                "token_j": "2",
                "material_union_count": "16",
                "shared_tube_count": "4",
                "shared_tube_ids": "[1, 2, 3, 4]",
                "scene_tube_ids": "[1, 2, 3, 4]",
                "object_tube_ids": "[]",
                "part_tube_ids": "[]",
                "unknown_tube_ids": "[]",
            }
        ],
    )


class V46CarrierIncidenceTest(unittest.TestCase):
    def test_incidence_records_proxy_not_raw_uv(self) -> None:
        payload = incidence_audit([_artifact()])
        self.assertFalse(payload["gate"]["uses_raw_uv_containment"])
        self.assertEqual(payload["scene_rows"][0]["mask_with_ge16_carrier_ratio"], 1.0)


if __name__ == "__main__":
    unittest.main()
