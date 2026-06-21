from __future__ import annotations

import unittest
from pathlib import Path

from stream4d_native.v44_typed_mask_assembly import SceneArtifact
from stream4d_native.v46_signed_mask_graph import object_field_export


class V46NoD4RTTubeBirthTest(unittest.TestCase):
    def test_object_export_requires_mask_clusters(self) -> None:
        artifact = SceneArtifact(
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
                    "semantic_affinity": "0.9",
                    "object_affinity": "0.9",
                    "diagnostic_same_gt": "True",
                    "same_frame_cannot_link": "False",
                    "shared_tube_count": "2",
                    "trusted_material_tube_count": "4",
                    "object_part_tube_count": "4",
                    "material_union_count": "10",
                    "visible_outside_conflict_ratio": "0.0",
                }
            ],
        )
        payload = object_field_export([artifact])
        self.assertEqual(payload["scene_rows"][0]["birth_from_d4rt_tube_count"], 0)
        self.assertEqual(payload["scene_rows"][0]["maskless_object_count"], 0)
        self.assertTrue(all(not row["birth_from_d4rt_tube"] for row in payload["object_rows"]))


if __name__ == "__main__":
    unittest.main()
