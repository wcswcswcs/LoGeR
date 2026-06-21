from __future__ import annotations

import unittest
from pathlib import Path

from stream4d_native.v44_typed_mask_assembly import SceneArtifact
from stream4d_native.v46_signed_mask_graph import edge_scores_for_artifact


class V46SignedEdgesTest(unittest.TestCase):
    def test_visible_outside_same_frame_becomes_hard_negative(self) -> None:
        artifact = SceneArtifact(
            scene="scene_test",
            root=Path("."),
            token_rows=[
                {"token_id": "1", "frame_id": "0", "mask_id": "1", "area": "100", "boundary_contrast": "0.1", "diagnostic_gt_instance": "1", "diagnostic_gt_purity": "0.9"},
                {"token_id": "2", "frame_id": "0", "mask_id": "2", "area": "100", "boundary_contrast": "0.1", "diagnostic_gt_instance": "2", "diagnostic_gt_purity": "0.9"},
            ],
            edge_rows=[],
            source_rows=[],
            alignment_rows=[
                {
                    "token_i": "1",
                    "token_j": "2",
                    "semantic_affinity": "0.2",
                    "object_affinity": "0.2",
                    "diagnostic_same_gt": "False",
                    "same_frame_cannot_link": "True",
                    "shared_tube_count": "0",
                    "trusted_material_tube_count": "0",
                    "object_part_tube_count": "0",
                    "material_union_count": "8",
                    "visible_outside_conflict_ratio": "0.9",
                }
            ],
        )
        edge = edge_scores_for_artifact(artifact, negative_variant="N7_full")[0]
        self.assertEqual(edge["negative_weight"], 1.0)
        self.assertIn("same_frame", edge["edge_type"])


if __name__ == "__main__":
    unittest.main()
