from __future__ import annotations

import unittest
from pathlib import Path

from stream4d_native.v44_typed_mask_assembly import SceneArtifact
from stream4d_native.v46_signed_mask_graph import edge_scores_for_artifact


class V46SupporterReliabilityTest(unittest.TestCase):
    def test_low_q_supporter_does_not_create_strong_positive_alone(self) -> None:
        artifact = SceneArtifact(
            scene="scene_test",
            root=Path("."),
            token_rows=[
                {"token_id": "1", "frame_id": "0", "mask_id": "1", "area": "10000", "boundary_contrast": "1.0", "diagnostic_gt_instance": "1", "diagnostic_gt_purity": "0.1"},
                {"token_id": "2", "frame_id": "1", "mask_id": "2", "area": "10000", "boundary_contrast": "1.0", "diagnostic_gt_instance": "2", "diagnostic_gt_purity": "0.1"},
            ],
            edge_rows=[],
            source_rows=[],
            alignment_rows=[
                {
                    "token_i": "1",
                    "token_j": "2",
                    "semantic_affinity": "0.99",
                    "object_affinity": "0.99",
                    "diagnostic_same_gt": "False",
                    "same_frame_cannot_link": "True",
                    "shared_tube_count": "0",
                    "trusted_material_tube_count": "0",
                    "object_part_tube_count": "0",
                    "material_union_count": "20",
                    "visible_outside_conflict_ratio": "1.0",
                }
            ],
        )
        edge = edge_scores_for_artifact(artifact, positive_variant="P5_full", negative_variant="N7_full")[0]
        self.assertLess(edge["positive_weight"], 0.25)
        self.assertGreaterEqual(edge["negative_weight"], 0.7)


if __name__ == "__main__":
    unittest.main()
