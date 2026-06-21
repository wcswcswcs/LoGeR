from __future__ import annotations

import unittest

from stream4d_native.stage1_scale_aware_typed_assembly import typed_operation_audit_v45
from stream4d_native.v44_typed_mask_assembly import SceneArtifact


class V45AbsorbUsesD4RTTest(unittest.TestCase):
    def test_visible_outside_blocks_absorb(self) -> None:
        artifact = SceneArtifact(
            scene="s",
            root=None,  # type: ignore[arg-type]
            token_rows=[
                {"token_id": 1, "area": 1000, "boundary_contrast": 0.1, "diagnostic_gt_instance": 1, "diagnostic_gt_purity": 0.8},
                {"token_id": 2, "area": 10, "boundary_contrast": 0.1, "diagnostic_gt_instance": 1, "diagnostic_gt_purity": 0.5},
            ],
            edge_rows=[{"token_i": 1, "token_j": 2, "same_frame_cannot_link": "False", "semantic_affinity": 0.9, "object_affinity": 0.9, "diagnostic_same_gt": "True"}],
            source_rows=[],
            alignment_rows=[{"token_i": 1, "token_j": 2, "diagnostic_same_gt": "True", "shared_tube_count": 1, "trusted_material_tube_count": 1, "material_union_count": 2, "visible_outside_conflict_ratio": 1.0}],
        )
        payload = typed_operation_audit_v45([artifact])
        strict = [row for row in payload["rows"] if row["operation_profile"] == "O7_full_typed_operations"][0]
        self.assertEqual(strict["absorb_precision"], 0.0)


if __name__ == "__main__":
    unittest.main()

