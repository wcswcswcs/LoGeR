from __future__ import annotations

import unittest

from stream4d_native.stage1_scale_aware_typed_assembly import typed_energy_selection_diagnostic_v45
from stream4d_native.v44_typed_mask_assembly import SceneArtifact


class V45MixedMaskCannotBirthTest(unittest.TestCase):
    def test_mixed_tokens_do_not_birth_objects(self) -> None:
        artifact = SceneArtifact(
            scene="s",
            root=None,  # type: ignore[arg-type]
            token_rows=[
                {"token_id": 1, "area": 10000, "boundary_contrast": 0.9, "diagnostic_gt_instance": 1, "diagnostic_gt_purity": 0.1},
                {"token_id": 2, "area": 9000, "boundary_contrast": 0.9, "diagnostic_gt_instance": 2, "diagnostic_gt_purity": 0.1},
            ],
            edge_rows=[{"token_i": 1, "token_j": 2, "same_frame_cannot_link": "True", "semantic_affinity": 0.99, "object_affinity": 0.99, "diagnostic_same_gt": "False"}],
            source_rows=[],
            alignment_rows=[],
        )
        payload = typed_energy_selection_diagnostic_v45([artifact])
        self.assertTrue(all(row["mixed_birth_count"] == 0 for row in payload["variant_rows"]))


if __name__ == "__main__":
    unittest.main()

