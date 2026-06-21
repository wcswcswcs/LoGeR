from __future__ import annotations

import unittest

from stream4d_native.stage1_scale_aware_typed_assembly import descriptor_audit_v45
from stream4d_native.v44_typed_mask_assembly import SceneArtifact


class V45MaskDescriptorTest(unittest.TestCase):
    def test_descriptor_without_signal_fails_gate(self) -> None:
        artifact = SceneArtifact(
            scene="s",
            root=None,  # type: ignore[arg-type]
            token_rows=[
                {"token_id": 1, "area": 100, "boundary_contrast": 0.1, "diagnostic_gt_instance": 1, "diagnostic_gt_purity": 0.8},
                {"token_id": 2, "area": 100, "boundary_contrast": 0.1, "diagnostic_gt_instance": 2, "diagnostic_gt_purity": 0.8},
            ],
            edge_rows=[{"token_i": 1, "token_j": 2, "semantic_affinity": 0.5, "object_affinity": 0.5, "diagnostic_same_gt": "False"}],
            source_rows=[{"feature_backend": "dinov2_timm"}],
            alignment_rows=[],
        )
        payload = descriptor_audit_v45([artifact], feature_smokes=[])
        self.assertFalse(payload["gate"]["pass"])


if __name__ == "__main__":
    unittest.main()

