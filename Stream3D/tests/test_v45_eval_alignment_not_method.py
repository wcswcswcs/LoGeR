from __future__ import annotations

import unittest

from evaluation.eval_aligned_ap_bridge import EvalAlignmentManifest, validate_eval_alignment_manifest


class V45EvalAlignmentPolicyTest(unittest.TestCase):
    def test_eval_alignment_must_not_be_prediction_input(self) -> None:
        manifest = EvalAlignmentManifest(
            alignment_protocol="scene_level_eval_sim3",
            uses_gt_for_prediction=False,
            uses_gt_for_evaluation_alignment=True,
            scale_aligned_eval_protocol=True,
            is_method_result=False,
        )
        self.assertTrue(validate_eval_alignment_manifest(manifest)["pass"])

    def test_gt_prediction_leak_fails(self) -> None:
        manifest = EvalAlignmentManifest(
            alignment_protocol="bad",
            uses_gt_for_prediction=True,
            uses_gt_for_evaluation_alignment=True,
            scale_aligned_eval_protocol=True,
            is_method_result=True,
        )
        self.assertFalse(validate_eval_alignment_manifest(manifest)["pass"])


if __name__ == "__main__":
    unittest.main()

