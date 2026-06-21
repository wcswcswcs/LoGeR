from __future__ import annotations

import unittest

from stream4d_native.v46_signed_mask_graph import eval_aligned_ap_policy


class V46EvalAlignmentPolicyTest(unittest.TestCase):
    def test_ap_blocked_when_stage1_fails(self) -> None:
        payload = eval_aligned_ap_policy({"gate": {"pass": False}})
        self.assertEqual(payload["status"], "blocked_stage1_not_method")
        self.assertFalse(payload["uses_gt_for_prediction"])
        self.assertFalse(payload["gate"]["pass"])

    def test_eval_alignment_only_after_stage1_pass(self) -> None:
        payload = eval_aligned_ap_policy({"gate": {"pass": True}})
        self.assertFalse(payload["uses_gt_for_prediction"])
        self.assertTrue(payload["uses_gt_for_evaluation_alignment"])


if __name__ == "__main__":
    unittest.main()
