from __future__ import annotations

import unittest

from stream4d_native.v65_final_eval import build_v65_final_decision


class V65FinalEvalTest(unittest.TestCase):
    def test_final_decision_includes_soma_no_gt_policy_gate(self) -> None:
        payload = build_v65_final_decision()
        summary = payload["summary"]
        self.assertIn("GO_SOMA_NO_GT_INFERENCE_POLICY", summary["decision_labels"])
        self.assertTrue(summary["gate"]["SOMA_no_GT_inference_policy_locked"])
        self.assertEqual(summary["key_metrics"]["soma_policy_violation_count"], 0)
        self.assertEqual(summary["key_metrics"]["soma_method_inference_gt_geometry_record_count"], 0)
        self.assertTrue(summary["key_metrics"]["soma_no_gt_inference_policy_pass"])

    def test_final_evidence_hashes_include_soma_policy_audit(self) -> None:
        payload = build_v65_final_decision()
        paths = {row["path"] for row in payload["evidence_hash_rows"]}
        self.assertIn(
            "outputs/audit/v65_soma_inference_policy_audit/soma_inference_policy_audit_summary.json",
            paths,
        )
        self.assertIn(
            "outputs/audit/v65_soma_inference_policy_audit/soma_inference_policy_violations.csv",
            paths,
        )


if __name__ == "__main__":
    unittest.main()
