from __future__ import annotations

import unittest

from stream4d_native.v56_quarantine_promotion_veto import build_v56_quarantine_promotion_veto


class V56QuarantinePromotionVetoTest(unittest.TestCase):
    def test_quarantine_veto_payload_has_no_gt_prediction(self) -> None:
        payload = build_v56_quarantine_promotion_veto(
            promotion_rows_path="outputs/audit/v56_promotion_repeated_masks_p4_relax_c1_m1_nocompete/promotion_rows.csv",
            ratio_thresholds=(0.0, 0.04),
        )
        summary = payload["summary"]
        self.assertFalse(summary["uses_gt_for_prediction"])
        self.assertTrue(summary["uses_gt_for_diagnostic_labels"])
        self.assertIn("pass", summary["gate"])
        self.assertGreaterEqual(summary["promoted_component_count"], 0)
        for row in payload["veto_metric_rows"]:
            self.assertFalse(row["uses_gt_for_prediction"])
            self.assertTrue(row["uses_gt_for_diagnostic_labels"])


if __name__ == "__main__":
    unittest.main()

