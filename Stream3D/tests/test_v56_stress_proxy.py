from __future__ import annotations

import unittest

from stream4d_native.v56_stress_proxy import build_v56_stress_proxy


class V56StressProxyTest(unittest.TestCase):
    def test_stress_proxy_payload_has_no_gt_prediction(self) -> None:
        payload = build_v56_stress_proxy()
        summary = payload["summary"]
        self.assertFalse(summary["uses_gt_for_prediction"])
        self.assertTrue(summary["uses_gt_for_diagnostic_labels"])
        self.assertIn("pass", summary["gate"])
        self.assertGreater(summary["stress_setting_count"], 0)
        for row in payload["stress_metric_rows"]:
            self.assertFalse(row["uses_gt_for_prediction"])
            self.assertTrue(row["uses_gt_for_diagnostic_labels"])


if __name__ == "__main__":
    unittest.main()

