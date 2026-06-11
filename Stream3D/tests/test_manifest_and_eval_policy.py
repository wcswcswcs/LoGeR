from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.prediction_manifest import build_prediction_manifest, write_prediction_manifest
from tools.scan_reportable_configs import scan_configs
from tools.summarize_v10_unified_eval import _method_table_allowed


class ManifestAndEvalPolicyTests(unittest.TestCase):
    def test_manifest_scanner_rejects_gt_method_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = build_prediction_manifest(
                root=root,
                output_config="bad_gt_method",
                is_method_result=True,
                is_diagnostic_only=False,
                uses_gt=True,
                gt_usage="oracle_selection",
                pre_points_policy="own_recompute_paper_style",
                support_policy="oracle",
                extra={
                    "eval_policy": "own_recompute_paper_style",
                    "uses_gt_for_prediction": True,
                    "uses_gt_for_diagnostic": True,
                },
            )
            write_prediction_manifest("bad_gt_method", manifest, root=root)

            payload = scan_configs(root=root, configs=["bad_gt_method"])

        row = payload["rows"][0]
        self.assertTrue(row["suspicious"])
        self.assertIn("uses_gt_for_prediction", row["suspicious_reasons"])
        self.assertEqual(payload["summary"]["num_uses_gt_for_prediction"], 1)

    def test_metric_table_disallows_diagnostic_only_method(self) -> None:
        row = {
            "is_method_result": True,
            "is_diagnostic_only": True,
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic": False,
        }

        self.assertFalse(_method_table_allowed(row))


if __name__ == "__main__":
    unittest.main()
