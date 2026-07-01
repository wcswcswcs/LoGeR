from __future__ import annotations

import json
import tempfile
from pathlib import Path
import unittest

from stream4d_native.v64r2_dynamic_metrics import build_v64r2_dynamic_metrics


class V64R2DynamicMetricsTest(unittest.TestCase):
    def test_official_metrics_block_when_gt_level_is_too_low(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env_path = root / "dynamic_env_summary.json"
            adapter_path = root / "adapter_summary.json"
            env_path.write_text(
                json.dumps(
                    {
                        "dyn_level": 1,
                        "dyn_level_label": "DYN_LEVEL_1",
                        "can_report_official_object_tracking": False,
                        "can_report_3d_4d_trajectory_metrics": False,
                    }
                ),
                encoding="utf-8",
            )
            adapter_path.write_text(json.dumps({"gate": {"pass": False}, "blocked_reason": "missing masks"}), encoding="utf-8")
            payload = build_v64r2_dynamic_metrics(
                dynamic_env_summary_path=env_path,
                adapter_summary_path=adapter_path,
            )
            summary = payload["summary"]
            self.assertFalse(summary["gate"]["pass"])
            self.assertTrue(summary["gate"]["official_metrics_blocked_when_gt_missing"])
            for row in payload["dynamic_metric_rows"]:
                self.assertEqual(row["status"], "blocked_gt_level_or_adapter_not_available")


if __name__ == "__main__":
    unittest.main()
