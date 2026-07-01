from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path

from tools.prediction_manifest import build_prediction_manifest
from tools.scan_reportable_configs import scan_configs
from tools.summarize_v10_unified_eval import _method_table_allowed


class ManifestAndEvalPolicyTests(unittest.TestCase):
    def test_manifest_builder_rejects_gt_method_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "uses GT geometry"):
                build_prediction_manifest(
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

    def test_manifest_scanner_marks_legacy_gt_artifact_suspicious(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = {
                "schema_version": "stream4d_prediction_manifest_v1",
                "output_config": "bad_gt_method",
                "is_method_result": True,
                "is_diagnostic_only": False,
                "uses_gt": True,
                "uses_gt_for_prediction": True,
                "uses_gt_for_diagnostic": True,
                "pre_points_policy": "own_recompute_paper_style",
                "support_policy": "oracle",
            }
            pred_dir = root / "data" / "prediction" / "bad_gt_method_class_agnostic"
            tmp_dir = root / "data" / "TMP" / "bad_gt_method"
            pred_dir.mkdir(parents=True)
            tmp_dir.mkdir(parents=True)
            (pred_dir / "config_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (tmp_dir / "config_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            payload = scan_configs(root=root, configs=["bad_gt_method"])

        row = payload["rows"][0]
        self.assertTrue(row["suspicious"])
        self.assertIn("uses_gt_for_prediction", row["suspicious_reasons"])
        self.assertEqual(payload["summary"]["num_uses_gt_for_prediction"], 1)

    def test_diagnostic_gt_geometry_manifest_is_forbidden_not_method(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = build_prediction_manifest(
                root=root,
                output_config="diag_gt_geometry",
                is_method_result=False,
                is_diagnostic_only=False,
                uses_gt=False,
                gt_usage="none",
                pre_points_policy="evaluation_adapter",
                support_policy="used_frame_depth_pose_visible_mask_support",
                extra={
                    "uses_rgbd_for_evaluation_support": True,
                    "uses_rgbd_pose_mesh_for_export": True,
                },
            )

        self.assertFalse(manifest["is_method_result"])
        self.assertTrue(manifest["is_diagnostic_only"])
        self.assertTrue(manifest["forbidden_for_method_table"])

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
