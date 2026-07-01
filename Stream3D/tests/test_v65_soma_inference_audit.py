from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.prediction_manifest import build_prediction_manifest
from stream4d_native.soma_inference_policy import policy_violation_reasons
from stream4d_native.v47_common import write_csv, write_json
from stream4d_native.v65_soma_inference_audit import (
    V65SOMAInferenceArtifact,
    build_v65_soma_inference_audit,
)


class V65SOMAInferenceAuditTest(unittest.TestCase):
    def test_default_manifest_does_not_claim_rgbd_bridge(self) -> None:
        manifest = build_prediction_manifest(output_config="unit_default_manifest")
        self.assertTrue(manifest["is_method_result"])
        self.assertFalse(manifest["forbidden_for_method_table"])
        self.assertEqual(manifest["geometry_source"], "unknown")

    def test_diagnostic_eval_bridge_manifest_is_forbidden_not_method(self) -> None:
        manifest = build_prediction_manifest(
            output_config="unit_eval_bridge_manifest",
            is_method_result=False,
            is_diagnostic_only=True,
            uses_gt=False,
            gt_usage="scannet_rgbd_pose_mesh_mask_backproject",
            extra={
                "geometry_source": "scannet_rgbd_pose_mesh_mask_backproject_eval_adapter",
                "uses_rgbd_for_evaluation_support": True,
                "uses_rgbd_pose_mesh_for_export": True,
                "uses_gt_for_diagnostic": True,
                "forbidden_for_method_table": True,
            },
        )
        self.assertFalse(manifest["is_method_result"])
        self.assertTrue(manifest["is_diagnostic_only"])
        self.assertTrue(manifest["forbidden_for_method_table"])

    def test_policy_catches_explicit_gt_geometry_inference_flag(self) -> None:
        violations = policy_violation_reasons(
            {
                "is_method_result": True,
                "uses_gt_geometry_for_inference": True,
                "forbidden_for_method_table": False,
                "is_diagnostic_only": False,
            }
        )
        self.assertIn("method_result_uses_gt_geometry", violations)
        self.assertIn("gt_geometry_not_forbidden_for_method_table", violations)
        self.assertIn("gt_geometry_inference_not_marked_diagnostic", violations)

    def test_audit_allows_diagnostic_eval_bridge_but_not_method_gt_inference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            method_path = root / "method.csv"
            eval_path = root / "eval.json"
            write_csv(
                method_path,
                [
                    {
                        "row_id": "m0",
                        "is_method_result": True,
                        "method_safe_inference_artifact": True,
                        "uses_gt_for_prediction": False,
                        "uses_gt_geometry_for_inference": False,
                        "uses_rgbd_pose_mesh_for_export": False,
                        "forbidden_for_method_table": False,
                        "is_diagnostic_only": False,
                    }
                ],
            )
            write_json(
                eval_path,
                {
                    "phase": "eval_bridge",
                    "is_method_result": False,
                    "uses_gt_for_prediction": False,
                    "uses_gt_geometry_for_inference": False,
                    "uses_rgbd_pose_mesh_for_export": True,
                    "forbidden_for_method_table": True,
                    "is_diagnostic_only": True,
                },
            )
            payload = build_v65_soma_inference_audit(
                (
                    V65SOMAInferenceArtifact("method", method_path, "csv", required=True),
                    V65SOMAInferenceArtifact("eval", eval_path, "json", required=True),
                )
            )
            self.assertTrue(payload["summary"]["gate"]["pass"])
            self.assertEqual(payload["summary"]["policy_violation_count"], 0)
            self.assertEqual(payload["summary"]["gt_eval_or_export_record_count"], 1)

    def test_audit_rejects_method_row_with_gt_geometry_inference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            method_path = root / "bad_method.csv"
            write_csv(
                method_path,
                [
                    {
                        "row_id": "bad",
                        "is_method_result": True,
                        "uses_gt_geometry_for_inference": True,
                        "forbidden_for_method_table": False,
                        "is_diagnostic_only": False,
                    }
                ],
            )
            payload = build_v65_soma_inference_audit(
                (V65SOMAInferenceArtifact("bad_method", method_path, "csv", required=True),)
            )
            self.assertFalse(payload["summary"]["gate"]["pass"])
            self.assertEqual(payload["summary"]["method_inference_gt_geometry_record_count"], 1)
            self.assertEqual(payload["summary"]["policy_violation_count"], 1)


if __name__ == "__main__":
    unittest.main()
