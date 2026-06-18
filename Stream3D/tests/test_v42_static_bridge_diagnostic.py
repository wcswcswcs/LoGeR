from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from tools.run_v42_static_bridge_diagnostic import build_static_bridge


class V42StaticBridgeDiagnosticTests(unittest.TestCase):
    def test_build_static_bridge_marks_native_support_not_ap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            memory_summary = root / "memory_summary.json"
            memory_scene_rows = root / "memory_scene_rows.csv"
            reference_table = root / "reference.csv"
            memory_summary.write_text(
                json.dumps(
                    {
                        "scene_count": 2,
                        "expanded_object_field_count": 34,
                        "aggregate_tube_4D_ARI": 0.54,
                        "aggregate_tube_purity": 0.90,
                        "aggregate_tube_completeness": 0.62,
                        "mean_unknown_labeled_tube_ratio": 0.32,
                        "native_support_metric_proxy_pass": True,
                        "phase8_gate_pass": False,
                        "AP_bridge_status": "not_evaluated_native_support_not_scannet_ap",
                        "object_field_root": "objects",
                    }
                ),
                encoding="utf-8",
            )
            with memory_scene_rows.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["exported_tube_count", "cache_tube_count"])
                writer.writeheader()
                writer.writerow({"exported_tube_count": "60", "cache_tube_count": "100"})
                writer.writerow({"exported_tube_count": "80", "cache_tube_count": "100"})
            with reference_table.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "table4_label",
                        "source_row_id",
                        "status",
                        "AP",
                        "AP50",
                        "AP25",
                        "mean_predictions_per_scene",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "table4_label": "reference",
                        "source_row_id": "R0",
                        "status": "ok",
                        "AP": "0.1",
                        "AP50": "0.2",
                        "AP25": "0.3",
                        "mean_predictions_per_scene": "4",
                    }
                )
            rows, summary = build_static_bridge(
                memory_summary_path=memory_summary,
                memory_scene_rows_path=memory_scene_rows,
                reference_table_path=reference_table,
            )
        v42 = [row for row in rows if row["source_row_id"] == "V42-O-D4RT-native-support-memory"][0]
        self.assertEqual(v42["status"], "ok_native_support_not_ap")
        self.assertEqual(v42["AP"], "")
        self.assertEqual(v42["D4RT_hit_rate"], 0.7)
        self.assertFalse(summary["method_ap_goal_reached"])
        self.assertTrue(summary["v42_native_support_metric_proxy_pass"])

    def test_build_static_bridge_imports_gtgeo_diagnostic_ap_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            memory_summary = root / "memory_summary.json"
            memory_scene_rows = root / "memory_scene_rows.csv"
            reference_table = root / "reference.csv"
            gtgeo_summary = root / "gtgeo.json"
            memory_summary.write_text(
                json.dumps(
                    {
                        "scene_count": 1,
                        "expanded_object_field_count": 2,
                        "aggregate_tube_4D_ARI": 0.5,
                        "aggregate_tube_purity": 0.9,
                        "aggregate_tube_completeness": 0.6,
                        "native_support_metric_proxy_pass": True,
                        "phase8_gate_pass": False,
                        "AP_bridge_status": "not_evaluated_native_support_not_scannet_ap",
                    }
                ),
                encoding="utf-8",
            )
            with memory_scene_rows.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["exported_tube_count", "cache_tube_count"])
                writer.writeheader()
                writer.writerow({"exported_tube_count": "1", "cache_tube_count": "2"})
            with reference_table.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["table4_label", "source_row_id", "status"])
                writer.writeheader()
                writer.writerow({"table4_label": "reference", "source_row_id": "R0", "status": "ok"})
            gtgeo_summary.write_text(
                json.dumps(
                    {
                        "status": "OK_DIAGNOSTIC_GTGEO_AP_COMPUTED",
                        "memory_object_rows": "memory.csv",
                        "phase8_gate_pass": False,
                        "phase8_gate_blocker": "diagnostic only",
                        "aggregate": {
                            "scene_count": 1,
                            "AP": 0.12,
                            "AP50": 0.3,
                            "AP25": 0.4,
                            "mean_num_predictions": 2,
                            "mean_export_conflict_rate": 0.1,
                            "per_GT_best_IoU_ge_50": 0.2,
                        },
                    }
                ),
                encoding="utf-8",
            )
            rows, summary = build_static_bridge(
                memory_summary_path=memory_summary,
                memory_scene_rows_path=memory_scene_rows,
                reference_table_path=reference_table,
                gtgeo_summary_path=gtgeo_summary,
            )
        gtgeo = [row for row in rows if row["source_row_id"] == "V42-O-GTGeo-diagnostic"][0]
        self.assertEqual(gtgeo["status"], "ok_diagnostic_gtgeo_not_method")
        self.assertEqual(gtgeo["AP"], 0.12)
        self.assertTrue(gtgeo["forbidden_for_method_table"])
        self.assertEqual(summary["v42_gtgeo_diagnostic_status"], "ok_diagnostic_gtgeo_not_method")
        self.assertFalse(summary["method_ap_goal_reached"])

    def test_build_static_bridge_imports_calibrated_native_as_diagnostic_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            memory_summary = root / "memory_summary.json"
            memory_scene_rows = root / "memory_scene_rows.csv"
            reference_table = root / "reference.csv"
            calibrated_summary = root / "calibrated.json"
            memory_summary.write_text(
                json.dumps(
                    {
                        "scene_count": 1,
                        "expanded_object_field_count": 2,
                        "aggregate_tube_4D_ARI": 0.5,
                        "aggregate_tube_purity": 0.9,
                        "aggregate_tube_completeness": 0.6,
                        "native_support_metric_proxy_pass": True,
                        "phase8_gate_pass": False,
                        "AP_bridge_status": "not_evaluated_native_support_not_scannet_ap",
                    }
                ),
                encoding="utf-8",
            )
            with memory_scene_rows.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["exported_tube_count", "cache_tube_count"])
                writer.writeheader()
                writer.writerow({"exported_tube_count": "1", "cache_tube_count": "2"})
            with reference_table.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["table4_label", "source_row_id", "status"])
                writer.writeheader()
                writer.writerow({"table4_label": "reference", "source_row_id": "R0", "status": "ok"})
            calibrated_summary.write_text(
                json.dumps(
                    {
                        "status": "OK_CALIBRATED_NATIVE_DIAGNOSTIC_AP_COMPUTED",
                        "native_point_rows": "native.csv",
                        "phase8_gate_pass": False,
                        "phase8_gate_blocker": "diagnostic calibration",
                        "calibration_rows": [{"scene": "scene0000_00", "kept_anchor_count": 10}],
                        "best_by_AP": {
                            "AP": 0.2,
                            "AP50": 0.4,
                            "AP25": 0.6,
                            "num_predictions": 3,
                            "calibrated_native_hit_rate": 0.9,
                            "nn_radius": 0.05,
                            "per_GT_best_IoU_ge_50": 0.3,
                        },
                    }
                ),
                encoding="utf-8",
            )
            rows, summary = build_static_bridge(
                memory_summary_path=memory_summary,
                memory_scene_rows_path=memory_scene_rows,
                reference_table_path=reference_table,
                calibrated_native_summary_path=calibrated_summary,
            )
        calibrated = [row for row in rows if row["source_row_id"] == "V42-O-D4RT-native-Calibrated-diagnostic"][0]
        self.assertEqual(calibrated["status"], "OK_CALIBRATED_NATIVE_DIAGNOSTIC_AP_COMPUTED")
        self.assertEqual(calibrated["AP"], 0.2)
        self.assertTrue(calibrated["is_diagnostic_only"])
        self.assertTrue(calibrated["forbidden_for_method_table"])
        self.assertFalse(summary["method_ap_goal_reached"])
        self.assertEqual(summary["v42_calibrated_native_status"], "OK_CALIBRATED_NATIVE_DIAGNOSTIC_AP_COMPUTED")

    def test_build_static_bridge_imports_native_projection_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            memory_summary = root / "memory_summary.json"
            memory_scene_rows = root / "memory_scene_rows.csv"
            reference_table = root / "reference.csv"
            projection_summary = root / "projection.json"
            memory_summary.write_text(
                json.dumps(
                    {
                        "scene_count": 1,
                        "expanded_object_field_count": 2,
                        "aggregate_tube_4D_ARI": 0.5,
                        "aggregate_tube_purity": 0.9,
                        "aggregate_tube_completeness": 0.6,
                        "native_support_metric_proxy_pass": True,
                        "phase8_gate_pass": False,
                        "AP_bridge_status": "not_evaluated_native_support_not_scannet_ap",
                    }
                ),
                encoding="utf-8",
            )
            with memory_scene_rows.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["exported_tube_count", "cache_tube_count"])
                writer.writeheader()
                writer.writerow({"exported_tube_count": "1", "cache_tube_count": "2"})
            with reference_table.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["table4_label", "source_row_id", "status"])
                writer.writeheader()
                writer.writerow({"table4_label": "reference", "source_row_id": "R0", "status": "ok"})
            projection_summary.write_text(
                json.dumps(
                    {
                        "status": "NO_GO_NATIVE_PROJECTION_NOT_AP_MATERIALIZER_READY",
                        "cache_root": "cache",
                        "method_ap_materializer_ready": False,
                        "projection_all_scenes_gate_pass": False,
                        "rows": [
                            {
                                "projection_error_p90": 0.12,
                                "projection_within_0p02": 0.1,
                            }
                        ],
                        "scene_summaries": [
                            {
                                "scene": "scene0000_00",
                                "available_scene_transform_keys": [],
                                "has_mesh_vertex_ids": False,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            rows, summary = build_static_bridge(
                memory_summary_path=memory_summary,
                memory_scene_rows_path=memory_scene_rows,
                reference_table_path=reference_table,
                native_projection_summary_path=projection_summary,
            )
        projection = [row for row in rows if row["source_row_id"] == "V42-O-D4RT-native-projection-audit"][0]
        self.assertEqual(projection["status"], "NO_GO_NATIVE_PROJECTION_NOT_AP_MATERIALIZER_READY")
        self.assertEqual(projection["AP"], "")
        self.assertEqual(projection["native_projection_error_p90_mean"], 0.12)
        self.assertFalse(projection["forbidden_for_method_table"])
        self.assertFalse(summary["method_ap_goal_reached"])
        self.assertEqual(summary["v42_native_projection_status"], "NO_GO_NATIVE_PROJECTION_NOT_AP_MATERIALIZER_READY")

    def test_build_static_bridge_imports_native_tube_ap_without_scannet_ap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            memory_summary = root / "memory_summary.json"
            memory_scene_rows = root / "memory_scene_rows.csv"
            reference_table = root / "reference.csv"
            native_tube_summary = root / "native_tube.json"
            memory_summary.write_text(
                json.dumps(
                    {
                        "scene_count": 1,
                        "expanded_object_field_count": 2,
                        "aggregate_tube_4D_ARI": 0.5,
                        "aggregate_tube_purity": 0.9,
                        "aggregate_tube_completeness": 0.6,
                        "native_support_metric_proxy_pass": True,
                        "phase8_gate_pass": False,
                        "AP_bridge_status": "not_evaluated_native_support_not_scannet_ap",
                    }
                ),
                encoding="utf-8",
            )
            with memory_scene_rows.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["exported_tube_count", "cache_tube_count"])
                writer.writeheader()
                writer.writerow({"exported_tube_count": "1", "cache_tube_count": "2"})
            with reference_table.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["table4_label", "source_row_id", "status"])
                writer.writeheader()
                writer.writerow({"table4_label": "reference", "source_row_id": "R0", "status": "ok"})
            native_tube_summary.write_text(
                json.dumps(
                    {
                        "status": "OK_NATIVE_TUBE_AP_COMPUTED",
                        "memory_object_rows": "memory_object_field_rows.csv",
                        "prediction_count": 4,
                        "scenes": ["scene0000_00"],
                        "native_tube_AP": 0.25,
                        "native_tube_AP50": 0.5,
                        "native_tube_AP25": 0.75,
                        "per_gt_best_tube_iou_ge_50": 0.6,
                        "labeled_tube_coverage": 0.7,
                        "metric_scope": "d4rt_native_tube_space",
                        "phase8_gate_pass": False,
                        "phase8_gate_blocker": "native tube-space AP is not ScanNet AP",
                    }
                ),
                encoding="utf-8",
            )
            rows, summary = build_static_bridge(
                memory_summary_path=memory_summary,
                memory_scene_rows_path=memory_scene_rows,
                reference_table_path=reference_table,
                native_tube_ap_summary_path=native_tube_summary,
            )
        native = [row for row in rows if row["source_row_id"] == "V42-O-D4RT-native-tube-AP"][0]
        self.assertEqual(native["status"], "OK_NATIVE_TUBE_AP_COMPUTED")
        self.assertEqual(native["AP"], "")
        self.assertEqual(native["native_tube_AP50"], 0.5)
        self.assertFalse(native["is_scannet_ap_result"])
        self.assertFalse(native["forbidden_for_method_table"])
        self.assertFalse(summary["method_ap_goal_reached"])
        self.assertEqual(summary["v42_native_tube_ap_status"], "OK_NATIVE_TUBE_AP_COMPUTED")
        self.assertEqual(summary["v42_native_tube_AP50"], 0.5)

    def test_build_static_bridge_imports_native_tube_score_repair_as_separate_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            memory_summary = root / "memory_summary.json"
            memory_scene_rows = root / "memory_scene_rows.csv"
            reference_table = root / "reference.csv"
            score_summary = root / "native_tube_score.json"
            memory_summary.write_text(
                json.dumps(
                    {
                        "scene_count": 1,
                        "expanded_object_field_count": 2,
                        "aggregate_tube_4D_ARI": 0.5,
                        "aggregate_tube_purity": 0.9,
                        "aggregate_tube_completeness": 0.6,
                        "native_support_metric_proxy_pass": True,
                        "phase8_gate_pass": False,
                        "AP_bridge_status": "not_evaluated_native_support_not_scannet_ap",
                    }
                ),
                encoding="utf-8",
            )
            with memory_scene_rows.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["exported_tube_count", "cache_tube_count"])
                writer.writeheader()
                writer.writerow({"exported_tube_count": "1", "cache_tube_count": "2"})
            with reference_table.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["table4_label", "source_row_id", "status"])
                writer.writeheader()
                writer.writerow({"table4_label": "reference", "source_row_id": "R0", "status": "ok"})
            score_summary.write_text(
                json.dumps(
                    {
                        "status": "OK_NATIVE_TUBE_AP_COMPUTED",
                        "memory_object_rows": "memory_object_field_rows.csv",
                        "prediction_count": 4,
                        "scenes": ["scene0000_00"],
                        "score_mode": "confidence_log_tube_count",
                        "min_pred_tube_count": 1,
                        "max_pred_tube_count": 0,
                        "native_tube_AP": 0.1,
                        "native_tube_AP50": 0.2,
                        "native_tube_AP25": 0.3,
                        "metric_scope": "d4rt_native_tube_space",
                        "phase8_gate_pass": False,
                    }
                ),
                encoding="utf-8",
            )
            rows, summary = build_static_bridge(
                memory_summary_path=memory_summary,
                memory_scene_rows_path=memory_scene_rows,
                reference_table_path=reference_table,
                native_tube_score_repair_summary_path=score_summary,
            )
        repair = [row for row in rows if row["source_row_id"] == "V42-O-D4RT-native-tube-AP-score-repair"][0]
        self.assertEqual(repair["AP"], "")
        self.assertEqual(repair["native_tube_AP50"], 0.2)
        self.assertEqual(repair["native_tube_score_mode"], "confidence_log_tube_count")
        self.assertTrue(repair["repair_attempt"])
        self.assertFalse(repair["is_scannet_ap_result"])
        self.assertFalse(summary["method_ap_goal_reached"])
        self.assertEqual(summary["v42_native_tube_score_repair_AP50"], 0.2)


if __name__ == "__main__":
    unittest.main()
