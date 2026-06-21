from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from stream4d_native.v53_ap_diagnostic import ap_diagnostic_identity_gate, ap_smoke_gate, native_method_export_repair_audit


class V53APDiagnosticTest(unittest.TestCase):
    def test_ap_gates_require_non_gt_and_exportable_objects(self) -> None:
        best = {
            "4D_ARI": 0.5,
            "4D_purity": 0.9,
            "4D_completeness": 0.6,
            "conflict_rate": 0.1,
            "mean_predictions_per_scene": 10,
            "maskless_object_count": 0,
            "birth_from_d4rt_tube_count": 0,
            "uses_gt_for_prediction": False,
        }
        self.assertTrue(ap_smoke_gate(best, 3, True)["pass"])
        self.assertTrue(ap_diagnostic_identity_gate(best)["pass"])
        best["uses_gt_for_prediction"] = True
        self.assertFalse(ap_smoke_gate(best, 3, True)["pass"])
        self.assertFalse(ap_diagnostic_identity_gate(best)["pass"])

    def test_native_repair_audit_blocks_without_native_mapping_fields(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            carrier = root / "carrier.csv"
            objectlets = root / "objectlets.csv"
            chunks = root / "chunks.csv"
            v42_summary = root / "v42.json"
            native_exporter = root / "native.py"
            carrier.write_text("scene,carrier_id,uv_x,uv_y,observed_mask_id\ns,1,0.1,0.2,m\n", encoding="utf-8")
            objectlets.write_text("variant,objectlet_id,source_mask_observation_id\nv,o,m\n", encoding="utf-8")
            chunks.write_text("chunk_id,scene,component_id\nc,s,k\n", encoding="utf-8")
            v42_summary.write_text(
                '{"native_export_smoke_pass": true, "native_point_count": 3, '
                '"AP_bridge_status": "not_evaluated_native_support_not_scannet_ap", '
                '"uses_gt_for_prediction": false, "uses_gt_for_scoring": true}\n',
                encoding="utf-8",
            )
            native_exporter.write_text("# stub\n", encoding="utf-8")
            audit = native_method_export_repair_audit(
                carrier_table_path=carrier,
                objectlet_rows_path=objectlets,
                chunk_component_rows_path=chunks,
                v53_native_carrier_summary_path=root / "missing_v53_native_carrier_summary.json",
                v42_native_summary_path=v42_summary,
                native_exporter_path=native_exporter,
            )
        self.assertTrue(audit["repair_attempted"])
        self.assertFalse(audit["method_safe_native_ap_export_available"])
        self.assertEqual(
            audit["repair_result"],
            "blocked_current_v53_artifacts_missing_native_point_or_mesh_vertex_mapping",
        )

    def test_native_repair_audit_recognizes_d4rt_support_without_promoting_ap(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            carrier = root / "carrier.csv"
            objectlets = root / "objectlets.csv"
            chunks = root / "chunks.csv"
            v53_summary = root / "native_carrier_summary.json"
            carrier.write_text("scene,carrier_id,uv_x,uv_y,observed_mask_id\ns,1,0.1,0.2,m\n", encoding="utf-8")
            objectlets.write_text("variant,objectlet_id,source_mask_observation_id\nv,o,m\n", encoding="utf-8")
            chunks.write_text("chunk_id,scene,component_id\nc,s,k\n", encoding="utf-8")
            v53_summary.write_text(
                '{"native_carrier_materialization_pass": true, '
                '"method_safe_native_support_available": true, '
                '"uses_gt_for_prediction": false, '
                '"uses_gt_for_diagnostic_labels": false, '
                '"uses_rgbd_pose_mesh_for_export": false, '
                '"is_scannet_ap_export": false, '
                '"native_support_kind": "d4rt_carrier_global_id", '
                '"native_observation_row_count": 4, '
                '"native_unique_carrier_count": 3, '
                '"AP_bridge_status": "not_evaluated_native_carrier_support_not_scannet_ap"}\n',
                encoding="utf-8",
            )
            audit = native_method_export_repair_audit(
                carrier_table_path=carrier,
                objectlet_rows_path=objectlets,
                chunk_component_rows_path=chunks,
                v53_native_carrier_summary_path=v53_summary,
            )
        self.assertTrue(audit["method_safe_native_support_available"])
        self.assertTrue(audit["v53_native_carrier_support_available"])
        self.assertFalse(audit["method_safe_native_ap_export_available"])
        self.assertEqual(
            audit["repair_result"],
            "native_d4rt_carrier_support_available_scannet_ap_still_blocked",
        )


if __name__ == "__main__":
    unittest.main()
