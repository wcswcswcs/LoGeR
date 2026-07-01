from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from stream4d_native.v47_common import write_csv
from stream4d_native.v65_soma_object_bank import (
    V65SOMAObjectBankConfig,
    build_v65_soma_object_bank,
    write_v65_soma_object_bank,
)


class V65SOMAObjectBankTest(unittest.TestCase):
    def test_object_bank_keeps_object_material_and_support_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            history_id = "scene0000_00|synthetic|obj00001"
            object_path = root / "object_rows.csv"
            material_path = root / "material_rows.csv"
            carrier_path = root / "carrier_rows.csv"
            write_csv(
                object_path,
                [
                    {
                        "scene_id": "scene0000_00",
                        "history_id": history_id,
                        "semantic_modes": [],
                        "confirmed_material_ids": ["a:c001"],
                        "tentative_material_ids": [],
                        "shared_material_ids": [],
                        "quarantine_material_ids": [],
                        "unknown_material_ids": [],
                        "supporting_mask_ids": ["mask0"],
                        "supporting_frame_ids": [5],
                        "state_timeline": [{"material_id": "a:c001", "state": "confirmed"}],
                        "confidence": 0.9,
                        "score_policy": "synthetic",
                        "material_count": 1,
                    }
                ],
            )
            write_csv(
                material_path,
                [
                    {
                        "scene_id": "scene0000_00",
                        "material_id": "a:c001",
                        "component_id": "c001",
                        "carrier_id_if_available": "42",
                        "history_id": history_id,
                        "state": "confirmed",
                        "state_confidence": 1.0,
                        "support_observation_ids": ["obsA"],
                        "source_evidence_types": ["synthetic"],
                        "frame_ids": [5],
                        "uv_tracks_if_available": [[0.1, 0.2]],
                        "xyz_tracks_if_available": [[1.0, 2.0, 3.0]],
                        "uses_gt_for_prediction": False,
                        "uses_rgbd_pose_mesh_for_export": False,
                    }
                ],
            )
            write_csv(
                carrier_path,
                [
                    {
                        "variant": "synthetic",
                        "objectlet_id": "synthetic-objectlet",
                        "scene": "scene0000_00",
                        "chunk_id": "scene0000_00:chunk000",
                        "component_id": "c001",
                        "carrier_global_id": "scene0000_00:42",
                        "carrier_id": "42",
                        "frame_id": 5,
                        "carrier_observation_chunk_id": 0,
                        "submap_id": 0,
                        "window_index": 0,
                        "carrier_index": 7,
                        "uv_x": 0.1,
                        "uv_y": 0.2,
                        "confidence": 0.99,
                        "visibility_prob": 0.98,
                        "visible": True,
                        "valid": True,
                        "valid_uv": True,
                        "observed_mask_id": "mask0",
                        "support_mask_observation_id": "obsA",
                        "native_support_kind": "d4rt_carrier_global_id",
                        "is_scannet_ap_export": False,
                        "uses_gt_for_prediction": False,
                        "uses_gt_for_diagnostic_labels": False,
                        "uses_rgbd_pose_mesh_for_export": False,
                    }
                ],
            )
            result = build_v65_soma_object_bank(
                V65SOMAObjectBankConfig(
                    object_rows_path=object_path,
                    material_rows_path=material_path,
                    carrier_rows_path=carrier_path,
                    output_root=root / "out",
                    allow_unverified_component_join=True,
                )
            )
            summary = result["summary"]
            self.assertEqual(summary["object_count"], 1)
            self.assertEqual(summary["material_assignment_count"], 1)
            self.assertGreaterEqual(summary["object_support_row_count"], 3)
            self.assertTrue(summary["native_support_join_available"])
            self.assertEqual(summary["blockers"], [])
            self.assertTrue(summary["method_safe_inference_artifact"])
            self.assertTrue(summary["is_method_result"])
            self.assertFalse(summary["is_scannet_ap_export"])
            self.assertFalse(summary["uses_gt_for_prediction"])
            self.assertFalse(summary["uses_rgbd_pose_mesh_for_export"])
            self.assertTrue(result["object_bank_rows"][0]["has_view_mask_support"])
            self.assertTrue(result["object_bank_rows"][0]["has_native_point_or_carrier_support"])
            self.assertTrue(
                any(
                    row["support_kind"] == "experimental_native_carrier_component_id_overlap"
                    for row in result["object_support_rows"]
                )
            )
            paths = write_v65_soma_object_bank(result, root / "out")
            self.assertTrue((root / "out" / "soma_object_bank_summary.json").exists())
            self.assertEqual(paths["summary"], str(root / "out" / "soma_object_bank_summary.json"))

    def test_missing_support_is_blocked_instead_of_fabricated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            history_id = "scene0000_00|synthetic|obj00002"
            object_path = root / "object_rows.csv"
            material_path = root / "material_rows.csv"
            carrier_path = root / "carrier_rows.csv"
            write_csv(
                object_path,
                [
                    {
                        "scene_id": "scene0000_00",
                        "history_id": history_id,
                        "semantic_modes": [],
                        "confirmed_material_ids": ["a:c002"],
                        "tentative_material_ids": [],
                        "shared_material_ids": [],
                        "quarantine_material_ids": [],
                        "unknown_material_ids": [],
                        "supporting_mask_ids": [],
                        "supporting_frame_ids": [],
                        "state_timeline": [{"material_id": "a:c002", "state": "confirmed"}],
                        "confidence": 1.0,
                        "score_policy": "synthetic",
                        "material_count": 1,
                    }
                ],
            )
            write_csv(
                material_path,
                [
                    {
                        "scene_id": "scene0000_00",
                        "material_id": "a:c002",
                        "component_id": "c002",
                        "carrier_id_if_available": "c002",
                        "history_id": history_id,
                        "state": "confirmed",
                        "state_confidence": 1.0,
                        "support_observation_ids": [],
                        "source_evidence_types": ["synthetic"],
                        "frame_ids": [],
                        "uv_tracks_if_available": [],
                        "xyz_tracks_if_available": [],
                        "uses_gt_for_prediction": False,
                        "uses_rgbd_pose_mesh_for_export": False,
                    }
                ],
            )
            write_csv(carrier_path, [])
            result = build_v65_soma_object_bank(
                V65SOMAObjectBankConfig(
                    object_rows_path=object_path,
                    material_rows_path=material_path,
                    carrier_rows_path=carrier_path,
                    output_root=root / "out",
                )
            )
            summary = result["summary"]
            self.assertEqual(summary["object_count"], 1)
            self.assertEqual(summary["object_support_row_count"], 0)
            self.assertFalse(summary["native_support_join_available"])
            self.assertIn("object_to_view_mask_or_point_support_missing", summary["blockers"])
            self.assertIn("verified_object_to_native_carrier_mapping_missing", summary["blockers"])
            self.assertFalse(summary["gate"]["has_view_mask_or_native_point_support"])
            self.assertEqual(result["object_support_rows"], [])

    def test_component_overlap_is_not_verified_support_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            history_id = "scene0000_00|synthetic|obj00003"
            object_path = root / "object_rows.csv"
            material_path = root / "material_rows.csv"
            carrier_path = root / "carrier_rows.csv"
            write_csv(
                object_path,
                [
                    {
                        "scene_id": "scene0000_00",
                        "history_id": history_id,
                        "semantic_modes": [],
                        "confirmed_material_ids": ["a:c003"],
                        "tentative_material_ids": [],
                        "shared_material_ids": [],
                        "quarantine_material_ids": [],
                        "unknown_material_ids": [],
                        "supporting_mask_ids": [],
                        "supporting_frame_ids": [],
                        "state_timeline": [{"material_id": "a:c003", "state": "confirmed"}],
                        "confidence": 1.0,
                        "score_policy": "synthetic",
                        "material_count": 1,
                    }
                ],
            )
            write_csv(
                material_path,
                [
                    {
                        "scene_id": "scene0000_00",
                        "material_id": "a:c003",
                        "component_id": "c003",
                        "carrier_id_if_available": "c003",
                        "history_id": history_id,
                        "state": "confirmed",
                        "state_confidence": 1.0,
                        "support_observation_ids": [],
                        "source_evidence_types": ["synthetic"],
                        "frame_ids": [],
                        "uv_tracks_if_available": [],
                        "xyz_tracks_if_available": [],
                        "uses_gt_for_prediction": False,
                        "uses_rgbd_pose_mesh_for_export": False,
                    }
                ],
            )
            write_csv(
                carrier_path,
                [
                    {
                        "scene": "scene0000_00",
                        "component_id": "c003",
                        "carrier_global_id": "scene0000_00:100",
                        "carrier_id": "100",
                        "frame_id": 5,
                        "observed_mask_id": "9",
                        "support_mask_observation_id": "scene0000_00:5:9",
                        "uv_x": 0.25,
                        "uv_y": 0.75,
                        "is_scannet_ap_export": False,
                        "uses_gt_for_prediction": False,
                        "uses_rgbd_pose_mesh_for_export": False,
                    }
                ],
            )
            result = build_v65_soma_object_bank(
                V65SOMAObjectBankConfig(
                    object_rows_path=object_path,
                    material_rows_path=material_path,
                    carrier_rows_path=carrier_path,
                    output_root=root / "out",
                )
            )
            summary = result["summary"]
            self.assertEqual(summary["unverified_component_id_overlap_count"], 1)
            self.assertFalse(summary["unverified_component_id_overlap_is_support"])
            self.assertEqual(summary["object_support_row_count"], 0)
            self.assertEqual(result["object_support_rows"], [])


if __name__ == "__main__":
    unittest.main()
