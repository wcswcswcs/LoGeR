from __future__ import annotations

import unittest

from stream4d_native.v65_visualization_export import (
    _load_prediction_overlay,
    _load_scene_mesh,
    check_viser_import,
    export_v65_visualization,
)


class V65VisualizationExportTest(unittest.TestCase):
    def test_viser_import_records_python_executable(self) -> None:
        status = check_viser_import()
        self.assertIn("python_executable", status)
        self.assertIn("viser_import_ok", status)

    def test_export_records_ownership_3d_blocker(self) -> None:
        payload = export_v65_visualization()
        summary = payload["summary"]
        self.assertEqual(summary["scene_count"], 5)
        self.assertTrue(summary["gate"]["D4RT_geometry_layer_exported"])
        self.assertTrue(summary["gate"]["SOMA_ownership_summary_layer_exported"])
        self.assertFalse(summary["gate"]["SOMA_semantic_ownership_3d_layer_visible"])

    def test_prediction_overlay_records_exact_ap_inputs(self) -> None:
        points, _colors, _mesh_path = _load_scene_mesh("scene0050_00")
        overlay = _load_prediction_overlay(
            "scene0050_00",
            "v64r2_d4rt_chunk_scale_first_ap_probe5_g11",
            points.shape[0],
        )
        self.assertEqual(overlay["mask_contract"], "full_scene_vertex_mask")
        self.assertEqual(overlay["pred_vertex_count"], 14775)
        self.assertEqual(overlay["pre_points_count"], 14775)
        self.assertEqual(overlay["pred_instance_count"], 94)
        self.assertIn("data/prediction/v64r2_d4rt_chunk_scale_first_ap_probe5_g11_class_agnostic", overlay["pred_path"])
        self.assertIn("data/TMP/v64r2_d4rt_chunk_scale_first_ap_probe5_g11", overlay["pre_points_path"])
        self.assertEqual(overlay["error"], "")


if __name__ == "__main__":
    unittest.main()
