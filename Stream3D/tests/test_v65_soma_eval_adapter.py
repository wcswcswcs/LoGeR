from __future__ import annotations

import unittest

from stream4d_native.v65_soma_eval_adapter import (
    V65SOMAEvalAdapterConfig,
    build_eval_adapter_summary,
    build_scene_object_dicts,
    summarize_scene_object_dicts,
)


class V65SOMAEvalAdapterTest(unittest.TestCase):
    def test_builds_per_scene_object_mask_lists_from_support_rows(self) -> None:
        object_rows = [
            {"scene_id": "scene0000_00", "history_id": "scene0000_00|objA", "object_id": "objA", "confidence": "0.7"},
            {"scene_id": "scene0000_00", "history_id": "scene0000_00|objB", "object_id": "objB", "confidence": "0.8"},
        ]
        support_rows = [
            {
                "scene_id": "scene0000_00",
                "history_id": "scene0000_00|objA",
                "frame_id": "5",
                "observed_mask_id": "12",
                "support_mask_observation_id": "m:scene0000_00:5:12",
            },
            {
                "scene_id": "scene0000_00",
                "history_id": "scene0000_00|objA",
                "frame_id": "",
                "observed_mask_id": "",
                "support_mask_observation_id": "m:scene0000_00:6:13",
            },
            {
                "scene_id": "scene0000_00",
                "history_id": "scene0000_00|objA",
                "frame_id": "5",
                "observed_mask_id": "12",
                "support_mask_observation_id": "m:scene0000_00:5:12",
            },
            {
                "scene_id": "scene0000_00",
                "history_id": "scene0000_00|objB",
                "frame_id": "7",
                "observed_mask_id": "3",
                "support_mask_observation_id": "m:scene0000_00:7:3",
            },
        ]
        scene_dicts = build_scene_object_dicts(object_rows, support_rows)
        self.assertEqual(set(scene_dicts), {"scene0000_00"})
        self.assertEqual(len(scene_dicts["scene0000_00"]), 2)
        masks_a = scene_dicts["scene0000_00"][0]["mask_list"]
        masks_b = scene_dicts["scene0000_00"][1]["mask_list"]
        self.assertEqual([(m[0], m[1]) for m in masks_a], [(5, 12), (6, 13)])
        self.assertEqual([(m[0], m[1]) for m in masks_b], [(7, 3)])
        summary = summarize_scene_object_dicts(scene_dicts)
        self.assertEqual(summary["object_count"], 2)
        self.assertEqual(summary["mask_observation_count"], 3)

    def test_summary_is_diagnostic_only_not_method_ap(self) -> None:
        scene_dicts = {"scene0000_00": {0: {"mask_list": [(1, 2, 1.0)]}}}
        cfg = V65SOMAEvalAdapterConfig(output_config="synthetic_eval")
        summary = build_eval_adapter_summary(
            cfg=cfg,
            scene_dicts=scene_dicts,
            scene_rows=[{"scene": "scene0000_00", "ok": True, "num_exported_objects": 1, "num_exported_points": 10}],
        )
        self.assertFalse(summary["uses_gt_for_prediction"])
        self.assertFalse(summary["uses_gt_geometry_for_inference"])
        self.assertTrue(summary["uses_rgbd_for_evaluation_support"])
        self.assertTrue(summary["uses_rgbd_pose_mesh_for_export"])
        self.assertTrue(summary["is_diagnostic_only"])
        self.assertTrue(summary["forbidden_for_method_table"])
        self.assertFalse(summary["method_ap_available"])
        self.assertTrue(summary["diagnostic_ap_export_available"])


if __name__ == "__main__":
    unittest.main()
