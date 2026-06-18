from __future__ import annotations

import unittest

import numpy as np

from stream4d_native.object_field import ObjectField
from stream4d_native.object_field_native_export import (
    NativeObjectFieldExportConfig,
    export_object_fields_to_native_points,
)
from stream4d_native.object_tube_io import TubeRecord


def _tube(
    tube_id: int,
    *,
    visibility: tuple[float, ...] = (0.9, 0.2, 0.8),
    confidence: tuple[float, ...] = (0.8, 0.9, 0.1),
    coordinate_frame: str = "d4rt_canonical",
    alignment_source: str = "same_chunk_identity",
    allow_metric_merge: bool = True,
    pass_gate: bool = True,
) -> TubeRecord:
    n = len(visibility)
    xyz = np.asarray([[float(tube_id), float(idx), 0.25] for idx in range(n)], dtype=np.float32)
    return TubeRecord(
        tube_id=tube_id,
        persistent_tube_id=tube_id,
        chunk_id=0,
        submap_id=0,
        source_frame_global=0,
        source_xy=(10, 10),
        source_uv=(0.5, 0.5),
        target_frames_global=np.arange(n, dtype=np.int64),
        uv=np.asarray([[0.5, 0.5] for _ in range(n)], dtype=np.float32),
        visibility=np.asarray(visibility, dtype=np.float32),
        confidence=np.asarray(confidence, dtype=np.float32),
        xyz_local=xyz.copy(),
        xyz_ref0=xyz.copy(),
        xyz_canonical=xyz.copy() if coordinate_frame == "d4rt_canonical" else None,
        alignment_quality={"pass_gate": bool(pass_gate)},
        coordinate_frame=coordinate_frame,
        allow_metric_merge=bool(allow_metric_merge),
        alignment_source=alignment_source,
    )


class V41NativeObjectFieldExportTests(unittest.TestCase):
    def test_exports_only_visible_confident_canonical_points(self) -> None:
        fields = [
            ObjectField(
                object_id=7,
                primary_field_id=0,
                semantic_masklet_ids=[1, 2],
                attached_tube_ids=[10],
                confidence=0.95,
            )
        ]
        result = export_object_fields_to_native_points(
            fields,
            [_tube(10)],
            config=NativeObjectFieldExportConfig(min_visibility=0.5, min_confidence=0.5),
        )
        self.assertTrue(result.summary["native_export_smoke_pass"])
        self.assertEqual(result.summary["native_point_count"], 1)
        self.assertEqual(result.summary["exported_tube_count"], 1)
        self.assertEqual(result.point_rows[0]["object_id"], 7)
        self.assertFalse(result.summary["uses_gt_for_prediction"])
        self.assertFalse(result.summary["uses_rgbd_for_prediction"])
        self.assertFalse(result.summary["uses_scannet_mesh_for_prediction"])
        self.assertEqual(result.summary["AP_bridge_status"], "not_evaluated_native_support_not_scannet_ap")

    def test_rejects_forbidden_geometry_and_semanticless_fields(self) -> None:
        fields = [
            ObjectField(0, 0, [], [10], 0.99),
            ObjectField(1, 1, [3], [11, 12, 13], 0.88),
        ]
        tubes = [
            _tube(10),
            _tube(11, alignment_source="eval_gt_sim3"),
            _tube(12, coordinate_frame="ref0_local"),
            _tube(13, alignment_source="submap_identity_after_failed_self_sim3", allow_metric_merge=False, pass_gate=False),
        ]
        result = export_object_fields_to_native_points(fields, tubes)
        self.assertFalse(result.summary["native_export_smoke_pass"])
        self.assertEqual(result.summary["invalid_field_count_by_reason"]["missing_semantic_masklet_birth"], 1)
        reasons = result.summary["rejected_tube_count_by_reason"]
        self.assertEqual(reasons["eval_aligned_geometry_forbidden"], 1)
        self.assertEqual(reasons["noncanonical_coordinate_frame"], 1)
        self.assertEqual(reasons["metric_merge_disabled_by_alignment"], 1)
        self.assertEqual(result.summary["native_point_count"], 0)


if __name__ == "__main__":
    unittest.main()

