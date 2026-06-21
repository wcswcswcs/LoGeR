from __future__ import annotations

import unittest
from collections import Counter

import numpy as np

from stream4d_native.v44_native_typed_assembly import (
    MaskKey,
    MaskMeasurement,
    Objectlet,
    V44Config,
    _assign_tubes_to_components,
    _infer_role,
    cluster_metrics,
    stage1_gate,
)


class V44NativeTypedAssemblyTests(unittest.TestCase):
    def _measurement(self, *, area_ratio: float, rgb_variance: float, support: int) -> MaskMeasurement:
        area = max(1, int(area_ratio * 100000))
        return MaskMeasurement(
            index=0,
            scene="s",
            frame_id=0,
            mask_id=1,
            area=area,
            image_area=100000,
            bbox_xyxy=(0, 0, 10, 10),
            center_xy=(5.0, 5.0),
            area_ratio=area_ratio,
            mean_rgb=np.zeros((3,), dtype=np.float32),
            std_rgb=np.ones((3,), dtype=np.float32) * rgb_variance,
            feature=np.ones((6,), dtype=np.float32),
            rgb_variance=rgb_variance,
            core_nonempty=True,
            boundary_nonempty=True,
            boundary_contrast=0.05,
            boundary_gradient=0.05,
            prototype_count=4 if rgb_variance > 0.2 else 1,
            d4rt_support_count=support,
            d4rt_observation_count=support,
        )

    def test_large_high_variance_mask_is_mixed(self) -> None:
        config = V44Config()
        meas = self._measurement(area_ratio=0.08, rgb_variance=0.24, support=20)
        _infer_role(meas, config)
        self.assertEqual(meas.role, "mixed")

    def test_unknown_tubes_do_not_fabricate_good_completeness(self) -> None:
        pred = {1: 10, 2: 1_000_000, 3: 1_000_001}
        gt = {1: 5, 2: 5, 3: 5}
        metrics = cluster_metrics(pred, gt)
        self.assertLess(metrics["completeness"], 1.0)

    def test_assignment_keeps_d4rt_birth_zero_by_construction(self) -> None:
        obj = Objectlet(
            objectlet_id=0,
            scene="s",
            frame_id=0,
            primary=MaskKey(0, 1),
            role="core",
            feature=np.ones((6,), dtype=np.float32),
            center_xy=(0.0, 0.0),
            area=10,
            support=Counter({7: 2}),
        )
        labels, diag = _assign_tubes_to_components(
            [obj],
            {0: [0]},
            {7: 42, 8: 42},
            unknown_label_base=1_000_000,
        )
        self.assertEqual(labels[7], 0)
        self.assertGreaterEqual(labels[8], 1_000_000)
        self.assertEqual(diag["assigned_labeled_tube_count"], 1)

    def test_stage1_gate_requires_all_hard_constraints(self) -> None:
        gate = stage1_gate(
            {
                "4D_ARI": 0.50,
                "4D_purity": 0.90,
                "4D_completeness": 0.60,
                "temporal_span_mean": 2.0,
                "scene0081_ARI": 0.30,
                "mean_predictions_per_scene": 100,
                "duplicate_rate": 0.0,
                "conflict_rate": 0.0,
                "unknown_tube_ratio": 0.1,
                "birth_from_d4rt_tube_count": 0,
                "mixed_birth_count": 1,
            }
        )
        self.assertFalse(gate["stage1_significant_gate_pass"])
        self.assertFalse(gate["no_mixed_birth_pass"])


if __name__ == "__main__":
    unittest.main()
