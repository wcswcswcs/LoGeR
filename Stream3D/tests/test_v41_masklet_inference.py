from __future__ import annotations

import unittest

from stream4d_native.geosemantic_masklet_refiner import split_mixed_same_frame_masks
from stream4d_native.semantic_masklet_inference import (
    MaskletMeasurement,
    evaluate_masklet_assignments,
    infer_semantic_masklets,
)


class V41MaskletInferenceTests(unittest.TestCase):
    def _measurements(self) -> list[MaskletMeasurement]:
        return [
            MaskletMeasurement(0, 0, 1, (1.0, 0.0), "a", 10),
            MaskletMeasurement(1, 1, 1, (0.98, 0.02), "a", 10),
            MaskletMeasurement(2, 2, 7, (0.99, 0.01), "a", 10),
            MaskletMeasurement(3, 0, 2, (0.0, 1.0), "b", 20),
            MaskletMeasurement(4, 1, 2, (0.02, 0.98), "b", 20),
            MaskletMeasurement(5, 2, 2, (0.01, 0.99), "b", 20),
        ]

    def test_full_semantic_material_masklets_improve_completeness_without_conflict(self) -> None:
        measurements = self._measurements()
        e0 = infer_semantic_masklets(measurements, use_visual=False, use_d4rt=False)
        e3 = infer_semantic_masklets(measurements, use_visual=True, use_d4rt=True)
        m0 = evaluate_masklet_assignments(measurements, e0)
        m3 = evaluate_masklet_assignments(measurements, e3)
        self.assertGreaterEqual(m3["masklet_purity"], m0["masklet_purity"])
        self.assertGreater(m3["masklet_completeness"], m0["masklet_completeness"])
        self.assertEqual(m3["same_frame_conflict_violation"], 0)

    def test_mixed_same_frame_mask_split_is_detected_by_material_support(self) -> None:
        measurements = [
            MaskletMeasurement(0, 0, 1, (1.0, 0.0), "a", 10),
            MaskletMeasurement(1, 0, 1, (0.0, 1.0), "b", 20),
        ]
        splits = split_mixed_same_frame_masks(measurements)
        self.assertEqual(len(splits[(0, 1)]), 2)


if __name__ == "__main__":
    unittest.main()

