from __future__ import annotations

import unittest

from stream4d_native.v49_mosaic_stage1 import evaluate_component_assignment


class TestV49NoD4RTBirth(unittest.TestCase):
    def test_assignment_reports_no_d4rt_birth_or_maskless_object(self) -> None:
        rows = [
            {"scene": "s", "predicted_component_object_id": "c0", "mask_observation_id": "m0", "frame_id": "0", "diagnostic_gt_instance": "1"}
        ]
        metrics = evaluate_component_assignment(rows, {"s|c0": "s|o0"})
        self.assertEqual(metrics["birth_from_d4rt_tube_count"], 0)
        self.assertEqual(metrics["maskless_object_count"], 0)


if __name__ == "__main__":
    unittest.main()
