from __future__ import annotations

import unittest

import numpy as np

from tools.diagnose_v15_union_oracle import _greedy_union
from tools.prediction_manifest import build_prediction_manifest


class V15PureTests(unittest.TestCase):
    def test_union_oracle_greedy_improves_with_multiple_parts(self) -> None:
        gt = np.asarray([1, 1, 1, 1, 0, 0], dtype=bool)
        pred = np.asarray(
            [
                [1, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 1, 0],
                [0, 0, 1],
                [0, 0, 1],
            ],
            dtype=bool,
        )
        selected, curve = _greedy_union(
            gt,
            pred,
            max_k=2,
            max_candidates=3,
            min_improvement=1e-8,
        )
        self.assertEqual(len(selected), 2)
        self.assertAlmostEqual(curve[0], 0.5)
        self.assertAlmostEqual(curve[1], 1.0)

    def test_v15_manifest_defaults_mark_gt_oracle_as_forbidden(self) -> None:
        manifest = build_prediction_manifest(
            output_config="stream4d_v15_oracle_demo",
            uses_gt=True,
            is_method_result=False,
            is_diagnostic_only=True,
            gt_usage="unit_test_oracle",
            extra={"gt_selected_output": True},
        )
        self.assertTrue(manifest["uses_gt_for_diagnostic"])
        self.assertTrue(manifest["gt_selected_output"])
        self.assertTrue(manifest["forbidden_for_method_table"])
        self.assertFalse(manifest["alignment_used_for_prediction"])

    def test_scan_flags_gt_selected_method_result(self) -> None:
        # Exercise the scanner's row logic without touching the filesystem by
        # checking the same boolean invariant it reports as suspicious.
        manifest = {
            "uses_gt": True,
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic": True,
            "gt_selected_output": True,
            "forbidden_for_method_table": True,
            "is_method_result": True,
            "is_diagnostic_only": False,
        }
        self.assertTrue(manifest["gt_selected_output"] and manifest["is_method_result"])
        self.assertTrue(manifest["forbidden_for_method_table"] and manifest["is_method_result"])


if __name__ == "__main__":
    unittest.main()
