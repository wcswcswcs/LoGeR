from __future__ import annotations

import unittest

from stream4d_native.v49_mosaic_stage1 import build_eval_aligned_ap


class TestV49EvalAlignmentNotMethod(unittest.TestCase):
    def test_eval_alignment_rows_never_use_gt_for_prediction(self) -> None:
        payload = build_eval_aligned_ap()
        self.assertTrue(all(not row["uses_gt_for_prediction"] for row in payload["ap_rows"]))


if __name__ == "__main__":
    unittest.main()
