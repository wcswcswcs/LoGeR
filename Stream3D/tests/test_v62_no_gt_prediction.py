from __future__ import annotations

import unittest

from stream4d_native.v62_decircularization import build_v62_decircularization
from stream4d_native.v62_native_field import build_v62_native_field
from stream4d_native.v62_solver_v2 import build_v62_solver_v2


class V62NoGTPredictionTest(unittest.TestCase):
    def test_method_summaries_do_not_use_gt_for_prediction(self) -> None:
        for result in [build_v62_decircularization(), build_v62_solver_v2(), build_v62_native_field()]:
            summary = result.get("summary") or {}
            self.assertFalse(summary.get("uses_gt_for_prediction"))


if __name__ == "__main__":
    unittest.main()

