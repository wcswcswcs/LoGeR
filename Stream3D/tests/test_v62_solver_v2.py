from __future__ import annotations

import unittest

from stream4d_native.v62_solver_v2 import build_v62_solver_v2


class V62SolverV2Test(unittest.TestCase):
    def test_full_solver_passes_and_semantic_only_does_not_confirm_core(self) -> None:
        result = build_v62_solver_v2()
        self.assertTrue(result["summary"]["gate"]["pass"])
        semantic_confirmed = sum(1 for row in result["semantic_only_state_rows"] if row["state"] == "confirmed")
        self.assertEqual(semantic_confirmed, 0)


if __name__ == "__main__":
    unittest.main()

