from __future__ import annotations

import unittest

from stream4d_native.material_residual_matching import run_material_residual


class V432MaterialResidualTest(unittest.TestCase):
    def test_material_residual_without_controls_is_not_discriminative(self) -> None:
        summary = run_material_residual(".", semantic_summary={"metrics": {"birth_from_d4rt_tube_count": 0}})
        self.assertEqual(summary["status"], "NO_GO_MATERIAL_NOT_DISCRIMINATIVE")
        self.assertEqual(summary["accepted_corrections"], [])
        self.assertFalse(summary["gate"]["pass"])


if __name__ == "__main__":
    unittest.main()
