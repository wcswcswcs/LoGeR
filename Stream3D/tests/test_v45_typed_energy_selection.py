from __future__ import annotations

import unittest

from stream4d_native.stage1_scale_aware_typed_assembly import preconditioned_stage1_status


class V45TypedEnergySelectionTest(unittest.TestCase):
    def test_stage1_precondition_blocks_when_role_fails(self) -> None:
        ok = {"gate": {"pass": True}}
        failed = {"gate": {"pass": False}}
        status = preconditioned_stage1_status(descriptor=ok, role=failed, operations=ok, temporal=ok)
        self.assertFalse(status["stage1_run_as_method"])
        self.assertIn("role_lattice_gate_failed", status["blocked_reasons"])


if __name__ == "__main__":
    unittest.main()

