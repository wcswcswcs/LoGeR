from __future__ import annotations

import unittest

from stream4d_native.v62_stress_regen import build_v62_stress_regen


class V62StressRegenTest(unittest.TestCase):
    def test_regen_stress_has_at_least_three_mask_only_wins(self) -> None:
        result = build_v62_stress_regen()
        self.assertGreaterEqual(result["summary"]["stress_regen_real_minus_mask_only_pass_count"], 3)


if __name__ == "__main__":
    unittest.main()

