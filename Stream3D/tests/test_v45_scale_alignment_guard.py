from __future__ import annotations

import unittest

from stream4d_native.scale_aware_d4rt_scene import scale_alignment_guard_audit


class V45ScaleAlignmentGuardTest(unittest.TestCase):
    def test_outside_10pct_pair_is_blocked(self) -> None:
        payload = scale_alignment_guard_audit(
            ratio_rows=[
                {
                    "scene": "s",
                    "window_pair": "1-2",
                    "scale_next_over_prev": "0.88",
                    "scale_aligned_within_10pct": "False",
                }
            ],
            window_rows=[
                {"allow_metric_merge": "True", "weak_alignment": "False"},
            ],
            block_outside_10pct=True,
        )
        self.assertEqual(payload["outside_10pct_scale_pair_count"], 1)
        self.assertEqual(payload["blocked_outside_10pct_pair_count"], 1)
        self.assertTrue(payload["gate"]["outside_10pct_pairs_zero_or_blocked"])


if __name__ == "__main__":
    unittest.main()

