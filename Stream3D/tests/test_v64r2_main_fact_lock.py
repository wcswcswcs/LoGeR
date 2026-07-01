from __future__ import annotations

import unittest

from stream4d_native.v64r2_main_fact_lock import build_v64r2_main_fact_lock


class V64R2MainFactLockTest(unittest.TestCase):
    def test_v62_fact_lock_passes_from_landed_artifacts(self) -> None:
        payload = build_v64r2_main_fact_lock()
        self.assertTrue(payload["gate"]["pass"])
        self.assertEqual(payload["summary"]["main_ownership_status"], "GO_MAIN_OWNERSHIP_FIELD")
        self.assertFalse(payload["summary"]["uses_gt_for_prediction"])


if __name__ == "__main__":
    unittest.main()
