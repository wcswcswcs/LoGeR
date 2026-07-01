from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from stream4d_native.v58_fact_lock import build_v58_fact_lock, write_v58_fact_lock


class V58FactLockTest(unittest.TestCase):
    def test_v58_fact_lock_gate_and_no_gt_prediction(self) -> None:
        payload = build_v58_fact_lock()
        fact_lock = payload["fact_lock"]
        self.assertFalse(fact_lock["uses_gt_for_prediction"])
        self.assertFalse(fact_lock["uses_gt_for_diagnostic_labels"])
        self.assertEqual(fact_lock["v56_final_label"], "NO_GO_D4RT_CONTROL")
        self.assertEqual(fact_lock["v56_partial_label"], "PARTIAL_TENTATIVE_SUPPORT_SIGNAL")
        self.assertTrue(fact_lock["gate"]["pass"])
        self.assertGreater(len(payload["v56_baseline_rows"]), 0)

    def test_v58_fact_lock_writes_outputs(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="v58_fact_lock_test_"))
        self.addCleanup(lambda: shutil.rmtree(tmp, ignore_errors=True))
        payload = build_v58_fact_lock()
        outputs = write_v58_fact_lock(tmp, payload)
        self.assertTrue((tmp / "fact_lock.json").exists())
        self.assertTrue((tmp / "v56_baseline_rows.csv").exists())
        self.assertIn("dashboard_status", outputs)


if __name__ == "__main__":
    unittest.main()

