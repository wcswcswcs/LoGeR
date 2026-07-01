from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from stream4d_native.v59_fact_lock import build_v59_fact_lock, write_v59_fact_lock


class V59FactLockTest(unittest.TestCase):
    def test_phase0_gate_locks_v58_failure_chain(self) -> None:
        payload = build_v59_fact_lock()
        fact = payload["fact_lock"]
        self.assertFalse(fact["uses_gt_for_prediction"])
        self.assertTrue(fact["uses_gt_for_diagnostic_labels"])
        self.assertGreaterEqual(fact["v58_phase1_dino_recall@3"], 0.85)
        self.assertGreater(fact["v58_phase2_deferred_count"], fact["v58_phase2_actionable_count"])
        self.assertFalse(fact["v58_phase3_strict_gate_pass"])
        self.assertTrue(fact["expanded_candidate_quality_drop_observed"])
        self.assertTrue(fact["gate"]["pass"])

    def test_phase0_outputs_are_written(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="v59_fact_lock_test_"))
        self.addCleanup(lambda: shutil.rmtree(tmp, ignore_errors=True))
        payload = build_v59_fact_lock()
        outputs = write_v59_fact_lock(tmp / "out", payload, tmp / "viz")
        self.assertTrue((tmp / "out" / "fact_lock.json").exists())
        self.assertTrue((tmp / "out" / "v59_phase0_metric_rows.csv").exists())
        self.assertTrue((tmp / "out" / "v59_phase0_failure_chain_rows.csv").exists())
        self.assertEqual(outputs["visualization_status"], "created")


if __name__ == "__main__":
    unittest.main()
