from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from stream4d_native.v47_common import write_json
from stream4d_native.v59_final_decision import build_v59_final_decision, write_v59_final_decision


class V59FinalDecisionTest(unittest.TestCase):
    def test_phase2_same_category_blocker_labels_no_go(self) -> None:
        root = Path(tempfile.mkdtemp(prefix="v59_final_test_"))
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        phase0 = root / "phase0.json"
        phase1 = root / "phase1.json"
        phase2 = root / "phase2.json"
        write_json(phase0, {"gate": {"pass": True}, "v58_phase1_dino_recall@3": 0.9, "v58_phase2_deferred_count": 10})
        write_json(phase1, {"gate": {"pass": True}, "history_manifold_count": 2, "underseg_bridge_edge_count": 1})
        write_json(
            phase2,
            {
                "gate": {
                    "pass": False,
                    "path_precision_diagnostic_ge_0_80": True,
                    "part_to_core_path_precision_ge_0_80": True,
                    "shortcut_quarantine_precision_ge_0_75": True,
                    "same_category_false_path_rate_metric_available": True,
                    "same_category_false_path_rate_le_semantic_pairwise_baseline_minus_0_05": False,
                }
            },
        )
        decision = build_v59_final_decision(phase0, phase1, phase2)
        self.assertFalse(decision["goal_achieved"])
        self.assertEqual(decision["final_label"], "NO_GO_PHASE2_SAME_CATEGORY_GATE")
        outputs = write_v59_final_decision(decision, root / "out")
        self.assertTrue((root / "out" / "final_decision.json").exists())
        self.assertIn("final_decision", outputs)


if __name__ == "__main__":
    unittest.main()
