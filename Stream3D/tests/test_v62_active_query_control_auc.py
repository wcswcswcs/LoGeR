from __future__ import annotations

import unittest
import importlib.util
from pathlib import Path


_TOOL_PATH = Path(__file__).resolve().parents[1] / "tools" / "run_v62_active_query_control_auc.py"
_SPEC = importlib.util.spec_from_file_location("run_v62_active_query_control_auc", _TOOL_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
_rank_auc = _MODULE._rank_auc
_summarize = _MODULE._summarize


class V62ActiveQueryControlAucTest(unittest.TestCase):
    def test_rank_auc_handles_ties_and_separation(self) -> None:
        self.assertEqual(_rank_auc([True, False], [1.0, 0.0]), 1.0)
        self.assertEqual(_rank_auc([True, False], [0.5, 0.5]), 0.5)
        self.assertIsNone(_rank_auc([True, True], [1.0, 0.0]))

    def test_summary_never_claims_independent_gate_pass(self) -> None:
        rows = [
            {
                "control_success_label": True,
                "real_evidence_score": 0.9,
                "shuffled_evidence_score": 0.1,
                "no_temporal_evidence_score": 0.1,
                "candidate_prior_score": 0.4,
            },
            {
                "control_success_label": False,
                "real_evidence_score": 0.1,
                "shuffled_evidence_score": 0.9,
                "no_temporal_evidence_score": 0.9,
                "candidate_prior_score": 0.6,
            },
        ]
        summary = _summarize(rows, best_fixed=0.0)
        self.assertGreaterEqual(summary["real_minus_shuffled_query_AUC"], 0.15)
        self.assertGreaterEqual(summary["real_minus_no_temporal_query_AUC"], 0.10)
        self.assertFalse(summary["gate"]["independent_gt_or_external_outcome_labels"])
        self.assertFalse(summary["gate"]["pass"])
        self.assertFalse(summary["uses_gt_for_prediction"])


if __name__ == "__main__":
    unittest.main()
