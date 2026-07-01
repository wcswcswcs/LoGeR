from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np


_TOOL_PATH = Path(__file__).resolve().parents[1] / "tools" / "run_v62_active_query_material_outcome.py"
_SPEC = importlib.util.spec_from_file_location("run_v62_active_query_material_outcome", _TOOL_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
_material_rows_for_run = _MODULE._material_rows_for_run


class V62ActiveQueryMaterialOutcomeTest(unittest.TestCase):
    def test_conservative_material_outcomes_from_carrier_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "v62_d4rt_smoke_summary.json").write_text(
                json.dumps({"scene": "scene_test", "query_budget": 3}),
                encoding="utf-8",
            )
            with (root / "v62_d4rt_smoke_query_rows.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "d4rt_query_index",
                        "query_candidate_id",
                        "candidate_source",
                        "state",
                        "has_K_mat",
                        "uses_gt_for_prediction",
                        "uses_gt_for_diagnostic_labels",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "d4rt_query_index": 0,
                        "query_candidate_id": "q_confirm",
                        "candidate_source": "bridge_low_support",
                        "state": "confirmed",
                        "has_K_mat": "True",
                        "uses_gt_for_prediction": "False",
                        "uses_gt_for_diagnostic_labels": "False",
                    }
                )
                writer.writerow(
                    {
                        "d4rt_query_index": 1,
                        "query_candidate_id": "q_quarantine",
                        "candidate_source": "shared_shortcut_boundary",
                        "state": "shared",
                        "has_K_mat": "False",
                        "uses_gt_for_prediction": "False",
                        "uses_gt_for_diagnostic_labels": "False",
                    }
                )
                writer.writerow(
                    {
                        "d4rt_query_index": 2,
                        "query_candidate_id": "q_unresolved",
                        "candidate_source": "bridge_low_support",
                        "state": "confirmed",
                        "has_K_mat": "True",
                        "uses_gt_for_prediction": "False",
                        "uses_gt_for_diagnostic_labels": "False",
                    }
                )
            valid = np.ones((4, 3), dtype=bool)
            visibility = np.asarray(
                [
                    [0.9, 0.9, 0.9],
                    [0.9, 0.9, 0.1],
                    [0.9, 0.9, 0.1],
                    [0.9, 0.9, 0.1],
                ],
                dtype=np.float32,
            )
            confidence = visibility.copy()
            uv = np.full((4, 3, 2), 0.5, dtype=np.float32)
            np.savez_compressed(
                root / "carrier_batch_smoke.npz",
                valid=valid,
                visibility_prob=visibility,
                confidence_prob=confidence,
                uv_pred=uv,
            )
            args = types.SimpleNamespace(
                min_visibility=0.5,
                min_confidence=0.5,
                min_accepted_frames=2,
                min_accepted_ratio=0.0,
                confirm_min_accepted_ratio=0.25,
                min_in_bounds_ratio=0.80,
                quarantine_outside_residual_rate=0.20,
            )
            summary, rows = _material_rows_for_run(root, args)

        labels = {row["query_candidate_id"]: row["material_outcome_label"] for row in rows}
        self.assertEqual(labels["q_confirm"], "confirm")
        self.assertEqual(labels["q_quarantine"], "quarantine")
        self.assertEqual(labels["q_unresolved"], "unresolved")
        self.assertEqual(summary["query_to_confirm_rate"], 1 / 3)
        self.assertEqual(summary["query_to_quarantine_rate"], 1 / 3)
        self.assertTrue(all(row["uses_gt_for_prediction"] is False for row in rows))
        self.assertTrue(all(row["uses_gt_for_diagnostic_labels"] is False for row in rows))


if __name__ == "__main__":
    unittest.main()
