from __future__ import annotations

import csv
import shutil
import tempfile
import unittest
from pathlib import Path

from stream4d_native.v61_manifold_query import V61ManifoldQueryConfig, build_v61_manifold_query


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


class V61ManifoldQueryTest(unittest.TestCase):
    def test_q7_selects_underseg_quarantine_without_gt_prediction(self) -> None:
        root = Path(tempfile.mkdtemp(prefix="v61_query_test_"))
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        query_root = root / "v58"
        _write_csv(
            query_root / "query_metric_rows.csv",
            [
                {
                    "baseline_id": "Q0",
                    "baseline_name": "random",
                    "query_budget": 2,
                    "query_count": 2,
                    "valid_material_evidence_rate": 0.1,
                    "query_to_confirm_rate": 0.0,
                    "query_to_quarantine_rate": 0.0,
                    "entropy_reduction": 0.1,
                    "real_minus_shuffled_query_AUC": 0.0,
                    "real_minus_no_temporal_query_AUC": 0.0,
                }
            ],
        )
        _write_csv(
            query_root / "query_rows.csv",
            [
                {
                    "baseline_id": "Q0",
                    "baseline_name": "random",
                    "query_id": "src_low",
                    "observation_id": "scene:0:1",
                    "candidate_id": "cand_low",
                    "estimated_information_gain": 0.1,
                    "entropy_before": 1.0,
                    "entropy_after": 1.0,
                    "actual_entropy_reduction": 0.0,
                    "valid_material_evidence": "False",
                    "query_to_confirm": "False",
                    "query_to_quarantine": "False",
                    "real_evidence_score": 0.1,
                    "shuffled_evidence_score": 0.1,
                    "no_temporal_evidence_score": 0.1,
                    "diagnostic_query_success_same_gt": "False",
                    "uses_gt_for_prediction": "False",
                    "uses_gt_for_diagnostic_labels": "True",
                },
                {
                    "baseline_id": "Q1",
                    "baseline_name": "boundary",
                    "query_id": "src_hi",
                    "observation_id": "scene:0:2",
                    "candidate_id": "cand_hi",
                    "estimated_information_gain": 0.2,
                    "entropy_before": 2.0,
                    "entropy_after": 0.5,
                    "actual_entropy_reduction": 1.5,
                    "valid_material_evidence": "True",
                    "query_to_confirm": "False",
                    "query_to_quarantine": "True",
                    "real_evidence_score": 0.9,
                    "shuffled_evidence_score": 0.2,
                    "no_temporal_evidence_score": 0.2,
                    "diagnostic_query_success_same_gt": "True",
                    "uses_gt_for_prediction": "False",
                    "uses_gt_for_diagnostic_labels": "True",
                },
            ],
        )
        _write_csv(query_root / "material_evidence_rows.csv", [{"query_id": "src_hi", "candidate_id": "cand_hi"}])
        states = root / "states.csv"
        _write_csv(
            states,
            [
                {
                    "material_node_id": "a:1",
                    "state": "confirmed",
                    "predicted_history_id": "h1",
                    "support_observation_ids_json": "[\"m:scene:0:1\"]",
                },
                {
                    "material_node_id": "a:2",
                    "state": "quarantine",
                    "predicted_history_id": "h2||h3",
                    "support_observation_ids_json": "[\"m:scene:0:2\"]",
                    "has_K_underseg": "True",
                },
            ],
        )
        observations = root / "obs.csv"
        _write_csv(
            observations,
            [
                {
                    "observation_node_id": "m:scene:0:1",
                    "explanation_type": "assign",
                    "candidate_history_ids_json": "[\"h1\"]",
                },
                {
                    "observation_node_id": "m:scene:0:2",
                    "explanation_type": "underseg",
                    "candidate_history_ids_json": "[\"h2\", \"h3\"]",
                },
            ],
        )
        result = build_v61_manifold_query(
            V61ManifoldQueryConfig(
                refined_state_rows_path=states,
                observation_explanation_rows_path=observations,
                v58_query_root=query_root,
                query_budget=1,
            )
        )
        summary = result["summary"]
        self.assertFalse(summary["uses_gt_for_prediction"])
        self.assertEqual(summary["query_count"], 1)
        self.assertEqual(result["query_rows"][0]["source_query_id"], "src_hi")
        self.assertEqual(result["query_rows"][0]["query_type"], "Q_shortcut")
        self.assertEqual(summary["valid_material_evidence_rate"], 1.0)
        self.assertEqual(summary["query_to_quarantine_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
