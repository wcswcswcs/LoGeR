from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from stream4d_native.v58_active_material_query import (
    V58ActiveMaterialQueryConfig,
    build_v58_active_material_query,
    write_v58_active_material_query,
)


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


class V58ActiveMaterialQueryTest(unittest.TestCase):
    def _fixture(self, root: Path) -> V58ActiveMaterialQueryConfig:
        phase2 = root / "phase2"
        phase2.mkdir(parents=True, exist_ok=True)
        (phase2 / "explanation_summary.json").write_text(
            json.dumps(
                {
                    "gate": {"pass": False},
                    "deferred_count": 3,
                    "actionable_count": 1,
                }
            ),
            encoding="utf-8",
        )
        _write_csv(
            phase2 / "explanation_rows.csv",
            [
                {
                    "variant": "E6_counterfactual_semantic_material_underseg",
                    "is_selected": "True",
                    "decision_state": "defer_to_active_query",
                    "observation_id": "sceneA:1:1",
                    "scene": "sceneA",
                    "frame_id": 1,
                    "mask_id": 1,
                    "explanation_type": "assign_to_existing",
                    "history_id": "sceneA|h1",
                    "candidate_history_ids_json": json.dumps(["sceneA|h1", "sceneA|h2"]),
                    "posterior_entropy": 1.6,
                    "posterior_top1_margin": 0.05,
                    "material_competition": 0.4,
                    "component_entropy": 0.3,
                    "semantic_advantage": 0.1,
                },
                {
                    "variant": "E6_counterfactual_semantic_material_underseg",
                    "is_selected": "True",
                    "decision_state": "defer_to_active_query",
                    "observation_id": "sceneA:1:2",
                    "scene": "sceneA",
                    "frame_id": 1,
                    "mask_id": 2,
                    "explanation_type": "underseg_mixture",
                    "history_id": "sceneA|h1||sceneA|h2",
                    "candidate_history_ids_json": json.dumps(["sceneA|h1", "sceneA|h2"]),
                    "posterior_entropy": 2.0,
                    "posterior_top1_margin": 0.03,
                    "material_competition": 0.9,
                    "component_entropy": 0.8,
                    "semantic_advantage": 0.0,
                },
                {
                    "variant": "E6_counterfactual_semantic_material_underseg",
                    "is_selected": "True",
                    "decision_state": "actionable",
                    "observation_id": "sceneA:1:3",
                    "scene": "sceneA",
                    "frame_id": 1,
                    "mask_id": 3,
                    "explanation_type": "assign_to_existing",
                    "history_id": "sceneA|h3",
                    "candidate_history_ids_json": json.dumps(["sceneA|h3"]),
                    "posterior_entropy": 0.2,
                    "posterior_top1_margin": 0.7,
                    "material_competition": 0.0,
                    "component_entropy": 0.0,
                    "semantic_advantage": 0.9,
                },
            ],
        )
        candidate_rows = root / "candidate_rows.csv"
        ledger_rows = root / "ledger_rows.csv"
        _write_csv(
            candidate_rows,
            [
                {
                    "candidate_id": "cand001",
                    "candidate_source": "R0_single_representative_mask",
                    "scene": "sceneA",
                    "chunk_id": "c0",
                    "source_mask_observation_id": "sceneA:1:1",
                    "source_frame_id": 1,
                    "source_mask_id": 1,
                    "component_count": 1,
                    "uses_gt_for_prediction": "False",
                    "uses_gt_for_diagnostic_labels": "True",
                },
                {
                    "candidate_id": "cand002",
                    "candidate_source": "R5_repeated_support_signature",
                    "scene": "sceneA",
                    "chunk_id": "c0",
                    "source_mask_observation_id": "sceneA:1:2",
                    "source_frame_id": 1,
                    "source_mask_id": 2,
                    "component_count": 2,
                    "uses_gt_for_prediction": "False",
                    "uses_gt_for_diagnostic_labels": "True",
                },
            ],
        )
        _write_csv(
            ledger_rows,
            [
                {
                    "candidate_id": "cand001",
                    "candidate_source": "R0_single_representative_mask",
                    "scene": "sceneA",
                    "chunk_id": "c0",
                    "target_frame_id": 2,
                    "visible_carrier_count": 4,
                    "inside_best_mask_ratio": 0.95,
                    "inside_any_mask_ratio": 1.0,
                    "outside_all_related_masks_ratio": 0.02,
                    "mask_explained_ratio": 0.9,
                    "same_frame_exclusion_violation": "False",
                    "related_mask_count": 1,
                    "reprojection_success": "True",
                    "diagnostic_success_same_gt": "True",
                },
                {
                    "candidate_id": "cand002",
                    "candidate_source": "R5_repeated_support_signature",
                    "scene": "sceneA",
                    "chunk_id": "c0",
                    "target_frame_id": 2,
                    "visible_carrier_count": 3,
                    "inside_best_mask_ratio": 0.55,
                    "inside_any_mask_ratio": 0.7,
                    "outside_all_related_masks_ratio": 0.2,
                    "mask_explained_ratio": 0.6,
                    "same_frame_exclusion_violation": "True",
                    "related_mask_count": 3,
                    "reprojection_success": "True",
                    "diagnostic_success_same_gt": "False",
                },
            ],
        )
        return V58ActiveMaterialQueryConfig(
            phase2_root=phase2,
            reprojection_candidate_rows_path=candidate_rows,
            reprojection_ledger_rows_path=ledger_rows,
            output_root=root / "out",
            visualization_root=root / "viz",
            query_budget=2,
            max_target_frames=1,
        )

    def test_active_query_outputs_equal_budget_and_gt_guard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._fixture(Path(tmp))
            result = build_v58_active_material_query(cfg)
            summary = result["summary"]
            self.assertFalse(summary["uses_gt_for_prediction"])
            self.assertTrue(summary["uses_gt_for_diagnostic_labels"])
            self.assertEqual(summary["eligible_deferred_observation_count"], 2)
            self.assertEqual(summary["query_count"], 2)
            baselines = {row["baseline_id"]: row for row in result["query_metric_rows"]}
            self.assertEqual(set(baselines), {"Q0", "Q1", "Q2", "Q3", "Q4", "Q5", "Q6"})
            self.assertTrue(all(row["query_count"] == 2 for row in baselines.values()))
            self.assertTrue(all(row["uses_gt_for_prediction"] is False for row in result["query_rows"]))
            self.assertTrue(result["material_evidence_rows"])
            paths = write_v58_active_material_query(result, cfg.output_root)
            for path in paths.values():
                self.assertTrue(Path(path).exists() or (Path(tmp) / "out" / Path(path).name).exists())


if __name__ == "__main__":
    unittest.main()
