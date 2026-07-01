from __future__ import annotations

import csv
import shutil
import tempfile
import unittest
from pathlib import Path

from stream4d_native.v47_common import write_json
from stream4d_native.v61_refinement import V61RefinementConfig, build_v61_refinement, write_v61_refinement


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


class V61RefinementTest(unittest.TestCase):
    def test_composite_shared_shortcut_becomes_quarantine(self) -> None:
        root = Path(tempfile.mkdtemp(prefix="v61_refinement_test_"))
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        embedding = root / "embedding.json"
        states = root / "states.csv"
        candidates = root / "candidates.csv"
        nodes = root / "nodes.csv"
        write_json(
            embedding,
            {
                "core_purity": 1.0,
                "core_completeness": 1.0,
                "expanded_completeness": 1.0,
                "real_minus_shuffled_ARI": 1.0,
                "real_minus_no_temporal_ARI": 1.0,
            },
        )
        common = {
            "scene": "s",
            "candidate_rank": 1,
            "candidate_total_cost": 0.1,
            "has_K_mat": "False",
            "has_K_mask": "True",
            "has_K_sem": "False",
            "has_K_underseg": "False",
            "can_enter_confirmed_core": "False",
            "can_enter_shared": "True",
            "can_enter_quarantine": "True",
            "uses_gt_for_prediction": "False",
            "uses_gt_for_diagnostic_labels": "True",
        }
        _write_csv(
            states,
            [
                {
                    **common,
                    "variant": "M7",
                    "material_node_id": "a:c1",
                    "component_id": "c1",
                    "state": "confirmed",
                    "predicted_history_id": "s|h1",
                    "candidate_history_id": "s|h1",
                    "state_reason": "K_mat_material_continuity",
                    "candidate_evidence_types": "K_mat",
                    "support_observation_ids_json": "[\"m:o1\"]",
                    "diagnostic_expected_history_id": "s|h1",
                    "diagnostic_exact_match": "True",
                    "diagnostic_contains_expected": "True",
                },
                {
                    **common,
                    "variant": "M7",
                    "material_node_id": "a:c2",
                    "component_id": "c2",
                    "state": "shared",
                    "predicted_history_id": "s|h2||s|h3",
                    "candidate_history_id": "s|h2||s|h3",
                    "state_reason": "underseg_or_composite_shared_support",
                    "candidate_evidence_types": "K_mask",
                    "support_observation_ids_json": "[\"m:o2\"]",
                    "diagnostic_expected_history_id": "",
                    "diagnostic_exact_match": "False",
                    "diagnostic_contains_expected": "False",
                },
            ],
        )
        _write_csv(candidates, [{"material_node_id": "a:c1"}, {"material_node_id": "a:c2"}])
        _write_csv(
            nodes,
            [
                {"node_id": "m:o1", "node_type": "mask_observation", "history_id": "s|h1"},
                {"node_id": "m:o2", "node_type": "mask_observation", "history_id": "s|h2||s|h3"},
            ],
        )
        result = build_v61_refinement(
            V61RefinementConfig(
                embedding_summary_path=embedding,
                material_state_rows_path=states,
                material_candidate_rows_path=candidates,
                v60_node_rows_path=nodes,
            )
        )
        self.assertEqual(result["summary"]["selected_variant"], "R2_shortcut_quarantine")
        self.assertEqual(result["summary"]["quarantined_node_count"], 1)
        self.assertEqual(result["summary"]["quarantine_precision_diagnostic"], 1.0)
        after = {row["material_node_id"]: row for row in result["material_state_after_refinement"]}
        self.assertEqual(after["a:c2"]["state"], "quarantine")
        outputs = write_v61_refinement(result, root / "out")
        self.assertTrue((root / "out" / "refinement_summary.json").exists())
        self.assertIn("material_state_after_refinement", outputs)


if __name__ == "__main__":
    unittest.main()
