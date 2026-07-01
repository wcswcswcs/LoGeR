from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from stream4d_native.v63_query_policy import V63QueryPolicyConfig, build_v63_query_policy


class V63QueryPolicyTest(unittest.TestCase):
    def test_policy_selects_balanced_real_actions_and_equal_controls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "candidates.csv"
            rows = []
            for candidate_type in ["heldout_recovery", "shortcut_quarantine", "decoy_rejection", "unknown_defer"]:
                for idx in range(2):
                    rows.append(_candidate(f"{candidate_type}_{idx}", "method_candidate", candidate_type, "", 0.8 - idx * 0.1))
            for control_id in ["C0_v62_original", "C1_random_matched", "C2_mask_boundary", "C3_semantic_only", "C4_K_mask_only_ablation"]:
                for idx in range(4):
                    rows.append(_candidate(f"{control_id}_{idx}", "baseline_control", "control", control_id, 0.7 - idx * 0.01))
            _write_csv(path, rows)
            cfg = V63QueryPolicyConfig(query_candidate_rows=path, query_budget=4)

            result = build_v63_query_policy(cfg)

        summary = result["summary"]
        selected = result["selected_query_rows"]
        real_actions = [row["planned_action"] for row in selected if row["policy_id"] == "R0_real_policy"]
        self.assertTrue(summary["gate"]["pass"])
        self.assertEqual(summary["real_policy_query_count"], 4)
        self.assertEqual(summary["control_query_counts"]["C4_K_mask_only_ablation"], 4)
        self.assertEqual(sorted(real_actions), ["confirm", "defer", "quarantine", "reject_decoy"])
        self.assertTrue(all(row["uses_gt_for_prediction"] is False for row in selected))


def _candidate(candidate_id: str, role: str, candidate_type: str, control_id: str, score: float) -> dict[str, object]:
    return {
        "v63_candidate_id": candidate_id,
        "row_role": role,
        "candidate_type": candidate_type,
        "control_id": control_id,
        "material_node_id": candidate_id,
        "scene": "scene_test",
        "component_id": candidate_id,
        "support_observation_count": 2,
        "selection_score": score,
        "query_history_id": f"history_{candidate_id}",
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": False,
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
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


if __name__ == "__main__":
    unittest.main()
