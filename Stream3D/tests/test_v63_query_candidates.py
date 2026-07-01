from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from stream4d_native.v63_query_candidates import V63QueryCandidateConfig, build_v63_query_candidates


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


class V63QueryCandidatesTest(unittest.TestCase):
    def test_balanced_protocol_uses_no_gt_selection_and_decoy_differs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            query_path = root / "query.csv"
            novelty_path = root / "novelty.csv"
            query_rows = [
                _query("q1", "m1", "bridge_low_support", "confirmed", "bridge_overlap", 2),
                _query("q2", "m2", "bridge_low_support", "confirmed", "bridge_overlap", 2),
                _query("q3", "m3", "shared_shortcut_boundary", "shared", "shortcut_shared", 1),
                _query("q4", "m4", "shared_shortcut_boundary", "shared", "shortcut_shared", 1),
                _query("q5", "m5", "bridge_low_support", "confirmed", "bridge_overlap", 1),
            ]
            novelty_rows = [
                _novelty("m1", "confirmed", "bridge_overlap", "h1", has_k_mat=True),
                _novelty("m2", "confirmed", "bridge_overlap", "h2", has_k_mat=True),
                _novelty("m3", "shared", "shortcut_shared", "h3", has_k_sem=True),
                _novelty("m4", "shared", "shortcut_shared", "h4", has_k_sem=True),
                _novelty("m5", "confirmed", "bridge_overlap", "h5", has_k_mat=True),
                _novelty("m6", "unknown", "bridge_overlap", "h6"),
                _novelty("m7", "tentative", "anchor_known", "h7", has_k_sem=True),
                _novelty("m8", "tentative", "anchor_known", "h8", has_k_sem=True),
                _novelty("m9", "confirmed", "bridge_overlap", "h9", has_k_mask=True),
            ]
            _write_csv(query_path, query_rows)
            _write_csv(novelty_path, novelty_rows)
            cfg = V63QueryCandidateConfig(
                v62_query_candidate_rows=query_path,
                v62_novelty_material_rows=novelty_path,
                per_type_budget=2,
                per_control_budget=1,
            )

            result = build_v63_query_candidates(cfg)

        summary = result["summary"]
        rows = result["query_candidate_rows"]
        decoys = [row for row in rows if row["candidate_type"] == "decoy_rejection"]
        self.assertTrue(summary["gate"]["pass"])
        self.assertEqual(summary["method_candidate_type_counts"]["heldout_recovery"], 2)
        self.assertEqual(summary["method_candidate_type_counts"]["decoy_rejection"], 2)
        self.assertTrue(all(row["uses_gt_for_prediction"] is False for row in rows))
        self.assertTrue(all(row["uses_gt_for_diagnostic_labels"] is False for row in rows))
        self.assertTrue(all(row["decoy_source_history_id"] != row["decoy_history_id"] for row in decoys))


def _query(
    query_id: str,
    material: str,
    source: str,
    state: str,
    novelty_type: str,
    support: int,
) -> dict[str, object]:
    return {
        "query_candidate_id": query_id,
        "material_node_id": material,
        "scene": "scene_test",
        "component_id": material,
        "candidate_source": source,
        "state": state,
        "novelty_type": novelty_type,
        "support_observation_count": support,
        "has_material_boundary_source": True,
        "support_observation_ids_json": '["m:scene_test:0:1"]',
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": False,
    }


def _novelty(
    material: str,
    state: str,
    novelty_type: str,
    history: str,
    *,
    has_k_mat: bool = False,
    has_k_mask: bool = False,
    has_k_sem: bool = False,
) -> dict[str, object]:
    return {
        "material_node_id": material,
        "scene": "scene_test",
        "component_id": material,
        "state": state,
        "novelty_type": novelty_type,
        "support_observation_count": 2,
        "candidate_history_id": history,
        "predicted_history_id": history,
        "support_observation_ids_json": '["m:scene_test:0:1"]',
        "has_K_mat": has_k_mat,
        "has_K_mask": has_k_mask,
        "has_K_sem": has_k_sem,
        "diagnostic_exact_match": True,
    }


if __name__ == "__main__":
    unittest.main()
