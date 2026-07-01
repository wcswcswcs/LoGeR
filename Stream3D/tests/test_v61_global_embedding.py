from __future__ import annotations

import csv
import shutil
import tempfile
import unittest
from pathlib import Path

from stream4d_native.v47_common import write_json
from stream4d_native.v61_global_embedding import V61GlobalEmbeddingConfig, build_v61_global_embedding, write_v61_global_embedding


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


class V61GlobalEmbeddingTest(unittest.TestCase):
    def test_k_mat_candidate_becomes_confirmed_material_state(self) -> None:
        root = Path(tempfile.mkdtemp(prefix="v61_embedding_test_"))
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        graph = root / "graph.json"
        candidates = root / "candidates.csv"
        v56_core = root / "v56_core.json"
        v56_tentative = root / "v56_tentative.json"
        write_json(graph, {"gate": {"pass": True}})
        write_json(v56_core, {"core_completeness": 0.0, "real_minus_shuffled_ARI": 0.0, "real_minus_no_temporal_ARI": 0.0})
        write_json(v56_tentative, {"expanded_completeness": 0.0})
        rows = []
        for component_id, history_id in (("c1", "s|h1"), ("c2", "s|h1"), ("c3", "s|h2"), ("c4", "s|h2")):
            rows.append(
                {
                    "material_node_id": f"a:{component_id}",
                    "scene": "s",
                    "component_id": component_id,
                    "candidate_history_id": history_id,
                    "candidate_rank": 1,
                    "candidate_total_cost": 0.1,
                    "candidate_evidence_types": "K_mat",
                    "has_K_mat": "True",
                    "has_K_mask": "False",
                    "has_K_sem": "False",
                    "has_K_underseg": "False",
                    "can_enter_confirmed_core": "True",
                    "can_enter_shared": "False",
                    "can_enter_quarantine": "False",
                    "hard_constraint_violation": "False",
                    "support_observation_ids_json": f"[\"m:o{component_id[-1]}\"]",
                    "diagnostic_expected_history_id": history_id,
                }
            )
        _write_csv(candidates, rows)
        cfg = V61GlobalEmbeddingConfig(
            graph_v3_summary_path=graph,
            material_candidate_rows_path=candidates,
            v56_core_summary_path=v56_core,
            v56_tentative_summary_path=v56_tentative,
        )
        result = build_v61_global_embedding(cfg)
        self.assertTrue(result["summary"]["gate"]["pass"])
        self.assertEqual(result["material_state_rows"][0]["state"], "confirmed")
        self.assertEqual(result["summary"]["core_purity"], 1.0)
        outputs = write_v61_global_embedding(result, root / "out")
        self.assertTrue((root / "out" / "embedding_summary.json").exists())
        self.assertIn("energy_rows", outputs)


if __name__ == "__main__":
    unittest.main()
