from __future__ import annotations

import csv
import shutil
import tempfile
import unittest
from pathlib import Path

from stream4d_native.v47_common import write_json
from stream4d_native.v60_manifold_query import V60QueryConfig, build_v60_manifold_query, write_v60_manifold_query


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


class V60ManifoldQueryTest(unittest.TestCase):
    def test_q7_selects_tentative_or_quarantine_states(self) -> None:
        root = Path(tempfile.mkdtemp(prefix="v60_query_test_"))
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        emb = root / "embedding"
        query = root / "query"
        write_json(emb / "embedding_summary.json", {"gate": {"pass": False}})
        _write_csv(
            emb / "node_state_rows.csv",
            [
                {"observation_id": "o1", "state": "tentative", "state_reason": "fixture", "posterior_top1_margin": 0.5},
                {"observation_id": "o2", "state": "unknown", "state_reason": "fixture", "posterior_top1_margin": 0.5},
            ],
        )
        write_json(query / "query_summary.json", {"gate": {"pass": False}})
        _write_csv(
            query / "query_metric_rows.csv",
            [{"baseline_id": "Q0", "baseline_name": "random", "entropy_reduction": 0.1, "query_to_confirm_rate": 0.0, "query_to_quarantine_rate": 0.0}],
        )
        _write_csv(
            query / "query_rows.csv",
            [
                {
                    "baseline_id": "Q0",
                    "baseline_name": "random",
                    "query_id": "q0",
                    "observation_id": "o1",
                    "candidate_id": "c1",
                    "estimated_information_gain": 0.5,
                    "valid_material_evidence": "True",
                    "query_to_confirm": "True",
                    "query_to_quarantine": "False",
                    "entropy_before": 1.0,
                    "entropy_after": 0.2,
                    "actual_entropy_reduction": 0.8,
                    "real_evidence_score": 1.0,
                    "shuffled_evidence_score": 0.1,
                    "no_temporal_evidence_score": 0.1,
                    "diagnostic_query_success_same_gt": "True",
                },
                {"baseline_id": "Q0", "baseline_name": "random", "query_id": "q1", "observation_id": "o2", "candidate_id": "c2", "estimated_information_gain": 1.0},
            ],
        )
        _write_csv(query / "material_evidence_rows.csv", [{"query_id": "q0", "valid_material_evidence": "True"}])
        cfg = V60QueryConfig(embedding_root=emb, v58_query_root=query, query_budget=1)
        result = build_v60_manifold_query(cfg)
        self.assertEqual(result["summary"]["query_count"], 1)
        self.assertEqual(result["summary"]["candidate_pool_count"], 1)
        self.assertEqual(result["query_rows"][0]["observation_id"], "o1")
        self.assertEqual(len(result["material_evidence_rows"]), 1)
        outputs = write_v60_manifold_query(result, root / "out")
        self.assertTrue((root / "out" / "query_summary.json").exists())
        self.assertIn("query_rows", outputs)


if __name__ == "__main__":
    unittest.main()
