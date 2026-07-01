from __future__ import annotations

import csv
import shutil
import tempfile
import unittest
from pathlib import Path

from stream4d_native.v47_common import write_json
from stream4d_native.v61_failure_lock import V61FailureLockConfig, build_v61_failure_lock, write_v61_failure_lock


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


class V61FailureLockTest(unittest.TestCase):
    def test_material_state_coverage_detects_unit_mismatch(self) -> None:
        root = Path(tempfile.mkdtemp(prefix="v61_failure_lock_test_"))
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        final = root / "final.json"
        graph = root / "graph.json"
        path = root / "path.json"
        emb = root / "embedding.json"
        refinement = root / "refinement.json"
        query = root / "query.json"
        nodes = root / "nodes.csv"
        edges = root / "edges.csv"
        states = root / "states.csv"
        write_json(final, {"final_label": "NO_GO_EMBEDDING", "partial_label": "PARTIAL_GRAPH_PATH_SIGNAL"})
        write_json(graph, {"gate": {"pass": True}})
        write_json(path, {"gate": {"pass": True}, "accepted_path_count": 1, "path_precision_diagnostic": 1.0, "shortcut_quarantine_precision": 1.0})
        write_json(emb, {"gate": {"pass": False}, "confirmed_node_count": 1, "tentative_node_count": 0, "quarantine_node_count": 0, "unknown_node_count": 0})
        write_json(refinement, {"gate": {"pass": False}})
        write_json(query, {"gate": {"pass": False}})
        _write_csv(nodes, [{"node_id": "a:c1", "node_type": "material"}, {"node_id": "m:s:1:2", "node_type": "mask_observation"}])
        _write_csv(edges, [{"edge_id": "e1", "edge_type": "mask_support"}])
        _write_csv(states, [{"observation_id": "s:1:2", "state": "confirmed"}])
        cfg = V61FailureLockConfig(
            v60_final_path=final,
            v60_graph_summary_path=graph,
            v60_graph_node_rows_path=nodes,
            v60_graph_edge_rows_path=edges,
            v60_path_summary_path=path,
            v60_embedding_summary_path=emb,
            v60_node_state_rows_path=states,
            v60_refinement_summary_path=refinement,
            v60_query_summary_path=query,
        )
        result = build_v61_failure_lock(cfg)
        self.assertTrue(result["summary"]["gate"]["pass"])
        self.assertEqual(result["summary"]["material_state_coverage_rate"], 0.0)
        outputs = write_v61_failure_lock(result, root / "out")
        self.assertTrue((root / "out" / "failure_lock.json").exists())
        self.assertIn("unit_mismatch_rows", outputs)


if __name__ == "__main__":
    unittest.main()
