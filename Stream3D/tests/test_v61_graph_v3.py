from __future__ import annotations

import csv
import shutil
import tempfile
import unittest
from pathlib import Path

from stream4d_native.v47_common import write_json
from stream4d_native.v61_graph_v3 import V61GraphV3Config, build_v61_graph_v3, write_v61_graph_v3


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


class V61GraphV3Test(unittest.TestCase):
    def test_material_candidate_from_material_and_mask_edges(self) -> None:
        root = Path(tempfile.mkdtemp(prefix="v61_graph_v3_test_"))
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        phase0 = root / "phase0.json"
        graph = root / "graph.json"
        nodes = root / "nodes.csv"
        edges = root / "edges.csv"
        write_json(phase0, {"gate": {"pass": True}})
        write_json(graph, {"gate": {"pass": True}})
        _write_csv(
            nodes,
            [
                {"node_id": "a:c1", "node_type": "material", "scene": "s", "component_id": "c1", "history_id": "s|h1"},
                {"node_id": "m:s:1:1", "node_type": "mask_observation", "scene": "s", "frame_id": 1, "mask_id": 1, "history_id": "s|h1"},
                {"node_id": "s:s|h1:mode0", "node_type": "semantic_mode", "scene": "s", "history_id": "s|h1"},
                {"node_id": "h:s|h1", "node_type": "history_core", "scene": "s", "history_id": "s|h1"},
            ],
        )
        _write_csv(
            edges,
            [
                {"edge_id": "e1", "edge_type": "material_continuity", "src_node_id": "a:c1", "dst_node_id": "h:s|h1", "confidence": 1.0, "edge_cost": 0.0},
                {"edge_id": "e2", "edge_type": "mask_support", "src_node_id": "m:s:1:1", "dst_node_id": "a:c1", "confidence": 1.0, "edge_cost": 0.0},
                {"edge_id": "e3", "edge_type": "mask_support", "src_node_id": "m:s:1:1", "dst_node_id": "h:s|h1", "confidence": 1.0, "edge_cost": 0.0},
                {"edge_id": "e4", "edge_type": "semantic_compatibility", "src_node_id": "m:s:1:1", "dst_node_id": "s:s|h1:mode0", "confidence": 1.0, "edge_cost": 0.0},
                {"edge_id": "e5", "edge_type": "semantic_compatibility", "src_node_id": "s:s|h1:mode0", "dst_node_id": "h:s|h1", "confidence": 1.0, "edge_cost": 0.0},
            ],
        )
        cfg = V61GraphV3Config(phase0_failure_lock_path=phase0, v60_graph_summary_path=graph, v60_node_rows_path=nodes, v60_edge_rows_path=edges)
        result = build_v61_graph_v3(cfg)
        self.assertTrue(result["summary"]["gate"]["pass"])
        self.assertEqual(result["summary"]["candidate_recall_at_5"], 1.0)
        self.assertEqual(result["material_candidate_rows"][0]["candidate_history_id"], "s|h1")
        outputs = write_v61_graph_v3(result, root / "out")
        self.assertTrue((root / "out" / "graph_v3_summary.json").exists())
        self.assertIn("material_candidate_rows", outputs)


if __name__ == "__main__":
    unittest.main()
