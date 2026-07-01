from __future__ import annotations

import csv
import shutil
import tempfile
import unittest
from pathlib import Path

from stream4d_native.v47_common import write_json
from stream4d_native.v60_graph_v2 import V60GraphV2Config, build_v60_graph_v2, write_v60_graph_v2


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


class V60GraphV2Test(unittest.TestCase):
    def test_graph_v2_adds_costs_and_preserves_invariants(self) -> None:
        root = Path(tempfile.mkdtemp(prefix="v60_graph_v2_test_"))
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        graph = root / "v59"
        write_json(graph / "graph_summary.json", {"gate": {"pass": True}, "history_manifold_count": 1})
        _write_csv(graph / "node_rows.csv", [{"node_id": "h:h1", "node_type": "history_core", "history_id": "h1"}])
        _write_csv(
            graph / "edge_rows.csv",
            [
                {"edge_id": "e1", "edge_type": "semantic_compatibility", "confidence": 0.9, "can_merge_histories": False},
                {"edge_id": "e2", "edge_type": "material_continuity", "confidence": 1.0, "can_create_birth": False, "can_confirm_identity": True},
                {"edge_id": "e3", "edge_type": "mask_support", "confidence": 0.8, "can_create_birth": False},
                {"edge_id": "e4", "edge_type": "reprojection", "confidence": 0.7},
                {"edge_id": "e5", "edge_type": "underseg_bridge", "confidence": 0.5, "can_merge_histories": False},
            ],
        )
        result = build_v60_graph_v2(V60GraphV2Config(v59_graph_root=graph, output_root=root / "out"))
        summary = result["summary"]
        self.assertTrue(summary["gate"]["pass"])
        self.assertEqual(summary["hard_constraint_violation_count"], 0)
        self.assertGreater(len(result["edge_cost_rows"]), 0)
        outputs = write_v60_graph_v2(result, root / "out")
        self.assertTrue((root / "out" / "graph_summary.json").exists())
        self.assertIn("edge_cost_rows", outputs)


if __name__ == "__main__":
    unittest.main()
