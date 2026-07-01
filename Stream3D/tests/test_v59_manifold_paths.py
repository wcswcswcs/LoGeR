from __future__ import annotations

import csv
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from stream4d_native.v47_common import write_json
from stream4d_native.v59_manifold_paths import V59PathConfig, build_v59_manifold_paths, write_v59_manifold_paths


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


class V59ManifoldPathsTest(unittest.TestCase):
    def test_path_diagnostics_do_not_promote_same_category_proxy(self) -> None:
        root = Path(tempfile.mkdtemp(prefix="v59_paths_test_"))
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        graph = root / "graph"
        explanation = root / "explanation"
        write_json(graph / "graph_summary.json", {"gate": {"pass": True}})
        _write_csv(
            graph / "edge_rows.csv",
            [
                {"edge_type": "semantic_compatibility", "src_node_id": "m:sceneA:0:1", "dst_node_id": "s:sceneA|obj1:mode0"},
                {"edge_type": "mask_support", "src_node_id": "m:sceneA:0:1", "dst_node_id": "a:c1"},
                {"edge_type": "material_continuity", "src_node_id": "a:c1", "dst_node_id": "h:sceneA|obj1"},
                {"edge_type": "underseg_bridge", "src_node_id": "m:sceneA:0:2", "dst_node_id": "h:sceneA|obj1"},
                {"edge_type": "underseg_bridge", "src_node_id": "m:sceneA:0:2", "dst_node_id": "h:sceneA|obj2"},
            ],
        )
        _write_csv(
            explanation / "explanation_rows.csv",
            [
                {
                    "variant": "E6_counterfactual_semantic_material_underseg",
                    "observation_id": "sceneA:0:1",
                    "scene": "sceneA",
                    "frame_id": 0,
                    "mask_id": 1,
                    "explanation_type": "assign_to_existing",
                    "history_id": "sceneA|obj1",
                    "candidate_history_ids_json": json.dumps(["sceneA|obj1"]),
                    "diagnostic_expected_type": "assign_to_existing",
                    "diagnostic_expected_history_ids_json": json.dumps(["sceneA|obj1"]),
                    "diagnostic_correct": True,
                    "uses_gt_for_diagnostic_labels": True,
                    "is_selected": True,
                },
                {
                    "variant": "E6_counterfactual_semantic_material_underseg",
                    "observation_id": "sceneA:0:2",
                    "scene": "sceneA",
                    "frame_id": 0,
                    "mask_id": 2,
                    "explanation_type": "underseg_mixture",
                    "history_id": "sceneA|obj1",
                    "candidate_history_ids_json": json.dumps(["sceneA|obj1", "sceneA|obj2"]),
                    "diagnostic_expected_type": "underseg_mixture",
                    "diagnostic_expected_history_ids_json": json.dumps(["sceneA|obj1", "sceneA|obj2"]),
                    "diagnostic_correct": True,
                    "uses_gt_for_diagnostic_labels": True,
                    "is_selected": True,
                },
                {
                    "variant": "E1_semantic_only",
                    "observation_id": "sceneA:0:1",
                    "explanation_type": "assign_to_existing",
                    "history_id": "sceneA|obj1",
                    "diagnostic_expected_history_ids_json": json.dumps(["sceneA|obj1"]),
                    "is_selected": True,
                },
            ],
        )
        cfg = V59PathConfig(graph_root=graph, explanation_root=explanation, output_root=root / "out")
        result = build_v59_manifold_paths(cfg)
        summary = result["summary"]
        self.assertEqual(summary["path_precision_diagnostic"], 1.0)
        self.assertEqual(summary["shortcut_quarantine_precision"], 1.0)
        self.assertFalse(summary["same_category_metric_available"])
        self.assertFalse(summary["gate"]["pass"])
        outputs = write_v59_manifold_paths(result, root / "out")
        for rel_path in outputs.values():
            self.assertTrue((root / "out" / Path(rel_path).name).exists() or Path(rel_path).exists())


if __name__ == "__main__":
    unittest.main()
