from __future__ import annotations

import csv
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from stream4d_native.v59_graph_builder import V59GraphBuilderConfig, build_v59_graph, write_v59_graph


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


class V59GraphBuilderTest(unittest.TestCase):
    def _fixture(self, root: Path) -> V59GraphBuilderConfig:
        semantic = root / "semantic"
        explanation = root / "explanation"
        reproj = root / "reprojection"
        _write_csv(
            root / "history_rows.csv",
            [
                {"history_id": "sceneA|obj1", "scene": "sceneA"},
                {"history_id": "sceneA|obj2", "scene": "sceneA"},
            ],
        )
        _write_csv(
            semantic / "history_semantic_rows.csv",
            [
                {"history_id": "sceneA|obj1", "scene": "sceneA", "mode_index": 0, "mode_weight": 1.0},
                {"history_id": "sceneA|obj2", "scene": "sceneA", "mode_index": 0, "mode_weight": 1.0},
            ],
        )
        _write_csv(
            root / "native_rows.csv",
            [
                {"history_id": "sceneA|obj1", "scene": "sceneA", "component_id": "c1", "state": "confirmed"},
                {"history_id": "sceneA|obj2", "scene": "sceneA", "component_id": "c2", "state": "confirmed"},
            ],
        )
        _write_csv(
            explanation / "explanation_rows.csv",
            [
                {
                    "variant": "E6_counterfactual_semantic_material_underseg",
                    "row_role": "candidate",
                    "observation_id": "sceneA:0:1",
                    "scene": "sceneA",
                    "frame_id": 0,
                    "mask_id": 1,
                    "explanation_type": "underseg_mixture",
                    "history_id": "sceneA|obj1",
                    "candidate_history_ids_json": json.dumps(["sceneA|obj1", "sceneA|obj2"]),
                    "semantic_score": 0.9,
                    "posterior": 0.7,
                    "is_selected": True,
                    "decision_state": "defer_to_active_query",
                },
                {
                    "variant": "E6_counterfactual_semantic_material_underseg",
                    "row_role": "candidate",
                    "observation_id": "sceneA:0:1",
                    "scene": "sceneA",
                    "frame_id": 0,
                    "mask_id": 1,
                    "explanation_type": "assign_to_existing",
                    "history_id": "sceneA|obj2",
                    "candidate_history_ids_json": json.dumps(["sceneA|obj2"]),
                    "semantic_score": 0.8,
                    "posterior": 0.2,
                    "is_selected": False,
                    "decision_state": "not_selected",
                },
            ],
        )
        _write_csv(
            root / "support_rows.csv",
            [
                {"mask_observation_id": "sceneA:0:1", "scene": "sceneA", "frame_id": 0, "mask_id": 1, "component_id": "c1", "selection_score": 1.0, "support_count": 10},
                {"mask_observation_id": "sceneA:0:1", "scene": "sceneA", "frame_id": 0, "mask_id": 1, "component_id": "c2", "selection_score": 0.9, "support_count": 9},
            ],
        )
        _write_csv(
            reproj / "candidate_rows.csv",
            [
                {"candidate_id": "cand1", "source_mask_observation_id": "sceneA:0:1", "scene": "sceneA", "candidate_source": "fixture"},
            ],
        )
        _write_csv(
            reproj / "reprojection_ledger_rows.csv",
            [
                {
                    "candidate_id": "cand1",
                    "scene": "sceneA",
                    "target_frame_id": 1,
                    "best_mask_observation_id": "sceneA:1:2",
                    "inside_best_mask_ratio": 0.8,
                    "reprojection_success": True,
                    "same_frame_exclusion_violation": False,
                    "outside_all_related_masks_ratio": 0.0,
                },
                {
                    "candidate_id": "cand1",
                    "scene": "sceneA",
                    "target_frame_id": 2,
                    "best_mask_observation_id": "sceneA:2:3",
                    "inside_best_mask_ratio": 0.1,
                    "reprojection_success": False,
                    "same_frame_exclusion_violation": True,
                    "outside_all_related_masks_ratio": 0.9,
                },
            ],
        )
        _write_csv(root / "history_update_rows.csv", [])
        return V59GraphBuilderConfig(
            semantic_root=semantic,
            explanation_root=explanation,
            support_rows_path=root / "support_rows.csv",
            history_rows_path=root / "history_rows.csv",
            history_update_rows_path=root / "history_update_rows.csv",
            reprojection_root=reproj,
            native_carrier_state_rows_path=root / "native_rows.csv",
            output_root=root / "out",
            visualization_root=root / "viz",
        )

    def test_graph_builder_records_typed_invariants(self) -> None:
        root = Path(tempfile.mkdtemp(prefix="v59_graph_test_"))
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        cfg = self._fixture(root)
        result = build_v59_graph(cfg)
        summary = result["summary"]
        self.assertFalse(summary["uses_gt_for_prediction"])
        self.assertTrue(summary["gate"]["pass"])
        self.assertGreater(summary["underseg_bridge_edge_count"], 0)
        self.assertEqual(summary["no_D4RT_birth_edge_count"], 0)
        edge_types = summary["edge_count_by_type"]
        for edge_type in ["semantic_compatibility", "material_continuity", "mask_support", "reprojection", "exclusion"]:
            self.assertIn(edge_type, edge_types)
        outputs = write_v59_graph(result, root / "out")
        for rel_path in outputs.values():
            self.assertTrue((root / "out" / Path(rel_path).name).exists() or Path(rel_path).exists())


if __name__ == "__main__":
    unittest.main()
