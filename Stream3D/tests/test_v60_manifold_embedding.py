from __future__ import annotations

import csv
import shutil
import tempfile
import unittest
from pathlib import Path

from stream4d_native.v47_common import write_json
from stream4d_native.v60_manifold_embedding import (
    V60EmbeddingConfig,
    adjusted_rand_index,
    build_v60_manifold_embedding,
    write_v60_manifold_embedding,
)


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


class V60ManifoldEmbeddingTest(unittest.TestCase):
    def test_adjusted_rand_index_perfect(self) -> None:
        self.assertEqual(adjusted_rand_index(["a", "a", "b"], ["x", "x", "y"]), 1.0)

    def test_margin_repair_keeps_low_margin_tentative(self) -> None:
        root = Path(tempfile.mkdtemp(prefix="v60_embedding_test_"))
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        path_root = root / "v60_path"
        write_json(
            path_root / "path_summary.json",
            {"same_category_false_path_rate_calibrated": 0.0},
        )
        _write_csv(
            path_root / "path_rows.csv",
            [
                {
                    "observation_id": "o1",
                    "scene": "s",
                    "frame_id": 1,
                    "mask_id": 1,
                    "target_history_id": "h1",
                    "expected_histories_json": "[\"h1\"]",
                    "accepted_path": "True",
                    "diagnostic_correct": "True",
                    "independent_path_count": 2,
                    "path_confidence": 0.1,
                    "crosses_shortcut_or_exclusion": "False",
                    "touches_competing_history_core": "False",
                    "has_exclusion": "False",
                },
                {
                    "observation_id": "o2",
                    "scene": "s",
                    "frame_id": 2,
                    "mask_id": 2,
                    "target_history_id": "h2",
                    "expected_histories_json": "[\"h2\"]",
                    "accepted_path": "True",
                    "diagnostic_correct": "True",
                    "independent_path_count": 2,
                    "path_confidence": 0.1,
                    "crosses_shortcut_or_exclusion": "False",
                    "touches_competing_history_core": "False",
                    "has_exclusion": "False",
                },
                {
                    "observation_id": "o3",
                    "scene": "s",
                    "frame_id": 3,
                    "mask_id": 3,
                    "target_history_id": "h3",
                    "expected_histories_json": "[\"h4\"]",
                    "accepted_path": "True",
                    "diagnostic_correct": "False",
                    "independent_path_count": 2,
                    "path_confidence": 0.1,
                    "crosses_shortcut_or_exclusion": "True",
                    "touches_competing_history_core": "True",
                    "has_exclusion": "False",
                },
            ],
        )
        _write_csv(path_root / "shortcut_rows.csv", [{"observation_id": "o3"}])
        explanation_rows = root / "explanation_rows.csv"
        _write_csv(
            explanation_rows,
            [
                {"variant": "E6_counterfactual_semantic_material_underseg", "observation_id": "o1", "is_selected": "True", "posterior_top1_margin": 0.7},
                {"variant": "E6_counterfactual_semantic_material_underseg", "observation_id": "o2", "is_selected": "True", "posterior_top1_margin": 0.2},
                {"variant": "E6_counterfactual_semantic_material_underseg", "observation_id": "o3", "is_selected": "True", "posterior_top1_margin": 0.9},
            ],
        )
        explanation_summary = root / "explanation_summary.json"
        write_json(explanation_summary, {"phase": "fixture"})
        v56_core = root / "v56_core.json"
        v56_tentative = root / "v56_tentative.json"
        write_json(v56_core, {"core_purity": 0.9, "core_completeness": 0.5, "real_minus_shuffled_ARI": 0.1, "real_minus_no_temporal_ARI": 0.1})
        write_json(v56_tentative, {"expanded_completeness": 0.6, "tentative_underseg_rate": None})
        cfg = V60EmbeddingConfig(
            v60_path_root=path_root,
            v58_explanation_rows_path=explanation_rows,
            v58_explanation_summary_path=explanation_summary,
            v56_core_summary_path=v56_core,
            v56_tentative_summary_path=v56_tentative,
        )
        result = build_v60_manifold_embedding(cfg)
        states = {row["observation_id"]: row["state"] for row in result["node_state_rows"]}
        self.assertEqual(states["o1"], "confirmed")
        self.assertEqual(states["o2"], "tentative")
        self.assertEqual(states["o3"], "quarantine")
        outputs = write_v60_manifold_embedding(result, root / "out")
        self.assertTrue((root / "out" / "embedding_summary.json").exists())
        self.assertIn("manifold_metric_rows", outputs)


if __name__ == "__main__":
    unittest.main()
