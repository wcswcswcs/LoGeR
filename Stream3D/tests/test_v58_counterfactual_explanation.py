from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from stream4d_native.v58_counterfactual_explanation import (
    V58CounterfactualConfig,
    build_v58_counterfactual_explanation,
    write_v58_counterfactual_explanation,
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


class V58CounterfactualExplanationTest(unittest.TestCase):
    def _fixture(self, root: Path) -> V58CounterfactualConfig:
        semantic = root / "semantic"
        semantic.mkdir(parents=True, exist_ok=True)
        (semantic / "semantic_memory_summary.json").write_text(
            json.dumps(
                {
                    "backend": "dinov2_timm",
                    "gate": {"pass": False},
                    "semantic_claim_allowed": True,
                }
            ),
            encoding="utf-8",
        )
        features = {
            "sceneA:0:1": [1.0, 0.0],
            "sceneA:1:2": [0.98, 0.02],
            "sceneA:1:3": [0.7, 0.7],
            "sceneA:1:4": [-1.0, 0.0],
            "sceneA:0:9": [0.0, 1.0],
        }
        _write_csv(
            semantic / "mask_feature_rows.csv",
            [
                {
                    "mask_observation_id": key,
                    "scene": "sceneA",
                    "frame_id": key.split(":")[1],
                    "mask_id": key.split(":")[2],
                    "feature_available": "True",
                    "feature_json": json.dumps(value),
                }
                for key, value in features.items()
            ],
        )
        _write_csv(
            semantic / "history_sample_rows.csv",
            [
                {"history_id": "sceneA|L11|obj001", "mask_observation_id": "sceneA:0:1", "feature_available": "True"},
                {"history_id": "sceneA|L11|obj002", "mask_observation_id": "sceneA:0:9", "feature_available": "True"},
            ],
        )
        history_rows = root / "history_rows.csv"
        update_rows = root / "update_rows.csv"
        support_rows = root / "support_rows.csv"
        objectlet_rows = root / "objectlet_rows.csv"
        v56_core = root / "v56_core.json"
        v56_tentative = root / "v56_tentative.json"
        _write_csv(
            history_rows,
            [
                {"history_id": "sceneA|L11|obj001", "scene": "sceneA", "dominant_gt_diagnostic": "1"},
                {"history_id": "sceneA|L11|obj002", "scene": "sceneA", "dominant_gt_diagnostic": "2"},
            ],
        )
        _write_csv(
            update_rows,
            [
                {"history_id": "sceneA|L11|obj001", "candidate_id": "sceneA:1:2", "update_state": "confirmed_update"},
                {"history_id": "sceneA|L11|obj001", "candidate_id": "sceneA:1:3", "update_state": "partial_update"},
            ],
        )
        _write_csv(
            support_rows,
            [
                {"variant": "I0_visible_tau0.10", "mask_observation_id": "sceneA:0:1", "scene": "sceneA", "frame_id": 0, "mask_id": 1, "component_id": "c1", "support_count": 10, "diagnostic_gt_instance": "1"},
                {"variant": "I0_visible_tau0.10", "mask_observation_id": "sceneA:1:2", "scene": "sceneA", "frame_id": 1, "mask_id": 2, "component_id": "c1", "support_count": 9, "diagnostic_gt_instance": "1"},
                {"variant": "I0_visible_tau0.10", "mask_observation_id": "sceneA:1:3", "scene": "sceneA", "frame_id": 1, "mask_id": 3, "component_id": "c1", "support_count": 5, "diagnostic_gt_instance": "1"},
                {"variant": "I0_visible_tau0.10", "mask_observation_id": "sceneA:1:3", "scene": "sceneA", "frame_id": 1, "mask_id": 3, "component_id": "c2", "support_count": 5, "diagnostic_gt_instance": "2"},
                {"variant": "I0_visible_tau0.10", "mask_observation_id": "sceneA:1:4", "scene": "sceneA", "frame_id": 1, "mask_id": 4, "component_id": "c4", "support_count": 7, "diagnostic_gt_instance": "4"},
                {"variant": "I0_visible_tau0.10", "mask_observation_id": "sceneA:0:9", "scene": "sceneA", "frame_id": 0, "mask_id": 9, "component_id": "c2", "support_count": 8, "diagnostic_gt_instance": "2"},
            ],
        )
        _write_csv(
            objectlet_rows,
            [
                {"variant": "L11_dynamic_uncovered_gain_dup010", "objectlet_id": "sceneA|L11|obj001", "source_mask_observation_id": "sceneA:0:1", "component_ids": json.dumps(["c1"]), "underseg_proxy": "False"},
                {"variant": "L11_dynamic_uncovered_gain_dup010", "objectlet_id": "sceneA|L11|obj002", "source_mask_observation_id": "sceneA:0:9", "component_ids": json.dumps(["c2"]), "underseg_proxy": "False"},
                {"variant": "L11_dynamic_uncovered_gain_dup010", "objectlet_id": "sceneA|L11|obj999", "source_mask_observation_id": "sceneA:1:3", "component_ids": json.dumps(["c1", "c2"]), "underseg_proxy": "True"},
            ],
        )
        v56_core.write_text(
            json.dumps(
                {
                    "core_ARI": 0.5,
                    "core_purity": 0.9,
                    "core_completeness": 0.6,
                    "history_temporal_span_mean": 1.0,
                    "update_precision_diagnostic": 0.8,
                }
            ),
            encoding="utf-8",
        )
        v56_tentative.write_text(json.dumps({"expanded_ARI": 0.55, "expanded_purity": 0.88, "expanded_completeness": 0.65}), encoding="utf-8")
        return V58CounterfactualConfig(
            semantic_root=semantic,
            support_rows_path=support_rows,
            history_rows_path=history_rows,
            history_update_rows_path=update_rows,
            objectlet_rows_path=objectlet_rows,
            v56_core_summary_path=v56_core,
            v56_tentative_summary_path=v56_tentative,
            output_root=root / "out",
            visualization_root=root / "viz",
        )

    def test_counterfactual_outputs_metrics_and_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._fixture(Path(tmp))
            result = build_v58_counterfactual_explanation(cfg)
            summary = result["summary"]
            self.assertEqual(summary["observation_count"], 5)
            self.assertFalse(summary["uses_gt_for_prediction"])
            self.assertTrue(summary["uses_gt_for_diagnostic_labels"])
            self.assertIn("gate", summary)
            self.assertIsNone(summary["same_category_confusion_rate"])
            underseg_rows = [
                row
                for row in result["explanation_rows"]
                if row["explanation_type"] == "underseg_mixture" and row["observation_id"] == "sceneA:1:3"
            ]
            self.assertTrue(underseg_rows)
            histories = json.loads(underseg_rows[0]["candidate_history_ids_json"])
            self.assertIn("sceneA|L11|obj001", histories)
            paths = write_v58_counterfactual_explanation(result, cfg.output_root)
            for rel in paths.values():
                self.assertTrue((Path(tmp) / "out" / Path(rel).name).exists() or Path(rel).exists())


if __name__ == "__main__":
    unittest.main()
