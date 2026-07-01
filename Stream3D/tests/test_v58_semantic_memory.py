from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from stream4d_native.v58_semantic_memory import (
    V58SemanticMemoryConfig,
    build_v58_semantic_memory,
    write_v58_semantic_memory,
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


class V58SemanticMemoryTest(unittest.TestCase):
    def _fixture(self, root: Path) -> V58SemanticMemoryConfig:
        support_rows = root / "support.csv"
        history_rows = root / "history.csv"
        update_rows = root / "updates.csv"
        objectlet_rows = root / "objectlets.csv"
        _write_csv(
            support_rows,
            [
                {"variant": "I0_visible_tau0.10", "mask_observation_id": "sceneA:0:1", "scene": "sceneA", "frame_id": 0, "mask_id": 1, "component_id": "c1", "support_count": 10},
                {"variant": "I0_visible_tau0.10", "mask_observation_id": "sceneA:1:2", "scene": "sceneA", "frame_id": 1, "mask_id": 2, "component_id": "c1", "support_count": 9},
                {"variant": "I0_visible_tau0.10", "mask_observation_id": "sceneA:2:3", "scene": "sceneA", "frame_id": 2, "mask_id": 3, "component_id": "c2", "support_count": 8},
                {"variant": "I0_visible_tau0.10", "mask_observation_id": "sceneA:3:4", "scene": "sceneA", "frame_id": 3, "mask_id": 4, "component_id": "c2", "support_count": 7},
            ],
        )
        _write_csv(
            history_rows,
            [
                {"history_id": "h1", "scene": "sceneA", "dominant_gt_diagnostic": "1"},
                {"history_id": "h2", "scene": "sceneA", "dominant_gt_diagnostic": "2"},
            ],
        )
        _write_csv(
            update_rows,
            [
                {"history_id": "h1", "candidate_id": "sceneA:1:2", "update_state": "confirmed_update"},
                {"history_id": "h2", "candidate_id": "sceneA:3:4", "update_state": "confirmed_update"},
            ],
        )
        _write_csv(
            objectlet_rows,
            [
                {"variant": "L11_dynamic_uncovered_gain_dup010", "objectlet_id": "h1", "source_mask_observation_id": "sceneA:0:1", "underseg_proxy": "False"},
                {"variant": "L11_dynamic_uncovered_gain_dup010", "objectlet_id": "h2", "source_mask_observation_id": "sceneA:2:3", "underseg_proxy": "True"},
            ],
        )
        return V58SemanticMemoryConfig(
            support_rows_path=support_rows,
            history_rows_path=history_rows,
            history_update_rows_path=update_rows,
            objectlet_rows_path=objectlet_rows,
            output_root=root / "out",
            backend="dinov2_timm",
            device="cpu",
            write_mask_feature_vectors=True,
        )

    def test_semantic_memory_builds_feature_bank_and_shortlist(self) -> None:
        features = {
            "sceneA:0:1": [1.0, 0.0],
            "sceneA:1:2": [0.98, 0.02],
            "sceneA:2:3": [0.0, 1.0],
            "sceneA:3:4": [0.03, 0.97],
        }

        def fake_feature(mask_observation_id: str, **_kwargs):
            return features[mask_observation_id], {
                "scene": "sceneA",
                "frame_id": int(mask_observation_id.split(":")[1]),
                "mask_id": int(mask_observation_id.split(":")[2]),
                "semantic_feature_missing_reason": "",
                "semantic_mask_pixel_count": 12,
            }

        with tempfile.TemporaryDirectory() as tmp, patch(
            "stream4d_native.v58_semantic_memory._semantic_mask_feature",
            side_effect=fake_feature,
        ):
            cfg = self._fixture(Path(tmp))
            result = build_v58_semantic_memory(cfg)
            summary = result["summary"]
            self.assertEqual(summary["feature_success_rate"], 1.0)
            self.assertEqual(summary["mask_feature_count"], 4)
            self.assertEqual(summary["component_feature_count"], 2)
            self.assertEqual(summary["history_shortlist_recall@3"], 1.0)
            self.assertFalse(summary["gate"]["pass"])
            self.assertIn("same_category_confusion_rate_pass", summary["semantic_claim_blockers"])
            self.assertFalse(summary["uses_gt_for_prediction"])
            paths = write_v58_semantic_memory(result, cfg.output_root)
            for rel in paths.values():
                self.assertTrue((Path(tmp) / "out" / Path(rel).name).exists() or Path(rel).exists())

    def test_colorhist_backend_blocks_semantic_claim(self) -> None:
        def fake_feature(mask_observation_id: str, **_kwargs):
            return [1.0, 0.0] if mask_observation_id.endswith(("1", "2")) else [0.0, 1.0], {
                "scene": "sceneA",
                "frame_id": 0,
                "mask_id": 1,
                "semantic_feature_missing_reason": "",
                "semantic_mask_pixel_count": 12,
            }

        with tempfile.TemporaryDirectory() as tmp, patch(
            "stream4d_native.v58_semantic_memory._semantic_mask_feature",
            side_effect=fake_feature,
        ):
            cfg = self._fixture(Path(tmp))
            cfg = V58SemanticMemoryConfig(**{**cfg.__dict__, "backend": "colorhist"})
            result = build_v58_semantic_memory(cfg)
            self.assertFalse(result["summary"]["semantic_claim_allowed"])
            self.assertIn("only_colorhist_backend_available", result["summary"]["semantic_claim_blockers"])


if __name__ == "__main__":
    unittest.main()
