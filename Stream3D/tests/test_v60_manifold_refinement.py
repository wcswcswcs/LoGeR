from __future__ import annotations

import csv
import shutil
import tempfile
import unittest
from pathlib import Path

from stream4d_native.v47_common import write_json
from stream4d_native.v60_manifold_refinement import V60RefinementConfig, build_v60_manifold_refinement, write_v60_manifold_refinement


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


class V60ManifoldRefinementTest(unittest.TestCase):
    def test_refinement_rejects_unsafe_margin_promotion(self) -> None:
        root = Path(tempfile.mkdtemp(prefix="v60_refinement_test_"))
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        emb = root / "embedding"
        write_json(
            emb / "embedding_summary.json",
            {
                "core_purity": 0.91,
                "expanded_completeness": 0.2,
                "real_minus_shuffled_ARI": 0.1,
                "real_minus_no_temporal_ARI": 0.1,
            },
        )
        _write_csv(
            emb / "node_state_rows.csv",
            [
                {"observation_id": "a", "state": "tentative", "posterior_top1_margin": 0.6, "independent_path_count": 2, "crosses_shortcut_or_exclusion": "False", "diagnostic_correct": "False"},
                {"observation_id": "b", "state": "quarantine", "posterior_top1_margin": 0.7, "independent_path_count": 1, "crosses_shortcut_or_exclusion": "True", "diagnostic_correct": "False"},
            ],
        )
        result = build_v60_manifold_refinement(V60RefinementConfig(embedding_root=emb))
        self.assertTrue(result["summary"]["diagnostic_only_bypass"])
        self.assertFalse(result["summary"]["gate"]["pass"])
        self.assertEqual(result["summary"]["quarantine_precision_diagnostic"], 1.0)
        outputs = write_v60_manifold_refinement(result, root / "out")
        self.assertTrue((root / "out" / "refinement_summary.json").exists())
        self.assertIn("refinement_rows", outputs)


if __name__ == "__main__":
    unittest.main()
