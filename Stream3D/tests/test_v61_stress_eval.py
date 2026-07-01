from __future__ import annotations

import csv
import shutil
import tempfile
import unittest
from pathlib import Path

from stream4d_native.v47_common import write_json
from stream4d_native.v61_stress_eval import V61StressEvalConfig, build_v61_stress_eval


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


class V61StressEvalTest(unittest.TestCase):
    def test_memory_beats_mask_only_under_dropout_proxy(self) -> None:
        root = Path(tempfile.mkdtemp(prefix="v61_stress_test_"))
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        rows = []
        for idx in range(4):
            rows.append(
                {
                    "material_node_id": f"a:{idx}",
                    "scene": "s",
                    "state": "confirmed",
                    "predicted_history_id": "h1" if idx < 2 else "h2",
                    "candidate_history_id": "h1" if idx < 2 else "h2",
                    "support_observation_ids_json": f"[\"m:s:{idx}:1\"]",
                    "diagnostic_expected_history_id": "h1" if idx < 2 else "h2",
                    "has_K_sem": "True",
                    "has_K_mat": "True",
                }
            )
        states = root / "states.csv"
        _write_csv(states, rows)
        query_summary = root / "query.json"
        write_json(query_summary, {"gate": {"pass": False}})
        v56 = root / "v56.csv"
        _write_csv(
            v56,
            [
                {
                    "stress_type": "mask_dropout",
                    "stress_strength": "0.50",
                    "core_ARI": 0.1,
                    "core_purity": 0.5,
                    "core_completeness": 0.5,
                    "expanded_ARI": 0.2,
                    "expanded_purity": 0.5,
                    "expanded_completeness": 0.5,
                }
            ],
        )
        result = build_v61_stress_eval(
            V61StressEvalConfig(
                global_state_rows_path=states,
                refined_state_rows_path=states,
                query_summary_path=query_summary,
                v56_stress_rows_path=v56,
            )
        )
        summary = result["summary"]
        self.assertFalse(summary["uses_gt_for_prediction"])
        self.assertGreaterEqual(summary["stress_setting_count"], 6)
        first = result["stress_setting_rows"][0]
        self.assertIn("v61_refined_minus_mask_only_ARI", first)
        self.assertIn("D5_v61_refined_manifold", {row["row"] for row in result["stress_metric_rows"]})


if __name__ == "__main__":
    unittest.main()
