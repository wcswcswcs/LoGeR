from __future__ import annotations

import csv
import shutil
import tempfile
import unittest
from pathlib import Path

from stream4d_native.v61_native_field import V61NativeFieldConfig, build_v61_native_field


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


class V61NativeFieldTest(unittest.TestCase):
    def test_exports_state_labels_without_method_gt(self) -> None:
        root = Path(tempfile.mkdtemp(prefix="v61_native_test_"))
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        states = root / "states.csv"
        _write_csv(
            states,
            [
                {
                    "material_node_id": "a:1",
                    "scene": "s",
                    "component_id": "c1",
                    "state": "confirmed",
                    "predicted_history_id": "h1",
                    "support_observation_ids_json": "[\"m:s:0:1\"]",
                    "candidate_evidence_types": "K_mat|K_sem",
                    "uses_gt_for_prediction": "False",
                },
                {
                    "material_node_id": "a:2",
                    "scene": "s",
                    "component_id": "c2",
                    "state": "quarantine",
                    "predicted_history_id": "h1||h2",
                    "support_observation_ids_json": "[\"m:s:0:2\"]",
                    "candidate_evidence_types": "K_mask",
                    "uses_gt_for_prediction": "False",
                },
            ],
        )
        queries = root / "query.csv"
        _write_csv(queries, [{"observation_id": "s:0:1", "query_id": "q1"}])
        nodes = root / "nodes.csv"
        _write_csv(nodes, [{"node_id": "s:h1:mode0", "node_type": "semantic_mode", "history_id": "h1", "semantic_mode_id": "mode0"}])
        result = build_v61_native_field(
            V61NativeFieldConfig(
                refined_state_rows_path=states,
                query_rows_path=queries,
                v60_node_rows_path=nodes,
            )
        )
        summary = result["summary"]
        self.assertTrue(summary["method_safe_native_support_available"])
        self.assertFalse(summary["uses_gt_for_prediction"])
        self.assertEqual(summary["confirmed_carrier_count"], 1)
        self.assertEqual(summary["quarantine_carrier_count"], 1)
        self.assertTrue(all(row["state"] in {"confirmed", "tentative", "shared", "quarantine", "unknown"} for row in result["native_carrier_state_rows"]))


if __name__ == "__main__":
    unittest.main()
