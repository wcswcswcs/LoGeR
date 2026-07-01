from __future__ import annotations

import csv
import shutil
import tempfile
import unittest
from pathlib import Path

from stream4d_native.v47_common import write_json
from stream4d_native.v60_manifold_paths_v2 import V60PathV2Config, build_v60_manifold_paths_v2, write_v60_manifold_paths_v2


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


class V60ManifoldPathsV2Test(unittest.TestCase):
    def test_path_v2_uses_calibrated_gate(self) -> None:
        root = Path(tempfile.mkdtemp(prefix="v60_path_v2_test_"))
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        fact = root / "fact.json"
        graph = root / "graph.json"
        v59 = root / "v59_path"
        write_json(fact, {"phase2_same_category_calibrated": {"pass": True, "method_false_rate": 0.0, "method_pair_count": 50, "method_false_count": 0, "method_wilson_upper95": 0.04}})
        write_json(graph, {"gate": {"pass": True}})
        write_json(
            v59 / "path_summary.json",
            {
                "accepted_path_count": 1,
                "path_precision_diagnostic": 1.0,
                "path_recall_proxy": 0.5,
                "part_to_core_path_precision": 1.0,
                "shortcut_quarantine_precision": 1.0,
                "shortcut_quarantine_count": 1,
                "false_shortcut_count": 0,
                "mean_path_length": 3.0,
                "paths_with_both_semantic_and_material_rate": 1.0,
            },
        )
        _write_csv(v59 / "path_rows.csv", [{"observation_id": "o1", "accepted_path": "True", "path_length": 3, "has_semantic_path": "True", "has_material_path": "True"}])
        _write_csv(v59 / "shortcut_rows.csv", [{"observation_id": "o2"}])
        cfg = V60PathV2Config(v60_fact_lock_path=fact, v60_graph_summary_path=graph, v59_path_root=v59)
        result = build_v60_manifold_paths_v2(cfg)
        self.assertTrue(result["summary"]["gate"]["pass"])
        self.assertGreater(result["path_rows"][0]["path_confidence"], 0.0)
        outputs = write_v60_manifold_paths_v2(result, root / "out")
        self.assertTrue((root / "out" / "path_summary.json").exists())
        self.assertIn("shortcut_rows", outputs)


if __name__ == "__main__":
    unittest.main()
