from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from stream4d.measurement_bank import json_safe


BASELINE_ROWS = {
    "P0 on S0": {"ap": 0.235730, "ap50": 0.414306, "ap25": 0.537786},
    "P0 on S1": {"ap": 0.399213, "ap50": 0.597171, "ap25": 0.742535},
    "O38 own": {"ap": 0.081038, "ap50": 0.219225, "ap25": 0.492501},
    "repair_cmask own": {"ap": 0.101653, "ap50": 0.248464, "ap25": 0.494844},
    "repair_cmask on S1": {"ap": 0.102883, "ap50": 0.242779, "ap25": 0.576250},
    "P_v6compact on S1": {"ap": 0.284832, "ap50": 0.503962, "ap25": 0.671915},
}

GRAPH_INPUTS = {
    "v18 Phase1 pre-cut k16 d0.15 graph": "outputs/audit/v18_phase1_repair_precut_k16_d015/signed_surfel_graph_probe5.json",
    "v18 Phase1 bank16 k8 d0.25 graph": "outputs/audit/v18_phase1_repair_precut_k8/signed_surfel_graph_probe5.json",
    "v18 Phase1 grid48 k16 d0.15 graph": "outputs/audit/v18_phase1_grid48_precut_k16_d015/signed_surfel_graph_probe5.json",
}

ORACLE_INPUTS = {
    "v18 Phase2 bank16 k16 d0.15 GT edge oracle": "outputs/audit/v18_phase2_precut_k16_d015/edge_oracle_probe5.json",
    "v18 Phase2 bank16 k8 d0.25 GT edge oracle": "outputs/audit/v18_phase2_repair_bank16_k8_d025/edge_oracle_probe5.json",
    "v18 Phase2 grid48 k16 d0.15 GT edge oracle": "outputs/audit/v18_phase2_grid48_precut_k16_d015/edge_oracle_probe5.json",
}


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_error(row: dict[str, Any], expected: dict[str, float]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    ok = True
    for key in ("ap", "ap50", "ap25"):
        value = row.get(key)
        if value is None:
            out[f"{key}_abs_error_to_reference"] = None
            ok = False
            continue
        err = abs(float(value) - float(expected[key]))
        out[f"{key}_abs_error_to_reference"] = err
        ok = ok and err <= 1e-4
    out["reference_reproduced"] = bool(ok)
    return out


def _baseline_rows(root: Path) -> list[dict[str, Any]]:
    payload = _load_json(root / "outputs/audit/v18_phase0/unified_eval_matrix_probe5.json")
    if payload is None:
        return [
            {
                "kind": "baseline",
                "name": name,
                "status": "missing_v18_phase0_json",
                **expected,
                "reference_reproduced": False,
            }
            for name, expected in BASELINE_ROWS.items()
        ]
    by_name = {str(row.get("method")): row for row in payload.get("rows", [])}
    rows: list[dict[str, Any]] = []
    for name, expected in BASELINE_ROWS.items():
        src = by_name.get(name)
        if src is None:
            rows.append({"kind": "baseline", "name": name, "status": "missing_row", "reference_reproduced": False})
            continue
        row = {
            "kind": "baseline",
            "name": name,
            "status": "ok",
            "config": src.get("config"),
            "support": src.get("support"),
            "ap": src.get("ap"),
            "ap50": src.get("ap50"),
            "ap25": src.get("ap25"),
            "pre_points_ratio": src.get("pre_points_ratio"),
            "prediction_union_ratio": src.get("prediction_union_ratio"),
            "manifest_integrity_pass": src.get("manifest_integrity_pass"),
        }
        row.update(_metric_error(row, expected))
        rows.append(row)
    return rows


def _graph_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, rel in GRAPH_INPUTS.items():
        payload = _load_json(root / rel)
        if payload is None:
            rows.append({"kind": "graph", "name": name, "status": "missing_json"})
            continue
        numeric = payload.get("aggregate", {}).get("numeric_mean", {})
        gate = payload.get("aggregate", {}).get("gate", {})
        rows.append(
            {
                "kind": "graph",
                "name": name,
                "status": "ok",
                "num_nodes": numeric.get("num_nodes"),
                "num_edges": numeric.get("num_edges"),
                "track_length_visible_mean": numeric.get("track_length_visible_mean"),
                "uv_in01_rate": numeric.get("uv_in01_rate"),
                "cycle_uv_error_p90": numeric.get("cycle_uv_error_p90"),
                "raw_largest_graph_component_ratio": numeric.get("raw_largest_graph_component_ratio"),
                "largest_graph_component_ratio": numeric.get("largest_graph_component_ratio"),
                "precut_removed_edge_ratio": numeric.get("precut_removed_edge_ratio"),
                "phase1_pass_without_edge_count": gate.get("phase1_pass_without_edge_count"),
            }
        )
    return rows


def _oracle_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, rel in ORACLE_INPUTS.items():
        payload = _load_json(root / rel)
        if payload is None:
            rows.append({"kind": "oracle", "name": name, "status": "missing_json"})
            continue
        aggregate = payload.get("aggregate", {})
        numeric = aggregate.get("numeric_mean", {})
        rows.append(
            {
                "kind": "oracle",
                "name": name,
                "status": "ok",
                "config": aggregate.get("oracle_output_config"),
                "ap": aggregate.get("ap"),
                "ap50": aggregate.get("ap50"),
                "ap25": aggregate.get("ap25"),
                "node_gt_label_coverage": numeric.get("node_gt_label_coverage"),
                "edge_gt_label_coverage": numeric.get("edge_gt_label_coverage"),
                "num_exported_objects": numeric.get("num_exported_objects"),
                "num_exported_points": numeric.get("num_exported_points"),
                "num_oracle_components": numeric.get("num_oracle_components"),
                "phase2_min_gate": aggregate.get("phase2_min_gate"),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row.keys() if not isinstance(row.get(key), (dict, list))})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _write_md(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Stream4D v19 Phase0 Reproduction Summary",
        "",
        "## Baselines",
        "",
        "| row | AP | AP50 | AP25 | pre ratio | manifest | reproduced |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for row in payload["rows"]:
        if row.get("kind") != "baseline":
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("name")),
                    str(row.get("ap")),
                    str(row.get("ap50")),
                    str(row.get("ap25")),
                    str(row.get("pre_points_ratio")),
                    str(row.get("manifest_integrity_pass")),
                    str(row.get("reference_reproduced")),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Graphs", "", "| row | nodes | edges | raw largest | largest after pre-cut | pre-cut removed | gate |", "|---|---:|---:|---:|---:|---:|---|"])
    for row in payload["rows"]:
        if row.get("kind") != "graph":
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("name")),
                    str(row.get("num_nodes")),
                    str(row.get("num_edges")),
                    str(row.get("raw_largest_graph_component_ratio")),
                    str(row.get("largest_graph_component_ratio")),
                    str(row.get("precut_removed_edge_ratio")),
                    str(row.get("phase1_pass_without_edge_count")),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Oracles", "", "| row | AP | AP50 | AP25 | node cov | edge cov | objects | points | gate |", "|---|---:|---:|---:|---:|---:|---:|---:|---|"])
    for row in payload["rows"]:
        if row.get("kind") != "oracle":
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("name")),
                    str(row.get("ap")),
                    str(row.get("ap50")),
                    str(row.get("ap25")),
                    str(row.get("node_gt_label_coverage")),
                    str(row.get("edge_gt_label_coverage")),
                    str(row.get("num_exported_objects")),
                    str(row.get("num_exported_points")),
                    str(row.get("phase2_min_gate")),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--output-prefix", default="outputs/audit/v19_phase0/phase0_reproduction_probe5")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    rows = [*_baseline_rows(root), *_graph_rows(root), *_oracle_rows(root)]
    aggregate = {
        "phase": "v19_phase0_reproduction",
        "num_rows": len(rows),
        "num_missing": int(sum(1 for row in rows if row.get("status") != "ok")),
        "all_reference_baselines_reproduced": bool(
            all(row.get("reference_reproduced") for row in rows if row.get("kind") == "baseline")
        ),
    }
    payload = {"args": vars(args), "aggregate": aggregate, "rows": rows}
    output = Path(args.output_prefix)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")
    _write_csv(output.with_suffix(".csv"), rows)
    _write_md(output.with_suffix(".md"), payload)
    print(json.dumps(json_safe(aggregate), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
