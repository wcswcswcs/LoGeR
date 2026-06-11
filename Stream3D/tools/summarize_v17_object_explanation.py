from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"_missing": True, "_path": str(path)}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload["_path"] = str(path)
    return payload


def _row_by_method(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    for row in rows:
        if row.get("method") == method:
            return row
    return {}


def _own_row(rows: list[dict[str, Any]], config: str) -> dict[str, Any]:
    for row in rows:
        if row.get("prediction_config") == config and row.get("pre_points_config") == config:
            return row
    return {}


def _oracle_row(oracle_payload: dict[str, Any], k: int) -> dict[str, Any]:
    for row in oracle_payload.get("rows", []):
        if int(row.get("k", -1)) == int(k):
            return row
    return {}


def _oracle_selected_sets(oracle_payload: dict[str, Any], k: int) -> dict[str, set[int]]:
    out: dict[str, set[int]] = {}
    for scene in oracle_payload.get("scenes", []):
        selected: set[int] = set()
        for gt_row in scene.get("per_gt", []):
            for idx in gt_row.get("selected_pred_indices", [])[: int(k)]:
                selected.add(int(idx))
        out[str(scene.get("scene"))] = selected
    return out


def _solver_selected_sets(solver_payload: dict[str, Any]) -> dict[str, set[int]]:
    return {
        str(scene.get("scene")): {int(v) for v in scene.get("selected_measurement_indices", [])}
        for scene in solver_payload.get("scenes", [])
    }


def _overlap_metrics(oracle_payload: dict[str, Any], solver_payload: dict[str, Any], k: int) -> dict[str, Any]:
    oracle = _oracle_selected_sets(oracle_payload, k)
    solver = _solver_selected_sets(solver_payload)
    oracle_total = 0
    solver_total = 0
    inter_total = 0
    scene_rows = []
    for scene, oracle_set in oracle.items():
        solver_set = solver.get(scene, set())
        inter = oracle_set & solver_set
        oracle_total += len(oracle_set)
        solver_total += len(solver_set)
        inter_total += len(inter)
        scene_rows.append(
            {
                "scene": scene,
                "oracle_selected_count": int(len(oracle_set)),
                "solver_selected_count": int(len(solver_set)),
                "selected_overlap_with_oracle": int(len(inter)),
                "oracle_recall_by_solver": float(len(inter) / max(len(oracle_set), 1)),
                "solver_precision_vs_oracle": float(len(inter) / max(len(solver_set), 1)),
            }
        )
    return {
        "oracle_selected_count": int(oracle_total),
        "solver_selected_count": int(solver_total),
        "selected_overlap_with_oracle": int(inter_total),
        "oracle_recall_by_solver": float(inter_total / max(oracle_total, 1)),
        "solver_precision_vs_oracle": float(inter_total / max(solver_total, 1)),
        "scene_rows": scene_rows,
    }


def _recovery(real: dict[str, Any], baseline: dict[str, Any], oracle: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ("ap", "ap50", "ap25"):
        r = real.get(key)
        b = baseline.get(key)
        o = oracle.get(key)
        if r is None or b is None or o is None or abs(float(o) - float(b)) < 1e-12:
            out[f"{key}_recovery"] = None
        else:
            out[f"{key}_recovery"] = float((float(r) - float(b)) / (float(o) - float(b)))
    return out


def _gate(real_own: dict[str, Any], real_s1: dict[str, Any], shuffle_own: dict[str, Any], no_temporal_own: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "own_ap_ge_0p16": bool((real_own.get("ap") or 0.0) >= 0.16),
        "own_ap50_ge_0p35": bool((real_own.get("ap50") or 0.0) >= 0.35),
        "own_ap25_ge_0p55": bool((real_own.get("ap25") or 0.0) >= 0.55),
        "own_pre_ge_25pct": bool((real_own.get("pre_points_ratio") or 0.0) >= 0.25),
        "s1_ap_ge_0p10": bool((real_s1.get("ap") or 0.0) >= 0.10),
        "s1_ap50_ge_0p22": bool((real_s1.get("ap50") or 0.0) >= 0.22),
        "s1_ap25_ge_0p40": bool((real_s1.get("ap25") or 0.0) >= 0.40),
        "real_minus_shuffle_ap50_ge_0p05": bool(
            (real_own.get("ap50") is not None)
            and (shuffle_own.get("ap50") is not None)
            and float(real_own["ap50"]) - float(shuffle_own["ap50"]) >= 0.05
        ),
        "real_minus_no_temporal_ap25_ge_0p05": bool(
            (real_own.get("ap25") is not None)
            and (no_temporal_own.get("ap25") is not None)
            and float(real_own["ap25"]) - float(no_temporal_own["ap25"]) >= 0.05
        ),
    }
    checks["minimum_pass"] = bool(all(checks.values()))
    checks["strong_pass"] = bool(
        (real_own.get("ap") or 0.0) >= 0.22
        and (real_own.get("ap50") or 0.0) >= 0.45
        and (real_own.get("ap25") or 0.0) >= 0.65
        and (real_s1.get("ap") or 0.0) >= 0.16
    )
    return checks


def _fmt(value: Any, scale: float = 1.0) -> str:
    if value is None or value == "":
        return "NA"
    try:
        return f"{float(value) * float(scale):.6f}"
    except Exception:
        return str(value)


def _write_outputs(prefix: Path, payload: dict[str, Any]) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".json").write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")
    rows = payload.get("matrix_rows", [])
    if rows:
        flat = [{key: value for key, value in row.items() if key not in {"scene_rows", "manifest"}} for row in rows]
        fieldnames = sorted({key for row in flat for key in row.keys()})
        with prefix.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in flat:
                writer.writerow({key: row.get(key, "") for key in fieldnames})
    summary = payload["summary"]
    lines = [
        "# Stream4D v17 Object Explanation Summary",
        "",
        "This file summarizes actual generated artifacts; it does not recompute AP.",
        "",
        "## Gates",
        "",
    ]
    for key, value in summary["gate"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Key Metrics",
            "",
            f"- M17-real own AP/AP50/AP25: `{_fmt(summary['real_own'].get('ap'))} / {_fmt(summary['real_own'].get('ap50'))} / {_fmt(summary['real_own'].get('ap25'))}`",
            f"- M17-real own pre%: `{_fmt(summary['real_own'].get('pre_points_ratio'), 100.0)}`",
            f"- M17-real on S1 AP/AP50/AP25: `{_fmt(summary['real_s1'].get('ap'))} / {_fmt(summary['real_s1'].get('ap50'))} / {_fmt(summary['real_s1'].get('ap25'))}`",
            f"- shuffle own AP/AP50/AP25: `{_fmt(summary['shuffle_own'].get('ap'))} / {_fmt(summary['shuffle_own'].get('ap50'))} / {_fmt(summary['shuffle_own'].get('ap25'))}`",
            f"- no-temporal own AP/AP50/AP25: `{_fmt(summary['no_temporal_own'].get('ap'))} / {_fmt(summary['no_temporal_own'].get('ap50'))} / {_fmt(summary['no_temporal_own'].get('ap25'))}`",
            f"- area-only own AP/AP50/AP25: `{_fmt(summary['area_only_own'].get('ap'))} / {_fmt(summary['area_only_own'].get('ap50'))} / {_fmt(summary['area_only_own'].get('ap25'))}`",
            "",
            "## Oracle Recovery",
            "",
        ]
    )
    for key, value in summary["recovery"].items():
        lines.append(f"- {key}: `{value}`")
    overlap = summary["oracle_overlap"]
    lines.extend(
        [
            f"- oracle_selected_count: `{overlap.get('oracle_selected_count')}`",
            f"- solver_selected_count: `{overlap.get('solver_selected_count')}`",
            f"- selected_overlap_with_oracle: `{overlap.get('selected_overlap_with_oracle')}`",
            f"- oracle_recall_by_solver: `{overlap.get('oracle_recall_by_solver')}`",
            f"- solver_precision_vs_oracle: `{overlap.get('solver_precision_vs_oracle')}`",
            "",
            "## Unified Matrix",
            "",
            "| method | prediction | pre_points | AP | AP50 | AP25 | pre% | union% | manifest pass | method table |",
            "|---|---|---|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("method", "")),
                    str(row.get("prediction_config", "")),
                    str(row.get("pre_points_config", "")),
                    _fmt(row.get("ap")),
                    _fmt(row.get("ap50")),
                    _fmt(row.get("ap25")),
                    _fmt(row.get("pre_points_ratio"), 100.0),
                    _fmt(row.get("prediction_union_ratio"), 100.0),
                    str(row.get("manifest_integrity_pass")),
                    str(row.get("method_table_allowed")),
                ]
            )
            + " |"
        )
    prefix.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--matrix-json", required=True)
    parser.add_argument("--real-config", default="stream4d_v17_m17_real_probe5")
    parser.add_argument("--shuffle-config", default="stream4d_v17_m17_shuffle_probe5")
    parser.add_argument("--no-temporal-config", default="stream4d_v17_m17_no_temporal_probe5")
    parser.add_argument("--area-only-config", default="stream4d_v17_m17_area_only_probe5")
    parser.add_argument("--real-s1-method", default="M17-real on S1")
    parser.add_argument("--oracle-json", default="outputs/audit/v16_phase1/c_hybrid_union_oracle_probe5.json")
    parser.add_argument("--solver-summary", default="outputs/v17_object_explanation_solver/stream4d_v17_m17_real_probe5_summary.json")
    parser.add_argument("--feature-json", default="outputs/audit/v17_phase2/c_hybrid_oracle_feature_separation_probe5.json")
    parser.add_argument("--measurement-bank-json", default="outputs/audit/v17_phase1/measurement_bank_fixed_probe5.json")
    parser.add_argument("--oracle-k", type=int, default=8)
    parser.add_argument("--output-prefix", default="outputs/audit/v17_phase4/object_explanation_summary_probe5")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    matrix = _load_json(root / args.matrix_json)
    rows = matrix.get("rows", [])
    oracle = _load_json(root / args.oracle_json)
    solver = _load_json(root / args.solver_summary)
    feature = _load_json(root / args.feature_json)
    bank = _load_json(root / args.measurement_bank_json)
    real_own = _own_row(rows, args.real_config)
    shuffle_own = _own_row(rows, args.shuffle_config)
    no_temporal_own = _own_row(rows, args.no_temporal_config)
    area_only_own = _own_row(rows, args.area_only_config)
    real_s1 = _row_by_method(rows, args.real_s1_method)
    oracle_k = _oracle_row(oracle, int(args.oracle_k))
    gate = _gate(real_own, real_s1, shuffle_own, no_temporal_own)
    recovery = _recovery(real_own, area_only_own, oracle_k)
    overlap = _overlap_metrics(oracle, solver, int(args.oracle_k))
    recovery["ap50_recovery_ge_0p40"] = bool((recovery.get("ap50_recovery") or -999.0) >= 0.40)
    recovery["ap25_recovery_ge_0p40"] = bool((recovery.get("ap25_recovery") or -999.0) >= 0.40)
    overlap["oracle_recall_ge_0p35"] = bool(overlap["oracle_recall_by_solver"] >= 0.35)
    overlap["solver_precision_ge_0p30"] = bool(overlap["solver_precision_vs_oracle"] >= 0.30)
    payload = {
        "args": vars(args),
        "summary": {
            "gate": gate,
            "real_own": real_own,
            "real_s1": real_s1,
            "shuffle_own": shuffle_own,
            "no_temporal_own": no_temporal_own,
            "area_only_own": area_only_own,
            "oracle_k": oracle_k,
            "recovery": recovery,
            "oracle_overlap": overlap,
            "feature_summary": feature.get("summary", {}),
            "measurement_bank_gate": bank.get("aggregate", {}).get("gate", {}),
        },
        "matrix_rows": rows,
    }
    _write_outputs(root / args.output_prefix, payload)
    print(json.dumps(_json_safe(payload["summary"]["gate"]), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
