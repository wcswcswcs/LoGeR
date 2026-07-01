#!/usr/bin/env python3
"""Audit D4RT support for the current best v100 union history links."""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
if str(STREAM3D_ROOT) not in sys.path:
    sys.path.insert(0, str(STREAM3D_ROOT))

from tools import build_v100_phase4h_overlap3_exact_history_memory as p4h  # noqa: E402
from tools import build_v100_phase5_da3_d4rt_verifier_audit as p5  # noqa: E402


AUDIT_ROOT = STREAM3D_ROOT / "outputs/audit"
OUT_DIR = AUDIT_ROOT / "v100_phase5b_current_union_d4rt_support_audit"
PHASE4P_DIR = AUDIT_ROOT / "v100_phase4p_multi_semantic_union_repair"
PHASE5_DIR = AUDIT_ROOT / "v100_phase5_da3_d4rt_verifier_audit"
D4RT_SUPPORT_OVERLAP_TAU = 0.20


def _rel(path: Path | str) -> str:
    return p4h._rel(path)


def _num(value: Any, default: float = 0.0) -> float:
    return p4h._num(value, default)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    p4h._write_csv(path, rows)


def _write_json(path: Path, payload: Any) -> None:
    p4h._write_json(path, payload)


def _artifact_rows(paths: list[tuple[Path, str, str]]) -> list[dict[str, Any]]:
    return [
        {
            "schema_version": "stream4d_v100_phase5b_artifact_manifest_row_v1",
            "phase_id": "v100_phase5b_current_union_d4rt_support_audit",
            "artifact_path": _rel(path),
            "artifact_type": kind,
            "size_bytes": path.stat().st_size if path.exists() else 0,
            "sha256": p4h._sha256(path) if path.exists() and path.is_file() else "",
            "note": note,
        }
        for path, kind, note in paths
    ]


def main() -> int:
    started = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    phase4p = json.loads((PHASE4P_DIR / "summary.json").read_text(encoding="utf-8"))
    phase5 = json.loads((PHASE5_DIR / "summary.json").read_text(encoding="utf-8"))
    best_variant = str(phase4p["best_variant_id"])
    d4rt_lookup = p5._d4rt_candidate_lookup()
    edges = pd.read_csv(PHASE4P_DIR / "union_edge_rows.csv")
    edges = edges[edges["variant_id"] == best_variant]

    support_rows: list[dict[str, Any]] = []
    for item in edges.to_dict(orient="records"):
        a = p5._canon_oid(item.get("mv_object_id_a", ""))
        b = p5._canon_oid(item.get("mv_object_id_b", ""))
        cand = d4rt_lookup.get(frozenset({a, b}))
        overlap = _num(cand.get("object_anchor_overlap")) if cand else 0.0
        supported = bool(cand is not None and overlap >= D4RT_SUPPORT_OVERLAP_TAU)
        support_rows.append(
            {
                "schema_version": "stream4d_v100_phase5b_d4rt_support_row_v1",
                "phase_id": "v100_phase5b_current_union_d4rt_support_audit",
                "variant_id": best_variant,
                "dataset_split": str(item.get("dataset_split", "")),
                "source_id": str(item.get("source_id", "")),
                "source_variant_id": str(item.get("source_variant_id", "")),
                "source_family": str(item.get("source_family", "")),
                "mv_object_id_a": str(item.get("mv_object_id_a", "")),
                "mv_object_id_b": str(item.get("mv_object_id_b", "")),
                "canonical_object_id_a": a,
                "canonical_object_id_b": b,
                "d4rt_supported": supported,
                "best_d4rt_overlap": overlap,
                "best_d4rt_anchor_family": cand.get("anchor_family", "") if cand else "",
                "best_d4rt_shared_anchor_count": cand.get("shared_anchor_count", "") if cand else "",
                "support_tau": D4RT_SUPPORT_OVERLAP_TAU,
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )

    summary_rows: list[dict[str, Any]] = []
    for split in sorted({row["dataset_split"] for row in support_rows}):
        rows = [row for row in support_rows if row["dataset_split"] == split]
        supported = [row for row in rows if row["d4rt_supported"]]
        summary_rows.append(
            {
                "schema_version": "stream4d_v100_phase5b_summary_row_v1",
                "phase_id": "v100_phase5b_current_union_d4rt_support_audit",
                "variant_id": best_variant,
                "dataset_split": split,
                "union_edge_count": len(rows),
                "d4rt_supported_edge_count": len(supported),
                "d4rt_support_rate": float(len(supported) / max(1, len(rows))),
                "max_d4rt_overlap": max([_num(row["best_d4rt_overlap"]) for row in rows], default=0.0),
                "support_tau": D4RT_SUPPORT_OVERLAP_TAU,
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )

    holdout_summary = next((row for row in summary_rows if row["dataset_split"] == "holdout"), {})
    support_rate_gate = _num(holdout_summary.get("d4rt_support_rate")) >= 0.5
    d4rt_usefulness_gate = _num(phase5.get("d4rt_real_minus_control_MV_AP_scene")) >= 0.005
    phase5b_pass = bool(support_rate_gate and d4rt_usefulness_gate)
    gate_rows = [
        {
            "gate_id": "current_best_holdout_union_edges_d4rt_support_rate_ge_0p5",
            "pass": support_rate_gate,
            "expected": ">=0.5 of current best holdout union edges supported by D4RT real_R20/R40 anchors",
            "observed": holdout_summary.get("d4rt_support_rate", ""),
            "severity": "verifier_required",
        },
        {
            "gate_id": "existing_d4rt_real_minus_control_ge_0p005",
            "pass": d4rt_usefulness_gate,
            "expected": ">=0.005 scene AP real-control signal from Phase5",
            "observed": phase5.get("d4rt_real_minus_control_MV_AP_scene"),
            "severity": "control_required",
        },
    ]
    failure_rows = [
        {
            "schema_version": "stream4d_v100_phase5b_failure_row_v1",
            "phase_id": "v100_phase5b_current_union_d4rt_support_audit",
            "failure_id": row["gate_id"],
            "severity": row["severity"],
            "expected": row["expected"],
            "observed": row["observed"],
            "repair_direction": "If D4RT support rate is low, current D4RT anchors cannot validate or expand the best v100 union links; keep D4RT diagnostic-only.",
        }
        for row in gate_rows
        if not bool(row["pass"])
    ]

    support_csv = OUT_DIR / "current_union_d4rt_support_rows.csv"
    summary_csv = OUT_DIR / "summary_rows.csv"
    gate_csv = OUT_DIR / "variant_gate_rows.csv"
    failure_csv = OUT_DIR / "variant_failure_rows.csv"
    performance_csv = OUT_DIR / "performance_rows.csv"
    artifact_csv = OUT_DIR / "artifact_manifest_rows.csv"
    _write_csv(support_csv, support_rows)
    _write_csv(summary_csv, summary_rows)
    _write_csv(gate_csv, gate_rows)
    _write_csv(failure_csv, failure_rows)
    _write_csv(
        performance_csv,
        [
            {
                "schema_version": "stream4d_v100_phase5b_performance_row_v1",
                "phase_id": "v100_phase5b_current_union_d4rt_support_audit",
                "case_id": "current_phase4p_best_union_d4rt_support_lookup",
                "runtime_sec": time.time() - started,
                "union_edge_count": len(support_rows),
                "d4rt_lookup_edge_count": len(d4rt_lookup),
                "v65_evaluator_runs": 0,
            }
        ],
    )
    _write_csv(
        artifact_csv,
        _artifact_rows(
            [
                (support_csv, "csv", "D4RT support labels for Phase4p best union edges"),
                (summary_csv, "csv", "D4RT support summary by split"),
                (gate_csv, "csv", "Phase5b gates"),
                (failure_csv, "csv", "Phase5b failures"),
                (performance_csv, "csv", "Phase5b runtime"),
            ]
        ),
    )

    summary = {
        "schema_version": "stream4d_v100_phase5b_current_union_d4rt_support_audit_summary_v1",
        "phase_id": "v100_phase5b_current_union_d4rt_support_audit",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "runtime_sec": time.time() - started,
        "decision": "PASS_D4RT_SUPPORT_FOR_CURRENT_UNION" if phase5b_pass else "NO_GO_D4RT_SUPPORT_FOR_CURRENT_UNION",
        "phase5b_pass": phase5b_pass,
        "failure_count": len(failure_rows),
        "best_phase4p_variant_id": best_variant,
        "phase4p_best_holdout_MV_AP_scene": phase4p.get("best_holdout_MV_AP_scene"),
        "phase4p_best_holdout_MV_AP50_scene": phase4p.get("best_holdout_MV_AP50_scene"),
        "summary_rows": summary_rows,
        "d4rt_real_minus_control_MV_AP_scene": phase5.get("d4rt_real_minus_control_MV_AP_scene"),
        "uses_gt_for_prediction": False,
        "uses_future": False,
        "outputs": {
            "summary": _rel(OUT_DIR / "summary.json"),
            "current_union_d4rt_support_rows": _rel(support_csv),
            "summary_rows": _rel(summary_csv),
            "variant_gate_rows": _rel(gate_csv),
            "variant_failure_rows": _rel(failure_csv),
            "performance_rows": _rel(performance_csv),
            "artifact_manifest_rows": _rel(artifact_csv),
        },
    }
    _write_json(OUT_DIR / "summary.json", summary)
    print(json.dumps(p4h._jsonable(summary), indent=2, sort_keys=True))
    return 0 if phase5b_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
