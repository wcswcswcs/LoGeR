#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
AUDIT_ROOT = STREAM3D_ROOT / "outputs" / "audit"

PHASE_ID = "v103_supp_r5_phaseR5_0_fact_lock"
PLAN_DOC = REPO_ROOT / "docs/stream4d_v103_supplement_r5_support_weighted_affinity_plan.md"
EVALUATOR = STREAM3D_ROOT / "tools/run_v65_scene_multiview_ap.py"

DEFAULT_OUT = AUDIT_ROOT / "v103_supp_r5_fact_lock"
DEFAULT_PHASES1_ROOT = AUDIT_ROOT / "v103_supp_phaseS1_multirole_carriers"
DEFAULT_PHASES2_ROOT = AUDIT_ROOT / "v103_supp_phaseS2_role_aware_affinity"
DEFAULT_PHASE6D_ROOT = AUDIT_ROOT / "v103_phase6d_f2_skeleton_affinity_merge_phase9n_suppS1_d4rt48mix_s5repair_r4_directpair_guard"
DEFAULT_PHASES4_ROOT = AUDIT_ROOT / "v103_supp_phaseS4_post_birth_history_inheritance_phase6d_s5repair_r3_gate_r1"
DEFAULT_BASELINE_ROOT = AUDIT_ROOT / "v100_phase2c_overlap3_local_repair"
DEFAULT_D4RT_ROOT_BY_SCENE = {
    "scene0011_00": AUDIT_ROOT / "v103_phase2_stratified_q5c_objlike16384_scene0011_c0001_cap24576_qchunk16384_d4rt48mix_maskbalanced8",
    "scene0050_00": AUDIT_ROOT / "v103_phase2_stratified_q5c_objlike16384_scene0050_c0001_cap24576_qchunk16384_d4rt48mix_maskbalanced8",
}


def _project(path: Path | str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if p.parts and p.parts[0] == "Stream3D":
        return REPO_ROOT / p
    return STREAM3D_ROOT / p


def _rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return p.resolve().relative_to(REPO_ROOT).as_posix()
    except Exception:
        return p.as_posix()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return _rel(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields or ["schema_version"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key, "")) for key in fields})


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact_row(role: str, path: Path, required: bool = True) -> dict[str, Any]:
    return {
        "schema_version": "stream4d_v103_supp_r5_phaseR5_0_current_artifact_row_v1",
        "phase_id": PHASE_ID,
        "artifact_role": role,
        "path": _rel(path),
        "exists": path.exists(),
        "required": bool(required),
        "size_bytes": path.stat().st_size if path.exists() and path.is_file() else "",
        "uses_gt_for_prediction": False,
        "uses_future": False,
    }


def _gate_row(name: str, passed: bool, observed: Any, required: Any, repair: str) -> dict[str, Any]:
    return {
        "schema_version": "stream4d_v103_supp_r5_phaseR5_0_gate_row_v1",
        "phase_id": PHASE_ID,
        "gate_name": name,
        "pass": bool(passed),
        "observed": json.dumps(_jsonable(observed), sort_keys=True) if isinstance(observed, (dict, list, tuple)) else observed,
        "required": required,
        "repair_direction": repair,
        "uses_gt_for_prediction": False,
        "uses_future": False,
    }


def _metric_rows(phase6d_root: Path, baseline_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    phase6d_summary = _read_json(phase6d_root / "summary.json")
    d9: dict[str, Any] = {}
    metric_path = phase6d_root / "merge_metric_rows.csv"
    if metric_path.exists():
        df = pd.read_csv(metric_path)
        hit = df[df["variant_id"].astype(str) == "D9_affinity_merge_tau065_top1_broad_support_veto"]
        if not hit.empty:
            rec = hit.iloc[0].to_dict()
            d9 = {
                "variant_id": str(rec["variant_id"]),
                "MV_AP_window": float(rec["MV_AP_window"]),
                "MV_AP50_window": float(rec["MV_AP50_window"]),
                "MV_AP25_window": float(rec.get("MV_AP25_window", np.nan)),
                "ScoreFreeMatch50_window": float(rec["ScoreFreeMatch50_window"]),
                "same_frame_collision_count": int(rec["same_frame_collision_count"]),
                "pixel_collision_rate": float(rec["pixel_collision_rate"]),
                "missing_mask_raster_count": int(rec["missing_mask_raster_count"]),
                "dataset_split": str(rec.get("dataset_split", "")),
                "chunk_id": str(rec.get("chunk_id", "")),
            }
    if d9:
        rows.append(
            {
                "schema_version": "stream4d_v103_supp_r5_phaseR5_0_current_metric_row_v1",
                "phase_id": PHASE_ID,
                "metric_role": "current_phase6d_d9",
                "artifact_root": _rel(phase6d_root),
                "decision": phase6d_summary.get("decision", ""),
                **d9,
                "uses_gt_for_prediction": False,
                "uses_gt_for_eval": True,
                "uses_future": False,
            }
        )
    baseline_metric = baseline_root / "local_metric_rows.csv"
    if baseline_metric.exists():
        try:
            bdf = pd.read_csv(baseline_metric)
            for _, rec in bdf.head(10).iterrows():
                rows.append(
                    {
                        "schema_version": "stream4d_v103_supp_r5_phaseR5_0_current_metric_row_v1",
                        "phase_id": PHASE_ID,
                        "metric_role": "candidate_current_strong_baseline",
                        "artifact_root": _rel(baseline_root),
                        "variant_id": str(rec.get("variant_id", rec.get("method", ""))),
                        "MV_AP_window": rec.get("MV_AP_window", rec.get("AP", "")),
                        "MV_AP50_window": rec.get("MV_AP50_window", rec.get("AP50", "")),
                        "MV_AP25_window": rec.get("MV_AP25_window", rec.get("AP25", "")),
                        "uses_gt_for_prediction": False,
                        "uses_gt_for_eval": True,
                        "uses_future": False,
                    }
                )
        except Exception:
            pass
    return rows, d9


def build(args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.time()
    out = _project(args.output_root)
    out.mkdir(parents=True, exist_ok=True)

    phaseS1_root = _project(args.phaseS1_root)
    phaseS2_root = _project(args.phaseS2_root)
    phase6d_root = _project(args.phase6d_root)
    phaseS4_root = _project(args.phaseS4_root)
    baseline_root = _project(args.current_baseline_root)
    d4rt_roots = {
        "scene0011_00": _project(args.scene0011_d4rt_root),
        "scene0050_00": _project(args.scene0050_d4rt_root),
    }

    artifact_rows = [
        _artifact_row("plan_doc", PLAN_DOC),
        _artifact_row("canonical_evaluator", EVALUATOR),
        _artifact_row("phaseS1_summary", phaseS1_root / "summary.json"),
        _artifact_row("phaseS1_carrier_role_rows", phaseS1_root / "carrier_role_rows.parquet"),
        _artifact_row("phaseS2_summary", phaseS2_root / "summary.json"),
        _artifact_row("phase6d_summary", phase6d_root / "summary.json"),
        _artifact_row("phase6d_merge_metric_rows", phase6d_root / "merge_metric_rows.csv"),
        _artifact_row("phase6d_merge_selected_rows", phase6d_root / "merge_selected_rows.csv"),
        _artifact_row("phaseS4_summary", phaseS4_root / "summary.json"),
        _artifact_row("current_baseline_root", baseline_root, required=False),
    ]
    for scene, root in d4rt_roots.items():
        artifact_rows.extend(
            [
                _artifact_row(f"{scene}_d4rt_summary", root / "summary.json"),
                _artifact_row(f"{scene}_carrier_batch", root / "carrier_batch.npz"),
                _artifact_row(f"{scene}_carrier_sources", root / "carrier_sources.npz"),
                _artifact_row(f"{scene}_query_source_count_rows", root / "query_source_count_rows.csv"),
            ]
        )

    phaseS1_summary = _read_json(phaseS1_root / "summary.json")
    phaseS2_summary = _read_json(phaseS2_root / "summary.json")
    phase6d_summary = _read_json(phase6d_root / "summary.json")
    phaseS4_summary = _read_json(phaseS4_root / "summary.json")
    metric_rows, d9 = _metric_rows(phase6d_root, baseline_root)

    scope_rows = [
        {
            "schema_version": "stream4d_v103_supp_r5_phaseR5_0_current_scope_row_v1",
            "phase_id": PHASE_ID,
            "scope_role": "current_method_scope",
            "dataset_split": d9.get("dataset_split", "dev") if d9 else "unknown",
            "chunk_id": d9.get("chunk_id", "c0001") if d9 else "unknown",
            "frame_ids": "145..300 stride 5",
            "frame_count": 32,
            "scene_ids": "scene0011_00,scene0050_00",
            "scope_note": "current c0001 / first32-style dev subset; not full-dev or holdout",
            "uses_gt_for_prediction": False,
            "uses_future": False,
        },
        {
            "schema_version": "stream4d_v103_supp_r5_phaseR5_0_current_scope_row_v1",
            "phase_id": PHASE_ID,
            "scope_role": "clip_backfill_policy_from_user",
            "dataset_split": "n/a",
            "chunk_id": "n/a",
            "frame_ids": "n/a",
            "frame_count": "",
            "scene_ids": "",
            "scope_note": "mask-crop CLIP may be relaxed to a compact low-resolution feature map; high-resolution/high-dimensional dense pixel CLIP map remains disallowed for this plan.",
            "uses_gt_for_prediction": False,
            "uses_future": False,
        },
    ]

    missing_required = [row for row in artifact_rows if bool(row["required"]) and not bool(row["exists"])]
    gates = [
        _gate_row(
            "required_artifacts_exist",
            not missing_required,
            [row["artifact_role"] for row in missing_required],
            "[]",
            "Re-bind the missing current v103 artifact roots before R5-1.",
        ),
        _gate_row(
            "canonical_evaluator_is_v65",
            EVALUATOR.exists(),
            _rel(EVALUATOR),
            "Stream3D/tools/run_v65_scene_multiview_ap.py exists",
            "Repair metric contract before running R5 AP phases.",
        ),
        _gate_row(
            "phaseS1_passed",
            phaseS1_summary.get("decision") == "PASS_ENTER_PHASES2_ROLE_AWARE_AFFINITY",
            phaseS1_summary.get("decision", ""),
            "PASS_ENTER_PHASES2_ROLE_AWARE_AFFINITY",
            "Rerun or repair S1 multi-role carriers before support-weighted feature construction.",
        ),
        _gate_row(
            "phaseS2_passed",
            phaseS2_summary.get("decision") == "PASS_ENTER_PHASES3_SCAFFOLDED_MASK_GRAPH",
            phaseS2_summary.get("decision", ""),
            "PASS_ENTER_PHASES3_SCAFFOLDED_MASK_GRAPH",
            "Repair S2 feature arithmetic before R5-1.",
        ),
        _gate_row(
            "phase6d_d9_available",
            phase6d_summary.get("decision") == "PASS_PHASE6D_S3_STYLE_LOCAL_GATE" and bool(d9),
            {"decision": phase6d_summary.get("decision", ""), "d9": d9},
            "PASS_PHASE6D_S3_STYLE_LOCAL_GATE with D9 metric row",
            "Re-bind current Phase6d D9 root or rerun the c0001 local replay before R5-2/R5-4.",
        ),
        _gate_row(
            "phaseS4_history_no_go_recorded",
            phaseS4_summary.get("decision") == "NO_GO_REPAIR_PHASES4_POST_BIRTH_HISTORY_INHERITANCE",
            phaseS4_summary.get("decision", ""),
            "NO_GO_REPAIR_PHASES4_POST_BIRTH_HISTORY_INHERITANCE",
            "Use the current S4 control-bias result as boundary; do not run R5 history before local full-dev passes.",
        ),
        _gate_row(
            "uses_gt_for_prediction_false",
            True,
            False,
            "false",
            "Remove any GT dependency from prediction/provider paths.",
        ),
        _gate_row(
            "uses_future_false",
            True,
            False,
            "false",
            "Remove any future-frame dependency.",
        ),
    ]
    failures = [
        {
            "schema_version": "stream4d_v103_supp_r5_phaseR5_0_failure_row_v1",
            "phase_id": PHASE_ID,
            "failure_id": row["gate_name"],
            "severity": "blocking",
            "evidence": row["observed"],
            "repair_direction": row["repair_direction"],
        }
        for row in gates
        if not bool(row["pass"])
    ]

    _write_csv(out / "current_artifact_rows.csv", artifact_rows)
    _write_csv(out / "current_metric_rows.csv", metric_rows)
    _write_csv(out / "current_scope_rows.csv", scope_rows)
    _write_csv(out / "gate_rows.csv", gates)
    _write_csv(out / "failure_rows.csv", failures)

    phase_pass = not failures
    summary = {
        "schema_version": "stream4d_v103_supp_r5_phaseR5_0_summary_v1",
        "phase_id": PHASE_ID,
        "created_unix_time": time.time(),
        "decision": "PASS_ENTER_PHASE_R5_1_SUPPORT_WEIGHTED_FEATURE" if phase_pass else "NO_GO_REPAIR_PHASE_R5_0_FACT_LOCK",
        "phase_r5_0_pass": bool(phase_pass),
        "failure_count": int(len(failures)),
        "canonical_evaluator": _rel(EVALUATOR),
        "local_metric": "MV_AP_window",
        "scene_metric": "MV_AP_scene",
        "current_strong_baseline": "F2_v100_chunk32_overlap3_surfel_maskview_thr018_p2d2",
        "current_baseline_root": _rel(baseline_root),
        "current_phase6d_d9_root": _rel(phase6d_root),
        "current_phase6d_d9_variant_id": d9.get("variant_id", ""),
        "current_phase6d_d9_MV_AP_window": d9.get("MV_AP_window", ""),
        "current_phase6d_d9_MV_AP50_window": d9.get("MV_AP50_window", ""),
        "current_scope": "dev current c0001 first32-style subset, scenes scene0011_00 and scene0050_00",
        "phaseS1_root": _rel(phaseS1_root),
        "phaseS2_root": _rel(phaseS2_root),
        "phaseS4_root": _rel(phaseS4_root),
        "d4rt_roots": {scene: _rel(root) for scene, root in d4rt_roots.items()},
        "clip_backfill_policy": {
            "mask_level_sparse_table_default": True,
            "low_resolution_compact_feature_map_allowed": True,
            "high_resolution_high_dim_dense_pixel_map_allowed": False,
            "compact_dim_default": 64,
        },
        "runs_AP": False,
        "uses_gt_for_prediction": False,
        "uses_future": False,
        "runtime_sec": time.time() - t0,
        "outputs": {
            "summary": _rel(out / "summary.json"),
            "current_artifact_rows": _rel(out / "current_artifact_rows.csv"),
            "current_metric_rows": _rel(out / "current_metric_rows.csv"),
            "current_scope_rows": _rel(out / "current_scope_rows.csv"),
            "gate_rows": _rel(out / "gate_rows.csv"),
            "failure_rows": _rel(out / "failure_rows.csv"),
        },
        "truthfulness_note": "R5-0 is read-only. It does not generate predictions, run AP, tune thresholds, or claim v103 completion.",
    }
    _write_json(out / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Stream4D v103 R5 Phase R5-0 fact lock.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--phaseS1-root", default=str(DEFAULT_PHASES1_ROOT))
    parser.add_argument("--phaseS2-root", default=str(DEFAULT_PHASES2_ROOT))
    parser.add_argument("--phase6d-root", default=str(DEFAULT_PHASE6D_ROOT))
    parser.add_argument("--phaseS4-root", default=str(DEFAULT_PHASES4_ROOT))
    parser.add_argument("--current-baseline-root", default=str(DEFAULT_BASELINE_ROOT))
    parser.add_argument("--scene0011-d4rt-root", default=str(DEFAULT_D4RT_ROOT_BY_SCENE["scene0011_00"]))
    parser.add_argument("--scene0050-d4rt-root", default=str(DEFAULT_D4RT_ROOT_BY_SCENE["scene0050_00"]))
    args = parser.parse_args()
    summary = build(args)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if summary["phase_r5_0_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
