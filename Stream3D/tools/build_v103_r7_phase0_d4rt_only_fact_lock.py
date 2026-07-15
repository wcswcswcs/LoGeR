#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
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

PHASE_ID = "v103_r7_phase0_d4rt_only_fact_lock"
PLAN_DOC = REPO_ROOT / "docs/stream4d_v103_r7_d4rt_only_support_conditioned_affinity_l2h_plan.md"
EVALUATOR = STREAM3D_ROOT / "tools/run_v65_scene_multiview_ap.py"
DEFAULT_OUT = AUDIT_ROOT / "v103_r7_phase0_d4rt_only_fact_lock"
DEFAULT_R6_FACT_LOCK_ROOT = AUDIT_ROOT / "v103_supp_r6_phase0_fact_lock"
DEFAULT_R6_FINAL_ROOT = AUDIT_ROOT / "v103_supp_r6_final_decision"
DEFAULT_R6_FEATURE_ROOT = AUDIT_ROOT / "v103_supp_r6_phase2_support_conditioned_feature"
DEFAULT_R6_DIAG_ROOT = AUDIT_ROOT / "v103_supp_r6_phase6_gt_coverage_inconsistency"
DEFAULT_PHASES1_ROOT = AUDIT_ROOT / "v103_supp_phaseS1_multirole_carriers"
DEFAULT_D4RT_ROOT_BY_SCENE = {
    "scene0011_00": AUDIT_ROOT / "v103_phase2_stratified_q5c_objlike16384_scene0011_c0001_cap24576_qchunk16384_d4rt48mix_maskbalanced8",
    "scene0050_00": AUDIT_ROOT / "v103_phase2_stratified_q5c_objlike16384_scene0050_c0001_cap24576_qchunk16384_d4rt48mix_maskbalanced8",
}

FORBIDDEN_TOKENS = (
    "da3",
    "da3-giant",
    "3dgs",
    "gaussian",
    "phase9n",
    "phase9b",
    "phase9c",
    "phase9d",
    "da3_pair",
)


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


def _sha256(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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


def _num(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None or value == "":
            return default
        out = float(value)
        if not np.isfinite(out):
            return default
        return out
    except Exception:
        return default


def _forbidden_hits(value: Any) -> list[str]:
    text = json.dumps(_jsonable(value), sort_keys=True).lower() if not isinstance(value, str) else value.lower()
    return sorted({token for token in FORBIDDEN_TOKENS if token in text})


def _artifact_row(role: str, path: Path, *, required: bool = True, note: str = "") -> dict[str, Any]:
    exists = path.exists()
    row_count: int | str = ""
    if exists and path.is_file() and path.suffix.lower() in {".csv", ".parquet"}:
        try:
            if path.suffix.lower() == ".csv":
                row_count = max(sum(1 for _ in path.open("r", encoding="utf-8")) - 1, 0)
            else:
                row_count = int(pd.read_parquet(path).shape[0])
        except Exception:
            row_count = ""
    hits = _forbidden_hits(_rel(path))
    return {
        "schema_version": "stream4d_v103_r7_phase0_artifact_row_v1",
        "phase_id": PHASE_ID,
        "artifact_role": role,
        "path": _rel(path),
        "exists": bool(exists),
        "required": bool(required),
        "is_file": path.is_file() if exists else False,
        "is_dir": path.is_dir() if exists else False,
        "size_bytes": path.stat().st_size if exists and path.is_file() else "",
        "sha256": _sha256(path),
        "row_count": row_count,
        "forbidden_token_hits_in_path": ";".join(hits),
        "note": note,
        "uses_gt_for_prediction": False,
        "uses_future": False,
    }


def _gate_row(name: str, passed: bool, observed: Any, required: Any, repair: str) -> dict[str, Any]:
    return {
        "schema_version": "stream4d_v103_r7_phase0_gate_row_v1",
        "phase_id": PHASE_ID,
        "gate_name": name,
        "pass": bool(passed),
        "observed": observed,
        "required": required,
        "repair_direction": repair,
        "uses_gt_for_prediction": False,
        "uses_future": False,
    }


def _failure_row(failure_id: str, evidence: Any, repair: str) -> dict[str, Any]:
    return {
        "schema_version": "stream4d_v103_r7_phase0_failure_row_v1",
        "phase_id": PHASE_ID,
        "failure_id": failure_id,
        "severity": "blocking",
        "evidence": json.dumps(_jsonable(evidence), sort_keys=True) if isinstance(evidence, (dict, list, tuple)) else evidence,
        "repair_direction": repair,
        "uses_gt_for_prediction": False,
        "uses_future": False,
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.time()
    out = _project(args.output_root)
    if out.exists() and any(out.iterdir()) and not args.force:
        raise SystemExit(f"output root already exists and is non-empty; pass --force: {_rel(out)}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "last_command.txt").write_text(" ".join([sys.executable, str(Path(__file__)), *sys.argv[1:]]) + "\n", encoding="utf-8")

    r6_fact_root = _project(args.r6_fact_lock_root)
    r6_final_root = _project(args.r6_final_root)
    r6_feature_root = _project(args.r6_feature_root)
    r6_diag_root = _project(args.r6_diag_root)
    phases1_root = _project(args.phaseS1_root)
    d4rt_roots = {
        "scene0011_00": _project(args.scene0011_d4rt_root),
        "scene0050_00": _project(args.scene0050_d4rt_root),
    }

    r6_fact = _read_json(r6_fact_root / "summary.json")
    r6_final = _read_json(r6_final_root / "summary.json")
    r6_feature = _read_json(r6_feature_root / "summary.json")
    r6_diag = _read_json(r6_diag_root / "summary.json")
    d4rt_summaries = {scene: _read_json(root / "summary.json") for scene, root in d4rt_roots.items()}

    method_inputs = {
        "plan_doc": _rel(PLAN_DOC),
        "canonical_evaluator": _rel(EVALUATOR),
        "r6_fact_summary": _rel(r6_fact_root / "summary.json"),
        "r6_final_summary": _rel(r6_final_root / "summary.json"),
        "r6_feature_summary": _rel(r6_feature_root / "summary.json"),
        "r6_diag_summary": _rel(r6_diag_root / "summary.json"),
        "phaseS1_carrier_roles": _rel(phases1_root / "carrier_role_rows.parquet"),
        "d4rt_scene0011_summary": _rel(d4rt_roots["scene0011_00"] / "summary.json"),
        "d4rt_scene0050_summary": _rel(d4rt_roots["scene0050_00"] / "summary.json"),
    }
    forbidden_hits = _forbidden_hits(method_inputs)
    da3_used = bool(forbidden_hits)
    gs_used = bool({"3dgs", "gaussian"} & set(forbidden_hits))

    artifact_rows = [
        _artifact_row("plan_doc", PLAN_DOC),
        _artifact_row("canonical_evaluator", EVALUATOR),
        _artifact_row("r6_fact_summary_reference_only", r6_fact_root / "summary.json"),
        _artifact_row("r6_final_summary_reference_only", r6_final_root / "summary.json"),
        _artifact_row("r6_feature_summary", r6_feature_root / "summary.json"),
        _artifact_row("r6_feature_rows", r6_feature_root / "role_feature_summary_rows.csv"),
        _artifact_row("r6_diag_summary", r6_diag_root / "summary.json"),
        _artifact_row("r6_diag_gt_coverage_summary", r6_diag_root / "gt_object_coverage_summary_rows.csv"),
        _artifact_row("phaseS1_carrier_role_rows", phases1_root / "carrier_role_rows.parquet"),
        _artifact_row("scene0011_d4rt_summary", d4rt_roots["scene0011_00"] / "summary.json"),
        _artifact_row("scene0011_d4rt_carrier_batch", d4rt_roots["scene0011_00"] / "carrier_batch.npz"),
        _artifact_row("scene0050_d4rt_summary", d4rt_roots["scene0050_00"] / "summary.json"),
        _artifact_row("scene0050_d4rt_carrier_batch", d4rt_roots["scene0050_00"] / "carrier_batch.npz"),
        _artifact_row("last_command", out / "last_command.txt", required=False),
    ]

    selected_scenes = list(r6_fact.get("selected_scene_ids", ["scene0011_00", "scene0050_00"]))
    selected_chunks = list(r6_fact.get("selected_chunk_ids", ["c0001"]))
    current_replay = _num(r6_fact.get("current_replay_MV_AP_window"))
    current_replay_ap50 = _num(r6_fact.get("current_replay_MV_AP50_window"))
    current_d9 = _num(r6_fact.get("current_locked_D9_MV_AP_window"))
    current_d9_ap50 = _num(r6_fact.get("current_locked_D9_MV_AP50_window"))
    s_hit = _num(r6_fact.get("current_S_support_hit_rate"))
    a_hit = _num(r6_fact.get("current_A_anchor_hit_rate"))
    role_path = phases1_root / "carrier_role_rows.parquet"
    role_available = role_path.exists()
    d4rt_available = all((root / "carrier_batch.npz").exists() and (root / "summary.json").exists() for root in d4rt_roots.values())
    formal_metric = EVALUATOR.exists() and "run_v65_scene_multiview_ap.py" in EVALUATOR.as_posix()
    selected_scope_match = selected_scenes == ["scene0011_00", "scene0050_00"] and selected_chunks == ["c0001"]
    uses_gt_for_prediction = False
    uses_future = False

    metric_contract_rows = [
        {
            "schema_version": "stream4d_v103_r7_phase0_metric_contract_row_v1",
            "phase_id": PHASE_ID,
            "canonical_evaluator_path": _rel(EVALUATOR),
            "formal_metric_source_eq_v65": bool(formal_metric),
            "local_metric": "MV_AP_window",
            "scene_metric": "MV_AP_scene",
            "AP_thresholds": "0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90",
            "uses_gt_for_prediction": False,
            "uses_future": False,
        }
    ]
    baseline_rows = [
        {
            "schema_version": "stream4d_v103_r7_phase0_baseline_row_v1",
            "phase_id": PHASE_ID,
            "metric_role": "current_replay_D0_reference_from_R6_fact_lock",
            "variant_id": r6_fact.get("current_replay_variant_id", "D0_f2_original_replay"),
            "MV_AP_window": current_replay,
            "MV_AP50_window": current_replay_ap50,
            "uses_gt_for_prediction": False,
            "uses_future": False,
        },
        {
            "schema_version": "stream4d_v103_r7_phase0_baseline_row_v1",
            "phase_id": PHASE_ID,
            "metric_role": "current_locked_D9_reference_from_R6_fact_lock",
            "variant_id": r6_fact.get("current_locked_D9_variant_id", "D9_affinity_merge_tau065_top1_broad_support_veto"),
            "MV_AP_window": current_d9,
            "MV_AP50_window": current_d9_ap50,
            "same_frame_collision_count": r6_fact.get("current_locked_D9_same_frame_collision_count", ""),
            "missing_mask_raster_count": r6_fact.get("current_locked_D9_missing_mask_raster_count", ""),
            "uses_gt_for_prediction": False,
            "uses_future": False,
        },
    ]
    role_rows = [
        {
            "schema_version": "stream4d_v103_r7_phase0_role_artifact_row_v1",
            "phase_id": PHASE_ID,
            "role_artifact": "phaseS1_multirole_carriers",
            "path": _rel(role_path),
            "exists": role_available,
            "D4RT_only": True,
            "uses_gt_for_prediction": False,
            "uses_future": False,
        },
        {
            "schema_version": "stream4d_v103_r7_phase0_role_artifact_row_v1",
            "phase_id": PHASE_ID,
            "role_artifact": "r6_support_coverage_reference",
            "path": _rel(r6_diag_root / "gt_object_coverage_summary_rows.csv"),
            "exists": (r6_diag_root / "gt_object_coverage_summary_rows.csv").exists(),
            "S_support_hit_rate": s_hit,
            "A_anchor_hit_rate": a_hit,
            "uses_gt_for_prediction": False,
            "uses_gt_for_eval": True,
            "uses_future": False,
        },
    ]
    scope_rows = [
        {
            "schema_version": "stream4d_v103_r7_phase0_selected_scope_row_v1",
            "phase_id": PHASE_ID,
            "selected_scope": "current c0001 / first32-style dev subset",
            "selected_scenes": ",".join(selected_scenes),
            "selected_chunk_id": ",".join(selected_chunks),
            "d4rt_frame_ids_scene0011": ",".join(map(str, d4rt_summaries.get("scene0011_00", {}).get("frame_ids", [])[:5])),
            "d4rt_frame_ids_scene0050": ",".join(map(str, d4rt_summaries.get("scene0050_00", {}).get("frame_ids", [])[:5])),
            "uses_gt_for_prediction": False,
            "uses_future": False,
        }
    ]

    gates = [
        _gate_row("formal_metric_source_eq_v65", formal_metric, _rel(EVALUATOR), "run_v65_scene_multiview_ap.py", "Repair evaluator path before R7."),
        _gate_row("selected_scope_matches_current_c0001", selected_scope_match, {"scenes": selected_scenes, "chunks": selected_chunks}, "scene0011_00,scene0050_00 / c0001", "Repair scope before R7-1."),
        _gate_row("D4RT_role_artifacts_available", role_available and d4rt_available, {"role": role_available, "d4rt": d4rt_available}, True, "Repair D4RT/S1 artifact paths."),
        _gate_row("S_support_coverage_available", s_hit is not None, s_hit, "available", "Regenerate R6/R7 GT coverage diagnostic."),
        _gate_row("A_anchor_available", a_hit is not None, a_hit, "available", "Regenerate R6/R7 role coverage diagnostic."),
        _gate_row("current_replay_and_D9_baselines_readable", current_replay is not None and current_d9 is not None, {"replay": current_replay, "D9": current_d9}, "both readable", "Repair R6 fact lock baseline fields."),
        _gate_row("DA3_USED_false", not da3_used, forbidden_hits, "[]", "Remove DA3/phase9n inputs from R7 method path."),
        _gate_row("DA3_ROWS_LOADED_false", not da3_used, forbidden_hits, "[]", "Do not read DA3 row artifacts in R7."),
        _gate_row("GS_USED_false", not gs_used, forbidden_hits, "[]", "Remove 3DGS/Gaussian inputs from R7."),
        _gate_row("uses_gt_for_prediction_false", not uses_gt_for_prediction, uses_gt_for_prediction, False, "Keep GT diagnostic-only."),
        _gate_row("uses_future_false", not uses_future, uses_future, False, "Do not read future chunk memory."),
    ]
    failures = []
    for row in gates:
        if not bool(row["pass"]):
            fid = str(row["gate_name"])
            if fid in {"DA3_USED_false", "DA3_ROWS_LOADED_false", "GS_USED_false"}:
                fid = "R7_DA3_ARTIFACT_LEAKAGE"
            failures.append(_failure_row(fid, row["observed"], row["repair_direction"]))
    missing_required = [row for row in artifact_rows if row["required"] and not row["exists"]]
    if missing_required:
        failures.append(_failure_row("R7_REQUIRED_ARTIFACT_MISSING", missing_required, "Repair required R7 input paths before R7-1."))

    phase_pass = not failures
    summary = {
        "schema_version": "stream4d_v103_r7_phase0_summary_v1",
        "phase": "R7-0",
        "phase_id": PHASE_ID,
        "phase_pass": bool(phase_pass),
        "decision": "PASS_ENTER_R7_1_EXACT_EDGE_ATTRIBUTION" if phase_pass else "NO_GO_R7_0_REPAIR_D4RT_ONLY_FACT_LOCK",
        "canonical_evaluator_path": _rel(EVALUATOR),
        "formal_metric_source_eq_v65": bool(formal_metric),
        "AP_thresholds": [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90],
        "selected_scope": "current c0001 / first32-style dev subset",
        "selected_scenes": selected_scenes,
        "selected_chunk_id": ",".join(selected_chunks),
        "current_replay_MV_AP_window": current_replay,
        "current_replay_MV_AP50_window": current_replay_ap50,
        "current_locked_D9_MV_AP_window": current_d9,
        "current_locked_D9_MV_AP50_window": current_d9_ap50,
        "current_locked_D9_collision_count": r6_fact.get("current_locked_D9_same_frame_collision_count", ""),
        "current_locked_D9_missing_mask_raster_count": r6_fact.get("current_locked_D9_missing_mask_raster_count", ""),
        "S_support_hit_rate": s_hit,
        "A_anchor_hit_rate": a_hit,
        "current_accepted_diff_gt_edge_count": r6_fact.get("current_accepted_diff_gt_edge_count", ""),
        "r6_final_decision": r6_final.get("decision", ""),
        "r6_feature_root": _rel(r6_feature_root),
        "r6_feature_decision": r6_feature.get("decision", ""),
        "r6_diag_decision": r6_diag.get("decision", ""),
        "DA3_USED": bool(da3_used),
        "DA3_ROWS_LOADED": bool(da3_used),
        "GS_USED": bool(gs_used),
        "forbidden_token_hits": forbidden_hits,
        "uses_gt_for_prediction": False,
        "uses_gt_for_eval": True,
        "uses_future": False,
        "failure_count": len(failures),
        "runtime_sec": time.time() - t0,
        "outputs": {
            "summary": _rel(out / "summary.json"),
            "gate_rows": _rel(out / "gate_rows.csv"),
            "failure_rows": _rel(out / "failure_rows.csv"),
            "variant_rows": _rel(out / "variant_rows.csv"),
            "metric_rows": _rel(out / "metric_rows.csv"),
            "control_rows": _rel(out / "control_rows.csv"),
            "artifact_rows": _rel(out / "artifact_rows.csv"),
            "metric_contract_rows": _rel(out / "metric_contract_rows.csv"),
            "selected_scope_rows": _rel(out / "selected_scope_rows.csv"),
            "baseline_rows": _rel(out / "baseline_rows.csv"),
            "role_artifact_rows": _rel(out / "role_artifact_rows.csv"),
        },
        "truthfulness_note": (
            "R7-0 is read-only and locks D4RT-only method inputs. It uses R6 fact/final summaries as reference metrics, "
            "but intentionally does not read the old phase9n/DA3 scaffold rows."
        ),
    }
    variant_rows = [
        {
            "schema_version": "stream4d_v103_r7_phase0_variant_row_v1",
            "phase_id": PHASE_ID,
            "variant_id": "R7_0_fact_lock_only",
            "variant_role": "read_only_fact_lock",
            "runs_AP": False,
            "uses_gt_for_prediction": False,
            "uses_future": False,
        }
    ]
    control_rows = [
        {
            "schema_version": "stream4d_v103_r7_phase0_control_row_v1",
            "phase_id": PHASE_ID,
            "control_id": "DA3_GS_leakage_scan",
            "pass": not da3_used and not gs_used,
            "forbidden_token_hits": ";".join(forbidden_hits),
            "uses_gt_for_prediction": False,
            "uses_future": False,
        }
    ]

    _write_json(out / "summary.json", summary)
    _write_csv(out / "gate_rows.csv", gates)
    _write_csv(out / "failure_rows.csv", failures)
    _write_csv(out / "variant_rows.csv", variant_rows)
    _write_csv(out / "metric_rows.csv", baseline_rows)
    _write_csv(out / "control_rows.csv", control_rows)
    _write_csv(out / "artifact_rows.csv", artifact_rows)
    _write_csv(out / "metric_contract_rows.csv", metric_contract_rows)
    _write_csv(out / "selected_scope_rows.csv", scope_rows)
    _write_csv(out / "baseline_rows.csv", baseline_rows)
    _write_csv(out / "role_artifact_rows.csv", role_rows)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Stream4D v103 R7-0 D4RT-only fact lock.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--r6-fact-lock-root", default=str(DEFAULT_R6_FACT_LOCK_ROOT))
    parser.add_argument("--r6-final-root", default=str(DEFAULT_R6_FINAL_ROOT))
    parser.add_argument("--r6-feature-root", default=str(DEFAULT_R6_FEATURE_ROOT))
    parser.add_argument("--r6-diag-root", default=str(DEFAULT_R6_DIAG_ROOT))
    parser.add_argument("--phaseS1-root", default=str(DEFAULT_PHASES1_ROOT))
    parser.add_argument("--scene0011-d4rt-root", default=str(DEFAULT_D4RT_ROOT_BY_SCENE["scene0011_00"]))
    parser.add_argument("--scene0050-d4rt-root", default=str(DEFAULT_D4RT_ROOT_BY_SCENE["scene0050_00"]))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    summary = build(args)
    return 0 if summary["phase_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
