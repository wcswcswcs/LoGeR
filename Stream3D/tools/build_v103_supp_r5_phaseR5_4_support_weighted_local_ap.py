#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
AUDIT_ROOT = STREAM3D_ROOT / "outputs" / "audit"

PHASE_ID = "v103_supp_r5_phaseR5_4_support_weighted_local_ap"
DEFAULT_OUT = AUDIT_ROOT / "v103_supp_r5_support_weighted_local_ap_diag"
DEFAULT_R5_FEATURE_ROOT = AUDIT_ROOT / "v103_supp_r5_support_weighted_affinity"
DEFAULT_PHASE6D_SCRIPT = STREAM3D_ROOT / "tools/build_v103_phase6d_f2_skeleton_affinity_merge.py"
DEFAULT_F2_ROOT = AUDIT_ROOT / "v100_phase2c_overlap3_local_repair"
DEFAULT_PHASE9N_ROOT = AUDIT_ROOT / "v103_phase9n_da3_bridge_pair_fused_phase4_suppS1_d4rt48mix_s5repair_r3_allclean"
DEFAULT_SUBSET_BASELINE = AUDIT_ROOT / "v103_phase6_baseline_subset_contract_r1/baseline_subset_metric_rows.csv"
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


def _variant_list(summary: dict[str, Any], requested: str, max_variants: int) -> list[str]:
    if requested.strip():
        variants = [v.strip() for v in requested.split(",") if v.strip()]
    else:
        variants = [str(v) for v in summary.get("passing_support_variants", [])]
    return variants[: int(max_variants)]


def _as_tensor(value: Any, dtype: torch.dtype | None = None) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(dtype=dtype) if dtype is not None else value
    return torch.as_tensor(value, dtype=dtype)


def _write_phase5_like_roots(feature_payload: dict[str, Any], variants: list[str], out: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant_id in variants:
        variant_root = out / "phase5_like_features" / variant_id
        for scene, scene_payload in feature_payload["scenes"].items():
            scene_dir = variant_root / scene
            scene_dir.mkdir(parents=True, exist_ok=True)
            variant_payload = scene_payload["variants"][variant_id]
            support_count = _as_tensor(variant_payload["all_support_count"], torch.int64)
            payload = {
                "schema_version": "stream4d_v103_supp_r5_phaseR5_4_phase5_like_mask_level_feature_v1",
                "phase_id": PHASE_ID,
                "source_phase_id": feature_payload.get("phase_id", ""),
                "source_r5_feature_variant_id": variant_id,
                "feature": _as_tensor(variant_payload["mask_feature"], torch.float16),
                "mask_frame": _as_tensor(scene_payload["mask_frame"], torch.int64),
                "mask_label": _as_tensor(scene_payload["mask_label"], torch.int64),
                "mask_is_broad": _as_tensor(scene_payload["mask_is_broad"], torch.bool),
                "mask_is_object_like": _as_tensor(scene_payload["mask_is_object_like"], torch.bool),
                "support_count": support_count,
                "uses_gt": False,
                "uses_future": False,
            }
            path = scene_dir / "mask_level_feature.pt"
            torch.save(payload, path)
            rows.append(
                {
                    "schema_version": "stream4d_v103_supp_r5_phaseR5_4_phase5_like_feature_row_v1",
                    "phase_id": PHASE_ID,
                    "variant_id": variant_id,
                    "scene_id": scene,
                    "path": _rel(path),
                    "exists": path.exists(),
                    "size_bytes": path.stat().st_size if path.exists() else 0,
                    "uses_gt": False,
                    "uses_future": False,
                }
            )
    return rows


def _run_phase6d_for_variant(args: argparse.Namespace, out: Path, variant_id: str, phase5_like_root: Path) -> tuple[int, str]:
    run_root = out / "phase6d_runs" / variant_id
    cmd = [
        sys.executable,
        str(_project(args.phase6d_script)),
        "--output-root",
        str(run_root),
        "--f2-root",
        str(_project(args.f2_root)),
        "--phase5-root",
        str(phase5_like_root),
        "--phase9n-root",
        str(_project(args.phase9n_root)),
        "--scene0011-phase2-root",
        str(_project(args.scene0011_d4rt_root)),
        "--scene0050-phase2-root",
        str(_project(args.scene0050_d4rt_root)),
        "--subset-baseline-rows",
        str(_project(args.subset_baseline_rows)),
        "--dataset-split",
        str(args.dataset_split),
        "--chunk-id",
        str(args.chunk_id),
        "--cupy-device-id",
        str(args.cupy_device_id),
    ]
    (run_root.parent).mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log_path = out / f"phase6d_{variant_id}.log"
    log_path.write_text(proc.stdout, encoding="utf-8")
    return int(proc.returncode), str(log_path)


def _collect_metrics(out: Path, variants: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    metric_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    for feature_variant in variants:
        run_root = out / "phase6d_runs" / feature_variant
        metric_path = run_root / "merge_metric_rows.csv"
        summary_path = run_root / "summary.json"
        summary = _read_json(summary_path)
        if not metric_path.exists():
            failure_rows.append(
                {
                    "schema_version": "stream4d_v103_supp_r5_phaseR5_4_failure_row_v1",
                    "phase_id": PHASE_ID,
                    "r5_feature_variant_id": feature_variant,
                    "blocker": "PHASE6D_METRIC_ROWS_MISSING",
                    "detail": _rel(metric_path),
                    "repair_direction": "Inspect phase6d log and rerun this feature variant.",
                }
            )
            continue
        df = pd.read_csv(metric_path)
        for _, rec in df.iterrows():
            row = {
                "schema_version": "stream4d_v103_supp_r5_phaseR5_4_variant_metric_row_v1",
                "phase_id": PHASE_ID,
                "r5_feature_variant_id": feature_variant,
                "phase6d_variant_id": str(rec["variant_id"]),
                "MV_AP_window": float(rec["MV_AP_window"]),
                "MV_AP50_window": float(rec["MV_AP50_window"]),
                "MV_AP25_window": float(rec.get("MV_AP25_window", 0.0)),
                "ScoreFreeMatch50_window": float(rec["ScoreFreeMatch50_window"]),
                "same_frame_collision_count": int(rec["same_frame_collision_count"]),
                "pixel_collision_rate": float(rec["pixel_collision_rate"]),
                "missing_mask_raster_count": int(rec["missing_mask_raster_count"]),
                "accepted_edge_count": int(rec.get("accepted_merge_count", 0)),
                "candidate_edge_count": int(rec.get("candidate_edge_count", 0)),
                "dataset_split": str(rec.get("dataset_split", "")),
                "chunk_id": str(rec.get("chunk_id", "")),
                "uses_gt_for_prediction": bool(rec.get("uses_gt_for_prediction", False)),
                "uses_future": bool(rec.get("uses_future", False)),
                "source_phase6d_root": _rel(run_root),
            }
            metric_rows.append(row)
            if str(rec["variant_id"]).startswith("R"):
                control_rows.append(row | {"schema_version": "stream4d_v103_supp_r5_phaseR5_4_control_metric_row_v1"})
        d9 = df[df["variant_id"].astype(str) == "D9_affinity_merge_tau065_top1_broad_support_veto"]
        replay = df[df["variant_id"].astype(str) == "D0_f2_original_replay"]
        shuffled = df[df["variant_id"].astype(str) == "R5_shuffled_affinity_merge_tau065_top1_broad_support_veto_control"]
        if d9.empty or replay.empty or shuffled.empty:
            failure_rows.append(
                {
                    "schema_version": "stream4d_v103_supp_r5_phaseR5_4_failure_row_v1",
                    "phase_id": PHASE_ID,
                    "r5_feature_variant_id": feature_variant,
                    "blocker": "REQUIRED_D9_REPLAY_OR_SHUFFLED_ROW_MISSING",
                    "detail": f"d9={not d9.empty} replay={not replay.empty} shuffled={not shuffled.empty}",
                    "repair_direction": "Repair Phase6d variant registry or metric output before R5-4 gate.",
                }
            )
            continue
        d9r = d9.iloc[0]
        rr = replay.iloc[0]
        sr = shuffled.iloc[0]
        specs = [
            ("MV_AP_window_ge_replay_plus_0p005", float(d9r["MV_AP_window"]), float(rr["MV_AP_window"]) + 0.005),
            ("MV_AP50_window_ge_replay_plus_0p010", float(d9r["MV_AP50_window"]), float(rr["MV_AP50_window"]) + 0.010),
            ("real_minus_shuffled_MV_AP_window_ge_0p003", float(d9r["MV_AP_window"]) - float(sr["MV_AP_window"]), 0.003),
            ("same_frame_collision_count_eq_0", int(d9r["same_frame_collision_count"]), 0),
            ("pixel_collision_rate_eq_0", float(d9r["pixel_collision_rate"]), 0.0),
            ("missing_mask_raster_count_eq_0", int(d9r["missing_mask_raster_count"]), 0),
        ]
        for gate_name, observed, required in specs:
            if gate_name.endswith("_eq_0"):
                passed = observed == required
            else:
                passed = observed >= required
            gate_rows.append(
                {
                    "schema_version": "stream4d_v103_supp_r5_phaseR5_4_gate_row_v1",
                    "phase_id": PHASE_ID,
                    "r5_feature_variant_id": feature_variant,
                    "phase6d_variant_id": "D9_affinity_merge_tau065_top1_broad_support_veto",
                    "gate_name": gate_name,
                    "pass": bool(passed),
                    "observed": observed,
                    "required": required,
                    "phase6d_decision": summary.get("decision", ""),
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )
    for row in gate_rows:
        if not bool(row["pass"]):
            failure_rows.append(
                {
                    "schema_version": "stream4d_v103_supp_r5_phaseR5_4_failure_row_v1",
                    "phase_id": PHASE_ID,
                    "r5_feature_variant_id": row["r5_feature_variant_id"],
                    "blocker": row["gate_name"],
                    "detail": f"observed={row['observed']} required={row['required']}",
                    "repair_direction": "Do not promote this feature variant; inspect support feature/edge attribution and controls.",
                }
            )
    return metric_rows, control_rows, gate_rows, failure_rows


def build(args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.time()
    out = _project(args.output_root)
    if out.exists() and any(out.iterdir()) and not args.force:
        raise SystemExit(f"output root already exists and is non-empty; pass --force: {_rel(out)}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "last_command.txt").write_text(" ".join([sys.executable, str(Path(__file__)), *sys.argv[1:]]) + "\n", encoding="utf-8")

    feature_root = _project(args.r5_feature_root)
    feature_summary = _read_json(feature_root / "summary.json")
    if not bool(feature_summary.get("phase_r5_1_pass", False)):
        raise RuntimeError(f"R5-1 feature root has not passed: {feature_root / 'summary.json'}")
    variants = _variant_list(feature_summary, str(args.variants), int(args.max_variants))
    if not variants:
        raise RuntimeError("no R5 feature variants selected for local AP diagnostic")
    feature_payload = torch.load(feature_root / "role_mask_level_feature.pt", map_location="cpu", weights_only=False)
    phase5_rows = _write_phase5_like_roots(feature_payload, variants, out)
    run_rows: list[dict[str, Any]] = []
    for variant_id in variants:
        rc, log_path = _run_phase6d_for_variant(args, out, variant_id, out / "phase5_like_features" / variant_id)
        run_rows.append(
            {
                "schema_version": "stream4d_v103_supp_r5_phaseR5_4_phase6d_run_row_v1",
                "phase_id": PHASE_ID,
                "r5_feature_variant_id": variant_id,
                "returncode": int(rc),
                "log_path": _rel(log_path),
                "phase6d_run_root": _rel(out / "phase6d_runs" / variant_id),
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
    metric_rows, control_rows, gate_rows, failure_rows = _collect_metrics(out, variants)
    phase6d_failures = [
        {
            "schema_version": "stream4d_v103_supp_r5_phaseR5_4_failure_row_v1",
            "phase_id": PHASE_ID,
            "r5_feature_variant_id": row["r5_feature_variant_id"],
            "blocker": "PHASE6D_RUN_FAILED",
            "detail": f"returncode={row['returncode']} log={row['log_path']}",
            "repair_direction": "Inspect the Phase6d log and repair the feature adapter or evaluator input.",
        }
        for row in run_rows
        if int(row["returncode"]) != 0
    ]
    failure_rows = phase6d_failures + failure_rows
    passing_variants = sorted({str(row["r5_feature_variant_id"]) for row in gate_rows if bool(row["pass"])})
    fully_passing_variants: list[str] = []
    for variant_id in variants:
        sub = [row for row in gate_rows if str(row["r5_feature_variant_id"]) == variant_id]
        if sub and all(bool(row["pass"]) for row in sub):
            fully_passing_variants.append(variant_id)

    _write_csv(out / "phase5_like_feature_rows.csv", phase5_rows)
    _write_csv(out / "phase6d_run_rows.csv", run_rows)
    _write_csv(out / "variant_metric_rows.csv", metric_rows)
    _write_csv(out / "control_metric_rows.csv", control_rows)
    _write_csv(out / "variant_gate_rows.csv", gate_rows)
    _write_csv(out / "failure_rows.csv", failure_rows)

    artifact_rows = [
        {
            "schema_version": "stream4d_v103_supp_r5_phaseR5_4_artifact_row_v1",
            "phase_id": PHASE_ID,
            "artifact_role": role,
            "path": _rel(path),
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else 0,
            "uses_gt_for_prediction": False,
            "uses_future": False,
        }
        for role, path in [
            ("phase5_like_feature_rows", out / "phase5_like_feature_rows.csv"),
            ("phase6d_run_rows", out / "phase6d_run_rows.csv"),
            ("variant_metric_rows", out / "variant_metric_rows.csv"),
            ("control_metric_rows", out / "control_metric_rows.csv"),
            ("variant_gate_rows", out / "variant_gate_rows.csv"),
            ("failure_rows", out / "failure_rows.csv"),
            ("last_command", out / "last_command.txt"),
        ]
    ]
    _write_csv(out / "artifact_rows.csv", artifact_rows)
    phase_pass = bool(fully_passing_variants) and not phase6d_failures
    summary = {
        "schema_version": "stream4d_v103_supp_r5_phaseR5_4_summary_v1",
        "phase_id": PHASE_ID,
        "created_unix_time": time.time(),
        "decision": "PASS_R5_4_DIAGNOSTIC_LOCAL_AP_SIGNAL" if phase_pass else "NO_GO_R5_4_SUPPORT_WEIGHTED_LOCAL_AP_DIAGNOSTIC",
        "phase_r5_4_diag_pass": bool(phase_pass),
        "failure_count": int(len(failure_rows)),
        "r5_feature_root": _rel(feature_root),
        "tested_r5_feature_variants": variants,
        "fully_passing_r5_feature_variants": fully_passing_variants,
        "partially_passing_gate_variants": passing_variants,
        "phase6d_script": _rel(_project(args.phase6d_script)),
        "f2_root": _rel(_project(args.f2_root)),
        "phase9n_root": _rel(_project(args.phase9n_root)),
        "metric_scope": f"dev {args.chunk_id} current subset; diagnostic local AP only",
        "runs_AP": True,
        "uses_gt_for_prediction": False,
        "uses_gt_for_eval": True,
        "uses_future": False,
        "runtime_sec": time.time() - t0,
        "outputs": {
            "summary": _rel(out / "summary.json"),
            "phase5_like_feature_rows": _rel(out / "phase5_like_feature_rows.csv"),
            "phase6d_run_rows": _rel(out / "phase6d_run_rows.csv"),
            "variant_metric_rows": _rel(out / "variant_metric_rows.csv"),
            "control_metric_rows": _rel(out / "control_metric_rows.csv"),
            "variant_gate_rows": _rel(out / "variant_gate_rows.csv"),
            "failure_rows": _rel(out / "failure_rows.csv"),
            "artifact_rows": _rel(out / "artifact_rows.csv"),
        },
        "truthfulness_note": (
            "This is an R5-4 local-AP diagnostic wrapper over existing Phase6d/v65 evaluator. "
            "It does not complete R5-2 edge attribution, R5-3 GT diagnostics, full-dev, holdout, or local2history. "
            "It tests whether R5-1 support-weighted mask features produce immediate c0001 subset local AP signal."
        ),
    }
    _write_json(out / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="R5 support-weighted local AP diagnostic using the existing Phase6d evaluator.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--r5-feature-root", default=str(DEFAULT_R5_FEATURE_ROOT))
    parser.add_argument("--variants", default="")
    parser.add_argument("--max-variants", type=int, default=4)
    parser.add_argument("--phase6d-script", default=str(DEFAULT_PHASE6D_SCRIPT))
    parser.add_argument("--f2-root", default=str(DEFAULT_F2_ROOT))
    parser.add_argument("--phase9n-root", default=str(DEFAULT_PHASE9N_ROOT))
    parser.add_argument("--scene0011-d4rt-root", default=str(DEFAULT_D4RT_ROOT_BY_SCENE["scene0011_00"]))
    parser.add_argument("--scene0050-d4rt-root", default=str(DEFAULT_D4RT_ROOT_BY_SCENE["scene0050_00"]))
    parser.add_argument("--subset-baseline-rows", default=str(DEFAULT_SUBSET_BASELINE))
    parser.add_argument("--dataset-split", default="dev")
    parser.add_argument("--chunk-id", default="c0001")
    parser.add_argument("--cupy-device-id", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    summary = build(args)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if summary["phase_r5_4_diag_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
