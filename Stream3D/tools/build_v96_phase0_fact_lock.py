#!/usr/bin/env python3
"""Build Stream4D v96 Phase0 fact lock and artifact boundary audit."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
OUT = ROOT / "outputs/audit/v96_phase0_fact_lock"
PHASE_ID = "v96_phase0_fact_lock"
RUN_ID = "v96_phase0_fact_lock"

PLAN = REPO_ROOT / "docs/stream4d_v96_d4rt_micro_primitive_affinity_field_plan.md"
V65_EVALUATOR = ROOT / "tools/run_v65_scene_multiview_ap.py"
V93_PHASE0 = ROOT / "outputs/audit/v93_phase0_contract/summary.json"
V93_BASELINES = ROOT / "outputs/audit/v93_phase0_contract/baseline_metric_rows.csv"
V94_PHASE0 = ROOT / "outputs/audit/v94_phase0_fact_lock/summary.json"
V94_CONTROLS = ROOT / "outputs/audit/v94_phase4_controls/summary.json"
V94_FINAL = ROOT / "outputs/audit/v94_phase8_dev_decision/summary.json"
V95_PHASE0 = ROOT / "outputs/audit/v95_phase0_fact_lock/summary.json"
V95_FINAL = ROOT / "outputs/audit/v95_phase8_dev_decision/summary.json"
V95_CORE_GT_DIAG = ROOT / "outputs/audit/v95_phase5_core_gt_alignment_diagnostic/summary.json"
D4RT_ADAPTER = ROOT / "stream4d/d4rt_adapter.py"
D4RT_ROOT = REPO_ROOT / "Open-d4rt"
D4RT_CONFIG = D4RT_ROOT / "checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml"
D4RT_CKPT = D4RT_ROOT / "checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt"

EXPECTED_AP_THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
EXPECTED_V95_NO_GO = "NO_GO_V95_LOCAL_MV_AP_WINDOW"


def _created_at() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return _rel(value)
    return value


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key, "")) for key in fields})


def _sha256(path: Path, *, max_hash_bytes: int = 1024 * 1024 * 1024) -> tuple[str, str]:
    if not path.exists() or not path.is_file():
        return "", "missing_or_not_file"
    size = path.stat().st_size
    if size > max_hash_bytes:
        return "", f"skipped_large_file_size_bytes_{size}"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest(), "computed"


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        try:
            return path.resolve().relative_to(REPO_ROOT).as_posix()
        except ValueError:
            return path.as_posix()


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return float(default)
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _artifact_row(name: str, path: Path, *, kind: str = "file", role: str = "input") -> dict[str, Any]:
    digest, digest_status = _sha256(path)
    stat = path.stat() if path.exists() else None
    return {
        "schema_version": "stream4d_v96_phase0_artifact_boundary_v1",
        "phase_id": PHASE_ID,
        "run_id": RUN_ID,
        "artifact_name": name,
        "artifact_path": _rel(path),
        "artifact_exists": path.exists(),
        "artifact_kind": kind,
        "artifact_role": role,
        "artifact_size_bytes": int(stat.st_size) if stat and path.is_file() else "",
        "artifact_sha256": digest,
        "artifact_sha256_status": digest_status,
        "uses_gt_for_prediction": False,
        "uses_future": False,
    }


def _copy_baseline_rows(created_at: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in _read_csv(V93_BASELINES):
        out: dict[str, Any] = dict(row)
        out["schema_version"] = "stream4d_v96_phase0_baseline_metric_v1"
        out["phase_id"] = PHASE_ID
        out["run_id"] = RUN_ID
        out["source_phase"] = "v93_phase0_contract"
        out["created_at"] = created_at
        rows.append(out)

    v95_final = _read_json(V95_FINAL)
    if v95_final:
        rows.append(
            {
                "schema_version": "stream4d_v96_phase0_baseline_metric_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "variant_id": v95_final.get("best_real_variant_id", ""),
                "variant_family": "v95_best_real",
                "source_version": "v95_phase8_dev_decision",
                "support_policy": "local_window_gt_projection",
                "metric_source": "tools.run_v65_scene_multiview_ap.SparseSceneIoU/_summarize_iou",
                "MV_AP_window": v95_final.get("best_real_MV_AP_window", ""),
                "MV_AP50_window": v95_final.get("best_real_MV_AP50_window", ""),
                "MV_AP25_window": v95_final.get("best_real_MV_AP25_window", ""),
                "ScoreFreeMatch50_window": v95_final.get("best_real_ScoreFreeMatch50_window", ""),
                "ScoreFreeMatch25_window": v95_final.get("best_real_ScoreFreeMatch25_window", ""),
                "same_frame_collision_count": v95_final.get("same_frame_collision_count", 0),
                "missing_mask_raster_count": v95_final.get("missing_mask_raster_count", 0),
                "uses_gt_for_prediction_count": int(_bool(v95_final.get("uses_gt_for_prediction"))),
                "uses_future_count": int(_bool(v95_final.get("uses_future"))),
                "notes": "v95 final best real is a locked failure boundary, not a v96 success.",
                "scene_id": "ALL_DEV",
                "split": "dev",
                "window_id": "ALL_DEV_WINDOWS",
                "source_artifact": _rel(V95_FINAL),
                "created_at": created_at,
            }
        )
    return rows


def _forbidden_rows(created_at: str) -> list[dict[str, Any]]:
    entries = [
        (
            "local_window_gt_projection",
            ROOT / "data/scannet/processed",
            "GT masks are evaluator-only support; not allowed for query selection, affinity, clustering, rendering, or scoring.",
        ),
        (
            "v65_evaluator_gt_path",
            V65_EVALUATOR,
            "SparseSceneIoU/_summarize_iou reads GT for final evaluation only.",
        ),
        (
            "stream3d_corrected_local_window",
            V93_BASELINES,
            "Stream3D corrected local-window row is diagnostic comparator only, not a v96 method row.",
        ),
        (
            "v95_core_gt_alignment_diagnostic",
            V95_CORE_GT_DIAG,
            "GT alignment diagnostic can explain failure but cannot select or score v96 method outputs.",
        ),
        (
            "holdout_or_local2history_before_dev_gate",
            ROOT / "outputs/audit",
            "Plan forbids holdout/local2history promotion unless dev gate passes.",
        ),
    ]
    rows: list[dict[str, Any]] = []
    for name, path, reason in entries:
        rows.append(
            {
                "schema_version": "stream4d_v96_phase0_forbidden_artifact_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "artifact_name": name,
                "artifact_path": _rel(path),
                "artifact_exists": path.exists(),
                "may_use_for_prediction": False,
                "may_use_for_query_selection": False,
                "may_use_for_variant_selection": False,
                "may_use_for_diagnostic": True,
                "reason": reason,
                "created_at": created_at,
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
    return rows


def _provenance_counts(rows: list[dict[str, Any]], summaries: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"uses_gt_for_prediction_count": 0, "uses_future_count": 0}
    for row in rows:
        counts["uses_gt_for_prediction_count"] += int(_bool(row.get("uses_gt_for_prediction")) or _bool(row.get("uses_gt_for_prediction_count")))
        counts["uses_future_count"] += int(_bool(row.get("uses_future")) or _bool(row.get("uses_future_count")))
    for summary in summaries:
        counts["uses_gt_for_prediction_count"] += int(_bool(summary.get("uses_gt_for_prediction")))
        counts["uses_future_count"] += int(_bool(summary.get("uses_future")))
    return counts


def run() -> dict[str, Any]:
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    created_at = _created_at()

    v93_phase0 = _read_json(V93_PHASE0)
    v94_phase0 = _read_json(V94_PHASE0)
    v94_controls = _read_json(V94_CONTROLS)
    v94_final = _read_json(V94_FINAL)
    v95_phase0 = _read_json(V95_PHASE0)
    v95_final = _read_json(V95_FINAL)
    evaluator_text = V65_EVALUATOR.read_text(encoding="utf-8") if V65_EVALUATOR.exists() else ""

    ap_thresholds = v95_phase0.get("AP_thresholds_actual", v94_phase0.get("AP_thresholds_actual", v93_phase0.get("AP_thresholds_actual", [])))
    formal_metric_source_eq_v65 = bool(
        V65_EVALUATOR.exists()
        and "SparseSceneIoU" in evaluator_text
        and "_summarize_iou" in evaluator_text
        and "AP_THRESHOLDS" in evaluator_text
        and _bool(v95_phase0.get("formal_metric_source_eq_v65", v93_phase0.get("formal_metric_source_eq_v65")))
    )
    local_support_policy = str(v95_phase0.get("local_support_policy", v93_phase0.get("local_support_policy", "")))
    baseline_rows = _copy_baseline_rows(created_at)
    provenance = _provenance_counts(baseline_rows, [v94_final, v95_final])
    required_ap = _num(v95_phase0.get("required_MV_AP_window"), 0.07399544580104074)
    required_ap50 = _num(v95_phase0.get("required_MV_AP50_window"), 0.19217992227130698)

    gate_rows = [
        {
            "schema_version": "stream4d_v96_phase0_gate_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "gate": "formal_metric_source_eq_v65",
            "pass": formal_metric_source_eq_v65,
            "observed": formal_metric_source_eq_v65,
            "required": True,
        },
        {
            "schema_version": "stream4d_v96_phase0_gate_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "gate": "AP_thresholds_actual",
            "pass": [float(x) for x in ap_thresholds] == EXPECTED_AP_THRESHOLDS,
            "observed": ap_thresholds,
            "required": EXPECTED_AP_THRESHOLDS,
        },
        {
            "schema_version": "stream4d_v96_phase0_gate_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "gate": "local_support_policy",
            "pass": local_support_policy == "local_window_gt_projection",
            "observed": local_support_policy,
            "required": "local_window_gt_projection",
        },
        {
            "schema_version": "stream4d_v96_phase0_gate_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "gate": "v95_final_decision_locked_no_go",
            "pass": v95_final.get("final_decision") == EXPECTED_V95_NO_GO,
            "observed": v95_final.get("final_decision", ""),
            "required": EXPECTED_V95_NO_GO,
        },
        {
            "schema_version": "stream4d_v96_phase0_gate_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "gate": "required_metrics_available",
            "pass": required_ap > 0 and required_ap50 > 0,
            "required_MV_AP_window": required_ap,
            "required_MV_AP50_window": required_ap50,
        },
        {
            "schema_version": "stream4d_v96_phase0_gate_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "gate": "no_prediction_gt_or_future_in_locked_method_rows",
            "pass": provenance["uses_gt_for_prediction_count"] == 0 and provenance["uses_future_count"] == 0,
            **provenance,
        },
    ]
    for row in gate_rows:
        row["uses_gt_for_prediction"] = False
        row["uses_future"] = False

    phase0_pass = all(bool(row.get("pass")) for row in gate_rows)
    artifact_rows = [
        _artifact_row("v96_plan", PLAN),
        _artifact_row("v65_scene_multiview_ap_evaluator", V65_EVALUATOR),
        _artifact_row("v93_phase0_summary", V93_PHASE0),
        _artifact_row("v93_phase0_baseline_rows", V93_BASELINES),
        _artifact_row("v94_phase0_summary", V94_PHASE0),
        _artifact_row("v94_control_summary", V94_CONTROLS),
        _artifact_row("v94_final_summary", V94_FINAL),
        _artifact_row("v95_phase0_summary", V95_PHASE0),
        _artifact_row("v95_final_summary", V95_FINAL),
        _artifact_row("d4rt_adapter", D4RT_ADAPTER),
        _artifact_row("opend4rt_root", D4RT_ROOT, kind="directory"),
        _artifact_row("opend4rt_32clip_config", D4RT_CONFIG),
        _artifact_row("opend4rt_32clip_checkpoint", D4RT_CKPT),
    ]
    forbidden_rows = _forbidden_rows(created_at)

    evaluator_contract = {
        "schema": "stream4d_v96_evaluator_contract_v1",
        "phase_id": PHASE_ID,
        "run_id": RUN_ID,
        "formal_metric_source": "tools.run_v65_scene_multiview_ap.SparseSceneIoU/_summarize_iou",
        "formal_metric_source_eq_v65": formal_metric_source_eq_v65,
        "AP_thresholds_actual": ap_thresholds,
        "AP_thresholds_expected": EXPECTED_AP_THRESHOLDS,
        "local_support_policy": local_support_policy,
        "support_policy_note": "local-window GT projection is evaluator support only; it is forbidden for v96 prediction/query/variant selection.",
        "score_protocol_note": "AP uses the v65 score-threshold precision envelope; score-free Match50/25 are auxiliary diagnostics.",
        "uses_gt_for_prediction": False,
        "uses_future": False,
    }

    stream3d_rows = [row for row in baseline_rows if row.get("variant_family") == "stream3d_local_diagnostic"]
    stream3d_row = stream3d_rows[0] if stream3d_rows else {}
    summary = {
        "schema": "stream4d_v96_phase0_fact_lock_summary_v1",
        "phase_id": PHASE_ID,
        "run_id": RUN_ID,
        "created_at": created_at,
        "decision": "PASS_V96_PHASE0_FACT_LOCK" if phase0_pass else "NO_GO_V96_PHASE0_FACT_LOCK",
        "formal_metric_source_eq_v65": formal_metric_source_eq_v65,
        "AP_thresholds_actual": ap_thresholds,
        "local_support_policy": local_support_policy,
        "B0_MV_AP_window": _num(v93_phase0.get("B0_MV_AP_window")),
        "B0_MV_AP50_window": _num(v93_phase0.get("B0_MV_AP50_window")),
        "best_locked_control": v94_controls.get("best_control_variant_id", "P3_C0_area_semantic_hybrid_score"),
        "best_control_MV_AP_window": _num(v94_controls.get("best_control_MV_AP_window"), _num(v95_phase0.get("best_control_MV_AP_window"))),
        "best_control_MV_AP50_window": _num(v94_controls.get("best_control_MV_AP50_window"), _num(v95_phase0.get("best_control_MV_AP50_window"))),
        "v91_best": "V91_AD4_sr2_adapt_sig8_b05_j075_r12",
        "v91_best_MV_AP_window": _num(v95_phase0.get("v91_best_MV_AP_window"), _num(v93_phase0.get("v91_best_MV_AP_window"))),
        "v91_best_MV_AP50_window": _num(v95_phase0.get("v91_best_MV_AP50_window"), _num(v93_phase0.get("v91_best_MV_AP50_window"))),
        "stream3d_corrected_local_window_MV_AP_window": _num(stream3d_row.get("MV_AP_window")),
        "stream3d_corrected_local_window_MV_AP50_window": _num(stream3d_row.get("MV_AP50_window")),
        "v95_best_real_variant_id": v95_final.get("best_real_variant_id", ""),
        "v95_best_real_MV_AP_window": _num(v95_final.get("best_real_MV_AP_window")),
        "v95_best_real_MV_AP50_window": _num(v95_final.get("best_real_MV_AP50_window")),
        "v95_final_decision": v95_final.get("final_decision", ""),
        "required_MV_AP_window": required_ap,
        "required_MV_AP50_window": required_ap50,
        "same_frame_collision_required": 0,
        "missing_mask_raster_required": 0,
        "uses_gt_for_prediction_count": provenance["uses_gt_for_prediction_count"],
        "uses_future_count": provenance["uses_future_count"],
        "d4rt_root_exists": D4RT_ROOT.exists(),
        "d4rt_config_exists": D4RT_CONFIG.exists(),
        "d4rt_ckpt_exists": D4RT_CKPT.exists(),
        "d4rt_ckpt_size_bytes": D4RT_CKPT.stat().st_size if D4RT_CKPT.exists() else 0,
        "d4rt_default_v63_path_note": "v63 defaults use ../Open-d4rt; v96 uses repo-local Open-d4rt unless overridden.",
        "phase0_gate_rows": _rel(OUT / "phase0_gate_rows.csv"),
        "baseline_metric_rows": _rel(OUT / "baseline_metric_rows.csv"),
        "evaluator_contract": _rel(OUT / "evaluator_contract.json"),
        "forbidden_artifact_rows": _rel(OUT / "forbidden_artifact_rows.csv"),
        "artifact_boundary_rows": _rel(OUT / "artifact_boundary_rows.csv"),
        "runtime_sec": float(time.time() - started),
        "uses_gt_for_prediction": False,
        "uses_future": False,
    }

    _write_csv(OUT / "baseline_metric_rows.csv", baseline_rows)
    _write_json(OUT / "evaluator_contract.json", evaluator_contract)
    _write_csv(OUT / "forbidden_artifact_rows.csv", forbidden_rows)
    _write_csv(OUT / "artifact_boundary_rows.csv", artifact_rows)
    _write_csv(OUT / "phase0_gate_rows.csv", gate_rows)
    _write_json(OUT / "summary.json", summary)
    return summary


def main() -> None:
    summary = run()
    print(json.dumps({"decision": summary["decision"], "output_root": _rel(OUT)}, sort_keys=True))


if __name__ == "__main__":
    main()
