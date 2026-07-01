from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


PHASE0_WINDOWS = ROOT / "outputs/audit/v90_phase0_mv_ap_contract/window_support_rows.csv"
PHASE1_AGG = ROOT / "outputs/audit/v90_phase1_variant_resurrection/mv_metric_aggregate_rows.csv"
PHASE3_AGG = ROOT / "outputs/audit/v90_phase3_carrier_supported_carving/mv_metric_aggregate_rows.csv"
PHASE4_ROOT = ROOT / "outputs/audit/v90_phase4_geo_semantic_witness_cover"
PHASE4_AGG = PHASE4_ROOT / "mv_metric_aggregate_rows.csv"
PHASE4_SUMMARY = PHASE4_ROOT / "summary.json"
PHASE8_OUT = ROOT / "outputs/audit/v90_phase8_dev_decision"
PHASE9_OUT = ROOT / "outputs/audit/v90_phase9_holdout_mv_ap"
PHASE11_OUT = ROOT / "outputs/audit/v90_phase11_casebook"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(_jsonable(row))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_delta(row: dict[str, Any], baseline: dict[str, Any], key: str) -> float:
    return _num(row.get(key)) - _num(baseline.get(key))


def _best(rows: list[dict[str, Any]], predicate) -> dict[str, Any]:
    candidates = [row for row in rows if predicate(row)]
    return max(candidates, key=lambda row: _num(row.get("mean_MV_AP_window")), default={})


def _holdout_splits_available() -> tuple[bool, dict[str, Any]]:
    rows = _read_csv(PHASE0_WINDOWS)
    split_counts: dict[str, int] = {}
    scene_split_counts: dict[str, int] = {}
    for row in rows:
        split = row.get("split", "")
        scene = row.get("scene_id", "")
        split_counts[split] = split_counts.get(split, 0) + 1
        scene_split_counts[f"{split}:{scene}"] = scene_split_counts.get(f"{split}:{scene}", 0) + 1
    has_holdout = any(split and split != "dev" for split in split_counts)
    return has_holdout, {"split_counts": split_counts, "scene_split_counts": scene_split_counts}


def run() -> dict[str, Any]:
    t0 = time.time()
    PHASE8_OUT.mkdir(parents=True, exist_ok=True)
    PHASE9_OUT.mkdir(parents=True, exist_ok=True)
    PHASE11_OUT.mkdir(parents=True, exist_ok=True)

    phase1_rows = _read_csv(PHASE1_AGG)
    phase3_rows = _read_csv(PHASE3_AGG)
    phase4_rows = _read_csv(PHASE4_AGG)
    phase4_summary = _load_json(PHASE4_SUMMARY)
    b0 = next((row for row in phase1_rows if row.get("variant_id") == "B0_local_only"), {})
    stream3d = next((row for row in phase1_rows if row.get("variant_id") == "S3D_L1_local_merged_masks"), {})
    if not stream3d:
        stream3d = next((row for row in phase1_rows if "S3D" in row.get("variant_id", "") or "Stream3D" in row.get("variant_id", "")), {})
    control_candidates = [
        row
        for row in phase1_rows + phase3_rows + phase4_rows
        if row.get("variant_id", "").startswith("C")
    ]
    best_control = max(control_candidates, key=lambda row: _num(row.get("mean_MV_AP_window")), default={})
    real_candidates = [
        row
        for row in phase4_rows
        if row.get("variant_id", "").startswith("W")
        and not _bool(row.get("uses_gt_for_prediction"))
        and not _bool(row.get("uses_future"))
        and _num(row.get("risk_penalty_mean"), 1.0) <= _num(phase4_summary.get("B0_risk_penalty_mean_proxy"), 1.0) + 1e-12
    ]
    best_real = max(real_candidates, key=lambda row: _num(row.get("mean_MV_AP_window")), default={})

    progress_gate = bool(best_real) and (
        _num(best_real.get("mean_MV_AP_window")) >= _num(b0.get("mean_MV_AP_window")) + 0.01
        and _num(best_real.get("mean_MV_AP50_window")) >= _num(b0.get("mean_MV_AP50_window")) + 0.02
        and _num(best_real.get("mean_MV_AP_window")) >= _num(best_control.get("mean_MV_AP_window")) + 0.005
        and _num(best_real.get("mean_MV_AP50_window")) >= _num(best_control.get("mean_MV_AP50_window")) + 0.01
        and int(float(best_real.get("same_frame_collision_count", 0) or 0)) == 0
        and not _bool(best_real.get("uses_gt_for_prediction"))
        and not _bool(best_real.get("uses_future"))
    )
    strong_gate = bool(best_real) and (
        _num(best_real.get("mean_MV_AP_window")) >= 0.05
        and _num(best_real.get("mean_MV_AP50_window")) >= 0.12
        and _num(best_real.get("mean_MV_AP_window")) >= _num(best_control.get("mean_MV_AP_window")) + 0.015
        and _num(best_real.get("mean_MV_AP50_window")) >= _num(best_control.get("mean_MV_AP50_window")) + 0.03
    )

    frozen_config = {
        "schema": "stream4d_v90_frozen_local_readout_config_v1",
        "selected_variant": best_real.get("variant_id", ""),
        "parent_phase": "v90_phase4_geo_semantic_witness_cover",
        "selection_reason": "best risk-safe real variant passing Phase8 progress and strong local gates on dev",
        "native_support_rows": _rel(ROOT / "outputs/audit/v90_phase3_v82_full_support/native_carrier_support_rows.csv"),
        "semantic_feature_rows": [
            _rel(ROOT / "outputs/audit/v71_semantic_features/mask_feature_rows.csv"),
            _rel(ROOT / "outputs/audit/v81_dino_feature_json_scene0011_scene0050/mask_feature_rows.csv"),
        ],
        "risk_gate": {
            "b0_risk_penalty_mean_proxy": phase4_summary.get("B0_risk_penalty_mean_proxy"),
            "selected_risk_penalty_mean": best_real.get("risk_penalty_mean"),
        },
        "uses_gt_for_prediction": False,
        "uses_future": False,
        "holdout_selection_allowed": False,
    }
    frozen_text = json.dumps(_jsonable(frozen_config), indent=2, sort_keys=True) + "\n"
    frozen_config["config_payload_sha256"] = _sha256_text(frozen_text)

    dev_metric_rows = []
    for row in [b0, best_control, best_real, stream3d]:
        if row:
            dev_metric_rows.append(row)
    control_comparison_rows = [
        {
            "comparison": "best_real_minus_B0",
            "best_real_variant": best_real.get("variant_id", ""),
            "baseline_variant": b0.get("variant_id", "B0_local_only"),
            "MV_AP_window_gap": _num(best_real.get("mean_MV_AP_window")) - _num(b0.get("mean_MV_AP_window")),
            "MV_AP50_window_gap": _num(best_real.get("mean_MV_AP50_window")) - _num(b0.get("mean_MV_AP50_window")),
        },
        {
            "comparison": "best_real_minus_best_control",
            "best_real_variant": best_real.get("variant_id", ""),
            "baseline_variant": best_control.get("variant_id", ""),
            "MV_AP_window_gap": _num(best_real.get("mean_MV_AP_window")) - _num(best_control.get("mean_MV_AP_window")),
            "MV_AP50_window_gap": _num(best_real.get("mean_MV_AP50_window")) - _num(best_control.get("mean_MV_AP50_window")),
        },
        {
            "comparison": "best_real_minus_Stream3D_diagnostic",
            "best_real_variant": best_real.get("variant_id", ""),
            "baseline_variant": stream3d.get("variant_id", ""),
            "MV_AP_window_gap": _num(best_real.get("mean_MV_AP_window")) - _num(stream3d.get("mean_MV_AP_window")),
            "MV_AP50_window_gap": _num(best_real.get("mean_MV_AP50_window")) - _num(stream3d.get("mean_MV_AP50_window")),
            "note": "Stream3D remains diagnostic upper/reference, not required by v90 dev gate.",
        },
    ]
    decision_matrix_rows = [
        {"gate": "Phase8_progress_gate", "pass": progress_gate},
        {"gate": "Phase8_strong_local_gate", "pass": strong_gate},
        {"gate": "same_frame_collision_count_zero", "pass": int(float(best_real.get("same_frame_collision_count", 0) or 0)) == 0},
        {"gate": "uses_gt_for_prediction_false", "pass": not _bool(best_real.get("uses_gt_for_prediction"))},
        {"gate": "uses_future_false", "pass": not _bool(best_real.get("uses_future"))},
        {"gate": "risk_safe_vs_B0_proxy", "pass": _num(best_real.get("risk_penalty_mean"), 1.0) <= _num(phase4_summary.get("B0_risk_penalty_mean_proxy"), 1.0) + 1e-12},
    ]
    _write_csv(PHASE8_OUT / "dev_metric_rows.csv", dev_metric_rows)
    _write_csv(PHASE8_OUT / "control_comparison_rows.csv", control_comparison_rows)
    _write_csv(PHASE8_OUT / "decision_matrix_rows.csv", decision_matrix_rows)
    frozen_config_path = PHASE8_OUT / "frozen_candidate_config.json"
    _write_json(frozen_config_path, frozen_config)
    frozen_file_sha256 = _sha256(frozen_config_path)
    phase8_summary = {
        "phase": "v90_phase8_dev_decision",
        "schema": "stream4d_v90_phase8_dev_decision_v1",
        "phase8_progress_gate": progress_gate,
        "phase8_strong_local_gate": strong_gate,
        "best_real_variant": best_real.get("variant_id", ""),
        "best_real_metrics": best_real,
        "best_control_variant": best_control.get("variant_id", ""),
        "best_control_metrics": best_control,
        "B0_local_only": b0,
        "Stream3D_local_diagnostic": stream3d,
        "frozen_config": _rel(PHASE8_OUT / "frozen_candidate_config.json"),
        "frozen_config_payload_sha256": frozen_config["config_payload_sha256"],
        "frozen_config_file_sha256": frozen_file_sha256,
        "outputs": {
            "dev_metric_rows": _rel(PHASE8_OUT / "dev_metric_rows.csv"),
            "control_comparison_rows": _rel(PHASE8_OUT / "control_comparison_rows.csv"),
            "decision_matrix_rows": _rel(PHASE8_OUT / "decision_matrix_rows.csv"),
            "frozen_candidate_config": _rel(PHASE8_OUT / "frozen_candidate_config.json"),
        },
    }
    _write_json(PHASE8_OUT / "summary.json", phase8_summary)

    has_holdout, holdout_diag = _holdout_splits_available()
    existing_phase9_summary = _load_json(PHASE9_OUT / "summary.json")
    holdout_run_executed = bool(existing_phase9_summary.get("holdout_run_executed"))
    if holdout_run_executed:
        phase9_summary = existing_phase9_summary
        holdout_blocked = bool(phase9_summary.get("blocked"))
    else:
        holdout_blocked = not has_holdout
        phase9_summary = {
            "phase": "v90_phase9_holdout_mv_ap",
            "schema": "stream4d_v90_phase9_holdout_mv_ap_blocked_v1",
            "phase9_pass": False,
            "holdout_run_executed": False,
            "blocked": holdout_blocked,
            "blocked_reason": "NO_HOLDOUT_SPLIT_AVAILABLE_IN_V90_WINDOW_SUPPORT_ROWS" if holdout_blocked else "",
            "holdout_diagnostic": holdout_diag,
            "frozen_config": _rel(PHASE8_OUT / "frozen_candidate_config.json"),
            "config_payload_sha256": frozen_config["config_payload_sha256"],
            "config_file_sha256": frozen_file_sha256,
            "note": "Current v90 Phase0 support rows contain only dev splits for scene0011_00 and scene0050_00; using the same rows as holdout would leak dev selection.",
        }
        _write_json(PHASE9_OUT / "blocked_reason.json", phase9_summary if holdout_blocked else {})
        _write_json(
            PHASE9_OUT / "config_sha256_check.json",
            {
                "config_payload_sha256": frozen_config["config_payload_sha256"],
                "config_file_sha256": frozen_file_sha256,
                "config_path": _rel(PHASE8_OUT / "frozen_candidate_config.json"),
                "holdout_not_run": True,
            },
        )
        _write_json(PHASE9_OUT / "summary.json", phase9_summary)

    holdout_pass = bool(phase9_summary.get("phase9_pass"))
    if progress_gate and strong_gate and holdout_pass:
        final_label = "GO_LOCAL_MV_AP_WINDOW"
    elif progress_gate:
        final_label = "DIAGNOSTIC_PROGRESS_LOCAL_ONLY"
    else:
        final_label = "NO_GO_EXTENT_BLOCKER"
    holdout_b0 = phase9_summary.get("holdout_B0", {}) if isinstance(phase9_summary.get("holdout_B0"), dict) else {}
    holdout_best_real = (
        phase9_summary.get("holdout_best_real", {}) if isinstance(phase9_summary.get("holdout_best_real"), dict) else {}
    )
    holdout_best_control = (
        phase9_summary.get("holdout_best_control", {}) if isinstance(phase9_summary.get("holdout_best_control"), dict) else {}
    )
    holdout_status = (
        "pass"
        if holdout_pass
        else "failed"
        if holdout_run_executed and not phase9_summary.get("blocked")
        else phase9_summary.get("blocked_reason", "not_run")
    )
    if holdout_pass:
        main_conclusion = "Dev and frozen holdout local MV_AP_window pass; v90 local readout may claim GO_LOCAL_MV_AP_WINDOW, while MV_AP_scene/local2history remains pending or diagnostic."
        next_action = "Proceed to Phase10 MV_AP_scene/local2history using the frozen Phase9 local outputs without changing local mask extents."
    elif holdout_run_executed and not phase9_summary.get("blocked"):
        main_conclusion = "Dev local MV_AP_window improved, but the single frozen holdout run did not clear the B0-margin gates, so v90 remains diagnostic local progress only."
        next_action = "Stop v90 method claim and open v91 design work; do not retune or rerun v90 holdout as success. Focus v91 on improving holdout B0 margin while preserving no-GT/no-future/no-collision gates."
    else:
        main_conclusion = "Dev local MV_AP_window improved with risk-safe geo-semantic witness cover plus carving, but no non-leaky holdout result is available, so this is diagnostic local progress only."
        next_action = "Create a true v90 holdout split or fresh-scene holdout inputs, then run the frozen W9b config exactly once before any method success claim or MV_AP_scene/local2history work."
    final_decision = {
        "phase": "v90_phase11_casebook",
        "schema": "stream4d_v90_phase11_final_decision_v2",
        "decision_label": final_label,
        "dev_progress_gate": progress_gate,
        "dev_strong_local_gate": strong_gate,
        "holdout_run_executed": holdout_run_executed,
        "holdout_phase9_pass": holdout_pass,
        "holdout_status": holdout_status,
        "holdout_blocked_reason": phase9_summary.get("blocked_reason", ""),
        "holdout_gate": phase9_summary.get("gate", {}),
        "holdout_best_real_metrics": holdout_best_real,
        "holdout_B0_local_only": holdout_b0,
        "holdout_best_control_metrics": holdout_best_control,
        "holdout_margin_vs_B0": {
            "MV_AP_window": _metric_delta(holdout_best_real, holdout_b0, "mean_MV_AP_window"),
            "MV_AP50_window": _metric_delta(holdout_best_real, holdout_b0, "mean_MV_AP50_window"),
        },
        "holdout_margin_vs_best_control": {
            "MV_AP_window": _metric_delta(holdout_best_real, holdout_best_control, "mean_MV_AP_window"),
            "MV_AP50_window": _metric_delta(holdout_best_real, holdout_best_control, "mean_MV_AP50_window"),
        },
        "holdout_summary": _rel(PHASE9_OUT / "summary.json"),
        "holdout_provenance_caveat": phase9_summary.get("provenance_caveat", ""),
        "best_real_variant": best_real.get("variant_id", ""),
        "best_real_metrics": best_real,
        "best_control_variant": best_control.get("variant_id", ""),
        "B0_local_only": b0,
        "Stream3D_local_diagnostic": stream3d,
        "main_conclusion": main_conclusion,
        "next_action": next_action,
        "runtime_sec": time.time() - t0,
    }
    success_rows = [
        {
            "case_type": "dev_success",
            "variant_id": best_real.get("variant_id", ""),
            "MV_AP_window": best_real.get("mean_MV_AP_window", ""),
            "MV_AP50_window": best_real.get("mean_MV_AP50_window", ""),
            "B0_gap": _num(best_real.get("mean_MV_AP_window")) - _num(b0.get("mean_MV_AP_window")),
            "control_gap": _num(best_real.get("mean_MV_AP_window")) - _num(best_control.get("mean_MV_AP_window")),
            "risk_penalty_mean": best_real.get("risk_penalty_mean", ""),
            "witness_coverage_rate": best_real.get("witness_coverage_rate", ""),
        }
    ]
    failure_rows = []
    if holdout_run_executed and not holdout_pass:
        failure_rows.append(
            {
                "case_type": "holdout_failed",
                "best_real_variant": holdout_best_real.get("variant_id", ""),
                "B0_variant": holdout_b0.get("variant_id", "B0_local_only"),
                "best_control_variant": holdout_best_control.get("variant_id", ""),
                "MV_AP_window": holdout_best_real.get("mean_MV_AP_window", ""),
                "B0_MV_AP_window": holdout_b0.get("mean_MV_AP_window", ""),
                "best_control_MV_AP_window": holdout_best_control.get("mean_MV_AP_window", ""),
                "MV_AP_window_gap_vs_B0": _metric_delta(holdout_best_real, holdout_b0, "mean_MV_AP_window"),
                "MV_AP_window_required_gap_vs_B0": 0.008,
                "MV_AP50_window_gap_vs_B0": _metric_delta(holdout_best_real, holdout_b0, "mean_MV_AP50_window"),
                "MV_AP50_window_required_gap_vs_B0": 0.015,
                "gate": json.dumps(phase9_summary.get("gate", {}), sort_keys=True),
                "next_action": final_decision["next_action"],
            }
        )
    else:
        failure_rows.append(
            {
                "case_type": "holdout_blocked",
                "blocked_reason": phase9_summary.get("blocked_reason", ""),
                "evidence": json.dumps(holdout_diag, sort_keys=True),
                "next_action": final_decision["next_action"],
            }
        )
    failure_rows.append(
        {
            "case_type": "stream3d_gap_remaining",
            "best_real_variant": best_real.get("variant_id", ""),
            "stream3d_variant": stream3d.get("variant_id", ""),
            "MV_AP_window_gap": _num(best_real.get("mean_MV_AP_window")) - _num(stream3d.get("mean_MV_AP_window")),
            "note": "Remaining gap is evidence that local readout is improved but not yet Stream3D-level.",
        }
    )
    method_summary = "\n".join(
        [
            "# v90 Method vs Control Summary",
            "",
            f"Decision: {final_label}",
            f"Best risk-safe dev variant: {best_real.get('variant_id', '')}",
            f"Best real MV_AP_window: {best_real.get('mean_MV_AP_window', '')}",
            f"Best real MV_AP50_window: {best_real.get('mean_MV_AP50_window', '')}",
            f"B0 MV_AP_window: {b0.get('mean_MV_AP_window', '')}",
            f"Best control MV_AP_window: {best_control.get('mean_MV_AP_window', '')}",
            f"Holdout status: {holdout_status}",
            f"Holdout best real MV_AP_window: {holdout_best_real.get('mean_MV_AP_window', '')}",
            f"Holdout B0 MV_AP_window: {holdout_b0.get('mean_MV_AP_window', '')}",
            f"Holdout best control MV_AP_window: {holdout_best_control.get('mean_MV_AP_window', '')}",
            "",
        ]
    )
    theory_update = "\n".join(
        [
            "# v90 Theory Update",
            "",
            "Phase4 supports H4: D4RT geo-semantic witnesses can replace simple whole-mask adapter selection on dev MV_AP_window.",
            "Phase3 support-only/carving alone was insufficient; witness cover changed masklet selection, and carving then improved selected tubes.",
            "The frozen holdout run is the boundary for v90 method claims; if it fails, v90 remains diagnostic even when dev improves.",
            "Phase10 MV_AP_scene/local2history is not a method-success path until dev and holdout local MV_AP_window both pass.",
            "",
        ]
    )
    _write_json(PHASE11_OUT / "final_decision.json", final_decision)
    _write_csv(PHASE11_OUT / "success_casebook_rows.csv", success_rows)
    _write_csv(PHASE11_OUT / "failure_casebook_rows.csv", failure_rows)
    _write_text(PHASE11_OUT / "method_vs_control_summary.md", method_summary)
    _write_text(PHASE11_OUT / "theory_update.md", theory_update)
    sha_paths = [
        PHASE8_OUT / "dev_metric_rows.csv",
        PHASE8_OUT / "control_comparison_rows.csv",
        PHASE8_OUT / "decision_matrix_rows.csv",
        PHASE8_OUT / "frozen_candidate_config.json",
        PHASE8_OUT / "summary.json",
        PHASE9_OUT / "config_sha256_check.json",
        PHASE9_OUT / "holdout_metric_rows.csv",
        PHASE9_OUT / "holdout_metric_aggregate_rows.csv",
        PHASE9_OUT / "holdout_casebook_rows.csv",
        PHASE9_OUT / "holdout_iou_matrix_rows.csv",
        PHASE9_OUT / "summary.json",
        PHASE11_OUT / "final_decision.json",
        PHASE11_OUT / "success_casebook_rows.csv",
        PHASE11_OUT / "failure_casebook_rows.csv",
        PHASE11_OUT / "method_vs_control_summary.md",
        PHASE11_OUT / "theory_update.md",
    ]
    for out_dir in [PHASE8_OUT, PHASE9_OUT, PHASE11_OUT]:
        _write_json(out_dir / "SHA256SUMS.json", {_rel(path): _sha256(path) for path in sha_paths if path.exists() and path.parent == out_dir})
    print(json.dumps(_jsonable({"phase8": phase8_summary, "phase9": phase9_summary, "phase11": final_decision}), indent=2, sort_keys=True), flush=True)
    return final_decision


if __name__ == "__main__":
    run()
