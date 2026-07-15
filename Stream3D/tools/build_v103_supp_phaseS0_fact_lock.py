#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
AUDIT_ROOT = STREAM3D_ROOT / "outputs" / "audit"

PHASE_ID = "v103_supp_phaseS0_fact_lock"
DEFAULT_OUT = AUDIT_ROOT / "v103_supp_phaseS0_fact_lock"
ADJUSTED_METHOD_DOC = REPO_ROOT / "docs/stream4d_v103_stream4d_method_thinking_training_free_primitive_affinity_field_adjusted.md"
SUPPLEMENT_PLAN_DOC = REPO_ROOT / "docs/stream4d_v103_supplement_multirole_carrier_affinity_field_plan.md"


DEFAULT_ARTIFACTS = {
    "phase0_contract": AUDIT_ROOT / "v103_phase0_contract",
    "phase1_gpu_data_model": AUDIT_ROOT / "v103_phase1_gpu_data_model_parity",
    "phase2_scene0011_d4rt48mix_maskbalanced8": AUDIT_ROOT
    / "v103_phase2_stratified_q5c_objlike16384_scene0011_c0001_cap24576_qchunk16384_d4rt48mix_maskbalanced8",
    "phase2_scene0050_d4rt48mix_maskbalanced8": AUDIT_ROOT
    / "v103_phase2_stratified_q5c_objlike16384_scene0050_c0001_cap24576_qchunk16384_d4rt48mix_maskbalanced8",
    "phase3_all_d4rt48mix_maskbalanced8_e5": AUDIT_ROOT
    / "v103_phase3_carrier_reliability_filter_all_d4rt48mix_maskbalanced8_competing_repair5",
    "phase4_affinity_feature_e5": AUDIT_ROOT / "v103_phase4_primitive_affinity_all_d4rt48mix_maskbalanced8_e5_r1",
    "phase4_arithmetic_audit_e5": AUDIT_ROOT / "v103_phase4_affinity_correctness_d4rt48mix_maskbalanced8_e5_r1",
    "phase5_mask_pooling_e5": AUDIT_ROOT / "v103_phase5_local_pooling_d4rt48mix_maskbalanced8_e5_r1",
    "phase6_raw_birth_e5": AUDIT_ROOT / "v103_phase6_local_clustering_d4rt48mix_maskbalanced8_e5_r1",
    "phase6_skeleton_best_current": AUDIT_ROOT
    / "v103_phase6d_f2_skeleton_affinity_merge_phase9n_r5_no_broad_or_rel070_broad_support_veto",
    "phase7_history_readiness_e5": AUDIT_ROOT
    / "v103_phase7_causal_history_token_readiness_r11_all_d4rt48mix_maskbalanced8_e5",
    "phase8_history_injection_real_lam010": AUDIT_ROOT / "v103_phase8_history_aware_clustering_r2_d4rt48mix_e5_lam010",
    "phase8_history_injection_shuffle_lam010": AUDIT_ROOT / "v103_phase8_history_aware_clustering_r5_d4rt48mix_e5_lam010_shuffle",
}


def _rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return p.resolve().relative_to(REPO_ROOT).as_posix()
    except Exception:
        return p.as_posix()


def _project(path: Path | str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if p.parts and p.parts[0] == "Stream3D":
        return REPO_ROOT / p
    return STREAM3D_ROOT / p


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return _rel(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
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


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _boolish(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _floatish(value: Any, default: float = 0.0) -> float:
    try:
        if value == "" or value is None:
            return default
        return float(value)
    except Exception:
        return default


def _sha256_first(path: Path, max_bytes: int = 64 * 1024 * 1024) -> str:
    if not path.exists() or not path.is_file():
        return ""
    h = hashlib.sha256()
    remaining = int(max_bytes)
    with path.open("rb") as f:
        while remaining > 0:
            chunk = f.read(min(1024 * 1024, remaining))
            if not chunk:
                break
            h.update(chunk)
            remaining -= len(chunk)
    return h.hexdigest()


def _artifact_file_rows(artifact_id: str, root: Path, required: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rel_name in required:
        path = root / rel_name
        exists = path.exists()
        rows.append(
            {
                "schema_version": "stream4d_v103_supp_phaseS0_input_artifact_row_v1",
                "phase_id": PHASE_ID,
                "artifact_id": artifact_id,
                "file_role": rel_name,
                "path": _rel(path),
                "exists": bool(exists),
                "size_bytes": path.stat().st_size if exists and path.is_file() else "",
                "sha256_first64m": _sha256_first(path) if exists and path.is_file() else "",
                "checksum_policy": "first64m_for_large_files; existence/size locked for huge tensor caches",
            }
        )
    return rows


def _summary_at(root: Path) -> dict[str, Any]:
    return _read_json(root / "summary.json")


def _decision(summary: dict[str, Any]) -> str:
    return str(summary.get("decision", summary.get("phase_decision", "")))


def _gate(gate_id: str, passed: bool, observed: Any, required: Any, repair_direction: str = "") -> dict[str, Any]:
    return {
        "schema_version": "stream4d_v103_supp_phaseS0_gate_row_v1",
        "phase_id": PHASE_ID,
        "gate_id": gate_id,
        "pass": bool(passed),
        "observed": observed,
        "required": required,
        "repair_direction": repair_direction,
    }


def _copy_baseline_rows(phase0_root: Path) -> list[dict[str, Any]]:
    rows = _read_csv_rows(phase0_root / "baseline_metric_rows.csv")
    out: list[dict[str, Any]] = []
    for row in rows:
        if row.get("baseline_role") != "current_strong_local_baseline":
            continue
        out.append(
            {
                "schema_version": "stream4d_v103_supp_phaseS0_baseline_metric_row_v1",
                "phase_id": PHASE_ID,
                "source_phase_id": row.get("phase_id", ""),
                "source_artifact": row.get("source_artifact", ""),
                "baseline_role": row.get("baseline_role", ""),
                "variant_id": row.get("variant_id", ""),
                "dataset_split": row.get("dataset_split", ""),
                "MV_AP_window": row.get("MV_AP_window", ""),
                "MV_AP50_window": row.get("MV_AP50_window", ""),
                "MV_AP25_window": row.get("MV_AP25_window", ""),
                "ScoreFreeMatch50_window": row.get("ScoreFreeMatch50_window", ""),
                "fragmented_MV_AP_scene": row.get("fragmented_MV_AP_scene", ""),
                "fragmented_MV_AP50_scene": row.get("fragmented_MV_AP50_scene", ""),
                "same_frame_collision_count": row.get("same_frame_collision_count", ""),
                "pixel_collision_rate": row.get("pixel_collision_rate", ""),
                "missing_mask_raster_count": row.get("missing_mask_raster_count", ""),
                "uses_gt_for_prediction": row.get("uses_gt_for_prediction", ""),
                "uses_future": row.get("uses_future", ""),
                "metric_source": row.get("metric_source", ""),
            }
        )
    return out


def _count_truth_flags(summaries: list[dict[str, Any]], baseline_rows: list[dict[str, Any]]) -> tuple[int, int]:
    uses_gt = 0
    uses_future = 0
    for summary in summaries:
        if _boolish(summary.get("uses_gt_for_prediction", False)):
            uses_gt += 1
        if _boolish(summary.get("uses_future", False)):
            uses_future += 1
    for row in baseline_rows:
        if _boolish(row.get("uses_gt_for_prediction", False)):
            uses_gt += 1
        if _boolish(row.get("uses_future", False)):
            uses_future += 1
    return uses_gt, uses_future


def build(args: argparse.Namespace) -> dict[str, Any]:
    output_root = _project(args.output_root)
    roots = {name: _project(path) for name, path in DEFAULT_ARTIFACTS.items()}

    phase0 = _summary_at(roots["phase0_contract"])
    phase1 = _summary_at(roots["phase1_gpu_data_model"])
    phase2_11 = _summary_at(roots["phase2_scene0011_d4rt48mix_maskbalanced8"])
    phase2_50 = _summary_at(roots["phase2_scene0050_d4rt48mix_maskbalanced8"])
    phase3 = _summary_at(roots["phase3_all_d4rt48mix_maskbalanced8_e5"])
    phase4 = _summary_at(roots["phase4_affinity_feature_e5"])
    phase4_audit = _summary_at(roots["phase4_arithmetic_audit_e5"])
    phase5 = _summary_at(roots["phase5_mask_pooling_e5"])
    phase6 = _summary_at(roots["phase6_raw_birth_e5"])
    phase6_skeleton = _summary_at(roots["phase6_skeleton_best_current"])
    phase7 = _summary_at(roots["phase7_history_readiness_e5"])
    phase8_real = _summary_at(roots["phase8_history_injection_real_lam010"])
    phase8_shuffle = _summary_at(roots["phase8_history_injection_shuffle_lam010"])

    baseline_rows = _copy_baseline_rows(roots["phase0_contract"])
    baseline_by_split = {str(row.get("dataset_split")): row for row in baseline_rows}
    dev_base = baseline_by_split.get("dev", {})
    hold_base = baseline_by_split.get("holdout", {})
    summaries_for_truth = [
        phase1,
        phase2_11,
        phase2_50,
        phase4_audit,
        phase5,
        phase6,
        phase6_skeleton,
        phase7,
        phase8_real,
        phase8_shuffle,
    ]
    uses_gt_count, uses_future_count = _count_truth_flags(summaries_for_truth, baseline_rows)

    phase8_real_ap = _floatish(phase8_real.get("best_MV_AP_window"))
    phase8_shuffle_ap = _floatish(phase8_shuffle.get("best_MV_AP_window"))
    phase8_real_minus_shuffle = phase8_real_ap - phase8_shuffle_ap

    input_rows: list[dict[str, Any]] = []
    input_rows.extend(_artifact_file_rows("adjusted_method_doc", ADJUSTED_METHOD_DOC.parent, [ADJUSTED_METHOD_DOC.name]))
    input_rows.extend(_artifact_file_rows("supplement_plan_doc", SUPPLEMENT_PLAN_DOC.parent, [SUPPLEMENT_PLAN_DOC.name]))
    input_rows.extend(_artifact_file_rows("phase0_contract", roots["phase0_contract"], ["summary.json", "baseline_metric_rows.csv", "gate_rows.csv"]))
    input_rows.extend(_artifact_file_rows("phase1_gpu_data_model", roots["phase1_gpu_data_model"], ["summary.json", "gate_rows.csv"]))
    for artifact_id in ["phase2_scene0011_d4rt48mix_maskbalanced8", "phase2_scene0050_d4rt48mix_maskbalanced8"]:
        input_rows.extend(
            _artifact_file_rows(
                artifact_id,
                roots[artifact_id],
                ["summary.json", "carrier_batch.npz", "carrier_sources.npz", "mask_balance_rows.csv", "query_source_count_rows.csv", "gate_rows.csv"],
            )
        )
    input_rows.extend(
        _artifact_file_rows(
            "phase3_all_d4rt48mix_maskbalanced8_e5",
            roots["phase3_all_d4rt48mix_maskbalanced8_e5"],
            ["summary.json", "carrier_filter_metric_rows.csv", "gate_rows.csv"],
        )
    )
    input_rows.extend(
        _artifact_file_rows(
            "phase4_affinity_feature_e5",
            roots["phase4_affinity_feature_e5"],
            ["summary.json", "primitive_feature_metric_rows.csv", "gate_rows.csv"],
        )
    )
    input_rows.extend(_artifact_file_rows("phase4_arithmetic_audit_e5", roots["phase4_arithmetic_audit_e5"], ["summary.json"]))
    input_rows.extend(
        _artifact_file_rows(
            "phase5_mask_pooling_e5",
            roots["phase5_mask_pooling_e5"],
            ["summary.json", "mask_pooling_metric_rows.csv", "gate_rows.csv"],
        )
    )
    input_rows.extend(
        _artifact_file_rows(
            "phase6_raw_birth_e5",
            roots["phase6_raw_birth_e5"],
            ["summary.json", "local_mv_metric_rows.csv", "gate_rows.csv"],
        )
    )
    input_rows.extend(
        _artifact_file_rows(
            "phase6_skeleton_best_current",
            roots["phase6_skeleton_best_current"],
            ["summary.json", "merge_metric_rows.csv", "merge_edge_rows.csv", "merge_cluster_rows.csv"],
        )
    )
    input_rows.extend(
        _artifact_file_rows(
            "phase7_history_readiness_e5",
            roots["phase7_history_readiness_e5"],
            ["summary.json", "history_assignment_metric_rows.csv", "history_control_rows.csv", "gate_rows.csv"],
        )
    )
    for artifact_id in ["phase8_history_injection_real_lam010", "phase8_history_injection_shuffle_lam010"]:
        input_rows.extend(
            _artifact_file_rows(
                artifact_id,
                roots[artifact_id],
                ["summary.json", "local_mv_metric_rows.csv", "gate_rows.csv"],
            )
        )

    all_required_artifacts_exist = all(bool(row["exists"]) for row in input_rows)
    phase2_cache_available = all(
        (roots[name] / "carrier_batch.npz").exists()
        and (roots[name] / "carrier_sources.npz").exists()
        and int(_summary_at(roots[name]).get("failure_count", 999)) == 0
        for name in ["phase2_scene0011_d4rt48mix_maskbalanced8", "phase2_scene0050_d4rt48mix_maskbalanced8"]
    )

    gates = [
        _gate(
            "formal_metric_source_eq_v65",
            bool(phase0.get("formal_metric_source_eq_v65")),
            phase0.get("formal_metric_source_eq_v65", ""),
            "true",
            "repair evaluator adapter/path; do not run algorithm experiments",
        ),
        _gate(
            "current_strong_baseline_rows_readable",
            len(baseline_rows) >= 2 and bool(dev_base) and bool(hold_base),
            f"rows={len(baseline_rows)} dev={dev_base.get('MV_AP_window', '')} holdout={hold_base.get('MV_AP_window', '')}",
            "dev and holdout current_strong_local_baseline rows",
            "repair Phase0 baseline_metric_rows reading without rewriting AP evaluator",
        ),
        _gate(
            "phase1_gpu_artifact_available",
            _decision(phase1) == "PASS_ENTER_PHASE2_D4RT_QUERY" and int(phase1.get("failure_count", 999)) == 0,
            _decision(phase1),
            "PASS_ENTER_PHASE2_D4RT_QUERY",
            "repair Phase1 artifact path before carrier role work",
        ),
        _gate(
            "selected_d4rt48mix_provider_cache_available",
            phase2_cache_available,
            f"scene0011={_decision(phase2_11) or 'no_decision'} failures={phase2_11.get('failure_count', '')}; "
            f"scene0050={_decision(phase2_50) or 'no_decision'} failures={phase2_50.get('failure_count', '')}",
            "both selected Phase2 roots have carrier_batch/carrier_sources and failure_count=0",
            "repair selected Phase2 roots/cache/source parity before S1",
        ),
        _gate(
            "phase3_selected_carrier_filter_artifact_available",
            _decision(phase3) == "PASS_ENTER_PHASE4_PRIMITIVE_AFFINITY" and int(phase3.get("failure_count", 999)) == 0,
            _decision(phase3),
            "PASS_ENTER_PHASE4_PRIMITIVE_AFFINITY",
            "repair selected Phase3 artifact root before S1/S2",
        ),
        _gate(
            "phase4_affinity_feature_pass_artifact_available",
            _decision(phase4) == "PASS_ENTER_PHASE5_MASK_LEVEL_POOLING" and int(phase4.get("failure_count", 999)) == 0,
            _decision(phase4),
            "PASS_ENTER_PHASE5_MASK_LEVEL_POOLING",
            "repair Phase4 feature artifact before role-aware S2",
        ),
        _gate(
            "phase4_affinity_arithmetic_audit_pass",
            _decision(phase4_audit) == "PASS_PHASE4_AFFINITY_ARITHMETIC_AUDIT",
            _decision(phase4_audit),
            "PASS_PHASE4_AFFINITY_ARITHMETIC_AUDIT",
            "fix arithmetic/provenance bug before S2",
        ),
        _gate(
            "phase5_mask_level_pooling_pass_artifact_available",
            _decision(phase5) == "PASS_ENTER_PHASE6_MASK_CLUSTERING" and int(phase5.get("failure_count", 999)) == 0,
            _decision(phase5),
            "PASS_ENTER_PHASE6_MASK_CLUSTERING",
            "repair Phase5 pooling artifact before S2/S3",
        ),
        _gate(
            "phase6_raw_birth_no_go_recorded",
            _decision(phase6) == "NO_GO_REPAIR_PHASE6_MASK_CLUSTERING",
            f"decision={_decision(phase6)} best_MV_AP_window={phase6.get('best_MV_AP_window', '')}",
            "NO_GO_REPAIR_PHASE6_MASK_CLUSTERING recorded as boundary",
            "do not continue raw carrier-born birth as mainline",
        ),
        _gate(
            "phase8_history_injection_no_go_and_control_recorded",
            _decision(phase8_real) == "NO_GO_REPAIR_PHASE6_MASK_CLUSTERING"
            and _decision(phase8_shuffle) == "NO_GO_REPAIR_PHASE6_MASK_CLUSTERING"
            and abs(phase8_real_minus_shuffle) < 0.003,
            f"real={phase8_real_ap} shuffle={phase8_shuffle_ap} real_minus_shuffle={phase8_real_minus_shuffle}",
            "real/shuffle both No-Go and direct injection not control-separated",
            "do not restore direct history feature injection; use post-birth inheritance",
        ),
        _gate(
            "all_required_artifact_files_exist",
            all_required_artifacts_exist,
            f"missing={sum(1 for row in input_rows if not row['exists'])}",
            "0 missing required files",
            "repair artifact root/variant id drift before S1",
        ),
        _gate(
            "uses_gt_for_prediction_count_zero",
            uses_gt_count == 0,
            uses_gt_count,
            0,
            "remove any prediction-time GT dependency before continuing",
        ),
        _gate(
            "uses_future_count_zero",
            uses_future_count == 0,
            uses_future_count,
            0,
            "remove any future-frame dependency before continuing",
        ),
    ]

    failures = [
        {
            "schema_version": "stream4d_v103_supp_phaseS0_failure_row_v1",
            "phase_id": PHASE_ID,
            "gate_id": row["gate_id"],
            "observed": row["observed"],
            "required": row["required"],
            "repair_direction": row["repair_direction"],
        }
        for row in gates
        if not row["pass"]
    ]
    phaseS0_pass = not failures
    decision = "PASS_ENTER_PHASES1_MULTIROLE_CARRIER_SETS" if phaseS0_pass else "NO_GO_REPAIR_PHASES0_FACT_LOCK"

    output_root.mkdir(parents=True, exist_ok=True)
    input_csv = output_root / "input_artifact_rows.csv"
    baseline_csv = output_root / "baseline_metric_rows.csv"
    gate_csv = output_root / "gate_rows.csv"
    failure_csv = output_root / "failure_rows.csv"
    summary_path = output_root / "summary.json"

    _write_csv(input_csv, input_rows)
    _write_csv(baseline_csv, baseline_rows)
    _write_csv(gate_csv, gates)
    _write_csv(failure_csv, failures)

    summary = {
        "schema_version": "stream4d_v103_supp_phaseS0_summary_v1",
        "phase_id": PHASE_ID,
        "created_unix_time": time.time(),
        "decision": decision,
        "phaseS0_pass": bool(phaseS0_pass),
        "failure_count": len(failures),
        "formal_metric_source_eq_v65": bool(phase0.get("formal_metric_source_eq_v65")),
        "current_baseline_variant": dev_base.get("variant_id", ""),
        "baseline_dev_MV_AP_window": dev_base.get("MV_AP_window", ""),
        "baseline_dev_MV_AP50_window": dev_base.get("MV_AP50_window", ""),
        "baseline_holdout_MV_AP_window": hold_base.get("MV_AP_window", ""),
        "baseline_holdout_MV_AP50_window": hold_base.get("MV_AP50_window", ""),
        "fragmented_dev_MV_AP_scene": dev_base.get("fragmented_MV_AP_scene", ""),
        "fragmented_dev_MV_AP50_scene": dev_base.get("fragmented_MV_AP50_scene", ""),
        "fragmented_holdout_MV_AP_scene": hold_base.get("fragmented_MV_AP_scene", ""),
        "fragmented_holdout_MV_AP50_scene": hold_base.get("fragmented_MV_AP50_scene", ""),
        "local_support_policy": phase0.get("local_support_policy", ""),
        "scene_support_policy": phase0.get("scene_support_policy", ""),
        "AP_thresholds": phase0.get("AP_thresholds_actual", ""),
        "phase1_gpu_artifact_available": _decision(phase1) == "PASS_ENTER_PHASE2_D4RT_QUERY",
        "phase2_selected_d4rt_provider": "OpenD4RT_48CLIP_9Mix_NoCropAUG maskbalanced8 qchunk16384 chunk_index1 cap24576",
        "phase3_selected_carrier_filter_artifact": _rel(roots["phase3_all_d4rt48mix_maskbalanced8_e5"]),
        "phase4_affinity_feature_artifact": _rel(roots["phase4_affinity_feature_e5"]),
        "phase5_mask_level_feature_artifact": _rel(roots["phase5_mask_pooling_e5"]),
        "phase6_raw_birth_decision": _decision(phase6),
        "phase6_raw_birth_best_MV_AP_window": phase6.get("best_MV_AP_window", ""),
        "phase6_raw_birth_best_MV_AP50_window": phase6.get("best_MV_AP50_window", ""),
        "phase6_current_scaffolded_skeleton_best": _rel(roots["phase6_skeleton_best_current"]),
        "phase6_current_scaffolded_skeleton_best_MV_AP_window": phase6_skeleton.get("best_MV_AP_window", ""),
        "phase6_current_scaffolded_skeleton_best_MV_AP50_window": phase6_skeleton.get("best_MV_AP50_window", ""),
        "phase8_history_injection_decision": _decision(phase8_real),
        "phase8_history_injection_real_MV_AP_window": phase8_real.get("best_MV_AP_window", ""),
        "phase8_history_injection_shuffle_MV_AP_window": phase8_shuffle.get("best_MV_AP_window", ""),
        "phase8_history_injection_real_minus_shuffle_MV_AP_window": phase8_real_minus_shuffle,
        "uses_gt_for_prediction_count": uses_gt_count,
        "uses_future_count": uses_future_count,
        "outputs": {
            "summary": _rel(summary_path),
            "input_artifact_rows": _rel(input_csv),
            "baseline_metric_rows": _rel(baseline_csv),
            "gate_rows": _rel(gate_csv),
            "failure_rows": _rel(failure_csv),
        },
        "truthfulness_note": (
            "Phase S0 is read-only. It locks current v103 artifacts and boundaries for the supplement plan; "
            "it does not generate method predictions, tune thresholds, or rewrite the AP evaluator."
        ),
    }
    _write_json(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    summary = build(args)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    raise SystemExit(0 if summary["phaseS0_pass"] else 2)


if __name__ == "__main__":
    main()
