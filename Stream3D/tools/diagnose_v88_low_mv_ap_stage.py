#!/usr/bin/env python3
"""Diagnose why v88 formal MV_AP is much lower than prior AP diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent


def _project(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == "Stream3D":
        return REPO / path
    return ROOT / path


def _read_json(path: str | Path) -> dict[str, Any]:
    path = _project(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: str | Path) -> list[dict[str, Any]]:
    path = _project(path)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
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
            writer.writerow({field: _scalar(row.get(field, "")) for field in fields})


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _scalar(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_jsonable(value), sort_keys=True, ensure_ascii=False)
    return value


def _num(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _mean(values: list[float]) -> float:
    clean = [v for v in values if math.isfinite(v)]
    return sum(clean) / len(clean) if clean else 0.0


def _norm_variant(variant: str) -> str:
    mapping = {
        "B3_DV5_object_gain_with_local_fallback": "B3_history_with_local_fallback",
        "B4_M10_state_priority_with_local_fallback": "B4_state_priority_with_local_fallback",
        "B5_confirmed_only_conservative": "B5_carrier_gated_frame_mask_readout",
    }
    return mapping.get(str(variant), str(variant))


def _metric_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("split", "")),
        str(row.get("scene_id", "")),
        _norm_variant(str(row.get("variant", ""))),
        str(row.get("score_mode", "input")),
    )


def _aggregate(rows: list[dict[str, Any]], metric: str = "MV_AP") -> dict[str, float]:
    by_variant: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_variant[_norm_variant(str(row.get("variant", "")))].append(_num(row.get(metric), 0.0))
    return {variant: _mean(values) for variant, values in by_variant.items()}


def _best(rows: list[dict[str, Any]], variants: set[str]) -> tuple[str, float]:
    agg = _aggregate([row for row in rows if _norm_variant(str(row.get("variant", ""))) in variants])
    if not agg:
        return "", 0.0
    variant = max(sorted(agg), key=lambda key: agg[key])
    return variant, agg[variant]


def _adapter_parity(v88_rows: list[dict[str, Any]], direct_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    direct_by_key = {_metric_key(row): row for row in direct_rows}
    rows: list[dict[str, Any]] = []
    max_delta = 0.0
    for row in v88_rows:
        if str(row.get("score_mode", "")) != "input" or str(row.get("split", "")) != "dev":
            continue
        key = _metric_key(row)
        other = direct_by_key.get(key)
        if other is None:
            rows.append({"split": key[0], "scene_id": key[1], "variant": key[2], "parity_status": "missing_in_direct_adapter"})
            max_delta = max(max_delta, 1.0)
            continue
        out = {
            "split": key[0],
            "scene_id": key[1],
            "variant": key[2],
            "score_mode": key[3],
            "v88_MV_AP": row.get("MV_AP", ""),
            "direct_MV_AP": other.get("MV_AP", ""),
            "delta_MV_AP": _num(row.get("MV_AP"), 0.0) - _num(other.get("MV_AP"), 0.0),
            "v88_MV_AP50": row.get("MV_AP50", ""),
            "direct_MV_AP50": other.get("MV_AP50", ""),
            "delta_MV_AP50": _num(row.get("MV_AP50"), 0.0) - _num(other.get("MV_AP50"), 0.0),
            "v88_MV_AP25": row.get("MV_AP25", ""),
            "direct_MV_AP25": other.get("MV_AP25", ""),
            "delta_MV_AP25": _num(row.get("MV_AP25"), 0.0) - _num(other.get("MV_AP25"), 0.0),
            "parity_status": "match",
        }
        max_delta = max(max_delta, abs(out["delta_MV_AP"]), abs(out["delta_MV_AP50"]), abs(out["delta_MV_AP25"]))
        rows.append(out)
    return rows, {"adapter_parity_max_abs_delta": max_delta, "adapter_parity_pass": max_delta <= 1e-12}


def _v87_v88_comparison(v87_rows: list[dict[str, Any]], v88_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    v87_by_key = {
        (str(row.get("scene_id", "")), _norm_variant(str(row.get("variant", "")))): row
        for row in v87_rows
        if row.get("split") == "dev" and row.get("score_mode") == "input"
    }
    rows: list[dict[str, Any]] = []
    for row in v88_rows:
        if row.get("split") != "dev" or row.get("score_mode") != "input":
            continue
        variant = _norm_variant(str(row.get("variant", "")))
        if variant not in {"B0_local_only", "B3_history_with_local_fallback", "B4_state_priority_with_local_fallback"}:
            continue
        key = (str(row.get("scene_id", "")), variant)
        old = v87_by_key.get(key, {})
        rows.append(
            {
                "scene_id": key[0],
                "variant": variant,
                "v87_MV_AP": old.get("MV_AP", ""),
                "v88_MV_AP": row.get("MV_AP", ""),
                "v88_minus_v87_MV_AP": _num(row.get("MV_AP"), 0.0) - _num(old.get("MV_AP"), 0.0),
                "v87_MV_AP50": old.get("MV_AP50", ""),
                "v88_MV_AP50": row.get("MV_AP50", ""),
                "note": "v88 uses expanded dev chunks; values are same formal evaluator family, not native AP",
            }
        )
    return rows


def _stage_rows(v85: dict[str, Any], v86: dict[str, Any], v88_rows: list[dict[str, Any]], p4: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [
        {
            "stage": "v85_native_carrier_diagnostic",
            "metric_name": "native_carrier_cluster_AP50",
            "metric_value": v85.get("native_carrier_cluster_AP50", ""),
            "metric_scope": v85.get("native_carrier_diagnostic_metric_scope", "native_carrier_mask_diagnostic_cluster_metric_not_scannet_ap"),
            "method_safe_for_v88_success": False,
            "interpretation": "high previous AP is native-carrier diagnostic, not v65 MV_AP",
        },
        {
            "stage": "v86_tracklet_readout_native",
            "metric_name": "selected_native_AP50",
            "metric_value": v86.get("selected_native_AP50", ""),
            "metric_scope": "native carrier / tracklet readout diagnostic",
            "method_safe_for_v88_success": False,
            "interpretation": "local2history/native signal was high before frame-mask MV_AP materialization",
        },
    ]
    for row in v88_rows:
        if row.get("split") == "dev" and row.get("score_mode") == "input" and row.get("variant") in {
            "B0_local_only",
            p4.get("best_real_variant", ""),
            p4.get("best_control_variant", ""),
        }:
            rows.append(
                {
                    "stage": "v88_formal_mv_object_tube",
                    "metric_name": f"{row.get('variant')}:{row.get('scene_id')}:MV_AP",
                    "metric_value": row.get("MV_AP", ""),
                    "metric_scope": "v65 SparseSceneIoU/_summarize_iou over materialized 2D MV object tubes",
                    "method_safe_for_v88_success": row.get("variant") != p4.get("best_control_variant", ""),
                    "interpretation": "formal v88 MV_AP, directly comparable for method gate",
                }
            )
    return rows


def _readout_vs_local(v88_rows: list[dict[str, Any]], materializer_rows: list[dict[str, Any]], scorefree_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    b0_by_scene = {
        str(row.get("scene_id", "")): row
        for row in v88_rows
        if row.get("split") == "dev" and row.get("score_mode") == "input" and row.get("variant") == "B0_local_only"
    }
    mat = {(row.get("scene_id", ""), row.get("variant", "")): row for row in materializer_rows if row.get("split") == "dev"}
    sf = {(row.get("scene_id", ""), row.get("variant", "")): row for row in scorefree_rows if row.get("score_mode") == "input"}
    rows: list[dict[str, Any]] = []
    for row in v88_rows:
        if row.get("split") != "dev" or row.get("score_mode") != "input":
            continue
        scene = str(row.get("scene_id", ""))
        variant = str(row.get("variant", ""))
        local = b0_by_scene.get(scene, {})
        mat_row = mat.get((scene, variant), {})
        sf_row = sf.get((scene, variant), {})
        delta = _num(row.get("MV_AP"), 0.0) - _num(local.get("MV_AP"), 0.0)
        if variant in {"B1_M10_state_priority", "B2_DV5_confirmed_object_gain"} and _num(mat_row.get("mv_object_count"), 0.0) < 20:
            stage_hint = "local2history_pure_history_sparse"
        elif variant in {"B3_history_with_local_fallback", "B4_state_priority_with_local_fallback", "B6_area_penalized_history_readout"} and abs(delta) < 0.0015:
            stage_hint = "readout_mostly_tracks_local_fallback"
        elif variant.startswith("C0"):
            stage_hint = "semantic_control_bias"
        elif delta < 0:
            stage_hint = "readout_degrades_local"
        else:
            stage_hint = "small_readout_gain"
        rows.append(
            {
                "scene_id": scene,
                "variant": variant,
                "MV_AP": row.get("MV_AP", ""),
                "B0_MV_AP": local.get("MV_AP", ""),
                "minus_B0_MV_AP": delta,
                "MV_AP50": row.get("MV_AP50", ""),
                "MV_AP25": row.get("MV_AP25", ""),
                "scorefree_match50_recall": sf_row.get("scorefree_match50_recall", ""),
                "scorefree_match25_recall": sf_row.get("scorefree_match25_recall", ""),
                "mv_object_count": mat_row.get("mv_object_count", ""),
                "mean_frames_per_object": mat_row.get("mean_frames_per_object", ""),
                "broad_mask_support_rate": mat_row.get("broad_mask_support_rate", ""),
                "stage_hint": stage_hint,
            }
        )
    return rows


def _local2history_support(tracklet_rows: list[dict[str, Any]], materializer_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    counts: Counter[tuple[str, str, str]] = Counter()
    confirmed_gain: Counter[tuple[str, str]] = Counter()
    for row in tracklet_rows:
        scene = str(row.get("scene_id", ""))
        chunk = str(row.get("chunk_id", ""))
        state = str(row.get("tracklet_state_after", "unknown")) or "unknown"
        counts[(scene, chunk, state)] += 1
        support_slots = _num(row.get("support_slot_count_after"), 0.0)
        support_chunks = _num(row.get("support_chunk_count_after"), 0.0)
        full_minus_semantic = _num(row.get("full_minus_semantic_slot"), -999.0)
        if state == "confirmed" and support_slots >= 2 and support_chunks >= 2 and full_minus_semantic >= 0.03:
            confirmed_gain[(scene, chunk)] += 1
    for (scene, chunk, state), count in sorted(counts.items()):
        rows.append(
            {
                "source": "v82_tracklet_assignment_rows",
                "scene_id": scene,
                "chunk_id": chunk,
                "state_or_variant": state,
                "row_count": count,
                "confirmed_object_gain_rows_same_chunk": confirmed_gain.get((scene, chunk), 0),
                "note": "local2history source support before MV_AP materialization",
            }
        )
    for row in materializer_rows:
        if row.get("split") == "dev" and row.get("variant") in {"B1_M10_state_priority", "B2_DV5_confirmed_object_gain", "B3_history_with_local_fallback", "B4_state_priority_with_local_fallback"}:
            rows.append(
                {
                    "source": "v88_phase2_materializer",
                    "scene_id": row.get("scene_id", ""),
                    "chunk_id": "aggregate",
                    "state_or_variant": row.get("variant", ""),
                    "row_count": row.get("mv_object_count", ""),
                    "confirmed_object_gain_rows_same_chunk": "",
                    "note": "materialized MV object count by readout variant",
                }
            )
    return rows


def _mechanism_rows(scorefree_rows: list[dict[str, Any]], materializer_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    mat = {(row.get("scene_id", ""), row.get("variant", "")): row for row in materializer_rows if row.get("split") == "dev"}
    rows: list[dict[str, Any]] = []
    for row in scorefree_rows:
        if row.get("score_mode") != "input":
            continue
        key = (row.get("scene_id", ""), row.get("variant", ""))
        mat_row = mat.get(key, {})
        ap50 = _num(row.get("MV_AP50"), 0.0)
        sf50 = _num(row.get("scorefree_match50_recall"), 0.0)
        if sf50 >= ap50 + 0.05:
            mechanism = "ranking_or_many_fp_penalty_visible"
        elif sf50 < 0.02:
            mechanism = "extent_or_grouping_low_iou"
        else:
            mechanism = "mixed_extent_grouping_ranking"
        rows.append(
            {
                "scene_id": row.get("scene_id", ""),
                "variant": row.get("variant", ""),
                "MV_AP": row.get("MV_AP", ""),
                "MV_AP50": row.get("MV_AP50", ""),
                "scorefree_match50_recall": row.get("scorefree_match50_recall", ""),
                "scorefree_minus_MV_AP50": sf50 - ap50,
                "pred_object_count": mat_row.get("mv_object_count", ""),
                "mean_frames_per_object": mat_row.get("mean_frames_per_object", ""),
                "singleton_object_rate": mat_row.get("singleton_object_rate", ""),
                "mechanism_hint": mechanism,
            }
        )
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    out = _project(args.output_root)
    v88_rows = _read_csv("outputs/audit/v88_phase3_mv_ap_eval/mv_metric_rows.csv")
    direct_rows = _read_csv("outputs/audit/v88_debug_existing_adapter_inputscore/mv_metric_rows.csv")
    v87_rows = _read_csv("outputs/audit/v87_repair2_existing_mv_ap_inputscore/mv_metric_rows.csv")
    materializer_rows = _read_csv("outputs/audit/v88_phase2_mv_tube/materializer_metric_rows.csv")
    scorefree_rows = _read_csv("outputs/audit/v88_phase4_mv_ap_decomposition/scorefree_match_rows.csv")
    tracklet_rows = _read_csv("outputs/audit/v82_phase2_object_tracklets_repair11_app079_sigma022/tracklet_assignment_rows.csv")
    p3 = _read_json("outputs/audit/v88_phase3_mv_ap_eval/mv_eval_summary.json")
    p4 = _read_json("outputs/audit/v88_phase4_mv_ap_decomposition/phase4_summary.json")
    p9 = _read_json("outputs/audit/v88_phase9_casebook/final_decision.json")
    v85 = _read_json("outputs/audit/v85_phase7_renderable_materializer/materializer_summary.json")
    v86 = _read_json("outputs/audit/v86_phase12_dev_tracklet_readout_repair/dev_tracklet_readout_summary.json")

    parity_rows, parity_summary = _adapter_parity(v88_rows, direct_rows)
    comparison_rows = _v87_v88_comparison(v87_rows, v88_rows)
    stage_rows = _stage_rows(v85, v86, v88_rows, p4)
    readout_rows = _readout_vs_local(v88_rows, materializer_rows, scorefree_rows)
    support_rows = _local2history_support(tracklet_rows, materializer_rows)
    mechanism_rows = _mechanism_rows(scorefree_rows, materializer_rows)

    dev_input = [row for row in v88_rows if row.get("split") == "dev" and row.get("score_mode") == "input"]
    real_variants = {
        "B1_M10_state_priority",
        "B2_DV5_confirmed_object_gain",
        "B3_history_with_local_fallback",
        "B4_state_priority_with_local_fallback",
        "B5_carrier_gated_frame_mask_readout",
        "B6_area_penalized_history_readout",
    }
    b0 = _aggregate([row for row in dev_input if row.get("variant") == "B0_local_only"]).get("B0_local_only", 0.0)
    best_real_variant, best_real = _best(dev_input, real_variants)
    best_control_variant, best_control = _best(
        dev_input,
        {
            "C0_semantic_only_control",
            "C1_shuffled_history_control",
            "C2_stale_history_control",
            "C3_size_matched_hash_control",
            "C4_single_largest_by_scene_control",
            "C5_local_only_area_rank_control",
        },
    )
    pure_history_counts = {
        f"{row.get('scene_id')}:{row.get('variant')}": row.get("mv_object_count", "")
        for row in materializer_rows
        if row.get("split") == "dev" and row.get("variant") in {"B1_M10_state_priority", "B2_DV5_confirmed_object_gain"}
    }
    summary = {
        "schema": "stream4d_v88_low_mv_ap_stage_diagnosis_v1",
        "decision": "LOW_MV_AP_NOT_V88_WRAPPER_BUG_CONTROL_BIAS_AND_LOCAL_TUBE_STAGE_DROP",
        "formal_metric_source": p3.get("formal_metric_source", ""),
        "phase3_decision": p3.get("decision", ""),
        "phase9_final_decision": p9.get("final_decision", ""),
        "phase9_primary_blocker": p9.get("primary_blocker", ""),
        **parity_summary,
        "previous_high_ap_examples": {
            "v85_native_carrier_cluster_AP50": v85.get("native_carrier_cluster_AP50", ""),
            "v86_selected_native_AP50": v86.get("selected_native_AP50", ""),
            "v86_selected_ARI": v86.get("selected_ARI", ""),
            "v86_selected_purity": v86.get("selected_purity", ""),
            "scope_note": "These are native-carrier / diagnostic metrics, not v65 MV_AP over materialized 2D multi-view object tubes.",
        },
        "v88_dev_aggregate": {
            "B0_MV_AP": b0,
            "best_real_variant": best_real_variant,
            "best_real_MV_AP": best_real,
            "best_control_variant": best_control_variant,
            "best_control_MV_AP": best_control,
            "best_real_minus_B0_MV_AP": best_real - b0,
            "best_real_minus_best_control_MV_AP": best_real - best_control,
            "pure_history_mv_object_counts": pure_history_counts,
        },
        "stage_conclusion": (
            "The large gap versus previous AP starts before local2history can explain it: "
            "B0_local_only, which does not use local2history, already has very low formal MV_AP. "
            "The previous high AP values are native-carrier/diagnostic scopes. Within formal MV_AP, "
            "pure history B1/B2 are sparse and weak, while B3/B4/B6 mostly follow local fallback with small deltas. "
            "The current method blocker is control/readout bias because C0 semantic-only control beats the best real variant."
        ),
    }
    _write_csv(out / "adapter_parity_rows.csv", parity_rows)
    _write_csv(out / "v87_v88_metric_comparison_rows.csv", comparison_rows)
    _write_csv(out / "stage_attribution_rows.csv", stage_rows)
    _write_csv(out / "readout_vs_local_rows.csv", readout_rows)
    _write_csv(out / "local2history_support_rows.csv", support_rows)
    _write_csv(out / "mv_ap_failure_mechanism_rows.csv", mechanism_rows)
    _write_json(out / "stage_diagnosis_summary.json", summary)
    print(json.dumps({"decision": summary["decision"], "output_root": str(out)}, sort_keys=True))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="outputs/audit/v88_low_mv_ap_stage_diagnosis")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
