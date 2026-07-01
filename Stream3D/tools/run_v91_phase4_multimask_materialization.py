from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import run_v90_dev_extent_score_cross_audit as phase7d  # noqa: E402
from tools import run_v90_geo_semantic_witness_cover as phase4  # noqa: E402
from tools import run_v91_phase4_ap50_control_repair as v91repair  # noqa: E402
from tools import run_v91_phase4_radius_sweep as radius_sweep  # noqa: E402


OUT = ROOT / "outputs/audit/v91_phase4_witness_cover_multimask_materialization"
SUPPORT_ROWS = ROOT / "outputs/audit/v90_phase3_v82_full_support/native_carrier_support_rows.csv"
SCORE_SOURCE_VARIANT = "W8a_risk_balanced_p135_witnesses"


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


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return float(default)
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return int(default)
        return int(float(value))
    except Exception:
        return int(default)


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _mean(values: list[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else 0.0


def _variant_specs() -> list[dict[str, Any]]:
    return [
        {
            "variant_id": "V91_M1_W8a_top2_r16_drop5_sceneorig",
            "max_masks": 2,
            "extra_score_delta": 0.35,
            "allow_broad_extra": False,
            "radius": 16,
            "support_point_radius": 3,
        },
        {
            "variant_id": "V91_M2_W8a_top3_r16_drop5_sceneorig",
            "max_masks": 3,
            "extra_score_delta": 0.50,
            "allow_broad_extra": False,
            "radius": 16,
            "support_point_radius": 3,
        },
        {
            "variant_id": "V91_M3_W8a_top4_r16_drop5_sceneorig",
            "max_masks": 4,
            "extra_score_delta": 0.65,
            "allow_broad_extra": False,
            "radius": 16,
            "support_point_radius": 3,
        },
        {
            "variant_id": "V91_M4_W8a_top3_r12_drop5_sceneorig",
            "max_masks": 3,
            "extra_score_delta": 0.50,
            "allow_broad_extra": False,
            "radius": 12,
            "support_point_radius": 3,
        },
        {
            "variant_id": "V91_M5_W8a_top3_r16_allow_broad_drop5_sceneorig",
            "max_masks": 3,
            "extra_score_delta": 0.50,
            "allow_broad_extra": True,
            "radius": 16,
            "support_point_radius": 3,
        },
    ]


def _select_multimask_rows(
    candidates: list[dict[str, Any]],
    slot_to_obj: dict[tuple[str, str], str],
    slot_to_proto: dict[tuple[str, str], str],
    slot_to_area: dict[tuple[str, str], float],
    frame_to_window_index: dict[tuple[str, int], int],
    frame_to_window_id: dict[tuple[str, int], str],
    spec: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_variant = f"{spec['variant_id']}_source"
    by_slot_frame: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_slot_frame[(str(row["scene_id"]), str(row["local_slot_id"]), _int(row.get("frame_id"), -1))].append(row)

    pre_rows: list[dict[str, Any]] = []
    for (scene, slot, frame_id), items in sorted(by_slot_frame.items()):
        slot_key = (scene, slot)
        if slot_key not in slot_to_obj:
            continue
        scored: list[tuple[float, dict[str, Any]]] = []
        for item in items:
            score = phase4._score_candidate(
                item,
                SCORE_SOURCE_VARIANT,
                slot_to_proto.get(slot_key, ""),
                slot_to_area.get(slot_key, 0.0),
            )
            scored.append((score, item))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        if not scored:
            continue
        selected: list[tuple[float, dict[str, Any]]] = [scored[0]]
        top_score = scored[0][0]
        for score, item in scored[1:]:
            if len(selected) >= int(spec["max_masks"]):
                break
            if not bool(spec["allow_broad_extra"]) and _bool(item.get("broad_background_risk")):
                continue
            if score >= top_score - float(spec["extra_score_delta"]):
                selected.append((score, item))
        for rank, (score, item) in enumerate(selected, start=1):
            pre_rows.append(
                {
                    **item,
                    "variant_id": source_variant,
                    "mv_object_id": f"{source_variant}:{slot_to_obj[slot_key]}",
                    "window_index": int(frame_to_window_index.get((scene, frame_id), -1)),
                    "window_id": frame_to_window_id.get((scene, frame_id), ""),
                    "selection_score": float(score),
                    "selection_rank": int(rank),
                    "selection_stage": "pre_conflict_wta_multimask",
                    "selection_reason": (
                        f"v91_multimask_{SCORE_SOURCE_VARIANT}_max{int(spec['max_masks'])}"
                        f"_delta{float(spec['extra_score_delta']):.2f}_allowBroad{bool(spec['allow_broad_extra'])}"
                    ),
                    "risk_penalty": 1.0 if _bool(item.get("broad_background_risk")) else 0.0,
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )

    kept: dict[tuple[str, int, int], dict[str, Any]] = {}
    dropped: list[dict[str, Any]] = []
    for row in sorted(pre_rows, key=lambda r: _num(r.get("selection_score")), reverse=True):
        key = (str(row["scene_id"]), _int(row.get("frame_id"), -1), _int(row.get("mask_id"), -1))
        old = kept.get(key)
        if old is None:
            kept[key] = {**row, "selection_stage": "post_conflict_wta_multimask", "conflict_dropped": False}
        else:
            dropped.append(
                {
                    **row,
                    "selection_stage": "dropped_by_same_frame_mask_wta_multimask",
                    "conflict_dropped": True,
                    "kept_mv_object_id": old.get("mv_object_id", ""),
                }
            )
    final_rows = sorted(
        kept.values(),
        key=lambda r: (r["variant_id"], r["scene_id"], r["local_slot_id"], _int(r.get("frame_id")), -_num(r.get("selection_score"))),
    )
    return final_rows, dropped


def run(_args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    phase4.OUT = OUT
    radius_sweep.OUT = OUT
    mask_dirs = phase4._mask_dir_by_scene()
    frame_to_window_index, frame_to_window_id = phase4._window_maps()
    _source_rows, slot_to_obj, slot_to_proto, slot_to_area = phase4._load_source_rows()
    semantic_features = phase4._load_semantic_features()
    candidates, support_points = phase4._load_support_candidates(SUPPORT_ROWS, set(slot_to_obj), semantic_features, mask_dirs)
    slot_to_proto, slot_to_area = phase4._fill_slot_priors_from_candidates(candidates, slot_to_proto, slot_to_area)
    baselines = v91repair._phase8_baselines()

    config_rows: list[dict[str, Any]] = []
    source_selection_rows_all: list[dict[str, Any]] = []
    dropped_source_rows_all: list[dict[str, Any]] = []
    generated_rows_all: list[dict[str, Any]] = []
    selected_rows_all: list[dict[str, Any]] = []
    eval_rows_all: list[dict[str, Any]] = []
    scored_rows_all: list[dict[str, Any]] = []
    support_quality_all: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    risk_rows: list[dict[str, Any]] = []

    for spec in _variant_specs():
        variant_id = str(spec["variant_id"])
        source_variant = f"{variant_id}_source"
        source_rows, dropped_source_rows = _select_multimask_rows(
            candidates,
            slot_to_obj,
            slot_to_proto,
            slot_to_area,
            frame_to_window_index,
            frame_to_window_id,
            spec,
        )
        generated_rows, selected_rows, eval_rows = phase4._generate_carved_masks(
            source_rows,
            support_points,
            mask_dirs,
            radius=int(spec["radius"]),
            support_point_radius=int(spec["support_point_radius"]),
            variant=variant_id,
            source_variant=source_variant,
        )
        feature_map = radius_sweep._feature_map(selected_rows, generated_rows)
        scored_rows, keep_flags = v91repair._variant_rows(
            eval_rows,
            feature_map,
            variant_id=variant_id,
            risk_mode="drop_broad_low_h9_5",
            score_mode="broad_scene_orig_ge065",
            group_name="multimask_materialization",
        )
        support_quality = v91repair._support_quality_rows(scored_rows, feature_map, variant_id)
        metrics, cases = radius_sweep._evaluate_variant(variant_id, scored_rows)

        source_selection_rows_all.extend(source_rows)
        dropped_source_rows_all.extend(dropped_source_rows)
        generated_rows_all.extend(generated_rows)
        selected_rows_all.extend(selected_rows)
        eval_rows_all.extend(eval_rows)
        scored_rows_all.extend(scored_rows)
        support_quality_all.extend(support_quality)
        metric_rows.extend(metrics)
        case_rows.extend({**row, "variant_id": variant_id} for row in cases)
        broad_values = [1.0 if _bool(row.get("broad_risk")) else 0.0 for row in support_quality]
        risk_rows.append(
            {
                "variant_id": variant_id,
                "score_source_variant": SCORE_SOURCE_VARIANT,
                "max_masks": int(spec["max_masks"]),
                "extra_score_delta": float(spec["extra_score_delta"]),
                "allow_broad_extra": bool(spec["allow_broad_extra"]),
                "radius": int(spec["radius"]),
                "support_point_radius": int(spec["support_point_radius"]),
                "risk_filter_mode": "drop_broad_low_h9_5",
                "score_mode": "broad_scene_orig_ge065",
                "source_selected_rows": len(source_rows),
                "source_conflict_dropped_rows": len(dropped_source_rows),
                "selected_rows": len(scored_rows),
                "pre_filter_eval_rows": len(eval_rows),
                "dropped_rows": len(eval_rows) - int(sum(1 for flag in keep_flags if flag)),
                "risk_penalty_mean": _mean(broad_values),
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
        config_rows.append(
            {
                "variant_id": variant_id,
                "changed_module": "phase4_witness_cover_multimask_materialization",
                "changed_parameters": (
                    f"score_source={SCORE_SOURCE_VARIANT}; max_masks={int(spec['max_masks'])}; "
                    f"extra_score_delta={float(spec['extra_score_delta'])}; allow_broad_extra={bool(spec['allow_broad_extra'])}; "
                    f"carving_radius={int(spec['radius'])}; support_point_radius={int(spec['support_point_radius'])}; "
                    "risk_filter=drop_broad_low_h9_5; score=broad_scene_orig_ge065"
                ),
                "reason_for_change": (
                    "v91 Phase8 refreshed best r16 still fails AP50 control margin; scene0011 has low score-free Match50, "
                    "so test whether multi-masklet D4RT witness materialization improves extent without using GT/holdout"
                ),
                "uses_gt_for_prediction": False,
                "uses_future": False,
                "dev_only_or_holdout": "dev_only",
                "expected_blocker": "EXTENT_BLOCKER+CONTROL_BIAS_BLOCKER",
            }
        )

    aggregate_rows = phase7d._aggregate(metric_rows)
    control_rows = v91repair._add_gate_rows(aggregate_rows, baselines)
    best = max(
        control_rows,
        key=lambda row: (
            _num(row.get("dev_gate_min_margin"), -999.0),
            _num(row.get("mean_MV_AP50_window"), -999.0),
            _num(row.get("mean_MV_AP_window"), -999.0),
        ),
        default={},
    )
    passing = [row for row in control_rows if _bool(row.get("v91_phase8_progress_gate_pass"))]
    summary = {
        "phase": "v91_phase4_witness_cover_multimask_materialization",
        "schema": "stream4d_v91_phase4_multimask_materialization_v1",
        "score_source_variant": SCORE_SOURCE_VARIANT,
        "variant_count": len(config_rows),
        "best_variant_id": best.get("variant_id", ""),
        "best_variant_gate": best,
        "any_v91_phase8_progress_gate_pass": bool(passing),
        "decision": "DEV_GATE_PASS_READY_FOR_PHASE8_FREEZE" if passing else "DEV_GATE_FAIL_CONTINUE_REPAIR",
        "next_action": "refresh Phase8 and freeze before holdout" if passing else "continue dev-only extent/control repair; do not run Phase9 holdout",
        "row_counts": {
            "variant_config_rows": len(config_rows),
            "source_selection_rows": len(source_selection_rows_all),
            "dropped_source_rows": len(dropped_source_rows_all),
            "generated_mask_rows": len(generated_rows_all),
            "selected_masklet_rows": len(selected_rows_all),
            "pre_filter_eval_rows": len(eval_rows_all),
            "scored_frame_mask_rows": len(scored_rows_all),
            "support_quality_rows": len(support_quality_all),
            "mv_metric_rows": len(metric_rows),
            "control_metric_rows": len(control_rows),
            "casebook_rows": len(case_rows),
        },
        "uses_gt_for_prediction": False,
        "uses_future": False,
        "holdout_policy": "No holdout is used or touched by this dev-only materialization repair.",
        "runtime_sec": time.time() - started,
    }
    _write_csv(OUT / "variant_config_rows.csv", config_rows)
    _write_csv(OUT / "source_selection_rows.csv", source_selection_rows_all)
    _write_csv(OUT / "dropped_source_rows.csv", dropped_source_rows_all)
    _write_csv(OUT / "generated_mask_rows.csv", generated_rows_all)
    _write_csv(OUT / "selected_masklet_rows.csv", selected_rows_all)
    _write_csv(OUT / "mv_object_frame_mask_rows.csv", scored_rows_all)
    _write_csv(OUT / "support_quality_rows.csv", support_quality_all)
    _write_csv(OUT / "risk_rows.csv", risk_rows)
    _write_csv(OUT / "mv_metric_rows.csv", metric_rows)
    _write_csv(OUT / "mv_metric_aggregate_rows.csv", aggregate_rows)
    _write_csv(OUT / "control_metric_rows.csv", control_rows)
    _write_csv(OUT / "casebook_rows.csv", case_rows)
    _write_json(OUT / "summary.json", summary)
    outputs = [
        OUT / "variant_config_rows.csv",
        OUT / "source_selection_rows.csv",
        OUT / "dropped_source_rows.csv",
        OUT / "generated_mask_rows.csv",
        OUT / "selected_masklet_rows.csv",
        OUT / "mv_object_frame_mask_rows.csv",
        OUT / "support_quality_rows.csv",
        OUT / "risk_rows.csv",
        OUT / "mv_metric_rows.csv",
        OUT / "mv_metric_aggregate_rows.csv",
        OUT / "control_metric_rows.csv",
        OUT / "casebook_rows.csv",
        OUT / "summary.json",
    ]
    _write_json(OUT / "SHA256SUMS.json", {_rel(path): _sha256(path) for path in outputs if path.exists()})
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run v91 Phase4 multi-masklet materialization repair on dev only.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
