from __future__ import annotations

import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
STREAM3D = ROOT / "Stream3D"
AUDIT_ROOT = STREAM3D / "outputs" / "audit"

PHASE1_DIR = AUDIT_ROOT / "v101_phase1_f2_fragmentation_casebook"
PHASE2_DIR = AUDIT_ROOT / "v101_phase2_geometry_provider_capability"
PHASE2C_DIR = AUDIT_ROOT / "v100_phase2c_overlap3_local_repair"
OUT_DIR = AUDIT_ROOT / "v101_phase2b_false_bridge_repair_probe"

TAUS = [0.10, 0.20, 0.30, 0.40, 0.50]


def _rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return _rel(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
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


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key, "")) for key in keys})


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        if isinstance(value, float) and math.isnan(value):
            return default
        return float(value)
    except Exception:
        return default


def _chunk_index(chunk_id: str) -> int:
    digits = "".join(ch for ch in str(chunk_id) if ch.isdigit())
    return int(digits) if digits else 0


def _roc_auc(scores: list[float], labels: list[int]) -> float | None:
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return None
    wins = 0.0
    for ps in pos:
        for ns in neg:
            if ps > ns:
                wins += 1.0
            elif ps == ns:
                wins += 0.5
    return float(wins / (len(pos) * len(neg)))


def _load_best_gt() -> dict[str, dict[str, Any]]:
    overlap = pd.read_csv(PHASE1_DIR / "pred_gt_overlap_rows.csv")
    hold_overlap = overlap[overlap["dataset_split"].astype(str) == "holdout"].copy()
    best_gt: dict[str, dict[str, Any]] = {}
    for oid, sub in hold_overlap.groupby("mv_object_id"):
        idx = sub["IoU"].astype(float).idxmax()
        row = sub.loc[idx].to_dict()
        best_gt[str(oid)] = {
            "raw_gt_object_id": int(row["raw_gt_object_id"]),
            "best_gt_iou": float(row["IoU"]),
            "scene_id": str(row["scene_id"]),
            "chunk_id": str(row["chunk_id"]),
        }
    return best_gt


def _object_stats() -> dict[str, dict[str, Any]]:
    rows = pd.read_csv(PHASE1_DIR / "pred_object_fragment_rows.csv")
    rows = rows[rows["dataset_split"].astype(str) == "holdout"].copy()
    stats: dict[str, dict[str, Any]] = {}
    for row in rows.to_dict("records"):
        oid = str(row["mv_object_id"])
        stats[oid] = {
            "mean_mask_area_ratio": _num(row.get("mean_mask_area_ratio")),
            "broad_mask_share": _num(row.get("broad_mask_share")),
            "matched_GT_count": int(_num(row.get("matched_GT_count"))),
            "best_GT_IoU_diagnostic": _num(row.get("best_GT_IoU")),
            "semantic_residual_coherence_proxy": _num(row.get("semantic_residual_coherence")),
        }
    return stats


def _frame_maps() -> dict[str, dict[int, int]]:
    frame_rows = pd.read_parquet(PHASE2C_DIR / "mv_object_frame_mask_rows.parquet")
    frame_rows = frame_rows[frame_rows["dataset_split"].astype(str) == "holdout"].copy()
    maps: dict[str, dict[int, int]] = defaultdict(dict)
    for row in frame_rows.to_dict("records"):
        maps[str(row["mv_object_id"])][int(row["frame_id"])] = int(row["selected_mask_id"])
    return dict(maps)


def _pair_scores() -> dict[tuple[str, str], dict[str, Any]]:
    rows = pd.read_csv(PHASE2_DIR / "mask_pair_bridge_rows.csv")
    scores: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows.to_dict("records"):
        a = str(row["obj_i"])
        b = str(row["obj_j"])
        key = tuple(sorted([a, b]))
        score = _num(row.get("Bridge"))
        old = scores.get(key)
        if old is None or score > _num(old.get("Bridge")):
            scores[key] = {
                "Bridge": score,
                "shared_anchor_count": int(_num(row.get("shared_anchor_count"))),
                "anchor_family": row.get("anchor_family", ""),
            }
    return scores


def _build_universe() -> list[dict[str, Any]]:
    object_rows = pd.read_parquet(PHASE2C_DIR / "mv_object_rows.parquet")
    hold_obj = object_rows[object_rows["dataset_split"].astype(str) == "holdout"].copy()
    best_gt = _load_best_gt()
    obj_stats = _object_stats()
    frame_maps = _frame_maps()
    pair_scores = _pair_scores()

    by_scene_chunk: dict[tuple[str, str], list[str]] = defaultdict(list)
    obj_meta: dict[str, dict[str, Any]] = {}
    for row in hold_obj.to_dict("records"):
        oid = str(row["mv_object_id"])
        scene = str(row["scene_id"])
        chunk = str(row["chunk_id"])
        by_scene_chunk[(scene, chunk)].append(oid)
        obj_meta[oid] = {"scene_id": scene, "chunk_id": chunk}

    rows: list[dict[str, Any]] = []
    for scene in sorted({key[0] for key in by_scene_chunk}):
        chunks = sorted([key[1] for key in by_scene_chunk if key[0] == scene], key=_chunk_index)
        for left, right in zip(chunks[:-1], chunks[1:]):
            for a in by_scene_chunk[(scene, left)]:
                for b in by_scene_chunk[(scene, right)]:
                    ga = best_gt.get(a, {})
                    gb = best_gt.get(b, {})
                    valid_label = _num(ga.get("best_gt_iou")) >= 0.05 and _num(gb.get("best_gt_iou")) >= 0.05
                    same = bool(valid_label and ga.get("raw_gt_object_id") == gb.get("raw_gt_object_id"))
                    key = tuple(sorted([a, b]))
                    score_row = pair_scores.get(key, {})
                    frames_a = frame_maps.get(a, {})
                    frames_b = frame_maps.get(b, {})
                    common_frames = sorted(set(frames_a) & set(frames_b))
                    same_mask_frames = sum(1 for frame in common_frames if frames_a[frame] == frames_b[frame])
                    competing_frames = sum(1 for frame in common_frames if frames_a[frame] != frames_b[frame])
                    stats_a = obj_stats.get(a, {})
                    stats_b = obj_stats.get(b, {})
                    broad_a = _num(stats_a.get("broad_mask_share")) > 0.0 or _num(stats_a.get("mean_mask_area_ratio")) >= 0.20
                    broad_b = _num(stats_b.get("broad_mask_share")) > 0.0 or _num(stats_b.get("mean_mask_area_ratio")) >= 0.20
                    rows.append(
                        {
                            "schema_version": "stream4d_v101_phase2b_pair_diagnostic_row_v1",
                            "phase_id": "v101_phase2b_false_bridge_repair_probe",
                            "scene_id": scene,
                            "left_chunk_id": left,
                            "right_chunk_id": right,
                            "obj_i": a,
                            "obj_j": b,
                            "Bridge": _num(score_row.get("Bridge")),
                            "shared_anchor_count": int(_num(score_row.get("shared_anchor_count"))),
                            "anchor_family": score_row.get("anchor_family", ""),
                            "common_frame_count": len(common_frames),
                            "same_mask_overlap_frame_count": same_mask_frames,
                            "same_frame_competing_mask_count": competing_frames,
                            "obj_i_broad_proxy": bool(broad_a),
                            "obj_j_broad_proxy": bool(broad_b),
                            "obj_i_mean_mask_area_ratio": _num(stats_a.get("mean_mask_area_ratio")),
                            "obj_j_mean_mask_area_ratio": _num(stats_b.get("mean_mask_area_ratio")),
                            "obj_i_broad_mask_share": _num(stats_a.get("broad_mask_share")),
                            "obj_j_broad_mask_share": _num(stats_b.get("broad_mask_share")),
                            "obj_i_semantic_coherence_proxy": _num(stats_a.get("semantic_residual_coherence_proxy")),
                            "obj_j_semantic_coherence_proxy": _num(stats_b.get("semantic_residual_coherence_proxy")),
                            "semantic_coherence_proxy_abs_delta": abs(
                                _num(stats_a.get("semantic_residual_coherence_proxy"))
                                - _num(stats_b.get("semantic_residual_coherence_proxy"))
                            ),
                            "same_object_GT_diagnostic": same,
                            "obj_i_best_gt": ga.get("raw_gt_object_id", ""),
                            "obj_j_best_gt": gb.get("raw_gt_object_id", ""),
                            "obj_i_best_gt_iou": ga.get("best_gt_iou", ""),
                            "obj_j_best_gt_iou": gb.get("best_gt_iou", ""),
                            "uses_gt_for_prediction": False,
                            "uses_gt_for_diagnostic": True,
                        }
                    )
    return rows


def _passes_filter(row: dict[str, Any], config: dict[str, Any]) -> bool:
    if config.get("reject_same_frame_competing_mask") and int(row["same_frame_competing_mask_count"]) > 0:
        return False
    if config.get("require_same_mask_when_frame_overlap") and int(row["common_frame_count"]) > 0:
        if int(row["same_mask_overlap_frame_count"]) == 0:
            return False
    if config.get("object_like_only") and (bool(row["obj_i_broad_proxy"]) or bool(row["obj_j_broad_proxy"])):
        return False
    if int(row["shared_anchor_count"]) < int(config.get("shared_anchor_min", 0)):
        return False
    if _num(row["semantic_coherence_proxy_abs_delta"]) > _num(config.get("semantic_coherence_proxy_delta_max"), 999.0):
        return False
    return True


def _evaluate_config(rows: list[dict[str, Any]], config: dict[str, Any], tau: float) -> dict[str, Any]:
    same_total = sum(1 for row in rows if bool(row["same_object_GT_diagnostic"]))
    filtered_rows = [row for row in rows if _passes_filter(row, config)]
    selected = [row for row in filtered_rows if _num(row["Bridge"]) >= tau]
    same_selected = sum(1 for row in selected if bool(row["same_object_GT_diagnostic"]))
    diff_selected = len(selected) - same_selected
    same_scores = [_num(row["Bridge"]) for row in filtered_rows]
    labels = [1 if bool(row["same_object_GT_diagnostic"]) else 0 for row in filtered_rows]
    auc = _roc_auc(same_scores, labels)
    return {
        "schema_version": "stream4d_v101_phase2b_filter_curve_row_v1",
        "phase_id": "v101_phase2b_false_bridge_repair_probe",
        "provider_name": "G0_D4RT_reliable_anchors_only",
        "filter_name": config["filter_name"],
        "tau": tau,
        "universe_pair_count": len(rows),
        "filtered_pair_count": len(filtered_rows),
        "selected_pair_count": len(selected),
        "same_object_pair_universe_count": same_total,
        "same_object_selected_count": same_selected,
        "same_object_bridge_recall_at_tau": float(same_selected / max(1, same_total)),
        "false_bridge_rate_at_tau": float(diff_selected / max(1, len(selected))),
        "bridge_AUC_diagnostic_after_filter": auc if auc is not None else "",
        "passes_v101_bridge_gate": bool(
            (same_selected / max(1, same_total)) >= 0.35
            and (diff_selected / max(1, len(selected))) <= 0.20
            and (auc if auc is not None else 0.0) >= 0.65
        ),
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic": True,
        "filter_config_json": json.dumps(config, sort_keys=True),
    }


def _configs() -> list[dict[str, Any]]:
    return [
        {"filter_name": "R0_no_filter"},
        {"filter_name": "R1_same_frame_competing_mask_cannot_link", "reject_same_frame_competing_mask": True},
        {"filter_name": "R2_require_same_mask_on_overlap_frames", "require_same_mask_when_frame_overlap": True},
        {"filter_name": "R3_object_like_only_broad_proxy_removed", "object_like_only": True},
        {
            "filter_name": "R4_competing_cannot_link_plus_object_like_only",
            "reject_same_frame_competing_mask": True,
            "object_like_only": True,
        },
        {
            "filter_name": "R5_high_anchor_min128_plus_competing_cannot_link",
            "shared_anchor_min": 128,
            "reject_same_frame_competing_mask": True,
        },
        {
            "filter_name": "R6_high_anchor_min512_plus_competing_cannot_link",
            "shared_anchor_min": 512,
            "reject_same_frame_competing_mask": True,
        },
        {
            "filter_name": "R7_semantic_coherence_proxy_delta_le_0p10_diagnostic_only",
            "semantic_coherence_proxy_delta_max": 0.10,
        },
        {
            "filter_name": "R8_semantic_proxy_plus_competing_cannot_link",
            "semantic_coherence_proxy_delta_max": 0.10,
            "reject_same_frame_competing_mask": True,
        },
    ]


def _attempt_rows() -> list[dict[str, Any]]:
    return [
        {
            "schema_version": "stream4d_v101_phase2b_repair_attempt_row_v1",
            "phase_id": "v101_phase2b_false_bridge_repair_probe",
            "attempt_name": "same_frame_competing_mask_cannot_link",
            "plan_source": "v101 plan 7.8 false bridge repair",
            "gt_free_signal_available": True,
            "implementation": "Reject D4RT object-pair bridge when adjacent chunks share a global frame but selected CropFormer mask ids differ.",
            "artifact_columns": "mv_object_frame_mask_rows.parquet: frame_id, selected_mask_id",
        },
        {
            "schema_version": "stream4d_v101_phase2b_repair_attempt_row_v1",
            "phase_id": "v101_phase2b_false_bridge_repair_probe",
            "attempt_name": "split_broad_mask_support_from_object_like",
            "plan_source": "v101 plan 7.8 false bridge repair",
            "gt_free_signal_available": True,
            "implementation": "Reject pairs where either object has broad_mask_share>0 or mean_mask_area_ratio>=0.20 from Phase1 pred-object diagnostics.",
            "artifact_columns": "pred_object_fragment_rows.csv: broad_mask_share, mean_mask_area_ratio",
        },
        {
            "schema_version": "stream4d_v101_phase2b_repair_attempt_row_v1",
            "phase_id": "v101_phase2b_false_bridge_repair_probe",
            "attempt_name": "raise_semantic_residual_margin",
            "plan_source": "v101 plan 7.8 false bridge repair",
            "gt_free_signal_available": False,
            "implementation": "Not promoted: current artifacts expose only a scalar semantic_residual_coherence proxy, not pairwise residual descriptors.",
            "artifact_columns": "pred_object_fragment_rows.csv: semantic_residual_coherence_source=mean_v100_semantic_norm_proxy_not_pairwise_cosine",
        },
        {
            "schema_version": "stream4d_v101_phase2b_repair_attempt_row_v1",
            "phase_id": "v101_phase2b_false_bridge_repair_probe",
            "attempt_name": "depth_normal_discontinuity_barrier",
            "plan_source": "v101 plan 7.8 false bridge repair",
            "gt_free_signal_available": False,
            "implementation": "Not run: Phase2 D4RT proxy rows do not expose per-pair DA3 depth/normal boundary statistics.",
            "artifact_columns": "mask_pair_bridge_rows.csv lacks depth/normal discontinuity fields",
        },
    ]


def main() -> int:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    phase2 = _read_json(PHASE2_DIR / "summary.json")
    if not bool(phase2.get("phase2_completed")):
        raise RuntimeError("Phase2 summary is not complete; refusing Phase2b probe.")

    pair_rows = _build_universe()
    curve_rows: list[dict[str, Any]] = []
    for config in _configs():
        for tau in TAUS:
            curve_rows.append(_evaluate_config(pair_rows, config, tau))

    passing_rows = [row for row in curve_rows if bool(row["passes_v101_bridge_gate"])]
    best_at_010 = sorted(
        [row for row in curve_rows if abs(float(row["tau"]) - 0.10) < 1e-9],
        key=lambda r: (
            bool(r["passes_v101_bridge_gate"]),
            -_num(r["false_bridge_rate_at_tau"], 1.0),
            _num(r["same_object_bridge_recall_at_tau"]),
        ),
        reverse=True,
    )
    best_overall = sorted(
        curve_rows,
        key=lambda r: (
            bool(r["passes_v101_bridge_gate"]),
            -_num(r["false_bridge_rate_at_tau"], 1.0),
            _num(r["same_object_bridge_recall_at_tau"]),
        ),
        reverse=True,
    )

    failure_rows = [
        {
            "schema_version": "stream4d_v101_phase2b_failure_row_v1",
            "phase_id": "v101_phase2b_false_bridge_repair_probe",
            "failure_type": "depth_normal_discontinuity_barrier_unavailable",
            "reason": "Current Phase2 D4RT bridge proxy does not expose per-mask DA3 depth/normal discontinuity statistics.",
            "severity": "plan_7_8_repair_direction_blocked_by_artifact_schema",
        },
        {
            "schema_version": "stream4d_v101_phase2b_failure_row_v1",
            "phase_id": "v101_phase2b_false_bridge_repair_probe",
            "failure_type": "pairwise_semantic_residual_descriptor_unavailable",
            "reason": "Only scalar semantic coherence proxy is available; no pairwise semantic residual descriptor can be used as a method margin.",
            "severity": "plan_7_8_repair_direction_blocked_by_artifact_schema",
        },
    ]
    if not passing_rows:
        failure_rows.append(
            {
                "schema_version": "stream4d_v101_phase2b_failure_row_v1",
                "phase_id": "v101_phase2b_false_bridge_repair_probe",
                "failure_type": "false_bridge_repair_probe_no_gate_pass",
                "reason": "Pre-registered GT-free filters did not satisfy recall>=0.35, false_bridge<=0.20, and AUC>=0.65 together.",
                "severity": "blocks_phase3_fragment_repair",
            }
        )

    files = {
        "pair_diagnostic_rows": OUT_DIR / "pair_diagnostic_rows.csv",
        "filter_curve_rows": OUT_DIR / "filter_curve_rows.csv",
        "repair_attempt_rows": OUT_DIR / "repair_attempt_rows.csv",
        "failure_rows": OUT_DIR / "failure_rows.csv",
    }
    _write_csv(files["pair_diagnostic_rows"], pair_rows)
    _write_csv(files["filter_curve_rows"], curve_rows)
    _write_csv(files["repair_attempt_rows"], _attempt_rows())
    _write_csv(files["failure_rows"], failure_rows)

    summary = {
        "schema_version": "stream4d_v101_phase2b_false_bridge_repair_probe_summary_v1",
        "phase_id": "v101_phase2b_false_bridge_repair_probe",
        "created_unix_time": time.time(),
        "runtime_sec": time.time() - t0,
        "phase2b_completed": True,
        "decision": "PASS_FALSE_BRIDGE_FILTER_FOUND_ENTER_PHASE3"
        if passing_rows
        else "NO_GO_FALSE_BRIDGE_REPAIR_FILTERS_STILL_BLOCK_PHASE3",
        "provider_bridge_potential_confirmed_after_repair": bool(passing_rows),
        "pair_universe_count": len(pair_rows),
        "same_object_pair_universe_count": sum(1 for row in pair_rows if bool(row["same_object_GT_diagnostic"])),
        "filter_curve_row_count": len(curve_rows),
        "passing_filter_count": len(passing_rows),
        "best_tau0p10_row": best_at_010[0] if best_at_010 else {},
        "best_overall_row": best_overall[0] if best_overall else {},
        "repair_attempt_count": 4,
        "repair_attempts_with_available_gt_free_signal": 2,
        "artifact_schema_blockers": [
            "pairwise semantic residual descriptor unavailable",
            "per-pair depth/normal discontinuity unavailable",
        ],
        "analysis": {
            "reason": "Phase2 D4RT bridge proxy had high false bridge; Phase2b tested plan 7.8 GT-free filters before stopping.",
            "main_blocker": "No tested filter produced v101 bridge potential, so Phase3 fragment repair is still not allowed.",
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic": True,
        },
        "outputs": {name: _rel(path) for name, path in files.items()},
        "summary": _rel(OUT_DIR / "summary.json"),
    }
    _write_json(OUT_DIR / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if passing_rows else 2


if __name__ == "__main__":
    sys.exit(main())
