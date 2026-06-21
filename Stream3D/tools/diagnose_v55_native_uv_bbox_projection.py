from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from stream4d_native.v47_common import ROOT, parse_int, read_csv, read_json
from stream4d_native.v55_history_update import (
    _bbox_iou,
    _center_distance,
    _dominant,
    _load_list,
    _objectlet_frame_projection_stats,
    _support_component_gt,
)


def _project(path: str | Path) -> Path:
    path_obj = Path(path)
    return path_obj if path_obj.is_absolute() else ROOT / path_obj


def _candidate_dominant_gt(
    *,
    scene: str,
    component_ids: set[str],
    component_gt: dict[tuple[str, str], Counter[str]],
) -> str | None:
    counter: Counter[str] = Counter()
    for component_id in component_ids:
        counter.update(component_gt.get((scene, component_id), Counter()))
    return _dominant(counter)


def _score_candidates(
    *,
    role_rows: list[dict[str, Any]],
    anchor_birth_rows: list[dict[str, Any]],
    objectlet_rows: list[dict[str, Any]],
    best_variant: str,
    evidence_roles: set[str],
    projection_stats: dict[str, dict[int, dict[str, float]]],
    component_gt: dict[tuple[str, str], Counter[str]],
) -> list[dict[str, Any]]:
    histories: dict[str, dict[str, Any]] = {}
    for row in anchor_birth_rows:
        if str(row.get("accepted_birth")).lower() != "true":
            continue
        scene = str(row.get("scene"))
        history_id = str(row.get("birth_object_id"))
        history_gt = _candidate_dominant_gt(
            scene=scene,
            component_ids=set(_load_list(row.get("component_ids"))),
            component_gt=component_gt,
        )
        histories[history_id] = {"scene": scene, "dominant_gt": history_gt}

    histories_by_scene: dict[str, list[str]] = defaultdict(list)
    for history_id, history in histories.items():
        histories_by_scene[str(history["scene"])].append(history_id)

    evidence_chunks = {str(row.get("chunk_id")) for row in role_rows if str(row.get("role")) in evidence_roles}
    candidates = [
        row for row in objectlet_rows if str(row.get("variant")) == best_variant and str(row.get("chunk_id")) in evidence_chunks
    ]
    scored_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        scene = str(candidate.get("scene"))
        objectlet_id = str(candidate.get("objectlet_id") or "")
        candidate_stats = projection_stats.get(objectlet_id, {})
        if not candidate_stats:
            continue
        candidate_support = sum(frame_stats["n"] for frame_stats in candidate_stats.values())
        scored: list[tuple[float, int, float, float, float, float, float, str]] = []
        for history_id in histories_by_scene.get(scene, []):
            history_stats = projection_stats.get(history_id, {})
            shared_frame_ids = set(candidate_stats) & set(history_stats)
            if not shared_frame_ids:
                continue
            uv_support = 0.0
            weighted_iou = 0.0
            min_center_dist = float("inf")
            accepted_frame_count = 0
            for frame_id in shared_frame_ids:
                candidate_frame = candidate_stats[frame_id]
                history_frame = history_stats[frame_id]
                iou = _bbox_iou(candidate_frame, history_frame)
                center_dist = _center_distance(candidate_frame, history_frame)
                if iou <= 0.0 and center_dist > 0.25:
                    continue
                frame_support = min(candidate_frame["n"], history_frame["n"])
                uv_support += frame_support
                weighted_iou += iou * frame_support
                min_center_dist = min(min_center_dist, center_dist)
                accepted_frame_count += 1
            if uv_support <= 0.0:
                continue
            history_support = sum(frame_stats["n"] for frame_stats in history_stats.values())
            candidate_ratio = float(uv_support / max(candidate_support, 1.0))
            history_ratio = float(uv_support / max(history_support, 1.0))
            uv_jaccard = float(uv_support / max(candidate_support + history_support - uv_support, 1.0))
            mean_iou = float(weighted_iou / max(uv_support, 1.0))
            scored.append(
                (
                    uv_support,
                    accepted_frame_count,
                    candidate_ratio,
                    history_ratio,
                    uv_jaccard,
                    mean_iou,
                    min_center_dist,
                    history_id,
                )
            )
        if not scored:
            continue
        scored.sort(key=lambda item: (-item[0], -item[2], -item[5], item[6], item[7]))
        uv_support, shared_frames, candidate_ratio, history_ratio, uv_jaccard, mean_iou, min_center_dist, history_id = scored[0]
        candidate_gt = _candidate_dominant_gt(
            scene=scene,
            component_ids=set(_load_list(candidate.get("component_ids"))),
            component_gt=component_gt,
        )
        history_gt = histories[history_id]["dominant_gt"]
        scored_rows.append(
            {
                "scene": scene,
                "objectlet_id": objectlet_id,
                "chunk_id": str(candidate.get("chunk_id")),
                "history_id": history_id,
                "uv_support": uv_support,
                "shared_frames": shared_frames,
                "candidate_ratio": candidate_ratio,
                "history_ratio": history_ratio,
                "uv_jaccard": uv_jaccard,
                "mean_iou": mean_iou,
                "min_center_dist": min_center_dist,
                "candidate_dominant_gt": candidate_gt,
                "history_dominant_gt": history_gt,
                "diagnostic_hit": bool(candidate_gt and history_gt and candidate_gt == history_gt),
            }
        )
    return scored_rows


def _summary_for(rows: list[dict[str, Any]], *, label: str, filters: dict[str, float]) -> dict[str, Any]:
    kept: list[dict[str, Any]] = []
    for row in rows:
        if row["uv_support"] < filters.get("min_support", 0.0):
            continue
        if row["candidate_ratio"] < filters.get("min_candidate_ratio", 0.0):
            continue
        if row["uv_jaccard"] < filters.get("min_jaccard", 0.0):
            continue
        if row["mean_iou"] < filters.get("min_mean_iou", 0.0):
            continue
        if row["min_center_dist"] > filters.get("max_center_dist", float("inf")):
            continue
        if row["shared_frames"] < filters.get("min_shared_frames", 0.0):
            continue
        kept.append(row)
    hits = sum(1 for row in kept if row["diagnostic_hit"])
    return {
        "label": label,
        **filters,
        "count": len(kept),
        "diagnostic_precision": None if not kept else float(hits / len(kept)),
    }


def diagnose(args: argparse.Namespace) -> dict[str, Any]:
    role_rows = read_csv(_project(args.chunk_role_rows))
    anchor_birth_rows = read_csv(_project(args.anchor_birth_rows))
    objectlet_rows = read_csv(_project(args.objectlet_rows))
    local_summary = read_json(_project(args.local_summary))
    best_variant = str(local_summary.get("best_method_variant") or "")
    evidence_roles = {role.strip() for role in args.history_evidence_roles.split(",") if role.strip()}
    scenes = {str(row.get("scene")) for row in role_rows}
    component_gt = _support_component_gt(
        _project(args.support_rows),
        support_variant=args.support_variant,
        scenes=scenes,
    )
    accepted_history_ids = {
        str(row.get("birth_object_id"))
        for row in anchor_birth_rows
        if str(row.get("accepted_birth")).lower() == "true"
    }
    evidence_chunks = {str(row.get("chunk_id")) for row in role_rows if str(row.get("role")) in evidence_roles}
    candidate_ids = {
        str(row.get("objectlet_id") or "")
        for row in objectlet_rows
        if str(row.get("variant")) == best_variant and str(row.get("chunk_id")) in evidence_chunks
    }
    needed_objectlets = {objectlet_id for objectlet_id in candidate_ids | accepted_history_ids if objectlet_id}
    projection_stats = _objectlet_frame_projection_stats(_project(args.native_carrier_rows), needed_objectlets)
    rows = _score_candidates(
        role_rows=role_rows,
        anchor_birth_rows=anchor_birth_rows,
        objectlet_rows=objectlet_rows,
        best_variant=best_variant,
        evidence_roles=evidence_roles,
        projection_stats=projection_stats,
        component_gt=component_gt,
    )
    threshold_summaries = [
        _summary_for(rows, label="uv_support_ge_1", filters={"min_support": 1.0}),
        _summary_for(rows, label="uv_support_ge_20", filters={"min_support": 20.0}),
        _summary_for(rows, label="uv_support_ge_100", filters={"min_support": 100.0}),
        _summary_for(rows, label="mean_iou_ge_005", filters={"min_mean_iou": 0.05}),
        _summary_for(rows, label="min_center_le_008", filters={"max_center_dist": 0.08}),
        _summary_for(
            rows,
            label="u5_selected_thresholds",
            filters={
                "min_support": 20.0,
                "min_candidate_ratio": 0.10,
                "min_jaccard": 0.0,
                "min_mean_iou": 0.05,
                "max_center_dist": 0.10,
                "min_shared_frames": 3.0,
            },
        ),
    ]
    grid: list[dict[str, Any]] = []
    for min_support in (20.0, 100.0):
        for min_candidate_ratio in (0.0, 0.05, 0.10):
            for min_mean_iou in (0.0, 0.05):
                for max_center_dist in (0.08, 0.10):
                    for min_shared_frames in (1.0, 3.0):
                        grid.append(
                            _summary_for(
                                rows,
                                label="grid",
                                filters={
                                    "min_support": min_support,
                                    "min_candidate_ratio": min_candidate_ratio,
                                    "min_jaccard": 0.0,
                                    "min_mean_iou": min_mean_iou,
                                    "max_center_dist": max_center_dist,
                                    "min_shared_frames": min_shared_frames,
                                },
                            )
                        )
    grid.sort(
        key=lambda row: (
            -(row["diagnostic_precision"] if row["diagnostic_precision"] is not None else -1.0),
            -row["count"],
            row["min_support"],
            row["min_candidate_ratio"],
            row["min_mean_iou"],
            row["max_center_dist"],
            row["min_shared_frames"],
        )
    )
    return {
        "phase": "v55_native_uv_bbox_projection_diagnostic",
        "native_carrier_rows": str(_project(args.native_carrier_rows)),
        "best_variant": best_variant,
        "history_evidence_roles": sorted(evidence_roles),
        "accepted_anchor_history_count": len(accepted_history_ids),
        "candidate_objectlet_count": len(candidate_ids),
        "native_rows_used_objectlet_count": len(projection_stats),
        "scored_candidate_count": len(rows),
        "threshold_summaries": threshold_summaries,
        "top_grid": grid[: int(args.top_grid)],
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose v55 native UV/bbox projection candidate evidence.")
    parser.add_argument("--chunk-role-rows", default="outputs/audit/v55_chunk_roles/chunk_role_rows.csv")
    parser.add_argument("--anchor-birth-rows", default="outputs/audit/v55_anchor_birth/anchor_birth_rows.csv")
    parser.add_argument(
        "--objectlet-rows",
        default="outputs/audit/v54_local_objectlets_k0all_conflict_veto018_skip_repeated_sig_l12_stride1_probe5_q4096_notopup_max4000_skip/objectlet_rows.csv",
    )
    parser.add_argument(
        "--local-summary",
        default="outputs/audit/v54_local_reproduction_stride1_probe5_q4096_notopup_k0all_max4000_veto018_w123_fullchunk/local_reproduction_summary.json",
    )
    parser.add_argument(
        "--support-rows",
        default="outputs/audit/v54_mask_component_support_tau005_stride1_probe5_q4096_notopup/mask_component_support_rows.csv",
    )
    parser.add_argument("--support-variant", default="R0_visible_tau0.05")
    parser.add_argument(
        "--native-carrier-rows",
        default="outputs/audit/v55_native_carrier_materialization_q4096_l11/objectlet_native_carrier_rows.csv",
    )
    parser.add_argument("--history-evidence-roles", default="bridge,update")
    parser.add_argument("--top-grid", type=int, default=10)
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()
    payload = diagnose(args)
    if args.output_json:
        output_path = _project(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
