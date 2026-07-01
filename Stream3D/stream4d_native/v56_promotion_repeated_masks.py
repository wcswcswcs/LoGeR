from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .v47_common import ROOT, parse_bool, parse_int, read_csv, read_json, utc_now, write_csv, write_json
from .v55_history_update import (
    _build_assignment_map,
    _dominant,
    _metrics_from_support,
    _next_history_by_scene,
    _support_component_gt,
)


def _project(path: str | Path) -> Path:
    path_obj = Path(path)
    return path_obj if path_obj.is_absolute() else ROOT / path_obj


def _rel(path: str | Path) -> str:
    path_obj = _project(path)
    try:
        return str(path_obj.relative_to(ROOT))
    except ValueError:
        return str(path_obj)


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _history_components(path: Path) -> dict[str, dict[str, Any]]:
    histories: dict[str, dict[str, Any]] = {}
    for row in read_csv(path):
        history_id = str(row.get("history_id"))
        scene = str(row.get("scene"))
        history = histories.setdefault(
            history_id,
            {
                "history_id": history_id,
                "scene": scene,
                "history_components": set(),
                "dominant_gt": str(row.get("history_dominant_gt_diagnostic") or ""),
                "chunks": set(),
            },
        )
        history["history_components"].add(str(row.get("component_id")))
        if not history.get("dominant_gt") and row.get("history_dominant_gt_diagnostic"):
            history["dominant_gt"] = str(row.get("history_dominant_gt_diagnostic"))
    return histories


def _multi_history_components(histories: dict[str, dict[str, Any]]) -> set[tuple[str, str]]:
    history_ids_by_component: dict[tuple[str, str], set[str]] = defaultdict(set)
    for history_id, history in histories.items():
        scene = str(history["scene"])
        for component_id in history["history_components"]:
            history_ids_by_component[(scene, str(component_id))].add(history_id)
    return {
        component_key
        for component_key, history_ids in history_ids_by_component.items()
        if len(history_ids) > 1
    }


def _chunk_lookup(
    chunk_rows: list[dict[str, Any]],
    role_rows: list[dict[str, Any]],
    roles: set[str],
) -> dict[tuple[str, int], str]:
    role_by_chunk = {str(row.get("chunk_id")): str(row.get("role")) for row in role_rows}
    lookup: dict[tuple[str, int], str] = {}
    for row in chunk_rows:
        chunk_id = str(row.get("chunk_id"))
        if role_by_chunk.get(chunk_id) not in roles:
            continue
        scene = str(row.get("scene"))
        start = parse_int(row.get("raw_frame_start"))
        end = parse_int(row.get("raw_frame_end"))
        for frame_id in range(start, end + 1):
            lookup[(scene, frame_id)] = chunk_id
    return lookup


def build_v56_repeated_mask_promotion(
    *,
    core_history_component_rows_path: str | Path = "outputs/audit/v56_reuse_core_update_C3_e1_boundary_uv/history_component_rows.csv",
    selected_history_component_rows_path: str | Path = "outputs/audit/v55_history_update_repair_native_boundary_uv_historymask_s100_r09_d15_plus_cosupport_seed038_componentrows_probe/history_component_rows.csv",
    core_history_update_rows_path: str | Path = "outputs/audit/v56_reuse_core_update_C3_e1_boundary_uv/history_update_rows.csv",
    core_summary_path: str | Path = "outputs/audit/v56_reuse_core_update_C3_e1_boundary_uv/history_update_summary.json",
    anchor_birth_rows_path: str | Path = "outputs/audit/v55_anchor_birth/anchor_birth_rows.csv",
    chunk_role_rows_path: str | Path = "outputs/audit/v55_chunk_roles/chunk_role_rows.csv",
    chunk_rows_path: str | Path = "outputs/audit/v54_chunk_universe_stride1_probe5_q4096_notopup/chunk_rows.csv",
    support_rows_path: str | Path = "outputs/audit/v54_mask_component_support_tau005_stride1_probe5_q4096_notopup/mask_component_support_rows.csv",
    support_variant: str = "R0_visible_tau0.05",
    evidence_roles: tuple[str, ...] = ("bridge", "update"),
    min_independent_chunks: int = 2,
    min_co_support_masks: int = 2,
    min_component_support_count: int = 1,
    require_no_competing_history: bool = False,
    max_competing_history_count: int = 0,
    exclude_multi_history_tentative_components: bool = False,
) -> dict[str, Any]:
    core_histories = _history_components(_project(core_history_component_rows_path))
    selected_histories = _history_components(_project(selected_history_component_rows_path))
    multi_history_components = _multi_history_components(selected_histories)
    core_summary = read_json(_project(core_summary_path))
    role_rows = read_csv(_project(chunk_role_rows_path))
    chunk_rows = read_csv(_project(chunk_rows_path))
    evidence_role_set = {str(role) for role in evidence_roles if str(role)}
    chunk_by_scene_frame = _chunk_lookup(chunk_rows, role_rows, evidence_role_set)
    scenes = {str(history["scene"]) for history in core_histories.values()}
    component_gt = _support_component_gt(_project(support_rows_path), support_variant=support_variant, scenes=scenes)

    for row in read_csv(_project(anchor_birth_rows_path)):
        if not parse_bool(row.get("accepted_birth")):
            continue
        history_id = str(row.get("birth_object_id"))
        if history_id in core_histories:
            core_histories[history_id]["chunks"].add(str(row.get("anchor_chunk_id")))
    for row in read_csv(_project(core_history_update_rows_path)):
        if str(row.get("update_state")) == "confirmed_update" and str(row.get("history_id")) in core_histories:
            core_histories[str(row.get("history_id"))]["chunks"].add(str(row.get("chunk_id")))

    tentative_components: dict[tuple[str, str, str], dict[str, Any]] = {}
    core_component_to_histories: dict[tuple[str, str], set[str]] = defaultdict(set)
    for history_id, history in core_histories.items():
        scene = str(history["scene"])
        for component_id in history["history_components"]:
            core_component_to_histories[(scene, component_id)].add(history_id)
    for history_id, selected in selected_histories.items():
        if history_id not in core_histories:
            continue
        scene = str(selected["scene"])
        core_components = core_histories[history_id]["history_components"]
        for component_id in selected["history_components"] - core_components:
            tentative_components[(history_id, scene, component_id)] = {
                "history_id": history_id,
                "scene": scene,
                "component_id": component_id,
            }

    candidate_by_component: dict[tuple[str, str], list[tuple[str, str, str]]] = defaultdict(list)
    for key in tentative_components:
        _history_id, scene, component_id = key
        candidate_by_component[(scene, component_id)].append(key)

    core_masks_by_history: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
    core_histories_by_mask: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    candidate_masks: dict[tuple[str, str, str], set[tuple[str, str, str]]] = defaultdict(set)
    candidate_support_by_mask: dict[tuple[str, str, str], Counter[tuple[str, str, str]]] = defaultdict(Counter)
    with _project(support_rows_path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("variant")) != support_variant:
                continue
            support_count = max(parse_int(row.get("support_count")), 1)
            if support_count < int(min_component_support_count):
                continue
            scene = str(row.get("scene"))
            frame_id = parse_int(row.get("frame_id"))
            chunk_id = chunk_by_scene_frame.get((scene, frame_id))
            if not chunk_id:
                continue
            mask_key = (scene, chunk_id, str(row.get("mask_observation_id")))
            component_id = str(row.get("component_id"))
            for history_id in core_component_to_histories.get((scene, component_id), set()):
                core_masks_by_history[history_id].add(mask_key)
                core_histories_by_mask[mask_key].add(history_id)
            for candidate_key in candidate_by_component.get((scene, component_id), []):
                candidate_masks[candidate_key].add(mask_key)
                candidate_support_by_mask[candidate_key][mask_key] += support_count

    promoted_by_history: dict[str, set[str]] = defaultdict(set)
    promotion_rows: list[dict[str, Any]] = []
    precision_hits = 0
    precision_total = 0
    source_breakdown: Counter[str] = Counter()
    competing_reject_count = 0
    repeated_mask_reject_count = 0
    missing_support_reject_count = 0
    quarantine_reject_count = 0
    for key, candidate in sorted(tentative_components.items()):
        history_id, scene, component_id = key
        is_quarantine_candidate = (
            bool(exclude_multi_history_tentative_components)
            and (scene, component_id) in multi_history_components
        )
        masks = candidate_masks.get(key, set())
        co_support_masks = masks & core_masks_by_history.get(history_id, set())
        co_support_chunks = {chunk_id for _scene, chunk_id, _mask_id in co_support_masks}
        competing_histories = {
            other_history
            for mask_key in masks
            for other_history in core_histories_by_mask.get(mask_key, set())
            if other_history != history_id
        }
        same_component_core_histories = {
            other_history
            for other_history in core_component_to_histories.get((scene, component_id), set())
            if other_history != history_id
        }
        competing_count = len(competing_histories | same_component_core_histories)
        repeated_pass = len(co_support_chunks) >= int(min_independent_chunks) and len(co_support_masks) >= int(
            min_co_support_masks
        )
        competing_pass = (not require_no_competing_history) or competing_count <= int(max_competing_history_count)
        state = "promoted" if repeated_pass and competing_pass and not is_quarantine_candidate else "not_promoted"
        reason = "P2_repeated_independent_masks"
        if is_quarantine_candidate:
            reason = "quarantine_multi_history_component"
            quarantine_reject_count += 1
        elif not masks:
            reason = "no_evidence_chunk_mask_support"
            missing_support_reject_count += 1
        elif not repeated_pass:
            reason = "insufficient_repeated_core_cosupport"
            repeated_mask_reject_count += 1
        elif not competing_pass:
            reason = "competing_history_core_cosupport"
            competing_reject_count += 1
        if state == "promoted":
            promoted_by_history[history_id].add(component_id)
            source_breakdown["P2_repeated_independent_masks"] += 1
            history_gt = core_histories[history_id].get("dominant_gt")
            component_label = _dominant(component_gt.get((scene, component_id), Counter()))
            if history_gt and component_label:
                precision_total += 1
                if str(history_gt) == str(component_label):
                    precision_hits += 1
        promotion_rows.append(
            {
                "history_id": history_id,
                "scene": scene,
                "component_id": component_id,
                "candidate_state": "tentative",
                "promotion_state": state,
                "reason": reason,
                "co_support_mask_count": len(co_support_masks),
                "co_support_chunk_count": len(co_support_chunks),
                "candidate_evidence_mask_count": len(masks),
                "candidate_evidence_chunk_count": len({chunk_id for _scene, chunk_id, _mask_id in masks}),
                "competing_history_count": competing_count,
                "is_quarantine_candidate": is_quarantine_candidate,
                "support_count_sum": int(sum(candidate_support_by_mask.get(key, Counter()).values())),
                "history_dominant_gt_diagnostic": core_histories[history_id].get("dominant_gt"),
                "component_dominant_gt_diagnostic": _dominant(component_gt.get((scene, component_id), Counter())),
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic_labels": True,
            }
        )

    promoted_histories: dict[str, dict[str, Any]] = {}
    for history_id, history in core_histories.items():
        promoted_histories[history_id] = {
            **history,
            "history_components": set(history["history_components"]) | promoted_by_history.get(history_id, set()),
            "chunks": set(history["chunks"]),
        }
    for row in promotion_rows:
        if row["promotion_state"] == "promoted":
            for mask_key in candidate_masks.get((row["history_id"], row["scene"], row["component_id"]), set()):
                promoted_histories[row["history_id"]]["chunks"].add(mask_key[1])

    core_assignment, core_duplicate_count = _build_assignment_map(core_histories, "history_components")
    promoted_assignment, promoted_duplicate_count = _build_assignment_map(promoted_histories, "history_components")
    core_metrics = _metrics_from_support(
        _project(support_rows_path), support_variant=support_variant, scenes=scenes, component_to_history=core_assignment
    )
    promoted_metrics = _metrics_from_support(
        _project(support_rows_path),
        support_variant=support_variant,
        scenes=scenes,
        component_to_history=promoted_assignment,
    )

    shuffled_histories: dict[str, dict[str, Any]] = {}
    for history_id, history in core_histories.items():
        shuffled_histories[history_id] = {**history, "history_components": set(history["history_components"])}
    next_history = _next_history_by_scene(core_histories)
    for history_id, components in promoted_by_history.items():
        target_id = next_history.get(history_id)
        if target_id:
            shuffled_histories[target_id]["history_components"].update(components)
    shuffled_assignment, shuffled_duplicate_count = _build_assignment_map(shuffled_histories, "history_components")
    shuffled_metrics = _metrics_from_support(
        _project(support_rows_path),
        support_variant=support_variant,
        scenes=scenes,
        component_to_history=shuffled_assignment,
    )

    promoted_count = sum(len(components) for components in promoted_by_history.values())
    precision = float(precision_hits / max(precision_total, 1)) if precision_total else None
    real_minus_shuffled = float(promoted_metrics["ARI"] - shuffled_metrics["ARI"])
    real_minus_no_temporal = float(promoted_metrics["ARI"] - core_metrics["ARI"])
    core_real_minus_shuffled = float(core_summary.get("real_minus_shuffled_ARI", 0.0))
    core_real_minus_no_temporal = float(core_summary.get("real_minus_no_temporal_ARI", 0.0))
    temporal_spans = [float(len(history["chunks"])) for history in promoted_histories.values()]
    summary = {
        "phase": "v56_repeated_mask_promotion",
        "created_at": utc_now(),
        "input_paths": {
            "core_history_component_rows_path": _rel(core_history_component_rows_path),
            "selected_history_component_rows_path": _rel(selected_history_component_rows_path),
            "support_rows_path": _rel(support_rows_path),
        },
        "min_independent_chunks": int(min_independent_chunks),
        "min_co_support_masks": int(min_co_support_masks),
        "min_component_support_count": int(min_component_support_count),
        "require_no_competing_history": bool(require_no_competing_history),
        "max_competing_history_count": int(max_competing_history_count),
        "exclude_multi_history_tentative_components": bool(exclude_multi_history_tentative_components),
        "promotion_candidate_count": len(tentative_components),
        "eligible_promotion_candidate_count": len(tentative_components) - quarantine_reject_count,
        "quarantine_candidate_reject_count": quarantine_reject_count,
        "promoted_component_count": promoted_count,
        "promotion_precision_diagnostic": precision,
        "false_promotion_count": None if precision is None else int(precision_total - precision_hits),
        "promotion_source_breakdown": dict(source_breakdown),
        "missing_support_reject_count": missing_support_reject_count,
        "repeated_mask_reject_count": repeated_mask_reject_count,
        "competing_reject_count": competing_reject_count,
        "confirmed_core_ARI": promoted_metrics["ARI"],
        "confirmed_core_purity": promoted_metrics["purity"],
        "confirmed_core_completeness": promoted_metrics["completeness"],
        "confirmed_core_temporal_span": _mean(temporal_spans),
        "P0_core_ARI": core_metrics["ARI"],
        "P0_core_purity": core_metrics["purity"],
        "P0_core_completeness": core_metrics["completeness"],
        "confirmed_core_completeness_gain_vs_P0": float(promoted_metrics["completeness"] - core_metrics["completeness"]),
        "confirmed_core_purity_drop_vs_P0": float(core_metrics["purity"] - promoted_metrics["purity"]),
        "real_minus_shuffled_ARI": real_minus_shuffled,
        "real_minus_no_temporal_ARI": real_minus_no_temporal,
        "real_minus_shuffled_ARI_gain_vs_P0": float(real_minus_shuffled - core_real_minus_shuffled),
        "real_minus_no_temporal_ARI_gain_vs_P0": float(real_minus_no_temporal - core_real_minus_no_temporal),
        "shuffled_history_ARI": shuffled_metrics["ARI"],
        "core_duplicate_component_count": core_duplicate_count,
        "promoted_duplicate_component_count": promoted_duplicate_count,
        "shuffled_duplicate_component_count": shuffled_duplicate_count,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }
    gate = {
        "promotion_precision_diagnostic_ge_0.85": (precision or 0.0) >= 0.85,
        "promoted_component_count_gt_0": promoted_count > 0,
        "confirmed_core_completeness_gain_vs_P0_ge_0.02": summary["confirmed_core_completeness_gain_vs_P0"] >= 0.02,
        "confirmed_core_purity_drop_vs_P0_le_0.005": summary["confirmed_core_purity_drop_vs_P0"] <= 0.005,
        "real_minus_shuffled_ARI_gain_vs_P0_ge_0.03": summary["real_minus_shuffled_ARI_gain_vs_P0"] >= 0.03,
        "real_minus_no_temporal_ARI_gain_vs_P0_ge_0.02": summary["real_minus_no_temporal_ARI_gain_vs_P0"] >= 0.02,
    }
    gate["pass"] = bool(all(gate.values()))
    summary["gate"] = gate
    metric_rows = [
        {"row": "P0_no_promotion_core", **core_metrics, "real_minus_shuffled_ARI": core_real_minus_shuffled, "real_minus_no_temporal_ARI": core_real_minus_no_temporal},
        {"row": "P2_repeated_mask_promotion", **promoted_metrics, "real_minus_shuffled_ARI": real_minus_shuffled, "real_minus_no_temporal_ARI": real_minus_no_temporal},
        {"row": "P7_shuffled_promotion_control", **shuffled_metrics},
    ]
    return {"summary": summary, "promotion_rows": promotion_rows, "promotion_metric_rows": metric_rows}


def write_v56_repeated_mask_promotion(output_root: str | Path, payload: dict[str, Any]) -> None:
    out = _project(output_root)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "promotion_summary.json", payload["summary"])
    write_csv(out / "promotion_rows.csv", payload["promotion_rows"])
    write_csv(out / "promotion_metric_rows.csv", payload["promotion_metric_rows"])
