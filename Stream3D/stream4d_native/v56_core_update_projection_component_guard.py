from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .v47_common import ROOT, parse_bool, parse_float, parse_int, read_csv, read_json, utc_now, write_csv, write_json
from .v55_history_update import (
    _build_assignment_map,
    _component_atom_map,
    _dominant,
    _load_list,
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


def _history_state(
    anchor_birth_rows: list[dict[str, Any]],
    component_to_atom: dict[tuple[str, str], str],
    component_gt: dict[tuple[str, str], Counter[str]],
) -> dict[str, dict[str, Any]]:
    histories: dict[str, dict[str, Any]] = {}
    for row in anchor_birth_rows:
        if not parse_bool(row.get("accepted_birth")):
            continue
        scene = str(row.get("scene"))
        history_id = str(row.get("birth_object_id"))
        components = set(_load_list(row.get("component_ids")))
        atoms = {component_to_atom.get((scene, component_id), "") for component_id in components}
        atoms = {atom for atom in atoms if atom}
        gt_counter: Counter[str] = Counter()
        for component_id in components:
            gt_counter.update(component_gt.get((scene, component_id), Counter()))
        histories[history_id] = {
            "history_id": history_id,
            "scene": scene,
            "anchor_chunk_id": str(row.get("anchor_chunk_id")),
            "anchor_components": set(components),
            "history_components": set(components),
            "anchor_atoms": set(atoms),
            "history_atoms": set(atoms),
            "chunks": {str(row.get("anchor_chunk_id"))},
            "dominant_gt": _dominant(gt_counter),
        }
    return histories


def _native_projection_maps(
    native_rows_path: Path,
    needed_objectlets: set[str],
) -> tuple[
    dict[str, Counter[tuple[int, int]]],
    dict[str, dict[str, Counter[tuple[int, int]]]],
    dict[str, Counter[int]],
    dict[str, dict[str, Counter[int]]],
]:
    objectlet_frame_masks: dict[str, Counter[tuple[int, int]]] = defaultdict(Counter)
    component_frame_masks: dict[str, dict[str, Counter[tuple[int, int]]]] = defaultdict(lambda: defaultdict(Counter))
    objectlet_frames: dict[str, Counter[int]] = defaultdict(Counter)
    component_frames: dict[str, dict[str, Counter[int]]] = defaultdict(lambda: defaultdict(Counter))
    with native_rows_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            objectlet_id = str(row.get("objectlet_id") or "")
            if objectlet_id not in needed_objectlets:
                continue
            if not parse_bool(row.get("visible")) or not parse_bool(row.get("valid_uv")):
                continue
            frame_id = parse_int(row.get("frame_id"))
            observed_mask_id = parse_int(row.get("observed_mask_id"))
            component_id = str(row.get("component_id") or "")
            objectlet_frames[objectlet_id][frame_id] += 1
            if component_id:
                component_frames[objectlet_id][component_id][frame_id] += 1
            if observed_mask_id <= 0:
                continue
            key = (frame_id, observed_mask_id)
            objectlet_frame_masks[objectlet_id][key] += 1
            if component_id:
                component_frame_masks[objectlet_id][component_id][key] += 1
    return objectlet_frame_masks, component_frame_masks, objectlet_frames, component_frames


def build_v56_projection_component_guard_core_update(
    *,
    anchor_birth_rows_path: str | Path = "outputs/audit/v55_anchor_birth/anchor_birth_rows.csv",
    objectlet_rows_path: str | Path = "outputs/audit/v54_local_objectlets_k0all_conflict_veto018_skip_repeated_sig_l12_stride1_probe5_q4096_notopup_max4000_skip/objectlet_rows.csv",
    component_atom_rows_path: str | Path = "outputs/audit/v55_atoms/component_atom_rows.csv",
    support_rows_path: str | Path = "outputs/audit/v54_mask_component_support_tau005_stride1_probe5_q4096_notopup/mask_component_support_rows.csv",
    native_carrier_rows_path: str | Path = "outputs/audit/v55_native_carrier_materialization_q4096_l11/objectlet_native_carrier_rows.csv",
    c3_history_update_rows_path: str | Path = "outputs/audit/v56_reuse_core_update_C3_e1_boundary_uv/history_update_rows.csv",
    support_variant: str = "R0_visible_tau0.05",
    boundary_component_min_support: int = 1,
    boundary_component_min_ratio: float = 0.0,
    uv_component_min_support: int = 1,
    uv_component_min_ratio: float = 0.0,
    include_boundary: bool = True,
    include_uv: bool = True,
) -> dict[str, Any]:
    anchor_birth_rows = read_csv(_project(anchor_birth_rows_path))
    objectlet_rows = read_csv(_project(objectlet_rows_path))
    component_atom_rows = read_csv(_project(component_atom_rows_path))
    c3_rows = read_csv(_project(c3_history_update_rows_path))
    component_to_atom = _component_atom_map(component_atom_rows)
    scenes = {str(row.get("scene")) for row in anchor_birth_rows}
    component_gt = _support_component_gt(_project(support_rows_path), support_variant=support_variant, scenes=scenes)
    histories = _history_state(anchor_birth_rows, component_to_atom, component_gt)
    objectlet_by_id = {str(row.get("objectlet_id")): row for row in objectlet_rows}
    accepted_c3_rows = [
        row
        for row in c3_rows
        if str(row.get("update_state")) == "confirmed_update"
        and (
            (include_boundary and str(row.get("update_source")) == "native_boundary_projection")
            or (include_uv and str(row.get("update_source")) == "native_uv_bbox_projection")
        )
    ]
    needed_objectlets = {str(row.get("history_id") or "") for row in accepted_c3_rows} | {
        str(row.get("objectlet_id") or "") for row in accepted_c3_rows
    }
    (
        objectlet_frame_masks,
        component_frame_masks,
        objectlet_frames,
        component_frames,
    ) = _native_projection_maps(_project(native_carrier_rows_path), needed_objectlets)

    update_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    added_components_by_history: dict[str, set[str]] = defaultdict(set)
    update_precision_hits = 0
    update_precision_total = 0
    accepted_update_count = 0
    accepted_component_count = 0
    filtered_component_count = 0
    no_component_update_count = 0

    for row in accepted_c3_rows:
        scene = str(row.get("scene"))
        history_id = str(row.get("history_id") or "")
        objectlet_id = str(row.get("objectlet_id") or "")
        source = str(row.get("update_source"))
        history = histories.get(history_id)
        candidate = objectlet_by_id.get(objectlet_id)
        if history is None or candidate is None:
            continue
        components = set(_load_list(candidate.get("component_ids")))
        eligible: list[tuple[str, int, int, float]] = []
        if source == "native_boundary_projection":
            shared_keys = set(objectlet_frame_masks.get(history_id, Counter())) & set(
                objectlet_frame_masks.get(objectlet_id, Counter())
            )
            for component_id in sorted(components):
                counter = component_frame_masks.get(objectlet_id, {}).get(component_id, Counter())
                if not counter:
                    continue
                shared_support = sum(counter.get(key, 0) for key in shared_keys)
                total_support = sum(counter.values())
                ratio = float(shared_support / max(total_support, 1))
                if shared_support >= int(boundary_component_min_support) and ratio >= float(boundary_component_min_ratio):
                    eligible.append((component_id, int(shared_support), int(total_support), ratio))
        elif source == "native_uv_bbox_projection":
            shared_frames = set(objectlet_frames.get(history_id, Counter())) & set(objectlet_frames.get(objectlet_id, Counter()))
            for component_id in sorted(components):
                counter = component_frames.get(objectlet_id, {}).get(component_id, Counter())
                if not counter:
                    continue
                shared_support = sum(counter.get(frame_id, 0) for frame_id in shared_frames)
                total_support = sum(counter.values())
                ratio = float(shared_support / max(total_support, 1))
                if shared_support >= int(uv_component_min_support) and ratio >= float(uv_component_min_ratio):
                    eligible.append((component_id, int(shared_support), int(total_support), ratio))
        newly_added: list[str] = []
        for component_id, shared_support, total_support, ratio in eligible:
            if component_id in history["history_components"]:
                continue
            history_gt = history.get("dominant_gt")
            component_label = _dominant(component_gt.get((scene, component_id), Counter()))
            if history_gt and component_label:
                update_precision_total += 1
                if component_label == history_gt:
                    update_precision_hits += 1
            history["history_components"].add(component_id)
            added_components_by_history[history_id].add(component_id)
            atom_id = component_to_atom.get((scene, component_id), "")
            if atom_id:
                history["history_atoms"].add(atom_id)
            newly_added.append(component_id)
            component_rows.append(
                {
                    "scene": scene,
                    "chunk_id": row.get("chunk_id"),
                    "history_id": history_id,
                    "objectlet_id": objectlet_id,
                    "component_id": component_id,
                    "source_c3_update_source": source,
                    "component_shared_projection_support": shared_support,
                    "component_total_projection_support": total_support,
                    "component_projection_support_ratio": ratio,
                    "component_dominant_gt_diagnostic": component_label,
                    "history_dominant_gt_diagnostic": history_gt,
                    "uses_gt_for_prediction": False,
                    "uses_gt_for_diagnostic_labels": True,
                }
            )
        filtered_component_count += max(len(components) - len(eligible), 0)
        if newly_added:
            accepted_update_count += 1
            accepted_component_count += len(newly_added)
            history["chunks"].add(str(row.get("chunk_id")))
        else:
            no_component_update_count += 1
        update_rows.append(
            {
                "scene": scene,
                "chunk_id": row.get("chunk_id"),
                "history_id": history_id,
                "objectlet_id": objectlet_id,
                "update_state": "confirmed_update" if newly_added else "component_guard_reject",
                "update_source": f"projection_component_guard_from_{source}",
                "source_c3_update_source": source,
                "accepted_component_count": len(newly_added),
                "candidate_component_count": len(components),
                "eligible_component_count": len(eligible),
                "filtered_component_count": max(len(components) - len(eligible), 0),
                "same_frame_exclusion_violation_rate": row.get("same_frame_exclusion_violation_rate"),
                "outside_all_related_masks_ratio_mean": row.get("outside_all_related_masks_ratio_mean"),
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic_labels": True,
            }
        )

    anchor_assignment, anchor_duplicate_count = _build_assignment_map(histories, "anchor_components")
    history_assignment, history_duplicate_count = _build_assignment_map(histories, "history_components")
    anchor_metrics = _metrics_from_support(
        _project(support_rows_path),
        support_variant=support_variant,
        scenes=scenes,
        component_to_history=anchor_assignment,
    )
    history_metrics = _metrics_from_support(
        _project(support_rows_path),
        support_variant=support_variant,
        scenes=scenes,
        component_to_history=history_assignment,
    )
    shuffled_histories: dict[str, dict[str, Any]] = {}
    for history_id, history in histories.items():
        shuffled_histories[history_id] = {**history, "history_components": set(history["anchor_components"])}
    next_history = _next_history_by_scene(histories)
    for history_id, added_components in added_components_by_history.items():
        target_id = next_history.get(history_id)
        if not target_id:
            continue
        shuffled_histories[target_id]["history_components"].update(added_components)
    shuffled_assignment, shuffled_duplicate_count = _build_assignment_map(shuffled_histories, "history_components")
    shuffled_metrics = _metrics_from_support(
        _project(support_rows_path),
        support_variant=support_variant,
        scenes=scenes,
        component_to_history=shuffled_assignment,
    )
    temporal_spans = [float(len(history["chunks"])) for history in histories.values()]
    anchor_temporal_spans = [1.0 for _history in histories.values()]
    update_precision = float(update_precision_hits / max(update_precision_total, 1)) if update_precision_total else None
    conflict_values = [
        parse_float(row["same_frame_exclusion_violation_rate"])
        for row in update_rows
        if row["update_state"] == "confirmed_update" and row["same_frame_exclusion_violation_rate"] not in (None, "")
    ]
    summary = {
        "phase": "v56_projection_component_guard_core_update",
        "created_at": utc_now(),
        "input_paths": {
            "anchor_birth_rows_path": _rel(anchor_birth_rows_path),
            "objectlet_rows_path": _rel(objectlet_rows_path),
            "native_carrier_rows_path": _rel(native_carrier_rows_path),
            "c3_history_update_rows_path": _rel(c3_history_update_rows_path),
            "support_rows_path": _rel(support_rows_path),
        },
        "history_update_variant": "C4c_projection_component_guard_from_C3",
        "include_boundary": bool(include_boundary),
        "include_uv": bool(include_uv),
        "boundary_component_min_support": int(boundary_component_min_support),
        "boundary_component_min_ratio": float(boundary_component_min_ratio),
        "uv_component_min_support": int(uv_component_min_support),
        "uv_component_min_ratio": float(uv_component_min_ratio),
        "history_object_count": len(histories),
        "source_c3_confirmed_update_count": len(accepted_c3_rows),
        "confirmed_update_count": accepted_update_count,
        "confirmed_added_component_count": accepted_component_count,
        "filtered_component_count": filtered_component_count,
        "component_guard_reject_count": no_component_update_count,
        "update_precision_diagnostic": update_precision,
        "update_precision_total": update_precision_total,
        "anchor_only_temporal_span_mean": _mean(anchor_temporal_spans),
        "history_temporal_span_mean": _mean(temporal_spans),
        "anchor_only_ARI": anchor_metrics["ARI"],
        "anchor_only_purity": anchor_metrics["purity"],
        "anchor_only_completeness": anchor_metrics["completeness"],
        "history_ARI": history_metrics["ARI"],
        "history_purity": history_metrics["purity"],
        "history_completeness": history_metrics["completeness"],
        "shuffled_history_ARI": shuffled_metrics["ARI"],
        "shuffled_history_purity": shuffled_metrics["purity"],
        "shuffled_history_completeness": shuffled_metrics["completeness"],
        "real_minus_shuffled_ARI": float(history_metrics["ARI"] - shuffled_metrics["ARI"]),
        "real_minus_no_temporal_ARI": float(history_metrics["ARI"] - anchor_metrics["ARI"]),
        "anchor_duplicate_component_count": anchor_duplicate_count,
        "history_duplicate_component_count": history_duplicate_count,
        "shuffled_duplicate_component_count": shuffled_duplicate_count,
        "conflict_rate": _mean(conflict_values),
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }
    gate = {
        "core_purity_ge_anchor_core_minus_0.005": summary["history_purity"] >= summary["anchor_only_purity"] - 0.005,
        "core_ARI_ge_anchor_core_minus_0.005": summary["history_ARI"] >= summary["anchor_only_ARI"] - 0.005,
        "history_temporal_span_mean_ge_anchor_plus_0.15": (summary["history_temporal_span_mean"] or 0.0)
        >= (summary["anchor_only_temporal_span_mean"] or 0.0) + 0.15,
        "update_precision_diagnostic_ge_0.90": (update_precision or 0.0) >= 0.90,
        "real_minus_shuffled_ARI_ge_0.15": summary["real_minus_shuffled_ARI"] >= 0.15,
        "real_minus_no_temporal_ARI_ge_0.10": summary["real_minus_no_temporal_ARI"] >= 0.10,
    }
    gate["pass"] = bool(all(gate.values()))
    summary["gate"] = gate
    history_rows = [
        {
            "history_id": history_id,
            "scene": history["scene"],
            "anchor_chunk_id": history["anchor_chunk_id"],
            "history_chunk_count": len(history["chunks"]),
            "anchor_component_count": len(history["anchor_components"]),
            "history_component_count": len(history["history_components"]),
            "dominant_gt_diagnostic": history.get("dominant_gt"),
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic_labels": True,
        }
        for history_id, history in histories.items()
    ]
    metric_rows = [
        {"row": "anchor_only", **anchor_metrics},
        {"row": "projection_component_guard_core_update", **history_metrics},
        {"row": "shuffled_update_control", **shuffled_metrics},
    ]
    return {
        "summary": summary,
        "core_update_rows": update_rows,
        "core_component_rows": component_rows,
        "history_rows": history_rows,
        "core_metric_rows": metric_rows,
    }


def write_v56_projection_component_guard_core_update(
    output_root: str | Path,
    payload: dict[str, Any],
) -> None:
    out = _project(output_root)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "core_update_summary.json", payload["summary"])
    write_csv(out / "core_update_rows.csv", payload["core_update_rows"])
    write_csv(out / "core_component_rows.csv", payload["core_component_rows"])
    write_csv(out / "history_rows.csv", payload["history_rows"])
    write_csv(out / "core_metric_rows.csv", payload["core_metric_rows"])
