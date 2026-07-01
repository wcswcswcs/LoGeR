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


def _load_native_component_continuity(
    native_rows_path: Path,
    *,
    history_ids: set[str],
    candidate_objectlet_ids: set[str],
) -> tuple[
    dict[str, Counter[tuple[int, int]]],
    dict[str, dict[str, Counter[tuple[int, int]]]],
    dict[str, dict[tuple[int, int], list[tuple[str, int]]]],
]:
    history_counters: dict[str, Counter[tuple[int, int]]] = defaultdict(Counter)
    candidate_component_counters: dict[str, dict[str, Counter[tuple[int, int]]]] = defaultdict(lambda: defaultdict(Counter))
    with native_rows_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            objectlet_id = str(row.get("objectlet_id") or "")
            if objectlet_id not in history_ids and objectlet_id not in candidate_objectlet_ids:
                continue
            if not parse_bool(row.get("visible")) or not parse_bool(row.get("valid_uv")):
                continue
            observed_mask_id = parse_int(row.get("observed_mask_id"))
            if observed_mask_id <= 0:
                continue
            key = (parse_int(row.get("frame_id")), observed_mask_id)
            if objectlet_id in history_ids:
                history_counters[objectlet_id][key] += 1
            if objectlet_id in candidate_objectlet_ids:
                component_id = str(row.get("component_id") or "")
                if component_id:
                    candidate_component_counters[objectlet_id][component_id][key] += 1

    history_by_scene_key: dict[str, dict[tuple[int, int], list[tuple[str, int]]]] = defaultdict(lambda: defaultdict(list))
    for history_id, counter in history_counters.items():
        scene = history_id.split("|", 1)[0]
        for key, count in counter.items():
            history_by_scene_key[scene][key].append((history_id, int(count)))
    return history_counters, candidate_component_counters, history_by_scene_key


def build_v56_component_native_core_update(
    *,
    chunk_role_rows_path: str | Path = "outputs/audit/v55_chunk_roles/chunk_role_rows.csv",
    anchor_birth_rows_path: str | Path = "outputs/audit/v55_anchor_birth/anchor_birth_rows.csv",
    objectlet_rows_path: str | Path = "outputs/audit/v54_local_objectlets_k0all_conflict_veto018_skip_repeated_sig_l12_stride1_probe5_q4096_notopup_max4000_skip/objectlet_rows.csv",
    local_summary_path: str | Path = "outputs/audit/v54_local_reproduction_stride1_probe5_q4096_notopup_k0all_max4000_veto018_w123_fullchunk/local_reproduction_summary.json",
    component_atom_rows_path: str | Path = "outputs/audit/v55_atoms/component_atom_rows.csv",
    support_rows_path: str | Path = "outputs/audit/v54_mask_component_support_tau005_stride1_probe5_q4096_notopup/mask_component_support_rows.csv",
    native_carrier_rows_path: str | Path = "outputs/audit/v55_native_carrier_materialization_q4096_l11/objectlet_native_carrier_rows.csv",
    support_variant: str = "R0_visible_tau0.05",
    history_evidence_roles: tuple[str, ...] = ("bridge", "update"),
    component_min_shared_support: int = 5,
    component_min_shared_ratio: float = 0.30,
    objectlet_min_component_count: int = 2,
    objectlet_min_total_shared_support: int = 20,
) -> dict[str, Any]:
    role_rows = read_csv(_project(chunk_role_rows_path))
    anchor_birth_rows = read_csv(_project(anchor_birth_rows_path))
    objectlet_rows = read_csv(_project(objectlet_rows_path))
    local_summary = read_json(_project(local_summary_path))
    component_atom_rows = read_csv(_project(component_atom_rows_path))
    component_to_atom = _component_atom_map(component_atom_rows)
    best_variant = str(local_summary.get("best_method_variant") or "")
    scenes = {str(row.get("scene")) for row in role_rows}
    evidence_roles = {str(role) for role in history_evidence_roles if str(role)}
    evidence_chunks = {str(row.get("chunk_id")) for row in role_rows if str(row.get("role")) in evidence_roles}
    update_chunks = {str(row.get("chunk_id")) for row in role_rows if str(row.get("role")) == "update"}
    component_gt = _support_component_gt(_project(support_rows_path), support_variant=support_variant, scenes=scenes)
    histories = _history_state(anchor_birth_rows, component_to_atom, component_gt)
    histories_by_scene: dict[str, list[str]] = defaultdict(list)
    for history_id, history in histories.items():
        histories_by_scene[str(history["scene"])].append(history_id)

    candidates = [
        row
        for row in objectlet_rows
        if str(row.get("variant")) == best_variant and str(row.get("chunk_id")) in evidence_chunks
    ]
    candidate_by_objectlet = {str(row.get("objectlet_id")): row for row in candidates}
    history_counters, candidate_component_counters, history_by_scene_key = _load_native_component_continuity(
        _project(native_carrier_rows_path),
        history_ids=set(histories),
        candidate_objectlet_ids=set(candidate_by_objectlet),
    )

    update_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    added_components_by_history: dict[str, set[str]] = defaultdict(set)
    update_precision_hits = 0
    update_precision_total = 0
    candidate_count = 0
    accepted_update_count = 0
    accepted_component_count = 0
    rejected_objectlet_count = 0

    for candidate in candidates:
        scene = str(candidate.get("scene"))
        objectlet_id = str(candidate.get("objectlet_id"))
        component_counters = candidate_component_counters.get(objectlet_id, {})
        if not component_counters:
            continue
        candidate_count += 1
        shared_by_history_component: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        total_by_component = {component_id: sum(counter.values()) for component_id, counter in component_counters.items()}
        for component_id, counter in component_counters.items():
            for key, count in counter.items():
                for history_id, history_count in history_by_scene_key.get(scene, {}).get(key, []):
                    shared_by_history_component[history_id][component_id] += min(int(count), int(history_count))

        scored: list[tuple[int, int, float, str, list[str]]] = []
        for history_id in histories_by_scene.get(scene, []):
            history = histories[history_id]
            eligible_components: list[str] = []
            total_shared = 0
            total_support = 0
            for component_id, shared in shared_by_history_component.get(history_id, {}).items():
                if component_id in history["history_components"]:
                    continue
                support = max(total_by_component.get(component_id, 0), 1)
                ratio = float(shared / support)
                if shared >= int(component_min_shared_support) and ratio >= float(component_min_shared_ratio):
                    eligible_components.append(component_id)
                    total_shared += int(shared)
                    total_support += int(support)
            if not eligible_components:
                continue
            mean_ratio = float(total_shared / max(total_support, 1))
            scored.append((len(eligible_components), total_shared, mean_ratio, history_id, sorted(eligible_components)))
        scored.sort(key=lambda item: (-item[0], -item[1], -item[2], item[3]))
        if not scored:
            rejected_objectlet_count += 1
            continue
        eligible_count, total_shared, mean_ratio, history_id, eligible_components = scored[0]
        if eligible_count < int(objectlet_min_component_count) or total_shared < int(objectlet_min_total_shared_support):
            rejected_objectlet_count += 1
            continue
        history = histories[history_id]
        newly_added: list[str] = []
        for component_id in eligible_components:
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
                    "history_id": history_id,
                    "scene": scene,
                    "chunk_id": candidate.get("chunk_id"),
                    "objectlet_id": objectlet_id,
                    "component_id": component_id,
                    "component_shared_support": shared_by_history_component[history_id][component_id],
                    "component_total_native_support": total_by_component.get(component_id, 0),
                    "component_shared_ratio": float(
                        shared_by_history_component[history_id][component_id]
                        / max(total_by_component.get(component_id, 0), 1)
                    ),
                    "component_dominant_gt_diagnostic": _dominant(component_gt.get((scene, component_id), Counter())),
                    "history_dominant_gt_diagnostic": history.get("dominant_gt"),
                    "uses_gt_for_prediction": False,
                    "uses_gt_for_diagnostic_labels": True,
                }
            )
        if newly_added:
            accepted_update_count += 1
            accepted_component_count += len(newly_added)
            history["chunks"].add(str(candidate.get("chunk_id")))
        update_rows.append(
            {
                "scene": scene,
                "chunk_id": candidate.get("chunk_id"),
                "history_id": history_id,
                "objectlet_id": objectlet_id,
                "update_state": "confirmed_update" if newly_added else "duplicate_noop",
                "update_source": "component_native_frame_mask_continuity",
                "accepted_component_count": len(newly_added),
                "candidate_component_count": len(_load_list(candidate.get("component_ids"))),
                "eligible_component_count": eligible_count,
                "total_shared_support": total_shared,
                "mean_component_shared_ratio": mean_ratio,
                "same_frame_exclusion_violation_rate": parse_float(candidate.get("same_frame_exclusion_violation_rate")),
                "outside_all_related_masks_ratio_mean": parse_float(candidate.get("outside_all_related_masks_ratio_mean")),
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic_labels": True,
            }
        )

    for history in histories.values():
        for chunk_id in sorted(chunk for chunk in update_chunks if chunk.startswith(str(history["scene"]))):
            if not any(row["history_id"] == history["history_id"] and row["chunk_id"] == chunk_id for row in update_rows):
                update_rows.append(
                    {
                        "scene": history["scene"],
                        "chunk_id": chunk_id,
                        "history_id": history["history_id"],
                        "objectlet_id": "",
                        "update_state": "occluded_or_absent",
                        "update_source": "no_evidence",
                        "accepted_component_count": 0,
                        "candidate_component_count": 0,
                        "eligible_component_count": 0,
                        "total_shared_support": 0,
                        "mean_component_shared_ratio": 0.0,
                        "same_frame_exclusion_violation_rate": None,
                        "outside_all_related_masks_ratio_mean": None,
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
        float(row["same_frame_exclusion_violation_rate"])
        for row in update_rows
        if row["update_state"] == "confirmed_update" and row["same_frame_exclusion_violation_rate"] not in (None, "")
    ]
    summary = {
        "phase": "v56_component_native_core_update",
        "created_at": utc_now(),
        "input_paths": {
            "chunk_role_rows_path": _rel(chunk_role_rows_path),
            "anchor_birth_rows_path": _rel(anchor_birth_rows_path),
            "objectlet_rows_path": _rel(objectlet_rows_path),
            "native_carrier_rows_path": _rel(native_carrier_rows_path),
            "support_rows_path": _rel(support_rows_path),
        },
        "history_update_variant": "C4_component_native_frame_mask_continuity",
        "history_object_count": len(histories),
        "history_evidence_roles": sorted(evidence_roles),
        "component_min_shared_support": int(component_min_shared_support),
        "component_min_shared_ratio": float(component_min_shared_ratio),
        "objectlet_min_component_count": int(objectlet_min_component_count),
        "objectlet_min_total_shared_support": int(objectlet_min_total_shared_support),
        "candidate_objectlet_count": candidate_count,
        "accepted_update_count": accepted_update_count,
        "rejected_objectlet_count": rejected_objectlet_count,
        "confirmed_update_count": accepted_update_count,
        "confirmed_added_component_count": accepted_component_count,
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
        {"row": "component_native_core_update", **history_metrics},
        {"row": "shuffled_update_control", **shuffled_metrics},
    ]
    return {
        "summary": summary,
        "core_update_rows": update_rows,
        "core_component_rows": component_rows,
        "history_rows": history_rows,
        "core_metric_rows": metric_rows,
    }


def write_v56_component_native_core_update(
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

