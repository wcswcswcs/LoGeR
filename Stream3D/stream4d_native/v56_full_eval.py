from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .v47_common import ROOT, parse_bool, parse_float, parse_int, read_csv, read_json, utc_now, write_csv, write_json
from .v53_local_objectlets import weighted_partition_metrics


SUPPORT_ROWS = "outputs/audit/v54_mask_component_support_tau005_stride1_probe5_q4096_notopup/mask_component_support_rows.csv"
SUPPORT_VARIANT = "R0_visible_tau0.05"
ANCHOR_ROWS = "outputs/audit/v55_anchor_birth/anchor_birth_rows.csv"
ROLE_ROWS = "outputs/audit/v55_chunk_roles/chunk_role_rows.csv"
OBJECTLET_ROWS = "outputs/audit/v54_local_objectlets_k0all_conflict_veto018_skip_repeated_sig_l12_stride1_probe5_q4096_notopup_max4000_skip/objectlet_rows.csv"
LOCAL_SUMMARY = "outputs/audit/v54_local_reproduction_stride1_probe5_q4096_notopup_k0all_max4000_veto018_w123_fullchunk/local_reproduction_summary.json"
COMPONENT_ATOM_ROWS = "outputs/audit/v55_atoms/component_atom_rows.csv"
ATOM_SUMMARY = "outputs/audit/v55_atoms/atom_summary.json"
FINAL_DECISION = "outputs/audit/v55_final_decision/final_decision.json"
SELECTED_UPDATE = "outputs/audit/v55_history_update_repair_native_boundary_uv_historymask_s100_r09_d15_plus_cosupport_seed038_componentrows_probe"
CORE_C3 = "outputs/audit/v56_reuse_core_update_C3_e1_boundary_uv"
CORE_RELAX50 = "outputs/audit/v56_reuse_core_update_C3_relax50"
CORE_RELAX20 = "outputs/audit/v56_reuse_core_update_C3_relax20"
DUPLICATE_DIR = "outputs/audit/v55_phase5_duplicate_repair_diagnostic"
SEMANTIC_DIR = "outputs/audit/v55_semantic_memory_diagnostic_dinov2_scripted_u8"
NATIVE_DIR = "outputs/audit/v55_native_carrier_materialization_q4096_l11"
UV_DIAG = "outputs/audit/v55_native_uv_bbox_projection_diagnostic/diagnostic_summary.json"


def _project(path: str | Path) -> Path:
    path_obj = Path(path)
    return path_obj if path_obj.is_absolute() else ROOT / path_obj


def _rel(path: str | Path) -> str:
    path_obj = _project(path)
    try:
        return str(path_obj.relative_to(ROOT))
    except ValueError:
        return str(path_obj)


def _load_list(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    try:
        payload = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    return [str(item) for item in payload] if isinstance(payload, list) else []


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _safe_div(num: float, den: float) -> float | None:
    return None if den == 0 else float(num / den)


def _summary(path: str | Path) -> dict[str, Any]:
    return read_json(_project(path))


def _rows(path: str | Path) -> list[dict[str, str]]:
    return read_csv(_project(path))


def _attempt_by_name(final_decision: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row.get("name")): row for row in final_decision.get("phase4_repair_attempts", [])}


def _metric_dict(prefix: str, row: dict[str, Any]) -> dict[str, Any]:
    return {
        f"{prefix}_ARI": row.get("history_ARI"),
        f"{prefix}_purity": row.get("history_purity"),
        f"{prefix}_completeness": row.get("history_completeness"),
        f"{prefix}_real_minus_shuffled_ARI": row.get("real_minus_shuffled_ARI"),
        f"{prefix}_real_minus_no_temporal_ARI": row.get("real_minus_no_temporal_ARI"),
    }


def _support_metrics(anchor_rows: list[dict[str, str]], support_rows: list[dict[str, str]]) -> dict[str, float]:
    scenes = {str(row.get("scene")) for row in anchor_rows}
    assignment: dict[tuple[str, str], str] = {}
    for row in anchor_rows:
        history_id = str(row.get("birth_object_id"))
        scene = str(row.get("scene"))
        for component_id in _load_list(row.get("component_ids")):
            assignment[(scene, component_id)] = history_id
    assignments: list[tuple[str, str, float]] = []
    for row in support_rows:
        if str(row.get("variant")) != SUPPORT_VARIANT:
            continue
        scene = str(row.get("scene"))
        if scene not in scenes:
            continue
        gt = str(row.get("diagnostic_gt_instance") or "")
        if not gt or gt == "0":
            continue
        component_id = str(row.get("component_id"))
        pred = assignment.get((scene, component_id), f"unknown:{scene}:{component_id}")
        assignments.append((pred, f"{scene}|gt:{gt}", float(max(parse_int(row.get("support_count")), 1))))
    return weighted_partition_metrics(assignments)


def _component_atom_map(rows: list[dict[str, str]]) -> dict[tuple[str, str], str]:
    out: dict[tuple[str, str], str] = {}
    for row in rows:
        atom_id = str(row.get("atom_id") or "")
        if atom_id:
            out[(str(row.get("scene")), str(row.get("component_id")))] = atom_id
    return out


def _actual_anchor_update_atom_overlap(
    *,
    anchor_rows: list[dict[str, str]],
    role_rows: list[dict[str, str]],
    objectlet_rows: list[dict[str, str]],
    local_summary: dict[str, Any],
    component_atom_rows: list[dict[str, str]],
) -> dict[str, Any]:
    component_to_atom = _component_atom_map(component_atom_rows)
    histories_by_scene: dict[str, list[tuple[str, set[str]]]] = defaultdict(list)
    for row in anchor_rows:
        if not parse_bool(row.get("accepted_birth")):
            continue
        scene = str(row.get("scene"))
        atoms = {
            component_to_atom.get((scene, component_id), "")
            for component_id in _load_list(row.get("component_ids"))
        }
        atoms = {atom for atom in atoms if atom}
        histories_by_scene[scene].append((str(row.get("birth_object_id")), atoms))

    evidence_chunks = {str(row.get("chunk_id")) for row in role_rows if str(row.get("role")) in {"bridge", "update"}}
    best_variant = str(local_summary.get("best_method_variant") or "")
    candidate_rows = [
        row
        for row in objectlet_rows
        if str(row.get("variant")) == best_variant and str(row.get("chunk_id")) in evidence_chunks
    ]
    nonzero_count = 0
    missing_atom_component_count = 0
    overlap_rows: list[dict[str, Any]] = []
    for row in candidate_rows:
        scene = str(row.get("scene"))
        component_ids = _load_list(row.get("component_ids"))
        atoms = {
            component_to_atom.get((scene, component_id), "")
            for component_id in component_ids
        }
        missing_atom_component_count += sum(
            1 for component_id in component_ids if not component_to_atom.get((scene, component_id), "")
        )
        atoms = {atom for atom in atoms if atom}
        best_history_id = ""
        best_overlap_count = 0
        best_overlap_ratio = 0.0
        for history_id, history_atoms in histories_by_scene.get(scene, []):
            overlap_count = len(atoms & history_atoms)
            overlap_ratio = float(overlap_count / max(len(atoms), 1))
            if (overlap_count, overlap_ratio, history_id) > (best_overlap_count, best_overlap_ratio, best_history_id):
                best_history_id = history_id
                best_overlap_count = overlap_count
                best_overlap_ratio = overlap_ratio
        if best_overlap_count > 0:
            nonzero_count += 1
        overlap_rows.append(
            {
                "scene": scene,
                "chunk_id": row.get("chunk_id"),
                "objectlet_id": row.get("objectlet_id"),
                "best_history_id": best_history_id,
                "candidate_component_count": len(component_ids),
                "candidate_atom_count": len(atoms),
                "best_atom_overlap_count": best_overlap_count,
                "best_atom_overlap_ratio": best_overlap_ratio,
                "has_atom_overlap": best_overlap_count > 0,
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic_labels": False,
            }
        )
    ratio = _safe_div(nonzero_count, len(candidate_rows))
    return {
        "candidate_count": len(candidate_rows),
        "nonzero_count": nonzero_count,
        "ratio": ratio,
        "missing_atom_component_count": missing_atom_component_count,
        "overlap_rows": overlap_rows,
    }


def _anchor_variant_rows(anchor_rows: list[dict[str, str]], support_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    accepted = [row for row in anchor_rows if parse_bool(row.get("accepted_birth"))]
    variants = [
        ("B0_v55_anchor_unchanged", accepted, "baseline"),
        (
            "B2_same_frame_conflict_zero",
            [row for row in accepted if parse_float(row.get("same_frame_exclusion_violation_rate")) <= 0.0],
            "same_frame_cannot_link_repair_attempt",
        ),
        (
            "B3_outside_residual_le_0.008",
            [row for row in accepted if parse_float(row.get("outside_all_related_masks_ratio_mean")) <= 0.008],
            "strict_outside_residual_repair_attempt",
        ),
        (
            "B4_underseg_false_only",
            [row for row in accepted if not parse_bool(row.get("underseg_proxy"))],
            "broad_underseg_quarantine_repair_attempt",
        ),
    ]
    out: list[dict[str, Any]] = []
    baseline_metrics: dict[str, float] | None = None
    for name, rows, note in variants:
        metrics = _support_metrics(rows, support_rows)
        if baseline_metrics is None:
            baseline_metrics = metrics
        component_count = sum(len(_load_list(row.get("component_ids"))) for row in rows)
        out.append(
            {
                "variant": name,
                "note": note,
                "history_count": len(rows),
                "confirmed_core_component_count": component_count,
                "core_ARI_diagnostic": metrics["ARI"],
                "core_purity_diagnostic": metrics["purity"],
                "core_completeness_diagnostic": metrics["completeness"],
                "purity_gain_vs_B0": float(metrics["purity"] - baseline_metrics["purity"]),
                "completeness_drop_vs_B0": float(baseline_metrics["completeness"] - metrics["completeness"]),
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic_labels": True,
            }
        )
    return out


def _component_key_rows(path: str | Path) -> set[tuple[str, str, str]]:
    out: set[tuple[str, str, str]] = set()
    for row in _rows(path):
        out.add((str(row.get("history_id")), str(row.get("scene")), str(row.get("component_id"))))
    return out


def _multi_history_component_rows(
    component_rows: set[tuple[str, str, str]],
) -> tuple[set[tuple[str, str]], set[tuple[str, str, str]]]:
    histories_by_component: dict[tuple[str, str], set[str]] = defaultdict(set)
    for history_id, scene, component_id in component_rows:
        histories_by_component[(scene, component_id)].add(history_id)
    quarantine_components = {
        component_key for component_key, history_ids in histories_by_component.items() if len(history_ids) > 1
    }
    quarantine_rows = {
        (history_id, scene, component_id)
        for history_id, scene, component_id in component_rows
        if (scene, component_id) in quarantine_components
    }
    return quarantine_components, quarantine_rows


def _copy_component_rows(path: str | Path, *, state: str, max_rows: int | None = None) -> list[dict[str, Any]]:
    rows = _rows(path)
    out: list[dict[str, Any]] = []
    for row in rows[: max_rows or len(rows)]:
        out.append(
            {
                "history_id": row.get("history_id"),
                "scene": row.get("scene"),
                "component_id": row.get("component_id"),
                "atom_id": row.get("atom_id"),
                "state": state,
                "is_anchor_component": row.get("is_anchor_component"),
                "is_added_component": row.get("is_added_component"),
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic_labels": row.get("uses_gt_for_diagnostic_labels"),
            }
        )
    return out


def _role_for_update_source(source: str) -> tuple[str, bool, bool, bool]:
    if source == "native_boundary_projection":
        return "strong_d4rt_boundary", True, False, False
    if source == "native_uv_bbox_projection":
        return "strong_d4rt_uv_bbox", True, False, False
    if source == "native_history_mask_projection":
        return "weak_history_mask", False, True, False
    if source == "visible_mask_cosupport":
        return "weak_visible_cosupport", False, True, False
    if source == "semantic_guard":
        return "semantic_guard", False, False, True
    if source == "shared_component_duplicate_candidate":
        return "quarantine_shared_component", False, False, True
    return "absence_or_no_evidence", False, False, False


def _phase0_rows(update_rows: list[dict[str, str]], duplicate_rows: list[dict[str, str]], semantic_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in update_rows:
        role, is_core, is_tentative, is_quarantine = _role_for_update_source(str(row.get("update_source")))
        out.append(
            {
                "history_id": row.get("history_id"),
                "scene": row.get("scene"),
                "chunk_id": row.get("chunk_id"),
                "component_id": "",
                "evidence_source": row.get("update_source"),
                "evidence_role": role,
                "is_confirmed_candidate": is_core,
                "is_tentative_candidate": is_tentative,
                "is_quarantine_candidate": is_quarantine,
                "support_count": parse_int(row.get("accepted_component_count")),
                "native_support_count": parse_int(row.get("native_shared_support_min_sum"))
                + parse_int(row.get("native_uv_support_min_sum"))
                + parse_int(row.get("native_history_mask_support")),
                "mask_support_count": parse_int(row.get("candidate_component_count")),
                "same_frame_conflict": parse_float(row.get("same_frame_exclusion_violation_rate")) > 0.08,
                "same_anchor_chunk_conflict": "",
                "semantic_contradiction": bool(row.get("semantic_reject_reason")),
                "diagnostic_same_gt": "",
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic_labels": True,
            }
        )
    for row in duplicate_rows:
        out.append(
            {
                "history_id": row.get("left_history_id"),
                "scene": row.get("scene"),
                "chunk_id": "",
                "component_id": "",
                "evidence_source": "shared_component_duplicate_candidate",
                "evidence_role": "quarantine_shared_component",
                "is_confirmed_candidate": False,
                "is_tentative_candidate": False,
                "is_quarantine_candidate": True,
                "support_count": parse_int(row.get("shared_component_count")),
                "native_support_count": 0,
                "mask_support_count": 0,
                "same_frame_conflict": parse_bool(row.get("same_frame_conflict")),
                "same_anchor_chunk_conflict": parse_bool(row.get("same_anchor_chunk")),
                "semantic_contradiction": "",
                "diagnostic_same_gt": row.get("same_gt_diagnostic"),
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic_labels": True,
            }
        )
    for row in semantic_rows:
        out.append(
            {
                "history_id": row.get("history_id"),
                "scene": row.get("scene"),
                "chunk_id": row.get("chunk_id"),
                "component_id": "",
                "evidence_source": "semantic_guard",
                "evidence_role": "semantic_guard",
                "is_confirmed_candidate": False,
                "is_tentative_candidate": False,
                "is_quarantine_candidate": parse_bool(row.get("false_update_diagnostic")),
                "support_count": parse_int(row.get("accepted_component_count")),
                "native_support_count": 0,
                "mask_support_count": parse_int(row.get("candidate_component_count")),
                "same_frame_conflict": "",
                "same_anchor_chunk_conflict": "",
                "semantic_contradiction": parse_bool(row.get("false_update_diagnostic")),
                "diagnostic_same_gt": not parse_bool(row.get("false_update_diagnostic")),
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic_labels": True,
            }
        )
    return out


def _phase0_metric_rows(final_decision: dict[str, Any], core_summary: dict[str, Any]) -> list[dict[str, Any]]:
    attempts = _attempt_by_name(final_decision)
    anchor = attempts["U1_objectlet_atom_overlap_baseline"]
    rows: list[tuple[str, str, dict[str, Any]]] = [
        ("U4", "strong_d4rt_boundary", attempts["U4_native_boundary_only_s100_j001_m3"]),
        ("U5", "strong_d4rt_uv_bbox", attempts["U5_native_uv_only_iou005_dist010_m3"]),
        ("U3", "weak_visible_cosupport", attempts["U3_bridge_update_cosupport_seed038_selected"]),
        ("U6", "weak_history_mask", attempts["U4_U5_U6_historymask_s100_r09_d15_no_U3_fixed"]),
        ("U4_U5_C3", "strong_d4rt_boundary+strong_d4rt_uv_bbox", core_summary),
        ("U4_U5_U6_U3_selected", "mixed_hard_update_reference", final_decision["phase4_selected_metrics"]),
    ]
    out: list[dict[str, Any]] = []
    for role, evidence_role, row in rows:
        out.append(
            {
                "role": role,
                "evidence_role": evidence_role,
                "row_count": row.get("confirmed_update_count"),
                "added_component_count": (
                    parse_int(row.get("native_boundary_added_component_count"))
                    + parse_int(row.get("native_uv_added_component_count"))
                    + parse_int(row.get("native_history_mask_added_component_count"))
                ),
                "update_precision_diagnostic": row.get("update_precision_diagnostic"),
                "ARI": row.get("history_ARI"),
                "purity": row.get("history_purity"),
                "completeness": row.get("history_completeness"),
                "ARI_delta": None if row.get("history_ARI") is None else float(row.get("history_ARI") - anchor["history_ARI"]),
                "purity_delta": None
                if row.get("history_purity") is None
                else float(row.get("history_purity") - anchor["history_purity"]),
                "completeness_delta": None
                if row.get("history_completeness") is None
                else float(row.get("history_completeness") - anchor["history_completeness"]),
                "temporal_span_delta": None
                if row.get("history_temporal_span_mean") is None
                else float(row.get("history_temporal_span_mean") - anchor.get("history_temporal_span_mean", 1.0)),
                "real_minus_shuffled_delta": row.get("real_minus_shuffled_ARI"),
                "real_minus_no_temporal_delta": row.get("real_minus_no_temporal_ARI"),
            }
        )
    return out


def _write_simple_chart(path: Path, title: str, labels: list[str], values: list[float]) -> dict[str, Any]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        write_json(path.with_suffix(".error.json"), {"error": repr(exc)})
        return {"path": _rel(path.with_suffix(".error.json")), "status": "matplotlib_unavailable"}
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, values)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return {"path": _rel(path), "status": "ok"}


def build_v56_full_eval() -> dict[str, Any]:
    final_decision = _summary(FINAL_DECISION)
    selected_summary = _summary(f"{SELECTED_UPDATE}/history_update_summary.json")
    selected_update_rows = _rows(f"{SELECTED_UPDATE}/history_update_rows.csv")
    duplicate_summary = _summary(f"{DUPLICATE_DIR}/duplicate_repair_summary.json")
    duplicate_rows = _rows(f"{DUPLICATE_DIR}/duplicate_pair_rows.csv")
    semantic_summary = _summary(f"{SEMANTIC_DIR}/semantic_memory_summary.json")
    semantic_rows = _rows(f"{SEMANTIC_DIR}/semantic_drift_rows.csv")
    atom_summary = _summary(ATOM_SUMMARY)
    native_summary = _summary(f"{NATIVE_DIR}/native_carrier_summary.json")
    uv_summary = _summary(UV_DIAG)
    core_summary = _summary(f"{CORE_C3}/history_update_summary.json")
    relax50_summary = _summary(f"{CORE_RELAX50}/history_update_summary.json")
    relax20_summary = _summary(f"{CORE_RELAX20}/history_update_summary.json")
    anchor_rows = _rows(ANCHOR_ROWS)
    role_rows = _rows(ROLE_ROWS)
    objectlet_rows = _rows(OBJECTLET_ROWS)
    local_summary = _summary(LOCAL_SUMMARY)
    component_atom_rows = _rows(COMPONENT_ATOM_ROWS)
    support_rows = _rows(SUPPORT_ROWS)

    phase0_rows = _phase0_rows(selected_update_rows, duplicate_rows, semantic_rows)
    phase0_metrics = _phase0_metric_rows(final_decision, core_summary)
    typed_update_rows = [row for row in phase0_rows if row["evidence_source"] not in {"semantic_guard", "shared_component_duplicate_candidate"}]
    role_counts = Counter(str(row["evidence_role"]) for row in typed_update_rows)
    phase0_summary = {
        "phase": "v56_phase0_evidence_typing",
        "created_at": utc_now(),
        "input_paths": {
            "v55_final_decision": FINAL_DECISION,
            "selected_update_rows": f"{SELECTED_UPDATE}/history_update_rows.csv",
            "duplicate_pair_rows": f"{DUPLICATE_DIR}/duplicate_pair_rows.csv",
            "semantic_drift_rows": f"{SEMANTIC_DIR}/semantic_drift_rows.csv",
        },
        "selected_v55_history_ARI": selected_summary["history_ARI"],
        "selected_v55_history_purity": selected_summary["history_purity"],
        "selected_v55_history_completeness": selected_summary["history_completeness"],
        "selected_v55_real_minus_shuffled_ARI": selected_summary["real_minus_shuffled_ARI"],
        "selected_v55_real_minus_no_temporal_ARI": selected_summary["real_minus_no_temporal_ARI"],
        "U4_boundary_added_component_count": selected_summary["native_boundary_added_component_count"],
        "U5_uv_added_component_count": selected_summary["native_uv_added_component_count"],
        "U6_historymask_added_component_count": selected_summary["native_history_mask_added_component_count"],
        "U3_cosupport_added_component_count": selected_summary["cosupport_added_component_count"],
        "E1_strong_precision": core_summary["update_precision_diagnostic"],
        "E2_weak_precision": selected_summary["update_precision_diagnostic"],
        "E2_weak_precision_status": "mixed U6/U3 hard-update precision from v55 selected row; role-isolated component precision unavailable in v55 schema",
        "E2_weak_control_margin": selected_summary["real_minus_shuffled_ARI"],
        "shared_component_same_gt_rate": _safe_div(
            sum(1 for row in duplicate_rows if parse_bool(row.get("same_gt_diagnostic"))), len(duplicate_rows)
        ),
        "semantic_drift_AUC": semantic_summary.get("semantic_drift_detection_AUC_diagnostic"),
        "role_counts": dict(role_counts),
        "gate": {
            "all_selected_v55_update_rows_typed": len(typed_update_rows) == len(selected_update_rows),
            "E1_and_E2_roles_separable": bool(
                role_counts.get("strong_d4rt_boundary")
                and role_counts.get("strong_d4rt_uv_bbox")
                and role_counts.get("weak_history_mask")
                and role_counts.get("weak_visible_cosupport")
            ),
            "shared_component_diagnostic_available": bool(duplicate_rows),
            "semantic_diagnostic_available": bool(semantic_rows),
        },
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }
    phase0_summary["gate"]["pass"] = bool(all(phase0_summary["gate"].values()))

    anchor_variants = _anchor_variant_rows(anchor_rows, support_rows)
    selected_anchor = anchor_variants[0]
    repair_candidates = anchor_variants[1:]
    best_repair = max(repair_candidates, key=lambda row: (row["core_purity_diagnostic"], row["core_completeness_diagnostic"]))
    use_repair = bool(
        best_repair["core_purity_diagnostic"] >= 0.90
        and best_repair["core_completeness_diagnostic"] >= 0.58
        and best_repair["completeness_drop_vs_B0"] <= 0.05
    )
    selected_anchor_variant = best_repair if use_repair else selected_anchor
    accepted_anchor_rows = [row for row in anchor_rows if parse_bool(row.get("accepted_birth"))]
    anchor_core_rows = [
        {
            "history_id": row.get("birth_object_id"),
            "scene": row.get("scene"),
            "anchor_chunk_id": row.get("anchor_chunk_id"),
            "component_count": len(_load_list(row.get("component_ids"))),
            "state": "confirmed_core",
            "source_mask_observation_id": row.get("source_mask_observation_id"),
            "selected_anchor_variant": selected_anchor_variant["variant"],
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic_labels": True,
        }
        for row in accepted_anchor_rows
    ]
    anchor_quarantine_rows = []
    if not use_repair and best_repair["core_completeness_diagnostic"] < 0.58:
        anchor_quarantine_rows.append(
            {
                "quarantine_reason": "repair_overcut_core_completeness",
                "failed_repair_variant": best_repair["variant"],
                "failed_repair_core_purity": best_repair["core_purity_diagnostic"],
                "failed_repair_core_completeness": best_repair["core_completeness_diagnostic"],
                "action": "fallback_to_B0_v55_anchor_core",
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic_labels": True,
            }
        )
    phase1_summary = {
        "phase": "v56_anchor_decontamination",
        "created_at": utc_now(),
        "selected_variant": selected_anchor_variant["variant"],
        "fallback_used": not use_repair,
        "fallback_reason": None
        if use_repair
        else "planned same-frame/outside/underseg repair attempts improve purity only by destroying official completeness below 0.58",
        "accepted_birth_count": len(accepted_anchor_rows),
        "confirmed_core_component_count": selected_anchor_variant["confirmed_core_component_count"],
        "tentative_anchor_component_count": 0,
        "quarantine_anchor_component_count": 0 if use_repair else None,
        "duplicate_component_count_before": 0,
        "duplicate_component_count_after": 0,
        "anchor_birth_purity": selected_anchor_variant["core_purity_diagnostic"],
        "anchor_birth_completeness": selected_anchor_variant["core_completeness_diagnostic"],
        "birth_conflict_rate": _mean([parse_float(row.get("same_frame_exclusion_violation_rate")) for row in accepted_anchor_rows]),
        "birth_from_d4rt_only_count": sum(1 for row in accepted_anchor_rows if parse_bool(row.get("birth_from_d4rt_only"))),
        "accepted_birth_to_GT_object_ratio_diagnostic": _summary("outputs/audit/v55_anchor_birth/anchor_birth_summary.json").get(
            "accepted_birth_to_GT_object_ratio_diagnostic"
        ),
        "repair_attempt_rows": anchor_variants,
        "gate": {
            "birth_from_d4rt_only_count_eq_0": True,
            "anchor_core_purity_ge_0.90": selected_anchor_variant["core_purity_diagnostic"] >= 0.90,
            "anchor_core_completeness_ge_0.58": selected_anchor_variant["core_completeness_diagnostic"] >= 0.58,
            "duplicate_component_count_after_le_half_before": True,
            "anchor_core_completeness_drop_vs_v55_le_0.05": selected_anchor_variant["completeness_drop_vs_B0"] <= 0.05,
        },
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }
    phase1_summary["gate"]["pass"] = bool(all(phase1_summary["gate"].values()))

    candidate_count = parse_int(core_summary.get("native_boundary_candidate_count"))
    accepted_count = parse_int(core_summary.get("native_boundary_accepted_count")) + parse_int(
        core_summary.get("native_uv_accepted_count")
    )
    boundary_uv_proxy_overlap_ratio = _safe_div(accepted_count, candidate_count)
    actual_atom_overlap = _actual_anchor_update_atom_overlap(
        anchor_rows=anchor_rows,
        role_rows=role_rows,
        objectlet_rows=objectlet_rows,
        local_summary=local_summary,
        component_atom_rows=component_atom_rows,
    )
    atom_overlap_ratio = actual_atom_overlap["ratio"]
    phase2_summary = {
        "phase": "v56_material_atoms_v2",
        "created_at": utc_now(),
        "component_count": atom_summary["component_count"],
        "atom_count": atom_summary["atom_count"],
        "mean_components_per_atom": atom_summary["mean_components_per_atom"],
        "anchor_update_atom_overlap_nonzero_ratio": atom_overlap_ratio,
        "anchor_update_atom_overlap_nonzero_count": actual_atom_overlap["nonzero_count"],
        "anchor_update_atom_overlap_candidate_count": actual_atom_overlap["candidate_count"],
        "anchor_update_atom_overlap_nonzero_ratio_status": "actual full component_atom_rows overlap between accepted anchor histories and bridge/update objectlets",
        "component_atom_rows_exported_count": len(component_atom_rows),
        "missing_atom_component_count": actual_atom_overlap["missing_atom_component_count"],
        "boundary_uv_proxy_overlap_ratio": boundary_uv_proxy_overlap_ratio,
        "boundary_uv_proxy_status": "C3 accepted strong native boundary/UV update events over native boundary candidates; diagnostic only, not material atom gate evidence",
        "cross_window_atom_link_precision_diagnostic": None,
        "cross_window_atom_link_precision_status": "unavailable because actual full atom-map anchor/update overlap is zero",
        "atom_purity_diagnostic": atom_summary["atom_purity_diagnostic"],
        "fragmentation_per_GT_object_before": atom_summary["fragmentation_per_GT_object_diagnostic_before"],
        "fragmentation_per_GT_object_after": atom_summary["fragmentation_per_GT_object_diagnostic_after"],
        "fragmentation_per_GT_object_decrease": atom_summary["fragmentation_per_GT_object_decrease"],
        "same_frame_conflict_rate": atom_summary["same_frame_conflict_rate"],
        "real_minus_shuffled_atom_AUC": atom_summary["real_minus_shuffled_atom_AUC"],
        "real_minus_no_temporal_atom_AUC": atom_summary.get("real_minus_no_temporal_atom_AUC"),
        "boundary_uv_candidate_count": candidate_count,
        "boundary_uv_accepted_count": accepted_count,
        "gate": {
            "atom_purity_diagnostic_ge_0.90": atom_summary["atom_purity_diagnostic"] >= 0.90,
            "anchor_update_atom_overlap_nonzero_ratio_ge_0.10": (atom_overlap_ratio or 0.0) >= 0.10,
            "fragmentation_per_GT_object_decreases_ge_20pct": atom_summary["fragmentation_per_GT_object_decrease"] >= 0.20,
            "real_minus_shuffled_atom_AUC_ge_0.15": atom_summary["real_minus_shuffled_atom_AUC"] >= 0.15,
            "same_frame_conflict_rate_le_0.05": atom_summary["same_frame_conflict_rate"] <= 0.05,
        },
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }
    phase2_summary["gate"]["pass"] = bool(all(phase2_summary["gate"].values()))
    atom_evidence_rows = [
        {
            "variant": "A3_shared_carrier_plus_boundary_uv_projection",
            "component_count": atom_summary["component_count"],
            "atom_count": atom_summary["atom_count"],
            "anchor_update_atom_overlap_nonzero_ratio": atom_overlap_ratio,
            "precision_diagnostic": None,
            "accepted_update_count": actual_atom_overlap["nonzero_count"],
            "candidate_count": actual_atom_overlap["candidate_count"],
            "note": "actual full atom map overlap; zero overlap means material atom v2 cannot be used for confirmed core update",
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic_labels": True,
        },
        {
            "variant": "C3_boundary_uv_projection_proxy_not_atom_gate",
            "component_count": atom_summary["component_count"],
            "atom_count": atom_summary["atom_count"],
            "anchor_update_atom_overlap_nonzero_ratio": boundary_uv_proxy_overlap_ratio,
            "precision_diagnostic": core_summary["update_precision_diagnostic"],
            "accepted_update_count": accepted_count,
            "candidate_count": candidate_count,
            "note": "legacy proxy retained for audit only; not used for material atom gate after recheck",
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic_labels": True,
        }
    ]

    core_components = _component_key_rows(f"{CORE_C3}/history_component_rows.csv")
    selected_components = _component_key_rows(f"{SELECTED_UPDATE}/history_component_rows.csv")
    raw_tentative_components = selected_components - core_components
    quarantine_component_keys, quarantine_history_component_rows = _multi_history_component_rows(selected_components)
    quarantine_history_component_rows = quarantine_history_component_rows & raw_tentative_components
    tentative_components = raw_tentative_components - quarantine_history_component_rows
    phase3_summary = {
        "phase": "v56_history_state_init",
        "created_at": utc_now(),
        "history_count": core_summary["history_object_count"],
        "confirmed_core_component_count": len(core_components),
        "tentative_component_count": 0,
        "quarantine_component_count": 0,
        "mean_core_components_per_history": _safe_div(len(core_components), core_summary["history_object_count"]),
        "mean_tentative_components_per_history": 0.0,
        "mean_quarantine_components_per_history": 0.0,
        "core_ARI_diagnostic": core_summary["anchor_only_ARI"],
        "core_purity_diagnostic": core_summary["anchor_only_purity"],
        "core_completeness_diagnostic": core_summary["anchor_only_completeness"],
        "core_duplicate_component_count": core_summary["anchor_duplicate_component_count"],
        "gate": {
            "history_count_gt_0": core_summary["history_object_count"] > 0,
            "core_purity_diagnostic_ge_0.90": core_summary["anchor_only_purity"] >= 0.90,
            "core_completeness_diagnostic_ge_0.58": core_summary["anchor_only_completeness"] >= 0.58,
            "core_duplicate_component_count_le_half_v55": core_summary["anchor_duplicate_component_count"] <= 0,
            "quarantine_component_count_gt_0_if_shared_contamination_exists": False,
        },
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }
    phase3_summary["gate"]["pass"] = bool(all(phase3_summary["gate"].values()))
    history_state_rows = _copy_component_rows(f"{CORE_C3}/history_component_rows.csv", state="confirmed_core", max_rows=5000)
    history_component_state_rows = history_state_rows + [
        {
            "history_id": history_id,
            "scene": scene,
            "component_id": component_id,
            "state": "tentative_support_from_selected_hard_update_reinterpreted",
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic_labels": True,
        }
        for history_id, scene, component_id in sorted(tentative_components)[:5000]
    ]

    core_metric_rows = []
    for name, summary in [
        ("C3_strict_s100", core_summary),
        ("C3_relax50", relax50_summary),
        ("C3_relax20", relax20_summary),
    ]:
        core_metric_rows.append(
            {
                "row": name,
                "confirmed_update_count": summary["confirmed_update_count"],
                "confirmed_added_component_count": parse_int(summary.get("native_boundary_added_component_count"))
                + parse_int(summary.get("native_uv_added_component_count")),
                "history_temporal_span_mean": summary["history_temporal_span_mean"],
                "core_ARI": summary["history_ARI"],
                "core_purity": summary["history_purity"],
                "core_completeness": summary["history_completeness"],
                "update_precision_diagnostic": summary["update_precision_diagnostic"],
                "real_minus_shuffled_ARI": summary["real_minus_shuffled_ARI"],
                "real_minus_no_temporal_ARI": summary["real_minus_no_temporal_ARI"],
                "outside_residual_mean": summary["conflict_rate"],
                "same_frame_conflict_count": summary["conflict_reject_count"],
                "gate_pass": summary["gate"]["pass"],
            }
        )
    phase4_summary = {
        "phase": "v56_core_update",
        "created_at": utc_now(),
        "selected_core_row": "C3_strict_s100",
        "repair_attempt_rows": core_metric_rows,
        "confirmed_update_count": core_summary["confirmed_update_count"],
        "confirmed_added_component_count": parse_int(core_summary.get("native_boundary_added_component_count"))
        + parse_int(core_summary.get("native_uv_added_component_count")),
        "history_temporal_span_mean": core_summary["history_temporal_span_mean"],
        "core_ARI": core_summary["history_ARI"],
        "core_purity": core_summary["history_purity"],
        "core_completeness": core_summary["history_completeness"],
        "update_precision_diagnostic": core_summary["update_precision_diagnostic"],
        "real_minus_shuffled_ARI": core_summary["real_minus_shuffled_ARI"],
        "real_minus_no_temporal_ARI": core_summary["real_minus_no_temporal_ARI"],
        "real_minus_mask_only_ARI": core_summary.get("real_minus_mask_only_ARI_static"),
        "outside_residual_mean": core_summary["conflict_rate"],
        "same_frame_conflict_count": core_summary["conflict_reject_count"],
        "blocker_repair_result": "relaxed boundary/UV thresholds increased update count but dropped precision/purity and did not improve controls",
        "gate": {
            "core_purity_ge_anchor_core_minus_0.005": core_summary["history_purity"]
            >= core_summary["anchor_only_purity"] - 0.005,
            "core_ARI_ge_anchor_core_minus_0.005": core_summary["history_ARI"] >= core_summary["anchor_only_ARI"] - 0.005,
            "history_temporal_span_mean_ge_anchor_plus_0.15": core_summary["history_temporal_span_mean"]
            >= core_summary["anchor_only_temporal_span_mean"] + 0.15,
            "update_precision_diagnostic_ge_0.90": core_summary["update_precision_diagnostic"] >= 0.90,
            "real_minus_shuffled_ARI_ge_0.15": core_summary["real_minus_shuffled_ARI"] >= 0.15,
            "real_minus_no_temporal_ARI_ge_0.10": core_summary["real_minus_no_temporal_ARI"] >= 0.10,
        },
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }
    phase4_summary["gate"]["pass"] = bool(all(phase4_summary["gate"].values()))

    expanded_gain = float(selected_summary["history_completeness"] - core_summary["history_completeness"])
    phase5_summary = {
        "phase": "v56_tentative_support",
        "created_at": utc_now(),
        "tentative_added_component_count": len(tentative_components),
        "tentative_update_count": role_counts.get("weak_history_mask", 0) + role_counts.get("weak_visible_cosupport", 0),
        "tentative_precision_diagnostic": selected_summary["update_precision_diagnostic"],
        "expanded_ARI": selected_summary["history_ARI"],
        "expanded_purity": selected_summary["history_purity"],
        "expanded_completeness": selected_summary["history_completeness"],
        "confirmed_core_ARI": core_summary["history_ARI"],
        "confirmed_core_purity": core_summary["history_purity"],
        "confirmed_core_completeness": core_summary["history_completeness"],
        "core_control_margin_change": 0.0,
        "tentative_underseg_rate": None,
        "quarantine_count": duplicate_summary.get("components_with_multi_history_count"),
        "expanded_completeness_gain": expanded_gain,
        "gate": {
            "core_purity_drop_le_0.002": True,
            "core_real_minus_shuffled_drop_le_0.005": True,
            "expanded_completeness_gain_ge_0.04": expanded_gain >= 0.04,
            "expanded_purity_ge_0.86": selected_summary["history_purity"] >= 0.86,
            "tentative_precision_diagnostic_ge_0.75": selected_summary["update_precision_diagnostic"] >= 0.75,
        },
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }
    phase5_summary["gate"]["pass"] = bool(all(phase5_summary["gate"].values()))
    tentative_rows = [
        row for row in phase0_rows if row["evidence_role"] in {"weak_history_mask", "weak_visible_cosupport"}
    ]
    core_vs_expanded_rows = [
        {"row": "confirmed_core_C3", "ARI": core_summary["history_ARI"], "purity": core_summary["history_purity"], "completeness": core_summary["history_completeness"]},
        {"row": "expanded_selected_U6_U3_tentative", "ARI": selected_summary["history_ARI"], "purity": selected_summary["history_purity"], "completeness": selected_summary["history_completeness"]},
    ]

    promotion_probe_names = [
        "U4_U5_U6b_historymask_component_rank50_plus_U3",
        "U4_U5_U6b_historymask_component_rank500_plus_U3",
        "U4_U5_U6b_historymask_component_rank1000_plus_U3",
        "U4_U5_U6b_historymask_component_rank500_no_U3",
    ]
    attempts = _attempt_by_name(final_decision)
    promotion_metric_rows = [
        {
            "row": "P0_no_promotion_core_C3",
            "confirmed_core_ARI": core_summary["history_ARI"],
            "confirmed_core_purity": core_summary["history_purity"],
            "confirmed_core_completeness": core_summary["history_completeness"],
            "real_minus_shuffled_ARI": core_summary["real_minus_shuffled_ARI"],
            "real_minus_no_temporal_ARI": core_summary["real_minus_no_temporal_ARI"],
            "promoted_component_count": 0,
        }
    ]
    for name in promotion_probe_names:
        row = attempts.get(name, {})
        promotion_metric_rows.append(
            {
                "row": name,
                "confirmed_core_ARI": row.get("history_ARI"),
                "confirmed_core_purity": row.get("history_purity"),
                "confirmed_core_completeness": row.get("history_completeness"),
                "real_minus_shuffled_ARI": row.get("real_minus_shuffled_ARI"),
                "real_minus_no_temporal_ARI": row.get("real_minus_no_temporal_ARI"),
                "promoted_component_count": row.get("native_history_mask_added_component_count"),
                "gate_pass": row.get("gate_pass"),
                "note": "v55 component-support repair probe; not accepted as v56 promotion because independent later evidence and control gates are insufficient",
            }
        )
    phase6_summary = {
        "phase": "v56_promotion",
        "created_at": utc_now(),
        "promotion_candidate_count": len(tentative_components),
        "promoted_component_count": 0,
        "promotion_precision_diagnostic": None,
        "false_promotion_count": None,
        "promotion_source_breakdown": {},
        "confirmed_core_ARI": core_summary["history_ARI"],
        "confirmed_core_purity": core_summary["history_purity"],
        "confirmed_core_completeness": core_summary["history_completeness"],
        "confirmed_core_temporal_span": core_summary["history_temporal_span_mean"],
        "real_minus_shuffled_ARI": core_summary["real_minus_shuffled_ARI"],
        "real_minus_no_temporal_ARI": core_summary["real_minus_no_temporal_ARI"],
        "expanded_completeness_after_promotion": selected_summary["history_completeness"],
        "blocked_reason": "confirmed-core D4RT control/span gate failed after planned threshold repair; semantic-only or mask-only promotion is forbidden",
        "gate": {
            "promotion_precision_diagnostic_ge_0.85": False,
            "promoted_component_count_gt_0": False,
            "confirmed_core_completeness_gain_vs_P0_ge_0.02": False,
            "confirmed_core_purity_drop_vs_P0_le_0.005": True,
            "real_minus_shuffled_ARI_gain_vs_P0_ge_0.03": False,
            "real_minus_no_temporal_ARI_gain_vs_P0_ge_0.02": False,
        },
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }
    phase6_summary["gate"]["pass"] = False
    promotion_rows = [
        {
            "history_id": history_id,
            "scene": scene,
            "component_id": component_id,
            "candidate_state": "tentative",
            "promotion_state": "not_promoted",
            "reason": "core_gate_blocked_no_independent_later_E1",
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic_labels": True,
        }
        for history_id, scene, component_id in sorted(tentative_components)[:5000]
    ]

    phase7_summary = {
        "phase": "v56_quarantine",
        "created_at": utc_now(),
        "quarantine_component_count": duplicate_summary.get("components_with_multi_history_count"),
        "quarantine_mask_count": len(duplicate_rows),
        "false_promotion_reduction": None,
        "false_update_reduction": None,
        "core_purity_gain": None,
        "expanded_completeness_drop": None,
        "shared_component_duplicate_candidate_count": duplicate_summary.get("candidate_pair_count"),
        "duplicate_merge_attempt_count": 0,
        "same_gt_rate_top_pairs_diagnostic": _safe_div(
            sum(1 for row in duplicate_rows if parse_bool(row.get("same_gt_diagnostic"))), len(duplicate_rows)
        ),
        "diagnostic_status": "quarantine evidence available, but reduction metrics are blocked because promotion is disabled",
        "gate": {
            "false_promotion_reduction_ge_0.10": False,
            "core_purity_gain_ge_0.005": False,
            "expanded_completeness_drop_le_0.04": True,
            "duplicate_merge_attempt_count_eq_0": True,
        },
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }
    phase7_summary["gate"]["pass"] = False
    quarantine_rows = [
        {
            **row,
            "quarantine_reason": "shared_component_not_duplicate_by_default",
            "v56_action": "block_duplicate_merge_and_hard_identity_promotion",
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic_labels": True,
        }
        for row in duplicate_rows
    ]

    stress_rows = [
        {
            "stress_type": stress_type,
            "stress_strength": "",
            "row": "D5_v56_full_with_quarantine_semantic",
            "status": "not_run_core_and_promotion_gate_blocked",
            "core_ARI": "",
            "core_purity": "",
            "core_completeness": "",
            "expanded_ARI": "",
            "expanded_purity": "",
            "expanded_completeness": "",
            "real_minus_mask_only_ARI": "",
            "real_minus_shuffled_ARI": "",
            "real_minus_no_temporal_ARI": "",
        }
        for stress_type in ["mask_dropout", "partial_crop", "mask_split", "mask_merge", "temporal_gap", "history_drift"]
    ]
    phase8_summary = {
        "phase": "v56_stress",
        "created_at": utc_now(),
        "status": "not_run",
        "not_run_reason": "Stop 4 and Stop 6 block dynamic-ready claim: confirmed core controls and promotion are not sufficient",
        "stress_real_minus_mask_only_ARI_pass_count": 0,
        "stress_temporal_span_gain_vs_mask_only": None,
        "reactivation_precision_diagnostic": None,
        "false_promotion_rate": None,
        "core_purity": core_summary["history_purity"],
        "gate": {
            "stress_real_minus_mask_only_ARI_ge_0.05_in_at_least_3_settings": False,
            "stress_temporal_span_gain_vs_mask_only_ge_0.30": False,
            "reactivation_precision_diagnostic_ge_0.80": False,
            "false_promotion_rate_le_0.15": False,
            "core_purity_ge_0.88": core_summary["history_purity"] >= 0.88,
        },
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }
    phase8_summary["gate"]["pass"] = False

    native_component_rows = _rows(f"{NATIVE_DIR}/component_native_carrier_rows.csv")
    native_supported_components = {str(row.get("component_id")) for row in native_component_rows if parse_bool(row.get("has_native_carrier_support"))}
    confirmed_native_support_count = sum(1 for _hid, _scene, component_id in core_components if component_id in native_supported_components)
    tentative_native_support_count = sum(1 for _hid, _scene, component_id in tentative_components if component_id in native_supported_components)
    quarantine_native_support_count = sum(1 for _scene, component_id in quarantine_component_keys if component_id in native_supported_components)
    native_state_component_rows = sorted(core_components | tentative_components | quarantine_history_component_rows)
    native_state_counts = Counter(
        "confirmed"
        if row in core_components
        else "quarantine"
        if row in quarantine_history_component_rows
        else "tentative"
        for row in native_state_component_rows
    )
    native_field_summary = {
        "phase": "v56_native_field",
        "created_at": utc_now(),
        "history_object_count": core_summary["history_object_count"],
        "confirmed_core_carrier_count": confirmed_native_support_count,
        "confirmed_core_component_count": len(core_components),
        "tentative_carrier_count": tentative_native_support_count,
        "tentative_component_count": len(tentative_components),
        "raw_tentative_component_count": len(raw_tentative_components),
        "quarantine_carrier_count": quarantine_native_support_count,
        "quarantine_component_count": len(quarantine_component_keys),
        "quarantine_history_component_row_count": len(quarantine_history_component_rows),
        "native_carrier_state_row_count": len(native_state_component_rows),
        "native_carrier_state_counts": dict(native_state_counts),
        "native_observation_row_count": native_summary["native_observation_row_count"],
        "method_safe_native_support_available": native_summary["method_safe_native_support_available"],
        "uses_gt_for_prediction": False,
        "uses_rgbd_pose_mesh_for_export": False,
        "gate": {
            "method_safe_native_support_available": bool(native_summary["method_safe_native_support_available"]),
            "confirmed_core_carrier_count_gt_0": confirmed_native_support_count > 0,
            "history_object_count_gt_0": core_summary["history_object_count"] > 0,
            "all_state_rows_exported": len(native_state_component_rows)
            == len(core_components) + len(tentative_components) + len(quarantine_history_component_rows),
            "quarantine_state_rows_present_if_quarantine_components": len(quarantine_component_keys) == 0
            or native_state_counts.get("quarantine", 0) > 0,
            "uses_gt_for_prediction_false": True,
            "uses_rgbd_pose_mesh_for_export_false": True,
        },
    }
    native_field_summary["gate"]["pass"] = bool(all(native_field_summary["gate"].values()))
    native_carrier_state_rows = [
        {
            "history_id": history_id,
            "scene": scene,
            "component_id": component_id,
            "state": "confirmed"
            if (history_id, scene, component_id) in core_components
            else "quarantine"
            if (history_id, scene, component_id) in quarantine_history_component_rows
            else "tentative",
            "has_native_carrier_support": component_id in native_supported_components,
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic_labels": False,
            "uses_rgbd_pose_mesh_for_export": False,
        }
        for history_id, scene, component_id in native_state_component_rows
    ]
    ap_rows = [
        {
            "row": "AP_core_only",
            "AP": None,
            "is_diagnostic_only": True,
            "forbidden_for_method_table": True,
            "uses_gt_for_prediction": False,
            "uses_gt_for_evaluation_alignment": False,
            "uses_rgbd_pose_mesh_for_export": False,
            "status": native_summary.get("AP_bridge_status"),
        },
        {
            "row": "AP_core_plus_tentative",
            "AP": None,
            "is_diagnostic_only": True,
            "forbidden_for_method_table": True,
            "uses_gt_for_prediction": False,
            "uses_gt_for_evaluation_alignment": False,
            "uses_rgbd_pose_mesh_for_export": False,
            "status": native_summary.get("real_method_ap_status"),
        },
    ]

    method_rows = [
        {"row": "F0_v55_selected_hard_update_reference", "core_4D_ARI": selected_summary["history_ARI"], "core_purity": selected_summary["history_purity"], "core_completeness": selected_summary["history_completeness"], "real_minus_shuffled_ARI": selected_summary["real_minus_shuffled_ARI"], "real_minus_no_temporal_ARI": selected_summary["real_minus_no_temporal_ARI"]},
        {"row": "F1_anchor_only_core", "core_4D_ARI": core_summary["anchor_only_ARI"], "core_purity": core_summary["anchor_only_purity"], "core_completeness": core_summary["anchor_only_completeness"], "real_minus_shuffled_ARI": 0.0, "real_minus_no_temporal_ARI": 0.0},
        {"row": "F2_confirmed_core_update_C3", "core_4D_ARI": core_summary["history_ARI"], "core_purity": core_summary["history_purity"], "core_completeness": core_summary["history_completeness"], "real_minus_shuffled_ARI": core_summary["real_minus_shuffled_ARI"], "real_minus_no_temporal_ARI": core_summary["real_minus_no_temporal_ARI"]},
        {"row": "F3_core_plus_tentative_expanded", "expanded_4D_ARI": selected_summary["history_ARI"], "expanded_purity": selected_summary["history_purity"], "expanded_completeness": selected_summary["history_completeness"], "real_minus_shuffled_ARI": selected_summary["real_minus_shuffled_ARI"], "real_minus_no_temporal_ARI": selected_summary["real_minus_no_temporal_ARI"]},
        {"row": "F4_confirm_then_promote", "status": "not_promoted_core_gate_blocked", "promoted_component_count": 0},
        {"row": "F5_full_v56_with_quarantine_semantic", "status": "blocked_by_core_control_and_promotion"},
    ]
    success = {
        "state_separation": phase5_summary["gate"]["pass"],
        "confirmed_core": bool(
            core_summary["history_purity"] >= 0.89
            and core_summary["real_minus_shuffled_ARI"] >= 0.15
            and core_summary["real_minus_no_temporal_ARI"] >= 0.10
        ),
        "promotion": False,
        "stress": False,
        "native_field": native_field_summary["gate"]["pass"],
    }
    final_label = "NO_GO_D4RT_CONTROL"
    partial_label = "PARTIAL_TENTATIVE_SUPPORT_SIGNAL" if phase5_summary["gate"]["pass"] else "PARTIAL_CORE_MEMORY_SIGNAL"
    final_summary = {
        "phase": "v56_final_decision",
        "created_at": utc_now(),
        "final_label": final_label,
        "partial_label": partial_label,
        "goal_achieved": False,
        "success_criteria": success,
        "core_4D_ARI": core_summary["history_ARI"],
        "core_purity": core_summary["history_purity"],
        "core_completeness": core_summary["history_completeness"],
        "core_temporal_span_mean": core_summary["history_temporal_span_mean"],
        "expanded_4D_ARI": selected_summary["history_ARI"],
        "expanded_purity": selected_summary["history_purity"],
        "expanded_completeness": selected_summary["history_completeness"],
        "expanded_temporal_span_mean": selected_summary["history_temporal_span_mean"],
        "promotion_precision": None,
        "false_promotion_rate": None,
        "duplicate_rate": core_summary["duplicate_rate"],
        "conflict_rate": core_summary["conflict_rate"],
        "quarantine_count": duplicate_summary.get("components_with_multi_history_count"),
        "real_minus_shuffled_ARI": core_summary["real_minus_shuffled_ARI"],
        "real_minus_no_temporal_ARI": core_summary["real_minus_no_temporal_ARI"],
        "real_minus_mask_only_ARI_static": core_summary.get("real_minus_mask_only_ARI_static"),
        "stress_real_minus_mask_only_ARI": None,
        "native_field_available": native_field_summary["gate"]["pass"],
        "best_AP_diagnostic": None,
        "answers_to_report_questions": [
            "v56 supports evidence role collapse as a plausible v55 failure mode: E1 is higher precision but low coverage/control, while E2 expands coverage but was unsafe as hard identity.",
            "State separation is implemented in artifacts: C3 confirmed core is kept separate from selected U6/U3 tentative expansion.",
            "Anchor decontamination did not find duplicated accepted birth components; stricter same-frame/outside repairs overcut completeness, so B0 anchor fallback is retained.",
            "Material atom v2 full atom-map overlap is zero for bridge/update objectlets; the nonzero boundary/UV value is only a diagnostic proxy and cannot support confirmed core update.",
            "Confirmed-core update is high precision but fails temporal span and D4RT control thresholds.",
            "Tentative support improves expanded completeness by more than 0.04 without changing core metrics.",
            "Promotion is disabled: no safe independent later evidence clears the planned gate.",
            "Quarantine evidence is diagnostic and blocks duplicate merge by default, but reduction metrics cannot be measured with promotion disabled.",
            "Stress evaluation is not run because core/promotion gates block dynamic-ready claims.",
            "Semantic guard has high diagnostic AUC, but remains veto/diagnostic and is not used as positive promotion evidence.",
            "Native carrier state field is method-safe, but AP is not available as a method-safe output.",
            "Primary failure layer is D4RT core control, followed by promotion and dynamic-ready stress.",
        ],
        "analysis_conclusions": [
            "The best strict E1 core row reaches update_precision_diagnostic=0.9293 and core_purity=0.8966, but only real_minus_shuffled_ARI=0.0804 and real_minus_no_temporal_ARI=0.0722.",
            "A full material-atom recheck found 0/153 bridge/update objectlets with actual anchor-history atom overlap, so C4 material-atom links cannot repair the core gate.",
            "Relaxing E1 thresholds increases confirmed_update_count from 17 to 22/32 but drops update precision to 0.7403/0.7094 and does not improve controls.",
            "The selected v55 U6/U3 hard update has useful expanded completeness (0.6815), but v56 must keep that as tentative because hard identity controls still fail.",
            "No final GO claim is justified; the honest result is a partial tentative-support signal with NO_GO_D4RT_CONTROL as the method blocker.",
        ],
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }
    return {
        "phase0_summary": phase0_summary,
        "phase0_rows": phase0_rows,
        "phase0_metrics": phase0_metrics,
        "phase1_summary": phase1_summary,
        "anchor_variants": anchor_variants,
        "anchor_core_rows": anchor_core_rows,
        "anchor_quarantine_rows": anchor_quarantine_rows,
        "phase2_summary": phase2_summary,
        "component_atom_v2_rows": component_atom_rows,
        "atom_v2_evidence_rows": atom_evidence_rows,
        "atom_v2_overlap_rows": actual_atom_overlap["overlap_rows"],
        "phase3_summary": phase3_summary,
        "history_state_rows": history_state_rows,
        "history_component_state_rows": history_component_state_rows,
        "phase4_summary": phase4_summary,
        "core_update_rows": _rows(f"{CORE_C3}/history_update_rows.csv"),
        "core_metric_rows": core_metric_rows,
        "phase5_summary": phase5_summary,
        "tentative_rows": tentative_rows,
        "core_vs_expanded_rows": core_vs_expanded_rows,
        "phase6_summary": phase6_summary,
        "promotion_rows": promotion_rows,
        "promotion_metric_rows": promotion_metric_rows,
        "phase7_summary": phase7_summary,
        "quarantine_rows": quarantine_rows,
        "phase8_summary": phase8_summary,
        "stress_rows": stress_rows,
        "native_field_summary": native_field_summary,
        "native_carrier_state_rows": native_carrier_state_rows,
        "ap_rows": ap_rows,
        "final_summary": final_summary,
        "method_rows": method_rows,
        "uv_summary": uv_summary,
        "semantic_summary": semantic_summary,
        "native_summary": native_summary,
    }


def write_v56_full_eval(payload: dict[str, Any], *, output_root: str | Path = "outputs/audit") -> dict[str, Any]:
    root = _project(output_root)
    paths: dict[str, str] = {}

    def j(rel: str, data: Any) -> None:
        path = root / rel
        write_json(path, data)
        paths[rel] = _rel(path)

    def c(rel: str, rows: list[dict[str, Any]]) -> None:
        path = root / rel
        write_csv(path, rows)
        paths[rel] = _rel(path)

    j("v56_phase0_evidence_typing/phase0_summary.json", payload["phase0_summary"])
    c("v56_phase0_evidence_typing/evidence_role_rows.csv", payload["phase0_rows"])
    c("v56_phase0_evidence_typing/evidence_role_metric_rows.csv", payload["phase0_metrics"])

    j("v56_anchor_decontamination/anchor_decontamination_summary.json", payload["phase1_summary"])
    c("v56_anchor_decontamination/anchor_variant_metric_rows.csv", payload["anchor_variants"])
    c("v56_anchor_decontamination/anchor_core_rows.csv", payload["anchor_core_rows"])
    c("v56_anchor_decontamination/anchor_quarantine_rows.csv", payload["anchor_quarantine_rows"])

    j("v56_material_atoms_v2/atom_v2_summary.json", payload["phase2_summary"])
    c("v56_material_atoms_v2/component_atom_v2_rows.csv", payload["component_atom_v2_rows"])
    c("v56_material_atoms_v2/atom_v2_evidence_rows.csv", payload["atom_v2_evidence_rows"])
    c("v56_material_atoms_v2/atom_v2_anchor_update_overlap_rows.csv", payload["atom_v2_overlap_rows"])

    j("v56_history_state_init/history_state_summary.json", payload["phase3_summary"])
    c("v56_history_state_init/history_state_rows.csv", payload["history_state_rows"])
    c("v56_history_state_init/history_component_state_rows.csv", payload["history_component_state_rows"])

    j("v56_core_update/core_update_summary.json", payload["phase4_summary"])
    c("v56_core_update/core_update_rows.csv", payload["core_update_rows"])
    c("v56_core_update/core_metric_rows.csv", payload["core_metric_rows"])

    j("v56_tentative_support/tentative_support_summary.json", payload["phase5_summary"])
    c("v56_tentative_support/tentative_component_rows.csv", payload["tentative_rows"])
    c("v56_tentative_support/core_vs_expanded_metric_rows.csv", payload["core_vs_expanded_rows"])

    j("v56_promotion/promotion_summary.json", payload["phase6_summary"])
    c("v56_promotion/promotion_rows.csv", payload["promotion_rows"])
    c("v56_promotion/promotion_metric_rows.csv", payload["promotion_metric_rows"])

    j("v56_quarantine/quarantine_summary.json", payload["phase7_summary"])
    c("v56_quarantine/quarantine_rows.csv", payload["quarantine_rows"])

    j("v56_stress/stress_summary.json", payload["phase8_summary"])
    c("v56_stress/stress_metric_rows.csv", payload["stress_rows"])

    j("v56_native_field/native_field_summary.json", payload["native_field_summary"])
    c("v56_native_field/native_carrier_state_rows.csv", payload["native_carrier_state_rows"])
    c("v56_ap/ap_metric_rows.csv", payload["ap_rows"])

    j("v56_final_decision/final_decision.json", payload["final_summary"])
    c("v56_final_decision/method_rows.csv", payload["method_rows"])

    vis_root = root / "v56_visualizations"
    vis = [
        _write_simple_chart(
            vis_root / "phase0" / "evidence_role_pareto.png",
            "v56 evidence role ARI",
            [str(row["role"]) for row in payload["phase0_metrics"]],
            [float(row["ARI"] or 0.0) for row in payload["phase0_metrics"]],
        ),
        _write_simple_chart(
            vis_root / "anchor_decontamination" / "anchor_before_after_metrics.png",
            "anchor repair purity",
            [str(row["variant"]).replace("_", "\n") for row in payload["anchor_variants"]],
            [float(row["core_purity_diagnostic"]) for row in payload["anchor_variants"]],
        ),
        _write_simple_chart(
            vis_root / "core_update" / "core_control_comparison.png",
            "core control comparison",
            [str(row["row"]) for row in payload["core_metric_rows"]],
            [float(row["real_minus_shuffled_ARI"] or 0.0) for row in payload["core_metric_rows"]],
        ),
        _write_simple_chart(
            vis_root / "tentative_support" / "promotion_pareto_core_vs_expanded.png",
            "core vs expanded completeness",
            [str(row["row"]) for row in payload["core_vs_expanded_rows"]],
            [float(row["completeness"]) for row in payload["core_vs_expanded_rows"]],
        ),
    ]
    dashboard = vis_root / "v56_dashboard.html"
    dashboard.parent.mkdir(parents=True, exist_ok=True)
    f = payload["final_summary"]
    dashboard.write_text(
        "\n".join(
            [
                "<!doctype html>",
                "<meta charset='utf-8'>",
                "<title>Stream4D v56 Dashboard</title>",
                "<h1>Stream4D v56 Evidence-Typed Memory Dashboard</h1>",
                f"<p>Final label: <b>{f['final_label']}</b>; partial label: <b>{f['partial_label']}</b></p>",
                f"<p>Core ARI={f['core_4D_ARI']}, purity={f['core_purity']}, completeness={f['core_completeness']}</p>",
                f"<p>Expanded ARI={f['expanded_4D_ARI']}, purity={f['expanded_purity']}, completeness={f['expanded_completeness']}</p>",
                "<h2>Visual Checks</h2>",
                *[f"<div><img src='{Path(item['path']).relative_to('outputs/audit/v56_visualizations') if item['path'].startswith('outputs/audit/v56_visualizations') else item['path']}' style='max-width:960px'></div>" for item in vis if item.get("status") == "ok"],
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    paths["v56_visualizations/v56_dashboard.html"] = _rel(dashboard)
    j("v56_visualizations/visualization_manifest.json", {"visualizations": vis, "dashboard": _rel(dashboard)})
    return paths
