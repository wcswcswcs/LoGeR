#!/usr/bin/env python3
"""Build ACL2 v83 Phase1 unified clue matrix.

The matrix is intentionally evidence-preserving: unavailable READ/SWA/RADIO
fields stay empty and are reported as missing instead of being backfilled from
semantic labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping


DEFAULT_OUT_DIR = Path(
    "results/acl2_v83tf_clue_sufficiency_vs_action_misuse/phase1_unified_clue_matrix"
)
DEFAULT_V82_PAIR_BANK = Path(
    "results/acl2_v82tf_swa_carrier_semantic_scale_handoff/"
    "phase2_swa_pair_bank_v2/swa_pair_bank_v2.csv"
)
DEFAULT_V82_ROUTE_JOINED = Path(
    "results/acl2_v82tf_swa_carrier_semantic_scale_handoff/"
    "phase12_contextual_route_rule_search/contextual_route_rule_rows_joined.csv"
)
DEFAULT_V80_CASE_BANK_DIR = Path(
    "results/kitti01_hmc_v2/acl2_v80tf_multiseq_goodbad_semantic_three_memory_control/"
    "report_final/phase1_three_memory_case_bank"
)
DEFAULT_DIRECT_HOOK_AUDIT = Path(
    "results/acl2_v83tf_clue_sufficiency_vs_action_misuse/"
    "phase1_direct_hook_repair_audit/direct_hook_repair_audit.csv"
)


GROUP_FIELDS = {
    "G0_geometry_basic": [
        "raw_overlap_residual",
        "overlap_scale_residual",
        "boundary_jump",
        "future_after_overlap",
        "confidence_weighted_residual",
        "prev_to_curr_scale_jump",
        "window5_joint_sim3_rmse",
        "window5_subchunk_scale_cv",
        "downstream_future_consistency",
        "low_observability_score",
        "regime_shift_score",
        "local_sim3_ate",
        "head_to_tail",
        "scale_cv",
        "intra_scale_variance",
        "J_mid",
        "J_long",
        "J_short",
    ],
    "G1_dense_semantic": [
        "stable_mass",
        "harm_mass",
        "context_mass",
        "semantic_confidence_mean",
        "patch_purity_mean",
        "lowtrust_stuff_ratio",
        "sky_context_ratio",
        "dynamic_thing_ratio",
        "road_edge_continuity",
        "corridor_stability",
        "overlap_semantic_agreement",
    ],
    "G2_thingstuff_panoptic": [
        "thing_moving_ratio",
        "thing_static_ratio",
        "stuff_static_ratio",
        "structure_ratio",
        "track_lifespan_proxy",
        "source_type_consistency",
    ],
    "G3_RADIO_topology": [
        "same_object_ratio",
        "cross_object_boundary_ratio",
        "object_boundary_score",
        "object_interior_score",
        "temporal_stability",
        "radio_lowtrust_ratio",
    ],
    "G4_internal_READ": [
        "READ_used_stable_mass",
        "READ_used_harm_mass",
        "QK_pair_compatibility",
        "query_risk_mass",
        "read_entropy",
    ],
    "G5_internal_SWA": [
        "current_Q_alignment",
        "cache_K_alignment",
        "cache_V_alignment",
        "K_risk_delta",
        "V_protect_delta",
        "route_mass",
        "head_layer_sensitivity",
        "actual_vs_random_route_delta",
    ],
    "G6_merge_gauge": [
        "boundary_transform_residual",
        "merge_raw_overlap_residual",
        "postmerge_pose_sensitivity",
        "reset_relative_position",
        "gauge_hold_signal",
    ],
    "G7_TTT": [
        "selected_low_support_ratio",
        "continuous_low_support_cluster_len",
        "update_conflict",
        "post_zp_delta",
        "write_mass_stable",
        "write_mass_harm",
        "write_mass_context",
    ],
}


APPLICABLE_FIELDS = {
    "v82_swa_adjacent_pair": {
        "G0_geometry_basic": [
            "raw_overlap_residual",
            "overlap_scale_residual",
            "boundary_jump",
            "future_after_overlap",
            "confidence_weighted_residual",
            "prev_to_curr_scale_jump",
        ],
        "G1_dense_semantic": [
            "stable_mass",
            "harm_mass",
            "context_mass",
            "semantic_confidence_mean",
        ],
        "G2_thingstuff_panoptic": [],
        "G3_RADIO_topology": [
            "same_object_ratio",
            "cross_object_boundary_ratio",
            "temporal_stability",
        ],
        "G4_internal_READ": GROUP_FIELDS["G4_internal_READ"],
        "G5_internal_SWA": GROUP_FIELDS["G5_internal_SWA"],
        "G6_merge_gauge": ["merge_raw_overlap_residual", "boundary_transform_residual"],
        "G7_TTT": [],
    },
    "v80_long_window": {
        "G0_geometry_basic": [
            "window5_joint_sim3_rmse",
            "window5_subchunk_scale_cv",
            "downstream_future_consistency",
            "low_observability_score",
            "regime_shift_score",
            "J_long",
        ],
        "G1_dense_semantic": ["road_edge_continuity", "corridor_stability"],
        "G2_thingstuff_panoptic": [],
        "G3_RADIO_topology": [],
        "G4_internal_READ": [],
        "G5_internal_SWA": [],
        "G6_merge_gauge": [],
        "G7_TTT": ["update_conflict", "post_zp_delta"],
    },
    "v80_mid_adjacent_pair": {
        "G0_geometry_basic": [
            "raw_overlap_residual",
            "boundary_jump",
            "future_after_overlap",
            "scale_cv",
            "J_mid",
        ],
        "G1_dense_semantic": [
            "stable_mass",
            "harm_mass",
            "context_mass",
            "overlap_semantic_agreement",
        ],
        "G2_thingstuff_panoptic": [],
        "G3_RADIO_topology": ["same_object_ratio", "cross_object_boundary_ratio"],
        "G4_internal_READ": GROUP_FIELDS["G4_internal_READ"],
        "G5_internal_SWA": GROUP_FIELDS["G5_internal_SWA"],
        "G6_merge_gauge": ["merge_raw_overlap_residual", "boundary_transform_residual"],
        "G7_TTT": [],
    },
    "v80_short_read_case": {
        "G0_geometry_basic": [
            "local_sim3_ate",
            "head_to_tail",
            "scale_cv",
            "intra_scale_variance",
            "J_short",
        ],
        "G1_dense_semantic": [
            "stable_mass",
            "harm_mass",
            "context_mass",
            "semantic_confidence_mean",
            "lowtrust_stuff_ratio",
        ],
        "G2_thingstuff_panoptic": [
            "thing_moving_ratio",
            "thing_static_ratio",
            "stuff_static_ratio",
        ],
        "G3_RADIO_topology": ["object_boundary_score", "temporal_stability"],
        "G4_internal_READ": GROUP_FIELDS["G4_internal_READ"],
        "G5_internal_SWA": GROUP_FIELDS["G5_internal_SWA"],
        "G6_merge_gauge": [],
        "G7_TTT": [],
    },
}

META_FIELDS = [
    "row_id",
    "row_scope",
    "seq",
    "prev_chunk",
    "curr_chunk",
    "chunk_id",
    "chunk_start",
    "chunk_end",
    "frame_start",
    "frame_end",
    "case_type",
    "base_case_type",
    "target_label",
    "label_role",
    "has_radio",
    "source_path",
    "source_note",
    "missing_fields_from_source",
    "derived_feature_notes",
]

OUTPUT_FIELDS = META_FIELDS + [field for fields in GROUP_FIELDS.values() for field in fields]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v82-pair-bank", type=Path, default=DEFAULT_V82_PAIR_BANK)
    parser.add_argument("--v82-route-joined", type=Path, default=DEFAULT_V82_ROUTE_JOINED)
    parser.add_argument("--v80-case-bank-dir", type=Path, default=DEFAULT_V80_CASE_BANK_DIR)
    parser.add_argument("--direct-hook-audit-csv", type=Path, default=DEFAULT_DIRECT_HOOK_AUDIT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: list[str] | None = None) -> None:
    rows = list(rows)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: serialize_cell(row.get(key, "")) for key in fields})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def serialize_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def is_present(value: Any) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    if text == "":
        return False
    if text.lower() in {"nan", "none", "null"}:
        return False
    return True


def safe_float(value: Any) -> float | None:
    if not is_present(value):
        return None
    try:
        out = float(str(value))
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def safe_int(value: Any) -> int | None:
    fval = safe_float(value)
    return None if fval is None else int(fval)


def boolish(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def seq_norm(value: Any) -> str:
    text = str(value or "").strip()
    if len(text) >= 2 and text[:2].isdigit():
        return text[:2]
    return text.zfill(2) if text else ""


def base_case(case_type: Any, base_case_type: Any = "") -> str:
    if is_present(base_case_type):
        return str(base_case_type)
    text = str(case_type)
    return "bad" if text.startswith("bad") else "good"


def label_role(case_type: Any, base_case_type: Any = "") -> str:
    text = str(case_type)
    if "false_positive" in text:
        return "false_positive"
    return base_case(case_type, base_case_type)


def target_label(case_type: Any, base_case_type: Any = "") -> int:
    return 1 if base_case(case_type, base_case_type) == "bad" else 0


def maybe_product(a: Any, b: Any) -> float | None:
    aval = safe_float(a)
    bval = safe_float(b)
    if aval is None or bval is None:
        return None
    return aval * bval


def finite_float(value: Any) -> float | None:
    fval = safe_float(value)
    return fval if fval is not None and math.isfinite(fval) else None


def blank_row(row_scope: str) -> dict[str, Any]:
    row = {field: "" for field in OUTPUT_FIELDS}
    row["row_scope"] = row_scope
    row["has_radio"] = "false"
    return row


def route_key(row: Mapping[str, Any]) -> tuple[str, int] | None:
    seq = seq_norm(row.get("seq_base") or row.get("seq"))
    chunk = safe_int(row.get("chunk") or row.get("curr_chunk"))
    if not seq or chunk is None:
        return None
    return (seq, chunk)


def audit_key(row: Mapping[str, Any]) -> tuple[str, int] | None:
    seq = seq_norm(row.get("seq"))
    chunk = safe_int(row.get("chunk"))
    if not seq or chunk is None:
        return None
    return (seq, chunk)


def _load_torch_payload(path: Path) -> Any:
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _tensor_mean(value: Any) -> float | None:
    import torch

    if not torch.is_tensor(value):
        return None
    tensor = value.float()
    if tensor.numel() == 0:
        return None
    return float(tensor.mean().item())


def _normalized_entropy(value: Any) -> float | None:
    import torch

    if not torch.is_tensor(value):
        return None
    tensor = value.float().flatten().clamp_min(0)
    if tensor.numel() == 0:
        return None
    total = tensor.sum()
    if not torch.isfinite(total) or float(total.item()) <= 0:
        return None
    prob = tensor / total
    entropy = -(prob * torch.log(prob.clamp_min(1e-12))).sum()
    denom = math.log(float(prob.numel()))
    if denom <= 0:
        return None
    return float((entropy / denom).item())


def _cosine_mean(a: Any, b: Any) -> float | None:
    import torch

    if not torch.is_tensor(a) or not torch.is_tensor(b):
        return None
    if a.shape != b.shape or a.numel() == 0:
        return None
    avec = a.float().reshape(-1, int(a.shape[-1]))
    bvec = b.float().reshape(-1, int(b.shape[-1]))
    denom = avec.norm(dim=-1) * bvec.norm(dim=-1)
    valid = denom > 1e-8
    if not bool(valid.any()):
        return None
    cos = (avec * bvec).sum(dim=-1)[valid] / denom[valid]
    return float(cos.mean().item())


def _clip01(value: float | None) -> float | None:
    if value is None:
        return None
    return max(0.0, min(1.0, float(value)))


def _extract_read_features(path: Path) -> tuple[dict[str, Any], str]:
    payload = _load_torch_payload(path)
    if not isinstance(payload, dict):
        return {}, "read_feature_extract_failed: payload_not_dict"
    debug = payload.get("debug") if isinstance(payload.get("debug"), dict) else {}
    tensors = payload.get("tensors") if isinstance(payload.get("tensors"), dict) else {}
    qk_var_mean = _tensor_mean(tensors.get("qk_var_patch"))
    features: dict[str, Any] = {
        "READ_used_stable_mass": finite_float(debug.get("v78_l07_l13_v68_read_semantic_support_mean")),
        "READ_used_harm_mass": finite_float(debug.get("v78_l07_l13_v68_read_semantic_risk_mean")),
        "QK_pair_compatibility": _clip01(1.0 - qk_var_mean) if qk_var_mean is not None else None,
        "query_risk_mass": _tensor_mean(tensors.get("uncertainty_patch")),
        "read_entropy": _normalized_entropy(tensors.get("read_patch_final")),
    }
    note = (
        "G4 direct READ from read_cue_patch: stable=support_mean, harm=risk_mean, "
        "QK_pair_compatibility=1-mean(qk_var_patch), query_risk_mass=mean(uncertainty_patch), "
        "read_entropy=normalized entropy(read_patch_final)"
    )
    return {key: value for key, value in features.items() if value is not None}, note


def _extract_swa_features(path: Path) -> tuple[dict[str, Any], str]:
    payload = _load_torch_payload(path)
    if not isinstance(payload, dict):
        return {}, "swa_feature_extract_failed: payload_not_dict"
    cur_q = payload.get("tap::pca_swa_current_q_layers")
    cur_k = payload.get("tap::pca_swa_current_k_layers")
    cur_v = payload.get("tap::pca_swa_current_v_layers")
    cache_k = payload.get("tap::pca_swa_cache_k_layers")
    cache_v = payload.get("tap::pca_swa_cache_v_layers")
    qk = _cosine_mean(cur_q, cur_k)
    cache_k_align = _cosine_mean(cur_k, cache_k)
    cache_v_align = _cosine_mean(cur_v, cache_v)
    features: dict[str, Any] = {
        "current_Q_alignment": qk,
        "cache_K_alignment": cache_k_align,
        "cache_V_alignment": cache_v_align,
        "K_risk_delta": _clip01(1.0 - cache_k_align) if cache_k_align is not None else None,
        "V_protect_delta": cache_v_align,
    }
    note = (
        "G5 direct SWA from PCA taps: current_Q_alignment=cos(current_q,current_k), "
        "cache_K_alignment=cos(current_k,cache_k), cache_V_alignment=cos(current_v,cache_v), "
        "K_risk_delta=1-cache_K_alignment, V_protect_delta=cache_V_alignment"
    )
    return {key: value for key, value in features.items() if value is not None}, note


def build_direct_hook_index(path: Path) -> tuple[dict[tuple[str, int], dict[str, Any]], dict[str, Any]]:
    if not path.exists():
        return {}, {"direct_hook_audit_csv": str(path), "exists": False, "feature_chunks": 0, "errors": []}
    rows = read_csv(path)
    out: dict[tuple[str, int], dict[str, Any]] = defaultdict(dict)
    notes: dict[tuple[str, int], list[str]] = defaultdict(list)
    errors: list[str] = []
    for row in rows:
        if row.get("status") != "complete":
            continue
        key = audit_key(row)
        if key is None:
            continue
        artifact_type = row.get("artifact_type")
        group = row.get("artifact_group")
        artifact_path = Path(str(row.get("path", "")))
        try:
            if group == "read" and artifact_type == "read_cue_patch_dump":
                features, note = _extract_read_features(artifact_path)
            elif group == "swa" and artifact_type == "v68_layer_pca_feature_dump":
                features, note = _extract_swa_features(artifact_path)
            else:
                continue
        except Exception as exc:
            errors.append(f"{artifact_path}: {type(exc).__name__}: {exc}")
            continue
        out[key].update(features)
        if features:
            notes[key].append(f"{note}; path={artifact_path}")
    for key, key_notes in notes.items():
        out[key]["direct_hook_feature_notes"] = " | ".join(key_notes)
    summary = {
        "direct_hook_audit_csv": str(path),
        "exists": True,
        "feature_chunks": len(out),
        "feature_cells": sum(1 for row in out.values() for key in row if key != "direct_hook_feature_notes"),
        "errors": errors[:20],
        "error_count": len(errors),
    }
    return dict(out), summary


def build_route_index(rows: list[dict[str, str]]) -> dict[tuple[str, int], dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = route_key(row)
        if key is not None and str(row.get("pair_complete", "")).lower() == "true":
            grouped[key].append(row)

    out: dict[tuple[str, int], dict[str, Any]] = {}
    for key, group_rows in grouped.items():
        route_mass_values = [
            v
            for row in group_rows
            for v in [safe_float(row.get("actual_mean_swa_overlap_attention_mass_selected_lift"))]
            if v is not None
        ]
        delta_values = [
            v
            for row in group_rows
            if "random" in str(row.get("control_kind", "")).lower()
            for v in [safe_float(row.get("actual_minus_control_mean_swa_overlap_attention_mass_selected_lift"))]
            if v is not None
        ]
        all_delta_values = [
            v
            for row in group_rows
            for v in [safe_float(row.get("actual_minus_control_mean_swa_overlap_attention_mass_selected_lift"))]
            if v is not None
        ]
        out[key] = {
            "route_mass": mean(route_mass_values) if route_mass_values else "",
            "actual_vs_random_route_delta": mean(delta_values) if delta_values else "",
            "head_layer_sensitivity": (max(all_delta_values) - min(all_delta_values)) if len(all_delta_values) >= 2 else "",
            "swa_route_rows": len(group_rows),
            "swa_route_groups": sorted({str(row.get("route_group", "")) for row in group_rows if row.get("route_group")}),
            "swa_control_kinds": sorted({str(row.get("control_kind", "")) for row in group_rows if row.get("control_kind")}),
        }
    return out


def attach_route_features(row: dict[str, Any], route_index: Mapping[tuple[str, int], Mapping[str, Any]]) -> None:
    seq = seq_norm(row.get("seq"))
    curr = safe_int(row.get("curr_chunk") or row.get("chunk_id") or row.get("chunk_end"))
    if not seq or curr is None:
        return
    route = route_index.get((seq, curr))
    if not route:
        return
    row["route_mass"] = route.get("route_mass", "")
    row["actual_vs_random_route_delta"] = route.get("actual_vs_random_route_delta", "")
    row["head_layer_sensitivity"] = route.get("head_layer_sensitivity", "")
    note = {
        "G5_route_mass": "mean actual_mean_swa_overlap_attention_mass_selected_lift from v82 contextual route rows",
        "G5_actual_vs_random_route_delta": "mean actual-control selected_lift delta for random controls",
        "G5_head_layer_sensitivity": "max-min actual-control selected_lift delta across joined route rows",
        "swa_route_rows": route.get("swa_route_rows"),
        "swa_route_groups": route.get("swa_route_groups"),
        "swa_control_kinds": route.get("swa_control_kinds"),
    }
    existing = str(row.get("derived_feature_notes", "")).strip()
    row["derived_feature_notes"] = "; ".join(part for part in [existing, json.dumps(note, sort_keys=True)] if part)


def attach_direct_features(row: dict[str, Any], direct_index: Mapping[tuple[str, int], Mapping[str, Any]]) -> None:
    if row.get("row_scope") == "v80_long_window":
        return
    seq = seq_norm(row.get("seq"))
    chunk = safe_int(row.get("curr_chunk") or row.get("chunk_id"))
    if not seq or chunk is None:
        return
    features = direct_index.get((seq, chunk))
    if not features:
        return
    copied: list[str] = []
    for field in GROUP_FIELDS["G4_internal_READ"] + GROUP_FIELDS["G5_internal_SWA"]:
        if is_present(features.get(field)) and not is_present(row.get(field)):
            row[field] = features[field]
            copied.append(field)
    if copied:
        existing = str(row.get("derived_feature_notes", "")).strip()
        note = f"direct_hook_features={','.join(copied)}; {features.get('direct_hook_feature_notes', '')}"
        row["derived_feature_notes"] = "; ".join(part for part in [existing, note] if part)


def build_adjacent_rows(
    pair_rows: list[dict[str, str]],
    route_index: Mapping[tuple[str, int], Mapping[str, Any]],
    direct_index: Mapping[tuple[str, int], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for src in pair_rows:
        row = blank_row("v82_swa_adjacent_pair")
        seq = seq_norm(src.get("seq"))
        prev_chunk = safe_int(src.get("prev_chunk"))
        curr_chunk = safe_int(src.get("curr_chunk"))
        row.update(
            {
                "row_id": f"v82_adj_{seq}_{prev_chunk:03d}_{curr_chunk:03d}",
                "seq": seq,
                "prev_chunk": prev_chunk,
                "curr_chunk": curr_chunk,
                "case_type": src.get("case_type", ""),
                "base_case_type": base_case(src.get("case_type"), src.get("base_case_type")),
                "target_label": target_label(src.get("case_type"), src.get("base_case_type")),
                "label_role": label_role(src.get("case_type"), src.get("base_case_type")),
                "has_radio": str(boolish(src.get("has_radio"))).lower(),
                "source_path": src.get("source_path", ""),
                "source_note": "v82 phase2 SWA pair bank v2",
                "missing_fields_from_source": src.get("carrier_missing_fields", ""),
                "raw_overlap_residual": src.get("raw_overlap_residual", ""),
                "overlap_scale_residual": src.get("overlap_scale_residual", ""),
                "boundary_jump": src.get("boundary_jump", ""),
                "future_after_overlap": src.get("future_after_overlap", ""),
                "confidence_weighted_residual": maybe_product(src.get("raw_overlap_residual"), src.get("semantic_confidence_mean")),
                "prev_to_curr_scale_jump": src.get("prev_to_curr_scale_jump", ""),
                "stable_mass": src.get("stable_overlap_mass", ""),
                "harm_mass": src.get("harm_overlap_mass", ""),
                "context_mass": src.get("context_overlap_mass", ""),
                "semantic_confidence_mean": src.get("semantic_confidence_mean", ""),
                "READ_used_stable_mass": src.get("READ_used_stable_mass", ""),
                "READ_used_harm_mass": src.get("READ_used_harm_mass", ""),
                "K_risk_delta": src.get("K_risk_delta", ""),
                "V_protect_delta": src.get("V_alignment_delta", ""),
                "same_object_ratio": src.get("same_object_overlap_ratio", ""),
                "cross_object_boundary_ratio": src.get("cross_object_boundary_ratio", ""),
                "temporal_stability": src.get("RADIO_temporal_stability", ""),
                "boundary_transform_residual": src.get("boundary_jump", ""),
                "merge_raw_overlap_residual": src.get("raw_overlap_residual", ""),
                "J_mid": src.get("J_mid", ""),
            }
        )
        if is_present(row["confidence_weighted_residual"]):
            row["derived_feature_notes"] = "confidence_weighted_residual=raw_overlap_residual*semantic_confidence_mean"
        attach_route_features(row, route_index)
        attach_direct_features(row, direct_index)
        out.append(row)
    return out


def build_long_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for src in rows:
        row = blank_row("v80_long_window")
        seq = seq_norm(src.get("seq"))
        start = safe_int(src.get("chunk_start"))
        end = safe_int(src.get("chunk_end"))
        row.update(
            {
                "row_id": f"v80_long_{seq}_{start:03d}_{end:03d}",
                "seq": seq,
                "chunk_start": start,
                "chunk_end": end,
                "frame_start": src.get("frame_start", ""),
                "frame_end": src.get("frame_end", ""),
                "case_type": src.get("case_type", ""),
                "base_case_type": base_case(src.get("case_type")),
                "target_label": target_label(src.get("case_type")),
                "label_role": label_role(src.get("case_type")),
                "has_radio": "false",
                "source_path": src.get("trajectory", ""),
                "source_note": "v80 phase1 long_five_chunk_cases",
                "missing_fields_from_source": src.get("missing_fields", ""),
                "window5_joint_sim3_rmse": src.get("window5_joint_sim3_rmse", ""),
                "window5_subchunk_scale_cv": src.get("window5_subchunk_scale_cv", ""),
                "downstream_future_consistency": src.get("downstream_future_consistency", ""),
                "low_observability_score": src.get("low_observability_score", ""),
                "regime_shift_score": src.get("regime_shift_score", ""),
                "road_edge_continuity": src.get("road_edge_continuity", ""),
                "corridor_stability": src.get("corridor_stability", ""),
                "update_conflict": src.get("TTT_update_conflict", ""),
                "post_zp_delta": src.get("post_zp_delta", ""),
                "J_long": src.get("J_long", ""),
            }
        )
        out.append(row)
    return out


def build_mid_rows(
    rows: list[dict[str, str]],
    route_index: Mapping[tuple[str, int], Mapping[str, Any]],
    direct_index: Mapping[tuple[str, int], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for src in rows:
        row = blank_row("v80_mid_adjacent_pair")
        seq = seq_norm(src.get("seq"))
        prev_chunk = safe_int(src.get("prev_chunk"))
        curr_chunk = safe_int(src.get("curr_chunk"))
        row.update(
            {
                "row_id": f"v80_mid_{seq}_{prev_chunk:03d}_{curr_chunk:03d}",
                "seq": seq,
                "prev_chunk": prev_chunk,
                "curr_chunk": curr_chunk,
                "frame_start": src.get("frame_start", ""),
                "frame_end": src.get("frame_end", ""),
                "case_type": src.get("case_type", ""),
                "base_case_type": base_case(src.get("case_type")),
                "target_label": target_label(src.get("case_type")),
                "label_role": label_role(src.get("case_type")),
                "has_radio": "false",
                "source_path": src.get("trajectory", ""),
                "source_note": "v80 phase1 mid_adjacent_pair_cases",
                "missing_fields_from_source": src.get("missing_fields", ""),
                "raw_overlap_residual": src.get("raw_overlap_residual", ""),
                "boundary_jump": src.get("boundary_jump", ""),
                "future_after_overlap": src.get("future_after_overlap", ""),
                "scale_cv": src.get("scale_cv", ""),
                "J_mid": src.get("J_mid", ""),
                "stable_mass": src.get("stable_overlap_mass", ""),
                "harm_mass": src.get("harm_overlap_mass", ""),
                "context_mass": src.get("context_overlap_mass", ""),
                "overlap_semantic_agreement": src.get("overlap_semantic_agreement", ""),
                "same_object_ratio": src.get("same_object_overlap_ratio", ""),
                "cross_object_boundary_ratio": src.get("cross_object_boundary_ratio", ""),
                "K_risk_delta": src.get("K_risk_delta", ""),
                "V_protect_delta": src.get("V_alignment_delta", ""),
                "route_mass": src.get("SWA_gate_mass", ""),
                "merge_raw_overlap_residual": src.get("raw_overlap_residual", ""),
                "boundary_transform_residual": src.get("boundary_jump", ""),
            }
        )
        attach_route_features(row, route_index)
        attach_direct_features(row, direct_index)
        out.append(row)
    return out


def build_short_rows(rows: list[dict[str, str]], direct_index: Mapping[tuple[str, int], Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for src in rows:
        row = blank_row("v80_short_read_case")
        seq = seq_norm(src.get("seq"))
        chunk_id = safe_int(src.get("chunk_id"))
        row.update(
            {
                "row_id": f"v80_short_{seq}_{chunk_id:03d}",
                "seq": seq,
                "chunk_id": chunk_id,
                "frame_start": src.get("frame_start", ""),
                "frame_end": src.get("frame_end", ""),
                "case_type": src.get("case_type", ""),
                "base_case_type": base_case(src.get("case_type")),
                "target_label": target_label(src.get("case_type")),
                "label_role": label_role(src.get("case_type")),
                "has_radio": str(boolish(src.get("radio_available"))).lower(),
                "source_path": src.get("trajectory", ""),
                "source_note": "v80 phase1 short_single_chunk_cases",
                "missing_fields_from_source": src.get("missing_fields", ""),
                "local_sim3_ate": src.get("local_sim3_ate", ""),
                "head_to_tail": src.get("head_to_tail", ""),
                "scale_cv": src.get("scale_cv", ""),
                "intra_scale_variance": src.get("intra_scale_variance", ""),
                "J_short": src.get("J_short", ""),
                "stable_mass": src.get("stable_mass", ""),
                "harm_mass": src.get("harm_mass", ""),
                "context_mass": src.get("context_mass", ""),
                "semantic_confidence_mean": src.get("semantic_confidence_mean", ""),
                "lowtrust_stuff_ratio": src.get("lowtrust_stuff_ratio", ""),
                "thing_moving_ratio": src.get("thing_moving_ratio", ""),
                "thing_static_ratio": src.get("thing_static_ratio", ""),
                "stuff_static_ratio": src.get("stuff_stable_ratio", ""),
                "object_boundary_score": src.get("RADIO_boundary_ratio", ""),
                "temporal_stability": src.get("RADIO_temporal_stability", ""),
                "read_entropy": src.get("READ_attention_entropy", ""),
            }
        )
        if is_present(src.get("stuff_stable_ratio")):
            row["derived_feature_notes"] = "stuff_static_ratio mapped from source stuff_stable_ratio"
        attach_direct_features(row, direct_index)
        out.append(row)
    return out


def present_count(rows: list[Mapping[str, Any]], fields: list[str]) -> tuple[int, int]:
    total = len(rows) * len(fields)
    present = sum(1 for row in rows for field in fields if is_present(row.get(field)))
    return present, total


def applicable_count(rows: list[Mapping[str, Any]], group: str) -> tuple[int, int]:
    present = 0
    total = 0
    for row in rows:
        fields = APPLICABLE_FIELDS[str(row.get("row_scope", ""))].get(group, [])
        present += sum(1 for field in fields if is_present(row.get(field)))
        total += len(fields)
    return present, total


def rate(present: int, total: int) -> float | None:
    return None if total == 0 else present / total


def summarize(
    rows: list[dict[str, Any]],
    source_paths: Mapping[str, Path],
    direct_hook_summary: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    row_counts = Counter(str(row.get("row_scope")) for row in rows)
    seqs = sorted({str(row.get("seq")) for row in rows if is_present(row.get("seq"))})
    case_counts = Counter(str(row.get("case_type")) for row in rows)
    label_counts = Counter(str(row.get("label_role")) for row in rows)

    availability_rows: list[dict[str, Any]] = []
    group_gate: dict[str, Any] = {}
    for group, fields in GROUP_FIELDS.items():
        canonical_present, canonical_total = present_count(rows, fields)
        applicable_present, applicable_total = applicable_count(rows, group)
        gate_mode = (
            "applicable_by_row_scope"
            if group in {"G0_geometry_basic", "G1_dense_semantic", "G4_internal_READ", "G5_internal_SWA"}
            else "canonical_all_rows"
        )
        gate_present, gate_total = (
            (applicable_present, applicable_total)
            if gate_mode == "applicable_by_row_scope"
            else (canonical_present, canonical_total)
        )
        missing_by_field = {
            field: len(rows) - sum(1 for row in rows if is_present(row.get(field))) for field in fields
        }
        row = {
            "group": group,
            "field_count": len(fields),
            "canonical_present_cells": canonical_present,
            "canonical_total_cells": canonical_total,
            "canonical_fill_rate": rate(canonical_present, canonical_total),
            "applicable_present_cells": applicable_present,
            "applicable_total_cells": applicable_total,
            "applicable_fill_rate": rate(applicable_present, applicable_total),
            "gate_mode": gate_mode,
            "gate_present_cells": gate_present,
            "gate_total_cells": gate_total,
            "gate_completeness": rate(gate_present, gate_total),
            "missing_by_field_json": missing_by_field,
        }
        availability_rows.append(row)
        group_gate[group] = row

    phase1_gate = {
        "rows": len(rows),
        "adjacent_pair_rows": row_counts.get("v82_swa_adjacent_pair", 0),
        "long_window_rows": row_counts.get("v80_long_window", 0),
        "sequence_coverage_count": len(seqs),
        "sequence_coverage": seqs,
        "g0_geometry_basic_completeness": group_gate["G0_geometry_basic"]["gate_completeness"],
        "g1_dense_semantic_completeness": group_gate["G1_dense_semantic"]["gate_completeness"],
        "g4_internal_READ_completeness": group_gate["G4_internal_READ"]["gate_completeness"],
        "g5_internal_SWA_completeness": group_gate["G5_internal_SWA"]["gate_completeness"],
    }
    checks = {
        "adjacent_pair_rows_ge_24": phase1_gate["adjacent_pair_rows"] >= 24,
        "long_window_rows_ge_24": phase1_gate["long_window_rows"] >= 24,
        "sequence_coverage_ge_3": phase1_gate["sequence_coverage_count"] >= 3,
        "g0_geometry_basic_completeness_ge_90": (phase1_gate["g0_geometry_basic_completeness"] or 0.0) >= 0.90,
        "g1_dense_semantic_completeness_ge_90": (phase1_gate["g1_dense_semantic_completeness"] or 0.0) >= 0.90,
        "g4_internal_READ_completeness_ge_70": (phase1_gate["g4_internal_READ_completeness"] or 0.0) >= 0.70,
        "g5_internal_SWA_completeness_ge_70": (phase1_gate["g5_internal_SWA_completeness"] or 0.0) >= 0.70,
    }
    phase1_gate["checks"] = checks
    phase1_gate["phase1_gate_pass"] = all(checks.values())
    phase1_gate["blocked_next_action_if_false"] = (
        "Run direct READ/SWA dumps before action; if unavailable, mark G4/G5 incomplete and only run geometry/semantic diagnostics."
    )

    summary = {
        "schema": "acl2_v83_phase1_unified_clue_matrix_summary_v1",
        "source_paths": {name: str(path) for name, path in source_paths.items()},
        "row_counts_by_scope": dict(row_counts),
        "case_counts": dict(case_counts),
        "label_role_counts": dict(label_counts),
        "has_radio_true_rows": sum(1 for row in rows if str(row.get("has_radio")).lower() == "true"),
        "has_radio_false_rows": sum(1 for row in rows if str(row.get("has_radio")).lower() != "true"),
        "direct_hook_feature_summary": dict(direct_hook_summary),
        "phase1_gate": phase1_gate,
        "feature_groups": {
            row["group"]: {
                "fields": GROUP_FIELDS[row["group"]],
                "gate_mode": row["gate_mode"],
                "gate_completeness": row["gate_completeness"],
                "canonical_fill_rate": row["canonical_fill_rate"],
                "applicable_fill_rate": row["applicable_fill_rate"],
                "missing_by_field": row["missing_by_field_json"],
            }
            for row in availability_rows
        },
        "notes": [
            "G0/G1 gate completeness uses row-scope applicable fields because adjacent, long, mid, and short sources expose different geometry/semantic surfaces.",
            "G4/G5 internal READ/SWA gate completeness uses row-scope applicable internal fields; long-window rows are covered by G7 TTT and are not counted as READ/SWA rows.",
            "Direct READ/SWA values are derived only from audited direct-hook tensors/debug stats; feature provenance is written in derived_feature_notes.",
            "RADIO fields are left empty when unavailable and has_radio remains false.",
        ],
    }
    return availability_rows, summary


def write_missing_report(path: Path, summary: Mapping[str, Any]) -> None:
    gate = summary["phase1_gate"]
    lines = [
        "# ACL2 v83 Phase1 Missing Feature Report",
        "",
        "## Gate",
        "",
        f"phase1_gate_pass: `{gate['phase1_gate_pass']}`",
        "",
        "| Check | Value |",
        "| --- | --- |",
    ]
    for key, value in gate["checks"].items():
        lines.append(f"| {key} | {value} |")
    lines.extend(
        [
            "",
            "## Counts",
            "",
            f"- rows: {gate['rows']}",
            f"- adjacent_pair_rows: {gate['adjacent_pair_rows']}",
            f"- long_window_rows: {gate['long_window_rows']}",
            f"- sequence_coverage: {gate['sequence_coverage']}",
            f"- has_radio_true_rows: {summary['has_radio_true_rows']}",
            f"- has_radio_false_rows: {summary['has_radio_false_rows']}",
            "",
            "## Group Completeness",
            "",
            "| Group | Gate Mode | Gate Completeness | Canonical Fill | Applicable Fill |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for group, payload in summary["feature_groups"].items():
        lines.append(
            "| {group} | {mode} | {gate:.6f} | {canon:.6f} | {app:.6f} |".format(
                group=group,
                mode=payload["gate_mode"],
                gate=payload["gate_completeness"] if payload["gate_completeness"] is not None else -1,
                canon=payload["canonical_fill_rate"] if payload["canonical_fill_rate"] is not None else -1,
                app=payload["applicable_fill_rate"] if payload["applicable_fill_rate"] is not None else -1,
            )
        )
    lines.extend(["", "## Missing Fields By Group", ""])
    for group, payload in summary["feature_groups"].items():
        missing = payload["missing_by_field"]
        worst = sorted(missing.items(), key=lambda item: (-item[1], item[0]))
        lines.append(f"### {group}")
        lines.append("")
        for field, count in worst:
            if count:
                lines.append(f"- {field}: missing_rows={count}")
        if not any(count for _, count in worst):
            lines.append("- no missing canonical fields")
        lines.append("")
    lines.extend(["## Phase1 Blocker Handling", ""])
    if gate["phase1_gate_pass"]:
        lines.append(
            "Phase1 gate passes after direct-hook READ/SWA feature extraction. This only permits Phase2 "
            "clue sufficiency diagnostics; it does not permit runtime action."
        )
    else:
        lines.append(
            "G4/G5 or another Phase1 gate item remains below threshold. Per plan section 19.1, "
            "the next legal repair direction is to run or repair direct READ/SWA dump tooling. "
            "If direct dumps are unavailable for the required rows, G4/G5 must remain incomplete "
            "and runtime action must not start from this label-only matrix."
        )
    lines.extend(
        [
            "",
            "RADIO fields are partially unavailable; rows without RADIO evidence keep `has_radio=false`.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    source_paths = {
        "v82_pair_bank": args.v82_pair_bank,
        "v82_route_joined": args.v82_route_joined,
        "v80_long_cases": args.v80_case_bank_dir / "long_five_chunk_cases.csv",
        "v80_mid_cases": args.v80_case_bank_dir / "mid_adjacent_pair_cases.csv",
        "v80_short_cases": args.v80_case_bank_dir / "short_single_chunk_cases.csv",
    }
    for name, path in source_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{name}: {path}")

    route_index = build_route_index(read_csv(args.v82_route_joined))
    direct_index, direct_hook_summary = build_direct_hook_index(args.direct_hook_audit_csv)
    rows: list[dict[str, Any]] = []
    rows.extend(build_adjacent_rows(read_csv(args.v82_pair_bank), route_index, direct_index))
    rows.extend(build_long_rows(read_csv(source_paths["v80_long_cases"])))
    rows.extend(build_mid_rows(read_csv(source_paths["v80_mid_cases"]), route_index, direct_index))
    rows.extend(build_short_rows(read_csv(source_paths["v80_short_cases"]), direct_index))

    availability_rows, summary = summarize(rows, source_paths, direct_hook_summary)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "unified_clue_matrix.csv", rows, OUTPUT_FIELDS)
    write_csv(args.out_dir / "feature_group_availability.csv", availability_rows)
    write_json(args.out_dir / "clue_matrix_summary.json", summary)
    write_missing_report(args.out_dir / "missing_feature_report.md", summary)
    print(json.dumps(summary["phase1_gate"], ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
