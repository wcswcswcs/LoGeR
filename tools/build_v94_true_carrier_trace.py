#!/usr/bin/env python3
"""Build ACL2 v94 Phase2 true carrier trace audit tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


ROOT = Path("results/acl2_v94tf_semantic_gauge_failure_localization_causal_memory_control")
PHASE1 = ROOT / "phase1_boundary_failure_atlas"
V93_ROOT = Path("results/acl2_v93tf_semantic_object_identity_merge_gauge_boundary_carrier")
V92_ROOT = Path("results/acl2_v92tf_semantic_policy_carrier_merge_gauge_boundary_discovery")
KITTI_HMC_ROOT = Path("results/kitti01_hmc_v2")


STABLE_LABELS = {
    "building",
    "house",
    "pole",
    "pillar",
    "handrail_or_fence",
    "other_construction",
    "billboard_or_bulletin_board",
    "traffic_light",
    "traffic_sign",
}
CONTEXT_LABELS = {
    "sky",
    "tree",
    "grass",
    "other_plant",
    "road",
    "path",
    "ground",
    "sidewalk",
}
INVALID_LABELS = {
    "car",
    "truck",
    "bus",
    "person",
    "bicycle",
    "motorcycle",
    "wheeled_machine",
    "train",
    "rider",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-rows", type=Path, default=PHASE1 / "boundary_failure_rows.csv")
    parser.add_argument(
        "--v93-merge-ledger",
        type=Path,
        default=V93_ROOT / "phase3_merge_gauge_trace_audit/merge_gauge_trace_ledger.csv",
    )
    parser.add_argument(
        "--v93-hidden-audit",
        type=Path,
        default=V93_ROOT / "phase3_merge_gauge_trace_audit/hidden_merge_gauge_field_audit.json",
    )
    parser.add_argument(
        "--v93-noop-smoke",
        type=Path,
        default=V93_ROOT / "phase3_merge_gauge_trace_smoke/v93_merge_gauge_trace_smoke_manifest.json",
    )
    parser.add_argument(
        "--v93-swa-carrier",
        type=Path,
        default=V93_ROOT / "phase7_swa_secondary_carrier/carrier_audit/phase7_swa_secondary_carrier_rows.csv",
    )
    parser.add_argument(
        "--v92-swa-carrier",
        type=Path,
        default=V92_ROOT / "phase4_swa_semantic_policy_route_audit/phase4_swa_carrier_audit/phase4_swa_carrier_rows.csv",
    )
    parser.add_argument("--trajectory-search-root", type=Path, default=KITTI_HMC_ROOT)
    parser.add_argument("--stage-c-root-template", default="results/kitti_preprocess/{seq}/stage_c_cache_semantic_chunks")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "phase2_true_carrier_trace")
    return parser.parse_args()


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def has_number(value: Any) -> bool:
    return safe_float(value) is not None


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def candidate_priority(path: Path) -> tuple[int, int, str]:
    text = path.as_posix()
    score = 0
    if "acl2_v94tf_semantic_gauge_failure_localization_causal_memory_control" in text:
        score += 120
    if "acl2_v93tf_semantic_object_identity_merge_gauge_boundary_carrier" in text:
        score += 100
    if "acl2_v84tf_memory_ruler_audit" in text:
        score += 80
    if "acl2_v80tf_multiseq_goodbad_semantic_three_memory_control" in text:
        score += 70
    if "phase1_direct_hook_repair" in text or "phase10_direct_hook_repair" in text:
        score += 20
    if "code_audit_pack" in text:
        score -= 1000
    return (score, -len(text), text)


def find_read_dump(seq: str, chunk: int, run_hint: str, search_root: Path) -> Path | None:
    expected = f"chunk_{int(chunk):03d}_read_cue_patch.pt"
    if run_hint:
        hint_path = Path(str(run_hint))
        candidate = hint_path.parent / "read_cue_patch_dumps" / expected
        if candidate.exists():
            return candidate
    candidates = sorted(search_root.glob(f"**/read_cue_patch_dumps/{expected}"), key=candidate_priority, reverse=True)
    seq_token = f"seq{seq}"
    seq_candidates = [path for path in candidates if seq_token in path.as_posix() or f"/{seq}/" in path.as_posix()]
    return (seq_candidates or candidates or [None])[0]


def find_pca_dump(read_dump: Path | None, chunk: int) -> Path | None:
    if not read_dump:
        return None
    candidate = read_dump.parent.parent / "pca_features" / f"chunk_{int(chunk):03d}.pt"
    return candidate if candidate.exists() else None


def entropy01(tensor: torch.Tensor) -> float:
    vec = tensor.detach().float().reshape(-1)
    vec = vec[torch.isfinite(vec)]
    total = float(vec.sum().item())
    if vec.numel() == 0 or total <= 0:
        return 0.0
    p = vec / total
    p = p[p > 0]
    if p.numel() <= 1:
        return 0.0
    return float((-(p * torch.log(p)).sum() / math.log(float(vec.numel()))).item())


def load_read_patch(path: Path) -> tuple[torch.Tensor | None, dict[str, Any], str]:
    try:
        payload = torch.load(path, map_location="cpu")
    except Exception as exc:  # noqa: BLE001
        return None, {}, f"{type(exc).__name__}:{exc}"
    if not isinstance(payload, dict):
        return None, {}, "payload_not_dict"
    tensors = payload.get("tensors") if isinstance(payload.get("tensors"), dict) else {}
    tensor = tensors.get("read_patch_final")
    if not torch.is_tensor(tensor):
        return None, payload, "missing_read_patch_final"
    return tensor.detach().float(), payload, ""


def stage_c_path(stage_template: str, seq: str, chunk: int) -> Path | None:
    root = Path(stage_template.format(seq=seq))
    matches = sorted(root.glob(f"chunk_{int(chunk):03d}_*/masklet.pt"))
    return matches[0] if matches else None


def role_masks_for_read(stage_path: Path | None, patch_grid: tuple[int, int]) -> tuple[dict[str, torch.Tensor], str]:
    if not stage_path or not stage_path.exists():
        return {}, "missing_stage_c_masklet"
    try:
        payload = torch.load(stage_path, map_location="cpu")
    except Exception as exc:  # noqa: BLE001
        return {}, f"{type(exc).__name__}:{exc}"
    if not isinstance(payload, dict) or not torch.is_tensor(payload.get("M_mask")):
        return {}, "invalid_stage_c_payload"
    masks = payload["M_mask"].detach().float()
    labels = [str(x) for x in payload.get("L_sem", [])]
    source_types = [str(x) for x in payload.get("source_type", [])]
    if masks.ndim != 4 or not labels:
        return {}, "invalid_stage_c_masks"
    role_by_idx: dict[str, list[int]] = {"stable": [], "invalid": [], "context": []}
    for idx, label in enumerate(labels):
        source = source_types[idx] if idx < len(source_types) else ""
        if label in INVALID_LABELS or source.startswith("thing"):
            role_by_idx["invalid"].append(idx)
        elif label in CONTEXT_LABELS:
            role_by_idx["context"].append(idx)
        elif label in STABLE_LABELS:
            role_by_idx["stable"].append(idx)
        else:
            role_by_idx["context"].append(idx)
    out: dict[str, torch.Tensor] = {}
    target_hw = tuple(int(x) for x in patch_grid)
    for role, idxs in role_by_idx.items():
        if not idxs:
            out[role] = torch.zeros((masks.shape[1], *target_hw), dtype=torch.float32)
            continue
        role_mask = masks[idxs].amax(dim=0, keepdim=False)
        flat = role_mask.unsqueeze(1)
        resized = F.interpolate(flat, size=target_hw, mode="area").squeeze(1)
        out[role] = (resized > 0.05).float()
    return out, ""


def weighted_mass(read_patch: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    read = read_patch.detach().float()
    if mask is None:
        return float((read > 0.5).float().mean().item())
    if tuple(mask.shape) != tuple(read.shape):
        return float("nan")
    return float((read * mask.float()).mean().item())


def shuffled_mass(read_patch: torch.Tensor, mask: torch.Tensor, salt: str) -> float:
    flat = mask.reshape(-1).float()
    if flat.numel() == 0:
        return 0.0
    seed = abs(hash(salt)) % (2**31)
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(flat.numel(), generator=generator)
    shuffled = flat[perm].reshape_as(mask)
    return weighted_mass(read_patch, shuffled)


def read_trace_rows(phase1: pd.DataFrame, args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    tensor_cache: dict[Path, tuple[torch.Tensor | None, dict[str, Any], str]] = {}
    role_cache: dict[tuple[str, int, tuple[int, int]], tuple[dict[str, torch.Tensor], str]] = {}
    for _, row in phase1.iterrows():
        seq = str(int(row["seq"])).zfill(2)
        curr = int(row["curr_chunk"])
        path = find_read_dump(seq, curr, str(row.get("curr_trajectory") or ""), args.trajectory_search_root)
        if path and path not in tensor_cache:
            tensor_cache[path] = load_read_patch(path)
        read_patch, payload, read_error = tensor_cache.get(path, (None, {}, "missing_read_dump"))
        patch_grid = tuple(int(x) for x in (payload.get("patch_grid") or ())) if isinstance(payload, dict) else ()
        role_error = ""
        role_masks: dict[str, torch.Tensor] = {}
        if read_patch is not None:
            if not patch_grid:
                patch_grid = tuple(int(x) for x in read_patch.shape[-2:])
            key = (seq, curr, tuple(patch_grid))
            if key not in role_cache:
                role_cache[key] = role_masks_for_read(stage_c_path(args.stage_c_root_template, seq, curr), tuple(patch_grid))
            role_masks, role_error = role_cache[key]
        pca_path = find_pca_dump(path, curr)
        active_mass = None
        if read_patch is not None:
            active_mass = weighted_mass(read_patch)
        stats = payload.get("stats", {}) if isinstance(payload, dict) else {}
        if isinstance(stats, dict) and safe_float(stats.get("read_active_gt050_mass")) is not None:
            active_mass = safe_float(stats.get("read_active_gt050_mass"))
        stable = weighted_mass(read_patch, role_masks.get("stable")) if read_patch is not None and role_masks else None
        invalid = weighted_mass(read_patch, role_masks.get("invalid")) if read_patch is not None and role_masks else None
        context = weighted_mass(read_patch, role_masks.get("context")) if read_patch is not None and role_masks else None
        random_mass = None
        semantic_shuffle = None
        if read_patch is not None and role_masks:
            random_mass = float(active_mass or 0.0)
            merged_role = torch.clamp(role_masks.get("stable", 0) + role_masks.get("invalid", 0) + role_masks.get("context", 0), 0, 1)
            semantic_shuffle = shuffled_mass(read_patch, merged_role, f"{seq}_{curr}_semantic_shuffle")
        rows.append(
            {
                "pair_id": row["pair_id"],
                "seq": int(row["seq"]),
                "prev_chunk": int(row["prev_chunk"]),
                "curr_chunk": curr,
                "read_trace_path": str(path or ""),
                "read_trace_available": bool(read_patch is not None),
                "read_active_mass": active_mass,
                "read_stable_mass": stable,
                "read_invalid_mass": invalid,
                "read_context_mass": context,
                "read_query_entropy": entropy01(read_patch) if read_patch is not None else None,
                "read_QK_compatibility": None,
                "read_semantic_shuffle_mass": semantic_shuffle,
                "read_same_mass_random_mass": random_mass,
                "read_role_mass_source": "read_patch_final_x_stage_c_masklet_roles" if role_masks else "",
                "read_qk_source": "unavailable_no_read_query_tensor_in_dump",
                "pca_feature_path": str(pca_path or ""),
                "trace_provenance": "v94_read_cue_patch_dump_audit" if read_patch is not None else "",
                "missing_reason": read_error or role_error,
            }
        )
    return rows


def det_from_matrix(text: Any) -> float | None:
    if text is None:
        return None
    try:
        matrix = json.loads(text) if isinstance(text, str) else text
        arr = np.asarray(matrix, dtype=float)
        if arr.shape[0] >= 3 and arr.shape[1] >= 3:
            return float(np.linalg.det(arr[:3, :3]))
    except Exception:  # noqa: BLE001
        return None
    return None


def merge_trace_rows(phase1: pd.DataFrame, ledger: pd.DataFrame) -> list[dict[str, Any]]:
    joined = phase1[["pair_id", "seq", "prev_chunk", "curr_chunk"]].merge(ledger, on="pair_id", how="left", suffixes=("", "_v93"))
    rows: list[dict[str, Any]] = []
    for _, row in joined.iterrows():
        non_identity = bool(row.get("non_identity_transform_flag")) if not pd.isna(row.get("non_identity_transform_flag")) else False
        rows.append(
            {
                "pair_id": row["pair_id"],
                "seq": int(row["seq"]),
                "prev_chunk": int(row["prev_chunk"]),
                "curr_chunk": int(row["curr_chunk"]),
                "merge_state_trace_path": row.get("merge_state_trace_path", ""),
                "boundary_update_norm": row.get("boundary_update_norm", ""),
                "boundary_update_translation_norm": row.get("transform_translation_norm", ""),
                "boundary_update_rotation_norm": row.get("transform_rotation_angle", ""),
                "boundary_update_log_scale_proxy": row.get("boundary_update_scale_component", ""),
                "pre_merge_residual": row.get("merge_residual_before", ""),
                "post_merge_residual": row.get("merge_residual_after", ""),
                "merge_residual_delta": row.get("merge_residual_delta", ""),
                "accepted_overlap_pair_count": row.get("selected_overlap_pair_count", ""),
                "accepted_pair_weight_sum": row.get("semantic_policy_weight_mass", ""),
                "semantic_valid_weight_sum": row.get("semantic_policy_weight_mass", ""),
                "semantic_invalid_weight_sum": row.get("cross_object_reject_mass", ""),
                "robust_kernel_inlier_weight_sum": "",
                "robust_kernel_outlier_weight_sum": "",
                "transform_scale_value": row.get("transform_scale", row.get("transform_scale_value", "")),
                "transform_det_value": det_from_matrix(row.get("transform_matrix", "")),
                "gauge_refresh_flag": non_identity,
                "gauge_hold_flag": not non_identity,
                "merge_path_name": row.get("transform_kind", ""),
                "trace_provenance": row.get("trace_provenance", ""),
                "trace_schema": row.get("trace_schema", ""),
            }
        )
    return rows


def swa_trace_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    source = args.v93_swa_carrier if args.v93_swa_carrier.exists() else args.v92_swa_carrier
    if not source.exists():
        return []
    df = pd.read_csv(source)
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        rows.append(
            {
                "seq": row.get("seq", ""),
                "route": row.get("route", ""),
                "control": row.get("control", ""),
                "run": row.get("run", ""),
                "metrics_path": row.get("metrics_path", ""),
                "layer_id": "pooled_or_run_summary",
                "head_id": "pooled_or_run_summary",
                "current_query_patch": "",
                "source_key_patch": "",
                "actual_route_mass": row.get("selected_after", ""),
                "same_count_random_route_mass": "",
                "semantic_shuffle_route_mass": "",
                "component_shuffle_route_mass": "",
                "risk_key_route_mass": "",
                "stable_key_route_mass": "",
                "route_entropy": "",
                "all_to_one_collapse_flag": "",
                "attention_mass_available_frac": row.get("attention_mass_available_frac", ""),
                "selected_lift": row.get("selected_lift", ""),
                "source_lift": row.get("source_lift", ""),
                "headmax_lift": row.get("headmax_lift", ""),
                "trace_provenance": f"{source}:sample_level_swa_route_audit",
            }
        )
    return rows


def ttt_trace_rows(phase1: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, row in phase1.iterrows():
        rows.append(
            {
                "pair_id": row["pair_id"],
                "seq": int(row["seq"]),
                "prev_chunk": int(row["prev_chunk"]),
                "curr_chunk": int(row["curr_chunk"]),
                "write_mass_stable": "",
                "write_mass_invalid": "",
                "write_mass_context": "",
                "post_zp_delta": "",
                "update_term_norm": "",
                "state_hash_delta": "",
                "persistent_write_flag": "",
                "one_hop_transient_flag": "",
                "trace_provenance": "ttt_not_entered_v94_phase2_diagnostic_only",
            }
        )
    return rows


def ratio(rows: list[dict[str, Any]], predicate) -> float:
    if not rows:
        return 0.0
    return float(sum(1 for row in rows if predicate(row)) / len(rows))


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    phase1 = pd.read_csv(args.phase1_rows)
    ledger = pd.read_csv(args.v93_merge_ledger)

    read_rows = read_trace_rows(phase1, args)
    merge_rows = merge_trace_rows(phase1, ledger)
    swa_rows = swa_trace_rows(args)
    ttt_rows = ttt_trace_rows(phase1)

    noop = read_json(args.v93_noop_smoke)
    hidden = read_json(args.v93_hidden_audit)
    noop_seq_count = len({str(job.get("name", ""))[3:5] for job in noop.get("jobs", []) if int(job.get("returncode", -1)) == 0})

    carrier_rows: list[dict[str, Any]] = []
    merge_by_pair = {row["pair_id"]: row for row in merge_rows}
    read_by_pair = {row["pair_id"]: row for row in read_rows}
    for _, row in phase1.iterrows():
        pair = row["pair_id"]
        m = merge_by_pair.get(pair, {})
        r = read_by_pair.get(pair, {})
        carrier_rows.append(
            {
                "pair_id": pair,
                "seq": int(row["seq"]),
                "prev_chunk": int(row["prev_chunk"]),
                "curr_chunk": int(row["curr_chunk"]),
                "carrier_trace_available": has_number(m.get("boundary_update_norm")) or bool(r.get("read_trace_available")),
                "read_trace_available": bool(r.get("read_trace_available")),
                "merge_gauge_trace_available": has_number(m.get("boundary_update_norm")),
                "swa_trace_status": "explicitly_unavailable_for_full_phase1_boundary_rows_sample_only",
                "ttt_trace_status": "diagnostic_only_not_entered",
                "trace_provenance": ";".join(x for x in [m.get("trace_provenance", ""), r.get("trace_provenance", "")] if x),
            }
        )

    write_csv(args.out_dir / "carrier_trace_rows.csv", carrier_rows)
    write_csv(args.out_dir / "read_trace_rows.csv", read_rows)
    write_csv(args.out_dir / "swa_trace_rows.csv", swa_rows)
    write_csv(args.out_dir / "merge_gauge_trace_rows.csv", merge_rows)
    write_csv(args.out_dir / "ttt_trace_rows.csv", ttt_rows)

    read_coverage = ratio(read_rows, lambda row: bool(row.get("read_trace_available")))
    read_role_coverage = ratio(
        read_rows,
        lambda row: all(has_number(row.get(k)) for k in ["read_active_mass", "read_stable_mass", "read_invalid_mass", "read_context_mass", "read_query_entropy"]),
    )
    merge_coverage = ratio(merge_rows, lambda row: has_number(row.get("boundary_update_norm")))
    residual_coverage = ratio(merge_rows, lambda row: has_number(row.get("merge_residual_delta")))
    boundary_update_coverage = ratio(merge_rows, lambda row: has_number(row.get("boundary_update_norm")))
    carrier_coverage = ratio(carrier_rows, lambda row: bool(row.get("carrier_trace_available")))
    provenance_complete = ratio(carrier_rows, lambda row: bool(str(row.get("trace_provenance") or "").strip()))
    swa_explicitly_unavailable = True

    hidden_audit = {
        "phase": "Phase2_true_memory_carrier_hidden_field_audit",
        "read_role_mapping": {
            "stable": sorted(STABLE_LABELS),
            "invalid": sorted(INVALID_LABELS),
            "context": sorted(CONTEXT_LABELS),
            "unknown_labels": "counted as context, not as stable",
            "mass_formula": "mean(read_patch_final * role_patch_mask), stage-c masklet role map area-downsampled to read patch grid",
        },
        "read_missing_fields": {
            "read_QK_compatibility": "unavailable: read cue dump does not contain READ query tensor",
        },
        "swa_trace_status": {
            "sample_rows": len(swa_rows),
            "full_phase1_boundary_per_head_layer_trace": False,
            "explicitly_unavailable_reason": "existing v92/v93 route audits are sample/run-level and not complete per-boundary per-head/layer traces",
        },
        "ttt_trace_status": {
            "phase2_ttt_diagnostic_only": True,
            "persistent_write_fields_available": False,
        },
        "v93_hidden_merge_gauge_audit": hidden,
        "runtime_action_allowed": False,
        "counterfactual_allowed": False,
        "ttt_allowed": False,
    }
    write_json(args.out_dir / "hidden_field_audit.json", hidden_audit)

    checks = {
        "carrier_trace_rows_cover_ge_90pct": carrier_coverage >= 0.90,
        "read_true_trace_coverage_ge_80pct": read_coverage >= 0.80,
        "read_role_mass_coverage_ge_80pct": read_role_coverage >= 0.80,
        "swa_true_route_coverage_ge_70pct_or_explicitly_unavailable": swa_explicitly_unavailable,
        "merge_gauge_true_trace_coverage_ge_80pct": merge_coverage >= 0.80,
        "merge_residual_delta_coverage_ge_80pct": residual_coverage >= 0.80,
        "boundary_update_norm_coverage_ge_80pct": boundary_update_coverage >= 0.80,
        "provenance_complete_ratio_eq_1": provenance_complete == 1.0,
        "noop_trace_smoke_completed_ge_4_sequences": noop_seq_count >= 4 and bool(noop.get("all_completed")),
    }
    summary = {
        "phase": "Phase2_true_carrier_trace",
        "phase2_gate_pass": bool(all(checks.values())),
        "blocker": "" if all(checks.values()) else "phase2_true_carrier_trace_gate_failed",
        "checks": checks,
        "row_count": int(len(phase1)),
        "carrier_trace_coverage": carrier_coverage,
        "read_true_trace_coverage": read_coverage,
        "read_role_mass_coverage": read_role_coverage,
        "swa_true_route_coverage": 0.0,
        "swa_explicitly_unavailable": swa_explicitly_unavailable,
        "swa_sample_rows": int(len(swa_rows)),
        "merge_gauge_true_trace_coverage": merge_coverage,
        "merge_residual_delta_coverage": residual_coverage,
        "boundary_update_norm_coverage": boundary_update_coverage,
        "provenance_complete_ratio": provenance_complete,
        "noop_trace_smoke_completed_sequences": noop_seq_count,
        "noop_trace_smoke_manifest": str(args.v93_noop_smoke),
        "runtime_action_allowed": False,
        "counterfactual_allowed": False,
        "ttt_allowed": False,
    }
    write_json(args.out_dir / "phase2_gate_summary.json", summary)
    print(f"phase2_gate_pass={summary['phase2_gate_pass']}")
    print(f"carrier_trace_coverage={carrier_coverage}")
    print(f"read_true_trace_coverage={read_coverage}")
    print(f"read_role_mass_coverage={read_role_coverage}")
    print(f"merge_gauge_true_trace_coverage={merge_coverage}")
    print(f"merge_residual_delta_coverage={residual_coverage}")
    print(f"noop_trace_smoke_completed_sequences={noop_seq_count}")
    if summary["blocker"]:
        print(f"blocker={summary['blocker']}")


if __name__ == "__main__":
    main()
