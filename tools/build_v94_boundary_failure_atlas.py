#!/usr/bin/env python3
"""Build ACL2 v94 Phase1 boundary failure atlas from measured audit artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.kitti_trajectory_diagnostics import _load_kitti_gt, _load_tum_prediction, _rmse, _umeyama_sim3  # noqa: E402


ROOT = Path("results/acl2_v94tf_semantic_gauge_failure_localization_causal_memory_control")
DEFAULT_GT_ROOT = Path("/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/poses")
V93_POLICY = Path(
    "results/acl2_v93tf_semantic_object_identity_merge_gauge_boundary_carrier/phase2_object_topology_policy/object_topology_policy_rows.csv"
)
V93_TRACE = Path(
    "results/acl2_v93tf_semantic_object_identity_merge_gauge_boundary_carrier/phase3_merge_gauge_trace_audit/merge_gauge_trace_ledger.csv"
)
V92_POLICY = Path(
    "results/acl2_v92tf_semantic_policy_carrier_merge_gauge_boundary_discovery/phase1_semantic_policy_row_bank/semantic_policy_rows.csv"
)
V86_SCALE = Path("results/acl2_v86tf_robust_soft_latent_gauge_transport/phase4_offline_scale_labels/offline_scale_jump_rows.csv")
V88_SCALE_MODE = Path(
    "results/acl2_v88tf_scale_mode_consensus_gauge_update_attribution/phase1_scale_mode_consensus_universe/scale_mode_pair_rows.csv"
)
V90_TOPOLOGY = Path(
    "results/acl2_v90tf_semantic_object_topology_scale_mode_memory_control/phase2_semantic_topology_scale_mode_ledger/topology_pair_rows.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "phase1_boundary_failure_atlas")
    parser.add_argument("--gt-root", type=Path, default=DEFAULT_GT_ROOT)
    parser.add_argument("--trajectory-search-root", type=Path, default=Path("results/kitti01_hmc_v2"))
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
        for row in rows:
            writer.writerow(row)


def seq_text(value: Any) -> str:
    text = str(value).strip()
    if text in {"", "nan", "None"}:
        return ""
    try:
        return f"{int(float(text)):02d}"
    except ValueError:
        return text.zfill(2)


def pair_id(seq: Any, prev: Any, curr: Any) -> str:
    return f"{seq_text(seq)}_{int(float(prev)):03d}_{int(float(curr)):03d}"


def safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        out = float(value)
        if not math.isfinite(out):
            return None
        return out
    except (TypeError, ValueError):
        return None


def normalize_keyed(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["seq"] = frame["seq"].map(seq_text)
    frame["prev_chunk"] = frame["prev_chunk"].astype(int)
    frame["curr_chunk"] = frame["curr_chunk"].astype(int)
    if "pair_id" not in frame.columns:
        frame["pair_id"] = [pair_id(s, p, c) for s, p, c in zip(frame["seq"], frame["prev_chunk"], frame["curr_chunk"])]
    return frame


def q(series: pd.Series, quantile: float) -> float | None:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return None
    return float(numeric.quantile(quantile))


def norm_by_q95(value: Any, q95_value: float | None) -> float | None:
    v = safe_float(value)
    if v is None:
        return None
    denom = q95_value if q95_value and q95_value > 0 else None
    if denom is None:
        return 0.0 if v == 0 else 1.0
    return float(min(max(v / denom, 0.0), 3.0))


def candidate_priority(path: Path) -> tuple[int, int, str]:
    text = path.as_posix()
    score = 0
    if "acl2_v80tf_multiseq_goodbad_semantic_three_memory_control" in text:
        score += 70
    if "acl2_v84tf_memory_ruler_audit" in text:
        score += 65
    if "phase2_direct_hook_repair" in text:
        score += 25
    if "phase1_direct_hook_repair" in text or "phase10_direct_hook_repair" in text:
        score += 20
    if "acl2_v68_integrated_cueconstruction" in text:
        score += 5
    if "code_audit_pack" in text:
        score -= 1000
    return (score, -len(text), text)


def find_trajectory(seq: str, chunk: int, search_root: Path) -> Path | None:
    if chunk < 0 or not search_root.exists():
        return None
    names = {f"chunk{chunk:03d}", f"chunk{chunk}"}
    candidates: list[Path] = []
    for path in search_root.rglob(f"{seq}.txt"):
        if path.parent.name in names:
            candidates.append(path)
    if not candidates:
        return None
    return sorted(candidates, key=candidate_priority, reverse=True)[0]


def eval_trajectory(seq: str, traj: Path, gt_root: Path) -> dict[str, Any]:
    gt_path = gt_root / f"{seq}.txt"
    if not traj or not traj.exists():
        return {"valid": False, "trajectory": str(traj) if traj else "", "missing_reason": "trajectory_missing"}
    if not gt_path.exists():
        return {"valid": False, "trajectory": str(traj), "missing_reason": f"missing_gt:{gt_path}"}
    try:
        _, _, gt_pos = _load_kitti_gt(gt_path)
        frames, raw_poses, raw_pos = _load_tum_prediction(traj, gt_pos.shape[0])
        scale, rot, trans = _umeyama_sim3(raw_pos, gt_pos[frames], with_scale=True)
        aligned_pos = (scale * (rot @ raw_pos.T)).T + trans[None]
        errors = np.linalg.norm(aligned_pos - gt_pos[frames], axis=1)
        rmse = _rmse(errors)
        head_tail = float(
            abs(np.linalg.norm(aligned_pos[-1] - aligned_pos[0]) - np.linalg.norm(gt_pos[frames[-1]] - gt_pos[frames[0]]))
        )
        step_raw = np.linalg.norm(np.diff(raw_pos, axis=0), axis=1)
        step_gt = np.linalg.norm(np.diff(gt_pos[frames], axis=0), axis=1)
        ratios = step_raw / np.maximum(step_gt, 1e-12)
        ratios = ratios[np.isfinite(ratios)]
        scale_cv = float(np.std(ratios) / max(np.mean(ratios), 1e-12)) if ratios.size else None
        return {
            "valid": True,
            "trajectory": str(traj),
            "frame_start": int(frames.min()) if frames.size else "",
            "frame_end": int(frames.max()) if frames.size else "",
            "frame_count": int(frames.size),
            "chunk_sim3_scale": float(scale),
            "local_sim3_ate_rmse": float(rmse),
            "head_tail_error": head_tail,
            "scale_cv": scale_cv,
            "missing_reason": "",
        }
    except Exception as exc:  # noqa: BLE001
        return {"valid": False, "trajectory": str(traj), "missing_reason": f"{type(exc).__name__}:{exc}"}


def semantic_evidence_type(row: pd.Series) -> str:
    s_stable = safe_float(row.get("S_valid")) or safe_float(row.get("v92_S_valid")) or safe_float(row.get("topo_topology_valid_mass")) or 0.0
    s_invalid = safe_float(row.get("S_invalid")) or 0.0
    s_context = max(
        safe_float(row.get("S_context")) or safe_float(row.get("v92_S_context")) or 0.0,
        safe_float(row.get("S_lowobs")) or safe_float(row.get("v92_S_lowobs")) or 0.0,
    )
    s_multi = safe_float(row.get("S_multimode")) or 0.0
    same_obj = safe_float(row.get("same_object_ratio")) or 0.0
    cross_obj = safe_float(row.get("cross_object_ratio")) or 0.0
    if s_multi >= 0.75:
        return "SEM_MULTIMODE_UNSAFE"
    if s_invalid >= 0.35 or cross_obj >= 0.35:
        return "SEM_INVALID_BOUNDARY"
    if s_context >= 0.35:
        return "SEM_LOWOBS_ABSTAIN"
    if s_stable >= 0.25 or same_obj >= 0.75:
        return "SEM_STABLE_REFERENCE"
    if s_context > 0:
        return "SEM_WEAK_CONTEXT"
    return "SEM_UNKNOWN"


def positive_guard(value: Any, threshold: float | None, op: str) -> bool:
    v = safe_float(value)
    if v is None or threshold is None:
        return False
    if threshold <= 0:
        return v > 0
    return v < threshold if op == "lt" else v > threshold


def build_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    policy = normalize_keyed(pd.read_csv(V93_POLICY))
    trace = normalize_keyed(pd.read_csv(V93_TRACE))
    v92 = normalize_keyed(pd.read_csv(V92_POLICY))
    scale = normalize_keyed(pd.read_csv(V86_SCALE))
    mode = normalize_keyed(pd.read_csv(V88_SCALE_MODE))
    topo = normalize_keyed(pd.read_csv(V90_TOPOLOGY))

    merged = policy.merge(trace.add_prefix("trace_"), left_on="pair_id", right_on="trace_pair_id", how="left")
    merged = merged.merge(v92.add_prefix("v92_"), left_on="pair_id", right_on="v92_pair_id", how="left")
    merged = merged.merge(scale.add_prefix("scale_"), left_on="pair_id", right_on="scale_pair_id", how="left")
    merged = merged.merge(mode.add_prefix("mode_"), left_on="pair_id", right_on="mode_pair_id", how="left")
    merged = merged.merge(topo.add_prefix("topo_"), left_on="pair_id", right_on="topo_pair_id", how="left")

    scale_q75 = q(merged["scale_abs_log_scale_jump"], 0.75)
    boundary_q75 = q(merged["trace_boundary_update_norm"], 0.75)
    residual_q95 = q(merged["trace_merge_residual_after"].abs() if "trace_merge_residual_after" in merged else pd.Series(dtype=float), 0.95)
    future_q95 = q(merged["scale_full_ATE_contribution_proxy"], 0.95)
    boundary_q95 = q(merged["trace_boundary_update_norm"], 0.95)
    scale_q95 = q(merged["scale_abs_log_scale_jump"], 0.95)
    observability_q25 = q(merged["v92_observability_score"].combine_first(merged["mode_observability_score"]), 0.25)
    invalid_q75 = q(merged["S_invalid"], 0.75)
    stable_q50 = q(merged["v92_S_valid"].combine_first(merged["topo_topology_valid_mass"]), 0.50)
    mode_entropy_q75 = q(merged["mode_mode_entropy"].combine_first(merged["v92_mode_entropy"]), 0.75)
    mode_mad_q75 = q(merged["mode_weighted_mode_mad"], 0.75)

    trajectory_cache: dict[tuple[str, int], dict[str, Any]] = {}
    recovery_rows: list[dict[str, Any]] = []

    def get_eval(seq: str, chunk: int, known_path: Any) -> dict[str, Any]:
        key = (seq, chunk)
        if key in trajectory_cache:
            return trajectory_cache[key]
        path_text = str(known_path or "").strip()
        traj = Path(path_text) if path_text and path_text.lower() != "nan" else find_trajectory(seq, chunk, args.trajectory_search_root)
        result = eval_trajectory(seq, traj, args.gt_root) if traj else {
            "valid": False,
            "trajectory": "",
            "missing_reason": "trajectory_not_found_by_search",
        }
        result["recovered_by_v94_search"] = bool((not path_text or path_text.lower() == "nan") and result.get("trajectory"))
        trajectory_cache[key] = result
        recovery_rows.append(
            {
                "seq": seq,
                "chunk": chunk,
                "known_path": path_text,
                "selected_trajectory": result.get("trajectory", ""),
                "valid": result.get("valid"),
                "chunk_sim3_scale": result.get("chunk_sim3_scale", ""),
                "local_sim3_ate_rmse": result.get("local_sim3_ate_rmse", ""),
                "recovered_by_v94_search": result.get("recovered_by_v94_search"),
                "missing_reason": result.get("missing_reason", ""),
            }
        )
        return result

    # Pre-evaluate trajectories before local-quality quantiles.
    eval_pairs: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for _, row in merged.iterrows():
        seq = seq_text(row.get("seq"))
        prev = int(row.get("prev_chunk"))
        curr = int(row.get("curr_chunk"))
        prev_eval = get_eval(seq, prev, row.get("scale_prev_trajectory"))
        curr_eval = get_eval(seq, curr, row.get("scale_curr_trajectory"))
        eval_pairs[str(row.get("pair_id"))] = (prev_eval, curr_eval)

    local_values = [
        item.get("local_sim3_ate_rmse")
        for pair in eval_pairs.values()
        for item in pair
        if item.get("valid") and safe_float(item.get("local_sim3_ate_rmse")) is not None
    ]
    local_q75 = float(pd.Series(local_values).quantile(0.75)) if local_values else None

    rows: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        pid = str(row.get("pair_id"))
        seq = seq_text(row.get("seq"))
        prev = int(row.get("prev_chunk"))
        curr = int(row.get("curr_chunk"))
        prev_eval, curr_eval = eval_pairs[pid]
        prev_scale = safe_float(prev_eval.get("chunk_sim3_scale"))
        curr_scale = safe_float(curr_eval.get("chunk_sim3_scale"))
        original_scale_available = str(row.get("scale_scale_label_available")).strip().lower() == "true"
        recovered_scale_available = prev_scale is not None and curr_scale is not None and prev_scale > 0 and curr_scale > 0
        if recovered_scale_available:
            adjacent_log_scale_jump = math.log(curr_scale) - math.log(prev_scale)
            abs_log_scale_jump = abs(adjacent_log_scale_jump)
            scale_jump_source = "v86_original" if original_scale_available else "v94_recovered_trajectory_search"
            scale_missing_reason = ""
        else:
            adjacent_log_scale_jump = None
            abs_log_scale_jump = None
            scale_jump_source = ""
            reasons = [prev_eval.get("missing_reason"), curr_eval.get("missing_reason")]
            scale_missing_reason = ";".join(str(x) for x in reasons if x)

        prev_local = safe_float(prev_eval.get("local_sim3_ate_rmse"))
        curr_local = safe_float(curr_eval.get("local_sim3_ate_rmse"))
        local_bad = (prev_local is not None and local_q75 is not None and prev_local > local_q75) or (
            curr_local is not None and local_q75 is not None and curr_local > local_q75
        )
        local_good = (
            prev_local is not None
            and curr_local is not None
            and local_q75 is not None
            and prev_local <= local_q75
            and curr_local <= local_q75
        )
        handoff_scale = local_good and positive_guard(abs_log_scale_jump, scale_q75, "gt")
        handoff_gauge = local_good and positive_guard(row.get("trace_boundary_update_norm"), boundary_q75, "gt")
        lowobs = positive_guard(row.get("v92_observability_score"), observability_q25, "lt")
        sem_invalid = positive_guard(row.get("S_invalid"), invalid_q75, "gt") and not positive_guard(
            row.get("v92_S_valid"), stable_q50, "gt"
        )
        multimode = positive_guard(row.get("mode_mode_entropy"), mode_entropy_q75, "gt") or positive_guard(
            row.get("mode_weighted_mode_mad"), mode_mad_q75, "gt"
        )

        future_after_overlap = safe_float(row.get("scale_full_ATE_contribution_proxy"))
        boundary_jump = safe_float(row.get("trace_boundary_update_norm"))
        raw_overlap_residual = safe_float(row.get("trace_merge_residual_after"))
        confidence_weighted_overlap_residual = safe_float(row.get("trace_merge_residual_delta"))
        j_parts = [
            0.35 * (norm_by_q95(future_after_overlap, future_q95) or 0.0),
            0.25 * (norm_by_q95(boundary_jump, boundary_q95) or 0.0),
            0.25 * (norm_by_q95(abs_log_scale_jump, scale_q95) or 0.0),
            0.15 * (norm_by_q95(raw_overlap_residual, residual_q95) or 0.0),
        ]
        j_handoff = float(sum(j_parts))
        j_prop_3 = future_after_overlap
        j_prop_5 = future_after_overlap
        growth = None

        tags = []
        if local_bad:
            tags.append("LOCAL_BAD")
        if handoff_scale:
            tags.append("HANDOFF_SCALE")
        if handoff_gauge:
            tags.append("HANDOFF_GAUGE")
        if lowobs:
            tags.append("LOW_OBSERVABILITY")
        if sem_invalid:
            tags.append("SEMANTIC_INVALID")
        if multimode:
            tags.append("MULTIMODE_CONFLICT")
        if not tags and future_after_overlap is not None and positive_guard(future_after_overlap, q(merged["scale_full_ATE_contribution_proxy"], 0.75), "gt"):
            tags.append("LONG_ACCUMULATION")
        if not tags:
            tags.append("SAFE_OR_UNASSIGNED")
        primary = tags[0]
        secondary = ";".join(tags[1:])
        reason = []
        if local_bad:
            reason.append(f"local_sim3_gt_q75={local_q75}")
        if handoff_scale:
            reason.append(f"abs_log_scale_jump>{scale_q75}")
        if handoff_gauge:
            reason.append(f"boundary_update_norm>{boundary_q75}")
        if lowobs:
            reason.append(f"observability_score<{observability_q25}")
        if sem_invalid:
            reason.append(f"S_invalid>{invalid_q75} and S_valid<=q50")
        if multimode:
            reason.append("mode_entropy_or_mad_above_q75")

        item = {
            "seq": seq,
            "prev_chunk": prev,
            "curr_chunk": curr,
            "frame_start": curr_eval.get("frame_start", ""),
            "frame_end": curr_eval.get("frame_end", ""),
            "pair_id": pid,
            "source_pair_bank": str(V93_POLICY),
            "case_label_offline_only": row.get("base_case_type"),
            "prev_local_sim3_ate": prev_local,
            "curr_local_sim3_ate": curr_local,
            "prev_head_tail_error": prev_eval.get("head_tail_error", ""),
            "curr_head_tail_error": curr_eval.get("head_tail_error", ""),
            "prev_scale_cv": prev_eval.get("scale_cv", ""),
            "curr_scale_cv": curr_eval.get("scale_cv", ""),
            "local_chunk_good_flag": bool(local_good),
            "local_label_available": bool(prev_local is not None and curr_local is not None),
            "future_after_overlap": future_after_overlap,
            "boundary_jump": boundary_jump,
            "overlap_scale_residual": abs_log_scale_jump,
            "raw_overlap_residual": raw_overlap_residual,
            "confidence_weighted_overlap_residual": confidence_weighted_overlap_residual,
            "adjacent_log_scale_jump_offline": adjacent_log_scale_jump,
            "adjacent_gauge_jump_proxy": boundary_jump,
            "J_handoff": j_handoff,
            "future_error_1chunk": future_after_overlap,
            "future_error_3chunk": j_prop_3,
            "future_error_5chunk": j_prop_5,
            "J_prop_3": j_prop_3,
            "J_prop_5": j_prop_5,
            "propagation_growth_rate": growth,
            "baseline_proxy": row.get("v92_B_proxy", ""),
            "median_depth_proxy": row.get("v92_d_med_proxy", ""),
            "baseline_over_depth": row.get("v92_B_proxy", ""),
            "verified_match_count": row.get("feature_match_support_count", row.get("v92_feature_match_support_count", "")),
            "raw_overlap_inlier_count": row.get("v92_raw_overlap_support_count", ""),
            "local_shape_mode_entropy": row.get("mode_mode_entropy", row.get("v92_mode_entropy", "")),
            "local_shape_mode_mad": row.get("mode_weighted_mode_mad", ""),
            "observability_score": row.get("v92_observability_score", row.get("mode_observability_score", "")),
            "semantic_valid_mass": row.get("v92_S_valid", row.get("topo_topology_valid_mass", "")),
            "semantic_invalid_mass": row.get("S_invalid", row.get("topo_topology_invalid_mass", "")),
            "semantic_context_mass": row.get("v92_S_context", row.get("topo_topology_context_mass", "")),
            "component_boundary_ratio": row.get("object_boundary_ratio", row.get("topo_topology_boundary_conflict", "")),
            "cross_component_ratio": row.get("cross_object_ratio", row.get("topo_topology_boundary_conflict", "")),
            "same_component_ratio": row.get("same_object_ratio", row.get("topo_topology_component_support", "")),
            "dynamic_or_transient_ratio": row.get("radio_boundary_mean", ""),
            "semantic_mode_entropy": row.get("mode_mode_entropy", row.get("topo_topology_valid_mode_entropy", "")),
            "semantic_evidence_type_majority": semantic_evidence_type(row),
            "failure_type_primary": primary,
            "failure_type_secondary": secondary,
            "failure_confidence": "audit_quantile_rule",
            "assignment_reason": ";".join(reason),
            "scale_label_available": bool(recovered_scale_available),
            "scale_gt_unavailable": not recovered_scale_available,
            "scale_jump_source": scale_jump_source,
            "scale_missing_reason": scale_missing_reason,
            "prev_trajectory": prev_eval.get("trajectory", ""),
            "curr_trajectory": curr_eval.get("trajectory", ""),
            "prev_trajectory_recovered_by_v94_search": prev_eval.get("recovered_by_v94_search", ""),
            "curr_trajectory_recovered_by_v94_search": curr_eval.get("recovered_by_v94_search", ""),
            "trace_path": row.get("trace_merge_state_trace_path", ""),
            "trace_provenance": row.get("trace_trace_provenance", ""),
            "offline_audit_label_only": True,
            "no_gt_runtime_feature": False,
        }
        rows.append(item)

    quantiles = {
        "local_sim3_ate_q75": local_q75,
        "abs_log_scale_jump_q75": scale_q75,
        "boundary_update_norm_q75": boundary_q75,
        "observability_score_q25": observability_q25,
        "S_invalid_q75": invalid_q75,
        "S_valid_q50": stable_q50,
        "mode_entropy_q75": mode_entropy_q75,
        "mode_mad_q75": mode_mad_q75,
        "future_after_overlap_q95": future_q95,
        "boundary_update_norm_q95": boundary_q95,
        "abs_log_scale_jump_q95": scale_q95,
        "merge_residual_after_q95": residual_q95,
    }
    return rows, quantiles, recovery_rows


def summarize(rows: list[dict[str, Any]], quantiles: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    df = pd.DataFrame(rows)
    by_seq = []
    for seq, group in df.groupby("seq", sort=True):
        by_seq.append(
            {
                "seq": seq,
                "row_count": int(len(group)),
                "scale_label_available_rows": int(group["scale_label_available"].sum()),
                "local_label_available_rows": int(group["local_label_available"].sum()),
                "handoff_scale_or_gauge_rows": int(
                    group["failure_type_primary"].isin(["HANDOFF_SCALE", "HANDOFF_GAUGE"]).sum()
                    + group["failure_type_secondary"].astype(str).str.contains("HANDOFF_SCALE|HANDOFF_GAUGE").sum()
                ),
                "good_safe_rows": int(
                    group["case_label_offline_only"].astype(str).eq("good").sum()
                    + group["failure_type_primary"].astype(str).eq("SAFE_OR_UNASSIGNED").sum()
                ),
                "primary_failure_counts": json.dumps(group["failure_type_primary"].value_counts().to_dict(), ensure_ascii=False),
            }
        )
    type_counts = []
    primary_counts = df["failure_type_primary"].value_counts().to_dict()
    for key, value in primary_counts.items():
        type_counts.append({"failure_type_primary": key, "row_count": int(value), "fraction": float(value / len(df))})

    handoff_mask = df["failure_type_primary"].isin(["HANDOFF_SCALE", "HANDOFF_GAUGE"]) | df["failure_type_secondary"].astype(str).str.contains(
        "HANDOFF_SCALE|HANDOFF_GAUGE", regex=True
    )
    labelled_or_scale = df["scale_label_available"].astype(bool) | df["case_label_offline_only"].astype(str).isin(["bad", "good"])
    local_classified = df["local_label_available"].astype(bool) & df["failure_type_primary"].notna()
    safe_rows = df["case_label_offline_only"].astype(str).eq("good") | df["failure_type_primary"].astype(str).eq("SAFE_OR_UNASSIGNED")
    max_primary_frac = max(primary_counts.values()) / len(df) if primary_counts else 1.0
    checks = {
        "boundary_rows_ge_49": len(df) >= 49,
        "sequence_coverage_ge_4": df["seq"].nunique() >= 4,
        "scale_or_audit_labels_ge_70pct": float(labelled_or_scale.mean()) >= 0.70,
        "true_scale_label_rows_ge_70pct": float(df["scale_label_available"].mean()) >= 0.70,
        "local_vs_handoff_classification_ge_80pct": float(local_classified.mean()) >= 0.80,
        "handoff_scale_or_gauge_rows_ge_10": int(handoff_mask.sum()) >= 10,
        "good_safe_rows_ge_10": int(safe_rows.sum()) >= 10,
        "no_single_failure_type_gt_80pct": max_primary_frac <= 0.80,
    }
    gate = all(checks.values())
    summary = {
        "phase": "Phase1_boundary_failure_atlas",
        "phase1_gate_pass": gate,
        "blocker": "" if gate else "phase1_boundary_failure_atlas_gate_failed",
        "checks": checks,
        "row_count": int(len(df)),
        "sequence_coverage": int(df["seq"].nunique()),
        "scale_label_available_rows": int(df["scale_label_available"].sum()),
        "scale_label_available_ratio": float(df["scale_label_available"].mean()),
        "scale_or_audit_label_available_ratio": float(labelled_or_scale.mean()),
        "local_vs_handoff_classification_ratio": float(local_classified.mean()),
        "handoff_scale_or_gauge_rows": int(handoff_mask.sum()),
        "good_safe_rows": int(safe_rows.sum()),
        "primary_failure_type_counts": primary_counts,
        "max_primary_failure_type_fraction": float(max_primary_frac),
        "quantiles": quantiles,
        "scale_recovery_policy": "audit-only recompute from trajectory+GT when historical chunk trajectory exists; proxy-only rows are not counted in true_scale_label_rows_ge_70pct",
        "runtime_action_allowed": False,
        "counterfactual_allowed": False,
        "ttt_allowed": False,
    }
    return by_seq, type_counts, summary


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows, quantiles, recovery_rows = build_rows(args)
    by_seq, type_counts, summary = summarize(rows, quantiles)
    write_csv(args.out_dir / "boundary_failure_rows.csv", rows)
    write_csv(args.out_dir / "boundary_failure_by_seq.csv", by_seq)
    write_csv(args.out_dir / "boundary_failure_type_counts.csv", type_counts)
    write_csv(args.out_dir / "scale_recovery_rows.csv", recovery_rows)
    write_json(args.out_dir / "local_vs_handoff_diagnostic.json", summary)
    write_json(args.out_dir / "phase1_gate_summary.json", summary)
    write_json(args.out_dir / "phase1_quantile_tables.json", quantiles)
    print(f"phase1_gate_pass={summary['phase1_gate_pass']}")
    print(f"row_count={summary['row_count']}")
    print(f"sequence_coverage={summary['sequence_coverage']}")
    print(f"scale_label_available_rows={summary['scale_label_available_rows']}")
    print(f"scale_label_available_ratio={summary['scale_label_available_ratio']}")
    print(f"local_vs_handoff_classification_ratio={summary['local_vs_handoff_classification_ratio']}")
    print(f"handoff_scale_or_gauge_rows={summary['handoff_scale_or_gauge_rows']}")
    print(f"good_safe_rows={summary['good_safe_rows']}")
    if summary["blocker"]:
        print(f"blocker={summary['blocker']}")


if __name__ == "__main__":
    main()
