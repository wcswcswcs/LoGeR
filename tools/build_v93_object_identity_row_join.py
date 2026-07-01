#!/usr/bin/env python3
"""Build v93 Phase1 row-level object/RADIO/tracklet source join table."""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

from tools.v86_soft_latent_utils import safe_float, write_csv, write_json  # noqa: E402
from tools.v93_semantic_object_identity_utils import ROOT, V91_ROOT, V92_ROOT, pair_id, seq_text  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v92-root", type=Path, default=V92_ROOT)
    parser.add_argument("--v91-root", type=Path, default=V91_ROOT)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "phase1_object_identity_row_join")
    return parser.parse_args()


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _parse_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(text)
        except Exception:
            return []
    if isinstance(parsed, list):
        return [str(v) for v in parsed]
    return []


def _bool(value: Any) -> bool:
    return str(value or "").strip().lower() in {"true", "1", "yes", "y"}


def _mean(values: list[float]) -> float | None:
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return None
    return float(sum(clean) / len(clean))


def _seq_track_metadata(seq: str) -> dict[str, Any]:
    path = Path(f"results/kitti_preprocess/{seq}/sam31_textmatch_caronly_signmerge/debug/track_metadata.json")
    data = _load_json(path)
    return data if isinstance(data, dict) else {}


def _stage_c_frame_range(paths: list[str]) -> tuple[int | None, int | None]:
    starts: list[int] = []
    ends: list[int] = []
    for text in paths:
        data = _load_json(Path(text))
        if not isinstance(data, dict):
            continue
        if data.get("start_frame") is not None:
            starts.append(int(data["start_frame"]))
        if data.get("end_frame") is not None:
            ends.append(int(data["end_frame"]))
    if not starts or not ends:
        return None, None
    return min(starts), max(ends)


def _chunk_id_from_sidecar_path(path: Path) -> int | None:
    match = re.search(r"chunk_(\d+)_", path.as_posix())
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _discover_radio_sidecars(seq: str, prev_chunk: Any, curr_chunk: Any) -> list[str]:
    """Find loadable RADIO/RADSeg chunk sidecars for this policy row.

    v92 candidate rows only contained paths for the sidecars available at the
    time they were built. v93 may generate additional audited sidecars under
    results/kitti_preprocess/<seq>/radio_sidecar_chunks* during Case-A repair,
    so Phase1 must discover those files explicitly instead of relying on stale
    v92 manifests.
    """

    seq_root = Path(f"results/kitti_preprocess/{seq_text(seq)}")
    if not seq_root.exists():
        return []
    chunks: list[int] = []
    for value in [prev_chunk, curr_chunk]:
        try:
            chunks.append(int(float(str(value).strip())))
        except (TypeError, ValueError):
            continue
    out: list[str] = []
    for root_pattern in ["radio_sidecar_chunks*", "radseg_sidecar_chunks*"]:
        for root in sorted(seq_root.glob(root_pattern)):
            if not root.is_dir():
                continue
            for chunk in chunks:
                for path in sorted(root.glob(f"chunk_{chunk:03d}_*/radio_sidecar.pt")):
                    if path.exists():
                        out.append(str(path))
    return sorted(set(out))


def _spanning_track_stats(seq: str, stage_paths: list[str]) -> dict[str, Any]:
    start, end = _stage_c_frame_range(stage_paths)
    metadata = _seq_track_metadata(seq)
    if start is None or end is None or not metadata:
        return {
            "global_object_candidate_count": 0,
            "global_object_candidate_confirmed_count": 0,
            "global_object_candidate_ids": [],
            "object_id_source": "sam31_track_metadata_unmapped",
            "object_id_confidence": 0.0,
        }
    candidates = []
    confirmed = []
    for item in metadata.values():
        if not isinstance(item, dict):
            continue
        birth = int(item.get("birth_frame", 10**9))
        last = int(item.get("last_frame", -1))
        if birth <= start and last >= end - 1:
            gid = item.get("global_id")
            candidates.append(gid)
            if str(item.get("state", "")) == "confirmed":
                confirmed.append(gid)
    return {
        "global_object_candidate_count": len(candidates),
        "global_object_candidate_confirmed_count": len(confirmed),
        "global_object_candidate_ids": confirmed[:8],
        "object_id_source": "sam31_track_metadata_sequence_candidate_unmapped_to_policy_component",
        "object_id_confidence": 0.0,
    }


def _clean_global_id(value: Any) -> str:
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return ""
    try:
        return str(int(float(text)))
    except ValueError:
        return text


def _mean_numeric(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.mean())


def _ratio(num_value: int, den_value: int) -> float:
    return float(num_value / den_value) if den_value else 0.0


@lru_cache(maxsize=256)
def _boundary_identity_stats(seq: str, prev_chunk: int, curr_chunk: int) -> dict[str, Any]:
    """Join exact SAM31 global IDs from adjacent chunk boundary audit rows.

    The SAM31 sign-merge front end writes one row per cross-boundary track
    handoff. These rows contain prev_global_id and assigned_global_id, so they
    are a stronger Strategy-A source than sequence-level metadata alone.
    """

    path = Path(f"results/kitti_preprocess/{seq}/sam31_textmatch_caronly_signmerge/metrics/chunk_boundary_audit.csv")
    base: dict[str, Any] = {
        "boundary_audit_path": str(path),
        "boundary_audit_rows": 0,
        "boundary_valid_global_rows": 0,
        "boundary_same_global_rows": 0,
        "boundary_cross_global_rows": 0,
        "boundary_kept_global_id_rows": 0,
        "boundary_new_id_rows": 0,
        "boundary_global_same_ratio": 0.0,
        "boundary_global_cross_ratio": 0.0,
        "boundary_new_id_ratio": 0.0,
        "boundary_mean_mask_iou": None,
        "boundary_mean_box_iou": None,
        "boundary_mean_containment": None,
        "boundary_mean_area_similarity": None,
        "boundary_mean_center_dist_norm": None,
        "boundary_decisions": [],
        "boundary_prev_global_ids": [],
        "boundary_curr_global_ids": [],
        "boundary_identity_confidence": 0.0,
        "boundary_identity_source": "",
        "has_boundary_global_identity": False,
    }
    if not path.exists():
        return base
    try:
        audit = pd.read_csv(path)
    except Exception:
        return base
    if "left_chunk" not in audit or "right_chunk" not in audit:
        return base
    left = pd.to_numeric(audit["left_chunk"], errors="coerce")
    right = pd.to_numeric(audit["right_chunk"], errors="coerce")
    rows = audit[(left == int(prev_chunk)) & (right == int(curr_chunk))].copy()
    if rows.empty:
        return base

    prev_ids = [_clean_global_id(value) for value in rows.get("prev_global_id", pd.Series("", index=rows.index))]
    curr_ids = [_clean_global_id(value) for value in rows.get("assigned_global_id", pd.Series("", index=rows.index))]
    valid_pairs = [(prev, curr) for prev, curr in zip(prev_ids, curr_ids) if prev and curr]
    same_count = sum(1 for prev, curr in valid_pairs if prev == curr)
    cross_count = sum(1 for prev, curr in valid_pairs if prev != curr)
    decisions = rows.get("decision", pd.Series("", index=rows.index)).fillna("").astype(str)
    kept_count = int(decisions.eq("kept_global_id").sum())
    new_id_count = int(decisions.str.contains("new_id", case=False, na=False).sum())

    metric_values = [
        value
        for value in [
            _mean_numeric(rows, "mean_mask_iou"),
            _mean_numeric(rows, "mean_box_iou"),
            _mean_numeric(rows, "mean_containment"),
            _mean_numeric(rows, "area_similarity"),
        ]
        if value is not None
    ]
    support_quality = max(0.0, min(1.0, float(sum(metric_values) / len(metric_values)))) if metric_values else 0.0
    valid_ratio = _ratio(len(valid_pairs), len(rows))
    confidence = float(valid_ratio * support_quality) if valid_pairs else 0.0
    prev_unique = sorted({prev for prev, _ in valid_pairs}, key=lambda x: (len(x), x))
    curr_unique = sorted({curr for _, curr in valid_pairs}, key=lambda x: (len(x), x))
    return {
        **base,
        "boundary_audit_rows": int(len(rows)),
        "boundary_valid_global_rows": int(len(valid_pairs)),
        "boundary_same_global_rows": int(same_count),
        "boundary_cross_global_rows": int(cross_count),
        "boundary_kept_global_id_rows": kept_count,
        "boundary_new_id_rows": new_id_count,
        "boundary_global_same_ratio": _ratio(same_count, len(valid_pairs)),
        "boundary_global_cross_ratio": _ratio(cross_count, len(valid_pairs)),
        "boundary_new_id_ratio": _ratio(new_id_count, len(rows)),
        "boundary_mean_mask_iou": _mean_numeric(rows, "mean_mask_iou"),
        "boundary_mean_box_iou": _mean_numeric(rows, "mean_box_iou"),
        "boundary_mean_containment": _mean_numeric(rows, "mean_containment"),
        "boundary_mean_area_similarity": _mean_numeric(rows, "area_similarity"),
        "boundary_mean_center_dist_norm": _mean_numeric(rows, "center_dist_norm"),
        "boundary_decisions": sorted({text for text in decisions if text}),
        "boundary_prev_global_ids": prev_unique[:16],
        "boundary_curr_global_ids": curr_unique[:16],
        "boundary_identity_confidence": confidence,
        "boundary_identity_source": "sam31_chunk_boundary_audit_global_id_overlap",
        "has_boundary_global_identity": bool(valid_pairs),
    }


def _tracklet_stats(tracklets: pd.DataFrame) -> dict[str, Any]:
    if tracklets.empty:
        return {
            "component_tracklet_available": False,
            "component_tracklet_id": "",
            "same_component_tracklet": False,
            "tracklet_iou_mean": None,
            "tracklet_temporal_consistency": None,
            "feature_match_support_count": 0,
            "match_backed_component_relation": False,
            "match_support_confidence": None,
            "cross_component_match_ratio": None,
        }
    feature_sum = float(pd.to_numeric(tracklets.get("feature_match_support_count", 0), errors="coerce").fillna(0).sum())
    verified_sum = float(pd.to_numeric(tracklets.get("verified_inlier_count", 0), errors="coerce").fillna(0).sum())
    spatial = pd.to_numeric(tracklets.get("match_spatial_coverage", 0), errors="coerce").fillna(0)
    cross = pd.to_numeric(tracklets.get("cross_component_boundary_ratio", 0), errors="coerce").fillna(0)
    same_component = tracklets.get("same_role_proxy", pd.Series(False, index=tracklets.index)).astype(str).str.lower().isin(["true", "1"])
    return {
        "component_tracklet_available": True,
        "component_tracklet_id": "v91_component_proxy_set",
        "same_component_tracklet": bool(same_component.any()),
        "tracklet_iou_mean": None,
        "tracklet_temporal_consistency": float(spatial.mean()) if len(spatial) else None,
        "feature_match_support_count": int(feature_sum),
        "match_backed_component_relation": feature_sum > 0,
        "match_support_confidence": float(verified_sum / feature_sum) if feature_sum > 0 else None,
        "cross_component_match_ratio": float(cross.mean()) if len(cross) else None,
    }


@lru_cache(maxsize=128)
def _radio_tensor_means(path_text: str) -> dict[str, float | None]:
    path = Path(path_text)
    if not path.exists():
        return {"interior": None, "boundary": None, "temporal_stability": None, "lowtrust": None}
    try:
        import torch

        try:
            data = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            data = torch.load(path, map_location="cpu")
    except Exception:
        return {"interior": None, "boundary": None, "temporal_stability": None, "lowtrust": None}
    out: dict[str, float | None] = {}
    for out_key, src_key in [
        ("interior", "object_interior_score"),
        ("boundary", "object_boundary_score"),
        ("temporal_stability", "temporal_stability"),
        ("lowtrust", "radio_lowtrust_score"),
    ]:
        tensor = data.get(src_key) if isinstance(data, dict) else None
        if tensor is None:
            out[out_key] = None
            continue
        try:
            out[out_key] = float(tensor.float().mean().item())
        except Exception:
            out[out_key] = None
    return out


def _radio_stats(paths: list[str], *, prev_chunk: Any = None, curr_chunk: Any = None) -> dict[str, Any]:
    unique = sorted(set(paths))
    means = [_radio_tensor_means(path) for path in unique if Path(path).exists()]
    found_chunks = sorted(
        {
            chunk
            for path in unique
            for chunk in [_chunk_id_from_sidecar_path(Path(path))]
            if chunk is not None and Path(path).exists()
        }
    )
    expected_chunks = []
    for value in [prev_chunk, curr_chunk]:
        try:
            expected_chunks.append(int(float(str(value).strip())))
        except (TypeError, ValueError):
            continue
    return {
        "radio_available": bool(means),
        "radio_sidecar_file_count": len(means),
        "radio_sidecar_paths": unique,
        "radio_sidecar_chunk_ids": found_chunks,
        "radio_expected_chunk_ids": expected_chunks,
        "radio_pair_sidecar_complete": bool(expected_chunks) and set(expected_chunks).issubset(set(found_chunks)),
        "radio_interior_mean": _mean([m["interior"] for m in means]),
        "radio_boundary_mean": _mean([m["boundary"] for m in means]),
        "radio_temporal_stability_mean": _mean([m["temporal_stability"] for m in means]),
        "radio_lowtrust_mean": _mean([m["lowtrust"] for m in means]),
        "radio_join_scope": "chunk_sidecar_tensor_means_unmapped_to_policy_component" if means else "",
    }


def _labelled(df: pd.DataFrame) -> pd.Series:
    return df.get("base_case_type", pd.Series("", index=df.index)).astype(str).isin(["bad", "good"])


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    policy_path = args.v92_root / "phase1_semantic_policy_row_bank/semantic_policy_rows.csv"
    candidate_path = args.v92_root / "phase7_data_source_expansion/semantic_source_expansion_candidate_rows.csv"
    tracklet_path = args.v91_root / "phase1_semantic_topology_tracklets/semantic_topology_tracklet_rows.csv"
    policy = pd.read_csv(policy_path)
    candidates = pd.read_csv(candidate_path)
    tracklets = pd.read_csv(tracklet_path) if tracklet_path.exists() else pd.DataFrame()
    policy["seq"] = policy["seq"].map(seq_text)
    candidates["seq"] = candidates["seq"].map(seq_text)
    if not tracklets.empty:
        tracklets["seq"] = tracklets["seq"].map(seq_text)
    merged = policy.merge(
        candidates,
        on=["seq", "prev_chunk", "curr_chunk", "pair_id"],
        how="left",
        suffixes=("_v92", "_candidate"),
    )
    tracklet_groups = {pid: frame.copy() for pid, frame in tracklets.groupby("pair_id")} if not tracklets.empty else {}
    rows: list[dict[str, Any]] = []
    strategy_rows: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        seq = seq_text(row.get("seq"))
        pid = str(row.get("pair_id") or pair_id(seq, row.get("prev_chunk"), row.get("curr_chunk")))
        stage_paths = _parse_list(row.get("stage_c_manifest_paths"))
        candidate_radio_paths = _parse_list(row.get("radio_sidecar_paths")) + _parse_list(row.get("radio_r5_sidecar_paths"))
        discovered_radio_paths = _discover_radio_sidecars(seq, row.get("prev_chunk"), row.get("curr_chunk"))
        radio_paths = candidate_radio_paths + discovered_radio_paths
        span = _spanning_track_stats(seq, stage_paths)
        boundary = _boundary_identity_stats(seq, int(row.get("prev_chunk")), int(row.get("curr_chunk")))
        tr = _tracklet_stats(tracklet_groups.get(pid, pd.DataFrame()))
        radio = _radio_stats(radio_paths, prev_chunk=row.get("prev_chunk"), curr_chunk=row.get("curr_chunk"))
        has_object_identity = bool(boundary["has_boundary_global_identity"])
        radio_available = bool(radio["radio_available"])
        component_available = bool(tr["component_tracklet_available"]) or _bool(row.get("component_tracklet_available"))
        feature_available = bool(tr["match_backed_component_relation"])
        scope_parts = []
        if has_object_identity:
            scope_parts.append("sam31_boundary_global_id")
        if radio_available:
            scope_parts.append("radio_chunk_tensor_means")
        if component_available:
            scope_parts.append("component_proxy")
        source_scope = "+".join(scope_parts) if scope_parts else "no_joined_object_or_component_source"
        failure_reasons = []
        if has_object_identity:
            failure_reasons.append("sam31_boundary_global_id_join_available")
        elif span["global_object_candidate_confirmed_count"] > 0:
            failure_reasons.append("sam31_global_ids_exist_but_no_policy_component_to_global_id_mapping")
        else:
            failure_reasons.append("no_spanning_sam31_global_track_candidate_for_pair_window")
        if component_available:
            failure_reasons.append("component_tracklet_available_only_as_compact_proxy")
        if radio_available:
            failure_reasons.append("radio_sidecar_tensor_means_available_but_unmapped_to_policy_component")
        elif _bool(row.get("has_radio")):
            failure_reasons.append("radio_candidate_flag_without_loadable_sidecar_mean")
        out = {
            "pair_id": pid,
            "seq": seq,
            "prev_chunk": int(row.get("prev_chunk")),
            "curr_chunk": int(row.get("curr_chunk")),
            "base_case_type": row.get("base_case_type_v92", row.get("base_case_type_candidate", "")),
            "quality_type": row.get("quality_type_v92", row.get("quality_type_candidate", "")),
            "policy_state_v92": row.get("policy_state_v92", row.get("policy_state_candidate", "")),
            "has_object_identity": has_object_identity,
            "has_global_object_id": has_object_identity,
            "prev_global_object_id": boundary["boundary_prev_global_ids"],
            "curr_global_object_id": boundary["boundary_curr_global_ids"],
            "same_global_object_id": bool(
                boundary["boundary_same_global_rows"] > 0 and boundary["boundary_cross_global_rows"] == 0
            )
            if has_object_identity
            else "",
            "object_identity_source": boundary["boundary_identity_source"] if has_object_identity else "none_row_level",
            "object_identity_confidence": boundary["boundary_identity_confidence"],
            "object_id_source": boundary["boundary_identity_source"] if has_object_identity else span["object_id_source"],
            "object_id_confidence": boundary["boundary_identity_confidence"] if has_object_identity else span["object_id_confidence"],
            "global_object_candidate_count": span["global_object_candidate_count"],
            "global_object_candidate_confirmed_count": span["global_object_candidate_confirmed_count"],
            "global_object_candidate_ids": span["global_object_candidate_ids"],
            "same_object_ratio": boundary["boundary_global_same_ratio"]
            if has_object_identity
            else safe_float(row.get("same_object_ratio")),
            "cross_object_ratio": boundary["boundary_global_cross_ratio"]
            if has_object_identity
            else safe_float(row.get("cross_object_boundary_ratio")),
            "object_boundary_ratio": safe_float(row.get("object_boundary_ratio")),
            "object_interior_ratio": safe_float(row.get("object_interior_ratio")),
            "temporal_stability": safe_float(row.get("temporal_stability")),
            "boundary_audit_rows": boundary["boundary_audit_rows"],
            "boundary_valid_global_rows": boundary["boundary_valid_global_rows"],
            "boundary_same_global_rows": boundary["boundary_same_global_rows"],
            "boundary_cross_global_rows": boundary["boundary_cross_global_rows"],
            "boundary_kept_global_id_rows": boundary["boundary_kept_global_id_rows"],
            "boundary_new_id_rows": boundary["boundary_new_id_rows"],
            "boundary_global_same_ratio": boundary["boundary_global_same_ratio"],
            "boundary_global_cross_ratio": boundary["boundary_global_cross_ratio"],
            "boundary_new_id_ratio": boundary["boundary_new_id_ratio"],
            "boundary_mean_mask_iou": boundary["boundary_mean_mask_iou"],
            "boundary_mean_box_iou": boundary["boundary_mean_box_iou"],
            "boundary_mean_containment": boundary["boundary_mean_containment"],
            "boundary_mean_area_similarity": boundary["boundary_mean_area_similarity"],
            "boundary_mean_center_dist_norm": boundary["boundary_mean_center_dist_norm"],
            "boundary_decisions": boundary["boundary_decisions"],
            "boundary_audit_path": boundary["boundary_audit_path"],
            "component_tracklet_id": tr["component_tracklet_id"],
            "tracklet_iou": "",
            "tracklet_iou_mean": tr["tracklet_iou_mean"],
            "tracklet_length": "",
            "tracklet_length_mean": "",
            "tracklet_temporal_consistency": tr["tracklet_temporal_consistency"],
            "same_component_tracklet": tr["same_component_tracklet"],
            "match_backed_component_relation": tr["match_backed_component_relation"],
            "feature_match_support_count": tr["feature_match_support_count"],
            "match_support_count": tr["feature_match_support_count"],
            "match_support_confidence": tr["match_support_confidence"],
            "cross_component_match_ratio": tr["cross_component_match_ratio"],
            "radio_available": radio_available,
            "has_radio": radio_available,
            "radio_sidecar_paths": radio["radio_sidecar_paths"],
            "radio_discovered_sidecar_file_count": len(sorted(set(discovered_radio_paths))),
            "radio_pair_sidecar_complete": radio["radio_pair_sidecar_complete"],
            "radio_sidecar_chunk_ids": radio["radio_sidecar_chunk_ids"],
            "radio_expected_chunk_ids": radio["radio_expected_chunk_ids"],
            "radio_object_interior": radio["radio_interior_mean"],
            "radio_boundary": radio["radio_boundary_mean"],
            "radio_temporal_stability": radio["radio_temporal_stability_mean"],
            "radio_lowtrust": radio["radio_lowtrust_mean"],
            "radio_join_scope": radio["radio_join_scope"],
            "radio_sidecar_file_count": radio["radio_sidecar_file_count"],
            "radio_interior_mean": radio["radio_interior_mean"],
            "radio_boundary_mean": radio["radio_boundary_mean"],
            "radio_temporal_stability_mean": radio["radio_temporal_stability_mean"],
            "source_scope": source_scope,
            "join_failure_reason": ";".join(failure_reasons),
            "no_object_identity_success_claim": not has_object_identity,
            "component_tracklet_available": component_available,
            "feature_match_component_available": feature_available,
            "compact_proxy_only": component_available and not has_object_identity and not radio_available,
            "labelled": str(row.get("base_case_type_v92", row.get("base_case_type_candidate", ""))) in {"bad", "good"},
            "abs_log_scale_jump_gt": row.get("abs_log_scale_jump_gt"),
        }
        rows.append(out)
        strategy_rows.extend(
            [
                {
                    "pair_id": pid,
                    "strategy": "A_exact_global_object_track_id",
                    "status": "sam31_boundary_global_id_join_available"
                    if has_object_identity
                    else "candidate_metadata_found_but_row_level_mapping_absent"
                    if span["global_object_candidate_confirmed_count"] > 0
                    else "no_pair_spanning_confirmed_track_candidate",
                    "evidence": {**span, **boundary},
                    "success_claim": has_object_identity,
                },
                {
                    "pair_id": pid,
                    "strategy": "B_temporal_iou_component_tracklet_join",
                    "status": "component_proxy_available_no_mask_iou_materialized"
                    if component_available
                    else "component_proxy_unavailable",
                    "evidence": {
                        "component_tracklet_available": component_available,
                        "tracklet_iou_mean": tr["tracklet_iou_mean"],
                        "tracklet_temporal_consistency": tr["tracklet_temporal_consistency"],
                    },
                    "success_claim": False,
                },
                {
                    "pair_id": pid,
                    "strategy": "C_feature_match_backed_component_relation",
                    "status": "feature_match_component_proxy_available" if feature_available else "feature_match_component_proxy_unavailable",
                    "evidence": {
                        "feature_match_support_count": tr["feature_match_support_count"],
                        "match_support_confidence": tr["match_support_confidence"],
                    },
                    "success_claim": False,
                },
                {
                    "pair_id": pid,
                    "strategy": "D_radio_radseg_sidecar_join",
                    "status": "radio_chunk_tensor_means_available_unmapped" if radio_available else "radio_unavailable_for_pair",
                    "evidence": {
                        "radio_sidecar_file_count": radio["radio_sidecar_file_count"],
                        "radio_pair_sidecar_complete": radio["radio_pair_sidecar_complete"],
                        "radio_sidecar_chunk_ids": radio["radio_sidecar_chunk_ids"],
                        "radio_interior_mean": radio["radio_interior_mean"],
                        "radio_boundary_mean": radio["radio_boundary_mean"],
                        "radio_temporal_stability_mean": radio["radio_temporal_stability_mean"],
                    },
                    "success_claim": False,
                },
                {
                    "pair_id": pid,
                    "strategy": "E_compact_component_fallback",
                    "status": "diagnostic_component_proxy_only" if component_available else "fallback_unavailable",
                    "evidence": {"source_scope": source_scope},
                    "success_claim": False,
                },
            ]
        )

    out_df = pd.DataFrame(rows)
    labelled = out_df["labelled"].astype(bool)
    row_count = len(out_df)
    labelled_count = int(labelled.sum())
    object_mask = out_df["has_object_identity"].astype(bool)
    radio_mask = out_df["radio_available"].astype(bool)
    component_mask = out_df["component_tracklet_available"].astype(bool)
    feature_mask = out_df["feature_match_component_available"].astype(bool)
    summary = {
        "phase": "Phase1_object_identity_row_join",
        "row_count": row_count,
        "labelled_row_count": labelled_count,
        "sequence_coverage": int(out_df["seq"].nunique()),
        "object_identity_available_ratio": float(object_mask.mean()) if row_count else 0.0,
        "object_identity_labelled_coverage": float(object_mask[labelled].mean()) if labelled_count else 0.0,
        "object_identity_seq_coverage": int(out_df.loc[object_mask, "seq"].nunique()),
        "radio_available_ratio": float(radio_mask.mean()) if row_count else 0.0,
        "radio_labelled_coverage": float(radio_mask[labelled].mean()) if labelled_count else 0.0,
        "radio_seq_coverage": int(out_df.loc[radio_mask, "seq"].nunique()),
        "radio_fields_include_interior_boundary_stability": bool(
            out_df.loc[radio_mask, ["radio_interior_mean", "radio_boundary_mean", "radio_temporal_stability_mean"]]
            .notna()
            .all(axis=None)
        )
        if radio_mask.any()
        else False,
        "component_tracklet_available_ratio": float(component_mask.mean()) if row_count else 0.0,
        "component_tracklet_labelled_coverage": float(component_mask[labelled].mean()) if labelled_count else 0.0,
        "feature_match_component_available_ratio": float(feature_mask.mean()) if row_count else 0.0,
        "feature_match_component_labelled_coverage": float(feature_mask[labelled].mean()) if labelled_count else 0.0,
        "compact_proxy_only_ratio": float(out_df["compact_proxy_only"].astype(bool).mean()) if row_count else 0.0,
        "no_object_identity_success_claim": not (
            (float(object_mask[labelled].mean()) if labelled_count else 0.0) >= 0.50
            and int(out_df.loc[object_mask, "seq"].nunique()) >= 3
        ),
        "strategy_order_attempted": [
            "A_exact_global_object_track_id",
            "B_temporal_iou_component_tracklet_join",
            "C_feature_match_backed_component_relation",
            "D_radio_radseg_sidecar_join",
            "E_compact_component_fallback",
        ],
        "source_scope_counts": out_df["source_scope"].value_counts().to_dict(),
        "join_failure_reason_counts": out_df["join_failure_reason"].value_counts().to_dict(),
        "runtime_action_allowed": False,
        "counterfactual_allowed": False,
        "ttt_allowed": False,
    }
    write_csv(args.out_dir / "object_identity_row_join.csv", rows)
    write_csv(args.out_dir / "object_identity_strategy_audit_rows.csv", strategy_rows)
    write_json(args.out_dir / "object_identity_source_summary.json", summary)
    print(f"row_count={summary['row_count']}")
    print(f"object_identity_available_ratio={summary['object_identity_available_ratio']}")
    print(f"radio_available_ratio={summary['radio_available_ratio']}")
    print(f"component_tracklet_available_ratio={summary['component_tracklet_available_ratio']}")
    print(f"out_dir={args.out_dir}")


if __name__ == "__main__":
    main()
