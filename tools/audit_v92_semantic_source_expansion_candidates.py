#!/usr/bin/env python3
"""Audit v92 semantic source expansion candidates without claiming unjoined sources."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.v86_soft_latent_utils import read_json, safe_float, write_csv, write_json  # noqa: E402
from tools.v91_semantic_regime_utils import normalize_pair_columns  # noqa: E402
from tools.v92_semantic_policy_carrier_utils import ROOT, V91_ROOT, seq_text  # noqa: E402


DEFAULT_POLICY_ROWS = ROOT / "phase1_semantic_policy_row_bank/semantic_policy_rows.csv"
DEFAULT_TRACKLET_ROWS = V91_ROOT / "phase1_semantic_topology_tracklets/semantic_topology_tracklet_rows.csv"
DEFAULT_OUT = ROOT / "phase7_data_source_expansion"
KITTI_PREPROCESS = Path("results/kitti_preprocess")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-rows", type=Path, default=DEFAULT_POLICY_ROWS)
    parser.add_argument("--tracklet-rows", type=Path, default=DEFAULT_TRACKLET_ROWS)
    parser.add_argument("--kitti-preprocess-root", type=Path, default=KITTI_PREPROCESS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def _chunk_globs(seq_root: Path, chunk: int, names: list[str]) -> list[Path]:
    out: list[Path] = []
    pattern = f"chunk_{int(chunk):03d}_*"
    for name in names:
        for base in sorted(seq_root.glob(name)):
            if not base.is_dir():
                continue
            out.extend(sorted(base.glob(f"{pattern}/radio_sidecar.pt")))
    return [p for p in out if p.exists()]


def _stage_chunk_manifests(seq_root: Path, chunk: int) -> list[Path]:
    out: list[Path] = []
    pattern = f"chunk_{int(chunk):03d}_*"
    for base in sorted(seq_root.glob("stage_c_cache_semantic_chunks*")):
        if not base.is_dir():
            continue
        out.extend(sorted(base.glob(f"{pattern}/manifest.json")))
    return [p for p in out if p.exists()]


def _has_file(path: Path) -> bool:
    return path.exists() and path.is_file()


def _load_track_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _track_metadata_summary(seq_root: Path, prev_chunk: int, curr_chunk: int) -> dict[str, Any]:
    path = seq_root / "sam31_textmatch_caronly_signmerge/debug/track_metadata.json"
    payload = _load_track_metadata(path)
    if not payload:
        return {
            "sam31_track_metadata_available": False,
            "sam31_track_metadata_path": "",
            "sam31_confirmed_tracks": 0,
            "sam31_boundary_spanning_tracks": 0,
        }
    prev_start = int(prev_chunk) * 29
    prev_end = prev_start + 32
    curr_start = int(curr_chunk) * 29
    curr_end = curr_start + 32
    confirmed = 0
    spanning = 0
    for obj in payload.values():
        if not isinstance(obj, dict):
            continue
        if str(obj.get("state", "")).lower() != "confirmed":
            continue
        confirmed += 1
        birth = int(float(obj.get("birth_frame", -10**9) or -10**9))
        last = int(float(obj.get("last_frame", -10**9) or -10**9))
        prev_hit = birth <= prev_end and last >= prev_start
        curr_hit = birth <= curr_end and last >= curr_start
        if prev_hit and curr_hit:
            spanning += 1
    return {
        "sam31_track_metadata_available": True,
        "sam31_track_metadata_path": str(path),
        "sam31_confirmed_tracks": int(confirmed),
        "sam31_boundary_spanning_tracks": int(spanning),
    }


def _tracklet_pair_metrics(tracklets: pd.DataFrame) -> dict[str, Any]:
    if tracklets.empty:
        return {
            "component_tracklet_available": False,
            "same_object_ratio": None,
            "cross_object_boundary_ratio": None,
            "object_interior_ratio": None,
            "object_boundary_ratio": None,
            "temporal_stability": None,
            "tracklet_rows_from_v91": 0,
        }
    same_label = tracklets.get("same_label", pd.Series(False, index=tracklets.index)).astype(str).str.lower().isin({"true", "1", "yes"})
    same_role = tracklets.get("same_role_proxy", pd.Series(False, index=tracklets.index)).astype(str).str.lower().isin({"true", "1", "yes"})
    cross_boundary = pd.to_numeric(tracklets.get("cross_component_boundary_ratio", pd.Series(0.0, index=tracklets.index)), errors="coerce").fillna(0.0)
    interior_prev = pd.to_numeric(tracklets.get("interior_mass_prev", pd.Series(0.0, index=tracklets.index)), errors="coerce").fillna(0.0)
    interior_curr = pd.to_numeric(tracklets.get("interior_mass_curr", pd.Series(0.0, index=tracklets.index)), errors="coerce").fillna(0.0)
    boundary_prev = pd.to_numeric(tracklets.get("boundary_mass_prev", pd.Series(0.0, index=tracklets.index)), errors="coerce").fillna(0.0)
    boundary_curr = pd.to_numeric(tracklets.get("boundary_mass_curr", pd.Series(0.0, index=tracklets.index)), errors="coerce").fillna(0.0)
    match_cov = pd.to_numeric(tracklets.get("match_spatial_coverage", pd.Series(0.0, index=tracklets.index)), errors="coerce").fillna(0.0)
    raw_support = pd.to_numeric(tracklets.get("raw_overlap_support_count", pd.Series(0.0, index=tracklets.index)), errors="coerce").fillna(0.0)
    feature_support = pd.to_numeric(tracklets.get("feature_match_support_count", pd.Series(0.0, index=tracklets.index)), errors="coerce").fillna(0.0)
    return {
        "component_tracklet_available": bool(((raw_support + feature_support) > 0).any()),
        "same_object_ratio": float((same_label & same_role).mean()),
        "cross_object_boundary_ratio": float(cross_boundary.mean()),
        "object_interior_ratio": float(((interior_prev + interior_curr) * 0.5).mean()),
        "object_boundary_ratio": float(((boundary_prev + boundary_curr) * 0.5).mean()),
        "temporal_stability": float(match_cov.mean()),
        "tracklet_rows_from_v91": int(len(tracklets)),
    }


def _mean_bool(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return float(sum(1 for row in rows if bool(row.get(key))) / len(rows))


def _mean_numeric(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [safe_float(row.get(key)) for row in rows]
    clean = [float(v) for v in vals if v is not None]
    if not clean:
        return None
    return float(sum(clean) / len(clean))


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    policy = normalize_pair_columns(pd.read_csv(args.policy_rows))
    policy["seq"] = policy["seq"].map(seq_text)
    tracklet_rows = normalize_pair_columns(pd.read_csv(args.tracklet_rows))
    tracklet_rows["seq"] = tracklet_rows["seq"].map(seq_text)
    tracklets_by_pair = {str(pid): group.copy() for pid, group in tracklet_rows.groupby("pair_id")}

    rows: list[dict[str, Any]] = []
    for _, prow in policy.iterrows():
        seq = seq_text(prow["seq"])
        prev_chunk = int(prow["prev_chunk"])
        curr_chunk = int(prow["curr_chunk"])
        pid = str(prow["pair_id"])
        seq_root = args.kitti_preprocess_root / seq
        radio_names = ["radio_sidecar_chunks*", "radseg_sidecar_chunks*"]
        prev_radio = _chunk_globs(seq_root, prev_chunk, radio_names)
        curr_radio = _chunk_globs(seq_root, curr_chunk, radio_names)
        prev_radio_r5 = _chunk_globs(seq_root, prev_chunk, ["radio_sidecar_chunks_r5_overlap", "radseg_sidecar_chunks*"])
        curr_radio_r5 = _chunk_globs(seq_root, curr_chunk, ["radio_sidecar_chunks_r5_overlap", "radseg_sidecar_chunks*"])
        prev_stage = _stage_chunk_manifests(seq_root, prev_chunk)
        curr_stage = _stage_chunk_manifests(seq_root, curr_chunk)
        thingstuff = seq_root / "videomt_l_vspw_w32_thingstuff/sparse_masklets.pt"
        sam31_sparse = seq_root / "sam31_textmatch_caronly_signmerge/sparse_masklets.pt"
        sam31_metrics = seq_root / "sam31_textmatch_caronly_signmerge/metrics/metrics_per_track.csv"
        meta = _track_metadata_summary(seq_root, prev_chunk, curr_chunk)
        pair_tracklets = tracklets_by_pair.get(pid, pd.DataFrame())
        proxy = _tracklet_pair_metrics(pair_tracklets)
        row = {
            "seq": seq,
            "prev_chunk": prev_chunk,
            "curr_chunk": curr_chunk,
            "pair_id": pid,
            "base_case_type": str(prow.get("base_case_type", "")),
            "policy_state": str(prow.get("policy_state", "")),
            "semantic_shuffle_state": str(prow.get("semantic_shuffle_state", "")),
            "component_shuffle_state": str(prow.get("component_shuffle_state", "")),
            "regime_shuffle_state": str(prow.get("regime_shuffle_state", "")),
            "quality_type": str(prow.get("quality_type", "")),
            "has_radio_ratio_source": "filesystem_sidecar_candidate_not_joined",
            "prev_radio_sidecar_available": bool(prev_radio),
            "curr_radio_sidecar_available": bool(curr_radio),
            "radio_pair_sidecar_available": bool(prev_radio and curr_radio),
            "prev_radio_r5_sidecar_available": bool(prev_radio_r5),
            "curr_radio_r5_sidecar_available": bool(curr_radio_r5),
            "radio_r5_pair_sidecar_available": bool(prev_radio_r5 and curr_radio_r5),
            "radio_sidecar_paths": [str(p) for p in (prev_radio[:3] + curr_radio[:3])],
            "radio_r5_sidecar_paths": [str(p) for p in (prev_radio_r5[:3] + curr_radio_r5[:3])],
            "thingstuff_seq_available": _has_file(thingstuff),
            "thingstuff_path": str(thingstuff) if _has_file(thingstuff) else "",
            "stage_c_prev_manifest_available": bool(prev_stage),
            "stage_c_curr_manifest_available": bool(curr_stage),
            "stage_c_pair_manifest_available": bool(prev_stage and curr_stage),
            "stage_c_manifest_paths": [str(p) for p in (prev_stage[:2] + curr_stage[:2])],
            "sam31_sparse_available": _has_file(sam31_sparse),
            "sam31_metrics_available": _has_file(sam31_metrics),
            "object_identity_available": False,
            "object_identity_availability_reason": "sam31/global_id sidecars exist only as sequence-level metadata; no audited row-level component-to-global-id join exists in v92",
            **meta,
            **proxy,
        }
        row["has_radio"] = bool(row["radio_pair_sidecar_available"] or row["radio_r5_pair_sidecar_available"])
        row["has_track"] = bool(row["object_identity_available"])
        row["source_scope"] = "component_tracklet_proxy_plus_unjoined_sidecar_candidates"
        rows.append(row)

    labelled = [row for row in rows if row.get("base_case_type") in {"bad", "good"}]
    direct_object_labelled = [row for row in labelled if row.get("object_identity_available")]
    fallback_labelled = [row for row in labelled if row.get("component_tracklet_available")]
    source_inventory = []
    for seq, group in pd.DataFrame(rows).groupby("seq"):
        source_inventory.append(
            {
                "seq": seq,
                "rows": int(len(group)),
                "radio_pair_rows": int(group["has_radio"].astype(bool).sum()),
                "thingstuff_seq_available": bool(group["thingstuff_seq_available"].any()),
                "stage_c_pair_rows": int(group["stage_c_pair_manifest_available"].astype(bool).sum()),
                "sam31_metadata_rows": int(group["sam31_track_metadata_available"].astype(bool).sum()),
                "object_identity_rows": int(group["object_identity_available"].astype(bool).sum()),
                "component_tracklet_rows": int(group["component_tracklet_available"].astype(bool).sum()),
            }
        )
    summary = {
        "phase": "Phase7_semantic_source_expansion_candidate_audit",
        "row_count": int(len(rows)),
        "labelled_row_count": int(len(labelled)),
        "sequence_coverage": int(pd.DataFrame(rows)["seq"].nunique()) if rows else 0,
        "has_radio_ratio": _mean_bool(rows, "has_radio"),
        "has_radio_labelled_ratio": float(len([row for row in labelled if row.get("has_radio")]) / max(1, len(labelled))),
        "has_track_ratio": _mean_bool(rows, "has_track"),
        "has_track_labelled_ratio": float(len(direct_object_labelled) / max(1, len(labelled))),
        "object_identity_available_ratio": _mean_bool(rows, "object_identity_available"),
        "object_identity_labelled_coverage": float(len(direct_object_labelled) / max(1, len(labelled))),
        "component_tracklet_available_ratio": _mean_bool(rows, "component_tracklet_available"),
        "component_tracklet_labelled_coverage": float(len(fallback_labelled) / max(1, len(labelled))),
        "thingstuff_seq_available_ratio": _mean_bool(rows, "thingstuff_seq_available"),
        "stage_c_pair_manifest_available_ratio": _mean_bool(rows, "stage_c_pair_manifest_available"),
        "sam31_track_metadata_available_ratio": _mean_bool(rows, "sam31_track_metadata_available"),
        "sam31_boundary_spanning_rows": int(sum(1 for row in rows if int(row.get("sam31_boundary_spanning_tracks") or 0) > 0)),
        "same_object_ratio": _mean_numeric(rows, "same_object_ratio"),
        "cross_object_boundary_ratio": _mean_numeric(rows, "cross_object_boundary_ratio"),
        "object_interior_ratio": _mean_numeric(rows, "object_interior_ratio"),
        "object_boundary_ratio": _mean_numeric(rows, "object_boundary_ratio"),
        "temporal_stability": _mean_numeric(rows, "temporal_stability"),
        "source_scope": "component_tracklet_proxy_plus_unjoined_sidecar_candidates",
        "radio_join_status": "candidate_files_found_but_not_joined_to_policy_rows",
        "object_identity_join_status": "unavailable_no_row_level_component_to_global_id_join",
        "runtime_action_allowed": False,
        "ttt_allowed": False,
    }
    summary["phase7_source_candidate_audit_gate_pass"] = bool(
        summary["row_count"] >= 49
        and summary["sequence_coverage"] >= 4
        and summary["component_tracklet_available_ratio"] >= 0.90
    )
    if not summary["phase7_source_candidate_audit_gate_pass"]:
        summary["blocker"] = "phase7_source_candidate_audit_incomplete"

    write_csv(args.out_dir / "semantic_source_expansion_candidate_rows.csv", rows)
    write_csv(args.out_dir / "semantic_source_expansion_inventory_by_seq.csv", source_inventory)
    write_json(args.out_dir / "semantic_source_expansion_candidate_summary.json", summary)
    print(f"phase7_source_candidate_audit_gate_pass={summary['phase7_source_candidate_audit_gate_pass']}")
    print(f"row_count={summary['row_count']}")
    print(f"labelled_row_count={summary['labelled_row_count']}")
    print(f"has_radio_ratio={summary['has_radio_ratio']}")
    print(f"object_identity_available_ratio={summary['object_identity_available_ratio']}")
    print(f"component_tracklet_available_ratio={summary['component_tracklet_available_ratio']}")
    print(f"source_scope={summary['source_scope']}")


if __name__ == "__main__":
    main()
