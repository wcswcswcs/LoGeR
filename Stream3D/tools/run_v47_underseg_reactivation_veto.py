from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from typing import Any

from stream4d_native.v47_common import (
    ROOT,
    UnionFind,
    adjusted_rand_score,
    cluster_completeness,
    cluster_purity,
    parse_bool,
    parse_float,
    parse_int,
    read_csv,
    safe_mean,
    write_csv,
    write_json,
)


def _tracklet_meta(tracklet_rows: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[tuple[str, int, int], str]]:
    meta: dict[str, dict[str, Any]] = {}
    mask_to_track: dict[tuple[str, int, int], str] = {}
    for row in tracklet_rows:
        tid = str(row.get("tracklet_id"))
        scene = str(row.get("scene"))
        frame = parse_int(row.get("frame_id"))
        mask_id = parse_int(row.get("mask_id"))
        item = meta.setdefault(tid, {"tracklet_id": tid, "scene": scene, "frames": set(), "rows": [], "gt_counts": Counter()})
        item["frames"].add(frame)
        item["rows"].append(row)
        gt = str(row.get("diagnostic_gt_instance", ""))
        if gt:
            item["gt_counts"][gt] += 1
        mask_to_track[(scene, frame, mask_id)] = tid
    for item in meta.values():
        frames = sorted(item["frames"])
        item["start_frame"] = frames[0] if frames else -1
        item["end_frame"] = frames[-1] if frames else -1
        if item["gt_counts"]:
            item["dominant_gt"], item["dominant_gt_count"] = item["gt_counts"].most_common(1)[0]
        else:
            item["dominant_gt"], item["dominant_gt_count"] = "", 0
    return meta, mask_to_track


def _primary_track_by_carrier(
    carrier_rows: list[dict[str, Any]],
    mask_to_track: dict[tuple[str, int, int], str],
) -> dict[str, str]:
    votes: dict[str, Counter[str]] = defaultdict(Counter)
    for row in carrier_rows:
        mask_id = parse_int(row.get("observed_mask_id"), 0)
        if mask_id <= 0:
            continue
        if not (parse_bool(row.get("visible")) and parse_bool(row.get("valid_uv")) and parse_bool(row.get("mask_label_available"))):
            continue
        scene = str(row.get("scene"))
        frame = parse_int(row.get("frame_id"))
        tid = mask_to_track.get((scene, frame, mask_id))
        if not tid:
            continue
        carrier_id = str(row.get("carrier_global_id") or f"{scene}:{parse_int(row.get('carrier_id'))}")
        votes[carrier_id][tid] += 1
    return {carrier: counts.most_common(1)[0][0] for carrier, counts in votes.items() if counts}


def _shared_masks(
    carrier_rows: list[dict[str, Any]],
    carrier_to_track: dict[str, str],
) -> list[dict[str, Any]]:
    tracks_by_mask: dict[tuple[str, int, int], set[str]] = defaultdict(set)
    track_support_by_mask: dict[tuple[str, int, int], Counter[str]] = defaultdict(Counter)
    carrier_count_by_mask: Counter[tuple[str, int, int]] = Counter()
    for row in carrier_rows:
        mask_id = parse_int(row.get("observed_mask_id"), 0)
        if mask_id <= 0:
            continue
        if not (parse_bool(row.get("visible")) and parse_bool(row.get("valid_uv")) and parse_bool(row.get("mask_label_available"))):
            continue
        scene = str(row.get("scene"))
        key = (scene, parse_int(row.get("frame_id")), mask_id)
        carrier_id = str(row.get("carrier_global_id") or f"{scene}:{parse_int(row.get('carrier_id'))}")
        tid = carrier_to_track.get(carrier_id)
        if tid:
            tracks_by_mask[key].add(tid)
            track_support_by_mask[key][tid] += 1
            carrier_count_by_mask[key] += 1
    rows = []
    for (scene, frame, mask_id), tids in sorted(tracks_by_mask.items()):
        support_counts = track_support_by_mask[(scene, frame, mask_id)]
        dominant_tid, dominant_count = support_counts.most_common(1)[0] if support_counts else ("", 0)
        carrier_count = carrier_count_by_mask[(scene, frame, mask_id)]
        rows.append(
            {
                "scene": scene,
                "frame_id": frame,
                "mask_id": mask_id,
                "supporting_primary_track_count": len(tids),
                "supporting_carrier_count": carrier_count,
                "supporting_primary_track_ids": sorted(tids),
                "dominant_primary_track_id": dominant_tid,
                "dominant_primary_track_support_count": dominant_count,
                "dominant_primary_track_fraction": float(dominant_count / max(carrier_count, 1)),
                "shared_observation": len(tids) > 1,
                "can_create_identity_merge_edge": False,
                "uses_gt_for_prediction": False,
            }
        )
    return rows


def _endpoint_key(meta: dict[str, dict[str, Any]], tid: str, which: str) -> tuple[str, int, int] | None:
    item = meta.get(tid)
    if not item:
        return None
    target = item["end_frame"] if which == "src" else item["start_frame"]
    rows = [row for row in item["rows"] if parse_int(row.get("frame_id")) == int(target)]
    if not rows:
        return None
    row = rows[0]
    return str(row.get("scene")), parse_int(row.get("frame_id")), parse_int(row.get("mask_id"))


def _metrics(tracklet_rows: list[dict[str, Any]], meta: dict[str, dict[str, Any]], selected_pairs: list[dict[str, Any]]) -> dict[str, Any]:
    ids = sorted(meta)
    idx = {tid: i for i, tid in enumerate(ids)}
    uf = UnionFind(idx.values())
    for row in selected_pairs:
        src = str(row.get("src_tracklet_id"))
        dst = str(row.get("dst_tracklet_id"))
        if src in idx and dst in idx:
            uf.union(idx[src], idx[dst])
    true_labels: list[str] = []
    pred_labels: list[str] = []
    scene_true: dict[str, list[str]] = defaultdict(list)
    scene_pred: dict[str, list[str]] = defaultdict(list)
    cluster_frames: dict[str, set[int]] = defaultdict(set)
    for tid in ids:
        pred = f"u{uf.find(idx[tid])}"
        cluster_frames[pred].update(meta[tid]["frames"])
    for row in tracklet_rows:
        gt = str(row.get("diagnostic_gt_instance", ""))
        if not gt:
            continue
        tid = str(row.get("tracklet_id"))
        pred = f"u{uf.find(idx[tid])}"
        true_labels.append(gt)
        pred_labels.append(pred)
        scene = str(row.get("scene"))
        scene_true[scene].append(gt)
        scene_pred[scene].append(pred)
    labeled = [row for row in selected_pairs if str(row.get("diagnostic_src_gt")) and str(row.get("diagnostic_dst_gt"))]
    false_pairs = [row for row in labeled if not parse_bool(row.get("diagnostic_same_gt"))]
    return {
        "ARI": adjusted_rand_score(true_labels, pred_labels),
        "purity": cluster_purity(true_labels, pred_labels),
        "completeness": cluster_completeness(true_labels, pred_labels),
        "temporal_span_mean": safe_mean(len(frames) for frames in cluster_frames.values()),
        "cluster_count": len(cluster_frames),
        "selected_pair_count": len(selected_pairs),
        "labeled_selected_pair_count": len(labeled),
        "false_merge_pair_count": len(false_pairs),
        "false_merge_rate": float(len(false_pairs) / max(len(labeled), 1)),
        "scene0011_purity": cluster_purity(scene_true["scene0011_00"], scene_pred["scene0011_00"]) if scene_true.get("scene0011_00") else None,
        "scene0050_purity": cluster_purity(scene_true["scene0050_00"], scene_pred["scene0050_00"]) if scene_true.get("scene0050_00") else None,
        "scene0591_purity": cluster_purity(scene_true["scene0591_00"], scene_pred["scene0591_00"]) if scene_true.get("scene0591_00") else None,
    }


def _parse_float_list(text: str) -> list[float]:
    values: list[float] = []
    for item in str(text).split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    return values


def _parse_int_list(text: str) -> list[int]:
    values: list[int] = []
    for item in str(text).split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    return values


def _endpoint_stats(
    key: tuple[str, int, int] | None,
    shared_by_key: dict[tuple[str, int, int], dict[str, Any]],
) -> dict[str, Any]:
    if key is None:
        return {
            "endpoint_key": "",
            "shared": False,
            "supporting_primary_track_count": 0,
            "supporting_carrier_count": 0,
            "dominant_primary_track_fraction": 0.0,
        }
    row = shared_by_key.get(key)
    scene, frame, mask_id = key
    if not row:
        return {
            "endpoint_key": f"{scene}:{frame}:{mask_id}",
            "shared": False,
            "supporting_primary_track_count": 0,
            "supporting_carrier_count": 0,
            "dominant_primary_track_fraction": 0.0,
        }
    return {
        "endpoint_key": f"{scene}:{frame}:{mask_id}",
        "shared": parse_bool(row.get("shared_observation")),
        "supporting_primary_track_count": parse_int(row.get("supporting_primary_track_count")),
        "supporting_carrier_count": parse_int(row.get("supporting_carrier_count")),
        "dominant_primary_track_fraction": parse_float(row.get("dominant_primary_track_fraction")),
    }


def _endpoint_is_ambiguous(
    stats: dict[str, Any],
    min_supporting_tracks: int,
    max_dominant_fraction: float,
    min_supporting_carriers: int,
) -> bool:
    if not parse_bool(stats.get("shared")):
        return False
    if parse_int(stats.get("supporting_primary_track_count")) < int(min_supporting_tracks):
        return False
    if parse_int(stats.get("supporting_carrier_count")) < int(min_supporting_carriers):
        return False
    return parse_float(stats.get("dominant_primary_track_fraction")) <= float(max_dominant_fraction)


def _mark_pairs_for_policy(
    selected_pairs: list[dict[str, Any]],
    meta: dict[str, dict[str, Any]],
    shared_by_key: dict[tuple[str, int, int], dict[str, Any]],
    *,
    endpoint_scope: str,
    min_supporting_tracks: int,
    max_dominant_fraction: float,
    min_supporting_carriers: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    vetoed: list[dict[str, Any]] = []
    for row in selected_pairs:
        src_key = _endpoint_key(meta, str(row.get("src_tracklet_id")), "src")
        dst_key = _endpoint_key(meta, str(row.get("dst_tracklet_id")), "dst")
        src_stats = _endpoint_stats(src_key, shared_by_key)
        dst_stats = _endpoint_stats(dst_key, shared_by_key)
        src_veto = _endpoint_is_ambiguous(
            src_stats,
            min_supporting_tracks=min_supporting_tracks,
            max_dominant_fraction=max_dominant_fraction,
            min_supporting_carriers=min_supporting_carriers,
        )
        dst_veto = _endpoint_is_ambiguous(
            dst_stats,
            min_supporting_tracks=min_supporting_tracks,
            max_dominant_fraction=max_dominant_fraction,
            min_supporting_carriers=min_supporting_carriers,
        )
        if endpoint_scope == "both_endpoint":
            should_veto = src_veto and dst_veto
        elif endpoint_scope == "support_only_no_veto":
            should_veto = False
        else:
            should_veto = src_veto or dst_veto
        marked = dict(
            row,
            src_endpoint_key=src_stats["endpoint_key"],
            dst_endpoint_key=dst_stats["endpoint_key"],
            src_endpoint_shared=src_stats["shared"],
            dst_endpoint_shared=dst_stats["shared"],
            src_endpoint_supporting_primary_track_count=src_stats["supporting_primary_track_count"],
            dst_endpoint_supporting_primary_track_count=dst_stats["supporting_primary_track_count"],
            src_endpoint_supporting_carrier_count=src_stats["supporting_carrier_count"],
            dst_endpoint_supporting_carrier_count=dst_stats["supporting_carrier_count"],
            src_endpoint_dominant_primary_track_fraction=src_stats["dominant_primary_track_fraction"],
            dst_endpoint_dominant_primary_track_fraction=dst_stats["dominant_primary_track_fraction"],
            src_endpoint_policy_veto=src_veto,
            dst_endpoint_policy_veto=dst_veto,
            endpoint_scope=endpoint_scope,
            min_supporting_tracks=min_supporting_tracks,
            max_dominant_fraction=max_dominant_fraction,
            min_supporting_carriers=min_supporting_carriers,
        )
        if should_veto:
            vetoed.append(marked)
        else:
            kept.append(marked)
    return kept, vetoed


def _gate_from_metrics(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    false_reduction = (
        float((before["false_merge_pair_count"] - after["false_merge_pair_count"]) / before["false_merge_pair_count"])
        if before["false_merge_pair_count"] > 0
        else 0.0
    )
    purity_gain = float(after["purity"] - before["purity"])
    completeness_drop = float(before["completeness"] - after["completeness"])
    gate = {
        "underseg_false_merge_reduction_pass": bool(false_reduction >= 0.20),
        "purity_gain_pass": bool(purity_gain >= 0.03),
        "completeness_drop_pass": bool(completeness_drop <= 0.03),
        "scene0011_purity_pass": bool((after["scene0011_purity"] or 0.0) >= (before["scene0011_purity"] or 0.0) - 0.005),
        "scene0050_purity_pass": bool((after["scene0050_purity"] or 0.0) >= (before["scene0050_purity"] or 0.0) - 0.005),
    }
    gate["pass"] = bool(all(gate.values()))
    gate["gate_pass_count"] = sum(1 for value in gate.values() if value is True)
    return {
        "underseg_false_merge_reduction": false_reduction,
        "purity_gain": purity_gain,
        "completeness_drop": completeness_drop,
        "ARI_change": float(after["ARI"] - before["ARI"]),
        "gate": gate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v47 underseg shared-observation veto diagnostic on R5 reactivation.")
    parser.add_argument("--observation-root", default="outputs/audit/v47_observation_tables_metricfix")
    parser.add_argument("--tracklet-root", default="outputs/audit/v47_tracklets_strict_veto_A5")
    parser.add_argument("--reactivation-root", default="outputs/audit/v47_reactivation_gap2_tracklet_strict")
    parser.add_argument("--selected-pairs-file", default="reactivation_best_R5_selected_pairs.csv")
    parser.add_argument("--output-root", default="outputs/audit/v47_underseg_reactivation_veto")
    parser.add_argument("--endpoint-scopes", default="either_endpoint,both_endpoint,support_only_no_veto")
    parser.add_argument("--min-supporting-tracks", default="2,3,4,5,8")
    parser.add_argument("--max-dominant-fractions", default="0.50,0.60,0.70,0.80,0.90,0.95,1.00")
    parser.add_argument("--min-supporting-carriers", default="0,4,8,16")
    args = parser.parse_args()

    carrier_rows = read_csv(ROOT / str(args.observation_root) / "carrier_observation_table.csv")
    tracklet_rows = read_csv(ROOT / str(args.tracklet_root) / "tracklet_rows.csv")
    selected_pairs = read_csv(ROOT / str(args.reactivation_root) / str(args.selected_pairs_file))
    meta, mask_to_track = _tracklet_meta(tracklet_rows)
    carrier_to_track = _primary_track_by_carrier(carrier_rows, mask_to_track)
    shared_rows = _shared_masks(carrier_rows, carrier_to_track)
    shared_by_key = {
        (str(row["scene"]), parse_int(row["frame_id"]), parse_int(row["mask_id"])): row
        for row in shared_rows
    }

    before = _metrics(tracklet_rows, meta, selected_pairs)
    scan_rows: list[dict[str, Any]] = []
    policy_pairs: dict[int, tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {}
    row_index = 0
    for endpoint_scope in [item.strip() for item in str(args.endpoint_scopes).split(",") if item.strip()]:
        for min_supporting_tracks in _parse_int_list(str(args.min_supporting_tracks)):
            for max_dominant_fraction in _parse_float_list(str(args.max_dominant_fractions)):
                for min_supporting_carriers in _parse_int_list(str(args.min_supporting_carriers)):
                    kept, vetoed = _mark_pairs_for_policy(
                        selected_pairs,
                        meta,
                        shared_by_key,
                        endpoint_scope=endpoint_scope,
                        min_supporting_tracks=min_supporting_tracks,
                        max_dominant_fraction=max_dominant_fraction,
                        min_supporting_carriers=min_supporting_carriers,
                    )
                    after = _metrics(tracklet_rows, meta, kept)
                    deltas = _gate_from_metrics(before, after)
                    scan_row = {
                        "scan_row_id": row_index,
                        "endpoint_scope": endpoint_scope,
                        "min_supporting_tracks": min_supporting_tracks,
                        "max_dominant_fraction": max_dominant_fraction,
                        "min_supporting_carriers": min_supporting_carriers,
                        "kept_pair_count": len(kept),
                        "vetoed_pair_count": len(vetoed),
                        **after,
                        "underseg_false_merge_reduction": deltas["underseg_false_merge_reduction"],
                        "purity_gain": deltas["purity_gain"],
                        "completeness_drop": deltas["completeness_drop"],
                        "ARI_change": deltas["ARI_change"],
                        **{f"gate_{key}": value for key, value in deltas["gate"].items()},
                    }
                    scan_rows.append(scan_row)
                    policy_pairs[row_index] = (kept, vetoed)
                    row_index += 1

    def rank_scan(row: dict[str, Any]) -> tuple[Any, ...]:
        return (
            int(parse_bool(row.get("gate_pass"))),
            parse_int(row.get("gate_gate_pass_count")),
            int(parse_bool(row.get("gate_underseg_false_merge_reduction_pass"))),
            int(parse_bool(row.get("gate_completeness_drop_pass"))),
            parse_float(row.get("ARI_change")),
            parse_float(row.get("purity_gain")),
            parse_float(row.get("underseg_false_merge_reduction")),
            parse_int(row.get("kept_pair_count")),
        )

    best_scan_row = max(scan_rows, key=rank_scan) if scan_rows else {}
    best_row_id = parse_int(best_scan_row.get("scan_row_id"), 0)
    kept, vetoed = policy_pairs.get(best_row_id, ([], []))
    after = _metrics(tracklet_rows, meta, kept)
    deltas = _gate_from_metrics(before, after)
    summary = {
        "phase": "v47_underseg_reactivation_veto_scan",
        "selected_pairs_file": str(args.selected_pairs_file),
        "shared_observation_count": sum(1 for row in shared_rows if parse_bool(row.get("shared_observation"))),
        "shared_observation_rows": len(shared_rows),
        "shared_observation_tracks_mean": safe_mean(
            parse_int(row.get("supporting_primary_track_count"))
            for row in shared_rows
            if parse_bool(row.get("shared_observation"))
        ),
        "carrier_primary_track_count": len(carrier_to_track),
        "scan_row_count": len(scan_rows),
        "gate_passing_rows": sum(1 for row in scan_rows if parse_bool(row.get("gate_pass"))),
        "best_scan_row": best_scan_row,
        "before": before,
        "after": after,
        "vetoed_pair_count": len(vetoed),
        "kept_pair_count": len(kept),
        "underseg_false_merge_reduction": deltas["underseg_false_merge_reduction"],
        "purity_gain": deltas["purity_gain"],
        "completeness_drop": deltas["completeness_drop"],
        "ARI_change": deltas["ARI_change"],
        "gate": deltas["gate"],
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }

    out_root = ROOT / str(args.output_root)
    write_csv(out_root / "underseg_shared_observation_rows.csv", shared_rows)
    write_csv(out_root / "underseg_veto_scan_rows.csv", scan_rows)
    write_csv(out_root / "underseg_reactivation_kept_pairs.csv", kept)
    write_csv(out_root / "underseg_reactivation_vetoed_pairs.csv", vetoed)
    write_csv(out_root / "underseg_reactivation_best_kept_pairs.csv", kept)
    write_csv(out_root / "underseg_reactivation_best_vetoed_pairs.csv", vetoed)
    write_json(out_root / "underseg_reactivation_summary.json", summary)
    print({"summary": str(out_root / "underseg_reactivation_summary.json"), "gate": summary["gate"]})


if __name__ == "__main__":
    main()
