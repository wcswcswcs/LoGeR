from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from typing import Any

import numpy as np

from stream4d_native.v47_common import (
    ROOT,
    UnionFind,
    adjusted_rand_score,
    cluster_completeness,
    cluster_purity,
    cosine,
    parse_bool,
    parse_float,
    parse_int,
    read_csv,
    safe_mean,
    write_csv,
    write_json,
)


def _loads_feature(value: Any) -> list[float]:
    if isinstance(value, list):
        return [float(v) for v in value]
    text = str(value or "").strip()
    if not text:
        return []
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        return []
    if not isinstance(loaded, list):
        return []
    return [float(v) for v in loaded]


def _mean_feature(features: list[list[float]]) -> list[float]:
    if not features:
        return []
    arr = np.asarray(features, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return []
    mean = arr.mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if norm > 0:
        mean = mean / norm
    return [float(v) for v in mean.tolist()]


def _stable_shuffle(values: list[str], seed: str) -> dict[str, str]:
    keyed = [(hashlib.sha1(f"{seed}:{idx}:{value}".encode("utf-8")).hexdigest(), value) for idx, value in enumerate(values)]
    keyed.sort()
    ordered = [value for _key, value in keyed]
    if len(ordered) <= 1:
        return {value: value for value in ordered}
    shifted = ordered[1:] + ordered[:1]
    return dict(zip(ordered, shifted))


def _build_tracklets(
    tracklet_rows: list[dict[str, Any]],
    mask_rows: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[int, str]]:
    mask_by_node = {parse_int(row.get("node_id")): row for row in mask_rows}
    tracklets: dict[str, dict[str, Any]] = {}
    node_to_track: dict[int, str] = {}
    for row in tracklet_rows:
        track_id = str(row.get("tracklet_id"))
        node_id = parse_int(row.get("node_id"))
        node_to_track[node_id] = track_id
        meta = tracklets.setdefault(
            track_id,
            {
                "tracklet_id": track_id,
                "scene": str(row.get("scene")),
                "nodes": [],
                "frames": set(),
                "mask_ids": [],
                "gt_counts": Counter(),
                "features": [],
            },
        )
        frame_id = parse_int(row.get("frame_id"))
        meta["nodes"].append(node_id)
        meta["frames"].add(frame_id)
        meta["mask_ids"].append(parse_int(row.get("mask_id")))
        gt = str(row.get("diagnostic_gt_instance", ""))
        if gt:
            meta["gt_counts"][gt] += 1
        mask_row = mask_by_node.get(node_id, {})
        feat = _loads_feature(mask_row.get("core_feature"))
        if feat:
            meta["features"].append(feat)
    for meta in tracklets.values():
        frames = sorted(meta["frames"])
        meta["start_frame"] = frames[0] if frames else -1
        meta["end_frame"] = frames[-1] if frames else -1
        meta["length"] = len(meta["nodes"])
        meta["feature"] = _mean_feature(meta["features"])
        gt_counts = meta["gt_counts"]
        if gt_counts:
            dominant, count = gt_counts.most_common(1)[0]
            meta["dominant_gt"] = dominant
            meta["dominant_gt_count"] = int(count)
            meta["labeled_count"] = int(sum(gt_counts.values()))
            meta["diagnostic_purity"] = float(count / max(sum(gt_counts.values()), 1))
        else:
            meta["dominant_gt"] = ""
            meta["dominant_gt_count"] = 0
            meta["labeled_count"] = 0
            meta["diagnostic_purity"] = None
    return tracklets, node_to_track


def _baseline_metrics(tracklets: dict[str, dict[str, Any]], tracklet_rows: list[dict[str, Any]], uf: UnionFind | None = None) -> dict[str, Any]:
    if uf is None:
        ids = list(range(len(tracklets)))
        del ids
    true_labels: list[str] = []
    pred_labels: list[str] = []
    track_index = {tid: idx for idx, tid in enumerate(sorted(tracklets))}
    for row in tracklet_rows:
        gt = str(row.get("diagnostic_gt_instance", ""))
        if not gt:
            continue
        tid = str(row.get("tracklet_id"))
        pred = tid if uf is None else f"r{uf.find(track_index[tid])}"
        true_labels.append(gt)
        pred_labels.append(pred)
    cluster_frames: dict[str, set[int]] = defaultdict(set)
    if uf is None:
        for tid, meta in tracklets.items():
            cluster_frames[tid].update(meta["frames"])
    else:
        for tid, meta in tracklets.items():
            cluster_frames[f"r{uf.find(track_index[tid])}"].update(meta["frames"])
    return {
        "ARI": adjusted_rand_score(true_labels, pred_labels),
        "purity": cluster_purity(true_labels, pred_labels),
        "completeness": cluster_completeness(true_labels, pred_labels),
        "temporal_span_mean": safe_mean(len(frames) for frames in cluster_frames.values()),
        "cluster_count": len(cluster_frames),
    }


def _candidate_rows(
    *,
    edge_rows: list[dict[str, Any]],
    tracklets: dict[str, dict[str, Any]],
    node_to_track: dict[int, str],
    max_gap: int,
    edge_types: set[str],
) -> list[dict[str, Any]]:
    by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    tracklet_ids = sorted(tracklets)
    shuffled_map = _stable_shuffle(tracklet_ids, "v47_reactivation_semantic_shuffle")
    shuffled_feature = {tid: tracklets.get(shuffled_map.get(tid, tid), {}).get("feature", []) for tid in tracklet_ids}
    for row in edge_rows:
        if str(row.get("edge_type")) not in edge_types:
            continue
        src_track = node_to_track.get(parse_int(row.get("src_node_id")))
        dst_track = node_to_track.get(parse_int(row.get("dst_node_id")))
        if not src_track or not dst_track or src_track == dst_track:
            continue
        left = tracklets[src_track]
        right = tracklets[dst_track]
        if left["scene"] != right["scene"]:
            continue
        src_frame = parse_int(row.get("src_frame_id"))
        dst_frame = parse_int(row.get("dst_frame_id"))
        if src_frame != parse_int(left.get("end_frame")) or dst_frame != parse_int(right.get("start_frame")):
            continue
        gap = int(dst_frame - src_frame)
        if gap <= 0 or gap > int(max_gap):
            continue
        if set(left["frames"]) & set(right["frames"]):
            continue
        key = (src_track, dst_track)
        semantic = cosine(left.get("feature", []), right.get("feature", []))
        semantic_shuffled = cosine(left.get("feature", []), shuffled_feature.get(dst_track, []))
        d4rt = parse_float(row.get("A5_d4rt_semantic_confirmation"))
        item = {
            "src_tracklet_id": src_track,
            "dst_tracklet_id": dst_track,
            "scene": left["scene"],
            "src_end_frame": src_frame,
            "dst_start_frame": dst_frame,
            "gap": gap,
            "edge_type": str(row.get("edge_type")),
            "A5_d4rt_semantic_confirmation": d4rt,
            "A4_d4rt_visible_veto": parse_float(row.get("A4_d4rt_visible_veto")),
            "A7_shuffled_D4RT": parse_float(row.get("A7_shuffled_D4RT")),
            "A8_no_temporal_control": parse_float(row.get("A8_no_temporal_control")),
            "semantic_memory_similarity": semantic,
            "semantic_memory_shuffled": semantic_shuffled,
            "visible_outside": parse_float(row.get("visible_outside"), 1.0),
            "forward_visible_carrier_count": parse_int(row.get("forward_visible_carrier_count")),
            "backward_visible_carrier_count": parse_int(row.get("backward_visible_carrier_count")),
            "diagnostic_src_gt": left.get("dominant_gt", ""),
            "diagnostic_dst_gt": right.get("dominant_gt", ""),
            "diagnostic_same_gt": bool(left.get("dominant_gt") and left.get("dominant_gt") == right.get("dominant_gt")),
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic_labels": True,
        }
        prev = by_pair.get(key)
        if prev is None or d4rt > parse_float(prev.get("A5_d4rt_semantic_confirmation")):
            by_pair[key] = item
    return list(by_pair.values())


def _evaluate_variant(
    *,
    name: str,
    tracklets: dict[str, dict[str, Any]],
    tracklet_rows: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    score_key: str,
    semantic_key: str,
    min_d4rt: float,
    min_semantic: float,
    max_visible_outside: float,
    min_margin: float,
    max_accepts: int,
    require_both: bool,
    use_d4rt: bool,
    use_semantic: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    base = _baseline_metrics(tracklets, tracklet_rows)
    track_index = {tid: idx for idx, tid in enumerate(sorted(tracklets))}
    uf = UnionFind(track_index.values())
    by_src: dict[str, list[dict[str, Any]]] = defaultdict(list)
    filtered: list[dict[str, Any]] = []
    for row in candidates:
        d4rt_score = parse_float(row.get(score_key))
        sem_score = parse_float(row.get(semantic_key))
        if parse_float(row.get("visible_outside"), 1.0) > float(max_visible_outside):
            continue
        if use_d4rt and d4rt_score < float(min_d4rt):
            continue
        if use_semantic and sem_score < float(min_semantic):
            continue
        if require_both and (d4rt_score < float(min_d4rt) or sem_score < float(min_semantic)):
            continue
        combined = (d4rt_score if use_d4rt else 0.0) + (sem_score if use_semantic else 0.0)
        item = dict(row, variant=name, selected_score=float(combined), selected_d4rt_score=float(d4rt_score), selected_semantic_score=float(sem_score))
        filtered.append(item)
        by_src[str(row["src_tracklet_id"])].append(item)

    margin_ok: set[tuple[str, str]] = set()
    for src, rows in by_src.items():
        rows.sort(key=lambda r: parse_float(r.get("selected_score")), reverse=True)
        if not rows:
            continue
        top = parse_float(rows[0].get("selected_score"))
        second = parse_float(rows[1].get("selected_score")) if len(rows) > 1 else 0.0
        if top - second >= float(min_margin):
            margin_ok.add((src, str(rows[0]["dst_tracklet_id"])))

    filtered.sort(key=lambda r: (parse_float(r.get("selected_score")), -parse_float(r.get("visible_outside"), 1.0)), reverse=True)
    selected: list[dict[str, Any]] = []
    used_src: set[str] = set()
    used_dst: set[str] = set()
    cluster_frames = {tid: set(meta["frames"]) for tid, meta in tracklets.items()}
    skipped_frame_conflict = 0
    for row in filtered:
        src = str(row["src_tracklet_id"])
        dst = str(row["dst_tracklet_id"])
        if (src, dst) not in margin_ok:
            continue
        if src in used_src or dst in used_dst:
            continue
        root_src = uf.find(track_index[src])
        root_dst = uf.find(track_index[dst])
        if root_src == root_dst:
            continue
        root_src_id = sorted(tracklets)[root_src]
        root_dst_id = sorted(tracklets)[root_dst]
        if cluster_frames[root_src_id] & cluster_frames[root_dst_id]:
            skipped_frame_conflict += 1
            continue
        uf.union(root_src, root_dst)
        new_root = uf.find(root_src)
        new_id = sorted(tracklets)[new_root]
        cluster_frames[new_id] = cluster_frames[root_src_id] | cluster_frames[root_dst_id]
        selected.append(dict(row, selected_rank=len(selected)))
        used_src.add(src)
        used_dst.add(dst)
        if max_accepts >= 0 and len(selected) >= int(max_accepts):
            break

    after = _baseline_metrics(tracklets, tracklet_rows, uf)
    labeled_selected = [row for row in selected if str(row.get("diagnostic_src_gt")) and str(row.get("diagnostic_dst_gt"))]
    same_gt_selected = [row for row in labeled_selected if parse_bool(row.get("diagnostic_same_gt"))]
    same_gt_candidates = [row for row in filtered if parse_bool(row.get("diagnostic_same_gt"))]
    result = {
        "variant": name,
        "score_key": score_key,
        "semantic_key": semantic_key,
        "min_d4rt": float(min_d4rt),
        "min_semantic": float(min_semantic),
        "max_visible_outside": float(max_visible_outside),
        "min_margin": float(min_margin),
        "max_accepts": int(max_accepts),
        "require_both": bool(require_both),
        "candidate_count_after_filter": len(filtered),
        "reactivation_accept_count": len(selected),
        "skipped_frame_conflict": skipped_frame_conflict,
        "reactivation_precision": float(len(same_gt_selected) / max(len(labeled_selected), 1)),
        "reactivation_recall": float(len(same_gt_selected) / max(len(same_gt_candidates), 1)),
        "base_ARI": base["ARI"],
        "ARI": after["ARI"],
        "ARI_gain": float(after["ARI"] - base["ARI"]),
        "base_purity": base["purity"],
        "purity": after["purity"],
        "purity_change": float(after["purity"] - base["purity"]),
        "base_completeness": base["completeness"],
        "completeness": after["completeness"],
        "completeness_change": float(after["completeness"] - base["completeness"]),
        "base_temporal_span_mean": base["temporal_span_mean"],
        "temporal_span_mean": after["temporal_span_mean"],
        "temporal_span_gain": float((after["temporal_span_mean"] or 0.0) - (base["temporal_span_mean"] or 0.0)),
        "base_cluster_count": base["cluster_count"],
        "cluster_count": after["cluster_count"],
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }
    return result, selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v47 tracklet semantic/D4RT reactivation audit.")
    parser.add_argument("--observation-root", default="outputs/audit/v47_observation_tables_metricfix")
    parser.add_argument("--tracklet-root", default="outputs/audit/v47_tracklets_strict_veto_A5")
    parser.add_argument("--edge-table", default="outputs/audit/v47_adjacent_edges_gap2_only/temporal_candidate_edge_table.csv")
    parser.add_argument("--edge-types", default="adjacent,skip")
    parser.add_argument("--max-gap", type=int, default=5)
    parser.add_argument("--min-d4rt-values", default="0.97,0.90,0.80,0.70")
    parser.add_argument("--min-semantic-values", default="0.97,0.90,0.80")
    parser.add_argument("--max-visible-outside-values", default="0.45,0.75,1.0")
    parser.add_argument("--min-margin-values", default="0.00,0.05,0.10")
    parser.add_argument("--max-accept-values", default="25,50,100,-1")
    parser.add_argument("--output-root", default="outputs/audit/v47_reactivation_gap2_tracklet_strict")
    args = parser.parse_args()

    mask_rows = read_csv(ROOT / str(args.observation_root) / "mask_observation_table.csv")
    tracklet_rows = read_csv(ROOT / str(args.tracklet_root) / "tracklet_rows.csv")
    edge_rows = read_csv(ROOT / str(args.edge_table))
    tracklets, node_to_track = _build_tracklets(tracklet_rows, mask_rows)
    candidates = _candidate_rows(
        edge_rows=edge_rows,
        tracklets=tracklets,
        node_to_track=node_to_track,
        max_gap=int(args.max_gap),
        edge_types={item.strip() for item in str(args.edge_types).split(",") if item.strip()},
    )

    rows: list[dict[str, Any]] = []
    selected_by_signature: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for min_d4rt in [float(v) for v in str(args.min_d4rt_values).split(",") if v.strip()]:
        for min_sem in [float(v) for v in str(args.min_semantic_values).split(",") if v.strip()]:
            for max_out in [float(v) for v in str(args.max_visible_outside_values).split(",") if v.strip()]:
                for margin in [float(v) for v in str(args.min_margin_values).split(",") if v.strip()]:
                    for max_accept in [int(float(v)) for v in str(args.max_accept_values).split(",") if v.strip()]:
                        specs = [
                            ("R1_d4rt_only", "A5_d4rt_semantic_confirmation", "semantic_memory_similarity", True, False, False),
                            ("R2_semantic_only", "A5_d4rt_semantic_confirmation", "semantic_memory_similarity", False, True, False),
                            ("R5_d4rt_semantic_margin", "A5_d4rt_semantic_confirmation", "semantic_memory_similarity", True, True, True),
                            ("R6_shuffled_d4rt", "A7_shuffled_D4RT", "semantic_memory_similarity", True, True, True),
                            ("R7_semantic_shuffled", "A5_d4rt_semantic_confirmation", "semantic_memory_shuffled", True, True, True),
                        ]
                        for name, score_key, semantic_key, use_d4rt, use_semantic, require_both in specs:
                            result, selected = _evaluate_variant(
                                name=name,
                                tracklets=tracklets,
                                tracklet_rows=tracklet_rows,
                                candidates=candidates,
                                score_key=score_key,
                                semantic_key=semantic_key,
                                min_d4rt=min_d4rt,
                                min_semantic=min_sem,
                                max_visible_outside=max_out,
                                min_margin=margin,
                                max_accepts=max_accept,
                                require_both=require_both,
                                use_d4rt=use_d4rt,
                                use_semantic=use_semantic,
                            )
                            signature = (
                                result["variant"],
                                result["score_key"],
                                result["semantic_key"],
                                result["min_d4rt"],
                                result["min_semantic"],
                                result["max_visible_outside"],
                                result["min_margin"],
                                result["max_accepts"],
                            )
                            selected_by_signature[signature] = selected
                            rows.append(result)

    rows.sort(key=lambda row: (parse_float(row.get("ARI_gain")), parse_float(row.get("reactivation_precision"))), reverse=True)
    r1_rows = [row for row in rows if row["variant"] == "R1_d4rt_only"]
    r2_rows = [row for row in rows if row["variant"] == "R2_semantic_only"]
    r5_rows = [row for row in rows if row["variant"] == "R5_d4rt_semantic_margin"]
    r6_rows = [row for row in rows if row["variant"] == "R6_shuffled_d4rt"]
    r7_rows = [row for row in rows if row["variant"] == "R7_semantic_shuffled"]
    best_r1 = max(r1_rows, key=lambda row: (parse_float(row.get("ARI_gain")), parse_float(row.get("reactivation_precision"))), default={})
    best_r2 = max(r2_rows, key=lambda row: (parse_float(row.get("ARI_gain")), parse_float(row.get("reactivation_precision"))), default={})
    best_r5 = max(r5_rows, key=lambda row: (parse_float(row.get("ARI_gain")), parse_float(row.get("reactivation_precision"))), default={})
    best_r6 = max(r6_rows, key=lambda row: (parse_float(row.get("ARI_gain")), parse_float(row.get("reactivation_precision"))), default={})
    best_r7 = max(r7_rows, key=lambda row: (parse_float(row.get("ARI_gain")), parse_float(row.get("reactivation_precision"))), default={})
    feasible_main = [
        row
        for row in [best_r1, best_r5]
        if row
        and parse_float(row.get("reactivation_precision")) >= 0.80
        and parse_float(row.get("temporal_span_gain")) >= 0.20
        and parse_float(row.get("ARI_gain")) >= 0.02
        and parse_float(row.get("purity_change")) >= -0.005
    ]
    recommended = max(
        feasible_main,
        key=lambda row: (parse_float(row.get("ARI_gain")), parse_float(row.get("reactivation_precision"))),
        default=best_r5,
    )

    def signature(row: dict[str, Any]) -> tuple[Any, ...]:
        return (
            row.get("variant"),
            row.get("score_key"),
            row.get("semantic_key"),
            row.get("min_d4rt"),
            row.get("min_semantic"),
            row.get("max_visible_outside"),
            row.get("min_margin"),
            row.get("max_accepts"),
        )

    summary = {
        "phase": "v47_reactivation_audit",
        "tracklet_root": str(ROOT / str(args.tracklet_root)),
        "edge_table": str(ROOT / str(args.edge_table)),
        "max_gap": int(args.max_gap),
        "tracklet_count": len(tracklets),
        "reactivation_candidate_count": len(candidates),
        "rows": len(rows),
        "best_row": rows[0] if rows else None,
        "best_R1_row": best_r1,
        "best_R2_row": best_r2,
        "best_R5_row": best_r5,
        "best_R6_shuffled_d4rt_row": best_r6,
        "best_R7_semantic_shuffled_row": best_r7,
        "recommended_reactivation_row": recommended,
        "recommended_reactivation_variant": recommended.get("variant"),
        "semantic_memory_helped_over_d4rt_only": bool(parse_float(best_r5.get("ARI_gain")) > parse_float(best_r1.get("ARI_gain"))),
        "d4rt_only_reactivation_gate": {
            "precision_pass": bool(parse_float(best_r1.get("reactivation_precision")) >= 0.80),
            "temporal_span_gain_pass": bool(parse_float(best_r1.get("temporal_span_gain")) >= 0.20),
            "ARI_gain_pass": bool(parse_float(best_r1.get("ARI_gain")) >= 0.02),
            "purity_drop_pass": bool(parse_float(best_r1.get("purity_change")) >= -0.005),
        },
        "R5_real_minus_R6_shuffled_d4rt_ARI_gain": parse_float(best_r5.get("ARI_gain")) - parse_float(best_r6.get("ARI_gain")),
        "R5_real_minus_R7_semantic_shuffled_ARI_gain": parse_float(best_r5.get("ARI_gain")) - parse_float(best_r7.get("ARI_gain")),
        "gate": {
            "precision_pass": bool(parse_float(best_r5.get("reactivation_precision")) >= 0.80),
            "temporal_span_gain_pass": bool(parse_float(best_r5.get("temporal_span_gain")) >= 0.20),
            "ARI_gain_pass": bool(parse_float(best_r5.get("ARI_gain")) >= 0.02),
            "purity_drop_pass": bool(parse_float(best_r5.get("purity_change")) >= -0.005),
            "real_d4rt_beats_shuffled_pass": bool(
                parse_float(best_r5.get("ARI_gain")) > parse_float(best_r6.get("ARI_gain"))
            ),
            "real_semantic_beats_shuffled_pass": bool(
                parse_float(best_r5.get("ARI_gain")) > parse_float(best_r7.get("ARI_gain"))
            ),
        },
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }
    summary["gate"]["pass"] = bool(all(summary["gate"].values()))
    summary["d4rt_only_reactivation_gate"]["pass"] = bool(all(summary["d4rt_only_reactivation_gate"].values()))

    out_root = ROOT / str(args.output_root)
    write_csv(out_root / "reactivation_scan_rows.csv", rows)
    write_csv(out_root / "reactivation_candidate_rows.csv", candidates)
    write_csv(out_root / "reactivation_best_R1_selected_pairs.csv", selected_by_signature.get(signature(best_r1), []))
    write_csv(out_root / "reactivation_best_R2_selected_pairs.csv", selected_by_signature.get(signature(best_r2), []))
    write_csv(out_root / "reactivation_best_R5_selected_pairs.csv", selected_by_signature.get(signature(best_r5), []))
    write_csv(out_root / "reactivation_best_R6_selected_pairs.csv", selected_by_signature.get(signature(best_r6), []))
    write_csv(out_root / "reactivation_best_R7_selected_pairs.csv", selected_by_signature.get(signature(best_r7), []))
    write_json(out_root / "reactivation_summary.json", summary)
    print({"summary": str(out_root / "reactivation_summary.json"), "gate": summary["gate"]})


if __name__ == "__main__":
    main()
