from __future__ import annotations

import argparse
from collections import defaultdict
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
from tools.run_v47_reactivation_audit import _build_tracklets


def _parse_float_list(spec: str) -> list[float]:
    return [float(item) for item in str(spec).split(",") if item.strip()]


def _parse_int_list(spec: str) -> list[int]:
    return [int(float(item)) for item in str(spec).split(",") if item.strip()]


def _track_index(tracklets: dict[str, dict[str, Any]]) -> tuple[list[str], dict[str, int]]:
    ids = sorted(tracklets)
    return ids, {tid: idx for idx, tid in enumerate(ids)}


def _init_state(
    *,
    tracklets: dict[str, dict[str, Any]],
    tracklet_ids: list[str],
    track_index: dict[str, int],
    base_pairs: list[dict[str, Any]],
) -> tuple[UnionFind, dict[int, set[int]], int]:
    uf = UnionFind(range(len(tracklet_ids)))
    cluster_frames = {idx: set(tracklets[tid]["frames"]) for tid, idx in track_index.items()}
    base_merge_count = 0
    for row in base_pairs:
        src = str(row.get("src_tracklet_id"))
        dst = str(row.get("dst_tracklet_id"))
        if src not in track_index or dst not in track_index:
            continue
        left = uf.find(track_index[src])
        right = uf.find(track_index[dst])
        if left == right:
            continue
        if cluster_frames[left] & cluster_frames[right]:
            continue
        merged = uf.union(left, right)
        if merged:
            root = uf.find(left)
            other = right if root == left else left
            cluster_frames[root] = set(cluster_frames.get(left, set())) | set(cluster_frames.get(right, set()))
            cluster_frames.pop(other, None)
            base_merge_count += 1
    return uf, cluster_frames, base_merge_count


def _metrics(
    *,
    tracklet_rows: list[dict[str, Any]],
    tracklets: dict[str, dict[str, Any]],
    track_index: dict[str, int],
    uf: UnionFind,
    hard_negative_pairs: set[tuple[int, int]],
) -> dict[str, Any]:
    true_labels: list[str] = []
    pred_labels: list[str] = []
    scene_true: dict[str, list[str]] = defaultdict(list)
    scene_pred: dict[str, list[str]] = defaultdict(list)
    cluster_frames: dict[str, set[int]] = defaultdict(set)
    scene_clusters: dict[str, set[str]] = defaultdict(set)
    for tid, idx in track_index.items():
        pred = f"u{uf.find(idx)}"
        cluster_frames[pred].update(tracklets[tid]["frames"])
        scene_clusters[str(tracklets[tid]["scene"])].add(pred)
    for row in tracklet_rows:
        gt = str(row.get("diagnostic_gt_instance", ""))
        if not gt:
            continue
        tid = str(row.get("tracklet_id"))
        if tid not in track_index:
            continue
        pred = f"u{uf.find(track_index[tid])}"
        true_labels.append(gt)
        pred_labels.append(pred)
        scene = str(row.get("scene"))
        scene_true[scene].append(gt)
        scene_pred[scene].append(pred)
    hard_negative_violation_count = sum(1 for left, right in hard_negative_pairs if uf.find(left) == uf.find(right))

    def first_scene(prefix: str) -> str:
        return next((scene for scene in sorted(scene_true) if scene.startswith(prefix)), "")

    def scene_ari(prefix: str) -> float | None:
        scene = first_scene(prefix)
        return adjusted_rand_score(scene_true[scene], scene_pred[scene]) if scene else None

    def scene_purity(prefix: str) -> float | None:
        scene = first_scene(prefix)
        return cluster_purity(scene_true[scene], scene_pred[scene]) if scene else None

    def scene_completeness(prefix: str) -> float | None:
        scene = first_scene(prefix)
        return cluster_completeness(scene_true[scene], scene_pred[scene]) if scene else None

    return {
        "ARI": adjusted_rand_score(true_labels, pred_labels),
        "purity": cluster_purity(true_labels, pred_labels),
        "completeness": cluster_completeness(true_labels, pred_labels),
        "temporal_span_mean": safe_mean(len(frames) for frames in cluster_frames.values()),
        "cluster_count": len(cluster_frames),
        "mean_predictions_per_scene": safe_mean(len(clusters) for clusters in scene_clusters.values()),
        "hard_negative_count": len(hard_negative_pairs),
        "hard_negative_violation_count": hard_negative_violation_count,
        "hard_negative_violation_rate": float(hard_negative_violation_count / max(len(hard_negative_pairs), 1)),
        "scene0081_ARI": scene_ari("scene0081"),
        "scene0011_purity": scene_purity("scene0011"),
        "scene0050_purity": scene_purity("scene0050"),
        "scene0591_ARI": scene_ari("scene0591"),
        "scene0591_purity": scene_purity("scene0591"),
        "scene0591_completeness": scene_completeness("scene0591"),
        "birth_from_d4rt_tube_count": 0,
        "maskless_object_count": 0,
    }


def _hard_negative_pairs(
    *,
    candidates: list[dict[str, Any]],
    track_index: dict[str, int],
    min_visible_carriers: int,
    min_visible_outside: float,
) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for row in candidates:
        if parse_float(row.get("visible_outside"), 0.0) < float(min_visible_outside):
            continue
        carriers = parse_int(row.get("forward_visible_carrier_count")) + parse_int(row.get("backward_visible_carrier_count"))
        if carriers < int(min_visible_carriers):
            continue
        src = str(row.get("src_tracklet_id"))
        dst = str(row.get("dst_tracklet_id"))
        if src not in track_index or dst not in track_index:
            continue
        left, right = sorted([track_index[src], track_index[dst]])
        if left != right:
            pairs.add((left, right))
    return pairs


def _would_violate_hard_negative(
    *,
    uf: UnionFind,
    left_root: int,
    right_root: int,
    hard_negative_pairs: set[tuple[int, int]],
) -> bool:
    left_members = uf.members.get(left_root, {left_root})
    right_members = uf.members.get(right_root, {right_root})
    if len(left_members) > len(right_members):
        left_members, right_members = right_members, left_members
    for left in left_members:
        for right in right_members:
            pair = tuple(sorted([left, right]))
            if pair in hard_negative_pairs:
                return True
    return False


def _candidate_score(row: dict[str, Any], variant: str, negative_weight: float) -> float:
    if variant == "G5_dense_semantic_control":
        return parse_float(row.get("semantic_memory_similarity"))
    d4rt = parse_float(row.get("A5_d4rt_semantic_confirmation"))
    semantic = parse_float(row.get("semantic_memory_similarity"))
    score = d4rt + semantic
    if variant == "G2_local_signed_soft":
        score -= float(negative_weight) * parse_float(row.get("visible_outside"), 0.0)
    return float(score)


def _filter_candidates(
    *,
    candidates: list[dict[str, Any]],
    variant: str,
    min_d4rt: float,
    min_semantic: float,
    max_visible_outside: float,
    min_margin: float,
    topk_per_src: int,
    negative_weight: float,
) -> list[dict[str, Any]]:
    by_src: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        if variant != "G5_dense_semantic_control" and parse_float(row.get("A5_d4rt_semantic_confirmation")) < float(min_d4rt):
            continue
        if parse_float(row.get("semantic_memory_similarity")) < float(min_semantic):
            continue
        if variant in {"G1_local_positive_edges", "G3_local_hard_negative_veto"}:
            if parse_float(row.get("visible_outside"), 1.0) > float(max_visible_outside):
                continue
        score = _candidate_score(row, variant, negative_weight)
        item = dict(row, local_graph_score=score, variant=variant)
        by_src[str(row.get("src_tracklet_id"))].append(item)
    filtered: list[dict[str, Any]] = []
    for rows in by_src.values():
        rows.sort(key=lambda item: parse_float(item.get("local_graph_score")), reverse=True)
        for rank, row in enumerate(rows[: max(int(topk_per_src), 1)]):
            second = parse_float(rows[rank + 1].get("local_graph_score")) if rank + 1 < len(rows) else 0.0
            if parse_float(row.get("local_graph_score")) - second < float(min_margin):
                continue
            filtered.append(dict(row, local_graph_src_rank=rank))
    filtered.sort(
        key=lambda item: (
            parse_float(item.get("local_graph_score")),
            parse_float(item.get("A5_d4rt_semantic_confirmation")),
            parse_float(item.get("semantic_memory_similarity")),
            -parse_float(item.get("visible_outside"), 1.0),
        ),
        reverse=True,
    )
    return filtered


def _evaluate(
    *,
    variant: str,
    tracklets: dict[str, dict[str, Any]],
    tracklet_rows: list[dict[str, Any]],
    tracklet_ids: list[str],
    track_index: dict[str, int],
    candidates: list[dict[str, Any]],
    base_pairs: list[dict[str, Any]],
    base_metrics: dict[str, Any],
    hard_negative_pairs: set[tuple[int, int]],
    min_d4rt: float,
    min_semantic: float,
    max_visible_outside: float,
    min_margin: float,
    max_accepts: int,
    topk_per_src: int,
    negative_weight: float,
    enforce_hard_negative: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    uf, cluster_frames, base_merge_count = _init_state(
        tracklets=tracklets,
        tracklet_ids=tracklet_ids,
        track_index=track_index,
        base_pairs=base_pairs,
    )
    filtered = _filter_candidates(
        candidates=candidates,
        variant=variant,
        min_d4rt=min_d4rt,
        min_semantic=min_semantic,
        max_visible_outside=max_visible_outside,
        min_margin=min_margin,
        topk_per_src=topk_per_src,
        negative_weight=negative_weight,
    )
    selected: list[dict[str, Any]] = []
    skipped_frame_conflict = 0
    skipped_hard_negative = 0
    for row in filtered:
        src = str(row.get("src_tracklet_id"))
        dst = str(row.get("dst_tracklet_id"))
        if src not in track_index or dst not in track_index:
            continue
        left = uf.find(track_index[src])
        right = uf.find(track_index[dst])
        if left == right:
            continue
        if cluster_frames[left] & cluster_frames[right]:
            skipped_frame_conflict += 1
            continue
        if enforce_hard_negative and _would_violate_hard_negative(
            uf=uf,
            left_root=left,
            right_root=right,
            hard_negative_pairs=hard_negative_pairs,
        ):
            skipped_hard_negative += 1
            continue
        merged = uf.union(left, right)
        if not merged:
            continue
        root = uf.find(left)
        other = right if root == left else left
        cluster_frames[root] = set(cluster_frames.get(left, set())) | set(cluster_frames.get(right, set()))
        cluster_frames.pop(other, None)
        selected.append(dict(row, local_graph_selected_rank=len(selected)))
        if max_accepts >= 0 and len(selected) >= int(max_accepts):
            break
    after = _metrics(
        tracklet_rows=tracklet_rows,
        tracklets=tracklets,
        track_index=track_index,
        uf=uf,
        hard_negative_pairs=hard_negative_pairs,
    )
    result = {
        "variant": variant,
        "min_d4rt": float(min_d4rt),
        "min_semantic": float(min_semantic),
        "max_visible_outside": float(max_visible_outside),
        "min_margin": float(min_margin),
        "max_accepts": int(max_accepts),
        "topk_per_src": int(topk_per_src),
        "negative_weight": float(negative_weight),
        "enforce_hard_negative": bool(enforce_hard_negative),
        "candidate_count_after_filter": len(filtered),
        "local_graph_node_count": len(tracklet_ids),
        "local_edge_count": len(filtered),
        "base_merge_count": base_merge_count,
        "refinement_merge_count": len(selected),
        "refinement_split_count": 0,
        "skipped_frame_conflict_count": skipped_frame_conflict,
        "skipped_hard_negative_count": skipped_hard_negative,
        **after,
        "base_ARI": base_metrics["ARI"],
        "ARI_change": float(after["ARI"] - base_metrics["ARI"]),
        "base_purity": base_metrics["purity"],
        "purity_change": float(after["purity"] - base_metrics["purity"]),
        "base_completeness": base_metrics["completeness"],
        "completeness_change": float(after["completeness"] - base_metrics["completeness"]),
        "base_temporal_span_mean": base_metrics["temporal_span_mean"],
        "temporal_span_change": float((after["temporal_span_mean"] or 0.0) - (base_metrics["temporal_span_mean"] or 0.0)),
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }
    return result, selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v47 Phase8 local signed graph refinement over R5 reactivation output.")
    parser.add_argument("--observation-root", default="outputs/audit/v47_observation_tables_metricfix")
    parser.add_argument("--tracklet-root", default="outputs/audit/v47_tracklets_strict_veto_A5")
    parser.add_argument("--reactivation-root", default="outputs/audit/v47_reactivation_gap2_tracklet_strict")
    parser.add_argument("--selected-pairs-file", default="reactivation_best_R5_selected_pairs.csv")
    parser.add_argument("--min-d4rt-values", default="0.97,0.90,0.80")
    parser.add_argument("--min-semantic-values", default="0.97,0.90,0.80")
    parser.add_argument("--max-visible-outside-values", default="0.45,0.75")
    parser.add_argument("--min-margin-values", default="0.00,0.05")
    parser.add_argument("--max-accept-values", default="25,50,100,-1")
    parser.add_argument("--topk-per-src-values", default="1,2")
    parser.add_argument("--negative-visible-carrier-values", default="8,16")
    parser.add_argument("--negative-visible-outside", type=float, default=1.0)
    parser.add_argument("--negative-weight-values", default="0.25,0.50")
    parser.add_argument("--output-root", default="outputs/audit/v47_local_graph_refinement_r5")
    args = parser.parse_args()

    mask_rows = read_csv(ROOT / str(args.observation_root) / "mask_observation_table.csv")
    tracklet_rows = read_csv(ROOT / str(args.tracklet_root) / "tracklet_rows.csv")
    candidates = read_csv(ROOT / str(args.reactivation_root) / "reactivation_candidate_rows.csv")
    base_pairs = read_csv(ROOT / str(args.reactivation_root) / str(args.selected_pairs_file))
    tracklets, _node_to_track = _build_tracklets(tracklet_rows, mask_rows)
    tracklet_ids, track_index = _track_index(tracklets)

    base_uf, _base_cluster_frames, base_merge_count = _init_state(
        tracklets=tracklets,
        tracklet_ids=tracklet_ids,
        track_index=track_index,
        base_pairs=base_pairs,
    )
    empty_hard_pairs: set[tuple[int, int]] = set()
    base_metrics = _metrics(
        tracklet_rows=tracklet_rows,
        tracklets=tracklets,
        track_index=track_index,
        uf=base_uf,
        hard_negative_pairs=empty_hard_pairs,
    )

    rows: list[dict[str, Any]] = []
    selected_by_signature: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for negative_visible_carriers in _parse_int_list(args.negative_visible_carrier_values):
        hard_pairs = _hard_negative_pairs(
            candidates=candidates,
            track_index=track_index,
            min_visible_carriers=negative_visible_carriers,
            min_visible_outside=float(args.negative_visible_outside),
        )
        for min_d4rt in _parse_float_list(args.min_d4rt_values):
            for min_semantic in _parse_float_list(args.min_semantic_values):
                for max_visible_outside in _parse_float_list(args.max_visible_outside_values):
                    for min_margin in _parse_float_list(args.min_margin_values):
                        for max_accepts in _parse_int_list(args.max_accept_values):
                            for topk_per_src in _parse_int_list(args.topk_per_src_values):
                                specs = [
                                    ("G1_local_positive_edges", 0.0, False),
                                    ("G3_local_hard_negative_veto", 0.0, True),
                                ]
                                for weight in _parse_float_list(args.negative_weight_values):
                                    specs.append(("G2_local_signed_soft", weight, False))
                                specs.append(("G5_dense_semantic_control", 0.0, False))
                                for variant, negative_weight, enforce_hard_negative in specs:
                                    result, selected = _evaluate(
                                        variant=variant,
                                        tracklets=tracklets,
                                        tracklet_rows=tracklet_rows,
                                        tracklet_ids=tracklet_ids,
                                        track_index=track_index,
                                        candidates=candidates,
                                        base_pairs=base_pairs,
                                        base_metrics=base_metrics,
                                        hard_negative_pairs=hard_pairs,
                                        min_d4rt=min_d4rt,
                                        min_semantic=min_semantic,
                                        max_visible_outside=max_visible_outside,
                                        min_margin=min_margin,
                                        max_accepts=max_accepts,
                                        topk_per_src=topk_per_src,
                                        negative_weight=negative_weight,
                                        enforce_hard_negative=enforce_hard_negative,
                                    )
                                    result["negative_visible_carriers"] = int(negative_visible_carriers)
                                    result["negative_visible_outside"] = float(args.negative_visible_outside)
                                    signature = (
                                        result["variant"],
                                        result["min_d4rt"],
                                        result["min_semantic"],
                                        result["max_visible_outside"],
                                        result["min_margin"],
                                        result["max_accepts"],
                                        result["topk_per_src"],
                                        result["negative_weight"],
                                        result["negative_visible_carriers"],
                                    )
                                    selected_by_signature[signature] = selected
                                    rows.append(result)

    rows.sort(
        key=lambda row: (
            parse_float(row.get("ARI_change")),
            parse_float(row.get("purity_change")),
            -parse_float(row.get("hard_negative_violation_rate")),
        ),
        reverse=True,
    )
    local_rows = [row for row in rows if row["variant"] in {"G2_local_signed_soft", "G3_local_hard_negative_veto"}]
    dense_rows = [row for row in rows if row["variant"] == "G5_dense_semantic_control"]
    best_local = max(
        local_rows,
        key=lambda row: (
            parse_float(row.get("ARI_change")),
            parse_float(row.get("purity_change")),
            -parse_float(row.get("hard_negative_violation_rate")),
        ),
        default={},
    )
    best_dense = max(
        dense_rows,
        key=lambda row: (
            parse_float(row.get("ARI_change")),
            parse_float(row.get("purity_change")),
        ),
        default={},
    )
    safe_local_rows = [
        row
        for row in local_rows
        if parse_float(row.get("ARI_change")) >= 0.015
        and parse_float(row.get("purity_change")) >= -0.005
        and parse_float(row.get("hard_negative_violation_rate")) <= 0.05
    ]
    best_safe_local = max(
        safe_local_rows,
        key=lambda row: (
            parse_float(row.get("ARI")),
            parse_float(row.get("completeness")),
            parse_float(row.get("ARI_change")),
            parse_float(row.get("purity_change")),
        ),
        default={},
    )
    recommended_local = best_safe_local or best_local
    best_any = rows[0] if rows else {}

    def signature(row: dict[str, Any]) -> tuple[Any, ...]:
        return (
            row.get("variant"),
            row.get("min_d4rt"),
            row.get("min_semantic"),
            row.get("max_visible_outside"),
            row.get("min_margin"),
            row.get("max_accepts"),
            row.get("topk_per_src"),
            row.get("negative_weight"),
            row.get("negative_visible_carriers"),
        )

    gate = {
        "ARI_gain_pass": bool(parse_float(recommended_local.get("ARI_change")) >= 0.015),
        "purity_drop_pass": bool(parse_float(recommended_local.get("purity_change")) >= -0.005),
        "hard_negative_violation_rate_pass": bool(parse_float(recommended_local.get("hard_negative_violation_rate")) <= 0.05),
        "dense_control_not_promoted": bool(
            parse_float(best_dense.get("purity_change")) < -0.005
            or parse_float(best_dense.get("ARI_change")) <= parse_float(recommended_local.get("ARI_change"))
        ),
    }
    gate["pass"] = bool(all(gate.values()))
    summary = {
        "phase": "v47_local_graph_refinement",
        "tracklet_root": str(ROOT / str(args.tracklet_root)),
        "reactivation_root": str(ROOT / str(args.reactivation_root)),
        "selected_pairs_file": str(args.selected_pairs_file),
        "base": {
            **base_metrics,
            "base_pair_count": len(base_pairs),
            "base_merge_count": base_merge_count,
        },
        "tracklet_count": len(tracklet_ids),
        "reactivation_candidate_count": len(candidates),
        "rows": len(rows),
        "best_any_row": best_any,
        "best_local_row": recommended_local,
        "best_raw_local_row": best_local,
        "best_safe_local_row": best_safe_local,
        "safe_local_row_count": len(safe_local_rows),
        "recommended_local_row": recommended_local,
        "best_dense_control_row": best_dense,
        "dense_graph_vs_local_graph_delta": parse_float(best_dense.get("ARI_change")) - parse_float(recommended_local.get("ARI_change")),
        "gate": gate,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }

    out_root = ROOT / str(args.output_root)
    write_csv(out_root / "local_graph_refinement_scan_rows.csv", rows)
    write_csv(out_root / "local_graph_refinement_best_local_selected_pairs.csv", selected_by_signature.get(signature(recommended_local), []))
    write_csv(out_root / "local_graph_refinement_best_raw_local_selected_pairs.csv", selected_by_signature.get(signature(best_local), []))
    write_csv(out_root / "local_graph_refinement_best_safe_local_selected_pairs.csv", selected_by_signature.get(signature(best_safe_local), []))
    write_csv(out_root / "local_graph_refinement_best_dense_selected_pairs.csv", selected_by_signature.get(signature(best_dense), []))
    write_json(out_root / "local_graph_refinement_summary.json", summary)
    print({"summary": str(out_root / "local_graph_refinement_summary.json"), "gate": summary["gate"]})


if __name__ == "__main__":
    main()
