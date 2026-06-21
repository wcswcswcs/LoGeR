from __future__ import annotations

import argparse
import hashlib
import math
from collections import Counter, defaultdict
from typing import Any

from stream4d_native.v47_common import (
    ROOT,
    adjusted_rand_score,
    cluster_completeness,
    cluster_purity,
    parse_float,
    parse_int,
    rank_auc,
    read_csv,
    safe_mean,
    write_csv,
    write_json,
)
from stream4d_native.v48_data_contract import utc_now


class StringUnionFind:
    def __init__(self, nodes: list[str]) -> None:
        self.parent = {node: node for node in nodes}
        self.members = {node: {node} for node in nodes}

    def find(self, node: str) -> str:
        node = str(node)
        if node not in self.parent:
            self.parent[node] = node
            self.members[node] = {node}
        parent = self.parent[node]
        if parent != node:
            self.parent[node] = self.find(parent)
        return self.parent[node]

    def can_union(
        self,
        left: str,
        right: str,
        *,
        pair_distance_norm: dict[tuple[str, str], float],
        max_root_same_frame_distance_norm: float,
    ) -> bool:
        root_l = self.find(left)
        root_r = self.find(right)
        if root_l == root_r:
            return False
        for member_l in self.members.get(root_l, {root_l}):
            for member_r in self.members.get(root_r, {root_r}):
                key = tuple(sorted([member_l, member_r]))
                distance = pair_distance_norm.get(key)
                if distance is None:
                    continue
                if distance > float(max_root_same_frame_distance_norm):
                    return False
        return True

    def union(
        self,
        left: str,
        right: str,
        *,
        pair_distance_norm: dict[tuple[str, str], float],
        max_root_same_frame_distance_norm: float,
    ) -> bool:
        if not self.can_union(
            left,
            right,
            pair_distance_norm=pair_distance_norm,
            max_root_same_frame_distance_norm=max_root_same_frame_distance_norm,
        ):
            return False
        root_l = self.find(left)
        root_r = self.find(right)
        if len(self.members[root_l]) < len(self.members[root_r]):
            root_l, root_r = root_r, root_l
        for node in self.members[root_r]:
            self.parent[node] = root_l
        self.members[root_l].update(self.members[root_r])
        del self.members[root_r]
        return True


def _parse_csv_values(spec: str) -> list[str]:
    return [item.strip() for item in str(spec).split(",") if item.strip()]


def _parse_float_values(spec: str) -> list[float]:
    return [float(item.strip()) for item in str(spec).split(",") if item.strip()]


def _component_key(scene: str, component: str) -> str:
    return f"{scene}|{component}"


def _real_component(component: str) -> bool:
    return bool(component) and not str(component).startswith("uncovered:")


def _mask_by_node(mask_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("node_id")): row for row in mask_rows}


def _component_tracks(
    mask_vote_rows: list[dict[str, Any]],
    mask_rows: list[dict[str, Any]],
) -> tuple[dict[str, dict[int, tuple[float, float]]], dict[str, dict[int, tuple[float, float, float, float]]]]:
    masks = _mask_by_node(mask_rows)
    centers_accum: dict[str, dict[int, list[tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
    boxes_accum: dict[str, dict[int, list[tuple[float, float, float, float]]]] = defaultdict(lambda: defaultdict(list))
    for row in mask_vote_rows:
        component = str(row.get("predicted_component_object_id") or "")
        if not _real_component(component):
            continue
        mask = masks.get(str(row.get("node_id")))
        if not mask:
            continue
        scene = str(row.get("scene"))
        key = _component_key(scene, component)
        frame = parse_int(row.get("frame_id"))
        x0 = parse_float(mask.get("bbox_x0"))
        y0 = parse_float(mask.get("bbox_y0"))
        x1 = parse_float(mask.get("bbox_x1"))
        y1 = parse_float(mask.get("bbox_y1"))
        centers_accum[key][frame].append(((x0 + x1) * 0.5, (y0 + y1) * 0.5))
        boxes_accum[key][frame].append((x0, y0, x1, y1))
    tracks: dict[str, dict[int, tuple[float, float]]] = {}
    boxes: dict[str, dict[int, tuple[float, float, float, float]]] = {}
    for key, by_frame in centers_accum.items():
        tracks[key] = {}
        for frame, centers in by_frame.items():
            tracks[key][frame] = (
                sum(x for x, _y in centers) / len(centers),
                sum(y for _x, y in centers) / len(centers),
            )
    for key, by_frame in boxes_accum.items():
        boxes[key] = {}
        for frame, items in by_frame.items():
            boxes[key][frame] = (
                sum(item[0] for item in items) / len(items),
                sum(item[1] for item in items) / len(items),
                sum(item[2] for item in items) / len(items),
                sum(item[3] for item in items) / len(items),
            )
    return tracks, boxes


def _shuffle_track(track: dict[int, tuple[float, float]], key: str) -> dict[int, tuple[float, float]]:
    frames = sorted(track)
    if len(frames) <= 1:
        return dict(track)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    shift = int(digest[:8], 16) % len(frames)
    if shift == 0:
        shift = 1
    values = [track[frame] for frame in frames]
    shifted = values[shift:] + values[:shift]
    return {frame: value for frame, value in zip(frames, shifted)}


def _mean_motion(track: dict[int, tuple[float, float]], frames: list[int]) -> float:
    if len(frames) < 2:
        return 0.0
    deltas: list[float] = []
    for a, b in zip(frames, frames[1:]):
        ax, ay = track[a]
        bx, by = track[b]
        deltas.append(math.hypot(bx - ax, by - ay))
    return safe_mean(deltas) or 0.0


def _pair_features(
    left_track: dict[int, tuple[float, float]],
    right_track: dict[int, tuple[float, float]],
    left_boxes: dict[int, tuple[float, float, float, float]],
    right_boxes: dict[int, tuple[float, float, float, float]],
    *,
    sigma_motion: float,
    sigma_layout: float,
    sigma_distance_norm: float,
) -> dict[str, Any] | None:
    common = sorted(set(left_track) & set(right_track))
    if len(common) < 2:
        return None
    motion_diffs: list[float] = []
    distances: list[float] = []
    distance_norms: list[float] = []
    for a, b in zip(common, common[1:]):
        la, lb = left_track[a], left_track[b]
        ra, rb = right_track[a], right_track[b]
        dl = (lb[0] - la[0], lb[1] - la[1])
        dr = (rb[0] - ra[0], rb[1] - ra[1])
        motion_diffs.append(math.hypot(dl[0] - dr[0], dl[1] - dr[1]))
    for frame in common:
        lx, ly = left_track[frame]
        rx, ry = right_track[frame]
        distance = math.hypot(lx - rx, ly - ry)
        left_box = left_boxes[frame]
        right_box = right_boxes[frame]
        left_diag = math.hypot(left_box[2] - left_box[0], left_box[3] - left_box[1])
        right_diag = math.hypot(right_box[2] - right_box[0], right_box[3] - right_box[1])
        mean_diag = max((left_diag + right_diag) * 0.5, 1e-6)
        distances.append(distance)
        distance_norms.append(distance / mean_diag)
    mean_motion_delta = safe_mean(motion_diffs) or 0.0
    mean_distance = safe_mean(distances) or 0.0
    mean_distance_norm = safe_mean(distance_norms) or 0.0
    layout_std = math.sqrt(safe_mean((value - mean_distance) ** 2 for value in distances) or 0.0)
    motion_score = math.exp(-mean_motion_delta / max(float(sigma_motion), 1e-6))
    layout_score = math.exp(-layout_std / max(float(sigma_layout), 1e-6))
    proximity_score = math.exp(-mean_distance_norm / max(float(sigma_distance_norm), 1e-6))
    real_score = 0.45 * motion_score + 0.25 * layout_score + 0.30 * proximity_score
    no_temporal_score = 0.55 * layout_score + 0.45 * proximity_score
    return {
        "overlap_frame_count": len(common),
        "mean_motion_delta": mean_motion_delta,
        "left_mean_motion": _mean_motion(left_track, common),
        "right_mean_motion": _mean_motion(right_track, common),
        "layout_std": layout_std,
        "mean_same_frame_distance": mean_distance,
        "mean_same_frame_distance_norm": mean_distance_norm,
        "min_same_frame_distance_norm": min(distance_norms),
        "max_same_frame_distance_norm": max(distance_norms),
        "motion_score": motion_score,
        "layout_score": layout_score,
        "proximity_score": proximity_score,
        "real_score": real_score,
        "no_temporal_score": no_temporal_score,
    }


def _build_pair_rows(
    pair_rows: list[dict[str, Any]],
    tracks: dict[str, dict[int, tuple[float, float]]],
    boxes: dict[str, dict[int, tuple[float, float, float, float]]],
    *,
    sigma_motion: float,
    sigma_layout: float,
    sigma_distance_norm: float,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in pair_rows:
        scene = str(row.get("scene"))
        left = _component_key(scene, str(row.get("component_left")))
        right = _component_key(scene, str(row.get("component_right")))
        if left not in tracks or right not in tracks:
            continue
        real = _pair_features(
            tracks[left],
            tracks[right],
            boxes[left],
            boxes[right],
            sigma_motion=sigma_motion,
            sigma_layout=sigma_layout,
            sigma_distance_norm=sigma_distance_norm,
        )
        if real is None:
            continue
        shuffled_right_track = _shuffle_track(tracks[right], right)
        shuffled = _pair_features(
            tracks[left],
            shuffled_right_track,
            boxes[left],
            boxes[right],
            sigma_motion=sigma_motion,
            sigma_layout=sigma_layout,
            sigma_distance_norm=sigma_distance_norm,
        )
        shuffled_score = shuffled["real_score"] if shuffled else None
        out.append(
            {
                "component_left": left,
                "component_right": right,
                "scene": scene,
                "diagnostic_same_gt": parse_int(row.get("diagnostic_same_gt_edge_count")) > 0,
                "edge_count": row.get("edge_count"),
                "same_frame_conflict": row.get("same_frame_conflict"),
                "A5_d4rt_semantic_confirmation": row.get("max_A5_d4rt_semantic_confirmation"),
                "A8_no_temporal_control": row.get("max_A8_no_temporal_control"),
                "A7_shuffled_D4RT": row.get("max_A7_shuffled_D4RT"),
                "separation_real_score": real["real_score"],
                "separation_no_temporal_score": real["no_temporal_score"],
                "separation_shuffled_score": shuffled_score,
                **real,
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic_labels": True,
            }
        )
    return out


def _evaluate(mask_vote_rows: list[dict[str, Any]], uf: StringUnionFind) -> dict[str, Any]:
    true_labels: list[str] = []
    pred_labels: list[str] = []
    scene_true: dict[str, list[str]] = defaultdict(list)
    scene_pred: dict[str, list[str]] = defaultdict(list)
    frames: dict[str, set[int]] = defaultdict(set)
    object_gt: dict[str, Counter[str]] = defaultdict(Counter)
    unknown_count = 0
    for row in mask_vote_rows:
        scene = str(row.get("scene"))
        component = str(row.get("predicted_component_object_id") or "")
        if _real_component(component):
            pred = uf.find(_component_key(scene, component))
        else:
            pred = f"{scene}|uncovered:{row.get('mask_observation_id') or row.get('node_id')}"
            unknown_count += 1
        gt = str(row.get("diagnostic_gt_instance") or "")
        frames[pred].add(parse_int(row.get("frame_id")))
        if gt:
            true_labels.append(gt)
            pred_labels.append(pred)
            object_gt[pred][gt] += 1
            scene_true[scene].append(gt)
            scene_pred[scene].append(pred)
    conflict_objects = sum(1 for counts in object_gt.values() if len(counts) > 1)
    return {
        "ARI": adjusted_rand_score(true_labels, pred_labels),
        "purity": cluster_purity(true_labels, pred_labels),
        "completeness": cluster_completeness(true_labels, pred_labels),
        "selected_candidate_count": len(frames),
        "selected_object_count": len(frames),
        "temporal_span_mean": safe_mean(len(value) for value in frames.values()),
        "mean_predictions_per_scene": safe_mean(len(set(scene_pred[scene])) for scene in sorted(scene_pred)),
        "conflict_rate": float(conflict_objects / max(len(object_gt), 1)),
        "unknown_tube_ratio": float(unknown_count / max(len(mask_vote_rows), 1)),
        "duplicate_rate": 0.0,
        "birth_from_d4rt_tube_count": 0,
        "maskless_object_count": 0,
        "scene0081_ARI": adjusted_rand_score(scene_true["scene0081_01"], scene_pred["scene0081_01"])
        if scene_true.get("scene0081_01")
        else None,
        "scene0011_purity": cluster_purity(scene_true["scene0011_00"], scene_pred["scene0011_00"])
        if scene_true.get("scene0011_00")
        else None,
        "scene0050_purity": cluster_purity(scene_true["scene0050_00"], scene_pred["scene0050_00"])
        if scene_true.get("scene0050_00")
        else None,
        "scene0591_completeness": cluster_completeness(scene_true["scene0591_00"], scene_pred["scene0591_00"])
        if scene_true.get("scene0591_00")
        else None,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }


def _run_variant(
    *,
    mask_vote_rows: list[dict[str, Any]],
    seeds: list[str],
    pair_rows: list[dict[str, Any]],
    pair_distance_norm: dict[tuple[str, str], float],
    score_key: str,
    min_score: float,
    min_a5: float,
    max_same_frame_distance_norm: float,
    max_root_same_frame_distance_norm: float,
    max_selected_pairs: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    uf = StringUnionFind(seeds)
    selected: list[dict[str, Any]] = []
    rejected = Counter()
    candidates: list[dict[str, Any]] = []
    for row in pair_rows:
        score = parse_float(row.get(score_key))
        if score < float(min_score):
            continue
        if parse_float(row.get("A5_d4rt_semantic_confirmation")) < float(min_a5):
            continue
        if parse_float(row.get("mean_same_frame_distance_norm")) > float(max_same_frame_distance_norm):
            continue
        candidates.append(dict(row, selected_score=score, score_key=score_key))
    candidates.sort(
        key=lambda row: (
            parse_float(row.get("selected_score")),
            parse_float(row.get("A5_d4rt_semantic_confirmation")),
            parse_int(row.get("edge_count")),
        ),
        reverse=True,
    )
    for row in candidates:
        left = str(row.get("component_left"))
        right = str(row.get("component_right"))
        if uf.find(left) == uf.find(right):
            rejected["already_same_root"] += 1
            continue
        if not uf.can_union(
            left,
            right,
            pair_distance_norm=pair_distance_norm,
            max_root_same_frame_distance_norm=max_root_same_frame_distance_norm,
        ):
            rejected["root_same_frame_separation"] += 1
            continue
        if not uf.union(
            left,
            right,
            pair_distance_norm=pair_distance_norm,
            max_root_same_frame_distance_norm=max_root_same_frame_distance_norm,
        ):
            rejected["union_noop"] += 1
            continue
        selected.append(
            {
                **row,
                "merge_index": len(selected),
                "min_score": min_score,
                "min_a5": min_a5,
                "max_same_frame_distance_norm": max_same_frame_distance_norm,
                "max_root_same_frame_distance_norm": max_root_same_frame_distance_norm,
            }
        )
        if len(selected) >= int(max_selected_pairs):
            rejected["max_selected_pairs"] += 1
            break
    metrics = _evaluate(mask_vote_rows, uf)
    row = {
        "variant": f"D_repair_{score_key}_score{min_score}_a5{min_a5}_sep{max_same_frame_distance_norm}_rootsep{max_root_same_frame_distance_norm}",
        "score_key": score_key,
        "min_score": min_score,
        "min_A5_d4rt_semantic_confirmation": min_a5,
        "max_same_frame_distance_norm": max_same_frame_distance_norm,
        "max_root_same_frame_distance_norm": max_root_same_frame_distance_norm,
        "candidate_pair_count": len(candidates),
        "selected_pair_count": len(selected),
        "rejected": dict(rejected),
        **metrics,
    }
    return row, selected


def _delta_if_present(left: dict[str, Any], right: dict[str, Any]) -> float | None:
    if not left or not right or left.get("ARI") in (None, "") or right.get("ARI") in (None, ""):
        return None
    return parse_float(left.get("ARI")) - parse_float(right.get("ARI"))


def _row_signature(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("score_key"),
        row.get("min_score"),
        row.get("min_A5_d4rt_semantic_confirmation"),
        row.get("max_same_frame_distance_norm"),
        row.get("max_root_same_frame_distance_norm"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair v48 Scheme D with same-frame separation and relative-layout common-fate guards.")
    parser.add_argument("--mask-observation-table", default="outputs/audit/v47_observation_tables_metricfix/mask_observation_table.csv")
    parser.add_argument("--mask-vote-rows", default="outputs/audit/v47_carrier_supertrack_union_32_fine_metricfix/carrier_supertrack_mask_vote_rows.csv")
    parser.add_argument("--pair-stats", default="outputs/audit/v47_component_constrained_merge_union32_gap2_narrow/component_constrained_merge_pair_stats.csv")
    parser.add_argument("--score-keys", default="separation_real_score,separation_no_temporal_score,separation_shuffled_score")
    parser.add_argument("--real-score-keys", default="separation_real_score")
    parser.add_argument("--no-temporal-score-keys", default="separation_no_temporal_score")
    parser.add_argument("--shuffled-score-keys", default="separation_shuffled_score")
    parser.add_argument("--min-scores", default="0.75,0.85,0.90,0.95")
    parser.add_argument("--min-a5-values", default="0.3,0.4,0.5,0.6,0.7,0.8")
    parser.add_argument("--max-same-frame-distance-norm-values", default="0.5,1.0,1.5,2.0")
    parser.add_argument("--max-root-same-frame-distance-norm-values", default="0.75,1.0,1.5,2.0")
    parser.add_argument("--max-selected-pairs", type=int, default=120)
    parser.add_argument("--sigma-motion", type=float, default=40.0)
    parser.add_argument("--sigma-layout", type=float, default=80.0)
    parser.add_argument("--sigma-distance-norm", type=float, default=1.0)
    parser.add_argument("--output-root", default="outputs/audit/v48_common_fate_separation_repair")
    args = parser.parse_args()

    mask_rows = read_csv(ROOT / str(args.mask_observation_table))
    mask_vote_rows = read_csv(ROOT / str(args.mask_vote_rows))
    component_pair_rows = read_csv(ROOT / str(args.pair_stats))
    tracks, boxes = _component_tracks(mask_vote_rows, mask_rows)
    pair_rows = _build_pair_rows(
        component_pair_rows,
        tracks,
        boxes,
        sigma_motion=float(args.sigma_motion),
        sigma_layout=float(args.sigma_layout),
        sigma_distance_norm=float(args.sigma_distance_norm),
    )
    pair_distance_norm = {
        tuple(sorted([str(row["component_left"]), str(row["component_right"])])): parse_float(row.get("mean_same_frame_distance_norm"))
        for row in pair_rows
    }
    seeds = sorted(
        {
            _component_key(str(row.get("scene")), str(row.get("predicted_component_object_id")))
            for row in mask_vote_rows
            if _real_component(str(row.get("predicted_component_object_id") or ""))
        }
    )
    base = _evaluate(mask_vote_rows, StringUnionFind(seeds))
    score_keys = _parse_csv_values(args.score_keys)
    real_keys = set(_parse_csv_values(args.real_score_keys))
    no_temporal_keys = set(_parse_csv_values(args.no_temporal_score_keys))
    shuffled_keys = set(_parse_csv_values(args.shuffled_score_keys))
    min_scores = _parse_float_values(args.min_scores)
    min_a5_values = _parse_float_values(args.min_a5_values)
    max_same_frame_distance_norm_values = _parse_float_values(args.max_same_frame_distance_norm_values)
    max_root_same_frame_distance_norm_values = _parse_float_values(args.max_root_same_frame_distance_norm_values)

    labels = [bool(row.get("diagnostic_same_gt")) for row in pair_rows]
    auc_by_score = {
        score_key: rank_auc(labels, [parse_float(row.get(score_key)) for row in pair_rows])
        for score_key in score_keys
    }
    precision_top100_by_score = {}
    for score_key in score_keys:
        ranked = sorted(pair_rows, key=lambda row: parse_float(row.get(score_key)), reverse=True)
        top = ranked[: min(100, len(ranked))]
        precision_top100_by_score[score_key] = (sum(1 for row in top if row.get("diagnostic_same_gt")) / len(top)) if top else None

    rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    selected_by_signature: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for score_key in score_keys:
        for min_score in min_scores:
            for min_a5 in min_a5_values:
                for max_same_frame_distance_norm in max_same_frame_distance_norm_values:
                    for max_root_same_frame_distance_norm in max_root_same_frame_distance_norm_values:
                        row, selected = _run_variant(
                            mask_vote_rows=mask_vote_rows,
                            seeds=seeds,
                            pair_rows=pair_rows,
                            pair_distance_norm=pair_distance_norm,
                            score_key=score_key,
                            min_score=min_score,
                            min_a5=min_a5,
                            max_same_frame_distance_norm=max_same_frame_distance_norm,
                            max_root_same_frame_distance_norm=max_root_same_frame_distance_norm,
                            max_selected_pairs=int(args.max_selected_pairs),
                        )
                        row.update(
                            {
                                "motion_pair_AUC": auc_by_score.get(score_key),
                                "common_fate_precision@top100": precision_top100_by_score.get(score_key),
                                "delta_ARI_vs_nomotion": parse_float(row.get("ARI")) - parse_float(base.get("ARI")),
                                "delta_completeness_vs_nomotion": parse_float(row.get("completeness")) - parse_float(base.get("completeness")),
                                "purity_drop_vs_nomotion": parse_float(base.get("purity")) - parse_float(row.get("purity")),
                            }
                        )
                        rows.append(row)
                        selected_by_signature[_row_signature(row)] = selected
                        for item in selected:
                            selected_rows.append({"variant": row["variant"], **item})
    rows.sort(key=lambda row: (parse_float(row.get("ARI")), parse_float(row.get("completeness")), parse_float(row.get("purity"))), reverse=True)
    real_rows = [row for row in rows if row["score_key"] in real_keys]
    no_temporal_rows = [row for row in rows if row["score_key"] in no_temporal_keys]
    shuffled_rows = [row for row in rows if row["score_key"] in shuffled_keys]
    best_real = real_rows[0] if real_rows else {}
    best_no_temporal = no_temporal_rows[0] if no_temporal_rows else {}
    best_shuffled = shuffled_rows[0] if shuffled_rows else {}
    real_minus_no_temporal = _delta_if_present(best_real, best_no_temporal)
    real_minus_shuffled = _delta_if_present(best_real, best_shuffled)
    real_auc = auc_by_score.get("separation_real_score")
    no_temporal_auc = auc_by_score.get("separation_no_temporal_score")
    shuffled_auc = auc_by_score.get("separation_shuffled_score")
    gate = {
        "motion_pair_AUC_pass": real_auc is not None and real_auc >= 0.80,
        "top100_precision_pass": precision_top100_by_score.get("separation_real_score") is not None
        and precision_top100_by_score["separation_real_score"] >= 0.80,
        "delta_ARI_pass": parse_float(best_real.get("delta_ARI_vs_nomotion")) >= 0.02,
        "delta_completeness_pass": parse_float(best_real.get("delta_completeness_vs_nomotion")) >= 0.04,
        "purity_drop_pass": parse_float(best_real.get("purity_drop_vs_nomotion")) <= 0.005,
        "real_minus_no_temporal_motion_AUC_pass": real_auc is not None and no_temporal_auc is not None and real_auc - no_temporal_auc >= 0.08,
        "real_minus_shuffled_motion_AUC_pass": real_auc is not None and shuffled_auc is not None and real_auc - shuffled_auc >= 0.08,
        "real_minus_no_temporal_ARI_pass": real_minus_no_temporal is not None and real_minus_no_temporal >= 0.10,
        "real_minus_shuffled_ARI_pass": real_minus_shuffled is not None and real_minus_shuffled >= 0.10,
    }
    gate["pass"] = bool(
        (gate["motion_pair_AUC_pass"] or gate["top100_precision_pass"])
        and gate["delta_ARI_pass"]
        and gate["delta_completeness_pass"]
        and gate["purity_drop_pass"]
        and gate["real_minus_no_temporal_motion_AUC_pass"]
    )
    summary = {
        "phase": "v48_common_fate_separation_repair",
        "created_at": utc_now(),
        "repair_basis": "Scheme D repair: for static-object common-fate ambiguity, add same-frame separation and relative-layout proximity guards.",
        "base_metrics": base,
        "pair_count": len(pair_rows),
        "row_count": len(rows),
        "auc_by_score": auc_by_score,
        "precision_top100_by_score": precision_top100_by_score,
        "best_real_row": best_real,
        "best_no_temporal_row": best_no_temporal,
        "best_shuffled_row": best_shuffled,
        "real_minus_no_temporal_ARI": real_minus_no_temporal,
        "real_minus_shuffled_ARI": real_minus_shuffled,
        "real_minus_no_temporal_motion_AUC": real_auc - no_temporal_auc if real_auc is not None and no_temporal_auc is not None else None,
        "real_minus_shuffled_motion_AUC": real_auc - shuffled_auc if real_auc is not None and shuffled_auc is not None else None,
        "gate": gate,
        "failure_label": None if gate["pass"] else "NO_GO_COMMON_FATE_SEPARATION_REPAIR",
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }
    out = ROOT / str(args.output_root)
    write_json(out / "common_fate_separation_repair_summary.json", summary)
    write_csv(out / "common_fate_separation_repair_rows.csv", rows)
    write_csv(out / "common_fate_separation_repair_pair_rows.csv", pair_rows)
    write_csv(out / "common_fate_separation_repair_selected_pairs.csv", selected_rows)
    for name, row in [
        ("best_real", best_real),
        ("best_no_temporal", best_no_temporal),
        ("best_shuffled", best_shuffled),
    ]:
        if row:
            write_csv(out / f"common_fate_separation_repair_{name}_selected_pairs.csv", selected_by_signature.get(_row_signature(row), []))
    print({"summary": str(out / "common_fate_separation_repair_summary.json"), "gate": gate, "failure_label": summary["failure_label"]})


if __name__ == "__main__":
    main()
