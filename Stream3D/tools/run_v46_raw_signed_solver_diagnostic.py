from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _parse_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _parse_int(value: Any, default: int = 0) -> int:
    if value is None or value == "":
        return int(default)
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _comb2(n: int) -> float:
    return float(n * (n - 1) / 2)


def _adjusted_rand_score(true_labels: list[str], pred_labels: list[str]) -> float:
    n = len(true_labels)
    if n <= 1:
        return 1.0
    table: Counter[tuple[str, str]] = Counter(zip(true_labels, pred_labels))
    true_counts: Counter[str] = Counter(true_labels)
    pred_counts: Counter[str] = Counter(pred_labels)
    sum_comb = sum(_comb2(v) for v in table.values())
    sum_true = sum(_comb2(v) for v in true_counts.values())
    sum_pred = sum(_comb2(v) for v in pred_counts.values())
    total = _comb2(n)
    if total <= 0:
        return 1.0
    expected = sum_true * sum_pred / total
    max_index = 0.5 * (sum_true + sum_pred)
    denom = max_index - expected
    if abs(denom) <= 1e-12:
        return 0.0
    return float((sum_comb - expected) / denom)


def _cluster_purity(true_labels: list[str], pred_labels: list[str]) -> float:
    clusters: dict[str, Counter[str]] = defaultdict(Counter)
    for true, pred in zip(true_labels, pred_labels):
        clusters[pred][true] += 1
    if not true_labels:
        return 0.0
    return float(sum(max(counts.values()) for counts in clusters.values()) / len(true_labels))


def _cluster_completeness(true_labels: list[str], pred_labels: list[str]) -> float:
    gt_groups: dict[str, Counter[str]] = defaultdict(Counter)
    for true, pred in zip(true_labels, pred_labels):
        gt_groups[true][pred] += 1
    if not true_labels:
        return 0.0
    return float(sum(max(counts.values()) for counts in gt_groups.values()) / len(true_labels))


def _pairwise_metrics(true_labels: list[str], pred_labels: list[str]) -> tuple[float | None, float | None]:
    tp = fp = fn = 0
    n = len(true_labels)
    for i in range(n):
        for j in range(i + 1, n):
            same_true = true_labels[i] == true_labels[j]
            same_pred = pred_labels[i] == pred_labels[j]
            if same_true and same_pred:
                tp += 1
            elif (not same_true) and same_pred:
                fp += 1
            elif same_true and (not same_pred):
                fn += 1
    precision = None if tp + fp == 0 else float(tp / (tp + fp))
    recall = None if tp + fn == 0 else float(tp / (tp + fn))
    return precision, recall


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _spec_name(semantic_threshold: float, p4_max: float) -> str:
    sem = f"{semantic_threshold:g}".replace(".", "p")
    p4 = f"{p4_max:g}".replace(".", "p")
    return f"derived_semantic_le_{sem}_p4lt_{p4}"


def _parse_derived_negative_specs(raw: str) -> dict[str, tuple[float, float]]:
    specs: dict[str, tuple[float, float]] = {}
    for item in str(raw or "").split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) == 2:
            semantic_threshold = float(parts[0])
            p4_max = float(parts[1])
            name = _spec_name(semantic_threshold, p4_max)
        elif len(parts) == 3:
            name = parts[0]
            semantic_threshold = float(parts[1])
            p4_max = float(parts[2])
        else:
            raise ValueError(f"invalid derived negative spec: {item}")
        specs[name] = (semantic_threshold, p4_max)
    return specs


def _is_negative_row(row: dict[str, Any], negative_key: str, derived_specs: dict[str, tuple[float, float]]) -> bool:
    if negative_key in derived_specs:
        semantic_threshold, p4_max = derived_specs[negative_key]
        return bool(
            _parse_float(row.get("P6_feature_only")) <= float(semantic_threshold)
            and _parse_float(row.get("P4_vc_q_temporal")) < float(p4_max)
        )
    return _parse_bool(row.get(negative_key))


class UnionFind:
    def __init__(self, node_ids: list[int]) -> None:
        self.parent = {node_id: node_id for node_id in node_ids}
        self.members = {node_id: {node_id} for node_id in node_ids}

    def find(self, node_id: int) -> int:
        parent = self.parent[node_id]
        if parent != node_id:
            self.parent[node_id] = self.find(parent)
        return self.parent[node_id]

    def component_members(self, node_id: int) -> set[int]:
        return self.members[self.find(node_id)]

    def union(self, left: int, right: int) -> bool:
        root_l = self.find(left)
        root_r = self.find(right)
        if root_l == root_r:
            return False
        if len(self.members[root_l]) < len(self.members[root_r]):
            root_l, root_r = root_r, root_l
        for node_id in self.members[root_r]:
            self.parent[node_id] = root_l
        self.members[root_l].update(self.members[root_r])
        del self.members[root_r]
        return True


def _read_edge_rows(input_root: Path) -> list[dict[str, Any]]:
    path = input_root / "raw_visual_semantic_edge_rows.csv"
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _scene_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[str(row["scene"])].append(row)
    return dict(out)


def _node_info(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    nodes: dict[int, dict[str, Any]] = {}
    for row in rows:
        for prefix in ["left", "right"]:
            node_id = _parse_int(row[f"{prefix}_node_id"])
            if node_id in nodes:
                continue
            nodes[node_id] = {
                "node_id": node_id,
                "gt": str(row.get(f"{prefix}_gt", "")),
                "frame_id": _parse_int(row.get(f"{prefix}_frame_id")),
                "frame_rank": _parse_int(row.get(f"{prefix}_frame_rank")),
                "mask_id": _parse_int(row.get(f"{prefix}_mask_id")),
            }
    return nodes


def _pair_key(left: int, right: int) -> tuple[int, int]:
    return (left, right) if left <= right else (right, left)


def _would_violate(uf: UnionFind, left: int, right: int, hard_negative_pairs: set[tuple[int, int]]) -> bool:
    root_l = uf.find(left)
    root_r = uf.find(right)
    if root_l == root_r:
        return False
    for node_l in uf.members[root_l]:
        for node_r in uf.members[root_r]:
            if _pair_key(node_l, node_r) in hard_negative_pairs:
                return True
    return False


def _component_negative_conflicts(
    uf: UnionFind, left: int, right: int, hard_negative_pairs: set[tuple[int, int]]
) -> tuple[int, int, float]:
    root_l = uf.find(left)
    root_r = uf.find(right)
    if root_l == root_r:
        return 0, 0, 0.0
    members_l = uf.members[root_l]
    members_r = uf.members[root_r]
    cross_count = len(members_l) * len(members_r)
    conflict_count = 0
    for node_l in members_l:
        for node_r in members_r:
            if _pair_key(node_l, node_r) in hard_negative_pairs:
                conflict_count += 1
    ratio = float(conflict_count / cross_count) if cross_count else 0.0
    return conflict_count, cross_count, ratio


def _cluster_labels(uf: UnionFind, node_ids: list[int]) -> list[str]:
    root_to_label: dict[int, str] = {}
    labels: list[str] = []
    for node_id in node_ids:
        root = uf.find(node_id)
        if root not in root_to_label:
            root_to_label[root] = f"c{len(root_to_label):04d}"
        labels.append(root_to_label[root])
    return labels


def _p90(values: list[int]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(math.ceil(0.9 * len(ordered))) - 1)
    return float(ordered[idx])


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _evaluate_clusters(
    *,
    input_root: Path,
    scene: str,
    rows: list[dict[str, Any]],
    node_infos: dict[int, dict[str, Any]],
    uf: UnionFind,
    positive_key: str,
    positive_threshold: float,
    negative_key: str,
    negative_mode: str,
    negative_weight: float | None,
    hard_negative_precision: float | None,
    hard_negative_false_same_gt_count: int,
    hard_negative_true_diff_gt_count: int,
    solver_variant: str,
    accepted_merge_count: int,
    rejected_negative_veto_count: int,
    skipped_positive_hard_negative_count: int,
    hard_negative_pairs: set[tuple[int, int]],
    positive_candidate_count: int,
) -> dict[str, Any]:
    node_ids = sorted(node_infos)
    true_labels = [str(node_infos[node_id]["gt"]) for node_id in node_ids]
    pred_labels = _cluster_labels(uf, node_ids)
    precision, recall = _pairwise_metrics(true_labels, pred_labels)
    cluster_sizes = sorted((len(members) for members in uf.members.values()), reverse=True)
    temporal_spans = []
    for members in uf.members.values():
        ranks = [int(node_infos[node_id]["frame_rank"]) for node_id in members]
        temporal_spans.append(float(max(ranks) - min(ranks)) if ranks else 0.0)
    hard_negative_violation_count = 0
    for left, right in hard_negative_pairs:
        if uf.find(left) == uf.find(right):
            hard_negative_violation_count += 1
    positive_inside_score_sum = 0.0
    positive_cut_score_sum = 0.0
    for row in rows:
        score = _parse_float(row.get(positive_key))
        if score < positive_threshold:
            continue
        left = _parse_int(row["left_node_id"])
        right = _parse_int(row["right_node_id"])
        if uf.find(left) == uf.find(right):
            positive_inside_score_sum += score
        else:
            positive_cut_score_sum += score
    return {
        "input_root": str(input_root),
        "input_root_name": input_root.name,
        "scene": scene,
        "solver_variant": solver_variant,
        "positive_key": positive_key,
        "positive_threshold": positive_threshold,
        "negative_key": negative_key,
        "negative_mode": negative_mode,
        "negative_weight": negative_weight,
        "node_count": len(node_ids),
        "edge_row_count": len(rows),
        "positive_candidate_count": positive_candidate_count,
        "accepted_merge_count": accepted_merge_count,
        "rejected_negative_veto_count": rejected_negative_veto_count,
        "skipped_positive_hard_negative_count": skipped_positive_hard_negative_count,
        "cluster_count": len(uf.members),
        "largest_cluster_size": cluster_sizes[0] if cluster_sizes else 0,
        "cluster_size_p90": _p90(cluster_sizes),
        "temporal_span_mean": _mean(temporal_spans),
        "birth_from_d4rt_tube_count": 0,
        "diagnostic_subset_purity": _cluster_purity(true_labels, pred_labels),
        "diagnostic_subset_completeness": _cluster_completeness(true_labels, pred_labels),
        "diagnostic_subset_ari": _adjusted_rand_score(true_labels, pred_labels),
        "diagnostic_pairwise_precision": precision,
        "diagnostic_pairwise_recall": recall,
        "hard_negative_count": len(hard_negative_pairs),
        "hard_negative_precision": hard_negative_precision,
        "hard_negative_false_same_gt_count": hard_negative_false_same_gt_count,
        "hard_negative_true_diff_gt_count": hard_negative_true_diff_gt_count,
        "hard_negative_violation_count": hard_negative_violation_count,
        "hard_negative_violation_rate": (
            float(hard_negative_violation_count / len(hard_negative_pairs)) if hard_negative_pairs else 0.0
        ),
        "positive_inside_score_sum": positive_inside_score_sum,
        "positive_cut_score_sum": positive_cut_score_sum,
        "diagnostic_only": True,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }


def _solve_scene(
    *,
    input_root: Path,
    scene: str,
    rows: list[dict[str, Any]],
    positive_key: str,
    positive_threshold: float,
    negative_key: str,
    negative_mode: str,
    negative_weight: float | None,
    derived_specs: dict[str, tuple[float, float]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    node_infos = _node_info(rows)
    node_ids = sorted(node_infos)
    hard_negative_rows = [
        row
        for row in rows
        if negative_key != "none" and _is_negative_row(row, negative_key, derived_specs)
    ]
    hard_negative_pairs = {
        _pair_key(_parse_int(row["left_node_id"]), _parse_int(row["right_node_id"]))
        for row in hard_negative_rows
    }
    hard_negative_false_same_gt_count = int(sum(1 for row in hard_negative_rows if _parse_bool(row.get("diagnostic_same_gt"))))
    hard_negative_true_diff_gt_count = int(len(hard_negative_rows) - hard_negative_false_same_gt_count)
    hard_negative_precision = (
        None if not hard_negative_rows else float(hard_negative_true_diff_gt_count / len(hard_negative_rows))
    )
    if negative_key == "none":
        solver_variant = "S0_positive_cc"
    elif negative_mode == "soft_penalty":
        solver_variant = "S4_signed_soft_penalty_cc"
    elif negative_mode == "hard_veto_expand":
        solver_variant = "S5_signed_hard_veto_expand_cc"
    else:
        solver_variant = "S3_signed_hard_veto_cc"
    uf = UnionFind(node_ids)
    candidate_rows = [
        row
        for row in rows
        if _parse_float(row.get(positive_key)) >= float(positive_threshold)
    ]
    candidate_rows.sort(key=lambda row: _parse_float(row.get(positive_key)), reverse=True)
    accepted_merge_count = 0
    rejected_negative_veto_count = 0
    skipped_positive_hard_negative_count = 0
    trace_rows: list[dict[str, Any]] = []
    for row in candidate_rows:
        left = _parse_int(row["left_node_id"])
        right = _parse_int(row["right_node_id"])
        score = _parse_float(row.get(positive_key))
        pair = _pair_key(left, right)
        decision = "already_same_component"
        conflict_count = 0
        cross_count = 0
        conflict_ratio = 0.0
        decision_score = score
        if negative_key != "none" and uf.find(left) != uf.find(right):
            conflict_count, cross_count, conflict_ratio = _component_negative_conflicts(uf, left, right, hard_negative_pairs)
            if negative_mode == "soft_penalty":
                decision_score = float(score - float(negative_weight or 0.0) * conflict_ratio)
        if negative_key != "none" and negative_mode in {"hard_veto", "hard_veto_expand"} and pair in hard_negative_pairs:
            skipped_positive_hard_negative_count += 1
            decision = "skip_pair_is_hard_negative"
        elif uf.find(left) != uf.find(right):
            if negative_key != "none" and negative_mode in {"hard_veto", "hard_veto_expand"} and _would_violate(uf, left, right, hard_negative_pairs):
                rejected_negative_veto_count += 1
                decision = "reject_component_hard_negative"
            elif negative_key != "none" and negative_mode == "soft_penalty" and decision_score < float(positive_threshold):
                rejected_negative_veto_count += 1
                decision = "reject_soft_negative_penalty"
            else:
                uf.union(left, right)
                accepted_merge_count += 1
                decision = "accept_merge"
        trace_rows.append(
            {
                "input_root": str(input_root),
                "input_root_name": input_root.name,
                "scene": scene,
                "solver_variant": solver_variant,
                "positive_key": positive_key,
                "positive_threshold": positive_threshold,
                "negative_key": negative_key,
                "negative_mode": negative_mode,
                "negative_weight": negative_weight,
                "left_node_id": left,
                "right_node_id": right,
                "score": score,
                "negative_conflict_count": conflict_count,
                "negative_cross_pair_count": cross_count,
                "negative_conflict_ratio": conflict_ratio,
                "decision_score": decision_score,
                "diagnostic_same_gt": _parse_bool(row.get("diagnostic_same_gt")),
                "decision": decision,
            }
        )
    if negative_mode == "hard_veto_expand" and negative_key != "none":
        max_conflict_ratio = float(negative_weight or 0.0)
        for row in candidate_rows:
            left = _parse_int(row["left_node_id"])
            right = _parse_int(row["right_node_id"])
            if uf.find(left) == uf.find(right):
                continue
            score = _parse_float(row.get(positive_key))
            conflict_count, cross_count, conflict_ratio = _component_negative_conflicts(uf, left, right, hard_negative_pairs)
            if conflict_ratio <= max_conflict_ratio:
                uf.union(left, right)
                accepted_merge_count += 1
                decision = "expand_accept_merge"
            else:
                rejected_negative_veto_count += 1
                decision = "expand_reject_conflict_ratio"
            trace_rows.append(
                {
                    "input_root": str(input_root),
                    "input_root_name": input_root.name,
                    "scene": scene,
                    "solver_variant": solver_variant,
                    "positive_key": positive_key,
                    "positive_threshold": positive_threshold,
                    "negative_key": negative_key,
                    "negative_mode": negative_mode,
                    "negative_weight": negative_weight,
                    "left_node_id": left,
                    "right_node_id": right,
                    "score": score,
                    "negative_conflict_count": conflict_count,
                    "negative_cross_pair_count": cross_count,
                    "negative_conflict_ratio": conflict_ratio,
                    "decision_score": score,
                    "diagnostic_same_gt": _parse_bool(row.get("diagnostic_same_gt")),
                    "decision": decision,
                }
            )
    metrics = _evaluate_clusters(
        input_root=input_root,
        scene=scene,
        rows=rows,
        node_infos=node_infos,
        uf=uf,
        positive_key=positive_key,
        positive_threshold=positive_threshold,
        negative_key=negative_key,
        negative_mode=negative_mode,
        negative_weight=negative_weight,
        hard_negative_precision=hard_negative_precision,
        hard_negative_false_same_gt_count=hard_negative_false_same_gt_count,
        hard_negative_true_diff_gt_count=hard_negative_true_diff_gt_count,
        solver_variant=solver_variant,
        accepted_merge_count=accepted_merge_count,
        rejected_negative_veto_count=rejected_negative_veto_count,
        skipped_positive_hard_negative_count=skipped_positive_hard_negative_count,
        hard_negative_pairs=hard_negative_pairs,
        positive_candidate_count=len(candidate_rows),
    )
    return metrics, trace_rows


def _add_baseline_deltas(rows: list[dict[str, Any]]) -> None:
    baselines: dict[tuple[str, str, str, float], dict[str, Any]] = {}
    for row in rows:
        if row["negative_key"] == "none":
            baselines[
                (
                    str(row["input_root"]),
                    str(row["scene"]),
                    str(row["positive_key"]),
                    float(row["positive_threshold"]),
                )
            ] = row
    for row in rows:
        base = baselines.get(
            (
                str(row["input_root"]),
                str(row["scene"]),
                str(row["positive_key"]),
                float(row["positive_threshold"]),
            )
        )
        if base is None or row["negative_key"] == "none":
            row["purity_minus_s0"] = None
            row["ari_minus_s0"] = None
            row["completeness_minus_s0"] = None
            row["hard_negative_violation_rate_minus_s0"] = None
            row["diagnostic_gate_pass"] = False
            continue
        row["purity_minus_s0"] = float(row["diagnostic_subset_purity"] - base["diagnostic_subset_purity"])
        row["ari_minus_s0"] = float(row["diagnostic_subset_ari"] - base["diagnostic_subset_ari"])
        row["completeness_minus_s0"] = float(
            row["diagnostic_subset_completeness"] - base["diagnostic_subset_completeness"]
        )
        row["hard_negative_violation_rate_minus_s0"] = float(
            row["hard_negative_violation_rate"] - base["hard_negative_violation_rate"]
        )
        row["diagnostic_gate_pass"] = bool(
            row["purity_minus_s0"] >= 0.05
            and row["ari_minus_s0"] >= -0.02
            and row["completeness_minus_s0"] >= -0.10
            and row["diagnostic_subset_completeness"] >= 0.80
            and row["hard_negative_violation_rate"] <= 0.01
        )


def _best_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["input_root"]), str(row["scene"]))].append(row)
    best: list[dict[str, Any]] = []
    for (_root, _scene), group in grouped.items():
        signed = [row for row in group if row["negative_key"] != "none"]
        candidates = signed or group
        candidates = sorted(
            candidates,
            key=lambda row: (
                row.get("diagnostic_gate_pass") is True,
                row.get("purity_minus_s0") if row.get("purity_minus_s0") is not None else -999.0,
                row.get("diagnostic_subset_ari") if row.get("diagnostic_subset_ari") is not None else -999.0,
            ),
            reverse=True,
        )
        best.append(dict(candidates[0]))
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="v46 raw-edge signed solver diagnostic on P5/N4 edge rows.")
    parser.add_argument("--input-roots", required=True, help="Comma-separated raw visual semantic repair roots.")
    parser.add_argument(
        "--positive-keys",
        default="P5_p4_semantic_boost_capped,P5_p4_semantic_product_rescore_capped",
    )
    parser.add_argument("--positive-thresholds", default="0.05,0.10,0.20,0.30,0.50")
    parser.add_argument(
        "--negative-keys",
        default="none,N4_semantic_contradiction_guarded_le_0p7,N4_semantic_contradiction_guarded_le_0p75,N4_semantic_contradiction_guarded_le_0p8",
    )
    parser.add_argument("--soft-negative-weights", default="0.10,0.20,0.30,0.50")
    parser.add_argument("--expansion-conflict-ratios", default="")
    parser.add_argument(
        "--derived-negative-specs",
        default="",
        help="Comma-separated specs name:semantic_threshold:p4_max or semantic_threshold:p4_max.",
    )
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    input_roots = [Path(item) for item in str(args.input_roots).split(",") if item]
    positive_keys = [item for item in str(args.positive_keys).split(",") if item]
    positive_thresholds = [float(item) for item in str(args.positive_thresholds).split(",") if item]
    negative_keys = [item for item in str(args.negative_keys).split(",") if item]
    soft_negative_weights = [float(item) for item in str(args.soft_negative_weights).split(",") if item]
    expansion_conflict_ratios = [float(item) for item in str(args.expansion_conflict_ratios).split(",") if item]
    derived_specs = _parse_derived_negative_specs(str(args.derived_negative_specs))
    out = Path(args.output_root)

    scene_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    for input_root in input_roots:
        rows_by_scene = _scene_rows(_read_edge_rows(input_root))
        for scene, rows in rows_by_scene.items():
            for positive_key in positive_keys:
                for positive_threshold in positive_thresholds:
                    for negative_key in negative_keys:
                        if negative_key == "none":
                            mode_configs: list[tuple[str, float | None]] = [("none", None)]
                        else:
                            mode_configs = [("hard_veto", None)] + [
                                ("soft_penalty", weight) for weight in soft_negative_weights
                            ] + [("hard_veto_expand", ratio) for ratio in expansion_conflict_ratios]
                        for negative_mode, negative_weight in mode_configs:
                            metrics, trace = _solve_scene(
                                input_root=input_root,
                                scene=scene,
                                rows=rows,
                                positive_key=positive_key,
                                positive_threshold=positive_threshold,
                                negative_key=negative_key,
                                negative_mode=negative_mode,
                                negative_weight=negative_weight,
                                derived_specs=derived_specs,
                            )
                            scene_rows.append(metrics)
                            trace_rows.extend(trace)
    _add_baseline_deltas(scene_rows)
    best_rows = _best_rows(scene_rows)
    gate = {
        "pass": False,
        "diagnostic_only": True,
        "any_scene_config_diagnostic_gate_pass": any(bool(row.get("diagnostic_gate_pass")) for row in scene_rows),
        "all_best_scene_diagnostic_gate_pass": bool(best_rows and all(bool(row.get("diagnostic_gate_pass")) for row in best_rows)),
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "phase": "v46_raw_signed_solver_diagnostic",
        "input_roots": [str(root) for root in input_roots],
        "positive_keys": positive_keys,
        "positive_thresholds": positive_thresholds,
        "negative_keys": negative_keys,
        "soft_negative_weights": soft_negative_weights,
        "expansion_conflict_ratios": expansion_conflict_ratios,
        "derived_negative_specs": {
            name: {"semantic_threshold": spec[0], "p4_max": spec[1]} for name, spec in derived_specs.items()
        },
        "gate": gate,
        "diagnostic_only": True,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
        "scene_rows": scene_rows,
        "best_rows": best_rows,
    }
    _write_json(out / "raw_signed_solver_diagnostic.json", payload)
    _write_csv(out / "raw_signed_solver_scene_rows.csv", scene_rows)
    _write_csv(out / "raw_signed_solver_best_rows.csv", best_rows)
    _write_csv(out / "raw_signed_solver_merge_trace_rows.csv", trace_rows)
    print(json.dumps({"gate": gate, "summary": str(out / "raw_signed_solver_diagnostic.json")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
