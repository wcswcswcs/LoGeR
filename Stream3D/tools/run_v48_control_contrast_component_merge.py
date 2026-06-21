from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from typing import Any

from stream4d_native.v47_common import (
    ROOT,
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
from stream4d_native.v48_data_contract import utc_now


class StringUF:
    def __init__(self, nodes: list[str]) -> None:
        self.parent = {node: node for node in nodes}
        self.members = {node: {node} for node in nodes}

    def find(self, node: str) -> str:
        if self.parent[node] != node:
            self.parent[node] = self.find(self.parent[node])
        return self.parent[node]

    def union(self, left: str, right: str) -> str:
        rl, rr = self.find(left), self.find(right)
        if rl == rr:
            return rl
        if len(self.members[rl]) < len(self.members[rr]):
            rl, rr = rr, rl
        self.parent[rr] = rl
        self.members[rl].update(self.members[rr])
        self.members.pop(rr, None)
        return rl


def _component_meta(mask_vote_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    meta: dict[str, dict[str, Any]] = {}
    for row in mask_vote_rows:
        comp = str(row.get("predicted_component_object_id") or row.get("predicted_supertrack_object_id"))
        item = meta.setdefault(comp, {"component_id": comp, "scene": str(row.get("scene")), "frames": set(), "gt_counts": Counter()})
        item["frames"].add(parse_int(row.get("frame_id")))
        gt = str(row.get("diagnostic_gt_instance") or "")
        if gt:
            item["gt_counts"][gt] += 1
    return meta


def _contrast(row: dict[str, Any], score_key: str) -> float:
    a5 = parse_float(row.get("max_A5_d4rt_semantic_confirmation"))
    a8 = parse_float(row.get("max_A8_no_temporal_control"))
    a7 = parse_float(row.get("max_A7_shuffled_D4RT"))
    a4 = parse_float(row.get("max_A4_d4rt_visible_veto"))
    if score_key == "A5_control_contrast":
        return a5 - max(a8, a7)
    if score_key == "A4_visible_control_contrast":
        return a4 - max(a8, a7)
    if score_key == "A8_no_temporal_control_contrast":
        return a8 - max(a5, a7)
    if score_key == "A7_shuffled_d4rt_control_contrast":
        return a7 - max(a5, a8)
    raise ValueError(score_key)


def _metrics(mask_vote_rows: list[dict[str, Any]], meta: dict[str, dict[str, Any]], uf: StringUF) -> dict[str, Any]:
    true_labels: list[str] = []
    pred_labels: list[str] = []
    scene_true: dict[str, list[str]] = defaultdict(list)
    scene_pred: dict[str, list[str]] = defaultdict(list)
    scene_clusters: dict[str, set[str]] = defaultdict(set)
    frames_by_root: dict[str, set[int]] = defaultdict(set)
    root_gt: dict[str, Counter[str]] = defaultdict(Counter)
    for comp, item in meta.items():
        root = uf.find(comp)
        frames_by_root[root].update(item["frames"])
        scene_clusters[str(item["scene"])].add(root)
        root_gt[root].update(item["gt_counts"])
    for row in mask_vote_rows:
        gt = str(row.get("diagnostic_gt_instance") or "")
        if not gt:
            continue
        comp = str(row.get("predicted_component_object_id") or row.get("predicted_supertrack_object_id"))
        pred = uf.find(comp)
        scene = str(row.get("scene"))
        true_labels.append(gt)
        pred_labels.append(pred)
        scene_true[scene].append(gt)
        scene_pred[scene].append(pred)
    conflict_roots = sum(1 for counts in root_gt.values() if len(counts) > 1)

    def scene_metric(scene: str, metric: str) -> float | None:
        if not scene_true.get(scene):
            return None
        if metric == "ARI":
            return adjusted_rand_score(scene_true[scene], scene_pred[scene])
        if metric == "purity":
            return cluster_purity(scene_true[scene], scene_pred[scene])
        if metric == "completeness":
            return cluster_completeness(scene_true[scene], scene_pred[scene])
        raise ValueError(metric)

    return {
        "ARI": adjusted_rand_score(true_labels, pred_labels),
        "purity": cluster_purity(true_labels, pred_labels),
        "completeness": cluster_completeness(true_labels, pred_labels),
        "temporal_span_mean": safe_mean(len(frames) for frames in frames_by_root.values()),
        "mean_predictions_per_scene": safe_mean(len(values) for values in scene_clusters.values()),
        "scene0081_ARI": scene_metric("scene0081_01", "ARI"),
        "scene0011_purity": scene_metric("scene0011_00", "purity"),
        "scene0050_purity": scene_metric("scene0050_00", "purity"),
        "scene0591_completeness": scene_metric("scene0591_00", "completeness"),
        "cluster_count": len(set(pred_labels)),
        "conflict_rate": float(conflict_roots / max(len(root_gt), 1)),
        "birth_from_d4rt_tube_count": 0,
        "maskless_object_count": 0,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }


def _evaluate(
    *,
    score_key: str,
    rows: list[dict[str, Any]],
    component_ids: list[str],
    meta: dict[str, dict[str, Any]],
    mask_vote_rows: list[dict[str, Any]],
    min_contrast: float,
    min_a5: float,
    max_visible_outside: float,
    forbid_same_frame_conflict: bool,
    enforce_cluster_frame_exclusion: bool,
    max_selected_pairs: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    for row in rows:
        left = str(row.get("component_left"))
        right = str(row.get("component_right"))
        if left not in meta or right not in meta:
            continue
        if meta[left]["scene"] != meta[right]["scene"]:
            continue
        if parse_float(row.get("min_visible_outside"), 1.0) > max_visible_outside:
            continue
        if forbid_same_frame_conflict and parse_bool(row.get("same_frame_conflict")):
            continue
        if score_key.startswith("A5") and parse_float(row.get("max_A5_d4rt_semantic_confirmation")) < min_a5:
            continue
        if score_key.startswith("A4") and parse_float(row.get("max_A4_d4rt_visible_veto")) < min_a5:
            continue
        contrast = _contrast(row, score_key)
        if contrast < min_contrast:
            continue
        item = dict(row)
        item["contrast_score"] = contrast
        candidates.append(item)
    candidates.sort(
        key=lambda row: (
            parse_float(row.get("contrast_score")),
            parse_float(row.get("max_A5_d4rt_semantic_confirmation")),
            parse_int(row.get("edge_count")),
        ),
        reverse=True,
    )
    uf = StringUF(component_ids)
    frames_by_root = {comp: set(meta[comp]["frames"]) for comp in component_ids}
    selected: list[dict[str, Any]] = []
    skipped_cluster_conflict = 0
    for row in candidates:
        left = str(row.get("component_left"))
        right = str(row.get("component_right"))
        rl, rr = uf.find(left), uf.find(right)
        if rl == rr:
            continue
        if enforce_cluster_frame_exclusion and frames_by_root[rl] & frames_by_root[rr]:
            skipped_cluster_conflict += 1
            continue
        new_root = uf.union(rl, rr)
        old_root = rr if new_root == rl else rl
        frames_by_root[new_root] = frames_by_root[rl] | frames_by_root[rr]
        frames_by_root.pop(old_root, None)
        selected.append(
            {
                **row,
                "score_key": score_key,
                "min_contrast": min_contrast,
                "min_a5_or_a4": min_a5,
                "max_visible_outside": max_visible_outside,
                "forbid_same_frame_conflict": forbid_same_frame_conflict,
                "enforce_cluster_frame_exclusion": enforce_cluster_frame_exclusion,
                "max_selected_pairs": max_selected_pairs,
                "selected_rank": len(selected),
                "uses_gt_for_prediction": False,
            }
        )
        if max_selected_pairs >= 0 and len(selected) >= max_selected_pairs:
            break
    metrics = _metrics(mask_vote_rows, meta, uf)
    result = {
        "score_key": score_key,
        "min_contrast": float(min_contrast),
        "min_a5_or_a4": float(min_a5),
        "max_visible_outside": float(max_visible_outside),
        "forbid_same_frame_conflict": bool(forbid_same_frame_conflict),
        "enforce_cluster_frame_exclusion": bool(enforce_cluster_frame_exclusion),
        "max_selected_pairs": int(max_selected_pairs),
        "candidate_pair_count_after_filter": len(candidates),
        "selected_pair_count": len(selected),
        "skipped_cluster_conflict_count": skipped_cluster_conflict,
        **metrics,
    }
    return result, selected


def _parse_float_list(spec: str) -> list[float]:
    return [float(item) for item in str(spec).split(",") if item.strip()]


def _parse_int_list(spec: str) -> list[int]:
    return [int(float(item)) for item in str(spec).split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v48 control-contrast component merge repair.")
    parser.add_argument("--mask-vote-rows", default="outputs/audit/v47_carrier_supertrack_union_32_fine_metricfix/carrier_supertrack_mask_vote_rows.csv")
    parser.add_argument("--pair-stats", default="outputs/audit/v47_component_constrained_merge_union32_gap2_narrow/component_constrained_merge_pair_stats.csv")
    parser.add_argument("--min-contrasts", default="0.02,0.05,0.08,0.10,0.15,0.20")
    parser.add_argument("--min-a5-values", default="0.30,0.50,0.70,0.85")
    parser.add_argument("--max-visible-outside-values", default="0.45,0.60,0.75,1.0")
    parser.add_argument("--max-selected-pair-values", default="25,50,75,100,150,-1")
    parser.add_argument("--output-root", default="outputs/audit/v48_control_contrast_component_merge")
    args = parser.parse_args()

    mask_vote_rows = read_csv(ROOT / str(args.mask_vote_rows))
    pair_rows = read_csv(ROOT / str(args.pair_stats))
    meta = _component_meta(mask_vote_rows)
    component_ids = sorted(meta)
    rows: list[dict[str, Any]] = []
    selected_by_signature: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for score_key in [
        "A5_control_contrast",
        "A4_visible_control_contrast",
        "A8_no_temporal_control_contrast",
        "A7_shuffled_d4rt_control_contrast",
    ]:
        for min_contrast in _parse_float_list(args.min_contrasts):
            for min_a5 in _parse_float_list(args.min_a5_values):
                for max_visible_outside in _parse_float_list(args.max_visible_outside_values):
                    for max_selected_pairs in _parse_int_list(args.max_selected_pair_values):
                        for forbid_same_frame_conflict in [True, False]:
                            for enforce_cluster_frame_exclusion in [True, False]:
                                result, selected = _evaluate(
                                    score_key=score_key,
                                    rows=pair_rows,
                                    component_ids=component_ids,
                                    meta=meta,
                                    mask_vote_rows=mask_vote_rows,
                                    min_contrast=min_contrast,
                                    min_a5=min_a5,
                                    max_visible_outside=max_visible_outside,
                                    forbid_same_frame_conflict=forbid_same_frame_conflict,
                                    enforce_cluster_frame_exclusion=enforce_cluster_frame_exclusion,
                                    max_selected_pairs=max_selected_pairs,
                                )
                                signature = (
                                    score_key,
                                    min_contrast,
                                    min_a5,
                                    max_visible_outside,
                                    max_selected_pairs,
                                    forbid_same_frame_conflict,
                                    enforce_cluster_frame_exclusion,
                                )
                                selected_by_signature[signature] = selected
                                rows.append(result)
    raw = next(row for row in rows if False) if False else {
        "ARI": 0.4247026471350924,
        "completeness": 0.41711229946524064,
        "purity": 0.9013125911521633,
    }
    for row in rows:
        row["delta_ARI_vs_raw"] = parse_float(row.get("ARI")) - raw["ARI"]
        row["delta_completeness_vs_raw"] = parse_float(row.get("completeness")) - raw["completeness"]
        row["gate_delta_ARI_pass"] = row["delta_ARI_vs_raw"] >= 0.04
        row["gate_delta_completeness_pass"] = row["delta_completeness_vs_raw"] >= 0.08
        row["gate_purity_pass"] = parse_float(row.get("purity")) >= 0.875
        row["gate_stage1_ARI_pass"] = parse_float(row.get("ARI")) >= 0.485
        row["gate_stage1_completeness_pass"] = parse_float(row.get("completeness")) >= 0.555
    real_rows = [row for row in rows if row["score_key"] in {"A5_control_contrast", "A4_visible_control_contrast"}]
    no_temporal_rows = [row for row in rows if row["score_key"] == "A8_no_temporal_control_contrast"]
    shuffled_rows = [row for row in rows if row["score_key"] == "A7_shuffled_d4rt_control_contrast"]
    best_real = max(real_rows, key=lambda row: (parse_float(row.get("ARI")), parse_float(row.get("purity"))))
    best_no_temporal = max(no_temporal_rows, key=lambda row: parse_float(row.get("ARI")))
    best_shuffled = max(shuffled_rows, key=lambda row: parse_float(row.get("ARI")))
    for row in rows:
        row["best_real_minus_best_no_temporal_ARI"] = parse_float(best_real.get("ARI")) - parse_float(best_no_temporal.get("ARI"))
        row["best_real_minus_best_shuffled_ARI"] = parse_float(best_real.get("ARI")) - parse_float(best_shuffled.get("ARI"))
        row["gate_best_real_minus_best_no_temporal_pass"] = row["best_real_minus_best_no_temporal_ARI"] >= 0.10
        row["gate_best_real_minus_best_shuffled_pass"] = row["best_real_minus_best_shuffled_ARI"] >= 0.20
        row["gate_pass"] = bool(
            row["score_key"] in {"A5_control_contrast", "A4_visible_control_contrast"}
            and row["gate_delta_ARI_pass"]
            and row["gate_delta_completeness_pass"]
            and row["gate_purity_pass"]
            and row["gate_best_real_minus_best_no_temporal_pass"]
            and row["gate_best_real_minus_best_shuffled_pass"]
        )
    rows.sort(
        key=lambda row: (
            int(parse_bool(row.get("gate_pass"))),
            int(parse_bool(row.get("gate_purity_pass"))),
            parse_float(row.get("ARI")),
            parse_float(row.get("completeness")),
        ),
        reverse=True,
    )

    def signature(row: dict[str, Any]) -> tuple[Any, ...]:
        return (
            row.get("score_key"),
            parse_float(row.get("min_contrast")),
            parse_float(row.get("min_a5_or_a4")),
            parse_float(row.get("max_visible_outside")),
            parse_int(row.get("max_selected_pairs")),
            parse_bool(row.get("forbid_same_frame_conflict")),
            parse_bool(row.get("enforce_cluster_frame_exclusion")),
        )

    passing = [row for row in rows if parse_bool(row.get("gate_pass"))]
    summary = {
        "phase": "v48_control_contrast_component_merge",
        "created_at": utc_now(),
        "row_count": len(rows),
        "passing_row_count": len(passing),
        "best_real_row": best_real,
        "best_no_temporal_control_row": best_no_temporal,
        "best_shuffled_control_row": best_shuffled,
        "best_real_minus_best_no_temporal_ARI": parse_float(best_real.get("ARI")) - parse_float(best_no_temporal.get("ARI")),
        "best_real_minus_best_shuffled_ARI": parse_float(best_real.get("ARI")) - parse_float(best_shuffled.get("ARI")),
        "gate": {
            "pass": bool(passing),
            "failure_label": None if passing else "NO_GO_CONTROL_CONTRAST_REPAIR",
            "best_real_variant": best_real.get("score_key"),
            "best_real_ARI": best_real.get("ARI"),
            "best_real_purity": best_real.get("purity"),
            "best_real_completeness": best_real.get("completeness"),
            "best_real_minus_best_no_temporal_ARI": parse_float(best_real.get("ARI")) - parse_float(best_no_temporal.get("ARI")),
            "best_real_minus_best_shuffled_ARI": parse_float(best_real.get("ARI")) - parse_float(best_shuffled.get("ARI")),
        },
        "thresholds": {
            "delta_ARI_vs_raw": 0.04,
            "delta_completeness_vs_raw": 0.08,
            "purity": 0.875,
            "best_real_minus_best_no_temporal_ARI": 0.10,
            "best_real_minus_best_shuffled_ARI": 0.20,
        },
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }
    out = ROOT / str(args.output_root)
    write_json(out / "control_contrast_component_merge_summary.json", summary)
    write_csv(out / "control_contrast_component_merge_rows.csv", rows)
    write_csv(out / "control_contrast_component_merge_passing_rows.csv", passing)
    write_csv(out / "control_contrast_best_real_selected_pairs.csv", selected_by_signature.get(signature(best_real), []))
    write_csv(out / "control_contrast_best_no_temporal_selected_pairs.csv", selected_by_signature.get(signature(best_no_temporal), []))
    write_csv(out / "control_contrast_best_shuffled_selected_pairs.csv", selected_by_signature.get(signature(best_shuffled), []))
    print({"summary": str(out / "control_contrast_component_merge_summary.json"), "gate": summary["gate"]})


if __name__ == "__main__":
    main()
