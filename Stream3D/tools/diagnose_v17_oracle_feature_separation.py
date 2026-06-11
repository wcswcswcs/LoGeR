from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from tools.v11_candidate_pool_oracle import _json_safe, _load_prediction, _read_seq_list


def _safe_auc(labels: np.ndarray, values: np.ndarray) -> float | None:
    labels = labels.astype(bool, copy=False)
    values = values.astype(np.float64, copy=False)
    pos = values[labels]
    neg = values[~labels]
    if pos.size == 0 or neg.size == 0:
        return None
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, values.size + 1, dtype=np.float64)
    # Average tied ranks.
    sorted_values = values[order]
    start = 0
    while start < sorted_values.size:
        end = start + 1
        while end < sorted_values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        if end - start > 1:
            ranks[order[start:end]] = float(np.mean(np.arange(start + 1, end + 1)))
        start = end
    rank_sum_pos = float(np.sum(ranks[labels]))
    auc = (rank_sum_pos - pos.size * (pos.size + 1) / 2.0) / float(pos.size * neg.size)
    return float(auc)


def _spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < 2 or y.size < 2 or np.std(x) <= 0.0 or np.std(y) <= 0.0:
        return None
    xr = np.argsort(np.argsort(x, kind="mergesort"), kind="mergesort").astype(np.float64)
    yr = np.argsort(np.argsort(y, kind="mergesort"), kind="mergesort").astype(np.float64)
    return float(np.corrcoef(xr, yr)[0, 1])


def _overlap_features(candidates: np.ndarray, anchors: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = candidates.shape[1]
    if anchors.shape[1] == 0 or n == 0:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    cand_area = candidates.sum(axis=0).astype(np.float64)
    anchor_area = anchors.sum(axis=0).astype(np.float64)
    inter = candidates.astype(np.int64).T @ anchors.astype(np.int64)
    union = cand_area[:, None] + anchor_area[None, :] - inter
    iou = inter / np.maximum(union, 1.0)
    min_ioc = inter / np.maximum(np.minimum(cand_area[:, None], anchor_area[None, :]), 1.0)
    cand_ioc = inter / np.maximum(cand_area[:, None], 1.0)
    return (
        np.max(iou, axis=1).astype(np.float64),
        np.max(min_ioc, axis=1).astype(np.float64),
        np.max(cand_ioc, axis=1).astype(np.float64),
    )


def _conflict_features(masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = masks.shape[1]
    if n == 0:
        return np.zeros(0), np.zeros(0)
    areas = masks.sum(axis=0).astype(np.float64)
    point_counts = masks.astype(np.int16).sum(axis=1)
    overlap_mass = (masks.T @ np.maximum(point_counts - 1, 0)).astype(np.float64)
    conflict_rate = overlap_mass / np.maximum(areas, 1.0)
    inter = masks.astype(np.int64).T @ masks.astype(np.int64)
    np.fill_diagonal(inter, 0)
    max_ioc = inter / np.maximum(np.minimum(areas[:, None], areas[None, :]), 1.0)
    return conflict_rate, np.max(max_ioc, axis=1).astype(np.float64)


def _oracle_labels(oracle_payload: dict[str, Any], scene: str, k: int) -> tuple[set[int], dict[int, float]]:
    selected: set[int] = set()
    gain: dict[int, float] = {}
    for scene_row in oracle_payload.get("scenes", []):
        if scene_row.get("scene") != scene:
            continue
        for gt_row in scene_row.get("per_gt", []):
            chosen = [int(v) for v in gt_row.get("selected_pred_indices", [])[: int(k)]]
            best_iou = float(gt_row.get(f"best_iou_k{int(k)}", 0.0) or 0.0)
            for rank, idx in enumerate(chosen):
                selected.add(idx)
                gain[idx] = max(float(gain.get(idx, 0.0)), best_iou / float(rank + 1))
    return selected, gain


def _scene_features(root: Path, args: argparse.Namespace, oracle_payload: dict[str, Any], scene: str) -> list[dict[str, Any]]:
    hybrid = _load_prediction(root, args.pred_config, args.pred_suffix, scene)
    regionlet = _load_prediction(root, args.regionlet_config, args.pred_suffix, scene)
    mask = _load_prediction(root, args.mask_config, args.pred_suffix, scene)
    surfel = _load_prediction(root, args.surfel_config, args.pred_suffix, scene)
    h_masks = np.asarray(hybrid["pred_masks"], dtype=bool)
    h_scores = np.asarray(hybrid["pred_score"], dtype=np.float64)
    support = np.load(root / "data" / "TMP" / args.pred_config / f"{scene}_pre_points.npy").astype(np.int64)
    selected, gain = _oracle_labels(oracle_payload, scene, int(args.k))

    area = h_masks.sum(axis=0).astype(np.float64)
    support_area = h_masks[support, :].sum(axis=0).astype(np.float64) if support.size else area
    conflict_rate, max_candidate_overlap = _conflict_features(h_masks)
    reg_iou, reg_minioc, reg_candioc = _overlap_features(h_masks, np.asarray(regionlet["pred_masks"], dtype=bool))
    mask_iou, mask_minioc, mask_candioc = _overlap_features(h_masks, np.asarray(mask["pred_masks"], dtype=bool))
    surf_iou, surf_minioc, surf_candioc = _overlap_features(h_masks, np.asarray(surfel["pred_masks"], dtype=bool))
    rows: list[dict[str, Any]] = []
    denom = float(max(h_masks.shape[0], 1))
    for idx in range(h_masks.shape[1]):
        rows.append(
            {
                "scene": scene,
                "candidate_index": int(idx),
                "oracle_selected": bool(idx in selected),
                "oracle_marginal_gain_proxy": float(gain.get(idx, 0.0)),
                "area_points": float(area[idx]),
                "area_scene_ratio": float(area[idx] / denom),
                "support_area_points": float(support_area[idx]),
                "score": float(h_scores[idx]),
                "log_score": float(np.log1p(max(h_scores[idx], 0.0))),
                "score_per_sqrt_area": float(h_scores[idx] / max(np.sqrt(max(area[idx], 1.0)), 1.0)),
                "conflict_rate": float(conflict_rate[idx]),
                "max_candidate_overlap": float(max_candidate_overlap[idx]),
                "regionlet_max_iou": float(reg_iou[idx]),
                "regionlet_max_min_ioc": float(reg_minioc[idx]),
                "regionlet_candidate_coverage": float(reg_candioc[idx]),
                "mask_max_iou": float(mask_iou[idx]),
                "mask_max_min_ioc": float(mask_minioc[idx]),
                "mask_candidate_coverage": float(mask_candioc[idx]),
                "surfel_max_iou": float(surf_iou[idx]),
                "surfel_max_min_ioc": float(surf_minioc[idx]),
                "surfel_candidate_coverage": float(surf_candioc[idx]),
                "anchor_agreement_score": float(max(reg_minioc[idx], surf_minioc[idx], mask_minioc[idx])),
                "visible_outside_negative_proxy": float(conflict_rate[idx] * (1.0 - max(reg_candioc[idx], surf_candioc[idx]))),
                "boundary_risk_proxy": float(max_candidate_overlap[idx] * conflict_rate[idx]),
            }
        )
    return rows


def _summarize(rows: list[dict[str, Any]], k: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    forbidden_feature_keys = {"candidate_index", "oracle_selected", "oracle_marginal_gain_proxy"}
    feature_names = [
        key
        for key, value in rows[0].items()
        if isinstance(value, (float, int)) and not isinstance(value, bool) and key not in forbidden_feature_keys
    ]
    labels = np.asarray([bool(row["oracle_selected"]) for row in rows], dtype=bool)
    gain = np.asarray([float(row["oracle_marginal_gain_proxy"]) for row in rows], dtype=np.float64)
    selected_count = int(np.count_nonzero(labels))
    feature_rows: list[dict[str, Any]] = []
    for feature in feature_names:
        values = np.asarray([float(row[feature]) for row in rows], dtype=np.float64)
        auc = _safe_auc(labels, values)
        if auc is not None and auc < 0.5:
            direction = "low"
            auc_directional = 1.0 - float(auc)
            rank_values = -values
        else:
            direction = "high"
            auc_directional = auc
            rank_values = values
        selected_vals = values[labels]
        rejected_vals = values[~labels]
        order = np.argsort(-rank_values, kind="mergesort")[:selected_count] if selected_count > 0 else np.zeros(0, dtype=np.int64)
        recovered = int(np.count_nonzero(labels[order])) if order.size else 0
        feature_rows.append(
            {
                "feature": feature,
                "selected_mean": float(np.mean(selected_vals)) if selected_vals.size else None,
                "selected_median": float(np.median(selected_vals)) if selected_vals.size else None,
                "selected_std": float(np.std(selected_vals)) if selected_vals.size else None,
                "rejected_mean": float(np.mean(rejected_vals)) if rejected_vals.size else None,
                "rejected_median": float(np.median(rejected_vals)) if rejected_vals.size else None,
                "rejected_std": float(np.std(rejected_vals)) if rejected_vals.size else None,
                "selected_rejected_ratio": (
                    float(np.mean(selected_vals) / max(abs(float(np.mean(rejected_vals))), 1e-9))
                    if selected_vals.size and rejected_vals.size
                    else None
                ),
                "auc_high": auc,
                "auc_directional": auc_directional,
                "direction": direction,
                "spearman_with_oracle_gain_proxy": _spearman(values, gain),
                "topk_recovery_count": recovered,
                "topk_recovery_recall": float(recovered / max(selected_count, 1)),
                "topk_recovery_precision": float(recovered / max(int(order.size), 1)),
            }
        )
    best_auc = max((row.get("auc_directional") or 0.0 for row in feature_rows), default=0.0)
    num_auc_ge_062 = int(sum(1 for row in feature_rows if (row.get("auc_directional") or 0.0) >= 0.62))
    summary = {
        "diagnostic_only": True,
        "uses_gt_for_diagnostic": True,
        "uses_gt_for_prediction": False,
        "is_method_result": False,
        "forbidden_for_method_table": True,
        "gt_selected_output": False,
        "k": int(k),
        "num_candidates": int(len(rows)),
        "num_oracle_selected_candidates": selected_count,
        "best_feature_auc_directional": float(best_auc),
        "num_features_auc_ge_0p62": num_auc_ge_062,
        "has_single_feature_auc_ge_0p70": bool(best_auc >= 0.70),
        "has_three_features_auc_ge_0p62": bool(num_auc_ge_062 >= 3),
    }
    return sorted(feature_rows, key=lambda row: float(row.get("auc_directional") or 0.0), reverse=True), summary


def _write_outputs(prefix: Path, payload: dict[str, Any]) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".json").write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")
    for name, rows in (("features", payload["feature_rows"]), ("candidates", payload["candidate_rows"])):
        if rows:
            with (prefix.parent / f"{prefix.name}_{name}.csv").open("w", encoding="utf-8", newline="") as handle:
                fieldnames = sorted({key for row in rows for key in row.keys()})
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for row in rows:
                    writer.writerow({key: row.get(key, "") for key in fieldnames})
    lines = [
        "# Stream4D v17 Oracle-Selected vs Rejected Feature Separation",
        "",
        "Oracle labels are GT-read diagnostic labels from v16 union oracle. Candidate features below are non-GT mask/score/overlap proxies.",
        "",
        "## Summary",
        "",
    ]
    for key, value in payload["summary"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Feature AUC",
            "",
            "| feature | direction | AUC | selected mean | rejected mean | top-k recall | spearman gain |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["feature_rows"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["feature"]),
                    str(row["direction"]),
                    f"{float(row.get('auc_directional') or 0.0):.6f}",
                    "NA" if row.get("selected_mean") is None else f"{float(row['selected_mean']):.6f}",
                    "NA" if row.get("rejected_mean") is None else f"{float(row['rejected_mean']):.6f}",
                    f"{float(row.get('topk_recovery_recall') or 0.0):.6f}",
                    "NA"
                    if row.get("spearman_with_oracle_gain_proxy") is None
                    else f"{float(row['spearman_with_oracle_gain_proxy']):.6f}",
                ]
            )
            + " |"
        )
    prefix.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--seq-list", required=True)
    parser.add_argument("--pred-config", default="stream4d_v13_c_hybrid_unsup_probe5")
    parser.add_argument("--regionlet-config", default="stream4d_v13_c_regionlet_unsup_probe5")
    parser.add_argument("--mask-config", default="stream4d_v13_c_mask_unsup_probe5")
    parser.add_argument("--surfel-config", default="stream4d_v13_c_surfel_unsup_probe5")
    parser.add_argument("--oracle-json", default="outputs/audit/v16_phase1/c_hybrid_union_oracle_probe5.json")
    parser.add_argument("--pred-suffix", default="class_agnostic")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--output-prefix", default="outputs/audit/v17_phase2/c_hybrid_oracle_feature_separation_probe5")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    oracle_payload = json.loads((root / args.oracle_json).read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for scene in _read_seq_list((root / args.seq_list).resolve()):
        rows.extend(_scene_features(root, args, oracle_payload, scene))
    feature_rows, summary = _summarize(rows, int(args.k))
    payload = {"args": vars(args), "summary": summary, "feature_rows": feature_rows, "candidate_rows": rows}
    _write_outputs(root / args.output_prefix, payload)
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
