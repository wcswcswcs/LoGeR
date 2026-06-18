from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def _auc_fast(labels: list[bool], scores: list[float]) -> float | None:
    pairs = sorted(zip(scores, labels), key=lambda item: item[0])
    n_pos = sum(1 for _score, label in pairs if label)
    n_neg = len(pairs) - n_pos
    if not n_pos or not n_neg:
        return None
    rank_sum = 0.0
    index = 0
    while index < len(pairs):
        next_index = index + 1
        while next_index < len(pairs) and pairs[next_index][0] == pairs[index][0]:
            next_index += 1
        avg_rank = (index + 1 + next_index) / 2.0
        rank_sum += avg_rank * sum(1 for _score, label in pairs[index:next_index] if label)
        index = next_index
    return float((rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.quantile(np.asarray(values, dtype=np.float64), float(q)))


def _source_row(scene: str, source: str, rows: list[dict[str, str]]) -> dict[str, Any]:
    labeled = [row for row in rows if row.get("source") == source and row.get("diagnostic_same_gt") in {"True", "False"}]
    labels = [row["diagnostic_same_gt"] == "True" for row in labeled]
    scores = [float(row["semantic_affinity"]) for row in labeled]
    all_auc = _auc_fast(labels, scores)
    negative_scores = [float(row["semantic_affinity"]) for row in labeled if row.get("diagnostic_same_gt") == "False"]

    positives_by_gt: dict[str, list[float]] = defaultdict(list)
    for row in labeled:
        if row.get("diagnostic_same_gt") != "True" or row.get("gt_i") != row.get("gt_j"):
            continue
        gt = str(row.get("gt_i", ""))
        if not gt or gt == "0":
            continue
        positives_by_gt[gt].append(float(row["semantic_affinity"]))

    per_instance_auc: list[float] = []
    per_instance_low_rate: list[float] = []
    positive_counts = Counter({gt: len(vals) for gt, vals in positives_by_gt.items()})
    for gt, pos_scores in positives_by_gt.items():
        instance_auc = _auc_fast([True] * len(pos_scores) + [False] * len(negative_scores), pos_scores + negative_scores)
        if instance_auc is not None:
            per_instance_auc.append(float(instance_auc))
        per_instance_low_rate.append(float(sum(1 for score in pos_scores if float(score) < 0.50) / max(len(pos_scores), 1)))

    top = positive_counts.most_common(3)
    positive_edge_count = int(sum(positive_counts.values()))
    top1_share = float(top[0][1] / max(positive_edge_count, 1)) if top else 0.0
    top3_share = float(sum(count for _gt, count in top) / max(positive_edge_count, 1)) if top else 0.0
    return {
        "scene": scene,
        "source": source,
        "all_pair_semantic_affinity_AUC": all_auc,
        "instance_balanced_semantic_affinity_AUC_mean": _mean(per_instance_auc),
        "instance_balanced_semantic_affinity_AUC_p10": _quantile(per_instance_auc, 0.10),
        "instance_balanced_semantic_affinity_AUC_min": min(per_instance_auc) if per_instance_auc else None,
        "instance_balanced_low_positive_rate_mean": _mean(per_instance_low_rate),
        "positive_gt_instance_count": int(len(positives_by_gt)),
        "positive_edge_count": positive_edge_count,
        "negative_edge_count": int(len(negative_scores)),
        "top_positive_gt_ids": ",".join(gt for gt, _count in top),
        "top_positive_gt_edge_counts": ",".join(str(count) for _gt, count in top),
        "top1_positive_edge_share": top1_share,
        "top3_positive_edge_share": top3_share,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-root", required=True)
    parser.add_argument("--scenes", default="scene0081_01,scene0591_00")
    parser.add_argument("--sources", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    audit_root = Path(args.audit_root)
    scenes = [item.strip() for item in str(args.scenes).split(",") if item.strip()]
    sources = [item.strip() for item in str(args.sources).split(",") if item.strip()]
    out: list[dict[str, Any]] = []
    for scene in scenes:
        rows = _read_csv(audit_root / scene / "part_edge_rows.csv")
        for source in sources:
            out.append(_source_row(scene, source, rows))
    _write_csv(Path(args.output_csv), out)
    print(json.dumps({"output_csv": str(args.output_csv), "row_count": len(out)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
