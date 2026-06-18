from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


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


def _label_maps(scene: str, scan_root: Path) -> tuple[dict[str, str], dict[str, str]]:
    path = scan_root / scene / f"{scene}.aggregation.json"
    if not path.exists():
        return {}, {}
    data = json.loads(path.read_text(encoding="utf-8"))
    exact: dict[str, str] = {}
    minus_one: dict[str, str] = {}
    for group in data.get("segGroups", []):
        idx = int(group.get("id", group.get("objectId", -1)))
        label = str(group.get("label", ""))
        exact[str(idx)] = label
        minus_one[str(idx + 1)] = label
    return exact, minus_one


def _source_rows(
    *,
    scene: str,
    source: str,
    rows: list[dict[str, str]],
    exact_labels_by_gt: dict[str, str],
    minus_one_labels_by_gt: dict[str, str],
) -> dict[str, Any]:
    labeled = [row for row in rows if row.get("source") == source and row.get("diagnostic_same_gt") in {"True", "False"}]
    labels = [row["diagnostic_same_gt"] == "True" for row in labeled]
    scores = [float(row["semantic_affinity"]) for row in labeled]
    all_auc = _auc_fast(labels, scores)
    low_positive_gt = Counter(
        row["gt_i"]
        for row in labeled
        if row.get("diagnostic_same_gt") == "True"
        and row.get("gt_i") == row.get("gt_j")
        and float(row.get("semantic_affinity", "0")) < 0.50
    )
    top = [gt for gt, _count in low_positive_gt.most_common(3)]
    row: dict[str, Any] = {
        "scene": scene,
        "source": source,
        "all_semantic_affinity_AUC": all_auc,
        "gt_labeled_edge_count": int(len(labeled)),
        "top_low_positive_gt_ids": ",".join(top),
        "top_low_positive_gt_labels_exact_id": ",".join(exact_labels_by_gt.get(gt, "") for gt in top),
        "top_low_positive_gt_labels_id_minus_1": ",".join(minus_one_labels_by_gt.get(gt, "") for gt in top),
        "top_low_positive_counts": ",".join(str(low_positive_gt[gt]) for gt in top),
    }
    for count in (1, 3):
        excluded = set(top[:count])
        filtered = [edge for edge in labeled if edge.get("gt_i") not in excluded and edge.get("gt_j") not in excluded]
        row[f"auc_without_top{count}_low_positive_gt"] = _auc_fast(
            [edge["diagnostic_same_gt"] == "True" for edge in filtered],
            [float(edge["semantic_affinity"]) for edge in filtered],
        )
        row[f"edges_dropped_top{count}"] = int(len(labeled) - len(filtered))
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-root", required=True)
    parser.add_argument("--scenes", default="scene0081_01,scene0591_00")
    parser.add_argument(
        "--sources",
        default="prepared,dinov2_maskcut,dinov2_maskcut_material_split,hybrid_union_feature_merge,hybrid_union_feature_merge_material_split",
    )
    parser.add_argument("--scan-root", default="data/scannet/processed")
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    audit_root = Path(args.audit_root)
    scan_root = Path(args.scan_root)
    scenes = [item.strip() for item in str(args.scenes).split(",") if item.strip()]
    sources = [item.strip() for item in str(args.sources).split(",") if item.strip()]
    out: list[dict[str, Any]] = []
    for scene in scenes:
        exact_labels_by_gt, minus_one_labels_by_gt = _label_maps(scene, scan_root)
        edge_rows = _read_csv(audit_root / scene / "part_edge_rows.csv")
        for source in sources:
            out.append(
                _source_rows(
                    scene=scene,
                    source=source,
                    rows=edge_rows,
                    exact_labels_by_gt=exact_labels_by_gt,
                    minus_one_labels_by_gt=minus_one_labels_by_gt,
                )
            )
    _write_csv(Path(args.output_csv), out)
    print(json.dumps({"output_csv": str(args.output_csv), "row_count": len(out)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
