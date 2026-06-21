from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from typing import Any

from .v47_common import UnionFind, adjusted_rand_score, cluster_completeness, cluster_purity, parse_bool, parse_int, safe_mean


def _mask_key(row: dict[str, Any]) -> tuple[str, int, int]:
    return str(row.get("scene")), parse_int(row.get("frame_id")), parse_int(row.get("mask_id"))


def _stable_shuffle(values: list[str], seed: str) -> list[str]:
    keyed = [(hashlib.sha1(f"{seed}:{idx}:{value}".encode("utf-8")).hexdigest(), value) for idx, value in enumerate(values)]
    keyed.sort()
    ordered = [value for _key, value in keyed]
    if not ordered:
        return []
    return ordered[1:] + ordered[:1]


def build_carrier_supertrack_summary(
    carrier_rows: list[dict[str, Any]],
    mask_rows: list[dict[str, Any]],
    max_union_unique_carriers: int | None = None,
) -> dict[str, Any]:
    mask_by_key = {_mask_key(row): row for row in mask_rows}
    lengths: Counter[str] = Counter()
    carrier_gt_counts: dict[str, Counter[str]] = defaultdict(Counter)
    carrier_mask_counts: dict[str, Counter[str]] = defaultdict(Counter)
    mask_carrier_counts: dict[tuple[str, int, int], Counter[str]] = defaultdict(Counter)
    scene_labeled_mask_keys: dict[str, set[tuple[str, int, int]]] = defaultdict(set)
    scene_supported_mask_keys: dict[str, set[tuple[str, int, int]]] = defaultdict(set)

    for row in mask_rows:
        gt = str(row.get("diagnostic_gt_instance", ""))
        if gt:
            scene_labeled_mask_keys[str(row.get("scene"))].add(_mask_key(row))

    for row in carrier_rows:
        if not (parse_bool(row.get("visible")) and parse_bool(row.get("valid_uv"))):
            continue
        carrier_global_id = str(row.get("carrier_global_id") or f"{row.get('scene')}:{parse_int(row.get('carrier_id'))}")
        lengths[carrier_global_id] += 1
        observed_mask_id = parse_int(row.get("observed_mask_id"))
        if observed_mask_id <= 0:
            continue
        key = (str(row.get("scene")), parse_int(row.get("frame_id")), observed_mask_id)
        mask_row = mask_by_key.get(key)
        if not mask_row:
            continue
        mask_obs_id = str(mask_row.get("mask_observation_id"))
        carrier_mask_counts[carrier_global_id][mask_obs_id] += 1
        mask_carrier_counts[key][carrier_global_id] += 1
        scene_supported_mask_keys[str(row.get("scene"))].add(key)
        gt = str(mask_row.get("diagnostic_gt_instance", ""))
        if gt:
            carrier_gt_counts[carrier_global_id][gt] += 1

    supertrack_rows: list[dict[str, Any]] = []
    purity_numer = 0
    purity_denom = 0
    for carrier_global_id, length in lengths.items():
        gt_counts = carrier_gt_counts.get(carrier_global_id, Counter())
        labeled_count = sum(gt_counts.values())
        dominant_gt = ""
        dominant_count = 0
        if gt_counts:
            dominant_gt, dominant_count = gt_counts.most_common(1)[0]
            purity_numer += dominant_count
            purity_denom += labeled_count
        supertrack_rows.append(
            {
                "carrier_global_id": carrier_global_id,
                "visible_observation_count": int(length),
                "mask_support_count": int(sum(carrier_mask_counts.get(carrier_global_id, Counter()).values())),
                "unique_mask_support_count": int(len(carrier_mask_counts.get(carrier_global_id, Counter()))),
                "diagnostic_labeled_observation_count": int(labeled_count),
                "diagnostic_dominant_gt": dominant_gt,
                "diagnostic_supertrack_purity": None if labeled_count <= 0 else float(dominant_count / labeled_count),
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic_labels": True,
            }
        )

    mask_vote_rows: list[dict[str, Any]] = []
    true_labels: list[str] = []
    pred_labels: list[str] = []
    carrier_ids = sorted(lengths)
    carrier_index = {carrier_id: idx for idx, carrier_id in enumerate(carrier_ids)}
    carrier_by_index = {idx: carrier_id for carrier_id, idx in carrier_index.items()}
    uf = UnionFind(carrier_index.values())
    for carrier_counts in mask_carrier_counts.values():
        ids = list(carrier_counts)
        if len(ids) <= 1:
            continue
        if max_union_unique_carriers is not None and len(ids) > int(max_union_unique_carriers):
            continue
        first = carrier_index[ids[0]]
        for other in ids[1:]:
            uf.union(first, carrier_index[other])
    component_labels_by_root: dict[str, str] = {}
    component_true_labels: list[str] = []
    component_pred_labels: list[str] = []
    for row in mask_rows:
        key = _mask_key(row)
        gt = str(row.get("diagnostic_gt_instance", ""))
        carrier_counts = mask_carrier_counts.get(key, Counter())
        if carrier_counts:
            pred = carrier_counts.most_common(1)[0][0]
            support = carrier_counts.most_common(1)[0][1]
            root_idx = uf.find(carrier_index[pred])
            root = carrier_by_index[root_idx]
            if root not in component_labels_by_root:
                component_labels_by_root[root] = f"c{len(component_labels_by_root):05d}"
            component_pred = component_labels_by_root[root]
        else:
            pred = f"uncovered:{row.get('mask_observation_id')}"
            support = 0
            component_pred = pred
        mask_vote_rows.append(
            {
                "node_id": row.get("node_id"),
                "mask_observation_id": row.get("mask_observation_id"),
                "scene": row.get("scene"),
                "frame_id": row.get("frame_id"),
                "mask_id": row.get("mask_id"),
                "predicted_supertrack_object_id": pred,
                "predicted_component_object_id": component_pred,
                "supporting_carrier_observation_count": int(support),
                "supporting_unique_carrier_count": int(len(carrier_counts)),
                "diagnostic_gt_instance": gt,
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic_labels": True,
            }
        )
        if gt:
            true_labels.append(gt)
            pred_labels.append(pred)
            component_true_labels.append(gt)
            component_pred_labels.append(component_pred)

    shuffled_pred = _stable_shuffle(pred_labels, "v47_carrier_supertrack_shuffle")
    component_shuffled_pred = _stable_shuffle(component_pred_labels, "v47_carrier_component_shuffle")
    scene_rows: list[dict[str, Any]] = []
    for scene, labeled_keys in sorted(scene_labeled_mask_keys.items()):
        supported = scene_supported_mask_keys.get(scene, set())
        scene_supertrack_rows = [row for row in supertrack_rows if str(row["carrier_global_id"]).startswith(f"{scene}:")]
        scene_purities = [
            row.get("diagnostic_supertrack_purity")
            for row in scene_supertrack_rows
            if row.get("diagnostic_supertrack_purity") is not None
        ]
        scene_rows.append(
            {
                "scene": scene,
                "labeled_mask_count": int(len(labeled_keys)),
                "supported_labeled_mask_count": int(len(labeled_keys & supported)),
                "supertrack_coverage": float(len(labeled_keys & supported) / max(len(labeled_keys), 1)),
                "supertrack_purity": safe_mean(scene_purities),
            }
        )

    real_ari = adjusted_rand_score(true_labels, pred_labels)
    shuffled_ari = adjusted_rand_score(true_labels, shuffled_pred) if shuffled_pred else 0.0
    component_ari = adjusted_rand_score(component_true_labels, component_pred_labels)
    component_shuffled_ari = (
        adjusted_rand_score(component_true_labels, component_shuffled_pred) if component_shuffled_pred else 0.0
    )
    summary = {
        "phase": "v47_carrier_supertrack_diagnostic",
        "supertrack_count": len(lengths),
        "supertrack_length_mean": safe_mean(lengths.values()),
        "supertrack_purity": None if purity_denom <= 0 else float(purity_numer / purity_denom),
        "supertrack_coverage": float(
            sum(len(scene_labeled_mask_keys[scene] & scene_supported_mask_keys.get(scene, set())) for scene in scene_labeled_mask_keys)
            / max(sum(len(values) for values in scene_labeled_mask_keys.values()), 1)
        ),
        "mask_support_per_supertrack": safe_mean(row["unique_mask_support_count"] for row in supertrack_rows),
        "object_from_supertrack_ARI": real_ari,
        "object_from_supertrack_purity": cluster_purity(true_labels, pred_labels),
        "object_from_supertrack_completeness": cluster_completeness(true_labels, pred_labels),
        "shuffled_supertrack_ARI": shuffled_ari,
        "real_minus_shuffled_supertrack_ARI": float(real_ari - shuffled_ari),
        "component_count": len(component_labels_by_root),
        "max_union_unique_carriers": max_union_unique_carriers,
        "object_from_component_ARI": component_ari,
        "object_from_component_purity": cluster_purity(component_true_labels, component_pred_labels),
        "object_from_component_completeness": cluster_completeness(component_true_labels, component_pred_labels),
        "shuffled_component_ARI": component_shuffled_ari,
        "real_minus_shuffled_component_ARI": float(component_ari - component_shuffled_ari),
        "scene_rows": scene_rows,
        "gate": {
            "supertrack_purity_pass": bool((0.0 if purity_denom <= 0 else purity_numer / purity_denom) >= 0.90),
            "real_minus_shuffled_supertrack_ARI_pass": bool((real_ari - shuffled_ari) >= 0.20),
        },
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }
    summary["gate"]["pass"] = bool(all(summary["gate"].values()))
    return {"summary": summary, "supertrack_rows": supertrack_rows, "mask_vote_rows": mask_vote_rows, "scene_rows": scene_rows}


__all__ = ["build_carrier_supertrack_summary"]
