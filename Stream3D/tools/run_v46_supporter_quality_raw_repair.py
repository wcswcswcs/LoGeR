from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from tools.run_v46_raw_carrier_incidence_repair import (
    ROOT,
    MaskNode,
    WindowTrace,
    _build_nodes,
    _deterministic_permutation,
    _json_safe,
    _load_mask_label,
    _load_scene_windows,
    _positive_label_counts,
    _rank_auc,
    _safe_mean,
    _safe_median,
    _safe_quantile,
    _shared_jaccard,
)


CARRIER_OVERLAP_SUFFIX = "_carrier_overlap"
CARRIER_OVERLAP_SOFT25_SUFFIX = "_carrier_overlap_soft25"
CARRIER_OVERLAP_SOFT50_SUFFIX = "_carrier_overlap_soft50"
HUB_SOFT_Q005_SUFFIX = "_hubsoft_q005"
HUB_SOFT_Q010_SUFFIX = "_hubsoft_q010"
HUB_SOFT_Q020_SUFFIX = "_hubsoft_q020"
HUB_CAP32_Q020_SUFFIX = "_hubcap32_q020"
HUB_CAP64_Q020_SUFFIX = "_hubcap64_q020"
HUB_CAP128_Q020_SUFFIX = "_hubcap128_q020"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
            writer.writerow({key: _csv_cell(row.get(key)) for key in keys})


def _csv_cell(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_json_safe(value), sort_keys=True)
    return value


def _entropy_from_counts(counts: dict[int, int]) -> float:
    total = float(sum(counts.values()))
    if total <= 0.0 or len(counts) <= 1:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        p = float(count) / total
        if p > 0.0:
            entropy -= p * math.log(p)
    return float(entropy / max(math.log(float(len(counts))), 1e-12))


def _base_quality_variant(variant: str) -> str:
    text = str(variant)
    suffixes = [
        CARRIER_OVERLAP_SOFT25_SUFFIX,
        CARRIER_OVERLAP_SOFT50_SUFFIX,
        CARRIER_OVERLAP_SUFFIX,
        HUB_SOFT_Q005_SUFFIX,
        HUB_SOFT_Q010_SUFFIX,
        HUB_SOFT_Q020_SUFFIX,
        HUB_CAP32_Q020_SUFFIX,
        HUB_CAP64_Q020_SUFFIX,
        HUB_CAP128_Q020_SUFFIX,
    ]
    changed = True
    while changed:
        changed = False
        for suffix in suffixes:
            if text.endswith(suffix):
                text = text[: -len(suffix)]
                changed = True
                break
    return text


def _carrier_overlap_mode(variant: str) -> str | None:
    text = _strip_hub_suffix(str(variant))
    if text.endswith(CARRIER_OVERLAP_SOFT25_SUFFIX):
        return "soft25"
    if text.endswith(CARRIER_OVERLAP_SOFT50_SUFFIX):
        return "soft50"
    if text.endswith(CARRIER_OVERLAP_SUFFIX):
        return "hard"
    return None


def _uses_carrier_overlap(variant: str) -> bool:
    return _carrier_overlap_mode(variant) is not None


def _strip_hub_suffix(variant: str) -> str:
    text = str(variant)
    for suffix in [
        HUB_SOFT_Q005_SUFFIX,
        HUB_SOFT_Q010_SUFFIX,
        HUB_SOFT_Q020_SUFFIX,
        HUB_CAP32_Q020_SUFFIX,
        HUB_CAP64_Q020_SUFFIX,
        HUB_CAP128_Q020_SUFFIX,
    ]:
        if text.endswith(suffix):
            return text[: -len(suffix)]
    return text


def _hub_penalty_config(variant: str) -> dict[str, Any] | None:
    text = str(variant)
    if text.endswith(HUB_SOFT_Q005_SUFFIX):
        return {"mode": "sqrt", "candidate_threshold": 0.05}
    if text.endswith(HUB_SOFT_Q010_SUFFIX):
        return {"mode": "sqrt", "candidate_threshold": 0.10}
    if text.endswith(HUB_SOFT_Q020_SUFFIX):
        return {"mode": "sqrt", "candidate_threshold": 0.20}
    if text.endswith(HUB_CAP32_Q020_SUFFIX):
        return {"mode": "cap_sqrt", "candidate_threshold": 0.20, "fanout_cap": 32.0}
    if text.endswith(HUB_CAP64_Q020_SUFFIX):
        return {"mode": "cap_sqrt", "candidate_threshold": 0.20, "fanout_cap": 64.0}
    if text.endswith(HUB_CAP128_Q020_SUFFIX):
        return {"mode": "cap_sqrt", "candidate_threshold": 0.20, "fanout_cap": 128.0}
    return None


def _uses_hub_penalty(variant: str) -> bool:
    return _hub_penalty_config(variant) is not None


def _carrier_overlap_weight(local_overlap: float, mode: str | None) -> float:
    value = max(0.0, min(1.0, float(local_overlap)))
    if mode == "hard":
        return value
    if mode == "soft25":
        return 0.75 + 0.25 * value
    if mode == "soft50":
        return 0.50 + 0.50 * value
    return 1.0


def _hub_penalty_weight(fanout: int | float | None, config: dict[str, Any] | None) -> float:
    if config is None:
        return 1.0
    value = max(1.0, float(fanout or 1.0))
    if config.get("mode") == "sqrt":
        return float(1.0 / math.sqrt(value))
    if config.get("mode") == "cap_sqrt":
        cap = max(1.0, float(config.get("fanout_cap") or 1.0))
        if value <= cap:
            return 1.0
        return float(math.sqrt(cap / value))
    return 1.0


def _hub_fanout_column(variant: str) -> str:
    return f"{variant}_supporter_fanout"


def _hub_weight_column(variant: str) -> str:
    return f"{variant}_hub_weight"


def _image_area(scene: str, frame_id: int) -> int | None:
    label = _load_mask_label(scene, frame_id)
    if label is None:
        return None
    return int(label.shape[0] * label.shape[1])


def _supporter_quality_rows(
    *,
    scene: str,
    nodes: list[MaskNode],
    windows_by_index: dict[int, WindowTrace],
    min_visible_carriers: int,
    underseg_purity_threshold: float,
    low_q_threshold: float,
) -> tuple[dict[tuple[int, int], dict[str, Any]], list[dict[str, Any]]]:
    quality_by_key: dict[tuple[int, int], dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for node in nodes:
        split_entropy_values: list[float] = []
        outside_values: list[float] = []
        fragmentation_values: list[float] = []
        observation_count = 0
        for window_index, carrier_indices in node.inc_by_window.items():
            window = windows_by_index.get(window_index)
            if window is None:
                continue
            base_idx = np.asarray(sorted(carrier_indices), dtype=np.int64)
            if base_idx.size == 0:
                continue
            for local_index, frame_id in enumerate(window.frame_ids):
                # Reliability is defined by how the support behaves in other views.
                if int(frame_id) == int(node.frame_id):
                    continue
                labels_at_carrier = window.labels_by_frame.get(int(frame_id))
                if labels_at_carrier is None:
                    continue
                visible_idx = base_idx[window.visible[local_index, base_idx]]
                if visible_idx.size < int(min_visible_carriers):
                    continue
                observed = labels_at_carrier[visible_idx]
                counts = _positive_label_counts(observed)
                covered_count = int(sum(counts.values()))
                observation_count += 1
                split_entropy_values.append(_entropy_from_counts(counts))
                fragmentation_values.append(1.0 if len(counts) >= 2 else 0.0)
                outside_values.append(float(1.0 - covered_count / max(int(visible_idx.size), 1)))
        split_entropy = float(np.mean(split_entropy_values)) if split_entropy_values else 0.0
        visible_outside = float(np.mean(outside_values)) if outside_values else 1.0
        fragmentation = float(np.mean(fragmentation_values)) if fragmentation_values else 0.0
        q_split = math.exp(-1.50 * split_entropy)
        q_full_soft = math.exp(-1.50 * split_entropy - 0.75 * visible_outside - 0.75 * fragmentation)
        support_density = float(node.support_density)
        support_count = float(node.support_count)
        density_factor_006 = max(0.0, min(1.0, support_density / 0.006))
        density_factor_008 = max(0.0, min(1.0, support_density / 0.008))
        support_count_factor_300 = max(0.0, min(1.0, support_count / 300.0))
        support_count_factor_500 = max(0.0, min(1.0, support_count / 500.0))
        q_rows = {
            "Q0_no_filter": 1.0,
            "Q1_split_entropy_soft": float(max(0.0, min(1.0, q_split))),
            "Q5_split_outside_fragment_soft": float(max(0.0, min(1.0, q_full_soft))),
            "Q5_threshold_055": float(q_full_soft if q_full_soft >= 0.55 else 0.0),
            "Q5_threshold_070": float(q_full_soft if q_full_soft >= 0.70 else 0.0),
            "Q6_density_soft_0p006_c300": float(
                max(0.0, min(1.0, q_full_soft * density_factor_006 * support_count_factor_300))
            ),
            "Q6_density_soft_0p008_c500": float(
                max(0.0, min(1.0, q_full_soft * density_factor_008 * support_count_factor_500))
            ),
            "Q6_density_hard_0p006_c300": float(
                q_full_soft if support_density >= 0.006 and support_count >= 300.0 else 0.0
            ),
            "Q6_density_hard_0p008_c500": float(
                q_full_soft if support_density >= 0.008 and support_count >= 500.0 else 0.0
            ),
        }
        area_total = _image_area(scene, node.frame_id)
        diagnostic_underseg = (
            None
            if node.dominant_gt_purity is None
            else bool(float(node.dominant_gt_purity) < float(underseg_purity_threshold))
        )
        row = {
            "scene": scene,
            "node_id": node.node_id,
            "frame_id": node.frame_id,
            "mask_id": node.mask_id,
            "area": node.area,
            "area_fraction": None if area_total is None else float(node.area / max(area_total, 1)),
            "carrier_support_count": node.support_count,
            "support_density": node.support_density,
            "observer_quality_sample_count": observation_count,
            "split_entropy": split_entropy,
            "visible_outside": visible_outside,
            "fragmentation_rate": fragmentation,
            "diagnostic_gt_instance": node.dominant_gt,
            "diagnostic_gt_purity": node.dominant_gt_purity,
            "diagnostic_underseg_gt_purity_lt_threshold": diagnostic_underseg,
            "low_q_threshold": float(low_q_threshold),
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic_labels": True,
            **q_rows,
        }
        quality_by_key[(node.frame_id, node.mask_id)] = row
        rows.append(row)
    return quality_by_key, rows


def _weighted_view_consensus(
    left: MaskNode,
    right: MaskNode,
    windows_by_index: dict[int, WindowTrace],
    quality_by_key: dict[tuple[int, int], dict[str, Any]],
    *,
    variant: str,
    min_visible_carriers: int,
    observer_frame_mode: str,
    near_endpoint_frame_gap: int,
    supporter_fanout: dict[tuple[int, int], int] | None = None,
) -> tuple[float, int, float, float]:
    observer_scores: list[float] = []
    q_values_used: list[float] = []
    base_variant = _base_quality_variant(variant)
    carrier_overlap_mode = _carrier_overlap_mode(variant)
    hub_config = _hub_penalty_config(variant)
    common_windows = sorted(set(left.inc_by_window) & set(right.inc_by_window))
    for window_index in common_windows:
        window = windows_by_index[window_index]
        left_idx = np.asarray(sorted(left.inc_by_window[window_index]), dtype=np.int64)
        right_idx = np.asarray(sorted(right.inc_by_window[window_index]), dtype=np.int64)
        if left_idx.size == 0 or right_idx.size == 0:
            continue
        for local_index, frame_id in enumerate(window.frame_ids):
            if _skip_observer_frame(
                observer_frame_id=int(frame_id),
                left_frame_id=int(left.frame_id),
                right_frame_id=int(right.frame_id),
                observer_frame_mode=observer_frame_mode,
                near_endpoint_frame_gap=int(near_endpoint_frame_gap),
            ):
                continue
            labels_at_carrier = window.labels_by_frame.get(int(frame_id))
            if labels_at_carrier is None:
                continue
            left_visible = left_idx[window.visible[local_index, left_idx]]
            right_visible = right_idx[window.visible[local_index, right_idx]]
            if left_visible.size < int(min_visible_carriers) or right_visible.size < int(min_visible_carriers):
                continue
            left_counts = _positive_label_counts(labels_at_carrier[left_visible])
            right_counts = _positive_label_counts(labels_at_carrier[right_visible])
            if not left_counts or not right_counts:
                observer_scores.append(0.0)
                q_values_used.append(0.0)
                continue
            score = 0.0
            best_q = 0.0
            for label in set(left_counts) & set(right_counts):
                quality = quality_by_key.get((int(frame_id), int(label)), {})
                q = float(quality.get(base_variant, 0.0))
                local_overlap = 1.0
                if carrier_overlap_mode is not None:
                    left_label_idx = left_visible[labels_at_carrier[left_visible] == int(label)]
                    right_label_idx = right_visible[labels_at_carrier[right_visible] == int(label)]
                    if left_label_idx.size == 0 or right_label_idx.size == 0:
                        local_overlap = 0.0
                    else:
                        common_carriers = np.intersect1d(left_label_idx, right_label_idx, assume_unique=False)
                        local_overlap = float(common_carriers.size / max(min(left_label_idx.size, right_label_idx.size), 1))
                hub_weight = _hub_penalty_weight(
                    (supporter_fanout or {}).get((int(frame_id), int(label)), 1),
                    hub_config,
                )
                candidate = (
                    q
                    * min(left_counts[label] / left_visible.size, right_counts[label] / right_visible.size)
                    * _carrier_overlap_weight(local_overlap, carrier_overlap_mode)
                    * hub_weight
                )
                if candidate > score:
                    score = float(candidate)
                    best_q = float(q * hub_weight)
            observer_scores.append(float(score))
            q_values_used.append(float(best_q))
    if not observer_scores:
        return 0.0, 0, 0.0, 0.0
    return (
        float(np.mean(observer_scores)),
        int(len(observer_scores)),
        float(np.max(observer_scores)),
        float(np.mean(q_values_used)),
    )


def _skip_observer_frame(
    *,
    observer_frame_id: int,
    left_frame_id: int,
    right_frame_id: int,
    observer_frame_mode: str,
    near_endpoint_frame_gap: int,
) -> bool:
    if observer_frame_mode == "all":
        return False
    if observer_frame_mode == "exclude_endpoints":
        return int(observer_frame_id) in {int(left_frame_id), int(right_frame_id)}
    if observer_frame_mode == "exclude_near_endpoints":
        return (
            abs(int(observer_frame_id) - int(left_frame_id)) <= int(near_endpoint_frame_gap)
            or abs(int(observer_frame_id) - int(right_frame_id)) <= int(near_endpoint_frame_gap)
        )
    raise ValueError(f"unsupported observer_frame_mode: {observer_frame_mode}")


def _supporter_fanout_for_variant(
    *,
    capped_nodes: list[MaskNode],
    windows_by_index: dict[int, WindowTrace],
    quality_by_key: dict[tuple[int, int], dict[str, Any]],
    variant: str,
    min_visible_carriers: int,
    observer_frame_mode: str,
    near_endpoint_frame_gap: int,
) -> dict[tuple[int, int], int]:
    config = _hub_penalty_config(variant)
    if config is None:
        return {}
    base_variant = _base_quality_variant(variant)
    carrier_overlap_mode = _carrier_overlap_mode(variant)
    threshold = float(config["candidate_threshold"])
    edge_sets_by_supporter: dict[tuple[int, int], set[tuple[int, int]]] = defaultdict(set)
    for i, left in enumerate(capped_nodes):
        for right in capped_nodes[i + 1 :]:
            edge_key = (int(left.node_id), int(right.node_id))
            common_windows = sorted(set(left.inc_by_window) & set(right.inc_by_window))
            for window_index in common_windows:
                window = windows_by_index[window_index]
                left_idx = np.asarray(sorted(left.inc_by_window[window_index]), dtype=np.int64)
                right_idx = np.asarray(sorted(right.inc_by_window[window_index]), dtype=np.int64)
                if left_idx.size == 0 or right_idx.size == 0:
                    continue
                for local_index, frame_id in enumerate(window.frame_ids):
                    if _skip_observer_frame(
                        observer_frame_id=int(frame_id),
                        left_frame_id=int(left.frame_id),
                        right_frame_id=int(right.frame_id),
                        observer_frame_mode=observer_frame_mode,
                        near_endpoint_frame_gap=int(near_endpoint_frame_gap),
                    ):
                        continue
                    labels_at_carrier = window.labels_by_frame.get(int(frame_id))
                    if labels_at_carrier is None:
                        continue
                    left_visible = left_idx[window.visible[local_index, left_idx]]
                    right_visible = right_idx[window.visible[local_index, right_idx]]
                    if left_visible.size < int(min_visible_carriers) or right_visible.size < int(min_visible_carriers):
                        continue
                    left_counts = _positive_label_counts(labels_at_carrier[left_visible])
                    right_counts = _positive_label_counts(labels_at_carrier[right_visible])
                    if not left_counts or not right_counts:
                        continue
                    for label in set(left_counts) & set(right_counts):
                        quality = quality_by_key.get((int(frame_id), int(label)), {})
                        q = float(quality.get(base_variant, 0.0))
                        local_overlap = 1.0
                        if carrier_overlap_mode is not None:
                            left_label_idx = left_visible[labels_at_carrier[left_visible] == int(label)]
                            right_label_idx = right_visible[labels_at_carrier[right_visible] == int(label)]
                            if left_label_idx.size == 0 or right_label_idx.size == 0:
                                local_overlap = 0.0
                            else:
                                common_carriers = np.intersect1d(left_label_idx, right_label_idx, assume_unique=False)
                                local_overlap = float(
                                    common_carriers.size / max(min(left_label_idx.size, right_label_idx.size), 1)
                                )
                        candidate = (
                            q
                            * min(left_counts[label] / left_visible.size, right_counts[label] / right_visible.size)
                            * _carrier_overlap_weight(local_overlap, carrier_overlap_mode)
                        )
                        if candidate >= threshold:
                            edge_sets_by_supporter[(int(frame_id), int(label))].add(edge_key)
    return {key: len(edge_set) for key, edge_set in edge_sets_by_supporter.items()}


def _precision_at_k(rows: list[dict[str, Any]], score_key: str, k: int) -> float | None:
    labeled = [row for row in rows if row.get("diagnostic_same_gt") is not None]
    ranked = sorted(labeled, key=lambda row: float(row.get(score_key) or 0.0), reverse=True)[: min(int(k), len(labeled))]
    if not ranked:
        return None
    return float(sum(1 for row in ranked if row.get("diagnostic_same_gt") is True) / len(ranked))


def _pollution_at_k(rows: list[dict[str, Any]], score_key: str, k: int) -> float | None:
    precision = _precision_at_k(rows, score_key, k)
    return None if precision is None else float(1.0 - precision)


def _same_frame_false_rate_at_k(rows: list[dict[str, Any]], score_key: str, k: int) -> float | None:
    labeled = [row for row in rows if row.get("diagnostic_same_gt") is not None]
    ranked = sorted(labeled, key=lambda row: float(row.get(score_key) or 0.0), reverse=True)[: min(int(k), len(labeled))]
    if not ranked:
        return None
    false_same_frame = [
        row
        for row in ranked
        if row.get("diagnostic_same_gt") is False and int(row["left_frame_id"]) == int(row["right_frame_id"])
    ]
    return float(len(false_same_frame) / max(len(ranked), 1))


def _supporter_detection_metrics(
    quality_rows: list[dict[str, Any]],
    *,
    variant: str,
    low_q_threshold: float,
) -> dict[str, Any]:
    base_variant = _base_quality_variant(variant)
    diagnostic = [row for row in quality_rows if row.get("diagnostic_underseg_gt_purity_lt_threshold") is not None]
    low_q = [row for row in diagnostic if float(row.get(base_variant) or 0.0) <= float(low_q_threshold)]
    underseg = [row for row in diagnostic if row.get("diagnostic_underseg_gt_purity_lt_threshold") is True]
    low_and_underseg = [row for row in low_q if row.get("diagnostic_underseg_gt_purity_lt_threshold") is True]
    return {
        "underseg_supporter_precision": None if not low_q else float(len(low_and_underseg) / len(low_q)),
        "underseg_supporter_recall": None if not underseg else float(len(low_and_underseg) / len(underseg)),
        "low_reliability_mask_ratio": None if not diagnostic else float(len(low_q) / len(diagnostic)),
    }


def _edge_rows_for_variants(
    *,
    scene: str,
    nodes: list[MaskNode],
    windows: list[WindowTrace],
    quality_by_key: dict[tuple[int, int], dict[str, Any]],
    variants: list[str],
    max_edge_nodes: int,
    min_node_carriers: int,
    min_visible_carriers: int,
    observer_frame_mode: str,
    near_endpoint_frame_gap: int,
) -> list[dict[str, Any]]:
    eligible = [
        node
        for node in nodes
        if node.support_count >= int(min_node_carriers) and node.dominant_gt is not None and node.dominant_gt_purity is not None
    ]
    eligible.sort(key=lambda node: (node.support_count, node.area), reverse=True)
    capped = eligible[: int(max_edge_nodes)]
    windows_by_index = {window.window_index: window for window in windows}
    permuted_indices = _deterministic_permutation(capped, seed_text=f"v46_supporter_quality_shuffle:{scene}") if capped else []
    shuffled_by_node_id: dict[int, MaskNode] = {}
    for idx, node in enumerate(capped):
        shuffled_by_node_id[node.node_id] = capped[permuted_indices[idx % len(permuted_indices)]]
    fanout_by_variant: dict[str, dict[tuple[int, int], int]] = {}
    for variant in variants:
        if not _uses_hub_penalty(variant):
            continue
        fanout = _supporter_fanout_for_variant(
            capped_nodes=capped,
            windows_by_index=windows_by_index,
            quality_by_key=quality_by_key,
            variant=variant,
            min_visible_carriers=int(min_visible_carriers),
            observer_frame_mode=observer_frame_mode,
            near_endpoint_frame_gap=int(near_endpoint_frame_gap),
        )
        fanout_by_variant[variant] = fanout
        config = _hub_penalty_config(variant)
        for key, quality in quality_by_key.items():
            supporter_fanout = int(fanout.get(key, 0))
            quality[_hub_fanout_column(variant)] = supporter_fanout
            quality[_hub_weight_column(variant)] = _hub_penalty_weight(max(supporter_fanout, 1), config)
    rows: list[dict[str, Any]] = []
    for i, left in enumerate(capped):
        for right in capped[i + 1 :]:
            same_gt = bool(left.dominant_gt == right.dominant_gt)
            shuffled_right = shuffled_by_node_id.get(right.node_id, right)
            row: dict[str, Any] = {
                "scene": scene,
                "left_node_id": left.node_id,
                "right_node_id": right.node_id,
                "left_frame_id": left.frame_id,
                "right_frame_id": right.frame_id,
                "left_mask_id": left.mask_id,
                "right_mask_id": right.mask_id,
                "left_support_count": left.support_count,
                "right_support_count": right.support_count,
                "left_gt": left.dominant_gt,
                "right_gt": right.dominant_gt,
                "left_gt_purity": left.dominant_gt_purity,
                "right_gt_purity": right.dominant_gt_purity,
                "diagnostic_same_gt": same_gt,
                "shared_carrier_jaccard": _shared_jaccard(left, right),
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic_labels": True,
            }
            for variant in variants:
                vc, observer_count, vc_max, mean_q = _weighted_view_consensus(
                    left,
                    right,
                    windows_by_index,
                    quality_by_key,
                    variant=variant,
                    min_visible_carriers=int(min_visible_carriers),
                    observer_frame_mode=observer_frame_mode,
                    near_endpoint_frame_gap=int(near_endpoint_frame_gap),
                    supporter_fanout=fanout_by_variant.get(variant),
                )
                shuffled_vc, shuffled_observer_count, _shuffled_max, shuffled_mean_q = _weighted_view_consensus(
                    left,
                    shuffled_right,
                    windows_by_index,
                    quality_by_key,
                    variant=variant,
                    min_visible_carriers=int(min_visible_carriers),
                    observer_frame_mode=observer_frame_mode,
                    near_endpoint_frame_gap=int(near_endpoint_frame_gap),
                    supporter_fanout=fanout_by_variant.get(variant),
                )
                row[f"{variant}_view_consensus"] = vc
                row[f"{variant}_view_consensus_max_observer"] = vc_max
                row[f"{variant}_observer_count"] = observer_count
                row[f"{variant}_mean_supporter_q_used"] = mean_q
                row[f"{variant}_shuffled_view_consensus"] = shuffled_vc
                row[f"{variant}_shuffled_observer_count"] = shuffled_observer_count
                row[f"{variant}_shuffled_mean_supporter_q_used"] = shuffled_mean_q
            rows.append(row)
    return rows


def _summarize_edges(
    *,
    scene: str,
    edge_rows: list[dict[str, Any]],
    quality_rows: list[dict[str, Any]],
    variants: list[str],
    low_q_threshold: float,
) -> list[dict[str, Any]]:
    labels = [bool(row["diagnostic_same_gt"]) for row in edge_rows]
    shared_scores = [float(row["shared_carrier_jaccard"]) for row in edge_rows]
    shared_auc = _rank_auc(labels, shared_scores)
    shared_p1 = _precision_at_k(edge_rows, "shared_carrier_jaccard", 1000)
    shared_p5 = _precision_at_k(edge_rows, "shared_carrier_jaccard", 5000)
    q0_pollution = _pollution_at_k(edge_rows, "Q0_no_filter_view_consensus", 5000)
    q0_same_frame_false = _same_frame_false_rate_at_k(edge_rows, "Q0_no_filter_view_consensus", 5000)
    rows: list[dict[str, Any]] = [
        {
            "scene": scene,
            "variant": "P0_shared_carrier_jaccard",
            "score_key": "shared_carrier_jaccard",
            "edge_count": len(edge_rows),
            "edge_same_gt_AUC": shared_auc,
            "edge_precision@top1k": shared_p1,
            "edge_precision@top5k": shared_p5,
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic_labels": True,
        }
    ]
    for variant in variants:
        base_variant = _base_quality_variant(variant)
        score_key = f"{variant}_view_consensus"
        shuffled_key = f"{variant}_shuffled_view_consensus"
        scores = [float(row[score_key]) for row in edge_rows]
        shuffled_scores = [float(row[shuffled_key]) for row in edge_rows]
        auc = _rank_auc(labels, scores)
        shuffled_auc = _rank_auc(labels, shuffled_scores)
        p1 = _precision_at_k(edge_rows, score_key, 1000)
        p5 = _precision_at_k(edge_rows, score_key, 5000)
        pollution = _pollution_at_k(edge_rows, score_key, 5000)
        same_frame_false = _same_frame_false_rate_at_k(edge_rows, score_key, 5000)
        supporter_metrics = _supporter_detection_metrics(quality_rows, variant=variant, low_q_threshold=float(low_q_threshold))
        q_values = [float(row.get(base_variant) or 0.0) for row in quality_rows]
        hub_config = _hub_penalty_config(variant)
        hub_fanouts = [float(row.get(_hub_fanout_column(variant)) or 0.0) for row in quality_rows] if hub_config else []
        hub_weights = [float(row.get(_hub_weight_column(variant)) or 1.0) for row in quality_rows] if hub_config else []
        hub_adjusted_q_values = (
            [float(row.get(base_variant) or 0.0) * float(row.get(_hub_weight_column(variant)) or 1.0) for row in quality_rows]
            if hub_config
            else []
        )
        observer_counts = [float(row.get(f"{variant}_observer_count") or 0.0) for row in edge_rows]
        raw_auc_margin = None if auc is None or shared_auc is None else float(auc - shared_auc)
        shuffled_margin = None if auc is None or shuffled_auc is None else float(auc - shuffled_auc)
        p5_margin = None if p5 is None or shared_p5 is None else float(p5 - shared_p5)
        rows.append(
            {
                "scene": scene,
                "variant": variant,
                "score_key": score_key,
                "edge_count": len(edge_rows),
                "mean_observer_count": _safe_mean(observer_counts),
                "median_observer_count": _safe_median(observer_counts),
                "supporter_reliability_mean": _safe_mean(q_values),
                "supporter_reliability_p10": _safe_quantile(q_values, 0.10),
                "supporter_reliability_min": min(q_values) if q_values else None,
                "supporter_hub_candidate_threshold": None if hub_config is None else float(hub_config["candidate_threshold"]),
                "supporter_hub_fanout_cap": None if hub_config is None else hub_config.get("fanout_cap"),
                "supporter_hub_fanout_mean": _safe_mean(hub_fanouts) if hub_config else None,
                "supporter_hub_fanout_p90": _safe_quantile(hub_fanouts, 0.90) if hub_config else None,
                "supporter_hub_fanout_max": max(hub_fanouts) if hub_fanouts else None,
                "supporter_hub_weight_mean": _safe_mean(hub_weights) if hub_config else None,
                "supporter_hub_adjusted_reliability_mean": _safe_mean(hub_adjusted_q_values) if hub_config else None,
                "split_entropy_mean": _safe_mean(row["split_entropy"] for row in quality_rows),
                "visible_outside_mean": _safe_mean(row["visible_outside"] for row in quality_rows),
                "fragmentation_rate_mean": _safe_mean(row["fragmentation_rate"] for row in quality_rows),
                "reliable_supporter_count": sum(1 for value in q_values if value > float(low_q_threshold)),
                "reliable_supporter_drop_vs_q0": float(1.0 - sum(1 for value in q_values if value > float(low_q_threshold)) / max(len(q_values), 1)),
                "underseg_supporter_precision": supporter_metrics["underseg_supporter_precision"],
                "underseg_supporter_recall": supporter_metrics["underseg_supporter_recall"],
                "low_reliability_mask_ratio": supporter_metrics["low_reliability_mask_ratio"],
                "view_consensus_mean": _safe_mean(scores),
                "view_consensus_p90": _safe_quantile(scores, 0.90),
                "shuffled_view_consensus_mean": _safe_mean(shuffled_scores),
                "edge_same_gt_AUC": auc,
                "shared_carrier_jaccard_AUC": shared_auc,
                "shuffled_edge_same_gt_AUC": shuffled_auc,
                "edge_precision@top1k": p1,
                "edge_precision@top5k": p5,
                "shared_precision@top5k": shared_p5,
                "real_minus_shared_edge_AUC": raw_auc_margin,
                "real_minus_shuffled_edge_AUC": shuffled_margin,
                "precision_top5k_minus_shared": p5_margin,
                "positive_edge_pollution_rate": pollution,
                "q0_positive_edge_pollution_rate": q0_pollution,
                "positive_edge_pollution_delta_vs_q0": None if pollution is None or q0_pollution is None else float(pollution - q0_pollution),
                "same_frame_false_merge_supporter_rate": same_frame_false,
                "q0_same_frame_false_merge_supporter_rate": q0_same_frame_false,
                "same_frame_false_merge_delta_vs_q0": None
                if same_frame_false is None or q0_same_frame_false is None
                else float(same_frame_false - q0_same_frame_false),
                "gate_pass": bool(
                    raw_auc_margin is not None
                    and raw_auc_margin >= 0.08
                    and shuffled_margin is not None
                    and shuffled_margin >= 0.10
                    and p5_margin is not None
                    and p5_margin >= 0.10
                ),
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic_labels": True,
            }
        )
    return rows


def _scene_payload(
    *,
    scene: str,
    carrier_cache_root: Path,
    visibility_threshold: float,
    confidence_threshold: float,
    min_mask_area: int,
    max_edge_nodes: int,
    min_node_carriers: int,
    min_visible_carriers: int,
    underseg_purity_threshold: float,
    low_q_threshold: float,
    variants: list[str],
    observer_frame_mode: str,
    near_endpoint_frame_gap: int,
) -> dict[str, Any]:
    windows, window_rows, manifest_diag = _load_scene_windows(
        scene=scene,
        carrier_cache_root=carrier_cache_root,
        visibility_threshold=float(visibility_threshold),
        confidence_threshold=float(confidence_threshold),
        min_mask_area=int(min_mask_area),
    )
    nodes, frame_rows, node_diag = _build_nodes(scene, windows, min_mask_area=int(min_mask_area))
    windows_by_index = {window.window_index: window for window in windows}
    quality_by_key, quality_rows = _supporter_quality_rows(
        scene=scene,
        nodes=nodes,
        windows_by_index=windows_by_index,
        min_visible_carriers=int(min_visible_carriers),
        underseg_purity_threshold=float(underseg_purity_threshold),
        low_q_threshold=float(low_q_threshold),
    )
    edge_rows = _edge_rows_for_variants(
        scene=scene,
        nodes=nodes,
        windows=windows,
        quality_by_key=quality_by_key,
        variants=variants,
        max_edge_nodes=int(max_edge_nodes),
        min_node_carriers=int(min_node_carriers),
        min_visible_carriers=int(min_visible_carriers),
        observer_frame_mode=observer_frame_mode,
        near_endpoint_frame_gap=int(near_endpoint_frame_gap),
    )
    summary_rows = _summarize_edges(
        scene=scene,
        edge_rows=edge_rows,
        quality_rows=quality_rows,
        variants=variants,
        low_q_threshold=float(low_q_threshold),
    )
    return {
        "quality_rows": quality_rows,
        "edge_rows": edge_rows,
        "summary_rows": summary_rows,
        "frame_rows": frame_rows,
        "window_rows": window_rows,
        "diag": {**manifest_diag, **node_diag},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Raw-cache supporter quality repair for v46 D4RT view-consensus edges.")
    parser.add_argument("--carrier-cache-root", required=True)
    parser.add_argument("--scenes", default="scene0081_01,scene0591_00")
    parser.add_argument("--visibility-threshold", type=float, default=0.3)
    parser.add_argument("--confidence-threshold", type=float, default=0.3)
    parser.add_argument("--min-mask-area", type=int, default=64)
    parser.add_argument("--max-edge-nodes", type=int, default=120)
    parser.add_argument("--min-node-carriers", type=int, default=5)
    parser.add_argument("--min-visible-carriers", type=int, default=3)
    parser.add_argument("--underseg-purity-threshold", type=float, default=0.70)
    parser.add_argument("--low-q-threshold", type=float, default=0.55)
    parser.add_argument("--observer-frame-mode", choices=["all", "exclude_endpoints", "exclude_near_endpoints"], default="all")
    parser.add_argument("--near-endpoint-frame-gap", type=int, default=10)
    parser.add_argument(
        "--variants",
        default="Q0_no_filter,Q1_split_entropy_soft,Q5_split_outside_fragment_soft,Q5_threshold_055,Q5_threshold_070",
    )
    parser.add_argument("--output-root", default="outputs/audit/v46_supporter_quality_raw_repair")
    args = parser.parse_args()

    carrier_cache_root = ROOT / str(args.carrier_cache_root)
    scenes = [item.strip() for item in str(args.scenes).split(",") if item.strip()]
    variants = [item.strip() for item in str(args.variants).split(",") if item.strip()]
    all_quality_rows: list[dict[str, Any]] = []
    all_edge_rows: list[dict[str, Any]] = []
    all_summary_rows: list[dict[str, Any]] = []
    all_frame_rows: list[dict[str, Any]] = []
    all_window_rows: list[dict[str, Any]] = []
    diags: dict[str, Any] = {}
    for scene in scenes:
        payload = _scene_payload(
            scene=scene,
            carrier_cache_root=carrier_cache_root,
            visibility_threshold=float(args.visibility_threshold),
            confidence_threshold=float(args.confidence_threshold),
            min_mask_area=int(args.min_mask_area),
            max_edge_nodes=int(args.max_edge_nodes),
            min_node_carriers=int(args.min_node_carriers),
            min_visible_carriers=int(args.min_visible_carriers),
            underseg_purity_threshold=float(args.underseg_purity_threshold),
            low_q_threshold=float(args.low_q_threshold),
            variants=variants,
            observer_frame_mode=str(args.observer_frame_mode),
            near_endpoint_frame_gap=int(args.near_endpoint_frame_gap),
        )
        all_quality_rows.extend(payload["quality_rows"])
        all_edge_rows.extend(payload["edge_rows"])
        all_summary_rows.extend(payload["summary_rows"])
        all_frame_rows.extend(payload["frame_rows"])
        all_window_rows.extend(payload["window_rows"])
        diags[scene] = payload["diag"]
    raw_variant_rows = [row for row in all_summary_rows if str(row.get("variant")) != "P0_shared_carrier_jaccard"]
    gate = {
        "any_scene_variant_gate_pass": any(bool(row.get("gate_pass")) for row in raw_variant_rows),
        "all_scene_variant_gate_pass": False,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }
    for variant in variants:
        selected = [row for row in raw_variant_rows if str(row.get("variant")) == variant]
        if selected and all(bool(row.get("gate_pass")) for row in selected):
            gate["all_scene_variant_gate_pass"] = True
    gate["pass"] = bool(gate["all_scene_variant_gate_pass"])
    payload = {
        "phase": "v46_supporter_quality_raw_repair",
        "created_at": _utc_now(),
        "carrier_cache_root": str(carrier_cache_root.relative_to(ROOT) if carrier_cache_root.is_relative_to(ROOT) else carrier_cache_root),
        "scenes": scenes,
        "visibility_threshold": float(args.visibility_threshold),
        "confidence_threshold": float(args.confidence_threshold),
        "min_mask_area": int(args.min_mask_area),
        "max_edge_nodes": int(args.max_edge_nodes),
        "min_node_carriers": int(args.min_node_carriers),
        "min_visible_carriers": int(args.min_visible_carriers),
        "underseg_purity_threshold": float(args.underseg_purity_threshold),
        "low_q_threshold": float(args.low_q_threshold),
        "observer_frame_mode": str(args.observer_frame_mode),
        "near_endpoint_frame_gap": int(args.near_endpoint_frame_gap),
        "variants": variants,
        "summary_rows": all_summary_rows,
        "diag": diags,
        "gate": gate,
    }
    out = ROOT / str(args.output_root)
    _write_json(out / "supporter_quality_raw_repair.json", payload)
    _write_csv(out / "supporter_quality_rows.csv", all_quality_rows)
    _write_csv(out / "supporter_quality_edge_rows.csv", all_edge_rows)
    _write_csv(out / "supporter_quality_summary_rows.csv", all_summary_rows)
    _write_csv(out / "supporter_quality_frame_rows.csv", all_frame_rows)
    _write_csv(out / "supporter_quality_window_rows.csv", all_window_rows)
    print(json.dumps({"summary": str(out / "supporter_quality_raw_repair.json"), "gate": gate}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
