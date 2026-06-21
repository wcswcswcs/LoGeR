from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .v47_common import ROOT, parse_bool, parse_float, parse_int, read_csv, read_json, utc_now, write_csv, write_json
from .v53_local_objectlets import _component_scene, _evaluate_variant, _load_component_ids


def _project(path: str | Path) -> Path:
    path_obj = Path(path)
    return path_obj if path_obj.is_absolute() else ROOT / path_obj


def _rel(path: str | Path) -> str:
    path_obj = _project(path)
    try:
        return str(path_obj.relative_to(ROOT))
    except ValueError:
        return str(path_obj)


def _load_best_variant(summary_path: str | Path, fallback: str) -> str:
    path = _project(summary_path)
    if not path.exists():
        return fallback
    payload = read_json(path)
    if not isinstance(payload, dict):
        return fallback
    best = payload.get("best_real_row", {}) if isinstance(payload.get("best_real_row"), dict) else {}
    return str(payload.get("best_real_variant") or best.get("variant") or fallback)


def _load_vector(value: Any) -> list[float]:
    text = str(value or "").strip()
    if not text:
        return []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    out: list[float] = []
    for item in payload:
        try:
            out.append(float(item))
        except (TypeError, ValueError):
            return []
    return out


def _mean_vector(items: list[tuple[list[float], float]]) -> tuple[list[float], float]:
    valid = [(vec, float(weight)) for vec, weight in items if vec and float(weight) > 0.0]
    if not valid:
        return [], 0.0
    dim = len(valid[0][0])
    valid = [(vec, weight) for vec, weight in valid if len(vec) == dim]
    if not valid:
        return [], 0.0
    total = sum(weight for _vec, weight in valid)
    arr = sum(np.asarray(vec, dtype=np.float64) * weight for vec, weight in valid) / max(total, 1e-12)
    return arr.astype(float).tolist(), float(total)


def _variance_to_mean(items: list[tuple[list[float], float]], mean: list[float]) -> float | None:
    if not mean:
        return None
    mean_arr = np.asarray(mean, dtype=np.float64)
    distances: list[float] = []
    for vec, weight in items:
        if not vec or len(vec) != mean_arr.shape[0] or float(weight) <= 0.0:
            continue
        dist = float(np.linalg.norm(np.asarray(vec, dtype=np.float64) - mean_arr))
        distances.extend([dist] * max(parse_int(weight), 1))
    return float(np.mean(distances)) if distances else None


def _entropy(counts: Counter[str]) -> float:
    total = float(sum(counts.values()))
    if total <= 0.0:
        return 0.0
    return float(-sum((count / total) * np.log(count / total) for count in counts.values() if count > 0))


def _auc(labels: list[bool], scores: list[float]) -> float | None:
    positives = [score for label, score in zip(labels, scores) if label]
    negatives = [score for label, score in zip(labels, scores) if not label]
    if not positives or not negatives:
        return None
    wins = 0.0
    total = 0.0
    for pos in positives:
        for neg in negatives:
            total += 1.0
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
    return float(wins / max(total, 1.0))


def _component_features(
    *,
    support_rows: list[dict[str, Any]],
    mask_rows_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    feature_items_by_component: dict[str, list[tuple[list[float], float]]] = defaultdict(list)
    gt_counts_by_component: dict[str, Counter[str]] = defaultdict(Counter)
    support_weight_by_component: Counter[str] = Counter()
    mask_count_by_component: Counter[str] = Counter()
    scene_by_component: dict[str, Counter[str]] = defaultdict(Counter)
    backend_counts: Counter[str] = Counter()
    for row in support_rows:
        component_id = str(row.get("component_id"))
        mask_id = str(row.get("mask_observation_id"))
        support_count = float(parse_int(row.get("support_count"), 1))
        mask_row = mask_rows_by_id.get(mask_id, {})
        vector = _load_vector(mask_row.get("core_feature"))
        if vector and parse_bool(mask_row.get("core_feature_valid", True)):
            feature_items_by_component[component_id].append((vector, support_count))
            backend_counts[str(mask_row.get("feature_backend") or "unknown")] += 1
            mask_count_by_component[component_id] += 1
        support_weight_by_component[component_id] += int(support_count)
        scene_by_component[component_id][str(row.get("scene"))] += int(support_count)
        gt = str(row.get("diagnostic_gt_instance") or "")
        if gt:
            gt_counts_by_component[component_id][gt] += int(support_count)

    rows: list[dict[str, Any]] = []
    for component_id in sorted(support_weight_by_component):
        items = feature_items_by_component.get(component_id, [])
        mean, weight = _mean_vector(items)
        variance = _variance_to_mean(items, mean)
        gt_counts = gt_counts_by_component.get(component_id, Counter())
        dominant_gt, dominant_count = gt_counts.most_common(1)[0] if gt_counts else ("", 0)
        rows.append(
            {
                "component_id": component_id,
                "scene": scene_by_component[component_id].most_common(1)[0][0] if scene_by_component[component_id] else "",
                "feature_available": bool(mean),
                "feature_backend": backend_counts.most_common(1)[0][0] if backend_counts else "colorhist_fallback",
                "component_mean_feature": mean,
                "component_feature_variance": variance,
                "feature_support_weight": weight,
                "feature_mask_count": int(mask_count_by_component.get(component_id, 0)),
                "support_weight": int(support_weight_by_component.get(component_id, 0)),
                "diagnostic_gt_entropy": _entropy(gt_counts),
                "dominant_gt_instance": dominant_gt,
                "dominant_gt_ratio": float(dominant_count / max(sum(gt_counts.values()), 1)),
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic_labels": True,
            }
        )
    return rows


def _objectlet_semantic_rows(
    objectlet_rows: list[dict[str, Any]],
    component_rows_by_id: dict[str, dict[str, Any]],
    variant: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for objectlet in objectlet_rows:
        if str(objectlet.get("variant")) != str(variant):
            continue
        components = _load_component_ids(objectlet.get("component_ids"))
        feature_items: list[tuple[list[float], float]] = []
        gt_counts: Counter[str] = Counter()
        for component_id in components:
            component = component_rows_by_id.get(component_id)
            if not component:
                continue
            vector = component.get("component_mean_feature")
            if isinstance(vector, str):
                vector = _load_vector(vector)
            if vector:
                feature_items.append((list(vector), parse_float(component.get("feature_support_weight"), 1.0)))
            gt = str(component.get("dominant_gt_instance") or "")
            if gt:
                gt_counts[gt] += parse_int(component.get("support_weight"), 1)
        prototype, feature_weight = _mean_vector(feature_items)
        distances: list[float] = []
        if prototype:
            proto = np.asarray(prototype, dtype=np.float64)
            for vector, _weight in feature_items:
                if len(vector) == proto.shape[0]:
                    distances.append(float(np.linalg.norm(np.asarray(vector, dtype=np.float64) - proto)))
        dominant_gt, dominant_count = gt_counts.most_common(1)[0] if gt_counts else ("", 0)
        component_count = len(components)
        feature_count = len(feature_items)
        semantic_diversity = float(np.mean(distances)) if distances else None
        contradiction_score = float(max(distances)) if distances else None
        dominant_ratio = float(dominant_count / max(sum(gt_counts.values()), 1))
        rows.append(
            {
                "variant": variant,
                "objectlet_id": objectlet.get("objectlet_id"),
                "scene": objectlet.get("scene"),
                "chunk_id": objectlet.get("chunk_id"),
                "candidate_id": objectlet.get("candidate_id"),
                "source_mask_observation_id": objectlet.get("source_mask_observation_id"),
                "component_count": component_count,
                "component_feature_count": feature_count,
                "objectlet_feature_success_rate": float(feature_count / max(component_count, 1)),
                "objectlet_semantic_prototype": prototype,
                "objectlet_semantic_diversity": semantic_diversity,
                "semantic_contradiction_score": contradiction_score,
                "feature_support_weight": feature_weight,
                "diagnostic_gt_entropy": _entropy(gt_counts),
                "diagnostic_dominant_gt_instance": dominant_gt,
                "diagnostic_dominant_gt_ratio": dominant_ratio,
                "diagnostic_mixed_gt": dominant_ratio < 0.75 if gt_counts else False,
                "underseg_proxy": parse_bool(objectlet.get("underseg_proxy")),
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic_labels": True,
            }
        )
    return rows


def _filtered_objectlet_metrics(
    *,
    threshold: float,
    objectlet_rows: list[dict[str, Any]],
    objectlet_semantic_by_id: dict[str, dict[str, Any]],
    support_rows: list[dict[str, Any]],
    all_components: set[str],
    component_scene: dict[str, str],
) -> dict[str, Any]:
    kept_rows: list[dict[str, Any]] = []
    removed_mixed = 0
    total_mixed = 0
    for row in objectlet_rows:
        semantic = objectlet_semantic_by_id.get(str(row.get("objectlet_id")), {})
        score = parse_float(semantic.get("semantic_contradiction_score"), -1.0)
        mixed = parse_bool(semantic.get("diagnostic_mixed_gt"))
        total_mixed += int(mixed)
        if score > float(threshold):
            removed_mixed += int(mixed)
            continue
        kept = dict(row)
        kept["semantic_contradiction_score"] = semantic.get("semantic_contradiction_score")
        kept["semantic_veto_threshold"] = threshold
        kept_rows.append(kept)
    component_to_object: dict[str, str] = {}
    for row in kept_rows:
        objectlet_id = str(row.get("objectlet_id"))
        for component_id in _load_component_ids(row.get("component_ids")):
            component_to_object[component_id] = objectlet_id
    metrics = _evaluate_variant(
        f"L13_colorhist_semantic_veto_le_{threshold:.6f}",
        support_rows,
        component_to_object,
        kept_rows,
        all_components,
        component_scene,
    )
    metrics["semantic_veto_threshold"] = float(threshold)
    metrics["removed_objectlet_count"] = int(len(objectlet_rows) - len(kept_rows))
    metrics["kept_objectlet_count"] = int(len(kept_rows))
    metrics["diagnostic_mixed_objectlet_count"] = int(total_mixed)
    metrics["removed_diagnostic_mixed_objectlet_count"] = int(removed_mixed)
    metrics["false_objectlet_merge_reduction"] = float(removed_mixed / max(total_mixed, 1))
    metrics["uses_gt_for_prediction"] = False
    metrics["uses_gt_for_diagnostic_labels"] = True
    return metrics


def build_semantic_guard(
    *,
    support_rows_path: str | Path = "outputs/audit/v53_mask_component_support_tau005/mask_component_support_rows.csv",
    mask_table_path: str | Path = "outputs/audit/v47_observation_tables_metricfix/mask_observation_table.csv",
    objectlet_summary_path: str | Path = "outputs/audit/v53_local_objectlets_k0_conflict_veto025_skip_repeated_sig_l12/local_objectlet_summary.json",
    objectlet_rows_path: str | Path = "outputs/audit/v53_local_objectlets_k0_conflict_veto025_skip_repeated_sig_l12/objectlet_rows.csv",
    support_variant: str = "R0_visible_tau0.05",
    objectlet_variant: str | None = None,
) -> dict[str, Any]:
    variant = objectlet_variant or _load_best_variant(objectlet_summary_path, "L6_coverage_first_minnew025")
    support_rows = [row for row in read_csv(_project(support_rows_path)) if str(row.get("variant")) == support_variant]
    mask_rows = read_csv(_project(mask_table_path))
    objectlet_rows_all = read_csv(_project(objectlet_rows_path))
    objectlet_rows = [row for row in objectlet_rows_all if str(row.get("variant")) == str(variant)]
    mask_rows_by_id = {str(row.get("mask_observation_id")): row for row in mask_rows}
    component_rows = _component_features(support_rows=support_rows, mask_rows_by_id=mask_rows_by_id)
    component_rows_by_id = {str(row.get("component_id")): row for row in component_rows}
    objectlet_semantic = _objectlet_semantic_rows(objectlet_rows, component_rows_by_id, variant)
    objectlet_semantic_by_id = {str(row.get("objectlet_id")): row for row in objectlet_semantic}

    labels = [parse_bool(row.get("diagnostic_mixed_gt")) for row in objectlet_semantic]
    scores = [parse_float(row.get("semantic_contradiction_score")) for row in objectlet_semantic]
    underseg_labels = [parse_bool(row.get("underseg_proxy")) for row in objectlet_semantic]
    semantic_auc = _auc(labels, scores)
    underseg_auc = _auc(underseg_labels, scores)

    all_components = {str(row.get("component_id")) for row in support_rows}
    component_scene = _component_scene(support_rows)
    thresholds = sorted(
        {
            float(np.quantile(scores, q))
            for q in [0.50, 0.65, 0.75, 0.85, 0.90, 0.95]
            if scores
        }
    )
    metric_rows = [
        _filtered_objectlet_metrics(
            threshold=threshold,
            objectlet_rows=objectlet_rows,
            objectlet_semantic_by_id=objectlet_semantic_by_id,
            support_rows=support_rows,
            all_components=all_components,
            component_scene=component_scene,
        )
        for threshold in thresholds
    ]
    base_summary = read_json(_project(objectlet_summary_path)) if _project(objectlet_summary_path).exists() else {}
    base_best = base_summary.get("best_real_row", {}) if isinstance(base_summary.get("best_real_row"), dict) else {}
    for row in metric_rows:
        row["purity_change"] = parse_float(row.get("4D_purity")) - parse_float(base_best.get("4D_purity"))
        row["completeness_change"] = parse_float(row.get("4D_completeness")) - parse_float(base_best.get("4D_completeness"))
        row["ARI_change"] = parse_float(row.get("4D_ARI")) - parse_float(base_best.get("4D_ARI"))
        row["semantic_veto_candidate_pass"] = (
            parse_float(row.get("purity_change")) >= 0.005
            and parse_float(row.get("completeness_change")) >= -0.04
        )
    best_metric = max(
        metric_rows,
        key=lambda row: (
            bool(row.get("semantic_veto_candidate_pass")),
            parse_float(row.get("purity_change")),
            parse_float(row.get("4D_ARI")),
        ),
        default={},
    )
    feature_success_rate = sum(1 for row in mask_rows if _load_vector(row.get("core_feature"))) / max(len(mask_rows), 1)
    component_feature_success_rate = sum(1 for row in component_rows if parse_bool(row.get("feature_available"))) / max(len(component_rows), 1)
    objectlet_feature_success_rate = (
        float(np.mean([parse_float(row.get("objectlet_feature_success_rate")) for row in objectlet_semantic]))
        if objectlet_semantic
        else 0.0
    )
    semantic_pass = bool(
        (semantic_auc is not None and semantic_auc >= 0.70)
        or (underseg_auc is not None and underseg_auc >= 0.70)
        or parse_float(best_metric.get("false_objectlet_merge_reduction")) >= 0.10
    )
    veto_candidate_pass = bool(best_metric.get("semantic_veto_candidate_pass"))
    summary = {
        "phase": "v53_semantic_guard",
        "created_at": utc_now(),
        "support_rows_path": _rel(support_rows_path),
        "mask_table_path": _rel(mask_table_path),
        "objectlet_summary_path": _rel(objectlet_summary_path),
        "objectlet_rows_path": _rel(objectlet_rows_path),
        "support_variant": support_variant,
        "objectlet_variant": variant,
        "dense_semantic_available": False,
        "feature_backend": "colorhist_fallback",
        "semantic_claim_allowed": False,
        "feature_success_rate": feature_success_rate,
        "component_feature_success_rate": component_feature_success_rate,
        "objectlet_feature_success_rate": objectlet_feature_success_rate,
        "semantic_contradiction_AUC_diagnostic": semantic_auc,
        "underseg_detection_AUC_diagnostic": underseg_auc,
        "semantic_guard_signal_pass": semantic_pass,
        "semantic_veto_candidate_pass": veto_candidate_pass,
        "semantic_guard_method_enabled": bool(veto_candidate_pass and semantic_pass and False),
        "disable_from_method_selection": True,
        "disable_reason": (
            "dense semantic unavailable; colorhist fallback did not produce a promotable method guard"
            if not (semantic_pass and veto_candidate_pass)
            else "dense semantic unavailable, so semantic claim remains diagnostic even though fallback signal improved a metric"
        ),
        "base_best_local": {
            "variant": base_best.get("variant"),
            "4D_ARI": base_best.get("4D_ARI"),
            "4D_purity": base_best.get("4D_purity"),
            "4D_completeness": base_best.get("4D_completeness"),
            "real_minus_mask_only_ARI": base_best.get("real_minus_mask_only_ARI"),
        },
        "best_veto_metric_row": best_metric,
        "semantic_only_ran": False,
        "semantic_only_not_run_reason": "only colorhist fallback is available; plan requires no dense semantic claim",
        "DINO_vs_colorhist_delta": None,
        "RADIO_vs_colorhist_delta": None,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }
    return {
        "summary": summary,
        "component_feature_rows": component_rows,
        "objectlet_semantic_rows": objectlet_semantic,
        "semantic_metric_rows": metric_rows,
    }


def write_semantic_guard(output_root: str | Path, payload: dict[str, Any]) -> None:
    out = _project(output_root)
    write_json(out / "semantic_summary.json", payload["summary"])
    write_csv(out / "component_feature_rows.csv", payload["component_feature_rows"])
    write_csv(out / "objectlet_semantic_rows.csv", payload["objectlet_semantic_rows"])
    write_csv(out / "semantic_metric_rows.csv", payload["semantic_metric_rows"])


__all__ = ["build_semantic_guard", "write_semantic_guard"]
