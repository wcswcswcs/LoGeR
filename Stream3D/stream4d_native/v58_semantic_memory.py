from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .v47_common import ROOT, cosine, json_safe, parse_float, parse_int, rank_auc, write_csv, write_json
from .v55_history_update import _semantic_mask_feature


DEFAULT_SUPPORT_ROWS = "outputs/audit/v54_mask_component_support_tau005_stride1_probe5_q4096_notopup/mask_component_support_rows.csv"
DEFAULT_HISTORY_ROWS = "outputs/audit/v55_history_update/history_rows.csv"
DEFAULT_HISTORY_UPDATE_ROWS = "outputs/audit/v55_history_update/history_update_rows.csv"
DEFAULT_OBJECTLET_ROWS = (
    "outputs/audit/v54_local_objectlets_k0all_conflict_veto018_skip_repeated_sig_l12_stride1_probe5_q4096_notopup_max4000_skip/objectlet_rows.csv"
)


@dataclass(frozen=True)
class V58SemanticMemoryConfig:
    support_rows_path: str | Path = DEFAULT_SUPPORT_ROWS
    history_rows_path: str | Path = DEFAULT_HISTORY_ROWS
    history_update_rows_path: str | Path = DEFAULT_HISTORY_UPDATE_ROWS
    objectlet_rows_path: str | Path = DEFAULT_OBJECTLET_ROWS
    output_root: str | Path = "outputs/audit/v58_semantic_memory"
    visualization_root: str | Path = "outputs/audit/v58_visualizations/semantic_memory"
    support_variant: str = "I0_visible_tau0.10"
    objectlet_underseg_variant: str = "L11_dynamic_uncovered_gain_dup010"
    backend: str = "dinov2_timm"
    device: str = "cuda:0"
    checkpoint: str | None = None
    short_side: int = 518
    max_modes: int = 4
    max_mask_observations: int | None = None
    write_mask_feature_vectors: bool = True


def build_v58_semantic_memory(config: V58SemanticMemoryConfig | None = None) -> dict[str, Any]:
    cfg = config or V58SemanticMemoryConfig()
    support_rows = _read_csv(_project(cfg.support_rows_path))
    history_rows = _read_csv(_project(cfg.history_rows_path))
    update_rows = _read_csv(_project(cfg.history_update_rows_path))
    objectlet_rows = _read_csv(_project(cfg.objectlet_rows_path))

    support_rows = [row for row in support_rows if str(row.get("variant")) == str(cfg.support_variant)]
    selected_masks = _select_mask_observations(support_rows, max_count=cfg.max_mask_observations)
    mask_features, mask_feature_rows = _extract_mask_features(selected_masks, cfg)
    component_features, component_feature_rows = _build_component_features(support_rows, mask_features)
    history_samples, history_sample_rows = _build_history_samples(
        history_rows=history_rows,
        update_rows=update_rows,
        objectlet_rows=objectlet_rows,
        mask_features=mask_features,
    )
    history_modes, history_semantic_rows = _build_history_modes(history_rows, history_samples, cfg.max_modes)
    shortlist_rows, shortlist_metrics = _build_shortlist_rows(update_rows, history_modes, history_samples, mask_features, cfg.max_modes)
    underseg_metrics = _build_underseg_metrics(
        support_rows,
        objectlet_rows,
        mask_features,
        history_modes,
        component_features,
        objectlet_underseg_variant=cfg.objectlet_underseg_variant,
    )

    feature_attempt_count = len(selected_masks)
    feature_success_count = sum(1 for row in mask_feature_rows if _as_bool(row.get("feature_available")))
    history_feature_count = sum(len(samples) for samples in history_samples.values())
    history_with_memory = sum(1 for modes in history_modes.values() if modes)
    history_count = len(history_rows)
    mode_counts = [len(modes) for modes in history_modes.values() if modes]
    variances = [
        float(row["mode_variance"])
        for row in history_semantic_rows
        if row.get("mode_variance") not in (None, "")
    ]

    same_category_confusion_rate = None
    same_category_metric_note = (
        "not_available: current v55/v56 diagnostic rows expose dominant instance ids, "
        "but no reliable semantic category label; not substituting instance id for category"
    )

    gate = {
        "feature_success_rate_ge_0_95": _safe_div(feature_success_count, feature_attempt_count) >= 0.95,
        "history_shortlist_recall_at_3_ge_0_85": _num(shortlist_metrics.get("history_shortlist_recall@3"), -1.0) >= 0.85,
        "underseg_or_outlier_auc_ge_0_70": _num(underseg_metrics.get("underseg_detection_AUC"), -1.0) >= 0.70
        or _num(underseg_metrics.get("outlier_detection_AUC"), -1.0) >= 0.70,
        "same_category_confusion_metric_available": same_category_confusion_rate is not None,
        "same_category_confusion_rate_pass": False,
    }
    gate["pass"] = bool(
        gate["feature_success_rate_ge_0_95"]
        and gate["history_shortlist_recall_at_3_ge_0_85"]
        and gate["underseg_or_outlier_auc_ge_0_70"]
        and gate["same_category_confusion_rate_pass"]
    )

    semantic_claim_allowed = bool(
        str(cfg.backend) != "colorhist"
        and gate["feature_success_rate_ge_0_95"]
        and gate["history_shortlist_recall_at_3_ge_0_85"]
    )
    summary = {
        "phase": "v58_semantic_memory",
        "backend": str(cfg.backend),
        "device": str(cfg.device),
        "checkpoint": cfg.checkpoint,
        "short_side": int(cfg.short_side),
        "support_variant": str(cfg.support_variant),
        "objectlet_underseg_variant": str(cfg.objectlet_underseg_variant),
        "feature_attempt_count": int(feature_attempt_count),
        "feature_success_count": int(feature_success_count),
        "feature_success_rate": _safe_div(feature_success_count, feature_attempt_count),
        "mask_feature_count": int(feature_success_count),
        "component_feature_count": int(len(component_features)),
        "history_feature_count": int(history_feature_count),
        "history_count": int(history_count),
        "history_with_semantic_memory_count": int(history_with_memory),
        "history_mode_count_mean": _mean(mode_counts),
        "history_semantic_variance_mean": _mean(variances),
        "semantic_memory_init_success_rate": _safe_div(history_with_memory, history_count),
        "history_shortlist_recall@1": shortlist_metrics.get("history_shortlist_recall@1"),
        "history_shortlist_recall@3": shortlist_metrics.get("history_shortlist_recall@3"),
        "shortlist_query_count": shortlist_metrics.get("shortlist_query_count"),
        "same_category_confusion_rate": same_category_confusion_rate,
        "same_category_metric_note": same_category_metric_note,
        "underseg_detection_AUC": underseg_metrics.get("underseg_detection_AUC"),
        "underseg_label_positive_count": underseg_metrics.get("underseg_label_positive_count"),
        "underseg_label_negative_count": underseg_metrics.get("underseg_label_negative_count"),
        "underseg_score_note": underseg_metrics.get("underseg_score_note"),
        "outlier_detection_AUC": None,
        "semantic_mode_purity_diagnostic": None,
        "semantic_mode_purity_note": (
            "not_available: current rows do not provide per-mode semantic category labels; "
            "dominant_gt_diagnostic is an instance diagnostic and is not used as semantic purity"
        ),
        "semantic_claim_allowed": semantic_claim_allowed,
        "semantic_claim_blockers": _semantic_claim_blockers(gate, cfg.backend),
        "gate": gate,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
        "input_paths": {
            "support_rows_path": _rel(cfg.support_rows_path),
            "history_rows_path": _rel(cfg.history_rows_path),
            "history_update_rows_path": _rel(cfg.history_update_rows_path),
            "objectlet_rows_path": _rel(cfg.objectlet_rows_path),
        },
        "output_paths": {
            "semantic_memory_summary": _rel(Path(cfg.output_root) / "semantic_memory_summary.json"),
            "mask_feature_rows": _rel(Path(cfg.output_root) / "mask_feature_rows.csv"),
            "component_feature_rows": _rel(Path(cfg.output_root) / "component_feature_rows.csv"),
            "history_semantic_rows": _rel(Path(cfg.output_root) / "history_semantic_rows.csv"),
            "semantic_shortlist_rows": _rel(Path(cfg.output_root) / "semantic_shortlist_rows.csv"),
        },
    }
    return {
        "summary": summary,
        "mask_feature_rows": mask_feature_rows,
        "component_feature_rows": component_feature_rows,
        "history_semantic_rows": history_semantic_rows,
        "semantic_shortlist_rows": shortlist_rows,
        "history_sample_rows": history_sample_rows,
    }


def write_v58_semantic_memory(result: dict[str, Any], output_root: str | Path) -> dict[str, str]:
    root = _project(output_root)
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "semantic_memory_summary": root / "semantic_memory_summary.json",
        "mask_feature_rows": root / "mask_feature_rows.csv",
        "component_feature_rows": root / "component_feature_rows.csv",
        "history_semantic_rows": root / "history_semantic_rows.csv",
        "semantic_shortlist_rows": root / "semantic_shortlist_rows.csv",
        "history_sample_rows": root / "history_sample_rows.csv",
    }
    write_json(paths["semantic_memory_summary"], result["summary"])
    write_csv(paths["mask_feature_rows"], result["mask_feature_rows"])
    write_csv(paths["component_feature_rows"], result["component_feature_rows"])
    write_csv(paths["history_semantic_rows"], result["history_semantic_rows"])
    write_csv(paths["semantic_shortlist_rows"], result["semantic_shortlist_rows"])
    write_csv(paths["history_sample_rows"], result["history_sample_rows"])
    return {name: _rel(path) for name, path in paths.items()}


def write_v58_semantic_memory_visualization(
    result: dict[str, Any],
    visualization_root: str | Path,
    *,
    tag: str,
) -> str:
    root = _project(visualization_root)
    root.mkdir(parents=True, exist_ok=True)
    summary = result["summary"]
    png_path = root / f"semantic_memory_modes_{tag}.png"
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        labels = ["feature", "recall@1", "recall@3", "underseg"]
        values = [
            _num(summary.get("feature_success_rate")),
            _num(summary.get("history_shortlist_recall@1")),
            _num(summary.get("history_shortlist_recall@3")),
            _num(summary.get("underseg_detection_AUC")),
        ]
        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(labels, values, color=["#247BA0", "#70C1B3", "#B2DBBF", "#F25F5C"])
        ax.axhline(0.85, color="#444444", linestyle="--", linewidth=1, label="shortlist gate")
        ax.axhline(0.70, color="#888888", linestyle=":", linewidth=1, label="underseg gate")
        ax.set_ylim(0.0, 1.05)
        ax.set_ylabel("score")
        ax.set_title(f"v58 semantic memory summary: {tag}")
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2.0, min(value + 0.03, 1.02), f"{value:.3f}", ha="center", va="bottom", fontsize=9)
        blocker_text = ", ".join(str(item) for item in summary.get("semantic_claim_blockers", [])) or "none"
        ax.text(0.01, -0.28, f"claim_allowed={summary.get('semantic_claim_allowed')} blockers={blocker_text}", transform=ax.transAxes, fontsize=8)
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        fig.savefig(png_path, dpi=160)
        plt.close(fig)
        return _rel(png_path)
    except Exception as exc:  # pragma: no cover - matplotlib availability is environment-specific.
        fallback = png_path.with_suffix(".txt")
        fallback.write_text(f"visualization_failed: {type(exc).__name__}: {exc}\n", encoding="utf-8")
        return _rel(fallback)


def _project(path: str | Path) -> Path:
    path_obj = Path(path)
    return path_obj if path_obj.is_absolute() else ROOT / path_obj


def _rel(path: str | Path) -> str:
    path_obj = _project(path)
    try:
        return str(path_obj.relative_to(ROOT))
    except ValueError:
        return str(path_obj)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _select_mask_observations(rows: list[dict[str, str]], *, max_count: int | None) -> list[str]:
    ids = sorted({str(row.get("mask_observation_id") or "") for row in rows if row.get("mask_observation_id")})
    if max_count is not None:
        ids = ids[: int(max_count)]
    return ids


def _extract_mask_features(mask_observations: list[str], cfg: V58SemanticMemoryConfig) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    adapter_cache: dict[str, Any] = {}
    feature_map_cache: dict[tuple[str, int, str], Any] = {}
    feature_cache: dict[tuple[str, str, str, int], tuple[list[float], dict[str, Any]]] = {}
    features: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    for mask_observation_id in mask_observations:
        feature, diag = _semantic_mask_feature(
            mask_observation_id,
            backend=str(cfg.backend),
            device=str(cfg.device),
            checkpoint=cfg.checkpoint,
            short_side=int(cfg.short_side),
            adapter_cache=adapter_cache,
            feature_map_cache=feature_map_cache,
            feature_cache=feature_cache,
        )
        vector = _normalize(np.asarray(feature, dtype=np.float32))
        if vector.size:
            features[mask_observation_id] = vector
        rows.append(
            {
                "mask_observation_id": mask_observation_id,
                "scene": diag.get("scene"),
                "frame_id": diag.get("frame_id"),
                "mask_id": diag.get("mask_id"),
                "backend": str(cfg.backend),
                "feature_available": bool(vector.size),
                "feature_missing_reason": diag.get("semantic_feature_missing_reason"),
                "feature_dim": int(vector.size),
                "feature_sha256": _feature_sha(vector) if vector.size else "",
                "feature_head_json": json.dumps([float(x) for x in vector[:8]], sort_keys=True),
                "feature_json": json.dumps([float(x) for x in vector], sort_keys=True) if vector.size and cfg.write_mask_feature_vectors else "",
                "semantic_mask_pixel_count": diag.get("semantic_mask_pixel_count"),
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic_labels": False,
            }
        )
    return features, rows


def _build_component_features(
    support_rows: list[dict[str, str]],
    mask_features: dict[str, np.ndarray],
) -> tuple[dict[tuple[str, str], np.ndarray], list[dict[str, Any]]]:
    accum: dict[tuple[str, str], list[tuple[float, np.ndarray, str]]] = defaultdict(list)
    total_support: Counter[tuple[str, str]] = Counter()
    for row in support_rows:
        key = (str(row.get("scene") or ""), str(row.get("component_id") or ""))
        if not key[0] or not key[1]:
            continue
        support = max(parse_float(row.get("support_count")), 0.0)
        total_support[key] += int(support)
        mask_id = str(row.get("mask_observation_id") or "")
        feature = mask_features.get(mask_id)
        if feature is not None and feature.size and support > 0.0:
            accum[key].append((support, feature, mask_id))
    features: dict[tuple[str, str], np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    all_keys = sorted(set(total_support) | set(accum))
    for scene, component_id in all_keys:
        items = accum.get((scene, component_id), [])
        if items:
            weight_sum = sum(weight for weight, _feature, _mask_id in items)
            vector = _normalize(sum(weight * feature for weight, feature, _mask_id in items) / max(weight_sum, 1e-8))
            features[(scene, component_id)] = vector
            top_mask = max(items, key=lambda item: item[0])[2]
        else:
            vector = np.asarray([], dtype=np.float32)
            weight_sum = 0.0
            top_mask = ""
        rows.append(
            {
                "scene": scene,
                "component_id": component_id,
                "backend": "weighted_mask_pool",
                "feature_available": bool(vector.size),
                "feature_dim": int(vector.size),
                "feature_sha256": _feature_sha(vector) if vector.size else "",
                "feature_head_json": json.dumps([float(x) for x in vector[:8]], sort_keys=True) if vector.size else "[]",
                "support_weight_sum": float(weight_sum),
                "support_row_count": int(len(items)),
                "total_component_support_count": int(total_support.get((scene, component_id), 0)),
                "top_mask_observation_id": top_mask,
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic_labels": False,
            }
        )
    return features, rows


def _build_history_samples(
    *,
    history_rows: list[dict[str, str]],
    update_rows: list[dict[str, str]],
    objectlet_rows: list[dict[str, str]],
    mask_features: dict[str, np.ndarray],
) -> tuple[dict[str, list[tuple[str, np.ndarray, str]]], list[dict[str, Any]]]:
    history_ids = {str(row.get("history_id") or "") for row in history_rows if row.get("history_id")}
    source_by_history: dict[str, str] = {}
    for row in objectlet_rows:
        objectlet_id = str(row.get("objectlet_id") or "")
        if objectlet_id in history_ids and objectlet_id not in source_by_history:
            source = str(row.get("source_mask_observation_id") or "")
            if source:
                source_by_history[objectlet_id] = source
    samples: dict[str, list[tuple[str, np.ndarray, str]]] = defaultdict(list)
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    def add(history_id: str, mask_observation_id: str, source: str) -> None:
        if not history_id or not mask_observation_id:
            return
        key = (history_id, mask_observation_id, source)
        if key in seen:
            return
        seen.add(key)
        feature = mask_features.get(mask_observation_id)
        available = feature is not None and feature.size > 0
        if available:
            samples[history_id].append((mask_observation_id, feature, source))
        rows.append(
            {
                "history_id": history_id,
                "mask_observation_id": mask_observation_id,
                "sample_source": source,
                "feature_available": bool(available),
                "feature_sha256": _feature_sha(feature) if available else "",
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic_labels": False,
            }
        )

    for history_id, mask_observation_id in source_by_history.items():
        add(history_id, mask_observation_id, "history_source_mask")
    for row in update_rows:
        state = str(row.get("update_state") or "")
        if state not in {"confirmed_update", "partial_update"}:
            continue
        add(str(row.get("history_id") or ""), str(row.get("candidate_id") or ""), f"v55_{state}")
    return samples, rows


def _build_history_modes(
    history_rows: list[dict[str, str]],
    history_samples: dict[str, list[tuple[str, np.ndarray, str]]],
    max_modes: int,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    modes_by_history: dict[str, list[dict[str, Any]]] = {}
    rows: list[dict[str, Any]] = []
    history_meta = {str(row.get("history_id") or ""): row for row in history_rows}
    for history_id in sorted(history_meta):
        samples = history_samples.get(history_id, [])
        vectors = [feature for _mask_id, feature, _source in samples]
        sample_ids = [mask_id for mask_id, _feature, _source in samples]
        modes = _fit_modes(vectors, max_modes=max_modes)
        modes_by_history[history_id] = modes
        for mode in modes:
            vector = np.asarray(mode["feature"], dtype=np.float32)
            rows.append(
                {
                    "history_id": history_id,
                    "scene": history_meta[history_id].get("scene"),
                    "mode_index": int(mode["mode_index"]),
                    "mode_sample_count": int(mode["sample_count"]),
                    "mode_weight": float(mode["weight"]),
                    "mode_variance": float(mode["variance"]),
                    "feature_dim": int(vector.size),
                    "feature_sha256": _feature_sha(vector),
                    "feature_head_json": json.dumps([float(x) for x in vector[:8]], sort_keys=True),
                    "sample_mask_observation_ids_json": json.dumps(sample_ids, sort_keys=True),
                    "dominant_gt_diagnostic": history_meta[history_id].get("dominant_gt_diagnostic"),
                    "uses_gt_for_prediction": False,
                    "uses_gt_for_diagnostic_labels": True,
                }
            )
    return modes_by_history, rows


def _build_shortlist_rows(
    update_rows: list[dict[str, str]],
    history_modes: dict[str, list[dict[str, Any]]],
    history_samples: dict[str, list[tuple[str, np.ndarray, str]]],
    mask_features: dict[str, np.ndarray],
    max_modes: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    hit1 = 0
    hit3 = 0
    total = 0
    for row in update_rows:
        state = str(row.get("update_state") or "")
        if state not in {"confirmed_update", "partial_update"}:
            continue
        query_id = str(row.get("candidate_id") or "")
        expected = str(row.get("history_id") or "")
        feature = mask_features.get(query_id)
        if feature is None or not feature.size or not expected:
            continue
        scored: list[tuple[float, str]] = []
        used_self_exclusion = False
        for history_id, modes in history_modes.items():
            candidate_modes = modes
            if history_id == expected:
                remaining = [sample_feature for mask_id, sample_feature, _source in history_samples.get(history_id, []) if mask_id != query_id]
                if remaining:
                    candidate_modes = _fit_modes(remaining, max_modes=max_modes)
                    used_self_exclusion = True
            scored.append((_score_modes(feature, candidate_modes), history_id))
        scored.sort(key=lambda item: (-item[0], item[1]))
        top = [history_id for _score, history_id in scored[:3]]
        total += 1
        hit1 += int(bool(top) and top[0] == expected)
        hit3 += int(expected in top)
        rows.append(
            {
                "query_mask_observation_id": query_id,
                "expected_history_id_diagnostic": expected,
                "update_state": state,
                "top1_history_id": top[0] if top else "",
                "top1_score": scored[0][0] if scored else None,
                "top3_history_ids_json": json.dumps(top, sort_keys=True),
                "hit_at_1": bool(top and top[0] == expected),
                "hit_at_3": bool(expected in top),
                "used_leave_one_out_for_expected_history": bool(used_self_exclusion),
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic_labels": True,
                "diagnostic_label_source": "v55_history_update_assignment_not_training_label",
            }
        )
    return rows, {
        "shortlist_query_count": int(total),
        "history_shortlist_recall@1": _safe_div(hit1, total),
        "history_shortlist_recall@3": _safe_div(hit3, total),
    }


def _build_underseg_metrics(
    support_rows: list[dict[str, str]],
    objectlet_rows: list[dict[str, str]],
    mask_features: dict[str, np.ndarray],
    history_modes: dict[str, list[dict[str, Any]]],
    component_features: dict[tuple[str, str], np.ndarray],
    *,
    objectlet_underseg_variant: str,
) -> dict[str, Any]:
    components_by_mask: dict[str, Counter[str]] = defaultdict(Counter)
    scene_by_mask: dict[str, str] = {}
    for row in support_rows:
        mask_id = str(row.get("mask_observation_id") or "")
        comp = str(row.get("component_id") or "")
        scene = str(row.get("scene") or "")
        if mask_id and comp:
            components_by_mask[mask_id][comp] += max(parse_int(row.get("support_count")), 1)
            scene_by_mask[mask_id] = scene
    underseg_votes: dict[str, Counter[bool]] = defaultdict(Counter)
    for row in objectlet_rows:
        if str(row.get("variant") or "") != str(objectlet_underseg_variant):
            continue
        mask_id = str(row.get("source_mask_observation_id") or "")
        if not mask_id:
            continue
        raw = str(row.get("underseg_proxy") or "").strip().lower()
        if raw not in {"true", "false", "1", "0"}:
            continue
        underseg_votes[mask_id][raw in {"true", "1"}] += 1
    labels: list[bool] = []
    scores: list[float] = []
    for mask_id, votes in underseg_votes.items():
        feature = mask_features.get(mask_id)
        if feature is None or not feature.size:
            continue
        components = components_by_mask.get(mask_id, Counter())
        scored = sorted((_score_modes(feature, modes) for modes in history_modes.values()), reverse=True)
        if len(scored) < 2:
            history_ambiguity = 0.0
        else:
            history_ambiguity = 1.0 - max(0.0, min(1.0, scored[0] - scored[1]))
        component_entropy = _counter_entropy(components)
        component_diversity = _component_semantic_diversity(scene_by_mask.get(mask_id, ""), components, component_features)
        underseg_score = max(float(history_ambiguity), float(component_entropy), float(component_diversity))
        labels.append(votes[True] >= votes[False])
        scores.append(float(underseg_score))
    auc = rank_auc(labels, scores) if labels else None
    return {
        "underseg_detection_AUC": auc,
        "underseg_label_positive_count": int(sum(labels)),
        "underseg_label_negative_count": int(len(labels) - sum(labels)),
        "outlier_detection_AUC": None,
        "underseg_score_note": (
            "max(history_top2_ambiguity, component_support_entropy, component_feature_diversity); "
            f"labels use objectlet underseg_proxy majority from variant={objectlet_underseg_variant}"
        ),
    }


def _fit_modes(vectors: list[np.ndarray], *, max_modes: int) -> list[dict[str, Any]]:
    clean = [_normalize(np.asarray(vec, dtype=np.float32)) for vec in vectors if np.asarray(vec).size]
    if not clean:
        return []
    if len(clean) < 4:
        mode_count = 1
    else:
        mode_count = max(1, min(int(max_modes), int(round(math.sqrt(len(clean))))))
    centers = _init_centers(clean, mode_count)
    assignments = [0 for _ in clean]
    for _iter in range(6):
        assignments = [int(np.argmax([_cosine(vec, center) for center in centers])) for vec in clean]
        next_centers: list[np.ndarray] = []
        for mode_idx in range(mode_count):
            members = [vec for vec, assignment in zip(clean, assignments) if assignment == mode_idx]
            if members:
                next_centers.append(_normalize(np.mean(np.stack(members, axis=0), axis=0)))
            else:
                next_centers.append(centers[mode_idx])
        centers = next_centers
    modes: list[dict[str, Any]] = []
    for mode_idx, center in enumerate(centers):
        members = [vec for vec, assignment in zip(clean, assignments) if assignment == mode_idx]
        if not members:
            continue
        variance = float(np.mean([max(0.0, 1.0 - _cosine(vec, center)) for vec in members]))
        modes.append(
            {
                "mode_index": int(len(modes)),
                "sample_count": int(len(members)),
                "weight": float(len(members) / max(len(clean), 1)),
                "variance": variance,
                "feature": center,
            }
        )
    return modes


def _counter_entropy(counter: Counter[str]) -> float:
    total = float(sum(counter.values()))
    if total <= 0.0 or len(counter) <= 1:
        return 0.0
    probs = [float(value) / total for value in counter.values() if value > 0]
    entropy = -sum(prob * math.log(prob) for prob in probs)
    return float(entropy / max(math.log(len(probs)), 1e-8))


def _component_semantic_diversity(
    scene: str,
    counter: Counter[str],
    component_features: dict[tuple[str, str], np.ndarray],
) -> float:
    vectors: list[np.ndarray] = []
    for component_id, _support in counter.most_common(12):
        feature = component_features.get((scene, component_id))
        if feature is not None and feature.size:
            vectors.append(feature)
    if len(vectors) <= 1:
        return 0.0
    distances: list[float] = []
    for idx, left in enumerate(vectors):
        for right in vectors[idx + 1 :]:
            distances.append(max(0.0, 1.0 - _cosine(left, right)))
    return float(max(distances)) if distances else 0.0


def _init_centers(vectors: list[np.ndarray], mode_count: int) -> list[np.ndarray]:
    centers = [vectors[0]]
    while len(centers) < mode_count:
        candidate = min(vectors, key=lambda vec: max(_cosine(vec, center) for center in centers))
        centers.append(candidate)
    return [_normalize(center) for center in centers]


def _score_modes(feature: np.ndarray, modes: list[dict[str, Any]]) -> float:
    if not modes:
        return -1.0
    return max(_cosine(feature, np.asarray(mode["feature"], dtype=np.float32)) for mode in modes)


def _normalize(vector: np.ndarray) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if arr.size == 0 or norm <= 1e-8:
        return np.asarray([], dtype=np.float32)
    return arr / norm


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    return cosine([float(x) for x in left], [float(x) for x in right])


def _feature_sha(vector: np.ndarray | None) -> str:
    if vector is None:
        return ""
    arr = np.asarray(vector, dtype=np.float32).reshape(-1)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _safe_div(num: int | float, den: int | float) -> float | None:
    return None if float(den) == 0.0 else float(num) / float(den)


def _mean(values: Iterable[Any]) -> float | None:
    vals = [float(value) for value in values if value not in (None, "") and math.isfinite(float(value))]
    return float(np.mean(vals)) if vals else None


def _num(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return float(default)
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _semantic_claim_blockers(gate: dict[str, Any], backend: str) -> list[str]:
    blockers: list[str] = []
    if str(backend) == "colorhist":
        blockers.append("only_colorhist_backend_available")
    for key, value in gate.items():
        if key == "pass":
            continue
        if value is False:
            blockers.append(key)
    return blockers


def dumps_json(payload: Any) -> str:
    return json.dumps(json_safe(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
