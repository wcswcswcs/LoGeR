from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .v47_common import (
    ROOT,
    adjusted_rand_score,
    cluster_completeness,
    cluster_purity,
    json_safe,
    parse_bool,
    parse_float,
    parse_int,
    read_json,
    safe_mean,
    utc_now,
    write_csv,
    write_json,
)
from .v58_semantic_memory import _fit_modes, _normalize, _score_modes


DEFAULT_SEMANTIC_ROOT = "outputs/audit/v58_semantic_memory_dino_full_repair2"
DEFAULT_SUPPORT_ROWS = "outputs/audit/v54_mask_component_support_tau005_stride1_probe5_q4096_notopup/mask_component_support_rows.csv"
DEFAULT_HISTORY_ROWS = "outputs/audit/v55_history_update/history_rows.csv"
DEFAULT_HISTORY_UPDATE_ROWS = "outputs/audit/v55_history_update/history_update_rows.csv"
DEFAULT_OBJECTLET_ROWS = (
    "outputs/audit/v54_local_objectlets_k0all_conflict_veto018_skip_repeated_sig_l12_stride1_probe5_q4096_notopup_max4000_skip/objectlet_rows.csv"
)
DEFAULT_V56_CORE_SUMMARY = "outputs/audit/v56_core_update/core_update_summary.json"
DEFAULT_V56_TENTATIVE_SUMMARY = "outputs/audit/v56_tentative_support/tentative_support_summary.json"


@dataclass(frozen=True)
class V58CounterfactualConfig:
    semantic_root: str | Path = DEFAULT_SEMANTIC_ROOT
    support_rows_path: str | Path = DEFAULT_SUPPORT_ROWS
    history_rows_path: str | Path = DEFAULT_HISTORY_ROWS
    history_update_rows_path: str | Path = DEFAULT_HISTORY_UPDATE_ROWS
    objectlet_rows_path: str | Path = DEFAULT_OBJECTLET_ROWS
    v56_core_summary_path: str | Path = DEFAULT_V56_CORE_SUMMARY
    v56_tentative_summary_path: str | Path = DEFAULT_V56_TENTATIVE_SUMMARY
    output_root: str | Path = "outputs/audit/v58_counterfactual_explanation"
    visualization_root: str | Path = "outputs/audit/v58_visualizations/explanation"
    support_variant: str = "I0_visible_tau0.10"
    objectlet_underseg_variant: str = "L11_dynamic_uncovered_gain_dup010"
    primary_variant: str = "E6_counterfactual_semantic_material_underseg"
    k_sem: int = 5
    k_mat: int = 5
    max_modes: int = 4
    max_observations: int | None = None
    entropy_fraction_for_ambiguous: float = 0.80
    margin_for_ambiguous: float = 0.15


def build_v58_counterfactual_explanation(config: V58CounterfactualConfig | None = None) -> dict[str, Any]:
    cfg = config or V58CounterfactualConfig()
    semantic_root = _project(cfg.semantic_root)
    semantic_summary = read_json(semantic_root / "semantic_memory_summary.json")
    mask_features, mask_meta = _load_mask_features(semantic_root / "mask_feature_rows.csv")
    history_sample_rows = _read_csv(semantic_root / "history_sample_rows.csv")
    history_rows = _read_csv(_project(cfg.history_rows_path))
    update_rows = _read_csv(_project(cfg.history_update_rows_path))
    support_rows = [
        row
        for row in _read_csv(_project(cfg.support_rows_path))
        if str(row.get("variant") or "") == str(cfg.support_variant)
    ]
    objectlet_rows = [
        row
        for row in _read_csv(_project(cfg.objectlet_rows_path))
        if str(row.get("variant") or "") == str(cfg.objectlet_underseg_variant)
    ]
    v56_core = read_json(_project(cfg.v56_core_summary_path))
    v56_tentative = read_json(_project(cfg.v56_tentative_summary_path))

    history_meta = {str(row.get("history_id") or ""): row for row in history_rows if row.get("history_id")}
    history_modes = _rebuild_history_modes(history_sample_rows, mask_features, history_meta, max_modes=cfg.max_modes)
    support_index = _build_support_index(support_rows)
    history_index = _build_history_index(objectlet_rows, history_meta)
    update_index = _build_update_index(update_rows)
    objectlet_index = _build_objectlet_index(objectlet_rows)
    observations = _select_observations(mask_features, support_index, update_index, objectlet_index, max_count=cfg.max_observations)

    variants = _variant_specs()
    explanation_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    rows_by_variant: dict[str, list[dict[str, Any]]] = {}
    for variant in variants:
        rows = _score_variant(
            variant,
            observations,
            mask_features,
            mask_meta,
            history_modes,
            support_index,
            history_index,
            update_index,
            objectlet_index,
            history_meta,
            cfg,
        )
        rows_by_variant[variant["name"]] = rows
        explanation_rows.extend(rows)
        metric_rows.append(
            _evaluate_explanation_metrics(
                variant["name"],
                rows,
                v56_core=v56_core,
                v56_tentative=v56_tentative,
            )
        )

    primary_rows = rows_by_variant.get(cfg.primary_variant) or rows_by_variant[variants[-1]["name"]]
    history_metric_rows = _build_history_metric_rows(primary_rows, support_index, v56_core, v56_tentative)
    primary_metrics = next((row for row in metric_rows if row["variant"] == cfg.primary_variant), metric_rows[-1])
    primary_history_metrics = next(
        (row for row in history_metric_rows if row.get("variant") == cfg.primary_variant),
        {},
    )
    v56_false_rate = _v56_false_history_rate(v56_core)
    false_rate_threshold = None if v56_false_rate is None else float(v56_false_rate - 0.10)
    gate = {
        "assign_precision_ge_0_85": _num(primary_metrics.get("assign_precision_diagnostic"), -1.0) >= 0.85,
        "partial_precision_ge_0_80": _num(primary_metrics.get("partial_precision_diagnostic"), -1.0) >= 0.80,
        "underseg_precision_ge_0_75": _num(primary_metrics.get("underseg_precision_diagnostic"), -1.0) >= 0.75,
        "new_birth_precision_ge_0_85": _num(primary_metrics.get("new_birth_precision_diagnostic"), -1.0) >= 0.85,
        "false_history_update_rate_le_v56_minus_0_10": (
            false_rate_threshold is not None
            and _num(primary_metrics.get("false_history_update_rate"), 999.0) <= false_rate_threshold
        ),
        "purity_ge_v56_core_minus_0_005": (
            primary_history_metrics.get("metric_scope") == "diagnostic_observation_support_projection"
            and _num(primary_history_metrics.get("purity"), -1.0) >= _num(v56_core.get("core_purity"), 999.0) - 0.005
        ),
        "completeness_ge_v56_expanded_minus_0_03": (
            primary_history_metrics.get("metric_scope") == "diagnostic_observation_support_projection"
            and _num(primary_history_metrics.get("completeness"), -1.0)
            >= _num(v56_tentative.get("expanded_completeness"), 999.0) - 0.03
        ),
    }
    gate["pass"] = bool(all(gate.values()))
    summary = {
        "phase": "v58_counterfactual_explanation",
        "created_at": utc_now(),
        "primary_variant": cfg.primary_variant,
        "semantic_backend": semantic_summary.get("backend"),
        "semantic_phase1_gate_pass": bool((semantic_summary.get("gate") or {}).get("pass")),
        "semantic_phase1_gate_note": (
            "Phase2 uses DINO semantic memory as a partial mechanism input; Phase1 full gate remained false "
            "because same-category category labels were unavailable."
        ),
        "observation_count": int(len(observations)),
        "explanation_candidate_count": int(sum(1 for row in primary_rows if row.get("row_role") == "candidate")),
        "posterior_entropy_mean": primary_metrics.get("posterior_entropy_mean"),
        "selected_count_total": primary_metrics.get("selected_count_total"),
        "deferred_count": primary_metrics.get("deferred_count"),
        "actionable_count": primary_metrics.get("actionable_count"),
        "assign_count": primary_metrics.get("assign_count"),
        "partial_count": primary_metrics.get("partial_count"),
        "new_object_count": primary_metrics.get("new_object_count"),
        "underseg_count": primary_metrics.get("underseg_count"),
        "outlier_count": primary_metrics.get("outlier_count"),
        "ambiguous_count": primary_metrics.get("ambiguous_count"),
        "assign_precision_diagnostic": primary_metrics.get("assign_precision_diagnostic"),
        "partial_precision_diagnostic": primary_metrics.get("partial_precision_diagnostic"),
        "new_birth_precision_diagnostic": primary_metrics.get("new_birth_precision_diagnostic"),
        "underseg_precision_diagnostic": primary_metrics.get("underseg_precision_diagnostic"),
        "outlier_precision_diagnostic": primary_metrics.get("outlier_precision_diagnostic"),
        "false_history_update_rate": primary_metrics.get("false_history_update_rate"),
        "false_new_birth_rate": primary_metrics.get("false_new_birth_rate"),
        "same_category_confusion_rate": None,
        "same_category_confusion_note": (
            "not_available: scene-level semantic category labels are absent; instance ids are not used as same-category labels"
        ),
        "v56_hard_update_false_rate_source": "v56_core_update.update_precision_diagnostic",
        "v56_hard_update_false_rate": v56_false_rate,
        "v56_false_history_update_threshold_minus_0_10": false_rate_threshold,
        "history_metric_scope": primary_history_metrics.get("metric_scope"),
        "ARI": primary_history_metrics.get("ARI"),
        "purity": primary_history_metrics.get("purity"),
        "completeness": primary_history_metrics.get("completeness"),
        "temporal_span_mean": primary_history_metrics.get("temporal_span_mean"),
        "duplicate_rate": primary_history_metrics.get("duplicate_rate"),
        "conflict_rate": primary_history_metrics.get("conflict_rate"),
        "real_minus_shuffled_ARI": primary_history_metrics.get("real_minus_shuffled_ARI"),
        "real_minus_no_temporal_ARI": primary_history_metrics.get("real_minus_no_temporal_ARI"),
        "real_minus_mask_only_ARI": primary_history_metrics.get("real_minus_mask_only_ARI"),
        "gate": gate,
        "gate_note": (
            "history ARI/purity/completeness are computed as a diagnostic observation-support projection, "
            "not as a full Phase4/Phase8 object-field update."
        ),
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
        "diagnostic_label_sources": [
            "v55_history_update.update_state/history_id",
            "v54_objectlet.underseg_proxy",
            "support_rows.diagnostic_gt_instance for new-vs-existing diagnostics only",
        ],
        "input_paths": {
            "semantic_root": _rel(semantic_root),
            "support_rows_path": _rel(cfg.support_rows_path),
            "history_rows_path": _rel(cfg.history_rows_path),
            "history_update_rows_path": _rel(cfg.history_update_rows_path),
            "objectlet_rows_path": _rel(cfg.objectlet_rows_path),
            "v56_core_summary_path": _rel(cfg.v56_core_summary_path),
            "v56_tentative_summary_path": _rel(cfg.v56_tentative_summary_path),
        },
        "output_paths": {
            "explanation_summary": _rel(Path(cfg.output_root) / "explanation_summary.json"),
            "explanation_rows": _rel(Path(cfg.output_root) / "explanation_rows.csv"),
            "explanation_metric_rows": _rel(Path(cfg.output_root) / "explanation_metric_rows.csv"),
            "history_metric_rows": _rel(Path(cfg.output_root) / "history_metric_rows.csv"),
        },
    }
    return {
        "summary": summary,
        "explanation_rows": explanation_rows,
        "explanation_metric_rows": metric_rows,
        "history_metric_rows": history_metric_rows,
    }


def write_v58_counterfactual_explanation(result: dict[str, Any], output_root: str | Path) -> dict[str, str]:
    root = _project(output_root)
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "explanation_summary": root / "explanation_summary.json",
        "explanation_rows": root / "explanation_rows.csv",
        "explanation_metric_rows": root / "explanation_metric_rows.csv",
        "history_metric_rows": root / "history_metric_rows.csv",
    }
    write_json(paths["explanation_summary"], result["summary"])
    write_csv(paths["explanation_rows"], result["explanation_rows"])
    write_csv(paths["explanation_metric_rows"], result["explanation_metric_rows"])
    write_csv(paths["history_metric_rows"], result["history_metric_rows"])
    return {name: _rel(path) for name, path in paths.items()}


def write_v58_counterfactual_visualization(
    result: dict[str, Any],
    visualization_root: str | Path,
    *,
    tag: str = "primary",
) -> str:
    root = _project(visualization_root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"explanation_posterior_panel_{tag}.png"
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        summary = result["summary"]
        labels = ["assign", "partial", "new", "underseg", "outlier", "ambig"]
        values = [
            _num(summary.get("assign_count")),
            _num(summary.get("partial_count")),
            _num(summary.get("new_object_count")),
            _num(summary.get("underseg_count")),
            _num(summary.get("outlier_count")),
            _num(summary.get("ambiguous_count")),
        ]
        fig, ax = plt.subplots(figsize=(8.5, 4.2))
        ax.bar(labels, values, color=["#36688D", "#66A182", "#F0A202", "#C73E1D", "#6D6875", "#8E7DBE"])
        ax.set_title(f"v58 Phase2 explanation counts: {summary.get('primary_variant')}")
        ax.set_ylabel("selected observations")
        for idx, value in enumerate(values):
            ax.text(idx, value + max(values + [1.0]) * 0.02, str(int(value)), ha="center", va="bottom", fontsize=8)
        text = (
            f"assignP={_fmt(summary.get('assign_precision_diagnostic'))} "
            f"partialP={_fmt(summary.get('partial_precision_diagnostic'))} "
            f"undersegP={_fmt(summary.get('underseg_precision_diagnostic'))} "
            f"newP={_fmt(summary.get('new_birth_precision_diagnostic'))} "
            f"gate={summary.get('gate', {}).get('pass')}"
        )
        ax.text(0.01, -0.24, text, transform=ax.transAxes, fontsize=8)
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        return _rel(path)
    except Exception as exc:  # pragma: no cover
        fallback = path.with_suffix(".txt")
        fallback.write_text(f"visualization_failed: {type(exc).__name__}: {exc}\n", encoding="utf-8")
        return _rel(fallback)


def _score_variant(
    variant: dict[str, Any],
    observations: list[str],
    mask_features: dict[str, np.ndarray],
    mask_meta: dict[str, dict[str, Any]],
    history_modes: dict[str, list[dict[str, Any]]],
    support_index: dict[str, Any],
    history_index: dict[str, Any],
    update_index: dict[str, Any],
    objectlet_index: dict[str, Any],
    history_meta: dict[str, dict[str, str]],
    cfg: V58CounterfactualConfig,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for observation_id in observations:
        feature = mask_features.get(observation_id)
        if feature is None or not feature.size:
            continue
        scene = str(mask_meta.get(observation_id, {}).get("scene") or support_index["scene_by_mask"].get(observation_id) or "")
        sem_scores = _semantic_scores(feature, history_modes, scene, history_meta)
        mat_scores = _material_scores(observation_id, scene, support_index, history_index)
        candidates = _candidate_histories(sem_scores, mat_scores, cfg.k_sem, cfg.k_mat)
        diagnostic = _diagnostic_label(observation_id, scene, update_index, objectlet_index, support_index, history_index)
        features = _observation_evidence(
            observation_id,
            scene,
            sem_scores,
            mat_scores,
            support_index,
            objectlet_index,
        )
        features["has_update_candidate"] = bool(update_index["rows_by_candidate"].get(observation_id))
        features["has_objectlet_candidate"] = bool(objectlet_index["rows_by_mask"].get(observation_id))
        candidate_rows = _candidate_rows_for_observation(
            variant,
            observation_id,
            scene,
            candidates,
            sem_scores,
            mat_scores,
            features,
            diagnostic,
        )
        _normalize_posteriors(candidate_rows)
        top = max(candidate_rows, key=lambda row: (float(row["posterior"]), row["explanation_type"], row.get("history_id", "")))
        entropy = _posterior_entropy([float(row["posterior"]) for row in candidate_rows])
        margin = _posterior_margin([float(row["posterior"]) for row in candidate_rows])
        ambiguous = entropy >= float(cfg.entropy_fraction_for_ambiguous) * math.log(max(len(candidate_rows), 2)) or margin <= float(cfg.margin_for_ambiguous)
        for row in candidate_rows:
            row["posterior_entropy"] = entropy
            row["posterior_top1_margin"] = margin
            row["is_selected"] = row is top
            row["is_ambiguous"] = bool(ambiguous)
            row["decision_state"] = _decision_state(row, top, entropy, margin) if row is top else "not_selected"
            rows.append(row)
    return rows


def _candidate_rows_for_observation(
    variant: dict[str, Any],
    observation_id: str,
    scene: str,
    candidates: list[str],
    sem_scores: dict[str, float],
    mat_scores: dict[str, float],
    evidence: dict[str, Any],
    diagnostic: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sem_weight = float(variant["sem_weight"])
    mat_weight = float(variant["mat_weight"])
    use_partial = bool(variant["partial"])
    use_underseg = bool(variant["underseg"])
    top_sem = float(evidence["top_semantic_score"])
    top_mat = float(evidence["top_material_score"])
    top2_gap = float(evidence["semantic_top2_gap"])
    competition = float(evidence["material_competition"])
    entropy = float(evidence["component_entropy"])
    diversity = float(evidence["component_diversity"])
    objectness = max(0.0, min(1.0, 0.55 * ((top_sem + 1.0) / 2.0) + 0.45 * top_mat))
    underseg_score = max(entropy, competition, diversity, 1.0 - min(max(top2_gap, 0.0), 1.0))
    for history_id in candidates:
        sem = float(sem_scores.get(history_id, -1.0))
        mat = float(mat_scores.get(history_id, 0.0))
        sem_adv = sem - max([score for hid, score in sem_scores.items() if hid != history_id] or [-1.0])
        low_adv_high_entropy = entropy >= 0.92 and sem_adv <= 0.22
        assign_logit = (
            sem_weight * (2.0 * sem)
            + mat_weight * (2.3 * mat)
            + 1.0 * sem_adv
            - 1.3 * competition
            - 0.9 * entropy
            - 1.2 * (1.0 - min(max(mat, 0.0), 1.0))
            - (1.4 if low_adv_high_entropy else 0.0)
            - 0.25
        )
        rows.append(
            _base_row(
                variant["name"],
                observation_id,
                scene,
                "assign_to_existing",
                history_id,
                assign_logit,
                sem,
                mat,
                sem_adv,
                evidence,
                diagnostic,
            )
        )
        if use_partial:
            partial_preference = max(0.0, 1.0 - abs(mat - 0.45) / 0.45)
            partial_logit = (
                sem_weight * (1.65 * sem)
                + mat_weight * (1.6 * mat)
                + 1.6 * partial_preference
                + 0.7 * sem_adv
                - 0.6 * competition
                - 0.55
            )
            rows.append(
                _base_row(
                    variant["name"],
                    observation_id,
                    scene,
                    "partial_of_existing",
                    history_id,
                    partial_logit,
                    sem,
                    mat,
                    sem_adv,
                    evidence,
                    diagnostic,
                )
            )
    if use_underseg and len(candidates) >= 2:
        pair = candidates[:2]
        low_adv_high_entropy = entropy >= 0.92 and top2_gap <= 0.22
        underseg_logit = (
            2.6 * underseg_score
            + mat_weight * 1.5 * competition
            + sem_weight * 1.4 * (1.0 - min(max(top2_gap, 0.0), 1.0))
            + 0.8 * entropy
            + (1.3 if low_adv_high_entropy else 0.0)
            - 1.2
        )
        rows.append(
            _base_row(
                variant["name"],
                observation_id,
                scene,
                "underseg_mixture",
                "||".join(pair),
                underseg_logit,
                top_sem,
                top_mat,
                top2_gap,
                evidence,
                diagnostic,
            )
        )
    new_logit = 1.45 * (1.0 - top_mat) + 1.05 * (1.0 - ((top_sem + 1.0) / 2.0)) + 0.35 * objectness - 1.05 * underseg_score - 0.35
    if variant["name"] == "E2_material_only":
        new_logit = 1.8 * (1.0 - top_mat) - 0.4 * competition
    rows.append(_base_row(variant["name"], observation_id, scene, "new_object", "", new_logit, top_sem, top_mat, top2_gap, evidence, diagnostic))
    outlier_logit = 1.9 * (1.0 - objectness) + 0.8 * float(evidence["outside_residual"]) - 0.8 * float(evidence["mask_evidence_strength"])
    rows.append(
        _base_row(
            variant["name"],
            observation_id,
            scene,
            "outlier_or_background",
            "",
            outlier_logit,
            top_sem,
            top_mat,
            top2_gap,
            evidence,
            diagnostic,
        )
    )
    return rows


def _base_row(
    variant: str,
    observation_id: str,
    scene: str,
    explanation_type: str,
    history_id: str,
    logit: float,
    semantic_score: float,
    material_score: float,
    semantic_advantage: float,
    evidence: dict[str, Any],
    diagnostic: dict[str, Any],
) -> dict[str, Any]:
    correct = _diagnostic_correct(explanation_type, history_id, diagnostic)
    return {
        "variant": variant,
        "row_role": "candidate",
        "observation_id": observation_id,
        "scene": scene,
        "frame_id": _frame_from_observation(observation_id),
        "mask_id": _mask_from_observation(observation_id),
        "explanation_type": explanation_type,
        "history_id": history_id,
        "candidate_history_ids_json": json.dumps(_row_history_ids(explanation_type, history_id)),
        "logit": float(logit),
        "posterior": None,
        "semantic_score": float(semantic_score),
        "material_score": float(material_score),
        "semantic_advantage": float(semantic_advantage),
        "component_entropy": float(evidence["component_entropy"]),
        "component_diversity": float(evidence["component_diversity"]),
        "material_competition": float(evidence["material_competition"]),
        "outside_residual": float(evidence["outside_residual"]),
        "mask_evidence_strength": float(evidence["mask_evidence_strength"]),
        "support_component_count": int(evidence["support_component_count"]),
        "has_update_candidate": bool(evidence.get("has_update_candidate")),
        "has_objectlet_candidate": bool(evidence.get("has_objectlet_candidate")),
        "diagnostic_expected_type": diagnostic.get("expected_type"),
        "diagnostic_expected_history_ids_json": json.dumps(sorted(diagnostic.get("expected_history_ids") or [])),
        "diagnostic_label_source": diagnostic.get("label_source"),
        "diagnostic_correct": correct,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }


def _normalize_posteriors(rows: list[dict[str, Any]]) -> None:
    logits = np.asarray([float(row["logit"]) for row in rows], dtype=np.float64)
    logits = logits - float(np.max(logits))
    probs = np.exp(logits)
    denom = float(probs.sum())
    probs = probs / denom if denom > 0.0 else np.ones_like(probs) / max(len(probs), 1)
    for row, prob in zip(rows, probs):
        row["posterior"] = float(prob)


def _evaluate_explanation_metrics(
    variant: str,
    rows: list[dict[str, Any]],
    *,
    v56_core: dict[str, Any],
    v56_tentative: dict[str, Any],
) -> dict[str, Any]:
    selected_total = [row for row in rows if parse_bool(row.get("is_selected"))]
    deferred = [row for row in selected_total if row.get("decision_state") == "defer_to_active_query"]
    selected = [row for row in selected_total if row.get("decision_state") == "actionable"]
    counts = Counter(str(row.get("explanation_type")) for row in selected)
    correct_by_type: dict[str, list[bool]] = defaultdict(list)
    for row in selected:
        correct = row.get("diagnostic_correct")
        if correct in (None, ""):
            continue
        correct_by_type[str(row.get("explanation_type"))].append(parse_bool(correct))
    update_selected = [
        row
        for row in selected
        if str(row.get("explanation_type")) in {"assign_to_existing", "partial_of_existing"}
    ]
    false_updates = [
        row
        for row in update_selected
        if not parse_bool(row.get("diagnostic_correct"))
    ]
    new_selected = [row for row in selected if row.get("explanation_type") == "new_object"]
    false_new = [row for row in new_selected if not parse_bool(row.get("diagnostic_correct"))]
    metric = {
        "variant": variant,
        "observation_count": int(len(selected_total)),
        "selected_count_total": int(len(selected_total)),
        "deferred_count": int(len(deferred)),
        "actionable_count": int(len(selected)),
        "explanation_candidate_count": int(len(rows)),
        "posterior_entropy_mean": safe_mean(row.get("posterior_entropy") for row in selected_total),
        "posterior_top1_margin_mean": safe_mean(row.get("posterior_top1_margin") for row in selected_total),
        "assign_count": int(counts.get("assign_to_existing", 0)),
        "partial_count": int(counts.get("partial_of_existing", 0)),
        "new_object_count": int(counts.get("new_object", 0)),
        "underseg_count": int(counts.get("underseg_mixture", 0)),
        "outlier_count": int(counts.get("outlier_or_background", 0)),
        "ambiguous_count": int(sum(1 for row in selected if parse_bool(row.get("is_ambiguous")))),
        "assign_precision_diagnostic": _precision(correct_by_type.get("assign_to_existing", [])),
        "partial_precision_diagnostic": _precision(correct_by_type.get("partial_of_existing", [])),
        "new_birth_precision_diagnostic": _precision(correct_by_type.get("new_object", [])),
        "underseg_precision_diagnostic": _precision(correct_by_type.get("underseg_mixture", [])),
        "outlier_precision_diagnostic": _precision(correct_by_type.get("outlier_or_background", [])),
        "false_history_update_rate": _safe_div(len(false_updates), len(update_selected)),
        "false_new_birth_rate": _safe_div(len(false_new), len(new_selected)),
        "same_category_confusion_rate": None,
        "v56_core_update_precision_diagnostic": v56_core.get("update_precision_diagnostic"),
        "v56_tentative_precision_diagnostic": v56_tentative.get("tentative_precision_diagnostic"),
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }
    return metric


def _build_history_metric_rows(
    selected_rows: list[dict[str, Any]],
    support_index: dict[str, Any],
    v56_core: dict[str, Any],
    v56_tentative: dict[str, Any],
) -> list[dict[str, Any]]:
    selected = [
        row
        for row in selected_rows
        if parse_bool(row.get("is_selected")) and row.get("decision_state") == "actionable"
    ]
    primary_assignments = _support_projection_assignments(selected, support_index, mode="primary")
    shuffled_assignments = _support_projection_assignments(selected, support_index, mode="shuffled")
    mask_only_assignments = _support_projection_assignments(selected, support_index, mode="mask_only")
    primary = _projection_metrics(primary_assignments)
    shuffled = _projection_metrics(shuffled_assignments)
    mask_only = _projection_metrics(mask_only_assignments)
    temporal_span = _temporal_span(selected)
    duplicate_rate = _duplicate_rate(selected, support_index)
    conflict_rate = _conflict_rate(selected)
    row = {
        "variant": "E6_counterfactual_semantic_material_underseg",
        "metric_scope": "diagnostic_observation_support_projection",
        "ARI": primary["ARI"],
        "purity": primary["purity"],
        "completeness": primary["completeness"],
        "temporal_span_mean": temporal_span,
        "duplicate_rate": duplicate_rate,
        "conflict_rate": conflict_rate,
        "real_minus_shuffled_ARI": None if shuffled["ARI"] is None else float(primary["ARI"] - shuffled["ARI"]),
        "real_minus_no_temporal_ARI": None if mask_only["ARI"] is None else float(primary["ARI"] - mask_only["ARI"]),
        "real_minus_mask_only_ARI": None if mask_only["ARI"] is None else float(primary["ARI"] - mask_only["ARI"]),
        "shuffled_ARI": shuffled["ARI"],
        "mask_only_ARI": mask_only["ARI"],
        "note": "diagnostic projection from selected explanations to mask-component support rows; not a full object-field update",
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }
    return [
        {
            "variant": "v56_core_baseline",
            "metric_scope": "v56_core_update_summary",
            "ARI": v56_core.get("core_ARI"),
            "purity": v56_core.get("core_purity"),
            "completeness": v56_core.get("core_completeness"),
            "temporal_span_mean": v56_core.get("history_temporal_span_mean"),
            "real_minus_shuffled_ARI": v56_core.get("real_minus_shuffled_ARI"),
            "real_minus_no_temporal_ARI": v56_core.get("real_minus_no_temporal_ARI"),
            "real_minus_mask_only_ARI": v56_core.get("real_minus_mask_only_ARI"),
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic_labels": True,
        },
        {
            "variant": "v56_expanded_baseline",
            "metric_scope": "v56_tentative_support_summary",
            "ARI": v56_tentative.get("expanded_ARI"),
            "purity": v56_tentative.get("expanded_purity"),
            "completeness": v56_tentative.get("expanded_completeness"),
            "temporal_span_mean": None,
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic_labels": True,
        },
        row,
    ]


def _support_projection_assignments(selected: list[dict[str, Any]], support_index: dict[str, Any], *, mode: str) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    next_history = _next_history_map([str(row.get("history_id") or "") for row in selected if row.get("history_id")])
    for row in selected:
        observation_id = str(row.get("observation_id") or "")
        explanation = str(row.get("explanation_type") or "")
        history_id = str(row.get("history_id") or "")
        for component_id, support_count in support_index["components_by_mask"].get(observation_id, Counter()).items():
            gt = support_index["gt_by_mask"].get(observation_id)
            if not gt:
                continue
            if mode == "mask_only":
                pred = f"mask:{observation_id}"
            elif explanation in {"assign_to_existing", "partial_of_existing"} and history_id:
                pred = next_history.get(history_id, history_id) if mode == "shuffled" else history_id
            elif explanation == "underseg_mixture":
                pred = f"shared:{observation_id}"
            elif explanation == "new_object":
                pred = f"new:{observation_id}"
            else:
                pred = f"outlier:{observation_id}"
            for _ in range(max(1, int(support_count))):
                rows.append((pred, gt))
    return rows


def _projection_metrics(assignments: list[tuple[str, str]]) -> dict[str, float | None]:
    if not assignments:
        return {"ARI": None, "purity": None, "completeness": None}
    pred = [item[0] for item in assignments]
    true = [item[1] for item in assignments]
    return {
        "ARI": adjusted_rand_score(true, pred),
        "purity": cluster_purity(true, pred),
        "completeness": cluster_completeness(true, pred),
    }


def _semantic_scores(
    feature: np.ndarray,
    history_modes: dict[str, list[dict[str, Any]]],
    scene: str,
    history_meta: dict[str, dict[str, str]],
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for history_id, modes in history_modes.items():
        if scene and str(history_meta.get(history_id, {}).get("scene") or "") != scene:
            continue
        scores[history_id] = float(_score_modes(feature, modes))
    return scores


def _material_scores(observation_id: str, scene: str, support_index: dict[str, Any], history_index: dict[str, Any]) -> dict[str, float]:
    components = support_index["components_by_mask"].get(observation_id, Counter())
    total = float(sum(components.values()))
    scores: dict[str, float] = {}
    if total <= 0.0:
        return scores
    for history_id, hist_components in history_index["components_by_history"].items():
        if scene and history_index["scene_by_history"].get(history_id) != scene:
            continue
        hit = sum(count for component_id, count in components.items() if component_id in hist_components)
        if hit > 0:
            scores[history_id] = float(hit / total)
    return scores


def _candidate_histories(sem_scores: dict[str, float], mat_scores: dict[str, float], k_sem: int, k_mat: int) -> list[str]:
    sem = [history_id for history_id, _score in sorted(sem_scores.items(), key=lambda item: (-item[1], item[0]))[: int(k_sem)]]
    mat = [history_id for history_id, _score in sorted(mat_scores.items(), key=lambda item: (-item[1], item[0]))[: int(k_mat)]]
    out: list[str] = []
    for history_id in sem + mat:
        if history_id and history_id not in out:
            out.append(history_id)
    return out


def _observation_evidence(
    observation_id: str,
    scene: str,
    sem_scores: dict[str, float],
    mat_scores: dict[str, float],
    support_index: dict[str, Any],
    objectlet_index: dict[str, Any],
) -> dict[str, Any]:
    sem_sorted = sorted(sem_scores.values(), reverse=True)
    mat_sorted = sorted(mat_scores.values(), reverse=True)
    components = support_index["components_by_mask"].get(observation_id, Counter())
    total = float(sum(components.values()))
    objectlet_rows = objectlet_index["rows_by_mask"].get(observation_id, [])
    outside_vals = [parse_float(row.get("outside_all_related_masks_ratio_mean"), 0.0) for row in objectlet_rows if row.get("outside_all_related_masks_ratio_mean") not in (None, "")]
    return {
        "top_semantic_score": float(sem_sorted[0]) if sem_sorted else -1.0,
        "semantic_top2_gap": float(sem_sorted[0] - sem_sorted[1]) if len(sem_sorted) >= 2 else 1.0,
        "top_material_score": float(mat_sorted[0]) if mat_sorted else 0.0,
        "material_competition": float(mat_sorted[1] / max(mat_sorted[0], 1e-8)) if len(mat_sorted) >= 2 and mat_sorted[0] > 0 else 0.0,
        "component_entropy": _counter_entropy(components),
        "component_diversity": _component_diversity(scene, components, support_index["component_feature_by_key"]),
        "outside_residual": safe_mean(outside_vals) if outside_vals else 0.0,
        "mask_evidence_strength": min(1.0, math.log1p(total) / math.log(256.0)),
        "support_component_count": len(components),
    }


def _diagnostic_label(
    observation_id: str,
    scene: str,
    update_index: dict[str, Any],
    objectlet_index: dict[str, Any],
    support_index: dict[str, Any],
    history_index: dict[str, Any],
) -> dict[str, Any]:
    updates = update_index["rows_by_candidate"].get(observation_id, [])
    histories = {str(row.get("history_id") or "") for row in updates if row.get("history_id")}
    states = {str(row.get("update_state") or "") for row in updates}
    underseg_votes = objectlet_index["underseg_votes_by_mask"].get(observation_id, Counter())
    if underseg_votes and underseg_votes[True] > underseg_votes[False]:
        return {
            "expected_type": "underseg_mixture",
            "expected_history_ids": sorted(histories),
            "label_source": "v54_objectlet_underseg_proxy_diagnostic",
        }
    if len(histories) > 1:
        return {
            "expected_type": "underseg_mixture",
            "expected_history_ids": sorted(histories),
            "label_source": "multiple_v55_histories_for_same_candidate_diagnostic",
        }
    if "confirmed_update" in states and histories:
        return {
            "expected_type": "assign_to_existing",
            "expected_history_ids": sorted(histories),
            "label_source": "v55_confirmed_update_diagnostic",
        }
    if "partial_update" in states and histories:
        return {
            "expected_type": "partial_of_existing",
            "expected_history_ids": sorted(histories),
            "label_source": "v55_partial_update_diagnostic",
        }
    gt = support_index["gt_by_mask"].get(observation_id)
    if gt:
        history_ids = history_index["histories_by_scene_gt"].get((scene, gt), [])
        if history_ids:
            return {
                "expected_type": "assign_to_existing",
                "expected_history_ids": sorted(history_ids),
                "label_source": "support_dominant_gt_matches_existing_history_diagnostic",
            }
        return {
            "expected_type": "new_object",
            "expected_history_ids": [],
            "label_source": "support_dominant_gt_absent_from_history_diagnostic",
        }
    return {
        "expected_type": "outlier_or_background",
        "expected_history_ids": [],
        "label_source": "no_support_dominant_gt_diagnostic",
    }


def _diagnostic_correct(explanation_type: str, history_id: str, diagnostic: dict[str, Any]) -> bool | None:
    expected_type = diagnostic.get("expected_type")
    expected_histories = set(diagnostic.get("expected_history_ids") or [])
    if expected_type in (None, ""):
        return None
    if explanation_type != expected_type:
        return False
    if explanation_type in {"assign_to_existing", "partial_of_existing"}:
        return history_id in expected_histories
    if explanation_type == "underseg_mixture":
        return True
    return True


def _load_mask_features(path: Path) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    features: dict[str, np.ndarray] = {}
    meta: dict[str, dict[str, Any]] = {}
    for row in _read_csv(path):
        mask_id = str(row.get("mask_observation_id") or "")
        if not mask_id or not parse_bool(row.get("feature_available")):
            continue
        raw = str(row.get("feature_json") or "").strip()
        if not raw:
            continue
        vector = _normalize(np.asarray(json.loads(raw), dtype=np.float32))
        if not vector.size:
            continue
        features[mask_id] = vector
        meta[mask_id] = row
    return features, meta


def _rebuild_history_modes(
    history_sample_rows: list[dict[str, str]],
    mask_features: dict[str, np.ndarray],
    history_meta: dict[str, dict[str, str]],
    *,
    max_modes: int,
) -> dict[str, list[dict[str, Any]]]:
    samples: dict[str, list[np.ndarray]] = defaultdict(list)
    for row in history_sample_rows:
        history_id = str(row.get("history_id") or "")
        mask_id = str(row.get("mask_observation_id") or "")
        feature = mask_features.get(mask_id)
        if history_id in history_meta and feature is not None and feature.size:
            samples[history_id].append(feature)
    return {history_id: _fit_modes(vectors, max_modes=max_modes) for history_id, vectors in samples.items()}


def _build_support_index(rows: list[dict[str, str]]) -> dict[str, Any]:
    components_by_mask: dict[str, Counter[str]] = defaultdict(Counter)
    gt_votes_by_mask: dict[str, Counter[str]] = defaultdict(Counter)
    scene_by_mask: dict[str, str] = {}
    component_feature_accum: dict[tuple[str, str], list[tuple[float, np.ndarray]]] = defaultdict(list)
    for row in rows:
        mask_id = str(row.get("mask_observation_id") or "")
        scene = str(row.get("scene") or "")
        component = str(row.get("component_id") or "")
        support = max(parse_int(row.get("support_count"), 1), 1)
        if mask_id and component:
            components_by_mask[mask_id][component] += support
            scene_by_mask[mask_id] = scene
        gt = str(row.get("diagnostic_gt_instance") or "")
        if mask_id and gt:
            gt_votes_by_mask[mask_id][f"{scene}|{gt}"] += support
    gt_by_mask = {mask_id: votes.most_common(1)[0][0] for mask_id, votes in gt_votes_by_mask.items() if votes}
    # Component feature vectors are reconstructed from dominant mask support lazily by the caller when available.
    return {
        "components_by_mask": components_by_mask,
        "scene_by_mask": scene_by_mask,
        "gt_by_mask": gt_by_mask,
        "component_feature_by_key": {},
    }


def _build_history_index(rows: list[dict[str, str]], history_meta: dict[str, dict[str, str]]) -> dict[str, Any]:
    components_by_history: dict[str, set[str]] = defaultdict(set)
    scene_by_history: dict[str, str] = {}
    histories_by_scene_gt: dict[tuple[str, str], list[str]] = defaultdict(list)
    for history_id, row in history_meta.items():
        scene = str(row.get("scene") or "")
        scene_by_history[history_id] = scene
        gt = str(row.get("dominant_gt_diagnostic") or "")
        if scene and gt:
            histories_by_scene_gt[(scene, f"{scene}|{gt}")].append(history_id)
    for row in rows:
        history_id = str(row.get("objectlet_id") or "")
        if history_id not in history_meta:
            continue
        components_by_history[history_id].update(_load_json_list(row.get("component_ids")))
    return {
        "components_by_history": components_by_history,
        "scene_by_history": scene_by_history,
        "histories_by_scene_gt": histories_by_scene_gt,
    }


def _build_update_index(rows: list[dict[str, str]]) -> dict[str, Any]:
    rows_by_candidate: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        candidate = str(row.get("candidate_id") or "")
        if candidate:
            rows_by_candidate[candidate].append(row)
    return {"rows_by_candidate": rows_by_candidate}


def _build_objectlet_index(rows: list[dict[str, str]]) -> dict[str, Any]:
    rows_by_mask: dict[str, list[dict[str, str]]] = defaultdict(list)
    votes: dict[str, Counter[bool]] = defaultdict(Counter)
    for row in rows:
        mask_id = str(row.get("source_mask_observation_id") or "")
        if not mask_id:
            continue
        rows_by_mask[mask_id].append(row)
        raw = str(row.get("underseg_proxy") or "").strip().lower()
        if raw in {"true", "false", "1", "0"}:
            votes[mask_id][raw in {"true", "1"}] += 1
    return {"rows_by_mask": rows_by_mask, "underseg_votes_by_mask": votes}


def _select_observations(
    mask_features: dict[str, np.ndarray],
    support_index: dict[str, Any],
    update_index: dict[str, Any],
    objectlet_index: dict[str, Any],
    *,
    max_count: int | None,
) -> list[str]:
    priority: list[str] = []
    for source in [
        sorted(update_index["rows_by_candidate"]),
        sorted(objectlet_index["rows_by_mask"]),
        sorted(support_index["components_by_mask"]),
        sorted(mask_features),
    ]:
        for mask_id in source:
            if mask_id in mask_features and mask_id not in priority:
                priority.append(mask_id)
    if max_count is not None:
        priority = priority[: int(max_count)]
    return priority


def _counter_entropy(counter: Counter[str]) -> float:
    total = float(sum(counter.values()))
    if total <= 0.0 or len(counter) <= 1:
        return 0.0
    probs = [value / total for value in counter.values() if value > 0]
    entropy = -sum(prob * math.log(prob) for prob in probs)
    return float(entropy / max(math.log(len(probs)), 1e-8))


def _component_diversity(scene: str, counter: Counter[str], component_features: dict[tuple[str, str], np.ndarray]) -> float:
    # Phase2 currently uses support entropy for component-level multimodality.
    # The hook remains explicit so future dense component vectors can be wired without changing row schema.
    return 0.0


def _posterior_entropy(probs: list[float]) -> float:
    return float(-sum(prob * math.log(max(prob, 1e-12)) for prob in probs if prob > 0.0))


def _posterior_margin(probs: list[float]) -> float:
    top = sorted(probs, reverse=True)
    return float(top[0] - top[1]) if len(top) >= 2 else 1.0


def _decision_state(row: dict[str, Any], top: dict[str, Any], entropy: float, margin: float) -> str:
    explanation = str(top.get("explanation_type") or "")
    posterior = float(top.get("posterior") or 0.0)
    has_update = parse_bool(top.get("has_update_candidate"))
    has_objectlet = parse_bool(top.get("has_objectlet_candidate"))
    if explanation == "underseg_mixture" and (margin < 0.64 or posterior < 0.75):
        return "defer_to_active_query"
    if explanation == "underseg_mixture" and not (has_update or has_objectlet):
        return "defer_to_active_query"
    if explanation == "assign_to_existing" and not (has_update or has_objectlet) and _num(top.get("semantic_score"), -1.0) < 0.84:
        return "defer_to_active_query"
    if explanation == "partial_of_existing" and margin < 0.30:
        return "defer_to_active_query"
    if explanation == "assign_to_existing" and margin < 0.32:
        return "defer_to_active_query"
    if entropy > 2.2 and margin < 0.20:
        return "defer_to_active_query"
    return "actionable"


def _precision(values: list[bool]) -> float | None:
    return _safe_div(sum(1 for value in values if value), len(values))


def _safe_div(num: int | float, den: int | float) -> float | None:
    return None if float(den) == 0.0 else float(num) / float(den)


def _num(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return float(default)
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def _v56_false_history_rate(v56_core: dict[str, Any]) -> float | None:
    precision = v56_core.get("update_precision_diagnostic")
    if precision in (None, ""):
        return None
    return float(1.0 - float(precision))


def _load_json_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    text = str(value or "").strip()
    if not text:
        return []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return [item.strip() for item in text.split(";") if item.strip()]
    return [str(item) for item in payload]


def _variant_specs() -> list[dict[str, Any]]:
    return [
        {"name": "E1_semantic_only", "sem_weight": 1.0, "mat_weight": 0.0, "partial": True, "underseg": False},
        {"name": "E2_material_only", "sem_weight": 0.0, "mat_weight": 1.0, "partial": True, "underseg": False},
        {"name": "E3_semantic_material_pair_assignment", "sem_weight": 1.0, "mat_weight": 1.0, "partial": False, "underseg": False},
        {"name": "E4_counterfactual_without_underseg", "sem_weight": 1.0, "mat_weight": 1.0, "partial": True, "underseg": False},
        {"name": "E5_counterfactual_with_underseg", "sem_weight": 1.0, "mat_weight": 1.0, "partial": True, "underseg": True},
        {"name": "E6_counterfactual_semantic_material_underseg", "sem_weight": 1.2, "mat_weight": 1.1, "partial": True, "underseg": True},
    ]


def _temporal_span(selected: list[dict[str, Any]]) -> float | None:
    frames_by_cluster: dict[str, set[int]] = defaultdict(set)
    for row in selected:
        explanation = str(row.get("explanation_type") or "")
        history_id = str(row.get("history_id") or "")
        if explanation in {"assign_to_existing", "partial_of_existing"} and history_id:
            frames_by_cluster[history_id].add(parse_int(row.get("frame_id")))
    spans = [len(frames) for frames in frames_by_cluster.values() if frames]
    return safe_mean(spans)


def _duplicate_rate(selected: list[dict[str, Any]], support_index: dict[str, Any]) -> float | None:
    pred_by_component: dict[str, set[str]] = defaultdict(set)
    for row in selected:
        explanation = str(row.get("explanation_type") or "")
        history_id = str(row.get("history_id") or "")
        if explanation not in {"assign_to_existing", "partial_of_existing"} or not history_id:
            continue
        for component_id in support_index["components_by_mask"].get(str(row.get("observation_id") or ""), Counter()):
            pred_by_component[component_id].add(history_id)
    if not pred_by_component:
        return None
    return float(sum(1 for histories in pred_by_component.values() if len(histories) > 1) / len(pred_by_component))


def _conflict_rate(selected: list[dict[str, Any]]) -> float | None:
    histories_by_scene_frame: dict[tuple[str, int], set[str]] = defaultdict(set)
    total = 0
    conflict = 0
    for row in selected:
        explanation = str(row.get("explanation_type") or "")
        history_id = str(row.get("history_id") or "")
        if explanation not in {"assign_to_existing", "partial_of_existing"} or not history_id:
            continue
        key = (str(row.get("scene") or ""), parse_int(row.get("frame_id")))
        before = len(histories_by_scene_frame[key])
        was_present = history_id in histories_by_scene_frame[key]
        histories_by_scene_frame[key].add(history_id)
        total += 1
        if before and not was_present:
            conflict += 1
    return _safe_div(conflict, total)


def _next_history_map(history_ids: list[str]) -> dict[str, str]:
    by_scene: dict[str, list[str]] = defaultdict(list)
    for history_id in history_ids:
        scene = history_id.split("|", 1)[0] if "|" in history_id else ""
        if history_id not in by_scene[scene]:
            by_scene[scene].append(history_id)
    out: dict[str, str] = {}
    for ids in by_scene.values():
        ids = sorted(ids)
        for idx, history_id in enumerate(ids):
            out[history_id] = ids[(idx + 1) % len(ids)] if ids else history_id
    return out


def _frame_from_observation(observation_id: str) -> int | None:
    parts = observation_id.split(":")
    return parse_int(parts[1]) if len(parts) >= 3 else None


def _mask_from_observation(observation_id: str) -> int | None:
    parts = observation_id.split(":")
    return parse_int(parts[2]) if len(parts) >= 3 else None


def _row_history_ids(explanation_type: str, history_id: str) -> list[str]:
    if not history_id:
        return []
    if explanation_type == "underseg_mixture":
        return [item for item in history_id.split("||") if item]
    return [history_id]


def _fmt(value: Any) -> str:
    if value in (None, ""):
        return "NA"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


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


def dumps_json(payload: Any) -> str:
    return json.dumps(json_safe(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
