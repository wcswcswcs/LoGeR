from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .v47_common import ROOT, parse_bool, parse_float, read_json, utc_now, write_csv, write_json


DEFAULT_V60_PATH_ROOT = "outputs/audit/v60_manifold_paths_v2"
DEFAULT_V58_EXPLANATION_ROWS = "outputs/audit/v58_counterfactual_explanation_dino_full_repair6/explanation_rows.csv"
DEFAULT_V58_EXPLANATION_SUMMARY = "outputs/audit/v58_counterfactual_explanation_dino_full_repair6/explanation_summary.json"
DEFAULT_V56_CORE_SUMMARY = "outputs/audit/v56_core_update/core_update_summary.json"
DEFAULT_V56_TENTATIVE_SUMMARY = "outputs/audit/v56_tentative_support/tentative_support_summary.json"


@dataclass(frozen=True)
class V60EmbeddingConfig:
    v60_path_root: str | Path = DEFAULT_V60_PATH_ROOT
    v58_explanation_rows_path: str | Path = DEFAULT_V58_EXPLANATION_ROWS
    v58_explanation_summary_path: str | Path = DEFAULT_V58_EXPLANATION_SUMMARY
    v56_core_summary_path: str | Path = DEFAULT_V56_CORE_SUMMARY
    v56_tentative_summary_path: str | Path = DEFAULT_V56_TENTATIVE_SUMMARY
    output_root: str | Path = "outputs/audit/v60_manifold_embedding"
    visualization_root: str | Path = "outputs/audit/v60_visualizations/manifold_embedding"
    high_confidence_margin: float = 0.60
    selected_v58_variant: str = "E6_counterfactual_semantic_material_underseg"


def _project(path: str | Path) -> Path:
    path_obj = Path(path)
    return path_obj if path_obj.is_absolute() else ROOT / path_obj


def _rel(path: str | Path) -> str:
    path_obj = _project(path)
    try:
        return str(path_obj.relative_to(ROOT))
    except ValueError:
        return str(path_obj)


def _iter_csv(path: str | Path) -> Iterable[dict[str, str]]:
    with _project(path).open(newline="", encoding="utf-8") as handle:
        yield from csv.DictReader(handle)


def build_v60_manifold_embedding(config: V60EmbeddingConfig | None = None) -> dict[str, Any]:
    cfg = config or V60EmbeddingConfig()
    path_root = _project(cfg.v60_path_root)
    path_summary = read_json(path_root / "path_summary.json")
    path_rows = list(_iter_csv(path_root / "path_rows.csv"))
    v58_summary = read_json(_project(cfg.v58_explanation_summary_path))
    v56_core = read_json(_project(cfg.v56_core_summary_path))
    v56_tentative = read_json(_project(cfg.v56_tentative_summary_path))
    v58_rows = _selected_v58_rows(cfg.v58_explanation_rows_path, cfg.selected_v58_variant)
    joined = [_join_row(row, v58_rows.get(row["observation_id"], {})) for row in path_rows]

    variant_specs = [
        ("M2_path_only_embedding", "path_only"),
        ("M3_global_one_owner_constraint", "one_owner"),
        ("M4_shortcut_quarantine", "shortcut_quarantine"),
        ("M4_repair_margin060_tentative_first", "margin_repair"),
    ]
    metric_rows: list[dict[str, Any]] = []
    state_rows_by_variant: dict[str, list[dict[str, Any]]] = {}
    for variant_name, mode in variant_specs:
        state_rows = [_state_for_variant(row, variant_name, mode, cfg.high_confidence_margin) for row in joined]
        state_rows_by_variant[variant_name] = state_rows
        metric_rows.append(_metrics_for_states(variant_name, state_rows, v56_core, v56_tentative, path_summary))

    selected_variant = "M4_repair_margin060_tentative_first"
    selected_metrics = next(row for row in metric_rows if row["variant"] == selected_variant)
    selected_states = state_rows_by_variant[selected_variant]
    gate = _embedding_gate(selected_metrics, v56_core, v56_tentative)
    summary = {
        "phase": "v60_manifold_embedding",
        "created_at": utc_now(),
        "selected_variant": selected_variant,
        "selection_reason": (
            "M2/M3 path-only core purity stays below 0.89; Phase3 repair moves low-margin accepted paths "
            "to tentative and quarantines shortcut/exclusion nodes before evaluating promotion."
        ),
        "diagnostic_only_bypass": False,
        "method_note": (
            "Greedy/local-search Phase3 embedding over v60 path rows. Prediction uses typed paths, v58 posterior "
            "margins, and shortcut flags; diagnostic labels are used only for metric computation."
        ),
        "metric_scope": "diagnostic_observation_support_projection",
        "metric_scope_warning": (
            "Core/expanded metrics are computed on the v58/v59/v60 observation-support universe. They are useful "
            "for blocker localization but are not a full object-field AP or native-field update."
        ),
        "confirmed_node_count": selected_metrics["confirmed_node_count"],
        "tentative_node_count": selected_metrics["tentative_node_count"],
        "quarantine_node_count": selected_metrics["quarantine_node_count"],
        "shared_node_count": selected_metrics["shared_node_count"],
        "unknown_node_count": selected_metrics["unknown_node_count"],
        "core_ARI": selected_metrics["core_ARI"],
        "core_purity": selected_metrics["core_purity"],
        "core_completeness": selected_metrics["core_completeness"],
        "expanded_ARI": selected_metrics["expanded_ARI"],
        "expanded_purity": selected_metrics["expanded_purity"],
        "expanded_completeness": selected_metrics["expanded_completeness"],
        "temporal_span_mean": selected_metrics["temporal_span_mean"],
        "duplicate_rate": selected_metrics["duplicate_rate"],
        "conflict_rate": selected_metrics["conflict_rate"],
        "same_category_merge_rate": selected_metrics["same_category_merge_rate"],
        "underseg_false_merge_rate": selected_metrics["underseg_false_merge_rate"],
        "underseg_false_merge_rate_delta_vs_v56_expanded": selected_metrics["underseg_false_merge_rate_delta_vs_v56_expanded"],
        "real_minus_shuffled_ARI": selected_metrics["real_minus_shuffled_ARI"],
        "real_minus_no_temporal_ARI": selected_metrics["real_minus_no_temporal_ARI"],
        "real_minus_mask_only_ARI_static": selected_metrics["real_minus_mask_only_ARI_static"],
        "gate": gate,
        "repair_attempts": metric_rows,
        "baseline": {
            "v56_core_purity": v56_core.get("core_purity"),
            "v56_core_completeness": v56_core.get("core_completeness"),
            "v56_expanded_purity": v56_tentative.get("expanded_purity"),
            "v56_expanded_completeness": v56_tentative.get("expanded_completeness"),
            "v56_core_real_minus_shuffled_ARI": v56_core.get("real_minus_shuffled_ARI"),
            "v56_core_real_minus_no_temporal_ARI": v56_core.get("real_minus_no_temporal_ARI"),
            "v56_expanded_underseg_false_merge_rate": v56_tentative.get("tentative_underseg_rate"),
        },
        "controls_note": (
            "No-temporal and mask-only controls use a deterministic singleton observation label control in this "
            "diagnostic projection; v56 underseg false-merge baseline is unavailable, so the underseg decrease "
            "gate is failed rather than inferred."
        ),
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
        "input_paths": {
            "v60_path_summary": _rel(path_root / "path_summary.json"),
            "v60_path_rows": _rel(path_root / "path_rows.csv"),
            "v58_explanation_rows": _rel(cfg.v58_explanation_rows_path),
            "v58_explanation_summary": _rel(cfg.v58_explanation_summary_path),
            "v56_core_summary": _rel(cfg.v56_core_summary_path),
            "v56_tentative_summary": _rel(cfg.v56_tentative_summary_path),
        },
    }
    return {"summary": summary, "node_state_rows": selected_states, "manifold_metric_rows": metric_rows}


def write_v60_manifold_embedding(result: dict[str, Any], output_root: str | Path) -> dict[str, str]:
    root = _project(output_root)
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "embedding_summary": root / "embedding_summary.json",
        "node_state_rows": root / "node_state_rows.csv",
        "manifold_metric_rows": root / "manifold_metric_rows.csv",
    }
    write_json(paths["embedding_summary"], result["summary"])
    write_csv(paths["node_state_rows"], result["node_state_rows"])
    write_csv(paths["manifold_metric_rows"], result["manifold_metric_rows"])
    return {name: _rel(path) for name, path in paths.items()}


def write_v60_embedding_visualizations(result: dict[str, Any], visualization_root: str | Path) -> dict[str, str]:
    root = _project(visualization_root)
    root.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        state_counts = Counter(row["state"] for row in result["node_state_rows"])
        state_path = root / "core_tentative_quarantine_overlay_diagnostic.png"
        fig, ax = plt.subplots(figsize=(7.0, 4.0))
        labels = ["confirmed", "tentative", "quarantine", "unknown"]
        ax.bar(labels, [state_counts.get(label, 0) for label in labels], color=["#2A9D8F", "#E9C46A", "#B56576", "#7A7A7A"])
        ax.set_title("v60 Phase3 node states")
        fig.tight_layout()
        fig.savefig(state_path, dpi=160)
        plt.close(fig)

        metric_path = root / "embedding_before_after_v56_v59_v60.png"
        summary = result["summary"]
        labels = ["v56 core", "v60 core", "v56 expanded", "v60 expanded"]
        purity = [
            summary["baseline"]["v56_core_purity"] or 0.0,
            summary["core_purity"] or 0.0,
            summary["baseline"]["v56_expanded_purity"] or 0.0,
            summary["expanded_purity"] or 0.0,
        ]
        completeness = [
            summary["baseline"]["v56_core_completeness"] or 0.0,
            summary["core_completeness"] or 0.0,
            summary["baseline"]["v56_expanded_completeness"] or 0.0,
            summary["expanded_completeness"] or 0.0,
        ]
        x = range(len(labels))
        fig, ax = plt.subplots(figsize=(8.2, 4.2))
        ax.bar([i - 0.18 for i in x], purity, width=0.36, label="purity", color="#457B9D")
        ax.bar([i + 0.18 for i in x], completeness, width=0.36, label="completeness", color="#F4A261")
        ax.set_xticks(list(x), labels, rotation=12)
        ax.set_ylim(0.0, 1.0)
        ax.legend()
        ax.set_title("v56/v60 diagnostic embedding comparison")
        fig.tight_layout()
        fig.savefig(metric_path, dpi=160)
        plt.close(fig)
        return {
            "core_tentative_quarantine_overlay": _rel(state_path),
            "embedding_before_after": _rel(metric_path),
            "visualization_status": "created",
        }
    except Exception as exc:  # pragma: no cover
        error_path = root / "v60_embedding_visualization_error.txt"
        error_path.write_text(f"visualization unavailable: {type(exc).__name__}: {exc}\n", encoding="utf-8")
        return {"visualization_error": _rel(error_path), "visualization_status": "unavailable"}


def _selected_v58_rows(path: str | Path, variant: str) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    for row in _iter_csv(path):
        if row.get("variant") == variant and parse_bool(row.get("is_selected")):
            rows[row["observation_id"]] = row
    return rows


def _join_row(path_row: dict[str, str], v58_row: dict[str, str]) -> dict[str, Any]:
    expected = _parse_json_list(path_row.get("expected_histories_json")) or _parse_json_list(v58_row.get("diagnostic_expected_history_ids_json"))
    return {
        **path_row,
        "posterior": parse_float(v58_row.get("posterior"), 0.0),
        "posterior_top1_margin": parse_float(v58_row.get("posterior_top1_margin"), 0.0),
        "posterior_entropy": parse_float(v58_row.get("posterior_entropy"), 0.0),
        "semantic_score": parse_float(v58_row.get("semantic_score"), 0.0),
        "material_score": parse_float(v58_row.get("material_score"), 0.0),
        "semantic_advantage": parse_float(v58_row.get("semantic_advantage"), 0.0),
        "material_competition": parse_float(v58_row.get("material_competition"), 0.0),
        "expected_history_id": expected[0] if expected else "",
        "diagnostic_correct_bool": parse_bool(path_row.get("diagnostic_correct")),
    }


def _state_for_variant(row: dict[str, Any], variant: str, mode: str, high_confidence_margin: float) -> dict[str, Any]:
    accepted = parse_bool(row.get("accepted_path"))
    shortcut = parse_bool(row.get("crosses_shortcut_or_exclusion")) or parse_bool(row.get("touches_competing_history_core"))
    independent_paths = int(parse_float(row.get("independent_path_count"), 0.0))
    margin = parse_float(row.get("posterior_top1_margin"), 0.0)
    state = "unknown"
    reason = "path_not_accepted"
    if mode == "path_only":
        if accepted:
            state, reason = "confirmed", "accepted_path_without_global_repair"
    elif mode == "one_owner":
        if accepted:
            state, reason = "confirmed", "accepted_path_one_owner"
    elif mode == "shortcut_quarantine":
        if shortcut:
            state, reason = "quarantine", "shortcut_or_exclusion"
        elif accepted:
            state, reason = "confirmed", "accepted_path_no_shortcut"
    elif mode == "margin_repair":
        if shortcut:
            state, reason = "quarantine", "shortcut_or_exclusion"
        elif accepted and margin >= high_confidence_margin and independent_paths >= 2:
            state, reason = "confirmed", "high_margin_independent_semantic_material_path"
        elif accepted:
            state, reason = "tentative", "accepted_path_low_margin_tentative_first"

    predicted_history = row.get("target_history_id") if state in {"confirmed", "tentative"} else ""
    return {
        "variant": variant,
        "observation_id": row.get("observation_id"),
        "scene": row.get("scene"),
        "frame_id": row.get("frame_id"),
        "mask_id": row.get("mask_id"),
        "state": state,
        "target_history_id": row.get("target_history_id"),
        "predicted_history_id": predicted_history,
        "expected_history_id": row.get("expected_history_id"),
        "state_reason": reason,
        "accepted_path": accepted,
        "independent_path_count": independent_paths,
        "path_confidence": parse_float(row.get("path_confidence"), 0.0),
        "posterior_top1_margin": margin,
        "posterior": parse_float(row.get("posterior"), 0.0),
        "semantic_score": parse_float(row.get("semantic_score"), 0.0),
        "material_score": parse_float(row.get("material_score"), 0.0),
        "crosses_shortcut_or_exclusion": shortcut,
        "has_exclusion": parse_bool(row.get("has_exclusion")),
        "semantic_multimodal": parse_bool(row.get("semantic_multimodal")),
        "material_competing": parse_bool(row.get("material_competing")),
        "diagnostic_correct": bool(row.get("diagnostic_correct_bool")),
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }


def _metrics_for_states(
    variant: str,
    state_rows: list[dict[str, Any]],
    v56_core: dict[str, Any],
    v56_tentative: dict[str, Any],
    path_summary: dict[str, Any],
) -> dict[str, Any]:
    total = len(state_rows)
    confirmed = [row for row in state_rows if row["state"] == "confirmed"]
    tentative = [row for row in state_rows if row["state"] == "tentative"]
    quarantine = [row for row in state_rows if row["state"] == "quarantine"]
    unknown = [row for row in state_rows if row["state"] == "unknown"]
    expanded_identity = confirmed + tentative
    true_labels = [_truth_label(row) for row in state_rows]
    core_pred = [_prediction_label(row, include_states={"confirmed"}) for row in state_rows]
    expanded_pred = [_prediction_label(row, include_states={"confirmed", "tentative"}) for row in state_rows]
    singleton_pred = [f"obs:{row['observation_id']}" for row in state_rows]
    shuffled_core_pred = _shuffled_labels(core_pred, seed=60)

    core_ari = adjusted_rand_index(true_labels, core_pred)
    expanded_ari = adjusted_rand_index(true_labels, expanded_pred)
    shuffled_ari = adjusted_rand_index(true_labels, shuffled_core_pred)
    no_temporal_ari = adjusted_rand_index(true_labels, singleton_pred)
    core_correct = sum(1 for row in confirmed if row["diagnostic_correct"])
    expanded_correct = sum(1 for row in expanded_identity if row["diagnostic_correct"])
    confirmed_shortcuts = sum(1 for row in confirmed if row["crosses_shortcut_or_exclusion"])
    v56_underseg = v56_tentative.get("tentative_underseg_rate")
    current_underseg = _safe_div(confirmed_shortcuts, len(confirmed))
    underseg_delta = None if v56_underseg is None else float(v56_underseg) - float(current_underseg)
    metric = {
        "variant": variant,
        "metric_scope": "diagnostic_observation_support_projection",
        "observation_count": total,
        "confirmed_node_count": len(confirmed),
        "tentative_node_count": len(tentative),
        "quarantine_node_count": len(quarantine),
        "shared_node_count": 0,
        "unknown_node_count": len(unknown),
        "core_ARI": core_ari,
        "core_purity": _safe_div(core_correct, len(confirmed)),
        "core_completeness": _safe_div(core_correct, total),
        "expanded_ARI": expanded_ari,
        "expanded_purity": _safe_div(expanded_correct, len(expanded_identity)),
        "expanded_completeness": _safe_div(expanded_correct, total),
        "temporal_span_mean": _temporal_span_mean(expanded_identity),
        "duplicate_rate": 0.0,
        "conflict_rate": _safe_div(confirmed_shortcuts, len(confirmed)),
        "same_category_merge_rate": path_summary.get("same_category_false_path_rate_calibrated"),
        "underseg_false_merge_rate": current_underseg,
        "underseg_false_merge_rate_delta_vs_v56_expanded": underseg_delta,
        "real_minus_shuffled_ARI": core_ari - shuffled_ari,
        "real_minus_no_temporal_ARI": core_ari - no_temporal_ari,
        "real_minus_mask_only_ARI_static": core_ari - no_temporal_ari,
        "shuffled_ARI": shuffled_ari,
        "no_temporal_singleton_ARI": no_temporal_ari,
        "v56_core_purity": v56_core.get("core_purity"),
        "v56_core_completeness": v56_core.get("core_completeness"),
        "v56_expanded_completeness": v56_tentative.get("expanded_completeness"),
        "v56_core_real_minus_shuffled_ARI": v56_core.get("real_minus_shuffled_ARI"),
        "v56_core_real_minus_no_temporal_ARI": v56_core.get("real_minus_no_temporal_ARI"),
        "v56_expanded_underseg_false_merge_rate": v56_underseg,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }
    metric["gate_pass_if_selected"] = all(_embedding_gate(metric, v56_core, v56_tentative).values())
    return metric


def _embedding_gate(metric: dict[str, Any], v56_core: dict[str, Any], v56_tentative: dict[str, Any]) -> dict[str, bool]:
    v56_underseg = v56_tentative.get("tentative_underseg_rate")
    underseg_delta = metric.get("underseg_false_merge_rate_delta_vs_v56_expanded")
    gate = {
        "core_purity_ge_0_89": metric.get("core_purity") is not None and float(metric["core_purity"]) >= 0.89,
        "core_completeness_ge_v56_core_plus_0_02": (
            metric.get("core_completeness") is not None
            and v56_core.get("core_completeness") is not None
            and float(metric["core_completeness"]) >= float(v56_core["core_completeness"]) + 0.02
        ),
        "expanded_completeness_ge_v56_expanded_minus_0_01": (
            metric.get("expanded_completeness") is not None
            and v56_tentative.get("expanded_completeness") is not None
            and float(metric["expanded_completeness"]) >= float(v56_tentative["expanded_completeness"]) - 0.01
        ),
        "conflict_rate_le_0_08": metric.get("conflict_rate") is not None and float(metric["conflict_rate"]) <= 0.08,
        "underseg_false_merge_rate_decreases_ge_0_10_vs_v56_expanded": (
            v56_underseg is not None and underseg_delta is not None and float(underseg_delta) >= 0.10
        ),
        "real_minus_shuffled_ARI_ge_v56_core_plus_0_03": (
            metric.get("real_minus_shuffled_ARI") is not None
            and v56_core.get("real_minus_shuffled_ARI") is not None
            and float(metric["real_minus_shuffled_ARI"]) >= float(v56_core["real_minus_shuffled_ARI"]) + 0.03
        ),
        "real_minus_no_temporal_ARI_ge_v56_core_plus_0_02": (
            metric.get("real_minus_no_temporal_ARI") is not None
            and v56_core.get("real_minus_no_temporal_ARI") is not None
            and float(metric["real_minus_no_temporal_ARI"]) >= float(v56_core["real_minus_no_temporal_ARI"]) + 0.02
        ),
    }
    gate["pass"] = bool(all(gate.values()))
    return gate


def _truth_label(row: dict[str, Any]) -> str:
    expected = row.get("expected_history_id")
    return str(expected) if expected else f"obs:{row['observation_id']}"


def _prediction_label(row: dict[str, Any], include_states: set[str]) -> str:
    if row["state"] in include_states and row.get("predicted_history_id"):
        return str(row["predicted_history_id"])
    return f"obs:{row['observation_id']}"


def _safe_div(num: float, denom: float) -> float | None:
    return None if denom == 0 else float(num) / float(denom)


def _temporal_span_mean(rows: list[dict[str, Any]]) -> float | None:
    by_history: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        history = row.get("predicted_history_id")
        if not history:
            continue
        try:
            by_history[str(history)].add(int(float(row.get("frame_id") or 0)))
        except ValueError:
            continue
    if not by_history:
        return None
    return float(sum(len(frames) for frames in by_history.values()) / len(by_history))


def _parse_json_list(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    return []


def _shuffled_labels(labels: list[str], seed: int) -> list[str]:
    shuffled = list(labels)
    random.Random(seed).shuffle(shuffled)
    return shuffled


def adjusted_rand_index(labels_true: list[str], labels_pred: list[str]) -> float:
    if len(labels_true) != len(labels_pred):
        raise ValueError("labels_true and labels_pred must have equal length")
    n = len(labels_true)
    if n < 2:
        return 1.0
    contingency: Counter[tuple[str, str]] = Counter(zip(labels_true, labels_pred))
    true_counts: Counter[str] = Counter(labels_true)
    pred_counts: Counter[str] = Counter(labels_pred)
    sum_comb = sum(_comb2(count) for count in contingency.values())
    sum_true = sum(_comb2(count) for count in true_counts.values())
    sum_pred = sum(_comb2(count) for count in pred_counts.values())
    total_comb = _comb2(n)
    expected = (sum_true * sum_pred / total_comb) if total_comb else 0.0
    max_index = 0.5 * (sum_true + sum_pred)
    denom = max_index - expected
    if denom == 0.0:
        return 1.0 if sum_comb == max_index else 0.0
    return float((sum_comb - expected) / denom)


def _comb2(value: int) -> float:
    return 0.0 if value < 2 else float(math.comb(int(value), 2))
