from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .v47_common import ROOT, parse_bool, parse_float, rank_auc, read_json, utc_now, write_csv, write_json


DEFAULT_REFINED_STATES = "outputs/audit/v61_refinement/material_state_after_refinement.csv"
DEFAULT_OBSERVATIONS = "outputs/audit/v61_global_embedding/observation_explanation_rows.csv"
DEFAULT_V58_QUERY_ROOT = "outputs/audit/v58_active_material_query_q128_repair5_expanded_all_minvis1"


@dataclass(frozen=True)
class V61ManifoldQueryConfig:
    refined_state_rows_path: str | Path = DEFAULT_REFINED_STATES
    observation_explanation_rows_path: str | Path = DEFAULT_OBSERVATIONS
    v58_query_root: str | Path = DEFAULT_V58_QUERY_ROOT
    output_root: str | Path = "outputs/audit/v61_manifold_query"
    visualization_root: str | Path = "outputs/audit/v61_visualizations/query"
    query_budget: int = 128


def build_v61_manifold_query(config: V61ManifoldQueryConfig | None = None) -> dict[str, Any]:
    cfg = config or V61ManifoldQueryConfig()
    query_root = _project(cfg.v58_query_root)
    baseline_rows = [_baseline_metric(row) for row in _iter_csv(query_root / "query_metric_rows.csv")]
    source_query_rows = list(_iter_csv(query_root / "query_rows.csv"))
    source_material_rows = list(_iter_csv(query_root / "material_evidence_rows.csv"))
    state_rows = list(_iter_csv(cfg.refined_state_rows_path))
    observation_rows = list(_iter_csv(cfg.observation_explanation_rows_path))
    obs_stats = _observation_stats_from_states(state_rows)
    obs_explanations = _observation_explanations(observation_rows)
    selected_rows, pool_count = _select_q7_rows(
        source_query_rows,
        obs_stats,
        obs_explanations,
        query_budget=cfg.query_budget,
    )
    upper_bound = _diagnostic_upper_bound(
        source_query_rows,
        obs_stats,
        obs_explanations,
        query_budget=cfg.query_budget,
    )
    selected_source_query_ids = {row["source_query_id"] for row in selected_rows}
    material_rows = [row for row in source_material_rows if row.get("query_id") in selected_source_query_ids]
    q7_metrics = _metrics_for_query_rows("Q7", "v61 manifold boundary/gap/shortcut query", selected_rows, cfg.query_budget)
    q7_metrics["candidate_pool_count"] = pool_count
    q7_metrics["query_type_breakdown"] = dict(Counter(row["query_type"] for row in selected_rows))

    best_fixed_entropy = max((row["state_entropy_reduction"] for row in baseline_rows if row["baseline_id"] != "Q7"), default=0.0)
    best_fixed_confirm_or_quarantine = max(
        (row["query_to_confirm_or_quarantine_rate"] for row in baseline_rows if row["baseline_id"] != "Q7"),
        default=0.0,
    )
    gate = {
        "Q7_valid_material_evidence_rate_ge_0_50": q7_metrics["valid_material_evidence_rate"] >= 0.50,
        "Q7_query_to_confirm_or_quarantine_rate_ge_best_fixed_plus_0_15": (
            q7_metrics["query_to_confirm_or_quarantine_rate"] >= best_fixed_confirm_or_quarantine + 0.15
        ),
        "Q7_state_entropy_reduction_ge_best_fixed_times_1_20": (
            q7_metrics["state_entropy_reduction"] >= best_fixed_entropy * 1.20
        ),
        "Q7_real_minus_shuffled_query_AUC_ge_0_15": (
            q7_metrics["real_minus_shuffled_query_AUC"] is not None
            and q7_metrics["real_minus_shuffled_query_AUC"] >= 0.15
        ),
        "Q7_real_minus_no_temporal_query_AUC_ge_0_10": (
            q7_metrics["real_minus_no_temporal_query_AUC"] is not None
            and q7_metrics["real_minus_no_temporal_query_AUC"] >= 0.10
        ),
    }
    gate["pass"] = bool(all(gate.values()))
    summary = {
        "phase": "v61_manifold_query",
        "created_at": utc_now(),
        "method_note": (
            "Phase4 is an equal-budget reranking over the existing v58 material-query evidence cache. "
            "Q7 selection uses v61 prediction-side manifold state, observation explanation, estimated information gain, "
            "and non-GT D4RT visibility/support proxy fields from the frozen query cache. "
            "It is not a newly executed D4RT tracker pass."
        ),
        "repair_note": (
            "After the initial manifold-only ranking underperformed, Phase4 applied the plan-directed high-visibility material-node "
            "repair by adding target-visible and inside-history support proxies. Diagnostic upper-bound fields below use post-query "
            "outcome columns only to show current-pool feasibility; they are not used by Q7 selection."
        ),
        "query_budget": cfg.query_budget,
        "query_count": q7_metrics["query_count"],
        "candidate_pool_count": pool_count,
        "valid_material_evidence_rate": q7_metrics["valid_material_evidence_rate"],
        "query_to_confirm_rate": q7_metrics["query_to_confirm_rate"],
        "query_to_quarantine_rate": q7_metrics["query_to_quarantine_rate"],
        "query_to_confirm_or_quarantine_rate": q7_metrics["query_to_confirm_or_quarantine_rate"],
        "state_entropy_before": q7_metrics["state_entropy_before"],
        "state_entropy_after": q7_metrics["state_entropy_after"],
        "state_entropy_reduction": q7_metrics["state_entropy_reduction"],
        "promotion_precision": q7_metrics["promotion_precision"],
        "quarantine_precision": q7_metrics["quarantine_precision"],
        "real_minus_shuffled_query_AUC": q7_metrics["real_minus_shuffled_query_AUC"],
        "real_minus_no_temporal_query_AUC": q7_metrics["real_minus_no_temporal_query_AUC"],
        "query_type_breakdown": q7_metrics["query_type_breakdown"],
        "best_fixed_query_entropy_reduction": best_fixed_entropy,
        "best_fixed_query_to_confirm_or_quarantine_rate": best_fixed_confirm_or_quarantine,
        "diagnostic_pool_upper_bound": upper_bound,
        "gate": gate,
        "baseline_rows": baseline_rows,
        "input_paths": {
            "refined_state_rows": _rel(cfg.refined_state_rows_path),
            "observation_explanation_rows": _rel(cfg.observation_explanation_rows_path),
            "v58_query_rows": _rel(query_root / "query_rows.csv"),
            "v58_query_metric_rows": _rel(query_root / "query_metric_rows.csv"),
            "v58_material_evidence_rows": _rel(query_root / "material_evidence_rows.csv"),
        },
        "output_paths": {
            "query_summary": _rel(Path(cfg.output_root) / "query_summary.json"),
            "query_rows": _rel(Path(cfg.output_root) / "query_rows.csv"),
            "material_evidence_rows": _rel(Path(cfg.output_root) / "material_evidence_rows.csv"),
            "query_metric_rows": _rel(Path(cfg.output_root) / "query_metric_rows.csv"),
        },
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }
    metric_rows = baseline_rows + [q7_metrics]
    return {
        "summary": summary,
        "query_rows": selected_rows,
        "material_evidence_rows": material_rows,
        "query_metric_rows": metric_rows,
    }


def write_v61_manifold_query(result: dict[str, Any], output_root: str | Path) -> dict[str, str]:
    root = _project(output_root)
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "query_summary": root / "query_summary.json",
        "query_rows": root / "query_rows.csv",
        "material_evidence_rows": root / "material_evidence_rows.csv",
        "query_metric_rows": root / "query_metric_rows.csv",
    }
    write_json(paths["query_summary"], result["summary"])
    write_csv(paths["query_rows"], result["query_rows"])
    write_csv(paths["material_evidence_rows"], result["material_evidence_rows"])
    write_csv(paths["query_metric_rows"], result["query_metric_rows"])
    return {name: _rel(path) for name, path in paths.items()}


def write_v61_query_visualizations(result: dict[str, Any], visualization_root: str | Path) -> dict[str, str]:
    root = _project(visualization_root)
    root.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        metric_rows = result["query_metric_rows"]
        labels = [row["baseline_id"] for row in metric_rows]
        reductions = [row["state_entropy_reduction"] for row in metric_rows]
        entropy_path = root / "query_before_after_state_entropy.png"
        fig, ax = plt.subplots(figsize=(8.8, 4.2))
        ax.bar(labels, reductions, color=["#7A8B99" if label != "Q7" else "#2A9D8F" for label in labels])
        ax.set_title("v61 query entropy reduction")
        ax.set_ylabel("mean reduction")
        fig.tight_layout()
        fig.savefig(entropy_path, dpi=160)
        plt.close(fig)

        confirm_rates = [row["query_to_confirm_or_quarantine_rate"] for row in metric_rows]
        control_path = root / "query_control_comparison.png"
        fig, ax = plt.subplots(figsize=(8.8, 4.2))
        ax.bar(labels, confirm_rates, color=["#7A8B99" if label != "Q7" else "#E76F51" for label in labels])
        ax.set_title("v61 query confirm-or-quarantine rate")
        ax.set_ylabel("rate")
        fig.tight_layout()
        fig.savefig(control_path, dpi=160)
        plt.close(fig)

        breakdown = result["summary"].get("query_type_breakdown") or {}
        type_path = root / "manifold_query_type_breakdown.png"
        fig, ax = plt.subplots(figsize=(7.2, 4.0))
        ax.bar(list(breakdown), list(breakdown.values()), color="#457B9D")
        ax.set_title("v61 query type breakdown")
        ax.tick_params(axis="x", labelrotation=20)
        fig.tight_layout()
        fig.savefig(type_path, dpi=160)
        plt.close(fig)

        return {
            "query_entropy_plot": _rel(entropy_path),
            "query_control_comparison": _rel(control_path),
            "query_type_breakdown": _rel(type_path),
            "visualization_status": "created",
        }
    except Exception as exc:  # pragma: no cover
        error_path = root / "v61_query_visualization_error.txt"
        error_path.write_text(f"visualization unavailable: {type(exc).__name__}: {exc}\n", encoding="utf-8")
        return {"visualization_error": _rel(error_path), "visualization_status": "unavailable"}


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


def _observation_stats_from_states(rows: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = defaultdict(lambda: {"state_counts": Counter(), "history_ids": set(), "material_count": 0})
    for row in rows:
        support_ids = _parse_json_list(row.get("support_observation_ids_json") or row.get("support_observation_ids"))
        for obs_id in support_ids:
            obs = _normalize_observation_id(obs_id)
            item = stats[obs]
            item["material_count"] += 1
            item["state_counts"][row.get("state") or "unknown"] += 1
            pred = row.get("predicted_history_id") or ""
            if pred:
                item["history_ids"].update(part for part in pred.split("||") if part)
            if parse_bool(row.get("has_K_underseg")):
                item["has_underseg_material"] = True
    return stats


def _observation_explanations(rows: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        obs_id = _normalize_observation_id(row.get("observation_node_id") or row.get("observation_id") or "")
        histories = _parse_json_list(row.get("candidate_history_ids_json"))
        out[obs_id] = {
            "explanation_type": row.get("explanation_type") or "",
            "candidate_history_count": len(histories),
            "support_material_count": parse_float(row.get("support_material_count"), 0.0),
            "confirmed_support_count": parse_float(row.get("confirmed_support_count"), 0.0),
            "shared_support_count": parse_float(row.get("shared_support_count"), 0.0),
            "quarantine_support_count": parse_float(row.get("quarantine_support_count"), 0.0),
        }
    return out


def _select_q7_rows(
    source_rows: list[dict[str, str]],
    obs_stats: dict[str, dict[str, Any]],
    obs_explanations: dict[str, dict[str, Any]],
    *,
    query_budget: int,
) -> tuple[list[dict[str, Any]], int]:
    best_by_obs: dict[str, tuple[float, dict[str, Any]]] = {}
    for row in source_rows:
        obs_id = _normalize_observation_id(row.get("observation_id") or "")
        stats = obs_stats.get(obs_id)
        explanation = obs_explanations.get(obs_id)
        if not stats and not explanation:
            continue
        query_type = _query_type(stats, explanation)
        score = _query_score(row, stats, explanation, query_type)
        out = {**row}
        out["source_baseline_id"] = row.get("baseline_id")
        out["source_query_id"] = row.get("query_id")
        out["baseline_id"] = "Q7"
        out["baseline_name"] = "v61 manifold boundary/gap/shortcut query"
        out["query_type"] = query_type
        out["v61_query_score"] = score
        out["v61_observation_id"] = obs_id
        out["v61_state_counts_json"] = dict((stats or {}).get("state_counts", {}))
        out["v61_history_count"] = len((stats or {}).get("history_ids", set()))
        out["v61_explanation_type"] = (explanation or {}).get("explanation_type", "")
        out["v61_candidate_history_count"] = (explanation or {}).get("candidate_history_count", 0)
        prev = best_by_obs.get(obs_id)
        if prev is None or score > prev[0]:
            best_by_obs[obs_id] = (score, out)
    pool = sorted(best_by_obs.values(), key=lambda item: item[0], reverse=True)
    selected = [row for _score, row in pool[: int(query_budget)]]
    for idx, row in enumerate(selected, start=1):
        row["query_rank"] = idx
        row["query_id"] = f"Q7_q{idx:04d}_{row.get('candidate_id', 'candidate')}"
    return selected, len(pool)


def _query_type(stats: dict[str, Any] | None, explanation: dict[str, Any] | None) -> str:
    counts = (stats or {}).get("state_counts", Counter())
    history_count = len((stats or {}).get("history_ids", set()))
    explanation_type = (explanation or {}).get("explanation_type", "")
    candidate_count = int((explanation or {}).get("candidate_history_count", 0))
    if counts.get("quarantine", 0) > 0 or explanation_type == "underseg" or candidate_count > 1:
        return "Q_shortcut"
    if counts.get("tentative", 0) > 0:
        return "Q_gap"
    if history_count > 1:
        return "Q_boundary"
    return "Q_boundary"


def _query_score(row: dict[str, str], stats: dict[str, Any] | None, explanation: dict[str, Any] | None, query_type: str) -> float:
    counts = (stats or {}).get("state_counts", Counter())
    candidate_count = float((explanation or {}).get("candidate_history_count", 0))
    material_count = float((stats or {}).get("material_count", 0))
    entropy = parse_float(row.get("entropy_before") or row.get("explanation_entropy_before"), 0.0)
    est_ig = parse_float(row.get("estimated_information_gain") or row.get("expected_information_gain_estimate"), 0.0)
    target_visible = parse_float(row.get("target_visible_rate"), 0.0)
    inside_support = parse_float(row.get("track_inside_history_rate"), 0.0)
    outside_residual = parse_float(row.get("track_outside_rate"), 0.0)
    shortcut_bonus = 0.45 if query_type == "Q_shortcut" else 0.0
    boundary_bonus = 0.18 * max(0.0, candidate_count - 1.0)
    gap_bonus = 0.30 if query_type == "Q_gap" else 0.0
    quarantine_bonus = 0.08 * math.log1p(float(counts.get("quarantine", 0)))
    tentative_bonus = 0.05 * math.log1p(float(counts.get("tentative", 0)))
    support_bonus = 0.01 * math.log1p(material_count)
    visibility_bonus = 0.80 * target_visible + 0.30 * inside_support - 0.10 * outside_residual
    return est_ig + 0.05 * entropy + shortcut_bonus + boundary_bonus + gap_bonus + quarantine_bonus + tentative_bonus + support_bonus + visibility_bonus


def _diagnostic_upper_bound(
    source_rows: list[dict[str, str]],
    obs_stats: dict[str, dict[str, Any]],
    obs_explanations: dict[str, dict[str, Any]],
    *,
    query_budget: int,
) -> dict[str, Any]:
    by_obs: dict[str, dict[str, Any]] = {}
    for row in source_rows:
        obs_id = _normalize_observation_id(row.get("observation_id") or "")
        if obs_id not in obs_stats and obs_id not in obs_explanations:
            continue
        item = by_obs.setdefault(obs_id, {"valid": False, "confirm_or_quarantine": False, "best_entropy_reduction": 0.0})
        item["valid"] = item["valid"] or parse_bool(row.get("valid_material_evidence"))
        item["confirm_or_quarantine"] = item["confirm_or_quarantine"] or parse_bool(row.get("query_to_confirm")) or parse_bool(row.get("query_to_quarantine"))
        item["best_entropy_reduction"] = max(item["best_entropy_reduction"], parse_float(row.get("actual_entropy_reduction"), 0.0))
    budget = max(int(query_budget), 1)
    valid_count = sum(1 for item in by_obs.values() if item["valid"])
    confirm_or_quarantine_count = sum(1 for item in by_obs.values() if item["confirm_or_quarantine"])
    entropy_values = sorted((float(item["best_entropy_reduction"]) for item in by_obs.values()), reverse=True)[:budget]
    return {
        "candidate_observation_count": len(by_obs),
        "valid_observation_count": valid_count,
        "confirm_or_quarantine_observation_count": confirm_or_quarantine_count,
        "valid_rate_at_budget_upper_bound": min(valid_count, budget) / float(budget),
        "confirm_or_quarantine_rate_at_budget_upper_bound": min(confirm_or_quarantine_count, budget) / float(budget),
        "entropy_reduction_at_budget_upper_bound": _mean_float(entropy_values),
        "uses_post_query_outcomes": True,
        "uses_gt_for_prediction": False,
    }


def _metrics_for_query_rows(baseline_id: str, baseline_name: str, rows: list[dict[str, Any]], budget: int) -> dict[str, Any]:
    valid = [parse_bool(row.get("valid_material_evidence")) for row in rows]
    confirm = [parse_bool(row.get("query_to_confirm")) for row in rows]
    quarantine = [parse_bool(row.get("query_to_quarantine")) for row in rows]
    successes = [parse_bool(row.get("diagnostic_query_success_same_gt")) for row in rows]
    real_scores = [parse_float(row.get("real_evidence_score"), 0.0) for row in rows]
    shuffled_scores = [parse_float(row.get("shuffled_evidence_score"), 0.0) for row in rows]
    no_temporal_scores = [parse_float(row.get("no_temporal_evidence_score"), 0.0) for row in rows]
    confirm_success = [success for success, is_confirm in zip(successes, confirm) if is_confirm]
    quarantine_success = [success for success, is_quarantine in zip(successes, quarantine) if is_quarantine]
    real_auc = rank_auc(successes, real_scores)
    shuffled_auc = rank_auc(successes, shuffled_scores)
    no_temporal_auc = rank_auc(successes, no_temporal_scores)
    return {
        "baseline_id": baseline_id,
        "baseline_name": baseline_name,
        "query_budget": budget,
        "query_count": len(rows),
        "valid_material_evidence_rate": _mean_bool(valid),
        "query_to_confirm_rate": _mean_bool(confirm),
        "query_to_quarantine_rate": _mean_bool(quarantine),
        "query_to_confirm_or_quarantine_rate": _mean_bool([left or right for left, right in zip(confirm, quarantine)]),
        "state_entropy_before": _mean_float(row.get("entropy_before") for row in rows),
        "state_entropy_after": _mean_float(row.get("entropy_after") for row in rows),
        "state_entropy_reduction": _mean_float(row.get("actual_entropy_reduction") for row in rows),
        "promotion_precision": _mean_bool(confirm_success) if confirm_success else None,
        "quarantine_precision": _mean_bool(quarantine_success) if quarantine_success else None,
        "real_query_AUC": real_auc,
        "shuffled_query_AUC": shuffled_auc,
        "no_temporal_query_AUC": no_temporal_auc,
        "real_minus_shuffled_query_AUC": None if real_auc is None or shuffled_auc is None else real_auc - shuffled_auc,
        "real_minus_no_temporal_query_AUC": None if real_auc is None or no_temporal_auc is None else real_auc - no_temporal_auc,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }


def _baseline_metric(row: dict[str, str]) -> dict[str, Any]:
    confirm = parse_float(row.get("query_to_confirm_rate"), 0.0)
    quarantine = parse_float(row.get("query_to_quarantine_rate"), 0.0)
    return {
        "baseline_id": row.get("baseline_id") or "",
        "baseline_name": row.get("baseline_name") or "",
        "query_budget": int(parse_float(row.get("query_budget"), 0.0)),
        "query_count": int(parse_float(row.get("query_count"), 0.0)),
        "valid_material_evidence_rate": parse_float(row.get("valid_material_evidence_rate"), 0.0),
        "query_to_confirm_rate": confirm,
        "query_to_quarantine_rate": quarantine,
        "query_to_confirm_or_quarantine_rate": min(1.0, confirm + quarantine),
        "state_entropy_before": parse_float(row.get("explanation_entropy_before"), 0.0),
        "state_entropy_after": parse_float(row.get("explanation_entropy_after"), 0.0),
        "state_entropy_reduction": parse_float(row.get("entropy_reduction"), 0.0),
        "promotion_precision": None,
        "quarantine_precision": None,
        "real_query_AUC": parse_float(row.get("real_query_AUC"), 0.0),
        "shuffled_query_AUC": parse_float(row.get("shuffled_query_AUC"), 0.0),
        "no_temporal_query_AUC": parse_float(row.get("no_temporal_query_AUC"), 0.0),
        "real_minus_shuffled_query_AUC": parse_float(row.get("real_minus_shuffled_query_AUC"), 0.0),
        "real_minus_no_temporal_query_AUC": parse_float(row.get("real_minus_no_temporal_query_AUC"), 0.0),
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }


def _normalize_observation_id(value: str) -> str:
    value = str(value or "")
    return value if value.startswith("m:") else f"m:{value}"


def _parse_json_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if not value:
        return []
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    return [str(item) for item in parsed] if isinstance(parsed, list) else []


def _mean_bool(values: list[bool]) -> float:
    return 0.0 if not values else float(sum(1 for value in values if value)) / float(len(values))


def _mean_float(values: Iterable[Any]) -> float:
    nums = [parse_float(value, 0.0) for value in values]
    return 0.0 if not nums else float(sum(nums) / len(nums))
