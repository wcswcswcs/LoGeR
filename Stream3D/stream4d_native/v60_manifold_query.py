from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .v47_common import ROOT, parse_bool, parse_float, rank_auc, read_json, utc_now, write_csv, write_json


DEFAULT_EMBEDDING_ROOT = "outputs/audit/v60_manifold_embedding"
DEFAULT_V58_QUERY_ROOT = "outputs/audit/v58_active_material_query_q128_repair5_expanded_all_minvis1"


@dataclass(frozen=True)
class V60QueryConfig:
    embedding_root: str | Path = DEFAULT_EMBEDDING_ROOT
    v58_query_root: str | Path = DEFAULT_V58_QUERY_ROOT
    output_root: str | Path = "outputs/audit/v60_manifold_query"
    visualization_root: str | Path = "outputs/audit/v60_visualizations/manifold_query"
    query_budget: int = 128


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


def build_v60_manifold_query(config: V60QueryConfig | None = None) -> dict[str, Any]:
    cfg = config or V60QueryConfig()
    embedding_root = _project(cfg.embedding_root)
    query_root = _project(cfg.v58_query_root)
    embedding_summary = read_json(embedding_root / "embedding_summary.json")
    states = {row["observation_id"]: row for row in _iter_csv(embedding_root / "node_state_rows.csv")}
    v58_summary = read_json(query_root / "query_summary.json")
    v58_metric_rows = list(_iter_csv(query_root / "query_metric_rows.csv"))
    v58_query_rows = list(_iter_csv(query_root / "query_rows.csv"))
    v58_material_rows = list(_iter_csv(query_root / "material_evidence_rows.csv"))

    selected_rows, candidate_pool_count = _select_q7_rows(v58_query_rows, states, cfg.query_budget)
    q7_metrics = _metrics_for_query_rows("Q7", "full manifold-aware query over v60 tentative/quarantine states", selected_rows, cfg.query_budget)
    q7_metrics["candidate_pool_count"] = candidate_pool_count
    baseline_rows = [_baseline_metric(row) for row in v58_metric_rows]
    best_entropy = max((row["entropy_reduction"] for row in baseline_rows), default=0.0)
    best_confirm_or_quarantine = max((row["query_to_confirm_or_quarantine_rate"] for row in baseline_rows), default=0.0)
    gate = {
        "Q7_entropy_reduction_ge_best_fixed_plus_20pct_relative": q7_metrics["state_entropy_reduction"] >= best_entropy * 1.20,
        "Q7_query_to_confirm_or_quarantine_rate_ge_best_fixed_plus_0_15": q7_metrics["query_to_confirm_or_quarantine_rate"] >= best_confirm_or_quarantine + 0.15,
        "Q7_valid_material_evidence_rate_ge_0_50": q7_metrics["valid_material_evidence_rate"] >= 0.50,
        "Q7_promotion_precision_ge_0_85": q7_metrics["promotion_precision"] is not None and q7_metrics["promotion_precision"] >= 0.85,
        "Q7_quarantine_precision_ge_0_80": q7_metrics["quarantine_precision"] is not None and q7_metrics["quarantine_precision"] >= 0.80,
        "Q7_real_minus_shuffled_query_AUC_ge_0_15": q7_metrics["real_minus_shuffled_query_AUC"] is not None and q7_metrics["real_minus_shuffled_query_AUC"] >= 0.15,
        "Q7_real_minus_no_temporal_query_AUC_ge_0_10": q7_metrics["real_minus_no_temporal_query_AUC"] is not None and q7_metrics["real_minus_no_temporal_query_AUC"] >= 0.10,
    }
    gate["pass"] = bool(all(gate.values()))
    selected_source_query_ids = {row["source_query_id"] for row in selected_rows}
    material_rows = [row for row in v58_material_rows if row.get("query_id") in selected_source_query_ids]
    summary = {
        "phase": "v60_manifold_query",
        "created_at": utc_now(),
        "diagnostic_only_bypass": True,
        "bypass_reason": "Phase3/Phase4 gates failed; Phase5 is run over existing material-query evidence to test repair feasibility only.",
        "query_budget": cfg.query_budget,
        "query_count": q7_metrics["query_count"],
        "candidate_pool_count": q7_metrics["candidate_pool_count"],
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
        "best_fixed_query_entropy_reduction": best_entropy,
        "best_fixed_query_to_confirm_or_quarantine_rate": best_confirm_or_quarantine,
        "gate": gate,
        "method_note": (
            "Q7 is a diagnostic v60 reranking over v58 selected query evidence, restricted to v60 tentative/quarantine states. "
            "It is not a newly executed material tracker run and cannot rescue the failed Phase3 embedding gate."
        ),
        "baseline_rows": baseline_rows,
        "embedding_gate_pass": bool((embedding_summary.get("gate") or {}).get("pass")),
        "input_paths": {
            "embedding_summary": _rel(embedding_root / "embedding_summary.json"),
            "node_state_rows": _rel(embedding_root / "node_state_rows.csv"),
            "v58_query_summary": _rel(query_root / "query_summary.json"),
            "v58_query_rows": _rel(query_root / "query_rows.csv"),
            "v58_query_metric_rows": _rel(query_root / "query_metric_rows.csv"),
            "v58_material_evidence_rows": _rel(query_root / "material_evidence_rows.csv"),
        },
        "v58_reference": {
            "Q6_entropy_reduction": v58_summary.get("Q6_entropy_reduction"),
            "Q6_valid_material_evidence_rate": v58_summary.get("Q6_valid_material_evidence_rate"),
            "Q6_query_to_confirm_rate": v58_summary.get("Q6_query_to_confirm_rate"),
            "Q6_real_minus_shuffled_query_AUC": v58_summary.get("Q6_real_minus_shuffled_query_AUC"),
            "Q6_real_minus_no_temporal_query_AUC": v58_summary.get("Q6_real_minus_no_temporal_query_AUC"),
            "gate_pass": (v58_summary.get("gate") or {}).get("pass"),
        },
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }
    return {"summary": summary, "query_rows": selected_rows, "material_evidence_rows": material_rows}


def write_v60_manifold_query(result: dict[str, Any], output_root: str | Path) -> dict[str, str]:
    root = _project(output_root)
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "query_summary": root / "query_summary.json",
        "query_rows": root / "query_rows.csv",
        "material_evidence_rows": root / "material_evidence_rows.csv",
    }
    write_json(paths["query_summary"], result["summary"])
    write_csv(paths["query_rows"], result["query_rows"])
    write_csv(paths["material_evidence_rows"], result["material_evidence_rows"])
    return {name: _rel(path) for name, path in paths.items()}


def write_v60_query_visualizations(result: dict[str, Any], visualization_root: str | Path) -> dict[str, str]:
    root = _project(visualization_root)
    root.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        path = root / "query_state_entropy_plot.png"
        summary = result["summary"]
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        ax.bar(["before", "after"], [summary["state_entropy_before"], summary["state_entropy_after"]], color=["#457B9D", "#E9C46A"])
        ax.set_title("v60 Q7 diagnostic state entropy")
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        return {"query_state_entropy_plot": _rel(path), "visualization_status": "created"}
    except Exception as exc:  # pragma: no cover
        error_path = root / "v60_query_visualization_error.txt"
        error_path.write_text(f"visualization unavailable: {type(exc).__name__}: {exc}\n", encoding="utf-8")
        return {"visualization_error": _rel(error_path), "visualization_status": "unavailable"}


def _select_q7_rows(query_rows: list[dict[str, str]], states: dict[str, dict[str, str]], budget: int) -> tuple[list[dict[str, Any]], int]:
    pool: list[tuple[float, dict[str, Any]]] = []
    seen: set[str] = set()
    for row in query_rows:
        state = states.get(row.get("observation_id", ""))
        if not state or state.get("state") not in {"tentative", "quarantine"}:
            continue
        observation_id = row["observation_id"]
        if observation_id in seen:
            continue
        seen.add(observation_id)
        state_bonus = 0.20 if state["state"] == "tentative" else 0.15
        score = parse_float(row.get("estimated_information_gain"), 0.0) + state_bonus + 0.10 * parse_float(state.get("posterior_top1_margin"), 0.0)
        out = {**row}
        out["source_query_id"] = row.get("query_id")
        out["baseline_id"] = "Q7"
        out["baseline_name"] = "full manifold-aware query over v60 tentative/quarantine states"
        out["v60_state"] = state["state"]
        out["v60_state_reason"] = state.get("state_reason")
        out["v60_query_score"] = score
        pool.append((score, out))
    pool.sort(key=lambda item: item[0], reverse=True)
    selected = [row for _score, row in pool[:budget]]
    for idx, row in enumerate(selected, start=1):
        row["query_rank"] = idx
        row["query_id"] = f"Q7_q{idx:04d}_{row.get('candidate_id', 'candidate')}"
    return selected, len(pool)


def _metrics_for_query_rows(baseline_id: str, baseline_name: str, rows: list[dict[str, Any]], budget: int) -> dict[str, Any]:
    query_count = len(rows)
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
        "query_count": query_count,
        "candidate_pool_count": query_count,
        "valid_material_evidence_rate": _mean_bool(valid),
        "query_to_confirm_rate": _mean_bool(confirm),
        "query_to_quarantine_rate": _mean_bool(quarantine),
        "query_to_confirm_or_quarantine_rate": _mean_bool([c or q for c, q in zip(confirm, quarantine)]),
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
    }


def _baseline_metric(row: dict[str, str]) -> dict[str, Any]:
    confirm = parse_float(row.get("query_to_confirm_rate"), 0.0)
    quarantine = parse_float(row.get("query_to_quarantine_rate"), 0.0)
    return {
        "baseline_id": row.get("baseline_id"),
        "baseline_name": row.get("baseline_name"),
        "entropy_reduction": parse_float(row.get("entropy_reduction"), 0.0),
        "query_to_confirm_rate": confirm,
        "query_to_quarantine_rate": quarantine,
        "query_to_confirm_or_quarantine_rate": min(1.0, confirm + quarantine),
        "valid_material_evidence_rate": parse_float(row.get("valid_material_evidence_rate"), 0.0),
        "real_minus_shuffled_query_AUC": parse_float(row.get("real_minus_shuffled_query_AUC"), 0.0),
        "real_minus_no_temporal_query_AUC": parse_float(row.get("real_minus_no_temporal_query_AUC"), 0.0),
    }


def _mean_bool(values: list[bool]) -> float:
    return 0.0 if not values else float(sum(1 for value in values if value)) / float(len(values))


def _mean_float(values: Iterable[Any]) -> float:
    nums = [parse_float(value, 0.0) for value in values]
    return 0.0 if not nums else float(sum(nums) / len(nums))
