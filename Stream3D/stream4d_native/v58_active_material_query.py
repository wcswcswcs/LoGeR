from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .v47_common import (
    ROOT,
    parse_bool,
    parse_float,
    parse_int,
    read_json,
    safe_mean,
    utc_now,
    write_csv,
    write_json,
)


DEFAULT_PHASE2_ROOT = "outputs/audit/v58_counterfactual_explanation_dino_full_repair6"
DEFAULT_REPROJECTION_ROOT = "outputs/audit/v54_reprojection_ledger_k0all_conflict_veto018_skip_repeated_sig_stride1_probe5_q4096_notopup_max4000_skip"


@dataclass(frozen=True)
class V58ActiveMaterialQueryConfig:
    phase2_root: str | Path = DEFAULT_PHASE2_ROOT
    reprojection_candidate_rows_path: str | Path = Path(DEFAULT_REPROJECTION_ROOT) / "candidate_rows.csv"
    reprojection_ledger_rows_path: str | Path = Path(DEFAULT_REPROJECTION_ROOT) / "reprojection_ledger_rows.csv"
    output_root: str | Path = "outputs/audit/v58_active_material_query"
    visualization_root: str | Path = "outputs/audit/v58_visualizations/active_query"
    primary_variant: str = "E6_counterfactual_semantic_material_underseg"
    query_budget: int = 128
    max_target_frames: int = 5
    random_seed: int = 58
    max_visual_queries: int = 6


def build_v58_active_material_query(config: V58ActiveMaterialQueryConfig | None = None) -> dict[str, Any]:
    cfg = config or V58ActiveMaterialQueryConfig()
    phase2_root = _project(cfg.phase2_root)
    phase2_summary = read_json(phase2_root / "explanation_summary.json")
    selected_rows = _load_phase2_selected_rows(phase2_root / "explanation_rows.csv", cfg.primary_variant)
    deferred_rows = [row for row in selected_rows if str(row.get("decision_state") or "") == "defer_to_active_query"]
    candidate_rows = _read_csv(_project(cfg.reprojection_candidate_rows_path))
    ledger_rows = _read_csv(_project(cfg.reprojection_ledger_rows_path))
    evidence_by_candidate = _aggregate_candidate_evidence(candidate_rows, ledger_rows)
    candidates_by_observation = _candidate_options_by_observation(candidate_rows, evidence_by_candidate)
    eligible_rows = [
        row for row in deferred_rows if str(row.get("observation_id") or "") in candidates_by_observation
    ]
    query_count = _query_count(cfg.query_budget, len(eligible_rows))

    baseline_specs = _baseline_specs()
    query_rows: list[dict[str, Any]] = []
    material_evidence_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for spec in baseline_specs:
        selected_queries = _select_queries(
            spec,
            eligible_rows,
            candidates_by_observation,
            query_count=query_count,
            max_target_frames=cfg.max_target_frames,
            random_seed=cfg.random_seed,
        )
        query_rows.extend(selected_queries)
        material_evidence_rows.extend(
            _material_rows_for_queries(selected_queries, ledger_rows, max_target_frames=cfg.max_target_frames)
        )
        metric_rows.append(_metric_row(spec, selected_queries))

    metrics_by_baseline = {row["baseline_id"]: row for row in metric_rows}
    best_fixed = max(
        (
            _num(row.get("entropy_reduction"), 0.0)
            for row in metric_rows
            if str(row.get("baseline_id") or "") in {"Q0", "Q1", "Q2", "Q3", "Q4", "Q5"}
        ),
        default=0.0,
    )
    q0 = metrics_by_baseline.get("Q0", {})
    q6 = metrics_by_baseline.get("Q6", {})
    q6_entropy = _num(q6.get("entropy_reduction"), 0.0)
    q6_valid = _num(q6.get("valid_material_evidence_rate"), 0.0)
    q6_confirm = _num(q6.get("query_to_confirm_rate"), 0.0)
    q0_confirm = _num(q0.get("query_to_confirm_rate"), 0.0)
    q6_real_minus_shuffled = _num(q6.get("real_minus_shuffled_query_AUC"), -999.0)
    q6_real_minus_no_temporal = _num(q6.get("real_minus_no_temporal_query_AUC"), -999.0)
    entropy_required = best_fixed + 0.20 * abs(best_fixed)
    posthoc_upper = _posthoc_actual_reduction_upper_bound(eligible_rows, candidates_by_observation, query_count)
    gate = {
        "Q6_entropy_reduction_ge_best_fixed_plus_20pct_relative": q6_entropy >= entropy_required,
        "Q6_valid_material_evidence_rate_ge_0_50": q6_valid >= 0.50,
        "Q6_query_to_confirm_rate_ge_Q0_plus_0_15": q6_confirm >= q0_confirm + 0.15,
        "Q6_real_minus_shuffled_query_AUC_ge_0_15": q6_real_minus_shuffled >= 0.15,
        "Q6_real_minus_no_temporal_query_AUC_ge_0_10": q6_real_minus_no_temporal >= 0.10,
    }
    gate["pass"] = bool(all(gate.values()))
    summary = {
        "phase": "v58_active_material_query",
        "created_at": utc_now(),
        "method_note": (
            "Phase3 uses the frozen v54 D4RT reprojection ledger as the material-evidence cache. "
            "Query selection uses Phase2 posterior and non-GT material ledger fields only; diagnostic GT fields are used only for AUC evaluation."
        ),
        "primary_variant": cfg.primary_variant,
        "phase2_root": _rel(phase2_root),
        "phase2_gate_pass": bool((phase2_summary.get("gate") or {}).get("pass")),
        "phase2_deferred_count": phase2_summary.get("deferred_count"),
        "phase2_actionable_count": phase2_summary.get("actionable_count"),
        "deferred_observation_count_loaded": int(len(deferred_rows)),
        "eligible_deferred_observation_count": int(len(eligible_rows)),
        "candidate_row_count": int(len(candidate_rows)),
        "candidate_unique_observation_count": int(len(candidates_by_observation)),
        "ledger_row_count": int(len(ledger_rows)),
        "query_budget": int(cfg.query_budget),
        "query_count": int(query_count),
        "max_target_frames": int(cfg.max_target_frames),
        "baselines": [spec["baseline_id"] for spec in baseline_specs],
        "best_fixed_query_entropy_reduction": best_fixed,
        "Q6_entropy_reduction_required_for_gate": entropy_required,
        "Q0_query_to_confirm_rate": q0.get("query_to_confirm_rate"),
        "Q6_entropy_reduction": q6.get("entropy_reduction"),
        "Q6_valid_material_evidence_rate": q6.get("valid_material_evidence_rate"),
        "Q6_query_to_confirm_rate": q6.get("query_to_confirm_rate"),
        "Q6_real_minus_shuffled_query_AUC": q6.get("real_minus_shuffled_query_AUC"),
        "Q6_real_minus_no_temporal_query_AUC": q6.get("real_minus_no_temporal_query_AUC"),
        "posthoc_actual_entropy_reduction_upper_bound": posthoc_upper.get("entropy_reduction"),
        "posthoc_actual_confirm_rate_upper_bound": posthoc_upper.get("query_to_confirm_rate"),
        "posthoc_upper_bound_note": (
            "Diagnostic only: uses current non-GT material-evidence proxy after ledger aggregation to estimate the best possible "
            "entropy reduction from the eligible pool. It is not used for query selection."
        ),
        "entropy_gate_feasible_under_current_proxy": (
            posthoc_upper.get("entropy_reduction") is not None
            and _num(posthoc_upper.get("entropy_reduction")) >= entropy_required
        ),
        "gate": gate,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
        "diagnostic_label_sources": [
            "v54_reprojection_ledger.diagnostic_success_same_gt for query AUC only",
            "v54_reprojection_ledger.diagnostic_source_gt/best_gt for audit only",
        ],
        "input_paths": {
            "phase2_summary": _rel(phase2_root / "explanation_summary.json"),
            "phase2_rows": _rel(phase2_root / "explanation_rows.csv"),
            "reprojection_candidate_rows": _rel(cfg.reprojection_candidate_rows_path),
            "reprojection_ledger_rows": _rel(cfg.reprojection_ledger_rows_path),
        },
        "output_paths": {
            "query_summary": _rel(Path(cfg.output_root) / "query_summary.json"),
            "query_rows": _rel(Path(cfg.output_root) / "query_rows.csv"),
            "material_evidence_rows": _rel(Path(cfg.output_root) / "material_evidence_rows.csv"),
            "query_metric_rows": _rel(Path(cfg.output_root) / "query_metric_rows.csv"),
        },
    }
    return {
        "summary": summary,
        "query_rows": query_rows,
        "material_evidence_rows": material_evidence_rows,
        "query_metric_rows": metric_rows,
    }


def write_v58_active_material_query(result: dict[str, Any], output_root: str | Path) -> dict[str, str]:
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


def write_v58_active_material_query_visualizations(
    result: dict[str, Any],
    visualization_root: str | Path,
    *,
    max_visual_queries: int = 6,
) -> list[str]:
    root = _project(visualization_root)
    root.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        metrics = result["query_metric_rows"]
        labels = [str(row.get("baseline_id")) for row in metrics]
        reductions = [_num(row.get("entropy_reduction"), 0.0) for row in metrics]
        fig, ax = plt.subplots(figsize=(8.6, 4.4))
        ax.bar(labels, reductions, color=["#4C6A92", "#5B8C5A", "#B88B4A", "#7B6D8D", "#B45B5B", "#4F8B8B", "#2F5D62"])
        ax.set_title("v58 Phase3 query entropy reduction")
        ax.set_ylabel("mean entropy reduction")
        for idx, value in enumerate(reductions):
            ax.text(idx, value + max(reductions + [1.0]) * 0.02, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
        path = root / "query_entropy_reduction_plot.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(_rel(path))

        real = [_num(row.get("real_query_AUC"), 0.0) for row in metrics]
        shuffled = [_num(row.get("shuffled_query_AUC"), 0.0) for row in metrics]
        no_temporal = [_num(row.get("no_temporal_query_AUC"), 0.0) for row in metrics]
        x = np.arange(len(labels))
        width = 0.25
        fig, ax = plt.subplots(figsize=(9.2, 4.4))
        ax.bar(x - width, real, width, label="real", color="#4C6A92")
        ax.bar(x, shuffled, width, label="shuffled", color="#B88B4A")
        ax.bar(x + width, no_temporal, width, label="no temporal", color="#8B5E83")
        ax.set_title("v58 Phase3 material evidence controls")
        ax.set_ylabel("diagnostic AUC")
        ax.set_xticks(x, labels)
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = root / "query_control_comparison.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(_rel(path))

        q6_rows = [row for row in result["query_rows"] if row.get("baseline_id") == "Q6"][: int(max_visual_queries)]
        for row in q6_rows:
            values = [
                _num(row.get("entropy_before"), 0.0),
                _num(row.get("estimated_information_gain"), 0.0),
                _num(row.get("actual_entropy_reduction"), 0.0),
                _num(row.get("track_inside_history_rate"), 0.0),
                _num(row.get("track_competing_history_rate"), 0.0),
                _num(row.get("track_outside_rate"), 0.0),
            ]
            labels2 = ["entropy", "est IG", "actual dH", "inside", "compete", "outside"]
            fig, ax = plt.subplots(figsize=(8.0, 3.8))
            ax.bar(labels2, values, color=["#4C6A92", "#5B8C5A", "#2F5D62", "#7EA16B", "#B88B4A", "#B45B5B"])
            ax.set_title(f"active query {row.get('scene')} {row.get('observation_id')}")
            ax.tick_params(axis="x", labelrotation=20)
            fig.tight_layout()
            safe_obs = str(row.get("observation_id") or "").replace(":", "_").replace("/", "_")
            path = root / f"active_query_map_{row.get('scene')}_{safe_obs}.png"
            fig.savefig(path, dpi=160)
            plt.close(fig)
            paths.append(_rel(path))
    except Exception as exc:  # pragma: no cover
        fallback = root / "active_query_visualization_failed.txt"
        fallback.write_text(f"visualization_failed: {type(exc).__name__}: {exc}\n", encoding="utf-8")
        paths.append(_rel(fallback))
    return paths


def _load_phase2_selected_rows(path: Path, primary_variant: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("variant") or "") != str(primary_variant):
                continue
            if not parse_bool(row.get("is_selected")):
                continue
            rows.append(row)
    return rows


def _aggregate_candidate_evidence(
    candidate_rows: list[dict[str, str]],
    ledger_rows: list[dict[str, str]],
) -> dict[str, dict[str, Any]]:
    candidate_meta = {str(row.get("candidate_id") or ""): row for row in candidate_rows}
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in ledger_rows:
        grouped[str(row.get("candidate_id") or "")].append(row)
    out: dict[str, dict[str, Any]] = {}
    for candidate_id, meta in candidate_meta.items():
        rows = grouped.get(candidate_id, [])
        success = [1.0 if parse_bool(row.get("reprojection_success")) else 0.0 for row in rows]
        visible = [1.0 if parse_int(row.get("visible_carrier_count")) > 0 else 0.0 for row in rows]
        inside = [parse_float(row.get("inside_best_mask_ratio")) for row in rows]
        inside_any = [parse_float(row.get("inside_any_mask_ratio")) for row in rows]
        outside = [parse_float(row.get("outside_all_related_masks_ratio")) for row in rows]
        explained = [parse_float(row.get("mask_explained_ratio")) for row in rows]
        conflict = [1.0 if parse_bool(row.get("same_frame_exclusion_violation")) else 0.0 for row in rows]
        related = [parse_float(row.get("related_mask_count")) for row in rows]
        diag = [1.0 if parse_bool(row.get("diagnostic_success_same_gt")) else 0.0 for row in rows]
        source_frame = parse_int(meta.get("source_frame_id"))
        no_temporal_rows = [row for row in rows if parse_int(row.get("target_frame_id")) == source_frame]
        no_temporal_scores = [_ledger_evidence_score(row) for row in no_temporal_rows]
        real_score = _clip(
            0.40 * _mean(success)
            + 0.30 * _mean(inside)
            + 0.15 * (1.0 - _mean(outside))
            + 0.10 * _mean(explained)
            + 0.05 * (1.0 - _mean(conflict)),
            0.0,
            1.0,
        )
        competing_rate = _clip(max(_mean(conflict), min(_mean(related) / 6.0, 1.0) * 0.35), 0.0, 1.0)
        confirm_score = _clip(_mean(success) * _mean(inside) * (1.0 - _mean(outside)) * (1.0 - _mean(conflict)), 0.0, 1.0)
        quarantine_score = _clip(_mean(success) * max(competing_rate, _mean(outside)), 0.0, 1.0)
        out[candidate_id] = {
            "candidate_id": candidate_id,
            "candidate_source": meta.get("candidate_source", ""),
            "scene": meta.get("scene", ""),
            "chunk_id": meta.get("chunk_id", ""),
            "source_mask_observation_id": meta.get("source_mask_observation_id", ""),
            "source_frame_id": source_frame,
            "source_mask_id": parse_int(meta.get("source_mask_id")),
            "component_count": parse_int(meta.get("component_count")),
            "ledger_row_count": len(rows),
            "candidate_success_rate": _mean(success),
            "target_visible_rate": _mean(visible),
            "inside_best_mask_ratio_mean": _mean(inside),
            "inside_any_mask_ratio_mean": _mean(inside_any),
            "track_outside_rate": _mean(outside),
            "mask_explained_ratio_mean": _mean(explained),
            "same_frame_exclusion_violation_rate": _mean(conflict),
            "related_mask_count_mean": _mean(related),
            "track_competing_history_rate": competing_rate,
            "diagnostic_success_rate": _mean(diag),
            "real_evidence_score": real_score,
            "no_temporal_evidence_score": _mean(no_temporal_scores),
            "confirm_score": confirm_score,
            "quarantine_score": quarantine_score,
            "valid_material_evidence": bool(_mean(success) >= 0.50 and _mean(visible) >= 0.50 and len(rows) > 0),
            "query_to_confirm": bool(confirm_score >= 0.72 and _mean(outside) <= 0.08 and competing_rate <= 0.45),
            "query_to_quarantine": bool(quarantine_score >= 0.18 and not (confirm_score >= 0.72 and _mean(outside) <= 0.08 and competing_rate <= 0.45)),
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic_labels": True,
        }
    return out


def _candidate_options_by_observation(
    candidate_rows: list[dict[str, str]],
    evidence_by_candidate: dict[str, dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        candidate_id = str(row.get("candidate_id") or "")
        evidence = evidence_by_candidate.get(candidate_id)
        if not evidence:
            continue
        observation_id = str(row.get("source_mask_observation_id") or "")
        if not observation_id:
            continue
        out[observation_id].append(evidence)
    for observation_id, options in out.items():
        options.sort(key=lambda item: (-_num(item.get("real_evidence_score")), str(item.get("candidate_id"))))
    return out


def _baseline_specs() -> list[dict[str, str]]:
    return [
        {"baseline_id": "Q0", "baseline_name": "random query", "source_region_type": "random_control"},
        {"baseline_id": "Q1", "baseline_name": "mask interior query", "source_region_type": "mask_interior"},
        {"baseline_id": "Q2", "baseline_name": "mask boundary query", "source_region_type": "mask_boundary"},
        {"baseline_id": "Q3", "baseline_name": "high semantic entropy query without material competition", "source_region_type": "semantic_entropy"},
        {"baseline_id": "Q4", "baseline_name": "v57 tentative bridgelet heuristic", "source_region_type": "tentative_bridgelet"},
        {"baseline_id": "Q5", "baseline_name": "material uncertainty query without semantics", "source_region_type": "material_uncertainty"},
        {"baseline_id": "Q6", "baseline_name": "semantic-conditioned information-gain query", "source_region_type": "semantic_conditioned_ig"},
    ]


def _select_queries(
    spec: dict[str, str],
    eligible_rows: list[dict[str, str]],
    candidates_by_observation: dict[str, list[dict[str, Any]]],
    *,
    query_count: int,
    max_target_frames: int,
    random_seed: int,
) -> list[dict[str, Any]]:
    scored: list[tuple[float, str, dict[str, Any]]] = []
    for row in eligible_rows:
        observation_id = str(row.get("observation_id") or "")
        options = candidates_by_observation.get(observation_id) or []
        if not options:
            continue
        candidate = _choose_candidate_for_baseline(spec["baseline_id"], row, options, random_seed)
        query = _build_query_row(spec, row, candidate, max_target_frames=max_target_frames, random_seed=random_seed)
        score = _selection_score(spec["baseline_id"], row, candidate, random_seed)
        scored.append((score, observation_id, query))
    scored.sort(key=lambda item: (-item[0], item[1], str(item[2].get("candidate_id"))))
    selected = [item[2] for item in scored[: int(query_count)]]
    for idx, row in enumerate(selected):
        row["query_rank"] = idx + 1
        row["query_id"] = f"{row['baseline_id']}_q{idx + 1:04d}_{row['candidate_id']}"
    return selected


def _choose_candidate_for_baseline(
    baseline_id: str,
    phase2_row: dict[str, str],
    options: list[dict[str, Any]],
    random_seed: int,
) -> dict[str, Any]:
    if baseline_id == "Q0":
        return sorted(options, key=lambda item: _stable_unit(f"{random_seed}|{phase2_row.get('observation_id')}|{item.get('candidate_id')}"))[0]
    if baseline_id == "Q3":
        return sorted(
            options,
            key=lambda item: (
                0 if str(item.get("candidate_source") or "") == "R0_single_representative_mask" else 1,
                str(item.get("candidate_id")),
            ),
        )[0]
    key_fn = {
        "Q1": lambda item: _num(item.get("inside_best_mask_ratio_mean")) - _num(item.get("track_outside_rate")) - _num(item.get("track_competing_history_rate")) * 0.25,
        "Q2": lambda item: _num(item.get("track_competing_history_rate")) + _num(item.get("track_outside_rate")) + 0.10 * _num(item.get("component_count")),
        "Q4": lambda item: _num(item.get("candidate_success_rate")) + 0.02 * _num(item.get("ledger_row_count")) + 0.05 * _num(item.get("component_count")),
        "Q5": lambda item: _num(item.get("track_competing_history_rate")) + _num(item.get("same_frame_exclusion_violation_rate")) + _num(item.get("track_outside_rate")),
        "Q6": lambda item: _estimated_information_gain(phase2_row, item)
        + 0.30 * _num(item.get("confirm_score"))
        - 0.15 * _num(item.get("quarantine_score")),
    }.get(baseline_id, lambda item: _num(item.get("real_evidence_score")))
    return sorted(options, key=lambda item: (-key_fn(item), str(item.get("candidate_id"))))[0]


def _build_query_row(
    spec: dict[str, str],
    phase2_row: dict[str, str],
    candidate: dict[str, Any],
    *,
    max_target_frames: int,
    random_seed: int,
) -> dict[str, Any]:
    entropy_before = _num(phase2_row.get("posterior_entropy"))
    estimated_ig = _estimated_information_gain(phase2_row, candidate)
    actual_reduction = _actual_entropy_reduction(entropy_before, candidate)
    entropy_after = max(0.0, entropy_before - actual_reduction)
    candidate_histories = _load_json_list(phase2_row.get("candidate_history_ids_json"))
    return {
        "baseline_id": spec["baseline_id"],
        "baseline_name": spec["baseline_name"],
        "query_id": "",
        "query_rank": "",
        "observation_id": phase2_row.get("observation_id", ""),
        "scene": phase2_row.get("scene", ""),
        "source_frame": parse_int(phase2_row.get("frame_id")),
        "source_mask_id": parse_int(phase2_row.get("mask_id")),
        "source_point_uv": _pseudo_source_uv(phase2_row.get("observation_id", ""), random_seed),
        "source_region_type": spec["source_region_type"],
        "candidate_id": candidate.get("candidate_id"),
        "candidate_source": candidate.get("candidate_source"),
        "candidate_explanations_to_disambiguate": json.dumps(
            {
                "selected_explanation_type": phase2_row.get("explanation_type", ""),
                "selected_history_id": phase2_row.get("history_id", ""),
                "candidate_history_ids": candidate_histories,
            },
            sort_keys=True,
        ),
        "expected_information_gain_estimate": estimated_ig,
        "estimated_information_gain": estimated_ig,
        "valid_material_evidence": bool(candidate.get("valid_material_evidence")),
        "target_visible_rate": candidate.get("target_visible_rate"),
        "track_inside_history_rate": candidate.get("inside_best_mask_ratio_mean"),
        "track_competing_history_rate": candidate.get("track_competing_history_rate"),
        "track_outside_rate": candidate.get("track_outside_rate"),
        "explanation_entropy_before": entropy_before,
        "explanation_entropy_after": entropy_after,
        "entropy_before": entropy_before,
        "entropy_after": entropy_after,
        "actual_entropy_reduction": actual_reduction,
        "query_to_confirm": bool(candidate.get("query_to_confirm")),
        "query_to_quarantine": bool(candidate.get("query_to_quarantine")),
        "promotion_candidate_gain": 1.0 if candidate.get("query_to_confirm") and actual_reduction > 0.05 else 0.0,
        "real_evidence_score": candidate.get("real_evidence_score"),
        "shuffled_evidence_score": "",
        "no_temporal_evidence_score": candidate.get("no_temporal_evidence_score"),
        "diagnostic_query_success_same_gt": _num(candidate.get("diagnostic_success_rate"), 0.0) >= 0.50,
        "diagnostic_success_rate": candidate.get("diagnostic_success_rate"),
        "max_target_frames": int(max_target_frames),
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }


def _selection_score(
    baseline_id: str,
    phase2_row: dict[str, str],
    candidate: dict[str, Any],
    random_seed: int,
) -> float:
    entropy = _num(phase2_row.get("posterior_entropy"))
    margin = _num(phase2_row.get("posterior_top1_margin"))
    material_comp = _num(phase2_row.get("material_competition"))
    component_entropy = _num(phase2_row.get("component_entropy"))
    if baseline_id == "Q0":
        return _stable_unit(f"{random_seed}|select|{phase2_row.get('observation_id')}|{candidate.get('candidate_id')}")
    if baseline_id == "Q1":
        return _num(candidate.get("inside_best_mask_ratio_mean")) - _num(candidate.get("track_outside_rate"))
    if baseline_id == "Q2":
        return _num(candidate.get("track_competing_history_rate")) + _num(candidate.get("track_outside_rate")) + (1.0 - margin)
    if baseline_id == "Q3":
        return entropy
    if baseline_id == "Q4":
        return _num(candidate.get("candidate_success_rate")) + 0.03 * _num(candidate.get("ledger_row_count")) + 0.05 * _num(candidate.get("component_count"))
    if baseline_id == "Q5":
        return (
            _num(candidate.get("track_competing_history_rate"))
            + _num(candidate.get("same_frame_exclusion_violation_rate"))
            + _num(candidate.get("track_outside_rate"))
            + 0.2 * component_entropy
        )
    if baseline_id == "Q6":
        return (
            _estimated_information_gain(phase2_row, candidate)
            + 0.35 * entropy * _num(candidate.get("confirm_score"))
            + 0.10 * entropy * (1.0 - _num(candidate.get("track_outside_rate")))
            + 0.05 * material_comp
        )
    return _num(candidate.get("real_evidence_score"))


def _estimated_information_gain(phase2_row: dict[str, str], candidate: dict[str, Any]) -> float:
    entropy = _num(phase2_row.get("posterior_entropy"))
    margin = _num(phase2_row.get("posterior_top1_margin"))
    material_comp = _num(phase2_row.get("material_competition"))
    component_entropy = _num(phase2_row.get("component_entropy"))
    semantic_advantage = abs(_num(phase2_row.get("semantic_advantage")))
    evidence = _num(candidate.get("real_evidence_score"))
    confirm = _num(candidate.get("confirm_score"))
    competition = max(material_comp, _num(candidate.get("track_competing_history_rate")))
    ambiguity = _clip(1.0 - margin, 0.0, 1.0)
    boundary_or_competition = max(competition, _num(candidate.get("track_outside_rate")))
    expected_resolution = _clip(
        0.45 * evidence
        + 0.25 * confirm
        + 0.12 * ambiguity
        + 0.10 * boundary_or_competition
        + 0.05 * component_entropy
        + 0.03 * (1.0 - min(semantic_advantage, 1.0)),
        0.0,
        0.95,
    )
    return float(entropy * expected_resolution)


def _actual_entropy_reduction(entropy_before: float, candidate: dict[str, Any]) -> float:
    valid = 1.0 if candidate.get("valid_material_evidence") else 0.0
    confirm = _num(candidate.get("confirm_score"))
    quarantine = _num(candidate.get("quarantine_score"))
    evidence = _num(candidate.get("real_evidence_score"))
    resolution = _clip(valid * (0.45 * evidence + 0.35 * max(confirm, quarantine) + 0.20 * _num(candidate.get("target_visible_rate"))), 0.0, 0.85)
    return float(max(0.0, entropy_before) * resolution)


def _material_rows_for_queries(
    query_rows: list[dict[str, Any]],
    ledger_rows: list[dict[str, str]],
    *,
    max_target_frames: int,
) -> list[dict[str, Any]]:
    query_by_candidate: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in query_rows:
        query_by_candidate[str(row.get("candidate_id") or "")].append(row)
    out: list[dict[str, Any]] = []
    emitted_counts: dict[tuple[str, str], int] = defaultdict(int)
    for ledger in ledger_rows:
        candidate_id = str(ledger.get("candidate_id") or "")
        for query in query_by_candidate.get(candidate_id, []):
            key = (str(query.get("query_id")), candidate_id)
            if emitted_counts[key] >= int(max_target_frames):
                continue
            emitted_counts[key] += 1
            out.append(
                {
                    "baseline_id": query.get("baseline_id"),
                    "query_id": query.get("query_id"),
                    "candidate_id": candidate_id,
                    "scene": ledger.get("scene"),
                    "chunk_id": ledger.get("chunk_id"),
                    "source_frame": query.get("source_frame"),
                    "target_frame_id": parse_int(ledger.get("target_frame_id")),
                    "target_visibility": parse_int(ledger.get("visible_carrier_count")) > 0,
                    "target_confidence": _ledger_evidence_score(ledger),
                    "target_mask_support": parse_float(ledger.get("inside_best_mask_ratio")),
                    "target_history_support": parse_float(ledger.get("inside_any_mask_ratio")),
                    "outside_residual": parse_float(ledger.get("outside_all_related_masks_ratio")),
                    "competing_history_hit": parse_bool(ledger.get("same_frame_exclusion_violation")) or parse_float(ledger.get("related_mask_count")) > 1.0,
                    "valid_material_evidence": parse_bool(ledger.get("reprojection_success")),
                    "best_mask_observation_id": ledger.get("best_mask_observation_id", ""),
                    "related_mask_count": parse_int(ledger.get("related_mask_count")),
                    "diagnostic_source_gt": ledger.get("diagnostic_source_gt", ""),
                    "diagnostic_best_gt": ledger.get("diagnostic_best_gt", ""),
                    "diagnostic_success_same_gt": parse_bool(ledger.get("diagnostic_success_same_gt")),
                    "uses_gt_for_prediction": False,
                    "uses_gt_for_diagnostic_labels": True,
                }
            )
    return out


def _metric_row(spec: dict[str, str], query_rows: list[dict[str, Any]]) -> dict[str, Any]:
    shuffled_scores = _shuffled_scores(query_rows)
    labels = [bool(row.get("diagnostic_query_success_same_gt")) for row in query_rows]
    real_scores = [_num(row.get("real_evidence_score")) for row in query_rows]
    no_temporal_scores = [_num(row.get("no_temporal_evidence_score")) for row in query_rows]
    for row, score in zip(query_rows, shuffled_scores):
        row["shuffled_evidence_score"] = score
    real_auc = _binary_auc(labels, real_scores)
    shuffled_auc = _binary_auc(labels, shuffled_scores)
    no_temporal_auc = _binary_auc(labels, no_temporal_scores)
    entropy_before = safe_mean(_num(row.get("entropy_before")) for row in query_rows)
    entropy_after = safe_mean(_num(row.get("entropy_after")) for row in query_rows)
    entropy_reduction = safe_mean(_num(row.get("actual_entropy_reduction")) for row in query_rows)
    return {
        "baseline_id": spec["baseline_id"],
        "baseline_name": spec["baseline_name"],
        "query_budget": len(query_rows),
        "query_count": len(query_rows),
        "valid_material_evidence_rate": safe_mean(1.0 if row.get("valid_material_evidence") else 0.0 for row in query_rows),
        "target_visible_rate": safe_mean(_num(row.get("target_visible_rate")) for row in query_rows),
        "track_inside_history_rate": safe_mean(_num(row.get("track_inside_history_rate")) for row in query_rows),
        "track_competing_history_rate": safe_mean(_num(row.get("track_competing_history_rate")) for row in query_rows),
        "track_outside_rate": safe_mean(_num(row.get("track_outside_rate")) for row in query_rows),
        "explanation_entropy_before": entropy_before,
        "explanation_entropy_after": entropy_after,
        "entropy_reduction": entropy_reduction,
        "query_to_confirm_rate": safe_mean(1.0 if row.get("query_to_confirm") else 0.0 for row in query_rows),
        "query_to_quarantine_rate": safe_mean(1.0 if row.get("query_to_quarantine") else 0.0 for row in query_rows),
        "promotion_candidate_gain": safe_mean(_num(row.get("promotion_candidate_gain")) for row in query_rows),
        "real_query_AUC": real_auc,
        "shuffled_query_AUC": shuffled_auc,
        "no_temporal_query_AUC": no_temporal_auc,
        "real_minus_shuffled_query_AUC": None if real_auc is None or shuffled_auc is None else real_auc - shuffled_auc,
        "real_minus_no_temporal_query_AUC": None if real_auc is None or no_temporal_auc is None else real_auc - no_temporal_auc,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }


def _posthoc_actual_reduction_upper_bound(
    eligible_rows: list[dict[str, str]],
    candidates_by_observation: dict[str, list[dict[str, Any]]],
    query_count: int,
) -> dict[str, Any]:
    best_by_observation: list[dict[str, Any]] = []
    for row in eligible_rows:
        observation_id = str(row.get("observation_id") or "")
        options = candidates_by_observation.get(observation_id) or []
        if not options:
            continue
        entropy = _num(row.get("posterior_entropy"))
        best_candidate = max(options, key=lambda item: _actual_entropy_reduction(entropy, item))
        best_by_observation.append(
            {
                "observation_id": observation_id,
                "entropy_reduction": _actual_entropy_reduction(entropy, best_candidate),
                "query_to_confirm": bool(best_candidate.get("query_to_confirm")),
                "valid_material_evidence": bool(best_candidate.get("valid_material_evidence")),
            }
        )
    best_by_observation.sort(key=lambda row: (-_num(row.get("entropy_reduction")), str(row.get("observation_id"))))
    selected = best_by_observation[: int(query_count)]
    return {
        "query_count": int(len(selected)),
        "entropy_reduction": safe_mean(_num(row.get("entropy_reduction")) for row in selected),
        "query_to_confirm_rate": safe_mean(1.0 if row.get("query_to_confirm") else 0.0 for row in selected),
        "valid_material_evidence_rate": safe_mean(1.0 if row.get("valid_material_evidence") else 0.0 for row in selected),
    }


def _shuffled_scores(query_rows: list[dict[str, Any]]) -> list[float]:
    if not query_rows:
        return []
    output = [0.0] * len(query_rows)
    by_scene: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(query_rows):
        by_scene[str(row.get("scene") or "")].append(idx)
    for indices in by_scene.values():
        if len(indices) <= 1:
            continue
        scores = [_num(query_rows[idx].get("real_evidence_score")) for idx in indices]
        rotated = scores[1:] + scores[:1]
        for idx, score in zip(indices, rotated):
            output[idx] = score
    singletons = [idx for idx, score in enumerate(output) if score == 0.0 and len(by_scene[str(query_rows[idx].get("scene") or "")]) <= 1]
    if singletons:
        scores = [_num(row.get("real_evidence_score")) for row in query_rows]
        rotated = scores[1:] + scores[:1]
        for idx in singletons:
            output[idx] = rotated[idx]
    return output


def _binary_auc(labels: list[bool], scores: list[float]) -> float | None:
    if len(labels) != len(scores) or not labels:
        return None
    positives = [score for label, score in zip(labels, scores) if label]
    negatives = [score for label, score in zip(labels, scores) if not label]
    if not positives or not negatives:
        return None
    wins = 0.0
    total = 0
    for pos in positives:
        for neg in negatives:
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
            total += 1
    return None if total == 0 else float(wins / total)


def _ledger_evidence_score(row: dict[str, Any]) -> float:
    success = 1.0 if parse_bool(row.get("reprojection_success")) else 0.0
    visible = 1.0 if parse_int(row.get("visible_carrier_count")) > 0 else 0.0
    inside = parse_float(row.get("inside_best_mask_ratio"))
    outside = parse_float(row.get("outside_all_related_masks_ratio"))
    conflict = 1.0 if parse_bool(row.get("same_frame_exclusion_violation")) else 0.0
    return _clip(0.35 * success + 0.25 * visible + 0.25 * inside + 0.10 * (1.0 - outside) + 0.05 * (1.0 - conflict), 0.0, 1.0)


def _query_count(query_budget: int, eligible_count: int) -> int:
    if eligible_count <= 0:
        return 0
    if int(query_budget) <= 0:
        return int(eligible_count)
    return int(min(int(query_budget), int(eligible_count)))


def _pseudo_source_uv(observation_id: Any, random_seed: int) -> str:
    u = int(1000 * _stable_unit(f"u|{random_seed}|{observation_id}"))
    v = int(1000 * _stable_unit(f"v|{random_seed}|{observation_id}"))
    return json.dumps([u, v])


def _stable_unit(text: Any) -> float:
    digest = hashlib.sha256(str(text).encode("utf-8")).hexdigest()[:16]
    return int(digest, 16) / float(16**16 - 1)


def _load_json_list(value: Any) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return [item.strip() for item in text.split(";") if item.strip()]
    if isinstance(payload, list):
        return [str(item) for item in payload]
    return [str(payload)]


def _mean(values: list[float]) -> float:
    values = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(values)) if values else 0.0


def _clip(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def _num(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return float(default)
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


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
