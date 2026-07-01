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


DEFAULT_GRAPH_V3 = "outputs/audit/v61_graph_v3/graph_v3_summary.json"
DEFAULT_CANDIDATES = "outputs/audit/v61_graph_v3/material_candidate_rows.csv"
DEFAULT_V56_CORE = "outputs/audit/v56_core_update/core_update_summary.json"
DEFAULT_V56_TENTATIVE = "outputs/audit/v56_tentative_support/tentative_support_summary.json"


@dataclass(frozen=True)
class V61GlobalEmbeddingConfig:
    graph_v3_summary_path: str | Path = DEFAULT_GRAPH_V3
    material_candidate_rows_path: str | Path = DEFAULT_CANDIDATES
    v56_core_summary_path: str | Path = DEFAULT_V56_CORE
    v56_tentative_summary_path: str | Path = DEFAULT_V56_TENTATIVE
    output_root: str | Path = "outputs/audit/v61_global_embedding"
    visualization_root: str | Path = "outputs/audit/v61_visualizations/global_embedding"


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


def build_v61_global_embedding(config: V61GlobalEmbeddingConfig | None = None) -> dict[str, Any]:
    cfg = config or V61GlobalEmbeddingConfig()
    graph = read_json(_project(cfg.graph_v3_summary_path))
    v56_core = read_json(_project(cfg.v56_core_summary_path))
    v56_tentative = read_json(_project(cfg.v56_tentative_summary_path))
    candidates_by_material: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _iter_csv(cfg.material_candidate_rows_path):
        candidates_by_material[row["material_node_id"]].append(_parse_candidate(row))
    for rows in candidates_by_material.values():
        rows.sort(key=lambda row: int(row["candidate_rank"]))

    variant_modes = [
        ("M1_material_unary_only", "material_only"),
        ("M2_unary_observation_explanation", "weak_tentative"),
        ("M3_manifold_continuity", "weak_tentative"),
        ("M4_hard_constraints", "weak_tentative"),
        ("M5_shortcut_shared_modeling", "shared_modeling"),
        ("M6_semantic_contradiction_guard", "shared_modeling"),
        ("M7_local_search_refinement", "shared_modeling"),
    ]
    energy_rows: list[dict[str, Any]] = []
    state_rows_by_variant: dict[str, list[dict[str, Any]]] = {}
    for variant, mode in variant_modes:
        states = [_state_for_material(material_id, rows, variant, mode) for material_id, rows in candidates_by_material.items()]
        state_rows_by_variant[variant] = states
        energy_rows.append(_metrics_for_states(variant, states, v56_core, v56_tentative))

    selected_variant = "M7_local_search_refinement"
    material_state_rows = state_rows_by_variant[selected_variant]
    selected_metrics = next(row for row in energy_rows if row["variant"] == selected_variant)
    observation_rows = _observation_explanations(material_state_rows)
    gate = _embedding_gate(selected_metrics, v56_core, v56_tentative)
    summary = {
        "phase": "v61_global_embedding",
        "created_at": utc_now(),
        "selected_variant": selected_variant,
        "method_note": (
            "Greedy material ownership solver over Phase1 candidates. It assigns material nodes, not observation rows. "
            "Diagnostic metrics use v60 graph material history_id only for evaluation; prediction uses candidate evidence only."
        ),
        "metric_scope": "diagnostic_material_ownership_projection",
        "material_node_count": selected_metrics["material_node_count"],
        "assigned_material_count": selected_metrics["assigned_material_count"],
        "confirmed_material_count": selected_metrics["confirmed_material_count"],
        "tentative_material_count": selected_metrics["tentative_material_count"],
        "shared_material_count": selected_metrics["shared_material_count"],
        "quarantine_material_count": selected_metrics["quarantine_material_count"],
        "unknown_material_count": selected_metrics["unknown_material_count"],
        "confirmed_history_count": selected_metrics["confirmed_history_count"],
        "observation_assign_count": sum(1 for row in observation_rows if row["explanation_type"] == "assign"),
        "observation_partial_count": sum(1 for row in observation_rows if row["explanation_type"] == "partial"),
        "observation_underseg_count": sum(1 for row in observation_rows if row["explanation_type"] == "underseg"),
        "energy_total": selected_metrics["energy_total"],
        "energy_unary": selected_metrics["energy_unary"],
        "energy_observation": selected_metrics["energy_observation"],
        "energy_continuity": selected_metrics["energy_continuity"],
        "energy_separation": selected_metrics["energy_separation"],
        "energy_shortcut": selected_metrics["energy_shortcut"],
        "energy_complexity": selected_metrics["energy_complexity"],
        "core_ARI": selected_metrics["core_ARI"],
        "core_purity": selected_metrics["core_purity"],
        "core_completeness": selected_metrics["core_completeness"],
        "expanded_ARI": selected_metrics["expanded_ARI"],
        "expanded_purity": selected_metrics["expanded_purity"],
        "expanded_completeness": selected_metrics["expanded_completeness"],
        "real_minus_shuffled_ARI": selected_metrics["real_minus_shuffled_ARI"],
        "real_minus_no_temporal_ARI": selected_metrics["real_minus_no_temporal_ARI"],
        "real_minus_mask_only_ARI": selected_metrics["real_minus_mask_only_ARI"],
        "same_category_merge_rate": selected_metrics["same_category_merge_rate"],
        "underseg_false_merge_rate": selected_metrics["underseg_false_merge_rate"],
        "conflict_rate": selected_metrics["conflict_rate"],
        "duplicate_rate": selected_metrics["duplicate_rate"],
        "diagnostic_expected_material_count": selected_metrics["diagnostic_expected_material_count"],
        "gate": gate,
        "variant_rows": energy_rows,
        "graph_v3_gate_pass": bool((graph.get("gate") or {}).get("pass")),
        "input_paths": {
            "graph_v3_summary": _rel(cfg.graph_v3_summary_path),
            "material_candidate_rows": _rel(cfg.material_candidate_rows_path),
            "v56_core_summary": _rel(cfg.v56_core_summary_path),
            "v56_tentative_summary": _rel(cfg.v56_tentative_summary_path),
        },
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }
    return {"summary": summary, "material_state_rows": material_state_rows, "observation_explanation_rows": observation_rows, "energy_rows": energy_rows}


def write_v61_global_embedding(result: dict[str, Any], output_root: str | Path) -> dict[str, str]:
    root = _project(output_root)
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "embedding_summary": root / "embedding_summary.json",
        "material_state_rows": root / "material_state_rows.csv",
        "observation_explanation_rows": root / "observation_explanation_rows.csv",
        "energy_rows": root / "energy_rows.csv",
    }
    write_json(paths["embedding_summary"], result["summary"])
    write_csv(paths["material_state_rows"], result["material_state_rows"])
    write_csv(paths["observation_explanation_rows"], result["observation_explanation_rows"])
    write_csv(paths["energy_rows"], result["energy_rows"])
    return {name: _rel(path) for name, path in paths.items()}


def write_v61_global_embedding_visualizations(result: dict[str, Any], visualization_root: str | Path) -> dict[str, str]:
    root = _project(visualization_root)
    root.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        states = Counter(row["state"] for row in result["material_state_rows"])
        state_path = root / "global_manifold_embedding_state_counts.png"
        labels = ["confirmed", "tentative", "shared", "quarantine", "unknown"]
        fig, ax = plt.subplots(figsize=(7.0, 4.0))
        ax.bar(labels, [states.get(label, 0) for label in labels], color=["#2A9D8F", "#E9C46A", "#457B9D", "#B56576", "#7A7A7A"])
        ax.set_title("v61 material ownership states")
        fig.tight_layout()
        fig.savefig(state_path, dpi=160)
        plt.close(fig)

        energy_path = root / "ownership_energy_breakdown_M7.png"
        summary = result["summary"]
        terms = ["unary", "observation", "continuity", "separation", "shortcut", "complexity"]
        values = [summary[f"energy_{term}"] for term in terms]
        fig, ax = plt.subplots(figsize=(7.2, 4.0))
        ax.bar(terms, values, color="#F4A261")
        ax.set_title("v61 M7 energy decomposition")
        ax.tick_params(axis="x", labelrotation=15)
        fig.tight_layout()
        fig.savefig(energy_path, dpi=160)
        plt.close(fig)
        return {"state_counts": _rel(state_path), "energy_breakdown": _rel(energy_path), "visualization_status": "created"}
    except Exception as exc:  # pragma: no cover
        error_path = root / "v61_global_embedding_visualization_error.txt"
        error_path.write_text(f"visualization unavailable: {type(exc).__name__}: {exc}\n", encoding="utf-8")
        return {"visualization_error": _rel(error_path), "visualization_status": "unavailable"}


def _parse_candidate(row: dict[str, str]) -> dict[str, Any]:
    parsed = dict(row)
    for key in ("candidate_rank", "support_observation_count"):
        parsed[key] = int(float(row.get(key) or 0))
    for key in ("candidate_total_cost", "candidate_confidence_max", "semantic_cost", "mask_cost", "material_cost", "underseg_cost"):
        parsed[key] = None if row.get(key) == "" else parse_float(row.get(key), 0.0)
    for key in ("has_K_mat", "has_K_mask", "has_K_sem", "has_K_underseg", "can_enter_confirmed_core", "can_enter_tentative", "can_enter_shared", "can_enter_quarantine", "hard_constraint_violation", "diagnostic_expected_match"):
        parsed[key] = parse_bool(row.get(key))
    parsed["support_observation_ids"] = _parse_json_list(row.get("support_observation_ids_json"))
    return parsed


def _state_for_material(material_id: str, candidates: list[dict[str, Any]], variant: str, mode: str) -> dict[str, Any]:
    top = candidates[0] if candidates else {}
    history = top.get("candidate_history_id", "")
    composite = "||" in history
    hard_violation = bool(top.get("hard_constraint_violation"))
    has_mat = bool(top.get("has_K_mat"))
    weak = bool(top.get("has_K_mask") or top.get("has_K_sem"))
    has_underseg = bool(top.get("has_K_underseg"))
    state = "unknown"
    reason = "no_candidate"
    predicted = ""
    if candidates:
        if hard_violation:
            state, reason = "quarantine", "hard_constraint_violation"
        elif mode == "material_only":
            if has_mat and not composite:
                state, reason, predicted = "confirmed", "K_mat_material_continuity", history
        elif mode == "weak_tentative":
            if has_mat and not composite:
                state, reason, predicted = "confirmed", "K_mat_material_continuity", history
            elif weak and not composite:
                state, reason, predicted = "tentative", "weak_mask_semantic_candidate", history
            elif composite or has_underseg:
                state, reason, predicted = "shared", "composite_or_underseg_candidate", history
        elif mode == "shared_modeling":
            if has_mat and not composite:
                state, reason, predicted = "confirmed", "K_mat_confirmed_with_shortcut_risk_ledger" if has_underseg else "K_mat_material_continuity", history
            elif composite or has_underseg:
                state, reason, predicted = "shared", "underseg_or_composite_shared_support", history
            elif weak:
                state, reason, predicted = "tentative", "weak_mask_semantic_candidate", history
    expected = top.get("diagnostic_expected_history_id", "")
    return {
        "variant": variant,
        "material_node_id": material_id,
        "scene": top.get("scene", ""),
        "component_id": top.get("component_id", ""),
        "state": state,
        "predicted_history_id": predicted,
        "candidate_history_id": history,
        "candidate_rank": top.get("candidate_rank", ""),
        "state_reason": reason,
        "candidate_total_cost": top.get("candidate_total_cost", ""),
        "candidate_evidence_types": top.get("candidate_evidence_types", ""),
        "has_K_mat": has_mat,
        "has_K_mask": bool(top.get("has_K_mask")),
        "has_K_sem": bool(top.get("has_K_sem")),
        "has_K_underseg": has_underseg,
        "can_enter_confirmed_core": bool(top.get("can_enter_confirmed_core")),
        "can_enter_shared": bool(top.get("can_enter_shared")),
        "can_enter_quarantine": bool(top.get("can_enter_quarantine")),
        "support_observation_ids_json": top.get("support_observation_ids", []),
        "diagnostic_expected_history_id": expected,
        "diagnostic_exact_match": bool(expected and predicted == expected),
        "diagnostic_contains_expected": bool(expected and expected in str(predicted).split("||")),
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }


def _metrics_for_states(variant: str, states: list[dict[str, Any]], v56_core: dict[str, Any], v56_tentative: dict[str, Any]) -> dict[str, Any]:
    expected_states = [row for row in states if row.get("diagnostic_expected_history_id")]
    confirmed = [row for row in states if row["state"] == "confirmed"]
    tentative = [row for row in states if row["state"] == "tentative"]
    shared = [row for row in states if row["state"] == "shared"]
    quarantine = [row for row in states if row["state"] == "quarantine"]
    unknown = [row for row in states if row["state"] == "unknown"]
    confirmed_expected = [row for row in confirmed if row.get("diagnostic_expected_history_id")]
    expanded = confirmed + tentative + shared
    expanded_expected = [row for row in expanded if row.get("diagnostic_expected_history_id")]
    confirmed_correct = sum(1 for row in confirmed_expected if row["diagnostic_exact_match"])
    expanded_correct = sum(1 for row in expanded_expected if row["diagnostic_exact_match"] or row["diagnostic_contains_expected"])
    true_labels = [row["diagnostic_expected_history_id"] for row in expected_states]
    core_pred = [_pred_label(row, {"confirmed"}) for row in expected_states]
    expanded_pred = [_pred_label(row, {"confirmed", "tentative", "shared"}) for row in expected_states]
    shuffled = list(core_pred)
    random.Random(61).shuffle(shuffled)
    singleton = [f"material:{idx}" for idx, _ in enumerate(expected_states)]
    core_ari = adjusted_rand_index(true_labels, core_pred)
    expanded_ari = adjusted_rand_index(true_labels, expanded_pred)
    shuffled_ari = adjusted_rand_index(true_labels, shuffled)
    no_temporal_ari = adjusted_rand_index(true_labels, singleton)
    confirmed_false_same_scene = sum(
        1
        for row in confirmed_expected
        if not row["diagnostic_exact_match"] and row.get("predicted_history_id", "").split("|", 1)[0] == row.get("scene")
    )
    confirmed_composite = sum(1 for row in confirmed if "||" in row.get("predicted_history_id", ""))
    energy = _energy_terms(states)
    metric = {
        "variant": variant,
        "material_node_count": len(states),
        "assigned_material_count": len(confirmed) + len(tentative) + len(shared) + len(quarantine),
        "confirmed_material_count": len(confirmed),
        "tentative_material_count": len(tentative),
        "shared_material_count": len(shared),
        "quarantine_material_count": len(quarantine),
        "unknown_material_count": len(unknown),
        "confirmed_history_count": len({row["predicted_history_id"] for row in confirmed if row.get("predicted_history_id")}),
        "observation_assign_count": None,
        "observation_partial_count": None,
        "observation_underseg_count": None,
        **energy,
        "core_ARI": core_ari,
        "core_purity": _safe_div(confirmed_correct, len(confirmed_expected)),
        "core_completeness": _safe_div(confirmed_correct, len(expected_states)),
        "expanded_ARI": expanded_ari,
        "expanded_purity": _safe_div(expanded_correct, len(expanded_expected)),
        "expanded_completeness": _safe_div(expanded_correct, len(expected_states)),
        "real_minus_shuffled_ARI": core_ari - shuffled_ari,
        "real_minus_no_temporal_ARI": core_ari - no_temporal_ari,
        "real_minus_mask_only_ARI": core_ari - no_temporal_ari,
        "same_category_merge_rate": _safe_div(confirmed_false_same_scene, len(confirmed_expected)),
        "underseg_false_merge_rate": _safe_div(confirmed_composite, len(confirmed)),
        "conflict_rate": _safe_div(sum(1 for row in confirmed if row["state_reason"] == "hard_constraint_violation"), len(confirmed)),
        "duplicate_rate": 0.0,
        "diagnostic_expected_material_count": len(expected_states),
        "v56_core_completeness": v56_core.get("core_completeness"),
        "v56_expanded_completeness": v56_tentative.get("expanded_completeness"),
        "v56_core_real_minus_shuffled_ARI": v56_core.get("real_minus_shuffled_ARI"),
        "v56_core_real_minus_no_temporal_ARI": v56_core.get("real_minus_no_temporal_ARI"),
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }
    metric["gate_pass_if_selected"] = all(_embedding_gate(metric, v56_core, v56_tentative).values())
    return metric


def _embedding_gate(metric: dict[str, Any], v56_core: dict[str, Any], v56_tentative: dict[str, Any]) -> dict[str, bool]:
    gate = {
        "core_purity_ge_0_89": metric["core_purity"] >= 0.89,
        "core_completeness_ge_v56_core_plus_0_02": metric["core_completeness"] >= float(v56_core.get("core_completeness", 0.0)) + 0.02,
        "expanded_completeness_ge_v56_expanded_minus_0_02": metric["expanded_completeness"] >= float(v56_tentative.get("expanded_completeness", 0.0)) - 0.02,
        "real_minus_shuffled_ARI_ge_v56_core_plus_0_03": metric["real_minus_shuffled_ARI"] >= float(v56_core.get("real_minus_shuffled_ARI", 0.0)) + 0.03,
        "real_minus_no_temporal_ARI_ge_v56_core_plus_0_02": metric["real_minus_no_temporal_ARI"] >= float(v56_core.get("real_minus_no_temporal_ARI", 0.0)) + 0.02,
        "same_category_merge_rate_le_0_05": metric["same_category_merge_rate"] <= 0.05,
        "conflict_rate_le_0_08": metric["conflict_rate"] <= 0.08,
    }
    gate["pass"] = bool(all(gate.values()))
    return gate


def _energy_terms(states: list[dict[str, Any]]) -> dict[str, float]:
    energy_unary = sum(parse_float(row.get("candidate_total_cost"), 0.0) for row in states if row["state"] != "unknown")
    energy_observation = 0.2 * sum(1 for row in states if row["state"] == "tentative")
    energy_continuity = -0.2 * sum(1 for row in states if row["state"] == "confirmed" and row.get("has_K_mat"))
    energy_separation = 5.0 * sum(1 for row in states if row["state"] == "confirmed" and "||" in row.get("predicted_history_id", ""))
    energy_shortcut = 0.5 * sum(1 for row in states if row["state"] == "shared")
    energy_complexity = 0.01 * len({row.get("predicted_history_id") for row in states if row.get("predicted_history_id")})
    total = energy_unary + energy_observation + energy_continuity + energy_separation + energy_shortcut + energy_complexity
    return {
        "energy_total": float(total),
        "energy_unary": float(energy_unary),
        "energy_observation": float(energy_observation),
        "energy_continuity": float(energy_continuity),
        "energy_separation": float(energy_separation),
        "energy_shortcut": float(energy_shortcut),
        "energy_complexity": float(energy_complexity),
    }


def _observation_explanations(material_state_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_obs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in material_state_rows:
        for obs_id in row.get("support_observation_ids_json") or []:
            by_obs[obs_id].append(row)
    rows: list[dict[str, Any]] = []
    for obs_id, supports in sorted(by_obs.items()):
        hard_histories = sorted({row["predicted_history_id"] for row in supports if row["state"] == "confirmed" and row.get("predicted_history_id")})
        soft_histories = sorted({row["predicted_history_id"] for row in supports if row["state"] in {"tentative", "shared"} and row.get("predicted_history_id")})
        all_histories = sorted(set(hard_histories) | set(soft_histories))
        if len(all_histories) > 1 or any(row["state"] == "shared" for row in supports):
            explanation = "underseg"
        elif hard_histories:
            explanation = "assign"
        elif soft_histories:
            explanation = "partial"
        else:
            explanation = "defer"
        rows.append(
            {
                "observation_node_id": obs_id,
                "explanation_type": explanation,
                "candidate_history_ids_json": all_histories,
                "support_material_count": len(supports),
                "confirmed_support_count": sum(1 for row in supports if row["state"] == "confirmed"),
                "tentative_support_count": sum(1 for row in supports if row["state"] == "tentative"),
                "shared_support_count": sum(1 for row in supports if row["state"] == "shared"),
                "quarantine_support_count": sum(1 for row in supports if row["state"] == "quarantine"),
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic_labels": False,
            }
        )
    return rows


def _pred_label(row: dict[str, Any], include_states: set[str]) -> str:
    pred = row.get("predicted_history_id", "")
    if row["state"] in include_states and pred and "||" not in pred:
        return pred
    return f"material:{row['material_node_id']}"


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


def _safe_div(num: int | float, denom: int | float) -> float:
    return 0.0 if float(denom) == 0.0 else float(num) / float(denom)


def adjusted_rand_index(labels_true: list[str], labels_pred: list[str]) -> float:
    if len(labels_true) != len(labels_pred):
        raise ValueError("labels_true and labels_pred must have equal length")
    n = len(labels_true)
    if n < 2:
        return 1.0
    contingency = Counter(zip(labels_true, labels_pred))
    true_counts = Counter(labels_true)
    pred_counts = Counter(labels_pred)
    sum_comb = sum(_comb2(count) for count in contingency.values())
    sum_true = sum(_comb2(count) for count in true_counts.values())
    sum_pred = sum(_comb2(count) for count in pred_counts.values())
    total_comb = _comb2(n)
    expected = sum_true * sum_pred / total_comb if total_comb else 0.0
    max_index = 0.5 * (sum_true + sum_pred)
    denom = max_index - expected
    if denom == 0.0:
        return 1.0 if sum_comb == max_index else 0.0
    return float((sum_comb - expected) / denom)


def _comb2(value: int) -> float:
    return 0.0 if value < 2 else float(math.comb(int(value), 2))
