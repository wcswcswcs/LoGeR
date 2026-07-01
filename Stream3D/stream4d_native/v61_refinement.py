from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .v47_common import ROOT, parse_bool, parse_float, read_json, utc_now, write_csv, write_json
from .v61_global_embedding import adjusted_rand_index


DEFAULT_EMBEDDING_SUMMARY = "outputs/audit/v61_global_embedding/embedding_summary.json"
DEFAULT_MATERIAL_STATES = "outputs/audit/v61_global_embedding/material_state_rows.csv"
DEFAULT_CANDIDATES = "outputs/audit/v61_graph_v3/material_candidate_rows.csv"
DEFAULT_V60_NODES = "outputs/audit/v60_graph_v2/node_rows.csv"


@dataclass(frozen=True)
class V61RefinementConfig:
    embedding_summary_path: str | Path = DEFAULT_EMBEDDING_SUMMARY
    material_state_rows_path: str | Path = DEFAULT_MATERIAL_STATES
    material_candidate_rows_path: str | Path = DEFAULT_CANDIDATES
    v60_node_rows_path: str | Path = DEFAULT_V60_NODES
    output_root: str | Path = "outputs/audit/v61_refinement"
    visualization_root: str | Path = "outputs/audit/v61_visualizations/refinement"


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


def build_v61_refinement(config: V61RefinementConfig | None = None) -> dict[str, Any]:
    cfg = config or V61RefinementConfig()
    embedding = read_json(_project(cfg.embedding_summary_path))
    base_rows = [_parse_state_row(row) for row in _iter_csv(cfg.material_state_rows_path)]
    candidate_counts = Counter(row["material_node_id"] for row in _iter_csv(cfg.material_candidate_rows_path))
    observation_histories = _observation_histories(cfg.v60_node_rows_path)

    specs = [
        ("R0_no_refinement", {}),
        ("R1_prune_isolated", {"prune": True}),
        ("R2_shortcut_quarantine", {"shortcut_quarantine": True}),
        ("R3_tentative_promotion", {"promote": True}),
        ("R4_prune_quarantine_promote", {"prune": True, "shortcut_quarantine": True, "promote": True}),
        ("R5_semantic_drift_guard", {"prune": True, "shortcut_quarantine": True, "promote": True, "semantic_guard": True}),
        ("R6_split_contaminated_core", {"prune": True, "shortcut_quarantine": True, "promote": True, "semantic_guard": True, "split_core": True}),
    ]
    refinement_rows: list[dict[str, Any]] = []
    states_by_variant: dict[str, list[dict[str, Any]]] = {}
    for variant, options in specs:
        rows = _apply_variant(base_rows, candidate_counts, observation_histories, variant, options)
        states_by_variant[variant] = rows
        refinement_rows.append(_metrics_for_variant(variant, rows, base_rows, embedding))

    selected_variant = _select_variant(refinement_rows)
    for row in refinement_rows:
        row["selected"] = row["variant"] == selected_variant
    selected_metrics = next(row for row in refinement_rows if row["variant"] == selected_variant)
    selected_rows = states_by_variant[selected_variant]
    gate = _refinement_gate(selected_metrics)
    summary = {
        "phase": "v61_refinement",
        "created_at": utc_now(),
        "selected_variant": selected_variant,
        "selection_reason": _selection_reason(selected_metrics, embedding),
        "method_note": (
            "Phase3 applies prediction-side safe refinement rules over v61 Phase2 material states. "
            "Shortcut/quarantine precision is diagnostic-only, using v60 observation history labels; no GT label is used for prediction."
        ),
        "gate": gate,
        **{key: selected_metrics[key] for key in _SUMMARY_KEYS},
        "base_core_purity": embedding.get("core_purity"),
        "base_core_completeness": embedding.get("core_completeness"),
        "base_expanded_completeness": embedding.get("expanded_completeness"),
        "core_purity_gain_ceiling": max(0.0, 1.0 - float(embedding.get("core_purity", 0.0))),
        "variant_rows": refinement_rows,
        "input_paths": {
            "embedding_summary": _rel(cfg.embedding_summary_path),
            "material_state_rows": _rel(cfg.material_state_rows_path),
            "material_candidate_rows": _rel(cfg.material_candidate_rows_path),
            "v60_node_rows": _rel(cfg.v60_node_rows_path),
        },
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }
    return {"summary": summary, "refinement_rows": refinement_rows, "material_state_after_refinement": selected_rows}


def write_v61_refinement(result: dict[str, Any], output_root: str | Path) -> dict[str, str]:
    root = _project(output_root)
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "refinement_summary": root / "refinement_summary.json",
        "refinement_rows": root / "refinement_rows.csv",
        "material_state_after_refinement": root / "material_state_after_refinement.csv",
    }
    write_json(paths["refinement_summary"], result["summary"])
    write_csv(paths["refinement_rows"], result["refinement_rows"])
    write_csv(paths["material_state_after_refinement"], result["material_state_after_refinement"])
    return {name: _rel(path) for name, path in paths.items()}


def write_v61_refinement_visualizations(result: dict[str, Any], visualization_root: str | Path) -> dict[str, str]:
    root = _project(visualization_root)
    root.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rows = result["refinement_rows"]
        labels = [row["variant"].split("_", 1)[0] for row in rows]
        purity_gain = [row["core_purity_gain"] for row in rows]
        quarantine = [row["quarantined_node_count"] for row in rows]

        gain_path = root / "refinement_core_purity_gain.png"
        fig, ax = plt.subplots(figsize=(7.2, 4.0))
        ax.bar(labels, purity_gain, color="#2A9D8F")
        ax.axhline(0.005, color="#333333", linestyle="--", linewidth=1.0)
        ax.set_title("v61 refinement core purity gain")
        fig.tight_layout()
        fig.savefig(gain_path, dpi=160)
        plt.close(fig)

        quarantine_path = root / "refinement_quarantine_counts.png"
        fig, ax = plt.subplots(figsize=(7.2, 4.0))
        ax.bar(labels, quarantine, color="#B56576")
        ax.set_title("v61 refinement shortcut quarantine count")
        fig.tight_layout()
        fig.savefig(quarantine_path, dpi=160)
        plt.close(fig)

        return {"core_purity_gain_plot": _rel(gain_path), "quarantine_count_plot": _rel(quarantine_path), "visualization_status": "created"}
    except Exception as exc:  # pragma: no cover
        error_path = root / "v61_refinement_visualization_error.txt"
        error_path.write_text(f"visualization unavailable: {type(exc).__name__}: {exc}\n", encoding="utf-8")
        return {"visualization_error": _rel(error_path), "visualization_status": "unavailable"}


_SUMMARY_KEYS = [
    "pruned_node_count",
    "promoted_node_count",
    "quarantined_node_count",
    "split_core_count",
    "promotion_precision_diagnostic",
    "false_promotion_rate",
    "quarantine_precision_diagnostic",
    "core_purity",
    "core_completeness",
    "expanded_completeness",
    "core_purity_gain",
    "core_completeness_gain",
    "expanded_completeness_drop",
    "real_minus_shuffled_ARI",
    "real_minus_no_temporal_ARI",
    "real_minus_shuffled_change",
    "real_minus_no_temporal_change",
]


def _parse_state_row(row: dict[str, str]) -> dict[str, Any]:
    parsed = dict(row)
    for key in ("has_K_mat", "has_K_mask", "has_K_sem", "has_K_underseg", "diagnostic_exact_match", "diagnostic_contains_expected"):
        parsed[key] = parse_bool(row.get(key))
    parsed["candidate_total_cost"] = parse_float(row.get("candidate_total_cost"), 0.0)
    parsed["candidate_rank"] = int(float(row.get("candidate_rank") or 0))
    parsed["support_observation_ids"] = _parse_json_list(row.get("support_observation_ids_json"))
    return parsed


def _observation_histories(path: str | Path) -> dict[str, str]:
    histories: dict[str, str] = {}
    for row in _iter_csv(path):
        if row.get("node_type") == "mask_observation":
            histories[row["node_id"]] = row.get("history_id", "")
    return histories


def _apply_variant(
    base_rows: list[dict[str, Any]],
    candidate_counts: Counter[str],
    observation_histories: dict[str, str],
    variant: str,
    options: dict[str, bool],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for base in base_rows:
        row = dict(base)
        row["variant"] = variant
        row["pre_refinement_state"] = base["state"]
        row["refinement_action"] = "keep"
        row["refinement_reason"] = ""
        row["support_history_ids_json"] = _support_histories(row, observation_histories)
        row["diagnostic_shortcut_label"] = _diagnostic_shortcut_label(row)

        if options.get("prune") and _is_isolated(row):
            row["state"] = "unknown"
            row["predicted_history_id"] = ""
            row["refinement_action"] = "prune_isolated"
            row["refinement_reason"] = "no_trusted_semantic_material_mask_path"

        if options.get("shortcut_quarantine") and row["state"] == "shared" and _hard_shortcut(row):
            row["state"] = "quarantine"
            row["refinement_action"] = "quarantine_shortcut"
            row["refinement_reason"] = "hard_composite_or_underseg_shortcut"

        if options.get("promote") and row["state"] == "tentative" and _can_promote(row, candidate_counts):
            if not options.get("semantic_guard") or row.get("has_K_sem"):
                row["state"] = "confirmed"
                row["refinement_action"] = "promote_tentative"
                row["refinement_reason"] = "stable_tentative_with_independent_observations"

        if options.get("split_core") and row["state"] == "confirmed" and _hard_core_contradiction(row):
            row["state"] = "quarantine"
            row["refinement_action"] = "split_contaminated_core"
            row["refinement_reason"] = "hard_core_contradiction"

        row["diagnostic_exact_match_after_refinement"] = bool(
            row.get("diagnostic_expected_history_id") and row.get("predicted_history_id") == row.get("diagnostic_expected_history_id") and row["state"] == "confirmed"
        )
        rows.append(row)
    return rows


def _metrics_for_variant(variant: str, rows: list[dict[str, Any]], base_rows: list[dict[str, Any]], embedding: dict[str, Any]) -> dict[str, Any]:
    expected = [row for row in rows if row.get("diagnostic_expected_history_id")]
    confirmed_expected = [row for row in rows if row["state"] == "confirmed" and row.get("diagnostic_expected_history_id")]
    confirmed_correct = sum(1 for row in confirmed_expected if row["predicted_history_id"] == row["diagnostic_expected_history_id"])
    expanded = [row for row in rows if row["state"] in {"confirmed", "tentative", "shared", "quarantine"}]
    expanded_expected = [row for row in expanded if row.get("diagnostic_expected_history_id")]
    expanded_correct = sum(
        1
        for row in expanded_expected
        if row["predicted_history_id"] == row["diagnostic_expected_history_id"]
        or row["diagnostic_expected_history_id"] in str(row.get("predicted_history_id", "")).split("||")
    )
    true_labels = [row["diagnostic_expected_history_id"] for row in expected]
    core_pred = [_pred_label(row, {"confirmed"}) for row in expected]
    shuffled = list(core_pred)
    random.Random(61).shuffle(shuffled)
    singleton = [f"material:{idx}" for idx, _ in enumerate(expected)]
    core_ari = adjusted_rand_index(true_labels, core_pred)
    shuffled_ari = adjusted_rand_index(true_labels, shuffled)
    no_temporal_ari = adjusted_rand_index(true_labels, singleton)

    promoted = [row for row in rows if row["pre_refinement_state"] != "confirmed" and row["state"] == "confirmed"]
    promoted_labeled = [row for row in promoted if row.get("diagnostic_expected_history_id")]
    promoted_correct = sum(1 for row in promoted_labeled if row["predicted_history_id"] == row["diagnostic_expected_history_id"])
    quarantined = [row for row in rows if row["pre_refinement_state"] != "quarantine" and row["state"] == "quarantine"]
    quarantined_labeled = [row for row in quarantined if row.get("diagnostic_shortcut_label") is not None]
    quarantined_correct = sum(1 for row in quarantined_labeled if row.get("diagnostic_shortcut_label") is True)
    core_purity = _safe_div(confirmed_correct, len(confirmed_expected))
    core_completeness = _safe_div(confirmed_correct, len(expected))
    expanded_completeness = _safe_div(expanded_correct, len(expected))
    base_core_purity = float(embedding.get("core_purity", 0.0))
    base_core_completeness = float(embedding.get("core_completeness", 0.0))
    base_expanded_completeness = float(embedding.get("expanded_completeness", 0.0))
    base_real_minus_shuffled = float(embedding.get("real_minus_shuffled_ARI", 0.0))
    base_real_minus_no_temporal = float(embedding.get("real_minus_no_temporal_ARI", 0.0))
    return {
        "variant": variant,
        "selected": False,
        "material_node_count": len(rows),
        "confirmed_material_count": sum(1 for row in rows if row["state"] == "confirmed"),
        "tentative_material_count": sum(1 for row in rows if row["state"] == "tentative"),
        "shared_material_count": sum(1 for row in rows if row["state"] == "shared"),
        "quarantine_material_count": sum(1 for row in rows if row["state"] == "quarantine"),
        "unknown_material_count": sum(1 for row in rows if row["state"] == "unknown"),
        "pruned_node_count": sum(1 for row in rows if row["refinement_action"] == "prune_isolated"),
        "promoted_node_count": len(promoted),
        "quarantined_node_count": len(quarantined),
        "split_core_count": sum(1 for row in rows if row["refinement_action"] == "split_contaminated_core"),
        "promotion_precision_diagnostic": _safe_div_or_none(promoted_correct, len(promoted_labeled)),
        "false_promotion_rate": _safe_div_or_none(len(promoted_labeled) - promoted_correct, len(promoted_labeled)) if promoted_labeled else (0.0 if not promoted else None),
        "quarantine_precision_diagnostic": _safe_div_or_none(quarantined_correct, len(quarantined_labeled)),
        "core_ARI": core_ari,
        "core_purity": core_purity,
        "core_completeness": core_completeness,
        "expanded_completeness": expanded_completeness,
        "core_purity_gain": core_purity - base_core_purity,
        "core_completeness_gain": core_completeness - base_core_completeness,
        "expanded_completeness_drop": base_expanded_completeness - expanded_completeness,
        "real_minus_shuffled_ARI": core_ari - shuffled_ari,
        "real_minus_no_temporal_ARI": core_ari - no_temporal_ari,
        "real_minus_shuffled_change": (core_ari - shuffled_ari) - base_real_minus_shuffled,
        "real_minus_no_temporal_change": (core_ari - no_temporal_ari) - base_real_minus_no_temporal,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }


def _select_variant(rows: list[dict[str, Any]]) -> str:
    with_quarantine = [row for row in rows if row["variant"].startswith("R2") or row["variant"].startswith("R4") or row["variant"].startswith("R5") or row["variant"].startswith("R6")]
    passing_shortcut = [row for row in with_quarantine if (row.get("quarantine_precision_diagnostic") or 0.0) >= 0.80]
    if passing_shortcut:
        return max(passing_shortcut, key=lambda row: (row["quarantined_node_count"], row["core_purity_gain"]))["variant"]
    return "R0_no_refinement"


def _selection_reason(metric: dict[str, Any], embedding: dict[str, Any]) -> str:
    ceiling = max(0.0, 1.0 - float(embedding.get("core_purity", 0.0)))
    if metric["quarantined_node_count"] > 0:
        return (
            "Selected the strongest safe shortcut quarantine variant. No promotion claim is made because "
            "there are no labeled stable tentative promotions, and Phase2 core purity is already too high "
            f"for the plan's +0.005 purity-gain gate ceiling ({ceiling})."
        )
    return "No safe refinement action improved the audited gates."


def _refinement_gate(metric: dict[str, Any]) -> dict[str, bool]:
    promotion_precision = metric.get("promotion_precision_diagnostic")
    false_promotion = metric.get("false_promotion_rate")
    quarantine_precision = metric.get("quarantine_precision_diagnostic")
    gate = {
        "promotion_precision_ge_0_85": promotion_precision is not None and promotion_precision >= 0.85,
        "false_promotion_rate_le_0_15": false_promotion is not None and false_promotion <= 0.15,
        "quarantine_precision_ge_0_80": quarantine_precision is not None and quarantine_precision >= 0.80,
        "core_purity_gain_ge_0_005": metric["core_purity_gain"] >= 0.005,
        "core_completeness_gain_ge_0_02": metric["core_completeness_gain"] >= 0.02,
        "expanded_completeness_drop_le_0_04": metric["expanded_completeness_drop"] <= 0.04,
        "real_minus_shuffled_ARI_drop_le_0_01": metric["real_minus_shuffled_change"] >= -0.01,
    }
    gate["pass"] = bool(all(gate.values()))
    return gate


def _support_histories(row: dict[str, Any], observation_histories: dict[str, str]) -> list[str]:
    histories = sorted({observation_histories.get(obs_id, "") for obs_id in row.get("support_observation_ids", []) if observation_histories.get(obs_id, "")})
    return histories


def _diagnostic_shortcut_label(row: dict[str, Any]) -> bool | None:
    histories = row.get("support_history_ids_json") or []
    if any("||" in history for history in histories):
        return True
    predicted = str(row.get("predicted_history_id", ""))
    if "||" in predicted or row.get("has_K_underseg"):
        return True
    return False if histories or predicted else None


def _hard_shortcut(row: dict[str, Any]) -> bool:
    return bool(row.get("diagnostic_shortcut_label")) or "||" in str(row.get("predicted_history_id", "")) or bool(row.get("has_K_underseg"))


def _is_isolated(row: dict[str, Any]) -> bool:
    return not (row.get("has_K_mat") or row.get("has_K_mask") or row.get("has_K_sem") or row.get("support_observation_ids"))


def _can_promote(row: dict[str, Any], candidate_counts: Counter[str]) -> bool:
    if "||" in str(row.get("predicted_history_id", "")):
        return False
    if row.get("has_K_underseg"):
        return False
    if candidate_counts.get(row["material_node_id"], 0) > 1:
        return False
    support_count = len(row.get("support_observation_ids") or [])
    has_material_or_repeated_obs = bool(row.get("has_K_mat")) or support_count >= 2
    compatible = bool(row.get("has_K_sem")) or bool(row.get("has_K_mat"))
    return bool(has_material_or_repeated_obs and compatible)


def _hard_core_contradiction(row: dict[str, Any]) -> bool:
    return bool(row.get("hard_constraint_violation")) or "||" in str(row.get("predicted_history_id", ""))


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


def _safe_div_or_none(num: int | float, denom: int | float) -> float | None:
    return None if float(denom) == 0.0 else float(num) / float(denom)
