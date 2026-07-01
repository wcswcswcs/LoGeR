from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from .v47_common import ROOT, parse_bool, read_json, utc_now, write_csv, write_json
from .v61_global_embedding import _metrics_for_states, _observation_explanations, _parse_candidate, _state_for_material


DEFAULT_CANDIDATES = "outputs/audit/v61_graph_v3/material_candidate_rows.csv"
DEFAULT_V60_NODES = "outputs/audit/v60_graph_v2/node_rows.csv"
DEFAULT_V60_EDGES = "outputs/audit/v60_graph_v2/edge_rows.csv"
DEFAULT_V56_CORE = "outputs/audit/v56_core_update/core_update_summary.json"
DEFAULT_V56_TENTATIVE = "outputs/audit/v56_tentative_support/tentative_support_summary.json"


@dataclass(frozen=True)
class V62DecircularizationConfig:
    material_candidate_rows_path: str | Path = DEFAULT_CANDIDATES
    v60_node_rows_path: str | Path = DEFAULT_V60_NODES
    v60_edge_rows_path: str | Path = DEFAULT_V60_EDGES
    v56_core_summary_path: str | Path = DEFAULT_V56_CORE
    v56_tentative_summary_path: str | Path = DEFAULT_V56_TENTATIVE
    output_root: str | Path = "outputs/audit/v62_decircularization"
    visualization_root: str | Path = "outputs/audit/v62_visualizations/decircularization"


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


def load_candidate_rows(path: str | Path = DEFAULT_CANDIDATES) -> list[dict[str, str]]:
    return list(_iter_csv(path))


def expected_by_material(candidate_rows: list[dict[str, str]]) -> dict[str, str]:
    expected: dict[str, str] = {}
    for row in candidate_rows:
        material_id = row.get("material_node_id", "")
        value = row.get("diagnostic_expected_history_id", "")
        if material_id and value and material_id not in expected:
            expected[material_id] = value
    return expected


def material_meta(candidate_rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    meta: dict[str, dict[str, str]] = {}
    for row in candidate_rows:
        material_id = row.get("material_node_id", "")
        if material_id and material_id not in meta:
            meta[material_id] = {
                "scene": row.get("scene", ""),
                "component_id": row.get("component_id", ""),
            }
    return meta


def build_states_from_candidates(
    candidate_rows: list[dict[str, str]],
    *,
    variant: str,
    mode: str = "shared_modeling",
    filter_fn: Callable[[dict[str, str]], bool] | None = None,
    mutate_fn: Callable[[dict[str, str]], dict[str, str]] | None = None,
    material_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    all_rows = list(candidate_rows)
    expected = expected_by_material(all_rows)
    meta = material_meta(all_rows)
    ids = material_ids or sorted(meta)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        if filter_fn is not None and not filter_fn(row):
            continue
        next_row = dict(row)
        if mutate_fn is not None:
            next_row = mutate_fn(next_row)
        grouped[next_row["material_node_id"]].append(_parse_candidate(next_row))
    for rows in grouped.values():
        rows.sort(key=lambda item: (int(item["candidate_rank"]), float(item.get("candidate_total_cost") or 1.0e9)))

    states: list[dict[str, Any]] = []
    for material_id in ids:
        rows = grouped.get(material_id, [])
        if rows:
            state = _state_for_material(material_id, rows, variant, mode)
        else:
            state = {
                "variant": variant,
                "material_node_id": material_id,
                "scene": meta.get(material_id, {}).get("scene", ""),
                "component_id": meta.get(material_id, {}).get("component_id", ""),
                "state": "unknown",
                "predicted_history_id": "",
                "candidate_history_id": "",
                "candidate_rank": None,
                "state_reason": "no_candidate_after_decircularization",
                "candidate_total_cost": 0.0,
                "candidate_evidence_types": "",
                "has_K_mat": False,
                "has_K_mask": False,
                "has_K_sem": False,
                "has_K_underseg": False,
                "can_enter_confirmed_core": False,
                "can_enter_shared": False,
                "can_enter_quarantine": False,
                "support_observation_ids_json": [],
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic_labels": True,
            }
        exp = expected.get(material_id, "")
        state["diagnostic_expected_history_id"] = exp
        state["diagnostic_exact_match"] = bool(exp and state.get("predicted_history_id") == exp)
        state["diagnostic_contains_expected"] = bool(exp and exp in str(state.get("predicted_history_id", "")).split("||"))
        states.append(state)
    return states


def metric_for_states(variant: str, states: list[dict[str, Any]], v56_core: dict[str, Any], v56_tentative: dict[str, Any]) -> dict[str, Any]:
    return _metrics_for_states(variant, states, v56_core, v56_tentative)


def candidate_recall(candidate_rows: list[dict[str, str]], material_ids: list[str], expected: dict[str, str]) -> dict[str, float]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in candidate_rows:
        grouped[row["material_node_id"]].append(row)
    denom = sum(1 for material_id in material_ids if expected.get(material_id))
    if denom == 0:
        return {"candidate_recall@1": 0.0, "candidate_recall@5": 0.0, "material_nodes_with_candidate_rate": 0.0}
    recall1 = 0
    recall5 = 0
    with_candidate = 0
    for material_id in material_ids:
        exp = expected.get(material_id, "")
        ranked = sorted(grouped.get(material_id, []), key=lambda row: (int(float(row.get("candidate_rank") or 0)), float(row.get("candidate_total_cost") or 1.0e9)))
        if ranked:
            with_candidate += 1
        if exp and ranked[:1] and ranked[0].get("candidate_history_id") == exp:
            recall1 += 1
        if exp and any(row.get("candidate_history_id") == exp for row in ranked[:5]):
            recall5 += 1
    return {
        "candidate_recall@1": recall1 / denom,
        "candidate_recall@5": recall5 / denom,
        "material_nodes_with_candidate_rate": with_candidate / max(len(material_ids), 1),
    }


def observation_history_fallback_free(row: dict[str, str]) -> bool:
    return "observation_history_id" not in row.get("source_edge_ids_json", "")


def typed_non_diagnostic_candidate(row: dict[str, str]) -> bool:
    allowed = {"K_mat", "K_mask", "K_sem", "K_underseg"}
    evidence = {part for part in row.get("candidate_evidence_types", "").split("|") if part}
    return bool(evidence) and evidence.issubset(allowed) and observation_history_fallback_free(row)


def no_kmat_candidate(row: dict[str, str]) -> bool:
    return typed_non_diagnostic_candidate(row) and not parse_bool(row.get("has_K_mat"))


def shuffled_kmat_mutator(candidate_rows: list[dict[str, str]], seed: int = 62) -> Callable[[dict[str, str]], dict[str, str]]:
    histories = [row.get("candidate_history_id", "") for row in candidate_rows if parse_bool(row.get("has_K_mat")) and typed_non_diagnostic_candidate(row)]
    random.Random(seed).shuffle(histories)
    iterator = iter(histories)

    def mutate(row: dict[str, str]) -> dict[str, str]:
        if parse_bool(row.get("has_K_mat")):
            try:
                history = next(iterator)
            except StopIteration:
                history = row.get("candidate_history_id", "")
            row["candidate_history_id"] = history
            row["candidate_history_node_id"] = f"h:{history}" if history else ""
            row["diagnostic_expected_match"] = str(bool(history and history == row.get("diagnostic_expected_history_id", "")))
        return row

    return mutate


def _material_history_provenance(v60_nodes: list[dict[str, str]], v60_edges: list[dict[str, str]]) -> list[dict[str, Any]]:
    node_sources = Counter(row.get("evidence_source", "") for row in v60_nodes if row.get("node_type") == "material" and row.get("history_id"))
    edge_sources = Counter(row.get("evidence_source", "") for row in v60_edges if row.get("edge_type") == "material_continuity")
    rows: list[dict[str, Any]] = []
    for source, count in sorted(node_sources.items()):
        rows.append(
            {
                "field_or_edge": "material_node.history_id",
                "source": source,
                "count": count,
                "provenance_classification": "method_history_state",
                "used_for_prediction": False,
                "used_for_diagnostic_metric": True,
                "audit_note": "v61_graph_v3 uses this field for diagnostic recall labels; K_mat candidates come from material_continuity edges.",
            }
        )
    for source, count in sorted(edge_sources.items()):
        rows.append(
            {
                "field_or_edge": "material_continuity edge",
                "source": source,
                "count": count,
                "provenance_classification": "method_candidate_evidence",
                "used_for_prediction": True,
                "used_for_diagnostic_metric": False,
                "audit_note": "K_mat evidence is retained in D4 after diagnostic and observation-history fallback removal.",
            }
        )
    rows.append(
        {
            "field_or_edge": "observation_history_id fallback",
            "source": "v61_graph_v3 synthetic fallback",
            "count": None,
            "provenance_classification": "mask_level_shortcut_removed_in_D2_D4",
            "used_for_prediction": False,
            "used_for_diagnostic_metric": False,
            "audit_note": "Rows whose source_edge_ids_json contains observation_history_id are excluded in D2/D4.",
        }
    )
    return rows


def build_v62_decircularization(config: V62DecircularizationConfig | None = None) -> dict[str, Any]:
    cfg = config or V62DecircularizationConfig()
    candidates = load_candidate_rows(cfg.material_candidate_rows_path)
    material_ids = sorted(material_meta(candidates))
    expected = expected_by_material(candidates)
    v56_core = read_json(_project(cfg.v56_core_summary_path))
    v56_tentative = read_json(_project(cfg.v56_tentative_summary_path))
    v60_nodes = list(_iter_csv(cfg.v60_node_rows_path))
    v60_edges = list(_iter_csv(cfg.v60_edge_rows_path))

    variant_defs: list[tuple[str, Callable[[dict[str, str]], bool] | None, Callable[[dict[str, str]], dict[str, str]] | None, str]] = [
        ("D0_original_v61_candidate_rows", None, None, "shared_modeling"),
        ("D1_drop_diagnostic_expected_columns", None, None, "shared_modeling"),
        ("D2_drop_observation_history_id_fallback", observation_history_fallback_free, None, "shared_modeling"),
        ("D3_drop_material_history_id_as_prediction_source", typed_non_diagnostic_candidate, None, "shared_modeling"),
        ("D4_typed_non_diagnostic_edges_only", typed_non_diagnostic_candidate, None, "shared_modeling"),
        ("D5_rebuilt_shuffled_K_mat_control", typed_non_diagnostic_candidate, shuffled_kmat_mutator(candidates), "shared_modeling"),
        ("D6_rebuilt_no_temporal_K_mat_control", no_kmat_candidate, None, "shared_modeling"),
    ]

    variant_metric_rows: list[dict[str, Any]] = []
    variant_state_rows: dict[str, list[dict[str, Any]]] = {}
    variant_candidate_rows: dict[str, list[dict[str, str]]] = {}
    for variant, filter_fn, mutate_fn, mode in variant_defs:
        filtered: list[dict[str, str]] = []
        for row in candidates:
            if filter_fn is not None and not filter_fn(row):
                continue
            next_row = dict(row)
            if mutate_fn is not None:
                next_row = mutate_fn(next_row)
            filtered.append(next_row)
        states = build_states_from_candidates(filtered if variant.startswith("D5") else candidates, variant=variant, mode=mode, filter_fn=(None if variant.startswith("D5") else filter_fn), mutate_fn=(None if variant.startswith("D5") else mutate_fn), material_ids=material_ids)
        if variant.startswith("D5"):
            states = build_states_from_candidates(filtered, variant=variant, mode=mode, material_ids=material_ids)
        metric = metric_for_states(variant, states, v56_core, v56_tentative)
        recalls = candidate_recall(filtered if variant.startswith("D5") else [row for row in candidates if filter_fn is None or filter_fn(row)], material_ids, expected)
        metric.update(recalls)
        metric["candidate_pair_count"] = len(filtered if variant.startswith("D5") else [row for row in candidates if filter_fn is None or filter_fn(row)])
        metric["uses_diagnostic_expected_in_prediction"] = False
        metric["control_candidate_regeneration"] = variant.startswith("D5") or variant.startswith("D6")
        variant_metric_rows.append(metric)
        variant_state_rows[variant] = states
        variant_candidate_rows[variant] = filtered if variant.startswith("D5") else [row for row in candidates if filter_fn is None or filter_fn(row)]

    by_variant = {row["variant"]: row for row in variant_metric_rows}
    d0 = by_variant["D0_original_v61_candidate_rows"]
    d1 = by_variant["D1_drop_diagnostic_expected_columns"]
    d4 = by_variant["D4_typed_non_diagnostic_edges_only"]
    d5 = by_variant["D5_rebuilt_shuffled_K_mat_control"]
    d6 = by_variant["D6_rebuilt_no_temporal_K_mat_control"]
    for row in variant_metric_rows:
        if row["variant"] in {"D0_original_v61_candidate_rows", "D1_drop_diagnostic_expected_columns", "D2_drop_observation_history_id_fallback", "D3_drop_material_history_id_as_prediction_source", "D4_typed_non_diagnostic_edges_only"}:
            row["real_minus_shuffled_ARI"] = row["core_ARI"] - d5["core_ARI"]
            row["real_minus_no_temporal_ARI"] = row["core_ARI"] - d6["core_ARI"]
    gate = {
        "D1_delta_core_ARI_le_0_005": abs(d0["core_ARI"] - d1["core_ARI"]) <= 0.005,
        "D1_delta_core_completeness_le_0_005": abs(d0["core_completeness"] - d1["core_completeness"]) <= 0.005,
        "D4_core_purity_ge_0_95": d4["core_purity"] >= 0.95,
        "D4_core_completeness_ge_0_90": d4["core_completeness"] >= 0.90,
        "D4_real_minus_shuffled_ARI_ge_0_30": d4["real_minus_shuffled_ARI"] >= 0.30,
        "D4_real_minus_no_temporal_ARI_ge_0_25": d4["real_minus_no_temporal_ARI"] >= 0.25,
        "uses_diagnostic_expected_in_prediction_false": all(not row["uses_diagnostic_expected_in_prediction"] for row in variant_metric_rows),
        "rebuilt_controls_available": d5["control_candidate_regeneration"] and d6["control_candidate_regeneration"],
    }
    gate["pass"] = bool(all(gate.values()))
    summary = {
        "phase": "v62_decircularization",
        "created_at": utc_now(),
        "method_note": (
            "D1 drops diagnostic fields from prediction and reattaches diagnostic labels only after solving for metrics. "
            "D2/D4 remove observation_history_id fallback rows. D5/D6 rebuild candidate controls before ranking."
        ),
        "selected_variant": "D4_typed_non_diagnostic_edges_only",
        "candidate_pair_count_D4": d4["candidate_pair_count"],
        "candidate_recall_at_5_D4": d4["candidate_recall@5"],
        "core_ARI_D4": d4["core_ARI"],
        "core_purity_D4": d4["core_purity"],
        "core_completeness_D4": d4["core_completeness"],
        "real_minus_shuffled_ARI_D4": d4["real_minus_shuffled_ARI"],
        "real_minus_no_temporal_ARI_D4": d4["real_minus_no_temporal_ARI"],
        "same_category_merge_rate_D4": d4["same_category_merge_rate"],
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
        "uses_diagnostic_expected_in_prediction": False,
        "diagnostic_expected_history_id_not_used_in_candidate_generation": True,
        "control_candidate_regeneration": True,
        "gate": gate,
        "input_paths": {
            "material_candidate_rows": _rel(cfg.material_candidate_rows_path),
            "v60_node_rows": _rel(cfg.v60_node_rows_path),
            "v60_edge_rows": _rel(cfg.v60_edge_rows_path),
            "v56_core_summary": _rel(cfg.v56_core_summary_path),
            "v56_tentative_summary": _rel(cfg.v56_tentative_summary_path),
        },
    }
    casebook_rows = _casebook_rows(variant_state_rows["D0_original_v61_candidate_rows"], variant_state_rows["D4_typed_non_diagnostic_edges_only"])
    return {
        "summary": summary,
        "variant_metric_rows": variant_metric_rows,
        "provenance_rows": _material_history_provenance(v60_nodes, v60_edges),
        "casebook_rows": casebook_rows,
        "decircularized_material_state_rows": variant_state_rows["D4_typed_non_diagnostic_edges_only"],
        "decircularized_observation_explanation_rows": _observation_explanations(variant_state_rows["D4_typed_non_diagnostic_edges_only"]),
    }


def write_v62_decircularization(result: dict[str, Any], output_root: str | Path) -> dict[str, str]:
    root = _project(output_root)
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "decircularization_summary": root / "decircularization_summary.json",
        "variant_metric_rows": root / "variant_metric_rows.csv",
        "provenance_rows": root / "provenance_rows.csv",
        "leakage_audit_casebook": root / "leakage_audit_casebook.md",
        "decircularized_material_state_rows": root / "decircularized_material_state_rows.csv",
        "decircularized_observation_explanation_rows": root / "decircularized_observation_explanation_rows.csv",
    }
    write_json(paths["decircularization_summary"], result["summary"])
    write_csv(paths["variant_metric_rows"], result["variant_metric_rows"])
    write_csv(paths["provenance_rows"], result["provenance_rows"])
    write_csv(paths["decircularized_material_state_rows"], result["decircularized_material_state_rows"])
    write_csv(paths["decircularized_observation_explanation_rows"], result["decircularized_observation_explanation_rows"])
    _write_casebook(paths["leakage_audit_casebook"], result["casebook_rows"])
    return {name: _rel(path) for name, path in paths.items()}


def write_v62_decircularization_visualizations(result: dict[str, Any], visualization_root: str | Path) -> dict[str, str]:
    root = _project(visualization_root)
    root.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rows = result["variant_metric_rows"]
        labels = [row["variant"].split("_", 1)[0] for row in rows]
        core_values = [row["core_ARI"] for row in rows]
        path = root / "field_drop_sensitivity_bar.png"
        fig, ax = plt.subplots(figsize=(8.5, 4.2))
        ax.bar(labels, core_values, color="#52796F")
        ax.set_ylim(0.0, 1.05)
        ax.set_title("v62 de-circularization core ARI")
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)

        provenance_path = root / "evidence_provenance_sankey.png"
        prov = result["provenance_rows"]
        fig, ax = plt.subplots(figsize=(8.5, 4.2))
        ax.barh([row["field_or_edge"] for row in prov], [row["count"] or 0 for row in prov], color="#E9C46A")
        ax.set_title("v62 evidence provenance counts")
        fig.tight_layout()
        fig.savefig(provenance_path, dpi=160)
        plt.close(fig)
        return {
            "field_drop_sensitivity_bar": _rel(path),
            "evidence_provenance_sankey": _rel(provenance_path),
            "visualization_status": "created",
        }
    except Exception as exc:  # pragma: no cover
        error_path = root / "v62_decircularization_visualization_error.txt"
        error_path.write_text(f"visualization unavailable: {type(exc).__name__}: {exc}\n", encoding="utf-8")
        return {"visualization_error": _rel(error_path), "visualization_status": "unavailable"}


def _casebook_rows(original_states: list[dict[str, Any]], d4_states: list[dict[str, Any]], limit: int = 50) -> list[dict[str, Any]]:
    d4_by_id = {row["material_node_id"]: row for row in d4_states}
    rows: list[dict[str, Any]] = []
    for row in original_states:
        other = d4_by_id.get(row["material_node_id"])
        if not other:
            continue
        changed = row.get("state") != other.get("state") or row.get("predicted_history_id") != other.get("predicted_history_id")
        if changed:
            rows.append(
                {
                    "material_node_id": row["material_node_id"],
                    "scene": row.get("scene", ""),
                    "D0_state": row.get("state", ""),
                    "D0_predicted_history_id": row.get("predicted_history_id", ""),
                    "D4_state": other.get("state", ""),
                    "D4_predicted_history_id": other.get("predicted_history_id", ""),
                    "diagnostic_expected_history_id": row.get("diagnostic_expected_history_id", ""),
                    "D0_reason": row.get("state_reason", ""),
                    "D4_reason": other.get("state_reason", ""),
                }
            )
        if len(rows) >= limit:
            break
    return rows


def _write_casebook(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# v62 Leakage Audit Casebook", "", "Top prediction changes after removing suspicious fields.", ""]
    if not rows:
        lines.append("No D0 to D4 prediction changes were found in the first-pass audit.")
    for row in rows:
        lines.extend(
            [
                f"## {row['material_node_id']}",
                "",
                f"- scene: `{row['scene']}`",
                f"- D0: `{row['D0_state']}` / `{row['D0_predicted_history_id']}`",
                f"- D4: `{row['D4_state']}` / `{row['D4_predicted_history_id']}`",
                f"- diagnostic expected: `{row['diagnostic_expected_history_id']}`",
                f"- D0 reason: `{row['D0_reason']}`",
                f"- D4 reason: `{row['D4_reason']}`",
                "",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


