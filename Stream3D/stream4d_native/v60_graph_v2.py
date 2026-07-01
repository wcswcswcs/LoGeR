from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .v47_common import ROOT, parse_bool, parse_float, read_json, safe_mean, utc_now, write_csv, write_json


DEFAULT_V59_GRAPH_ROOT = "outputs/audit/v59_phase1_graph"


@dataclass(frozen=True)
class V60GraphV2Config:
    v59_graph_root: str | Path = DEFAULT_V59_GRAPH_ROOT
    output_root: str | Path = "outputs/audit/v60_graph_v2"
    visualization_root: str | Path = "outputs/audit/v60_visualizations/graph_v2"
    epsilon: float = 1.0e-6


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


def build_v60_graph_v2(config: V60GraphV2Config | None = None) -> dict[str, Any]:
    cfg = config or V60GraphV2Config()
    graph_root = _project(cfg.v59_graph_root)
    v59_summary = read_json(graph_root / "graph_summary.json")
    node_rows = list(_iter_csv(graph_root / "node_rows.csv"))
    raw_edges = list(_iter_csv(graph_root / "edge_rows.csv"))

    edge_rows: list[dict[str, Any]] = []
    cost_rows: list[dict[str, Any]] = []
    violation_count = 0
    cost_by_type: dict[str, list[float]] = defaultdict(list)
    for edge in raw_edges:
        enriched = _enrich_edge(edge, epsilon=cfg.epsilon)
        edge_rows.append(enriched)
        if enriched["edge_cost"] is not None:
            cost = float(enriched["edge_cost"])
            cost_by_type[str(enriched["edge_type"])].append(cost)
            cost_rows.append(
                {
                    "edge_id": enriched["edge_id"],
                    "edge_type": enriched["edge_type"],
                    "edge_cost": cost,
                    "edge_reliability": enriched["edge_reliability"],
                    "edge_role": enriched["edge_role"],
                    "hard_constraint_type": enriched["hard_constraint_type"],
                }
            )
        if bool(enriched["hard_constraint_violation"]):
            violation_count += 1

    edge_count_by_type = dict(Counter(row["edge_type"] for row in edge_rows))
    node_count_by_type = dict(Counter(row["node_type"] for row in node_rows))
    cost_distribution = {
        edge_type: {
            "count": len(values),
            "mean": safe_mean(values),
            "min": min(values) if values else None,
            "max": max(values) if values else None,
        }
        for edge_type, values in sorted(cost_by_type.items())
    }
    required_cost_types = ["semantic_compatibility", "material_continuity", "mask_support", "reprojection"]
    nonempty_rates = {
        edge_type: (
            len(cost_by_type.get(edge_type, [])) / edge_count_by_type.get(edge_type, 1)
            if edge_count_by_type.get(edge_type, 0)
            else 0.0
        )
        for edge_type in required_cost_types
    }
    v59_history_count = int(v59_summary.get("history_manifold_count") or 0)
    history_ids = {row.get("history_id") for row in node_rows if row.get("node_type") == "history_core" and row.get("history_id")}
    history_coverage = len(history_ids) / v59_history_count if v59_history_count else None
    gate = {
        "v59_graph_invariants_still_pass": bool((v59_summary.get("gate") or {}).get("pass")),
        "edge_cost_nonempty_rate_ge_0_95_for_required_types": all(rate >= 0.95 for rate in nonempty_rates.values()),
        "shortcut_edge_count_gt_0": edge_count_by_type.get("underseg_bridge", 0) > 0,
        "hard_constraint_violation_count_eq_0": violation_count == 0,
        "history_coverage_ge_v59": history_coverage is not None and history_coverage >= 1.0,
    }
    gate["pass"] = bool(all(gate.values()))
    summary = {
        "phase": "v60_graph_v2",
        "created_at": utc_now(),
        "v59_graph_root": _rel(graph_root),
        "node_count": len(node_rows),
        "edge_count": len(edge_rows),
        "node_count_by_type": node_count_by_type,
        "edge_count_by_type": edge_count_by_type,
        "edge_cost_distribution_by_type": cost_distribution,
        "edge_cost_nonempty_rate_by_type": nonempty_rates,
        "history_coverage": history_coverage,
        "mask_observation_coverage": node_count_by_type.get("mask_observation", 0) / max(node_count_by_type.get("mask_observation", 0), 1),
        "material_node_coverage": node_count_by_type.get("material", 0) / max(node_count_by_type.get("material", 0), 1),
        "shortcut_edge_count": edge_count_by_type.get("underseg_bridge", 0),
        "cannot_link_edge_count": edge_count_by_type.get("exclusion", 0),
        "hard_constraint_violation_count": violation_count,
        "material_no_birth_invariant": all(
            not parse_bool(row.get("can_create_birth")) for row in edge_rows if row["edge_type"] in {"material_continuity", "mask_support"}
        ),
        "semantic_no_merge_invariant": all(
            not parse_bool(row.get("can_merge_histories")) for row in edge_rows if row["edge_type"] == "semantic_compatibility"
        ),
        "underseg_no_merge_invariant": all(
            not parse_bool(row.get("can_merge_histories")) for row in edge_rows if row["edge_type"] == "underseg_bridge"
        ),
        "gate": gate,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
        "input_paths": {
            "v59_graph_summary": _rel(graph_root / "graph_summary.json"),
            "v59_node_rows": _rel(graph_root / "node_rows.csv"),
            "v59_edge_rows": _rel(graph_root / "edge_rows.csv"),
        },
    }
    return {
        "summary": summary,
        "node_rows": node_rows,
        "edge_rows": edge_rows,
        "edge_cost_rows": cost_rows,
    }


def write_v60_graph_v2(result: dict[str, Any], output_root: str | Path) -> dict[str, str]:
    root = _project(output_root)
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "graph_summary": root / "graph_summary.json",
        "node_rows": root / "node_rows.csv",
        "edge_rows": root / "edge_rows.csv",
        "edge_cost_rows": root / "edge_cost_rows.csv",
    }
    write_json(paths["graph_summary"], result["summary"])
    write_csv(paths["node_rows"], result["node_rows"])
    write_csv(paths["edge_rows"], result["edge_rows"])
    write_csv(paths["edge_cost_rows"], result["edge_cost_rows"])
    return {name: _rel(path) for name, path in paths.items()}


def write_v60_graph_v2_visualizations(result: dict[str, Any], visualization_root: str | Path) -> dict[str, str]:
    root = _project(visualization_root)
    root.mkdir(parents=True, exist_ok=True)
    summary = result["summary"]
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        overview = root / "semantic_material_graph_v2_overview.png"
        labels = list(summary["edge_count_by_type"].keys())
        counts = [summary["edge_count_by_type"][label] for label in labels]
        fig, ax = plt.subplots(figsize=(9.0, 4.2))
        ax.bar(labels, counts, color="#52796F")
        ax.set_title("v60 graph v2 edge types")
        ax.tick_params(axis="x", labelrotation=25)
        fig.tight_layout()
        fig.savefig(overview, dpi=160)
        plt.close(fig)

        costs = root / "edge_cost_histograms_by_type.png"
        fig, ax = plt.subplots(figsize=(9.0, 4.2))
        for edge_type, dist in summary["edge_cost_distribution_by_type"].items():
            mean = dist.get("mean")
            if mean is not None:
                ax.bar(edge_type, mean)
        ax.set_title("mean edge cost by type")
        ax.tick_params(axis="x", labelrotation=25)
        fig.tight_layout()
        fig.savefig(costs, dpi=160)
        plt.close(fig)
        return {
            "graph_overview": _rel(overview),
            "edge_cost_histograms": _rel(costs),
            "visualization_status": "created",
        }
    except Exception as exc:  # pragma: no cover
        error_path = root / "v60_graph_v2_visualization_error.txt"
        error_path.write_text(f"visualization unavailable: {type(exc).__name__}: {exc}\n", encoding="utf-8")
        return {"visualization_error": _rel(error_path), "visualization_status": "unavailable"}


def _enrich_edge(edge: dict[str, str], *, epsilon: float) -> dict[str, Any]:
    edge_type = str(edge.get("edge_type") or "")
    confidence = max(0.0, min(1.0, parse_float(edge.get("confidence"), 0.0)))
    role, hard_constraint = _edge_role(edge_type)
    cost = _edge_cost(edge_type, confidence, edge, epsilon)
    reliability = math.exp(-cost) if cost is not None and cost < 100.0 else 0.0
    can_enter_core = edge_type == "material_continuity" and parse_bool(edge.get("can_confirm_identity"))
    can_enter_tentative = edge_type in {"semantic_compatibility", "mask_support", "reprojection", "material_continuity"}
    can_enter_quarantine = edge_type in {"underseg_bridge", "exclusion"}
    violation = (
        (edge_type in {"material_continuity", "mask_support"} and parse_bool(edge.get("can_create_birth")))
        or (edge_type == "semantic_compatibility" and parse_bool(edge.get("can_merge_histories")))
        or (edge_type == "underseg_bridge" and parse_bool(edge.get("can_merge_histories")))
        or (edge_type in {"underseg_bridge", "exclusion"} and parse_bool(edge.get("can_confirm_identity")))
    )
    return {
        **edge,
        "edge_cost": cost,
        "edge_reliability": reliability,
        "edge_direction": "directed",
        "edge_role": role,
        "hard_constraint_type": hard_constraint,
        "can_enter_core": can_enter_core,
        "can_enter_tentative": can_enter_tentative,
        "can_enter_quarantine": can_enter_quarantine,
        "can_bridge_manifold": edge_type == "underseg_bridge",
        "can_only_explain_shared_observation": edge_type in {"underseg_bridge", "exclusion"},
        "hard_constraint_violation": violation,
        "uses_gt_for_prediction": False,
    }


def _edge_cost(edge_type: str, confidence: float, edge: dict[str, str], epsilon: float) -> float | None:
    if edge_type in {"semantic_compatibility", "material_continuity", "mask_support"}:
        return float(-math.log(epsilon + max(confidence, 0.0)))
    if edge_type == "reprojection":
        outside = parse_float(edge.get("outside_all_related_masks_ratio"), 0.0)
        conflict = 1.0 if parse_bool(edge.get("same_frame_exclusion_violation")) else 0.0
        inside = confidence
        return float(max(0.0, 1.5 * outside + 2.0 * conflict - inside))
    if edge_type == "underseg_bridge":
        return 50.0
    if edge_type == "exclusion":
        return 100.0
    return None


def _edge_role(edge_type: str) -> tuple[str, str]:
    if edge_type == "semantic_compatibility":
        return "semantic_support", "none"
    if edge_type == "material_continuity":
        return "material_continuity", "no_birth"
    if edge_type == "mask_support":
        return "mask_material_support", "no_birth"
    if edge_type == "reprojection":
        return "view_consistency", "soft_visibility"
    if edge_type == "underseg_bridge":
        return "shortcut_candidate", "quarantine_or_shared"
    if edge_type == "exclusion":
        return "cannot_link", "hard_veto"
    return "unknown", "unknown"
