from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .v47_common import ROOT, read_json, utc_now, write_csv, write_json


DEFAULT_V60_FINAL = "outputs/audit/v60_final_decision/final_decision.json"
DEFAULT_V60_GRAPH = "outputs/audit/v60_graph_v2/graph_summary.json"
DEFAULT_V60_GRAPH_NODES = "outputs/audit/v60_graph_v2/node_rows.csv"
DEFAULT_V60_GRAPH_EDGES = "outputs/audit/v60_graph_v2/edge_rows.csv"
DEFAULT_V60_PATH = "outputs/audit/v60_manifold_paths_v2/path_summary.json"
DEFAULT_V60_EMBEDDING = "outputs/audit/v60_manifold_embedding/embedding_summary.json"
DEFAULT_V60_NODE_STATES = "outputs/audit/v60_manifold_embedding/node_state_rows.csv"
DEFAULT_V60_REFINEMENT = "outputs/audit/v60_manifold_refinement/refinement_summary.json"
DEFAULT_V60_QUERY = "outputs/audit/v60_manifold_query/query_summary.json"


@dataclass(frozen=True)
class V61FailureLockConfig:
    v60_final_path: str | Path = DEFAULT_V60_FINAL
    v60_graph_summary_path: str | Path = DEFAULT_V60_GRAPH
    v60_graph_node_rows_path: str | Path = DEFAULT_V60_GRAPH_NODES
    v60_graph_edge_rows_path: str | Path = DEFAULT_V60_GRAPH_EDGES
    v60_path_summary_path: str | Path = DEFAULT_V60_PATH
    v60_embedding_summary_path: str | Path = DEFAULT_V60_EMBEDDING
    v60_node_state_rows_path: str | Path = DEFAULT_V60_NODE_STATES
    v60_refinement_summary_path: str | Path = DEFAULT_V60_REFINEMENT
    v60_query_summary_path: str | Path = DEFAULT_V60_QUERY


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


def build_v61_failure_lock(config: V61FailureLockConfig | None = None) -> dict[str, Any]:
    cfg = config or V61FailureLockConfig()
    final = read_json(_project(cfg.v60_final_path))
    graph = read_json(_project(cfg.v60_graph_summary_path))
    path = read_json(_project(cfg.v60_path_summary_path))
    embedding = read_json(_project(cfg.v60_embedding_summary_path))
    refinement = read_json(_project(cfg.v60_refinement_summary_path))
    query = read_json(_project(cfg.v60_query_summary_path))

    node_rows = list(_iter_csv(cfg.v60_graph_node_rows_path))
    state_rows = list(_iter_csv(cfg.v60_node_state_rows_path))
    edge_rows = list(_iter_csv(cfg.v60_graph_edge_rows_path))
    node_count_by_type = Counter(row.get("node_type", "") for row in node_rows)
    edge_count_by_type = Counter(row.get("edge_type", "") for row in edge_rows)
    material_node_ids = {row["node_id"] for row in node_rows if row.get("node_type") == "material"}
    observation_node_ids = {row["node_id"] for row in node_rows if row.get("node_type") == "mask_observation"}
    state_ids = _state_row_ids(state_rows)
    material_nodes_with_state = material_node_ids & state_ids
    observation_nodes_with_state = observation_node_ids & state_ids
    material_node_count = int(node_count_by_type.get("material", 0))
    observation_node_count = int(node_count_by_type.get("mask_observation", 0))
    material_state_coverage_rate = _safe_div(len(material_nodes_with_state), material_node_count)
    observation_state_coverage_rate = _safe_div(len(observation_nodes_with_state), observation_node_count)

    unit_rows = [
        {
            "node_type": node_type,
            "node_count": int(node_count_by_type.get(node_type, 0)),
            "state_count": {
                "material": len(material_nodes_with_state),
                "mask_observation": len(observation_nodes_with_state),
            }.get(node_type, ""),
            "state_coverage_rate": {
                "material": material_state_coverage_rate,
                "mask_observation": observation_state_coverage_rate,
            }.get(node_type, ""),
        }
        for node_type in ("history_core", "semantic_mode", "mask_observation", "material")
    ]
    unit_rows.append(
        {
            "node_type": "v60_node_state_rows",
            "node_count": len(state_rows),
            "state_count": len(state_rows),
            "state_coverage_rate": 1.0 if state_rows else 0.0,
        }
    )

    gate = {
        "v60_final_label_is_no_go_embedding": final.get("final_label") == "NO_GO_EMBEDDING",
        "v60_graph_gate_pass": bool((graph.get("gate") or {}).get("pass")),
        "v60_path_gate_pass": bool((path.get("gate") or {}).get("pass")),
        "v60_embedding_gate_fail": not bool((embedding.get("gate") or {}).get("pass")),
        "material_state_coverage_rate_lt_0_10": material_state_coverage_rate < 0.10,
    }
    gate["pass"] = bool(all(gate.values()))

    summary = {
        "phase": "v61_phase0_failure_lock",
        "created_at": utc_now(),
        "v60_final_label": final.get("final_label"),
        "v60_partial_label": final.get("partial_label"),
        "v60_graph_node_count_by_type": dict(node_count_by_type),
        "v60_edge_count_by_type": dict(edge_count_by_type),
        "v60_accepted_path_count": path.get("accepted_path_count"),
        "v60_path_precision": path.get("path_precision_diagnostic"),
        "v60_shortcut_quarantine_precision": path.get("shortcut_quarantine_precision"),
        "v60_embedding_confirmed_node_count": embedding.get("confirmed_node_count"),
        "v60_embedding_tentative_node_count": embedding.get("tentative_node_count"),
        "v60_embedding_quarantine_node_count": embedding.get("quarantine_node_count"),
        "v60_embedding_unknown_node_count": embedding.get("unknown_node_count"),
        "v60_embedding_core_purity": embedding.get("core_purity"),
        "v60_embedding_core_completeness": embedding.get("core_completeness"),
        "v60_embedding_expanded_completeness": embedding.get("expanded_completeness"),
        "v60_material_node_count": material_node_count,
        "v60_observation_node_count": observation_node_count,
        "material_nodes_with_state_count": len(material_nodes_with_state),
        "material_state_coverage_rate": material_state_coverage_rate,
        "observation_nodes_with_state_count": len(observation_nodes_with_state),
        "observation_state_coverage_rate": observation_state_coverage_rate,
        "v60_refinement_gate_pass": bool((refinement.get("gate") or {}).get("pass")),
        "v60_query_gate_pass": bool((query.get("gate") or {}).get("pass")),
        "gate": gate,
        "conclusion": (
            "v60 Phase3 labels observation rows but not material graph nodes; v61 may proceed to material ownership candidates."
            if gate["pass"]
            else "v60 unit-mismatch lock did not pass; inspect missing artifacts or schema before Phase1."
        ),
        "input_paths": {
            "v60_final": _rel(cfg.v60_final_path),
            "v60_graph_summary": _rel(cfg.v60_graph_summary_path),
            "v60_graph_node_rows": _rel(cfg.v60_graph_node_rows_path),
            "v60_graph_edge_rows": _rel(cfg.v60_graph_edge_rows_path),
            "v60_path_summary": _rel(cfg.v60_path_summary_path),
            "v60_embedding_summary": _rel(cfg.v60_embedding_summary_path),
            "v60_node_state_rows": _rel(cfg.v60_node_state_rows_path),
            "v60_refinement_summary": _rel(cfg.v60_refinement_summary_path),
            "v60_query_summary": _rel(cfg.v60_query_summary_path),
        },
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }
    return {"summary": summary, "unit_mismatch_rows": unit_rows}


def write_v61_failure_lock(result: dict[str, Any], output_root: str | Path) -> dict[str, str]:
    root = _project(output_root)
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "failure_lock": root / "failure_lock.json",
        "unit_mismatch_rows": root / "unit_mismatch_rows.csv",
    }
    write_json(paths["failure_lock"], result["summary"])
    write_csv(paths["unit_mismatch_rows"], result["unit_mismatch_rows"])
    return {name: _rel(path) for name, path in paths.items()}


def write_v61_phase0_visualizations(result: dict[str, Any], visualization_root: str | Path) -> dict[str, str]:
    root = _project(visualization_root)
    root.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rows = result["unit_mismatch_rows"]
        node_rows = [row for row in rows if row["node_type"] in {"mask_observation", "material"}]
        dashboard = root / "v61_phase0_unit_mismatch_dashboard.png"
        labels = [row["node_type"] for row in node_rows]
        counts = [float(row["node_count"]) for row in node_rows]
        states = [float(row["state_count"] or 0.0) for row in node_rows]
        fig, ax = plt.subplots(figsize=(7.2, 4.0))
        x = range(len(labels))
        ax.bar([i - 0.18 for i in x], counts, width=0.36, label="graph nodes", color="#457B9D")
        ax.bar([i + 0.18 for i in x], states, width=0.36, label="nodes with v60 state", color="#E76F51")
        ax.set_xticks(list(x), labels)
        ax.set_yscale("log")
        ax.legend()
        ax.set_title("v61 Phase0 unit mismatch")
        fig.tight_layout()
        fig.savefig(dashboard, dpi=160)
        plt.close(fig)

        coverage_path = root / "v60_embedding_state_by_node_type.png"
        fig, ax = plt.subplots(figsize=(7.2, 4.0))
        ax.bar(labels, [float(row["state_coverage_rate"] or 0.0) for row in node_rows], color=["#2A9D8F", "#B56576"])
        ax.axhline(0.10, color="#333333", linestyle="--", linewidth=1.0)
        ax.set_ylim(0.0, 1.05)
        ax.set_title("v60 state coverage by node type")
        fig.tight_layout()
        fig.savefig(coverage_path, dpi=160)
        plt.close(fig)
        return {
            "unit_mismatch_dashboard": _rel(dashboard),
            "state_by_node_type": _rel(coverage_path),
            "visualization_status": "created",
        }
    except Exception as exc:  # pragma: no cover
        error_path = root / "v61_phase0_visualization_error.txt"
        error_path.write_text(f"visualization unavailable: {type(exc).__name__}: {exc}\n", encoding="utf-8")
        return {"visualization_error": _rel(error_path), "visualization_status": "unavailable"}


def _state_row_ids(rows: list[dict[str, str]]) -> set[str]:
    ids: set[str] = set()
    for row in rows:
        observation_id = row.get("observation_id")
        if observation_id:
            ids.add(f"m:{observation_id}")
        node_id = row.get("node_id")
        if node_id:
            ids.add(node_id)
        component_id = row.get("component_id")
        if component_id:
            ids.add(f"a:{component_id}")
    return ids


def _safe_div(num: int, denom: int) -> float:
    return 0.0 if denom == 0 else float(num) / float(denom)
