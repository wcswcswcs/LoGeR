from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .v47_common import ROOT, parse_bool, parse_float, parse_int, utc_now, write_csv, write_json


DEFAULT_SEMANTIC_ROOT = "outputs/audit/v58_semantic_memory_dino_full_repair2"
DEFAULT_EXPLANATION_ROOT = "outputs/audit/v58_counterfactual_explanation_dino_full_repair6"
DEFAULT_SUPPORT_ROWS = "outputs/audit/v54_mask_component_support_tau005_stride1_probe5_q4096_notopup/mask_component_support_rows.csv"
DEFAULT_HISTORY_ROWS = "outputs/audit/v55_history_update/history_rows.csv"
DEFAULT_HISTORY_UPDATE_ROWS = "outputs/audit/v55_history_update/history_update_rows.csv"
DEFAULT_REPROJECTION_ROOT = "outputs/audit/v58_active_query_reprojection_ledger_deferred_max1600_noveto_minvis1"
DEFAULT_NATIVE_ROWS = "outputs/audit/v56_native_field/native_carrier_state_rows.csv"


@dataclass(frozen=True)
class V59GraphBuilderConfig:
    semantic_root: str | Path = DEFAULT_SEMANTIC_ROOT
    explanation_root: str | Path = DEFAULT_EXPLANATION_ROOT
    support_rows_path: str | Path = DEFAULT_SUPPORT_ROWS
    history_rows_path: str | Path = DEFAULT_HISTORY_ROWS
    history_update_rows_path: str | Path = DEFAULT_HISTORY_UPDATE_ROWS
    reprojection_root: str | Path = DEFAULT_REPROJECTION_ROOT
    native_carrier_state_rows_path: str | Path = DEFAULT_NATIVE_ROWS
    output_root: str | Path = "outputs/audit/v59_phase1_graph"
    visualization_root: str | Path = "outputs/audit/v59_visualizations/phase1"
    primary_variant: str = "E6_counterfactual_semantic_material_underseg"
    semantic_top_k_per_observation: int = 3
    material_top_k_per_mask: int = 5
    reprojection_max_edges_per_candidate: int = 3
    semantic_min_score: float = 0.35


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


def _parse_list(value: Any) -> list[Any]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return value
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def _node(node_id: str, node_type: str, **fields: Any) -> dict[str, Any]:
    return {
        "node_id": node_id,
        "node_type": node_type,
        "scene": fields.get("scene", ""),
        "chunk_id": fields.get("chunk_id", ""),
        "frame_id": fields.get("frame_id", ""),
        "mask_id": fields.get("mask_id", ""),
        "history_id": fields.get("history_id", ""),
        "component_id": fields.get("component_id", ""),
        "semantic_mode_id": fields.get("semantic_mode_id", ""),
        "state": fields.get("state", "unknown"),
        "evidence_source": fields.get("evidence_source", ""),
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": bool(fields.get("uses_gt_for_diagnostic_labels", False)),
    }


def _edge(
    edge_id: str,
    edge_type: str,
    src: str,
    dst: str,
    *,
    evidence_source: str,
    confidence: float,
    can_confirm_identity: bool = False,
    can_create_birth: bool = False,
    can_merge_histories: bool = False,
    can_create_quarantine: bool = False,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        "edge_id": edge_id,
        "edge_type": edge_type,
        "src_node_id": src,
        "dst_node_id": dst,
        "evidence_source": evidence_source,
        "confidence": float(confidence),
        "can_confirm_identity": bool(can_confirm_identity),
        "can_create_birth": bool(can_create_birth),
        "can_merge_histories": bool(can_merge_histories),
        "can_create_quarantine": bool(can_create_quarantine),
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": bool((extra or {}).get("uses_gt_for_diagnostic_labels", False)),
    }
    if extra:
        row.update({key: value for key, value in extra.items() if key not in row})
    return row


def build_v59_graph(config: V59GraphBuilderConfig | None = None) -> dict[str, Any]:
    cfg = config or V59GraphBuilderConfig()
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []
    source_counts: dict[str, int] = {}

    history_ids: set[str] = set()
    semantic_histories: set[str] = set()
    material_histories: set[str] = set()
    broad_or_shared_mask_observed = False

    for row in _iter_csv(cfg.history_rows_path):
        source_counts["history_rows"] = source_counts.get("history_rows", 0) + 1
        history_id = str(row.get("history_id") or "")
        if not history_id:
            continue
        history_ids.add(history_id)
        node_id = f"h:{history_id}"
        nodes[node_id] = _node(
            node_id,
            "history_core",
            scene=row.get("scene"),
            history_id=history_id,
            state="confirmed",
            evidence_source="v55_history_rows",
            uses_gt_for_diagnostic_labels=parse_bool(row.get("uses_gt_for_diagnostic_labels")),
        )

    for row in _iter_csv(Path(cfg.semantic_root) / "history_semantic_rows.csv"):
        source_counts["history_semantic_rows"] = source_counts.get("history_semantic_rows", 0) + 1
        history_id = str(row.get("history_id") or "")
        if not history_id:
            continue
        semantic_histories.add(history_id)
        mode_index = str(row.get("mode_index") or "0")
        node_id = f"s:{history_id}:mode{mode_index}"
        nodes[node_id] = _node(
            node_id,
            "semantic_mode",
            scene=row.get("scene"),
            history_id=history_id,
            semantic_mode_id=f"{history_id}:mode{mode_index}",
            state="confirmed",
            evidence_source="v58_history_semantic_rows",
            uses_gt_for_diagnostic_labels=parse_bool(row.get("uses_gt_for_diagnostic_labels")),
        )
        h_node = f"h:{history_id}"
        if h_node in nodes:
            edges.append(
                _edge(
                    f"e{len(edges):08d}",
                    "semantic_compatibility",
                    node_id,
                    h_node,
                    evidence_source="v58_history_semantic_rows",
                    confidence=parse_float(row.get("mode_weight"), 0.0),
                    can_confirm_identity=False,
                    can_merge_histories=False,
                    extra={"semantic_mode_weight": row.get("mode_weight")},
                )
            )

    native_components_by_history: dict[str, set[str]] = defaultdict(set)
    for row in _iter_csv(cfg.native_carrier_state_rows_path):
        source_counts["native_carrier_state_rows"] = source_counts.get("native_carrier_state_rows", 0) + 1
        history_id = str(row.get("history_id") or "")
        component_id = str(row.get("component_id") or "")
        if not history_id or not component_id:
            continue
        state = str(row.get("state") or "unknown")
        if state in {"confirmed", "tentative"}:
            material_histories.add(history_id)
        native_components_by_history[history_id].add(component_id)
        material_node = f"a:{component_id}"
        nodes.setdefault(
            material_node,
            _node(
                material_node,
                "material",
                scene=row.get("scene"),
                history_id=history_id,
                component_id=component_id,
                state=state,
                evidence_source="v56_native_carrier_state_rows",
                uses_gt_for_diagnostic_labels=parse_bool(row.get("uses_gt_for_diagnostic_labels")),
            ),
        )
        h_node = f"h:{history_id}"
        if h_node in nodes:
            edges.append(
                _edge(
                    f"e{len(edges):08d}",
                    "material_continuity",
                    material_node,
                    h_node,
                    evidence_source="v56_native_carrier_state_rows",
                    confidence=1.0 if state == "confirmed" else 0.6,
                    can_confirm_identity=state == "confirmed",
                    can_create_birth=False,
                    can_merge_histories=False,
                    extra={"native_state": state},
                )
            )

    selected_observations, selected_by_obs, semantic_candidates = _load_explanation_edges(cfg)
    for obs_id, row in selected_by_obs.items():
        explanation_type = str(row.get("explanation_type") or "")
        decision_state = str(row.get("decision_state") or "")
        candidate_histories = [str(item) for item in _parse_list(row.get("candidate_history_ids_json"))]
        state = "quarantine" if explanation_type == "underseg_mixture" or len(candidate_histories) > 1 else "tentative"
        if decision_state != "defer_to_active_query" and explanation_type == "assign_to_existing":
            state = "tentative"
        if state == "quarantine":
            broad_or_shared_mask_observed = True
        node_id = f"m:{obs_id}"
        nodes[node_id] = _node(
            node_id,
            "mask_observation",
            scene=row.get("scene"),
            frame_id=row.get("frame_id"),
            mask_id=row.get("mask_id"),
            history_id=row.get("history_id"),
            state=state,
            evidence_source="v58_counterfactual_explanation_rows",
            uses_gt_for_diagnostic_labels=parse_bool(row.get("uses_gt_for_diagnostic_labels")),
        )
        for history_id in candidate_histories:
            if not history_id:
                continue
            h_node = f"h:{history_id}"
            if h_node not in nodes:
                continue
            edge_type = "underseg_bridge" if state == "quarantine" else "mask_support"
            edges.append(
                _edge(
                    f"e{len(edges):08d}",
                    edge_type,
                    node_id,
                    h_node,
                    evidence_source="v58_counterfactual_explanation_rows",
                    confidence=parse_float(row.get("posterior"), 0.0),
                    can_confirm_identity=False,
                    can_create_birth=False,
                    can_merge_histories=False,
                    can_create_quarantine=state == "quarantine",
                    extra={
                        "explanation_type": explanation_type,
                        "decision_state": decision_state,
                        "uses_gt_for_diagnostic_labels": parse_bool(row.get("uses_gt_for_diagnostic_labels")),
                    },
                )
            )

    for obs_id, candidates in semantic_candidates.items():
        for rank, row in enumerate(candidates[: cfg.semantic_top_k_per_observation]):
            score = parse_float(row.get("semantic_score"), 0.0)
            if score < cfg.semantic_min_score:
                continue
            history_id = str(row.get("history_id") or "")
            semantic_node = _first_semantic_node(nodes, history_id)
            mask_node = f"m:{obs_id}"
            if not semantic_node or mask_node not in nodes:
                continue
            edges.append(
                _edge(
                    f"e{len(edges):08d}",
                    "semantic_compatibility",
                    mask_node,
                    semantic_node,
                    evidence_source="v58_counterfactual_explanation_rows",
                    confidence=score,
                    can_confirm_identity=False,
                    can_create_birth=False,
                    can_merge_histories=False,
                    extra={"semantic_rank": rank, "posterior": row.get("posterior")},
                )
            )

    material_support = _top_support_rows(cfg.support_rows_path, selected_observations, cfg.material_top_k_per_mask)
    for obs_id, rows in material_support.items():
        mask_node = f"m:{obs_id}"
        if mask_node not in nodes:
            continue
        for row in rows:
            component_id = str(row.get("component_id") or "")
            if not component_id:
                continue
            material_node = f"a:{component_id}"
            nodes.setdefault(
                material_node,
                _node(
                    material_node,
                    "material",
                    scene=row.get("scene"),
                    frame_id=row.get("frame_id"),
                    mask_id=row.get("mask_id"),
                    component_id=component_id,
                    state="unknown",
                    evidence_source="v54_mask_component_support_rows",
                    uses_gt_for_diagnostic_labels=parse_bool(row.get("uses_gt_for_diagnostic_labels")),
                ),
            )
            edges.append(
                _edge(
                    f"e{len(edges):08d}",
                    "mask_support",
                    mask_node,
                    material_node,
                    evidence_source="v54_mask_component_support_rows",
                    confidence=parse_float(row.get("selection_score"), parse_float(row.get("W_visible"), 0.0)),
                    can_confirm_identity=False,
                    can_create_birth=False,
                    can_merge_histories=False,
                    extra={
                        "support_count": row.get("support_count"),
                        "component_visible_count_in_frame": row.get("component_visible_count_in_frame"),
                    },
                )
            )
        if len(rows) > 1:
            broad_or_shared_mask_observed = True

    _add_reprojection_edges(cfg, selected_observations, nodes, edges)
    _add_history_update_source_edges(cfg, nodes, edges)

    node_rows = list(nodes.values())
    node_count_by_type = dict(Counter(row["node_type"] for row in node_rows))
    edge_count_by_type = dict(Counter(row["edge_type"] for row in edges))
    histories_without_semantic = sorted(history_ids - semantic_histories)
    histories_without_material = sorted(history_ids - material_histories)
    no_d4rt_birth_edge_count = sum(
        1 for row in edges if row["edge_type"] in {"material_continuity", "mask_support"} and row["can_create_birth"]
    )
    semantic_merge_violation_count = sum(
        1 for row in edges if row["edge_type"] == "semantic_compatibility" and row["can_merge_histories"]
    )
    underseg_merge_violation_count = sum(
        1 for row in edges if row["edge_type"] == "underseg_bridge" and row["can_merge_histories"]
    )
    quarantine_confirm_violation_count = sum(
        1 for row in edges if row["edge_type"] in {"underseg_bridge", "exclusion"} and row["can_confirm_identity"]
    )
    required_edge_types = {"semantic_compatibility", "material_continuity", "mask_support", "reprojection", "exclusion"}
    gate = {
        "all_histories_have_semantic_node_and_confirmed_or_tentative_material_node": (
            not histories_without_semantic and not histories_without_material
        ),
        "edge_types_cover_semantic_material_support_reprojection_exclusion": required_edge_types.issubset(edge_count_by_type),
        "no_D4RT_birth_edge_count_eq_0": no_d4rt_birth_edge_count == 0,
        "semantic_edges_do_not_merge_histories": semantic_merge_violation_count == 0,
        "underseg_bridge_edges_do_not_merge_histories": underseg_merge_violation_count == 0,
        "quarantine_edges_do_not_confirm_core": quarantine_confirm_violation_count == 0,
        "underseg_bridge_edges_recorded_when_broad_shared_masks_exist": (
            edge_count_by_type.get("underseg_bridge", 0) > 0 if broad_or_shared_mask_observed else True
        ),
    }
    gate["pass"] = bool(all(gate.values()))
    summary = {
        "phase": "v59_phase1_typed_graph",
        "created_at": utc_now(),
        "primary_variant": cfg.primary_variant,
        "node_count": len(node_rows),
        "edge_count": len(edges),
        "source_row_counts": source_counts,
        "node_count_by_type": node_count_by_type,
        "edge_count_by_type": edge_count_by_type,
        "history_manifold_count": len(history_ids),
        "mean_nodes_per_manifold": _mean_nodes_per_history(node_rows, history_ids),
        "confirmed_node_count": sum(1 for row in node_rows if row.get("state") == "confirmed"),
        "tentative_node_count": sum(1 for row in node_rows if row.get("state") == "tentative"),
        "quarantine_node_count": sum(1 for row in node_rows if row.get("state") == "quarantine"),
        "underseg_bridge_edge_count": edge_count_by_type.get("underseg_bridge", 0),
        "same_frame_cannot_link_edge_count": edge_count_by_type.get("exclusion", 0),
        "material_continuity_edge_count": edge_count_by_type.get("material_continuity", 0),
        "semantic_compatibility_edge_count": edge_count_by_type.get("semantic_compatibility", 0),
        "histories_without_semantic_count": len(histories_without_semantic),
        "histories_without_material_count": len(histories_without_material),
        "histories_without_semantic_examples": histories_without_semantic[:10],
        "histories_without_material_examples": histories_without_material[:10],
        "no_D4RT_birth_edge_count": no_d4rt_birth_edge_count,
        "semantic_merge_violation_count": semantic_merge_violation_count,
        "underseg_merge_violation_count": underseg_merge_violation_count,
        "quarantine_confirm_violation_count": quarantine_confirm_violation_count,
        "broad_or_shared_mask_observed": broad_or_shared_mask_observed,
        "gate": gate,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
        "diagnostic_label_sources": [
            "prior v58/v56 diagnostic fields are carried only for audit/evaluation provenance",
            "graph node/edge construction does not use GT labels for prediction decisions",
        ],
        "input_paths": {
            "history_rows": _rel(cfg.history_rows_path),
            "history_update_rows": _rel(cfg.history_update_rows_path),
            "history_semantic_rows": _rel(Path(cfg.semantic_root) / "history_semantic_rows.csv"),
            "explanation_rows": _rel(Path(cfg.explanation_root) / "explanation_rows.csv"),
            "support_rows": _rel(cfg.support_rows_path),
            "reprojection_candidate_rows": _rel(Path(cfg.reprojection_root) / "candidate_rows.csv"),
            "reprojection_ledger_rows": _rel(Path(cfg.reprojection_root) / "reprojection_ledger_rows.csv"),
            "native_carrier_state_rows": _rel(cfg.native_carrier_state_rows_path),
        },
    }
    invariant_rows = [
        {"invariant": name, "pass": value}
        for name, value in gate.items()
        if name != "pass"
    ]
    return {
        "summary": summary,
        "node_rows": node_rows,
        "edge_rows": edges,
        "graph_invariant_rows": invariant_rows,
    }


def write_v59_graph(result: dict[str, Any], output_root: str | Path) -> dict[str, str]:
    root = _project(output_root)
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "graph_summary": root / "graph_summary.json",
        "node_rows": root / "node_rows.csv",
        "edge_rows": root / "edge_rows.csv",
        "graph_invariant_rows": root / "graph_invariant_rows.csv",
    }
    write_json(paths["graph_summary"], result["summary"])
    write_csv(paths["node_rows"], result["node_rows"])
    write_csv(paths["edge_rows"], result["edge_rows"])
    write_csv(paths["graph_invariant_rows"], result["graph_invariant_rows"])
    return {name: _rel(path) for name, path in paths.items()}


def write_v59_graph_visualizations(result: dict[str, Any], visualization_root: str | Path) -> dict[str, str]:
    root = _project(visualization_root)
    root.mkdir(parents=True, exist_ok=True)
    summary = result["summary"]
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        hist_path = root / "manifold_node_type_histogram.png"
        labels = list(summary["node_count_by_type"].keys())
        counts = [summary["node_count_by_type"][label] for label in labels]
        fig, ax = plt.subplots(figsize=(8.0, 4.2))
        ax.bar(labels, counts, color="#52796F")
        ax.set_title("v59 typed graph node types")
        ax.set_ylabel("node count")
        ax.tick_params(axis="x", labelrotation=20)
        fig.tight_layout()
        fig.savefig(hist_path, dpi=160)
        plt.close(fig)

        edge_path = root / "semantic_material_graph_overview_all.png"
        labels = list(summary["edge_count_by_type"].keys())
        counts = [summary["edge_count_by_type"][label] for label in labels]
        fig, ax = plt.subplots(figsize=(9.0, 4.2))
        ax.bar(labels, counts, color="#6D597A")
        ax.set_title("v59 typed graph edge types")
        ax.set_ylabel("edge count")
        ax.tick_params(axis="x", labelrotation=25)
        fig.tight_layout()
        fig.savefig(edge_path, dpi=160)
        plt.close(fig)
        return {
            "node_type_histogram": _rel(hist_path),
            "graph_overview": _rel(edge_path),
            "visualization_status": "created",
        }
    except Exception as exc:  # pragma: no cover - optional visualization backend
        error_path = root / "v59_phase1_visualization_error.txt"
        error_path.write_text(f"visualization unavailable: {type(exc).__name__}: {exc}\n", encoding="utf-8")
        return {"visualization_error": _rel(error_path), "visualization_status": "unavailable"}


def _load_explanation_edges(
    cfg: V59GraphBuilderConfig,
) -> tuple[set[str], dict[str, dict[str, str]], dict[str, list[dict[str, str]]]]:
    selected_by_obs: dict[str, dict[str, str]] = {}
    semantic_candidates: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in _iter_csv(Path(cfg.explanation_root) / "explanation_rows.csv"):
        if str(row.get("variant") or "") != cfg.primary_variant:
            continue
        obs_id = str(row.get("observation_id") or "")
        if not obs_id:
            continue
        if parse_bool(row.get("is_selected")):
            selected_by_obs[obs_id] = row
        if str(row.get("row_role") or "") == "candidate":
            semantic_candidates[obs_id].append(row)
    selected_observations = set(selected_by_obs)
    for obs_id in list(semantic_candidates):
        semantic_candidates[obs_id].sort(
            key=lambda item: (
                parse_float(item.get("semantic_score"), 0.0),
                parse_float(item.get("posterior"), 0.0),
            ),
            reverse=True,
        )
    return selected_observations, selected_by_obs, semantic_candidates


def _top_support_rows(path: str | Path, observation_ids: set[str], top_k: int) -> dict[str, list[dict[str, str]]]:
    by_obs: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in _iter_csv(path):
        obs_id = str(row.get("mask_observation_id") or "")
        if obs_id not in observation_ids:
            continue
        rows = by_obs[obs_id]
        rows.append(row)
        rows.sort(
            key=lambda item: (
                parse_float(item.get("selection_score"), 0.0),
                parse_float(item.get("support_count"), 0.0),
            ),
            reverse=True,
        )
        if len(rows) > top_k:
            del rows[top_k:]
    return by_obs


def _first_semantic_node(nodes: dict[str, dict[str, Any]], history_id: str) -> str | None:
    prefix = f"s:{history_id}:"
    for node_id in nodes:
        if node_id.startswith(prefix):
            return node_id
    return None


def _add_reprojection_edges(
    cfg: V59GraphBuilderConfig,
    observation_ids: set[str],
    nodes: dict[str, dict[str, Any]],
    edges: list[dict[str, Any]],
) -> None:
    candidate_to_obs: dict[str, str] = {}
    candidate_meta: dict[str, dict[str, str]] = {}
    for row in _iter_csv(Path(cfg.reprojection_root) / "candidate_rows.csv"):
        obs_id = str(row.get("source_mask_observation_id") or "")
        if obs_id not in observation_ids:
            continue
        candidate_id = str(row.get("candidate_id") or "")
        if not candidate_id:
            continue
        candidate_to_obs[candidate_id] = obs_id
        candidate_meta[candidate_id] = row

    edge_count_by_candidate: dict[str, int] = defaultdict(int)
    for row in _iter_csv(Path(cfg.reprojection_root) / "reprojection_ledger_rows.csv"):
        candidate_id = str(row.get("candidate_id") or "")
        source_obs = candidate_to_obs.get(candidate_id)
        if not source_obs:
            continue
        if edge_count_by_candidate[candidate_id] >= cfg.reprojection_max_edges_per_candidate:
            continue
        source_node = f"m:{source_obs}"
        target_obs = str(row.get("best_mask_observation_id") or "")
        if not target_obs:
            continue
        target_node = f"m:{target_obs}"
        meta = candidate_meta.get(candidate_id, {})
        nodes.setdefault(
            target_node,
            _node(
                target_node,
                "mask_observation",
                scene=row.get("scene"),
                frame_id=row.get("target_frame_id"),
                mask_id=target_obs.rsplit(":", 1)[-1],
                state="unknown",
                evidence_source="v58_reprojection_ledger_rows",
                uses_gt_for_diagnostic_labels=parse_bool(row.get("uses_gt_for_diagnostic_labels")),
            ),
        )
        success = parse_bool(row.get("reprojection_success"))
        exclusion = parse_bool(row.get("same_frame_exclusion_violation")) or parse_float(
            row.get("outside_all_related_masks_ratio"), 0.0
        ) > 0.35
        edge_type = "exclusion" if exclusion else "reprojection"
        edges.append(
            _edge(
                f"e{len(edges):08d}",
                edge_type,
                source_node,
                target_node,
                evidence_source="v58_reprojection_ledger_rows",
                confidence=parse_float(row.get("inside_best_mask_ratio"), 0.0) if success else 0.0,
                can_confirm_identity=False,
                can_create_birth=False,
                can_merge_histories=False,
                can_create_quarantine=exclusion,
                extra={
                    "candidate_id": candidate_id,
                    "candidate_source": meta.get("candidate_source"),
                    "target_frame_id": row.get("target_frame_id"),
                    "same_frame_exclusion_violation": row.get("same_frame_exclusion_violation"),
                    "outside_all_related_masks_ratio": row.get("outside_all_related_masks_ratio"),
                    "uses_gt_for_diagnostic_labels": parse_bool(row.get("uses_gt_for_diagnostic_labels")),
                },
            )
        )
        edge_count_by_candidate[candidate_id] += 1


def _add_history_update_source_edges(
    cfg: V59GraphBuilderConfig,
    nodes: dict[str, dict[str, Any]],
    edges: list[dict[str, Any]],
) -> None:
    for row in _iter_csv(cfg.history_update_rows_path):
        history_id = str(row.get("history_id") or "")
        candidate_id = str(row.get("candidate_id") or "")
        h_node = f"h:{history_id}"
        m_node = f"m:{candidate_id}"
        if h_node not in nodes or m_node not in nodes:
            continue
        update_state = str(row.get("update_state") or "")
        edges.append(
            _edge(
                f"e{len(edges):08d}",
                "mask_support",
                m_node,
                h_node,
                evidence_source="v55_history_update_rows",
                confidence=parse_float(row.get("overlap_atom_ratio"), 0.0),
                can_confirm_identity=update_state == "confirmed_update",
                can_create_birth=False,
                can_merge_histories=False,
                extra={"update_state": update_state},
            )
        )


def _mean_nodes_per_history(node_rows: list[dict[str, Any]], history_ids: set[str]) -> float | None:
    if not history_ids:
        return None
    counts = Counter(row.get("history_id") for row in node_rows if row.get("history_id"))
    return float(sum(counts.get(history_id, 0) for history_id in history_ids) / len(history_ids))
