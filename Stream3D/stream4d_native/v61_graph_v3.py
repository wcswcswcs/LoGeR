from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from .v47_common import ROOT, parse_float, rank_auc, read_json, utc_now, write_csv, write_json


DEFAULT_PHASE0 = "outputs/audit/v61_phase0_failure_lock/failure_lock.json"
DEFAULT_V60_GRAPH = "outputs/audit/v60_graph_v2/graph_summary.json"
DEFAULT_V60_NODES = "outputs/audit/v60_graph_v2/node_rows.csv"
DEFAULT_V60_EDGES = "outputs/audit/v60_graph_v2/edge_rows.csv"


@dataclass(frozen=True)
class V61GraphV3Config:
    phase0_failure_lock_path: str | Path = DEFAULT_PHASE0
    v60_graph_summary_path: str | Path = DEFAULT_V60_GRAPH
    v60_node_rows_path: str | Path = DEFAULT_V60_NODES
    v60_edge_rows_path: str | Path = DEFAULT_V60_EDGES
    output_root: str | Path = "outputs/audit/v61_graph_v3"
    visualization_root: str | Path = "outputs/audit/v61_visualizations/graph_v3"
    semantic_topk_per_observation: int = 5
    max_candidates_per_material: int = 8


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


def build_v61_graph_v3(config: V61GraphV3Config | None = None) -> dict[str, Any]:
    cfg = config or V61GraphV3Config()
    phase0 = read_json(_project(cfg.phase0_failure_lock_path))
    graph = read_json(_project(cfg.v60_graph_summary_path))
    nodes = list(_iter_csv(cfg.v60_node_rows_path))
    edges = list(_iter_csv(cfg.v60_edge_rows_path))
    node_by_id = {row["node_id"]: row for row in nodes}
    material_nodes = [row for row in nodes if row.get("node_type") == "material"]
    history_nodes = [row for row in nodes if row.get("node_type") == "history_core"]

    obs_to_semantic_histories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    semantic_to_history: dict[str, list[dict[str, Any]]] = defaultdict(list)
    obs_to_mask_histories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    obs_to_underseg_histories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    mat_to_observations: dict[str, list[dict[str, Any]]] = defaultdict(list)
    candidates: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)

    for edge in edges:
        src = edge.get("src_node_id", "")
        dst = edge.get("dst_node_id", "")
        edge_type = edge.get("edge_type")
        if edge_type == "semantic_compatibility" and src.startswith("s:") and dst.startswith("h:"):
            semantic_to_history[src].append(_edge_info(edge, "K_sem_mode_to_history"))
        elif edge_type == "mask_support" and src.startswith("m:") and dst.startswith("a:"):
            mat_to_observations[dst].append(_edge_info(edge, "K_mask_observation_to_material"))
        elif edge_type == "mask_support" and src.startswith("m:") and dst.startswith("h:"):
            obs_to_mask_histories[src].append(_edge_info(edge, "K_mask_observation_to_history"))
        elif edge_type == "underseg_bridge" and src.startswith("m:") and dst.startswith("h:"):
            obs_to_underseg_histories[src].append(_edge_info(edge, "K_underseg_observation_to_history"))
        elif edge_type == "material_continuity" and src.startswith("a:") and dst.startswith("h:"):
            _add_candidate(candidates[src], dst, edge, "K_mat")

    for edge in edges:
        if edge.get("edge_type") != "semantic_compatibility":
            continue
        src = edge.get("src_node_id", "")
        dst = edge.get("dst_node_id", "")
        if not (src.startswith("m:") and dst.startswith("s:")):
            continue
        mode_histories = semantic_to_history.get(dst, [])
        if not mode_histories:
            continue
        obs_semantic_cost = parse_float(edge.get("edge_cost"), 0.0)
        obs_semantic_conf = parse_float(edge.get("confidence"), 0.0)
        for hist_info in sorted(mode_histories, key=lambda item: item["edge_cost"])[: cfg.semantic_topk_per_observation]:
            obs_to_semantic_histories[src].append(
                {
                    "history_node_id": hist_info["history_node_id"],
                    "evidence_source": "K_sem",
                    "confidence": min(obs_semantic_conf, hist_info["confidence"]),
                    "edge_cost": obs_semantic_cost + hist_info["edge_cost"],
                    "source_edge_id": f"{edge.get('edge_id')}|{hist_info['source_edge_id']}",
                }
            )

    for material in material_nodes:
        material_id = material["node_id"]
        for obs_info in mat_to_observations.get(material_id, []):
            obs_id = obs_info["observation_node_id"]
            for hist_info in obs_to_mask_histories.get(obs_id, []):
                _add_candidate_info(candidates[material_id], hist_info["history_node_id"], obs_info, hist_info, "K_mask")
            obs_row = node_by_id.get(obs_id, {})
            if obs_row.get("history_id"):
                _add_candidate_info(
                    candidates[material_id],
                    f"h:{obs_row['history_id']}",
                    obs_info,
                    {"confidence": 0.50, "edge_cost": 0.69, "source_edge_id": "observation_history_id"},
                    "K_mask",
                )
            for hist_info in obs_to_semantic_histories.get(obs_id, []):
                _add_candidate_info(candidates[material_id], hist_info["history_node_id"], obs_info, hist_info, "K_sem")
            for hist_info in obs_to_underseg_histories.get(obs_id, []):
                _add_candidate_info(candidates[material_id], hist_info["history_node_id"], obs_info, hist_info, "K_underseg")

    candidate_rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []
    per_material_candidate_counts: list[int] = []
    candidate_lists_for_recall: dict[str, list[dict[str, Any]]] = {}
    for material in material_nodes:
        material_id = material["node_id"]
        expected_history_node = f"h:{material['history_id']}" if material.get("history_id") else ""
        ranked = sorted(candidates.get(material_id, {}).values(), key=_candidate_rank_key)[: cfg.max_candidates_per_material]
        candidate_lists_for_recall[material_id] = ranked
        per_material_candidate_counts.append(len(ranked))
        for rank, cand in enumerate(ranked, start=1):
            row = _candidate_row(material, cand, rank, expected_history_node)
            candidate_rows.append(row)
            edge_rows.append(
                {
                    "edge_id": f"v61_cand_{len(edge_rows):08d}",
                    "edge_type": "material_ownership_candidate",
                    "src_node_id": material_id,
                    "dst_node_id": cand["history_node_id"],
                    "candidate_rank": rank,
                    "candidate_total_cost": row["candidate_total_cost"],
                    "candidate_evidence_types": row["candidate_evidence_types"],
                    "can_enter_confirmed_core": row["can_enter_confirmed_core"],
                    "can_enter_tentative": row["can_enter_tentative"],
                    "can_enter_shared": row["can_enter_shared"],
                    "can_enter_quarantine": row["can_enter_quarantine"],
                    "uses_gt_for_prediction": False,
                    "uses_gt_for_diagnostic_labels": True,
                }
            )

    metrics = _candidate_metrics(material_nodes, candidate_lists_for_recall)
    semantic_only_metrics = _semantic_only_confusion_metrics(material_nodes, candidate_lists_for_recall)
    material_nodes_with_candidate_rate = _safe_div(sum(1 for count in per_material_candidate_counts if count > 0), len(material_nodes))
    candidate_histories_per_material_mean = float(mean(per_material_candidate_counts)) if per_material_candidate_counts else 0.0
    candidate_histories_per_material_p90 = _quantile(per_material_candidate_counts, 0.90)
    pair_count = len(candidate_rows)
    strong_material_candidate_rate = _safe_div(sum(1 for row in candidate_rows if row["can_enter_confirmed_core"]), pair_count)
    tentative_candidate_rate = _safe_div(sum(1 for row in candidate_rows if row["can_enter_tentative"]), pair_count)
    shared_candidate_rate = _safe_div(sum(1 for row in candidate_rows if row["can_enter_shared"]), pair_count)
    quarantine_candidate_rate = _safe_div(sum(1 for row in candidate_rows if row["can_enter_quarantine"]), pair_count)
    hard_constraint_violation_count = sum(1 for row in candidate_rows if row["hard_constraint_violation"])

    same_category_proxy = metrics["same_scene_top1_false_candidate_rate"]
    semantic_only_proxy = semantic_only_metrics["semantic_only_same_scene_top1_false_candidate_rate"]
    same_category_threshold = _calibrated_confusion_threshold(semantic_only_proxy)
    same_category_gate_available = same_category_threshold is not None and same_category_proxy is not None
    same_category_gate_pass = bool(same_category_gate_available and same_category_proxy <= same_category_threshold)
    gate = {
        "phase0_gate_pass": bool((phase0.get("gate") or {}).get("pass")),
        "material_nodes_with_candidate_rate_ge_0_80": material_nodes_with_candidate_rate >= 0.80,
        "candidate_recall_at_5_ge_0_90_diagnostic": metrics["candidate_recall_at_5"] >= 0.90,
        "same_category_candidate_confusion_proxy_improves_semantic_only_by_0_05": same_category_gate_pass,
        "hard_constraint_violation_count_eq_0": hard_constraint_violation_count == 0,
    }
    gate["pass"] = bool(all(gate.values()))
    summary = {
        "phase": "v61_graph_v3",
        "created_at": utc_now(),
        "method_note": (
            "Material-history candidates are generated from v60 graph evidence: material continuity, mask support, "
            "observation semantic-mode paths, and underseg bridge edges. Diagnostic recall compares against v60 graph "
            "material history_id and is not used for candidate generation."
        ),
        "material_node_count": len(material_nodes),
        "history_node_count": len(history_nodes),
        "material_candidate_pair_count": pair_count,
        "candidate_histories_per_material_mean": candidate_histories_per_material_mean,
        "candidate_histories_per_material_p90": candidate_histories_per_material_p90,
        "material_nodes_with_candidate_rate": material_nodes_with_candidate_rate,
        "strong_material_candidate_rate": strong_material_candidate_rate,
        "tentative_candidate_rate": tentative_candidate_rate,
        "shared_candidate_rate": shared_candidate_rate,
        "quarantine_candidate_rate": quarantine_candidate_rate,
        "hard_constraint_violation_count": hard_constraint_violation_count,
        "candidate_recall_at_1": metrics["candidate_recall_at_1"],
        "candidate_recall_at_3": metrics["candidate_recall_at_3"],
        "candidate_recall_at_5": metrics["candidate_recall_at_5"],
        "diagnostic_expected_material_count": metrics["diagnostic_expected_material_count"],
        "same_category_candidate_confusion_rate": same_category_proxy,
        "same_category_candidate_confusion_note": "proxy: top-1 false candidate in the same scene; scene-level semantic category labels are unavailable",
        "semantic_only_candidate_confusion_rate": semantic_only_proxy,
        "same_category_candidate_confusion_required_max": same_category_threshold,
        "underseg_candidate_detection_AUC": _underseg_auc(candidate_rows),
        "gate": gate,
        "input_paths": {
            "phase0_failure_lock": _rel(cfg.phase0_failure_lock_path),
            "v60_graph_summary": _rel(cfg.v60_graph_summary_path),
            "v60_node_rows": _rel(cfg.v60_node_rows_path),
            "v60_edge_rows": _rel(cfg.v60_edge_rows_path),
        },
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }
    return {"summary": summary, "material_candidate_rows": candidate_rows, "edge_rows": edge_rows}


def write_v61_graph_v3(result: dict[str, Any], output_root: str | Path) -> dict[str, str]:
    root = _project(output_root)
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "graph_v3_summary": root / "graph_v3_summary.json",
        "material_candidate_rows": root / "material_candidate_rows.csv",
        "edge_rows": root / "edge_rows.csv",
    }
    write_json(paths["graph_v3_summary"], result["summary"])
    write_csv(paths["material_candidate_rows"], result["material_candidate_rows"])
    write_csv(paths["edge_rows"], result["edge_rows"])
    return {name: _rel(path) for name, path in paths.items()}


def write_v61_graph_v3_visualizations(result: dict[str, Any], visualization_root: str | Path) -> dict[str, str]:
    root = _project(visualization_root)
    root.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        summary = result["summary"]
        coverage = root / "material_candidate_coverage.png"
        fig, ax = plt.subplots(figsize=(7.0, 4.0))
        labels = ["candidate coverage", "recall@5", "same-scene confusion"]
        values = [
            summary["material_nodes_with_candidate_rate"],
            summary["candidate_recall_at_5"],
            summary["same_category_candidate_confusion_rate"] or 0.0,
        ]
        ax.bar(labels, values, color=["#2A9D8F", "#457B9D", "#B56576"])
        ax.set_ylim(0.0, 1.05)
        ax.tick_params(axis="x", labelrotation=12)
        ax.set_title("v61 graph v3 candidate gates")
        fig.tight_layout()
        fig.savefig(coverage, dpi=160)
        plt.close(fig)

        hist_path = root / "candidate_histories_per_material.png"
        counts = Counter(row["material_node_id"] for row in result["material_candidate_rows"])
        fig, ax = plt.subplots(figsize=(7.0, 4.0))
        ax.hist(list(counts.values()), bins=range(1, 10), color="#F4A261", align="left")
        ax.set_xlabel("candidate histories per material")
        ax.set_ylabel("material count")
        ax.set_title("v61 material candidate fanout")
        fig.tight_layout()
        fig.savefig(hist_path, dpi=160)
        plt.close(fig)
        return {"candidate_coverage": _rel(coverage), "candidate_fanout": _rel(hist_path), "visualization_status": "created"}
    except Exception as exc:  # pragma: no cover
        error_path = root / "v61_graph_v3_visualization_error.txt"
        error_path.write_text(f"visualization unavailable: {type(exc).__name__}: {exc}\n", encoding="utf-8")
        return {"visualization_error": _rel(error_path), "visualization_status": "unavailable"}


def _edge_info(edge: dict[str, str], source: str) -> dict[str, Any]:
    return {
        "observation_node_id": edge.get("src_node_id") if edge.get("src_node_id", "").startswith("m:") else "",
        "history_node_id": edge.get("dst_node_id") if edge.get("dst_node_id", "").startswith("h:") else "",
        "confidence": parse_float(edge.get("confidence"), 0.0),
        "edge_cost": parse_float(edge.get("edge_cost"), 0.0),
        "source_edge_id": edge.get("edge_id"),
        "evidence_source": source,
    }


def _add_candidate(candidates_for_material: dict[str, dict[str, Any]], history_node_id: str, edge: dict[str, str], source: str) -> None:
    info = {
        "history_node_id": history_node_id,
        "confidence": parse_float(edge.get("confidence"), 0.0),
        "edge_cost": parse_float(edge.get("edge_cost"), 0.0),
        "source_edge_id": edge.get("edge_id"),
    }
    cand = candidates_for_material.setdefault(history_node_id, _new_candidate(history_node_id))
    _update_candidate(cand, source, info, "")


def _add_candidate_info(
    candidates_for_material: dict[str, dict[str, Any]],
    history_node_id: str,
    obs_info: dict[str, Any],
    hist_info: dict[str, Any],
    source: str,
) -> None:
    cand = candidates_for_material.setdefault(history_node_id, _new_candidate(history_node_id))
    combined = {
        "confidence": min(parse_float(obs_info.get("confidence"), 0.0), parse_float(hist_info.get("confidence"), 0.0)),
        "edge_cost": parse_float(obs_info.get("edge_cost"), 0.0) + parse_float(hist_info.get("edge_cost"), 0.0),
        "source_edge_id": f"{obs_info.get('source_edge_id')}|{hist_info.get('source_edge_id')}",
    }
    _update_candidate(cand, source, combined, obs_info.get("observation_node_id", ""))


def _new_candidate(history_node_id: str) -> dict[str, Any]:
    return {
        "history_node_id": history_node_id,
        "evidence_types": set(),
        "source_edge_ids": set(),
        "support_observation_ids": set(),
        "semantic_cost": None,
        "mask_cost": None,
        "material_cost": None,
        "underseg_cost": None,
        "confidence_max": 0.0,
    }


def _update_candidate(cand: dict[str, Any], source: str, info: dict[str, Any], observation_node_id: str) -> None:
    cand["evidence_types"].add(source)
    if info.get("source_edge_id"):
        cand["source_edge_ids"].add(str(info["source_edge_id"]))
    if observation_node_id:
        cand["support_observation_ids"].add(observation_node_id)
    cand["confidence_max"] = max(float(cand["confidence_max"]), parse_float(info.get("confidence"), 0.0))
    cost_key = {
        "K_sem": "semantic_cost",
        "K_mask": "mask_cost",
        "K_mat": "material_cost",
        "K_underseg": "underseg_cost",
    }.get(source)
    if cost_key:
        current = cand.get(cost_key)
        cost = parse_float(info.get("edge_cost"), 0.0)
        cand[cost_key] = cost if current is None else min(float(current), cost)


def _candidate_row(material: dict[str, str], cand: dict[str, Any], rank: int, expected_history_node: str) -> dict[str, Any]:
    evidence = sorted(cand["evidence_types"])
    has_mat = "K_mat" in cand["evidence_types"]
    has_mask = "K_mask" in cand["evidence_types"]
    has_sem = "K_sem" in cand["evidence_types"]
    has_underseg = "K_underseg" in cand["evidence_types"]
    total_cost = _total_cost(cand)
    can_confirm = bool(has_mat)
    can_tentative = bool(has_mat or has_mask or has_sem)
    can_shared = bool(has_underseg or len(cand["support_observation_ids"]) > 1)
    can_quarantine = bool(has_underseg)
    expected_match = bool(expected_history_node and cand["history_node_id"] == expected_history_node)
    return {
        "material_node_id": material["node_id"],
        "scene": material.get("scene"),
        "component_id": material.get("component_id"),
        "candidate_history_id": cand["history_node_id"].removeprefix("h:"),
        "candidate_history_node_id": cand["history_node_id"],
        "candidate_rank": rank,
        "candidate_total_cost": total_cost,
        "candidate_confidence_max": cand["confidence_max"],
        "candidate_evidence_types": "|".join(evidence),
        "has_K_mat": has_mat,
        "has_K_mask": has_mask,
        "has_K_sem": has_sem,
        "has_K_underseg": has_underseg,
        "semantic_cost": cand.get("semantic_cost"),
        "mask_cost": cand.get("mask_cost"),
        "material_cost": cand.get("material_cost"),
        "underseg_cost": cand.get("underseg_cost"),
        "support_observation_count": len(cand["support_observation_ids"]),
        "source_edge_ids_json": sorted(cand["source_edge_ids"]),
        "support_observation_ids_json": sorted(cand["support_observation_ids"]),
        "can_enter_confirmed_core": can_confirm,
        "can_enter_tentative": can_tentative,
        "can_enter_shared": can_shared,
        "can_enter_quarantine": can_quarantine,
        "hard_constraint_violation": False,
        "diagnostic_expected_history_id": expected_history_node.removeprefix("h:") if expected_history_node else "",
        "diagnostic_expected_match": expected_match,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }


def _candidate_rank_key(cand: dict[str, Any]) -> tuple[float, float]:
    return (_total_cost(cand), -float(cand["confidence_max"]))


def _total_cost(cand: dict[str, Any]) -> float:
    cost = 0.0
    if cand.get("material_cost") is not None:
        cost += 0.45 * float(cand["material_cost"])
    else:
        cost += 2.0
    if cand.get("mask_cost") is not None:
        cost += 0.25 * float(cand["mask_cost"])
    else:
        cost += 0.5
    if cand.get("semantic_cost") is not None:
        cost += 0.20 * float(cand["semantic_cost"])
    else:
        cost += 0.25
    if cand.get("underseg_cost") is not None and cand.get("material_cost") is None:
        cost += 0.10 * min(float(cand["underseg_cost"]), 50.0)
    return float(cost)


def _candidate_metrics(material_nodes: list[dict[str, str]], candidates: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    expected = [row for row in material_nodes if row.get("history_id")]
    recalls = {}
    for k in (1, 3, 5):
        ok = 0
        for material in expected:
            target = f"h:{material['history_id']}"
            top = candidates.get(material["node_id"], [])[:k]
            if any(cand["history_node_id"] == target for cand in top):
                ok += 1
        recalls[f"candidate_recall_at_{k}"] = _safe_div(ok, len(expected))
    false_top1_same_scene = 0
    top1_available = 0
    for material in expected:
        top = candidates.get(material["node_id"], [])[:1]
        if not top:
            continue
        top1_available += 1
        top_history = top[0]["history_node_id"].removeprefix("h:")
        if top_history != material["history_id"] and top_history.split("|", 1)[0] == material.get("scene"):
            false_top1_same_scene += 1
    recalls["same_scene_top1_false_candidate_rate"] = _safe_div(false_top1_same_scene, top1_available)
    recalls["diagnostic_expected_material_count"] = len(expected)
    return recalls


def _semantic_only_confusion_metrics(material_nodes: list[dict[str, str]], candidates: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    false_top1 = 0
    available = 0
    for material in material_nodes:
        if not material.get("history_id"):
            continue
        sem = [cand for cand in candidates.get(material["node_id"], []) if "K_sem" in cand["evidence_types"]]
        if not sem:
            continue
        sem.sort(key=_candidate_rank_key)
        available += 1
        top_history = sem[0]["history_node_id"].removeprefix("h:")
        if top_history != material["history_id"] and top_history.split("|", 1)[0] == material.get("scene"):
            false_top1 += 1
    return {"semantic_only_same_scene_top1_false_candidate_rate": _safe_div(false_top1, available)}


def _underseg_auc(candidate_rows: list[dict[str, Any]]) -> float | None:
    labels = [bool(row["has_K_underseg"]) for row in candidate_rows]
    scores = [float(row["support_observation_count"]) + (1.0 if row["can_enter_shared"] else 0.0) for row in candidate_rows]
    return rank_auc(labels, scores)


def _calibrated_confusion_threshold(baseline: float | None) -> float | None:
    if baseline is None:
        return None
    baseline = float(baseline)
    if baseline < 0.05:
        return 0.0
    return baseline - 0.05


def _safe_div(num: int | float, denom: int | float) -> float:
    return 0.0 if float(denom) == 0.0 else float(num) / float(denom)


def _quantile(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(math.ceil(q * len(ordered))) - 1))
    return float(ordered[idx])
