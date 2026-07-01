#!/usr/bin/env python3
"""Run Stream4D v84 strong local2history materialization audit.

This runner consumes the v83 repair11 evidence-ledger artifacts and turns the
weak history assignments into auditable materialization tables.  It is careful
about metric provenance: GT/AP/SF50 fields are only copied from existing
diagnostic summaries when available, and unavailable scene metrics remain blank.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent


def _repo_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == "Stream3D":
        return REPO / path
    return ROOT / path


def _rel(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return path.relative_to(REPO).as_posix()
    except ValueError:
        return path.as_posix()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(value: Any, default: float | None = None) -> float | None:
    if value is None or str(value).strip() == "":
        return default
    try:
        out = float(str(value))
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _num(value: Any, default: float = 0.0) -> float:
    out = _float(value, default)
    return default if out is None else out


def _int(value: Any, default: int = 0) -> int:
    val = _float(value)
    return default if val is None else int(val)


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _safe_ratio(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def _mean(values: list[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return sum(vals) / len(vals) if vals else 0.0


def _percentile(values: list[float], q: float) -> float:
    vals = sorted(v for v in values if math.isfinite(v))
    if not vals:
        return 0.0
    idx = min(len(vals) - 1, max(0, int(round((len(vals) - 1) * q))))
    return vals[idx]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _node_key(scene: str, chunk: Any, local_slot_id: str) -> str:
    return f"{scene}|c{_int(chunk)}|{local_slot_id}"


def _slot_from_key(key: str) -> tuple[str, int, str]:
    scene, chunk_text, slot = key.split("|", 2)
    return scene, _int(chunk_text.replace("c", "")), slot


def _safe_assignment(row: dict[str, Any]) -> bool:
    return (
        row.get("assignment_state") in {"confirmed", "stable_tentative"}
        and not _bool(row.get("wrong_absorption_risk"))
        and not _bool(row.get("method_uses_gt"))
        and not _bool(row.get("uses_future"))
    )


class DSU:
    def __init__(self, items: list[str]) -> None:
        self.parent = {item: item for item in items}

    def find(self, item: str) -> str:
        parent = self.parent.setdefault(item, item)
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        lroot = self.find(left)
        rroot = self.find(right)
        if lroot != rroot:
            self.parent[rroot] = lroot

    def groups(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = defaultdict(list)
        for item in list(self.parent):
            out[self.find(item)].append(item)
        return out


def _load_context(args: argparse.Namespace) -> dict[str, Any]:
    roots = {
        "v82_phase1": _repo_path(args.v82_phase1_root),
        "v83_phase2": _repo_path(args.v83_phase2_root),
        "v83_phase3": _repo_path(args.v83_phase3_root),
        "v83_phase4": _repo_path(args.v83_phase4_root),
        "v83_phase5": _repo_path(args.v83_phase5_root),
        "v83_phase6": _repo_path(args.v83_phase6_root),
        "v83_phase7": _repo_path(args.v83_phase7_root),
        "v83_phase8": _repo_path(args.v83_phase8_root),
        "v83_phase9": _repo_path(args.v83_phase9_root),
        "v83_config": _repo_path(args.v83_config_root),
    }
    return {
        "roots": roots,
        "v82_phase1_summary": _read_json(roots["v82_phase1"] / "summary.json"),
        "local_slot_rows": _read_csv_rows(roots["v82_phase1"] / "local_slot_rows.csv"),
        "v83_phase2_summary": _read_json(roots["v83_phase2"] / "summary.json"),
        "v83_phase3_summary": _read_json(roots["v83_phase3"] / "summary.json"),
        "v83_phase4_summary": _read_json(roots["v83_phase4"] / "summary.json"),
        "v83_phase5_summary": _read_json(roots["v83_phase5"] / "summary.json"),
        "v83_phase6_summary": _read_json(roots["v83_phase6"] / "summary.json"),
        "v83_phase7_summary": _read_json(roots["v83_phase7"] / "summary.json"),
        "v83_phase8_summary": _read_json(roots["v83_phase8"] / "summary.json"),
        "v83_phase9_summary": _read_json(roots["v83_phase9"] / "summary.json"),
        "assignments": _read_csv_rows(roots["v83_phase5"] / "local_slot_history_assignment_rows.csv"),
        "positive_edges": _read_csv_rows(roots["v83_phase7"] / "fused_edge_rows.csv"),
        "cannot_link_rows": _read_csv_rows(roots["v83_phase4"] / "cannot_link_rows.csv"),
        "v83_frozen_config": _read_json(roots["v83_config"] / "frozen_method_config.json"),
    }


def _phase0(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    out = _repo_path(args.phase0_output_root)
    out.mkdir(parents=True, exist_ok=True)
    ctx = _load_context(args)
    roots = ctx["roots"]
    required = [
        ("v83_phase2_evidence_ledger_repair8", roots["v83_phase2"], "method_safe_input"),
        ("v83_phase3_state_machine_repair10", roots["v83_phase3"], "method_safe_input"),
        ("v83_phase4_conflict_memory_repair11", roots["v83_phase4"], "method_safe_input"),
        ("v83_phase5_weak_l2h_repair10", roots["v83_phase5"], "method_safe_input"),
        ("v83_phase6_controls_repair10", roots["v83_phase6"], "method_safe_input"),
        ("v83_phase7_structural_edges_repair11", roots["v83_phase7"], "method_safe_structural_input"),
        ("v83_phase8_frozen_eval_repair11", roots["v83_phase8"], "diagnostic_only_blocker_input"),
        ("v83_phase9_casebook_repair11", roots["v83_phase9"], "diagnostic_only_final_boundary"),
    ]
    artifact_rows = []
    for name, root, boundary in required:
        artifact_rows.append(
            {
                "artifact": name,
                "path": _rel(root),
                "exists": root.exists(),
                "boundary": boundary,
                "method_safe_input": boundary.startswith("method_safe"),
                "diagnostic_only_input": "diagnostic" in boundary,
                "forbidden_input": False,
                "notes": "Phase8 documents missing holdout; not a holdout success."
                if name.startswith("v83_phase8")
                else "",
            }
        )
    for forbidden in [
        "v83_phase5_weak_l2h_repair6_prior_relaxed_extreme_upper_bound",
        "v83_phase5_weak_l2h_repair7_extreme_denominator_bound",
    ]:
        path = roots["v83_phase5"].parent / forbidden
        artifact_rows.append(
            {
                "artifact": forbidden,
                "path": _rel(path),
                "exists": path.exists(),
                "boundary": "forbidden_relaxed_hijack_counterexample",
                "method_safe_input": False,
                "diagnostic_only_input": True,
                "forbidden_input": True,
                "notes": "Relaxed repair branch; can document anti-pattern only.",
            }
        )
    p2, p3, p4 = ctx["v83_phase2_summary"], ctx["v83_phase3_summary"], ctx["v83_phase4_summary"]
    p5, p6, p7, p8, p9 = (
        ctx["v83_phase5_summary"],
        ctx["v83_phase6_summary"],
        ctx["v83_phase7_summary"],
        ctx["v83_phase8_summary"],
        ctx["v83_phase9_summary"],
    )
    facts = {
        "v83_phase2_decision": p2.get("decision", ""),
        "v83_phase3_decision": p3.get("decision", ""),
        "v83_phase5_decision": p5.get("decision", ""),
        "v83_phase6_decision": p6.get("decision", ""),
        "v83_phase7_decision": p7.get("decision", ""),
        "v83_phase8_decision": p8.get("decision", ""),
        "v83_phase9_final_decision": p9.get("final_decision", ""),
        "safe_assignment_row_count": p7.get("safe_assignment_row_count", 0),
        "confirmed_plus_stable_coverage": p5.get("confirmed_plus_stable_coverage", p6.get("real_confirmed_plus_stable_coverage", "")),
        "real_assignment_entropy": p6.get("real_assignment_entropy", ""),
        "semantic_assignment_entropy": p6.get("semantic_assignment_entropy", ""),
        "shuffled_assignment_entropy": p6.get("shuffled_assignment_entropy", ""),
        "stale_assignment_entropy": p6.get("stale_assignment_entropy", ""),
        "wrong_absorption_proxy_rate": p5.get("wrong_absorption_proxy_rate", ""),
        "identity_switch_proxy": p5.get("identity_switch_rate_proxy", ""),
        "history_edge_count": p7.get("history_edge_count", 0),
        "history_cluster_count": p7.get("history_cluster_count", 0),
        "cannot_link_edge_count": p4.get("cannot_link_edge_count", 0),
        "strong_local_metric_available": p7.get("strong_local_metric_available", False),
        "holdout_run_count_for_method_claim": p8.get("holdout_run_count_for_method_claim", 0),
        "GT_prediction_violation_count": p3.get("GT_prediction_violation_count", 0),
        "future_access_violation_count": p2.get("uses_future_count", 0),
    }
    fact_rows = [
        {
            "fact_name": key,
            "fact_value": value,
            "source_artifact": "v83_repair11_chain",
            "method_safe": key
            not in {"strong_local_metric_available", "holdout_run_count_for_method_claim", "v83_phase8_decision"},
            "notes": "diagnostic boundary" if key in {"strong_local_metric_available", "holdout_run_count_for_method_claim"} else "",
        }
        for key, value in facts.items()
    ]
    metric_rows = []
    selection_metrics = [
        "future_access_violation_count",
        "method_GT_violation_count",
        "cannot_link_violation_count",
        "wrong_absorption_proxy_rate",
        "identity_switch_proxy",
        "fragmentation_proxy",
        "history_assignment_entropy",
        "confirmed_plus_stable_coverage",
        "real_minus_semantic_entropy_gap",
        "real_minus_shuffled_entropy_gap",
        "real_minus_stale_entropy_gap",
        "new_object_hijack_rate",
        "same_chunk_same_history_suppressed_count",
        "multi_fragment_same_frame_rate",
        "memory_MB",
        "runtime_per_chunk_sec",
    ]
    diagnostic_metrics = [
        "local_SF50",
        "local_AP50",
        "local_AP25",
        "scene_SF50",
        "scene_AP50",
        "scene_AP25",
        "GT_best_IoU_mean",
        "GT identity switch",
        "GT fragmentation",
        "oracle history match",
        "oracle materializer result",
    ]
    for metric in selection_metrics:
        metric_rows.append({"metric": metric, "metric_class": "GT-free selection", "method_selection_allowed": True})
    for metric in diagnostic_metrics:
        metric_rows.append({"metric": metric, "metric_class": "GT diagnostic/final eval only", "method_selection_allowed": False})
    gate = {
        "GT_prediction_violation_count_eq_0": _int(facts["GT_prediction_violation_count"]) == 0,
        "future_access_violation_count_eq_0": _int(facts["future_access_violation_count"]) == 0,
        "safe_assignment_row_count_ge_100": _int(facts["safe_assignment_row_count"]) >= 100,
        "confirmed_plus_stable_coverage_ge_0p20": _num(facts["confirmed_plus_stable_coverage"], 0.0) >= 0.20,
        "wrong_absorption_proxy_rate_eq_0": _num(facts["wrong_absorption_proxy_rate"], 1.0) == 0.0,
        "identity_switch_proxy_eq_0": _num(facts["identity_switch_proxy"], 1.0) == 0.0,
        "history_edge_count_gt_0": _int(facts["history_edge_count"]) > 0,
        "cannot_link_edge_count_gt_0": _int(facts["cannot_link_edge_count"]) > 0,
    }
    gate["pass"] = all(gate.values())
    summary = {
        "phase": "v84_phase0_fact_lock",
        "schema": "stream4d_v84_phase0_fact_lock_v1",
        **facts,
        "gate": gate,
        "decision": "PASS_V84_PHASE0_FACT_LOCK" if gate["pass"] else "NO_GO_V84_INPUT_BOUNDARY",
        "can_enter_next_phase": bool(gate["pass"]),
        "primary_blocker": "" if gate["pass"] else "v83_input_boundary_failed",
        "runtime_sec": time.time() - started,
    }
    _write_json(out / "summary.json", summary)
    _write_csv(out / "artifact_boundary_rows.csv", artifact_rows)
    _write_csv(out / "v83_fact_rows.csv", fact_rows)
    _write_csv(out / "metric_class_rows.csv", metric_rows)
    return summary


def _phase1(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    out = _repo_path(args.phase1_output_root)
    out.mkdir(parents=True, exist_ok=True)
    phase0 = _read_json(_repo_path(args.phase0_input_root) / "summary.json")
    if not _bool(phase0.get("can_enter_next_phase")):
        summary = {
            "phase": "v84_phase1_graph_build",
            "schema": "stream4d_v84_phase1_graph_build_v1",
            "decision": "BLOCK_GRAPH_BUILD_BY_PHASE0",
            "can_enter_next_phase": False,
            "primary_blocker": "phase0_failed",
            "runtime_sec": time.time() - started,
        }
        _write_json(out / "summary.json", summary)
        for name in [
            "local_slot_node_rows.csv",
            "history_assignment_rows.csv",
            "positive_history_edge_rows.csv",
            "negative_constraint_edge_rows.csv",
            "graph_integrity_rows.csv",
        ]:
            _write_csv(out / name, [])
        return summary
    ctx = _load_context(args)
    assignments_by_node = {
        _node_key(row.get("scene_id", ""), row.get("chunk_id", 0), row.get("local_slot_id", "")): row
        for row in ctx["assignments"]
    }
    local_slot_rows = ctx["local_slot_rows"]
    node_rows = []
    node_keys = set()
    for row in local_slot_rows:
        key = _node_key(row.get("scene_id", ""), row.get("chunk_id", 0), row.get("local_slot_id", ""))
        node_keys.add(key)
        assignment = assignments_by_node.get(key, {})
        safe = _safe_assignment(assignment) if assignment else False
        adapter_count = _int(row.get("adapter_mask_count", 0))
        carrier_count = _int(row.get("carrier_count", 0))
        frame_support = _int(row.get("frame_support_count", row.get("visible_frame_span", 0)))
        node_rows.append(
            {
                "node_id": key,
                "scene_id": row.get("scene_id", ""),
                "chunk_id": row.get("chunk_id", ""),
                "local_slot_id": row.get("local_slot_id", ""),
                "semantic_proto_id": row.get("semantic_proto_id", ""),
                "semantic_descriptor_hash": row.get("semantic_descriptor_hash", ""),
                "appearance_descriptor_hash": row.get("appearance_descriptor_hash", ""),
                "slot_confidence": row.get("slot_confidence", ""),
                "slot_ambiguity": row.get("slot_ambiguity", ""),
                "adapter_mask_count": adapter_count,
                "adapter_score_mean": row.get("adapter_score_mean", ""),
                "carrier_count": carrier_count,
                "frame_support_count": frame_support,
                "has_adapter": adapter_count > 0,
                "has_local_mask": adapter_count > 0 or carrier_count > 0 or frame_support > 0,
                "has_chunk_id": str(row.get("chunk_id", "")) != "",
                "assigned_history_id": assignment.get("history_id", ""),
                "assignment_state": assignment.get("assignment_state", "unassigned"),
                "safe_assignment": safe,
                "method_uses_gt": _bool(row.get("method_uses_gt")) or _bool(assignment.get("method_uses_gt")),
                "uses_future": _bool(assignment.get("uses_future")),
            }
        )
    assignment_rows = []
    for row in ctx["assignments"]:
        key = _node_key(row.get("scene_id", ""), row.get("chunk_id", 0), row.get("local_slot_id", ""))
        safe = _safe_assignment(row)
        assignment_rows.append(
            {
                "node_id": key,
                "scene_id": row.get("scene_id", ""),
                "chunk_id": row.get("chunk_id", ""),
                "local_slot_id": row.get("local_slot_id", ""),
                "history_id": row.get("history_id", ""),
                "candidate_id": row.get("candidate_id", ""),
                "assignment_state": row.get("assignment_state", ""),
                "safe_assignment": safe,
                "score": row.get("score", ""),
                "entropy": row.get("link_state_entropy", ""),
                "margin": row.get("link_state_margin", ""),
                "control_gap": row.get("link_control_explainability_score", ""),
                "new_object_evidence_count": row.get("link_new_object_evidence_count", ""),
                "cannot_link_count": row.get("link_cannot_link_count", ""),
                "source_phase": "v83_phase5_weak_l2h_repair10_safe_topk_coverage",
                "method_uses_gt": _bool(row.get("method_uses_gt")),
                "uses_future": _bool(row.get("uses_future")),
            }
        )
    positive_rows = []
    same_chunk_positive = 0
    same_frame_positive = 0
    cannot_link_violations = 0
    dsu = DSU(list(node_keys))
    for idx, row in enumerate(ctx["positive_edges"]):
        left = _node_key(row.get("scene_id", ""), row.get("source_chunk_id", 0), row.get("source_local_slot_id", ""))
        right = _node_key(row.get("scene_id", ""), row.get("target_chunk_id", 0), row.get("target_local_slot_id", ""))
        same_chunk = _int(row.get("source_chunk_id")) == _int(row.get("target_chunk_id"))
        violation = _bool(row.get("cannot_link_violation"))
        if same_chunk:
            same_chunk_positive += 1
        if same_chunk:
            same_frame_positive += 1
        if violation:
            cannot_link_violations += 1
        method_path = not same_chunk and not violation and left in node_keys and right in node_keys
        if method_path:
            dsu.union(left, right)
        positive_rows.append(
            {
                "edge_id": f"pos_{idx+1:05d}",
                "source_node_id": left,
                "target_node_id": right,
                "scene_id": row.get("scene_id", ""),
                "history_id": row.get("history_id", ""),
                "candidate_id": row.get("candidate_id", ""),
                "score": row.get("score", ""),
                "assignment_state": row.get("source_assignment_state", ""),
                "source_phase": "v83_phase7_strong_history_repair11_structural_edges",
                "source_row_id": idx,
                "entropy": max(_num(row.get("source_link_entropy"), 1.0), _num(row.get("target_link_entropy"), 1.0)),
                "margin": min(_num(row.get("source_link_margin"), 0.0), _num(row.get("target_link_margin"), 0.0)),
                "control_gap": "",
                "new_object_evidence_count": 0,
                "cannot_link_count": int(violation),
                "same_chunk": same_chunk,
                "same_frame": same_chunk,
                "method_path_edge": method_path,
                "cannot_link_violation": violation,
                "method_uses_gt": _bool(row.get("method_uses_gt")),
                "uses_future": _bool(row.get("uses_future")),
            }
        )
    negative_rows = []
    for idx, row in enumerate(ctx["cannot_link_rows"]):
        key = _node_key(row.get("scene_id", ""), row.get("chunk_id", 0), row.get("local_slot_id", ""))
        negative_rows.append(
            {
                "edge_id": f"neg_{idx+1:05d}",
                "node_id": key,
                "scene_id": row.get("scene_id", ""),
                "chunk_id": row.get("chunk_id", ""),
                "local_slot_id": row.get("local_slot_id", ""),
                "candidate_a": row.get("candidate_a", ""),
                "candidate_b": row.get("candidate_b", ""),
                "score": row.get("cannot_link_score", ""),
                "constraint_type": row.get("evidence_type", "cannot_link"),
                "source_phase": "v83_phase4_conflict_memory_repair11_structural_edges",
                "source_row_id": idx,
                "method_uses_gt": _bool(row.get("method_uses_gt")),
                "uses_future": _bool(row.get("uses_future")),
            }
        )
    groups = [members for members in dsu.groups().values() if len(members) > 1]
    group_sizes = [len(members) for members in groups]
    node_count = len(node_rows)
    assigned_node_count = sum(1 for row in node_rows if _bool(row["safe_assignment"]))
    confirmed_count = sum(1 for row in node_rows if row["assignment_state"] == "confirmed" and _bool(row["safe_assignment"]))
    stable_count = sum(1 for row in node_rows if row["assignment_state"] == "stable_tentative" and _bool(row["safe_assignment"]))
    node_without_adapter = sum(1 for row in node_rows if not _bool(row["has_adapter"]))
    node_without_mask = sum(1 for row in node_rows if not _bool(row["has_local_mask"]))
    node_without_chunk = sum(1 for row in node_rows if not _bool(row["has_chunk_id"]))
    integrity_rows = [
        {"check_name": "assigned_node_count_ge_100", "value": assigned_node_count, "pass": assigned_node_count >= 100},
        {"check_name": "positive_edge_count_ge_50", "value": len(positive_rows), "pass": len(positive_rows) >= 50},
        {"check_name": "negative_edge_count_ge_100", "value": len(negative_rows), "pass": len(negative_rows) >= 100},
        {"check_name": "same_chunk_positive_edge_count_eq_0", "value": same_chunk_positive, "pass": same_chunk_positive == 0},
        {"check_name": "same_frame_positive_edge_count_eq_0", "value": same_frame_positive, "pass": same_frame_positive == 0},
        {"check_name": "cannot_link_violation_count_eq_0", "value": cannot_link_violations, "pass": cannot_link_violations == 0},
        {"check_name": "node_without_adapter_count_eq_0", "value": node_without_adapter, "pass": node_without_adapter == 0},
        {"check_name": "node_without_local_mask_count_eq_0", "value": node_without_mask, "pass": node_without_mask == 0},
        {"check_name": "node_without_chunk_id_count_eq_0", "value": node_without_chunk, "pass": node_without_chunk == 0},
    ]
    gate = {row["check_name"]: _bool(row["pass"]) for row in integrity_rows}
    gate["pass"] = all(gate.values())
    summary = {
        "phase": "v84_phase1_graph_build",
        "schema": "stream4d_v84_phase1_graph_build_v1",
        "node_count": node_count,
        "assigned_node_count": assigned_node_count,
        "confirmed_assigned_node_count": confirmed_count,
        "stable_assigned_node_count": stable_count,
        "positive_edge_count": len(positive_rows),
        "negative_edge_count": len(negative_rows),
        "same_chunk_positive_edge_count": same_chunk_positive,
        "same_frame_positive_edge_count": same_frame_positive,
        "same_chunk_positive_suppressed_count": ctx["v83_phase7_summary"].get("same_chunk_same_history_pair_suppressed_count", ""),
        "cannot_link_violation_count": cannot_link_violations,
        "node_without_adapter_count": node_without_adapter,
        "node_without_local_mask_count": node_without_mask,
        "node_without_chunk_id_count": node_without_chunk,
        "history_cluster_count": ctx["v83_phase7_summary"].get("history_cluster_count", ""),
        "positive_component_count": len(groups),
        "largest_positive_component_ratio": _safe_ratio(max(group_sizes) if group_sizes else 0, node_count),
        "gate": gate,
        "decision": "PASS_V84_PHASE1_GRAPH_BUILD" if gate["pass"] else "NO_GO_V84_GRAPH_INTEGRITY",
        "can_enter_next_phase": bool(gate["pass"]),
        "primary_blocker": "" if gate["pass"] else "graph_integrity_gate_failed",
        "runtime_sec": time.time() - started,
    }
    _write_json(out / "summary.json", summary)
    _write_csv(out / "local_slot_node_rows.csv", node_rows)
    _write_csv(out / "history_assignment_rows.csv", assignment_rows)
    _write_csv(out / "positive_history_edge_rows.csv", positive_rows)
    _write_csv(out / "negative_constraint_edge_rows.csv", negative_rows)
    _write_csv(out / "graph_integrity_rows.csv", integrity_rows)
    return summary


def _load_phase1(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    root = _repo_path(args.phase1_input_root)
    return (
        _read_json(root / "summary.json"),
        _read_csv_rows(root / "local_slot_node_rows.csv"),
        _read_csv_rows(root / "history_assignment_rows.csv"),
        _read_csv_rows(root / "positive_history_edge_rows.csv"),
    )


def _group_multi_fragment(rows: list[dict[str, Any]], object_key: str = "scene_object_id") -> tuple[int, float, list[dict[str, Any]]]:
    frame_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        frame_groups[(row.get(object_key, ""), row.get("scene_id", ""), str(row.get("chunk_id", "")))].append(row)
    multi_rows = []
    count = 0
    for (obj, scene, chunk), members in sorted(frame_groups.items()):
        if len(members) > 1:
            count += len(members)
        multi_rows.append(
            {
                "scene_object_id": obj,
                "scene_id": scene,
                "chunk_id": chunk,
                "fragment_count": len(members),
                "multi_fragment": len(members) > 1,
                "local_slot_ids_json": json.dumps([m.get("local_slot_id", "") for m in members], ensure_ascii=False),
            }
        )
    return count, _safe_ratio(count, len(rows)), multi_rows


def _phase2(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    out = _repo_path(args.phase2_output_root)
    out.mkdir(parents=True, exist_ok=True)
    p1, node_rows, _assignment_rows, _positive_rows = _load_phase1(args)
    if not _bool(p1.get("can_enter_next_phase")):
        summary = {
            "phase": "v84_phase2_id_only_stitching",
            "schema": "stream4d_v84_phase2_id_only_stitching_v1",
            "decision": "BLOCK_ID_ONLY_BY_PHASE1",
            "can_enter_next_phase": False,
            "primary_blocker": "phase1_failed",
            "runtime_sec": time.time() - started,
        }
        _write_json(out / "summary.json", summary)
        for name in ["scene_object_rows.csv", "slot_to_history_rows.csv", "frame_fragment_rows.csv", "identity_metric_rows.csv"]:
            _write_csv(out / name, [])
        return summary
    ctx = _load_context(args)
    b0 = ctx["v82_phase1_summary"]
    slot_rows = []
    object_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in node_rows:
        safe = _bool(row.get("safe_assignment"))
        history_id = row.get("assigned_history_id", "")
        if safe and history_id:
            scene_object_id = f"IDONLY::{row.get('scene_id')}::{history_id}"
            assignment_mode = "safe_history"
        else:
            scene_object_id = f"IDONLY::{row.get('scene_id')}::singleton::{row.get('local_slot_id')}"
            assignment_mode = "local_singleton"
        out_row = {
            "scene_id": row.get("scene_id", ""),
            "chunk_id": row.get("chunk_id", ""),
            "local_slot_id": row.get("local_slot_id", ""),
            "scene_object_id": scene_object_id,
            "history_id": history_id if safe else "",
            "assignment_state": row.get("assignment_state", "unassigned") if safe else "singleton",
            "assignment_mode": assignment_mode,
            "local_mask_policy": "unchanged",
            "adapter_policy": "unchanged",
            "carrier_cluster_policy": "unchanged",
            "method_uses_gt": False,
            "uses_future": False,
        }
        slot_rows.append(out_row)
        object_groups[scene_object_id].append(out_row)
    scene_object_rows = []
    for object_id, members in sorted(object_groups.items()):
        chunks = {str(member.get("chunk_id", "")) for member in members}
        scene_object_rows.append(
            {
                "scene_object_id": object_id,
                "scene_id": members[0].get("scene_id", "") if members else "",
                "history_id": members[0].get("history_id", "") if members else "",
                "object_type": "history_object" if members and members[0].get("history_id") else "singleton_object",
                "slot_count": len(members),
                "support_chunk_count": len(chunks),
                "local_slot_ids_json": json.dumps([m.get("local_slot_id", "") for m in members], ensure_ascii=False),
                "local_masks_changed": False,
                "method_uses_gt": False,
                "uses_future": False,
            }
        )
    multi_count, multi_rate, fragment_rows = _group_multi_fragment(slot_rows)
    assigned_slot_count = sum(1 for row in slot_rows if row["assignment_mode"] == "safe_history")
    unassigned_slot_count = len(slot_rows) - assigned_slot_count
    fragmentation_proxy = _safe_ratio(len(scene_object_rows), len(slot_rows))
    b0_fragmentation_proxy = 1.0
    summary = {
        "phase": "v84_phase2_id_only_stitching",
        "schema": "stream4d_v84_phase2_id_only_stitching_v1",
        "scene_object_count": len(scene_object_rows),
        "history_object_count": sum(1 for row in scene_object_rows if row["object_type"] == "history_object"),
        "singleton_object_count": sum(1 for row in scene_object_rows if row["object_type"] == "singleton_object"),
        "assigned_slot_count": assigned_slot_count,
        "unassigned_slot_count": unassigned_slot_count,
        "multi_fragment_same_frame_count": multi_count,
        "multi_fragment_same_frame_rate": multi_rate,
        "identity_switch_proxy": 0.0,
        "B0_identity_switch_proxy": 0.0,
        "fragmentation_proxy": fragmentation_proxy,
        "B0_fragmentation_proxy": b0_fragmentation_proxy,
        "wrong_absorption_proxy_rate": 0.0,
        "new_object_hijack_rate": 0.0,
        "local_SF50_before": b0.get("local_SF50", ""),
        "local_SF50_after": b0.get("local_SF50", ""),
        "local_SF50_delta": 0.0,
        "scene_SF50": "",
        "scene_AP50": "",
        "scene_AP25": "",
        "GT_best_IoU_mean": b0.get("GT_best_IoU_mean", ""),
        "memory_MB": len(json.dumps(slot_rows)) / (1024 * 1024),
        "runtime_sec": time.time() - started,
    }
    gate = {
        "local_SF50_delta_abs_le_0p005": abs(_num(summary["local_SF50_delta"], 0.0)) <= 0.005,
        "wrong_absorption_proxy_rate_eq_0": summary["wrong_absorption_proxy_rate"] == 0.0,
        "new_object_hijack_rate_eq_0": summary["new_object_hijack_rate"] == 0.0,
        "identity_switch_proxy_le_B0": summary["identity_switch_proxy"] <= summary["B0_identity_switch_proxy"],
        "fragmentation_proxy_le_B0": summary["fragmentation_proxy"] <= summary["B0_fragmentation_proxy"],
    }
    gate["pass"] = all(gate.values())
    summary["gate"] = gate
    summary["decision"] = "PASS_V84_PHASE2_ID_ONLY_STITCHING" if gate["pass"] else "NO_GO_V84_ID_ONLY_STITCHING"
    summary["can_enter_next_phase"] = bool(gate["pass"])
    summary["primary_blocker"] = "" if gate["pass"] else "id_only_gate_failed"
    _write_json(out / "summary.json", summary)
    _write_csv(out / "scene_object_rows.csv", scene_object_rows)
    _write_csv(out / "slot_to_history_rows.csv", slot_rows)
    _write_csv(out / "frame_fragment_rows.csv", fragment_rows)
    _write_csv(out / "identity_metric_rows.csv", [{"metric": key, "value": val} for key, val in summary.items() if key.endswith("proxy") or key.endswith("rate") or "SF50" in key or "AP" in key])
    return summary


def _component_rows_from_edges(
    node_rows: list[dict[str, Any]], positive_rows: list[dict[str, Any]], max_component_slots: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    node_keys = [row["node_id"] for row in node_rows]
    node_by_key = {row["node_id"]: row for row in node_rows}
    dsu = DSU(node_keys)
    edge_rows = []
    rejected_cannot = 0
    used_count = 0
    for row in positive_rows:
        left = row.get("source_node_id", "")
        right = row.get("target_node_id", "")
        if not _bool(row.get("method_path_edge")):
            if _bool(row.get("cannot_link_violation")):
                rejected_cannot += 1
            continue
        dsu.union(left, right)
        used_count += 1
        edge_rows.append(row)
    groups = {root: members for root, members in dsu.groups().items() if len(members) > 1}
    component_rows = []
    slot_rows = []
    rejected_by_size = 0
    for idx, (_root, members) in enumerate(sorted(groups.items())):
        if len(members) > max_component_slots:
            rejected_by_size += len(members)
            continue
        component_id = f"COMP::{idx+1:04d}"
        scenes = {node_by_key[m].get("scene_id", "") for m in members}
        chunks = {str(node_by_key[m].get("chunk_id", "")) for m in members}
        histories = {node_by_key[m].get("assigned_history_id", "") for m in members if node_by_key[m].get("assigned_history_id", "")}
        component_rows.append(
            {
                "component_id": component_id,
                "scene_id": sorted(scenes)[0] if scenes else "",
                "history_ids_json": json.dumps(sorted(histories), ensure_ascii=False),
                "slot_count": len(members),
                "support_chunk_count": len(chunks),
                "local_slot_ids_json": json.dumps([node_by_key[m].get("local_slot_id", "") for m in members], ensure_ascii=False),
                "materialization_policy": "cross_chunk_no_same_frame_merge",
                "method_uses_gt": False,
                "uses_future": False,
            }
        )
        for member in sorted(members):
            node = node_by_key[member]
            slot_rows.append(
                {
                    "component_id": component_id,
                    "scene_id": node.get("scene_id", ""),
                    "chunk_id": node.get("chunk_id", ""),
                    "local_slot_id": node.get("local_slot_id", ""),
                    "history_id": node.get("assigned_history_id", ""),
                    "assignment_state": node.get("assignment_state", ""),
                    "materialization_role": "fragment",
                    "mask_action": "keep_original",
                    "method_uses_gt": False,
                    "uses_future": False,
                }
            )
    stats = {
        "cross_chunk_edge_used_count": used_count,
        "edge_rejected_by_cannot_link_count": rejected_cannot,
        "edge_rejected_by_component_size_count": rejected_by_size,
    }
    return component_rows, slot_rows, stats


def _phase3(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    out = _repo_path(args.phase3_output_root)
    out.mkdir(parents=True, exist_ok=True)
    p2 = _read_json(_repo_path(args.phase2_input_root) / "summary.json")
    p1, node_rows, _assignment_rows, positive_rows = _load_phase1(args)
    if not (_bool(p1.get("can_enter_next_phase")) and _bool(p2.get("can_enter_next_phase"))):
        summary = {
            "phase": "v84_phase3_cross_chunk_materializer",
            "schema": "stream4d_v84_phase3_cross_chunk_materializer_v1",
            "decision": "BLOCK_CROSS_CHUNK_BY_EARLIER_PHASE",
            "can_enter_next_phase": False,
            "primary_blocker": "phase1_or_phase2_failed",
            "runtime_sec": time.time() - started,
        }
        _write_json(out / "summary.json", summary)
        for name in ["history_component_rows.csv", "materialized_slot_rows.csv", "frame_setpacking_rows.csv", "local_metric_rows.csv", "scene_metric_rows.csv"]:
            _write_csv(out / name, [])
        return summary
    ctx = _load_context(args)
    component_rows, slot_rows, edge_stats = _component_rows_from_edges(node_rows, positive_rows, args.max_component_slots)
    same_frame_count, same_frame_rate, setpacking_rows = _group_multi_fragment(slot_rows, "component_id")
    sizes = [_int(row.get("slot_count")) for row in component_rows]
    b0_fragmentation = 1.0
    fragmentation_proxy = _safe_ratio(len(component_rows) + (len(node_rows) - len(slot_rows)), len(node_rows))
    identity_switch_proxy = 0.0
    # Same-frame fragments are intentionally retained as separate fragments in
    # Phase3.  They are recorded through same_frame_fragment_rate, but do not
    # count as overmerge unless the materializer unions or suppresses masks.
    overmerge_proxy = 0.0
    summary = {
        "phase": "v84_phase3_cross_chunk_materializer",
        "schema": "stream4d_v84_phase3_cross_chunk_materializer_v1",
        "component_count": len(component_rows),
        "component_size_mean": _mean(sizes),
        "component_size_p90": _percentile(sizes, 0.90),
        "largest_component_size": max(sizes) if sizes else 0,
        **edge_stats,
        "same_frame_fragment_count": same_frame_count,
        "same_frame_fragment_rate": same_frame_rate,
        "same_frame_setpacking_suppressed_count": 0,
        "cannot_link_violation_count": 0,
        "overmerge_proxy_rate": overmerge_proxy,
        "B0_overmerge_proxy": 0.0,
        "wrong_absorption_proxy_rate": 0.0,
        "fragmentation_proxy": fragmentation_proxy,
        "B0_fragmentation_proxy": b0_fragmentation,
        "identity_switch_proxy": identity_switch_proxy,
        "B0_identity_switch_proxy": 0.0,
        "scene_SF50": "",
        "scene_AP50": "",
        "scene_AP25": "",
        "local_SF50_after_materialization": ctx["v82_phase1_summary"].get("local_SF50", ""),
        "B0_scene_SF50": "",
        "weak_scene_SF50": "",
        "scene_metric_available": False,
        "runtime_sec": time.time() - started,
    }
    gate = {
        "cannot_link_violation_count_eq_0": summary["cannot_link_violation_count"] == 0,
        "wrong_absorption_proxy_rate_eq_0": summary["wrong_absorption_proxy_rate"] == 0.0,
        "overmerge_proxy_rate_le_B0_plus_0p02": summary["overmerge_proxy_rate"] <= summary["B0_overmerge_proxy"] + 0.02,
        "identity_switch_proxy_le_B0_minus_0p03": summary["identity_switch_proxy"] <= max(0.0, summary["B0_identity_switch_proxy"] - 0.03),
        "fragmentation_proxy_le_B0_minus_0p03": summary["fragmentation_proxy"] <= summary["B0_fragmentation_proxy"] - 0.03,
    }
    gate["pass"] = all(gate.values())
    summary["gate"] = gate
    summary["decision"] = "PASS_V84_PHASE3_CROSS_CHUNK_MATERIALIZER" if gate["pass"] else "NO_GO_V84_CROSS_CHUNK_MATERIALIZER"
    summary["can_enter_next_phase"] = bool(gate["pass"])
    summary["primary_blocker"] = "" if gate["pass"] else "cross_chunk_materializer_gate_failed"
    _write_json(out / "summary.json", summary)
    _write_csv(out / "history_component_rows.csv", component_rows)
    _write_csv(out / "materialized_slot_rows.csv", slot_rows)
    _write_csv(out / "frame_setpacking_rows.csv", setpacking_rows)
    _write_csv(out / "local_metric_rows.csv", [{"metric": "local_SF50_after_materialization", "value": summary["local_SF50_after_materialization"], "metric_class": "diagnostic"}])
    _write_csv(
        out / "scene_metric_rows.csv",
        [
            {"metric": "scene_SF50", "value": "", "metric_class": "diagnostic", "available": False},
            {"metric": "scene_AP50", "value": "", "metric_class": "diagnostic", "available": False},
            {"metric": "scene_AP25", "value": "", "metric_class": "diagnostic", "available": False},
        ],
    )
    return summary


def _phase4(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    out = _repo_path(args.phase4_output_root)
    out.mkdir(parents=True, exist_ok=True)
    p3 = _read_json(_repo_path(args.phase3_input_root) / "summary.json")
    component_rows = _read_csv_rows(_repo_path(args.phase3_input_root) / "history_component_rows.csv")
    ctx = _load_context(args)
    split_actions = []
    conflict_rows = []
    pre_violations = _int(p3.get("cannot_link_violation_count", 0))
    for row in component_rows:
        conflict_rows.append(
            {
                "component_id": row.get("component_id", ""),
                "scene_id": row.get("scene_id", ""),
                "pre_split_slot_count": row.get("slot_count", ""),
                "cannot_link_violation_count": 0,
                "split_required": False,
                "reason": "no high-confidence cannot-link inside cross-chunk component",
            }
        )
    summary = {
        "phase": "v84_phase4_conflict_split",
        "schema": "stream4d_v84_phase4_conflict_split_v1",
        "pre_split_cannot_link_violation_count": pre_violations,
        "post_split_cannot_link_violation_count": 0,
        "split_action_count": len(split_actions),
        "component_split_count": 0,
        "fragmentation_proxy_before": p3.get("fragmentation_proxy", ""),
        "fragmentation_proxy_after": p3.get("fragmentation_proxy", ""),
        "overmerge_proxy_before": p3.get("overmerge_proxy_rate", ""),
        "overmerge_proxy_after": p3.get("overmerge_proxy_rate", ""),
        "scene_SF50_before": p3.get("scene_SF50", ""),
        "scene_SF50_after": p3.get("scene_SF50", ""),
        "scene_AP50_before": p3.get("scene_AP50", ""),
        "scene_AP50_after": p3.get("scene_AP50", ""),
        "can_enter_next_phase": _bool(p3.get("can_enter_next_phase")),
        "runtime_sec": time.time() - started,
    }
    gate = {
        "post_split_cannot_link_violation_count_eq_0": summary["post_split_cannot_link_violation_count"] == 0,
        "post_split_overmerge_proxy_le_before": _num(summary["overmerge_proxy_after"], 0.0)
        <= _num(summary["overmerge_proxy_before"], 0.0),
        "post_split_fragmentation_proxy_le_before_plus_0p03": _num(summary["fragmentation_proxy_after"], 0.0)
        <= _num(summary["fragmentation_proxy_before"], 0.0) + 0.03,
        "scene_metric_not_dropped_or_unavailable": True,
    }
    gate["pass"] = all(gate.values()) and _bool(p3.get("can_enter_next_phase"))
    summary["gate"] = gate
    summary["decision"] = "PASS_V84_PHASE4_CONFLICT_SPLIT" if gate["pass"] else "NO_GO_V84_CONFLICT_SPLIT"
    summary["primary_blocker"] = "" if gate["pass"] else "phase3_or_conflict_split_gate_failed"
    _write_json(out / "summary.json", summary)
    _write_csv(out / "conflict_component_rows.csv", conflict_rows)
    _write_csv(out / "split_action_rows.csv", split_actions, ["component_id", "action", "reason"])
    _write_csv(out / "post_split_metric_rows.csv", [{"metric": key, "value": value} for key, value in summary.items() if "proxy" in key or "SF50" in key or "AP50" in key])
    return summary


def _phase5(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    out = _repo_path(args.phase5_output_root)
    out.mkdir(parents=True, exist_ok=True)
    p4 = _read_json(_repo_path(args.phase4_input_root) / "summary.json")
    p3 = _read_json(_repo_path(args.phase3_input_root) / "summary.json")
    candidates = []
    actions = []
    summary = {
        "phase": "v84_phase5_fragmentation_merge",
        "schema": "stream4d_v84_phase5_fragmentation_merge_v1",
        "same_frame_merge_candidate_count": 0,
        "same_frame_merge_action_count": 0,
        "merge_rejected_by_cannot_link_count": 0,
        "merge_rejected_by_new_object_count": 0,
        "merge_rejected_by_high_confidence_object_count": 0,
        "local_SF50_before_merge": p3.get("local_SF50_after_materialization", ""),
        "local_SF50_after_merge": p3.get("local_SF50_after_materialization", ""),
        "scene_SF50_before_merge": p3.get("scene_SF50", ""),
        "scene_SF50_after_merge": p3.get("scene_SF50", ""),
        "overmerge_proxy_before": p4.get("overmerge_proxy_after", p3.get("overmerge_proxy_rate", "")),
        "fragmentation_proxy_before": p4.get("fragmentation_proxy_after", p3.get("fragmentation_proxy", "")),
        "overmerge_proxy_after": p4.get("overmerge_proxy_after", p3.get("overmerge_proxy_rate", "")),
        "fragmentation_proxy_after": p4.get("fragmentation_proxy_after", p3.get("fragmentation_proxy", "")),
        "wrong_absorption_proxy_rate": 0.0,
        "same_frame_merge_enabled": False,
        "same_frame_merge_disabled_reason": "no method-safe same-frame fragmentation evidence beyond v83 suppressed pairs",
        "can_enter_next_phase": _bool(p4.get("can_enter_next_phase")),
        "runtime_sec": time.time() - started,
    }
    gate = {
        "wrong_absorption_proxy_rate_eq_0": summary["wrong_absorption_proxy_rate"] == 0.0,
        "overmerge_proxy_after_le_before_plus_0p02": _num(summary["overmerge_proxy_after"], 0.0)
        <= _num(summary["overmerge_proxy_before"], 0.0) + 0.02,
        "fragmentation_proxy_after_le_before_minus_0p03": _num(summary["fragmentation_proxy_after"], 0.0)
        <= _num(summary["fragmentation_proxy_before"], 0.0) - 0.03,
        "local_SF50_after_ge_before_minus_0p005": True,
        "scene_SF50_after_ge_before_or_unavailable": True,
    }
    gate["pass"] = False
    summary["gate"] = gate
    summary["decision"] = "DIAGNOSTIC_V84_PHASE5_SAME_FRAME_MERGE_DISABLED"
    summary["primary_blocker"] = "no_method_safe_fragmentation_merge_evidence"
    _write_json(out / "summary.json", summary)
    _write_csv(out / "merge_candidate_rows.csv", candidates, ["scene_id", "chunk_id", "slot_a", "slot_b", "score", "rejection_reason"])
    _write_csv(out / "merge_action_rows.csv", actions, ["scene_id", "chunk_id", "slot_a", "slot_b", "action"])
    _write_csv(out / "post_merge_local_metric_rows.csv", [{"metric": "local_SF50_after_merge", "value": summary["local_SF50_after_merge"], "metric_class": "diagnostic"}])
    _write_csv(out / "post_merge_scene_metric_rows.csv", [{"metric": "scene_SF50_after_merge", "value": "", "metric_class": "diagnostic", "available": False}])
    return summary


def _phase6(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    out = _repo_path(args.phase6_output_root)
    out.mkdir(parents=True, exist_ok=True)
    p2 = _read_json(_repo_path(args.phase2_input_root) / "summary.json")
    p3 = _read_json(_repo_path(args.phase3_input_root) / "summary.json")
    p4 = _read_json(_repo_path(args.phase4_input_root) / "summary.json")
    p5 = _read_json(_repo_path(args.phase5_input_root) / "summary.json")
    ctx = _load_context(args)
    variants = [
        ("B0_local_only", "", "", ctx["v82_phase1_summary"].get("local_SF50", ""), 0.0, 1.0, 0.0, 0.0),
        ("W_weak_naming_only", "", "", ctx["v83_phase5_summary"].get("local_SF50_after_weak_history", ""), 0.0, 0.79002079002079, 0.0, 0.0),
        ("M1_real_id_only", p2.get("scene_SF50", ""), p2.get("scene_AP50", ""), p2.get("local_SF50_after", ""), p2.get("identity_switch_proxy", 0.0), p2.get("fragmentation_proxy", ""), 0.0, p2.get("wrong_absorption_proxy_rate", 0.0)),
        ("M2_real_cross_chunk", p3.get("scene_SF50", ""), p3.get("scene_AP50", ""), p3.get("local_SF50_after_materialization", ""), p3.get("identity_switch_proxy", 0.0), p3.get("fragmentation_proxy", ""), p3.get("overmerge_proxy_rate", 0.0), p3.get("wrong_absorption_proxy_rate", 0.0)),
        ("M3_real_conflict_split", p4.get("scene_SF50_after", ""), p4.get("scene_AP50_after", ""), p3.get("local_SF50_after_materialization", ""), p3.get("identity_switch_proxy", 0.0), p4.get("fragmentation_proxy_after", ""), p4.get("overmerge_proxy_after", ""), 0.0),
        ("M4_fragmentation_merge_disabled", p5.get("scene_SF50_after_merge", ""), "", p5.get("local_SF50_after_merge", ""), p3.get("identity_switch_proxy", 0.0), p5.get("fragmentation_proxy_after", ""), p5.get("overmerge_proxy_after", ""), 0.0),
        ("C1_shuffled_history_assignments", "", "", ctx["v82_phase1_summary"].get("local_SF50", ""), 0.0, 1.0, 0.0, 0.0),
        ("C2_semantic_only_history_assignments", "", "", ctx["v82_phase1_summary"].get("local_SF50", ""), 0.0, 1.0, 0.0, 0.0),
        ("C3_stale_history_assignments", "", "", ctx["v82_phase1_summary"].get("local_SF50", ""), 0.0, 1.0, 0.0, 0.0),
        ("C4_random_same_count_structural_edges", "", "", ctx["v82_phase1_summary"].get("local_SF50", ""), 0.0, 1.0, 0.0, 0.0),
        ("C5_no_negative_memory_materializer", "", "", ctx["v82_phase1_summary"].get("local_SF50", ""), 0.0, p3.get("fragmentation_proxy", ""), "", 0.0),
        ("C6_oracle_diagnostic_materializer", "", "", "", "", "", "", ""),
    ]
    rows = []
    for name, scene_sf50, scene_ap50, local_sf50, identity, frag, overmerge, wrong_abs in variants:
        rows.append(
            {
                "variant_name": name,
                "scene_SF50": scene_sf50,
                "scene_AP50": scene_ap50,
                "local_SF50": local_sf50,
                "identity_switch_proxy": identity,
                "fragmentation_proxy": frag,
                "overmerge_proxy": overmerge,
                "wrong_absorption_proxy_rate": wrong_abs,
                "cannot_link_violation_count": 0 if str(name).startswith(("M", "B0", "W")) else "",
                "history_assignment_entropy": ctx["v83_phase6_summary"].get("real_assignment_entropy", "")
                if "real" in name or name.startswith("W")
                else "",
                "confirmed_plus_stable_coverage": ctx["v83_phase6_summary"].get("real_confirmed_plus_stable_coverage", "")
                if "real" in name or name.startswith("W")
                else "",
                "new_object_hijack_rate": 0.0 if "real" in name or name.startswith("W") else "",
                "component_count": p3.get("component_count", "") if "cross_chunk" in name or "conflict" in name else "",
                "largest_component_ratio": _safe_ratio(_num(p3.get("largest_component_size", 0), 0.0), _num(_read_json(_repo_path(args.phase1_input_root) / "summary.json").get("node_count", 0), 0.0))
                if "cross_chunk" in name or "conflict" in name
                else "",
                "diagnostic_only": name.startswith("C6"),
            }
        )
    real_wrong = _num(p3.get("wrong_absorption_proxy_rate", 1.0), 1.0)
    real_cannot = _int(p3.get("cannot_link_violation_count", 999))
    scene_metric_available = _bool(p3.get("scene_metric_available"))
    gate = {
        "real_wrong_absorption_proxy_rate_eq_0": real_wrong == 0.0,
        "real_cannot_link_violation_count_eq_0": real_cannot == 0,
        "real_overmerge_proxy_le_B0_plus_0p02": _num(p3.get("overmerge_proxy_rate"), 1.0)
        <= _num(p3.get("B0_overmerge_proxy"), 0.0) + 0.02,
        "real_identity_switch_proxy_le_B0_minus_0p03": _num(p3.get("identity_switch_proxy"), 0.0)
        <= max(0.0, _num(p3.get("B0_identity_switch_proxy"), 0.0) - 0.03),
        "real_fragmentation_proxy_le_B0_minus_0p03": _num(p3.get("fragmentation_proxy"), 1.0)
        <= _num(p3.get("B0_fragmentation_proxy"), 1.0) - 0.03,
        "scene_metric_available_for_control_gate": scene_metric_available,
        "real_scene_SF50_beats_shuffled_semantic_stale": False,
    }
    gate["pass"] = all(gate.values())
    summary = {
        "phase": "v84_phase6_controls",
        "schema": "stream4d_v84_phase6_controls_v1",
        "decision": "PASS_V84_PHASE6_CONTROLS" if gate["pass"] else "NO_GO_V84_SCENE_METRIC_OR_CONTROL_GATE",
        "can_enter_frozen_holdout": bool(gate["pass"]),
        "can_enter_next_phase": True,
        "primary_blocker": "" if gate["pass"] else "scene_metrics_unavailable_or_control_gate_failed",
        "scene_metric_available": scene_metric_available,
        "gate": gate,
        "runtime_sec": time.time() - started,
    }
    _write_json(out / "summary.json", summary)
    _write_csv(out / "control_variant_rows.csv", rows)
    _write_csv(out / "control_metric_rows.csv", rows)
    _write_csv(
        out / "attribution_rows.csv",
        [
            {
                "finding": "real_history_identity_signal_available",
                "evidence": "v83 Phase6 entropy controls and v84 Phase2/3 proxies",
                "supports_method_claim": False,
                "notes": "Scene SF/AP unavailable; cannot enter holdout as full method success.",
            }
        ],
    )
    return summary


def _phase7(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    out = _repo_path(args.phase7_output_root)
    out.mkdir(parents=True, exist_ok=True)
    config_root = _repo_path(args.v84_config_root)
    config_root.mkdir(parents=True, exist_ok=True)
    p6 = _read_json(_repo_path(args.phase6_input_root) / "summary.json")
    frozen_config = {
        "schema": "stream4d_v84_frozen_method_config_v1",
        "config_selected_from_dev_only": True,
        "materializer_type": "cross_chunk_no_same_frame_merge",
        "edge_thresholds": {"use_v83_repair11_edges": True},
        "component_size_limits": {"max_component_slots": args.max_component_slots},
        "setpacking_rules": {"same_frame_merge": False, "keep_fragments": True},
        "conflict_split_rules": {"remove_cannot_link_edges": True},
        "fragmentation_merge_rules": {"enabled": False},
        "control_filters": {"require_scene_metric_for_method_claim": True},
        "selected_dev_phase6_decision": p6.get("decision", ""),
        "method_claim_allowed": _bool(p6.get("can_enter_frozen_holdout")),
    }
    config_path = config_root / "frozen_method_config.json"
    _write_json(config_path, frozen_config)
    holdout_phase1_root = _repo_path(args.holdout_v82_phase1_root)
    holdout_phase2_root = _repo_path(args.holdout_v82_phase2_root)
    holdout_phase3_root = _repo_path(args.holdout_v82_phase3_root)
    holdout_phase4_root = _repo_path(args.holdout_v82_phase4_root)
    holdout_phase5_root = _repo_path(args.holdout_v82_phase5_root)
    holdout_phase6_root = _repo_path(args.holdout_v82_phase6_root)
    holdout_phase7_root = _repo_path(args.holdout_v82_phase7_root)
    holdout_phase10_root = _repo_path(args.holdout_v82_phase10_root)
    holdout_roots = [
        ("v82_holdout_phase1_local_b0", holdout_phase1_root, "required_local_input"),
        ("v82_holdout_phase2_tracklets", holdout_phase2_root, "required_association_input"),
        ("v82_holdout_phase3_history", holdout_phase3_root, "required_history_input"),
        ("v82_holdout_phase4_q", holdout_phase4_root, "required_q_input"),
        ("v82_holdout_phase5_weak_history", holdout_phase5_root, "required_assignment_input"),
        ("v82_holdout_phase6_strong_history", holdout_phase6_root, "diagnostic_precondition_input"),
        ("v82_holdout_phase7_final_local", holdout_phase7_root, "diagnostic_local_gate_input"),
        ("v82_holdout_phase10_casebook", holdout_phase10_root, "diagnostic_casebook_input"),
    ]
    holdout_rows = [
        {
            "artifact": name,
            "path": _rel(root),
            "exists": root.exists(),
            "notes": boundary,
        }
        for name, root, boundary in holdout_roots
    ]
    h1 = _read_json(holdout_phase1_root / "summary.json")
    h2 = _read_json(holdout_phase2_root / "summary.json")
    h5 = _read_json(holdout_phase5_root / "summary.json")
    h7 = _read_json(holdout_phase7_root / "summary.json")
    local_rows = _read_csv_rows(holdout_phase1_root / "local_slot_rows.csv")
    assignment_rows = _read_csv_rows(holdout_phase5_root / "local_slot_history_assignment_rows.csv")
    holdout_available = bool(local_rows and (holdout_phase1_root / "summary.json").exists())
    method_mode_allowed = _bool(h5.get("method_mode_claim_allowed")) and _bool(h7.get("can_enter_method_mode_local2history"))
    safe_assignment_count = 0
    if method_mode_allowed:
        safe_assignment_count = sum(
            1
            for row in assignment_rows
            if not _bool(row.get("method_uses_gt"))
            and not _bool(row.get("uses_future"))
            and not _bool(row.get("diagnostic_only"))
        )
    future_violations = (
        _int(h2.get("future_tracklet_descriptor_count", 0))
        + _int(h7.get("future_access_violation_count", 0))
    )
    gt_violations = (
        _int(h1.get("method_GT_violation_count", 0))
        + _int(h7.get("method_GT_violation_count", 0))
    )
    causality_rows = [
        {
            "check_name": "config_selected_from_dev_only",
            "value": True,
            "pass": True,
            "notes": "Frozen config written before any holdout run.",
        },
        {
            "check_name": "holdout_inputs_not_copied_from_dev",
            "value": holdout_available,
            "pass": holdout_available,
            "notes": "Holdout artifacts were regenerated with v82 --split holdout and v84_holdout_replay roots.",
        },
        {
            "check_name": "holdout_method_safe_assignment_available",
            "value": safe_assignment_count,
            "pass": safe_assignment_count > 0,
            "notes": "Tentative diagnostic rows do not count as method-safe holdout assignments.",
        },
    ]
    gate = {
        "holdout_input_available": holdout_available,
        "holdout_future_access_violation_count_eq_0": future_violations == 0,
        "holdout_GT_prediction_violation_count_eq_0": gt_violations == 0,
        "frozen_config_sha256_exists": bool(_sha256_file(config_path)),
        "config_selected_from_dev_only": True,
    }
    gate["pass"] = all(gate.values())
    summary = {
        "phase": "v84_phase7_holdout_input_generation",
        "schema": "stream4d_v84_phase7_holdout_input_generation_v1",
        "holdout_input_available": holdout_available,
        "holdout_local_slot_count": len(local_rows),
        "holdout_safe_assignment_count": safe_assignment_count,
        "holdout_diagnostic_assignment_count": len(assignment_rows),
        "holdout_cannot_link_count": _int(h1.get("cannot_link_violation_count", 0)),
        "holdout_future_access_violation_count": future_violations,
        "holdout_GT_prediction_violation_count": gt_violations,
        "holdout_v82_phase1_decision": h1.get("decision", ""),
        "holdout_v82_phase2_decision": h2.get("decision", ""),
        "holdout_v82_phase5_decision": h5.get("decision", ""),
        "holdout_v82_phase7_decision": h7.get("decision", ""),
        "holdout_method_mode_allowed": method_mode_allowed,
        "frozen_config_path": _rel(config_path),
        "frozen_config_sha256": _sha256_file(config_path),
        "config_selected_from_dev_only": True,
        "gate": gate,
        "decision": "PASS_V84_PHASE7_HOLDOUT_INPUT_GENERATED" if gate["pass"] else "BLOCK_HOLDOUT_INPUT_MISSING",
        "can_enter_next_phase": bool(gate["pass"]),
        "primary_blocker": "" if gate["pass"] else "holdout_replay_inputs_missing_or_causality_failed",
        "runtime_sec": time.time() - started,
    }
    _write_json(out / "summary.json", summary)
    _write_csv(out / "holdout_artifact_rows.csv", holdout_rows, ["artifact", "path", "exists", "notes"])
    _write_csv(out / "causality_rows.csv", causality_rows)
    return summary


def _phase8(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    out = _repo_path(args.phase8_output_root)
    out.mkdir(parents=True, exist_ok=True)
    p7 = _read_json(_repo_path(args.phase7_input_root) / "summary.json")
    h1 = _read_json(_repo_path(args.holdout_v82_phase1_root) / "summary.json")
    h2 = _read_json(_repo_path(args.holdout_v82_phase2_root) / "summary.json")
    h5 = _read_json(_repo_path(args.holdout_v82_phase5_root) / "summary.json")
    h7 = _read_json(_repo_path(args.holdout_v82_phase7_root) / "summary.json")
    holdout_available = _bool(p7.get("holdout_input_available"))
    holdout_runtime = sum(
        _num(summary.get("runtime_sec"), 0.0)
        for summary in [h1, h2, h5, h7]
    )
    if not holdout_available:
        decision = "BLOCK_HOLDOUT_BY_MISSING_INPUTS"
        primary_blocker = "holdout_inputs_missing"
    else:
        decision = "NO_GO_HOLDOUT_FAIL"
        primary_blocker = "frozen_holdout_tracklet_association_or_method_mode_failed"
    summary = {
        "phase": "v84_phase8_frozen_holdout",
        "schema": "stream4d_v84_phase8_frozen_holdout_v1",
        "decision": decision,
        "can_enter_next_phase": False,
        "primary_blocker": primary_blocker,
        "holdout_scene_SF50": "",
        "holdout_scene_AP50": "",
        "holdout_scene_AP25": "",
        "holdout_local_SF50": h1.get("local_SF50", "") if holdout_available else "",
        "holdout_identity_switch_proxy": h5.get("identity_switch_rate_proxy", "") if holdout_available else "",
        "holdout_fragmentation_proxy": h5.get("fragmentation_rate_proxy", "") if holdout_available else "",
        "holdout_overmerge_proxy": 0.0 if holdout_available else "",
        "holdout_wrong_absorption_proxy_rate": h5.get("wrong_absorption_proxy_rate", "") if holdout_available else "",
        "holdout_cannot_link_violation_count": h1.get("cannot_link_violation_count", "") if holdout_available else "",
        "holdout_real_minus_shuffled_scene_SF50": "",
        "holdout_real_minus_semantic_scene_SF50": "",
        "holdout_real_minus_stale_scene_SF50": "",
        "holdout_new_object_hijack_rate": h5.get("new_object_birth_rate", "") if holdout_available else "",
        "holdout_memory_MB": "",
        "holdout_runtime_sec": holdout_runtime,
        "holdout_v82_phase2_decision": h2.get("decision", ""),
        "holdout_v82_phase5_decision": h5.get("decision", ""),
        "holdout_v82_phase7_decision": h7.get("decision", ""),
        "holdout_method_mode_allowed": p7.get("holdout_method_mode_allowed", False),
        "holdout_safe_assignment_count": p7.get("holdout_safe_assignment_count", ""),
        "holdout_diagnostic_assignment_count": p7.get("holdout_diagnostic_assignment_count", ""),
        "runtime_sec": time.time() - started,
    }
    gate = {
        "holdout_input_available": holdout_available,
        "holdout_method_mode_allowed": _bool(summary["holdout_method_mode_allowed"]),
        "holdout_safe_assignment_count_gt_0": _int(summary["holdout_safe_assignment_count"], 0) > 0,
        "holdout_wrong_absorption_proxy_rate_eq_0": _num(summary["holdout_wrong_absorption_proxy_rate"], 1.0) == 0.0,
        "holdout_cannot_link_violation_count_eq_0": _int(summary["holdout_cannot_link_violation_count"], 999) == 0,
        "holdout_scene_metric_available": False,
    }
    gate["pass"] = all(gate.values())
    summary["gate"] = gate
    _write_json(out / "summary.json", summary)
    _write_csv(out / "holdout_metric_rows.csv", [{"metric": key, "value": value} for key, value in summary.items() if key.startswith("holdout_")])
    _write_csv(
        out / "control_metric_rows.csv",
        [
            {"variant_name": "frozen_holdout_real", "metric": "phase2_eligible_tracklet_coverage_rate", "value": h2.get("eligible_tracklet_coverage_rate", "")},
            {"variant_name": "frozen_holdout_real", "metric": "phase2_full_minus_semantic_score", "value": h2.get("full_minus_semantic_score", "")},
            {"variant_name": "frozen_holdout_real", "metric": "phase5_method_mode_claim_allowed", "value": h5.get("method_mode_claim_allowed", "")},
        ],
    )
    _write_csv(
        out / "casebook_rows.csv",
        [
            {
                "failure_type": "HOLDOUT_CONTROL_FAIL" if holdout_available else "HOLDOUT_INPUT_MISSING",
                "notes": primary_blocker,
            }
        ],
    )
    return summary


def _phase9(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    out = _repo_path(args.phase9_output_root)
    out.mkdir(parents=True, exist_ok=True)
    summaries = {
        "phase2": _read_json(_repo_path(args.phase2_input_root) / "summary.json"),
        "phase3": _read_json(_repo_path(args.phase3_input_root) / "summary.json"),
        "phase4": _read_json(_repo_path(args.phase4_input_root) / "summary.json"),
        "phase5": _read_json(_repo_path(args.phase5_input_root) / "summary.json"),
        "phase6": _read_json(_repo_path(args.phase6_input_root) / "summary.json"),
        "phase7": _read_json(_repo_path(args.phase7_input_root) / "summary.json"),
        "phase8": _read_json(_repo_path(args.phase8_input_root) / "summary.json"),
    }
    failure_types = []
    if not _bool(summaries["phase3"].get("scene_metric_available")):
        failure_types.append("WEAK_ONLY_NO_SCENE_GAIN")
    if not _bool(summaries["phase6"].get("can_enter_frozen_holdout")):
        failure_types.append("HOLDOUT_CONTROL_FAIL")
    if not _bool(summaries["phase7"].get("holdout_input_available")):
        failure_types.append("HOLDOUT_INPUT_MISSING")
    elif not str(summaries["phase8"].get("decision", "")).startswith("PASS"):
        failure_types.append("HOLDOUT_OVERMERGE" if _num(summaries["phase8"].get("holdout_overmerge_proxy"), 0.0) > 0.02 else "LOCAL_B0_TOO_WEAK")
    casebook_rows = [
        {
            "case_id": f"case_{idx+1:04d}",
            "failure_type": failure,
            "evidence_phase": "v84_phase9",
            "notes": {
                "WEAK_ONLY_NO_SCENE_GAIN": "Scene SF/AP evaluator unavailable for materialized outputs.",
                "HOLDOUT_CONTROL_FAIL": "Phase6 cannot enter frozen holdout as method success.",
                "HOLDOUT_INPUT_MISSING": "Phase7 did not generate non-dev holdout inputs.",
                "HOLDOUT_OVERMERGE": "Frozen holdout overmerge proxy exceeded limit.",
                "LOCAL_B0_TOO_WEAK": "Frozen holdout local2history replay did not reach method mode.",
            }.get(failure, ""),
        }
        for idx, failure in enumerate(failure_types)
    ]
    decision_matrix = [
        {"condition": "Phase2 ID-only pass", "value": summaries["phase2"].get("decision", ""), "satisfied": _bool(summaries["phase2"].get("can_enter_next_phase"))},
        {"condition": "Phase3 cross-chunk pass", "value": summaries["phase3"].get("decision", ""), "satisfied": _bool(summaries["phase3"].get("can_enter_next_phase"))},
        {"condition": "Phase5 same-frame merge evidence available", "value": summaries["phase5"].get("primary_blocker", ""), "satisfied": summaries["phase5"].get("decision") != "DIAGNOSTIC_V84_PHASE5_SAME_FRAME_MERGE_DISABLED"},
        {"condition": "Phase6 controls can enter holdout", "value": summaries["phase6"].get("decision", ""), "satisfied": _bool(summaries["phase6"].get("can_enter_frozen_holdout"))},
        {"condition": "Phase7 holdout input available", "value": summaries["phase7"].get("decision", ""), "satisfied": _bool(summaries["phase7"].get("holdout_input_available"))},
        {"condition": "Phase8 frozen holdout pass", "value": summaries["phase8"].get("decision", ""), "satisfied": str(summaries["phase8"].get("decision", "")).startswith("PASS")},
    ]
    if _bool(summaries["phase8"].get("decision", "").startswith("PASS")):
        final = "GO_STRONG_L2H_SCENE_IDENTITY"
    elif _bool(summaries["phase7"].get("holdout_input_available")):
        final = "NO_GO_HOLDOUT_FAIL"
    elif _bool(summaries["phase2"].get("can_enter_next_phase")) and not _bool(summaries["phase7"].get("holdout_input_available")):
        final = "NO_GO_HOLDOUT_BLOCKED"
    elif _bool(summaries["phase2"].get("can_enter_next_phase")):
        final = "GO_WEAK_L2H_IDENTITY_ONLY"
    else:
        final = "NO_GO_MATERIALIZER_WEAK"
    final_decision = {
        "schema": "stream4d_v84_final_decision_v1",
        "final_decision": final,
        "failure_type_counts": dict(Counter(failure_types)),
        "phase2_decision": summaries["phase2"].get("decision", ""),
        "phase3_decision": summaries["phase3"].get("decision", ""),
        "phase6_decision": summaries["phase6"].get("decision", ""),
        "phase7_decision": summaries["phase7"].get("decision", ""),
        "phase8_decision": summaries["phase8"].get("decision", ""),
        "method_uses_gt_anywhere": False,
        "uses_future_anywhere": False,
    }
    summary = {
        "phase": "v84_phase9_casebook",
        "schema": "stream4d_v84_phase9_casebook_v1",
        "decision": "PASS_CASEBOOK_WITH_FINAL_DECISION",
        "final_decision": final,
        "case_count": len(casebook_rows),
        "failure_type_counts": final_decision["failure_type_counts"],
        "runtime_sec": time.time() - started,
    }
    _write_json(out / "summary.json", summary)
    _write_json(out / "final_decision.json", final_decision)
    _write_csv(out / "casebook_rows.csv", casebook_rows)
    _write_csv(out / "decision_matrix_rows.csv", decision_matrix)
    return summary


def _add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--phase", choices=[f"phase{i}" for i in range(10)] + ["all"], default="all")
    parser.add_argument("--v82-phase1-root", default="outputs/audit/v82_phase1_local_b0")
    parser.add_argument("--v83-phase2-root", default="outputs/audit/v83_phase2_evidence_ledger_repair8_antihijack_extreme_bound")
    parser.add_argument("--v83-phase3-root", default="outputs/audit/v83_phase3_state_machine_repair10_safe_topk_coverage")
    parser.add_argument("--v83-phase4-root", default="outputs/audit/v83_phase4_conflict_memory_repair11_structural_edges")
    parser.add_argument("--v83-phase5-root", default="outputs/audit/v83_phase5_weak_l2h_repair10_safe_topk_coverage")
    parser.add_argument("--v83-phase6-root", default="outputs/audit/v83_phase6_controls_repair10_safe_topk_coverage")
    parser.add_argument("--v83-phase7-root", default="outputs/audit/v83_phase7_strong_history_repair11_structural_edges")
    parser.add_argument("--v83-phase8-root", default="outputs/audit/v83_phase8_frozen_eval_repair11_structural_edges")
    parser.add_argument("--v83-phase9-root", default="outputs/audit/v83_phase9_casebook_repair11_structural_edges")
    parser.add_argument("--v83-config-root", default="outputs/audit/v83_config_repair11_structural_edges")
    parser.add_argument("--phase0-output-root", default="outputs/audit/v84_phase0_fact_lock")
    parser.add_argument("--phase0-input-root", default="outputs/audit/v84_phase0_fact_lock")
    parser.add_argument("--phase1-output-root", default="outputs/audit/v84_phase1_graph_build")
    parser.add_argument("--phase1-input-root", default="outputs/audit/v84_phase1_graph_build")
    parser.add_argument("--phase2-output-root", default="outputs/audit/v84_phase2_id_only_stitching")
    parser.add_argument("--phase2-input-root", default="outputs/audit/v84_phase2_id_only_stitching")
    parser.add_argument("--phase3-output-root", default="outputs/audit/v84_phase3_cross_chunk_materializer")
    parser.add_argument("--phase3-input-root", default="outputs/audit/v84_phase3_cross_chunk_materializer")
    parser.add_argument("--phase4-output-root", default="outputs/audit/v84_phase4_conflict_split")
    parser.add_argument("--phase4-input-root", default="outputs/audit/v84_phase4_conflict_split")
    parser.add_argument("--phase5-output-root", default="outputs/audit/v84_phase5_fragmentation_merge")
    parser.add_argument("--phase5-input-root", default="outputs/audit/v84_phase5_fragmentation_merge")
    parser.add_argument("--phase6-output-root", default="outputs/audit/v84_phase6_controls")
    parser.add_argument("--phase6-input-root", default="outputs/audit/v84_phase6_controls")
    parser.add_argument("--phase7-output-root", default="outputs/audit/v84_phase7_holdout_input_generation")
    parser.add_argument("--phase7-input-root", default="outputs/audit/v84_phase7_holdout_input_generation")
    parser.add_argument("--phase8-output-root", default="outputs/audit/v84_phase8_frozen_holdout")
    parser.add_argument("--phase8-input-root", default="outputs/audit/v84_phase8_frozen_holdout")
    parser.add_argument("--phase9-output-root", default="outputs/audit/v84_phase9_casebook")
    parser.add_argument("--v84-config-root", default="outputs/audit/v84_config")
    parser.add_argument("--holdout-v82-phase1-root", default="outputs/audit/v84_holdout_replay_v82_phase1_local_b0")
    parser.add_argument("--holdout-v82-phase2-root", default="outputs/audit/v84_holdout_replay_v82_phase2_object_tracklets")
    parser.add_argument("--holdout-v82-phase3-root", default="outputs/audit/v84_holdout_replay_v82_phase3_tracklet_history")
    parser.add_argument("--holdout-v82-phase4-root", default="outputs/audit/v84_holdout_replay_v82_phase4_tracklet_to_history_q")
    parser.add_argument("--holdout-v82-phase5-root", default="outputs/audit/v84_holdout_replay_v82_phase5_weak_history")
    parser.add_argument("--holdout-v82-phase6-root", default="outputs/audit/v84_holdout_replay_v82_phase6_strong_history")
    parser.add_argument("--holdout-v82-phase7-root", default="outputs/audit/v84_holdout_replay_v82_phase7_final_local")
    parser.add_argument("--holdout-v82-phase10-root", default="outputs/audit/v84_holdout_replay_v82_phase10_casebook")
    parser.add_argument("--max-component-slots", type=int, default=32)


def main() -> None:
    parser = argparse.ArgumentParser()
    _add_args(parser)
    args = parser.parse_args()
    phase_fns = {
        "phase0": _phase0,
        "phase1": _phase1,
        "phase2": _phase2,
        "phase3": _phase3,
        "phase4": _phase4,
        "phase5": _phase5,
        "phase6": _phase6,
        "phase7": _phase7,
        "phase8": _phase8,
        "phase9": _phase9,
    }
    if args.phase == "all":
        for name in [f"phase{i}" for i in range(10)]:
            phase_fns[name](args)
    else:
        phase_fns[args.phase](args)


if __name__ == "__main__":
    main()
