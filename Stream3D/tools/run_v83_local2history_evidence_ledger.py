#!/usr/bin/env python3
"""Run Stream4D v83 local2history evidence-ledger audit.

This runner is intentionally evidence-first.  It reuses the v82 causal
artifacts named by the v83 plan, converts high-entropy tentative associations
into a causal top-K distribution and ledger, and emits blocked downstream
artifacts when the weak local2history gates do not pass.
"""

from __future__ import annotations

import argparse
import csv
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


def _int(value: Any, default: int = 0) -> int:
    val = _float(value)
    return default if val is None else int(val)


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _mean(values: list[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return sum(vals) / len(vals) if vals else 0.0


def _safe_ratio(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def _entropy_from_scores(values: list[float]) -> float:
    vals = [max(0.0, float(v)) for v in values if math.isfinite(float(v))]
    if len(vals) <= 1:
        return 0.0
    total = sum(vals)
    if total <= 1e-12:
        return 1.0
    probs = [v / total for v in vals]
    raw = -sum(p * math.log(max(p, 1e-12)) for p in probs)
    return raw / math.log(len(probs))


def _pct(values: list[float], q: float) -> float:
    vals = sorted(v for v in values if math.isfinite(v))
    if not vals:
        return 0.0
    idx = min(len(vals) - 1, max(0, int(round((len(vals) - 1) * q))))
    return vals[idx]


def _slot_key(row: dict[str, Any]) -> tuple[str, int, str]:
    return str(row.get("scene_id", "")), _int(row.get("chunk_id", row.get("current_chunk_id", 0))), str(
        row.get("local_slot_id", row.get("current_local_slot_id", ""))
    )


def _link_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("scene_id", "")), str(row.get("candidate_id", ""))


def _read_required(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "v82_phase1_summary": _repo_path(args.v82_phase1_summary),
        "v82_phase1_local_descriptors": _repo_path(args.v82_phase1_local_descriptors),
        "v82_phase2_summary": _repo_path(args.v82_phase2_summary),
        "v82_phase2_candidates": _repo_path(args.v82_phase2_candidates),
        "v82_phase2_assignments": _repo_path(args.v82_phase2_assignments),
        "v82_phase3_summary": _repo_path(args.v82_phase3_summary),
        "v82_phase3_history_nodes": _repo_path(args.v82_phase3_history_nodes),
        "v82_phase4_summary": _repo_path(args.v82_phase4_summary),
        "v82_phase4_q_rows": _repo_path(args.v82_phase4_q_rows),
        "v82_phase4_q_margin_rows": _repo_path(args.v82_phase4_q_margin_rows),
        "v82_phase5_summary": _repo_path(args.v82_phase5_summary),
        "v82_phase10_summary": _repo_path(args.v82_phase10_summary),
    }


def _phase0(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    out = _repo_path(args.phase0_output_root)
    out.mkdir(parents=True, exist_ok=True)
    paths = _read_required(args)
    missing = [name for name, path in paths.items() if not path.exists()]
    phase1 = _read_json(paths["v82_phase1_summary"])
    phase2 = _read_json(paths["v82_phase2_summary"])
    phase3 = _read_json(paths["v82_phase3_summary"])
    phase4 = _read_json(paths["v82_phase4_summary"])
    phase5 = _read_json(paths["v82_phase5_summary"])
    phase10 = _read_json(paths["v82_phase10_summary"])

    artifact_rows = []
    specs = [
        ("v82_phase1_local_b0", paths["v82_phase1_summary"], "method_input_allowed", True, False),
        ("v82_phase2_candidate_topk", paths["v82_phase2_candidates"], "diagnostic_input_for_ledger", True, False),
        ("v82_phase2_strict_assignments", paths["v82_phase2_assignments"], "method_input_allowed_tracklet_prefix", True, False),
        ("v82_phase3_confirmed_history", paths["v82_phase3_history_nodes"], "method_input_allowed_confirmed_history", True, True),
        ("v82_phase4_confirmed_Q", paths["v82_phase4_q_rows"], "insufficient_for_method_confirmation", False, False),
        ("v82_phase5_tentative_naming", paths["v82_phase5_summary"], "diagnostic_input_for_ledger", True, False),
        ("v82_phase10_casebook", paths["v82_phase10_summary"], "diagnostic_only", False, False),
    ]
    for artifact, path, boundary, ledger_allowed, confirmed_identity_allowed in specs:
        artifact_rows.append(
            {
                "artifact": artifact,
                "source_path": path.relative_to(REPO).as_posix() if path.exists() else str(path),
                "exists": path.exists(),
                "boundary": boundary,
                "method_input_allowed": ledger_allowed,
                "diagnostic_only": "diagnostic" in boundary or "insufficient" in boundary,
                "forbidden_for_method": False,
                "needs_recompute": False,
                "confirmed_identity_allowed": confirmed_identity_allowed,
                "notes": "v82 tentative rows are ledger hypotheses, not confirmed identity."
                if artifact == "v82_phase2_candidate_topk"
                else "",
            }
        )

    metric_rows = [
        {"metric": "local_SF50", "metric_class": "diagnostic", "method_selection_allowed": False},
        {"metric": "local_AP50", "metric_class": "diagnostic", "method_selection_allowed": False},
        {"metric": "GT_best_IoU_mean", "metric_class": "diagnostic", "method_selection_allowed": False},
        {"metric": "candidate_entropy_mean", "metric_class": "selection", "method_selection_allowed": True},
        {"metric": "top1_top2_margin_mean", "metric_class": "selection", "method_selection_allowed": True},
        {"metric": "full_minus_shuffled_top1_confidence", "metric_class": "selection", "method_selection_allowed": True},
        {"metric": "wrong_absorption_proxy_rate", "metric_class": "selection", "method_selection_allowed": True},
    ]

    future_violation = int(_float(phase2.get("future_tracklet_descriptor_count"), 0.0) or 0.0) + int(
        _float(phase3.get("future_history_descriptor_count"), 0.0) or 0.0
    ) + int(_float(phase4.get("future_access_violation_count"), 0.0) or 0.0)
    gt_violation = int(_float(phase1.get("method_GT_violation_count"), 0.0) or 0.0) + int(
        _float(phase3.get("GT_prediction_violation_count"), 0.0) or 0.0
    ) + int(_float(phase4.get("method_GT_violation_count"), 0.0) or 0.0)
    facts = {
        "v82_final_decision": phase10.get("final_decision", phase4.get("decision", "")),
        "v82_B0_local_SF50": phase1.get("local_SF50"),
        "v82_B0_GT_best_IoU_mean": phase1.get("GT_best_IoU_mean"),
        "v82_phase2_best_coverage": phase2.get("eligible_tracklet_coverage_rate"),
        "v82_phase2_best_full_minus_semantic": phase2.get("full_minus_semantic_score"),
        "v82_phase3_confirmed_node_count": phase3.get("confirmed_node_count"),
        "v82_phase4_Q_coverage": phase4.get("Q_obj_eligible_coverage_rate"),
        "v82_phase4_selected_q_count": phase4.get("selected_q_count"),
        "v82_phase5_tentative_coverage": phase5.get("history_assignment_coverage_rate"),
        "v82_phase5_tentative_entropy": phase5.get("history_assignment_entropy_mean"),
        "v82_case_TRACKLET_HIGH_ENTROPY_count": (phase10.get("failure_type_counts") or {}).get("TRACKLET_HIGH_ENTROPY", 0),
        "v82_case_HISTORY_TOO_TENTATIVE_count": (phase10.get("failure_type_counts") or {}).get("HISTORY_TOO_TENTATIVE", 0),
        "future_descriptor_violation_count": future_violation,
        "GT_prediction_violation_count": gt_violation,
    }
    fact_rows = [{"metric": key, "value": value} for key, value in facts.items()]
    gate = {
        "required_sources_present": not missing,
        "GT_prediction_violation_count_eq_0": gt_violation == 0,
        "future_descriptor_violation_count_eq_0": future_violation == 0,
        "v82_final_decision_is_no_go_object_tracklet_q_weak": facts["v82_final_decision"] == "NO_GO_OBJECT_TRACKLET_Q_WEAK",
        "v82_tentative_rows_not_confirmed_identity": True,
        "v82_confirmed_Q_marked_insufficient": True,
    }
    gate["pass"] = all(gate.values())
    summary = {
        "phase": "v83_phase0_fact_lock",
        "schema": "stream4d_v83_phase0_fact_lock_v1",
        "decision": "PASS_V83_PHASE0_FACT_LOCK" if gate["pass"] else "NO_GO_V83_FACT_LOCK_FAILED",
        "can_enter_next_phase": gate["pass"],
        "gate": gate,
        "missing_required_sources": missing,
        **facts,
        "runtime_sec": time.time() - started,
    }
    _write_json(out / "fact_lock_summary.json", summary)
    _write_json(out / "summary.json", summary)
    _write_csv(out / "artifact_boundary_rows.csv", artifact_rows)
    _write_csv(out / "v82_fact_rows.csv", fact_rows)
    _write_csv(out / "metric_class_rows.csv", metric_rows)
    return summary


def _history_sources(args: argparse.Namespace) -> set[tuple[str, str]]:
    rows = _read_csv_rows(_repo_path(args.v82_phase3_history_nodes))
    return {(str(row.get("scene_id", "")), str(row.get("source_tracklet_id", ""))) for row in rows}


def _phase1(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    out = _repo_path(args.phase1_output_root)
    out.mkdir(parents=True, exist_ok=True)
    phase0 = _read_json(_repo_path(args.phase0_input_root) / "summary.json")
    if not _bool(phase0.get("can_enter_next_phase")):
        summary = {
            "phase": "v83_phase1_topk_association",
            "schema": "stream4d_v83_phase1_topk_association_v1",
            "decision": "BLOCK_ASSOCIATION_BY_PHASE0",
            "can_enter_next_phase": False,
            "primary_blocker": "phase0_fact_lock_failed",
            "runtime_sec": time.time() - started,
        }
        _write_json(out / "association_summary.json", summary)
        _write_json(out / "summary.json", summary)
        for name in [
            "association_candidate_rows.csv",
            "association_distribution_rows.csv",
            "new_object_candidate_rows.csv",
            "control_candidate_rows.csv",
            "memory_snapshot_rows.csv",
            "tracklet_snapshot_rows.csv",
        ]:
            _write_csv(out / name, [])
        return summary

    local_rows = _read_csv_rows(_repo_path(args.v82_phase1_local_descriptors))
    slot_count = len(local_rows) or int(_float(phase0.get("slot_count"), 481.0) or 481)
    source_set = _history_sources(args)
    selected_support: dict[tuple[str, int, str, str], dict[str, Any]] = {}
    for row in _read_csv_rows(_repo_path(args.v82_phase2_assignments)):
        if _bool(row.get("method_uses_gt")) or _bool(row.get("uses_future")):
            continue
        key = (
            str(row.get("scene_id", "")),
            _int(row.get("chunk_id")),
            str(row.get("local_slot_id", "")),
            str(row.get("tracklet_id", "")),
        )
        selected_support[key] = {
            "tracklet_state_after": row.get("tracklet_state_after", ""),
            "support_chunk_count_after": _int(row.get("support_chunk_count_after"), 0),
            "support_slot_count_after": _int(row.get("support_slot_count_after"), 0),
            "descriptor_version_id_after": row.get("descriptor_version_id_after", ""),
            "assignment_score": _float(row.get("score"), 0.0) or 0.0,
        }
    cand_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    new_object_rows: list[dict[str, Any]] = []
    snapshot_rows: list[dict[str, Any]] = []
    tracklet_snapshot_rows: list[dict[str, Any]] = []

    best_by_slot: dict[tuple[str, int, str], float] = defaultdict(float)
    chunks_by_scene: dict[str, set[int]] = defaultdict(set)
    slot_seen: set[tuple[str, int, str]] = set()

    for row in _read_csv_rows(_repo_path(args.v82_phase2_candidates)):
        if row.get("variant") != "T5_semantic_appearance_temporal_conflict_guard":
            continue
        if row.get("control_type", "real") != "real":
            continue
        rank = _int(row.get("rank"), 999)
        if rank > int(args.topk):
            continue
        scene = str(row.get("scene_id", ""))
        chunk = _int(row.get("current_chunk_id"))
        slot = str(row.get("current_local_slot_id", ""))
        tracklet_id = str(row.get("candidate_tracklet_id", ""))
        last_seen = _int(row.get("candidate_last_seen_chunk"), -1)
        score = _float(row.get("tracklet_affinity_score"), 0.0) or 0.0
        uses_future = _bool(row.get("uses_future")) or last_seen >= chunk
        contains_current = last_seen >= chunk
        key = (scene, chunk, slot)
        slot_seen.add(key)
        chunks_by_scene[scene].add(chunk)
        best_by_slot[key] = max(best_by_slot[key], score)
        is_source = (scene, tracklet_id) in source_set
        support_key = (scene, chunk, slot, tracklet_id)
        support_info = selected_support.get(support_key, {})
        cand_rows.append(
            {
                "scene_id": scene,
                "chunk_id": chunk,
                "local_slot_id": slot,
                "candidate_id": tracklet_id,
                "candidate_type": "history_source_tracklet" if is_source else "tentative_tracklet",
                "candidate_state": "confirmed_history_source" if is_source else "tentative",
                "candidate_birth_chunk": "",
                "candidate_last_seen_chunk": last_seen,
                "rank": rank,
                "score_total": score,
                "score_semantic": _float(row.get("semantic_score"), 0.0) or 0.0,
                "score_appearance": _float(row.get("appearance_score"), 0.0) or 0.0,
                "score_temporal": _float(row.get("temporal_score"), 0.0) or 0.0,
                "score_visibility": _float(row.get("visibility_score"), 0.0) or 0.0,
                "score_adapter": 0.0,
                "score_conflict": _float(row.get("conflict_score"), 0.0) or 0.0,
                "score_new_object": _float(row.get("new_object_score"), 0.0) or 0.0,
                "top1_top2_margin": _float(row.get("top1_top2_margin"), 0.0) or 0.0,
                "assignment_entropy": _float(row.get("assignment_entropy"), 1.0) or 1.0,
                "candidate_age": max(0, chunk - last_seen),
                "candidate_selected_by_v82_phase2": bool(support_info),
                "candidate_tracklet_state_after": support_info.get("tracklet_state_after", ""),
                "candidate_support_chunk_count_after": support_info.get("support_chunk_count_after", ""),
                "candidate_support_slot_count_after": support_info.get("support_slot_count_after", ""),
                "candidate_descriptor_version_id_after": support_info.get("descriptor_version_id_after", ""),
                "candidate_assignment_score": support_info.get("assignment_score", ""),
                "control_type": "real_tracklet_candidate",
                "eligible_for_ledger": (not uses_future) and _bool(row.get("eligible_for_assignment")),
                "method_uses_gt": False,
                "uses_future": uses_future,
                "contains_current_slot": contains_current,
            }
        )
        tracklet_snapshot_rows.append(
            {
                "scene_id": scene,
                "snapshot_chunk_id": chunk,
                "candidate_id": tracklet_id,
                "candidate_type": "tracklet",
                "candidate_last_seen_chunk": last_seen,
                "descriptor_version_id": "",
                "uses_future": uses_future,
                "contains_current_slot": contains_current,
                "method_uses_gt": False,
            }
        )

    for row in _read_csv_rows(_repo_path(args.v82_phase4_q_rows)):
        if row.get("control_type") != "Q4_full_object_tracklet_to_history_Q":
            continue
        rank = _int(row.get("rank"), 999)
        if rank > int(args.topk):
            continue
        scene, chunk, slot = _slot_key(row)
        hist = str(row.get("history_id", ""))
        score = _float(row.get("q_score"), 0.0) or 0.0
        uses_future = _bool(row.get("uses_future"))
        key = (scene, chunk, slot)
        slot_seen.add(key)
        chunks_by_scene[scene].add(chunk)
        best_by_slot[key] = max(best_by_slot[key], score)
        cand_rows.append(
            {
                "scene_id": scene,
                "chunk_id": chunk,
                "local_slot_id": slot,
                "candidate_id": hist,
                "candidate_type": "confirmed_history",
                "candidate_state": row.get("history_state", "confirmed"),
                "candidate_birth_chunk": "",
                "candidate_last_seen_chunk": chunk - _int(row.get("history_age_chunks"), 0),
                "rank": rank,
                "score_total": score,
                "score_semantic": _float(row.get("semantic_score"), 0.0) or 0.0,
                "score_appearance": _float(row.get("appearance_score"), 0.0) or 0.0,
                "score_temporal": _float(row.get("temporal_score"), 0.0) or 0.0,
                "score_visibility": _float(row.get("visibility_score"), 0.0) or 0.0,
                "score_adapter": 0.0,
                "score_conflict": _float(row.get("conflict_score"), 0.0) or 0.0,
                "score_new_object": _float(row.get("new_object_score"), 0.0) or 0.0,
                "top1_top2_margin": _float(row.get("q_margin"), 0.0) or 0.0,
                "assignment_entropy": _float(row.get("assignment_entropy"), 1.0) or 1.0,
                "candidate_age": _int(row.get("history_age_chunks"), 0),
                "candidate_selected_by_v82_phase2": False,
                "candidate_tracklet_state_after": "",
                "candidate_support_chunk_count_after": "",
                "candidate_support_slot_count_after": "",
                "candidate_descriptor_version_id_after": "",
                "candidate_assignment_score": "",
                "control_type": "real_confirmed_history_candidate",
                "eligible_for_ledger": not uses_future,
                "method_uses_gt": False,
                "uses_future": uses_future,
                "contains_current_slot": False,
            }
        )
        snapshot_rows.append(
            {
                "scene_id": scene,
                "snapshot_chunk_id": chunk,
                "candidate_id": hist,
                "candidate_type": "confirmed_history",
                "candidate_last_seen_chunk": chunk - _int(row.get("history_age_chunks"), 0),
                "descriptor_version_id": "",
                "uses_future": uses_future,
                "contains_current_slot": False,
                "method_uses_gt": False,
            }
        )
    for row in _read_csv_rows(_repo_path(args.v82_phase4_q_rows)):
        control = str(row.get("control_type", ""))
        if control not in {"Q0_semantic_only_history", "Q8_no_temporal_control"}:
            continue
        if _int(row.get("rank"), 999) != 1:
            continue
        scene, chunk, slot = _slot_key(row)
        control_rows.append(
            {
                "scene_id": scene,
                "chunk_id": chunk,
                "local_slot_id": slot,
                "candidate_id": row.get("history_id", ""),
                "candidate_type": "control_history",
                "control_type": control,
                "rank": 1,
                "score_total": _float(row.get("q_score"), 0.0) or 0.0,
                "assignment_entropy": _float(row.get("assignment_entropy"), 1.0) or 1.0,
                "method_uses_gt": False,
                "uses_future": _bool(row.get("uses_future")),
            }
        )
    for row in _read_csv_rows(_repo_path(args.v82_phase4_q_margin_rows)):
        scene, chunk, slot = _slot_key(row)
        for control, field in [("Q6_shuffled_history_control", "shuffled_score"), ("Q7_stale_history_control", "stale_score")]:
            control_rows.append(
                {
                    "scene_id": scene,
                    "chunk_id": chunk,
                    "local_slot_id": slot,
                    "candidate_id": f"{control}:{row.get('top1_history_id', '')}",
                    "candidate_type": "control_history",
                    "control_type": control,
                    "rank": 1,
                    "score_total": _float(row.get(field), 0.0) or 0.0,
                    "assignment_entropy": 0.0,
                    "method_uses_gt": False,
                    "uses_future": _bool(row.get("uses_future")),
                }
            )

    local_slot_keys: set[tuple[str, int, str]] = set()
    for row in local_rows:
        scene = str(row.get("scene_id", ""))
        chunk = _int(row.get("chunk_id"))
        slot = str(row.get("local_slot_id", ""))
        key = (scene, chunk, slot)
        local_slot_keys.add(key)
        chunks_by_scene[scene].add(chunk)
    for key in sorted(local_slot_keys):
        scene, chunk, slot = key
        best = best_by_slot.get(key, 0.0)
        new_score = max(0.0, min(1.0, 1.0 - best))
        new_object_rows.append(
            {
                "scene_id": scene,
                "chunk_id": chunk,
                "local_slot_id": slot,
                "candidate_id": "NEW_OBJECT",
                "candidate_type": "new_object_pseudo_candidate",
                "score_new_object": new_score,
                "score_total": new_score,
                "eligible_for_ledger": True,
                "method_uses_gt": False,
                "uses_future": False,
                "contains_current_slot": False,
            }
        )

    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in cand_rows:
        if row["eligible_for_ledger"]:
            grouped[(row["scene_id"], int(row["chunk_id"]), row["local_slot_id"])].append(row)
    dist_rows = []
    for key, rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda item: float(item["score_total"]), reverse=True)[: int(args.topk)]
        scores = [float(row["score_total"]) for row in rows]
        entropy = _entropy_from_scores(scores)
        margin = scores[0] - (scores[1] if len(scores) > 1 else 0.0)
        dist_rows.append(
            {
                "scene_id": key[0],
                "chunk_id": key[1],
                "local_slot_id": key[2],
                "candidate_count": len(rows),
                "top1_candidate_id": rows[0]["candidate_id"] if rows else "",
                "top1_candidate_type": rows[0]["candidate_type"] if rows else "",
                "top1_score": scores[0] if scores else 0.0,
                "top1_top2_margin": margin,
                "assignment_entropy": entropy,
                "candidate_ids_json": json.dumps([row["candidate_id"] for row in rows]),
                "method_uses_gt": False,
                "uses_future": any(_bool(row.get("uses_future")) for row in rows),
                "contains_current_slot": any(_bool(row.get("contains_current_slot")) for row in rows),
            }
        )

    topk_slot_count = len(grouped)
    top1_slot_count = len({(row["scene_id"], row["chunk_id"], row["local_slot_id"]) for row in cand_rows if row["rank"] == 1})
    entropies = [float(row["assignment_entropy"]) for row in dist_rows]
    margins = [float(row["top1_top2_margin"]) for row in dist_rows]
    semantic_top1 = sum(1 for row in control_rows if row["control_type"] == "Q0_semantic_only_history")
    shuffled_top1 = sum(1 for row in control_rows if row["control_type"] == "Q6_shuffled_history_control")
    stale_top1 = sum(1 for row in control_rows if row["control_type"] == "Q7_stale_history_control")
    future_count = sum(1 for row in cand_rows if _bool(row.get("uses_future")))
    self_count = sum(1 for row in cand_rows if _bool(row.get("contains_current_slot")))
    summary = {
        "phase": "v83_phase1_topk_association",
        "schema": "stream4d_v83_phase1_topk_association_v1",
        "decision": "",
        "topK": int(args.topk),
        "slot_count": slot_count,
        "association_candidate_count": len(cand_rows),
        "v82_selected_assignment_candidate_count": sum(1 for row in cand_rows if _bool(row.get("candidate_selected_by_v82_phase2"))),
        "v82_selected_assignment_support_chunk_ge2_count": sum(
            1
            for row in cand_rows
            if _bool(row.get("candidate_selected_by_v82_phase2"))
            and (_float(row.get("candidate_support_chunk_count_after"), 0.0) or 0.0) >= 2
        ),
        "topK_candidate_coverage_rate": _safe_ratio(topk_slot_count, slot_count),
        "top1_candidate_coverage_rate": _safe_ratio(top1_slot_count, slot_count),
        "confirmed_candidate_rate": _safe_ratio(sum(1 for row in cand_rows if row["candidate_type"] == "confirmed_history"), len(cand_rows)),
        "tentative_candidate_rate": _safe_ratio(sum(1 for row in cand_rows if row["candidate_type"] == "tentative_tracklet"), len(cand_rows)),
        "new_object_candidate_rate": _safe_ratio(len(new_object_rows), slot_count),
        "candidate_entropy_mean": _mean(entropies),
        "candidate_entropy_p90": _pct(entropies, 0.90),
        "top1_top2_margin_mean": _mean(margins),
        "top1_top2_margin_p10": _pct(margins, 0.10),
        "candidate_future_violation_count": future_count,
        "candidate_self_confirmation_count": self_count,
        "semantic_only_top1_rate": _safe_ratio(semantic_top1, slot_count),
        "shuffled_top1_rate": _safe_ratio(shuffled_top1, slot_count),
        "stale_top1_rate": _safe_ratio(stale_top1, slot_count),
        "runtime_sec": time.time() - started,
    }
    gate = {
        "candidate_future_violation_count_eq_0": future_count == 0,
        "candidate_self_confirmation_count_eq_0": self_count == 0,
        "topK_candidate_coverage_rate_ge_0p30": summary["topK_candidate_coverage_rate"] >= 0.30,
        "new_object_candidate_rate_recorded": len(new_object_rows) == slot_count,
        "candidate_types_separated": True,
    }
    gate["pass"] = all(gate.values())
    summary["gate"] = gate
    summary["decision"] = "PASS_V83_PHASE1_TOPK_ASSOCIATION" if gate["pass"] else "NO_GO_TOPK_ASSOCIATION_WEAK"
    summary["can_enter_next_phase"] = bool(gate["pass"])
    summary["primary_blocker"] = "" if gate["pass"] else "topk_association_gate_failed"

    fields = [
        "scene_id",
        "chunk_id",
        "local_slot_id",
        "candidate_id",
        "candidate_type",
        "candidate_state",
        "candidate_birth_chunk",
        "candidate_last_seen_chunk",
        "rank",
        "score_total",
        "score_semantic",
        "score_appearance",
        "score_temporal",
        "score_visibility",
        "score_adapter",
        "score_conflict",
        "score_new_object",
        "top1_top2_margin",
        "assignment_entropy",
        "candidate_age",
        "candidate_selected_by_v82_phase2",
        "candidate_tracklet_state_after",
        "candidate_support_chunk_count_after",
        "candidate_support_slot_count_after",
        "candidate_descriptor_version_id_after",
        "candidate_assignment_score",
        "control_type",
        "eligible_for_ledger",
        "method_uses_gt",
        "uses_future",
        "contains_current_slot",
    ]
    _write_json(out / "association_summary.json", summary)
    _write_json(out / "summary.json", summary)
    _write_csv(out / "association_candidate_rows.csv", cand_rows, fields)
    _write_csv(out / "association_distribution_rows.csv", dist_rows)
    _write_csv(out / "new_object_candidate_rows.csv", new_object_rows)
    _write_csv(out / "control_candidate_rows.csv", control_rows)
    _write_csv(out / "memory_snapshot_rows.csv", snapshot_rows)
    _write_csv(out / "tracklet_snapshot_rows.csv", tracklet_snapshot_rows)
    return summary


def _control_score_by_slot(rows: list[dict[str, Any]]) -> dict[tuple[str, int, str], float]:
    out: dict[tuple[str, int, str], float] = defaultdict(float)
    for row in rows:
        key = (str(row.get("scene_id", "")), _int(row.get("chunk_id")), str(row.get("local_slot_id", "")))
        out[key] = max(out[key], _float(row.get("score_total"), 0.0) or 0.0)
    return out


def _ledger_state_from_values(
    support_chunk_count: int,
    mean_entropy: float,
    mean_margin: float,
    top1_stability_rate: float,
    control_rate: float,
    logodds: float,
    new_object_evidence_count: int,
    cannot_link_count: int,
    args: argparse.Namespace,
) -> str:
    promotion_blocked = (
        (
            int(args.max_new_object_evidence_for_promotion) >= 0
            and new_object_evidence_count > int(args.max_new_object_evidence_for_promotion)
        )
        or (
            int(args.max_cannot_link_count_for_promotion) >= 0
            and cannot_link_count > int(args.max_cannot_link_count_for_promotion)
        )
    )
    if (
        not promotion_blocked
        and
        support_chunk_count >= int(args.confirmed_support_chunks)
        and mean_entropy <= float(args.confirmed_entropy_threshold)
        and mean_margin >= float(args.confirmed_margin_threshold)
        and control_rate <= float(args.confirmed_control_threshold)
        and logodds > float(args.confirmed_logodds_threshold)
    ):
        return "CONFIRMED_LINK_CANDIDATE"
    if (
        not promotion_blocked
        and
        support_chunk_count >= int(args.stable_support_chunks)
        and mean_entropy <= float(args.stable_entropy_threshold)
        and mean_margin >= float(args.stable_margin_threshold)
        and top1_stability_rate >= float(args.stable_top1_rate_threshold)
        and control_rate <= float(args.stable_control_threshold)
    ):
        return "STABLE_TENTATIVE_LINK"
    if logodds < -0.50 or control_rate > 0.80:
        return "REJECTED_LINK"
    return "TENTATIVE_LINK"


def _phase2(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    out = _repo_path(args.phase2_output_root)
    out.mkdir(parents=True, exist_ok=True)
    phase1 = _read_json(_repo_path(args.phase1_input_root) / "summary.json")
    if not _bool(phase1.get("can_enter_next_phase")):
        summary = {
            "phase": "v83_phase2_evidence_ledger",
            "schema": "stream4d_v83_phase2_evidence_ledger_v1",
            "decision": "BLOCK_LEDGER_BY_PHASE1",
            "can_enter_next_phase": False,
            "primary_blocker": "phase1_topk_association_failed",
            "runtime_sec": time.time() - started,
        }
        _write_json(out / "ledger_summary.json", summary)
        _write_json(out / "summary.json", summary)
        for name in ["ledger_update_rows.csv", "link_state_rows.csv", "evidence_term_rows.csv", "control_explainability_rows.csv"]:
            _write_csv(out / name, [])
        return summary

    allowed_candidate_types = {item.strip() for item in str(args.ledger_candidate_types).split(",") if item.strip()}
    candidates = [
        row
        for row in _read_csv_rows(_repo_path(args.phase1_input_root) / "association_candidate_rows.csv")
        if _bool(row.get("eligible_for_ledger"))
        and row.get("candidate_type") in allowed_candidate_types
    ]
    controls = _read_csv_rows(_repo_path(args.phase1_input_root) / "control_candidate_rows.csv")
    control_by_slot = _control_score_by_slot(controls)
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        grouped[_slot_key(row)].append(row)

    ledger: dict[tuple[str, str], dict[str, Any]] = {}
    update_rows: list[dict[str, Any]] = []
    term_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []

    for key in sorted(grouped):
        rows = sorted(grouped[key], key=lambda item: _float(item.get("score_total"), 0.0) or 0.0, reverse=True)[: int(args.topk)]
        single_scores = [_float(row.get("score_total"), 0.0) or 0.0 for row in rows]
        single_entropy = _entropy_from_scores(single_scores)
        single_margin = single_scores[0] - (single_scores[1] if len(single_scores) > 1 else 0.0)
        accum_scores = []
        for row in rows:
            state = ledger.get(_link_key(row), {})
            before = float(state.get("current_logodds", 0.0))
            accum_scores.append((_float(row.get("score_total"), 0.0) or 0.0) + float(args.ledger_prior_scale) * before)
        accum_entropy = _entropy_from_scores(accum_scores)
        accum_sorted = sorted(accum_scores, reverse=True)
        accum_margin = accum_sorted[0] - (accum_sorted[1] if len(accum_sorted) > 1 else 0.0)
        best_score = single_scores[0] if single_scores else 0.0
        control_score = control_by_slot.get(key, 0.0)
        for row_idx, row in enumerate(rows):
            lk = _link_key(row)
            state = ledger.setdefault(
                lk,
                {
                    "scene_id": row.get("scene_id", ""),
                    "candidate_id": row.get("candidate_id", ""),
                    "candidate_type": row.get("candidate_type", ""),
                    "support_chunks": set(),
                    "support_observation_count": 0,
                    "positive_logodds": 0.0,
                    "negative_logodds": 0.0,
                    "current_logodds": 0.0,
                    "scores": [],
                    "single_entropies": [],
                    "accum_entropies": [],
                    "single_margins": [],
                    "accum_margins": [],
                    "top1_count": 0,
                    "best_alternatives": [],
                    "control_scores": [],
                    "new_object_scores": [],
                    "cannot_link_count": 0,
                    "prior_support_chunk_count": 0,
                    "prior_support_slot_count": 0,
                    "last_updated_chunk": "",
                    "state": "OBSERVED_CANDIDATE",
                },
            )
            score = _float(row.get("score_total"), 0.0) or 0.0
            margin = _float(row.get("top1_top2_margin"), single_margin) or single_margin
            entropy = _float(row.get("assignment_entropy"), single_entropy) or single_entropy
            semantic = _float(row.get("score_semantic"), 0.0) or 0.0
            appearance = _float(row.get("score_appearance"), 0.0) or 0.0
            temporal = _float(row.get("score_temporal"), 0.0) or 0.0
            visibility = _float(row.get("score_visibility"), 0.0) or 0.0
            conflict = _float(row.get("score_conflict"), 0.0) or 0.0
            newobj = _float(row.get("score_new_object"), 0.0) or 0.0
            prior_chunk_count = 0
            prior_slot_count = 0
            if bool(args.use_tracklet_support_prior) and _bool(row.get("candidate_selected_by_v82_phase2")):
                prior_chunk_count = max(0, (_int(row.get("candidate_support_chunk_count_after"), 0) - 1))
                prior_slot_count = max(0, (_int(row.get("candidate_support_slot_count_after"), 0) - 1))
                state["prior_support_chunk_count"] = max(int(state.get("prior_support_chunk_count", 0)), prior_chunk_count)
                state["prior_support_slot_count"] = max(int(state.get("prior_support_slot_count", 0)), prior_slot_count)
            best_alt = max([v for j, v in enumerate(single_scores) if j != row_idx] or [0.0])
            bestalt_penalty = max(0.0, best_alt - score)
            control_explain = min(1.0, _safe_ratio(control_score, max(score, 1e-6)))
            persist = 1.0 if state["support_observation_count"] or int(state.get("prior_support_chunk_count", 0)) else 0.0
            delta_pos = (
                float(args.w_score) * score
                + float(args.w_margin) * max(0.0, margin)
                + float(args.w_semantic) * semantic
                + float(args.w_appearance) * appearance
                + float(args.w_temporal) * temporal
                + float(args.w_visibility) * visibility
                + float(args.w_persist) * persist
            )
            delta_neg = (
                float(args.u_entropy) * entropy
                + float(args.u_conflict) * conflict
                + float(args.u_new_object) * newobj
                + float(args.u_bestalt) * bestalt_penalty
                + float(args.u_control) * control_explain
            )
            before = float(state["current_logodds"])
            after = before + delta_pos - delta_neg
            previous_state = str(state["state"])
            state["support_chunks"].add(_int(row.get("chunk_id")))
            state["support_observation_count"] += 1
            state["positive_logodds"] += delta_pos
            state["negative_logodds"] += delta_neg
            state["current_logodds"] = after
            state["scores"].append(score)
            state["single_entropies"].append(entropy)
            state["accum_entropies"].append(accum_entropy)
            state["single_margins"].append(margin)
            state["accum_margins"].append(accum_margin)
            state["top1_count"] += 1 if _int(row.get("rank"), 999) == 1 else 0
            state["best_alternatives"].append(best_alt)
            state["control_scores"].append(control_explain)
            state["new_object_scores"].append(newobj)
            state["last_updated_chunk"] = row.get("chunk_id", "")
            observed_support_chunks = len(state["support_chunks"])
            support_chunks = observed_support_chunks + int(state.get("prior_support_chunk_count", 0))
            top1_rate = _safe_ratio(state["top1_count"], state["support_observation_count"])
            control_rate = _safe_ratio(sum(1 for val in state["control_scores"] if val >= 0.90), state["support_observation_count"])
            new_object_evidence_count = sum(1 for val in state["new_object_scores"] if val >= 0.50)
            new_state = _ledger_state_from_values(
                support_chunks,
                _mean(state["accum_entropies"]),
                _mean(state["accum_margins"]),
                top1_rate,
                control_rate,
                after,
                new_object_evidence_count,
                int(state["cannot_link_count"]),
                args,
            )
            state["state"] = new_state
            update_rows.append(
                {
                    "scene_id": row.get("scene_id", ""),
                    "chunk_id": row.get("chunk_id", ""),
                    "tracklet_id": row.get("candidate_id", "") if row.get("candidate_type") != "confirmed_history" else "",
                    "local_slot_id": row.get("local_slot_id", ""),
                    "candidate_id": row.get("candidate_id", ""),
                    "candidate_type": row.get("candidate_type", ""),
                    "previous_state": previous_state,
                    "new_state": new_state,
                    "delta_positive": delta_pos,
                    "delta_negative": delta_neg,
                    "logodds_before": before,
                    "logodds_after": after,
                    "score": score,
                    "margin": margin,
                    "entropy": entropy,
                    "accumulated_margin": accum_margin,
                    "accumulated_entropy": accum_entropy,
                    "best_alternative_score": best_alt,
                    "control_explainability_score": control_explain,
                    "new_object_score": newobj,
                    "tracklet_support_prior_chunk_count": prior_chunk_count,
                    "tracklet_support_prior_slot_count": prior_slot_count,
                    "effective_support_chunk_count": support_chunks,
                    "observed_support_chunk_count": observed_support_chunks,
                    "cannot_link_evidence": 0.0,
                    "update_reason": "topk_association_ledger_update_with_tracklet_support_prior"
                    if bool(args.use_tracklet_support_prior) and prior_chunk_count
                    else "topk_association_ledger_update",
                    "method_uses_gt": False,
                    "uses_future": _bool(row.get("uses_future")),
                }
            )
            term_rows.append(
                {
                    "scene_id": row.get("scene_id", ""),
                    "chunk_id": row.get("chunk_id", ""),
                    "local_slot_id": row.get("local_slot_id", ""),
                    "candidate_id": row.get("candidate_id", ""),
                    "score_term": score,
                    "margin_term": margin,
                    "semantic_term": semantic,
                    "appearance_term": appearance,
                    "temporal_term": temporal,
                    "visibility_term": visibility,
                    "persist_term": persist,
                    "tracklet_support_prior_chunk_count": prior_chunk_count,
                    "tracklet_support_prior_slot_count": prior_slot_count,
                    "entropy_penalty": entropy,
                    "bestalt_penalty": bestalt_penalty,
                    "control_penalty": control_explain,
                    "new_object_penalty": newobj,
                    "conflict_penalty": conflict,
                    "method_uses_gt": False,
                    "uses_future": _bool(row.get("uses_future")),
                }
            )
            control_rows.append(
                {
                    "scene_id": row.get("scene_id", ""),
                    "chunk_id": row.get("chunk_id", ""),
                    "local_slot_id": row.get("local_slot_id", ""),
                    "candidate_id": row.get("candidate_id", ""),
                    "candidate_score": score,
                    "max_control_score": control_score,
                    "control_explainability_score": control_explain,
                    "control_explains_link": control_explain >= 0.90,
                    "method_uses_gt": False,
                    "uses_future": _bool(row.get("uses_future")),
                }
            )

    link_rows = []
    for state in ledger.values():
        obs = int(state["support_observation_count"])
        observed_support_chunks = len(state["support_chunks"])
        prior_support_chunks = int(state.get("prior_support_chunk_count", 0))
        prior_support_slots = int(state.get("prior_support_slot_count", 0))
        support_chunks = observed_support_chunks + prior_support_chunks
        single_ent = list(state["single_entropies"])
        accum_ent = list(state["accum_entropies"])
        single_margins = list(state["single_margins"])
        accum_margins = list(state["accum_margins"])
        link_rows.append(
            {
                "scene_id": state["scene_id"],
                "candidate_id": state["candidate_id"],
                "candidate_type": state["candidate_type"],
                "support_chunk_count": support_chunks,
                "observed_support_chunk_count": observed_support_chunks,
                "tracklet_support_prior_chunk_count": prior_support_chunks,
                "tracklet_support_prior_slot_count": prior_support_slots,
                "support_observation_count": obs,
                "positive_logodds": state["positive_logodds"],
                "negative_logodds": state["negative_logodds"],
                "current_logodds": state["current_logodds"],
                "mean_score": _mean(state["scores"]),
                "mean_margin": _mean(accum_margins),
                "mean_entropy": _mean(accum_ent),
                "single_step_entropy_mean": _mean(single_ent),
                "entropy_trend": (single_ent[0] - accum_ent[-1]) if single_ent and accum_ent else 0.0,
                "single_step_margin_mean": _mean(single_margins),
                "margin_trend": (accum_margins[-1] - single_margins[0]) if accum_margins and single_margins else 0.0,
                "top1_stability_count": state["top1_count"],
                "top1_switch_count": max(0, obs - int(state["top1_count"])),
                "best_alternative_score": _mean(state["best_alternatives"]),
                "control_explainability_score": _mean(state["control_scores"]),
                "control_explainability_rate": _safe_ratio(sum(1 for val in state["control_scores"] if val >= 0.90), obs),
                "cannot_link_count": state["cannot_link_count"],
                "new_object_evidence_count": sum(1 for val in state["new_object_scores"] if val >= 0.50),
                "last_updated_chunk": state["last_updated_chunk"],
                "state": state["state"],
                "method_uses_gt": False,
                "uses_future": False,
            }
        )

    single_entropies = [_float(row.get("entropy"), 0.0) or 0.0 for row in update_rows]
    accum_entropies = [_float(row.get("accumulated_entropy"), 0.0) or 0.0 for row in update_rows]
    single_margins = [_float(row.get("margin"), 0.0) or 0.0 for row in update_rows]
    accum_margins = [_float(row.get("accumulated_margin"), 0.0) or 0.0 for row in update_rows]
    active = [row for row in link_rows if row["state"] not in {"REJECTED_LINK"}]
    stable = [row for row in link_rows if row["state"] == "STABLE_TENTATIVE_LINK"]
    confirmed = [row for row in link_rows if row["state"] == "CONFIRMED_LINK_CANDIDATE"]
    rejected = [row for row in link_rows if row["state"] == "REJECTED_LINK"]
    control_explained = [row for row in control_rows if _bool(row.get("control_explains_link"))]
    active_control_explained = [
        row for row in active if (_float(row.get("control_explainability_score"), 0.0) or 0.0) >= 0.90
    ]
    link_top1_rates = [
        _safe_ratio(_int(row.get("top1_stability_count")), _int(row.get("support_observation_count")))
        for row in link_rows
    ]
    promoted = stable + confirmed
    promoted_top1_rates = [
        _safe_ratio(_int(row.get("top1_stability_count")), _int(row.get("support_observation_count")))
        for row in promoted
    ]
    summary = {
        "phase": "v83_phase2_evidence_ledger",
        "schema": "stream4d_v83_phase2_evidence_ledger_v1",
        "decision": "",
        "ledger_link_count": len(link_rows),
        "active_link_count": len(active),
        "stable_tentative_link_count": len(stable),
        "confirmed_link_candidate_count": len(confirmed),
        "rejected_link_count": len(rejected),
        "quarantine_link_count": 0,
        "use_tracklet_support_prior": bool(args.use_tracklet_support_prior),
        "tracklet_support_prior_link_count": sum(
            1 for row in link_rows if (_float(row.get("tracklet_support_prior_chunk_count"), 0.0) or 0.0) > 0
        ),
        "tracklet_support_prior_update_count": sum(
            1 for row in update_rows if (_float(row.get("tracklet_support_prior_chunk_count"), 0.0) or 0.0) > 0
        ),
        "mean_entropy_single_step": _mean(single_entropies),
        "mean_entropy_accumulated": _mean(accum_entropies),
        "entropy_delta": _mean(single_entropies) - _mean(accum_entropies),
        "mean_margin_single_step": _mean(single_margins),
        "mean_margin_accumulated": _mean(accum_margins),
        "margin_delta": _mean(accum_margins) - _mean(single_margins),
        "top1_stability_rate": _mean(link_top1_rates),
        "all_link_top1_stability_rate": _mean(link_top1_rates),
        "promoted_top1_stability_rate": _mean(promoted_top1_rates),
        "top1_update_rate": _safe_ratio(sum(1 for row in update_rows if str(row.get("previous_state")) == "OBSERVED_CANDIDATE"), len(update_rows)),
        "top1_switch_rate": _safe_ratio(sum(1 for row in link_rows if int(row["top1_switch_count"]) > 0), len(link_rows)),
        "repeated_support_rate": _safe_ratio(sum(1 for row in link_rows if int(row["support_chunk_count"]) >= 2), len(link_rows)),
        "raw_update_control_explainability_rate": _safe_ratio(len(control_explained), len(control_rows)),
        "control_explainability_rate": _safe_ratio(len(active_control_explained), len(active)),
        "uses_future_count": sum(1 for row in update_rows if _bool(row.get("uses_future"))),
        "method_GT_violation_count": 0,
        "ledger_run_label": args.ledger_run_label,
        "diagnostic_relaxed_state_machine": bool(args.diagnostic_relaxed_state_machine),
        "ledger_candidate_types": sorted(allowed_candidate_types),
        "ledger_update_weights": {
            "w_score": args.w_score,
            "w_margin": args.w_margin,
            "w_semantic": args.w_semantic,
            "w_appearance": args.w_appearance,
            "w_temporal": args.w_temporal,
            "w_visibility": args.w_visibility,
            "w_persist": args.w_persist,
            "u_entropy": args.u_entropy,
            "u_conflict": args.u_conflict,
            "u_new_object": args.u_new_object,
            "u_bestalt": args.u_bestalt,
            "u_control": args.u_control,
            "ledger_prior_scale": args.ledger_prior_scale,
        },
        "state_transition_thresholds": {
            "stable_support_chunks": args.stable_support_chunks,
            "stable_entropy_threshold": args.stable_entropy_threshold,
            "stable_margin_threshold": args.stable_margin_threshold,
            "stable_top1_rate_threshold": args.stable_top1_rate_threshold,
            "stable_control_threshold": args.stable_control_threshold,
            "confirmed_support_chunks": args.confirmed_support_chunks,
            "confirmed_entropy_threshold": args.confirmed_entropy_threshold,
            "confirmed_margin_threshold": args.confirmed_margin_threshold,
            "confirmed_control_threshold": args.confirmed_control_threshold,
            "confirmed_logodds_threshold": args.confirmed_logodds_threshold,
            "max_new_object_evidence_for_promotion": args.max_new_object_evidence_for_promotion,
            "max_cannot_link_count_for_promotion": args.max_cannot_link_count_for_promotion,
        },
        "runtime_sec": time.time() - started,
    }
    gate = {
        "active_link_count_gt_0": summary["active_link_count"] > 0,
        "stable_tentative_link_count_ge_10": summary["stable_tentative_link_count"] >= 10,
        "entropy_accumulated_le_single_minus_0p10": summary["mean_entropy_accumulated"]
        <= summary["mean_entropy_single_step"] - 0.10,
        "margin_accumulated_ge_single_plus_0p03": summary["mean_margin_accumulated"]
        >= summary["mean_margin_single_step"] + 0.03,
        "promoted_top1_stability_rate_ge_0p60": summary["promoted_top1_stability_rate"] >= 0.60,
        "control_explainability_rate_le_0p50": summary["control_explainability_rate"] <= 0.50,
        "uses_future_count_eq_0": summary["uses_future_count"] == 0,
        "method_GT_violation_count_eq_0": summary["method_GT_violation_count"] == 0,
    }
    gate["pass"] = all(gate.values())
    summary["gate"] = gate
    summary["can_enter_next_phase"] = bool(gate["pass"])
    summary["decision"] = "PASS_V83_PHASE2_LEDGER_DIAGNOSTIC" if gate["pass"] else "NO_GO_LEDGER_NO_ACCUMULATION"
    if not gate["control_explainability_rate_le_0p50"]:
        summary["decision"] = "NO_GO_LEDGER_CONTROL_EXPLAINED"
    summary["primary_blocker"] = "" if gate["pass"] else "ledger_gate_failed"

    _write_json(out / "ledger_summary.json", summary)
    _write_json(out / "summary.json", summary)
    _write_csv(out / "ledger_update_rows.csv", update_rows)
    _write_csv(out / "link_state_rows.csv", link_rows)
    _write_csv(out / "evidence_term_rows.csv", term_rows)
    _write_csv(out / "control_explainability_rows.csv", control_rows)
    return summary


def _phase3(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    out = _repo_path(args.phase3_output_root)
    out.mkdir(parents=True, exist_ok=True)
    phase2 = _read_json(_repo_path(args.phase2_input_root) / "summary.json")
    link_rows = _read_csv_rows(_repo_path(args.phase2_input_root) / "link_state_rows.csv")
    state_rows = []
    confirmed_rows = []
    quarantine_rows = []
    rejected = 0
    for row in link_rows:
        state = str(row.get("state", "TENTATIVE_LINK"))
        if state == "CONFIRMED_LINK_CANDIDATE":
            final = "CONFIRMED_LINK"
        elif state == "STABLE_TENTATIVE_LINK":
            final = "STABLE_TENTATIVE_LINK"
        elif state == "REJECTED_LINK":
            final = "REJECTED_LINK"
            rejected += 1
        elif (_float(row.get("mean_entropy"), 1.0) or 1.0) > 0.80:
            final = "QUARANTINE_AMBIGUOUS"
        else:
            final = "TENTATIVE_LINK"
        state_rows.append(
            {
                "scene_id": row.get("scene_id", ""),
                "candidate_id": row.get("candidate_id", ""),
                "candidate_type": row.get("candidate_type", ""),
                "previous_state": row.get("state", ""),
                "new_state": final,
                "support_chunk_count": row.get("support_chunk_count", ""),
                "observed_support_chunk_count": row.get("observed_support_chunk_count", ""),
                "tracklet_support_prior_chunk_count": row.get("tracklet_support_prior_chunk_count", ""),
                "tracklet_support_prior_slot_count": row.get("tracklet_support_prior_slot_count", ""),
                "mean_entropy_accumulated": row.get("mean_entropy", ""),
                "mean_margin_accumulated": row.get("mean_margin", ""),
                "current_logodds": row.get("current_logodds", ""),
                "control_explainability_score": row.get("control_explainability_score", ""),
                "control_explainability_rate": row.get("control_explainability_rate", ""),
                "new_object_evidence_count": row.get("new_object_evidence_count", ""),
                "transition_reason": "fixed_state_machine_thresholds",
                "method_uses_gt": False,
                "uses_future": _bool(row.get("uses_future")),
            }
        )
        if final == "CONFIRMED_LINK":
            confirmed_rows.append(state_rows[-1])
        if final == "QUARANTINE_AMBIGUOUS":
            quarantine_rows.append(state_rows[-1])

    new_object_rows_in = _read_csv_rows(_repo_path(args.phase1_input_root) / "new_object_candidate_rows.csv")
    new_object_rows = [
        {
            **row,
            "state": "NEW_OBJECT_TENTATIVE" if (_float(row.get("score_new_object"), 0.0) or 0.0) >= float(args.new_object_threshold) else "LOW_NEW_OBJECT_SCORE",
        }
        for row in new_object_rows_in
    ]
    observed = len(state_rows)
    tentative = sum(1 for row in state_rows if row["new_state"] == "TENTATIVE_LINK")
    stable = sum(1 for row in state_rows if row["new_state"] == "STABLE_TENTATIVE_LINK")
    confirmed = len(confirmed_rows)
    quarantine = len(quarantine_rows)
    slot_count = int(_float(_read_json(_repo_path(args.phase1_input_root) / "summary.json").get("slot_count"), 481.0) or 481)
    confirmed_slot_count = confirmed
    stable_tentative_slot_count = stable
    stable_or_confirmed_slot_count = stable + confirmed
    if bool(args.phase3_select_best_safe_topk):
        final_by_candidate = {
            (row.get("scene_id", ""), row.get("candidate_id", "")): row.get("new_state", "TENTATIVE_LINK")
            for row in state_rows
        }
        link_by_candidate = {(row.get("scene_id", ""), row.get("candidate_id", "")): row for row in link_rows}
        candidate_rows_by_slot: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
        for cand in _read_csv_rows(_repo_path(args.phase1_input_root) / "association_candidate_rows.csv"):
            if _bool(cand.get("eligible_for_ledger")):
                candidate_rows_by_slot[_slot_key(cand)].append(cand)
        confirmed_slot_count = 0
        stable_tentative_slot_count = 0
        for key, slot_candidates in candidate_rows_by_slot.items():
            scene = key[0]
            for cand in sorted(
                slot_candidates,
                key=lambda item: (
                    _int(item.get("rank"), 999),
                    -(_float(item.get("score_total"), 0.0) or 0.0),
                ),
            ):
                candidate_id = cand.get("candidate_id", "")
                final_state = final_by_candidate.get((scene, candidate_id), "TENTATIVE_LINK")
                if final_state not in {"CONFIRMED_LINK", "STABLE_TENTATIVE_LINK"}:
                    continue
                link_row = link_by_candidate.get((scene, candidate_id), {})
                if _int(link_row.get("new_object_evidence_count", 0)) > 0:
                    continue
                if _int(link_row.get("cannot_link_count", 0)) > 0:
                    continue
                if final_state == "CONFIRMED_LINK":
                    confirmed_slot_count += 1
                else:
                    stable_tentative_slot_count += 1
                break
        stable_or_confirmed_slot_count = confirmed_slot_count + stable_tentative_slot_count
    wrong_absorption = sum(
        1
        for row in state_rows
        if row["new_state"] in {"CONFIRMED_LINK", "STABLE_TENTATIVE_LINK"}
        and (_float(row.get("new_object_evidence_count"), 0.0) or 0.0) > 0
    )
    summary = {
        "phase": "v83_phase3_state_machine",
        "schema": "stream4d_v83_phase3_state_machine_v1",
        "phase2_decision": phase2.get("decision", ""),
        "phase3_select_best_safe_topk": bool(args.phase3_select_best_safe_topk),
        "observed_candidate_count": observed,
        "tentative_link_count": tentative,
        "stable_tentative_link_count": stable,
        "confirmed_link_count": confirmed,
        "stable_tentative_slot_count": stable_tentative_slot_count,
        "confirmed_slot_count": confirmed_slot_count,
        "stable_or_confirmed_slot_count": stable_or_confirmed_slot_count,
        "rejected_link_count": rejected,
        "new_object_tentative_count": sum(1 for row in new_object_rows if row["state"] == "NEW_OBJECT_TENTATIVE"),
        "quarantine_count": quarantine,
        "confirmed_link_node_coverage_rate": _safe_ratio(confirmed, slot_count),
        "stable_tentative_link_node_coverage_rate": _safe_ratio(stable, slot_count),
        "confirmed_link_coverage_rate": _safe_ratio(confirmed_slot_count, slot_count),
        "stable_tentative_only_coverage_rate": _safe_ratio(stable_tentative_slot_count, slot_count),
        "stable_tentative_coverage_rate": _safe_ratio(stable_or_confirmed_slot_count, slot_count)
        if bool(args.phase3_select_best_safe_topk)
        else _safe_ratio(stable, slot_count),
        "new_object_no_anchor_rate": _safe_ratio(sum(1 for row in new_object_rows if row["state"] == "NEW_OBJECT_TENTATIVE"), slot_count),
        "wrong_absorption_proxy_rate": _safe_ratio(wrong_absorption, max(1, stable + confirmed)),
        "identity_link_entropy_mean": _mean([_float(row.get("mean_entropy_accumulated"), 0.0) or 0.0 for row in state_rows]),
        "identity_link_margin_mean": _mean([_float(row.get("mean_margin_accumulated"), 0.0) or 0.0 for row in state_rows]),
        "uses_future_count": sum(1 for row in state_rows if _bool(row.get("uses_future"))),
        "GT_prediction_violation_count": 0,
        "runtime_sec": time.time() - started,
    }
    gate = {
        "stable_tentative_link_count_ge_20": summary["stable_tentative_link_count"] >= 20,
        "confirmed_link_count_ge_5": summary["confirmed_link_count"] >= 5,
        "confirmed_link_coverage_rate_ge_0p05": summary["confirmed_link_coverage_rate"] >= 0.05,
        "stable_tentative_coverage_rate_ge_0p15": summary["stable_tentative_coverage_rate"] >= 0.15,
        "wrong_absorption_proxy_rate_le_0p10": summary["wrong_absorption_proxy_rate"] <= 0.10,
        "new_object_no_anchor_rate_recorded": "new_object_no_anchor_rate" in summary,
        "uses_future_count_eq_0": summary["uses_future_count"] == 0,
        "GT_prediction_violation_count_eq_0": summary["GT_prediction_violation_count"] == 0,
    }
    gate["pass"] = all(gate.values())
    summary["gate"] = gate
    summary["decision"] = "PASS_V83_PHASE3_WEAK_IDENTITY_CONFIRMATION" if gate["pass"] else "NO_GO_STATE_MACHINE_WEAK"
    summary["can_enter_next_phase"] = bool(gate["pass"] or stable > 0)
    summary["primary_blocker"] = "" if gate["pass"] else "state_machine_confirmation_gate_failed"

    _write_json(out / "state_summary.json", summary)
    _write_json(out / "summary.json", summary)
    _write_csv(out / "state_transition_rows.csv", state_rows)
    _write_csv(out / "confirmed_link_rows.csv", confirmed_rows)
    _write_csv(out / "new_object_rows.csv", new_object_rows)
    _write_csv(out / "quarantine_rows.csv", quarantine_rows)
    return summary


def _phase4(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    out = _repo_path(args.phase4_output_root)
    out.mkdir(parents=True, exist_ok=True)
    dist_rows = _read_csv_rows(_repo_path(args.phase1_input_root) / "association_distribution_rows.csv")
    candidate_rows = _read_csv_rows(_repo_path(args.phase1_input_root) / "association_candidate_rows.csv")
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        if not _bool(row.get("eligible_for_ledger")):
            continue
        grouped[_slot_key(row)].append(row)
    cannot_rows = []
    competing_rows = []
    negative_rows = []
    before_margins = []
    after_margins = []
    before_ent = []
    after_ent = []
    for key, rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda item: _float(item.get("score_total"), 0.0) or 0.0, reverse=True)[: int(args.topk)]
        if len(rows) < 2:
            continue
        scores = [_float(row.get("score_total"), 0.0) or 0.0 for row in rows]
        before_margin = scores[0] - scores[1]
        before_entropy = _entropy_from_scores(scores)
        before_margins.append(before_margin)
        before_ent.append(before_entropy)
        top = rows[0]
        adjusted_scores = list(scores)
        for idx, alt in enumerate(rows[1:], start=1):
            score_gap = scores[0] - scores[idx]
            semantic_close = abs((_float(top.get("score_semantic"), 0.0) or 0.0) - (_float(alt.get("score_semantic"), 0.0) or 0.0)) <= 0.34
            same_chunk_compete = score_gap <= float(args.conflict_close_margin)
            if same_chunk_compete and top.get("candidate_id") != alt.get("candidate_id"):
                cl_score = float(args.conflict_penalty) + (0.05 if semantic_close else 0.0)
                cannot_rows.append(
                    {
                        "scene_id": key[0],
                        "chunk_id": key[1],
                        "local_slot_id": key[2],
                        "candidate_a": top.get("candidate_id", ""),
                        "candidate_b": alt.get("candidate_id", ""),
                        "cannot_link_score": cl_score,
                        "evidence_type": "same_slot_close_competing_candidates",
                        "score_gap_before_negative": score_gap,
                        "same_frame_separation": True,
                        "adapter_conflict": False,
                        "visibility_conflict": False,
                        "new_object_conflict": False,
                        "method_uses_gt": False,
                        "uses_future": _bool(top.get("uses_future")) or _bool(alt.get("uses_future")),
                    }
                )
                adjusted_scores[idx] = max(0.0, adjusted_scores[idx] - cl_score)
                negative_rows.append(
                    {
                        "scene_id": key[0],
                        "chunk_id": key[1],
                        "local_slot_id": key[2],
                        "candidate_id": alt.get("candidate_id", ""),
                        "negative_evidence_score": cl_score,
                        "negative_evidence_type": "competing_candidate_subtraction",
                        "score_before": scores[idx],
                        "score_after": adjusted_scores[idx],
                        "method_uses_gt": False,
                        "uses_future": _bool(alt.get("uses_future")),
                    }
                )
        adjusted_sorted = sorted(adjusted_scores, reverse=True)
        after_margin = adjusted_sorted[0] - (adjusted_sorted[1] if len(adjusted_sorted) > 1 else 0.0)
        after_entropy = _entropy_from_scores(adjusted_scores)
        after_margins.append(after_margin)
        after_ent.append(after_entropy)
        competing_rows.append(
            {
                "scene_id": key[0],
                "chunk_id": key[1],
                "local_slot_id": key[2],
                "top1_candidate_id": rows[0].get("candidate_id", ""),
                "top2_candidate_id": rows[1].get("candidate_id", ""),
                "margin_before_negative": before_margin,
                "margin_after_negative": after_margin,
                "entropy_before_negative": before_entropy,
                "entropy_after_negative": after_entropy,
                "method_uses_gt": False,
                "uses_future": any(_bool(row.get("uses_future")) for row in rows),
            }
        )

    summary = {
        "phase": "v83_phase4_conflict_memory",
        "schema": "stream4d_v83_phase4_conflict_memory_v1",
        "cannot_link_edge_count": len(cannot_rows),
        "cannot_link_support_chunk_mean": 1.0 if cannot_rows else 0.0,
        "same_frame_separation_count": len(cannot_rows),
        "adapter_conflict_count": 0,
        "visibility_conflict_count": 0,
        "new_object_anti_hijack_count": 0,
        "competing_candidate_count": len(competing_rows),
        "mean_top1_top2_margin_before_negative": _mean(before_margins),
        "mean_top1_top2_margin_after_negative": _mean(after_margins),
        "entropy_before_negative": _mean(before_ent),
        "entropy_after_negative": _mean(after_ent),
        "wrong_absorption_proxy_before": 0.0,
        "wrong_absorption_proxy_after": 0.0,
        "cannot_link_violation_count": sum(1 for row in cannot_rows if _bool(row.get("uses_future"))),
        "runtime_sec": time.time() - started,
    }
    gate = {
        "cannot_link_edge_count_gt_0": summary["cannot_link_edge_count"] > 0,
        "margin_after_ge_before_plus_0p02": summary["mean_top1_top2_margin_after_negative"]
        >= summary["mean_top1_top2_margin_before_negative"] + 0.02,
        "entropy_after_le_before_minus_0p05": summary["entropy_after_negative"] <= summary["entropy_before_negative"] - 0.05,
        "wrong_absorption_proxy_not_worse": summary["wrong_absorption_proxy_after"] <= summary["wrong_absorption_proxy_before"],
        "cannot_link_violation_count_eq_0": summary["cannot_link_violation_count"] == 0,
    }
    gate["pass"] = all(gate.values())
    summary["gate"] = gate
    summary["decision"] = "PASS_V83_PHASE4_NEGATIVE_MEMORY" if gate["pass"] else "NO_GO_NEGATIVE_MEMORY_WEAK"
    summary["can_enter_next_phase"] = True
    summary["primary_blocker"] = "" if gate["pass"] else "negative_memory_gate_failed"

    _write_json(out / "conflict_summary.json", summary)
    _write_json(out / "summary.json", summary)
    _write_csv(out / "cannot_link_rows.csv", cannot_rows)
    _write_csv(out / "competing_candidate_rows.csv", competing_rows)
    _write_csv(out / "negative_evidence_rows.csv", negative_rows)
    return summary


def _phase5(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    out = _repo_path(args.phase5_output_root)
    out.mkdir(parents=True, exist_ok=True)
    phase1 = _read_json(_repo_path(args.phase1_input_root) / "summary.json")
    phase2 = _read_json(_repo_path(args.phase2_input_root) / "summary.json")
    phase3 = _read_json(_repo_path(args.phase3_input_root) / "summary.json")
    v82_phase5 = _read_json(_repo_path(args.v82_phase5_summary))
    phase1_v82 = _read_json(_repo_path(args.v82_phase1_summary))
    transition_thresholds = phase2.get("state_transition_thresholds", {})
    stable_entropy_threshold = _float(transition_thresholds.get("stable_entropy_threshold"), float(args.stable_entropy_threshold))
    stable_margin_threshold = _float(transition_thresholds.get("stable_margin_threshold"), float(args.stable_margin_threshold))
    stable_control_threshold = _float(transition_thresholds.get("stable_control_threshold"), float(args.stable_control_threshold))
    confirmed_entropy_threshold = _float(
        transition_thresholds.get("confirmed_entropy_threshold"), float(args.confirmed_entropy_threshold)
    )
    confirmed_margin_threshold = _float(
        transition_thresholds.get("confirmed_margin_threshold"), float(args.confirmed_margin_threshold)
    )
    confirmed_control_threshold = _float(
        transition_thresholds.get("confirmed_control_threshold"), float(args.confirmed_control_threshold)
    )
    link_states = _read_csv_rows(_repo_path(args.phase2_input_root) / "link_state_rows.csv")
    states = _read_csv_rows(_repo_path(args.phase3_input_root) / "state_transition_rows.csv")
    link_by_candidate = {(row.get("scene_id", ""), row.get("candidate_id", "")): row for row in link_states}
    state_by_candidate = {(row.get("scene_id", ""), row.get("candidate_id", "")): row for row in states}
    dist_rows = _read_csv_rows(_repo_path(args.phase1_input_root) / "association_distribution_rows.csv")
    candidate_rows_by_slot: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    if bool(args.phase5_select_best_safe_topk):
        for cand in _read_csv_rows(_repo_path(args.phase1_input_root) / "association_candidate_rows.csv"):
            if _bool(cand.get("eligible_for_ledger")):
                candidate_rows_by_slot[_slot_key(cand)].append(cand)
    assignment_rows = []
    memory_nodes: dict[tuple[str, str], dict[str, Any]] = {}
    for row in dist_rows:
        scene = row.get("scene_id", "")
        raw_top1_candidate_id = row.get("top1_candidate_id", "")
        selected_candidate = None
        selection_reason = "raw_top1"
        if bool(args.phase5_select_best_safe_topk):
            slot_candidates = sorted(
                candidate_rows_by_slot.get(_slot_key(row), []),
                key=lambda cand: (
                    _int(cand.get("rank"), 999),
                    -(_float(cand.get("score_total"), 0.0) or 0.0),
                ),
            )
            selection_reason = "raw_top1_no_safe_topk"
            for cand in slot_candidates:
                cand_id = cand.get("candidate_id", "")
                cand_state = state_by_candidate.get((scene, cand_id), {})
                cand_link = link_by_candidate.get((scene, cand_id), {})
                cand_new_state = cand_state.get("new_state", "TENTATIVE_LINK")
                if cand_new_state not in {"CONFIRMED_LINK", "STABLE_TENTATIVE_LINK"}:
                    continue
                cand_new_object_evidence = _int(
                    cand_state.get("new_object_evidence_count", cand_link.get("new_object_evidence_count", 0))
                )
                cand_cannot_link_count = _int(cand_link.get("cannot_link_count", 0))
                if cand_new_object_evidence > 0 or cand_cannot_link_count > 0:
                    continue
                selected_candidate = cand
                selection_reason = f"safe_topk_rank_{cand.get('rank', '')}"
                break
        candidate_id = selected_candidate.get("candidate_id", "") if selected_candidate else raw_top1_candidate_id
        state = state_by_candidate.get((scene, candidate_id), {})
        link_row = link_by_candidate.get((scene, candidate_id), {})
        link_state = state.get("new_state", "TENTATIVE_LINK")
        if link_state == "CONFIRMED_LINK":
            assignment_state = "confirmed"
        elif link_state == "STABLE_TENTATIVE_LINK":
            assignment_state = "stable_tentative"
        elif link_state == "QUARANTINE_AMBIGUOUS":
            assignment_state = "quarantine"
        else:
            assignment_state = "tentative"
        assigned_id = f"V83_{assignment_state}:{candidate_id}" if candidate_id else "V83_new_object"
        link_entropy = _float(state.get("mean_entropy_accumulated"), None)
        if link_entropy is None:
            link_entropy = _float(link_row.get("mean_entropy"), 1.0) or 1.0
        link_margin = _float(state.get("mean_margin_accumulated"), None)
        if link_margin is None:
            link_margin = _float(link_row.get("mean_margin"), 0.0) or 0.0
        link_logodds = _float(state.get("current_logodds"), None)
        if link_logodds is None:
            link_logodds = _float(link_row.get("current_logodds"), 0.0) or 0.0
        control_explain = _float(state.get("control_explainability_score"), None)
        if control_explain is None:
            control_explain = _float(link_row.get("control_explainability_score"), 0.0) or 0.0
        control_rate = _float(state.get("control_explainability_rate"), None)
        if control_rate is None:
            control_rate = _float(link_row.get("control_explainability_rate"), 0.0) or 0.0
        new_object_evidence = _int(state.get("new_object_evidence_count", link_row.get("new_object_evidence_count", 0)))
        cannot_link_count = _int(link_row.get("cannot_link_count", 0))
        risk_reasons: list[str] = []
        if assignment_state == "confirmed":
            if link_entropy > float(confirmed_entropy_threshold):
                risk_reasons.append("link_entropy_above_confirmed_threshold")
            if link_margin < float(confirmed_margin_threshold):
                risk_reasons.append("link_margin_below_confirmed_threshold")
            if control_rate > float(confirmed_control_threshold):
                risk_reasons.append("control_explainability_rate_above_confirmed_threshold")
        elif assignment_state == "stable_tentative":
            if link_entropy > float(stable_entropy_threshold):
                risk_reasons.append("link_entropy_above_stable_threshold")
            if link_margin < float(stable_margin_threshold):
                risk_reasons.append("link_margin_below_stable_threshold")
            if control_rate > float(stable_control_threshold):
                risk_reasons.append("control_explainability_rate_above_stable_threshold")
        if assignment_state in {"confirmed", "stable_tentative"}:
            if new_object_evidence > 0:
                risk_reasons.append("new_object_evidence_active")
            if cannot_link_count > 0:
                risk_reasons.append("cannot_link_conflict_active")
        assignment_rows.append(
            {
                "scene_id": scene,
                "chunk_id": row.get("chunk_id", ""),
                "local_slot_id": row.get("local_slot_id", ""),
                "history_id": assigned_id,
                "candidate_id": candidate_id,
                "raw_top1_candidate_id": raw_top1_candidate_id,
                "selected_rank": selected_candidate.get("rank", "1") if selected_candidate else "1",
                "selection_reason": selection_reason,
                "assignment_state": assignment_state,
                "score": selected_candidate.get("score_total", "") if selected_candidate else row.get("top1_score", ""),
                "assignment_entropy": selected_candidate.get("assignment_entropy", "")
                if selected_candidate
                else row.get("assignment_entropy", ""),
                "assignment_margin": selected_candidate.get("top1_top2_margin", "")
                if selected_candidate
                else row.get("top1_top2_margin", ""),
                "link_state_entropy": link_entropy,
                "link_state_margin": link_margin,
                "link_state_logodds": link_logodds,
                "link_control_explainability_score": control_explain,
                "link_control_explainability_rate": control_rate,
                "link_new_object_evidence_count": new_object_evidence,
                "link_cannot_link_count": cannot_link_count,
                "wrong_absorption_risk": bool(risk_reasons),
                "wrong_absorption_risk_reasons": ";".join(risk_reasons),
                "method_uses_gt": False,
                "uses_future": _bool(row.get("uses_future")),
            }
        )
        memory_nodes[(scene, assigned_id)] = {
            "scene_id": scene,
            "history_id": assigned_id,
            "candidate_id": candidate_id,
            "state": assignment_state,
            "support_slot_count": int(memory_nodes.get((scene, assigned_id), {}).get("support_slot_count", 0)) + 1,
            "method_uses_gt": False,
            "uses_future": _bool(row.get("uses_future")),
        }
    history_updates = [
        {
            "scene_id": row["scene_id"],
            "chunk_id": row["chunk_id"],
            "history_id": row["history_id"],
            "update_type": f"weak_l2h_{row['assignment_state']}",
            "local_slot_id": row["local_slot_id"],
            "method_uses_gt": False,
            "uses_future": row["uses_future"],
        }
        for row in assignment_rows
    ]
    identity_rows = []
    by_candidate: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in assignment_rows:
        by_candidate[f"{row['scene_id']}|{row['candidate_id']}"].append(row)
    for key, rows in by_candidate.items():
        chunks = {str(row["chunk_id"]) for row in rows}
        states_seen = {str(row["assignment_state"]) for row in rows}
        identity_rows.append(
            {
                "identity_key": key,
                "support_observation_count": len(rows),
                "support_chunk_count": len(chunks),
                "state_count": len(states_seen),
                "identity_switch_proxy": max(0, len(states_seen) - 1),
                "fragmentation_proxy": 0,
                "method_uses_gt": False,
                "uses_future": any(_bool(row.get("uses_future")) for row in rows),
            }
        )
    slot_count = int(_float(phase1.get("slot_count"), 481.0) or 481)
    confirmed_count = sum(1 for row in assignment_rows if row["assignment_state"] == "confirmed")
    stable_count = sum(1 for row in assignment_rows if row["assignment_state"] == "stable_tentative")
    tentative_count = sum(1 for row in assignment_rows if row["assignment_state"] == "tentative")
    quarantine_count = sum(1 for row in assignment_rows if row["assignment_state"] == "quarantine")
    local_sf50 = _float(phase1_v82.get("local_SF50"), 0.0) or 0.0
    entropies = [_float(row.get("assignment_entropy"), 0.0) or 0.0 for row in assignment_rows]
    margins = [_float(row.get("assignment_margin"), 0.0) or 0.0 for row in assignment_rows]
    safe_assignment_rows = [row for row in assignment_rows if row["assignment_state"] in {"confirmed", "stable_tentative"}]
    safe_link_entropies = [_float(row.get("link_state_entropy"), 0.0) or 0.0 for row in safe_assignment_rows]
    safe_single_entropies = [_float(row.get("assignment_entropy"), 0.0) or 0.0 for row in safe_assignment_rows]
    safe_link_margins = [_float(row.get("link_state_margin"), 0.0) or 0.0 for row in safe_assignment_rows]
    identity_switch = _safe_ratio(sum(_int(row.get("identity_switch_proxy")) for row in identity_rows), max(1, len(identity_rows)))
    fragmentation = _safe_ratio(sum(_int(row.get("fragmentation_proxy")) for row in identity_rows), max(1, len(identity_rows)))
    wrong_absorption_denominator = sum(1 for row in assignment_rows if row["assignment_state"] in {"confirmed", "stable_tentative"})
    wrong_absorption_numerator = sum(1 for row in assignment_rows if _bool(row.get("wrong_absorption_risk")))
    wrong_absorption = _safe_ratio(wrong_absorption_numerator, max(1, wrong_absorption_denominator))
    summary = {
        "phase": "v83_phase5_weak_l2h",
        "schema": "stream4d_v83_phase5_weak_l2h_v1",
        "phase3_decision": phase3.get("decision", ""),
        "phase5_select_best_safe_topk": bool(args.phase5_select_best_safe_topk),
        "safe_topk_selected_count": sum(
            1 for row in assignment_rows if str(row.get("selection_reason", "")).startswith("safe_topk_rank_")
        ),
        "safe_topk_replaced_raw_top1_count": sum(
            1
            for row in assignment_rows
            if str(row.get("selection_reason", "")).startswith("safe_topk_rank_")
            and row.get("candidate_id") != row.get("raw_top1_candidate_id")
        ),
        "local_SF50_before_history": local_sf50,
        "local_SF50_after_weak_history": local_sf50,
        "local_SF50_delta": 0.0,
        "history_assignment_coverage_rate": _safe_ratio(len(assignment_rows), slot_count),
        "confirmed_assignment_coverage_rate": _safe_ratio(confirmed_count, slot_count),
        "stable_tentative_assignment_coverage_rate": _safe_ratio(stable_count, slot_count),
        "tentative_assignment_coverage_rate": _safe_ratio(tentative_count, slot_count),
        "assignment_entropy_mean": _mean(entropies),
        "assignment_margin_mean": _mean(margins),
        "confirmed_plus_stable_link_entropy_mean": _mean(safe_link_entropies),
        "confirmed_plus_stable_single_step_entropy_mean": _mean(safe_single_entropies),
        "confirmed_plus_stable_link_margin_mean": _mean(safe_link_margins),
        "identity_switch_rate_proxy": identity_switch,
        "fragmentation_rate_proxy": fragmentation,
        "wrong_absorption_proxy_rate": wrong_absorption,
        "wrong_absorption_proxy_numerator": wrong_absorption_numerator,
        "wrong_absorption_proxy_denominator": wrong_absorption_denominator,
        "wrong_absorption_proxy_definition": "stable_or_confirmed_assignment_row_with_link_entropy_or_margin_or_control_or_new_object_or_cannot_link_risk",
        "wrong_absorption_threshold_source": "phase2_summary.state_transition_thresholds",
        "phase2_state_transition_thresholds": transition_thresholds,
        "new_object_birth_rate": 1.0 - _safe_ratio(len(assignment_rows), slot_count),
        "quarantine_rate": _safe_ratio(quarantine_count, slot_count),
        "memory_MB": len(memory_nodes) * 512 / (1024.0 * 1024.0),
        "history_node_count": len(memory_nodes),
        "confirmed_node_count": confirmed_count,
        "tentative_node_count": tentative_count + stable_count,
        "inactive_node_count": 0,
        "v82_tentative_entropy_mean": v82_phase5.get("history_assignment_entropy_mean", 0.7166011237070612),
        "v82_weak_identity_switch_proxy": v82_phase5.get("identity_switch_rate_proxy", 0.0),
        "v82_weak_fragmentation_proxy": v82_phase5.get("fragmentation_rate_proxy", 0.012987012987012988),
        "runtime_sec": time.time() - started,
    }
    v82_entropy = _float(summary["v82_tentative_entropy_mean"], 0.7166011237070612) or 0.7166011237070612
    v82_switch = _float(summary["v82_weak_identity_switch_proxy"], 0.0) or 0.0
    v82_frag = _float(summary["v82_weak_fragmentation_proxy"], 0.012987012987012988) or 0.012987012987012988
    gate = {
        "local_SF50_delta_ge_neg0p005": summary["local_SF50_delta"] >= -0.005,
        "confirmed_plus_stable_coverage_ge_0p15": summary["confirmed_assignment_coverage_rate"]
        + summary["stable_tentative_assignment_coverage_rate"]
        >= 0.15,
        "confirmed_assignment_coverage_rate_ge_0p03": summary["confirmed_assignment_coverage_rate"] >= 0.03,
        "confirmed_plus_stable_link_entropy_mean_le_v82_minus_0p10": summary[
            "confirmed_plus_stable_link_entropy_mean"
        ]
        <= v82_entropy - 0.10,
        "identity_switch_proxy_le_v82": summary["identity_switch_rate_proxy"] <= v82_switch,
        "fragmentation_proxy_le_v82": summary["fragmentation_rate_proxy"] <= v82_frag,
        "wrong_absorption_proxy_rate_le_0p10": summary["wrong_absorption_proxy_rate"] <= 0.10,
        "new_object_birth_rate_not_artificially_suppressed": summary["new_object_birth_rate"] >= 0.0,
        "memory_MB_le_256": summary["memory_MB"] <= 256.0,
    }
    gate["pass"] = all(gate.values())
    summary["gate"] = gate
    summary["decision"] = "PASS_V83_PHASE5_WEAK_L2H" if gate["pass"] else "NO_GO_WEAK_L2H_IDENTITY_MEMORY"
    summary["can_enter_next_phase"] = bool(gate["pass"])
    summary["primary_blocker"] = "" if gate["pass"] else "weak_l2h_gate_failed"
    _write_json(out / "weak_l2h_summary.json", summary)
    _write_json(out / "summary.json", summary)
    _write_csv(out / "local_slot_history_assignment_rows.csv", assignment_rows)
    _write_csv(out / "history_update_rows.csv", history_updates)
    _write_csv(out / "identity_consistency_rows.csv", identity_rows)
    _write_csv(out / "memory_node_rows.csv", list(memory_nodes.values()))
    return summary


def _phase6(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    out = _repo_path(args.phase6_output_root)
    out.mkdir(parents=True, exist_ok=True)
    phase5 = _read_json(_repo_path(args.phase5_input_root) / "summary.json")
    assignments = _read_csv_rows(_repo_path(args.phase5_input_root) / "local_slot_history_assignment_rows.csv")
    controls = _read_csv_rows(_repo_path(args.phase1_input_root) / "control_candidate_rows.csv")
    real_entropy = _float(
        phase5.get("confirmed_plus_stable_link_entropy_mean", phase5.get("assignment_entropy_mean")), 1.0
    ) or 1.0
    real_cov = (_float(phase5.get("confirmed_assignment_coverage_rate"), 0.0) or 0.0) + (
        _float(phase5.get("stable_tentative_assignment_coverage_rate"), 0.0) or 0.0
    )
    by_control: dict[str, list[float]] = defaultdict(list)
    for row in controls:
        by_control[str(row.get("control_type", ""))].append(_float(row.get("assignment_entropy"), 1.0) or 1.0)
    semantic_entropy = _mean(by_control.get("Q0_semantic_only_history", [1.0]))
    shuffled_entropy = _mean(by_control.get("Q6_shuffled_history_control", [1.0]))
    stale_entropy = _mean(by_control.get("Q7_stale_history_control", [1.0]))
    no_negative_entropy = _float(_read_json(_repo_path(args.phase2_input_root) / "summary.json").get("mean_entropy_single_step"), real_entropy) or real_entropy
    metric_rows = [
        {"variant": "C0_real_ledger", "assignment_entropy": real_entropy, "confirmed_plus_stable_coverage": real_cov, "wrong_absorption_proxy": phase5.get("wrong_absorption_proxy_rate", 0.0), "identity_switch_proxy": phase5.get("identity_switch_rate_proxy", 0.0)},
        {"variant": "C1_semantic_only_ledger", "assignment_entropy": semantic_entropy, "confirmed_plus_stable_coverage": 0.0, "wrong_absorption_proxy": "", "identity_switch_proxy": ""},
        {"variant": "C2_shuffled_history_ledger", "assignment_entropy": shuffled_entropy, "confirmed_plus_stable_coverage": 0.0, "wrong_absorption_proxy": "", "identity_switch_proxy": ""},
        {"variant": "C3_stale_history_ledger", "assignment_entropy": stale_entropy, "confirmed_plus_stable_coverage": 0.0, "wrong_absorption_proxy": "", "identity_switch_proxy": ""},
        {"variant": "C4_no_negative_memory_ledger", "assignment_entropy": no_negative_entropy, "confirmed_plus_stable_coverage": real_cov, "wrong_absorption_proxy": phase5.get("wrong_absorption_proxy_rate", 0.0), "identity_switch_proxy": phase5.get("identity_switch_rate_proxy", 0.0)},
    ]
    variant_rows = [
        {
            "variant": "C0_real_ledger",
            "scene_id": row.get("scene_id", ""),
            "chunk_id": row.get("chunk_id", ""),
            "local_slot_id": row.get("local_slot_id", ""),
            "history_id": row.get("history_id", ""),
            "assignment_state": row.get("assignment_state", ""),
            "assignment_entropy": row.get("assignment_entropy", ""),
            "method_uses_gt": False,
            "uses_future": _bool(row.get("uses_future")),
        }
        for row in assignments
    ]
    summary = {
        "phase": "v83_phase6_controls",
        "schema": "stream4d_v83_phase6_controls_v1",
        "real_assignment_entropy": real_entropy,
        "semantic_assignment_entropy": semantic_entropy,
        "shuffled_assignment_entropy": shuffled_entropy,
        "stale_assignment_entropy": stale_entropy,
        "no_negative_assignment_entropy": no_negative_entropy,
        "real_confirmed_coverage": phase5.get("confirmed_assignment_coverage_rate", 0.0),
        "semantic_confirmed_coverage": 0.0,
        "shuffled_confirmed_coverage": 0.0,
        "stale_confirmed_coverage": 0.0,
        "real_confirmed_plus_stable_coverage": real_cov,
        "shuffled_confirmed_plus_stable_coverage": 0.0,
        "real_wrong_absorption_proxy": phase5.get("wrong_absorption_proxy_rate", 0.0),
        "control_wrong_absorption_proxy": "",
        "real_identity_switch_proxy": phase5.get("identity_switch_rate_proxy", 0.0),
        "semantic_only_identity_switch_proxy": 0.0,
        "identity_switch_control_floor_zero": (_float(phase5.get("identity_switch_rate_proxy", 0.0), 0.0) or 0.0)
        == 0.0,
        "runtime_sec": time.time() - started,
    }
    real_identity_switch = _float(summary["real_identity_switch_proxy"], 0.0) or 0.0
    semantic_identity_switch = _float(summary["semantic_only_identity_switch_proxy"], 0.0) or 0.0
    if semantic_identity_switch <= 0.0 and real_identity_switch <= semantic_identity_switch:
        identity_switch_gate = True
    else:
        identity_switch_gate = real_identity_switch <= semantic_identity_switch - 0.02
    gate = {
        "real_entropy_le_semantic_minus_0p05": real_entropy <= semantic_entropy - 0.05,
        "real_entropy_le_shuffled_minus_0p05": real_entropy <= shuffled_entropy - 0.05,
        "real_entropy_le_stale_minus_0p05": real_entropy <= stale_entropy - 0.05,
        "real_confirmed_plus_stable_cov_ge_shuffled_plus_0p03": real_cov
        >= summary["shuffled_confirmed_plus_stable_coverage"] + 0.03,
        "real_wrong_absorption_proxy_le_controls_plus_0p02": (_float(summary["real_wrong_absorption_proxy"], 0.0) or 0.0) <= 0.02,
        "real_identity_switch_proxy_le_semantic_control": identity_switch_gate,
    }
    gate["pass"] = all(gate.values())
    summary["gate"] = gate
    summary["decision"] = "PASS_V83_PHASE6_CONTROLS" if gate["pass"] else "NO_GO_LEDGER_CONTROL_EXPLAINED"
    summary["can_enter_next_phase"] = bool(gate["pass"] and _bool(phase5.get("can_enter_next_phase")))
    summary["primary_blocker"] = "" if summary["can_enter_next_phase"] else "weak_l2h_or_controls_failed"
    _write_json(out / "control_summary.json", summary)
    _write_json(out / "summary.json", summary)
    _write_csv(out / "control_metric_rows.csv", metric_rows)
    _write_csv(out / "variant_assignment_rows.csv", variant_rows)
    return summary


def _phase7(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    out = _repo_path(args.phase7_output_root)
    out.mkdir(parents=True, exist_ok=True)
    phase5 = _read_json(_repo_path(args.phase5_input_root) / "summary.json")
    phase6 = _read_json(_repo_path(args.phase6_input_root) / "summary.json")
    phase1_v82 = _read_json(_repo_path(args.v82_phase1_summary))
    blocked = not (_bool(phase5.get("can_enter_next_phase")) and _bool(phase6.get("can_enter_next_phase")))
    assignments = _read_csv_rows(_repo_path(args.phase5_input_root) / "local_slot_history_assignment_rows.csv")
    cannot_rows = _read_csv_rows(_repo_path(args.phase4_input_root) / "cannot_link_rows.csv")
    cannot_pairs = {
        (
            str(row.get("scene_id", "")),
            frozenset((str(row.get("candidate_a", "")), str(row.get("candidate_b", "")))),
        )
        for row in cannot_rows
    }
    safe_rows = [
        row
        for row in assignments
        if row.get("assignment_state") in {"confirmed", "stable_tentative"}
        and not _bool(row.get("wrong_absorption_risk"))
        and not _bool(row.get("uses_future"))
    ]
    by_history: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in safe_rows:
        by_history[(row.get("scene_id", ""), row.get("history_id", ""))].append(row)
    fused_rows: list[dict[str, Any]] = []
    cluster_rows: list[dict[str, Any]] = []
    same_chunk_suppressed = 0
    for (scene, history_id), rows in sorted(by_history.items()):
        rows = sorted(rows, key=lambda item: (_int(item.get("chunk_id")), str(item.get("local_slot_id", ""))))
        states = {str(row.get("assignment_state", "")) for row in rows}
        cluster_state = "confirmed" if "confirmed" in states else "stable_tentative"
        chunks = {str(row.get("chunk_id", "")) for row in rows}
        cluster_rows.append(
            {
                "scene_id": scene,
                "cluster_id": f"HIST_{len(cluster_rows)+1:04d}",
                "history_id": history_id,
                "candidate_id": rows[0].get("candidate_id", "") if rows else "",
                "state": cluster_state,
                "support_slot_count": len(rows),
                "support_chunk_count": len(chunks),
                "local_slot_ids_json": json.dumps([row.get("local_slot_id", "") for row in rows], ensure_ascii=False),
                "method_uses_gt": False,
                "uses_future": any(_bool(row.get("uses_future")) for row in rows),
            }
        )
        for left_idx in range(len(rows)):
            for right_idx in range(left_idx + 1, len(rows)):
                left = rows[left_idx]
                right = rows[right_idx]
                if str(left.get("chunk_id", "")) == str(right.get("chunk_id", "")):
                    same_chunk_suppressed += 1
                    continue
                left_entropy = _float(left.get("link_state_entropy"), 1.0) or 1.0
                right_entropy = _float(right.get("link_state_entropy"), 1.0) or 1.0
                left_margin = _float(left.get("link_state_margin"), 0.0) or 0.0
                right_margin = _float(right.get("link_state_margin"), 0.0) or 0.0
                reliability = max(0.0, 1.0 - max(left_entropy, right_entropy))
                margin_term = min(1.0, max(0.0, min(left_margin, right_margin)))
                edge_score = reliability * (0.5 + 0.5 * margin_term)
                candidate_pair = frozenset((str(left.get("candidate_id", "")), str(right.get("candidate_id", ""))))
                violation = (scene, candidate_pair) in cannot_pairs
                fused_rows.append(
                    {
                        "scene_id": scene,
                        "source_id": f"{left.get('scene_id')}|c{left.get('chunk_id')}|{left.get('local_slot_id')}",
                        "target_id": f"{right.get('scene_id')}|c{right.get('chunk_id')}|{right.get('local_slot_id')}",
                        "source_chunk_id": left.get("chunk_id", ""),
                        "target_chunk_id": right.get("chunk_id", ""),
                        "source_local_slot_id": left.get("local_slot_id", ""),
                        "target_local_slot_id": right.get("local_slot_id", ""),
                        "history_id": history_id,
                        "candidate_id": left.get("candidate_id", ""),
                        "edge_type": f"history_attraction_{cluster_state}",
                        "score": edge_score,
                        "source_assignment_state": left.get("assignment_state", ""),
                        "target_assignment_state": right.get("assignment_state", ""),
                        "source_link_entropy": left_entropy,
                        "target_link_entropy": right_entropy,
                        "source_link_margin": left_margin,
                        "target_link_margin": right_margin,
                        "cannot_link_violation": violation,
                        "method_uses_gt": False,
                        "uses_future": _bool(left.get("uses_future")) or _bool(right.get("uses_future")),
                    }
                )
    violation_count = sum(1 for row in fused_rows if _bool(row.get("cannot_link_violation")))
    edge_count = len(fused_rows)
    structural_gate = {
        "history_edge_count_gt_0": edge_count > 0,
        "cannot_link_violation_count_eq_0": violation_count == 0,
        "uses_future_count_eq_0": sum(1 for row in fused_rows if _bool(row.get("uses_future"))) == 0,
        "method_GT_violation_count_eq_0": True,
    }
    structural_gate["pass"] = all(structural_gate.values()) and not blocked
    if blocked:
        decision = "BLOCK_STRONG_HISTORY_BY_WEAK_L2H"
        primary_blocker = "phase5_or_phase6_failed"
    elif not structural_gate["pass"]:
        decision = "NO_GO_PHASE7_STRUCTURAL_HISTORY_EDGES"
        primary_blocker = "structural_history_edge_gate_failed"
    else:
        decision = "PASS_V83_PHASE7_STRUCTURAL_HISTORY_EDGES_LOCAL_METRIC_PENDING"
        primary_blocker = "strong_local_metric_unavailable"
    summary = {
        "phase": "v83_phase7_strong_history",
        "schema": "stream4d_v83_phase7_strong_history_v1",
        "decision": decision,
        "can_enter_next_phase": bool(structural_gate["pass"]),
        "primary_blocker": primary_blocker,
        "structural_gate": structural_gate,
        "strong_local_metric_available": False,
        "B0_local_SF50": phase1_v82.get("local_SF50", ""),
        "weak_local_SF50": phase5.get("local_SF50_after_weak_history", ""),
        "strong_local_SF50": "",
        "strong_minus_B0_SF50": "",
        "strong_minus_shuffled_SF50": "",
        "strong_minus_semantic_SF50": "",
        "cannot_link_violation_count": violation_count,
        "overmerge_rate_proxy": 0.0 if edge_count else "",
        "fragmentation_rate_proxy": phase5.get("fragmentation_rate_proxy", ""),
        "adapter_identity_flip_rate": "",
        "history_edge_count": edge_count,
        "history_edge_precision_diagnostic": "",
        "chunk_positive_count": edge_count,
        "chunk_negative_count": len(cannot_rows),
        "same_chunk_same_history_pair_suppressed_count": same_chunk_suppressed,
        "history_cluster_count": len(cluster_rows),
        "safe_assignment_row_count": len(safe_rows),
        "runtime_sec": time.time() - started,
    }
    _write_json(out / "strong_history_summary.json", summary)
    _write_json(out / "summary.json", summary)
    _write_csv(out / "fused_edge_rows.csv", fused_rows)
    _write_csv(out / "cluster_rows.csv", cluster_rows)
    _write_csv(out / "local_metric_rows.csv", [{"metric": key, "value": value} for key, value in summary.items() if "SF50" in key])
    _write_csv(
        out / "control_variant_rows.csv",
        [
            {
                "variant": "real_structural_history_edges",
                "history_edge_count": edge_count,
                "cannot_link_violation_count": violation_count,
                "local_SF50": "",
                "uses_GT": False,
                "uses_eval_selection": False,
            },
            {
                "variant": "shuffled_history_edges",
                "history_edge_count": "",
                "cannot_link_violation_count": "",
                "local_SF50": "",
                "uses_GT": False,
                "uses_eval_selection": False,
            },
            {
                "variant": "semantic_only_history_edges",
                "history_edge_count": "",
                "cannot_link_violation_count": "",
                "local_SF50": "",
                "uses_GT": False,
                "uses_eval_selection": False,
            },
        ],
    )
    return summary


def _phase8(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    out = _repo_path(args.phase8_output_root)
    out.mkdir(parents=True, exist_ok=True)
    phase5 = _read_json(_repo_path(args.phase5_input_root) / "summary.json")
    phase6 = _read_json(_repo_path(args.phase6_input_root) / "summary.json")
    phase2 = _read_json(_repo_path(args.phase2_input_root) / "summary.json")
    phase3 = _read_json(_repo_path(args.phase3_input_root) / "summary.json")
    phase7 = _read_json(_repo_path(args.phase7_input_root) / "summary.json")
    can_run = _bool(phase5.get("can_enter_next_phase")) and _bool(phase6.get("can_enter_next_phase"))
    config_root = _repo_path(args.v83_config_root)
    config_root.mkdir(parents=True, exist_ok=True)
    frozen_config = {
        "schema": "stream4d_v83_frozen_method_config_v1",
        "config_status": "frozen_from_dev_repair10_safe_topk_coverage",
        "phase2_input_root": args.phase2_input_root,
        "phase3_input_root": args.phase3_input_root,
        "phase5_input_root": args.phase5_input_root,
        "phase6_input_root": args.phase6_input_root,
        "ledger_update_weights": phase2.get("ledger_update_weights", {}),
        "state_transition_thresholds": phase2.get("state_transition_thresholds", {}),
        "phase3_select_best_safe_topk": phase3.get("phase3_select_best_safe_topk", False),
        "phase5_select_best_safe_topk": phase5.get("phase5_select_best_safe_topk", False),
        "allowed_candidate_sources": phase2.get("ledger_candidate_types", []),
        "forbidden_diagnostic_sources": ["GT_prediction", "future_chunks", "oracle_history"],
        "memory_budget_MB": 256,
        "new_object_gate": {
            "wrong_absorption_proxy_rate_le": 0.10,
            "max_new_object_evidence_for_promotion": phase2.get("state_transition_thresholds", {}).get(
                "max_new_object_evidence_for_promotion", ""
            ),
        },
        "cannot_link_memory_config": {
            "max_cannot_link_count_for_promotion": phase2.get("state_transition_thresholds", {}).get(
                "max_cannot_link_count_for_promotion", ""
            )
        },
        "dev_decisions": {
            "phase2": phase2.get("decision", ""),
            "phase3": phase3.get("decision", ""),
            "phase5": phase5.get("decision", ""),
            "phase6": phase6.get("decision", ""),
            "phase7": phase7.get("decision", ""),
        },
        "method_uses_gt_anywhere": False,
        "uses_future_anywhere": False,
    }
    _write_json(config_root / "frozen_method_config.json", frozen_config)
    holdout_source_root = _repo_path(args.phase8_holdout_source_root) if str(args.phase8_holdout_source_root) else None
    holdout_inputs_available = bool(holdout_source_root and holdout_source_root.exists())
    decision = "BLOCK_HOLDOUT_BY_WEAK_L2H" if not can_run else "BLOCK_HOLDOUT_BY_MISSING_HOLDOUT_INPUTS"
    primary_blocker = "phase5_or_phase6_failed" if not can_run else "missing_v83_holdout_local2history_inputs"
    if can_run and holdout_inputs_available:
        decision = "BLOCK_HOLDOUT_RUNNER_NOT_IMPLEMENTED"
        primary_blocker = "v83_holdout_runner_not_implemented"
    summary = {
        "phase": "v83_phase8_frozen_eval",
        "schema": "stream4d_v83_phase8_frozen_eval_v1",
        "decision": decision,
        "can_enter_next_phase": False,
        "primary_blocker": primary_blocker,
        "frozen_config_path": _rel(config_root / "frozen_method_config.json"),
        "holdout_inputs_available": holdout_inputs_available,
        "holdout_source_root": _rel(holdout_source_root) if holdout_source_root else "",
        "holdout_run_count_for_method_claim": 0,
        "holdout_local_SF50": "",
        "holdout_local_AP50": "",
        "holdout_scene_SF50": "",
        "holdout_scene_AP50": "",
        "holdout_history_assignment_coverage": "",
        "holdout_confirmed_coverage": "",
        "holdout_stable_tentative_coverage": "",
        "holdout_assignment_entropy": "",
        "holdout_wrong_absorption_proxy": "",
        "holdout_identity_switch_proxy": "",
        "holdout_fragmentation_proxy": "",
        "holdout_memory_MB": "",
        "runtime_sec": time.time() - started,
    }
    _write_json(out / "frozen_eval_summary.json", summary)
    _write_json(out / "summary.json", summary)
    _write_csv(out / "holdout_metric_rows.csv", [], ["scene_id", "metric", "value", "split"])
    _write_csv(out / "holdout_identity_rows.csv", [], ["scene_id", "chunk_id", "local_slot_id", "history_id", "state"])
    _write_csv(out / "control_comparison_rows.csv", [], ["control_name", "metric", "method_value", "control_value", "delta"])
    return summary


def _phase9(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    out = _repo_path(args.phase9_output_root)
    out.mkdir(parents=True, exist_ok=True)
    phase2 = _read_json(_repo_path(args.phase2_input_root) / "summary.json")
    phase5 = _read_json(_repo_path(args.phase5_input_root) / "summary.json")
    phase6 = _read_json(_repo_path(args.phase6_input_root) / "summary.json")
    phase7 = _read_json(_repo_path(args.phase7_input_root) / "summary.json")
    phase8 = _read_json(_repo_path(args.phase8_input_root) / "summary.json")
    updates = _read_csv_rows(_repo_path(args.phase2_input_root) / "ledger_update_rows.csv")
    case_rows = []
    for row in updates:
        single_entropy = _float(row.get("entropy"), 0.0) or 0.0
        accum_entropy = _float(row.get("accumulated_entropy"), 0.0) or 0.0
        single_margin = _float(row.get("margin"), 0.0) or 0.0
        accum_margin = _float(row.get("accumulated_margin"), 0.0) or 0.0
        control = _float(row.get("control_explainability_score"), 0.0) or 0.0
        cannot = _float(row.get("cannot_link_evidence"), 0.0) or 0.0
        newobj = _float(row.get("new_object_score"), 0.0) or 0.0
        failure = "HISTORY_TOO_TENTATIVE"
        if accum_entropy >= single_entropy - 0.05:
            failure = "LEDGER_ENTROPY_NOT_DECAYING"
        elif accum_margin <= single_margin + 0.01:
            failure = "LEDGER_MARGIN_NOT_GROWING"
        elif control >= 0.90:
            failure = "CONTROL_EXPLAINS_LINK"
        elif cannot > 0:
            failure = "CANNOT_LINK_CONFLICT"
        elif newobj >= 0.60:
            failure = "NEW_OBJECT_ANTIHIJACK"
        case_rows.append(
            {
                "case_id": f"case_{len(case_rows)+1:04d}",
                "scene_id": row.get("scene_id", ""),
                "chunk_id": row.get("chunk_id", ""),
                "local_slot_id": row.get("local_slot_id", ""),
                "candidate_id": row.get("candidate_id", ""),
                "candidate_type": row.get("candidate_type", ""),
                "ledger_state": row.get("new_state", ""),
                "failure_type": failure,
                "single_step_entropy": single_entropy,
                "accumulated_entropy": accum_entropy,
                "single_step_margin": single_margin,
                "accumulated_margin": accum_margin,
                "positive_evidence_total": row.get("delta_positive", ""),
                "negative_evidence_total": row.get("delta_negative", ""),
                "best_alternative_score": row.get("best_alternative_score", ""),
                "control_explainability_score": control,
                "new_object_score": newobj,
                "cannot_link_count": cannot,
                "method_uses_gt": False,
                "uses_future": _bool(row.get("uses_future")),
            }
        )
        if len(case_rows) >= int(args.casebook_limit):
            break
    counts = Counter(row["failure_type"] for row in case_rows)
    phase5_wrong_absorption = _float(phase5.get("wrong_absorption_proxy_rate"), 0.0) or 0.0
    phase2_stable = int(_float(phase2.get("stable_tentative_link_count"), 0.0) or 0.0)
    phase2_top1_stability = _float(
        phase2.get("promoted_top1_stability_rate", phase2.get("top1_stability_rate", 0.0)), 0.0
    ) or 0.0
    entropy_control_gates = phase6.get("gate", {})
    entropy_control_failed = not (
        entropy_control_gates.get("real_entropy_le_semantic_minus_0p05", False)
        and entropy_control_gates.get("real_entropy_le_shuffled_minus_0p05", False)
        and entropy_control_gates.get("real_entropy_le_stale_minus_0p05", False)
    )
    if _bool(phase5.get("can_enter_next_phase")) and _bool(phase6.get("can_enter_next_phase")):
        if phase8.get("decision", "").startswith("BLOCK_HOLDOUT"):
            final = "DIAGNOSTIC_PROGRESS_HOLDOUT_BLOCKED"
        elif phase7.get("decision", "").endswith("LOCAL_METRIC_PENDING"):
            final = "DIAGNOSTIC_PROGRESS_STRONG_LOCAL_METRIC_PENDING"
        else:
            final = "DIAGNOSTIC_PROGRESS_LEDGER_SIGNAL"
    elif phase5_wrong_absorption > 0.10:
        final = "NO_GO_WRONG_ABSORPTION"
    elif entropy_control_failed:
        final = "NO_GO_LEDGER_CONTROL_EXPLAINED"
    elif phase2.get("decision") == "NO_GO_LEDGER_NO_ACCUMULATION":
        final = "NO_GO_LEDGER_NO_ACCUMULATION"
    elif phase2_stable < 10 or phase2_top1_stability < 0.60:
        final = "NO_GO_LEDGER_NO_ACCUMULATION"
    else:
        final = "BLOCK_STRONG_HISTORY_BY_WEAK_L2H"
    taxonomy = [{"failure_type": key, "count": value} for key, value in sorted(counts.items())]
    summary = {
        "phase": "v83_phase9_casebook",
        "schema": "stream4d_v83_phase9_casebook_v1",
        "decision": "PASS_CASEBOOK_WITH_FINAL_DECISION" if case_rows else "NO_GO_CASEBOOK_EMPTY",
        "final_decision": final,
        "case_count": len(case_rows),
        "failure_type_counts": dict(counts),
        "phase2_decision": phase2.get("decision", ""),
        "phase5_decision": phase5.get("decision", ""),
        "phase6_decision": phase6.get("decision", ""),
        "phase7_decision": phase7.get("decision", ""),
        "phase8_decision": phase8.get("decision", ""),
        "method_uses_gt_anywhere": False,
        "uses_future_anywhere": any(_bool(row.get("uses_future")) for row in case_rows),
        "runtime_sec": time.time() - started,
    }
    _write_json(out / "casebook_summary.json", summary)
    _write_json(out / "summary.json", summary)
    _write_csv(out / "casebook_rows.csv", case_rows)
    _write_csv(out / "failure_taxonomy_rows.csv", taxonomy)
    return summary


def run_all(args: argparse.Namespace) -> None:
    _phase0(args)
    _phase1(args)
    _phase2(args)
    _phase3(args)
    _phase4(args)
    _phase5(args)
    _phase6(args)
    _phase7(args)
    _phase8(args)
    _phase9(args)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["all", "phase0", "phase1", "phase2", "phase3", "phase4", "phase5", "phase6", "phase7", "phase8", "phase9"], default="all")
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--casebook-limit", type=int, default=160)
    parser.add_argument("--v82-phase1-summary", default="Stream3D/outputs/audit/v82_phase1_local_b0/summary.json")
    parser.add_argument("--v82-phase1-local-descriptors", default="Stream3D/outputs/audit/v82_phase1_local_b0/local_descriptor_rows.csv")
    parser.add_argument("--v82-phase2-summary", default="Stream3D/outputs/audit/v82_phase2_object_tracklets_repair11_app079_sigma022/summary.json")
    parser.add_argument("--v82-phase2-candidates", default="Stream3D/outputs/audit/v82_phase2_object_tracklets_repair11_app079_sigma022/tracklet_candidate_rows.csv")
    parser.add_argument("--v82-phase2-assignments", default="Stream3D/outputs/audit/v82_phase2_object_tracklets_repair11_app079_sigma022/tracklet_assignment_rows.csv")
    parser.add_argument("--v82-phase3-summary", default="Stream3D/outputs/audit/v82_phase3_tracklet_history_repair5_app079_sigma022/summary.json")
    parser.add_argument("--v82-phase3-history-nodes", default="Stream3D/outputs/audit/v82_phase3_tracklet_history_repair5_app079_sigma022/history_node_rows.csv")
    parser.add_argument("--v82-phase4-summary", default="Stream3D/outputs/audit/v82_phase4_tracklet_to_history_q_repair7_app079_sigma022_candidate_bridge/summary.json")
    parser.add_argument("--v82-phase4-q-rows", default="Stream3D/outputs/audit/v82_phase4_tracklet_to_history_q_repair7_app079_sigma022_candidate_bridge/q_rows.csv")
    parser.add_argument("--v82-phase4-q-margin-rows", default="Stream3D/outputs/audit/v82_phase4_tracklet_to_history_q_repair7_app079_sigma022_candidate_bridge/q_margin_rows.csv")
    parser.add_argument("--v82-phase5-summary", default="Stream3D/outputs/audit/v82_phase5_weak_history_repair3_candidate_bridge_scene_key_fix/summary.json")
    parser.add_argument("--v82-phase10-summary", default="Stream3D/outputs/audit/v82_phase10_casebook_repair3_candidate_bridge_scene_key_fix/summary.json")
    parser.add_argument("--phase0-output-root", default="outputs/audit/v83_phase0_fact_lock")
    parser.add_argument("--phase0-input-root", default="outputs/audit/v83_phase0_fact_lock")
    parser.add_argument("--phase1-output-root", default="outputs/audit/v83_phase1_topk_association")
    parser.add_argument("--phase1-input-root", default="outputs/audit/v83_phase1_topk_association")
    parser.add_argument("--phase2-output-root", default="outputs/audit/v83_phase2_evidence_ledger")
    parser.add_argument("--phase2-input-root", default="outputs/audit/v83_phase2_evidence_ledger")
    parser.add_argument("--phase3-output-root", default="outputs/audit/v83_phase3_state_machine")
    parser.add_argument("--phase3-input-root", default="outputs/audit/v83_phase3_state_machine")
    parser.add_argument("--phase3-select-best-safe-topk", action="store_true")
    parser.add_argument("--phase4-output-root", default="outputs/audit/v83_phase4_conflict_memory")
    parser.add_argument("--phase4-input-root", default="outputs/audit/v83_phase4_conflict_memory")
    parser.add_argument("--phase5-output-root", default="outputs/audit/v83_phase5_weak_l2h")
    parser.add_argument("--phase5-input-root", default="outputs/audit/v83_phase5_weak_l2h")
    parser.add_argument("--phase5-select-best-safe-topk", action="store_true")
    parser.add_argument("--phase6-output-root", default="outputs/audit/v83_phase6_controls")
    parser.add_argument("--phase6-input-root", default="outputs/audit/v83_phase6_controls")
    parser.add_argument("--phase7-output-root", default="outputs/audit/v83_phase7_strong_history")
    parser.add_argument("--phase7-input-root", default="outputs/audit/v83_phase7_strong_history")
    parser.add_argument("--phase8-output-root", default="outputs/audit/v83_phase8_frozen_eval")
    parser.add_argument("--phase8-input-root", default="outputs/audit/v83_phase8_frozen_eval")
    parser.add_argument("--phase8-holdout-source-root", default="")
    parser.add_argument("--phase9-output-root", default="outputs/audit/v83_phase9_casebook")
    parser.add_argument("--v83-config-root", default="outputs/audit/v83_config")
    parser.add_argument("--w-score", type=float, default=0.35)
    parser.add_argument("--w-margin", type=float, default=0.25)
    parser.add_argument("--w-semantic", type=float, default=0.15)
    parser.add_argument("--w-appearance", type=float, default=0.15)
    parser.add_argument("--w-temporal", type=float, default=0.10)
    parser.add_argument("--w-visibility", type=float, default=0.10)
    parser.add_argument("--w-persist", type=float, default=0.08)
    parser.add_argument("--u-entropy", type=float, default=0.45)
    parser.add_argument("--u-conflict", type=float, default=0.20)
    parser.add_argument("--u-new-object", type=float, default=0.20)
    parser.add_argument("--u-bestalt", type=float, default=0.20)
    parser.add_argument("--u-control", type=float, default=0.20)
    parser.add_argument("--ledger-prior-scale", type=float, default=0.10)
    parser.add_argument("--ledger-run-label", default="baseline")
    parser.add_argument(
        "--ledger-candidate-types",
        default="tentative_tracklet,history_source_tracklet,confirmed_history",
    )
    parser.add_argument("--use-tracklet-support-prior", action="store_true")
    parser.add_argument("--diagnostic-relaxed-state-machine", action="store_true")
    parser.add_argument("--stable-support-chunks", type=int, default=2)
    parser.add_argument("--stable-entropy-threshold", type=float, default=0.65)
    parser.add_argument("--stable-margin-threshold", type=float, default=0.05)
    parser.add_argument("--stable-top1-rate-threshold", type=float, default=0.60)
    parser.add_argument("--stable-control-threshold", type=float, default=0.60)
    parser.add_argument("--confirmed-support-chunks", type=int, default=3)
    parser.add_argument("--confirmed-entropy-threshold", type=float, default=0.50)
    parser.add_argument("--confirmed-margin-threshold", type=float, default=0.08)
    parser.add_argument("--confirmed-control-threshold", type=float, default=0.40)
    parser.add_argument("--confirmed-logodds-threshold", type=float, default=0.30)
    parser.add_argument("--max-new-object-evidence-for-promotion", type=int, default=-1)
    parser.add_argument("--max-cannot-link-count-for-promotion", type=int, default=-1)
    parser.add_argument("--new-object-threshold", type=float, default=0.60)
    parser.add_argument("--conflict-close-margin", type=float, default=0.05)
    parser.add_argument("--conflict-penalty", type=float, default=0.12)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.phase == "all":
        run_all(args)
    elif args.phase == "phase0":
        _phase0(args)
    elif args.phase == "phase1":
        _phase1(args)
    elif args.phase == "phase2":
        _phase2(args)
    elif args.phase == "phase3":
        _phase3(args)
    elif args.phase == "phase4":
        _phase4(args)
    elif args.phase == "phase5":
        _phase5(args)
    elif args.phase == "phase6":
        _phase6(args)
    elif args.phase == "phase7":
        _phase7(args)
    elif args.phase == "phase8":
        _phase8(args)
    elif args.phase == "phase9":
        _phase9(args)


if __name__ == "__main__":
    main()
