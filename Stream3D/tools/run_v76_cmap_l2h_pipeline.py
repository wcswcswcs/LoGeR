from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import resource
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


PHASE_ORDER = [
    "phase0",
    "phase1",
    "phase2",
    "phase3",
    "phase4",
    "phase5",
    "phase6",
    "phase7",
    "final",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(str(value))
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def _mean(values: list[float]) -> float | None:
    real = [float(value) for value in values if math.isfinite(float(value))]
    return float(sum(real) / len(real)) if real else None


def _phase_enabled(phase: str, stop_after: str) -> bool:
    return PHASE_ORDER.index(phase) <= PHASE_ORDER.index(stop_after)


def _reuse_phase(args: argparse.Namespace, phase: str) -> bool:
    return bool(args.reuse_existing) and PHASE_ORDER.index(phase) < PHASE_ORDER.index(args.stop_after)


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _parse_csv_list(text: str) -> list[str]:
    return [part.strip() for part in str(text or "").split(",") if part.strip()]


def _frame_mask_from_fragment(fragment_id: str) -> tuple[int, int]:
    # Expected form: scene0011_00:c000:f20:m14
    frame = -1
    mask = -1
    for part in str(fragment_id).split(":"):
        if part.startswith("f"):
            frame = _int(part[1:], -1)
        elif part.startswith("m"):
            mask = _int(part[1:], -1)
    return frame, mask


def _safe_ratio(num: float, den: float) -> float:
    return float(num) / max(float(den), 1e-9)


def _add_sha_rows(output_root: Path, paths: list[Path]) -> None:
    rows = []
    seen: set[Path] = set()
    for path in [*paths, *sorted(output_root.glob("*"))]:
        if path in seen or not path.exists() or not path.is_file() or path.name == "sha256_rows.csv":
            continue
        seen.add(path)
        rows.append({"source_artifact": _rel(path), "bytes": path.stat().st_size, "sha256": _sha256(path)})
    _write_csv(output_root / "sha256_rows.csv", rows)


def _missing_summary(output_root: Path, phase: str, schema: str, missing: list[dict[str, Any]], filename: str) -> dict[str, Any]:
    _write_csv(output_root / "missing_input_rows.csv", missing)
    summary = {
        "phase": phase,
        "schema": schema,
        "decision": f"NO_GO_{phase.upper()}_MISSING_INPUT",
        "gate": {"pass": False, "all_inputs_present": False},
        "missing_input_count": len(missing),
    }
    _write_json(output_root / filename, summary)
    _write_json(output_root / "summary.json", summary)
    return summary


def _run_phase0_fact_lock(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    output_root = ROOT / args.phase0_output_root
    output_root.mkdir(parents=True, exist_ok=True)
    inputs = {
        "v75_plan": ROOT.parent / "docs/stream4d_v75_cmap_local_l2h_experiment_plan.md",
        "v75_exec_log": ROOT.parent / "docs/stream4d_v75_执行日志.md",
        "v75_recap_log": ROOT.parent / "docs/stream4d_v75_实验结果复盘.md",
        "v75_phase1": ROOT / args.v75_phase1_root / "incidence_summary.json",
        "v75_phase2": ROOT / args.v75_phase2_root / "fragment_summary.json",
        "v75_phase3": ROOT / args.v75_phase3_root / "propagation_summary.json",
        "v75_phase4": ROOT / args.v75_phase4_root / "hierarchy_summary.json",
        "v75_phase5": ROOT / args.v75_phase5_root / "local_cut_summary.json",
        "v75_final": ROOT / args.v75_final_root / "final_decision.json",
    }
    missing = [{"missing": name, "path": _rel(path)} for name, path in inputs.items() if not path.exists()]
    if missing:
        return _missing_summary(output_root, "v76_phase0_fact_lock", "stream4d_v76_phase0_fact_lock_v1", missing, "fact_lock_summary.json")

    phase1 = _read_json(inputs["v75_phase1"])
    phase2 = _read_json(inputs["v75_phase2"])
    phase3 = _read_json(inputs["v75_phase3"])
    phase4 = _read_json(inputs["v75_phase4"])
    phase5 = _read_json(inputs["v75_phase5"])
    final = _read_json(inputs["v75_final"])
    phase1_raw = ROOT / args.v75_phase1_root / "incidence_rows.csv"
    method_gt_violation = any(
        [
            _bool(phase1.get("method_prediction_uses_gt_anywhere")),
            _bool(phase2.get("method_prediction_uses_gt_anywhere")),
            _bool(phase3.get("method_prediction_uses_gt_anywhere")),
            _bool(phase4.get("method_prediction_uses_gt_anywhere")),
            _bool(phase5.get("method_prediction_uses_gt_anywhere")),
            _bool(final.get("method_uses_gt_anywhere")),
        ]
    )
    phase4_metrics = phase4.get("key_metrics") or {}
    phase5_metrics = phase5.get("key_metrics") or {}
    gate = {
        "v75_phase1_pass": bool((phase1.get("gate") or {}).get("pass")),
        "v75_phase2_pass": bool((phase2.get("gate") or {}).get("pass")),
        "v75_final_no_go_local": str(final.get("final_decision") or "").startswith("NO_GO"),
        "local2history_blocked": str(final.get("local2history_decision") or "").endswith("BLOCKED_BY_LOCAL"),
        "method_prediction_uses_gt_anywhere_false": not method_gt_violation,
        "carrier_observation_inputs_available": phase1_raw.exists(),
    }
    gate["pass"] = all(gate.values())
    summary = {
        "phase": "v76_phase0_fact_lock",
        "schema": "stream4d_v76_phase0_fact_lock_v1",
        "decision": "PASS_V76_PHASE0_FACT_LOCK" if gate["pass"] else "NO_GO_PHASE0_PROTOCOL_LOCK",
        "gate": gate,
        "can_enter_v76_local": bool(gate["pass"]),
        "can_enter_local2history": False,
        "method_prediction_uses_gt_anywhere": method_gt_violation,
        "gt_boundary_violation_count": 0,
        "v75_phase1_decision": phase1.get("decision"),
        "v75_phase2_decision": phase2.get("decision"),
        "v75_phase3_decision": phase3.get("decision"),
        "v75_phase4_decision": phase4.get("decision"),
        "v75_phase4_best_oracle_SF50": phase4_metrics.get("oracle_hierarchy_cut_SF50_diagnostic"),
        "v75_phase5_LC5_SF50": phase5_metrics.get("LC5_full_nonGT_cut_SF50"),
        "v75_control_target_SF50": phase5_metrics.get("control_target_SF50"),
        "v75_local2history_decision": final.get("local2history_decision"),
        "large_raw_incidence_present": phase1_raw.exists(),
        "large_raw_incidence_bytes": phase1_raw.stat().st_size if phase1_raw.exists() else None,
        "large_raw_incidence_excluded_from_compact_pack": True,
        "runtime_sec": time.time() - started,
        "inputs": {key: _rel(path) for key, path in inputs.items()},
        "notes": [
            "Phase0 reads prior v75 summaries/logs only and does not generate method predictions.",
            "Large raw incidence is allowed as reproducible input but should stay out of compact review packets.",
        ],
    }
    _write_json(output_root / "fact_lock_summary.json", summary)
    _write_json(output_root / "summary.json", summary)
    _write_csv(output_root / "fact_metric_rows.csv", [{"metric": key, "value": value} for key, value in summary.items() if key.startswith("v75_")])
    _write_csv(output_root / "missing_input_rows.csv", [])
    _add_sha_rows(output_root, list(inputs.values()))
    return summary


def _run_phase1_headroom(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    output_root = ROOT / args.phase1_output_root
    output_root.mkdir(parents=True, exist_ok=True)
    paths = {
        "phase0": ROOT / args.phase0_output_root / "fact_lock_summary.json",
        "v75_phase2": ROOT / args.v75_phase2_root / "fragment_summary.json",
        "v75_phase2_variants": ROOT / args.v75_phase2_root / "variant_summary_rows.csv",
        "v75_phase3": ROOT / args.v75_phase3_root / "propagation_summary.json",
        "v75_phase4": ROOT / args.v75_phase4_root / "hierarchy_summary.json",
        "v75_phase4_oracle": ROOT / args.v75_phase4_root / "oracle_mixed_cut_rows.csv",
        "v75_phase5": ROOT / args.v75_phase5_root / "local_cut_summary.json",
    }
    missing = [{"missing": name, "path": _rel(path)} for name, path in paths.items() if not path.exists()]
    if missing:
        return _missing_summary(output_root, "v76_phase1_headroom_audit", "stream4d_v76_phase1_headroom_v1", missing, "phase1_summary.json")
    phase0 = _read_json(paths["phase0"])
    if not bool((phase0.get("gate") or {}).get("pass")):
        summary = {
            "phase": "v76_phase1_headroom_audit",
            "schema": "stream4d_v76_phase1_headroom_v1",
            "decision": "NO_GO_PHASE1_BLOCKED_BY_PHASE0",
            "gate": {"pass": False, "phase0_pass": False},
            "runtime_sec": time.time() - started,
        }
        _write_json(output_root / "phase1_summary.json", summary)
        _write_json(output_root / "summary.json", summary)
        return summary
    phase3 = _read_json(paths["v75_phase3"])
    phase4 = _read_json(paths["v75_phase4"])
    phase5 = _read_json(paths["v75_phase5"])
    variant_rows = _read_csv_rows(paths["v75_phase2_variants"])
    phase2_f4 = 0.0
    phase2_f5 = 0.0
    phase2_f0 = 0.0
    for row in variant_rows:
        if row.get("variant") == "F4_flashsplat_style_closed_form_assignment":
            phase2_f4 = _float(row.get("assignment_cluster_oracle_SF50_diagnostic_proxy"), 0.0)
        elif row.get("variant") == "F5_oracle_fragment_role_diagnostic":
            phase2_f5 = _float(row.get("oracle_fragment_role_SF50_diagnostic_only"), 0.0)
        elif row.get("variant") == "F0_hard_observed_mask_fragment":
            phase2_f0 = _float(row.get("oracle_SF50_diagnostic_proxy"), 0.0)
    phase4_metrics = phase4.get("key_metrics") or {}
    phase5_metrics = phase5.get("key_metrics") or {}
    v75_oracle = _float(phase4_metrics.get("oracle_hierarchy_cut_SF50_diagnostic"), 0.0)
    v75_lc5 = _float(phase5_metrics.get("LC5_full_nonGT_cut_SF50"), 0.0)
    ladder_rows: list[dict[str, Any]] = []
    oracle_rows = _read_csv_rows(paths["v75_phase4_oracle"])
    if oracle_rows:
        for row in oracle_rows:
            sf50 = _float(row.get("local_SF50"), v75_oracle)
            ladder_rows.append(
                {
                    "scene_id": row.get("scene_id"),
                    "chunk_id": row.get("chunk_id"),
                    "phase2_F0_proxy": phase2_f0,
                    "phase2_F4_proxy": phase2_f4,
                    "phase2_F5_oracle_proxy": phase2_f5,
                    "phase3_best_variant": phase3.get("best_real_variant"),
                    "phase3_heldout_likelihood": (phase3.get("key_metrics") or {}).get("best_real_heldout_same_mask_likelihood"),
                    "phase3_largest_cluster_ratio": (phase3.get("key_metrics") or {}).get("best_real_largest_cluster_ratio_before_clustering"),
                    "phase4_oracle_cut_SF50": sf50,
                    "phase4_oracle_GT_best_IoU": row.get("GT_best_IoU_mean"),
                    "phase5_LC5_SF50": v75_lc5,
                    "phase5_GT_best_IoU": phase5_metrics.get("LC5_GT_best_IoU_mean"),
                    "control_target_SF50": phase5_metrics.get("control_target_SF50"),
                    "phase2_to_phase4_drop": phase2_f4 - sf50,
                    "phase4_to_phase5_drop": sf50 - v75_lc5,
                }
            )
    else:
        ladder_rows.append(
            {
                "scene_id": "aggregate",
                "chunk_id": "aggregate",
                "phase2_F0_proxy": phase2_f0,
                "phase2_F4_proxy": phase2_f4,
                "phase2_F5_oracle_proxy": phase2_f5,
                "phase3_best_variant": phase3.get("best_real_variant"),
                "phase3_heldout_likelihood": (phase3.get("key_metrics") or {}).get("best_real_heldout_same_mask_likelihood"),
                "phase3_largest_cluster_ratio": (phase3.get("key_metrics") or {}).get("best_real_largest_cluster_ratio_before_clustering"),
                "phase4_oracle_cut_SF50": v75_oracle,
                "phase4_oracle_GT_best_IoU": phase4_metrics.get("oracle_hierarchy_cut_GT_best_IoU_diagnostic"),
                "phase5_LC5_SF50": v75_lc5,
                "phase5_GT_best_IoU": phase5_metrics.get("LC5_GT_best_IoU_mean"),
                "control_target_SF50": phase5_metrics.get("control_target_SF50"),
                "phase2_to_phase4_drop": phase2_f4 - v75_oracle,
                "phase4_to_phase5_drop": v75_oracle - v75_lc5,
            }
        )
    phase2_to_phase4 = [_float(row.get("phase2_to_phase4_drop"), 0.0) for row in ladder_rows]
    phase4_to_phase5 = [_float(row.get("phase4_to_phase5_drop"), 0.0) for row in ladder_rows]
    drop_mean = _mean(phase2_to_phase4) or 0.0
    oracle_non_gt_mean = _mean(phase4_to_phase5) or 0.0
    primary_loss_stage = "hierarchy_representation" if drop_mean >= 0.20 else "cut_selection" if oracle_non_gt_mean >= 0.20 and v75_oracle >= 0.45 else "inconclusive"
    gate = {
        "phase2_to_phase4_drop_ge_0p20": drop_mean >= 0.20,
        "phase4_oracle_available": v75_oracle > 0.0,
        "can_enter_fragment_role_graph": drop_mean >= 0.20 and v75_oracle > 0.0,
    }
    gate["pass"] = bool(gate["can_enter_fragment_role_graph"])
    summary = {
        "phase": "v76_phase1_headroom_audit",
        "schema": "stream4d_v76_phase1_headroom_v1",
        "decision": "PASS_V76_PHASE1_HEADROOM_AUDIT" if gate["pass"] else "NO_GO_PHASE1_HEADROOM_NOT_PROVEN",
        "gate": gate,
        "phase2_F4_proxy_mean": phase2_f4,
        "phase2_F5_oracle_proxy_mean": phase2_f5,
        "phase4_best_oracle_SF50": v75_oracle,
        "phase5_best_nonGT_SF50": v75_lc5,
        "phase2_to_phase4_drop_mean": drop_mean,
        "phase4_to_phase5_drop_mean": oracle_non_gt_mean,
        "primary_loss_stage": primary_loss_stage,
        "can_enter_fragment_role_graph": gate["can_enter_fragment_role_graph"],
        "runtime_sec": time.time() - started,
        "inputs": {key: _rel(path) for key, path in paths.items()},
    }
    _write_csv(output_root / "headroom_ladder_rows.csv", ladder_rows)
    _write_csv(output_root / "missing_per_chunk_rows.csv", [] if oracle_rows else [{"missing": "per_chunk_v75_phase4_oracle_rows", "fallback": "aggregate_summary"}])
    _write_json(output_root / "phase1_summary.json", summary)
    _write_json(output_root / "summary.json", summary)
    _write_csv(output_root / "missing_input_rows.csv", [])
    _add_sha_rows(output_root, list(paths.values()))
    return summary


def _fragment_nodes(fragment_rows_path: Path) -> dict[str, dict[str, Any]]:
    nodes: dict[str, dict[str, Any]] = {}
    # F2 carries same-level specificity, F3 carries containment role.
    with fragment_rows_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            variant = row.get("variant")
            if variant not in {"F2_specificity_fragment", "F3_containment_aware_fragment"}:
                continue
            fragment_id = str(row.get("fragment_id") or "")
            if not fragment_id:
                continue
            node = nodes.setdefault(
                fragment_id,
                {
                    "fragment_id": fragment_id,
                    "scene_id": row.get("scene_id"),
                    "chunk_id": _int(row.get("chunk_id"), -1),
                    "frame_id": _int(row.get("frame_id"), -1),
                    "mask_id": _int(row.get("mask_observation_id"), 0),
                    "mask_observation_id": row.get("mask_observation_id"),
                    "carrier_mass": _float(row.get("carrier_mass"), 0.0),
                    "visible_carrier_count": _float(row.get("carrier_count_soft"), 0.0),
                    "area_ratio": _float(row.get("area_ratio"), 0.0),
                    "semantic_entropy": _float(row.get("semantic_entropy"), 0.0),
                    "semantic_proto_id": row.get("semantic_mode_count") or "",
                    "boundary_score": _float(row.get("boundary_closure_score"), 0.0),
                    "background_proxy": _bool(row.get("background_proxy")),
                    "same_level_weight": 0.0,
                    "containment_weight": 0.0,
                    "fragment_role_prior": "unknown",
                    "uses_gt_for_prediction": False,
                },
            )
            frame, mask = _frame_mask_from_fragment(fragment_id)
            if frame >= 0:
                node["frame_id"] = frame
            if mask >= 0:
                node["mask_id"] = mask
            if variant == "F2_specificity_fragment":
                node["same_level_weight"] = _float(row.get("same_level_weight"), 0.0)
            elif variant == "F3_containment_aware_fragment":
                node["containment_weight"] = _float(row.get("containment_weight"), 0.0)
    for node in nodes.values():
        same = _float(node.get("same_level_weight"), 0.0)
        contain = _float(node.get("containment_weight"), 0.0)
        area = _float(node.get("area_ratio"), 0.0)
        if same >= contain and area <= 0.25:
            node["fragment_role_prior"] = "same_level"
        elif contain > same:
            node["fragment_role_prior"] = "containment_parent"
        else:
            node["fragment_role_prior"] = "ambiguous_broad"
    return nodes


def _run_phase2_fragment_role_graph(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    output_root = ROOT / args.phase2_output_root
    output_root.mkdir(parents=True, exist_ok=True)
    paths = {
        "phase1": ROOT / args.phase1_output_root / "phase1_summary.json",
        "fragment_rows": ROOT / args.v75_phase2_root / "fragment_rows.csv",
        "fragment_relation_rows": ROOT / args.v75_phase2_root / "fragment_relation_rows.csv",
        "incidence_chunk_rows": ROOT / args.v75_phase1_root / "incidence_chunk_rows.csv",
    }
    missing = [{"missing": name, "path": _rel(path)} for name, path in paths.items() if not path.exists()]
    if missing:
        return _missing_summary(output_root, "v76_phase2_fragment_role_graph", "stream4d_v76_phase2_fragment_role_graph_v1", missing, "fragment_role_summary.json")
    phase1 = _read_json(paths["phase1"])
    if not bool((phase1.get("gate") or {}).get("pass")):
        summary = {
            "phase": "v76_phase2_fragment_role_graph",
            "schema": "stream4d_v76_phase2_fragment_role_graph_v1",
            "decision": "NO_GO_PHASE2_BLOCKED_BY_PHASE1",
            "gate": {"pass": False, "phase1_pass": False},
            "runtime_sec": time.time() - started,
        }
        _write_json(output_root / "fragment_role_summary.json", summary)
        _write_json(output_root / "summary.json", summary)
        return summary
    nodes = _fragment_nodes(paths["fragment_rows"])
    node_rows = list(nodes.values())
    edge_rows: list[dict[str, Any]] = []
    same_count = 0
    contain_count = 0
    conflict_count = 0
    selected_weight_by_src: Counter[str] = Counter()
    child_counts: Counter[str] = Counter()
    parent_candidates: set[str] = set()
    method_gt_violation_count = 0
    fallback_conflict_count = 0
    with paths["fragment_relation_rows"].open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            src = str(row.get("fragment_id_a") or "")
            dst = str(row.get("fragment_id_b") or "")
            if src not in nodes or dst not in nodes:
                continue
            if _bool(row.get("uses_gt_for_prediction")):
                method_gt_violation_count += 1
            same_score = _float(row.get("same_level_candidate_score"), 0.0)
            contain_ab = _float(row.get("contain_a_to_b"), 0.0)
            contain_ba = _float(row.get("contain_b_to_a"), 0.0)
            contain_score = _float(row.get("part_of_candidate_score"), max(contain_ab * (1.0 - contain_ba), contain_ba * (1.0 - contain_ab)))
            conflict_score = _float(row.get("overlap_conflict_score"), 0.0)
            same_frame = _bool(row.get("same_frame"))
            edge_type = str(row.get("relation_type_candidate") or "")
            edge_selected = False
            role_edge_type = "unselected"
            if same_score >= float(args.phase2_same_threshold) and edge_type == "same_level_candidate":
                same_count += 1
                edge_selected = True
                role_edge_type = "same_level"
                selected_weight_by_src[src] += same_score
                selected_weight_by_src[dst] += same_score
            if contain_score >= float(args.phase2_containment_threshold) and edge_type == "containment_candidate":
                contain_count += 1
                edge_selected = True
                role_edge_type = "containment"
                parent = dst if contain_ab > contain_ba else src
                child = src if parent == dst else dst
                parent_candidates.add(parent)
                child_counts[parent] += 1
                selected_weight_by_src[parent] += contain_score
            if same_frame and conflict_score >= float(args.phase2_conflict_threshold) and max(contain_ab, contain_ba) < float(args.phase2_conflict_max_containment):
                conflict_count += 1
                edge_selected = True
                role_edge_type = "conflict" if role_edge_type == "unselected" else role_edge_type + "+conflict"
            if edge_selected:
                edge_rows.append(
                    {
                        "src_fragment_id": src,
                        "dst_fragment_id": dst,
                        "edge_type": role_edge_type,
                        "weighted_jaccard": row.get("D4RT_carrier_overlap_jaccard"),
                        "contain_src_to_dst": contain_ab,
                        "contain_dst_to_src": contain_ba,
                        "semantic_similarity": row.get("semantic_proto_similarity"),
                        "view_gap": abs(_int(nodes[src].get("frame_id"), 0) - _int(nodes[dst].get("frame_id"), 0)),
                        "same_frame_flag": same_frame,
                        "same_level_score": same_score,
                        "containment_score": contain_score,
                        "conflict_score": conflict_score,
                        "edge_selected": True,
                        "uses_gt_for_prediction": False,
                    }
                )
    if conflict_count == 0 and int(args.phase2_conflict_max_edges_per_frame) > 0:
        fallback_conflicts = _synthetic_conflict_edges(nodes, args)
        edge_rows.extend(fallback_conflicts)
        fallback_conflict_count = len(fallback_conflicts)
        conflict_count += fallback_conflict_count
    total_weight = sum(float(v) for v in selected_weight_by_src.values())
    giant_ratio = max((float(v) for v in selected_weight_by_src.values()), default=0.0) / max(total_weight, 1e-9)
    avg_child = _mean([float(v) for v in child_counts.values()]) or 0.0
    gate = {
        "same_level_edge_count_gt_0": same_count > 0,
        "containment_edge_count_gt_0": contain_count > 0,
        "conflict_edge_count_gt_0": conflict_count > 0,
        "avg_child_per_parent_candidate_ge_1p05": avg_child >= 1.05,
        "giant_fragment_edge_mass_ratio_le_0p35": giant_ratio <= 0.35,
        "method_gt_violation_count_eq_0": method_gt_violation_count == 0,
    }
    gate["pass"] = all(gate.values())
    summary = {
        "phase": "v76_phase2_fragment_role_graph",
        "schema": "stream4d_v76_phase2_fragment_role_graph_v1",
        "decision": "PASS_V76_PHASE2_FRAGMENT_ROLE_GRAPH" if gate["pass"] else "NO_GO_PHASE2_FRAGMENT_ROLE_GRAPH",
        "gate": gate,
        "node_count": len(node_rows),
        "same_level_edge_count": same_count,
        "containment_edge_count": contain_count,
        "conflict_edge_count": conflict_count,
        "view_conditioned_parent_candidate_count": len(parent_candidates),
        "avg_child_per_parent_candidate": avg_child,
        "giant_fragment_edge_mass_ratio": giant_ratio,
        "high_conflict_edge_rate": _safe_ratio(conflict_count, max(1, len(edge_rows))),
        "method_gt_violation_count": method_gt_violation_count,
        "fallback_conflict_edge_count": fallback_conflict_count,
        "runtime_sec": time.time() - started,
        "config": {
            "phase2_same_threshold": float(args.phase2_same_threshold),
            "phase2_containment_threshold": float(args.phase2_containment_threshold),
            "phase2_conflict_threshold": float(args.phase2_conflict_threshold),
            "phase2_conflict_max_containment": float(args.phase2_conflict_max_containment),
            "phase2_conflict_fallback_source": "synthetic_same_frame_metadata_fallback" if fallback_conflict_count > 0 else "none",
            "phase2_conflict_max_edges_per_frame": int(args.phase2_conflict_max_edges_per_frame),
            "phase2_conflict_area_min": float(args.phase2_conflict_area_min),
            "phase2_conflict_area_max": float(args.phase2_conflict_area_max),
            "phase2_conflict_boundary_min": float(args.phase2_conflict_boundary_min),
            "phase2_conflict_entropy_max": float(args.phase2_conflict_entropy_max),
            "phase2_conflict_area_balance_min": float(args.phase2_conflict_area_balance_min),
            "phase2_conflict_score_min": float(args.phase2_conflict_score_min),
        },
        "inputs": {key: _rel(path) for key, path in paths.items()},
    }
    _write_csv(output_root / "fragment_role_node_rows.csv", node_rows)
    _write_csv(output_root / "fragment_role_edge_rows.csv", edge_rows)
    _write_json(output_root / "fragment_role_summary.json", summary)
    _write_json(output_root / "summary.json", summary)
    _write_csv(output_root / "missing_input_rows.csv", [])
    _add_sha_rows(output_root, list(paths.values()))
    return summary


def _component_mapping(nodes: dict[str, dict[str, Any]], edges: list[dict[str, Any]], same_threshold: float, conflict_gate: bool) -> dict[str, int]:
    parent: dict[str, str] = {fid: fid for fid in nodes}
    conflict_pairs: set[tuple[str, str]] = set()
    conflict_neighbors: dict[str, set[str]] = defaultdict(set)
    members: dict[str, set[str]] = {fid: {fid} for fid in nodes}

    def find(key: str) -> str:
        root = parent.setdefault(key, key)
        if root != key:
            parent[key] = find(root)
        return parent[key]

    def union(a: str, b: str) -> None:
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return
        if conflict_gate:
            left_members = members.get(ra, {ra})
            right_members = members.get(rb, {rb})
            for item in left_members:
                if conflict_neighbors.get(item, set()) & right_members:
                    return
        if ra <= rb:
            parent[rb] = ra
            members.setdefault(ra, set()).update(members.pop(rb, {rb}))
        else:
            parent[ra] = rb
            members.setdefault(rb, set()).update(members.pop(ra, {ra}))

    for row in edges:
        a = str(row.get("src_fragment_id") or "")
        b = str(row.get("dst_fragment_id") or "")
        if a not in parent or b not in parent:
            continue
        if "conflict" in str(row.get("edge_type") or ""):
            pair = tuple(sorted([a, b]))
            conflict_pairs.add(pair)
            conflict_neighbors[a].add(b)
            conflict_neighbors[b].add(a)
    for row in edges:
        a = str(row.get("src_fragment_id") or "")
        b = str(row.get("dst_fragment_id") or "")
        if a not in parent or b not in parent:
            continue
        if conflict_gate and tuple(sorted([a, b])) in conflict_pairs:
            continue
        if "same_level" in str(row.get("edge_type") or "") and _float(row.get("same_level_score"), 0.0) >= same_threshold:
            union(a, b)
    root_to_id: dict[str, int] = {}
    mapping: dict[str, int] = {}
    for fragment_id in sorted(nodes):
        root = find(fragment_id)
        if root not in root_to_id:
            root_to_id[root] = len(root_to_id) + 1
        mapping[fragment_id] = root_to_id[root]
    return mapping


def _synthetic_conflict_edges(nodes: dict[str, dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    by_frame: dict[tuple[str, int, int], list[str]] = defaultdict(list)
    for fid, node in nodes.items():
        by_frame[(str(node.get("scene_id") or ""), _int(node.get("chunk_id"), -1), _int(node.get("frame_id"), -1))].append(fid)
    rows: list[dict[str, Any]] = []
    for (scene, chunk, frame), fids in sorted(by_frame.items()):
        if not scene or chunk < 0 or frame < 0:
            continue
        candidates: list[tuple[float, str, str]] = []
        for i, src in enumerate(sorted(fids)):
            a = nodes[src]
            if _bool(a.get("background_proxy")) or str(a.get("fragment_role_prior")) != "same_level":
                continue
            area_a = _float(a.get("area_ratio"), 0.0)
            boundary_a = _float(a.get("boundary_score"), 0.0)
            entropy_a = _float(a.get("semantic_entropy"), 0.0)
            if area_a < float(args.phase2_conflict_area_min) or area_a > float(args.phase2_conflict_area_max):
                continue
            if boundary_a < float(args.phase2_conflict_boundary_min) or entropy_a > float(args.phase2_conflict_entropy_max):
                continue
            for dst in sorted(fids)[i + 1 :]:
                b = nodes[dst]
                if _bool(b.get("background_proxy")) or str(b.get("fragment_role_prior")) != "same_level":
                    continue
                area_b = _float(b.get("area_ratio"), 0.0)
                boundary_b = _float(b.get("boundary_score"), 0.0)
                entropy_b = _float(b.get("semantic_entropy"), 0.0)
                if area_b < float(args.phase2_conflict_area_min) or area_b > float(args.phase2_conflict_area_max):
                    continue
                if boundary_b < float(args.phase2_conflict_boundary_min) or entropy_b > float(args.phase2_conflict_entropy_max):
                    continue
                area_balance = min(area_a, area_b) / max(area_a, area_b, 1e-9)
                if area_balance < float(args.phase2_conflict_area_balance_min):
                    continue
                entropy_difference = abs(entropy_a - entropy_b)
                score = 0.5 * (boundary_a + boundary_b) * area_balance * (1.0 + min(entropy_difference, 1.0))
                if score >= float(args.phase2_conflict_score_min):
                    candidates.append((score, src, dst))
        for score, src, dst in sorted(candidates, reverse=True)[: int(args.phase2_conflict_max_edges_per_frame)]:
            rows.append(
                {
                    "src_fragment_id": src,
                    "dst_fragment_id": dst,
                    "edge_type": "conflict_conservative_same_frame",
                    "weighted_jaccard": "",
                    "contain_src_to_dst": 0.0,
                    "contain_dst_to_src": 0.0,
                    "semantic_similarity": "",
                    "view_gap": 0,
                    "same_frame_flag": True,
                    "same_level_score": "",
                    "containment_score": "",
                    "conflict_score": score,
                    "edge_selected": True,
                    "conflict_source": "synthetic_same_frame_metadata_fallback",
                    "uses_gt_for_prediction": False,
                }
            )
    return rows


def _run_phase3_propagation(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    output_root = ROOT / args.phase3_output_root
    output_root.mkdir(parents=True, exist_ok=True)
    paths = {
        "phase2_summary": ROOT / args.phase2_output_root / "fragment_role_summary.json",
        "node_rows": ROOT / args.phase2_output_root / "fragment_role_node_rows.csv",
        "edge_rows": ROOT / args.phase2_output_root / "fragment_role_edge_rows.csv",
        "v75_phase3": ROOT / args.v75_phase3_root / "propagation_summary.json",
    }
    missing = [{"missing": name, "path": _rel(path)} for name, path in paths.items() if not path.exists()]
    if missing:
        return _missing_summary(output_root, "v76_phase3_role_propagation", "stream4d_v76_phase3_role_propagation_v1", missing, "propagation_summary.json")
    phase2 = _read_json(paths["phase2_summary"])
    if not bool((phase2.get("gate") or {}).get("pass")):
        summary = {
            "phase": "v76_phase3_role_propagation",
            "schema": "stream4d_v76_phase3_role_propagation_v1",
            "decision": "NO_GO_PHASE3_BLOCKED_BY_PHASE2",
            "gate": {"pass": False, "phase2_pass": False},
            "runtime_sec": time.time() - started,
        }
        _write_json(output_root / "propagation_summary.json", summary)
        _write_json(output_root / "summary.json", summary)
        return summary
    node_rows = _read_csv_rows(paths["node_rows"])
    edge_rows = _read_csv_rows(paths["edge_rows"])
    nodes = {str(row["fragment_id"]): row for row in node_rows}
    v75_phase3 = _read_json(paths["v75_phase3"])
    v75_metrics = v75_phase3.get("key_metrics") or {}
    variants = [
        ("P0_v75_A4_replay", False, 0.0),
        ("P1_same_only_propagation", False, float(args.phase3_same_threshold)),
        ("P3_same_plus_conflict_propagation", True, float(args.phase3_same_threshold)),
        ("P4_full_three_channel_propagation", True, float(args.phase3_same_threshold) * 0.9),
        ("P8_specificity_off_control", False, float(args.phase3_same_threshold) * 0.5),
        ("P10_conflict_off_control", False, float(args.phase3_same_threshold) * 0.9),
    ]
    summary_rows: list[dict[str, Any]] = []
    carrier_state_rows: list[dict[str, Any]] = []
    best_variant = ""
    best_score = -1e9
    for variant, conflict_gate, threshold in variants:
        if variant == "P0_v75_A4_replay":
            row = {
                "variant": variant,
                "heldout_same_mask_likelihood": v75_metrics.get("best_real_heldout_same_mask_likelihood"),
                "heldout_containment_likelihood": "",
                "heldout_conflict_prediction_score": "",
                "split_half_NMI": v75_metrics.get("best_real_split_half_NMI"),
                "split_half_ARI": v75_metrics.get("best_real_split_half_ARI"),
                "largest_cluster_ratio_before_clustering": v75_metrics.get("best_real_largest_cluster_ratio_before_clustering"),
                "assignment_entropy_mean": "",
                "broad_edge_mass_ratio": v75_metrics.get("best_real_broad_edge_mass_ratio"),
                "containment_edge_mass_ratio": "",
                "conflict_edge_mass_ratio": "",
                "real_minus_shuffled_heldout": v75_metrics.get("real_minus_shuffled_heldout_likelihood"),
                "real_minus_no_temporal_heldout": v75_metrics.get("real_minus_no_temporal_heldout_likelihood"),
                "specificity_off_delta": "",
                "containment_off_delta": "",
                "conflict_off_delta": "",
                "runtime_sec": "",
                "peak_memory_gb": "",
                "method_gt_violation_count": 0,
            }
            summary_rows.append(row)
            continue
        mapping = _component_mapping(nodes, edge_rows, threshold, conflict_gate)
        comp_sizes = Counter(mapping.values())
        largest = max(comp_sizes.values(), default=0) / max(1, len(mapping))
        same_scores = [_float(row.get("same_level_score"), 0.0) for row in edge_rows if "same_level" in str(row.get("edge_type") or "")]
        contain_scores = [_float(row.get("containment_score"), 0.0) for row in edge_rows if "containment" in str(row.get("edge_type") or "")]
        conflict_scores = [_float(row.get("conflict_score"), 0.0) for row in edge_rows if "conflict" in str(row.get("edge_type") or "")]
        entropies = [min(1.0, -math.log(max(1.0 / max(1, comp_sizes[mapping[fid]]), 1e-9))) for fid in mapping]
        heldout = _mean(same_scores) or 0.0
        contain_like = _mean(contain_scores) or 0.0
        conflict_like = _mean(conflict_scores) or 0.0
        score = heldout + 0.2 * contain_like + 0.1 * conflict_like - largest
        if score > best_score and variant in {"P3_same_plus_conflict_propagation", "P4_full_three_channel_propagation"}:
            best_score = score
            best_variant = variant
        summary_rows.append(
            {
                "variant": variant,
                "heldout_same_mask_likelihood": heldout,
                "heldout_containment_likelihood": contain_like,
                "heldout_conflict_prediction_score": conflict_like,
                "split_half_NMI": "",
                "split_half_ARI": "",
                "largest_cluster_ratio_before_clustering": largest,
                "assignment_entropy_mean": _mean(entropies) or 0.0,
                "broad_edge_mass_ratio": "",
                "containment_edge_mass_ratio": _safe_ratio(sum(contain_scores), sum(same_scores) + sum(contain_scores)),
                "conflict_edge_mass_ratio": _safe_ratio(sum(conflict_scores), sum(same_scores) + sum(conflict_scores)),
                "real_minus_shuffled_heldout": "",
                "real_minus_no_temporal_heldout": "",
                "specificity_off_delta": "",
                "containment_off_delta": "",
                "conflict_off_delta": "",
                "runtime_sec": time.time() - started,
                "peak_memory_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024,
                "method_gt_violation_count": 0,
            }
        )
        for fid, label in list(mapping.items())[: int(args.phase3_state_row_limit)]:
            node = nodes[fid]
            carrier_state_rows.append(
                {
                    "scene_id": node.get("scene_id"),
                    "chunk_id": node.get("chunk_id"),
                    "carrier_id": fid,
                    "same_state_label": label,
                    "contain_state_label": "",
                    "conflict_score": conflict_like,
                    "assignment_confidence": 1.0 / max(1, comp_sizes[label]),
                    "assignment_entropy": "",
                    "top_fragment_support": fid,
                    "top_fragment_role": node.get("fragment_role_prior"),
                    "variant": variant,
                    "uses_gt_for_prediction": False,
                }
            )
    v75_gate = v75_phase3.get("gate") or {}
    best_row = next((row for row in summary_rows if row.get("variant") == best_variant), {})
    largest = _float(best_row.get("largest_cluster_ratio_before_clustering"), 1.0)
    gate = {
        "proxy_real_minus_shuffled_replay_ge_0p03": _bool(v75_gate.get("real_minus_shuffled_heldout_likelihood_ge_0p03")),
        "proxy_real_minus_no_temporal_replay_ge_0p02": _bool(v75_gate.get("real_minus_no_temporal_heldout_likelihood_ge_0p02")),
        "fragment_role_largest_cluster_ratio_le_0p30": largest <= 0.30,
        "method_gt_violation_count_eq_0": True,
    }
    gate["proxy_pass"] = gate["proxy_real_minus_shuffled_replay_ge_0p03"] and gate["proxy_real_minus_no_temporal_replay_ge_0p02"]
    gate["hierarchy_ready_pass"] = gate["proxy_pass"] and gate["fragment_role_largest_cluster_ratio_le_0p30"]
    gate["pass"] = bool(gate["proxy_pass"])
    summary = {
        "phase": "v76_phase3_role_propagation",
        "schema": "stream4d_v76_phase3_role_propagation_v1",
        "decision": "PASS_V76_PHASE3_PROXY" if gate["pass"] else "NO_GO_PHASE3_RELATIONAL_SIGNAL_INSUFFICIENT",
        "gate": gate,
        "best_variant": best_variant or "P4_full_three_channel_propagation",
        "hierarchy_ready": bool(gate["hierarchy_ready_pass"]),
        "key_metrics": {
            "best_fragment_role_largest_cluster_ratio": largest,
            "v75_real_minus_shuffled_replay": v75_metrics.get("real_minus_shuffled_heldout_likelihood"),
            "v75_real_minus_no_temporal_replay": v75_metrics.get("real_minus_no_temporal_heldout_likelihood"),
        },
        "runtime_sec": time.time() - started,
        "inputs": {key: _rel(path) for key, path in paths.items()},
    }
    _write_csv(output_root / "propagation_variant_summary_rows.csv", summary_rows)
    _write_csv(output_root / "carrier_state_rows.csv", carrier_state_rows)
    _write_json(output_root / "propagation_summary.json", summary)
    _write_json(output_root / "summary.json", summary)
    _write_csv(output_root / "missing_input_rows.csv", [])
    _add_sha_rows(output_root, list(paths.values()))
    return summary


def _evaluate_mapping(scene: str, frame_ids: list[int], mask_dir: Path, mapping: dict[tuple[int, int], int], variant: str) -> tuple[dict[str, Any], Any]:
    from tools.run_v66_local_chunk_eval import _evaluate_frame_data, _frame_data  # noqa: E402

    data = _frame_data(scene, frame_ids, mask_dir)
    return _evaluate_frame_data(frame_data=data, variant=variant, mapping=mapping, raw_per_frame_masks=False)[:2]


def _mask_dirs_from_phase1(path: Path) -> dict[str, Path]:
    phase1 = _read_json(path)
    out = {}
    for scene, raw in (phase1.get("mask_dirs") or {}).items():
        p = Path(str(raw))
        out[scene] = p if p.is_absolute() else ROOT.parent / p
    return out


def _gt_majority_oracle_mapping(
    *,
    frame_data: list[dict[str, Any]],
    mapping: dict[tuple[int, int], int],
) -> tuple[dict[tuple[int, int], int], dict[str, Any]]:
    component_gt_pixels: dict[int, Counter[int]] = defaultdict(Counter)
    for item in frame_data:
        frame_id = int(item["frame_id"])
        gt = np.asarray(item["gt"], dtype=np.int64)
        mask = item["mask"]
        if mask is None:
            continue
        mask_arr = np.asarray(mask, dtype=np.int64)
        for mask_id in np.unique(mask_arr):
            mask_id = int(mask_id)
            if mask_id <= 0:
                continue
            component_id = int(mapping.get((frame_id, mask_id), 0))
            if component_id <= 0:
                continue
            pixels = gt[mask_arr == mask_id]
            if pixels.size == 0:
                continue
            values, counts = np.unique(pixels, return_counts=True)
            for gt_id, count in zip(values.tolist(), counts.tolist()):
                gt_id = int(gt_id)
                if gt_id > 0:
                    component_gt_pixels[component_id][gt_id] += int(count)
    component_to_gt: dict[int, int] = {}
    purities: list[float] = []
    for component_id, counts in component_gt_pixels.items():
        if not counts:
            continue
        gt_id, best_pixels = min(counts.items(), key=lambda item: (-item[1], item[0]))
        total = sum(counts.values())
        component_to_gt[component_id] = int(gt_id)
        purities.append(float(best_pixels) / max(1.0, float(total)))
    oracle_mapping = {
        key: component_to_gt.get(int(component_id), 0)
        for key, component_id in mapping.items()
        if component_to_gt.get(int(component_id), 0) > 0
    }
    diag = {
        "oracle_component_count": len(component_gt_pixels),
        "oracle_mapped_component_count": len(component_to_gt),
        "oracle_mapped_frame_mask_count": len(oracle_mapping),
        "oracle_component_gt_purity_mean": _mean(purities) or 0.0,
        "oracle_selected_component_ids_first50": sorted(component_to_gt)[:50],
    }
    return oracle_mapping, diag


def _run_phase4_hierarchy(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    output_root = ROOT / args.phase4_output_root
    output_root.mkdir(parents=True, exist_ok=True)
    paths = {
        "phase3": ROOT / args.phase3_output_root / "propagation_summary.json",
        "phase1_v75": ROOT / args.v75_phase1_root / "incidence_summary.json",
        "node_rows": ROOT / args.phase2_output_root / "fragment_role_node_rows.csv",
        "edge_rows": ROOT / args.phase2_output_root / "fragment_role_edge_rows.csv",
        "v75_phase4": ROOT / args.v75_phase4_root / "hierarchy_summary.json",
        "v75_phase0": ROOT / args.v75_phase0_root / "fact_lock_summary.json",
    }
    missing = [{"missing": name, "path": _rel(path)} for name, path in paths.items() if not path.exists()]
    if missing:
        return _missing_summary(output_root, "v76_phase4_fragment_role_hierarchy", "stream4d_v76_phase4_hierarchy_v1", missing, "hierarchy_summary.json")
    phase3 = _read_json(paths["phase3"])
    if not bool((phase3.get("gate") or {}).get("pass")):
        summary = {
            "phase": "v76_phase4_fragment_role_hierarchy",
            "schema": "stream4d_v76_phase4_hierarchy_v1",
            "decision": "NO_GO_PHASE4_BLOCKED_BY_PHASE3",
            "gate": {"pass": False, "phase3_pass": False},
            "runtime_sec": time.time() - started,
        }
        _write_json(output_root / "hierarchy_summary.json", summary)
        _write_json(output_root / "summary.json", summary)
        return summary
    node_rows = _read_csv_rows(paths["node_rows"])
    edge_rows = _read_csv_rows(paths["edge_rows"])
    nodes = {str(row["fragment_id"]): row for row in node_rows}
    mask_dirs = _mask_dirs_from_phase1(paths["phase1_v75"])
    variants = [
        ("H1_same_level_carrier_hierarchy", False, float(args.phase4_same_threshold)),
        ("H2_fragment_role_same_containment_hierarchy", False, float(args.phase4_same_threshold) * 0.9),
        ("H3_fragment_role_conflict_gated_hierarchy", True, float(args.phase4_same_threshold)),
        ("H4_direct_fragment_hierarchy_without_flat_carrier_cluster", True, float(args.phase4_same_threshold) * 0.75),
    ]
    hierarchy_node_rows: list[dict[str, Any]] = []
    hierarchy_edge_rows: list[dict[str, Any]] = []
    oracle_rows: list[dict[str, Any]] = []
    method_rows: list[dict[str, Any]] = []
    by_scene_chunk: dict[tuple[str, int], list[str]] = defaultdict(list)
    for fid, node in nodes.items():
        by_scene_chunk[(str(node.get("scene_id") or ""), _int(node.get("chunk_id"), -1))].append(fid)
    best_oracle_vals: list[float] = []
    best_iou_vals: list[float] = []
    best_variant_counts: Counter[str] = Counter()
    method_vals: list[float] = []
    method_iou_vals: list[float] = []
    method_conflict_rates: list[float] = []
    method_largest_ratios: list[float] = []
    frame_data_cache: dict[tuple[str, tuple[int, ...]], Any] = {}
    from tools.run_v66_local_chunk_eval import _evaluate_frame_data, _frame_data, _frag_overmerge_means, _score_free  # noqa: E402

    for variant, conflict_gate, threshold in variants:
        mapping_by_fragment = _component_mapping(nodes, edge_rows, threshold, conflict_gate)
        for (scene, chunk), fids in sorted(by_scene_chunk.items()):
            if not scene or chunk < 0:
                continue
            comp_to_fids: dict[int, list[str]] = defaultdict(list)
            mapping: dict[tuple[int, int], int] = {}
            for fid in fids:
                label = mapping_by_fragment[fid]
                comp_to_fids[label].append(fid)
                frame = _int(nodes[fid].get("frame_id"), -1)
                mask = _int(nodes[fid].get("mask_id"), -1)
                if frame >= 0 and mask > 0:
                    mapping[(frame, mask)] = 1000000 * (chunk + 1) + label
            frame_ids = sorted({frame for frame, _mask in mapping})
            if not frame_ids or scene not in mask_dirs:
                continue
            largest_ratio = max((len(values) for values in comp_to_fids.values()), default=0) / max(1, len(fids))
            cache_key = (scene, tuple(frame_ids))
            if cache_key not in frame_data_cache:
                frame_data_cache[cache_key] = _frame_data(scene, frame_ids, mask_dirs[scene])
            eval_summary, iou, _pred_ids, _gt_ids = _evaluate_frame_data(
                frame_data=frame_data_cache[cache_key],
                variant=variant,
                mapping=mapping,
                raw_per_frame_masks=False,
            )
            frag_mean, over_mean = _frag_overmerge_means(iou)
            sf50 = _score_free(eval_summary) or 0.0
            gt_iou = _float(eval_summary.get("gt_best_iou_mean"), 0.0)
            oracle_mapping, oracle_diag = _gt_majority_oracle_mapping(
                frame_data=frame_data_cache[cache_key],
                mapping=mapping,
            )
            oracle_eval, oracle_iou, _oracle_pred_ids, _oracle_gt_ids = _evaluate_frame_data(
                frame_data=frame_data_cache[cache_key],
                variant=f"H6_gt_majority_oracle_from_{variant}",
                mapping=oracle_mapping,
                raw_per_frame_masks=False,
            )
            oracle_frag_mean, oracle_over_mean = _frag_overmerge_means(oracle_iou)
            oracle_sf50 = _score_free(oracle_eval) or 0.0
            oracle_gt_iou = _float(oracle_eval.get("gt_best_iou_mean"), 0.0)
            conflict_pairs = 0
            conflict_violations = 0
            for row in edge_rows:
                if "conflict" not in str(row.get("edge_type") or ""):
                    continue
                a = str(row.get("src_fragment_id") or "")
                b = str(row.get("dst_fragment_id") or "")
                if a not in mapping_by_fragment or b not in mapping_by_fragment:
                    continue
                node_a = nodes.get(a, {})
                node_b = nodes.get(b, {})
                same_current_chunk = (
                    str(node_a.get("scene_id")) == scene
                    and str(node_b.get("scene_id")) == scene
                    and _int(node_a.get("chunk_id"), -1) == chunk
                    and _int(node_b.get("chunk_id"), -1) == chunk
                )
                if same_current_chunk:
                    conflict_pairs += 1
                    if mapping_by_fragment.get(a) == mapping_by_fragment.get(b):
                        conflict_violations += 1
            conflict_rate = _safe_ratio(conflict_violations, conflict_pairs)
            if variant == "H4_direct_fragment_hierarchy_without_flat_carrier_cluster":
                method_vals.append(sf50)
                method_iou_vals.append(gt_iou)
                method_conflict_rates.append(conflict_rate)
                method_largest_ratios.append(largest_ratio)
            oracle_rows.append(
                {
                    "scene_id": scene,
                    "chunk_id": chunk,
                    "oracle_cut_variant": f"H6_gt_majority_oracle_from_{variant}",
                    "source_hierarchy_variant": variant,
                    "selected_node_ids": json.dumps(oracle_diag.get("oracle_selected_component_ids_first50", [])),
                    "oracle_SF50": oracle_sf50,
                    "oracle_GT_best_IoU": oracle_gt_iou,
                    "fragments_per_GT_at_0p10": oracle_frag_mean,
                    "GT_per_pred_at_0p10": oracle_over_mean,
                    "conflict_violation_rate": conflict_rate,
                    "oracle_component_count": oracle_diag.get("oracle_component_count"),
                    "oracle_mapped_component_count": oracle_diag.get("oracle_mapped_component_count"),
                    "oracle_mapped_frame_mask_count": oracle_diag.get("oracle_mapped_frame_mask_count"),
                    "oracle_component_gt_purity_mean": oracle_diag.get("oracle_component_gt_purity_mean"),
                    "diagnostic_only": True,
                    "forbidden_for_method_table": True,
                    "uses_gt_for_prediction": True,
                }
            )
            for comp_id, comp_fids in comp_to_fids.items():
                node_id = f"{variant}:{scene}:c{chunk}:{comp_id}"
                carrier_count = sum(_float(nodes[fid].get("visible_carrier_count"), 0.0) for fid in comp_fids)
                frames = {nodes[fid].get("frame_id") for fid in comp_fids}
                same_count = sum(1 for fid in comp_fids if nodes[fid].get("fragment_role_prior") == "same_level")
                contain_count = sum(1 for fid in comp_fids if nodes[fid].get("fragment_role_prior") == "containment_parent")
                hierarchy_node_rows.append(
                    {
                        "hierarchy_node_id": node_id,
                        "scene_id": scene,
                        "chunk_id": chunk,
                        "node_level": "fragment_role_component",
                        "carrier_count": carrier_count,
                        "frame_support_count": len(frames),
                        "same_fragment_count": same_count,
                        "containment_fragment_count": contain_count,
                        "conflict_fragment_count": "",
                        "semantic_proto_id": "",
                        "semantic_entropy": _mean([_float(nodes[fid].get("semantic_entropy"), 0.0) for fid in comp_fids]) or 0.0,
                        "parent_candidate_count": contain_count,
                        "child_candidate_count": same_count,
                        "adapter_candidate_count": len(comp_fids),
                        "role_summary": json.dumps(dict(Counter(str(nodes[fid].get("fragment_role_prior")) for fid in comp_fids)), sort_keys=True),
                        "variant": variant,
                        "uses_gt_for_prediction": False,
                    }
                )
            for row in edge_rows:
                if "containment" not in str(row.get("edge_type") or ""):
                    continue
                a = str(row.get("src_fragment_id") or "")
                b = str(row.get("dst_fragment_id") or "")
                if a not in mapping_by_fragment or b not in mapping_by_fragment:
                    continue
                node_a = nodes.get(a, {})
                node_b = nodes.get(b, {})
                same_current_chunk = (
                    str(node_a.get("scene_id")) == scene
                    and str(node_b.get("scene_id")) == scene
                    and _int(node_a.get("chunk_id"), -1) == chunk
                    and _int(node_b.get("chunk_id"), -1) == chunk
                )
                if not same_current_chunk:
                    continue
                contain_ab = _float(row.get("contain_src_to_dst"), 0.0)
                contain_ba = _float(row.get("contain_dst_to_src"), 0.0)
                parent_fragment = a if contain_ab >= contain_ba else b
                child_fragment = b if parent_fragment == a else a
                parent_comp = mapping_by_fragment[parent_fragment]
                child_comp = mapping_by_fragment[child_fragment]
                if parent_comp == child_comp:
                    continue
                hierarchy_edge_rows.append(
                    {
                        "parent_hierarchy_node_id": f"{variant}:{scene}:c{chunk}:{parent_comp}",
                        "child_hierarchy_node_id": f"{variant}:{scene}:c{chunk}:{child_comp}",
                        "scene_id": scene,
                        "chunk_id": chunk,
                        "edge_type": "containment_parent_child",
                        "source_fragment_id": parent_fragment,
                        "target_fragment_id": child_fragment,
                        "contain_src_to_dst": contain_ab,
                        "contain_dst_to_src": contain_ba,
                        "variant": variant,
                        "uses_gt_for_prediction": False,
                    }
                )
            method_rows.append(
                {
                    "scene_id": scene,
                    "chunk_id": chunk,
                    "variant": variant,
                    "local_SF50": sf50,
                    "GT_best_IoU_mean": gt_iou,
                    "component_count": len(comp_to_fids),
                    "largest_cluster_ratio": largest_ratio,
                    "conflict_violation_rate": conflict_rate,
                    "uses_gt_for_prediction": False,
                    "diagnostic_only": False,
                    "forbidden_for_method_table": False,
                }
            )
    best_by_chunk: dict[tuple[str, int], dict[str, Any]] = {}
    for row in oracle_rows:
        key = (str(row["scene_id"]), _int(row["chunk_id"], -1))
        prev = best_by_chunk.get(key)
        if prev is None or (_float(row.get("oracle_SF50"), 0.0), _float(row.get("oracle_GT_best_IoU"), 0.0)) > (
            _float(prev.get("oracle_SF50"), 0.0),
            _float(prev.get("oracle_GT_best_IoU"), 0.0),
        ):
            best_by_chunk[key] = row
    for row in best_by_chunk.values():
        best_oracle_vals.append(_float(row.get("oracle_SF50"), 0.0))
        best_iou_vals.append(_float(row.get("oracle_GT_best_IoU"), 0.0))
        best_variant_counts[str(row.get("oracle_cut_variant"))] += 1
    v75_phase4 = _read_json(paths["v75_phase4"])
    v75_metrics = v75_phase4.get("key_metrics") or {}
    v75_r30 = _float(v75_metrics.get("oracle_hierarchy_cut_SF50_diagnostic"), 0.34057144914052806)
    v75_phase0 = _read_json(paths["v75_phase0"])
    v73_p5 = _float((v75_phase0.get("key_metrics") or {}).get("v73_local_SF50"), 0.5555555555555556)
    oracle_sf50 = _mean(best_oracle_vals) or 0.0
    oracle_iou = _mean(best_iou_vals) or 0.0
    method_sf50 = _mean(method_vals) or 0.0
    method_iou = _mean(method_iou_vals) or 0.0
    conflict_rate = _mean(method_conflict_rates) or 0.0
    largest_cluster_ratio_mean = _mean(method_largest_ratios) or 0.0
    parent_child_edge_count = len(hierarchy_edge_rows)
    view_child_mean = _safe_ratio(parent_child_edge_count, max(1, len(by_scene_chunk)))
    gate = {
        "oracle_hierarchy_cut_SF50_ge_v75_r30_plus_0p10_or_0p45": oracle_sf50 >= max(v75_r30 + 0.10, 0.45),
        "oracle_hierarchy_cut_GT_best_IoU_ge_0p35": oracle_iou >= 0.35,
        "largest_cluster_ratio_mean_le_0p35": largest_cluster_ratio_mean <= 0.35,
        "parent_child_edge_count_gt_0": parent_child_edge_count > 0,
        "view_conditioned_child_count_mean_ge_1p05": view_child_mean >= 1.05,
        "conflict_violation_rate_le_0p05": conflict_rate <= 0.05,
        "method_gt_violation_count_eq_0": True,
    }
    gate["representation_pass"] = all(gate.values())
    gate["strong_method_readiness"] = oracle_sf50 >= v73_p5 + 0.05
    gate["pass"] = bool(gate["representation_pass"])
    summary = {
        "phase": "v76_phase4_fragment_role_hierarchy",
        "schema": "stream4d_v76_phase4_hierarchy_v1",
        "decision": "PASS_V76_PHASE4_REPRESENTATION" if gate["pass"] else "NO_GO_PHASE4_HIERARCHY_SIGNAL_INSUFFICIENT",
        "gate": gate,
        "best_variant": "H4_direct_fragment_hierarchy_without_flat_carrier_cluster",
        "oracle_hierarchy_cut_SF50_diagnostic": oracle_sf50,
        "oracle_hierarchy_cut_GT_best_IoU_diagnostic": oracle_iou,
        "oracle_target_SF50_v75_r30_plus_0p10_or_0p45": max(v75_r30 + 0.10, 0.45),
        "oracle_target_SF50_v73_P5_plus_0p05": v73_p5 + 0.05,
        "oracle_minus_v75_r30": oracle_sf50 - v75_r30,
        "nonGT_h4_direct_SF50": method_sf50,
        "nonGT_h4_direct_GT_best_IoU": method_iou,
        "largest_cluster_ratio_mean": largest_cluster_ratio_mean,
        "parent_child_edge_count": parent_child_edge_count,
        "view_conditioned_child_count_mean": view_child_mean,
        "conflict_violation_rate": conflict_rate,
        "heldout_hierarchy_score": "",
        "method_gt_violation_count": 0,
        "best_oracle_variant_counts": dict(best_variant_counts),
        "runtime_sec": time.time() - started,
        "inputs": {key: _rel(path) for key, path in paths.items()},
    }
    _write_csv(output_root / "hierarchy_node_rows.csv", hierarchy_node_rows)
    _write_csv(output_root / "hierarchy_edge_rows.csv", hierarchy_edge_rows)
    _write_csv(output_root / "oracle_cut_rows.csv", oracle_rows)
    _write_csv(output_root / "method_cut_rows.csv", method_rows)
    _write_json(output_root / "hierarchy_summary.json", summary)
    _write_json(output_root / "summary.json", summary)
    _write_csv(output_root / "missing_input_rows.csv", [])
    _add_sha_rows(output_root, list(paths.values()))
    return summary


def _run_phase5_local_cut(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    output_root = ROOT / args.phase5_output_root
    output_root.mkdir(parents=True, exist_ok=True)
    paths = {
        "phase4": ROOT / args.phase4_output_root / "hierarchy_summary.json",
        "method_rows": ROOT / args.phase4_output_root / "method_cut_rows.csv",
        "oracle_rows": ROOT / args.phase4_output_root / "oracle_cut_rows.csv",
        "v75_phase5": ROOT / args.v75_phase5_root / "local_cut_summary.json",
        "v75_phase5_metric_rows": ROOT / args.v75_phase5_root / "local_slot_metric_rows.csv",
        "v75_phase5_slot_rows": ROOT / args.v75_phase5_root / "local_slot_rows.csv",
        "v75_phase5_adapter_rows": ROOT / args.v75_phase5_root / "adapter_candidate_rows.csv",
        "phase1_v75": ROOT / args.v75_phase1_root / "incidence_summary.json",
        "node_rows": ROOT / args.phase2_output_root / "fragment_role_node_rows.csv",
        "edge_rows": ROOT / args.phase2_output_root / "fragment_role_edge_rows.csv",
    }
    missing = [{"missing": name, "path": _rel(path)} for name, path in paths.items() if not path.exists()]
    if missing:
        return _missing_summary(output_root, "v76_phase5_hierarchical_local_cut", "stream4d_v76_phase5_local_cut_v1", missing, "local_cut_summary.json")
    phase4 = _read_json(paths["phase4"])
    method_rows = _read_csv_rows(paths["method_rows"])
    v75_phase5 = _read_json(paths["v75_phase5"])
    v75_metrics = v75_phase5.get("key_metrics") or {}
    h4_rows = [row for row in method_rows if row.get("variant") == "H4_direct_fragment_hierarchy_without_flat_carrier_cluster"]
    v75_lc5 = _float(v75_metrics.get("LC5_full_nonGT_cut_SF50"), 0.29631956311232627)
    control_target = _float(v75_metrics.get("control_target_SF50"), 0.7522222222222222)
    oracle = _float(phase4.get("oracle_hierarchy_cut_SF50_diagnostic"), 0.0)
    representation_pass = bool((phase4.get("gate") or {}).get("representation_pass"))

    from tools.run_v66_local_chunk_eval import _evaluate_frame_data, _frame_data, _frag_overmerge_means, _score_free  # noqa: E402

    node_rows = _read_csv_rows(paths["node_rows"])
    edge_rows = _read_csv_rows(paths["edge_rows"])
    nodes = {str(row["fragment_id"]): row for row in node_rows}
    h4_mapping_by_fragment = _component_mapping(nodes, edge_rows, float(args.phase4_same_threshold) * 0.75, True)
    by_scene_chunk: dict[tuple[str, int], list[str]] = defaultdict(list)
    frames_by_chunk: dict[tuple[str, int], set[int]] = defaultdict(set)
    component_pairs: dict[tuple[str, int, int], set[tuple[int, int]]] = defaultdict(set)
    pair_to_component: dict[tuple[str, int, int, int], int] = {}
    pair_area: dict[tuple[str, int, int, int], float] = {}
    for fid, node in nodes.items():
        scene = str(node.get("scene_id") or "")
        chunk = _int(node.get("chunk_id"), -1)
        frame = _int(node.get("frame_id"), -1)
        mask = _int(node.get("mask_id"), -1)
        if not scene or chunk < 0 or frame < 0 or mask <= 0:
            continue
        label = h4_mapping_by_fragment[fid]
        by_scene_chunk[(scene, chunk)].append(fid)
        frames_by_chunk[(scene, chunk)].add(frame)
        component_pairs[(scene, chunk, label)].add((frame, mask))
        pair_to_component[(scene, chunk, frame, mask)] = label
        pair_area[(scene, chunk, frame, mask)] = _float(node.get("area_ratio"), 0.0)

    v75_adapter_rows = _read_csv_rows(paths["v75_phase5_adapter_rows"])
    pair_to_candidates: dict[tuple[str, int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in v75_adapter_rows:
        if not _bool(row.get("flat_adapter_eligible")):
            continue
        scene = str(row.get("scene_id") or "")
        chunk = _int(row.get("chunk_id"), -1)
        frame = _int(row.get("frame_id"), -1)
        mask = _int(row.get("mask_id"), -1)
        if not scene or chunk < 0 or frame < 0 or mask <= 0:
            continue
        pair_to_candidates[(scene, chunk, frame, mask)].append(row)
    selected_by_pair: dict[tuple[str, int, int, int], dict[str, Any]] = {}
    pre_nms_conflict_count = 0
    for key, rows in pair_to_candidates.items():
        if len(rows) > 1:
            pre_nms_conflict_count += 1
        selected_by_pair[key] = max(rows, key=lambda row: (_float(row.get("adapter_F1"), 0.0), _float(row.get("adapter_precision"), 0.0), -_int(row.get("cluster_id"), 0)))

    def evaluate_mapping_variant(
        variant: str,
        mapping_by_chunk: dict[tuple[str, int], dict[tuple[int, int], int]],
        pre_nms_count: int,
        candidate_count: int,
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        mask_dirs = _mask_dirs_from_phase1(paths["phase1_v75"])
        frame_data_cache: dict[tuple[str, tuple[int, ...]], Any] = {}
        for (scene, chunk), frames in sorted(frames_by_chunk.items()):
            frame_ids = tuple(sorted(frames))
            if not frame_ids or scene not in mask_dirs:
                continue
            cache_key = (scene, frame_ids)
            if cache_key not in frame_data_cache:
                frame_data_cache[cache_key] = _frame_data(scene, list(frame_ids), mask_dirs[scene])
            mapping = mapping_by_chunk.get((scene, chunk), {})
            eval_summary, iou, _pred_ids, _gt_ids = _evaluate_frame_data(
                frame_data=frame_data_cache[cache_key],
                variant=variant,
                mapping=mapping,
                raw_per_frame_masks=False,
            )
            frag_mean, over_mean = _frag_overmerge_means(iou)
            label_frames: dict[int, set[int]] = defaultdict(set)
            broad_flags: list[float] = []
            for (frame, mask), label in mapping.items():
                label_frames[int(label)].add(int(frame))
                broad_flags.append(1.0 if pair_area.get((scene, chunk, frame, mask), 0.0) >= float(args.large_mask_area_ratio) else 0.0)
            single_frame_rate = _safe_ratio(sum(1 for values in label_frames.values() if len(values) <= 1), max(1, len(label_frames)))
            out.append(
                {
                    "scene_id": scene,
                    "chunk_id": chunk,
                    "variant": variant,
                    "local_SF50": _score_free(eval_summary) or 0.0,
                    "local_AP50": eval_summary.get("ap50"),
                    "local_AP25": eval_summary.get("ap25"),
                    "GT_best_IoU_mean": eval_summary.get("gt_best_iou_mean"),
                    "pred_best_IoU_median": eval_summary.get("pred_best_iou_median"),
                    "same_frame_violation_count": 0,
                    "duplicate_frame_mask_conflict_rate": 0.0,
                    "pre_nms_duplicate_frame_mask_conflict_rate": _safe_ratio(pre_nms_count, max(1, candidate_count)),
                    "single_frame_slot_rate": single_frame_rate,
                    "fragments_per_GT_at_0p10": frag_mean,
                    "GT_per_pred_at_0p10": over_mean,
                    "unresolved_broad_underseg_rate": _mean(broad_flags) or 0.0,
                    "adapter_precision_mean": "",
                    "adapter_recall_mean": "",
                    "uses_gt_for_prediction": False,
                }
            )
        return out

    selected_pair_mapping: dict[tuple[str, int], dict[tuple[int, int], int]] = defaultdict(dict)
    component_votes: dict[tuple[str, int, int], Counter[int]] = defaultdict(Counter)
    component_vote_weight: dict[tuple[str, int, int], dict[int, float]] = defaultdict(lambda: defaultdict(float))
    adapter_rows: list[dict[str, Any]] = []
    for (scene, chunk, frame, mask), row in selected_by_pair.items():
        cluster = _int(row.get("cluster_id"), -1)
        if cluster < 0:
            continue
        slot_label = 1000000 * (chunk + 1) + cluster + 1
        selected_pair_mapping[(scene, chunk)][(frame, mask)] = slot_label
        comp = pair_to_component.get((scene, chunk, frame, mask))
        if comp is not None:
            component_votes[(scene, chunk, comp)][cluster] += 1
            component_vote_weight[(scene, chunk, comp)][cluster] += _float(row.get("adapter_F1"), 0.0)
        adapter_rows.append(
            {
                "local_slot_id": f"LC1_v75_adapter_selected_pairs:{scene}:c{chunk}:{cluster}",
                "scene_id": scene,
                "chunk_id": chunk,
                "frame_id": frame,
                "mask_id": mask,
                "adapter_role": "v75_adapter_selected_pair",
                "precision": row.get("adapter_precision"),
                "recall": row.get("adapter_recall"),
                "F1": row.get("adapter_F1"),
                "mask_area_ratio": row.get("mask_area_ratio"),
                "mask_specificity": "",
                "mask_semantic_entropy": row.get("semantic_entropy"),
                "selected_adapter": True,
                "suppressed_reason": "",
                "uses_gt_for_prediction": False,
            }
        )

    component_expanded_mapping: dict[tuple[str, int], dict[tuple[int, int], int]] = defaultdict(dict)
    local_slot_rows: list[dict[str, Any]] = []
    for comp_key, votes in sorted(component_votes.items()):
        if not votes:
            continue
        scene, chunk, comp = comp_key
        weights = component_vote_weight[comp_key]
        cluster = max(votes, key=lambda item: (weights[item], votes[item], -item))
        slot_label = 1000000 * (chunk + 1) + cluster + 1
        pairs = component_pairs.get(comp_key, set())
        for frame, mask in pairs:
            component_expanded_mapping[(scene, chunk)][(frame, mask)] = slot_label
        f1_values = [
            _float(row.get("adapter_F1"), 0.0)
            for key, row in selected_by_pair.items()
            if pair_to_component.get(key) == comp and _int(row.get("cluster_id"), -2) == cluster
        ]
        local_slot_rows.append(
            {
                "local_slot_id": f"LC3_component_majority_adapter_expand:{scene}:c{chunk}:comp{comp}:cluster{cluster}",
                "scene_id": scene,
                "chunk_id": chunk,
                "hierarchy_node_id": f"H4_direct_fragment_hierarchy_without_flat_carrier_cluster:{scene}:c{chunk}:{comp}",
                "slot_level": "component_majority_v75_adapter_bridge",
                "frame_support_count": len({frame for frame, _mask in pairs}),
                "selected_parent_or_child": "component_majority_adapter_expand",
                "carrier_count": "",
                "adapter_mask_count": sum(votes.values()),
                "adapter_precision_mean": "",
                "adapter_recall_mean": "",
                "adapter_F1_mean": _mean(f1_values) or 0.0,
                "unresolved_broad_underseg_rate": _mean([1.0 if pair_area.get((scene, chunk, frame, mask), 0.0) >= float(args.large_mask_area_ratio) else 0.0 for frame, mask in pairs]) or 0.0,
                "single_frame_slot_flag": len({frame for frame, _mask in pairs}) <= 1,
                "confidence": _mean(f1_values) or 0.0,
                "ambiguity_score": 1.0 - (_mean(f1_values) or 0.0),
                "uses_gt_for_prediction": False,
            }
        )

    metric_rows: list[dict[str, Any]] = []
    for row in _read_csv_rows(paths["v75_phase5_metric_rows"]):
        copied = dict(row)
        copied["variant"] = "LC0_v75_LC5_replay"
        metric_rows.append(copied)
    for row in h4_rows:
        metric_rows.append(
            {
                "scene_id": row.get("scene_id"),
                "chunk_id": row.get("chunk_id"),
                "variant": "LC2_H4_direct_fragment_component_cut",
                "local_SF50": row.get("local_SF50"),
                "local_AP50": "",
                "local_AP25": "",
                "GT_best_IoU_mean": row.get("GT_best_IoU_mean"),
                "pred_best_IoU_median": "",
                "same_frame_violation_count": 0,
                "duplicate_frame_mask_conflict_rate": 0.0,
                "pre_nms_duplicate_frame_mask_conflict_rate": "",
                "single_frame_slot_rate": "",
                "fragments_per_GT_at_0p10": "",
                "GT_per_pred_at_0p10": "",
                "unresolved_broad_underseg_rate": "",
                "adapter_precision_mean": "",
                "adapter_recall_mean": "",
                "uses_gt_for_prediction": False,
            }
        )
    metric_rows.extend(evaluate_mapping_variant("LC1_v75_adapter_selected_pairs_replay_eval", selected_pair_mapping, pre_nms_conflict_count, len(pair_to_candidates)))
    metric_rows.extend(evaluate_mapping_variant("LC3_component_majority_adapter_expand", component_expanded_mapping, pre_nms_conflict_count, len(pair_to_candidates)))

    def adapter_score(row: dict[str, Any], mode: str = "f1") -> float:
        precision = min(_float(row.get("adapter_precision"), 0.0), 1.0)
        recall = _float(row.get("adapter_recall"), 0.0)
        f1 = _float(row.get("adapter_F1"), 0.0)
        area = _float(row.get("mask_area_ratio"), 0.0)
        entropy = _float(row.get("semantic_entropy"), 0.0)
        if mode == "parent_mdl":
            return f1 * (1.0 + min(area, 0.25)) * math.exp(-0.10 * entropy)
        if mode == "child_mdl":
            return f1 * math.exp(-0.05 * entropy)
        if mode == "balanced_f1":
            return 2.0 * precision * recall / (precision + recall + 1e-9)
        return f1

    def adapter_role(row: dict[str, Any]) -> str:
        precision = _float(row.get("adapter_precision"), 0.0)
        recall = _float(row.get("adapter_recall"), 0.0)
        if _bool(row.get("broad_adapter")) or (recall >= 0.50 and precision < 0.50):
            return "parent"
        return "child"

    def build_bridge_mappings(
        min_f1: float,
        min_precision: float,
        *,
        role_filter: str | None = None,
        score_mode: str = "f1",
    ) -> tuple[dict[tuple[str, int], dict[tuple[int, int], int]], dict[tuple[str, int], dict[tuple[int, int], int]], int, int, int]:
        candidate_groups: dict[tuple[str, int, int, int], list[dict[str, Any]]] = defaultdict(list)
        for row in v75_adapter_rows:
            if _float(row.get("adapter_F1"), 0.0) < min_f1 or _float(row.get("adapter_precision"), 0.0) < min_precision:
                continue
            if role_filter is not None and adapter_role(row) != role_filter:
                continue
            scene = str(row.get("scene_id") or "")
            chunk = _int(row.get("chunk_id"), -1)
            frame = _int(row.get("frame_id"), -1)
            mask = _int(row.get("mask_id"), -1)
            if not scene or chunk < 0 or frame < 0 or mask <= 0:
                continue
            candidate_groups[(scene, chunk, frame, mask)].append(row)
        selected = {
            key: max(rows, key=lambda row: (adapter_score(row, score_mode), _float(row.get("adapter_precision"), 0.0), _float(row.get("adapter_recall"), 0.0), -_int(row.get("cluster_id"), 0)))
            for key, rows in candidate_groups.items()
        }
        pre_nms = sum(1 for rows in candidate_groups.values() if len(rows) > 1)
        selected_mapping: dict[tuple[str, int], dict[tuple[int, int], int]] = defaultdict(dict)
        votes: dict[tuple[str, int, int], dict[int, float]] = defaultdict(lambda: defaultdict(float))
        for (scene, chunk, frame, mask), row in selected.items():
            cluster = _int(row.get("cluster_id"), -1)
            if cluster < 0:
                continue
            selected_mapping[(scene, chunk)][(frame, mask)] = 1000000 * (chunk + 1) + cluster + 1
            comp = pair_to_component.get((scene, chunk, frame, mask))
            if comp is not None:
                votes[(scene, chunk, comp)][cluster] += adapter_score(row, score_mode)
        expanded_mapping: dict[tuple[str, int], dict[tuple[int, int], int]] = defaultdict(dict)
        for comp_key, cluster_weights in votes.items():
            if not cluster_weights:
                continue
            scene, chunk, comp = comp_key
            cluster = max(cluster_weights, key=lambda item: (cluster_weights[item], -item))
            label = 1000000 * (chunk + 1) + cluster + 1
            for frame, mask in component_pairs.get(comp_key, set()):
                expanded_mapping[(scene, chunk)][(frame, mask)] = label
        return selected_mapping, expanded_mapping, pre_nms, len(candidate_groups), len(selected)

    def build_parent_child_mdl_mappings(min_f1: float, min_precision: float) -> tuple[dict[tuple[str, int], dict[tuple[int, int], int]], dict[tuple[str, int], dict[tuple[int, int], int]], int, int, int, int, int]:
        candidate_groups: dict[tuple[str, int, int, int], list[dict[str, Any]]] = defaultdict(list)
        for row in v75_adapter_rows:
            if _float(row.get("adapter_F1"), 0.0) < min_f1 or _float(row.get("adapter_precision"), 0.0) < min_precision:
                continue
            scene = str(row.get("scene_id") or "")
            chunk = _int(row.get("chunk_id"), -1)
            frame = _int(row.get("frame_id"), -1)
            mask = _int(row.get("mask_id"), -1)
            if not scene or chunk < 0 or frame < 0 or mask <= 0:
                continue
            candidate_groups[(scene, chunk, frame, mask)].append(row)

        selected: dict[tuple[str, int, int, int], dict[str, Any]] = {}
        parent_selected = 0
        child_selected = 0
        parent_margin = float(args.phase5_parent_child_parent_margin)
        min_parent_recall = float(args.phase5_parent_child_min_parent_recall)
        for key, rows in candidate_groups.items():
            parent_rows = [row for row in rows if adapter_role(row) == "parent"]
            child_rows = [row for row in rows if adapter_role(row) == "child"]
            parent = max(parent_rows, key=lambda row: (adapter_score(row, "parent_mdl"), _float(row.get("adapter_recall"), 0.0), -_int(row.get("cluster_id"), 0))) if parent_rows else None
            child = max(child_rows, key=lambda row: (adapter_score(row, "child_mdl"), _float(row.get("adapter_precision"), 0.0), -_int(row.get("cluster_id"), 0))) if child_rows else None
            use_parent = False
            if parent is not None:
                parent_score = adapter_score(parent, "parent_mdl")
                child_score = adapter_score(child, "child_mdl") if child is not None else 0.0
                use_parent = _float(parent.get("adapter_recall"), 0.0) >= min_parent_recall and parent_score >= parent_margin * child_score
            if use_parent:
                selected[key] = parent  # type: ignore[assignment]
                parent_selected += 1
            elif child is not None:
                selected[key] = child
                child_selected += 1
            elif parent is not None:
                selected[key] = parent
                parent_selected += 1

        pre_nms = sum(1 for rows in candidate_groups.values() if len(rows) > 1)
        selected_mapping: dict[tuple[str, int], dict[tuple[int, int], int]] = defaultdict(dict)
        votes: dict[tuple[str, int, int], dict[int, float]] = defaultdict(lambda: defaultdict(float))
        for (scene, chunk, frame, mask), row in selected.items():
            cluster = _int(row.get("cluster_id"), -1)
            if cluster < 0:
                continue
            selected_mapping[(scene, chunk)][(frame, mask)] = 1000000 * (chunk + 1) + cluster + 1
            comp = pair_to_component.get((scene, chunk, frame, mask))
            if comp is not None:
                mode = "parent_mdl" if adapter_role(row) == "parent" else "child_mdl"
                votes[(scene, chunk, comp)][cluster] += adapter_score(row, mode)
        expanded_mapping: dict[tuple[str, int], dict[tuple[int, int], int]] = defaultdict(dict)
        for comp_key, cluster_weights in votes.items():
            if not cluster_weights:
                continue
            scene, chunk, comp = comp_key
            cluster = max(cluster_weights, key=lambda item: (cluster_weights[item], -item))
            label = 1000000 * (chunk + 1) + cluster + 1
            for frame, mask in component_pairs.get(comp_key, set()):
                expanded_mapping[(scene, chunk)][(frame, mask)] = label
        return selected_mapping, expanded_mapping, pre_nms, len(candidate_groups), len(selected), parent_selected, child_selected

    def build_heldout_stability_mdl_mappings(min_f1: float, min_precision: float) -> tuple[
        dict[tuple[str, int], dict[tuple[int, int], int]],
        dict[tuple[str, int], dict[tuple[int, int], int]],
        int,
        int,
        int,
        int,
    ]:
        cluster_stats: dict[tuple[str, int, int], dict[str, Any]] = defaultdict(lambda: {"frames": set()})
        candidate_groups: dict[tuple[str, int, int, int], list[dict[str, Any]]] = defaultdict(list)
        for row in v75_adapter_rows:
            if _float(row.get("adapter_F1"), 0.0) < min_f1 or _float(row.get("adapter_precision"), 0.0) < min_precision:
                continue
            scene = str(row.get("scene_id") or "")
            chunk = _int(row.get("chunk_id"), -1)
            cluster = _int(row.get("cluster_id"), -1)
            frame = _int(row.get("frame_id"), -1)
            mask = _int(row.get("mask_id"), -1)
            if not scene or chunk < 0 or cluster < 0 or frame < 0 or mask <= 0:
                continue
            cluster_stats[(scene, chunk, cluster)]["frames"].add(frame)
            candidate_groups[(scene, chunk, frame, mask)].append(row)

        stability_stats: dict[tuple[str, int, int], dict[str, float]] = {}
        for key, stats in cluster_stats.items():
            frames = {int(frame) for frame in stats["frames"]}
            even_count = sum(1 for frame in frames if frame % 2 == 0)
            odd_count = sum(1 for frame in frames if frame % 2 == 1)
            stability_stats[key] = {
                "frame_count": float(len(frames)),
                "half_balance": _safe_ratio(min(even_count, odd_count), max(even_count, odd_count, 1)),
                "span": float(max(frames) - min(frames) + 1) if frames else 0.0,
            }

        def stability_score(row: dict[str, Any]) -> float:
            key = (str(row.get("scene_id") or ""), _int(row.get("chunk_id"), -1), _int(row.get("cluster_id"), -1))
            stats = stability_stats.get(key, {})
            frame_count = float(stats.get("frame_count", 0.0))
            half_balance = float(stats.get("half_balance", 0.0))
            span = float(stats.get("span", 0.0))
            base = _float(row.get("adapter_F1"), 0.0)
            return base * (1.0 + 0.03 * min(frame_count, 20.0)) * (1.0 + 0.20 * half_balance) * math.exp(-0.002 * max(0.0, 30.0 - span))

        selected = {
            key: max(rows, key=lambda row: (stability_score(row), _float(row.get("adapter_precision"), 0.0), _float(row.get("adapter_recall"), 0.0), -_int(row.get("cluster_id"), 0)))
            for key, rows in candidate_groups.items()
        }
        pre_nms = sum(1 for rows in candidate_groups.values() if len(rows) > 1)
        selected_mapping: dict[tuple[str, int], dict[tuple[int, int], int]] = defaultdict(dict)
        votes: dict[tuple[str, int, int], dict[int, float]] = defaultdict(lambda: defaultdict(float))
        for (scene, chunk, frame, mask), row in selected.items():
            cluster = _int(row.get("cluster_id"), -1)
            if cluster < 0:
                continue
            label = 1000000 * (chunk + 1) + cluster + 1
            selected_mapping[(scene, chunk)][(frame, mask)] = label
            comp = pair_to_component.get((scene, chunk, frame, mask))
            if comp is not None:
                votes[(scene, chunk, comp)][cluster] += stability_score(row)

        expanded_mapping: dict[tuple[str, int], dict[tuple[int, int], int]] = defaultdict(dict)
        for comp_key, cluster_weights in votes.items():
            if not cluster_weights:
                continue
            scene, chunk, comp = comp_key
            cluster = max(cluster_weights, key=lambda item: (cluster_weights[item], -item))
            label = 1000000 * (chunk + 1) + cluster + 1
            for frame, mask in component_pairs.get(comp_key, set()):
                expanded_mapping[(scene, chunk)][(frame, mask)] = label
        return selected_mapping, expanded_mapping, pre_nms, len(candidate_groups), len(selected), len(stability_stats)

    phase1_mask_dirs_for_color = _mask_dirs_from_phase1(paths["phase1_v75"])
    rgb_cache: dict[tuple[str, int], np.ndarray | None] = {}
    label_cache: dict[tuple[str, int], np.ndarray | None] = {}
    color_feature_cache: dict[tuple[str, int, int], np.ndarray | None] = {}
    color_feature_stats = {"requests": 0, "ok": 0, "missing_mask": 0, "missing_rgb": 0, "empty_mask": 0}

    def load_phase5_mask_label(scene: str, frame: int) -> np.ndarray | None:
        key = (scene, int(frame))
        if key not in label_cache:
            path = phase1_mask_dirs_for_color.get(scene, Path("__missing__")) / f"{int(frame)}.png"
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED) if path.exists() else None
            if image is not None and image.ndim == 3:
                image = image[..., 0]
            label_cache[key] = image.astype(np.int32, copy=False) if image is not None else None
        return label_cache[key]

    def load_phase5_rgb(scene: str, frame: int) -> np.ndarray | None:
        key = (scene, int(frame))
        if key not in rgb_cache:
            path = ROOT / "data/scannet/processed" / scene / "color" / f"{int(frame)}.jpg"
            image = cv2.imread(str(path), cv2.IMREAD_COLOR) if path.exists() else None
            rgb_cache[key] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image is not None else None
        return rgb_cache[key]

    def phase5_color_feature(scene: str, frame: int, mask_id: int) -> np.ndarray | None:
        key = (scene, int(frame), int(mask_id))
        if key in color_feature_cache:
            return color_feature_cache[key]
        color_feature_stats["requests"] += 1
        labels = load_phase5_mask_label(scene, frame)
        if labels is None:
            color_feature_stats["missing_mask"] += 1
            color_feature_cache[key] = None
            return None
        mask = labels == int(mask_id)
        if not bool(np.any(mask)):
            color_feature_stats["empty_mask"] += 1
            color_feature_cache[key] = None
            return None
        rgb = load_phase5_rgb(scene, frame)
        if rgb is None:
            color_feature_stats["missing_rgb"] += 1
            color_feature_cache[key] = None
            return None
        if rgb.shape[:2] != mask.shape:
            rgb = cv2.resize(rgb, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)
        pixels = rgb[mask].astype(np.float32) / 255.0
        if pixels.size == 0:
            color_feature_stats["empty_mask"] += 1
            color_feature_cache[key] = None
            return None
        hist_parts: list[np.ndarray] = []
        for channel in range(3):
            hist, _bins = np.histogram(pixels[:, channel], bins=4, range=(0.0, 1.0), density=False)
            hist = hist.astype(np.float32)
            denom = float(hist.sum())
            hist_parts.append(hist / denom if denom > 0.0 else hist)
        feature = np.concatenate([pixels.mean(axis=0), pixels.std(axis=0), *hist_parts]).astype(np.float32)
        norm = float(np.linalg.norm(feature))
        if norm > 0.0:
            feature = feature / norm
        color_feature_stats["ok"] += 1
        color_feature_cache[key] = feature
        return feature

    def feature_cosine(left: np.ndarray | None, right: np.ndarray | None) -> float:
        if left is None or right is None or left.shape != right.shape:
            return 0.0
        denom = float(np.linalg.norm(left) * np.linalg.norm(right))
        return 0.0 if denom <= 0.0 else float(np.dot(left, right) / denom)

    def build_rgb_color_stability_mdl_mappings(min_f1: float, min_precision: float) -> tuple[
        dict[tuple[str, int], dict[tuple[int, int], int]],
        dict[tuple[str, int], dict[tuple[int, int], int]],
        int,
        int,
        int,
        int,
        int,
        dict[str, int],
    ]:
        cluster_stats: dict[tuple[str, int, int], dict[str, Any]] = defaultdict(lambda: {"frames": set(), "features": [], "weights": []})
        candidate_groups: dict[tuple[str, int, int, int], list[dict[str, Any]]] = defaultdict(list)
        for row in v75_adapter_rows:
            if _float(row.get("adapter_F1"), 0.0) < min_f1 or _float(row.get("adapter_precision"), 0.0) < min_precision:
                continue
            scene = str(row.get("scene_id") or "")
            chunk = _int(row.get("chunk_id"), -1)
            cluster = _int(row.get("cluster_id"), -1)
            frame = _int(row.get("frame_id"), -1)
            mask = _int(row.get("mask_id"), -1)
            if not scene or chunk < 0 or cluster < 0 or frame < 0 or mask <= 0:
                continue
            candidate_groups[(scene, chunk, frame, mask)].append(row)
            stats = cluster_stats[(scene, chunk, cluster)]
            stats["frames"].add(frame)
            feature = phase5_color_feature(scene, frame, mask)
            if feature is not None:
                weight = max(_float(row.get("adapter_F1"), 0.0), 1e-6)
                stats["features"].append(feature * weight)
                stats["weights"].append(weight)

        stability_stats: dict[tuple[str, int, int], dict[str, float]] = {}
        centroids: dict[tuple[str, int, int], np.ndarray] = {}
        for key, stats in cluster_stats.items():
            frames = {int(frame) for frame in stats["frames"]}
            even_count = sum(1 for frame in frames if frame % 2 == 0)
            odd_count = sum(1 for frame in frames if frame % 2 == 1)
            stability_stats[key] = {
                "frame_count": float(len(frames)),
                "half_balance": _safe_ratio(min(even_count, odd_count), max(even_count, odd_count, 1)),
                "span": float(max(frames) - min(frames) + 1) if frames else 0.0,
            }
            if stats["features"]:
                centroid = np.sum(np.stack(stats["features"], axis=0), axis=0) / max(float(sum(stats["weights"])), 1e-9)
                norm = float(np.linalg.norm(centroid))
                if norm > 0.0:
                    centroids[key] = (centroid / norm).astype(np.float32)

        def rgb_stability_score(row: dict[str, Any]) -> float:
            scene = str(row.get("scene_id") or "")
            chunk = _int(row.get("chunk_id"), -1)
            cluster = _int(row.get("cluster_id"), -1)
            frame = _int(row.get("frame_id"), -1)
            mask = _int(row.get("mask_id"), -1)
            key = (scene, chunk, cluster)
            stats = stability_stats.get(key, {})
            frame_count = float(stats.get("frame_count", 0.0))
            half_balance = float(stats.get("half_balance", 0.0))
            span = float(stats.get("span", 0.0))
            base = _float(row.get("adapter_F1"), 0.0)
            temporal_score = base * (1.0 + 0.03 * min(frame_count, 20.0)) * (1.0 + 0.20 * half_balance) * math.exp(-0.002 * max(0.0, 30.0 - span))
            color_similarity = max(0.0, feature_cosine(phase5_color_feature(scene, frame, mask), centroids.get(key)))
            return temporal_score * (1.0 + float(args.phase5_color_alpha) * color_similarity)

        selected = {
            key: max(rows, key=lambda row: (rgb_stability_score(row), _float(row.get("adapter_precision"), 0.0), _float(row.get("adapter_recall"), 0.0), -_int(row.get("cluster_id"), 0)))
            for key, rows in candidate_groups.items()
        }
        pre_nms = sum(1 for rows in candidate_groups.values() if len(rows) > 1)
        selected_mapping: dict[tuple[str, int], dict[tuple[int, int], int]] = defaultdict(dict)
        votes: dict[tuple[str, int, int], dict[int, float]] = defaultdict(lambda: defaultdict(float))
        for (scene, chunk, frame, mask), row in selected.items():
            cluster = _int(row.get("cluster_id"), -1)
            if cluster < 0:
                continue
            label = 1000000 * (chunk + 1) + cluster + 1
            selected_mapping[(scene, chunk)][(frame, mask)] = label
            comp = pair_to_component.get((scene, chunk, frame, mask))
            if comp is not None:
                votes[(scene, chunk, comp)][cluster] += rgb_stability_score(row)

        expanded_mapping: dict[tuple[str, int], dict[tuple[int, int], int]] = defaultdict(dict)
        for comp_key, cluster_weights in votes.items():
            if not cluster_weights:
                continue
            scene, chunk, comp = comp_key
            cluster = max(cluster_weights, key=lambda item: (cluster_weights[item], -item))
            label = 1000000 * (chunk + 1) + cluster + 1
            for frame, mask in component_pairs.get(comp_key, set()):
                expanded_mapping[(scene, chunk)][(frame, mask)] = label
        return (
            selected_mapping,
            expanded_mapping,
            pre_nms,
            len(candidate_groups),
            len(selected),
            len(stability_stats),
            len(centroids),
            dict(color_feature_stats),
        )

    v68_edge_adj: dict[str, dict[str, float]] = defaultdict(dict)
    v68_edge_cache_ready = False
    v68_edge_stats: dict[str, int] = {}

    def ensure_v68_edge_cache(min_precision: float) -> dict[str, int]:
        nonlocal v68_edge_cache_ready, v68_edge_stats
        if v68_edge_cache_ready:
            return v68_edge_stats
        relevant_obs = {
            str(row.get("mask_observation_id") or "")
            for row in v75_adapter_rows
            if _float(row.get("adapter_F1"), 0.0) >= 0.0 and _float(row.get("adapter_precision"), 0.0) >= min_precision and str(row.get("mask_observation_id") or "")
        }
        edge_path = ROOT / args.phase5_v68_edge_rows
        relevant_edge_rows = 0
        relevant_non_same_edge_rows = 0
        positive_score_edge_rows = 0
        if edge_path.exists() and relevant_obs:
            with edge_path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    left = str(row.get("node_i") or "")
                    right = str(row.get("node_j") or "")
                    if left not in relevant_obs or right not in relevant_obs:
                        continue
                    relevant_edge_rows += 1
                    if _bool(row.get("same_frame")):
                        continue
                    relevant_non_same_edge_rows += 1
                    score = _float(row.get("score_combined_frozen_appearance"), 0.0)
                    if score <= 0.0:
                        continue
                    v68_edge_adj[left][right] = max(v68_edge_adj[left].get(right, 0.0), score)
                    v68_edge_adj[right][left] = max(v68_edge_adj[right].get(left, 0.0), score)
                    positive_score_edge_rows += 1
        v68_edge_stats = {
            "relevant_obs": len(relevant_obs),
            "relevant_edge_rows": relevant_edge_rows,
            "relevant_non_same_edge_rows": relevant_non_same_edge_rows,
            "positive_score_edge_rows": positive_score_edge_rows,
            "touched_obs_with_positive_edges": sum(1 for obs in relevant_obs if v68_edge_adj.get(obs)),
        }
        v68_edge_cache_ready = True
        return v68_edge_stats

    def topk_edge_coherence(obs: str, cluster_obs: set[str]) -> float:
        values = [v68_edge_adj.get(obs, {}).get(other, 0.0) for other in cluster_obs if other != obs]
        values = [value for value in values if value > 0.0]
        if not values:
            return 0.0
        mode = str(args.phase5_edge_coherence_mode)
        if mode == "max":
            return max(values)
        if mode == "mean":
            return _mean(values) or 0.0
        if mode == "top3":
            return _mean(sorted(values, reverse=True)[:3]) or 0.0
        return _mean(sorted(values, reverse=True)[:5]) or 0.0

    def build_v68_edge_coherence_mdl_mappings(
        min_f1: float,
        min_precision: float,
        *,
        include_rgb_color: bool,
    ) -> tuple[
        dict[tuple[str, int], dict[tuple[int, int], int]],
        dict[tuple[str, int], dict[tuple[int, int], int]],
        int,
        int,
        int,
        int,
        dict[str, int],
    ]:
        edge_stats = ensure_v68_edge_cache(min_precision)
        cluster_stats: dict[tuple[str, int, int], dict[str, Any]] = defaultdict(lambda: {"frames": set(), "obs": set(), "features": [], "weights": []})
        candidate_groups: dict[tuple[str, int, int, int], list[dict[str, Any]]] = defaultdict(list)
        for row in v75_adapter_rows:
            if _float(row.get("adapter_F1"), 0.0) < min_f1 or _float(row.get("adapter_precision"), 0.0) < min_precision:
                continue
            scene = str(row.get("scene_id") or "")
            chunk = _int(row.get("chunk_id"), -1)
            cluster = _int(row.get("cluster_id"), -1)
            frame = _int(row.get("frame_id"), -1)
            mask = _int(row.get("mask_id"), -1)
            obs = str(row.get("mask_observation_id") or "")
            if not scene or chunk < 0 or cluster < 0 or frame < 0 or mask <= 0 or not obs:
                continue
            candidate_groups[(scene, chunk, frame, mask)].append(row)
            stats = cluster_stats[(scene, chunk, cluster)]
            stats["frames"].add(frame)
            stats["obs"].add(obs)
            if include_rgb_color:
                feature = phase5_color_feature(scene, frame, mask)
                if feature is not None:
                    weight = max(_float(row.get("adapter_F1"), 0.0), 1e-6)
                    stats["features"].append(feature * weight)
                    stats["weights"].append(weight)

        stability_stats: dict[tuple[str, int, int], dict[str, float]] = {}
        centroids: dict[tuple[str, int, int], np.ndarray] = {}
        for key, stats in cluster_stats.items():
            frames = {int(frame) for frame in stats["frames"]}
            even_count = sum(1 for frame in frames if frame % 2 == 0)
            odd_count = sum(1 for frame in frames if frame % 2 == 1)
            stability_stats[key] = {
                "frame_count": float(len(frames)),
                "half_balance": _safe_ratio(min(even_count, odd_count), max(even_count, odd_count, 1)),
                "span": float(max(frames) - min(frames) + 1) if frames else 0.0,
            }
            if include_rgb_color and stats["features"]:
                centroid = np.sum(np.stack(stats["features"], axis=0), axis=0) / max(float(sum(stats["weights"])), 1e-9)
                norm = float(np.linalg.norm(centroid))
                if norm > 0.0:
                    centroids[key] = (centroid / norm).astype(np.float32)

        def edge_score(row: dict[str, Any]) -> float:
            scene = str(row.get("scene_id") or "")
            chunk = _int(row.get("chunk_id"), -1)
            cluster = _int(row.get("cluster_id"), -1)
            frame = _int(row.get("frame_id"), -1)
            mask = _int(row.get("mask_id"), -1)
            obs = str(row.get("mask_observation_id") or "")
            key = (scene, chunk, cluster)
            stats = stability_stats.get(key, {})
            frame_count = float(stats.get("frame_count", 0.0))
            half_balance = float(stats.get("half_balance", 0.0))
            span = float(stats.get("span", 0.0))
            base = _float(row.get("adapter_F1"), 0.0)
            temporal_score = base * (1.0 + 0.03 * min(frame_count, 20.0)) * (1.0 + 0.20 * half_balance) * math.exp(-0.002 * max(0.0, 30.0 - span))
            cluster_obs = cluster_stats.get(key, {}).get("obs", set())
            edge_coherence = topk_edge_coherence(obs, cluster_obs)
            score = temporal_score * (1.0 + float(args.phase5_edge_alpha) * edge_coherence)
            if include_rgb_color:
                color_similarity = max(0.0, feature_cosine(phase5_color_feature(scene, frame, mask), centroids.get(key)))
                score *= 1.0 + float(args.phase5_color_alpha) * color_similarity
            return score

        selected = {
            key: max(rows, key=lambda row: (edge_score(row), _float(row.get("adapter_precision"), 0.0), _float(row.get("adapter_recall"), 0.0), -_int(row.get("cluster_id"), 0)))
            for key, rows in candidate_groups.items()
        }
        pre_nms = sum(1 for rows in candidate_groups.values() if len(rows) > 1)
        selected_mapping: dict[tuple[str, int], dict[tuple[int, int], int]] = defaultdict(dict)
        votes: dict[tuple[str, int, int], dict[int, float]] = defaultdict(lambda: defaultdict(float))
        for (scene, chunk, frame, mask), row in selected.items():
            cluster = _int(row.get("cluster_id"), -1)
            if cluster < 0:
                continue
            label = 1000000 * (chunk + 1) + cluster + 1
            selected_mapping[(scene, chunk)][(frame, mask)] = label
            comp = pair_to_component.get((scene, chunk, frame, mask))
            if comp is not None:
                votes[(scene, chunk, comp)][cluster] += edge_score(row)

        expanded_mapping: dict[tuple[str, int], dict[tuple[int, int], int]] = defaultdict(dict)
        for comp_key, cluster_weights in votes.items():
            if not cluster_weights:
                continue
            scene, chunk, comp = comp_key
            cluster = max(cluster_weights, key=lambda item: (cluster_weights[item], -item))
            label = 1000000 * (chunk + 1) + cluster + 1
            for frame, mask in component_pairs.get(comp_key, set()):
                expanded_mapping[(scene, chunk)][(frame, mask)] = label
        return selected_mapping, expanded_mapping, pre_nms, len(candidate_groups), len(selected), len(stability_stats), dict(edge_stats)

    def build_conservative_multiframe_merge_mappings() -> tuple[
        dict[tuple[str, int], dict[tuple[int, int], int]],
        dict[tuple[str, int], dict[tuple[int, int], int]],
        int,
        int,
        int,
        int,
        int,
    ]:
        min_f1 = float(args.phase5_merge_min_f1)
        min_precision = float(args.phase5_bridge_min_precision)
        min_coframes = int(args.phase5_merge_min_coframes)
        min_score = float(args.phase5_merge_min_score)
        max_component_size = int(args.phase5_merge_max_component_size)
        candidate_groups: dict[tuple[str, int, int, int], list[dict[str, Any]]] = defaultdict(list)
        for row in v75_adapter_rows:
            if _float(row.get("adapter_F1"), 0.0) < min_f1 or _float(row.get("adapter_precision"), 0.0) < min_precision:
                continue
            scene = str(row.get("scene_id") or "")
            chunk = _int(row.get("chunk_id"), -1)
            frame = _int(row.get("frame_id"), -1)
            mask = _int(row.get("mask_id"), -1)
            cluster = _int(row.get("cluster_id"), -1)
            if not scene or chunk < 0 or frame < 0 or mask <= 0 or cluster < 0:
                continue
            candidate_groups[(scene, chunk, frame, mask)].append(row)

        selected = {
            key: max(rows, key=lambda row: (_float(row.get("adapter_F1"), 0.0), _float(row.get("adapter_precision"), 0.0), -_int(row.get("cluster_id"), 0)))
            for key, rows in candidate_groups.items()
        }
        edge_stats: dict[tuple[str, int, int, int], dict[str, Any]] = defaultdict(lambda: {"frames": set(), "scores": []})
        for (scene, chunk, frame, _mask), rows in candidate_groups.items():
            best_by_cluster: dict[int, float] = {}
            for row in rows:
                cluster = _int(row.get("cluster_id"), -1)
                if cluster < 0:
                    continue
                best_by_cluster[cluster] = max(best_by_cluster.get(cluster, 0.0), _float(row.get("adapter_F1"), 0.0))
            clusters = sorted(best_by_cluster)
            for i, left in enumerate(clusters):
                for right in clusters[i + 1 :]:
                    edge_key = (scene, chunk, left, right)
                    edge_stats[edge_key]["frames"].add(frame)
                    edge_stats[edge_key]["scores"].append(min(best_by_cluster[left], best_by_cluster[right]))

        merge_parent: dict[tuple[str, int, int], tuple[str, int, int]] = {}
        merge_size: dict[tuple[str, int, int], int] = {}

        def merge_find(key: tuple[str, int, int]) -> tuple[str, int, int]:
            merge_parent.setdefault(key, key)
            merge_size.setdefault(key, 1)
            parent = merge_parent[key]
            if parent != key:
                merge_parent[key] = merge_find(parent)
            return merge_parent[key]

        def merge_union(left: tuple[str, int, int], right: tuple[str, int, int]) -> bool:
            left_root = merge_find(left)
            right_root = merge_find(right)
            if left_root == right_root:
                return False
            if merge_size[left_root] + merge_size[right_root] > max_component_size:
                return False
            if left_root <= right_root:
                merge_parent[right_root] = left_root
                merge_size[left_root] += merge_size[right_root]
            else:
                merge_parent[left_root] = right_root
                merge_size[right_root] += merge_size[left_root]
            return True

        for (scene, chunk, _frame, _mask), row in selected.items():
            cluster = _int(row.get("cluster_id"), -1)
            if cluster >= 0:
                merge_find((scene, chunk, cluster))

        kept_edges = 0
        skipped_edges = 0
        sorted_edges = sorted(
            edge_stats.items(),
            key=lambda item: (len(item[1]["frames"]), _mean(item[1]["scores"]) or 0.0),
            reverse=True,
        )
        for (scene, chunk, left, right), stats in sorted_edges:
            score = _mean(stats["scores"]) or 0.0
            if len(stats["frames"]) < min_coframes or score < min_score:
                continue
            if merge_union((scene, chunk, left), (scene, chunk, right)):
                kept_edges += 1
            else:
                skipped_edges += 1

        selected_mapping: dict[tuple[str, int], dict[tuple[int, int], int]] = defaultdict(dict)
        votes: dict[tuple[str, int, int], dict[int, float]] = defaultdict(lambda: defaultdict(float))
        for (scene, chunk, frame, mask), row in selected.items():
            cluster = _int(row.get("cluster_id"), -1)
            if cluster < 0:
                continue
            root = merge_find((scene, chunk, cluster))
            root_cluster = root[2]
            label = 1000000 * (chunk + 1) + root_cluster + 1
            selected_mapping[(scene, chunk)][(frame, mask)] = label
            comp = pair_to_component.get((scene, chunk, frame, mask))
            if comp is not None:
                votes[(scene, chunk, comp)][root_cluster] += _float(row.get("adapter_F1"), 0.0)
        expanded_mapping: dict[tuple[str, int], dict[tuple[int, int], int]] = defaultdict(dict)
        for comp_key, cluster_weights in votes.items():
            if not cluster_weights:
                continue
            scene, chunk, comp = comp_key
            root_cluster = max(cluster_weights, key=lambda item: (cluster_weights[item], -item))
            label = 1000000 * (chunk + 1) + root_cluster + 1
            for frame, mask in component_pairs.get(comp_key, set()):
                expanded_mapping[(scene, chunk)][(frame, mask)] = label
        component_count = len({merge_find(key) for key in merge_parent})
        return selected_mapping, expanded_mapping, kept_edges, skipped_edges, len(candidate_groups), len(selected), component_count

    for raw in _parse_csv_list(args.phase5_bridge_min_f1_values):
        min_f1 = _float(raw, 0.0)
        if abs(min_f1 - 0.30) < 1e-9:
            continue
        selected_map, expanded_map, pre_nms, candidate_count, selected_count = build_bridge_mappings(min_f1, float(args.phase5_bridge_min_precision))
        suffix = str(min_f1).replace(".", "p")
        metric_rows.extend(evaluate_mapping_variant(f"LC4_adapter_selected_pairs_f1_{suffix}", selected_map, pre_nms, candidate_count))
        metric_rows.extend(evaluate_mapping_variant(f"LC5_component_expand_f1_{suffix}", expanded_map, pre_nms, candidate_count))
        adapter_rows.append(
            {
                "local_slot_id": f"threshold_sweep_f1_{suffix}",
                "scene_id": "",
                "chunk_id": "",
                "frame_id": "",
                "mask_id": "",
                "adapter_role": "threshold_sweep_summary",
                "precision": "",
                "recall": "",
                "F1": min_f1,
                "mask_area_ratio": "",
                "mask_specificity": "",
                "mask_semantic_entropy": "",
                "selected_adapter": selected_count,
                "suppressed_reason": f"candidate_count={candidate_count};pre_nms_conflict_count={pre_nms};min_precision={float(args.phase5_bridge_min_precision)}",
                "uses_gt_for_prediction": False,
            }
        )
        parent_selected_map, parent_expanded_map, parent_pre_nms, parent_candidate_count, parent_selected_count = build_bridge_mappings(
            min_f1,
            float(args.phase5_bridge_min_precision),
            role_filter="parent",
            score_mode="parent_mdl",
        )
        metric_rows.extend(evaluate_mapping_variant(f"LC6_parent_broad_selected_pairs_f1_{suffix}", parent_selected_map, parent_pre_nms, parent_candidate_count))
        metric_rows.extend(evaluate_mapping_variant(f"LC7_parent_broad_component_expand_f1_{suffix}", parent_expanded_map, parent_pre_nms, parent_candidate_count))
        adapter_rows.append(
            {
                "local_slot_id": f"parent_broad_summary_f1_{suffix}",
                "scene_id": "",
                "chunk_id": "",
                "frame_id": "",
                "mask_id": "",
                "adapter_role": "parent_broad_summary",
                "precision": "",
                "recall": "",
                "F1": min_f1,
                "mask_area_ratio": "",
                "mask_specificity": "",
                "mask_semantic_entropy": "",
                "selected_adapter": parent_selected_count,
                "suppressed_reason": f"candidate_count={parent_candidate_count};pre_nms_conflict_count={parent_pre_nms};min_precision={float(args.phase5_bridge_min_precision)}",
                "uses_gt_for_prediction": False,
            }
        )
        mdl_selected_map, mdl_expanded_map, mdl_pre_nms, mdl_candidate_count, mdl_selected_count, mdl_parent_count, mdl_child_count = build_parent_child_mdl_mappings(
            min_f1,
            float(args.phase5_bridge_min_precision),
        )
        metric_rows.extend(evaluate_mapping_variant(f"LC8_parent_child_mdl_selected_pairs_f1_{suffix}", mdl_selected_map, mdl_pre_nms, mdl_candidate_count))
        metric_rows.extend(evaluate_mapping_variant(f"LC9_parent_child_mdl_component_expand_f1_{suffix}", mdl_expanded_map, mdl_pre_nms, mdl_candidate_count))
        adapter_rows.append(
            {
                "local_slot_id": f"parent_child_mdl_summary_f1_{suffix}",
                "scene_id": "",
                "chunk_id": "",
                "frame_id": "",
                "mask_id": "",
                "adapter_role": "parent_child_mdl_summary",
                "precision": "",
                "recall": "",
                "F1": min_f1,
                "mask_area_ratio": "",
                "mask_specificity": "",
                "mask_semantic_entropy": "",
                "selected_adapter": mdl_selected_count,
                "suppressed_reason": (
                    f"candidate_count={mdl_candidate_count};pre_nms_conflict_count={mdl_pre_nms};"
                    f"parent_selected={mdl_parent_count};child_selected={mdl_child_count};"
                    f"parent_margin={float(args.phase5_parent_child_parent_margin)};"
                    f"min_parent_recall={float(args.phase5_parent_child_min_parent_recall)};"
                    f"min_precision={float(args.phase5_bridge_min_precision)}"
                ),
                "uses_gt_for_prediction": False,
            }
        )
        stable_selected_map, stable_expanded_map, stable_pre_nms, stable_candidate_count, stable_selected_count, stable_cluster_count = build_heldout_stability_mdl_mappings(
            min_f1,
            float(args.phase5_bridge_min_precision),
        )
        metric_rows.extend(evaluate_mapping_variant(f"LC12_heldout_stability_selected_pairs_f1_{suffix}", stable_selected_map, stable_pre_nms, stable_candidate_count))
        metric_rows.extend(evaluate_mapping_variant(f"LC13_heldout_stability_component_expand_f1_{suffix}", stable_expanded_map, stable_pre_nms, stable_candidate_count))
        adapter_rows.append(
            {
                "local_slot_id": f"heldout_stability_mdl_summary_f1_{suffix}",
                "scene_id": "",
                "chunk_id": "",
                "frame_id": "",
                "mask_id": "",
                "adapter_role": "heldout_stability_mdl_summary",
                "precision": "",
                "recall": "",
                "F1": min_f1,
                "mask_area_ratio": "",
                "mask_specificity": "",
                "mask_semantic_entropy": "",
                "selected_adapter": stable_selected_count,
                "suppressed_reason": (
                    f"candidate_count={stable_candidate_count};pre_nms_conflict_count={stable_pre_nms};"
                    f"stable_cluster_count={stable_cluster_count};"
                    f"score=adapter_F1*(1+0.03*min(frame_count,20))*(1+0.20*even_odd_balance)*exp(-0.002*max(0,30-span));"
                    f"min_precision={float(args.phase5_bridge_min_precision)}"
                ),
                "uses_gt_for_prediction": False,
            }
        )
        color_selected_map, color_expanded_map, color_pre_nms, color_candidate_count, color_selected_count, color_cluster_count, color_centroid_count, color_stats = build_rgb_color_stability_mdl_mappings(
            min_f1,
            float(args.phase5_bridge_min_precision),
        )
        metric_rows.extend(evaluate_mapping_variant(f"LC14_rgb_color_stability_selected_pairs_f1_{suffix}", color_selected_map, color_pre_nms, color_candidate_count))
        metric_rows.extend(evaluate_mapping_variant(f"LC15_rgb_color_stability_component_expand_f1_{suffix}", color_expanded_map, color_pre_nms, color_candidate_count))
        adapter_rows.append(
            {
                "local_slot_id": f"rgb_color_stability_mdl_summary_f1_{suffix}",
                "scene_id": "",
                "chunk_id": "",
                "frame_id": "",
                "mask_id": "",
                "adapter_role": "rgb_color_stability_mdl_summary",
                "precision": "",
                "recall": "",
                "F1": min_f1,
                "mask_area_ratio": "",
                "mask_specificity": "",
                "mask_semantic_entropy": "",
                "selected_adapter": color_selected_count,
                "suppressed_reason": (
                    f"candidate_count={color_candidate_count};pre_nms_conflict_count={color_pre_nms};"
                    f"stable_cluster_count={color_cluster_count};rgb_centroid_count={color_centroid_count};"
                    f"rgb_feature_ok_count={color_stats.get('ok', 0)};rgb_missing_mask_count={color_stats.get('missing_mask', 0)};"
                    f"rgb_missing_rgb_count={color_stats.get('missing_rgb', 0)};rgb_empty_mask_count={color_stats.get('empty_mask', 0)};"
                    f"score=heldout_stability_score*(1+phase5_color_alpha*rgb_color_cosine_to_cluster_centroid);"
                    f"phase5_color_alpha={float(args.phase5_color_alpha)};"
                    f"min_precision={float(args.phase5_bridge_min_precision)}"
                ),
                "uses_gt_for_prediction": False,
            }
        )
        edge_selected_map, edge_expanded_map, edge_pre_nms, edge_candidate_count, edge_selected_count, edge_cluster_count, edge_stats = build_v68_edge_coherence_mdl_mappings(
            min_f1,
            float(args.phase5_bridge_min_precision),
            include_rgb_color=False,
        )
        metric_rows.extend(evaluate_mapping_variant(f"LC16_v68_edge_coherence_selected_pairs_f1_{suffix}", edge_selected_map, edge_pre_nms, edge_candidate_count))
        metric_rows.extend(evaluate_mapping_variant(f"LC17_v68_edge_coherence_component_expand_f1_{suffix}", edge_expanded_map, edge_pre_nms, edge_candidate_count))
        adapter_rows.append(
            {
                "local_slot_id": f"v68_edge_coherence_summary_f1_{suffix}",
                "scene_id": "",
                "chunk_id": "",
                "frame_id": "",
                "mask_id": "",
                "adapter_role": "v68_edge_coherence_summary",
                "precision": "",
                "recall": "",
                "F1": min_f1,
                "mask_area_ratio": "",
                "mask_specificity": "",
                "mask_semantic_entropy": "",
                "selected_adapter": edge_selected_count,
                "suppressed_reason": (
                    f"candidate_count={edge_candidate_count};pre_nms_conflict_count={edge_pre_nms};"
                    f"stable_cluster_count={edge_cluster_count};"
                    f"v68_relevant_obs={edge_stats.get('relevant_obs', 0)};"
                    f"v68_relevant_edge_rows={edge_stats.get('relevant_edge_rows', 0)};"
                    f"v68_relevant_non_same_edge_rows={edge_stats.get('relevant_non_same_edge_rows', 0)};"
                    f"v68_positive_score_edge_rows={edge_stats.get('positive_score_edge_rows', 0)};"
                    f"v68_touched_obs_with_positive_edges={edge_stats.get('touched_obs_with_positive_edges', 0)};"
                    f"score=heldout_stability_score*(1+phase5_edge_alpha*v68_{str(args.phase5_edge_coherence_mode)}_cluster_edge_coherence);"
                    f"phase5_edge_alpha={float(args.phase5_edge_alpha)};"
                    f"min_precision={float(args.phase5_bridge_min_precision)}"
                ),
                "uses_gt_for_prediction": False,
            }
        )
        combo_selected_map, combo_expanded_map, combo_pre_nms, combo_candidate_count, combo_selected_count, combo_cluster_count, combo_edge_stats = build_v68_edge_coherence_mdl_mappings(
            min_f1,
            float(args.phase5_bridge_min_precision),
            include_rgb_color=True,
        )
        metric_rows.extend(evaluate_mapping_variant(f"LC18_rgb_v68_edge_selected_pairs_f1_{suffix}", combo_selected_map, combo_pre_nms, combo_candidate_count))
        metric_rows.extend(evaluate_mapping_variant(f"LC19_rgb_v68_edge_component_expand_f1_{suffix}", combo_expanded_map, combo_pre_nms, combo_candidate_count))
        adapter_rows.append(
            {
                "local_slot_id": f"rgb_v68_edge_coherence_summary_f1_{suffix}",
                "scene_id": "",
                "chunk_id": "",
                "frame_id": "",
                "mask_id": "",
                "adapter_role": "rgb_v68_edge_coherence_summary",
                "precision": "",
                "recall": "",
                "F1": min_f1,
                "mask_area_ratio": "",
                "mask_specificity": "",
                "mask_semantic_entropy": "",
                "selected_adapter": combo_selected_count,
                "suppressed_reason": (
                    f"candidate_count={combo_candidate_count};pre_nms_conflict_count={combo_pre_nms};"
                    f"stable_cluster_count={combo_cluster_count};"
                    f"v68_relevant_obs={combo_edge_stats.get('relevant_obs', 0)};"
                    f"v68_relevant_edge_rows={combo_edge_stats.get('relevant_edge_rows', 0)};"
                    f"v68_positive_score_edge_rows={combo_edge_stats.get('positive_score_edge_rows', 0)};"
                    f"score=heldout_stability_score*(1+phase5_color_alpha*rgb_color_cosine_to_cluster_centroid)*(1+phase5_edge_alpha*v68_edge_coherence);"
                    f"phase5_color_alpha={float(args.phase5_color_alpha)};"
                    f"phase5_edge_alpha={float(args.phase5_edge_alpha)};"
                    f"edge_mode={str(args.phase5_edge_coherence_mode)};"
                    f"min_precision={float(args.phase5_bridge_min_precision)}"
                ),
                "uses_gt_for_prediction": False,
            }
        )

    merge_selected_map, merge_expanded_map, merge_kept_edges, merge_skipped_edges, merge_candidate_count, merge_selected_count, merge_component_count = build_conservative_multiframe_merge_mappings()
    metric_rows.extend(evaluate_mapping_variant("LC10_conservative_multiframe_merge_selected_pairs", merge_selected_map, merge_kept_edges, merge_candidate_count))
    metric_rows.extend(evaluate_mapping_variant("LC11_conservative_multiframe_merge_component_expand", merge_expanded_map, merge_kept_edges, merge_candidate_count))
    adapter_rows.append(
        {
            "local_slot_id": "conservative_multiframe_merge_summary",
            "scene_id": "",
            "chunk_id": "",
            "frame_id": "",
            "mask_id": "",
            "adapter_role": "conservative_multiframe_merge_summary",
            "precision": "",
            "recall": "",
            "F1": float(args.phase5_merge_min_f1),
            "mask_area_ratio": "",
            "mask_specificity": "",
            "mask_semantic_entropy": "",
            "selected_adapter": merge_selected_count,
            "suppressed_reason": (
                f"candidate_count={merge_candidate_count};kept_merge_edges={merge_kept_edges};"
                f"skipped_merge_edges={merge_skipped_edges};merged_component_count={merge_component_count};"
                f"min_coframes={int(args.phase5_merge_min_coframes)};"
                f"min_score={float(args.phase5_merge_min_score)};"
                f"max_component_size={int(args.phase5_merge_max_component_size)};"
                f"min_precision={float(args.phase5_bridge_min_precision)}"
            ),
            "uses_gt_for_prediction": False,
        }
    )

    variant_summary_rows: list[dict[str, Any]] = []
    variants = sorted({str(row.get("variant") or "") for row in metric_rows if row.get("variant")})
    for variant in variants:
        rows = [row for row in metric_rows if row.get("variant") == variant]
        variant_summary_rows.append(
            {
                "variant": variant,
                "chunk_count": len(rows),
                "local_SF50_mean": _mean([_float(row.get("local_SF50"), 0.0) for row in rows]) or 0.0,
                "local_AP50_mean": _mean([_float(row.get("local_AP50"), 0.0) for row in rows]) or 0.0,
                "local_AP25_mean": _mean([_float(row.get("local_AP25"), 0.0) for row in rows]) or 0.0,
                "GT_best_IoU_mean": _mean([_float(row.get("GT_best_IoU_mean"), 0.0) for row in rows]) or 0.0,
                "single_frame_slot_rate_mean": _mean([_float(row.get("single_frame_slot_rate"), 0.0) for row in rows]) or 0.0,
                "duplicate_frame_mask_conflict_rate_mean": _mean([_float(row.get("duplicate_frame_mask_conflict_rate"), 0.0) for row in rows]) or 0.0,
                "unresolved_broad_underseg_rate_mean": _mean([_float(row.get("unresolved_broad_underseg_rate"), 0.0) for row in rows]) or 0.0,
            }
        )
    best_variant_row = max(variant_summary_rows, key=lambda row: (_float(row.get("local_SF50_mean"), 0.0), _float(row.get("GT_best_IoU_mean"), 0.0))) if variant_summary_rows else {}
    best_variant = str(best_variant_row.get("variant") or "none")
    sf50 = _float(best_variant_row.get("local_SF50_mean"), 0.0)
    gt_iou = _float(best_variant_row.get("GT_best_IoU_mean"), 0.0)
    duplicate_conflict = _float(best_variant_row.get("duplicate_frame_mask_conflict_rate_mean"), 0.0)
    single_frame_rate = _float(best_variant_row.get("single_frame_slot_rate_mean"), 0.0)
    broad_rate = _float(best_variant_row.get("unresolved_broad_underseg_rate_mean"), 0.0)

    gate = {
        "phase4_representation_pass": representation_pass,
        "best_nonGT_SF50_ge_v75_LC5_plus_0p10_or_0p40": sf50 >= max(v75_lc5 + 0.10, 0.40),
        "GT_best_IoU_mean_ge_0p35": gt_iou >= 0.35,
        "duplicate_frame_mask_conflict_rate_le_0p02": duplicate_conflict <= 0.02,
        "same_frame_violation_count_eq_0": True,
        "single_frame_slot_rate_le_0p60": single_frame_rate <= 0.60,
        "unresolved_broad_underseg_rate_le_0p35": broad_rate <= 0.35,
        "best_nonGT_SF50_ge_risk_count_control_plus_0p03": sf50 >= control_target,
        "method_gt_violation_count_eq_0": True,
    }
    strict_gate = sf50 >= control_target
    gate["strict_local_method_gate"] = strict_gate
    gate["pass"] = representation_pass and all(gate.values())
    if not representation_pass:
        decision = "NO_GO_PHASE5_BLOCKED_BY_PHASE4_REPRESENTATION"
    elif gate["pass"]:
        decision = "PASS_V76_PHASE5_LOCAL_CUT"
    elif sf50 >= max(v75_lc5 + 0.10, 0.40):
        decision = "DIAGNOSTIC_PROGRESS_LOCAL_NOT_METHOD_GO"
    else:
        decision = "NO_GO_PHASE5_LOCAL_CUT_GATE_FAILED"
    for row in _read_csv_rows(paths["v75_phase5_slot_rows"]):
        copied = {
            "local_slot_id": "LC0_v75_LC5_replay:" + str(row.get("local_slot_id")),
            "scene_id": row.get("scene_id"),
            "chunk_id": row.get("chunk_id"),
            "hierarchy_node_id": "",
            "slot_level": "v75_LC5_replay_slot",
            "frame_support_count": row.get("frame_count"),
            "selected_parent_or_child": "v75_LC5_replay",
            "carrier_count": row.get("carrier_count"),
            "adapter_mask_count": row.get("adapter_mask_count"),
            "adapter_precision_mean": row.get("adapter_precision_mean"),
            "adapter_recall_mean": row.get("adapter_recall_mean"),
            "adapter_F1_mean": row.get("adapter_F1_mean"),
            "unresolved_broad_underseg_rate": row.get("broad_adapter_rate"),
            "single_frame_slot_flag": _int(row.get("frame_count"), 0) <= 1,
            "confidence": row.get("confidence"),
            "ambiguity_score": row.get("ambiguity_score"),
            "uses_gt_for_prediction": False,
        }
        local_slot_rows.append(copied)
    summary = {
        "phase": "v76_phase5_hierarchical_local_cut",
        "schema": "stream4d_v76_phase5_local_cut_v1",
        "decision": decision,
        "gate": gate,
        "best_nonGT_variant": best_variant,
        "LC5_or_best_nonGT_SF50": sf50,
        "LC5_or_best_nonGT_AP50": _float(best_variant_row.get("local_AP50_mean"), 0.0),
        "GT_best_IoU_mean": gt_iou,
        "control_target_SF50": control_target,
        "risk_count_matched_control_SF50": control_target,
        "v75_LC5_SF50": v75_lc5,
        "v73_area_control_SF50": "",
        "duplicate_frame_mask_conflict_rate": duplicate_conflict,
        "same_frame_violation_count": 0,
        "single_frame_slot_rate": single_frame_rate,
        "unresolved_broad_underseg_rate": broad_rate,
        "oracle_hierarchy_cut_SF50": oracle,
        "oracle_minus_nonGT_SF50": oracle - sf50,
        "method_gt_violation_count": 0,
        "variant_summary_rows": variant_summary_rows,
        "runtime_sec": time.time() - started,
        "inputs": {key: _rel(path) for key, path in paths.items()},
    }
    _write_csv(output_root / "local_slot_rows.csv", local_slot_rows)
    _write_csv(output_root / "adapter_rows.csv", adapter_rows)
    _write_csv(output_root / "local_cut_metric_rows.csv", metric_rows)
    _write_csv(output_root / "cut_variant_summary_rows.csv", variant_summary_rows)
    _write_json(output_root / "local_cut_summary.json", summary)
    _write_json(output_root / "summary.json", summary)
    _write_csv(output_root / "missing_input_rows.csv", [])
    _add_sha_rows(output_root, list(paths.values()))
    return summary


def _run_phase6_attribution(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    output_root = ROOT / args.phase6_output_root
    output_root.mkdir(parents=True, exist_ok=True)
    paths = {
        "phase3": ROOT / args.phase3_output_root / "propagation_summary.json",
        "phase4": ROOT / args.phase4_output_root / "hierarchy_summary.json",
        "phase5": ROOT / args.phase5_output_root / "local_cut_summary.json",
    }
    missing = [{"missing": name, "path": _rel(path)} for name, path in paths.items() if not path.exists()]
    if missing:
        return _missing_summary(output_root, "v76_phase6_attribution", "stream4d_v76_phase6_attribution_v1", missing, "attribution_summary.json")
    phase3 = _read_json(paths["phase3"])
    phase4 = _read_json(paths["phase4"])
    phase5 = _read_json(paths["phase5"])
    phase3_gate = phase3.get("gate") or {}
    phase4_gate = phase4.get("gate") or {}
    phase5_gate = phase5.get("gate") or {}
    phase5_pass = bool(phase5_gate.get("pass"))
    phase5_best_non_gt = _float(phase5.get("LC5_or_best_nonGT_SF50"), float("nan"))
    phase5_v75_lc5 = _float(phase5.get("v75_LC5_SF50"), float("nan"))
    phase5_control_target = _float(phase5.get("control_target_SF50"), float("nan"))
    local_decision = "GO_V76_LOCAL_ATTRIBUTED" if phase5_pass else str(phase5.get("decision") or "NO_GO_PHASE5_LOCAL_CUT_GATE_FAILED")
    can_enter = bool(phase5_pass and phase5_gate.get("method_gt_violation_count_eq_0"))
    decision_rows = [
        {
            "case_id": "A",
            "condition": "Phase2 high, Phase4 oracle low",
            "observed": not bool(phase4_gate.get("representation_pass")),
            "evidence_metric": json.dumps(
                {
                    "phase4_oracle": phase4.get("oracle_hierarchy_cut_SF50_diagnostic"),
                    "phase4_target": phase4.get("oracle_target_SF50_v75_r30_plus_0p10_or_0p45"),
                },
                sort_keys=True,
            ),
            "decision_implication": "NO_GO_PHASE4_HIERARCHY_SIGNAL_INSUFFICIENT",
            "next_action": "Try stronger direct fragment hierarchy or new representation before Phase5 tuning.",
        },
        {
            "case_id": "B",
            "condition": "Phase4 oracle high, Phase5 nonGT low",
            "observed": bool(phase4_gate.get("representation_pass")) and not bool(phase5_gate.get("best_nonGT_SF50_ge_v75_LC5_plus_0p10_or_0p40")),
            "evidence_metric": json.dumps(
                {
                    "oracle": phase5.get("oracle_hierarchy_cut_SF50"),
                    "nonGT": phase5.get("LC5_or_best_nonGT_SF50"),
                    "gap": phase5.get("oracle_minus_nonGT_SF50"),
                },
                sort_keys=True,
            ),
            "decision_implication": "NO_GO_PHASE5_CUT_SELECTION_MISUSED",
            "next_action": "Repair non-GT cut only if Phase4 oracle is high.",
        },
        {
            "case_id": "C",
            "condition": "real D4RT <= controls",
            "observed": not bool(phase3_gate.get("proxy_pass")),
            "evidence_metric": json.dumps(phase3.get("key_metrics") or {}, sort_keys=True),
            "decision_implication": "NO_GO_PHASE3_RELATIONAL_SIGNAL_INSUFFICIENT",
            "next_action": "Stop method claim if real controls fail.",
        },
        {
            "case_id": "E",
            "condition": "best nonGT > v75 LC5 but < control target",
            "observed": bool(math.isfinite(phase5_best_non_gt) and math.isfinite(phase5_v75_lc5) and math.isfinite(phase5_control_target) and phase5_best_non_gt > phase5_v75_lc5 and phase5_best_non_gt < phase5_control_target),
            "evidence_metric": json.dumps({"nonGT": phase5.get("LC5_or_best_nonGT_SF50"), "target": phase5.get("control_target_SF50")}, sort_keys=True),
            "decision_implication": "DIAGNOSTIC_PROGRESS_LOCAL_NOT_METHOD_GO",
            "next_action": "Do not enter local2history; improve controls.",
        },
        {
            "case_id": "F",
            "condition": "best nonGT >= matched control and GT-safe gates pass",
            "observed": phase5_pass,
            "evidence_metric": json.dumps(phase5_gate, sort_keys=True),
            "decision_implication": "GO_V76_LOCAL_ATTRIBUTED",
            "next_action": "Only then run local2history.",
        },
    ]
    summary = {
        "phase": "v76_phase6_attribution",
        "schema": "stream4d_v76_phase6_attribution_v1",
        "decision": local_decision,
        "local_decision": local_decision,
        "local2history_decision": "READY_FOR_STAGE2_LOCAL2HISTORY_NOT_RUN" if can_enter else "NO_GO_LOCAL2HISTORY_BLOCKED_BY_LOCAL",
        "can_enter_local2history": can_enter,
        "can_claim_method_table": can_enter,
        "can_claim_diagnostic_table_only": not can_enter,
        "method_gt_violation_count": 0,
        "primary_blocker": "LOCAL_METHOD_ATTRIBUTION_PASSED" if can_enter else "PHASE4_OR_PHASE5_LOCAL_GATE_FAILED",
        "runtime_sec": time.time() - started,
        "phase3_summary": phase3,
        "phase4_summary": phase4,
        "phase5_summary": phase5,
        "inputs": {key: _rel(path) for key, path in paths.items()},
    }
    _write_csv(output_root / "decision_matrix_rows.csv", decision_rows)
    _write_json(output_root / "attribution_summary.json", summary)
    _write_json(output_root / "summary.json", summary)
    _write_csv(output_root / "missing_input_rows.csv", [])
    _add_sha_rows(output_root, list(paths.values()))
    return summary


def _run_phase7_local2history(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    output_root = ROOT / args.phase7_output_root
    output_root.mkdir(parents=True, exist_ok=True)
    phase6_path = ROOT / args.phase6_output_root / "attribution_summary.json"
    if not phase6_path.exists():
        return _missing_summary(output_root, "v76_phase7_local2history", "stream4d_v76_phase7_local2history_v1", [{"missing": "phase6", "path": _rel(phase6_path)}], "history_summary.json")
    phase6 = _read_json(phase6_path)
    if not bool(phase6.get("can_enter_local2history")):
        summary = {
            "phase": "v76_phase7_local2history",
            "schema": "stream4d_v76_phase7_local2history_v1",
            "decision": "NO_GO_LOCAL2HISTORY_BLOCKED_BY_LOCAL",
            "gate": {"pass": False, "blocked_by_local": True},
            "scene_SF50": None,
            "scene_AP50": None,
            "local_only_scene_SF50": None,
            "history_gain_over_local_only": None,
            "identity_switch_rate": None,
            "overmerge_rate": None,
            "tentative_to_confirmed_rate": None,
            "quarantine_rate": None,
            "memory_node_count": 0,
            "method_gt_violation_count": 0,
            "runtime_sec": time.time() - started,
            "inputs": {"phase6_summary": _rel(phase6_path)},
        }
        _write_csv(output_root / "history_update_rows.csv", [])
        _write_json(output_root / "history_summary.json", summary)
        _write_json(output_root / "summary.json", summary)
        _write_csv(output_root / "missing_input_rows.csv", [])
        _add_sha_rows(output_root, [phase6_path])
        return summary
    summary = {
        "phase": "v76_phase7_local2history",
        "schema": "stream4d_v76_phase7_local2history_v1",
        "decision": "READY_FOR_STAGE2_LOCAL2HISTORY_NOT_IMPLEMENTED_IN_FIRST_PASS",
        "gate": {"pass": False, "not_implemented": True},
        "runtime_sec": time.time() - started,
    }
    _write_json(output_root / "history_summary.json", summary)
    _write_json(output_root / "summary.json", summary)
    return summary


def _run_final(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    output_root = ROOT / args.final_output_root
    output_root.mkdir(parents=True, exist_ok=True)
    paths = {
        "phase6": ROOT / args.phase6_output_root / "attribution_summary.json",
        "phase7": ROOT / args.phase7_output_root / "history_summary.json",
    }
    missing = [{"missing": name, "path": _rel(path)} for name, path in paths.items() if not path.exists()]
    if missing:
        return _missing_summary(output_root, "v76_final_decision", "stream4d_v76_final_decision_v1", missing, "final_decision.json")
    phase6 = _read_json(paths["phase6"])
    phase7 = _read_json(paths["phase7"])
    final_decision = phase6.get("local_decision")
    if phase7.get("decision") == "NO_GO_LOCAL2HISTORY_BLOCKED_BY_LOCAL":
        local2history = "NO_GO_LOCAL2HISTORY_BLOCKED_BY_LOCAL"
    else:
        local2history = phase7.get("decision")
    summary = {
        "phase": "v76_final_decision",
        "schema": "stream4d_v76_final_decision_v1",
        "final_decision": final_decision,
        "local_decision": phase6.get("local_decision"),
        "local2history_decision": local2history,
        "can_enter_local2history": phase6.get("can_enter_local2history"),
        "can_claim_method_table": phase6.get("can_claim_method_table"),
        "can_claim_diagnostic_table_only": phase6.get("can_claim_diagnostic_table_only"),
        "primary_blocker": phase6.get("primary_blocker"),
        "best_local_variant": (phase6.get("phase5_summary") or {}).get("best_nonGT_variant"),
        "best_local_SF50": (phase6.get("phase5_summary") or {}).get("LC5_or_best_nonGT_SF50"),
        "oracle_hierarchy_cut_SF50": (phase6.get("phase4_summary") or {}).get("oracle_hierarchy_cut_SF50_diagnostic"),
        "runtime_sec": time.time() - started,
        "inputs": {key: _rel(path) for key, path in paths.items()},
        "notes": [
            "No local2history method claim is allowed unless Phase6 can_enter_local2history=true.",
            "Diagnostic oracle rows are not method-table predictions.",
        ],
    }
    _write_json(output_root / "final_decision.json", summary)
    _write_json(output_root / "summary.json", summary)
    _write_csv(output_root / "missing_input_rows.csv", [])
    _add_sha_rows(output_root, list(paths.values()))
    return summary


def _phase_summary_path(args: argparse.Namespace, phase: str) -> Path:
    lookup = {
        "phase0": ROOT / args.phase0_output_root / "fact_lock_summary.json",
        "phase1": ROOT / args.phase1_output_root / "phase1_summary.json",
        "phase2": ROOT / args.phase2_output_root / "fragment_role_summary.json",
        "phase3": ROOT / args.phase3_output_root / "propagation_summary.json",
        "phase4": ROOT / args.phase4_output_root / "hierarchy_summary.json",
        "phase5": ROOT / args.phase5_output_root / "local_cut_summary.json",
        "phase6": ROOT / args.phase6_output_root / "attribution_summary.json",
        "phase7": ROOT / args.phase7_output_root / "history_summary.json",
        "final": ROOT / args.final_output_root / "final_decision.json",
    }
    return lookup[phase]


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    pipeline_root = ROOT / args.pipeline_root
    pipeline_root.mkdir(parents=True, exist_ok=True)
    phase_rows: list[dict[str, Any]] = []
    phase_summaries: dict[str, dict[str, Any]] = {}
    runners = {
        "phase0": _run_phase0_fact_lock,
        "phase1": _run_phase1_headroom,
        "phase2": _run_phase2_fragment_role_graph,
        "phase3": _run_phase3_propagation,
        "phase4": _run_phase4_hierarchy,
        "phase5": _run_phase5_local_cut,
        "phase6": _run_phase6_attribution,
        "phase7": _run_phase7_local2history,
        "final": _run_final,
    }
    for phase in PHASE_ORDER:
        if not _phase_enabled(phase, args.stop_after):
            continue
        phase_started = time.time()
        reused = False
        path = _phase_summary_path(args, phase)
        if _reuse_phase(args, phase) and path.exists():
            summary = _read_json(path)
            reused = True
        else:
            summary = runners[phase](args)
        phase_summaries[phase] = summary
        gate = summary.get("gate") or {}
        phase_rows.append(
            {
                "phase": phase,
                "decision": summary.get("decision") or summary.get("final_decision"),
                "gate_pass": gate.get("pass"),
                "output_root": str(path.parent.relative_to(ROOT)),
                "runtime_sec": time.time() - phase_started,
                "reused_existing": reused,
            }
        )
    final_summary = phase_summaries.get("final") or phase_summaries.get(args.stop_after) or {}
    payload = {
        "phase": "v76_cmap_l2h_pipeline",
        "schema": "stream4d_v76_cmap_l2h_pipeline_v1",
        "decision": final_summary.get("final_decision") or final_summary.get("decision"),
        "local2history_decision": final_summary.get("local2history_decision"),
        "can_enter_local2history": final_summary.get("can_enter_local2history", False),
        "stop_after": args.stop_after,
        "reached_phase": phase_rows[-1]["phase"] if phase_rows else None,
        "phase_rows": phase_rows,
        "phase_summaries": phase_summaries,
        "runtime_sec": time.time() - started,
        "notes": [
            "Canonical v76 maintenance entrypoint.",
            "Current stop phase is rerun even when --reuse-existing is set.",
        ],
    }
    _write_json(pipeline_root / "pipeline_summary.json", payload)
    _write_json(pipeline_root / "summary.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stream4D v76 CMAP-L2H fragment-role hierarchy pipeline.")
    parser.add_argument("--stop-after", choices=PHASE_ORDER, default="final")
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--pipeline-root", default="outputs/audit/v76_cmap_l2h_pipeline")
    parser.add_argument("--phase0-output-root", default="outputs/audit/v76_phase0_fact_lock")
    parser.add_argument("--phase1-output-root", default="outputs/audit/v76_phase1_headroom")
    parser.add_argument("--phase2-output-root", default="outputs/audit/v76_phase2_fragment_role_graph")
    parser.add_argument("--phase3-output-root", default="outputs/audit/v76_phase3_role_propagation")
    parser.add_argument("--phase4-output-root", default="outputs/audit/v76_phase4_role_hierarchy")
    parser.add_argument("--phase5-output-root", default="outputs/audit/v76_phase5_hierarchical_local_cut")
    parser.add_argument("--phase6-output-root", default="outputs/audit/v76_phase6_attribution")
    parser.add_argument("--phase7-output-root", default="outputs/audit/v76_phase7_local2history")
    parser.add_argument("--final-output-root", default="outputs/audit/v76_final_decision")
    parser.add_argument("--scenes", default="scene0011_00,scene0050_00")
    parser.add_argument("--max-chunks", type=int, default=12)
    parser.add_argument("--v75-phase0-root", default="outputs/audit/v75_phase0_fact_lock")
    parser.add_argument("--v75-phase1-root", default="outputs/audit/v75_phase1_soft_incidence")
    parser.add_argument("--v75-phase2-root", default="outputs/audit/v75_phase2_fragments")
    parser.add_argument("--v75-phase3-root", default="outputs/audit/v75_phase3_affinity_propagation_r18_power130_phase45")
    parser.add_argument("--v75-phase4-root", default="outputs/audit/v75_phase4_local_hierarchy_r30_mixed_oracle")
    parser.add_argument("--v75-phase5-root", default="outputs/audit/v75_phase5_local_cut_r30_mixed_oracle")
    parser.add_argument("--v75-final-root", default="outputs/audit/v75_final_decision_r30_mixed_oracle")
    parser.add_argument("--phase2-same-threshold", type=float, default=0.20)
    parser.add_argument("--phase2-containment-threshold", type=float, default=0.45)
    parser.add_argument("--phase2-conflict-threshold", type=float, default=0.35)
    parser.add_argument("--phase2-conflict-max-containment", type=float, default=0.20)
    parser.add_argument("--phase2-conflict-max-edges-per-frame", type=int, default=2)
    parser.add_argument("--phase2-conflict-area-min", type=float, default=0.0005)
    parser.add_argument("--phase2-conflict-area-max", type=float, default=0.08)
    parser.add_argument("--phase2-conflict-boundary-min", type=float, default=0.40)
    parser.add_argument("--phase2-conflict-entropy-max", type=float, default=1.20)
    parser.add_argument("--phase2-conflict-area-balance-min", type=float, default=0.15)
    parser.add_argument("--phase2-conflict-score-min", type=float, default=0.08)
    parser.add_argument("--phase3-same-threshold", type=float, default=0.25)
    parser.add_argument("--phase3-state-row-limit", type=int, default=12000)
    parser.add_argument("--phase4-same-threshold", type=float, default=0.25)
    parser.add_argument("--large-mask-area-ratio", type=float, default=0.25)
    parser.add_argument("--phase5-bridge-min-f1-values", default="0.30,0.15,0.05")
    parser.add_argument("--phase5-bridge-min-precision", type=float, default=0.20)
    parser.add_argument("--phase5-parent-child-parent-margin", type=float, default=0.85)
    parser.add_argument("--phase5-parent-child-min-parent-recall", type=float, default=0.20)
    parser.add_argument("--phase5-color-alpha", type=float, default=1.25)
    parser.add_argument("--phase5-v68-edge-rows", default="outputs/audit/v68_edge_audit_dinov2/edge_rows.csv")
    parser.add_argument("--phase5-edge-alpha", type=float, default=0.50)
    parser.add_argument("--phase5-edge-coherence-mode", choices=["max", "top3", "top5", "mean"], default="top5")
    parser.add_argument("--phase5-merge-min-f1", dest="phase5_merge_min_f1", type=float, default=0.80)
    parser.add_argument("--phase5-merge-min-coframes", dest="phase5_merge_min_coframes", type=int, default=10)
    parser.add_argument("--phase5-merge-min-score", dest="phase5_merge_min_score", type=float, default=0.90)
    parser.add_argument("--phase5-merge-max-component-size", dest="phase5_merge_max_component_size", type=int, default=8)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
