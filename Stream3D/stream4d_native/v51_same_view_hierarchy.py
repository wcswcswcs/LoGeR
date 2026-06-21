from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from stream4d_native.v47_common import ROOT, utc_now, write_csv, write_json
from stream4d_native.v51_remask_source_discovery import _frame_id


PLAN_PATH = "docs/stream4d_v51_r2_mosaic_remask_lift_codex_plan.md"


def _rel(path: str | Path) -> str:
    path_obj = Path(path)
    if not path_obj.is_absolute():
        path_obj = ROOT / path_obj
    try:
        return str(path_obj.relative_to(ROOT))
    except ValueError:
        return str(path_obj)


def _scene_dirs(root: Path) -> list[Path]:
    if any(root.glob("*_masks.npz")):
        return [root]
    return sorted([path for path in root.iterdir() if path.is_dir()])


def _load_masks(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = np.load(path, allow_pickle=True)
    masks = np.asarray(payload["masks"])
    if masks.dtype != bool:
        masks = masks != 0
    if "areas" in payload.files:
        areas = np.asarray(payload["areas"], dtype=np.int64)
    else:
        areas = masks.reshape(masks.shape[0], -1).sum(axis=1).astype(np.int64)
    return masks, areas


def _proposal_id(scene: str, frame_id: int, index: int) -> str:
    return f"{scene}|frame{int(frame_id):06d}|sam2_filtered|p{int(index):05d}"


def _p10(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[int(0.10 * (len(ordered) - 1))]


def build_v51_same_view_hierarchy(
    input_root: str | Path,
    contain_threshold: float = 0.85,
    area_ratio_threshold: float = 1.30,
    duplicate_iou_threshold: float = 0.80,
    max_relation_rows: int = 50000,
) -> dict[str, Any]:
    root = ROOT / input_root if not Path(input_root).is_absolute() else Path(input_root)
    relation_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    part_edge_count = 0
    duplicate_edge_count = 0
    sibling_edge_count = 0
    overlap_pair_count = 0
    whole_candidates: set[str] = set()
    part_candidates: set[str] = set()
    underseg_candidates: set[str] = set()
    confidence_values: list[float] = []
    frame_count = 0
    proposal_count = 0
    for scene_dir in _scene_dirs(root):
        scene = scene_dir.name
        for path in sorted(scene_dir.glob("*_masks.npz"), key=lambda p: _frame_id(p) if _frame_id(p) is not None else 10**12):
            frame_id = _frame_id(path)
            if frame_id is None:
                continue
            masks, areas = _load_masks(path)
            n = int(masks.shape[0])
            frame_count += 1
            proposal_count += n
            children_by_parent: dict[str, list[tuple[str, float]]] = defaultdict(list)
            frame_part_edges = 0
            frame_duplicate_edges = 0
            frame_overlap_pairs = 0
            for i in range(n):
                if int(areas[i]) <= 0:
                    continue
                left = masks[i]
                for j in range(i + 1, n):
                    if int(areas[j]) <= 0:
                        continue
                    inter = int(np.count_nonzero(left & masks[j]))
                    if inter <= 0:
                        continue
                    frame_overlap_pairs += 1
                    union = int(areas[i]) + int(areas[j]) - inter
                    iou = inter / max(float(union), 1.0)
                    contain_i_j = inter / max(float(areas[i]), 1.0)
                    contain_j_i = inter / max(float(areas[j]), 1.0)
                    area_ratio = max(float(areas[i]), float(areas[j])) / max(min(float(areas[i]), float(areas[j])), 1.0)
                    pid_i = _proposal_id(scene, frame_id, i)
                    pid_j = _proposal_id(scene, frame_id, j)
                    if iou >= duplicate_iou_threshold:
                        frame_duplicate_edges += 1
                        if len(relation_rows) < max_relation_rows:
                            relation_rows.append(
                                {
                                    "scene": scene,
                                    "frame_id": frame_id,
                                    "relation_id": f"duplicate::{pid_i}<->{pid_j}",
                                    "relation_type": "duplicate",
                                    "child_proposal_id": "",
                                    "parent_proposal_id": "",
                                    "proposal_a_id": pid_i,
                                    "proposal_b_id": pid_j,
                                    "containment_score": max(contain_i_j, contain_j_i),
                                    "iou_score": iou,
                                    "area_ratio": area_ratio,
                                    "relation_confidence": iou,
                                    "source_type": "sam2_rerun_containment_filtered",
                                    "uses_real_overlap_masks": True,
                                    "uses_flat_lattice_fallback": False,
                                    "uses_gt_for_prediction": False,
                                }
                            )
                    child = parent = ""
                    containment = 0.0
                    if contain_i_j >= contain_threshold and float(areas[j]) >= area_ratio_threshold * float(areas[i]):
                        child, parent, containment = pid_i, pid_j, contain_i_j
                    elif contain_j_i >= contain_threshold and float(areas[i]) >= area_ratio_threshold * float(areas[j]):
                        child, parent, containment = pid_j, pid_i, contain_j_i
                    if child and parent:
                        confidence = containment * min(1.0, area_ratio / max(area_ratio_threshold, 1e-6))
                        frame_part_edges += 1
                        whole_candidates.add(parent)
                        part_candidates.add(child)
                        children_by_parent[parent].append((child, confidence))
                        confidence_values.append(float(confidence))
                        if area_ratio >= 4.0:
                            underseg_candidates.add(parent)
                        if len(relation_rows) < max_relation_rows:
                            relation_rows.append(
                                {
                                    "scene": scene,
                                    "frame_id": frame_id,
                                    "relation_id": f"part_of::{child}->{parent}",
                                    "relation_type": "part_of",
                                    "child_proposal_id": child,
                                    "parent_proposal_id": parent,
                                    "proposal_a_id": child,
                                    "proposal_b_id": parent,
                                    "containment_score": containment,
                                    "iou_score": iou,
                                    "area_ratio": area_ratio,
                                    "relation_confidence": confidence,
                                    "source_type": "sam2_rerun_containment_filtered",
                                    "uses_real_overlap_masks": True,
                                    "uses_flat_lattice_fallback": False,
                                    "uses_gt_for_prediction": False,
                                }
                            )
            for parent, children in children_by_parent.items():
                unique_children = sorted({child for child, _confidence in children})
                confidence_by_child = {child: confidence for child, confidence in children}
                for idx, left in enumerate(unique_children):
                    for right in unique_children[idx + 1 :]:
                        sibling_edge_count += 1
                        confidence = min(confidence_by_child.get(left, 0.0), confidence_by_child.get(right, 0.0))
                        if len(relation_rows) < max_relation_rows:
                            relation_rows.append(
                                {
                                    "scene": scene,
                                    "frame_id": frame_id,
                                    "relation_id": f"sibling::{left}<->{right}@{parent}",
                                    "relation_type": "sibling_under_parent",
                                    "child_proposal_id": "",
                                    "parent_proposal_id": parent,
                                    "proposal_a_id": left,
                                    "proposal_b_id": right,
                                    "containment_score": "",
                                    "iou_score": "",
                                    "area_ratio": "",
                                    "relation_confidence": confidence,
                                    "source_type": "sam2_rerun_containment_filtered",
                                    "uses_real_overlap_masks": True,
                                    "uses_flat_lattice_fallback": False,
                                    "uses_gt_for_prediction": False,
                                }
                            )
            part_edge_count += frame_part_edges
            duplicate_edge_count += frame_duplicate_edges
            overlap_pair_count += frame_overlap_pairs
            metric_rows.append(
                {
                    "scene": scene,
                    "frame_id": frame_id,
                    "proposal_count": n,
                    "overlap_pair_count": frame_overlap_pairs,
                    "part_edge_count": frame_part_edges,
                    "duplicate_edge_count": frame_duplicate_edges,
                    "uses_real_overlap_masks": True,
                    "uses_flat_lattice_fallback": False,
                    "uses_gt_for_prediction": False,
                }
            )
    structural_gate = {
        "uses_real_overlap_masks": True,
        "uses_flat_lattice_fallback": False,
        "part_edge_count_pass": part_edge_count >= 200,
        "sibling_edge_count_pass": sibling_edge_count >= 100,
        "whole_candidate_count_pass": len(whole_candidates) >= 1,
        "diagnostic_precision_not_evaluated": True,
    }
    structural_gate["pass"] = bool(
        structural_gate["uses_real_overlap_masks"]
        and not structural_gate["uses_flat_lattice_fallback"]
        and structural_gate["part_edge_count_pass"]
        and structural_gate["sibling_edge_count_pass"]
        and structural_gate["whole_candidate_count_pass"]
    )
    return {
        "phase": "v51_r2_same_view_hierarchy",
        "created_at": utc_now(),
        "plan": PLAN_PATH,
        "input_root": _rel(root),
        "params": {
            "contain_threshold": float(contain_threshold),
            "area_ratio_threshold": float(area_ratio_threshold),
            "duplicate_iou_threshold": float(duplicate_iou_threshold),
        },
        "summary": {
            "frame_count": frame_count,
            "proposal_count": proposal_count,
            "overlap_pair_count": overlap_pair_count,
            "part_edge_count": part_edge_count,
            "sibling_edge_count": sibling_edge_count,
            "duplicate_edge_count": duplicate_edge_count,
            "conflict_edge_count": 0,
            "whole_candidate_count": len(whole_candidates),
            "part_candidate_count": len(part_candidates),
            "underseg_candidate_count": len(underseg_candidates),
            "relation_source_breakdown": {
                "sam2_rerun_containment_filtered_part_of": part_edge_count,
                "sam2_rerun_containment_filtered_sibling": sibling_edge_count,
                "sam2_rerun_containment_filtered_duplicate": duplicate_edge_count,
                "flat_lattice_fallback": 0,
            },
            "relation_confidence_mean": sum(confidence_values) / max(len(confidence_values), 1),
            "relation_confidence_p10": _p10(confidence_values),
            "uses_real_overlap_masks": True,
            "uses_flat_lattice_fallback": False,
            "uses_gt_for_prediction": False,
            "part_relation_precision": None,
            "sibling_relation_precision": None,
            "whole_candidate_purity": None,
            "diagnostic_precision_not_evaluated": True,
        },
        "gate": structural_gate,
        "relation_rows": relation_rows,
        "relation_metric_rows": metric_rows,
    }


def write_v51_same_view_hierarchy(output_root: str | Path, payload: dict[str, Any]) -> None:
    out = ROOT / output_root if not Path(output_root).is_absolute() else Path(output_root)
    write_json(out / "hierarchy_summary.json", payload)
    write_csv(out / "relation_rows.csv", payload["relation_rows"])
    write_csv(out / "relation_metric_rows.csv", payload["relation_metric_rows"])
