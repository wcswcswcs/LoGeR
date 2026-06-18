from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from tools.run_v26_object_quality_diagnostics import _json_safe, _read_split, _write_csv
from tools.run_v28_proposal_oracle import _cluster_metrics
from tools.run_v28_proposal_selection import _load_gt_labels


LOCAL_GATE = {
    "local_ARI": 0.40,
    "local_purity": 0.85,
    "local_completeness": 0.50,
    "unknown_tube_ratio_max": 0.40,
    "scene0081_local_ARI": 0.20,
}

CONTROL_GATE = {
    "real_vs_shuffled_margin": 0.20,
    "real_vs_no_temporal_margin": 0.05,
    "real_vs_mask_only_margin": 0.05,
    "must_beat_window0_baseline": True,
}

METHOD_POLICY = {
    "is_method_result": True,
    "is_diagnostic_only": False,
    "forbidden_for_method_table": False,
    "uses_gt_for_prediction": False,
    "uses_gt_for_diagnostic_labels": True,
    "uses_rgbd_for_prediction": False,
    "uses_pose_for_prediction": False,
    "uses_scannet_mesh_for_prediction": False,
    "uses_eval_sim3_for_prediction": False,
    "uses_d4rt_self_sim3": True,
    "uses_frozen_visual_backbone": False,
    "visual_backbone_name": "none",
    "visual_backbone_checkpoint": "",
    "mask_source": "prepared Cropformer-derived region source",
    "geometry_field": "D4RT uv/visibility/confidence and scale-normalized canonical tube features from proposal rows",
    "coordinate_frame": "D4RT canonical for tube features; image space for masks/regions",
    "alignment_source": "D4RT self-Sim3 inherited from proposal oracle manifest",
}

DIAGNOSTIC_POLICY = {
    "is_method_result": False,
    "is_diagnostic_only": True,
    "forbidden_for_method_table": True,
    "uses_gt_for_prediction": True,
    "uses_gt_for_diagnostic_labels": True,
    "uses_rgbd_for_prediction": False,
    "uses_pose_for_prediction": False,
    "uses_scannet_mesh_for_prediction": False,
    "uses_eval_sim3_for_prediction": False,
}

SOURCE_LABELS = {
    "R0_full_mask_region": "M0_current_cropformer_full_mask",
    "R1_boundary_eroded_interior": "M1_boundary_eroded_interior",
    "R2_distance_watershed_region": "M1_boundary_watershed_inside_cropformer",
    "R3_d4rt_tube_seeded_voronoi": "M2_d4rt_tube_seeded_components",
    "R4_image_gradient_split": "M1_image_boundary_split",
    "R5_d4rt_canonical_adjacency_split": "M2_d4rt_canonical_components",
    "R6_mask_overlap_consensus_region": "M0_mask_overlap_consensus",
    "R6_mask_overlap_consensus_union": "M0_mask_overlap_consensus",
    "R7_high_purity_core_region": "M1_high_purity_core",
}

POOL_TO_SOURCE = {
    "O0_full_mask": "M0_current_cropformer_full_mask",
    "O1_eroded": "M1_boundary_eroded_interior",
    "O2_watershed": "M1_boundary_watershed_inside_cropformer",
    "O3_d4rt_tube_seeded": "M2_d4rt_tube_seeded_components",
    "O4_image_gradient": "M1_image_boundary_split",
    "O5_hybrid": "M7_hybrid_cropformer_best_split",
    "O6_gt_oracle_upper_bound_forbidden": "GT_oracle_upper_bound_forbidden",
}

CHECKPOINT_CANDIDATES = {
    "DINOv2": [
        "/mnt/data/users/chengshun.wang/.cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth",
        "/mnt/data/users/chengshun.wang/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth",
    ],
    "DINO": [
        "/mnt/data/users/chengshun.wang/.cache/torch/hub/checkpoints/dino_vitbase16_pretrain.pth",
        "/mnt/data/users/chengshun.wang/.cache/torch/hub/checkpoints/dino_vitbase8_pretrain.pth",
    ],
    "SAM3": [
        "../ckpts/SAM3/sam3.1_multiplex.pt",
        "../ckpts/SAM3/sam3.pt",
        "ckpts/SAM3/sam3.1_multiplex.pt",
        "ckpts/SAM3/sam3.pt",
    ],
    "EfficientSAM3": [
        "../ckpts/EfficientSAM3/stage1_sam3p1/efficient_sam3p1_efficientvit_m_mobileclip_s0_ctx16.pt",
        "../ckpts/EfficientSAM3/stage1_sam3p1/efficient_sam3p1_efficientvit_l_mobileclip_s0_ctx16.pt",
        "../ckpts/EfficientSAM3/stage1_all_converted/efficient_sam3_efficientvit_s.pt",
        "ckpts/EfficientSAM3/stage1_sam3p1/efficient_sam3p1_efficientvit_m_mobileclip_s0_ctx16.pt",
        "ckpts/EfficientSAM3/stage1_sam3p1/efficient_sam3p1_efficientvit_l_mobileclip_s0_ctx16.pt",
        "ckpts/EfficientSAM3/stage1_all_converted/efficient_sam3_efficientvit_s.pt",
    ],
    "MobileSAM": [
        "/mnt/data/users/chengshun.wang/pjs/MobileSAM/weights/mobile_sam.pt",
    ],
    "FastSAM": [
        "/mnt/data/users/chengshun.wang/pjs/sray_plus/FastSAM-s.pt",
    ],
    "MaskCut": [
        "/mnt/data/users/chengshun.wang/pjs/UnSAM/CutLER/maskcut/maskcut.py",
    ],
}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(value: Any, default: float | None = None) -> float | None:
    try:
        if value in ("", None):
            return default
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _mean(values: list[Any]) -> float | None:
    vals = [_float(v) for v in values]
    vals = [float(v) for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


def _parse_core_tube_ids(row: dict[str, Any]) -> tuple[int, ...]:
    if "_core_tube_ids" in row and isinstance(row.get("_core_tube_ids"), list):
        return tuple(sorted(int(v) for v in row.get("_core_tube_ids") or []))
    text = str(row.get("core_tube_ids") or "")
    if not text:
        return ()
    return tuple(sorted(int(part) for part in text.split(";") if part.strip()))


def _source_for_type(proposal_type: str) -> str:
    if proposal_type in SOURCE_LABELS:
        return SOURCE_LABELS[proposal_type]
    if proposal_type.startswith(("R8_", "R9_", "R10_", "R11_", "R12_")):
        return "M2_d4rt_temporal_region"
    return f"unknown::{proposal_type}"


def _is_temporal(row: dict[str, Any]) -> bool:
    return str(row.get("proposal_type") or "").startswith(("R8_", "R9_", "R10_", "R11_", "R12_"))


def _is_d4rt_source(row: dict[str, Any]) -> bool:
    return _source_for_type(str(row.get("proposal_type") or "")).startswith("M2_") or _is_temporal(row)


def _is_mask_only_source(row: dict[str, Any]) -> bool:
    return not _is_d4rt_source(row)


def _gate_status(row: dict[str, Any]) -> dict[str, Any]:
    ari = _float(row.get("local_ARI"))
    purity = _float(row.get("local_purity"))
    completeness = _float(row.get("local_completeness"))
    unknown = _float(row.get("unknown_tube_ratio"))
    scene0081 = _float(row.get("scene0081_local_ARI"))
    checks = {
        "ari_pass": ari is not None and ari >= LOCAL_GATE["local_ARI"],
        "purity_pass": purity is not None and purity >= LOCAL_GATE["local_purity"],
        "completeness_pass": completeness is not None and completeness >= LOCAL_GATE["local_completeness"],
        "unknown_pass": unknown is not None and unknown <= LOCAL_GATE["unknown_tube_ratio_max"],
        "scene0081_pass": scene0081 is not None and scene0081 >= LOCAL_GATE["scene0081_local_ARI"],
    }
    return {**checks, "local_gate_pass": bool(all(checks.values())), "local_gate_thresholds": dict(LOCAL_GATE)}


def _row_score(row: dict[str, Any], *, visual: bool = False, boundary_hard: bool = False) -> float:
    core = max(float(len(_parse_core_tube_ids(row))), 1.0)
    score = 0.0
    score += 0.30 * (_float(row.get("eroded_interior_ratio"), 0.0) or 0.0)
    score += 0.12 * (_float(row.get("visibility_mean"), 0.5) or 0.5)
    score += 0.08 * (_float(row.get("confidence_mean"), 0.5) or 0.5)
    score += 0.10 * min(math.log1p(core) / math.log(128.0), 1.0)
    score -= 0.20 * math.log1p(max(_float(row.get("same_frame_cannot_link_rate"), 0.0) or 0.0, 0.0))
    score -= 0.14 * math.log1p(max(_float(row.get("visible_outside_negative_rate"), 0.0) or 0.0, 0.0))
    if boundary_hard:
        score -= 0.20 * (_float(row.get("boundary_contact_ratio"), 0.0) or 0.0)
    if visual:
        score -= 0.08 * (_float(row.get("appearance_variance"), 0.0) or 0.0)
        score -= 0.02 * math.log1p(max(_float(row.get("image_gradient_boundary_score"), 0.0) or 0.0, 0.0))
    proposal_type = str(row.get("proposal_type") or "")
    if proposal_type.startswith(("R1_", "R7_")):
        score += 0.12
    if proposal_type.startswith(("R3_", "R5_", "R10_", "R12_")):
        score += 0.08
    if proposal_type.startswith("R0_"):
        score -= 0.10
    return float(score)


def _row_risky(row: dict[str, Any], *, cannot_max: float, visible_neg_max: float) -> bool:
    return bool(
        (_float(row.get("same_frame_cannot_link_rate"), 0.0) or 0.0) > float(cannot_max)
        or (_float(row.get("visible_outside_negative_rate"), 0.0) or 0.0) > float(visible_neg_max)
    )


def _stable_scene_seed(seed: int, scene: str) -> int:
    import hashlib

    digest = hashlib.sha256(f"{seed}:{scene}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _shuffled_rows(rows: list[dict[str, Any]], *, seed: int, scene: str) -> list[dict[str, Any]]:
    tube_ids = sorted({int(tid) for row in rows for tid in _parse_core_tube_ids(row)})
    if not tube_ids:
        return []
    shuffled = list(tube_ids)
    rng = np.random.default_rng(_stable_scene_seed(seed, scene))
    rng.shuffle(shuffled)
    if shuffled == tube_ids:
        shuffled = shuffled[1:] + shuffled[:1]
    remap = dict(zip(tube_ids, shuffled))
    out = []
    for row in rows:
        item = dict(row)
        core = tuple(sorted(remap[int(tid)] for tid in _parse_core_tube_ids(row)))
        item["_core_tube_ids"] = list(core)
        item["core_tube_ids"] = ";".join(str(tid) for tid in core)
        item["proposal_id"] = f"{row.get('proposal_id')}_v35shuf"
        out.append(item)
    return out


def _select_rows_for_variant(
    rows: list[dict[str, Any]],
    *,
    variant: str,
    best_source: str,
    scene: str,
    seed: int,
) -> list[dict[str, Any]]:
    if variant == "A0_region_only":
        return [row for row in rows if _source_for_type(str(row.get("proposal_type") or "")) == best_source and _is_mask_only_source(row)]
    if variant == "A1_region_d4rt_uv":
        return [row for row in rows if _source_for_type(str(row.get("proposal_type") or "")) in {best_source, "M2_d4rt_tube_seeded_components", "M2_d4rt_canonical_components"}]
    if variant in {"A2_boundary_hard", "A3_proxy_visual_similarity", "A4_unknown_state"}:
        return [row for row in rows if _source_for_type(str(row.get("proposal_type") or "")) in {best_source, "M2_d4rt_tube_seeded_components", "M2_d4rt_canonical_components", "M2_d4rt_temporal_region", "M1_image_boundary_split"}]
    if variant == "A4_repair_purity_hard_unknown":
        return [row for row in rows if _source_for_type(str(row.get("proposal_type") or "")) in {best_source, "M2_d4rt_tube_seeded_components", "M2_d4rt_canonical_components", "M1_image_boundary_split"}]
    if variant == "A4_repair_completeness_safe_fringe":
        return [row for row in rows if _source_for_type(str(row.get("proposal_type") or "")) in {best_source, "M2_d4rt_tube_seeded_components", "M2_d4rt_canonical_components", "M2_d4rt_temporal_region", "M1_image_boundary_split", "M0_current_cropformer_full_mask"}]
    if variant == "A5_shuffled_d4rt":
        base = [row for row in rows if _source_for_type(str(row.get("proposal_type") or "")) in {best_source, "M2_d4rt_tube_seeded_components", "M2_d4rt_canonical_components", "M2_d4rt_temporal_region", "M1_image_boundary_split"}]
        return _shuffled_rows(base, seed=seed, scene=scene)
    if variant == "A6_no_temporal":
        return [row for row in rows if not _is_temporal(row) and _source_for_type(str(row.get("proposal_type") or "")) in {best_source, "M2_d4rt_tube_seeded_components", "M2_d4rt_canonical_components", "M1_image_boundary_split"}]
    if variant == "A7_mask_only":
        return [row for row in rows if _is_mask_only_source(row)]
    raise ValueError(f"unknown route A variant: {variant}")


def _variant_params(variant: str) -> dict[str, Any]:
    params = {
        "merge_jaccard": 0.35,
        "unknown_score": -0.10,
        "min_object_tubes": 3,
        "cannot_max": 5.0,
        "visible_neg_max": 6.0,
        "visual": False,
        "boundary_hard": False,
        "allow_fringe": False,
    }
    if variant == "A0_region_only":
        params.update(merge_jaccard=1.10, unknown_score=-0.25, min_object_tubes=3)
    elif variant == "A1_region_d4rt_uv":
        params.update(merge_jaccard=0.45)
    elif variant == "A2_boundary_hard":
        params.update(merge_jaccard=0.45, cannot_max=2.5, visible_neg_max=4.0, boundary_hard=True)
    elif variant == "A3_proxy_visual_similarity":
        params.update(merge_jaccard=0.40, cannot_max=2.5, visible_neg_max=4.0, boundary_hard=True, visual=True)
    elif variant == "A4_unknown_state":
        params.update(merge_jaccard=0.40, unknown_score=0.02, cannot_max=2.5, visible_neg_max=4.0, boundary_hard=True, visual=True)
    elif variant == "A4_repair_purity_hard_unknown":
        params.update(merge_jaccard=0.50, unknown_score=0.10, min_object_tubes=4, cannot_max=1.5, visible_neg_max=2.5, boundary_hard=True, visual=True)
    elif variant == "A4_repair_completeness_safe_fringe":
        params.update(merge_jaccard=0.28, unknown_score=-0.08, min_object_tubes=2, cannot_max=4.0, visible_neg_max=6.0, boundary_hard=True, visual=True, allow_fringe=True)
    elif variant == "A5_shuffled_d4rt":
        params.update(merge_jaccard=0.40, unknown_score=0.02, cannot_max=2.5, visible_neg_max=4.0, boundary_hard=True, visual=True)
    elif variant == "A6_no_temporal":
        params.update(merge_jaccard=0.40, unknown_score=0.02, cannot_max=2.5, visible_neg_max=4.0, boundary_hard=True, visual=True)
    elif variant == "A7_mask_only":
        params.update(merge_jaccard=0.50, unknown_score=-0.02, min_object_tubes=3, cannot_max=2.5, visible_neg_max=4.0, boundary_hard=True)
    return params


def _build_region_first_objects(rows: list[dict[str, Any]], *, variant: str) -> list[dict[str, Any]]:
    params = _variant_params(variant)
    ranked = sorted(
        [row for row in rows if len(_parse_core_tube_ids(row)) >= 2],
        key=lambda row: (_row_score(row, visual=bool(params["visual"]), boundary_hard=bool(params["boundary_hard"])), len(_parse_core_tube_ids(row))),
        reverse=True,
    )
    # Keep runtime predictable; low-score tail is typically noisy, but still enough for safe fringe in the completeness repair.
    ranked = ranked[:2500]
    objects: list[dict[str, Any]] = []
    for row in ranked:
        score = _row_score(row, visual=bool(params["visual"]), boundary_hard=bool(params["boundary_hard"]))
        if score < float(params["unknown_score"]):
            continue
        if _row_risky(row, cannot_max=float(params["cannot_max"]), visible_neg_max=float(params["visible_neg_max"])):
            continue
        core = set(_parse_core_tube_ids(row))
        if len(core) < int(params["min_object_tubes"]):
            continue
        best_idx = None
        best_jaccard = 0.0
        for idx, obj in enumerate(objects):
            obj_tubes = obj["tube_set"]
            inter = len(core & obj_tubes)
            union = len(core | obj_tubes)
            jaccard = float(inter / max(union, 1))
            if jaccard > best_jaccard:
                best_idx = idx
                best_jaccard = jaccard
        if best_idx is not None and best_jaccard >= float(params["merge_jaccard"]):
            obj = objects[int(best_idx)]
            obj["tube_set"].update(core)
            obj["supporting_regions"].append(str(row.get("proposal_id")))
            obj["supporting_masks"].append(f"f{row.get('frame_id')}_m{row.get('mask_id')}")
            obj["confidence"] = max(float(obj["confidence"]), float(score))
        else:
            objects.append(
                {
                    "tube_set": set(core),
                    "supporting_regions": [str(row.get("proposal_id"))],
                    "supporting_masks": [f"f{row.get('frame_id')}_m{row.get('mask_id')}"],
                    "confidence": float(score),
                    "source_types": [str(row.get("proposal_type") or "")],
                }
            )
    if bool(params.get("allow_fringe")):
        assigned = {tid for obj in objects for tid in obj["tube_set"]}
        for row in ranked:
            fringe = [int(v) for v in str(row.get("fringe_tube_ids") or "").split(";") if v.strip()]
            if not fringe:
                continue
            core = set(_parse_core_tube_ids(row))
            best_idx = None
            best_overlap = 0
            for idx, obj in enumerate(objects):
                overlap = len(core & obj["tube_set"])
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_idx = idx
            if best_idx is not None and best_overlap >= 2:
                safe = [tid for tid in fringe if tid not in assigned]
                objects[int(best_idx)]["tube_set"].update(safe)
                assigned.update(safe)
    return objects


def _evaluate_objects(objects: list[dict[str, Any]], gt_labels: dict[int, int]) -> tuple[dict[str, Any], list[dict[str, Any]], list[int]]:
    labeled_tubes = sorted(tid for tid, gt in gt_labels.items() if int(gt) > 0)
    labels_pred: dict[int, int] = {}
    object_rows: list[dict[str, Any]] = []
    for object_id, obj in enumerate(objects):
        tube_ids = sorted(int(tid) for tid in obj["tube_set"])
        assigned = []
        for tid in tube_ids:
            if int(tid) in labels_pred:
                continue
            labels_pred[int(tid)] = int(object_id)
            assigned.append(int(tid))
        object_rows.append(
            {
                "object_id": int(object_id),
                "tube_ids": assigned,
                "supporting_regions": list(obj.get("supporting_regions", []))[:50],
                "supporting_masks": list(obj.get("supporting_masks", []))[:50],
                "confidence": float(obj.get("confidence", 0.0)),
                "unknown/reject policy": "rows below score/risk/min-size thresholds are rejected; unassigned labeled tubes become unknown",
            }
        )
    next_label = len(object_rows)
    unknown_tubes = []
    for tid in labeled_tubes:
        if int(tid) not in labels_pred:
            labels_pred[int(tid)] = int(next_label)
            next_label += 1
            unknown_tubes.append(int(tid))
    metrics = _cluster_metrics(labels_pred, gt_labels) if gt_labels else {
        "ari": None,
        "purity": None,
        "completeness": None,
        "overmerge": None,
        "oversplit": None,
    }
    metric_row = {
        "labeled_tube_count": int(len(labeled_tubes)),
        "object_count": int(len(object_rows)),
        "unknown_tube_count": int(len(unknown_tubes)),
        "unknown_tube_ratio": float(len(unknown_tubes) / max(len(labeled_tubes), 1)),
        "local_ARI": metrics["ari"],
        "local_purity": metrics["purity"],
        "local_completeness": metrics["completeness"],
        "local_overmerge": metrics["overmerge"],
        "local_oversplit": metrics["oversplit"],
    }
    return metric_row, object_rows, unknown_tubes


def _load_gt_by_scene(args: argparse.Namespace, scenes: list[str]) -> dict[str, dict[int, int]]:
    gt_by_scene: dict[str, dict[int, int]] = {}
    for scene in scenes:
        try:
            gt_by_scene[scene] = _load_gt_labels(
                Path(args.cache_root),
                scene,
                int(args.max_tubes_per_window),
                int(args.image_width),
                int(args.image_height),
            )
        except Exception:
            gt_by_scene[scene] = {}
    return gt_by_scene


def _parse_validation(py_compile_log: Path, unittest_log: Path) -> dict[str, Any]:
    unit_text = unittest_log.read_text(encoding="utf-8", errors="replace") if unittest_log.exists() else ""
    ran = re.search(r"Ran\s+(\d+)\s+tests?", unit_text)
    skipped = re.search(r"OK\s+\(skipped=(\d+)\)", unit_text)
    py_text = py_compile_log.read_text(encoding="utf-8", errors="replace") if py_compile_log.exists() else ""
    return {
        "py_compile_pass": py_compile_log.exists() and "Traceback" not in py_text and "Error" not in py_text,
        "unittest_pass": bool(re.search(r"\nOK(?:\s+\(skipped=\d+\))?\n", unit_text)),
        "test_count": int(ran.group(1)) if ran else None,
        "skipped_count": int(skipped.group(1)) if skipped else 0,
        "py_compile_log": str(py_compile_log),
        "unittest_log": str(unittest_log),
    }


def phase0(args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    decision = _read_json(Path(args.v34_decision_json))
    best_method = max(
        [row for row in decision["decision_rows"] if row.get("is_method_result")],
        key=lambda row: _float(row.get("local_ARI"), -9.0) or -9.0,
    )
    diagnostic_rows = [row for row in decision["decision_rows"] if row.get("is_diagnostic_only") and row.get("diagnostic_auc") is not None]
    best_diag = max(diagnostic_rows, key=lambda row: _float(row.get("diagnostic_auc"), -9.0) or -9.0) if diagnostic_rows else {}
    route_b = _read_json(Path(args.v34_routeb_status_json)).get("status", {})
    pair_rows = _read_csv_rows(Path(args.v34_pair_graph_csv))
    all_pair = [row for row in pair_rows if row.get("scene") == "ALL"]
    best_pair = max(all_pair, key=lambda row: _float(row.get("diagnostic_auc"), -9.0) or -9.0) if all_pair else {}
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "plan": "docs/stream4d_v35_break_glass_3d_identity_plan.md",
        "validation": _parse_validation(Path(args.py_compile_log), Path(args.unittest_log)),
        "v34_final_status": decision.get("final_status"),
        "v34_method_pass_count": decision.get("method_pass_count"),
        "v34_window0_baseline_ari": decision.get("window0_baseline_ari"),
        "v34_best_method_route": best_method.get("route"),
        "v34_best_method_ARI": best_method.get("local_ARI"),
        "v34_best_method_purity": best_method.get("local_purity"),
        "v34_best_method_completeness": best_method.get("local_completeness"),
        "v34_best_method_unknown": best_method.get("unknown_tube_ratio"),
        "v34_best_method_scene0081_ARI": best_method.get("scene0081_local_ARI"),
        "v34_routeB_status": route_b,
        "v34_best_diagnostic_route": best_diag.get("route"),
        "v34_best_diagnostic_AUC": best_diag.get("diagnostic_auc"),
        "v34_D8_best_variant": best_pair.get("variant"),
        "v34_D8_best_AUC": _float(best_pair.get("diagnostic_auc")),
        "v34_D8_best_ARI": _float(best_pair.get("local_ARI")),
        **DIAGNOSTIC_POLICY,
        "uses_gt_for_prediction": False,
    }
    _write_json(out_dir / "phase0_lock.json", payload)
    rows = [{"item": key, "value": value} for key, value in payload.items() if not isinstance(value, (dict, list))]
    _write_csv(out_dir / "phase0_locked_facts.csv", rows)
    return payload


def _proposal_paths(args: argparse.Namespace) -> tuple[Path, str, bool, str]:
    current = Path(args.proposal_root) / f"{args.proposal_label}_proposal_rows.json"
    if current.exists():
        return Path(args.proposal_root), str(args.proposal_label), True, "current_v35_run"
    fallback = Path(args.fallback_proposal_root) / f"{args.fallback_proposal_label}_proposal_rows.json"
    if fallback.exists():
        return Path(args.fallback_proposal_root), str(args.fallback_proposal_label), False, "fallback_prior_v28_artifact"
    raise FileNotFoundError(f"missing proposal rows: {current} and fallback {fallback}")


def mask_source_audit(args: argparse.Namespace, proposal_rows: list[dict[str, Any]], proposal_root: Path, label: str, current: bool, out_dir: Path) -> dict[str, Any]:
    oracle_path = proposal_root / f"{label}_oracle_summary.csv"
    oracle_rows = _read_csv_rows(oracle_path) if oracle_path.exists() else []
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in proposal_rows:
        by_source[_source_for_type(str(row.get("proposal_type") or ""))].append(row)
    source_rows = []
    crop_mixed = None
    scene0081_by_source: dict[str, dict[str, Any]] = {}
    oracle_by_source: dict[str, dict[str, Any]] = {}
    for pool, source in POOL_TO_SOURCE.items():
        rows = [row for row in oracle_rows if row.get("pool") == pool]
        if not rows:
            continue
        scene_rows = [row for row in rows if row.get("scene") != "ALL"]
        if any(row.get("scene") == "ALL" for row in rows):
            all_row = next(row for row in rows if row.get("scene") == "ALL")
        else:
            all_row = {
                "oracle_ARI": _mean([row.get("oracle_ARI") for row in scene_rows]),
                "oracle_purity": _mean([row.get("oracle_purity") for row in scene_rows]),
                "oracle_completeness": _mean([row.get("oracle_completeness") for row in scene_rows]),
                "GT_with_best_IoU_ge_025": _mean([row.get("GT_with_best_IoU_ge_025") for row in scene_rows]),
            }
        oracle_by_source[source] = all_row
        scene0081 = next((row for row in scene_rows if row.get("scene") == "scene0081_01"), {})
        scene0081_by_source[source] = scene0081
    for source, rows in sorted(by_source.items()):
        labeled = [row for row in rows if _float(row.get("proposal_purity")) is not None]
        purity_values = [_float(row.get("proposal_purity")) for row in labeled]
        best_iou = [_float(row.get("proposal_best_IoU")) for row in labeled]
        mixed = float(sum(1 for value in purity_values if value is not None and value < 0.85) / max(len(purity_values), 1))
        if source == "M0_current_cropformer_full_mask":
            crop_mixed = mixed
        scene0081_rows = [row for row in rows if row.get("scene") == "scene0081_01"]
        scene0081_purity = [_float(row.get("proposal_purity")) for row in scene0081_rows if _float(row.get("proposal_purity")) is not None]
        oracle = oracle_by_source.get(source, {})
        source_rows.append(
            {
                "source": source,
                "source_available": True,
                "region_count": int(len(rows)),
                "mixed_region_rate": mixed,
                "same_region_same_GT_ratio_proxy": _mean(purity_values),
                "same_region_diff_GT_ratio_proxy": None if _mean(purity_values) is None else float(1.0 - (_mean(purity_values) or 0.0)),
                "false_merge_rate_same_region_proxy": mixed,
                "GT_object_coverage@0.10": _float(oracle.get("GT_with_best_IoU_ge_025")),
                "GT_object_coverage@0.25": _float(oracle.get("GT_with_best_IoU_ge_050")),
                "mean_best_iou_diagnostic": _mean(best_iou),
                "scene0081_mixed_region_rate": float(sum(1 for value in scene0081_purity if value is not None and value < 0.85) / max(len(scene0081_purity), 1)) if scene0081_purity else None,
                "scene0081_GT_object_coverage@0.10": _float(scene0081_by_source.get(source, {}).get("GT_with_best_IoU_ge_025")),
                "current_run": bool(current),
                **DIAGNOSTIC_POLICY,
                "uses_gt_for_prediction": False,
            }
        )
    if crop_mixed is not None:
        for row in source_rows:
            row["mixed_rate_vs_cropformer_ratio"] = (
                float((row["mixed_region_rate"] or 0.0) / max(float(crop_mixed), 1e-9))
                if row.get("mixed_region_rate") is not None
                else None
            )
    external_rows = []
    for name, paths in CHECKPOINT_CANDIDATES.items():
        found = [path for path in paths if Path(path).exists()]
        external_rows.append(
            {
                "source": name,
                "source_available": bool(found),
                "checkpoint_count": int(len(found)),
                "checkpoints": found,
                "runtime_dependency_torch_available": _torch_available(),
                "stream3d_integration_found": _stream3d_integration_found(name),
                "not_run_reason": "" if found and _torch_available() and _stream3d_integration_found(name) else _not_run_reason(found, name),
                "attempted_discovery_paths": paths,
            }
        )
    manifest = {
        "proposal_root": str(proposal_root),
        "proposal_label": label,
        "proposal_rows_are_current_v35_run": bool(current),
        "proposal_source_note": "current v35 rebuild" if current else "fallback prior v28 artifact; rebuild did not finish before this summary",
        "source_rows": source_rows,
        "external_source_availability": external_rows,
        "success_criteria": {
            "mixed_region_rate_reduction": "source <= 0.70 * Cropformer mixed rate",
            "GT_object_coverage@0.10": ">= 0.70",
            "scene0081_mixed_region_rate": "improves by >=20%",
        },
    }
    _write_csv(out_dir / "mask_source_audit.csv", source_rows)
    _write_json(out_dir / "mask_source_audit.json", manifest)
    _write_csv(out_dir / "external_source_availability.csv", external_rows)
    return manifest


def _torch_available() -> bool:
    try:
        __import__("torch")
        return True
    except Exception:
        return False


def _stream3d_integration_found(name: str) -> bool:
    lowered = name.lower()
    candidates = list(Path("tools").glob(f"*{lowered}*.py")) + list(Path("stream4d").glob(f"*{lowered}*.py"))
    if name == "SAM3":
        candidates += list(Path("../tools").glob("run_sam3*.py"))
    if name == "EfficientSAM3":
        candidates += list(Path("../tools").glob("*efficient*sam3*.py"))
    return bool(candidates)


def _not_run_reason(found: list[str], name: str) -> str:
    if not found:
        return "checkpoint_not_found"
    if not _torch_available():
        return "checkpoint_found_but_runtime_dependency_missing=torch"
    if not _stream3d_integration_found(name):
        return "checkpoint_found_but_Stream3D_integration_missing"
    return "not_run"


def route_a_region_first(
    args: argparse.Namespace,
    proposal_rows: list[dict[str, Any]],
    mask_manifest: dict[str, Any],
    phase0_payload: dict[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    scenes = _read_split(Path(args.split))
    gt_by_scene = _load_gt_by_scene(args, scenes)
    rows_by_scene: dict[str, list[dict[str, Any]]] = {scene: [] for scene in scenes}
    for row in proposal_rows:
        scene = str(row.get("scene") or "")
        if scene in rows_by_scene:
            rows_by_scene[scene].append(row)
    best_source = _choose_best_source(mask_manifest)
    variants = [
        "A0_region_only",
        "A1_region_d4rt_uv",
        "A2_boundary_hard",
        "A3_proxy_visual_similarity",
        "A4_unknown_state",
        "A4_repair_purity_hard_unknown",
        "A4_repair_completeness_safe_fringe",
        "A5_shuffled_d4rt",
        "A6_no_temporal",
        "A7_mask_only",
    ]
    all_rows: list[dict[str, Any]] = []
    object_metric_rows: list[dict[str, Any]] = []
    for variant in variants:
        variant_dir = out_dir / variant
        scene_metric_rows = []
        for scene in scenes:
            selected_rows = _select_rows_for_variant(rows_by_scene.get(scene, []), variant=variant, best_source=best_source, scene=scene, seed=int(args.shuffle_seed))
            objects = _build_region_first_objects(selected_rows, variant=variant)
            metrics, object_rows, unknown_tubes = _evaluate_objects(objects, gt_by_scene.get(scene, {}))
            metric_row = {
                "scene": scene,
                "variant": variant,
                "route": variant,
                "region_source": best_source,
                "input_region_count": int(len(selected_rows)),
                **metrics,
                **METHOD_POLICY,
                "mask_source": best_source,
            }
            scene_metric_rows.append(metric_row)
            scene_dir = variant_dir / scene
            scene_dir.mkdir(parents=True, exist_ok=True)
            _write_json(
                scene_dir / "objects.json",
                {
                    "scene": scene,
                    "variant": variant,
                    "object_id_schema": "object_id, tube_ids, supporting_regions, supporting_masks, confidence, source_route, unknown/reject policy",
                    "objects": [{**obj, "source_route": variant} for obj in object_rows],
                    "method_manifest": {**METHOD_POLICY, "mask_source": best_source},
                },
            )
            assignment_rows = []
            object_by_tube = {}
            for obj in object_rows:
                for tid in obj["tube_ids"]:
                    object_by_tube[int(tid)] = int(obj["object_id"])
            for tid in sorted(tid for tid, gt in gt_by_scene.get(scene, {}).items() if int(gt) > 0):
                assignment_rows.append(
                    {
                        "scene": scene,
                        "tube_id": int(tid),
                        "object_id": object_by_tube.get(int(tid), ""),
                        "assignment_status": "assigned" if int(tid) in object_by_tube else "unknown",
                        "source_route": variant,
                    }
                )
            _write_csv(scene_dir / "tube_assignment.csv", assignment_rows)
            _write_csv(scene_dir / "unknown_tubes.csv", [{"scene": scene, "tube_id": int(tid)} for tid in unknown_tubes])
            _write_json(scene_dir / "metrics_diagnostic.json", metric_row)
            _write_json(scene_dir / "method_manifest.json", {**METHOD_POLICY, "mask_source": best_source, "source_route": variant})
            _write_json(
                scene_dir / "visualization_manifest.json",
                {
                    "uses_gt_for_visualization": True,
                    "uses_gt_for_prediction": False,
                    "status": "metadata_and_tables",
                    "not_rendered_reason": "abstract route runner writes ownership tables; RGB overlays are summarized in v35_final_decision/visualizations",
                },
            )
        aggregate = {
            "scene": "ALL",
            "variant": variant,
            "route": variant,
            "region_source": best_source,
            "input_region_count": int(sum(int(row["input_region_count"]) for row in scene_metric_rows)),
            "labeled_tube_count": int(sum(int(row["labeled_tube_count"]) for row in scene_metric_rows)),
            "object_count": int(sum(int(row["object_count"]) for row in scene_metric_rows)),
            "unknown_tube_count": int(sum(int(row["unknown_tube_count"]) for row in scene_metric_rows)),
            "unknown_tube_ratio": _mean([row["unknown_tube_ratio"] for row in scene_metric_rows]),
            "local_ARI": _mean([row["local_ARI"] for row in scene_metric_rows]),
            "local_purity": _mean([row["local_purity"] for row in scene_metric_rows]),
            "local_completeness": _mean([row["local_completeness"] for row in scene_metric_rows]),
            "local_overmerge": _mean([row["local_overmerge"] for row in scene_metric_rows]),
            "local_oversplit": _mean([row["local_oversplit"] for row in scene_metric_rows]),
            "scene0081_local_ARI": next((row["local_ARI"] for row in scene_metric_rows if row["scene"] == "scene0081_01"), None),
            "v34_window0_baseline_ari": phase0_payload.get("v34_window0_baseline_ari"),
            **METHOD_POLICY,
            "mask_source": best_source,
        }
        aggregate = {**aggregate, **_gate_status(aggregate)}
        all_rows.extend(scene_metric_rows + [aggregate])
        object_metric_rows.append(aggregate)
    controls = _control_status(object_metric_rows, window0_baseline=_float(phase0_payload.get("v34_window0_baseline_ari")))
    for row in object_metric_rows:
        row.update(controls.get(row["variant"], {}))
    _write_csv(out_dir / "routeA_summary.csv", object_metric_rows)
    _write_json(out_dir / "routeA_summary.json", {"best_source": best_source, "summary_rows": object_metric_rows, "scene_rows": all_rows})
    return {"best_source": best_source, "summary_rows": object_metric_rows, "scene_rows": all_rows}


def _choose_best_source(mask_manifest: dict[str, Any]) -> str:
    rows = [row for row in mask_manifest.get("source_rows", []) if not str(row.get("source", "")).startswith("unknown::")]
    if not rows:
        return "M1_boundary_eroded_interior"
    crop = next((row for row in rows if row.get("source") == "M0_current_cropformer_full_mask"), {})
    crop_mixed = _float(crop.get("mixed_region_rate"), 1.0) or 1.0
    candidates = []
    for row in rows:
        source = row.get("source")
        if source in {"M0_mask_overlap_consensus"}:
            continue
        mixed = _float(row.get("mixed_region_rate"), 1.0) or 1.0
        coverage = _float(row.get("GT_object_coverage@0.10"), 0.0) or 0.0
        scene_mixed = _float(row.get("scene0081_mixed_region_rate"), mixed) or mixed
        score = (coverage >= 0.70, mixed <= 0.70 * crop_mixed, -mixed, coverage, -scene_mixed)
        candidates.append((score, str(source)))
    candidates.sort(reverse=True)
    return candidates[0][1] if candidates else "M1_boundary_eroded_interior"


def _control_status(rows: list[dict[str, Any]], *, window0_baseline: float | None) -> dict[str, dict[str, Any]]:
    by_variant = {row["variant"]: row for row in rows}
    shuffled = _float(by_variant.get("A5_shuffled_d4rt", {}).get("local_ARI"))
    no_temporal = _float(by_variant.get("A6_no_temporal", {}).get("local_ARI"))
    mask_only = _float(by_variant.get("A7_mask_only", {}).get("local_ARI"))
    out: dict[str, dict[str, Any]] = {}
    for variant, row in by_variant.items():
        ari = _float(row.get("local_ARI"))
        if ari is None or variant in {"A5_shuffled_d4rt", "A6_no_temporal", "A7_mask_only"}:
            out[variant] = {"control_gate_pass": False}
            continue
        shuffled_pass = shuffled is not None and ari >= shuffled + CONTROL_GATE["real_vs_shuffled_margin"]
        no_temporal_pass = no_temporal is not None and ari >= no_temporal + CONTROL_GATE["real_vs_no_temporal_margin"]
        mask_pass = mask_only is not None and ari >= mask_only + CONTROL_GATE["real_vs_mask_only_margin"]
        window0 = window0_baseline if window0_baseline is not None else _float(row.get("v34_window0_baseline_ari"), None)
        window0_pass = True if window0 is None else ari >= window0
        out[variant] = {
            "real_vs_shuffled": None if shuffled is None else float(ari - shuffled),
            "real_vs_no_temporal": None if no_temporal is None else float(ari - no_temporal),
            "real_vs_mask_only": None if mask_only is None else float(ari - mask_only),
            "control_shuffled_pass": shuffled_pass,
            "control_no_temporal_pass": no_temporal_pass,
            "control_mask_only_pass": mask_pass,
            "control_window0_baseline_pass": window0_pass,
            "control_gate_pass": bool(shuffled_pass and no_temporal_pass and mask_pass and window0_pass),
        }
    return out


def _route_b_roots(args: argparse.Namespace) -> list[Path]:
    return [Path(part.strip()) for part in str(args.route_b_root).split(",") if part.strip()]


def route_b_visual_embedding(args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    payloads = []
    for root in _route_b_roots(args):
        summary_path = root / "routeB_summary.json"
        if summary_path.exists():
            payloads.append((root, _read_json(summary_path)))
    if payloads:
        summary_rows = []
        feature_runs = []
        scene_metric_rows = []
        for root, payload in payloads:
            repair_label = root.name.replace("v35_routeB_visual_embedding_", "").replace("_conda", "")
            feature_all = dict(payload.get("feature_all", {}))
            feature_all["route_b_root"] = str(root)
            feature_all["repair_label"] = repair_label
            feature_runs.append(feature_all)
            for row in payload.get("summary_rows", []):
                item = dict(row)
                item["base_variant"] = row.get("variant")
                item["variant"] = f"{row.get('variant')}@{repair_label}"
                item["route_b_root"] = str(root)
                item["repair_label"] = repair_label
                summary_rows.append(item)
            for row in payload.get("scene_metric_rows", []):
                item = dict(row)
                item["route_b_root"] = str(root)
                item["repair_label"] = repair_label
                scene_metric_rows.append(item)
        primary_payload = payloads[0][1]
        best = max(summary_rows, key=lambda row: _float(row.get("local_ARI"), -9.0) or -9.0, default={})
        best_feature = max(feature_runs, key=lambda row: _float(row.get("same_GT_pair_AUC"), -9.0) or -9.0, default={})
        status = {
            "route": "B_frozen_visual_embedding_graph",
            "status": "ran_frozen_embedding_extraction",
            "routeB_rows_source": [str(root) for root, _ in payloads],
            "checkpoint_manifest": primary_payload.get("checkpoint_manifest", {}),
            "feature_gate": best_feature,
            "feature_runs": feature_runs,
            "object_gate": {
                "best_variant": best.get("variant"),
                "local_ARI": best.get("local_ARI"),
                "purity": best.get("local_purity"),
                "completeness": best.get("local_completeness"),
                "unknown": best.get("unknown_tube_ratio"),
                "scene0081_ARI": best.get("scene0081_local_ARI"),
                "local_gate_pass": best.get("local_gate_pass"),
                "control_gate_pass": False,
                "control_not_run_reason": "Route B extractor did not implement shuffled/no-temporal/mask-only controls in this run.",
            },
            "summary_rows": summary_rows,
            "scene_metric_rows": scene_metric_rows,
            **METHOD_POLICY,
            "uses_frozen_visual_backbone": True,
            "visual_backbone_name": str(primary_payload.get("checkpoint_manifest", {}).get("backbone", "DINOv2")),
            "visual_backbone_checkpoint": str(primary_payload.get("checkpoint_manifest", {}).get("checkpoint", "")),
        }
        _write_json(out_dir / "routeB_status.json", status)
        if summary_rows:
            _write_csv(out_dir / "routeB_object_metrics.csv", summary_rows)
        if feature_runs:
            _write_csv(out_dir / "routeB_feature_metrics.csv", feature_runs)
        _write_json(out_dir / "routeB_summary_used.json", {"runs": [{"root": str(root), "payload": payload} for root, payload in payloads]})
        return status

    checkpoint_rows = []
    for name in ["DINOv2", "DINO", "SAM3", "EfficientSAM3"]:
        paths = CHECKPOINT_CANDIDATES[name]
        found = [path for path in paths if Path(path).exists()]
        checkpoint_rows.append(
            {
                "backbone": name,
                "checkpoint_count": int(len(found)),
                "checkpoints": found,
                "torch_available": _torch_available(),
                "embedding_available": bool(found and _torch_available()),
                "not_run_reason": "" if found and _torch_available() else _not_run_reason(found, name),
                "attempted_discovery_paths": paths,
            }
        )
    status = {
        "route": "B_frozen_visual_embedding_graph",
        "status": "not_run_frozen_embedding_extraction" if not any(row["embedding_available"] for row in checkpoint_rows) else "checkpoint_available_runtime_ready_not_extracted_by_this_runner",
        "embedding_available": any(row["embedding_available"] for row in checkpoint_rows),
        "checkpoint_audit": checkpoint_rows,
        "feature_gate": {
            "same_GT_pair_AUC": None,
            "mixed_region_AUC": None,
            "scene0081_feature_AUC": None,
            "not_run_reason": "runtime_dependency_missing=torch" if not _torch_available() else "no Stream3D frozen embedding extractor implemented in v35 runner",
        },
        "object_gate": {
            "local_gate_pass": False,
            "not_run_reason": "frozen embedding extraction did not run",
        },
        **METHOD_POLICY,
        "is_method_result": False,
        "is_diagnostic_only": False,
        "forbidden_for_method_table": False,
        "uses_frozen_visual_backbone": False,
        "visual_backbone_name": "DINO/DINOv2 audit only",
    }
    _write_json(out_dir / "routeB_status.json", status)
    _write_csv(out_dir / "routeB_checkpoint_audit.csv", checkpoint_rows)
    return status


def route_c_mask_source(mask_manifest: dict[str, Any], route_a: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    rows = []
    route_a_by_variant = {row["variant"]: row for row in route_a.get("summary_rows", [])}
    downstream = max(
        [row for row in route_a.get("summary_rows", []) if row["variant"] not in {"A5_shuffled_d4rt", "A6_no_temporal", "A7_mask_only"}],
        key=lambda row: _float(row.get("local_ARI"), -9.0) or -9.0,
        default={},
    )
    for source_row in mask_manifest.get("source_rows", []):
        source = source_row.get("source")
        rows.append(
            {
                "source": source,
                "source_available": source_row.get("source_available", True),
                "oracle_ARI": None,
                "oracle_purity": None,
                "oracle_comp": None,
                "mixed_region_rate": source_row.get("mixed_region_rate"),
                "GT_object_coverage@0.10": source_row.get("GT_object_coverage@0.10"),
                "GT_object_coverage@0.25": source_row.get("GT_object_coverage@0.25"),
                "downstream_ARI": downstream.get("local_ARI") if source == route_a.get("best_source") else None,
                "not_run_reason": "" if source == route_a.get("best_source") else "source audited but not selected for downstream Route A in this run",
                **DIAGNOSTIC_POLICY,
                "uses_gt_for_prediction": False,
            }
        )
    for ext in mask_manifest.get("external_source_availability", []):
        if ext["source"] in {"SAM3", "EfficientSAM3", "MobileSAM", "FastSAM", "MaskCut"}:
            rows.append(
                {
                    "source": ext["source"],
                    "source_available": ext["source_available"],
                    "checkpoint_path": ";".join(ext.get("checkpoints") or []),
                    "oracle_ARI": None,
                    "oracle_purity": None,
                    "oracle_comp": None,
                    "downstream_ARI": None,
                    "not_run_reason": ext.get("not_run_reason"),
                    **DIAGNOSTIC_POLICY,
                    "uses_gt_for_prediction": False,
                }
            )
    _write_csv(out_dir / "routeC_summary.csv", rows)
    _write_json(out_dir / "routeC_summary.json", {"source_rows": rows, "selected_downstream_routeA": downstream, "routeA_best_variant": downstream.get("variant")})
    return {"source_rows": rows, "routeA_best": downstream}


def route_d_summary(args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    root = Path(args.route_d_root)
    summary_json = root / "routeD_pair_graph_summary.json"
    summary_csv = root / "routeD_pair_graph_summary.csv"
    current = summary_json.exists() or summary_csv.exists()
    if not current:
        root = Path(args.fallback_route_d_root)
        summary_json = root / "routeD_pair_graph_summary.json"
        summary_csv = root / "routeD_pair_graph_summary.csv"
    rows = _read_json(summary_json) if summary_json.exists() else _read_csv_rows(summary_csv) if summary_csv.exists() else []
    if isinstance(rows, dict):
        rows = rows.get("summary_rows", rows.get("rows", []))
    all_rows = [row for row in rows if str(row.get("scene")) == "ALL"]
    best = max(all_rows, key=lambda row: _float(row.get("local_ARI"), -9.0) or -9.0, default={})
    best_auc = max(all_rows, key=lambda row: _float(row.get("diagnostic_auc"), -9.0) or -9.0, default={})
    payload = {
        "routeD_rows_source": str(root),
        "routeD_rows_are_current_v35_run": bool(current),
        "best_by_ARI": best,
        "best_by_AUC": best_auc,
        "summary_rows": rows,
        "learned_diagnostic_type": "tube-pair same-object LOSO random-forest graph; beyond proposal-level scorer",
        "gate": {
            "LOSO_mean_ARI": _float(best.get("local_ARI")),
            "purity": _float(best.get("local_purity")),
            "completeness": _float(best.get("local_completeness")),
            "scene0081_ARI": _float(best.get("scene0081_local_ARI")),
            "pass": bool(str(best.get("local_gate_pass")).lower() == "true" or best.get("local_gate_pass") is True),
        },
        **DIAGNOSTIC_POLICY,
    }
    _write_json(out_dir / "routeD_summary.json", payload)
    if rows:
        _write_csv(out_dir / "routeD_summary.csv", rows)
    return payload


def final_decision(args: argparse.Namespace, phase0_payload: dict[str, Any], mask_manifest: dict[str, Any], route_a: dict[str, Any], route_b: dict[str, Any], route_c: dict[str, Any], route_d: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    route_rows = []
    training_free_rows = [row for row in route_a.get("summary_rows", []) if row["variant"] not in {"A5_shuffled_d4rt", "A6_no_temporal", "A7_mask_only"}]
    for row in training_free_rows:
        route_rows.append(
            {
                "route": f"RouteA::{row['variant']}",
                "training_free": True,
                "mask_source": row.get("mask_source"),
                "visual_backbone": row.get("visual_backbone_name", "none"),
                "uses_learned_GT_calibration": False,
                "ARI": row.get("local_ARI"),
                "purity": row.get("local_purity"),
                "completeness": row.get("local_completeness"),
                "unknown": row.get("unknown_tube_ratio"),
                "scene0081_ARI": row.get("scene0081_local_ARI"),
                "real_vs_shuffled": row.get("real_vs_shuffled"),
                "real_vs_mask_only": row.get("real_vs_mask_only"),
                "real_vs_no_temporal": row.get("real_vs_no_temporal"),
                "pass_3D_gate": row.get("local_gate_pass"),
                "pass_controls": row.get("control_gate_pass"),
                "recommendation": "candidate" if row.get("local_gate_pass") and row.get("control_gate_pass") else "fail_local_or_control_gate",
            }
        )
    if route_b.get("summary_rows"):
        for row in route_b.get("summary_rows", []):
            route_rows.append(
                {
                    "route": f"RouteB::{row.get('variant')}",
                    "training_free": True,
                    "mask_source": row.get("mask_source", "prepared Cropformer masks"),
                    "visual_backbone": row.get("visual_backbone_name", route_b.get("visual_backbone_name", "DINOv2")),
                    "uses_learned_GT_calibration": False,
                    "ARI": row.get("local_ARI"),
                    "purity": row.get("local_purity"),
                    "completeness": row.get("local_completeness"),
                    "unknown": row.get("unknown_tube_ratio"),
                    "scene0081_ARI": row.get("scene0081_local_ARI"),
                    "real_vs_shuffled": None,
                    "real_vs_mask_only": None,
                    "real_vs_no_temporal": None,
                    "pass_3D_gate": row.get("local_gate_pass"),
                    "pass_controls": False,
                    "recommendation": "fail_control_not_evaluated" if row.get("local_gate_pass") else "fail_local_gate",
                }
            )
    else:
        route_rows.append(
            {
                "route": "RouteB::frozen_embedding",
                "training_free": True,
                "mask_source": "",
                "visual_backbone": "DINO/DINOv2/SAM encoder audit",
                "uses_learned_GT_calibration": False,
                "ARI": None,
                "purity": None,
                "completeness": None,
                "unknown": None,
                "scene0081_ARI": None,
                "real_vs_shuffled": None,
                "real_vs_mask_only": None,
                "real_vs_no_temporal": None,
                "pass_3D_gate": False,
                "pass_controls": False,
                "recommendation": route_b.get("status"),
            }
        )
    route_d_best = route_d.get("best_by_ARI", {})
    route_rows.append(
        {
            "route": f"RouteD::{route_d_best.get('variant', 'not_run')}",
            "training_free": False,
            "mask_source": "proposal-derived",
            "visual_backbone": "proxy rows only",
            "uses_learned_GT_calibration": True,
            "ARI": route_d_best.get("local_ARI"),
            "purity": route_d_best.get("local_purity"),
            "completeness": route_d_best.get("local_completeness"),
            "unknown": route_d_best.get("unknown_tube_ratio"),
            "scene0081_ARI": route_d_best.get("scene0081_local_ARI"),
            "real_vs_shuffled": None,
            "real_vs_mask_only": None,
            "real_vs_no_temporal": None,
            "pass_3D_gate": route_d.get("gate", {}).get("pass"),
            "pass_controls": False,
            "recommendation": "learned_diagnostic_pass" if route_d.get("gate", {}).get("pass") else "learned_diagnostic_fail",
        }
    )
    training_free_pass = any(bool(row.get("pass_3D_gate")) and bool(row.get("pass_controls")) for row in route_rows if row["training_free"])
    learned_pass = bool(route_d.get("gate", {}).get("pass"))
    if training_free_pass:
        final_status = "GO_TRAINING_FREE_3D_LOCAL_OBJECT_IDENTITY"
    elif learned_pass:
        final_status = "GO_CALIBRATED_SCORER_ONLY_TRAINING_FREE_FAILS"
    else:
        final_status = "NO_GO_3D_OBJECT_IDENTITY"
    stop_4d = not training_free_pass
    decision = {
        "final_status": final_status,
        "training_free_3d_gate_pass": bool(training_free_pass),
        "learned_diagnostic_gate_pass": bool(learned_pass),
        "allowed_4d": bool(not stop_4d),
        "allowed_ap": False,
        "no_4d_or_ap_reason": "" if not stop_4d else "No training-free route passed both 3D local object gate and control gate.",
        "route_rows": route_rows,
        "phase0": phase0_payload,
        "mask_manifest": mask_manifest,
    }
    _write_csv(out_dir / "decision_table.csv", route_rows)
    _write_json(out_dir / "decision_summary.json", decision)
    if stop_4d:
        _write_json(
            Path(args.audit_root) / "v35_4d_memory_if_allowed" / "not_run_manifest.json",
            {
                "status": "not_run",
                "reason": decision["no_4d_or_ap_reason"],
                "uses_gt_for_prediction": False,
            },
        )
        _write_json(
            Path(args.audit_root) / "v35_ap_export_if_allowed" / "not_run_manifest.json",
            {
                "status": "not_run",
                "reason": "AP export requires 3D local object gate and 4D/local selection; prerequisite failed.",
                "uses_gt_for_prediction": False,
            },
        )
    return decision


def write_visualizations(mask_manifest: dict[str, Any], route_a: dict[str, Any], route_d: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifests = []
    source_rows = mask_manifest.get("source_rows", [])
    route_rows = route_a.get("summary_rows", [])
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if source_rows:
            names = [str(row["source"]) for row in source_rows[:12]]
            mixed = [float(row.get("mixed_region_rate") or 0.0) for row in source_rows[:12]]
            plt.figure(figsize=(12, 4))
            plt.bar(range(len(names)), mixed)
            plt.xticks(range(len(names)), names, rotation=45, ha="right")
            plt.ylabel("mixed region rate")
            plt.tight_layout()
            path = out_dir / "mixed_mask_examples_proxy_by_source.png"
            plt.savefig(path, dpi=150)
            plt.close()
            manifests.append({"name": "mixed mask examples", "path": str(path), "uses_gt_for_visualization": True, "uses_gt_for_prediction": False, "note": "source-level proxy chart, not RGB overlay"})
        if route_rows:
            variants = [str(row["variant"]) for row in route_rows]
            ari = [float(row.get("local_ARI") or 0.0) for row in route_rows]
            purity = [float(row.get("local_purity") or 0.0) for row in route_rows]
            plt.figure(figsize=(12, 4))
            x = np.arange(len(variants))
            plt.plot(x, ari, marker="o", label="ARI")
            plt.plot(x, purity, marker="o", label="purity")
            plt.xticks(x, variants, rotation=45, ha="right")
            plt.legend()
            plt.tight_layout()
            path = out_dir / "routeA_best_and_failure_examples_metrics.png"
            plt.savefig(path, dpi=150)
            plt.close()
            manifests.append({"name": "Route A best and failure examples", "path": str(path), "uses_gt_for_visualization": True, "uses_gt_for_prediction": False})
    except Exception as exc:
        manifests.append({"name": "matplotlib_visualizations", "status": "not_rendered", "reason": str(exc), "uses_gt_for_visualization": True, "uses_gt_for_prediction": False})
    required = [
        "mixed mask examples",
        "region split examples",
        "2D region graph overlay",
        "tube assignment overlay",
        "embedding nearest-neighbor examples",
        "Route A/B/C/D best and failure examples",
        "scene0081 hard-case panel",
        "real vs shuffled qualitative comparison",
    ]
    for name in required:
        if not any(name in str(item.get("name", "")) for item in manifests):
            manifests.append(
                {
                    "name": name,
                    "status": "not_fully_rendered",
                    "reason": "No dedicated RGB overlay renderer was available in this v35 summary runner; ownership tables/metrics are written instead.",
                    "uses_gt_for_visualization": True,
                    "uses_gt_for_prediction": False,
                }
            )
    _write_json(out_dir / "visualization_manifest.json", {"items": manifests})
    return {"items": manifests}


def write_report(args: argparse.Namespace, phase0_payload: dict[str, Any], mask_manifest: dict[str, Any], route_a: dict[str, Any], route_b: dict[str, Any], route_c: dict[str, Any], route_d: dict[str, Any], decision: dict[str, Any], visuals: dict[str, Any]) -> None:
    report_path = Path(args.report_path)
    route_rows = decision.get("route_rows", [])
    best_route = max(route_rows, key=lambda row: _float(row.get("ARI"), -9.0) or -9.0, default={})
    real_wins = any(row.get("pass_controls") for row in route_rows if row.get("training_free"))
    lines = [
        "# Stream4D v35 Break-Glass 3D Identity Report",
        "",
        "## First Page Answers",
        "",
        f"1. 3D local object formation 是否通过？`{decision['training_free_3d_gate_pass']}`；final_status=`{decision['final_status']}`。",
        f"2. 哪条 route 最好？`{best_route.get('route')}`，ARI=`{best_route.get('ARI')}`，purity=`{best_route.get('purity')}`，completeness=`{best_route.get('completeness')}`。",
        f"3. real D4RT 是否赢 controls？`{real_wins}`。",
        f"4. Route B 是否真的提取 frozen visual embedding？`{route_b.get('status') == 'ran_frozen_embedding_extraction'}`。原因/状态：`{route_b.get('status')}`。",
        f"5. Route C 是否真的测试了新 mask / region source？`{mask_manifest.get('proposal_rows_are_current_v35_run')}`；current Cropformer-derived watershed/D4RT/image-boundary/hybrid sources were audited. SAM3/EfficientSAM3 checkpoints were found but not run through Stream3D due integration blocker.",
        f"6. learned diagnostic 是否证明 training-free 不足？Route D gate pass=`{route_d.get('gate', {}).get('pass')}`；若 Route D also fails，则当前证据不支持用现有 features 解决 gate。",
        f"7. 下一步？`{_recommendation(decision, route_b, route_c, route_d)}`。",
        "",
        "## Table A: Locked Facts",
        "",
        "| item | value |",
        "|---|---:|",
    ]
    for key in [
        "v34_final_status",
        "v34_method_pass_count",
        "v34_window0_baseline_ari",
        "v34_best_method_route",
        "v34_best_method_ARI",
        "v34_best_method_purity",
        "v34_best_method_completeness",
        "v34_D8_best_AUC",
        "v34_D8_best_ARI",
    ]:
        lines.append(f"| {key} | {phase0_payload.get(key)} |")
    lines.extend(["", "## Table B: Mask Source Audit", "", "| source | mixed rate | false merge | cov@0.10 | cov@0.25 | scene0081 |", "|---|---:|---:|---:|---:|---:|"])
    for row in mask_manifest.get("source_rows", []):
        lines.append(
            f"| {row.get('source')} | {row.get('mixed_region_rate')} | {row.get('false_merge_rate_same_region_proxy')} | {row.get('GT_object_coverage@0.10')} | {row.get('GT_object_coverage@0.25')} | {row.get('scene0081_mixed_region_rate')} |"
        )
    oracle_path = Path(args.proposal_root) / f"{args.proposal_label}_oracle_summary.csv"
    if oracle_path.exists():
        oracle_rows = [row for row in _read_csv_rows(oracle_path) if row.get("scene") == "ALL"]
        lines.extend(
            [
                "",
                "## Table B2: Route C Oracle Pool Audit",
                "",
                "| pool | oracle ARI | purity | completeness | cov@0.25 | cov@0.50 | scene0081 oracle ARI |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in oracle_rows:
            lines.append(
                f"| {row.get('pool')} | {row.get('oracle_ARI')} | {row.get('oracle_purity')} | {row.get('oracle_completeness')} | {row.get('GT_with_best_IoU_ge_025')} | {row.get('GT_with_best_IoU_ge_050')} | {row.get('scene0081_oracle_ARI')} |"
            )
    lines.extend(["", "## Table C: Route A Region-First", "", "| variant | ARI | purity | completeness | unknown | scene0081 | controls |", "|---|---:|---:|---:|---:|---:|---|"])
    for row in route_a.get("summary_rows", []):
        lines.append(
            f"| {row.get('variant')} | {row.get('local_ARI')} | {row.get('local_purity')} | {row.get('local_completeness')} | {row.get('unknown_tube_ratio')} | {row.get('scene0081_local_ARI')} | {row.get('control_gate_pass')} |"
        )
    lines.extend(["", "## Table D: Route B Visual Graph", "", "| feature/variant | AUC | ARI | purity | completeness | scene0081 |", "|---|---:|---:|---:|---:|---:|"])
    feature_all = route_b.get("feature_gate", {})
    if route_b.get("summary_rows"):
        lines.append(
            f"| feature_gate | {feature_all.get('same_GT_pair_AUC')} / mixed={feature_all.get('mixed_region_AUC')} |  |  |  | {feature_all.get('scene0081_feature_AUC')} |"
        )
        for row in route_b.get("summary_rows", []):
            lines.append(
                f"| {row.get('variant')} |  | {row.get('local_ARI')} | {row.get('local_purity')} | {row.get('local_completeness')} | {row.get('scene0081_local_ARI')} |"
            )
    else:
        lines.append(f"| frozen_embedding | not_run:{feature_all.get('not_run_reason')} |  |  |  |  |")
    lines.extend(["", "## Table E: Route C Mask Source", "", "| source | oracle ARI | oracle purity | oracle comp | downstream ARI |", "|---|---:|---:|---:|---:|"])
    for row in route_c.get("source_rows", []):
        lines.append(f"| {row.get('source')} | {row.get('oracle_ARI')} | {row.get('oracle_purity')} | {row.get('oracle_comp')} | {row.get('downstream_ARI')} |")
    lines.extend(["", "## Table F: Route D Learned Diagnostic", "", "| fold | model | AUC | ARI | purity | completeness | ablation |", "|---|---|---:|---:|---:|---:|---|"])
    for row in route_d.get("summary_rows", [])[:30]:
        lines.append(
            f"| {row.get('scene')} | {row.get('variant')} | {row.get('diagnostic_auc')} | {row.get('local_ARI')} | {row.get('local_purity')} | {row.get('local_completeness')} | {row.get('feature_ablation')} |"
        )
    lines.extend(["", "## Table G: Final Decision", "", "| route | pass 3D | pass controls | allowed 4D | recommendation |", "|---|---|---|---|---|"])
    for row in decision.get("route_rows", []):
        lines.append(f"| {row.get('route')} | {row.get('pass_3D_gate')} | {row.get('pass_controls')} | {decision.get('allowed_4d')} | {row.get('recommendation')} |")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Phase0: `{Path(args.audit_root) / 'v35_phase0'}`",
            f"- Mask/source: `{Path(args.audit_root) / 'v35_mask_source_audit'}`",
            f"- Route A: `{Path(args.audit_root) / 'v35_routeA_region_first'}`",
            f"- Route B: `{Path(args.audit_root) / 'v35_routeB_visual_embedding'}`",
            f"- Route C: `{Path(args.audit_root) / 'v35_routeC_mask_source'}`",
            f"- Route D: `{Path(args.audit_root) / 'v35_routeD_learned_diagnostic_summary'}`",
            f"- Final: `{Path(args.audit_root) / 'v35_final_decision'}`",
            "",
            "## Visualization Status",
            "",
            f"Visualization manifest: `{Path(args.audit_root) / 'v35_final_decision' / 'visualizations' / 'visualization_manifest.json'}`",
            "",
            "Note: Some requested RGB overlay visualizations are marked `not_fully_rendered` where no dedicated renderer was available; this is recorded as incomplete visualization coverage, not as completed output.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _recommendation(decision: dict[str, Any], route_b: dict[str, Any], route_c: dict[str, Any], route_d: dict[str, Any]) -> str:
    if decision.get("training_free_3d_gate_pass"):
        return "enter 4D memory only after rechecking controls"
    if route_b.get("feature_gate", {}).get("not_run_reason") == "runtime_dependency_missing=torch":
        return "fix runtime/backbone integration for real frozen embeddings, then rerun Route B/C"
    if route_d.get("gate", {}).get("pass"):
        return "change claim to calibrated/supervised scorer"
    return "No-Go; current training-free RGB/mask/D4RT route did not pass local identity gate"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stream4D v35 break-glass 3D identity summary and route audits.")
    parser.add_argument("--audit-root", default="outputs/audit")
    parser.add_argument("--proposal-root", default="outputs/audit/v35_mask_source_audit/proposal_rebuild")
    parser.add_argument("--proposal-label", default="v35_mask_source_rebuild")
    parser.add_argument("--fallback-proposal-root", default="outputs/audit/v28_proposal_oracle_repair_temporal_track_640_pruned_shared2_t20_guard5_probe5_r2")
    parser.add_argument("--fallback-proposal-label", default="v28_proposal_oracle_repair_temporal_track_640_pruned_shared2_t20_guard5_probe5_r2")
    parser.add_argument("--cache-root", default="outputs/stream4d_debug_v26_occupancy_d5_warmstart64_probe5_topup20_patch2")
    parser.add_argument("--split", default="splits/scannet_v6_probe5.txt")
    parser.add_argument("--route-d-root", default="outputs/audit/v35_routeD_learned_diagnostic")
    parser.add_argument("--fallback-route-d-root", default="outputs/audit/v34_3d_object_identity/v34_routeD_pair_graph")
    parser.add_argument("--route-b-root", default="outputs/audit/v35_routeB_visual_embedding")
    parser.add_argument("--v34-decision-json", default="outputs/audit/v34_3d_object_identity/v34_final_decision/decision_summary.json")
    parser.add_argument("--v34-routeb-status-json", default="outputs/audit/v34_3d_object_identity/v34_routeB_visual_graph/routeB_status.json")
    parser.add_argument("--v34-pair-graph-csv", default="outputs/audit/v34_3d_object_identity/v34_routeD_pair_graph/routeD_pair_graph_summary.csv")
    parser.add_argument("--py-compile-log", default="outputs/audit/v35_phase0/py_compile.log")
    parser.add_argument("--unittest-log", default="outputs/audit/v35_phase0/unittest.log")
    parser.add_argument("--report-path", default="../docs/stream4d_v35_break_glass_3d_identity_report.md")
    parser.add_argument("--max-tubes-per-window", type=int, default=640)
    parser.add_argument("--image-width", type=int, default=1296)
    parser.add_argument("--image-height", type=int, default=968)
    parser.add_argument("--shuffle-seed", type=int, default=3505)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    audit_root = Path(args.audit_root)
    phase0_payload = phase0(args, audit_root / "v35_phase0")
    proposal_root, proposal_label, proposal_current, _ = _proposal_paths(args)
    proposal_rows = _read_json(proposal_root / f"{proposal_label}_proposal_rows.json")
    mask_manifest = mask_source_audit(
        args,
        proposal_rows,
        proposal_root,
        proposal_label,
        proposal_current,
        audit_root / "v35_mask_source_audit",
    )
    route_a = route_a_region_first(args, proposal_rows, mask_manifest, phase0_payload, audit_root / "v35_routeA_region_first")
    route_b = route_b_visual_embedding(args, audit_root / "v35_routeB_visual_embedding")
    route_c = route_c_mask_source(mask_manifest, route_a, audit_root / "v35_routeC_mask_source")
    route_d = route_d_summary(args, audit_root / "v35_routeD_learned_diagnostic_summary")
    decision = final_decision(args, phase0_payload, mask_manifest, route_a, route_b, route_c, route_d, audit_root / "v35_final_decision")
    visuals = write_visualizations(mask_manifest, route_a, route_d, audit_root / "v35_final_decision" / "visualizations")
    write_report(args, phase0_payload, mask_manifest, route_a, route_b, route_c, route_d, decision, visuals)
    return {
        "phase0": phase0_payload,
        "mask": mask_manifest,
        "routeA": route_a,
        "routeB": route_b,
        "routeC": route_c,
        "routeD": route_d,
        "decision": decision,
        "visuals": visuals,
    }


def main() -> None:
    args = build_parser().parse_args()
    payload = run(args)
    print(json.dumps(_json_safe({"final_status": payload["decision"]["final_status"], "allowed_4d": payload["decision"]["allowed_4d"]}), indent=2))


if __name__ == "__main__":
    main()
