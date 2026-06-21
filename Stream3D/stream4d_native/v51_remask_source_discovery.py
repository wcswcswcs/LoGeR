from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from stream4d_native.v47_common import ROOT, read_json, utc_now, write_csv, write_json
from stream4d_native.v50_stage1 import build_v50_fact_lock


PLAN_PATH = "docs/stream4d_v51_r2_mosaic_remask_lift_codex_plan.md"
PROBE_SCENES = ["scene0011_00", "scene0030_00", "scene0050_00", "scene0081_01", "scene0591_00"]


def _rel(path: str | Path) -> str:
    path_obj = Path(path)
    if not path_obj.is_absolute():
        path_obj = ROOT / path_obj
    try:
        return str(path_obj.relative_to(ROOT))
    except ValueError:
        return str(path_obj)


def _read_optional_json(path: str | Path) -> dict[str, Any]:
    path_obj = ROOT / path if not isinstance(path, Path) else path
    if not path_obj.is_absolute():
        path_obj = ROOT / path_obj
    if not path_obj.exists():
        return {"missing": True, "path": _rel(path_obj)}
    payload = read_json(path_obj)
    return payload if isinstance(payload, dict) else {"payload": payload}


def _bool_path(path: str | Path) -> bool:
    path_obj = ROOT / path if not isinstance(path, Path) else path
    if not path_obj.is_absolute():
        path_obj = ROOT / path_obj
    return path_obj.exists()


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _median(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _frame_id(path: Path) -> int | None:
    stem = path.stem
    if stem.isdigit():
        return int(stem)
    match = re.search(r"frame0*([0-9]+)", stem)
    return int(match.group(1)) if match else None


def _load_mask_stack_from_npz(path: Path) -> tuple[np.ndarray | None, str, list[str]]:
    errors: list[str] = []
    try:
        data = np.load(path, allow_pickle=True)
    except Exception as exc:  # pragma: no cover - defensive audit path
        return None, "", [f"{type(exc).__name__}: {exc}"]
    key = ""
    for candidate in ("masks", "pred_masks", "mask_stack"):
        if candidate in data.files:
            key = candidate
            break
    if not key and data.files:
        key = data.files[0]
    if not key:
        return None, "", ["npz has no arrays"]
    try:
        arr = np.asarray(data[key])
    except Exception as exc:  # pragma: no cover
        return None, key, [f"{type(exc).__name__}: {exc}"]
    if arr.ndim != 3:
        errors.append(f"array key {key} has ndim={arr.ndim}, expected 3")
        return None, key, errors
    if arr.dtype != bool:
        arr = arr != 0
    return arr, key, errors


def _pair_metrics(mask_stack: np.ndarray, max_masks: int = 96) -> dict[str, Any]:
    n_total = int(mask_stack.shape[0])
    if n_total == 0:
        return {
            "mask_count": 0,
            "pair_count_evaluated": 0,
            "overlap_pair_count": 0,
            "containment_pair_count": 0,
            "duplicate_pair_count": 0,
            "whole_candidate_count": 0,
            "part_candidate_count": 0,
            "sampled_mask_count": 0,
            "mask_sampling_note": "",
        }
    if n_total > max_masks:
        areas_all = mask_stack.reshape(n_total, -1).sum(axis=1)
        keep = np.argsort(areas_all)[-max_masks:]
        masks = mask_stack[np.sort(keep)]
        sampling_note = f"largest_area_top{max_masks}_of_{n_total}"
    else:
        masks = mask_stack
        sampling_note = ""
    n = int(masks.shape[0])
    flat = masks.reshape(n, -1)
    areas = flat.sum(axis=1).astype(np.float64)
    overlap_pair_count = 0
    containment_pair_count = 0
    duplicate_pair_count = 0
    pair_count = 0
    parents: set[int] = set()
    parts: set[int] = set()
    for i in range(n):
        if areas[i] <= 0:
            continue
        left = masks[i]
        for j in range(i + 1, n):
            if areas[j] <= 0:
                continue
            pair_count += 1
            inter = int(np.count_nonzero(left & masks[j]))
            if inter <= 0:
                continue
            overlap_pair_count += 1
            ci = inter / float(areas[i])
            cj = inter / float(areas[j])
            ratio = max(float(areas[i]), float(areas[j])) / max(min(float(areas[i]), float(areas[j])), 1.0)
            if min(ci, cj) >= 0.85 and ratio <= 1.10:
                duplicate_pair_count += 1
            if max(ci, cj) >= 0.85 and ratio >= 1.30:
                containment_pair_count += 1
                if ci >= cj:
                    parts.add(i)
                    parents.add(j)
                else:
                    parts.add(j)
                    parents.add(i)
    return {
        "mask_count": n_total,
        "pair_count_evaluated": pair_count,
        "overlap_pair_count": overlap_pair_count,
        "containment_pair_count": containment_pair_count,
        "duplicate_pair_count": duplicate_pair_count,
        "whole_candidate_count": len(parents),
        "part_candidate_count": len(parts),
        "sampled_mask_count": n,
        "mask_sampling_note": sampling_note,
    }


def _inspect_png_label_source(
    source_id: str,
    source_dir_name: str,
    role: str,
    max_sample_scenes: int = 5,
    max_files_per_scene: int = 4,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    base = ROOT / "data/scannet/processed"
    scenes = sorted([p for p in base.glob("scene*") if p.is_dir()])
    available_scene_count = 0
    png_total = 0
    unique_counts: list[float] = []
    sample_rows: list[dict[str, Any]] = []
    sampled_scene_count = 0
    for scene_dir in scenes:
        mask_dir = scene_dir / source_dir_name / "mask"
        files = sorted(mask_dir.glob("*.png"), key=lambda p: _frame_id(p) if _frame_id(p) is not None else 10**12)
        if not files:
            continue
        available_scene_count += 1
        png_total += len(files)
        if sampled_scene_count >= max_sample_scenes:
            continue
        sampled_scene_count += 1
        if len(files) > max_files_per_scene:
            idx = np.linspace(0, len(files) - 1, max_files_per_scene).round().astype(int)
            chosen = [files[int(i)] for i in sorted(set(idx.tolist()))]
        else:
            chosen = files
        for path in chosen:
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image is None:
                sample_rows.append(
                    {
                        "source_id": source_id,
                        "scene": scene_dir.name,
                        "path": _rel(path),
                        "load_ok": False,
                        "error": "cv2.imread returned None",
                    }
                )
                continue
            if image.ndim == 3:
                image = image[..., 0]
            values = np.unique(image)
            unique_counts.append(float(len(values)))
            sample_rows.append(
                {
                    "source_id": source_id,
                    "scene": scene_dir.name,
                    "frame_id": _frame_id(path),
                    "path": _rel(path),
                    "load_ok": True,
                    "format": "single_channel_png_label_map",
                    "shape": list(image.shape),
                    "dtype": str(image.dtype),
                    "unique_label_count": int(len(values)),
                    "min_label": int(values.min()) if len(values) else None,
                    "max_label": int(values.max()) if len(values) else None,
                    "first_unique_labels": [int(v) for v in values[:10].tolist()],
                    "overlap_capable": False,
                }
            )
    row = {
        "source_id": source_id,
        "source_family": source_dir_name,
        "role": role,
        "path_pattern": f"data/scannet/processed/<scene>/{source_dir_name}/mask/*.png",
        "available": available_scene_count > 0,
        "available_scene_count": available_scene_count,
        "file_count": png_total,
        "sampled_scene_count": sampled_scene_count,
        "sampled_file_count": sum(1 for r in sample_rows if r.get("source_id") == source_id),
        "format": "single_channel_png_label_map",
        "overlap_capable": False,
        "preserves_nxhxw_stack": False,
        "stream3d_current_mainline": source_id == "stream3d_cropformer_flat_png",
        "main_branch_eligible": False,
        "diagnostic_only": True,
        "mean_unique_labels_sample": _mean(unique_counts),
        "median_unique_labels_sample": _median(unique_counts),
        "source_note": "Stream3D flat PNG label map; one label per pixel so exact overlapping proposals are not recoverable from this file.",
    }
    return row, sample_rows


def _discover_npz_roots() -> list[Path]:
    roots: set[Path] = set()
    for base in [ROOT / "outputs/audit", ROOT / "data"]:
        if not base.exists():
            continue
        for path in base.rglob("*_masks.npz"):
            lowered = str(path).lower()
            if any(token in lowered for token in ("sam", "maskcut", "watershed", "source", "objectlet", "proposal")):
                roots.add(path.parent)
    return sorted(roots)


def _source_family_from_path(path: Path) -> str:
    lowered = str(path).lower()
    if "sam2" in lowered:
        return "sam2_npz"
    if "efficientsam" in lowered:
        return "efficientsam_npz"
    if re.search(r"(^|/|_)sam($|/|_)", lowered):
        return "sam_npz"
    if "maskcut" in lowered:
        return "dinov2_maskcut_npz"
    if "watershed" in lowered:
        return "watershed_npz"
    return "external_npz"


def _inspect_npz_root(root: Path, max_files: int = 8) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    files_all = sorted(root.glob("*_masks.npz"), key=lambda p: _frame_id(p) if _frame_id(p) is not None else 10**12)
    if not files_all:
        files_all = sorted(root.glob("*.npz"), key=lambda p: _frame_id(p) if _frame_id(p) is not None else 10**12)
    if len(files_all) > max_files:
        idx = np.linspace(0, len(files_all) - 1, max_files).round().astype(int)
        files = [files_all[int(i)] for i in sorted(set(idx.tolist()))]
        sampled = True
    else:
        files = files_all
        sampled = False
    source_family = _source_family_from_path(root)
    source_id = f"{source_family}:{_rel(root)}"
    sample_rows: list[dict[str, Any]] = []
    totals = Counter()
    shapes: Counter[str] = Counter()
    errors: list[str] = []
    for path in files:
        masks, key, load_errors = _load_mask_stack_from_npz(path)
        errors.extend(load_errors)
        row: dict[str, Any] = {
            "source_id": source_id,
            "source_family": source_family,
            "path": _rel(path),
            "frame_id": _frame_id(path),
            "load_ok": masks is not None,
            "npz_key": key,
            "errors": load_errors,
        }
        if masks is not None:
            metrics = _pair_metrics(masks)
            row.update(
                {
                    "format": "npz_mask_stack",
                    "shape": list(masks.shape),
                    "dtype_after_load": "bool",
                    "overlap_capable": True,
                    **metrics,
                }
            )
            for key2 in [
                "mask_count",
                "pair_count_evaluated",
                "overlap_pair_count",
                "containment_pair_count",
                "duplicate_pair_count",
                "whole_candidate_count",
                "part_candidate_count",
            ]:
                totals[key2] += int(metrics.get(key2, 0) or 0)
            shapes[str(tuple(masks.shape[1:]))] += 1
        sample_rows.append(row)
    frame_count = sum(1 for row in sample_rows if row.get("load_ok"))
    mask_count = totals["mask_count"]
    containment_ratio = totals["containment_pair_count"] / max(totals["pair_count_evaluated"], 1)
    whole_ratio = totals["whole_candidate_count"] / max(mask_count, 1)
    row = {
        "source_id": source_id,
        "source_family": source_family,
        "role": "external_discovery_diagnostic",
        "path_pattern": f"{_rel(root)}/*_masks.npz",
        "available": bool(files_all),
        "available_scene_count": None,
        "file_count": len(files_all),
        "sampled_file_count": len(files),
        "sampled": sampled,
        "format": "npz_mask_stack",
        "overlap_capable": frame_count > 0,
        "preserves_nxhxw_stack": frame_count > 0,
        "stream3d_current_mainline": False,
        "main_branch_eligible": source_family in {"sam_npz", "sam2_npz"},
        "diagnostic_only": source_family not in {"sam_npz", "sam2_npz"},
        "frame_count_sampled": frame_count,
        "mask_count_sampled": mask_count,
        "mean_masks_per_frame_sampled": mask_count / frame_count if frame_count else 0.0,
        "overlap_pair_count_sampled": totals["overlap_pair_count"],
        "containment_pair_count_sampled": totals["containment_pair_count"],
        "containment_pair_ratio_sampled": containment_ratio,
        "whole_candidate_count_sampled": totals["whole_candidate_count"],
        "whole_candidate_ratio_sampled": whole_ratio,
        "duplicate_pair_count_sampled": totals["duplicate_pair_count"],
        "shape_counts_sampled": dict(shapes),
        "errors": errors[:5],
        "source_note": "Sampled external NPZ discovery. Non-SAM families are backup/diagnostic only under the user source constraint.",
    }
    return row, sample_rows


def build_v51_fact_lock() -> dict[str, Any]:
    v50_fact = build_v50_fact_lock()
    v50_final_path = "outputs/audit/v50_final_decision/v50_final_decision.json"
    v50_full_path = "outputs/audit/v50_full_stage1/full_stage1_summary.json"
    v50_ap_path = "outputs/audit/v50_ap_diagnostic/ap_export_summary.json"
    v50_mask_path = "outputs/audit/v50_mask_source_audit/mask_source_summary.json"
    v50_final = _read_optional_json(v50_final_path)
    v50_full = _read_optional_json(v50_full_path)
    v50_ap = _read_optional_json(v50_ap_path)
    v50_mask = _read_optional_json(v50_mask_path)

    stream3d_rows, _samples = _inspect_png_label_source(
        "stream3d_cropformer_flat_png", "output_Cropformer", "stream3d_current_mainline"
    )
    sam2_rows, _ = _inspect_png_label_source("stream3d_sam2_flat_png", "output_SAM2", "stream3d_supported_but_absent")
    sam_rows, _ = _inspect_png_label_source("stream3d_sam_flat_png", "output_SAM", "legacy_sam_candidate_absent")

    fact_map = dict(v50_fact.get("fact_map", {}))
    fact_map.update(
        {
            "plan": PLAN_PATH,
            "stream3d_current_mask_source": "Cropformer_flat_png",
            "stream3d_current_mask_source_evidence": "run.py Step1 CropFormer active; SAM2 command commented; configs/default backbone Cropformer",
            "stream3d_current_cropformer_scene_count": stream3d_rows["available_scene_count"],
            "stream3d_current_cropformer_png_total": stream3d_rows["file_count"],
            "stream3d_current_mask_overlap_capable": False,
            "stream3d_sam2_png_scene_count": sam2_rows["available_scene_count"],
            "stream3d_sam_png_scene_count": sam_rows["available_scene_count"],
            "stream3d_sam2_checkpoint_at_script_path_exists": _bool_path("third_party/sam2/checkpoints/sam2.1_hiera_large.pt"),
            "cropformer_checkpoint_exists": _bool_path("third_party/seg_models/Mask2Former_hornet_3x_576d0b.pth"),
            "v50_final_label": v50_final.get("final_label"),
            "v50_method_claim_eligible": v50_final.get("method_claim_eligible"),
            "v50_no_go_labels": v50_final.get("no_go_labels") or v50_final.get("no_go"),
            "v50_best_4D_ARI": v50_full.get("best_4D_ARI") or v50_full.get("final_candidate", {}).get("4D_ARI"),
            "v50_best_4D_purity": v50_full.get("best_4D_purity") or v50_full.get("final_candidate", {}).get("4D_purity"),
            "v50_best_4D_completeness": v50_full.get("best_4D_completeness") or v50_full.get("final_candidate", {}).get("4D_completeness"),
            "v50_ap_ran": bool(v50_ap.get("ap_rows") or v50_ap.get("ap_metric_rows")),
            "v50_same_view_hierarchy_available": v50_mask.get("gate", {}).get("same_view_hierarchy_available"),
            "v50_effective_hierarchy_route": v50_mask.get("gate", {}).get("effective_hierarchy_route"),
        }
    )
    rows = list(v50_fact.get("fact_rows", []))
    for key in [
        "stream3d_current_mask_source",
        "stream3d_current_cropformer_scene_count",
        "stream3d_current_cropformer_png_total",
        "stream3d_current_mask_overlap_capable",
        "stream3d_sam2_png_scene_count",
        "stream3d_sam_png_scene_count",
        "stream3d_sam2_checkpoint_at_script_path_exists",
        "cropformer_checkpoint_exists",
        "v50_final_label",
        "v50_method_claim_eligible",
        "v50_same_view_hierarchy_available",
        "v50_effective_hierarchy_route",
    ]:
        rows.append(
            {
                "key": key,
                "value": fact_map.get(key),
                "available": fact_map.get(key) not in (None, ""),
                "required": key not in {"stream3d_sam2_png_scene_count", "stream3d_sam_png_scene_count"},
                "source": "v51_fact_lock",
                "note": "v51-r2 fact lock extension",
            }
        )
    gate = {
        **v50_fact.get("gate", {}),
        "stream3d_current_source_identified": fact_map["stream3d_current_mask_source"] == "Cropformer_flat_png",
        "stream3d_current_source_overlap_capable": False,
        "sam2_local_flat_outputs_available": bool(fact_map["stream3d_sam2_png_scene_count"]),
        "sam2_checkpoint_at_script_path_exists": bool(fact_map["stream3d_sam2_checkpoint_at_script_path_exists"]),
        "phase1_needs_source_discovery": True,
        "phase2_remask_needed_for_overlap_stack": True,
    }
    gate["pass"] = bool(v50_fact.get("gate", {}).get("pass") and gate["stream3d_current_source_identified"])
    return {
        "phase": "v51_r2_fact_lock",
        "created_at": utc_now(),
        "plan": PLAN_PATH,
        "fact_map": fact_map,
        "fact_rows": rows,
        "gate": gate,
        "missing_required": v50_fact.get("missing_required", []),
        "artifact_sources": {
            **v50_fact.get("artifact_sources", {}),
            "v50_final_decision": v50_final_path,
            "v50_full_stage1": v50_full_path,
            "v50_ap_diagnostic": v50_ap_path,
            "v50_mask_source_audit": v50_mask_path,
        },
        "source_constraint_note": "User asked to verify Stream3D masks and avoid silently switching to non-Stream3D/SAM sources.",
    }


def write_v51_fact_lock(output_root: str | Path, payload: dict[str, Any]) -> None:
    out = ROOT / output_root if not Path(output_root).is_absolute() else Path(output_root)
    write_json(out / "fact_lock.json", payload)
    write_csv(out / "fact_lock_rows.csv", payload.get("fact_rows", []))


def build_v51_source_discovery(include_external_npz: bool = True, max_npz_roots: int | None = None) -> dict[str, Any]:
    source_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    for source_id, dir_name, role in [
        ("stream3d_cropformer_flat_png", "output_Cropformer", "stream3d_current_mainline"),
        ("stream3d_sam2_flat_png", "output_SAM2", "stream3d_supported_but_absent"),
        ("stream3d_sam_flat_png", "output_SAM", "legacy_sam_candidate_absent"),
    ]:
        row, samples = _inspect_png_label_source(source_id, dir_name, role)
        source_rows.append(row)
        sample_rows.extend(samples)

    npz_roots_scanned = 0
    if include_external_npz:
        roots = _discover_npz_roots()
        if max_npz_roots is not None:
            roots = roots[: int(max_npz_roots)]
        for root in roots:
            row, samples = _inspect_npz_root(root)
            source_rows.append(row)
            sample_rows.extend(samples)
            npz_roots_scanned += 1

    current_row = next(row for row in source_rows if row["source_id"] == "stream3d_cropformer_flat_png")
    sam_candidate_rows = [
        row
        for row in source_rows
        if row.get("source_family") in {"sam_npz", "sam2_npz", "output_SAM2", "output_SAM"} or "sam" in str(row.get("source_id", "")).lower()
    ]
    main_overlap_rows = [
        row
        for row in source_rows
        if bool(row.get("main_branch_eligible")) and bool(row.get("overlap_capable")) and bool(row.get("preserves_nxhxw_stack"))
    ]
    any_external_overlap = [row for row in source_rows if bool(row.get("overlap_capable")) and bool(row.get("preserves_nxhxw_stack"))]
    gate = {
        "stream3d_current_source_identified": bool(current_row.get("available")),
        "stream3d_current_source_is_cropformer_flat_png": True,
        "stream3d_current_source_overlap_capable": False,
        "sam_or_sam2_overlap_stack_available": bool(main_overlap_rows),
        "external_overlap_stack_available_diagnostic": bool(any_external_overlap),
        "needs_phase2_remask": not bool(main_overlap_rows),
        "can_claim_same_view_hierarchy_from_current_stream3d_source": False,
        "pass": bool(main_overlap_rows),
    }
    summary = {
        "source_row_count": len(source_rows),
        "sample_row_count": len(sample_rows),
        "npz_roots_scanned": npz_roots_scanned,
        "stream3d_current_source": "Cropformer_flat_png",
        "stream3d_current_file_count": current_row.get("file_count"),
        "stream3d_current_scene_count": current_row.get("available_scene_count"),
        "sam_candidate_row_count": len(sam_candidate_rows),
        "sam_or_sam2_overlap_stack_candidate_count": len(main_overlap_rows),
        "external_overlap_stack_diagnostic_count": len(any_external_overlap),
        "selected_source_id": main_overlap_rows[0]["source_id"] if main_overlap_rows else None,
        "selected_source_reason": "SAM/SAM2 overlap-capable stack found" if main_overlap_rows else "No SAM/SAM2 overlap-capable stack found in current workspace",
    }
    return {
        "phase": "v51_r2_source_discovery",
        "created_at": utc_now(),
        "plan": PLAN_PATH,
        "summary": summary,
        "gate": gate,
        "source_discovery_rows": source_rows,
        "source_file_sample_rows": sample_rows,
        "source_constraint_note": "Non-SAM external NPZ sources are diagnostic/backup only and are not selected as the Stream3D main branch.",
    }


def write_v51_source_discovery(output_root: str | Path, payload: dict[str, Any]) -> None:
    out = ROOT / output_root if not Path(output_root).is_absolute() else Path(output_root)
    write_json(out / "source_discovery_summary.json", payload)
    write_csv(out / "source_discovery_rows.csv", payload.get("source_discovery_rows", []))
    write_csv(out / "source_file_sample_rows.csv", payload.get("source_file_sample_rows", []))
