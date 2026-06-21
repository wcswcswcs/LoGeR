from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Any

from stream4d_native.v47_common import ROOT, utc_now, write_csv, write_json
from stream4d_native.v51_mask_bank_filter import filter_mask_bank_by_containment
from stream4d_native.v51_overlap_mask_bank import build_v51_overlap_mask_bank, write_v51_overlap_mask_bank
from stream4d_native.v51_remask_runner import run_sam2_remask


PLAN_PATH = "docs/stream4d_v51_r2_mosaic_remask_lift_codex_plan.md"


def _parse_csv(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def _rel(path: str | Path) -> str:
    path_obj = Path(path)
    if not path_obj.is_absolute():
        path_obj = ROOT / path_obj
    try:
        return str(path_obj.relative_to(ROOT))
    except ValueError:
        return str(path_obj)


def _empty_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()


def _scene_row(
    scene: str,
    remask: dict[str, Any],
    filtered: dict[str, Any],
    bank: dict[str, Any],
    audit_scene_root: Path,
) -> dict[str, Any]:
    sample_rows = bank.get("sample_rows", [])
    pair_count = sum(int(row.get("pair_count_evaluated") or 0) for row in sample_rows)
    summary = bank.get("summary", {})
    gate = bank.get("gate", {})
    return {
        "scene": scene,
        "frame_count": int(summary.get("frame_count") or 0),
        "raw_mask_count": int(remask.get("total_kept_masks") or 0),
        "filtered_mask_count": int(summary.get("mask_count") or 0),
        "mean_masks_per_frame": float(summary.get("mean_masks_per_frame") or 0.0),
        "overlap_pair_count": int(summary.get("overlap_pair_count") or 0),
        "containment_pair_count": int(summary.get("containment_pair_count") or 0),
        "pair_count_evaluated": int(pair_count),
        "containment_pair_ratio": float(summary.get("containment_pair_ratio") or 0.0),
        "whole_candidate_count": int(summary.get("whole_candidate_count") or 0),
        "whole_candidate_ratio": float(summary.get("whole_candidate_ratio") or 0.0),
        "gate_pass": bool(gate.get("pass")),
        "mean_masks_per_frame_pass": bool(gate.get("mean_masks_per_frame_pass")),
        "containment_pass": bool(gate.get("containment_pass")),
        "whole_candidate_pass": bool(gate.get("whole_candidate_pass")),
        "preserves_nxhxw_stack": bool(summary.get("preserves_nxhxw_stack")),
        "uses_gt_for_prediction": bool(summary.get("uses_gt_for_prediction")),
        "remask_runtime_sec": float(remask.get("runtime_sec") or 0.0),
        "remask_summary": _rel(Path(remask["output_root"]) / scene / "sam2_remask_prepare_summary.json"),
        "filter_summary": _rel(Path(filtered["output_root"]) / "mask_bank_filter_summary.json"),
        "bank_summary": _rel(audit_scene_root / "overlap_mask_bank_summary.json"),
    }


def run_probe5_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    scenes = _parse_csv(args.scenes)
    raw_root = ROOT / args.raw_output_root
    filtered_root = ROOT / args.filtered_output_root
    audit_root = ROOT / args.audit_output_root
    audit_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for scene in scenes:
        print({"stage": "remask", "scene": scene, "frames": args.frame_ids}, flush=True)
        remask = run_sam2_remask(
            scene=scene,
            frame_ids=args.frame_ids,
            output_root=raw_root,
            sam2_root=args.sam2_root,
            checkpoint=args.checkpoint,
            model_cfg=args.model_cfg,
            device=args.device,
            points_per_side=args.points_per_side,
            pred_iou_thresh=args.pred_iou_thresh,
            stability_score_thresh=args.stability_score_thresh,
            box_nms_thresh=args.box_nms_thresh,
            crop_n_layers=args.crop_n_layers,
            crop_nms_thresh=args.crop_nms_thresh,
            crop_overlap_ratio=args.crop_overlap_ratio,
            crop_n_points_downscale_factor=args.crop_n_points_downscale_factor,
            min_mask_region_area=args.min_mask_region_area,
            min_area=args.min_area,
        )
        write_json(raw_root / scene / "sam2_remask_prepare_summary.json", remask)
        _empty_cuda_cache()
        raw_scene_root = raw_root / scene
        filtered_scene_root = filtered_root / scene
        print({"stage": "filter", "scene": scene, "input": _rel(raw_scene_root)}, flush=True)
        filtered = filter_mask_bank_by_containment(
            input_root=raw_scene_root,
            output_root=filtered_scene_root,
            contain_threshold=args.contain_threshold,
            area_ratio_threshold=args.area_ratio_threshold,
            min_masks_per_frame=args.min_masks_per_frame,
        )
        print({"stage": "bank", "scene": scene, "input": _rel(filtered_scene_root)}, flush=True)
        bank = build_v51_overlap_mask_bank(filtered_scene_root, max_files=args.max_files)
        audit_scene_root = audit_root / scene
        write_v51_overlap_mask_bank(audit_scene_root, bank)
        row = _scene_row(scene, remask, filtered, bank, audit_scene_root)
        rows.append(row)
        write_json(audit_root / "probe5_source_gate_partial.json", _aggregate(rows, args, partial=True))
        write_csv(audit_root / "probe5_source_gate_partial_rows.csv", rows)
        print({"stage": "scene_done", "scene": scene, "gate_pass": row["gate_pass"]}, flush=True)
    payload = _aggregate(rows, args, partial=False)
    write_json(audit_root / "probe5_source_gate_summary.json", payload)
    write_csv(audit_root / "probe5_source_gate_rows.csv", rows)
    print(
        {
            "summary": _rel(audit_root / "probe5_source_gate_summary.json"),
            "gate": payload["gate"],
            "total": payload["summary"],
        },
        flush=True,
    )
    return payload


def _aggregate(rows: list[dict[str, Any]], args: argparse.Namespace, partial: bool) -> dict[str, Any]:
    frame_count = sum(int(row["frame_count"]) for row in rows)
    mask_count = sum(int(row["filtered_mask_count"]) for row in rows)
    pair_count = sum(int(row["pair_count_evaluated"]) for row in rows)
    containment = sum(int(row["containment_pair_count"]) for row in rows)
    whole = sum(int(row["whole_candidate_count"]) for row in rows)
    summary = {
        "scene_count": len(rows),
        "frame_count": frame_count,
        "raw_mask_count": sum(int(row["raw_mask_count"]) for row in rows),
        "filtered_mask_count": mask_count,
        "mean_masks_per_frame": mask_count / max(frame_count, 1),
        "overlap_pair_count": sum(int(row["overlap_pair_count"]) for row in rows),
        "containment_pair_count": containment,
        "pair_count_evaluated": pair_count,
        "containment_pair_ratio": containment / max(pair_count, 1),
        "whole_candidate_count": whole,
        "whole_candidate_ratio": whole / max(mask_count, 1),
        "remask_runtime_sec": sum(float(row["remask_runtime_sec"]) for row in rows),
        "all_scene_gate_pass": all(bool(row["gate_pass"]) for row in rows) if rows else False,
        "failed_scenes": [row["scene"] for row in rows if not bool(row["gate_pass"])],
    }
    gate = {
        "probe5_status": (len(rows) == len(_parse_csv(args.scenes))) and not partial,
        "mean_masks_per_frame_pass": summary["mean_masks_per_frame"] >= 10.0,
        "containment_pass": containment >= 200 or summary["containment_pair_ratio"] >= 0.02,
        "whole_candidate_pass": whole >= 0.20 * max(mask_count, 1),
        "preserves_nxhxw_stack": all(bool(row["preserves_nxhxw_stack"]) for row in rows) if rows else False,
        "uses_gt_for_prediction": any(bool(row["uses_gt_for_prediction"]) for row in rows),
        "all_scene_gate_pass": summary["all_scene_gate_pass"],
    }
    gate["pass"] = bool(
        gate["probe5_status"]
        and gate["mean_masks_per_frame_pass"]
        and gate["containment_pass"]
        and gate["whole_candidate_pass"]
        and gate["preserves_nxhxw_stack"]
        and not gate["uses_gt_for_prediction"]
    )
    return {
        "phase": "v51_r2_probe5_remask_source_gate",
        "created_at": utc_now(),
        "plan": PLAN_PATH,
        "partial": partial,
        "source": "sam2.1_tiny_automatic_mask_generator",
        "raw_output_root": _rel(args.raw_output_root),
        "filtered_output_root": _rel(args.filtered_output_root),
        "audit_output_root": _rel(args.audit_output_root),
        "scenes": _parse_csv(args.scenes),
        "frame_ids": [int(x) for x in _parse_csv(args.frame_ids)],
        "aligned_frame_manifest_root": args.aligned_frame_manifest_root,
        "sam2_root": str(Path(args.sam2_root).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "model_cfg": args.model_cfg,
        "device": args.device,
        "sam2_params": {
            "points_per_side": args.points_per_side,
            "pred_iou_thresh": args.pred_iou_thresh,
            "stability_score_thresh": args.stability_score_thresh,
            "box_nms_thresh": args.box_nms_thresh,
            "crop_n_layers": args.crop_n_layers,
            "crop_nms_thresh": args.crop_nms_thresh,
            "crop_overlap_ratio": args.crop_overlap_ratio,
            "crop_n_points_downscale_factor": args.crop_n_points_downscale_factor,
            "min_mask_region_area": args.min_mask_region_area,
            "min_area": args.min_area,
        },
        "filter_params": {
            "contain_threshold": args.contain_threshold,
            "area_ratio_threshold": args.area_ratio_threshold,
            "min_masks_per_frame": args.min_masks_per_frame,
        },
        "summary": summary,
        "gate": gate,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v51-r2 SAM2 remask, containment filter, and OverlapMaskBank on probe5.")
    parser.add_argument("--scenes", default="scene0011_00,scene0030_00,scene0050_00,scene0081_01,scene0591_00")
    parser.add_argument("--frame-ids", default="0,10,20,30")
    parser.add_argument("--aligned-frame-manifest-root", default="outputs/stream4d_v5_cache_96f_probe5")
    parser.add_argument("--raw-output-root", default="outputs/remask/v51_r2/sam2_tiny_probe5_4f_p64_crop1_relaxed")
    parser.add_argument("--filtered-output-root", default="outputs/remask/v51_r2/sam2_tiny_probe5_4f_p64_crop1_containment_filtered")
    parser.add_argument("--audit-output-root", default="outputs/audit/v51_r2_probe5_sam2_tiny_4f_p64_crop1_containment_filtered")
    parser.add_argument("--sam2-root", default="../Grounded-SAM-2")
    parser.add_argument("--checkpoint", default="../Grounded-SAM-2/checkpoints/sam2.1_hiera_tiny.pt")
    parser.add_argument("--model-cfg", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--points-per-side", type=int, default=64)
    parser.add_argument("--pred-iou-thresh", type=float, default=0.80)
    parser.add_argument("--stability-score-thresh", type=float, default=0.88)
    parser.add_argument("--box-nms-thresh", type=float, default=0.95)
    parser.add_argument("--crop-n-layers", type=int, default=1)
    parser.add_argument("--crop-nms-thresh", type=float, default=0.95)
    parser.add_argument("--crop-overlap-ratio", type=float, default=0.50)
    parser.add_argument("--crop-n-points-downscale-factor", type=int, default=1)
    parser.add_argument("--min-mask-region-area", type=int, default=0)
    parser.add_argument("--min-area", type=int, default=50)
    parser.add_argument("--contain-threshold", type=float, default=0.85)
    parser.add_argument("--area-ratio-threshold", type=float, default=1.30)
    parser.add_argument("--min-masks-per-frame", type=int, default=10)
    parser.add_argument("--max-files", type=int, default=32)
    args = parser.parse_args()
    run_probe5_pipeline(args)


if __name__ == "__main__":
    main()
