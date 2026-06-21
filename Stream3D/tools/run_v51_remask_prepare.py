from __future__ import annotations

import argparse

from stream4d_native.v51_remask_runner import run_sam2_remask


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stream4D v51-r2 Phase2 SAM2 remask preparation.")
    parser.add_argument("--source", choices=["sam2"], default="sam2")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--frame-ids", required=True)
    parser.add_argument("--output-root", default="outputs/remask/v51_r2/sam2_tiny_smoke")
    parser.add_argument("--sam2-root", default="../Grounded-SAM-2")
    parser.add_argument("--checkpoint", default="../Grounded-SAM-2/checkpoints/sam2.1_hiera_tiny.pt")
    parser.add_argument("--model-cfg", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--points-per-side", type=int, default=16)
    parser.add_argument("--pred-iou-thresh", type=float, default=0.88)
    parser.add_argument("--stability-score-thresh", type=float, default=0.95)
    parser.add_argument("--box-nms-thresh", type=float, default=0.7)
    parser.add_argument("--crop-n-layers", type=int, default=0)
    parser.add_argument("--crop-nms-thresh", type=float, default=0.7)
    parser.add_argument("--crop-overlap-ratio", type=float, default=0.3413333333333333)
    parser.add_argument("--crop-n-points-downscale-factor", type=int, default=1)
    parser.add_argument("--min-mask-region-area", type=int, default=0)
    parser.add_argument("--min-area", type=int, default=400)
    args = parser.parse_args()
    payload = run_sam2_remask(
        scene=args.scene,
        frame_ids=args.frame_ids,
        output_root=args.output_root,
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
    print(
        {
            "summary": f"{args.output_root}/sam2_remask_prepare_summary.json",
            "scene": payload["scene"],
            "frame_count": payload["frame_count"],
            "total_kept_masks": payload["total_kept_masks"],
            "mean_masks_per_frame": payload["mean_masks_per_frame"],
            "checkpoint_found": payload["checkpoint_found"],
        }
    )


if __name__ == "__main__":
    main()
