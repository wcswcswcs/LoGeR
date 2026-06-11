from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from stream4d.measurement_bank import (
    build_scene_measurement_bank,
    json_safe,
    read_seq_list,
    write_summary_bundle,
)


def _write_overlay(bank_path: Path, output_dir: Path) -> str:
    from stream4d.measurement_bank import MeasurementBank, load_rgb

    bank = MeasurementBank.load(bank_path)
    available = np.flatnonzero(bank.mask_frame_available)
    if available.size == 0:
        return ""
    local_idx = int(available[0])
    frame_id = int(bank.frame_ids[local_idx])
    scene_root = Path(bank.meta["scannet_root"]) / bank.scene
    rgb = load_rgb(scene_root / "color" / f"{frame_id}.jpg")
    if rgb is None:
        return ""
    image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    h, w = image.shape[:2]
    uv = bank.uv_pred[local_idx]
    x = np.rint(uv[:, 0] * float(max(w - 1, 1))).astype(np.int64)
    y = np.rint(uv[:, 1] * float(max(h - 1, 1))).astype(np.int64)
    in_bounds = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    positive = bank.positive_observation[local_idx] & in_bounds
    negative = bank.negative_observation[local_idx] & in_bounds
    visible = bank.visible_ok[local_idx] & in_bounds
    for mask, color in ((visible, (160, 160, 160)), (negative, (0, 0, 255)), (positive, (0, 255, 0))):
        idx = np.flatnonzero(mask)
        if idx.size > 2000:
            idx = idx[np.linspace(0, idx.size - 1, 2000, dtype=np.int64)]
        for i in idx.tolist():
            cv2.circle(image, (int(x[i]), int(y[i])), 1, color, -1, lineType=cv2.LINE_AA)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{bank.scene}_measurement_bank_overlay_frame_{frame_id}.png"
    cv2.imwrite(str(path), image)
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug-root", required=True)
    parser.add_argument("--seq-list", required=True)
    parser.add_argument("--scannet-root", default="data/scannet/processed")
    parser.add_argument("--backbone", default="Cropformer")
    parser.add_argument("--output-root", default="outputs/v12_measurement_bank")
    parser.add_argument("--audit-prefix", default="outputs/audit/v12_measurement_bank/measurement_bank_probe5")
    parser.add_argument("--min-visibility", type=float, default=0.5)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--boundary-safe-px", type=float, default=3.0)
    args = parser.parse_args()

    debug_root = Path(args.debug_root)
    seq_list = Path(args.seq_list)
    scannet_root = Path(args.scannet_root)
    output_root = Path(args.output_root)
    rows = []
    overlay_paths = []
    for scene in read_seq_list(seq_list):
        bank, summary = build_scene_measurement_bank(
            debug_root=debug_root,
            scannet_root=scannet_root,
            scene=scene,
            backbone=args.backbone,
            min_visibility=float(args.min_visibility),
            min_confidence=float(args.min_confidence),
            boundary_safe_px=float(args.boundary_safe_px),
        )
        bank_path = output_root / scene / "measurement_bank.npz"
        bank.save(bank_path)
        summary["bank_path"] = str(bank_path)
        summary_path = output_root / scene / "measurement_bank_summary.json"
        summary_path.write_text(json.dumps(json_safe(summary), indent=2, sort_keys=True), encoding="utf-8")
        summary["summary_path"] = str(summary_path)
        overlay_path = _write_overlay(bank_path, Path(args.audit_prefix).parent / "visuals")
        if overlay_path:
            overlay_paths.append(overlay_path)
            summary["overlay_path"] = overlay_path
        rows.append(summary)
    aggregate = write_summary_bundle(Path(args.audit_prefix), rows)
    print(json.dumps(json_safe({"aggregate": aggregate, "overlays": overlay_paths}), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
