from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from stream4d.measurement_bank import MeasurementBank, json_safe, read_seq_list
from stream4d.video_masklet import (
    VideoMaskletParams,
    build_video_masklet_bank,
    summarize_original_sparse,
)


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    numeric_keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, (int, float, np.generic)) and value is not None and not isinstance(value, bool)
        }
    )
    by_mode: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_mode.setdefault(str(row["mode"]), []).append(row)
    return {
        "algorithm": "v13_video_masklet_measurement_density",
        "uses_gt": False,
        "is_method_result": False,
        "num_rows": int(len(rows)),
        "modes": {
            mode: {
                "num_scenes": int(len(mode_rows)),
                "numeric_mean": {
                    key: float(np.mean([float(row[key]) for row in mode_rows if row.get(key) is not None]))
                    for key in numeric_keys
                    if any(row.get(key) is not None for row in mode_rows)
                },
            }
            for mode, mode_rows in sorted(by_mode.items())
        },
    }


def _write_bundle(prefix: Path, rows: list[dict[str, Any]], aggregate: dict[str, Any]) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    payload = {"aggregate": aggregate, "rows": rows}
    prefix.with_suffix(".json").write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")
    if rows:
        fieldnames = sorted({key for row in rows for key in row.keys()})
        with prefix.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in fieldnames})
    lines = [
        "# Stream4D v13 Video Masklet Density",
        "",
        "This diagnostic does not read GT and does not report AP. GT-linked candidate/oracle diagnostics are reported in Phase B.",
        "",
        "## Aggregate",
        "",
        "| mode | semantic frames/surfel | obs/surfel | unobserved | masklets | frames/birth | agreement | neg outside |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, payload_mode in sorted(aggregate["modes"].items()):
        means = payload_mode["numeric_mean"]
        lines.append(
            "| "
            + " | ".join(
                [
                    mode,
                    f"{float(means.get('num_effective_semantic_frames_per_surfel', 0.0)):.4f}",
                    f"{float(means.get('positive_observations_per_surfel', 0.0)):.4f}",
                    f"{float(means.get('unobserved_surfel_ratio', 0.0)):.4f}",
                    f"{float(means.get('masklet_count', 0.0)):.2f}",
                    f"{float(means.get('masklet_frames_per_object_birth', 0.0)):.4f}",
                    f"{float(means.get('available_mask_agreement_iou', 0.0)):.4f}",
                    f"{float(means.get('negative_visible_outside_ratio', 0.0)):.4f}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Scenes",
            "",
            "| scene | mode | semantic frames/surfel | obs/surfel | unobserved | masklets | frames/birth | compactness | area growth | agreement | neg outside |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["scene"]),
                    str(row["mode"]),
                    f"{float(row.get('num_effective_semantic_frames_per_surfel', 0.0)):.4f}",
                    f"{float(row.get('positive_observations_per_surfel', 0.0)):.4f}",
                    f"{float(row.get('unobserved_surfel_ratio', 0.0)):.4f}",
                    str(row.get("masklet_count", 0)),
                    f"{float(row.get('masklet_frames_per_object_birth', 0.0)):.4f}",
                    f"{float(row.get('masklet_compactness', 0.0)):.4f}",
                    f"{float(row.get('masklet_area_growth_ratio', 0.0)):.4f}",
                    f"{float(row.get('available_mask_agreement_iou', 0.0)):.4f}",
                    f"{float(row.get('negative_visible_outside_ratio', 0.0)):.4f}",
                ]
            )
            + " |"
        )
    prefix.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank-root", default="outputs/v12_measurement_bank")
    parser.add_argument("--seq-list", required=True)
    parser.add_argument("--output-root", default="outputs/v13_masklet_measurements")
    parser.add_argument("--output-prefix", default="outputs/audit/v13_masklet_density/masklet_density_probe5")
    parser.add_argument("--modes", default="C1,C2,C3")
    parser.add_argument("--min-birth-surfels", type=int, default=12)
    parser.add_argument("--min-frame-surfels", type=int, default=6)
    parser.add_argument("--boundary-safe-px", type=float, default=3.0)
    parser.add_argument("--c2-min-available-mask-agreement", type=float, default=0.30)
    parser.add_argument("--c3-min-available-mask-agreement", type=float, default=0.40)
    parser.add_argument("--c3-min-boundary-safe-ratio", type=float, default=0.35)
    parser.add_argument("--c3-min-confidence", type=float, default=0.45)
    parser.add_argument("--c3-max-negative-visible-outside-ratio", type=float, default=0.75)
    args = parser.parse_args()

    params = VideoMaskletParams(
        min_birth_surfels=int(args.min_birth_surfels),
        min_frame_surfels=int(args.min_frame_surfels),
        boundary_safe_px=float(args.boundary_safe_px),
        c2_min_available_mask_agreement=float(args.c2_min_available_mask_agreement),
        c3_min_available_mask_agreement=float(args.c3_min_available_mask_agreement),
        c3_min_boundary_safe_ratio=float(args.c3_min_boundary_safe_ratio),
        c3_min_confidence=float(args.c3_min_confidence),
        c3_max_negative_visible_outside_ratio=float(args.c3_max_negative_visible_outside_ratio),
    )
    rows: list[dict[str, Any]] = []
    modes = [item.strip() for item in str(args.modes).split(",") if item.strip()]
    for scene in read_seq_list(Path(args.seq_list)):
        bank_path = Path(args.bank_root) / scene / "measurement_bank.npz"
        bank = MeasurementBank.load(bank_path)
        rows.append({**summarize_original_sparse(bank), "bank_path": str(bank_path), "masklet_path": ""})
        for mode in modes:
            masklets, summary = build_video_masklet_bank(bank, mode=mode, params=params)
            out_path = Path(args.output_root) / mode / scene / "masklets.npz"
            masklets.save(out_path)
            rows.append({**summary, "bank_path": str(bank_path), "masklet_path": str(out_path)})
    aggregate = _aggregate(rows)
    _write_bundle(Path(args.output_prefix), rows, aggregate)
    print(json.dumps(json_safe(aggregate), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
