from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from stream4d.export_scannet import ScanNetExporter
from stream4d.measurement_bank import MeasurementBank, json_safe, read_seq_list
from stream4d.scannet_stream import ScanNetStream
from stream4d.video_masklet import VideoMaskletBank
from tools.prediction_manifest import build_prediction_manifest, write_prediction_manifest


def _dominant_vote(bank: MeasurementBank, masklets: VideoMaskletBank, row_idx: int) -> tuple[int, int, float] | None:
    frame_id = int(masklets.frame_id[row_idx])
    local = np.flatnonzero(bank.frame_ids == frame_id)
    if local.size == 0:
        return None
    frame_idx = int(local[0])
    if not bool(bank.mask_frame_available[frame_idx]):
        return None
    surfels = masklets.surfels_for_row(row_idx)
    ids = bank.target_mask_id[frame_idx, surfels]
    ids = ids[ids > 0]
    if ids.size == 0:
        return None
    mask_id, count = Counter(int(v) for v in ids.tolist()).most_common(1)[0]
    return frame_id, int(mask_id), float(count) * float(masklets.confidence[row_idx])


def _records_for_scene(
    bank: MeasurementBank,
    masklets: VideoMaskletBank,
    *,
    min_rows_per_candidate: int,
    min_core_surfels: int,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    rejected = 0
    for key, row_indices in sorted(masklets.rows_by_birth().items()):
        if len(row_indices) < int(min_rows_per_candidate):
            rejected += 1
            continue
        counts: Counter[int] = Counter()
        votes: dict[tuple[int, int], float] = {}
        for row_idx in row_indices:
            surfels = masklets.surfels_for_row(row_idx)
            counts.update(int(v) for v in surfels.tolist())
            vote = _dominant_vote(bank, masklets, row_idx)
            if vote is not None:
                frame_id, mask_id, score = vote
                vote_key = (frame_id, mask_id)
                votes[vote_key] = max(votes.get(vote_key, 0.0), float(score))
        if not counts:
            rejected += 1
            continue
        core = np.asarray([idx for idx, count in counts.items() if count >= 2], dtype=np.int64)
        if core.size == 0:
            core = np.asarray(sorted(counts.keys()), dtype=np.int64)
        fringe = np.asarray([idx for idx, count in counts.items() if count == 1], dtype=np.int64)
        if core.shape[0] < int(min_core_surfels):
            rejected += 1
            continue
        birth_frame, birth_mask = key
        votes[(int(birth_frame), int(birth_mask))] = max(votes.get((int(birth_frame), int(birth_mask)), 0.0), float(core.shape[0]))
        object_id = len(records)
        records[object_id] = {
            "mask_list": [(int(frame), int(mask), float(score)) for (frame, mask), score in sorted(votes.items())],
            "carrier_ids": core,
            "core_surfels": core,
            "fringe_surfels": fringe,
            "unknown_surfels": np.empty((0,), dtype=np.int64),
            "reject_surfels": np.empty((0,), dtype=np.int64),
            "v13_masklet_candidate": {
                "object_id": int(object_id),
                "birth_frame": int(birth_frame),
                "birth_mask_id": int(birth_mask),
                "num_masklet_rows": int(len(row_indices)),
                "num_core_surfels": int(core.shape[0]),
                "num_fringe_surfels": int(fringe.shape[0]),
            },
        }
    diag = {
        "num_masklet_rows": int(masklets.masklet_id.shape[0]),
        "num_candidate_slots": int(len(records)),
        "num_rejected_candidates": int(rejected),
        "mean_mask_observations_per_candidate": float(np.mean([len(v["mask_list"]) for v in records.values()])) if records else 0.0,
        "mean_core_surfels_per_candidate": float(np.mean([np.asarray(v["core_surfels"]).shape[0] for v in records.values()])) if records else 0.0,
    }
    return records, diag


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    numeric_keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, (int, float, np.generic)) and value is not None and not isinstance(value, bool)
        }
    )
    return {
        "algorithm": "v13_masklet_candidate",
        "num_scenes": int(len(rows)),
        "numeric_mean": {
            key: float(np.mean([float(row[key]) for row in rows if row.get(key) is not None]))
            for key in numeric_keys
            if any(row.get(key) is not None for row in rows)
        },
        "scenes": rows,
    }


def _write_summary(args: argparse.Namespace, rows: list[dict[str, Any]], aggregate: dict[str, Any]) -> None:
    out_dir = Path(args.summary_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{args.output_config}_summary.json").write_text(
        json.dumps(json_safe(aggregate), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if rows:
        fieldnames = sorted({key for row in rows for key in row.keys()})
        with (out_dir / f"{args.output_config}_summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank-root", default="outputs/v12_measurement_bank")
    parser.add_argument("--masklet-root", default="outputs/v13_masklet_measurements")
    parser.add_argument("--masklet-mode", default="C3")
    parser.add_argument("--seq-list", required=True)
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--backbone", default="Cropformer")
    parser.add_argument("--summary-root", default="outputs/v13_masklet_candidates")
    parser.add_argument("--min-rows-per-candidate", type=int, default=2)
    parser.add_argument("--min-core-surfels", type=int, default=8)
    parser.add_argument("--min-export-points-per-object", type=int, default=60)
    parser.add_argument("--export-nn-radius", type=float, default=0.05)
    parser.add_argument("--export-core-nn-radius", type=float, default=0.05)
    parser.add_argument("--export-fringe-nn-radius", type=float, default=0.05)
    parser.add_argument("--export-fringe-radius", type=float, default=0.05)
    parser.add_argument("--export-fringe-max-ratio", type=float, default=0.35)
    parser.add_argument("--export-score-mode", choices=["one", "area", "reliability", "observations"], default="reliability")
    parser.add_argument("--export-enable-wta", action="store_true")
    parser.add_argument("--diagnostic-candidate-only", action="store_true")
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for scene in read_seq_list(Path(args.seq_list)):
        bank = MeasurementBank.load(Path(args.bank_root) / scene / "measurement_bank.npz")
        masklets = VideoMaskletBank.load(Path(args.masklet_root) / args.masklet_mode / scene / "masklets.npz")
        object_dict, diag = _records_for_scene(
            bank,
            masklets,
            min_rows_per_candidate=int(args.min_rows_per_candidate),
            min_core_surfels=int(args.min_core_surfels),
        )
        stream = ScanNetStream(seq_name=scene, backbone=args.backbone)
        errors = stream.validate(require_masks=True)
        if errors:
            raise RuntimeError("; ".join(errors))
        exporter = ScanNetExporter(
            stream,
            output_config=args.output_config,
            export_support_mode="posterior_support",
            export_nn_radius=float(args.export_nn_radius),
            export_core_nn_radius=float(args.export_core_nn_radius),
            export_fringe_nn_radius=float(args.export_fringe_nn_radius),
            export_fringe_radius=float(args.export_fringe_radius),
            export_fringe_max_ratio=float(args.export_fringe_max_ratio),
            export_min_points_per_object=int(args.min_export_points_per_object),
            export_score_mode=args.export_score_mode,
            export_enable_wta=bool(args.export_enable_wta),
        )
        export_diag = exporter.export_object_slot_posterior_support(object_dict, bank)
        row = {
            "seq_name": scene,
            "algorithm": "v13_masklet_candidate",
            "masklet_mode": args.masklet_mode,
            "uses_gt": False,
            "is_method_result": not bool(args.diagnostic_candidate_only),
            "is_diagnostic_only": bool(args.diagnostic_candidate_only),
            **diag,
            **export_diag,
        }
        rows.append(row)
    aggregate = _aggregate(rows)
    _write_summary(args, rows, aggregate)
    manifest = build_prediction_manifest(
        root=".",
        output_config=args.output_config,
        is_method_result=not bool(args.diagnostic_candidate_only),
        is_diagnostic_only=bool(args.diagnostic_candidate_only),
        uses_gt=False,
        gt_usage="none",
        source_configs=[str(args.bank_root), str(args.masklet_root)],
        pre_points_policy="recompute",
        support_policy="v13_masklet_candidate:posterior_support",
        notes="v13 unsupervised masklet candidate baseline. GT is not read.",
        extra={
            "algorithm": "v13_masklet_candidate",
            "eval_policy": "own_recompute_paper_style",
            "prediction_config": args.output_config,
            "pre_points_config": args.output_config,
            "support_source": "own",
            "geometry_source": "rgbd_eval_bridge",
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic": False,
            "is_method_result": not bool(args.diagnostic_candidate_only),
            "is_diagnostic_only": bool(args.diagnostic_candidate_only),
            "summary_path": str(Path(args.summary_root) / f"{args.output_config}_summary.json"),
            "masklet_mode": args.masklet_mode,
        },
    )
    write_prediction_manifest(args.output_config, manifest, root=".")
    print(json.dumps(json_safe(aggregate), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
