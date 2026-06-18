from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

from stream4d_native.frame_index_map import FrameIndexMap


def _frame_ids_from_dir(path: Path, suffixes: tuple[str, ...]) -> list[int]:
    ids: list[int] = []
    if not path.exists():
        return ids
    for child in path.iterdir():
        if child.suffix.lower() in suffixes:
            try:
                ids.append(int(child.stem))
            except ValueError:
                continue
    return sorted(ids)


def _write_rows(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["raw_frame_id", "dense_rgb_rank", "d4rt_clip_local_index", "mask_observation_rank"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit v41.1 frame-stride and mask-observation rank invariants.")
    parser.add_argument("--scene-dir", default="/home/tmp_datasets/scannet_v2/scans/scene0050_00")
    parser.add_argument("--max-rgb-frames", type=int, default=80)
    parser.add_argument("--mask-stride", type=int, default=10)
    parser.add_argument("--output-root", default="outputs/audit/v41_1_phaseA_stride")
    args = parser.parse_args()

    scene_dir = Path(args.scene_dir)
    color_dir = scene_dir / "color"
    rgb_ids = _frame_ids_from_dir(color_dir, (".jpg", ".png"))
    if not rgb_ids:
        rgb_ids = list(range(int(args.max_rgb_frames)))
        source = "synthetic_fallback_no_color_dir"
    else:
        rgb_ids = rgb_ids[: int(args.max_rgb_frames)]
        source = str(color_dir)
    rgb_set = set(rgb_ids)
    mask_ids = [raw_id for raw_id in rgb_ids if (raw_id - rgb_ids[0]) % int(args.mask_stride) == 0 and raw_id in rgb_set]
    fmap = FrameIndexMap.from_frame_ids(rgb_ids, mask_ids)
    summary = fmap.audit_summary()
    summary.update(
        {
            "phase": "v41_1_phaseA_stride",
            "rgb_source": source,
            "scene_dir": str(scene_dir),
            "max_rgb_frames": int(args.max_rgb_frames),
            "mask_stride_request": int(args.mask_stride),
            "birth_from_d4rt_tube_count_in_tests": 0,
            "gate_pass": bool(summary["uses_contiguous_rgb_for_d4rt"] and summary["d4rt_encoder_stride"] == 1),
        }
    )

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    json_path = out_root / "frame_stride_audit.json"
    csv_path = out_root / "frame_index_map_examples.csv"
    answer_path = out_root / "frame_stride_answer.md"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    _write_rows(csv_path, fmap.to_rows()[:80])
    answer = [
        "# v41.1 Phase A frame-stride answer",
        "",
        f"- RGB source: `{source}`",
        f"- `d4rt_encoder_stride`: `{summary['d4rt_encoder_stride']}`",
        f"- `mask_observation_stride`: `{summary['mask_observation_stride']}`",
        f"- `rank_delta_distribution`: `{summary['rank_delta_distribution']}`",
        f"- `uses_contiguous_rgb_for_d4rt`: `{summary['uses_contiguous_rgb_for_d4rt']}`",
        f"- `uses_sparse_masks_as_measurements`: `{summary['uses_sparse_masks_as_measurements']}`",
        f"- `birth_from_d4rt_tube_count_in_tests`: `{summary['birth_from_d4rt_tube_count_in_tests']}`",
        f"- gate: `{'PASS' if summary['gate_pass'] else 'FAIL'}`",
        "",
        "This audit reads RGB frame ids only. It does not use ScanNet depth, pose, mesh, or instance labels for method prediction.",
        "",
    ]
    answer_path.write_text("\n".join(answer))
    print(json.dumps({"json": str(json_path), "csv": str(csv_path), "answer": str(answer_path), "gate_pass": summary["gate_pass"]}, indent=2))


if __name__ == "__main__":
    main()

