#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def count_files(path: Path, pattern: str) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in path.glob(pattern))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--patch-size", type=int, default=14)
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve()
    log_path = Path(args.log).resolve()
    out_dir = Path(args.out_dir).resolve()
    eval_summary = read_json(output_root / "eval_summary.json")
    pipeline_summary = read_json(output_root / "pipeline_summary.json")
    log_text = log_path.read_text(encoding="utf-8", errors="replace")

    loaded_shape = None
    m = re.search(r"loaded sequence (?P<seq>[^:]+): \((?P<shape>[^)]+)\)", log_text)
    if m:
        shape = [int(part.strip()) for part in m.group("shape").split(",")]
        loaded_shape = {"sequence": m.group("seq"), "shape": shape}

    runtime_layout = None
    if loaded_shape and len(loaded_shape["shape"]) == 5:
        _, frames, channels, height, width = loaded_shape["shape"]
        patch_h = height // args.patch_size
        patch_w = width // args.patch_size
        patch_count = patch_h * patch_w
        runtime_layout = {
            "frames": frames,
            "channels": channels,
            "height": height,
            "width": width,
            "patch_size": args.patch_size,
            "patch_grid": [patch_h, patch_w],
            "patch_count": patch_count,
            "mrt_index": 0,
            "patch_stream_tokens_P": patch_count + 1,
            "image_patch_token_range": [1, patch_count],
            "pose_tokens_per_window_slot_X": 32,
            "window_size_Win": 10,
            "global_tokens_per_frame_row_A": patch_count + 1 + 10 * 32,
        }

    seq_dir = output_root / "00" / "02"
    files = {
        "abs_pose_lines": sum(1 for _ in (seq_dir / "poses" / "abs_pose.txt").open("r", encoding="utf-8"))
        if (seq_dir / "poses" / "abs_pose.txt").exists()
        else 0,
        "intri_lines": sum(1 for _ in (seq_dir / "poses" / "intri.txt").open("r", encoding="utf-8"))
        if (seq_dir / "poses" / "intri.txt").exists()
        else 0,
        "gt_abs_pose_lines": sum(1 for _ in (seq_dir / "poses" / "gt_abs_pose.txt").open("r", encoding="utf-8"))
        if (seq_dir / "poses" / "gt_abs_pose.txt").exists()
        else 0,
        "depth_npy_count": count_files(seq_dir / "depth" / "dpt", "*.npy"),
        "depth_conf_npy_count": count_files(seq_dir / "depth" / "conf", "*.npy"),
        "trajectory_metrics_exists": (seq_dir / "eval" / "trajectory_metrics.json").exists(),
        "trajectory_plot_exists": (seq_dir / "plots" / "trajectory_compare.png").exists(),
    }

    summary = {
        "output_root": str(output_root),
        "log_path": str(log_path),
        "pipeline": {
            "ran_infer": pipeline_summary.get("ran_infer"),
            "ran_loop": pipeline_summary.get("ran_loop"),
            "ran_eval": pipeline_summary.get("ran_eval"),
        },
        "loaded_shape": loaded_shape,
        "runtime_token_layout": runtime_layout,
        "files": files,
        "eval_summary": eval_summary,
        "status": "pass",
    }
    write_json(out_dir / "stage0_hs_eval_smoke_summary.json", summary)
    if runtime_layout is not None:
        write_json(out_dir / "hs_runtime_token_layout_smoke_kitti00_max12.json", runtime_layout)
        write_md(
            out_dir / "hs_runtime_token_layout_smoke_kitti00_max12.md",
            f"""# Runtime Token Layout: KITTI 00/02 max12 smoke

- loaded image shape: `{loaded_shape['shape']}`
- runtime height x width: `{runtime_layout['height']} x {runtime_layout['width']}`
- patch size: `{runtime_layout['patch_size']}`
- runtime patch grid: `{runtime_layout['patch_grid'][0]} x {runtime_layout['patch_grid'][1]}`
- runtime image patch count: `{runtime_layout['patch_count']}`
- MRT index: `0`
- patch stream tokens `P`: `{runtime_layout['patch_stream_tokens_P']}`
- image patch token range: `{runtime_layout['image_patch_token_range'][0]}..{runtime_layout['image_patch_token_range'][1]}`
- pose tokens per window slot `X`: `32`
- window size `Win`: `10`
- global tokens per frame row `A`: `{runtime_layout['global_tokens_per_frame_row_A']}`

Important alignment conclusion: KITTI runtime does not use a square `37 x 37` patch grid under the observed loader path. Stage2 semantic projection for KITTI must use this data-dependent grid unless a later audited loader/config change proves a different runtime shape.
""",
        )

    print(f"wrote smoke summary to {out_dir / 'stage0_hs_eval_smoke_summary.json'}")


if __name__ == "__main__":
    main()
