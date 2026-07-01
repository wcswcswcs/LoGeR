from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent


def _project(path: str | Path) -> Path:
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj
    if path_obj.parts and path_obj.parts[0] == ROOT.name:
        return REPO_ROOT / path_obj
    return ROOT / path_obj


def _rel(path: str | Path) -> str:
    path_obj = _project(path)
    try:
        return str(path_obj.relative_to(ROOT))
    except ValueError:
        try:
            return str(path_obj.relative_to(REPO_ROOT))
        except ValueError:
            return str(path_obj)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _numeric_stems(path: Path, suffix: str) -> list[int]:
    if not path.exists():
        return []
    out: list[int] = []
    for item in path.glob(f"*{suffix}"):
        if item.stem.isdigit():
            out.append(int(item.stem))
    return sorted(set(out))


def _expected_stride_frames(scene: str, stride: int, max_frames: int) -> list[int]:
    color_dir = ROOT / "data/scannet/processed" / scene / "color"
    ids = _numeric_stems(color_dir, ".jpg")
    if not ids:
        raise FileNotFoundError(f"no RGB jpg frames found: {color_dir}")
    id_set = set(ids)
    frames = [frame for frame in range(min(ids), max(ids) + 1, int(stride)) if frame in id_set]
    return frames[: int(max_frames)] if max_frames > 0 else frames


def _prepare_split_inputs(
    *,
    scene: str,
    stride: int,
    output_root: Path,
    gpus: list[str],
    max_frames: int,
) -> tuple[Path, list[dict[str, Any]], list[int]]:
    processed_root = output_root / "scannet_processed"
    frames = _expected_stride_frames(scene, stride, max_frames)
    rows: list[dict[str, Any]] = []
    for split_index, gpu in enumerate(gpus):
        split_scene = f"{scene}_gpu{gpu}"
        split_color_dir = processed_root / split_scene / "color"
        split_color_dir.mkdir(parents=True, exist_ok=True)
        for frame in frames[split_index:: len(gpus)]:
            src = ROOT / "data/scannet/processed" / scene / "color" / f"{int(frame)}.jpg"
            dst = split_color_dir / f"{int(frame)}.jpg"
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            os.symlink(src, dst)
            rows.append(
                {
                    "frame_id": int(frame),
                    "split_index": int(split_index),
                    "gpu": str(gpu),
                    "split_scene": split_scene,
                    "source_rgb": _rel(src),
                    "shadow_rgb": _rel(dst),
                    "source_rgb_sha256": _sha256(src),
                    "source_rgb_bytes": src.stat().st_size,
                }
            )
    _write_csv(output_root / "stride5_cropformer_input_frames.csv", sorted(rows, key=lambda row: int(row["frame_id"])))
    return processed_root, rows, frames


def _cropformer_command(
    *,
    processed_root: Path,
    split_scene: str,
    confidence_threshold: float,
) -> list[str]:
    return [
        sys.executable,
        "third_party/detectron2/projects/CropFormer/demo_cropformer/Cropformer.py",
        "--config-file",
        "third_party/detectron2/projects/CropFormer/configs/entityv2/entity_segmentation/mask2former_hornet_3x.yaml",
        "--root",
        str(processed_root),
        "--image_path_pattern",
        "color/*.jpg",
        "--dataset",
        "scannet",
        "--seq_name_list",
        split_scene,
        "--confidence-threshold",
        str(float(confidence_threshold)),
        "--opts",
        "MODEL.WEIGHTS",
        "third_party/seg_models/Mask2Former_hornet_3x_576d0b.pth",
    ]


def _run_cropformer(
    *,
    output_root: Path,
    processed_root: Path,
    scene: str,
    gpus: list[str],
    confidence_threshold: float,
) -> list[dict[str, Any]]:
    processes: list[tuple[subprocess.Popen[Any], Any, dict[str, Any]]] = []
    for gpu in gpus:
        split_scene = f"{scene}_gpu{gpu}"
        log_path = output_root / f"cropformer_gpu{gpu}.log"
        command = _cropformer_command(
            processed_root=processed_root,
            split_scene=split_scene,
            confidence_threshold=confidence_threshold,
        )
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        handle = log_path.open("w", encoding="utf-8")
        start = time.perf_counter()
        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, text=True)
        row = {
            "gpu": str(gpu),
            "split_scene": split_scene,
            "command": " ".join(command),
            "log_path": _rel(log_path),
            "start_monotonic": float(start),
        }
        processes.append((proc, handle, row))

    rows: list[dict[str, Any]] = []
    for proc, handle, row in processes:
        returncode = proc.wait()
        handle.close()
        end = time.perf_counter()
        row["returncode"] = int(returncode)
        row["elapsed_sec"] = float(end - float(row["start_monotonic"]))
        rows.append(row)
    _write_csv(output_root / "cropformer_process_rows.csv", rows)
    return rows


def _merge_and_hash_masks(
    *,
    output_root: Path,
    processed_root: Path,
    scene: str,
    expected_frames: list[int],
    gpus: list[str],
) -> dict[str, Any]:
    final_mask_dir = processed_root / scene / "output_Cropformer" / "mask"
    final_mask_dir.mkdir(parents=True, exist_ok=True)
    duplicate_frames: list[int] = []
    copied_rows: list[dict[str, Any]] = []
    for gpu in gpus:
        split_scene = f"{scene}_gpu{gpu}"
        split_mask_dir = processed_root / split_scene / "output_Cropformer" / "mask"
        for path in sorted(split_mask_dir.glob("*.png")):
            if not path.stem.isdigit():
                continue
            frame = int(path.stem)
            dst = final_mask_dir / path.name
            if dst.exists():
                duplicate_frames.append(frame)
                continue
            shutil.copy2(path, dst)
            digest = _sha256(dst)
            copied_rows.append(
                {
                    "frame_id": int(frame),
                    "gpu": str(gpu),
                    "source_mask": _rel(path),
                    "final_mask": _rel(dst),
                    "sha256": digest,
                    "bytes": dst.stat().st_size,
                }
            )
    copied_rows.sort(key=lambda row: int(row["frame_id"]))
    _write_csv(output_root / "stride5_cropformer_mask_hashes.csv", copied_rows)
    expected = set(int(frame) for frame in expected_frames)
    present = set(int(row["frame_id"]) for row in copied_rows)
    missing = sorted(expected - present)
    non_stride = sorted(present - expected)
    aggregate = hashlib.sha256()
    for row in copied_rows:
        aggregate.update(f"{int(row['frame_id'])} {row['sha256']}\n".encode("utf-8"))
    return {
        "final_mask_dir": _rel(final_mask_dir),
        "expected_stride_frame_count": int(len(expected_frames)),
        "available_stride_mask_frame_count": int(len(expected & present)),
        "missing_stride_mask_frame_count": int(len(missing)),
        "coverage_ratio": float(len(expected & present) / max(len(expected_frames), 1)),
        "first_missing_stride_frames": missing[:50],
        "last_missing_stride_frames": missing[-20:],
        "non_stride_mask_frame_count": int(len(non_stride)),
        "duplicate_frames_skipped": duplicate_frames[:100],
        "mask_hash_csv": _rel(output_root / "stride5_cropformer_mask_hashes.csv"),
        "present_stride_mask_file_hash_aggregate_sha256": aggregate.hexdigest() if copied_rows else None,
        "gate": {
            "all_expected_stride_frames_have_2d_masks": len(missing) == 0,
            "pass": len(missing) == 0,
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    scene = str(args.scene)
    output_root = _project(args.output_root)
    gpus = [item.strip() for item in str(args.gpus).split(",") if item.strip()]
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU id")
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"output root exists; pass --overwrite: {output_root}")
        audit_root = ROOT / "outputs" / "audit"
        output_root.relative_to(audit_root)
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    processed_root, input_rows, expected_frames = _prepare_split_inputs(
        scene=scene,
        stride=int(args.stride),
        output_root=output_root,
        gpus=gpus,
        max_frames=int(args.max_frames),
    )
    process_rows = _run_cropformer(
        output_root=output_root,
        processed_root=processed_root,
        scene=scene,
        gpus=gpus,
        confidence_threshold=float(args.confidence_threshold),
    )
    merge_summary = _merge_and_hash_masks(
        output_root=output_root,
        processed_root=processed_root,
        scene=scene,
        expected_frames=expected_frames,
        gpus=gpus,
    )
    ok = all(int(row.get("returncode", 1)) == 0 for row in process_rows) and bool(merge_summary["gate"]["pass"])
    summary = {
        "phase": "v65_fresh_cropformer_stride_masks",
        "scene": scene,
        "stride": int(args.stride),
        "gpus": gpus,
        "confidence_threshold": float(args.confidence_threshold),
        "max_frames": int(args.max_frames),
        "cwd": os.getcwd(),
        "python": sys.executable,
        "argv": sys.argv,
        "output_root": _rel(output_root),
        "processed_root": _rel(processed_root),
        "fresh_cache_policy": "shadow stride input + fresh CropFormer run; old data/scannet/processed output_Cropformer/mask is not read or overwritten",
        "input_frame_count": int(len(input_rows)),
        "input_frames_csv": _rel(output_root / "stride5_cropformer_input_frames.csv"),
        "cropformer_process_rows_csv": _rel(output_root / "cropformer_process_rows.csv"),
        "process_rows": process_rows,
        "merge_summary": merge_summary,
        "gate": {
            "all_cropformer_processes_return_zero": all(int(row.get("returncode", 1)) == 0 for row in process_rows),
            "all_expected_stride_frames_have_2d_masks": bool(merge_summary["gate"]["pass"]),
            "pass": bool(ok),
        },
    }
    _write_json(output_root / "fresh_cropformer_stride_mask_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if not ok:
        raise SystemExit(1)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate fresh stride-N CropFormer masks without using the historical cache.")
    parser.add_argument("--scene", default="scene0050_00")
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--output-root", default="outputs/audit/v65_cropformer_stride5_scene0050_fresh")
    parser.add_argument("--gpus", default="6,7")
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
