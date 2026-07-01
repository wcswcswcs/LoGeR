#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import yaml
from huggingface_hub import snapshot_download


ROOT = Path(__file__).resolve().parents[2]
PHASE1 = ROOT / "Stream3D" / "outputs" / "audit" / "v98_phase1_provider_contract"
SOURCE_INPUT = PHASE1 / "da3_streaming_full_scene0050_input"
HF_CACHE = PHASE1 / "hf_cache"

VARIANTS = {
    "base": {
        "repo_id": "depth-anything/DA3-BASE",
    },
    "small": {
        "repo_id": "depth-anything/DA3-SMALL",
    },
    "large": {
        "repo_id": "depth-anything/DA3-LARGE",
    },
    "giant": {
        "repo_id": "depth-anything/DA3-GIANT",
    },
    "nested_giant_large": {
        "repo_id": "depth-anything/DA3NESTED-GIANT-LARGE",
    },
}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _input_dir(frame_count: int) -> Path:
    return PHASE1 / f"da3_streaming_d4rt32o3_scene0050_input{frame_count}"


def _output_root(variant: str, frame_count: int) -> Path:
    return PHASE1 / f"da3_streaming_d4rt32o3_scene0050_{variant}_input{frame_count}"


def _expected_chunk_count(frame_count: int, chunk_size: int = 32, overlap: int = 3) -> int:
    if frame_count <= 0:
        return 0
    if frame_count <= chunk_size:
        return 1
    stride = chunk_size - overlap
    return 1 + ((frame_count - chunk_size + stride - 1) // stride)


def _prepare_input(count: int) -> dict[str, Any]:
    manifest_path = SOURCE_INPUT / "frame_manifest_rows.csv"
    rows: list[dict[str, str]]
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))[:count]
    if not rows:
        raise ValueError(f"no source rows found in {manifest_path}")
    if len(rows) != count:
        raise ValueError(f"requested {count} frames but only found {len(rows)} rows in {manifest_path}")
    input_dir = _input_dir(count)
    input_dir.mkdir(parents=True, exist_ok=True)
    for old in input_dir.glob("*.jpg"):
        old.unlink()
    for idx, row in enumerate(rows):
        src = Path(row["source_rgb"])
        dst = input_dir / row["image_name"]
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(src, dst)
        row["da3_frame_index"] = str(idx)
        row["symlink_path"] = str(dst)
    with (input_dir / "frame_manifest_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "scene_id": "scene0050_00",
        "input_dir": str(input_dir),
        "source_input_dir": str(SOURCE_INPUT),
        "source_manifest": str(manifest_path),
        "frame_count": len(rows),
        "frame_id_min": int(rows[0]["frame_id"]),
        "frame_id_max": int(rows[-1]["frame_id"]),
        "stride_frame_id": int(rows[1]["frame_id"]) - int(rows[0]["frame_id"]) if len(rows) > 1 else None,
        "uses_gt_for_prediction": False,
        "uses_future": False,
    }
    _write_json(input_dir / "summary.json", summary)
    return summary


def _download_variant(variant: str) -> dict[str, Any]:
    row = VARIANTS[variant]
    repo_id = row["repo_id"]
    local_dir = snapshot_download(
        repo_id=repo_id,
        cache_dir=str(HF_CACHE),
        allow_patterns=["model.safetensors", "config.json"],
    )
    model_path = Path(local_dir) / "model.safetensors"
    config_path = Path(local_dir) / "config.json"
    return {
        "variant": variant,
        "repo_id": repo_id,
        "snapshot_dir": str(local_dir),
        "model_safetensors": str(model_path),
        "config_json": str(config_path),
        "model_exists": model_path.exists(),
        "config_exists": config_path.exists(),
        "model_size_bytes": model_path.stat().st_size if model_path.exists() else None,
        "config_size_bytes": config_path.stat().st_size if config_path.exists() else None,
    }


def _find_cached_snapshot(repo_id: str) -> Path:
    repo_name = repo_id.replace("/", "--")
    candidates = sorted((HF_CACHE / "hub").glob(f"models--{repo_name}/snapshots/*"))
    candidates.extend(sorted(HF_CACHE.glob(f"models--{repo_name}/snapshots/*")))
    if not candidates:
        raise FileNotFoundError(f"missing cached weights for {repo_id}; rerun with --download")
    return candidates[-1]


def _write_config(variant: str, weights: dict[str, Any], frame_count: int) -> Path:
    output_root = _output_root(variant, frame_count)
    output_root.mkdir(parents=True, exist_ok=True)
    config = {
        "Weights": {
            "DA3": weights["model_safetensors"],
            "DA3_CONFIG": weights["config_json"],
            "SALAD": "",
        },
        "Model": {
            "chunk_size": 32,
            "overlap": 3,
            "loop_chunk_size": 32,
            "loop_enable": False,
            "useDBoW": False,
            "delete_temp_files": False,
            "align_lib": "torch",
            "align_method": "sim3",
            "scale_compute_method": "auto",
            "align_type": "dense",
            "ref_view_strategy": "middle",
            "ref_view_strategy_loop": "middle",
            "depth_threshold": 15.0,
            "save_depth_conf_result": True,
            "save_debug_info": True,
            "Sparse_Align": {
                "keypoint_select": "orb",
                "keypoint_num": 5000,
            },
            "IRLS": {
                "delta": 0.1,
                "max_iters": 5,
                "tol": "1e-9",
            },
            "Pointcloud_Save": {
                "sample_ratio": 0.0005,
                "conf_threshold_coef": 0.75,
            },
        },
        "Loop": {
            "SALAD": {
                "image_size": [336, 336],
                "batch_size": 2,
                "similarity_threshold": 0.85,
                "top_k": 2,
                "use_nms": True,
                "nms_threshold": 25,
            },
            "SIM3_Optimizer": {
                "lang_version": "python",
                "max_iterations": 5,
                "lambda_init": 1e-6,
            },
        },
    }
    path = output_root / "da3_streaming_d4rt32o3_config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare v98.1 DA3 variant smoke configs for chunk=32 overlap=3.")
    parser.add_argument("--frame-count", type=int, default=35)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS.keys()), choices=list(VARIANTS.keys()))
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()

    input_summary = _prepare_input(args.frame_count)
    weights_rows: list[dict[str, Any]] = []
    config_rows: list[dict[str, Any]] = []
    for variant in args.variants:
        if args.download:
            weights = _download_variant(variant)
        else:
            snapshot = _find_cached_snapshot(VARIANTS[variant]["repo_id"])
            weights = {
                "variant": variant,
                "repo_id": VARIANTS[variant]["repo_id"],
                "snapshot_dir": str(snapshot),
                "model_safetensors": str(snapshot / "model.safetensors"),
                "config_json": str(snapshot / "config.json"),
                "model_exists": (snapshot / "model.safetensors").exists(),
                "config_exists": (snapshot / "config.json").exists(),
                "model_size_bytes": (snapshot / "model.safetensors").stat().st_size if (snapshot / "model.safetensors").exists() else None,
                "config_size_bytes": (snapshot / "config.json").stat().st_size if (snapshot / "config.json").exists() else None,
            }
        config_path = _write_config(variant, weights, int(input_summary["frame_count"]))
        weights_rows.append(weights)
        config_rows.append(
            {
                "variant": variant,
                "repo_id": VARIANTS[variant]["repo_id"],
                "config_path": str(config_path),
                "output_root": str(_output_root(variant, int(input_summary["frame_count"]))),
                "input_dir": str(_input_dir(int(input_summary["frame_count"]))),
                "chunk_size": 32,
                "overlap": 3,
                "frame_count": int(input_summary["frame_count"]),
                "expected_chunk_count": _expected_chunk_count(int(input_summary["frame_count"])),
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
    _write_json(
        PHASE1 / f"da3_streaming_d4rt32o3_variant_input{int(input_summary['frame_count'])}_prepare_summary.json",
        {
            "input_summary": input_summary,
            "weights": weights_rows,
            "configs": config_rows,
        },
    )
    print(json.dumps({"input": input_summary, "configs": config_rows, "weights": weights_rows}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
