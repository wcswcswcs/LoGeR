#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default="third_party/HorizonStream")
    parser.add_argument("--config", default="third_party/HorizonStream/configs/horizonstream_infer.yaml")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--kitti-src", required=True)
    parser.add_argument("--kitti-meta-root", required=True)
    parser.add_argument("--checkpoint", default="third_party/HorizonStream/checkpoints/HorizonStream.pt")
    parser.add_argument("--skyseg", default="third_party/HorizonStream/checkpoints/skyseg.onnx")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    config_path = Path(args.config).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model_cfg = cfg["model"]["horizonstream_cfg"]
    agg = model_cfg["agg_regator_cfg"]
    data = cfg["data"]
    infer = cfg["inference"]

    image_size = int(data.get("size", agg.get("img_size", 518)))
    patch_size = int(data.get("patch_size", agg.get("patch_size", 14)))
    patch_h = image_size // patch_size
    patch_w = image_size // patch_size
    patch_count = patch_h * patch_w
    enable_mrt = bool(model_cfg.get("enable_metric_readout_token", False))
    use_register = bool(agg.get("use_register_token", False))
    register_count = 4 if use_register else 0
    mrt_count = 1 if enable_mrt else 0
    special_patch_tokens = register_count + mrt_count
    patch_stream_tokens = special_patch_tokens + patch_count
    pose_tokens = int(agg.get("num_pose_tokens", 32))
    window_size = int(infer.get("window_size", 10))

    token_layout = {
        "config_path": str(config_path),
        "image_size": image_size,
        "crop": bool(data.get("crop", True)),
        "patch_size": patch_size,
        "patch_grid": [patch_h, patch_w],
        "patch_count": patch_count,
        "enable_metric_readout_token": enable_mrt,
        "metric_readout_token_index": 0 if enable_mrt and not use_register else None,
        "use_register_token": use_register,
        "register_token_count": register_count,
        "special_patch_tokens": special_patch_tokens,
        "patch_stream_tokens_P": patch_stream_tokens,
        "image_patch_token_range": [special_patch_tokens, patch_stream_tokens - 1],
        "pose_tokens_per_window_slot_X": pose_tokens,
        "window_size_Win": window_size,
        "global_tokens_per_frame_row_A": patch_stream_tokens + window_size * pose_tokens,
        "global_attn_arch": str(agg.get("global_attn_arch")),
        "global_attn_impl": str(agg.get("global_attn_impl")),
        "gla_serial_layers": list(agg.get("gla_serial_layers", [])),
        "intermediate_layer_idx": list(agg.get("intermediate_layer_idx", [])),
        "gate_attn": str(agg.get("gate_attn")),
    }
    write_json(out_dir / "hs_token_layout_report.json", token_layout)
    write_md(
        out_dir / "hs_token_layout_report.md",
        f"""# HorizonStream v113-HS Token Layout Report

- config: `{config_path}`
- input size: `{image_size}`
- patch size: `{patch_size}`
- patch grid: `{patch_h} x {patch_w}`
- image patch count: `{patch_count}`
- use register token: `{use_register}`
- enable MRT: `{enable_mrt}`
- MRT index: `{token_layout['metric_readout_token_index']}`
- patch stream tokens `P`: `{patch_stream_tokens}`
- image patch token range: `{special_patch_tokens}..{patch_stream_tokens - 1}`
- pose tokens per local-window slot `X`: `{pose_tokens}`
- window size `Win`: `{window_size}`
- global tokens per frame row `A=P+Win*X`: `{token_layout['global_tokens_per_frame_row_A']}`
- global attention arch: `{token_layout['global_attn_arch']}`
- GLA serial layers: `{token_layout['gla_serial_layers']}`
- intermediate layers: `{token_layout['intermediate_layer_idx']}`
- gate attention: `{token_layout['gate_attn']}`

Audit note: with default config, semantic projection must target only image patch tokens, not MRT or pose tokens. MRT is patch-stream index 0, and pose tokens are appended later in the global path.
""",
    )

    checkpoints = []
    for label, raw_path in [
        ("HorizonStream.pt", args.checkpoint),
        ("skyseg.onnx", args.skyseg),
    ]:
        path = Path(raw_path).resolve()
        checkpoints.append(
            {
                "label": label,
                "path": str(path),
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else None,
                "sha256": sha256_file(path) if path.exists() else None,
            }
        )
    write_json(
        out_dir / "stage0_hs_checkpoint_audit.json",
        {
            "repo_root": str(repo_root),
            "config_path": str(config_path),
            "checkpoints": checkpoints,
            "loop_closure_main_result": "disabled_for_main_v113hs_baseline",
        },
    )

    code_loci_rows = [
        {
            "module_path": "horizonstream/runtime/models/horizonstream.py",
            "function": "HorizonStreamAggregator._run_gla_global_block",
            "hook_type": "GLA pre-input scaling",
            "status": "feasible",
            "next_attempt": "add identity-gated frame scale map before gla_block",
        },
        {
            "module_path": "horizonstream/runtime/models/horizonstream.py",
            "function": "GLACache.update",
            "hook_type": "GLA state trace/state-delta scaling",
            "status": "feasible_for_state_delta_not_direct_gamma",
            "next_attempt": "trace recurrent_state and optional delta scaling; do not call it gamma modulation",
        },
        {
            "module_path": "horizonstream/runtime/models/horizonstream.py",
            "function": "HorizonStreamAggregator._process_causal_frame_attention",
            "hook_type": "local KV value scaling",
            "status": "feasible",
            "next_attempt": "scale local_kv_cache.kv_ori value slice before pose attention",
        },
        {
            "module_path": "horizonstream/runtime/layers/attention.py",
            "function": "Attention.forward",
            "hook_type": "manual attention trace/head gate",
            "status": "partial",
            "next_attempt": "requires unfused/manual attention weights for head-specific risk",
        },
        {
            "module_path": "horizonstream/models/horizonstream.py",
            "function": "HorizonStream._predict_metric_scale",
            "hook_type": "MRT feature gate/readout probe",
            "status": "feasible_high_risk",
            "next_attempt": "trace feature norm and predicted_metric_scale before any feature gate",
        },
        {
            "module_path": "horizonstream/runtime/heads/camera_head.py",
            "function": "BatchCameraHead.forward",
            "hook_type": "pose readout probe",
            "status": "feasible",
            "next_attempt": "trace win_pose_tokens norm and pose residual stats",
        },
        {
            "module_path": "horizonstream/core/infer.py",
            "function": "run_inference_cfg",
            "hook_type": "chunk loop/state advance",
            "status": "feasible",
            "next_attempt": "record chunk schedule and state transition sidecar",
        },
    ]
    fields = ["module_path", "function", "hook_type", "status", "next_attempt"]
    write_csv(out_dir / "hs_code_loci.csv", code_loci_rows, fields)
    write_csv(out_dir / "stage0_hs_code_loci.csv", code_loci_rows, fields)

    write_md(
        out_dir / "hs_hook_contract_gla.md",
        """# HS-A GLA Hook Contract

Primary trace hook: `GLACache.update`.

- Allowed trace-only fields: recurrent state shape/norm, conv state shape/norm, previous/new/delta norm when previous state exists.
- Allowed first action: pre-GLA input scaling in `HorizonStreamAggregator._run_gla_global_block`.
- Allowed second action: state-delta scaling in `GLACache.update` only when state shapes match.
- Not allowed: report state-delta scaling as direct gamma modulation.
- Direct KDA gamma modulation is blocked until a concrete KDA retention/gamma tensor is located.
""",
    )
    write_md(
        out_dir / "hs_hook_contract_local.md",
        """# HS-L Local Hook Contract

Primary trace/action hook: `HorizonStreamAggregator._process_causal_frame_attention`.

- Local KV cache shape after expansion is `[2, B*F*Win, H, P, D]`.
- Index 0 is K and index 1 is V.
- First action family scales only V before pose tokens read local patch evidence.
- MRT/source special tokens stay untouched in the first HS-L candidate.
- Attention-logit bias/head-specific gates are deferred until mask shape or manual attention trace is validated.
""",
    )
    write_md(
        out_dir / "hs_hook_contract_mrt.md",
        """# HS-M MRT Hook Contract

Primary diagnosis/action hook: `HorizonStream._predict_metric_scale`.

- MRT index is `patch_start_idx - 1`, which is 0 under the current config.
- Allowed diagnosis: feature norm, predicted metric scale, chunk semantic risk/stable mass, scale/error correlations.
- Allowed first action: feature-space gate or residual blend before `metric_readout_head`.
- Not allowed as a promoted method: direct GT/ATE-informed scale manipulation or post-hoc scale correction.
""",
    )
    write_md(
        out_dir / "stage0_hs_repo_audit.md",
        f"""# Stage0 HorizonStream Repo Audit

- repo root: `{repo_root}`
- config: `{config_path}`
- checkpoint audit: `stage0_hs_checkpoint_audit.json`
- code loci: `hs_code_loci.csv`
- token layout: `hs_token_layout_report.md`
- KITTI source: `{Path(args.kitti_src).resolve()}`
- planned generalizable meta-root: `{Path(args.kitti_meta_root).resolve()}`
- main result loop closure: disabled
- runtime env: `conda activate loger`
""",
    )

    command_rows = []
    conda_prefix = "source /mnt/data/users/chengshun.wang/miniconda3/etc/profile.d/conda.sh && conda activate loger"
    common = (
        "cd third_party/HorizonStream && "
        "CUDA_VISIBLE_DEVICES={gpu} python run_pipeline.py "
        "--config configs/horizonstream_infer.yaml "
        "--img-path {meta_root} --seq-list {seq} --camera 02 "
        "--checkpoint checkpoints/HorizonStream.pt "
        "--output-root {output_root} "
        "--no-camera-preprocess --offload-outputs-to-cpu "
        "--no-save-videos --no-save-points --no-save-images --no-save-depth --no-save-depth-conf "
        "--no-mask-sky --no-point-mask-sky --no-loop "
        "--eval-pose-variants main"
    )
    for seq, gpu in [("00", "4"), ("02", "5"), ("05", "6")]:
        command_rows.append(
            {
                "stage": "stage1_baseline",
                "seq": seq,
                "gpu": gpu,
                "command": f"{conda_prefix} && " + common.format(
                    gpu=gpu,
                    meta_root=str(Path(args.kitti_meta_root).resolve()),
                    seq=seq,
                    output_root=str((out_dir.parent / "outputs" / f"baseline_kitti_{seq}").resolve()),
                ),
            }
        )
    command_rows.append(
        {
            "stage": "stage0_smoke",
            "seq": "00_max12",
            "gpu": "4",
            "command": f"{conda_prefix} && cd third_party/HorizonStream && CUDA_VISIBLE_DEVICES=4 python run_pipeline.py --config configs/horizonstream_infer.yaml --img-path {Path(args.kitti_meta_root).resolve()} --seq-list 00 --camera 02 --checkpoint checkpoints/HorizonStream.pt --output-root {(out_dir.parent / 'outputs' / 'smoke_kitti00_max12').resolve()} --max-frames 12 --no-camera-preprocess --no-save-videos --no-save-points --no-save-images --save-depth --save-depth-conf --no-mask-sky --no-point-mask-sky --no-loop --eval-pose-variants main",
        }
    )
    write_csv(out_dir / "hs_baseline_repro_command_manifest.csv", command_rows, ["stage", "seq", "gpu", "command"])

    print(f"wrote audit artifacts to {out_dir}")


if __name__ == "__main__":
    main()
