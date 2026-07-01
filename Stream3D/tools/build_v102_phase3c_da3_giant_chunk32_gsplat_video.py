from __future__ import annotations

import csv
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
STREAM3D = ROOT / "Stream3D"
AUDIT_ROOT = STREAM3D / "outputs" / "audit"
OUT_DIR = AUDIT_ROOT / "v102_phase3c_da3_giant_chunk32_gsplat_video"
V98_PROVIDER = AUDIT_ROOT / "v98_phase1_provider_contract"
DA3_REPO = ROOT.parent / "Depth-Anything-3"
PLAN_DOC = ROOT / "docs" / "stream4d_v102_mask_pair_primitive_bridge_da3_giant_3dgs_plan.md"
PYTHON = Path("/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python")


def _rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return _rel(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields or ["schema_version"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key, "")) for key in fields})


def _input_images(frame_count: int) -> list[Path]:
    image_root = V98_PROVIDER / "da3_streaming_d4rt32o3_scene0050_input119"
    images = sorted(image_root.glob("*.png")) + sorted(image_root.glob("*.jpg"))
    return images[:frame_count]


def _render_script(output_dir: Path, image_paths: list[Path], model_id: str) -> str:
    return f"""
import json
import numpy as np
import torch
from pathlib import Path
from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.export.gs import export_to_gs_video

out = Path({str(output_dir)!r})
out.mkdir(parents=True, exist_ok=True)
images = {[str(p) for p in image_paths]!r}
model = DepthAnything3.from_pretrained({model_id!r}).to('cuda').eval()
pred = model.inference(
    images,
    infer_gs=True,
    process_res={int(os.environ.get("V102_GSPLAT_PROCESS_RES", "252"))},
    export_dir=None,
    export_format='mini_npz',
)
export_to_gs_video(
    pred,
    str(out),
    out_image_hw=tuple({tuple(int(x) for x in os.environ.get("V102_GSPLAT_OUT_HW", "182,252").split(","))!r}),
    chunk_size={int(os.environ.get("V102_GSPLAT_RENDER_CHUNK_SIZE", "1"))},
    trj_mode={os.environ.get("V102_GSPLAT_TRJ_MODE", "interpolate_smooth")!r},
    color_mode={os.environ.get("V102_GSPLAT_COLOR_MODE", "RGB+D")!r},
    vis_depth=None,
    enable_tqdm=True,
    output_name={os.environ.get("V102_GSPLAT_OUTPUT_NAME", "chunk32_interpolate_smooth_rgb")!r},
    video_quality={os.environ.get("V102_GSPLAT_VIDEO_QUALITY", "medium")!r},
)
video_files = [str(p.relative_to(out)) for p in (out / 'gs_video').glob('*.mp4')]
summary = {{
    'model_id': {model_id!r},
    'frame_count': len(images),
    'process_res': {int(os.environ.get("V102_GSPLAT_PROCESS_RES", "252"))},
    'trj_mode': {os.environ.get("V102_GSPLAT_TRJ_MODE", "interpolate_smooth")!r},
    'render_chunk_size': {int(os.environ.get("V102_GSPLAT_RENDER_CHUNK_SIZE", "1"))},
    'out_image_hw': list(tuple({tuple(int(x) for x in os.environ.get("V102_GSPLAT_OUT_HW", "182,252").split(","))!r})),
    'depth_shape': list(pred.depth.shape) if getattr(pred, 'depth', None) is not None else None,
    'extrinsics_shape': list(pred.extrinsics.shape) if getattr(pred, 'extrinsics', None) is not None else None,
    'intrinsics_shape': list(pred.intrinsics.shape) if getattr(pred, 'intrinsics', None) is not None else None,
    'gaussian_means_shape': list(pred.gaussians.means.shape) if getattr(pred, 'gaussians', None) is not None else None,
    'video_files': video_files,
}}
(out / 'render_summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\\n')
torch.cuda.empty_cache()
"""


def main() -> int:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model_id = os.environ.get("V102_DA3_GIANT_MODEL", "depth-anything/DA3-GIANT-1.1")
    cuda_device = os.environ.get("V102_CUDA_DEVICE", "6")
    frame_count = int(os.environ.get("V102_GSPLAT_FRAME_COUNT", "32"))
    image_paths = _input_images(frame_count)
    if len(image_paths) < frame_count:
        raise RuntimeError(f"Only found {len(image_paths)} input images for requested frame_count={frame_count}")

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{DA3_REPO / 'src'}:{env.get('PYTHONPATH', '')}"
    env["CUDA_VISIBLE_DEVICES"] = cuda_device
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    cmd = [str(PYTHON), "-c", _render_script(OUT_DIR, image_paths, model_id)]
    proc_t0 = time.time()
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        env=env,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=int(os.environ.get("V102_GSPLAT_TIMEOUT_SEC", "3600")),
    )
    runtime = time.time() - proc_t0
    (OUT_DIR / "stdout.log").write_text(proc.stdout, encoding="utf-8")
    (OUT_DIR / "stderr.log").write_text(proc.stderr, encoding="utf-8")
    lower = (proc.stdout + "\n" + proc.stderr).lower()
    render_summary_path = OUT_DIR / "render_summary.json"
    render_summary = json.loads(render_summary_path.read_text(encoding="utf-8")) if render_summary_path.exists() else {}
    video_files = sorted((OUT_DIR / "gs_video").glob("*.mp4"))
    rows = [
        {
            "schema_version": "stream4d_v102_phase3c_gsplat_video_row_v1",
            "phase_id": "v102_phase3c_da3_giant_chunk32_gsplat_video",
            "model_id": model_id,
            "cuda_device": cuda_device,
            "frame_count": frame_count,
            "process_res": int(os.environ.get("V102_GSPLAT_PROCESS_RES", "252")),
            "trj_mode": os.environ.get("V102_GSPLAT_TRJ_MODE", "interpolate_smooth"),
            "render_chunk_size": int(os.environ.get("V102_GSPLAT_RENDER_CHUNK_SIZE", "1")),
            "out_image_hw": os.environ.get("V102_GSPLAT_OUT_HW", "182,252"),
            "exit_code": proc.returncode,
            "runtime_sec": runtime,
            "OOM_flag": "out of memory" in lower or ("cuda" in lower and "memory" in lower),
            "render_success": proc.returncode == 0 and bool(video_files),
            "video_file": _rel(video_files[0]) if video_files else "",
            "video_file_size_MB": float(video_files[0].stat().st_size / (1024 * 1024)) if video_files else "",
            "gaussian_means_shape": render_summary.get("gaussian_means_shape", ""),
            "stderr_tail": proc.stderr[-1200:],
        }
    ]
    video_csv = OUT_DIR / "gsplat_video_rows.csv"
    _write_csv(video_csv, rows)
    summary = {
        "schema_version": "stream4d_v102_phase3c_da3_giant_chunk32_gsplat_video_summary_v1",
        "phase_id": "v102_phase3c_da3_giant_chunk32_gsplat_video",
        "created_unix_time": time.time(),
        "runtime_sec": time.time() - t0,
        "decision": "PASS_GSPLAT_INTERPOLATED_POSE_VIDEO" if rows[0]["render_success"] else "NO_GO_GSPLAT_VIDEO_RENDER_FAILED",
        "render_success": rows[0]["render_success"],
        "model_id": model_id,
        "frame_count": frame_count,
        "process_res": int(os.environ.get("V102_GSPLAT_PROCESS_RES", "252")),
        "trj_mode": os.environ.get("V102_GSPLAT_TRJ_MODE", "interpolate_smooth"),
        "render_chunk_size": int(os.environ.get("V102_GSPLAT_RENDER_CHUNK_SIZE", "1")),
        "out_image_hw": os.environ.get("V102_GSPLAT_OUT_HW", "182,252"),
        "video_file": rows[0]["video_file"],
        "video_file_size_MB": rows[0]["video_file_size_MB"],
        "gaussian_means_shape": render_summary.get("gaussian_means_shape", ""),
        "truthfulness_note": "Video is rendered via gsplat from a fresh DA3-GIANT-1.1 prediction using interpolated input-view poses.",
        "plan_doc": _rel(PLAN_DOC),
        "outputs": {
            "summary": _rel(OUT_DIR / "summary.json"),
            "render_summary": _rel(render_summary_path) if render_summary_path.exists() else "",
            "gsplat_video_rows": _rel(video_csv),
            "stdout_log": _rel(OUT_DIR / "stdout.log"),
            "stderr_log": _rel(OUT_DIR / "stderr.log"),
        },
    }
    _write_json(OUT_DIR / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if rows[0]["render_success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
