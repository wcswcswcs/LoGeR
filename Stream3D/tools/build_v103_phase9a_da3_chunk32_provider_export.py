from __future__ import annotations

import argparse
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
OUT_DIR = AUDIT_ROOT / "v103_phase9a_da3_chunk32_provider_export"
DA3_REPO = ROOT.parent / "Depth-Anything-3"
PLAN_DOC = ROOT / "docs" / "stream4d_v103_training_free_primitive_affinity_field_experiment_plan.md"
PYTHON = Path("/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python")


SCENES = {
    "scene0011_00": {
        "input_manifest": AUDIT_ROOT
        / "v98_phase1_provider_contract"
        / "da3_streaming_full_scene0011_input"
        / "frame_manifest_rows.csv",
        "default_cuda": "6",
    },
    "scene0050_00": {
        "input_manifest": AUDIT_ROOT
        / "v98_phase1_provider_contract"
        / "da3_streaming_d4rt32o3_scene0050_input119"
        / "frame_manifest_rows.csv",
        "default_cuda": "7",
        "reuse_ply": AUDIT_ROOT
        / "v102_phase2b_da3_giant_chunk32_audit"
        / "chunk32_process252"
        / "gs_ply"
        / "0000.ply",
        "reuse_mini_npz": AUDIT_ROOT
        / "v102_phase2b_da3_giant_chunk32_audit"
        / "chunk32_process252"
        / "exports"
        / "mini_npz"
        / "results.npz",
    },
}


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
    if isinstance(value, np.generic):
        return value.item()
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


def _read_manifest(path: Path, frame_count: int) -> list[Path]:
    rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
    rows = sorted(rows, key=lambda row: int(row["da3_frame_index"]))
    image_paths: list[Path] = []
    for row in rows[:frame_count]:
        symlink = Path(row.get("symlink_path", ""))
        source = Path(row.get("source_rgb", ""))
        image_paths.append(symlink if symlink.exists() else source)
    return image_paths


def _infer_script(output_dir: Path, image_paths: list[Path], model_id: str, process_res: int) -> str:
    return f"""
import json
import numpy as np
from pathlib import Path
from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.export.gs import export_to_gs_ply

out = Path({str(output_dir)!r})
out.mkdir(parents=True, exist_ok=True)
images = {[str(p) for p in image_paths]!r}
model = DepthAnything3.from_pretrained({model_id!r}).to('cuda').eval()
pred = model.inference(
    images,
    infer_gs=True,
    process_res={int(process_res)},
    export_dir=None,
    export_format='mini_npz',
)
export_to_gs_ply(pred, str(out))
mini_dir = out / 'exports' / 'mini_npz'
mini_dir.mkdir(parents=True, exist_ok=True)
save_dict = {{'depth': np.round(pred.depth, 8)}}
if getattr(pred, 'conf', None) is not None:
    save_dict['conf'] = np.round(pred.conf, 2)
if getattr(pred, 'extrinsics', None) is not None:
    save_dict['extrinsics'] = pred.extrinsics
if getattr(pred, 'intrinsics', None) is not None:
    save_dict['intrinsics'] = pred.intrinsics
np.savez_compressed(mini_dir / 'results.npz', **save_dict)
summary = {{
    'frame_count': len(images),
    'process_res': {int(process_res)},
    'depth_available': getattr(pred, 'depth', None) is not None,
    'gaussians_available': getattr(pred, 'gaussians', None) is not None,
    'depth_shape': list(pred.depth.shape) if getattr(pred, 'depth', None) is not None else None,
    'extrinsics_shape': list(pred.extrinsics.shape) if getattr(pred, 'extrinsics', None) is not None else None,
    'intrinsics_shape': list(pred.intrinsics.shape) if getattr(pred, 'intrinsics', None) is not None else None,
    'export_files': [str(p.relative_to(out)) for p in out.rglob('*') if p.is_file()][:500],
}}
(out / 'provider_export_summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\\n')
"""


def _run_scene(
    scene_id: str,
    *,
    frame_count: int,
    process_res: int,
    cuda_device: str | None,
    model_id: str,
    force: bool,
    timeout_sec: int,
) -> dict[str, Any]:
    spec = SCENES[scene_id]
    attempt_id = f"{scene_id}_chunk32_process{process_res}"
    attempt_dir = OUT_DIR / attempt_id
    ply = attempt_dir / "gs_ply" / "0000.ply"
    mini_npz = attempt_dir / "exports" / "mini_npz" / "results.npz"
    device = cuda_device or str(spec.get("default_cuda", "6"))
    row: dict[str, Any] = {
        "schema_version": "stream4d_v103_phase9a_da3_chunk32_export_row_v1",
        "phase_id": "v103_phase9a_da3_chunk32_provider_export",
        "scene_id": scene_id,
        "attempt_id": attempt_id,
        "model_id": model_id,
        "cuda_device": device,
        "frame_count": frame_count,
        "process_res": process_res,
        "export_format": "official_DA3_GIANT_1_1_gs_ply_plus_mini_npz",
        "output_dir": _rel(attempt_dir),
        "reused_from": "",
        "exit_code": "",
        "runtime_sec": "",
        "OOM_flag": False,
        "export_success": False,
        "ply_file": "",
        "ply_file_size_MB": "",
        "mini_npz_file": "",
        "mini_npz_file_size_MB": "",
        "stderr_tail": "",
        "uses_gt_for_prediction": False,
        "uses_future": False,
    }
    if not force and ply.exists() and mini_npz.exists():
        row.update(
            {
                "reused_from": "existing_v103_artifact",
                "exit_code": "skipped_existing",
                "export_success": True,
                "ply_file": _rel(ply),
                "ply_file_size_MB": float(ply.stat().st_size / (1024 * 1024)),
                "mini_npz_file": _rel(mini_npz),
                "mini_npz_file_size_MB": float(mini_npz.stat().st_size / (1024 * 1024)),
            }
        )
        return row

    reuse_ply = spec.get("reuse_ply")
    reuse_mini = spec.get("reuse_mini_npz")
    if not force and isinstance(reuse_ply, Path) and isinstance(reuse_mini, Path) and reuse_ply.exists() and reuse_mini.exists():
        row.update(
            {
                "reused_from": "v102_phase2b_da3_giant_chunk32_audit",
                "exit_code": "skipped_reuse_prior_verified_artifact",
                "export_success": True,
                "ply_file": _rel(reuse_ply),
                "ply_file_size_MB": float(reuse_ply.stat().st_size / (1024 * 1024)),
                "mini_npz_file": _rel(reuse_mini),
                "mini_npz_file_size_MB": float(reuse_mini.stat().st_size / (1024 * 1024)),
            }
        )
        return row

    image_paths = _read_manifest(Path(spec["input_manifest"]), frame_count)
    missing = [p for p in image_paths if not p.exists()]
    if len(image_paths) < frame_count or missing:
        row["exit_code"] = "skipped_missing_input_images"
        row["stderr_tail"] = f"image_count={len(image_paths)} missing_count={len(missing)}"
        return row

    attempt_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{DA3_REPO / 'src'}:{env.get('PYTHONPATH', '')}"
    env["CUDA_VISIBLE_DEVICES"] = device
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    cmd = [str(PYTHON), "-c", _infer_script(attempt_dir, image_paths, model_id, process_res)]
    t0 = time.time()
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        env=env,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout_sec,
    )
    runtime = time.time() - t0
    (attempt_dir / "stdout.log").write_text(proc.stdout, encoding="utf-8")
    (attempt_dir / "stderr.log").write_text(proc.stderr, encoding="utf-8")
    lower = (proc.stdout + "\n" + proc.stderr).lower()
    oom = "out of memory" in lower or ("cuda" in lower and "memory" in lower)
    row.update(
        {
            "exit_code": proc.returncode,
            "runtime_sec": runtime,
            "OOM_flag": oom,
            "export_success": proc.returncode == 0 and ply.exists() and mini_npz.exists(),
            "ply_file": _rel(ply) if ply.exists() else "",
            "ply_file_size_MB": float(ply.stat().st_size / (1024 * 1024)) if ply.exists() else "",
            "mini_npz_file": _rel(mini_npz) if mini_npz.exists() else "",
            "mini_npz_file_size_MB": float(mini_npz.stat().st_size / (1024 * 1024)) if mini_npz.exists() else "",
            "stderr_tail": proc.stderr[-1200:],
        }
    )
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", choices=["all", *SCENES.keys()], default="all")
    parser.add_argument("--frame-count", type=int, default=32)
    parser.add_argument("--process-res", type=int, default=252)
    parser.add_argument("--cuda-device", default=None)
    parser.add_argument("--model-id", default=os.environ.get("V103_DA3_GIANT_MODEL", "depth-anything/DA3-GIANT-1.1"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--timeout-sec", type=int, default=int(os.environ.get("V103_DA3_EXPORT_TIMEOUT_SEC", "2400")))
    args = parser.parse_args()

    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scenes = list(SCENES) if args.scene == "all" else [args.scene]
    rows = [
        _run_scene(
            scene_id,
            frame_count=args.frame_count,
            process_res=args.process_res,
            cuda_device=args.cuda_device,
            model_id=args.model_id,
            force=args.force,
            timeout_sec=args.timeout_sec,
        )
        for scene_id in scenes
    ]
    row_path = OUT_DIR / "chunk32_export_rows.csv"
    _write_csv(row_path, rows)
    success_scenes = [row["scene_id"] for row in rows if bool(row.get("export_success"))]
    failure_rows = [row for row in rows if not bool(row.get("export_success"))]
    failure_path = OUT_DIR / "failure_rows.csv"
    _write_csv(failure_path, failure_rows)
    summary = {
        "schema_version": "stream4d_v103_phase9a_da3_chunk32_export_summary_v1",
        "phase_id": "v103_phase9a_da3_chunk32_provider_export",
        "created_unix_time": time.time(),
        "runtime_sec": time.time() - t0,
        "plan_doc": _rel(PLAN_DOC),
        "scene_count": len(rows),
        "success_scene_count": len(success_scenes),
        "success_scenes": success_scenes,
        "failure_count": len(failure_rows),
        "decision": "PASS_DA3_CHUNK32_EXPORT_AVAILABLE"
        if len(success_scenes) == len(rows)
        else "PARTIAL_DA3_CHUNK32_EXPORT_AVAILABLE",
        "truthfulness_note": (
            "This phase only establishes official DA3-GIANT-1.1 chunk32 3DGS provider artifacts. "
            "It does not claim primitive bridge quality, AP, or DA3_PROVIDER_READY."
        ),
        "outputs": {
            "summary": _rel(OUT_DIR / "summary.json"),
            "chunk32_export_rows": _rel(row_path),
            "failure_rows": _rel(failure_path),
        },
    }
    _write_json(OUT_DIR / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if not failure_rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
