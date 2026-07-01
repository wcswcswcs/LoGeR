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
OUT_DIR = AUDIT_ROOT / "v102_phase2b_da3_giant_chunk32_audit"
V98_PROVIDER = AUDIT_ROOT / "v98_phase1_provider_contract"
DA3_REPO = ROOT.parent / "Depth-Anything-3"
PLAN_DOC = ROOT / "docs" / "stream4d_v102_mask_pair_primitive_bridge_da3_giant_3dgs_plan.md"
PYTHON = Path("/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python")


ATTEMPTS = [
    {"attempt_id": "smoke8_process252", "frame_count": 8, "process_res": 252},
    {"attempt_id": "chunk32_process252", "frame_count": 32, "process_res": 252},
    {"attempt_id": "chunk32_process196", "frame_count": 32, "process_res": 196},
    {"attempt_id": "chunk32_process168", "frame_count": 32, "process_res": 168},
]


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


def _script(output_dir: Path, image_paths: list[Path], model_id: str, process_res: int) -> str:
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
(out / 'smoke_summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\\n')
"""


def _run_attempt(spec: dict[str, Any]) -> dict[str, Any]:
    model_id = os.environ.get("V102_DA3_GIANT_MODEL", "depth-anything/DA3-GIANT-1.1")
    cuda_device = os.environ.get("V102_CUDA_DEVICE", "6")
    image_paths = _input_images(int(spec["frame_count"]))
    attempt_dir = OUT_DIR / str(spec["attempt_id"])
    attempt_dir.mkdir(parents=True, exist_ok=True)
    row = {
        "schema_version": "stream4d_v102_phase2b_chunk_attempt_row_v1",
        "phase_id": "v102_phase2b_da3_giant_chunk32_audit",
        "attempt_id": spec["attempt_id"],
        "model_id": model_id,
        "cuda_device": cuda_device,
        "frame_count": int(spec["frame_count"]),
        "process_res": int(spec["process_res"]),
        "export_format": "gs_ply_only_plus_mini_npz",
        "output_dir": _rel(attempt_dir),
        "exit_code": "",
        "runtime_sec": "",
        "OOM_flag": False,
        "export_success": False,
        "ply_file": "",
        "ply_file_size_MB": "",
        "mini_npz_file": "",
        "mini_npz_file_size_MB": "",
        "stderr_tail": "",
    }
    if len(image_paths) < int(spec["frame_count"]):
        row["exit_code"] = "skipped"
        row["stderr_tail"] = f"Only found {len(image_paths)} input images."
        return row

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{DA3_REPO / 'src'}:{env.get('PYTHONPATH', '')}"
    env["CUDA_VISIBLE_DEVICES"] = cuda_device
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    cmd = [str(PYTHON), "-c", _script(attempt_dir, image_paths, model_id, int(spec["process_res"]))]
    t0 = time.time()
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        env=env,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=int(os.environ.get("V102_CHUNK_ATTEMPT_TIMEOUT_SEC", "2400")),
    )
    runtime = time.time() - t0
    (attempt_dir / "stdout.log").write_text(proc.stdout, encoding="utf-8")
    (attempt_dir / "stderr.log").write_text(proc.stderr, encoding="utf-8")
    lower = (proc.stdout + "\n" + proc.stderr).lower()
    ply_files = sorted((attempt_dir / "gs_ply").glob("*.ply"))
    mini_npz = attempt_dir / "exports" / "mini_npz" / "results.npz"
    row.update(
        {
            "exit_code": proc.returncode,
            "runtime_sec": runtime,
            "OOM_flag": "out of memory" in lower or ("cuda" in lower and "memory" in lower),
            "export_success": proc.returncode == 0 and bool(ply_files) and mini_npz.exists(),
            "ply_file": _rel(ply_files[0]) if ply_files else "",
            "ply_file_size_MB": float(ply_files[0].stat().st_size / (1024 * 1024)) if ply_files else "",
            "mini_npz_file": _rel(mini_npz) if mini_npz.exists() else "",
            "mini_npz_file_size_MB": float(mini_npz.stat().st_size / (1024 * 1024)) if mini_npz.exists() else "",
            "stderr_tail": proc.stderr[-1000:],
        }
    )
    return row


def main() -> int:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    chunk_success_seen = False
    for spec in ATTEMPTS:
        if chunk_success_seen and int(spec["frame_count"]) == 32:
            continue
        row = _run_attempt(spec)
        rows.append(row)
        if int(spec["frame_count"]) == 32 and bool(row.get("export_success")):
            chunk_success_seen = True

    attempt_csv = OUT_DIR / "chunk_attempt_rows.csv"
    _write_csv(attempt_csv, rows)
    smoke8_success = any(int(r.get("frame_count", 0)) == 8 and bool(r.get("export_success")) for r in rows)
    chunk32_success_rows = [r for r in rows if int(r.get("frame_count", 0)) == 32 and bool(r.get("export_success"))]
    best_chunk = chunk32_success_rows[0] if chunk32_success_rows else {}
    decision = (
        "PASS_CHUNK32_3DGS_EXPORT"
        if chunk32_success_rows
        else "NO_GO_CHUNK32_3DGS_EXPORT_FAILED__SMOKE8_PASS"
        if smoke8_success
        else "NO_GO_DA3_GIANT_3DGS_CHUNK_AUDIT_FAILED"
    )
    summary = {
        "schema_version": "stream4d_v102_phase2b_da3_giant_chunk32_audit_summary_v1",
        "phase_id": "v102_phase2b_da3_giant_chunk32_audit",
        "created_unix_time": time.time(),
        "runtime_sec": time.time() - t0,
        "decision": decision,
        "smoke8_success": smoke8_success,
        "chunk32_success": bool(chunk32_success_rows),
        "chunk32_attempt_count": sum(1 for r in rows if int(r.get("frame_count", 0)) == 32),
        "best_chunk32_attempt_id": best_chunk.get("attempt_id", ""),
        "best_chunk32_frame_count": best_chunk.get("frame_count", ""),
        "best_chunk32_process_res": best_chunk.get("process_res", ""),
        "best_chunk32_ply_file": best_chunk.get("ply_file", ""),
        "best_chunk32_ply_file_size_MB": best_chunk.get("ply_file_size_MB", ""),
        "best_chunk32_mini_npz_file": best_chunk.get("mini_npz_file", ""),
        "truthfulness_note": "This is the requested chunk-level DA3-GIANT-1.1 3DGS audit; success is claimed only for a real 32-frame export.",
        "plan_doc": _rel(PLAN_DOC),
        "outputs": {
            "summary": _rel(OUT_DIR / "summary.json"),
            "chunk_attempt_rows": _rel(attempt_csv),
        },
    }
    _write_json(OUT_DIR / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if chunk32_success_rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
