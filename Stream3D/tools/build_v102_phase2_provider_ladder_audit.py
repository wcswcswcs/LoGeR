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
OUT_DIR = AUDIT_ROOT / "v102_phase2_provider_ladder_audit"

PHASE0_DIR = AUDIT_ROOT / "v102_phase0_fact_lock"
PHASE1_DIR = AUDIT_ROOT / "v102_phase1_fragment_casebook"
V98_PROVIDER = AUDIT_ROOT / "v98_phase1_provider_contract"
DA3_REPO = ROOT.parent / "Depth-Anything-3"
DA_CPP_REPO = ROOT.parent / "depth-anything.cpp"
PLAN_DOC = ROOT / "docs" / "stream4d_v102_mask_pair_primitive_bridge_da3_giant_3dgs_plan.md"

PYTHON = Path("/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python")

PROVIDERS = [
    ("P0_DA3_STREAMING_SMALL", "depth-anything/DA3-SMALL", V98_PROVIDER / "da3_streaming_d4rt32o3_scene0050_small_input119"),
    ("P1_DA3_STREAMING_BASE", "depth-anything/DA3-BASE", V98_PROVIDER / "da3_streaming_d4rt32o3_scene0050_base_input119"),
    ("P2_DA3_STREAMING_LARGE", "depth-anything/DA3-LARGE", V98_PROVIDER / "da3_streaming_d4rt32o3_scene0050_large_input119"),
    ("P3_DA3METRIC_LARGE", "depth-anything/DA3METRIC-LARGE", Path("")),
    ("P4_DA3NESTED_GIANT_LARGE", "depth-anything/DA3NESTED-GIANT-LARGE", V98_PROVIDER / "da3_streaming_d4rt32o3_scene0050_nested_giant_large_input119"),
    ("P5_DA3_GIANT_depth_pose", "depth-anything/DA3-GIANT", V98_PROVIDER / "da3_streaming_d4rt32o3_scene0050_giant_input119"),
    ("P6_DA3_GIANT_1_1_3DGS_official", "depth-anything/DA3-GIANT-1.1", Path("")),
    ("P10_DA3_GIANT_1_1_subchunk8_3DGS", "depth-anything/DA3-GIANT-1.1", Path("")),
    ("P11_DA3_GIANT_1_1_subchunk4_3DGS", "depth-anything/DA3-GIANT-1.1", Path("")),
]

MODEL_CACHE_ROOTS = [
    V98_PROVIDER / "hf_cache" / "models--depth-anything--DA3-GIANT",
    V98_PROVIDER / "hf_cache" / "models--depth-anything--DA3NESTED-GIANT-LARGE",
    V98_PROVIDER / "hf_cache" / "models--depth-anything--DA3-LARGE",
    V98_PROVIDER / "hf_cache" / "models--depth-anything--DA3-BASE",
    V98_PROVIDER / "hf_cache" / "models--depth-anything--DA3-SMALL",
    V98_PROVIDER / "hf_cache" / "models--depth-anything--DA3-GIANT-1.1",
    Path.home() / ".cache" / "huggingface" / "hub" / "models--depth-anything--DA3-GIANT-1.1",
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


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _file_size_mb(path: Path) -> float:
    return float(path.stat().st_size / (1024 * 1024)) if path.exists() else 0.0


def _count_files(path: Path, pattern: str) -> int:
    return len(list(path.glob(pattern))) if path.exists() else 0


def _first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _cache_row(path: Path) -> dict[str, Any]:
    blobs = list((path / "blobs").glob("*")) if (path / "blobs").exists() else []
    snapshots = list((path / "snapshots").glob("*")) if (path / "snapshots").exists() else []
    return {
        "schema_version": "stream4d_v102_phase2_model_cache_row_v1",
        "phase_id": "v102_phase2_provider_ladder_audit",
        "cache_path": _rel(path),
        "exists": path.exists(),
        "blob_count": len(blobs),
        "blob_size_MB_total": float(sum(p.stat().st_size for p in blobs if p.is_file()) / (1024 * 1024)),
        "snapshot_count": len(snapshots),
        "snapshot_paths": "|".join(_rel(p) for p in snapshots[:8]),
        "ref_main": (path / "refs" / "main").read_text(encoding="utf-8").strip() if (path / "refs" / "main").exists() else "",
    }


def _provider_contract(provider_id: str, model_id: str, root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    failures: list[dict[str, Any]] = []
    if str(root) == "." or str(root) == "":
        reason = (
            "No verified local artifact for this provider variant in current workspace; "
            "runtime attempt or model-id resolution is required before promotion."
        )
        failures.append(_failure(provider_id, "provider_artifact_missing", reason, "provider_contract"))
        return (
            {
                "schema_version": "stream4d_v102_phase2_provider_contract_row_v1",
                "phase_id": "v102_phase2_provider_ladder_audit",
                "provider_id": provider_id,
                "implementation": "official_DA3_or_runtime_required",
                "model_id": model_id,
                "source_root": "",
                "input_frame_count": "",
                "chunk_size": "",
                "subchunk_size": "8" if "subchunk8" in provider_id else "4" if "subchunk4" in provider_id else "",
                "overlap": "",
                "outputs_depth": False,
                "outputs_confidence": False,
                "outputs_pose": False,
                "outputs_intrinsics": False,
                "outputs_point_cloud": False,
                "outputs_3d_gaussians": False,
                "export_format": "",
                "OOM_flag": False,
                "runtime_sec": "",
                "peak_gpu_memory_MB": "",
                "peak_cpu_memory_MB": "",
                "output_file_size_MB": "",
                "frame_npz_count": 0,
                "contract_ok": False,
                "method_result_allowed": False,
                "contract_note": reason,
            },
            failures,
        )
    results = root / "results_output"
    pcd = root / "pcd" / "combined_pcd.ply"
    camera = root / "camera_poses.txt"
    intrinsic = root / "intrinsic.txt"
    frame_count = _count_files(results, "frame_*.npz")
    pcd_size = _file_size_mb(pcd)
    outputs_depth = frame_count > 0
    outputs_pose = camera.exists()
    outputs_intrinsics = intrinsic.exists()
    outputs_point_cloud = pcd.exists()
    log_path = Path(str(root) + ".log")
    log_text = log_path.read_text(encoding="utf-8", errors="replace")[-4000:] if log_path.exists() else ""
    oom_flag = "out of memory" in log_text.lower() or "cuda oom" in log_text.lower()
    done_flag = "DA3-Streaming done" in log_text or frame_count > 0
    contract_ok = bool(root.exists() and outputs_depth and outputs_pose and outputs_intrinsics and outputs_point_cloud and done_flag and not oom_flag)
    if not root.exists():
        failures.append(_failure(provider_id, "provider_root_missing", "Provider output root does not exist.", "provider_contract"))
    if root.exists() and not outputs_depth:
        failures.append(_failure(provider_id, "depth_outputs_missing", "No results_output/frame_*.npz files found.", "provider_contract"))
    if root.exists() and not outputs_point_cloud:
        failures.append(_failure(provider_id, "point_cloud_missing", "No pcd/combined_pcd.ply found.", "provider_contract"))
    return (
        {
            "schema_version": "stream4d_v102_phase2_provider_contract_row_v1",
            "phase_id": "v102_phase2_provider_ladder_audit",
            "provider_id": provider_id,
            "implementation": "official_DA3_streaming",
            "model_id": model_id,
            "source_root": _rel(root),
            "input_frame_count": 119 if "input119" in root.name else "",
            "chunk_size": 32,
            "subchunk_size": "",
            "overlap": 3,
            "outputs_depth": outputs_depth,
            "outputs_confidence": outputs_depth,
            "outputs_pose": outputs_pose,
            "outputs_intrinsics": outputs_intrinsics,
            "outputs_point_cloud": outputs_point_cloud,
            "outputs_3d_gaussians": False,
            "export_format": "DA3-Streaming results_output npz + pcd ply",
            "OOM_flag": oom_flag,
            "runtime_sec": "",
            "peak_gpu_memory_MB": "",
            "peak_cpu_memory_MB": "",
            "output_file_size_MB": pcd_size,
            "frame_npz_count": frame_count,
            "contract_ok": contract_ok,
            "method_result_allowed": False,
            "contract_note": "Existing v98 provider artifact reused for v102 provider boundary audit; this is not a new v102 method result.",
        },
        failures,
    )


def _failure(provider_id: str, failure_type: str, reason: str, severity: str) -> dict[str, Any]:
    return {
        "schema_version": "stream4d_v102_phase2_provider_failure_row_v1",
        "phase_id": "v102_phase2_provider_ladder_audit",
        "provider_id": provider_id,
        "failure_type": failure_type,
        "reason": reason,
        "severity": severity,
    }


def _inspect_da3_api() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    failures: list[dict[str, Any]] = []
    api_path = DA3_REPO / "src" / "depth_anything_3" / "api.py"
    text = api_path.read_text(encoding="utf-8", errors="replace") if api_path.exists() else ""
    row = {
        "schema_version": "stream4d_v102_phase2_da3_api_row_v1",
        "phase_id": "v102_phase2_provider_ladder_audit",
        "repo_path": _rel(DA3_REPO),
        "api_path": _rel(api_path),
        "repo_exists": DA3_REPO.exists(),
        "api_exists": api_path.exists(),
        "supports_infer_gs": "infer_gs" in text,
        "supports_gs_export_assert": 'if "gs" in export_format' in text and "assert infer_gs" in text,
        "mentions_gs_ply": "gs_ply" in text,
        "mentions_gs_video": "gs_video" in text,
        "supported_export_format_doc": "mini_npz, npz, glb, ply, gs, gs_video" if "Export format (mini_npz, npz, glb, ply, gs, gs_video)" in text else "",
    }
    if not row["repo_exists"] or not row["api_exists"]:
        failures.append(_failure("P6_DA3_GIANT_1_1_3DGS_official", "official_da3_api_missing", "Depth-Anything-3 API not available.", "provider_contract"))
    if row["api_exists"] and not row["supports_infer_gs"]:
        failures.append(_failure("P6_DA3_GIANT_1_1_3DGS_official", "infer_gs_api_missing", "API does not expose infer_gs.", "provider_contract"))
    return row, failures


def _find_cpp_ggufs() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    candidates = list(DA_CPP_REPO.glob("**/*.gguf")) if DA_CPP_REPO.exists() else []
    for path in sorted(candidates):
        lower = path.name.lower()
        if "da3" not in lower and "giant" not in lower:
            continue
        quant = "f16" if "f16" in lower else "q8_0" if "q8" in lower else "q4_k" if "q4" in lower else "unknown"
        rows.append(
            {
                "schema_version": "stream4d_v102_phase2_cpp_provider_row_v1",
                "phase_id": "v102_phase2_provider_ladder_audit",
                "provider_id": f"depth_anything_cpp_{quant}",
                "repo_path": _rel(DA_CPP_REPO),
                "gguf_path": _rel(path),
                "quantization": quant,
                "model_size_MB": _file_size_mb(path),
                "outputs_depth": True,
                "outputs_confidence": False,
                "outputs_pose": "",
                "outputs_3d_gaussians": False,
                "exports_ply_or_glb": False,
                "provider_contract_note": "Static file discovery only; cpp 3DGS export is not claimed.",
            }
        )
    return rows


def _smoke_command(output_dir: Path, model_id_or_path: str, image_paths: list[Path], process_res: int, export_format: str) -> list[str]:
    if export_format in {"gs_ply", "gs_ply_only"}:
        script = f"""
import json
import numpy as np
from pathlib import Path
from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.export.gs import export_to_gs_ply
out = Path({str(output_dir)!r})
out.mkdir(parents=True, exist_ok=True)
model = DepthAnything3.from_pretrained({model_id_or_path!r}).to('cuda').eval()
pred = model.inference(
    {[str(p) for p in image_paths]!r},
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
    'depth_available': getattr(pred, 'depth', None) is not None,
    'gaussians_available': getattr(pred, 'gaussians', None) is not None,
    'export_files': [str(p.relative_to(out)) for p in out.rglob('*') if p.is_file()][:200],
}}
(out / 'smoke_summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\\n')
"""
    else:
        script = f"""
import json
from pathlib import Path
import torch
from depth_anything_3.api import DepthAnything3
out = Path({str(output_dir)!r})
out.mkdir(parents=True, exist_ok=True)
model = DepthAnything3.from_pretrained({model_id_or_path!r}).to('cuda').eval()
pred = model.inference(
    {[str(p) for p in image_paths]!r},
    infer_gs={'True' if 'gs' in export_format else 'False'},
    process_res={int(process_res)},
    export_dir=str(out),
    export_format={export_format!r},
)
summary = {{
    'depth_available': getattr(pred, 'depth', None) is not None,
    'gaussians_available': getattr(pred, 'gaussians', None) is not None,
    'export_files': [str(p.relative_to(out)) for p in out.rglob('*') if p.is_file()][:200],
}}
(out / 'smoke_summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\\n')
"""
    return [
        str(PYTHON),
        "-c",
        script,
    ]


def _maybe_run_smoke() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    run_smoke = os.environ.get("V102_RUN_DA3_GIANT_GS_SMOKE", "0") == "1"
    image_root = V98_PROVIDER / "da3_streaming_d4rt32o3_scene0050_input119"
    image_paths = sorted(image_root.glob("*.png"))[:2] + sorted(image_root.glob("*.jpg"))[:2]
    image_paths = image_paths[:2]
    giant_cache = V98_PROVIDER / "hf_cache" / "models--depth-anything--DA3-GIANT"
    snapshot_paths = sorted((giant_cache / "snapshots").glob("*")) if (giant_cache / "snapshots").exists() else []
    model_to_use = os.environ.get("V102_DA3_GIANT_MODEL", "depth-anything/DA3-GIANT-1.1")
    local_fallback_model = str(snapshot_paths[0]) if snapshot_paths else "depth-anything/DA3-GIANT"
    export_format = os.environ.get("V102_DA3_EXPORT_FORMAT", "gs_ply_only")
    output_dir = OUT_DIR / f"official_da3_giant_smoke2_{export_format}"
    row = {
        "schema_version": "stream4d_v102_phase2_oom_repair_row_v1",
        "phase_id": "v102_phase2_provider_ladder_audit",
        "provider_id": "P6_DA3_GIANT_1_1_3DGS_official",
        "attempt_id": "smoke2_da3_giant_gs_export",
        "attempt_order": 0,
        "model_id_requested": "depth-anything/DA3-GIANT-1.1",
        "model_id_used": model_to_use,
        "local_da3_giant_fallback_model": local_fallback_model,
        "input_frame_count": len(image_paths),
        "process_res": 252,
        "export_format": export_format,
        "run_requested": run_smoke,
        "output_dir": _rel(output_dir),
        "exit_code": "",
        "runtime_sec": "",
        "OOM_flag": False,
        "export_success": False,
        "repair_note": "Set V102_RUN_DA3_GIANT_GS_SMOKE=1 to launch this potentially heavy GPU smoke. This script records the command boundary when not launched.",
    }
    if not run_smoke:
        row["skip_reason"] = "Heavy official DA3-GIANT GS smoke not launched in static audit pass."
        failures.append(
            _failure(
                "P6_DA3_GIANT_1_1_3DGS_official",
                "official_3dgs_smoke_not_run",
                "Static provider audit did not launch heavy GPU GS smoke; Phase3 cannot be promoted from this run.",
                "provider_runtime_required_for_3dgs",
            )
        )
        rows.append(row)
        return rows, failures
    if len(image_paths) < 2:
        row["skip_reason"] = "Not enough input images for smoke2."
        failures.append(_failure("P6_DA3_GIANT_1_1_3DGS_official", "smoke_input_missing", row["skip_reason"], "provider_runtime"))
        rows.append(row)
        return rows, failures

    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = _smoke_command(output_dir, model_to_use, image_paths, process_res=252, export_format=export_format)
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{DA3_REPO / 'src'}:{env.get('PYTHONPATH', '')}"
    env["CUDA_VISIBLE_DEVICES"] = "6"
    t0 = time.time()
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        env=env,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=900,
    )
    runtime = time.time() - t0
    (output_dir / "stdout.log").write_text(proc.stdout, encoding="utf-8")
    (output_dir / "stderr.log").write_text(proc.stderr, encoding="utf-8")
    lower = (proc.stdout + "\n" + proc.stderr).lower()
    exported = [p for p in output_dir.rglob("*") if p.is_file() and p.name not in {"stdout.log", "stderr.log"}]
    row.update(
        {
            "exit_code": proc.returncode,
            "runtime_sec": runtime,
            "OOM_flag": "out of memory" in lower or "cuda" in lower and "memory" in lower,
            "export_success": proc.returncode == 0 and bool(exported),
            "output_file_size_MB": float(sum(p.stat().st_size for p in exported) / (1024 * 1024)),
            "exported_file_count": len(exported),
        }
    )
    if proc.returncode != 0:
        failures.append(
            _failure(
                "P6_DA3_GIANT_1_1_3DGS_official",
                "official_3dgs_smoke_failed_or_oom",
                f"exit_code={proc.returncode}; see {_rel(output_dir / 'stderr.log')}",
                "provider_runtime",
            )
        )
    rows.append(row)
    return rows, failures


def main() -> int:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    phase0 = _read_json(PHASE0_DIR / "summary.json")
    phase1 = _read_json(PHASE1_DIR / "summary.json")
    if not bool(phase0.get("phase0_pass")):
        raise RuntimeError("Phase0 did not pass; refusing Phase2 provider audit.")
    if not bool(phase1.get("phase1_pass")):
        raise RuntimeError("Phase1 did not pass; refusing Phase2 provider audit.")

    provider_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    for provider_id, model_id, root in PROVIDERS:
        row, failures = _provider_contract(provider_id, model_id, root)
        provider_rows.append(row)
        failure_rows.extend(failures)

    api_row, api_failures = _inspect_da3_api()
    failure_rows.extend(api_failures)
    cache_rows = [_cache_row(path) for path in MODEL_CACHE_ROOTS]
    cpp_rows = _find_cpp_ggufs()
    if not any("giant" in str(r.get("gguf_path", "")).lower() for r in cpp_rows):
        failure_rows.append(
            _failure(
                "P7_P8_P9_depth_anything_cpp_DA3_GIANT",
                "cpp_giant_gguf_missing",
                "No local depth-anything.cpp DA3-GIANT f16/q8_0/q4_k GGUF was found.",
                "provider_fallback",
            )
        )

    model_1_1_cache = _first_existing(
        [
            V98_PROVIDER / "hf_cache" / "models--depth-anything--DA3-GIANT-1.1",
            Path.home() / ".cache" / "huggingface" / "hub" / "models--depth-anything--DA3-GIANT-1.1",
        ]
    )
    if model_1_1_cache is None:
        failure_rows.append(
            _failure(
                "P6_DA3_GIANT_1_1_3DGS_official",
                "da3_giant_1_1_model_cache_missing",
                "No local HF cache for depth-anything/DA3-GIANT-1.1 was found; only DA3-GIANT/DA3NESTED-GIANT-LARGE cache roots are present.",
                "provider_model_id_resolution",
            )
        )

    oom_rows, smoke_failures = _maybe_run_smoke()
    failure_rows.extend(smoke_failures)

    provider_contract_ok_count = sum(1 for row in provider_rows if row.get("contract_ok"))
    usable_depth_provider = any(row.get("contract_ok") and row.get("outputs_depth") and row.get("outputs_pose") for row in provider_rows)
    official_3dgs_export_success = any(bool(row.get("export_success")) for row in oom_rows)
    outputs_3dgs = official_3dgs_export_success

    phase2_pass = bool(usable_depth_provider and api_row.get("supports_infer_gs") and not outputs_3dgs is False)
    decision = (
        "PASS_DEPTH_PROVIDER_STATIC_CONTRACT__BLOCK_3DGS_RUNTIME_PROMOTION"
        if usable_depth_provider and not outputs_3dgs
        else "PASS_PROVIDER_AND_3DGS_SMOKE_ENTER_PHASE3"
        if usable_depth_provider and outputs_3dgs
        else "NO_GO_PROVIDER_CONTRACT_BLOCK_PHASE3"
    )

    gate_rows = [
        {
            "gate_id": "at_least_one_usable_depth_pose_provider",
            "pass": usable_depth_provider,
            "expected": ">=1 provider outputs depth and pose/ray without OOM",
            "observed": provider_contract_ok_count,
            "severity": "required_for_phase4_depth_bridge",
        },
        {
            "gate_id": "official_da3_api_supports_infer_gs",
            "pass": bool(api_row.get("supports_infer_gs")),
            "expected": True,
            "observed": api_row.get("supports_infer_gs"),
            "severity": "required_for_3dgs",
        },
        {
            "gate_id": "da3_giant_1_1_model_id_resolved",
            "pass": model_1_1_cache is not None,
            "expected": "local cache or verified model id for depth-anything/DA3-GIANT-1.1",
            "observed": _rel(model_1_1_cache) if model_1_1_cache else "",
            "severity": "required_for_3dgs_promotion",
        },
        {
            "gate_id": "official_3dgs_export_success",
            "pass": official_3dgs_export_success,
            "expected": "Smoke-2 exports gs artifact",
            "observed": official_3dgs_export_success,
            "severity": "required_for_phase3_3dgs_visual_audit",
        },
        {
            "gate_id": "cpp_giant_gguf_available",
            "pass": any("giant" in str(r.get("gguf_path", "")).lower() for r in cpp_rows),
            "expected": "DA3-GIANT f16/q8_0/q4_k GGUF fallback exists",
            "observed": len(cpp_rows),
            "severity": "fallback_diagnostic",
        },
    ]

    provider_csv = OUT_DIR / "provider_contract_rows.csv"
    api_csv = OUT_DIR / "da3_api_rows.csv"
    cache_csv = OUT_DIR / "model_cache_rows.csv"
    cpp_csv = OUT_DIR / "depth_anything_cpp_provider_rows.csv"
    oom_csv = OUT_DIR / "oom_repair_rows.csv"
    gate_csv = OUT_DIR / "variant_gate_rows.csv"
    failure_csv = OUT_DIR / "provider_failure_rows.csv"

    _write_csv(provider_csv, provider_rows)
    _write_csv(api_csv, [api_row])
    _write_csv(cache_csv, cache_rows)
    _write_csv(cpp_csv, cpp_rows)
    _write_csv(oom_csv, oom_rows)
    _write_csv(gate_csv, gate_rows)
    _write_csv(failure_csv, failure_rows)

    summary = {
        "schema_version": "stream4d_v102_phase2_provider_ladder_audit_summary_v1",
        "phase_id": "v102_phase2_provider_ladder_audit",
        "created_unix_time": time.time(),
        "runtime_sec": time.time() - t0,
        "decision": decision,
        "phase2_completed": True,
        "phase2_pass_for_depth_provider_bridge_diagnostic": usable_depth_provider,
        "phase2_pass_for_3dgs_promotion": official_3dgs_export_success and model_1_1_cache is not None,
        "provider_count": len(provider_rows),
        "provider_contract_ok_count": provider_contract_ok_count,
        "usable_depth_provider_exists": usable_depth_provider,
        "official_da3_api_supports_gs": bool(api_row.get("supports_infer_gs") and api_row.get("supports_gs_export_assert")),
        "da3_giant_1_1_cache_exists": model_1_1_cache is not None,
        "official_3dgs_export_success": official_3dgs_export_success,
        "depth_anything_cpp_giant_gguf_count": sum(1 for r in cpp_rows if "giant" in str(r.get("gguf_path", "")).lower()),
        "failure_count": len(failure_rows),
        "truthfulness_note": "This phase does not claim DA3-GIANT-1.1 3DGS success unless an actual gs export succeeds and files exist.",
        "phase1_context": {
            "decision": phase1.get("decision"),
            "repair_candidate_pair_count": phase1.get("repair_candidate_pair_count"),
            "broad_contamination_rate": phase1.get("broad_contamination_rate"),
        },
        "plan_doc": _rel(PLAN_DOC),
        "outputs": {
            "summary": _rel(OUT_DIR / "summary.json"),
            "provider_contract_rows": _rel(provider_csv),
            "da3_api_rows": _rel(api_csv),
            "model_cache_rows": _rel(cache_csv),
            "depth_anything_cpp_provider_rows": _rel(cpp_csv),
            "oom_repair_rows": _rel(oom_csv),
            "variant_gate_rows": _rel(gate_csv),
            "provider_failure_rows": _rel(failure_csv),
        },
    }
    _write_json(OUT_DIR / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if usable_depth_provider else 2


if __name__ == "__main__":
    raise SystemExit(main())
