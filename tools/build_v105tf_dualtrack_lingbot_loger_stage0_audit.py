#!/usr/bin/env python3
"""Build ACL2 v105-TF Stage 0 repo/env audit artifacts.

Stage 0 is an engineering audit, not a method experiment.  It verifies the
LingBot-Map repo path and code surfaces, local checkpoint/data/env availability,
and LoGeR carry-forward artifact readability before any baseline metrics are
claimed.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/acl2_v105tf_dualtrack_lingbot_loger_evidence_eligibility"
STAGE0 = RESULT_ROOT / "stage0_repo_env_audit"
CODE_AUDIT = STAGE0 / "stage0_lingbot_code_audit"
LINGBOT_REPO = ROOT / "third_party/lingbot-map"
PLAN = ROOT / "docs/ACL2_v105TF_DualTrack_LingBotMap_LoGeR_EvidenceEligibility_ExperimentPlan.md"
V104_ROOT = ROOT / "results/acl2_v104tf_strict_provider_evidence_eligibility_state_machine_memory_control"
V103_ROOT = ROOT / "results/acl2_v103tf_semantic_geometric_evidence_eligibility_readswa_ttt_memory_control"


REQUIRED_CODE_FILES = [
    "README.md",
    "benchmark/configs/kitti.yaml",
    "benchmark/configs/methods/lingbot_map.yaml",
    "benchmark/configs/datasets/kitti_504x280.yaml",
    "benchmark/methods/lingbot_map.py",
    "benchmark/datasets/kitti.py",
    "lingbot_map/models/gct_stream.py",
    "lingbot_map/models/gct_stream_window.py",
    "lingbot_map/aggregator/stream.py",
    "lingbot_map/layers/attention.py",
    "lingbot_map/layers/block.py",
    "lingbot_map/layers/flashinfer_cache.py",
]


CODE_POINTER_ROWS = [
    {
        "file": "benchmark/methods/lingbot_map.py",
        "line_hint": 24,
        "symbol": "_resolve_keyframe_interval",
        "observation": "auto keyframe interval resolves to 1 for short sequences else ceil(num_frames/threshold)",
    },
    {
        "file": "benchmark/methods/lingbot_map.py",
        "line_hint": 40,
        "symbol": "LingbotMapMethod.__init__",
        "observation": "method config surface includes mode, use_sdpa, image_size, num_scale_frames, kv cache windows, window_size, overlap_size, keyframe_interval",
    },
    {
        "file": "benchmark/methods/lingbot_map.py",
        "line_hint": 108,
        "symbol": "LingbotMapMethod._load_model",
        "observation": "streaming loads lingbot_map.models.gct_stream.GCTStream; windowed loads lingbot_map.models.gct_stream_window.GCTStream; checkpoint is optional in code but required for valid baseline",
    },
    {
        "file": "benchmark/methods/lingbot_map.py",
        "line_hint": 149,
        "symbol": "LingbotMapMethod._run_inference",
        "observation": "streaming calls inference_streaming(..., num_scale_frames, keyframe_interval, output_device=cpu); windowed calls inference_windowed(..., window_size, overlap_size, num_scale_frames, keyframe_interval)",
    },
    {
        "file": "benchmark/methods/lingbot_map.py",
        "line_hint": 190,
        "symbol": "LingbotMapMethod._process_outputs",
        "observation": "decodes pose_enc to C2W extrinsic/intrinsic, emits depth, pose, intrinsics and optional depth_conf confidence",
    },
    {
        "file": "benchmark/methods/lingbot_map.py",
        "line_hint": 258,
        "symbol": "LingbotMapMethod.process_scene",
        "observation": "loads BSS RGB frames, runs inference, checks output frame count against input count, returns frame/global BSS dict",
    },
    {
        "file": "lingbot_map/models/gct_stream.py",
        "line_hint": 350,
        "symbol": "GCTStream.inference_streaming",
        "observation": "streaming first processes scale frames, then one frame at a time; returns pose_enc plus depth/depth_conf when present",
    },
    {
        "file": "lingbot_map/models/gct_stream.py",
        "line_hint": 300,
        "symbol": "GCTStream._set_skip_append",
        "observation": "non-keyframe mode toggles _skip_append in SDPA dict cache and FlashInfer manager",
    },
    {
        "file": "lingbot_map/models/gct_stream_window.py",
        "line_hint": 959,
        "symbol": "GCTStream.inference_windowed",
        "observation": "window_size counts keyframes; overlap may be specified as actual frames or overlap_keyframes and is used for handoff-style evaluation",
    },
    {
        "file": "lingbot_map/aggregator/stream.py",
        "line_hint": 91,
        "symbol": "AggregatorStream backend selection",
        "observation": "use_sdpa selects SDPABlock; default FlashInferBlock uses paged KV cache",
    },
    {
        "file": "lingbot_map/aggregator/stream.py",
        "line_hint": 207,
        "symbol": "AggregatorStream._get_flashinfer_manager",
        "observation": "lazy FlashInferKVCacheManager receives kv_cache_scale_frames and kv_cache_sliding_window",
    },
    {
        "file": "lingbot_map/layers/flashinfer_cache.py",
        "line_hint": 55,
        "symbol": "FlashInferKVCacheManager",
        "observation": "two-stream paged cache keeps scale patch pages, live window patch pages, and append-only special pages",
    },
    {
        "file": "lingbot_map/layers/flashinfer_cache.py",
        "line_hint": 303,
        "symbol": "get_cache_stats",
        "observation": "can expose scale/live/special page counts for provenance; compute_attention returns attention output, not weights",
    },
    {
        "file": "lingbot_map/layers/attention.py",
        "line_hint": 562,
        "symbol": "SDPAAttention",
        "observation": "SDPA path uses torch scaled_dot_product_attention and does not directly return attention weights; trace needs instrumentation or offline QK recompute",
    },
    {
        "file": "benchmark/datasets/kitti.py",
        "line_hint": 1,
        "symbol": "KittiDataset",
        "observation": "expects raw_data_root with poses/ and sequences/; reads image_2 RGB, P2 intrinsics, C2W cam_0 poses; cam0/cam2 offset absorbed by Sim3 evaluation",
    },
]


CONFIG_SURFACE_ROWS = [
    {
        "surface": "workspace",
        "source": "benchmark/configs/kitti.yaml",
        "default_or_required": "/path/to/workspace/kitti placeholder",
        "stage0_status": "must be rewritten to v105 result workspace before Stage1",
    },
    {
        "surface": "raw_data_root",
        "source": "benchmark/configs/datasets/kitti_504x280.yaml",
        "default_or_required": "/path/to/kitti/dataset placeholder",
        "stage0_status": "requires local KITTI odometry root with poses/ and sequences/",
    },
    {
        "surface": "_checkpoint",
        "source": "benchmark/configs/methods/lingbot_map.yaml",
        "default_or_required": "/path/to/lingbot-map.pt placeholder",
        "stage0_status": "requires lingbot-map-long.pt or equivalent released checkpoint",
    },
    {
        "surface": "env",
        "source": "benchmark/configs/methods/lingbot_map.yaml",
        "default_or_required": "lingbot-map",
        "stage0_status": "benchmark run_worker expects this conda env; current shell must be able to resolve conda or equivalent launcher",
    },
    {
        "surface": "_mode",
        "source": "benchmark/configs/methods/lingbot_map.yaml",
        "default_or_required": "streaming",
        "stage0_status": "Stage1 first priority is streaming default",
    },
    {
        "surface": "_use_sdpa",
        "source": "benchmark/configs/methods/lingbot_map.yaml",
        "default_or_required": "false",
        "stage0_status": "fail-forward should set true for debug if FlashInfer/env fails",
    },
    {
        "surface": "_num_scale_frames",
        "source": "benchmark/configs/methods/lingbot_map.yaml",
        "default_or_required": "8",
        "stage0_status": "maps to inference_streaming/inference_windowed num_scale_frames",
    },
    {
        "surface": "_keyframe_interval",
        "source": "benchmark/configs/methods/lingbot_map.yaml",
        "default_or_required": "auto",
        "stage0_status": "resolved by _resolve_keyframe_interval; sensitivity settings kf1/kf4 required in Stage1",
    },
    {
        "surface": "_kv_cache_sliding_window",
        "source": "benchmark/configs/methods/lingbot_map.yaml",
        "default_or_required": "64",
        "stage0_status": "controls live_window_patch_pages retention in FlashInferKVCacheManager",
    },
    {
        "surface": "_kv_cache_scale_frames",
        "source": "benchmark/configs/methods/lingbot_map.yaml",
        "default_or_required": "8",
        "stage0_status": "controls scale_patch_pages retention in FlashInferKVCacheManager",
    },
]


def run_text(args: list[str], *, cwd: Path = ROOT, timeout: int = 20) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            args,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
        return {"returncode": proc.returncode, "stdout": proc.stdout.strip()}
    except Exception as exc:  # noqa: BLE001
        return {"returncode": None, "stdout": f"{type(exc).__name__}: {exc}"}


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        seen: list[str] = []
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.append(key)
        fieldnames = seen
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def repo_resolution() -> dict[str, Any]:
    candidates = [
        ROOT / "third_party/lingbot-map",
        ROOT / "third_party/lingbot-map-main",
        ROOT / "third_party/lingbot_map",
    ]
    method_hits = list(ROOT.glob("**/benchmark/methods/lingbot_map.py"))
    return {
        "resolved_repo": str(LINGBOT_REPO.relative_to(ROOT)) if LINGBOT_REPO.is_dir() else None,
        "primary_exists": LINGBOT_REPO.is_dir(),
        "candidate_paths": [
            {"path": str(path.relative_to(ROOT)), "exists": path.exists()} for path in candidates
        ],
        "method_hits": [str(path.relative_to(ROOT)) for path in method_hits[:20]],
    }


def required_file_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rel_path in REQUIRED_CODE_FILES:
        path = LINGBOT_REPO / rel_path
        rows.append(
            {
                "path": f"third_party/lingbot-map/{rel_path}",
                "exists": path.is_file(),
                "size_bytes": path.stat().st_size if path.is_file() else "",
            }
        )
    return rows


def current_python_imports() -> dict[str, bool]:
    return {name: importlib.util.find_spec(name) is not None for name in ["torch", "flashinfer", "lingbot_map"]}


def conda_state() -> dict[str, Any]:
    conda_candidates = [
        shutil.which("conda"),
        "/mnt/data/users/chengshun.wang/miniconda3/bin/conda",
        "/mnt/data/users/chengshun.wang/miniconda3/condabin/conda",
    ]
    conda_path = next((item for item in conda_candidates if item and Path(item).is_file()), None)
    mamba_path = shutil.which("mamba")
    micromamba_path = shutil.which("micromamba")
    state: dict[str, Any] = {
        "conda": conda_path,
        "mamba": mamba_path,
        "micromamba": micromamba_path,
        "lingbot_map_env_found": False,
        "viable_existing_envs": [],
        "env_probe_rows": [],
    }
    if conda_path:
        out = run_text([conda_path, "env", "list"], timeout=30)
        state["conda_env_list_returncode"] = out["returncode"]
        state["conda_env_list_stdout"] = out["stdout"]
        state["lingbot_map_env_found"] = "lingbot-map" in out["stdout"] or "lingbot_map" in out["stdout"]
        py_path = f"{(LINGBOT_REPO).as_posix()}:{(LINGBOT_REPO / 'benchmark').as_posix()}"
        probe_code = (
            "import importlib.util,json,sys;"
            "mods=['torch','torchvision','flashinfer','lingbot_map','benchmark'];"
            "print(json.dumps({'python':sys.version.split()[0],"
            "'imports':{m:(importlib.util.find_spec(m) is not None) for m in mods}}))"
        )
        for env_name in ["lingbot-map", "lingbot_map", "loger", "infinitevggt", "4D", "sem2", "sem"]:
            proc_env = os.environ.copy()
            proc_env["PYTHONPATH"] = py_path + (":" + proc_env["PYTHONPATH"] if proc_env.get("PYTHONPATH") else "")
            try:
                proc = subprocess.run(
                    [conda_path, "run", "-n", env_name, "python", "-c", probe_code],
                    cwd=ROOT,
                    env=proc_env,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    timeout=20,
                    check=False,
                )
                raw = proc.stdout.strip()
                parsed: dict[str, Any] = {}
                for line in raw.splitlines():
                    if line.startswith("{") and line.endswith("}"):
                        parsed = json.loads(line)
                imports = parsed.get("imports", {})
                viable = bool(imports.get("torch") and imports.get("torchvision") and imports.get("lingbot_map") and imports.get("benchmark"))
                row = {
                    "env": env_name,
                    "returncode": proc.returncode,
                    "python": parsed.get("python", ""),
                    "torch": imports.get("torch", False),
                    "torchvision": imports.get("torchvision", False),
                    "flashinfer": imports.get("flashinfer", False),
                    "lingbot_map": imports.get("lingbot_map", False),
                    "benchmark": imports.get("benchmark", False),
                    "viable_with_local_pythonpath": viable,
                    "stdout": raw,
                }
            except Exception as exc:  # noqa: BLE001
                row = {
                    "env": env_name,
                    "returncode": None,
                    "python": "",
                    "torch": False,
                    "torchvision": False,
                    "flashinfer": False,
                    "lingbot_map": False,
                    "benchmark": False,
                    "viable_with_local_pythonpath": False,
                    "stdout": f"{type(exc).__name__}: {exc}",
                }
            state["env_probe_rows"].append(row)
            if row["viable_with_local_pythonpath"]:
                state["viable_existing_envs"].append(env_name)
        state["stage1_env_resolved"] = bool(state["lingbot_map_env_found"] or state["viable_existing_envs"])
        state["recommended_env"] = (
            "lingbot-map"
            if state["lingbot_map_env_found"]
            else (state["viable_existing_envs"][0] if state["viable_existing_envs"] else "")
        )
        state["recommended_pythonpath"] = py_path if state["recommended_env"] and not state["lingbot_map_env_found"] else ""
        state["flashinfer_available_in_recommended_env"] = next(
            (row["flashinfer"] for row in state["env_probe_rows"] if row["env"] == state["recommended_env"]),
            False,
        )
    else:
        state["stage1_env_resolved"] = False
        state["recommended_env"] = ""
        state["recommended_pythonpath"] = ""
        state["flashinfer_available_in_recommended_env"] = False
    return state


def gpu_state() -> dict[str, Any]:
    out = run_text(["nvidia-smi", "-L"], timeout=10)
    lines = [line for line in out["stdout"].splitlines() if line.startswith("GPU ")]
    return {
        "nvidia_smi_returncode": out["returncode"],
        "gpu_lines": lines,
        "gpu_count": len(lines),
        "requested_gpus_0_5_visible": all(any(line.startswith(f"GPU {idx}:") for line in lines) for idx in range(6)),
    }


def search_checkpoints() -> dict[str, Any]:
    env_path = os.environ.get("ACL2_LINGBOT_CKPT", "")
    roots = [
        LINGBOT_REPO,
        Path("/mnt/data/users/chengshun.wang/.cache/huggingface/hub"),
        Path("/mnt/data/users/chengshun.wang/.cache/modelscope/hub"),
        Path("/mnt/data/users/chengshun.wang/checkpoints"),
        ROOT / "checkpoints",
    ]
    patterns = ("lingbot-map-long.pt", "lingbot-map.pt", "lingbot_map_long.pt", "lingbot_map.pt")
    hits: list[str] = []
    scanned: list[dict[str, Any]] = []
    for root in roots:
        start = time.time()
        file_count = 0
        hit_count_before = len(hits)
        if root.exists():
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [d for d in dirnames if d not in {".git", "__pycache__", "code_audit_pack", "results"}]
                for filename in filenames:
                    file_count += 1
                    lower = filename.lower()
                    if lower in patterns or ("lingbot" in lower and "map" in lower and lower.endswith((".pt", ".pth", ".ckpt"))):
                        hits.append(str((Path(dirpath) / filename).resolve()))
                if time.time() - start > 8:
                    break
        scanned.append(
            {
                "root": str(root),
                "exists": root.exists(),
                "files_seen_limited": file_count,
                "hits_added": len(hits) - hit_count_before,
                "scan_seconds": round(time.time() - start, 3),
            }
        )
    resolved = env_path if env_path and Path(env_path).is_file() else (hits[0] if hits else "")
    return {
        "env_ACL2_LINGBOT_CKPT": env_path,
        "env_path_exists": bool(env_path and Path(env_path).is_file()),
        "search_roots": scanned,
        "hits": hits,
        "resolved_checkpoint": resolved,
        "checkpoint_resolved": bool(resolved),
    }


def check_kitti_root(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "sequences_00_image_2": (path / "sequences/00/image_2").is_dir(),
        "poses_00": (path / "poses/00.txt").is_file(),
        "seq00_times": (path / "sequences/00/times.txt").is_file(),
        "seq00_calib": (path / "sequences/00/calib.txt").is_file(),
    }


def search_kitti() -> dict[str, Any]:
    env_candidates = [os.environ.get(name, "") for name in ["KITTI_ROOT", "KITTIRAW", "DATA_ROOT"]]
    roots = [
        *(Path(item) for item in env_candidates if item),
        ROOT / "data/kitti/dataset",
        ROOT / "data/kitti",
        ROOT / "datasets/kitti/dataset",
        ROOT / "datasets/kitti",
        Path("/mnt/data/users/chengshun.wang/data/kitti/dataset"),
        Path("/mnt/data/users/chengshun.wang/data/kitti"),
        Path("/mnt/data/users/chengshun.wang/datasets/kitti/dataset"),
        Path("/mnt/data/users/chengshun.wang/datasets/kitti"),
        Path("/mnt/data/datasets/kitti/dataset"),
        Path("/mnt/data/datasets/kitti"),
    ]
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for root in roots:
        key = str(root)
        if key in seen:
            continue
        seen.add(key)
        rows.append(check_kitti_root(root))
    resolved = next(
        (row["path"] for row in rows if row["sequences_00_image_2"] and row["poses_00"] and row["seq00_times"] and row["seq00_calib"]),
        "",
    )
    return {
        "env": {"KITTI_ROOT": os.environ.get("KITTI_ROOT", ""), "KITTIRAW": os.environ.get("KITTIRAW", ""), "DATA_ROOT": os.environ.get("DATA_ROOT", "")},
        "candidate_rows": rows,
        "resolved_kitti_root": resolved,
        "kitti_resolved": bool(resolved),
        "wide_search_note": "A prior manual wide find over /mnt/data/users/chengshun.wang was interrupted after about 90s with no output; Stage0 script uses bounded candidate checks.",
    }


def write_code_audit_report(summary: dict[str, Any]) -> None:
    checkpoint = summary["checkpoint"]
    kitti = summary["kitti"]
    env = summary["environment"]
    text = f"""# Stage 0 LingBot Code/Environment Audit

This is an engineering audit for ACL2 v105-TF. It is not a scientific No-Go.

## Repository Resolution

- primary_repo: `third_party/lingbot-map`
- primary_exists: `{summary["repo_resolution"]["primary_exists"]}`
- method_hits: `{summary["repo_resolution"]["method_hits"]}`

## Required File Availability

- required_files_present: `{summary["required_files_present"]}`
- required_file_count: `{len(summary["required_files"])}`

## Interface Answers Required by Plan

1. Streaming and windowed entry functions:
   - Benchmark adapter: `benchmark/methods/lingbot_map.py::LingbotMapMethod._run_inference`.
   - Streaming path calls `GCTStream.inference_streaming(...)` from `lingbot_map/models/gct_stream.py`.
   - Windowed path calls `GCTStream.inference_windowed(...)` from `lingbot_map/models/gct_stream_window.py`.

2. `num_scale_frames` path:
   - Method config `_num_scale_frames` is stored on `LingbotMapMethod`.
   - `_run_inference` passes it to `inference_streaming` or `inference_windowed`.
   - Streaming code converts it to `scale_frames` and processes those initial frames as scale/reference frames.

3. `keyframe_interval` path:
   - `_resolve_keyframe_interval` handles `auto`, `0`, `None`, or explicit ints.
   - Streaming marks frames after scale frames as keyframes when `(i - scale_frames) % keyframe_interval == 0`.
   - Non-keyframes use `_set_skip_append`, toggling cache append behavior.

4. `kv_cache_sliding_window` and `kv_cache_scale_frames`:
   - `AggregatorStream` forwards both into `FlashInferKVCacheManager`.
   - `kv_cache_scale_frames` controls persistent `scale_patch_pages`.
   - `kv_cache_sliding_window` controls evicted/recycled `live_window_patch_pages`.
   - Special tokens are append-only in `all_special_pages`.

5. Camera/register/scale special tokens:
   - `AggregatorStream._setup_special_tokens` creates camera, optional register, and scale tokens.
   - `patch_start_idx = 1 + num_register_tokens + 1`, so specials precede patch tokens.
   - FlashInfer cache writes special K/V separately from patch-page K/V.

6. Attention weights / trace route:
   - FlashInfer `compute_attention` returns attention output, not attention weights.
   - SDPA path uses `torch.nn.functional.scaled_dot_product_attention`, which also returns output only.
   - First trace implementation should therefore use SDPA instrumentation or offline QK recompute from captured Q/K/cache provenance, matching the v105 plan.

7. Benchmark output format:
   - `process_scene` returns `{{"frame": {{"rgb", "depth", "pose", "intrinsics", optional "confidence"}}, "global": {{}}}}`.
   - `pose_enc` is decoded to C2W extrinsic/intrinsic; depth is emitted as float32 frames; confidence comes from `depth_conf`.

8. KITTI loader pose convention / image_2-cam0 offset:
   - `benchmark/datasets/kitti.py` expects `raw_data_root/poses` and `raw_data_root/sequences`.
   - It reads `image_2` RGB but uses cam0 C2W poses unchanged.
   - The code comments state the cam0/cam2 rectified rigid offset is absorbed by Sim(3) trajectory alignment.

## Environment Snapshot

- checkpoint_resolved: `{checkpoint["checkpoint_resolved"]}`
- resolved_checkpoint: `{checkpoint["resolved_checkpoint"]}`
- kitti_resolved: `{kitti["kitti_resolved"]}`
- resolved_kitti_root: `{kitti["resolved_kitti_root"]}`
- conda_path: `{env["conda"].get("conda")}`
- lingbot_map_env_found: `{env["conda"].get("lingbot_map_env_found")}`
- stage1_env_resolved: `{env["conda"].get("stage1_env_resolved")}`
- recommended_env: `{env["conda"].get("recommended_env")}`
- recommended_pythonpath: `{env["conda"].get("recommended_pythonpath")}`
- flashinfer_available_in_recommended_env: `{env["conda"].get("flashinfer_available_in_recommended_env")}`
- current_python_imports: `{env["python_imports"]}`
- requested_gpus_0_5_visible: `{env["gpu"].get("requested_gpus_0_5_visible")}`

## Stage 0 Status

- stage0_engineering_partial_pass: `{summary["stage0_engineering_partial_pass"]}`
- stage1_baseline_allowed: `{summary["stage1_baseline_allowed"]}`
- blocker_classes: `{summary["blocker_classes"]}`

"""
    (CODE_AUDIT / "code_audit_report.md").write_text(text, encoding="utf-8")


def write_blocker_docs(summary: dict[str, Any]) -> None:
    for stale in ["CHECKPOINT_MISSING.md", "KITTI_DATASET_MISSING.md", "ENV_MISSING.md"]:
        path = STAGE0 / stale
        if path.exists():
            path.unlink()
    ckpt = summary["checkpoint"]
    if not ckpt["checkpoint_resolved"]:
        (STAGE0 / "CHECKPOINT_MISSING.md").write_text(
            """# CHECKPOINT_MISSING

Stage 0 did not resolve a LingBot checkpoint. This blocks Stage 1 baseline metrics.

Searched:
"""
            + "\n".join(f"- `{row['root']}` exists={row['exists']} files_seen_limited={row['files_seen_limited']} hits_added={row['hits_added']}" for row in ckpt["search_roots"])
            + f"""

Environment:
- ACL2_LINGBOT_CKPT=`{ckpt['env_ACL2_LINGBOT_CKPT']}`
- env path exists=`{ckpt['env_path_exists']}`

Plan-backed repair direction:
1. Set `ACL2_LINGBOT_CKPT` to an existing released checkpoint, preferably `lingbot-map-long.pt`.
2. Or place `lingbot-map-long.pt` under one searched checkpoint/cache root and rerun:
   `python3 tools/build_v105tf_dualtrack_lingbot_loger_stage0_audit.py`

No baseline or scientific No-Go is allowed while this is unresolved.
""",
            encoding="utf-8",
        )
    kitti = summary["kitti"]
    if not kitti["kitti_resolved"]:
        rows = "\n".join(
            f"- `{row['path']}` exists={row['exists']} image_2={row['sequences_00_image_2']} poses_00={row['poses_00']} times={row['seq00_times']} calib={row['seq00_calib']}"
            for row in kitti["candidate_rows"]
        )
        (STAGE0 / "KITTI_DATASET_MISSING.md").write_text(
            f"""# KITTI_DATASET_MISSING

Stage 0 did not resolve a runnable KITTI odometry root. This blocks Stage 1 baseline metrics.

Required structure:
- `raw_data_root/poses/00.txt`
- `raw_data_root/sequences/00/image_2/`
- `raw_data_root/sequences/00/times.txt`
- `raw_data_root/sequences/00/calib.txt`

Bounded candidate checks:
{rows}

Environment:
- KITTI_ROOT=`{kitti['env']['KITTI_ROOT']}`
- KITTIRAW=`{kitti['env']['KITTIRAW']}`
- DATA_ROOT=`{kitti['env']['DATA_ROOT']}`

Plan-backed repair direction:
1. Point `KITTI_ROOT` at the KITTI odometry dataset root.
2. Rerun Stage 0 audit.
3. Only then generate and run Stage 1 LingBot configs.

Note: {kitti['wide_search_note']}
""",
            encoding="utf-8",
        )
    env = summary["environment"]
    if not env["conda"].get("stage1_env_resolved"):
        (STAGE0 / "ENV_MISSING.md").write_text(
            f"""# ENV_MISSING

Stage 0 did not resolve a runnable LingBot environment from the current shell.

Observed:
- conda=`{env['conda'].get('conda')}`
- mamba=`{env['conda'].get('mamba')}`
- micromamba=`{env['conda'].get('micromamba')}`
- lingbot_map_env_found=`{env['conda'].get('lingbot_map_env_found')}`
- stage1_env_resolved=`{env['conda'].get('stage1_env_resolved')}`
- viable_existing_envs=`{env['conda'].get('viable_existing_envs')}`
- current_python_imports={json.dumps(env['python_imports'], sort_keys=True)}

Plan-backed repair direction:
1. Make the `lingbot-map` conda env visible to this shell, or provide the correct launcher.
2. If FlashInfer is unavailable, first use `_use_sdpa: true` for debug/parity.
3. Do not classify LingBot scientifically until env/checkpoint/data are resolved.
""",
            encoding="utf-8",
        )


def build_summary() -> dict[str, Any]:
    repo = repo_resolution()
    required = required_file_rows()
    checkpoint = search_checkpoints()
    kitti = search_kitti()
    environment = {
        "python": run_text(["python3", "--version"], timeout=5),
        "python_imports": current_python_imports(),
        "conda": conda_state(),
        "gpu": gpu_state(),
        "env_vars": {
            "ACL2_LINGBOT_CKPT": os.environ.get("ACL2_LINGBOT_CKPT", ""),
            "KITTI_ROOT": os.environ.get("KITTI_ROOT", ""),
            "KITTIRAW": os.environ.get("KITTIRAW", ""),
            "DATA_ROOT": os.environ.get("DATA_ROOT", ""),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        },
    }
    required_present = all(row["exists"] for row in required)
    blocker_classes: list[str] = []
    if not checkpoint["checkpoint_resolved"]:
        blocker_classes.append("checkpoint_missing")
    if not kitti["kitti_resolved"]:
        blocker_classes.append("dataset_path_error")
    if not environment["conda"].get("stage1_env_resolved"):
        blocker_classes.append("env_missing")
    if not environment["conda"].get("stage1_env_resolved") and not environment["python_imports"].get("torch"):
        blocker_classes.append("current_python_torch_missing")
    summary = {
        "schema": "acl2_v105tf_stage0_repo_env_audit_v1",
        "plan": str(PLAN.relative_to(ROOT)),
        "repo_resolution": repo,
        "required_files": required,
        "required_files_present": required_present,
        "checkpoint": checkpoint,
        "kitti": kitti,
        "environment": environment,
        "loger_comparison_artifacts": {
            "v104_root": str(V104_ROOT.relative_to(ROOT)),
            "v104_root_readable": V104_ROOT.is_dir(),
            "v103_root": str(V103_ROOT.relative_to(ROOT)),
            "v103_root_readable": V103_ROOT.is_dir(),
        },
        "code_audit_report_written": True,
        "stage0_engineering_partial_pass": bool(repo["primary_exists"] and required_present and (V104_ROOT.is_dir() or V103_ROOT.is_dir())),
        "stage1_baseline_allowed": bool(repo["primary_exists"] and required_present and checkpoint["checkpoint_resolved"] and kitti["kitti_resolved"] and environment["conda"].get("stage1_env_resolved")),
        "blocker_classes": blocker_classes,
        "scientific_no_go_allowed": False,
    }
    return summary


def main() -> int:
    CODE_AUDIT.mkdir(parents=True, exist_ok=True)
    summary = build_summary()
    write_csv(CODE_AUDIT / "code_pointer_rows.csv", CODE_POINTER_ROWS)
    write_csv(CODE_AUDIT / "config_surface_rows.csv", CONFIG_SURFACE_ROWS)
    write_csv(CODE_AUDIT / "required_file_rows.csv", summary["required_files"])
    write_csv(STAGE0 / "kitti_candidate_rows.csv", summary["kitti"]["candidate_rows"])
    write_csv(STAGE0 / "checkpoint_search_rows.csv", summary["checkpoint"]["search_roots"])
    write_csv(STAGE0 / "conda_env_probe_rows.csv", summary["environment"]["conda"].get("env_probe_rows", []))
    repo_text = "# repo_path_resolution\n\n" + json.dumps(summary["repo_resolution"], ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    (STAGE0 / "repo_path_resolution.md").write_text(repo_text, encoding="utf-8")
    unsupported = [
        "FlashInfer/SDPA attention weights are not directly returned by current code; use SDPA instrumentation or offline QK recompute.",
        "Stage1 baseline is not allowed until checkpoint, KITTI root, and LingBot env are resolved.",
    ]
    if summary["blocker_classes"]:
        unsupported.append("Current blocker classes: " + ", ".join(summary["blocker_classes"]))
    (CODE_AUDIT / "unsupported_or_missing_interfaces.md").write_text(
        "# Unsupported Or Missing Interfaces\n\n" + "\n".join(f"- {item}" for item in unsupported) + "\n",
        encoding="utf-8",
    )
    write_code_audit_report(summary)
    write_blocker_docs(summary)
    write_json(STAGE0 / "stage0_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
