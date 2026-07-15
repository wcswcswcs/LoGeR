#!/usr/bin/env python3
"""Build chunk-windowed v105 P6 outputs and assemble scene-level videos.

This is an audit/repair helper, not the v105 primary pipeline runner.  It
executes the already frozen holdout Phase6 chain on missing stride windows:

  X1 gapadaptive SAM2.1-L -> AllTracker proxy -> Phase6 P6 candidate

The final scene videos are complete over the selected stride frames, but the
internal computation is chunk-windowed and does not claim continuous scene-level
tracking identity across chunk boundaries.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import queue
import shlex
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python")
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "Stream3D/outputs/audit/v105_fullscene_multichunk_repair_20260711"
DEFAULT_CONFIG = REPO_ROOT / "configs/v105/baseline_chunk_table/baseline_x_gapadaptive_sam2.generated.yaml"
DEFAULT_X0_ROOT = REPO_ROOT / "Stream3D/outputs/audit"
DEFAULT_FIRST64_TEMPLATE = "v105_specgap_holdout_phase6_period4_force50k_max1_64_{scene_short}_20260711"
DEFAULT_VARIANT_ID = "P6_period4_force50k_max1_multichunk_windowed_v1"
DEFAULT_X0_SUMMARY_TEMPLATE = (
    "Stream3D/outputs/audit/v105_specgap_holdout_x0_64_{scene_short}_20260711/"
    "baseline_x_sam2_twostage_sam2/summary.json"
)
DEFAULT_FIRST_PREFIX_LABEL_ROOT_TEMPLATE = (
    "Stream3D/outputs/audit/" + DEFAULT_FIRST64_TEMPLATE + "/labels"
)
_BLEND_LUT_CACHE: dict[float, np.ndarray] = {}


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return p.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except Exception:
        return p.as_posix()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any], lock: threading.Lock) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    with lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def numeric_stem(path: Path) -> int:
    try:
        return int(path.stem)
    except Exception:
        return 10**15


def scene_short(scene_id: str) -> str:
    return scene_id.split("_")[0]


def format_scene_template(value: str, scene_id: str) -> str:
    return str(value).format(scene_id=scene_id, scene_short=scene_short(scene_id))


def format_chunk_template(value: str, scene_id: str, frame_start: int, frame_count: int) -> str:
    return str(value).format(
        scene_id=scene_id,
        scene_short=scene_short(scene_id),
        frame_start=int(frame_start),
        frame_count=int(frame_count),
    )


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def selected_frame_ids(scene_id: str, stride: int) -> list[int]:
    color_dir = REPO_ROOT / "Stream3D/data/scannet/processed" / scene_id / "color"
    files = sorted(color_dir.glob("*.jpg"), key=numeric_stem)
    if not files:
        files = sorted(color_dir.glob("*.png"), key=numeric_stem)
    ids = [numeric_stem(path) for path in files if numeric_stem(path) < 10**12]
    return ids[:: max(int(stride), 1)]


def chunk_specs(frame_ids: list[int], *, start_index: int, chunk_size: int) -> list[dict[str, int]]:
    specs: list[dict[str, int]] = []
    idx = int(start_index)
    while idx < len(frame_ids):
        count = min(int(chunk_size), len(frame_ids) - idx)
        if count <= 0:
            break
        specs.append(
            {
                "start_index": int(idx),
                "frame_start": int(frame_ids[idx]),
                "frame_count": int(count),
            }
        )
        idx += int(chunk_size)
    return specs


def run_command(
    *,
    cmd: list[str],
    log_path: Path,
    env: dict[str, str],
    timeout_sec: int,
    step_name: str,
    manifest_path: Path,
    manifest_lock: threading.Lock,
    context: dict[str, Any],
) -> bool:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    append_jsonl(
        manifest_path,
        {
            "event": "step_start",
            "step_name": step_name,
            "cmd": cmd,
            "log_path": rel(log_path),
            "timeout_sec": int(timeout_sec),
            **context,
        },
        manifest_lock,
    )
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=timeout_sec,
            check=False,
        )
    runtime = time.time() - start
    ok = proc.returncode == 0
    append_jsonl(
        manifest_path,
        {
            "event": "step_end",
            "step_name": step_name,
            "returncode": int(proc.returncode),
            "runtime_sec": float(runtime),
            "ok": bool(ok),
            **context,
        },
        manifest_lock,
    )
    print(
        json.dumps(
            {
                "event": "step_end",
                "step": step_name,
                "ok": ok,
                "runtime_sec": round(runtime, 3),
                **context,
            },
            ensure_ascii=True,
        ),
        flush=True,
    )
    return ok


def run_chunk_job(
    *,
    gpu: str,
    scene_id: str,
    frame_start: int,
    frame_count: int,
    output_root: Path,
    manifest_path: Path,
    manifest_lock: threading.Lock,
    timeout_x1: int,
    timeout_alltracker: int,
    timeout_phase6: int,
    choice_policy: str,
    min_component_area: int,
    max_points_per_frame: int,
    base_points_per_component: int,
    area_per_extra_point: int,
    max_points_per_component: int,
    min_candidate_touch_area: int,
    min_candidate_touch_ratio: float,
    max_births_per_frame: int,
    birth_anchor_period: int,
    birth_anchor_offset: int,
    birth_anchor_force_candidate_area_thresh: int,
    birth_anchor_force_max_events: int,
    video_gpu_hot_window: int,
    reuse_video_state_template: bool,
    skip_phase6_chunk_visuals: bool,
    skip_phase6_candidate: bool,
    seed_source: str,
    x0_summary: str,
    x0_summary_template: str,
    reuse_x1_root_template: str,
    reuse_alltracker_root_template: str,
    allow_missing_x1_diagnostics: bool,
    phase6_extra_args: list[str],
) -> dict[str, Any]:
    short = scene_short(scene_id)
    suffix = f"{short}_start{int(frame_start):04d}_f{int(frame_count)}"
    x1_frame_count = int(frame_count) if str(seed_source) == "full_x1" else 1
    x0_summary_value = str(x0_summary).strip() or format_scene_template(str(x0_summary_template), scene_id)
    x0_summary_path = resolve_repo_path(x0_summary_value)
    context = {
        "scene_id": scene_id,
        "scene_short": short,
        "frame_start": int(frame_start),
        "frame_count": int(frame_count),
        "x1_frame_count": int(x1_frame_count),
        "gpu": str(gpu),
        "seed_source": str(seed_source),
        "x0_summary": rel(x0_summary_path),
        "allow_missing_x1_diagnostics": bool(allow_missing_x1_diagnostics),
        "phase6_extra_args": list(phase6_extra_args),
        "choice_policy": str(choice_policy),
        "min_component_area": int(min_component_area),
        "max_points_per_frame": int(max_points_per_frame),
        "base_points_per_component": int(base_points_per_component),
        "area_per_extra_point": int(area_per_extra_point),
        "max_points_per_component": int(max_points_per_component),
        "min_candidate_touch_area": int(min_candidate_touch_area),
        "min_candidate_touch_ratio": float(min_candidate_touch_ratio),
        "max_births_per_frame": int(max_births_per_frame),
        "birth_anchor_period": int(birth_anchor_period),
        "birth_anchor_offset": int(birth_anchor_offset),
        "birth_anchor_force_candidate_area_thresh": int(birth_anchor_force_candidate_area_thresh),
        "birth_anchor_force_max_events": int(birth_anchor_force_max_events),
        "video_gpu_hot_window": int(video_gpu_hot_window),
        "reuse_video_state_template": bool(reuse_video_state_template),
        "skip_phase6_chunk_visuals": bool(skip_phase6_chunk_visuals),
        "skip_phase6_candidate": bool(skip_phase6_candidate),
    }
    log_dir = output_root / "run_logs"
    x1_prefix = "x1" if str(seed_source) == "full_x1" else "x1seed"
    default_x1_root = output_root / f"{x1_prefix}_{suffix}"
    default_alltracker_root = output_root / f"alltracker_{suffix}"
    x1_root = default_x1_root
    alltracker_root = default_alltracker_root
    x1_reuse_source: str | None = None
    alltracker_reuse_source: str | None = None
    if str(reuse_x1_root_template).strip():
        candidate = resolve_repo_path(format_chunk_template(str(reuse_x1_root_template), scene_id, frame_start, frame_count))
        if (candidate / "baseline_x_gapadaptive_sam2" / "summary.json").exists() and (candidate / "birth_bank" / "birth_records.json").exists():
            x1_root = candidate
            x1_reuse_source = rel(candidate)
        else:
            context["reuse_x1_root_missing"] = rel(candidate)
    if str(reuse_alltracker_root_template).strip():
        candidate = resolve_repo_path(
            format_chunk_template(str(reuse_alltracker_root_template), scene_id, frame_start, frame_count)
        )
        if (candidate / "alltracker_contract_summary.json").exists():
            alltracker_root = candidate
            alltracker_reuse_source = rel(candidate)
        else:
            context["reuse_alltracker_root_missing"] = rel(candidate)
    phase6_root = output_root / f"phase6_{suffix}"
    x1_summary = x1_root / "baseline_x_gapadaptive_sam2" / "summary.json"
    x1_labels = x1_root / "baseline_x_gapadaptive_sam2" / "labels"
    birth_records = x1_root / "birth_bank" / "birth_records.json"
    alltracker_summary = alltracker_root / "alltracker_contract_summary.json"
    phase6_summary = phase6_root / "phase6_speculative_gap_birth_summary.json"
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "PYTHONUNBUFFERED": "1",
        }
    )
    if not x1_summary.exists() or not birth_records.exists():
        ok = run_command(
            cmd=[
                str(PYTHON),
                "tools/audit_v105_baseline_x_sam2_twostage_tracking.py",
                "--config",
                rel(DEFAULT_CONFIG),
                "--scene-id",
                scene_id,
                "--frame-start",
                str(int(frame_start)),
                "--frame-count",
                str(int(x1_frame_count)),
                "--output-root",
                rel(x1_root),
                "--birth-dump-dir",
                rel(x1_root / "birth_bank"),
            ],
            log_path=log_dir / f"x1_{suffix}.log",
            env=env,
            timeout_sec=timeout_x1,
            step_name="x1_gapadaptive",
            manifest_path=manifest_path,
            manifest_lock=manifest_lock,
            context=context,
        )
        if not ok:
            return {**context, "status": "failed", "failed_step": "x1_gapadaptive"}
    else:
        append_jsonl(
            manifest_path,
            {
                "event": "step_skip",
                "step_name": "x1_gapadaptive",
                "reuse_source": x1_reuse_source,
                **context,
            },
            manifest_lock,
        )

    if not alltracker_summary.exists():
        alltracker_cmd = [
            str(PYTHON),
            "tools/build_v105_phase3_alltracker_contract.py",
            "--scene-id",
            scene_id,
            "--frame-start",
            str(int(frame_start)),
            "--frame-count",
            str(int(frame_count)),
            "--x1-label-dir",
            rel(x1_labels),
            "--output-root",
            rel(alltracker_root),
        ]
        if str(seed_source) != "full_x1" or bool(allow_missing_x1_diagnostics):
            alltracker_cmd.append("--allow-missing-diagnostic-labels")
        ok = run_command(
            cmd=alltracker_cmd,
            log_path=log_dir / f"alltracker_{suffix}.log",
            env=env,
            timeout_sec=timeout_alltracker,
            step_name="alltracker_proxy",
            manifest_path=manifest_path,
            manifest_lock=manifest_lock,
            context=context,
        )
        if not ok:
            return {**context, "status": "failed", "failed_step": "alltracker_proxy"}
    else:
        append_jsonl(
            manifest_path,
            {
                "event": "step_skip",
                "step_name": "alltracker_proxy",
                "reuse_source": alltracker_reuse_source,
                **context,
            },
            manifest_lock,
        )

    if bool(skip_phase6_candidate):
        append_jsonl(
            manifest_path,
            {"event": "step_skip", "step_name": "phase6_p6_candidate", "skip_reason": "skip_phase6_candidate", **context},
            manifest_lock,
        )
    elif not phase6_summary.exists():
        phase6_cmd = [
            str(PYTHON),
            "tools/build_v105_phase6_speculative_gap_birth.py",
            "--config",
            rel(DEFAULT_CONFIG),
            "--scene-id",
            scene_id,
            "--frame-start",
            str(int(frame_start)),
            "--frame-count",
            str(int(frame_count)),
            "--frame0-birth-records",
            rel(birth_records),
            "--alltracker-dir",
            rel(alltracker_root),
            "--x0-summary",
            rel(x0_summary_path),
            "--x1-summary",
            rel(x1_summary),
            "--output-root",
            rel(phase6_root),
            "--variant",
            "p4_anchor_period_definite",
            "--use-video-feature-bank",
            "--video-feature-bank-storage-device",
            "cuda",
            "--video-gpu-hot-window",
            str(int(video_gpu_hot_window)),
            "--birth-anchor-period",
            str(int(birth_anchor_period)),
            "--birth-anchor-offset",
            str(int(birth_anchor_offset)),
            "--birth-anchor-force-candidate-area-thresh",
            str(int(birth_anchor_force_candidate_area_thresh)),
            "--birth-anchor-force-max-events",
            str(int(birth_anchor_force_max_events)),
            "--max-births-per-frame",
            str(int(max_births_per_frame)),
            "--choice-policy",
            str(choice_policy),
            "--min-component-area",
            str(int(min_component_area)),
            "--max-points-per-frame",
            str(int(max_points_per_frame)),
            "--base-points-per-component",
            str(int(base_points_per_component)),
            "--area-per-extra-point",
            str(int(area_per_extra_point)),
            "--max-points-per-component",
            str(int(max_points_per_component)),
            "--min-candidate-touch-area",
            str(int(min_candidate_touch_area)),
            "--min-candidate-touch-ratio",
            str(float(min_candidate_touch_ratio)),
            "--enable-frame0-residual-repair",
            "--scheduler-mode",
            "independent_anchor",
            "--stream-state-repair-mode",
            "reconsolidate",
            "--allow-missing-x0-diagnostics",
        ]
        if bool(allow_missing_x1_diagnostics):
            phase6_cmd.append("--allow-missing-x1-diagnostics")
        if bool(reuse_video_state_template):
            phase6_cmd.append("--reuse-video-state-template")
        if bool(skip_phase6_chunk_visuals):
            phase6_cmd.extend(["--skip-chunk-overlays", "--skip-chunk-sheets", "--skip-chunk-video"])
        phase6_cmd.extend(phase6_extra_args)
        ok = run_command(
            cmd=phase6_cmd,
            log_path=log_dir / f"phase6_{suffix}.log",
            env=env,
            timeout_sec=timeout_phase6,
            step_name="phase6_p6_candidate",
            manifest_path=manifest_path,
            manifest_lock=manifest_lock,
            context=context,
        )
        if not ok:
            return {**context, "status": "failed", "failed_step": "phase6_p6_candidate"}
    else:
        append_jsonl(manifest_path, {"event": "step_skip", "step_name": "phase6_p6_candidate", **context}, manifest_lock)

    return {
        **context,
        "status": "completed",
        "phase6_skipped_by_cli": bool(skip_phase6_candidate),
        "x1_summary": rel(x1_summary),
        "x1_reuse_source": x1_reuse_source,
        "alltracker_summary": rel(alltracker_summary),
        "alltracker_reuse_source": alltracker_reuse_source,
        "phase6_summary": rel(phase6_summary),
    }


def color_for_id(label_id: int) -> tuple[int, int, int]:
    value = int(label_id) * 1103515245 + 12345
    r = 80 + ((value >> 0) & 127)
    g = 80 + ((value >> 8) & 127)
    b = 80 + ((value >> 16) & 127)
    return int(b), int(g), int(r)


def blend_lut_for_max_label(max_label_id: int, *, alpha: float) -> np.ndarray:
    key = float(alpha)
    existing = _BLEND_LUT_CACHE.get(key)
    if existing is not None and existing.shape[0] > int(max_label_id):
        return existing
    old_count = int(existing.shape[0]) if existing is not None else 0
    new_count = int(max_label_id) + 1
    lut = np.zeros((new_count, 256, 3), dtype=np.uint8)
    if existing is not None:
        lut[:old_count] = existing
    values = np.arange(256, dtype=np.float32)[:, None]
    for label_id in range(max(1, old_count), new_count):
        color = np.array(color_for_id(label_id), dtype=np.float32)[None, :]
        lut[label_id] = np.clip(
            values * np.float32(1.0 - alpha) + color * np.float32(alpha),
            0,
            255,
        ).astype(np.uint8)
    _BLEND_LUT_CACHE[key] = lut
    return lut


def overlay_label(rgb_bgr: np.ndarray, label: np.ndarray | None, *, alpha: float = 0.58) -> np.ndarray:
    out = rgb_bgr.copy()
    if label is None:
        return out
    if label.ndim == 3:
        label = label[:, :, 0]
    if label.shape[:2] != out.shape[:2]:
        label = cv2.resize(label.astype(np.uint16), (out.shape[1], out.shape[0]), interpolation=cv2.INTER_NEAREST)
    ids = [int(v) for v in np.unique(label) if int(v) > 0]
    if not ids:
        return out
    mask_any = label > 0
    blend_lut = blend_lut_for_max_label(max(ids), alpha=alpha)
    labels = label[mask_any]
    pixels = out[mask_any]
    pixels[:, 0] = blend_lut[labels, pixels[:, 0], 0]
    pixels[:, 1] = blend_lut[labels, pixels[:, 1], 1]
    pixels[:, 2] = blend_lut[labels, pixels[:, 2], 2]
    out[mask_any] = pixels
    edges = np.zeros(label.shape[:2], dtype=np.uint8)
    for label_id in ids:
        m = (label == label_id).astype(np.uint8)
        if np.count_nonzero(m) == 0:
            continue
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(edges, contours, -1, 255, 1)
    out[edges > 0] = (255, 255, 255)
    return out


def put_text(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    out = frame.copy()
    height = 22 * len(lines) + 6
    cv2.rectangle(out, (0, 0), (out.shape[1], height), (0, 0, 0), thickness=-1)
    for idx, line in enumerate(lines):
        cv2.putText(out, line, (8, 18 + 22 * idx), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def empty_assembly_timing() -> dict[str, float]:
    return {
        "total_sec": 0.0,
        "label_path_sec": 0.0,
        "label_link_or_copy_sec": 0.0,
        "label_read_sec": 0.0,
        "rgb_read_sec": 0.0,
        "overlay_label_sec": 0.0,
        "put_text_sec": 0.0,
        "overlay_jpeg_write_sec": 0.0,
        "video_writer_init_sec": 0.0,
        "video_write_sec": 0.0,
        "video_release_sec": 0.0,
        "video_decode_count_sec": 0.0,
    }


def assemble_frame_payload(
    *,
    frame_index: int,
    frame_id: int,
    scene_id: str,
    scene_color: Path,
    src_label: Path,
    dst_label: Path,
    overlay_path: Path,
    path_lookup_sec: float,
    skip_assembled_overlay_jpegs: bool,
    hardlink_assembled_labels: bool,
) -> dict[str, Any]:
    timing = empty_assembly_timing()
    timing["label_path_sec"] = float(path_lookup_sec)
    missing_label = False
    copied = 0
    hardlinked = 0
    copy_fallbacks = 0
    written_overlay_jpegs = 0
    if src_label.exists():
        t0 = time.time()
        if bool(hardlink_assembled_labels):
            try:
                if dst_label.exists():
                    dst_label.unlink()
                os.link(src_label, dst_label)
                hardlinked += 1
            except OSError:
                shutil.copy2(src_label, dst_label)
                copy_fallbacks += 1
        else:
            shutil.copy2(src_label, dst_label)
            copy_fallbacks += 1
        copied += 1
        timing["label_link_or_copy_sec"] += time.time() - t0
        t0 = time.time()
        label = cv2.imread(str(src_label), cv2.IMREAD_UNCHANGED)
        timing["label_read_sec"] += time.time() - t0
    else:
        missing_label = True
        label = None

    t0 = time.time()
    rgb = cv2.imread(str(scene_color / f"{int(frame_id)}.jpg"), cv2.IMREAD_COLOR)
    if rgb is None:
        rgb = cv2.imread(str(scene_color / f"{int(frame_id)}.png"), cv2.IMREAD_COLOR)
    timing["rgb_read_sec"] += time.time() - t0
    if rgb is None:
        return {
            "frame_index": int(frame_index),
            "frame_id": int(frame_id),
            "overlay": None,
            "missing_label": bool(missing_label),
            "copied": int(copied),
            "hardlinked": int(hardlinked),
            "copy_fallbacks": int(copy_fallbacks),
            "written_overlay_jpegs": int(written_overlay_jpegs),
            "timing": timing,
        }

    t0 = time.time()
    overlay = overlay_label(rgb, label)
    timing["overlay_label_sec"] += time.time() - t0
    t0 = time.time()
    overlay = put_text(
        overlay,
        [
            f"{scene_id} stride5 full-scene video frame_index={int(frame_index):04d} frame_id={int(frame_id):06d}",
            "P6 chunk-windowed masks; not a continuous scene-level ID claim",
        ],
    )
    timing["put_text_sec"] += time.time() - t0
    if not bool(skip_assembled_overlay_jpegs):
        t0 = time.time()
        cv2.imwrite(str(overlay_path), overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        timing["overlay_jpeg_write_sec"] += time.time() - t0
        written_overlay_jpegs += 1
    return {
        "frame_index": int(frame_index),
        "frame_id": int(frame_id),
        "overlay": overlay,
        "missing_label": bool(missing_label),
        "copied": int(copied),
        "hardlinked": int(hardlinked),
        "copy_fallbacks": int(copy_fallbacks),
        "written_overlay_jpegs": int(written_overlay_jpegs),
        "timing": timing,
    }


def first_prefix_label_root(scene_id: str, template: str) -> Path:
    return resolve_repo_path(format_scene_template(template, scene_id))


def generated_phase6_label_root(output_root: Path, scene_id: str, frame_start: int, frame_count: int) -> Path:
    suffix = f"{scene_short(scene_id)}_start{int(frame_start):04d}_f{int(frame_count)}"
    return output_root / f"phase6_{suffix}" / "labels"


def source_label_for_frame(
    *,
    output_root: Path,
    scene_id: str,
    frame_ids: list[int],
    frame_index: int,
    first64_count: int,
    first_prefix_label_root_template: str,
    chunk_size: int,
) -> Path:
    frame_id = int(frame_ids[frame_index])
    if frame_index < int(first64_count):
        return first_prefix_label_root(scene_id, first_prefix_label_root_template) / f"frame_{frame_id:06d}.png"
    chunk_start_index = int(first64_count) + ((int(frame_index) - int(first64_count)) // int(chunk_size)) * int(chunk_size)
    count = min(int(chunk_size), len(frame_ids) - chunk_start_index)
    start_frame = int(frame_ids[chunk_start_index])
    return generated_phase6_label_root(output_root, scene_id, start_frame, count) / f"frame_{frame_id:06d}.png"


def assemble_scene(
    *,
    output_root: Path,
    scene_id: str,
    frame_ids: list[int],
    first64_count: int,
    first_prefix_label_root_template: str,
    chunk_size: int,
    fps: float,
    variant_id: str,
    skip_assembled_overlay_jpegs: bool,
    hardlink_assembled_labels: bool,
    assembly_workers: int,
) -> dict[str, Any]:
    assembled_root = output_root / "assembled_scene_videos"
    mask_dir = assembled_root / "sgq_local" / "masks" / variant_id / scene_id / "mask"
    overlay_dir = assembled_root / "overlays" / scene_id
    video_path = assembled_root / "videos" / f"{variant_id}_{scene_id}_full_stride5.mp4"
    if mask_dir.exists():
        shutil.rmtree(mask_dir)
    if overlay_dir.exists() and not bool(skip_assembled_overlay_jpegs):
        shutil.rmtree(overlay_dir)
    mask_dir.mkdir(parents=True, exist_ok=True)
    if not bool(skip_assembled_overlay_jpegs):
        overlay_dir.mkdir(parents=True, exist_ok=True)
    video_path.parent.mkdir(parents=True, exist_ok=True)

    scene_color = REPO_ROOT / "Stream3D/data/scannet/processed" / scene_id / "color"
    missing_labels: list[int] = []
    copied = 0
    hardlinked = 0
    copy_fallbacks = 0
    writer: cv2.VideoWriter | None = None
    written_video_frames = 0
    written_overlay_jpegs = 0
    timing = empty_assembly_timing()
    total_t0 = time.time()

    def submit_payload(executor: concurrent.futures.Executor, frame_index: int) -> concurrent.futures.Future[dict[str, Any]]:
        frame_id = int(frame_ids[frame_index])
        t0 = time.time()
        src_label = source_label_for_frame(
            output_root=output_root,
            scene_id=scene_id,
            frame_ids=frame_ids,
            frame_index=frame_index,
            first64_count=first64_count,
            first_prefix_label_root_template=first_prefix_label_root_template,
            chunk_size=chunk_size,
        )
        dst_label = mask_dir / f"{int(frame_id)}.png"
        overlay_path = overlay_dir / f"{frame_index:04d}_frame_{int(frame_id):06d}.jpg"
        path_lookup_sec = time.time() - t0
        return executor.submit(
            assemble_frame_payload,
            frame_index=int(frame_index),
            frame_id=int(frame_id),
            scene_id=scene_id,
            scene_color=scene_color,
            src_label=src_label,
            dst_label=dst_label,
            overlay_path=overlay_path,
            path_lookup_sec=float(path_lookup_sec),
            skip_assembled_overlay_jpegs=bool(skip_assembled_overlay_jpegs),
            hardlink_assembled_labels=bool(hardlink_assembled_labels),
        )

    def consume_payload(row: dict[str, Any]) -> None:
        nonlocal copied, hardlinked, copy_fallbacks, written_overlay_jpegs, writer, written_video_frames
        copied += int(row["copied"])
        hardlinked += int(row["hardlinked"])
        copy_fallbacks += int(row["copy_fallbacks"])
        written_overlay_jpegs += int(row["written_overlay_jpegs"])
        if bool(row["missing_label"]):
            missing_labels.append(int(row["frame_id"]))
        for key, value in row["timing"].items():
            timing[key] += float(value)
        overlay = row["overlay"]
        if overlay is None:
            return
        if writer is None:
            t0 = time.time()
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                float(fps),
                (overlay.shape[1], overlay.shape[0]),
            )
            timing["video_writer_init_sec"] += time.time() - t0
        if writer is not None and writer.isOpened():
            t0 = time.time()
            writer.write(overlay)
            timing["video_write_sec"] += time.time() - t0
            written_video_frames += 1

    workers = max(1, int(assembly_workers))
    if workers == 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            for frame_index in range(len(frame_ids)):
                consume_payload(submit_payload(executor, frame_index).result())
    else:
        window = max(workers * 2, workers)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            pending: dict[int, concurrent.futures.Future[dict[str, Any]]] = {}
            next_submit = 0
            while next_submit < len(frame_ids) and len(pending) < window:
                pending[next_submit] = submit_payload(executor, next_submit)
                next_submit += 1
            for frame_index in range(len(frame_ids)):
                row = pending.pop(frame_index).result()
                consume_payload(row)
                while next_submit < len(frame_ids) and len(pending) < window:
                    pending[next_submit] = submit_payload(executor, next_submit)
                    next_submit += 1
    if writer is not None:
        t0 = time.time()
        writer.release()
        timing["video_release_sec"] += time.time() - t0
    cap_count = 0
    if video_path.exists():
        t0 = time.time()
        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            cap_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        timing["video_decode_count_sec"] += time.time() - t0
    timing["total_sec"] = time.time() - total_t0
    return {
        "schema_version": "stream4d_v105_fullscene_multichunk_scene_video_row_v1",
        "scene_id": scene_id,
        "variant_id": variant_id,
        "frame_count_expected": int(len(frame_ids)),
        "label_count_copied": int(copied),
        "missing_label_count": int(len(missing_labels)),
        "missing_label_frame_ids_first50": missing_labels[:50],
        "missing_label_frame_ids_truncated": bool(len(missing_labels) > 50),
        "mask_dir": rel(mask_dir),
        "overlay_dir": rel(overlay_dir) if not bool(skip_assembled_overlay_jpegs) else "",
        "skip_assembled_overlay_jpegs": bool(skip_assembled_overlay_jpegs),
        "assembled_overlay_jpegs_written": int(written_overlay_jpegs),
        "hardlink_assembled_labels": bool(hardlink_assembled_labels),
        "label_hardlink_count": int(hardlinked),
        "label_copy_fallback_count": int(copy_fallbacks),
        "video_path": rel(video_path),
        "video_exists": bool(video_path.exists() and video_path.stat().st_size > 0),
        "video_frame_count_written": int(written_video_frames),
        "video_frame_count_decoded": int(cap_count),
        "complete_scene_video": bool(copied == len(frame_ids) and cap_count == len(frame_ids)),
        "chunk_windowed_internal": True,
        "continuous_scene_level_id_claim": False,
        "alpha": 0.58,
        "assembly_workers": int(workers),
        "assembly_timing_sec": {key: float(value) for key, value in timing.items()},
    }


def run_posthoc_boundary_relabel(
    *,
    output_root: Path,
    scene_row: dict[str, Any],
    chunk_size: int,
    fps: float,
    match_mode: str,
    min_iou: float,
    min_intersection_pixels: int,
    diagnostic_iou_threshold: float,
    workers: int,
    skip_video: bool,
    output_subdir: str,
    zip_name: str,
) -> dict[str, Any]:
    scene_id = str(scene_row.get("scene_id", ""))
    source_mask_dir = resolve_repo_path(str(scene_row.get("mask_dir", "")))
    posthoc_root = output_root / str(output_subdir) / scene_id
    variant_id = f"{scene_row.get('variant_id', 'variant')}_boundary_iou_{match_mode}"
    generated_zip_name = str(zip_name).strip()
    if not generated_zip_name:
        generated_zip_name = f"{scene_short(scene_id)}_{variant_id}.zip"
    elif len(str(scene_id)) and "{scene_id}" in generated_zip_name:
        generated_zip_name = generated_zip_name.format(scene_id=scene_id, scene_short=scene_short(scene_id))

    cmd = [
        str(PYTHON),
        "tools/relabel_v105_chunk_ids_by_boundary_iou.py",
        "--source-mask-dir",
        rel(source_mask_dir),
        "--output-root",
        rel(posthoc_root),
        "--scene-id",
        scene_id,
        "--variant-id",
        variant_id,
        "--chunk-size",
        str(int(chunk_size)),
        "--min-iou",
        str(float(min_iou)),
        "--min-intersection-pixels",
        str(int(min_intersection_pixels)),
        "--match-mode",
        str(match_mode),
        "--diagnostic-iou-threshold",
        str(float(diagnostic_iou_threshold)),
        "--fps",
        str(float(fps)),
        "--workers",
        str(int(workers)),
        "--zip-name",
        generated_zip_name,
    ]
    if bool(skip_video):
        cmd.append("--skip-video")

    log_path = output_root / "run_logs" / f"posthoc_boundary_relabel_{scene_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env={**os.environ.copy(), "PYTHONUNBUFFERED": "1"}, stdout=log, stderr=subprocess.STDOUT, check=False)
    wall_sec = time.time() - started

    summary_path = posthoc_root / "chunk_boundary_relabel_summary.json"
    summary_payload: dict[str, Any] | None = None
    if summary_path.exists():
        summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))

    row: dict[str, Any] = {
        "schema_version": "stream4d_v105_posthoc_boundary_relabel_row_v1",
        "scene_id": scene_id,
        "source_mask_dir": rel(source_mask_dir),
        "output_root": rel(posthoc_root),
        "variant_id": variant_id,
        "cmd": cmd,
        "log_path": rel(log_path),
        "returncode": int(proc.returncode),
        "wall_sec": float(wall_sec),
        "summary_path": rel(summary_path),
        "summary_exists": bool(summary_path.exists()),
        "summary_sha256": sha256_file(summary_path) if summary_path.exists() else "",
        "match_mode": str(match_mode),
        "min_iou": float(min_iou),
        "min_intersection_pixels": int(min_intersection_pixels),
        "diagnostic_iou_threshold": float(diagnostic_iou_threshold),
        "mask_geometry_claim": "ids_only_foreground_geometry_must_match_source",
        "posthoc_only": True,
        "continuous_scene_level_id_claim": False,
    }
    if summary_payload:
        row.update(
            {
                "geometry_verification": summary_payload.get("geometry_verification"),
                "raw_boundary_summary": summary_payload.get("raw_boundary_summary"),
                "relabeled_boundary_summary": summary_payload.get("relabeled_boundary_summary"),
                "relabeled_mask_dir": summary_payload.get("relabeled_mask_dir"),
                "video_summary": summary_payload.get("video_summary"),
                "zip_summary": summary_payload.get("zip_summary"),
            }
        )
    return row


def worker_loop(
    *,
    gpu: str,
    jobs: "queue.Queue[dict[str, Any]]",
    results: list[dict[str, Any]],
    results_lock: threading.Lock,
    output_root: Path,
    manifest_path: Path,
    manifest_lock: threading.Lock,
    timeout_x1: int,
    timeout_alltracker: int,
    timeout_phase6: int,
    choice_policy: str,
    min_component_area: int,
    max_points_per_frame: int,
    base_points_per_component: int,
    area_per_extra_point: int,
    max_points_per_component: int,
    min_candidate_touch_area: int,
    min_candidate_touch_ratio: float,
    max_births_per_frame: int,
    birth_anchor_period: int,
    birth_anchor_offset: int,
    birth_anchor_force_candidate_area_thresh: int,
    birth_anchor_force_max_events: int,
    video_gpu_hot_window: int,
    reuse_video_state_template: bool,
    skip_phase6_chunk_visuals: bool,
    skip_phase6_candidate: bool,
    seed_source: str,
    x0_summary: str,
    x0_summary_template: str,
    reuse_x1_root_template: str,
    reuse_alltracker_root_template: str,
    allow_missing_x1_diagnostics: bool,
    phase6_extra_args: list[str],
) -> None:
    while True:
        try:
            job = jobs.get_nowait()
        except queue.Empty:
            return
        try:
            result = run_chunk_job(
                gpu=gpu,
                scene_id=str(job["scene_id"]),
                frame_start=int(job["frame_start"]),
                frame_count=int(job["frame_count"]),
                output_root=output_root,
                manifest_path=manifest_path,
                manifest_lock=manifest_lock,
                timeout_x1=timeout_x1,
                timeout_alltracker=timeout_alltracker,
                timeout_phase6=timeout_phase6,
                choice_policy=choice_policy,
                min_component_area=min_component_area,
                max_points_per_frame=max_points_per_frame,
                base_points_per_component=base_points_per_component,
                area_per_extra_point=area_per_extra_point,
                max_points_per_component=max_points_per_component,
                min_candidate_touch_area=min_candidate_touch_area,
                min_candidate_touch_ratio=min_candidate_touch_ratio,
                max_births_per_frame=max_births_per_frame,
                birth_anchor_period=birth_anchor_period,
                birth_anchor_offset=birth_anchor_offset,
                birth_anchor_force_candidate_area_thresh=birth_anchor_force_candidate_area_thresh,
                birth_anchor_force_max_events=birth_anchor_force_max_events,
                video_gpu_hot_window=video_gpu_hot_window,
                reuse_video_state_template=reuse_video_state_template,
                skip_phase6_chunk_visuals=skip_phase6_chunk_visuals,
                skip_phase6_candidate=skip_phase6_candidate,
                seed_source=seed_source,
                x0_summary=x0_summary,
                x0_summary_template=x0_summary_template,
                reuse_x1_root_template=reuse_x1_root_template,
                reuse_alltracker_root_template=reuse_alltracker_root_template,
                allow_missing_x1_diagnostics=allow_missing_x1_diagnostics,
                phase6_extra_args=phase6_extra_args,
            )
        except Exception as exc:
            result = {
                "scene_id": str(job["scene_id"]),
                "frame_start": int(job["frame_start"]),
                "frame_count": int(job["frame_count"]),
                "gpu": str(gpu),
                "status": "exception",
                "exception": repr(exc),
            }
            append_jsonl(manifest_path, {"event": "job_exception", **result}, manifest_lock)
        with results_lock:
            results.append(result)
        jobs.task_done()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", default="scene0030_00,scene0591_00")
    parser.add_argument("--output-root", default=rel(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--gpus", default="6,7")
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--max-frames-per-scene", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--first-prefix-count", type=int, default=64)
    parser.add_argument("--first-prefix-label-root-template", default=DEFAULT_FIRST_PREFIX_LABEL_ROOT_TEMPLATE)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--skip-run", action="store_true")
    parser.add_argument("--skip-assemble", action="store_true")
    parser.add_argument(
        "--skip-assembled-overlay-jpegs",
        action="store_true",
        default=False,
        help="During full-scene assembly, write the MP4 but skip the per-frame assembled overlay JPEG sidecars.",
    )
    parser.add_argument(
        "--hardlink-assembled-labels",
        action="store_true",
        default=False,
        help="Hardlink labels into assembled mask_dir when possible, falling back to copy2. This preserves bytes and reduces same-filesystem assembly I/O.",
    )
    parser.add_argument(
        "--assembly-workers",
        type=int,
        default=1,
        help="Number of worker threads used to precompute full-scene overlay frames before ordered MP4 writing. Default 1 preserves the historical serial assembly path.",
    )
    parser.add_argument(
        "--posthoc-boundary-relabel",
        action="store_true",
        default=False,
        help="After assembly, write an extra post-hoc boundary-IoU id-relabel candidate. Default off; original assembled masks/video are unchanged.",
    )
    parser.add_argument("--posthoc-boundary-relabel-match-mode", choices=("one_to_one", "split_friendly"), default="one_to_one")
    parser.add_argument("--posthoc-boundary-relabel-min-iou", type=float, default=0.05)
    parser.add_argument("--posthoc-boundary-relabel-min-intersection-pixels", type=int, default=512)
    parser.add_argument("--posthoc-boundary-relabel-diagnostic-iou-threshold", type=float, default=0.05)
    parser.add_argument("--posthoc-boundary-relabel-workers", type=int, default=8)
    parser.add_argument("--posthoc-boundary-relabel-skip-video", action="store_true", default=False)
    parser.add_argument("--posthoc-boundary-relabel-output-subdir", default="posthoc_boundary_relabel")
    parser.add_argument(
        "--posthoc-boundary-relabel-zip-name",
        default="",
        help="Optional zip filename for the post-hoc relabel artifact. Supports {scene_id} and {scene_short}; blank generates one per scene.",
    )
    parser.add_argument("--timeout-x1", type=int, default=1800)
    parser.add_argument("--timeout-alltracker", type=int, default=900)
    parser.add_argument("--timeout-phase6", type=int, default=1200)
    parser.add_argument("--variant-id", default=DEFAULT_VARIANT_ID)
    parser.add_argument("--choice-policy", default="smallest_valid_mask_per_point")
    parser.add_argument("--min-component-area", type=int, default=200)
    parser.add_argument("--max-points-per-frame", type=int, default=64)
    parser.add_argument("--base-points-per-component", type=int, default=1)
    parser.add_argument("--area-per-extra-point", type=int, default=40000)
    parser.add_argument("--max-points-per-component", type=int, default=4)
    parser.add_argument("--min-candidate-touch-area", type=int, default=16)
    parser.add_argument("--min-candidate-touch-ratio", type=float, default=0.001)
    parser.add_argument("--max-births-per-frame", type=int, default=2)
    parser.add_argument("--birth-anchor-period", type=int, default=4)
    parser.add_argument("--birth-anchor-offset", type=int, default=1)
    parser.add_argument("--birth-anchor-force-candidate-area-thresh", type=int, default=50000)
    parser.add_argument("--birth-anchor-force-max-events", type=int, default=1)
    parser.add_argument("--video-gpu-hot-window", type=int, default=0)
    parser.add_argument(
        "--reuse-video-state-template",
        action="store_true",
        default=False,
        help="Forward --reuse-video-state-template to Phase6 and record it in runner summaries.",
    )
    parser.add_argument(
        "--skip-phase6-chunk-visuals",
        action="store_true",
        default=False,
        help="Forward Phase6 flags that skip per-chunk overlays, sheets, and chunk MP4s; labels are still written for full-scene assembly.",
    )
    parser.add_argument(
        "--skip-phase6-candidate",
        action="store_true",
        default=False,
        help="Run/prepare X1 and AllTracker chunk inputs but skip Phase6 generation. Useful before a persistent Phase6 batch wrapper.",
    )
    parser.add_argument("--seed-source", choices=["full_x1", "frame0_x1"], default="full_x1")
    parser.add_argument("--x0-summary", default="")
    parser.add_argument("--x0-summary-template", default=DEFAULT_X0_SUMMARY_TEMPLATE)
    parser.add_argument("--reuse-x1-root-template", default="")
    parser.add_argument("--reuse-alltracker-root-template", default="")
    parser.add_argument("--allow-missing-x1-diagnostics", action="store_true", default=False)
    parser.add_argument(
        "--phase6-extra-args",
        default="",
        help="Extra arguments forwarded verbatim to build_v105_phase6_speculative_gap_birth.py after the runner defaults.",
    )
    parser.add_argument("--zip-name", default="v105_fullscene_multichunk_scene_videos.zip")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_root = (REPO_ROOT / args.output_root).resolve() if not Path(args.output_root).is_absolute() else Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "fullscene_multichunk_manifest.jsonl"
    manifest_lock = threading.Lock()
    scenes = [part.strip() for part in str(args.scenes).split(",") if part.strip()]
    gpus = [part.strip() for part in str(args.gpus).split(",") if part.strip()]
    variant_id = str(args.variant_id)
    phase6_extra_args = shlex.split(str(args.phase6_extra_args))
    phase6_parameters = {
        "seed_source": str(args.seed_source),
        "x0_summary": str(args.x0_summary),
        "x0_summary_template": str(args.x0_summary_template),
        "reuse_x1_root_template": str(args.reuse_x1_root_template),
        "reuse_alltracker_root_template": str(args.reuse_alltracker_root_template),
        "allow_missing_x1_diagnostics": bool(args.allow_missing_x1_diagnostics),
        "phase6_extra_args": phase6_extra_args,
        "choice_policy": str(args.choice_policy),
        "min_component_area": int(args.min_component_area),
        "max_points_per_frame": int(args.max_points_per_frame),
        "base_points_per_component": int(args.base_points_per_component),
        "area_per_extra_point": int(args.area_per_extra_point),
        "max_points_per_component": int(args.max_points_per_component),
        "min_candidate_touch_area": int(args.min_candidate_touch_area),
        "min_candidate_touch_ratio": float(args.min_candidate_touch_ratio),
        "max_births_per_frame": int(args.max_births_per_frame),
        "birth_anchor_period": int(args.birth_anchor_period),
        "birth_anchor_offset": int(args.birth_anchor_offset),
        "birth_anchor_force_candidate_area_thresh": int(args.birth_anchor_force_candidate_area_thresh),
        "birth_anchor_force_max_events": int(args.birth_anchor_force_max_events),
        "video_gpu_hot_window": int(args.video_gpu_hot_window),
        "reuse_video_state_template": bool(args.reuse_video_state_template),
        "skip_phase6_chunk_visuals": bool(args.skip_phase6_chunk_visuals),
        "skip_phase6_candidate": bool(args.skip_phase6_candidate),
        "skip_assembled_overlay_jpegs": bool(args.skip_assembled_overlay_jpegs),
        "hardlink_assembled_labels": bool(args.hardlink_assembled_labels),
        "assembly_workers": int(args.assembly_workers),
        "posthoc_boundary_relabel": bool(args.posthoc_boundary_relabel),
        "posthoc_boundary_relabel_match_mode": str(args.posthoc_boundary_relabel_match_mode),
        "posthoc_boundary_relabel_min_iou": float(args.posthoc_boundary_relabel_min_iou),
        "posthoc_boundary_relabel_min_intersection_pixels": int(args.posthoc_boundary_relabel_min_intersection_pixels),
        "posthoc_boundary_relabel_diagnostic_iou_threshold": float(args.posthoc_boundary_relabel_diagnostic_iou_threshold),
        "posthoc_boundary_relabel_workers": int(args.posthoc_boundary_relabel_workers),
        "posthoc_boundary_relabel_skip_video": bool(args.posthoc_boundary_relabel_skip_video),
        "posthoc_boundary_relabel_output_subdir": str(args.posthoc_boundary_relabel_output_subdir),
        "posthoc_boundary_relabel_zip_name": str(args.posthoc_boundary_relabel_zip_name),
    }
    if not gpus:
        raise ValueError("at least one GPU id is required")

    frame_ids_by_scene = {scene: selected_frame_ids(scene, int(args.frame_stride)) for scene in scenes}
    if int(args.max_frames_per_scene) > 0:
        frame_ids_by_scene = {
            scene: ids[: int(args.max_frames_per_scene)]
            for scene, ids in frame_ids_by_scene.items()
        }
    jobs: "queue.Queue[dict[str, Any]]" = queue.Queue()
    planned_jobs: list[dict[str, Any]] = []
    for scene in scenes:
        specs = chunk_specs(
            frame_ids_by_scene[scene],
            start_index=int(args.first_prefix_count),
            chunk_size=int(args.chunk_size),
        )
        for spec in specs:
            job = {"scene_id": scene, **spec}
            planned_jobs.append(job)
            jobs.put(job)
    write_json(
        output_root / "fullscene_multichunk_plan.json",
        {
            "schema_version": "stream4d_v105_fullscene_multichunk_plan_v1",
            "variant_id": variant_id,
            "scenes": scenes,
            "frame_stride": int(args.frame_stride),
            "max_frames_per_scene": int(args.max_frames_per_scene),
            "chunk_size": int(args.chunk_size),
            "first_prefix_count": int(args.first_prefix_count),
            "first_prefix_label_root_template": str(args.first_prefix_label_root_template),
            "phase6_parameters": phase6_parameters,
            "chunk_windowed_internal": True,
            "continuous_scene_level_id_claim": False,
            "planned_jobs": planned_jobs,
            "frame_counts_by_scene": {scene: len(ids) for scene, ids in frame_ids_by_scene.items()},
        },
    )

    results: list[dict[str, Any]] = []
    if not bool(args.skip_run):
        results_lock = threading.Lock()
        threads = [
            threading.Thread(
                target=worker_loop,
                kwargs={
                    "gpu": gpu,
                    "jobs": jobs,
                    "results": results,
                    "results_lock": results_lock,
                    "output_root": output_root,
                    "manifest_path": manifest_path,
                    "manifest_lock": manifest_lock,
                    "timeout_x1": int(args.timeout_x1),
                    "timeout_alltracker": int(args.timeout_alltracker),
                    "timeout_phase6": int(args.timeout_phase6),
                    "choice_policy": str(args.choice_policy),
                    "min_component_area": int(args.min_component_area),
                    "max_points_per_frame": int(args.max_points_per_frame),
                    "base_points_per_component": int(args.base_points_per_component),
                    "area_per_extra_point": int(args.area_per_extra_point),
                    "max_points_per_component": int(args.max_points_per_component),
                    "min_candidate_touch_area": int(args.min_candidate_touch_area),
                    "min_candidate_touch_ratio": float(args.min_candidate_touch_ratio),
                    "max_births_per_frame": int(args.max_births_per_frame),
                    "birth_anchor_period": int(args.birth_anchor_period),
                    "birth_anchor_offset": int(args.birth_anchor_offset),
                    "birth_anchor_force_candidate_area_thresh": int(args.birth_anchor_force_candidate_area_thresh),
                    "birth_anchor_force_max_events": int(args.birth_anchor_force_max_events),
                    "video_gpu_hot_window": int(args.video_gpu_hot_window),
                    "reuse_video_state_template": bool(args.reuse_video_state_template),
                    "skip_phase6_chunk_visuals": bool(args.skip_phase6_chunk_visuals),
                    "skip_phase6_candidate": bool(args.skip_phase6_candidate),
                    "seed_source": str(args.seed_source),
                    "x0_summary": str(args.x0_summary),
                    "x0_summary_template": str(args.x0_summary_template),
                    "reuse_x1_root_template": str(args.reuse_x1_root_template),
                    "reuse_alltracker_root_template": str(args.reuse_alltracker_root_template),
                    "allow_missing_x1_diagnostics": bool(args.allow_missing_x1_diagnostics),
                    "phase6_extra_args": phase6_extra_args,
                },
                daemon=False,
            )
            for gpu in gpus
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    else:
        for job in planned_jobs:
            results.append({**job, "status": "skipped_by_cli"})

    scene_rows: list[dict[str, Any]] = []
    if not bool(args.skip_assemble):
        for scene in scenes:
            scene_rows.append(
                assemble_scene(
                    output_root=output_root,
                    scene_id=scene,
                    frame_ids=frame_ids_by_scene[scene],
                    first64_count=int(args.first_prefix_count),
                    first_prefix_label_root_template=str(args.first_prefix_label_root_template),
                    chunk_size=int(args.chunk_size),
                    fps=float(args.fps),
                    variant_id=variant_id,
                    skip_assembled_overlay_jpegs=bool(args.skip_assembled_overlay_jpegs),
                    hardlink_assembled_labels=bool(args.hardlink_assembled_labels),
                    assembly_workers=int(args.assembly_workers),
                )
            )

    posthoc_boundary_relabel_rows: list[dict[str, Any]] = []
    if bool(args.posthoc_boundary_relabel):
        for row in scene_rows:
            posthoc_row = run_posthoc_boundary_relabel(
                output_root=output_root,
                scene_row=row,
                chunk_size=int(args.chunk_size),
                fps=float(args.fps),
                match_mode=str(args.posthoc_boundary_relabel_match_mode),
                min_iou=float(args.posthoc_boundary_relabel_min_iou),
                min_intersection_pixels=int(args.posthoc_boundary_relabel_min_intersection_pixels),
                diagnostic_iou_threshold=float(args.posthoc_boundary_relabel_diagnostic_iou_threshold),
                workers=int(args.posthoc_boundary_relabel_workers),
                skip_video=bool(args.posthoc_boundary_relabel_skip_video),
                output_subdir=str(args.posthoc_boundary_relabel_output_subdir),
                zip_name=str(args.posthoc_boundary_relabel_zip_name),
            )
            posthoc_boundary_relabel_rows.append(posthoc_row)
            if int(posthoc_row.get("returncode", 1)) != 0:
                raise RuntimeError({"failed_posthoc_boundary_relabel": posthoc_row})

    videos_dir = output_root / "assembled_scene_videos" / "videos"
    zip_path = output_root / str(args.zip_name)
    summary = {
        "schema_version": "stream4d_v105_fullscene_multichunk_summary_v1",
        "variant_id": variant_id,
        "phase6_parameters": phase6_parameters,
        "job_count": len(planned_jobs),
        "job_completed_count": sum(1 for row in results if row.get("status") == "completed"),
        "job_failed_count": sum(1 for row in results if row.get("status") not in {"completed", "skipped_by_cli"}),
        "job_results": results,
        "scene_video_rows": scene_rows,
        "posthoc_boundary_relabel_enabled": bool(args.posthoc_boundary_relabel),
        "posthoc_boundary_relabel_rows": posthoc_boundary_relabel_rows,
        "posthoc_boundary_relabel_all_ok": bool(not args.posthoc_boundary_relabel)
        or all(int(row.get("returncode", 1)) == 0 and bool(row.get("summary_exists")) for row in posthoc_boundary_relabel_rows),
        "all_scene_videos_complete": bool(scene_rows) and all(row.get("complete_scene_video") for row in scene_rows),
        "chunk_windowed_internal": True,
        "continuous_scene_level_id_claim": False,
        "not_claimed": [
            "continuous full-scene SAM2 tracking state",
            "local2history-verified scene-level identity",
            "complete-scene MV_AP_scene method success",
        ],
        "zip_path": rel(zip_path),
        "manifest_jsonl": rel(manifest_path),
    }
    summary_path = output_root / "fullscene_multichunk_summary.json"
    write_json(summary_path, summary)

    if videos_dir.exists() and scene_rows:
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(summary_path, arcname="fullscene_multichunk_summary.json")
            for row in scene_rows:
                video = REPO_ROOT / str(row.get("video_path", ""))
                if video.exists():
                    zf.write(video, arcname=video.name)
        summary["zip_exists"] = zip_path.exists()
        summary["zip_sha256"] = sha256_file(zip_path) if zip_path.exists() else ""
        write_json(summary_path, summary)

    print(json.dumps({"summary": rel(summary_path), "zip": rel(zip_path), "complete": summary["all_scene_videos_complete"]}, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
