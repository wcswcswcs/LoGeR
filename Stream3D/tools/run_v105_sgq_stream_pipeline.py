#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "v105" / "sgq_stream4d_default.yaml"
DEFAULT_OUTPUT = STREAM3D_ROOT / "outputs" / "audit" / "v105_sgq_default_dev"
RUNNER_PATH = Path(__file__).resolve()
SPECGAP_MODULE_DOCS = [
    "sam2_frame_feature_bank_design.md",
    "baseline_x_api_contract.md",
    "alltracker_chunk_provider_design.md",
    "speculative_coverage_field_design.md",
    "speculative_gap_tube_graph_design.md",
    "sam2_batch_birth_decoder_design.md",
    "sam2_cohort_tracker_design.md",
    "coverage_preserving_reconciliation_design.md",
    "lingbot_stream_provider_design.md",
    "local2history_memory_design.md",
    "cache_contract.md",
    "visual_audit_protocol.md",
    "profiling_contract.md",
]
SPECGAP_REQUIRED_DOC_SECTIONS = [
    "Responsibility",
    "Inputs/Outputs",
    "Persistent State",
    "Causality",
    "Memory Ownership",
    "Cache Contract",
    "API Calls",
    "Failure Modes",
    "Metrics",
    "Tests",
]


def _rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return p.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except Exception:
        return p.as_posix()


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
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_records_json(path: Path, rows: list[dict[str, Any]], *, schema_version: str | None = None) -> None:
    payload = {
        "schema_version": schema_version or "stream4d_v105_record_table_v1",
        "row_count": len(rows),
        "rows": rows,
    }
    _write_json(path, payload)


def _read_records_json(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
        return [row for row in payload["rows"] if isinstance(row, dict)]
    return []


def _sha256_file(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _hash_payload(payload: Any) -> str:
    raw = json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _stable_int_seed(*parts: Any, modulo: int = 2**31 - 1) -> int:
    raw = "::".join(str(part) for part in parts).encode("utf-8")
    return int(hashlib.sha256(raw).hexdigest()[:16], 16) % modulo


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return payload


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(_jsonable(payload), sort_keys=False, allow_unicode=False), encoding="utf-8")


def _parse_comma_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def _as_repo_path(value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _numeric_stem(path: Path) -> int:
    try:
        return int(path.stem)
    except ValueError:
        return 10**12


def _read_image(path: Path, flags: int = cv2.IMREAD_UNCHANGED) -> np.ndarray | None:
    if not path.exists():
        return None
    return cv2.imread(str(path), flags)


def _resize(image: np.ndarray, target_hw: tuple[int, int], interpolation: int) -> np.ndarray:
    h, w = target_hw
    if image.shape[:2] == (h, w):
        return image.copy()
    return cv2.resize(image, (w, h), interpolation=interpolation)


def _crop_scaled(image: np.ndarray, crop_rgb: tuple[int, int, int, int], rgb_hw: tuple[int, int]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    rgb_h, rgb_w = rgb_hw
    h, w = image.shape[:2]
    x0, y0, x1, y1 = crop_rgb
    sx = w / float(rgb_w)
    sy = h / float(rgb_h)
    ix0 = int(round(x0 * sx))
    iy0 = int(round(y0 * sy))
    ix1 = int(round(x1 * sx))
    iy1 = int(round(y1 * sy))
    ix0 = max(0, min(ix0, w - 1))
    iy0 = max(0, min(iy0, h - 1))
    ix1 = max(ix0 + 1, min(ix1, w))
    iy1 = max(iy0 + 1, min(iy1, h))
    return image[iy0:iy1, ix0:ix1].copy(), (ix0, iy0, ix1, iy1)


def _black_ratio(rgb: np.ndarray, threshold: int) -> float:
    if rgb.size == 0:
        return 1.0
    if rgb.ndim == 2:
        dark = rgb <= threshold
    else:
        dark = np.all(rgb <= threshold, axis=2)
    return float(np.count_nonzero(dark)) / float(dark.size)


def _detect_nonblack_crop(rgb: np.ndarray, threshold: int, margin: int = 0) -> tuple[int, int, int, int] | None:
    if rgb.ndim == 2:
        nonblack = rgb > threshold
    else:
        nonblack = np.any(rgb > threshold, axis=2)
    ys, xs = np.nonzero(nonblack)
    if xs.size == 0 or ys.size == 0:
        return None
    h, w = rgb.shape[:2]
    x0 = max(0, int(xs.min()) - margin)
    y0 = max(0, int(ys.min()) - margin)
    x1 = min(w, int(xs.max()) + 1 + margin)
    y1 = min(h, int(ys.max()) + 1 + margin)
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _load_intrinsics(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        return np.loadtxt(path).astype(np.float64)
    except Exception:
        return None


def _update_intrinsics(k: np.ndarray | None, crop: tuple[int, int, int, int], target_hw: tuple[int, int]) -> np.ndarray | None:
    if k is None:
        return None
    out = np.asarray(k, dtype=np.float64).copy()
    if out.shape[0] < 3 or out.shape[1] < 3:
        return None
    x0, y0, x1, y1 = crop
    target_h, target_w = target_hw
    sx = target_w / float(max(x1 - x0, 1))
    sy = target_h / float(max(y1 - y0, 1))
    out[0, 2] = (out[0, 2] - x0) * sx
    out[1, 2] = (out[1, 2] - y0) * sy
    out[0, 0] *= sx
    out[1, 1] *= sy
    return out


_INSTANCE_COLOR_PALETTE_BGR: tuple[tuple[int, int, int], ...] = (
    (44, 160, 44),
    (31, 119, 180),
    (214, 39, 40),
    (255, 127, 14),
    (148, 103, 189),
    (140, 86, 75),
    (227, 119, 194),
    (127, 127, 127),
    (188, 189, 34),
    (23, 190, 207),
    (0, 114, 178),
    (213, 94, 0),
    (0, 158, 115),
    (204, 121, 167),
    (240, 228, 66),
    (86, 180, 233),
)


def _instance_color_bgr(label: int) -> tuple[int, int, int]:
    if label > 0 and label <= len(_INSTANCE_COLOR_PALETTE_BGR):
        return _INSTANCE_COLOR_PALETTE_BGR[label - 1]
    value = int(label)
    # Scramble larger ids so adjacent ids remain visually distinct.
    return (
        int((53 * value + 97) % 255),
        int((157 * value + 29) % 255),
        int((223 * value + 191) % 255),
    )


def _colorize_labels(labels: np.ndarray) -> np.ndarray:
    labels64 = labels.astype(np.int64, copy=False)
    colors = np.zeros(labels64.shape + (3,), dtype=np.uint8)
    ids = np.unique(labels64)
    for label in ids:
        value = int(label)
        if value <= 0:
            continue
        colors[labels64 == value] = _instance_color_bgr(value)
    return colors


def _draw_instance_id_labels(frame_bgr: np.ndarray, labels: np.ndarray, *, min_pixels: int = 80) -> np.ndarray:
    out = frame_bgr.copy()
    labels64 = labels.astype(np.int64, copy=False)
    h, w = labels64.shape[:2]
    font_scale = max(0.35, min(0.55, w / 900.0))
    thickness = 1 if w < 700 else 2
    for label in [int(v) for v in np.unique(labels64) if int(v) > 0]:
        ys, xs = np.where(labels64 == label)
        if xs.size < int(min_pixels):
            continue
        x = int(np.median(xs))
        y = int(np.median(ys))
        text = f"id={label}"
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        x0 = max(0, min(w - tw - 4, x - tw // 2))
        y0 = max(th + 4, min(h - baseline - 2, y))
        color = _instance_color_bgr(label)
        cv2.rectangle(out, (x0 - 2, y0 - th - 3), (x0 + tw + 2, y0 + baseline + 2), (0, 0, 0), -1)
        cv2.putText(out, text, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
    return out


def _overlay(rgb_bgr: np.ndarray, labels: np.ndarray | None, *, draw_instance_ids: bool = False) -> np.ndarray:
    base = rgb_bgr.copy()
    if labels is None:
        return base
    color = _colorize_labels(labels)
    mask = np.any(color > 0, axis=2)
    if not np.any(mask):
        return base
    blended = cv2.addWeighted(base[mask], 0.55, color[mask], 0.45, 0.0)
    if blended is None:
        return base
    base[mask] = blended
    if draw_instance_ids:
        base = _draw_instance_id_labels(base, labels)
    return base


def _write_contact_sheet(tiles: list[np.ndarray], path: Path, columns: int = 4) -> bool:
    if not tiles:
        return False
    h, w = tiles[0].shape[:2]
    rows = (len(tiles) + max(int(columns), 1) - 1) // max(int(columns), 1)
    canvas = np.zeros((rows * h, max(int(columns), 1) * w, 3), dtype=np.uint8)
    for idx, tile in enumerate(tiles):
        if tile.shape[:2] != (h, w):
            tile = cv2.resize(tile, (w, h), interpolation=cv2.INTER_AREA)
        r, c = divmod(idx, max(int(columns), 1))
        canvas[r * h : (r + 1) * h, c * w : (c + 1) * w] = tile
    path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(path), canvas))


def _label_image_ids(path: Path) -> list[int]:
    image = _read_image(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        return []
    if image.ndim == 3:
        image = image[..., 0]
    return [int(value) for value in np.unique(image) if int(value) > 0]


def _line_hits(path: Path, patterns: dict[str, str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for lineno, line in enumerate(lines, start=1):
        for role, pattern in patterns.items():
            if re.search(pattern, line):
                rows.append(
                    {
                        "file": _rel(path),
                        "line": lineno,
                        "role": role,
                        "pattern": pattern,
                        "code_excerpt": line.strip()[:240],
                    }
                )
    return rows


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def _safe_error(exc: BaseException) -> str:
    return "".join(traceback.format_exception_only(type(exc), exc)).strip()


@dataclass
class PipelineContext:
    config: dict[str, Any]
    config_path: Path
    output_root: Path
    stages: list[str]
    force: bool
    provider_rows: list[dict[str, Any]] = field(default_factory=list)
    failure_rows: list[dict[str, Any]] = field(default_factory=list)
    stage_rows: list[dict[str, Any]] = field(default_factory=list)
    gpu_rows: list[dict[str, Any]] = field(default_factory=list)
    cache_rows: list[dict[str, Any]] = field(default_factory=list)
    gates: dict[str, Any] = field(default_factory=dict)
    runtime_state: dict[str, Any] = field(default_factory=dict)
    config_hash: str = ""

    def add_failure(self, **row: Any) -> None:
        base = {
            "schema_version": "stream4d_v105_failure_row_v1",
            "run_id": self.config["run"]["name"],
            "timestamp_unix": time.time(),
        }
        base.update(row)
        self.failure_rows.append(base)
        self.flush_common()

    def flush_common(self) -> None:
        _write_records_json(self.output_root / "provider_registry_records.json", self.provider_rows)
        _write_records_json(self.output_root / "cache_registry_records.json", self.cache_rows)
        _write_records_json(self.output_root / "stage_profile_records.json", self.stage_rows)
        _write_records_json(self.output_root / "gpu_memory_records.json", self.gpu_rows)
        _write_records_json(self.output_root / "failure_records.json", self.failure_rows)

    def write_summary(self, final_status: str = "in_progress") -> None:
        payload = {
            "schema_version": "stream4d_v105_summary_v1",
            "run_id": self.config["run"]["name"],
            "status": final_status,
            "stages_requested": self.stages,
            "config_hash": self.config_hash,
            "output_root": _rel(self.output_root),
            "gates": self.gates,
            "failure_count": len(self.failure_rows),
            "uses_gt_for_prediction": False,
            "uses_future": False,
            "formal_ap_generated": False,
            "formal_evaluator": self.config.get("evaluation", {}).get("evaluator", ""),
            "cache_read_count": sum(1 for row in self.cache_rows if str(row.get("cache_read", "")).lower() == "true"),
        }
        _write_json(self.output_root / "summary.json", payload)


class StageTimer:
    def __init__(self, ctx: PipelineContext, stage_name: str) -> None:
        self.ctx = ctx
        self.stage_name = stage_name
        self.start = 0.0

    def __enter__(self) -> "StageTimer":
        self.start = time.time()
        self._record_gpu("start")
        return self

    def __exit__(self, exc_type: Any, exc: BaseException | None, tb: Any) -> bool:
        runtime = time.time() - self.start
        self._record_gpu("end")
        row = {
            "schema_version": "stream4d_v105_stage_profile_row_v1",
            "run_id": self.ctx.config["run"]["name"],
            "stage_name": self.stage_name,
            "runtime_sec": runtime,
            "status": "failed" if exc is not None else "completed",
            "error": _safe_error(exc) if exc is not None else "",
        }
        self.ctx.stage_rows.append(row)
        if exc is not None:
            self.ctx.add_failure(
                stage_name=self.stage_name,
                failure_type="STAGE_EXCEPTION",
                severity="error",
                symptom=_safe_error(exc),
                suggested_repair="Inspect traceback, repair stage implementation or input contract, then rerun the same stage.",
            )
        self.ctx.flush_common()
        return False

    def _record_gpu(self, event: str) -> None:
        if not bool(self.ctx.config.get("profiling", {}).get("use_nvml", True)):
            return
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            for gpu_id in self.ctx.config.get("run", {}).get("gpu_ids", []):
                handle = pynvml.nvmlDeviceGetHandleByIndex(int(gpu_id))
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                self.ctx.gpu_rows.append(
                    {
                        "schema_version": "stream4d_v105_gpu_memory_row_v1",
                        "run_id": self.ctx.config["run"]["name"],
                        "stage_name": self.stage_name,
                        "event": event,
                        "gpu_index": int(gpu_id),
                        "memory_used_mb": float(mem.used) / (1024.0 * 1024.0),
                        "memory_total_mb": float(mem.total) / (1024.0 * 1024.0),
                        "timestamp_unix": time.time(),
                    }
                )
        except Exception as exc:
            if event == "start":
                self.ctx.gpu_rows.append(
                    {
                        "schema_version": "stream4d_v105_gpu_memory_row_v1",
                        "run_id": self.ctx.config["run"]["name"],
                        "stage_name": self.stage_name,
                        "event": event,
                        "gpu_index": "",
                        "memory_used_mb": "",
                        "memory_total_mb": "",
                        "nvml_error": _safe_error(exc),
                        "timestamp_unix": time.time(),
                    }
                )


def build_provider_registry(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    providers = config.get("providers", {})
    mask_candidates = providers.get("mask", {}).get("candidates", [])
    tracker_candidates = providers.get("tracker", {}).get("candidates", [])
    for name in mask_candidates:
        rows.append(
            {
                "schema_version": "stream4d_v105_provider_registry_row_v1",
                "provider_role": "mask",
                "provider_name": name,
                "configured_default": name == providers.get("mask", {}).get("name"),
                "enabled": True,
                "adapter_required": True,
            }
        )
    for name in tracker_candidates:
        rows.append(
            {
                "schema_version": "stream4d_v105_provider_registry_row_v1",
                "provider_role": "tracker",
                "provider_name": name,
                "configured_default": name == providers.get("tracker", {}).get("name"),
                "enabled": True,
                "adapter_required": True,
            }
        )
    for role in ("geometry", "correspondence", "semantic"):
        info = providers.get(role, {})
        rows.append(
            {
                "schema_version": "stream4d_v105_provider_registry_row_v1",
                "provider_role": role,
                "provider_name": info.get("name", role),
                "configured_default": True,
                "enabled": bool(info.get("enabled", True)),
                "adapter_required": True,
            }
        )
    return rows


def prepare_context(args: argparse.Namespace) -> PipelineContext:
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    config = _load_yaml(config_path)
    overrides: dict[str, Any] = {"run": {}, "paths": {}}
    if args.split:
        overrides["run"]["split"] = args.split
    scenes = _parse_comma_list(args.scenes)
    if scenes is not None:
        overrides["run"]["scenes"] = scenes
    if args.chunk_size is not None:
        overrides["run"]["chunk_size"] = int(args.chunk_size)
    if args.frame_stride is not None:
        overrides["run"]["frame_stride"] = int(args.frame_stride)
    if args.overlap is not None:
        overrides["run"]["overlap"] = int(args.overlap)
    if args.cache_mode:
        overrides["run"]["cache_mode"] = args.cache_mode
    if args.max_frames_per_scene is not None:
        overrides["run"]["max_frames_per_scene"] = int(args.max_frames_per_scene)
    if args.scannet_processed_root:
        overrides["paths"]["scannet_processed_root"] = args.scannet_processed_root
    if args.construct_sam2_models:
        overrides.setdefault("provider_smoke", {})["construct_sam2_models"] = True
    overrides = {k: v for k, v in overrides.items() if v}
    _deep_update(config, overrides)

    stages = _parse_comma_list(args.stages) or ["phase0", "preprocess", "provider_smoke"]
    output_root = Path(args.output_root) if args.output_root else DEFAULT_OUTPUT
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root
    resume_existing_output = set(stages).issubset({"local2history", "full_dev", "holdout", "casebook"})
    if output_root.exists() and any(output_root.iterdir()) and not args.force and not resume_existing_output:
        raise SystemExit(f"output root exists and is non-empty; pass --force to overwrite v105 artifacts: {_rel(output_root)}")
    output_root.mkdir(parents=True, exist_ok=True)

    ctx = PipelineContext(config=config, config_path=config_path, output_root=output_root, stages=stages, force=args.force)
    ctx.config_hash = _hash_payload(config)
    ctx.provider_rows = build_provider_registry(config)

    for subdir in [
        "preprocessing",
        "fourdpm_code_audit",
        "geometry",
        "masks",
        "tracking",
        "alltracker",
        "query_planning",
        "local_objects",
        "local2history",
        "evaluation",
        "videos",
        "diagnostics",
        "casebook",
        "phase0",
        "provider_smoke",
        "baselines",
        "sgq_local",
        "full_dev",
        "holdout",
    ]:
        (output_root / subdir).mkdir(parents=True, exist_ok=True)

    _write_yaml(output_root / "config_resolved.yaml", config)
    (output_root / "config_hash.txt").write_text(ctx.config_hash + "\n", encoding="utf-8")
    (output_root / "last_command.txt").write_text(" ".join([sys.executable, str(RUNNER_PATH), *sys.argv[1:]]) + "\n", encoding="utf-8")
    ctx.flush_common()
    ctx.write_summary()
    return ctx


def audit_fourdpm(ctx: PipelineContext) -> dict[str, Any]:
    configured = _as_repo_path(ctx.config.get("paths", {}).get("fourdpm_root"))
    candidates = [
        configured,
        REPO_ROOT / "third_party" / "4D_PM",
        REPO_ROOT / "third_party" / "4DPM",
        REPO_ROOT / "third_party" / "4dpm",
    ]
    root = next((p for p in candidates if p is not None and p.exists()), None)
    out_dir = ctx.output_root / "fourdpm_code_audit"
    phase_dir = ctx.output_root / "phase0"
    if root is None:
        row = {
            "schema_version": "stream4d_v105_fourdpm_code_audit_row_v1",
            "audit_status": "blocked",
            "failure_type": "FOURDPM_CODE_AUDIT_BLOCKED",
            "searched_paths": [_rel(p) for p in candidates if p is not None],
        }
        _write_records_json(phase_dir / "fourdpm_code_audit_records.json", [row])
        _write_records_json(phase_dir / "sam2_callsite_records.json", [])
        _write_records_json(phase_dir / "gap_sampling_records.json", [])
        _write_json(out_dir / "summary.json", row)
        ctx.add_failure(
            stage_name="phase0",
            failure_type="FOURDPM_CODE_AUDIT_BLOCKED",
            severity="blocker",
            symptom="third_party/4D_PM and fallback 4DPM paths do not exist",
            suggested_repair="Restore or clone the local 4DPM code, then rerun Phase 0.",
        )
        return row

    files = {
        "samv2_tools": root / "frontend" / "segment" / "samv2_tools.py",
        "video_matcher": root / "frontend" / "segment" / "video_matcher.py",
        "process_sequence": root / "frontend" / "process_sequence.py",
        "frontend_utils": root / "frontend" / "utils.py",
        "run_pipeline": root / "run_pipeline.py",
    }
    sam2_rows: list[dict[str, Any]] = []
    gap_rows: list[dict[str, Any]] = []
    for file_path in files.values():
        sam2_rows.extend(
            _line_hits(
                file_path,
                {
                    "sam2_image_predictor": r"SAM2ImagePredictor",
                    "build_sam2": r"build_sam2\(",
                    "build_sam2_video_predictor": r"build_sam2_video_predictor",
                    "sam2_checkpoint": r"sam2_checkpoint|checkpoint",
                    "video_predictor": r"video_predictor|predictor",
                },
            )
        )
        gap_rows.extend(
            _line_hits(
                file_path,
                {
                    "run_video_overseg": r"def run_video_overseg",
                    "first_frame_oversegment": r"oversegment\(",
                    "propagate_maskbundle": r"propagate_maskbundle",
                    "sample_gap_masks": r"sample_gap_masks",
                    "gap_comment": r"uncovered|gap|Fill uncovered",
                    "forward_propagation": r"propagate_in_video|future",
                    "disjoint_overlap_suppression": r"disjoin_segments",
                },
            )
        )
    _write_records_json(out_dir / "sam2_callsite_records.json", sam2_rows)
    _write_records_json(out_dir / "gap_sampling_records.json", gap_rows)
    _write_records_json(phase_dir / "sam2_callsite_records.json", sam2_rows)
    _write_records_json(phase_dir / "gap_sampling_records.json", gap_rows)

    answers = {
        "initializes_first_frame_masks": any(row["role"] == "first_frame_oversegment" for row in gap_rows),
        "uses_sam2_or_samv2": any("sam2" in str(row["role"]) for row in sam2_rows),
        "propagates_masks_to_future_frames": any(row["role"] in {"propagate_maskbundle", "forward_propagation"} for row in gap_rows),
        "detects_uncovered_gap_regions": any(row["role"] in {"sample_gap_masks", "gap_comment"} for row in gap_rows),
        "initializes_new_mask_in_gap": any(row["role"] == "sample_gap_masks" for row in gap_rows),
        "new_mask_forward_propagates": any(row["role"] == "forward_propagation" for row in gap_rows),
        "suppresses_duplicate_or_overlap_masks": any(row["role"] == "disjoint_overlap_suppression" for row in gap_rows),
        "runtime_memory_recording_found": any("measure_peak_mb" in row.get("code_excerpt", "") for row in gap_rows + sam2_rows),
    }
    completed = all(
        [
            answers["initializes_first_frame_masks"],
            answers["uses_sam2_or_samv2"],
            answers["propagates_masks_to_future_frames"],
            answers["detects_uncovered_gap_regions"],
            answers["initializes_new_mask_in_gap"],
            answers["new_mask_forward_propagates"],
        ]
    )
    audit_rows = [
        {
            "schema_version": "stream4d_v105_fourdpm_code_audit_row_v1",
            "audit_status": "completed" if completed else "partial",
            "fourdpm_root": _rel(root),
            "file_role": role,
            "path": _rel(path),
            "exists": path.exists(),
            "sha256": _sha256_file(path),
        }
        for role, path in files.items()
    ]
    summary = {
        "schema_version": "stream4d_v105_fourdpm_code_audit_summary_v1",
        "audit_status": "completed" if completed else "partial",
        "fourdpm_root": _rel(root),
        "answers": answers,
        "sam2_callsite_count": len(sam2_rows),
        "gap_sampling_evidence_count": len(gap_rows),
        "code_reproduced_baseline_run": False,
        "paper_compatible_baseline_allowed_if_runtime_unavailable": True,
        "runtime_baseline_status": "not_run_in_phase0",
    }
    _write_records_json(phase_dir / "fourdpm_code_audit_records.json", audit_rows)
    _write_json(out_dir / "summary.json", summary)
    _write_json(out_dir / "segmentation_frontend_summary.json", summary)
    return summary


def audit_specgap_module_design_docs(ctx: PipelineContext) -> dict[str, Any]:
    phase_dir = ctx.output_root / "phase0"
    docs_root = REPO_ROOT / "docs" / "v105" / "modules"
    records: list[dict[str, Any]] = []
    missing_docs: list[str] = []
    section_failures: list[str] = []
    for filename in SPECGAP_MODULE_DOCS:
        path = docs_root / filename
        exists = path.exists()
        text = path.read_text(encoding="utf-8", errors="replace") if exists else ""
        missing_sections = [section for section in SPECGAP_REQUIRED_DOC_SECTIONS if section not in text]
        if not exists:
            missing_docs.append(filename)
        if exists and missing_sections:
            section_failures.append(filename)
        records.append(
            {
                "schema_version": "stream4d_v105_module_design_doc_row_v1",
                "doc_name": filename,
                "path": _rel(path),
                "exists": exists,
                "sha256": _sha256_file(path),
                "line_count": len(text.splitlines()) if exists else 0,
                "required_sections": SPECGAP_REQUIRED_DOC_SECTIONS,
                "missing_sections": missing_sections,
                "section_gate_pass": exists and not missing_sections,
            }
        )
    summary = {
        "schema_version": "stream4d_v105_module_design_doc_status_v1",
        "docs_root": _rel(docs_root),
        "required_doc_count": len(SPECGAP_MODULE_DOCS),
        "present_doc_count": sum(1 for row in records if row["exists"]),
        "all_required_docs_present": not missing_docs,
        "all_required_sections_present": not section_failures,
        "missing_docs": missing_docs,
        "section_failure_docs": section_failures,
        "phase0_design_doc_gate_pass": not missing_docs and not section_failures,
    }
    _write_records_json(phase_dir / "module_design_doc_records.json", records)
    _write_json(phase_dir / "module_design_doc_status.json", summary)
    if not summary["phase0_design_doc_gate_pass"]:
        ctx.add_failure(
            stage_name="phase0",
            failure_type="MODULE_DESIGN_DOCS_MISSING",
            severity="blocker",
            symptom=f"module design docs missing or incomplete: {summary}",
            suggested_repair="Create every docs/v105/modules design document with Responsibility, I/O, state, causality, memory, cache, API, failure, metrics, and tests sections before implementing modules.",
        )
    return summary


def audit_baseline_x_code_contract(ctx: PipelineContext) -> dict[str, Any]:
    phase_dir = ctx.output_root / "phase0"
    cfg = ctx.config.get("baseline_x", {})
    runner = REPO_ROOT / str(cfg.get("parity_reference_runner", "tools/audit_v105_baseline_x_sam2_twostage_tracking.py"))
    x0_config = _as_repo_path(str(cfg.get("x0_config", "configs/v105/baseline_chunk_table/baseline_x_sam2_twostage_sam2.generated.yaml")))
    x1_config = _as_repo_path(str(cfg.get("x1_config", "configs/v105/baseline_chunk_table/baseline_x_gapadaptive_sam2.generated.yaml")))
    fourdpm_root = _as_repo_path(ctx.config.get("paths", {}).get("fourdpm_root")) or (REPO_ROOT / "third_party" / "4D_PM")
    required_files = {
        "baseline_x_runner": runner,
        "x0_config": x0_config,
        "x1_gapadaptive_config": x1_config,
        "4dpm_video_matcher": fourdpm_root / "frontend" / "segment" / "video_matcher.py",
        "4dpm_oversegmentation": fourdpm_root / "frontend" / "segment" / "oversegmentation.py",
        "4dpm_infer": fourdpm_root / "frontend" / "segment" / "infer.py",
        "4dpm_active_sampling": fourdpm_root / "frontend" / "segment" / "active_sampling.py",
    }
    file_rows = [
        {
            "schema_version": "stream4d_v105_baseline_x_contract_file_row_v1",
            "role": role,
            "path": _rel(path),
            "exists": path.exists(),
            "sha256": _sha256_file(path),
        }
        for role, path in required_files.items()
    ]
    runner_hits = _line_hits(
        runner,
        {
            "frame0_stage1": r"stage1|frame0_stage1",
            "frame0_stage2_uncovered": r"stage2|frame0_stage2",
            "largest_valid": r"largest_valid",
            "smallest_valid": r"smallest_valid",
            "component_adaptive_gap": r"component_adaptive",
            "propagate_new_masks": r"propagate_new_masks",
            "reseed_full_video": r"reseed_full_video",
            "sheets_and_video": r"make_sheet_grid|write_video",
        },
    )
    x0_payload = _load_yaml(x0_config) if x0_config is not None and x0_config.exists() else {}
    x1_payload = _load_yaml(x1_config) if x1_config is not None and x1_config.exists() else {}
    x0_ok = bool(x0_payload) and x0_payload.get("baseline", {}).get("id") in {"baseline-x", "baseline-x-sam2-twostage"}
    x1_ok = bool(x1_payload) and x1_payload.get("gap", {}).get("sampler") == "component_adaptive"
    required_files_exist = all(row["exists"] for row in file_rows)
    runner_semantics_ok = all(
        any(hit["role"] == role for hit in runner_hits)
        for role in ["frame0_stage1", "frame0_stage2_uncovered", "largest_valid", "smallest_valid", "propagate_new_masks"]
    )
    summary = {
        "schema_version": "stream4d_v105_baseline_x_code_contract_v1",
        "required_files_exist": required_files_exist,
        "runner_semantics_ok": runner_semantics_ok,
        "x0_config_ok": x0_ok,
        "x1_gapadaptive_config_ok": x1_ok,
        "x0_config": _rel(x0_config) if x0_config else "",
        "x1_config": _rel(x1_config) if x1_config else "",
        "x0_variant": x0_payload.get("baseline", {}).get("variant", ""),
        "x1_variant": x1_payload.get("baseline", {}).get("variant", ""),
        "x1_gap_sampler": x1_payload.get("gap", {}).get("sampler", ""),
        "baseline_x_contract_gate_pass": bool(required_files_exist and runner_semantics_ok and x0_ok and x1_ok),
        "note": "This is a Phase 0 code/config contract audit only; it does not generate new baseline metrics.",
    }
    _write_records_json(phase_dir / "baseline_x_contract_file_records.json", file_rows)
    _write_records_json(phase_dir / "baseline_x_runner_semantic_hit_records.json", runner_hits)
    _write_json(phase_dir / "baseline_x_code_contract.json", summary)
    if not summary["baseline_x_contract_gate_pass"]:
        ctx.add_failure(
            stage_name="phase0",
            failure_type="BASELINE_X_CONTRACT_BLOCKED",
            severity="blocker",
            symptom=f"baseline-x contract audit failed: {summary}",
            suggested_repair="Read and repair the baseline-x runner/config paths before any feature-bank or speculative-gap stage.",
        )
    return summary


def audit_sam2_real_api(ctx: PipelineContext) -> dict[str, Any]:
    phase_dir = ctx.output_root / "phase0"
    sam2_root = REPO_ROOT / "Grounded-SAM-2" / "sam2"
    files = {
        "sam2_image_predictor": sam2_root / "sam2_image_predictor.py",
        "sam2_video_predictor": sam2_root / "sam2_video_predictor.py",
    }
    rows: list[dict[str, Any]] = []
    for path in files.values():
        rows.extend(
            _line_hits(
                path,
                {
                    "class_sam2_image_predictor": r"class SAM2ImagePredictor",
                    "image_set_image": r"def set_image\(",
                    "image_set_image_batch": r"def set_image_batch\(",
                    "image_predict": r"def predict\(",
                    "class_sam2_video_predictor": r"class SAM2VideoPredictor",
                    "video_init_state": r"def init_state\(",
                    "video_add_new_mask": r"def add_new_mask\(",
                    "video_propagate_in_video": r"def propagate_in_video\(",
                    "video_get_image_feature": r"def _get_image_feature\(",
                    "new_object_before_tracking_started": r"tracking_has_started|allow_new_object",
                    "cached_features": r"cached_features",
                    "forward_image": r"forward_image",
                    "prepare_backbone_features": r"_prepare_backbone_features",
                },
            )
        )
    file_rows = [
        {
            "schema_version": "stream4d_v105_sam2_api_file_row_v1",
            "role": role,
            "path": _rel(path),
            "exists": path.exists(),
            "sha256": _sha256_file(path),
        }
        for role, path in files.items()
    ]
    required_roles = [
        "class_sam2_image_predictor",
        "image_set_image",
        "class_sam2_video_predictor",
        "video_init_state",
        "video_add_new_mask",
        "video_propagate_in_video",
        "video_get_image_feature",
        "cached_features",
    ]
    role_set = {row["role"] for row in rows}
    summary = {
        "schema_version": "stream4d_v105_sam2_real_api_audit_v1",
        "api_files_exist": all(row["exists"] for row in file_rows),
        "required_roles": required_roles,
        "missing_roles": [role for role in required_roles if role not in role_set],
        "sam2_api_gate_pass": all(row["exists"] for row in file_rows) and all(role in role_set for role in required_roles),
        "feature_access_hook": "video_get_image_feature",
        "image_feature_reference": "SAM2ImagePredictor.set_image / set_image_batch",
        "new_object_constraint_observed": "new_object_before_tracking_started" in role_set,
        "note": "Phase 0 API audit only; feature-bank parity must still prove output equivalence.",
    }
    _write_records_json(phase_dir / "sam2_api_file_records.json", file_rows)
    _write_records_json(phase_dir / "sam2_api_callsite_records.json", rows)
    _write_json(phase_dir / "sam2_api_audit_summary.json", summary)
    if not summary["sam2_api_gate_pass"]:
        ctx.add_failure(
            stage_name="phase0",
            failure_type="SAM2_API_AUDIT_BLOCKED",
            severity="blocker",
            symptom=f"SAM2 API audit failed: {summary}",
            suggested_repair="Locate SAM2ImagePredictor/SAM2VideoPredictor APIs and verify feature access before implementing feature bank.",
        )
    return summary


def run_phase0(ctx: PipelineContext) -> None:
    with StageTimer(ctx, "phase0"):
        phase_dir = ctx.output_root / "phase0"
        fourdpm_summary = audit_fourdpm(ctx)
        module_doc_summary = audit_specgap_module_design_docs(ctx)
        baseline_contract = audit_baseline_x_code_contract(ctx)
        sam2_api_summary = audit_sam2_real_api(ctx)
        run_contract = {
            "schema_version": "stream4d_v105_specgap_run_contract_v1",
            "run_id": ctx.config["run"]["name"],
            "causality_scope": ctx.config.get("run", {}).get("causality_scope", "chunk_causal"),
            "frame_causal": bool(ctx.config.get("run", {}).get("frame_causal", False)),
            "uses_next_chunk_nonoverlap_frames": bool(ctx.config.get("run", {}).get("uses_next_chunk_nonoverlap_frames", False)),
            "final_mask_model": "SAM2.1-Hiera-L",
            "final_tracking_model": "SAM2.1-Hiera-L",
            "final_refinement_model": "SAM2.1-Hiera-L",
            "alltracker_role": "approximate correspondence and proxy coverage only",
            "lingbot_role": "view novelty and soft 3D/history support only",
            "mv_ap_role": "diagnostic_not_gate",
            "visual_gate_primary": True,
            "cache_policy_for_final": "write_only_verified_no_read / cross_run_cache_read_count=0",
        }
        _write_json(phase_dir / "run_contract.json", run_contract)
        skeleton_rows = [
            {
                "schema_version": "stream4d_v105_pipeline_skeleton_row_v1",
                "runner_path": _rel(RUNNER_PATH),
                "runner_exists": RUNNER_PATH.exists(),
                "config_path": _rel(ctx.config_path),
                "config_parsed": True,
                "provider_registry_constructed": bool(ctx.provider_rows),
                "single_user_entrypoint": True,
                "package_path": _rel(STREAM3D_ROOT / "sgq_v105"),
                "package_exists": (STREAM3D_ROOT / "sgq_v105" / "__init__.py").exists(),
                "specgap_config": _rel(ctx.config_path),
                "specgap_chunk_causal_contract": run_contract["causality_scope"] == "chunk_causal",
            }
        ]
        _write_records_json(phase_dir / "pipeline_skeleton_records.json", skeleton_rows)
        _write_records_json(phase_dir / "failure_records.json", [row for row in ctx.failure_rows if row.get("stage_name") == "phase0"])
        gate_pass = (
            bool(skeleton_rows[0]["runner_exists"])
            and bool(skeleton_rows[0]["config_parsed"])
            and bool(skeleton_rows[0]["provider_registry_constructed"])
            and bool(module_doc_summary.get("phase0_design_doc_gate_pass"))
            and bool(baseline_contract.get("baseline_x_contract_gate_pass"))
            and bool(sam2_api_summary.get("sam2_api_gate_pass"))
            and fourdpm_summary.get("audit_status") in {"completed", "blocked", "partial"}
        )
        if fourdpm_summary.get("audit_status") == "partial":
            ctx.add_failure(
                stage_name="phase0",
                failure_type="FOURDPM_CODE_AUDIT_PARTIAL",
                severity="warning",
                symptom="4DPM code exists but one or more planned frontend audit questions lack direct evidence rows.",
                suggested_repair="Inspect third_party/4D_PM frontend files and extend audit patterns or baseline wrapper before claiming code-reproduced 4DPM baseline.",
            )
        ctx.gates["phase0_pass"] = bool(gate_pass)
        ctx.gates["phase0_no_ap_generated"] = True
        _write_json(
            phase_dir / "summary.json",
            {
                "schema_version": "stream4d_v105_phase0_summary_v1",
                "phase0_pass": bool(gate_pass),
                "runner_exists": RUNNER_PATH.exists(),
                "config_parsed": True,
                "provider_registry_constructed": bool(ctx.provider_rows),
                "fourdpm_code_audit_status": fourdpm_summary.get("audit_status"),
                "module_design_doc_status": module_doc_summary,
                "baseline_x_contract_gate_pass": baseline_contract.get("baseline_x_contract_gate_pass"),
                "sam2_api_gate_pass": sam2_api_summary.get("sam2_api_gate_pass"),
                "run_contract": run_contract,
                "no_ap_generated": True,
            },
        )
        ctx.write_summary()


def _selected_frame_ids(scene_root: Path, stride: int, max_frames: int) -> list[int]:
    color_dir = scene_root / "color"
    files = sorted(color_dir.glob("*.jpg"), key=_numeric_stem)
    if not files:
        files = sorted(color_dir.glob("*.png"), key=_numeric_stem)
    ids = [_numeric_stem(path) for path in files if _numeric_stem(path) < 10**12]
    ids = ids[:: max(int(stride), 1)]
    if max_frames > 0:
        ids = ids[: int(max_frames)]
    return ids


def _transform_frame(
    *,
    scene_root: Path,
    scene_id: str,
    frame_id: int,
    target_hw: tuple[int, int],
    fixed_crop_px: int,
    auto_crop_fallback: bool,
    black_threshold: int,
    overlay_frame_path: Path | None,
) -> tuple[dict[str, Any], dict[str, Any], np.ndarray | None, float | None]:
    color_path = scene_root / "color" / f"{frame_id}.jpg"
    if not color_path.exists():
        color_path = scene_root / "color" / f"{frame_id}.png"
    rgb = _read_image(color_path, cv2.IMREAD_COLOR)
    if rgb is None:
        raise FileNotFoundError(f"missing RGB frame: {color_path}")
    h, w = rgb.shape[:2]
    crop = (fixed_crop_px, fixed_crop_px, max(w - fixed_crop_px, fixed_crop_px + 1), max(h - fixed_crop_px, fixed_crop_px + 1))
    crop_policy_used = "fixed_black_padding"
    cropped = rgb[crop[1] : crop[3], crop[0] : crop[2]].copy()
    final_rgb = _resize(cropped, target_hw, cv2.INTER_LINEAR)
    black_before = _black_ratio(rgb, black_threshold)
    black_after = _black_ratio(final_rgb, black_threshold)
    if auto_crop_fallback and black_after > 0.005:
        detected = _detect_nonblack_crop(rgb, black_threshold, margin=0)
        if detected is not None:
            trial = _resize(rgb[detected[1] : detected[3], detected[0] : detected[2]], target_hw, cv2.INTER_LINEAR)
            trial_ratio = _black_ratio(trial, black_threshold)
            if trial_ratio < black_after:
                crop = detected
                final_rgb = trial
                black_after = trial_ratio
                crop_policy_used = "auto_nonblack_bbox"
    x0, y0, x1, y1 = crop

    sem_path = scene_root / "label-filt" / f"{frame_id}.png"
    inst_path = scene_root / "instance" / "instance" / f"{frame_id}.png"
    cropformer_path = scene_root / "output_Cropformer" / "mask" / f"{frame_id}.png"
    depth_path = scene_root / "depth" / f"{frame_id}.png"
    semantic = _read_image(sem_path, cv2.IMREAD_UNCHANGED)
    instance = _read_image(inst_path, cv2.IMREAD_UNCHANGED)
    crop_mask = _read_image(cropformer_path, cv2.IMREAD_UNCHANGED)
    depth = _read_image(depth_path, cv2.IMREAD_UNCHANGED)

    final_semantic = None
    final_instance = None
    final_crop_mask = None
    final_depth = None
    if semantic is not None:
        final_semantic = _resize(_crop_scaled(semantic, crop, (h, w))[0], target_hw, cv2.INTER_NEAREST)
    if instance is not None:
        final_instance = _resize(_crop_scaled(instance, crop, (h, w))[0], target_hw, cv2.INTER_NEAREST)
    if crop_mask is not None:
        final_crop_mask = _resize(_crop_scaled(crop_mask, crop, (h, w))[0], target_hw, cv2.INTER_NEAREST)
    if depth is not None:
        final_depth = _resize(_crop_scaled(depth, crop, (h, w))[0], target_hw, cv2.INTER_NEAREST)

    before_nonzero = float(np.count_nonzero(crop_mask)) / float(crop_mask.size) if crop_mask is not None and crop_mask.size else None
    after_nonzero = float(np.count_nonzero(final_crop_mask)) / float(final_crop_mask.size) if final_crop_mask is not None and final_crop_mask.size else None
    mask_area_change_ratio = None
    if before_nonzero and after_nonzero is not None:
        mask_area_change_ratio = after_nonzero / max(before_nonzero, 1e-12)

    intr_path = scene_root / "intrinsic" / "intrinsic_color.txt"
    k = _load_intrinsics(intr_path)
    k_updated = _update_intrinsics(k, crop, target_hw)
    intr_row = {
        "schema_version": "stream4d_v105_intrinsics_row_v1",
        "scene_id": scene_id,
        "frame_id": int(frame_id),
        "intrinsics_path": _rel(intr_path),
        "intrinsics_read": k is not None,
        "intrinsics_updated": k_updated is not None,
        "fx": float(k[0, 0]) if k is not None else "",
        "fy": float(k[1, 1]) if k is not None else "",
        "cx": float(k[0, 2]) if k is not None else "",
        "cy": float(k[1, 2]) if k is not None else "",
        "fx_updated": float(k_updated[0, 0]) if k_updated is not None else "",
        "fy_updated": float(k_updated[1, 1]) if k_updated is not None else "",
        "cx_updated": float(k_updated[0, 2]) if k_updated is not None else "",
        "cy_updated": float(k_updated[1, 2]) if k_updated is not None else "",
    }

    label_for_overlay = final_crop_mask if final_crop_mask is not None else final_instance
    overlay = _overlay(final_rgb, label_for_overlay)
    if overlay_frame_path is not None:
        overlay_frame_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(overlay_frame_path), overlay)

    meta = {
        "schema_version": "stream4d_v105_preprocess_meta_row_v1",
        "scene_id": scene_id,
        "frame_id": int(frame_id),
        "rgb_path": _rel(color_path),
        "original_height": int(h),
        "original_width": int(w),
        "crop_x0": int(x0),
        "crop_y0": int(y0),
        "crop_x1": int(x1),
        "crop_y1": int(y1),
        "crop_policy_used": crop_policy_used,
        "scale_x": float(target_hw[1]) / float(max(x1 - x0, 1)),
        "scale_y": float(target_hw[0]) / float(max(y1 - y0, 1)),
        "final_height": int(final_rgb.shape[0]),
        "final_width": int(final_rgb.shape[1]),
        "black_border_ratio_before": black_before,
        "black_border_ratio_after": black_after,
        "rgb_transform_hash": _hash_payload({"source": _sha256_file(color_path), "crop": crop, "target_hw": target_hw, "resize": "bilinear"}),
        "mask_transform_hash": _hash_payload({"source": _sha256_file(cropformer_path), "crop": crop, "target_hw": target_hw, "resize": "nearest"}) if cropformer_path.exists() else "",
        "semantic_transform_hash": _hash_payload({"source": _sha256_file(sem_path), "crop": crop, "target_hw": target_hw, "resize": "nearest"}) if sem_path.exists() else "",
        "depth_transform_hash": _hash_payload({"source": _sha256_file(depth_path), "crop": crop, "target_hw": target_hw, "resize": "nearest"}) if depth_path.exists() else "",
        "intrinsics_updated": k_updated is not None,
        "semantic_uses_nearest_resize": semantic is not None,
        "mask_uses_nearest_resize": crop_mask is not None,
        "depth_uses_nearest_resize": depth is not None,
        "mask_area_change_ratio": mask_area_change_ratio if mask_area_change_ratio is not None else "",
        "rgb_exists": True,
        "cropformer_mask_exists": crop_mask is not None,
        "semantic_exists": semantic is not None,
        "instance_exists": instance is not None,
        "depth_exists": depth is not None,
    }
    return meta, intr_row, overlay, mask_area_change_ratio


def run_preprocess(ctx: PipelineContext) -> None:
    with StageTimer(ctx, "preprocess"):
        cfg = ctx.config
        pp = cfg.get("preprocess", {})
        run_cfg = cfg.get("run", {})
        target_hw = (int(pp.get("target_height", 240)), int(pp.get("target_width", 320)))
        fixed_crop_px = int(pp.get("crop_black_padding_px", 15))
        stride = int(run_cfg.get("frame_stride", 5))
        max_frames = int(run_cfg.get("max_frames_per_scene", run_cfg.get("chunk_size", 32)))
        scannet_root = _as_repo_path(cfg.get("paths", {}).get("scannet_processed_root")) or (STREAM3D_ROOT / "data" / "scannet" / "processed")
        out_dir = ctx.output_root / "preprocessing"
        overlay_dir = out_dir / "overlay_sanity_frames"
        meta_rows: list[dict[str, Any]] = []
        intr_rows: list[dict[str, Any]] = []
        overlays: list[np.ndarray] = []
        area_ratios: list[float] = []
        scenes = [str(scene) for scene in run_cfg.get("scenes", [])]
        for scene_id in scenes:
            scene_root = scannet_root / scene_id
            if not scene_root.exists():
                ctx.add_failure(
                    stage_name="preprocess",
                    scene_id=scene_id,
                    failure_type="PREPROCESS_ALIGNMENT_FAILURE",
                    severity="blocker",
                    symptom=f"scene root does not exist: {_rel(scene_root)}",
                    suggested_repair="Fix paths.scannet_processed_root or scene list before provider/model stages.",
                )
                continue
            frame_ids = _selected_frame_ids(scene_root, stride, max_frames)
            if not frame_ids:
                ctx.add_failure(
                    stage_name="preprocess",
                    scene_id=scene_id,
                    failure_type="PREPROCESS_ALIGNMENT_FAILURE",
                    severity="blocker",
                    symptom=f"no color frames found under {_rel(scene_root / 'color')}",
                    suggested_repair="Restore ScanNet processed color frames before provider/model stages.",
                )
                continue
            for frame_id in frame_ids:
                overlay_path = overlay_dir / f"{scene_id}_{frame_id:06d}.jpg" if pp.get("export_overlay_frames", True) else None
                try:
                    meta, intr, overlay, area_ratio = _transform_frame(
                        scene_root=scene_root,
                        scene_id=scene_id,
                        frame_id=frame_id,
                        target_hw=target_hw,
                        fixed_crop_px=fixed_crop_px,
                        auto_crop_fallback=bool(pp.get("auto_crop_fallback", True)),
                        black_threshold=int(pp.get("black_pixel_threshold", 5)),
                        overlay_frame_path=overlay_path,
                    )
                    meta_rows.append(meta)
                    intr_rows.append(intr)
                    overlays.append(overlay)
                    if area_ratio is not None and np.isfinite(area_ratio):
                        area_ratios.append(float(area_ratio))
                except Exception as exc:
                    ctx.add_failure(
                        stage_name="preprocess",
                        scene_id=scene_id,
                        frame_id=int(frame_id),
                        failure_type="PREPROCESS_ALIGNMENT_FAILURE",
                        severity="blocker",
                        symptom=_safe_error(exc),
                        suggested_repair="Check RGB/mask/depth/intrinsics file presence and transform code; rerun preprocessing only.",
                    )

        _write_records_json(out_dir / "preprocess_meta_records.json", meta_rows)
        _write_records_json(out_dir / "intrinsics_records.json", intr_rows)
        video_path = out_dir / "overlay_sanity_video.mp4"
        video_ok = False
        if overlays:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(video_path), fourcc, float(ctx.config.get("video_export", {}).get("fps", 8)), (target_hw[1], target_hw[0]))
            if writer.isOpened():
                for frame in overlays:
                    writer.write(frame)
                writer.release()
                video_ok = video_path.exists() and video_path.stat().st_size > 0
            else:
                ctx.add_failure(
                    stage_name="preprocess",
                    failure_type="VIDEO_EXPORT_FAILURE",
                    severity="blocker",
                    symptom=f"cv2.VideoWriter failed to open {_rel(video_path)}",
                    suggested_repair="Check OpenCV codec support or switch video codec/container before claiming preprocessing pass.",
                )

        final_size_pass = bool(meta_rows) and all(int(row["final_height"]) == target_hw[0] and int(row["final_width"]) == target_hw[1] for row in meta_rows)
        black_max = max([float(row["black_border_ratio_after"]) for row in meta_rows], default=1.0)
        black_pass = black_max <= float(pp.get("black_border_ratio_after_max", 0.005))
        semantic_nearest_pass = bool(meta_rows) and all(bool(row["semantic_uses_nearest_resize"]) for row in meta_rows)
        mask_delta_p90 = float(np.percentile(np.abs(np.asarray(area_ratios) - 1.0), 90)) if area_ratios else None
        mask_area_pass = bool(area_ratios) and mask_delta_p90 is not None and mask_delta_p90 <= float(pp.get("mask_area_change_ratio_p90_abs_delta_max", 0.35))
        projection_sanity_available = False
        projection_sanity_pass = True
        preprocess_pass = final_size_pass and black_pass and semantic_nearest_pass and mask_area_pass and video_ok and projection_sanity_pass
        summary = {
            "schema_version": "stream4d_v105_preprocess_summary_v1",
            "preprocess_pass": bool(preprocess_pass),
            "scene_count": len(scenes),
            "frame_count": len(meta_rows),
            "target_height": target_hw[0],
            "target_width": target_hw[1],
            "final_size_pass": bool(final_size_pass),
            "black_border_ratio_after_max": black_max,
            "black_border_pass": bool(black_pass),
            "semantic_nearest_resize_pass": bool(semantic_nearest_pass),
            "mask_area_change_ratio_p90_abs_delta": mask_delta_p90,
            "mask_area_change_pass": bool(mask_area_pass),
            "projection_sanity_available": projection_sanity_available,
            "projection_sanity_error_px_p90": None,
            "projection_sanity_pass": projection_sanity_pass,
            "overlay_sanity_video": _rel(video_path) if video_ok else "",
            "overlay_sanity_video_pass": bool(video_ok),
            "cache_read_count": 0,
            "uses_gt_for_prediction": False,
            "uses_future": False,
        }
        _write_json(out_dir / "preprocess_summary.json", summary)
        if not preprocess_pass:
            ctx.add_failure(
                stage_name="preprocess",
                failure_type="PREPROCESS_ALIGNMENT_FAILURE",
                severity="blocker",
                symptom=f"preprocess gate failed: {summary}",
                suggested_repair="Follow plan repair order: fixed crop -> auto black-border detection -> center crop/resize-only with updated intrinsics -> inspect overlay video.",
            )
        ctx.gates["phase1_preprocess_pass"] = bool(preprocess_pass)
        ctx.gates["phase1_preprocess_summary"] = summary
        ctx.write_summary()


def _smoke_cropformer(ctx: PipelineContext) -> dict[str, Any]:
    t0 = time.time()
    demo_script = STREAM3D_ROOT / "third_party" / "detectron2" / "projects" / "CropFormer" / "demo_cropformer" / "Cropformer.py"
    config_path = STREAM3D_ROOT / "third_party" / "detectron2" / "projects" / "CropFormer" / "configs" / "entityv2" / "entity_segmentation" / "mask2former_hornet_3x.yaml"
    weight_path = STREAM3D_ROOT / "third_party" / "seg_models" / "Mask2Former_hornet_3x_576d0b.pth"
    ok = bool(demo_script.exists() and config_path.exists() and weight_path.exists())
    return {
        "provider_role": "mask",
        "provider_name": "cropformer",
        "provider_mode": "live_cropformer_subprocess_assets_available",
        "available": ok,
        "schema_adapter_pass": ok,
        "runtime_sec": time.time() - t0,
        "peak_gpu_memory_mb": "",
        "demo_script": _rel(demo_script),
        "config_path": _rel(config_path),
        "weight_path": _rel(weight_path),
        "weight_sha256": _sha256_file(weight_path),
        "failure_reason": "" if ok else "CropFormer demo/config/weights missing",
    }


def _cropformer_live_root(ctx: PipelineContext) -> Path:
    return ctx.output_root / "baselines" / "cropformer_live" / "scannet_processed"


def _cropformer_live_mask_dir(ctx: PipelineContext, scene_id: str) -> Path:
    return _cropformer_live_root(ctx) / scene_id / "output_Cropformer" / "mask"


def _run_cropformer_live_scene(ctx: PipelineContext, scene_id: str, frame_ids: list[int]) -> dict[str, Any]:
    t0 = time.time()
    baselines = ctx.config.get("baselines", {})
    confidence_threshold = float(baselines.get("cropformer_confidence_threshold", baselines.get("cropformer_live_confidence_threshold", 0.5)))
    cropformer_cuda_visible_devices = str(
        baselines.get("cropformer_cuda_visible_devices", os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    ).strip()
    cropformer_model_device = str(baselines.get("cropformer_model_device", "cuda:0")).strip() or "cuda:0"
    scannet_root = _as_repo_path(ctx.config.get("paths", {}).get("scannet_processed_root")) or (STREAM3D_ROOT / "data" / "scannet" / "processed")
    live_root = _cropformer_live_root(ctx)
    shadow_scene_dir = live_root / scene_id
    shadow_color_dir = shadow_scene_dir / "color"
    mask_dir = _cropformer_live_mask_dir(ctx, scene_id)
    log_path = ctx.output_root / "baselines" / "cropformer_live" / "logs" / f"{scene_id}_cropformer.log"
    input_records_path = ctx.output_root / "baselines" / "cropformer_live" / f"{scene_id}_input_frame_records.json"
    demo_script = STREAM3D_ROOT / "third_party" / "detectron2" / "projects" / "CropFormer" / "demo_cropformer" / "Cropformer.py"
    config_path = STREAM3D_ROOT / "third_party" / "detectron2" / "projects" / "CropFormer" / "configs" / "entityv2" / "entity_segmentation" / "mask2former_hornet_3x.yaml"
    weight_path = STREAM3D_ROOT / "third_party" / "seg_models" / "Mask2Former_hornet_3x_576d0b.pth"
    command = [
        sys.executable,
        "third_party/detectron2/projects/CropFormer/demo_cropformer/Cropformer.py",
        "--config-file",
        "third_party/detectron2/projects/CropFormer/configs/entityv2/entity_segmentation/mask2former_hornet_3x.yaml",
        "--root",
        str(live_root),
        "--image_path_pattern",
        "color/*.jpg",
        "--dataset",
        "scannet",
        "--seq_name_list",
        scene_id,
        "--confidence-threshold",
        str(float(confidence_threshold)),
        "--opts",
        "MODEL.WEIGHTS",
        "third_party/seg_models/Mask2Former_hornet_3x_576d0b.pth",
        "MODEL.DEVICE",
        cropformer_model_device,
    ]
    try:
        for required in (demo_script, config_path, weight_path):
            if not required.exists():
                raise FileNotFoundError(str(required))
        if shadow_scene_dir.exists():
            shutil.rmtree(shadow_scene_dir)
        shadow_color_dir.mkdir(parents=True, exist_ok=True)
        input_rows: list[dict[str, Any]] = []
        for frame_id in [int(frame_id) for frame_id in frame_ids]:
            src = scannet_root / scene_id / "color" / f"{frame_id}.jpg"
            if not src.exists():
                raise FileNotFoundError(f"missing CropFormer RGB input: {src}")
            dst = shadow_color_dir / f"{frame_id}.jpg"
            native_rgb = _native_rgb_for_segmentor(ctx, scene_id, int(frame_id))
            cv2.imwrite(str(dst), cv2.cvtColor(np.ascontiguousarray(native_rgb[:, :, :3]), cv2.COLOR_RGB2BGR))
            input_rows.append(
                {
                    "schema_version": "stream4d_v105_cropformer_live_input_row_v1",
                    "scene_id": scene_id,
                    "frame_id": int(frame_id),
                    "source_rgb": _rel(src),
                    "shadow_rgb": _rel(dst),
                    "shadow_rgb_policy": "native_scannet_rgb_no_pipeline_resize",
                    "shadow_rgb_height": int(native_rgb.shape[0]),
                    "shadow_rgb_width": int(native_rgb.shape[1]),
                    "source_rgb_sha256": _sha256_file(src),
                    "shadow_rgb_sha256": _sha256_file(dst),
                }
            )
        _write_records_json(input_records_path, input_rows, schema_version="stream4d_v105_cropformer_live_input_records_v1")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        if cropformer_cuda_visible_devices:
            env["CUDA_VISIBLE_DEVICES"] = cropformer_cuda_visible_devices
        with log_path.open("w", encoding="utf-8") as handle:
            handle.write(f"cropformer_cuda_visible_devices={env.get('CUDA_VISIBLE_DEVICES', '')}\n")
            handle.write(f"cropformer_model_device={cropformer_model_device}\n")
            proc = subprocess.run(command, cwd=STREAM3D_ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, text=True)
        expected = [int(frame_id) for frame_id in frame_ids]
        missing: list[int] = []
        empty: list[int] = []
        mask_id_counts: dict[int, int] = {}
        total_mask_ids = 0
        for frame_id in expected:
            mask_path = mask_dir / f"{frame_id}.png"
            image = _read_image(mask_path, cv2.IMREAD_UNCHANGED)
            if image is None:
                missing.append(int(frame_id))
                continue
            if image.ndim == 3:
                image = image[..., 0]
            ids = [int(value) for value in np.unique(image) if int(value) > 0]
            mask_id_counts[int(frame_id)] = int(len(ids))
            total_mask_ids += int(len(ids))
            if not ids:
                empty.append(int(frame_id))
        status = "completed" if proc.returncode == 0 and not missing else "failed"
        return {
            "schema_version": "stream4d_v105_cropformer_live_runtime_row_v1",
            "status": status,
            "scene_id": scene_id,
            "frame_ids": ";".join(str(int(frame_id)) for frame_id in frame_ids),
            "frame_count": len(frame_ids),
            "mask_frame_count": len(frame_ids) - len(missing),
            "nonzero_output_frame_count": len(frame_ids) - len(missing) - len(empty),
            "zero_output_frame_ids": empty,
            "missing_frame_ids": missing,
            "total_mask_ids": int(total_mask_ids),
            "mask_id_count_by_frame": mask_id_counts,
            "latency_sec": time.time() - t0,
            "peak_gpu_memory_mb": "",
            "provider_mode": "live_cropformer_subprocess_stride_input_frames",
            "input_policy": "native_scannet_rgb_no_pipeline_resize",
            "confidence_threshold": confidence_threshold,
            "cuda_visible_devices": cropformer_cuda_visible_devices,
            "model_device": cropformer_model_device,
            "command": " ".join(command),
            "log_path": _rel(log_path),
            "input_frame_records_json": _rel(input_records_path),
            "mask_dir": _rel(mask_dir),
            "weight_path": _rel(weight_path),
            "weight_sha256": _sha256_file(weight_path),
            "returncode": int(proc.returncode),
            "failure_reason": "" if status == "completed" else f"CropFormer returncode={proc.returncode}, missing_frame_ids={missing}",
        }
    except Exception as exc:
        return {
            "schema_version": "stream4d_v105_cropformer_live_runtime_row_v1",
            "status": "failed",
            "scene_id": scene_id,
            "frame_ids": ";".join(str(int(frame_id)) for frame_id in frame_ids),
            "frame_count": len(frame_ids),
            "mask_frame_count": len(list(mask_dir.glob("*.png"))) if mask_dir.exists() else 0,
            "nonzero_output_frame_count": 0,
            "zero_output_frame_ids": [],
            "missing_frame_ids": [],
            "total_mask_ids": 0,
            "latency_sec": time.time() - t0,
            "peak_gpu_memory_mb": "",
            "provider_mode": "live_cropformer_subprocess_stride_input_frames",
            "confidence_threshold": confidence_threshold,
            "cuda_visible_devices": cropformer_cuda_visible_devices,
            "model_device": cropformer_model_device,
            "command": " ".join(command),
            "log_path": _rel(log_path),
            "input_frame_records_json": _rel(input_records_path),
            "mask_dir": _rel(mask_dir),
            "weight_path": _rel(weight_path),
            "weight_sha256": _sha256_file(weight_path),
            "returncode": "",
            "failure_reason": _safe_error(exc),
        }


def _smoke_sam2(ctx: PipelineContext, role: str) -> dict[str, Any]:
    t0 = time.time()
    paths = ctx.config.get("paths", {})
    checkpoint = _as_repo_path(paths.get("sam2_checkpoint"))
    model_cfg = paths.get("sam2_model_cfg", "")
    import_ok = _module_available("sam2")
    checkpoint_ok = bool(checkpoint and checkpoint.exists())
    construct_requested = bool(ctx.config.get("provider_smoke", {}).get("construct_sam2_models", False))
    construct_ok = False
    construct_error = ""
    peak_mb = ""
    if import_ok and checkpoint_ok and construct_requested:
        try:
            import torch  # type: ignore

            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            if role == "tracker":
                from sam2.build_sam import build_sam2_video_predictor  # type: ignore

                model = build_sam2_video_predictor(model_cfg, str(checkpoint), device=device)
            else:
                from sam2.build_sam import build_sam2  # type: ignore

                model = build_sam2(model_cfg, str(checkpoint), device=device)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                peak_mb = float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)
            construct_ok = model is not None
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            construct_error = _safe_error(exc)
    available = bool(import_ok and checkpoint_ok and (construct_ok if construct_requested else True))
    return {
        "provider_role": role,
        "provider_name": "sam2",
        "provider_mode": "sam2_checkpoint_import" if not construct_requested else "sam2_checkpoint_construct",
        "available": available,
        "schema_adapter_pass": bool(import_ok and checkpoint_ok),
        "runtime_sec": time.time() - t0,
        "peak_gpu_memory_mb": peak_mb,
        "checkpoint": _rel(checkpoint) if checkpoint else "",
        "checkpoint_sha256": _sha256_file(checkpoint) if checkpoint else "",
        "model_cfg": model_cfg,
        "import_ok": import_ok,
        "checkpoint_ok": checkpoint_ok,
        "construct_requested": construct_requested,
        "construct_ok": construct_ok,
        "failure_reason": "" if available else construct_error or "sam2 import/checkpoint/model construction unavailable",
    }


def _smoke_simple_import(
    ctx: PipelineContext,
    role: str,
    provider_name: str,
    module_names: list[str],
    checkpoint_key: str | None = None,
    require_checkpoint: bool = False,
) -> dict[str, Any]:
    t0 = time.time()
    import_hits = {name: _module_available(name) for name in module_names}
    checkpoint = _as_repo_path(ctx.config.get("paths", {}).get(checkpoint_key)) if checkpoint_key else None
    checkpoint_ok = True if checkpoint_key is None and not require_checkpoint else bool(checkpoint and checkpoint.exists())
    available = any(import_hits.values()) and checkpoint_ok
    return {
        "provider_role": role,
        "provider_name": provider_name,
        "provider_mode": "import_and_checkpoint_presence",
        "available": available,
        "schema_adapter_pass": any(import_hits.values()),
        "runtime_sec": time.time() - t0,
        "peak_gpu_memory_mb": "",
        "import_hits": import_hits,
        "checkpoint": _rel(checkpoint) if checkpoint else "",
        "checkpoint_ok": checkpoint_ok,
        "failure_reason": "" if available else "module import or required checkpoint missing",
    }


def _smoke_fastsam(ctx: PipelineContext) -> dict[str, Any]:
    t0 = time.time()
    checkpoint = _as_repo_path(ctx.config.get("paths", {}).get("fastsam_checkpoint"))
    import_ok = _module_available("ultralytics")
    checkpoint_ok = bool(checkpoint and checkpoint.exists())
    construct_ok = False
    inference_ok = False
    output_mask_count = 0
    sample_shape: list[int] = []
    peak_mb: float | str = ""
    error = ""
    if import_ok and checkpoint_ok:
        try:
            import torch  # type: ignore
            from ultralytics import FastSAM  # type: ignore

            scannet_root = _as_repo_path(ctx.config.get("paths", {}).get("scannet_processed_root")) or (STREAM3D_ROOT / "data" / "scannet" / "processed")
            scene = str(ctx.config.get("run", {}).get("scenes", [""])[0])
            sample = scannet_root / scene / "color" / "0.jpg"
            model = FastSAM(str(checkpoint))
            construct_ok = model is not None
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()
            if sample.exists():
                result = model(
                    str(sample),
                    device=0 if torch.cuda.is_available() else "cpu",
                    imgsz=int(ctx.config.get("baselines", {}).get("fastsam_imgsz", 1024)),
                    conf=float(ctx.config.get("baselines", {}).get("fastsam_conf", 0.3)),
                    iou=float(ctx.config.get("baselines", {}).get("fastsam_iou", 0.5)),
                    retina_masks=True,
                    verbose=False,
                )[0]
                if result.masks is not None:
                    sample_shape = list(result.masks.data.shape)
                    output_mask_count = int(result.masks.data.shape[0])
                inference_ok = output_mask_count > 0
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                peak_mb = float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)
                torch.cuda.empty_cache()
            del model
        except Exception as exc:
            error = _safe_error(exc)
    available = bool(import_ok and checkpoint_ok and construct_ok and inference_ok)
    return {
        "provider_role": "mask",
        "provider_name": "fastsam",
        "provider_mode": "real_inference_smoke",
        "available": available,
        "schema_adapter_pass": bool(import_ok and checkpoint_ok and construct_ok),
        "runtime_sec": time.time() - t0,
        "peak_gpu_memory_mb": peak_mb,
        "import_hits": {"ultralytics": import_ok},
        "checkpoint": _rel(checkpoint) if checkpoint else "",
        "checkpoint_ok": checkpoint_ok,
        "checkpoint_sha256": _sha256_file(checkpoint) if checkpoint else "",
        "construct_ok": construct_ok,
        "inference_ok": inference_ok,
        "output_mask_count": output_mask_count,
        "sample_mask_shape": sample_shape,
        "failure_reason": "" if available else error or "FastSAM import/checkpoint/construction/inference unavailable",
    }


def _smoke_lingbot(ctx: PipelineContext) -> dict[str, Any]:
    t0 = time.time()
    sys.path.insert(0, str(STREAM3D_ROOT))
    import_ok = False
    synthetic_pass = False
    error = ""
    try:
        from geometry_provider.lingbot_map_provider import LingBotMapGeometryProvider  # type: ignore

        import tempfile

        import_ok = True
        with tempfile.TemporaryDirectory(prefix="stream4d_v105_lingbot_smoke_") as d:
            root = Path(d)
            (root / "points").mkdir(parents=True)
            np.save(root / "points" / "000000.npy", np.asarray([[0, 0, 0], [1, 0, 0], [9, 9, 9]], dtype=np.float32))
            provider = LingBotMapGeometryProvider(geometry_root=root, nn_radius=0.2)
            result = provider.project_frame_masks(
                dataset=object(),
                scene_points=np.asarray([[0.02, 0, 0], [1.01, 0, 0], [5, 5, 5]], dtype=np.float32),
                mask_image=np.ones((4, 4), dtype=np.int32),
                frame_id=0,
                depth_max_pre=0.0,
            )
            synthetic_pass = result.frame_point_ids == [0, 1] and result.mask_info == {1: {0, 1}}
    except Exception as exc:
        error = _safe_error(exc)
    return {
        "provider_role": "geometry",
        "provider_name": "lingbot_map",
        "provider_mode": "synthetic_projection_adapter",
        "available": bool(import_ok and synthetic_pass),
        "schema_adapter_pass": bool(import_ok and synthetic_pass),
        "runtime_sec": time.time() - t0,
        "peak_gpu_memory_mb": "",
        "import_ok": import_ok,
        "synthetic_projection_pass": synthetic_pass,
        "failure_reason": "" if import_ok and synthetic_pass else error or "LingBot adapter synthetic projection failed",
    }


def _smoke_alltracker(ctx: PipelineContext) -> dict[str, Any]:
    t0 = time.time()
    root = _as_repo_path(ctx.config.get("paths", {}).get("fourdpm_root")) or (REPO_ROOT / "third_party" / "4D_PM")
    code_path = root / "frontend" / "alltracker" / "wrapper.py"
    checkpoint = _as_repo_path(ctx.config.get("paths", {}).get("alltracker_checkpoint"))
    code_ok = code_path.exists()
    checkpoint_ok = bool(checkpoint and checkpoint.exists())
    construct_ok = False
    inference_ok = False
    peak_mb: float | str = ""
    output_shapes: dict[str, Any] = {}
    confidence_mean: float | str = ""
    visibility_mean: float | str = ""
    error = ""
    if code_ok and checkpoint_ok:
        old_sys_path = list(sys.path)
        try:
            import torch  # type: ignore

            sys.path.insert(0, str(root))
            from frontend.alltracker.wrapper import forward_alltracker, setup_model  # type: ignore

            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()
            model = setup_model(str(checkpoint), window_len=16, weights_only=False, device=device)
            construct_ok = model is not None
            if torch.cuda.is_available():
                scannet_root = _as_repo_path(ctx.config.get("paths", {}).get("scannet_processed_root")) or (STREAM3D_ROOT / "data" / "scannet" / "processed")
                scene = str(ctx.config.get("run", {}).get("scenes", [""])[0])
                frame_ids = _selected_frame_ids(scannet_root / scene, int(ctx.config.get("run", {}).get("frame_stride", 5)), 4)
                frames = []
                for frame_id in frame_ids:
                    bgr = _read_image(scannet_root / scene / "color" / f"{int(frame_id)}.jpg", cv2.IMREAD_COLOR)
                    if bgr is None:
                        continue
                    rgb = cv2.cvtColor(cv2.resize(bgr, (320, 240), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
                    frames.append(torch.from_numpy(rgb).permute(2, 0, 1).float().to(device) / 255.0)
                if len(frames) >= 2:
                    out = forward_alltracker(frames, model, inference_iters=1)
                    torch.cuda.synchronize()
                    output_shapes = {
                        "traj_maps_normalised": list(out["traj_maps_normalised"].shape),
                        "visibilty": list(out["visibilty"].shape),
                        "confidence": list(out["confidence"].shape),
                    }
                    confidence_mean = float(out["confidence"].mean())
                    visibility_mean = float(out["visibilty"].float().mean())
                    inference_ok = True
            if torch.cuda.is_available():
                peak_mb = float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)
                torch.cuda.empty_cache()
            del model
        except Exception as exc:
            error = _safe_error(exc)
        finally:
            sys.path[:] = old_sys_path
    available = bool(code_ok and checkpoint_ok and construct_ok and (inference_ok or peak_mb == ""))
    return {
        "provider_role": "correspondence",
        "provider_name": "alltracker",
        "provider_mode": "construction_forward_smoke",
        "available": available,
        "schema_adapter_pass": bool(code_ok),
        "runtime_sec": time.time() - t0,
        "peak_gpu_memory_mb": peak_mb,
        "code_path": _rel(code_path),
        "code_ok": code_ok,
        "checkpoint": _rel(checkpoint) if checkpoint else "",
        "checkpoint_ok": checkpoint_ok,
        "checkpoint_sha256": _sha256_file(checkpoint) if checkpoint else "",
        "construct_ok": construct_ok,
        "inference_ok": inference_ok,
        "output_shapes": output_shapes,
        "confidence_mean": confidence_mean,
        "visibility_mean": visibility_mean,
        "explicit_blocked_allowed": bool(code_ok and not checkpoint_ok),
        "failure_reason": "" if available else error or "AllTracker code/checkpoint/construction/forward unavailable",
    }


def _smoke_edgetam(ctx: PipelineContext) -> dict[str, Any]:
    t0 = time.time()
    root = REPO_ROOT / "third_party" / "EdgeTAM"
    checkpoint = _as_repo_path(ctx.config.get("paths", {}).get("edgetam_checkpoint"))
    model_cfg = root / "sam2" / "configs" / "edgetam.yaml"
    code_paths = [
        root / "sam2" / "build_sam.py",
        root / "sam2" / "sam2_video_predictor.py",
        root / "sam2" / "sam2_image_predictor.py",
    ]
    code_ok = root.exists() and all(path.exists() for path in code_paths)
    checkpoint_ok = bool(checkpoint and checkpoint.exists())
    cfg_ok = model_cfg.exists()
    return {
        "provider_role": "tracker",
        "provider_name": "edgetam",
        "provider_mode": "code_config_checkpoint_presence",
        "available": bool(code_ok and cfg_ok and checkpoint_ok),
        "schema_adapter_pass": bool(code_ok and cfg_ok),
        "runtime_sec": time.time() - t0,
        "peak_gpu_memory_mb": "",
        "code_path": _rel(root),
        "code_ok": code_ok,
        "model_cfg": _rel(model_cfg),
        "model_cfg_ok": cfg_ok,
        "checkpoint": _rel(checkpoint) if checkpoint else "",
        "checkpoint_ok": checkpoint_ok,
        "checkpoint_sha256": _sha256_file(checkpoint) if checkpoint else "",
        "failure_reason": "" if code_ok and cfg_ok and checkpoint_ok else "EdgeTAM code/config/checkpoint missing",
    }


def run_provider_smoke(ctx: PipelineContext) -> None:
    with StageTimer(ctx, "provider_smoke"):
        out_dir = ctx.output_root / "provider_smoke"
        rows: list[dict[str, Any]] = []
        rows.append(_smoke_cropformer(ctx))
        rows.append(_smoke_fastsam(ctx))
        rows.append(_smoke_sam2(ctx, "mask"))
        rows.append(_smoke_simple_import(ctx, "mask", "sam31", ["sam3", "sam31"], "sam3_image_checkpoint", require_checkpoint=True))
        rows.append(_smoke_sam2(ctx, "tracker"))
        rows.append(_smoke_simple_import(ctx, "tracker", "sam31_multiplex", ["sam3", "sam31"], "sam31_checkpoint", require_checkpoint=True))
        rows.append(_smoke_edgetam(ctx))
        rows.append(_smoke_lingbot(ctx))
        rows.append(_smoke_alltracker(ctx))
        for row in rows:
            row["schema_version"] = "stream4d_v105_provider_smoke_row_v1"
            row["run_id"] = ctx.config["run"]["name"]
        failure_rows = []
        for row in rows:
            if not row.get("available"):
                failure_rows.append(
                    {
                        "schema_version": "stream4d_v105_provider_failure_row_v1",
                        "provider_role": row.get("provider_role"),
                        "provider_name": row.get("provider_name"),
                        "failure_type": "PROVIDER_UNAVAILABLE",
                        "failure_reason": row.get("failure_reason", ""),
                        "explicit_blocked_allowed": bool(row.get("explicit_blocked_allowed", False)),
                    }
                )
        _write_records_json(out_dir / "provider_smoke_records.json", rows)
        _write_records_json(out_dir / "provider_failure_records.json", failure_rows)
        _write_records_json(out_dir / "provider_registry_records.json", ctx.provider_rows)

        mask_available = any(row.get("provider_role") == "mask" and row.get("available") for row in rows)
        tracker_available = any(row.get("provider_role") == "tracker" and row.get("available") for row in rows)
        geometry_ok_or_blocked = any(row.get("provider_role") == "geometry" and row.get("available") for row in rows) or any(
            row.get("provider_role") == "geometry" and row.get("explicit_blocked_allowed") for row in rows
        )
        alltracker_ok_or_blocked = any(row.get("provider_role") == "correspondence" and (row.get("available") or row.get("explicit_blocked_allowed")) for row in rows)
        phase2_pass = bool(mask_available and tracker_available and geometry_ok_or_blocked and alltracker_ok_or_blocked)
        summary = {
            "schema_version": "stream4d_v105_provider_smoke_summary_v1",
            "phase2_provider_smoke_pass": phase2_pass,
            "mask_provider_available": mask_available,
            "tracker_provider_available": tracker_available,
            "geometry_provider_available_or_blocked": geometry_ok_or_blocked,
            "alltracker_available_or_blocked": alltracker_ok_or_blocked,
            "provider_failure_count": len(failure_rows),
            "construct_sam2_models": bool(ctx.config.get("provider_smoke", {}).get("construct_sam2_models", False)),
        }
        _write_json(out_dir / "provider_smoke_summary.json", summary)
        if not phase2_pass:
            ctx.add_failure(
                stage_name="provider_smoke",
                failure_type="PROVIDER_UNAVAILABLE",
                severity="blocker",
                symptom=f"provider smoke gate failed: {summary}",
                suggested_repair="Follow plan fallback order: reduce SAM batch/objects if OOM, fallback SAM2, fallback EdgeTAM; configure missing checkpoints before baselines.",
            )
        ctx.gates["phase2_provider_smoke_pass"] = phase2_pass
        ctx.gates["phase2_provider_smoke_summary"] = summary
        ctx.write_summary()


def _scene_frame_ids_for_baseline(ctx: PipelineContext, scene_id: str, max_frames: int) -> list[int]:
    scannet_root = _as_repo_path(ctx.config.get("paths", {}).get("scannet_processed_root")) or (STREAM3D_ROOT / "data" / "scannet" / "processed")
    stride = int(ctx.config.get("baselines", {}).get("eval_stride", ctx.config.get("run", {}).get("frame_stride", 5)))
    return _selected_frame_ids(scannet_root / scene_id, stride, int(max_frames))


def _write_pipeline_support_root(
    *,
    ctx: PipelineContext,
    pipeline_root: Path,
    variant_id: str,
    mask_root: str | None,
    mask_dir_by_scene: dict[str, Path],
    frame_ids_by_scene: dict[str, list[int]],
    object_id_policy: str,
) -> dict[str, Any]:
    objectlet_rows: list[dict[str, Any]] = []
    ledger_rows: list[dict[str, Any]] = []
    object_count = 0
    support_pair_count = 0
    for scene_id, frame_ids in frame_ids_by_scene.items():
        mask_dir = mask_dir_by_scene[scene_id]
        for frame_id in frame_ids:
            mask_path = mask_dir / f"{int(frame_id)}.png"
            for mask_id in _label_image_ids(mask_path):
                if object_id_policy == "frame_mask_is_object":
                    objectlet_id = f"{variant_id}:{scene_id}:f{int(frame_id):06d}:m{int(mask_id):04d}"
                elif object_id_policy == "mask_id_is_track":
                    objectlet_id = f"{variant_id}:{scene_id}:track{int(mask_id):04d}"
                else:
                    raise ValueError(f"unknown object_id_policy={object_id_policy}")
                candidate_id = f"{objectlet_id}:candidate"
                objectlet_rows.append(
                    {
                        "scene": scene_id,
                        "variant": variant_id,
                        "objectlet_id": objectlet_id,
                        "candidate_id": candidate_id,
                    }
                )
                ledger_rows.append(
                    {
                        "scene": scene_id,
                        "candidate_id": candidate_id,
                        "reprojection_success": True,
                        "best_mask_observation_id": f"{scene_id}:{int(frame_id)}:{int(mask_id)}",
                        "mask_path": _rel(mask_path),
                    }
                )
                support_pair_count += 1
        object_count += len({row["objectlet_id"] for row in objectlet_rows if row["scene"] == scene_id})

    local_dir = pipeline_root / "local_objectlets"
    ledger_dir = pipeline_root / "reprojection_ledger"
    _write_records_json(local_dir / "objectlet_records.json", objectlet_rows, schema_version="stream4d_v105_objectlet_manifest_v1")
    _write_records_json(ledger_dir / "reprojection_ledger_records.json", ledger_rows, schema_version="stream4d_v105_reprojection_manifest_v1")
    local_summary = {
        "schema_version": "stream4d_v105_baseline_local_objectlet_summary_v1",
        "best_real_variant": variant_id,
        "variant_id": variant_id,
        "object_id_policy": object_id_policy,
        "record_format": "json_manifest",
        "objectlet_records": _rel(local_dir / "objectlet_records.json"),
        "reprojection_ledger_records": _rel(ledger_dir / "reprojection_ledger_records.json"),
        "objectlet_row_count": len(objectlet_rows),
        "ledger_row_count": len(ledger_rows),
        "support_pair_count": support_pair_count,
    }
    _write_json(local_dir / "local_objectlet_summary.json", local_summary)
    pipeline_summary = {
        "schema_version": "stream4d_v105_baseline_pipeline_summary_v1",
        "variant_id": variant_id,
        "mask_root": mask_root or "",
        "mask_frame_coverage": {},
        "object_id_policy": object_id_policy,
        "frame_ids_by_scene": frame_ids_by_scene,
        "uses_gt_for_prediction": False,
        "uses_future": False,
        "note": "Support ledger is generated by v105 runner for v65 2D MV-AP diagnostics; it is not a trained model output.",
    }
    _write_json(pipeline_root / "pipeline_summary.json", pipeline_summary)
    return {
        "pipeline_root": _rel(pipeline_root),
        "record_format": "json_manifest",
        "objectlet_row_count": len(objectlet_rows),
        "ledger_row_count": len(ledger_rows),
        "support_pair_count": support_pair_count,
        "object_count_upper_bound": object_count,
    }


def _run_v65_soma_eval(
    *,
    scene_id: str,
    pipeline_root: Path,
    output_root: Path,
    stride: int,
    max_frames: int,
) -> list[dict[str, Any]]:
    if str(STREAM3D_ROOT) not in sys.path:
        sys.path.insert(0, str(STREAM3D_ROOT))
    from tools.run_v65_scene_multiview_ap import run as run_v65  # type: ignore

    ns = argparse.Namespace(
        scene=scene_id,
        methods="soma",
        strides=str(int(stride)),
        pipeline_root=str(pipeline_root),
        stream3d_config="scannet",
        output_root=str(output_root),
        score_mode="constant",
        min_pred_pixels=1,
        min_gt_pixels=1,
        vertex_nn_radius=0.08,
        vertex_cache_root=str(ctx_cache_root(output_root)),
        use_cache=0,
        max_frames=int(max_frames),
    )
    payload = run_v65(ns)
    rows = payload.get("rows", [])
    return rows if isinstance(rows, list) else []


def ctx_cache_root(output_root: Path) -> Path:
    return output_root / "vertex_cache_unused_for_soma"


def _write_baseline_overlay_video(
    *,
    ctx: PipelineContext,
    scene_id: str,
    frame_ids: list[int],
    mask_dir: Path,
    video_path: Path,
) -> bool:
    scannet_root = _as_repo_path(ctx.config.get("paths", {}).get("scannet_processed_root")) or (STREAM3D_ROOT / "data" / "scannet" / "processed")
    scene_root = scannet_root / scene_id
    frames: list[np.ndarray] = []
    for frame_id in frame_ids:
        rgb = _read_image(scene_root / "color" / f"{int(frame_id)}.jpg", cv2.IMREAD_COLOR)
        mask = _read_image(mask_dir / f"{int(frame_id)}.png", cv2.IMREAD_UNCHANGED)
        if rgb is None:
            continue
        if mask is not None and mask.shape[:2] != rgb.shape[:2]:
            mask = cv2.resize(mask, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        frames.append(_overlay(rgb, mask, draw_instance_ids=True))
    if not frames:
        return False
    video_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(ctx.config.get("video_export", {}).get("fps", 8)),
        (frames[0].shape[1], frames[0].shape[0]),
    )
    if not writer.isOpened():
        return False
    for frame in frames:
        writer.write(frame)
    writer.release()
    return video_path.exists() and video_path.stat().st_size > 0


def _write_full_frame_visual_audit(
    *,
    video_rows: list[dict[str, Any]],
    audit_root: Path,
    expected_frame_count: int,
    frames_per_sheet: int = 8,
) -> dict[str, Any]:
    audit_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    frames_per_sheet = max(int(frames_per_sheet), 1)
    for row in video_rows:
        video_path = Path(str(row.get("video_path", "")))
        if not video_path.is_absolute():
            video_path = REPO_ROOT / video_path
        stem = video_path.stem
        frame_dir = audit_root / stem / "frames"
        group_sheet_dir = audit_root / stem / "sheet_groups"
        frame_dir.mkdir(parents=True, exist_ok=True)
        group_sheet_dir.mkdir(parents=True, exist_ok=True)
        frames: list[np.ndarray] = []
        cap = cv2.VideoCapture(str(video_path))
        decoded = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            cv2.imwrite(str(frame_dir / f"frame_{decoded:03d}.jpg"), frame)
            frames.append(frame)
            decoded += 1
        cap.release()
        sheet_path = ""
        sheet_group_rows: list[dict[str, Any]] = []
        if frames:
            cols = 8
            thumbs: list[np.ndarray] = []
            for idx, frame in enumerate(frames):
                thumb = cv2.resize(frame, (240, 180), interpolation=cv2.INTER_AREA)
                cv2.rectangle(thumb, (0, 0), (88, 22), (0, 0, 0), -1)
                cv2.putText(
                    thumb,
                    f"frame {idx:02d}",
                    (4, 16),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.46,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                thumbs.append(thumb)
            row_count = (len(thumbs) + cols - 1) // cols
            sheet = np.full((row_count * 180, cols * 240, 3), 255, dtype=np.uint8)
            for idx, thumb in enumerate(thumbs):
                y = (idx // cols) * 180
                x = (idx % cols) * 240
                sheet[y : y + 180, x : x + 240] = thumb
            sheet_file = audit_root / f"{stem}_all_{len(frames)}_frames_sheet.jpg"
            cv2.imwrite(str(sheet_file), sheet)
            sheet_path = _rel(sheet_file)
            for group_start in range(0, len(thumbs), frames_per_sheet):
                group_end_excl = min(group_start + frames_per_sheet, len(thumbs))
                group = thumbs[group_start:group_end_excl]
                group_sheet = np.full((180, frames_per_sheet * 240, 3), 255, dtype=np.uint8)
                for local_idx, thumb in enumerate(group):
                    x = local_idx * 240
                    group_sheet[0:180, x : x + 240] = thumb
                group_file = group_sheet_dir / f"frames_{group_start:03d}_{group_end_excl - 1:03d}_sheet.jpg"
                cv2.imwrite(str(group_file), group_sheet)
                sheet_group_rows.append(
                    {
                        "schema_version": "stream4d_v105_visual_sheet_group_row_v1",
                        "sheet_group_index": len(sheet_group_rows),
                        "frame_start_index": int(group_start),
                        "frame_end_index": int(group_end_excl - 1),
                        "frame_count": int(group_end_excl - group_start),
                        "sheet_path": _rel(group_file),
                    }
                )
        expected_sheet_group_count = (int(expected_frame_count) + frames_per_sheet - 1) // frames_per_sheet
        rows.append(
            {
                **row,
                "decoded_frame_count": int(decoded),
                "expected_frame_count": int(expected_frame_count),
                "all_expected_frames_decoded": decoded == int(expected_frame_count),
                "frame_dir": _rel(frame_dir),
                "sheet": sheet_path,
                "all_frames_sheet": sheet_path,
                "frames_per_sheet": int(frames_per_sheet),
                "sheet_group_dir": _rel(group_sheet_dir),
                "sheet_groups": sheet_group_rows,
                "sheet_group_count": len(sheet_group_rows),
                "expected_sheet_group_count": int(expected_sheet_group_count),
                "sheet_groups_cover_expected_frames": decoded == int(expected_frame_count)
                and len(sheet_group_rows) == int(expected_sheet_group_count),
            }
        )
    summary = {
        "schema_version": "stream4d_v105_full_frame_visual_audit_v1",
        "output_root": _rel(audit_root),
        "no_interval_sampling": True,
        "frames_per_sheet": int(frames_per_sheet),
        "video_count": len(rows),
        "all_videos_decode_expected_frames": all(row.get("all_expected_frames_decoded") for row in rows),
        "all_sheet_groups_cover_expected_frames": all(row.get("sheet_groups_cover_expected_frames") for row in rows),
        "rows": rows,
    }
    _write_json(audit_root / "full_frame_visual_audit.json", summary)
    return summary


def _preprocessed_rgb_for_sam2(ctx: PipelineContext, scene_id: str, frame_id: int) -> np.ndarray:
    cfg = ctx.config
    pp = cfg.get("preprocess", {})
    target_hw = (int(pp.get("target_height", 240)), int(pp.get("target_width", 320)))
    scannet_root = _as_repo_path(cfg.get("paths", {}).get("scannet_processed_root")) or (STREAM3D_ROOT / "data" / "scannet" / "processed")
    color_path = scannet_root / scene_id / "color" / f"{int(frame_id)}.jpg"
    bgr = _read_image(color_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"missing RGB frame for SAM2 baseline: {color_path}")
    h, w = bgr.shape[:2]
    pad = int(pp.get("crop_black_padding_px", 15))
    crop = (pad, pad, max(w - pad, pad + 1), max(h - pad, pad + 1))
    final_bgr = _resize(bgr[crop[1] : crop[3], crop[0] : crop[2]], target_hw, cv2.INTER_LINEAR)
    return cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)


def _native_rgb_for_segmentor(ctx: PipelineContext, scene_id: str, frame_id: int) -> np.ndarray:
    cfg = ctx.config
    scannet_root = _as_repo_path(cfg.get("paths", {}).get("scannet_processed_root")) or (STREAM3D_ROOT / "data" / "scannet" / "processed")
    color_path = scannet_root / scene_id / "color" / f"{int(frame_id)}.jpg"
    bgr = _read_image(color_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"missing native RGB frame for segmentor: {color_path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _preprocessed_label_for_sgq(ctx: PipelineContext, scene_id: str, frame_id: int, path: Path) -> np.ndarray | None:
    label = _read_image(path, cv2.IMREAD_UNCHANGED)
    if label is None:
        return None
    if label.ndim == 3:
        label = label[..., 0]
    cfg = ctx.config
    pp = cfg.get("preprocess", {})
    target_hw = (int(pp.get("target_height", 240)), int(pp.get("target_width", 320)))
    scannet_root = _as_repo_path(cfg.get("paths", {}).get("scannet_processed_root")) or (STREAM3D_ROOT / "data" / "scannet" / "processed")
    rgb = _read_image(scannet_root / scene_id / "color" / f"{int(frame_id)}.jpg", cv2.IMREAD_COLOR)
    if rgb is None:
        return None
    h, w = rgb.shape[:2]
    pad = int(pp.get("crop_black_padding_px", 15))
    crop = (pad, pad, max(w - pad, pad + 1), max(h - pad, pad + 1))
    return _resize(_crop_scaled(label, crop, (h, w))[0], target_hw, cv2.INTER_NEAREST)


def _preprocessed_depth_for_sgq(ctx: PipelineContext, scene_id: str, frame_id: int) -> np.ndarray | None:
    cfg = ctx.config
    pp = cfg.get("preprocess", {})
    target_hw = (int(pp.get("target_height", 240)), int(pp.get("target_width", 320)))
    scannet_root = _as_repo_path(cfg.get("paths", {}).get("scannet_processed_root")) or (STREAM3D_ROOT / "data" / "scannet" / "processed")
    depth = _read_image(scannet_root / scene_id / "depth" / f"{int(frame_id)}.png", cv2.IMREAD_UNCHANGED)
    rgb = _read_image(scannet_root / scene_id / "color" / f"{int(frame_id)}.jpg", cv2.IMREAD_COLOR)
    if depth is None or rgb is None:
        return None
    if depth.ndim == 3:
        depth = depth[..., 0]
    h, w = rgb.shape[:2]
    pad = int(pp.get("crop_black_padding_px", 15))
    crop = (pad, pad, max(w - pad, pad + 1), max(h - pad, pad + 1))
    return _resize(_crop_scaled(depth, crop, (h, w))[0], target_hw, cv2.INTER_NEAREST)


def _preprocessed_intrinsics_for_sgq(ctx: PipelineContext, scene_id: str, frame_id: int) -> np.ndarray | None:
    cfg = ctx.config
    pp = cfg.get("preprocess", {})
    target_hw = (int(pp.get("target_height", 240)), int(pp.get("target_width", 320)))
    scannet_root = _as_repo_path(cfg.get("paths", {}).get("scannet_processed_root")) or (STREAM3D_ROOT / "data" / "scannet" / "processed")
    rgb = _read_image(scannet_root / scene_id / "color" / f"{int(frame_id)}.jpg", cv2.IMREAD_COLOR)
    if rgb is None:
        return None
    h, w = rgb.shape[:2]
    pad = int(pp.get("crop_black_padding_px", 15))
    crop = (pad, pad, max(w - pad, pad + 1), max(h - pad, pad + 1))
    return _update_intrinsics(_load_intrinsics(scannet_root / scene_id / "intrinsic" / "intrinsic_color.txt"), crop, target_hw)


def _pose_for_sgq(ctx: PipelineContext, scene_id: str, frame_id: int) -> np.ndarray | None:
    cfg = ctx.config
    scannet_root = _as_repo_path(cfg.get("paths", {}).get("scannet_processed_root")) or (STREAM3D_ROOT / "data" / "scannet" / "processed")
    pose_path = scannet_root / scene_id / "pose" / f"{int(frame_id)}.txt"
    if not pose_path.exists():
        return None
    try:
        pose = np.loadtxt(pose_path).astype(np.float64)
    except Exception:
        return None
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        return None
    return pose


def _mask_world_points_from_depth(
    mask: np.ndarray,
    *,
    depth_mm: np.ndarray | None,
    intrinsics: np.ndarray | None,
    pose_c2w: np.ndarray | None,
    max_points: int,
) -> np.ndarray | None:
    if depth_mm is None or intrinsics is None or pose_c2w is None:
        return None
    mask_bool = np.asarray(mask).astype(bool)
    depth = np.asarray(depth_mm)
    if depth.ndim == 3:
        depth = depth[..., 0]
    if mask_bool.shape[:2] != depth.shape[:2]:
        mask_bool = cv2.resize(mask_bool.astype(np.uint8), (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    valid = mask_bool & (depth > 0)
    ys, xs = np.nonzero(valid)
    if xs.size == 0:
        return None
    max_points = max(int(max_points), 1)
    if xs.size > max_points:
        take = np.linspace(0, xs.size - 1, max_points, dtype=np.int64)
        xs = xs[take]
        ys = ys[take]
    z = depth[ys, xs].astype(np.float64) / 1000.0
    finite = np.isfinite(z) & (z > 0.0)
    if not np.any(finite):
        return None
    xs = xs[finite].astype(np.float64)
    ys = ys[finite].astype(np.float64)
    z = z[finite]
    k = np.asarray(intrinsics, dtype=np.float64)
    if k.shape[0] < 3 or k.shape[1] < 3:
        return None
    fx = float(k[0, 0])
    fy = float(k[1, 1])
    cx = float(k[0, 2])
    cy = float(k[1, 2])
    if abs(fx) < 1e-9 or abs(fy) < 1e-9:
        return None
    x_cam = (xs - cx) * z / fx
    y_cam = (ys - cy) * z / fy
    cam_points = np.stack([x_cam, y_cam, z, np.ones_like(z)], axis=1)
    world = (np.asarray(pose_c2w, dtype=np.float64) @ cam_points.T).T[:, :3]
    world = world[np.all(np.isfinite(world), axis=1)]
    if world.shape[0] == 0:
        return None
    return world.astype(np.float32)


def _depth_structure_stats(mask: np.ndarray, depth_mm: np.ndarray | None) -> dict[str, Any]:
    pixel_count = int(np.count_nonzero(mask))
    if depth_mm is None or pixel_count <= 0:
        return {
            "depth_structure_pixel_count": pixel_count,
            "depth_structure_valid_pixel_count": 0,
            "depth_structure_valid_fraction": 0.0,
            "depth_structure_iqr_mm": 0.0,
            "depth_structure_median_mm": "",
            "depth_structure_min_mm": "",
            "depth_structure_max_mm": "",
        }
    if depth_mm.shape[:2] != mask.shape[:2]:
        depth_mm = _resize(depth_mm, mask.shape[:2], cv2.INTER_NEAREST)
    values = depth_mm[(mask > 0) & (depth_mm > 0)].astype(np.float32)
    if values.size == 0:
        return {
            "depth_structure_pixel_count": pixel_count,
            "depth_structure_valid_pixel_count": 0,
            "depth_structure_valid_fraction": 0.0,
            "depth_structure_iqr_mm": 0.0,
            "depth_structure_median_mm": "",
            "depth_structure_min_mm": "",
            "depth_structure_max_mm": "",
        }
    q25, q50, q75 = np.percentile(values, [25, 50, 75])
    return {
        "depth_structure_pixel_count": pixel_count,
        "depth_structure_valid_pixel_count": int(values.size),
        "depth_structure_valid_fraction": float(values.size) / float(max(pixel_count, 1)),
        "depth_structure_iqr_mm": float(q75 - q25),
        "depth_structure_median_mm": float(q50),
        "depth_structure_min_mm": float(values.min()),
        "depth_structure_max_mm": float(values.max()),
    }


class Sam31BoxPromptImagePredictor:
    supports_mask_input = False
    supports_point_input = False

    def __init__(self, predictor: Any, frame_root: Path, output_prob_thresh: float = 0.5) -> None:
        self.predictor = predictor
        self.frame_root = frame_root
        self.output_prob_thresh = float(output_prob_thresh)
        self.frame_index = 0
        self.session_id: str | None = None
        self.image_hw: tuple[int, int] | None = None
        self.frame_root.mkdir(parents=True, exist_ok=True)

    def close(self) -> None:
        if self.session_id is None:
            return
        try:
            self.predictor.handle_request(
                {"type": "close_session", "session_id": self.session_id, "run_gc_collect": True}
            )
        except Exception:
            pass
        self.session_id = None

    def set_image(self, image: np.ndarray) -> None:
        self.close()
        if image.ndim != 3:
            raise ValueError(f"SAM3.1 wrapper expects HWC RGB image, got shape={image.shape}")
        h, w = image.shape[:2]
        self.image_hw = (int(h), int(w))
        frame_dir = self.frame_root / f"frame_{self.frame_index:06d}"
        self.frame_index += 1
        frame_dir.mkdir(parents=True, exist_ok=True)
        frame_path = frame_dir / "00000.jpg"
        rgb = np.ascontiguousarray(image[:, :, :3])
        cv2.imwrite(str(frame_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        response = self.predictor.handle_request({"type": "start_session", "resource_path": str(frame_dir)})
        self.session_id = str(response["session_id"])

    def predict(
        self,
        *,
        box: np.ndarray,
        multimask_output: bool = True,
        return_logits: bool = False,
        normalize_coords: bool = True,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.session_id is None or self.image_hw is None:
            raise RuntimeError("set_image must be called before SAM3.1 predict")
        if kwargs.get("mask_input") is not None:
            raise ValueError("SAM3.1 multiplex box wrapper does not support mask_input")
        if kwargs.get("point_coords") is not None or kwargs.get("point_labels") is not None:
            raise ValueError("SAM3.1 multiplex box wrapper does not support point prompts in this path")
        h, w = self.image_hw
        box_arr = np.asarray(box, dtype=np.float32).reshape(-1)
        if box_arr.size != 4:
            raise ValueError(f"expected one xyxy box with 4 values, got {box_arr}")
        x0, y0, x1, y1 = [float(v) for v in box_arr]
        if not normalize_coords:
            x0 *= w
            x1 *= w
            y0 *= h
            y1 *= h
        x0 = min(max(x0, 0.0), float(max(w - 1, 1)))
        x1 = min(max(x1, 0.0), float(max(w, 1)))
        y0 = min(max(y0, 0.0), float(max(h - 1, 1)))
        y1 = min(max(y1, 0.0), float(max(h, 1)))
        bw = max(x1 - x0, 1.0)
        bh = max(y1 - y0, 1.0)
        xywh = np.asarray([[x0 / w, y0 / h, bw / w, bh / h]], dtype=np.float32)
        labels = np.asarray([1], dtype=np.int32)
        response = self.predictor.handle_request(
            {
                "type": "add_prompt",
                "session_id": self.session_id,
                "frame_index": 0,
                "bounding_boxes": xywh,
                "bounding_box_labels": labels,
                "output_prob_thresh": self.output_prob_thresh,
            }
        )
        outputs = response["outputs"]
        masks = np.asarray(outputs.get("out_binary_masks", np.zeros((0, h, w), dtype=bool)))
        if masks.ndim == 2:
            masks = masks[None, :, :]
        scores = np.asarray(outputs.get("out_probs", np.ones((masks.shape[0],), dtype=np.float32))).reshape(-1)
        if scores.size == 0 and masks.shape[0] > 0:
            scores = np.ones((masks.shape[0],), dtype=np.float32)
        lowres = np.zeros((masks.shape[0], 256, 256), dtype=np.float32)
        return masks.astype(bool), scores.astype(np.float32), lowres

    def __del__(self) -> None:
        self.close()


class _V105NoopDetector:
    def detect(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        return []


def _build_sam31_multiplex_frontend(ctx: PipelineContext, *, max_num_objects: int, device: str) -> tuple[Any, Any]:
    root = REPO_ROOT / "third_party" / "sam3"
    checkpoint = _as_repo_path(ctx.config.get("paths", {}).get("sam31_checkpoint"))
    if checkpoint is None or not checkpoint.exists():
        raise FileNotFoundError(f"missing SAM3.1 checkpoint: {checkpoint}")
    old_sys_path = list(sys.path)
    try:
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        sys.path.insert(0, str(root))
        from sam3.model_builder import build_sam3_multiplex_video_predictor  # type: ignore
        from loger.pipeline.video_masklet_frontend import VideoMaskletFrontend  # type: ignore

        predictor = build_sam3_multiplex_video_predictor(
            checkpoint_path=str(checkpoint),
            use_fa3=False,
            use_rope_real=False,
            max_num_objects=max(int(max_num_objects), 1),
            multiplex_count=16,
            compile=False,
            warm_up=False,
            async_loading_frames=False,
        )
        if hasattr(predictor, "model"):
            model = predictor.model
            if hasattr(model, "postprocess_batch_size"):
                model.postprocess_batch_size = max(int(ctx.config.get("baselines", {}).get("full_sam31_postprocess_batch_size", 1)), 1)
            if hasattr(model, "batched_grounding_batch_size"):
                model.batched_grounding_batch_size = max(int(ctx.config.get("baselines", {}).get("full_sam31_batched_grounding_batch_size", 1)), 1)
            detector = getattr(model, "detector", None)
            if detector is not None and hasattr(detector, "offload_outputs_to_cpu_for_eval"):
                detector.offload_outputs_to_cpu_for_eval = bool(ctx.config.get("baselines", {}).get("full_sam31_offload_outputs_to_cpu", True))
        frontend = VideoMaskletFrontend(
            video_predictor=predictor,
            detector=_V105NoopDetector(),
            sam_backend="sam31_multiplex",
            device=device,
            thing_prompts=[],
            stuff_prompts=[],
            max_thing_objects=max(int(max_num_objects), 1),
            prompt_type="mask",
            sam31_offload_video_to_cpu=bool(ctx.config.get("baselines", {}).get("full_sam31_offload_video_to_cpu", True)),
            sam31_offload_state_to_cpu=bool(ctx.config.get("baselines", {}).get("full_sam31_offload_state_to_cpu", True)),
            sam31_enable_backward=bool(ctx.config.get("baselines", {}).get("full_sam31_enable_backward", False)),
        )
        return predictor, frontend
    finally:
        sys.path[:] = old_sys_path


def _write_sam31_frame_resource(ctx: PipelineContext, scene_id: str, frame_ids: list[int]) -> tuple[str, dict[int, int]]:
    frame_dir = tempfile.mkdtemp(prefix="stream4d_v105_sam31_frames_")
    local_index_by_frame: dict[int, int] = {}
    for local_idx, frame_id in enumerate(frame_ids):
        rgb = _preprocessed_rgb_for_sam2(ctx, scene_id, int(frame_id))
        frame_path = Path(frame_dir) / f"{int(local_idx):05d}.jpg"
        cv2.imwrite(str(frame_path), cv2.cvtColor(np.ascontiguousarray(rgb[:, :, :3]), cv2.COLOR_RGB2BGR))
        local_index_by_frame[int(frame_id)] = int(local_idx)
    return frame_dir, local_index_by_frame


def _build_sgq_image_refiner(ctx: PipelineContext, provider_name: str, device: str) -> Any:
    provider = provider_name.strip().lower()
    if provider == "sam2":
        from sam2.build_sam import build_sam2  # type: ignore
        from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore

        checkpoint = _as_repo_path(ctx.config.get("paths", {}).get("sam2_checkpoint"))
        model_cfg = ctx.config.get("paths", {}).get("sam2_model_cfg")
        return SAM2ImagePredictor(build_sam2(model_cfg, str(checkpoint), device=device))
    if provider == "edgetam":
        root = REPO_ROOT / "third_party" / "EdgeTAM"
        checkpoint = _as_repo_path(ctx.config.get("paths", {}).get("edgetam_checkpoint"))
        if checkpoint is None or not checkpoint.exists():
            raise FileNotFoundError(f"missing EdgeTAM checkpoint: {checkpoint}")
        old_cwd = Path.cwd()
        old_sys_path = list(sys.path)
        try:
            os.chdir(root)
            sys.path.insert(0, str(root))
            for name in list(sys.modules):
                if name == "sam2" or name.startswith("sam2."):
                    del sys.modules[name]
            try:
                from hydra.core.global_hydra import GlobalHydra  # type: ignore

                if GlobalHydra.instance().is_initialized():
                    GlobalHydra.instance().clear()
            except Exception:
                pass
            from sam2.build_sam import build_sam2  # type: ignore
            from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore

            return SAM2ImagePredictor(build_sam2("edgetam.yaml", str(checkpoint), device=device))
        finally:
            os.chdir(old_cwd)
            sys.path[:] = old_sys_path
    if provider in {"sam31", "sam3.1", "sam31_multiplex"}:
        root = REPO_ROOT / "third_party" / "sam3"
        checkpoint = _as_repo_path(ctx.config.get("paths", {}).get("sam31_checkpoint"))
        if checkpoint is None or not checkpoint.exists():
            raise FileNotFoundError(f"missing SAM3.1 checkpoint: {checkpoint}")
        old_sys_path = list(sys.path)
        try:
            sys.path.insert(0, str(root))
            from sam3 import build_sam3_predictor  # type: ignore

            predictor = build_sam3_predictor(
                checkpoint_path=str(checkpoint),
                version="sam3.1",
                compile=False,
                warm_up=False,
                max_num_objects=16,
                multiplex_count=16,
                use_fa3=False,
                use_rope_real=False,
                async_loading_frames=False,
            )
        finally:
            sys.path[:] = old_sys_path
        output_prob_thresh = float(ctx.config.get("local", {}).get("sgq_sam31_output_prob_thresh", 0.5))
        frame_root = ctx.output_root / "sgq_local" / "sam31_frame_sessions"
        return Sam31BoxPromptImagePredictor(predictor, frame_root=frame_root, output_prob_thresh=output_prob_thresh)
    raise ValueError(f"unknown SGQ image refiner provider: {provider_name}")


def _max_iou_with_label(mask: np.ndarray, label: np.ndarray) -> float:
    ids = [int(value) for value in np.unique(label) if int(value) > 0]
    if not ids or not np.any(mask):
        return 0.0
    best = 0.0
    for value in ids:
        other = label == value
        inter = np.count_nonzero(mask & other)
        union = np.count_nonzero(mask | other)
        if union:
            best = max(best, float(inter) / float(union))
    return best


def _best_iou_with_label(mask: np.ndarray, label: np.ndarray | None) -> tuple[float, int]:
    if label is None or not np.any(mask):
        return 0.0, 0
    ids = [int(value) for value in np.unique(label) if int(value) > 0]
    best = 0.0
    best_id = 0
    for value in ids:
        other = label == value
        inter = np.count_nonzero(mask & other)
        union = np.count_nonzero(mask | other)
        if union:
            score = float(inter) / float(union)
            if score > best:
                best = score
                best_id = value
    return best, best_id


def _best_overlap_with_label(mask: np.ndarray, label: np.ndarray | None) -> tuple[int, float, float]:
    if label is None or not np.any(mask):
        return 0, 0.0, 0.0
    ids = [int(value) for value in np.unique(label) if int(value) > 0]
    best_id = 0
    best_iou = 0.0
    best_coverage = 0.0
    mask_area = float(max(np.count_nonzero(mask), 1))
    for value in ids:
        other = label == value
        inter = float(np.count_nonzero(mask & other))
        union = float(np.count_nonzero(mask | other))
        iou = inter / union if union else 0.0
        coverage = inter / mask_area
        if (iou, coverage) > (best_iou, best_coverage):
            best_id = value
            best_iou = iou
            best_coverage = coverage
    return best_id, best_iou, best_coverage


def _mask_track_state(
    mask: np.ndarray,
    *,
    min_pixels: int,
    local_frame_idx: int,
    frame_id: int,
    world_points: np.ndarray | None = None,
) -> dict[str, Any] | None:
    mask_bool = np.asarray(mask).astype(bool)
    area = int(np.count_nonzero(mask_bool))
    if area < int(min_pixels):
        return None
    bbox = _bbox_xyxy(mask_bool)
    if bbox is None:
        return None
    state = {
        "mask": mask_bool.copy(),
        "bbox": [int(v) for v in bbox],
        "area": int(area),
        "center": (0.5 * (float(bbox[0]) + float(bbox[2])), 0.5 * (float(bbox[1]) + float(bbox[3]))),
        "last_local_frame_idx": int(local_frame_idx),
        "last_frame_id": int(frame_id),
    }
    if world_points is not None:
        points = np.asarray(world_points, dtype=np.float32)
        if points.ndim == 2 and points.shape[1] == 3 and points.shape[0] > 0:
            points = points[np.all(np.isfinite(points), axis=1)]
            if points.shape[0] > 0:
                state["world_points"] = points.copy()
                state["world_point_count"] = int(points.shape[0])
                state["world_centroid"] = np.mean(points, axis=0).astype(np.float32)
    return state


def _append_mask_track_state(
    track_registry: dict[int, dict[str, Any]],
    track_id: int,
    mask: np.ndarray,
    *,
    min_pixels: int,
    local_frame_idx: int,
    frame_id: int,
    max_history: int,
    world_points: np.ndarray | None = None,
) -> bool:
    state = _mask_track_state(
        mask,
        min_pixels=min_pixels,
        local_frame_idx=local_frame_idx,
        frame_id=frame_id,
        world_points=world_points,
    )
    if state is None:
        return False
    track_id = int(track_id)
    previous = track_registry.get(track_id, {})
    raw_history = previous.get("history", []) if isinstance(previous, dict) else []
    history = [dict(row) for row in raw_history if isinstance(row, dict)]
    history = [row for row in history if int(row.get("last_local_frame_idx", -1)) != int(local_frame_idx)]
    history.append(dict(state))
    history.sort(key=lambda row: (int(row.get("last_local_frame_idx", -1)), int(row.get("last_frame_id", -1))))
    if len(history) > max(int(max_history), 1):
        history = history[-max(int(max_history), 1) :]
    state_with_history = dict(state)
    state_with_history["history"] = history
    track_registry[track_id] = state_with_history
    return True


def _score_world_track_continuity(
    current_points: np.ndarray | None,
    state: dict[str, Any],
    *,
    near_threshold_m: float,
    centroid_threshold_m: float,
) -> dict[str, Any] | None:
    if current_points is None:
        return None
    prev_points = state.get("world_points")
    if prev_points is None:
        return None
    current = np.asarray(current_points, dtype=np.float32)
    previous = np.asarray(prev_points, dtype=np.float32)
    if current.ndim != 2 or previous.ndim != 2 or current.shape[1] != 3 or previous.shape[1] != 3:
        return None
    current = current[np.all(np.isfinite(current), axis=1)]
    previous = previous[np.all(np.isfinite(previous), axis=1)]
    if current.shape[0] == 0 or previous.shape[0] == 0:
        return None
    current_centroid = np.mean(current, axis=0)
    previous_centroid = np.mean(previous, axis=0)
    centroid_dist = float(np.linalg.norm(current_centroid - previous_centroid))
    # Sample sizes are capped by the caller, so a dense pairwise distance block is cheap and deterministic.
    distances = np.linalg.norm(current[:, None, :] - previous[None, :, :], axis=2)
    current_to_prev = np.min(distances, axis=1)
    prev_to_current = np.min(distances, axis=0)
    near_threshold = max(float(near_threshold_m), 1e-6)
    current_near = float(np.mean(current_to_prev <= near_threshold))
    previous_near = float(np.mean(prev_to_current <= near_threshold))
    near_fraction = 0.5 * (current_near + previous_near)
    centroid_threshold = max(float(centroid_threshold_m), near_threshold)
    centroid_score = max(0.0, 1.0 - centroid_dist / centroid_threshold)
    geometry_score = 0.70 * near_fraction + 0.30 * centroid_score
    return {
        "geometry_score": float(geometry_score),
        "world_near_fraction": float(near_fraction),
        "world_current_near_fraction": float(current_near),
        "world_previous_near_fraction": float(previous_near),
        "world_centroid_distance_m": float(centroid_dist),
        "world_point_count": int(current.shape[0]),
        "world_prev_point_count": int(previous.shape[0]),
    }


def _score_mask_track_continuity(
    mask: np.ndarray,
    state: dict[str, Any],
    *,
    min_pixels: int,
    world_points: np.ndarray | None = None,
    geometry_near_threshold_m: float = 0.08,
    geometry_centroid_threshold_m: float = 0.45,
    geometry_min_score: float = 0.25,
) -> dict[str, Any] | None:
    current = _mask_track_state(mask, min_pixels=min_pixels, local_frame_idx=0, frame_id=0, world_points=world_points)
    if current is None:
        return None
    prev_mask = np.asarray(state.get("mask")).astype(bool)
    if prev_mask.shape[:2] != current["mask"].shape[:2]:
        return None
    inter = float(np.count_nonzero(current["mask"] & prev_mask))
    union = float(np.count_nonzero(current["mask"] | prev_mask))
    mask_iou = inter / union if union > 0.0 else 0.0
    bbox_iou = _bbox_iou_xyxy(current["bbox"], state["bbox"])
    current_area = float(max(int(current["area"]), 1))
    coverage = inter / current_area if current_area > 0.0 else 0.0
    prev_area = float(max(int(state.get("area", 0)), 1))
    area_score = min(current_area, prev_area) / max(current_area, prev_area)
    h, w = current["mask"].shape[:2]
    diag = float(max(np.hypot(float(h), float(w)), 1.0))
    cx, cy = current["center"]
    pcx, pcy = state.get("center", (cx, cy))
    center_score = max(0.0, 1.0 - float(np.hypot(float(cx) - float(pcx), float(cy) - float(pcy))) / (0.45 * diag))
    score = 0.45 * mask_iou + 0.30 * bbox_iou + 0.15 * area_score + 0.10 * center_score
    geometry_match = _score_world_track_continuity(
        current.get("world_points"),
        state,
        near_threshold_m=geometry_near_threshold_m,
        centroid_threshold_m=geometry_centroid_threshold_m,
    )
    geometry_score = float(geometry_match.get("geometry_score", 0.0)) if geometry_match else 0.0
    geometry_gate = bool(geometry_match and geometry_score >= float(geometry_min_score))
    if geometry_match:
        score = max(score, 0.55 * geometry_score + 0.45 * score)
    continuity_gate = (
        mask_iou >= 0.03
        or coverage >= 0.20
        or (bbox_iou >= 0.20 and center_score >= 0.55 and area_score >= 0.35)
        or geometry_gate
    )
    if not continuity_gate:
        return None
    result = {
        "score": float(score),
        "mask_iou": float(mask_iou),
        "coverage": float(coverage),
        "bbox_iou": float(bbox_iou),
        "area_score": float(area_score),
        "center_score": float(center_score),
        "current_area": int(current["area"]),
        "geometry_gate": bool(geometry_gate),
    }
    if geometry_match:
        result.update(geometry_match)
    return result


def _best_track_registry_match(
    mask: np.ndarray,
    track_registry: dict[int, dict[str, Any]],
    *,
    local_frame_idx: int,
    min_pixels: int,
    lookback: int,
    min_score: float,
    world_points: np.ndarray | None = None,
    geometry_near_threshold_m: float = 0.08,
    geometry_centroid_threshold_m: float = 0.45,
    geometry_min_score: float = 0.25,
) -> tuple[int, dict[str, Any] | None]:
    best_track_id = 0
    best_match: dict[str, Any] | None = None
    for track_id, state in track_registry.items():
        raw_history = state.get("history", []) if isinstance(state, dict) else []
        states: list[dict[str, Any]] = [row for row in raw_history if isinstance(row, dict)]
        if not states and isinstance(state, dict):
            states = [state]
        for history_state in states:
            age = int(local_frame_idx) - int(history_state.get("last_local_frame_idx", local_frame_idx))
            if age <= 0 or age > max(int(lookback), 1):
                continue
            match = _score_mask_track_continuity(
                mask,
                history_state,
                min_pixels=min_pixels,
                world_points=world_points,
                geometry_near_threshold_m=geometry_near_threshold_m,
                geometry_centroid_threshold_m=geometry_centroid_threshold_m,
                geometry_min_score=geometry_min_score,
            )
            if match is None:
                continue
            match["age"] = int(age)
            match["matched_local_frame_idx"] = int(history_state.get("last_local_frame_idx", local_frame_idx))
            match["matched_frame_id"] = int(history_state.get("last_frame_id", -1))
            score = float(match["score"]) / (1.0 + 0.08 * float(max(age, 0)))
            match["score"] = float(score)
            if score < float(min_score):
                continue
            if best_match is None or score > float(best_match["score"]):
                best_track_id = int(track_id)
                best_match = match
    return best_track_id, best_match


def _best_proposal_support_match(
    mask: np.ndarray,
    supports: list[dict[str, Any]],
    *,
    prompt_area: int,
    min_pixels: int,
    min_score: float,
    min_iou: float,
    min_mask_coverage: float,
    min_support_coverage: float,
    max_area_expand: float,
    track_state: dict[str, Any] | None = None,
    require_geometry: bool = False,
    geometry_near_threshold_m: float = 0.08,
    geometry_centroid_threshold_m: float = 0.45,
    geometry_min_score: float = 0.25,
    write_full_support: bool = False,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    mask_bool = np.asarray(mask).astype(bool)
    mask_area = int(np.count_nonzero(mask_bool))
    if mask_area < int(min_pixels):
        return None, {"matched": False, "reason": "mask_below_min_pixels", "mask_area": int(mask_area)}
    prompt_area = int(max(prompt_area, 1))
    best_mask: np.ndarray | None = None
    best: dict[str, Any] | None = None
    for support in supports:
        support_mask = np.asarray(support.get("mask")).astype(bool)
        if support_mask.shape[:2] != mask_bool.shape[:2]:
            continue
        support_area = int(support.get("pixel_area", 0) or np.count_nonzero(support_mask))
        if support_area < int(min_pixels):
            continue
        area_expand = float(support_area) / float(max(prompt_area, 1))
        if area_expand > float(max_area_expand):
            continue
        overlap = mask_bool & support_mask
        inter = int(np.count_nonzero(overlap))
        if inter < int(min_pixels):
            continue
        union = float(max(mask_area + support_area - inter, 1))
        iou = float(inter) / union
        mask_coverage = float(inter) / float(max(mask_area, 1))
        support_coverage = float(inter) / float(max(support_area, 1))
        area_score = min(float(prompt_area), float(support_area)) / max(float(prompt_area), float(support_area), 1.0)
        if (
            iou < float(min_iou)
            and mask_coverage < float(min_mask_coverage)
            and support_coverage < float(min_support_coverage)
        ):
            continue
        score = 0.45 * iou + 0.25 * mask_coverage + 0.20 * support_coverage + 0.10 * area_score
        if score < float(min_score):
            continue
        geometry_match: dict[str, Any] | None = None
        geometry_gate = False
        if track_state is not None:
            support_world_points = support.get("world_points")
            states: list[dict[str, Any]] = []
            raw_history = track_state.get("history", []) if isinstance(track_state, dict) else []
            states.extend(row for row in raw_history if isinstance(row, dict))
            if isinstance(track_state, dict):
                states.append(track_state)
            for state in states:
                candidate_geometry = _score_world_track_continuity(
                    support_world_points,
                    state,
                    near_threshold_m=geometry_near_threshold_m,
                    centroid_threshold_m=geometry_centroid_threshold_m,
                )
                if candidate_geometry is None:
                    continue
                if geometry_match is None or float(candidate_geometry["geometry_score"]) > float(
                    geometry_match["geometry_score"]
                ):
                    geometry_match = candidate_geometry
            if geometry_match is not None:
                geometry_score = float(geometry_match.get("geometry_score", 0.0))
                geometry_gate = geometry_score >= float(geometry_min_score)
                score = max(score, 0.60 * geometry_score + 0.40 * score)
        if bool(require_geometry) and not geometry_gate:
            continue
        row = {
            "matched": True,
            "score": float(score),
            "proposal_mask_id": int(support.get("proposal_mask_id", -1)),
            "write_full_support": bool(write_full_support),
            "support_pixel_area": int(support_area),
            "mask_pixel_area": int(mask_area),
            "prompt_pixel_area": int(prompt_area),
            "intersection_pixel_area": int(inter),
            "iou": float(iou),
            "mask_coverage": float(mask_coverage),
            "support_coverage": float(support_coverage),
            "area_score": float(area_score),
            "area_expand": float(area_expand),
            "geometry_gate": bool(geometry_gate),
        }
        if geometry_match is not None:
            row.update(geometry_match)
        if best is None or float(row["score"]) > float(best["score"]):
            best = row
            best_mask = support_mask.copy() if bool(write_full_support) else overlap
    if best is None or best_mask is None:
        return None, {
            "matched": False,
            "reason": "no_support_match",
            "mask_area": int(mask_area),
            "prompt_pixel_area": int(prompt_area),
            "require_geometry": bool(require_geometry),
        }
    return best_mask, best


def _bbox_xyxy(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.nonzero(mask)
    if xs.size == 0 or ys.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _bbox_iou_xyxy(a: list[int] | tuple[int, int, int, int], b: list[int] | tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = [int(v) for v in a]
    bx0, by0, bx1, by1 = [int(v) for v in b]
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    iw = max(ix1 - ix0, 0)
    ih = max(iy1 - iy0, 0)
    inter = float(iw * ih)
    if inter <= 0.0:
        return 0.0
    area_a = float(max(ax1 - ax0, 0) * max(ay1 - ay0, 0))
    area_b = float(max(bx1 - bx0, 0) * max(by1 - by0, 0))
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def _bbox_gap_norm_xyxy(
    a: list[int] | tuple[int, int, int, int],
    b: list[int] | tuple[int, int, int, int],
    shape_hw: tuple[int, int],
) -> float:
    ax0, ay0, ax1, ay1 = [int(v) for v in a]
    bx0, by0, bx1, by1 = [int(v) for v in b]
    dx = max(bx0 - ax1, ax0 - bx1, 0)
    dy = max(by0 - ay1, ay0 - by1, 0)
    diag = float(max(np.hypot(float(shape_hw[0]), float(shape_hw[1])), 1.0))
    return float(np.hypot(float(dx), float(dy))) / diag


def _candidate_objectness_gate(
    mask: np.ndarray,
    *,
    min_pixels: int,
    min_area_ratio: float,
    max_area_ratio: float,
    broad_area_ratio: float,
    max_bbox_area_ratio: float,
    max_span_ratio: float,
    max_edge_pixel_ratio: float,
    max_edge_touch_sides: int,
    enabled: bool,
) -> dict[str, Any]:
    mask_bool = np.asarray(mask).astype(bool)
    h, w = mask_bool.shape[:2]
    frame_area = float(max(h * w, 1))
    pixel_area = int(np.count_nonzero(mask_bool))
    area_ratio = float(pixel_area) / frame_area
    reject_reasons: list[str] = []
    bbox = _bbox_xyxy(mask_bool)
    if pixel_area < int(min_pixels):
        reject_reasons.append("below_min_pixels")
    if area_ratio < float(min_area_ratio):
        reject_reasons.append("below_min_area_ratio")
    if area_ratio > float(max_area_ratio):
        reject_reasons.append("above_max_area_ratio")
    if bbox is None:
        reject_reasons.append("empty_bbox")
        bbox_values = [0, 0, 0, 0]
        bbox_area_ratio = 0.0
        fill_ratio = 0.0
        span_w = 0.0
        span_h = 0.0
        edge_touch_sides = 0
        edge_pixel_ratio = 0.0
    else:
        x0, y0, x1, y1 = [int(v) for v in bbox]
        bbox_values = [x0, y0, x1, y1]
        bbox_area = float(max((x1 - x0) * (y1 - y0), 1))
        bbox_area_ratio = bbox_area / frame_area
        fill_ratio = float(pixel_area) / bbox_area
        span_w = float(max(x1 - x0, 0)) / float(max(w, 1))
        span_h = float(max(y1 - y0, 0)) / float(max(h, 1))
        ys, xs = np.nonzero(mask_bool)
        edge_pixels = int(
            np.count_nonzero(xs == 0)
            + np.count_nonzero(xs == w - 1)
            + np.count_nonzero(ys == 0)
            + np.count_nonzero(ys == h - 1)
        )
        edge_pixel_ratio = float(edge_pixels) / float(max(pixel_area, 1))
        edge_touch_sides = int(x0 <= 0) + int(y0 <= 0) + int(x1 >= w) + int(y1 >= h)
    if enabled and area_ratio >= float(broad_area_ratio):
        if bbox_area_ratio >= float(max_bbox_area_ratio):
            reject_reasons.append("broad_bbox_area")
        if max(span_w, span_h) >= float(max_span_ratio) and edge_touch_sides >= int(max_edge_touch_sides):
            reject_reasons.append("broad_span_edge")
        if edge_pixel_ratio >= float(max_edge_pixel_ratio) and edge_touch_sides >= int(max_edge_touch_sides):
            reject_reasons.append("broad_edge_pixels")
    return {
        "accepted": not reject_reasons,
        "reject_reasons": reject_reasons,
        "pixel_area": int(pixel_area),
        "area_ratio": float(area_ratio),
        "bbox": bbox_values,
        "bbox_area_ratio": float(bbox_area_ratio),
        "fill_ratio": float(fill_ratio),
        "span_w": float(span_w),
        "span_h": float(span_h),
        "edge_touch_sides": int(edge_touch_sides),
        "edge_pixel_ratio": float(edge_pixel_ratio),
    }


def _candidate_uncovered_stats(mask: np.ndarray, label: np.ndarray | None) -> tuple[int, float, np.ndarray]:
    mask_bool = np.asarray(mask).astype(bool)
    pixel_area = int(np.count_nonzero(mask_bool))
    if pixel_area <= 0:
        return 0, 0.0, np.zeros(mask_bool.shape[:2], dtype=bool)
    if label is None:
        return pixel_area, 1.0, mask_bool.copy()
    label_np = np.asarray(label)
    if label_np.shape[:2] != mask_bool.shape[:2]:
        return pixel_area, 1.0, mask_bool.copy()
    uncovered = mask_bool & (label_np == 0)
    uncovered_pixels = int(np.count_nonzero(uncovered))
    return uncovered_pixels, float(uncovered_pixels) / float(max(pixel_area, 1)), uncovered


def _stabilize_label_ids_temporally(
    *,
    mask_dir: Path,
    frame_ids: list[int],
    min_pixels: int,
    lookback: int,
    min_match_score: float,
    ctx: PipelineContext | None = None,
    scene_id: str = "",
    geometry_enabled: bool = False,
    geometry_max_points: int = 512,
    geometry_near_threshold_m: float = 0.08,
    geometry_centroid_threshold_m: float = 0.45,
    geometry_min_score: float = 0.25,
) -> dict[str, Any]:
    """Rewrite per-frame instance ids so colors denote temporal object identity."""
    track_states: dict[int, dict[str, Any]] = {}
    next_track_id = 1
    frame_rows: list[dict[str, Any]] = []
    remap_rows: list[dict[str, Any]] = []
    reused_assignment_count = 0
    new_assignment_count = 0
    geometry_assignment_count = 0
    same_frame_append_assignment_count = 0
    geometry_only_reject_count = 0
    skipped_tiny_new_seed_count = 0
    skipped_small_region_count = 0
    history_assignment_count = 0
    long_history_assignment_count = 0
    history_max_len = max(len(frame_ids), max(int(lookback), 1))
    h = 0
    w = 0

    for local_idx, frame_id in enumerate(frame_ids):
        path = mask_dir / f"{int(frame_id)}.png"
        raw = _read_image(path, cv2.IMREAD_UNCHANGED)
        if raw is None:
            frame_rows.append(
                {
                    "frame_id": int(frame_id),
                    "source_label_count": 0,
                    "stable_label_count": 0,
                    "reused_assignment_count": 0,
                    "new_assignment_count": 0,
                    "missing_mask": True,
                }
            )
            continue
        if raw.ndim == 3:
            raw = raw[..., 0]
        if h == 0 or w == 0:
            h, w = raw.shape[:2]
        depth_mm = None
        intrinsics = None
        pose_c2w = None
        if geometry_enabled and ctx is not None and scene_id:
            depth_mm = _preprocessed_depth_for_sgq(ctx, scene_id, int(frame_id))
            intrinsics = _preprocessed_intrinsics_for_sgq(ctx, scene_id, int(frame_id))
            pose_c2w = _pose_for_sgq(ctx, scene_id, int(frame_id))
        stable = np.zeros(raw.shape[:2], dtype=np.uint16)
        objects: list[dict[str, Any]] = []
        for source_id in [int(value) for value in np.unique(raw) if int(value) > 0]:
            mask = raw == source_id
            area = int(np.count_nonzero(mask))
            if area < min_pixels:
                skipped_small_region_count += 1
                continue
            bbox = _bbox_xyxy(mask)
            if bbox is None:
                skipped_small_region_count += 1
                continue
            cx = 0.5 * (float(bbox[0]) + float(bbox[2]))
            cy = 0.5 * (float(bbox[1]) + float(bbox[3]))
            objects.append(
                {
                    "source_id": int(source_id),
                    "mask": mask,
                    "area": area,
                    "bbox": bbox,
                    "center": (cx, cy),
                    "world_points": _mask_world_points_from_depth(
                        mask,
                        depth_mm=depth_mm,
                        intrinsics=intrinsics,
                        pose_c2w=pose_c2w,
                        max_points=geometry_max_points,
                    )
                    if geometry_enabled
                    else None,
                }
            )
        objects.sort(key=lambda row: (-int(row["area"]), int(row["source_id"])))
        matched_track_ids: set[int] = set()
        frame_reused = 0
        frame_new = 0
        diag = float(max(np.hypot(*raw.shape[:2]), 1.0))
        new_seed_min_pixels = max(int(min_pixels), 80)
        for obj in objects:
            best_track_id = 0
            best_score = 0.0
            best_terms = (0.0, 0.0, 0.0, 0.0)
            best_shape_continuity_gate = False
            best_geometry_terms: dict[str, Any] | None = None
            best_same_frame_append = False
            best_match_age = 0
            best_match_local_idx = -1
            best_match_frame_id = -1
            mask = obj["mask"]
            area = float(max(int(obj["area"]), 1))
            bbox = obj["bbox"]
            cx, cy = obj["center"]
            for track_id, state in track_states.items():
                if int(track_id) in matched_track_ids:
                    continue
                raw_history = state.get("history", []) if isinstance(state, dict) else []
                candidate_states = [row for row in raw_history if isinstance(row, dict)]
                if not candidate_states and isinstance(state, dict):
                    candidate_states = [state]
                for candidate_state in sorted(
                    candidate_states,
                    key=lambda row: int(row.get("last_local_idx", -1)),
                    reverse=True,
                ):
                    age = int(local_idx) - int(candidate_state.get("last_local_idx", local_idx))
                    if age <= 0 or age > history_max_len:
                        continue
                    prev_mask = candidate_state["mask"]
                    inter = float(np.count_nonzero(mask & prev_mask))
                    union = float(np.count_nonzero(mask | prev_mask))
                    mask_iou = inter / union if union > 0.0 else 0.0
                    box_iou = _bbox_iou_xyxy(bbox, candidate_state["bbox"])
                    prev_area = float(max(int(candidate_state["area"]), 1))
                    area_score = min(area, prev_area) / max(area, prev_area)
                    pcx, pcy = candidate_state["center"]
                    center_dist = float(np.hypot(float(cx) - float(pcx), float(cy) - float(pcy)))
                    center_score = max(0.0, 1.0 - center_dist / (0.35 * diag))
                    score = 0.55 * mask_iou + 0.25 * box_iou + 0.10 * area_score + 0.10 * center_score
                    overlap_gate = (
                        mask_iou >= 0.03
                        or box_iou >= 0.08
                        or (center_score >= 0.72 and area_score >= 0.45)
                    )
                    geometry_terms = None
                    geometry_gate = False
                    if geometry_enabled:
                        geometry_terms = _score_world_track_continuity(
                            obj.get("world_points"),
                            candidate_state,
                            near_threshold_m=geometry_near_threshold_m,
                            centroid_threshold_m=geometry_centroid_threshold_m,
                        )
                        if geometry_terms is not None:
                            geometry_score = float(geometry_terms.get("geometry_score", 0.0))
                            geometry_gate = geometry_score >= float(geometry_min_score)
                            score = max(score, 0.55 * geometry_score + 0.45 * score)
                    shape_continuity_gate = bool(box_iou >= 0.20 and center_score >= 0.60 and area_score >= 0.45)
                    geometry_only_gate = bool(geometry_gate and not overlap_gate and not shape_continuity_gate)
                    geometry_only_sane = (
                        not geometry_only_gate
                        or (
                            area_score >= 0.15
                            and (
                                mask_iou >= 0.01
                                or box_iou >= 0.05
                                or center_score >= 0.60
                            )
                        )
                    )
                    if geometry_only_gate and not geometry_only_sane:
                        geometry_only_reject_count += 1
                        continue
                    if (overlap_gate or geometry_gate) and score > best_score:
                        best_track_id = int(track_id)
                        best_score = float(score)
                        best_terms = (float(mask_iou), float(box_iou), float(area_score), float(center_score))
                        best_shape_continuity_gate = shape_continuity_gate
                        best_geometry_terms = geometry_terms
                        best_match_age = int(age)
                        best_match_local_idx = int(candidate_state.get("last_local_idx", -1))
                        best_match_frame_id = int(candidate_state.get("last_frame_id", -1))
            if best_track_id <= 0 and matched_track_ids:
                for track_id in sorted(matched_track_ids):
                    state = track_states.get(int(track_id))
                    if not isinstance(state, dict) or int(state.get("last_local_idx", -1)) != int(local_idx):
                        continue
                    append_match = _score_mask_track_continuity(
                        mask,
                        state,
                        min_pixels=min_pixels,
                        world_points=obj.get("world_points"),
                        geometry_near_threshold_m=geometry_near_threshold_m,
                        geometry_centroid_threshold_m=geometry_centroid_threshold_m,
                        geometry_min_score=geometry_min_score,
                    )
                    if append_match is None:
                        continue
                    append_geometry_score = float(append_match.get("geometry_score", 0.0))
                    append_gap = _bbox_gap_norm_xyxy(bbox, state["bbox"], raw.shape[:2])
                    append_gate = (
                        append_geometry_score >= max(float(geometry_min_score), 0.70)
                        and float(append_match.get("area_score", 0.0)) >= 0.15
                        and (
                            float(append_match.get("mask_iou", 0.0)) >= 0.01
                            or float(append_match.get("bbox_iou", 0.0)) >= 0.05
                            or float(append_match.get("center_score", 0.0)) >= 0.60
                            or append_gap <= 0.08
                        )
                    )
                    if not append_gate:
                        continue
                    append_score = max(float(append_match.get("score", 0.0)), append_geometry_score)
                    if append_score > best_score:
                        best_track_id = int(track_id)
                        best_score = float(append_score)
                        best_terms = (
                            float(append_match.get("mask_iou", 0.0)),
                            float(append_match.get("bbox_iou", 0.0)),
                            float(append_match.get("area_score", 0.0)),
                            float(append_match.get("center_score", 0.0)),
                        )
                        best_shape_continuity_gate = bool(
                            float(append_match.get("bbox_iou", 0.0)) >= 0.20
                            and float(append_match.get("center_score", 0.0)) >= 0.60
                            and float(append_match.get("area_score", 0.0)) >= 0.45
                        )
                        best_geometry_terms = dict(append_match)
                        best_geometry_terms["same_frame_bbox_gap_norm"] = float(append_gap)
                        best_same_frame_append = True
            if best_track_id > 0 and (
                best_score >= float(min_match_score)
                or best_shape_continuity_gate
                or best_same_frame_append
                or (
                    best_geometry_terms is not None
                    and float(best_geometry_terms.get("geometry_score", 0.0)) >= float(geometry_min_score)
                )
            ):
                assigned_track_id = best_track_id
                frame_reused += 1
                reused_assignment_count += 1
                matched_track_ids.add(assigned_track_id)
                if best_same_frame_append:
                    assignment_kind = "reused_same_frame_identity_append"
                    same_frame_append_assignment_count += 1
                elif best_geometry_terms is not None and float(best_geometry_terms.get("geometry_score", 0.0)) >= float(geometry_min_score):
                    assignment_kind = "reused_geometry_continuity"
                    geometry_assignment_count += 1
                else:
                    assignment_kind = "reused" if best_score >= float(min_match_score) else "reused_shape_continuity"
                if best_match_age > 1:
                    history_assignment_count += 1
                if best_match_age > max(int(lookback), 1):
                    long_history_assignment_count += 1
            else:
                if int(obj["area"]) < int(new_seed_min_pixels):
                    skipped_tiny_new_seed_count += 1
                    continue
                assigned_track_id = int(next_track_id)
                next_track_id += 1
                frame_new += 1
                new_assignment_count += 1
                matched_track_ids.add(assigned_track_id)
                assignment_kind = "new"
            region = mask & (stable == 0)
            if int(np.count_nonzero(region)) >= min_pixels:
                stable[region] = int(assigned_track_id)
            state_mask = mask.copy()
            previous_state = track_states.get(int(assigned_track_id))
            if isinstance(previous_state, dict) and int(previous_state.get("last_local_idx", -1)) == int(local_idx):
                previous_mask = np.asarray(previous_state.get("mask")).astype(bool)
                if previous_mask.shape[:2] == state_mask.shape[:2]:
                    state_mask = previous_mask | state_mask
            state_bbox = _bbox_xyxy(state_mask) or bbox
            state_ys, state_xs = np.nonzero(state_mask)
            if state_xs.size > 0:
                state_center = (float(np.mean(state_xs)), float(np.mean(state_ys)))
            else:
                state_center = (float(cx), float(cy))
            state_world_points = (
                _mask_world_points_from_depth(
                    state_mask,
                    depth_mm=depth_mm,
                    intrinsics=intrinsics,
                    pose_c2w=pose_c2w,
                    max_points=geometry_max_points,
                )
                if geometry_enabled
                else None
            )
            state_record = {
                "mask": state_mask.copy(),
                "bbox": [int(v) for v in state_bbox],
                "area": int(np.count_nonzero(state_mask)),
                "center": (float(state_center[0]), float(state_center[1])),
                "last_local_idx": int(local_idx),
                "last_frame_id": int(frame_id),
            }
            if state_world_points is not None:
                state_record["world_points"] = state_world_points
                state_record["world_point_count"] = int(np.asarray(state_world_points).shape[0])
            raw_history = previous_state.get("history", []) if isinstance(previous_state, dict) else []
            history = [dict(row) for row in raw_history if isinstance(row, dict)]
            history = [row for row in history if int(row.get("last_local_idx", -1)) != int(local_idx)]
            history.append(dict(state_record))
            history.sort(key=lambda row: (int(row.get("last_local_idx", -1)), int(row.get("last_frame_id", -1))))
            if len(history) > history_max_len:
                history = history[-history_max_len:]
            state_record["history"] = history
            track_states[int(assigned_track_id)] = state_record
            remap_rows.append(
                {
                    "frame_id": int(frame_id),
                    "source_id_before_stabilization": int(obj["source_id"]),
                    "stable_track_id": int(assigned_track_id),
                    "best_candidate_track_id": int(best_track_id),
                    "assignment_kind": assignment_kind,
                    "match_score": best_score,
                    "match_mask_iou": best_terms[0],
                    "match_bbox_iou": best_terms[1],
                    "match_area_score": best_terms[2],
                    "match_center_score": best_terms[3],
                    "match_shape_continuity_gate": bool(best_shape_continuity_gate),
                    "match_geometry_score": float(best_geometry_terms.get("geometry_score", 0.0)) if best_geometry_terms else 0.0,
                    "match_world_near_fraction": float(best_geometry_terms.get("world_near_fraction", 0.0)) if best_geometry_terms else 0.0,
                    "match_world_centroid_distance_m": float(best_geometry_terms.get("world_centroid_distance_m", 0.0)) if best_geometry_terms else "",
                    "match_same_frame_bbox_gap_norm": float(best_geometry_terms["same_frame_bbox_gap_norm"]) if best_geometry_terms and "same_frame_bbox_gap_norm" in best_geometry_terms else "",
                    "match_age": int(best_match_age),
                    "match_local_frame_idx": int(best_match_local_idx),
                    "match_frame_id": int(best_match_frame_id),
                    "area": int(obj["area"]),
                }
            )
        cv2.imwrite(str(path), stable)
        frame_rows.append(
            {
                "frame_id": int(frame_id),
                "source_label_count": len(objects),
                "stable_label_count": len([int(value) for value in np.unique(stable) if int(value) > 0]),
                "reused_assignment_count": frame_reused,
                "new_assignment_count": frame_new,
                "missing_mask": False,
            }
        )
    return {
        "schema_version": "stream4d_v105_temporal_id_stabilization_summary_v2",
        "enabled": True,
        "policy": "greedy_adjacent_mask_bbox_area_center_shape_plus_optional_3d_geometry_association",
        "lookback": int(max(lookback, 1)),
        "min_match_score": float(min_match_score),
        "geometry_enabled": bool(geometry_enabled and ctx is not None and bool(scene_id)),
        "geometry_max_points": int(max(geometry_max_points, 1)),
        "geometry_near_threshold_m": float(geometry_near_threshold_m),
        "geometry_centroid_threshold_m": float(geometry_centroid_threshold_m),
        "geometry_min_score": float(geometry_min_score),
        "geometry_assignment_count": int(geometry_assignment_count),
        "same_frame_append_assignment_count": int(same_frame_append_assignment_count),
        "geometry_only_reject_count": int(geometry_only_reject_count),
        "skipped_tiny_new_seed_count": int(skipped_tiny_new_seed_count),
        "history_assignment_count": int(history_assignment_count),
        "long_history_assignment_count": int(long_history_assignment_count),
        "history_max_len": int(history_max_len),
        "frame_count": len(frame_ids),
        "track_count": int(max(next_track_id - 1, 0)),
        "reused_assignment_count": int(reused_assignment_count),
        "new_assignment_count": int(new_assignment_count),
        "skipped_small_region_count": int(skipped_small_region_count),
        "frame_rows": frame_rows,
        "remap_rows": remap_rows,
    }


def _run_fastsam_scene(ctx: PipelineContext, scene_id: str, frame_ids: list[int], mask_dir: Path) -> dict[str, Any]:
    t0 = time.time()
    peak_gpu_mb: float | str = ""
    mask_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = _as_repo_path(ctx.config.get("paths", {}).get("fastsam_checkpoint"))
    baselines = ctx.config.get("baselines", {})
    fastsam_gate_enabled = bool(baselines.get("fastsam_candidate_objectness_gate_enabled", False))
    fastsam_min_pixels = int(baselines.get("fastsam_min_mask_pixels", baselines.get("baseline_matrix_min_mask_pixels", 16)))
    fastsam_min_mask_len_div = float(baselines.get("fastsam_min_mask_len_div", 0.0))
    fastsam_min_area_ratio = float(baselines.get("fastsam_min_area_ratio", baselines.get("baseline_matrix_min_area_ratio", 0.0)))
    fastsam_max_area_ratio = float(baselines.get("fastsam_max_area_ratio", baselines.get("baseline_matrix_max_area_ratio", 1.0)))
    fastsam_broad_area_ratio = float(
        baselines.get("fastsam_objectness_broad_area_ratio", baselines.get("baseline_matrix_objectness_broad_area_ratio", 1.0))
    )
    fastsam_max_bbox_area_ratio = float(
        baselines.get("fastsam_objectness_max_bbox_area_ratio", baselines.get("baseline_matrix_objectness_max_bbox_area_ratio", 1.0))
    )
    fastsam_max_span_ratio = float(
        baselines.get("fastsam_objectness_max_span_ratio", baselines.get("baseline_matrix_objectness_max_span_ratio", 1.0))
    )
    fastsam_max_edge_pixel_ratio = float(
        baselines.get("fastsam_objectness_max_edge_pixel_ratio", baselines.get("baseline_matrix_objectness_max_edge_pixel_ratio", 1.0))
    )
    fastsam_max_edge_touch_sides = int(
        baselines.get("fastsam_objectness_max_edge_touch_sides", baselines.get("baseline_matrix_objectness_max_edge_touch_sides", 4))
    )
    preserve_small_overlaps = bool(baselines.get("fastsam_preserve_small_overlaps", True))
    if checkpoint is None or not checkpoint.exists():
        return {
            "status": "failed",
            "scene_id": scene_id,
            "frame_ids": ";".join(str(int(frame_id)) for frame_id in frame_ids),
            "frame_count": len(frame_ids),
            "mask_frame_count": 0,
            "total_masks": 0,
            "latency_sec": time.time() - t0,
            "peak_gpu_memory_mb": peak_gpu_mb,
            "checkpoint": _rel(checkpoint) if checkpoint else "",
            "checkpoint_sha256": "",
            "failure_reason": "FastSAM checkpoint missing",
        }
    try:
        import torch  # type: ignore
        from ultralytics import FastSAM  # type: ignore

        model = FastSAM(str(checkpoint))
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        scannet_root = _as_repo_path(ctx.config.get("paths", {}).get("scannet_processed_root")) or (STREAM3D_ROOT / "data" / "scannet" / "processed")
        native_input_records: list[dict[str, Any]] = []
        min_pixel_records: list[dict[str, Any]] = []
        total_masks = 0
        max_mask_id = 0
        raw_mask_count = 0
        accepted_raw_mask_count = 0
        rejected_raw_mask_count = 0
        reject_reason_counts: dict[str, int] = {}
        for frame_id in frame_ids:
            source_image_path = scannet_root / scene_id / "color" / f"{int(frame_id)}.jpg"
            if not source_image_path.exists():
                raise FileNotFoundError(f"missing RGB frame for FastSAM baseline: {source_image_path}")
            rgb = _native_rgb_for_segmentor(ctx, scene_id, int(frame_id))
            frame_min_pixels = int(fastsam_min_pixels)
            if fastsam_min_mask_len_div > 0.0:
                proportional_min_pixels = int(round(float(rgb.shape[0] * rgb.shape[1]) / (fastsam_min_mask_len_div**2)))
                frame_min_pixels = max(frame_min_pixels, proportional_min_pixels)
            min_pixel_records.append(
                {
                    "frame_id": int(frame_id),
                    "min_mask_pixels": int(frame_min_pixels),
                    "min_mask_len_div": float(fastsam_min_mask_len_div),
                }
            )
            image_path = source_image_path
            native_input_records.append(
                {
                    "frame_id": int(frame_id),
                    "native_rgb_path": _rel(image_path),
                    "native_rgb_sha256": _sha256_file(image_path),
                    "native_rgb_height": int(rgb.shape[0]),
                    "native_rgb_width": int(rgb.shape[1]),
                }
            )
            result = model(
                str(image_path),
                device=0 if torch.cuda.is_available() else "cpu",
                imgsz=int(baselines.get("fastsam_imgsz", 1024)),
                conf=float(baselines.get("fastsam_conf", 0.3)),
                iou=float(baselines.get("fastsam_iou", 0.5)),
                retina_masks=True,
                verbose=False,
            )[0]
            label = np.zeros(rgb.shape[:2], dtype=np.uint16)
            if result.masks is not None:
                masks = result.masks.data.detach().cpu().numpy() > 0.5
                if masks.ndim == 2:
                    masks = masks[None, :, :]
                raw_mask_count += int(masks.shape[0])
                if result.boxes is not None and getattr(result.boxes, "conf", None) is not None:
                    scores = result.boxes.conf.detach().cpu().numpy().reshape(-1)
                else:
                    scores = np.ones((masks.shape[0],), dtype=np.float32)
                candidates: list[dict[str, Any]] = []
                for idx in range(int(masks.shape[0])):
                    mask = masks[int(idx)]
                    if mask.shape[:2] != rgb.shape[:2]:
                        mask = cv2.resize(mask.astype(np.uint8), (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
                    gate = _candidate_objectness_gate(
                        mask,
                        min_pixels=frame_min_pixels,
                        min_area_ratio=fastsam_min_area_ratio,
                        max_area_ratio=fastsam_max_area_ratio,
                        broad_area_ratio=fastsam_broad_area_ratio,
                        max_bbox_area_ratio=fastsam_max_bbox_area_ratio,
                        max_span_ratio=fastsam_max_span_ratio,
                        max_edge_pixel_ratio=fastsam_max_edge_pixel_ratio,
                        max_edge_touch_sides=fastsam_max_edge_touch_sides,
                        enabled=fastsam_gate_enabled,
                    )
                    if not bool(gate.get("accepted", False)):
                        rejected_raw_mask_count += 1
                        for reason in gate.get("reject_reasons", []):
                            reject_reason_counts[str(reason)] = reject_reason_counts.get(str(reason), 0) + 1
                        continue
                    accepted_raw_mask_count += 1
                    candidates.append(
                        {
                            "source_idx": int(idx),
                            "score": float(scores[int(idx)]) if int(idx) < len(scores) else 1.0,
                            "area": int(gate.get("pixel_area", int(np.count_nonzero(mask)))),
                            "mask": mask,
                        }
                    )
                if preserve_small_overlaps:
                    write_order = sorted(candidates, key=lambda row: (int(row["area"]), float(row["score"])), reverse=True)
                else:
                    write_order = sorted(candidates, key=lambda row: float(row["score"]), reverse=True)
                next_id = 1
                for cand in write_order:
                    mask = np.asarray(cand["mask"]).astype(bool)
                    region = mask if preserve_small_overlaps else (mask & (label == 0))
                    if int(np.count_nonzero(region)) < frame_min_pixels:
                        continue
                    label[region] = int(next_id)
                    next_id += 1
                visible_ids = [int(v) for v in np.unique(label) if int(v) > 0]
                if visible_ids:
                    compact = np.zeros_like(label)
                    for compact_id, visible_id in enumerate(visible_ids, start=1):
                        compact[label == int(visible_id)] = int(compact_id)
                    label = compact
                total_masks += int(len(visible_ids))
                max_mask_id = max(max_mask_id, int(len(visible_ids)))
            cv2.imwrite(str(mask_dir / f"{int(frame_id)}.png"), label)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            peak_gpu_mb = float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)
            torch.cuda.empty_cache()
        del model
        return {
            "status": "completed",
            "scene_id": scene_id,
            "frame_ids": ";".join(str(int(frame_id)) for frame_id in frame_ids),
            "frame_count": len(frame_ids),
            "mask_frame_count": len(list(mask_dir.glob("*.png"))),
            "total_masks": int(total_masks),
            "max_mask_id": int(max_mask_id),
            "raw_mask_count": int(raw_mask_count),
            "accepted_raw_mask_count": int(accepted_raw_mask_count),
            "rejected_raw_mask_count": int(rejected_raw_mask_count),
            "reject_reason_counts": reject_reason_counts,
            "fastsam_generation_objectness_gate_enabled": bool(fastsam_gate_enabled),
            "fastsam_min_mask_pixels_base": int(fastsam_min_pixels),
            "fastsam_min_mask_len_div": float(fastsam_min_mask_len_div),
            "fastsam_min_pixel_records": min_pixel_records,
            "fastsam_preserve_small_overlaps": bool(preserve_small_overlaps),
            "fastsam_write_policy": "large_to_small_area_then_score_with_small_overlap_preservation"
            if preserve_small_overlaps
            else "score_desc_first_claim",
            "latency_sec": time.time() - t0,
            "peak_gpu_memory_mb": peak_gpu_mb,
            "checkpoint": _rel(checkpoint),
            "checkpoint_sha256": _sha256_file(checkpoint),
            "input_policy": "native_scannet_rgb_no_pipeline_resize",
            "native_input_records": native_input_records,
            "failure_reason": "",
        }
    except Exception as exc:
        return {
            "status": "failed",
            "scene_id": scene_id,
            "frame_ids": ";".join(str(int(frame_id)) for frame_id in frame_ids),
            "frame_count": len(frame_ids),
            "mask_frame_count": len(list(mask_dir.glob("*.png"))),
            "total_masks": 0,
            "max_mask_id": 0,
            "latency_sec": time.time() - t0,
            "peak_gpu_memory_mb": peak_gpu_mb,
            "checkpoint": _rel(checkpoint),
            "checkpoint_sha256": _sha256_file(checkpoint),
            "failure_reason": _safe_error(exc),
        }


def _proposal_label_path_for_baseline(
    ctx: PipelineContext,
    scene_id: str,
    frame_id: int,
    proposal_source: str,
) -> Path:
    source = str(proposal_source).strip().lower()
    source_key = _promptable_segmentor_source_key(source)
    scannet_root = _as_repo_path(ctx.config.get("paths", {}).get("scannet_processed_root")) or (STREAM3D_ROOT / "data" / "scannet" / "processed")
    if source == "cropformer":
        return _cropformer_live_mask_dir(ctx, scene_id) / f"{int(frame_id)}.png"
    if source == "fastsam":
        return ctx.output_root / "baselines" / "masks" / "B1_fastsam_only" / scene_id / "mask" / f"{int(frame_id)}.png"
    if source == "sam2_b2":
        return ctx.output_root / "baselines" / "masks" / "B2_4dpm_sam2_gap_tracking" / scene_id / "mask" / f"{int(frame_id)}.png"
    if source_key is not None:
        return ctx.output_root / "baselines" / "masks" / f"_proposal_{source_key}" / scene_id / "mask" / f"{int(frame_id)}.png"
    raise ValueError(f"unsupported baseline proposal_source={proposal_source}")


def _promptable_segmentor_source_key(proposal_source: str) -> str | None:
    source = str(proposal_source).strip().lower()
    if source in {"sam2", "sam2_amg", "sam2_auto", "sam2_automatic"}:
        return "sam2"
    if source in {"edgetam", "edgetam_amg", "edgetam_auto", "edgetam_automatic"}:
        return "edgetam"
    if source in {"sam31", "sam3.1", "sam31_multiplex", "sam3"}:
        return "sam31"
    return None


def _build_promptable_amg(ctx: PipelineContext, source_key: str, device: str) -> Any:
    baselines = ctx.config.get("baselines", {})
    points_per_side = int(baselines.get("promptable_segmentor_amg_points_per_side", 64))
    points_per_batch = int(baselines.get("promptable_segmentor_amg_points_per_batch", 128))
    pred_iou_thresh = float(baselines.get("promptable_segmentor_amg_pred_iou_thresh", 0.7))
    stability_score_thresh = float(baselines.get("promptable_segmentor_amg_stability_score_thresh", 0.92))
    stability_score_offset = float(baselines.get("promptable_segmentor_amg_stability_score_offset", 0.7))
    min_mask_region_area = int(baselines.get("promptable_segmentor_amg_min_mask_region_area", 25))
    crop_n_layers = int(baselines.get("promptable_segmentor_amg_crop_n_layers", 1))
    box_nms_thresh = float(baselines.get("promptable_segmentor_amg_box_nms_thresh", 0.7))
    crop_n_points_downscale_factor = int(baselines.get("promptable_segmentor_amg_crop_n_points_downscale_factor", 2))
    use_m2m = bool(baselines.get("promptable_segmentor_amg_use_m2m", True))
    if source_key == "sam2":
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator  # type: ignore
        from sam2.build_sam import build_sam2  # type: ignore

        checkpoint = _as_repo_path(ctx.config.get("paths", {}).get("sam2_checkpoint"))
        model_cfg = ctx.config.get("paths", {}).get("sam2_model_cfg")
        if checkpoint is None or not checkpoint.exists():
            raise FileNotFoundError(f"missing SAM2 checkpoint: {checkpoint}")
        model = build_sam2(str(model_cfg), str(checkpoint), device=device, apply_postprocessing=False)
        return SAM2AutomaticMaskGenerator(
            model,
            points_per_side=points_per_side,
            points_per_batch=points_per_batch,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            stability_score_offset=stability_score_offset,
            crop_n_layers=crop_n_layers,
            box_nms_thresh=box_nms_thresh,
            crop_n_points_downscale_factor=crop_n_points_downscale_factor,
            min_mask_region_area=min_mask_region_area,
            use_m2m=use_m2m,
            output_mode="binary_mask",
        )
    if source_key == "edgetam":
        root = REPO_ROOT / "third_party" / "EdgeTAM"
        checkpoint = _as_repo_path(ctx.config.get("paths", {}).get("edgetam_checkpoint"))
        if checkpoint is None or not checkpoint.exists():
            raise FileNotFoundError(f"missing EdgeTAM checkpoint: {checkpoint}")
        old_cwd = Path.cwd()
        old_sys_path = list(sys.path)
        try:
            os.chdir(root)
            sys.path.insert(0, str(root))
            for name in list(sys.modules):
                if name == "sam2" or name.startswith("sam2."):
                    del sys.modules[name]
            try:
                from hydra.core.global_hydra import GlobalHydra  # type: ignore

                if GlobalHydra.instance().is_initialized():
                    GlobalHydra.instance().clear()
            except Exception:
                pass
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator  # type: ignore
            from sam2.build_sam import build_sam2  # type: ignore

            model = build_sam2("edgetam.yaml", str(checkpoint), device=device, apply_postprocessing=False)
            return SAM2AutomaticMaskGenerator(
                model,
                points_per_side=points_per_side,
                points_per_batch=points_per_batch,
                pred_iou_thresh=pred_iou_thresh,
                stability_score_thresh=stability_score_thresh,
                stability_score_offset=stability_score_offset,
                crop_n_layers=crop_n_layers,
                box_nms_thresh=box_nms_thresh,
                crop_n_points_downscale_factor=crop_n_points_downscale_factor,
                min_mask_region_area=min_mask_region_area,
                use_m2m=use_m2m,
                output_mode="binary_mask",
            )
        finally:
            os.chdir(old_cwd)
            sys.path[:] = old_sys_path
    raise ValueError(f"unsupported AMG source_key={source_key}")


def _sam31_segmentor_4dpm_point_prompts(
    ctx: PipelineContext,
    scene_id: str,
    frame_id: int,
) -> tuple[list[list[float]], dict[str, Any]]:
    baselines = ctx.config.get("baselines", {})
    num_pts = max(int(baselines.get("promptable_segmentor_sam31_num_pts", baselines.get("sam2_gap_num_pts", 300))), 1)
    deterministic_grid = bool(baselines.get("sam2_gap_deterministic_point_grid", False))
    if deterministic_grid:
        side = int(math.ceil(math.sqrt(float(num_pts))))
        coords = np.linspace(-0.95, 0.95, side, dtype=np.float32)
        yy, xx = np.meshgrid(coords, coords, indexing="ij")
        points_yx = np.stack([yy.reshape(-1), xx.reshape(-1)], axis=1)[:num_pts]
        seed = ""
        sampling = "4dpm_deterministic_point_grid"
    else:
        base_seed = int(ctx.config.get("run", {}).get("seed", 0))
        seed = _stable_int_seed(ctx.config.get("run", {}).get("name", "v105"), scene_id, int(frame_id), base_seed, "sam31_4dpm_point_prompts")
        rng = np.random.default_rng(int(seed) % (2**32 - 1))
        points_yx = rng.random((num_pts, 2), dtype=np.float32) * 2.0 - 1.0
        sampling = "4dpm_seeded_random"
    points_xy01 = np.stack(
        [
            np.clip((points_yx[:, 1] + 1.0) * 0.5, 0.0, 1.0),
            np.clip((points_yx[:, 0] + 1.0) * 0.5, 0.0, 1.0),
        ],
        axis=1,
    ).astype(np.float32)
    return points_xy01.tolist(), {
        "sam31_4dpm_num_pts": int(num_pts),
        "sam31_4dpm_point_sampling": sampling,
        "sam31_4dpm_point_seed": seed,
        "sam31_point_coordinate_conversion": "4dpm_yx_minus1_1_to_sam31_xy_0_1",
    }


def _write_promptable_segmentor_label(
    masks: list[dict[str, Any]],
    output_path: Path,
    *,
    h: int,
    w: int,
    min_pixels: int,
    min_area_ratio: float,
    max_area_ratio: float,
    broad_area_ratio: float,
    max_bbox_area_ratio: float,
    max_span_ratio: float,
    max_edge_pixel_ratio: float,
    max_edge_touch_sides: int,
    objectness_gate_enabled: bool,
    max_masks_per_frame: int,
    selection_policy: str = "score",
    target_area_ratio: float = 0.035,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    reject_counts: dict[str, int] = {}
    for idx, row in enumerate(masks):
        mask = np.asarray(row.get("segmentation")).astype(bool)
        while mask.ndim == 3 and 1 in mask.shape:
            mask = np.squeeze(mask, axis=int([dim for dim, size in enumerate(mask.shape) if size == 1][0]))
        if mask.ndim != 2:
            reject_counts["invalid_mask_rank"] = reject_counts.get("invalid_mask_rank", 0) + 1
            continue
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
        gate = _candidate_objectness_gate(
            mask,
            min_pixels=min_pixels,
            min_area_ratio=min_area_ratio,
            max_area_ratio=max_area_ratio,
            broad_area_ratio=broad_area_ratio,
            max_bbox_area_ratio=max_bbox_area_ratio,
            max_span_ratio=max_span_ratio,
            max_edge_pixel_ratio=max_edge_pixel_ratio,
            max_edge_touch_sides=max_edge_touch_sides,
            enabled=objectness_gate_enabled,
        )
        if not bool(gate["accepted"]):
            for reason in gate["reject_reasons"]:
                reject_counts[str(reason)] = reject_counts.get(str(reason), 0) + 1
            continue
        score = float(row.get("predicted_iou", row.get("stability_score", row.get("score", 0.0))) or 0.0)
        candidates.append(
            {
                "mask": mask,
                "pixel_area": int(gate["pixel_area"]),
                "area_ratio": float(gate["area_ratio"]),
                "score": score,
                "source_index": int(idx),
            }
        )
    policy = str(selection_policy).strip().lower()
    if policy in {"score", "best_score", "highest_score"}:
        candidates.sort(
            key=lambda item: (
                -float(item["score"]),
                abs(float(item["area_ratio"]) - float(target_area_ratio)),
                -float(item["area_ratio"]),
                int(item["source_index"]),
            )
        )
    elif policy in {"smallest", "4dpm_smallest", "select_smallest"}:
        candidates.sort(key=lambda item: (float(item["area_ratio"]), -float(item["score"]), int(item["source_index"])))
    elif policy in {"target_area", "target", "nearest_target"}:
        candidates.sort(
            key=lambda item: (
                abs(float(item["area_ratio"]) - float(target_area_ratio)),
                float(item["area_ratio"]),
                -float(item["score"]),
                int(item["source_index"]),
            )
        )
    else:
        candidates.sort(key=lambda item: (-float(item["pixel_area"]), -float(item["score"]), int(item["source_index"])))
    selected = candidates[: max(int(max_masks_per_frame), 1)]
    label = np.zeros((h, w), dtype=np.uint16)
    # Large masks are written first; smaller accepted masks overwrite overlaps.
    for label_id, item in enumerate(sorted(selected, key=lambda row: -float(row["pixel_area"])), start=1):
        label[np.asarray(item["mask"]).astype(bool)] = int(label_id)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), label)
    return {
        "raw_mask_count": int(len(masks)),
        "accepted_mask_count": int(len(candidates)),
        "written_mask_count": int(len(selected)),
        "rejected_mask_count": int(len(masks) - len(candidates)),
        "reject_counts": reject_counts,
        "nonzero": bool(np.count_nonzero(label) > 0),
        "selection_policy": policy or "largest",
        "target_area_ratio": float(target_area_ratio),
    }


def _run_promptable_segmentor_proposals_scene(
    ctx: PipelineContext,
    scene_id: str,
    frame_ids: list[int],
    proposal_source: str,
    mask_dir: Path,
) -> dict[str, Any]:
    t0 = time.time()
    peak_gpu_mb: float | str = ""
    source_key = _promptable_segmentor_source_key(proposal_source)
    if source_key is None:
        raise ValueError(f"unsupported promptable segmentor source={proposal_source}")
    mask_dir.mkdir(parents=True, exist_ok=True)
    for stale_mask in mask_dir.glob("*.png"):
        stale_mask.unlink()
    baselines = ctx.config.get("baselines", {})
    min_pixels = int(baselines.get("promptable_segmentor_min_mask_pixels", baselines.get("baseline_matrix_min_mask_pixels", 16)))
    min_area_ratio = float(baselines.get("promptable_segmentor_min_area_ratio", baselines.get("baseline_matrix_min_area_ratio", 0.0002)))
    max_area_ratio = float(baselines.get("promptable_segmentor_max_area_ratio", 1.0))
    objectness_gate_enabled = bool(
        baselines.get("promptable_segmentor_objectness_gate_enabled", False)
    )
    broad_area_ratio = float(baselines.get("promptable_segmentor_objectness_broad_area_ratio", baselines.get("baseline_matrix_objectness_broad_area_ratio", 0.08)))
    max_bbox_area_ratio = float(baselines.get("promptable_segmentor_objectness_max_bbox_area_ratio", baselines.get("baseline_matrix_objectness_max_bbox_area_ratio", 0.70)))
    max_span_ratio = float(baselines.get("promptable_segmentor_objectness_max_span_ratio", baselines.get("baseline_matrix_objectness_max_span_ratio", 0.90)))
    max_edge_pixel_ratio = float(baselines.get("promptable_segmentor_objectness_max_edge_pixel_ratio", baselines.get("baseline_matrix_objectness_max_edge_pixel_ratio", 0.08)))
    max_edge_touch_sides = int(baselines.get("promptable_segmentor_objectness_max_edge_touch_sides", baselines.get("baseline_matrix_objectness_max_edge_touch_sides", 1)))
    max_masks_per_frame = int(baselines.get("promptable_segmentor_max_masks_per_frame", 96))
    selection_policy = str(baselines.get("promptable_segmentor_selection_policy", "score"))
    frame_rows: list[dict[str, Any]] = []
    try:
        import torch  # type: ignore

        if not torch.cuda.is_available():
            raise RuntimeError(f"{source_key} promptable segmentor requires CUDA")
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        first_image = _native_rgb_for_segmentor(ctx, scene_id, int(frame_ids[0]))
        h, w = first_image.shape[:2]
        if source_key in {"sam2", "edgetam"}:
            generator = _build_promptable_amg(ctx, source_key, device="cuda:0")
            for frame_id in frame_ids:
                image = _native_rgb_for_segmentor(ctx, scene_id, int(frame_id))
                masks = list(generator.generate(image))
                row = _write_promptable_segmentor_label(
                    masks,
                    mask_dir / f"{int(frame_id)}.png",
                    h=h,
                    w=w,
                    min_pixels=min_pixels,
                    min_area_ratio=min_area_ratio,
                    max_area_ratio=max_area_ratio,
                    broad_area_ratio=broad_area_ratio,
                    max_bbox_area_ratio=max_bbox_area_ratio,
                    max_span_ratio=max_span_ratio,
                    max_edge_pixel_ratio=max_edge_pixel_ratio,
                    max_edge_touch_sides=max_edge_touch_sides,
                    objectness_gate_enabled=objectness_gate_enabled,
                    max_masks_per_frame=max_masks_per_frame,
                    selection_policy=selection_policy,
                )
                row.update({"frame_id": int(frame_id)})
                frame_rows.append(row)
            del generator
        else:
            old_sys_path = list(sys.path)
            root = REPO_ROOT / "third_party" / "sam3"
            checkpoint = _as_repo_path(
                ctx.config.get("paths", {}).get(
                    "sam3_image_checkpoint",
                    ctx.config.get("paths", {}).get("sam31_checkpoint"),
                )
            )
            if checkpoint is None or not checkpoint.exists():
                raise FileNotFoundError(f"missing SAM3/SAM3.1 image checkpoint: {checkpoint}")
            points_per_batch = max(int(baselines.get("promptable_segmentor_amg_points_per_batch", 64)), 1)
            confidence_threshold = float(baselines.get("promptable_segmentor_sam31_confidence_threshold", 0.25))
            quality_threshold = float(baselines.get("promptable_segmentor_sam31_quality_threshold", 0.0))
            per_point_selection_policy = str(baselines.get("promptable_segmentor_sam31_per_point_selection_policy", "best_score"))
            selection_policy = str(baselines.get("promptable_segmentor_sam31_selection_policy", "score"))
            target_area_ratio = float(
                baselines.get(
                    "promptable_segmentor_sam31_target_area_ratio",
                    baselines.get("baseline_matrix_objectness_target_area_ratio", 0.035),
                )
            )
            predictor = None
            try:
                sys.path.insert(0, str(root))
                from PIL import Image  # type: ignore
                from sam3.model.sam3_image_processor import Sam3Processor  # type: ignore
                from sam3.model_builder import build_sam3_image_model  # type: ignore

                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    model = build_sam3_image_model(
                        checkpoint_path=str(checkpoint),
                        load_from_HF=False,
                        device="cuda",
                        compile=False,
                        enable_inst_interactivity=True,
                    )
                    predictor = Sam3Processor(
                        model,
                        confidence_threshold=confidence_threshold,
                        device="cuda",
                    )
                    for frame_id in frame_ids:
                        image = _native_rgb_for_segmentor(ctx, scene_id, int(frame_id))
                        point_prompts, point_prompt_meta = _sam31_segmentor_4dpm_point_prompts(ctx, scene_id, int(frame_id))
                        state = predictor.set_image(Image.fromarray(np.ascontiguousarray(image[:, :, :3]).astype(np.uint8)))
                        point_prompts_np = np.asarray(point_prompts, dtype=np.float32)
                        point_prompts_px = point_prompts_np.copy()
                        point_prompts_px[:, 0] *= float(max(w - 1, 1))
                        point_prompts_px[:, 1] *= float(max(h - 1, 1))
                        masks_for_frame: list[dict[str, Any]] = []
                        prompt_batch_mask_counts: list[int] = []
                        prompt_batch_raw_option_counts: list[int] = []
                        prompt_prefilter_reject_counts: dict[str, int] = {}
                        prompt_batch_failure_count = 0
                        raw_multimask_option_count = 0
                        for prompt_batch_idx, prompt_start in enumerate(range(0, len(point_prompts), points_per_batch)):
                            prompt_batch = point_prompts_px[prompt_start : prompt_start + points_per_batch]
                            try:
                                point_coords = np.asarray(prompt_batch, dtype=np.float32).reshape(-1, 1, 2)
                                point_labels = np.ones((point_coords.shape[0], 1), dtype=np.int32)
                                mask_options, score_values, _ = model.predict_inst(
                                    state,
                                    point_coords=point_coords,
                                    point_labels=point_labels,
                                    multimask_output=True,
                                )
                            except Exception:
                                prompt_batch_failure_count += 1
                                continue
                            mask_options_np = np.asarray(mask_options)
                            score_values_np = np.asarray(score_values, dtype=np.float32)
                            if mask_options_np.ndim == 2:
                                mask_options_np = mask_options_np[None, None, :, :]
                            elif mask_options_np.ndim == 3:
                                if score_values_np.ndim == 1 and score_values_np.size == mask_options_np.shape[0]:
                                    mask_options_np = mask_options_np[None, :, :, :]
                                else:
                                    mask_options_np = mask_options_np[:, None, :, :]
                            if score_values_np.ndim == 0:
                                score_values_np = score_values_np.reshape(1, 1)
                            elif score_values_np.ndim == 1:
                                if mask_options_np.shape[0] == 1 and score_values_np.size == mask_options_np.shape[1]:
                                    score_values_np = score_values_np[None, :]
                                else:
                                    score_values_np = score_values_np[:, None]
                            prompt_batch_raw_option_counts.append(int(mask_options_np.shape[0] * mask_options_np.shape[1]))
                            raw_multimask_option_count += int(mask_options_np.shape[0] * mask_options_np.shape[1])
                            selected_from_batch = 0
                            for local_prompt_idx in range(mask_options_np.shape[0]):
                                accepted_options: list[dict[str, Any]] = []
                                for option_idx in range(mask_options_np.shape[1]):
                                    option = mask_options_np[local_prompt_idx, option_idx]
                                    score = (
                                        float(score_values_np[local_prompt_idx, option_idx])
                                        if local_prompt_idx < score_values_np.shape[0] and option_idx < score_values_np.shape[1]
                                        else 0.0
                                    )
                                    if quality_threshold > 0.0 and score < quality_threshold:
                                        prompt_prefilter_reject_counts["below_quality_threshold"] = (
                                            prompt_prefilter_reject_counts.get("below_quality_threshold", 0) + 1
                                        )
                                        continue
                                    mask = np.asarray(option).astype(bool)
                                    if mask.shape[:2] != (h, w):
                                        mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
                                    gate = _candidate_objectness_gate(
                                        mask,
                                        min_pixels=min_pixels,
                                        min_area_ratio=min_area_ratio,
                                        max_area_ratio=max_area_ratio,
                                        broad_area_ratio=broad_area_ratio,
                                        max_bbox_area_ratio=max_bbox_area_ratio,
                                        max_span_ratio=max_span_ratio,
                                        max_edge_pixel_ratio=max_edge_pixel_ratio,
                                        max_edge_touch_sides=max_edge_touch_sides,
                                        enabled=objectness_gate_enabled,
                                    )
                                    if not bool(gate["accepted"]):
                                        for reason in gate["reject_reasons"]:
                                            key = str(reason)
                                            prompt_prefilter_reject_counts[key] = prompt_prefilter_reject_counts.get(key, 0) + 1
                                        continue
                                    accepted_options.append(
                                        {
                                            "mask": mask,
                                            "pixel_area": int(gate["pixel_area"]),
                                            "area_ratio": float(gate["area_ratio"]),
                                            "score": score,
                                            "option_idx": int(option_idx),
                                        }
                                    )
                                if not accepted_options:
                                    prompt_prefilter_reject_counts["no_accepted_option_for_prompt"] = (
                                        prompt_prefilter_reject_counts.get("no_accepted_option_for_prompt", 0) + 1
                                    )
                                    continue
                                per_point_policy = per_point_selection_policy.strip().lower()
                                if per_point_policy in {"score", "best_score", "highest_score"}:
                                    point_key = lambda item: (
                                        -float(item["score"]),
                                        abs(float(item["area_ratio"]) - float(target_area_ratio)),
                                        int(item["option_idx"]),
                                    )
                                elif per_point_policy in {"target_area", "target", "nearest_target"}:
                                    point_key = lambda item: (
                                        abs(float(item["area_ratio"]) - float(target_area_ratio)),
                                        -float(item["score"]),
                                        int(item["option_idx"]),
                                    )
                                elif per_point_policy in {"largest", "large"}:
                                    point_key = lambda item: (-float(item["area_ratio"]), -float(item["score"]), int(item["option_idx"]))
                                else:
                                    point_key = lambda item: (float(item["area_ratio"]), -float(item["score"]), int(item["option_idx"]))
                                selected = sorted(accepted_options, key=point_key)[0]
                                selected_from_batch += 1
                                masks_for_frame.append(
                                    {
                                        "segmentation": selected["mask"],
                                        "predicted_iou": float(selected["score"]),
                                        "score": float(selected["score"]),
                                        "visual_prompt_type": "point",
                                        "source_prompt_batch_index": int(prompt_batch_idx),
                                        "source_prompt_batch_start": int(prompt_start),
                                        "source_prompt_index": int(prompt_start + local_prompt_idx),
                                        "source_prompt_option_index": int(selected["option_idx"]),
                                        "source_prompt_area_ratio": float(selected["area_ratio"]),
                                        "sam31_predict_inst_selection": per_point_selection_policy,
                                    }
                                )
                            prompt_batch_mask_counts.append(int(selected_from_batch))
                        row = _write_promptable_segmentor_label(
                            masks_for_frame,
                            mask_dir / f"{int(frame_id)}.png",
                            h=h,
                            w=w,
                            min_pixels=min_pixels,
                            min_area_ratio=min_area_ratio,
                            max_area_ratio=max_area_ratio,
                            broad_area_ratio=broad_area_ratio,
                            max_bbox_area_ratio=max_bbox_area_ratio,
                            max_span_ratio=max_span_ratio,
                            max_edge_pixel_ratio=max_edge_pixel_ratio,
                            max_edge_touch_sides=max_edge_touch_sides,
                            objectness_gate_enabled=objectness_gate_enabled,
                            max_masks_per_frame=max_masks_per_frame,
                            selection_policy=selection_policy,
                            target_area_ratio=target_area_ratio,
                        )
                        row.update(
                            {
                                "frame_id": int(frame_id),
                                "sam31_visual_prompt_type": "point",
                                "sam31_visual_prompt_count": int(len(point_prompts)),
                                "sam31_visual_prompt_batch_size": int(points_per_batch),
                                "sam31_visual_prompt_batch_count": int((len(point_prompts) + points_per_batch - 1) // points_per_batch),
                                "sam31_visual_prompt_batch_failure_count": int(prompt_batch_failure_count),
                                "sam31_prompt_batch_mask_counts": prompt_batch_mask_counts,
                                "sam31_prompt_batch_raw_option_counts": prompt_batch_raw_option_counts,
                                "sam31_raw_multimask_option_count": int(raw_multimask_option_count),
                                "sam31_prefilter_reject_counts": prompt_prefilter_reject_counts,
                                "sam31_quality_threshold": float(quality_threshold),
                                "sam31_segmentor_policy": "official_predict_inst_4dpm_num_pts_best_score",
                                "sam31_predict_inst_enable_inst_interactivity": True,
                                "sam31_predict_inst_multimask_output": True,
                                "sam31_predict_inst_point_coordinate_format": "pixel_xy",
                                "sam31_per_point_selection_policy": per_point_selection_policy,
                                "sam31_selection_policy": selection_policy,
                                **point_prompt_meta,
                                "sam31_image_checkpoint": _rel(checkpoint),
                                "sam31_image_checkpoint_sha256": _sha256_file(checkpoint),
                                "sam31_confidence_threshold": confidence_threshold,
                            }
                        )
                        frame_rows.append(row)
            finally:
                if predictor is not None and hasattr(predictor, "close"):
                    predictor.close()
                del predictor
                sys.path[:] = old_sys_path
        torch.cuda.synchronize()
        peak_gpu_mb = float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)
        nonzero_frames = sum(1 for row in frame_rows if bool(row.get("nonzero")))
        status = "completed" if nonzero_frames == len(frame_ids) else "partial" if nonzero_frames > 0 else "failed"
        return {
            "schema_version": "stream4d_v105_promptable_segmentor_proposal_runtime_row_v1",
            "segmentor_provider": source_key,
            "proposal_source": proposal_source,
            "status": status,
            "scene_id": scene_id,
            "frame_ids": ";".join(str(int(frame_id)) for frame_id in frame_ids),
            "frame_count": len(frame_ids),
            "frame_output_count": len(frame_rows),
            "nonzero_output_frame_count": int(nonzero_frames),
            "zero_output_frame_ids": [int(row["frame_id"]) for row in frame_rows if not bool(row.get("nonzero"))],
            "raw_mask_count": int(sum(int(row.get("raw_mask_count", 0)) for row in frame_rows)),
            "accepted_mask_count": int(sum(int(row.get("accepted_mask_count", 0)) for row in frame_rows)),
            "written_mask_count": int(sum(int(row.get("written_mask_count", 0)) for row in frame_rows)),
            "latency_sec": time.time() - t0,
            "peak_gpu_memory_mb": peak_gpu_mb,
            "mask_dir": _rel(mask_dir),
            "frame_rows": frame_rows,
        }
    except Exception as exc:
        return {
            "schema_version": "stream4d_v105_promptable_segmentor_proposal_runtime_row_v1",
            "segmentor_provider": source_key,
            "proposal_source": proposal_source,
            "status": "failed",
            "failure_type": "runtime_error",
            "scene_id": scene_id,
            "frame_ids": ";".join(str(int(frame_id)) for frame_id in frame_ids),
            "frame_count": len(frame_ids),
            "frame_output_count": len(frame_rows),
            "nonzero_output_frame_count": sum(1 for row in frame_rows if bool(row.get("nonzero"))),
            "latency_sec": time.time() - t0,
            "peak_gpu_memory_mb": peak_gpu_mb,
            "mask_dir": _rel(mask_dir),
            "failure_reason": _safe_error(exc),
            "frame_rows": frame_rows,
        }


def _load_dense_sam31_proposal_label(
    ctx: PipelineContext,
    scene_id: str,
    frame_id: int,
    proposal_source: str,
    frame_ids: list[int],
) -> tuple[np.ndarray | None, int | None, str, bool]:
    proposal_path = _proposal_label_path_for_baseline(ctx, scene_id, frame_id, proposal_source)
    label = _preprocessed_label_for_sgq(ctx, scene_id, frame_id, proposal_path)
    if label is not None:
        return label, int(frame_id), _rel(proposal_path), False
    if not bool(ctx.config.get("baselines", {}).get("full_sam31_allow_nearest_proposal_for_missing_frames", True)):
        return None, None, _rel(proposal_path), False
    available: list[tuple[int, int, Path]] = []
    for candidate_frame_id in frame_ids:
        candidate_path = _proposal_label_path_for_baseline(ctx, scene_id, int(candidate_frame_id), proposal_source)
        if candidate_path.exists():
            available.append((abs(int(candidate_frame_id) - int(frame_id)), int(candidate_frame_id), candidate_path))
    if not available:
        return None, None, _rel(proposal_path), False
    _distance, support_frame_id, support_path = min(
        available,
        key=lambda item: (item[0], 0 if item[1] <= int(frame_id) else 1, item[1]),
    )
    label = _preprocessed_label_for_sgq(ctx, scene_id, int(support_frame_id), support_path)
    return label, int(support_frame_id), _rel(support_path), True


def _run_dense_sam31_keyframe_scene(
    ctx: PipelineContext,
    scene_id: str,
    frame_ids: list[int],
    mask_dir: Path,
    *,
    variant_id: str,
    proposal_source: str,
    checkpoint: Path,
) -> dict[str, Any]:
    t0 = time.time()
    peak_gpu_mb: float | str = ""
    mask_dir.mkdir(parents=True, exist_ok=True)
    for stale_mask in mask_dir.glob("*.png"):
        stale_mask.unlink()
    baselines = ctx.config.get("baselines", {})
    max_queries = int(baselines.get("full_sam31_max_queries_per_scene", 96))
    min_pixels = int(baselines.get("full_sam31_min_mask_pixels", 16))
    min_area_ratio = float(baselines.get("full_sam31_min_area_ratio", 0.0002))
    max_area_ratio = float(baselines.get("full_sam31_max_area_ratio", 0.45))
    max_candidates = max(int(baselines.get("full_sam31_max_candidates_per_anchor_frame", 2)), 1)
    min_uncovered_pixels = max(int(baselines.get("full_sam31_min_uncovered_candidate_pixels", min_pixels)), 1)
    candidate_order = str(baselines.get("full_sam31_candidate_order", "small_to_large_uncovered")).strip().lower()
    id_stabilization_enabled = bool(baselines.get("full_sam31_temporal_id_stabilization_enabled", True))
    id_stabilization_lookback = max(int(baselines.get("full_sam31_temporal_id_lookback", 3)), 1)
    id_stabilization_min_score = float(baselines.get("full_sam31_temporal_id_min_match_score", 0.18))
    output_prob_thresh = float(
        baselines.get("full_sam31_output_prob_thresh", ctx.config.get("local", {}).get("sgq_sam31_output_prob_thresh", 0.5))
    )
    query_count = 0
    candidate_count = 0
    output_mask_count = 0
    frame_output_count = 0
    nonzero_output_frame_count = 0
    zero_output_frame_ids: list[int] = []
    missing_direct_proposal_frame_ids: list[int] = []
    nearest_support_count = 0
    failed_object_count = 0
    redundant_object_count = 0
    candidate_prompt_failure_count = 0
    covered_candidate_skip_count = 0
    prompt_cap_skipped_candidate_count = 0
    per_frame_candidate_stats: list[dict[str, Any]] = []
    track_rows: list[dict[str, Any]] = []
    id_stabilization_summary: dict[str, Any] = {"enabled": False}
    prompt_policy = "dense_input_frame_sam31_box_prompt_uncovered_proposal_sweep"
    scope_note = (
        "SAM3.1 dense input-frame box prompting followed by temporal id stabilization; every configured stride input frame is prompted. "
        "Candidates are swept on the current frame with uncovered proposal priority instead of stopping after only the largest masks. "
        "When the segmentor output is missing for an input frame, nearest segmentor support supplies the box prompt on the current RGB frame."
    )
    predictor = None
    capped_by_query_limit = False
    try:
        import torch  # type: ignore

        if not torch.cuda.is_available():
            raise RuntimeError("SAM3.1 dense keyframe baseline requires CUDA")
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        predictor = _build_sgq_image_refiner(ctx, "sam31", "cuda:0")
        predictor.output_prob_thresh = output_prob_thresh
        first_image = _preprocessed_rgb_for_sam2(ctx, scene_id, int(frame_ids[0]))
        h, w = first_image.shape[:2]
        frame_area = float(max(h * w, 1))
        next_track_id = 1
        persistent_track_ids: dict[tuple[str, int], int] = {}

        for local_idx, frame_id in enumerate(frame_ids):
            image = _preprocessed_rgb_for_sam2(ctx, scene_id, int(frame_id))
            if image.shape[:2] != (h, w):
                image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
            label = np.zeros((h, w), dtype=np.uint16)
            proposal_label, support_frame_id, proposal_path, used_nearest = _load_dense_sam31_proposal_label(
                ctx, scene_id, int(frame_id), proposal_source, frame_ids
            )
            if proposal_label is None:
                zero_output_frame_ids.append(int(frame_id))
                missing_direct_proposal_frame_ids.append(int(frame_id))
                cv2.imwrite(str(mask_dir / f"{int(frame_id)}.png"), label)
                frame_output_count += 1
                continue
            if used_nearest:
                nearest_support_count += 1
                missing_direct_proposal_frame_ids.append(int(frame_id))
            if proposal_label.shape[:2] != (h, w):
                proposal_label = cv2.resize(proposal_label, (w, h), interpolation=cv2.INTER_NEAREST)
            proposal_ids = [int(value) for value in np.unique(proposal_label) if int(value) > 0]
            proposal_ids.sort(key=lambda value: int(np.count_nonzero(proposal_label == value)), reverse=True)
            valid_candidates: list[tuple[int, np.ndarray, tuple[float, float, float, float], float]] = []
            for proposal_id in proposal_ids:
                proposal_mask = proposal_label == proposal_id
                pixel_area = int(np.count_nonzero(proposal_mask))
                area_ratio = float(pixel_area) / frame_area
                if pixel_area < min_pixels or area_ratio < min_area_ratio or area_ratio > max_area_ratio:
                    continue
                bbox = _bbox_xyxy(proposal_mask)
                if bbox is None:
                    continue
                valid_candidates.append((int(proposal_id), proposal_mask.astype(bool), bbox, area_ratio))
                candidate_count += 1
            if candidate_order in {"small_to_large", "small_to_large_uncovered", "small_first"}:
                valid_candidates.sort(key=lambda item: (item[3], int(np.count_nonzero(item[1]))))
            elif candidate_order in {"large_to_small", "large_first"}:
                valid_candidates.sort(key=lambda item: (-item[3], -int(np.count_nonzero(item[1]))))
            else:
                # Prefer object-scale regions over room-scale surfaces while keeping deterministic ordering.
                valid_candidates.sort(key=lambda item: (abs(item[3] - 0.02), item[3]))
            predictor.set_image(image)
            successful_this_frame = 0
            prompted_this_frame = 0
            covered_skip_this_frame = 0
            cap_skip_this_frame = 0
            failure_this_frame = 0
            for candidate_idx, (proposal_id, proposal_mask, bbox, area_ratio) in enumerate(valid_candidates):
                if successful_this_frame >= max_candidates:
                    skipped = len(valid_candidates) - candidate_idx
                    prompt_cap_skipped_candidate_count += skipped
                    cap_skip_this_frame += skipped
                    break
                if query_count >= max_queries:
                    capped_by_query_limit = True
                    break
                uncovered_candidate_pixels = int(np.count_nonzero(proposal_mask & (label == 0)))
                if uncovered_candidate_pixels < min_uncovered_pixels:
                    covered_candidate_skip_count += 1
                    covered_skip_this_frame += 1
                    continue
                query_count += 1
                prompted_this_frame += 1
                try:
                    masks, scores, _lowres = predictor.predict(
                        box=np.asarray(bbox, dtype=np.float32),
                        multimask_output=True,
                        return_logits=False,
                        normalize_coords=True,
                    )
                    mask_options = np.asarray(masks)
                    if mask_options.ndim == 2:
                        mask_options = mask_options[None, :, :]
                    if mask_options.size == 0 or mask_options.shape[0] == 0:
                        raise RuntimeError("sam31_box_prompt_returned_no_masks")
                    score_values = np.asarray(scores).reshape(-1)
                    best_key = (-1.0, -1.0, -1.0)
                    best_idx = 0
                    best_iou = 0.0
                    best_coverage = 0.0
                    target_area = float(max(np.count_nonzero(proposal_mask), 1))
                    for idx, option in enumerate(mask_options):
                        option_mask = option.astype(bool)
                        if option_mask.shape[:2] != (h, w):
                            option_mask = cv2.resize(option_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
                        inter = float(np.count_nonzero(option_mask & proposal_mask))
                        union = float(np.count_nonzero(option_mask | proposal_mask))
                        iou = inter / union if union else 0.0
                        coverage = inter / target_area
                        score = float(score_values[idx]) if idx < score_values.size else 0.0
                        key = (iou, coverage, score)
                        if key > best_key:
                            best_key = key
                            best_idx = int(idx)
                            best_iou = float(iou)
                            best_coverage = float(coverage)
                    refined = mask_options[best_idx].astype(bool)
                    if refined.shape[:2] != (h, w):
                        refined = cv2.resize(refined.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
                    region = refined & (label == 0)
                    output_area = int(np.count_nonzero(region))
                    score = float(score_values[best_idx]) if best_idx < score_values.size else 0.0
                    row = {
                        "prompt_policy": prompt_policy,
                        "prompt_frame_id": int(frame_id),
                        "prompt_local_frame_idx": int(local_idx),
                        "proposal_frame_id": int(support_frame_id) if support_frame_id is not None else "",
                        "proposal_mask_id": int(proposal_id),
                        "proposal_source": proposal_source,
                        "proposal_path": proposal_path,
                        "used_nearest_proposal_support": bool(used_nearest),
                        "sam31_score": score,
                        "sam31_candidate_overlap_iou": best_iou,
                        "sam31_candidate_overlap_coverage": best_coverage,
                        "uncovered_candidate_pixels_before_prompt": uncovered_candidate_pixels,
                    }
                    if output_area < min_pixels:
                        candidate_prompt_failure_count += 1
                        failure_this_frame += 1
                        track_rows.append(
                            {
                                **row,
                                "candidate_failure": True,
                                "written_frame_count": 0,
                                "written_pixel_count": 0,
                                "failure_reason": "sam31_box_prompt_output_below_min_pixels",
                            }
                        )
                        continue
                    track_key = (str(proposal_source), int(proposal_id))
                    assigned_track_id = persistent_track_ids.get(track_key)
                    if assigned_track_id is None:
                        assigned_track_id = int(next_track_id)
                        persistent_track_ids[track_key] = assigned_track_id
                        next_track_id += 1
                    label[region] = int(assigned_track_id)
                    output_mask_count += 1
                    successful_this_frame += 1
                    track_rows.append(
                        {
                            **row,
                            "candidate_failure": False,
                            "track_id": int(assigned_track_id),
                            "written_frame_count": 1,
                            "written_pixel_count": int(output_area),
                            "failure_reason": "",
                        }
                    )
                except Exception as exc:
                    candidate_prompt_failure_count += 1
                    failure_this_frame += 1
                    track_rows.append(
                        {
                            "prompt_policy": prompt_policy,
                            "prompt_frame_id": int(frame_id),
                            "prompt_local_frame_idx": int(local_idx),
                            "proposal_frame_id": int(support_frame_id) if support_frame_id is not None else "",
                            "proposal_mask_id": int(proposal_id),
                            "proposal_source": proposal_source,
                            "proposal_path": proposal_path,
                            "used_nearest_proposal_support": bool(used_nearest),
                            "candidate_failure": True,
                            "written_frame_count": 0,
                            "written_pixel_count": 0,
                            "failure_reason": _safe_error(exc),
                        }
                    )
            predictor.close()
            if int(np.count_nonzero(label)) > 0:
                nonzero_output_frame_count += 1
            else:
                zero_output_frame_ids.append(int(frame_id))
            cv2.imwrite(str(mask_dir / f"{int(frame_id)}.png"), label)
            frame_output_count += 1
            per_frame_candidate_stats.append(
                {
                    "frame_id": int(frame_id),
                    "valid_candidate_count": len(valid_candidates),
                    "prompted_candidate_count": prompted_this_frame,
                    "successful_candidate_count": successful_this_frame,
                    "candidate_prompt_failure_count": failure_this_frame,
                    "covered_candidate_skip_count": covered_skip_this_frame,
                    "prompt_cap_skipped_candidate_count": cap_skip_this_frame,
                    "nonzero_output": int(np.count_nonzero(label)) > 0,
                    "output_label_count": len([int(value) for value in np.unique(label) if int(value) > 0]),
                }
            )
            if capped_by_query_limit and query_count >= max_queries and local_idx + 1 < len(frame_ids):
                for remaining_frame_id in frame_ids[local_idx + 1 :]:
                    cv2.imwrite(str(mask_dir / f"{int(remaining_frame_id)}.png"), np.zeros((h, w), dtype=np.uint16))
                    zero_output_frame_ids.append(int(remaining_frame_id))
                    frame_output_count += 1
                break
        torch.cuda.synchronize()
        peak_gpu_mb = float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)
        if id_stabilization_enabled:
            id_stabilization_summary = _stabilize_label_ids_temporally(
                mask_dir=mask_dir,
                frame_ids=[int(frame_id) for frame_id in frame_ids],
                min_pixels=min_pixels,
                lookback=id_stabilization_lookback,
                min_match_score=id_stabilization_min_score,
                ctx=ctx,
                scene_id=scene_id,
                geometry_enabled=track_registry_geometry_enabled,
                geometry_max_points=track_registry_geometry_max_points,
                geometry_near_threshold_m=track_registry_geometry_near_threshold_m,
                geometry_centroid_threshold_m=track_registry_geometry_centroid_threshold_m,
                geometry_min_score=track_registry_geometry_min_score,
            )
        if output_mask_count <= 0:
            status = "failed"
        elif zero_output_frame_ids or capped_by_query_limit or failed_object_count > 0:
            status = "partial"
        else:
            status = "completed"
        if zero_output_frame_ids:
            failure_reason = "zero_output_input_frames=" + ";".join(str(frame_id) for frame_id in zero_output_frame_ids)
        elif capped_by_query_limit:
            failure_reason = "query_limit_capped_object_prompts"
        elif failed_object_count > 0:
            failure_reason = "one or more dense SAM3.1 box prompts produced no valid masks"
        else:
            failure_reason = ""
        return {
            "schema_version": "stream4d_v105_full_sam31_runtime_row_v1",
            "variant_id": variant_id,
            "proposal_source": proposal_source,
            "status": status,
            "scene_id": scene_id,
            "frame_ids": ";".join(str(int(frame_id)) for frame_id in frame_ids),
            "frame_count": len(frame_ids),
            "frame_output_count": frame_output_count,
            "candidate_count": candidate_count,
            "query_count": query_count,
            "query_limit": max_queries,
            "capped_by_query_limit": capped_by_query_limit,
            "output_mask_count": output_mask_count,
            "nonzero_output_frame_count": nonzero_output_frame_count,
            "zero_output_frame_ids": zero_output_frame_ids,
            "all_input_frames_have_nonzero_masks": len(zero_output_frame_ids) == 0,
            "tracked_object_count": int(id_stabilization_summary.get("track_count", len(persistent_track_ids))),
            "failed_object_count": failed_object_count,
            "candidate_prompt_failure_count": candidate_prompt_failure_count,
            "covered_candidate_skip_count": covered_candidate_skip_count,
            "prompt_cap_skipped_candidate_count": prompt_cap_skipped_candidate_count,
            "unprompted_valid_candidate_count": max(candidate_count - query_count - covered_candidate_skip_count, 0),
            "sam31_video_session_count": frame_output_count,
            "sam31_oom_retry_count": 0,
            "objects_per_session": max_candidates,
            "max_internal_objects": max_candidates,
            "anchor_frame_stride": 1,
            "max_candidates_per_anchor_frame": max_candidates,
            "candidate_order": candidate_order,
            "min_uncovered_candidate_pixels": min_uncovered_pixels,
            "prompt_policy": prompt_policy,
            "nearest_proposal_support_count": nearest_support_count,
            "missing_direct_proposal_frame_ids": sorted(set(missing_direct_proposal_frame_ids)),
            "per_frame_candidate_stats": per_frame_candidate_stats,
            "track_rows": track_rows,
            "track_rows_are_pre_id_stabilization": bool(id_stabilization_enabled),
            "id_stabilization_enabled": bool(id_stabilization_enabled),
            "id_stabilization_policy": id_stabilization_summary.get("policy", ""),
            "id_stabilization_track_count": id_stabilization_summary.get("track_count", ""),
            "id_stabilization_reused_assignment_count": id_stabilization_summary.get("reused_assignment_count", ""),
            "id_stabilization_new_assignment_count": id_stabilization_summary.get("new_assignment_count", ""),
            "id_stabilization_summary": id_stabilization_summary,
            "latency_sec": time.time() - t0,
            "peak_gpu_memory_mb": peak_gpu_mb,
            "checkpoint": _rel(checkpoint),
            "checkpoint_sha256": _sha256_file(checkpoint),
            "failure_reason": failure_reason,
            "reference_scope": scope_note,
            "eligible_for_phase4_speed_gate": status == "completed",
        }
    except Exception as exc:
        reason = _safe_error(exc)
        return {
            "schema_version": "stream4d_v105_full_sam31_runtime_row_v1",
            "variant_id": variant_id,
            "proposal_source": proposal_source,
            "status": "failed",
            "failure_type": "OOM" if "out of memory" in reason.lower() else "runtime_error",
            "scene_id": scene_id,
            "frame_ids": ";".join(str(int(frame_id)) for frame_id in frame_ids),
            "frame_count": len(frame_ids),
            "frame_output_count": frame_output_count,
            "candidate_count": candidate_count,
            "query_count": query_count,
            "query_limit": max_queries,
            "capped_by_query_limit": capped_by_query_limit,
            "output_mask_count": output_mask_count,
            "tracked_object_count": 0,
            "failed_object_count": failed_object_count,
            "candidate_prompt_failure_count": candidate_prompt_failure_count,
            "covered_candidate_skip_count": covered_candidate_skip_count,
            "prompt_cap_skipped_candidate_count": prompt_cap_skipped_candidate_count,
            "unprompted_valid_candidate_count": max(candidate_count - query_count - covered_candidate_skip_count, 0),
            "sam31_video_session_count": frame_output_count,
            "sam31_oom_retry_count": 0,
            "objects_per_session": max_candidates,
            "max_internal_objects": max_candidates,
            "latency_sec": time.time() - t0,
            "peak_gpu_memory_mb": peak_gpu_mb,
            "checkpoint": _rel(checkpoint),
            "checkpoint_sha256": _sha256_file(checkpoint),
            "failure_reason": reason,
            "reference_scope": scope_note,
            "eligible_for_phase4_speed_gate": False,
        }
    finally:
        if predictor is not None and hasattr(predictor, "close"):
            predictor.close()
        if predictor is not None and hasattr(predictor, "predictor") and hasattr(predictor.predictor, "shutdown"):
            try:
                predictor.predictor.shutdown()
            except Exception:
                pass
        gc.collect()
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def _run_sam31_gap_tracking_reference_scene(
    ctx: PipelineContext,
    scene_id: str,
    frame_ids: list[int],
    mask_dir: Path,
    *,
    variant_id: str,
    proposal_source: str,
    checkpoint: Path,
) -> dict[str, Any]:
    t0 = time.time()
    peak_gpu_mb: float | str = ""
    mask_dir.mkdir(parents=True, exist_ok=True)
    for stale_mask in mask_dir.glob("*.png"):
        stale_mask.unlink()

    baselines = ctx.config.get("baselines", {})
    max_queries = max(int(baselines.get("full_sam31_max_queries_per_scene", 192)), 1)
    min_pixels = int(baselines.get("full_sam31_min_mask_pixels", 16))
    min_area_ratio = float(baselines.get("full_sam31_min_area_ratio", 0.0002))
    max_area_ratio = float(baselines.get("full_sam31_max_area_ratio", 0.45))
    objectness_gate_enabled = bool(baselines.get("full_sam31_candidate_objectness_gate_enabled", True))
    objectness_broad_area_ratio = float(baselines.get("full_sam31_objectness_broad_area_ratio", 0.08))
    objectness_max_bbox_area_ratio = float(baselines.get("full_sam31_objectness_max_bbox_area_ratio", 0.70))
    objectness_max_span_ratio = float(baselines.get("full_sam31_objectness_max_span_ratio", 0.90))
    objectness_max_edge_pixel_ratio = float(baselines.get("full_sam31_objectness_max_edge_pixel_ratio", 0.08))
    objectness_max_edge_touch_sides = int(baselines.get("full_sam31_objectness_max_edge_touch_sides", 1))
    objectness_target_area_ratio = float(baselines.get("full_sam31_objectness_target_area_ratio", 0.035))
    max_candidates_per_frame = max(int(baselines.get("full_sam31_max_candidates_per_anchor_frame", 8)), 1)
    min_uncovered_pixels = max(int(baselines.get("full_sam31_min_uncovered_candidate_pixels", min_pixels)), 1)
    min_uncovered_fraction = float(baselines.get("full_sam31_min_uncovered_candidate_fraction", 0.50))
    prompt_covered_candidates_enabled = bool(
        baselines.get("full_sam31_prompt_covered_anchor_candidates_enabled", False)
    )
    objects_per_session = max(int(baselines.get("full_sam31_objects_per_session", 8)), 1)
    internal_cap = max(int(baselines.get("full_sam31_max_internal_objects", 16)), 1)
    objects_per_session = min(objects_per_session, internal_cap)
    candidate_order = str(baselines.get("full_sam31_candidate_order", "small_to_large_uncovered")).strip().lower()
    reuse_existing_id_enabled = bool(baselines.get("full_sam31_reuse_existing_anchor_id_enabled", True))
    reuse_existing_id_min_iou = float(baselines.get("full_sam31_reuse_existing_anchor_id_min_iou", 0.10))
    reuse_existing_id_min_coverage = float(baselines.get("full_sam31_reuse_existing_anchor_id_min_coverage", 0.25))
    reuse_tracked_output_id_enabled = bool(baselines.get("full_sam31_reuse_existing_tracked_output_id_enabled", True))
    reuse_tracked_output_id_min_iou = float(
        baselines.get("full_sam31_reuse_existing_tracked_output_id_min_iou", reuse_existing_id_min_iou)
    )
    reuse_tracked_output_id_min_coverage = float(
        baselines.get("full_sam31_reuse_existing_tracked_output_id_min_coverage", reuse_existing_id_min_coverage)
    )
    reuse_temporal_id_enabled = bool(baselines.get("full_sam31_reuse_existing_temporal_id_enabled", True))
    reuse_temporal_id_lookback = max(int(baselines.get("full_sam31_reuse_existing_temporal_id_lookback", 3)), 1)
    reuse_temporal_id_min_score = float(baselines.get("full_sam31_reuse_existing_temporal_id_min_score", 0.52))
    reuse_temporal_id_min_margin = float(baselines.get("full_sam31_reuse_existing_temporal_id_min_margin", 0.04))
    reuse_temporal_id_merge_enabled = bool(baselines.get("full_sam31_reuse_existing_temporal_id_merge_enabled", True))
    reuse_temporal_id_local_relabel_enabled = bool(
        baselines.get("full_sam31_reuse_existing_temporal_id_local_relabel_enabled", reuse_temporal_id_merge_enabled)
    )
    reuse_temporal_id_cross_min_mask_iou = float(
        baselines.get("full_sam31_reuse_existing_temporal_id_cross_min_mask_iou", 0.35)
    )
    reuse_temporal_id_cross_min_bbox_iou = float(
        baselines.get("full_sam31_reuse_existing_temporal_id_cross_min_bbox_iou", 0.35)
    )
    reuse_temporal_id_cross_min_area_score = float(
        baselines.get("full_sam31_reuse_existing_temporal_id_cross_min_area_score", 0.80)
    )
    track_registry_lookback = max(int(baselines.get("full_sam31_track_registry_lookback", 5)), 1)
    track_registry_min_score = float(baselines.get("full_sam31_track_registry_min_score", 0.18))
    track_registry_update_from_outputs_enabled = bool(
        baselines.get("full_sam31_track_registry_update_from_outputs_enabled", False)
    )
    track_registry_history_max_entries = max(int(baselines.get("full_sam31_track_registry_history_max_entries", 96)), 1)
    track_registry_geometry_enabled = bool(baselines.get("full_sam31_track_registry_geometry_enabled", False))
    track_registry_geometry_max_points = max(int(baselines.get("full_sam31_track_registry_geometry_max_points", 512)), 1)
    track_registry_geometry_near_threshold_m = float(baselines.get("full_sam31_track_registry_geometry_near_threshold_m", 0.08))
    track_registry_geometry_centroid_threshold_m = float(
        baselines.get("full_sam31_track_registry_geometry_centroid_threshold_m", 0.45)
    )
    track_registry_geometry_min_score = float(baselines.get("full_sam31_track_registry_geometry_min_score", 0.25))
    id_stabilization_enabled = bool(baselines.get("full_sam31_temporal_id_stabilization_enabled", False))
    id_stabilization_lookback = max(int(baselines.get("full_sam31_temporal_id_lookback", 3)), 1)
    id_stabilization_min_score = float(baselines.get("full_sam31_temporal_id_min_match_score", 0.18))
    tracked_mask_support_gate_enabled = bool(baselines.get("full_sam31_tracked_mask_support_gate_enabled", False))
    tracked_mask_support_gate_min_score = float(baselines.get("full_sam31_tracked_mask_support_gate_min_score", 0.16))
    tracked_mask_support_gate_min_iou = float(baselines.get("full_sam31_tracked_mask_support_gate_min_iou", 0.02))
    tracked_mask_support_gate_min_mask_coverage = float(
        baselines.get("full_sam31_tracked_mask_support_gate_min_mask_coverage", 0.05)
    )
    tracked_mask_support_gate_min_support_coverage = float(
        baselines.get("full_sam31_tracked_mask_support_gate_min_support_coverage", 0.20)
    )
    tracked_mask_support_gate_max_area_expand = float(
        baselines.get("full_sam31_tracked_mask_support_gate_max_area_expand", 4.0)
    )
    tracked_mask_support_gate_require_geometry = bool(
        baselines.get("full_sam31_tracked_mask_support_gate_require_geometry", False)
    )
    tracked_mask_support_gate_write_full_support = bool(
        baselines.get("full_sam31_tracked_mask_support_gate_write_full_support", False)
    )
    keyframe_proposal_direct_write_enabled = bool(
        baselines.get("full_sam31_keyframe_proposal_direct_write_enabled", False)
    )
    keyframe_proposal_prewrite_all_enabled = bool(
        baselines.get("full_sam31_keyframe_proposal_prewrite_all_enabled", False)
    )

    query_count = 0
    candidate_count = 0
    output_mask_count = 0
    tracked_object_count = 0
    failed_object_count = 0
    redundant_object_count = 0
    frame_output_count = 0
    sam31_session_count = 0
    capped_by_query_limit = False
    reused_anchor_track_count = 0
    reused_tracked_output_track_count = 0
    reused_temporal_track_count = 0
    temporal_track_merge_count = 0
    temporal_local_relabel_count = 0
    temporal_local_relabel_pixel_count = 0
    new_anchor_track_count = 0
    prompt_registry_reuse_count = 0
    prompt_registry_reserved_new_count = 0
    prompt_preassigned_override_count = 0
    track_registry_output_update_count = 0
    track_registry_geometry_match_count = 0
    tracked_mask_support_gate_applied_count = 0
    tracked_mask_support_gate_rejected_count = 0
    tracked_mask_support_gate_raw_pixel_count = 0
    tracked_mask_support_gate_clipped_pixel_count = 0
    keyframe_proposal_direct_write_count = 0
    keyframe_proposal_direct_write_pixel_count = 0
    keyframe_proposal_prewrite_count = 0
    keyframe_proposal_prewrite_pixel_count = 0
    proposal_support_preloaded_frame_count = 0
    proposal_support_preloaded_mask_count = 0
    candidate_reject_counts: dict[str, int] = {}
    objectness_rejected_candidate_count = 0
    batch_failure_reasons: list[str] = []
    zero_output_frame_ids: list[int] = []
    missing_proposal_frame_ids: list[int] = []
    per_frame_candidate_stats: list[dict[str, Any]] = []
    track_rows: list[dict[str, Any]] = []
    temporal_track_merge_rows: list[dict[str, Any]] = []
    id_stabilization_summary: dict[str, Any] = {"enabled": False}
    scope_note = (
        "SAM3.1 multiplex 4DPM-style gap video tracking: every stride input frame is scanned as an anchor; "
        "valid proposal regions initialize SAM3.1 mask-seeded object prompts even when already covered by a prior "
        "track, so anchor frames can correct identity drift; each "
        "prompt is propagated through the full window with start_session/mask_seed_internal/propagate_in_video."
    )

    predictor = None
    frontend = None
    frame_resource: str | None = None
    try:
        import torch  # type: ignore

        if not torch.cuda.is_available():
            raise RuntimeError("SAM3.1 gap video tracking baseline requires CUDA")
        if not frame_ids:
            raise RuntimeError(f"{variant_id} received an empty frame list")

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        first_image = _preprocessed_rgb_for_sam2(ctx, scene_id, int(frame_ids[0]))
        h, w = first_image.shape[:2]
        frame_area = float(max(h * w, 1))
        labels_by_local: dict[int, np.ndarray] = {int(idx): np.zeros((h, w), dtype=np.uint16) for idx in range(len(frame_ids))}
        proposal_supports_by_local: dict[int, list[dict[str, Any]]] = {int(idx): [] for idx in range(len(frame_ids))}
        preloaded_proposal_labels_by_local: dict[int, np.ndarray] = {}
        geometry_depth_by_local: dict[int, np.ndarray | None] = {}
        geometry_intrinsics_by_local: dict[int, np.ndarray | None] = {}
        geometry_pose_by_local: dict[int, np.ndarray | None] = {}
        if track_registry_geometry_enabled:
            for geom_local_idx, geom_frame_id in enumerate(frame_ids):
                geometry_depth_by_local[int(geom_local_idx)] = _preprocessed_depth_for_sgq(ctx, scene_id, int(geom_frame_id))
                geometry_intrinsics_by_local[int(geom_local_idx)] = _preprocessed_intrinsics_for_sgq(ctx, scene_id, int(geom_frame_id))
                geometry_pose_by_local[int(geom_local_idx)] = _pose_for_sgq(ctx, scene_id, int(geom_frame_id))
        proposal_supports_preloaded = bool(tracked_mask_support_gate_enabled)
        if proposal_supports_preloaded:
            for support_local_idx, support_frame_id in enumerate(frame_ids):
                support_path = _proposal_label_path_for_baseline(ctx, scene_id, int(support_frame_id), proposal_source)
                support_label = _preprocessed_label_for_sgq(ctx, scene_id, int(support_frame_id), support_path)
                if support_label is None:
                    continue
                if support_label.shape[:2] != (h, w):
                    support_label = cv2.resize(support_label, (w, h), interpolation=cv2.INTER_NEAREST)
                preloaded_proposal_labels_by_local[int(support_local_idx)] = support_label
                frame_support_count = 0
                for support_proposal_id in [int(value) for value in np.unique(support_label) if int(value) > 0]:
                    support_mask = support_label == int(support_proposal_id)
                    support_gate = _candidate_objectness_gate(
                        support_mask,
                        min_pixels=min_pixels,
                        min_area_ratio=min_area_ratio,
                        max_area_ratio=max_area_ratio,
                        broad_area_ratio=objectness_broad_area_ratio,
                        max_bbox_area_ratio=objectness_max_bbox_area_ratio,
                        max_span_ratio=objectness_max_span_ratio,
                        max_edge_pixel_ratio=objectness_max_edge_pixel_ratio,
                        max_edge_touch_sides=objectness_max_edge_touch_sides,
                        enabled=objectness_gate_enabled,
                    )
                    if not bool(support_gate["accepted"]):
                        continue
                    support_world_points = (
                        _mask_world_points_from_depth(
                            support_mask,
                            depth_mm=geometry_depth_by_local.get(int(support_local_idx)),
                            intrinsics=geometry_intrinsics_by_local.get(int(support_local_idx)),
                            pose_c2w=geometry_pose_by_local.get(int(support_local_idx)),
                            max_points=track_registry_geometry_max_points,
                        )
                        if track_registry_geometry_enabled
                        else None
                    )
                    proposal_supports_by_local.setdefault(int(support_local_idx), []).append(
                        {
                            "proposal_mask_id": int(support_proposal_id),
                            "mask": support_mask.astype(bool),
                            "pixel_area": int(support_gate["pixel_area"]),
                            "area_ratio": float(support_gate["area_ratio"]),
                            "world_points": support_world_points,
                        }
                    )
                    frame_support_count += 1
                if frame_support_count > 0:
                    proposal_support_preloaded_frame_count += 1
                    proposal_support_preloaded_mask_count += int(frame_support_count)
        frame_resource, _local_index_by_frame = _write_sam31_frame_resource(ctx, scene_id, [int(frame_id) for frame_id in frame_ids])
        predictor, frontend = _build_sam31_multiplex_frontend(ctx, max_num_objects=internal_cap, device="cuda:0")
        next_track_id = 1
        track_id_alias: dict[int, int] = {}
        track_registry: dict[int, dict[str, Any]] = {}

        def _canonical_track_id(track_id: int) -> int:
            track_id = int(track_id)
            seen: set[int] = set()
            while track_id > 0 and track_id in track_id_alias and track_id not in seen:
                seen.add(track_id)
                track_id = int(track_id_alias[track_id])
            return int(track_id)

        def _temporal_cross_id_match_is_strong(match: dict[str, Any] | None) -> bool:
            if not match:
                return False
            return (
                float(match.get("mask_iou", 0.0) or 0.0) >= reuse_temporal_id_cross_min_mask_iou
                and float(match.get("bbox_iou", 0.0) or 0.0) >= reuse_temporal_id_cross_min_bbox_iou
                and float(match.get("area_score", 0.0) or 0.0) >= reuse_temporal_id_cross_min_area_score
            )

        def _normalise_tracked_mask(mask: np.ndarray) -> np.ndarray:
            mask_np = np.asarray(mask)
            if mask_np.ndim == 3:
                if mask_np.shape[0] == 1:
                    mask_np = mask_np[0]
                elif mask_np.shape[-1] == 1:
                    mask_np = mask_np[..., 0]
                else:
                    mask_np = np.max(mask_np, axis=-1 if mask_np.shape[:2] == (h, w) else 0)
            mask_bool = mask_np.astype(bool)
            if mask_bool.shape[:2] != (h, w):
                mask_bool = cv2.resize(mask_bool.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
            return mask_bool

        def _world_points_for_local_mask(mask: np.ndarray, local_frame_idx: int) -> np.ndarray | None:
            if not track_registry_geometry_enabled:
                return None
            local_frame_idx = int(local_frame_idx)
            return _mask_world_points_from_depth(
                mask,
                depth_mm=geometry_depth_by_local.get(local_frame_idx),
                intrinsics=geometry_intrinsics_by_local.get(local_frame_idx),
                pose_c2w=geometry_pose_by_local.get(local_frame_idx),
                max_points=track_registry_geometry_max_points,
            )

        def _remember_prompt_state(track_id: int, det: dict[str, Any]) -> None:
            local_frame_idx = int(det.get("_v105_local_frame_idx", det.get("frame_idx", 0)))
            frame_id = int(det.get("_v105_frame_id", -1))
            identity_mask = np.asarray(det.get("_v105_identity_mask", det["mask"])).astype(bool)
            world_points = _world_points_for_local_mask(identity_mask, local_frame_idx)
            _append_mask_track_state(
                track_registry,
                int(track_id),
                identity_mask,
                min_pixels=min_pixels,
                local_frame_idx=local_frame_idx,
                frame_id=frame_id,
                max_history=track_registry_history_max_entries,
                world_points=world_points,
            )

        def _preassign_prompt_global_track(det: dict[str, Any]) -> None:
            nonlocal next_track_id, prompt_registry_reuse_count, prompt_registry_reserved_new_count
            nonlocal track_registry_geometry_match_count
            if int(det.get("_v105_assigned_track_id", 0) or 0) > 0:
                return
            local_frame_idx = int(det.get("_v105_local_frame_idx", det.get("frame_idx", 0)))
            frame_id = int(det.get("_v105_frame_id", -1))
            proposal_mask = np.asarray(det.get("_v105_identity_mask", det["mask"])).astype(bool)
            anchor_track_id, anchor_iou, anchor_coverage = _best_overlap_with_label(
                proposal_mask,
                labels_by_local.get(local_frame_idx),
            )
            proposal_world_points = _world_points_for_local_mask(proposal_mask, local_frame_idx)
            registry_track_id, registry_match = _best_track_registry_match(
                proposal_mask,
                track_registry,
                local_frame_idx=local_frame_idx,
                min_pixels=min_pixels,
                lookback=track_registry_lookback,
                min_score=track_registry_min_score,
                world_points=proposal_world_points,
                geometry_near_threshold_m=track_registry_geometry_near_threshold_m,
                geometry_centroid_threshold_m=track_registry_geometry_centroid_threshold_m,
                geometry_min_score=track_registry_geometry_min_score,
            )
            registry_track_id = _canonical_track_id(registry_track_id)
            if (
                reuse_existing_id_enabled
                and not keyframe_proposal_prewrite_all_enabled
                and anchor_track_id > 0
                and (
                    anchor_iou >= reuse_existing_id_min_iou
                    or anchor_coverage >= reuse_existing_id_min_coverage
                )
            ):
                assigned_track_id = _canonical_track_id(anchor_track_id)
                assignment_kind = "reused_anchor_label_overlap"
                prompt_registry_reuse_count += 1
                registry_match = registry_match or {}
            elif registry_track_id > 0:
                assigned_track_id = int(registry_track_id)
                assignment_kind = "reused_track_registry_continuity"
                prompt_registry_reuse_count += 1
                if bool((registry_match or {}).get("geometry_gate", False)):
                    track_registry_geometry_match_count += 1
            else:
                assigned_track_id = int(next_track_id)
                next_track_id += 1
                assignment_kind = "new_anchor_prompt_registry_reserved"
                prompt_registry_reserved_new_count += 1
            det["_v105_assigned_track_id"] = int(assigned_track_id)
            det["_v105_assignment_kind"] = assignment_kind
            det["_v105_registry_anchor_track_id"] = int(anchor_track_id)
            det["_v105_registry_anchor_iou"] = float(anchor_iou)
            det["_v105_registry_anchor_coverage"] = float(anchor_coverage)
            det["_v105_registry_match_track_id"] = int(registry_track_id)
            det["_v105_registry_match"] = registry_match or {}
            _remember_prompt_state(int(assigned_track_id), det)

        def _write_keyframe_proposal(det: dict[str, Any]) -> bool:
            nonlocal output_mask_count
            nonlocal keyframe_proposal_direct_write_count, keyframe_proposal_direct_write_pixel_count
            nonlocal keyframe_proposal_prewrite_count, keyframe_proposal_prewrite_pixel_count
            if not keyframe_proposal_direct_write_enabled:
                return False
            if bool(det.get("_v105_keyframe_proposal_written", False)):
                return False
            track_id = _canonical_track_id(int(det.get("_v105_assigned_track_id", 0) or 0))
            if track_id <= 0:
                return False
            local_frame_idx = int(det.get("_v105_local_frame_idx", det.get("frame_idx", 0)))
            if local_frame_idx not in labels_by_local:
                return False
            proposal_mask = np.asarray(det.get("_v105_identity_mask", det["mask"])).astype(bool)
            if proposal_mask.shape[:2] != (h, w):
                proposal_mask = cv2.resize(proposal_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
            label = labels_by_local[local_frame_idx]
            writable = (label == 0) | (label == int(track_id))
            region = proposal_mask & writable
            pixel_count = int(np.count_nonzero(region))
            if pixel_count < min_pixels:
                return False
            label[region] = int(track_id)
            labels_by_local[local_frame_idx] = label
            det["_v105_keyframe_proposal_written"] = True
            keyframe_proposal_direct_write_count += 1
            keyframe_proposal_direct_write_pixel_count += int(pixel_count)
            if keyframe_proposal_prewrite_all_enabled:
                keyframe_proposal_prewrite_count += 1
                keyframe_proposal_prewrite_pixel_count += int(pixel_count)
            output_mask_count += 1
            return True

        def _sort_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
            if candidate_order in {"small_to_large", "small_to_large_uncovered", "small_first"}:
                return sorted(rows, key=lambda row: (float(row["area_ratio"]), int(row["pixel_area"]), int(row["proposal_mask_id"])))
            if candidate_order in {"large_to_small", "large_first"}:
                return sorted(rows, key=lambda row: (-float(row["area_ratio"]), -int(row["pixel_area"]), int(row["proposal_mask_id"])))
            return sorted(rows, key=lambda row: (abs(float(row["area_ratio"]) - objectness_target_area_ratio), float(row["area_ratio"]), int(row["proposal_mask_id"])))

        def _best_overlap_with_existing_labels(frame_to_mask: dict[int, np.ndarray]) -> tuple[int, float, float]:
            intersection_by_track: dict[int, int] = {}
            mask_area_by_track: dict[int, int] = {}
            union_by_track: dict[int, int] = {}
            for local_frame_idx, mask in frame_to_mask.items():
                local_frame_idx = int(local_frame_idx)
                if local_frame_idx not in labels_by_local:
                    continue
                mask_bool = _normalise_tracked_mask(mask)
                if not mask_bool.any():
                    continue
                label = labels_by_local[local_frame_idx]
                mask_area = int(np.count_nonzero(mask_bool))
                for track_id in [int(value) for value in np.unique(label[mask_bool]) if int(value) > 0]:
                    track_mask = label == int(track_id)
                    inter = int(np.count_nonzero(mask_bool & track_mask))
                    if inter <= 0:
                        continue
                    union = int(np.count_nonzero(mask_bool | track_mask))
                    intersection_by_track[track_id] = intersection_by_track.get(track_id, 0) + inter
                    mask_area_by_track[track_id] = mask_area_by_track.get(track_id, 0) + mask_area
                    union_by_track[track_id] = union_by_track.get(track_id, 0) + union
            best_track_id = 0
            best_iou = 0.0
            best_coverage = 0.0
            for track_id, inter in intersection_by_track.items():
                union = float(max(union_by_track.get(track_id, 0), 1))
                mask_area = float(max(mask_area_by_track.get(track_id, 0), 1))
                iou = float(inter) / union
                coverage = float(inter) / mask_area
                if (iou, coverage) > (best_iou, best_coverage):
                    best_track_id = _canonical_track_id(int(track_id))
                    best_iou = iou
                    best_coverage = coverage
            return best_track_id, best_iou, best_coverage

        def _mask_summary(mask: np.ndarray) -> dict[str, Any] | None:
            mask_bool = _normalise_tracked_mask(mask)
            area = int(np.count_nonzero(mask_bool))
            if area < min_pixels:
                return None
            bbox = _bbox_xyxy(mask_bool)
            if bbox is None:
                return None
            return {
                "mask": mask_bool,
                "area": int(area),
                "bbox": [int(v) for v in bbox],
                "center": (0.5 * (float(bbox[0]) + float(bbox[2])), 0.5 * (float(bbox[1]) + float(bbox[3]))),
            }

        def _temporal_continuity_matches(frame_to_mask: dict[int, np.ndarray]) -> dict[int, dict[str, Any]]:
            matches: dict[int, dict[str, Any]] = {}
            diag = float(max(np.hypot(float(h), float(w)), 1.0))
            for local_frame_idx, mask in sorted(frame_to_mask.items()):
                local_frame_idx = int(local_frame_idx)
                current = _mask_summary(mask)
                if current is None:
                    continue
                current_mask = current["mask"]
                current_bbox = current["bbox"]
                current_area = float(max(int(current["area"]), 1))
                current_center = current["center"]
                for delta in range(1, int(reuse_temporal_id_lookback) + 1):
                    prev_idx = local_frame_idx - delta
                    if prev_idx < 0 or prev_idx not in labels_by_local:
                        continue
                    prev_label = labels_by_local[prev_idx]
                    for raw_track_id in [int(value) for value in np.unique(prev_label) if int(value) > 0]:
                        track_id = _canonical_track_id(raw_track_id)
                        prev_mask = prev_label == int(track_id)
                        prev_area_i = int(np.count_nonzero(prev_mask))
                        if prev_area_i < min_pixels:
                            continue
                        prev_bbox = _bbox_xyxy(prev_mask)
                        if prev_bbox is None:
                            continue
                        inter = float(np.count_nonzero(current_mask & prev_mask))
                        union = float(np.count_nonzero(current_mask | prev_mask))
                        mask_iou = inter / union if union > 0.0 else 0.0
                        coverage = inter / current_area if current_area > 0.0 else 0.0
                        box_iou = _bbox_iou_xyxy(current_bbox, prev_bbox)
                        prev_area = float(max(prev_area_i, 1))
                        area_score = min(current_area, prev_area) / max(current_area, prev_area)
                        pcx = 0.5 * (float(prev_bbox[0]) + float(prev_bbox[2]))
                        pcy = 0.5 * (float(prev_bbox[1]) + float(prev_bbox[3]))
                        center_score = max(
                            0.0,
                            1.0
                            - float(np.hypot(float(current_center[0]) - pcx, float(current_center[1]) - pcy))
                            / (0.45 * diag),
                        )
                        continuity_gate = (
                            mask_iou >= 0.03
                            or coverage >= 0.20
                            or (box_iou >= 0.20 and center_score >= 0.55 and area_score >= 0.35)
                        )
                        if not continuity_gate:
                            continue
                        score = (
                            0.35 * float(mask_iou)
                            + 0.15 * float(coverage)
                            + 0.25 * float(box_iou)
                            + 0.15 * float(area_score)
                            + 0.10 * float(center_score)
                        )
                        score /= 1.0 + 0.10 * float(delta - 1)
                        prev = matches.get(track_id)
                        if prev is None or score > float(prev["score"]):
                            matches[track_id] = {
                                "track_id": int(track_id),
                                "score": float(score),
                                "mask_iou": float(mask_iou),
                                "coverage": float(coverage),
                                "bbox_iou": float(box_iou),
                                "area_score": float(area_score),
                                "center_score": float(center_score),
                                "source_local_frame_idx": int(local_frame_idx),
                                "matched_local_frame_idx": int(prev_idx),
                                "lookback_delta": int(delta),
                            }
            return matches

        def _track_batch(batch: list[dict[str, Any]]) -> None:
            nonlocal output_mask_count, tracked_object_count, failed_object_count, redundant_object_count
            nonlocal sam31_session_count, next_track_id
            nonlocal reused_anchor_track_count, reused_tracked_output_track_count, reused_temporal_track_count
            nonlocal temporal_local_relabel_count, temporal_local_relabel_pixel_count
            nonlocal new_anchor_track_count, prompt_preassigned_override_count
            nonlocal track_registry_output_update_count
            nonlocal tracked_mask_support_gate_applied_count, tracked_mask_support_gate_rejected_count
            nonlocal tracked_mask_support_gate_raw_pixel_count, tracked_mask_support_gate_clipped_pixel_count
            session_id: str | None = None
            try:
                tracker_batch = sorted(
                    batch,
                    key=lambda row: (
                        int(row.get("_v105_proposal_pixel_area", row.get("pixel_area", row.get("prompt_pixel_area", 0))) or 0),
                        int(row.get("local_frame_idx", 0) or 0),
                        int(row.get("proposal_mask_id", 0) or 0),
                    ),
                )
                session_id = (
                    frontend._start_sam31_session(frame_resource)
                    if hasattr(frontend, "_start_sam31_session")
                    else frontend.video_predictor.handle_request({"type": "start_session", "resource_path": frame_resource})["session_id"]
                )
                sam31_session_count += 1
                obj_segments, obj_scores, obj_to_det = frontend._track_objects_once_sam31_multiplex_session(
                    session_id=session_id,
                    detections=tracker_batch,
                    H=h,
                    W=w,
                )
                for obj_id in sorted(obj_to_det.keys()):
                    det = obj_to_det[int(obj_id)]
                    preassigned_track_id = _canonical_track_id(int(det.get("_v105_assigned_track_id", 0) or 0))
                    preassignment_kind = str(det.get("_v105_assignment_kind", ""))
                    reuse_track_id = _canonical_track_id(int(det.get("_v105_reuse_track_id", 0) or 0))
                    tracked_overlap_id, tracked_overlap_iou, tracked_overlap_coverage = _best_overlap_with_existing_labels(
                        obj_segments.get(int(obj_id), {})
                    )
                    tracked_overlap_id = _canonical_track_id(tracked_overlap_id)
                    temporal_matches = (
                        _temporal_continuity_matches(obj_segments.get(int(obj_id), {}))
                        if reuse_temporal_id_enabled
                        else {}
                    )
                    temporal_best = max(temporal_matches.values(), key=lambda row: float(row["score"]), default=None)
                    temporal_best_id = int(temporal_best["track_id"]) if temporal_best is not None else 0
                    temporal_best_score = float(temporal_best["score"]) if temporal_best is not None else 0.0
                    direct_track_id = int(reuse_track_id or tracked_overlap_id)
                    direct_temporal_score = float(temporal_matches.get(direct_track_id, {}).get("score", 0.0)) if direct_track_id > 0 else 0.0
                    temporal_cross_id_strong = _temporal_cross_id_match_is_strong(temporal_best)
                    temporal_same_direct = temporal_best_id > 0 and temporal_best_id == direct_track_id
                    temporal_no_direct = temporal_best_id > 0 and direct_track_id <= 0
                    temporal_cross_direct = temporal_best_id > 0 and direct_track_id > 0 and temporal_best_id != direct_track_id
                    temporal_can_override = (
                        temporal_best_id > 0
                        and temporal_best_score >= reuse_temporal_id_min_score
                        and (
                            temporal_same_direct
                            or temporal_no_direct
                            or (
                                temporal_cross_direct
                                and temporal_cross_id_strong
                                and temporal_best_score >= direct_temporal_score + reuse_temporal_id_min_margin
                            )
                        )
                    )
                    temporal_relabel_source_track_id = 0
                    if keyframe_proposal_prewrite_all_enabled and preassigned_track_id > 0:
                        assigned_track_id = int(preassigned_track_id)
                        assignment_kind = preassignment_kind or "new_anchor_prompt_registry_reserved"
                        if assignment_kind == "reused_anchor_label_overlap":
                            reused_anchor_track_count += 1
                        elif assignment_kind == "reused_track_registry_continuity":
                            reused_temporal_track_count += 1
                        else:
                            new_anchor_track_count += 1
                    elif preassigned_track_id > 0 and not preassignment_kind.startswith("new_"):
                        assigned_track_id = int(preassigned_track_id)
                        assignment_kind = preassignment_kind
                        if assignment_kind == "reused_anchor_label_overlap":
                            reused_anchor_track_count += 1
                        elif assignment_kind == "reused_track_registry_continuity":
                            reused_temporal_track_count += 1
                    elif temporal_can_override:
                        assigned_track_id = int(temporal_best_id)
                        assignment_kind = "reused_temporal_continuity"
                        if (
                            reuse_temporal_id_local_relabel_enabled
                            and direct_track_id > 0
                            and direct_track_id != assigned_track_id
                        ):
                            temporal_relabel_source_track_id = int(direct_track_id)
                            assignment_kind = "reused_temporal_continuity_local_relabel"
                        reused_temporal_track_count += 1
                    elif reuse_track_id > 0:
                        assigned_track_id = reuse_track_id
                        assignment_kind = "reused_anchor_overlap"
                        reused_anchor_track_count += 1
                    elif (
                        reuse_tracked_output_id_enabled
                        and tracked_overlap_id > 0
                        and (
                            tracked_overlap_iou >= reuse_tracked_output_id_min_iou
                            or tracked_overlap_coverage >= reuse_tracked_output_id_min_coverage
                        )
                    ):
                        assigned_track_id = int(tracked_overlap_id)
                        assignment_kind = "reused_tracked_output_overlap"
                        reused_tracked_output_track_count += 1
                    elif preassigned_track_id > 0:
                        assigned_track_id = int(preassigned_track_id)
                        assignment_kind = preassignment_kind or "new_anchor_prompt_registry_reserved"
                        new_anchor_track_count += 1
                    else:
                        assigned_track_id = int(next_track_id)
                        next_track_id += 1
                        assignment_kind = "new_anchor_prompt"
                        new_anchor_track_count += 1
                    if preassigned_track_id > 0 and preassigned_track_id != assigned_track_id:
                        track_id_alias[int(preassigned_track_id)] = int(assigned_track_id)
                        track_registry.pop(int(preassigned_track_id), None)
                        prompt_preassigned_override_count += 1
                    _remember_prompt_state(int(assigned_track_id), det)
                    raw_segment_frame_count = 0
                    raw_segment_pixel_count = 0
                    written_frames = 0
                    written_pixels = 0
                    local_relabel_pixels = 0
                    support_gate_written_frames = 0
                    support_gate_rejected_frames = 0
                    support_gate_raw_pixels = 0
                    support_gate_clipped_pixels = 0
                    for local_frame_idx, mask in sorted(obj_segments.get(int(obj_id), {}).items()):
                        local_frame_idx = int(local_frame_idx)
                        if local_frame_idx not in labels_by_local:
                            continue
                        mask_bool = _normalise_tracked_mask(mask)
                        raw_pixels = int(np.count_nonzero(mask_bool))
                        if raw_pixels > 0:
                            raw_segment_frame_count += 1
                            raw_segment_pixel_count += raw_pixels
                        if tracked_mask_support_gate_enabled:
                            support_mask, support_info = _best_proposal_support_match(
                                mask_bool,
                                proposal_supports_by_local.get(local_frame_idx, []),
                                prompt_area=int(
                                    det.get("_v105_proposal_pixel_area", det.get("pixel_area", det.get("prompt_pixel_area", raw_pixels)))
                                    or raw_pixels
                                ),
                                min_pixels=min_pixels,
                                min_score=tracked_mask_support_gate_min_score,
                                min_iou=tracked_mask_support_gate_min_iou,
                                min_mask_coverage=tracked_mask_support_gate_min_mask_coverage,
                                min_support_coverage=tracked_mask_support_gate_min_support_coverage,
                                max_area_expand=tracked_mask_support_gate_max_area_expand,
                                track_state=track_registry.get(int(assigned_track_id)),
                                require_geometry=tracked_mask_support_gate_require_geometry,
                                geometry_near_threshold_m=track_registry_geometry_near_threshold_m,
                                geometry_centroid_threshold_m=track_registry_geometry_centroid_threshold_m,
                                geometry_min_score=track_registry_geometry_min_score,
                                write_full_support=tracked_mask_support_gate_write_full_support,
                            )
                            if support_mask is None:
                                tracked_mask_support_gate_rejected_count += 1
                                support_gate_rejected_frames += 1
                                continue
                            clipped_pixels = int(np.count_nonzero(support_mask))
                            tracked_mask_support_gate_applied_count += 1
                            tracked_mask_support_gate_raw_pixel_count += int(raw_pixels)
                            tracked_mask_support_gate_clipped_pixel_count += int(clipped_pixels)
                            support_gate_written_frames += 1
                            support_gate_raw_pixels += int(raw_pixels)
                            support_gate_clipped_pixels += int(clipped_pixels)
                            mask_bool = support_mask
                        label = labels_by_local[local_frame_idx]
                        if not assignment_kind.startswith("new_"):
                            writable = (label == 0) | (label == int(assigned_track_id))
                            if temporal_relabel_source_track_id > 0:
                                writable = writable | (label == int(temporal_relabel_source_track_id))
                            region = mask_bool & writable
                        else:
                            region = mask_bool & (label == 0)
                        pixel_count = int(np.count_nonzero(region))
                        if pixel_count < min_pixels:
                            continue
                        if temporal_relabel_source_track_id > 0:
                            local_relabel_pixels += int(np.count_nonzero(region & (label == int(temporal_relabel_source_track_id))))
                        label[region] = int(assigned_track_id)
                        labels_by_local[local_frame_idx] = label
                        if track_registry_update_from_outputs_enabled:
                            world_points = _world_points_for_local_mask(mask_bool, local_frame_idx)
                            if _append_mask_track_state(
                                track_registry,
                                int(assigned_track_id),
                                mask_bool,
                                min_pixels=min_pixels,
                                local_frame_idx=local_frame_idx,
                                frame_id=int(frame_ids[local_frame_idx]),
                                max_history=track_registry_history_max_entries,
                                world_points=world_points,
                            ):
                                track_registry_output_update_count += 1
                        output_mask_count += 1
                        written_frames += 1
                        written_pixels += pixel_count
                    if written_frames > 0:
                        tracked_object_count += 1
                        if local_relabel_pixels > 0:
                            temporal_local_relabel_count += 1
                            temporal_local_relabel_pixel_count += int(local_relabel_pixels)
                    elif raw_segment_frame_count > 0:
                        redundant_object_count += 1
                    else:
                        failed_object_count += 1
                    track_rows.append(
                        {
                            "track_id": int(assigned_track_id),
                            "prompt_policy": "sam31_gap_video_tracking_uncovered_proposal",
                            "prompt_local_frame_idx": int(det.get("_v105_local_frame_idx", det.get("frame_idx", 0))),
                            "prompt_frame_id": int(det.get("_v105_frame_id", -1)),
                            "proposal_mask_id": int(det.get("proposal_mask_id", det.get("_v105_proposal_mask_id", -1))),
                            "proposal_source": str(det.get("proposal_source", proposal_source)),
                            "prompt_mask_policy": str(det.get("prompt_mask_policy", "")),
                            "prompt_pixel_area": int(det.get("prompt_pixel_area", 0) or 0),
                            "identity_pixel_area": int(det.get("_v105_identity_pixel_area", 0) or 0),
                            "uncovered_pixels_before_prompt": int(det.get("uncovered_pixels", 0) or 0),
                            "uncovered_fraction_before_prompt": float(det.get("uncovered_fraction", 0.0) or 0.0),
                            "track_id_assignment_kind": assignment_kind,
                            "preassigned_track_id": int(preassigned_track_id),
                            "preassignment_kind": preassignment_kind,
                            "anchor_overlap_reuse_track_id": int(reuse_track_id),
                            "anchor_overlap_best_track_id": int(det.get("_v105_anchor_overlap_best_track_id", 0) or 0),
                            "anchor_overlap_best_iou": float(det.get("_v105_anchor_overlap_best_iou", 0.0) or 0.0),
                            "anchor_overlap_best_coverage": float(det.get("_v105_anchor_overlap_best_coverage", 0.0) or 0.0),
                            "registry_anchor_track_id": int(det.get("_v105_registry_anchor_track_id", 0) or 0),
                            "registry_anchor_iou": float(det.get("_v105_registry_anchor_iou", 0.0) or 0.0),
                            "registry_anchor_coverage": float(det.get("_v105_registry_anchor_coverage", 0.0) or 0.0),
                            "registry_match_track_id": int(det.get("_v105_registry_match_track_id", 0) or 0),
                            "registry_match": det.get("_v105_registry_match", {}),
                            "tracked_output_overlap_reuse_track_id": int(tracked_overlap_id),
                            "tracked_output_overlap_best_iou": float(tracked_overlap_iou),
                            "tracked_output_overlap_best_coverage": float(tracked_overlap_coverage),
                            "temporal_continuity_reuse_track_id": int(temporal_best_id),
                            "temporal_continuity_best_score": float(temporal_best_score),
                            "temporal_continuity_direct_track_id": int(direct_track_id),
                            "temporal_continuity_direct_score": float(direct_temporal_score),
                            "temporal_continuity_cross_id_strong": bool(temporal_cross_id_strong),
                            "temporal_relabel_source_track_id": int(temporal_relabel_source_track_id),
                            "temporal_local_relabel_pixel_count": int(local_relabel_pixels),
                            "temporal_merge_source_track_id": 0,
                            "temporal_continuity_best_match": temporal_best or {},
                            "sam31_obj_id": int(obj_id),
                            "sam31_obj_score": float(obj_scores.get(int(obj_id), 0.0)),
                            "sam31_prompt_type": str(det.get("_sam31_prompt_type", "")),
                            "sam31_mask_prompt_error": str(det.get("_sam31_mask_prompt_error", "")),
                            "raw_sam31_segment_frame_count": int(raw_segment_frame_count),
                            "raw_sam31_segment_pixel_count": int(raw_segment_pixel_count),
                            "tracked_mask_support_gate_written_frame_count": int(support_gate_written_frames),
                            "tracked_mask_support_gate_rejected_frame_count": int(support_gate_rejected_frames),
                            "tracked_mask_support_gate_raw_pixel_count": int(support_gate_raw_pixels),
                            "tracked_mask_support_gate_clipped_pixel_count": int(support_gate_clipped_pixels),
                            "written_frame_count": int(written_frames),
                            "written_pixel_count": int(written_pixels),
                        }
                    )
            finally:
                if session_id is not None:
                    try:
                        frontend.video_predictor.handle_request({"type": "close_session", "session_id": session_id, "run_gc_collect": True})
                    except Exception:
                        pass

        for local_idx, frame_id in enumerate(frame_ids):
            proposal_label = preloaded_proposal_labels_by_local.get(int(local_idx))
            if proposal_label is None:
                proposal_path = _proposal_label_path_for_baseline(ctx, scene_id, int(frame_id), proposal_source)
                proposal_label = _preprocessed_label_for_sgq(ctx, scene_id, int(frame_id), proposal_path)
            if proposal_label is None:
                missing_proposal_frame_ids.append(int(frame_id))
                per_frame_candidate_stats.append(
                    {
                        "frame_id": int(frame_id),
                        "valid_candidate_count": 0,
                        "prompted_candidate_count": 0,
                        "covered_candidate_skip_count": 0,
                        "missing_proposal": True,
                    }
                )
                continue
            if proposal_label.shape[:2] != (h, w):
                proposal_label = cv2.resize(proposal_label, (w, h), interpolation=cv2.INTER_NEAREST)

            if keyframe_proposal_prewrite_all_enabled:
                labels_by_local[int(local_idx)] = np.zeros((h, w), dtype=np.uint16)
            current_label = labels_by_local[int(local_idx)]
            proposal_ids = [int(value) for value in np.unique(proposal_label) if int(value) > 0]
            frame_candidates: list[dict[str, Any]] = []
            covered_skips = 0
            covered_prompt_candidates = 0
            reused_candidates = 0
            for proposal_id in proposal_ids:
                proposal_mask = proposal_label == int(proposal_id)
                gate = _candidate_objectness_gate(
                    proposal_mask,
                    min_pixels=min_pixels,
                    min_area_ratio=min_area_ratio,
                    max_area_ratio=max_area_ratio,
                    broad_area_ratio=objectness_broad_area_ratio,
                    max_bbox_area_ratio=objectness_max_bbox_area_ratio,
                    max_span_ratio=objectness_max_span_ratio,
                    max_edge_pixel_ratio=objectness_max_edge_pixel_ratio,
                    max_edge_touch_sides=objectness_max_edge_touch_sides,
                    enabled=objectness_gate_enabled,
                )
                if not bool(gate["accepted"]):
                    objectness_rejected_candidate_count += 1
                    for reason in gate["reject_reasons"]:
                        candidate_reject_counts[str(reason)] = candidate_reject_counts.get(str(reason), 0) + 1
                    continue
                pixel_area = int(gate["pixel_area"])
                area_ratio = float(gate["area_ratio"])
                bbox = gate["bbox"]
                candidate_count += 1
                if not proposal_supports_preloaded:
                    support_world_points = _world_points_for_local_mask(proposal_mask, int(local_idx))
                    proposal_supports_by_local.setdefault(int(local_idx), []).append(
                        {
                            "proposal_mask_id": int(proposal_id),
                            "mask": proposal_mask.astype(bool),
                            "pixel_area": int(pixel_area),
                            "area_ratio": float(area_ratio),
                            "world_points": support_world_points,
                        }
                    )
                uncovered_pixels, uncovered_fraction, uncovered_mask = _candidate_uncovered_stats(
                    proposal_mask,
                    current_label,
                )
                is_covered_candidate = (
                    uncovered_pixels < min_uncovered_pixels
                    or uncovered_fraction < min_uncovered_fraction
                )
                if is_covered_candidate and not (prompt_covered_candidates_enabled or keyframe_proposal_prewrite_all_enabled):
                    covered_skips += 1
                    continue
                prompt_mask = proposal_mask
                prompt_mask_policy = "full_proposal"
                if (not keyframe_proposal_prewrite_all_enabled) and 0 < uncovered_pixels < pixel_area:
                    prompt_mask = uncovered_mask
                    prompt_mask_policy = "uncovered_region_only"
                prompt_pixel_area = int(np.count_nonzero(prompt_mask))
                if prompt_pixel_area < min_pixels:
                    covered_skips += 1
                    continue
                prompt_bbox = _bbox_xyxy(prompt_mask)
                if prompt_bbox is None:
                    covered_skips += 1
                    continue
                overlap_track_id, overlap_iou, overlap_coverage = _best_overlap_with_label(prompt_mask, current_label)
                reuse_track_id = 0
                if (
                    reuse_existing_id_enabled
                    and overlap_track_id > 0
                    and (
                        overlap_iou >= reuse_existing_id_min_iou
                        or overlap_coverage >= reuse_existing_id_min_coverage
                    )
                ):
                    reuse_track_id = int(overlap_track_id)
                    reused_candidates += 1
                if is_covered_candidate:
                    covered_prompt_candidates += 1
                frame_candidates.append(
                    {
                        "frame_idx": int(local_idx),
                        "mask": prompt_mask.astype(np.uint8),
                        "box": np.asarray(prompt_bbox, dtype=np.float32),
                        "label": f"{proposal_source}_object",
                        "raw_label": f"{proposal_source}_mask_{proposal_id}",
                        "proposal_source": proposal_source,
                        "proposal_mask_id": int(proposal_id),
                        "confidence": float(min(0.99, max(0.05, area_ratio * 5.0))),
                        "area_ratio": area_ratio,
                        "pixel_area": int(pixel_area),
                        "prompt_pixel_area": int(prompt_pixel_area),
                        "uncovered_pixels": int(uncovered_pixels),
                        "uncovered_fraction": float(uncovered_fraction),
                        "covered_anchor_candidate": bool(is_covered_candidate),
                        "prompt_mask_policy": prompt_mask_policy,
                        "objectness_bbox_area_ratio": float(gate["bbox_area_ratio"]),
                        "objectness_fill_ratio": float(gate["fill_ratio"]),
                        "objectness_span_w": float(gate["span_w"]),
                        "objectness_span_h": float(gate["span_h"]),
                        "objectness_edge_touch_sides": int(gate["edge_touch_sides"]),
                        "objectness_edge_pixel_ratio": float(gate["edge_pixel_ratio"]),
                        "sem_group": 1,
                        "_v105_reuse_track_id": int(reuse_track_id),
                        "_v105_anchor_overlap_best_track_id": int(overlap_track_id),
                        "_v105_anchor_overlap_best_iou": float(overlap_iou),
                        "_v105_anchor_overlap_best_coverage": float(overlap_coverage),
                        "_v105_scene_id": scene_id,
                        "_v105_frame_id": int(frame_id),
                        "_v105_local_frame_idx": int(local_idx),
                        "_v105_proposal_mask_id": int(proposal_id),
                        "_v105_identity_mask": proposal_mask.astype(bool),
                        "_v105_identity_pixel_area": int(pixel_area),
                        "_v105_proposal_pixel_area": int(pixel_area),
                    }
                )

            frame_candidates = _sort_candidates(frame_candidates)
            selected = frame_candidates[:max_candidates_per_frame]
            if query_count + len(selected) > max_queries:
                selected = selected[: max(0, max_queries - query_count)]
                capped_by_query_limit = True
            prompted = len(selected)
            query_count += prompted
            for det in selected:
                _preassign_prompt_global_track(det)
            for det in sorted(
                selected,
                key=lambda row: (
                    int(row.get("_v105_identity_pixel_area", row.get("pixel_area", 0)) or 0),
                    int(row.get("_v105_local_frame_idx", row.get("frame_idx", 0)) or 0),
                    int(row.get("proposal_mask_id", 0) or 0),
                ),
            ):
                _write_keyframe_proposal(det)
            for start in range(0, len(selected), objects_per_session):
                batch = selected[start : start + objects_per_session]
                try:
                    _track_batch(batch)
                except RuntimeError as exc:
                    reason = _safe_error(exc)
                    if "out of memory" in reason.lower() and len(batch) > 1:
                        mid = max(1, len(batch) // 2)
                        for sub_batch in (batch[:mid], batch[mid:]):
                            try:
                                _track_batch(sub_batch)
                            except Exception as sub_exc:
                                failed_object_count += len(sub_batch)
                                batch_failure_reasons.append(_safe_error(sub_exc))
                        torch.cuda.empty_cache()
                    else:
                        failed_object_count += len(batch)
                        batch_failure_reasons.append(reason)
                        torch.cuda.empty_cache()
                except Exception as exc:
                    failed_object_count += len(batch)
                    batch_failure_reasons.append(_safe_error(exc))
                    torch.cuda.empty_cache()

            per_frame_candidate_stats.append(
                {
                    "frame_id": int(frame_id),
                    "valid_candidate_count": len(frame_candidates) + covered_skips,
                    "prompted_candidate_count": prompted,
                    "covered_candidate_skip_count": covered_skips,
                    "covered_candidate_prompt_count": covered_prompt_candidates,
                    "reused_anchor_candidate_count": reused_candidates,
                    "objectness_rejected_candidate_count": int(objectness_rejected_candidate_count),
                    "candidate_reject_counts": dict(candidate_reject_counts),
                    "capped_by_query_limit_after_frame": bool(capped_by_query_limit),
                    "nonzero_after_tracking": int(np.count_nonzero(labels_by_local[int(local_idx)])) > 0,
                    "missing_proposal": False,
                }
            )
            if capped_by_query_limit:
                break

        for local_idx, frame_id in enumerate(frame_ids):
            label = labels_by_local[int(local_idx)]
            if int(np.count_nonzero(label)) == 0:
                zero_output_frame_ids.append(int(frame_id))
            cv2.imwrite(str(mask_dir / f"{int(frame_id)}.png"), label)
            frame_output_count += 1

        if id_stabilization_enabled:
            id_stabilization_summary = _stabilize_label_ids_temporally(
                mask_dir=mask_dir,
                frame_ids=[int(frame_id) for frame_id in frame_ids],
                min_pixels=min_pixels,
                lookback=id_stabilization_lookback,
                min_match_score=id_stabilization_min_score,
                ctx=ctx,
                scene_id=scene_id,
                geometry_enabled=track_registry_geometry_enabled,
                geometry_max_points=track_registry_geometry_max_points,
                geometry_near_threshold_m=track_registry_geometry_near_threshold_m,
                geometry_centroid_threshold_m=track_registry_geometry_centroid_threshold_m,
                geometry_min_score=track_registry_geometry_min_score,
            )

        torch.cuda.synchronize()
        peak_gpu_mb = float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)
        nonzero_output_frame_count = len(frame_ids) - len(zero_output_frame_ids)
        if output_mask_count <= 0:
            status = "failed"
        elif batch_failure_reasons or capped_by_query_limit or failed_object_count > 0 or zero_output_frame_ids:
            status = "partial"
        else:
            status = "completed"
        if batch_failure_reasons:
            failure_reason = "; ".join(batch_failure_reasons)
        elif capped_by_query_limit:
            failure_reason = "query_limit_capped_gap_prompts"
        elif failed_object_count > 0:
            failure_reason = "one or more SAM3.1 gap object tracks produced no valid masks"
        elif zero_output_frame_ids:
            failure_reason = "zero_output_input_frames=" + ";".join(str(frame_id) for frame_id in zero_output_frame_ids)
        else:
            failure_reason = ""

        track_count = int(id_stabilization_summary.get("track_count", max(next_track_id - 1, 0)))
        return {
            "schema_version": "stream4d_v105_full_sam31_runtime_row_v1",
            "variant_id": variant_id,
            "proposal_source": proposal_source,
            "status": status,
            "scene_id": scene_id,
            "frame_ids": ";".join(str(int(frame_id)) for frame_id in frame_ids),
            "frame_count": len(frame_ids),
            "frame_output_count": frame_output_count,
            "candidate_count": candidate_count,
            "query_count": query_count,
            "query_limit": max_queries,
            "capped_by_query_limit": capped_by_query_limit,
            "output_mask_count": output_mask_count,
            "nonzero_output_frame_count": nonzero_output_frame_count,
            "zero_output_frame_ids": zero_output_frame_ids,
            "all_input_frames_have_nonzero_masks": len(zero_output_frame_ids) == 0,
            "tracked_object_count": track_count,
            "failed_object_count": failed_object_count,
            "redundant_object_count": int(redundant_object_count),
            "sam31_video_session_count": sam31_session_count,
            "sam31_oom_retry_count": 0,
            "objects_per_session": objects_per_session,
            "max_internal_objects": internal_cap,
            "anchor_frame_stride": 1,
            "max_candidates_per_anchor_frame": max_candidates_per_frame,
            "candidate_order": candidate_order,
            "candidate_objectness_gate_enabled": bool(objectness_gate_enabled),
            "candidate_objectness_broad_area_ratio": float(objectness_broad_area_ratio),
            "candidate_objectness_max_bbox_area_ratio": float(objectness_max_bbox_area_ratio),
            "candidate_objectness_max_span_ratio": float(objectness_max_span_ratio),
            "candidate_objectness_max_edge_pixel_ratio": float(objectness_max_edge_pixel_ratio),
            "candidate_objectness_max_edge_touch_sides": int(objectness_max_edge_touch_sides),
            "candidate_objectness_target_area_ratio": float(objectness_target_area_ratio),
            "objectness_rejected_candidate_count": int(objectness_rejected_candidate_count),
            "candidate_reject_counts": dict(candidate_reject_counts),
            "min_uncovered_candidate_pixels": min_uncovered_pixels,
            "min_uncovered_candidate_fraction": float(min_uncovered_fraction),
            "prompt_covered_anchor_candidates_enabled": bool(prompt_covered_candidates_enabled),
            "reuse_existing_anchor_id_enabled": bool(reuse_existing_id_enabled),
            "reuse_existing_anchor_id_min_iou": reuse_existing_id_min_iou,
            "reuse_existing_anchor_id_min_coverage": reuse_existing_id_min_coverage,
            "reuse_existing_anchor_id_gate": "iou_or_coverage",
            "reused_anchor_track_count": int(reused_anchor_track_count),
            "reuse_existing_tracked_output_id_enabled": bool(reuse_tracked_output_id_enabled),
            "reuse_existing_tracked_output_id_min_iou": reuse_tracked_output_id_min_iou,
            "reuse_existing_tracked_output_id_min_coverage": reuse_tracked_output_id_min_coverage,
            "reused_tracked_output_track_count": int(reused_tracked_output_track_count),
            "reuse_existing_temporal_id_enabled": bool(reuse_temporal_id_enabled),
            "reuse_existing_temporal_id_lookback": int(reuse_temporal_id_lookback),
            "reuse_existing_temporal_id_min_score": float(reuse_temporal_id_min_score),
            "reuse_existing_temporal_id_min_margin": float(reuse_temporal_id_min_margin),
            "reuse_existing_temporal_id_merge_enabled": bool(reuse_temporal_id_merge_enabled),
            "reuse_existing_temporal_id_merge_semantics": "disabled_global_history_rewrite",
            "reuse_existing_temporal_id_local_relabel_enabled": bool(reuse_temporal_id_local_relabel_enabled),
            "reuse_existing_temporal_id_cross_min_mask_iou": float(reuse_temporal_id_cross_min_mask_iou),
            "reuse_existing_temporal_id_cross_min_bbox_iou": float(reuse_temporal_id_cross_min_bbox_iou),
            "reuse_existing_temporal_id_cross_min_area_score": float(reuse_temporal_id_cross_min_area_score),
            "reused_temporal_track_count": int(reused_temporal_track_count),
            "temporal_track_merge_count": int(temporal_track_merge_count),
            "temporal_track_merge_rows": temporal_track_merge_rows,
            "temporal_local_relabel_count": int(temporal_local_relabel_count),
            "temporal_local_relabel_pixel_count": int(temporal_local_relabel_pixel_count),
            "new_anchor_track_count": int(new_anchor_track_count),
            "track_registry_enabled": True,
            "track_registry_semantics": "prompt_to_global_track_binding_no_mask_fusion",
            "track_registry_lookback": int(track_registry_lookback),
            "track_registry_min_score": float(track_registry_min_score),
            "track_registry_update_from_outputs_enabled": bool(track_registry_update_from_outputs_enabled),
            "track_registry_history_max_entries": int(track_registry_history_max_entries),
            "track_registry_output_update_count": int(track_registry_output_update_count),
            "track_registry_geometry_enabled": bool(track_registry_geometry_enabled),
            "track_registry_geometry_max_points": int(track_registry_geometry_max_points),
            "track_registry_geometry_near_threshold_m": float(track_registry_geometry_near_threshold_m),
            "track_registry_geometry_centroid_threshold_m": float(track_registry_geometry_centroid_threshold_m),
            "track_registry_geometry_min_score": float(track_registry_geometry_min_score),
            "track_registry_geometry_match_count": int(track_registry_geometry_match_count),
            "prompt_registry_reuse_count": int(prompt_registry_reuse_count),
            "prompt_registry_reserved_new_count": int(prompt_registry_reserved_new_count),
            "prompt_preassigned_override_count": int(prompt_preassigned_override_count),
            "tracked_mask_support_gate_enabled": bool(tracked_mask_support_gate_enabled),
            "tracked_mask_support_gate_min_score": float(tracked_mask_support_gate_min_score),
            "tracked_mask_support_gate_min_iou": float(tracked_mask_support_gate_min_iou),
            "tracked_mask_support_gate_min_mask_coverage": float(tracked_mask_support_gate_min_mask_coverage),
            "tracked_mask_support_gate_min_support_coverage": float(tracked_mask_support_gate_min_support_coverage),
            "tracked_mask_support_gate_max_area_expand": float(tracked_mask_support_gate_max_area_expand),
            "tracked_mask_support_gate_require_geometry": bool(tracked_mask_support_gate_require_geometry),
            "tracked_mask_support_gate_write_full_support": bool(tracked_mask_support_gate_write_full_support),
            "tracker_prompt_write_order": "small_to_large_identity_area",
            "keyframe_proposal_direct_write_enabled": bool(keyframe_proposal_direct_write_enabled),
            "keyframe_proposal_prewrite_all_enabled": bool(keyframe_proposal_prewrite_all_enabled),
            "keyframe_proposal_direct_write_count": int(keyframe_proposal_direct_write_count),
            "keyframe_proposal_direct_write_pixel_count": int(keyframe_proposal_direct_write_pixel_count),
            "keyframe_proposal_prewrite_count": int(keyframe_proposal_prewrite_count),
            "keyframe_proposal_prewrite_pixel_count": int(keyframe_proposal_prewrite_pixel_count),
            "tracked_mask_support_gate_applied_count": int(tracked_mask_support_gate_applied_count),
            "tracked_mask_support_gate_rejected_count": int(tracked_mask_support_gate_rejected_count),
            "tracked_mask_support_gate_raw_pixel_count": int(tracked_mask_support_gate_raw_pixel_count),
            "tracked_mask_support_gate_clipped_pixel_count": int(tracked_mask_support_gate_clipped_pixel_count),
            "proposal_support_preloaded_frame_count": int(proposal_support_preloaded_frame_count),
            "proposal_support_preloaded_mask_count": int(proposal_support_preloaded_mask_count),
            "each_input_frame_treated_as_anchor": True,
            "no_internal_keyframe_subsampling": True,
            "tracking_mode": "sam31_gap_video_tracking",
            "track_rows": track_rows,
            "id_stabilization_enabled": bool(id_stabilization_enabled),
            "id_stabilization_summary": id_stabilization_summary,
            "per_frame_candidate_stats": per_frame_candidate_stats,
            "missing_proposal_frame_ids": sorted(set(missing_proposal_frame_ids)),
            "latency_sec": time.time() - t0,
            "peak_gpu_memory_mb": peak_gpu_mb,
            "checkpoint": _rel(checkpoint),
            "checkpoint_sha256": _sha256_file(checkpoint),
            "failure_reason": failure_reason,
            "reference_scope": scope_note,
            "eligible_for_phase4_speed_gate": status == "completed",
        }
    except Exception as exc:
        reason = _safe_error(exc)
        return {
            "schema_version": "stream4d_v105_full_sam31_runtime_row_v1",
            "variant_id": variant_id,
            "proposal_source": proposal_source,
            "status": "failed",
            "failure_type": "OOM" if "out of memory" in reason.lower() else "runtime_error",
            "scene_id": scene_id,
            "frame_ids": ";".join(str(int(frame_id)) for frame_id in frame_ids),
            "frame_count": len(frame_ids),
            "frame_output_count": frame_output_count,
            "candidate_count": candidate_count,
            "query_count": query_count,
            "query_limit": max_queries,
            "capped_by_query_limit": capped_by_query_limit,
            "output_mask_count": output_mask_count,
            "tracked_object_count": 0,
            "failed_object_count": failed_object_count,
            "sam31_video_session_count": sam31_session_count,
            "latency_sec": time.time() - t0,
            "peak_gpu_memory_mb": peak_gpu_mb,
            "checkpoint": _rel(checkpoint),
            "checkpoint_sha256": _sha256_file(checkpoint),
            "failure_reason": reason,
            "reference_scope": scope_note,
            "eligible_for_phase4_speed_gate": False,
        }
    finally:
        if predictor is not None and hasattr(predictor, "shutdown"):
            try:
                predictor.shutdown()
            except Exception:
                pass
        if frame_resource and Path(frame_resource).exists():
            shutil.rmtree(frame_resource, ignore_errors=True)
        gc.collect()
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def _run_full_sam31_reference_scene(
    ctx: PipelineContext,
    scene_id: str,
    frame_ids: list[int],
    mask_dir: Path,
    *,
    variant_id: str = "B3_full_sam31_tracking",
    proposal_source: str = "cropformer",
) -> dict[str, Any]:
    t0 = time.time()
    peak_gpu_mb: float | str = ""
    mask_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = _as_repo_path(ctx.config.get("paths", {}).get("sam31_checkpoint"))
    baselines = ctx.config.get("baselines", {})
    max_queries = int(baselines.get("full_sam31_max_queries_per_scene", 96))
    min_pixels = int(baselines.get("full_sam31_min_mask_pixels", 16))
    min_area_ratio = float(baselines.get("full_sam31_min_area_ratio", 0.0002))
    max_area_ratio = float(baselines.get("full_sam31_max_area_ratio", 0.85))
    anchor_frame_stride = max(int(baselines.get("full_sam31_anchor_frame_stride", 4)), 1)
    max_candidates_per_anchor_frame = max(int(baselines.get("full_sam31_max_candidates_per_anchor_frame", 8)), 1)
    internal_cap = max(int(baselines.get("full_sam31_max_internal_objects", 16)), 1)
    objects_per_session = max(int(baselines.get("full_sam31_objects_per_session", 8)), 1)
    objects_per_session = min(objects_per_session, internal_cap)
    query_count = 0
    candidate_count = 0
    output_mask_count = 0
    frame_output_count = 0
    nonzero_output_frame_count = 0
    zero_output_frame_ids: list[int] = []
    tracked_object_count = 0
    failed_object_count = 0
    sam31_session_count = 0
    oom_retry_count = 0
    capped_by_query_limit = False
    batch_failure_reasons: list[str] = []
    track_rows: list[dict[str, Any]] = []
    scope_note = "SAM3.1 multiplex video object-prompt tracking via VideoMaskletFrontend._track_objects_once_sam31_multiplex_session; uses start_session/add_prompt(obj_id)/propagate_in_video"
    if checkpoint is None or not checkpoint.exists():
        return {
            "schema_version": "stream4d_v105_full_sam31_runtime_row_v1",
            "variant_id": variant_id,
            "proposal_source": proposal_source,
            "status": "failed",
            "scene_id": scene_id,
            "frame_ids": ";".join(str(int(frame_id)) for frame_id in frame_ids),
            "frame_count": len(frame_ids),
            "frame_output_count": 0,
            "candidate_count": 0,
            "query_count": 0,
            "query_limit": max_queries,
            "capped_by_query_limit": False,
            "output_mask_count": 0,
            "latency_sec": time.time() - t0,
            "peak_gpu_memory_mb": peak_gpu_mb,
            "checkpoint": _rel(checkpoint) if checkpoint else "",
            "checkpoint_sha256": "",
            "failure_reason": "SAM3.1 checkpoint missing",
            "reference_scope": scope_note,
            "eligible_for_phase4_speed_gate": False,
        }
    if bool(baselines.get("full_sam31_dense_keyframe_prompt", True)):
        return _run_dense_sam31_keyframe_scene(
            ctx,
            scene_id,
            frame_ids,
            mask_dir,
            variant_id=variant_id,
            proposal_source=proposal_source,
            checkpoint=checkpoint,
        )
    if bool(baselines.get("full_sam31_gap_tracking_enabled", True)):
        return _run_sam31_gap_tracking_reference_scene(
            ctx,
            scene_id,
            frame_ids,
            mask_dir,
            variant_id=variant_id,
            proposal_source=proposal_source,
            checkpoint=checkpoint,
        )
    predictor = None
    frontend = None
    frame_resource: str | None = None
    try:
        import torch  # type: ignore

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if not torch.cuda.is_available():
            return {
                "schema_version": "stream4d_v105_full_sam31_runtime_row_v1",
                "variant_id": variant_id,
                "proposal_source": proposal_source,
                "status": "failed",
                "failure_type": "runtime_error",
                "scene_id": scene_id,
                "frame_ids": ";".join(str(int(frame_id)) for frame_id in frame_ids),
                "frame_count": len(frame_ids),
                "frame_output_count": 0,
                "candidate_count": 0,
                "query_count": 0,
                "query_limit": max_queries,
                "capped_by_query_limit": False,
                "output_mask_count": 0,
                "tracked_object_count": 0,
                "failed_object_count": 0,
                "sam31_video_session_count": 0,
                "objects_per_session": objects_per_session,
                "max_internal_objects": internal_cap,
                "latency_sec": time.time() - t0,
                "peak_gpu_memory_mb": peak_gpu_mb,
                "checkpoint": _rel(checkpoint),
                "checkpoint_sha256": _sha256_file(checkpoint),
                "failure_reason": "SAM3.1 multiplex baseline requires CUDA",
                "reference_scope": scope_note,
                "eligible_for_phase4_speed_gate": False,
            }
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        if not frame_ids:
            raise RuntimeError(f"{variant_id} SAM3.1 baseline received an empty frame list")
        first_image = _preprocessed_rgb_for_sam2(ctx, scene_id, int(frame_ids[0]))
        h, w = first_image.shape[:2]
        frame_area = float(max(h * w, 1))
        anchor_indices = list(range(0, len(frame_ids), anchor_frame_stride))
        if len(frame_ids) > 0 and (len(frame_ids) - 1) not in anchor_indices:
            anchor_indices.append(len(frame_ids) - 1)
        detections: list[dict[str, Any]] = []
        for local_idx in anchor_indices:
            frame_id = int(frame_ids[int(local_idx)])
            proposal_path = _proposal_label_path_for_baseline(ctx, scene_id, frame_id, proposal_source)
            proposal_label = _preprocessed_label_for_sgq(ctx, scene_id, frame_id, proposal_path)
            if proposal_label is None:
                continue
            if proposal_label.shape[:2] != (h, w):
                proposal_label = cv2.resize(proposal_label, (w, h), interpolation=cv2.INTER_NEAREST)
            proposal_ids = [int(value) for value in np.unique(proposal_label) if int(value) > 0]
            proposal_ids.sort(key=lambda value: int(np.count_nonzero(proposal_label == value)), reverse=True)
            per_anchor_count = 0
            for proposal_id in proposal_ids:
                proposal_mask = proposal_label == proposal_id
                pixel_area = int(np.count_nonzero(proposal_mask))
                area_ratio = float(pixel_area) / frame_area
                if pixel_area < min_pixels or area_ratio < min_area_ratio or area_ratio > max_area_ratio:
                    continue
                bbox = _bbox_xyxy(proposal_mask)
                if bbox is None:
                    continue
                candidate_count += 1
                detections.append(
                    {
                        "frame_idx": int(local_idx),
                        "mask": proposal_mask.astype(np.uint8),
                        "box": np.asarray(bbox, dtype=np.float32),
                        "label": f"{proposal_source}_object",
                        "raw_label": f"{proposal_source}_mask_{proposal_id}",
                        "confidence": float(min(0.99, max(0.05, area_ratio * 5.0))),
                        "area_ratio": area_ratio,
                        "sem_group": 1,
                        "_v105_scene_id": scene_id,
                        "_v105_frame_id": frame_id,
                        "_v105_local_frame_idx": int(local_idx),
                        "_v105_proposal_mask_id": int(proposal_id),
                        "_v105_proposal_pixel_area": pixel_area,
                    }
                )
                per_anchor_count += 1
                if per_anchor_count >= max_candidates_per_anchor_frame:
                    break
        if not detections:
            for frame_id in frame_ids:
                cv2.imwrite(str(mask_dir / f"{int(frame_id)}.png"), np.zeros((h, w), dtype=np.uint16))
            frame_output_count = len(frame_ids)
            zero_output_frame_ids = [int(frame_id) for frame_id in frame_ids]
            return {
                "schema_version": "stream4d_v105_full_sam31_runtime_row_v1",
                "variant_id": variant_id,
                "proposal_source": proposal_source,
                "status": "failed",
                "failure_type": "no_candidates",
                "scene_id": scene_id,
                "frame_ids": ";".join(str(int(frame_id)) for frame_id in frame_ids),
                "frame_count": len(frame_ids),
                "frame_output_count": frame_output_count,
                "candidate_count": 0,
                "query_count": 0,
                "query_limit": max_queries,
                "capped_by_query_limit": False,
                "output_mask_count": 0,
                "nonzero_output_frame_count": 0,
                "zero_output_frame_ids": zero_output_frame_ids,
                "all_input_frames_have_nonzero_masks": False,
                "tracked_object_count": 0,
                "failed_object_count": 0,
                "sam31_video_session_count": 0,
                "objects_per_session": objects_per_session,
                "max_internal_objects": internal_cap,
                "latency_sec": time.time() - t0,
                "peak_gpu_memory_mb": peak_gpu_mb,
                "checkpoint": _rel(checkpoint),
                "checkpoint_sha256": _sha256_file(checkpoint),
                "failure_reason": f"no valid {proposal_source} proposal anchors for SAM3.1 object-prompt tracking",
                "reference_scope": scope_note,
                "eligible_for_phase4_speed_gate": False,
            }
        capped_by_query_limit = len(detections) > max_queries
        selected_detections = detections[:max_queries]
        query_count = len(selected_detections)
        frame_resource, _local_index_by_frame = _write_sam31_frame_resource(ctx, scene_id, [int(frame_id) for frame_id in frame_ids])
        predictor, frontend = _build_sam31_multiplex_frontend(ctx, max_num_objects=internal_cap, device=device)
        labels_by_local: dict[int, np.ndarray] = {int(idx): np.zeros((h, w), dtype=np.uint16) for idx in range(len(frame_ids))}
        next_track_id = 1

        def _normalise_tracked_mask(mask: np.ndarray) -> np.ndarray:
            mask_np = np.asarray(mask)
            if mask_np.ndim == 3:
                if mask_np.shape[0] == 1:
                    mask_np = mask_np[0]
                else:
                    mask_np = np.max(mask_np, axis=0)
            mask_bool = mask_np.astype(bool)
            if mask_bool.shape[:2] != (h, w):
                mask_bool = cv2.resize(mask_bool.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
            return mask_bool

        def _run_one_batch(batch: list[dict[str, Any]]) -> None:
            nonlocal output_mask_count, tracked_object_count, failed_object_count, sam31_session_count, next_track_id
            session_id: str | None = None
            try:
                session_id = frontend._start_sam31_session(frame_resource) if hasattr(frontend, "_start_sam31_session") else frontend.video_predictor.handle_request({"type": "start_session", "resource_path": frame_resource})["session_id"]
                sam31_session_count += 1
                obj_segments, obj_scores, obj_to_det = frontend._track_objects_once_sam31_multiplex_session(
                    session_id=session_id,
                    detections=batch,
                    H=h,
                    W=w,
                )
                for obj_id in sorted(obj_to_det.keys()):
                    det = obj_to_det[int(obj_id)]
                    frame_to_mask = obj_segments.get(int(obj_id), {})
                    assigned_track_id = int(next_track_id)
                    next_track_id += 1
                    written_frames = 0
                    written_pixels = 0
                    for local_frame_idx, mask in sorted(frame_to_mask.items()):
                        local_frame_idx = int(local_frame_idx)
                        if local_frame_idx not in labels_by_local:
                            continue
                        mask_bool = _normalise_tracked_mask(mask)
                        label = labels_by_local[local_frame_idx]
                        region = mask_bool & (label == 0)
                        pixel_count = int(np.count_nonzero(region))
                        if pixel_count < min_pixels:
                            continue
                        label[region] = assigned_track_id
                        labels_by_local[local_frame_idx] = label
                        output_mask_count += 1
                        written_frames += 1
                        written_pixels += pixel_count
                    if written_frames > 0:
                        tracked_object_count += 1
                    else:
                        failed_object_count += 1
                    track_rows.append(
                        {
                            "track_id": assigned_track_id,
                            "prompt_local_frame_idx": int(det.get("_v105_local_frame_idx", det.get("frame_idx", 0))),
                            "prompt_frame_id": int(det.get("_v105_frame_id", -1)),
                            "proposal_mask_id": int(det.get("_v105_proposal_mask_id", -1)),
                            "sam31_obj_id": int(obj_id),
                            "sam31_obj_score": float(obj_scores.get(int(obj_id), 0.0)),
                            "sam31_prompt_type": str(det.get("_sam31_prompt_type", "")),
                            "sam31_mask_prompt_error": str(det.get("_sam31_mask_prompt_error", "")),
                            "written_frame_count": int(written_frames),
                            "written_pixel_count": int(written_pixels),
                        }
                    )
            finally:
                if session_id is not None:
                    try:
                        frontend.video_predictor.handle_request({"type": "close_session", "session_id": session_id, "run_gc_collect": True})
                    except Exception:
                        pass

        def _track_batch(batch: list[dict[str, Any]]) -> None:
            nonlocal oom_retry_count, failed_object_count
            try:
                _run_one_batch(batch)
            except RuntimeError as exc:
                reason = _safe_error(exc)
                if "out of memory" in reason.lower() and len(batch) > 1:
                    oom_retry_count += 1
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    mid = max(1, len(batch) // 2)
                    _track_batch(batch[:mid])
                    _track_batch(batch[mid:])
                    return
                failed_object_count += len(batch)
                batch_failure_reasons.append(reason)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as exc:
                failed_object_count += len(batch)
                batch_failure_reasons.append(_safe_error(exc))

        for start in range(0, len(selected_detections), objects_per_session):
            _track_batch(selected_detections[start : start + objects_per_session])
        for local_idx, frame_id in enumerate(frame_ids):
            label = labels_by_local[int(local_idx)]
            if int(np.count_nonzero(label)) > 0:
                nonzero_output_frame_count += 1
            else:
                zero_output_frame_ids.append(int(frame_id))
            cv2.imwrite(str(mask_dir / f"{int(frame_id)}.png"), label)
            frame_output_count += 1
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            peak_gpu_mb = float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)
        if output_mask_count <= 0:
            status = "failed"
        elif batch_failure_reasons or capped_by_query_limit or failed_object_count > 0 or zero_output_frame_ids:
            status = "partial"
        else:
            status = "completed"
        eligible = status == "completed" and not capped_by_query_limit and output_mask_count > 0
        if batch_failure_reasons:
            failure_reason = "; ".join(batch_failure_reasons)
        elif capped_by_query_limit:
            failure_reason = "query_limit_capped_object_prompts"
        elif failed_object_count > 0:
            failure_reason = "one or more SAM3.1 object tracks produced no valid masks"
        elif zero_output_frame_ids:
            failure_reason = "zero_output_input_frames=" + ";".join(str(frame_id) for frame_id in zero_output_frame_ids)
        else:
            failure_reason = ""
        return {
            "schema_version": "stream4d_v105_full_sam31_runtime_row_v1",
            "variant_id": variant_id,
            "proposal_source": proposal_source,
            "status": status,
            "scene_id": scene_id,
            "frame_ids": ";".join(str(int(frame_id)) for frame_id in frame_ids),
            "frame_count": len(frame_ids),
            "frame_output_count": frame_output_count,
            "candidate_count": candidate_count,
            "query_count": query_count,
            "query_limit": max_queries,
            "capped_by_query_limit": capped_by_query_limit,
            "output_mask_count": output_mask_count,
            "nonzero_output_frame_count": nonzero_output_frame_count,
            "zero_output_frame_ids": zero_output_frame_ids,
            "all_input_frames_have_nonzero_masks": len(zero_output_frame_ids) == 0,
            "tracked_object_count": tracked_object_count,
            "failed_object_count": failed_object_count,
            "sam31_video_session_count": sam31_session_count,
            "sam31_oom_retry_count": oom_retry_count,
            "objects_per_session": objects_per_session,
            "max_internal_objects": internal_cap,
            "anchor_frame_stride": anchor_frame_stride,
            "max_candidates_per_anchor_frame": max_candidates_per_anchor_frame,
            "track_rows": track_rows,
            "latency_sec": time.time() - t0,
            "peak_gpu_memory_mb": peak_gpu_mb,
            "checkpoint": _rel(checkpoint),
            "checkpoint_sha256": _sha256_file(checkpoint),
            "failure_reason": failure_reason,
            "reference_scope": scope_note,
            "eligible_for_phase4_speed_gate": eligible,
        }
    except RuntimeError as exc:
        reason = _safe_error(exc)
        return {
            "schema_version": "stream4d_v105_full_sam31_runtime_row_v1",
            "variant_id": variant_id,
            "proposal_source": proposal_source,
            "status": "failed",
            "failure_type": "OOM" if "out of memory" in reason.lower() else "runtime_error",
            "scene_id": scene_id,
            "frame_ids": ";".join(str(int(frame_id)) for frame_id in frame_ids),
            "frame_count": len(frame_ids),
            "frame_output_count": frame_output_count,
            "candidate_count": candidate_count,
            "query_count": query_count,
            "query_limit": max_queries,
            "capped_by_query_limit": capped_by_query_limit,
            "output_mask_count": output_mask_count,
            "tracked_object_count": tracked_object_count,
            "failed_object_count": failed_object_count,
            "sam31_video_session_count": sam31_session_count,
            "sam31_oom_retry_count": oom_retry_count,
            "objects_per_session": objects_per_session,
            "max_internal_objects": internal_cap,
            "latency_sec": time.time() - t0,
            "peak_gpu_memory_mb": peak_gpu_mb,
            "checkpoint": _rel(checkpoint),
            "checkpoint_sha256": _sha256_file(checkpoint),
            "failure_reason": reason,
            "reference_scope": scope_note,
            "eligible_for_phase4_speed_gate": False,
        }
    except Exception as exc:
        return {
            "schema_version": "stream4d_v105_full_sam31_runtime_row_v1",
            "variant_id": variant_id,
            "proposal_source": proposal_source,
            "status": "failed",
            "failure_type": "runtime_error",
            "scene_id": scene_id,
            "frame_ids": ";".join(str(int(frame_id)) for frame_id in frame_ids),
            "frame_count": len(frame_ids),
            "frame_output_count": frame_output_count,
            "candidate_count": candidate_count,
            "query_count": query_count,
            "query_limit": max_queries,
            "capped_by_query_limit": capped_by_query_limit,
            "output_mask_count": output_mask_count,
            "tracked_object_count": tracked_object_count,
            "failed_object_count": failed_object_count,
            "sam31_video_session_count": sam31_session_count,
            "sam31_oom_retry_count": oom_retry_count,
            "objects_per_session": objects_per_session,
            "max_internal_objects": internal_cap,
            "latency_sec": time.time() - t0,
            "peak_gpu_memory_mb": peak_gpu_mb,
            "checkpoint": _rel(checkpoint),
            "checkpoint_sha256": _sha256_file(checkpoint),
            "failure_reason": _safe_error(exc),
            "reference_scope": scope_note,
            "eligible_for_phase4_speed_gate": False,
        }
    finally:
        if predictor is not None and hasattr(predictor, "close"):
            predictor.close()
        if predictor is not None and hasattr(predictor, "shutdown"):
            try:
                predictor.shutdown()
            except Exception:
                pass
        if frame_resource and Path(frame_resource).exists():
            shutil.rmtree(frame_resource, ignore_errors=True)
        gc.collect()
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def _build_mask_prompt_video_predictor(ctx: PipelineContext, tracker_provider: str, device: str) -> Any:
    provider = str(tracker_provider).strip().lower()
    if provider == "sam2_video":
        root = REPO_ROOT / "Grounded-SAM-2"
        checkpoint = _as_repo_path(ctx.config.get("paths", {}).get("sam2_checkpoint"))
        model_cfg = ctx.config.get("paths", {}).get("sam2_model_cfg")
        if checkpoint is None or not checkpoint.exists():
            raise FileNotFoundError(f"missing SAM2 checkpoint: {checkpoint}")
        old_sys_path = list(sys.path)
        try:
            for name in list(sys.modules):
                if name == "sam2" or name.startswith("sam2."):
                    del sys.modules[name]
            sys.path.insert(0, str(root))
            from sam2.build_sam import build_sam2_video_predictor  # type: ignore

            return build_sam2_video_predictor(str(model_cfg), str(checkpoint), device=device)
        finally:
            sys.path[:] = old_sys_path
    if provider == "edgetam_video":
        root = REPO_ROOT / "third_party" / "EdgeTAM"
        checkpoint = _as_repo_path(ctx.config.get("paths", {}).get("edgetam_checkpoint"))
        if checkpoint is None or not checkpoint.exists():
            raise FileNotFoundError(f"missing EdgeTAM checkpoint: {checkpoint}")
        old_cwd = Path.cwd()
        old_sys_path = list(sys.path)
        try:
            os.chdir(root)
            for name in list(sys.modules):
                if name == "sam2" or name.startswith("sam2."):
                    del sys.modules[name]
            sys.path.insert(0, str(root))
            try:
                from hydra.core.global_hydra import GlobalHydra  # type: ignore

                if GlobalHydra.instance().is_initialized():
                    GlobalHydra.instance().clear()
            except Exception:
                pass
            from sam2.build_sam import build_sam2_video_predictor  # type: ignore

            return build_sam2_video_predictor("edgetam.yaml", str(checkpoint), device=device)
        finally:
            os.chdir(old_cwd)
            sys.path[:] = old_sys_path
    raise ValueError(f"unsupported mask-prompt video tracker provider={tracker_provider}")


def _run_mask_prompt_video_tracker_scene(
    ctx: PipelineContext,
    scene_id: str,
    frame_ids: list[int],
    mask_dir: Path,
    *,
    variant_id: str,
    proposal_source: str,
    tracker_provider: str,
) -> dict[str, Any]:
    t0 = time.time()
    peak_gpu_mb: float | str = ""
    mask_dir.mkdir(parents=True, exist_ok=True)
    for stale_mask in mask_dir.glob("*.png"):
        stale_mask.unlink()
    baselines = ctx.config.get("baselines", {})
    min_pixels = int(baselines.get("baseline_matrix_min_mask_pixels", 16))
    min_area_ratio = float(baselines.get("baseline_matrix_min_area_ratio", 0.0002))
    max_area_ratio = float(baselines.get("baseline_matrix_max_area_ratio", 1.0))
    objectness_gate_enabled = bool(baselines.get("baseline_matrix_candidate_objectness_gate_enabled", False))
    objectness_broad_area_ratio = float(baselines.get("baseline_matrix_objectness_broad_area_ratio", 0.08))
    objectness_max_bbox_area_ratio = float(baselines.get("baseline_matrix_objectness_max_bbox_area_ratio", 0.70))
    objectness_max_span_ratio = float(baselines.get("baseline_matrix_objectness_max_span_ratio", 0.90))
    objectness_max_edge_pixel_ratio = float(baselines.get("baseline_matrix_objectness_max_edge_pixel_ratio", 0.08))
    objectness_max_edge_touch_sides = int(baselines.get("baseline_matrix_objectness_max_edge_touch_sides", 1))
    objectness_target_area_ratio = float(baselines.get("baseline_matrix_objectness_target_area_ratio", 0.035))
    max_candidates_per_frame = max(int(baselines.get("baseline_matrix_max_candidates_per_anchor_frame", 96)), 1)
    max_queries = max(int(baselines.get("baseline_matrix_max_queries_per_scene", 4096)), 1)
    objects_per_session = max(int(baselines.get("baseline_matrix_objects_per_session", 8)), 1)
    min_uncovered_pixels = max(
        int(baselines.get("baseline_matrix_min_uncovered_candidate_pixels", baselines.get("full_sam31_min_uncovered_candidate_pixels", min_pixels))),
        1,
    )
    min_uncovered_fraction = float(
        baselines.get("baseline_matrix_min_uncovered_candidate_fraction", baselines.get("full_sam31_min_uncovered_candidate_fraction", 0.50))
    )
    prompt_covered_candidates_enabled = bool(baselines.get("baseline_matrix_prompt_covered_anchor_candidates_enabled", False))
    reuse_existing_anchor_ids_enabled = bool(baselines.get("baseline_matrix_reuse_existing_anchor_id_enabled", False))
    reuse_existing_tracked_output_id_enabled = bool(baselines.get("baseline_matrix_reuse_existing_tracked_output_id_enabled", False))
    offload_video_to_cpu = bool(baselines.get("baseline_matrix_offload_video_to_cpu", True))
    offload_state_to_cpu = bool(baselines.get("baseline_matrix_offload_state_to_cpu", True))
    id_stabilization_enabled = bool(baselines.get("baseline_matrix_temporal_id_stabilization_enabled", True))
    id_stabilization_lookback = max(int(baselines.get("baseline_matrix_temporal_id_lookback", 3)), 1)
    id_stabilization_min_score = float(baselines.get("baseline_matrix_temporal_id_min_match_score", 0.18))
    track_registry_anchor_min_iou = float(baselines.get("baseline_matrix_track_registry_anchor_min_iou", 0.05))
    track_registry_anchor_min_coverage = float(baselines.get("baseline_matrix_track_registry_anchor_min_coverage", 0.10))
    track_registry_lookback = max(int(baselines.get("baseline_matrix_track_registry_lookback", 5)), 1)
    track_registry_min_score = float(baselines.get("baseline_matrix_track_registry_min_score", 0.18))
    track_registry_update_from_outputs_enabled = bool(
        baselines.get("baseline_matrix_track_registry_update_from_outputs_enabled", False)
    )
    track_registry_history_max_entries = max(int(baselines.get("baseline_matrix_track_registry_history_max_entries", 96)), 1)
    track_registry_geometry_enabled = bool(baselines.get("baseline_matrix_track_registry_geometry_enabled", False))
    track_registry_geometry_max_points = max(int(baselines.get("baseline_matrix_track_registry_geometry_max_points", 512)), 1)
    track_registry_geometry_near_threshold_m = float(baselines.get("baseline_matrix_track_registry_geometry_near_threshold_m", 0.08))
    track_registry_geometry_centroid_threshold_m = float(
        baselines.get("baseline_matrix_track_registry_geometry_centroid_threshold_m", 0.45)
    )
    track_registry_geometry_min_score = float(baselines.get("baseline_matrix_track_registry_geometry_min_score", 0.25))
    tracked_mask_support_gate_enabled = bool(baselines.get("baseline_matrix_tracked_mask_support_gate_enabled", False))
    tracked_mask_support_gate_min_score = float(baselines.get("baseline_matrix_tracked_mask_support_gate_min_score", 0.16))
    tracked_mask_support_gate_min_iou = float(baselines.get("baseline_matrix_tracked_mask_support_gate_min_iou", 0.02))
    tracked_mask_support_gate_min_mask_coverage = float(
        baselines.get("baseline_matrix_tracked_mask_support_gate_min_mask_coverage", 0.05)
    )
    tracked_mask_support_gate_min_support_coverage = float(
        baselines.get("baseline_matrix_tracked_mask_support_gate_min_support_coverage", 0.20)
    )
    tracked_mask_support_gate_max_area_expand = float(
        baselines.get("baseline_matrix_tracked_mask_support_gate_max_area_expand", 4.0)
    )
    tracked_mask_support_gate_require_geometry = bool(
        baselines.get("baseline_matrix_tracked_mask_support_gate_require_geometry", False)
    )
    tracked_mask_support_gate_write_full_support = bool(
        baselines.get("baseline_matrix_tracked_mask_support_gate_write_full_support", False)
    )
    keyframe_proposal_direct_write_enabled = bool(
        baselines.get("baseline_matrix_keyframe_proposal_direct_write_enabled", False)
    )
    keyframe_proposal_prewrite_all_enabled = bool(
        baselines.get("baseline_matrix_keyframe_proposal_prewrite_all_enabled", False)
    )
    candidate_count = 0
    query_count = 0
    output_mask_count = 0
    frame_output_count = 0
    nonzero_output_frame_count = 0
    zero_output_frame_ids: list[int] = []
    tracked_object_count = 0
    failed_object_count = 0
    tracker_session_count = 0
    capped_by_query_limit = False
    track_registry_reuse_count = 0
    track_registry_new_count = 0
    track_registry_output_update_count = 0
    track_registry_geometry_match_count = 0
    tracked_output_reuse_count = 0
    preassigned_override_count = 0
    tracked_mask_support_gate_applied_count = 0
    tracked_mask_support_gate_rejected_count = 0
    tracked_mask_support_gate_raw_pixel_count = 0
    tracked_mask_support_gate_clipped_pixel_count = 0
    keyframe_proposal_direct_write_count = 0
    keyframe_proposal_direct_write_pixel_count = 0
    keyframe_proposal_prewrite_count = 0
    keyframe_proposal_prewrite_pixel_count = 0
    covered_candidate_skip_count = 0
    covered_candidate_prompt_count = 0
    candidate_reject_counts: dict[str, int] = {}
    objectness_rejected_candidate_count = 0
    batch_failure_reasons: list[str] = []
    track_rows: list[dict[str, Any]] = []
    id_stabilization_summary: dict[str, Any] = {"enabled": False}
    frame_resource: str | None = None
    predictor = None
    try:
        import torch  # type: ignore

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if not torch.cuda.is_available():
            return {
                "schema_version": "stream4d_v105_mask_prompt_video_tracker_runtime_row_v1",
                "variant_id": variant_id,
                "status": "failed",
                "failure_type": "runtime_error",
                "scene_id": scene_id,
                "frame_ids": ";".join(str(int(frame_id)) for frame_id in frame_ids),
                "frame_count": len(frame_ids),
                "frame_output_count": 0,
                "segmentor_provider": proposal_source,
                "tracker_provider": tracker_provider,
                "candidate_count": 0,
                "query_count": 0,
                "query_limit": max_queries,
                "output_mask_count": 0,
                "tracked_object_count": 0,
                "failed_object_count": 0,
                "tracker_session_count": 0,
                "latency_sec": time.time() - t0,
                "peak_gpu_memory_mb": peak_gpu_mb,
                "failure_reason": f"{tracker_provider} baseline requires CUDA",
                "each_input_frame_treated_as_anchor": True,
                "no_internal_keyframe_subsampling": True,
            }
        if not frame_ids:
            raise RuntimeError(f"{variant_id} received an empty frame list")
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        first_image = _preprocessed_rgb_for_sam2(ctx, scene_id, int(frame_ids[0]))
        h, w = first_image.shape[:2]
        frame_area = float(max(h * w, 1))
        geometry_depth_by_local: dict[int, np.ndarray | None] = {}
        geometry_intrinsics_by_local: dict[int, np.ndarray | None] = {}
        geometry_pose_by_local: dict[int, np.ndarray | None] = {}
        if track_registry_geometry_enabled:
            for geom_local_idx, geom_frame_id in enumerate(frame_ids):
                geometry_depth_by_local[int(geom_local_idx)] = _preprocessed_depth_for_sgq(ctx, scene_id, int(geom_frame_id))
                geometry_intrinsics_by_local[int(geom_local_idx)] = _preprocessed_intrinsics_for_sgq(ctx, scene_id, int(geom_frame_id))
                geometry_pose_by_local[int(geom_local_idx)] = _pose_for_sgq(ctx, scene_id, int(geom_frame_id))
        detections: list[dict[str, Any]] = []
        for local_idx, frame_id in enumerate(frame_ids):
            proposal_path = _proposal_label_path_for_baseline(ctx, scene_id, int(frame_id), proposal_source)
            proposal_label = _preprocessed_label_for_sgq(ctx, scene_id, int(frame_id), proposal_path)
            if proposal_label is None:
                continue
            if proposal_label.shape[:2] != (h, w):
                proposal_label = cv2.resize(proposal_label, (w, h), interpolation=cv2.INTER_NEAREST)
            proposal_ids = [int(value) for value in np.unique(proposal_label) if int(value) > 0]
            proposal_ids.sort(key=lambda value: int(np.count_nonzero(proposal_label == value)), reverse=True)
            frame_detections: list[dict[str, Any]] = []
            for proposal_id in proposal_ids:
                proposal_mask = proposal_label == proposal_id
                gate = _candidate_objectness_gate(
                    proposal_mask,
                    min_pixels=min_pixels,
                    min_area_ratio=min_area_ratio,
                    max_area_ratio=max_area_ratio,
                    broad_area_ratio=objectness_broad_area_ratio,
                    max_bbox_area_ratio=objectness_max_bbox_area_ratio,
                    max_span_ratio=objectness_max_span_ratio,
                    max_edge_pixel_ratio=objectness_max_edge_pixel_ratio,
                    max_edge_touch_sides=objectness_max_edge_touch_sides,
                    enabled=objectness_gate_enabled,
                )
                if not bool(gate["accepted"]):
                    objectness_rejected_candidate_count += 1
                    for reason in gate["reject_reasons"]:
                        candidate_reject_counts[str(reason)] = candidate_reject_counts.get(str(reason), 0) + 1
                    continue
                candidate_count += 1
                pixel_area = int(gate["pixel_area"])
                area_ratio = float(gate["area_ratio"])
                frame_detections.append(
                    {
                        "local_frame_idx": int(local_idx),
                        "frame_id": int(frame_id),
                        "proposal_mask_id": int(proposal_id),
                        "mask": proposal_mask.astype(bool),
                        "pixel_area": pixel_area,
                        "area_ratio": area_ratio,
                        "objectness_bbox_area_ratio": float(gate["bbox_area_ratio"]),
                        "objectness_fill_ratio": float(gate["fill_ratio"]),
                        "objectness_span_w": float(gate["span_w"]),
                        "objectness_span_h": float(gate["span_h"]),
                        "objectness_edge_touch_sides": int(gate["edge_touch_sides"]),
                        "objectness_edge_pixel_ratio": float(gate["edge_pixel_ratio"]),
                    }
                )
            frame_detections.sort(
                key=lambda row: (
                    abs(float(row["area_ratio"]) - objectness_target_area_ratio),
                    float(row["area_ratio"]),
                    int(row["proposal_mask_id"]),
                )
            )
            detections.extend(frame_detections[:max_candidates_per_frame])
        if not detections:
            for frame_id in frame_ids:
                cv2.imwrite(str(mask_dir / f"{int(frame_id)}.png"), np.zeros((h, w), dtype=np.uint16))
            zero_output_frame_ids = [int(frame_id) for frame_id in frame_ids]
            return {
                "schema_version": "stream4d_v105_mask_prompt_video_tracker_runtime_row_v1",
                "variant_id": variant_id,
                "status": "failed",
                "failure_type": "no_candidates",
                "scene_id": scene_id,
                "frame_ids": ";".join(str(int(frame_id)) for frame_id in frame_ids),
                "frame_count": len(frame_ids),
                "frame_output_count": len(frame_ids),
                "segmentor_provider": proposal_source,
                "tracker_provider": tracker_provider,
                "candidate_count": 0,
                "query_count": 0,
                "query_limit": max_queries,
                "output_mask_count": 0,
                "nonzero_output_frame_count": 0,
                "zero_output_frame_ids": zero_output_frame_ids,
                "all_input_frames_have_nonzero_masks": False,
                "tracked_object_count": 0,
                "failed_object_count": 0,
                "tracker_session_count": 0,
                "latency_sec": time.time() - t0,
                "peak_gpu_memory_mb": peak_gpu_mb,
                "failure_reason": f"no valid {proposal_source} proposal anchors for {tracker_provider} mask-prompt tracking",
                "each_input_frame_treated_as_anchor": True,
                "no_internal_keyframe_subsampling": True,
            }
        capped_by_query_limit = len(detections) > max_queries
        selected = detections[:max_queries]
        query_count = len(selected)
        labels_by_local: dict[int, np.ndarray] = {int(idx): np.zeros((h, w), dtype=np.uint16) for idx in range(len(frame_ids))}
        proposal_supports_by_local: dict[int, list[dict[str, Any]]] = {int(idx): [] for idx in range(len(frame_ids))}
        for det in detections:
            support_local_idx = int(det["local_frame_idx"])
            support_mask = np.asarray(det["mask"]).astype(bool)
            support_world_points = _mask_world_points_from_depth(
                support_mask,
                depth_mm=geometry_depth_by_local.get(support_local_idx),
                intrinsics=geometry_intrinsics_by_local.get(support_local_idx),
                pose_c2w=geometry_pose_by_local.get(support_local_idx),
                max_points=track_registry_geometry_max_points,
            ) if track_registry_geometry_enabled else None
            proposal_supports_by_local.setdefault(int(det["local_frame_idx"]), []).append(
                {
                    "proposal_mask_id": int(det["proposal_mask_id"]),
                    "mask": support_mask,
                    "pixel_area": int(det["pixel_area"]),
                    "area_ratio": float(det["area_ratio"]),
                    "world_points": support_world_points,
                }
            )
        frame_resource, _local_index_by_frame = _write_sam31_frame_resource(ctx, scene_id, [int(frame_id) for frame_id in frame_ids])
        predictor = _build_mask_prompt_video_predictor(ctx, tracker_provider, device=device)
        next_track_id = 1
        track_registry: dict[int, dict[str, Any]] = {}

        def _normalise_video_mask(mask: np.ndarray) -> np.ndarray:
            mask_np = np.asarray(mask)
            if mask_np.ndim == 4 and mask_np.shape[1] == 1:
                mask_np = mask_np[:, 0]
            if mask_np.ndim == 3:
                if mask_np.shape[0] == 1:
                    mask_np = mask_np[0]
                else:
                    mask_np = np.max(mask_np, axis=0)
            mask_bool = mask_np > 0
            if mask_bool.shape[:2] != (h, w):
                mask_bool = cv2.resize(mask_bool.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
            return mask_bool

        def _world_points_for_local_mask(mask: np.ndarray, local_frame_idx: int) -> np.ndarray | None:
            if not track_registry_geometry_enabled:
                return None
            local_frame_idx = int(local_frame_idx)
            return _mask_world_points_from_depth(
                mask,
                depth_mm=geometry_depth_by_local.get(local_frame_idx),
                intrinsics=geometry_intrinsics_by_local.get(local_frame_idx),
                pose_c2w=geometry_pose_by_local.get(local_frame_idx),
                max_points=track_registry_geometry_max_points,
            )

        def _remember_prompt_state(track_id: int, det: dict[str, Any]) -> None:
            identity_mask = np.asarray(det.get("_v105_identity_mask", det["mask"])).astype(bool)
            local_frame_idx = int(det["local_frame_idx"])
            world_points = _world_points_for_local_mask(identity_mask, local_frame_idx)
            _append_mask_track_state(
                track_registry,
                int(track_id),
                identity_mask,
                min_pixels=min_pixels,
                local_frame_idx=local_frame_idx,
                frame_id=int(det["frame_id"]),
                max_history=track_registry_history_max_entries,
                world_points=world_points,
            )

        def _assign_prompt_global_track(det: dict[str, Any]) -> int:
            nonlocal next_track_id, track_registry_reuse_count, track_registry_new_count
            nonlocal track_registry_geometry_match_count
            if int(det.get("_v105_assigned_track_id", 0) or 0) > 0:
                return int(det["_v105_assigned_track_id"])
            proposal_mask = np.asarray(det.get("_v105_identity_mask", det["mask"])).astype(bool)
            local_frame_idx = int(det["local_frame_idx"])
            anchor_track_id, anchor_iou, anchor_coverage = _best_overlap_with_label(
                proposal_mask,
                labels_by_local.get(local_frame_idx),
            )
            proposal_world_points = _world_points_for_local_mask(proposal_mask, local_frame_idx)
            registry_track_id, registry_match = _best_track_registry_match(
                proposal_mask,
                track_registry,
                local_frame_idx=local_frame_idx,
                min_pixels=min_pixels,
                lookback=track_registry_lookback,
                min_score=track_registry_min_score,
                world_points=proposal_world_points,
                geometry_near_threshold_m=track_registry_geometry_near_threshold_m,
                geometry_centroid_threshold_m=track_registry_geometry_centroid_threshold_m,
                geometry_min_score=track_registry_geometry_min_score,
            )
            if (
                reuse_existing_anchor_ids_enabled
                and not keyframe_proposal_prewrite_all_enabled
                and
                anchor_track_id > 0
                and (
                    anchor_iou >= track_registry_anchor_min_iou
                    or anchor_coverage >= track_registry_anchor_min_coverage
                )
            ):
                assigned_track_id = int(anchor_track_id)
                assignment_kind = "reused_anchor_label_overlap"
                track_registry_reuse_count += 1
            elif registry_track_id > 0:
                assigned_track_id = int(registry_track_id)
                assignment_kind = "reused_track_registry_continuity"
                track_registry_reuse_count += 1
                if bool((registry_match or {}).get("geometry_gate", False)):
                    track_registry_geometry_match_count += 1
            else:
                assigned_track_id = int(next_track_id)
                next_track_id += 1
                assignment_kind = "new_anchor_prompt_registry_reserved"
                track_registry_new_count += 1
            det["_v105_assigned_track_id"] = int(assigned_track_id)
            det["_v105_assignment_kind"] = assignment_kind
            det["_v105_registry_anchor_track_id"] = int(anchor_track_id)
            det["_v105_registry_anchor_iou"] = float(anchor_iou)
            det["_v105_registry_anchor_coverage"] = float(anchor_coverage)
            det["_v105_registry_match_track_id"] = int(registry_track_id)
            det["_v105_registry_match"] = registry_match or {}
            _remember_prompt_state(int(assigned_track_id), det)
            return int(assigned_track_id)

        def _write_keyframe_proposal(det: dict[str, Any], track_id: int) -> bool:
            nonlocal output_mask_count
            nonlocal keyframe_proposal_direct_write_count, keyframe_proposal_direct_write_pixel_count
            nonlocal keyframe_proposal_prewrite_count, keyframe_proposal_prewrite_pixel_count
            if not keyframe_proposal_direct_write_enabled:
                return False
            if bool(det.get("_v105_keyframe_proposal_written", False)):
                return False
            track_id = int(track_id)
            if track_id <= 0:
                return False
            local_frame_idx = int(det["local_frame_idx"])
            if local_frame_idx not in labels_by_local:
                return False
            proposal_mask = np.asarray(det.get("_v105_identity_mask", det["mask"])).astype(bool)
            if proposal_mask.shape[:2] != (h, w):
                proposal_mask = cv2.resize(proposal_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
            label = labels_by_local[local_frame_idx]
            writable = (label == 0) | (label == int(track_id))
            region = proposal_mask & writable
            pixel_count = int(np.count_nonzero(region))
            if pixel_count < min_pixels:
                return False
            label[region] = int(track_id)
            labels_by_local[local_frame_idx] = label
            det["_v105_keyframe_proposal_written"] = True
            keyframe_proposal_direct_write_count += 1
            keyframe_proposal_direct_write_pixel_count += int(pixel_count)
            if keyframe_proposal_prewrite_all_enabled:
                keyframe_proposal_prewrite_count += 1
                keyframe_proposal_prewrite_pixel_count += int(pixel_count)
            output_mask_count += 1
            return True

        def _run_batch(batch: list[dict[str, Any]]) -> None:
            nonlocal output_mask_count, tracked_object_count, failed_object_count, tracker_session_count, next_track_id
            nonlocal tracked_output_reuse_count, preassigned_override_count
            nonlocal track_registry_output_update_count
            nonlocal covered_candidate_skip_count, covered_candidate_prompt_count
            nonlocal tracked_mask_support_gate_applied_count, tracked_mask_support_gate_rejected_count
            nonlocal tracked_mask_support_gate_raw_pixel_count, tracked_mask_support_gate_clipped_pixel_count
            inference_state = None
            obj_to_track_id: dict[int, int] = {}
            obj_to_assignment_kind: dict[int, str] = {}
            obj_to_det: dict[int, dict[str, Any]] = {}
            active_batch: list[dict[str, Any]] = []
            for det in batch:
                local_frame_idx = int(det["local_frame_idx"])
                anchor_label = labels_by_local.get(local_frame_idx)
                proposal_mask = np.asarray(det["mask"]).astype(bool)
                proposal_pixels = int(np.count_nonzero(proposal_mask))
                uncovered_pixels, uncovered_fraction, uncovered_mask = _candidate_uncovered_stats(
                    proposal_mask,
                    anchor_label,
                )
                is_covered_candidate = (
                    uncovered_pixels < min_uncovered_pixels
                    or uncovered_fraction < min_uncovered_fraction
                )
                if is_covered_candidate and not (prompt_covered_candidates_enabled or keyframe_proposal_prewrite_all_enabled):
                    covered_candidate_skip_count += 1
                    continue
                prompt_mask = proposal_mask
                prompt_mask_policy = "full_proposal"
                if (not keyframe_proposal_prewrite_all_enabled) and 0 < uncovered_pixels < proposal_pixels:
                    prompt_mask = uncovered_mask
                    prompt_mask_policy = "uncovered_region_only"
                prompt_pixels = int(np.count_nonzero(prompt_mask))
                if prompt_pixels < min_pixels:
                    covered_candidate_skip_count += 1
                    continue
                active_det = dict(det)
                active_det["mask"] = prompt_mask.astype(bool)
                active_det["_v105_prompt_mask_policy"] = prompt_mask_policy
                active_det["_v105_prompt_pixel_area"] = int(prompt_pixels)
                active_det["_v105_identity_mask"] = proposal_mask.astype(bool)
                active_det["_v105_identity_pixel_area"] = int(proposal_pixels)
                active_det["_v105_uncovered_pixels_before_prompt"] = int(uncovered_pixels)
                active_det["_v105_uncovered_fraction_before_prompt"] = float(uncovered_fraction)
                active_det["_v105_covered_anchor_candidate"] = bool(is_covered_candidate)
                if is_covered_candidate:
                    covered_candidate_prompt_count += 1
                active_batch.append(active_det)
            if not active_batch:
                return
            active_batch.sort(
                key=lambda row: (
                    int(row.get("_v105_identity_pixel_area", row.get("pixel_area", 0)) or 0),
                    int(row.get("local_frame_idx", 0) or 0),
                    int(row.get("proposal_mask_id", 0) or 0),
                )
            )
            try:
                inference_state = predictor.init_state(
                    video_path=frame_resource,
                    offload_video_to_cpu=offload_video_to_cpu,
                    offload_state_to_cpu=offload_state_to_cpu,
                    async_loading_frames=False,
                )
                tracker_session_count += 1
                for local_obj_id, det in enumerate(active_batch, start=1):
                    assigned_track_id = _assign_prompt_global_track(det)
                    obj_to_track_id[int(local_obj_id)] = assigned_track_id
                    obj_to_assignment_kind[int(local_obj_id)] = str(det.get("_v105_assignment_kind", ""))
                    obj_to_det[int(local_obj_id)] = det
                    if not keyframe_proposal_prewrite_all_enabled:
                        _write_keyframe_proposal(det, int(assigned_track_id))
                    predictor.add_new_mask(
                        inference_state,
                        frame_idx=int(det["local_frame_idx"]),
                        obj_id=int(local_obj_id),
                        mask=det["mask"],
                    )
                written_frame_counts: dict[int, int] = {obj_id: 0 for obj_id in obj_to_track_id}
                written_pixel_counts: dict[int, int] = {obj_id: 0 for obj_id in obj_to_track_id}
                support_gate_written_frame_counts: dict[int, int] = {obj_id: 0 for obj_id in obj_to_track_id}
                support_gate_rejected_frame_counts: dict[int, int] = {obj_id: 0 for obj_id in obj_to_track_id}
                support_gate_raw_pixel_counts: dict[int, int] = {obj_id: 0 for obj_id in obj_to_track_id}
                support_gate_clipped_pixel_counts: dict[int, int] = {obj_id: 0 for obj_id in obj_to_track_id}
                for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                    out_frame_idx = int(out_frame_idx)
                    if out_frame_idx not in labels_by_local:
                        continue
                    label = labels_by_local[out_frame_idx]
                    masks_np = out_mask_logits.detach().float().cpu().numpy()
                    for mask_idx, obj_id in enumerate([int(value) for value in out_obj_ids]):
                        if obj_id not in obj_to_track_id or mask_idx >= masks_np.shape[0]:
                            continue
                        mask_bool = _normalise_video_mask(masks_np[mask_idx])
                        raw_pixels = int(np.count_nonzero(mask_bool))
                        track_id = int(obj_to_track_id[obj_id])
                        if tracked_mask_support_gate_enabled:
                            support_mask, support_info = _best_proposal_support_match(
                                mask_bool,
                                proposal_supports_by_local.get(out_frame_idx, []),
                                prompt_area=int(
                                    obj_to_det[obj_id].get("pixel_area", obj_to_det[obj_id].get("_v105_prompt_pixel_area", raw_pixels))
                                    or raw_pixels
                                ),
                                min_pixels=min_pixels,
                                min_score=tracked_mask_support_gate_min_score,
                                min_iou=tracked_mask_support_gate_min_iou,
                                min_mask_coverage=tracked_mask_support_gate_min_mask_coverage,
                                min_support_coverage=tracked_mask_support_gate_min_support_coverage,
                                max_area_expand=tracked_mask_support_gate_max_area_expand,
                                track_state=track_registry.get(int(track_id)),
                                require_geometry=tracked_mask_support_gate_require_geometry,
                                geometry_near_threshold_m=track_registry_geometry_near_threshold_m,
                                geometry_centroid_threshold_m=track_registry_geometry_centroid_threshold_m,
                                geometry_min_score=track_registry_geometry_min_score,
                                write_full_support=tracked_mask_support_gate_write_full_support,
                            )
                            if support_mask is None:
                                tracked_mask_support_gate_rejected_count += 1
                                support_gate_rejected_frame_counts[obj_id] += 1
                                continue
                            clipped_pixels = int(np.count_nonzero(support_mask))
                            tracked_mask_support_gate_applied_count += 1
                            tracked_mask_support_gate_raw_pixel_count += int(raw_pixels)
                            tracked_mask_support_gate_clipped_pixel_count += int(clipped_pixels)
                            support_gate_written_frame_counts[obj_id] += 1
                            support_gate_raw_pixel_counts[obj_id] += int(raw_pixels)
                            support_gate_clipped_pixel_counts[obj_id] += int(clipped_pixels)
                            mask_bool = support_mask
                        assignment_kind = str(obj_to_assignment_kind.get(obj_id, ""))
                        if (
                            reuse_existing_tracked_output_id_enabled
                            and not keyframe_proposal_prewrite_all_enabled
                            and assignment_kind.startswith("new_")
                        ):
                            overlap_track_id, overlap_iou, overlap_coverage = _best_overlap_with_label(mask_bool, label)
                            if (
                                overlap_track_id > 0
                                and (
                                    overlap_iou >= track_registry_anchor_min_iou
                                    or overlap_coverage >= track_registry_anchor_min_coverage
                                )
                            ):
                                old_track_id = int(track_id)
                                track_id = int(overlap_track_id)
                                obj_to_track_id[obj_id] = track_id
                                obj_to_assignment_kind[obj_id] = "reused_tracked_output_overlap"
                                track_registry.pop(old_track_id, None)
                                _remember_prompt_state(track_id, obj_to_det[obj_id])
                                tracked_output_reuse_count += 1
                                preassigned_override_count += 1
                                assignment_kind = "reused_tracked_output_overlap"
                        if assignment_kind.startswith("new_"):
                            region = mask_bool & (label == 0)
                        else:
                            region = mask_bool & ((label == 0) | (label == int(track_id)))
                        pixel_count = int(np.count_nonzero(region))
                        if pixel_count < min_pixels:
                            continue
                        label[region] = track_id
                        labels_by_local[out_frame_idx] = label
                        if track_registry_update_from_outputs_enabled:
                            world_points = _world_points_for_local_mask(mask_bool, out_frame_idx)
                            if _append_mask_track_state(
                                track_registry,
                                int(track_id),
                                mask_bool,
                                min_pixels=min_pixels,
                                local_frame_idx=out_frame_idx,
                                frame_id=int(frame_ids[out_frame_idx]),
                                max_history=track_registry_history_max_entries,
                                world_points=world_points,
                            ):
                                track_registry_output_update_count += 1
                        output_mask_count += 1
                        written_frame_counts[obj_id] += 1
                        written_pixel_counts[obj_id] += pixel_count
                for obj_id, track_id in sorted(obj_to_track_id.items()):
                    det = obj_to_det[obj_id]
                    written_frames = int(written_frame_counts.get(obj_id, 0))
                    written_pixels = int(written_pixel_counts.get(obj_id, 0))
                    if written_frames > 0:
                        tracked_object_count += 1
                    else:
                        failed_object_count += 1
                    track_rows.append(
                        {
                            "track_id": int(track_id),
                            "track_id_assignment_kind": str(obj_to_assignment_kind.get(obj_id, "")),
                            "prompt_local_frame_idx": int(det["local_frame_idx"]),
                            "prompt_frame_id": int(det["frame_id"]),
                            "proposal_mask_id": int(det["proposal_mask_id"]),
                            "prompt_mask_policy": str(det.get("_v105_prompt_mask_policy", "")),
                            "prompt_pixel_area": int(det.get("_v105_prompt_pixel_area", 0) or 0),
                            "identity_pixel_area": int(det.get("_v105_identity_pixel_area", 0) or 0),
                            "uncovered_pixels_before_prompt": int(det.get("_v105_uncovered_pixels_before_prompt", 0) or 0),
                            "uncovered_fraction_before_prompt": float(det.get("_v105_uncovered_fraction_before_prompt", 0.0) or 0.0),
                            "covered_anchor_candidate": bool(det.get("_v105_covered_anchor_candidate", False)),
                            "registry_anchor_track_id": int(det.get("_v105_registry_anchor_track_id", 0) or 0),
                            "registry_anchor_iou": float(det.get("_v105_registry_anchor_iou", 0.0) or 0.0),
                            "registry_anchor_coverage": float(det.get("_v105_registry_anchor_coverage", 0.0) or 0.0),
                            "registry_match_track_id": int(det.get("_v105_registry_match_track_id", 0) or 0),
                            "registry_match": det.get("_v105_registry_match", {}),
                            "tracked_mask_support_gate_written_frame_count": int(support_gate_written_frame_counts.get(obj_id, 0)),
                            "tracked_mask_support_gate_rejected_frame_count": int(support_gate_rejected_frame_counts.get(obj_id, 0)),
                            "tracked_mask_support_gate_raw_pixel_count": int(support_gate_raw_pixel_counts.get(obj_id, 0)),
                            "tracked_mask_support_gate_clipped_pixel_count": int(support_gate_clipped_pixel_counts.get(obj_id, 0)),
                            "written_frame_count": written_frames,
                            "written_pixel_count": written_pixels,
                        }
                    )
            finally:
                if inference_state is not None and hasattr(predictor, "reset_state"):
                    try:
                        predictor.reset_state(inference_state)
                    except Exception:
                        pass

        if keyframe_proposal_prewrite_all_enabled:
            for det in sorted(
                selected,
                key=lambda row: (
                    int(row.get("local_frame_idx", 0) or 0),
                    int(row.get("pixel_area", 0) or 0),
                    int(row.get("proposal_mask_id", 0) or 0),
                ),
            ):
                assigned_track_id = _assign_prompt_global_track(det)
                _write_keyframe_proposal(det, int(assigned_track_id))

        for start in range(0, len(selected), objects_per_session):
            batch = selected[start : start + objects_per_session]
            try:
                _run_batch(batch)
            except RuntimeError as exc:
                reason = _safe_error(exc)
                if "out of memory" in reason.lower() and len(batch) > 1:
                    mid = max(1, len(batch) // 2)
                    for sub_batch in (batch[:mid], batch[mid:]):
                        try:
                            _run_batch(sub_batch)
                        except Exception as sub_exc:
                            failed_object_count += len(sub_batch)
                            batch_failure_reasons.append(_safe_error(sub_exc))
                    torch.cuda.empty_cache()
                else:
                    failed_object_count += len(batch)
                    batch_failure_reasons.append(reason)
                    torch.cuda.empty_cache()
            except Exception as exc:
                failed_object_count += len(batch)
                batch_failure_reasons.append(_safe_error(exc))
                torch.cuda.empty_cache()
        for local_idx, frame_id in enumerate(frame_ids):
            label = labels_by_local[int(local_idx)]
            if int(np.count_nonzero(label)) > 0:
                nonzero_output_frame_count += 1
            else:
                zero_output_frame_ids.append(int(frame_id))
            cv2.imwrite(str(mask_dir / f"{int(frame_id)}.png"), label)
            frame_output_count += 1
        if id_stabilization_enabled:
            id_stabilization_summary = _stabilize_label_ids_temporally(
                mask_dir=mask_dir,
                frame_ids=[int(frame_id) for frame_id in frame_ids],
                min_pixels=min_pixels,
                lookback=id_stabilization_lookback,
                min_match_score=id_stabilization_min_score,
                ctx=ctx,
                scene_id=scene_id,
                geometry_enabled=track_registry_geometry_enabled,
                geometry_max_points=track_registry_geometry_max_points,
                geometry_near_threshold_m=track_registry_geometry_near_threshold_m,
                geometry_centroid_threshold_m=track_registry_geometry_centroid_threshold_m,
                geometry_min_score=track_registry_geometry_min_score,
            )
        torch.cuda.synchronize()
        peak_gpu_mb = float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)
        if output_mask_count <= 0:
            status = "failed"
        elif batch_failure_reasons or capped_by_query_limit or failed_object_count > 0 or zero_output_frame_ids:
            status = "partial"
        else:
            status = "completed"
        if batch_failure_reasons:
            failure_reason = "; ".join(batch_failure_reasons)
        elif capped_by_query_limit:
            failure_reason = "query_limit_capped_object_prompts"
        elif failed_object_count > 0:
            failure_reason = "one or more object tracks produced no valid masks"
        elif zero_output_frame_ids:
            failure_reason = "zero_output_input_frames=" + ";".join(str(frame_id) for frame_id in zero_output_frame_ids)
        else:
            failure_reason = ""
        return {
            "schema_version": "stream4d_v105_mask_prompt_video_tracker_runtime_row_v1",
            "variant_id": variant_id,
            "status": status,
            "scene_id": scene_id,
            "frame_ids": ";".join(str(int(frame_id)) for frame_id in frame_ids),
            "frame_count": len(frame_ids),
            "frame_output_count": frame_output_count,
            "segmentor_provider": proposal_source,
            "tracker_provider": tracker_provider,
            "candidate_count": candidate_count,
            "query_count": query_count,
            "query_limit": max_queries,
            "capped_by_query_limit": capped_by_query_limit,
            "output_mask_count": output_mask_count,
            "nonzero_output_frame_count": nonzero_output_frame_count,
            "zero_output_frame_ids": zero_output_frame_ids,
            "all_input_frames_have_nonzero_masks": len(zero_output_frame_ids) == 0,
            "tracked_object_count": int(id_stabilization_summary.get("track_count", tracked_object_count)),
            "failed_object_count": failed_object_count,
            "tracker_session_count": tracker_session_count,
            "objects_per_session": objects_per_session,
            "max_candidates_per_anchor_frame": max_candidates_per_frame,
            "candidate_objectness_gate_enabled": bool(objectness_gate_enabled),
            "candidate_objectness_broad_area_ratio": float(objectness_broad_area_ratio),
            "candidate_objectness_max_bbox_area_ratio": float(objectness_max_bbox_area_ratio),
            "candidate_objectness_max_span_ratio": float(objectness_max_span_ratio),
            "candidate_objectness_max_edge_pixel_ratio": float(objectness_max_edge_pixel_ratio),
            "candidate_objectness_max_edge_touch_sides": int(objectness_max_edge_touch_sides),
            "candidate_objectness_target_area_ratio": float(objectness_target_area_ratio),
            "min_uncovered_candidate_pixels": int(min_uncovered_pixels),
            "min_uncovered_candidate_fraction": float(min_uncovered_fraction),
            "prompt_covered_anchor_candidates_enabled": bool(prompt_covered_candidates_enabled),
            "covered_candidate_skip_count": int(covered_candidate_skip_count),
            "covered_candidate_prompt_count": int(covered_candidate_prompt_count),
            "objectness_rejected_candidate_count": int(objectness_rejected_candidate_count),
            "candidate_reject_counts": dict(candidate_reject_counts),
            "track_registry_enabled": True,
            "track_registry_semantics": "prompt_to_global_track_binding_no_mask_fusion",
            "reuse_existing_anchor_id_enabled": bool(reuse_existing_anchor_ids_enabled),
            "track_registry_anchor_min_iou": float(track_registry_anchor_min_iou),
            "track_registry_anchor_min_coverage": float(track_registry_anchor_min_coverage),
            "track_registry_lookback": int(track_registry_lookback),
            "track_registry_min_score": float(track_registry_min_score),
            "track_registry_reuse_count": int(track_registry_reuse_count),
            "track_registry_new_count": int(track_registry_new_count),
            "track_registry_update_from_outputs_enabled": bool(track_registry_update_from_outputs_enabled),
            "track_registry_history_max_entries": int(track_registry_history_max_entries),
            "track_registry_output_update_count": int(track_registry_output_update_count),
            "track_registry_geometry_enabled": bool(track_registry_geometry_enabled),
            "track_registry_geometry_max_points": int(track_registry_geometry_max_points),
            "track_registry_geometry_near_threshold_m": float(track_registry_geometry_near_threshold_m),
            "track_registry_geometry_centroid_threshold_m": float(track_registry_geometry_centroid_threshold_m),
            "track_registry_geometry_min_score": float(track_registry_geometry_min_score),
            "track_registry_geometry_match_count": int(track_registry_geometry_match_count),
            "reuse_existing_tracked_output_id_enabled": bool(reuse_existing_tracked_output_id_enabled),
            "tracked_output_reuse_count": int(tracked_output_reuse_count),
            "preassigned_override_count": int(preassigned_override_count),
            "tracked_mask_support_gate_enabled": bool(tracked_mask_support_gate_enabled),
            "tracked_mask_support_gate_min_score": float(tracked_mask_support_gate_min_score),
            "tracked_mask_support_gate_min_iou": float(tracked_mask_support_gate_min_iou),
            "tracked_mask_support_gate_min_mask_coverage": float(tracked_mask_support_gate_min_mask_coverage),
            "tracked_mask_support_gate_min_support_coverage": float(tracked_mask_support_gate_min_support_coverage),
            "tracked_mask_support_gate_max_area_expand": float(tracked_mask_support_gate_max_area_expand),
            "tracked_mask_support_gate_require_geometry": bool(tracked_mask_support_gate_require_geometry),
            "tracked_mask_support_gate_write_full_support": bool(tracked_mask_support_gate_write_full_support),
            "tracker_prompt_write_order": "small_to_large_identity_area",
            "keyframe_proposal_direct_write_enabled": bool(keyframe_proposal_direct_write_enabled),
            "keyframe_proposal_direct_write_count": int(keyframe_proposal_direct_write_count),
            "keyframe_proposal_direct_write_pixel_count": int(keyframe_proposal_direct_write_pixel_count),
            "tracked_mask_support_gate_applied_count": int(tracked_mask_support_gate_applied_count),
            "tracked_mask_support_gate_rejected_count": int(tracked_mask_support_gate_rejected_count),
            "tracked_mask_support_gate_raw_pixel_count": int(tracked_mask_support_gate_raw_pixel_count),
            "tracked_mask_support_gate_clipped_pixel_count": int(tracked_mask_support_gate_clipped_pixel_count),
            "track_rows": track_rows,
            "track_rows_are_pre_id_stabilization": bool(id_stabilization_enabled),
            "id_stabilization_enabled": bool(id_stabilization_enabled),
            "id_stabilization_policy": id_stabilization_summary.get("policy", ""),
            "id_stabilization_track_count": id_stabilization_summary.get("track_count", ""),
            "id_stabilization_reused_assignment_count": id_stabilization_summary.get("reused_assignment_count", ""),
            "id_stabilization_new_assignment_count": id_stabilization_summary.get("new_assignment_count", ""),
            "id_stabilization_summary": id_stabilization_summary,
            "latency_sec": time.time() - t0,
            "peak_gpu_memory_mb": peak_gpu_mb,
            "failure_reason": failure_reason,
            "each_input_frame_treated_as_anchor": True,
            "no_internal_keyframe_subsampling": True,
        }
    except Exception as exc:
        return {
            "schema_version": "stream4d_v105_mask_prompt_video_tracker_runtime_row_v1",
            "variant_id": variant_id,
            "status": "failed",
            "failure_type": "runtime_error",
            "scene_id": scene_id,
            "frame_ids": ";".join(str(int(frame_id)) for frame_id in frame_ids),
            "frame_count": len(frame_ids),
            "frame_output_count": frame_output_count,
            "segmentor_provider": proposal_source,
            "tracker_provider": tracker_provider,
            "candidate_count": candidate_count,
            "query_count": query_count,
            "query_limit": max_queries,
            "capped_by_query_limit": capped_by_query_limit,
            "output_mask_count": output_mask_count,
            "tracked_object_count": tracked_object_count,
            "failed_object_count": failed_object_count,
            "tracker_session_count": tracker_session_count,
            "latency_sec": time.time() - t0,
            "peak_gpu_memory_mb": peak_gpu_mb,
            "failure_reason": _safe_error(exc),
            "each_input_frame_treated_as_anchor": True,
            "no_internal_keyframe_subsampling": True,
        }
    finally:
        if predictor is not None and hasattr(predictor, "reset_state"):
            pass
        if frame_resource and Path(frame_resource).exists():
            shutil.rmtree(frame_resource, ignore_errors=True)
        gc.collect()
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def _compute_alltracker_visibility_maps(
    ctx: PipelineContext,
    scene_id: str,
    frame_ids: list[int],
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    t0 = time.time()
    checkpoint = _as_repo_path(ctx.config.get("paths", {}).get("alltracker_checkpoint"))
    root = _as_repo_path(ctx.config.get("paths", {}).get("fourdpm_root")) or (REPO_ROOT / "third_party" / "4D_PM")
    maps: dict[int, np.ndarray] = {}
    row: dict[str, Any] = {
        "schema_version": "stream4d_v105_alltracker_visibility_row_v1",
        "scene_id": scene_id,
        "frame_ids": ";".join(str(int(frame_id)) for frame_id in frame_ids),
        "status": "not_run",
        "runtime_sec": 0.0,
        "peak_gpu_memory_mb": "",
        "checkpoint": _rel(checkpoint) if checkpoint else "",
        "checkpoint_sha256": _sha256_file(checkpoint) if checkpoint else "",
        "output_shape": "",
        "visibility_score_mean": "",
        "failure_reason": "",
    }
    if checkpoint is None or not checkpoint.exists():
        row.update({"status": "failed", "runtime_sec": time.time() - t0, "failure_reason": "AllTracker checkpoint missing"})
        return maps, row
    old_sys_path = list(sys.path)
    try:
        import torch  # type: ignore

        sys.path.insert(0, str(root))
        from frontend.alltracker.wrapper import forward_alltracker, setup_model  # type: ignore

        if not torch.cuda.is_available():
            row.update({"status": "failed", "runtime_sec": time.time() - t0, "failure_reason": "AllTracker wrapper path requires CUDA in this runner"})
            return maps, row
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        model = setup_model(str(checkpoint), window_len=16, weights_only=False, device="cuda:0")
        frames = []
        used_frame_ids = []
        for frame_id in frame_ids:
            rgb = _preprocessed_rgb_for_sam2(ctx, scene_id, int(frame_id))
            frames.append(torch.from_numpy(rgb).permute(2, 0, 1).float().cuda() / 255.0)
            used_frame_ids.append(int(frame_id))
        out = forward_alltracker(frames, model, inference_iters=1)
        torch.cuda.synchronize()
        combined = (out["confidence"][0].float() * out["visibilty"][0].float()).detach().cpu().numpy()
        target_h = int(ctx.config.get("preprocess", {}).get("target_height", 240))
        target_w = int(ctx.config.get("preprocess", {}).get("target_width", 320))
        for idx, frame_id in enumerate(used_frame_ids):
            maps[int(frame_id)] = cv2.resize(combined[idx].astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        peak_mb = float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)
        row.update(
            {
                "status": "completed",
                "runtime_sec": time.time() - t0,
                "peak_gpu_memory_mb": peak_mb,
                "output_shape": list(combined.shape),
                "visibility_score_mean": float(combined.mean()),
                "failure_reason": "",
            }
        )
        del model
        torch.cuda.empty_cache()
    except Exception as exc:
        row.update({"status": "failed", "runtime_sec": time.time() - t0, "failure_reason": _safe_error(exc)})
    finally:
        sys.path[:] = old_sys_path
    return maps, row


def _propagate_mask_with_alltracker(
    ctx: PipelineContext,
    scene_id: str,
    anchor_frame_id: int,
    target_frame_ids: list[int],
    anchor_mask: np.ndarray,
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    t0 = time.time()
    checkpoint = _as_repo_path(ctx.config.get("paths", {}).get("alltracker_checkpoint"))
    root = _as_repo_path(ctx.config.get("paths", {}).get("fourdpm_root")) or (REPO_ROOT / "third_party" / "4D_PM")
    local_cfg = ctx.config.get("local", {})
    conf_thr = float(local_cfg.get("sgq_alltracker_prop_conf_thr", 0.1))
    visibility_thr = float(local_cfg.get("sgq_alltracker_prop_visibility_thr", 0.5))
    min_area_ratio = float(local_cfg.get("sgq_alltracker_prop_min_area_ratio", 0.0002))
    propagated: dict[int, np.ndarray] = {}
    row: dict[str, Any] = {
        "schema_version": "stream4d_v105_alltracker_mask_propagation_record_v1",
        "scene_id": scene_id,
        "anchor_frame_id": int(anchor_frame_id),
        "target_frame_ids": [int(frame_id) for frame_id in target_frame_ids],
        "status": "not_run",
        "runtime_sec": 0.0,
        "peak_gpu_memory_mb": "",
        "checkpoint": _rel(checkpoint) if checkpoint else "",
        "checkpoint_sha256": _sha256_file(checkpoint) if checkpoint else "",
        "anchor_pixel_count": int(np.count_nonzero(anchor_mask)),
        "propagated_frame_count": 0,
        "propagated_pixel_count_by_frame": {},
        "failure_reason": "",
    }
    if checkpoint is None or not checkpoint.exists():
        row.update({"status": "failed", "runtime_sec": time.time() - t0, "failure_reason": "AllTracker checkpoint missing"})
        return propagated, row
    if not np.any(anchor_mask):
        row.update({"status": "failed", "runtime_sec": time.time() - t0, "failure_reason": "anchor mask is empty"})
        return propagated, row
    old_sys_path = list(sys.path)
    try:
        import torch  # type: ignore

        if not torch.cuda.is_available():
            row.update({"status": "failed", "runtime_sec": time.time() - t0, "failure_reason": "AllTracker propagation requires CUDA"})
            return propagated, row
        sys.path.insert(0, str(root))
        from frontend.alltracker.wrapper import forward_alltracker, setup_model  # type: ignore

        used_frame_ids = [int(anchor_frame_id)] + [int(frame_id) for frame_id in target_frame_ids if int(frame_id) != int(anchor_frame_id)]
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        model = setup_model(str(checkpoint), window_len=16, weights_only=False, device="cuda:0")
        frames = []
        for frame_id in used_frame_ids:
            rgb = _preprocessed_rgb_for_sam2(ctx, scene_id, int(frame_id))
            frames.append(torch.from_numpy(rgb).permute(2, 0, 1).float().cuda() / 255.0)
        out = forward_alltracker(frames, model, inference_iters=int(local_cfg.get("sgq_alltracker_prop_inference_iters", 1)), conf_thr=conf_thr, visibility_thr=visibility_thr)
        torch.cuda.synchronize()
        traj = out["traj_maps_e"][0].float()
        confidence = out["confidence"][0].float()
        visibility = out["visibilty"][0].float()
        _, internal_h, internal_w = traj.shape[1:]
        anchor_internal_np = cv2.resize(anchor_mask.astype(np.uint8), (int(internal_w), int(internal_h)), interpolation=cv2.INTER_NEAREST) > 0
        anchor_internal = torch.from_numpy(anchor_internal_np).bool().cuda()
        ys, xs = torch.nonzero(anchor_internal, as_tuple=True)
        target_h = int(ctx.config.get("preprocess", {}).get("target_height", 240))
        target_w = int(ctx.config.get("preprocess", {}).get("target_width", 320))
        min_pixels = max(1, int(round(min_area_ratio * float(target_h * target_w))))
        pixel_count_by_frame: dict[str, int] = {}
        propagated[int(anchor_frame_id)] = anchor_mask.astype(bool, copy=True)
        if ys.numel() > 0:
            for idx, frame_id in enumerate(used_frame_ids):
                if idx == 0:
                    continue
                valid = (confidence[idx, ys, xs] >= conf_thr) & (visibility[idx, ys, xs] >= visibility_thr)
                if not bool(valid.any()):
                    continue
                xy = traj[idx, :, ys[valid], xs[valid]]
                x = torch.round(xy[0]).long()
                y = torch.round(xy[1]).long()
                inbound = (x >= 0) & (x < internal_w) & (y >= 0) & (y < internal_h)
                if not bool(inbound.any()):
                    continue
                target_internal = torch.zeros((int(internal_h), int(internal_w)), dtype=torch.bool, device="cuda:0")
                target_internal[y[inbound], x[inbound]] = True
                target_np = target_internal.detach().cpu().numpy().astype(np.uint8)
                target_np = cv2.dilate(target_np, np.ones((3, 3), dtype=np.uint8), iterations=1) > 0
                target_np = cv2.resize(target_np.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST) > 0
                pixel_count = int(np.count_nonzero(target_np))
                if pixel_count < min_pixels:
                    continue
                propagated[int(frame_id)] = target_np
                pixel_count_by_frame[str(int(frame_id))] = pixel_count
        peak_mb = float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)
        row.update(
            {
                "status": "completed",
                "runtime_sec": time.time() - t0,
                "peak_gpu_memory_mb": peak_mb,
                "propagated_frame_count": len([frame_id for frame_id in propagated if int(frame_id) != int(anchor_frame_id)]),
                "propagated_pixel_count_by_frame": pixel_count_by_frame,
                "failure_reason": "",
            }
        )
        del model
        torch.cuda.empty_cache()
    except Exception as exc:
        row.update({"status": "failed", "runtime_sec": time.time() - t0, "failure_reason": _safe_error(exc)})
    finally:
        sys.path[:] = old_sys_path
    return propagated, row


def _run_b2_4dpm_sam2_scene(ctx: PipelineContext, scene_id: str, frame_ids: list[int], mask_dir: Path) -> dict[str, Any]:
    t0 = time.time()
    peak_gpu_mb = ""
    mask_dir.mkdir(parents=True, exist_ok=True)
    for stale_mask in mask_dir.glob("*.png"):
        stale_mask.unlink()
    diag_dir = ctx.output_root / "baselines" / "diagnostics" / "B2_4dpm_sam2_gap_tracking" / scene_id
    if diag_dir.exists():
        shutil.rmtree(diag_dir)
    diag_dir.mkdir(parents=True, exist_ok=True)
    paths = ctx.config.get("paths", {})
    baselines = ctx.config.get("baselines", {})
    base_seed = int(ctx.config.get("run", {}).get("seed", 0))
    deterministic_b2 = bool(baselines.get("deterministic_4dpm_sam2_gap", True))
    deterministic_point_grid = bool(baselines.get("sam2_gap_deterministic_point_grid", False))
    num_pts = int(baselines.get("sam2_gap_num_pts", 300))
    num_pts_active = int(baselines.get("sam2_gap_num_pts_active", 250))
    scene_seed = _stable_int_seed(ctx.config.get("run", {}).get("name", "v105"), scene_id, base_seed, "b2_4dpm_sam2_gap")
    fourdpm_root = _as_repo_path(paths.get("fourdpm_root")) or (REPO_ROOT / "third_party" / "4D_PM")
    if str(fourdpm_root) not in sys.path:
        sys.path.insert(0, str(fourdpm_root))
    config = {
        "sam_params": {
            "checkpoint": str(_as_repo_path(paths.get("sam2_checkpoint"))),
            "model_cfg": str(paths.get("sam2_model_cfg")),
            "nms_score_type": "stability",
            "select_smallest": True,
            "nms": True,
            "box_nms_thresh": 0.8,
            "iou_threshold": 0.5,
            "stability_threshold": 0.6,
            "active_stability_threshold": 0.6,
            "active_iou_threshold": 0.5,
            "model_mask_thresh": 0.0,
            "deterministic_point_grid": deterministic_point_grid,
            "deterministic_gap_sampling": deterministic_point_grid,
        },
        "frontend": {
            "num_pts": num_pts,
            "num_pts_active": num_pts_active,
        },
    }
    try:
        import torch  # type: ignore
        from frontend.segment.samv2_tools import setup_sam  # type: ignore
        from frontend.segment.video_matcher import run_video_overseg  # type: ignore
        from sam2.build_sam import build_sam2_video_predictor  # type: ignore

        if deterministic_b2:
            random.seed(scene_seed)
            np.random.seed(scene_seed % (2**32 - 1))
            torch.manual_seed(scene_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(scene_seed)
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except TypeError:
                torch.use_deterministic_algorithms(True)
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
            if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
                torch.backends.cuda.matmul.allow_tf32 = False
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.allow_tf32 = False
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        frames = [{"image": _preprocessed_rgb_for_sam2(ctx, scene_id, frame_id), "timestamp": int(frame_id)} for frame_id in frame_ids]
        model_sam = setup_sam(config, device="cuda:0" if torch.cuda.is_available() else "cpu")
        predictor = build_sam2_video_predictor(
            config["sam_params"]["model_cfg"],
            config["sam_params"]["checkpoint"],
            device="cuda:0" if torch.cuda.is_available() else "cpu",
        )
        out = run_video_overseg(model_sam, predictor, frames, config)
        diagnostics = dict(out.get("diagnostics", {}))
        total_masks = 0
        max_mask_id = 0
        frame_diag_rows: list[dict[str, Any]] = []
        contact_tiles: list[np.ndarray] = []
        for local_idx, frame_id in enumerate(frame_ids):
            masks = out["masks"][local_idx].detach().cpu().numpy().astype(bool)
            obj_ids = out["obj_ids"][local_idx].detach().cpu().numpy().astype(np.int64)
            label = np.zeros(masks.shape[1:], dtype=np.uint16)
            for mask, obj_id in zip(masks, obj_ids):
                mask_id = int(obj_id) + 1
                label[mask] = mask_id
                max_mask_id = max(max_mask_id, mask_id)
                total_masks += 1
            cv2.imwrite(str(mask_dir / f"{int(frame_id)}.png"), label)
            label_ids = [int(value) for value in np.unique(label) if int(value) > 0]
            nonzero_pixels = int(np.count_nonzero(label))
            previous_delta = int(frame_id) - int(frame_ids[local_idx - 1]) if local_idx > 0 else None
            frame_diag_rows.append(
                {
                    "schema_version": "stream4d_v105_b2_frame_diagnostic_v1",
                    "scene_id": scene_id,
                    "sequence_frame_index": int(local_idx),
                    "source_frame_id": int(frame_id),
                    "source_frame_delta_from_previous": previous_delta,
                    "sequence_frame_is_keyframe": True,
                    "no_internal_keyframe_subsampling": True,
                    "mask_path": _rel(mask_dir / f"{int(frame_id)}.png"),
                    "label_count": len(label_ids),
                    "nonzero_pixels": nonzero_pixels,
                    "max_label_id": max(label_ids) if label_ids else 0,
                    "empty_mask": len(label_ids) == 0 or nonzero_pixels == 0,
                }
            )
            frame_rgb = frames[local_idx]["image"]
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            overlay = _overlay(frame_bgr, label)
            color = _colorize_labels(label)
            tile = np.concatenate([overlay, color], axis=1)
            cv2.putText(
                tile,
                f"seq={local_idx:02d} src={int(frame_id)} n={len(label_ids)} px={nonzero_pixels}",
                (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            contact_tiles.append(tile)
        frame_diag_path = diag_dir / "frame_diagnostics.json"
        contact_sheet_path = diag_dir / "all_input_frames_overlay_mask_sheet.jpg"
        _write_records_json(frame_diag_path, frame_diag_rows, schema_version="stream4d_v105_b2_frame_diagnostic_table_v1")
        contact_sheet_ok = _write_contact_sheet(contact_tiles, contact_sheet_path, columns=4)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            peak_gpu_mb = float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)
        del out, predictor, model_sam
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {
            "status": "completed",
            "scene_id": scene_id,
            "baseline_scope": "4dpm_frontend_sam2_gap_tracking_code_audited_window_baseline",
            "plan_baseline_name": "baseline_4dpm_sam2_gap_tracking",
            "uses_fourdpm_frontend_run_video_overseg": True,
            "full_d4pm_reconstruction_backend": False,
            "full_sequence_run": False,
            "frame_ids": ";".join(str(int(frame_id)) for frame_id in frame_ids),
            "frame_count": len(frame_ids),
            "mask_frame_count": len(list(mask_dir.glob("*.png"))),
            "total_frame_masks": total_masks,
            "max_mask_id": max_mask_id,
            "each_input_frame_treated_as_keyframe": True,
            "no_internal_keyframe_subsampling": True,
            "sam2_point_sampling": "deterministic_grid" if deterministic_point_grid else "seeded_random" if deterministic_b2 else "random",
            "sam2_num_pts": num_pts,
            "sam2_num_pts_active": num_pts_active,
            "initial_mask_count": diagnostics.get("initial_mask_count", ""),
            "new_gap_mask_count": diagnostics.get("new_gap_mask_count", ""),
            "gap_region_count": diagnostics.get("gap_region_count", ""),
            "gap_region_area_ratio_mean": diagnostics.get("gap_region_area_ratio_mean", ""),
            "sam2_tracking_call_count": diagnostics.get("sam2_tracking_call_count", ""),
            "sam2_initialization_call_count": diagnostics.get("sam2_initialization_call_count", ""),
            "duplicate_mask_suppression_count": diagnostics.get("duplicate_mask_suppression_count", ""),
            "empty_frame_count": len([row for row in frame_diag_rows if row["empty_mask"]]),
            "frame_diagnostics_json": _rel(frame_diag_path),
            "frame_contact_sheet": _rel(contact_sheet_path) if contact_sheet_ok else "",
            "latency_sec": time.time() - t0,
            "peak_gpu_memory_mb": peak_gpu_mb,
            "deterministic_mode": deterministic_b2,
            "deterministic_seed": scene_seed,
            "failure_reason": "",
        }
    except Exception as exc:
        return {
            "status": "failed",
            "scene_id": scene_id,
            "baseline_scope": "4dpm_frontend_sam2_gap_tracking_code_audited_window_baseline",
            "plan_baseline_name": "baseline_4dpm_sam2_gap_tracking",
            "uses_fourdpm_frontend_run_video_overseg": True,
            "full_d4pm_reconstruction_backend": False,
            "full_sequence_run": False,
            "frame_ids": ";".join(str(int(frame_id)) for frame_id in frame_ids),
            "frame_count": len(frame_ids),
            "mask_frame_count": len(list(mask_dir.glob("*.png"))),
            "total_frame_masks": 0,
            "max_mask_id": 0,
            "each_input_frame_treated_as_keyframe": True,
            "no_internal_keyframe_subsampling": True,
            "sam2_point_sampling": "deterministic_grid" if deterministic_point_grid else "seeded_random" if deterministic_b2 else "random",
            "sam2_num_pts": num_pts,
            "sam2_num_pts_active": num_pts_active,
            "initial_mask_count": "",
            "new_gap_mask_count": "",
            "gap_region_count": "",
            "gap_region_area_ratio_mean": "",
            "sam2_tracking_call_count": "",
            "sam2_initialization_call_count": "",
            "duplicate_mask_suppression_count": "",
            "empty_frame_count": "",
            "frame_diagnostics_json": "",
            "frame_contact_sheet": "",
            "latency_sec": time.time() - t0,
            "peak_gpu_memory_mb": peak_gpu_mb,
            "deterministic_mode": deterministic_b2,
            "deterministic_seed": scene_seed,
            "failure_reason": _safe_error(exc),
        }


def run_baselines(ctx: PipelineContext) -> None:
    with StageTimer(ctx, "baselines"):
        if not ctx.gates.get("phase2_provider_smoke_pass", False):
            raise RuntimeError("Phase 2 provider smoke must pass before baselines.")
        out_dir = ctx.output_root / "baselines"
        baselines_cfg = ctx.config.get("baselines", {})
        scenes = [str(scene) for scene in ctx.config.get("run", {}).get("scenes", [])]
        eval_stride = int(baselines_cfg.get("eval_stride", ctx.config.get("run", {}).get("frame_stride", 5)))
        eval_max_frames = int(baselines_cfg.get("eval_max_frames_per_scene", 4))
        frame_ids_by_scene = {scene: _scene_frame_ids_for_baseline(ctx, scene, eval_max_frames) for scene in scenes}
        matrix_cfgs = baselines_cfg.get("baseline_matrix_configs", [])
        matrix_max_frames = int(baselines_cfg.get("baseline_matrix_max_frames_per_scene", baselines_cfg.get("sam2_gap_max_frames_per_scene", eval_max_frames)))
        matrix_frame_ids_by_scene = {scene: _scene_frame_ids_for_baseline(ctx, scene, matrix_max_frames) for scene in scenes}
        scannet_root = _as_repo_path(ctx.config.get("paths", {}).get("scannet_processed_root")) or (STREAM3D_ROOT / "data" / "scannet" / "processed")

        metric_rows: list[dict[str, Any]] = []
        latency_rows: list[dict[str, Any]] = []
        memory_rows: list[dict[str, Any]] = []
        budget_rows: list[dict[str, Any]] = []
        video_rows: list[dict[str, Any]] = []
        b2_scope_summary: dict[str, Any] | None = None
        b3_scope_summary: dict[str, Any] | None = None
        baseline_matrix_summary: dict[str, Any] | None = None
        cropformer_live_rows: list[dict[str, Any]] = []

        cropformer_needed = bool(baselines_cfg.get("run_cropformer_only", True))
        if isinstance(matrix_cfgs, list):
            cropformer_needed = cropformer_needed or any(
                isinstance(spec, dict)
                and bool(spec.get("enabled", True))
                and str(spec.get("segmentor", spec.get("proposal_source", ""))).strip().lower() == "cropformer"
                for spec in matrix_cfgs
            )
        if cropformer_needed:
            for scene in scenes:
                live_frame_ids = sorted(set(int(frame_id) for frame_id in frame_ids_by_scene[scene]) | set(int(frame_id) for frame_id in matrix_frame_ids_by_scene[scene]))
                cropformer_row = _run_cropformer_live_scene(ctx, scene, live_frame_ids)
                cropformer_live_rows.append(cropformer_row)
                if cropformer_row.get("status") != "completed":
                    ctx.add_failure(
                        stage_name="baselines",
                        failure_type="CROPFORMER_LIVE_PREDICTION_FAILED",
                        severity="blocker",
                        scene_id=scene,
                        symptom=f"CropFormer live prediction failed: {cropformer_row.get('failure_reason')}",
                        suggested_repair="Inspect CropFormer subprocess log, config, weights, and CUDA environment; do not fall back to cached output_Cropformer masks for v105 baseline.",
                    )
            _write_records_json(out_dir / "cropformer_live_runtime_records.json", cropformer_live_rows, schema_version="stream4d_v105_cropformer_live_runtime_records_v1")
        cropformer_live_latency_sec = sum(float(row.get("latency_sec") or 0.0) for row in cropformer_live_rows)

        if bool(baselines_cfg.get("run_cropformer_only", True)):
            variant = "B0_cropformer_only"
            t0 = time.time()
            pipeline_root = out_dir / "pipelines" / variant
            mask_dirs = {scene: _cropformer_live_mask_dir(ctx, scene) for scene in scenes}
            support_summary = _write_pipeline_support_root(
                ctx=ctx,
                pipeline_root=pipeline_root,
                variant_id=variant,
                mask_root=_rel(_cropformer_live_root(ctx)),
                mask_dir_by_scene=mask_dirs,
                frame_ids_by_scene=frame_ids_by_scene,
                object_id_policy="frame_mask_is_object",
            )
            eval_rows: list[dict[str, Any]] = []
            for scene in scenes:
                eval_out = out_dir / "evaluation" / variant / scene
                try:
                    eval_rows.extend(
                        _run_v65_soma_eval(
                            scene_id=scene,
                            pipeline_root=pipeline_root,
                            output_root=eval_out,
                            stride=eval_stride,
                            max_frames=eval_max_frames,
                        )
                    )
                except Exception as exc:
                    ctx.add_failure(
                        stage_name="baselines",
                        failure_type="BASELINE_EVAL_FAILURE",
                        severity="blocker",
                        scene_id=scene,
                        symptom=f"{variant} v65 eval failed: {_safe_error(exc)}",
                        suggested_repair="Repair generated v65 support ledger/pipeline_summary contract, then rerun baselines.",
                    )
            for scene in scenes:
                video_path = out_dir / "videos" / f"{variant}_{scene}.mp4"
                video_ok = _write_baseline_overlay_video(ctx=ctx, scene_id=scene, frame_ids=frame_ids_by_scene[scene], mask_dir=mask_dirs[scene], video_path=video_path)
                video_rows.append({"variant_id": variant, "scene_id": scene, "video_path": _rel(video_path), "video_exists": video_ok})
            for row in eval_rows:
                metric_rows.append(
                    {
                        "schema_version": "stream4d_v105_baseline_metric_row_v1",
                        "variant_id": variant,
                        "scene_id": row.get("scene"),
                        "metric_scope": "window_dev_smoke_v65_soma",
                        "frame_count": row.get("frame_count"),
                        "MV_AP_window": row.get("AP"),
                        "MV_AP50_window": row.get("AP50"),
                        "MV_AP25_window": row.get("AP25"),
                        "summary_json": row.get("summary_json"),
                    }
                )
            latency_rows.append({"schema_version": "stream4d_v105_baseline_latency_row_v1", "variant_id": variant, "latency_sec": cropformer_live_latency_sec + (time.time() - t0), "stage": "cropformer_live_generation_plus_support_ledger_plus_v65_eval_plus_video"})
            memory_rows.append({"schema_version": "stream4d_v105_baseline_memory_row_v1", "variant_id": variant, "peak_gpu_memory_mb": "", "note": "CropFormer live subprocess memory is recorded in cropformer_live_runtime_records if available; no cached mask source used."})
            budget_rows.append({"schema_version": "stream4d_v105_baseline_budget_row_v1", "variant_id": variant, "SAM_call_count": 0, "mask_provider_call_count": 0, "tracker_call_count": 0, **support_summary})

        if bool(baselines_cfg.get("run_fastsam_only", False)):
            variant = "B1_fastsam_only"
            t0 = time.time()
            fastsam_mask_root = out_dir / "masks" / variant
            fastsam_scene_rows: list[dict[str, Any]] = []
            for scene in scenes:
                scene_row = _run_fastsam_scene(ctx, scene, frame_ids_by_scene[scene], fastsam_mask_root / scene / "mask")
                fastsam_scene_rows.append(scene_row)
                if scene_row["status"] != "completed":
                    ctx.add_failure(
                        stage_name="baselines",
                        failure_type="PROVIDER_UNAVAILABLE",
                        severity="warning",
                        scene_id=scene,
                        symptom=f"{variant} failed: {scene_row['failure_reason']}",
                        suggested_repair="Repair FastSAM checkpoint/import/runtime or leave B1 disabled; B0+B2 remain the Phase3 minimum gate.",
                    )
            _write_records_json(out_dir / f"{variant}_runtime_records.json", fastsam_scene_rows)
            if all(row["status"] == "completed" for row in fastsam_scene_rows):
                pipeline_root = out_dir / "pipelines" / variant
                mask_dirs = {scene: fastsam_mask_root / scene / "mask" for scene in scenes}
                support_summary = _write_pipeline_support_root(
                    ctx=ctx,
                    pipeline_root=pipeline_root,
                    variant_id=variant,
                    mask_root=_rel(fastsam_mask_root),
                    mask_dir_by_scene=mask_dirs,
                    frame_ids_by_scene=frame_ids_by_scene,
                    object_id_policy="frame_mask_is_object",
                )
                eval_rows = []
                for scene in scenes:
                    try:
                        eval_rows.extend(
                            _run_v65_soma_eval(
                                scene_id=scene,
                                pipeline_root=pipeline_root,
                                output_root=out_dir / "evaluation" / variant / scene,
                                stride=eval_stride,
                                max_frames=eval_max_frames,
                            )
                        )
                    except Exception as exc:
                        ctx.add_failure(
                            stage_name="baselines",
                            failure_type="BASELINE_EVAL_FAILURE",
                            severity="warning",
                            scene_id=scene,
                            symptom=f"{variant} v65 eval failed: {_safe_error(exc)}",
                            suggested_repair="Repair generated FastSAM support ledger/pipeline_summary contract before using B1 as a comparison.",
                        )
                for scene in scenes:
                    video_path = out_dir / "videos" / f"{variant}_{scene}.mp4"
                    video_ok = _write_baseline_overlay_video(ctx=ctx, scene_id=scene, frame_ids=frame_ids_by_scene[scene], mask_dir=mask_dirs[scene], video_path=video_path)
                    video_rows.append({"variant_id": variant, "scene_id": scene, "video_path": _rel(video_path), "video_exists": video_ok})
                for row in eval_rows:
                    metric_rows.append(
                        {
                            "schema_version": "stream4d_v105_baseline_metric_row_v1",
                            "variant_id": variant,
                            "scene_id": row.get("scene"),
                            "metric_scope": "window_dev_smoke_v65_soma",
                            "frame_count": row.get("frame_count"),
                            "MV_AP_window": row.get("AP"),
                            "MV_AP50_window": row.get("AP50"),
                            "MV_AP25_window": row.get("AP25"),
                            "summary_json": row.get("summary_json"),
                        }
                    )
                latency_rows.append({"schema_version": "stream4d_v105_baseline_latency_row_v1", "variant_id": variant, "latency_sec": time.time() - t0, "stage": "fastsam_generation_plus_v65_eval_plus_video"})
                memory_rows.append(
                    {
                        "schema_version": "stream4d_v105_baseline_memory_row_v1",
                        "variant_id": variant,
                        "peak_gpu_memory_mb": max([float(row["peak_gpu_memory_mb"]) for row in fastsam_scene_rows if row["peak_gpu_memory_mb"] != ""], default=0.0),
                    }
                )
                budget_rows.append(
                    {
                        "schema_version": "stream4d_v105_baseline_budget_row_v1",
                        "variant_id": variant,
                        "SAM_call_count": 0,
                        "mask_provider_call_count": sum(len(frame_ids_by_scene[scene]) for scene in scenes),
                        "tracker_call_count": 0,
                        **support_summary,
                    }
                )

        if bool(baselines_cfg.get("run_4dpm_sam2_gap_tracking", True)):
            variant = "B2_4dpm_sam2_gap_tracking"
            b2_max_frames = int(baselines_cfg.get("sam2_gap_max_frames_per_scene", eval_max_frames))
            b2_frame_ids_by_scene = {scene: _scene_frame_ids_for_baseline(ctx, scene, b2_max_frames) for scene in scenes}
            b2_mask_root = out_dir / "masks" / variant
            b2_scene_rows: list[dict[str, Any]] = []
            for scene in scenes:
                scene_row = _run_b2_4dpm_sam2_scene(ctx, scene, b2_frame_ids_by_scene[scene], b2_mask_root / scene / "mask")
                b2_scene_rows.append(scene_row)
                if scene_row["status"] != "completed":
                    ctx.add_failure(
                        stage_name="baselines",
                        failure_type="SAM_PROMPT_FAILURE",
                        severity="blocker",
                        scene_id=scene,
                        symptom=f"{variant} failed: {scene_row['failure_reason']}",
                        suggested_repair="Reduce SAM2 points/frames, verify 4DPM image format and checkpoint config, then rerun B2 before SGQ.",
                    )
            _write_records_json(out_dir / f"{variant}_runtime_records.json", b2_scene_rows)
            b2_scope_summary = {
                "schema_version": "stream4d_v105_b2_4dpm_sam2_gap_tracking_scope_summary_v1",
                "variant_id": variant,
                "plan_baseline_name": "baseline_4dpm_sam2_gap_tracking",
                "baseline_scope": "4DPM SAM2/SAMv2 run_video_overseg frontend over every configured stride input frame",
                "claim_allowed": "code-audited 4DPM SAM2 frontend baseline over the stride-5 evaluation sequence; every input frame is treated as a keyframe for B2 diagnostics",
                "claim_forbidden": [
                    "full official D4PM system reproduction",
                    "4D reconstruction backend",
                    "SE(3) primitive motion optimization",
                    "full-sequence D4PM run",
                    "4DPM kf_interval sparse-keyframe schedule inside B2",
                ],
                "fourdpm_frontend_code_backed": bool(b2_scene_rows) and all(
                    row.get("uses_fourdpm_frontend_run_video_overseg") is True and row.get("status") == "completed"
                    for row in b2_scene_rows
                ),
                "fourdpm_frontend_entrypoint": "third_party/4D_PM/frontend/segment/video_matcher.py::run_video_overseg",
                "full_d4pm_reconstruction_backend": False,
                "full_sequence_run": False,
                "window_limited_run": True,
                "baseline_run_granularity": "every frame selected by config baselines.eval_stride/run.frame_stride, capped by baselines.sam2_gap_max_frames_per_scene",
                "each_input_frame_treated_as_keyframe": True,
                "no_internal_keyframe_subsampling": True,
                "sam2_num_pts": int(baselines_cfg.get("sam2_gap_num_pts", 300)),
                "sam2_num_pts_active": int(baselines_cfg.get("sam2_gap_num_pts_active", 250)),
                "sam2_point_sampling": (
                    "deterministic_grid"
                    if bool(baselines_cfg.get("sam2_gap_deterministic_point_grid", False))
                    else "seeded_random"
                    if bool(baselines_cfg.get("deterministic_4dpm_sam2_gap", True))
                    else "random"
                ),
                "dev_smoke_scene_count": len(scenes),
                "dev_window_scene_count": len(scenes),
                "scene_ids": scenes,
                "frame_count_per_scene": {scene: len(b2_frame_ids_by_scene[scene]) for scene in scenes},
                "frame_ids_by_scene": {scene: [int(frame_id) for frame_id in b2_frame_ids_by_scene[scene]] for scene in scenes},
                "runtime_records_json": _rel(out_dir / f"{variant}_runtime_records.json"),
                "baseline_metric_records_json": _rel(out_dir / "baseline_metric_records.json"),
                "baseline_latency_records_json": _rel(out_dir / "baseline_latency_records.json"),
                "baseline_memory_records_json": _rel(out_dir / "baseline_memory_records.json"),
                "baseline_budget_records_json": _rel(out_dir / "baseline_budget_records.json"),
                "video_index_records_json": _rel(out_dir / "video_index_records.json"),
                "mask_root": _rel(b2_mask_root),
                "not_claimed": [
                    "full official 4DPM reconstruction system",
                    "4D reconstruction backend",
                    "SE(3) primitive motion optimization",
                    "full-sequence ScanNet run",
                    "MV_AP_scene/local2history evaluation",
                ],
                "scene_runtime_rows": b2_scene_rows,
            }
            if all(row["status"] == "completed" for row in b2_scene_rows):
                pipeline_root = out_dir / "pipelines" / variant
                mask_dirs = {scene: b2_mask_root / scene / "mask" for scene in scenes}
                support_summary = _write_pipeline_support_root(
                    ctx=ctx,
                    pipeline_root=pipeline_root,
                    variant_id=variant,
                    mask_root=_rel(b2_mask_root),
                    mask_dir_by_scene=mask_dirs,
                    frame_ids_by_scene=b2_frame_ids_by_scene,
                    object_id_policy="mask_id_is_track",
                )
                eval_rows = []
                for scene in scenes:
                    try:
                        eval_rows.extend(
                            _run_v65_soma_eval(
                                scene_id=scene,
                                pipeline_root=pipeline_root,
                                output_root=out_dir / "evaluation" / variant / scene,
                                stride=eval_stride,
                                max_frames=b2_max_frames,
                            )
                        )
                    except Exception as exc:
                        ctx.add_failure(
                            stage_name="baselines",
                            failure_type="BASELINE_EVAL_FAILURE",
                            severity="blocker",
                            scene_id=scene,
                            symptom=f"{variant} v65 eval failed: {_safe_error(exc)}",
                            suggested_repair="Repair B2 mask-root/support-ledger contract, then rerun baselines.",
                        )
                for scene in scenes:
                    video_path = out_dir / "videos" / f"{variant}_{scene}.mp4"
                    video_ok = _write_baseline_overlay_video(ctx=ctx, scene_id=scene, frame_ids=b2_frame_ids_by_scene[scene], mask_dir=mask_dirs[scene], video_path=video_path)
                    video_rows.append({"variant_id": variant, "scene_id": scene, "video_path": _rel(video_path), "video_exists": video_ok})
                for row in eval_rows:
                    metric_rows.append(
                        {
                            "schema_version": "stream4d_v105_baseline_metric_row_v1",
                            "variant_id": variant,
                            "scene_id": row.get("scene"),
                            "metric_scope": "window_dev_smoke_v65_soma",
                            "frame_count": row.get("frame_count"),
                            "MV_AP_window": row.get("AP"),
                            "MV_AP50_window": row.get("AP50"),
                            "MV_AP25_window": row.get("AP25"),
                            "summary_json": row.get("summary_json"),
                        }
                    )
                latency_rows.append({"schema_version": "stream4d_v105_baseline_latency_row_v1", "variant_id": variant, "latency_sec": sum(float(row["latency_sec"]) for row in b2_scene_rows), "stage": "4dpm_sam2_gap_tracking_plus_v65_eval_plus_video"})
                memory_rows.append({"schema_version": "stream4d_v105_baseline_memory_row_v1", "variant_id": variant, "peak_gpu_memory_mb": max([float(row["peak_gpu_memory_mb"]) for row in b2_scene_rows if row["peak_gpu_memory_mb"] != ""], default=0.0)})
                budget_rows.append({"schema_version": "stream4d_v105_baseline_budget_row_v1", "variant_id": variant, "SAM_call_count": "recorded_indirectly_by_4DPM_runtime_rows", "mask_provider_call_count": len(scenes), "tracker_call_count": len(scenes), **support_summary})
            _write_json(out_dir / f"{variant}_scope_summary.json", b2_scope_summary)

        if bool(baselines_cfg.get("run_full_sam31_tracking", False)):
            variant = "B3_full_sam31_tracking"
            t0 = time.time()
            b3_mask_root = out_dir / "masks" / variant
            b3_scene_rows: list[dict[str, Any]] = []
            for scene in scenes:
                scene_row = _run_full_sam31_reference_scene(ctx, scene, frame_ids_by_scene[scene], b3_mask_root / scene / "mask")
                b3_scene_rows.append(scene_row)
                if scene_row.get("status") == "failed":
                    ctx.add_failure(
                        stage_name="baselines",
                        failure_type="SAM31_FULL_REFERENCE_UNAVAILABLE",
                        severity="warning",
                        scene_id=scene,
                        symptom=f"{variant} failed: {scene_row.get('failure_reason')}",
                        suggested_repair="Reduce SAM3.1 objects per session/query cap or crop anchor proposals, then rerun B3 full tracking reference.",
                    )
            _write_records_json(out_dir / f"{variant}_runtime_records.json", b3_scene_rows)
            b3_scope_summary = {
                "schema_version": "stream4d_v105_b3_full_sam31_scope_summary_v1",
                "variant_id": variant,
                "plan_baseline_name": "B3_full_sam31_tracking",
                "baseline_scope": "SAM3.1 multiplex video object-prompt tracking over CropFormer proposal anchors",
                "claim_allowed": "real SAM3.1 multiplex video tracking path using start_session/add_prompt(obj_id)/propagate_in_video over the full baseline window",
                "claim_forbidden": [
                    "official full-sequence all-scene SAM3.1 benchmark",
                    "uncapped all-proposal reference when query_limit is hit",
                ],
                "eligible_for_phase4_speed_gate": bool(b3_scene_rows) and all(row.get("eligible_for_phase4_speed_gate") is True for row in b3_scene_rows),
                "scene_runtime_records_json": _rel(out_dir / f"{variant}_runtime_records.json"),
                "reference_scope_by_scene": {row.get("scene_id"): row.get("reference_scope", "") for row in b3_scene_rows},
                "status_by_scene": {row.get("scene_id"): row.get("status") for row in b3_scene_rows},
                "query_count_by_scene": {row.get("scene_id"): row.get("query_count") for row in b3_scene_rows},
                "query_limit_by_scene": {row.get("scene_id"): row.get("query_limit") for row in b3_scene_rows},
                "sam31_video_session_count_by_scene": {row.get("scene_id"): row.get("sam31_video_session_count") for row in b3_scene_rows},
                "tracked_object_count_by_scene": {row.get("scene_id"): row.get("tracked_object_count") for row in b3_scene_rows},
                "capped_by_query_limit_by_scene": {row.get("scene_id"): row.get("capped_by_query_limit") for row in b3_scene_rows},
            }
            runnable_b3 = bool(b3_scene_rows) and all(row.get("status") in {"completed", "partial"} for row in b3_scene_rows)
            if runnable_b3:
                pipeline_root = out_dir / "pipelines" / variant
                mask_dirs = {scene: b3_mask_root / scene / "mask" for scene in scenes}
                support_summary = _write_pipeline_support_root(
                    ctx=ctx,
                    pipeline_root=pipeline_root,
                    variant_id=variant,
                    mask_root=_rel(b3_mask_root),
                    mask_dir_by_scene=mask_dirs,
                    frame_ids_by_scene=frame_ids_by_scene,
                    object_id_policy="mask_id_is_track",
                )
                eval_rows = []
                for scene in scenes:
                    try:
                        eval_rows.extend(
                            _run_v65_soma_eval(
                                scene_id=scene,
                                pipeline_root=pipeline_root,
                                output_root=out_dir / "evaluation" / variant / scene,
                                stride=eval_stride,
                                max_frames=eval_max_frames,
                            )
                        )
                    except Exception as exc:
                        ctx.add_failure(
                            stage_name="baselines",
                            failure_type="BASELINE_EVAL_FAILURE",
                            severity="warning",
                            scene_id=scene,
                            symptom=f"{variant} v65 eval failed: {_safe_error(exc)}",
                            suggested_repair="Repair B3 mask-root/support-ledger contract before using B3 as an acceleration comparator.",
                        )
                for scene in scenes:
                    video_path = out_dir / "videos" / f"{variant}_{scene}.mp4"
                    video_ok = _write_baseline_overlay_video(ctx=ctx, scene_id=scene, frame_ids=frame_ids_by_scene[scene], mask_dir=mask_dirs[scene], video_path=video_path)
                    video_rows.append({"variant_id": variant, "scene_id": scene, "video_path": _rel(video_path), "video_exists": video_ok})
                for row in eval_rows:
                    metric_rows.append(
                        {
                            "schema_version": "stream4d_v105_baseline_metric_row_v1",
                            "variant_id": variant,
                            "scene_id": row.get("scene"),
                            "metric_scope": "window_dev_smoke_v65_soma",
                            "frame_count": row.get("frame_count"),
                            "MV_AP_window": row.get("AP"),
                            "MV_AP50_window": row.get("AP50"),
                            "MV_AP25_window": row.get("AP25"),
                            "summary_json": row.get("summary_json"),
                        }
                    )
                latency_rows.append({"schema_version": "stream4d_v105_baseline_latency_row_v1", "variant_id": variant, "latency_sec": time.time() - t0, "stage": "sam31_multiplex_video_tracking_plus_v65_eval_plus_video"})
                memory_rows.append({"schema_version": "stream4d_v105_baseline_memory_row_v1", "variant_id": variant, "peak_gpu_memory_mb": max([float(row["peak_gpu_memory_mb"]) for row in b3_scene_rows if row.get("peak_gpu_memory_mb") not in {"", None}], default=0.0)})
                budget_rows.append({"schema_version": "stream4d_v105_baseline_budget_row_v1", "variant_id": variant, "SAM_call_count": sum(int(row.get("query_count") or 0) for row in b3_scene_rows), "mask_provider_call_count": sum(int(row.get("query_count") or 0) for row in b3_scene_rows), "tracker_call_count": sum(int(row.get("sam31_video_session_count") or 0) for row in b3_scene_rows), **support_summary})
            _write_json(out_dir / f"{variant}_scope_summary.json", b3_scope_summary)

        if bool(baselines_cfg.get("run_baseline_matrix_video_tracking", False)) and isinstance(matrix_cfgs, list):
            matrix_runtime_index: list[dict[str, Any]] = []
            for raw_spec in matrix_cfgs:
                if not isinstance(raw_spec, dict) or not bool(raw_spec.get("enabled", True)):
                    continue
                proposal_source = str(raw_spec.get("segmentor", raw_spec.get("proposal_source", "cropformer"))).strip().lower()
                tracker_provider = str(raw_spec.get("tracker", "sam31_multiplex")).strip().lower()
                variant = str(raw_spec.get("variant_id") or f"BMatrix_{proposal_source}_{tracker_provider}").strip()
                t0 = time.time()
                mask_root = out_dir / "masks" / variant
                segmentor_source_key = _promptable_segmentor_source_key(proposal_source)
                segmentor_scene_rows: list[dict[str, Any]] = []
                if segmentor_source_key is not None:
                    proposal_mask_root = out_dir / "masks" / f"_proposal_{segmentor_source_key}"
                    for scene in scenes:
                        proposal_row = _run_promptable_segmentor_proposals_scene(
                            ctx,
                            scene,
                            matrix_frame_ids_by_scene[scene],
                            proposal_source,
                            proposal_mask_root / scene / "mask",
                        )
                        segmentor_scene_rows.append(proposal_row)
                        if proposal_row.get("status") == "failed":
                            ctx.add_failure(
                                stage_name="baselines",
                                failure_type="BASELINE_SEGMENTOR_PROPOSAL_FAILED",
                                severity="warning",
                                scene_id=scene,
                                symptom=f"{variant} segmentor={proposal_source} proposal generation failed: {proposal_row.get('failure_reason')}",
                                suggested_repair="Inspect promptable segmentor checkpoint/API and reduce promptable_segmentor_* caps before trusting this baseline.",
                            )
                    _write_records_json(out_dir / f"{variant}_segmentor_runtime_records.json", segmentor_scene_rows)
                scene_rows: list[dict[str, Any]] = []
                for scene in scenes:
                    if tracker_provider in {"sam31", "sam31_multiplex", "sam3.1"}:
                        scene_row = _run_full_sam31_reference_scene(
                            ctx,
                            scene,
                            matrix_frame_ids_by_scene[scene],
                            mask_root / scene / "mask",
                            variant_id=variant,
                            proposal_source=proposal_source,
                        )
                    elif tracker_provider in {"sam2_video", "edgetam_video"}:
                        scene_row = _run_mask_prompt_video_tracker_scene(
                            ctx,
                            scene,
                            matrix_frame_ids_by_scene[scene],
                            mask_root / scene / "mask",
                            variant_id=variant,
                            proposal_source=proposal_source,
                            tracker_provider=tracker_provider,
                        )
                    else:
                        scene_row = {
                            "schema_version": "stream4d_v105_baseline_matrix_runtime_row_v1",
                            "variant_id": variant,
                            "status": "failed",
                            "scene_id": scene,
                            "frame_ids": ";".join(str(int(frame_id)) for frame_id in matrix_frame_ids_by_scene[scene]),
                            "frame_count": len(matrix_frame_ids_by_scene[scene]),
                            "frame_output_count": 0,
                            "segmentor_provider": proposal_source,
                            "tracker_provider": tracker_provider,
                            "failure_reason": f"unsupported baseline matrix tracker={tracker_provider}",
                        }
                    scene_rows.append(scene_row)
                    if scene_row.get("status") == "failed":
                        ctx.add_failure(
                            stage_name="baselines",
                            failure_type="BASELINE_MATRIX_VARIANT_FAILED",
                            severity="warning",
                            scene_id=scene,
                            symptom=f"{variant} failed: {scene_row.get('failure_reason')}",
                            suggested_repair="Inspect provider checkpoint/API and reduce baseline_matrix query/object caps before trusting this comparison.",
                        )
                runtime_path = out_dir / f"{variant}_runtime_records.json"
                _write_records_json(runtime_path, scene_rows)
                mask_dirs = {scene: mask_root / scene / "mask" for scene in scenes}
                mask_frames_exist = all(mask_dirs[scene].exists() and any(mask_dirs[scene].glob("*.png")) for scene in scenes)
                eval_rows = []
                support_summary: dict[str, Any] = {}
                if mask_frames_exist:
                    pipeline_root = out_dir / "pipelines" / variant
                    support_summary = _write_pipeline_support_root(
                        ctx=ctx,
                        pipeline_root=pipeline_root,
                        variant_id=variant,
                        mask_root=_rel(mask_root),
                        mask_dir_by_scene=mask_dirs,
                        frame_ids_by_scene=matrix_frame_ids_by_scene,
                        object_id_policy="mask_id_is_track",
                    )
                    for scene in scenes:
                        try:
                            eval_rows.extend(
                                _run_v65_soma_eval(
                                    scene_id=scene,
                                    pipeline_root=pipeline_root,
                                    output_root=out_dir / "evaluation" / variant / scene,
                                    stride=eval_stride,
                                    max_frames=matrix_max_frames,
                                )
                            )
                        except Exception as exc:
                            ctx.add_failure(
                                stage_name="baselines",
                                failure_type="BASELINE_EVAL_FAILURE",
                                severity="warning",
                                scene_id=scene,
                                symptom=f"{variant} v65 eval failed: {_safe_error(exc)}",
                                suggested_repair="Repair matrix mask-root/support-ledger contract before using this baseline comparison.",
                            )
                    for scene in scenes:
                        video_path = out_dir / "videos" / f"{variant}_{scene}.mp4"
                        video_ok = _write_baseline_overlay_video(
                            ctx=ctx,
                            scene_id=scene,
                            frame_ids=matrix_frame_ids_by_scene[scene],
                            mask_dir=mask_dirs[scene],
                            video_path=video_path,
                        )
                        video_rows.append({"variant_id": variant, "scene_id": scene, "video_path": _rel(video_path), "video_exists": video_ok})
                    for row in eval_rows:
                        metric_rows.append(
                            {
                                "schema_version": "stream4d_v105_baseline_metric_row_v1",
                                "variant_id": variant,
                                "scene_id": row.get("scene"),
                                "metric_scope": "window_dev_smoke_v65_soma",
                                "frame_count": row.get("frame_count"),
                                "MV_AP_window": row.get("AP"),
                                "MV_AP50_window": row.get("AP50"),
                                "MV_AP25_window": row.get("AP25"),
                                "summary_json": row.get("summary_json"),
                            }
                        )
                segmentor_extra_latency_sec = cropformer_live_latency_sec if proposal_source == "cropformer" else 0.0
                latency_rows.append(
                    {
                        "schema_version": "stream4d_v105_baseline_latency_row_v1",
                        "variant_id": variant,
                        "latency_sec": segmentor_extra_latency_sec + (time.time() - t0),
                        "stage": f"{proposal_source}_segmentor_plus_{tracker_provider}_tracker_plus_v65_eval_plus_video",
                    }
                )
                memory_rows.append(
                    {
                        "schema_version": "stream4d_v105_baseline_memory_row_v1",
                        "variant_id": variant,
                        "peak_gpu_memory_mb": max(
                            [float(row.get("peak_gpu_memory_mb")) for row in scene_rows + segmentor_scene_rows if row.get("peak_gpu_memory_mb") not in {"", None}],
                            default=0.0,
                        ),
                    }
                )
                budget_rows.append(
                    {
                        "schema_version": "stream4d_v105_baseline_budget_row_v1",
                        "variant_id": variant,
                        "SAM_call_count": sum(int(row.get("query_count") or 0) for row in scene_rows)
                        + sum(int(row.get("written_mask_count") or row.get("accepted_mask_count") or 0) for row in segmentor_scene_rows),
                        "mask_provider_call_count": sum(int(row.get("query_count") or 0) for row in scene_rows)
                        + sum(int(row.get("frame_count") or 0) for row in segmentor_scene_rows),
                        "tracker_call_count": sum(int(row.get("sam31_video_session_count") or row.get("tracker_session_count") or 0) for row in scene_rows),
                        **support_summary,
                    }
                )
                matrix_runtime_index.append(
                    {
                        "variant_id": variant,
                        "segmentor_provider": proposal_source,
                        "tracker_provider": tracker_provider,
                        "runtime_records_json": _rel(runtime_path),
                        "segmentor_runtime_records_json": _rel(out_dir / f"{variant}_segmentor_runtime_records.json") if segmentor_scene_rows else "",
                        "segmentor_status_by_scene": {row.get("scene_id"): row.get("status") for row in segmentor_scene_rows},
                        "status_by_scene": {row.get("scene_id"): row.get("status") for row in scene_rows},
                        "frame_count_by_scene": {row.get("scene_id"): row.get("frame_count") for row in scene_rows},
                        "frame_output_count_by_scene": {row.get("scene_id"): row.get("frame_output_count") for row in scene_rows},
                        "nonzero_output_frame_count_by_scene": {row.get("scene_id"): row.get("nonzero_output_frame_count") for row in scene_rows},
                        "zero_output_frame_ids_by_scene": {row.get("scene_id"): row.get("zero_output_frame_ids", []) for row in scene_rows},
                        "all_input_frames_have_nonzero_masks_by_scene": {row.get("scene_id"): row.get("all_input_frames_have_nonzero_masks") for row in scene_rows},
                        "query_count_by_scene": {row.get("scene_id"): row.get("query_count") for row in scene_rows},
                        "mask_frames_exist": mask_frames_exist,
                    }
                )
            baseline_matrix_summary = {
                "schema_version": "stream4d_v105_baseline_matrix_summary_v1",
                "enabled": True,
                "all_visualizations_use_only_input_frames": True,
                "each_input_frame_treated_as_anchor": True,
                "frame_ids_by_scene": {scene: [int(frame_id) for frame_id in matrix_frame_ids_by_scene[scene]] for scene in scenes},
                "variant_count": len(matrix_runtime_index),
                "variants": matrix_runtime_index,
            }
            _write_json(out_dir / "baseline_matrix_scope_summary.json", baseline_matrix_summary)

        _write_records_json(out_dir / "baseline_metric_records.json", metric_rows)
        _write_records_json(out_dir / "baseline_latency_records.json", latency_rows)
        _write_records_json(out_dir / "baseline_memory_records.json", memory_rows)
        _write_records_json(out_dir / "baseline_budget_records.json", budget_rows)
        _write_records_json(out_dir / "video_index_records.json", video_rows)
        expected_visual_frames = max(
            int(eval_max_frames),
            int(baselines_cfg.get("sam2_gap_max_frames_per_scene", eval_max_frames)),
            int(baselines_cfg.get("baseline_matrix_max_frames_per_scene", eval_max_frames)),
        )
        full_frame_visual_audit = _write_full_frame_visual_audit(
            video_rows=video_rows,
            audit_root=out_dir / "full_frame_visual_audit",
            expected_frame_count=expected_visual_frames,
        )
        ctx.gates["phase3_baseline_metric_records"] = metric_rows
        variants_with_metrics = {row["variant_id"] for row in metric_rows}
        variants_with_videos = {row["variant_id"] for row in video_rows if row.get("video_exists")}
        phase3_pass = {"B0_cropformer_only", "B2_4dpm_sam2_gap_tracking"}.issubset(variants_with_metrics) and {"B0_cropformer_only", "B2_4dpm_sam2_gap_tracking"}.issubset(variants_with_videos)
        baseline_matrix_variant_ids = (
            [str(row.get("variant_id")) for row in baseline_matrix_summary.get("variants", [])]
            if baseline_matrix_summary is not None
            else []
        )
        summary = {
            "schema_version": "stream4d_v105_baseline_summary_v1",
            "phase3_baseline_pass": bool(phase3_pass),
            "metric_scope": "dev_smoke_v65_soma_window",
            "eval_stride": eval_stride,
            "eval_max_frames_per_scene": eval_max_frames,
            "metric_variant_count": len(variants_with_metrics),
            "video_variant_count": len(variants_with_videos),
            "video_variant_ids": sorted(variants_with_videos),
            "failure_count": len([row for row in ctx.failure_rows if row.get("stage_name") == "baselines"]),
            "formal_evaluator": ctx.config.get("evaluation", {}).get("evaluator"),
            "cache_read_count": 0,
            "full_frame_visual_audit_json": _rel(out_dir / "full_frame_visual_audit" / "full_frame_visual_audit.json"),
            "full_frame_visual_audit": full_frame_visual_audit,
            "no_interval_sampling": True,
        }
        if b2_scope_summary is not None:
            summary["b2_scope_summary_json"] = _rel(out_dir / "B2_4dpm_sam2_gap_tracking_scope_summary.json")
            summary["b2_claim_allowed"] = b2_scope_summary["claim_allowed"]
            summary["b2_full_d4pm_reconstruction_backend"] = b2_scope_summary["full_d4pm_reconstruction_backend"]
            summary["b2_full_sequence_run"] = b2_scope_summary["full_sequence_run"]
        if b3_scope_summary is not None:
            summary["b3_scope_summary_json"] = _rel(out_dir / "B3_full_sam31_tracking_scope_summary.json")
            summary["b3_eligible_for_phase4_speed_gate"] = b3_scope_summary["eligible_for_phase4_speed_gate"]
            summary["b3_status_by_scene"] = b3_scope_summary["status_by_scene"]
        if baseline_matrix_summary is not None:
            summary["baseline_matrix_scope_summary_json"] = _rel(out_dir / "baseline_matrix_scope_summary.json")
            summary["baseline_matrix_variant_count"] = baseline_matrix_summary["variant_count"]
            summary["baseline_matrix_variant_ids"] = baseline_matrix_variant_ids
            summary["baseline_matrix_all_visualizations_use_only_input_frames"] = baseline_matrix_summary["all_visualizations_use_only_input_frames"]
        _write_json(out_dir / "baseline_summary.json", summary)
        if not phase3_pass:
            ctx.add_failure(
                stage_name="baselines",
                failure_type="BASELINE_PHASE3_INCOMPLETE",
                severity="blocker",
                symptom=f"Phase3 baseline gate failed: {summary}",
                suggested_repair="Repair missing B0/B2 metrics or videos before SGQ local query planner.",
            )
        ctx.gates["phase3_baseline_pass"] = bool(phase3_pass)
        ctx.gates["phase3_baseline_summary"] = summary
        ctx.write_summary()


def run_sgq_local(ctx: PipelineContext) -> None:
    with StageTimer(ctx, "sgq_local"):
        if not ctx.gates.get("phase3_baseline_pass", False):
            raise RuntimeError("Phase 3 baselines must pass before SGQ local.")
        out_dir = ctx.output_root / "sgq_local"
        local_cfg = ctx.config.get("local", {})
        repair_mode = str(local_cfg.get("sgq_repair_mode", "strict_box")).strip()
        refiner_provider = str(local_cfg.get("sgq_refiner_provider", "sam2")).strip().lower()
        macro_residual_repair = repair_mode in {
            "r22_macro_residual_sam2_mask_prompt",
            "macro_residual_sam2_mask_prompt",
        }
        post_sam_acceptance_repair = repair_mode in {
            "r22_macro_residual_sam2_mask_prompt",
            "r17_post_sam_acceptance_merge_to_b2_alltracker_propagation",
            "r18_high_overlap_b2_refine_post_sam_acceptance",
            "r19_dual_acceptance_high_merge_b2_refine",
            "r20_r17_post_sam_acceptance_relaxed_alltracker",
            "post_sam_acceptance_merge_to_b2_alltracker_propagation",
        }
        alltracker_mask_propagation_repair = repair_mode in {
            "r22_macro_residual_sam2_mask_prompt",
            "r12_alltracker_mask_propagation",
            "r13_sam2_maskinput_alltracker_propagation",
            "r14_merge_to_b2_alltracker_propagation",
            "r15_candidate_direct_merge_to_b2_alltracker_propagation",
            "r16_candidate_direct_new_object_alltracker_propagation",
            "r17_post_sam_acceptance_merge_to_b2_alltracker_propagation",
            "r18_high_overlap_b2_refine_post_sam_acceptance",
            "r19_dual_acceptance_high_merge_b2_refine",
            "r20_r17_post_sam_acceptance_relaxed_alltracker",
            "alltracker_mask_propagation",
            "alltracker_propagation_fastsam_depth_overlap",
        }
        merge_to_b2_repair = repair_mode in {
            "r22_macro_residual_sam2_mask_prompt",
            "r14_merge_to_b2_alltracker_propagation",
            "r15_candidate_direct_merge_to_b2_alltracker_propagation",
            "r17_post_sam_acceptance_merge_to_b2_alltracker_propagation",
            "r18_high_overlap_b2_refine_post_sam_acceptance",
            "r19_dual_acceptance_high_merge_b2_refine",
            "r20_r17_post_sam_acceptance_relaxed_alltracker",
            "merge_to_b2_alltracker_propagation",
        }
        candidate_direct_merge_repair = repair_mode in {
            "r15_candidate_direct_merge_to_b2_alltracker_propagation",
            "r17_post_sam_acceptance_merge_to_b2_alltracker_propagation",
            "r18_high_overlap_b2_refine_post_sam_acceptance",
            "r19_dual_acceptance_high_merge_b2_refine",
            "r20_r17_post_sam_acceptance_relaxed_alltracker",
            "candidate_direct_merge_to_b2_alltracker_propagation",
        }
        candidate_direct_output_repair = repair_mode in {
            "r15_candidate_direct_merge_to_b2_alltracker_propagation",
            "r16_candidate_direct_new_object_alltracker_propagation",
            "r17_post_sam_acceptance_merge_to_b2_alltracker_propagation",
            "r18_high_overlap_b2_refine_post_sam_acceptance",
            "r19_dual_acceptance_high_merge_b2_refine",
            "r20_r17_post_sam_acceptance_relaxed_alltracker",
            "candidate_direct_merge_to_b2_alltracker_propagation",
            "candidate_direct_new_object_alltracker_propagation",
        }
        alltracker_visibility_repair = repair_mode in {"alltracker_visibility_fastsam_depth_overlap", "r11_alltracker_visibility_fastsam_depth_overlap", "alltracker_visibility"} or alltracker_mask_propagation_repair
        fastsam_proposal_repair = repair_mode in {"fastsam_depth_overlap_multimask", "r10_fastsam_depth_overlap_multimask", "fastsam_proposal"} or alltracker_visibility_repair or alltracker_mask_propagation_repair
        wta_override_repair = repair_mode in {"depth_overlap_multimask_override", "r7_depth_overlap_multimask_override", "wta_override"}
        candidate_overlap_multimask_repair = repair_mode in {"candidate_overlap_multimask", "r6_candidate_overlap_multimask", "depth_overlap_multimask"} or wta_override_repair or fastsam_proposal_repair
        depth_structure_repair = repair_mode in {"depth_structure_veto", "r5_depth_structure_veto", "depth_structure"} or candidate_overlap_multimask_repair
        point_veto_repair = repair_mode in {"point_veto", "persistent_point_veto", "r3_point_veto"}
        mask_prompt_repair = repair_mode in {"mask_prompt_refine", "r4_mask_prompt_refine", "mask_prompt"} or depth_structure_repair
        precision_repair = point_veto_repair or mask_prompt_repair
        strict_repair = bool(local_cfg.get("sgq_strict_repair_enabled", True))
        if repair_mode == "r22_macro_residual_sam2_mask_prompt":
            variant = "SGQ5_r22_macro_residual_sam2_mask_prompt_sam2_smoke"
        elif repair_mode == "r20_r17_post_sam_acceptance_relaxed_alltracker":
            variant = "SGQ5_r20_r17_post_sam_acceptance_relaxed_alltracker_sam2_smoke"
        elif repair_mode == "r19_dual_acceptance_high_merge_b2_refine":
            variant = "SGQ5_r19_dual_acceptance_high_merge_b2_refine_sam2_smoke"
        elif repair_mode == "r18_high_overlap_b2_refine_post_sam_acceptance":
            variant = "SGQ5_r18_high_overlap_b2_refine_post_sam_acceptance_sam2_smoke"
        elif repair_mode == "r17_post_sam_acceptance_merge_to_b2_alltracker_propagation":
            variant = "SGQ5_r17_post_sam_acceptance_merge_to_b2_alltracker_propagation_sam2_smoke"
        elif repair_mode == "r16_candidate_direct_new_object_alltracker_propagation":
            variant = "SGQ5_r16_candidate_direct_new_object_alltracker_propagation_sam2_smoke"
        elif repair_mode == "r15_candidate_direct_merge_to_b2_alltracker_propagation":
            variant = "SGQ5_r15_candidate_direct_merge_to_b2_alltracker_propagation_sam2_smoke"
        elif repair_mode == "r14_merge_to_b2_alltracker_propagation":
            variant = "SGQ5_r14_merge_to_b2_alltracker_propagation_sam2_smoke"
        elif repair_mode == "r13_sam2_maskinput_alltracker_propagation":
            variant = "SGQ5_r13_sam2_maskinput_alltracker_propagation_sam2_smoke"
        elif alltracker_mask_propagation_repair:
            variant = "SGQ5_r12_alltracker_mask_propagation_fastsam_depth_overlap_sam2_smoke"
        elif alltracker_visibility_repair:
            variant = "SGQ5_r11_alltracker_visibility_fastsam_depth_overlap_sam2_smoke"
        elif fastsam_proposal_repair:
            variant = "SGQ5_r10_fastsam_depth_overlap_multimask_wta_sam2_smoke"
        elif wta_override_repair:
            variant = "SGQ5_r7_depth_overlap_multimask_override_sam2_smoke"
        elif candidate_overlap_multimask_repair:
            variant = "SGQ5_r6_depth_overlap_multimask_wta_sam2_smoke"
        elif depth_structure_repair:
            variant = "SGQ5_r5_depth_structure_veto_mask_prompt_sam2_smoke"
        elif mask_prompt_repair:
            variant = "SGQ5_r4_mask_prompt_refine_sam2_smoke"
        elif point_veto_repair:
            variant = "SGQ5_r3_persistent_point_veto_sam2_smoke"
        elif strict_repair:
            variant = "SGQ5_r2_strict_residual_sam2_box_smoke"
        else:
            variant = "SGQ5_sam2_fallback_cropformer_residual_topk_smoke"
        if refiner_provider != "sam2":
            variant = variant.replace("_sam2_smoke", f"_{refiner_provider}_smoke").replace("SGQ5_sam2_fallback", f"SGQ5_{refiner_provider}_fallback")
        scenes = [str(scene) for scene in ctx.config.get("run", {}).get("scenes", [])]
        baselines_cfg = ctx.config.get("baselines", {})
        eval_stride = int(baselines_cfg.get("eval_stride", ctx.config.get("run", {}).get("frame_stride", 5)))
        eval_max_frames = int(baselines_cfg.get("sam2_gap_max_frames_per_scene", baselines_cfg.get("eval_max_frames_per_scene", 4)))
        frame_ids_by_scene = {scene: _scene_frame_ids_for_baseline(ctx, scene, eval_max_frames) for scene in scenes}
        scannet_root = _as_repo_path(ctx.config.get("paths", {}).get("scannet_processed_root")) or (STREAM3D_ROOT / "data" / "scannet" / "processed")
        b2_mask_root = ctx.output_root / "baselines" / "masks" / "B2_4dpm_sam2_gap_tracking"
        sgq_mask_root = out_dir / "masks" / variant
        query_budget = int(local_cfg.get("query_budget_per_chunk", 16))
        if macro_residual_repair:
            query_budget = int(local_cfg.get("sgq_macro_residual_query_budget_per_scene", 8))
        elif depth_structure_repair:
            query_budget = int(local_cfg.get("sgq_depth_structure_query_budget_per_scene", local_cfg.get("sgq_mask_prompt_query_budget_per_scene", 4)))
        elif mask_prompt_repair:
            query_budget = int(local_cfg.get("sgq_mask_prompt_query_budget_per_scene", local_cfg.get("sgq_point_veto_query_budget_per_scene", 4)))
        elif point_veto_repair:
            query_budget = int(local_cfg.get("sgq_point_veto_query_budget_per_scene", 4))
        elif strict_repair:
            query_budget = int(local_cfg.get("sgq_strict_query_budget_per_scene", 4))
        query_budget_per_scene = int(query_budget)
        total_query_budget_available = int(query_budget_per_scene * max(len(scenes), 1))
        if depth_structure_repair:
            min_expected_gain = float(local_cfg.get("sgq_depth_structure_min_expected_gain", local_cfg.get("sgq_mask_prompt_min_expected_gain", 0.0005)))
            max_conflict_risk = float(local_cfg.get("sgq_depth_structure_max_conflict_risk", local_cfg.get("sgq_mask_prompt_max_conflict_risk", 0.65)))
        elif mask_prompt_repair:
            min_expected_gain = float(local_cfg.get("sgq_mask_prompt_min_expected_gain", local_cfg.get("sgq_point_veto_min_expected_gain", 0.0005)))
            max_conflict_risk = float(local_cfg.get("sgq_mask_prompt_max_conflict_risk", local_cfg.get("sgq_point_veto_max_conflict_risk", 0.65)))
        elif point_veto_repair:
            min_expected_gain = float(local_cfg.get("sgq_point_veto_min_expected_gain", 0.0005))
            max_conflict_risk = float(local_cfg.get("sgq_point_veto_max_conflict_risk", 0.65))
        else:
            min_expected_gain = float(local_cfg.get("sgq_strict_min_expected_gain", 0.01)) if strict_repair else 0.0
            max_conflict_risk = float(local_cfg.get("sgq_strict_max_conflict_risk", 0.35)) if strict_repair else 0.65
        if macro_residual_repair:
            min_expected_gain = float(local_cfg.get("sgq_macro_residual_min_expected_gain", 0.005))
            max_conflict_risk = float(local_cfg.get("sgq_macro_residual_max_conflict_risk", 0.35))
        point_veto_max_area_ratio = float(local_cfg.get("sgq_point_veto_max_area_ratio", 0.12))
        point_veto_min_persistence = int(local_cfg.get("sgq_point_veto_min_persistence_count", 2))
        fastsam_min_persistence = int(local_cfg.get("sgq_fastsam_min_persistence_count", 1))
        point_veto_min_output_area_ratio = float(local_cfg.get("sgq_point_veto_min_output_area_ratio", 0.0002))
        point_veto_min_candidate_fraction = float(local_cfg.get("sgq_point_veto_min_candidate_fraction", 0.05))
        mask_prompt_lowres_size = int(local_cfg.get("sgq_mask_prompt_lowres_size", 256))
        mask_prompt_positive_logit = float(local_cfg.get("sgq_mask_prompt_positive_logit", 10.0))
        mask_prompt_negative_logit = float(local_cfg.get("sgq_mask_prompt_negative_logit", -10.0))
        depth_structure_min_iqr_mm = float(local_cfg.get("sgq_depth_structure_min_iqr_mm", 120.0))
        depth_structure_min_valid_fraction = float(local_cfg.get("sgq_depth_structure_min_valid_fraction", 0.25))
        post_sam_acceptance_min_score = float(local_cfg.get("sgq_post_sam_acceptance_min_score", 0.5))
        post_sam_acceptance_min_overlap_iou = float(local_cfg.get("sgq_post_sam_acceptance_min_candidate_overlap_iou", 0.45))
        post_sam_acceptance_min_depth_iqr_mm = float(local_cfg.get("sgq_post_sam_acceptance_min_depth_iqr_mm", 500.0))
        post_sam_acceptance_require_merge = bool(local_cfg.get("sgq_post_sam_acceptance_require_merge_to_b2", True))
        post_sam_acceptance_allow_high_merge = bool(local_cfg.get("sgq_post_sam_acceptance_allow_high_merge_to_b2", False))
        post_sam_acceptance_high_merge_iou = float(local_cfg.get("sgq_post_sam_acceptance_high_merge_iou", 0.80))
        min_area_ratio = float(local_cfg.get("residual_tube_min_area_ratio", 0.002))
        macro_residual_min_area_ratio = float(local_cfg.get("sgq_macro_residual_min_area_ratio", min_area_ratio))
        macro_residual_max_area_ratio = float(local_cfg.get("sgq_macro_residual_max_area_ratio", 0.45))
        macro_residual_min_depth_iqr_mm = float(local_cfg.get("sgq_macro_residual_min_depth_iqr_mm", post_sam_acceptance_min_depth_iqr_mm))
        macro_residual_min_valid_fraction = float(local_cfg.get("sgq_macro_residual_min_valid_fraction", depth_structure_min_valid_fraction))
        macro_residual_min_alltracker_visibility = float(local_cfg.get("sgq_macro_residual_min_alltracker_visibility", 0.0))
        macro_residual_min_candidate_overlap_iou = float(local_cfg.get("sgq_macro_residual_min_candidate_overlap_iou", 0.20))
        macro_residual_min_candidate_coverage = float(local_cfg.get("sgq_macro_residual_min_candidate_coverage", 0.55))
        macro_residual_allow_unmerged = bool(local_cfg.get("sgq_macro_residual_allow_unmerged", True))
        alltracker_min_visibility_score = float(local_cfg.get("sgq_alltracker_min_visibility_score", 0.5))
        merge_to_b2_min_iou = float(local_cfg.get("sgq_merge_to_b2_min_iou", 0.25))
        merge_to_b2_min_coverage = float(local_cfg.get("sgq_merge_to_b2_min_coverage", 0.25))
        candidate_rows: list[dict[str, Any]] = []
        selected_rows: list[dict[str, Any]] = []
        sam_call_rows: list[dict[str, Any]] = []
        local_object_rows: list[dict[str, Any]] = []
        local_frame_rows: list[dict[str, Any]] = []
        gt_diag_rows: list[dict[str, Any]] = []
        alltracker_rows: list[dict[str, Any]] = []
        alltracker_propagation_rows: list[dict[str, Any]] = []
        gt_instance_keys: set[str] = set()
        latency_rows: list[dict[str, Any]] = []
        memory_rows: list[dict[str, Any]] = []
        video_rows: list[dict[str, Any]] = []
        metric_rows: list[dict[str, Any]] = []
        t_stage = time.time()
        peak_gpu_mb = 0.0

        try:
            import torch  # type: ignore

            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()
            predictor = _build_sgq_image_refiner(ctx, refiner_provider, device)
            provider_supports_mask_input = bool(getattr(predictor, "supports_mask_input", True))
            provider_supports_point_input = bool(getattr(predictor, "supports_point_input", True))
            for scene in scenes:
                scene_t0 = time.time()
                scene_candidates: list[dict[str, Any]] = []
                labels_by_frame: dict[int, np.ndarray] = {}
                b2_labels_by_frame: dict[int, np.ndarray] = {}
                alltracker_maps: dict[int, np.ndarray] = {}
                if alltracker_visibility_repair and bool(local_cfg.get("use_alltracker_proposals", False)):
                    alltracker_maps, alltracker_row = _compute_alltracker_visibility_maps(ctx, scene, frame_ids_by_scene[scene])
                    alltracker_rows.append(alltracker_row)
                    if alltracker_row.get("status") != "completed":
                        ctx.add_failure(
                            stage_name="sgq_local",
                            failure_type="ALLTRACKER_PROPAGATION_UNRELIABLE",
                            severity="warning",
                            scene_id=scene,
                            symptom=f"AllTracker visibility witness failed: {alltracker_row.get('failure_reason')}",
                            suggested_repair="Repair AllTracker checkpoint/runtime before using alltracker visibility as query evidence.",
                        )
                for frame_id in frame_ids_by_scene[scene]:
                    b2_path = b2_mask_root / scene / "mask" / f"{int(frame_id)}.png"
                    b2_label = _read_image(b2_path, cv2.IMREAD_UNCHANGED)
                    if b2_label is None:
                        ctx.add_failure(
                            stage_name="sgq_local",
                            failure_type="QUERY_RECALL_LOW",
                            severity="blocker",
                            scene_id=scene,
                            symptom=f"missing B2 mask for SGQ local: {_rel(b2_path)}",
                            suggested_repair="Rerun Phase3 B2 baseline before SGQ local.",
                        )
                        continue
                    if b2_label.ndim == 3:
                        b2_label = b2_label[..., 0]
                    labels_by_frame[int(frame_id)] = b2_label.astype(np.uint16, copy=True)
                    b2_labels_by_frame[int(frame_id)] = b2_label.astype(np.uint16, copy=True)
                    gt_path = scannet_root / scene / "instance" / "instance" / f"{int(frame_id)}.png"
                    gt_label = _preprocessed_label_for_sgq(ctx, scene, int(frame_id), gt_path)
                    if gt_label is not None:
                        for gt_id in [int(value) for value in np.unique(gt_label) if int(value) > 0]:
                            gt_instance_keys.add(f"{scene}:f{int(frame_id):06d}:gt{gt_id}")
                    depth_map = _preprocessed_depth_for_sgq(ctx, scene, int(frame_id))
                    proposal_sources: list[tuple[str, Path, str]] = [
                        ("cropformer", scannet_root / scene / "output_Cropformer" / "mask" / f"{int(frame_id)}.png", "cropformer_unexplained_proposal")
                    ]
                    if bool(local_cfg.get("use_fastsam_as_proposal", False)):
                        proposal_sources.append(
                            (
                                "fastsam",
                                ctx.output_root / "baselines" / "masks" / "B1_fastsam_only" / scene / "mask" / f"{int(frame_id)}.png",
                                "fastsam_unexplained_proposal",
                            )
                        )
                    for proposal_source, proposal_path, candidate_type in proposal_sources:
                        proposal_label = _preprocessed_label_for_sgq(ctx, scene, int(frame_id), proposal_path)
                        if proposal_label is None:
                            continue
                        frame_area = float(proposal_label.shape[0] * proposal_label.shape[1])
                        for mask_id in [int(value) for value in np.unique(proposal_label) if int(value) > 0]:
                            mask = proposal_label == mask_id
                            area_ratio = float(np.count_nonzero(mask)) / frame_area
                            if area_ratio < min_area_ratio:
                                continue
                            overlap_iou = _max_iou_with_label(mask, b2_label)
                            gt_best_iou, gt_best_id = _best_iou_with_label(mask, gt_label)
                            unexplained = mask & (b2_label == 0)
                            gain = float(np.count_nonzero(unexplained)) / frame_area
                            depth_stats = _depth_structure_stats(unexplained if np.any(unexplained) else mask, depth_map)
                            alltracker_score: float | str = ""
                            alltracker_p10: float | str = ""
                            alltracker_map = alltracker_maps.get(int(frame_id))
                            if alltracker_map is not None and np.any(mask):
                                values = alltracker_map[mask]
                                if values.size:
                                    alltracker_score = float(np.mean(values))
                                    alltracker_p10 = float(np.percentile(values, 10))
                            source_tag = "cf" if proposal_source == "cropformer" else "fs"
                            candidate_id = f"{variant}:{scene}:f{int(frame_id):06d}:{source_tag}{mask_id:04d}"
                            row = {
                                "schema_version": "stream4d_v105_query_candidate_row_v1",
                                "variant_id": variant,
                                "candidate_id": candidate_id,
                                "candidate_type": candidate_type,
                                "proposal_source": proposal_source,
                                "proposal_mask_id": int(mask_id),
                                "proposal_mask_path": _rel(proposal_path),
                                "cropformer_mask_id": int(mask_id) if proposal_source == "cropformer" else "",
                                "fastsam_mask_id": int(mask_id) if proposal_source == "fastsam" else "",
                                "scene_id": scene,
                                "chunk_id": "dev_smoke_c0000",
                                "frame_start": int(frame_id),
                                "frame_end": int(frame_id),
                                "anchor_frame_id": int(frame_id),
                                "area_ratio": area_ratio,
                                "geometry_residual_score": depth_stats["depth_structure_iqr_mm"],
                                "semantic_prior_score": "",
                                "tracking_uncertainty_score": 1.0 - overlap_iou,
                                "alltracker_visibility_score": alltracker_score,
                                "alltracker_visibility_p10": alltracker_p10,
                                "expected_coverage_gain": gain,
                                "expected_cost_sec": 1.0,
                                "expected_vram_mb": "",
                                "conflict_risk": overlap_iou,
                                "selected": False,
                                "selection_rank": "",
                                "gt_diagnostic_best_iou": gt_best_iou,
                                "gt_diagnostic_best_instance_id": gt_best_id,
                                "gt_diagnostic_hit_iou25": gt_best_iou >= 0.25,
                                "gt_diagnostic_note": "diagnostic_only_not_used_for_query_selection",
                                "depth_structure_source": "scannet_depth_preprocessed_non_gt",
                                **depth_stats,
                            }
                            scene_candidates.append(row)
                            gt_diag_rows.append(
                                {
                                    "schema_version": "stream4d_v105_gt_diagnostic_row_v1",
                                    "row_type": "candidate",
                                    "variant_id": variant,
                                    "scene_id": scene,
                                    "frame_id": int(frame_id),
                                    "candidate_id": candidate_id,
                                    "query_id": "",
                                    "source": f"{proposal_source}_candidate",
                                    "gt_best_iou": gt_best_iou,
                                    "gt_best_instance_id": gt_best_id,
                                    "gt_hit_iou25": gt_best_iou >= 0.25,
                                    "method_path_uses_gt": False,
                                }
                            )
                if precision_repair:
                    persistence_counts: dict[str, int] = {}
                    for row in scene_candidates:
                        key = f"{row.get('proposal_source', 'cropformer')}:{int(row.get('proposal_mask_id') or 0)}"
                        persistence_counts[key] = persistence_counts.get(key, 0) + 1
                    for row in scene_candidates:
                        key = f"{row.get('proposal_source', 'cropformer')}:{int(row.get('proposal_mask_id') or 0)}"
                        persistence = persistence_counts.get(key, 0)
                        row["semantic_prior_score"] = persistence
                        row["candidate_persistence_count"] = persistence
                    if macro_residual_repair:
                        scene_candidates.sort(
                            key=lambda row: (
                                float(row["expected_coverage_gain"]),
                                float(row.get("depth_structure_valid_fraction") or 0.0),
                                float(row.get("depth_structure_iqr_mm") or 0.0),
                                int(row.get("candidate_persistence_count") or 0),
                                float(row.get("alltracker_visibility_score") or 0.0),
                                -float(row["conflict_risk"]),
                            ),
                            reverse=True,
                        )
                        selected = [
                            row
                            for row in scene_candidates
                            if float(row["expected_coverage_gain"]) >= min_expected_gain
                            and float(row["conflict_risk"]) <= max_conflict_risk
                            and macro_residual_min_area_ratio <= float(row["area_ratio"]) <= macro_residual_max_area_ratio
                            and int(row.get("candidate_persistence_count") or 0) >= (fastsam_min_persistence if row.get("proposal_source") == "fastsam" else point_veto_min_persistence)
                            and float(row.get("depth_structure_iqr_mm") or 0.0) >= macro_residual_min_depth_iqr_mm
                            and float(row.get("depth_structure_valid_fraction") or 0.0) >= macro_residual_min_valid_fraction
                            and float(row.get("alltracker_visibility_score") or 0.0) >= macro_residual_min_alltracker_visibility
                        ][:query_budget]
                    else:
                        scene_candidates.sort(
                            key=lambda row: (
                                float(row.get("depth_structure_iqr_mm") or 0.0) if depth_structure_repair else 0.0,
                                float(row.get("depth_structure_valid_fraction") or 0.0) if depth_structure_repair else 0.0,
                                float(row.get("alltracker_visibility_score") or 0.0) if alltracker_visibility_repair else 0.0,
                                int(row.get("candidate_persistence_count") or 0),
                                -(float(row["area_ratio"]) > point_veto_max_area_ratio),
                                float(row["expected_coverage_gain"]) - 0.1 * float(row["conflict_risk"]),
                            ),
                            reverse=True,
                        )
                        selected = [
                            row
                            for row in scene_candidates
                            if float(row["expected_coverage_gain"]) >= min_expected_gain
                            and float(row["conflict_risk"]) <= max_conflict_risk
                            and float(row["area_ratio"]) <= point_veto_max_area_ratio
                            and int(row.get("candidate_persistence_count") or 0) >= (fastsam_min_persistence if row.get("proposal_source") == "fastsam" else point_veto_min_persistence)
                            and (not depth_structure_repair or float(row.get("depth_structure_iqr_mm") or 0.0) >= depth_structure_min_iqr_mm)
                            and (not depth_structure_repair or float(row.get("depth_structure_valid_fraction") or 0.0) >= depth_structure_min_valid_fraction)
                            and (not alltracker_visibility_repair or float(row.get("alltracker_visibility_score") or 0.0) >= alltracker_min_visibility_score)
                        ][:query_budget]
                else:
                    scene_candidates.sort(key=lambda row: (float(row["expected_coverage_gain"]) - 0.1 * float(row["conflict_risk"])), reverse=True)
                    selected = [
                        row
                        for row in scene_candidates
                        if float(row["expected_coverage_gain"]) >= min_expected_gain and float(row["conflict_risk"]) <= max_conflict_risk
                    ][:query_budget]
                selected_ids = {row["candidate_id"] for row in selected}
                for rank, row in enumerate(scene_candidates, start=1):
                    row["selected"] = row["candidate_id"] in selected_ids
                    row["selection_rank"] = rank if row["selected"] else ""
                    candidate_rows.append(row)
                for rank, row in enumerate(selected, start=1):
                    selected_rows.append({**row, "selection_rank": rank})

                next_mask_id = 1 + max([int(labels_by_frame[f].max()) for f in labels_by_frame], default=0)
                persistent_mask_ids: dict[int, int] = {}
                selected_by_frame: dict[int, list[dict[str, Any]]] = {}
                for row in selected:
                    selected_by_frame.setdefault(int(row["anchor_frame_id"]), []).append(row)
                for frame_id, rows in selected_by_frame.items():
                    image = _preprocessed_rgb_for_sam2(ctx, scene, frame_id)
                    predictor.set_image(image)
                    label = labels_by_frame[frame_id]
                    gt_path = scannet_root / scene / "instance" / "instance" / f"{int(frame_id)}.png"
                    gt_label = _preprocessed_label_for_sgq(ctx, scene, frame_id, gt_path)
                    for row in rows:
                        call_t0 = time.time()
                        proposal_path = Path(str(row.get("proposal_mask_path", "")))
                        if not proposal_path.is_absolute():
                            proposal_path = REPO_ROOT / proposal_path
                        proposal_label = _preprocessed_label_for_sgq(ctx, scene, frame_id, proposal_path)
                        if proposal_label is None:
                            continue
                        mask = proposal_label == int(row.get("proposal_mask_id") or 0)
                        bbox = _bbox_xyxy(mask)
                        success = False
                        failure_reason = ""
                        output_area = 0
                        sam_gt_best_iou: float | str = ""
                        sam_gt_best_id: int | str = ""
                        sam_gt_hit_iou25: bool | str = ""
                        sam_multimask_count: int | str = ""
                        sam_selected_mask_index: int | str = ""
                        candidate_overlap_iou: float | str = ""
                        candidate_overlap_coverage: float | str = ""
                        sam_score_selected: float | str = ""
                        merge_to_b2_mask_id: int | str = ""
                        merge_to_b2_iou: float | str = ""
                        merge_to_b2_coverage: float | str = ""
                        merge_to_b2_applied = False
                        candidate_direct_merge_applied = False
                        post_sam_acceptance_pass: bool | str = ""
                        post_sam_acceptance_reason = ""
                        output_acceptance_source = ""
                        try:
                            if bbox is None:
                                raise ValueError("empty candidate bbox")
                            candidate_unexplained = mask & (label == 0)
                            merge_candidate_mask_id = 0
                            merge_candidate_iou = 0.0
                            merge_candidate_coverage = 0.0
                            if merge_to_b2_repair:
                                merge_candidate_mask_id, merge_candidate_iou, merge_candidate_coverage = _best_overlap_with_label(
                                    mask,
                                    b2_labels_by_frame.get(int(frame_id)),
                                )
                                merge_to_b2_mask_id = int(merge_candidate_mask_id) if merge_candidate_mask_id > 0 else ""
                                merge_to_b2_iou = float(merge_candidate_iou)
                                merge_to_b2_coverage = float(merge_candidate_coverage)
                            predict_kwargs: dict[str, Any] = {
                                "box": np.asarray(bbox, dtype=np.float32),
                                "multimask_output": True,
                                "return_logits": False,
                                "normalize_coords": True,
                            }
                            if mask_prompt_repair and provider_supports_mask_input:
                                prompt_mask = candidate_unexplained if np.any(candidate_unexplained) else mask
                                lowres = cv2.resize(
                                    prompt_mask.astype(np.float32),
                                    (mask_prompt_lowres_size, mask_prompt_lowres_size),
                                    interpolation=cv2.INTER_NEAREST,
                                )
                                mask_input = np.where(lowres > 0.0, mask_prompt_positive_logit, mask_prompt_negative_logit).astype(np.float32)
                                predict_kwargs["mask_input"] = mask_input[None, :, :]
                                predict_kwargs["multimask_output"] = bool(candidate_overlap_multimask_repair)
                            elif point_veto_repair and provider_supports_point_input:
                                point_mask = candidate_unexplained if np.any(candidate_unexplained) else mask
                                point_ys, point_xs = np.nonzero(point_mask)
                                if point_xs.size == 0 or point_ys.size == 0:
                                    raise ValueError("empty point-veto candidate mask")
                                point_xy = np.asarray([[float(np.mean(point_xs)), float(np.mean(point_ys))]], dtype=np.float32)
                                point_labels = np.asarray([1], dtype=np.int32)
                                predict_kwargs["point_coords"] = point_xy
                                predict_kwargs["point_labels"] = point_labels
                            masks, scores, _lowres = predictor.predict(**predict_kwargs)
                            mask_options = np.asarray(masks)
                            if mask_options.ndim == 2:
                                mask_options = mask_options[None, :, :]
                            score_values = np.asarray(scores).reshape(-1)
                            candidate_overlap_iou: float | str = ""
                            candidate_overlap_coverage: float | str = ""
                            sam_score_selected: float | str = ""
                            if candidate_overlap_multimask_repair:
                                target = mask if wta_override_repair else (candidate_unexplained if np.any(candidate_unexplained) else mask)
                                target_area = float(max(np.count_nonzero(target), 1))
                                best_key = (-1.0, -1.0, -1.0)
                                best = 0
                                best_iou = 0.0
                                best_coverage = 0.0
                                for idx, option in enumerate(mask_options):
                                    option_mask = option > 0
                                    inter = float(np.count_nonzero(option_mask & target))
                                    union = float(np.count_nonzero(option_mask | target))
                                    iou = inter / union if union else 0.0
                                    coverage = inter / target_area
                                    score = float(score_values[idx]) if idx < score_values.size else 0.0
                                    key = (iou, coverage, score)
                                    if key > best_key:
                                        best_key = key
                                        best = idx
                                        best_iou = iou
                                        best_coverage = coverage
                                candidate_overlap_iou = float(best_iou)
                                candidate_overlap_coverage = float(best_coverage)
                            else:
                                best = int(np.argmax(score_values)) if score_values.size else 0
                            sam_multimask_count = int(mask_options.shape[0])
                            sam_selected_mask_index = int(best)
                            sam_score_selected = float(score_values[best]) if best < score_values.size else ""
                            refined = mask_options[best] > 0
                            merge_to_b2_applied = bool(
                                merge_to_b2_repair
                                and merge_candidate_mask_id > 0
                                and merge_candidate_iou >= merge_to_b2_min_iou
                                and merge_candidate_coverage >= merge_to_b2_min_coverage
                            )
                            if post_sam_acceptance_repair:
                                reject_reasons: list[str] = []
                                score_ok = sam_score_selected != "" and float(sam_score_selected) >= post_sam_acceptance_min_score
                                overlap_ok = candidate_overlap_iou != "" and float(candidate_overlap_iou) >= post_sam_acceptance_min_overlap_iou
                                depth_ok = float(row.get("depth_structure_iqr_mm") or 0.0) >= post_sam_acceptance_min_depth_iqr_mm
                                high_merge_ok = bool(
                                    post_sam_acceptance_allow_high_merge
                                    and merge_to_b2_applied
                                    and merge_to_b2_iou != ""
                                    and float(merge_to_b2_iou) >= post_sam_acceptance_high_merge_iou
                                    and score_ok
                                )
                                residual_acceptance_ok = bool(score_ok and overlap_ok and depth_ok)
                                macro_coverage_ok = bool(
                                    macro_residual_repair
                                    and score_ok
                                    and depth_ok
                                    and candidate_overlap_iou != ""
                                    and float(candidate_overlap_iou) >= macro_residual_min_candidate_overlap_iou
                                    and candidate_overlap_coverage != ""
                                    and float(candidate_overlap_coverage) >= macro_residual_min_candidate_coverage
                                )
                                merge_required_for_row = bool(
                                    post_sam_acceptance_require_merge
                                    and not (macro_residual_repair and macro_residual_allow_unmerged)
                                )
                                if merge_required_for_row and not merge_to_b2_applied:
                                    reject_reasons.append("merge_to_b2_not_applied")
                                if not (residual_acceptance_ok or high_merge_ok or macro_coverage_ok):
                                    if not score_ok:
                                        reject_reasons.append("sam_score_below_threshold")
                                    if not overlap_ok:
                                        reject_reasons.append("sam_candidate_overlap_iou_below_threshold")
                                    if not depth_ok:
                                        reject_reasons.append("depth_structure_iqr_below_threshold")
                                    if post_sam_acceptance_allow_high_merge and not high_merge_ok:
                                        reject_reasons.append("high_merge_to_b2_iou_below_threshold")
                                    if macro_residual_repair and not macro_coverage_ok:
                                        if candidate_overlap_iou == "" or float(candidate_overlap_iou) < macro_residual_min_candidate_overlap_iou:
                                            reject_reasons.append("macro_residual_candidate_overlap_iou_below_threshold")
                                        if candidate_overlap_coverage == "" or float(candidate_overlap_coverage) < macro_residual_min_candidate_coverage:
                                            reject_reasons.append("macro_residual_candidate_coverage_below_threshold")
                                post_sam_acceptance_pass = not reject_reasons
                                post_sam_acceptance_reason = ";".join(reject_reasons)
                                if reject_reasons:
                                    raise ValueError(f"post-SAM acceptance rejected: {post_sam_acceptance_reason}")
                            if candidate_direct_output_repair:
                                refined = mask.copy()
                                candidate_direct_merge_applied = True
                                output_acceptance_source = "candidate_direct_after_sam2_witness"
                                if post_sam_acceptance_repair:
                                    output_acceptance_source = "candidate_direct_after_post_sam_acceptance"
                            else:
                                output_acceptance_source = "sam2_refined_mask"
                            if precision_repair:
                                if merge_to_b2_applied:
                                    refined = refined & ((label == 0) | (b2_labels_by_frame.get(int(frame_id), label) == int(merge_candidate_mask_id)))
                                else:
                                    refined = refined & (mask if wta_override_repair else candidate_unexplained)
                            else:
                                refined = refined & (label == 0)
                            output_area = int(np.count_nonzero(refined))
                            if output_area <= 0:
                                raise ValueError(f"{refiner_provider} refined mask empty after WTA uncovered restriction")
                            sam_gt_best_iou, sam_gt_best_id = _best_iou_with_label(refined, gt_label)
                            sam_gt_hit_iou25 = bool(float(sam_gt_best_iou) >= 0.25)
                            if precision_repair:
                                frame_area = float(max(refined.shape[0] * refined.shape[1], 1))
                                candidate_area = float(max(np.count_nonzero(mask if wta_override_repair else candidate_unexplained), 1))
                                if float(output_area) / frame_area < point_veto_min_output_area_ratio:
                                    raise ValueError(f"{refiner_provider} precision-repair output below minimum frame-area ratio")
                                if float(output_area) / candidate_area < point_veto_min_candidate_fraction:
                                    raise ValueError(f"{refiner_provider} precision-repair output below minimum candidate fraction")
                                if merge_to_b2_applied:
                                    assigned_mask_id = int(merge_candidate_mask_id)
                                else:
                                    proposal_key = _stable_int_seed(row.get("proposal_source", "cropformer"), row.get("proposal_mask_id", 0), modulo=10**9)
                                    assigned_mask_id = persistent_mask_ids.get(proposal_key)
                                    if assigned_mask_id is None:
                                        assigned_mask_id = int(next_mask_id)
                                        persistent_mask_ids[proposal_key] = assigned_mask_id
                                        next_mask_id += 1
                            else:
                                assigned_mask_id = int(next_mask_id)
                                next_mask_id += 1
                            label[refined] = int(assigned_mask_id)
                            local_frame_rows.append(
                                {
                                    "schema_version": "stream4d_v105_local_object_frame_mask_row_v1",
                                    "variant_id": variant,
                                    "scene_id": scene,
                                    "frame_id": int(frame_id),
                                    "local_object_id": f"{variant}:{scene}:{refiner_provider}_query_{assigned_mask_id}",
                                    "mask_id": int(assigned_mask_id),
                                    "source": "merge_to_b2_selected_query" if merge_to_b2_applied else (f"{refiner_provider}_depth_overlap_multimask_override_selected_query" if wta_override_repair else (f"{refiner_provider}_depth_overlap_multimask_wta_selected_query" if candidate_overlap_multimask_repair else (f"{refiner_provider}_depth_structure_veto_mask_prompt_selected_query" if depth_structure_repair and provider_supports_mask_input else (f"{refiner_provider}_depth_structure_veto_box_prompt_selected_query" if depth_structure_repair else (f"{refiner_provider}_mask_prompt_refine_selected_query" if mask_prompt_repair and provider_supports_mask_input else (f"{refiner_provider}_point_veto_selected_query" if point_veto_repair and provider_supports_point_input else f"{refiner_provider}_box_prompt_selected_query")))))),
                                    "pixel_area": output_area,
                                }
                            )
                            if alltracker_mask_propagation_repair:
                                propagated_masks, propagation_row = _propagate_mask_with_alltracker(
                                    ctx,
                                    scene_id=scene,
                                    anchor_frame_id=int(frame_id),
                                    target_frame_ids=frame_ids_by_scene[scene],
                                    anchor_mask=refined,
                                )
                                propagation_row.update(
                                    {
                                        "variant_id": variant,
                                        "candidate_id": row["candidate_id"],
                                        "assigned_mask_id": int(assigned_mask_id),
                                    }
                                )
                                alltracker_propagation_rows.append(propagation_row)
                                for target_frame_id, propagated_mask in propagated_masks.items():
                                    target_frame_id = int(target_frame_id)
                                    if target_frame_id == int(frame_id):
                                        continue
                                    target_label = labels_by_frame.get(target_frame_id)
                                    if target_label is None:
                                        continue
                                    if propagated_mask.shape[:2] != target_label.shape[:2]:
                                        propagated_mask = cv2.resize(
                                            propagated_mask.astype(np.uint8),
                                            (target_label.shape[1], target_label.shape[0]),
                                            interpolation=cv2.INTER_NEAREST,
                                        ) > 0
                                    if merge_to_b2_applied:
                                        propagated_uncovered = propagated_mask & ((target_label == 0) | (b2_labels_by_frame.get(target_frame_id, target_label) == int(assigned_mask_id)))
                                    else:
                                        propagated_uncovered = propagated_mask & (target_label == 0)
                                    propagated_area = int(np.count_nonzero(propagated_uncovered))
                                    if propagated_area <= 0:
                                        continue
                                    target_label[propagated_uncovered] = int(assigned_mask_id)
                                    labels_by_frame[target_frame_id] = target_label
                                    local_frame_rows.append(
                                        {
                                            "schema_version": "stream4d_v105_local_object_frame_mask_row_v1",
                                            "variant_id": variant,
                                            "scene_id": scene,
                                            "frame_id": target_frame_id,
                                            "local_object_id": f"{variant}:{scene}:{refiner_provider}_query_{assigned_mask_id}",
                                            "mask_id": int(assigned_mask_id),
                                            "source": "alltracker_mask_propagated_merge_to_b2" if merge_to_b2_applied else "alltracker_mask_propagated_selected_query",
                                            "pixel_area": propagated_area,
                                        }
                                    )
                            success = True
                        except Exception as exc:
                            failure_reason = _safe_error(exc)
                        gt_diag_rows.append(
                            {
                                "schema_version": "stream4d_v105_gt_diagnostic_row_v1",
                                "row_type": "sam_output",
                                "variant_id": variant,
                                "scene_id": scene,
                                "frame_id": int(frame_id),
                                "candidate_id": row["candidate_id"],
                                "query_id": row["candidate_id"],
                                "source": f"{refiner_provider}_refined_output",
                                "gt_best_iou": sam_gt_best_iou,
                                "gt_best_instance_id": sam_gt_best_id,
                                "gt_hit_iou25": sam_gt_hit_iou25,
                                "method_path_uses_gt": False,
                                "success": success,
                                "failure_reason": failure_reason,
                            }
                        )
                        if refiner_provider in {"sam31", "sam3.1", "sam31_multiplex"}:
                            prompt_type = "normalized_xywh_box_sam31_no_mask_input_candidate_overlap_multimask_override" if wta_override_repair else ("normalized_xywh_box_sam31_no_mask_input_candidate_overlap_multimask_wta" if candidate_overlap_multimask_repair else "normalized_xywh_box_sam31_no_mask_input")
                        else:
                            prompt_type = "box_xyxy_plus_candidate_mask_input_candidate_overlap_multimask_override" if wta_override_repair else ("box_xyxy_plus_candidate_mask_input_candidate_overlap_multimask_wta" if candidate_overlap_multimask_repair else ("box_xyxy_plus_candidate_mask_input_depth_structure_veto" if depth_structure_repair else ("box_xyxy_plus_candidate_mask_input" if mask_prompt_repair else ("box_xyxy_plus_positive_point_candidate_veto" if point_veto_repair else "box_xyxy"))))
                        sam_call_rows.append(
                            {
                                "schema_version": "stream4d_v105_sam_call_row_v1",
                                "variant_id": variant,
                                "call_id": f"{row['candidate_id']}:{refiner_provider}_box",
                                "provider": refiner_provider,
                                "query_id": row["candidate_id"],
                                "prompt_type": prompt_type,
                                "input_frame_count": 1,
                                "object_count_in_call": 1,
                                "runtime_sec": time.time() - call_t0,
                                "peak_gpu_memory_mb": "",
                                "output_mask_count": int(success),
                                "output_area": output_area,
                                "gt_diagnostic_best_iou_after_sam": sam_gt_best_iou,
                                "gt_diagnostic_best_instance_id_after_sam": sam_gt_best_id,
                                "gt_diagnostic_hit_iou25_after_sam": sam_gt_hit_iou25,
                                "sam_multimask_count": sam_multimask_count,
                                "sam_selected_mask_index": sam_selected_mask_index,
                                "sam_candidate_overlap_iou": candidate_overlap_iou,
                                "sam_candidate_overlap_coverage": candidate_overlap_coverage,
                                "sam_score_selected": sam_score_selected,
                                "post_sam_acceptance_repair": post_sam_acceptance_repair,
                                "post_sam_acceptance_pass": post_sam_acceptance_pass,
                                "post_sam_acceptance_reason": post_sam_acceptance_reason,
                                "candidate_direct_merge_applied": candidate_direct_merge_applied,
                                "output_acceptance_source": output_acceptance_source,
                                "merge_to_b2_applied": merge_to_b2_applied,
                                "merge_to_b2_mask_id": merge_to_b2_mask_id,
                                "merge_to_b2_iou": merge_to_b2_iou,
                                "merge_to_b2_coverage": merge_to_b2_coverage,
                                "success": success,
                                "failure_reason": failure_reason,
                            }
                        )
                    labels_by_frame[frame_id] = label
                scene_mask_dir = sgq_mask_root / scene / "mask"
                scene_mask_dir.mkdir(parents=True, exist_ok=True)
                for frame_id, label in labels_by_frame.items():
                    cv2.imwrite(str(scene_mask_dir / f"{int(frame_id)}.png"), label)
                    for mask_id in [int(value) for value in np.unique(label) if int(value) > 0]:
                        local_object_rows.append(
                            {
                                "schema_version": "stream4d_v105_local_object_row_v1",
                                "variant_id": variant,
                                "scene_id": scene,
                                "local_object_id": f"{variant}:{scene}:mask{mask_id:04d}",
                                "state": "local_smoke",
                                "source": f"b2_track_or_selected_{refiner_provider}_query",
                            }
                        )
                video_path = out_dir / "videos" / f"{variant}_{scene}.mp4"
                video_ok = _write_baseline_overlay_video(ctx=ctx, scene_id=scene, frame_ids=frame_ids_by_scene[scene], mask_dir=scene_mask_dir, video_path=video_path)
                video_rows.append({"variant_id": variant, "scene_id": scene, "video_path": _rel(video_path), "video_exists": video_ok})
                latency_rows.append({"schema_version": "stream4d_v105_local_latency_row_v1", "variant_id": variant, "scene_id": scene, "latency_sec": time.time() - scene_t0})
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                peak_gpu_mb = float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)
            if hasattr(predictor, "close"):
                predictor.close()
            del predictor
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            ctx.add_failure(
                stage_name="sgq_local",
                failure_type="SAM_PROMPT_FAILURE",
                severity="blocker",
                symptom=f"SGQ local {refiner_provider} refiner failed: {_safe_error(exc)}",
                suggested_repair="Inspect provider-specific prompt setup, candidate masks, and WTA uncovered restriction before trying additional SGQ variants.",
            )

        sgq_mask_frames_written = any((sgq_mask_root / scene / "mask").exists() and any((sgq_mask_root / scene / "mask").glob("*.png")) for scene in scenes)
        if sgq_mask_frames_written:
            pipeline_root = out_dir / "pipelines" / variant
            mask_dirs = {scene: sgq_mask_root / scene / "mask" for scene in scenes}
            support_summary = _write_pipeline_support_root(
                ctx=ctx,
                pipeline_root=pipeline_root,
                variant_id=variant,
                mask_root=_rel(sgq_mask_root),
                mask_dir_by_scene=mask_dirs,
                frame_ids_by_scene=frame_ids_by_scene,
                object_id_policy="mask_id_is_track",
            )
            for scene in scenes:
                try:
                    rows = _run_v65_soma_eval(
                        scene_id=scene,
                        pipeline_root=pipeline_root,
                        output_root=out_dir / "evaluation" / variant / scene,
                        stride=eval_stride,
                        max_frames=eval_max_frames,
                    )
                    for row in rows:
                        metric_rows.append(
                            {
                                "schema_version": "stream4d_v105_local_metric_row_v1",
                                "variant_id": variant,
                                "scene_id": row.get("scene"),
                                "metric_scope": "window_dev_smoke_v65_soma",
                                "frame_count": row.get("frame_count"),
                                "MV_AP_window": row.get("AP"),
                                "MV_AP50_window": row.get("AP50"),
                                "MV_AP25_window": row.get("AP25"),
                                "summary_json": row.get("summary_json"),
                            }
                        )
                except Exception as exc:
                    ctx.add_failure(
                        stage_name="sgq_local",
                        failure_type="BASELINE_EVAL_FAILURE",
                        severity="blocker",
                        scene_id=scene,
                        symptom=f"SGQ local v65 eval failed: {_safe_error(exc)}",
                        suggested_repair="Repair SGQ mask-root/support-ledger contract before local2history.",
                    )
            budget_summary = {
                "schema_version": "stream4d_v105_local_budget_row_v1",
                "variant_id": variant,
                "SAM_call_count": len(sam_call_rows),
                "successful_SAM_call_count": sum(1 for row in sam_call_rows if row.get("success")),
                "candidate_query_count": len(candidate_rows),
                "selected_query_count": len(selected_rows),
                "query_budget_scope": "per_scene",
                "query_budget_per_scene": query_budget_per_scene,
                "scene_count": len(scenes),
                "total_query_budget_available": total_query_budget_available,
                "query_budget_available": total_query_budget_available,
                "query_budget_used": len(selected_rows),
                **support_summary,
            }
        else:
            budget_summary = {
                "schema_version": "stream4d_v105_local_budget_row_v1",
                "variant_id": variant,
                "SAM_call_count": len(sam_call_rows),
                "successful_SAM_call_count": 0,
                "candidate_query_count": len(candidate_rows),
                "selected_query_count": len(selected_rows),
                "query_budget_scope": "per_scene",
                "query_budget_per_scene": query_budget_per_scene,
                "scene_count": len(scenes),
                "total_query_budget_available": total_query_budget_available,
                "query_budget_available": total_query_budget_available,
                "query_budget_used": len(selected_rows),
            }

        _write_records_json(out_dir / "query_candidate_records.json", candidate_rows)
        _write_records_json(out_dir / "selected_query_records.json", selected_rows)
        _write_records_json(out_dir / "sam_call_records.json", sam_call_rows)
        _write_records_json(out_dir / "local_object_records.json", local_object_rows)
        _write_records_json(out_dir / "local_object_frame_mask_records.json", local_frame_rows)
        _write_records_json(out_dir / "gt_diagnostic_records.json", gt_diag_rows)
        _write_records_json(out_dir / "alltracker_visibility_records.json", alltracker_rows)
        _write_records_json(out_dir / "alltracker_mask_propagation_records.json", alltracker_propagation_rows)
        _write_records_json(out_dir / "local_metric_records.json", metric_rows)
        _write_records_json(out_dir / "local_latency_records.json", latency_rows)
        _write_records_json(out_dir / "local_memory_records.json", [{"schema_version": "stream4d_v105_local_memory_row_v1", "variant_id": variant, "peak_gpu_memory_mb": peak_gpu_mb}])
        _write_records_json(out_dir / "local_budget_records.json", [budget_summary])
        _write_records_json(out_dir / "videos" / "video_index_records.json", video_rows)

        b2_rows = []
        b2_rows = [
            row for row in ctx.gates.get("phase3_baseline_metric_records", [])
            if row.get("variant_id") == "B2_4dpm_sam2_gap_tracking"
        ]
        if not b2_rows:
            b2_metric_path = ctx.output_root / "baselines" / "baseline_metric_records.json"
            b2_rows = [row for row in _read_records_json(b2_metric_path) if row.get("variant_id") == "B2_4dpm_sam2_gap_tracking"]
        b2_by_scene = {row.get("scene_id"): row for row in b2_rows}
        improvement_rows = []
        for row in metric_rows:
            base = b2_by_scene.get(row.get("scene_id"))
            if not base:
                continue
            delta_ap = float(row.get("MV_AP_window") or 0.0) - float(base.get("MV_AP_window") or 0.0)
            delta_ap50 = float(row.get("MV_AP50_window") or 0.0) - float(base.get("MV_AP50_window") or 0.0)
            improvement_rows.append(
                {
                    "scene_id": row.get("scene_id"),
                    "variant_id": variant,
                    "delta_MV_AP_window_vs_B2": delta_ap,
                    "delta_MV_AP50_window_vs_B2": delta_ap50,
                }
            )
        _write_records_json(out_dir / "local_improvement_records.json", improvement_rows)
        aggregate_delta_ap = float(np.mean([float(row["delta_MV_AP_window_vs_B2"]) for row in improvement_rows])) if improvement_rows else None
        aggregate_delta_ap50 = float(np.mean([float(row["delta_MV_AP50_window_vs_B2"]) for row in improvement_rows])) if improvement_rows else None
        any_scene_quality_improve = any(
            float(row["delta_MV_AP_window_vs_B2"]) >= 0.005 or float(row["delta_MV_AP50_window_vs_B2"]) >= 0.010
            for row in improvement_rows
        )
        all_scene_non_degrade = bool(improvement_rows) and all(
            float(row["delta_MV_AP_window_vs_B2"]) >= -1e-12 and float(row["delta_MV_AP50_window_vs_B2"]) >= -1e-12
            for row in improvement_rows
        )
        aggregate_quality_improve = bool(
            aggregate_delta_ap is not None
            and aggregate_delta_ap50 is not None
            and aggregate_delta_ap >= 0.005
            and aggregate_delta_ap50 >= 0.010
        )
        candidate_gt_hit_keys = {
            f"{row.get('scene_id')}:f{int(row.get('anchor_frame_id')):06d}:gt{int(row.get('gt_diagnostic_best_instance_id'))}"
            for row in candidate_rows
            if row.get("gt_diagnostic_best_instance_id") not in {"", 0, "0"} and float(row.get("gt_diagnostic_best_iou") or 0.0) >= 0.25
        }
        selected_gt_rows = [row for row in selected_rows if row.get("gt_diagnostic_best_iou") not in {"", None}]
        selected_gt_hits = [row for row in selected_gt_rows if float(row.get("gt_diagnostic_best_iou") or 0.0) >= 0.25]
        successful_sam_rows = [row for row in sam_call_rows if row.get("success")]
        sam_gt_rows = [row for row in successful_sam_rows if row.get("gt_diagnostic_best_iou_after_sam") not in {"", None}]
        sam_gt_hits = [row for row in sam_gt_rows if float(row.get("gt_diagnostic_best_iou_after_sam") or 0.0) >= 0.25]
        total_gt_instances = len(gt_instance_keys)
        candidate_recall_gt = (len(candidate_gt_hit_keys) / float(total_gt_instances)) if total_gt_instances else None
        selected_precision_gt = (len(selected_gt_hits) / float(len(selected_gt_rows))) if selected_gt_rows else None
        actual_gain_after_sam = (len(sam_gt_hits) / float(len(sam_gt_rows))) if sam_gt_rows else None
        gt_diagnostic_summary = {
            "schema_version": "stream4d_v105_sgq_local_gt_diagnostic_summary_v1",
            "variant_id": variant,
            "diagnostic_only": True,
            "method_path_uses_gt": False,
            "iou_threshold": 0.25,
            "gt_instance_observation_count": total_gt_instances,
            "candidate_recall_gt_diagnostic": candidate_recall_gt,
            "candidate_hit_gt_observation_count": len(candidate_gt_hit_keys),
            "selected_query_precision_gt_diagnostic": selected_precision_gt,
            "selected_query_gt_rows": len(selected_gt_rows),
            "selected_query_gt_hit_count": len(selected_gt_hits),
            "actual_gain_after_sam": actual_gain_after_sam,
            "sam_output_gt_rows": len(sam_gt_rows),
            "sam_output_gt_hit_count": len(sam_gt_hits),
        }
        _write_json(out_dir / "gt_diagnostic_summary.json", gt_diagnostic_summary)
        eval_cfg = ctx.config.get("evaluation", {})
        metric_gate_enabled = bool(eval_cfg.get("metric_gate_enabled", True))
        visual_gate_required = bool(eval_cfg.get("visual_gate_required", not metric_gate_enabled))
        visual_artifact_gate_pass = bool(video_rows) and any(row.get("video_exists") for row in video_rows)
        if metric_gate_enabled:
            phase4_pass = bool(metric_rows) and aggregate_quality_improve and visual_artifact_gate_pass
        else:
            phase4_pass = visual_artifact_gate_pass if visual_gate_required else True
        summary = {
            "schema_version": "stream4d_v105_sgq_local_summary_v1",
            "phase4_sgq_local_pass": phase4_pass,
            "metric_gate_enabled": metric_gate_enabled,
            "visual_gate_required": visual_gate_required,
            "visual_gate_reason": eval_cfg.get("visual_gate_reason", ""),
            "visual_artifact_gate_pass": visual_artifact_gate_pass,
            "variant_id": variant,
            "sgq_refiner_provider": refiner_provider,
            "provider_boundary": "SGQ local refiner provider is recorded in sam_call_records; provider_smoke records are authoritative for SAM3.1/EdgeTAM/FastSAM/AllTracker availability.",
            "repair_mode": repair_mode,
            "strict_repair_enabled": strict_repair,
            "min_expected_gain": min_expected_gain,
            "max_conflict_risk": max_conflict_risk,
            "point_veto_repair": point_veto_repair,
            "mask_prompt_repair": mask_prompt_repair,
            "depth_structure_repair": depth_structure_repair,
            "candidate_overlap_multimask_repair": candidate_overlap_multimask_repair,
            "wta_override_repair": wta_override_repair,
            "macro_residual_repair": macro_residual_repair,
            "macro_residual_min_area_ratio": macro_residual_min_area_ratio if macro_residual_repair else "",
            "macro_residual_max_area_ratio": macro_residual_max_area_ratio if macro_residual_repair else "",
            "macro_residual_min_depth_iqr_mm": macro_residual_min_depth_iqr_mm if macro_residual_repair else "",
            "macro_residual_min_valid_fraction": macro_residual_min_valid_fraction if macro_residual_repair else "",
            "macro_residual_min_alltracker_visibility": macro_residual_min_alltracker_visibility if macro_residual_repair else "",
            "macro_residual_min_candidate_overlap_iou": macro_residual_min_candidate_overlap_iou if macro_residual_repair else "",
            "macro_residual_min_candidate_coverage": macro_residual_min_candidate_coverage if macro_residual_repair else "",
            "macro_residual_allow_unmerged": macro_residual_allow_unmerged if macro_residual_repair else "",
            "post_sam_acceptance_repair": post_sam_acceptance_repair,
            "post_sam_acceptance_min_score": post_sam_acceptance_min_score if post_sam_acceptance_repair else "",
            "post_sam_acceptance_min_candidate_overlap_iou": post_sam_acceptance_min_overlap_iou if post_sam_acceptance_repair else "",
            "post_sam_acceptance_min_depth_iqr_mm": post_sam_acceptance_min_depth_iqr_mm if post_sam_acceptance_repair else "",
            "post_sam_acceptance_require_merge_to_b2": post_sam_acceptance_require_merge if post_sam_acceptance_repair else "",
            "post_sam_acceptance_allow_high_merge_to_b2": post_sam_acceptance_allow_high_merge if post_sam_acceptance_repair else "",
            "post_sam_acceptance_high_merge_iou": post_sam_acceptance_high_merge_iou if post_sam_acceptance_repair and post_sam_acceptance_allow_high_merge else "",
            "fastsam_proposal_repair": fastsam_proposal_repair,
            "use_fastsam_as_proposal": bool(local_cfg.get("use_fastsam_as_proposal", False)),
            "sgq_fastsam_min_persistence_count": fastsam_min_persistence if bool(local_cfg.get("use_fastsam_as_proposal", False)) else "",
            "merge_to_b2_repair": merge_to_b2_repair,
            "candidate_direct_merge_repair": candidate_direct_merge_repair,
            "candidate_direct_output_repair": candidate_direct_output_repair,
            "merge_to_b2_min_iou": merge_to_b2_min_iou if merge_to_b2_repair else "",
            "merge_to_b2_min_coverage": merge_to_b2_min_coverage if merge_to_b2_repair else "",
            "merge_to_b2_applied_count": sum(1 for row in sam_call_rows if row.get("merge_to_b2_applied")),
            "candidate_direct_merge_applied_count": sum(1 for row in sam_call_rows if row.get("candidate_direct_merge_applied")),
            "alltracker_visibility_repair": alltracker_visibility_repair,
            "alltracker_mask_propagation_repair": alltracker_mask_propagation_repair,
            "use_alltracker_proposals": bool(local_cfg.get("use_alltracker_proposals", False)),
            "sgq_alltracker_min_visibility_score": alltracker_min_visibility_score if alltracker_visibility_repair else "",
            "alltracker_visibility_completed_count": sum(1 for row in alltracker_rows if row.get("status") == "completed"),
            "alltracker_mask_propagation_completed_count": sum(1 for row in alltracker_propagation_rows if row.get("status") == "completed"),
            "alltracker_mask_propagated_frame_count": sum(int(row.get("propagated_frame_count") or 0) for row in alltracker_propagation_rows),
            "precision_repair": precision_repair,
            "point_veto_max_area_ratio": point_veto_max_area_ratio if precision_repair else "",
            "point_veto_min_persistence_count": point_veto_min_persistence if precision_repair else "",
            "mask_prompt_lowres_size": mask_prompt_lowres_size if mask_prompt_repair else "",
            "mask_prompt_positive_logit": mask_prompt_positive_logit if mask_prompt_repair else "",
            "mask_prompt_negative_logit": mask_prompt_negative_logit if mask_prompt_repair else "",
            "depth_structure_min_iqr_mm": depth_structure_min_iqr_mm if depth_structure_repair else "",
            "depth_structure_min_valid_fraction": depth_structure_min_valid_fraction if depth_structure_repair else "",
            "candidate_count": len(candidate_rows),
            "selected_query_count": len(selected_rows),
            "selected_query_count_by_scene": {
                scene: sum(1 for row in selected_rows if row.get("scene_id") == scene)
                for scene in scenes
            },
            "query_budget_scope": "per_scene",
            "query_budget_per_scene": query_budget_per_scene,
            "total_query_budget_available": total_query_budget_available,
            "query_budget_used": len(selected_rows),
            "sam_call_count": len(sam_call_rows),
            "successful_sam_call_count": sum(1 for row in sam_call_rows if row.get("success")),
            "metric_row_count": len(metric_rows),
            "video_count": sum(1 for row in video_rows if row.get("video_exists")),
            "runtime_sec": time.time() - t_stage,
            "peak_gpu_memory_mb": peak_gpu_mb,
            "quality_gate_definition": (
                "visual artifact gate: SGQ local videos must exist for visual review; MV_AP rows are diagnostic only"
                if not metric_gate_enabled
                else "aggregate mean delta vs B2 over evaluated scenes must satisfy delta_MV_AP_window>=0.005 and delta_MV_AP50_window>=0.010; any-scene improvement is diagnostic only."
            ),
            "quality_improvement_gate_pass": aggregate_quality_improve,
            "aggregate_delta_MV_AP_window_vs_B2": aggregate_delta_ap,
            "aggregate_delta_MV_AP50_window_vs_B2": aggregate_delta_ap50,
            "any_scene_quality_improvement": any_scene_quality_improve,
            "all_scene_non_degrade": all_scene_non_degrade,
            "candidate_recall_gt_diagnostic": candidate_recall_gt,
            "selected_query_precision_gt_diagnostic": selected_precision_gt,
            "actual_gain_after_sam": actual_gain_after_sam,
            "gt_diagnostic_summary": _rel(out_dir / "gt_diagnostic_summary.json"),
            "cache_read_count": 0,
        }
        _write_json(out_dir / "sgq_local_summary.json", summary)
        if not phase4_pass:
            failure_type = "VISUAL_ARTIFACT_MISSING" if not metric_gate_enabled else "QUERY_PRECISION_LOW"
            suggested_repair = (
                "Write SGQ local visualization videos and run full-frame visual audit; do not use MV_AP as gate while GT/segmentor granularity is mismatched."
                if not metric_gate_enabled
                else "Try stricter persistence/semantic veto, mask prompt refinement, or lower residual threshold; do not proceed to local2history as method success."
            )
            ctx.add_failure(
                stage_name="sgq_local",
                failure_type=failure_type,
                severity="blocker",
                symptom=f"SGQ local gate failed: {summary}",
                suggested_repair=suggested_repair,
            )
        ctx.gates["phase4_sgq_local_pass"] = bool(phase4_pass)
        ctx.gates["phase4_sgq_local_summary"] = summary
        ctx.write_summary()


def _label_life_rows(mask_dir: Path, frame_ids: list[int]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    life: dict[int, list[int]] = {}
    zero_frame_ids: list[int] = []
    label_counts: list[int] = []
    for frame_id in frame_ids:
        mask = _read_image(mask_dir / f"{int(frame_id)}.png", cv2.IMREAD_UNCHANGED)
        if mask is None:
            zero_frame_ids.append(int(frame_id))
            label_counts.append(0)
            continue
        if mask.ndim == 3:
            mask = mask[..., 0]
        labels = [int(value) for value in np.unique(mask) if int(value) > 0]
        if not labels:
            zero_frame_ids.append(int(frame_id))
        label_counts.append(len(labels))
        for label_id in labels:
            life.setdefault(int(label_id), []).append(int(frame_id))
    rows = [
        {
            "history_object_id": int(label_id),
            "frame_count": len(frames),
            "first_frame_id": min(frames),
            "last_frame_id": max(frames),
            "frame_ids": frames,
        }
        for label_id, frames in sorted(life.items())
    ]
    lifespans = [int(row["frame_count"]) for row in rows]
    summary = {
        "zero_frame_ids": zero_frame_ids,
        "all_nonzero": len(zero_frame_ids) == 0,
        "label_count_min": min(label_counts) if label_counts else 0,
        "label_count_max": max(label_counts) if label_counts else 0,
        "label_count_mean": (sum(label_counts) / float(len(label_counts))) if label_counts else 0.0,
        "history_object_count": len(rows),
        "track_lifespan_min": min(lifespans) if lifespans else 0,
        "track_lifespan_max": max(lifespans) if lifespans else 0,
        "track_lifespan_mean": (sum(lifespans) / float(len(lifespans))) if lifespans else 0.0,
        "single_frame_track_count": sum(1 for value in lifespans if value == 1),
    }
    return rows, summary


def _read_label_png(path: Path) -> np.ndarray | None:
    label = _read_image(path, cv2.IMREAD_UNCHANGED)
    if label is None:
        return None
    if label.ndim == 3:
        label = label[..., 0]
    return label.astype(np.uint16, copy=False)


def _empty_label_like_scene_frame(ctx: PipelineContext, scene_id: str, frame_id: int) -> np.ndarray:
    rgb = _preprocessed_rgb_for_sam2(ctx, scene_id, int(frame_id))
    return np.zeros(rgb.shape[:2], dtype=np.uint16)


def _majority_nonzero_label(values: np.ndarray) -> int:
    values = values.reshape(-1)
    values = values[values > 0]
    if values.size == 0:
        return 0
    unique, counts = np.unique(values.astype(np.int64), return_counts=True)
    return int(unique[int(np.argmax(counts))])


def _connected_component_label(binary: np.ndarray) -> np.ndarray:
    mask = np.asarray(binary).astype(bool)
    if mask.size == 0 or not np.any(mask):
        return np.zeros(mask.shape, dtype=np.uint16)
    count, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    if count <= 1:
        return np.zeros(mask.shape, dtype=np.uint16)
    return np.minimum(labels, np.iinfo(np.uint16).max).astype(np.uint16, copy=False)


def _read_lingbot_view_novelty_timeline(summary_path: Path | None) -> tuple[dict[int, float], Path | None, bool]:
    if summary_path is None or not summary_path.exists():
        return {}, summary_path, True
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return {}, summary_path, True
    timeline_raw = summary.get("view_novelty_timeline_json") or summary.get("lingbot_view_novelty_timeline")
    timeline_path = _as_repo_path(str(timeline_raw)) if timeline_raw else summary_path.parent / "lingbot_view_novelty_timeline.json"
    if timeline_path is None or not timeline_path.exists():
        return {}, timeline_path, True
    try:
        payload = json.loads(timeline_path.read_text(encoding="utf-8"))
    except Exception:
        return {}, timeline_path, True
    rows = payload.get("rows", payload if isinstance(payload, list) else [])
    novelty_by_frame: dict[int, float] = {}
    for row in rows:
        if not isinstance(row, dict) or "frame_id" not in row:
            continue
        try:
            novelty_by_frame[int(row["frame_id"])] = float(row.get("view_novelty", 0.0))
        except Exception:
            continue
    return novelty_by_frame, timeline_path, not bool(novelty_by_frame)


def _make_local2history_control_label(
    ctx: PipelineContext,
    *,
    control_name: str,
    scene_id: str,
    frame_id: int,
    local_idx: int,
    frame_ids: list[int],
    source_mask_dir: Path,
    previous_source_label: np.ndarray | None,
    history_mask_dir: Path | None = None,
    alltracker_dir: Path | None = None,
    lingbot_view_novelty: dict[int, float] | None = None,
    lingbot_timeline_path: Path | None = None,
    lingbot_novelty_threshold: float = 0.08,
) -> tuple[np.ndarray, dict[str, Any]]:
    source_path = source_mask_dir / f"{int(frame_id)}.png"
    source = _read_label_png(source_path)
    if source is None:
        source = _empty_label_like_scene_frame(ctx, scene_id, int(frame_id))
        source_missing = True
    else:
        source_missing = False
    detail: dict[str, Any] = {
        "source_frame_id": int(frame_id),
        "source_mask_path": _rel(source_path),
        "source_mask_missing": bool(source_missing),
        "control_policy": control_name,
    }
    if control_name in {"no_history", "no_history_baseline"}:
        return source.copy(), {**detail, "control_semantics": "source SGQ local labels copied without additional history id stabilization"}
    if control_name == "full_history":
        history_path = (history_mask_dir / f"{int(frame_id)}.png") if history_mask_dir is not None else None
        history = _read_label_png(history_path) if history_path is not None else None
        missing_history = history is None
        if history is None:
            history = np.zeros_like(source, dtype=np.uint16)
        return history.astype(np.uint16, copy=True), {
            **detail,
            "control_semantics": "actual local2history ID-stabilized output; included as named Phase8 full_history row",
            "history_mask_path": _rel(history_path) if history_path is not None else "",
            "full_history_missing": bool(missing_history),
            "provider_missing": bool(missing_history),
        }
    if control_name == "stale_history":
        if previous_source_label is None:
            stale = np.zeros_like(source, dtype=np.uint16)
            stale_source_frame_id: int | str = ""
        else:
            stale = previous_source_label.astype(np.uint16, copy=True)
            stale_source_frame_id = int(frame_ids[max(local_idx - 1, 0)])
        return stale, {
            **detail,
            "control_semantics": "one-input-frame stale history replay; current frame receives previous input frame labels",
            "stale_source_frame_id": stale_source_frame_id,
        }
    ids = [int(value) for value in np.unique(source) if int(value) > 0]
    if control_name == "alltracker_only_history":
        alltracker_path = (
            alltracker_dir / "coverage_masks" / f"frame_{int(frame_id):06d}_envelope.png"
            if alltracker_dir is not None
            else None
        )
        envelope = _read_label_png(alltracker_path) if alltracker_path is not None else None
        missing_alltracker = envelope is None
        if envelope is None:
            out = np.zeros_like(source, dtype=np.uint16)
        else:
            if envelope.shape[:2] != source.shape[:2]:
                envelope = cv2.resize(envelope, (source.shape[1], source.shape[0]), interpolation=cv2.INTER_NEAREST)
            out = _connected_component_label(envelope > 0)
        return out, {
            **detail,
            "control_semantics": "AllTracker-only control: connected components of AllTracker envelope; no object-specific SGQ/LingBot identity witness",
            "alltracker_envelope_path": _rel(alltracker_path) if alltracker_path is not None else "",
            "alltracker_missing": bool(missing_alltracker),
            "provider_missing": bool(missing_alltracker),
            "alltracker_output_label_count": len([int(value) for value in np.unique(out) if int(value) > 0]),
        }
    if control_name == "lingbot_only_history":
        novelty_map = lingbot_view_novelty or {}
        missing_lingbot = int(frame_id) not in novelty_map
        novelty = float(novelty_map.get(int(frame_id), 0.0))
        threshold = max(float(lingbot_novelty_threshold), 1e-6)
        segment_id = 1 + sum(1 for fid in frame_ids[: local_idx + 1] if float(novelty_map.get(int(fid), 0.0)) >= threshold)
        out = np.zeros_like(source, dtype=np.uint16)
        if not missing_lingbot:
            out[source > 0] = int(min(segment_id, np.iinfo(np.uint16).max))
        return out, {
            **detail,
            "control_semantics": "LingBot-only control: source nonzero support is labeled only by LingBot view-novelty temporal segments; no object-specific pixel identity witness",
            "lingbot_timeline_path": _rel(lingbot_timeline_path) if lingbot_timeline_path is not None else "",
            "lingbot_view_novelty": novelty,
            "lingbot_novelty_threshold": threshold,
            "lingbot_segment_id": int(segment_id),
            "lingbot_missing": bool(missing_lingbot),
            "provider_missing": bool(missing_lingbot),
            "lingbot_output_label_count": len([int(value) for value in np.unique(out) if int(value) > 0]),
        }
    if control_name == "density_matched_history":
        out = np.zeros_like(source, dtype=np.uint16)
        ids.sort(key=lambda value: (-int(np.count_nonzero(source == value)), value))
        for new_id, old_id in enumerate(ids, start=1):
            out[source == old_id] = int(min(new_id, np.iinfo(np.uint16).max))
        return out, {
            **detail,
            "control_semantics": "same per-frame nonzero support and object count, but ids are reassigned by current-frame area rank",
            "source_label_count": len(ids),
        }
    if control_name == "shuffled_history":
        out = np.zeros_like(source, dtype=np.uint16)
        shuffled = ids[:]
        rng = np.random.default_rng(_stable_int_seed("local2history_control", control_name, scene_id, frame_id))
        rng.shuffle(shuffled)
        for old_id, new_id in zip(ids, shuffled):
            out[source == old_id] = int(new_id)
        return out, {
            **detail,
            "control_semantics": "same per-frame masks and label set, but deterministic frame-wise id permutation breaks temporal identity",
            "source_label_count": len(ids),
        }
    if control_name == "semantic_only_history":
        cfg = ctx.config
        scannet_root = _as_repo_path(cfg.get("paths", {}).get("scannet_processed_root")) or (
            STREAM3D_ROOT / "data" / "scannet" / "processed"
        )
        sem_path = scannet_root / scene_id / "label-filt" / f"{int(frame_id)}.png"
        semantic = _preprocessed_label_for_sgq(ctx, scene_id, int(frame_id), sem_path)
        out = np.zeros_like(source, dtype=np.uint16)
        missing_semantic = semantic is None
        if semantic is not None:
            if semantic.shape[:2] != source.shape[:2]:
                semantic = cv2.resize(semantic, (source.shape[1], source.shape[0]), interpolation=cv2.INTER_NEAREST)
            for source_id in ids:
                mask = source == source_id
                semantic_id = _majority_nonzero_label(semantic[mask])
                if semantic_id <= 0:
                    continue
                out[mask] = int(min(semantic_id + 1, np.iinfo(np.uint16).max))
        return out, {
            **detail,
            "control_semantics": "source object regions inherit only majority semantic class id; object-specific temporal witness is removed",
            "semantic_path": _rel(sem_path),
            "semantic_missing": bool(missing_semantic),
            "source_label_count": len(ids),
            "semantic_output_label_count": len([int(value) for value in np.unique(out) if int(value) > 0]),
        }
    raise ValueError(f"unsupported local2history control: {control_name}")


def _write_local2history_control_scene(
    ctx: PipelineContext,
    *,
    control_name: str,
    control_variant: str,
    scene_id: str,
    frame_ids: list[int],
    source_mask_dir: Path,
    dst_mask_dir: Path,
    video_path: Path,
    history_mask_dir: Path | None = None,
    alltracker_dir: Path | None = None,
    lingbot_view_novelty: dict[int, float] | None = None,
    lingbot_timeline_path: Path | None = None,
    lingbot_novelty_threshold: float = 0.08,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    dst_mask_dir.mkdir(parents=True, exist_ok=True)
    for stale in dst_mask_dir.glob("*.png"):
        stale.unlink()
    frame_rows: list[dict[str, Any]] = []
    previous_source_label: np.ndarray | None = None
    missing_source_count = 0
    missing_semantic_count = 0
    missing_provider_count = 0
    for local_idx, frame_id in enumerate(frame_ids):
        label, detail = _make_local2history_control_label(
            ctx,
            control_name=control_name,
            scene_id=scene_id,
            frame_id=int(frame_id),
            local_idx=int(local_idx),
            frame_ids=frame_ids,
            source_mask_dir=source_mask_dir,
            previous_source_label=previous_source_label,
            history_mask_dir=history_mask_dir,
            alltracker_dir=alltracker_dir,
            lingbot_view_novelty=lingbot_view_novelty,
            lingbot_timeline_path=lingbot_timeline_path,
            lingbot_novelty_threshold=lingbot_novelty_threshold,
        )
        if detail.get("source_mask_missing"):
            missing_source_count += 1
        if detail.get("semantic_missing"):
            missing_semantic_count += 1
        if detail.get("provider_missing"):
            missing_provider_count += 1
        cv2.imwrite(str(dst_mask_dir / f"{int(frame_id)}.png"), label)
        source_now = _read_label_png(source_mask_dir / f"{int(frame_id)}.png")
        previous_source_label = source_now if source_now is not None else label
        frame_rows.append(
            {
                "schema_version": "stream4d_v105_history_control_frame_row_v1",
                "variant_id": control_variant,
                "control_name": control_name,
                "scene_id": scene_id,
                "frame_id": int(frame_id),
                "mask_path": _rel(dst_mask_dir / f"{int(frame_id)}.png"),
                "nonzero_pixels": int(np.count_nonzero(label)),
                "label_count": len([int(value) for value in np.unique(label) if int(value) > 0]),
                **detail,
            }
        )
    object_rows, frag_summary = _label_life_rows(dst_mask_dir, frame_ids)
    control_object_rows = [
        {
            "schema_version": "stream4d_v105_history_control_object_row_v1",
            "variant_id": control_variant,
            "control_name": control_name,
            "scene_id": scene_id,
            **row,
        }
        for row in object_rows
    ]
    video_ok = _write_baseline_overlay_video(ctx=ctx, scene_id=scene_id, frame_ids=frame_ids, mask_dir=dst_mask_dir, video_path=video_path)
    status = "completed" if missing_source_count == 0 and missing_semantic_count == 0 and missing_provider_count == 0 and bool(video_ok) else "partial"
    if control_name == "stale_history" and missing_source_count == 0 and bool(video_ok):
        status = "completed"
    control_row = {
        "schema_version": "stream4d_v105_history_control_row_v1",
        "variant_id": control_variant,
        "control_name": control_name,
        "scene_id": scene_id,
        "status": status,
        "honest_no_go": False,
        "mask_dir": _rel(dst_mask_dir),
        "video_path": _rel(video_path),
        "video_exists": bool(video_ok),
        "frame_count": len(frame_ids),
        "missing_source_mask_count": int(missing_source_count),
        "missing_semantic_count": int(missing_semantic_count),
        "missing_provider_count": int(missing_provider_count),
        "visualization_uses_only_input_frames": True,
        **frag_summary,
    }
    return control_row, frame_rows, control_object_rows


def run_local2history(ctx: PipelineContext) -> None:
    with StageTimer(ctx, "local2history"):
        cfg = ctx.config
        run_cfg = cfg.get("run", {})
        l2h_cfg = cfg.get("local2history", {})
        out_dir = ctx.output_root / "local2history"
        out_dir.mkdir(parents=True, exist_ok=True)
        if not bool(l2h_cfg.get("enabled", True)):
            write_not_run_stage(ctx, "local2history", "local2history.enabled=false")
            return
        scenes = list(run_cfg.get("scenes", []))
        stride = int(run_cfg.get("frame_stride", 5))
        max_frames = int(run_cfg.get("max_frames_per_scene", run_cfg.get("chunk_size", 32)))
        frame_ids_by_scene = {scene: list(range(0, stride * max_frames, stride))[:max_frames] for scene in scenes}
        sgq_summary_path = ctx.output_root / "sgq_local" / "sgq_local_summary.json"
        if sgq_summary_path.exists():
            sgq_summary = json.loads(sgq_summary_path.read_text(encoding="utf-8"))
            source_variant = str(sgq_summary.get("variant_id") or "SGQ5_r22_macro_residual_sam2_mask_prompt_sam2_smoke")
        else:
            source_variant = "SGQ5_r22_macro_residual_sam2_mask_prompt_sam2_smoke"
        source_root = ctx.output_root / "sgq_local" / "masks" / source_variant
        if not source_root.exists():
            ctx.add_failure(
                stage_name="local2history",
                failure_type="MISSING_SGQ_LOCAL_MASKS",
                severity="blocker",
                symptom=f"local2history source masks not found: {_rel(source_root)}",
                suggested_repair="Run sgq_local first and ensure sgq_local/masks/<variant>/<scene>/mask/*.png exists.",
            )
            _write_json(
                out_dir / "local2history_summary.json",
                {
                    "schema_version": "stream4d_v105_local2history_summary_v1",
                    "status": "failed",
                    "source_variant_id": source_variant,
                    "failure_reason": "missing_sgq_local_masks",
                    "source_mask_root": _rel(source_root),
                },
            )
            return

        history_variant = f"{source_variant}_history_v1"
        min_pixels = int(cfg.get("baselines", {}).get("baseline_matrix_min_mask_pixels", 16))
        lookback = max(int(l2h_cfg.get("temporal_id_lookback", cfg.get("baselines", {}).get("baseline_matrix_temporal_id_lookback", 3))), 1)
        min_score = float(l2h_cfg.get("temporal_id_min_match_score", cfg.get("baselines", {}).get("baseline_matrix_temporal_id_min_match_score", 0.18)))
        history_object_rows: list[dict[str, Any]] = []
        history_assignment_rows: list[dict[str, Any]] = []
        history_candidate_rows: list[dict[str, Any]] = []
        history_confirmation_rows: list[dict[str, Any]] = []
        fragmentation_rows: list[dict[str, Any]] = []
        video_rows: list[dict[str, Any]] = []
        id_rows: list[dict[str, Any]] = []
        control_frame_rows: list[dict[str, Any]] = []
        control_object_rows: list[dict[str, Any]] = []
        control_video_rows: list[dict[str, Any]] = []

        for scene in scenes:
            frame_ids = frame_ids_by_scene[scene]
            src_mask_dir = source_root / scene / "mask"
            dst_mask_dir = out_dir / "masks" / history_variant / scene / "mask"
            dst_mask_dir.mkdir(parents=True, exist_ok=True)
            for stale in dst_mask_dir.glob("*.png"):
                stale.unlink()
            for frame_id in frame_ids:
                src = src_mask_dir / f"{int(frame_id)}.png"
                dst = dst_mask_dir / f"{int(frame_id)}.png"
                if src.exists():
                    shutil.copy2(src, dst)
            id_summary = _stabilize_label_ids_temporally(
                mask_dir=dst_mask_dir,
                frame_ids=frame_ids,
                min_pixels=min_pixels,
                lookback=lookback,
                min_match_score=min_score,
            )
            id_rows.append(
                {
                    "schema_version": "stream4d_v105_local2history_id_summary_row_v1",
                    "variant_id": history_variant,
                    "scene_id": scene,
                    "source_mask_dir": _rel(src_mask_dir),
                    "history_mask_dir": _rel(dst_mask_dir),
                    "id_stabilization_summary": id_summary,
                }
            )
            for remap in id_summary.get("remap_rows", []):
                history_assignment_rows.append(
                    {
                        "schema_version": "stream4d_v105_history_assignment_row_v1",
                        "variant_id": history_variant,
                        "scene_id": scene,
                        **remap,
                    }
                )
                history_candidate_rows.append(
                    {
                        "schema_version": "stream4d_v105_history_candidate_row_v1",
                        "variant_id": history_variant,
                        "scene_id": scene,
                        "frame_id": remap.get("frame_id"),
                        "source_id": remap.get("source_id_before_stabilization"),
                        "area": remap.get("area"),
                        "candidate_source": "sgq_local_mask",
                    }
                )
                history_confirmation_rows.append(
                    {
                        "schema_version": "stream4d_v105_history_confirmation_row_v1",
                        "variant_id": history_variant,
                        "scene_id": scene,
                        "frame_id": remap.get("frame_id"),
                        "stable_track_id": remap.get("stable_track_id"),
                        "assignment_kind": remap.get("assignment_kind"),
                        "match_score": remap.get("match_score"),
                        "confirmed": remap.get("assignment_kind") == "reused",
                        "confirmation_policy": "temporal_mask_bbox_area_center_match",
                    }
                )
            object_rows, frag_summary = _label_life_rows(dst_mask_dir, frame_ids)
            for row in object_rows:
                history_object_rows.append(
                    {
                        "schema_version": "stream4d_v105_history_object_row_v1",
                        "variant_id": history_variant,
                        "scene_id": scene,
                        **row,
                    }
                )
            fragmentation_rows.append(
                {
                    "schema_version": "stream4d_v105_fragmentation_diagnostic_row_v1",
                    "variant_id": history_variant,
                    "scene_id": scene,
                    **frag_summary,
                }
            )
            video_path = out_dir / "videos" / f"{history_variant}_{scene}.mp4"
            video_ok = _write_baseline_overlay_video(ctx=ctx, scene_id=scene, frame_ids=frame_ids, mask_dir=dst_mask_dir, video_path=video_path)
            video_rows.append(
                {
                    "schema_version": "stream4d_v105_local2history_video_row_v1",
                    "variant_id": history_variant,
                    "scene_id": scene,
                    "video_path": _rel(video_path),
                    "video_exists": bool(video_ok),
                    "frame_count": len(frame_ids),
                    "visualization_uses_only_input_frames": True,
                }
            )

        phase8_required_controls = [
            "no_history",
            "shuffled_history",
            "stale_history",
            "alltracker_only_history",
            "lingbot_only_history",
            "full_history",
        ]
        controls = [str(value) for value in l2h_cfg.get("controls", phase8_required_controls)]
        alltracker_dir_by_scene = {
            str(scene): _as_repo_path(str(path))
            for scene, path in dict(l2h_cfg.get("alltracker_dir_by_scene", {})).items()
        }
        lingbot_summary_by_scene = {
            str(scene): _as_repo_path(str(path))
            for scene, path in dict(l2h_cfg.get("lingbot_summary_by_scene", {})).items()
        }
        lingbot_novelty_by_scene: dict[str, dict[int, float]] = {}
        lingbot_timeline_by_scene: dict[str, Path | None] = {}
        for scene, summary_path in lingbot_summary_by_scene.items():
            novelty, timeline_path, _missing = _read_lingbot_view_novelty_timeline(summary_path)
            lingbot_novelty_by_scene[scene] = novelty
            lingbot_timeline_by_scene[scene] = timeline_path
        lingbot_novelty_threshold = float(l2h_cfg.get("lingbot_novelty_threshold", 0.08))
        history_control_rows: list[dict[str, Any]] = []
        for control_name in controls:
            control_variant = f"{history_variant}_{control_name}"
            for scene in scenes:
                frame_ids = frame_ids_by_scene[scene]
                source_mask_dir = source_root / scene / "mask"
                history_mask_dir = out_dir / "masks" / history_variant / scene / "mask"
                dst_mask_dir = out_dir / "controls" / control_name / "masks" / control_variant / scene / "mask"
                video_path = out_dir / "controls" / control_name / "videos" / f"{control_variant}_{scene}.mp4"
                control_row, frame_rows, object_rows = _write_local2history_control_scene(
                    ctx,
                    control_name=control_name,
                    control_variant=control_variant,
                    scene_id=scene,
                    frame_ids=frame_ids,
                    source_mask_dir=source_mask_dir,
                    dst_mask_dir=dst_mask_dir,
                    video_path=video_path,
                    history_mask_dir=history_mask_dir,
                    alltracker_dir=alltracker_dir_by_scene.get(scene),
                    lingbot_view_novelty=lingbot_novelty_by_scene.get(scene),
                    lingbot_timeline_path=lingbot_timeline_by_scene.get(scene),
                    lingbot_novelty_threshold=lingbot_novelty_threshold,
                )
                history_control_rows.append(control_row)
                control_frame_rows.extend(frame_rows)
                control_object_rows.extend(object_rows)
                control_video_rows.append(
                    {
                        "schema_version": "stream4d_v105_local2history_control_video_row_v1",
                        "variant_id": control_variant,
                        "control_name": control_name,
                        "scene_id": scene,
                        "video_path": _rel(video_path),
                        "video_exists": bool(control_row.get("video_exists")),
                        "frame_count": len(frame_ids),
                        "visualization_uses_only_input_frames": True,
                    }
                )
        visual_audit = _write_full_frame_visual_audit(
            video_rows=video_rows + control_video_rows,
            audit_root=out_dir / "full_frame_visual_audit",
            expected_frame_count=max_frames,
        )
        controls_pass = bool(history_control_rows) and all(row.get("status") == "completed" for row in history_control_rows)
        phase8_exact_controls_present = all(name in controls for name in phase8_required_controls)
        phase8_exact_controls_pass = bool(controls_pass and phase8_exact_controls_present)
        chunk_size = int(run_cfg.get("chunk_size", 32))
        required_two_chunk_frames = max(2 * chunk_size, 1)
        two_consecutive_chunks_pass = max_frames >= required_two_chunk_frames
        visual_gate_pass = bool(video_rows) and bool(control_video_rows) and all(row.get("video_exists") for row in video_rows + control_video_rows) and bool(
            visual_audit.get("all_videos_decode_expected_frames")
        )
        status = "completed_visual_controls" if visual_gate_pass and controls_pass else ("completed_visual_only" if visual_gate_pass else "failed_visual_gate")
        if not controls_pass:
            ctx.add_failure(
                stage_name="local2history",
                failure_type="LOCAL2HISTORY_CONTROLS_INCOMPLETE",
                severity="blocker",
                symptom=f"local2history visual controls incomplete: {[row for row in history_control_rows if row.get('status') != 'completed']}",
                suggested_repair="Inspect missing source/semantic masks and rerun local2history controls before full_dev/holdout.",
            )
        _write_records_json(out_dir / "history_object_records.json", history_object_rows)
        _write_records_json(out_dir / "history_assignment_records.json", history_assignment_rows)
        _write_records_json(out_dir / "history_candidate_records.json", history_candidate_rows)
        _write_records_json(out_dir / "history_confirmation_records.json", history_confirmation_rows)
        _write_records_json(out_dir / "history_control_records.json", history_control_rows)
        _write_records_json(out_dir / "history_control_frame_records.json", control_frame_rows)
        _write_records_json(out_dir / "history_control_object_records.json", control_object_rows)
        _write_records_json(out_dir / "fragmentation_diagnostic_records.json", fragmentation_rows)
        _write_records_json(out_dir / "video_index_records.json", video_rows)
        _write_records_json(out_dir / "control_video_index_records.json", control_video_rows)
        summary = {
            "schema_version": "stream4d_v105_local2history_summary_v1",
            "status": status,
            "source_variant_id": source_variant,
            "history_variant_id": history_variant,
            "source_mask_root": _rel(source_root),
            "visual_gate_pass": visual_gate_pass,
            "controls_pass": controls_pass,
            "control_status": "completed_visual_controls" if controls_pass else "incomplete_visual_controls",
            "control_names": controls,
            "phase8_required_control_names": phase8_required_controls,
            "phase8_exact_controls_present": phase8_exact_controls_present,
            "phase8_exact_controls_pass": phase8_exact_controls_pass,
            "two_consecutive_chunks_pass": bool(two_consecutive_chunks_pass),
            "two_consecutive_chunks_required_frame_count": int(required_two_chunk_frames),
            "observed_frame_count_per_scene": int(max_frames),
            "control_row_count": len(history_control_rows),
            "control_video_count": len(control_video_rows),
            "history_object_count": len(history_object_rows),
            "history_assignment_count": len(history_assignment_rows),
            "video_count": len(video_rows),
            "full_frame_visual_audit": _rel(out_dir / "full_frame_visual_audit" / "full_frame_visual_audit.json"),
            "no_interval_sampling": True,
        }
        _write_json(out_dir / "local2history_summary.json", summary)
        ctx.gates["local2history_visual_gate_pass"] = bool(visual_gate_pass)
        ctx.gates["local2history_controls_pass"] = bool(controls_pass)
        ctx.gates["local2history_summary"] = summary
        ctx.write_summary()


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def run_specgap_phase6_import(ctx: PipelineContext) -> None:
    with StageTimer(ctx, "specgap_phase6_import"):
        cfg = ctx.config
        import_cfg = cfg.get("specgap_phase6_import", {})
        out_dir = ctx.output_root / "sgq_local"
        out_dir.mkdir(parents=True, exist_ok=True)
        if not bool(import_cfg.get("enabled", True)):
            write_not_run_stage(ctx, "specgap_phase6_import", "specgap_phase6_import.enabled=false")
            return

        run_cfg = cfg.get("run", {})
        scenes = [str(scene) for scene in run_cfg.get("scenes", [])]
        variant = str(import_cfg.get("variant_id", "P6_specgap_anchor2_offset1_densepts_smallest_max2"))
        source_by_scene = import_cfg.get("source_summary_by_scene", {})
        if not isinstance(source_by_scene, dict):
            source_by_scene = {}

        frame_rows: list[dict[str, Any]] = []
        scene_rows: list[dict[str, Any]] = []
        video_rows: list[dict[str, Any]] = []
        imported_summaries: list[dict[str, Any]] = []
        mask_root = out_dir / "masks" / variant
        for scene in scenes:
            summary_text = str(source_by_scene.get(scene, "")).strip()
            if not summary_text:
                ctx.add_failure(
                    stage_name="specgap_phase6_import",
                    scene_id=scene,
                    failure_type="MISSING_PHASE6_IMPORT_SOURCE",
                    severity="blocker",
                    symptom=f"No specgap_phase6_import.source_summary_by_scene entry for {scene}",
                    suggested_repair="Add the verified Phase6 summary path to configs/v105/specgap_sam2l_default.yaml and rerun this stage.",
                )
                continue
            summary_path = _as_repo_path(summary_text) or Path(summary_text)
            summary = _read_json_if_exists(summary_path)
            if not summary:
                ctx.add_failure(
                    stage_name="specgap_phase6_import",
                    scene_id=scene,
                    failure_type="MISSING_PHASE6_IMPORT_SOURCE",
                    severity="blocker",
                    symptom=f"Could not read Phase6 summary: {_rel(summary_path)}",
                    suggested_repair="Rerun the verified Phase6 max2 candidate or correct the summary path.",
                )
                continue

            frame_ids = [int(v) for v in summary.get("frame_ids", [])]
            dst_mask_dir = mask_root / scene / "mask"
            dst_mask_dir.mkdir(parents=True, exist_ok=True)
            for stale in dst_mask_dir.glob("*.png"):
                stale.unlink()

            copied_count = 0
            missing_labels: list[int] = []
            for record in summary.get("records", []):
                if not isinstance(record, dict):
                    continue
                frame_id = int(record.get("frame_id", -1))
                if frame_id < 0:
                    continue
                src_label = _as_repo_path(str(record.get("label_path", "")))
                dst_label = dst_mask_dir / f"{frame_id}.png"
                if src_label is not None and src_label.exists():
                    shutil.copy2(src_label, dst_label)
                    copied_count += 1
                    frame_rows.append(
                        {
                            "schema_version": "stream4d_v105_specgap_phase6_import_frame_row_v1",
                            "variant_id": variant,
                            "scene_id": scene,
                            "frame_id": frame_id,
                            "source_label_path": _rel(src_label),
                            "imported_label_path": _rel(dst_label),
                            "source_record_x1_iou": (record.get("x1_foreground_metrics") or {}).get("fg_iou"),
                            "source_candidate_area": record.get("candidate_area"),
                            "source_accepted_birth_mask_count": record.get("accepted_birth_mask_count"),
                        }
                    )
                else:
                    missing_labels.append(frame_id)

            video_path = out_dir / "videos" / f"{variant}_{scene}.mp4"
            video_ok = _write_baseline_overlay_video(ctx=ctx, scene_id=scene, frame_ids=frame_ids, mask_dir=dst_mask_dir, video_path=video_path)
            video_rows.append(
                {
                    "schema_version": "stream4d_v105_sgq_local_video_row_v1",
                    "variant_id": variant,
                    "scene_id": scene,
                    "video_path": _rel(video_path),
                    "video_exists": bool(video_ok),
                    "frame_count": len(frame_ids),
                    "source_summary": _rel(summary_path),
                }
            )
            scene_row = {
                "schema_version": "stream4d_v105_specgap_phase6_import_scene_row_v1",
                "variant_id": variant,
                "scene_id": scene,
                "source_summary": _rel(summary_path),
                "source_summary_sha256": _sha256_file(summary_path),
                "source_scene_id": summary.get("scene_id"),
                "frame_count": len(frame_ids),
                "copied_label_count": copied_count,
                "missing_label_frame_ids": missing_labels,
                "video_exists": bool(video_ok),
                "latency_gate_le_0p7_x1": bool(summary.get("latency_gate_le_0p7_x1")),
                "candidate_main_gate_le_0p5_x1": bool(summary.get("candidate_main_gate_le_0p5_x1")),
                "single_gpu_vram_gate_le_20gb": bool(summary.get("single_gpu_vram_gate_le_20gb")),
                "runtime_ratio_vs_x1": summary.get("runtime_ratio_vs_x1"),
                "runtime_ratio_vs_x0": summary.get("runtime_ratio_vs_x0"),
                "total_runtime_sec": summary.get("total_runtime_sec"),
                "peak_cuda_memory_mb": summary.get("peak_cuda_memory_mb"),
                "mean_x1_foreground_iou": summary.get("mean_x1_foreground_iou"),
                "min_x1_foreground_iou": summary.get("min_x1_foreground_iou"),
                "birth_count_total": summary.get("birth_count_total"),
                "sheet_paths": summary.get("sheet_paths", []),
                "video_path": summary.get("video_path", ""),
            }
            scene_rows.append(scene_row)
            imported_summaries.append(summary)
            if copied_count != len(frame_ids) or missing_labels or not video_ok:
                ctx.add_failure(
                    stage_name="specgap_phase6_import",
                    scene_id=scene,
                    failure_type="PHASE6_IMPORT_INCOMPLETE",
                    severity="blocker",
                    symptom=f"Phase6 import incomplete for {scene}: copied={copied_count}, expected={len(frame_ids)}, missing={missing_labels}, video_ok={video_ok}",
                    suggested_repair="Inspect Phase6 records/labels and rerun the verified Phase6 candidate before local2history.",
                )

        expected_frame_count = max([int(row.get("frame_count") or 0) for row in video_rows], default=0)
        visual_audit = _write_full_frame_visual_audit(
            video_rows=video_rows,
            audit_root=out_dir / "full_frame_visual_audit",
            expected_frame_count=expected_frame_count,
        )
        peak_mb = max([float(row.get("peak_cuda_memory_mb") or 0.0) for row in scene_rows], default=0.0)
        total_runtime = sum(float(row.get("total_runtime_sec") or 0.0) for row in scene_rows)
        label_gate = bool(scene_rows) and all(int(row.get("copied_label_count") or 0) == int(row.get("frame_count") or -1) for row in scene_rows)
        latency_gate = bool(scene_rows) and all(bool(row.get("latency_gate_le_0p7_x1")) for row in scene_rows)
        vram_gate = bool(scene_rows) and all(bool(row.get("single_gpu_vram_gate_le_20gb")) for row in scene_rows)
        visual_gate = bool(video_rows) and all(bool(row.get("video_exists")) for row in video_rows) and bool(visual_audit.get("all_videos_decode_expected_frames"))
        phase6_import_pass = bool(label_gate and latency_gate and vram_gate and visual_gate)

        _write_records_json(out_dir / "specgap_phase6_import_scene_records.json", scene_rows)
        _write_records_json(out_dir / "specgap_phase6_import_frame_records.json", frame_rows)
        _write_records_json(out_dir / "videos" / "video_index_records.json", video_rows)
        _write_records_json(
            out_dir / "local_memory_records.json",
            [
                {
                    "schema_version": "stream4d_v105_local_memory_row_v1",
                    "variant_id": variant,
                    "peak_gpu_memory_mb": peak_mb,
                    "source": "imported_phase6_summary",
                }
            ],
        )
        _write_records_json(
            out_dir / "local_budget_records.json",
            [
                {
                    "schema_version": "stream4d_v105_local_budget_row_v1",
                    "variant_id": variant,
                    "source": "imported_phase6_summary",
                    "scene_count": len(scene_rows),
                    "total_runtime_sec": total_runtime,
                    "birth_count_total": sum(int(row.get("birth_count_total") or 0) for row in scene_rows),
                    "frame_count_total": sum(int(row.get("frame_count") or 0) for row in scene_rows),
                }
            ],
        )
        summary = {
            "schema_version": "stream4d_v105_sgq_local_summary_v1",
            "phase4_sgq_local_pass": phase6_import_pass,
            "phase6_research_signal_pass": phase6_import_pass,
            "phase6_import_pass": phase6_import_pass,
            "metric_gate_enabled": False,
            "visual_gate_required": True,
            "visual_artifact_gate_pass": visual_gate,
            "variant_id": variant,
            "sgq_refiner_provider": "sam2",
            "repair_mode": "import_verified_phase6_specgap_densepts_smallest_max2",
            "source_summary_count": len(scene_rows),
            "scene_import_records_json": _rel(out_dir / "specgap_phase6_import_scene_records.json"),
            "frame_import_records_json": _rel(out_dir / "specgap_phase6_import_frame_records.json"),
            "mask_root": _rel(mask_root),
            "video_count": sum(1 for row in video_rows if row.get("video_exists")),
            "full_frame_visual_audit": _rel(out_dir / "full_frame_visual_audit" / "full_frame_visual_audit.json"),
            "latency_gate_le_0p7_x1_all_scenes": latency_gate,
            "candidate_main_gate_le_0p5_x1_all_scenes": bool(scene_rows) and all(bool(row.get("candidate_main_gate_le_0p5_x1")) for row in scene_rows),
            "single_gpu_vram_gate_le_20gb_all_scenes": vram_gate,
            "quality_gate_definition": "Imported verified Phase6 max2 local masks; MV_AP is diagnostic only. Gate requires labels, videos, <=0.7*X1, <=20GB VRAM.",
            "cache_read_count": 0,
            "not_claimed": [
                "full unified speculative-gap implementation",
                "Phase7 LingBot view-novelty contribution",
                "Phase8 two-consecutive-chunk local2history success",
                "holdout generalization",
            ],
        }
        _write_json(out_dir / "sgq_local_summary.json", summary)
        if not phase6_import_pass:
            ctx.add_failure(
                stage_name="specgap_phase6_import",
                failure_type="PHASE6_IMPORT_GATE_FAILED",
                severity="blocker",
                symptom=f"Phase6 import gate failed: {summary}",
                suggested_repair="Repair missing labels/videos or rerun the Phase6 max2 candidate before local2history/full_dev.",
            )
        ctx.gates["phase4_sgq_local_pass"] = bool(phase6_import_pass)
        ctx.gates["phase4_sgq_local_summary"] = summary
        ctx.write_summary()


def _records_with_stage(stage_name: str, path: Path) -> list[dict[str, Any]]:
    rows = _read_records_json(path)
    out: list[dict[str, Any]] = []
    for row in rows:
        if isinstance(row, dict):
            out.append({"source_stage": stage_name, "source_records_json": _rel(path), **row})
    return out


def _collect_pipeline_video_rows(ctx: PipelineContext) -> list[dict[str, Any]]:
    root = ctx.output_root
    sources = [
        ("baselines", root / "baselines" / "video_index_records.json"),
        ("sgq_local", root / "sgq_local" / "videos" / "video_index_records.json"),
        ("local2history", root / "local2history" / "video_index_records.json"),
        ("local2history_controls", root / "local2history" / "control_video_index_records.json"),
    ]
    rows: list[dict[str, Any]] = []
    for stage_name, path in sources:
        for row in _read_records_json(path):
            if not isinstance(row, dict):
                continue
            rows.append(
                {
                    "schema_version": "stream4d_v105_stage_video_index_row_v1",
                    "source_stage": stage_name,
                    "source_records_json": _rel(path),
                    **row,
                }
            )
    return rows


def _visual_protocol_gate_summary(ctx: PipelineContext) -> dict[str, Any]:
    root = ctx.output_root
    baseline_summary = _read_json_if_exists(root / "baselines" / "baseline_summary.json")
    sgq_summary = _read_json_if_exists(root / "sgq_local" / "sgq_local_summary.json")
    l2h_summary = _read_json_if_exists(root / "local2history" / "local2history_summary.json")
    baseline_visual = baseline_summary.get("full_frame_visual_audit", {}) if isinstance(baseline_summary.get("full_frame_visual_audit"), dict) else {}
    sgq_visual = _read_json_if_exists(root / "sgq_local" / "full_frame_visual_audit" / "full_frame_visual_audit.json")
    l2h_visual = _read_json_if_exists(root / "local2history" / "full_frame_visual_audit" / "full_frame_visual_audit.json")
    video_rows = _collect_pipeline_video_rows(ctx)
    cache_read_count = int(baseline_summary.get("cache_read_count") or 0) + int(sgq_summary.get("cache_read_count") or 0)
    gates = {
        "baseline_phase3_pass": bool(baseline_summary.get("phase3_baseline_pass")),
        "baseline_visual_all_decode": bool(baseline_visual.get("all_videos_decode_expected_frames")),
        "baseline_visual_no_interval_sampling": bool(baseline_visual.get("no_interval_sampling")),
        "sgq_phase4_pass": bool(sgq_summary.get("phase4_sgq_local_pass")),
        "sgq_visual_gate_pass": bool(sgq_summary.get("visual_artifact_gate_pass")),
        "sgq_visual_all_decode": bool(sgq_visual.get("all_videos_decode_expected_frames")),
        "sgq_visual_no_interval_sampling": bool(sgq_visual.get("no_interval_sampling")),
        "local2history_visual_gate_pass": bool(l2h_summary.get("visual_gate_pass")),
        "local2history_controls_pass": bool(l2h_summary.get("controls_pass")),
        "local2history_visual_all_decode": bool(l2h_visual.get("all_videos_decode_expected_frames")),
        "local2history_visual_no_interval_sampling": bool(l2h_visual.get("no_interval_sampling")),
        "videos_exist": bool(video_rows) and all(bool(row.get("video_exists")) for row in video_rows),
        "cache_read_count_zero": cache_read_count == 0,
        "metric_gate_disabled_for_visual_protocol": bool(sgq_summary.get("metric_gate_enabled") is False),
    }
    return {
        "schema_version": "stream4d_v105_visual_protocol_gate_summary_v1",
        "gates": gates,
        "pass": all(gates.values()),
        "cache_read_count": cache_read_count,
        "video_count": len(video_rows),
        "baseline_summary_json": _rel(root / "baselines" / "baseline_summary.json"),
        "sgq_local_summary_json": _rel(root / "sgq_local" / "sgq_local_summary.json"),
        "local2history_summary_json": _rel(root / "local2history" / "local2history_summary.json"),
    }


def run_full_dev(ctx: PipelineContext) -> None:
    with StageTimer(ctx, "full_dev"):
        out_dir = ctx.output_root / "full_dev"
        out_dir.mkdir(parents=True, exist_ok=True)
        visual_gate = _visual_protocol_gate_summary(ctx)
        cache_mode = str(ctx.config.get("run", {}).get("cache_mode", ""))
        cache_policy_pass = cache_mode == "write_only_verified_no_read"
        video_rows = _collect_pipeline_video_rows(ctx)
        latency_rows = list(ctx.stage_rows)
        memory_rows = (
            _records_with_stage("baselines", ctx.output_root / "baselines" / "baseline_memory_records.json")
            + _records_with_stage("sgq_local", ctx.output_root / "sgq_local" / "local_memory_records.json")
        )
        budget_rows = (
            _records_with_stage("baselines", ctx.output_root / "baselines" / "baseline_budget_records.json")
            + _records_with_stage("sgq_local", ctx.output_root / "sgq_local" / "local_budget_records.json")
            + [
                {
                    "schema_version": "stream4d_v105_full_dev_budget_row_v1",
                    "source_stage": "local2history",
                    "history_object_count": _read_json_if_exists(ctx.output_root / "local2history" / "local2history_summary.json").get("history_object_count", 0),
                    "history_assignment_count": _read_json_if_exists(ctx.output_root / "local2history" / "local2history_summary.json").get("history_assignment_count", 0),
                    "control_row_count": _read_json_if_exists(ctx.output_root / "local2history" / "local2history_summary.json").get("control_row_count", 0),
                }
            ]
        )
        _write_records_json(out_dir / "full_dev_latency_records.json", latency_rows)
        _write_records_json(out_dir / "full_dev_memory_records.json", memory_rows)
        _write_records_json(out_dir / "full_dev_budget_records.json", budget_rows)
        _write_records_json(out_dir / "video_index_records.json", video_rows)
        status = "completed_visual_protocol" if visual_gate["pass"] and cache_policy_pass else "failed_visual_protocol_gate"
        summary = {
            "schema_version": "stream4d_v105_full_dev_summary_v1",
            "status": status,
            "split": ctx.config.get("run", {}).get("split"),
            "scenes": ctx.config.get("run", {}).get("scenes", []),
            "config_hash": ctx.config_hash,
            "cache_mode": cache_mode,
            "cache_policy_pass": cache_policy_pass,
            "cache_read_count": visual_gate["cache_read_count"],
            "visual_protocol_gate": visual_gate,
            "metric_gate_enabled": False,
            "formal_ap_generated": False,
            "metric_reason": "MV_AP/AP ignored for v105 visual protocol because GT granularity mismatches current segmentors.",
            "latency_records_json": _rel(out_dir / "full_dev_latency_records.json"),
            "memory_records_json": _rel(out_dir / "full_dev_memory_records.json"),
            "budget_records_json": _rel(out_dir / "full_dev_budget_records.json"),
            "video_index_json": _rel(out_dir / "video_index_records.json"),
        }
        _write_json(out_dir / "full_dev_summary.json", summary)
        if status != "completed_visual_protocol":
            ctx.add_failure(
                stage_name="full_dev",
                failure_type="CACHE_LEAKAGE" if not cache_policy_pass else "VIDEO_EXPORT_FAILURE",
                severity="blocker",
                symptom=f"full_dev visual protocol failed: {summary}",
                suggested_repair="Rerun full pipeline with --cache-mode write_only_verified_no_read after baseline/SGQ/local2history visual gates pass.",
            )
        ctx.gates["full_dev_visual_protocol_pass"] = status == "completed_visual_protocol"
        ctx.gates["full_dev_summary"] = summary
        ctx.write_summary()


def run_holdout(ctx: PipelineContext) -> None:
    with StageTimer(ctx, "holdout"):
        out_dir = ctx.output_root / "holdout"
        out_dir.mkdir(parents=True, exist_ok=True)
        split = str(ctx.config.get("run", {}).get("split", ""))
        visual_gate = _visual_protocol_gate_summary(ctx)
        cache_mode = str(ctx.config.get("run", {}).get("cache_mode", ""))
        cache_policy_pass = cache_mode == "write_only_verified_no_read"
        is_holdout_split = split == "holdout"
        status = "completed_visual_protocol" if is_holdout_split and visual_gate["pass"] and cache_policy_pass else "not_completed"
        reason = ""
        if not is_holdout_split:
            reason = "holdout stage requires run.split=holdout and an explicit holdout scene list; current run is not a holdout run."
        elif not cache_policy_pass:
            reason = "holdout requires cache_mode=write_only_verified_no_read."
        elif not visual_gate["pass"]:
            reason = "holdout visual protocol gates did not pass."
        video_rows = _collect_pipeline_video_rows(ctx)
        _write_records_json(out_dir / "video_index_records.json", video_rows)
        _write_records_json(out_dir / "holdout_latency_records.json", list(ctx.stage_rows))
        _write_records_json(
            out_dir / "holdout_memory_records.json",
            _records_with_stage("baselines", ctx.output_root / "baselines" / "baseline_memory_records.json")
            + _records_with_stage("sgq_local", ctx.output_root / "sgq_local" / "local_memory_records.json"),
        )
        summary = {
            "schema_version": "stream4d_v105_holdout_summary_v1",
            "status": status,
            "reason": reason,
            "split": split,
            "scenes": ctx.config.get("run", {}).get("scenes", []),
            "config_hash": ctx.config_hash,
            "cache_mode": cache_mode,
            "cache_policy_pass": cache_policy_pass,
            "visual_protocol_gate": visual_gate,
            "metric_gate_enabled": False,
            "formal_ap_generated": False,
            "video_index_json": _rel(out_dir / "video_index_records.json"),
        }
        _write_json(out_dir / "holdout_summary.json", summary)
        if status != "completed_visual_protocol":
            ctx.add_failure(
                stage_name="holdout",
                failure_type="PIPELINE_FRAGMENTATION_RISK",
                severity="blocker",
                symptom=f"holdout not completed: {reason}",
                suggested_repair="Run a separate frozen-config holdout invocation with --split holdout, explicit non-dev scenes, and --cache-mode write_only_verified_no_read.",
            )
        ctx.gates["holdout_visual_protocol_pass"] = status == "completed_visual_protocol"
        ctx.gates["holdout_summary"] = summary
        ctx.write_summary()


def run_casebook(ctx: PipelineContext) -> None:
    with StageTimer(ctx, "casebook"):
        out_dir = ctx.output_root / "casebook"
        out_dir.mkdir(parents=True, exist_ok=True)
        full_dev_summary = _read_json_if_exists(ctx.output_root / "full_dev" / "full_dev_summary.json")
        holdout_summary = _read_json_if_exists(ctx.output_root / "holdout" / "holdout_summary.json")
        sgq_summary = _read_json_if_exists(ctx.output_root / "sgq_local" / "sgq_local_summary.json")
        l2h_summary = _read_json_if_exists(ctx.output_root / "local2history" / "local2history_summary.json")
        video_rows = _collect_pipeline_video_rows(ctx)
        failure_rows = list(ctx.failure_rows)
        _write_records_json(out_dir / "failure_case_records.json", failure_rows)
        _write_records_json(out_dir / "video_index_records.json", video_rows)
        _write_records_json(out_dir / "latency_table_records.json", list(ctx.stage_rows))
        _write_records_json(
            out_dir / "memory_table_records.json",
            _records_with_stage("baselines", ctx.output_root / "baselines" / "baseline_memory_records.json")
            + _records_with_stage("sgq_local", ctx.output_root / "sgq_local" / "local_memory_records.json"),
        )
        _write_records_json(
            out_dir / "method_table_records.json",
            [
                {
                    "schema_version": "stream4d_v105_method_table_row_v1",
                    "config_hash": ctx.config_hash,
                    "split": ctx.config.get("run", {}).get("split"),
                    "scenes": ctx.config.get("run", {}).get("scenes", []),
                    "metric_gate_enabled": False,
                    "formal_ap_generated": False,
                    "sgq_refiner_provider": sgq_summary.get("sgq_refiner_provider", ""),
                    "repair_mode": sgq_summary.get("repair_mode", ""),
                    "local2history_status": l2h_summary.get("status", ""),
                    "controls_pass": l2h_summary.get("controls_pass", False),
                    "video_count": len(video_rows),
                }
            ],
        )
        full_dev_pass = full_dev_summary.get("status") == "completed_visual_protocol"
        holdout_pass = holdout_summary.get("status") == "completed_visual_protocol"
        controls_pass = bool(l2h_summary.get("controls_pass"))
        visual_artifacts_pass = bool(video_rows) and all(bool(row.get("video_exists")) for row in video_rows)
        method_success = bool(full_dev_pass and holdout_pass and controls_pass and visual_artifacts_pass and not failure_rows)
        final_status = "streaming_visual_success" if method_success else "diagnostic_or_no_go"
        decision = {
            "schema_version": "stream4d_v105_final_decision_v1",
            "status": final_status,
            "method_success": method_success,
            "full_dev_pass": full_dev_pass,
            "holdout_pass": holdout_pass,
            "controls_pass": controls_pass,
            "visual_artifacts_pass": visual_artifacts_pass,
            "failure_count": len(failure_rows),
            "metric_gate_enabled": False,
            "formal_ap_generated": False,
            "metric_reason": "MV_AP/AP ignored by user instruction because GT granularity mismatches current segmentors.",
            "honest_boundary": (
                "Full method success requires a completed holdout visual protocol and no blocker failures; "
                "otherwise this casebook is diagnostic/no-go even if dev visual artifacts exist."
            ),
            "full_dev_summary_json": _rel(ctx.output_root / "full_dev" / "full_dev_summary.json"),
            "holdout_summary_json": _rel(ctx.output_root / "holdout" / "holdout_summary.json"),
            "sgq_local_summary_json": _rel(ctx.output_root / "sgq_local" / "sgq_local_summary.json"),
            "local2history_summary_json": _rel(ctx.output_root / "local2history" / "local2history_summary.json"),
            "video_index_json": _rel(out_dir / "video_index_records.json"),
            "failure_case_records_json": _rel(out_dir / "failure_case_records.json"),
        }
        _write_json(out_dir / "final_decision.json", decision)
        ctx.gates["casebook_final_decision"] = decision
        ctx.write_summary()


def write_not_run_stage(ctx: PipelineContext, stage_name: str, reason: str) -> None:
    out_dir = ctx.output_root / stage_name
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": f"stream4d_v105_{stage_name}_summary_v1",
        "stage_name": stage_name,
        "status": "not_run",
        "reason": reason,
        "no_metrics_generated": True,
    }
    _write_json(out_dir / f"{stage_name}_summary.json", payload)
    ctx.stage_rows.append(
        {
            "schema_version": "stream4d_v105_stage_profile_row_v1",
            "run_id": ctx.config["run"]["name"],
            "stage_name": stage_name,
            "runtime_sec": 0.0,
            "status": "not_run",
            "error": reason,
        }
    )
    ctx.flush_common()


def run_requested_stages(ctx: PipelineContext) -> None:
    handlers = {
        "phase0": run_phase0,
        "preprocess": run_preprocess,
        "provider_smoke": run_provider_smoke,
        "baselines": run_baselines,
        "sgq_local": run_sgq_local,
        "specgap_phase6_import": run_specgap_phase6_import,
        "local2history": run_local2history,
        "full_dev": run_full_dev,
        "holdout": run_holdout,
        "casebook": run_casebook,
    }
    for stage in ctx.stages:
        if stage in handlers:
            if stage == "preprocess" and not ctx.gates.get("phase0_pass", False):
                write_not_run_stage(ctx, stage, "Phase 0 did not pass; preprocessing is gated.")
                continue
            if stage == "provider_smoke" and not ctx.gates.get("phase1_preprocess_pass", False):
                write_not_run_stage(ctx, stage, "Phase 1 preprocessing did not pass; provider smoke is gated.")
                continue
            handlers[stage](ctx)
        else:
            ctx.add_failure(
                stage_name=stage,
                failure_type="UNKNOWN_STAGE",
                severity="error",
                symptom=f"unknown stage: {stage}",
                suggested_repair="Use comma-separated stages from phase0,preprocess,provider_smoke,baselines,sgq_local,specgap_phase6_import,local2history,full_dev,holdout,casebook.",
            )
    ctx.write_summary(final_status="completed_requested_stages")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Stream4D v105 SGQ single-entry pipeline.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--split", default=None)
    parser.add_argument("--scenes", default=None, help="comma-separated scene ids")
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=None)
    parser.add_argument("--overlap", type=int, default=None)
    parser.add_argument("--cache-mode", default=None, choices=["off", "readwrite", "write_only_verified_no_read"])
    parser.add_argument("--max-frames-per-scene", type=int, default=None)
    parser.add_argument("--scannet-processed-root", default=None)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--stages", default="phase0,preprocess,provider_smoke")
    parser.add_argument("--construct-sam2-models", action="store_true", help="Instantiate configured SAM2 image/video models during provider smoke.")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ctx = prepare_context(args)
    run_requested_stages(ctx)


if __name__ == "__main__":
    main()
