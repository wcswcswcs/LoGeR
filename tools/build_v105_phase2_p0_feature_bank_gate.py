#!/usr/bin/env python3
"""Phase 2 P0 feature-bank-only gate for Stream4D v105.

This helper keeps the baseline-x-gapadaptive birth schedule and changes only the
SAM2 feature source. It can run a direct baseline, run a feature-bank variant,
and compare their emitted label PNGs and timing/instrumentation summaries.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import importlib.util
import json
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
STREAM3D_ROOT = REPO_ROOT / "Stream3D"
if str(STREAM3D_ROOT) not in sys.path:
    sys.path.insert(0, str(STREAM3D_ROOT))

from sgq_v105.sam2_feature_bank import Sam2FrameFeatureBank

BASELINE_RUNNER = REPO_ROOT / "tools/audit_v105_baseline_x_sam2_twostage_tracking.py"
DEFAULT_CONFIG = REPO_ROOT / "configs/v105/baseline_chunk_table/baseline_x_gapadaptive_sam2.generated.yaml"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return _rel(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return p.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except Exception:
        return p.as_posix()


def _load_baseline_runner() -> Any:
    spec = importlib.util.spec_from_file_location("v105_baseline_x_runner_for_p0", BASELINE_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load baseline runner: {BASELINE_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["v105_baseline_x_runner_for_p0"] = module
    spec.loader.exec_module(module)
    return module


def _hash_rgb(rgb: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(str(tuple(rgb.shape)).encode("ascii"))
    h.update(rgb.tobytes())
    return h.hexdigest()


def _move_tree(value: Any, device: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: _move_tree(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [_move_tree(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(_move_tree(v, device) for v in value)
    return value


def _tree_nbytes(value: Any) -> int:
    import torch

    if isinstance(value, torch.Tensor):
        return int(value.numel() * value.element_size())
    if isinstance(value, dict):
        return sum(_tree_nbytes(v) for v in value.values())
    if isinstance(value, list):
        return sum(_tree_nbytes(v) for v in value)
    if isinstance(value, tuple):
        return sum(_tree_nbytes(v) for v in value)
    return 0


def _empty_feature_bank_summary(*, storage_device: str, source: str) -> dict[str, Any]:
    return {
        "schema_version": "stream4d_v105_sam2_frame_feature_bank_summary_v1",
        "record_count": 0,
        "sam2_backbone_forward_count": 0,
        "expected_backbone_forward_count": 0,
        "feature_bank_hit_count": 0,
        "feature_bank_miss_count": 0,
        "feature_bank_prefetch_wait_sec": 0.0,
        "feature_bank_cpu_bytes": 0,
        "feature_bank_gpu_bytes": 0,
        "feature_bank_h2d_bytes": 0,
        "build_runtime_sec": 0.0,
        "storage_device": str(storage_device),
        "clone_tensors": False,
        "source": str(source),
        "records": [],
    }


def _set_reproducibility(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    try:
        import torch

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
        try:
            torch.set_float32_matmul_precision("highest")
        except Exception:
            pass
    except Exception:
        pass


def _resolve_output_variant(args: SimpleNamespace) -> Path:
    output_base = Path(args.output_root)
    if not output_base.is_absolute():
        output_base = REPO_ROOT / output_base
    return output_base / str(args.variant_id)


def _load_run_inputs(runner: Any, args: SimpleNamespace) -> tuple[list[int], list[Path], list[np.ndarray]]:
    frame_ids = runner.parse_frame_ids(str(args.frame_ids), int(args.frame_start), int(args.frame_stride), int(args.frame_count))
    rgb_root = Path(args.rgb_root)
    if not rgb_root.is_absolute():
        rgb_root = REPO_ROOT / rgb_root
    rgb_root = rgb_root / str(args.scene_id) / "color"
    frame_paths = [rgb_root / f"{int(frame_id)}.jpg" for frame_id in frame_ids]
    missing = [str(path) for path in frame_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(missing[:5])
    rgbs = [runner.read_rgb(path) for path in frame_paths]
    return [int(v) for v in frame_ids], frame_paths, rgbs


class ForwardCounter:
    def __init__(self, label: str, fn: Callable[..., Any]) -> None:
        self.label = label
        self.fn = fn
        self.count = 0
        self.runtime_sec = 0.0

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        out = self.fn(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.count += 1
        self.runtime_sec += time.time() - t0
        return out

    def summary(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "forward_count": int(self.count),
            "forward_runtime_sec": float(self.runtime_sec),
        }


def _make_cli_for_baseline(cli: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        config=str(cli.config),
        scene_id=str(cli.scene_id),
        rgb_root=None,
        frame_start=cli.frame_start,
        frame_stride=cli.frame_stride,
        frame_count=cli.frame_count,
        frame_ids=cli.frame_ids,
        output_root=str(cli.output_root),
        seed=cli.seed,
    )


def run_mode(cli: argparse.Namespace) -> None:
    runner = _load_baseline_runner()
    config_path = Path(cli.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    config = runner.load_config(config_path)
    baseline_cli = _make_cli_for_baseline(cli)
    baseline_cli.config = str(config_path)
    args = runner.make_args(config, baseline_cli)
    _set_reproducibility(int(args.seed))
    frame_ids, frame_paths, rgbs = _load_run_inputs(runner, args)
    output_variant = _resolve_output_variant(args)

    mode = str(cli.mode)
    original_setup_models = runner.setup_models
    instrumentation: dict[str, Any] = {
        "schema_version": "stream4d_v105_phase2_p0_feature_bank_instrumentation_v1",
        "mode": mode,
        "scene_id": str(args.scene_id),
        "frame_ids": frame_ids,
        "frame_count": len(frame_ids),
        "config_path": _rel(config_path),
        "output_variant": _rel(output_variant),
        "scope_note": "P0 changes only SAM2 feature source while keeping baseline-x-gapadaptive schedule. It does not implement speculative proxy birth.",
    }

    def patched_setup_models(patched_args: SimpleNamespace) -> dict[str, Any]:
        import torch

        models = original_setup_models(patched_args)
        segmentor = models["segmentor"]
        tracker_model = models["tracker_model"]

        if mode == "direct":
            image_counter = ForwardCounter("direct_image_model.forward_image", segmentor.model.forward_image)
            video_counter = ForwardCounter("direct_video_predictor.forward_image", tracker_model.forward_image)
            segmentor.model.forward_image = image_counter  # type: ignore[method-assign]
            tracker_model.forward_image = video_counter  # type: ignore[method-assign]
            instrumentation["direct_counters"] = {
                "image_model": image_counter,
                "video_predictor": video_counter,
            }
            return models

        if mode != "bank":
            raise ValueError(f"unsupported run mode: {mode}")

        bank_storage_device = str(getattr(cli, "bank_storage_device", "cuda"))
        image_bank = Sam2FrameFeatureBank(storage_device=bank_storage_device, clone_tensors=True)
        image_bank.build_for_image_predictor(segmentor, frame_ids=frame_ids, rgb_frames=rgbs)

        video_bank = Sam2FrameFeatureBank(storage_device=bank_storage_device, clone_tensors=True)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            video_bank.build_for_video_paths(tracker_model, frame_ids=frame_ids, frame_paths=frame_paths)

        rgb_hash_to_frame_id = {_hash_rgb(rgb): int(frame_id) for frame_id, rgb in zip(frame_ids, rgbs, strict=True)}
        original_set_image = segmentor.set_image
        original_init_state = tracker_model.init_state
        image_leak_counter = ForwardCounter("bank_leak_image_model.forward_image", segmentor.model.forward_image)
        video_leak_counter = ForwardCounter("bank_leak_video_predictor.forward_image", tracker_model.forward_image)
        segmentor.model.forward_image = image_leak_counter  # type: ignore[method-assign]
        tracker_model.forward_image = video_leak_counter  # type: ignore[method-assign]

        bind_stats = {
            "image_bind_count": 0,
            "image_bind_miss_count": 0,
            "video_state_bind_count": 0,
            "video_feature_inject_count": 0,
            "video_feature_inject_miss_count": 0,
        }

        def bank_set_image(rgb: np.ndarray, *set_args: Any, **set_kwargs: Any) -> Any:
            frame_id = rgb_hash_to_frame_id.get(_hash_rgb(rgb))
            if frame_id is None:
                bind_stats["image_bind_miss_count"] += 1
                return original_set_image(rgb, *set_args, **set_kwargs)
            image_bank.bind_image_predictor(segmentor, int(frame_id))
            bind_stats["image_bind_count"] += 1
            return None

        def bank_init_state(*init_args: Any, **init_kwargs: Any) -> dict[str, Any]:
            state = original_init_state(*init_args, **init_kwargs)
            bind_stats["video_state_bind_count"] += 1
            return state

        original_get_image_feature = tracker_model._get_image_feature

        def bank_get_image_feature(inference_state: dict[str, Any], frame_idx: int, batch_size: int) -> Any:
            chunk_index = int(frame_idx)
            if chunk_index in video_bank.records_by_chunk_index:
                record = video_bank.get_chunk_features(chunk_index)
                device = inference_state.get("device", video_bank.storage_device)
                image = record.input_image.to(device)
                backbone_out = _move_tree(record.backbone_out, device)
                cache = inference_state.setdefault("cached_features", {})
                cache[chunk_index] = (image, backbone_out)
                bind_stats["video_feature_inject_count"] += 1
            else:
                bind_stats["video_feature_inject_miss_count"] += 1
            return original_get_image_feature(inference_state, int(frame_idx), int(batch_size))

        segmentor.set_image = bank_set_image  # type: ignore[method-assign]
        tracker_model.init_state = bank_init_state  # type: ignore[method-assign]
        tracker_model._get_image_feature = bank_get_image_feature  # type: ignore[method-assign]
        instrumentation["feature_banks"] = {
            "bank_storage_device": bank_storage_device,
            "image_bank": image_bank,
            "video_bank": video_bank,
            "bind_stats": bind_stats,
            "leak_counters": {
                "image_model": image_leak_counter,
                "video_predictor": video_leak_counter,
            },
        }
        return models

    runner.setup_models = patched_setup_models
    t0 = time.time()
    runner.run(args)
    if "direct_counters" in instrumentation:
        counters = instrumentation["direct_counters"]
        instrumentation["direct_counters"] = {key: counter.summary() for key, counter in counters.items()}
    if "feature_banks" in instrumentation:
        banks = instrumentation["feature_banks"]
        leak_counters = banks["leak_counters"]
        instrumentation["feature_banks"] = {
            "bank_storage_device": banks.get("bank_storage_device", ""),
            "image_bank_summary": banks["image_bank"].summary(),
            "video_bank_summary": banks["video_bank"].summary(),
            "bind_stats": dict(banks["bind_stats"]),
            "leak_counters": {key: counter.summary() for key, counter in leak_counters.items()},
        }
    instrumentation["helper_wall_runtime_sec"] = float(time.time() - t0)
    summary_path = output_variant / "p0_instrumentation_summary.json"
    _write_json(summary_path, instrumentation)
    print(json.dumps({"instrumentation_summary": _rel(summary_path), "output_variant": _rel(output_variant)}, sort_keys=True), flush=True)


def _build_bank_patch(
    *,
    models: dict[str, Any],
    frame_ids: list[int],
    frame_paths: list[Path],
    rgbs: list[np.ndarray],
    instrumentation: dict[str, Any],
    bank_storage_device: str = "cuda",
    image_feature_source: str = "bank",
    image_bank_storage_device: str | None = None,
    video_bank_storage_device: str | None = None,
    video_gpu_hot_window: int = 0,
) -> None:
    import torch

    segmentor = models["segmentor"]
    tracker_model = models["tracker_model"]
    image_feature_source = str(image_feature_source)
    if image_feature_source not in {"bank", "direct"}:
        raise ValueError(f"unsupported image_feature_source={image_feature_source}")
    image_bank_device = str(image_bank_storage_device or bank_storage_device)
    video_bank_device = str(video_bank_storage_device or bank_storage_device)
    video_gpu_hot_window = max(int(video_gpu_hot_window), 0)

    image_bank: Sam2FrameFeatureBank | None = None
    if image_feature_source == "bank":
        image_bank = Sam2FrameFeatureBank(storage_device=image_bank_device, clone_tensors=True)
        image_bank.build_for_image_predictor(segmentor, frame_ids=frame_ids, rgb_frames=rgbs)

    video_bank = Sam2FrameFeatureBank(storage_device=video_bank_device, clone_tensors=True)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        video_bank.build_for_video_paths(tracker_model, frame_ids=frame_ids, frame_paths=frame_paths)

    rgb_hash_to_frame_id = {_hash_rgb(rgb): int(frame_id) for frame_id, rgb in zip(frame_ids, rgbs, strict=True)}
    original_set_image = segmentor.set_image
    original_init_state = tracker_model.init_state
    original_get_image_feature = tracker_model._get_image_feature
    image_counter_label = (
        "bank_leak_image_model.forward_image"
        if image_feature_source == "bank"
        else "bank_direct_image_model.forward_image"
    )
    image_leak_counter = ForwardCounter(image_counter_label, segmentor.model.forward_image)
    video_leak_counter = ForwardCounter("bank_leak_video_predictor.forward_image", tracker_model.forward_image)
    if image_feature_source == "bank":
        segmentor.model.forward_image = image_leak_counter  # type: ignore[method-assign]
    else:
        segmentor.model.forward_image = image_leak_counter  # type: ignore[method-assign]
    tracker_model.forward_image = video_leak_counter  # type: ignore[method-assign]

    bind_stats = {
        "image_feature_source": image_feature_source,
        "image_bind_count": 0,
        "image_bind_miss_count": 0,
        "image_direct_forward_count": 0,
        "video_state_bind_count": 0,
        "video_feature_inject_count": 0,
        "video_feature_inject_miss_count": 0,
        "video_feature_cache_hit_count": 0,
        "video_feature_cache_evict_count": 0,
        "video_feature_h2d_bytes": 0,
        "video_gpu_hot_window": int(video_gpu_hot_window),
    }

    def bank_set_image(rgb: np.ndarray, *set_args: Any, **set_kwargs: Any) -> Any:
        if image_feature_source == "direct":
            result = original_set_image(rgb, *set_args, **set_kwargs)
            bind_stats["image_direct_forward_count"] = int(image_leak_counter.count)
            return result
        frame_id = rgb_hash_to_frame_id.get(_hash_rgb(rgb))
        if frame_id is None:
            bind_stats["image_bind_miss_count"] += 1
            return original_set_image(rgb, *set_args, **set_kwargs)
        assert image_bank is not None
        image_bank.bind_image_predictor(segmentor, int(frame_id))
        bind_stats["image_bind_count"] += 1
        return None

    def bank_init_state(*init_args: Any, **init_kwargs: Any) -> dict[str, Any]:
        state = original_init_state(*init_args, **init_kwargs)
        bind_stats["video_state_bind_count"] += 1
        return state

    def bank_get_image_feature(inference_state: dict[str, Any], frame_idx: int, batch_size: int) -> Any:
        chunk_index = int(frame_idx)
        if chunk_index in video_bank.records_by_chunk_index:
            device = inference_state.get("device", video_bank.storage_device)
            cache = inference_state.setdefault("cached_features", {})
            hot_order = inference_state.setdefault("_v105_feature_bank_hot_order", OrderedDict())
            if not isinstance(hot_order, OrderedDict):
                hot_order = OrderedDict((int(k), None) for k in cache.keys())
                inference_state["_v105_feature_bank_hot_order"] = hot_order
            if chunk_index in cache:
                bind_stats["video_feature_cache_hit_count"] += 1
                hot_order[int(chunk_index)] = None
                hot_order.move_to_end(int(chunk_index))
            else:
                record = video_bank.get_chunk_features(chunk_index)
                image = record.input_image.to(device, non_blocking=True)
                backbone_out = _move_tree(record.backbone_out, device)
                moved_bytes = _tree_nbytes(image) + _tree_nbytes(backbone_out)
                if not str(record.input_image.device).startswith(str(device)):
                    video_bank.feature_bank_h2d_bytes += int(moved_bytes)
                    bind_stats["video_feature_h2d_bytes"] += int(moved_bytes)
                cache[chunk_index] = (image, backbone_out)
                hot_order[int(chunk_index)] = None
                hot_order.move_to_end(int(chunk_index))
                bind_stats["video_feature_inject_count"] += 1
            if video_gpu_hot_window > 0:
                while len(hot_order) > video_gpu_hot_window:
                    evict_idx, _ = hot_order.popitem(last=False)
                    if int(evict_idx) == chunk_index:
                        hot_order[int(evict_idx)] = None
                        hot_order.move_to_end(int(evict_idx))
                        break
                    if int(evict_idx) in cache:
                        cache.pop(int(evict_idx), None)
                        bind_stats["video_feature_cache_evict_count"] += 1
        else:
            bind_stats["video_feature_inject_miss_count"] += 1
        return original_get_image_feature(inference_state, int(frame_idx), int(batch_size))

    segmentor.set_image = bank_set_image  # type: ignore[method-assign]
    tracker_model.init_state = bank_init_state  # type: ignore[method-assign]
    tracker_model._get_image_feature = bank_get_image_feature  # type: ignore[method-assign]
    instrumentation["feature_banks"] = {
        "bank_storage_device": str(bank_storage_device),
        "image_feature_source": image_feature_source,
        "image_bank_storage_device": image_bank_device,
        "video_bank_storage_device": video_bank_device,
        "video_gpu_hot_window": int(video_gpu_hot_window),
        "image_bank": image_bank,
        "video_bank": video_bank,
        "bind_stats": bind_stats,
        "leak_counters": {
            "image_model": image_leak_counter if image_feature_source == "bank" else None,
            "video_predictor": video_leak_counter,
        },
        "direct_counters": {
            "image_model": image_leak_counter if image_feature_source == "direct" else None,
        },
    }


def pair_mode(cli: argparse.Namespace) -> None:
    """Run direct and bank sequentially in one process with one loaded model pair."""

    runner = _load_baseline_runner()
    config_path = Path(cli.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    config = runner.load_config(config_path)

    direct_cli_source = argparse.Namespace(**vars(cli))
    direct_cli_source.output_root = str(cli.direct_output_root)
    direct_cli = _make_cli_for_baseline(direct_cli_source)
    direct_cli.config = str(config_path)
    direct_cli.output_root = str(cli.direct_output_root)
    direct_args = runner.make_args(config, direct_cli)
    _set_reproducibility(int(direct_args.seed))
    frame_ids, frame_paths, rgbs = _load_run_inputs(runner, direct_args)

    original_setup_models = runner.setup_models
    models = original_setup_models(direct_args)
    segmentor = models["segmentor"]
    tracker_model = models["tracker_model"]
    image_forward_original = segmentor.model.forward_image
    video_forward_original = tracker_model.forward_image

    direct_inst: dict[str, Any] = {
        "schema_version": "stream4d_v105_phase2_p0_feature_bank_instrumentation_v1",
        "mode": "direct_pair",
        "scene_id": str(direct_args.scene_id),
        "frame_ids": frame_ids,
        "frame_count": len(frame_ids),
        "config_path": _rel(config_path),
        "output_variant": _rel(_resolve_output_variant(direct_args)),
        "scope_note": "Same-process direct control for P0 feature-bank parity.",
    }
    image_counter = ForwardCounter("direct_pair_image_model.forward_image", image_forward_original)
    video_counter = ForwardCounter("direct_pair_video_predictor.forward_image", video_forward_original)
    segmentor.model.forward_image = image_counter  # type: ignore[method-assign]
    tracker_model.forward_image = video_counter  # type: ignore[method-assign]

    def setup_direct(_: SimpleNamespace) -> dict[str, Any]:
        return models

    runner.setup_models = setup_direct
    t0 = time.time()
    runner.run(direct_args)
    direct_inst["helper_wall_runtime_sec"] = float(time.time() - t0)
    direct_inst["direct_counters"] = {
        "image_model": image_counter.summary(),
        "video_predictor": video_counter.summary(),
    }
    _write_json(_resolve_output_variant(direct_args) / "p0_instrumentation_summary.json", direct_inst)

    segmentor.model.forward_image = image_forward_original  # type: ignore[method-assign]
    tracker_model.forward_image = video_forward_original  # type: ignore[method-assign]
    try:
        import gc
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    bank_cli_source = argparse.Namespace(**vars(cli))
    bank_cli_source.output_root = str(cli.bank_output_root)
    bank_cli = _make_cli_for_baseline(bank_cli_source)
    bank_cli.config = str(config_path)
    bank_cli.output_root = str(cli.bank_output_root)
    bank_args = runner.make_args(config, bank_cli)
    bank_inst: dict[str, Any] = {
        "schema_version": "stream4d_v105_phase2_p0_feature_bank_instrumentation_v1",
        "mode": "bank_pair",
        "scene_id": str(bank_args.scene_id),
        "frame_ids": frame_ids,
        "frame_count": len(frame_ids),
        "config_path": _rel(config_path),
        "output_variant": _rel(_resolve_output_variant(bank_args)),
        "scope_note": "Same-process bank run for P0 feature-bank parity.",
    }
    _build_bank_patch(
        models=models,
        frame_ids=frame_ids,
        frame_paths=frame_paths,
        rgbs=rgbs,
        instrumentation=bank_inst,
        bank_storage_device=str(cli.bank_storage_device),
        image_feature_source=str(cli.image_feature_source),
        image_bank_storage_device=str(cli.image_bank_storage_device or cli.bank_storage_device),
        video_bank_storage_device=str(cli.video_bank_storage_device or cli.bank_storage_device),
        video_gpu_hot_window=int(cli.video_gpu_hot_window),
    )

    def setup_bank(_: SimpleNamespace) -> dict[str, Any]:
        return models

    runner.setup_models = setup_bank
    t0 = time.time()
    runner.run(bank_args)
    bank_inst["helper_wall_runtime_sec"] = float(time.time() - t0)
    banks = bank_inst["feature_banks"]
    leak_counters = banks["leak_counters"]
    direct_counters = banks.get("direct_counters", {})
    image_bank_summary = (
        banks["image_bank"].summary()
        if banks.get("image_bank") is not None
        else _empty_feature_bank_summary(
            storage_device=str(banks.get("image_bank_storage_device", "")),
            source=str(banks.get("image_feature_source", "direct")),
        )
    )
    bank_inst["feature_banks"] = {
        "bank_storage_device": banks.get("bank_storage_device", ""),
        "image_feature_source": banks.get("image_feature_source", "bank"),
        "image_bank_storage_device": banks.get("image_bank_storage_device", ""),
        "video_bank_storage_device": banks.get("video_bank_storage_device", ""),
        "video_gpu_hot_window": int(banks.get("video_gpu_hot_window", 0) or 0),
        "image_bank_summary": image_bank_summary,
        "video_bank_summary": banks["video_bank"].summary(),
        "bind_stats": dict(banks["bind_stats"]),
        "leak_counters": {key: counter.summary() for key, counter in leak_counters.items() if counter is not None},
        "direct_counters": {key: counter.summary() for key, counter in direct_counters.items() if counter is not None},
    }
    _write_json(_resolve_output_variant(bank_args) / "p0_instrumentation_summary.json", bank_inst)

    compare_out = Path(cli.compare_output_root)
    compare_cli = argparse.Namespace(
        direct_variant_dir=_resolve_output_variant(direct_args),
        bank_variant_dir=_resolve_output_variant(bank_args),
        output_root=compare_out,
        min_runtime_decrease_ratio=cli.min_runtime_decrease_ratio,
    )
    compare_mode(compare_cli)


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    aa = a.astype(bool)
    bb = b.astype(bool)
    union = int(np.count_nonzero(aa | bb))
    if union == 0:
        return 1.0
    return float(np.count_nonzero(aa & bb)) / float(union)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _int_field(payload: dict[str, Any], key: str, default: int) -> int:
    if key not in payload or payload.get(key) is None:
        return int(default)
    return int(payload.get(key))


def compare_mode(cli: argparse.Namespace) -> None:
    direct_dir = Path(cli.direct_variant_dir)
    bank_dir = Path(cli.bank_variant_dir)
    if not direct_dir.is_absolute():
        direct_dir = REPO_ROOT / direct_dir
    if not bank_dir.is_absolute():
        bank_dir = REPO_ROOT / bank_dir
    out_dir = Path(cli.output_root)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    direct_labels = sorted((direct_dir / "labels").glob("*.png"))
    bank_labels = sorted((bank_dir / "labels").glob("*.png"))
    direct_by_name = {path.name: path for path in direct_labels}
    bank_by_name = {path.name: path for path in bank_labels}
    frame_rows: list[dict[str, Any]] = []
    all_exact = True
    all_visible_sets = True
    min_foreground_iou = 1.0
    min_per_id_iou = 1.0

    for name in sorted(set(direct_by_name) | set(bank_by_name)):
        direct_path = direct_by_name.get(name)
        bank_path = bank_by_name.get(name)
        if direct_path is None or bank_path is None:
            frame_rows.append({"label_name": name, "missing": True, "exact_equal": False, "visible_id_sets_match": False})
            all_exact = False
            all_visible_sets = False
            min_foreground_iou = 0.0
            min_per_id_iou = 0.0
            continue
        direct = cv2.imread(str(direct_path), cv2.IMREAD_UNCHANGED)
        bank = cv2.imread(str(bank_path), cv2.IMREAD_UNCHANGED)
        if direct is None or bank is None or direct.shape != bank.shape:
            frame_rows.append({"label_name": name, "read_or_shape_failure": True, "exact_equal": False, "visible_id_sets_match": False})
            all_exact = False
            all_visible_sets = False
            min_foreground_iou = 0.0
            min_per_id_iou = 0.0
            continue
        direct_ids = sorted(int(v) for v in np.unique(direct) if int(v) != 0)
        bank_ids = sorted(int(v) for v in np.unique(bank) if int(v) != 0)
        visible_match = direct_ids == bank_ids
        fg_iou = _mask_iou(direct > 0, bank > 0)
        per_id_rows = []
        frame_min_id_iou = 1.0
        for label_id in sorted(set(direct_ids) | set(bank_ids)):
            iou = _mask_iou(direct == int(label_id), bank == int(label_id))
            frame_min_id_iou = min(frame_min_id_iou, iou)
            per_id_rows.append({"label_id": int(label_id), "iou": float(iou)})
        exact = bool(np.array_equal(direct, bank))
        all_exact = all_exact and exact
        all_visible_sets = all_visible_sets and visible_match
        min_foreground_iou = min(min_foreground_iou, fg_iou)
        min_per_id_iou = min(min_per_id_iou, frame_min_id_iou)
        frame_rows.append(
            {
                "label_name": name,
                "exact_equal": exact,
                "visible_id_sets_match": visible_match,
                "direct_visible_id_count": len(direct_ids),
                "bank_visible_id_count": len(bank_ids),
                "foreground_iou": float(fg_iou),
                "min_per_id_iou": float(frame_min_id_iou),
                "per_id_rows": per_id_rows,
            }
        )

    direct_summary = _read_json(direct_dir / "summary.json")
    bank_summary = _read_json(bank_dir / "summary.json")
    direct_inst = _read_json(direct_dir / "p0_instrumentation_summary.json")
    bank_inst = _read_json(bank_dir / "p0_instrumentation_summary.json")
    bank_feature = bank_inst.get("feature_banks", {}) if isinstance(bank_inst.get("feature_banks"), dict) else {}
    image_feature_source = str(bank_feature.get("image_feature_source", "bank"))
    image_bank_summary = bank_feature.get("image_bank_summary", {}) if isinstance(bank_feature.get("image_bank_summary"), dict) else {}
    video_bank_summary = bank_feature.get("video_bank_summary", {}) if isinstance(bank_feature.get("video_bank_summary"), dict) else {}
    leak_counters = bank_feature.get("leak_counters", {}) if isinstance(bank_feature.get("leak_counters"), dict) else {}
    direct_runtime = float(direct_summary.get("total_runtime_sec") or 0.0)
    bank_runtime = float(bank_summary.get("total_runtime_sec") or 0.0)
    image_build_runtime = float(image_bank_summary.get("build_runtime_sec") or 0.0)
    video_build_runtime = float(video_bank_summary.get("build_runtime_sec") or 0.0)
    bank_runtime_with_feature_build = bank_runtime + image_build_runtime + video_build_runtime
    runtime_decrease_ratio = None
    if direct_runtime > 0.0:
        runtime_decrease_ratio = (direct_runtime - bank_runtime) / direct_runtime
    runtime_decrease_ratio_with_feature_build = None
    if direct_runtime > 0.0:
        runtime_decrease_ratio_with_feature_build = (direct_runtime - bank_runtime_with_feature_build) / direct_runtime

    if image_feature_source == "direct":
        image_leaks = 0
    else:
        image_leaks = int((leak_counters.get("image_model") or {}).get("forward_count") or 0) if isinstance(leak_counters.get("image_model"), dict) else -1
    video_leaks = int((leak_counters.get("video_predictor") or {}).get("forward_count") or 0) if isinstance(leak_counters.get("video_predictor"), dict) else -1
    image_expected = _int_field(image_bank_summary, "expected_backbone_forward_count", -1)
    image_count = _int_field(image_bank_summary, "sam2_backbone_forward_count", -2)
    video_expected = _int_field(video_bank_summary, "expected_backbone_forward_count", -1)
    video_count = _int_field(video_bank_summary, "sam2_backbone_forward_count", -2)

    parity_pass = bool(
        all_visible_sets
        and min_per_id_iou >= 0.995
        and min_foreground_iou >= 0.999
        and image_count == image_expected
        and video_count == video_expected
        and image_leaks == 0
        and video_leaks == 0
    )
    efficiency_pass = bool(
        runtime_decrease_ratio_with_feature_build is not None
        and runtime_decrease_ratio_with_feature_build >= float(cli.min_runtime_decrease_ratio)
    )
    payload = {
        "schema_version": "stream4d_v105_phase2_p0_feature_bank_compare_v1",
        "direct_variant_dir": _rel(direct_dir),
        "bank_variant_dir": _rel(bank_dir),
        "frame_count": len(frame_rows),
        "all_labels_exact_equal": bool(all_exact),
        "visible_id_sets_match": bool(all_visible_sets),
        "min_foreground_iou": float(min_foreground_iou),
        "min_per_id_iou": float(min_per_id_iou),
        "parity_pass": parity_pass,
        "efficiency_pass": efficiency_pass,
        "min_runtime_decrease_ratio": float(cli.min_runtime_decrease_ratio),
        "direct_total_runtime_sec": direct_runtime,
        "bank_total_runtime_sec": bank_runtime,
        "runtime_decrease_ratio": runtime_decrease_ratio,
        "bank_image_feature_source": image_feature_source,
        "bank_image_bank_storage_device": str(bank_feature.get("image_bank_storage_device", "")),
        "bank_video_bank_storage_device": str(bank_feature.get("video_bank_storage_device", "")),
        "bank_video_gpu_hot_window": int(bank_feature.get("video_gpu_hot_window", 0) or 0),
        "bank_image_bank_build_runtime_sec": image_build_runtime,
        "bank_video_bank_build_runtime_sec": video_build_runtime,
        "bank_runtime_with_feature_build_sec": bank_runtime_with_feature_build,
        "runtime_decrease_ratio_with_feature_build": runtime_decrease_ratio_with_feature_build,
        "bank_image_backbone_forward_count": image_count,
        "bank_image_expected_backbone_forward_count": image_expected,
        "bank_video_backbone_forward_count": video_count,
        "bank_video_expected_backbone_forward_count": video_expected,
        "bank_image_forward_leak_count": image_leaks,
        "bank_video_forward_leak_count": video_leaks,
        "direct_instrumentation": direct_inst,
        "bank_instrumentation": bank_inst,
        "frame_rows": frame_rows,
    }
    compare_path = out_dir / "p0_feature_bank_compare_summary.json"
    _write_json(compare_path, payload)
    print(json.dumps({"compare_summary": _rel(compare_path), "parity_pass": parity_pass, "efficiency_pass": efficiency_pass}, sort_keys=True), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="run direct or bank baseline-x-gapadaptive")
    run_p.add_argument("--mode", choices=["direct", "bank"], required=True)
    run_p.add_argument("--config", default=str(DEFAULT_CONFIG))
    run_p.add_argument("--scene-id", required=True)
    run_p.add_argument("--frame-start", type=int, default=None)
    run_p.add_argument("--frame-stride", type=int, default=None)
    run_p.add_argument("--frame-count", type=int, default=None)
    run_p.add_argument("--frame-ids", default=None)
    run_p.add_argument("--output-root", required=True)
    run_p.add_argument("--bank-storage-device", default="cuda", choices=["cuda", "cpu"])
    run_p.add_argument("--seed", type=int, default=None)
    run_p.set_defaults(func=run_mode)

    pair_p = sub.add_parser("pair", help="run direct and bank in one process with one loaded model pair")
    pair_p.add_argument("--config", default=str(DEFAULT_CONFIG))
    pair_p.add_argument("--scene-id", required=True)
    pair_p.add_argument("--frame-start", type=int, default=None)
    pair_p.add_argument("--frame-stride", type=int, default=None)
    pair_p.add_argument("--frame-count", type=int, default=None)
    pair_p.add_argument("--frame-ids", default=None)
    pair_p.add_argument("--direct-output-root", required=True)
    pair_p.add_argument("--bank-output-root", required=True)
    pair_p.add_argument("--compare-output-root", required=True)
    pair_p.add_argument("--bank-storage-device", default="cuda", choices=["cuda", "cpu"])
    pair_p.add_argument("--image-feature-source", default="bank", choices=["bank", "direct"])
    pair_p.add_argument("--image-bank-storage-device", default=None, choices=["cuda", "cpu"])
    pair_p.add_argument("--video-bank-storage-device", default=None, choices=["cuda", "cpu"])
    pair_p.add_argument("--video-gpu-hot-window", type=int, default=0)
    pair_p.add_argument("--min-runtime-decrease-ratio", type=float, default=0.2)
    pair_p.add_argument("--seed", type=int, default=None)
    pair_p.set_defaults(func=pair_mode)

    cmp_p = sub.add_parser("compare", help="compare direct and bank variant directories")
    cmp_p.add_argument("--direct-variant-dir", required=True)
    cmp_p.add_argument("--bank-variant-dir", required=True)
    cmp_p.add_argument("--output-root", required=True)
    cmp_p.add_argument("--min-runtime-decrease-ratio", type=float, default=0.2)
    cmp_p.set_defaults(func=compare_mode)
    return parser


def main() -> None:
    cli = build_parser().parse_args()
    cli.func(cli)


if __name__ == "__main__":
    main()
