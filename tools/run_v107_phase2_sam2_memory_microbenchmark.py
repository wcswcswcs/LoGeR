#!/usr/bin/env python3
"""Run v107 Phase2 SAM2 memory mechanics microbenchmark A0-A6."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
STREAM3D_ROOT = ROOT / "Stream3D"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(STREAM3D_ROOT) not in sys.path:
    sys.path.insert(1, str(STREAM3D_ROOT))

import tools.audit_v105_baseline_x_sam2_twostage_tracking as base  # noqa: E402
from tools.audit_v106_sam2_rolling_state import (  # noqa: E402
    _rolling_add_frame,
    install_rolling_state_support,
    reset_rolling_stats,
)


DEFAULT_CONFIG = (
    ROOT
    / "configs/v106/v106_stateful_sam2_rolling_scene_stream_area20k_preprune6_maxvis45_labelcompact_noempty.yaml"
)
DEFAULT_REFERENCE = (
    ROOT
    / "Stream3D/outputs/audit/"
    / "v106_stateful_sam2_rolling_scene0050_area20k_e1_preprune6_maxvis45_labelcompact_noempty_full90_gpu6_20260713_1505"
    / "v106_stateful_sam2_rolling_scene_stream"
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def imread_label(path: Path) -> np.ndarray:
    label = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if label is None:
        raise FileNotFoundError(path)
    if label.ndim == 3:
        label = label[:, :, 0]
    return label


def label_path(reference_root: Path, frame_id: int) -> Path:
    return reference_root / "labels" / f"frame_{int(frame_id):06d}.png"


def rgb_path(scene_id: str, frame_id: int) -> Path:
    return ROOT / "Stream3D/data/scannet/processed" / scene_id / "color" / f"{int(frame_id)}.jpg"


def read_rgb(path: Path) -> np.ndarray:
    return base.read_rgb(path)


def masks_for_ids(label: np.ndarray, ids: list[int]) -> np.ndarray:
    return np.stack([(label == int(obj_id)) for obj_id in ids], axis=0).astype(bool)


def make_args(config_path: Path, scene_id: str, frame_ids: list[int], output_root: Path) -> SimpleNamespace:
    config = base.load_config(config_path)
    cli = SimpleNamespace(
        config=str(config_path),
        scene_id=scene_id,
        rgb_root=None,
        frame_start=int(frame_ids[0]),
        frame_stride=int(frame_ids[1] - frame_ids[0]) if len(frame_ids) > 1 else 5,
        frame_count=int(len(frame_ids)),
        frame_ids=",".join(str(v) for v in frame_ids),
        output_root=str(output_root),
        seed=107,
        birth_dump_dir="",
    )
    args = base.make_args(config, cli)
    args.model_dtype = str(config.get("sam2", {}).get("model_dtype", "bfloat16"))
    args.runtime_num_maskmem = int(config.get("sam2", {}).get("runtime_num_maskmem", 3))
    args.runtime_max_obj_ptrs_in_encoder = int(config.get("sam2", {}).get("runtime_max_obj_ptrs_in_encoder", 8))
    args.runtime_max_cond_frames_in_attn = int(config.get("sam2", {}).get("runtime_max_cond_frames_in_attn", 4))
    args.offload_video_to_cpu = False
    args.offload_state_to_cpu = False
    return args


def mask_for_id(ids: np.ndarray, masks: np.ndarray, obj_id: int) -> np.ndarray | None:
    matches = np.where(ids.astype(np.int64) == int(obj_id))[0]
    if matches.size == 0:
        return None
    return masks[int(matches[0])].astype(bool, copy=False)


def mask_iou(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None:
        return None
    union = np.logical_or(a, b)
    denom = int(np.count_nonzero(union))
    if denom == 0:
        return 1.0
    return float(np.count_nonzero(np.logical_and(a, b)) / denom)


def union_iou(ids_a: np.ndarray, masks_a: np.ndarray, ids_b: np.ndarray, masks_b: np.ndarray, compare_ids: list[int]) -> float | None:
    selected_a = [mask_for_id(ids_a, masks_a, obj_id) for obj_id in compare_ids]
    selected_b = [mask_for_id(ids_b, masks_b, obj_id) for obj_id in compare_ids]
    if any(mask is None for mask in selected_a + selected_b):
        return None
    union_a = np.any(np.stack(selected_a, axis=0), axis=0)
    union_b = np.any(np.stack(selected_b, axis=0), axis=0)
    return mask_iou(union_a, union_b)


def state_tensor_bytes(obj: Any, seen: set[int] | None = None) -> int:
    import torch

    if seen is None:
        seen = set()
    oid = id(obj)
    if oid in seen:
        return 0
    seen.add(oid)
    if torch.is_tensor(obj):
        return int(obj.numel() * obj.element_size())
    if isinstance(obj, dict):
        return sum(state_tensor_bytes(v, seen) for v in obj.values())
    if isinstance(obj, (list, tuple, set)):
        return sum(state_tensor_bytes(v, seen) for v in obj)
    return 0


@contextmanager
def timed_reconsolidation():
    timings: list[float] = []
    original = base.reconsolidate_stream_state_outputs

    def wrapped(predictor: Any, state: dict[str, Any]) -> None:
        started = time.time()
        try:
            return original(predictor, state)
        finally:
            timings.append(float(time.time() - started))

    base.reconsolidate_stream_state_outputs = wrapped
    try:
        yield timings
    finally:
        base.reconsolidate_stream_state_outputs = original


class Probe:
    def __init__(self, predictor: Any, rgbs: list[np.ndarray], labels: list[np.ndarray], selected: dict[str, list[int]]):
        self.predictor = predictor
        self.rgbs = rgbs
        self.labels = labels
        self.selected = selected
        self.logit_rows: list[dict[str, Any]] = []

    def trace_hook(self, frame_idx: int, obj_ids: Any, logits: Any, _masks: Any) -> None:
        import torch

        with torch.no_grad():
            tensor = logits.detach().float()
            flat = tensor.flatten(1)
            means = flat.mean(dim=1).detach().cpu().tolist()
            mins = flat.min(dim=1).values.detach().cpu().tolist()
            maxs = flat.max(dim=1).values.detach().cpu().tolist()
            stds = flat.std(dim=1, unbiased=False).detach().cpu().tolist()
            positive = (flat > 0.0).sum(dim=1).detach().cpu().tolist()
            shape = [int(v) for v in tensor.shape]
            ids = [int(v) for v in obj_ids]
        for idx, obj_id in enumerate(ids):
            self.logit_rows.append(
                {
                    "frame_idx": int(frame_idx),
                    "runtime_id": int(obj_id),
                    "logit_shape": shape,
                    "raw_logit_mean": float(means[idx]),
                    "raw_logit_min": float(mins[idx]),
                    "raw_logit_max": float(maxs[idx]),
                    "raw_logit_std": float(stds[idx]),
                    "positive_logit_pixel_count": int(positive[idx]),
                }
            )

    def add_frame(self, state: dict[str, Any], frame_idx: int) -> None:
        _rolling_add_frame(self.predictor, state, frame_idx=int(frame_idx), rgb=self.rgbs[int(frame_idx)])

    def add_masks(self, state: dict[str, Any], frame_idx: int, ids: list[int], masks: np.ndarray) -> tuple[float, float]:
        with timed_reconsolidation() as recon_times:
            started = time.time()
            base.add_masks_to_stream_state(
                self.predictor,
                state,
                tracker="sam2",
                frame_idx=int(frame_idx),
                obj_ids=np.asarray(ids, dtype=np.int64),
                masks=masks.astype(bool, copy=False),
            )
            add_runtime = float(time.time() - started)
        return add_runtime, float(sum(recon_times))

    def infer(self, state: dict[str, Any], frame_idx: int) -> tuple[np.ndarray, np.ndarray]:
        self.add_frame(state, int(frame_idx))
        return base.infer_stream_frame(self.predictor, state, frame_idx=int(frame_idx))

    def new_state(self, initial_ids: list[int]) -> dict[str, Any]:
        state = self.predictor.init_state(
            video_path=None,
            offload_video_to_cpu=False,
            offload_state_to_cpu=False,
            async_loading_frames=False,
        )
        self.add_frame(state, 0)
        self.add_masks(state, 0, initial_ids, masks_for_ids(self.labels[0], initial_ids))
        self.infer(state, 0)
        self.infer(state, 1)
        return state


def logit_stats_for(rows: list[dict[str, Any]], frame_idx: int, obj_id: int) -> dict[str, float] | None:
    for row in reversed(rows):
        if int(row["frame_idx"]) == int(frame_idx) and int(row["runtime_id"]) == int(obj_id):
            return row
    return None


def evaluate_variant(
    *,
    name: str,
    description: str,
    probe: Probe,
    control_state: dict[str, Any],
    variant_state: dict[str, Any],
    compare_ids: list[int],
    target_frame_idx: int,
    api_failures: list[str],
    add_runtime_sec: float,
    remove_runtime_sec: float,
    reconsolidation_runtime_sec: float,
    global_id_mapping: dict[int, int] | None = None,
) -> dict[str, Any]:
    import torch

    before_rows = len(probe.logit_rows)
    control_ids, control_masks = probe.infer(control_state, target_frame_idx)
    control_logit_rows = probe.logit_rows[before_rows:]
    before_rows = len(probe.logit_rows)
    variant_ids, variant_masks = probe.infer(variant_state, target_frame_idx)
    variant_logit_rows = probe.logit_rows[before_rows:]

    per_id_iou = {
        str(obj_id): mask_iou(
            mask_for_id(control_ids, control_masks, obj_id),
            mask_for_id(variant_ids, variant_masks, obj_id),
        )
        for obj_id in compare_ids
    }
    logit_deltas = {}
    for obj_id in compare_ids:
        c = logit_stats_for(control_logit_rows, target_frame_idx, obj_id)
        v = logit_stats_for(variant_logit_rows, target_frame_idx, obj_id)
        if c is None or v is None:
            logit_deltas[str(obj_id)] = None
            continue
        logit_deltas[str(obj_id)] = {
            key: abs(float(c[key]) - float(v[key]))
            for key in ("raw_logit_mean", "raw_logit_min", "raw_logit_max", "raw_logit_std")
        }
    valid_ious = [float(v) for v in per_id_iou.values() if v is not None]
    max_logit_delta = 0.0
    missing_logit_delta = False
    for value in logit_deltas.values():
        if value is None:
            missing_logit_delta = True
        else:
            max_logit_delta = max(max_logit_delta, max(float(v) for v in value.values()))
    fg_iou = union_iou(control_ids, control_masks, variant_ids, variant_masks, compare_ids)
    exact_other_object_mask_parity = bool(valid_ious and min(valid_ious) == 1.0 and fg_iou == 1.0)
    exact_other_object_logit_stat_parity = bool(not missing_logit_delta and max_logit_delta == 0.0)
    return {
        "case": name,
        "description": description,
        "compare_ids": [int(v) for v in compare_ids],
        "target_frame_idx": int(target_frame_idx),
        "control_obj_ids": [int(v) for v in control_ids.tolist()],
        "variant_obj_ids": [int(v) for v in variant_ids.tolist()],
        "per_id_iou": per_id_iou,
        "min_other_object_iou": min(valid_ious) if valid_ious else None,
        "foreground_union_iou": fg_iou,
        "logit_stat_abs_delta": logit_deltas,
        "max_logit_stat_abs_delta": max_logit_delta,
        "exact_other_object_mask_parity": exact_other_object_mask_parity,
        "exact_other_object_logit_stat_parity": exact_other_object_logit_stat_parity,
        "state_object_count": int(len(variant_state.get("obj_ids", []))),
        "state_tensor_bytes": int(state_tensor_bytes(variant_state)),
        "add_runtime_sec": float(add_runtime_sec),
        "remove_runtime_sec": float(remove_runtime_sec),
        "reconsolidation_runtime_sec": float(reconsolidation_runtime_sec),
        "peak_cuda_memory_mb": float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0),
        "api_failure_count": int(len(api_failures)),
        "api_failures": api_failures,
        "global_id_mapping": {str(k): int(v) for k, v in (global_id_mapping or {}).items()},
        "passes": bool(exact_other_object_mask_parity and exact_other_object_logit_stat_parity and not api_failures),
    }


def select_ids(labels: list[np.ndarray]) -> dict[str, list[int]]:
    common = set(int(v) for v in np.unique(labels[0]).tolist() if int(v) > 0)
    for label in labels[1:3]:
        common &= set(int(v) for v in np.unique(label).tolist() if int(v) > 0)
    areas = []
    for obj_id in common:
        area = int(np.count_nonzero(labels[0] == int(obj_id)))
        if area > 512:
            areas.append((area, int(obj_id)))
    areas.sort(reverse=True)
    ids = [obj_id for _area, obj_id in areas[:8]]
    if len(ids) < 8:
        raise RuntimeError(f"not enough common object IDs for Phase2 benchmark: {ids}")
    return {
        "base": ids[:4],
        "remove": ids[4:6],
        "admit": ids[6:8],
        "a0_all": ids[:6],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--reference-run-root", default=str(DEFAULT_REFERENCE))
    parser.add_argument("--scene-id", default="scene0050_00")
    parser.add_argument("--frame-start", type=int, default=4160)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--frame-count", type=int, default=4)
    parser.add_argument("--gpu", default="6")
    args = parser.parse_args()

    if str(args.gpu).strip():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu).strip()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    phase2 = output_root / "phase2"
    phase2.mkdir(parents=True, exist_ok=True)
    reference_root = Path(args.reference_run_root)
    if not reference_root.is_absolute():
        reference_root = ROOT / reference_root
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path

    frame_ids = [int(args.frame_start) + i * int(args.frame_stride) for i in range(int(args.frame_count))]
    labels = [imread_label(label_path(reference_root, frame_id)) for frame_id in frame_ids]
    rgbs = [read_rgb(rgb_path(str(args.scene_id), frame_id)) for frame_id in frame_ids]
    selected = select_ids(labels)

    reset_rolling_stats()
    model_args = make_args(config_path, str(args.scene_id), frame_ids, output_root)
    model_started = time.time()
    models = base.setup_models(model_args)
    predictor = models["tracker_model"]
    install_rolling_state_support(predictor)
    model_load_sec = float(time.time() - model_started)
    probe = Probe(predictor, rgbs, labels, selected)
    old_hook = base.STREAM_INFER_TRACE_HOOK
    base.STREAM_INFER_TRACE_HOOK = probe.trace_hook
    rows: list[dict[str, Any]] = []
    try:
        # A0: add N objects before tracking, mechanism pass if initial state runs.
        a0_state = probe.new_state(selected["a0_all"])
        a0_control = probe.new_state(selected["a0_all"])
        rows.append(
            evaluate_variant(
                name="A0_pretracking_add_N",
                description="Add six frozen reference masks before tracking starts.",
                probe=probe,
                control_state=a0_control,
                variant_state=a0_state,
                compare_ids=selected["a0_all"],
                target_frame_idx=2,
                api_failures=[],
                add_runtime_sec=0.0,
                remove_runtime_sec=0.0,
                reconsolidation_runtime_sec=0.0,
            )
        )

        # A1: post-start batch add.
        control = probe.new_state(selected["base"])
        variant = probe.new_state(selected["base"])
        probe.add_frame(variant, 1)
        add_sec, recon_sec = probe.add_masks(variant, 1, selected["remove"], masks_for_ids(labels[1], selected["remove"]))
        rows.append(
            evaluate_variant(
                name="A1_posttracking_batch_add",
                description="Add two frozen masks after tracking has started via v106 workaround.",
                probe=probe,
                control_state=control,
                variant_state=variant,
                compare_ids=selected["base"],
                target_frame_idx=2,
                api_failures=[],
                add_runtime_sec=add_sec,
                remove_runtime_sec=0.0,
                reconsolidation_runtime_sec=recon_sec,
            )
        )

        # A2: remove one object.
        initial = selected["base"] + selected["remove"]
        control = probe.new_state(initial)
        variant = probe.new_state(initial)
        remove_failures: list[str] = []
        remove_started = time.time()
        try:
            predictor.remove_object(variant, selected["remove"][0], strict=True, need_output=False)
        except Exception as exc:  # noqa: BLE001
            remove_failures.append(f"{type(exc).__name__}: {exc}")
        remove_sec = float(time.time() - remove_started)
        rows.append(
            evaluate_variant(
                name="A2_remove_one",
                description="Remove one active SAM2 object with public remove_object.",
                probe=probe,
                control_state=control,
                variant_state=variant,
                compare_ids=selected["base"],
                target_frame_idx=2,
                api_failures=remove_failures,
                add_runtime_sec=0.0,
                remove_runtime_sec=remove_sec,
                reconsolidation_runtime_sec=0.0,
            )
        )

        # A3: remove multiple objects.
        control = probe.new_state(initial)
        variant = probe.new_state(initial)
        failures = []
        remove_started = time.time()
        for obj_id in selected["remove"]:
            try:
                predictor.remove_object(variant, obj_id, strict=True, need_output=False)
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{obj_id}:{type(exc).__name__}: {exc}")
        remove_sec = float(time.time() - remove_started)
        rows.append(
            evaluate_variant(
                name="A3_remove_multiple",
                description="Remove two active SAM2 objects with public remove_object.",
                probe=probe,
                control_state=control,
                variant_state=variant,
                compare_ids=selected["base"],
                target_frame_idx=2,
                api_failures=failures,
                add_runtime_sec=0.0,
                remove_runtime_sec=remove_sec,
                reconsolidation_runtime_sec=0.0,
            )
        )

        # A4: remove then re-add same runtime ID.
        control = probe.new_state(initial)
        variant = probe.new_state(initial)
        failures = []
        remove_started = time.time()
        try:
            predictor.remove_object(variant, selected["remove"][0], strict=True, need_output=False)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"remove:{type(exc).__name__}: {exc}")
        remove_sec = float(time.time() - remove_started)
        probe.add_frame(variant, 1)
        add_sec, recon_sec = probe.add_masks(
            variant,
            1,
            [selected["remove"][0]],
            masks_for_ids(labels[1], [selected["remove"][0]]),
        )
        rows.append(
            evaluate_variant(
                name="A4_remove_readd_same_runtime_id",
                description="Remove one object, then re-add the same numeric runtime ID.",
                probe=probe,
                control_state=control,
                variant_state=variant,
                compare_ids=selected["base"] + [selected["remove"][1]],
                target_frame_idx=2,
                api_failures=failures,
                add_runtime_sec=add_sec,
                remove_runtime_sec=remove_sec,
                reconsolidation_runtime_sec=recon_sec,
                global_id_mapping={selected["remove"][0]: selected["remove"][0]},
            )
        )

        # A5: remove then re-add new runtime ID mapped to same global ID.
        control = probe.new_state(initial)
        variant = probe.new_state(initial)
        failures = []
        old_id = selected["remove"][0]
        new_id = int(old_id) + 10000
        remove_started = time.time()
        try:
            predictor.remove_object(variant, old_id, strict=True, need_output=False)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"remove:{type(exc).__name__}: {exc}")
        remove_sec = float(time.time() - remove_started)
        probe.add_frame(variant, 1)
        add_sec, recon_sec = probe.add_masks(variant, 1, [new_id], masks_for_ids(labels[1], [old_id]))
        rows.append(
            evaluate_variant(
                name="A5_remove_readd_new_runtime_id_same_global",
                description="Remove one object, then re-add with new runtime ID mapped to the old global ID.",
                probe=probe,
                control_state=control,
                variant_state=variant,
                compare_ids=selected["base"] + [selected["remove"][1]],
                target_frame_idx=2,
                api_failures=failures,
                add_runtime_sec=add_sec,
                remove_runtime_sec=remove_sec,
                reconsolidation_runtime_sec=recon_sec,
                global_id_mapping={new_id: old_id},
            )
        )

        # A6: batch demote + batch admit.
        control = probe.new_state(initial)
        variant = probe.new_state(initial)
        failures = []
        remove_started = time.time()
        for obj_id in selected["remove"]:
            try:
                predictor.remove_object(variant, obj_id, strict=True, need_output=False)
            except Exception as exc:  # noqa: BLE001
                failures.append(f"remove:{obj_id}:{type(exc).__name__}: {exc}")
        remove_sec = float(time.time() - remove_started)
        new_ids = [int(obj_id) + 10000 for obj_id in selected["admit"]]
        probe.add_frame(variant, 1)
        add_sec, recon_sec = probe.add_masks(variant, 1, new_ids, masks_for_ids(labels[1], selected["admit"]))
        rows.append(
            evaluate_variant(
                name="A6_batch_demote_batch_admit",
                description="Remove two active objects, then admit two new runtime IDs in one batch.",
                probe=probe,
                control_state=control,
                variant_state=variant,
                compare_ids=selected["base"],
                target_frame_idx=2,
                api_failures=failures,
                add_runtime_sec=add_sec,
                remove_runtime_sec=remove_sec,
                reconsolidation_runtime_sec=recon_sec,
                global_id_mapping={new_id: old_id for new_id, old_id in zip(new_ids, selected["admit"], strict=True)},
            )
        )
    finally:
        base.STREAM_INFER_TRACE_HOOK = old_hook

    logit_path = phase2 / "phase2_raw_logit_stats.parquet"
    pd.DataFrame(probe.logit_rows).to_parquet(logit_path, index=False)
    pd.DataFrame(probe.logit_rows).to_csv(phase2 / "phase2_raw_logit_stats.csv", index=False)
    all_cases_pass = all(row["passes"] for row in rows)
    summary = {
        "schema_version": "stream4d_v107_phase2_sam2_memory_microbenchmark_summary_v1",
        "created_unix_time": time.time(),
        "scene_id": str(args.scene_id),
        "frame_ids": [int(v) for v in frame_ids],
        "reference_run_root": {"path": rel(reference_root), "summary_sha256": sha256_file(reference_root / "summary.json")},
        "config": {"path": rel(config_path), "sha256": sha256_file(config_path)},
        "selected_ids": selected,
        "model_load_sec": model_load_sec,
        "raw_logit_stats": {"path": rel(logit_path), "row_count": int(len(probe.logit_rows))},
        "cases": rows,
        "case_pass_count": int(sum(1 for row in rows if row["passes"])),
        "case_count": int(len(rows)),
        "phase2_passes": bool(all_cases_pass),
        "live_lifecycle_mutation_allowed": bool(all_cases_pass),
        "decision": "PASS_PHASE2_SAM2_MEMORY_MECHANICS" if all_cases_pass else "NO_GO_PHASE2_SAM2_MEMORY_PARITY_FAILED",
        "gate": "Selected runtime/global-id strategy must not change other objects; exact mask and logit-stat parity required.",
        "repair_guidance_if_failed": (
            "Use new runtime IDs plus stable global mapping; do not force numeric SAM ID reuse. "
            "If add/remove still changes other-object masks/logits, live lifecycle mutation remains blocked."
        ),
    }
    write_json(phase2 / "memory_microbenchmark_summary.json", summary)
    write_json(phase2 / "memory_microbenchmark_rows.json", {"rows": rows, "row_count": len(rows)})
    write_json(
        output_root / "run_summary.json",
        {
            "schema_version": "stream4d_v107_phase2_sam2_memory_microbenchmark_run_v1",
            "summary": rel(phase2 / "memory_microbenchmark_summary.json"),
            "decision": summary["decision"],
            "phase2_passes": summary["phase2_passes"],
            "case_pass_count": summary["case_pass_count"],
            "case_count": summary["case_count"],
        },
    )
    print(json.dumps({"output_root": str(output_root), "decision": summary["decision"], "case_pass_count": summary["case_pass_count"], "case_count": summary["case_count"]}, sort_keys=True))
    return 0 if summary["phase2_passes"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
