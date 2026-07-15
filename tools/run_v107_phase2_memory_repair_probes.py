#!/usr/bin/env python3
"""Run v107 Phase2 blocker-repair probes without relaxing the Phase2 gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
STREAM3D_ROOT = ROOT / "Stream3D"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(STREAM3D_ROOT) not in sys.path:
    sys.path.insert(1, str(STREAM3D_ROOT))

import tools.audit_v105_baseline_x_sam2_twostage_tracking as base  # noqa: E402
import tools.run_v107_phase2_sam2_memory_microbenchmark as phase2  # noqa: E402
from tools.audit_v106_sam2_rolling_state import (  # noqa: E402
    install_rolling_state_support,
    reset_rolling_stats,
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


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@contextmanager
def timed_reconsolidation_call() -> Any:
    started = time.time()
    bucket = {"sec": 0.0}
    try:
        yield bucket
    finally:
        bucket["sec"] = float(time.time() - started)


def masks_for_source_ids(label: np.ndarray, source_ids: list[int]) -> np.ndarray:
    return np.stack([(label == int(obj_id)) for obj_id in source_ids], axis=0).astype(bool)


def add_masks_without_reconsolidation(
    predictor: Any,
    state: dict[str, Any],
    *,
    frame_idx: int,
    runtime_ids: list[int],
    masks: np.ndarray,
) -> float:
    import torch

    if masks.size == 0:
        return 0.0
    autocast_dtype = torch.float32
    try:
        autocast_dtype = next(predictor.parameters()).dtype
    except Exception:
        pass
    if bool(state.get("tracking_has_started", False)):
        state.get("frames_already_tracked", {}).pop(int(frame_idx), None)
    old_tracking_started = bool(state.get("tracking_has_started", False))
    state["tracking_has_started"] = False
    started = time.time()
    try:
        for obj_id, mask in zip(runtime_ids, masks.astype(bool), strict=False):
            mask_arg: Any = torch.from_numpy(mask.astype(np.float32))
            autocast_ctx = (
                torch.autocast("cuda", dtype=autocast_dtype)
                if autocast_dtype in {torch.bfloat16, torch.float16}
                else nullcontext()
            )
            with torch.inference_mode(), autocast_ctx:
                predictor.add_new_mask(
                    inference_state=state,
                    frame_idx=int(frame_idx),
                    obj_id=int(obj_id),
                    mask=mask_arg,
                )
    finally:
        state["tracking_has_started"] = old_tracking_started
    return float(time.time() - started)


def reconsolidate_with_predictor_dtype(predictor: Any, state: dict[str, Any]) -> None:
    import torch

    autocast_dtype = torch.float32
    try:
        autocast_dtype = next(predictor.parameters()).dtype
    except Exception:
        pass
    autocast_ctx = (
        torch.autocast("cuda", dtype=autocast_dtype)
        if autocast_dtype in {torch.bfloat16, torch.float16}
        else nullcontext()
    )
    with torch.inference_mode(), autocast_ctx:
        base.reconsolidate_stream_state_outputs(predictor, state)


def seed_state_at_frame(
    probe: phase2.Probe,
    seed_frame_idx: int,
    runtime_ids: list[int],
    source_ids: list[int],
) -> dict[str, Any]:
    state = probe.predictor.init_state(
        video_path=None,
        offload_video_to_cpu=False,
        offload_state_to_cpu=False,
        async_loading_frames=False,
    )
    probe.add_frame(state, int(seed_frame_idx))
    probe.add_masks(state, int(seed_frame_idx), runtime_ids, masks_for_source_ids(probe.labels[int(seed_frame_idx)], source_ids))
    probe.infer(state, int(seed_frame_idx))
    return state


def safe_evaluate(
    *,
    name: str,
    description: str,
    probe: phase2.Probe,
    control_state: dict[str, Any],
    variant_state: dict[str, Any],
    compare_ids: list[int],
    target_frame_idx: int,
    api_failures: list[str],
    add_runtime_sec: float = 0.0,
    remove_runtime_sec: float = 0.0,
    reconsolidation_runtime_sec: float = 0.0,
    global_id_mapping: dict[int, int] | None = None,
    mutation_kind: str = "sam2_state_mutation",
    notes: list[str] | None = None,
) -> dict[str, Any]:
    try:
        row = phase2.evaluate_variant(
            name=name,
            description=description,
            probe=probe,
            control_state=control_state,
            variant_state=variant_state,
            compare_ids=compare_ids,
            target_frame_idx=target_frame_idx,
            api_failures=api_failures,
            add_runtime_sec=add_runtime_sec,
            remove_runtime_sec=remove_runtime_sec,
            reconsolidation_runtime_sec=reconsolidation_runtime_sec,
            global_id_mapping=global_id_mapping,
        )
        row["probe_status"] = "completed"
    except Exception as exc:  # noqa: BLE001
        row = {
            "case": name,
            "description": description,
            "compare_ids": [int(v) for v in compare_ids],
            "target_frame_idx": int(target_frame_idx),
            "api_failure_count": int(len(api_failures) + 1),
            "api_failures": [*api_failures, f"evaluate:{type(exc).__name__}: {exc}"],
            "add_runtime_sec": float(add_runtime_sec),
            "remove_runtime_sec": float(remove_runtime_sec),
            "reconsolidation_runtime_sec": float(reconsolidation_runtime_sec),
            "global_id_mapping": {str(k): int(v) for k, v in (global_id_mapping or {}).items()},
            "passes": False,
            "probe_status": "evaluation_failed",
        }
    row["mutation_kind"] = mutation_kind
    row["notes"] = notes or []
    return row


def make_args(config_path: Path, scene_id: str, frame_ids: list[int], output_root: Path) -> SimpleNamespace:
    return phase2.make_args(config_path, scene_id, frame_ids, output_root)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--config", default=str(phase2.DEFAULT_CONFIG))
    parser.add_argument("--reference-run-root", default=str(phase2.DEFAULT_REFERENCE))
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
    phase2_dir = output_root / "phase2_repair"
    phase2_dir.mkdir(parents=True, exist_ok=True)
    reference_root = Path(args.reference_run_root)
    if not reference_root.is_absolute():
        reference_root = ROOT / reference_root
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path

    frame_ids = [int(args.frame_start) + i * int(args.frame_stride) for i in range(int(args.frame_count))]
    labels = [phase2.imread_label(phase2.label_path(reference_root, frame_id)) for frame_id in frame_ids]
    rgbs = [phase2.read_rgb(phase2.rgb_path(str(args.scene_id), frame_id)) for frame_id in frame_ids]
    selected = phase2.select_ids(labels)

    reset_rolling_stats()
    model_args = make_args(config_path, str(args.scene_id), frame_ids, output_root)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    model_started = time.time()
    models = base.setup_models(model_args)
    predictor = models["tracker_model"]
    install_rolling_state_support(predictor)
    model_load_sec = float(time.time() - model_started)

    probe = phase2.Probe(predictor, rgbs, labels, selected)
    old_hook = base.STREAM_INFER_TRACE_HOOK
    base.STREAM_INFER_TRACE_HOOK = probe.trace_hook
    rows: list[dict[str, Any]] = []
    try:
        base_ids = selected["base"]
        remove_ids = selected["remove"]
        admit_ids = selected["admit"]
        initial_ids = base_ids + remove_ids

        control = probe.new_state(base_ids)
        variant = probe.new_state(base_ids)
        with timed_reconsolidation_call() as bucket:
            reconsolidate_with_predictor_dtype(predictor, variant)
        rows.append(
            safe_evaluate(
                name="R0_noop_reconsolidate_on_base",
                description="Call reconsolidate_stream_state_outputs without changing the active object set.",
                probe=probe,
                control_state=control,
                variant_state=variant,
                compare_ids=base_ids,
                target_frame_idx=2,
                api_failures=[],
                reconsolidation_runtime_sec=bucket["sec"],
                notes=["Isolates whether reconsolidation alone perturbs other-object outputs."],
            )
        )

        control = probe.new_state(initial_ids)
        variant = probe.new_state(initial_ids)
        failures: list[str] = []
        remove_started = time.time()
        try:
            predictor.remove_object(variant, remove_ids[0], strict=True, need_output=True)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"remove_need_output_true:{type(exc).__name__}: {exc}")
        remove_sec = float(time.time() - remove_started)
        rows.append(
            safe_evaluate(
                name="R1_remove_one_need_output_true",
                description="Remove one active object with need_output=True instead of v106's need_output=False.",
                probe=probe,
                control_state=control,
                variant_state=variant,
                compare_ids=base_ids,
                target_frame_idx=2,
                api_failures=failures,
                remove_runtime_sec=remove_sec,
                notes=["Tests whether SAM2's optional remove output path preserves other objects better."],
            )
        )

        control = probe.new_state(initial_ids)
        variant = probe.new_state(initial_ids)
        failures = []
        remove_started = time.time()
        try:
            predictor.remove_object(variant, remove_ids[0], strict=True, need_output=False)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"remove:{type(exc).__name__}: {exc}")
        remove_sec = float(time.time() - remove_started)
        with timed_reconsolidation_call() as bucket:
            try:
                reconsolidate_with_predictor_dtype(predictor, variant)
            except Exception as exc:  # noqa: BLE001
                failures.append(f"reconsolidate_after_remove:{type(exc).__name__}: {exc}")
        rows.append(
            safe_evaluate(
                name="R2_remove_one_then_reconsolidate",
                description="Remove one active object, then explicitly reconsolidate packed state outputs.",
                probe=probe,
                control_state=control,
                variant_state=variant,
                compare_ids=base_ids,
                target_frame_idx=2,
                api_failures=failures,
                remove_runtime_sec=remove_sec,
                reconsolidation_runtime_sec=bucket["sec"],
                notes=["Tests whether explicit post-remove reconsolidation repairs the small remove-only drift."],
            )
        )

        control = probe.new_state(base_ids)
        variant = probe.new_state(base_ids)
        failures = []
        probe.add_frame(variant, 1)
        try:
            add_sec = add_masks_without_reconsolidation(
                predictor,
                variant,
                frame_idx=1,
                runtime_ids=remove_ids,
                masks=masks_for_source_ids(labels[1], remove_ids),
            )
        except Exception as exc:  # noqa: BLE001
            add_sec = 0.0
            failures.append(f"add_without_reconsolidation:{type(exc).__name__}: {exc}")
        rows.append(
            safe_evaluate(
                name="R3_posttracking_batch_add_without_reconsolidation",
                description="Add two new objects after tracking starts but skip v106 reconsolidation.",
                probe=probe,
                control_state=control,
                variant_state=variant,
                compare_ids=base_ids,
                target_frame_idx=2,
                api_failures=failures,
                add_runtime_sec=add_sec,
                notes=["Tests whether v106 reconsolidation is the cause of A1 drift or merely required for a usable state."],
            )
        )

        control = probe.new_state(initial_ids)
        variant = probe.new_state(initial_ids)
        failures = []
        old_id = remove_ids[0]
        new_id = int(old_id) + 10000
        remove_started = time.time()
        try:
            predictor.remove_object(variant, old_id, strict=True, need_output=False)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"remove:{type(exc).__name__}: {exc}")
        remove_sec = float(time.time() - remove_started)
        probe.add_frame(variant, 1)
        try:
            add_sec = add_masks_without_reconsolidation(
                predictor,
                variant,
                frame_idx=1,
                runtime_ids=[new_id],
                masks=masks_for_source_ids(labels[1], [old_id]),
            )
        except Exception as exc:  # noqa: BLE001
            add_sec = 0.0
            failures.append(f"readd_new_runtime_without_reconsolidation:{type(exc).__name__}: {exc}")
        rows.append(
            safe_evaluate(
                name="R4_remove_readd_new_runtime_without_reconsolidation",
                description="Use the planned new-runtime-id strategy but skip reconsolidation after re-add.",
                probe=probe,
                control_state=control,
                variant_state=variant,
                compare_ids=base_ids + [remove_ids[1]],
                target_frame_idx=2,
                api_failures=failures,
                add_runtime_sec=add_sec,
                remove_runtime_sec=remove_sec,
                global_id_mapping={new_id: old_id},
                notes=["Tests the documented repair direction plus delayed reconsolidation."],
            )
        )

        control = probe.new_state(initial_ids)
        variant = seed_state_at_frame(probe, 1, initial_ids, initial_ids)
        rows.append(
            safe_evaluate(
                name="R5_reset_rebuild_same_active_set_at_transaction_frame",
                description="Reset/rebuild a fresh state from frozen masks at transaction frame with the same active set.",
                probe=probe,
                control_state=control,
                variant_state=variant,
                compare_ids=initial_ids,
                target_frame_idx=2,
                api_failures=[],
                mutation_kind="sam2_state_reset_rebuild",
                notes=["Tests whether full state rebuild can preserve exact outputs for unchanged active IDs."],
            )
        )

        new_admit_ids = [int(obj_id) + 10000 for obj_id in admit_ids]
        control = probe.new_state(initial_ids)
        variant = seed_state_at_frame(probe, 1, base_ids + new_admit_ids, base_ids + admit_ids)
        rows.append(
            safe_evaluate(
                name="R6_reset_rebuild_demote_admit_at_transaction_frame",
                description="Reset/rebuild at transaction frame after demoting two IDs and admitting two new runtime IDs.",
                probe=probe,
                control_state=control,
                variant_state=variant,
                compare_ids=base_ids,
                target_frame_idx=2,
                api_failures=[],
                global_id_mapping={new_id: old_id for new_id, old_id in zip(new_admit_ids, admit_ids, strict=True)},
                mutation_kind="sam2_state_reset_rebuild",
                notes=["Tests whether reset/rebuild is a viable alternative to in-place add/remove transactions."],
            )
        )

        control = probe.new_state(base_ids)
        variant = probe.new_state(base_ids)
        rows.append(
            safe_evaluate(
                name="R7_shadow_pending_ledger_no_sam_mutation",
                description="Keep probation/reactivation decisions in an external ledger and do not mutate SAM2 state.",
                probe=probe,
                control_state=control,
                variant_state=variant,
                compare_ids=base_ids,
                target_frame_idx=2,
                api_failures=[],
                mutation_kind="external_shadow_ledger_only",
                global_id_mapping={int(remove_ids[0]) + 10000: remove_ids[0]},
                notes=[
                    "This can prove exact parity for shadow bookkeeping only.",
                    "It is not a pass for live SAM2 demotion/reactivation because active SAM2 memory is unchanged.",
                ],
            )
        )
    finally:
        base.STREAM_INFER_TRACE_HOOK = old_hook

    logit_path = phase2_dir / "phase2_repair_raw_logit_stats.parquet"
    pd.DataFrame(probe.logit_rows).to_parquet(logit_path, index=False)
    pd.DataFrame(probe.logit_rows).to_csv(phase2_dir / "phase2_repair_raw_logit_stats.csv", index=False)

    mutating_kinds = {"sam2_state_mutation", "sam2_state_reset_rebuild"}
    mutating_pass_rows = [row for row in rows if row.get("mutation_kind") in mutating_kinds and bool(row.get("passes"))]
    shadow_pass_rows = [row for row in rows if row.get("mutation_kind") == "external_shadow_ledger_only" and bool(row.get("passes"))]
    if mutating_pass_rows:
        decision = "PASS_PHASE2_REPAIR_FOUND_SAFE_SAM2_MUTATION_CANDIDATE"
    elif shadow_pass_rows:
        decision = "NO_GO_PHASE2_REPAIR_ONLY_SHADOW_LEDGER_EXACT"
    else:
        decision = "NO_GO_PHASE2_REPAIR_NO_SAFE_MUTATION_OR_SHADOW_FALLBACK"

    summary = {
        "schema_version": "stream4d_v107_phase2_memory_repair_probe_summary_v1",
        "created_unix_time": time.time(),
        "scene_id": str(args.scene_id),
        "frame_ids": [int(v) for v in frame_ids],
        "reference_run_root": {"path": rel(reference_root), "summary_sha256": sha256_file(reference_root / "summary.json")},
        "config": {"path": rel(config_path), "sha256": sha256_file(config_path)},
        "selected_ids": selected,
        "model_load_sec": model_load_sec,
        "raw_logit_stats": {"path": rel(logit_path), "row_count": int(len(probe.logit_rows))},
        "cases": rows,
        "case_count": int(len(rows)),
        "case_pass_count": int(sum(1 for row in rows if bool(row.get("passes")))),
        "mutating_case_pass_count": int(len(mutating_pass_rows)),
        "shadow_ledger_case_pass_count": int(len(shadow_pass_rows)),
        "safe_sam2_mutation_candidate_found": bool(mutating_pass_rows),
        "shadow_ledger_exact_parity_fallback_found": bool(shadow_pass_rows),
        "phase2_original_gate_still_passes": False,
        "live_lifecycle_mutation_allowed": bool(mutating_pass_rows),
        "shadow_only_lifecycle_bookkeeping_allowed": bool(shadow_pass_rows),
        "decision": decision,
        "gate_note": (
            "These are repair probes after the original Phase2 No-Go. "
            "A shadow-ledger-only pass does not satisfy the SAM2 memory demotion/reactivation gate."
        ),
    }
    write_json(phase2_dir / "memory_repair_probe_summary.json", summary)
    write_json(phase2_dir / "memory_repair_probe_rows.json", {"rows": rows, "row_count": len(rows)})
    write_json(
        output_root / "run_summary.json",
        {
            "schema_version": "stream4d_v107_phase2_memory_repair_probe_run_v1",
            "summary": rel(phase2_dir / "memory_repair_probe_summary.json"),
            "decision": decision,
            "safe_sam2_mutation_candidate_found": summary["safe_sam2_mutation_candidate_found"],
            "shadow_ledger_exact_parity_fallback_found": summary["shadow_ledger_exact_parity_fallback_found"],
            "case_pass_count": summary["case_pass_count"],
            "case_count": summary["case_count"],
        },
    )
    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "decision": decision,
                "case_pass_count": summary["case_pass_count"],
                "case_count": summary["case_count"],
                "safe_sam2_mutation_candidate_found": summary["safe_sam2_mutation_candidate_found"],
                "shadow_ledger_exact_parity_fallback_found": summary["shadow_ledger_exact_parity_fallback_found"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
