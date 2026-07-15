#!/usr/bin/env python3
"""Build a v107 Phase11 continuation freeze/casebook from measured artifacts.

This intentionally does not claim a formal holdout pass. It packages the
available Phase10/Phase9 evidence after the user requested continuing past
diagnostic gates.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIT_ROOT = REPO_ROOT / "Stream3D/outputs/audit"
SAM2_CHECKPOINT = REPO_ROOT / "Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt"
SCHEDULER_SCRIPT = REPO_ROOT / "tools/run_v107_phase8_g3_rolling_scheduler_smoke.py"
PLAN_DOC = REPO_ROOT / "docs/stream4d_v107_recoverability_aware_lifecycle_memory_plan.md"
BUILDER_SCRIPT = REPO_ROOT / "tools/build_v107_phase11_frozen_casebook.py"


RUN_SPECS: list[dict[str, str]] = [
    {
        "variant": "geometry_guided_method",
        "scene_id": "scene0011_00",
        "scheduler_root": "Stream3D/outputs/audit/v107_phase10_scene0011_g3_scheduler_smoke90_20260714_012021",
        "fidelity_root": "Stream3D/outputs/audit/v107_phase10_scene0011_fidelity_diag90_20260714_012311",
        "ledger_root": "Stream3D/outputs/audit/v107_phase10_scene0011_lifecycle_ledger90_20260714_012324",
    },
    {
        "variant": "geometry_guided_method",
        "scene_id": "scene0050_00",
        "scheduler_root": "Stream3D/outputs/audit/v107_phase10_scene0050_cold_main90_20260714_013015",
        "fidelity_root": "Stream3D/outputs/audit/v107_phase10_scene0050_cold_main90_fidelity_diag_20260714_013238",
        "ledger_root": "Stream3D/outputs/audit/v107_phase10_scene0050_cold_main90_lifecycle_ledger_20260714_013242",
    },
    {
        "variant": "no_transaction",
        "scene_id": "scene0011_00",
        "scheduler_root": "Stream3D/outputs/audit/v107_phase10_ablation_scene0011_no_transaction90_20260714_013418",
        "fidelity_root": "Stream3D/outputs/audit/v107_phase10_ablation_scene0011_no_transaction90_fidelity_diag_20260714_013652",
        "ledger_root": "Stream3D/outputs/audit/v107_phase10_ablation_scene0011_no_transaction90_lifecycle_ledger_20260714_013656",
    },
    {
        "variant": "no_transaction",
        "scene_id": "scene0050_00",
        "scheduler_root": "Stream3D/outputs/audit/v107_phase10_ablation_scene0050_no_transaction90_20260714_015511",
        "fidelity_root": "Stream3D/outputs/audit/v107_phase10_ablation_scene0050_no_transaction90_fidelity_diag_20260714_015710",
        "ledger_root": "Stream3D/outputs/audit/v107_phase10_ablation_scene0050_no_transaction90_lifecycle_ledger_20260714_015713",
    },
    {
        "variant": "no_watcher",
        "scene_id": "scene0011_00",
        "scheduler_root": "Stream3D/outputs/audit/v107_phase10_ablation_scene0011_no_watcher90_20260714_015905",
        "fidelity_root": "Stream3D/outputs/audit/v107_phase10_ablation_scene0011_no_watcher90_fidelity_diag_20260714_020046",
        "ledger_root": "Stream3D/outputs/audit/v107_phase10_ablation_scene0011_no_watcher90_lifecycle_ledger_20260714_020049",
    },
    {
        "variant": "no_watcher",
        "scene_id": "scene0050_00",
        "scheduler_root": "Stream3D/outputs/audit/v107_phase10_ablation_scene0050_no_watcher90_20260714_013716",
        "fidelity_root": "Stream3D/outputs/audit/v107_phase10_ablation_scene0050_no_watcher90_fidelity_diag_20260714_014206",
        "ledger_root": "Stream3D/outputs/audit/v107_phase10_ablation_scene0050_no_watcher90_lifecycle_ledger_20260714_014209",
    },
    {
        "variant": "no_probation",
        "scene_id": "scene0011_00",
        "scheduler_root": "Stream3D/outputs/audit/v107_phase10_ablation_scene0011_no_probation90_20260714_014746",
        "fidelity_root": "Stream3D/outputs/audit/v107_phase10_ablation_scene0011_no_probation90_fidelity_diag_20260714_014923",
        "ledger_root": "Stream3D/outputs/audit/v107_phase10_ablation_scene0011_no_probation90_lifecycle_ledger_20260714_014927",
    },
    {
        "variant": "no_probation",
        "scene_id": "scene0050_00",
        "scheduler_root": "Stream3D/outputs/audit/v107_phase10_ablation_scene0050_no_probation90_20260714_014948",
        "fidelity_root": "Stream3D/outputs/audit/v107_phase10_ablation_scene0050_no_probation90_fidelity_diag_20260714_015202",
        "ledger_root": "Stream3D/outputs/audit/v107_phase10_ablation_scene0050_no_probation90_lifecycle_ledger_20260714_015205",
    },
    {
        "variant": "no_recoverability",
        "scene_id": "scene0011_00",
        "scheduler_root": "Stream3D/outputs/audit/v107_phase10_ablation_scene0011_no_recoverability90_fixaddflag_20260714_020911",
        "fidelity_root": "Stream3D/outputs/audit/v107_phase10_ablation_scene0011_no_recoverability90_fixaddflag_fidelity_diag_20260714_021050",
        "ledger_root": "Stream3D/outputs/audit/v107_phase10_ablation_scene0011_no_recoverability90_fixaddflag_lifecycle_ledger_20260714_021053",
    },
    {
        "variant": "no_recoverability",
        "scene_id": "scene0050_00",
        "scheduler_root": "Stream3D/outputs/audit/v107_phase10_ablation_scene0050_no_recoverability90_20260714_021118",
        "fidelity_root": "Stream3D/outputs/audit/v107_phase10_ablation_scene0050_no_recoverability90_fidelity_diag_20260714_021312",
        "ledger_root": "Stream3D/outputs/audit/v107_phase10_ablation_scene0050_no_recoverability90_lifecycle_ledger_20260714_021315",
    },
    {
        "variant": "random_demotion",
        "scene_id": "scene0011_00",
        "scheduler_root": "Stream3D/outputs/audit/v107_phase10_ablation_scene0011_random_demotion90_20260714_021533",
        "fidelity_root": "Stream3D/outputs/audit/v107_phase10_ablation_scene0011_random_demotion90_fidelity_diag_20260714_021938",
        "ledger_root": "Stream3D/outputs/audit/v107_phase10_ablation_scene0011_random_demotion90_lifecycle_ledger_20260714_021941",
    },
    {
        "variant": "random_demotion",
        "scene_id": "scene0050_00",
        "scheduler_root": "Stream3D/outputs/audit/v107_phase10_ablation_scene0050_random_demotion90_20260714_022210",
        "fidelity_root": "Stream3D/outputs/audit/v107_phase10_ablation_scene0050_random_demotion90_fidelity_diag_20260714_022409",
        "ledger_root": "Stream3D/outputs/audit/v107_phase10_ablation_scene0050_random_demotion90_lifecycle_ledger_20260714_022412",
    },
    {
        "variant": "area_only_demotion",
        "scene_id": "scene0011_00",
        "scheduler_root": "Stream3D/outputs/audit/v107_phase10_ablation_scene0011_area_only_demotion90_20260714_022729",
        "fidelity_root": "Stream3D/outputs/audit/v107_phase10_ablation_scene0011_area_only_demotion90_fidelity_diag_20260714_022858",
        "ledger_root": "Stream3D/outputs/audit/v107_phase10_ablation_scene0011_area_only_demotion90_lifecycle_ledger_20260714_022901",
    },
    {
        "variant": "area_only_demotion",
        "scene_id": "scene0050_00",
        "scheduler_root": "Stream3D/outputs/audit/v107_phase10_ablation_scene0050_area_only_demotion90_20260714_022959",
        "fidelity_root": "Stream3D/outputs/audit/v107_phase10_ablation_scene0050_area_only_demotion90_fidelity_diag_20260714_023239",
        "ledger_root": "Stream3D/outputs/audit/v107_phase10_ablation_scene0050_area_only_demotion90_lifecycle_ledger_20260714_023242",
    },
    {
        "variant": "no_geometry",
        "scene_id": "scene0011_00",
        "scheduler_root": "Stream3D/outputs/audit/v107_phase10_ablation_scene0011_no_geometry90_20260714_024003",
        "fidelity_root": "Stream3D/outputs/audit/v107_phase10_ablation_scene0011_no_geometry90_fidelity_diag_20260714_024137",
        "ledger_root": "Stream3D/outputs/audit/v107_phase10_ablation_scene0011_no_geometry90_lifecycle_ledger_20260714_024140",
    },
    {
        "variant": "no_geometry",
        "scene_id": "scene0050_00",
        "scheduler_root": "Stream3D/outputs/audit/v107_phase10_ablation_scene0050_no_geometry90_20260714_024217",
        "fidelity_root": "Stream3D/outputs/audit/v107_phase10_ablation_scene0050_no_geometry90_fidelity_diag_20260714_024415",
        "ledger_root": "Stream3D/outputs/audit/v107_phase10_ablation_scene0050_no_geometry90_lifecycle_ledger_20260714_024418",
    },
]


HOLDOUT_SPECS: list[dict[str, str]] = [
    {
        "variant": "geometry_guided_method_holdout_subset",
        "scene_id": "scene0030_00",
        "holdout_scope": "scene0030_00_start0_stride5_90f_only_not_full_paper_holdout",
        "reference_root": "Stream3D/outputs/audit/v107_phase11_holdout_scene0030_reference90_20260714_0305/v106_stateful_sam2_rolling_scene_stream",
        "prompt_root": "Stream3D/outputs/audit/v107_phase11_holdout_scene0030_prompt_capsule90_20260714_0307",
        "pilot_root": "Stream3D/outputs/audit/v107_phase11_holdout_scene0030_live_reactivation_pilot90_20260714_0309",
        "probe_root": "Stream3D/outputs/audit/v107_phase11_holdout_scene0030_live_state_probe_events0_5_20260714_0310",
        "scheduler_root": "Stream3D/outputs/audit/v107_phase11_holdout_scene0030_g3_scheduler90_20260714_0311",
        "fidelity_root": "Stream3D/outputs/audit/v107_phase11_holdout_scene0030_g3_scheduler90_fidelity_diag_20260714_0313",
        "ledger_root": "Stream3D/outputs/audit/v107_phase11_holdout_scene0030_g3_scheduler90_lifecycle_ledger_20260714_0314",
        "scheduler_posthoc_visual_root": "Stream3D/outputs/audit/v107_phase11_holdout_scene0030_scheduler_posthoc_visuals_20260714_0340",
    },
]


APPEARANCE_CONTROL_SPECS: list[dict[str, str]] = [
    {
        "variant": "appearance_only_reactivation_control_holdout_subset",
        "scene_id": "scene0030_00",
        "scheduler_root": "Stream3D/outputs/audit/v107_phase11_holdout_scene0030_appearance_only_control90_20260714_0410",
        "fidelity_root": "Stream3D/outputs/audit/v107_phase11_holdout_scene0030_appearance_only_control90_fidelity_diag_20260714_0412",
        "ledger_root": "Stream3D/outputs/audit/v107_phase11_holdout_scene0030_appearance_only_control90_lifecycle_ledger_20260714_0413",
    },
]


TARGETED_VISUAL_SPOTCHECKS: list[dict[str, str]] = [
    {
        "scene_id": "scene0030_00",
        "event": "event000_obj3",
        "path": "Stream3D/outputs/audit/v107_phase11_holdout_scene0030_live_state_probe_events0_5_20260714_0310/event000/highres_live_state_visuals/event000_G2_pos_neg_confirm_zoom_x3.jpg",
        "observation": (
            "Codex spot-check: visible green positive prompt points lie on the same sofa cushion/arm target; "
            "visible red negative prompt point is off the target. No obvious projection-to-wall/occluder error "
            "was observed in this single high-resolution case."
        ),
    },
    {
        "scene_id": "scene0030_00",
        "event": "event001_obj4",
        "path": "Stream3D/outputs/audit/v107_phase11_holdout_scene0030_live_state_probe_events0_5_20260714_0310/event001/highres_live_state_visuals/event001_G2_pos_neg_confirm_zoom_x3.jpg",
        "observation": (
            "Codex spot-check: visible green positive prompt points lie on the same sofa cushion/arm target; "
            "visible red negative prompt points are off the target on nearby wall/sofa/background regions. "
            "This is plausible prompt placement evidence, not a user visual PASS."
        ),
    },
    {
        "scene_id": "scene0030_00",
        "event": "event003_scheduler_posthoc_ref7",
        "path": "Stream3D/outputs/audit/v107_phase11_holdout_scene0030_scheduler_posthoc_visuals_20260714_0340/highres_scheduler_posthoc_visuals/event003_confirm_G1_pos_f10_ref7_predlabel7.jpg",
        "observation": (
            "Codex spot-check of frozen scheduler labels: the post-hoc confirm mask covers the same blackboard "
            "region as the yellow reference contour; green positive points are on the blackboard and red negative "
            "points are off target on sofa/wall regions. This is plausible visual evidence, not a final user PASS."
        ),
    },
    {
        "scene_id": "scene0030_00",
        "event": "event004_scheduler_posthoc_ref8",
        "path": "Stream3D/outputs/audit/v107_phase11_holdout_scene0030_scheduler_posthoc_visuals_20260714_0340/highres_scheduler_posthoc_visuals/event004_confirm_G1_pos_f10_ref8_predlabel8.jpg",
        "observation": (
            "Codex spot-check of frozen scheduler labels: the post-hoc confirm mask and yellow reference contour "
            "align over the blackboard; visible positives are on target and visible negatives are off target. "
            "This is readable scheduler-result evidence only."
        ),
    },
    {
        "scene_id": "scene0030_00",
        "event": "event004_appearance_only_control_success_ref8",
        "path": "Stream3D/outputs/audit/v107_phase11_holdout_scene0030_appearance_only_control90_20260714_0410/highres_event_visuals/event004_appearance_only_confirm_f10_ref8_live7.jpg",
        "observation": (
            "Codex spot-check of the new appearance-only control: event004/ref8 visually recovers the blackboard "
            "mask without LingBot prompt points or SAM2 re-add. This is one readable success case, not final PASS."
        ),
    },
    {
        "scene_id": "scene0030_00",
        "event": "event003_appearance_only_control_failure_ref7",
        "path": "Stream3D/outputs/audit/v107_phase11_holdout_scene0030_appearance_only_control90_20260714_0410/highres_event_visuals/event003_appearance_only_confirm_f10_ref7_live6.jpg",
        "observation": (
            "Codex spot-check of the new appearance-only control: event003/ref7 did not output a mask because "
            "the appearance margin was below threshold, leaving only the reference contour visible. This records "
            "a real weak/failure case."
        ),
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_out = AUDIT_ROOT / f"v107_phase11_frozen_casebook_continue_gates_ignored_{stamp}"
    parser.add_argument("--out-root", type=Path, default=default_out)
    parser.add_argument("--skip-checkpoint-hash", action="store_true")
    return parser.parse_args()


def rel(path: Path | str) -> str:
    q = Path(path)
    try:
        return q.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return q.as_posix()


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"missing": True, "path": rel(path)}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - report malformed artifacts explicitly.
        return {"read_error": str(exc), "path": rel(path)}
    if isinstance(data, dict):
        return data
    return {"value": data}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git_value(args: list[str]) -> str:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError:
        return ""
    return proc.stdout.strip()


def num_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def count_dict(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, int] = {}
    for key, val in value.items():
        try:
            out[str(key)] = int(val)
        except (TypeError, ValueError):
            continue
    return out


def run_row(spec: dict[str, str]) -> dict[str, Any]:
    scheduler_root = REPO_ROOT / spec["scheduler_root"]
    fidelity_root = REPO_ROOT / spec["fidelity_root"]
    ledger_root = REPO_ROOT / spec["ledger_root"]
    summary = read_json(scheduler_root / "g3_scheduler_summary.json")
    fidelity = read_json(fidelity_root / "foreground_fidelity_summary.json")
    ledger = read_json(ledger_root / "lifecycle_ledger_summary.json")
    return {
        "variant": spec["variant"],
        "scene_id": spec["scene_id"],
        "scheduler_root": rel(scheduler_root),
        "fidelity_root": rel(fidelity_root),
        "ledger_root": rel(ledger_root),
        "scheduler_summary_sha256": sha256_file(scheduler_root / "g3_scheduler_summary.json"),
        "fidelity_summary_sha256": sha256_file(fidelity_root / "foreground_fidelity_summary.json"),
        "ledger_summary_sha256": sha256_file(ledger_root / "lifecycle_ledger_summary.json"),
        "artifact_complete": bool(
            (scheduler_root / "g3_scheduler_summary.json").exists()
            and (fidelity_root / "foreground_fidelity_summary.json").exists()
            and (ledger_root / "lifecycle_ledger_summary.json").exists()
        ),
        "status": summary.get("status"),
        "frame_count": summary.get("frame_count") or fidelity.get("frame_count"),
        "event_count": summary.get("event_count"),
        "runtime_sec": num_or_none(summary.get("runtime_sec")),
        "rolling_wall_sec": num_or_none(summary.get("rolling_wall_sec")),
        "auto_selection_policy": summary.get("auto_selection_policy"),
        "birth_transaction_enabled": summary.get("birth_transaction_enabled"),
        "recoverability_enabled": summary.get("recoverability_enabled"),
        "reactivation_prompt_mode": summary.get("reactivation_prompt_mode") or "lingbot_geometry_default",
        "geometry_prompts_enabled": summary.get("geometry_prompts_enabled"),
        "sam2_add_new_points_or_box_called": summary.get("sam2_add_new_points_or_box_called"),
        "appearance_only_reactivation_implemented": summary.get("appearance_only_reactivation_implemented"),
        "appearance_only_uses_lingbot_prompt_points": summary.get("appearance_only_uses_lingbot_prompt_points"),
        "appearance_only_descriptor": summary.get("appearance_only_descriptor"),
        "appearance_only_record_count": summary.get("appearance_only_record_count"),
        "appearance_only_output_record_count": summary.get("appearance_only_output_record_count"),
        "appearance_only_mean_score": num_or_none(summary.get("appearance_only_mean_score")),
        "appearance_only_mean_margin": num_or_none(summary.get("appearance_only_mean_margin")),
        "actual_video_readd_record_count": summary.get("actual_video_readd_record_count"),
        "diagnostic_confirm_mean_iou_to_reference": num_or_none(summary.get("diagnostic_confirm_mean_iou_to_reference")),
        "confirm_iou_ge_0_5_rate": num_or_none(summary.get("confirm_iou_ge_0_5_rate")),
        "long_term_memory_admitted_count": summary.get("long_term_memory_admitted_count"),
        "long_term_admission_skip_reasons": summary.get("long_term_admission_skip_reasons"),
        "prompt_new_object_assignment_count": summary.get("prompt_new_object_assignment_count"),
        "probation_output_mask_count": summary.get("probation_output_mask_count"),
        "shadow_output_mask_count": summary.get("shadow_output_mask_count"),
        "source_mapping_accepted_count": summary.get("source_mapping_accepted_count"),
        "source_mapping_skip_reasons": summary.get("source_mapping_skip_reasons"),
        "diagnostic_foreground_recall_mean": num_or_none(fidelity.get("foreground_recall_mean")),
        "diagnostic_foreground_recall_min": num_or_none(fidelity.get("foreground_recall_min")),
        "diagnostic_foreground_precision_mean": num_or_none(fidelity.get("foreground_precision_mean")),
        "diagnostic_foreground_iou_mean": num_or_none(fidelity.get("foreground_iou_mean")),
        "lifecycle_row_count": ledger.get("lifecycle_row_count"),
        "sam_slot_avoided_event_frame_count_estimate": ledger.get("sam_slot_avoided_event_frame_count_estimate"),
        "state_counts": count_dict(ledger.get("state_counts")),
        "visual_review_status": ledger.get("visual_review_status")
        or summary.get("visual_review_status")
        or "USER_VISUAL_REVIEW_PENDING",
        "reference_metrics_are_diagnostic_only": bool(
            summary.get("reference_metrics_are_diagnostic_only")
            or ledger.get("reference_metrics_are_diagnostic_only")
        ),
        "small_objects_may_skip_long_term_memory": bool(
            ledger.get("small_objects_may_skip_long_term_memory")
            or ledger.get("scheduler_small_objects_may_skip_long_term_memory")
        ),
        "highres_visuals": summary.get("highres_visuals") or [],
        "highres_probation_visuals": summary.get("highres_probation_visuals") or [],
        "highres_shadow_visuals": summary.get("highres_shadow_visuals") or [],
    }


def normalize_visual_items(items: Any, source_summary: Path | None = None, source_kind: str | None = None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not isinstance(items, list):
        return out
    for item in items:
        if isinstance(item, dict) and item.get("path"):
            copied = dict(item)
            if source_summary is not None:
                copied.setdefault("source_summary", rel(source_summary))
            if source_kind is not None:
                copied.setdefault("visual_source", source_kind)
            out.append(copied)
        elif isinstance(item, dict) and item.get("visual_path"):
            copied = dict(item)
            copied["path"] = str(item["visual_path"])
            if source_summary is not None:
                copied.setdefault("source_summary", rel(source_summary))
            if source_kind is not None:
                copied.setdefault("visual_source", source_kind)
            out.append(copied)
        elif isinstance(item, str):
            copied = {"path": item}
            if source_summary is not None:
                copied["source_summary"] = rel(source_summary)
            if source_kind is not None:
                copied["visual_source"] = source_kind
            out.append(copied)
    return out


def holdout_row(spec: dict[str, str]) -> dict[str, Any]:
    reference_root = REPO_ROOT / spec["reference_root"]
    prompt_root = REPO_ROOT / spec["prompt_root"]
    pilot_root = REPO_ROOT / spec["pilot_root"]
    probe_root = REPO_ROOT / spec["probe_root"]
    scheduler_root = REPO_ROOT / spec["scheduler_root"]
    fidelity_root = REPO_ROOT / spec["fidelity_root"]
    ledger_root = REPO_ROOT / spec["ledger_root"]
    scheduler_posthoc_visual_root = REPO_ROOT / spec["scheduler_posthoc_visual_root"]

    reference_summary_path = reference_root / "summary.json"
    prompt_summary_path = prompt_root / "prompt_capsule_visibility_probe_summary.json"
    pilot_summary_path = pilot_root / "lifecycle_metric_summary.json"
    probe_summary_path = probe_root / "live_state_reactivation_summary.json"
    scheduler_summary_path = scheduler_root / "g3_scheduler_summary.json"
    rolling_summary_path = scheduler_root / "v107_phase8_g3_rolling_scheduler_smoke/summary.json"
    fidelity_summary_path = fidelity_root / "foreground_fidelity_summary.json"
    fidelity_rows_path = fidelity_root / "foreground_fidelity_rows.csv"
    ledger_summary_path = ledger_root / "lifecycle_ledger_summary.json"
    scheduler_posthoc_visual_manifest_path = scheduler_posthoc_visual_root / "scheduler_posthoc_visual_manifest.json"

    reference = read_json(reference_summary_path)
    prompt = read_json(prompt_summary_path)
    pilot = read_json(pilot_summary_path)
    probe = read_json(probe_summary_path)
    summary = read_json(scheduler_summary_path)
    rolling = read_json(rolling_summary_path)
    fidelity = read_json(fidelity_summary_path)
    ledger = read_json(ledger_summary_path)
    scheduler_posthoc_visual_manifest = read_json(scheduler_posthoc_visual_manifest_path)
    raw_lingbot = prompt.get("raw_lingbot_geometry")
    if not isinstance(raw_lingbot, dict):
        raw_lingbot = {}
    model_info = raw_lingbot.get("model_info")
    if not isinstance(model_info, dict):
        model_info = {}

    artifact_paths = [
        reference_summary_path,
        prompt_summary_path,
        pilot_summary_path,
        probe_summary_path,
        scheduler_summary_path,
        rolling_summary_path,
        fidelity_summary_path,
        fidelity_rows_path,
        ledger_summary_path,
        scheduler_posthoc_visual_manifest_path,
    ]
    scheduler_visuals = normalize_visual_items(
        summary.get("highres_visuals"),
        scheduler_summary_path,
        "g3_scheduler",
    )
    probe_visuals = normalize_visual_items(
        probe.get("highres_visuals"),
        probe_summary_path,
        "live_state_probe",
    )
    scheduler_posthoc_visuals = normalize_visual_items(
        scheduler_posthoc_visual_manifest.get("rows"),
        scheduler_posthoc_visual_manifest_path,
        "g3_scheduler_posthoc_frozen_labels",
    )
    return {
        "variant": spec["variant"],
        "scene_id": spec["scene_id"],
        "holdout_scope": spec["holdout_scope"],
        "reference_root": rel(reference_root),
        "prompt_root": rel(prompt_root),
        "pilot_root": rel(pilot_root),
        "probe_root": rel(probe_root),
        "scheduler_root": rel(scheduler_root),
        "fidelity_root": rel(fidelity_root),
        "ledger_root": rel(ledger_root),
        "scheduler_posthoc_visual_root": rel(scheduler_posthoc_visual_root),
        "reference_summary_sha256": sha256_file(reference_summary_path),
        "prompt_summary_sha256": sha256_file(prompt_summary_path),
        "pilot_summary_sha256": sha256_file(pilot_summary_path),
        "probe_summary_sha256": sha256_file(probe_summary_path),
        "scheduler_summary_sha256": sha256_file(scheduler_summary_path),
        "rolling_summary_sha256": sha256_file(rolling_summary_path),
        "fidelity_summary_sha256": sha256_file(fidelity_summary_path),
        "fidelity_rows_sha256": sha256_file(fidelity_rows_path),
        "ledger_summary_sha256": sha256_file(ledger_summary_path),
        "scheduler_posthoc_visual_manifest_sha256": sha256_file(scheduler_posthoc_visual_manifest_path),
        "artifact_complete": all(path.exists() for path in artifact_paths),
        "status": summary.get("status"),
        "frame_count": summary.get("frame_count") or rolling.get("frame_count") or reference.get("frame_count"),
        "frame_start": prompt.get("frame_start"),
        "frame_stride": prompt.get("frame_stride"),
        "frame_ids_first": (raw_lingbot.get("frame_ids") or [None])[0],
        "frame_ids_last": (raw_lingbot.get("frame_ids") or [None])[-1],
        "event_count": summary.get("event_count"),
        "runtime_sec": num_or_none(summary.get("runtime_sec")),
        "rolling_wall_sec": num_or_none(summary.get("rolling_wall_sec")),
        "reference_wall_time_sec": num_or_none(reference.get("wall_time_sec")),
        "prompt_runtime_sec": num_or_none(prompt.get("runtime_sec")),
        "pilot_runtime_sec": num_or_none(pilot.get("runtime_sec")),
        "probe_runtime_sec": num_or_none(probe.get("runtime_sec")),
        "lingbot_forward_runtime_sec": num_or_none(raw_lingbot.get("forward_runtime_sec")),
        "lingbot_peak_memory_bytes": raw_lingbot.get("peak_memory_bytes"),
        "lingbot_npz_sha256": raw_lingbot.get("npz_sha256"),
        "lingbot_checkpoint_sha256": model_info.get("checkpoint_sha256"),
        "selected_pose_mode": prompt.get("selected_pose_mode"),
        "projection_geometry_source": summary.get("projection_geometry_source")
        or prompt.get("projection_geometry_source"),
        "uses_scannet_pose_or_depth_for_projection": bool(
            summary.get("uses_scannet_pose_or_depth_for_projection")
            or prompt.get("uses_scannet_pose_or_depth_for_projection")
        ),
        "sam2_add_new_points_or_box_called": summary.get("sam2_add_new_points_or_box_called"),
        "long_term_memory_admitted_count": summary.get("long_term_memory_admitted_count"),
        "long_term_admission_skip_reasons": summary.get("long_term_admission_skip_reasons"),
        "prompt_new_object_assignment_count": summary.get("prompt_new_object_assignment_count"),
        "probation_output_mask_count": summary.get("probation_output_mask_count"),
        "shadow_output_mask_count": summary.get("shadow_output_mask_count"),
        "selected_g2_count": summary.get("selected_g2_count"),
        "confirm_mean_iou": num_or_none(summary.get("confirm_mean_iou")),
        "diagnostic_foreground_recall_mean": num_or_none(fidelity.get("foreground_recall_mean")),
        "diagnostic_foreground_recall_min": num_or_none(fidelity.get("foreground_recall_min")),
        "diagnostic_foreground_precision_mean": num_or_none(fidelity.get("foreground_precision_mean")),
        "diagnostic_foreground_iou_mean": num_or_none(fidelity.get("foreground_iou_mean")),
        "lifecycle_row_count": ledger.get("lifecycle_row_count") or ledger.get("lifecycle_rows"),
        "sam_slot_avoided_event_frame_count_estimate": ledger.get("sam_slot_avoided_event_frame_count_estimate"),
        "state_counts": count_dict(ledger.get("state_counts")),
        "scheduler_highres_visual_count": summary.get("highres_visual_count"),
        "probe_highres_visual_count": probe.get("highres_visual_count"),
        "scheduler_posthoc_visual_count": scheduler_posthoc_visual_manifest.get("visual_count"),
        "scheduler_posthoc_no_model_rerun": scheduler_posthoc_visual_manifest.get("posthoc_no_model_rerun"),
        "highres_visuals": scheduler_posthoc_visuals + scheduler_visuals + probe_visuals,
        "highres_probation_visuals": normalize_visual_items(
            summary.get("highres_probation_visuals"),
            scheduler_summary_path,
            "g3_scheduler_probation",
        ),
        "highres_shadow_visuals": normalize_visual_items(
            summary.get("highres_shadow_visuals"),
            scheduler_summary_path,
            "g3_scheduler_shadow",
        ),
        "visual_review_status": "USER_VISUAL_REVIEW_PENDING",
        "metric_role": "diagnostic_only_not_acceptance_gate",
        "small_objects_may_skip_long_term_memory": True,
    }


def copy_review_artifact(src: Path, dst_dir: Path, prefix: str) -> dict[str, Any]:
    src_abs = src if src.is_absolute() else REPO_ROOT / src
    suffix = "".join(src_abs.suffixes) or ".bin"
    dst = dst_dir / f"{prefix}_{src_abs.stem}{suffix}"
    row: dict[str, Any] = {
        "source_path": rel(src_abs),
        "copied_path": rel(dst),
        "source_exists": src_abs.exists(),
        "sha256": sha256_file(src_abs),
    }
    if src_abs.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_abs, dst)
        row["copied_sha256"] = sha256_file(dst)
    return row


def build_review_manifest(rows: list[dict[str, Any]], out_root: Path) -> dict[str, Any]:
    review_entries: list[dict[str, Any]] = []
    visual_dst = out_root / "videos/highres_visual_cases"
    video_dst = out_root / "videos/rgb_overlay_mp4"
    for row in rows:
        prefix_base = f"{row['scene_id']}_{row['variant']}"
        for kind, field in [
            ("confirm", "highres_visuals"),
            ("probation", "highres_probation_visuals"),
            ("shadow", "highres_shadow_visuals"),
        ]:
            for idx, item in enumerate(row.get(field) or []):
                if isinstance(item, str):
                    item = {"path": item}
                if not isinstance(item, dict) or not item.get("path"):
                    continue
                copied = copy_review_artifact(Path(str(item["path"])), visual_dst, f"{prefix_base}_{kind}{idx:02d}")
                copied.update(
                    {
                        "schema_version": "stream4d_v107_phase11_user_review_entry_v1",
                        "scene_id": row["scene_id"],
                        "variant": row["variant"],
                        "visual_kind": kind,
                        "review_question": (
                            "Inspect at native resolution: are positive points on the target, "
                            "negative points off target/visible, and recovered mask visually correct?"
                        ),
                        "manual_review_status": "USER_REVIEW_PENDING",
                        "metric_role": "diagnostic_only",
                    }
                )
                review_entries.append(copied)
        scheduler_root = REPO_ROOT / row["scheduler_root"]
        for mp4 in sorted(scheduler_root.glob("v107_phase8_g3_rolling_scheduler_smoke/v106_visual_review/*.mp4")):
            copied = copy_review_artifact(mp4, video_dst, prefix_base)
            copied.update(
                {
                    "schema_version": "stream4d_v107_phase11_video_review_entry_v1",
                    "scene_id": row["scene_id"],
                    "variant": row["variant"],
                    "visual_kind": "rgb_overlay_video",
                    "manual_review_status": "USER_REVIEW_PENDING",
                    "metric_role": "diagnostic_only",
                }
            )
            review_entries.append(copied)
    return {
        "schema_version": "stream4d_v107_phase11_user_review_manifest_v1",
        "visual_protocol": {
            "final_acceptance_source": "targeted_high_resolution_visual_confirmation",
            "large_contact_sheets_allowed_for_final_pass": False,
            "metrics_are_acceptance_gates": False,
            "required_manual_actor": "user",
        },
        "entry_count": len(review_entries),
        "entries": review_entries,
    }


def build_targeted_visual_spotchecks(out_root: Path) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    dst_dir = out_root / "videos/targeted_visual_spotchecks"
    for idx, spec in enumerate(TARGETED_VISUAL_SPOTCHECKS):
        copied = copy_review_artifact(Path(spec["path"]), dst_dir, f"{spec['scene_id']}_{spec['event']}_{idx:02d}")
        copied.update(
            {
                "schema_version": "stream4d_v107_phase11_targeted_visual_spotcheck_entry_v1",
                "scene_id": spec["scene_id"],
                "event": spec["event"],
                "observer": "codex_assisted_visual_spotcheck",
                "observation": spec["observation"],
                "acceptance_boundary": "not_user_final_acceptance",
                "manual_review_status": "USER_REVIEW_PENDING",
                "metric_role": "visual_evidence_only_not_metric_gate",
            }
        )
        entries.append(copied)
    return {
        "schema_version": "stream4d_v107_phase11_targeted_visual_spotchecks_v1",
        "entry_count": len(entries),
        "acceptance_boundary": (
            "These observations document two readable high-resolution cases only. "
            "They do not establish final visual PASS."
        ),
        "entries": entries,
    }


def write_config(out_root: Path, code_hash: str | None, checkpoint_hash: str | None) -> tuple[Path, str | None]:
    lines = [
        "version: v107",
        "phase: phase11_continued_after_user_ignored_gate",
        "holdout_claim: false",
        "true_holdout_subset_scene_count: 1",
        "true_holdout_subset_scope: scene0030_00_start0_stride5_90f_only_not_full_paper_holdout",
        "cache_read_count: not_applicable_existing_artifacts_only",
        "same_scene_one_sequential_stream: true",
        "sam2_model: SAM2.1-L",
        f"sam2_checkpoint: {rel(SAM2_CHECKPOINT)}",
        f"sam2_checkpoint_sha256: {checkpoint_hash or 'missing_or_skipped'}",
        "lingbot_geometry_source: LingBot-Map decoded pose_enc + depth + depth_conf + intrinsics",
        "uses_scannet_pose_or_depth_for_projection: false",
        "metric_contract:",
        "  recall_iou_tor_mv_ap_fidelity: diagnostic_only",
        "  final_acceptance: targeted_high_resolution_visual_confirmation",
        "small_object_policy: small_objects_may_skip_long_term_memory",
        "reactivation_prompt:",
        "  positive: visible projected historical mask points",
        "  negative: visible projected co-visible sibling object points",
        "  occlusion_required: true",
        "appearance_only_live_scheduler_implemented: partial_scene0030_control_smoke",
        "scheduler_highres_visual_note: scene0030 scheduler summary had zero runtime highres visuals because requested visual event ids were long-term-policy skipped; post-hoc frozen-label scheduler visuals are packaged without model rerun",
        f"scheduler_script: {rel(SCHEDULER_SCRIPT)}",
        f"scheduler_script_sha256: {code_hash or 'missing'}",
    ]
    path = out_root / "config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path, sha256_file(path)


def missing_control_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scene in ["scene0011_00", "scene0050_00"]:
        rows.append(
            {
                "variant": "appearance_only_reactivation",
                "scene_id": scene,
                "artifact_complete": False,
                "status": "NOT_RUN_FOR_PHASE10_DEV_SCENES_PARTIAL_SCENE0030_CONTROL_EXISTS",
                "reason": (
                    "v107 live scheduler appearance-only reactivation now has a partial scene0030_00 "
                    "90-frame control smoke, but it has not been run for the Phase10 dev scenes or a full "
                    "ablation matrix. Keep this as missing for scene0011_00/scene0050_00 instead of "
                    "counting geometry-guided results as appearance-only evidence."
                ),
            }
        )
        rows.append(
            {
                "variant": "geometry_only_reactivation",
                "scene_id": scene,
                "artifact_complete": True,
                "status": "ALIAS_OF_GEOMETRY_GUIDED_METHOD",
                "reason": (
                    "Current method run is geometry-guided with LingBot projected prompts and no active "
                    "appearance re-ID branch; keep this alias explicit instead of double-counting a new run."
                ),
            }
        )
    return rows


def build_failure_casebook(
    rows: list[dict[str, Any]],
    holdout_rows: list[dict[str, Any]],
    appearance_control_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    method_roots = [row["scheduler_root"] for row in rows if row["variant"] == "geometry_guided_method"]
    no_geometry_roots = [row["scheduler_root"] for row in rows if row["variant"] == "no_geometry"]
    holdout_roots = [row["scheduler_root"] for row in holdout_rows]
    appearance_roots = [row["scheduler_root"] for row in appearance_control_rows]
    cases = [
            {
                "failure_type": "VISUAL_REVIEW_PENDING",
                "status": "active",
                "observed_artifact": method_roots,
                "root_cause_hypothesis": (
                    "Only targeted high-resolution visuals can decide final correctness; metrics are "
                    "diagnostic and large contact sheets are insufficient."
                ),
                "attempted_repairs": [
                    "Generated high-resolution event/probation/shadow visuals for selected cases.",
                    "Copied available review images and RGB overlay videos into Phase11 user review manifest.",
                ],
                "metric_movement": "not_used_as_gate",
                "allowed_next_family": "targeted_high_resolution_visual_review_or_video_rendering",
                "forbidden_next_action": "claim_final_visual_PASS_from_diagnostic_metrics_or_low_res_contact_sheet",
            },
            {
                "failure_type": "APPEARANCE_ONLY_LIVE_SCHEDULER_PARTIAL_WEAK_CONTROL",
                "nearest_plan_taxonomy": "REACTIVATION_FALSE_MERGE",
                "status": "active_limitation",
                "observed_artifact": appearance_roots,
                "root_cause_hypothesis": (
                    "Appearance-only reactivation is now wired into the v107 live scheduler, but the current "
                    "descriptor is a lightweight RGB mean/std plus shape match against current-frame base "
                    "masks. It can recover clear large targets, yet weak/missing masks still appear in "
                    "targeted visuals, so it is only partial control evidence."
                ),
                "attempted_repairs": [
                    "Ported appearance_only mode into the event-time scheduler without LingBot prompt points.",
                    "Kept SAM2 add_new_points/add_new_points_or_box disabled for the appearance-only control.",
                    "Ran a scene0030_00 90-frame control smoke and packaged one success and one failure visual spot-check.",
                ],
                "metric_movement": {
                    row["scene_id"]: {
                        "diagnostic_appearance_only_record_count": row.get("appearance_only_record_count"),
                        "diagnostic_appearance_only_output_record_count": row.get(
                            "appearance_only_output_record_count"
                        ),
                        "diagnostic_appearance_only_mean_score": row.get("appearance_only_mean_score"),
                        "diagnostic_appearance_only_mean_margin": row.get("appearance_only_mean_margin"),
                        "diagnostic_confirm_mean_iou_to_reference": row.get(
                            "diagnostic_confirm_mean_iou_to_reference"
                        ),
                        "diagnostic_confirm_iou_ge_0_5_rate": row.get("confirm_iou_ge_0_5_rate"),
                    }
                    for row in appearance_control_rows
                },
                "allowed_next_family": "targeted_visual_review_or_stronger_appearance_descriptor_control",
                "forbidden_next_action": "report_geometry_prompt_results_as_appearance_only_control",
            },
            {
                "failure_type": "REFERENCE_FIDELITY_DROP",
                "status": "diagnostic_only",
                "observed_artifact": [
                    row["fidelity_root"]
                    for row in rows
                    if row["variant"] == "geometry_guided_method" and row.get("diagnostic_foreground_recall_mean") is not None
                ],
                "root_cause_hypothesis": (
                    "Longer or lifecycle-virtualized streams can drift from the reference foreground; this is "
                    "expected and must be interpreted visually, not as exact parity failure."
                ),
                "attempted_repairs": [
                    "Separated reference metrics from online scheduler gates.",
                    "Recorded metrics as diagnostic-only in summaries and logs.",
                ],
                "metric_movement": {
                    row["scene_id"]: {
                        "diagnostic_foreground_recall_mean": row.get("diagnostic_foreground_recall_mean"),
                        "diagnostic_foreground_iou_mean": row.get("diagnostic_foreground_iou_mean"),
                    }
                    for row in rows
                    if row["variant"] == "geometry_guided_method"
                },
                "allowed_next_family": "targeted_visual_triage_on_worst_or_representative_cases",
                "forbidden_next_action": "block_implementation_on_exact_reference_parity",
            },
            {
                "failure_type": "GEOMETRY_PROMPT_GRANULARITY_LOSS",
                "status": "risk_pending_visual_review",
                "observed_artifact": no_geometry_roots + method_roots,
                "root_cause_hypothesis": (
                    "Projected point prompts may be too sparse or may land on ambiguous surfaces; only readable "
                    "per-case visuals can validate positive/negative prompt placement."
                ),
                "attempted_repairs": [
                    "Used LingBot-Map pose/depth/depth_conf/intrinsics for projection.",
                    "Required visible projected positive and sibling-negative points.",
                    "Generated high-resolution prompt visual cases instead of broad low-resolution sheets.",
                ],
                "metric_movement": "prompt placement is visual evidence, not an aggregate metric gate",
                "allowed_next_family": "sample_more_high_resolution_prompt_cases",
                "forbidden_next_action": "declare_prompt_geometry_correct_without_readable_visual_evidence",
            },
            {
                "failure_type": "SINGLE_SCENE_HOLDOUT_SUBSET_ONLY",
                "status": "active_limitation",
                "observed_artifact": holdout_roots,
                "root_cause_hypothesis": (
                    "scene0030_00 is a useful new-scene holdout subset, but one 90-frame scene is not a "
                    "complete paper-level holdout and cannot establish final method success."
                ),
                "attempted_repairs": [
                    "Ran v106 reference, LingBot prompt capsule, SAM2 image-predictor pilot, live-state mutation probe, G3 scheduler, fidelity diagnostic, and lifecycle ledger on scene0030_00.",
                    "Diagnosed that scheduler runtime high-resolution visuals were zero because the requested visual event ids were long-term-policy skipped.",
                    "Built post-hoc frozen-label scheduler visuals without rerunning SAM2 and packaged them with live-state probe high-resolution visuals.",
                ],
                "metric_movement": {
                    row["scene_id"]: {
                        "diagnostic_foreground_recall_mean": row.get("diagnostic_foreground_recall_mean"),
                        "diagnostic_foreground_iou_mean": row.get("diagnostic_foreground_iou_mean"),
                        "long_term_memory_admitted_count": row.get("long_term_memory_admitted_count"),
                    }
                    for row in holdout_rows
                },
                "allowed_next_family": "targeted_user_visual_review_or_additional_frozen_holdout_scenes",
                "forbidden_next_action": "promote_single_scene_subset_or_diagnostic_recall_to_final_pass",
            },
        ]
    return {
        "schema_version": "stream4d_v107_phase11_failure_casebook_v1",
        "case_count": len(cases),
        "cases": cases,
    }


def main() -> None:
    args = parse_args()
    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    code_hash = sha256_file(SCHEDULER_SCRIPT)
    checkpoint_hash = None if args.skip_checkpoint_hash else sha256_file(SAM2_CHECKPOINT)
    config_path, config_hash = write_config(out_root, code_hash, checkpoint_hash)
    rows = [run_row(spec) for spec in RUN_SPECS]
    holdout_rows = [holdout_row(spec) for spec in HOLDOUT_SPECS]
    appearance_control_rows = [run_row(spec) for spec in APPEARANCE_CONTROL_SPECS]
    missing_rows = missing_control_rows()
    review_manifest = build_review_manifest(rows + holdout_rows + appearance_control_rows, out_root)
    targeted_visual_spotchecks = build_targeted_visual_spotchecks(out_root)
    failure_casebook = build_failure_casebook(rows, holdout_rows, appearance_control_rows)

    metric_tables = {
        "schema_version": "stream4d_v107_phase11_paper_metric_tables_v1",
        "source_scope": "phase10_dev_and_ablation_artifacts_reused_after_user_requested_gate_ignore",
        "true_holdout_subset_scope": "scene0030_00_start0_stride5_90f_only_not_full_paper_holdout",
        "holdout_claim": False,
        "holdout_run_once_executed": True,
        "metric_contract": {
            "reference_fidelity": "diagnostic_only",
            "gt_diagnostics": "diagnostic_only",
            "mv_ap": "diagnostic_only_not_computed_in_this_builder",
            "visual_acceptance": "user_targeted_high_resolution_review_required",
        },
        "holdout_subset_rows": [
            {
                key: row.get(key)
                for key in [
                    "variant",
                    "scene_id",
                    "holdout_scope",
                    "artifact_complete",
                    "status",
                    "frame_count",
                    "frame_start",
                    "frame_stride",
                    "frame_ids_first",
                    "frame_ids_last",
                    "event_count",
                    "runtime_sec",
                    "rolling_wall_sec",
                    "diagnostic_foreground_recall_mean",
                    "diagnostic_foreground_recall_min",
                    "diagnostic_foreground_precision_mean",
                    "diagnostic_foreground_iou_mean",
                    "lifecycle_row_count",
                    "sam_slot_avoided_event_frame_count_estimate",
                    "visual_review_status",
                    "metric_role",
                ]
            }
            for row in holdout_rows
        ],
        "appearance_control_rows": [
            {
                key: row.get(key)
                for key in [
                    "variant",
                    "scene_id",
                    "artifact_complete",
                    "status",
                    "frame_count",
                    "event_count",
                    "runtime_sec",
                    "rolling_wall_sec",
                    "diagnostic_foreground_recall_mean",
                    "diagnostic_foreground_recall_min",
                    "diagnostic_foreground_precision_mean",
                    "diagnostic_foreground_iou_mean",
                    "lifecycle_row_count",
                    "sam_slot_avoided_event_frame_count_estimate",
                    "visual_review_status",
                ]
            }
            for row in appearance_control_rows
        ],
        "rows": [
            {
                key: row.get(key)
                for key in [
                    "variant",
                    "scene_id",
                    "artifact_complete",
                    "status",
                    "frame_count",
                    "event_count",
                    "runtime_sec",
                    "rolling_wall_sec",
                    "diagnostic_foreground_recall_mean",
                    "diagnostic_foreground_recall_min",
                    "diagnostic_foreground_precision_mean",
                    "diagnostic_foreground_iou_mean",
                    "lifecycle_row_count",
                    "sam_slot_avoided_event_frame_count_estimate",
                    "visual_review_status",
                ]
            }
            for row in rows
        ],
    }
    ablation_tables = {
        "schema_version": "stream4d_v107_phase11_paper_ablation_tables_v1",
        "source_scope": "phase10_dev_and_ablation_artifacts_reused_after_user_requested_gate_ignore",
        "true_holdout_subset_scope": "scene0030_00_start0_stride5_90f_only_not_full_paper_holdout",
        "holdout_claim": False,
        "holdout_run_once_executed": True,
        "holdout_subset_rows": [
            {
                key: row.get(key)
                for key in [
                    "variant",
                    "scene_id",
                    "holdout_scope",
                    "artifact_complete",
                    "status",
                    "projection_geometry_source",
                    "uses_scannet_pose_or_depth_for_projection",
                    "selected_pose_mode",
                    "sam2_add_new_points_or_box_called",
                    "long_term_memory_admitted_count",
                    "long_term_admission_skip_reasons",
                    "prompt_new_object_assignment_count",
                    "probation_output_mask_count",
                    "shadow_output_mask_count",
                    "selected_g2_count",
                    "state_counts",
                    "scheduler_highres_visual_count",
                    "scheduler_posthoc_visual_count",
                    "scheduler_posthoc_no_model_rerun",
                    "probe_highres_visual_count",
                    "small_objects_may_skip_long_term_memory",
                ]
            }
            for row in holdout_rows
        ],
        "appearance_control_rows": [
            {
                key: row.get(key)
                for key in [
                    "variant",
                    "scene_id",
                    "artifact_complete",
                    "status",
                    "reactivation_prompt_mode",
                    "geometry_prompts_enabled",
                    "sam2_add_new_points_or_box_called",
                    "long_term_memory_admitted_count",
                    "long_term_admission_skip_reasons",
                    "prompt_new_object_assignment_count",
                    "probation_output_mask_count",
                    "shadow_output_mask_count",
                    "selected_g2_count",
                    "state_counts",
                ]
            }
            for row in appearance_control_rows
        ],
        "rows": [
            {
                key: row.get(key)
                for key in [
                    "variant",
                    "scene_id",
                    "artifact_complete",
                    "status",
                    "auto_selection_policy",
                    "birth_transaction_enabled",
                    "recoverability_enabled",
                    "reactivation_prompt_mode",
                    "geometry_prompts_enabled",
                    "sam2_add_new_points_or_box_called",
                    "long_term_memory_admitted_count",
                    "long_term_admission_skip_reasons",
                    "prompt_new_object_assignment_count",
                    "probation_output_mask_count",
                    "shadow_output_mask_count",
                    "source_mapping_accepted_count",
                    "source_mapping_skip_reasons",
                    "state_counts",
                ]
            }
            for row in rows
        ]
        + missing_rows,
        "missing_or_alias_controls": missing_rows,
    }
    final_decision = {
        "schema_version": "stream4d_v107_phase11_final_decision_v1",
        "status": "FROZEN_HOLDOUT_SUBSET_RUN_NOT_FINAL_PASS",
        "goal_directive": "ignore_gate_continue_down",
        "final_pass": False,
        "holdout_run_once_executed": True,
        "holdout_claim": False,
        "reason": (
            "User requested continuing past gates. A scene0030_00 90-frame frozen holdout subset was run "
            "and packaged, but this is still not a complete paper-level holdout or visual PASS."
        ),
        "true_holdout_subset": {
            "scene_count": len({row["scene_id"] for row in holdout_rows}),
            "scene_ids": sorted({row["scene_id"] for row in holdout_rows}),
            "scope": "scene0030_00_start0_stride5_90f_only_not_full_paper_holdout",
            "artifact_complete_rows": sum(1 for row in holdout_rows if row.get("artifact_complete")),
            "scheduler_highres_visuals_missing": any(
                int(row.get("scheduler_highres_visual_count") or 0) == 0 for row in holdout_rows
            ),
            "scheduler_posthoc_visuals_packaged": sum(
                int(row.get("scheduler_posthoc_visual_count") or 0) for row in holdout_rows
            ),
            "scheduler_posthoc_no_model_rerun": all(
                bool(row.get("scheduler_posthoc_no_model_rerun")) for row in holdout_rows
            ),
            "probe_highres_visuals_packaged": sum(int(row.get("probe_highres_visual_count") or 0) for row in holdout_rows),
        },
        "appearance_only_control": {
            "scene_count": len({row["scene_id"] for row in appearance_control_rows}),
            "scene_ids": sorted({row["scene_id"] for row in appearance_control_rows}),
            "artifact_complete_rows": sum(1 for row in appearance_control_rows if row.get("artifact_complete")),
            "implemented_in_live_scheduler": True,
            "scope": "scene0030_00_start0_stride5_90f_control_only_not_full_ablation_matrix",
            "interpretation": "partial_and_weak_control_evidence_not_final_pass",
        },
        "code_revision": git_value(["rev-parse", "HEAD"]),
        "git_status_touched": git_value(
            [
                "status",
                "--short",
                "--",
                "tools/run_v107_phase8_g3_rolling_scheduler_smoke.py",
                "tools/build_v107_phase11_frozen_casebook.py",
                "docs/stream4d_v107_执行日志.md",
                "docs/stream4d_v107_实验结果复盘.md",
            ]
        ).splitlines(),
        "config_yaml": rel(config_path),
        "config_sha256": config_hash,
        "scheduler_script": rel(SCHEDULER_SCRIPT),
        "scheduler_script_sha256": code_hash,
        "builder_script": rel(BUILDER_SCRIPT),
        "builder_script_sha256": sha256_file(BUILDER_SCRIPT),
        "plan_doc": rel(PLAN_DOC),
        "plan_doc_sha256": sha256_file(PLAN_DOC),
        "checkpoint_hashes": [
            {
                "path": rel(SAM2_CHECKPOINT),
                "sha256": checkpoint_hash,
                "status": "hashed" if checkpoint_hash else "missing_or_skipped",
            }
        ],
        "freeze_contract": {
            "metric_contract": "diagnostic_only_not_acceptance_gate",
            "visual_protocol": "targeted_high_resolution_user_review_required",
            "small_object_policy": "small_objects_may_skip_long_term_memory",
            "exact_parity_gate": "ignored_by_user; drift_expected",
            "pose_depth_source": "LingBot-Map outputs for projection",
        },
        "artifact_counts": {
            "run_rows": len(rows),
            "holdout_subset_rows": len(holdout_rows),
            "appearance_control_rows": len(appearance_control_rows),
            "artifact_complete_rows": sum(1 for row in rows if row.get("artifact_complete")),
            "holdout_artifact_complete_rows": sum(1 for row in holdout_rows if row.get("artifact_complete")),
            "appearance_control_artifact_complete_rows": sum(
                1 for row in appearance_control_rows if row.get("artifact_complete")
            ),
            "review_entries": review_manifest["entry_count"],
            "targeted_visual_spotcheck_entries": targeted_visual_spotchecks["entry_count"],
            "missing_or_alias_control_rows": len(missing_rows),
        },
        "active_limitations": [
            "appearance_only_live_scheduler_partial_scene0030_control_only_weak",
            "user_visual_review_pending",
            "single_scene_holdout_subset_only_not_full_paper_holdout",
            "scheduler_runtime_highres_visuals_missing_repaired_with_posthoc_frozen_visuals",
            "metrics_diagnostic_only",
        ],
        "required_outputs": {
            "final_decision_json": rel(out_root / "final_decision.json"),
            "paper_metric_tables_json": rel(out_root / "paper_metric_tables.json"),
            "paper_ablation_tables_json": rel(out_root / "paper_ablation_tables.json"),
            "failure_casebook_json": rel(out_root / "failure_casebook.json"),
            "holdout_subset_runs_json": rel(out_root / "holdout_subset_runs.json"),
            "appearance_control_runs_json": rel(out_root / "appearance_control_runs.json"),
            "targeted_visual_spotchecks_json": rel(out_root / "targeted_visual_spotchecks.json"),
            "videos_dir": rel(out_root / "videos"),
            "user_review_manifest_json": rel(out_root / "user_review_manifest.json"),
        },
    }

    write_json(out_root / "holdout_subset_runs.json", {"schema_version": "stream4d_v107_phase11_holdout_subset_runs_v1", "rows": holdout_rows})
    write_json(out_root / "appearance_control_runs.json", {"schema_version": "stream4d_v107_phase11_appearance_control_runs_v1", "rows": appearance_control_rows})
    write_json(out_root / "paper_metric_tables.json", metric_tables)
    write_json(out_root / "paper_ablation_tables.json", ablation_tables)
    write_json(out_root / "failure_casebook.json", failure_casebook)
    write_json(out_root / "targeted_visual_spotchecks.json", targeted_visual_spotchecks)
    write_json(out_root / "user_review_manifest.json", review_manifest)
    write_json(out_root / "final_decision.json", final_decision)

    output_files = [
        out_root / "config.yaml",
        out_root / "final_decision.json",
        out_root / "paper_metric_tables.json",
        out_root / "paper_ablation_tables.json",
        out_root / "failure_casebook.json",
        out_root / "holdout_subset_runs.json",
        out_root / "appearance_control_runs.json",
        out_root / "targeted_visual_spotchecks.json",
        out_root / "user_review_manifest.json",
    ]
    output_manifest = {
        "schema_version": "stream4d_v107_phase11_artifact_manifest_v1",
        "out_root": rel(out_root),
        "files": [
            {"path": rel(path), "exists": path.exists(), "sha256": sha256_file(path)}
            for path in output_files
        ],
        "review_entry_count": review_manifest["entry_count"],
        "targeted_visual_spotcheck_count": targeted_visual_spotchecks["entry_count"],
        "video_or_visual_copy_count": (
            sum(1 for path in (out_root / "videos").rglob("*") if path.is_file())
            if (out_root / "videos").exists()
            else 0
        ),
    }
    output_manifest["files_all_exist"] = all(row["exists"] for row in output_manifest["files"])
    write_json(out_root / "phase11_artifact_manifest.json", output_manifest)
    print(json.dumps({"out_root": rel(out_root), **final_decision["artifact_counts"]}, sort_keys=True))


if __name__ == "__main__":
    main()
