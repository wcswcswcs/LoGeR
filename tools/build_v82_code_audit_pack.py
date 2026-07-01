#!/usr/bin/env python3
"""Build a compact Stream4D v82 causal tracklet memory code/artifact audit pack."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
PACK_ROOT = REPO_ROOT / "code_audit_pack"
AUDIT_ROOT = REPO_ROOT / "Stream3D/outputs/audit"

TAG_PREFIX = "stream4d_v82_revised_causal_tracklet_memory_core_audit"

DOC_PATHS = [
    "docs/stream4d_v82_revised_causal_tracklet_memory_plan.md",
    "docs/stream4d_v82_执行日志.md",
    "docs/stream4d_v82_实验结果复盘.md",
]

CODE_PATHS = [
    "tools/build_v82_code_audit_pack.py",
    "Stream3D/tools/run_v82_revised_causal_tracklet_memory.py",
    "Stream3D/tools/run_v81_history_anchored_cmap_af_l2h_pipeline.py",
    "Stream3D/tools/run_v80_cmap_af_l2h_pipeline.py",
]

V81_SOURCE_PATHS = [
    "Stream3D/outputs/audit/v81_history_anchored_cmap_af_l2h_pipeline_dev_r77_repair31b_snapshot_residualfloor_age1_margin005_max32/summary.json",
    "Stream3D/outputs/audit/v81_history_anchored_cmap_af_l2h_pipeline_dev_r77_repair31b_snapshot_residualfloor_age1_margin005_max32/pipeline_summary.json",
    "Stream3D/outputs/audit/v81_history_anchored_cmap_af_l2h_pipeline_dev_r77_repair31b_snapshot_residualfloor_age1_margin005_max32/appearance_feature_audit.json",
    "Stream3D/outputs/audit/v81_phase1_bootstrap_local_dev_r77_repair31b_snapshot_residualfloor_age1_margin005_max32/local_summary.json",
    "Stream3D/outputs/audit/v81_phase1_bootstrap_local_dev_r77_repair31b_snapshot_residualfloor_age1_margin005_max32/local_metric_rows.csv",
    "Stream3D/outputs/audit/v81_phase2_bootstrap_history_dev_r77_repair31b_snapshot_residualfloor_age1_margin005_max32/history_summary.json",
    "Stream3D/outputs/audit/v81_phase2_bootstrap_history_dev_r77_repair31b_snapshot_residualfloor_age1_margin005_max32/history_node_rows.csv",
    "Stream3D/outputs/audit/v81_phase2_bootstrap_history_dev_r77_repair31b_snapshot_residualfloor_age1_margin005_max32/history_update_rows.csv",
    "Stream3D/outputs/audit/v81_phase2_bootstrap_history_dev_r77_repair31b_snapshot_residualfloor_age1_margin005_max32/history_snapshot_rows.csv",
    "Stream3D/outputs/audit/v81_phase3_carrier_to_history_dev_r77_repair31b_snapshot_residualfloor_age1_margin005_max32/q_summary.json",
    "Stream3D/outputs/audit/v81_phase3_carrier_to_history_dev_r77_repair31b_snapshot_residualfloor_age1_margin005_max32/q_control_rows.csv",
    "Stream3D/outputs/audit/v81_final_decision_dev_r77_repair31b_snapshot_residualfloor_age1_margin005_max32/final_decision.json",
    "Stream3D/outputs/audit/v81_history_anchored_cmap_af_l2h_pipeline/repair2_to_repair31_summary_rows.csv",
    "Stream3D/outputs/audit/v81_dino_feature_json_scene0011_scene0050/semantic_summary.json",
    "Stream3D/outputs/audit/v81_dino_feature_json_scene0011_scene0050/semantic_edge_metric_rows.csv",
    "Stream3D/outputs/audit/v81_dino_feature_json_scene0011_scene0050/sha256_rows.csv",
    "Stream3D/outputs/audit/v81_dino_feature_json_scene0011_scene0050/missing_input_rows.csv",
]

V82_ARTIFACT_DIRS = [
    "Stream3D/outputs/audit/v82_phase0_fact_lock",
    "Stream3D/outputs/audit/v82_phase1_local_b0",
    "Stream3D/outputs/audit/v82_local_shadow",
    "Stream3D/outputs/audit/v82_phase2_object_tracklets",
    "Stream3D/outputs/audit/v82_phase2_object_tracklets_repair4_app080",
    "Stream3D/outputs/audit/v82_phase2_object_tracklets_repair5_app075",
    "Stream3D/outputs/audit/v82_phase2_object_tracklets_repair6_window4",
    "Stream3D/outputs/audit/v82_phase2_object_tracklets_repair7_app079",
    "Stream3D/outputs/audit/v82_phase2_object_tracklets_repair8_app078",
    "Stream3D/outputs/audit/v82_phase2_object_tracklets_repair9_app077",
    "Stream3D/outputs/audit/v82_phase2_object_tracklets_repair10_app076",
    "Stream3D/outputs/audit/v82_phase2_object_tracklets_repair11_app079_sigma022",
    "Stream3D/outputs/audit/v82_phase2_object_tracklets_repair12_app079_sigma026",
    "Stream3D/outputs/audit/v82_phase2_object_tracklets_repair13_app079_sigma030",
    "Stream3D/outputs/audit/v82_phase2_object_tracklets_repair14_app079_sigma022_residual_active_mean",
    "Stream3D/outputs/audit/v82_phase2_object_tracklets_repair15_app075_sigma022_residual_active_mean",
    "Stream3D/outputs/audit/v82_phase2_object_tracklets_repair16_app070_sigma022_residual_active_mean",
    "Stream3D/outputs/audit/v82_phase2_object_tracklets_repair17_app079_sigma022_residual_blend50",
    "Stream3D/outputs/audit/v82_phase3_tracklet_history",
    "Stream3D/outputs/audit/v82_phase3_tracklet_history_repair1_semctrl_mean0",
    "Stream3D/outputs/audit/v82_phase3_tracklet_history_repair2_semctrl_mean002",
    "Stream3D/outputs/audit/v82_phase3_tracklet_history_repair5_app079_sigma022",
    "Stream3D/outputs/audit/v82_phase4_tracklet_to_history_q",
    "Stream3D/outputs/audit/v82_phase4_tracklet_to_history_q_repair1_semctrl_mean0",
    "Stream3D/outputs/audit/v82_phase4_tracklet_to_history_q_repair2_semctrl_mean002",
    "Stream3D/outputs/audit/v82_phase4_tracklet_to_history_q_repair3_nonself_shuffled",
    "Stream3D/outputs/audit/v82_phase4_tracklet_to_history_q_repair5_app079_sigma022_nonself_shuffled",
    "Stream3D/outputs/audit/v82_phase4_tracklet_to_history_q_repair6_app079_sigma022_scene_key_fix",
    "Stream3D/outputs/audit/v82_phase4_tracklet_to_history_q_repair7_app079_sigma022_candidate_bridge",
    "Stream3D/outputs/audit/v82_phase4_tracklet_to_history_q_repair8_app079_sigma022_candidate_bridge_strict_source",
    "Stream3D/outputs/audit/v82_phase5_weak_history",
    "Stream3D/outputs/audit/v82_phase5_weak_history_repair2_candidate_tentative_app079_sigma022",
    "Stream3D/outputs/audit/v82_phase5_weak_history_repair3_candidate_bridge_scene_key_fix",
    "Stream3D/outputs/audit/v82_phase6_strong_history",
    "Stream3D/outputs/audit/v82_phase6_strong_history_repair2_candidate_tentative_app079_sigma022",
    "Stream3D/outputs/audit/v82_phase6_strong_history_repair3_candidate_bridge_scene_key_fix",
    "Stream3D/outputs/audit/v82_phase7_final_local",
    "Stream3D/outputs/audit/v82_phase7_final_local_repair2_candidate_tentative_app079_sigma022",
    "Stream3D/outputs/audit/v82_phase7_final_local_repair3_candidate_bridge_scene_key_fix",
    "Stream3D/outputs/audit/v82_phase8_frozen_holdout",
    "Stream3D/outputs/audit/v82_phase9_local2history",
    "Stream3D/outputs/audit/v82_phase10_casebook",
    "Stream3D/outputs/audit/v82_phase10_casebook_repair2_candidate_tentative_app079_sigma022",
    "Stream3D/outputs/audit/v82_phase10_casebook_repair3_candidate_bridge_scene_key_fix",
]

PHASE_SUMMARY_PATHS = {
    "phase0": "Stream3D/outputs/audit/v82_phase0_fact_lock/summary.json",
    "phase1": "Stream3D/outputs/audit/v82_phase1_local_b0/summary.json",
    "phase2_default_app085": "Stream3D/outputs/audit/v82_phase2_object_tracklets/summary.json",
    "phase2_repair4_app080": "Stream3D/outputs/audit/v82_phase2_object_tracklets_repair4_app080/summary.json",
    "phase2_repair5_app075": "Stream3D/outputs/audit/v82_phase2_object_tracklets_repair5_app075/summary.json",
    "phase2_repair6_window4": "Stream3D/outputs/audit/v82_phase2_object_tracklets_repair6_window4/summary.json",
    "phase2_repair7_app079": "Stream3D/outputs/audit/v82_phase2_object_tracklets_repair7_app079/summary.json",
    "phase2_repair8_app078": "Stream3D/outputs/audit/v82_phase2_object_tracklets_repair8_app078/summary.json",
    "phase2_repair9_app077": "Stream3D/outputs/audit/v82_phase2_object_tracklets_repair9_app077/summary.json",
    "phase2_repair10_app076": "Stream3D/outputs/audit/v82_phase2_object_tracklets_repair10_app076/summary.json",
    "phase2_repair11_app079_sigma022": "Stream3D/outputs/audit/v82_phase2_object_tracklets_repair11_app079_sigma022/summary.json",
    "phase2_repair12_app079_sigma026": "Stream3D/outputs/audit/v82_phase2_object_tracklets_repair12_app079_sigma026/summary.json",
    "phase2_repair13_app079_sigma030": "Stream3D/outputs/audit/v82_phase2_object_tracklets_repair13_app079_sigma030/summary.json",
    "phase2_repair14_app079_sigma022_residual_active_mean": "Stream3D/outputs/audit/v82_phase2_object_tracklets_repair14_app079_sigma022_residual_active_mean/summary.json",
    "phase2_repair15_app075_sigma022_residual_active_mean": "Stream3D/outputs/audit/v82_phase2_object_tracklets_repair15_app075_sigma022_residual_active_mean/summary.json",
    "phase2_repair16_app070_sigma022_residual_active_mean": "Stream3D/outputs/audit/v82_phase2_object_tracklets_repair16_app070_sigma022_residual_active_mean/summary.json",
    "phase2_repair17_app079_sigma022_residual_blend50": "Stream3D/outputs/audit/v82_phase2_object_tracklets_repair17_app079_sigma022_residual_blend50/summary.json",
    "phase3": "Stream3D/outputs/audit/v82_phase3_tracklet_history/summary.json",
    "phase3_repair1_semctrl_mean0": "Stream3D/outputs/audit/v82_phase3_tracklet_history_repair1_semctrl_mean0/summary.json",
    "phase3_repair2_semctrl_mean002": "Stream3D/outputs/audit/v82_phase3_tracklet_history_repair2_semctrl_mean002/summary.json",
    "phase3_current_repair5_app079_sigma022": "Stream3D/outputs/audit/v82_phase3_tracklet_history_repair5_app079_sigma022/summary.json",
    "phase4": "Stream3D/outputs/audit/v82_phase4_tracklet_to_history_q/summary.json",
    "phase4_repair1_semctrl_mean0": "Stream3D/outputs/audit/v82_phase4_tracklet_to_history_q_repair1_semctrl_mean0/summary.json",
    "phase4_repair2_semctrl_mean002": "Stream3D/outputs/audit/v82_phase4_tracklet_to_history_q_repair2_semctrl_mean002/summary.json",
    "phase4_repair3_nonself_shuffled": "Stream3D/outputs/audit/v82_phase4_tracklet_to_history_q_repair3_nonself_shuffled/summary.json",
    "phase4_repair5_app079_sigma022_nonself_shuffled": "Stream3D/outputs/audit/v82_phase4_tracklet_to_history_q_repair5_app079_sigma022_nonself_shuffled/summary.json",
    "phase4_repair6_app079_sigma022_scene_key_fix": "Stream3D/outputs/audit/v82_phase4_tracklet_to_history_q_repair6_app079_sigma022_scene_key_fix/summary.json",
    "phase4_current_candidate_bridge": "Stream3D/outputs/audit/v82_phase4_tracklet_to_history_q_repair7_app079_sigma022_candidate_bridge/summary.json",
    "phase4_repair8_candidate_bridge_strict_source": "Stream3D/outputs/audit/v82_phase4_tracklet_to_history_q_repair8_app079_sigma022_candidate_bridge_strict_source/summary.json",
    "phase5": "Stream3D/outputs/audit/v82_phase5_weak_history/summary.json",
    "phase5_repair2_candidate_tentative_app079_sigma022": "Stream3D/outputs/audit/v82_phase5_weak_history_repair2_candidate_tentative_app079_sigma022/summary.json",
    "phase5_current_candidate_bridge_scene_key_fix": "Stream3D/outputs/audit/v82_phase5_weak_history_repair3_candidate_bridge_scene_key_fix/summary.json",
    "phase6": "Stream3D/outputs/audit/v82_phase6_strong_history/summary.json",
    "phase6_repair2_candidate_tentative_app079_sigma022": "Stream3D/outputs/audit/v82_phase6_strong_history_repair2_candidate_tentative_app079_sigma022/summary.json",
    "phase6_current_candidate_bridge_scene_key_fix": "Stream3D/outputs/audit/v82_phase6_strong_history_repair3_candidate_bridge_scene_key_fix/summary.json",
    "phase7": "Stream3D/outputs/audit/v82_phase7_final_local/summary.json",
    "phase7_repair2_candidate_tentative_app079_sigma022": "Stream3D/outputs/audit/v82_phase7_final_local_repair2_candidate_tentative_app079_sigma022/summary.json",
    "phase7_current_candidate_bridge_scene_key_fix": "Stream3D/outputs/audit/v82_phase7_final_local_repair3_candidate_bridge_scene_key_fix/summary.json",
    "phase8": "Stream3D/outputs/audit/v82_phase8_frozen_holdout/summary.json",
    "phase9": "Stream3D/outputs/audit/v82_phase9_local2history/summary.json",
    "phase10": "Stream3D/outputs/audit/v82_phase10_casebook/summary.json",
    "phase10_repair2_candidate_tentative_app079_sigma022": "Stream3D/outputs/audit/v82_phase10_casebook_repair2_candidate_tentative_app079_sigma022/summary.json",
    "phase10_current_candidate_bridge_scene_key_fix": "Stream3D/outputs/audit/v82_phase10_casebook_repair3_candidate_bridge_scene_key_fix/summary.json",
}

LARGE_EXTERNAL_INPUTS = [
    {
        "path": "Stream3D/outputs/audit/v81_dino_feature_json_scene0011_scene0050/mask_feature_rows.csv",
        "reason": "large predecessor DINO feature CSV; summary and sha256 sidecars are packaged instead",
    },
    {
        "path": "Stream3D/outputs/audit/v81_phase3_carrier_to_history_dev_r77_repair31b_snapshot_residualfloor_age1_margin005_max32/q_rows.csv",
        "reason": "large predecessor q-row diagnostic table; q_summary and q_control_rows are packaged instead",
    },
]

LIGHT_EXTS = {".csv", ".json", ".jsonl", ".log", ".md", ".patch", ".py", ".txt", ".yaml", ".yml"}
EXCLUDED_SUFFIXES = {
    ".avi",
    ".bin",
    ".ckpt",
    ".gz",
    ".jpg",
    ".jpeg",
    ".mov",
    ".mp4",
    ".npy",
    ".npz",
    ".pickle",
    ".pkl",
    ".png",
    ".pt",
    ".pth",
    ".tar",
    ".zip",
}
EXCLUDED_PARTS = {".git", "__pycache__", "code_audit_pack", "checkpoints", "data", "weights"}


def repo_rel(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - malformed evidence should be visible.
        return {"read_error": str(exc)}


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def run_text(cmd: list[str], cwd: Path = REPO_ROOT) -> str:
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
    except FileNotFoundError as exc:
        return f"COMMAND_NOT_FOUND: {cmd[0]} ({exc})\n"
    return f"$ {' '.join(cmd)}\nreturncode={proc.returncode}\n\n{proc.stdout}"


def should_copy_file(path: Path, max_file_bytes: int) -> tuple[bool, str]:
    rel_parts = set(path.resolve().relative_to(REPO_ROOT).parts)
    if rel_parts & EXCLUDED_PARTS:
        return False, "excluded_path_part"
    suffix = path.suffix.lower()
    if suffix in EXCLUDED_SUFFIXES:
        return False, "excluded_suffix"
    if suffix not in LIGHT_EXTS:
        return False, "non_light_extension"
    size = path.stat().st_size
    if size > max_file_bytes:
        return False, f"size_gt_{max_file_bytes}"
    return True, ""


def copy_file(src: Path, packet_dir: Path, copied: list[str]) -> None:
    rel = repo_rel(src)
    dst = packet_dir / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    copied.append(rel)


def copy_selected(src_rel: str, packet_dir: Path, copied: list[str], missing: list[str], skipped: list[dict[str, Any]], max_file_bytes: int) -> None:
    src = REPO_ROOT / src_rel
    if not src.exists():
        missing.append(src_rel)
        return
    if src.is_file():
        ok, reason = should_copy_file(src, max_file_bytes)
        if ok:
            copy_file(src, packet_dir, copied)
        else:
            skipped.append({"path": src_rel, "size_bytes": src.stat().st_size, "reason": reason})
        return
    for path in sorted(src.rglob("*")):
        if not path.is_file():
            continue
        rel = repo_rel(path)
        ok, reason = should_copy_file(path, max_file_bytes)
        if ok:
            copy_file(path, packet_dir, copied)
        else:
            skipped.append({"path": rel, "size_bytes": path.stat().st_size, "reason": reason})


def load_phase_summaries() -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for name, rel in PHASE_SUMMARY_PATHS.items():
        path = REPO_ROOT / rel
        summaries[name] = read_json(path) if path.exists() else {"missing": True}
    return summaries


def select_phase2(summaries: dict[str, Any]) -> dict[str, Any]:
    candidates = []
    for name, summary in summaries.items():
        if not name.startswith("phase2_") or summary.get("missing"):
            continue
        gate = summary.get("gate", {})
        if summary.get("can_enter_next_phase") is True and gate.get("pass") is True:
            candidates.append(
                (
                    float(summary.get("eligible_tracklet_coverage_rate") or 0.0),
                    float(summary.get("full_minus_semantic_score") or 0.0),
                    float(summary.get("full_minus_shuffled_score") or 0.0),
                    name,
                    summary,
                )
            )
    if not candidates:
        return {"selected": "", "reason": "no passing phase2 summary found"}
    candidates.sort(reverse=True)
    coverage, full_semantic, full_shuffled, name, summary = candidates[0]
    return {
        "selected": name,
        "selection_rule": "max eligible_tracklet_coverage_rate, then full_minus_semantic_score, then full_minus_shuffled_score among GT-free passing phase2 gates",
        "eligible_tracklet_coverage_rate": coverage,
        "full_minus_semantic_score": full_semantic,
        "full_minus_shuffled_score": full_shuffled,
        "decision": summary.get("decision"),
    }


def packet_files(packet_dir: Path) -> list[Path]:
    return sorted(path for path in packet_dir.rglob("*") if path.is_file())


def write_payload_sidecars(packet_dir: Path) -> dict[str, int]:
    filelist = packet_dir / "PAYLOAD_FILELIST.txt"
    hashes = packet_dir / "PAYLOAD_SHA256SUMS.txt"
    write_text(filelist, "")
    write_text(hashes, "")
    files = packet_files(packet_dir)
    write_text(filelist, "\n".join(path.relative_to(packet_dir).as_posix() for path in files) + "\n")
    lines = []
    for path in files:
        rel = path.relative_to(packet_dir).as_posix()
        if rel in {"PAYLOAD_FILELIST.txt", "PAYLOAD_SHA256SUMS.txt"}:
            continue
        lines.append(f"{sha256_file(path)}  {rel}")
    write_text(hashes, "\n".join(lines) + "\n")
    return {"payload_file_count": len(packet_files(packet_dir)), "payload_hash_rows": len(lines)}


def write_git_and_compile_context(packet_dir: Path) -> None:
    write_text(packet_dir / "GIT_STATUS_SHORT.txt", run_text(["git", "status", "--short"]))
    diff_targets = [*DOC_PATHS, *CODE_PATHS]
    existing = [path for path in diff_targets if (REPO_ROOT / path).exists()]
    diff_text = run_text(["git", "diff", "--", *existing]) if existing else ""
    write_text(packet_dir / "SCOPED_GIT_DIFF.patch", diff_text)
    py_paths = [path for path in CODE_PATHS if path.endswith(".py") and (REPO_ROOT / path).is_file()]
    if py_paths:
        proc = subprocess.run(
            [sys.executable, "-m", "py_compile", *py_paths],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        write_text(
            packet_dir / "PY_COMPILE_CHECK.txt",
            f"$ {sys.executable} -m py_compile {' '.join(py_paths)}\n"
            f"returncode={proc.returncode}\n\n{proc.stdout}",
        )
    else:
        write_text(packet_dir / "PY_COMPILE_CHECK.txt", "no selected python files\n")


def write_phase_summary(packet_dir: Path) -> dict[str, Any]:
    summaries = load_phase_summaries()
    selected = select_phase2(summaries)
    phase3 = summaries.get("phase3_current_repair5_app079_sigma022", summaries.get("phase3", {}))
    phase4 = summaries.get("phase4_current_candidate_bridge", summaries.get("phase4", {}))
    phase5 = summaries.get("phase5_current_candidate_bridge_scene_key_fix", summaries.get("phase5", {}))
    phase6 = summaries.get("phase6_current_candidate_bridge_scene_key_fix", summaries.get("phase6", {}))
    phase7 = summaries.get("phase7_current_candidate_bridge_scene_key_fix", summaries.get("phase7", {}))
    phase9 = summaries.get("phase9", {})
    phase10 = summaries.get("phase10_current_candidate_bridge_scene_key_fix", summaries.get("phase10", {}))
    final_decision = (
        phase10.get("final_decision")
        or phase9.get("final_decision")
        or phase7.get("decision")
        or phase4.get("decision")
        or ""
    )
    phase10_done = not bool(phase10.get("missing"))
    method_success = bool(phase7.get("can_enter_method_mode_local2history"))
    phase_status = {
        "schema": "stream4d_v82_phase_status_pack_summary_v1",
        "objective_complete": phase10_done,
        "experiment_execution_complete": phase10_done,
        "status": "complete_no_go_audit_packet" if phase10_done and final_decision.startswith("NO_GO") else "partial_or_inconclusive_audit_packet",
        "highest_verified_phase": "phase10" if phase10_done else "phase2",
        "phase3_plus_run": not bool(summaries.get("phase3", {}).get("missing")),
        "method_success": method_success,
        "method_mode_local2history_success_claimed": method_success,
        "final_decision": final_decision,
        "phase0_decision": summaries.get("phase0", {}).get("decision"),
        "phase1_decision": summaries.get("phase1", {}).get("decision"),
        "phase1_can_enter_method_mode_local2history": summaries.get("phase1", {}).get(
            "can_enter_method_mode_local2history"
        ),
        "phase2_selected": selected,
        "phase3_decision": phase3.get("decision"),
        "phase3_confirmed_node_count": phase3.get("confirmed_node_count"),
        "phase4_decision": phase4.get("decision"),
        "phase4_Q_obj_eligible_coverage_rate": phase4.get("Q_obj_eligible_coverage_rate"),
        "phase4_full_minus_shuffled_top1_confidence": phase4.get("full_minus_shuffled_top1_confidence"),
        "phase5_decision": phase5.get("decision"),
        "phase5_history_assignment_coverage_rate": phase5.get("history_assignment_coverage_rate"),
        "phase5_history_assignment_entropy_mean": phase5.get("history_assignment_entropy_mean"),
        "phase6_decision": phase6.get("decision"),
        "phase7_decision": phase7.get("decision"),
        "phase8_decision": summaries.get("phase8", {}).get("decision"),
        "phase9_decision": phase9.get("decision"),
        "phase10_decision": phase10.get("decision"),
        "phase10_case_count": phase10.get("case_count"),
        "phase10_failure_type_counts": phase10.get("failure_type_counts"),
        "phase_summaries": summaries,
    }
    write_text(packet_dir / "PHASE_STATUS_SUMMARY.json", json.dumps(phase_status, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return phase_status


def write_readme(
    packet_dir: Path,
    tag: str,
    phase_status: dict[str, Any],
    missing: list[str],
    skipped: list[dict[str, Any]],
    max_file_bytes: int,
) -> None:
    phase2 = phase_status.get("phase2_selected", {})
    readme = f"""# {tag}

Stream4D v82 revised causal object-tracklet memory core code and artifact audit packet.

## Scope

This packet includes the v82 plan, execution log, retrospective log, v82 runner,
the predecessor v80/v81 runners used by the local replay, Phase0-Phase10 v82
artifacts, Phase3/Phase4 repair branches, and compact v81 predecessor evidence.
It intentionally excludes raw
datasets, tensor caches, checkpoints, media, old audit packs, and oversized
external input tables.

## Current Verified Status

- experiment execution complete: `{phase_status.get("experiment_execution_complete")}`
- highest verified phase: `{phase_status.get("highest_verified_phase")}`
- phase3_plus_run: `{phase_status.get("phase3_plus_run")}`
- method-mode local2history success claimed: `{phase_status.get("method_mode_local2history_success_claimed")}`
- final decision: `{phase_status.get("final_decision")}`
- phase0 decision: `{phase_status.get("phase0_decision")}`
- phase1 decision: `{phase_status.get("phase1_decision")}`
- phase1 can enter method mode: `{phase_status.get("phase1_can_enter_method_mode_local2history")}`
- selected Phase2 artifact: `{phase2.get("selected", "")}`
- selected Phase2 decision: `{phase2.get("decision", "")}`
- selected Phase2 coverage: `{phase2.get("eligible_tracklet_coverage_rate", "")}`
- selected Phase2 full-minus-semantic: `{phase2.get("full_minus_semantic_score", "")}`
- phase3 decision: `{phase_status.get("phase3_decision")}`
- phase3 confirmed node count: `{phase_status.get("phase3_confirmed_node_count")}`
- phase4 decision: `{phase_status.get("phase4_decision")}`
- phase4 Q eligible coverage: `{phase_status.get("phase4_Q_obj_eligible_coverage_rate")}`
- phase4 full-minus-shuffled: `{phase_status.get("phase4_full_minus_shuffled_top1_confidence")}`
- phase5 decision: `{phase_status.get("phase5_decision")}`
- phase5 coverage: `{phase_status.get("phase5_history_assignment_coverage_rate")}`
- phase5 entropy: `{phase_status.get("phase5_history_assignment_entropy_mean")}`
- phase6 decision: `{phase_status.get("phase6_decision")}`
- phase10 decision: `{phase_status.get("phase10_decision")}`
- phase10 case count: `{phase_status.get("phase10_case_count")}`

## Start Here

- plan: `docs/stream4d_v82_revised_causal_tracklet_memory_plan.md`
- execution log: `docs/stream4d_v82_执行日志.md`
- retrospective: `docs/stream4d_v82_实验结果复盘.md`
- runner: `Stream3D/tools/run_v82_revised_causal_tracklet_memory.py`
- phase status summary: `PHASE_STATUS_SUMMARY.json`
- manifest: `PACKET_MANIFEST.json`

## Validation Sidecars

- inside packet: `PAYLOAD_FILELIST.txt`, `PAYLOAD_SHA256SUMS.txt`, `PY_COMPILE_CHECK.txt`, `GIT_STATUS_SHORT.txt`, `SCOPED_GIT_DIFF.patch`
- beside zip: `.zip.sha256`, `.unzip_test.txt`, `.payload_sha256_check.txt`, `.entry_diff.txt`, `.expected_entries.txt`, `.actual_entries.txt`, `.build_summary.json`

## Rebuild

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/build_v82_code_audit_pack.py
```

## Missing Selected Paths

```text
{chr(10).join(missing) if missing else "none"}
```

## Skipped Files

Files larger than `{max_file_bytes}` bytes or non-light artifact formats are not
copied. See `EXCLUDED_PATHS.txt` for exact paths and reasons.

Skipped count: `{len(skipped)}`
"""
    write_text(packet_dir / "PACKET_README.md", readme)


def write_manifest(
    packet_dir: Path,
    tag: str,
    copied: list[str],
    missing: list[str],
    skipped: list[dict[str, Any]],
    phase_status: dict[str, Any],
    max_file_bytes: int,
) -> None:
    manifest = {
        "schema": "stream4d_v82_code_audit_pack_v1",
        "tag": tag,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "build_command": " ".join([sys.executable, *sys.argv]),
        "build_cwd": REPO_ROOT.as_posix(),
        "status": phase_status.get("status"),
        "objective_complete": phase_status.get("objective_complete"),
        "experiment_execution_complete": phase_status.get("experiment_execution_complete"),
        "highest_verified_phase": phase_status.get("highest_verified_phase"),
        "phase3_plus_run": phase_status.get("phase3_plus_run"),
        "method_success": phase_status.get("method_success"),
        "method_mode_local2history_success_claimed": phase_status.get("method_mode_local2history_success_claimed"),
        "final_decision": phase_status.get("final_decision"),
        "copied_path_count_before_sidecars": len(copied),
        "missing_selected_path_count": len(missing),
        "skipped_file_count": len(skipped),
        "max_file_bytes": max_file_bytes,
        "selected_paths": [*DOC_PATHS, *CODE_PATHS, *V81_SOURCE_PATHS, *V82_ARTIFACT_DIRS],
        "large_external_inputs_not_embedded": LARGE_EXTERNAL_INPUTS,
        "phase_status": {
            "phase0_decision": phase_status.get("phase0_decision"),
            "phase1_decision": phase_status.get("phase1_decision"),
            "phase1_can_enter_method_mode_local2history": phase_status.get(
                "phase1_can_enter_method_mode_local2history"
            ),
            "phase2_selected": phase_status.get("phase2_selected"),
            "phase3_decision": phase_status.get("phase3_decision"),
            "phase4_decision": phase_status.get("phase4_decision"),
            "phase5_decision": phase_status.get("phase5_decision"),
            "phase6_decision": phase_status.get("phase6_decision"),
            "phase7_decision": phase_status.get("phase7_decision"),
            "phase8_decision": phase_status.get("phase8_decision"),
            "phase9_decision": phase_status.get("phase9_decision"),
            "phase10_decision": phase_status.get("phase10_decision"),
            "phase10_case_count": phase_status.get("phase10_case_count"),
        },
        "missing_selected_paths": missing,
        "skipped_files": skipped,
        "exclusions": [
            "raw datasets",
            "*.pt/*.pth/*.npy/*.npz tensor artifacts",
            "checkpoints and weights",
            "media files",
            "old code_audit_pack archives",
            "__pycache__",
        ],
    }
    write_text(packet_dir / "PACKET_MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def zip_packet(packet_dir: Path, zip_path: Path) -> list[str]:
    entries: list[str] = []
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in packet_files(packet_dir):
            arcname = Path(packet_dir.name) / path.relative_to(packet_dir)
            zf.write(path, arcname.as_posix())
            entries.append(arcname.as_posix())
    return sorted(entries)


def validate_zip(packet_dir: Path, zip_path: Path, tag: str, entries: list[str]) -> dict[str, Any]:
    unzip_proc = subprocess.run(
        ["unzip", "-t", zip_path.name],
        cwd=PACK_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    unzip_test_path = PACK_ROOT / f"{tag}.unzip_test.txt"
    write_text(unzip_test_path, f"$ unzip -t {zip_path.name}\nreturncode={unzip_proc.returncode}\n\n{unzip_proc.stdout}")

    payload_proc = subprocess.run(
        ["sha256sum", "-c", "PAYLOAD_SHA256SUMS.txt"],
        cwd=packet_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    payload_check_path = PACK_ROOT / f"{tag}.payload_sha256_check.txt"
    write_text(payload_check_path, f"$ sha256sum -c PAYLOAD_SHA256SUMS.txt\nreturncode={payload_proc.returncode}\n\n{payload_proc.stdout}")

    expected_entries = sorted(f"{packet_dir.name}/{path.relative_to(packet_dir).as_posix()}" for path in packet_files(packet_dir))
    actual_entries = sorted(entries)
    expected_path = PACK_ROOT / f"{tag}.expected_entries.txt"
    actual_path = PACK_ROOT / f"{tag}.actual_entries.txt"
    write_text(expected_path, "\n".join(expected_entries) + "\n")
    write_text(actual_path, "\n".join(actual_entries) + "\n")
    missing_in_zip = sorted(set(expected_entries) - set(actual_entries))
    extra_in_zip = sorted(set(actual_entries) - set(expected_entries))
    entry_diff_path = PACK_ROOT / f"{tag}.entry_diff.txt"
    write_text(
        entry_diff_path,
        "missing_in_zip:\n"
        + ("\n".join(missing_in_zip) if missing_in_zip else "none")
        + "\n\nextra_in_zip:\n"
        + ("\n".join(extra_in_zip) if extra_in_zip else "none")
        + "\n",
    )

    zip_sha_path = PACK_ROOT / f"{tag}.zip.sha256"
    zip_sha = sha256_file(zip_path)
    write_text(zip_sha_path, f"{zip_sha}  {zip_path.name}\n")

    return {
        "zip_path": zip_path.as_posix(),
        "zip_sha256": zip_sha,
        "zip_size_bytes": zip_path.stat().st_size,
        "zip_entry_count": len(entries),
        "unzip_test_returncode": unzip_proc.returncode,
        "payload_sha256_check_returncode": payload_proc.returncode,
        "entry_diff_missing_count": len(missing_in_zip),
        "entry_diff_extra_count": len(extra_in_zip),
        "sidecars": [
            unzip_test_path.as_posix(),
            payload_check_path.as_posix(),
            entry_diff_path.as_posix(),
            expected_path.as_posix(),
            actual_path.as_posix(),
            zip_sha_path.as_posix(),
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="code_audit_pack")
    parser.add_argument("--tag", default="")
    parser.add_argument("--max-file-bytes", type=int, default=80 * 1024 * 1024)
    args = parser.parse_args()

    output_dir = REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"{TAG_PREFIX}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    packet_dir = output_dir / tag
    zip_path = output_dir / f"{tag}.zip"
    if packet_dir.exists() or zip_path.exists():
        raise SystemExit(f"Refusing to overwrite existing packet: {tag}")
    packet_dir.mkdir(parents=True)

    copied: list[str] = []
    missing: list[str] = []
    skipped: list[dict[str, Any]] = []

    for rel in [*DOC_PATHS, *CODE_PATHS, *V81_SOURCE_PATHS, *V82_ARTIFACT_DIRS]:
        copy_selected(rel, packet_dir, copied, missing, skipped, args.max_file_bytes)

    for item in LARGE_EXTERNAL_INPUTS:
        path = REPO_ROOT / str(item["path"])
        if path.exists():
            skipped.append({"path": item["path"], "size_bytes": path.stat().st_size, "reason": item["reason"]})

    write_text(
        packet_dir / "MISSING_PATHS.txt",
        "\n".join(missing) + ("\n" if missing else ""),
    )
    write_text(
        packet_dir / "EXCLUDED_PATHS.txt",
        "\n".join(f"{row['size_bytes']}\t{row['reason']}\t{row['path']}" for row in skipped)
        + ("\n" if skipped else ""),
    )
    write_git_and_compile_context(packet_dir)
    phase_status = write_phase_summary(packet_dir)
    write_readme(packet_dir, tag, phase_status, missing, skipped, args.max_file_bytes)
    write_manifest(packet_dir, tag, copied, missing, skipped, phase_status, args.max_file_bytes)
    payload_counts = write_payload_sidecars(packet_dir)

    entries = zip_packet(packet_dir, zip_path)
    validation = validate_zip(packet_dir, zip_path, tag, entries)
    summary = {
        "tag": tag,
        "packet_dir": packet_dir.as_posix(),
        "copied_path_count_before_sidecars": len(copied),
        "missing_selected_paths": missing,
        "skipped_file_count": len(skipped),
        **payload_counts,
        **validation,
    }
    summary_path = output_dir / f"{tag}.build_summary.json"
    write_text(summary_path, json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    write_text(output_dir / ".latest_stream4d_v82_pack_tag", tag + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
