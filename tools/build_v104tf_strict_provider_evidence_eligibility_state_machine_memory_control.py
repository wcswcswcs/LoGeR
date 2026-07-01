#!/usr/bin/env python3
"""Build ACL2 v104-TF strict-provider/state-machine audit artifacts.

This builder is intentionally artifact-backed.  It imports v103/v101/v102
evidence, rechecks the v104 strict provider gates, and refuses runtime action
unless the plan's provider/action-entry conditions are satisfied.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path("results/acl2_v104tf_strict_provider_evidence_eligibility_state_machine_memory_control")
V103 = Path("results/acl2_v103tf_semantic_geometric_evidence_eligibility_readswa_ttt_memory_control")
PLAN = Path("docs/ACL2_v104TF_StrictProvider_EvidenceEligibilityStateMachine_MemoryControl_ExperimentPlan.md")

SELECTED_ACTION_CASES = [
    "02_017_018",
    "00_019_020",
    "05_018_019",
    "05_019_020",
    "02_016_017",
    "01_013_014",
    "00_008_009",
    "01_001_002",
    "02_004_005",
    "05_002_003",
    "05_007_008",
]

STRICT_POSITIVES = {"02_017_018"}
EXPLORATORY_POSITIVES = {
    "00_019_020",
    "05_018_019",
    "05_019_020",
    "02_016_017",
    "01_013_014",
}
SAFE_GOOD_CONTROLS = {
    "00_008_009",
    "01_001_002",
    "02_004_005",
    "05_002_003",
    "05_007_008",
}

FORBIDDEN_REPEATS = [
    "READ weak-context skip / anchor rescue / beta ladder",
    "old Track E source gate / source replace / merge alpha",
    "R_same / query_hit / freshness single-threshold action",
    "aggregate stable-anchor gate",
    "old query-soft ge75/ge90 rho sweep",
    "TTT write mass / retention mass action",
    "boundary proxy promotion without true object-boundary provider",
]

ALLOWED_PROVIDER_ROUTES = [
    ("READ_CURRENT_SUPPORT_PROVIDER", "READ current-support provider", "provider_only"),
    ("SWA_CACHE_KV_STABILITY", "SWA cache K/V stability and top-k identity", "internal_cue_only"),
    ("TTT_IDENTITY_WRITE_TO_USE", "TTT identity write-to-use diagnostic", "diagnostic_provider_only"),
    ("SAME_SPACE_SB_STATE", "Same-space S-B hidden state representation", "state_representation_only"),
    ("STAGE_C_SEED_MASKLET_MAPPING", "Stage-C seed/masklet mapping provider candidate", "provider_candidate"),
    ("DELAY_HOLD_ACTION_CLUE", "DELAY/HOLD action-surface clue", "requires_new_provider_gate"),
]

ALLOWED_ACTION_CLUES = [
    ("DELAY_UPDATE", "v102/v103 delay can move a strict-positive trace", "blocked_without_provider"),
    ("HOLD_PREVIOUS", "hold previous reference can move trace but harmed controls historically", "blocked_without_provider"),
    ("CONTEXT_ONLY_DEMOTION", "context-only demotion is a diagnostic action family", "blocked_without_provider"),
    ("QUERY_HEAD_LOCAL_ROUTING", "A5 hook contract exists but action body/control readiness failed", "blocked_without_provider_controls"),
    ("TRANSMIT_SUPPORTED", "positive transmit path requires SCALE_ELIGIBLE strict anchors", "blocked_without_provider"),
]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: stringify(row.get(k, "")) for k in fieldnames})


def stringify(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return ";".join(str(v) for v in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def rel(path: Path) -> str:
    return path.as_posix()


def to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        s = str(value).strip()
        if not s or s.lower() == "nan":
            return default
        return float(s)
    except Exception:
        return default


def to_int(value: Any, default: int = 0) -> int:
    try:
        s = str(value).strip()
        if not s:
            return default
        return int(float(s))
    except Exception:
        return default


def index_by_case(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        case_id = row.get("case_id", "")
        if case_id and case_id not in out:
            out[case_id] = row
    return out


def latest_selected11_trace_root() -> Path:
    preferred = (
        ROOT
        / "stage1_provider/runtime_masklet_instance_trace_smoke_selected11_q128_currentuniverse_20260701_continue"
    )
    if preferred.exists():
        return preferred
    candidates = sorted((ROOT / "stage1_provider").glob("runtime_masklet_instance_trace_smoke_selected11*"))
    if not candidates:
        return preferred
    return max(candidates, key=lambda p: p.stat().st_mtime)


def latest_no_trace_pose_sha_parity_root() -> Path:
    preferred = ROOT / "stage1_provider/no_trace_pose_sha_parity_selected11_20260701_continue"
    if preferred.exists():
        return preferred
    candidates = sorted((ROOT / "stage1_provider").glob("no_trace_pose_sha_parity_selected11*"))
    if not candidates:
        return preferred
    return max(candidates, key=lambda p: p.stat().st_mtime)


def latest_selected11_a5_trace_root() -> Path:
    preferred = (
        ROOT
        / "stage1_provider/runtime_masklet_instance_trace_smoke_selected11_q128_a5trace_same_seed_20260701_continue"
    )
    if preferred.exists():
        return preferred
    candidates = sorted((ROOT / "stage1_provider").glob("runtime_masklet_instance_trace_smoke_selected11*q128_a5trace*"))
    if not candidates:
        return preferred
    return max(candidates, key=lambda p: p.stat().st_mtime)


def latest_a5_no_trace_pose_sha_parity_root() -> Path:
    preferred = (
        ROOT
        / "stage1_provider/no_trace_pose_sha_parity_selected11_q128_a5trace_same_seed_20260701_continue"
    )
    if preferred.exists():
        return preferred
    candidates = sorted((ROOT / "stage1_provider").glob("no_trace_pose_sha_parity_selected11*q128_a5trace*"))
    if not candidates:
        return preferred
    return max(candidates, key=lambda p: p.stat().st_mtime)


def latest_selected11_a5_same_masklet_trace_root() -> Path:
    preferred = (
        ROOT
        / "stage1_provider/runtime_masklet_instance_trace_smoke_selected11_q128_a5trace_same_masklet_20260702_continue"
    )
    if preferred.exists():
        return preferred
    candidates = sorted((ROOT / "stage1_provider").glob("runtime_masklet_instance_trace_smoke_selected11*q128_a5trace_same_masklet*"))
    if not candidates:
        return preferred
    return max(candidates, key=lambda p: p.stat().st_mtime)


def latest_a5_same_masklet_no_trace_pose_sha_parity_root() -> Path:
    preferred = (
        ROOT
        / "stage1_provider/no_trace_pose_sha_parity_selected11_q128_a5trace_same_masklet_20260702_continue"
    )
    if preferred.exists():
        return preferred
    candidates = sorted((ROOT / "stage1_provider").glob("no_trace_pose_sha_parity_selected11*q128_a5trace_same_masklet*"))
    if not candidates:
        return preferred
    return max(candidates, key=lambda p: p.stat().st_mtime)


def latest_selected5_a5_q4096raw_trace_root(mode: str) -> Path:
    preferred = (
        ROOT
        / f"stage1_provider/runtime_masklet_instance_trace_smoke_selected5_q4096raw_a5trace_{mode}_20260702_continue"
    )
    if preferred.exists():
        return preferred
    candidates = sorted((ROOT / "stage1_provider").glob(f"runtime_masklet_instance_trace_smoke_selected5*q4096raw_a5trace_{mode}*"))
    if not candidates:
        return preferred
    return max(candidates, key=lambda p: p.stat().st_mtime)


def latest_selected5_a5_q4096raw_no_trace_pose_sha_parity_root(mode: str) -> Path:
    preferred = (
        ROOT
        / f"stage1_provider/no_trace_pose_sha_parity_selected5_q4096raw_a5trace_{mode}_20260702_continue"
    )
    if preferred.exists():
        return preferred
    candidates = sorted((ROOT / "stage1_provider").glob(f"no_trace_pose_sha_parity_selected5*q4096raw_a5trace_{mode}*"))
    if not candidates:
        return preferred
    return max(candidates, key=lambda p: p.stat().st_mtime)


def tensor_nonnegative_stats(payload: dict[str, Any], key: str, torch_mod: Any) -> dict[str, Any]:
    value = payload.get(key)
    if value is None or not hasattr(value, "numel"):
        return {
            f"{key}_present": False,
            f"{key}_shape": "",
            f"{key}_numel": 0,
            f"{key}_nonnegative_count": 0,
            f"{key}_min_nonnegative": "",
            f"{key}_max_nonnegative": "",
            f"{key}_unique_nonnegative_count": 0,
        }
    tensor = value.detach().cpu()
    numel = int(tensor.numel())
    if numel == 0:
        nonnegative = 0
        min_value: int | str = ""
        max_value: int | str = ""
        unique_count = 0
    else:
        mask = tensor >= 0
        nonnegative = int(mask.sum().item())
        if nonnegative:
            valid = tensor[mask]
            min_value = int(valid.min().item())
            max_value = int(valid.max().item())
            unique_count = int(torch_mod.unique(valid).numel())
        else:
            min_value = ""
            max_value = ""
            unique_count = 0
    return {
        f"{key}_present": True,
        f"{key}_shape": list(tensor.shape),
        f"{key}_numel": numel,
        f"{key}_nonnegative_count": nonnegative,
        f"{key}_min_nonnegative": min_value,
        f"{key}_max_nonnegative": max_value,
        f"{key}_unique_nonnegative_count": unique_count,
    }


def tensor_bool_stats(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if value is None or not hasattr(value, "numel"):
        return {
            f"{key}_present": False,
            f"{key}_shape": "",
            f"{key}_numel": 0,
            f"{key}_true_count": 0,
            f"{key}_true_frac": 0.0,
        }
    tensor = value.detach().cpu().bool()
    numel = int(tensor.numel())
    true_count = int(tensor.sum().item()) if numel else 0
    return {
        f"{key}_present": True,
        f"{key}_shape": list(tensor.shape),
        f"{key}_numel": numel,
        f"{key}_true_count": true_count,
        f"{key}_true_frac": true_count / numel if numel else 0.0,
    }


def audit_selected11_fresh_trace(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows_by_case: dict[str, dict[str, Any]] = {}
    summary = {
        "schema": "acl2_v104_selected11_fresh_trace_payload_summary_v1",
        "trace_root": rel(root),
        "selected_case_count": len(SELECTED_ACTION_CASES),
        "trace_file_count": 0,
        "trace_loaded_count": 0,
        "trace_load_failed_count": 0,
        "direct_masklet_payload_case_count": 0,
        "direct_seed_payload_case_count": 0,
        "strict_current_cache_component_candidate_case_count": 0,
        "strict_current_cache_seed_candidate_case_count": 0,
        "ttt_prev_anchor_identity_available_case_count": 0,
        "per_chunk_geometry_sidecar_file_count": 0,
        "query_masklet_nonnegative_total": 0,
        "topk_masklet_nonnegative_total": 0,
        "query_seed_nonnegative_total": 0,
        "topk_seed_nonnegative_total": 0,
        "same_masklet_true_total": 0,
        "same_seed_true_total": 0,
        "runtime_action_allowed": False,
        "strict_provider_materializable_from_fresh_trace": False,
        "materialization_blocker": (
            "fresh READ_NO_ACTION traces materialize direct Stage-C masklet/seed payloads, "
            "but action-ready provider controls, nonproxy join, query/head-local carrier, "
            "and write/cache/current/read/L3 chain are still required"
        ),
    }

    if not root.exists():
        for case_id in SELECTED_ACTION_CASES:
            rows_by_case[case_id] = {
                "case_id": case_id,
                "trace_root": rel(root),
                "trace_path": "",
                "load_ok": False,
                "load_error": "trace_root_missing",
                "direct_masklet_payload_materialized": False,
                "direct_seed_payload_materialized": False,
                "strict_current_cache_component_candidate": False,
                "strict_current_cache_seed_candidate": False,
                "runtime_action_allowed": False,
            }
        return [rows_by_case[c] for c in SELECTED_ACTION_CASES], summary

    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment diagnostic
        for case_id in SELECTED_ACTION_CASES:
            rows_by_case[case_id] = {
                "case_id": case_id,
                "trace_root": rel(root),
                "trace_path": "",
                "load_ok": False,
                "load_error": f"torch_import_failed:{exc}",
                "direct_masklet_payload_materialized": False,
                "direct_seed_payload_materialized": False,
                "strict_current_cache_component_candidate": False,
                "strict_current_cache_seed_candidate": False,
                "runtime_action_allowed": False,
            }
        return [rows_by_case[c] for c in SELECTED_ACTION_CASES], summary

    trace_paths = sorted(root.glob("*/READ_NO_ACTION/swa_raw_transport_trace/*.pt"))
    summary["trace_file_count"] = len(trace_paths)
    geometry_count_by_case = {
        case_dir.name: len(list((case_dir / "READ_NO_ACTION/per_chunk_geometry").glob("*.pt")))
        for case_dir in sorted(root.iterdir())
        if case_dir.is_dir()
    }
    summary["per_chunk_geometry_sidecar_file_count"] = sum(geometry_count_by_case.values())

    for trace_path in trace_paths:
        case_id = trace_path.parents[2].name
        if case_id not in SELECTED_ACTION_CASES:
            continue
        row: dict[str, Any] = {
            "case_id": case_id,
            "trace_root": rel(root),
            "trace_path": rel(trace_path),
            "per_chunk_geometry_sidecar_file_count": geometry_count_by_case.get(case_id, 0),
            "runtime_action_allowed": False,
        }
        try:
            payload = torch.load(trace_path, map_location="cpu", weights_only=False)
            summary["trace_loaded_count"] += 1
            row.update(
                {
                    "load_ok": True,
                    "load_error": "",
                    "schema": payload.get("schema", ""),
                    "artifact": payload.get("artifact", ""),
                    "diagnostic_only": payload.get("diagnostic_only", ""),
                    "chunk_idx": payload.get("chunk_idx", ""),
                    "sampled_query_count": payload.get("sampled_query_count", ""),
                    "tokens_per_frame": payload.get("tokens_per_frame", ""),
                    "current_tokens": payload.get("current_tokens", ""),
                    "history_tokens": payload.get("history_tokens", ""),
                    "topk_identity_available": payload.get("topk_identity_available", False),
                    "current_stage_c_masklet_instance_idx_trace_available": payload.get(
                        "current_stage_c_masklet_instance_idx_trace_available", False
                    ),
                    "cache_stage_c_masklet_instance_idx_trace_available": payload.get(
                        "cache_stage_c_masklet_instance_idx_trace_available", False
                    ),
                    "current_stage_c_seed_global_track_idx_trace_available": payload.get(
                        "current_stage_c_seed_global_track_idx_trace_available", False
                    ),
                    "cache_stage_c_seed_global_track_idx_trace_available": payload.get(
                        "cache_stage_c_seed_global_track_idx_trace_available", False
                    ),
                    "current_stage_c_masklet_instance_idx_nonnegative_count": payload.get(
                        "current_stage_c_masklet_instance_idx_nonnegative_count", 0
                    ),
                    "current_stage_c_masklet_instance_idx_unique_count": payload.get(
                        "current_stage_c_masklet_instance_idx_unique_count", 0
                    ),
                    "current_stage_c_seed_global_track_idx_nonnegative_count": payload.get(
                        "current_stage_c_seed_global_track_idx_nonnegative_count", 0
                    ),
                    "current_stage_c_seed_global_track_idx_unique_count": payload.get(
                        "current_stage_c_seed_global_track_idx_unique_count", 0
                    ),
                    "ttt_prev_tracked_instance_anchor_identity_available": payload.get(
                        "ttt_prev_tracked_instance_anchor_identity_available", False
                    ),
                    "ttt_prev_tracked_instance_anchor_lifecycle_row_count": payload.get(
                        "ttt_prev_tracked_instance_anchor_lifecycle_row_count", 0
                    ),
                    "ttt_prev_tracked_instance_anchor_source_token_count": payload.get(
                        "ttt_prev_tracked_instance_anchor_source_token_count", 0
                    ),
                    "ttt_prev_tracked_instance_anchor_topk_hit_frac_mean": payload.get(
                        "ttt_prev_tracked_instance_anchor_topk_hit_frac_mean", ""
                    ),
                    "ttt_prev_tracked_instance_anchor_topk_same_masklet_frac_mean": payload.get(
                        "ttt_prev_tracked_instance_anchor_topk_same_masklet_frac_mean", ""
                    ),
                    "ttt_prev_tracked_instance_anchor_topk_same_seed_frac_mean": payload.get(
                        "ttt_prev_tracked_instance_anchor_topk_same_seed_frac_mean", ""
                    ),
                }
            )

            query_masklet = tensor_nonnegative_stats(payload, "sampled_query_stage_c_masklet_instance_idx", torch)
            topk_masklet = tensor_nonnegative_stats(
                payload, "current_Q_to_cache_K_topk_cache_stage_c_masklet_instance_idx", torch
            )
            query_seed = tensor_nonnegative_stats(payload, "sampled_query_stage_c_seed_global_track_idx", torch)
            topk_seed = tensor_nonnegative_stats(
                payload, "current_Q_to_cache_K_topk_cache_stage_c_seed_global_track_idx", torch
            )
            same_masklet = tensor_bool_stats(payload, "current_Q_to_cache_K_topk_same_stage_c_masklet_instance_idx")
            same_seed = tensor_bool_stats(payload, "current_Q_to_cache_K_topk_same_stage_c_seed_global_track_idx")
            anchor_hit = tensor_bool_stats(payload, "current_Q_to_cache_K_topk_ttt_prev_tracked_instance_anchor_hit_mask")
            anchor_same_masklet = tensor_bool_stats(
                payload, "current_Q_to_cache_K_topk_ttt_prev_tracked_instance_anchor_same_masklet"
            )
            anchor_same_seed = tensor_bool_stats(
                payload, "current_Q_to_cache_K_topk_ttt_prev_tracked_instance_anchor_same_seed"
            )
            for stats in (
                query_masklet,
                topk_masklet,
                query_seed,
                topk_seed,
                same_masklet,
                same_seed,
                anchor_hit,
                anchor_same_masklet,
                anchor_same_seed,
            ):
                row.update(stats)

            direct_masklet = (
                query_masklet["sampled_query_stage_c_masklet_instance_idx_nonnegative_count"] > 0
                and topk_masklet["current_Q_to_cache_K_topk_cache_stage_c_masklet_instance_idx_nonnegative_count"] > 0
                and same_masklet["current_Q_to_cache_K_topk_same_stage_c_masklet_instance_idx_present"]
            )
            direct_seed = (
                query_seed["sampled_query_stage_c_seed_global_track_idx_nonnegative_count"] > 0
                and topk_seed["current_Q_to_cache_K_topk_cache_stage_c_seed_global_track_idx_nonnegative_count"] > 0
                and same_seed["current_Q_to_cache_K_topk_same_stage_c_seed_global_track_idx_present"]
            )
            strict_component_candidate = (
                direct_masklet
                and same_masklet["current_Q_to_cache_K_topk_same_stage_c_masklet_instance_idx_true_count"] > 0
            )
            strict_seed_candidate = (
                direct_seed
                and same_seed["current_Q_to_cache_K_topk_same_stage_c_seed_global_track_idx_true_count"] > 0
            )
            row.update(
                {
                    "direct_masklet_payload_materialized": direct_masklet,
                    "direct_seed_payload_materialized": direct_seed,
                    "strict_current_cache_component_candidate": strict_component_candidate,
                    "strict_current_cache_seed_candidate": strict_seed_candidate,
                }
            )

            if direct_masklet:
                summary["direct_masklet_payload_case_count"] += 1
            if direct_seed:
                summary["direct_seed_payload_case_count"] += 1
            if strict_component_candidate:
                summary["strict_current_cache_component_candidate_case_count"] += 1
            if strict_seed_candidate:
                summary["strict_current_cache_seed_candidate_case_count"] += 1
            if payload.get("ttt_prev_tracked_instance_anchor_identity_available", False):
                summary["ttt_prev_anchor_identity_available_case_count"] += 1
            summary["query_masklet_nonnegative_total"] += query_masklet[
                "sampled_query_stage_c_masklet_instance_idx_nonnegative_count"
            ]
            summary["topk_masklet_nonnegative_total"] += topk_masklet[
                "current_Q_to_cache_K_topk_cache_stage_c_masklet_instance_idx_nonnegative_count"
            ]
            summary["query_seed_nonnegative_total"] += query_seed[
                "sampled_query_stage_c_seed_global_track_idx_nonnegative_count"
            ]
            summary["topk_seed_nonnegative_total"] += topk_seed[
                "current_Q_to_cache_K_topk_cache_stage_c_seed_global_track_idx_nonnegative_count"
            ]
            summary["same_masklet_true_total"] += same_masklet[
                "current_Q_to_cache_K_topk_same_stage_c_masklet_instance_idx_true_count"
            ]
            summary["same_seed_true_total"] += same_seed[
                "current_Q_to_cache_K_topk_same_stage_c_seed_global_track_idx_true_count"
            ]
        except Exception as exc:
            summary["trace_load_failed_count"] += 1
            row.update(
                {
                    "load_ok": False,
                    "load_error": str(exc),
                    "direct_masklet_payload_materialized": False,
                    "direct_seed_payload_materialized": False,
                    "strict_current_cache_component_candidate": False,
                    "strict_current_cache_seed_candidate": False,
                }
            )
        rows_by_case[case_id] = row

    for case_id in SELECTED_ACTION_CASES:
        rows_by_case.setdefault(
            case_id,
            {
                "case_id": case_id,
                "trace_root": rel(root),
                "trace_path": "",
                "per_chunk_geometry_sidecar_file_count": geometry_count_by_case.get(case_id, 0),
                "load_ok": False,
                "load_error": "selected_case_trace_missing",
                "direct_masklet_payload_materialized": False,
                "direct_seed_payload_materialized": False,
                "strict_current_cache_component_candidate": False,
                "strict_current_cache_seed_candidate": False,
                "runtime_action_allowed": False,
            },
        )

    loaded_selected = sum(1 for case_id in SELECTED_ACTION_CASES if to_bool(rows_by_case[case_id].get("load_ok")))
    summary.update(
        {
            "selected_trace_loaded_count": loaded_selected,
            "direct_masklet_payload_frac": summary["direct_masklet_payload_case_count"] / len(SELECTED_ACTION_CASES),
            "direct_seed_payload_frac": summary["direct_seed_payload_case_count"] / len(SELECTED_ACTION_CASES),
            "strict_current_cache_component_candidate_frac": summary[
                "strict_current_cache_component_candidate_case_count"
            ]
            / len(SELECTED_ACTION_CASES),
            "strict_current_cache_seed_candidate_frac": summary["strict_current_cache_seed_candidate_case_count"]
            / len(SELECTED_ACTION_CASES),
        }
    )
    return [rows_by_case[c] for c in SELECTED_ACTION_CASES], summary


def audit_selected11_no_action_parity(root: Path, pose_parity_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pose_parity_rows = index_by_case(read_csv(pose_parity_root / "pose_sha_parity_rows.csv"))
    pose_parity_summary = read_json(pose_parity_root / "summary.json")
    rows: list[dict[str, Any]] = []
    for case_id in SELECTED_ACTION_CASES:
        run_dir = root / case_id / "READ_NO_ACTION"
        correctness = read_json(run_dir / "hmc_correctness_summary.json")
        hook = read_json(run_dir / "hmc_hook_identity_check.json")
        pose_row = pose_parity_rows.get(case_id, {})
        pose_sha_available = (
            bool(pose_row)
            and to_bool(pose_row.get("trace_output_exists", False))
            and to_bool(pose_row.get("baseline_output_exists", False))
        )
        hook_bias = correctness.get("hook_max_abs_bias", {})
        max_hook_bias = max((to_float(v) for v in hook_bias.values()), default=0.0) if isinstance(hook_bias, dict) else 0.0
        row = {
            "case_id": case_id,
            "correctness_summary_path": rel(run_dir / "hmc_correctness_summary.json"),
            "hook_identity_check_path": rel(run_dir / "hmc_hook_identity_check.json"),
            "correctness_summary_available": bool(correctness),
            "hook_identity_check_available": bool(hook),
            "hook_identity_status": hook.get("status", ""),
            "identity_ttt_update": hook.get("identity_ttt_update", ""),
            "max_hook_abs_bias": max_hook_bias,
            "max_pass1_pass2_pose_matrix_abs_max": correctness.get("max_pass1_pass2_pose_matrix_abs_max", ""),
            "max_pass1_pass2_pose_t_max": correctness.get("max_pass1_pass2_pose_t_max", ""),
            "probe_no_commit_hash_equal_all": correctness.get("probe_no_commit_hash_equal_all", ""),
            "state_double_write_safe_all": correctness.get("state_double_write_safe_all", ""),
            "no_trace_pose_sha_parity_row_source": rel(pose_parity_root / "pose_sha_parity_rows.csv")
            if pose_parity_rows
            else "",
            "pose_sha_equal_available": pose_sha_available,
            "pose_sha_equal": pose_row.get("pose_sha_equal", ""),
            "pose_numeric_equal": pose_row.get("pose_numeric_equal", ""),
            "pose_max_abs_diff": pose_row.get("pose_max_abs_diff", ""),
            "no_trace_pose_parity_pass": pose_row.get("parity_pass", ""),
            "no_trace_pose_failure_reason": pose_row.get("failure_reason", ""),
            "runtime_action_allowed": False,
        }
        row["observable_no_action_parity_pass"] = (
            bool(correctness)
            and bool(hook)
            and hook.get("status") == "available"
            and to_float(row["max_hook_abs_bias"]) == 0.0
            and to_float(row["max_pass1_pass2_pose_matrix_abs_max"]) == 0.0
            and to_float(row["max_pass1_pass2_pose_t_max"]) == 0.0
            and to_bool(row["probe_no_commit_hash_equal_all"])
            and to_bool(row["state_double_write_safe_all"])
        )
        row["plan_pose_sha_gate_pass"] = (
            row["observable_no_action_parity_pass"]
            and pose_sha_available
            and to_bool(row["pose_sha_equal"])
            and to_bool(row["pose_numeric_equal"])
            and to_bool(row["no_trace_pose_parity_pass"])
        )
        row["plan_pose_sha_gate_note"] = (
            "pose_sha_equal validated by trace-vs-no-trace READ_NO_ACTION baseline"
            if row["plan_pose_sha_gate_pass"]
            else "pose_sha_equal gate not validated for this case"
        )
        rows.append(row)

    observable_pass_count = sum(1 for row in rows if row["observable_no_action_parity_pass"])
    pose_sha_available_count = sum(1 for row in rows if row["pose_sha_equal_available"])
    pose_sha_equal_count = sum(1 for row in rows if to_bool(row["pose_sha_equal"]))
    pose_numeric_equal_count = sum(1 for row in rows if to_bool(row["pose_numeric_equal"]))
    no_trace_pose_pass_count = sum(1 for row in rows if to_bool(row["no_trace_pose_parity_pass"]))
    plan_pose_sha_gate_pass_count = sum(1 for row in rows if row["plan_pose_sha_gate_pass"])
    summary = {
        "schema": "acl2_v104_selected11_no_action_parity_summary_v1",
        "trace_root": rel(root),
        "no_trace_pose_sha_parity_root": rel(pose_parity_root),
        "no_trace_pose_sha_parity_summary": rel(pose_parity_root / "summary.json")
        if pose_parity_summary
        else "",
        "selected_case_count": len(SELECTED_ACTION_CASES),
        "observable_no_action_parity_pass_case_count": observable_pass_count,
        "observable_no_action_parity_pass": observable_pass_count == len(SELECTED_ACTION_CASES),
        "max_hook_abs_bias_max": max((to_float(row["max_hook_abs_bias"]) for row in rows), default=0.0),
        "max_pass1_pass2_pose_matrix_abs_max": max(
            (to_float(row["max_pass1_pass2_pose_matrix_abs_max"]) for row in rows), default=0.0
        ),
        "max_pass1_pass2_pose_t_max": max(
            (to_float(row["max_pass1_pass2_pose_t_max"]) for row in rows), default=0.0
        ),
        "probe_no_commit_hash_equal_all_case_count": sum(
            1 for row in rows if to_bool(row["probe_no_commit_hash_equal_all"])
        ),
        "state_double_write_safe_all_case_count": sum(1 for row in rows if to_bool(row["state_double_write_safe_all"])),
        "pose_sha_equal_available_case_count": pose_sha_available_count,
        "pose_sha_equal_case_count": pose_sha_equal_count,
        "pose_numeric_equal_case_count": pose_numeric_equal_count,
        "no_trace_pose_parity_pass_case_count": no_trace_pose_pass_count,
        "no_trace_pose_sha_parity_pass": pose_parity_summary.get("parity_pass", False),
        "no_trace_pose_sha_parity_failed_job_count": pose_parity_summary.get("failed_job_count", ""),
        "plan_pose_sha_gate_pass_case_count": plan_pose_sha_gate_pass_count,
        "plan_pose_sha_gate_pass": plan_pose_sha_gate_pass_count == len(SELECTED_ACTION_CASES),
        "runtime_action_allowed": False,
        "interpretation": (
            "Observed no-action parity fields pass for all selected cases. The explicit pose SHA/numeric gate is "
            "validated by the trace-vs-no-trace READ_NO_ACTION baseline, so the trace instrumentation is recorded "
            "as pose-output neutral. This does not authorize Stage4 action because the strict provider join still fails."
        ),
    }
    return rows, summary


def audit_a5_query_head_local_trace_repair_attempt(
    trace_root: Path, pose_parity_root: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Audit the A5 same-seed query/head-local trace-only carrier branch."""

    trace_summary = read_json(trace_root / "summary.json")
    pose_parity_summary = read_json(pose_parity_root / "summary.json")
    rows: list[dict[str, Any]] = []
    summary = {
        "schema": "acl2_v104_a5_query_head_local_trace_repair_attempt_summary_v1",
        "trace_root": rel(trace_root),
        "trace_summary": rel(trace_root / "summary.json") if trace_summary else "",
        "no_trace_pose_sha_parity_root": rel(pose_parity_root),
        "no_trace_pose_sha_parity_summary": rel(pose_parity_root / "summary.json")
        if pose_parity_summary
        else "",
        "selected_case_count": len(SELECTED_ACTION_CASES),
        "runtime_action_allowed": False,
        "provider_pass_after_repair_attempt": False,
        "first_blocker": "trace_root_missing",
    }
    if not trace_root.exists():
        return rows, [], summary

    for case_id in SELECTED_ACTION_CASES:
        run_dir = trace_root / case_id / "READ_NO_ACTION"
        hook_path = run_dir / "hook_effect_summary.jsonl"
        row: dict[str, Any] = {
            "case_id": case_id,
            "case_role": classify_case(case_id),
            "selected_action_case": True,
            "focused_safe_good_control": case_id in SAFE_GOOD_CONTROLS,
            "focused_swa_bad_strict_candidate": case_id in STRICT_POSITIVES,
            "exploratory_positive": case_id in EXPLORATORY_POSITIVES,
            "hook_effect_summary_path": rel(hook_path),
            "hook_effect_summary_available": hook_path.exists(),
            "hook_rows": 0,
            "swa_read_rows": 0,
            "sum_swa_read_trace": 0,
            "sum_swa_read_carrier_rule_pass": 0,
            "max_swa_read_source_tokens": 0,
            "max_swa_read_selected_queries": 0,
            "max_swa_read_direct_witness_seed_count": 0,
            "max_swa_read_direct_witness_hit_count": 0,
            "max_swa_read_query_selected_frac": 0.0,
            "mean_swa_read_selected_frac_max": 0.0,
            "attention_trace_sum": 0,
            "attention_carrier_sum": 0,
            "ttt_apply_trace_sum": 0,
            "ttt_apply_carrier_sum": 0,
            "direct_match_modes": set(),
            "formal_strict_current_support_pass": False,
            "action_ready_query_head_local_edge": False,
            "runtime_action_allowed": False,
            "claim_level": "a5_query_head_local_trace_only_no_runtime",
        }
        if hook_path.exists():
            for line in hook_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except Exception as exc:
                    row["hook_parse_error"] = f"{type(exc).__name__}:{exc}"
                    continue
                row["hook_rows"] += 1
                hook_summary = obj.get("hook_effect_summary") or {}
                for key in ("chunk_attention", "frame_attention"):
                    hook_block = hook_summary.get(key) or {}
                    row["attention_trace_sum"] += to_int(
                        hook_block.get("num_swa_prev_ttt_tracked_instance_query_soft_trace", 0)
                    )
                    row["attention_carrier_sum"] += to_int(
                        hook_block.get("num_swa_prev_ttt_tracked_instance_query_soft_carrier_rule_pass", 0)
                    )
                hook_block = hook_summary.get("ttt_apply") or {}
                row["ttt_apply_trace_sum"] += to_int(
                    hook_block.get("num_swa_prev_ttt_tracked_instance_query_soft_trace", 0)
                )
                row["ttt_apply_carrier_sum"] += to_int(
                    hook_block.get("num_swa_prev_ttt_tracked_instance_query_soft_carrier_rule_pass", 0)
                )
                hook_block = hook_summary.get("swa_read") or {}
                if hook_block:
                    row["swa_read_rows"] += 1
                    row["sum_swa_read_trace"] += to_int(
                        hook_block.get("num_swa_prev_ttt_tracked_instance_query_soft_trace", 0)
                    )
                    row["sum_swa_read_carrier_rule_pass"] += to_int(
                        hook_block.get("num_swa_prev_ttt_tracked_instance_query_soft_carrier_rule_pass", 0)
                    )
                    row["max_swa_read_source_tokens"] = max(
                        row["max_swa_read_source_tokens"],
                        to_int(
                            hook_block.get(
                                "max_swa_prev_ttt_tracked_instance_query_soft_source_tokens",
                                hook_block.get("max_prev_ttt_tracked_instance_query_soft_source_tokens", 0),
                            )
                        ),
                    )
                    row["max_swa_read_selected_queries"] = max(
                        row["max_swa_read_selected_queries"],
                        to_int(
                            hook_block.get(
                                "max_swa_prev_ttt_tracked_instance_query_soft_selected_queries",
                                hook_block.get("max_prev_ttt_tracked_instance_query_soft_selected_query_count", 0),
                            )
                        ),
                    )
                    row["max_swa_read_direct_witness_seed_count"] = max(
                        row["max_swa_read_direct_witness_seed_count"],
                        to_int(
                            hook_block.get(
                                "max_swa_prev_ttt_tracked_instance_query_soft_direct_witness_seed_count",
                                hook_block.get(
                                    "max_prev_ttt_tracked_instance_query_soft_direct_witness_seed_count", 0
                                ),
                            )
                        ),
                    )
                    row["max_swa_read_direct_witness_hit_count"] = max(
                        row["max_swa_read_direct_witness_hit_count"],
                        to_int(hook_block.get("max_prev_ttt_tracked_instance_query_soft_direct_witness_hit_count", 0)),
                    )
                    row["max_swa_read_query_selected_frac"] = max(
                        row["max_swa_read_query_selected_frac"],
                        to_float(hook_block.get("max_prev_ttt_tracked_instance_query_soft_query_selected_frac", 0.0)),
                    )
                    row["mean_swa_read_selected_frac_max"] = max(
                        row["mean_swa_read_selected_frac_max"],
                        to_float(hook_block.get("mean_swa_prev_ttt_tracked_instance_query_soft_selected_frac", 0.0)),
                    )
                    for mode in hook_block.get("values_prev_ttt_tracked_instance_query_soft_direct_match_mode") or []:
                        row["direct_match_modes"].add(str(mode))
        row["direct_match_modes"] = sorted(row["direct_match_modes"])
        row["swa_read_trace_available"] = to_int(row.get("sum_swa_read_trace", 0)) > 0
        row["swa_read_carrier_rule_pass"] = to_int(row.get("sum_swa_read_carrier_rule_pass", 0)) > 0
        row["attention_or_ttt_apply_trace_available"] = (
            to_int(row.get("attention_trace_sum", 0)) > 0 or to_int(row.get("ttt_apply_trace_sum", 0)) > 0
        )
        row["attention_or_ttt_apply_carrier_available"] = (
            to_int(row.get("attention_carrier_sum", 0)) > 0 or to_int(row.get("ttt_apply_carrier_sum", 0)) > 0
        )
        if not row["hook_effect_summary_available"]:
            row["first_blocker"] = "hook_effect_summary_missing"
        elif row["focused_safe_good_control"] and row["swa_read_carrier_rule_pass"]:
            row["first_blocker"] = "a5_swa_read_carrier_hits_safe_good_control"
        elif not row["formal_strict_current_support_pass"]:
            row["first_blocker"] = "a5_trace_only_not_formal_strict_current_support"
        elif not row["action_ready_query_head_local_edge"]:
            row["first_blocker"] = "a5_trace_only_not_action_ready_query_head_local_edge"
        else:
            row["first_blocker"] = "runtime_action_still_forbidden_by_policy"
        rows.append(row)

    def _rule_row(rule_name: str, description: str, rule_rows: list[dict[str, Any]]) -> dict[str, Any]:
        selected_cases = sorted({str(row.get("case_id", "")) for row in rule_rows if row.get("case_id")})
        positive_cases = sorted(
            case_id for case_id in selected_cases if case_id in STRICT_POSITIVES or case_id in EXPLORATORY_POSITIVES
        )
        safe_good_cases = sorted(case_id for case_id in selected_cases if case_id in SAFE_GOOD_CONTROLS)
        formal_cases = sorted(
            {
                str(row.get("case_id", ""))
                for row in rule_rows
                if to_bool(row.get("formal_strict_current_support_pass", False))
            }
        )
        action_cases = sorted(
            {
                str(row.get("case_id", ""))
                for row in rule_rows
                if to_bool(row.get("action_ready_query_head_local_edge", False))
            }
        )
        selected_coverage = len(selected_cases) / len(SELECTED_ACTION_CASES) if SELECTED_ACTION_CASES else 0.0
        if not rule_rows:
            first_blocker = "no_a5_candidate_rows_for_rule"
        elif safe_good_cases:
            first_blocker = "safe_good_control_hit_by_a5_query_head_local_trace_rule"
        elif not formal_cases:
            first_blocker = "formal_nonproxy_strict_current_support_absent"
        elif not action_cases:
            first_blocker = "action_ready_query_head_local_edge_absent"
        elif selected_coverage < 0.8:
            first_blocker = "selected_action_case_coverage_below_stage1_gate"
        elif len(positive_cases) < 3:
            first_blocker = "positive_support_too_narrow_for_action_entry"
        else:
            first_blocker = ""
        repair_allowed = (
            not first_blocker
            and selected_coverage >= 0.8
            and not safe_good_cases
            and bool(formal_cases)
            and bool(action_cases)
        )
        return {
            "repair_rule": rule_name,
            "description": description,
            "candidate_row_count": len(rule_rows),
            "selected_case_count": len(selected_cases),
            "selected_case_ids": selected_cases,
            "selected_action_case_coverage": selected_coverage,
            "positive_case_count": len(positive_cases),
            "positive_case_ids": positive_cases,
            "safe_good_hit_count": len(safe_good_cases),
            "safe_good_case_ids": safe_good_cases,
            "formal_strict_current_support_case_count": len(formal_cases),
            "formal_strict_current_support_case_ids": formal_cases,
            "action_ready_query_head_local_case_count": len(action_cases),
            "action_ready_query_head_local_case_ids": action_cases,
            "repair_allowed_for_action": repair_allowed,
            "first_blocker": first_blocker,
            "runtime_action_allowed": False,
        }

    carrier_rows = [row for row in rows if to_bool(row.get("swa_read_carrier_rule_pass", False))]
    trace_rows = [row for row in rows if to_bool(row.get("swa_read_trace_available", False))]
    no_safe_good_carrier_rows = [
        row
        for row in carrier_rows
        if not to_bool(row.get("focused_safe_good_control", False))
    ]
    action_surface_rows = [
        row for row in rows if to_bool(row.get("attention_or_ttt_apply_carrier_available", False))
    ]
    rule_rows = [
        _rule_row(
            "promote_a5_same_seed_swa_read_carrier_rule",
            "Promote same-seed A5 swa_read carrier-rule-pass cases.",
            carrier_rows,
        ),
        _rule_row(
            "promote_a5_same_seed_trace_any",
            "Promote any selected case with same-seed A5 swa_read trace materialized.",
            trace_rows,
        ),
        _rule_row(
            "promote_a5_same_seed_carrier_without_safe_good",
            "Promote A5 carrier rows after excluding selected safe-good controls.",
            no_safe_good_carrier_rows,
        ),
        _rule_row(
            "promote_a5_attention_or_ttt_apply_carrier",
            "Promote only if A5 carrier reaches attention/TTT apply action surfaces.",
            action_surface_rows,
        ),
    ]
    allowed_rules = [row for row in rule_rows if to_bool(row.get("repair_allowed_for_action", False))]
    carrier_case_ids = sorted(row["case_id"] for row in carrier_rows)
    trace_case_ids = sorted(row["case_id"] for row in trace_rows)
    attention_action_case_ids = sorted(
        row["case_id"] for row in rows if to_bool(row.get("attention_or_ttt_apply_trace_available", False))
    )
    action_carrier_case_ids = sorted(row["case_id"] for row in action_surface_rows)
    summary.update(
        {
            "trace_status": trace_summary.get("status", ""),
            "trace_completed_job_count": trace_summary.get("completed_job_count", ""),
            "trace_failed_job_count": trace_summary.get("failed_job_count", ""),
            "trace_payload_file_count": trace_summary.get("trace_payload_file_count", ""),
            "trace_runtime_action_allowed": trace_summary.get("runtime_action_allowed", False),
            "trace_gate_pass": trace_summary.get("gate_pass", False),
            "trace_gate_pass_note": trace_summary.get("gate_pass_note", ""),
            "case_row_count": len(rows),
            "case_with_swa_read_trace_count": len(trace_case_ids),
            "case_with_swa_read_trace_ids": trace_case_ids,
            "case_with_swa_read_carrier_rule_pass_count": len(carrier_case_ids),
            "case_with_swa_read_carrier_rule_pass_ids": carrier_case_ids,
            "safe_good_swa_read_carrier_rule_pass_count": sum(
                1 for row in carrier_rows if to_bool(row.get("focused_safe_good_control", False))
            ),
            "safe_good_swa_read_carrier_rule_pass_ids": sorted(
                row["case_id"] for row in carrier_rows if to_bool(row.get("focused_safe_good_control", False))
            ),
            "strict_positive_swa_read_carrier_rule_pass_count": sum(
                1 for row in carrier_rows if row.get("case_id") in STRICT_POSITIVES
            ),
            "positive_swa_read_carrier_rule_pass_count": sum(
                1
                for row in carrier_rows
                if row.get("case_id") in STRICT_POSITIVES or row.get("case_id") in EXPLORATORY_POSITIVES
            ),
            "attention_or_ttt_apply_trace_case_count": len(attention_action_case_ids),
            "attention_or_ttt_apply_trace_case_ids": attention_action_case_ids,
            "attention_or_ttt_apply_carrier_case_count": len(action_carrier_case_ids),
            "attention_or_ttt_apply_carrier_case_ids": action_carrier_case_ids,
            "formal_strict_current_support_pass_case_count": sum(
                1 for row in rows if to_bool(row.get("formal_strict_current_support_pass", False))
            ),
            "action_ready_query_head_local_case_count": sum(
                1 for row in rows if to_bool(row.get("action_ready_query_head_local_edge", False))
            ),
            "max_swa_read_source_tokens": max(
                (to_int(row.get("max_swa_read_source_tokens", 0)) for row in rows), default=0
            ),
            "max_swa_read_selected_queries": max(
                (to_int(row.get("max_swa_read_selected_queries", 0)) for row in rows), default=0
            ),
            "max_swa_read_direct_witness_seed_count": max(
                (to_int(row.get("max_swa_read_direct_witness_seed_count", 0)) for row in rows), default=0
            ),
            "max_swa_read_query_selected_frac": max(
                (to_float(row.get("max_swa_read_query_selected_frac", 0.0)) for row in rows), default=0.0
            ),
            "direct_match_modes": sorted(
                {
                    mode
                    for row in rows
                    for mode in (row.get("direct_match_modes", []) if isinstance(row.get("direct_match_modes"), list) else [])
                }
            ),
            "no_trace_pose_sha_parity_pass": pose_parity_summary.get("parity_pass", False),
            "pose_sha_equal_case_count": pose_parity_summary.get("pose_sha_equal_case_count", ""),
            "pose_numeric_equal_case_count": pose_parity_summary.get("pose_numeric_equal_case_count", ""),
            "pose_parity_failed_job_count": pose_parity_summary.get("failed_job_count", ""),
            "repair_rule_count": len(rule_rows),
            "repair_allowed_rule_count": len(allowed_rules),
            "repair_allowed_rule_names": [row["repair_rule"] for row in allowed_rules],
            "first_blocker": (
                "a5_query_head_local_trace_remains_trace_only_and_hits_safe_good_controls"
                if not allowed_rules
                else ""
            ),
            "interpretation": (
                "The A5 same-seed query/head-local trace branch materializes swa_read trace in all selected cases "
                "and carrier-rule-pass in most selected cases, with pose SHA/numeric parity against no-trace "
                "READ_NO_ACTION baselines. It still cannot be promoted: the carrier hits selected safe-good controls, "
                "does not reach attention/TTT-apply action surfaces, and remains trace-only rather than a formal "
                "strict current-support provider."
            ),
        }
    )
    return rows, rule_rows, summary


def audit_a5_direct_match_threshold_repair_attempt(
    mode_rows: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Sweep A5 direct-match trace features as no-action repair candidates."""

    threshold_rows: list[dict[str, Any]] = []
    feature_specs = [
        ("direct_witness_seed_count", "max_swa_read_direct_witness_seed_count", "ge"),
        ("selected_query_count", "max_swa_read_selected_queries", "ge"),
        ("query_selected_frac", "max_swa_read_query_selected_frac", "ge"),
    ]

    for mode, rows in sorted(mode_rows.items()):
        carrier_rows = [row for row in rows if to_bool(row.get("swa_read_carrier_rule_pass", False))]
        for feature_name, row_key, comparator in feature_specs:
            values = sorted({to_float(row.get(row_key, 0.0)) for row in carrier_rows if to_float(row.get(row_key, 0.0)) > 0.0})
            for threshold in values:
                selected_rows = [row for row in carrier_rows if to_float(row.get(row_key, 0.0)) >= threshold]
                selected_cases = sorted({str(row.get("case_id", "")) for row in selected_rows if row.get("case_id")})
                positive_cases = sorted(
                    case_id
                    for case_id in selected_cases
                    if case_id in STRICT_POSITIVES or case_id in EXPLORATORY_POSITIVES
                )
                safe_good_cases = sorted(case_id for case_id in selected_cases if case_id in SAFE_GOOD_CONTROLS)
                formal_cases = sorted(
                    {
                        str(row.get("case_id", ""))
                        for row in selected_rows
                        if to_bool(row.get("formal_strict_current_support_pass", False))
                    }
                )
                action_cases = sorted(
                    {
                        str(row.get("case_id", ""))
                        for row in selected_rows
                        if to_bool(row.get("action_ready_query_head_local_edge", False))
                    }
                )
                selected_coverage = len(selected_cases) / len(SELECTED_ACTION_CASES) if SELECTED_ACTION_CASES else 0.0
                if safe_good_cases:
                    first_blocker = "safe_good_control_hit_by_a5_threshold_rule"
                elif not formal_cases:
                    first_blocker = "formal_nonproxy_strict_current_support_absent"
                elif not action_cases:
                    first_blocker = "action_ready_query_head_local_edge_absent"
                elif selected_coverage < 0.8:
                    first_blocker = "selected_action_case_coverage_below_stage1_gate"
                elif len(positive_cases) < 3:
                    first_blocker = "positive_support_too_narrow_for_action_entry"
                else:
                    first_blocker = ""
                repair_allowed = (
                    not first_blocker
                    and selected_coverage >= 0.8
                    and not safe_good_cases
                    and bool(formal_cases)
                    and bool(action_cases)
                )
                threshold_rows.append(
                    {
                        "direct_match_mode": mode,
                        "feature_name": feature_name,
                        "row_key": row_key,
                        "comparator": comparator,
                        "threshold": threshold,
                        "candidate_row_count": len(selected_rows),
                        "selected_case_count": len(selected_cases),
                        "selected_case_ids": selected_cases,
                        "selected_action_case_coverage": selected_coverage,
                        "positive_case_count": len(positive_cases),
                        "positive_case_ids": positive_cases,
                        "safe_good_hit_count": len(safe_good_cases),
                        "safe_good_case_ids": safe_good_cases,
                        "formal_strict_current_support_case_count": len(formal_cases),
                        "formal_strict_current_support_case_ids": formal_cases,
                        "action_ready_query_head_local_case_count": len(action_cases),
                        "action_ready_query_head_local_case_ids": action_cases,
                        "repair_allowed_for_action": repair_allowed,
                        "first_blocker": first_blocker,
                        "runtime_action_allowed": False,
                    }
                )

    allowed_rows = [row for row in threshold_rows if to_bool(row.get("repair_allowed_for_action", False))]
    no_safe_good_rows = [row for row in threshold_rows if to_int(row.get("safe_good_hit_count", 0)) == 0]
    no_safe_good_rows = sorted(
        no_safe_good_rows,
        key=lambda row: (
            to_int(row.get("positive_case_count", 0)),
            to_float(row.get("selected_action_case_coverage", 0.0)),
            -to_float(row.get("threshold", 0.0)),
        ),
        reverse=True,
    )
    best_no_safe_good = no_safe_good_rows[0] if no_safe_good_rows else {}
    summary = {
        "schema": "acl2_v104_a5_direct_match_threshold_repair_attempt_summary_v1",
        "mode_count": len(mode_rows),
        "mode_names": sorted(mode_rows),
        "feature_count": len(feature_specs),
        "threshold_rule_count": len(threshold_rows),
        "no_safe_good_threshold_rule_count": len(no_safe_good_rows),
        "repair_allowed_rule_count": len(allowed_rows),
        "repair_allowed_rule_names": [
            f"{row.get('direct_match_mode')}:{row.get('feature_name')}>={row.get('threshold')}"
            for row in allowed_rows
        ],
        "best_no_safe_good_rule": {
            "direct_match_mode": best_no_safe_good.get("direct_match_mode", ""),
            "feature_name": best_no_safe_good.get("feature_name", ""),
            "threshold": best_no_safe_good.get("threshold", ""),
            "selected_case_count": best_no_safe_good.get("selected_case_count", 0),
            "selected_case_ids": best_no_safe_good.get("selected_case_ids", []),
            "positive_case_count": best_no_safe_good.get("positive_case_count", 0),
            "positive_case_ids": best_no_safe_good.get("positive_case_ids", []),
            "first_blocker": best_no_safe_good.get("first_blocker", ""),
        },
        "provider_pass_after_repair_attempt": False,
        "runtime_action_allowed": False,
        "first_blocker": (
            "a5_threshold_candidates_without_safe_good_remain_nonformal_or_too_narrow"
            if no_safe_good_rows and not allowed_rows
            else "no_a5_threshold_rule_avoids_safe_good_controls"
            if not allowed_rows
            else ""
        ),
        "interpretation": (
            "A5 direct-match trace feature thresholds can create no-safe-good diagnostic candidates, but the best "
            "available candidates are still trace-only and lack formal strict current-support plus action-ready "
            "query/head-local edges. Thresholding alone does not repair Stage 1."
        ),
    }
    return threshold_rows, summary


def audit_a5_sampled_exact_edge_materialization(
    mode_roots: dict[str, Path],
    *,
    sample_label: str = "q128",
    selected_case_scope: str = "selected11",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Materialize sampled A5 direct-match edges from trace payload tensors."""

    edge_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    summary = {
        "schema": "acl2_v104_a5_sampled_exact_edge_materialization_summary_v1",
        "sample_label": sample_label,
        "selected_case_scope": selected_case_scope,
        "mode_roots": {mode: rel(root) for mode, root in sorted(mode_roots.items())},
        "mode_count": len(mode_roots),
        "trace_file_count": 0,
        "trace_loaded_count": 0,
        "trace_load_failed_count": 0,
        "edge_row_count": 0,
        "sampled_exact_nonproxy_current_support_edge_count": 0,
        "case_with_sampled_exact_nonproxy_current_support_edge_count": 0,
        "safe_good_with_sampled_exact_nonproxy_current_support_edge_count": 0,
        "formal_fullquery_strict_current_support_pass_case_count": 0,
        "action_ready_query_head_local_case_count": 0,
        "provider_pass_after_repair_attempt": False,
        "runtime_action_allowed": False,
        "first_blocker": "trace_root_missing",
    }
    existing_roots = {mode: root for mode, root in mode_roots.items() if root.exists()}
    if not existing_roots:
        return edge_rows, case_rows, summary

    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment diagnostic
        summary["torch_import_error"] = str(exc)
        summary["first_blocker"] = "torch_import_failed"
        return edge_rows, case_rows, summary

    nonproxy_case_keys: set[tuple[str, str]] = set()
    safe_good_nonproxy_case_keys: set[tuple[str, str]] = set()

    for mode, root in sorted(existing_roots.items()):
        trace_paths = sorted(root.glob("*/READ_NO_ACTION/swa_raw_transport_trace/*.pt"))
        summary["trace_file_count"] += len(trace_paths)
        for trace_path in trace_paths:
            case_id = trace_path.parents[2].name
            if case_id not in SELECTED_ACTION_CASES:
                continue
            case_row: dict[str, Any] = {
                "direct_match_mode": mode,
                "case_id": case_id,
                "case_role": classify_case(case_id),
                "selected_action_case": True,
                "focused_safe_good_control": case_id in SAFE_GOOD_CONTROLS,
                "focused_swa_bad_strict_candidate": case_id in STRICT_POSITIVES,
                "trace_path": rel(trace_path),
                "load_ok": False,
                "direct_match_edge_count": 0,
                "sampled_exact_nonproxy_current_support_edge_count": 0,
                "query_head_count": 0,
                "selected_query_count": 0,
                "anchor_id_count": 0,
                "anchor_seed_count": 0,
                "formal_fullquery_strict_current_support_pass": False,
                "action_ready_query_head_local_edge": False,
                "runtime_action_allowed": False,
                "claim_level": f"a5_sampled_{sample_label}_exact_edge_materialization_no_runtime",
                "first_blocker": f"sampled_{sample_label}_exact_edges_are_not_fullquery_formal_provider",
            }
            try:
                payload = torch.load(trace_path, map_location="cpu", weights_only=False)
                summary["trace_loaded_count"] += 1
                q_seed = payload["sampled_query_stage_c_seed_global_track_idx"].detach().cpu().long()
                q_inst = payload["sampled_query_stage_c_masklet_instance_idx"].detach().cpu().long()
                k_seed = payload["current_Q_to_cache_K_topk_cache_stage_c_seed_global_track_idx"].detach().cpu().long()
                k_inst = payload["current_Q_to_cache_K_topk_cache_stage_c_masklet_instance_idx"].detach().cpu().long()
                anchor_hit = payload["current_Q_to_cache_K_topk_ttt_prev_tracked_instance_anchor_hit_mask"].detach().cpu().bool()
                anchor_same_seed = payload["current_Q_to_cache_K_topk_ttt_prev_tracked_instance_anchor_same_seed"].detach().cpu().bool()
                anchor_same_masklet = payload[
                    "current_Q_to_cache_K_topk_ttt_prev_tracked_instance_anchor_same_masklet"
                ].detach().cpu().bool()
                direct_mask = anchor_hit & (anchor_same_masklet if mode == "same_masklet" else anchor_same_seed)
                q_seed_expanded = q_seed[:, None, :, None].expand_as(k_seed)
                q_inst_expanded = q_inst[:, None, :, None].expand_as(k_inst)
                current_cache_same_seed = (q_seed_expanded == k_seed) & (q_seed_expanded >= 0) & (k_seed >= 0)
                current_cache_same_masklet = (
                    (q_inst_expanded == k_inst) & (q_inst_expanded >= 0) & (k_inst >= 0)
                )
                nonproxy_current_support = direct_mask & current_cache_same_seed & current_cache_same_masklet
                anchor_ids = payload.get("current_Q_to_cache_K_topk_ttt_prev_tracked_instance_anchor_ids")
                if torch.is_tensor(anchor_ids):
                    anchor_ids = anchor_ids.detach().cpu().long()
                anchor_seeds = payload.get("current_Q_to_cache_K_topk_ttt_prev_tracked_instance_anchor_seeds")
                if torch.is_tensor(anchor_seeds):
                    anchor_seeds = anchor_seeds.detach().cpu().long()
                cache_indices = payload.get("current_Q_to_cache_K_topk_cache_indices")
                if torch.is_tensor(cache_indices):
                    cache_indices = cache_indices.detach().cpu().long()
                cache_frames = payload.get("current_Q_to_cache_K_topk_cache_frames")
                if torch.is_tensor(cache_frames):
                    cache_frames = cache_frames.detach().cpu().long()
                cache_scores = payload.get("current_Q_to_cache_K_topk_scores")
                if torch.is_tensor(cache_scores):
                    cache_scores = cache_scores.detach().cpu()
                sampled_query_indices = payload.get("sampled_query_indices")
                if torch.is_tensor(sampled_query_indices):
                    sampled_query_indices = sampled_query_indices.detach().cpu().long()

                nonzero = torch.nonzero(direct_mask, as_tuple=False)
                nonproxy_nonzero = torch.nonzero(nonproxy_current_support, as_tuple=False)
                case_row.update(
                    {
                        "load_ok": True,
                        "schema": payload.get("schema", ""),
                        "sampled_query_count": payload.get("sampled_query_count", ""),
                        "head_count": payload.get("head_count", ""),
                        "direct_match_edge_count": int(nonzero.shape[0]),
                        "sampled_exact_nonproxy_current_support_edge_count": int(nonproxy_nonzero.shape[0]),
                        "query_head_count": int(torch.unique(nonzero[:, 1]).numel()) if int(nonzero.shape[0]) else 0,
                        "selected_query_count": int(torch.unique(nonzero[:, 2]).numel()) if int(nonzero.shape[0]) else 0,
                    }
                )
                if int(nonproxy_nonzero.shape[0]) > 0:
                    nonproxy_case_keys.add((mode, case_id))
                    if case_id in SAFE_GOOD_CONTROLS:
                        safe_good_nonproxy_case_keys.add((mode, case_id))
                anchor_id_values = anchor_ids[direct_mask] if torch.is_tensor(anchor_ids) and int(nonzero.shape[0]) else None
                anchor_seed_values = (
                    anchor_seeds[direct_mask] if torch.is_tensor(anchor_seeds) and int(nonzero.shape[0]) else None
                )
                case_row["anchor_id_count"] = (
                    int(torch.unique(anchor_id_values[anchor_id_values >= 0]).numel())
                    if torch.is_tensor(anchor_id_values) and int((anchor_id_values >= 0).sum().item()) > 0
                    else 0
                )
                case_row["anchor_seed_count"] = (
                    int(torch.unique(anchor_seed_values[anchor_seed_values >= 0]).numel())
                    if torch.is_tensor(anchor_seed_values) and int((anchor_seed_values >= 0).sum().item()) > 0
                    else 0
                )
                for cell in nonzero.tolist():
                    batch_idx, head_idx, query_offset, topk_rank = [int(x) for x in cell]
                    query_token_index = (
                        int(sampled_query_indices[query_offset].item())
                        if torch.is_tensor(sampled_query_indices) and sampled_query_indices.ndim == 1
                        else query_offset
                    )
                    row = {
                        "direct_match_mode": mode,
                        "case_id": case_id,
                        "case_role": classify_case(case_id),
                        "selected_action_case": True,
                        "focused_safe_good_control": case_id in SAFE_GOOD_CONTROLS,
                        "focused_swa_bad_strict_candidate": case_id in STRICT_POSITIVES,
                        "trace_path": rel(trace_path),
                        "query_head": head_idx,
                        "query_offset": query_offset,
                        "query_token_index": query_token_index,
                        "topk_rank": topk_rank,
                        "query_stage_c_seed_global_track_idx": int(
                            q_seed_expanded[batch_idx, head_idx, query_offset, topk_rank].item()
                        ),
                        "cache_stage_c_seed_global_track_idx": int(
                            k_seed[batch_idx, head_idx, query_offset, topk_rank].item()
                        ),
                        "query_stage_c_masklet_instance_idx": int(
                            q_inst_expanded[batch_idx, head_idx, query_offset, topk_rank].item()
                        ),
                        "cache_stage_c_masklet_instance_idx": int(
                            k_inst[batch_idx, head_idx, query_offset, topk_rank].item()
                        ),
                        "current_cache_same_seed": bool(
                            current_cache_same_seed[batch_idx, head_idx, query_offset, topk_rank].item()
                        ),
                        "current_cache_same_masklet": bool(
                            current_cache_same_masklet[batch_idx, head_idx, query_offset, topk_rank].item()
                        ),
                        "anchor_hit": bool(anchor_hit[batch_idx, head_idx, query_offset, topk_rank].item()),
                        "anchor_same_seed": bool(anchor_same_seed[batch_idx, head_idx, query_offset, topk_rank].item()),
                        "anchor_same_masklet": bool(
                            anchor_same_masklet[batch_idx, head_idx, query_offset, topk_rank].item()
                        ),
                        "sampled_exact_nonproxy_current_support_edge": bool(
                            nonproxy_current_support[batch_idx, head_idx, query_offset, topk_rank].item()
                        ),
                        "formal_fullquery_strict_current_support_pass": False,
                        "action_ready_query_head_local_edge": False,
                        "runtime_action_allowed": False,
                        "claim_level": f"a5_sampled_{sample_label}_exact_edge_no_runtime",
                    }
                    if torch.is_tensor(anchor_ids):
                        row["anchor_id"] = int(anchor_ids[batch_idx, head_idx, query_offset, topk_rank].item())
                    if torch.is_tensor(anchor_seeds):
                        row["anchor_seed"] = int(anchor_seeds[batch_idx, head_idx, query_offset, topk_rank].item())
                    if torch.is_tensor(cache_indices):
                        row["cache_token_index"] = int(cache_indices[batch_idx, head_idx, query_offset, topk_rank].item())
                    if torch.is_tensor(cache_frames):
                        row["cache_frame"] = int(cache_frames[batch_idx, head_idx, query_offset, topk_rank].item())
                    if torch.is_tensor(cache_scores):
                        row["cache_score"] = float(cache_scores[batch_idx, head_idx, query_offset, topk_rank].item())
                    edge_rows.append(row)
            except Exception as exc:
                summary["trace_load_failed_count"] += 1
                case_row.update(
                    {
                        "load_error": f"{type(exc).__name__}:{exc}",
                        "first_blocker": "a5_sampled_exact_edge_payload_load_failed",
                    }
                )
            case_rows.append(case_row)

    summary.update(
        {
            "edge_row_count": len(edge_rows),
            "sampled_exact_nonproxy_current_support_edge_count": sum(
                1 for row in edge_rows if to_bool(row.get("sampled_exact_nonproxy_current_support_edge", False))
            ),
            "case_with_sampled_exact_nonproxy_current_support_edge_count": len(nonproxy_case_keys),
            "case_with_sampled_exact_nonproxy_current_support_edge_keys": [
                f"{mode}:{case_id}" for mode, case_id in sorted(nonproxy_case_keys)
            ],
            "safe_good_with_sampled_exact_nonproxy_current_support_edge_count": len(safe_good_nonproxy_case_keys),
            "safe_good_with_sampled_exact_nonproxy_current_support_edge_keys": [
                f"{mode}:{case_id}" for mode, case_id in sorted(safe_good_nonproxy_case_keys)
            ],
            "formal_fullquery_strict_current_support_pass_case_count": 0,
            "action_ready_query_head_local_case_count": 0,
            "first_blocker": "a5_sampled_exact_edges_are_not_fullquery_formal_provider_or_action_ready",
            "interpretation": (
                f"The A5 {sample_label} trace payloads already contain exact sampled query/cache/anchor edge tensors, and they "
                "can be materialized without rerunning inference. This repairs auditability of sampled edges only. "
                f"It still does not pass Stage 1 because the evidence is sampled {sample_label}, covers safe-good controls, and "
                "does not provide fullquery formal strict current-support or action-ready query/head-local edges."
            ),
        }
    )
    return edge_rows, case_rows, summary


def audit_selected11_deep_provider_blocker(paths: dict[str, Path]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fullquery_support_rows = read_csv(paths["fullquery_support_rows"])
    stable_source_rows = read_csv(paths["stable_anchor_gate_source_type_rows"])
    raw_tracked_rows = read_csv(paths["raw_tracked_candidate_sidecar_rows"])
    formal_rows = read_csv(paths["formal_provider_chain_rows"])
    formal_summary = read_json(paths["formal_provider_chain_summary"])

    def group(rows: list[dict[str, str]], key: str = "case_id") -> dict[str, list[dict[str, str]]]:
        grouped: dict[str, list[dict[str, str]]] = {}
        for row in rows:
            value = str(row.get(key, "")).strip()
            if value:
                grouped.setdefault(value, []).append(row)
        return grouped

    fq_by_case = group(fullquery_support_rows)
    stable_by_case = group(stable_source_rows)
    raw_by_case = group(raw_tracked_rows)
    formal_by_case = group(formal_rows)
    rows: list[dict[str, Any]] = []

    for case_id in SELECTED_ACTION_CASES:
        fq_rows = fq_by_case.get(case_id, [])
        stable_rows = stable_by_case.get(case_id, [])
        raw_rows = raw_by_case.get(case_id, [])
        formal_case_rows = formal_by_case.get(case_id, [])
        direct_witness_rows = [
            row for row in fq_rows if to_int(row.get("same_masklet_instance_topk_cell_count", 0)) > 0
        ]
        counterfactual_same_masklet_cells = sum(
            to_int(row.get("counterfactual_tracked_instance_anchor_same_masklet_cell_count", 0))
            for row in fq_rows
        )
        counterfactual_query_head_count = sum(
            to_int(row.get("counterfactual_tracked_instance_anchor_query_head_count", 0))
            for row in fq_rows
        )
        stable_lifecycle_cells = (
            sum(to_int(row.get("stable_gated_tracked_cell_count", 0)) for row in stable_rows)
            + sum(to_int(row.get("lifecycle_tracked_row_count", 0)) for row in stable_rows)
            + sum(to_int(row.get("stable_gated_cell_count", 0)) for row in raw_rows)
            + sum(to_int(row.get("lifecycle_row_count", 0)) for row in raw_rows)
        )
        if not direct_witness_rows:
            deep_blocker = "no_fullquery_direct_masklet_instance_witness"
        elif stable_lifecycle_cells <= 0 and counterfactual_same_masklet_cells > 0:
            deep_blocker = "counterfactual_tracked_instance_lifecycle_only_no_formal_stable_anchor_policy"
        elif stable_lifecycle_cells <= 0:
            deep_blocker = "no_stable_gated_or_lifecycle_tracked_candidate"
        elif counterfactual_query_head_count <= 0:
            deep_blocker = "no_counterfactual_query_head_candidate"
        else:
            deep_blocker = "requires_action_ready_query_head_controls_and_read_provider_promotion"
        rows.append(
            {
                "case_id": case_id,
                "case_role": classify_case(case_id),
                "fullquery_candidate_seed_row_count": len(fq_rows),
                "fullquery_full_query_coverage_row_count": sum(
                    1 for row in fq_rows if to_bool(row.get("full_query_coverage_pass", False))
                ),
                "fullquery_direct_witness_seed_count": len(direct_witness_rows),
                "fullquery_direct_witness_seed_ids": ";".join(
                    sorted(str(row.get("seed_global_track_idx", "")).strip() for row in direct_witness_rows)
                ),
                "fullquery_raw_topk_seed_cell_count": sum(
                    to_int(row.get("raw_topk_seed_cell_count", 0)) for row in fq_rows
                ),
                "fullquery_same_seed_topk_cell_count": sum(
                    to_int(row.get("same_seed_topk_cell_count", 0)) for row in fq_rows
                ),
                "fullquery_same_masklet_instance_topk_cell_count": sum(
                    to_int(row.get("same_masklet_instance_topk_cell_count", 0)) for row in fq_rows
                ),
                "counterfactual_tracked_instance_anchor_same_masklet_cell_count": counterfactual_same_masklet_cells,
                "counterfactual_tracked_instance_anchor_query_head_count": counterfactual_query_head_count,
                "stable_source_row_count": len(stable_rows),
                "stable_source_raw_tracked_cell_count": sum(
                    to_int(row.get("raw_tracked_cell_count", 0)) for row in stable_rows
                ),
                "stable_source_stable_gated_tracked_cell_count": sum(
                    to_int(row.get("stable_gated_tracked_cell_count", 0)) for row in stable_rows
                ),
                "stable_source_lifecycle_tracked_row_count": sum(
                    to_int(row.get("lifecycle_tracked_row_count", 0)) for row in stable_rows
                ),
                "raw_tracked_candidate_seed_row_count": len(raw_rows),
                "raw_tracked_candidate_topk_cell_count": sum(
                    to_int(row.get("raw_topk_cell_count", 0)) for row in raw_rows
                ),
                "raw_tracked_candidate_same_query_seed_cell_count": sum(
                    to_int(row.get("raw_same_query_seed_cell_count", 0)) for row in raw_rows
                ),
                "raw_tracked_candidate_stable_gated_cell_count": sum(
                    to_int(row.get("stable_gated_cell_count", 0)) for row in raw_rows
                ),
                "raw_tracked_candidate_lifecycle_row_count": sum(
                    to_int(row.get("lifecycle_row_count", 0)) for row in raw_rows
                ),
                "formal_provider_chain_row_count": len(formal_case_rows),
                "formal_provider_chain_first_blockers": ";".join(
                    sorted({str(row.get("first_blocker", "")).strip() for row in formal_case_rows if row.get("first_blocker")})
                ),
                "formal_provider_chain_counterfactual_promoted_first_blockers": ";".join(
                    sorted(
                        {
                            str(row.get("counterfactual_raw_tracked_promoted_first_blocker", "")).strip()
                            for row in formal_case_rows
                            if row.get("counterfactual_raw_tracked_promoted_first_blocker")
                        }
                    )
                ),
                "deep_provider_blocker": deep_blocker,
                "runtime_action_allowed": False,
            }
        )

    fullquery_direct_case_count = sum(1 for row in rows if to_int(row["fullquery_direct_witness_seed_count"]) > 0)
    stable_lifecycle_case_count = sum(
        1
        for row in rows
        if to_int(row["stable_source_stable_gated_tracked_cell_count"])
        + to_int(row["stable_source_lifecycle_tracked_row_count"])
        + to_int(row["raw_tracked_candidate_stable_gated_cell_count"])
        + to_int(row["raw_tracked_candidate_lifecycle_row_count"])
        > 0
    )
    counterfactual_lifecycle_case_count = sum(
        1 for row in rows if to_int(row["counterfactual_tracked_instance_anchor_same_masklet_cell_count"]) > 0
    )
    counterfactual_query_head_case_count = sum(
        1 for row in rows if to_int(row["counterfactual_tracked_instance_anchor_query_head_count"]) > 0
    )
    blocker_counts = Counter(row["deep_provider_blocker"] for row in rows)
    summary = {
        "schema": "acl2_v104_selected11_deep_provider_blocker_summary_v1",
        "selected_case_count": len(SELECTED_ACTION_CASES),
        "source_fullquery_support_rows": rel(paths["fullquery_support_rows"]),
        "source_stable_anchor_gate_source_type_rows": rel(paths["stable_anchor_gate_source_type_rows"]),
        "source_raw_tracked_candidate_sidecar_rows": rel(paths["raw_tracked_candidate_sidecar_rows"]),
        "source_formal_provider_chain_rows": rel(paths["formal_provider_chain_rows"]),
        "fullquery_candidate_case_count": sum(1 for row in rows if to_int(row["fullquery_candidate_seed_row_count"]) > 0),
        "fullquery_direct_witness_case_count": fullquery_direct_case_count,
        "fullquery_direct_witness_seed_count": sum(to_int(row["fullquery_direct_witness_seed_count"]) for row in rows),
        "selected_fullquery_raw_topk_seed_cell_count": sum(
            to_int(row["fullquery_raw_topk_seed_cell_count"]) for row in rows
        ),
        "selected_fullquery_same_masklet_instance_topk_cell_count": sum(
            to_int(row["fullquery_same_masklet_instance_topk_cell_count"]) for row in rows
        ),
        "selected_stable_source_raw_tracked_cell_count": sum(
            to_int(row["stable_source_raw_tracked_cell_count"]) for row in rows
        ),
        "selected_stable_source_stable_gated_tracked_cell_count": sum(
            to_int(row["stable_source_stable_gated_tracked_cell_count"]) for row in rows
        ),
        "selected_stable_source_lifecycle_tracked_row_count": sum(
            to_int(row["stable_source_lifecycle_tracked_row_count"]) for row in rows
        ),
        "selected_raw_tracked_candidate_topk_cell_count": sum(
            to_int(row["raw_tracked_candidate_topk_cell_count"]) for row in rows
        ),
        "selected_raw_tracked_candidate_stable_gated_cell_count": sum(
            to_int(row["raw_tracked_candidate_stable_gated_cell_count"]) for row in rows
        ),
        "selected_raw_tracked_candidate_lifecycle_row_count": sum(
            to_int(row["raw_tracked_candidate_lifecycle_row_count"]) for row in rows
        ),
        "counterfactual_tracked_instance_lifecycle_candidate_case_count": counterfactual_lifecycle_case_count,
        "counterfactual_tracked_instance_query_head_candidate_case_count": counterfactual_query_head_case_count,
        "formal_provider_chain_provider_pass": formal_summary.get(
            "tracked_instance_query_soft_formal_provider_chain_provider_pass", False
        ),
        "formal_provider_chain_first_blocker": formal_summary.get(
            "tracked_instance_query_soft_formal_provider_chain_first_blocker", ""
        ),
        "formal_provider_chain_first_blocker_counts": formal_summary.get(
            "tracked_instance_query_soft_formal_provider_chain_first_blocker_counts", {}
        ),
        "deep_provider_blocker_counts": dict(blocker_counts),
        "stable_lifecycle_materialized_case_count": stable_lifecycle_case_count,
        "runtime_action_allowed": False,
        "provider_pass": False,
        "interpretation": (
            "Full-query traces already cover all current query tokens where candidate rows exist. Direct same-masklet "
            "witnesses remain limited, and the tracked candidates that do exist are counterfactual-only because formal "
            "stable-anchor/lifecycle tracked evidence is zero. This confirms the next blocker is stable lifecycle and "
            "action-carrier policy, not q128 sampling."
        ),
    }
    return rows, summary


def audit_stable_lifecycle_policy_repair_attempt(
    deep_rows: list[dict[str, Any]],
    stable_summary: dict[str, Any],
    raw_summary: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Try policy-level repairs for the stable/lifecycle blocker without running action."""

    repair_rules = [
        (
            "promote_any_raw_tracked_candidate",
            "Counterfactually treat any raw tracked top-k/anchor source as stable lifecycle evidence.",
            lambda row: (
                to_int(row.get("stable_source_raw_tracked_cell_count", 0))
                + to_int(row.get("raw_tracked_candidate_topk_cell_count", 0))
            )
            > 0,
            False,
        ),
        (
            "promote_fullquery_direct_witness_with_tracked_anchor",
            "Require a full-query direct witness plus a tracked raw/anchor counterfactual.",
            lambda row: to_int(row.get("fullquery_direct_witness_seed_count", 0)) > 0
            and (
                to_int(row.get("counterfactual_tracked_instance_anchor_same_masklet_cell_count", 0))
                + to_int(row.get("raw_tracked_candidate_topk_cell_count", 0))
                + to_int(row.get("stable_source_raw_tracked_cell_count", 0))
            )
            > 0,
            False,
        ),
        (
            "promote_counterfactual_lifecycle_candidate",
            "Promote cases already labeled counterfactual tracked-instance lifecycle candidates.",
            lambda row: str(row.get("deep_provider_blocker", ""))
            == "counterfactual_tracked_instance_lifecycle_only_no_formal_stable_anchor_policy",
            False,
        ),
        (
            "label_oracle_promote_strict_positive_only",
            "Oracle-only upper bound: promote the known strict positive if it has a direct witness.",
            lambda row: row.get("case_id") in STRICT_POSITIVES
            and to_int(row.get("fullquery_direct_witness_seed_count", 0)) > 0,
            True,
        ),
    ]

    case_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    selected_case_count = len([row for row in deep_rows if row.get("case_id")])

    for rule_name, description, predicate, uses_label_oracle in repair_rules:
        selected_cases: list[str] = []
        strict_positive_cases: list[str] = []
        exploratory_positive_cases: list[str] = []
        safe_good_cases: list[str] = []
        for row in deep_rows:
            case_id = str(row.get("case_id", ""))
            selected = bool(predicate(row))
            if selected:
                selected_cases.append(case_id)
                if case_id in STRICT_POSITIVES:
                    strict_positive_cases.append(case_id)
                if case_id in EXPLORATORY_POSITIVES:
                    exploratory_positive_cases.append(case_id)
                if case_id in SAFE_GOOD_CONTROLS:
                    safe_good_cases.append(case_id)
            case_rows.append(
                {
                    "repair_rule": rule_name,
                    "case_id": case_id,
                    "case_role": row.get("case_role", classify_case(case_id)),
                    "selected_by_repair_rule": selected,
                    "uses_label_oracle": uses_label_oracle,
                    "fullquery_direct_witness_seed_count": row.get("fullquery_direct_witness_seed_count", 0),
                    "counterfactual_anchor_same_masklet_cell_count": row.get(
                        "counterfactual_tracked_instance_anchor_same_masklet_cell_count", 0
                    ),
                    "raw_tracked_candidate_topk_cell_count": row.get("raw_tracked_candidate_topk_cell_count", 0),
                    "stable_source_raw_tracked_cell_count": row.get("stable_source_raw_tracked_cell_count", 0),
                    "formal_stable_lifecycle_cells": (
                        to_int(row.get("stable_source_stable_gated_tracked_cell_count", 0))
                        + to_int(row.get("stable_source_lifecycle_tracked_row_count", 0))
                        + to_int(row.get("raw_tracked_candidate_stable_gated_cell_count", 0))
                        + to_int(row.get("raw_tracked_candidate_lifecycle_row_count", 0))
                    ),
                    "safe_good_hit": case_id in SAFE_GOOD_CONTROLS and selected,
                    "runtime_action_allowed": False,
                    "claim_level": "stable_lifecycle_policy_repair_counterfactual_no_runtime",
                }
            )

        safe_good_hit_count = len(safe_good_cases)
        strict_positive_hit_count = len(strict_positive_cases)
        exploratory_positive_hit_count = len(exploratory_positive_cases)
        selected_count = len(selected_cases)
        selected_coverage = selected_count / selected_case_count if selected_case_count else 0.0
        repair_allowed = (
            selected_coverage >= 0.8
            and safe_good_hit_count == 0
            and not uses_label_oracle
            and (strict_positive_hit_count + exploratory_positive_hit_count) >= 3
        )
        if uses_label_oracle:
            first_blocker = "uses_label_oracle_not_provider_rule"
        elif safe_good_hit_count:
            first_blocker = "safe_good_control_hit_by_policy_repair"
        elif selected_coverage < 0.8:
            first_blocker = "selected_action_case_coverage_below_stage1_gate"
        elif (strict_positive_hit_count + exploratory_positive_hit_count) < 3:
            first_blocker = "positive_support_too_narrow_for_action_entry"
        else:
            first_blocker = ""
        summary_rows.append(
            {
                "repair_rule": rule_name,
                "description": description,
                "selected_case_count": selected_count,
                "selected_case_ids": selected_cases,
                "selected_action_case_coverage": selected_coverage,
                "strict_positive_hit_count": strict_positive_hit_count,
                "strict_positive_case_ids": strict_positive_cases,
                "exploratory_positive_hit_count": exploratory_positive_hit_count,
                "exploratory_positive_case_ids": exploratory_positive_cases,
                "safe_good_hit_count": safe_good_hit_count,
                "safe_good_case_ids": safe_good_cases,
                "uses_label_oracle": uses_label_oracle,
                "repair_allowed_for_action": repair_allowed,
                "first_blocker": first_blocker,
                "runtime_action_allowed": False,
            }
        )

    allowed_rows = [row for row in summary_rows if to_bool(row.get("repair_allowed_for_action", False))]
    summary = {
        "schema": "acl2_v104_stable_lifecycle_policy_repair_attempt_summary_v1",
        "source": "selected11_deep_provider_blocker_rows.csv",
        "source_stable_anchor_gate_summary": "v103 stage6 branch_d_stable_anchor_gate_source_type_summary.json",
        "source_raw_tracked_candidate_summary": "v103 stage6 branch_d_raw_tracked_candidate_sidecar_summary.json",
        "stage6_lifecycle_row_count": stable_summary.get("lifecycle_row_count", 0),
        "stage6_lifecycle_tracked_row_count": stable_summary.get("lifecycle_tracked_row_count", 0),
        "stage6_stable_gated_anchor_count": stable_summary.get("stable_gated_anchor_count", 0),
        "stage6_stable_gated_tracked_cell_count": stable_summary.get("stable_gated_tracked_cell_count", 0),
        "stage6_raw_tracked_cell_count": stable_summary.get("raw_tracked_cell_count", 0),
        "raw_candidate_seed_row_count": raw_summary.get("candidate_seed_row_count", 0),
        "raw_candidate_lifecycle_row_count": raw_summary.get("lifecycle_row_count", 0),
        "repair_rule_count": len(summary_rows),
        "repair_allowed_rule_count": len(allowed_rows),
        "repair_allowed_rule_names": [row["repair_rule"] for row in allowed_rows],
        "provider_pass_after_repair_attempt": False,
        "runtime_action_allowed": False,
        "first_blocker": (
            "no_policy_repair_rule_satisfies_coverage_without_safe_good_or_oracle"
            if not allowed_rows
            else ""
        ),
        "interpretation": (
            "Relaxing the stable/lifecycle policy is not a safe repair on the current evidence. Broad tracked "
            "promotion hits safe-good controls, while the only safe single-case promotion uses label knowledge and "
            "does not satisfy selected-case provider coverage. Formal lifecycle/stable tracked evidence remains zero."
        ),
    }
    return case_rows, summary_rows, summary


def audit_row_level_current_support_carrier_repair_attempt(
    paths: dict[str, Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Compare row-level current-support/carrier repair candidates without authorizing runtime."""

    q128_rows = read_csv(paths["target6_q128_stage_c_seed_current_support_rows"])
    fullquery_rows = read_csv(paths["fullquery_support_rows"])
    stuff_rows = read_csv(paths["lifecycle_stuff_static_strict_support_carrier_rows"])
    sidecar_rows = read_csv(paths["lifecycle_repair_sidecar_rows"])

    candidate_rows: list[dict[str, Any]] = []

    for row in q128_rows:
        case_id = str(row.get("case_id", "")).strip()
        if not case_id:
            continue
        current_cache_supported = to_bool(row.get("current_cache_same_seed_supported", False))
        proxy_only = to_bool(row.get("proxy_only", True))
        strict_support = to_bool(row.get("strict_current_support_pass", False))
        if current_cache_supported or case_id in SELECTED_ACTION_CASES:
            candidate_rows.append(
                {
                    "candidate_source": "q128_stage_c_seed_current_support",
                    "case_id": case_id,
                    "case_role": classify_case(case_id),
                    "selected_action_case": case_id in SELECTED_ACTION_CASES,
                    "seed_global_track_idx": row.get("stage_c_seed_global_track_idx", ""),
                    "current_cache_same_seed_supported": current_cache_supported,
                    "proxy_only": proxy_only,
                    "strict_current_support_pass": strict_support,
                    "identity_resolution_level": row.get("identity_resolution_level", ""),
                    "support_quality": row.get("support_quality", ""),
                    "query_head_local_available": False,
                    "action_ready_carrier_available": False,
                    "first_blocker": (
                        "support_proxy_only_or_non_strict_current_support"
                        if current_cache_supported and (proxy_only or not strict_support)
                        else "no_current_cache_same_seed_support"
                    ),
                    "runtime_action_allowed": False,
                    "claim_level": "row_level_current_support_carrier_repair_no_runtime",
                }
            )

    for row in fullquery_rows:
        case_id = str(row.get("case_id", "")).strip()
        if not case_id:
            continue
        direct_witness = to_bool(row.get("direct_masklet_instance_witness", False))
        if direct_witness:
            candidate_rows.append(
                {
                    "candidate_source": "fullquery_direct_masklet_witness",
                    "case_id": case_id,
                    "case_role": classify_case(case_id),
                    "selected_action_case": case_id in SELECTED_ACTION_CASES,
                    "seed_global_track_idx": row.get("seed_global_track_idx", ""),
                    "current_cache_same_seed_supported": direct_witness,
                    "proxy_only": True,
                    "strict_current_support_pass": False,
                    "identity_resolution_level": "fullquery_direct_masklet_instance_witness_diagnostic",
                    "support_quality": "fullquery_direct_witness",
                    "query_head_local_available": to_int(row.get("counterfactual_tracked_instance_anchor_query_head_count", 0))
                    > 0,
                    "action_ready_carrier_available": False,
                    "first_blocker": "fullquery_direct_witness_lacks_formal_lifecycle_action_carrier",
                    "runtime_action_allowed": False,
                    "claim_level": "row_level_current_support_carrier_repair_no_runtime",
                }
            )

    for row in stuff_rows:
        case_id = str(row.get("case_id", "")).strip()
        if not case_id:
            continue
        current_cache_supported = to_bool(row.get("support_current_cache_same_seed_supported", False))
        proxy_only = to_bool(row.get("support_proxy_only", True))
        strict_support = to_bool(row.get("support_strict_current_support_pass", False))
        candidate_rows.append(
            {
                "candidate_source": "stuff_static_raw_present_same_seed_support",
                "case_id": case_id,
                "case_role": classify_case(case_id),
                "selected_action_case": case_id in SELECTED_ACTION_CASES,
                "seed_global_track_idx": row.get("seed_global_track_idx", ""),
                "current_cache_same_seed_supported": current_cache_supported,
                "proxy_only": proxy_only,
                "strict_current_support_pass": strict_support,
                "identity_resolution_level": row.get("support_identity_resolution_level", ""),
                "support_quality": row.get("support_quality", ""),
                "query_head_local_available": to_bool(row.get("a5_query_head_local_action_ready", False)),
                "action_ready_carrier_available": to_bool(row.get("action_ready_carrier_available", False)),
                "raw_trace_present": row.get("raw_trace_present", ""),
                "raw_trace_stable_gated_cell_count": row.get("raw_trace_stable_gated_cell_count", ""),
                "raw_trace_lifecycle_row_count": row.get("raw_trace_lifecycle_row_count", ""),
                "first_blocker": row.get("first_blocker", "support_proxy_only_or_non_strict_current_support"),
                "runtime_action_allowed": False,
                "claim_level": "row_level_current_support_carrier_repair_no_runtime",
            }
        )

    for row in sidecar_rows:
        case_id = str(row.get("case_id", "")).strip()
        if not case_id:
            continue
        materialized = to_bool(row.get("diagnostic_thing_tracked_lifecycle_sidecar_materializable", False))
        if materialized or case_id in SELECTED_ACTION_CASES:
            candidate_rows.append(
                {
                    "candidate_source": "diagnostic_thing_tracked_lifecycle_sidecar",
                    "case_id": case_id,
                    "case_role": classify_case(case_id),
                    "selected_action_case": case_id in SELECTED_ACTION_CASES,
                    "seed_global_track_idx": row.get("witness_seed_ids", ""),
                    "current_cache_same_seed_supported": to_bool(row.get("fullquery_nonproxy_current_support_candidate", False)),
                    "proxy_only": True,
                    "strict_current_support_pass": False,
                    "identity_resolution_level": "diagnostic_thing_tracked_lifecycle_sidecar",
                    "support_quality": row.get("next_blocker_if_diagnostic_sidecar_materialized", ""),
                    "query_head_local_available": to_bool(row.get("action_ready_query_head_local_available", False)),
                    "action_ready_carrier_available": False,
                    "diagnostic_lifecycle_payload_materialized": row.get(
                        "diagnostic_lifecycle_payload_materialized", ""
                    ),
                    "read_provider_promotable": row.get("read_provider_promotable", ""),
                    "first_blocker": row.get(
                        "next_blocker_if_diagnostic_sidecar_materialized",
                        "diagnostic_sidecar_not_promotable",
                    ),
                    "runtime_action_allowed": False,
                    "claim_level": "row_level_current_support_carrier_repair_no_runtime",
                }
            )

    def _rule_row(rule_name: str, description: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
        selected_cases = sorted(
            {str(row.get("case_id", "")) for row in rows if row.get("case_id") in SELECTED_ACTION_CASES}
        )
        safe_good_cases = sorted(case_id for case_id in selected_cases if case_id in SAFE_GOOD_CONTROLS)
        positive_cases = sorted(
            case_id for case_id in selected_cases if case_id in STRICT_POSITIVES or case_id in EXPLORATORY_POSITIVES
        )
        strict_support_cases = sorted(
            {
                str(row.get("case_id", ""))
                for row in rows
                if row.get("case_id") in SELECTED_ACTION_CASES
                and to_bool(row.get("strict_current_support_pass", False))
            }
        )
        action_ready_cases = sorted(
            {
                str(row.get("case_id", ""))
                for row in rows
                if row.get("case_id") in SELECTED_ACTION_CASES
                and to_bool(row.get("action_ready_carrier_available", False))
            }
        )
        selected_coverage = len(selected_cases) / len(SELECTED_ACTION_CASES) if SELECTED_ACTION_CASES else 0.0
        if safe_good_cases:
            first_blocker = "safe_good_control_hit_by_repair_rule"
        elif len(strict_support_cases) <= 0:
            first_blocker = "nonproxy_strict_current_support_absent"
        elif len(action_ready_cases) <= 0:
            first_blocker = "action_ready_query_head_local_carrier_absent"
        elif selected_coverage < 0.8:
            first_blocker = "selected_action_case_coverage_below_stage1_gate"
        elif len(positive_cases) < 3:
            first_blocker = "positive_support_too_narrow_for_action_entry"
        else:
            first_blocker = ""
        repair_allowed = (
            not first_blocker
            and selected_coverage >= 0.8
            and not safe_good_cases
            and bool(strict_support_cases)
            and bool(action_ready_cases)
        )
        return {
            "repair_rule": rule_name,
            "description": description,
            "candidate_row_count": len(rows),
            "selected_case_count": len(selected_cases),
            "selected_case_ids": selected_cases,
            "selected_action_case_coverage": selected_coverage,
            "positive_case_count": len(positive_cases),
            "positive_case_ids": positive_cases,
            "safe_good_hit_count": len(safe_good_cases),
            "safe_good_case_ids": safe_good_cases,
            "strict_current_support_case_count": len(strict_support_cases),
            "strict_current_support_case_ids": strict_support_cases,
            "action_ready_carrier_case_count": len(action_ready_cases),
            "action_ready_carrier_case_ids": action_ready_cases,
            "repair_allowed_for_action": repair_allowed,
            "first_blocker": first_blocker,
            "runtime_action_allowed": False,
        }

    by_source: dict[str, list[dict[str, Any]]] = {}
    for row in candidate_rows:
        by_source.setdefault(str(row.get("candidate_source", "")), []).append(row)

    fullquery_direct_rows = by_source.get("fullquery_direct_masklet_witness", [])
    fullquery_direct_counts = Counter(
        str(row.get("case_id", "")) for row in fullquery_direct_rows if row.get("case_id")
    )
    selected_safe_direct_max = max(
        [fullquery_direct_counts.get(case_id, 0) for case_id in SAFE_GOOD_CONTROLS] or [0]
    )
    fullquery_gt_safe_rows = [
        row
        for row in fullquery_direct_rows
        if fullquery_direct_counts.get(str(row.get("case_id", "")), 0) > selected_safe_direct_max
    ]

    rule_rows = [
        _rule_row(
            "promote_q128_sampled_current_cache_same_seed_support",
            "Promote sampled Stage-C current/cache same-seed support rows.",
            [
                row
                for row in by_source.get("q128_stage_c_seed_current_support", [])
                if to_bool(row.get("current_cache_same_seed_supported", False))
            ],
        ),
        _rule_row(
            "promote_fullquery_direct_masklet_witness_any",
            "Promote any full-query direct masklet-instance witness.",
            fullquery_direct_rows,
        ),
        _rule_row(
            "promote_fullquery_direct_witness_gt_safe_good_max",
            f"Promote full-query direct witness cases whose direct-seed count exceeds selected safe-good max={selected_safe_direct_max}.",
            fullquery_gt_safe_rows,
        ),
        _rule_row(
            "promote_stuff_static_raw_present_same_seed_support",
            "Promote raw-present stuff/static candidates with current/cache same-seed support.",
            [
                row
                for row in by_source.get("stuff_static_raw_present_same_seed_support", [])
                if to_bool(row.get("current_cache_same_seed_supported", False))
            ],
        ),
        _rule_row(
            "promote_diagnostic_thing_tracked_lifecycle_sidecar",
            "Promote materialized diagnostic thing-tracked lifecycle sidecar rows.",
            [
                row
                for row in by_source.get("diagnostic_thing_tracked_lifecycle_sidecar", [])
                if to_bool(row.get("diagnostic_lifecycle_payload_materialized", False))
            ],
        ),
    ]

    allowed_rows = [row for row in rule_rows if to_bool(row.get("repair_allowed_for_action", False))]
    source_counts = Counter(str(row.get("candidate_source", "")) for row in candidate_rows)
    summary = {
        "schema": "acl2_v104_row_level_current_support_carrier_repair_attempt_summary_v1",
        "candidate_row_count": len(candidate_rows),
        "candidate_source_counts": dict(source_counts),
        "repair_rule_count": len(rule_rows),
        "repair_allowed_rule_count": len(allowed_rows),
        "repair_allowed_rule_names": [row["repair_rule"] for row in allowed_rows],
        "fullquery_direct_safe_good_max_seed_count": selected_safe_direct_max,
        "q128_stage_c_seed_support_proxy_only_count": sum(
            1
            for row in by_source.get("q128_stage_c_seed_current_support", [])
            if to_bool(row.get("proxy_only", False))
        ),
        "q128_stage_c_strict_current_support_pass_count": sum(
            1
            for row in by_source.get("q128_stage_c_seed_current_support", [])
            if to_bool(row.get("strict_current_support_pass", False))
        ),
        "stuff_static_candidate_count": len(by_source.get("stuff_static_raw_present_same_seed_support", [])),
        "stuff_static_strict_current_support_pass_count": sum(
            1
            for row in by_source.get("stuff_static_raw_present_same_seed_support", [])
            if to_bool(row.get("strict_current_support_pass", False))
        ),
        "stuff_static_action_ready_carrier_count": sum(
            1
            for row in by_source.get("stuff_static_raw_present_same_seed_support", [])
            if to_bool(row.get("action_ready_carrier_available", False))
        ),
        "provider_pass_after_repair_attempt": False,
        "runtime_action_allowed": False,
        "first_blocker": (
            "no_row_level_repair_rule_satisfies_strict_support_carrier_safe_good_coverage"
            if not allowed_rows
            else ""
        ),
        "interpretation": (
            "All available row-level repair routes remain diagnostic-only. Sampled support and stuff/static rows "
            "are proxy-only/non-strict, full-query direct witnesses either hit safe-good or collapse to a single case, "
            "and no route has an action-ready query/head-local carrier."
        ),
    }
    return candidate_rows, rule_rows, summary


def audit_fresh_q128_tracked_lifecycle_repair_attempt(
    trace_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Audit fresh v104 q128 tracked-lifecycle rows without promoting them to action."""

    candidate_rows: list[dict[str, Any]] = []
    summary = {
        "schema": "acl2_v104_fresh_q128_tracked_lifecycle_repair_attempt_summary_v1",
        "trace_root": rel(trace_root),
        "trace_file_count": 0,
        "trace_loaded_count": 0,
        "trace_load_failed_count": 0,
        "candidate_row_count": 0,
        "same_seed_candidate_case_count": 0,
        "same_masklet_candidate_case_count": 0,
        "same_seed_and_masklet_candidate_case_count": 0,
        "formal_strict_current_support_pass_case_count": 0,
        "action_ready_carrier_case_count": 0,
        "repair_rule_count": 0,
        "repair_allowed_rule_count": 0,
        "repair_allowed_rule_names": [],
        "runtime_action_allowed": False,
        "provider_pass_after_repair_attempt": False,
        "first_blocker": "trace_root_missing",
    }
    if not trace_root.exists():
        return candidate_rows, [], summary

    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment diagnostic
        summary["torch_import_error"] = str(exc)
        summary["first_blocker"] = "torch_import_failed"
        return candidate_rows, [], summary

    trace_paths = sorted(trace_root.glob("*/READ_NO_ACTION/swa_raw_transport_trace/*.pt"))
    summary["trace_file_count"] = len(trace_paths)
    for trace_path in trace_paths:
        case_id = trace_path.parents[2].name
        if case_id not in SELECTED_ACTION_CASES:
            continue
        try:
            payload = torch.load(trace_path, map_location="cpu", weights_only=False)
            summary["trace_loaded_count"] += 1
        except Exception as exc:
            summary["trace_load_failed_count"] += 1
            candidate_rows.append(
                {
                    "candidate_source": "fresh_q128_tracked_lifecycle_payload",
                    "case_id": case_id,
                    "case_role": classify_case(case_id),
                    "trace_path": rel(trace_path),
                    "load_ok": False,
                    "load_error": f"{type(exc).__name__}:{exc}",
                    "runtime_action_allowed": False,
                    "claim_level": "fresh_q128_tracked_lifecycle_load_failed_no_runtime",
                }
            )
            continue

        lifecycle_rows = payload.get("ttt_prev_tracked_instance_anchor_lifecycle_rows") or []
        for row_index, lifecycle_row in enumerate(lifecycle_rows):
            same_seed_count = to_int(lifecycle_row.get("topk_same_seed_position_count", 0))
            same_masklet_count = to_int(lifecycle_row.get("topk_same_masklet_position_count", 0))
            query_head_ge75_frac = to_float(lifecycle_row.get("query_head_ge75_frac", 0.0))
            query_head_hit_max = to_float(lifecycle_row.get("query_head_hit_max", 0.0))
            same_seed_candidate = same_seed_count > 0
            same_masklet_candidate = same_masklet_count > 0
            candidate_rows.append(
                {
                    "candidate_source": "fresh_q128_tracked_lifecycle_payload",
                    "case_id": case_id,
                    "case_role": classify_case(case_id),
                    "selected_action_case": case_id in SELECTED_ACTION_CASES,
                    "trace_path": rel(trace_path),
                    "schema": payload.get("schema", ""),
                    "sampled_query_count": payload.get("sampled_query_count", ""),
                    "row_index": row_index,
                    "anchor_id": lifecycle_row.get("anchor_id", ""),
                    "source_type": lifecycle_row.get("source_type", ""),
                    "source_token_count": lifecycle_row.get("source_token_count", 0),
                    "topk_hit_position_count": lifecycle_row.get("topk_hit_position_count", 0),
                    "topk_same_seed_position_count": same_seed_count,
                    "topk_same_masklet_position_count": same_masklet_count,
                    "same_seed_current_support_candidate": same_seed_candidate,
                    "same_masklet_current_support_candidate": same_masklet_candidate,
                    "same_seed_and_masklet_current_support_candidate": (
                        same_seed_candidate and same_masklet_candidate
                    ),
                    "query_head_hit_frac": lifecycle_row.get("query_head_hit_frac", ""),
                    "query_head_hit_max": lifecycle_row.get("query_head_hit_max", ""),
                    "query_head_ge50_frac": lifecycle_row.get("query_head_ge50_frac", ""),
                    "query_head_ge75_frac": lifecycle_row.get("query_head_ge75_frac", ""),
                    "query_head_ge75_candidate": query_head_ge75_frac > 0.0,
                    "query_head_hit_max_ge75_candidate": query_head_hit_max >= 0.75,
                    "source_stage_c_seed_global_track_idx_mode": lifecycle_row.get(
                        "source_stage_c_seed_global_track_idx_mode", ""
                    ),
                    "source_stage_c_seed_global_track_idx_mode_frac": lifecycle_row.get(
                        "source_stage_c_seed_global_track_idx_mode_frac", ""
                    ),
                    "source_chunk_idx": lifecycle_row.get("source_chunk_idx", ""),
                    "current_chunk_idx": lifecycle_row.get("current_chunk_idx", ""),
                    "source_claim_level": lifecycle_row.get("claim_level", ""),
                    "formal_strict_current_support_pass": False,
                    "action_ready_carrier_available": False,
                    "first_blocker": "fresh_q128_lifecycle_rows_are_diagnostic_not_formal_provider",
                    "runtime_action_allowed": False,
                    "claim_level": "fresh_q128_tracked_lifecycle_repair_no_runtime",
                }
            )

    def _case_set(rows: list[dict[str, Any]], key: str) -> list[str]:
        return sorted(
            {
                str(row.get("case_id", ""))
                for row in rows
                if row.get("case_id") in SELECTED_ACTION_CASES and to_bool(row.get(key, False))
            }
        )

    safe_good_seed_max = max(
        [
            to_int(row.get("topk_same_seed_position_count", 0))
            for row in candidate_rows
            if row.get("case_id") in SAFE_GOOD_CONTROLS
        ]
        or [0]
    )
    safe_good_masklet_max = max(
        [
            to_int(row.get("topk_same_masklet_position_count", 0))
            for row in candidate_rows
            if row.get("case_id") in SAFE_GOOD_CONTROLS
        ]
        or [0]
    )

    def _rule_row(rule_name: str, description: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
        selected_cases = sorted(
            {str(row.get("case_id", "")) for row in rows if row.get("case_id") in SELECTED_ACTION_CASES}
        )
        positive_cases = sorted(
            case_id for case_id in selected_cases if case_id in STRICT_POSITIVES or case_id in EXPLORATORY_POSITIVES
        )
        safe_good_cases = sorted(case_id for case_id in selected_cases if case_id in SAFE_GOOD_CONTROLS)
        formal_cases = sorted(
            {
                str(row.get("case_id", ""))
                for row in rows
                if row.get("case_id") in SELECTED_ACTION_CASES
                and to_bool(row.get("formal_strict_current_support_pass", False))
            }
        )
        action_cases = sorted(
            {
                str(row.get("case_id", ""))
                for row in rows
                if row.get("case_id") in SELECTED_ACTION_CASES
                and to_bool(row.get("action_ready_carrier_available", False))
            }
        )
        selected_coverage = len(selected_cases) / len(SELECTED_ACTION_CASES) if SELECTED_ACTION_CASES else 0.0
        if safe_good_cases:
            first_blocker = "safe_good_control_hit_by_fresh_q128_lifecycle_rule"
        elif not formal_cases:
            first_blocker = "formal_nonproxy_strict_current_support_absent"
        elif not action_cases:
            first_blocker = "action_ready_query_head_local_carrier_absent"
        elif selected_coverage < 0.8:
            first_blocker = "selected_action_case_coverage_below_stage1_gate"
        elif len(positive_cases) < 3:
            first_blocker = "positive_support_too_narrow_for_action_entry"
        else:
            first_blocker = ""
        repair_allowed = (
            not first_blocker
            and selected_coverage >= 0.8
            and not safe_good_cases
            and bool(formal_cases)
            and bool(action_cases)
        )
        return {
            "repair_rule": rule_name,
            "description": description,
            "candidate_row_count": len(rows),
            "selected_case_count": len(selected_cases),
            "selected_case_ids": selected_cases,
            "selected_action_case_coverage": selected_coverage,
            "positive_case_count": len(positive_cases),
            "positive_case_ids": positive_cases,
            "safe_good_hit_count": len(safe_good_cases),
            "safe_good_case_ids": safe_good_cases,
            "formal_strict_current_support_case_count": len(formal_cases),
            "formal_strict_current_support_case_ids": formal_cases,
            "action_ready_carrier_case_count": len(action_cases),
            "action_ready_carrier_case_ids": action_cases,
            "repair_allowed_for_action": repair_allowed,
            "first_blocker": first_blocker,
            "runtime_action_allowed": False,
        }

    rule_rows = [
        _rule_row(
            "promote_fresh_q128_tracked_lifecycle_same_seed_candidate",
            "Promote fresh q128 tracked-lifecycle rows with same-seed top-k support.",
            [row for row in candidate_rows if to_bool(row.get("same_seed_current_support_candidate", False))],
        ),
        _rule_row(
            "promote_fresh_q128_tracked_lifecycle_same_masklet_candidate",
            "Promote fresh q128 tracked-lifecycle rows with same-masklet top-k support.",
            [row for row in candidate_rows if to_bool(row.get("same_masklet_current_support_candidate", False))],
        ),
        _rule_row(
            "promote_fresh_q128_tracked_lifecycle_same_seed_and_masklet_candidate",
            "Promote fresh q128 tracked-lifecycle rows with both same-seed and same-masklet support.",
            [
                row
                for row in candidate_rows
                if to_bool(row.get("same_seed_and_masklet_current_support_candidate", False))
            ],
        ),
        _rule_row(
            "promote_fresh_q128_tracked_lifecycle_qh75_same_seed_candidate",
            "Promote fresh q128 same-seed rows with nonzero query-head ge75 concentration.",
            [
                row
                for row in candidate_rows
                if to_bool(row.get("same_seed_current_support_candidate", False))
                and to_bool(row.get("query_head_ge75_candidate", False))
            ],
        ),
        _rule_row(
            "promote_fresh_q128_tracked_lifecycle_same_seed_gt_safe_good_max",
            f"Promote rows whose same-seed count exceeds selected safe-good max={safe_good_seed_max}.",
            [
                row
                for row in candidate_rows
                if to_int(row.get("topk_same_seed_position_count", 0)) > safe_good_seed_max
            ],
        ),
        _rule_row(
            "promote_fresh_q128_tracked_lifecycle_same_masklet_gt_safe_good_max",
            f"Promote rows whose same-masklet count exceeds selected safe-good max={safe_good_masklet_max}.",
            [
                row
                for row in candidate_rows
                if to_int(row.get("topk_same_masklet_position_count", 0)) > safe_good_masklet_max
            ],
        ),
    ]

    same_seed_cases = _case_set(candidate_rows, "same_seed_current_support_candidate")
    same_masklet_cases = _case_set(candidate_rows, "same_masklet_current_support_candidate")
    same_both_cases = _case_set(candidate_rows, "same_seed_and_masklet_current_support_candidate")
    formal_strict_cases = _case_set(candidate_rows, "formal_strict_current_support_pass")
    action_ready_cases = _case_set(candidate_rows, "action_ready_carrier_available")
    allowed_rows = [row for row in rule_rows if to_bool(row.get("repair_allowed_for_action", False))]
    source_counts = Counter(str(row.get("candidate_source", "")) for row in candidate_rows)
    summary.update(
        {
            "candidate_row_count": len(candidate_rows),
            "candidate_source_counts": dict(source_counts),
            "same_seed_candidate_case_count": len(same_seed_cases),
            "same_seed_candidate_case_ids": same_seed_cases,
            "same_masklet_candidate_case_count": len(same_masklet_cases),
            "same_masklet_candidate_case_ids": same_masklet_cases,
            "same_seed_and_masklet_candidate_case_count": len(same_both_cases),
            "same_seed_and_masklet_candidate_case_ids": same_both_cases,
            "formal_strict_current_support_pass_case_count": len(formal_strict_cases),
            "formal_strict_current_support_pass_case_ids": formal_strict_cases,
            "action_ready_carrier_case_count": len(action_ready_cases),
            "action_ready_carrier_case_ids": action_ready_cases,
            "safe_good_same_seed_max_position_count": safe_good_seed_max,
            "safe_good_same_masklet_max_position_count": safe_good_masklet_max,
            "repair_rule_count": len(rule_rows),
            "repair_allowed_rule_count": len(allowed_rows),
            "repair_allowed_rule_names": [row["repair_rule"] for row in allowed_rows],
            "first_blocker": (
                "no_fresh_q128_tracked_lifecycle_rule_satisfies_strict_support_carrier_safe_good_coverage"
                if not allowed_rows
                else ""
            ),
            "interpretation": (
                "Fresh q128 traces do materialize tracked-instance lifecycle rows and broad same-seed/same-masklet "
                "top-k support, but those candidates also cover safe-good controls and remain diagnostic-only: "
                "formal nonproxy strict current-support and action-ready query/head-local carriers are absent."
            ),
        }
    )
    return candidate_rows, rule_rows, summary


def audit_tracked_local_provider_demotion_rule(
    paths: dict[str, Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Audit focused full-query tracked-instance signals as diagnostic-only demotion evidence."""

    fullquery_support_rows = read_csv(paths["fullquery_support_rows"])
    direct_case_rows = read_csv(paths["direct_match_control_case_rows"])
    direct_threshold_rows = read_csv(paths["direct_match_control_threshold_rows"])
    direct_summary = read_json(paths["direct_match_control_summary"])

    labels_by_case: dict[str, str] = {}
    cats_by_case: dict[str, str] = {}
    direct_by_case_mode: dict[tuple[str, str], dict[str, str]] = {}
    for row in direct_case_rows:
        case_id = row.get("case_id", "")
        if not case_id:
            continue
        labels_by_case.setdefault(case_id, row.get("binary_label", ""))
        cats_by_case.setdefault(case_id, row.get("v103_category_memberships", ""))
        mode = row.get("root_direct_match_mode") or row.get("direct_match_modes", "")
        if mode:
            direct_by_case_mode[(case_id, mode)] = row

    agg: dict[str, dict[str, Any]] = {}
    for row in fullquery_support_rows:
        case_id = row.get("case_id", "")
        if not case_id:
            continue
        out = agg.setdefault(
            case_id,
            {
                "case_id": case_id,
                "fullquery_seed_row_count": 0,
                "fullquery_full_query_coverage_row_count": 0,
                "fullquery_direct_witness_seed_count": 0,
                "fullquery_same_masklet_instance_topk_cell_count": 0,
                "fullquery_same_masklet_instance_topk_cell_count_max": 0,
                "fullquery_same_seed_topk_cell_count": 0,
                "fullquery_raw_topk_seed_cell_count": 0,
                "fullquery_counterfactual_query_head_count": 0,
                "fullquery_counterfactual_query_head_count_max": 0,
            },
        )
        same_masklet = to_int(row.get("same_masklet_instance_topk_cell_count", 0))
        query_head = to_int(row.get("counterfactual_tracked_instance_anchor_query_head_count", 0))
        out["fullquery_seed_row_count"] += 1
        if to_bool(row.get("full_query_coverage_pass", False)):
            out["fullquery_full_query_coverage_row_count"] += 1
        if to_bool(row.get("direct_masklet_instance_witness", False)) or same_masklet > 0:
            out["fullquery_direct_witness_seed_count"] += 1
        out["fullquery_same_masklet_instance_topk_cell_count"] += same_masklet
        out["fullquery_same_masklet_instance_topk_cell_count_max"] = max(
            out["fullquery_same_masklet_instance_topk_cell_count_max"], same_masklet
        )
        out["fullquery_same_seed_topk_cell_count"] += to_int(row.get("same_seed_topk_cell_count", 0))
        out["fullquery_raw_topk_seed_cell_count"] += to_int(row.get("raw_topk_seed_cell_count", 0))
        out["fullquery_counterfactual_query_head_count"] += query_head
        out["fullquery_counterfactual_query_head_count_max"] = max(
            out["fullquery_counterfactual_query_head_count_max"], query_head
        )

    case_ids = sorted(set(agg) | set(labels_by_case))
    case_rows: list[dict[str, Any]] = []
    for case_id in case_ids:
        row = dict(agg.get(case_id, {"case_id": case_id}))
        for key in (
            "fullquery_seed_row_count",
            "fullquery_full_query_coverage_row_count",
            "fullquery_direct_witness_seed_count",
            "fullquery_same_masklet_instance_topk_cell_count",
            "fullquery_same_masklet_instance_topk_cell_count_max",
            "fullquery_same_seed_topk_cell_count",
            "fullquery_raw_topk_seed_cell_count",
            "fullquery_counterfactual_query_head_count",
            "fullquery_counterfactual_query_head_count_max",
        ):
            row.setdefault(key, 0)
        label = labels_by_case.get(case_id, "")
        cats = cats_by_case.get(case_id, "")
        focused_safe_good = label == "good" or "GOOD_PROTECTION_SAFE_GOOD" in cats
        focused_swa_bad = label == "bad" and "SWA_HANDOFF_SCALE_GAUGE" in cats
        row.update(
            {
                "case_role": classify_case(case_id),
                "selected_action_case": case_id in SELECTED_ACTION_CASES,
                "v103_binary_label": label,
                "v103_category_memberships": cats,
                "focused_safe_good_control": focused_safe_good,
                "focused_swa_bad_strict_candidate": focused_swa_bad,
                "focused_read_local_bad": label == "bad" and "READ_LOCAL_SCALE" in cats,
                "runtime_action_allowed": False,
            }
        )
        querysoft_conflict = False
        for mode in ("same_masklet", "same_seed"):
            direct = direct_by_case_mode.get((case_id, mode), {})
            carrier_pass = to_bool(direct.get("carrier_rule_pass", False))
            row[f"querysoft_{mode}_carrier_rule_pass"] = carrier_pass
            row[f"querysoft_{mode}_max_direct_witness_seed_count"] = to_int(
                direct.get("max_direct_witness_seed_count", 0)
            )
            row[f"querysoft_{mode}_max_selected_query_count"] = to_int(direct.get("max_selected_query_count", 0))
            row[f"querysoft_{mode}_max_query_selected_frac"] = direct.get("max_query_selected_frac", "")
            if carrier_pass and to_int(row["fullquery_direct_witness_seed_count"]) <= 0:
                querysoft_conflict = True
        row["querysoft_carrier_pass_but_fullquery_direct_witness_missing"] = querysoft_conflict
        case_rows.append(row)

    safe_good_max_direct_seed = max(
        (
            to_int(row["fullquery_direct_witness_seed_count"])
            for row in case_rows
            if to_bool(row["focused_safe_good_control"])
        ),
        default=0,
    )
    demotion_threshold = safe_good_max_direct_seed + 1

    def evaluate_rule(rule_name: str, feature: str, threshold: int) -> dict[str, Any]:
        selected = [row for row in case_rows if to_int(row.get(feature, 0)) >= threshold and threshold > 0]
        safe = [row["case_id"] for row in selected if to_bool(row["focused_safe_good_control"])]
        strict = [row["case_id"] for row in selected if to_bool(row["focused_swa_bad_strict_candidate"])]
        bad = [row["case_id"] for row in selected if row.get("v103_binary_label") == "bad"]
        selected_action_cases = [row["case_id"] for row in selected if to_bool(row["selected_action_case"])]
        single_case = len(selected) == 1
        safe_for_diagnostic_demote = len(safe) == 0 and len(strict) > 0
        first_blocker = (
            "single_case_provider_pass_only_no_formal_lifecycle_or_controls"
            if safe_for_diagnostic_demote and single_case
            else (
                "rule_selects_safe_good_control"
                if safe
                else "rule_does_not_select_strict_positive"
            )
        )
        return {
            "rule_name": rule_name,
            "feature": feature,
            "direction": "ge",
            "threshold": threshold,
            "selected_case_ids": selected_action_cases,
            "focused_selected_case_ids": [row["case_id"] for row in selected],
            "focused_selected_case_count": len(selected),
            "selected_action_case_count": len(selected_action_cases),
            "focused_safe_good_hit_case_ids": safe,
            "focused_safe_good_hit_case_count": len(safe),
            "focused_strict_swa_bad_hit_case_ids": strict,
            "focused_strict_swa_bad_hit_case_count": len(strict),
            "focused_bad_hit_case_ids": bad,
            "focused_bad_hit_case_count": len(bad),
            "safe_for_diagnostic_demote": safe_for_diagnostic_demote,
            "single_case_provider_pass_only": single_case,
            "provider_pass": False,
            "runtime_action_allowed": False,
            "first_blocker": first_blocker,
            "claim_level": "focused_fullquery_tracked_local_diagnostic_no_runtime",
        }

    strict_02 = next((row for row in case_rows if row["case_id"] == "02_017_018"), {})
    rule_rows = [
        evaluate_rule(
            "direct_seed_count_gt_focused_safe_good_max",
            "fullquery_direct_witness_seed_count",
            demotion_threshold,
        ),
        evaluate_rule(
            "same_masklet_cell_count_ge_02_017_018_value",
            "fullquery_same_masklet_instance_topk_cell_count",
            to_int(strict_02.get("fullquery_same_masklet_instance_topk_cell_count", 0)),
        ),
        evaluate_rule(
            "query_head_count_ge_02_017_018_value",
            "fullquery_counterfactual_query_head_count",
            to_int(strict_02.get("fullquery_counterfactual_query_head_count", 0)),
        ),
        evaluate_rule("direct_seed_count_ge_1", "fullquery_direct_witness_seed_count", 1),
    ]

    best_rule_name = direct_summary.get("tracked_instance_query_soft_direct_match_control_best_rule_name", "")
    best_querysoft_selected = direct_summary.get(
        "tracked_instance_query_soft_direct_match_control_best_selected_case_ids", []
    )
    if isinstance(best_querysoft_selected, str):
        best_querysoft_selected = [case_id for case_id in best_querysoft_selected.split(";") if case_id]
    case_by_id = {row["case_id"]: row for row in case_rows}
    best_rule_conflicts = [
        case_id
        for case_id in best_querysoft_selected
        if to_int(case_by_id.get(case_id, {}).get("fullquery_direct_witness_seed_count", 0)) <= 0
    ]
    threshold_best_row = next((row for row in direct_threshold_rows if row.get("rule_name") == best_rule_name), {})
    accepted_rule = rule_rows[0]
    summary = {
        "schema": "acl2_v104_tracked_local_provider_demotion_rule_audit_v1",
        "source_fullquery_support_rows": rel(paths["fullquery_support_rows"]),
        "source_direct_match_control_case_rows": rel(paths["direct_match_control_case_rows"]),
        "source_direct_match_control_threshold_rows": rel(paths["direct_match_control_threshold_rows"]),
        "focused_fullquery_case_count": len(case_rows),
        "focused_safe_good_control_case_count": sum(1 for row in case_rows if to_bool(row["focused_safe_good_control"])),
        "focused_swa_bad_strict_candidate_case_count": sum(
            1 for row in case_rows if to_bool(row["focused_swa_bad_strict_candidate"])
        ),
        "focused_safe_good_max_fullquery_direct_witness_seed_count": safe_good_max_direct_seed,
        "direct_seed_demotion_threshold": demotion_threshold,
        "demotion_rule_name": accepted_rule["rule_name"],
        "demotion_rule_selected_case_ids": accepted_rule["focused_selected_case_ids"],
        "demotion_rule_selected_action_case_ids": accepted_rule["selected_case_ids"],
        "demotion_rule_selected_case_count": accepted_rule["focused_selected_case_count"],
        "demotion_rule_safe_good_hit_case_count": accepted_rule["focused_safe_good_hit_case_count"],
        "demotion_rule_strict_swa_bad_hit_case_count": accepted_rule["focused_strict_swa_bad_hit_case_count"],
        "demotion_rule_single_case_provider_pass_only": accepted_rule["single_case_provider_pass_only"],
        "demotion_rule_safe_for_diagnostic_demote": accepted_rule["safe_for_diagnostic_demote"],
        "demotion_rule_provider_pass": False,
        "runtime_action_allowed": False,
        "unsafe_cell_strength_counterexample_case_id": "00_012_013",
        "unsafe_cell_strength_counterexample_same_masklet_count": to_int(
            case_by_id.get("00_012_013", {}).get("fullquery_same_masklet_instance_topk_cell_count", 0)
        ),
        "unsafe_cell_strength_counterexample_query_head_count": to_int(
            case_by_id.get("00_012_013", {}).get("fullquery_counterfactual_query_head_count", 0)
        ),
        "strict_positive_02_017_018_direct_seed_count": to_int(
            strict_02.get("fullquery_direct_witness_seed_count", 0)
        ),
        "strict_positive_02_017_018_same_masklet_count": to_int(
            strict_02.get("fullquery_same_masklet_instance_topk_cell_count", 0)
        ),
        "strict_positive_02_017_018_query_head_count": to_int(
            strict_02.get("fullquery_counterfactual_query_head_count", 0)
        ),
        "querysoft_fullquery_conflict_case_count": sum(
            1 for row in case_rows if to_bool(row["querysoft_carrier_pass_but_fullquery_direct_witness_missing"])
        ),
        "v103_querysoft_best_rule_name": best_rule_name,
        "v103_querysoft_best_selected_case_ids": best_querysoft_selected,
        "v103_querysoft_best_rule_first_blocker": threshold_best_row.get("first_blocker", ""),
        "v103_querysoft_best_rule_metric_good_control_candidate_pass": to_bool(
            threshold_best_row.get("metric_good_control_candidate_pass", False)
        ),
        "v103_querysoft_best_selected_cases_missing_v104_fullquery_direct_witness": best_rule_conflicts,
        "interpretation": (
            "Full-query tracked-local demotion can exclude focused safe-good controls only by using direct witness seed "
            "count above the observed safe-good maximum. That rule selects only 02_017_018, so it is a single-case "
            "diagnostic demotion candidate, not a Stage 1 provider pass. Cell-count and query-head-count rules are "
            "unsafe because a focused safe-good case has stronger counts than the strict positive. The v103 query-soft "
            "best rule is also not promotable because one of its selected cases lacks v104 full-query direct witness."
        ),
    }
    return case_rows, rule_rows, summary


def audit_provider_control_payload_feasibility(
    paths: dict[str, Path], demotion_case_rows: list[dict[str, Any]], threshold: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Check whether existing trace tensors can safely stand in for provider controls."""

    trace_root = paths["fullquery_trace_root"]
    demotion_by_case = index_by_case([{k: stringify(v) for k, v in row.items()} for row in demotion_case_rows])
    rows: list[dict[str, Any]] = []
    summary = {
        "schema": "acl2_v104_provider_control_payload_feasibility_v1",
        "source_fullquery_trace_root": rel(trace_root),
        "direct_seed_threshold": threshold,
        "trace_payload_case_count": 0,
        "per_head_tensor_available_case_count": 0,
        "naive_raw_tensor_direct_count_matches_reported_case_count": 0,
        "naive_query_shift_pass_ge_threshold_case_count": 0,
        "naive_query_shift_safe_good_hit_case_count": 0,
        "naive_anchor_rotation_pass_ge_threshold_case_count": 0,
        "naive_anchor_rotation_safe_good_hit_case_count": 0,
        "query_head_random_control_available": False,
        "anchor_id_rotation_control_available": False,
        "runtime_action_allowed": False,
        "first_blocker": "raw_tensor_surrogate_controls_are_not_strict_support_controls",
    }
    if not trace_root.exists():
        summary["trace_root_missing"] = True
        return rows, summary

    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment diagnostic
        summary["torch_import_error"] = str(exc)
        return rows, summary

    shifts = [997, 4093, 8191]
    for case_dir in sorted(trace_root.iterdir()):
        if not case_dir.is_dir():
            continue
        trace_paths = sorted((case_dir / "READ_NO_ACTION/swa_raw_transport_trace").glob("*.pt"))
        if not trace_paths:
            continue
        case_id = case_dir.name
        meta = demotion_by_case.get(case_id, {})
        row: dict[str, Any] = {
            "case_id": case_id,
            "case_role": meta.get("case_role", classify_case(case_id)),
            "focused_safe_good_control": meta.get("focused_safe_good_control", False),
            "focused_swa_bad_strict_candidate": meta.get("focused_swa_bad_strict_candidate", False),
            "trace_path": rel(trace_paths[0]),
            "load_ok": False,
            "runtime_action_allowed": False,
        }
        try:
            payload = torch.load(trace_paths[0], map_location="cpu", weights_only=False)
            q_seed = payload["sampled_query_stage_c_seed_global_track_idx"][0].to(torch.int64)
            q_mask = payload["sampled_query_stage_c_masklet_instance_idx"][0].to(torch.int64)
            cache_mask = payload["current_Q_to_cache_K_topk_cache_stage_c_masklet_instance_idx"][0].to(torch.int64)
            valid_q = (q_seed >= 0) & (q_mask >= 0)
            same_mask = (cache_mask == q_mask[None, :, None]) & valid_q[None, :, None] & (cache_mask >= 0)
            raw_direct_seed_count = int(torch.unique(q_seed[(same_mask.any(dim=(0, 2))) & valid_q]).numel())
            query_shift_counts = []
            for shift in shifts:
                shifted_mask = torch.roll(cache_mask, shifts=shift, dims=1)
                shifted_same = (
                    (shifted_mask == q_mask[None, :, None]) & valid_q[None, :, None] & (shifted_mask >= 0)
                )
                query_shift_counts.append(
                    int(torch.unique(q_seed[(shifted_same.any(dim=(0, 2))) & valid_q]).numel())
                )
            anchor_seed = payload["current_Q_to_cache_K_topk_ttt_prev_tracked_instance_anchor_seeds"][0].to(torch.int64)
            valid_anchor = anchor_seed >= 0
            rotated = anchor_seed.clone()
            if bool(valid_anchor.any()):
                unique = torch.sort(torch.unique(anchor_seed[valid_anchor])).values
                if int(unique.numel()) > 1:
                    for idx in range(int(unique.numel())):
                        rotated[anchor_seed == unique[idx]] = unique[(idx + 1) % int(unique.numel())]
            anchor_rot_same = (rotated == q_seed[None, :, None]) & valid_q[None, :, None] & valid_anchor
            anchor_rotated_seed_count = int(
                torch.unique(q_seed[(anchor_rot_same.any(dim=(0, 2))) & valid_q]).numel()
            )
            reported_direct_seed_count = to_int(meta.get("fullquery_direct_witness_seed_count", 0))
            query_shift_max = max(query_shift_counts) if query_shift_counts else 0
            query_shift_pass = query_shift_max >= threshold
            anchor_rotation_pass = anchor_rotated_seed_count >= threshold
            row.update(
                {
                    "load_ok": True,
                    "head_count": payload.get("head_count", ""),
                    "sampled_query_count": payload.get("sampled_query_count", ""),
                    "reported_strict_support_direct_seed_count": reported_direct_seed_count,
                    "naive_raw_tensor_direct_seed_count": raw_direct_seed_count,
                    "naive_raw_tensor_direct_count_matches_reported": raw_direct_seed_count
                    == reported_direct_seed_count,
                    "query_shift_offsets": shifts,
                    "naive_query_shift_direct_seed_counts": query_shift_counts,
                    "naive_query_shift_max_direct_seed_count": query_shift_max,
                    "naive_query_shift_pass_ge_threshold": query_shift_pass,
                    "naive_anchor_rotation_direct_seed_count": anchor_rotated_seed_count,
                    "naive_anchor_rotation_pass_ge_threshold": anchor_rotation_pass,
                    "claim_level": "payload_feasibility_negative_control_no_runtime",
                    "first_blocker": (
                        "naive_raw_tensor_control_hits_safe_good_or_does_not_match_strict_support_rows"
                    ),
                }
            )
        except Exception as exc:
            row.update({"load_error": str(exc), "first_blocker": "payload_control_feasibility_load_failed"})
        rows.append(row)

    summary["trace_payload_case_count"] = len(rows)
    summary["per_head_tensor_available_case_count"] = sum(1 for row in rows if to_bool(row.get("load_ok", False)))
    summary["naive_raw_tensor_direct_count_matches_reported_case_count"] = sum(
        1 for row in rows if to_bool(row.get("naive_raw_tensor_direct_count_matches_reported", False))
    )
    summary["naive_query_shift_pass_ge_threshold_case_count"] = sum(
        1 for row in rows if to_bool(row.get("naive_query_shift_pass_ge_threshold", False))
    )
    summary["naive_query_shift_safe_good_hit_case_count"] = sum(
        1
        for row in rows
        if to_bool(row.get("naive_query_shift_pass_ge_threshold", False))
        and to_bool(row.get("focused_safe_good_control", False))
    )
    summary["naive_anchor_rotation_pass_ge_threshold_case_count"] = sum(
        1 for row in rows if to_bool(row.get("naive_anchor_rotation_pass_ge_threshold", False))
    )
    summary["naive_anchor_rotation_safe_good_hit_case_count"] = sum(
        1
        for row in rows
        if to_bool(row.get("naive_anchor_rotation_pass_ge_threshold", False))
        and to_bool(row.get("focused_safe_good_control", False))
    )
    summary["interpretation"] = (
        "The trace payloads expose per-head query/top-k tensors, but naive query-shift and anchor-seed rotation "
        "surrogates are not valid Stage 1 controls: they pass many cases including safe-good controls, and raw tensor "
        "direct-seed counts do not match the stricter support-row direct witness counts. A proper control must preserve "
        "the strict full-query support-row universe and provider policy, not recompute from all raw same-masklet cells."
    )
    return rows, summary


def audit_provider_support_row_control_feasibility(
    paths: dict[str, Path], demotion_case_rows: list[dict[str, Any]], threshold: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evaluate control surrogates while preserving the strict full-query support-row seed universe."""

    support_rows = read_csv(paths["fullquery_support_rows"])
    by_case: dict[str, list[dict[str, str]]] = {}
    for row in support_rows:
        case_id = row.get("case_id", "")
        if case_id:
            by_case.setdefault(case_id, []).append(row)
    demotion_by_case = index_by_case([{k: stringify(v) for k, v in row.items()} for row in demotion_case_rows])
    case_ids = sorted(set(demotion_by_case) | set(by_case))
    rows: list[dict[str, Any]] = []
    summary = {
        "schema": "acl2_v104_provider_support_row_control_feasibility_v1",
        "source_fullquery_support_rows": rel(paths["fullquery_support_rows"]),
        "direct_seed_threshold": threshold,
        "focused_case_count": len(demotion_by_case),
        "support_row_case_count": len(by_case),
        "missing_support_row_case_count": 0,
        "support_row_recompute_mismatch_case_count": 0,
        "query_shift_pass_ge_threshold_case_count": 0,
        "query_shift_safe_good_hit_case_count": 0,
        "anchor_rotation_pass_ge_threshold_case_count": 0,
        "anchor_rotation_safe_good_hit_case_count": 0,
        "support_row_query_shift_control_available": False,
        "support_row_anchor_rotation_surrogate_negative_pass": False,
        "query_head_random_control_available": False,
        "anchor_id_rotation_control_available": False,
        "runtime_action_allowed": False,
        "first_blocker": "support_row_controls_are_diagnostic_surrogates_not_plan_controls",
    }
    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment diagnostic
        summary["torch_import_error"] = str(exc)
        return rows, summary

    shifts = [997, 4093, 8191]
    for case_id in case_ids:
        case_support_rows = by_case.get(case_id, [])
        meta = demotion_by_case.get(case_id, {})
        out: dict[str, Any] = {
            "case_id": case_id,
            "case_role": meta.get("case_role", classify_case(case_id)),
            "selected_action_case": case_id in SELECTED_ACTION_CASES,
            "focused_safe_good_control": meta.get("focused_safe_good_control", False),
            "focused_swa_bad_strict_candidate": meta.get("focused_swa_bad_strict_candidate", False),
            "support_row_count": len(case_support_rows),
            "load_ok": False,
            "runtime_action_allowed": False,
            "claim_level": "support_row_preserving_provider_control_feasibility_no_runtime",
        }
        if not case_support_rows:
            out.update(
                {
                    "first_blocker": "no_fullquery_support_rows_for_case",
                    "reported_direct_seed_count": 0,
                    "recomputed_direct_seed_count": 0,
                    "support_row_recompute_mismatch_count": 0,
                    "query_shift_pass_ge_threshold": False,
                    "anchor_rotation_pass_ge_threshold": False,
                }
            )
            rows.append(out)
            continue
        trace_path = Path(case_support_rows[0].get("trace_path", ""))
        out["trace_path"] = rel(trace_path)
        try:
            payload = torch.load(trace_path, map_location="cpu", weights_only=False)
            q_seed_0 = payload["sampled_query_stage_c_seed_global_track_idx"].detach().cpu().long()
            q_inst_0 = payload["sampled_query_stage_c_masklet_instance_idx"].detach().cpu().long()
            k_seed = payload["current_Q_to_cache_K_topk_cache_stage_c_seed_global_track_idx"].detach().cpu().long()
            k_inst = payload["current_Q_to_cache_K_topk_cache_stage_c_masklet_instance_idx"].detach().cpu().long()
            q_seed = q_seed_0[:, None, :, None].expand_as(k_seed)
            q_inst = q_inst_0[:, None, :, None].expand_as(k_inst)
            anchor_seed = payload.get("current_Q_to_cache_K_topk_ttt_prev_tracked_instance_anchor_seeds")
            rotated_anchor = None
            if torch.is_tensor(anchor_seed):
                anchor_seed_tensor = anchor_seed.detach().cpu().long()
                valid_anchor = anchor_seed_tensor >= 0
                rotated_anchor = anchor_seed_tensor.clone()
                if bool(valid_anchor.any()):
                    unique_anchor = torch.sort(torch.unique(anchor_seed_tensor[valid_anchor])).values
                    if int(unique_anchor.numel()) > 1:
                        for idx in range(int(unique_anchor.numel())):
                            rotated_anchor[anchor_seed_tensor == unique_anchor[idx]] = unique_anchor[
                                (idx + 1) % int(unique_anchor.numel())
                            ]

            actual_direct_seeds: list[int] = []
            mismatch_details: list[str] = []
            anchor_rotated_direct_seeds: list[int] = []
            for support in case_support_rows:
                seed = to_int(support.get("seed_global_track_idx", -1), -1)
                if seed < 0:
                    continue
                raw = k_seed == seed
                same = raw & (q_seed == seed) & (k_seed >= 0) & (k_inst >= 0) & (q_inst >= 0) & (k_inst == q_inst)
                count = int(same.sum().item())
                reported = to_int(support.get("same_masklet_instance_topk_cell_count", 0))
                if count != reported:
                    mismatch_details.append(f"{seed}:{reported}->{count}")
                if count > 0:
                    actual_direct_seeds.append(seed)
                if rotated_anchor is not None:
                    rotated_same = same & (rotated_anchor == seed)
                    if int(rotated_same.sum().item()) > 0:
                        anchor_rotated_direct_seeds.append(seed)

            query_shift_counts: list[int] = []
            query_shift_seed_sets: list[list[int]] = []
            for shift in shifts:
                shifted_q_seed = torch.roll(q_seed_0, shifts=shift, dims=1)[:, None, :, None].expand_as(k_seed)
                shifted_q_inst = torch.roll(q_inst_0, shifts=shift, dims=1)[:, None, :, None].expand_as(k_inst)
                shift_direct_seeds: list[int] = []
                for support in case_support_rows:
                    seed = to_int(support.get("seed_global_track_idx", -1), -1)
                    if seed < 0:
                        continue
                    raw = k_seed == seed
                    shifted_same = (
                        raw
                        & (shifted_q_seed == seed)
                        & (k_seed >= 0)
                        & (k_inst >= 0)
                        & (shifted_q_inst >= 0)
                        & (k_inst == shifted_q_inst)
                    )
                    if int(shifted_same.sum().item()) > 0:
                        shift_direct_seeds.append(seed)
                query_shift_counts.append(len(shift_direct_seeds))
                query_shift_seed_sets.append(shift_direct_seeds)

            reported_direct_count = sum(
                1 for support in case_support_rows if to_int(support.get("same_masklet_instance_topk_cell_count", 0)) > 0
            )
            query_shift_max = max(query_shift_counts) if query_shift_counts else 0
            query_shift_pass = query_shift_max >= threshold
            anchor_rotation_pass = len(anchor_rotated_direct_seeds) >= threshold
            out.update(
                {
                    "load_ok": True,
                    "head_count": payload.get("head_count", ""),
                    "sampled_query_count": payload.get("sampled_query_count", ""),
                    "reported_direct_seed_count": reported_direct_count,
                    "recomputed_direct_seed_count": len(actual_direct_seeds),
                    "support_row_recompute_mismatch_count": len(mismatch_details),
                    "support_row_recompute_mismatch_details": mismatch_details,
                    "actual_direct_seed_ids": actual_direct_seeds,
                    "query_shift_offsets": shifts,
                    "query_shift_direct_seed_counts": query_shift_counts,
                    "query_shift_direct_seed_sets": query_shift_seed_sets,
                    "query_shift_max_direct_seed_count": query_shift_max,
                    "query_shift_pass_ge_threshold": query_shift_pass,
                    "anchor_rotated_direct_seed_count": len(anchor_rotated_direct_seeds),
                    "anchor_rotated_direct_seed_ids": anchor_rotated_direct_seeds,
                    "anchor_rotation_pass_ge_threshold": anchor_rotation_pass,
                    "first_blocker": (
                        "query_shift_surrogate_preserves_single_case_signal"
                        if query_shift_pass
                        else "support_row_control_surrogate_diagnostic_only"
                    ),
                }
            )
        except Exception as exc:
            out.update({"load_error": f"{type(exc).__name__}:{exc}", "first_blocker": "support_row_control_load_failed"})
        rows.append(out)

    summary["missing_support_row_case_count"] = sum(1 for row in rows if to_int(row.get("support_row_count", 0)) == 0)
    summary["support_row_recompute_mismatch_case_count"] = sum(
        1 for row in rows if to_int(row.get("support_row_recompute_mismatch_count", 0)) > 0
    )
    summary["query_shift_pass_ge_threshold_case_count"] = sum(
        1 for row in rows if to_bool(row.get("query_shift_pass_ge_threshold", False))
    )
    summary["query_shift_safe_good_hit_case_count"] = sum(
        1
        for row in rows
        if to_bool(row.get("query_shift_pass_ge_threshold", False))
        and to_bool(row.get("focused_safe_good_control", False))
    )
    summary["anchor_rotation_pass_ge_threshold_case_count"] = sum(
        1 for row in rows if to_bool(row.get("anchor_rotation_pass_ge_threshold", False))
    )
    summary["anchor_rotation_safe_good_hit_case_count"] = sum(
        1
        for row in rows
        if to_bool(row.get("anchor_rotation_pass_ge_threshold", False))
        and to_bool(row.get("focused_safe_good_control", False))
    )
    summary["actual_direct_ge_threshold_case_ids"] = [
        row["case_id"] for row in rows if to_int(row.get("recomputed_direct_seed_count", 0)) >= threshold
    ]
    summary["query_shift_ge_threshold_case_ids"] = [
        row["case_id"] for row in rows if to_bool(row.get("query_shift_pass_ge_threshold", False))
    ]
    summary["anchor_rotation_ge_threshold_case_ids"] = [
        row["case_id"] for row in rows if to_bool(row.get("anchor_rotation_pass_ge_threshold", False))
    ]
    summary["support_row_anchor_rotation_surrogate_negative_pass"] = (
        summary["anchor_rotation_pass_ge_threshold_case_count"] == 0
        and summary["support_row_recompute_mismatch_case_count"] == 0
        and summary["support_row_case_count"] > 0
    )
    summary["interpretation"] = (
        "Support-row-preserving recomputation matches the strict full-query support rows, so it is a better diagnostic "
        "than raw tensor all-cell controls. Query-shift still preserves the single-case 02_017_018 direct-seed signal, "
        "so it cannot establish a query-head random control margin. Anchor-seed rotation removes all ge-threshold "
        "support-row signals with no safe-good hits, but it remains a surrogate because support rows do not serialize "
        "the exact query/head/anchor-id witness sets or measured action-control margins. Stage 1 controls remain "
        "unavailable."
    )
    return rows, summary


def audit_support_row_query_head_witness_materialization(
    paths: dict[str, Path], demotion_case_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Materialize head-level witness counts for strict full-query support rows."""

    support_rows = read_csv(paths["fullquery_support_rows"])
    demotion_by_case = index_by_case([{k: stringify(v) for k, v in row.items()} for row in demotion_case_rows])
    rows: list[dict[str, Any]] = []
    summary = {
        "schema": "acl2_v104_support_row_query_head_witness_materialization_v1",
        "source_fullquery_support_rows": rel(paths["fullquery_support_rows"]),
        "support_row_count": len(support_rows),
        "support_row_with_direct_witness_count": sum(
            1 for row in support_rows if to_int(row.get("same_masklet_instance_topk_cell_count", 0)) > 0
        ),
        "head_witness_row_count": 0,
        "direct_witness_case_count": 0,
        "direct_witness_seed_count": 0,
        "selected_direct_witness_case_count": 0,
        "safe_good_direct_witness_case_count": 0,
        "strict_positive_direct_witness_case_count": 0,
        "runtime_action_allowed": False,
        "query_head_local_edges_available": False,
        "first_blocker": "head_level_witness_materialized_but_no_action_ready_query_head_control",
    }
    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment diagnostic
        summary["torch_import_error"] = str(exc)
        return rows, summary

    direct_case_ids: set[str] = set()
    direct_seed_keys: set[tuple[str, int]] = set()
    selected_case_ids: set[str] = set()
    safe_good_case_ids: set[str] = set()
    strict_positive_case_ids: set[str] = set()
    by_trace: dict[str, list[dict[str, str]]] = {}
    for support in support_rows:
        trace_path = support.get("trace_path", "")
        if trace_path:
            by_trace.setdefault(trace_path, []).append(support)

    for trace_text, trace_support_rows in sorted(by_trace.items()):
        trace_path = Path(trace_text)
        try:
            payload = torch.load(trace_path, map_location="cpu", weights_only=False)
            q_seed_0 = payload["sampled_query_stage_c_seed_global_track_idx"].detach().cpu().long()
            q_inst_0 = payload["sampled_query_stage_c_masklet_instance_idx"].detach().cpu().long()
            k_seed = payload["current_Q_to_cache_K_topk_cache_stage_c_seed_global_track_idx"].detach().cpu().long()
            k_inst = payload["current_Q_to_cache_K_topk_cache_stage_c_masklet_instance_idx"].detach().cpu().long()
            q_seed = q_seed_0[:, None, :, None].expand_as(k_seed)
            q_inst = q_inst_0[:, None, :, None].expand_as(k_inst)
            cf_hit = payload.get("current_Q_to_cache_K_topk_ttt_prev_tracked_instance_anchor_hit_mask")
            cf_hit_bool = (
                cf_hit.detach().cpu().bool()
                if torch.is_tensor(cf_hit) and tuple(cf_hit.shape) == tuple(k_seed.shape)
                else torch.zeros_like(k_seed, dtype=torch.bool)
            )
            head_count = int(k_seed.shape[1]) if len(k_seed.shape) >= 2 else to_int(payload.get("head_count", 0))
            for support in trace_support_rows:
                case_id = support.get("case_id", "")
                seed = to_int(support.get("seed_global_track_idx", -1), -1)
                if seed < 0:
                    continue
                meta = demotion_by_case.get(case_id, {})
                raw = k_seed == seed
                same = raw & (q_seed == seed) & (k_seed >= 0) & (k_inst >= 0) & (q_inst >= 0) & (k_inst == q_inst)
                cf_same = cf_hit_bool & same
                for head_idx in range(head_count):
                    head_same = same[:, head_idx, :, :]
                    cell_count = int(head_same.sum().item())
                    if cell_count <= 0:
                        continue
                    query_count = int(head_same.any(dim=-1).sum().item())
                    cf_head_same = cf_same[:, head_idx, :, :]
                    cf_cell_count = int(cf_head_same.sum().item())
                    cf_query_count = int(cf_head_same.any(dim=-1).sum().item())
                    direct_case_ids.add(case_id)
                    direct_seed_keys.add((case_id, seed))
                    if case_id in SELECTED_ACTION_CASES:
                        selected_case_ids.add(case_id)
                    if to_bool(meta.get("focused_safe_good_control", False)):
                        safe_good_case_ids.add(case_id)
                    if to_bool(meta.get("focused_swa_bad_strict_candidate", False)):
                        strict_positive_case_ids.add(case_id)
                    rows.append(
                        {
                            "case_id": case_id,
                            "case_role": meta.get("case_role", classify_case(case_id)),
                            "selected_action_case": case_id in SELECTED_ACTION_CASES,
                            "focused_safe_good_control": meta.get("focused_safe_good_control", False),
                            "focused_swa_bad_strict_candidate": meta.get("focused_swa_bad_strict_candidate", False),
                            "seed_global_track_idx": seed,
                            "query_head": head_idx,
                            "same_masklet_instance_topk_cell_count": cell_count,
                            "same_masklet_instance_query_count": query_count,
                            "counterfactual_anchor_same_masklet_cell_count": cf_cell_count,
                            "counterfactual_anchor_same_masklet_query_count": cf_query_count,
                            "trace_path": rel(trace_path),
                            "query_head_local_witness_materialized": True,
                            "action_ready_query_head_local_edge": False,
                            "runtime_action_allowed": False,
                            "claim_level": "support_row_query_head_witness_materialized_no_runtime",
                        }
                    )
        except Exception as exc:
            rows.append(
                {
                    "case_id": trace_path.parents[2].name if len(trace_path.parents) > 2 else "",
                    "trace_path": rel(trace_path),
                    "load_error": f"{type(exc).__name__}:{exc}",
                    "query_head_local_witness_materialized": False,
                    "action_ready_query_head_local_edge": False,
                    "runtime_action_allowed": False,
                    "claim_level": "support_row_query_head_witness_load_failed",
                }
            )

    summary.update(
        {
            "head_witness_row_count": len([row for row in rows if to_bool(row.get("query_head_local_witness_materialized", False))]),
            "direct_witness_case_count": len(direct_case_ids),
            "direct_witness_seed_count": len(direct_seed_keys),
            "selected_direct_witness_case_count": len(selected_case_ids),
            "safe_good_direct_witness_case_count": len(safe_good_case_ids),
            "strict_positive_direct_witness_case_count": len(strict_positive_case_ids),
            "direct_witness_case_ids": sorted(direct_case_ids),
            "selected_direct_witness_case_ids": sorted(selected_case_ids),
            "safe_good_direct_witness_case_ids": sorted(safe_good_case_ids),
            "strict_positive_direct_witness_case_ids": sorted(strict_positive_case_ids),
            "interpretation": (
                "Head-level witness counts can be materialized from the full-query trace payload and support-row seed "
                "universe. This improves auditability of the local carrier, but it is still not action-ready because "
                "provider controls, exact action-routing authorization, stable lifecycle, and L3/good-control margins "
                "remain missing."
            ),
        }
    )
    return rows, summary


def audit_exact_support_row_witness_set(
    paths: dict[str, Path], demotion_case_rows: list[dict[str, Any]], threshold: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Serialize exact direct-witness cells for support-row direct masklet matches."""

    support_rows = [
        row for row in read_csv(paths["fullquery_support_rows"])
        if to_int(row.get("same_masklet_instance_topk_cell_count", 0)) > 0
    ]
    demotion_by_case = index_by_case([{k: stringify(v) for k, v in row.items()} for row in demotion_case_rows])
    rows: list[dict[str, Any]] = []
    summary = {
        "schema": "acl2_v104_exact_support_row_witness_set_v1",
        "source_fullquery_support_rows": rel(paths["fullquery_support_rows"]),
        "direct_support_row_count": len(support_rows),
        "direct_seed_threshold": threshold,
        "exact_witness_cell_count": 0,
        "exact_witness_case_count": 0,
        "exact_witness_seed_count": 0,
        "safe_good_exact_witness_case_count": 0,
        "strict_positive_exact_witness_case_count": 0,
        "anchor_rotation_preserved_cell_count": 0,
        "anchor_rotation_ge_threshold_case_count": 0,
        "anchor_rotation_safe_good_hit_case_count": 0,
        "exact_anchor_rotation_surrogate_negative_pass": False,
        "query_head_random_control_available": False,
        "anchor_id_rotation_control_available": False,
        "runtime_action_allowed": False,
        "first_blocker": "exact_witness_set_materialized_but_action_control_margins_missing",
    }
    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment diagnostic
        summary["torch_import_error"] = str(exc)
        return rows, summary

    by_trace: dict[str, list[dict[str, str]]] = {}
    for support in support_rows:
        trace_path = support.get("trace_path", "")
        if trace_path:
            by_trace.setdefault(trace_path, []).append(support)

    for trace_text, trace_support_rows in sorted(by_trace.items()):
        trace_path = Path(trace_text)
        try:
            payload = torch.load(trace_path, map_location="cpu", weights_only=False)
            q_seed_0 = payload["sampled_query_stage_c_seed_global_track_idx"].detach().cpu().long()
            q_inst_0 = payload["sampled_query_stage_c_masklet_instance_idx"].detach().cpu().long()
            k_seed = payload["current_Q_to_cache_K_topk_cache_stage_c_seed_global_track_idx"].detach().cpu().long()
            k_inst = payload["current_Q_to_cache_K_topk_cache_stage_c_masklet_instance_idx"].detach().cpu().long()
            q_seed = q_seed_0[:, None, :, None].expand_as(k_seed)
            q_inst = q_inst_0[:, None, :, None].expand_as(k_inst)
            sampled_query_indices = payload.get("sampled_query_indices")
            if torch.is_tensor(sampled_query_indices):
                sampled_query_indices = sampled_query_indices.detach().cpu().long()
            cache_indices = payload.get("current_Q_to_cache_K_topk_cache_indices")
            if torch.is_tensor(cache_indices):
                cache_indices = cache_indices.detach().cpu().long()
            cache_scores = payload.get("current_Q_to_cache_K_topk_scores")
            if torch.is_tensor(cache_scores):
                cache_scores = cache_scores.detach().cpu()
            anchor_ids = payload.get("current_Q_to_cache_K_topk_ttt_prev_tracked_instance_anchor_ids")
            if torch.is_tensor(anchor_ids):
                anchor_ids = anchor_ids.detach().cpu().long()
            anchor_seeds = payload.get("current_Q_to_cache_K_topk_ttt_prev_tracked_instance_anchor_seeds")
            if torch.is_tensor(anchor_seeds):
                anchor_seeds = anchor_seeds.detach().cpu().long()
            cf_hit = payload.get("current_Q_to_cache_K_topk_ttt_prev_tracked_instance_anchor_hit_mask")
            cf_hit_bool = (
                cf_hit.detach().cpu().bool()
                if torch.is_tensor(cf_hit) and tuple(cf_hit.shape) == tuple(k_seed.shape)
                else torch.zeros_like(k_seed, dtype=torch.bool)
            )

            rotated_anchor_seeds = None
            if torch.is_tensor(anchor_seeds):
                rotated_anchor_seeds = anchor_seeds.clone()
                valid_anchor = anchor_seeds >= 0
                if bool(valid_anchor.any()):
                    unique_anchor = torch.sort(torch.unique(anchor_seeds[valid_anchor])).values
                    if int(unique_anchor.numel()) > 1:
                        for idx in range(int(unique_anchor.numel())):
                            rotated_anchor_seeds[anchor_seeds == unique_anchor[idx]] = unique_anchor[
                                (idx + 1) % int(unique_anchor.numel())
                            ]

            for support in trace_support_rows:
                case_id = support.get("case_id", "")
                meta = demotion_by_case.get(case_id, {})
                seed = to_int(support.get("seed_global_track_idx", -1), -1)
                if seed < 0:
                    continue
                same = (
                    (k_seed == seed)
                    & (q_seed == seed)
                    & (k_seed >= 0)
                    & (k_inst >= 0)
                    & (q_inst >= 0)
                    & (k_inst == q_inst)
                )
                nonzero = torch.nonzero(same, as_tuple=False)
                for cell in nonzero.tolist():
                    batch_idx, head_idx, query_offset, topk_rank = [int(x) for x in cell]
                    query_token_index = (
                        int(sampled_query_indices[query_offset].item())
                        if torch.is_tensor(sampled_query_indices)
                        else query_offset
                    )
                    cache_token_index = (
                        int(cache_indices[batch_idx, head_idx, query_offset, topk_rank].item())
                        if torch.is_tensor(cache_indices)
                        else ""
                    )
                    anchor_seed = (
                        int(anchor_seeds[batch_idx, head_idx, query_offset, topk_rank].item())
                        if torch.is_tensor(anchor_seeds)
                        else ""
                    )
                    rotated_anchor_seed = (
                        int(rotated_anchor_seeds[batch_idx, head_idx, query_offset, topk_rank].item())
                        if torch.is_tensor(rotated_anchor_seeds)
                        else ""
                    )
                    anchor_id = (
                        int(anchor_ids[batch_idx, head_idx, query_offset, topk_rank].item())
                        if torch.is_tensor(anchor_ids)
                        else ""
                    )
                    score = (
                        float(cache_scores[batch_idx, head_idx, query_offset, topk_rank].item())
                        if torch.is_tensor(cache_scores)
                        else ""
                    )
                    anchor_hit = bool(cf_hit_bool[batch_idx, head_idx, query_offset, topk_rank].item())
                    rows.append(
                        {
                            "case_id": case_id,
                            "case_role": meta.get("case_role", classify_case(case_id)),
                            "selected_action_case": case_id in SELECTED_ACTION_CASES,
                            "focused_safe_good_control": meta.get("focused_safe_good_control", False),
                            "focused_swa_bad_strict_candidate": meta.get("focused_swa_bad_strict_candidate", False),
                            "seed_global_track_idx": seed,
                            "query_head": head_idx,
                            "query_offset": query_offset,
                            "query_token_index": query_token_index,
                            "topk_rank": topk_rank,
                            "cache_token_index": cache_token_index,
                            "cache_score": score,
                            "query_stage_c_seed_global_track_idx": int(q_seed[batch_idx, head_idx, query_offset, topk_rank].item()),
                            "cache_stage_c_seed_global_track_idx": int(k_seed[batch_idx, head_idx, query_offset, topk_rank].item()),
                            "query_stage_c_masklet_instance_idx": int(q_inst[batch_idx, head_idx, query_offset, topk_rank].item()),
                            "cache_stage_c_masklet_instance_idx": int(k_inst[batch_idx, head_idx, query_offset, topk_rank].item()),
                            "anchor_hit": anchor_hit,
                            "anchor_id": anchor_id,
                            "anchor_seed": anchor_seed,
                            "rotated_anchor_seed": rotated_anchor_seed,
                            "anchor_rotation_preserves_seed": rotated_anchor_seed == seed,
                            "exact_same_seed": True,
                            "exact_same_masklet": True,
                            "trace_path": rel(trace_path),
                            "runtime_action_allowed": False,
                            "claim_level": "exact_support_row_direct_witness_cell_no_runtime",
                        }
                    )
        except Exception as exc:
            rows.append(
                {
                    "trace_path": rel(trace_path),
                    "load_error": f"{type(exc).__name__}:{exc}",
                    "runtime_action_allowed": False,
                    "claim_level": "exact_support_row_witness_set_load_failed",
                }
            )

    direct_cases = {row.get("case_id") for row in rows if row.get("case_id")}
    direct_seed_keys = {
        (str(row.get("case_id")), to_int(row.get("seed_global_track_idx", -1), -1))
        for row in rows
        if row.get("case_id") and to_int(row.get("seed_global_track_idx", -1), -1) >= 0
    }
    safe_good_cases = {
        row.get("case_id")
        for row in rows
        if row.get("case_id") and to_bool(row.get("focused_safe_good_control", False))
    }
    strict_positive_cases = {
        row.get("case_id")
        for row in rows
        if row.get("case_id") and to_bool(row.get("focused_swa_bad_strict_candidate", False))
    }
    rotated_preserved_rows = [row for row in rows if to_bool(row.get("anchor_rotation_preserves_seed", False))]
    rotated_preserved_seed_counts = Counter(
        row.get("case_id")
        for row in rotated_preserved_rows
        if row.get("case_id")
    )
    rotated_ge_case_ids = sorted(
        case_id for case_id, count in rotated_preserved_seed_counts.items() if count >= threshold
    )
    summary.update(
        {
            "exact_witness_cell_count": len([row for row in rows if row.get("case_id")]),
            "exact_witness_case_count": len(direct_cases),
            "exact_witness_case_ids": sorted(direct_cases),
            "exact_witness_seed_count": len(direct_seed_keys),
            "safe_good_exact_witness_case_count": len(safe_good_cases),
            "safe_good_exact_witness_case_ids": sorted(safe_good_cases),
            "strict_positive_exact_witness_case_count": len(strict_positive_cases),
            "strict_positive_exact_witness_case_ids": sorted(strict_positive_cases),
            "anchor_rotation_preserved_cell_count": len(rotated_preserved_rows),
            "anchor_rotation_ge_threshold_case_count": len(rotated_ge_case_ids),
            "anchor_rotation_ge_threshold_case_ids": rotated_ge_case_ids,
            "anchor_rotation_safe_good_hit_case_count": sum(
                1 for case_id in rotated_ge_case_ids if case_id in safe_good_cases
            ),
            "exact_anchor_rotation_surrogate_negative_pass": len(rotated_ge_case_ids) == 0 and bool(rows),
            "interpretation": (
                "Exact direct-witness cells are now serialized with query token, head, top-k rank, cache token, "
                "anchor id, and anchor seed. This is the required audit substrate for future query/head and anchor "
                "controls. It still shows safe-good exact witnesses and does not include measured action-control "
                "margins, so runtime action remains blocked."
            ),
        }
    )
    return rows, summary


def audit_exact_provider_control_diagnostics(
    exact_rows: list[dict[str, Any]], threshold: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run deterministic exact-witness provider-control diagnostics without runtime action."""

    rows: list[dict[str, Any]] = []
    summary = {
        "schema": "acl2_v104_exact_provider_control_diagnostics_v1",
        "source": "stage1_provider/exact_support_row_witness_set_rows.csv",
        "direct_seed_threshold": threshold,
        "exact_witness_cell_count": len([row for row in exact_rows if row.get("case_id")]),
        "observed_ge_threshold_case_ids": [],
        "query_head_rotation_ge_threshold_case_ids": [],
        "anchor_seed_rotation_ge_threshold_case_ids": [],
        "query_head_rotation_control_margin": 0.0,
        "anchor_seed_rotation_control_margin": 0.0,
        "query_head_random_control_available": False,
        "anchor_id_rotation_control_available": False,
        "runtime_action_allowed": False,
        "first_blocker": "exact_controls_not_evaluated",
    }
    if not exact_rows:
        return rows, summary

    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment diagnostic
        summary["torch_import_error"] = str(exc)
        summary["first_blocker"] = "torch_import_failed"
        return rows, summary

    rows_by_trace: dict[str, list[dict[str, Any]]] = {}
    for row in exact_rows:
        trace_path = str(row.get("trace_path", "")).strip()
        if trace_path and row.get("case_id"):
            rows_by_trace.setdefault(trace_path, []).append(row)

    control_shifts = [1, 5, 9]
    observed_seed_by_case: dict[str, set[int]] = {}
    qh_seed_by_case_shift: dict[int, dict[str, set[int]]] = {shift: {} for shift in control_shifts}
    anchor_seed_by_case: dict[str, set[int]] = {}
    case_meta: dict[str, dict[str, Any]] = {}

    for trace_text, trace_rows in sorted(rows_by_trace.items()):
        trace_path = Path(trace_text)
        try:
            payload = torch.load(trace_path, map_location="cpu", weights_only=False)
            q_seed_0 = payload["sampled_query_stage_c_seed_global_track_idx"].detach().cpu().long()
            q_inst_0 = payload["sampled_query_stage_c_masklet_instance_idx"].detach().cpu().long()
            k_seed = payload["current_Q_to_cache_K_topk_cache_stage_c_seed_global_track_idx"].detach().cpu().long()
            k_inst = payload["current_Q_to_cache_K_topk_cache_stage_c_masklet_instance_idx"].detach().cpu().long()
            head_count = int(k_seed.shape[1])
            q_seed = q_seed_0[:, None, :, None].expand_as(k_seed)
            q_inst = q_inst_0[:, None, :, None].expand_as(k_inst)
            for row in trace_rows:
                case_id = str(row.get("case_id", ""))
                seed = to_int(row.get("seed_global_track_idx", -1), -1)
                head = to_int(row.get("query_head", -1), -1)
                query_offset = to_int(row.get("query_offset", -1), -1)
                topk_rank = to_int(row.get("topk_rank", -1), -1)
                if not case_id or seed < 0 or head < 0 or query_offset < 0 or topk_rank < 0:
                    continue
                case_meta.setdefault(case_id, row)
                observed_seed_by_case.setdefault(case_id, set()).add(seed)
                anchor_preserved = to_bool(row.get("anchor_rotation_preserves_seed", False))
                if anchor_preserved:
                    anchor_seed_by_case.setdefault(case_id, set()).add(seed)
                for shift in control_shifts:
                    shifted_head = (head + shift) % head_count
                    same_after_head_rotation = (
                        int(k_seed[0, shifted_head, query_offset, topk_rank].item()) == seed
                        and int(q_seed[0, shifted_head, query_offset, topk_rank].item()) == seed
                        and int(k_inst[0, shifted_head, query_offset, topk_rank].item()) >= 0
                        and int(q_inst[0, shifted_head, query_offset, topk_rank].item()) >= 0
                        and int(k_inst[0, shifted_head, query_offset, topk_rank].item())
                        == int(q_inst[0, shifted_head, query_offset, topk_rank].item())
                    )
                    if same_after_head_rotation:
                        qh_seed_by_case_shift[shift].setdefault(case_id, set()).add(seed)
        except Exception as exc:
            rows.append(
                {
                    "trace_path": rel(trace_path),
                    "control_name": "exact_provider_control_load",
                    "load_error": f"{type(exc).__name__}:{exc}",
                    "runtime_action_allowed": False,
                    "claim_level": "exact_provider_control_diagnostic_load_failed",
                }
            )

    observed_ge_case_ids = sorted(
        case_id for case_id, seeds in observed_seed_by_case.items() if len(seeds) >= threshold
    )
    anchor_ge_case_ids = sorted(
        case_id for case_id, seeds in anchor_seed_by_case.items() if len(seeds) >= threshold
    )
    qh_ge_case_ids_by_shift = {
        shift: sorted(case_id for case_id, seeds in by_case.items() if len(seeds) >= threshold)
        for shift, by_case in qh_seed_by_case_shift.items()
    }
    qh_ge_union = sorted({case_id for ids in qh_ge_case_ids_by_shift.values() for case_id in ids})
    observed_count = len(observed_ge_case_ids)
    query_head_control_margin = (
        (observed_count - len(qh_ge_union)) / observed_count if observed_count else 0.0
    )
    anchor_control_margin = (
        (observed_count - len(anchor_ge_case_ids)) / observed_count if observed_count else 0.0
    )

    case_ids = sorted(set(observed_seed_by_case) | set(anchor_seed_by_case) | set(qh_ge_union))
    for case_id in case_ids:
        meta = case_meta.get(case_id, {})
        row = {
            "case_id": case_id,
            "case_role": meta.get("case_role", classify_case(case_id)),
            "selected_action_case": case_id in SELECTED_ACTION_CASES,
            "focused_safe_good_control": meta.get("focused_safe_good_control", False),
            "focused_swa_bad_strict_candidate": meta.get("focused_swa_bad_strict_candidate", False),
            "observed_direct_seed_count": len(observed_seed_by_case.get(case_id, set())),
            "observed_pass_ge_threshold": len(observed_seed_by_case.get(case_id, set())) >= threshold,
            "anchor_seed_rotation_direct_seed_count": len(anchor_seed_by_case.get(case_id, set())),
            "anchor_seed_rotation_pass_ge_threshold": len(anchor_seed_by_case.get(case_id, set())) >= threshold,
            "runtime_action_allowed": False,
            "claim_level": "exact_provider_control_diagnostic_no_runtime",
        }
        for shift in control_shifts:
            row[f"query_head_rotation_shift_{shift}_direct_seed_count"] = len(
                qh_seed_by_case_shift[shift].get(case_id, set())
            )
            row[f"query_head_rotation_shift_{shift}_pass_ge_threshold"] = (
                len(qh_seed_by_case_shift[shift].get(case_id, set())) >= threshold
            )
        row["query_head_rotation_any_shift_pass_ge_threshold"] = case_id in qh_ge_union
        rows.append(row)

    query_head_safe_good_hit_count = sum(
        1
        for row in rows
        if to_bool(row.get("query_head_rotation_any_shift_pass_ge_threshold", False))
        and to_bool(row.get("focused_safe_good_control", False))
    )
    anchor_safe_good_hit_count = sum(
        1
        for row in rows
        if to_bool(row.get("anchor_seed_rotation_pass_ge_threshold", False))
        and to_bool(row.get("focused_safe_good_control", False))
    )
    query_head_available = bool(observed_ge_case_ids) and not qh_ge_union and query_head_safe_good_hit_count == 0
    anchor_available = bool(observed_ge_case_ids) and not anchor_ge_case_ids and anchor_safe_good_hit_count == 0
    summary.update(
        {
            "observed_ge_threshold_case_count": observed_count,
            "observed_ge_threshold_case_ids": observed_ge_case_ids,
            "query_head_rotation_shifts": control_shifts,
            "query_head_rotation_ge_threshold_case_ids_by_shift": qh_ge_case_ids_by_shift,
            "query_head_rotation_ge_threshold_case_ids": qh_ge_union,
            "query_head_rotation_ge_threshold_case_count": len(qh_ge_union),
            "query_head_rotation_safe_good_hit_case_count": query_head_safe_good_hit_count,
            "anchor_seed_rotation_ge_threshold_case_ids": anchor_ge_case_ids,
            "anchor_seed_rotation_ge_threshold_case_count": len(anchor_ge_case_ids),
            "anchor_seed_rotation_safe_good_hit_case_count": anchor_safe_good_hit_count,
            "query_head_rotation_control_margin": query_head_control_margin,
            "anchor_seed_rotation_control_margin": anchor_control_margin,
            "query_head_random_control_available": query_head_available,
            "anchor_id_rotation_control_available": anchor_available,
            "provider_control_diagnostic_pass": query_head_available and anchor_available,
            "runtime_action_allowed": False,
            "first_blocker": (
                "exact_provider_controls_pass_diagnostic_but_stage1_coverage_and_lifecycle_still_block"
                if query_head_available and anchor_available
                else "exact_provider_control_diagnostic_failed"
            ),
            "interpretation": (
                "Exact provider-control diagnostics rotate query heads and anchor seeds on serialized direct-witness "
                "cells. They are no-action provider controls only; they do not measure Stage4 action effects. Passing "
                "these diagnostics can remove one Stage1 control blocker, but strict provider coverage, lifecycle, "
                "and action metrics still gate runtime."
            ),
        }
    )
    return rows, summary


def v103_paths() -> dict[str, Path]:
    return {
        "focused_cases": V103 / "stage1_focused_drift_source_case_preparation/focused_case_rows.csv",
        "oracle_case_rows": V103 / "stage3_branch_b_semantic_correspondence_oracle/branch_b_oracle_case_rows.csv",
        "oracle_metric_rows": V103 / "stage3_branch_b_semantic_correspondence_oracle/branch_b_oracle_metric_rows.csv",
        "stage3_summary": V103 / "stage3_branch_b_semantic_correspondence_oracle/stage3_branch_b_summary.json",
        "stage4_summary": V103 / "stage4_branch_a_swa_admission_action_surface/stage4_summary.json",
        "stage4_sim_rows": V103 / "stage4_branch_a_swa_admission_action_surface/branch_a_admission_filter_simulator_rows.csv",
        "stage4_inventory": V103 / "stage4_branch_a_swa_admission_action_surface/branch_a_action_surface_inventory_rows.csv",
        "stage5_summary": V103 / "stage5_branch_c_same_space_state_machine/stage5_summary.json",
        "stage6_summary": V103 / "stage6_branch_d_ttt_write_to_use_lifecycle/stage6_summary.json",
        "raw_payload_rows": V103 / "stage6_branch_d_ttt_write_to_use_lifecycle/branch_d_raw_payload_identity_schema_rows.csv",
        "raw_payload_summary": V103 / "stage6_branch_d_ttt_write_to_use_lifecycle/branch_d_raw_payload_identity_schema_summary.json",
        "runtime_current_rows": V103 / "stage6_branch_d_ttt_write_to_use_lifecycle/branch_d_runtime_masklet_instance_current_universe_rows.csv",
        "runtime_current_summary": V103 / "stage6_branch_d_ttt_write_to_use_lifecycle/branch_d_runtime_masklet_instance_current_universe_summary.json",
        "join_ladder_rows": V103 / "stage6_branch_d_ttt_write_to_use_lifecycle/branch_d_action_ready_provider_join_ladder_rows.csv",
        "per_anchor_gap_rows": V103 / "stage6_branch_d_ttt_write_to_use_lifecycle/branch_d_per_anchor_chain_gap_rows.csv",
        "stage7_summary": V103 / "stage7_read_current_support_provider_integration/stage7_summary.json",
        "stage7_inventory": V103 / "stage7_read_current_support_provider_integration/stage7_read_provider_inventory_rows.csv",
        "read_fp_filter_rows": V103 / "stage7_read_current_support_provider_integration/read_support_stage4_false_positive_filter_rows.csv",
        "fullquery_ladder_rows": (
            V103
            / "stage7_read_current_support_provider_integration/tracked_masklet_instance_fullquery_action_ready_ladder_rows.csv"
        ),
        "fullquery_ladder_summary": (
            V103
            / "stage7_read_current_support_provider_integration/tracked_masklet_instance_fullquery_action_ready_ladder_summary.json"
        ),
        "formal_provider_chain_summary": (
            V103
            / "stage7_read_current_support_provider_integration/tracked_instance_query_soft_formal_provider_chain_summary.json"
        ),
        "direct_match_control_summary": (
            V103
            / "stage7_read_current_support_provider_integration/tracked_instance_query_soft_direct_match_control_summary.json"
        ),
        "direct_match_control_case_rows": (
            V103
            / "stage7_read_current_support_provider_integration/tracked_instance_query_soft_direct_match_control_case_rows.csv"
        ),
        "direct_match_control_threshold_rows": (
            V103
            / "stage7_read_current_support_provider_integration/tracked_instance_query_soft_direct_match_control_threshold_rows.csv"
        ),
        "formal_provider_chain_rows": (
            V103
            / "stage7_read_current_support_provider_integration/tracked_instance_query_soft_formal_provider_chain_rows.csv"
        ),
        "fullquery_support_rows": (
            V103
            / "stage7_read_current_support_provider_integration/tracked_masklet_instance_fullquery_support_rows.csv"
        ),
        "fullquery_support_summary": (
            V103
            / "stage7_read_current_support_provider_integration/tracked_masklet_instance_fullquery_support_summary.json"
        ),
        "fullquery_trace_root": (
            V103
            / "stage7_read_current_support_provider_integration/tracked_masklet_instance_fullquery_trace_focused19_goodctrl_20260701_122615"
        ),
        "stable_anchor_gate_source_type_rows": (
            V103 / "stage6_branch_d_ttt_write_to_use_lifecycle/branch_d_stable_anchor_gate_source_type_rows.csv"
        ),
        "stable_anchor_gate_source_type_summary": (
            V103 / "stage6_branch_d_ttt_write_to_use_lifecycle/branch_d_stable_anchor_gate_source_type_summary.json"
        ),
        "raw_tracked_candidate_sidecar_rows": (
            V103 / "stage6_branch_d_ttt_write_to_use_lifecycle/branch_d_raw_tracked_candidate_sidecar_rows.csv"
        ),
        "raw_tracked_candidate_sidecar_summary": (
            V103 / "stage6_branch_d_ttt_write_to_use_lifecycle/branch_d_raw_tracked_candidate_sidecar_summary.json"
        ),
        "lifecycle_repair_sidecar_summary": (
            V103
            / "stage7_read_current_support_provider_integration/tracked_instance_query_soft_lifecycle_repair_sidecar_summary.json"
        ),
        "lifecycle_repair_sidecar_rows": (
            V103
            / "stage7_read_current_support_provider_integration/tracked_instance_query_soft_lifecycle_repair_sidecar_rows.csv"
        ),
        "target6_q128_stage_c_seed_current_support_rows": (
            V103
            / "stage7_read_current_support_provider_integration/target6_q128_stage_c_seed_current_support/target6_q128_stage_c_seed_current_support_rows.csv"
        ),
        "lifecycle_source_type_raw_closure_summary": (
            V103
            / "stage7_read_current_support_provider_integration/tracked_instance_lifecycle_payload_source_type_raw_closure_summary.json"
        ),
        "lifecycle_source_type_provider_policy_summary": (
            V103
            / "stage7_read_current_support_provider_integration/tracked_instance_lifecycle_payload_source_type_provider_policy_summary.json"
        ),
        "lifecycle_stuff_static_strict_support_carrier_summary": (
            V103
            / "stage7_read_current_support_provider_integration/tracked_instance_lifecycle_payload_stuff_static_strict_support_carrier_summary.json"
        ),
        "lifecycle_stuff_static_strict_support_carrier_rows": (
            V103
            / "stage7_read_current_support_provider_integration/tracked_instance_lifecycle_payload_stuff_static_strict_support_carrier_rows.csv"
        ),
        "stage8_summary": V103 / "stage8_training_free_cue_distillation_blocked/stage8_summary.json",
        "final_decision": V103 / "final_decision/final_decision.json",
        "selected11_fresh_trace_root": latest_selected11_trace_root(),
        "selected11_no_trace_pose_sha_parity_root": latest_no_trace_pose_sha_parity_root(),
        "selected11_a5_trace_root": latest_selected11_a5_trace_root(),
        "selected11_a5_no_trace_pose_sha_parity_root": latest_a5_no_trace_pose_sha_parity_root(),
        "selected11_a5_same_masklet_trace_root": latest_selected11_a5_same_masklet_trace_root(),
        "selected11_a5_same_masklet_no_trace_pose_sha_parity_root": (
            latest_a5_same_masklet_no_trace_pose_sha_parity_root()
        ),
        "selected5_a5_q4096raw_same_seed_trace_root": latest_selected5_a5_q4096raw_trace_root("same_seed"),
        "selected5_a5_q4096raw_same_masklet_trace_root": latest_selected5_a5_q4096raw_trace_root("same_masklet"),
        "selected5_a5_q4096raw_same_seed_no_trace_pose_sha_parity_root": (
            latest_selected5_a5_q4096raw_no_trace_pose_sha_parity_root("same_seed")
        ),
        "selected5_a5_q4096raw_same_masklet_no_trace_pose_sha_parity_root": (
            latest_selected5_a5_q4096raw_no_trace_pose_sha_parity_root("same_masklet")
        ),
    }


def classify_case(case_id: str) -> str:
    if case_id in STRICT_POSITIVES:
        return "strict_positive"
    if case_id in EXPLORATORY_POSITIVES:
        return "exploratory_positive"
    if case_id in SAFE_GOOD_CONTROLS:
        return "safe_good_control"
    return "focused_base"


def build_stage0() -> dict[str, Any]:
    out = ROOT / "stage0_evidence_freeze"
    paths = v103_paths()
    v103_final = read_json(paths["final_decision"])
    known_facts = {
        "schema": "acl2_v104_stage0_known_facts_v1",
        "plan": rel(PLAN),
        "v103_root": rel(V103),
        "v103_final_taxonomy": v103_final.get("final_taxonomy"),
        "v103_runtime_action_allowed": v103_final.get("runtime_action_allowed", False),
        "v103_full_validation_run": v103_final.get("full_validation_run", False),
        "v103_blocking_requirements": v103_final.get("blocking_requirements", []),
        "v104_interpretation": "v103 artifacts are input evidence only; proxy/provider diagnostics are not promoted to v104 success.",
    }
    write_json(out / "stage0_known_facts.json", known_facts)

    forbidden_md = ["# Stage 0 Forbidden Repeat List", ""]
    for item in FORBIDDEN_REPEATS:
        forbidden_md.append(f"- {item}")
    forbidden_md.extend(
        [
            "",
            "Builder behavior: no runtime action is launched from any forbidden family unless a new provider, target, action body, and metric are all recorded.",
            "",
        ]
    )
    write_text(out / "stage0_forbidden_repeat_list.md", "\n".join(forbidden_md))

    provider_rows = [
        {"route_id": rid, "route_name": name, "v104_status": status, "runtime_action_allowed": False}
        for rid, name, status in ALLOWED_PROVIDER_ROUTES
    ]
    action_rows = [
        {"action_clue": clue, "source_interpretation": interp, "v104_status": status, "runtime_action_allowed": False}
        for clue, interp, status in ALLOWED_ACTION_CLUES
    ]
    write_csv(out / "stage0_allowed_legacy_provider_routes.csv", provider_rows)
    write_csv(out / "stage0_allowed_action_surface_clues.csv", action_rows)
    write_text(
        out / "why_this_is_not_a_repeat.md",
        "# Why v104 is not a forbidden repeat\n\n"
        "v104 does not rerun old query-soft, READ beta, source-gate, merge-alpha, or aggregate delay sweeps. "
        "It first audits strict instance-level provider eligibility, then blocks action when the provider gate fails.\n",
    )

    summary = {
        "schema": "acl2_v104_stage0_summary_v1",
        "stage": 0,
        "stage0_pass": True,
        "known_facts_path": rel(out / "stage0_known_facts.json"),
        "forbidden_repeat_list_path": rel(out / "stage0_forbidden_repeat_list.md"),
        "allowed_legacy_provider_routes_path": rel(out / "stage0_allowed_legacy_provider_routes.csv"),
        "allowed_action_surface_clues_path": rel(out / "stage0_allowed_action_surface_clues.csv"),
        "builder_refuses_forbidden_action_without_new_reason": True,
    }
    write_json(out / "stage0_summary.json", summary)
    return summary


def build_stage1() -> dict[str, Any]:
    out = ROOT / "stage1_provider"
    paths = v103_paths()
    focused = index_by_case(read_csv(paths["focused_cases"]))
    raw_by_case = index_by_case(read_csv(paths["raw_payload_rows"]))
    join_by_case = index_by_case(read_csv(paths["join_ladder_rows"]))
    current_by_case = index_by_case(read_csv(paths["runtime_current_rows"]))
    fullquery_rows = read_csv(paths["fullquery_ladder_rows"])
    fullquery_by_case = index_by_case(fullquery_rows)
    fullquery_selected_rows = [row for row in fullquery_rows if row.get("case_id", "") in SELECTED_ACTION_CASES]
    fullquery_imported_summary = read_json(paths["fullquery_ladder_summary"])
    formal_provider_chain_summary = read_json(paths["formal_provider_chain_summary"])
    direct_match_control_summary = read_json(paths["direct_match_control_summary"])
    fresh_trace_rows, fresh_trace_summary = audit_selected11_fresh_trace(paths["selected11_fresh_trace_root"])
    parity_rows, parity_summary = audit_selected11_no_action_parity(
        paths["selected11_fresh_trace_root"], paths["selected11_no_trace_pose_sha_parity_root"]
    )
    deep_blocker_rows, deep_blocker_summary = audit_selected11_deep_provider_blocker(paths)
    stable_anchor_gate_summary = read_json(paths["stable_anchor_gate_source_type_summary"])
    raw_tracked_candidate_summary = read_json(paths["raw_tracked_candidate_sidecar_summary"])
    stable_policy_repair_rows, stable_policy_repair_rule_rows, stable_policy_repair_summary = (
        audit_stable_lifecycle_policy_repair_attempt(
            deep_blocker_rows, stable_anchor_gate_summary, raw_tracked_candidate_summary
        )
    )
    lifecycle_repair_sidecar_summary = read_json(paths["lifecycle_repair_sidecar_summary"])
    lifecycle_raw_closure_summary = read_json(paths["lifecycle_source_type_raw_closure_summary"])
    lifecycle_provider_policy_summary = read_json(paths["lifecycle_source_type_provider_policy_summary"])
    lifecycle_stuff_carrier_summary = read_json(paths["lifecycle_stuff_static_strict_support_carrier_summary"])
    lifecycle_payload_closure_summary = {
        "schema": "acl2_v104_imported_v103_lifecycle_payload_closure_summary_v1",
        "source_lifecycle_repair_sidecar_summary": rel(paths["lifecycle_repair_sidecar_summary"]),
        "source_lifecycle_source_type_raw_closure_summary": rel(paths["lifecycle_source_type_raw_closure_summary"]),
        "source_lifecycle_source_type_provider_policy_summary": rel(
            paths["lifecycle_source_type_provider_policy_summary"]
        ),
        "source_lifecycle_stuff_static_strict_support_carrier_summary": rel(
            paths["lifecycle_stuff_static_strict_support_carrier_summary"]
        ),
        "lifecycle_repair_sidecar_materializable_case_count": lifecycle_repair_sidecar_summary.get(
            "tracked_instance_query_soft_lifecycle_repair_sidecar_materializable_case_count", 0
        ),
        "lifecycle_repair_sidecar_provider_pass": lifecycle_repair_sidecar_summary.get(
            "tracked_instance_query_soft_lifecycle_repair_sidecar_provider_pass", False
        ),
        "lifecycle_repair_sidecar_next_blocker_counts": lifecycle_repair_sidecar_summary.get(
            "tracked_instance_query_soft_lifecycle_repair_sidecar_next_blocker_counts", {}
        ),
        "raw_closure_candidate_count": lifecycle_raw_closure_summary.get(
            "tracked_instance_lifecycle_payload_source_type_raw_closure_candidate_count", 0
        ),
        "raw_closure_raw_trace_present_count": lifecycle_raw_closure_summary.get(
            "tracked_instance_lifecycle_payload_source_type_raw_closure_raw_trace_present_count", 0
        ),
        "raw_closure_stuff_static_pair_count": lifecycle_raw_closure_summary.get(
            "tracked_instance_lifecycle_payload_source_type_raw_closure_stuff_static_pair_count", 0
        ),
        "raw_closure_first_blocker_counts": lifecycle_raw_closure_summary.get(
            "tracked_instance_lifecycle_payload_source_type_raw_closure_first_blocker_counts", {}
        ),
        "provider_policy_raw_tensor_absence_closed": lifecycle_provider_policy_summary.get(
            "tracked_instance_lifecycle_payload_source_type_provider_policy_raw_tensor_absence_closed", False
        ),
        "provider_policy_source_type_boundary_closed": lifecycle_provider_policy_summary.get(
            "tracked_instance_lifecycle_payload_source_type_provider_policy_source_type_boundary_closed", False
        ),
        "provider_policy_nonproxy_strict_current_support_pass": lifecycle_provider_policy_summary.get(
            "tracked_instance_lifecycle_payload_source_type_provider_policy_nonproxy_strict_current_support_pass",
            False,
        ),
        "provider_policy_strict_provider_materializable": lifecycle_provider_policy_summary.get(
            "tracked_instance_lifecycle_payload_source_type_provider_policy_strict_provider_materializable", False
        ),
        "provider_policy_action_ready_carrier_available": lifecycle_provider_policy_summary.get(
            "tracked_instance_lifecycle_payload_source_type_provider_policy_action_ready_carrier_available", False
        ),
        "provider_policy_allowed": lifecycle_provider_policy_summary.get(
            "tracked_instance_lifecycle_payload_source_type_provider_policy_allowed", False
        ),
        "stuff_static_carrier_candidate_count": lifecycle_stuff_carrier_summary.get(
            "tracked_instance_lifecycle_payload_stuff_static_strict_support_carrier_candidate_count", 0
        ),
        "stuff_static_current_cache_same_seed_supported_count": lifecycle_stuff_carrier_summary.get(
            "tracked_instance_lifecycle_payload_stuff_static_strict_support_carrier_current_cache_same_seed_supported_count",
            0,
        ),
        "stuff_static_support_strict_current_support_pass_count": lifecycle_stuff_carrier_summary.get(
            "tracked_instance_lifecycle_payload_stuff_static_strict_support_carrier_support_strict_current_support_pass_count",
            0,
        ),
        "stuff_static_strict_provider_materializable_count": lifecycle_stuff_carrier_summary.get(
            "tracked_instance_lifecycle_payload_stuff_static_strict_support_carrier_strict_provider_materializable_count",
            0,
        ),
        "stuff_static_action_ready_carrier_available": lifecycle_stuff_carrier_summary.get(
            "tracked_instance_lifecycle_payload_stuff_static_strict_support_carrier_action_ready_carrier_available",
            False,
        ),
        "stuff_static_first_blocker_counts": lifecycle_stuff_carrier_summary.get(
            "tracked_instance_lifecycle_payload_stuff_static_strict_support_carrier_first_blocker_counts", {}
        ),
        "runtime_action_allowed": False,
        "provider_pass": False,
        "first_blocker": "v103_raw_absence_closed_but_nonproxy_strict_current_support_and_action_carrier_missing",
        "interpretation": (
            "Imported v103 Stage7 closure artifacts show that the next wall is not raw tensor absence. A small "
            "stuff/static lifecycle surface has raw traces and current/cache same-seed support, but strict current "
            "support, strict provider materialization, and action-ready query/head-local carrier all remain absent."
        ),
    }
    row_level_repair_rows, row_level_repair_rule_rows, row_level_repair_summary = (
        audit_row_level_current_support_carrier_repair_attempt(paths)
    )
    fresh_q128_lifecycle_rows, fresh_q128_lifecycle_rule_rows, fresh_q128_lifecycle_summary = (
        audit_fresh_q128_tracked_lifecycle_repair_attempt(paths["selected11_fresh_trace_root"])
    )
    a5_trace_rows, a5_trace_rule_rows, a5_trace_summary = audit_a5_query_head_local_trace_repair_attempt(
        paths["selected11_a5_trace_root"], paths["selected11_a5_no_trace_pose_sha_parity_root"]
    )
    a5_same_masklet_rows, a5_same_masklet_rule_rows, a5_same_masklet_summary = (
        audit_a5_query_head_local_trace_repair_attempt(
            paths["selected11_a5_same_masklet_trace_root"],
            paths["selected11_a5_same_masklet_no_trace_pose_sha_parity_root"],
        )
    )
    a5_threshold_rows, a5_threshold_summary = audit_a5_direct_match_threshold_repair_attempt(
        {
            "same_seed": a5_trace_rows,
            "same_masklet": a5_same_masklet_rows,
        }
    )
    a5_sampled_edge_rows, a5_sampled_edge_case_rows, a5_sampled_edge_summary = (
        audit_a5_sampled_exact_edge_materialization(
            {
                "same_seed": paths["selected11_a5_trace_root"],
                "same_masklet": paths["selected11_a5_same_masklet_trace_root"],
            },
            sample_label="q128",
            selected_case_scope="selected11",
        )
    )
    a5_q4096_edge_rows, a5_q4096_edge_case_rows, a5_q4096_edge_summary = (
        audit_a5_sampled_exact_edge_materialization(
            {
                "same_seed": paths["selected5_a5_q4096raw_same_seed_trace_root"],
                "same_masklet": paths["selected5_a5_q4096raw_same_masklet_trace_root"],
            },
            sample_label="q4096raw",
            selected_case_scope="selected5",
        )
    )
    a5_q4096_same_seed_parity_summary = read_json(
        paths["selected5_a5_q4096raw_same_seed_no_trace_pose_sha_parity_root"] / "summary.json"
    )
    a5_q4096_same_masklet_parity_summary = read_json(
        paths["selected5_a5_q4096raw_same_masklet_no_trace_pose_sha_parity_root"] / "summary.json"
    )
    a5_q4096_parity_pass = bool(a5_q4096_same_seed_parity_summary.get("parity_pass")) and bool(
        a5_q4096_same_masklet_parity_summary.get("parity_pass")
    )
    a5_q4096_pose_sha_equal_count = to_int(a5_q4096_same_seed_parity_summary.get("pose_sha_equal_case_count", 0)) + to_int(
        a5_q4096_same_masklet_parity_summary.get("pose_sha_equal_case_count", 0)
    )
    a5_q4096_pose_numeric_equal_count = to_int(
        a5_q4096_same_seed_parity_summary.get("pose_numeric_equal_case_count", 0)
    ) + to_int(a5_q4096_same_masklet_parity_summary.get("pose_numeric_equal_case_count", 0))
    demotion_case_rows, demotion_rule_rows, demotion_summary = audit_tracked_local_provider_demotion_rule(paths)
    control_payload_rows, control_payload_summary = audit_provider_control_payload_feasibility(
        paths, demotion_case_rows, to_int(demotion_summary.get("direct_seed_demotion_threshold", 0))
    )
    support_control_rows, support_control_summary = audit_provider_support_row_control_feasibility(
        paths, demotion_case_rows, to_int(demotion_summary.get("direct_seed_demotion_threshold", 0))
    )
    query_head_witness_rows, query_head_witness_summary = audit_support_row_query_head_witness_materialization(
        paths, demotion_case_rows
    )
    exact_witness_rows, exact_witness_summary = audit_exact_support_row_witness_set(
        paths, demotion_case_rows, to_int(demotion_summary.get("direct_seed_demotion_threshold", 0))
    )
    exact_control_rows, exact_control_summary = audit_exact_provider_control_diagnostics(
        exact_witness_rows, to_int(demotion_summary.get("direct_seed_demotion_threshold", 0))
    )
    fresh_by_case = index_by_case(fresh_trace_rows)
    stage4 = read_json(paths["stage4_summary"])
    stage6 = read_json(paths["stage6_summary"])

    manifest_rows: list[dict[str, Any]] = []
    provider_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []

    for case_id in sorted(focused):
        base = focused[case_id]
        raw = raw_by_case.get(case_id, {})
        join = join_by_case.get(case_id, {})
        current = current_by_case.get(case_id, {})
        fullquery = fullquery_by_case.get(case_id, {})
        fresh = fresh_by_case.get(case_id, {})
        is_selected = case_id in SELECTED_ACTION_CASES
        old_ladder_strict_identity = to_int(join.get("strict_instance_identity_anchor_count", 0)) > 0
        old_ladder_nonproxy_support = to_int(join.get("nonproxy_current_support_anchor_count", 0)) > 0
        fullquery_strict_identity = to_bool(fullquery.get("fullquery_strict_identity_candidate", False))
        fullquery_nonproxy_support = to_bool(fullquery.get("fullquery_nonproxy_current_support_candidate", False))
        fullquery_stable_lifecycle = to_bool(
            fullquery.get("stable_lifecycle_materialized_for_fullquery_tracked_instance", False)
        )
        fullquery_query_head_local = to_bool(fullquery.get("action_ready_query_head_local_available", False))
        fullquery_action_ready_pass = to_bool(fullquery.get("case_fullquery_action_ready_provider_pass", False))
        old_ladder_action_ready_pass = to_bool(join.get("case_action_ready_provider_pass", False))
        strict_identity = old_ladder_strict_identity or fullquery_strict_identity
        nonproxy_support = old_ladder_nonproxy_support or fullquery_nonproxy_support
        action_ready_provider_join = old_ladder_action_ready_pass or fullquery_action_ready_pass
        raw_has_masklet = to_int(raw.get("component_instance_masklet_key_count", 0)) > 0
        fresh_direct_masklet = to_bool(fresh.get("direct_masklet_payload_materialized", False))
        fresh_direct_seed = to_bool(fresh.get("direct_seed_payload_materialized", False))
        strict_current_cache_component = to_bool(fresh.get("strict_current_cache_component_candidate", False))
        strict_current_cache_seed = to_bool(fresh.get("strict_current_cache_seed_candidate", False))
        ttt_anchor_identity = to_bool(fresh.get("ttt_prev_tracked_instance_anchor_identity_available", False))
        effective_raw_has_masklet = raw_has_masklet or fresh_direct_masklet
        semantic_fallback = "semantic_class_fallback" in join.get("support_identity_resolution_levels", "")
        proxy_only = to_bool(join.get("support_proxy_only_values", False)) or semantic_fallback

        if action_ready_provider_join and effective_raw_has_masklet:
            resolution = "strict_masklet_instance"
            first_blocker = ""
        elif fresh_direct_masklet and fullquery_strict_identity and not fullquery_stable_lifecycle:
            resolution = "strict_masklet_instance"
            first_blocker = "fullquery_strict_candidate_but_stable_lifecycle_missing"
        elif fresh_direct_masklet and fullquery_strict_identity and not fullquery_query_head_local:
            resolution = "strict_masklet_instance"
            first_blocker = "fullquery_strict_candidate_but_query_head_local_missing"
        elif fresh_direct_masklet and not fullquery_strict_identity:
            resolution = "strict_component_payload_candidate"
            first_blocker = "direct_sampled_payload_available_but_no_fullquery_direct_masklet_instance_witness"
        elif fresh_direct_masklet:
            resolution = "strict_component_payload_candidate"
            first_blocker = "direct_masklet_payload_materialized_but_action_ready_provider_join_missing"
        elif semantic_fallback:
            resolution = "semantic_class_fallback"
            first_blocker = "semantic_class_fallback_only"
        elif raw and not effective_raw_has_masklet:
            resolution = "missing"
            first_blocker = "raw_payload_lacks_component_instance_masklet_keys"
        elif current and to_int(current.get("tracked_instance_seed_overlap_anchor_count", 0)) == 0:
            resolution = "missing"
            first_blocker = "current_universe_has_no_tracked_instance_seed_overlap"
        else:
            resolution = "missing"
            first_blocker = "provider_join_missing"

        manifest_rows.append(
            {
                "case_id": case_id,
                "seq": base.get("seq", ""),
                "chunk_prev": base.get("prev_chunk", ""),
                "chunk_curr": base.get("curr_chunk", ""),
                "selected_action_case": is_selected,
                "case_role": classify_case(case_id),
                "raw_payload_trace_path": raw.get("trace_path", ""),
                "fresh_trace_path": fresh.get("trace_path", ""),
                "raw_payload_loaded": raw.get("load_ok", ""),
                "fresh_trace_loaded": fresh.get("load_ok", ""),
                "tokens_per_frame": raw.get("tokens_per_frame", ""),
                "sampled_query_count": raw.get("sampled_query_count", ""),
                "payload_has_stage_c_seed_trace": raw.get("has_stage_c_seed_trace", ""),
                "payload_component_instance_masklet_key_count": raw.get("component_instance_masklet_key_count", ""),
                "fresh_trace_direct_masklet_payload_materialized": fresh_direct_masklet,
                "fresh_trace_direct_seed_payload_materialized": fresh_direct_seed,
                "fresh_trace_strict_current_cache_component_candidate": strict_current_cache_component,
                "fresh_trace_strict_current_cache_seed_candidate": strict_current_cache_seed,
                "fresh_trace_same_masklet_true_count": fresh.get(
                    "current_Q_to_cache_K_topk_same_stage_c_masklet_instance_idx_true_count", ""
                ),
                "fresh_trace_same_seed_true_count": fresh.get(
                    "current_Q_to_cache_K_topk_same_stage_c_seed_global_track_idx_true_count", ""
                ),
                "fresh_trace_ttt_anchor_identity_available": ttt_anchor_identity,
                "fullquery_direct_witness_seed_count": fullquery.get("fullquery_direct_witness_seed_count", ""),
                "fullquery_same_masklet_instance_topk_cell_count": fullquery.get(
                    "fullquery_same_masklet_instance_topk_cell_count", ""
                ),
                "fullquery_strict_identity_candidate": fullquery_strict_identity,
                "fullquery_nonproxy_current_support_candidate": fullquery_nonproxy_support,
                "fullquery_stable_lifecycle_materialized": fullquery_stable_lifecycle,
                "fullquery_action_ready_query_head_local_available": fullquery_query_head_local,
                "fullquery_action_ready_provider_pass": fullquery_action_ready_pass,
                "payload_token_index_key_count": raw.get("token_index_key_count", ""),
                "materialization_blocker": first_blocker if fresh_direct_masklet else raw.get("materialization_blocker", first_blocker),
                "runtime_action_allowed": False,
            }
        )
        provider_rows.append(
            {
                "case_id": case_id,
                "seq": base.get("seq", ""),
                "chunk_prev": base.get("prev_chunk", ""),
                "chunk_curr": base.get("curr_chunk", ""),
                "frame_idx": "",
                "query_token_id": "",
                "source_token_id": "",
                "query_head": "",
                "layer": "",
                "seed_global_track_idx": "",
                "masklet_id": "",
                "component_id": "",
                "semantic_label_id": "",
                "semantic_label_name": "",
                "semantic_role": "",
                "identity_resolution_level": resolution,
                "proxy_only": proxy_only,
                "same_instance_current_cache": strict_identity and nonproxy_support,
                "same_instance_current_write": fullquery_stable_lifecycle,
                "current_support": nonproxy_support,
                "scale_observability": to_int(join.get("true_scale_observable_anchor_count", 0)) > 0,
                "same_space_residual": "",
                "stage_c_seed_continuity": base.get("stage_c_seed_continuity", ""),
                "topk_hit": join.get("external_token_masklet_topk_best_frac", ""),
                "query_hit": join.get("external_token_masklet_query_best_frac", ""),
                "route_mass": "",
                "claim_level": "artifact_backed_case_level_provider_audit_no_runtime",
                "first_blocker": first_blocker,
            }
        )
        case_pass = action_ready_provider_join and effective_raw_has_masklet
        case_rows.append(
            {
                "case_id": case_id,
                "seq": base.get("seq", ""),
                "case_role": classify_case(case_id),
                "selected_action_case": is_selected,
                "raw_sidecar_available": bool(raw) or bool(fresh),
                "raw_has_component_instance_masklet_keys": raw_has_masklet,
                "fresh_trace_loaded": to_bool(fresh.get("load_ok", False)),
                "fresh_trace_direct_masklet_payload_materialized": fresh_direct_masklet,
                "fresh_trace_direct_seed_payload_materialized": fresh_direct_seed,
                "strict_current_cache_component_candidate": strict_current_cache_component,
                "strict_current_cache_seed_candidate": strict_current_cache_seed,
                "fullquery_strict_identity_candidate": fullquery_strict_identity,
                "fullquery_nonproxy_current_support_candidate": fullquery_nonproxy_support,
                "fullquery_stable_lifecycle_materialized": fullquery_stable_lifecycle,
                "fullquery_action_ready_query_head_local_available": fullquery_query_head_local,
                "fullquery_action_ready_provider_pass": fullquery_action_ready_pass,
                "effective_raw_has_masklet_identity_payload": effective_raw_has_masklet,
                "ttt_anchor_identity_available": ttt_anchor_identity,
                "old_ladder_strict_identity_candidate": old_ladder_strict_identity,
                "old_ladder_nonproxy_current_support": old_ladder_nonproxy_support,
                "strict_identity_candidate": strict_identity,
                "nonproxy_current_support": nonproxy_support,
                "semantic_class_fallback": semantic_fallback,
                "proxy_only": proxy_only,
                "provider_join_pass": case_pass,
                "first_blocker": first_blocker,
            }
        )
        if is_selected and not case_pass:
            failure_rows.append(
                {
                    "case_id": case_id,
                    "case_role": classify_case(case_id),
                    "failure_reason": first_blocker,
                    "raw_sidecar_available": bool(raw) or bool(fresh),
                    "fresh_trace_direct_masklet_payload_materialized": fresh_direct_masklet,
                    "strict_current_cache_component_candidate": strict_current_cache_component,
                    "fullquery_strict_identity_candidate": fullquery_strict_identity,
                    "fullquery_nonproxy_current_support_candidate": fullquery_nonproxy_support,
                    "fullquery_stable_lifecycle_materialized": fullquery_stable_lifecycle,
                    "fullquery_action_ready_query_head_local_available": fullquery_query_head_local,
                    "join_row_available": bool(join),
                    "current_universe_row_available": bool(current),
                    "repair_route": (
                        "fresh direct masklet payload is not enough; action-ready strict identity/nonproxy current-support "
                        "join plus provider controls and query/head-local carrier are required"
                    ),
                }
            )

    selected_rows = [r for r in case_rows if r["selected_action_case"]]
    selected_count = len(selected_rows)
    selected_audited = sum(1 for case_id in SELECTED_ACTION_CASES if case_id in focused)
    strict_identity_count = sum(1 for r in selected_rows if r["strict_identity_candidate"])
    strict_current_cache_component_count = sum(
        1 for r in selected_rows if r["strict_current_cache_component_candidate"]
    )
    strict_current_cache_seed_count = sum(1 for r in selected_rows if r["strict_current_cache_seed_candidate"])
    nonproxy_count = sum(1 for r in selected_rows if r["nonproxy_current_support"])
    fallback_count = sum(1 for r in selected_rows if r["semantic_class_fallback"])
    proxy_only_count = sum(1 for r in selected_rows if r["proxy_only"])
    raw_missing_count = sum(
        1
        for r in selected_rows
        if not to_bool(r["fresh_trace_direct_masklet_payload_materialized"])
        and not to_bool(r["raw_has_component_instance_masklet_keys"])
    )
    fresh_direct_masklet_count = sum(1 for r in selected_rows if r["fresh_trace_direct_masklet_payload_materialized"])
    fresh_direct_seed_count = sum(1 for r in selected_rows if r["fresh_trace_direct_seed_payload_materialized"])
    ttt_anchor_identity_count = sum(1 for r in selected_rows if r["ttt_anchor_identity_available"])
    fullquery_strict_identity_count = sum(1 for r in selected_rows if r["fullquery_strict_identity_candidate"])
    fullquery_nonproxy_count = sum(1 for r in selected_rows if r["fullquery_nonproxy_current_support_candidate"])
    fullquery_stable_lifecycle_count = sum(1 for r in selected_rows if r["fullquery_stable_lifecycle_materialized"])
    fullquery_query_head_local_count = sum(
        1 for r in selected_rows if r["fullquery_action_ready_query_head_local_available"]
    )
    fullquery_action_ready_count = sum(1 for r in selected_rows if r["fullquery_action_ready_provider_pass"])
    provider_pass_count = sum(1 for r in selected_rows if r["provider_join_pass"])

    selected_action_case_coverage = selected_audited / len(SELECTED_ACTION_CASES)
    strict_identity_frac = strict_identity_count / selected_count if selected_count else 0.0
    strict_current_cache_component_frac = (
        strict_current_cache_component_count / selected_count if selected_count else 0.0
    )
    strict_current_cache_seed_frac = strict_current_cache_seed_count / selected_count if selected_count else 0.0
    nonproxy_frac = nonproxy_count / selected_count if selected_count else 0.0
    fallback_frac = fallback_count / selected_count if selected_count else 0.0
    proxy_only_frac = proxy_only_count / selected_count if selected_count else 0.0
    fresh_direct_masklet_frac = fresh_direct_masklet_count / selected_count if selected_count else 0.0
    fresh_direct_seed_frac = fresh_direct_seed_count / selected_count if selected_count else 0.0
    fullquery_strict_identity_frac = fullquery_strict_identity_count / selected_count if selected_count else 0.0
    fullquery_nonproxy_frac = fullquery_nonproxy_count / selected_count if selected_count else 0.0
    provider_failure_rate = 1.0 - (provider_pass_count / selected_count if selected_count else 0.0)

    query_head_random_control_available = bool(
        stage4.get("a5_control_test_ready", False)
        or stage4.get("a5_v54_provider_control_ready", False)
        or exact_control_summary.get("query_head_random_control_available", False)
    )
    anchor_id_rotation_control_available = bool(
        stage4.get("a5_v54_provider_control_ready", False)
        or exact_control_summary.get("anchor_id_rotation_control_available", False)
    )
    provider_pass = (
        selected_action_case_coverage >= 0.80
        and strict_identity_frac >= 0.80
        and nonproxy_frac >= 0.80
        and fallback_frac <= 0.20
        and raw_missing_count == 0
        and query_head_random_control_available
        and anchor_id_rotation_control_available
        and provider_failure_rate <= 0.20
    )

    control_rows = [
        {
            "control_name": "anchor_id_rotation",
            "available": anchor_id_rotation_control_available,
            "source": "v103 stage4 a5_v54_provider_control_ready",
            "runtime_action_allowed": False,
        },
        {
            "control_name": "semantic_label_rotation",
            "available": bool(read_json(paths["stage3_summary"]).get("semantic_increment_pass", False)),
            "source": "v103 Stage3 semantic rotation/increment audit; diagnostic only",
            "runtime_action_allowed": False,
        },
        {
            "control_name": "query_head_random",
            "available": query_head_random_control_available,
            "source": "v103 A5 hook contract/control readiness",
            "runtime_action_allowed": False,
        },
        {"control_name": "same_count_random", "available": True, "source": "v103 oracle/simulator same-count random", "runtime_action_allowed": False},
        {"control_name": "same_masklet_wrong_chunk", "available": False, "source": "not materialized with strict raw sidecar", "runtime_action_allowed": False},
        {"control_name": "same_semantic_wrong_instance", "available": False, "source": "semantic-class fallback cannot form wrong-instance control", "runtime_action_allowed": False},
    ]

    write_csv(out / "selected11_fresh_trace_payload_rows.csv", fresh_trace_rows)
    write_json(out / "selected11_fresh_trace_payload_summary.json", fresh_trace_summary)
    write_csv(out / "selected11_no_action_parity_rows.csv", parity_rows)
    write_json(out / "selected11_no_action_parity_summary.json", parity_summary)
    write_csv(out / "selected11_deep_provider_blocker_rows.csv", deep_blocker_rows)
    write_json(out / "selected11_deep_provider_blocker_summary.json", deep_blocker_summary)
    write_csv(out / "stable_lifecycle_policy_repair_attempt_rows.csv", stable_policy_repair_rows)
    write_csv(out / "stable_lifecycle_policy_repair_attempt_rule_rows.csv", stable_policy_repair_rule_rows)
    write_json(out / "stable_lifecycle_policy_repair_attempt_summary.json", stable_policy_repair_summary)
    write_json(out / "v103_lifecycle_payload_closure_import_summary.json", lifecycle_payload_closure_summary)
    write_csv(out / "row_level_current_support_carrier_repair_attempt_rows.csv", row_level_repair_rows)
    write_csv(out / "row_level_current_support_carrier_repair_attempt_rule_rows.csv", row_level_repair_rule_rows)
    write_json(out / "row_level_current_support_carrier_repair_attempt_summary.json", row_level_repair_summary)
    write_csv(out / "fresh_q128_tracked_lifecycle_repair_attempt_rows.csv", fresh_q128_lifecycle_rows)
    write_csv(
        out / "fresh_q128_tracked_lifecycle_repair_attempt_rule_rows.csv",
        fresh_q128_lifecycle_rule_rows,
    )
    write_json(
        out / "fresh_q128_tracked_lifecycle_repair_attempt_summary.json",
        fresh_q128_lifecycle_summary,
    )
    write_csv(out / "a5_query_head_local_trace_repair_attempt_rows.csv", a5_trace_rows)
    write_csv(out / "a5_query_head_local_trace_repair_attempt_rule_rows.csv", a5_trace_rule_rows)
    write_json(out / "a5_query_head_local_trace_repair_attempt_summary.json", a5_trace_summary)
    write_csv(out / "a5_same_masklet_query_head_local_trace_repair_attempt_rows.csv", a5_same_masklet_rows)
    write_csv(
        out / "a5_same_masklet_query_head_local_trace_repair_attempt_rule_rows.csv",
        a5_same_masklet_rule_rows,
    )
    write_json(
        out / "a5_same_masklet_query_head_local_trace_repair_attempt_summary.json",
        a5_same_masklet_summary,
    )
    write_csv(out / "a5_direct_match_threshold_repair_attempt_rows.csv", a5_threshold_rows)
    write_json(out / "a5_direct_match_threshold_repair_attempt_summary.json", a5_threshold_summary)
    write_csv(out / "a5_sampled_exact_edge_materialization_rows.csv", a5_sampled_edge_rows)
    write_csv(out / "a5_sampled_exact_edge_materialization_case_rows.csv", a5_sampled_edge_case_rows)
    write_json(out / "a5_sampled_exact_edge_materialization_summary.json", a5_sampled_edge_summary)
    write_csv(out / "a5_q4096raw_exact_edge_materialization_rows.csv", a5_q4096_edge_rows)
    write_csv(out / "a5_q4096raw_exact_edge_materialization_case_rows.csv", a5_q4096_edge_case_rows)
    write_json(out / "a5_q4096raw_exact_edge_materialization_summary.json", a5_q4096_edge_summary)
    write_csv(out / "tracked_local_provider_demotion_case_rows.csv", demotion_case_rows)
    write_csv(out / "tracked_local_provider_demotion_rule_rows.csv", demotion_rule_rows)
    write_json(out / "tracked_local_provider_demotion_summary.json", demotion_summary)
    write_csv(out / "provider_control_payload_feasibility_rows.csv", control_payload_rows)
    write_json(out / "provider_control_payload_feasibility_summary.json", control_payload_summary)
    write_csv(out / "provider_support_row_control_feasibility_rows.csv", support_control_rows)
    write_json(out / "provider_support_row_control_feasibility_summary.json", support_control_summary)
    write_csv(out / "support_row_query_head_witness_rows.csv", query_head_witness_rows)
    write_json(out / "support_row_query_head_witness_summary.json", query_head_witness_summary)
    write_csv(out / "exact_support_row_witness_set_rows.csv", exact_witness_rows)
    write_json(out / "exact_support_row_witness_set_summary.json", exact_witness_summary)
    write_csv(out / "exact_provider_control_diagnostic_rows.csv", exact_control_rows)
    write_json(out / "exact_provider_control_diagnostic_summary.json", exact_control_summary)
    local_provider_rows: list[dict[str, Any]] = []
    for row in deep_blocker_rows:
        candidate = (
            to_int(row.get("fullquery_direct_witness_seed_count", 0)) > 0
            and to_int(row.get("counterfactual_tracked_instance_anchor_query_head_count", 0)) > 0
        )
        case_role = row.get("case_role", "")
        if candidate and case_role == "safe_good_control":
            blocker = "tracked_local_provider_hits_safe_good_control"
        elif candidate and to_int(row.get("raw_tracked_candidate_stable_gated_cell_count", 0)) <= 0:
            blocker = "tracked_local_provider_candidate_lacks_formal_stable_lifecycle"
        elif not candidate:
            blocker = "no_tracked_local_provider_direct_query_head_candidate"
        else:
            blocker = "requires_provider_controls_before_runtime_action"
        local_provider_rows.append(
            {
                "case_id": row.get("case_id", ""),
                "case_role": case_role,
                "tracked_local_provider_candidate": candidate,
                "fullquery_direct_witness_seed_count": row.get("fullquery_direct_witness_seed_count", 0),
                "fullquery_same_masklet_instance_topk_cell_count": row.get(
                    "fullquery_same_masklet_instance_topk_cell_count", 0
                ),
                "counterfactual_tracked_instance_anchor_query_head_count": row.get(
                    "counterfactual_tracked_instance_anchor_query_head_count", 0
                ),
                "formal_stable_lifecycle_available": False,
                "proposed_eligibility_state": "LOCAL_REGISTRATION_ONLY_CANDIDATE" if candidate else "NO_LOCAL_PROVIDER",
                "runtime_action_allowed": False,
                "first_blocker": blocker,
            }
        )
    local_provider_candidate_count = sum(1 for row in local_provider_rows if row["tracked_local_provider_candidate"])
    local_provider_safe_good_hit_count = sum(
        1
        for row in local_provider_rows
        if row["tracked_local_provider_candidate"] and row["case_role"] == "safe_good_control"
    )
    local_provider_positive_hit_count = sum(
        1
        for row in local_provider_rows
        if row["tracked_local_provider_candidate"] and row["case_role"] in {"strict_positive", "exploratory_positive"}
    )
    local_provider_summary = {
        "schema": "acl2_v104_selected11_tracked_local_provider_feasibility_summary_v1",
        "selected_case_count": len(SELECTED_ACTION_CASES),
        "tracked_local_provider_candidate_case_count": local_provider_candidate_count,
        "tracked_local_provider_positive_hit_case_count": local_provider_positive_hit_count,
        "tracked_local_provider_safe_good_hit_case_count": local_provider_safe_good_hit_count,
        "tracked_local_provider_rule_safe_for_action": False,
        "runtime_action_allowed": False,
        "interpretation": (
            "A separate tracked-instance local provider path has diagnostic signal, but it also hits a selected "
            "safe-good control and still lacks formal stable lifecycle/provider controls. It can only be treated as "
            "LOCAL_REGISTRATION_ONLY diagnostic evidence, not a scale/gauge memory action provider."
        ),
    }
    write_csv(out / "selected11_tracked_local_provider_feasibility_rows.csv", local_provider_rows)
    write_json(out / "selected11_tracked_local_provider_feasibility_summary.json", local_provider_summary)
    selected11_fullquery_summary = {
        "schema": "acl2_v104_selected11_fullquery_action_ready_ladder_summary_v1",
        "source_rows": rel(paths["fullquery_ladder_rows"]),
        "source_summary": rel(paths["fullquery_ladder_summary"]),
        "source_case_count": fullquery_imported_summary.get("case_count"),
        "selected_case_count": selected_count,
        "fullquery_strict_identity_candidate_case_count": fullquery_strict_identity_count,
        "fullquery_strict_identity_candidate_frac": fullquery_strict_identity_frac,
        "fullquery_nonproxy_current_support_candidate_case_count": fullquery_nonproxy_count,
        "fullquery_nonproxy_current_support_candidate_frac": fullquery_nonproxy_frac,
        "fullquery_stable_lifecycle_materialized_case_count": fullquery_stable_lifecycle_count,
        "fullquery_action_ready_query_head_local_case_count": fullquery_query_head_local_count,
        "fullquery_action_ready_provider_pass_case_count": fullquery_action_ready_count,
        "formal_provider_chain_pass": formal_provider_chain_summary.get(
            "tracked_instance_query_soft_formal_provider_chain_provider_pass", False
        ),
        "direct_match_control_provider_pass": direct_match_control_summary.get(
            "tracked_instance_query_soft_direct_match_control_provider_pass", False
        ),
        "direct_match_control_metric_good_control_candidate_pass": direct_match_control_summary.get(
            "tracked_instance_query_soft_direct_match_control_metric_good_control_candidate_pass", False
        ),
        "direct_match_control_first_blocker": direct_match_control_summary.get(
            "tracked_instance_query_soft_direct_match_control_first_blocker", ""
        ),
        "runtime_action_allowed": False,
        "materialization_blocker": (
            "full-query ladder finds limited strict/nonproxy candidates but zero stable lifecycle/action-ready "
            "query-head-local provider pass on the selected action set"
        ),
    }
    write_csv(out / "selected11_fullquery_action_ready_ladder_rows.csv", fullquery_selected_rows)
    write_json(out / "selected11_fullquery_action_ready_ladder_summary.json", selected11_fullquery_summary)
    write_csv(out / "raw_sidecar_trace_manifest.csv", manifest_rows)
    write_csv(out / "provider_token_rows.csv", provider_rows)
    write_csv(out / "provider_case_summary.csv", case_rows)
    write_csv(out / "current_cache_write_identity_rows.csv", provider_rows)
    write_csv(out / "provider_join_failure_rows.csv", failure_rows)
    write_csv(out / "provider_control_rows.csv", control_rows)

    identity_summary = {
        "schema": "acl2_v104_current_cache_write_identity_summary_v1",
        "selected_action_case_count": selected_count,
        "provider_pass_case_count": provider_pass_count,
        "strict_identity_candidate_case_count": strict_identity_count,
        "fresh_direct_masklet_payload_case_count": fresh_direct_masklet_count,
        "fresh_direct_seed_payload_case_count": fresh_direct_seed_count,
        "strict_current_cache_component_candidate_case_count": strict_current_cache_component_count,
        "strict_current_cache_seed_candidate_case_count": strict_current_cache_seed_count,
        "ttt_anchor_identity_available_case_count": ttt_anchor_identity_count,
        "fullquery_strict_identity_candidate_case_count": fullquery_strict_identity_count,
        "fullquery_nonproxy_current_support_candidate_case_count": fullquery_nonproxy_count,
        "fullquery_stable_lifecycle_materialized_case_count": fullquery_stable_lifecycle_count,
        "fullquery_action_ready_query_head_local_case_count": fullquery_query_head_local_count,
        "fullquery_action_ready_provider_pass_case_count": fullquery_action_ready_count,
        "nonproxy_current_support_case_count": nonproxy_count,
        "semantic_class_fallback_case_count": fallback_count,
        "proxy_only_case_count": proxy_only_count,
        "provider_join_failure_rate": provider_failure_rate,
        "provider_join_pass": provider_pass,
        "runtime_action_allowed": False,
    }
    stage4_runtime_controls_available = bool(
        stage4.get("a5_control_test_ready", False) and stage4.get("a5_v54_provider_control_ready", False)
    )
    exact_provider_controls_available = bool(exact_control_summary.get("provider_control_diagnostic_pass", False))
    provider_controls_available = bool(query_head_random_control_available and anchor_id_rotation_control_available)
    control_claim_level = (
        "stage1_exact_provider_control_diagnostic_no_runtime"
        if exact_provider_controls_available
        else "stage4_or_legacy_provider_control_artifact"
        if provider_controls_available
        else "provider_controls_unavailable"
    )
    control_summary = {
        "schema": "acl2_v104_provider_control_summary_v1",
        "query_head_random_control_available": query_head_random_control_available,
        "anchor_id_rotation_control_available": anchor_id_rotation_control_available,
        "same_count_random_control_available": True,
        "provider_controls_available": provider_controls_available,
        "provider_control_claim_level": control_claim_level,
        "exact_provider_control_diagnostic_pass": exact_provider_controls_available,
        "stage4_runtime_action_controls_available": stage4_runtime_controls_available,
        "runtime_action_allowed": False,
    }
    write_json(out / "current_cache_write_identity_summary.json", identity_summary)
    write_json(out / "provider_control_summary.json", control_summary)

    stage4_outputs = stage4.get("outputs", {}) if isinstance(stage4.get("outputs", {}), dict) else {}
    control_gap_summary = {
        "schema": "acl2_v104_provider_control_gap_audit_v1",
        "source_stage4_summary": rel(paths["stage4_summary"]),
        "source_a5_hook_contract_report": stage4_outputs.get("a5_hook_contract_report", ""),
        "source_a5_v54_action_guard_report": stage4_outputs.get("a5_v54_action_guard_report", ""),
        "source_fullquery_control_gap_report": rel(
            V103
            / "stage7_read_current_support_provider_integration/tracked_masklet_instance_fullquery_trace_focused19_goodctrl_20260701_122615/control_gap_report.md"
        ),
        "source_querysoft_control_gap_report": rel(
            V103
            / "stage7_read_current_support_provider_integration/tracked_instance_query_soft_trace_focused19_same_masklet_20260701_1430/control_gap_report.md"
        ),
        "a5_source_hook_contract_observed": stage4.get("a5_source_hook_contract_observed", False),
        "a5_source_hook_body_present": stage4.get("a5_source_hook_body_present", False),
        "a5_hmc_runtime_wiring_present": stage4.get("a5_hmc_runtime_wiring_present", False),
        "a5_cli_runtime_wiring_present": stage4.get("a5_cli_runtime_wiring_present", False),
        "a5_action_body_contract_ready": stage4.get("a5_action_body_contract_ready", False),
        "a5_control_test_ready": stage4.get("a5_control_test_ready", False),
        "a5_v54_action_body_contract_ready": stage4.get("a5_v54_action_body_contract_ready", False),
        "a5_v54_provider_control_ready": stage4.get("a5_v54_provider_control_ready", False),
        "query_head_random_control_available": query_head_random_control_available,
        "anchor_id_rotation_control_available": anchor_id_rotation_control_available,
        "provider_controls_available": provider_controls_available,
        "provider_control_claim_level": control_claim_level,
        "exact_provider_control_diagnostic_pass": exact_provider_controls_available,
        "exact_provider_control_summary": rel(out / "exact_provider_control_diagnostic_summary.json"),
        "stage4_runtime_action_controls_available": stage4_runtime_controls_available,
        "stage4_runtime_action_controls_blocker": "a5_control_test_ready_and_a5_v54_provider_control_ready_are_false",
        "runtime_action_allowed": False,
        "first_blocker": (
            "stage1_provider_controls_diagnostic_available_but_stage4_runtime_controls_and_provider_coverage_blocked"
            if provider_controls_available
            else "a5_hook_exists_but_control_test_and_provider_control_ready_false"
        ),
        "interpretation": (
            "Exact witness-set diagnostics now provide Stage1 no-action query-head and anchor-seed provider controls "
            "when the corresponding fields are true. The query/head-local hook and default-off tracked-instance guard "
            "also exist, but v103 artifacts still report control_test_ready=false and provider_control_ready=false for "
            "Stage4 runtime action. These controls therefore remove only the provider-control availability blocker; "
            "they do not evaluate L3/runtime action effects, and Stage4 remains blocked by strict-provider coverage, "
            "lifecycle/action carrier, and action-entry gates."
        ),
    }
    control_gap_rows = [
        {
            "control_name": "query_head_random",
            "required_for_stage1_pass": True,
            "available": query_head_random_control_available,
            "source_field": "exact_provider_control_diagnostic_summary.query_head_random_control_available",
            "source_value": exact_control_summary.get("query_head_random_control_available", False),
            "stage4_runtime_control_ready": stage4.get("a5_control_test_ready", False),
            "claim_level": control_claim_level if query_head_random_control_available else "missing",
            "first_blocker": (
                "stage4_runtime_query_head_action_control_not_ready"
                if query_head_random_control_available
                else "no_query_head_random_control_artifact"
            ),
            "runtime_action_allowed": False,
        },
        {
            "control_name": "anchor_id_rotation",
            "required_for_stage1_pass": True,
            "available": anchor_id_rotation_control_available,
            "source_field": "exact_provider_control_diagnostic_summary.anchor_id_rotation_control_available",
            "source_value": exact_control_summary.get("anchor_id_rotation_control_available", False),
            "stage4_runtime_control_ready": stage4.get("a5_v54_provider_control_ready", False),
            "claim_level": control_claim_level if anchor_id_rotation_control_available else "missing",
            "first_blocker": (
                "stage4_runtime_anchor_action_control_not_ready"
                if anchor_id_rotation_control_available
                else "no_anchor_id_rotation_provider_control_artifact"
            ),
            "runtime_action_allowed": False,
        },
        {
            "control_name": "same_count_random",
            "required_for_stage1_pass": False,
            "available": True,
            "source_field": "same-count random simulator/control rows",
            "source_value": True,
            "stage4_runtime_control_ready": False,
            "claim_level": "legacy_diagnostic_control",
            "first_blocker": "available_but_insufficient_without_query_head_and_anchor_controls",
            "runtime_action_allowed": False,
        },
    ]
    write_csv(out / "provider_control_gap_rows.csv", control_gap_rows)
    write_json(out / "provider_control_gap_summary.json", control_gap_summary)
    write_text(
        out / "provider_control_gap_audit.md",
        "# Stage 1 Provider Control Gap Audit\n\n"
        "Exact witness-set diagnostics now provide no-action Stage 1 provider controls when "
        "`query_head_random_control_available=true` and `anchor_id_rotation_control_available=true`. These controls "
        "rotate query heads and anchor seeds on serialized witness cells and do not run Stage 4 action.\n\n"
        "Source-level query/head-local hooks and the default-off tracked-instance guard are present, but v103 reports "
        "`a5_control_test_ready=false` and `a5_v54_provider_control_ready=false` for runtime action. The focused "
        "full-query and query-soft trace roots are READ_NO_ACTION diagnostics; their control-gap reports state that "
        "no L3/runtime action effect is evaluated there.\n\n"
        "Result: the Stage 1 provider-control availability blocker is repaired at no-action diagnostic level only. "
        "Stage 4 runtime remains unauthorized until strict-provider coverage, stable/lifecycle carrier, "
        "query/head-local action carrier, and action-entry metrics pass together.\n",
    )

    schema_report = f"""# Stage 1 Raw Sidecar Schema Report

Imported source:

```text
{rel(paths["raw_payload_rows"])}
```

Fresh selected11 trace source:

```text
{rel(paths["selected11_fresh_trace_root"])}
```

Observed v103/v104-reused and fresh trace facts:

```text
selected_action_case_count={selected_count}
raw_sidecar_missing_case_count={raw_missing_count}
old_payload_with_component_instance_masklet_keys_case_count={sum(1 for r in selected_rows if r["raw_has_component_instance_masklet_keys"])}
fresh_direct_masklet_payload_case_count={fresh_direct_masklet_count}
fresh_direct_seed_payload_case_count={fresh_direct_seed_count}
strict_current_cache_component_candidate_case_count={strict_current_cache_component_count}
strict_current_cache_seed_candidate_case_count={strict_current_cache_seed_count}
ttt_anchor_identity_available_case_count={ttt_anchor_identity_count}
fullquery_strict_identity_candidate_case_count={fullquery_strict_identity_count}
fullquery_nonproxy_current_support_candidate_case_count={fullquery_nonproxy_count}
fullquery_stable_lifecycle_materialized_case_count={fullquery_stable_lifecycle_count}
fullquery_action_ready_query_head_local_case_count={fullquery_query_head_local_count}
fullquery_action_ready_provider_pass_case_count={fullquery_action_ready_count}
strict_identity_candidate_case_count={strict_identity_count}
nonproxy_current_support_case_count={nonproxy_count}
semantic_class_fallback_case_count={fallback_count}
provider_pass_case_count={provider_pass_count}
```

Interpretation:

The older imported raw payload table did not carry component/instance/masklet keys, but the selected11 fresh READ_NO_ACTION traces now materialize direct Stage-C masklet and seed payload tensors. This repairs the raw-payload materialization part of the Stage 1 blocker. The v103 full-query ladder still finds only limited selected-case strict/nonproxy candidates and zero action-ready query/head-local provider passes, so Stage 4 runtime action remains unauthorized.
"""
    write_text(out / "raw_sidecar_schema_report.md", schema_report)

    repair_report = f"""# Stage 1 Provider Repair Attempts

This report follows the v104 fail-forward checklist.

1. Patch token order / special-token offset:
   v103 raw rows expose `tokens_per_frame` and sampled-query/top-k tensor shapes, and the runtime masklet projection smoke passed. No evidence was found that token-order alone explains the blocker.

2. Frame offset / chunk index:
   raw rows are keyed by case/chunk and load successfully where present. The blocker persists after current-universe audits.

3. Stage-C seed_global_track_idx:
   Stage-C seed traces are present in raw payloads, so the seed id is not simply lost. The blocker is that seed-level support does not become strict instance/current-support evidence.

4. Masklet id only in sparse masklet file:
   repaired/validated for the selected11 diagnostic trace set. The fresh READ_NO_ACTION traces now carry sampled-query and top-k cache Stage-C masklet/seed tensors plus same-masklet/same-seed booleans. This is logged in `selected11_fresh_trace_payload_rows.csv`.

5. Current query/cache universe mismatch:
   q128 fresh traces show sampled current/cache direct same-masklet and same-seed candidates in all selected cases. The full-query ladder is stricter and finds selected strict/nonproxy candidates only in {fullquery_strict_identity_count}/{selected_count} cases. This remains a coverage/provider-join blocker, not a raw-payload-materialization blocker.

6. TTT write anchor identity:
   TTT/stable-anchor keys exist, and q128 traces carry anchor-hit/same-masklet/same-seed tensors. However full-query selected cases have stable lifecycle materialized in {fullquery_stable_lifecycle_count}/{selected_count}, action-ready query/head-local in {fullquery_query_head_local_count}/{selected_count}, and action-ready provider pass in {fullquery_action_ready_count}/{selected_count}.

7. No-action trace instrumentation parity:
   validated for selected11 by a trace-vs-no-trace READ_NO_ACTION baseline rooted at `{rel(paths["selected11_no_trace_pose_sha_parity_root"])}`. The helper was repaired to skip pose-text comment lines, reuse existing successful outputs, and schedule pending jobs without same-GPU overlap. Final parity summary reports pose SHA equal in {parity_summary.get("pose_sha_equal_case_count", 0)}/{selected_count}, numeric diff equal in {parity_summary.get("pose_numeric_equal_case_count", 0)}/{selected_count}, failed jobs={parity_summary.get("no_trace_pose_sha_parity_failed_job_count", "")}, and `plan_pose_sha_gate_pass={parity_summary.get("plan_pose_sha_gate_pass", False)}`. This validates the diagnostic trace instrumentation only; it is not an action success.

8. Deep provider blocker attribution:
   connected v103 full-query support rows, stable-anchor gate source-type rows, raw tracked candidate sidecar rows, and formal provider-chain rows into selected11. The deep blocker summary reports full-query direct witness cases={deep_blocker_summary.get("fullquery_direct_witness_case_count", 0)}/{selected_count}, selected stable-gated tracked cells={deep_blocker_summary.get("selected_stable_source_stable_gated_tracked_cell_count", 0)}, selected lifecycle tracked rows={deep_blocker_summary.get("selected_stable_source_lifecycle_tracked_row_count", 0)}, selected raw tracked candidate stable-gated cells={deep_blocker_summary.get("selected_raw_tracked_candidate_stable_gated_cell_count", 0)}, selected raw tracked candidate lifecycle rows={deep_blocker_summary.get("selected_raw_tracked_candidate_lifecycle_row_count", 0)}, and formal provider pass={deep_blocker_summary.get("formal_provider_chain_provider_pass", False)}. This confirms the remaining blocker is formal stable/lifecycle action carrier, not q128 sampling.

9. Separate tracked-instance local-provider feasibility:
   tested as diagnostic-only from the deep blocker rows. It finds tracked local-provider candidates in {local_provider_candidate_count}/{selected_count} selected cases, with positive hits={local_provider_positive_hit_count} and safe-good hits={local_provider_safe_good_hit_count}. Because a safe-good control is hit and formal stable lifecycle/provider controls remain unavailable, this route can only be `LOCAL_REGISTRATION_ONLY_CANDIDATE`; it is not safe to promote to scale/gauge action.

10. Focused full-query demotion rule audit:
   tested direct-seed, same-masklet-cell, and query-head-cell thresholds against focused full-query rows and v103 direct-match controls. The only no-safe-good direct-seed rule uses threshold={demotion_summary.get("direct_seed_demotion_threshold", "")} and selects {demotion_summary.get("demotion_rule_selected_case_ids", [])}; it is single-case diagnostic-only with provider_pass={demotion_summary.get("demotion_rule_provider_pass", False)}. Cell/query-head count rules are rejected because focused safe-good `{demotion_summary.get("unsafe_cell_strength_counterexample_case_id", "")}` has same-masklet count={demotion_summary.get("unsafe_cell_strength_counterexample_same_masklet_count", "")} and query-head count={demotion_summary.get("unsafe_cell_strength_counterexample_query_head_count", "")}, stronger than the strict-positive local signal. v103 query-soft best rule `{demotion_summary.get("v103_querysoft_best_rule_name", "")}` selected {demotion_summary.get("v103_querysoft_best_selected_case_ids", [])}, but v104 full-query direct witness is missing for {demotion_summary.get("v103_querysoft_best_selected_cases_missing_v104_fullquery_direct_witness", [])}; this blocks promotion of the old query-soft metric-good candidate.

11. Provider-control payload feasibility:
   loaded focused full-query READ_NO_ACTION trace payloads from `{rel(paths["fullquery_trace_root"])}`. Per-head tensors are present in {control_payload_summary.get("per_head_tensor_available_case_count", 0)} cases, but naive raw tensor controls are invalid: query-shift surrogate passes {control_payload_summary.get("naive_query_shift_pass_ge_threshold_case_count", 0)} cases with safe-good hits={control_payload_summary.get("naive_query_shift_safe_good_hit_case_count", 0)}, anchor-rotation surrogate passes {control_payload_summary.get("naive_anchor_rotation_pass_ge_threshold_case_count", 0)} cases with safe-good hits={control_payload_summary.get("naive_anchor_rotation_safe_good_hit_case_count", 0)}, and raw tensor direct-count matches strict support-row direct-count in only {control_payload_summary.get("naive_raw_tensor_direct_count_matches_reported_case_count", 0)} cases. Therefore these surrogates cannot be claimed as query_head_random/anchor_id_rotation controls.

12. Support-row-preserving provider-control feasibility:
   repeated the control smoke while preserving the full-query support-row seed universe. Recompute mismatch cases={support_control_summary.get("support_row_recompute_mismatch_case_count", 0)}, query-shift ge-threshold cases={support_control_summary.get("query_shift_ge_threshold_case_ids", [])}, query-shift safe-good hits={support_control_summary.get("query_shift_safe_good_hit_case_count", 0)}, anchor-rotation ge-threshold cases={support_control_summary.get("anchor_rotation_ge_threshold_case_ids", [])}, anchor-rotation safe-good hits={support_control_summary.get("anchor_rotation_safe_good_hit_case_count", 0)}. This is stronger diagnostic evidence than the raw-tensor all-cell smoke, but still not a plan-qualified provider control because query-shift preserves the single-case signal and support rows do not serialize exact query/head/anchor-id witness sets or action-control margins.

13. Query/head witness materialization:
   materialized head-level witness rows from the strict support-row seed universe. Head witness rows={query_head_witness_summary.get("head_witness_row_count", 0)}, direct witness cases={query_head_witness_summary.get("direct_witness_case_ids", [])}, safe-good direct witness cases={query_head_witness_summary.get("safe_good_direct_witness_case_ids", [])}, strict-positive direct witness cases={query_head_witness_summary.get("strict_positive_direct_witness_case_ids", [])}. This improves carrier auditability, but `action_ready_query_head_local_edge` remains false.

14. Exact support-row witness-set schema:
   serialized exact direct-witness cells with query token, query head, top-k rank, cache token, anchor id, and anchor seed. Exact witness cells={exact_witness_summary.get("exact_witness_cell_count", 0)}, cases={exact_witness_summary.get("exact_witness_case_ids", [])}, safe-good cases={exact_witness_summary.get("safe_good_exact_witness_case_ids", [])}, strict-positive cases={exact_witness_summary.get("strict_positive_exact_witness_case_ids", [])}, exact anchor-rotation ge-threshold cases={exact_witness_summary.get("anchor_rotation_ge_threshold_case_ids", [])}. This creates the missing audit substrate for future controls but still does not authorize action.

15. Exact provider-control diagnostics:
   ran no-action controls on the exact witness set. Observed ge-threshold cases={exact_control_summary.get("observed_ge_threshold_case_ids", [])}; query-head rotation ge-threshold cases={exact_control_summary.get("query_head_rotation_ge_threshold_case_ids", [])}, margin={exact_control_summary.get("query_head_rotation_control_margin", 0.0)}; anchor-seed rotation ge-threshold cases={exact_control_summary.get("anchor_seed_rotation_ge_threshold_case_ids", [])}, margin={exact_control_summary.get("anchor_seed_rotation_control_margin", 0.0)}; provider_control_diagnostic_pass={exact_control_summary.get("provider_control_diagnostic_pass", False)}. This can support provider-control availability only as READ_NO_ACTION diagnostics, not Stage4 action metrics.

16. Stable/lifecycle policy repair attempt:
   tested counterfactual policy repairs that would promote raw tracked candidates or direct-witness tracked anchors into stable/lifecycle evidence. Stage6 reports lifecycle_row_count={stable_policy_repair_summary.get("stage6_lifecycle_row_count", 0)} but lifecycle_tracked_row_count={stable_policy_repair_summary.get("stage6_lifecycle_tracked_row_count", 0)}, stable_gated_tracked_cell_count={stable_policy_repair_summary.get("stage6_stable_gated_tracked_cell_count", 0)}, and raw_tracked_cell_count={stable_policy_repair_summary.get("stage6_raw_tracked_cell_count", 0)}. Repair allowed rule count={stable_policy_repair_summary.get("repair_allowed_rule_count", 0)}; first_blocker={stable_policy_repair_summary.get("first_blocker", "")}. This means relaxing tracked stable/lifecycle policy is not a safe fix on current evidence.

17. Imported v103 lifecycle payload closure:
   imported the later v103 Stage7 source-type closure artifacts. Raw tensor absence is closed={lifecycle_payload_closure_summary.get("provider_policy_raw_tensor_absence_closed", False)} and source-type boundary is closed={lifecycle_payload_closure_summary.get("provider_policy_source_type_boundary_closed", False)}; candidate_count={lifecycle_payload_closure_summary.get("stuff_static_carrier_candidate_count", 0)}, current_cache_same_seed_supported_count={lifecycle_payload_closure_summary.get("stuff_static_current_cache_same_seed_supported_count", 0)}, strict_current_support_pass_count={lifecycle_payload_closure_summary.get("stuff_static_support_strict_current_support_pass_count", 0)}, strict_provider_materializable_count={lifecycle_payload_closure_summary.get("stuff_static_strict_provider_materializable_count", 0)}, action_ready_carrier_available={lifecycle_payload_closure_summary.get("stuff_static_action_ready_carrier_available", False)}. This closes the “maybe raw sidecar is just missing” branch; the remaining wall is nonproxy strict current-support plus action-ready carrier.

18. Row-level current-support/carrier repair attempt:
   compared q128 Stage-C same-seed support, full-query direct masklet witnesses, stuff/static raw-present support, and diagnostic thing-tracked lifecycle sidecar routes in one no-action audit. Candidate rows={row_level_repair_summary.get("candidate_row_count", 0)}, rule count={row_level_repair_summary.get("repair_rule_count", 0)}, allowed rule count={row_level_repair_summary.get("repair_allowed_rule_count", 0)}, first_blocker={row_level_repair_summary.get("first_blocker", "")}. This directly tests the next recommended repair direction and still finds no safe strict-support/action-carrier promotion.

19. Fresh q128 tracked-lifecycle repair attempt:
   audited the fresh v104 selected11 q128 trace payload rows `ttt_prev_tracked_instance_anchor_lifecycle_rows` directly. Candidate rows={fresh_q128_lifecycle_summary.get("candidate_row_count", 0)}, same-seed candidate cases={fresh_q128_lifecycle_summary.get("same_seed_candidate_case_count", 0)}, same-masklet candidate cases={fresh_q128_lifecycle_summary.get("same_masklet_candidate_case_count", 0)}, same-seed+same-masklet candidate cases={fresh_q128_lifecycle_summary.get("same_seed_and_masklet_candidate_case_count", 0)}, formal strict current-support cases={fresh_q128_lifecycle_summary.get("formal_strict_current_support_pass_case_count", 0)}, action-ready carrier cases={fresh_q128_lifecycle_summary.get("action_ready_carrier_case_count", 0)}, rule count={fresh_q128_lifecycle_summary.get("repair_rule_count", 0)}, allowed rule count={fresh_q128_lifecycle_summary.get("repair_allowed_rule_count", 0)}, first_blocker={fresh_q128_lifecycle_summary.get("first_blocker", "")}. This closes the fresh-trace lifecycle-row branch: broad q128 lifecycle signals exist, but they hit safe-good controls and remain diagnostic-only rather than formal strict providers.

20. A5 query/head-local trace repair attempt:
   ran a fresh selected11 q128 A5 same-seed trace-only branch rooted at `{rel(paths["selected11_a5_trace_root"])}` and validated it with trace-vs-no-trace pose parity rooted at `{rel(paths["selected11_a5_no_trace_pose_sha_parity_root"])}`. A5 swa_read trace cases={a5_trace_summary.get("case_with_swa_read_trace_count", 0)}/{selected_count}, swa_read carrier-rule-pass cases={a5_trace_summary.get("case_with_swa_read_carrier_rule_pass_count", 0)}/{selected_count}, safe-good carrier hits={a5_trace_summary.get("safe_good_swa_read_carrier_rule_pass_count", 0)}, attention/TTT-apply carrier cases={a5_trace_summary.get("attention_or_ttt_apply_carrier_case_count", 0)}, pose SHA equal cases={a5_trace_summary.get("pose_sha_equal_case_count", "")}, numeric equal cases={a5_trace_summary.get("pose_numeric_equal_case_count", "")}, parity_pass={a5_trace_summary.get("no_trace_pose_sha_parity_pass", False)}, allowed rule count={a5_trace_summary.get("repair_allowed_rule_count", 0)}, first_blocker={a5_trace_summary.get("first_blocker", "")}. This proves the A5 hook can materialize a same-seed query/head-local diagnostic carrier, but the route remains trace-only, hits safe-good controls, and does not provide a runtime action surface.

21. A5 same-masklet direct-match repair attempt:
   ran the stricter selected11 q128 A5 same-masklet trace-only branch rooted at `{rel(paths["selected11_a5_same_masklet_trace_root"])}` and recorded pose parity in `{rel(paths["selected11_a5_same_masklet_no_trace_pose_sha_parity_root"])}` using the already materialized no-trace baseline from the same-seed A5 branch. A5 same-masklet swa_read trace cases={a5_same_masklet_summary.get("case_with_swa_read_trace_count", 0)}/{selected_count}, carrier-rule-pass cases={a5_same_masklet_summary.get("case_with_swa_read_carrier_rule_pass_count", 0)}/{selected_count}, safe-good carrier hits={a5_same_masklet_summary.get("safe_good_swa_read_carrier_rule_pass_count", 0)}, attention/TTT-apply carrier cases={a5_same_masklet_summary.get("attention_or_ttt_apply_carrier_case_count", 0)}, pose SHA equal cases={a5_same_masklet_summary.get("pose_sha_equal_case_count", "")}, numeric equal cases={a5_same_masklet_summary.get("pose_numeric_equal_case_count", "")}, parity_pass={a5_same_masklet_summary.get("no_trace_pose_sha_parity_pass", False)}, allowed rule count={a5_same_masklet_summary.get("repair_allowed_rule_count", 0)}, first_blocker={a5_same_masklet_summary.get("first_blocker", "")}. This falsifies the idea that simply tightening same-seed to same-masklet repairs the A5 route: same-masklet still hits safe-good controls and remains trace-only.

22. A5 direct-match feature-threshold repair attempt:
   swept same-seed and same-masklet A5 trace features (`direct_witness_seed_count`, `selected_query_count`, `query_selected_frac`) as no-action provider candidates. Threshold rules tested={a5_threshold_summary.get("threshold_rule_count", 0)}, no-safe-good threshold rules={a5_threshold_summary.get("no_safe_good_threshold_rule_count", 0)}, allowed rule count={a5_threshold_summary.get("repair_allowed_rule_count", 0)}, best no-safe-good rule={a5_threshold_summary.get("best_no_safe_good_rule", {})}, first_blocker={a5_threshold_summary.get("first_blocker", "")}. This closes the obvious thresholding escape hatch: even when a threshold avoids selected safe-good controls, it remains trace-only and lacks formal strict current-support/action-ready query-head-local edges.

23. A5 sampled exact edge materialization:
   loaded the q128 A5 trace payload tensors and serialized sampled query/cache/anchor direct-match edges for same-seed and same-masklet modes. Edge rows={a5_sampled_edge_summary.get("edge_row_count", 0)}, sampled exact nonproxy current-support edges={a5_sampled_edge_summary.get("sampled_exact_nonproxy_current_support_edge_count", 0)}, mode:case keys with sampled exact nonproxy support={a5_sampled_edge_summary.get("case_with_sampled_exact_nonproxy_current_support_edge_count", 0)}, safe-good mode:case keys with sampled exact nonproxy support={a5_sampled_edge_summary.get("safe_good_with_sampled_exact_nonproxy_current_support_edge_count", 0)}, formal fullquery strict current-support cases={a5_sampled_edge_summary.get("formal_fullquery_strict_current_support_pass_case_count", 0)}, action-ready query/head-local cases={a5_sampled_edge_summary.get("action_ready_query_head_local_case_count", 0)}, first_blocker={a5_sampled_edge_summary.get("first_blocker", "")}. This makes the A5 sampled edge substrate auditable, but it still cannot pass Stage 1 because q128 sampled exact edges are not fullquery formal providers and also cover safe-good controls.

24. A5 q4096 raw-edge materialization attempt:
   first attempted selected5 A5 q4096 with query-soft attention_mass_max_queries=4096; that failed with CUDA OOM in `_append_source_soft_mass_stats`, although raw transport trace files were written. The repaired attempt kept raw_transport max_queries=4096 but restored query-soft attention mass sampling to 64, and completed both same-seed and same-masklet selected5 no-action branches. Pose parity against the existing no-trace baseline passed for both modes: combined pose SHA equal={a5_q4096_pose_sha_equal_count}/10, numeric equal={a5_q4096_pose_numeric_equal_count}/10, parity_pass={a5_q4096_parity_pass}. The materialized q4096raw audit reports edge rows={a5_q4096_edge_summary.get("edge_row_count", 0)}, sampled exact nonproxy current-support edges={a5_q4096_edge_summary.get("sampled_exact_nonproxy_current_support_edge_count", 0)}, mode:case keys with sampled exact nonproxy support={a5_q4096_edge_summary.get("case_with_sampled_exact_nonproxy_current_support_edge_count", 0)}, safe-good mode:case keys with sampled exact nonproxy support={a5_q4096_edge_summary.get("safe_good_with_sampled_exact_nonproxy_current_support_edge_count", 0)}, formal fullquery strict current-support cases={a5_q4096_edge_summary.get("formal_fullquery_strict_current_support_pass_case_count", 0)}, action-ready query/head-local cases={a5_q4096_edge_summary.get("action_ready_query_head_local_case_count", 0)}, first_blocker={a5_q4096_edge_summary.get("first_blocker", "")}. This confirms that a denser raw edge dump can be made stable and output-neutral, but it is still a sampled no-action provider audit and does not satisfy Stage1 fullquery/action-ready requirements.

Measured v104 gate values:

```text
selected_action_case_coverage={selected_action_case_coverage:.6f}
fresh_direct_masklet_payload_frac={fresh_direct_masklet_frac:.6f}
fresh_direct_seed_payload_frac={fresh_direct_seed_frac:.6f}
strict_current_cache_component_candidate_frac={strict_current_cache_component_frac:.6f}
strict_current_cache_seed_candidate_frac={strict_current_cache_seed_frac:.6f}
fullquery_strict_identity_candidate_frac={fullquery_strict_identity_frac:.6f}
fullquery_nonproxy_current_support_candidate_frac={fullquery_nonproxy_frac:.6f}
fullquery_stable_lifecycle_materialized_case_count={fullquery_stable_lifecycle_count}
fullquery_action_ready_query_head_local_case_count={fullquery_query_head_local_count}
fullquery_action_ready_provider_pass_case_count={fullquery_action_ready_count}
observable_no_action_parity_pass={parity_summary.get("observable_no_action_parity_pass", False)}
pose_sha_equal_case_count={parity_summary.get("pose_sha_equal_case_count", 0)}
pose_numeric_equal_case_count={parity_summary.get("pose_numeric_equal_case_count", 0)}
plan_pose_sha_gate_pass={parity_summary.get("plan_pose_sha_gate_pass", False)}
deep_fullquery_direct_witness_case_count={deep_blocker_summary.get("fullquery_direct_witness_case_count", 0)}
deep_selected_stable_gated_tracked_cell_count={deep_blocker_summary.get("selected_stable_source_stable_gated_tracked_cell_count", 0)}
deep_selected_lifecycle_tracked_row_count={deep_blocker_summary.get("selected_stable_source_lifecycle_tracked_row_count", 0)}
deep_formal_provider_chain_first_blocker={deep_blocker_summary.get("formal_provider_chain_first_blocker", "")}
tracked_local_provider_candidate_case_count={local_provider_candidate_count}
tracked_local_provider_positive_hit_case_count={local_provider_positive_hit_count}
tracked_local_provider_safe_good_hit_case_count={local_provider_safe_good_hit_count}
demotion_rule_name={demotion_summary.get("demotion_rule_name", "")}
demotion_direct_seed_threshold={demotion_summary.get("direct_seed_demotion_threshold", "")}
demotion_rule_selected_case_count={demotion_summary.get("demotion_rule_selected_case_count", "")}
demotion_rule_safe_good_hit_case_count={demotion_summary.get("demotion_rule_safe_good_hit_case_count", "")}
demotion_rule_single_case_provider_pass_only={demotion_summary.get("demotion_rule_single_case_provider_pass_only", "")}
demotion_rule_provider_pass={demotion_summary.get("demotion_rule_provider_pass", "")}
querysoft_fullquery_conflict_case_count={demotion_summary.get("querysoft_fullquery_conflict_case_count", "")}
control_payload_per_head_tensor_available_case_count={control_payload_summary.get("per_head_tensor_available_case_count", "")}
control_payload_naive_query_shift_safe_good_hit_case_count={control_payload_summary.get("naive_query_shift_safe_good_hit_case_count", "")}
control_payload_naive_anchor_rotation_safe_good_hit_case_count={control_payload_summary.get("naive_anchor_rotation_safe_good_hit_case_count", "")}
control_payload_naive_raw_tensor_direct_count_matches_reported_case_count={control_payload_summary.get("naive_raw_tensor_direct_count_matches_reported_case_count", "")}
support_row_control_recompute_mismatch_case_count={support_control_summary.get("support_row_recompute_mismatch_case_count", "")}
support_row_control_query_shift_ge_threshold_case_ids={support_control_summary.get("query_shift_ge_threshold_case_ids", "")}
support_row_control_query_shift_safe_good_hit_case_count={support_control_summary.get("query_shift_safe_good_hit_case_count", "")}
support_row_control_anchor_rotation_ge_threshold_case_ids={support_control_summary.get("anchor_rotation_ge_threshold_case_ids", "")}
support_row_control_anchor_rotation_safe_good_hit_case_count={support_control_summary.get("anchor_rotation_safe_good_hit_case_count", "")}
query_head_witness_head_row_count={query_head_witness_summary.get("head_witness_row_count", "")}
query_head_witness_direct_case_ids={query_head_witness_summary.get("direct_witness_case_ids", "")}
query_head_witness_safe_good_direct_case_ids={query_head_witness_summary.get("safe_good_direct_witness_case_ids", "")}
query_head_witness_action_ready_query_head_local_edge={query_head_witness_summary.get("query_head_local_edges_available", "")}
exact_witness_cell_count={exact_witness_summary.get("exact_witness_cell_count", "")}
exact_witness_case_ids={exact_witness_summary.get("exact_witness_case_ids", "")}
exact_witness_safe_good_case_ids={exact_witness_summary.get("safe_good_exact_witness_case_ids", "")}
exact_anchor_rotation_ge_threshold_case_ids={exact_witness_summary.get("anchor_rotation_ge_threshold_case_ids", "")}
exact_control_observed_ge_threshold_case_ids={exact_control_summary.get("observed_ge_threshold_case_ids", "")}
exact_control_query_head_rotation_ge_threshold_case_ids={exact_control_summary.get("query_head_rotation_ge_threshold_case_ids", "")}
exact_control_query_head_rotation_margin={exact_control_summary.get("query_head_rotation_control_margin", "")}
exact_control_anchor_seed_rotation_ge_threshold_case_ids={exact_control_summary.get("anchor_seed_rotation_ge_threshold_case_ids", "")}
exact_control_anchor_seed_rotation_margin={exact_control_summary.get("anchor_seed_rotation_control_margin", "")}
exact_provider_control_diagnostic_pass={exact_control_summary.get("provider_control_diagnostic_pass", "")}
stable_lifecycle_policy_repair_allowed_rule_count={stable_policy_repair_summary.get("repair_allowed_rule_count", "")}
stable_lifecycle_policy_repair_first_blocker={stable_policy_repair_summary.get("first_blocker", "")}
v103_lifecycle_payload_closure_first_blocker={lifecycle_payload_closure_summary.get("first_blocker", "")}
v103_lifecycle_payload_closure_raw_absence_closed={lifecycle_payload_closure_summary.get("provider_policy_raw_tensor_absence_closed", "")}
v103_stuff_static_strict_current_support_pass_count={lifecycle_payload_closure_summary.get("stuff_static_support_strict_current_support_pass_count", "")}
v103_stuff_static_strict_provider_materializable_count={lifecycle_payload_closure_summary.get("stuff_static_strict_provider_materializable_count", "")}
v103_stuff_static_action_ready_carrier_available={lifecycle_payload_closure_summary.get("stuff_static_action_ready_carrier_available", "")}
row_level_current_support_carrier_repair_allowed_rule_count={row_level_repair_summary.get("repair_allowed_rule_count", "")}
row_level_current_support_carrier_repair_first_blocker={row_level_repair_summary.get("first_blocker", "")}
fresh_q128_tracked_lifecycle_candidate_row_count={fresh_q128_lifecycle_summary.get("candidate_row_count", "")}
fresh_q128_tracked_lifecycle_same_seed_candidate_case_count={fresh_q128_lifecycle_summary.get("same_seed_candidate_case_count", "")}
fresh_q128_tracked_lifecycle_same_masklet_candidate_case_count={fresh_q128_lifecycle_summary.get("same_masklet_candidate_case_count", "")}
fresh_q128_tracked_lifecycle_same_seed_and_masklet_candidate_case_count={fresh_q128_lifecycle_summary.get("same_seed_and_masklet_candidate_case_count", "")}
fresh_q128_tracked_lifecycle_formal_strict_current_support_pass_case_count={fresh_q128_lifecycle_summary.get("formal_strict_current_support_pass_case_count", "")}
fresh_q128_tracked_lifecycle_action_ready_carrier_case_count={fresh_q128_lifecycle_summary.get("action_ready_carrier_case_count", "")}
fresh_q128_tracked_lifecycle_repair_allowed_rule_count={fresh_q128_lifecycle_summary.get("repair_allowed_rule_count", "")}
fresh_q128_tracked_lifecycle_repair_first_blocker={fresh_q128_lifecycle_summary.get("first_blocker", "")}
a5_query_head_local_trace_case_with_swa_read_trace_count={a5_trace_summary.get("case_with_swa_read_trace_count", "")}
a5_query_head_local_trace_case_with_swa_read_carrier_rule_pass_count={a5_trace_summary.get("case_with_swa_read_carrier_rule_pass_count", "")}
a5_query_head_local_trace_safe_good_carrier_rule_pass_count={a5_trace_summary.get("safe_good_swa_read_carrier_rule_pass_count", "")}
a5_query_head_local_trace_attention_or_ttt_apply_carrier_case_count={a5_trace_summary.get("attention_or_ttt_apply_carrier_case_count", "")}
a5_query_head_local_trace_pose_sha_equal_case_count={a5_trace_summary.get("pose_sha_equal_case_count", "")}
a5_query_head_local_trace_pose_numeric_equal_case_count={a5_trace_summary.get("pose_numeric_equal_case_count", "")}
a5_query_head_local_trace_no_trace_pose_sha_parity_pass={a5_trace_summary.get("no_trace_pose_sha_parity_pass", "")}
a5_query_head_local_trace_repair_allowed_rule_count={a5_trace_summary.get("repair_allowed_rule_count", "")}
a5_query_head_local_trace_repair_first_blocker={a5_trace_summary.get("first_blocker", "")}
a5_same_masklet_query_head_local_trace_case_with_swa_read_trace_count={a5_same_masklet_summary.get("case_with_swa_read_trace_count", "")}
a5_same_masklet_query_head_local_trace_case_with_swa_read_carrier_rule_pass_count={a5_same_masklet_summary.get("case_with_swa_read_carrier_rule_pass_count", "")}
a5_same_masklet_query_head_local_trace_safe_good_carrier_rule_pass_count={a5_same_masklet_summary.get("safe_good_swa_read_carrier_rule_pass_count", "")}
a5_same_masklet_query_head_local_trace_attention_or_ttt_apply_carrier_case_count={a5_same_masklet_summary.get("attention_or_ttt_apply_carrier_case_count", "")}
a5_same_masklet_query_head_local_trace_pose_sha_equal_case_count={a5_same_masklet_summary.get("pose_sha_equal_case_count", "")}
a5_same_masklet_query_head_local_trace_pose_numeric_equal_case_count={a5_same_masklet_summary.get("pose_numeric_equal_case_count", "")}
a5_same_masklet_query_head_local_trace_no_trace_pose_sha_parity_pass={a5_same_masklet_summary.get("no_trace_pose_sha_parity_pass", "")}
a5_same_masklet_query_head_local_trace_repair_allowed_rule_count={a5_same_masklet_summary.get("repair_allowed_rule_count", "")}
a5_same_masklet_query_head_local_trace_repair_first_blocker={a5_same_masklet_summary.get("first_blocker", "")}
a5_direct_match_threshold_rule_count={a5_threshold_summary.get("threshold_rule_count", "")}
a5_direct_match_threshold_no_safe_good_rule_count={a5_threshold_summary.get("no_safe_good_threshold_rule_count", "")}
a5_direct_match_threshold_best_no_safe_good_rule={a5_threshold_summary.get("best_no_safe_good_rule", "")}
a5_direct_match_threshold_repair_allowed_rule_count={a5_threshold_summary.get("repair_allowed_rule_count", "")}
a5_direct_match_threshold_repair_first_blocker={a5_threshold_summary.get("first_blocker", "")}
a5_sampled_exact_edge_row_count={a5_sampled_edge_summary.get("edge_row_count", "")}
a5_sampled_exact_nonproxy_current_support_edge_count={a5_sampled_edge_summary.get("sampled_exact_nonproxy_current_support_edge_count", "")}
a5_sampled_exact_case_with_nonproxy_current_support_edge_count={a5_sampled_edge_summary.get("case_with_sampled_exact_nonproxy_current_support_edge_count", "")}
a5_sampled_exact_safe_good_with_nonproxy_current_support_edge_count={a5_sampled_edge_summary.get("safe_good_with_sampled_exact_nonproxy_current_support_edge_count", "")}
a5_sampled_exact_formal_fullquery_strict_current_support_pass_case_count={a5_sampled_edge_summary.get("formal_fullquery_strict_current_support_pass_case_count", "")}
a5_sampled_exact_action_ready_query_head_local_case_count={a5_sampled_edge_summary.get("action_ready_query_head_local_case_count", "")}
a5_sampled_exact_edge_first_blocker={a5_sampled_edge_summary.get("first_blocker", "")}
a5_q4096raw_exact_edge_row_count={a5_q4096_edge_summary.get("edge_row_count", "")}
a5_q4096raw_exact_nonproxy_current_support_edge_count={a5_q4096_edge_summary.get("sampled_exact_nonproxy_current_support_edge_count", "")}
a5_q4096raw_exact_case_with_nonproxy_current_support_edge_count={a5_q4096_edge_summary.get("case_with_sampled_exact_nonproxy_current_support_edge_count", "")}
a5_q4096raw_exact_safe_good_with_nonproxy_current_support_edge_count={a5_q4096_edge_summary.get("safe_good_with_sampled_exact_nonproxy_current_support_edge_count", "")}
a5_q4096raw_exact_formal_fullquery_strict_current_support_pass_case_count={a5_q4096_edge_summary.get("formal_fullquery_strict_current_support_pass_case_count", "")}
a5_q4096raw_exact_action_ready_query_head_local_case_count={a5_q4096_edge_summary.get("action_ready_query_head_local_case_count", "")}
a5_q4096raw_exact_edge_first_blocker={a5_q4096_edge_summary.get("first_blocker", "")}
a5_q4096raw_pose_sha_equal_case_count={a5_q4096_pose_sha_equal_count}
a5_q4096raw_pose_numeric_equal_case_count={a5_q4096_pose_numeric_equal_count}
a5_q4096raw_no_trace_pose_sha_parity_pass={a5_q4096_parity_pass}
strict_identity_candidate_frac={strict_identity_frac:.6f}
nonproxy_current_support_frac={nonproxy_frac:.6f}
semantic_class_fallback_frac={fallback_frac:.6f}
proxy_only_frac={proxy_only_frac:.6f}
missing_raw_sidecar_case_count={raw_missing_count}
provider_join_failure_rate={provider_failure_rate:.6f}
query_head_random_control_available={query_head_random_control_available}
anchor_id_rotation_control_available={anchor_id_rotation_control_available}
```
"""
    write_text(out / "provider_repair_attempts.md", repair_report)

    if not provider_pass:
        write_text(
            out / "provider_external_projection_only_no_action.md",
            "# Provider Blocker\n\n"
            "Stage 1 did not pass. The selected11 fresh traces repair direct masklet/seed payload materialization, "
            "but action purposes still require the strict provider join, nonproxy current support, provider controls, "
            "query/head-local carrier, and full write/cache/current/read/L3 chain. Stage 4 runtime action is blocked "
            "by v104 gate policy.\n",
        )

    stage1_blocker = ""
    if not provider_pass:
        stage1_blocker = (
            "direct_payload_and_no_action_controls_repaired_but_strict_provider_join_lifecycle_query_head_local_blocked_no_action"
            if provider_controls_available
            else "direct_payload_repaired_but_strict_provider_join_controls_blocked_no_action"
        )

    summary = {
        "schema": "acl2_v104_stage1_provider_summary_v1",
        "stage": 1,
        "selected_action_case_count": selected_count,
        "selected_action_case_coverage": selected_action_case_coverage,
        "fresh_selected11_trace_root": rel(paths["selected11_fresh_trace_root"]),
        "selected11_no_trace_pose_sha_parity_root": rel(paths["selected11_no_trace_pose_sha_parity_root"]),
        "fresh_trace_loaded_case_count": fresh_trace_summary.get("selected_trace_loaded_count", 0),
        "observable_no_action_parity_pass": parity_summary.get("observable_no_action_parity_pass", False),
        "plan_pose_sha_gate_pass": parity_summary.get("plan_pose_sha_gate_pass", False),
        "pose_sha_equal_case_count": parity_summary.get("pose_sha_equal_case_count", 0),
        "pose_numeric_equal_case_count": parity_summary.get("pose_numeric_equal_case_count", 0),
        "no_trace_pose_sha_parity_pass": parity_summary.get("no_trace_pose_sha_parity_pass", False),
        "no_trace_pose_sha_parity_failed_job_count": parity_summary.get(
            "no_trace_pose_sha_parity_failed_job_count", ""
        ),
        "deep_provider_fullquery_direct_witness_case_count": deep_blocker_summary.get(
            "fullquery_direct_witness_case_count", 0
        ),
        "deep_provider_fullquery_direct_witness_seed_count": deep_blocker_summary.get(
            "fullquery_direct_witness_seed_count", 0
        ),
        "deep_provider_selected_fullquery_same_masklet_instance_topk_cell_count": deep_blocker_summary.get(
            "selected_fullquery_same_masklet_instance_topk_cell_count", 0
        ),
        "deep_provider_selected_stable_source_raw_tracked_cell_count": deep_blocker_summary.get(
            "selected_stable_source_raw_tracked_cell_count", 0
        ),
        "deep_provider_selected_stable_gated_tracked_cell_count": deep_blocker_summary.get(
            "selected_stable_source_stable_gated_tracked_cell_count", 0
        ),
        "deep_provider_selected_lifecycle_tracked_row_count": deep_blocker_summary.get(
            "selected_stable_source_lifecycle_tracked_row_count", 0
        ),
        "deep_provider_selected_raw_tracked_candidate_topk_cell_count": deep_blocker_summary.get(
            "selected_raw_tracked_candidate_topk_cell_count", 0
        ),
        "deep_provider_selected_raw_tracked_candidate_stable_gated_cell_count": deep_blocker_summary.get(
            "selected_raw_tracked_candidate_stable_gated_cell_count", 0
        ),
        "deep_provider_selected_raw_tracked_candidate_lifecycle_row_count": deep_blocker_summary.get(
            "selected_raw_tracked_candidate_lifecycle_row_count", 0
        ),
        "deep_provider_formal_chain_provider_pass": deep_blocker_summary.get(
            "formal_provider_chain_provider_pass", False
        ),
        "deep_provider_formal_chain_first_blocker": deep_blocker_summary.get(
            "formal_provider_chain_first_blocker", ""
        ),
        "tracked_local_provider_candidate_case_count": local_provider_candidate_count,
        "tracked_local_provider_positive_hit_case_count": local_provider_positive_hit_count,
        "tracked_local_provider_safe_good_hit_case_count": local_provider_safe_good_hit_count,
        "tracked_local_provider_rule_safe_for_action": local_provider_summary.get(
            "tracked_local_provider_rule_safe_for_action", False
        ),
        "demotion_rule_name": demotion_summary.get("demotion_rule_name", ""),
        "demotion_direct_seed_threshold": demotion_summary.get("direct_seed_demotion_threshold", ""),
        "demotion_rule_selected_case_ids": demotion_summary.get("demotion_rule_selected_case_ids", []),
        "demotion_rule_selected_case_count": demotion_summary.get("demotion_rule_selected_case_count", 0),
        "demotion_rule_safe_good_hit_case_count": demotion_summary.get(
            "demotion_rule_safe_good_hit_case_count", 0
        ),
        "demotion_rule_strict_swa_bad_hit_case_count": demotion_summary.get(
            "demotion_rule_strict_swa_bad_hit_case_count", 0
        ),
        "demotion_rule_single_case_provider_pass_only": demotion_summary.get(
            "demotion_rule_single_case_provider_pass_only", False
        ),
        "demotion_rule_provider_pass": demotion_summary.get("demotion_rule_provider_pass", False),
        "querysoft_fullquery_conflict_case_count": demotion_summary.get("querysoft_fullquery_conflict_case_count", 0),
        "control_payload_per_head_tensor_available_case_count": control_payload_summary.get(
            "per_head_tensor_available_case_count", 0
        ),
        "control_payload_naive_query_shift_safe_good_hit_case_count": control_payload_summary.get(
            "naive_query_shift_safe_good_hit_case_count", 0
        ),
        "control_payload_naive_anchor_rotation_safe_good_hit_case_count": control_payload_summary.get(
            "naive_anchor_rotation_safe_good_hit_case_count", 0
        ),
        "control_payload_naive_raw_tensor_direct_count_matches_reported_case_count": control_payload_summary.get(
            "naive_raw_tensor_direct_count_matches_reported_case_count", 0
        ),
        "support_row_control_recompute_mismatch_case_count": support_control_summary.get(
            "support_row_recompute_mismatch_case_count", 0
        ),
        "support_row_control_query_shift_ge_threshold_case_ids": support_control_summary.get(
            "query_shift_ge_threshold_case_ids", []
        ),
        "support_row_control_query_shift_safe_good_hit_case_count": support_control_summary.get(
            "query_shift_safe_good_hit_case_count", 0
        ),
        "support_row_control_anchor_rotation_ge_threshold_case_ids": support_control_summary.get(
            "anchor_rotation_ge_threshold_case_ids", []
        ),
        "support_row_control_anchor_rotation_safe_good_hit_case_count": support_control_summary.get(
            "anchor_rotation_safe_good_hit_case_count", 0
        ),
        "support_row_anchor_rotation_surrogate_negative_pass": support_control_summary.get(
            "support_row_anchor_rotation_surrogate_negative_pass", False
        ),
        "query_head_witness_head_row_count": query_head_witness_summary.get("head_witness_row_count", 0),
        "query_head_witness_direct_case_ids": query_head_witness_summary.get("direct_witness_case_ids", []),
        "query_head_witness_safe_good_direct_case_ids": query_head_witness_summary.get(
            "safe_good_direct_witness_case_ids", []
        ),
        "query_head_witness_strict_positive_direct_case_ids": query_head_witness_summary.get(
            "strict_positive_direct_witness_case_ids", []
        ),
        "query_head_witness_action_ready_query_head_local_edge": query_head_witness_summary.get(
            "query_head_local_edges_available", False
        ),
        "exact_witness_cell_count": exact_witness_summary.get("exact_witness_cell_count", 0),
        "exact_witness_case_ids": exact_witness_summary.get("exact_witness_case_ids", []),
        "exact_witness_safe_good_case_ids": exact_witness_summary.get("safe_good_exact_witness_case_ids", []),
        "exact_witness_strict_positive_case_ids": exact_witness_summary.get(
            "strict_positive_exact_witness_case_ids", []
        ),
        "exact_anchor_rotation_ge_threshold_case_ids": exact_witness_summary.get(
            "anchor_rotation_ge_threshold_case_ids", []
        ),
        "exact_anchor_rotation_surrogate_negative_pass": exact_witness_summary.get(
            "exact_anchor_rotation_surrogate_negative_pass", False
        ),
        "exact_control_observed_ge_threshold_case_ids": exact_control_summary.get(
            "observed_ge_threshold_case_ids", []
        ),
        "exact_control_query_head_rotation_ge_threshold_case_ids": exact_control_summary.get(
            "query_head_rotation_ge_threshold_case_ids", []
        ),
        "exact_control_query_head_rotation_margin": exact_control_summary.get(
            "query_head_rotation_control_margin", 0.0
        ),
        "exact_control_anchor_seed_rotation_ge_threshold_case_ids": exact_control_summary.get(
            "anchor_seed_rotation_ge_threshold_case_ids", []
        ),
        "exact_control_anchor_seed_rotation_margin": exact_control_summary.get(
            "anchor_seed_rotation_control_margin", 0.0
        ),
        "exact_provider_control_diagnostic_pass": exact_control_summary.get(
            "provider_control_diagnostic_pass", False
        ),
        "stable_lifecycle_policy_repair_attempt_rows": rel(out / "stable_lifecycle_policy_repair_attempt_rows.csv"),
        "stable_lifecycle_policy_repair_attempt_rule_rows": rel(
            out / "stable_lifecycle_policy_repair_attempt_rule_rows.csv"
        ),
        "stable_lifecycle_policy_repair_attempt_summary": rel(
            out / "stable_lifecycle_policy_repair_attempt_summary.json"
        ),
        "stable_lifecycle_policy_repair_allowed_rule_count": stable_policy_repair_summary.get(
            "repair_allowed_rule_count", 0
        ),
        "stable_lifecycle_policy_repair_first_blocker": stable_policy_repair_summary.get("first_blocker", ""),
        "stable_lifecycle_policy_stage6_lifecycle_row_count": stable_policy_repair_summary.get(
            "stage6_lifecycle_row_count", 0
        ),
        "stable_lifecycle_policy_stage6_lifecycle_tracked_row_count": stable_policy_repair_summary.get(
            "stage6_lifecycle_tracked_row_count", 0
        ),
        "v103_lifecycle_payload_closure_import_summary": rel(
            out / "v103_lifecycle_payload_closure_import_summary.json"
        ),
        "v103_lifecycle_payload_closure_first_blocker": lifecycle_payload_closure_summary.get(
            "first_blocker", ""
        ),
        "v103_lifecycle_payload_closure_raw_absence_closed": lifecycle_payload_closure_summary.get(
            "provider_policy_raw_tensor_absence_closed", False
        ),
        "v103_lifecycle_payload_closure_source_type_boundary_closed": lifecycle_payload_closure_summary.get(
            "provider_policy_source_type_boundary_closed", False
        ),
        "v103_stuff_static_current_cache_same_seed_supported_count": lifecycle_payload_closure_summary.get(
            "stuff_static_current_cache_same_seed_supported_count", 0
        ),
        "v103_stuff_static_strict_current_support_pass_count": lifecycle_payload_closure_summary.get(
            "stuff_static_support_strict_current_support_pass_count", 0
        ),
        "v103_stuff_static_strict_provider_materializable_count": lifecycle_payload_closure_summary.get(
            "stuff_static_strict_provider_materializable_count", 0
        ),
        "v103_stuff_static_action_ready_carrier_available": lifecycle_payload_closure_summary.get(
            "stuff_static_action_ready_carrier_available", False
        ),
        "row_level_current_support_carrier_repair_attempt_rows": rel(
            out / "row_level_current_support_carrier_repair_attempt_rows.csv"
        ),
        "row_level_current_support_carrier_repair_attempt_rule_rows": rel(
            out / "row_level_current_support_carrier_repair_attempt_rule_rows.csv"
        ),
        "row_level_current_support_carrier_repair_attempt_summary": rel(
            out / "row_level_current_support_carrier_repair_attempt_summary.json"
        ),
        "row_level_current_support_carrier_repair_allowed_rule_count": row_level_repair_summary.get(
            "repair_allowed_rule_count", 0
        ),
        "row_level_current_support_carrier_repair_first_blocker": row_level_repair_summary.get(
            "first_blocker", ""
        ),
        "row_level_current_support_carrier_candidate_source_counts": row_level_repair_summary.get(
            "candidate_source_counts", {}
        ),
        "fresh_q128_tracked_lifecycle_repair_attempt_rows": rel(
            out / "fresh_q128_tracked_lifecycle_repair_attempt_rows.csv"
        ),
        "fresh_q128_tracked_lifecycle_repair_attempt_rule_rows": rel(
            out / "fresh_q128_tracked_lifecycle_repair_attempt_rule_rows.csv"
        ),
        "fresh_q128_tracked_lifecycle_repair_attempt_summary": rel(
            out / "fresh_q128_tracked_lifecycle_repair_attempt_summary.json"
        ),
        "fresh_q128_tracked_lifecycle_candidate_row_count": fresh_q128_lifecycle_summary.get(
            "candidate_row_count", 0
        ),
        "fresh_q128_tracked_lifecycle_same_seed_candidate_case_count": fresh_q128_lifecycle_summary.get(
            "same_seed_candidate_case_count", 0
        ),
        "fresh_q128_tracked_lifecycle_same_masklet_candidate_case_count": fresh_q128_lifecycle_summary.get(
            "same_masklet_candidate_case_count", 0
        ),
        "fresh_q128_tracked_lifecycle_same_seed_and_masklet_candidate_case_count": (
            fresh_q128_lifecycle_summary.get("same_seed_and_masklet_candidate_case_count", 0)
        ),
        "fresh_q128_tracked_lifecycle_formal_strict_current_support_pass_case_count": (
            fresh_q128_lifecycle_summary.get("formal_strict_current_support_pass_case_count", 0)
        ),
        "fresh_q128_tracked_lifecycle_action_ready_carrier_case_count": fresh_q128_lifecycle_summary.get(
            "action_ready_carrier_case_count", 0
        ),
        "fresh_q128_tracked_lifecycle_repair_allowed_rule_count": fresh_q128_lifecycle_summary.get(
            "repair_allowed_rule_count", 0
        ),
        "fresh_q128_tracked_lifecycle_repair_first_blocker": fresh_q128_lifecycle_summary.get(
            "first_blocker", ""
        ),
        "a5_query_head_local_trace_repair_attempt_rows": rel(
            out / "a5_query_head_local_trace_repair_attempt_rows.csv"
        ),
        "a5_query_head_local_trace_repair_attempt_rule_rows": rel(
            out / "a5_query_head_local_trace_repair_attempt_rule_rows.csv"
        ),
        "a5_query_head_local_trace_repair_attempt_summary": rel(
            out / "a5_query_head_local_trace_repair_attempt_summary.json"
        ),
        "a5_query_head_local_trace_case_with_swa_read_trace_count": a5_trace_summary.get(
            "case_with_swa_read_trace_count", 0
        ),
        "a5_query_head_local_trace_case_with_swa_read_carrier_rule_pass_count": a5_trace_summary.get(
            "case_with_swa_read_carrier_rule_pass_count", 0
        ),
        "a5_query_head_local_trace_safe_good_carrier_rule_pass_count": a5_trace_summary.get(
            "safe_good_swa_read_carrier_rule_pass_count", 0
        ),
        "a5_query_head_local_trace_attention_or_ttt_apply_carrier_case_count": a5_trace_summary.get(
            "attention_or_ttt_apply_carrier_case_count", 0
        ),
        "a5_query_head_local_trace_pose_sha_equal_case_count": a5_trace_summary.get(
            "pose_sha_equal_case_count", ""
        ),
        "a5_query_head_local_trace_pose_numeric_equal_case_count": a5_trace_summary.get(
            "pose_numeric_equal_case_count", ""
        ),
        "a5_query_head_local_trace_no_trace_pose_sha_parity_pass": a5_trace_summary.get(
            "no_trace_pose_sha_parity_pass", False
        ),
        "a5_query_head_local_trace_repair_allowed_rule_count": a5_trace_summary.get(
            "repair_allowed_rule_count", 0
        ),
        "a5_query_head_local_trace_repair_first_blocker": a5_trace_summary.get("first_blocker", ""),
        "a5_same_masklet_query_head_local_trace_repair_attempt_rows": rel(
            out / "a5_same_masklet_query_head_local_trace_repair_attempt_rows.csv"
        ),
        "a5_same_masklet_query_head_local_trace_repair_attempt_rule_rows": rel(
            out / "a5_same_masklet_query_head_local_trace_repair_attempt_rule_rows.csv"
        ),
        "a5_same_masklet_query_head_local_trace_repair_attempt_summary": rel(
            out / "a5_same_masklet_query_head_local_trace_repair_attempt_summary.json"
        ),
        "a5_same_masklet_query_head_local_trace_case_with_swa_read_trace_count": (
            a5_same_masklet_summary.get("case_with_swa_read_trace_count", 0)
        ),
        "a5_same_masklet_query_head_local_trace_case_with_swa_read_carrier_rule_pass_count": (
            a5_same_masklet_summary.get("case_with_swa_read_carrier_rule_pass_count", 0)
        ),
        "a5_same_masklet_query_head_local_trace_safe_good_carrier_rule_pass_count": (
            a5_same_masklet_summary.get("safe_good_swa_read_carrier_rule_pass_count", 0)
        ),
        "a5_same_masklet_query_head_local_trace_attention_or_ttt_apply_carrier_case_count": (
            a5_same_masklet_summary.get("attention_or_ttt_apply_carrier_case_count", 0)
        ),
        "a5_same_masklet_query_head_local_trace_pose_sha_equal_case_count": a5_same_masklet_summary.get(
            "pose_sha_equal_case_count", ""
        ),
        "a5_same_masklet_query_head_local_trace_pose_numeric_equal_case_count": a5_same_masklet_summary.get(
            "pose_numeric_equal_case_count", ""
        ),
        "a5_same_masklet_query_head_local_trace_no_trace_pose_sha_parity_pass": a5_same_masklet_summary.get(
            "no_trace_pose_sha_parity_pass", False
        ),
        "a5_same_masklet_query_head_local_trace_repair_allowed_rule_count": a5_same_masklet_summary.get(
            "repair_allowed_rule_count", 0
        ),
        "a5_same_masklet_query_head_local_trace_repair_first_blocker": a5_same_masklet_summary.get(
            "first_blocker", ""
        ),
        "a5_direct_match_threshold_repair_attempt_rows": rel(
            out / "a5_direct_match_threshold_repair_attempt_rows.csv"
        ),
        "a5_direct_match_threshold_repair_attempt_summary": rel(
            out / "a5_direct_match_threshold_repair_attempt_summary.json"
        ),
        "a5_direct_match_threshold_rule_count": a5_threshold_summary.get("threshold_rule_count", 0),
        "a5_direct_match_threshold_no_safe_good_rule_count": a5_threshold_summary.get(
            "no_safe_good_threshold_rule_count", 0
        ),
        "a5_direct_match_threshold_best_no_safe_good_rule": a5_threshold_summary.get(
            "best_no_safe_good_rule", {}
        ),
        "a5_direct_match_threshold_repair_allowed_rule_count": a5_threshold_summary.get(
            "repair_allowed_rule_count", 0
        ),
        "a5_direct_match_threshold_repair_first_blocker": a5_threshold_summary.get("first_blocker", ""),
        "a5_sampled_exact_edge_materialization_rows": rel(out / "a5_sampled_exact_edge_materialization_rows.csv"),
        "a5_sampled_exact_edge_materialization_case_rows": rel(
            out / "a5_sampled_exact_edge_materialization_case_rows.csv"
        ),
        "a5_sampled_exact_edge_materialization_summary": rel(
            out / "a5_sampled_exact_edge_materialization_summary.json"
        ),
        "a5_sampled_exact_edge_row_count": a5_sampled_edge_summary.get("edge_row_count", 0),
        "a5_sampled_exact_nonproxy_current_support_edge_count": a5_sampled_edge_summary.get(
            "sampled_exact_nonproxy_current_support_edge_count", 0
        ),
        "a5_sampled_exact_case_with_nonproxy_current_support_edge_count": a5_sampled_edge_summary.get(
            "case_with_sampled_exact_nonproxy_current_support_edge_count", 0
        ),
        "a5_sampled_exact_safe_good_with_nonproxy_current_support_edge_count": a5_sampled_edge_summary.get(
            "safe_good_with_sampled_exact_nonproxy_current_support_edge_count", 0
        ),
        "a5_sampled_exact_formal_fullquery_strict_current_support_pass_case_count": (
            a5_sampled_edge_summary.get("formal_fullquery_strict_current_support_pass_case_count", 0)
        ),
        "a5_sampled_exact_action_ready_query_head_local_case_count": a5_sampled_edge_summary.get(
            "action_ready_query_head_local_case_count", 0
        ),
        "a5_sampled_exact_edge_first_blocker": a5_sampled_edge_summary.get("first_blocker", ""),
        "a5_q4096raw_exact_edge_materialization_rows": rel(out / "a5_q4096raw_exact_edge_materialization_rows.csv"),
        "a5_q4096raw_exact_edge_materialization_case_rows": rel(
            out / "a5_q4096raw_exact_edge_materialization_case_rows.csv"
        ),
        "a5_q4096raw_exact_edge_materialization_summary": rel(
            out / "a5_q4096raw_exact_edge_materialization_summary.json"
        ),
        "a5_q4096raw_exact_edge_row_count": a5_q4096_edge_summary.get("edge_row_count", 0),
        "a5_q4096raw_exact_nonproxy_current_support_edge_count": a5_q4096_edge_summary.get(
            "sampled_exact_nonproxy_current_support_edge_count", 0
        ),
        "a5_q4096raw_exact_case_with_nonproxy_current_support_edge_count": a5_q4096_edge_summary.get(
            "case_with_sampled_exact_nonproxy_current_support_edge_count", 0
        ),
        "a5_q4096raw_exact_safe_good_with_nonproxy_current_support_edge_count": a5_q4096_edge_summary.get(
            "safe_good_with_sampled_exact_nonproxy_current_support_edge_count", 0
        ),
        "a5_q4096raw_exact_formal_fullquery_strict_current_support_pass_case_count": (
            a5_q4096_edge_summary.get("formal_fullquery_strict_current_support_pass_case_count", 0)
        ),
        "a5_q4096raw_exact_action_ready_query_head_local_case_count": a5_q4096_edge_summary.get(
            "action_ready_query_head_local_case_count", 0
        ),
        "a5_q4096raw_exact_edge_first_blocker": a5_q4096_edge_summary.get("first_blocker", ""),
        "a5_q4096raw_same_seed_no_trace_pose_sha_parity_summary": rel(
            paths["selected5_a5_q4096raw_same_seed_no_trace_pose_sha_parity_root"] / "summary.json"
        ),
        "a5_q4096raw_same_masklet_no_trace_pose_sha_parity_summary": rel(
            paths["selected5_a5_q4096raw_same_masklet_no_trace_pose_sha_parity_root"] / "summary.json"
        ),
        "a5_q4096raw_pose_sha_equal_case_count": a5_q4096_pose_sha_equal_count,
        "a5_q4096raw_pose_numeric_equal_case_count": a5_q4096_pose_numeric_equal_count,
        "a5_q4096raw_no_trace_pose_sha_parity_pass": a5_q4096_parity_pass,
        "v103_querysoft_best_selected_cases_missing_v104_fullquery_direct_witness": demotion_summary.get(
            "v103_querysoft_best_selected_cases_missing_v104_fullquery_direct_witness", []
        ),
        "fresh_direct_masklet_payload_case_count": fresh_direct_masklet_count,
        "fresh_direct_masklet_payload_frac": fresh_direct_masklet_frac,
        "fresh_direct_seed_payload_case_count": fresh_direct_seed_count,
        "fresh_direct_seed_payload_frac": fresh_direct_seed_frac,
        "strict_current_cache_component_candidate_case_count": strict_current_cache_component_count,
        "strict_current_cache_component_candidate_frac": strict_current_cache_component_frac,
        "strict_current_cache_seed_candidate_case_count": strict_current_cache_seed_count,
        "strict_current_cache_seed_candidate_frac": strict_current_cache_seed_frac,
        "ttt_anchor_identity_available_case_count": ttt_anchor_identity_count,
        "fullquery_strict_identity_candidate_case_count": fullquery_strict_identity_count,
        "fullquery_strict_identity_candidate_frac": fullquery_strict_identity_frac,
        "fullquery_nonproxy_current_support_candidate_case_count": fullquery_nonproxy_count,
        "fullquery_nonproxy_current_support_candidate_frac": fullquery_nonproxy_frac,
        "fullquery_stable_lifecycle_materialized_case_count": fullquery_stable_lifecycle_count,
        "fullquery_action_ready_query_head_local_case_count": fullquery_query_head_local_count,
        "fullquery_action_ready_provider_pass_case_count": fullquery_action_ready_count,
        "strict_identity_candidate_frac": strict_identity_frac,
        "nonproxy_current_support_frac": nonproxy_frac,
        "semantic_class_fallback_frac": fallback_frac,
        "proxy_only_frac": proxy_only_frac,
        "missing_raw_sidecar_case_count": raw_missing_count,
        "query_head_random_control_available": query_head_random_control_available,
        "anchor_id_rotation_control_available": anchor_id_rotation_control_available,
        "provider_controls_available": provider_controls_available,
        "provider_control_claim_level": control_claim_level,
        "stage4_runtime_action_controls_available": stage4_runtime_controls_available,
        "provider_control_gap_summary": rel(out / "provider_control_gap_summary.json"),
        "provider_control_gap_rows": rel(out / "provider_control_gap_rows.csv"),
        "provider_control_gap_audit": rel(out / "provider_control_gap_audit.md"),
        "provider_join_failure_rate": provider_failure_rate,
        "provider_pass_case_count": provider_pass_count,
        "stage1_provider_pass": provider_pass,
        "runtime_action_allowed": False,
        "blocker": stage1_blocker,
        "selected11_fresh_trace_payload_rows": rel(out / "selected11_fresh_trace_payload_rows.csv"),
        "selected11_fresh_trace_payload_summary": rel(out / "selected11_fresh_trace_payload_summary.json"),
        "selected11_no_action_parity_rows": rel(out / "selected11_no_action_parity_rows.csv"),
        "selected11_no_action_parity_summary": rel(out / "selected11_no_action_parity_summary.json"),
        "selected11_deep_provider_blocker_rows": rel(out / "selected11_deep_provider_blocker_rows.csv"),
        "selected11_deep_provider_blocker_summary": rel(out / "selected11_deep_provider_blocker_summary.json"),
        "selected11_tracked_local_provider_feasibility_rows": rel(
            out / "selected11_tracked_local_provider_feasibility_rows.csv"
        ),
        "selected11_tracked_local_provider_feasibility_summary": rel(
            out / "selected11_tracked_local_provider_feasibility_summary.json"
        ),
        "tracked_local_provider_demotion_case_rows": rel(out / "tracked_local_provider_demotion_case_rows.csv"),
        "tracked_local_provider_demotion_rule_rows": rel(out / "tracked_local_provider_demotion_rule_rows.csv"),
        "tracked_local_provider_demotion_summary": rel(out / "tracked_local_provider_demotion_summary.json"),
        "provider_control_payload_feasibility_rows": rel(out / "provider_control_payload_feasibility_rows.csv"),
        "provider_control_payload_feasibility_summary": rel(
            out / "provider_control_payload_feasibility_summary.json"
        ),
        "provider_support_row_control_feasibility_rows": rel(
            out / "provider_support_row_control_feasibility_rows.csv"
        ),
        "provider_support_row_control_feasibility_summary": rel(
            out / "provider_support_row_control_feasibility_summary.json"
        ),
        "support_row_query_head_witness_rows": rel(out / "support_row_query_head_witness_rows.csv"),
        "support_row_query_head_witness_summary": rel(out / "support_row_query_head_witness_summary.json"),
        "exact_support_row_witness_set_rows": rel(out / "exact_support_row_witness_set_rows.csv"),
        "exact_support_row_witness_set_summary": rel(out / "exact_support_row_witness_set_summary.json"),
        "exact_provider_control_diagnostic_rows": rel(out / "exact_provider_control_diagnostic_rows.csv"),
        "exact_provider_control_diagnostic_summary": rel(out / "exact_provider_control_diagnostic_summary.json"),
        "selected11_fullquery_action_ready_ladder_rows": rel(out / "selected11_fullquery_action_ready_ladder_rows.csv"),
        "selected11_fullquery_action_ready_ladder_summary": rel(out / "selected11_fullquery_action_ready_ladder_summary.json"),
        "raw_sidecar_trace_manifest": rel(out / "raw_sidecar_trace_manifest.csv"),
        "provider_token_rows": rel(out / "provider_token_rows.csv"),
        "provider_case_summary": rel(out / "provider_case_summary.csv"),
        "provider_repair_attempts": rel(out / "provider_repair_attempts.md"),
    }
    write_json(out / "stage1_provider_summary.json", summary)
    return summary


def build_stage2(stage1: dict[str, Any]) -> dict[str, Any]:
    out = ROOT / "stage2_oracle"
    paths = v103_paths()
    oracle_rows = read_csv(paths["oracle_case_rows"])
    metric_rows = read_csv(paths["oracle_metric_rows"])
    case_by_id = index_by_case(oracle_rows)

    metric_by_name = {row.get("method_name", row.get("method", "")): row for row in metric_rows}
    geo = metric_by_name.get("geometry_only", {})
    sem = metric_by_name.get("semantic_plus_geometry", {})
    fpr_reduction = to_float(geo.get("good_FPR")) - to_float(sem.get("good_FPR"))
    ba_gain = to_float(sem.get("balanced_accuracy")) - to_float(geo.get("balanced_accuracy"))
    semantic_diagnostic_pass = fpr_reduction >= 0.10 or ba_gain >= 0.05
    semantic_action_entry_pass = (
        to_float(sem.get("bad_recall")) >= 0.60
        and to_float(sem.get("good_FPR")) <= 0.25
        and to_int(sem.get("selected_positive_sequence_coverage")) >= 2
        and to_float(sem.get("same_count_random_margin")) >= 0.05
        and bool(stage1.get("query_head_random_control_available"))
        and bool(stage1.get("anchor_id_rotation_control_available"))
        and bool(stage1.get("stage1_provider_pass"))
    )

    geo_selected = set(filter(None, geo.get("selected_case_ids", "").split(";")))
    sem_selected = set(filter(None, sem.get("selected_case_ids", "").split(";")))
    rescued_good = sorted(
        cid for cid in geo_selected - sem_selected if case_by_id.get(cid, {}).get("binary_label") == "good"
    )
    missed_bad = sorted(
        cid for cid in geo_selected - sem_selected if case_by_id.get(cid, {}).get("binary_label") == "bad"
    )

    evidence_rows: list[dict[str, Any]] = []
    for row in oracle_rows:
        case_id = row.get("case_id", "")
        geom_risk = to_float(row.get("B1_geometry_only_risk_score"))
        semgeom_risk = to_float(row.get("B3_semantic_plus_geometry_risk_score"))
        internal_risk = to_float(row.get("B4_internal_same_space_risk_score"))
        scale_obs = to_float(row.get("O_scale_repaired_mean"))
        current_support = bool(stage1.get("stage1_provider_pass"))
        if not current_support:
            state = "DELAY_UPDATE" if geom_risk >= to_float(geo.get("threshold")) else "CONTEXT_ONLY"
        elif geom_risk < 0.45 and scale_obs > 0.5:
            state = "SCALE_ELIGIBLE"
        elif scale_obs <= 0.3:
            state = "LOCAL_REGISTRATION_ONLY"
        elif semgeom_risk >= 0.7:
            state = "REJECT_SCALE_EVIDENCE"
        elif internal_risk >= 0.6:
            state = "HOLD_PREVIOUS"
        else:
            state = "CONTEXT_ONLY"
        evidence_rows.append(
            {
                "case_id": case_id,
                "seq": row.get("seq", ""),
                "binary_label": row.get("binary_label", ""),
                "case_role": classify_case(case_id),
                "geometry_risk_score": row.get("B1_geometry_only_risk_score", ""),
                "semantic_safety_score": row.get("B2_semantic_only_risk_score", ""),
                "semantic_plus_geometry_risk_score": row.get("B3_semantic_plus_geometry_risk_score", ""),
                "current_support_score": 1.0 if current_support else 0.0,
                "scale_observability_score": scale_obs,
                "internal_memory_risk_score": internal_risk,
                "evidence_state": state,
                "provider_pass": stage1.get("stage1_provider_pass", False),
                "claim_level": "diagnostic_oracle_no_runtime",
            }
        )

    case_summary = [
        {
            "case_id": row["case_id"],
            "seq": row["seq"],
            "binary_label": row["binary_label"],
            "case_role": row["case_role"],
            "evidence_state": row["evidence_state"],
            "selected_action_case": row["case_id"] in SELECTED_ACTION_CASES,
            "runtime_action_allowed": False,
        }
        for row in evidence_rows
    ]

    write_csv(out / "evidence_state_rows.csv", evidence_rows)
    write_csv(out / "evidence_state_case_summary.csv", case_summary)
    write_json(
        out / "oracle_comparison_summary.json",
        {
            "schema": "acl2_v104_stage2_oracle_comparison_summary_v1",
            "geometry_only_bad_recall": to_float(geo.get("bad_recall")),
            "geometry_only_good_FPR": to_float(geo.get("good_FPR")),
            "geometry_only_balanced_accuracy": to_float(geo.get("balanced_accuracy")),
            "semantic_plus_geometry_bad_recall": to_float(sem.get("bad_recall")),
            "semantic_plus_geometry_good_FPR": to_float(sem.get("good_FPR")),
            "semantic_plus_geometry_balanced_accuracy": to_float(sem.get("balanced_accuracy")),
            "semantic_good_FPR_reduction": fpr_reduction,
            "semantic_balanced_accuracy_gain": ba_gain,
            "semantic_diagnostic_pass": semantic_diagnostic_pass,
            "semantic_action_entry_pass": semantic_action_entry_pass,
            "provider_controls_available": bool(stage1.get("query_head_random_control_available"))
            and bool(stage1.get("anchor_id_rotation_control_available")),
            "runtime_action_allowed": False,
        },
    )

    write_text(
        out / "semantic_increment_report.md",
        f"""# Stage 2 Semantic Increment Report

```text
geometry_only_bad_recall={geo.get("bad_recall")}
geometry_only_good_FPR={geo.get("good_FPR")}
geometry_only_balanced_accuracy={geo.get("balanced_accuracy")}
semantic_plus_geometry_bad_recall={sem.get("bad_recall")}
semantic_plus_geometry_good_FPR={sem.get("good_FPR")}
semantic_plus_geometry_balanced_accuracy={sem.get("balanced_accuracy")}
semantic_good_FPR_reduction={fpr_reduction}
semantic_balanced_accuracy_gain={ba_gain}
semantic_diagnostic_pass={semantic_diagnostic_pass}
semantic_action_entry_pass={semantic_action_entry_pass}
```

Interpretation: semantic evidence shows safety-filter value by reducing good FPR, but it does not become action-entry evidence because bad recall, provider controls, and strict provider requirements are not satisfied together.
""",
    )
    write_text(
        out / "semantic_increment_failure.md",
        "# Stage 2 Action-Entry Failure\n\n"
        "Semantic+geometry did not exceed geometry-only as a high-recall/action-entry oracle. "
        "Its v104 role is safety filter / context demotion diagnostic, not runtime scale/gauge evidence.\n",
    )
    write_text(
        out / "false_positive_rescue_panels.md",
        "# False Positive Rescue Panels\n\n"
        "Artifact-backed case ids rescued by semantic+geometry relative to geometry-only:\n\n"
        f"```text\nrescued_good_cases={';'.join(rescued_good)}\n```\n\n"
        "Visual panel paths remain the imported v102/v103 case-level panels referenced in `stage1_focused_drift_source_case_preparation/focused_case_rows.csv`.\n",
    )
    write_text(
        out / "missed_positive_panels.md",
        "# Missed Positive Panels\n\n"
        "Bad cases selected by geometry-only but missed by semantic+geometry:\n\n"
        f"```text\nmissed_bad_cases={';'.join(missed_bad)}\n```\n\n"
        "These missed positives prevent semantic+geometry from becoming a high-recall action-entry oracle.\n",
    )

    state_counts = Counter(row["evidence_state"] for row in evidence_rows)
    summary = {
        "schema": "acl2_v104_stage2_summary_v1",
        "stage": 2,
        "semantic_diagnostic_pass": semantic_diagnostic_pass,
        "semantic_action_entry_pass": semantic_action_entry_pass,
        "semantic_good_FPR_reduction": fpr_reduction,
        "semantic_balanced_accuracy_gain": ba_gain,
        "semantic_rescued_good_case_count": len(rescued_good),
        "semantic_missed_bad_case_count": len(missed_bad),
        "state_counts": dict(state_counts),
        "runtime_action_allowed": False,
        "blocker": "" if semantic_action_entry_pass else "semantic_safety_filter_only_or_provider_controls_missing",
    }
    write_json(out / "stage2_summary.json", summary)
    return summary


def build_stage3(stage1: dict[str, Any], stage2: dict[str, Any]) -> dict[str, Any]:
    out = ROOT / "stage3_state_machine"
    evidence_rows = read_csv(ROOT / "stage2_oracle/evidence_state_rows.csv")
    demotion_case_rows = read_csv(ROOT / "stage1_provider/tracked_local_provider_demotion_case_rows.csv")
    demotion_by_case = index_by_case(demotion_case_rows)
    demotion_rule_case_ids = set(stage1.get("demotion_rule_selected_case_ids", []))
    boundary_rows: list[dict[str, Any]] = []
    anchor_rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []
    tracked_local_rows: list[dict[str, Any]] = []

    for row in evidence_rows:
        case_id = row.get("case_id", "")
        state = row.get("evidence_state", "CONTEXT_ONLY")
        demotion_row = demotion_by_case.get(case_id, {})
        demotion_rule_hit = case_id in demotion_rule_case_ids
        if not stage1.get("stage1_provider_pass") and demotion_rule_hit:
            boundary_state = "LOCAL_REGISTRATION_ONLY_DIAGNOSTIC"
            anchor_state = "LOCAL_TRACKED_INSTANCE_DIAGNOSTIC"
            edge_state = "BLOCK_SCALE_ROUTE_LOCAL_DIAGNOSTIC_ONLY"
        elif not stage1.get("stage1_provider_pass"):
            boundary_state = "DELAY_UPDATE" if state == "DELAY_UPDATE" else "CONTEXT_ONLY"
            anchor_state = "UNSUPPORTED" if state == "DELAY_UPDATE" else "CONTEXT_ONLY"
            edge_state = "BLOCK_SCALE_ROUTE"
        elif state == "SCALE_ELIGIBLE":
            boundary_state = "ALLOW_UPDATE"
            anchor_state = "LANDMARK"
            edge_state = "ALLOW_SCALE_ROUTE"
        elif state == "HOLD_PREVIOUS":
            boundary_state = "HOLD_PREVIOUS"
            anchor_state = "REFERENCE"
            edge_state = "DAMP_SCALE_ROUTE"
        elif state == "REJECT_SCALE_EVIDENCE":
            boundary_state = "REJECT_UPDATE"
            anchor_state = "DYNAMIC_OR_BOUNDARY_RISK"
            edge_state = "BLOCK_SCALE_ROUTE"
        else:
            boundary_state = "CONTEXT_ONLY"
            anchor_state = "CONTEXT_ONLY"
            edge_state = "ALLOW_CONTEXT_ROUTE"
        boundary_rows.append(
            {
                "case_id": case_id,
                "seq": row.get("seq", ""),
                "boundary_state": boundary_state,
                "source_evidence_state": state,
                "provider_pass": stage1.get("stage1_provider_pass", False),
                "runtime_action_allowed": False,
            }
        )
        anchor_rows.append(
            {
                "case_id": case_id,
                "anchor_id": "",
                "anchor_state": anchor_state,
                "semantic_role": "",
                "current_support": row.get("current_support_score", "0.0"),
                "scale_observability": row.get("scale_observability_score", ""),
                "provider_pass": stage1.get("stage1_provider_pass", False),
                "claim_level": "case_level_state_machine_diagnostic",
            }
        )
        edge_rows.append(
            {
                "case_id": case_id,
                "query_head": "",
                "anchor_id": "",
                "edge_state": edge_state,
                "query_head_local_edge_available": False,
                "provider_pass": stage1.get("stage1_provider_pass", False),
                "claim_level": "edge_state_placeholder_blocked_by_provider",
            }
        )
        if demotion_row:
            if demotion_rule_hit:
                tracked_local_state = "LOCAL_REGISTRATION_ONLY_DIAGNOSTIC"
                first_blocker = "single_case_provider_pass_only_no_formal_lifecycle_or_controls"
            elif to_bool(demotion_row.get("focused_safe_good_control", False)):
                tracked_local_state = "SAFE_GOOD_REJECT_LOCAL_PROVIDER"
                first_blocker = "safe_good_control_or_cell_count_counterexample"
            elif to_int(demotion_row.get("fullquery_direct_witness_seed_count", 0)) <= 0:
                tracked_local_state = "NO_LOCAL_TRACKED_DIRECT_WITNESS"
                first_blocker = "fullquery_direct_witness_missing"
            else:
                tracked_local_state = "LOCAL_TRACKED_DIAGNOSTIC_REJECTED_BY_RULE"
                first_blocker = "below_direct_seed_demotion_threshold_or_query_head_control_missing"
            tracked_local_rows.append(
                {
                    "case_id": case_id,
                    "case_role": demotion_row.get("case_role", classify_case(case_id)),
                    "selected_action_case": case_id in SELECTED_ACTION_CASES,
                    "tracked_local_state": tracked_local_state,
                    "demotion_rule_name": stage1.get("demotion_rule_name", ""),
                    "demotion_rule_hit": demotion_rule_hit,
                    "fullquery_direct_witness_seed_count": demotion_row.get(
                        "fullquery_direct_witness_seed_count", 0
                    ),
                    "fullquery_same_masklet_instance_topk_cell_count": demotion_row.get(
                        "fullquery_same_masklet_instance_topk_cell_count", 0
                    ),
                    "fullquery_counterfactual_query_head_count": demotion_row.get(
                        "fullquery_counterfactual_query_head_count", 0
                    ),
                    "focused_safe_good_control": demotion_row.get("focused_safe_good_control", False),
                    "focused_swa_bad_strict_candidate": demotion_row.get(
                        "focused_swa_bad_strict_candidate", False
                    ),
                    "scale_gauge_promotion_allowed": False,
                    "runtime_action_allowed": False,
                    "first_blocker": first_blocker,
                    "claim_level": "tracked_local_provider_state_machine_diagnostic_no_runtime",
                }
            )

    boundary_counts = Counter(row["boundary_state"] for row in boundary_rows)
    anchor_counts = Counter(row["anchor_state"] for row in anchor_rows)
    tracked_local_counts = Counter(row["tracked_local_state"] for row in tracked_local_rows)
    state_coverage = len(boundary_rows) / len(evidence_rows) if evidence_rows else 0.0
    selected_boundary_rows = [r for r in boundary_rows if r["case_id"] in SELECTED_ACTION_CASES]
    delay_hold_positive_cases = [
        r["case_id"]
        for r in selected_boundary_rows
        if r["case_id"] in STRICT_POSITIVES | EXPLORATORY_POSITIVES and r["boundary_state"] in {"DELAY_UPDATE", "HOLD_PREVIOUS"}
    ]
    query_head_edges_available = any(to_bool(r["query_head_local_edge_available"]) for r in edge_rows)
    context_covers_stable_scale = False
    tracked_local_diagnostic_case_ids = [
        row["case_id"]
        for row in tracked_local_rows
        if row["tracked_local_state"] == "LOCAL_REGISTRATION_ONLY_DIAGNOSTIC"
    ]
    tracked_local_safe_good_hit_count = sum(
        1
        for row in tracked_local_rows
        if row["tracked_local_state"] == "LOCAL_REGISTRATION_ONLY_DIAGNOSTIC"
        and to_bool(row.get("focused_safe_good_control", False))
    )
    stage3_pass = (
        state_coverage >= 0.80
        and len(boundary_counts) >= 2
        and len(set(delay_hold_positive_cases)) >= 2
        and not context_covers_stable_scale
        and query_head_edges_available
        and bool(stage1.get("stage1_provider_pass"))
        and bool(stage2.get("semantic_action_entry_pass"))
    )

    write_csv(out / "boundary_state_rows.csv", boundary_rows)
    write_csv(out / "anchor_state_rows.csv", anchor_rows)
    write_csv(out / "query_head_edge_state_rows.csv", edge_rows)
    write_csv(out / "tracked_local_state_rows.csv", tracked_local_rows)
    write_text(
        out / "state_transition_visual_panels.md",
        "# Stage 3 State Transition Visual Panels\n\n"
        "No new strict provider visual panels were generated because Stage 1 failed. "
        "Existing case-level RGB/semantic/residual panels remain referenced from v103 focused case rows; v104 action visualization is blocked.\n\n"
        "Tracked local-provider demotion was evaluated as a state-machine diagnostic only. "
        "It may label a case as LOCAL_REGISTRATION_ONLY_DIAGNOSTIC, but scale/gauge promotion and runtime action remain blocked until formal lifecycle/provider controls and query/head-local edges exist.\n",
    )
    summary = {
        "schema": "acl2_v104_stage3_state_machine_summary_v1",
        "stage": 3,
        "state_coverage": state_coverage,
        "boundary_state_counts": dict(boundary_counts),
        "anchor_state_counts": dict(anchor_counts),
        "tracked_local_state_counts": dict(tracked_local_counts),
        "tracked_local_diagnostic_case_count": len(tracked_local_diagnostic_case_ids),
        "tracked_local_diagnostic_case_ids": tracked_local_diagnostic_case_ids,
        "tracked_local_safe_good_hit_case_count": tracked_local_safe_good_hit_count,
        "tracked_local_scale_gauge_promotion_allowed": False,
        "state_class_count": len(boundary_counts),
        "delay_hold_positive_case_count": len(set(delay_hold_positive_cases)),
        "query_head_local_edges_available": query_head_edges_available,
        "context_only_does_not_cover_stable_scale_anchors": not context_covers_stable_scale,
        "stage3_state_machine_pass": stage3_pass,
        "runtime_action_allowed": False,
        "tracked_local_state_rows": rel(out / "tracked_local_state_rows.csv"),
        "blocker": "" if stage3_pass else "provider_or_query_head_local_edges_missing",
    }
    write_json(out / "state_machine_summary.json", summary)
    return summary


def build_stage4(stage1: dict[str, Any], stage2: dict[str, Any], stage3: dict[str, Any]) -> dict[str, Any]:
    out = ROOT / "stage4_action_surface"
    paths = v103_paths()
    sim_rows = read_csv(paths["stage4_sim_rows"])
    inventory_rows = read_csv(paths["stage4_inventory"])
    imported_stage4 = read_json(paths["stage4_summary"])

    stage4_prereq_pass = (
        bool(stage1.get("stage1_provider_pass"))
        and bool(stage2.get("semantic_action_entry_pass"))
        and bool(stage3.get("stage3_state_machine_pass"))
    )
    runtime_action_allowed = False
    action_surface_pass = False
    copied_rows = []
    for row in sim_rows:
        copied = dict(row)
        copied["v104_runtime_action_allowed"] = False
        copied["v104_claim_level"] = "imported_simulator_no_runtime"
        copied_rows.append(copied)
    write_csv(out / "action_surface_simulator_rows.csv", copied_rows)
    write_csv(out / "action_surface_inventory_rows.csv", inventory_rows)
    write_text(
        out / "no_runtime_action_reason.md",
        "# Stage 4 Runtime Action Blocked\n\n"
        "Stage 4 is not launched because v104 Stage 1 strict provider and provider controls did not pass. "
        "Imported v103 simulator/action-surface rows are retained only as diagnostic evidence.\n",
    )
    write_text(
        out / "admission_false_positive_panels.md",
        "# Admission False Positive Panels\n\n"
        "No new v104 runtime false-positive panels were generated. v103 diagnostic good-harm rows remain imported; "
        "runtime action is blocked before new action execution.\n",
    )

    best_rows = [r for r in sim_rows if to_bool(r.get("action_metric_gate_pass_without_provider"))]
    best = best_rows[0] if best_rows else {}
    summary = {
        "schema": "acl2_v104_stage4_action_surface_summary_v1",
        "stage": 4,
        "stage4_prereq_pass": stage4_prereq_pass,
        "runtime_action_allowed": runtime_action_allowed,
        "stage4_action_surface_pass": action_surface_pass,
        "imported_v103_action_surface_pass": imported_stage4.get("stage4_branch_a_action_surface_pass", False),
        "imported_v103_simulator_pass": imported_stage4.get("stage4_branch_a_simulator_pass", False),
        "best_imported_action_id_without_provider": best.get("action_id", ""),
        "best_imported_bad_l3_improvement_without_provider": to_float(best.get("sim_best_bad_l3_improvement", "")),
        "best_imported_safe_good_harm_max_without_provider": to_float(best.get("sim_safe_good_harm_max", "")),
        "blocker": "stage1_stage2_stage3_action_entry_not_authorized",
    }
    write_json(out / "action_surface_summary.json", summary)
    return summary


def build_stage5(stage1: dict[str, Any]) -> dict[str, Any]:
    out = ROOT / "stage5_ttt_write_to_use"
    paths = v103_paths()
    join_rows = read_csv(paths["join_ladder_rows"])
    gap_rows = read_csv(paths["per_anchor_gap_rows"])
    imported_stage6 = read_json(paths["stage6_summary"])

    ttt_rows: list[dict[str, Any]] = []
    for row in join_rows:
        ttt_rows.append(
            {
                "case_id": row.get("case_id", ""),
                "anchor_id": "",
                "masklet_id": "",
                "component_id": "",
                "z_write": "",
                "z_cache": "",
                "z_current": "",
                "r_write_cache": "",
                "r_cache_current": "",
                "r_ref_current": "",
                "TTT_write_time": "",
                "SWA_cache_presence": row.get("swa_raw_transport_metric_available", ""),
                "SWA_query_hit": row.get("external_token_masklet_query_best_frac", ""),
                "SWA_topk_use": row.get("external_token_masklet_topk_best_frac", ""),
                "READ_current_support": row.get("READ_current_support_mean", ""),
                "later_L3_handoff": row.get("relative_improvement_vs_baseline_head10_to_tail10_pose_sim3_rmse_m", ""),
                "later_L4_drift": "",
                "full_chain_available": False,
                "first_blocker": row.get("first_blocker", "per_anchor_write_cache_current_query_read_l3_chain_missing"),
            }
        )
    write_csv(out / "ttt_write_to_use_chain_rows.csv", ttt_rows)
    write_csv(out / "per_anchor_chain_gap_rows.csv", gap_rows)

    full_chain_case_coverage = 0.0
    r_write_cache_nonempty = 0
    r_cache_current_nonempty = 0
    r_ref_current_nonempty = 0
    later_l3_available = any(row.get("later_L3_handoff") not in {"", None} for row in ttt_rows)
    query_head_controls_available = bool(stage1.get("query_head_random_control_available"))
    stage5_pass = (
        full_chain_case_coverage >= 0.80
        and r_write_cache_nonempty > 0
        and r_cache_current_nonempty > 0
        and r_ref_current_nonempty > 0
        and later_l3_available
        and query_head_controls_available
    )
    write_text(
        out / "per_anchor_chain_missing_report.md",
        "# Stage 5 Per-Anchor Chain Missing Report\n\n"
        "The imported v103/v101 rows do not materialize per-anchor z_write/z_cache/z_current residual links with later READ/SWA/L3 evidence. "
        "TTT runtime refresh/expire/transient-only actions are blocked.\n",
    )
    write_text(
        out / "chain_materialization_blocker.md",
        "# TTT Chain Materialization Blocker\n\n"
        "Do not use write mass proxy. The write/cache/current/read/L3 full chain is incomplete, so Stage 4 scheme E remains diagnostic-only.\n",
    )
    summary = {
        "schema": "acl2_v104_stage5_ttt_write_to_use_summary_v1",
        "stage": 5,
        "full_chain_case_coverage": full_chain_case_coverage,
        "r_write_cache_nonempty": r_write_cache_nonempty,
        "r_cache_current_nonempty": r_cache_current_nonempty,
        "r_ref_current_nonempty": r_ref_current_nonempty,
        "later_L3_available": later_l3_available,
        "query_head_controls_available": query_head_controls_available,
        "stage5_ttt_full_chain_pass": stage5_pass,
        "imported_v103_stage6_branch_d_ttt_lifecycle_pass": imported_stage6.get("stage6_branch_d_ttt_lifecycle_pass", False),
        "runtime_action_allowed": False,
        "blocker": "TTT_CHAIN_INCOMPLETE_NO_ACTION",
    }
    write_json(out / "ttt_write_to_use_summary.json", summary)
    return summary


def build_stage6(stage2: dict[str, Any], stage3: dict[str, Any], stage4: dict[str, Any]) -> dict[str, Any]:
    out = ROOT / "stage6_cue_distillation_blocked"
    prereq_pass = (
        bool(stage2.get("semantic_action_entry_pass"))
        and bool(stage3.get("stage3_state_machine_pass"))
        and bool(stage4.get("stage4_action_surface_pass"))
    )
    rows = [
        {"cue_family": "geometry_only", "evaluated": True, "runtime_ready": False, "reason": "baseline diagnostic only"},
        {"cue_family": "semantic_only", "evaluated": True, "runtime_ready": False, "reason": "low recall diagnostic"},
        {"cue_family": "semantic_geometry", "evaluated": True, "runtime_ready": False, "reason": "safety filter only"},
        {"cue_family": "semantic_geometry_internal", "evaluated": True, "runtime_ready": False, "reason": "provider/action gate missing"},
        {"cue_family": "provider_backed_state_machine", "evaluated": False, "runtime_ready": False, "reason": "strict provider failed"},
    ]
    write_csv(out / "cue_family_rows.csv", rows)
    write_text(
        out / "cue_distillation_gap_report.md",
        "# Stage 6 Cue Distillation Blocked\n\n"
        "Stage 6 requires Stage 2/3/4 pass. v104 did not satisfy strict provider/action-entry gates, so no runtime cue distillation or pilot is authorized.\n",
    )
    summary = {
        "schema": "acl2_v104_stage6_cue_distillation_summary_v1",
        "stage": 6,
        "stage6_prereq_pass": prereq_pass,
        "stage6_cue_distillation_pass": False,
        "runtime_action_allowed": False,
        "blocker": "stage2_stage3_stage4_prerequisites_failed",
    }
    write_json(out / "cue_distillation_summary.json", summary)
    return summary


def build_stage7(
    stage1: dict[str, Any],
    stage2: dict[str, Any],
    stage3: dict[str, Any],
    stage4: dict[str, Any],
    stage5: dict[str, Any],
    stage6: dict[str, Any],
) -> dict[str, Any]:
    out = ROOT / "stage7_final_decision"
    runtime_allowed = bool(stage4.get("runtime_action_allowed"))
    full_validation_allowed = False
    if not stage1.get("stage1_provider_pass"):
        taxonomy = "PROVIDER_BLOCKED_NO_ACTION"
    elif not stage2.get("semantic_diagnostic_pass"):
        taxonomy = "SEMANTIC_INCREMENT_FAIL_GEOMETRY_ONLY"
    elif stage3.get("stage3_state_machine_pass") and not stage4.get("stage4_action_surface_pass"):
        taxonomy = "STATE_MACHINE_DIAGNOSTIC_PASS_ACTION_BLOCKED"
    elif stage4.get("stage4_action_surface_pass") and runtime_allowed:
        taxonomy = "ACTION_SURFACE_PASS_RUNTIME_READY"
    else:
        taxonomy = "STATE_MACHINE_DIAGNOSTIC_PASS_ACTION_BLOCKED"

    answers = {
        "did_strict_instance_level_provider_pass": bool(stage1.get("stage1_provider_pass")),
        "did_semantic_add_value_beyond_geometry": bool(stage2.get("semantic_diagnostic_pass")),
        "did_state_machine_avoid_good_false_positives": False,
        "did_any_action_improve_L3_by_ge_5pct": False,
        "did_good_controls_remain_safe": False,
        "did_TTT_write_to_use_chain_materialize": bool(stage5.get("stage5_ttt_full_chain_pass")),
        "is_runtime_action_allowed": runtime_allowed,
        "is_full_validation_allowed": full_validation_allowed,
    }
    blocking_requirements = []
    if not stage1.get("stage1_provider_pass"):
        blocking_requirements.append("strict_instance_provider_pass_required")
    if not stage1.get("provider_controls_available"):
        blocking_requirements.append("provider_controls_required")
    if not stage1.get("stage4_runtime_action_controls_available"):
        blocking_requirements.append("stage4_runtime_action_controls_required")
    if not stage1.get("query_head_witness_action_ready_query_head_local_edge"):
        blocking_requirements.append("query_head_local_edges_required")
    if not stage5.get("stage5_ttt_full_chain_pass"):
        blocking_requirements.append("TTT_full_chain_required_for_TTT_action")
    decision = {
        "schema": "acl2_v104_final_decision_v1",
        "final_taxonomy": taxonomy,
        "goal_achieved": taxonomy in {"FULL_METHOD_SUCCESS", "RUNTIME_PILOT_PASS_FULL_VALIDATION_READY"},
        "full_method_success": taxonomy == "FULL_METHOD_SUCCESS",
        "runtime_action_allowed": runtime_allowed,
        "full_validation_run": False,
        "full_validation_allowed": full_validation_allowed,
        "stage1_provider_pass": stage1.get("stage1_provider_pass", False),
        "stage2_semantic_diagnostic_pass": stage2.get("semantic_diagnostic_pass", False),
        "stage2_semantic_action_entry_pass": stage2.get("semantic_action_entry_pass", False),
        "stage3_state_machine_pass": stage3.get("stage3_state_machine_pass", False),
        "stage4_action_surface_pass": stage4.get("stage4_action_surface_pass", False),
        "stage5_ttt_full_chain_pass": stage5.get("stage5_ttt_full_chain_pass", False),
        "stage6_cue_distillation_pass": stage6.get("stage6_cue_distillation_pass", False),
        "answers": answers,
        "blocking_requirements": blocking_requirements,
        "next_decisive_experiment": "materialize stable/lifecycle raw-sidecar identity and nonproxy current-support on the selected action set, then add Stage4 runtime action controls before any action validation.",
    }
    write_json(out / "final_decision.json", decision)
    write_text(
        out / "final_report.md",
        f"""# ACL2 v104 Final Report

```text
final_taxonomy={taxonomy}
runtime_action_allowed={runtime_allowed}
full_validation_allowed={full_validation_allowed}
```

Required answers:

```text
1. strict instance-level provider pass: {answers["did_strict_instance_level_provider_pass"]}
2. semantic value beyond geometry: {answers["did_semantic_add_value_beyond_geometry"]}
3. state machine avoided good false positives: {answers["did_state_machine_avoid_good_false_positives"]}
4. any action improved L3 by >=5%: {answers["did_any_action_improve_L3_by_ge_5pct"]}
5. good controls remained safe: {answers["did_good_controls_remain_safe"]}
6. TTT write-to-use chain materialized: {answers["did_TTT_write_to_use_chain_materialize"]}
7. runtime action allowed: {answers["is_runtime_action_allowed"]}
8. full validation allowed: {answers["is_full_validation_allowed"]}
```

Conclusion:

v104 is blocked at the strict provider gate. Semantic evidence retains diagnostic safety-filter value, but no READ/SWA/TTT runtime action is authorized.
""",
    )
    write_text(
        ROOT / "visual_missing_reason.md",
        "# Visual Missing Reason\n\n"
        "No new v104 provider/action visual panels were generated because Stage 1 strict provider failed and Stage 4 runtime action was not authorized. "
        "Imported v102/v103 visual panel paths remain recorded in focused case rows.\n",
    )
    return decision


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    stage0 = build_stage0()
    stage1 = build_stage1()
    stage2 = build_stage2(stage1)
    stage3 = build_stage3(stage1, stage2)
    stage4 = build_stage4(stage1, stage2, stage3)
    stage5 = build_stage5(stage1)
    stage6 = build_stage6(stage2, stage3, stage4)
    decision = build_stage7(stage1, stage2, stage3, stage4, stage5, stage6)
    print(json.dumps({
        "root": rel(ROOT),
        "stage0_pass": stage0.get("stage0_pass"),
        "stage1_provider_pass": stage1.get("stage1_provider_pass"),
        "stage2_semantic_diagnostic_pass": stage2.get("semantic_diagnostic_pass"),
        "stage2_semantic_action_entry_pass": stage2.get("semantic_action_entry_pass"),
        "stage3_state_machine_pass": stage3.get("stage3_state_machine_pass"),
        "stage4_action_surface_pass": stage4.get("stage4_action_surface_pass"),
        "stage5_ttt_full_chain_pass": stage5.get("stage5_ttt_full_chain_pass"),
        "final_taxonomy": decision.get("final_taxonomy"),
        "runtime_action_allowed": decision.get("runtime_action_allowed"),
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
