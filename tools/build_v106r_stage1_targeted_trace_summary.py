#!/usr/bin/env python3
"""Summarize ACL2 v106R Stage1 targeted no-action traces."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "third_party/lingbot-map/benchmark"
TOOLS = ROOT / "tools"
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(TOOLS))

from benchmark.io.image import load_exr  # noqa: E402
from benchmark.io.intrinsics import read_intrinsics  # noqa: E402
from build_v105tf_lingbot_stage3_oracle import (  # noqa: E402
    SPECIAL_TOKENS,
    SemanticFrameLoader,
    semantic_role,
)
from build_v106r_stage1_memory_operation_map import (  # noqa: E402
    FEATURE_DEFS,
    pearson,
    quantile_thresholds,
    selected_metrics,
    topk_mask,
)


V106R = ROOT / "results/acl2_v106r_lingbot_memory_operation_discovery_semantic_aware_control"
STAGE1_TRACE = V106R / "stage1_memory_operation_map/targeted_trace"
WORKSPACE = STAGE1_TRACE / "workspace"
RAW_TRACE = STAGE1_TRACE / "raw_trace"
TARGET_MANIFEST = STAGE1_TRACE / "target_manifest.csv"
RUN_MANIFEST = STAGE1_TRACE / "run_manifest.csv"
NOTRACE_METHOD = "lingbot_map_v106r_stage1_targeted_notrace"
TRACE_METHOD = "lingbot_map_v106r_stage1_targeted_trace"
EPS = 1e-12


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def fnum(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    raw = row.get(key, "")
    if raw in {"", None}:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def bool_s(value: bool) -> str:
    return "true" if value else "false"


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def sha256_file(path: Path) -> str:
    if not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                rows.append(
                    {
                        "row_type": "trace_error",
                        "error": f"json_decode_error:{path.name}:{line_number}:{exc}",
                    }
                )
    return rows


def load_traj(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    mats: list[np.ndarray] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            vals = [float(x) for x in line.split()]
            if len(vals) != 13:
                raise ValueError(f"bad trajectory row in {path}: {line[:120]}")
            mat = np.eye(4, dtype=np.float64)
            mat[:3, :4] = np.asarray(vals[1:], dtype=np.float64).reshape(3, 4)
            mats.append(mat)
    if not mats:
        return None
    return np.stack(mats, axis=0)


def max_abs_diff_arrays(lhs: np.ndarray | None, rhs: np.ndarray | None) -> float | None:
    if lhs is None or rhs is None:
        return None
    if lhs.shape != rhs.shape:
        return math.inf
    diff = np.abs(lhs.astype(np.float64) - rhs.astype(np.float64))
    if diff.size == 0 or np.all(np.isnan(diff)):
        return 0.0
    return float(np.nanmax(diff))


def max_abs_diff_exr_dir(lhs_root: Path, rhs_root: Path, name: str) -> tuple[float | None, str]:
    lhs_files = sorted((lhs_root / name).glob("*.exr")) if (lhs_root / name).is_dir() else []
    rhs_files = sorted((rhs_root / name).glob("*.exr")) if (rhs_root / name).is_dir() else []
    if not lhs_files or not rhs_files:
        return None, f"{name}_exr_missing"
    if [path.name for path in lhs_files] != [path.name for path in rhs_files]:
        return math.inf, f"{name}_filename_mismatch"
    max_diff = 0.0
    for lhs_file, rhs_file in zip(lhs_files, rhs_files):
        lhs = load_exr(lhs_file)
        rhs = load_exr(rhs_file)
        if lhs.shape != rhs.shape:
            return math.inf, f"{name}_shape_mismatch:{lhs_file.name}"
        diff = np.abs(lhs.astype(np.float64) - rhs.astype(np.float64))
        if diff.size and not np.all(np.isnan(diff)):
            max_diff = max(max_diff, float(np.nanmax(diff)))
    return max_diff, ""


def load_intr(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    return read_intrinsics(path)


def target_rows_by_dataset() -> dict[str, dict[str, Any]]:
    targets = {row["target_id"]: row for row in read_csv(TARGET_MANIFEST)}
    out: dict[str, dict[str, Any]] = {}
    for row in read_csv(RUN_MANIFEST):
        if row["phase"] != "run_worker_trace":
            continue
        item = dict(targets[row["target_id"]])
        item.update(
            {
                "dataset": row["dataset"],
                "seq": row["seq"],
                "trace_file": row["trace_file"],
                "trace_global_idxs": row["trace_global_idxs"],
            }
        )
        out[row["dataset"]] = item
    return out


def compare_case(target: dict[str, Any], trace_rows: list[dict[str, Any]]) -> dict[str, Any]:
    dataset = target["dataset"]
    seq = target["seq"]
    notrace_root = WORKSPACE / dataset / seq / NOTRACE_METHOD
    trace_root = WORKSPACE / dataset / seq / TRACE_METHOD
    reasons: list[str] = []
    if not (notrace_root / ".complete.json").is_file():
        reasons.append("notrace_complete_missing")
    if not (trace_root / ".complete.json").is_file():
        reasons.append("trace_complete_missing")

    notrace_traj = notrace_root / "traj.txt"
    trace_traj = trace_root / "traj.txt"
    pose_sha_equal = sha256_file(notrace_traj) != "" and sha256_file(notrace_traj) == sha256_file(trace_traj)
    pose_diff = max_abs_diff_arrays(load_traj(notrace_traj), load_traj(trace_traj))
    intr_diff = max_abs_diff_arrays(load_intr(notrace_root / "intrinsics.txt"), load_intr(trace_root / "intrinsics.txt"))
    depth_diff, depth_reason = max_abs_diff_exr_dir(notrace_root, trace_root, "depth")
    conf_diff, conf_reason = max_abs_diff_exr_dir(notrace_root, trace_root, "confidence")

    if pose_diff is None:
        reasons.append("pose_missing")
    if intr_diff is None:
        reasons.append("intrinsics_missing")
    if depth_reason:
        reasons.append(depth_reason)
    if conf_reason:
        reasons.append(conf_reason)

    trace_error_count = sum(1 for row in trace_rows if row.get("row_type") == "trace_error")
    gca_count = sum(1 for row in trace_rows if row.get("row_type") == "gca_context_topk")
    kv_count = sum(1 for row in trace_rows if row.get("row_type") == "kv_cache_provenance")
    resolved_context = sum(
        1 for row in trace_rows
        if row.get("row_type") == "gca_context_topk" and str(row.get("key_context_role", "")).strip()
    )
    context_role_resolved_ratio = resolved_context / gca_count if gca_count else 0.0
    trace_payload_exists = bool(gca_count and kv_count)
    if not trace_payload_exists:
        reasons.append("trace_payload_empty_or_incomplete")
    if trace_error_count:
        reasons.append(f"trace_error_rows:{trace_error_count}")
    if context_role_resolved_ratio < 0.90:
        reasons.append(f"context_role_resolved_ratio_lt_0p90:{context_role_resolved_ratio}")

    pose_ok = pose_sha_equal or (pose_diff is not None and pose_diff <= 1e-6)
    depth_ok = depth_diff is not None and depth_diff <= 1e-6
    intr_ok = intr_diff is not None and intr_diff <= 1e-6
    conf_ok = conf_diff is not None and conf_diff <= 1e-6
    complete_ok = (notrace_root / ".complete.json").is_file() and (trace_root / ".complete.json").is_file()
    parity_pass = (
        complete_ok
        and pose_ok
        and depth_ok
        and intr_ok
        and conf_ok
        and trace_payload_exists
        and trace_error_count == 0
        and context_role_resolved_ratio >= 0.90
    )

    if pose_diff is not None and not pose_ok:
        reasons.append("pose_diff_gt_1e-6")
    if depth_diff is not None and not depth_ok:
        reasons.append("depth_diff_gt_1e-6")
    if intr_diff is not None and not intr_ok:
        reasons.append("intrinsics_diff_gt_1e-6")
    if conf_diff is not None and not conf_ok:
        reasons.append("confidence_diff_gt_1e-6")

    return {
        "schema": "acl2_v106r_stage1_targeted_trace_parity_row_v1",
        "target_id": target["target_id"],
        "target_kind": target["target_kind"],
        "dataset": dataset,
        "seq": seq,
        "notrace_method": NOTRACE_METHOD,
        "trace_method": TRACE_METHOD,
        "pose_sha_equal": bool_s(pose_sha_equal),
        "pose_max_abs_diff": "" if pose_diff is None else pose_diff,
        "depth_max_abs_diff": "" if depth_diff is None else depth_diff,
        "intrinsics_max_abs_diff": "" if intr_diff is None else intr_diff,
        "confidence_max_abs_diff": "" if conf_diff is None else conf_diff,
        "trace_payload_exists": bool_s(trace_payload_exists),
        "trace_row_count": len(trace_rows),
        "gca_context_topk_row_count": gca_count,
        "kv_cache_provenance_row_count": kv_count,
        "trace_error_row_count": trace_error_count,
        "context_role_resolved_ratio": context_role_resolved_ratio,
        "parity_pass": bool_s(parity_pass),
        "failure_reason": ";".join(dict.fromkeys(reasons)),
        "notrace_root": rel(notrace_root),
        "trace_root": rel(trace_root),
        "trace_file": rel(Path(target["trace_file"])),
    }


def aggregate_context(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], dict[str, Any]] = {}
    weights: defaultdict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in rows:
        if row.get("row_type") != "gca_context_topk":
            continue
        key = (
            row.get("target_id", ""),
            row.get("target_kind", ""),
            row.get("dataset", ""),
            row.get("seq", ""),
            row.get("method", ""),
            row.get("trace_backend", ""),
            row.get("cache_mode", ""),
            row.get("key_context_role", ""),
            row.get("key_token_role", ""),
            row.get("query_token_role", ""),
        )
        if key not in groups:
            groups[key] = {
                "schema": "acl2_v106r_stage1_targeted_context_role_token_row_v1",
                "target_id": key[0],
                "target_kind": key[1],
                "dataset": key[2],
                "seq": key[3],
                "method": key[4],
                "trace_backend": key[5],
                "cache_mode": key[6],
                "key_context_role": key[7],
                "key_token_role": key[8],
                "query_token_role": key[9],
                "row_count": 0,
                "global_idx_values": set(),
            }
        groups[key]["row_count"] += 1
        groups[key]["global_idx_values"].add(str(row.get("global_idx", "")))
        weights[key].append(float(row.get("attention_weight", 0.0) or 0.0))
    out: list[dict[str, Any]] = []
    for key, row in sorted(groups.items()):
        vals = weights[key]
        item = dict(row)
        item["attention_weight_sum"] = float(np.sum(vals)) if vals else 0.0
        item["attention_weight_mean"] = float(np.mean(vals)) if vals else 0.0
        item["attention_weight_max"] = float(np.max(vals)) if vals else 0.0
        item["global_idx_values"] = ",".join(sorted(x for x in item["global_idx_values"] if x != ""))
        out.append(item)
    return out


def read_sampling_frames(dataset: str, seq: str) -> list[int]:
    path = WORKSPACE / dataset / seq / "gt/sampling.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [int(item) for item in payload["frames"]]


def key_semantics_cached(
    seq: str,
    sample_position: int,
    token_offset: int,
    loader: SemanticFrameLoader,
    sample_frames: list[int],
    frame_cache: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    if not (0 <= sample_position < len(sample_frames)):
        return {
            "key_original_frame": "",
            "key_semantic_label": "",
            "key_semantic_role": "",
            "key_semantic_confidence": "",
        }
    original_frame = sample_frames[sample_position]
    if token_offset < SPECIAL_TOKENS:
        return {
            "key_original_frame": original_frame,
            "key_semantic_label": "special_token",
            "key_semantic_role": "TRAJECTORY_MEMORY_EVIDENCE",
            "key_semantic_confidence": "",
        }
    patch_index = int(token_offset) - SPECIAL_TOKENS
    if sample_position not in frame_cache:
        frame_cache[sample_position] = loader.load_frame(original_frame)
    sem = frame_cache[sample_position]
    patch_labels = sem["patch_labels"]
    patch_conf = sem["patch_confidence"]
    if not (0 <= patch_index < len(patch_labels)):
        return {
            "key_original_frame": original_frame,
            "key_semantic_label": "token_out_of_patch_grid",
            "key_semantic_role": "CONTEXT_ONLY",
            "key_semantic_confidence": "",
        }
    label_names = sem["label_names"]
    label_id = int(patch_labels[patch_index])
    label = label_names[label_id] if label_id < len(label_names) else "void"
    return {
        "key_original_frame": original_frame,
        "key_semantic_label": label,
        "key_semantic_role": semantic_role(label),
        "key_semantic_confidence": float(patch_conf[patch_index]),
    }


def summarize_semantic_frame(seq: str, original_frame: int, loader: SemanticFrameLoader) -> dict[str, Any]:
    sem = loader.load_frame(original_frame)
    labels = np.asarray(sem["patch_labels"], dtype=np.int64)
    conf = np.asarray(sem["patch_confidence"], dtype=np.float32)
    names = sem["label_names"]
    counts: defaultdict[str, int] = defaultdict(int)
    role_counts: defaultdict[str, int] = defaultdict(int)
    for label_id in labels:
        label = names[int(label_id)] if int(label_id) < len(names) else "void"
        counts[label] += 1
        role_counts[semantic_role(label)] += 1
    total = max(1, len(labels))
    top_labels = ";".join(f"{label}:{count}" for label, count in sorted(counts.items(), key=lambda item: item[1], reverse=True)[:8])
    dominant_label, dominant_count = max(counts.items(), key=lambda item: item[1]) if counts else ("", 0)
    return {
        "semantic_source": "pseudo_semantic_cache_from_kitti_preprocess_not_GT",
        "patch_count": total,
        "scale_reference_patch_frac": role_counts["SCALE_REFERENCE_EVIDENCE"] / total,
        "local_registration_patch_frac": role_counts["LOCAL_REGISTRATION_EVIDENCE"] / total,
        "context_only_patch_frac": role_counts["CONTEXT_ONLY"] / total,
        "reject_unreliable_patch_frac": role_counts["REJECT_UNRELIABLE"] / total,
        "semantic_confidence_mean": float(np.mean(conf)) if conf.size else 0.0,
        "semantic_confidence_p10": float(np.percentile(conf, 10)) if conf.size else 0.0,
        "dominant_semantic_label": dominant_label,
        "patch_purity": dominant_count / total,
        "top_labels": top_labels,
    }


def annotate_trace_rows(targets: dict[str, dict[str, Any]], trace_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_dataset = defaultdict(list)
    for row in trace_rows:
        by_dataset[row.get("dataset", "")].append(row)
    annotated: list[dict[str, Any]] = []
    loaders: dict[str, SemanticFrameLoader] = {}
    for dataset, rows in by_dataset.items():
        target = targets[dataset]
        seq = target["seq"]
        loader = loaders.setdefault(seq, SemanticFrameLoader(seq))
        sample_frames = read_sampling_frames(dataset, seq)
        last_invocation: dict[int, dict[str, Any]] = {}
        semantic_cache: dict[tuple[int, int], dict[str, Any]] = {}
        frame_cache: dict[int, dict[str, Any]] = {}
        for row in rows:
            row_type = row.get("row_type", "")
            global_idx = int(row.get("global_idx", -1))
            if row_type == "kv_cache_provenance":
                tokens_per_frame = int(row.get("tokens_per_frame", 0))
                q_seq_len = int(row.get("q_seq_len", 0))
                cached_frames = int(row.get("cached_frames", 0))
                last_invocation[global_idx] = {
                    "current_sample_position": cached_frames - 1,
                    "frame_eval_allowed": bool(tokens_per_frame > 0 and q_seq_len == tokens_per_frame),
                    "cached_frames": cached_frames,
                    "q_seq_len": q_seq_len,
                }
                continue
            if row_type != "gca_context_topk":
                continue
            inv = last_invocation.get(global_idx, {})
            current_pos = int(inv.get("current_sample_position", -1))
            allowed = bool(inv.get("frame_eval_allowed", False))
            key_frame_offset = int(row.get("key_frame_offset", -1))
            key_token_offset = int(row.get("key_token_offset", -1))
            sem_key = (key_frame_offset, key_token_offset)
            if sem_key not in semantic_cache:
                semantic_cache[sem_key] = key_semantics_cached(
                    seq,
                    key_frame_offset,
                    key_token_offset,
                    loader,
                    sample_frames,
                    frame_cache,
                )
            sem = semantic_cache[sem_key]
            annotated.append(
                {
                    "schema": "acl2_v106r_stage1_targeted_trace_semantic_key_row_v1",
                    "target_id": target["target_id"],
                    "target_kind": target["target_kind"],
                    "seq": seq,
                    "dataset": dataset,
                    "method": row.get("method", ""),
                    "global_idx": global_idx,
                    "head_idx": int(row.get("head_idx", -1)),
                    "current_sample_position": current_pos if allowed else "",
                    "current_original_frame": sample_frames[current_pos] if allowed and 0 <= current_pos < len(sample_frames) else "",
                    "frame_eval_allowed": allowed,
                    "key_sample_position": key_frame_offset,
                    "key_token_offset": key_token_offset,
                    "key_context_role": row.get("key_context_role", ""),
                    "key_token_role": row.get("key_token_role", ""),
                    "query_token_role": row.get("query_token_role", ""),
                    "query_token_index": row.get("query_token_index", ""),
                    "topk_rank": row.get("topk_rank", ""),
                    "attention_weight": float(row.get("attention_weight", 0.0) or 0.0),
                    **sem,
                }
            )
    return annotated


def build_head_features(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if not row.get("frame_eval_allowed"):
            continue
        current_pos = row.get("current_sample_position", "")
        if current_pos == "":
            continue
        grouped[(str(row["target_id"]), str(row["seq"]), int(current_pos), int(row["head_idx"]))].append(row)

    out: list[dict[str, Any]] = []
    for (target_id, seq, pos, head_idx), group in sorted(grouped.items()):
        total = sum(float(row["attention_weight"]) for row in group)
        total = max(total, EPS)
        by_context = defaultdict(float)
        by_role = defaultdict(float)
        by_context_role = defaultdict(float)
        target_kind = str(group[0]["target_kind"])
        dataset = str(group[0]["dataset"])
        original_frame = group[0].get("current_original_frame", "")
        for row in group:
            weight = float(row["attention_weight"])
            by_context[str(row.get("key_context_role", ""))] += weight
            by_role[str(row.get("key_semantic_role", ""))] += weight
            by_context_role[(str(row.get("key_context_role", "")), str(row.get("key_semantic_role", "")))] += weight
        out.append(
            {
                "schema": "acl2_v106r_stage1_targeted_headlocal_frame_head_feature_v1",
                "target_id": target_id,
                "target_kind": target_kind,
                "dataset": dataset,
                "seq": seq,
                "sample_position": pos,
                "original_frame": original_frame,
                "head_idx": head_idx,
                "head_trace_topk_rows": len(group),
                "head_trace_topk_attention_sum": total,
                "scale_reference_context_attention_frac": by_context["scale_reference_context"] / total,
                "local_window_context_attention_frac": by_context["local_window_context"] / total,
                "current_or_latest_frame_attention_frac": by_context["current_or_latest_frame"] / total,
                "semantic_scale_reference_attention_frac": by_role["SCALE_REFERENCE_EVIDENCE"] / total,
                "semantic_local_registration_attention_frac": by_role["LOCAL_REGISTRATION_EVIDENCE"] / total,
                "semantic_context_only_attention_frac": by_role["CONTEXT_ONLY"] / total,
                "semantic_reject_unreliable_attention_frac": by_role["REJECT_UNRELIABLE"] / total,
                "scale_context_reject_attention_frac": by_context_role[("scale_reference_context", "REJECT_UNRELIABLE")] / max(by_context["scale_reference_context"], EPS),
                "scale_context_structure_attention_frac": by_context_role[("scale_reference_context", "SCALE_REFERENCE_EVIDENCE")] / max(by_context["scale_reference_context"], EPS),
                "local_context_reject_attention_frac": by_context_role[("local_window_context", "REJECT_UNRELIABLE")] / max(by_context["local_window_context"], EPS),
            }
        )
    return out


def build_frame_rows(targets: dict[str, dict[str, Any]], head_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    identities = sorted({(row["target_id"], row["dataset"], row["seq"], row["sample_position"], row["original_frame"], row["target_kind"]) for row in head_rows})
    loaders: dict[str, SemanticFrameLoader] = {}
    out: list[dict[str, Any]] = []
    for target_id, dataset, seq, sample_position, original_frame, target_kind in identities:
        target = targets[dataset]
        loader = loaders.setdefault(seq, SemanticFrameLoader(seq))
        sem = summarize_semantic_frame(seq, int(original_frame), loader)
        semantic_trust = sem["semantic_confidence_mean"] * sem["patch_purity"] * sem["patch_purity"]
        handoff = fnum(target, "handoff_transfer_penalty")
        out.append(
            {
                "schema": "acl2_v106r_stage1_targeted_frame_semantic_geometry_row_v1",
                "target_id": target_id,
                "target_kind": target_kind,
                "dataset": dataset,
                "seq": seq,
                "sample_position": sample_position,
                "original_frame": original_frame,
                "window_index": target["window_index"],
                "target_frame_start": target["target_frame_start"],
                "target_frame_end": target["target_frame_end"],
                "L3_handoff_metric_nearby": handoff,
                "rolling_drift_metric_nearby": handoff,
                "rolling_metric_source": "target_window_handoff_transfer_penalty_same_as_l3_for_targeted_trace",
                "adjacent_log_scale_jump_nearby": target["adjacent_log_scale_jump"],
                "local_window_ATE_nearby": target["local_sim3_ate_rmse_m"],
                "bad_label": target_kind == "high_l3",
                "good_label": target_kind == "safe_good_low_drift",
                "semantic_trust": semantic_trust,
                **sem,
            }
        )
    return out


def build_operation_rows(head_rows: list[dict[str, Any]], frame_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frames = {(row["target_id"], str(row["sample_position"])): row for row in frame_rows}
    out: list[dict[str, Any]] = []
    for head in head_rows:
        frame = frames.get((head["target_id"], str(head["sample_position"])))
        if frame is None:
            continue
        for feature in FEATURE_DEFS:
            out.append(
                {
                    "schema": "acl2_v106r_stage1_targeted_memory_operation_row_v1",
                    "target_id": head["target_id"],
                    "target_kind": head["target_kind"],
                    "seq_id": head["seq"],
                    "frame_id": head["original_frame"],
                    "window_id": frame["window_index"],
                    "boundary_id": f"{head['target_id']}:{head['sample_position']}",
                    "sample_position": head["sample_position"],
                    "operation_type": feature["operation_type"],
                    "context_role": feature["context_role"],
                    "token_type": feature["token_type"],
                    "head_id": head["head_idx"],
                    "attention_mass": fnum(head, feature["column"]),
                    "feature_family": feature["feature_family"],
                    "feature_column": feature["column"],
                    "semantic_role": feature["semantic_role"],
                    "dominant_semantic_label": frame["dominant_semantic_label"],
                    "semantic_confidence": frame["semantic_confidence_mean"],
                    "patch_purity": frame["patch_purity"],
                    "semantic_trust": frame["semantic_trust"],
                    "geometry_support_score": 1.0 - fnum(frame, "reject_unreliable_patch_frac"),
                    "metric_consistency_score": 1.0 / (1.0 + fnum(frame, "L3_handoff_metric_nearby")),
                    "scale_observability_score": fnum(frame, "scale_reference_patch_frac"),
                    "L3_handoff_metric_nearby": frame["L3_handoff_metric_nearby"],
                    "rolling_drift_metric_nearby": frame["rolling_drift_metric_nearby"],
                    "good_or_bad_label": "bad" if parse_bool(frame["bad_label"]) else "good",
                    "bad_label": frame["bad_label"],
                    "good_label": frame["good_label"],
                    "trace_scope": "v106r_targeted_64f_or_96f_high_l3_and_safe_good_head_resolved",
                    "source_artifact": rel(STAGE1_TRACE / "targeted_headlocal_frame_head_features.csv"),
                }
            )
    return out


def best_threshold(values: list[float], bad: list[bool], good: list[bool]) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for direction in ("ge", "le"):
        for threshold in quantile_thresholds(values):
            mask = [(value >= threshold) if direction == "ge" else (value <= threshold) for value in values]
            metrics = selected_metrics(mask, bad, good)
            if metrics["selected_rows"] == 0:
                continue
            metrics["same_count_random_margin"] = metrics["bad_recall"] - (metrics["selected_rows"] / len(values))
            metrics["threshold_direction"] = direction
            metrics["threshold_value"] = threshold
            score = (
                metrics["balanced_accuracy"],
                metrics["same_count_random_margin"],
                metrics["bad_recall"],
                -metrics["good_FPR"],
            )
            if best is None or score > best["_score"]:
                best = {**metrics, "_score": score}
    if best is None:
        return {
            "selected_rows": 0,
            "selected_bad_rows": 0,
            "selected_good_rows": 0,
            "bad_recall": 0.0,
            "good_FPR": 0.0,
            "balanced_accuracy": 0.0,
            "same_count_random_margin": 0.0,
            "threshold_direction": "ge",
            "threshold_value": 0.0,
        }
    best.pop("_score", None)
    return best


def rank_levers(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_lever: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        lever = "|".join([row["operation_type"], row["context_role"], row["token_type"], row["feature_family"]])
        by_lever[lever].append(row)
    feature_by_lever = {
        f"{item['operation_type']}|{item['context_role']}|{item['token_type']}|{item['feature_family']}": item
        for item in FEATURE_DEFS
    }
    rank_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    for lever, group in sorted(by_lever.items()):
        values = [fnum(row, "attention_mass") for row in group]
        l3 = [fnum(row, "L3_handoff_metric_nearby") for row in group]
        rolling = [fnum(row, "rolling_drift_metric_nearby") for row in group]
        bad = [parse_bool(row["bad_label"]) for row in group]
        good = [parse_bool(row["good_label"]) for row in group]
        best = best_threshold(values, bad, good)
        mask = [(value >= best["threshold_value"]) if best["threshold_direction"] == "ge" else (value <= best["threshold_value"]) for value in values]
        shuffled_mask = topk_mask(values[7:] + values[:7], str(best["threshold_direction"]), int(best["selected_rows"])) if values else []
        shuffled = selected_metrics(shuffled_mask, bad, good) if values else {"bad_recall": 0.0}
        selected_bad_by_seq: defaultdict[str, int] = defaultdict(int)
        selected_by_seq: defaultdict[str, int] = defaultdict(int)
        selected_bad_by_target: defaultdict[str, int] = defaultdict(int)
        for keep, row, is_bad in zip(mask, group, bad):
            if not keep:
                continue
            selected_by_seq[str(row["seq_id"])] += 1
            if is_bad:
                selected_bad_by_seq[str(row["seq_id"])] += 1
                selected_bad_by_target[str(row["target_id"])] += 1
        total_selected_bad = sum(selected_bad_by_seq.values())
        feature = feature_by_lever[lever]
        rank_rows.append(
            {
                "schema": "acl2_v106r_stage1_targeted_memory_lever_rank_row_v1",
                "lever_id": lever,
                "operation_type": feature["operation_type"],
                "context_role": feature["context_role"],
                "token_type": feature["token_type"],
                "feature_family": feature["feature_family"],
                "feature_column": feature["column"],
                "case_count": len(group),
                "sequence_coverage": len({row["seq_id"] for row in group}),
                "target_coverage": len({row["target_id"] for row in group}),
                "selected_sequence_coverage": sum(1 for value in selected_by_seq.values() if value > 0),
                "bad_recall": best["bad_recall"],
                "good_FPR": best["good_FPR"],
                "balanced_accuracy": best["balanced_accuracy"],
                "selected_rows": best["selected_rows"],
                "selected_bad_rows": best["selected_bad_rows"],
                "selected_good_rows": best["selected_good_rows"],
                "threshold_direction": best["threshold_direction"],
                "threshold_value": best["threshold_value"],
                "abs_corr_L3": abs(pearson(values, l3)),
                "signed_corr_L3": pearson(values, l3),
                "abs_corr_rolling": abs(pearson(values, rolling)),
                "signed_corr_rolling": pearson(values, rolling),
                "same_count_random_margin": best["same_count_random_margin"],
                "semantic_shuffle_margin": best["bad_recall"] - float(shuffled["bad_recall"]),
                "positive_sequence_max_frac": max(selected_bad_by_seq.values()) / total_selected_bad if total_selected_bad else 0.0,
                "positive_target_max_frac": max(selected_bad_by_target.values()) / total_selected_bad if total_selected_bad else 0.0,
                "operation_interpretation": feature["interpretation"],
            }
        )
        for seq in sorted({row["seq_id"] for row in group}):
            idxs = [idx for idx, row in enumerate(group) if row["seq_id"] == seq]
            seq_metrics = selected_metrics([mask[idx] for idx in idxs], [bad[idx] for idx in idxs], [good[idx] for idx in idxs])
            split_rows.append(
                {
                    "schema": "acl2_v106r_stage1_targeted_memory_lever_sequence_split_row_v1",
                    "lever_id": lever,
                    "seq_id": seq,
                    "case_count": len(idxs),
                    "selected_rows": seq_metrics["selected_rows"],
                    "selected_bad_rows": seq_metrics["selected_bad_rows"],
                    "selected_good_rows": seq_metrics["selected_good_rows"],
                    "bad_recall": seq_metrics["bad_recall"],
                    "good_FPR": seq_metrics["good_FPR"],
                    "balanced_accuracy": seq_metrics["balanced_accuracy"],
                }
            )
    rank_rows.sort(
        key=lambda row: (
            max(float(row["abs_corr_L3"]), float(row["abs_corr_rolling"])),
            float(row["same_count_random_margin"]),
            float(row["balanced_accuracy"]),
        ),
        reverse=True,
    )
    return rank_rows, split_rows


def write_report(path: Path, summary: dict[str, Any], top_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Targeted Trace Summary",
        "",
        f"- targeted_trace_parity_pass: `{summary['targeted_trace_parity_pass']}`",
        f"- targeted_stage1_discovery_pass: `{summary['targeted_stage1_discovery_pass']}`",
        f"- parity_rows: `{summary['parity_rows']}`",
        f"- gca_context_topk_rows: `{summary['gca_context_topk_rows']}`",
        f"- kv_cache_provenance_rows: `{summary['kv_cache_provenance_rows']}`",
        f"- context_role_resolved_ratio_min: `{summary['context_role_resolved_ratio_min']}`",
        f"- targeted_memory_operation_rows: `{summary['targeted_memory_operation_rows']}`",
        f"- max_abs_corr_L3: `{summary['max_abs_corr_L3']}`",
        f"- max_same_count_random_margin: `{summary['max_same_count_random_margin']}`",
        "",
        "Top targeted levers:",
        "",
    ]
    for row in top_rows[:8]:
        lines.append(
            f"- {row['lever_id']}: abs_corr_L3={row['abs_corr_L3']}, "
            f"bad_recall={row['bad_recall']}, good_FPR={row['good_FPR']}, "
            f"same_count_random_margin={row['same_count_random_margin']}, "
            f"target_coverage={row['target_coverage']}"
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "These rows are no-action targeted traces over selected high-L3 and safe-good windows. "
            "A pass here is still a discovery pass, not a runtime action or full KITTI ATE claim.",
        ]
    )
    write_text(path, "\n".join(lines) + "\n")


def build() -> dict[str, Any]:
    targets = target_rows_by_dataset()
    all_trace_rows: list[dict[str, Any]] = []
    parity_rows: list[dict[str, Any]] = []
    for dataset, target in sorted(targets.items()):
        trace_rows = load_jsonl(Path(target["trace_file"]))
        for row in trace_rows:
            row.setdefault("dataset", dataset)
            row.setdefault("seq", target["seq"])
            row.setdefault("method", TRACE_METHOD)
            row["target_id"] = target["target_id"]
            row["target_kind"] = target["target_kind"]
        all_trace_rows.extend(trace_rows)
        parity_rows.append(compare_case(target, trace_rows))

    gca_rows = [row for row in all_trace_rows if row.get("row_type") == "gca_context_topk"]
    kv_rows = [row for row in all_trace_rows if row.get("row_type") == "kv_cache_provenance"]
    context_rows = aggregate_context(all_trace_rows)
    semantic_key_rows = annotate_trace_rows(targets, all_trace_rows) if all(row["parity_pass"] == "true" for row in parity_rows) else []
    head_rows = build_head_features(semantic_key_rows)
    frame_rows = build_frame_rows(targets, head_rows) if head_rows else []
    operation_rows = build_operation_rows(head_rows, frame_rows) if frame_rows else []
    rank_rows, split_rows = rank_levers(operation_rows) if operation_rows else ([], [])

    write_csv(STAGE1_TRACE / "no_action_parity_rows.csv", parity_rows)
    write_csv(STAGE1_TRACE / "targeted_gca_context_trace_rows.csv", gca_rows)
    write_csv(STAGE1_TRACE / "targeted_kv_cache_provenance_rows.csv", kv_rows)
    write_csv(STAGE1_TRACE / "targeted_context_role_token_rows.csv", context_rows)
    write_csv(STAGE1_TRACE / "targeted_trace_semantic_key_rows.csv", semantic_key_rows)
    write_csv(STAGE1_TRACE / "targeted_headlocal_frame_head_features.csv", head_rows)
    write_csv(STAGE1_TRACE / "targeted_frame_semantic_geometry_rows.csv", frame_rows)
    write_csv(STAGE1_TRACE / "targeted_memory_operation_rows.csv", operation_rows)
    write_csv(STAGE1_TRACE / "targeted_memory_lever_rank.csv", rank_rows)
    write_csv(STAGE1_TRACE / "targeted_memory_lever_sequence_split.csv", split_rows)

    failed = [row for row in parity_rows if row["parity_pass"] != "true"]
    max_abs_corr_l3 = max((float(row["abs_corr_L3"]) for row in rank_rows), default=0.0)
    max_abs_corr_rolling = max((float(row["abs_corr_rolling"]) for row in rank_rows), default=0.0)
    max_random_margin = max((float(row["same_count_random_margin"]) for row in rank_rows), default=0.0)
    levers_seq_ge2 = sum(1 for row in rank_rows if int(row["sequence_coverage"]) >= 2)
    operation_types_present = sorted({str(row["operation_type"]) for row in operation_rows})
    expected_operation_types = ["readout", "update", "retention", "initialization", "budget_eviction"]
    missing_operation_types = [item for item in expected_operation_types if item not in operation_types_present]
    targeted_stage1_pass = (
        not failed
        and levers_seq_ge2 >= 3
        and max(max_abs_corr_l3, max_abs_corr_rolling) >= 0.45
        and max_random_margin >= 0.05
    )
    summary = {
        "schema": "acl2_v106r_stage1_targeted_trace_summary_v1",
        "targeted_trace_parity_pass": not failed,
        "targeted_stage1_discovery_pass": targeted_stage1_pass,
        "parity_rows": len(parity_rows),
        "failed_parity_rows": len(failed),
        "gca_context_topk_rows": len(gca_rows),
        "kv_cache_provenance_rows": len(kv_rows),
        "trace_error_rows": sum(1 for row in all_trace_rows if row.get("row_type") == "trace_error"),
        "context_role_resolved_ratio_min": min((float(row["context_role_resolved_ratio"]) for row in parity_rows), default=0.0),
        "targeted_semantic_key_rows": len(semantic_key_rows),
        "targeted_headlocal_feature_rows": len(head_rows),
        "targeted_frame_rows": len(frame_rows),
        "targeted_memory_operation_rows": len(operation_rows),
        "targeted_lever_count": len(rank_rows),
        "levers_sequence_coverage_ge2": levers_seq_ge2,
        "operation_types_present": operation_types_present,
        "missing_operation_types": missing_operation_types,
        "max_abs_corr_L3": max_abs_corr_l3,
        "max_abs_corr_rolling": max_abs_corr_rolling,
        "max_same_count_random_margin": max_random_margin,
        "outputs": {
            "no_action_parity_rows": rel(STAGE1_TRACE / "no_action_parity_rows.csv"),
            "targeted_headlocal_frame_head_features": rel(STAGE1_TRACE / "targeted_headlocal_frame_head_features.csv"),
            "targeted_memory_operation_rows": rel(STAGE1_TRACE / "targeted_memory_operation_rows.csv"),
            "targeted_memory_lever_rank": rel(STAGE1_TRACE / "targeted_memory_lever_rank.csv"),
            "targeted_memory_lever_report": rel(STAGE1_TRACE / "targeted_memory_lever_report.md"),
        },
    }
    write_text(
        STAGE1_TRACE / "targeted_trace_summary.json",
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    write_report(STAGE1_TRACE / "targeted_memory_lever_report.md", summary, rank_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":
    build()
