#!/usr/bin/env python3
"""Build head-local trace diagnostics and policy sweep for ACL2 v105-TF Stage 4."""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

from build_v105tf_lingbot_stage3_oracle import (  # noqa: E402
    SPECIAL_TOKENS,
    SemanticFrameLoader,
    read_sampling_frames,
    semantic_role,
)


RESULT_ROOT = ROOT / "results/acl2_v105tf_dualtrack_lingbot_loger_evidence_eligibility"
STAGE3 = RESULT_ROOT / "stage3_lingbot_oracle"
STAGE4_HEAD = RESULT_ROOT / "stage4_lingbot_headlocal_trace"
RAW_TRACE = STAGE4_HEAD / "raw_trace"
SEQUENCES = ["00", "02"]
FEATURES = [
    "scale_reference_context_attention_frac",
    "local_window_context_attention_frac",
    "current_or_latest_frame_attention_frac",
    "semantic_scale_reference_attention_frac",
    "semantic_local_registration_attention_frac",
    "semantic_context_only_attention_frac",
    "semantic_reject_unreliable_attention_frac",
    "scale_context_reject_attention_frac",
    "scale_context_structure_attention_frac",
    "local_context_reject_attention_frac",
    "head_trace_topk_attention_sum",
    "head_trace_topk_rows",
]


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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def as_float(row: dict[str, Any], key: str) -> float:
    raw = row.get(key, 0.0)
    if raw in ("", None):
        return 0.0
    return float(raw)


def as_int(row: dict[str, Any], key: str) -> int:
    return int(float(row.get(key, 0) or 0))


def trace_path(seq: str) -> Path:
    dataset = f"kitti_v105_seq{seq}_trace32"
    method = f"lingbot_map_stage4_headlocal_trace_seq{seq}"
    return RAW_TRACE / f"{dataset}_{seq}_{method}.jsonl"


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


def load_stage3_frame_rows() -> dict[tuple[str, int], dict[str, Any]]:
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    for row in read_csv(STAGE3 / "frame_semantic_geometry_rows.csv"):
        seq = f"{as_int(row, 'seq'):02d}"
        pos = as_int(row, "sample_position")
        item = dict(row)
        item["seq"] = seq
        item["sample_position"] = pos
        item["bad_label"] = parse_bool(row.get("bad_label"))
        item["good_label"] = parse_bool(row.get("good_label"))
        rows[(seq, pos)] = item
    return rows


def annotate_trace_rows(seq: str, loader: SemanticFrameLoader, sample_frames: list[int]) -> list[dict[str, Any]]:
    rows = load_jsonl(trace_path(seq))
    last_invocation: dict[int, dict[str, Any]] = {}
    semantic_cache: dict[tuple[int, int], dict[str, Any]] = {}
    frame_cache: dict[int, dict[str, Any]] = {}
    annotated: list[dict[str, Any]] = []
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
                "schema": "acl2_v105tf_lingbot_stage4_headlocal_trace_semantic_key_row_v1",
                "seq": seq,
                "dataset": row.get("dataset", ""),
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
                "attention_weight": float(row.get("attention_weight", 0.0)),
                **sem,
            }
        )
    return annotated


def build_head_features(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if not row.get("frame_eval_allowed"):
            continue
        current_pos = row.get("current_sample_position", "")
        if current_pos == "":
            continue
        grouped[(str(row["seq"]), int(current_pos), int(row["head_idx"]))].append(row)

    out: list[dict[str, Any]] = []
    for (seq, pos, head_idx), group in sorted(grouped.items()):
        total = sum(float(row["attention_weight"]) for row in group)
        total = max(total, 1e-12)
        by_context = defaultdict(float)
        by_role = defaultdict(float)
        by_context_role = defaultdict(float)
        for row in group:
            weight = float(row["attention_weight"])
            by_context[str(row.get("key_context_role", ""))] += weight
            by_role[str(row.get("key_semantic_role", ""))] += weight
            by_context_role[(str(row.get("key_context_role", "")), str(row.get("key_semantic_role", "")))] += weight
        out.append(
            {
                "schema": "acl2_v105tf_lingbot_stage4_headlocal_frame_head_feature_v1",
                "seq": seq,
                "sample_position": pos,
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
                "scale_context_reject_attention_frac": by_context_role[("scale_reference_context", "REJECT_UNRELIABLE")] / max(by_context["scale_reference_context"], 1e-12),
                "scale_context_structure_attention_frac": by_context_role[("scale_reference_context", "SCALE_REFERENCE_EVIDENCE")] / max(by_context["scale_reference_context"], 1e-12),
                "local_context_reject_attention_frac": by_context_role[("local_window_context", "REJECT_UNRELIABLE")] / max(by_context["local_window_context"], 1e-12),
            }
        )
    return out


def thresholds(values: list[float]) -> list[float]:
    uniq = sorted({round(v, 12) for v in values if math.isfinite(v)})
    if len(uniq) > 60:
        qs = [5, 10, 15, 20, 25, 33, 40, 50, 60, 67, 75, 80, 85, 90, 95]
        vals = np.asarray(values, dtype=np.float64)
        uniq = sorted({round(float(np.percentile(vals, q)), 12) for q in qs})
    return uniq


def metric_for_selection(
    policy: str,
    selected: set[tuple[str, int]],
    frame_rows: dict[tuple[str, int], dict[str, Any]],
    selected_heads: dict[tuple[str, int], list[int]],
) -> dict[str, Any]:
    all_keys = sorted(frame_rows)
    bad_keys = [key for key in all_keys if bool(frame_rows[key]["bad_label"])]
    good_keys = [key for key in all_keys if bool(frame_rows[key]["good_label"])]
    selected_bad = [key for key in bad_keys if key in selected]
    selected_good = [key for key in good_keys if key in selected]
    neutral = [key for key in selected if key not in selected_bad and key not in selected_good]
    bad_recall = len(selected_bad) / max(len(bad_keys), 1)
    good_fpr = len(selected_good) / max(len(good_keys), 1)
    precision_vs_good = len(selected_bad) / max(len(selected_bad) + len(selected_good), 1)
    bad_seq_coverage = len({seq for seq, _ in selected_bad})
    deployable = len(selected_bad) >= 4 and bad_seq_coverage >= 2 and len(selected_good) <= 2
    strict_deployable = len(selected_bad) >= 4 and bad_seq_coverage >= 2 and len(selected_good) <= 1
    return {
        "schema": "acl2_v105tf_lingbot_stage4_headlocal_policy_metric_v1",
        "policy": policy,
        "selected_count": len(selected),
        "bad_selected": len(selected_bad),
        "good_selected": len(selected_good),
        "neutral_selected": len(neutral),
        "bad_total": len(bad_keys),
        "good_total": len(good_keys),
        "bad_recall": bad_recall,
        "good_FPR": good_fpr,
        "precision_vs_good": precision_vs_good,
        "bad_seq_coverage": bad_seq_coverage,
        "selected_positions": ";".join(f"{seq}:{pos}" for seq, pos in sorted(selected)),
        "bad_positions": ";".join(f"{seq}:{pos}" for seq, pos in selected_bad),
        "good_positions": ";".join(f"{seq}:{pos}" for seq, pos in selected_good),
        "selected_heads": ";".join(
            f"{seq}:{pos}:h{','.join(str(h) for h in sorted(set(selected_heads.get((seq, pos), []))))}"
            for seq, pos in sorted(selected)
        ),
        "deployable_candidate": deployable,
        "strict_deployable_candidate": strict_deployable,
    }


def sweep_policies(features: list[dict[str, Any]], frame_rows: dict[tuple[str, int], dict[str, Any]]) -> list[dict[str, Any]]:
    by_frame: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in features:
        by_frame[(str(row["seq"]), int(row["sample_position"]))].append(row)

    metrics: list[dict[str, Any]] = []
    for feature in FEATURES:
        vals = [as_float(row, feature) for row in features]
        for thr in thresholds(vals):
            for op in (">=", "<="):
                selected: set[tuple[str, int]] = set()
                heads: dict[tuple[str, int], list[int]] = defaultdict(list)
                for key, head_rows in by_frame.items():
                    for row in head_rows:
                        value = as_float(row, feature)
                        hit = value >= thr if op == ">=" else value <= thr
                        if hit:
                            selected.add(key)
                            heads[key].append(as_int(row, "head_idx"))
                if not selected:
                    continue
                policy = f"ANY_HEAD_{feature}_{'ge' if op == '>=' else 'le'}_{thr:.6g}"
                metrics.append(metric_for_selection(policy, selected, frame_rows, heads))

    # Two-clause safety form: high local-window evidence with a head-local reject/context cue.
    for reject_thr in thresholds([as_float(row, "semantic_reject_unreliable_attention_frac") for row in features]):
        for local_thr in thresholds([as_float(row, "local_window_context_attention_frac") for row in features]):
            selected = set()
            heads: dict[tuple[str, int], list[int]] = defaultdict(list)
            for row in features:
                key = (str(row["seq"]), int(row["sample_position"]))
                if (
                    as_float(row, "semantic_reject_unreliable_attention_frac") >= reject_thr
                    and as_float(row, "local_window_context_attention_frac") >= local_thr
                ):
                    selected.add(key)
                    heads[key].append(as_int(row, "head_idx"))
            if not selected:
                continue
            policy = (
                "ANY_HEAD_semantic_reject_unreliable_attention_frac"
                f"_ge_{reject_thr:.6g}_AND_local_window_context_attention_frac_ge_{local_thr:.6g}"
            )
            metrics.append(metric_for_selection(policy, selected, frame_rows, heads))

    metrics.sort(
        key=lambda row: (
            bool(row["strict_deployable_candidate"]),
            bool(row["deployable_candidate"]),
            int(row["bad_selected"]) - 3 * int(row["good_selected"]),
            float(row["precision_vs_good"]),
            float(row["bad_recall"]),
            -float(row["good_FPR"]),
            -int(row["selected_count"]),
        ),
        reverse=True,
    )
    return metrics


def selected_frame_rows(candidate: dict[str, Any] | None, frame_rows: dict[tuple[str, int], dict[str, Any]]) -> list[dict[str, Any]]:
    if not candidate:
        return []
    selected: list[dict[str, Any]] = []
    positions = str(candidate.get("selected_positions", ""))
    head_map: dict[tuple[str, int], str] = {}
    for part in str(candidate.get("selected_heads", "")).split(";"):
        if not part:
            continue
        seq, pos, heads = part.split(":", 2)
        head_map[(seq, int(pos))] = heads
    for part in positions.split(";"):
        if not part:
            continue
        seq, pos_s = part.split(":", 1)
        key = (seq, int(pos_s))
        source = frame_rows[key]
        selected.append(
            {
                "schema": "acl2_v105tf_lingbot_stage4_headlocal_selected_frame_v1",
                "policy": candidate["policy"],
                "seq": seq,
                "sample_position": key[1],
                "original_frame": as_int(source, "original_frame"),
                "bad_label": bool(source["bad_label"]),
                "good_label": bool(source["good_label"]),
                "sim3_residual_m": as_float(source, "sim3_residual_m"),
                "selected_heads": head_map.get(key, ""),
                "top_labels": source.get("top_labels", ""),
            }
        )
    return selected


def build() -> dict[str, Any]:
    STAGE4_HEAD.mkdir(parents=True, exist_ok=True)
    frame_rows = load_stage3_frame_rows()
    trace_semantic_rows: list[dict[str, Any]] = []
    for seq in SEQUENCES:
        sample_frames = read_sampling_frames(seq)
        loader = SemanticFrameLoader(seq)
        trace_semantic_rows.extend(annotate_trace_rows(seq, loader, sample_frames))

    head_features = build_head_features(trace_semantic_rows)
    metrics = sweep_policies(head_features, frame_rows)
    deployable = [row for row in metrics if row["deployable_candidate"]]
    strict = [row for row in metrics if row["strict_deployable_candidate"]]
    relaxed = [
        row for row in metrics
        if int(row["bad_selected"]) >= 4
        and int(row["bad_seq_coverage"]) >= 2
        and int(row["good_selected"]) <= 3
    ]
    candidate = strict[0] if strict else (deployable[0] if deployable else None)
    selected_rows = selected_frame_rows(candidate, frame_rows)
    relaxed_candidate = relaxed[0] if relaxed else None
    relaxed_selected_rows = selected_frame_rows(relaxed_candidate, frame_rows)

    write_csv(STAGE4_HEAD / "headlocal_trace_semantic_key_rows.csv", trace_semantic_rows)
    write_csv(STAGE4_HEAD / "headlocal_frame_head_features.csv", head_features)
    write_csv(STAGE4_HEAD / "headlocal_policy_metrics.csv", metrics[:500])
    write_csv(STAGE4_HEAD / "headlocal_selected_rows.csv", selected_rows)
    write_csv(STAGE4_HEAD / "headlocal_relaxed_selected_rows.csv", relaxed_selected_rows)

    summary = {
        "schema": "acl2_v105tf_lingbot_stage4_headlocal_policy_summary_v1",
        "trace_semantic_rows": len(trace_semantic_rows),
        "head_feature_rows": len(head_features),
        "candidate_policy_count": len(metrics),
        "deployable_candidate_count": len(deployable),
        "strict_deployable_candidate_count": len(strict),
        "selected_candidate": candidate,
        "relaxed_candidate_count": len(relaxed),
        "relaxed_candidate": relaxed_candidate,
        "outputs": {
            "trace_semantic_rows": str(STAGE4_HEAD / "headlocal_trace_semantic_key_rows.csv"),
            "head_features": str(STAGE4_HEAD / "headlocal_frame_head_features.csv"),
            "policy_metrics": str(STAGE4_HEAD / "headlocal_policy_metrics.csv"),
            "selected_rows": str(STAGE4_HEAD / "headlocal_selected_rows.csv"),
            "relaxed_selected_rows": str(STAGE4_HEAD / "headlocal_relaxed_selected_rows.csv"),
        },
    }
    (STAGE4_HEAD / "headlocal_policy_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":
    build()
