#!/usr/bin/env python3
"""Build ACL2 v105-TF LingBot Stage 3 semantic/geometry oracle diagnostics."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/acl2_v105tf_dualtrack_lingbot_loger_evidence_eligibility"
STAGE2 = RESULT_ROOT / "stage2_gca_trace"
STAGE3 = RESULT_ROOT / "stage3_lingbot_oracle"
WORKSPACE = STAGE2 / "workspace"
TRACE_DIR = STAGE2 / "raw_trace"
RAW_KITTI = ROOT / "data/kitti/dataset/sequences"
SEMANTIC_ROOT = ROOT / "results/kitti_preprocess"
TRACE_METHOD = "lingbot_map_stream_default_stage2_trace"
SEQUENCES = ["00", "02"]
TARGET_W = 504
TARGET_H = 280
PATCH_SIZE = 14
PATCH_W = TARGET_W // PATCH_SIZE
PATCH_H = TARGET_H // PATCH_SIZE
SPECIAL_TOKENS = 6


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


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_traj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    frames: list[int] = []
    mats: list[np.ndarray] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            vals = [float(x) for x in line.split()]
            if len(vals) != 13:
                raise ValueError(f"bad trajectory row in {path}: {line[:120]}")
            frames.append(int(vals[0]))
            mat = np.eye(4, dtype=np.float64)
            mat[:3, :4] = np.asarray(vals[1:], dtype=np.float64).reshape(3, 4)
            mats.append(mat)
    return np.asarray(frames, dtype=np.int64), np.stack(mats, axis=0)


def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    if len(src) < 3:
        return 1.0, np.eye(3), np.zeros(3)
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    x = src - mu_src
    y = dst - mu_dst
    var_src = float(np.mean(np.sum(x * x, axis=1)))
    if var_src <= 1e-12:
        return 1.0, np.eye(3), mu_dst - mu_src
    cov = (y.T @ x) / len(src)
    u, s, vt = np.linalg.svd(cov)
    d = np.ones(3)
    if np.linalg.det(u @ vt) < 0:
        d[-1] = -1
    r = u @ np.diag(d) @ vt
    scale = float(np.sum(s * d) / var_src)
    t = mu_dst - scale * (r @ mu_src)
    return scale, r, t


def sim3_residuals(gt_path: Path, pred_path: Path) -> dict[int, float]:
    gt_frames, gt = load_traj(gt_path)
    pred_frames, pred = load_traj(pred_path)
    if not np.array_equal(gt_frames, pred_frames):
        raise ValueError(f"frame mismatch: {gt_path} vs {pred_path}")
    gt_pos = gt[:, :3, 3]
    pred_pos = pred[:, :3, 3]
    scale, rot, trans = umeyama(pred_pos, gt_pos)
    aligned = scale * (pred_pos @ rot.T) + trans
    residual = np.linalg.norm(aligned - gt_pos, axis=1)
    return {int(frame): float(err) for frame, err in zip(gt_frames, residual)}


def semantic_role(label: str) -> str:
    label = str(label or "").strip().lower()
    scale_reference = {
        "building",
        "house",
        "wall",
        "handrail_or_fence",
        "fence",
        "pole",
        "pillar",
        "traffic sign",
        "billboard_or_bulletin_board",
        "other_construction",
    }
    local_registration = {"road", "ground", "path", "sidewalk", "stair"}
    reject = {
        "void",
        "sky",
        "grass",
        "tree",
        "flower",
        "other_plant",
        "vegetation",
        "mountain",
        "stone",
        "truck",
        "bicycle",
        "motorcycle",
        "person",
        "car",
        "wheeled_machine",
        "parasol_or_umbrella",
    }
    if label in scale_reference:
        return "SCALE_REFERENCE_EVIDENCE"
    if label in local_registration:
        return "LOCAL_REGISTRATION_EVIDENCE"
    if label in reject:
        return "REJECT_UNRELIABLE"
    return "CONTEXT_ONLY"


def image_transform(seq: str) -> dict[str, float]:
    image_path = RAW_KITTI / seq / "image_2/000000.png"
    with Image.open(image_path) as image:
        orig_w, orig_h = image.size
    scale = max(TARGET_W / orig_w, TARGET_H / orig_h)
    resized_w = int(round(orig_w * scale))
    resized_h = int(round(orig_h * scale))
    x0 = (resized_w - TARGET_W) // 2
    y0 = (resized_h - TARGET_H) // 2
    return {
        "orig_w": float(orig_w),
        "orig_h": float(orig_h),
        "scale": float(scale),
        "crop_x0": float(x0),
        "crop_y0": float(y0),
    }


def patch_sample_coords(seq: str, sem_h: int, sem_w: int) -> tuple[np.ndarray, np.ndarray]:
    transform = image_transform(seq)
    xs: list[int] = []
    ys: list[int] = []
    for row in range(PATCH_H):
        for col in range(PATCH_W):
            target_x = (col + 0.5) * PATCH_SIZE
            target_y = (row + 0.5) * PATCH_SIZE
            orig_x = (target_x + transform["crop_x0"]) / transform["scale"]
            orig_y = (target_y + transform["crop_y0"]) / transform["scale"]
            sem_x = int(round(orig_x * (sem_w - 1) / max(transform["orig_w"] - 1, 1)))
            sem_y = int(round(orig_y * (sem_h - 1) / max(transform["orig_h"] - 1, 1)))
            xs.append(int(np.clip(sem_x, 0, sem_w - 1)))
            ys.append(int(np.clip(sem_y, 0, sem_h - 1)))
    return np.asarray(ys, dtype=np.int64), np.asarray(xs, dtype=np.int64)


def read_sampling_frames(seq: str) -> list[int]:
    dataset = f"kitti_v105_seq{seq}_trace32"
    sampling_path = WORKSPACE / dataset / seq / "gt/sampling.json"
    payload = load_json(sampling_path)
    return [int(x) for x in payload["frames"]]


def chunk_index(seq: str) -> list[dict[str, Any]]:
    index_path = SEMANTIC_ROOT / seq / "stage_c_cache_semantic_chunks/cache_index.jsonl"
    chunks = load_jsonl(index_path)
    return sorted(chunks, key=lambda row: int(row["start_frame"]))


def find_chunk(chunks: list[dict[str, Any]], frame: int) -> dict[str, Any]:
    for chunk in chunks:
        if int(chunk["start_frame"]) <= frame < int(chunk["end_frame"]):
            return chunk
    raise KeyError(f"no semantic chunk for frame {frame}")


class SemanticFrameLoader:
    def __init__(self, seq: str):
        self.seq = seq
        self.chunks = chunk_index(seq)
        self.cache_name = ""
        self.cache: dict[str, Any] | None = None
        self.coords: tuple[np.ndarray, np.ndarray] | None = None

    def load_frame(self, frame: int) -> dict[str, Any]:
        chunk = find_chunk(self.chunks, frame)
        chunk_name = str(chunk["chunk"])
        if self.cache_name != chunk_name:
            path = SEMANTIC_ROOT / self.seq / "stage_c_cache_semantic_chunks" / chunk_name / "masklet.pt"
            self.cache = torch.load(path, map_location="cpu", weights_only=False)
            self.cache_name = chunk_name
            sem = self.cache["semantic_segmentation"]
            label_maps = sem["label_maps"]
            if isinstance(label_maps, torch.Tensor):
                sem_h, sem_w = int(label_maps.shape[1]), int(label_maps.shape[2])
            else:
                sem_h, sem_w = int(label_maps.shape[1]), int(label_maps.shape[2])
            self.coords = patch_sample_coords(self.seq, sem_h, sem_w)
        assert self.cache is not None and self.coords is not None
        sem = self.cache["semantic_segmentation"]
        local_idx = int(frame) - int(chunk["start_frame"])
        labels = sem["label_maps"][local_idx]
        conf = sem["confidence_maps"][local_idx]
        if isinstance(labels, torch.Tensor):
            labels_np = labels.detach().cpu().numpy()
        else:
            labels_np = np.asarray(labels)
        if isinstance(conf, torch.Tensor):
            conf_np = conf.detach().cpu().float().numpy()
        else:
            conf_np = np.asarray(conf, dtype=np.float32)
        ys, xs = self.coords
        patch_labels = labels_np[ys, xs].astype(np.int64)
        patch_conf = conf_np[ys, xs].astype(np.float32)
        label_names = list(sem.get("label_names", []))
        role_names = [semantic_role(label_names[int(x)] if int(x) < len(label_names) else "void") for x in patch_labels]
        return {
            "frame": int(frame),
            "chunk": chunk_name,
            "label_names": label_names,
            "patch_labels": patch_labels,
            "patch_confidence": patch_conf,
            "patch_roles": role_names,
            "semantic_source": sem.get("source", ""),
        }


def summarize_frame_semantics(seq: str, sample_pos: int, frame: int, loader: SemanticFrameLoader) -> dict[str, Any]:
    sem = loader.load_frame(frame)
    roles = sem["patch_roles"]
    role_counts = defaultdict(int)
    for role in roles:
        role_counts[role] += 1
    denom = float(len(roles)) if roles else 1.0
    label_counts = defaultdict(int)
    label_names = sem["label_names"]
    for label_id in sem["patch_labels"]:
        label = label_names[int(label_id)] if int(label_id) < len(label_names) else "void"
        label_counts[label] += 1
    top_labels = sorted(label_counts.items(), key=lambda item: (-item[1], item[0]))[:8]
    return {
        "seq": seq,
        "sample_position": sample_pos,
        "original_frame": frame,
        "semantic_chunk": sem["chunk"],
        "semantic_source": sem["semantic_source"],
        "patch_count": len(roles),
        "scale_reference_patch_frac": role_counts["SCALE_REFERENCE_EVIDENCE"] / denom,
        "local_registration_patch_frac": role_counts["LOCAL_REGISTRATION_EVIDENCE"] / denom,
        "context_only_patch_frac": role_counts["CONTEXT_ONLY"] / denom,
        "reject_unreliable_patch_frac": role_counts["REJECT_UNRELIABLE"] / denom,
        "semantic_confidence_mean": float(np.mean(sem["patch_confidence"])) if len(sem["patch_confidence"]) else 0.0,
        "semantic_confidence_p10": float(np.percentile(sem["patch_confidence"], 10)) if len(sem["patch_confidence"]) else 0.0,
        "top_labels": ";".join(f"{label}:{count}" for label, count in top_labels),
    }


def key_semantics(seq: str, sample_position: int, token_offset: int, loader: SemanticFrameLoader, sample_frames: list[int]) -> dict[str, Any]:
    if not (0 <= sample_position < len(sample_frames)):
        return {
            "key_original_frame": "",
            "key_semantic_label": "",
            "key_semantic_role": "",
            "key_semantic_confidence": "",
        }
    if token_offset < SPECIAL_TOKENS:
        return {
            "key_original_frame": sample_frames[sample_position],
            "key_semantic_label": "special_token",
            "key_semantic_role": "TRAJECTORY_MEMORY_EVIDENCE",
            "key_semantic_confidence": "",
        }
    patch_index = int(token_offset) - SPECIAL_TOKENS
    if not (0 <= patch_index < PATCH_H * PATCH_W):
        return {
            "key_original_frame": sample_frames[sample_position],
            "key_semantic_label": "token_out_of_patch_grid",
            "key_semantic_role": "CONTEXT_ONLY",
            "key_semantic_confidence": "",
        }
    sem = loader.load_frame(sample_frames[sample_position])
    label_id = int(sem["patch_labels"][patch_index])
    label_names = sem["label_names"]
    label = label_names[label_id] if label_id < len(label_names) else "void"
    return {
        "key_original_frame": sample_frames[sample_position],
        "key_semantic_label": label,
        "key_semantic_role": semantic_role(label),
        "key_semantic_confidence": float(sem["patch_confidence"][patch_index]),
    }


def annotate_trace_rows(seq: str, loader: SemanticFrameLoader, sample_frames: list[int]) -> list[dict[str, Any]]:
    trace_path = TRACE_DIR / f"kitti_v105_seq{seq}_trace32_{seq}_{TRACE_METHOD}.jsonl"
    rows = load_jsonl(trace_path)
    last_invocation: dict[int, dict[str, Any]] = {}
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
        sem = key_semantics(seq, key_frame_offset, key_token_offset, loader, sample_frames)
        out = {
            "schema": "acl2_v105tf_lingbot_stage3_trace_semantic_key_row_v1",
            "seq": seq,
            "dataset": row.get("dataset", ""),
            "global_idx": global_idx,
            "current_sample_position": current_pos if allowed else "",
            "current_original_frame": sample_frames[current_pos] if allowed and 0 <= current_pos < len(sample_frames) else "",
            "frame_eval_allowed": allowed,
            "key_sample_position": key_frame_offset,
            "key_token_offset": key_token_offset,
            "key_context_role": row.get("key_context_role", ""),
            "key_token_role": row.get("key_token_role", ""),
            "query_token_role": row.get("query_token_role", ""),
            "topk_rank": row.get("topk_rank", ""),
            "attention_weight": float(row.get("attention_weight", 0.0)),
            **sem,
        }
        annotated.append(out)
    return annotated


def frame_trace_features(rows: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if not row.get("frame_eval_allowed"):
            continue
        current_pos = row.get("current_sample_position", "")
        if current_pos == "":
            continue
        grouped[(str(row["seq"]), int(current_pos))].append(row)
    features: dict[tuple[str, int], dict[str, Any]] = {}
    for key, group in grouped.items():
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
        seq, pos = key
        features[key] = {
            "seq": seq,
            "sample_position": pos,
            "trace_topk_rows": len(group),
            "trace_topk_attention_sum": total,
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
    return features


def label_error_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_seq: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_seq[str(row["seq"])].append(row)
    out: list[dict[str, Any]] = []
    for seq, seq_rows in by_seq.items():
        errs = np.asarray([float(row["sim3_residual_m"]) for row in seq_rows], dtype=np.float64)
        q50 = float(np.percentile(errs, 50))
        q75 = float(np.percentile(errs, 75))
        for row in seq_rows:
            err = float(row["sim3_residual_m"])
            row = dict(row)
            row["seq_error_p50"] = q50
            row["seq_error_p75"] = q75
            row["bad_label"] = err >= q75
            row["good_label"] = err <= q50
            out.append(row)
    return out


def base_policy(row: dict[str, Any]) -> bool:
    return float(row["scale_reference_context_attention_frac"]) < 0.42


def semantic_only_policy(row: dict[str, Any]) -> bool:
    return (
        float(row["reject_unreliable_patch_frac"]) >= 0.45
        or float(row["scale_reference_patch_frac"]) <= 0.10
        or float(row["semantic_confidence_p10"]) <= 0.20
    )


def semantic_geometry_policy(row: dict[str, Any]) -> bool:
    return (
        float(row["scale_context_reject_attention_frac"]) >= 0.35
        or (
            float(row["reject_unreliable_patch_frac"]) >= 0.45
            and float(row["scale_context_structure_attention_frac"]) <= 0.25
        )
        or (
            float(row["semantic_reject_unreliable_attention_frac"]) >= 0.35
            and float(row["scale_reference_context_attention_frac"]) >= 0.35
        )
    )


def stable_semantic_policy(row: dict[str, Any]) -> bool:
    return (
        float(row["scale_reference_patch_frac"]) <= 0.12
        and float(row["semantic_scale_reference_attention_frac"]) <= 0.18
    )


def metrics_for_predictions(name: str, rows: list[dict[str, Any]], selected: list[bool]) -> dict[str, Any]:
    bad = [i for i, row in enumerate(rows) if row["bad_label"]]
    good = [i for i, row in enumerate(rows) if row["good_label"]]
    selected_bad = [i for i in bad if selected[i]]
    selected_good = [i for i in good if selected[i]]
    bad_recall = len(selected_bad) / max(len(bad), 1)
    good_fpr = len(selected_good) / max(len(good), 1)
    coverage = len({str(rows[i]["seq"]) for i in selected_bad})
    return {
        "schema": "acl2_v105tf_lingbot_stage3_oracle_policy_metric_v1",
        "policy": name,
        "rows": len(rows),
        "bad_rows": len(bad),
        "good_rows": len(good),
        "selected_rows": int(sum(1 for flag in selected if flag)),
        "selected_bad_rows": len(selected_bad),
        "selected_good_rows": len(selected_good),
        "bad_recall": bad_recall,
        "good_FPR": good_fpr,
        "balanced_accuracy": 0.5 * (bad_recall + (1.0 - good_fpr)),
        "selected_positive_sequence_coverage": coverage,
        "safe_good_harm_proxy": good_fpr,
    }


def shuffled_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = [dict(row) for row in rows]
    by_seq: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        by_seq[str(row["seq"])].append(idx)
    semantic_fields = [
        "scale_reference_patch_frac",
        "local_registration_patch_frac",
        "context_only_patch_frac",
        "reject_unreliable_patch_frac",
        "semantic_confidence_mean",
        "semantic_confidence_p10",
        "semantic_scale_reference_attention_frac",
        "semantic_local_registration_attention_frac",
        "semantic_context_only_attention_frac",
        "semantic_reject_unreliable_attention_frac",
        "scale_context_reject_attention_frac",
        "scale_context_structure_attention_frac",
        "local_context_reject_attention_frac",
    ]
    for indices in by_seq.values():
        shifted = indices[1:] + indices[:1]
        for dst, src in zip(indices, shifted):
            for field in semantic_fields:
                out[dst][field] = rows[src][field]
    return out


def rotated_context_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = [dict(row) for row in rows]
    for row in out:
        row["scale_reference_context_attention_frac"], row["local_window_context_attention_frac"] = (
            row["local_window_context_attention_frac"],
            row["scale_reference_context_attention_frac"],
        )
        row["scale_context_reject_attention_frac"], row["local_context_reject_attention_frac"] = (
            row["local_context_reject_attention_frac"],
            row["scale_context_reject_attention_frac"],
        )
        row["scale_context_structure_attention_frac"] = 1.0 - float(row["scale_context_reject_attention_frac"])
    return out


def random_same_count_metric(rows: list[dict[str, Any]], selected_count: int, trials: int = 64) -> dict[str, float]:
    rng = np.random.default_rng(105)
    recalls = []
    fprs = []
    n = len(rows)
    for _ in range(trials):
        picks = set(int(x) for x in rng.choice(n, size=min(selected_count, n), replace=False))
        metric = metrics_for_predictions("same_count_random", rows, [idx in picks for idx in range(n)])
        recalls.append(float(metric["bad_recall"]))
        fprs.append(float(metric["good_FPR"]))
    return {
        "same_count_random_bad_recall_mean": float(np.mean(recalls)) if recalls else 0.0,
        "same_count_random_bad_recall_p95": float(np.percentile(recalls, 95)) if recalls else 0.0,
        "same_count_random_good_FPR_mean": float(np.mean(fprs)) if fprs else 0.0,
    }


def evaluate_policy(name: str, rows: list[dict[str, Any]], fn: Callable[[dict[str, Any]], bool], geometry_baseline_ba: float) -> dict[str, Any]:
    selected = [fn(row) for row in rows]
    metric = metrics_for_predictions(name, rows, selected)
    rand = random_same_count_metric(rows, int(metric["selected_rows"]))
    metric.update(rand)
    shuffled = metrics_for_predictions(name + "_semantic_shuffle", shuffled_rows(rows), [fn(row) for row in shuffled_rows(rows)])
    rotated = metrics_for_predictions(name + "_context_role_rotation", rotated_context_rows(rows), [fn(row) for row in rotated_context_rows(rows)])
    metric["same_count_random_margin"] = float(metric["bad_recall"]) - float(metric["same_count_random_bad_recall_mean"])
    metric["semantic_shuffle_bad_recall"] = shuffled["bad_recall"]
    metric["semantic_shuffle_margin"] = float(metric["bad_recall"]) - float(shuffled["bad_recall"])
    metric["context_role_rotation_bad_recall"] = rotated["bad_recall"]
    metric["context_role_rotation_margin"] = float(metric["bad_recall"]) - float(rotated["bad_recall"])
    metric["semantic_increment_over_geometry_only"] = float(metric["balanced_accuracy"]) - geometry_baseline_ba
    metric["stage3_oracle_pass"] = (
        float(metric["bad_recall"]) >= 0.65
        and float(metric["good_FPR"]) <= 0.25
        and int(metric["selected_positive_sequence_coverage"]) >= 2
        and float(metric["same_count_random_margin"]) >= 0.05
        and float(metric["semantic_shuffle_margin"]) >= 0.05
    )
    return metric


def build() -> dict[str, Any]:
    STAGE3.mkdir(parents=True, exist_ok=True)
    source_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    trace_semantic_rows: list[dict[str, Any]] = []

    for seq in SEQUENCES:
        sample_frames = read_sampling_frames(seq)
        loader = SemanticFrameLoader(seq)
        semantic_pt = SEMANTIC_ROOT / seq / "sparse_masklets_with_semantic.pt"
        metrics_path = SEMANTIC_ROOT / seq / "sparse_masklets_with_semantic.metrics.json"
        metrics = load_json(metrics_path)
        source_rows.append(
            {
                "schema": "acl2_v105tf_lingbot_stage3_semantic_source_row_v1",
                "seq": seq,
                "semantic_pt": semantic_pt.relative_to(ROOT).as_posix(),
                "semantic_pt_size_bytes": semantic_pt.stat().st_size,
                "semantic_metrics_json": metrics_path.relative_to(ROOT).as_posix(),
                "semantic_pt_sha256": sha256_file(semantic_pt),
                "semantic_format": metrics.get("semantic_format", ""),
                "semantic_shape": metrics.get("semantic_shape", ""),
                "semantic_dtype": metrics.get("semantic_dtype", ""),
                "num_labels": metrics.get("num_labels", ""),
                "label_names": ";".join(metrics.get("label_names", [])),
                "has_confidence_maps": metrics.get("has_confidence_maps", False),
                "sampled_frames_required": len(sample_frames),
                "sampled_frames_covered": sum(1 for frame in sample_frames if find_chunk(loader.chunks, frame)),
                "source_scope": "pseudo_semantic_cache_from_kitti_preprocess_not_GT",
            }
        )
        for sample_pos, frame in enumerate(sample_frames):
            frame_rows.append(summarize_frame_semantics(seq, sample_pos, frame, loader))
        trace_semantic_rows.extend(annotate_trace_rows(seq, loader, sample_frames))

    trace_features = frame_trace_features(trace_semantic_rows)
    enriched_rows: list[dict[str, Any]] = []
    for row in frame_rows:
        key = (str(row["seq"]), int(row["sample_position"]))
        if key not in trace_features:
            continue
        seq = str(row["seq"])
        dataset = f"kitti_v105_seq{seq}_trace32"
        residuals = sim3_residuals(
            WORKSPACE / dataset / seq / "gt/traj.txt",
            WORKSPACE / dataset / seq / TRACE_METHOD / "traj.txt",
        )
        sample_pos = int(row["sample_position"])
        out = dict(row)
        out.update(trace_features[key])
        out["sim3_residual_m"] = residuals[sample_pos]
        enriched_rows.append(out)

    labelled_rows = label_error_rows(enriched_rows)
    geometry_metric = evaluate_policy("geometry_only_low_scale_reference_attention", labelled_rows, base_policy, 0.0)
    geometry_ba = float(geometry_metric["balanced_accuracy"])
    policy_rows = [
        {**geometry_metric, "semantic_increment_over_geometry_only": 0.0},
        evaluate_policy("semantic_only_unreliable_patch_prior", labelled_rows, semantic_only_policy, geometry_ba),
        evaluate_policy("semantic_geometry_scale_ref_unreliable", labelled_rows, semantic_geometry_policy, geometry_ba),
        evaluate_policy("semantic_geometry_low_structure_support", labelled_rows, stable_semantic_policy, geometry_ba),
    ]

    for row in policy_rows:
        row["stage3_oracle_pass"] = "true" if row["stage3_oracle_pass"] else "false"

    pass_rows = [row for row in policy_rows if row["stage3_oracle_pass"] == "true"]
    summary = {
        "schema": "acl2_v105tf_lingbot_stage3_oracle_summary_v1",
        "semantic_source_scope": "pseudo_semantic_cache_from_kitti_preprocess_not_GT",
        "trace_frame_alignment": "inferred_from_kv_cached_frames;scale_batch_rows_excluded",
        "frame_rows": len(labelled_rows),
        "trace_semantic_key_rows": len(trace_semantic_rows),
        "policy_rows": len(policy_rows),
        "stage3_lingbot_oracle_pass": bool(pass_rows),
        "stage4_action_allowed": bool(pass_rows),
        "passing_policies": [row["policy"] for row in pass_rows],
        "best_policy_by_balanced_accuracy": max(policy_rows, key=lambda row: float(row["balanced_accuracy"]))["policy"],
        "best_balanced_accuracy": max(float(row["balanced_accuracy"]) for row in policy_rows),
    }

    write_csv(STAGE3 / "semantic_source_rows.csv", source_rows)
    write_csv(STAGE3 / "trace_semantic_key_rows.csv", trace_semantic_rows)
    write_csv(STAGE3 / "frame_semantic_geometry_rows.csv", labelled_rows)
    write_csv(STAGE3 / "oracle_policy_metrics.csv", policy_rows)
    write_text(STAGE3 / "stage3_summary.json", json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    if pass_rows:
        failure_path = STAGE3 / "semantic_increment_failure.md"
        if failure_path.exists():
            failure_path.unlink()
    else:
        lines = [
            "# Stage3 Semantic Increment Failure",
            "",
            "No Stage3 oracle policy passed all required controls on the LingBot trace32 diagnostic universe.",
            "",
            "Important boundaries:",
            "- Semantic source is an existing pseudo-semantic cache from `results/kitti_preprocess`, not GT.",
            "- Frame alignment is inferred from KV cached-frame order; scale-batch trace rows are excluded.",
            "- This is an oracle diagnostic only; no LingBot routing action was run.",
            "",
            "Policy metrics are in `oracle_policy_metrics.csv`.",
            "",
        ]
        write_text(STAGE3 / "semantic_increment_failure.md", "\n".join(lines))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":
    build()
