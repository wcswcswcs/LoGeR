#!/usr/bin/env python3
"""Build ACL2 v108TF Stage2 full-sequence semantic cue bank.

This stage deliberately rebuilds semantic cues over the full KITTI 00/01/02/05
frame universe.  It may reuse the v107R projection helpers, but it must not
promote the earlier targeted 96F cue bank into full-sequence evidence.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import build_v107r_semantic_memory_decision_cue_operation_control as v107r  # noqa: E402


RESULT_ROOT = ROOT / "results/acl2_v108tf_lingbot_memory_operation_action_surface_search"
OUT = RESULT_ROOT / "stage2_semantic_cue_bank"
STAGE0 = RESULT_ROOT / "stage0_evidence_freeze"
V107TF_STAGE1 = ROOT / "results/acl2_v107tf_lingbot_cache_operation_observability_semantic_aware_update_retention/stage1_cache_operation_instrumentation"
OP_ROWS = V107TF_STAGE1 / "operation_trace_rows.csv"
SEQ_IDS = ("00", "01", "02", "05")
NEARBY_RADIUS = 4
EPS = 1.0e-9


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fnum(value: Any, default: float = float("nan")) -> float:
    try:
        if value == "":
            return default
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def inum(value: Any, default: int = -1) -> int:
    try:
        if value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def mean(values: list[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return sum(vals) / len(vals) if vals else float("nan")


def fmt(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    try:
        val = float(value)
    except Exception:
        return "" if value is None else str(value)
    if not math.isfinite(val):
        return ""
    return f"{val:.8g}"


def baseline_lengths() -> dict[str, int]:
    rows = read_csv(STAGE0 / "full_kitti_baseline_table.csv")
    lengths = {row.get("seq", ""): inum(row.get("frames", ""), -1) for row in rows}
    return {seq: lengths[seq] for seq in SEQ_IDS if lengths.get(seq, -1) > 0}


def target_ids_by_frame(op_rows: list[dict[str, str]]) -> dict[tuple[str, int], set[str]]:
    out: dict[tuple[str, int], set[str]] = defaultdict(set)
    for row in op_rows:
        seq = row.get("seq") or row.get("sequence_id", "")
        target_id = row.get("target_id", "")
        if not seq or not target_id:
            continue
        frame_id = inum(row.get("frame_id", ""), -1)
        if frame_id >= 0:
            out[(seq, frame_id)].add(target_id)
        trace_start = inum(row.get("trace_start_idx", ""), -1)
        span_start = inum(row.get("frame_span_start", ""), -1)
        span_end = inum(row.get("frame_span_end", ""), span_start)
        if trace_start >= 0 and span_start >= 0 and span_end >= span_start:
            for fid in range(trace_start + span_start, trace_start + span_end + 1):
                out[(seq, fid)].add(target_id)
    return out


def assign_frames_to_chunks(seq: str, length: int) -> tuple[dict[Path, list[int]], list[dict[str, Any]]]:
    chunks = v107r.build_chunk_index(seq)
    chunk_to_frames: dict[Path, list[int]] = defaultdict(list)
    missing: list[dict[str, Any]] = []
    for frame_id in range(length):
        chunk = v107r.find_chunk(chunks, frame_id)
        if chunk is None:
            missing.append({"seq_id": seq, "frame_id": frame_id, "reason": "no_stage_c_chunk_covering_frame"})
            continue
        _start, _end, path = chunk
        chunk_to_frames[path].append(frame_id)
    return chunk_to_frames, missing


def normalized_frame_fields(summary: dict[str, Any]) -> dict[str, float]:
    trust = max(float(summary.get("semantic_trust_mean", 0.0)), EPS)
    stable = float(summary.get("stable_structure_mass", 0.0)) / trust
    dynamic = float(summary.get("dynamic_transient_mass", 0.0)) / trust
    boundary = float(summary.get("semantic_boundary_mass", 0.0)) / trust
    weak = (
        float(summary.get("ground_or_road_weak_mass", 0.0))
        + float(summary.get("vegetation_weak_context_mass", 0.0))
    ) / trust
    road_ground = (
        float(summary.get("road_boundary_or_layout_mass", 0.0))
        + float(summary.get("ground_or_road_weak_mass", 0.0))
    ) / trust
    sky_lowobs = (
        float(summary.get("sky_or_lowobs_mass", 0.0))
        + float(summary.get("unknown_lowtrust_mass", 0.0))
    ) / trust
    purity = float(summary.get("semantic_patch_purity_mean", 0.0))
    return {
        "stable_structure_mass": stable,
        "dynamic_mass": dynamic,
        "boundary_mass": boundary,
        "weak_context_mass": weak,
        "road_ground_mass": road_ground,
        "sky_lowobs_mass": sky_lowobs,
        "semantic_trust_mean": float(summary.get("semantic_trust_mean", 0.0)),
        "semantic_purity_mean": purity,
        "semantic_boundary_risk": max(0.0, 1.0 - purity),
        "raw_stable_structure_trust_mass": float(summary.get("stable_structure_mass", 0.0)),
        "raw_road_boundary_or_layout_trust_mass": float(summary.get("road_boundary_or_layout_mass", 0.0)),
        "raw_ground_or_road_weak_trust_mass": float(summary.get("ground_or_road_weak_mass", 0.0)),
        "raw_dynamic_transient_trust_mass": float(summary.get("dynamic_transient_mass", 0.0)),
        "raw_vegetation_weak_context_trust_mass": float(summary.get("vegetation_weak_context_mass", 0.0)),
        "raw_sky_or_lowobs_trust_mass": float(summary.get("sky_or_lowobs_mass", 0.0)),
        "raw_semantic_boundary_trust_mass": float(summary.get("semantic_boundary_mass", 0.0)),
        "raw_unknown_lowtrust_trust_mass": float(summary.get("unknown_lowtrust_mass", 0.0)),
    }


def apply_q_ref(row: dict[str, Any]) -> None:
    risk = (
        fnum(row.get("dynamic_mass", 0.0))
        + fnum(row.get("boundary_mass", 0.0))
        + fnum(row.get("weak_context_mass", 0.0))
        + fnum(row.get("sky_lowobs_mass", 0.0))
    )
    stable = fnum(row.get("stable_structure_mass", 0.0))
    continuity = fnum(row.get("semantic_continuity_score", 0.0))
    boundary_risk = fnum(row.get("semantic_boundary_risk", 0.0))
    row["Q_ref_sem_balanced"] = stable + 0.5 * continuity - risk - 0.5 * boundary_risk
    row["Q_ref_sem_risk_strict"] = stable + 0.5 * continuity - 1.5 * risk - boundary_risk
    row["Q_ref_sem_stable_strict"] = 1.5 * stable + continuity - risk - 0.5 * boundary_risk


def visible_seed_set(payload: dict[str, Any], local_idx: int) -> set[str]:
    seed_ids = [str(seed) for seed in (payload.get("seed_global_track_idx", []) or [])]
    vmask = payload.get("V_mask")
    if vmask is None or not seed_ids:
        return set()
    if local_idx < 0 or local_idx >= int(vmask.shape[1]):
        return set()
    col = vmask[:, local_idx]
    out: set[str] = set()
    for idx in range(min(len(seed_ids), int(col.shape[0]))):
        try:
            visible = bool(col[idx].item())
        except Exception:
            visible = bool(col[idx])
        if visible and seed_ids[idx]:
            out.add(seed_ids[idx])
    return out


def operation_join_frames(row: dict[str, str]) -> list[int]:
    trace_start = inum(row.get("trace_start_idx", ""), -1)
    span_start = inum(row.get("frame_span_start", ""), -1)
    span_end = inum(row.get("frame_span_end", ""), span_start)
    if trace_start >= 0 and span_start >= 0 and span_end >= span_start:
        return list(range(trace_start + span_start, trace_start + span_end + 1))
    frame_id = inum(row.get("frame_id", ""), -1)
    return [frame_id] if frame_id >= 0 else []


def surface_ids_for_operation(row: dict[str, str]) -> str:
    op = row.get("operation_type", "")
    token_type = row.get("token_type", "")
    context_path = row.get("context_path", "")
    surfaces: set[str] = set()
    if op == "initialization":
        surfaces.add("A")
    if op in {"cache_append", "initialization"}:
        surfaces.add("B")
    if op in {"retention", "eviction", "budget_keep", "budget_drop"}:
        surfaces.add("C")
    if op == "trajectory_write" or row.get("is_trajectory_memory", "") == "True":
        surfaces.add("D")
    if context_path == "local_pose_reference_window":
        surfaces.add("E")
    if op == "special_token_update" or token_type != "image_patch":
        surfaces.add("F")
    return ";".join(sorted(surfaces))


def aggregate_frames(rows: list[dict[str, Any]]) -> dict[str, float]:
    cols = [
        "stable_structure_mass",
        "dynamic_mass",
        "boundary_mass",
        "weak_context_mass",
        "road_ground_mass",
        "sky_lowobs_mass",
        "semantic_trust_mean",
        "semantic_purity_mean",
        "semantic_boundary_risk",
        "semantic_continuity_score",
        "Q_ref_sem_balanced",
        "Q_ref_sem_risk_strict",
        "Q_ref_sem_stable_strict",
    ]
    return {f"{col}_mean": mean([fnum(row.get(col, "")) for row in rows]) for col in cols}


def build_operation_semantic_summary(op_rows: list[dict[str, str]], frame_rows: list[dict[str, Any]]) -> dict[str, Any]:
    frame_map = {(str(row["seq_id"]), int(row["frame_id"])): row for row in frame_rows}
    out_rows: list[dict[str, Any]] = []
    join_any = 0
    join_coverages: list[float] = []
    for idx, row in enumerate(op_rows):
        seq = row.get("seq") or row.get("sequence_id", "")
        join_frames = operation_join_frames(row)
        available = [frame_map[(seq, fid)] for fid in join_frames if (seq, fid) in frame_map]
        if not available:
            fid = inum(row.get("frame_id", ""), -1)
            if (seq, fid) in frame_map:
                available = [frame_map[(seq, fid)]]
        join_coverage = len(available) / len(join_frames) if join_frames else 0.0
        join_coverages.append(join_coverage)
        join_any += int(bool(available))
        semantic = aggregate_frames(available) if available else {}
        op = row.get("operation_type", "")
        out_rows.append(
            {
                "schema": "acl2_v108tf_operation_semantic_summary_row_v1",
                "operation_id": idx,
                "source_row_index": idx,
                "target_id": row.get("target_id", ""),
                "seq_id": seq,
                "target_kind": row.get("target_kind", ""),
                "current_frame": row.get("frame_id", ""),
                "semantic_join_frame_start": join_frames[0] if join_frames else "",
                "semantic_join_frame_end": join_frames[-1] if join_frames else "",
                "semantic_join_frame_count": len(join_frames),
                "semantic_join_available_frame_count": len(available),
                "semantic_join_coverage": join_coverage,
                "semantic_runtime_available": bool(available),
                "operation_type": op,
                "plan_operation_type": v107r.normalize_operation_type(op),
                "candidate_surface_ids": surface_ids_for_operation(row),
                "context_path": row.get("context_path", ""),
                "token_type": row.get("token_type", ""),
                "token_id": row.get("token_index", ""),
                "token_count": row.get("token_count", ""),
                "source_frame": row.get("source_frame", ""),
                "source_frame_age": row.get("source_age", ""),
                "keyframe_flag": row.get("is_keyframe", ""),
                "scale_frame_flag": row.get("is_scale_frame", ""),
                "trajectory_memory_flag": row.get("is_trajectory_memory", ""),
                "cache_keep_drop_status": "keep" if op == "budget_keep" else ("drop" if op in {"budget_drop", "eviction"} else ""),
                **semantic,
            }
        )
    fields = [
        "schema",
        "operation_id",
        "source_row_index",
        "target_id",
        "seq_id",
        "target_kind",
        "current_frame",
        "semantic_join_frame_start",
        "semantic_join_frame_end",
        "semantic_join_frame_count",
        "semantic_join_available_frame_count",
        "semantic_join_coverage",
        "semantic_runtime_available",
        "operation_type",
        "plan_operation_type",
        "candidate_surface_ids",
        "context_path",
        "token_type",
        "token_id",
        "token_count",
        "source_frame",
        "source_frame_age",
        "keyframe_flag",
        "scale_frame_flag",
        "trajectory_memory_flag",
        "cache_keep_drop_status",
        "stable_structure_mass_mean",
        "dynamic_mass_mean",
        "boundary_mass_mean",
        "weak_context_mass_mean",
        "road_ground_mass_mean",
        "sky_lowobs_mass_mean",
        "semantic_trust_mean_mean",
        "semantic_purity_mean_mean",
        "semantic_boundary_risk_mean",
        "semantic_continuity_score_mean",
        "Q_ref_sem_balanced_mean",
        "Q_ref_sem_risk_strict_mean",
        "Q_ref_sem_stable_strict_mean",
    ]
    write_csv(OUT / "operation_semantic_summary.csv", out_rows, fields)
    by_op = Counter(row.get("operation_type", "") for row in out_rows)
    by_surface = Counter()
    for row in out_rows:
        for sid in str(row.get("candidate_surface_ids", "")).split(";"):
            if sid:
                by_surface[sid] += 1
    return {
        "operation_row_count": len(out_rows),
        "operation_row_join_any_ratio": join_any / len(out_rows) if out_rows else 0.0,
        "operation_rows_join_coverage_mean": mean(join_coverages),
        "operation_types": dict(sorted(by_op.items())),
        "surface_row_counts": dict(sorted(by_surface.items())),
    }


def build_role_mapping_markdown(role_counts: dict[str, Counter[str]], label_name_set: set[str]) -> tuple[str, float]:
    rows = [
        "# v108TF Semantic Role Mapping",
        "",
        "Mapping uses the v107R keyword-plus-confidence/purity guard.  Labels not matched by static/dynamic/layout/ground/vegetation/sky rules map to `unknown_lowtrust`; low confidence and low purity patches may also become `unknown_lowtrust` or `semantic_boundary` at token level.",
        "",
        "| label_name | observed_patch_count | primary_observed_role | all_observed_roles | fallback_role_at_high_confidence |",
        "|---|---:|---|---|---|",
    ]
    covered = 0
    labels = sorted(label_name_set | set(role_counts))
    for label in labels:
        counter = role_counts.get(label, Counter())
        primary = counter.most_common(1)[0][0] if counter else ""
        observed = ";".join(f"{k}:{v}" for k, v in counter.most_common())
        fallback, _thing = v107r.role_for_label(label, 1.0, 1.0)
        role = primary or fallback or "unknown_lowtrust"
        covered += int(bool(role))
        rows.append(f"| {label or '(empty)'} | {sum(counter.values())} | {primary or '(not dominant)'} | {observed or '(not dominant)'} | {fallback} |")
    coverage = covered / len(labels) if labels else 0.0
    return "\n".join(rows) + "\n", coverage


def write_cue_definition_report(summary: dict[str, Any], grid: dict[str, Any]) -> None:
    text = f"""# v108TF Stage2 Cue Definition Report

This Stage2 build uses full KITTI frame universes from Stage0 baseline lengths:

```json
{json.dumps(summary.get("frame_universe_by_seq", {}), indent=2, sort_keys=True)}
```

Projection and token alignment:

- target size: `{grid["target_width"]}x{grid["target_height"]}`
- patch size: `{grid["patch_size"]}`
- patch grid: `{grid["patch_grid_h"]}x{grid["patch_grid_w"]}` = `{grid["inferred_patch_count"]}` image patch tokens
- patch start index: `{grid["patch_start_idx"]}`
- confidence maps required for the main run: `true`

Per-patch trust is `semantic_confidence * patch_purity^2`.  Per-frame role fields in `frame_semantic_summary.csv` are normalized by total frame trust.  Raw trust-mass columns are retained with the `raw_*_trust_mass` prefix for audit.

Continuity:

- source fields: `V_mask` and `seed_global_track_idx` from `masklet.pt`
- nearby radius: `+/-{NEARBY_RADIUS}` frames
- score: visible seed ids also visible in nearby frames divided by visible seed ids in the current frame

Predeclared Q_ref variants:

- balanced: `stable + 0.5 * continuity - risk - 0.5 * boundary_risk`
- risk_strict: `stable + 0.5 * continuity - 1.5 * risk - boundary_risk`
- stable_strict: `1.5 * stable + continuity - risk - 0.5 * boundary_risk`

Risk is `dynamic + boundary + weak_context + sky_lowobs`.  No GT pose, GT depth, external depth, SLAM, or full-ATE label is used by this cue bank.

Gate note: `semantic_nonvoid_frame_ratio` is the fraction of processed frames with at least one non-void semantic patch.  The stricter diagnostic `semantic_nonvoid_frame_ratio_ge_0p95` is also reported but is not silently substituted for the planned frame-level availability gate.
"""
    (OUT / "cue_definition_report.md").write_text(text, encoding="utf-8")


def main() -> None:
    import torch

    OUT.mkdir(parents=True, exist_ok=True)
    op_rows = read_csv(OP_ROWS)
    lengths = baseline_lengths()
    grid = v107r.parse_grid_from_trace(op_rows)
    target_ids = target_ids_by_frame(op_rows)

    frame_fields = [
        "schema",
        "seq_id",
        "frame_id",
        "target_ids",
        "source_chunk",
        "runtime_available",
        "confidence_maps_available",
        "semantic_projection_available",
        "patch_count",
        "semantic_patch_nonvoid_ratio",
        "stable_structure_mass",
        "dynamic_mass",
        "boundary_mass",
        "weak_context_mass",
        "road_ground_mass",
        "sky_lowobs_mass",
        "semantic_trust_mean",
        "semantic_purity_mean",
        "semantic_boundary_risk",
        "visible_seed_count",
        "semantic_continuity_score",
        "Q_ref_sem_balanced",
        "Q_ref_sem_risk_strict",
        "Q_ref_sem_stable_strict",
        "raw_stable_structure_trust_mass",
        "raw_road_boundary_or_layout_trust_mass",
        "raw_ground_or_road_weak_trust_mass",
        "raw_dynamic_transient_trust_mass",
        "raw_vegetation_weak_context_trust_mass",
        "raw_sky_or_lowobs_trust_mass",
        "raw_semantic_boundary_trust_mass",
        "raw_unknown_lowtrust_trust_mass",
    ]
    token_fields = [
        "schema",
        "seq_id",
        "frame_id",
        "token_id",
        "patch_y",
        "patch_x",
        "dominant_label",
        "label_name",
        "semantic_confidence",
        "patch_purity",
        "semantic_trust",
        "semantic_boundary_risk",
        "semantic_role",
        "thing_or_stuff",
    ]

    frame_rows: list[dict[str, Any]] = []
    missing_frames: list[dict[str, Any]] = []
    frame_seed_sets: dict[tuple[str, int], set[str]] = {}
    role_counts: dict[str, Counter[str]] = defaultdict(Counter)
    label_name_set: set[str] = set()
    frame_universe_by_seq: dict[str, int] = {}
    processed_by_seq: Counter[str] = Counter()
    token_row_count = 0
    total_patch_rows = 0
    total_nonvoid_patches = 0.0
    total_purity_sum = 0.0
    nonvoid_frame_count = 0
    nonvoid_frame_count_ge_0p95 = 0
    confidence_missing_frame_count = 0

    with (OUT / "token_semantic_rows.csv").open("w", encoding="utf-8", newline="") as token_handle:
        token_writer = csv.DictWriter(token_handle, fieldnames=token_fields, extrasaction="ignore")
        token_writer.writeheader()
        with torch.no_grad():
            for seq in SEQ_IDS:
                length = lengths.get(seq, 0)
                frame_universe_by_seq[seq] = length
                chunk_to_frames, seq_missing = assign_frames_to_chunks(seq, length)
                missing_frames.extend(seq_missing)
                for chunk_path, chunk_frames in sorted(chunk_to_frames.items(), key=lambda kv: kv[0].as_posix()):
                    payload = torch.load(chunk_path, map_location="cpu", weights_only=False)
                    sem = payload.get("semantic_segmentation", {})
                    label_maps = sem.get("label_maps")
                    conf_maps = sem.get("confidence_maps")
                    label_names = list(sem.get("label_names", []))
                    label_name_set.update(str(name) for name in label_names)
                    label_to_id = sem.get("label_to_id", {}) or {}
                    global_start = int(sem.get("global_start_frame", v107r.CHUNK_RE.search(chunk_path.parent.name).group(1)))
                    void_id = int(label_to_id.get("void", 0))
                    if label_maps is None or conf_maps is None:
                        confidence_missing_frame_count += len(chunk_frames)
                        for frame_id in chunk_frames:
                            missing_frames.append({"seq_id": seq, "frame_id": frame_id, "reason": "chunk_missing_label_or_confidence_maps"})
                        del payload
                        continue

                    for frame_id in chunk_frames:
                        local_idx = frame_id - global_start
                        if local_idx < 0 or local_idx >= int(label_maps.shape[0]):
                            missing_frames.append({"seq_id": seq, "frame_id": frame_id, "reason": "frame_not_in_loaded_chunk"})
                            continue
                        label_proj = v107r.cover_fit_resize_2d(label_maps[local_idx], grid["target_height"], grid["target_width"], "nearest").long()
                        conf_proj = v107r.cover_fit_resize_2d(conf_maps[local_idx], grid["target_height"], grid["target_width"], "bilinear").float()
                        patch_rows, patch_summary = v107r.patchify_projected_frame(label_proj, conf_proj, grid, label_names, void_id)
                        seeds = visible_seed_set(payload, local_idx)
                        frame_seed_sets[(seq, frame_id)] = seeds
                        normalized = normalized_frame_fields(patch_summary)
                        patch_count = int(patch_summary.get("patch_count", len(patch_rows)))
                        nonvoid_ratio = float(patch_summary.get("semantic_patch_nonvoid_ratio", 0.0))
                        purity_mean = float(patch_summary.get("semantic_patch_purity_mean", 0.0))
                        total_patch_rows += patch_count
                        total_nonvoid_patches += nonvoid_ratio * patch_count
                        total_purity_sum += purity_mean * patch_count
                        nonvoid_frame_count += int(nonvoid_ratio > 0.0)
                        nonvoid_frame_count_ge_0p95 += int(nonvoid_ratio >= 0.95)
                        processed_by_seq[seq] += 1
                        base_row: dict[str, Any] = {
                            "schema": "acl2_v108tf_frame_semantic_summary_row_v1",
                            "seq_id": seq,
                            "frame_id": frame_id,
                            "target_ids": ";".join(sorted(target_ids.get((seq, frame_id), set()))),
                            "source_chunk": rel(chunk_path),
                            "runtime_available": True,
                            "confidence_maps_available": True,
                            "semantic_projection_available": True,
                            "patch_count": patch_count,
                            "semantic_patch_nonvoid_ratio": nonvoid_ratio,
                            "visible_seed_count": len(seeds),
                            "semantic_continuity_score": 0.0,
                            **normalized,
                        }
                        frame_rows.append(base_row)
                        for prow in patch_rows:
                            label = str(prow["label_name"])
                            role = str(prow["semantic_role"])
                            role_counts[label][role] += 1
                            token_writer.writerow(
                                {
                                    "schema": "acl2_v108tf_token_semantic_row_v1",
                                    "seq_id": seq,
                                    "frame_id": frame_id,
                                    "token_id": prow["token_id"],
                                    "patch_y": prow["patch_y"],
                                    "patch_x": prow["patch_x"],
                                    "dominant_label": prow["dominant_label"],
                                    "label_name": label,
                                    "semantic_confidence": fmt(prow["semantic_confidence"]),
                                    "patch_purity": fmt(prow["patch_purity"]),
                                    "semantic_trust": fmt(prow["semantic_trust"]),
                                    "semantic_boundary_risk": fmt(prow["semantic_boundary_risk"]),
                                    "semantic_role": role,
                                    "thing_or_stuff": prow["thing_or_stuff"],
                                }
                            )
                        token_row_count += len(patch_rows)
                    del payload

    frame_row_by_key = {(str(row["seq_id"]), int(row["frame_id"])): row for row in frame_rows}
    for row in frame_rows:
        seq = str(row["seq_id"])
        frame_id = int(row["frame_id"])
        seeds = frame_seed_sets.get((seq, frame_id), set())
        if not seeds:
            row["semantic_continuity_score"] = 0.0
        else:
            nearby: set[str] = set()
            for offset in range(-NEARBY_RADIUS, NEARBY_RADIUS + 1):
                if offset == 0:
                    continue
                nearby.update(frame_seed_sets.get((seq, frame_id + offset), set()))
            row["semantic_continuity_score"] = len(seeds & nearby) / len(seeds)
        apply_q_ref(row)

    frame_rows.sort(key=lambda r: (str(r["seq_id"]), int(r["frame_id"])))
    write_csv(OUT / "frame_semantic_summary.csv", frame_rows, frame_fields)
    write_csv(OUT / "missing_semantic_frames.csv", missing_frames, ["seq_id", "frame_id", "reason"])
    role_markdown, role_mapping_coverage = build_role_mapping_markdown(role_counts, label_name_set)
    (OUT / "semantic_role_mapping.md").write_text(role_markdown, encoding="utf-8")
    op_summary = build_operation_semantic_summary(op_rows, frame_rows)

    expected_frame_count = sum(lengths.get(seq, 0) for seq in SEQ_IDS)
    processed_frame_count = len(frame_rows)
    coverage_by_seq = {
        seq: (processed_by_seq[seq] / lengths[seq] if lengths.get(seq, 0) else 0.0)
        for seq in SEQ_IDS
    }
    frame_coverage = processed_frame_count / expected_frame_count if expected_frame_count else 0.0
    semantic_nonvoid_frame_ratio = nonvoid_frame_count / processed_frame_count if processed_frame_count else 0.0
    semantic_nonvoid_frame_ratio_ge_0p95 = nonvoid_frame_count_ge_0p95 / processed_frame_count if processed_frame_count else 0.0
    semantic_patch_nonvoid_ratio = total_nonvoid_patches / total_patch_rows if total_patch_rows else 0.0
    semantic_patch_purity_mean = total_purity_sum / total_patch_rows if total_patch_rows else 0.0
    stage2_pass = bool(
        expected_frame_count > 0
        and processed_frame_count == expected_frame_count
        and not missing_frames
        and all(abs(coverage_by_seq.get(seq, 0.0) - 1.0) < 1.0e-12 for seq in SEQ_IDS)
        and semantic_nonvoid_frame_ratio >= 0.95
        and semantic_patch_purity_mean >= 0.80
        and op_summary["operation_rows_join_coverage_mean"] >= 0.95
        and role_mapping_coverage >= 1.0
        and confidence_missing_frame_count == 0
    )
    summary = {
        "schema": "acl2_v108tf_stage2_summary_v1",
        "stage2_pass": stage2_pass,
        "frame_universe_by_seq": frame_universe_by_seq,
        "expected_frame_count": expected_frame_count,
        "processed_frame_count": processed_frame_count,
        "frame_semantic_coverage": frame_coverage,
        "frame_semantic_coverage_by_seq": coverage_by_seq,
        "missing_frame_count": len(missing_frames),
        "confidence_missing_frame_count": confidence_missing_frame_count,
        "token_semantic_row_count": token_row_count,
        "semantic_nonvoid_frame_ratio": semantic_nonvoid_frame_ratio,
        "semantic_nonvoid_frame_ratio_ge_0p95": semantic_nonvoid_frame_ratio_ge_0p95,
        "semantic_patch_nonvoid_ratio": semantic_patch_nonvoid_ratio,
        "semantic_patch_purity_mean": semantic_patch_purity_mean,
        "semantic_role_mapping_coverage": role_mapping_coverage,
        "operation_rows_join_coverage_mean": op_summary["operation_rows_join_coverage_mean"],
        "operation_row_join_any_ratio": op_summary["operation_row_join_any_ratio"],
        "operation_row_count": op_summary["operation_row_count"],
        "operation_types": op_summary["operation_types"],
        "surface_row_counts": op_summary["surface_row_counts"],
        "token_alignment_pass": bool(grid["token_alignment_pass"]),
        "target_width": grid["target_width"],
        "target_height": grid["target_height"],
        "patch_size": grid["patch_size"],
        "patch_grid_h": grid["patch_grid_h"],
        "patch_grid_w": grid["patch_grid_w"],
        "patch_start_idx": grid["patch_start_idx"],
        "continuity_nearby_radius": NEARBY_RADIUS,
        "outputs": {
            "token_semantic_rows": rel(OUT / "token_semantic_rows.csv"),
            "frame_semantic_summary": rel(OUT / "frame_semantic_summary.csv"),
            "operation_semantic_summary": rel(OUT / "operation_semantic_summary.csv"),
            "semantic_role_mapping": rel(OUT / "semantic_role_mapping.md"),
            "cue_definition_report": rel(OUT / "cue_definition_report.md"),
        },
    }
    write_json(OUT / "stage2_summary.json", summary)
    write_cue_definition_report(summary, grid)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
