#!/usr/bin/env python3
"""Repair v99 holdout scene stitching with chunk32/overlap3.

Phase10K/L regenerated holdout object birth with non-overlapping chunks because
the Phase1 helper only exposed a single frame_to_chunk owner. This script keeps
the old artifacts intact and rebuilds the holdout branch with an explicit
frame_to_chunks scope:

* internal inference chunks are size 32 with 3-frame overlap;
* overlap frames may belong to two chunks and are used for identity stitching;
* metric emission keeps each real frame once, owned by the earliest chunk, so
  the scene evaluator does not double-count overlap frames.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
if str(STREAM3D_ROOT) not in sys.path:
    sys.path.insert(0, str(STREAM3D_ROOT))

from tools import build_v99_phase1_f2_base_reproduction as p1  # noqa: E402
from tools import build_v99_phase10k_holdout_chunk_object_birth_sweep as p10k  # noqa: E402
from tools import build_v99_phase10l_frozen_p2d2_regenerated_birth_holdout as p10l  # noqa: E402
from tools.build_v99_phase9_scene_local2history import DSU  # noqa: E402


AUDIT_ROOT = STREAM3D_ROOT / "outputs/audit"
OUT_DIR = AUDIT_ROOT / "v99_phase10o_overlap3_scene_stitch_repair"
PHASE0_DIR = AUDIT_ROOT / "v99_phase0_fact_lock"
PHASE10L_DIR = AUDIT_ROOT / "v99_phase10l_frozen_p2d2_regenerated_birth_holdout"
PHASE10N_DIR = AUDIT_ROOT / "v99_phase10n_scene_fragmentation_audit"

CHUNK_SIZE = 32
OVERLAP = 3
STEP = CHUNK_SIZE - OVERLAP
FROZEN_BIRTH_VARIANT = "F2_chunk32_surfel_maskview_birth_thr018"
FROZEN_SCORE_VARIANT = "P2_D2_frame_count_plus_semantic_tiebreak"
BASE_OVERLAP_VARIANT = "O0_overlap3_chunk_birth_primary_emit"
EPS = 1e-4


def _rel(path: Path | str) -> str:
    q = Path(path)
    try:
        return q.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return q.as_posix()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return _rel(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields or ["schema_version"])
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _jsonable(row.get(field, "")) for field in fields})


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return float(default)
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _build_overlap3_scope() -> dict[str, Any]:
    rows = _read_csv(p1.SOURCE_ROWS)
    frames_by_scene: dict[str, set[int]] = defaultdict(set)
    mask_path_by_frame: dict[tuple[str, int], Path] = {}
    source_uses_future = False
    source_uses_gt = False
    for row in rows:
        scene = row.get("scene_id", "")
        frame = p1._int(row.get("frame_id"), -1)
        if not scene or frame < 0:
            continue
        frames_by_scene[scene].add(frame)
        if row.get("mask_path"):
            mask_path_by_frame.setdefault((scene, frame), p1._project(row["mask_path"]))
        source_uses_future = source_uses_future or p1._bool(row.get("uses_future"))
        source_uses_gt = source_uses_gt or p1._bool(row.get("uses_gt_for_prediction"))

    chunks: dict[tuple[str, str], list[int]] = {}
    chunk_index_range: dict[tuple[str, str], tuple[int, int]] = {}
    frame_to_chunks: dict[tuple[str, int], list[str]] = defaultdict(list)
    primary_frame_to_chunk: dict[tuple[str, int], str] = {}
    eval_chunks: dict[tuple[str, str], list[int]] = defaultdict(list)

    for scene, frame_set in sorted(frames_by_scene.items()):
        frames = sorted(frame_set)
        for chunk_index, start in enumerate(range(0, len(frames), STEP)):
            chunk_frames = frames[start : start + CHUNK_SIZE]
            if not chunk_frames:
                continue
            chunk_id = f"c{chunk_index:04d}"
            chunks[(scene, chunk_id)] = chunk_frames
            chunk_index_range[(scene, chunk_id)] = (start, start + len(chunk_frames) - 1)
            for frame in chunk_frames:
                frame_to_chunks[(scene, frame)].append(chunk_id)
                primary_frame_to_chunk.setdefault((scene, frame), chunk_id)
            if len(chunk_frames) < CHUNK_SIZE:
                break
        for frame in frames:
            owner = primary_frame_to_chunk[(scene, frame)]
            eval_chunks[(scene, owner)].append(frame)

    overlap_pair_rows: list[dict[str, Any]] = []
    for scene in sorted(frames_by_scene):
        chunk_ids = sorted(chunk for row_scene, chunk in chunks if row_scene == scene)
        for left, right in zip(chunk_ids[:-1], chunk_ids[1:]):
            shared = sorted(set(chunks[(scene, left)]) & set(chunks[(scene, right)]))
            overlap_pair_rows.append(
                {
                    "schema_version": "stream4d_v99_phase10o_overlap_pair_v1",
                    "phase_id": "v99_phase10o_overlap3_scene_stitch_repair",
                    "scene_id": scene,
                    "left_chunk_id": left,
                    "right_chunk_id": right,
                    "shared_frame_count": len(shared),
                    "shared_frames": " ".join(str(v) for v in shared),
                    "chunk_size": CHUNK_SIZE,
                    "overlap": OVERLAP,
                    "step": STEP,
                }
            )

    return {
        "source_row_count": len(rows),
        "source_rows": rows,
        "frames_by_scene": {scene: sorted(vals) for scene, vals in frames_by_scene.items()},
        "chunks": dict(chunks),
        "eval_chunks": {key: sorted(vals) for key, vals in eval_chunks.items()},
        "frame_to_chunks": {key: sorted(vals) for key, vals in frame_to_chunks.items()},
        "primary_frame_to_chunk": primary_frame_to_chunk,
        "chunk_index_range": chunk_index_range,
        "mask_path_by_frame": mask_path_by_frame,
        "source_uses_future": source_uses_future,
        "source_uses_gt_for_prediction": source_uses_gt,
        "overlap_pair_rows": overlap_pair_rows,
    }


def _eval_scope_from_overlap(scope: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_row_count": scope["source_row_count"],
        "source_rows": scope["source_rows"],
        "frames_by_scene": scope["frames_by_scene"],
        "chunks": scope["eval_chunks"],
        "frame_to_chunk": scope["primary_frame_to_chunk"],
        "mask_path_by_frame": scope["mask_path_by_frame"],
        "source_uses_future": scope["source_uses_future"],
        "source_uses_gt_for_prediction": scope["source_uses_gt_for_prediction"],
    }


def _build_overlap_surfel_rows(scope: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    features, tau = p1._load_radio_residual_features()
    surfel_rows_raw = _read_csv(p1.SURFEL_ROWS)
    surfel_obs_raw = _read_csv(p1.SURFEL_OBS_ROWS)
    surfel_summary = json.loads(p1.SURFEL_SUMMARY.read_text(encoding="utf-8"))
    input_uses_future = bool(surfel_summary.get("uses_future", False))
    surfel_by_id = {row["surfel_id"]: row for row in surfel_rows_raw if row.get("surfel_id")}

    source_mask_area: dict[tuple[str, int, int], float] = {}
    for row in scope["source_rows"]:
        scene = row.get("scene_id", "")
        frame = p1._int(row.get("frame_id"), -1)
        mask_id = p1._int(row.get("mask_id_or_generated_id") or row.get("selected_mask_id"), -1)
        if scene and frame >= 0 and mask_id > 0:
            source_mask_area[(scene, frame, mask_id)] = _num(row.get("mask_area_ratio"))

    obs_by_chunk: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    skipped_obs: Counter[str] = Counter()
    for row in surfel_obs_raw:
        scene = row.get("scene_id", "")
        frame = p1._int(row.get("frame_id"), -1)
        mask_ids = [p1._int(tok, -1) for tok in str(row.get("mask_ids_covering", "")).replace(";", " ").split()]
        mask_ids = [m for m in mask_ids if m > 0]
        mask_id = mask_ids[0] if mask_ids else -1
        sid = row.get("surfel_id", "")
        chunk_ids = scope["frame_to_chunks"].get((scene, frame), [])
        if not scene or frame < 0 or mask_id <= 0 or not sid:
            skipped_obs["invalid_row"] += 1
            continue
        if sid not in surfel_by_id:
            skipped_obs["missing_surfel_row"] += 1
            continue
        if not chunk_ids:
            skipped_obs["outside_source_chunk"] += 1
            continue
        for chunk_id in chunk_ids:
            obs_by_chunk[(scene, chunk_id)].append(
                {
                    "surfel_id": sid,
                    "scene_id": scene,
                    "chunk_id": chunk_id,
                    "frame_id": frame,
                    "mask_id": mask_id,
                    "x_orig": p1._int(row.get("x_orig"), -1),
                    "y_orig": p1._int(row.get("y_orig"), -1),
                    "provider_confidence": _num(row.get("provider_confidence")),
                    "d4rt_anchor_ids_nearby": row.get("d4rt_anchor_ids_nearby", ""),
                    "feature": features.get((scene, frame, mask_id)),
                    "mask_area_ratio": source_mask_area.get((scene, frame, mask_id), 0.0),
                }
            )

    cfg = {"variant_id": FROZEN_BIRTH_VARIANT, "threshold": 0.18, "min_support": 3, "mask": 0.75, "semantic": 0.30, "edge": 0.75}
    rng = np.random.default_rng(9910)
    out_rows: list[dict[str, Any]] = []
    object_rows: list[dict[str, Any]] = []
    candidate_edge_count = 0
    cannot_edge_count = 0
    accepted_edge_count = 0
    rejected_union_count = 0
    chunk_stats: list[dict[str, Any]] = []

    for (scene, chunk_id), obs_rows in sorted(obs_by_chunk.items()):
        if not obs_rows:
            continue
        surfel_ids = sorted({row["surfel_id"] for row in obs_rows})
        surfel_to_idx = {sid: idx for idx, sid in enumerate(surfel_ids)}
        obs_by_surfel: dict[int, list[dict[str, Any]]] = defaultdict(list)
        obs_by_mask: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
        obs_by_frame: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
        for obs0 in obs_rows:
            obs = dict(obs0)
            idx = surfel_to_idx[obs["surfel_id"]]
            obs["surfel_index"] = idx
            obs_by_surfel[idx].append(obs)
            obs_by_mask[(scene, int(obs["frame_id"]), int(obs["mask_id"]))].append(obs)
            obs_by_frame[(scene, int(obs["frame_id"]))].append(obs)

        surfel_sem_vectors: dict[int, list[np.ndarray]] = defaultdict(list)
        for obs in obs_rows:
            feat = obs.get("feature")
            if feat is not None:
                surfel_sem_vectors[surfel_to_idx[obs["surfel_id"]]].append(feat)
        surfel_sem: dict[int, np.ndarray] = {}
        for idx, vecs in surfel_sem_vectors.items():
            if vecs:
                surfel_sem[idx] = p1._normalize_rows(np.mean(np.stack(vecs), axis=0, keepdims=True))[0]

        frame_totals = {key: len(rows) for key, rows in obs_by_frame.items()}
        positive_base: dict[tuple[int, int], dict[str, float]] = {}
        for key, rows in sorted(obs_by_mask.items()):
            unique_indices = sorted({int(row["surfel_index"]) for row in rows})
            if len(unique_indices) < 2:
                continue
            if len(unique_indices) <= 80:
                all_pairs = [(a, b) for pos, a in enumerate(unique_indices[:-1]) for b in unique_indices[pos + 1 :]]
            else:
                all_pairs = []
                for _ in range(p1.MAX_POSITIVE_PAIRS_PER_MASK * 3):
                    a, b = rng.choice(unique_indices, size=2, replace=False)
                    a_i, b_i = int(a), int(b)
                    if a_i > b_i:
                        a_i, b_i = b_i, a_i
                    all_pairs.append((a_i, b_i))
            if len(all_pairs) > p1.MAX_POSITIVE_PAIRS_PER_MASK:
                take = rng.choice(len(all_pairs), size=p1.MAX_POSITIVE_PAIRS_PER_MASK, replace=False)
                all_pairs = [all_pairs[int(i)] for i in take]
            _row_scene, frame, mask_id = key
            rho = float(len(rows) / max(1, frame_totals.get((scene, frame), len(rows))))
            b_centered = float((1.0 - rho) / math.sqrt(max(1e-6, rho * (1.0 - rho))))
            area = source_mask_area.get((scene, frame, mask_id), rows[0].get("mask_area_ratio", 0.0))
            _fine, obj_w, _coarse, broad_risk, _label = p1._mask_scale_weights(area)
            for a, b in all_pairs:
                stats = positive_base.setdefault((a, b), {"mask": 0.0, "count": 0.0, "risk": 0.0})
                stats["mask"] += b_centered * b_centered * obj_w
                stats["count"] += 1.0
                stats["risk"] = max(stats["risk"], broad_risk)

        cannot_edges: set[tuple[int, int]] = set()
        for (_scene, _frame), rows in sorted(obs_by_frame.items()):
            by_mask: dict[int, list[int]] = defaultdict(list)
            for obs in rows:
                by_mask[int(obs["mask_id"])].append(int(obs["surfel_index"]))
            mask_ids = sorted(by_mask)
            if len(mask_ids) < 2:
                continue
            candidate: list[tuple[int, int]] = []
            for i, ma in enumerate(mask_ids[:-1]):
                lhs = sorted(set(by_mask[ma]))[:40]
                for mb in mask_ids[i + 1 :]:
                    rhs = sorted(set(by_mask[mb]))[:40]
                    for a in lhs[:8]:
                        for b in rhs[:8]:
                            if a == b:
                                continue
                            candidate.append((min(a, b), max(a, b)))
            if len(candidate) > p1.MAX_NEGATIVE_PAIRS_PER_FRAME:
                take = rng.choice(len(candidate), size=p1.MAX_NEGATIVE_PAIRS_PER_FRAME, replace=False)
                candidate = [candidate[int(i)] for i in take]
            cannot_edges.update(candidate)
        cannot_edge_count += len(cannot_edges)

        base_edges: list[dict[str, Any]] = []
        for (a, b), stats in positive_base.items():
            sem_score = 0.0
            if a in surfel_sem and b in surfel_sem:
                raw_cos = p1._cosine(surfel_sem[a], surfel_sem[b])
                sem_score = float(max(0.0, (raw_cos - tau) / max(1e-6, 1.0 - tau)))
            base_edges.append(
                {
                    "surfel_index_a": a,
                    "surfel_index_b": b,
                    "A_mask_centered": float(stats["mask"] / max(1.0, stats["count"])),
                    "A_sem_residual": sem_score,
                    "edge_penalty": float(stats["risk"]),
                    "conflict_penalty": 1.0 if (a, b) in cannot_edges else 0.0,
                }
            )
        candidate_edge_count += len(base_edges)

        def signed(edge: dict[str, Any]) -> float:
            return float(
                cfg["mask"] * _num(edge.get("A_mask_centered"))
                + cfg["semantic"] * _num(edge.get("A_sem_residual"))
                - cfg["edge"] * _num(edge.get("edge_penalty"))
                - 2.0 * _num(edge.get("conflict_penalty"))
            )

        dsu = p1._ConstrainedDSU(len(surfel_ids), cannot_edges)
        for edge in sorted(base_edges, key=signed, reverse=True):
            a = int(edge["surfel_index_a"])
            b = int(edge["surfel_index_b"])
            if (a, b) in cannot_edges or signed(edge) < float(cfg["threshold"]):
                continue
            before_a = dsu.find(a)
            before_b = dsu.find(b)
            if dsu.union(a, b) and before_a != before_b:
                accepted_edge_count += 1
        rejected_union_count += dsu.rejected_unions

        object_index = 0
        for _root, indices in sorted(dsu.components().items(), key=lambda item: (-len(item[1]), item[0])):
            if len(indices) < 2:
                continue
            component_obs: list[dict[str, Any]] = []
            for idx in indices:
                component_obs.extend(obs_by_surfel.get(idx, []))
            by_frame_mask: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
            for obs in component_obs:
                by_frame_mask[(scene, int(obs["frame_id"]), int(obs["mask_id"]))].append(obs)
            emitted_keys = {
                key: rows
                for key, rows in by_frame_mask.items()
                if len({int(obs["surfel_index"]) for obs in rows}) >= int(cfg["min_support"])
            }
            emitted_frames = sorted({frame for _scene, frame, _mask_id in emitted_keys})
            if len(emitted_frames) < 2:
                continue
            object_id = f"{FROZEN_BIRTH_VARIANT}:{scene}:{chunk_id}:obj_{object_index:05d}"
            object_index += 1
            score = len(emitted_frames) / float(CHUNK_SIZE)
            object_rows.append(
                {
                    "schema_version": "stream4d_v99_phase10o_mv_object_row_v1",
                    "phase_id": "v99_phase10o_overlap3_scene_stitch_repair",
                    "variant_id": FROZEN_BIRTH_VARIANT,
                    "mv_object_id": object_id,
                    "legacy_mv_object_id": "",
                    "scene_id": scene,
                    "chunk_id": chunk_id,
                    "object_frame_count": len(emitted_frames),
                    "object_score": score,
                    "score_scope": "current_chunk",
                    "score_policy": "current_chunk_frame_count_over_32",
                    "object_id_policy": "chunk32_overlap3_surfel_maskview_component",
                    "object_birth_scope": "current_chunk32_overlap3_surfel_maskview_birth_from_v98_phase5_surfel_identity",
                    "surfel_identity_scope": "v98_phase13_holdout_phase5_surfel_identity_restricted_to_current_overlap_chunk_observations",
                    "surfel_dependency_proven_chunk_causal": False,
                    "threshold": cfg["threshold"],
                    "min_support": cfg["min_support"],
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )
            for (_row_scene, frame, mask_id), rows in sorted(emitted_keys.items()):
                support_surfel_count = len({int(obs["surfel_index"]) for obs in rows})
                out_rows.append(
                    {
                        "schema_version": "stream4d_v99_phase10o_mv_object_frame_mask_row_v1",
                        "phase_id": "v99_phase10o_overlap3_scene_stitch_repair",
                        "variant_id": FROZEN_BIRTH_VARIANT,
                        "mv_object_id": object_id,
                        "object_id": object_id,
                        "legacy_mv_object_id": "",
                        "scene_id": scene,
                        "chunk_id": chunk_id,
                        "window_id": chunk_id,
                        "frame_id": frame,
                        "selected_mask_id": mask_id,
                        "mask_id_or_generated_id": mask_id,
                        "readout_mode": "current_chunk32_overlap3_surfel_maskview_birth",
                        "score": score,
                        "score_scope": "current_chunk",
                        "score_policy": "current_chunk_frame_count_over_32",
                        "object_id_policy": "chunk32_overlap3_surfel_maskview_component",
                        "method_chunk_size": CHUNK_SIZE,
                        "method_chunk_overlap": OVERLAP,
                        "frame_stride": p1.FRAME_STRIDE,
                        "support_surfel_count": support_surfel_count,
                        "support_iou": "",
                        "support_recall": "",
                        "mask_precision": "",
                        "support_area": "",
                        "selected_mask_area": "",
                        "threshold": cfg["threshold"],
                        "min_support": cfg["min_support"],
                        "uses_gt_for_prediction": False,
                        "uses_future": False,
                        "object_birth_scope": "current_chunk32_overlap3_surfel_maskview_birth_from_v98_phase5_surfel_identity",
                        "surfel_identity_scope": "v98_phase13_holdout_phase5_surfel_identity_restricted_to_current_overlap_chunk_observations",
                        "surfel_dependency_proven_chunk_causal": False,
                        "emit_role": "internal_context_or_primary",
                    }
                )
        chunk_stats.append(
            {
                "scene_id": scene,
                "chunk_id": chunk_id,
                "observation_count": len(obs_rows),
                "local_surfel_count": len(surfel_ids),
                "mask_group_count": len(obs_by_mask),
                "candidate_edge_count": len(base_edges),
                "cannot_edge_count": len(cannot_edges),
            }
        )

    nms_drop_count = 0
    best_by_frame_mask: dict[tuple[str, str, str, int, int], dict[str, Any]] = {}
    for row in out_rows:
        key = (
            str(row["variant_id"]),
            str(row["scene_id"]),
            str(row["chunk_id"]),
            int(row["frame_id"]),
            int(row["selected_mask_id"]),
        )
        current = best_by_frame_mask.get(key)
        row_rank = (_num(row.get("score")), _num(row.get("support_surfel_count")), str(row.get("mv_object_id", "")))
        cur_rank = (
            _num(current.get("score")) if current else -1.0,
            _num(current.get("support_surfel_count")) if current else -1.0,
            str(current.get("mv_object_id", "")) if current else "",
        )
        if current is None or row_rank > cur_rank:
            if current is not None:
                nms_drop_count += 1
            best_by_frame_mask[key] = row
        else:
            nms_drop_count += 1
    out_rows = sorted(
        best_by_frame_mask.values(),
        key=lambda row: (
            str(row["variant_id"]),
            str(row["scene_id"]),
            str(row["chunk_id"]),
            int(row["frame_id"]),
            int(row["selected_mask_id"]),
            str(row["mv_object_id"]),
        ),
    )
    used_objects = {str(row["mv_object_id"]) for row in out_rows}
    object_rows = [row for row in object_rows if str(row["mv_object_id"]) in used_objects]

    post_nms_frames_by_object: dict[str, set[int]] = defaultdict(set)
    for row in out_rows:
        post_nms_frames_by_object[str(row["mv_object_id"])].add(int(row["frame_id"]))
    post_nms_score_by_object = {
        oid: len(frames) / float(CHUNK_SIZE)
        for oid, frames in post_nms_frames_by_object.items()
    }
    for row in out_rows:
        oid = str(row["mv_object_id"])
        row["score"] = post_nms_score_by_object.get(oid, _num(row.get("score")))
        row["score_policy"] = "current_chunk_frame_count_over_32_post_nms"
        row["object_frame_count_post_nms"] = len(post_nms_frames_by_object.get(oid, set()))
    for row in object_rows:
        oid = str(row["mv_object_id"])
        if oid in post_nms_score_by_object:
            row["object_score"] = post_nms_score_by_object[oid]
            row["score_policy"] = "current_chunk_frame_count_over_32_post_nms"
            row["object_frame_count"] = len(post_nms_frames_by_object[oid])

    stats = {
        "semantic_tau": tau,
        "input_surfel_row_count": len(surfel_rows_raw),
        "input_surfel_observation_row_count": len(surfel_obs_raw),
        "input_phase5_uses_future": input_uses_future,
        "surfel_identity_scope": "v98_phase13_holdout_phase5_surfel_identity_restricted_to_current_overlap_chunk_observations",
        "surfel_dependency_proven_chunk_causal": False,
        "chunk_count": len(obs_by_chunk),
        "chunk_size": CHUNK_SIZE,
        "overlap": OVERLAP,
        "step": STEP,
        "candidate_edge_count": candidate_edge_count,
        "cannot_edge_count": cannot_edge_count,
        "accepted_edge_count": int(accepted_edge_count),
        "rejected_union_due_cannot_link_count": int(rejected_union_count),
        "frame_mask_nms_dropped_claim_count": int(nms_drop_count),
        "post_nms_score_recomputed": True,
        "post_nms_score_policy": "current_chunk_frame_count_over_32_post_nms",
        "skipped_observation_counts": dict(skipped_obs),
        "chunk_stats": chunk_stats,
        "emitted_object_count": len(object_rows),
        "emitted_object_frame_mask_count": len(out_rows),
    }
    return out_rows, object_rows, stats


def _semantic_norm_by_object(rows: list[dict[str, Any]]) -> dict[str, float]:
    features = p10l._load_holdout_residual_features()
    by_object: dict[str, list[np.ndarray]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["scene_id"]),
            int(row["frame_id"]),
            int(row["selected_mask_id"]),
        )
        feat = features.get(key)
        if feat is not None:
            by_object[str(row["mv_object_id"])].append(feat)
    raw: dict[str, float] = {}
    for oid, vals in by_object.items():
        if len(vals) < 2:
            raw[oid] = 0.0
            continue
        stack = np.stack(vals).astype(np.float32)
        centroid = p1._normalize_rows(np.mean(stack, axis=0, keepdims=True))[0]
        raw[oid] = float(np.mean([p1._cosine(row, centroid) for row in stack]))
    for row in rows:
        raw.setdefault(str(row["mv_object_id"]), 0.0)
    return p10l._norm(raw)


def _apply_frozen_p2d2_score(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frames_by_object: dict[str, set[tuple[str, int]]] = defaultdict(set)
    for row in rows:
        frames_by_object[str(row["mv_object_id"])].add((str(row["scene_id"]), int(row["frame_id"])))
    semantic_norm = _semantic_norm_by_object(rows)
    out: list[dict[str, Any]] = []
    for row in rows:
        oid = str(row["mv_object_id"])
        frame_count_score = len(frames_by_object[oid]) / float(CHUNK_SIZE)
        new = dict(row)
        new["variant_id"] = BASE_OVERLAP_VARIANT
        new["variant"] = BASE_OVERLAP_VARIANT
        new["score"] = float(frame_count_score + EPS * semantic_norm.get(oid, 0.0))
        new["score_scope"] = "current_chunk"
        new["score_policy"] = "current_chunk_frame_count_over_32_plus_1e-4_semantic_consistency_tiebreak"
        new["fixed_dev_variant_id"] = FROZEN_SCORE_VARIANT
        new["fixed_birth_variant_id"] = FROZEN_BIRTH_VARIANT
        new["phase10o_parent_variant_id"] = row.get("variant_id", "")
        new["phase10o_frame_count_score"] = frame_count_score
        new["phase10o_semantic_norm"] = semantic_norm.get(oid, 0.0)
        new["uses_gt_for_prediction"] = False
        new["uses_future"] = False
        out.append(new)
    return out


def _primary_emit_rows(rows: list[dict[str, Any]], scope: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        scene = str(row["scene_id"])
        frame = int(row["frame_id"])
        chunk_id = str(row["chunk_id"])
        primary = str(scope["primary_frame_to_chunk"].get((scene, frame), ""))
        if primary != chunk_id:
            continue
        new = dict(row)
        new["emit_role"] = "primary_frame_owner"
        new["eval_emit_policy"] = "earliest_chunk_owns_frame_overlap_context_not_double_counted"
        out.append(new)
    return out


def _object_infos(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    features = p10l._load_holdout_residual_features()
    infos: dict[str, dict[str, Any]] = defaultdict(lambda: {"rows": [], "frames": set(), "frame_masks": set(), "features": []})
    for row in rows:
        oid = str(row["mv_object_id"])
        scene = str(row["scene_id"])
        chunk = str(row["chunk_id"])
        frame = int(row["frame_id"])
        mask = int(row["selected_mask_id"])
        infos[oid]["rows"].append(row)
        infos[oid]["scene_id"] = scene
        infos[oid]["chunk_id"] = chunk
        infos[oid]["frames"].add(frame)
        infos[oid]["frame_masks"].add((frame, mask))
        feat = features.get((scene, frame, mask))
        if feat is not None:
            infos[oid]["features"].append(feat)
    for oid, info in infos.items():
        frames = sorted(info["frames"])
        info["first_frame"] = frames[0] if frames else -1
        info["last_frame"] = frames[-1] if frames else -1
        info["frame_count"] = len(frames)
        feats = info["features"]
        if feats:
            info["feature"] = p1._normalize_rows(np.mean(np.stack(feats).astype(np.float32), axis=0, keepdims=True))[0]
        else:
            info["feature"] = None
    return dict(infos)


def _chunk_order(scope: dict[str, Any]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = defaultdict(list)
    for scene, chunk in sorted(scope["chunks"]):
        out[scene].append(chunk)
    return dict(out)


def _adjacent_pairs(infos: dict[str, dict[str, Any]], scope: dict[str, Any]) -> list[tuple[str, str, str, str]]:
    by_scene_chunk: dict[tuple[str, str], list[str]] = defaultdict(list)
    for oid, info in infos.items():
        by_scene_chunk[(str(info["scene_id"]), str(info["chunk_id"]))].append(oid)
    pairs: list[tuple[str, str, str, str]] = []
    for scene, chunks in _chunk_order(scope).items():
        for left, right in zip(chunks[:-1], chunks[1:]):
            for a in sorted(by_scene_chunk[(scene, left)]):
                for b in sorted(by_scene_chunk[(scene, right)]):
                    pairs.append((scene, left, a, b))
    return pairs


def _make_candidates(infos: dict[str, dict[str, Any]], scope: dict[str, Any], *, semantic_tau: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scene, left_chunk, a, b in _adjacent_pairs(infos, scope):
        left_masks = set(infos[a]["frame_masks"])
        right_masks = set(infos[b]["frame_masks"])
        shared_masks = left_masks & right_masks
        fa = infos[a].get("feature")
        fb = infos[b].get("feature")
        sem = float(np.dot(fa, fb)) if fa is not None and fb is not None else 0.0
        if shared_masks:
            shared_frames = {frame for frame, _mask in shared_masks}
            rows.append(
                {
                    "schema_version": "stream4d_v99_phase10o_local2history_candidate_v1",
                    "phase_id": "v99_phase10o_overlap3_scene_stitch_repair",
                    "scene_id": scene,
                    "left_chunk_id": left_chunk,
                    "right_chunk_id": str(infos[b]["chunk_id"]),
                    "mv_object_id_a": a,
                    "mv_object_id_b": b,
                    "candidate_family": "exact_overlap_frame_mask",
                    "shared_frame_mask_count": len(shared_masks),
                    "shared_frame_count": len(shared_frames),
                    "semantic_cosine": sem,
                    "affinity": float(1.0 + 0.01 * len(shared_masks) + 0.001 * sem),
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )
        elif sem >= semantic_tau:
            rows.append(
                {
                    "schema_version": "stream4d_v99_phase10o_local2history_candidate_v1",
                    "phase_id": "v99_phase10o_overlap3_scene_stitch_repair",
                    "scene_id": scene,
                    "left_chunk_id": left_chunk,
                    "right_chunk_id": str(infos[b]["chunk_id"]),
                    "mv_object_id_a": a,
                    "mv_object_id_b": b,
                    "candidate_family": f"semantic_residual_tau{semantic_tau:.2f}",
                    "shared_frame_mask_count": 0,
                    "shared_frame_count": 0,
                    "semantic_cosine": sem,
                    "affinity": sem,
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )
    return rows


def _one_to_one_mapping(ids: list[str], candidates: list[dict[str, Any]], *, variant_id: str) -> tuple[dict[str, str], list[dict[str, Any]]]:
    dsu = DSU(ids)
    used_left: set[tuple[str, str, str]] = set()
    used_right: set[tuple[str, str, str]] = set()
    accepted: list[dict[str, Any]] = []
    for row in sorted(candidates, key=lambda item: (_num(item.get("affinity")), _num(item.get("shared_frame_mask_count")), str(item.get("mv_object_id_a")), str(item.get("mv_object_id_b"))), reverse=True):
        a = str(row.get("mv_object_id_a"))
        b = str(row.get("mv_object_id_b"))
        scene = str(row.get("scene_id"))
        left_chunk = str(row.get("left_chunk_id"))
        right_chunk = str(row.get("right_chunk_id"))
        left_key = (scene, left_chunk, a)
        right_key = (scene, right_chunk, b)
        if a not in dsu.parent or b not in dsu.parent or left_key in used_left or right_key in used_right:
            continue
        if dsu.union(a, b):
            new = dict(row)
            new["schema_version"] = "stream4d_v99_phase10o_local2history_merge_v1"
            new["variant_id"] = variant_id
            new["merge_policy"] = "adjacent_chunk_overlap3_one_to_one_greedy"
            accepted.append(new)
            used_left.add(left_key)
            used_right.add(right_key)
    mapping = {oid: f"{variant_id}:{dsu.find(oid)}" for oid in ids}
    return mapping, accepted


def _apply_mapping(rows: list[dict[str, Any]], *, variant_id: str, mapping: dict[str, str], policy: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        oid = str(row["mv_object_id"])
        new = dict(row)
        new["variant_id"] = variant_id
        new["variant"] = variant_id
        new["phase10o_parent_mv_object_id"] = oid
        new["mv_object_id"] = mapping.get(oid, f"{variant_id}:{oid}")
        new["object_id"] = new["mv_object_id"]
        new["object_id_policy"] = policy
        new["score_scope"] = "current_chunk_score_scene_stitched_identity"
        new["score_policy"] = str(row.get("score_policy", "")) + "__phase10o_overlap3_scene_stitch"
        new["uses_gt_for_prediction"] = False
        new["uses_future"] = False
        out.append(new)
    return out


def _component_stats(mapping: dict[str, str]) -> dict[str, Any]:
    comps: dict[str, int] = defaultdict(int)
    for root in mapping.values():
        comps[root] += 1
    original = len(mapping)
    scene_objects = len(comps)
    merged = original - scene_objects
    large = sum(1 for size in comps.values() if size > 3)
    return {
        "scene_object_count": scene_objects,
        "history_merge_count": merged,
        "history_split_count": 0,
        "duplicate_scene_object_count": merged,
        "fragmentation_rate_proxy": float(scene_objects / original) if original else 0.0,
        "overmerge_rate_proxy_large_component_gt3": float(large / scene_objects) if scene_objects else 0.0,
        "max_component_size": max(comps.values()) if comps else 0,
    }


def _aggregate_decoupled(
    variant_id: str,
    local_rows: list[dict[str, Any]],
    scene_rows: list[dict[str, Any]],
    *,
    component_stats: dict[str, Any],
    candidate_count: int,
    accepted_count: int,
) -> dict[str, Any]:
    local_agg = p1._aggregate_metrics(local_rows)[0]
    scene_agg = p1._aggregate_metrics(scene_rows)[0]
    row = dict(local_agg)
    for key, value in scene_agg.items():
        if key.endswith("_scene") or key in {"gt_object_count_scene", "pred_object_count_scene"}:
            row[key] = value
    row["variant_id"] = variant_id
    row["metric_composition"] = "local_from_primary_chunk_ids_scene_from_overlap3_stitched_ids"
    row["history_candidate_count"] = int(candidate_count)
    row["accepted_history_merge_edge_count"] = int(accepted_count)
    row.update(component_stats)
    return row


def _eval_variant(variant_id: str, rows: list[dict[str, Any]], eval_scope: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    metric_rows, frame_rows = p1._evaluate_variant(variant_id, rows, eval_scope)
    agg = p1._aggregate_metrics(metric_rows)
    if len(agg) != 1:
        raise RuntimeError(f"expected one aggregate for {variant_id}, got {len(agg)}")
    return agg[0], metric_rows, frame_rows


def main() -> int:
    started = datetime.now()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p10k._patch_phase1_inputs()

    phase0 = json.loads((PHASE0_DIR / "summary.json").read_text(encoding="utf-8"))
    old10l = json.loads((PHASE10L_DIR / "summary.json").read_text(encoding="utf-8")) if (PHASE10L_DIR / "summary.json").exists() else {}
    old10n = json.loads((PHASE10N_DIR / "summary.json").read_text(encoding="utf-8")) if (PHASE10N_DIR / "summary.json").exists() else {}

    scope = _build_overlap3_scope()
    eval_scope = _eval_scope_from_overlap(scope)
    raw_rows, object_rows, birth_stats = _build_overlap_surfel_rows(scope)
    frozen_internal_rows = _apply_frozen_p2d2_score(raw_rows)
    primary_rows = _primary_emit_rows(frozen_internal_rows, scope)

    base_rows = []
    for row in primary_rows:
        new = dict(row)
        new["variant_id"] = BASE_OVERLAP_VARIANT
        new["variant"] = BASE_OVERLAP_VARIANT
        new["object_id_policy"] = "chunk32_overlap3_primary_emit_chunk_scoped_identity"
        base_rows.append(new)

    base_agg, base_metric_rows, base_frame_rows = _eval_variant(BASE_OVERLAP_VARIANT, base_rows, eval_scope)
    infos = _object_infos(frozen_internal_rows)
    ids = sorted(infos)

    variant_specs: list[dict[str, Any]] = []
    exact_candidates = _make_candidates(infos, scope, semantic_tau=1.01)
    exact_candidates = [row for row in exact_candidates if row["candidate_family"] == "exact_overlap_frame_mask"]
    variant_specs.append(
        {
            "variant_id": "O1_overlap3_exact_frame_mask_stitch",
            "policy": "overlap3_exact_shared_frame_mask_one_to_one",
            "family": "overlap3_exact_maskview",
            "candidates": exact_candidates,
        }
    )
    for tau, label in [(0.98, "0p98"), (0.95, "0p95"), (0.90, "0p90")]:
        sem_candidates = _make_candidates(infos, scope, semantic_tau=tau)
        variant_specs.append(
            {
                "variant_id": f"O2_overlap3_exact_plus_semantic_tau{label}",
                "policy": f"overlap3_exact_maskview_or_semantic_tau{tau:.2f}_one_to_one",
                "family": "overlap3_exact_plus_semantic",
                "candidates": sem_candidates,
            }
        )

    metric_rows: list[dict[str, Any]] = []
    single_identity_metric_rows: list[dict[str, Any]] = []
    scene_metric_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    merge_rows: list[dict[str, Any]] = []
    config_rows: list[dict[str, Any]] = []
    all_primary_rows: list[dict[str, Any]] = []

    base_stats = _component_stats({oid: f"{BASE_OVERLAP_VARIANT}:{oid}" for oid in ids})
    base_agg["metric_composition"] = "local_and_scene_from_primary_chunk_scoped_ids"
    base_agg.update(base_stats)
    base_agg["history_candidate_count"] = 0
    base_agg["accepted_history_merge_edge_count"] = 0
    metric_rows.append(base_agg)
    single_identity_metric_rows.append(base_agg)
    scene_metric_rows.extend(base_metric_rows)
    frame_rows.extend(base_frame_rows)
    all_primary_rows.extend(base_rows)
    config_rows.append(
        {
            "schema_version": "stream4d_v99_phase10o_variant_config_v1",
            "phase_id": "v99_phase10o_overlap3_scene_stitch_repair",
            "variant_id": BASE_OVERLAP_VARIANT,
            "family": "baseline",
            "policy": "overlap3_primary_emit_no_scene_stitch",
            "candidate_count": 0,
            "accepted_history_merge_edge_count": 0,
            "chunk_size": CHUNK_SIZE,
            "overlap": OVERLAP,
            "uses_gt_for_prediction": False,
            "uses_future": False,
        }
    )

    for spec in variant_specs:
        variant_id = str(spec["variant_id"])
        mapping, accepted = _one_to_one_mapping(ids, list(spec["candidates"]), variant_id=variant_id)
        mapped_rows = _apply_mapping(primary_rows, variant_id=variant_id, mapping=mapping, policy=str(spec["policy"]))
        single_agg, per_metric, frames = _eval_variant(variant_id, mapped_rows, eval_scope)
        stats = _component_stats(mapping)
        single_agg["metric_composition"] = "local_and_scene_from_same_stitched_ids"
        single_agg["history_candidate_count"] = len(spec["candidates"])
        single_agg["accepted_history_merge_edge_count"] = len(accepted)
        single_agg.update(stats)
        decoupled = _aggregate_decoupled(
            variant_id,
            base_metric_rows,
            per_metric,
            component_stats=stats,
            candidate_count=len(spec["candidates"]),
            accepted_count=len(accepted),
        )
        metric_rows.append(decoupled)
        single_identity_metric_rows.append(single_agg)
        scene_metric_rows.extend(per_metric)
        frame_rows.extend(frames)
        merge_rows.extend(accepted)
        all_primary_rows.extend(mapped_rows)
        config_rows.append(
            {
                "schema_version": "stream4d_v99_phase10o_variant_config_v1",
                "phase_id": "v99_phase10o_overlap3_scene_stitch_repair",
                "variant_id": variant_id,
                "family": spec["family"],
                "policy": spec["policy"],
                "candidate_count": len(spec["candidates"]),
                "accepted_history_merge_edge_count": len(accepted),
                "chunk_size": CHUNK_SIZE,
                "overlap": OVERLAP,
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )

    best_scene = max(metric_rows, key=lambda row: (_num(row.get("MV_AP_scene")), _num(row.get("MV_AP50_scene")), _num(row.get("MV_AP_window"))))
    best_single = max(single_identity_metric_rows, key=lambda row: (_num(row.get("MV_AP_scene")), _num(row.get("MV_AP50_scene")), _num(row.get("MV_AP_window"))))

    holdout_scene_gate = float(phase0["F2_base_holdout_MV_AP_scene"]) + 0.010
    holdout_scene_ap50_gate = float(phase0["F2_base_holdout_MV_AP50_scene"]) + 0.015
    holdout_window_floor = float(phase0["F2_base_holdout_MV_AP_window"]) - 0.003
    strict_local_gate_window = float(phase0["F2_base_holdout_MV_AP_window"]) + 0.005
    strict_local_gate_ap50 = float(phase0["F2_base_holdout_MV_AP50_window"]) + 0.010

    local_gate = _num(best_scene.get("MV_AP_window")) >= strict_local_gate_window and _num(best_scene.get("MV_AP50_window")) >= strict_local_gate_ap50
    scene_gate = (
        _num(best_scene.get("MV_AP_scene")) >= holdout_scene_gate
        and _num(best_scene.get("MV_AP50_scene")) >= holdout_scene_ap50_gate
        and _num(best_scene.get("MV_AP_window")) >= holdout_window_floor
    )
    safety_gate = (
        int(_num(best_scene.get("same_frame_collision_count"), 1)) == 0
        and int(_num(best_scene.get("missing_mask_raster_count"), 1)) == 0
        and not bool(scope.get("source_uses_future", False))
        and not bool(scope.get("source_uses_gt_for_prediction", False))
    )
    overlap_contract_gate = all(int(row["shared_frame_count"]) == OVERLAP for row in scope["overlap_pair_rows"])

    gate_rows = [
        {
            "gate_id": "overlap3_contract_shared_frames",
            "pass": overlap_contract_gate,
            "expected": f"every adjacent chunk pair shared_frame_count={OVERLAP}",
            "observed": sorted({int(row["shared_frame_count"]) for row in scope["overlap_pair_rows"]}),
            "severity": "config_contract",
        },
        {
            "gate_id": "strict_local_holdout_gate",
            "pass": local_gate,
            "expected": f"MV_AP_window>={strict_local_gate_window} and MV_AP50_window>={strict_local_gate_ap50}",
            "observed": f"{best_scene['variant_id']} MV_AP_window={best_scene.get('MV_AP_window')} MV_AP50_window={best_scene.get('MV_AP50_window')}",
            "severity": "method_gate",
        },
        {
            "gate_id": "scene_holdout_gate_vs_F2_holdout",
            "pass": scene_gate,
            "expected": f"MV_AP_scene>={holdout_scene_gate} and MV_AP50_scene>={holdout_scene_ap50_gate} and MV_AP_window>={holdout_window_floor}",
            "observed": f"{best_scene['variant_id']} MV_AP_scene={best_scene.get('MV_AP_scene')} MV_AP50_scene={best_scene.get('MV_AP50_scene')} MV_AP_window={best_scene.get('MV_AP_window')}",
            "severity": "scene_method_gate",
        },
        {
            "gate_id": "safety_no_gt_no_future_no_missing_no_collision",
            "pass": safety_gate,
            "expected": "uses_gt_for_prediction=false; uses_future=false; same_frame_collision_count=0; missing_mask_raster_count=0",
            "observed": f"uses_gt={scope.get('source_uses_gt_for_prediction')} uses_future={scope.get('source_uses_future')} same_frame_collision={best_scene.get('same_frame_collision_count')} missing_mask={best_scene.get('missing_mask_raster_count')}",
            "severity": "safety",
        },
        {
            "gate_id": "formal_claim_allowed_after_repair",
            "pass": False,
            "expected": "fresh pre-registered run and surfel identity chunk-causal proof",
            "observed": "post-final repair; surfel_dependency_proven_chunk_causal=false remains a formal proof blocker",
            "severity": "formal_claim_blocker",
        },
    ]
    failure_rows = [
        {
            "failure_id": row["gate_id"],
            "severity": row["severity"],
            "expected": row["expected"],
            "observed": row["observed"],
            "repair_direction": "If scene gate still fails after overlap3 exact stitch, inspect over/under-merge with per-scene rows and add DA3/D4RT geometric constraints only if they improve MV_AP_scene without local collapse.",
        }
        for row in gate_rows
        if not bool(row["pass"])
    ]

    decision = "GO_OVERLAP3_SCENE_STITCH_METRIC_REPAIRED_FORMAL_REVIEW_REQUIRED" if (scene_gate and safety_gate and overlap_contract_gate) else "NO_GO_OVERLAP3_SCENE_STITCH_REPAIR"
    summary = {
        "schema_version": "stream4d_v99_phase10o_overlap3_scene_stitch_summary_v1",
        "phase_id": "v99_phase10o_overlap3_scene_stitch_repair",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "runtime_sec": (datetime.now() - started).total_seconds(),
        "decision": decision,
        "metric_gate_pass": bool(local_gate and scene_gate and safety_gate and overlap_contract_gate),
        "local_gate_pass": bool(local_gate),
        "scene_gate_pass": bool(scene_gate),
        "safety_gate_pass": bool(safety_gate),
        "overlap_contract_gate_pass": bool(overlap_contract_gate),
        "formal_claim_allowed": False,
        "bug_repaired": "Phase9/Phase10K/L used non-overlap frame_to_chunk scope; Phase10O uses explicit frame_to_chunks for chunk32 overlap3.",
        "chunk_size": CHUNK_SIZE,
        "overlap": OVERLAP,
        "step": STEP,
        "eval_emit_policy": "earliest_chunk_owns_frame_overlap_context_not_double_counted",
        "metric_composition": "MV_AP_window from primary chunk-scoped local rows; MV_AP_scene from overlap3 stitched scene rows",
        "best_variant_id": best_scene["variant_id"],
        "best_MV_AP_window": float(_num(best_scene.get("MV_AP_window"))),
        "best_MV_AP50_window": float(_num(best_scene.get("MV_AP50_window"))),
        "best_MV_AP_scene": float(_num(best_scene.get("MV_AP_scene"))),
        "best_MV_AP50_scene": float(_num(best_scene.get("MV_AP50_scene"))),
        "best_history_merge_count": int(_num(best_scene.get("history_merge_count"))),
        "best_scene_object_count": int(_num(best_scene.get("scene_object_count"))),
        "best_single_identity_variant_id": best_single["variant_id"],
        "best_single_identity_MV_AP_window": float(_num(best_single.get("MV_AP_window"))),
        "best_single_identity_MV_AP_scene": float(_num(best_single.get("MV_AP_scene"))),
        "base_overlap_MV_AP_window": float(_num(base_agg.get("MV_AP_window"))),
        "base_overlap_MV_AP50_window": float(_num(base_agg.get("MV_AP50_window"))),
        "base_overlap_MV_AP_scene": float(_num(base_agg.get("MV_AP_scene"))),
        "base_overlap_MV_AP50_scene": float(_num(base_agg.get("MV_AP50_scene"))),
        "old_phase10l_nonoverlap_MV_AP_scene": old10l.get("holdout_MV_AP_scene"),
        "old_phase10l_nonoverlap_MV_AP_window": old10l.get("holdout_MV_AP_window"),
        "old_phase10n_decision": old10n.get("decision"),
        "F2_base_holdout_MV_AP_window": float(phase0["F2_base_holdout_MV_AP_window"]),
        "F2_base_holdout_MV_AP50_window": float(phase0["F2_base_holdout_MV_AP50_window"]),
        "F2_base_holdout_MV_AP_scene": float(phase0["F2_base_holdout_MV_AP_scene"]),
        "F2_base_holdout_MV_AP50_scene": float(phase0["F2_base_holdout_MV_AP50_scene"]),
        "birth_stats": birth_stats,
        "outputs": {
            "summary": _rel(OUT_DIR / "summary.json"),
            "variant_metric_rows": _rel(OUT_DIR / "variant_metric_rows.csv"),
            "single_identity_metric_rows": _rel(OUT_DIR / "single_identity_metric_rows.csv"),
            "variant_metric_scene_rows": _rel(OUT_DIR / "variant_metric_scene_rows.csv"),
            "variant_gate_rows": _rel(OUT_DIR / "variant_gate_rows.csv"),
            "variant_failure_rows": _rel(OUT_DIR / "variant_failure_rows.csv"),
            "variant_config_rows": _rel(OUT_DIR / "variant_config_rows.csv"),
            "local2history_candidate_rows": _rel(OUT_DIR / "local2history_candidate_rows.csv"),
            "local2history_merge_rows": _rel(OUT_DIR / "local2history_merge_rows.csv"),
            "overlap_pair_rows": _rel(OUT_DIR / "overlap_pair_rows.csv"),
            "mv_object_rows": _rel(OUT_DIR / "mv_object_rows.csv"),
            "mv_object_frame_mask_rows": _rel(OUT_DIR / "mv_object_frame_mask_rows.csv"),
            "frame_eval_rows": _rel(OUT_DIR / "frame_eval_rows.csv"),
        },
    }

    all_candidates: list[dict[str, Any]] = []
    for spec in variant_specs:
        all_candidates.extend(spec["candidates"])
    _write_csv(OUT_DIR / "variant_metric_rows.csv", metric_rows)
    _write_csv(OUT_DIR / "single_identity_metric_rows.csv", single_identity_metric_rows)
    _write_csv(OUT_DIR / "variant_metric_scene_rows.csv", scene_metric_rows)
    _write_csv(OUT_DIR / "variant_gate_rows.csv", gate_rows)
    _write_csv(OUT_DIR / "variant_failure_rows.csv", failure_rows)
    _write_csv(OUT_DIR / "variant_config_rows.csv", config_rows)
    _write_csv(OUT_DIR / "local2history_candidate_rows.csv", all_candidates)
    _write_csv(OUT_DIR / "local2history_merge_rows.csv", merge_rows)
    _write_csv(OUT_DIR / "overlap_pair_rows.csv", scope["overlap_pair_rows"])
    _write_csv(OUT_DIR / "mv_object_rows.csv", object_rows)
    _write_csv(OUT_DIR / "mv_object_frame_mask_rows.csv", all_primary_rows)
    _write_csv(OUT_DIR / "frame_eval_rows.csv", frame_rows)
    _write_json(OUT_DIR / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if summary["metric_gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
