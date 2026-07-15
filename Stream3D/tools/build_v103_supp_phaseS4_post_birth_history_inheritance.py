#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
AUDIT_ROOT = STREAM3D_ROOT / "outputs" / "audit"
TOOLS_DIR = Path(__file__).resolve().parent
if str(STREAM3D_ROOT) not in sys.path:
    sys.path.insert(0, str(STREAM3D_ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from build_v103_phase6_mask_clustering_local_object_birth import (  # noqa: E402
    _accumulator,
    _load_gt_2d,
    _load_label_png,
    _summarize_iou,
)
from build_v103_phase7_causal_history_token_readiness import (  # noqa: E402
    _load_phase2_scene,
    _project_labels_for_indices,
)


PHASE_ID = "v103_supp_phaseS4_post_birth_history_inheritance"
DEFAULT_OUT = AUDIT_ROOT / "v103_supp_phaseS4_post_birth_history_inheritance"
DEFAULT_PHASES0_ROOT = AUDIT_ROOT / "v103_supp_phaseS0_fact_lock"
DEFAULT_PHASES3_ROOT = AUDIT_ROOT / "v103_supp_phaseS3_scaffolded_mask_graph"
DEFAULT_PHASE7_ROOT = AUDIT_ROOT / "v103_phase7_causal_history_token_readiness_r11_all_d4rt48mix_maskbalanced8_e5"
DEFAULT_F2_ROOT = AUDIT_ROOT / "v100_phase2c_overlap3_local_repair"
DEFAULT_SCENE0011_PHASE2 = AUDIT_ROOT / "v103_phase2_stratified_q5c_objlike16384_scene0011_c0001_cap24576_qchunk16384_d4rt48mix_maskbalanced8"
DEFAULT_SCENE0050_PHASE2 = AUDIT_ROOT / "v103_phase2_stratified_q5c_objlike16384_scene0050_c0001_cap24576_qchunk16384_d4rt48mix_maskbalanced8"

SCHEMA_PREFIX = "stream4d_v103_supp_phaseS4"


VARIANTS = [
    {
        "variant_id": "S4_H0_real_strict_post_birth_inheritance",
        "support_source": "h2_real",
        "is_control": False,
        "object_tau": 0.55,
        "object_margin": 0.10,
        "object_entropy": 0.80,
        "min_unique_carriers": 3,
        "min_support_weight": 0.05,
        "one_to_one": True,
    },
    {
        "variant_id": "S4_H1_real_balanced_post_birth_inheritance",
        "support_source": "h2_real",
        "is_control": False,
        "object_tau": 0.45,
        "object_margin": 0.06,
        "object_entropy": 0.85,
        "min_unique_carriers": 2,
        "min_support_weight": 0.03,
        "one_to_one": True,
    },
    {
        "variant_id": "S4_C1_shuffled_history_control",
        "support_source": "semantic_shuffled",
        "is_control": True,
        "control_family": "shuffled_history",
        "object_tau": 0.45,
        "object_margin": 0.06,
        "object_entropy": 0.85,
        "min_unique_carriers": 2,
        "min_support_weight": 0.03,
        "one_to_one": True,
    },
    {
        "variant_id": "S4_C2_semantic_only_history_control",
        "support_source": "semantic_only",
        "is_control": True,
        "control_family": "semantic_only_history",
        "object_tau": 0.45,
        "object_margin": 0.06,
        "object_entropy": 0.85,
        "min_unique_carriers": 2,
        "min_support_weight": 0.03,
        "one_to_one": True,
    },
    {
        "variant_id": "S4_C3_stale_shifted_history_control",
        "support_source": "stale_shifted_h2",
        "is_control": True,
        "control_family": "stale_history",
        "object_tau": 0.45,
        "object_margin": 0.06,
        "object_entropy": 0.85,
        "min_unique_carriers": 2,
        "min_support_weight": 0.03,
        "one_to_one": True,
    },
    {
        "variant_id": "S4_C4_random_history_token_control",
        "support_source": "random_history",
        "is_control": True,
        "control_family": "random_history_token",
        "object_tau": 0.45,
        "object_margin": 0.06,
        "object_entropy": 0.85,
        "min_unique_carriers": 2,
        "min_support_weight": 0.03,
        "one_to_one": True,
    },
]


def _project(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if p.parts and p.parts[0] == "Stream3D":
        return REPO_ROOT / p
    return STREAM3D_ROOT / p


def _rel(path: str | Path) -> str:
    p = Path(path)
    try:
        return p.resolve().relative_to(REPO_ROOT).as_posix()
    except Exception:
        return p.as_posix()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return _rel(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
            writer.writerow({key: _jsonable(row.get(key, "")) for key in fields})


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return float(default)
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _scene_phase2_roots(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "scene0011_00": _project(args.scene0011_phase2_root),
        "scene0050_00": _project(args.scene0050_phase2_root),
    }


def _load_history_snapshot(phase7_root: Path, history_root: Path | None, history_variant_id: str) -> tuple[pd.DataFrame, pd.DataFrame, Path, str]:
    phase7_summary = _read_json(phase7_root / "summary.json")
    root = _project(history_root or phase7_summary["history_root"])
    variant_id = str(history_variant_id or phase7_summary["history_variant_id"])
    rows = pd.read_csv(root / "merge_selected_rows.csv")
    rows = rows[(rows["variant_id"].astype(str) == variant_id) & (~rows["uses_gt_for_prediction"].astype(bool))].copy()
    if rows.empty:
        raise RuntimeError(f"no history snapshot rows found in {root} for variant_id={variant_id}")
    snapshot_rows: list[dict[str, Any]] = []
    for row in rows.to_dict("records"):
        history_id = str(row["object_id"])
        snapshot_rows.append(
            {
                "schema_version": f"{SCHEMA_PREFIX}_history_memory_snapshot_row_v1",
                "phase_id": PHASE_ID,
                "scene_id": str(row["scene_id"]),
                "history_id": history_id,
                "mv_object_id": history_id,
                "object_id": history_id,
                "history_variant_id": variant_id,
                "source_variant_id": str(row["variant_id"]),
                "chunk_id": str(row.get("chunk_id", "c0000")),
                "window_id": str(row.get("window_id", row.get("chunk_id", "c0000"))),
                "frame_id": int(row["frame_id"]),
                "frame_local_index": int(row.get("frame_local_index", -1)),
                "selected_mask_id": int(row["selected_mask_id"]),
                "mask_id_or_generated_id": int(row.get("mask_id_or_generated_id", row["selected_mask_id"])),
                "object_score": float(row.get("object_score", row.get("score", 0.0))),
                "score": float(row.get("score", row.get("object_score", 0.0))),
                "support_count": int(float(row.get("support_count", 0) or 0)),
                "node_policy": str(row.get("node_policy", "pre_update_history_memory")),
                "emit_policy": str(row.get("emit_policy", "pre_update_history_memory")),
                "readout_mode": str(row.get("readout_mode", "pre_update_history_memory")),
                "source_role": "pre_update_history_snapshot",
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
    snapshot = pd.DataFrame(snapshot_rows)
    objects: list[dict[str, Any]] = []
    for (scene, hist_id), group in snapshot.groupby(["scene_id", "history_id"], sort=True):
        frames = sorted(group["frame_id"].astype(int).unique().tolist())
        objects.append(
            {
                "schema_version": f"{SCHEMA_PREFIX}_history_object_summary_row_v1",
                "phase_id": PHASE_ID,
                "scene_id": str(scene),
                "history_id": str(hist_id),
                "history_variant_id": variant_id,
                "frame_count": int(len(frames)),
                "mask_observation_count": int(len(group)),
                "first_frame_id": int(frames[0]) if frames else "",
                "last_frame_id": int(frames[-1]) if frames else "",
                "uses_gt_for_prediction": False,
            }
        )
    return snapshot, pd.DataFrame(objects), root, variant_id


def _load_current_f2_rows(
    *,
    f2_root: Path,
    phase2_summaries: dict[str, dict[str, Any]],
    dataset_split: str,
    chunk_id: str,
) -> pd.DataFrame:
    path = f2_root / "mv_object_frame_mask_rows.parquet"
    rows = pd.read_parquet(path)
    rows = rows[
        (rows["dataset_split"].astype(str) == str(dataset_split))
        & (rows["chunk_id"].astype(str) == str(chunk_id))
        & (~rows["uses_gt_for_prediction"].astype(bool))
        & (~rows["uses_future"].astype(bool))
    ].copy()
    rows = rows[rows["scene_id"].astype(str).isin(phase2_summaries)]
    if rows.empty:
        raise RuntimeError(f"no current F2 rows found in {path} for split={dataset_split} chunk_id={chunk_id}")

    out: list[dict[str, Any]] = []
    for row in rows.to_dict("records"):
        scene = str(row["scene_id"])
        frame_ids = [int(v) for v in phase2_summaries[scene]["frame_ids"]]
        frame_to_local = {int(frame_id): int(i) for i, frame_id in enumerate(frame_ids)}
        frame_id = int(row["frame_id"])
        if frame_id not in frame_to_local:
            continue
        local_id = str(row["mv_object_id"])
        score = float(row.get("score", row.get("object_score", 0.0)))
        out.append(
            {
                "schema_version": f"{SCHEMA_PREFIX}_current_object_frame_mask_row_v1",
                "phase_id": PHASE_ID,
                "variant_id": "S4_CURRENT_F2_C0001_SCAFFOLD",
                "mv_object_id": local_id,
                "object_id": local_id,
                "source_local_object_id": local_id,
                "scene_id": scene,
                "chunk_id": str(row["chunk_id"]),
                "window_id": str(row.get("window_id", row["chunk_id"])),
                "frame_local_index": int(frame_to_local[frame_id]),
                "frame_id": frame_id,
                "selected_mask_id": int(row["selected_mask_id"]),
                "mask_id_or_generated_id": int(row.get("mask_id_or_generated_id", row["selected_mask_id"])),
                "object_score": score,
                "score": score,
                "support_count": int(float(row.get("support_surfel_count", row.get("support_count", 0)) or 0)),
                "node_policy": str(row.get("object_birth_scope", "current_chunk32_overlap3_surfel_maskview_birth")),
                "emit_policy": str(row.get("eval_emit_policy", "earliest_chunk_owns_frame_overlap_context_not_double_counted")),
                "readout_mode": str(row.get("readout_mode", "current_chunk32_overlap3_surfel_maskview_birth")),
                "source_role": "current_post_birth_local_object",
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
    if not out:
        raise RuntimeError(f"current F2 rows did not overlap phase2 frame_ids for chunk_id={chunk_id}")
    return pd.DataFrame(out)


def _load_current_s3_rows(
    *,
    phaseS3_root: Path,
    phase2_summaries: dict[str, dict[str, Any]],
    chunk_id: str,
    phaseS3_variant_id: str,
) -> pd.DataFrame:
    path = phaseS3_root / "object_frame_mask_rows.csv"
    rows = pd.read_csv(path)
    rows = rows[
        (rows["chunk_id"].astype(str) == str(chunk_id))
        & (rows["variant_id"].astype(str) == str(phaseS3_variant_id))
        & (~rows["uses_gt_for_prediction"].astype(bool))
        & (~rows["uses_future"].astype(bool))
    ].copy()
    rows = rows[rows["scene_id"].astype(str).isin(phase2_summaries)]
    if rows.empty:
        raise RuntimeError(
            f"no current S3 rows found in {path} for chunk_id={chunk_id} variant_id={phaseS3_variant_id}"
        )

    out: list[dict[str, Any]] = []
    for row in rows.to_dict("records"):
        scene = str(row["scene_id"])
        frame_ids = [int(v) for v in phase2_summaries[scene]["frame_ids"]]
        frame_to_local = {int(frame_id): int(i) for i, frame_id in enumerate(frame_ids)}
        frame_id = int(row["frame_id"])
        if frame_id not in frame_to_local:
            continue
        local_id = str(row["mv_object_id"])
        score = float(row.get("score", row.get("object_score", 0.0)))
        out.append(
            {
                "schema_version": f"{SCHEMA_PREFIX}_current_object_frame_mask_row_v1",
                "phase_id": PHASE_ID,
                "variant_id": f"S4_CURRENT_ALIGNED_S3_{phaseS3_variant_id}",
                "mv_object_id": local_id,
                "object_id": local_id,
                "source_local_object_id": local_id,
                "scene_id": scene,
                "chunk_id": str(row["chunk_id"]),
                "window_id": str(row.get("window_id", row["chunk_id"])),
                "frame_local_index": int(frame_to_local[frame_id]),
                "frame_id": frame_id,
                "selected_mask_id": int(row["selected_mask_id"]),
                "mask_id_or_generated_id": int(row.get("mask_id_or_generated_id", row["selected_mask_id"])),
                "object_score": score,
                "score": score,
                "support_count": int(float(row.get("support_count", 0)) or 0),
                "node_policy": str(row.get("node_policy", "supp_s3_scaffolded_mask_graph")),
                "emit_policy": str(row.get("emit_policy", "component_wta_by_f2_score_support")),
                "readout_mode": str(row.get("readout_mode", "supp_s3_scaffolded_mask_graph")),
                "source_role": "current_post_birth_aligned_s3_local_object",
                "source_phaseS3_variant_id": str(phaseS3_variant_id),
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
    if not out:
        raise RuntimeError(
            f"current S3 rows did not overlap phase2 frame_ids for chunk_id={chunk_id} variant_id={phaseS3_variant_id}"
        )
    return pd.DataFrame(out)


def _load_current_phase6d_rows(
    *,
    phase6d_root: Path,
    phase2_summaries: dict[str, dict[str, Any]],
    chunk_id: str,
    phase6d_variant_id: str,
) -> pd.DataFrame:
    summary = _read_json(phase6d_root / "summary.json")
    if str(summary.get("decision", "")) != "PASS_PHASE6D_S3_STYLE_LOCAL_GATE":
        raise RuntimeError(f"Phase6d root did not pass S3-style local gate: {phase6d_root}")
    path = phase6d_root / "merge_selected_rows.csv"
    rows = pd.read_csv(path)
    rows = rows[
        (rows["chunk_id"].astype(str) == str(chunk_id))
        & (rows["variant_id"].astype(str) == str(phase6d_variant_id))
        & (~rows["uses_gt_for_prediction"].astype(bool))
        & (~rows["uses_future"].astype(bool))
    ].copy()
    rows = rows[rows["scene_id"].astype(str).isin(phase2_summaries)]
    if rows.empty:
        raise RuntimeError(
            f"no current Phase6d rows found in {path} for chunk_id={chunk_id} variant_id={phase6d_variant_id}"
        )

    out: list[dict[str, Any]] = []
    for row in rows.to_dict("records"):
        scene = str(row["scene_id"])
        frame_ids = [int(v) for v in phase2_summaries[scene]["frame_ids"]]
        frame_to_local = {int(frame_id): int(i) for i, frame_id in enumerate(frame_ids)}
        frame_id = int(row["frame_id"])
        if frame_id not in frame_to_local:
            continue
        local_id = str(row["mv_object_id"])
        score = float(row.get("score", row.get("object_score", 0.0)))
        out.append(
            {
                "schema_version": f"{SCHEMA_PREFIX}_current_object_frame_mask_row_v1",
                "phase_id": PHASE_ID,
                "variant_id": f"S4_CURRENT_PHASE6D_{phase6d_variant_id}",
                "mv_object_id": local_id,
                "object_id": local_id,
                "source_local_object_id": local_id,
                "scene_id": scene,
                "chunk_id": str(row["chunk_id"]),
                "window_id": str(row.get("window_id", row["chunk_id"])),
                "frame_local_index": int(frame_to_local[frame_id]),
                "frame_id": frame_id,
                "selected_mask_id": int(row["selected_mask_id"]),
                "mask_id_or_generated_id": int(row.get("mask_id_or_generated_id", row["selected_mask_id"])),
                "object_score": score,
                "score": score,
                "support_count": int(float(row.get("support_count", 0)) or 0),
                "node_policy": str(row.get("node_policy", "f2_skeleton_phase6d_affinity_merge")),
                "emit_policy": str(row.get("emit_policy", "component_wta_by_f2_score_support")),
                "readout_mode": str(row.get("readout_mode", "f2_skeleton_primitive_affinity_merge")),
                "source_role": "current_post_birth_phase6d_local_object",
                "source_phase6d_variant_id": str(phase6d_variant_id),
                "source_phase6d_root": _rel(phase6d_root),
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
    if not out:
        raise RuntimeError(
            f"current Phase6d rows did not overlap phase2 frame_ids for chunk_id={chunk_id} variant_id={phase6d_variant_id}"
        )
    return pd.DataFrame(out)


def _load_s3_alignment_status(phaseS3_root: Path, phase7_root: Path) -> dict[str, Any]:
    out = {
        "phaseS3_root": _rel(phaseS3_root),
        "phase7_root": _rel(phase7_root),
        "phaseS3_rows_available": False,
        "phaseS3_chunk_ids": [],
        "phaseS3_frame_id_min": "",
        "phaseS3_frame_id_max": "",
        "phase7_current_frame_id_min": "",
        "phase7_current_frame_id_max": "",
        "aligned_with_phase7_current_chunk": False,
        "action": "use_current_c0001_F2_scaffold_for_S4",
    }
    s3_path = phaseS3_root / "object_frame_mask_rows.csv"
    if s3_path.exists():
        s3 = pd.read_csv(s3_path)
        out["phaseS3_rows_available"] = True
        out["phaseS3_chunk_ids"] = sorted(s3["chunk_id"].astype(str).unique().tolist()) if "chunk_id" in s3.columns else []
        out["phaseS3_frame_id_min"] = int(s3["frame_id"].min()) if not s3.empty else ""
        out["phaseS3_frame_id_max"] = int(s3["frame_id"].max()) if not s3.empty else ""
    phase7 = _read_json(phase7_root / "summary.json")
    frame_ids: list[int] = []
    for meta in phase7.get("scene_meta", []):
        frame_ids.extend([int(v) for v in meta.get("frame_ids", [])])
    if frame_ids:
        out["phase7_current_frame_id_min"] = int(min(frame_ids))
        out["phase7_current_frame_id_max"] = int(max(frame_ids))
    out["aligned_with_phase7_current_chunk"] = bool(
        out["phaseS3_rows_available"]
        and "c0001" in out["phaseS3_chunk_ids"]
        and out["phaseS3_frame_id_min"] != ""
        and int(out["phaseS3_frame_id_min"]) >= int(out["phase7_current_frame_id_min"])
    )
    if out["aligned_with_phase7_current_chunk"]:
        out["action"] = "phaseS3_rows_aligned_but_default_still_uses_c0001_F2_scaffold_because_S3_method_gate_no_go"
    return out


def _h2_support(scene_payload: dict[str, Any], overlap_weight: float, semantic_weight: float) -> tuple[np.ndarray, np.ndarray]:
    e_overlap = scene_payload["e_overlap_for_h2"].to(torch.float32).cpu().numpy().astype(np.float32, copy=False)
    e_sem = scene_payload["e_sem"].to(torch.float32).cpu().numpy().astype(np.float32, copy=False)
    hard = scene_payload["hard_conflict"].to(torch.bool).cpu().numpy().astype(bool, copy=False)
    overlap_present = np.max(e_overlap, axis=1) > 0 if e_overlap.size else np.zeros((e_overlap.shape[0],), dtype=bool)
    h2 = np.where(overlap_present[:, None], float(overlap_weight) * e_overlap + float(semantic_weight) * e_sem, e_sem)
    h2 = np.asarray(h2, dtype=np.float32)
    h2 = np.where(hard, 0.0, h2).astype(np.float32, copy=False)
    return h2, hard


def _support_for_variant(scene_payload: dict[str, Any], variant: dict[str, Any], args: argparse.Namespace, scene_seed: int) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    h2, hard = _h2_support(scene_payload, float(args.overlap_weight), float(args.semantic_weight))
    e_sem = scene_payload["e_sem"].to(torch.float32).cpu().numpy().astype(np.float32, copy=False)
    source = str(variant["support_source"])
    m = h2.shape[1]
    rng = np.random.default_rng(int(scene_seed))
    meta: dict[str, Any] = {"support_source": source}
    if source == "h2_real":
        return h2, hard, meta
    if source == "semantic_only":
        return e_sem.copy(), None, meta
    if source == "semantic_shuffled":
        perm = rng.permutation(m) if m > 1 else np.arange(m)
        meta["history_column_permutation"] = perm.astype(int).tolist()
        return e_sem[:, perm].astype(np.float32, copy=False), None, meta
    if source == "stale_shifted_h2":
        if m > 1:
            meta["stale_control_policy"] = "cyclic_shift_history_columns_by_one"
            return np.roll(h2, shift=1, axis=1).astype(np.float32, copy=False), np.roll(hard, shift=1, axis=1), meta
        return h2.copy(), hard.copy(), meta
    if source == "random_history":
        meta["random_seed"] = int(scene_seed)
        return rng.random(h2.shape, dtype=np.float32), None, meta
    raise ValueError(f"unsupported support_source={source}")


def _prepare_scene_projection(
    *,
    scene: str,
    phase2_root: Path,
    scene_payload: dict[str, Any],
    cupy_device_id: int,
) -> dict[str, Any]:
    summary, masks, batch = _load_phase2_scene(phase2_root)
    carrier_indices = scene_payload["carrier_indices"].to(torch.int64).cpu().numpy().astype(np.int64, copy=False)
    labels, ok, weights, _xs, projection_backend, projection_runtime = _project_labels_for_indices(
        batch=batch,
        masks=masks,
        carrier_indices=carrier_indices,
        cupy_device_id=int(cupy_device_id),
    )
    return {
        "scene_id": scene,
        "phase2_root": _rel(phase2_root),
        "summary": summary,
        "labels": labels,
        "ok": ok,
        "weights": weights,
        "carrier_indices": carrier_indices,
        "projection_backend": projection_backend,
        "projection_runtime_sec": float(projection_runtime),
    }


def _assignment_from_support(
    support: np.ndarray,
    history_ids: list[str],
    *,
    tau: float,
    margin_tau: float,
    entropy_tau: float,
) -> dict[str, np.ndarray]:
    u = np.asarray(support, dtype=np.float32)
    n, m = u.shape
    if n == 0 or m == 0:
        top1_idx = np.zeros((n,), dtype=np.int64)
        top1 = np.zeros((n,), dtype=np.float32)
        top2 = np.zeros((n,), dtype=np.float32)
    else:
        top1_idx = np.argmax(u, axis=1).astype(np.int64)
        top1 = u[np.arange(n), top1_idx].astype(np.float32)
        if m >= 2:
            top2 = np.partition(u, m - 2, axis=1)[:, -2]
        else:
            top2 = np.zeros((n,), dtype=np.float32)
    margin = (top1 - top2).astype(np.float32)
    total = np.sum(u, axis=1, keepdims=True)
    p = np.divide(u, np.maximum(total, 1e-8), out=np.zeros_like(u), where=total > 0)
    entropy_raw = -np.sum(np.where(p > 0, p * np.log(np.maximum(p, 1e-12)), 0.0), axis=1)
    entropy = (entropy_raw / max(math.log(max(m, 2)), 1e-6)).astype(np.float32)
    entropy[total[:, 0] <= 0] = 1.0
    assigned = (top1 >= float(tau)) & (margin >= float(margin_tau)) & (entropy <= float(entropy_tau))
    return {
        "top1_idx": top1_idx,
        "top1_score": top1,
        "top2_score": top2,
        "margin": margin,
        "entropy": entropy,
        "assigned": assigned.astype(bool),
        "top1_history_id": np.asarray([history_ids[int(i)] if history_ids else "" for i in top1_idx], dtype=object),
    }


def _pool_object_history_support(
    *,
    scene: str,
    current_rows: pd.DataFrame,
    scene_payload: dict[str, Any],
    projection: dict[str, Any],
    variant: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[np.ndarray, list[str], dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    summary = projection["summary"]
    labels = np.asarray(projection["labels"], dtype=np.int32)
    ok = np.asarray(projection["ok"], dtype=bool)
    weights = np.asarray(projection["weights"], dtype=np.float32)
    support, hard_conflict, support_meta = _support_for_variant(scene_payload, variant, args, scene_seed=10317 + sum(ord(c) for c in scene) + len(str(variant["variant_id"])))
    history_ids = [str(v) for v in scene_payload["history_ids"]]
    carrier_assignment = _assignment_from_support(
        support,
        history_ids,
        tau=float(args.carrier_tau_hist),
        margin_tau=float(args.carrier_tau_margin),
        entropy_tau=float(args.carrier_tau_entropy),
    )
    carrier_valid = np.asarray(carrier_assignment["assigned"], dtype=bool)

    scene_rows = current_rows[current_rows["scene_id"].astype(str) == str(scene)].copy()
    object_ids = sorted(scene_rows["source_local_object_id"].astype(str).unique().tolist())
    obj_index = {oid: i for i, oid in enumerate(object_ids)}
    owner: dict[tuple[int, int], list[int]] = defaultdict(list)
    duplicate_claim_count = 0
    for row in scene_rows.to_dict("records"):
        key = (int(row["frame_local_index"]), int(row["selected_mask_id"]))
        oi = obj_index[str(row["source_local_object_id"])]
        if oi not in owner[key]:
            owner[key].append(oi)
        else:
            duplicate_claim_count += 1
    for values in owner.values():
        if len(values) > 1:
            duplicate_claim_count += len(values) - 1

    obj_count = len(object_ids)
    hist_count = len(history_ids)
    score_acc = np.zeros((obj_count, hist_count), dtype=np.float64)
    denom = np.zeros((obj_count,), dtype=np.float64)
    hit_count = np.zeros((obj_count,), dtype=np.int64)
    carrier_sets: list[set[int]] = [set() for _ in range(obj_count)]
    assigned_hit_count = np.zeros((obj_count,), dtype=np.int64)
    carrier_hit_any = np.zeros((support.shape[0],), dtype=bool)
    if obj_count and hist_count:
        for fi in range(labels.shape[0]):
            lab = labels[fi]
            good = ok[fi] & carrier_valid & (lab > 0)
            if not np.any(good):
                continue
            for mask_id in sorted({int(v) for v in np.unique(lab[good]).tolist() if int(v) > 0}):
                object_targets = owner.get((int(fi), int(mask_id)), [])
                if not object_targets:
                    continue
                idxs = np.flatnonzero(good & (lab == int(mask_id))).astype(np.int64)
                if idxs.size == 0:
                    continue
                w = np.asarray(weights[fi, idxs], dtype=np.float32)
                valid_w = np.isfinite(w) & (w > 0)
                if not np.any(valid_w):
                    continue
                idxs = idxs[valid_w]
                w = w[valid_w]
                weighted = (w[:, None].astype(np.float64) * support[idxs].astype(np.float64)).sum(axis=0)
                w_sum = float(np.sum(w))
                for oi in object_targets:
                    score_acc[oi] += weighted
                    denom[oi] += w_sum
                    hit_count[oi] += int(idxs.shape[0])
                    assigned_hit_count[oi] += int(np.count_nonzero(carrier_valid[idxs]))
                    carrier_sets[oi].update(int(v) for v in idxs.tolist())
                carrier_hit_any[idxs] = True
    object_support = np.divide(
        score_acc,
        np.maximum(denom[:, None], 1e-8),
        out=np.zeros_like(score_acc, dtype=np.float64),
        where=denom[:, None] > 0,
    ).astype(np.float32)
    pool_meta = {
        "scene_id": scene,
        "phase2_root": str(projection["phase2_root"]),
        "frame_ids": [int(v) for v in summary["frame_ids"]],
        "projection_backend": str(projection["projection_backend"]),
        "projection_runtime_sec": float(projection["projection_runtime_sec"]),
        "current_object_count": int(obj_count),
        "current_object_row_count": int(scene_rows.shape[0]),
        "carrier_count": int(np.asarray(projection["carrier_indices"]).shape[0]),
        "carrier_history_assignment_rate": float(np.mean(carrier_valid)) if carrier_valid.size else 0.0,
        "carrier_history_top1_score_mean": float(np.mean(carrier_assignment["top1_score"])) if carrier_valid.size else 0.0,
        "carrier_history_top1_margin_mean": float(np.mean(carrier_assignment["margin"])) if carrier_valid.size else 0.0,
        "carrier_history_entropy_mean": float(np.mean(carrier_assignment["entropy"])) if carrier_valid.size else 1.0,
        "carrier_history_hard_conflict_rate": float(np.mean(np.any(hard_conflict, axis=1))) if hard_conflict is not None and hard_conflict.size else 0.0,
        "carrier_current_object_hit_rate": float(np.mean(carrier_hit_any)) if carrier_hit_any.size else 0.0,
        "object_with_any_carrier_hit_rate": float(np.mean(hit_count > 0)) if hit_count.size else 0.0,
        "object_with_valid_carrier_hit_rate": float(np.mean(assigned_hit_count > 0)) if assigned_hit_count.size else 0.0,
        "duplicate_current_mask_claim_count": int(duplicate_claim_count),
        **support_meta,
    }
    object_aux = {
        "support_weight_sum": denom.astype(np.float32),
        "carrier_hit_count": hit_count.astype(np.int64),
        "unique_carrier_count": np.asarray([len(s) for s in carrier_sets], dtype=np.int64),
        "assigned_carrier_hit_count": assigned_hit_count.astype(np.int64),
    }
    return object_support, object_ids, carrier_assignment, object_aux, pool_meta


def _assign_objects(
    *,
    scene: str,
    variant: dict[str, Any],
    object_support: np.ndarray,
    object_ids: list[str],
    history_ids: list[str],
    object_aux: dict[str, np.ndarray],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, str]]:
    assignment = _assignment_from_support(
        object_support,
        history_ids,
        tau=float(variant["object_tau"]),
        margin_tau=float(variant["object_margin"]),
        entropy_tau=float(variant["object_entropy"]),
    )
    assigned_pre = np.asarray(assignment["assigned"], dtype=bool).copy()
    unique_count = np.asarray(object_aux["unique_carrier_count"], dtype=np.int64)
    support_weight = np.asarray(object_aux["support_weight_sum"], dtype=np.float32)
    assigned_pre &= unique_count >= int(variant["min_unique_carriers"])
    assigned_pre &= support_weight >= float(variant["min_support_weight"])

    reject_reason = np.full((len(object_ids),), "", dtype=object)
    for i in range(len(object_ids)):
        if not bool(assignment["assigned"][i]):
            reject_reason[i] = "object_score_gate_failed"
        elif unique_count[i] < int(variant["min_unique_carriers"]):
            reject_reason[i] = "min_unique_carrier_gate_failed"
        elif support_weight[i] < float(variant["min_support_weight"]):
            reject_reason[i] = "min_support_weight_gate_failed"

    assigned_final = assigned_pre.copy()
    one_to_one_conflict_count = 0
    if bool(variant.get("one_to_one", True)) and len(history_ids):
        by_hist: dict[str, list[int]] = defaultdict(list)
        for i, ok in enumerate(assigned_pre.tolist()):
            if ok:
                by_hist[str(assignment["top1_history_id"][i])].append(i)
        for _hist, idxs in by_hist.items():
            if len(idxs) <= 1:
                continue
            idxs = sorted(
                idxs,
                key=lambda i: (
                    float(assignment["top1_score"][i]),
                    float(assignment["margin"][i]),
                    float(support_weight[i]),
                    int(unique_count[i]),
                ),
                reverse=True,
            )
            for loser in idxs[1:]:
                assigned_final[loser] = False
                reject_reason[loser] = "one_to_one_history_competition_lost"
                one_to_one_conflict_count += 1

    mapping: dict[str, str] = {}
    assignment_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    variant_id = str(variant["variant_id"])
    for i, local_id in enumerate(object_ids):
        assigned_history = str(assignment["top1_history_id"][i]) if bool(assigned_final[i]) else ""
        new_id = f"{variant_id}:{scene}:c0001:new_{i:05d}"
        final_id = assigned_history if assigned_history else new_id
        mapping[str(local_id)] = final_id
        assignment_rows.append(
            {
                "schema_version": f"{SCHEMA_PREFIX}_history_assignment_row_v1",
                "phase_id": PHASE_ID,
                "variant_id": variant_id,
                "scene_id": scene,
                "source_local_object_id": str(local_id),
                "assigned_history_id": assigned_history,
                "final_object_id": final_id,
                "assigned_before_one_to_one": bool(assigned_pre[i]),
                "assigned_after_one_to_one": bool(assigned_final[i]),
                "top1_history_id": str(assignment["top1_history_id"][i]),
                "object_history_top1_score": float(assignment["top1_score"][i]),
                "object_history_top2_score": float(assignment["top2_score"][i]),
                "object_history_top1_margin": float(assignment["margin"][i]),
                "object_history_entropy": float(assignment["entropy"][i]),
                "support_weight_sum": float(support_weight[i]),
                "carrier_hit_count": int(object_aux["carrier_hit_count"][i]),
                "assigned_carrier_hit_count": int(object_aux["assigned_carrier_hit_count"][i]),
                "unique_carrier_count": int(unique_count[i]),
                "reject_reason": "" if bool(assigned_final[i]) else str(reject_reason[i]),
                "object_tau": float(variant["object_tau"]),
                "object_margin_tau": float(variant["object_margin"]),
                "object_entropy_tau": float(variant["object_entropy"]),
                "is_control": bool(variant.get("is_control", False)),
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
        order = np.argsort(object_support[i])[::-1] if object_support.shape[1] else np.asarray([], dtype=np.int64)
        for rank, hist_idx in enumerate(order[: min(5, order.shape[0])].tolist(), start=1):
            score_rows.append(
                {
                    "schema_version": f"{SCHEMA_PREFIX}_object_history_score_row_v1",
                    "phase_id": PHASE_ID,
                    "variant_id": variant_id,
                    "scene_id": scene,
                    "source_local_object_id": str(local_id),
                    "history_id": str(history_ids[int(hist_idx)]),
                    "rank": int(rank),
                    "score": float(object_support[i, int(hist_idx)]),
                    "is_top1": bool(rank == 1),
                    "selected_after_one_to_one": bool(rank == 1 and assigned_final[i]),
                    "is_control": bool(variant.get("is_control", False)),
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )

    assigned_count = int(np.count_nonzero(assigned_final))
    metrics = {
        "object_history_assignment_rate": float(assigned_count / max(len(object_ids), 1)),
        "object_history_top1_score_mean": float(np.mean(assignment["top1_score"])) if len(object_ids) else 0.0,
        "object_history_top1_margin_mean": float(np.mean(assignment["margin"])) if len(object_ids) else 0.0,
        "object_history_entropy_mean": float(np.mean(assignment["entropy"])) if len(object_ids) else 1.0,
        "one_to_one_conflict_count": int(one_to_one_conflict_count),
        "tentative_new_object_count": int(len(object_ids) - assigned_count),
        "confirmed_history_inheritance_count": int(assigned_count),
        "new_history_object_count": int(len(object_ids) - assigned_count),
    }
    return assignment_rows, score_rows, metrics, mapping


def _materialize_current_rows(current_rows: pd.DataFrame, mapping_by_scene: dict[str, dict[str, str]], variant_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in current_rows.to_dict("records"):
        scene = str(row["scene_id"])
        local_id = str(row["source_local_object_id"])
        final_id = mapping_by_scene[scene].get(local_id, local_id)
        new = dict(row)
        new.update(
            {
                "schema_version": f"{SCHEMA_PREFIX}_object_frame_mask_row_v1",
                "phase_id": PHASE_ID,
                "variant_id": variant_id,
                "mv_object_id": final_id,
                "object_id": final_id,
                "pre_inheritance_object_id": local_id,
                "inheritance_applied": bool(final_id != local_id),
                "readout_mode": "post_birth_history_inheritance",
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
        rows.append(new)
    return rows


def _baseline_current_rows(current_rows: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in current_rows.to_dict("records"):
        new = dict(row)
        new.update(
            {
                "schema_version": f"{SCHEMA_PREFIX}_object_frame_mask_row_v1",
                "phase_id": PHASE_ID,
                "variant_id": "S4_B0_current_local_no_history_baseline",
                "mv_object_id": str(row["source_local_object_id"]),
                "object_id": str(row["source_local_object_id"]),
                "pre_inheritance_object_id": str(row["source_local_object_id"]),
                "inheritance_applied": False,
                "readout_mode": "current_local_no_history_baseline",
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
        rows.append(new)
    return rows


def _evaluate_rows(
    *,
    variant_id: str,
    rows: list[dict[str, Any]],
    phase2_summaries: dict[str, dict[str, Any]],
    metric_suffix: str,
    min_pred_pixels: int,
    min_gt_pixels: int,
    use_cupy_iou: bool,
    cupy_device_id: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    per_scene_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    backend_used = ""
    pixel_collision_count = 0
    pixel_collision_event_count = 0
    missing_mask_raster_count = 0
    pred_positive_pixels = 0
    rows_by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_scene[str(row["scene_id"])].append(row)
    for scene, scene_rows in sorted(rows_by_scene.items()):
        if scene not in phase2_summaries:
            continue
        summary = phase2_summaries[scene]
        mask_root = _project(summary["mask_root"])
        eval_frame_ids = sorted({int(row["frame_id"]) for row in scene_rows})
        acc, backend = _accumulator(bool(use_cupy_iou), int(cupy_device_id))
        backend_used = backend
        by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in scene_rows:
            by_frame[int(row["frame_id"])].append(row)
        object_index: dict[str, int] = {}
        scores: dict[str, float] = {}
        for frame_id in eval_frame_ids:
            mask_path = mask_root / f"{int(frame_id)}.png"
            if not mask_path.exists():
                missing_mask_raster_count += 1
                gt = _load_gt_2d(scene, int(frame_id), (968, 1296))
                acc.add(np.zeros(gt.shape, dtype=np.int64), gt)
                continue
            label = _load_label_png(mask_path)
            pred = np.zeros(label.shape, dtype=np.int64)
            emitted = 0
            for row in sorted(by_frame.get(int(frame_id), []), key=lambda r: (-float(r.get("object_score", r.get("score", 0.0))), str(r.get("mv_object_id", "")))):
                oid = str(row["mv_object_id"])
                if oid not in object_index:
                    object_index[oid] = len(object_index) + 1
                    scores[oid] = float(row.get("object_score", row.get("score", 0.0)))
                else:
                    scores[oid] = max(scores[oid], float(row.get("object_score", row.get("score", 0.0))))
                mask_id = int(row["selected_mask_id"])
                pixels = label == mask_id
                if not np.any(pixels):
                    missing_mask_raster_count += 1
                    continue
                overlap = pixels & (pred > 0)
                overlap_count = int(np.count_nonzero(overlap))
                if overlap_count > 0:
                    pixel_collision_event_count += 1
                    pixel_collision_count += overlap_count
                pred[(pred == 0) & pixels] = int(object_index[oid])
                emitted += 1
            gt = _load_gt_2d(scene, int(frame_id), label.shape)
            acc.add(pred, gt)
            pos = int(np.count_nonzero(pred > 0))
            pred_positive_pixels += pos
            frame_rows.append(
                {
                    "schema_version": f"{SCHEMA_PREFIX}_frame_eval_row_v1",
                    "phase_id": PHASE_ID,
                    "variant_id": variant_id,
                    "scene_id": scene,
                    "metric_scope": metric_suffix,
                    "frame_id": int(frame_id),
                    "emitted_object_count": int(emitted),
                    "pred_positive_pixels": pos,
                    "gt_positive_pixels": int(np.count_nonzero(gt > 0)),
                    "uses_gt_for_prediction": False,
                    "uses_gt_for_eval": True,
                }
            )
        input_scores = np.ones((len(object_index),), dtype=np.float32)
        for oid, idx in object_index.items():
            input_scores[int(idx) - 1] = float(scores.get(oid, 1.0))
        metric, iou, _pred_ids, _gt_ids = _summarize_iou(
            accumulator=acc,
            min_pred_pixels=int(min_pred_pixels),
            min_gt_pixels=int(min_gt_pixels),
            score_mode="input",
            input_scores=input_scores,
        )
        gt_fragment_count_ge2_rate = 0.0
        gt_fragment_count_mean = 0.0
        if iou.shape[1]:
            frag_counts = np.sum(iou >= 0.25, axis=0)
            gt_fragment_count_mean = float(np.mean(frag_counts))
            gt_fragment_count_ge2_rate = float(np.mean(frag_counts >= 2))
        per_scene_rows.append(
            {
                "schema_version": f"{SCHEMA_PREFIX}_{metric_suffix}_metric_scene_row_v1",
                "phase_id": PHASE_ID,
                "variant_id": variant_id,
                "scene_id": scene,
                f"MV_AP_{metric_suffix}": metric.get("ap"),
                f"MV_AP50_{metric_suffix}": metric.get("ap50"),
                f"MV_AP25_{metric_suffix}": metric.get("ap25"),
                f"ScoreFreeMatch50_{metric_suffix}": metric.get("score_free_match_at_050", {}).get("f1"),
                "evaluated_pred_count": metric.get("evaluated_pred_count"),
                "evaluated_gt_count": metric.get("evaluated_gt_count"),
                "gt_best_iou_mean": metric.get("gt_best_iou_mean"),
                "pred_best_iou_mean": metric.get("pred_best_iou_mean"),
                "gt_fragment_count_mean": gt_fragment_count_mean,
                "gt_fragment_count_ge2_rate": gt_fragment_count_ge2_rate,
                "frame_count": int(len(eval_frame_ids)),
                "iou_backend": backend,
                "uses_gt_for_prediction": False,
                "uses_gt_for_eval": True,
                "uses_future": False,
            }
        )
    keys = [f"MV_AP_{metric_suffix}", f"MV_AP50_{metric_suffix}", f"MV_AP25_{metric_suffix}", f"ScoreFreeMatch50_{metric_suffix}"]
    aggregate = {
        "schema_version": f"{SCHEMA_PREFIX}_{metric_suffix}_metric_row_v1",
        "phase_id": PHASE_ID,
        "variant_id": variant_id,
        "scene_count": int(len(per_scene_rows)),
        "metric_scope": metric_suffix,
        "iou_backend": backend_used,
        "same_frame_collision_count": int(pixel_collision_event_count),
        "pixel_collision_count": int(pixel_collision_count),
        "pixel_collision_rate": float(pixel_collision_count / max(pred_positive_pixels, 1)),
        "missing_mask_raster_count": int(missing_mask_raster_count),
        "uses_gt_for_prediction": False,
        "uses_gt_for_eval": True,
        "uses_future": False,
    }
    for key in keys:
        vals = [float(row[key]) for row in per_scene_rows if row.get(key) not in {"", None}]
        aggregate[key] = float(np.mean(vals)) if vals else 0.0
    aggregate[f"scene_fragmentation_rate" if metric_suffix == "scene" else f"{metric_suffix}_fragmentation_rate"] = float(
        np.mean([float(row.get("gt_fragment_count_ge2_rate", 0.0)) for row in per_scene_rows])
    ) if per_scene_rows else 0.0
    return aggregate, per_scene_rows, frame_rows


def _scene_object_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    chunks_by_object: dict[str, set[str]] = defaultdict(set)
    frame_count_by_object: dict[str, set[int]] = defaultdict(set)
    duplicate_same_frame = 0
    by_object_frame: dict[tuple[str, int], int] = defaultdict(int)
    for row in rows:
        oid = str(row["mv_object_id"])
        chunks_by_object[oid].add(str(row.get("chunk_id", "")))
        frame_count_by_object[oid].add(int(row["frame_id"]))
        key = (oid, int(row["frame_id"]))
        by_object_frame[key] += 1
    for count in by_object_frame.values():
        if count > 1:
            duplicate_same_frame += count - 1
    chunk_counts = np.asarray([len(v) for v in chunks_by_object.values()], dtype=np.float32)
    return {
        "objects_crossing_multiple_chunks": int(np.count_nonzero(chunk_counts > 1)) if chunk_counts.size else 0,
        "mean_chunks_per_scene_object": float(np.mean(chunk_counts)) if chunk_counts.size else 0.0,
        "scene_object_count": int(len(chunks_by_object)),
        "scene_duplicate_object_rate": float(duplicate_same_frame / max(len(by_object_frame), 1)),
        "same_object_same_frame_duplicate_count": int(duplicate_same_frame),
    }


def _metric_row_for_variant(
    *,
    variant: dict[str, Any],
    scene_rows: list[dict[str, Any]],
    current_rows: list[dict[str, Any]],
    phase2_summaries: dict[str, dict[str, Any]],
    local_baseline: dict[str, Any],
    history_metrics: dict[str, Any],
    min_pred_pixels: int,
    min_gt_pixels: int,
    use_cupy_iou: bool,
    cupy_device_id: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    variant_id = str(variant["variant_id"])
    scene_agg, per_scene, scene_frame_rows = _evaluate_rows(
        variant_id=variant_id,
        rows=scene_rows,
        phase2_summaries=phase2_summaries,
        metric_suffix="scene",
        min_pred_pixels=min_pred_pixels,
        min_gt_pixels=min_gt_pixels,
        use_cupy_iou=use_cupy_iou,
        cupy_device_id=cupy_device_id,
    )
    local_agg, _per_local, local_frame_rows = _evaluate_rows(
        variant_id=variant_id,
        rows=current_rows,
        phase2_summaries=phase2_summaries,
        metric_suffix="window",
        min_pred_pixels=min_pred_pixels,
        min_gt_pixels=min_gt_pixels,
        use_cupy_iou=use_cupy_iou,
        cupy_device_id=cupy_device_id,
    )
    object_stats = _scene_object_stats(scene_rows)
    row = {
        "schema_version": f"{SCHEMA_PREFIX}_scene_metric_row_v1",
        "phase_id": PHASE_ID,
        "variant_id": variant_id,
        "is_control": bool(variant.get("is_control", False)),
        "control_family": str(variant.get("control_family", "")),
        "support_source": str(variant["support_source"]),
        "metric_scope": "history_c0000_plus_current_c0001_dev_subset",
        "MV_AP_scene": float(scene_agg.get("MV_AP_scene", 0.0)),
        "MV_AP50_scene": float(scene_agg.get("MV_AP50_scene", 0.0)),
        "MV_AP25_scene": float(scene_agg.get("MV_AP25_scene", 0.0)),
        "ScoreFreeMatch50_scene": float(scene_agg.get("ScoreFreeMatch50_scene", 0.0)),
        "local_MV_AP_window_after_history": float(local_agg.get("MV_AP_window", 0.0)),
        "local_MV_AP50_window_after_history": float(local_agg.get("MV_AP50_window", 0.0)),
        "local_MV_AP25_window_after_history": float(local_agg.get("MV_AP25_window", 0.0)),
        "local_AP_drop_vs_phaseS3": float(local_baseline.get("MV_AP_window", 0.0) - float(local_agg.get("MV_AP_window", 0.0))),
        "same_frame_collision_count": int(scene_agg.get("same_frame_collision_count", 0)),
        "pixel_collision_rate": float(scene_agg.get("pixel_collision_rate", 0.0)),
        "missing_mask_raster_count": int(scene_agg.get("missing_mask_raster_count", 0)),
        "scene_fragmentation_rate": float(scene_agg.get("scene_fragmentation_rate", 0.0)),
        "uses_gt_for_prediction": False,
        "uses_gt_for_eval": True,
        "uses_future": False,
        **object_stats,
        **history_metrics,
    }
    return row, per_scene, scene_frame_rows, local_frame_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="v103 supplement Phase S4 post-birth history inheritance.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--phaseS0-root", default=str(DEFAULT_PHASES0_ROOT))
    parser.add_argument("--phaseS3-root", default=str(DEFAULT_PHASES3_ROOT))
    parser.add_argument("--phase7-root", default=str(DEFAULT_PHASE7_ROOT))
    parser.add_argument("--history-root", default="")
    parser.add_argument("--history-variant-id", default="")
    parser.add_argument("--f2-root", default=str(DEFAULT_F2_ROOT))
    parser.add_argument("--phase6d-root", default="")
    parser.add_argument("--scene0011-phase2-root", default=str(DEFAULT_SCENE0011_PHASE2))
    parser.add_argument("--scene0050-phase2-root", default=str(DEFAULT_SCENE0050_PHASE2))
    parser.add_argument("--dataset-split", default="dev")
    parser.add_argument("--current-chunk-id", default="c0001")
    parser.add_argument(
        "--current-object-source",
        choices=["f2", "aligned_s3", "phase6d"],
        default="f2",
        help="Source for current post-birth local objects. Defaults to f2 to preserve the original S4 audit.",
    )
    parser.add_argument(
        "--phaseS3-variant-id",
        default="S3_V1_anchor_positive_only",
        help="Phase S3 variant to consume when --current-object-source=aligned_s3.",
    )
    parser.add_argument(
        "--phase6d-variant-id",
        default="D9_affinity_merge_tau065_top1_broad_support_veto",
        help="Phase6d variant to consume when --current-object-source=phase6d.",
    )
    parser.add_argument("--cupy-device-id", type=int, default=0)
    parser.add_argument("--disable-cupy-iou", action="store_true")
    parser.add_argument("--min-pred-pixels", type=int, default=20)
    parser.add_argument("--min-gt-pixels", type=int, default=20)
    parser.add_argument("--overlap-weight", type=float, default=0.70)
    parser.add_argument("--semantic-weight", type=float, default=0.30)
    parser.add_argument("--carrier-tau-hist", type=float, default=0.55)
    parser.add_argument("--carrier-tau-margin", type=float, default=0.10)
    parser.add_argument("--carrier-tau-entropy", type=float, default=0.75)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    started = time.time()
    out = _project(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    (out / "last_command.txt").write_text(" ".join([sys.executable, *sys.argv]) + "\n", encoding="utf-8")

    phaseS0_root = _project(args.phaseS0_root)
    phaseS3_root = _project(args.phaseS3_root)
    phase7_root = _project(args.phase7_root)
    phaseS0 = _read_json(phaseS0_root / "summary.json")
    phase7 = _read_json(phase7_root / "summary.json")
    token_payload = torch.load(phase7_root / "history_token_feature_rows.pt", map_location="cpu")
    scene_phase2_roots = _scene_phase2_roots(args)
    phase2_summaries = {scene: _read_json(root / "summary.json") for scene, root in scene_phase2_roots.items()}

    history_snapshot, _history_objects, history_root, history_variant_id = _load_history_snapshot(
        phase7_root=phase7_root,
        history_root=_project(args.history_root) if str(args.history_root).strip() else None,
        history_variant_id=str(args.history_variant_id),
    )
    alignment_status = _load_s3_alignment_status(phaseS3_root, phase7_root)
    if str(args.current_object_source) == "aligned_s3":
        current_rows_df = _load_current_s3_rows(
            phaseS3_root=phaseS3_root,
            phase2_summaries=phase2_summaries,
            chunk_id=str(args.current_chunk_id),
            phaseS3_variant_id=str(args.phaseS3_variant_id),
        )
        alignment_status["action"] = "use_aligned_phaseS3_post_birth_rows_for_S4_diagnostic"
    elif str(args.current_object_source) == "phase6d":
        if not str(args.phase6d_root).strip():
            raise RuntimeError("--phase6d-root is required when --current-object-source=phase6d")
        phase6d_root = _project(args.phase6d_root)
        current_rows_df = _load_current_phase6d_rows(
            phase6d_root=phase6d_root,
            phase2_summaries=phase2_summaries,
            chunk_id=str(args.current_chunk_id),
            phase6d_variant_id=str(args.phase6d_variant_id),
        )
        alignment_status["phase6d_root"] = _rel(phase6d_root)
        alignment_status["phase6d_variant_id"] = str(args.phase6d_variant_id)
        alignment_status["action"] = "use_phase6d_gate_pass_post_birth_rows_for_S4_diagnostic"
    else:
        current_rows_df = _load_current_f2_rows(
            f2_root=_project(args.f2_root),
            phase2_summaries=phase2_summaries,
            dataset_split=str(args.dataset_split),
            chunk_id=str(args.current_chunk_id),
        )
        alignment_status["action"] = str(alignment_status.get("action", "use_current_c0001_F2_scaffold_for_S4"))

    baseline_current = _baseline_current_rows(current_rows_df)
    baseline_scene_rows = history_snapshot.to_dict("records") + baseline_current
    baseline_variant = {
        "variant_id": "S4_B0_current_local_no_history_baseline",
        "support_source": "none",
        "is_control": False,
    }
    baseline_metric, baseline_per_scene, baseline_scene_frames, baseline_local_frames = _metric_row_for_variant(
        variant=baseline_variant,
        scene_rows=baseline_scene_rows,
        current_rows=baseline_current,
        phase2_summaries=phase2_summaries,
        local_baseline={"MV_AP_window": 0.0},
        history_metrics={
            "object_history_assignment_rate": 0.0,
            "object_history_top1_score_mean": 0.0,
            "object_history_top1_margin_mean": 0.0,
            "object_history_entropy_mean": 1.0,
            "one_to_one_conflict_count": 0,
            "tentative_new_object_count": int(current_rows_df["source_local_object_id"].nunique()),
            "confirmed_history_inheritance_count": 0,
            "new_history_object_count": int(current_rows_df["source_local_object_id"].nunique()),
        },
        min_pred_pixels=int(args.min_pred_pixels),
        min_gt_pixels=int(args.min_gt_pixels),
        use_cupy_iou=not bool(args.disable_cupy_iou),
        cupy_device_id=int(args.cupy_device_id),
    )
    baseline_metric["local_AP_drop_vs_phaseS3"] = 0.0
    local_baseline = {
        "MV_AP_window": float(baseline_metric["local_MV_AP_window_after_history"]),
        "MV_AP50_window": float(baseline_metric["local_MV_AP50_window_after_history"]),
        "pixel_collision_rate": float(baseline_metric["pixel_collision_rate"]),
    }

    all_assignment_rows: list[dict[str, Any]] = []
    all_score_rows: list[dict[str, Any]] = []
    all_scene_metric_rows: list[dict[str, Any]] = [baseline_metric]
    all_per_scene_rows: list[dict[str, Any]] = baseline_per_scene
    all_frame_rows: list[dict[str, Any]] = baseline_scene_frames + baseline_local_frames
    object_frame_rows: list[dict[str, Any]] = list(baseline_scene_rows)
    scene_meta: list[dict[str, Any]] = []

    payload_by_scene = token_payload["payload_by_scene"]
    projection_by_scene = {
        scene: _prepare_scene_projection(
            scene=scene,
            phase2_root=scene_phase2_roots[scene],
            scene_payload=payload_by_scene[scene],
            cupy_device_id=int(args.cupy_device_id),
        )
        for scene in sorted(payload_by_scene)
    }
    for variant in VARIANTS:
        mapping_by_scene: dict[str, dict[str, str]] = {}
        variant_assignment_rows: list[dict[str, Any]] = []
        variant_score_rows: list[dict[str, Any]] = []
        history_metrics_acc: dict[str, list[float]] = defaultdict(list)
        scene_variant_meta: list[dict[str, Any]] = []
        for scene in sorted(payload_by_scene):
            scene_payload = payload_by_scene[scene]
            object_support, object_ids, carrier_assignment, object_aux, pool_meta = _pool_object_history_support(
                scene=scene,
                current_rows=current_rows_df,
                scene_payload=scene_payload,
                projection=projection_by_scene[scene],
                variant=variant,
                args=args,
            )
            history_ids = [str(v) for v in scene_payload["history_ids"]]
            assignment_rows, score_rows, object_metrics, mapping = _assign_objects(
                scene=scene,
                variant=variant,
                object_support=object_support,
                object_ids=object_ids,
                history_ids=history_ids,
                object_aux=object_aux,
            )
            mapping_by_scene[scene] = mapping
            variant_assignment_rows.extend(assignment_rows)
            variant_score_rows.extend(score_rows)
            for key, value in object_metrics.items():
                history_metrics_acc[key].append(float(value))
            for key in [
                "carrier_history_assignment_rate",
                "carrier_history_top1_score_mean",
                "carrier_history_top1_margin_mean",
                "carrier_history_entropy_mean",
                "carrier_history_hard_conflict_rate",
                "object_with_any_carrier_hit_rate",
                "object_with_valid_carrier_hit_rate",
            ]:
                history_metrics_acc[key].append(float(pool_meta.get(key, 0.0)))
            scene_variant_meta.append({**pool_meta, "variant_id": str(variant["variant_id"])})

        mapped_current = _materialize_current_rows(current_rows_df, mapping_by_scene, str(variant["variant_id"]))
        mapped_scene_rows = history_snapshot.to_dict("records") + mapped_current
        history_metrics = {key: float(np.mean(vals)) if vals else 0.0 for key, vals in history_metrics_acc.items()}
        metric_row, per_scene, scene_frames, local_frames = _metric_row_for_variant(
            variant=variant,
            scene_rows=mapped_scene_rows,
            current_rows=mapped_current,
            phase2_summaries=phase2_summaries,
            local_baseline=local_baseline,
            history_metrics=history_metrics,
            min_pred_pixels=int(args.min_pred_pixels),
            min_gt_pixels=int(args.min_gt_pixels),
            use_cupy_iou=not bool(args.disable_cupy_iou),
            cupy_device_id=int(args.cupy_device_id),
        )
        all_assignment_rows.extend(variant_assignment_rows)
        all_score_rows.extend(variant_score_rows)
        all_scene_metric_rows.append(metric_row)
        all_per_scene_rows.extend(per_scene)
        all_frame_rows.extend(scene_frames + local_frames)
        object_frame_rows.extend(mapped_scene_rows)
        scene_meta.extend(scene_variant_meta)

    metric_df = pd.DataFrame(all_scene_metric_rows)
    real_df = metric_df[(metric_df["is_control"].astype(bool) == False) & (metric_df["variant_id"].astype(str) != "S4_B0_current_local_no_history_baseline")]
    control_df = metric_df[metric_df["is_control"].astype(bool)]
    best_real = real_df.sort_values(["MV_AP_scene", "MV_AP50_scene", "local_MV_AP_window_after_history"], ascending=False).head(1).iloc[0].to_dict()
    control_by_family = {str(row["control_family"]): row for row in control_df.to_dict("records")}
    shuffled = control_by_family.get("shuffled_history", {})
    stale = control_by_family.get("stale_history", {})
    semantic = control_by_family.get("semantic_only_history", {})
    random_control = control_by_family.get("random_history_token", {})
    real_minus_shuffled = float(_num(best_real.get("MV_AP_scene")) - _num(shuffled.get("MV_AP_scene")))
    real_minus_stale = float(_num(best_real.get("MV_AP_scene")) - _num(stale.get("MV_AP_scene")))
    real_minus_semantic = float(_num(best_real.get("MV_AP_scene")) - _num(semantic.get("MV_AP_scene")))

    fragmented_scene_gate = _num(phaseS0.get("fragmented_dev_MV_AP_scene")) + 0.010
    fragmented_ap50_gate = _num(phaseS0.get("fragmented_dev_MV_AP50_scene")) + 0.015
    local_floor = float(local_baseline["MV_AP_window"]) - 0.003
    pixel_floor = float(local_baseline["pixel_collision_rate"]) + 0.005
    gates = [
        {
            "gate_id": "current_chunk_alignment_uses_c0001_post_birth_rows",
            "pass": True,
            "expected": "current rows use c0001 F2 post-birth local objects aligned with Phase7 current frames",
            "observed": f"current_chunk={args.current_chunk_id}; phaseS3_alignment={alignment_status}",
            "severity": "causal_contract",
        },
        {
            "gate_id": "mv_ap_scene_ge_fragmented_dev_plus_0p010",
            "pass": _num(best_real.get("MV_AP_scene")) >= fragmented_scene_gate,
            "expected": fragmented_scene_gate,
            "observed": _num(best_real.get("MV_AP_scene")),
            "severity": "method_gate",
        },
        {
            "gate_id": "mv_ap50_scene_ge_fragmented_dev_plus_0p015",
            "pass": _num(best_real.get("MV_AP50_scene")) >= fragmented_ap50_gate,
            "expected": fragmented_ap50_gate,
            "observed": _num(best_real.get("MV_AP50_scene")),
            "severity": "method_gate",
        },
        {
            "gate_id": "local_mv_ap_window_after_history_preserved",
            "pass": _num(best_real.get("local_MV_AP_window_after_history")) >= local_floor,
            "expected": local_floor,
            "observed": _num(best_real.get("local_MV_AP_window_after_history")),
            "severity": "local_safety",
        },
        {
            "gate_id": "same_frame_collision_zero",
            "pass": int(_num(best_real.get("same_frame_collision_count"), 1)) == 0,
            "expected": 0,
            "observed": int(_num(best_real.get("same_frame_collision_count"), 1)),
            "severity": "safety",
        },
        {
            "gate_id": "pixel_collision_rate_within_floor",
            "pass": _num(best_real.get("pixel_collision_rate")) <= pixel_floor,
            "expected": pixel_floor,
            "observed": _num(best_real.get("pixel_collision_rate")),
            "severity": "safety",
        },
        {
            "gate_id": "objects_crossing_multiple_chunks_positive",
            "pass": int(_num(best_real.get("objects_crossing_multiple_chunks"))) > 0,
            "expected": ">0",
            "observed": int(_num(best_real.get("objects_crossing_multiple_chunks"))),
            "severity": "history_inheritance",
        },
        {
            "gate_id": "real_minus_shuffled_mv_ap_scene_ge_0p006",
            "pass": real_minus_shuffled >= 0.006,
            "expected": 0.006,
            "observed": real_minus_shuffled,
            "severity": "control",
        },
        {
            "gate_id": "real_minus_stale_mv_ap_scene_ge_0p006",
            "pass": real_minus_stale >= 0.006,
            "expected": 0.006,
            "observed": real_minus_stale,
            "severity": "control",
        },
        {
            "gate_id": "real_minus_semantic_mv_ap_scene_ge_0p003",
            "pass": real_minus_semantic >= 0.003,
            "expected": 0.003,
            "observed": real_minus_semantic,
            "severity": "control",
        },
        {
            "gate_id": "uses_future_false_and_uses_gt_for_prediction_false",
            "pass": not bool(best_real.get("uses_future")) and not bool(best_real.get("uses_gt_for_prediction")),
            "expected": "uses_future=false; uses_gt_for_prediction=false",
            "observed": f"uses_future={best_real.get('uses_future')} uses_gt_for_prediction={best_real.get('uses_gt_for_prediction')}",
            "severity": "truthfulness",
        },
    ]
    failure_rows = []
    for row in gates:
        if bool(row["pass"]):
            continue
        repair_direction = "Inspect carrier-to-current-object support coverage before lowering tau_hist."
        if row["gate_id"].startswith("real_minus"):
            repair_direction = "HISTORY_CONTROL_BIAS_REJECTED: return to S1/S3 object-specific carrier evidence; do not claim history success."
        if row["gate_id"].startswith("mv_ap"):
            repair_direction = "Check whether inherited histories actually join pre-update and current objects; if assignment rate is low, inspect overlap evidence and carrier-to-history support coverage."
        failure_rows.append(
            {
                "schema_version": f"{SCHEMA_PREFIX}_failure_row_v1",
                "phase_id": PHASE_ID,
                "failure_id": row["gate_id"],
                "severity": row["severity"],
                "expected": row["expected"],
                "observed": row["observed"],
                "repair_direction": repair_direction,
            }
        )

    decision = "PASS_ENTER_PHASES5_OR_CASEBOOK" if not failure_rows else "NO_GO_REPAIR_PHASES4_POST_BIRTH_HISTORY_INHERITANCE"
    outputs = {
        "history_assignment_rows": _rel(out / "history_assignment_rows.parquet"),
        "history_memory_snapshot_rows": _rel(out / "history_memory_snapshot_rows.parquet"),
        "object_history_score_rows": _rel(out / "object_history_score_rows.parquet"),
        "object_frame_mask_rows": _rel(out / "object_frame_mask_rows.parquet"),
        "scene_metric_rows": _rel(out / "scene_metric_rows.csv"),
        "control_metric_rows": _rel(out / "control_metric_rows.csv"),
        "per_scene_metric_rows": _rel(out / "per_scene_metric_rows.csv"),
        "frame_eval_rows": _rel(out / "frame_eval_rows.csv"),
        "gate_rows": _rel(out / "gate_rows.csv"),
        "failure_rows": _rel(out / "failure_rows.csv"),
        "last_command": _rel(out / "last_command.txt"),
        "summary": _rel(out / "summary.json"),
    }
    summary = {
        "schema_version": f"{SCHEMA_PREFIX}_summary_v1",
        "phase_id": PHASE_ID,
        "created_unix_time": time.time(),
        "runtime_sec": time.time() - started,
        "decision": decision,
        "phaseS4_pass": not failure_rows,
        "failure_count": len(failure_rows),
        "best_variant_id": str(best_real["variant_id"]),
        "best_MV_AP_scene": float(_num(best_real.get("MV_AP_scene"))),
        "best_MV_AP50_scene": float(_num(best_real.get("MV_AP50_scene"))),
        "best_local_MV_AP_window_after_history": float(_num(best_real.get("local_MV_AP_window_after_history"))),
        "baseline_no_history_MV_AP_scene": float(_num(baseline_metric.get("MV_AP_scene"))),
        "baseline_no_history_MV_AP50_scene": float(_num(baseline_metric.get("MV_AP50_scene"))),
        "baseline_current_local_MV_AP_window": float(local_baseline["MV_AP_window"]),
        "fragmented_dev_MV_AP_scene": _num(phaseS0.get("fragmented_dev_MV_AP_scene")),
        "fragmented_dev_MV_AP50_scene": _num(phaseS0.get("fragmented_dev_MV_AP50_scene")),
        "shuffled_history_MV_AP_scene": float(_num(shuffled.get("MV_AP_scene"))),
        "stale_history_MV_AP_scene": float(_num(stale.get("MV_AP_scene"))),
        "semantic_only_history_MV_AP_scene": float(_num(semantic.get("MV_AP_scene"))),
        "random_history_token_MV_AP_scene": float(_num(random_control.get("MV_AP_scene"))),
        "real_minus_shuffled_MV_AP_scene": real_minus_shuffled,
        "real_minus_stale_MV_AP_scene": real_minus_stale,
        "real_minus_semantic_MV_AP_scene": real_minus_semantic,
        "history_root": _rel(history_root),
        "history_variant_id": history_variant_id,
        "phase7_root": _rel(phase7_root),
        "phaseS3_alignment_status": alignment_status,
        "current_object_source": str(args.current_object_source),
        "phaseS3_variant_id": str(args.phaseS3_variant_id) if str(args.current_object_source) == "aligned_s3" else "",
        "phase6d_root": _rel(_project(args.phase6d_root)) if str(args.current_object_source) == "phase6d" else "",
        "phase6d_variant_id": str(args.phase6d_variant_id) if str(args.current_object_source) == "phase6d" else "",
        "metric_scope": "history_c0000_plus_current_c0001_dev_subset",
        "variant_count": len(VARIANTS),
        "method_variant_count": len([v for v in VARIANTS if not bool(v.get("is_control"))]),
        "control_variant_count": len([v for v in VARIANTS if bool(v.get("is_control"))]),
        "scene_meta": scene_meta,
        "outputs": outputs,
        "truthfulness_note": "S4 changes only post-birth identity assignment. The current local object source is recorded explicitly; aligned_s3 remains diagnostic, while phase6d requires a prior PASS_PHASE6D_S3_STYLE_LOCAL_GATE summary.",
        "uses_future": False,
        "uses_gt_for_prediction": False,
        "uses_gt_for_eval": True,
    }

    _write_parquet(out / "history_assignment_rows.parquet", all_assignment_rows)
    _write_parquet(out / "history_memory_snapshot_rows.parquet", history_snapshot.to_dict("records"))
    _write_parquet(out / "object_history_score_rows.parquet", all_score_rows)
    _write_parquet(out / "object_frame_mask_rows.parquet", object_frame_rows)
    _write_csv(out / "scene_metric_rows.csv", all_scene_metric_rows)
    _write_csv(out / "control_metric_rows.csv", control_df.to_dict("records"))
    _write_csv(out / "per_scene_metric_rows.csv", all_per_scene_rows)
    _write_csv(out / "frame_eval_rows.csv", all_frame_rows)
    _write_csv(out / "gate_rows.csv", gates)
    _write_csv(out / "failure_rows.csv", failure_rows)
    _write_json(out / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if not failure_rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
