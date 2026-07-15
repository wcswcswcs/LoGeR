#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent

PHASE_ID = "v103_supp_r2_phaseP3_carrier_segmentation"
DEFAULT_P2_ROOT = STREAM3D_ROOT / "outputs/audit/v103_supp_r2_phaseP2_observation_reliability"
DEFAULT_OUT = STREAM3D_ROOT / "outputs/audit/v103_supp_r2_phaseP3_carrier_segmentation"

ROLE_A = 1
ROLE_S = 2
ROLE_V = 3
ROLE_U = 4
ROLE_NAME = {
    ROLE_A: "A_obs_anchor",
    ROLE_S: "S_obs_support",
    ROLE_V: "V_obs_veto",
    ROLE_U: "U_obs_uncertain",
}

SEG_A = "A_anchor_segment"
SEG_S = "S_support_segment"
SEG_REJECT = "U_rejected_segment"

P2_COLUMNS = [
    "scene_id",
    "chunk_id",
    "carrier_id",
    "frame_id",
    "x",
    "y",
    "mask_id",
    "q_geo",
    "q_mask",
    "q_sem_final",
    "q_final",
    "visibility_prob",
    "confidence_prob",
    "broad_risk",
    "boundary_risk",
    "competing_mask_risk",
    "semantic_source_disagreement",
    "observation_role",
    "observation_role_code",
    "prev_whole_carrier_A_anchor",
]


def _rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return p.resolve().relative_to(REPO_ROOT).as_posix()
    except Exception:
        return p.as_posix()


def _project(path: Path | str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if p.parts and p.parts[0] == "Stream3D":
        return REPO_ROOT / p
    return STREAM3D_ROOT / p


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


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _gate_row(gate_id: str, passed: bool, observed: Any, required: Any, repair_direction: str, *, uses_gt: bool = False) -> dict[str, Any]:
    return {
        "schema_version": "stream4d_v103_supp_r2_phaseP3_segment_gate_row_v1",
        "phase_id": PHASE_ID,
        "gate_id": gate_id,
        "pass": bool(passed),
        "observed": json.dumps(_jsonable(observed), sort_keys=True) if isinstance(observed, (dict, list, tuple)) else observed,
        "required": required,
        "repair_direction": repair_direction,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic": bool(uses_gt),
    }


def _failure_row(scene_id: str, blocker: str, detail: Any, repair_direction: str) -> dict[str, Any]:
    return {
        "schema_version": "stream4d_v103_supp_r2_phaseP3_failure_row_v1",
        "phase_id": PHASE_ID,
        "scene_id": scene_id,
        "blocker": blocker,
        "detail": json.dumps(_jsonable(detail), sort_keys=True) if isinstance(detail, (dict, list, tuple)) else detail,
        "repair_direction": repair_direction,
    }


def _load_p2_summary(p2_root: Path) -> dict[str, dict[str, Any]]:
    path = p2_root / "carrier_observation_summary_rows.csv"
    rows = pd.read_csv(path)
    return {str(row["scene_id"]): dict(row) for _, row in rows.iterrows()}


def _scene_row_groups(pf: pq.ParquetFile, scene_id: str) -> list[int]:
    scene_idx = pf.schema.names.index("scene_id")
    groups: list[int] = []
    for idx in range(pf.num_row_groups):
        col = pf.metadata.row_group(idx).column(scene_idx)
        stats = col.statistics
        if stats is None:
            groups.append(idx)
            continue
        if str(stats.min) <= scene_id <= str(stats.max):
            groups.append(idx)
    return groups


def _read_scene_dataframe(path: Path, scene_id: str) -> pd.DataFrame:
    pf = pq.ParquetFile(path)
    groups = _scene_row_groups(pf, scene_id)
    if not groups:
        raise RuntimeError(f"no row groups found for {scene_id} in {path}")
    table = pf.read_row_groups(groups, columns=P2_COLUMNS)
    df = table.to_pandas(split_blocks=True, self_destruct=True)
    df = df[df["scene_id"].astype(str) == scene_id].copy()
    if df.empty:
        raise RuntimeError(f"empty P2 rows for {scene_id} in {path}")
    dtype_map = {
        "carrier_id": np.int64,
        "frame_id": np.int32,
        "x": np.int16,
        "y": np.int16,
        "mask_id": np.int16,
        "observation_role_code": np.int8,
    }
    for key, dtype in dtype_map.items():
        df[key] = df[key].astype(dtype, copy=False)
    for key in ["q_geo", "q_mask", "q_sem_final", "q_final", "visibility_prob", "confidence_prob", "semantic_source_disagreement"]:
        df[key] = df[key].astype(np.float32, copy=False)
    for key in ["broad_risk", "boundary_risk", "competing_mask_risk", "prev_whole_carrier_A_anchor"]:
        df[key] = df[key].astype(bool, copy=False)
    df.sort_values(["carrier_id", "frame_id"], kind="mergesort", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def _infer_frame_stride(frame_id: np.ndarray) -> int:
    uniq = np.unique(frame_id.astype(np.int64, copy=False))
    diffs = np.diff(uniq)
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return 1
    stride = int(diffs[0])
    for value in diffs[1:].tolist():
        stride = int(np.gcd(stride, int(value)))
    return max(stride, 1)


def _segment_scene(
    df: pd.DataFrame,
    scene_id: str,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    t0 = time.time()
    n = int(df.shape[0])
    carrier = df["carrier_id"].to_numpy(np.int64, copy=False)
    frame = df["frame_id"].to_numpy(np.int32, copy=False)
    role = df["observation_role_code"].to_numpy(np.int8, copy=False)
    x = df["x"].to_numpy(np.float32, copy=False)
    y = df["y"].to_numpy(np.float32, copy=False)
    q_sem = df["q_sem_final"].to_numpy(np.float32, copy=False)
    disagreement = df["semantic_source_disagreement"].to_numpy(np.float32, copy=False)
    stride = _infer_frame_stride(frame)
    width = max(float(np.max(x) + 1.0), 1.0)
    height = max(float(np.max(y) + 1.0), 1.0)
    diag = max(float(np.hypot(width, height)), 1.0)

    prev_same = np.zeros(n, dtype=bool)
    if n > 1:
        prev_same[1:] = carrier[1:] == carrier[:-1]
    frame_gap = np.zeros(n, dtype=np.int32)
    if n > 1:
        frame_gap[1:] = frame[1:] - frame[:-1]
    jump = np.zeros(n, dtype=np.float32)
    if n > 1:
        dx = x[1:] - x[:-1]
        dy = y[1:] - y[:-1]
        jump[1:] = (np.sqrt(dx * dx + dy * dy) / diag).astype(np.float32, copy=False)
        jump[~prev_same] = 0.0

    hard_break = np.zeros(n, dtype=bool)
    hard_break[0] = True
    if n > 1:
        gap_units = np.ceil(np.maximum(frame_gap[1:], 0) / float(stride)).astype(np.int32)
        current_role = role[1:]
        prev_role = role[:-1]
        current_veto = current_role == ROLE_V
        prev_veto = prev_role == ROLE_V
        gap_break = gap_units > int(args.max_frame_gap_units)
        jump_break = jump[1:] > float(args.hard_jump_norm)
        sustained_semantic_disagreement = (
            (disagreement[1:] > float(args.sustained_disagreement_break))
            & (disagreement[:-1] > float(args.sustained_disagreement_break))
        )
        sustained_low_sem = (q_sem[1:] < float(args.sustained_low_q_sem_break)) & (
            q_sem[:-1] < float(args.sustained_low_q_sem_break)
        )
        double_uncertain = (current_role == ROLE_U) & (prev_role == ROLE_U)
        hard_break[1:] = (
            (~prev_same[1:])
            | current_veto
            | prev_veto
            | gap_break
            | jump_break
            | sustained_semantic_disagreement
            | sustained_low_sem
            | double_uncertain
        )

    segment_group = np.cumsum(hard_break, dtype=np.int64) - 1
    usable = role != ROLE_V
    vdf = df.loc[usable, P2_COLUMNS].copy()
    vdf["segment_group"] = segment_group[usable]
    vdf["jump_norm_from_prev"] = np.where(hard_break[usable], 0.0, jump[usable]).astype(np.float32, copy=False)
    vdf["is_anchor"] = (vdf["observation_role_code"].to_numpy(np.int8, copy=False) == ROLE_A).astype(np.int16)
    vdf["is_support"] = (vdf["observation_role_code"].to_numpy(np.int8, copy=False) == ROLE_S).astype(np.int16)
    vdf["is_uncertain"] = (vdf["observation_role_code"].to_numpy(np.int8, copy=False) == ROLE_U).astype(np.int16)
    vdf["veto_zero"] = np.zeros(vdf.shape[0], dtype=np.int16)
    vdf["broad_int"] = vdf["broad_risk"].astype(np.int16, copy=False)
    vdf["competing_int"] = vdf["competing_mask_risk"].astype(np.int16, copy=False)

    gb = vdf.groupby("segment_group", sort=False, observed=True)
    seg = gb.agg(
        chunk_id=("chunk_id", "first"),
        carrier_id=("carrier_id", "first"),
        frame_start=("frame_id", "min"),
        frame_end=("frame_id", "max"),
        valid_frame_count=("frame_id", "count"),
        anchor_observation_count=("is_anchor", "sum"),
        support_observation_count=("is_support", "sum"),
        uncertain_observation_count=("is_uncertain", "sum"),
        veto_observation_count=("veto_zero", "sum"),
        mean_q_geo=("q_geo", "mean"),
        mean_q_mask=("q_mask", "mean"),
        mean_q_sem=("q_sem_final", "mean"),
        mean_q_final=("q_final", "mean"),
        segment_semantic_disagreement_mean=("semantic_source_disagreement", "mean"),
        segment_broad_rate=("broad_int", "mean"),
        segment_competing_rate=("competing_int", "mean"),
        mean_visibility=("visibility_prob", "mean"),
        mean_confidence=("confidence_prob", "mean"),
    )
    jitter = gb["jump_norm_from_prev"].quantile(0.90)
    seg["segment_jitter_norm_p90"] = jitter.astype(np.float32)
    seg["segment_semantic_stability"] = np.clip(1.0 - seg["segment_semantic_disagreement_mean"].astype(np.float32), 0.0, 1.0)
    seg.reset_index(inplace=True)

    anchor_cond = (
        (seg["anchor_observation_count"] >= int(args.min_anchor_observations))
        & (seg["valid_frame_count"] >= int(args.min_valid_frame_count))
        & (seg["mean_q_final"] >= float(args.min_anchor_mean_q_final))
        & (seg["mean_q_sem"] >= float(args.min_anchor_mean_q_sem))
        & (seg["segment_semantic_stability"] >= float(args.min_anchor_semantic_stability))
        & (seg["segment_jitter_norm_p90"] <= float(args.max_anchor_jitter_norm_p90))
        & (seg["segment_broad_rate"] <= float(args.max_anchor_broad_rate))
        & (seg["segment_competing_rate"] <= float(args.max_anchor_competing_rate))
    )
    support_cond = (
        (~anchor_cond)
        & (seg["valid_frame_count"] >= int(args.min_support_valid_frame_count))
        & ((seg["anchor_observation_count"] >= 1) | (seg["support_observation_count"] >= int(args.min_support_observations)))
        & (seg["mean_q_final"] >= float(args.min_support_mean_q_final))
        & (seg["mean_q_sem"] >= float(args.min_support_mean_q_sem))
        & (seg["segment_semantic_stability"] >= float(args.min_support_semantic_stability))
        & (seg["segment_jitter_norm_p90"] <= float(args.max_support_jitter_norm_p90))
        & (seg["segment_broad_rate"] <= float(args.max_support_broad_rate))
    )
    seg["segment_role"] = np.where(anchor_cond, SEG_A, np.where(support_cond, SEG_S, SEG_REJECT))
    rejected_count = int(np.count_nonzero(seg["segment_role"].to_numpy(dtype=object) == SEG_REJECT))
    seg = seg[seg["segment_role"] != SEG_REJECT].copy()
    seg.reset_index(drop=True, inplace=True)
    seg.insert(0, "schema_version", "stream4d_v103_supp_r2_phaseP3_carrier_segment_row_v1")
    seg.insert(1, "phase_id", PHASE_ID)
    seg.insert(2, "scene_id", scene_id)
    seg.insert(5, "segment_id", [f"{scene_id}:seg_{idx:09d}" for idx in range(int(seg.shape[0]))])
    seg["uses_gt_for_prediction"] = False
    seg["uses_future"] = False

    id_map = seg[["segment_group", "segment_id", "segment_role"]]
    link = vdf.merge(id_map, on="segment_group", how="inner", sort=False)
    link["observation_role"] = link["observation_role_code"].map(ROLE_NAME).fillna(link["observation_role"])
    link = link[
        [
            "scene_id",
            "chunk_id",
            "carrier_id",
            "segment_id",
            "segment_role",
            "frame_id",
            "x",
            "y",
            "mask_id",
            "observation_role",
            "observation_role_code",
            "q_geo",
            "q_mask",
            "q_sem_final",
            "q_final",
            "visibility_prob",
            "confidence_prob",
            "semantic_source_disagreement",
            "jump_norm_from_prev",
            "broad_risk",
            "boundary_risk",
            "competing_mask_risk",
            "prev_whole_carrier_A_anchor",
        ]
    ].copy()
    link.insert(0, "schema_version", "stream4d_v103_supp_r2_phaseP3_segment_observation_link_row_v1")
    link.insert(1, "phase_id", PHASE_ID)
    link["uses_gt_for_prediction"] = False
    link["uses_future"] = False

    meta = {
        "scene_id": scene_id,
        "input_observation_count": n,
        "usable_non_veto_observation_count": int(vdf.shape[0]),
        "raw_segment_group_count": int(rejected_count + seg.shape[0]),
        "rejected_segment_group_count": rejected_count,
        "output_segment_count": int(seg.shape[0]),
        "output_link_row_count": int(link.shape[0]),
        "frame_stride": stride,
        "image_width_est": width,
        "image_height_est": height,
        "hard_break_count": int(np.count_nonzero(hard_break)),
        "veto_hard_break_observation_count": int(np.count_nonzero(role == ROLE_V)),
        "runtime_sec": time.time() - t0,
    }
    return seg, link, meta


def _percentile(values: np.ndarray | pd.Series, q: float) -> float:
    arr = np.asarray(values)
    if arr.size == 0:
        return 0.0
    return float(np.percentile(arr.astype(np.float64, copy=False), q))


def _object_like_anchor_support_p10(link: pd.DataFrame, segment_role: str) -> float:
    sub = link[(link["segment_role"] == segment_role) & (link["observation_role_code"] == ROLE_A)]
    if sub.empty:
        return 0.0
    key = sub["frame_id"].astype(np.int64).to_numpy() * np.int64(100000) + sub["mask_id"].astype(np.int64).to_numpy()
    counts = np.unique(key, return_counts=True)[1]
    return float(np.percentile(counts.astype(np.float64), 10)) if counts.size else 0.0


def _scene_role_summary(
    scene_id: str,
    seg: pd.DataFrame,
    link: pd.DataFrame,
    meta: dict[str, Any],
    p2_summary: dict[str, Any],
    gt_diag: dict[str, Any] | None,
) -> dict[str, Any]:
    a_seg = seg[seg["segment_role"] == SEG_A]
    s_seg = seg[seg["segment_role"] == SEG_S]
    a_link = link[link["segment_role"] == SEG_A]
    row: dict[str, Any] = {
        "schema_version": "stream4d_v103_supp_r2_phaseP3_segment_role_summary_row_v1",
        "phase_id": PHASE_ID,
        "scene_id": scene_id,
        "chunk_id": str(seg["chunk_id"].iloc[0]) if not seg.empty else "",
        "previous_A_anchor_carrier_count": int(p2_summary.get("previous_A_anchor_carrier_count", 0)),
        "previous_A_anchor_foreground_observation_count": int(p2_summary.get("previous_A_anchor_foreground_observation_count", 0)),
        "previous_A_anchor_broad_rate": float(p2_summary.get("previous_broad_anchor_rate", 1.0)),
        "total_input_observation_count": int(meta["input_observation_count"]),
        "usable_non_veto_observation_count": int(meta["usable_non_veto_observation_count"]),
        "raw_segment_group_count": int(meta["raw_segment_group_count"]),
        "rejected_segment_group_count": int(meta["rejected_segment_group_count"]),
        "total_segment_count": int(seg.shape[0]),
        "A_anchor_segment_count": int(a_seg.shape[0]),
        "S_support_segment_count": int(s_seg.shape[0]),
        "segment_observation_link_row_count": int(link.shape[0]),
        "A_anchor_segment_link_row_count": int(a_link.shape[0]),
        "A_anchor_segment_object_like_mask_support_p10": _object_like_anchor_support_p10(link, SEG_A),
        "A_anchor_segment_broad_rate": float(a_seg["segment_broad_rate"].mean()) if not a_seg.empty else 0.0,
        "A_anchor_segment_competing_rate": float(a_seg["segment_competing_rate"].mean()) if not a_seg.empty else 0.0,
        "A_anchor_segment_mean_q_final": float(a_seg["mean_q_final"].mean()) if not a_seg.empty else 0.0,
        "A_anchor_segment_jitter_norm_p90": _percentile(a_seg["segment_jitter_norm_p90"], 90) if not a_seg.empty else 0.0,
        "S_support_segment_mean_q_final": float(s_seg["mean_q_final"].mean()) if not s_seg.empty else 0.0,
        "runtime_sec": float(meta["runtime_sec"]),
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic": bool(gt_diag),
    }
    if gt_diag:
        row.update(gt_diag)
    return row


def _read_label_png(path: Path, shape_hw: tuple[int, int] | None = None) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    if image.ndim == 3:
        image = image[..., 0]
    if shape_hw is not None and tuple(image.shape[:2]) != tuple(shape_hw):
        image = cv2.resize(image, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)
    return np.asarray(image)


def _sample_gt_labels(scene_id: str, rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    inst_out = np.zeros((int(rows.shape[0]),), dtype=np.int64)
    sem_out = np.zeros((int(rows.shape[0]),), dtype=np.int64)
    if rows.empty:
        return inst_out, sem_out
    scene_root = STREAM3D_ROOT / "data/scannet/processed" / scene_id
    index = np.arange(int(rows.shape[0]), dtype=np.int64)
    frame_values = rows["frame_id"].to_numpy(np.int32, copy=False)
    x_values = rows["x"].to_numpy(np.int64, copy=False)
    y_values = rows["y"].to_numpy(np.int64, copy=False)
    for frame_id in np.unique(frame_values).tolist():
        mask = frame_values == int(frame_id)
        local = index[mask]
        inst = _read_label_png(scene_root / "instance/instance" / f"{int(frame_id)}.png")
        sem = _read_label_png(scene_root / "label-filt" / f"{int(frame_id)}.png", tuple(inst.shape[:2]))
        h, w = inst.shape[:2]
        xs = np.clip(x_values[mask], 0, w - 1)
        ys = np.clip(y_values[mask], 0, h - 1)
        inst_out[local] = inst[ys, xs].astype(np.int64, copy=False)
        sem_out[local] = sem[ys, xs].astype(np.int64, copy=False)
    return inst_out, sem_out


def _entity_bridge_stats(entity: np.ndarray, instance_id: np.ndarray, semantic_id: np.ndarray, *, entity_label: str, min_gt_obs: int) -> dict[str, Any]:
    if entity.size == 0:
        return {
            f"{entity_label}_gt_labeled_count": 0,
            f"{entity_label}_gt_positive_observation_count": 0,
            f"{entity_label}_multi_gt_rate": 0.0,
            f"{entity_label}_same_semantic_false_bridge_rate": 0.0,
            f"{entity_label}_dominant_gt_purity_mean": 0.0,
        }
    lab = pd.DataFrame(
        {
            "entity_id": entity,
            "instance_id": instance_id.astype(np.int64, copy=False),
            "semantic_id": semantic_id.astype(np.int64, copy=False),
        }
    )
    lab = lab[(lab["instance_id"] > 0) & (lab["semantic_id"] > 0)].copy()
    if lab.empty:
        return {
            f"{entity_label}_gt_labeled_count": 0,
            f"{entity_label}_gt_positive_observation_count": 0,
            f"{entity_label}_multi_gt_rate": 0.0,
            f"{entity_label}_same_semantic_false_bridge_rate": 0.0,
            f"{entity_label}_dominant_gt_purity_mean": 0.0,
        }
    obs = lab.groupby("entity_id", sort=False).size()
    eligible = obs[obs >= int(min_gt_obs)]
    if eligible.empty:
        return {
            f"{entity_label}_gt_labeled_count": 0,
            f"{entity_label}_gt_positive_observation_count": int(obs.sum()),
            f"{entity_label}_multi_gt_rate": 0.0,
            f"{entity_label}_same_semantic_false_bridge_rate": 0.0,
            f"{entity_label}_dominant_gt_purity_mean": 0.0,
        }
    inst_unique = lab.groupby("entity_id", sort=False)["instance_id"].nunique()
    inst_pairs = lab.drop_duplicates(["entity_id", "semantic_id", "instance_id"])
    sem_inst_count = inst_pairs.groupby(["entity_id", "semantic_id"], sort=False).size()
    same_sem_entities = set(sem_inst_count[sem_inst_count >= 2].index.get_level_values(0).tolist())
    inst_count = lab.groupby(["entity_id", "instance_id"], sort=False).size()
    dominant = inst_count.groupby(level=0, sort=False).max()
    purity = dominant.reindex(eligible.index).astype(np.float64) / eligible.astype(np.float64)
    eligible_index = eligible.index
    same_sem_count = int(sum(1 for key in eligible_index.tolist() if key in same_sem_entities))
    return {
        f"{entity_label}_gt_labeled_count": int(eligible.shape[0]),
        f"{entity_label}_gt_positive_observation_count": int(eligible.sum()),
        f"{entity_label}_multi_gt_rate": float((inst_unique.reindex(eligible_index).fillna(0).to_numpy() >= 2).mean()),
        f"{entity_label}_same_semantic_false_bridge_rate": float(same_sem_count / max(1, int(eligible.shape[0]))),
        f"{entity_label}_dominant_gt_purity_mean": float(purity.mean()) if not purity.empty else 0.0,
    }


def _gt_diagnostic_for_scene(scene_id: str, df: pd.DataFrame, link: pd.DataFrame, args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    a_link = link[link["segment_role"] == SEG_A].copy()
    baseline = df[df["prev_whole_carrier_A_anchor"]].copy()

    seg_inst, seg_sem = _sample_gt_labels(scene_id, a_link[["frame_id", "x", "y"]]) if not a_link.empty else (
        np.zeros((0,), dtype=np.int64),
        np.zeros((0,), dtype=np.int64),
    )
    base_inst, base_sem = _sample_gt_labels(scene_id, baseline[["frame_id", "x", "y"]]) if not baseline.empty else (
        np.zeros((0,), dtype=np.int64),
        np.zeros((0,), dtype=np.int64),
    )

    seg_stats = _entity_bridge_stats(
        a_link["segment_id"].astype(str).to_numpy() if not a_link.empty else np.asarray([], dtype=object),
        seg_inst,
        seg_sem,
        entity_label="segment",
        min_gt_obs=int(args.min_gt_positive_observations),
    )
    base_stats = _entity_bridge_stats(
        baseline["carrier_id"].astype(np.int64).to_numpy() if not baseline.empty else np.asarray([], dtype=np.int64),
        base_inst,
        base_sem,
        entity_label="whole_carrier",
        min_gt_obs=int(args.min_gt_positive_observations),
    )
    segment_rate = float(seg_stats["segment_same_semantic_false_bridge_rate"])
    whole_rate = float(base_stats["whole_carrier_same_semantic_false_bridge_rate"])
    delta = segment_rate - whole_rate
    diag = {
        **seg_stats,
        **base_stats,
        "segment_minus_whole_carrier_same_semantic_false_bridge": float(delta),
        "segment_same_semantic_false_bridge_gate_pass": bool(segment_rate <= whole_rate + 0.05),
        "gt_diagnostic_status": "run",
    }
    for metric, value in diag.items():
        if metric == "segment_same_semantic_false_bridge_gate_pass":
            continue
        rows.append(
            {
                "schema_version": "stream4d_v103_supp_r2_phaseP3_segment_gt_diagnostic_row_v1",
                "phase_id": PHASE_ID,
                "scene_id": scene_id,
                "metric_name": metric,
                "value": value,
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic": True,
                "note": "GT instance/semantic labels are diagnostic-only and are not used to build segments.",
            }
        )
    return diag, rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="v103 supplement R2 Phase P3 carrier segmentation.")
    parser.add_argument("--p2-root", default=str(DEFAULT_P2_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--min-anchor-observations", type=int, default=2)
    parser.add_argument("--min-valid-frame-count", type=int, default=2)
    parser.add_argument("--min-anchor-mean-q-final", type=float, default=0.15)
    parser.add_argument("--min-anchor-mean-q-sem", type=float, default=0.50)
    parser.add_argument("--min-anchor-semantic-stability", type=float, default=0.70)
    parser.add_argument("--max-anchor-jitter-norm-p90", type=float, default=0.20)
    parser.add_argument("--max-anchor-broad-rate", type=float, default=0.02)
    parser.add_argument("--max-anchor-competing-rate", type=float, default=0.02)
    parser.add_argument("--min-support-observations", type=int, default=2)
    parser.add_argument("--min-support-valid-frame-count", type=int, default=2)
    parser.add_argument("--min-support-mean-q-final", type=float, default=0.05)
    parser.add_argument("--min-support-mean-q-sem", type=float, default=0.40)
    parser.add_argument("--min-support-semantic-stability", type=float, default=0.55)
    parser.add_argument("--max-support-jitter-norm-p90", type=float, default=0.25)
    parser.add_argument("--max-support-broad-rate", type=float, default=0.10)
    parser.add_argument("--max-frame-gap-units", type=int, default=2)
    parser.add_argument("--hard-jump-norm", type=float, default=0.22)
    parser.add_argument("--sustained-disagreement-break", type=float, default=0.35)
    parser.add_argument("--sustained-low-q-sem-break", type=float, default=0.35)
    parser.add_argument("--min-gt-positive-observations", type=int, default=2)
    parser.add_argument("--no-gt-diagnostic", action="store_true", help="Skip GT diagnostic rows; method gates can still run but P3 cannot fully pass.")
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    t0 = time.time()
    args = build_parser().parse_args()
    p2_root = _project(args.p2_root)
    out = _project(args.output_root)
    if out.exists() and any(out.iterdir()) and not args.force:
        raise SystemExit(f"output root already exists and is non-empty; pass --force to overwrite metadata: {_rel(out)}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "last_command.txt").write_text(" ".join([sys.executable, str(Path(__file__)), *sys.argv[1:]]) + "\n", encoding="utf-8")

    p2_summary = _load_p2_summary(p2_root)
    p2_rows_path = p2_root / "carrier_observation_reliability_rows.parquet"
    p2_summary_json = _read_json(p2_root / "summary.json")
    scene_ids = list(p2_summary.keys())

    segment_writer: pq.ParquetWriter | None = None
    link_writer: pq.ParquetWriter | None = None
    summary_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    gt_rows: list[dict[str, Any]] = []
    scene_meta: list[dict[str, Any]] = []

    try:
        for scene_id in scene_ids:
            scene_t0 = time.time()
            df = _read_scene_dataframe(p2_rows_path, scene_id)
            seg, link, meta = _segment_scene(df, scene_id, args)
            gt_diag: dict[str, Any] | None = None
            if args.no_gt_diagnostic:
                gt_diag = {
                    "gt_diagnostic_status": "skipped",
                    "segment_same_semantic_false_bridge_gate_pass": False,
                }
                failure_rows.append(
                    _failure_row(
                        scene_id,
                        "segment_same_semantic_false_bridge_diagnostic_not_run",
                        "P3 was run with --no-gt-diagnostic.",
                        "Run P3 without --no-gt-diagnostic before entering P4.",
                    )
                )
            else:
                gt_diag, rows = _gt_diagnostic_for_scene(scene_id, df, link, args)
                gt_rows.extend(rows)

            summary = _scene_role_summary(scene_id, seg, link, meta, p2_summary[scene_id], gt_diag)
            summary_rows.append(summary)
            previous_count = int(summary["previous_A_anchor_carrier_count"])
            a_count = int(summary["A_anchor_segment_count"])
            object_p10 = float(summary["A_anchor_segment_object_like_mask_support_p10"])
            broad_rate = float(summary["A_anchor_segment_broad_rate"])
            previous_broad = float(summary["previous_A_anchor_broad_rate"])
            false_pass = bool(summary.get("segment_same_semantic_false_bridge_gate_pass", False))

            scene_gates = [
                _gate_row(
                    f"{scene_id}_A_anchor_segment_count_ge_previous_A_anchor_carrier_count",
                    a_count >= previous_count,
                    {"current": a_count, "previous": previous_count},
                    "current >= previous whole-carrier A_anchor carrier count",
                    "Relax only segment-level continuity thresholds or inspect over-fragmentation.",
                ),
                _gate_row(
                    f"{scene_id}_A_anchor_segment_object_like_mask_support_p10_gt_0",
                    object_p10 > 0.0,
                    object_p10,
                    ">0",
                    "Repair object-like mask assignment before Phase P4.",
                ),
                _gate_row(
                    f"{scene_id}_A_anchor_segment_broad_rate_le_previous_plus_0p02",
                    broad_rate <= previous_broad + 0.02,
                    {"current": broad_rate, "previous": previous_broad},
                    "current <= previous + 0.02",
                    "Tighten broad-mask hard break or downgrade broad segments to support/veto.",
                ),
                _gate_row(
                    f"{scene_id}_segment_same_semantic_false_bridge_not_worse_than_whole_carrier_plus_0p05",
                    false_pass,
                    {
                        "segment": summary.get("segment_same_semantic_false_bridge_rate", ""),
                        "whole_carrier": summary.get("whole_carrier_same_semantic_false_bridge_rate", ""),
                        "delta": summary.get("segment_minus_whole_carrier_same_semantic_false_bridge", ""),
                        "status": summary.get("gt_diagnostic_status", ""),
                    },
                    "segment_rate <= whole_carrier_rate + 0.05",
                    "If failed, enforce stronger source agreement or downgrade high-risk A segments to support/veto.",
                    uses_gt=True,
                ),
            ]
            gate_rows.extend(scene_gates)
            for gate in scene_gates:
                if not bool(gate["pass"]):
                    failure_rows.append(
                        _failure_row(
                            scene_id,
                            str(gate["gate_id"]),
                            gate["observed"],
                            str(gate["repair_direction"]),
                        )
                    )

            seg_table = pa.Table.from_pandas(seg, preserve_index=False)
            if segment_writer is None:
                segment_writer = pq.ParquetWriter(out / "carrier_segment_rows.parquet", seg_table.schema, compression="zstd")
            segment_writer.write_table(seg_table)

            link_table = pa.Table.from_pandas(link, preserve_index=False)
            if link_writer is None:
                link_writer = pq.ParquetWriter(out / "carrier_segment_observation_link_rows.parquet", link_table.schema, compression="zstd")
            link_writer.write_table(link_table)

            scene_meta.append(
                {
                    **meta,
                    "scene_runtime_sec": time.time() - scene_t0,
                    "segment_rows": int(seg.shape[0]),
                    "link_rows": int(link.shape[0]),
                }
            )
            del df, seg, link
    finally:
        if segment_writer is not None:
            segment_writer.close()
        if link_writer is not None:
            link_writer.close()

    _write_csv(out / "segment_role_summary_rows.csv", summary_rows)
    _write_csv(out / "segment_gate_rows.csv", gate_rows)
    _write_csv(out / "failure_rows.csv", failure_rows)
    _write_csv(out / "segment_gt_diagnostic_rows.csv", gt_rows)

    all_gate_pass = all(bool(row["pass"]) for row in gate_rows)
    method_gate_pass = all(
        bool(row["pass"]) for row in gate_rows if "same_semantic_false_bridge" not in str(row["gate_id"])
    )
    decision = "PASS_ENTER_PHASEP4_SEGMENT_AFFINITY_FEATURE" if all_gate_pass else "NO_GO_REPAIR_PHASEP3_SEGMENTATION"
    if method_gate_pass and not all_gate_pass:
        decision = "PARTIAL_METHOD_GATES_PASS_GT_DIAGNOSTIC_OR_FALSE_BRIDGE_REPAIR_REQUIRED"
    summary = {
        "schema_version": "stream4d_v103_supp_r2_phaseP3_summary_v1",
        "phase_id": PHASE_ID,
        "decision": decision,
        "phaseP3_pass": bool(all_gate_pass),
        "phaseP3_method_gates_pass": bool(method_gate_pass),
        "failure_count": int(len(failure_rows)),
        "p2_root": _rel(p2_root),
        "p2_decision": p2_summary_json.get("decision", ""),
        "scene_ids": scene_ids,
        "total_A_anchor_segment_count": int(sum(int(row["A_anchor_segment_count"]) for row in summary_rows)),
        "total_S_support_segment_count": int(sum(int(row["S_support_segment_count"]) for row in summary_rows)),
        "total_segment_observation_link_row_count": int(sum(int(row["segment_observation_link_row_count"]) for row in summary_rows)),
        "segment_policy": {
            "veto_is_hard_break": True,
            "uncertain_gap_policy": "single-frame uncertain gaps may bridge; consecutive uncertain observations split",
            "max_frame_gap_units": int(args.max_frame_gap_units),
            "hard_jump_norm": float(args.hard_jump_norm),
            "sustained_disagreement_break": float(args.sustained_disagreement_break),
            "sustained_low_q_sem_break": float(args.sustained_low_q_sem_break),
            "min_anchor_observations": int(args.min_anchor_observations),
            "min_valid_frame_count": int(args.min_valid_frame_count),
        },
        "outputs": {
            "carrier_segment_rows": _rel(out / "carrier_segment_rows.parquet"),
            "carrier_segment_observation_link_rows": _rel(out / "carrier_segment_observation_link_rows.parquet"),
            "segment_role_summary_rows": _rel(out / "segment_role_summary_rows.csv"),
            "segment_gate_rows": _rel(out / "segment_gate_rows.csv"),
            "segment_gt_diagnostic_rows": _rel(out / "segment_gt_diagnostic_rows.csv"),
            "failure_rows": _rel(out / "failure_rows.csv"),
            "summary": _rel(out / "summary.json"),
        },
        "scene_meta": scene_meta,
        "runtime_sec": time.time() - t0,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic": not bool(args.no_gt_diagnostic),
        "uses_future": False,
        "truthfulness_note": (
            "P3 cuts D4RT carrier observations into segment primitives using P2 reliability only. "
            "GT instance/semantic labels, when enabled, are used only to score the false-bridge diagnostic gate."
        ),
    }
    _write_json(out / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if all_gate_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
