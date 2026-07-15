#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from build_v103_phase3_fast_carrier_reliability_filter import SCENE_INPUTS  # noqa: E402
from build_v103_phase4_primitive_affinity_feature import (  # noqa: E402
    _compute_scene_arrays,
    _countsketch,
    _exact_dense_subset,
    _mask_observations,
    _mask_weights,
    _pair_error,
)
from build_v103_phase5_mask_level_affinity_pooling import (  # noqa: E402
    _build_raw_sketch,
    _pair_values_strict_leave_two_out_bucket_zeroed,
    _pool_features,
    _sample_mask_pairs,
)


PHASE_ID = "v103_supp_r2_phaseP4_segment_affinity_feature"
DEFAULT_P3_ROOT = STREAM3D_ROOT / "outputs/audit/v103_supp_r2_phaseP3_carrier_segmentation"
DEFAULT_OUT = STREAM3D_ROOT / "outputs/audit/v103_supp_r2_phaseP4_segment_affinity_feature"
DEFAULT_PHASE4_BASELINE = STREAM3D_ROOT / "outputs/audit/v103_phase4_primitive_affinity_all_d4rt48mix_maskbalanced8_e5_r1"
DEFAULT_PHASE5_BASELINE = STREAM3D_ROOT / "outputs/audit/v103_phase5_mask_level_pooling_q5c_phase4r7_r4_control_gate_strict_l2o"
DEFAULT_DA3_SEMSOFT_ROOT = STREAM3D_ROOT / "outputs/audit/v103_r2_phase2_da3_semsoft_support_alpha_density_topk_reliable_veto_r6_emitclean"
DEFAULT_PHASE2_ROOT_BY_SCENE = {
    "scene0011_00": STREAM3D_ROOT / "outputs/audit/v103_phase2_stratified_q5c_objlike16384_scene0011_c0001_cap24576_qchunk16384_d4rt48mix_maskbalanced8",
    "scene0050_00": STREAM3D_ROOT / "outputs/audit/v103_phase2_stratified_q5c_objlike16384_scene0050_c0001_cap24576_qchunk16384_d4rt48mix_maskbalanced8",
}


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


def _torch_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _scene_row_groups(pf: pq.ParquetFile, scene_id: str) -> list[int]:
    names = pf.schema.names
    if "scene_id" not in names:
        return list(range(pf.num_row_groups))
    scene_idx = names.index("scene_id")
    groups: list[int] = []
    for idx in range(pf.num_row_groups):
        stats = pf.metadata.row_group(idx).column(scene_idx).statistics
        if stats is None or str(stats.min) <= scene_id <= str(stats.max):
            groups.append(idx)
    return groups


def _read_scene_parquet(path: Path, scene_id: str, columns: list[str]) -> pd.DataFrame:
    pf = pq.ParquetFile(path)
    groups = _scene_row_groups(pf, scene_id)
    table = pf.read_row_groups(groups, columns=columns)
    df = table.to_pandas(split_blocks=True, self_destruct=True)
    if "scene_id" in df.columns:
        df = df[df["scene_id"].astype(str) == scene_id].copy()
    return df


def _load_baselines(phase4_root: Path, phase5_root: Path) -> dict[str, dict[str, float]]:
    phase4_rows = pd.read_csv(phase4_root / "primitive_feature_metric_rows.csv")
    phase5_rows = pd.read_csv(phase5_root / "mask_pooling_metric_rows.csv")
    chosen = phase5_rows[phase5_rows["variant_id"] == "P7_strict_leave_two_out_from_P6_frame_centered_leave_one_out_b0p5"]
    out: dict[str, dict[str, float]] = {}
    for scene in sorted(set(phase4_rows["scene_id"].astype(str).tolist())):
        p4 = phase4_rows[phase4_rows["scene_id"].astype(str) == scene].iloc[0]
        p5s = chosen[chosen["scene_id"].astype(str) == scene]
        if p5s.empty:
            raise RuntimeError(f"missing Phase5 strict L2O baseline for {scene}: {phase5_root}")
        p5 = p5s.iloc[0]
        out[scene] = {
            "previous_broad_contribution_ratio": float(p4["broad_mask_feature_contribution_ratio"]),
            "phase4_d4rt48mix_hard_negative_separation": float(p4["pseudo_positive_minus_hard_negative_margin"]),
            "phase5_d4rt48mix_strict_l2o_hard_negative_separation": float(p5["pseudo_positive_minus_hard_negative_margin"]),
        }
    return out


def _load_da3_best_variant(da3_root: Path, scene_id: str) -> str:
    summary = _read_json(da3_root / "summary.json")
    return str(summary["best_by_scene"][scene_id]["variant_id"])


def _bool_mask(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    if str(series.dtype).startswith("bool"):
        return series.astype(bool)
    return series.astype(str).str.lower().isin({"1", "true", "yes"})


def _da3_support_for_scene(
    *,
    da3_root: Path,
    scene_id: str,
    next_local_index: int,
    mask_lookup: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    variant_id = _load_da3_best_variant(da3_root, scene_id)
    inc = pd.read_parquet(da3_root / "da3_semsoft_primitive_incidence_rows.parquet")
    inc = inc[(inc["scene_id"].astype(str) == scene_id) & (inc["variant_id"].astype(str) == variant_id) & _bool_mask(inc["emitted_to_support"])].copy()
    if inc.empty:
        return pd.DataFrame(), pd.DataFrame(), {
            "da3_semsoft_enabled": True,
            "da3_variant_id": variant_id,
            "da3_component_count": 0,
            "da3_incidence_row_count": 0,
        }
    comp_cols = [
        "scene_id",
        "variant_id",
        "component_id",
        "component_quality_score_mean",
        "component_alpha_mean",
        "component_density_log_mean",
        "component_semantic_risk_mean",
        "component_semantic_risk_p90",
        "is_clean_component",
    ]
    comp = pd.read_csv(da3_root / "da3_semsoft_component_rows.csv", usecols=comp_cols)
    comp = comp[(comp["scene_id"].astype(str) == scene_id) & (comp["variant_id"].astype(str) == variant_id) & _bool_mask(comp["is_clean_component"])].copy()
    comp = comp.drop_duplicates(["component_id"])
    inc = inc.merge(
        comp[
            [
                "component_id",
                "component_quality_score_mean",
                "component_alpha_mean",
                "component_density_log_mean",
                "component_semantic_risk_mean",
                "component_semantic_risk_p90",
            ]
        ],
        on="component_id",
        how="inner",
        sort=False,
    )
    inc = inc.merge(mask_lookup, on=["frame_local_index", "mask_id"], how="left", sort=False)
    inc = inc[inc["mask_observation_index"].notna()].copy()
    if inc.empty:
        return pd.DataFrame(), pd.DataFrame(), {
            "da3_semsoft_enabled": True,
            "da3_variant_id": variant_id,
            "da3_component_count": 0,
            "da3_incidence_row_count": 0,
        }

    component_ids = np.sort(inc["component_id"].astype(np.int64).unique())
    local_by_component = {int(cid): int(next_local_index + idx) for idx, cid in enumerate(component_ids.tolist())}
    inc["segment_local_index"] = inc["component_id"].map(local_by_component).astype(np.int64)
    inc["segment_id"] = [f"{scene_id}:da3_semsoft_{int(v):09d}" for v in inc["component_id"].astype(np.int64).tolist()]
    inc["segment_role"] = "DA3_semsoft_support_component"
    inc["carrier_id"] = -910_000_000_000 - inc["component_id"].astype(np.int64)

    log_count = np.log1p(inc["component_mask_gaussian_count"].astype(np.float32).to_numpy())
    p95 = float(np.percentile(log_count[np.isfinite(log_count)], 95)) if np.any(np.isfinite(log_count)) else 1.0
    count_q = np.clip(log_count / max(p95, 1e-6), 0.05, 1.0)
    quality = np.clip(inc["component_quality_score_mean"].astype(np.float32).to_numpy(), 0.0, 1.0)
    risk = np.clip(inc["risk_score"].astype(np.float32).to_numpy(), 0.0, 1.0)
    b = float(args.da3_support_scale) * quality * np.sqrt(count_q).astype(np.float32) * np.power(1.0 - risk, float(args.da3_risk_power)).astype(np.float32)
    inc["q_final"] = b.astype(np.float32)
    inc["q_geo"] = quality.astype(np.float32)
    inc["q_mask"] = (1.0 - risk).astype(np.float32)
    inc["q_sem_final"] = np.clip(1.0 - inc["component_semantic_risk_mean"].astype(np.float32).to_numpy(), 0.0, 1.0)
    inc["visibility_prob"] = np.ones(int(inc.shape[0]), dtype=np.float32)
    inc["confidence_prob"] = np.ones(int(inc.shape[0]), dtype=np.float32)
    inc["broad_risk"] = np.zeros(int(inc.shape[0]), dtype=bool)
    inc["competing_mask_risk"] = np.zeros(int(inc.shape[0]), dtype=bool)
    inc["observation_role_code"] = np.full(int(inc.shape[0]), 2, dtype=np.int8)
    inc["chunk_id"] = "c0001"

    seg_rows = []
    for cid in component_ids.tolist():
        sub = inc[inc["component_id"].astype(np.int64) == int(cid)]
        seg_rows.append(
            {
                "scene_id": scene_id,
                "segment_id": f"{scene_id}:da3_semsoft_{int(cid):09d}",
                "carrier_id": int(-910_000_000_000 - int(cid)),
                "segment_role": "DA3_semsoft_support_component",
                "mean_q_final": float(sub["q_final"].mean()),
                "segment_broad_rate": 0.0,
                "segment_competing_rate": 0.0,
                "segment_local_index": int(local_by_component[int(cid)]),
            }
        )
    seg = pd.DataFrame(seg_rows)
    link = inc[
        [
            "scene_id",
            "chunk_id",
            "carrier_id",
            "segment_id",
            "segment_role",
            "frame_id",
            "mask_id",
            "observation_role_code",
            "q_geo",
            "q_mask",
            "q_sem_final",
            "q_final",
            "visibility_prob",
            "confidence_prob",
            "broad_risk",
            "competing_mask_risk",
            "frame_local_index",
            "mask_observation_index",
            "segment_local_index",
        ]
    ].copy()
    meta = {
        "da3_semsoft_enabled": True,
        "da3_variant_id": variant_id,
        "da3_component_count": int(seg.shape[0]),
        "da3_incidence_row_count": int(link.shape[0]),
        "da3_B_formula": "da3_support_scale * component_quality_score_mean * sqrt(log1p(component_mask_gaussian_count)/p95) * (1-risk_score)^da3_risk_power",
        "da3_support_scale": float(args.da3_support_scale),
        "da3_risk_power": float(args.da3_risk_power),
        "da3_uses_alpha_density_quality": True,
    }
    return seg, link, meta


def _safe_p10(values: np.ndarray, mask: np.ndarray | None = None) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if mask is not None:
        arr = arr[np.asarray(mask, dtype=bool)]
    if arr.size == 0:
        return 0.0
    return float(np.percentile(arr, 10))


def _valid_rate(feature: np.ndarray, mask: np.ndarray | None = None) -> float:
    if feature.size == 0:
        return 0.0
    valid = np.linalg.norm(feature, axis=1) > 0.0
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        if not np.any(m):
            return 0.0
        valid = valid[m]
    return float(np.mean(valid))


def _build_lookup_frame(mask_frame: np.ndarray, mask_label: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "frame_local_index": mask_frame.astype(np.int32, copy=False),
            "mask_id": mask_label.astype(np.int32, copy=False),
            "mask_observation_index": np.arange(mask_frame.shape[0], dtype=np.int64),
        }
    )


def _support_counts(mask_idx: np.ndarray, segment_idx: np.ndarray, mask_count: int) -> np.ndarray:
    if mask_idx.size == 0:
        return np.zeros((mask_count,), dtype=np.int64)
    pair = np.stack([mask_idx.astype(np.int64, copy=False), segment_idx.astype(np.int64, copy=False)], axis=1)
    uniq = np.unique(pair, axis=0)
    return np.bincount(uniq[:, 0], minlength=int(mask_count)).astype(np.int64, copy=False)


def _hard_negative_separation(
    *,
    feature: np.ndarray,
    incidence_by_mask: list[np.ndarray],
    segment_idx: np.ndarray,
    mask_frame: np.ndarray,
    mask_is_object: np.ndarray,
    mask_is_broad: np.ndarray,
    max_pairs: int,
    device: torch.device,
) -> tuple[float, dict[str, Any]]:
    pseudo, hard, same_frame, broad = _sample_mask_pairs(
        incidence_by_mask,
        segment_idx,
        mask_frame,
        mask_is_object,
        mask_is_broad,
        int(max_pairs),
    )
    pos = _pair_values_strict_leave_two_out_bucket_zeroed(feature, pseudo, device=device)
    neg = _pair_values_strict_leave_two_out_bucket_zeroed(feature, hard, device=device)
    same_vals = _pair_values_strict_leave_two_out_bucket_zeroed(feature, same_frame, device=device)
    broad_vals = _pair_values_strict_leave_two_out_bucket_zeroed(feature, broad, device=device)
    pos_mean = float(np.mean(pos)) if pos.size else 0.0
    neg_mean = float(np.mean(neg)) if neg.size else 0.0
    return pos_mean - neg_mean, {
        "pseudo_positive_affinity_mean": pos_mean,
        "hard_negative_affinity_mean": neg_mean,
        "hard_negative_separation": pos_mean - neg_mean,
        "pseudo_positive_pair_count": int(pos.size),
        "hard_negative_pair_count": int(neg.size),
        "same_frame_pair_count": int(same_vals.size),
        "broad_pair_count": int(broad_vals.size),
        "same_frame_competing_mask_affinity_p95": float(np.percentile(same_vals, 95)) if same_vals.size else 0.0,
        "broad_mask_affinity_p95": float(np.percentile(broad_vals, 95)) if broad_vals.size else 0.0,
    }


def _gate_row(scene_id: str, gate_id: str, passed: bool, observed: Any, required: Any, repair_direction: str) -> dict[str, Any]:
    return {
        "schema_version": "stream4d_v103_supp_r2_phaseP4_gate_row_v1",
        "phase_id": PHASE_ID,
        "scene_id": scene_id,
        "gate_id": gate_id,
        "pass": bool(passed),
        "observed": json.dumps(_jsonable(observed), sort_keys=True) if isinstance(observed, (dict, list, tuple)) else observed,
        "required": required,
        "repair_direction": repair_direction,
        "uses_gt": False,
        "uses_future": False,
    }


def _failure_row(scene_id: str, blocker: str, detail: Any, repair_direction: str) -> dict[str, Any]:
    return {
        "schema_version": "stream4d_v103_supp_r2_phaseP4_failure_row_v1",
        "phase_id": PHASE_ID,
        "scene_id": scene_id,
        "blocker": blocker,
        "detail": json.dumps(_jsonable(detail), sort_keys=True) if isinstance(detail, (dict, list, tuple)) else detail,
        "repair_direction": repair_direction,
    }


def _run_scene(
    *,
    scene_id: str,
    p3_root: Path,
    phase2_root: Path,
    output_root: Path,
    baselines: dict[str, float],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    t0 = time.time()
    scene_out = output_root / scene_id
    scene_out.mkdir(parents=True, exist_ok=True)
    spec = dict(SCENE_INPUTS[scene_id])
    spec["phase2_root"] = phase2_root
    diag, _unused_a, _unused_b, _arrays = _compute_scene_arrays(scene_id, spec, scene_out, int(args.cupy_device_id))
    mask_frame, mask_label, mask_is_object, mask_is_broad, _obs_lookup = _mask_observations(diag)
    mask_count = int(mask_frame.shape[0])
    frame_id_by_local = np.asarray(diag["frame_ids"], dtype=np.int32)
    frame_lookup = {int(frame_id): int(i) for i, frame_id in enumerate(frame_id_by_local.tolist())}
    mask_lookup = _build_lookup_frame(mask_frame, mask_label)

    seg = _read_scene_parquet(
        p3_root / "carrier_segment_rows.parquet",
        scene_id,
        [
            "scene_id",
            "segment_id",
            "carrier_id",
            "segment_role",
            "mean_q_final",
            "segment_broad_rate",
            "segment_competing_rate",
        ],
    )
    link = _read_scene_parquet(
        p3_root / "carrier_segment_observation_link_rows.parquet",
        scene_id,
        [
            "scene_id",
            "chunk_id",
            "carrier_id",
            "segment_id",
            "segment_role",
            "frame_id",
            "mask_id",
            "observation_role_code",
            "q_geo",
            "q_mask",
            "q_sem_final",
            "q_final",
            "visibility_prob",
            "confidence_prob",
            "broad_risk",
            "competing_mask_risk",
        ],
    )
    if seg.empty or link.empty:
        raise RuntimeError(f"empty P3 segment/link input for {scene_id}")

    seg = seg.sort_values(["segment_role", "segment_id"], kind="mergesort").reset_index(drop=True)
    seg["segment_local_index"] = np.arange(int(seg.shape[0]), dtype=np.int64)
    segment_map = seg.set_index("segment_id")["segment_local_index"]
    role_map = seg.set_index("segment_id")["segment_role"]
    link["segment_local_index"] = link["segment_id"].map(segment_map).astype(np.int64)
    link["segment_role"] = link["segment_id"].map(role_map).astype(str)
    link["frame_local_index"] = link["frame_id"].map(frame_lookup).astype(np.int32)
    missing_frame = int(link["frame_local_index"].isna().sum()) if link["frame_local_index"].isna().any() else 0
    if missing_frame:
        raise RuntimeError(f"{scene_id} link rows have frame ids not present in Phase2 diag: {missing_frame}")
    link = link.merge(mask_lookup, on=["frame_local_index", "mask_id"], how="left", sort=False)
    missing_mask = int(link["mask_observation_index"].isna().sum())
    if missing_mask:
        raise RuntimeError(f"{scene_id} link rows have missing mask observation mapping: {missing_mask}")
    link["mask_observation_index"] = link["mask_observation_index"].astype(np.int64)

    da3_meta: dict[str, Any] = {
        "da3_semsoft_enabled": False,
        "da3_variant_id": "",
        "da3_component_count": 0,
        "da3_incidence_row_count": 0,
    }
    da3_root_arg = str(getattr(args, "da3_semsoft_root", "")).strip()
    if da3_root_arg:
        da3_root = _project(da3_root_arg)
        da3_seg, da3_link, da3_meta = _da3_support_for_scene(
            da3_root=da3_root,
            scene_id=scene_id,
            next_local_index=int(seg.shape[0]),
            mask_lookup=mask_lookup,
            args=args,
        )
        if not da3_seg.empty and not da3_link.empty:
            seg = pd.concat([seg, da3_seg], ignore_index=True, sort=False)
            link = pd.concat([link, da3_link], ignore_index=True, sort=False)

    link["B_sigma_a"] = link["q_final"].astype(np.float32)
    link = link[np.isfinite(link["B_sigma_a"].to_numpy(np.float32, copy=False)) & (link["B_sigma_a"].to_numpy(np.float32, copy=False) > 0.0)].copy()

    segment_idx = link["segment_local_index"].to_numpy(np.int64, copy=False)
    mask_idx = link["mask_observation_index"].to_numpy(np.int64, copy=False)
    frame_local = link["frame_local_index"].to_numpy(np.int64, copy=False)
    mask_id = link["mask_id"].to_numpy(np.int64, copy=False)
    b_ia = link["B_sigma_a"].to_numpy(np.float32, copy=False)
    incidence = np.stack(
        [
            segment_idx.astype(np.float64),
            mask_idx.astype(np.float64),
            frame_local.astype(np.float64),
            mask_id.astype(np.float64),
            b_ia.astype(np.float64),
        ],
        axis=1,
    )

    unique_frame_segment = link[["frame_local_index", "segment_local_index"]].drop_duplicates()
    visible_by_frame = (
        unique_frame_segment.groupby("frame_local_index", sort=False).size().reindex(np.arange(len(frame_id_by_local)), fill_value=0).to_numpy(np.int64)
    )
    weights, weight_meta = _mask_weights(
        incidence=incidence,
        mask_count=mask_count,
        mask_frame=mask_frame,
        mask_is_object=mask_is_object,
        mask_is_broad=mask_is_broad,
        visible_reliable_by_frame=visible_by_frame,
        specificity_mode=str(args.specificity_mode),
        specificity_alpha=float(args.specificity_alpha),
        no_idf=False,
    )
    primitive_feature, primitive_runtime = _countsketch(
        incidence,
        weights,
        int(seg.shape[0]),
        int(args.sketch_dim),
        device,
    )
    subset, exact = _exact_dense_subset(incidence, weights, int(seg.shape[0]), mask_count, int(args.exact_subset_size))
    p95_error, max_error = _pair_error(exact, primitive_feature, subset)

    incidence_by_mask = [np.flatnonzero(mask_idx == m).astype(np.int64) for m in range(mask_count)]
    raw = _build_raw_sketch(segment_idx, mask_idx, b_ia, weights, int(seg.shape[0]), int(args.sketch_dim), device)
    alpha = b_ia.astype(np.float32, copy=False)
    carrier_broad = seg["segment_broad_rate"].to_numpy(np.float32, copy=False)
    mask_feature, mask_runtime = _pool_features(
        variant_id="P0_mean_reliability_weighted",
        raw=raw,
        incidence_by_mask=incidence_by_mask,
        carrier_idx=segment_idx,
        mask_idx=mask_idx,
        b_ia=b_ia,
        alpha=alpha,
        carrier_broad=carrier_broad,
        mask_weight=weights,
        topk=int(args.topk_segments),
        trim_quantile=float(args.trim_quantile),
        device=device,
    )
    no_loo_feature, no_loo_runtime = _pool_features(
        variant_id="P4_no_leave_one_out_control",
        raw=raw,
        incidence_by_mask=incidence_by_mask,
        carrier_idx=segment_idx,
        mask_idx=mask_idx,
        b_ia=b_ia,
        alpha=alpha,
        carrier_broad=carrier_broad,
        mask_weight=weights,
        topk=int(args.topk_segments),
        trim_quantile=float(args.trim_quantile),
        device=device,
    )
    sep, pair_meta = _hard_negative_separation(
        feature=mask_feature,
        incidence_by_mask=incidence_by_mask,
        segment_idx=segment_idx,
        mask_frame=mask_frame,
        mask_is_object=mask_is_object,
        mask_is_broad=mask_is_broad,
        max_pairs=int(args.max_pair_rows),
        device=device,
    )

    role_values = link["segment_role"].astype(str).to_numpy()
    anchor_rows = role_values == "A_anchor_segment"
    d4rt_support_rows = role_values == "S_support_segment"
    da3_support_rows = role_values == "DA3_semsoft_support_component"
    support_rows = d4rt_support_rows | da3_support_rows
    anchor_support = _support_counts(mask_idx[anchor_rows], segment_idx[anchor_rows], mask_count)
    d4rt_support = _support_counts(mask_idx[d4rt_support_rows], segment_idx[d4rt_support_rows], mask_count)
    da3_support = _support_counts(mask_idx[da3_support_rows], segment_idx[da3_support_rows], mask_count)
    support_support = _support_counts(mask_idx[support_rows], segment_idx[support_rows], mask_count)
    all_support = _support_counts(mask_idx, segment_idx, mask_count)
    mass = np.sqrt(weights[mask_idx].astype(np.float64)) * np.abs(b_ia.astype(np.float64))
    total_mass = float(np.sum(mass))
    broad_mass = float(np.sum(mass[mask_is_broad[mask_idx]])) if total_mass > 0 else 0.0
    object_mass = float(np.sum(mass[mask_is_object[mask_idx]])) if total_mass > 0 else 0.0
    broad_ratio = float(broad_mass / max(total_mass, 1e-12))
    object_ratio = float(object_mass / max(total_mass, 1e-12))
    role_counts = seg["segment_role"].value_counts().to_dict()
    role_counts = {str(k): int(v) for k, v in role_counts.items()}
    mask_valid = np.linalg.norm(mask_feature, axis=1) > 0.0
    no_loo_valid = np.linalg.norm(no_loo_feature, axis=1) > 0.0
    both_valid = mask_valid & no_loo_valid
    loo_cos = float(np.mean(np.sum(mask_feature[both_valid] * no_loo_feature[both_valid], axis=1))) if np.any(both_valid) else 0.0

    metric = {
        "schema_version": "stream4d_v103_supp_r2_phaseP4_segment_feature_summary_row_v1",
        "phase_id": PHASE_ID,
        "scene_id": scene_id,
        "primitive_count_by_role": json.dumps(role_counts, sort_keys=True),
        "segment_primitive_count": int(seg.shape[0]),
        "incidence_row_count": int(incidence.shape[0]),
        "mask_observation_count": mask_count,
        "mask_with_min_anchor_segment_rate": float(np.mean(anchor_support[mask_is_object] >= int(args.min_segments_per_object_mask))) if np.any(mask_is_object) else 0.0,
        "mask_with_min_support_segment_rate": float(np.mean(support_support[mask_is_object] >= int(args.min_segments_per_object_mask))) if np.any(mask_is_object) else 0.0,
        "object_like_mask_anchor_support_p10": _safe_p10(anchor_support, mask_is_object),
        "object_like_mask_support_support_p10": _safe_p10(support_support, mask_is_object),
        "object_like_mask_d4rt_support_support_p10": _safe_p10(d4rt_support, mask_is_object),
        "object_like_mask_da3_support_support_p10": _safe_p10(da3_support, mask_is_object),
        "object_like_mask_all_support_p10": _safe_p10(all_support, mask_is_object),
        "da3_semsoft_mask_support_rate": float(np.mean(da3_support[mask_is_object] > 0)) if np.any(mask_is_object) else 0.0,
        "broad_contribution_ratio": broad_ratio,
        "object_like_contribution_ratio": object_ratio,
        "previous_broad_contribution_ratio": float(baselines["previous_broad_contribution_ratio"]),
        "exact_vs_sketch_cosine_p95_error": float(p95_error),
        "exact_vs_sketch_cosine_max_error": float(max_error),
        "exact_subset_count": int(subset.shape[0]),
        "leave_one_out_feature_valid_rate": _valid_rate(mask_feature, mask_is_object),
        "leave_one_out_feature_valid_rate_all": _valid_rate(mask_feature),
        "no_leave_one_out_feature_valid_rate_all": _valid_rate(no_loo_feature),
        "leave_one_out_vs_no_loo_cosine_mean": loo_cos,
        "hard_negative_separation": float(sep),
        "phase5_d4rt48mix_baseline_hard_negative_separation": float(baselines["phase5_d4rt48mix_strict_l2o_hard_negative_separation"]),
        "phase4_d4rt48mix_baseline_hard_negative_separation": float(baselines["phase4_d4rt48mix_hard_negative_separation"]),
        "primitive_runtime_sec": float(primitive_runtime),
        "mask_pool_runtime_sec": float(mask_runtime),
        "no_leave_one_out_pool_runtime_sec": float(no_loo_runtime),
        "uses_gt": False,
        "uses_future": False,
        **da3_meta,
        **{f"mask_weight_{k}": v for k, v in weight_meta.items()},
        **pair_meta,
    }
    parity = {
        "schema_version": "stream4d_v103_supp_r2_phaseP4_segment_affinity_parity_row_v1",
        "phase_id": PHASE_ID,
        "scene_id": scene_id,
        "exact_subset_count": int(subset.shape[0]),
        "exact_vs_sketch_cosine_p95_error": float(p95_error),
        "exact_vs_sketch_cosine_max_error": float(max_error),
        "sketch_dim": int(args.sketch_dim),
        "uses_gt": False,
        "uses_future": False,
    }

    gates = [
        _gate_row(
            scene_id,
            "exact_vs_sketch_cosine_p95_error_le_0p01",
            float(metric["exact_vs_sketch_cosine_p95_error"]) <= float(args.exact_p95_threshold),
            metric["exact_vs_sketch_cosine_p95_error"],
            f"<={args.exact_p95_threshold}",
            "Increase sketch_dim or repair incidence hash/parity if this fails.",
        ),
        _gate_row(
            scene_id,
            "leave_one_out_feature_valid_rate_ge_0p95",
            float(metric["leave_one_out_feature_valid_rate"]) >= float(args.leave_one_out_valid_threshold),
            metric["leave_one_out_feature_valid_rate"],
            f">={args.leave_one_out_valid_threshold}",
            "Repair segment coverage or mask pooling leave-one-out path.",
        ),
        _gate_row(
            scene_id,
            "object_like_mask_support_support_p10_gt_0",
            float(metric["object_like_mask_support_support_p10"]) > 0.0,
            metric["object_like_mask_support_support_p10"],
            ">0",
            "Return to P3 support segment construction if support segments do not cover object-like masks.",
        ),
        _gate_row(
            scene_id,
            "broad_contribution_ratio_le_previous_plus_0p05",
            float(metric["broad_contribution_ratio"]) <= float(metric["previous_broad_contribution_ratio"]) + 0.05,
            {"current": metric["broad_contribution_ratio"], "previous": metric["previous_broad_contribution_ratio"]},
            "current <= previous + 0.05",
            "Tighten broad veto or role-downgrade broad segment incidence.",
        ),
        _gate_row(
            scene_id,
            "hard_negative_separation_not_worse_than_phase5_d4rt48mix_baseline",
            float(metric["hard_negative_separation"]) + 1e-9 >= float(metric["phase5_d4rt48mix_baseline_hard_negative_separation"]),
            {"current": metric["hard_negative_separation"], "baseline": metric["phase5_d4rt48mix_baseline_hard_negative_separation"]},
            "current >= Phase5 D4RT48Mix strict L2O baseline",
            "Try frame-centered pooling, stricter source agreement, or anchor/support role separation before P5.",
        ),
    ]
    failures = [
        _failure_row(scene_id, str(gate["gate_id"]), gate["observed"], str(gate["repair_direction"]))
        for gate in gates
        if not bool(gate["pass"])
    ]

    incidence_df = pd.DataFrame(
        {
            "schema_version": "stream4d_v103_supp_r2_phaseP4_segment_primitive_mask_incidence_row_v1",
            "phase_id": PHASE_ID,
            "scene_id": scene_id,
            "chunk_id": link["chunk_id"].astype(str).to_numpy(),
            "segment_local_index": segment_idx,
            "segment_id": link["segment_id"].astype(str).to_numpy(),
            "segment_role": link["segment_role"].astype(str).to_numpy(),
            "carrier_id": link["carrier_id"].astype(np.int64).to_numpy(),
            "mask_observation_index": mask_idx,
            "frame_local_index": frame_local,
            "frame_id": link["frame_id"].astype(np.int32).to_numpy(),
            "mask_id": mask_id,
            "B_sigma_a": b_ia,
            "q_final": link["q_final"].astype(np.float32).to_numpy(),
            "q_geo": link["q_geo"].astype(np.float32).to_numpy(),
            "q_mask": link["q_mask"].astype(np.float32).to_numpy(),
            "q_sem_final": link["q_sem_final"].astype(np.float32).to_numpy(),
            "visibility_prob": link["visibility_prob"].astype(np.float32).to_numpy(),
            "confidence_prob": link["confidence_prob"].astype(np.float32).to_numpy(),
            "mask_is_object_like": mask_is_object[mask_idx],
            "mask_is_broad": mask_is_broad[mask_idx],
            "uses_gt": np.zeros(mask_idx.shape[0], dtype=bool),
            "uses_future": np.zeros(mask_idx.shape[0], dtype=bool),
        }
    )

    primitive_payload = {
        "segment_id": seg["segment_id"].astype(str).to_numpy(),
        "segment_role": seg["segment_role"].astype(str).to_numpy(),
        "carrier_id": seg["carrier_id"].to_numpy(np.int64, copy=False),
        "feature": primitive_feature.astype(np.float16, copy=False),
        "feature_norm_source_dtype": "float32",
        "segment_broad_rate": seg["segment_broad_rate"].to_numpy(np.float32, copy=False),
        "segment_competing_rate": seg["segment_competing_rate"].to_numpy(np.float32, copy=False),
        "incidence_row_count": int(incidence.shape[0]),
        "metrics": metric,
    }
    mask_payload = {
        "mask_observation_index": np.arange(mask_count, dtype=np.int64),
        "mask_frame": mask_frame.astype(np.int64, copy=False),
        "frame_id": frame_id_by_local[mask_frame.astype(np.int64, copy=False)].astype(np.int32, copy=False),
        "mask_label": mask_label.astype(np.int64, copy=False),
        "mask_is_object_like": mask_is_object.astype(bool, copy=False),
        "mask_is_broad": mask_is_broad.astype(bool, copy=False),
        "mask_weight": weights.astype(np.float32, copy=False),
        "anchor_segment_support_count": anchor_support.astype(np.int64, copy=False),
        "d4rt_support_segment_support_count": d4rt_support.astype(np.int64, copy=False),
        "da3_support_component_support_count": da3_support.astype(np.int64, copy=False),
        "support_segment_support_count": support_support.astype(np.int64, copy=False),
        "all_segment_support_count": all_support.astype(np.int64, copy=False),
        "feature": mask_feature.astype(np.float16, copy=False),
        "feature_norm_source_dtype": "float32",
        "metrics": metric,
    }
    metric["runtime_sec"] = time.time() - t0
    return primitive_payload, mask_payload, incidence_df, [parity], [metric], gates + failures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="v103 supplement R2 Phase P4 segment-level primitive affinity feature.")
    parser.add_argument("--p3-root", default=str(DEFAULT_P3_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--phase4-baseline-root", default=str(DEFAULT_PHASE4_BASELINE))
    parser.add_argument("--phase5-baseline-root", default=str(DEFAULT_PHASE5_BASELINE))
    parser.add_argument("--scene0011-phase2-root", default=str(DEFAULT_PHASE2_ROOT_BY_SCENE["scene0011_00"]))
    parser.add_argument("--scene0050-phase2-root", default=str(DEFAULT_PHASE2_ROOT_BY_SCENE["scene0050_00"]))
    parser.add_argument("--da3-semsoft-root", default="")
    parser.add_argument("--da3-support-scale", type=float, default=0.50)
    parser.add_argument("--da3-risk-power", type=float, default=1.0)
    parser.add_argument("--torch-device", default="auto")
    parser.add_argument("--cupy-device-id", type=int, default=0)
    parser.add_argument("--sketch-dim", type=int, default=256)
    parser.add_argument("--exact-subset-size", type=int, default=2048)
    parser.add_argument("--max-pair-rows", type=int, default=4096)
    parser.add_argument("--topk-segments", type=int, default=64)
    parser.add_argument("--trim-quantile", type=float, default=0.10)
    parser.add_argument("--specificity-mode", default="idf_object_preserve_downweight")
    parser.add_argument("--specificity-alpha", type=float, default=0.5)
    parser.add_argument("--min-segments-per-object-mask", type=int, default=4)
    parser.add_argument("--exact-p95-threshold", type=float, default=0.01)
    parser.add_argument("--leave-one-out-valid-threshold", type=float, default=0.95)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    t0 = time.time()
    args = build_parser().parse_args()
    p3_root = _project(args.p3_root)
    out = _project(args.output_root)
    if out.exists() and any(out.iterdir()) and not args.force:
        raise SystemExit(f"output root already exists and is non-empty; pass --force: {_rel(out)}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "last_command.txt").write_text(" ".join([sys.executable, str(Path(__file__)), *sys.argv[1:]]) + "\n", encoding="utf-8")
    p3_summary = _read_json(p3_root / "summary.json")
    if not bool(p3_summary.get("phaseP3_pass", False)):
        raise RuntimeError(f"P3 did not pass: {p3_root / 'summary.json'}")
    baselines = _load_baselines(_project(args.phase4_baseline_root), _project(args.phase5_baseline_root))
    device = _torch_device(str(args.torch_device))
    phase2_roots = {
        "scene0011_00": _project(args.scene0011_phase2_root),
        "scene0050_00": _project(args.scene0050_phase2_root),
    }
    da3_enabled = bool(str(args.da3_semsoft_root).strip())
    da3_root_rel = _rel(_project(args.da3_semsoft_root)) if da3_enabled else ""

    primitive_scenes: dict[str, Any] = {}
    mask_scenes: dict[str, Any] = {}
    parity_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    performance_rows: list[dict[str, Any]] = []
    incidence_writer: pq.ParquetWriter | None = None
    try:
        for scene_id in ["scene0011_00", "scene0050_00"]:
            scene_t0 = time.time()
            primitive_payload, mask_payload, incidence_df, p_rows, f_rows, gf_rows = _run_scene(
                scene_id=scene_id,
                p3_root=p3_root,
                phase2_root=phase2_roots[scene_id],
                output_root=out,
                baselines=baselines[scene_id],
                args=args,
                device=device,
            )
            primitive_scenes[scene_id] = {
                key: torch.as_tensor(value) if isinstance(value, np.ndarray) and value.dtype.kind in {"f", "i", "b"} else value
                for key, value in primitive_payload.items()
                if key != "metrics"
            }
            primitive_scenes[scene_id]["metrics"] = primitive_payload["metrics"]
            mask_scenes[scene_id] = {
                key: torch.as_tensor(value) if isinstance(value, np.ndarray) and value.dtype.kind in {"f", "i", "b"} else value
                for key, value in mask_payload.items()
                if key != "metrics"
            }
            mask_scenes[scene_id]["metrics"] = mask_payload["metrics"]
            parity_rows.extend(p_rows)
            feature_rows.extend(f_rows)
            for row in gf_rows:
                if "gate_id" in row:
                    gate_rows.append(row)
                else:
                    failure_rows.append(row)
            table = pa.Table.from_pandas(incidence_df, preserve_index=False)
            if incidence_writer is None:
                incidence_writer = pq.ParquetWriter(out / "segment_primitive_mask_incidence.parquet", table.schema, compression="zstd")
            incidence_writer.write_table(table)
            performance_rows.append(
                {
                    "schema_version": "stream4d_v103_supp_r2_phaseP4_performance_row_v1",
                    "phase_id": PHASE_ID,
                    "scene_id": scene_id,
                    "runtime_sec": time.time() - scene_t0,
                    "torch_device": str(device),
                    "cupy_device_id": int(args.cupy_device_id),
                    "uses_gt": False,
                    "uses_future": False,
                }
            )
            del incidence_df
    finally:
        if incidence_writer is not None:
            incidence_writer.close()

    primitive_artifact = {
        "schema_version": "stream4d_v103_supp_r2_phaseP4_segment_primitive_affinity_feature_v1",
        "phase_id": PHASE_ID,
        "sketch_seed": 10317,
        "sketch_dim": int(args.sketch_dim),
        "B_definition": "B_sigma_a=q_final, where P2 q_final=q_geo*q_mask*q_sem_final and q_geo=visibility*confidence*self_jitter_q. This avoids double-counting visibility/confidence.",
        "da3_semsoft_enabled": da3_enabled,
        "da3_semsoft_root": da3_root_rel,
        "da3_B_definition": "If enabled, DA3 semantic-soft clean components are support primitives only: B=da3_support_scale*component_quality_score_mean*sqrt(log1p(mask_gaussian_count)/p95)*(1-risk_score)^da3_risk_power.",
        "uses_gt": False,
        "uses_future": False,
        "scenes": primitive_scenes,
    }
    mask_artifact = {
        "schema_version": "stream4d_v103_supp_r2_phaseP4_segment_mask_level_feature_v1",
        "phase_id": PHASE_ID,
        "sketch_seed": 10317,
        "sketch_dim": int(args.sketch_dim),
        "pooling_policy": "P0_mean_reliability_weighted leave-one-out over segment primitive sketches; support-only is recorded but not allowed to trigger union in P5.",
        "da3_semsoft_enabled": da3_enabled,
        "da3_semsoft_root": da3_root_rel,
        "uses_gt": False,
        "uses_future": False,
        "scenes": mask_scenes,
    }
    primitive_path = out / "segment_primitive_affinity_feature.pt"
    mask_path = out / "segment_mask_level_feature.pt"
    torch.save(primitive_artifact, primitive_path)
    torch.save(mask_artifact, mask_path)

    _write_csv(out / "segment_affinity_parity_rows.csv", parity_rows)
    _write_csv(out / "segment_feature_summary_rows.csv", feature_rows)
    _write_csv(out / "segment_gate_rows.csv", gate_rows)
    _write_csv(out / "failure_rows.csv", failure_rows)
    _write_csv(out / "performance_rows.csv", performance_rows)
    artifact_rows = [
        {
            "schema_version": "stream4d_v103_supp_r2_phaseP4_artifact_row_v1",
            "phase_id": PHASE_ID,
            "artifact_role": role,
            "path": _rel(path),
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else 0,
            "uses_gt": False,
            "uses_future": False,
        }
        for role, path in [
            ("segment_primitive_mask_incidence", out / "segment_primitive_mask_incidence.parquet"),
            ("segment_primitive_affinity_feature", primitive_path),
            ("segment_mask_level_feature", mask_path),
            ("segment_affinity_parity_rows", out / "segment_affinity_parity_rows.csv"),
            ("segment_feature_summary_rows", out / "segment_feature_summary_rows.csv"),
        ]
    ]
    _write_csv(out / "artifact_rows.csv", artifact_rows)

    all_pass = all(bool(row["pass"]) for row in gate_rows)
    decision = "PASS_ENTER_PHASEP5_MASK_GRAPH_EDGE_INTERVENTION" if all_pass else "NO_GO_REPAIR_PHASEP4_SEGMENT_AFFINITY"
    summary = {
        "schema_version": "stream4d_v103_supp_r2_phaseP4_summary_v1",
        "phase_id": PHASE_ID,
        "decision": decision,
        "phaseP4_pass": bool(all_pass),
        "failure_count": int(len(failure_rows)),
        "p3_root": _rel(p3_root),
        "scene_ids": ["scene0011_00", "scene0050_00"],
        "sketch_dim": int(args.sketch_dim),
        "da3_semsoft_enabled": da3_enabled,
        "da3_semsoft_root": da3_root_rel,
        "da3_support_scale": float(args.da3_support_scale),
        "da3_risk_power": float(args.da3_risk_power),
        "torch_device": str(device),
        "runtime_sec": time.time() - t0,
        "outputs": {
            "segment_primitive_mask_incidence": _rel(out / "segment_primitive_mask_incidence.parquet"),
            "segment_primitive_affinity_feature": _rel(primitive_path),
            "segment_mask_level_feature": _rel(mask_path),
            "segment_affinity_parity_rows": _rel(out / "segment_affinity_parity_rows.csv"),
            "segment_feature_summary_rows": _rel(out / "segment_feature_summary_rows.csv"),
            "segment_gate_rows": _rel(out / "segment_gate_rows.csv"),
            "failure_rows": _rel(out / "failure_rows.csv"),
            "artifact_rows": _rel(out / "artifact_rows.csv"),
            "performance_rows": _rel(out / "performance_rows.csv"),
            "summary": _rel(out / "summary.json"),
        },
        "truthfulness_note": "P4 constructs segment-level primitive and mask-level affinity features only. It does not create objects, run AP, or use GT/future frames.",
        "uses_gt": False,
        "uses_future": False,
    }
    _write_json(out / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if all_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
