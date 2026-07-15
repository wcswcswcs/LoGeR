#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from diagnose_v103_phase9g_da3_seed_gaussian_neighborhood import (  # noqa: E402
    AUDIT_ROOT,
    SCENE_SPECS,
    _project_phase,
    _rel,
    _write_csv,
    _write_json,
)


PHASE_ID = "v103_phase9n_da3_bridge_pair_fused_phase4"
DEFAULT_PHASE4_ROOT = AUDIT_ROOT / "v103_phase4_positive_core_affinity_q5c_repair5_r12_dual_role"
DEFAULT_PHASE9E_ROOT = AUDIT_ROOT / "v103_phase9e_d4rt_anchor_da3_induced_carriers_r5_target_object"
DEFAULT_OUT = AUDIT_ROOT / "v103_phase9n_da3_bridge_pair_fused_phase4_r1"
SKETCH_SEED = 10317


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


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _project(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if p.parts and p.parts[0] == "Stream3D":
        return REPO_ROOT / p
    return STREAM3D_ROOT / p


def _countsketch(
    carrier_idx: np.ndarray,
    mask_idx: np.ndarray,
    b_ia: np.ndarray,
    mask_weight: np.ndarray,
    carrier_count: int,
    sketch_dim: int,
    device: torch.device,
) -> np.ndarray:
    if carrier_count == 0 or carrier_idx.size == 0:
        return np.zeros((carrier_count, sketch_dim), dtype=np.float32)
    with torch.no_grad():
        c_t = torch.as_tensor(carrier_idx, dtype=torch.long, device=device)
        m_t = torch.as_tensor(mask_idx, dtype=torch.long, device=device)
        b_t = torch.as_tensor(b_ia, dtype=torch.float32, device=device)
        w_t = torch.as_tensor(mask_weight, dtype=torch.float32, device=device)
        bucket = ((m_t * 2654435761 + SKETCH_SEED) % int(sketch_dim)).to(torch.long)
        sign = torch.where(((m_t * 1103515245 + SKETCH_SEED) % 2) == 0, 1.0, -1.0).to(torch.float32)
        values = torch.sqrt(w_t[m_t]) * b_t * sign
        out = torch.zeros((int(carrier_count), int(sketch_dim)), dtype=torch.float32, device=device)
        out.index_put_((c_t, bucket), values, accumulate=True)
        out = torch.nn.functional.normalize(out, p=2, dim=1, eps=1e-12)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return out.detach().cpu().numpy().astype(np.float32, copy=False)


def _scene_phase2_root(scene_id: str, args: argparse.Namespace) -> Path:
    attr = {
        "scene0011_00": "scene0011_phase2_root",
        "scene0050_00": "scene0050_phase2_root",
    }[scene_id]
    override = getattr(args, attr, None)
    if override:
        return _project(override)
    spec = dict(SCENE_SPECS[scene_id])
    return _project(spec["phase2_root"])


def _scene_pair_source(scene_id: str, args: argparse.Namespace) -> str:
    attr = {
        "scene0011_00": "scene0011_pair_source",
        "scene0050_00": "scene0050_pair_source",
    }[scene_id]
    return str(getattr(args, attr, None) or args.pair_source)


def _scene_pair_filter(scene_id: str, args: argparse.Namespace) -> str:
    attr = {
        "scene0011_00": "scene0011_pair_filter",
        "scene0050_00": "scene0050_pair_filter",
    }[scene_id]
    return str(getattr(args, attr, None) or args.pair_filter)


def _phase2_frame_ids(scene_id: str, args: argparse.Namespace) -> list[int]:
    summary = _read_json(_scene_phase2_root(scene_id, args) / "summary.json")
    return [int(v) for v in summary["frame_ids"]]


def _pair_reliability_array(df: pd.DataFrame) -> np.ndarray:
    bridge = df["final_bridge_score"].astype(float).to_numpy()
    sem = df["semantic_residual_cosine"].astype(float).to_numpy()
    broad = df["broad_contamination_score"].astype(float).to_numpy()
    sem = np.where(np.isfinite(sem), sem, 1.0)
    return np.clip(bridge * np.maximum(0.0, sem) * np.maximum(0.0, 1.0 - broad), 0.0, 1.0)


def _accepted_pair_path(phase9e_root: Path, scene_id: str, pair_source: str) -> Path:
    if pair_source.startswith("variant:"):
        variant_id = pair_source.split(":", 1)[1]
        return phase9e_root / scene_id / "accepted_pair_rows_by_variant" / f"{variant_id}_accepted_pair_rows.parquet"
    return phase9e_root / scene_id / f"{pair_source}_accepted_pair_rows.parquet"


def _load_pairs(phase9e_root: Path, scene_id: str, pair_source: str, pair_filter: str, min_pair_reliability: float) -> pd.DataFrame:
    pair_path = _accepted_pair_path(phase9e_root, scene_id, pair_source)
    if not pair_path.exists():
        raise FileNotFoundError(pair_path)
    df = pd.read_parquet(pair_path)
    rel = _pair_reliability_array(df)
    if pair_filter == "both_object_nonbroad":
        keep = (
            df["mask_a_is_object_like"].astype(bool)
            & ~df["mask_a_is_broad"].astype(bool)
            & df["mask_b_is_object_like"].astype(bool)
            & ~df["mask_b_is_broad"].astype(bool)
        )
        return df[keep].copy()
    if pair_filter == "no_broad":
        keep = ~df["mask_a_is_broad"].astype(bool) & ~df["mask_b_is_broad"].astype(bool)
        return df[keep].copy()
    if pair_filter == "high_reliability":
        keep = rel >= float(min_pair_reliability)
        return df[keep].copy()
    if pair_filter == "no_broad_or_high_reliability":
        no_broad = ~df["mask_a_is_broad"].astype(bool) & ~df["mask_b_is_broad"].astype(bool)
        keep = no_broad | (rel >= float(min_pair_reliability))
        return df[keep].copy()
    if pair_filter == "all_clean_pairs":
        return df.copy()
    raise ValueError(f"unsupported pair_filter={pair_filter}")


def _pair_reliability(row: dict[str, Any]) -> float:
    bridge = float(row.get("final_bridge_score", 0.0))
    sem = row.get("semantic_residual_cosine", 1.0)
    sem_val = float(sem) if sem is not None and np.isfinite(float(sem)) else 1.0
    broad = float(row.get("broad_contamination_score", 0.0))
    return float(np.clip(bridge * max(0.0, sem_val) * max(0.0, 1.0 - broad), 0.0, 1.0))


def _run_scene(scene_id: str, args: argparse.Namespace, device: torch.device) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    phase4_root = _project(args.phase4_root)
    phase9e_root = _project_phase(args.phase9e_root)
    out = _project_phase(args.output_root)
    scene_out = out / scene_id
    scene_out.mkdir(parents=True, exist_ok=True)

    base = torch.load(phase4_root / scene_id / "primitive_incidence_sparse.pt", map_location="cpu")
    base_carrier_id = base["carrier_id"].cpu().numpy().astype(np.int64)
    base_local = base["carrier_local_index"].cpu().numpy().astype(np.int64)
    base_mask_idx = base["mask_observation_index"].cpu().numpy().astype(np.int64)
    base_frame = base["frame_local_index"].cpu().numpy().astype(np.int64)
    base_mask_id = base["mask_id"].cpu().numpy().astype(np.int64)
    base_b = base["B_ia"].cpu().numpy().astype(np.float32)
    mask_frame = base["mask_frame"].cpu().numpy().astype(np.int64)
    mask_label = base["mask_label"].cpu().numpy().astype(np.int64)
    mask_is_object = base["mask_is_object_like"].cpu().numpy().astype(bool)
    mask_is_broad = base["mask_is_broad"].cpu().numpy().astype(bool)
    mask_weight = base["mask_weight"].cpu().numpy().astype(np.float32)

    pair_source = _scene_pair_source(scene_id, args)
    pair_filter = _scene_pair_filter(scene_id, args)
    phase2_root = _scene_phase2_root(scene_id, args)
    frame_ids = _phase2_frame_ids(scene_id, args)
    obs_to_idx = {
        f"{scene_id}:{int(frame_ids[int(frame_local)])}:{int(label)}": int(idx)
        for idx, (frame_local, label) in enumerate(zip(mask_frame.tolist(), mask_label.tolist()))
        if 0 <= int(frame_local) < len(frame_ids)
    }
    pairs_all = pd.read_parquet(_accepted_pair_path(phase9e_root, scene_id, pair_source))
    pairs = _load_pairs(
        phase9e_root,
        scene_id,
        pair_source,
        pair_filter,
        float(args.min_pair_reliability),
    )

    pair_rows: list[dict[str, Any]] = []
    incidence_rows: list[list[float]] = []
    rel_rows: list[float] = []
    broad_rows: list[float] = []
    missing = 0
    for pair_idx, row in enumerate(pairs.to_dict("records")):
        obs_a = str(row["mask_a_observation_id"])
        obs_b = str(row["mask_b_observation_id"])
        idx_a = obs_to_idx.get(obs_a)
        idx_b = obs_to_idx.get(obs_b)
        if idx_a is None or idx_b is None or idx_a == idx_b:
            missing += 1
            continue
        reliability = _pair_reliability(row)
        if reliability <= 0.0 or not math.isfinite(reliability):
            continue
        local = int(base_carrier_id.shape[0] + len(rel_rows))
        b_a = reliability
        b_b = reliability
        incidence_rows.append([float(local), float(idx_a), float(mask_frame[idx_a]), float(mask_label[idx_a]), float(b_a)])
        incidence_rows.append([float(local), float(idx_b), float(mask_frame[idx_b]), float(mask_label[idx_b]), float(b_b)])
        rel_rows.append(reliability)
        broad_risk = max(
            float(row.get("broad_contamination_score", 0.0)),
            float(bool(row.get("mask_a_is_broad", False))),
            float(bool(row.get("mask_b_is_broad", False))),
        )
        broad_rows.append(float(np.clip(broad_risk, 0.0, 1.0)))
        pair_rows.append(
            {
                "schema_version": "stream4d_v103_phase9n_bridge_pair_primitive_row_v1",
                "phase_id": PHASE_ID,
                "scene_id": scene_id,
                "synthetic_carrier_local_index": local,
                "synthetic_carrier_id": int(-910_000_000_000 - len(rel_rows)),
                "mask_a_observation_id": obs_a,
                "mask_b_observation_id": obs_b,
                "mask_a_observation_index": int(idx_a),
                "mask_b_observation_index": int(idx_b),
                "frame_a_local": int(mask_frame[idx_a]),
                "frame_b_local": int(mask_frame[idx_b]),
                "mask_a_id": int(mask_label[idx_a]),
                "mask_b_id": int(mask_label[idx_b]),
                "B_ia": float(reliability),
                "carrier_reliability": float(reliability),
                "carrier_broad_risk": float(broad_rows[-1]),
                "final_bridge_score": float(row.get("final_bridge_score", 0.0)),
                "semantic_residual_cosine": float(row.get("semantic_residual_cosine", 0.0)),
                "broad_contamination_score": float(row.get("broad_contamination_score", 0.0)),
                "phase9e_variant_id": str(row.get("phase9e_variant_id", "")),
                "diagnostic_different_gt": bool(row.get("diagnostic_different_gt", False)),
                "diagnostic_same_gt": bool(row.get("diagnostic_same_gt", False)),
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic": True,
            }
        )

    pair_incidence = np.asarray(incidence_rows, dtype=np.float64).reshape(-1, 5)
    if pair_incidence.size:
        fused_incidence = np.concatenate(
            [
                np.stack([base_local, base_mask_idx, base_frame, base_mask_id, base_b], axis=1).astype(np.float64),
                pair_incidence,
            ],
            axis=0,
        )
    else:
        fused_incidence = np.stack([base_local, base_mask_idx, base_frame, base_mask_id, base_b], axis=1).astype(np.float64)
    pair_carrier_ids = -910_000_000_000 - np.arange(1, len(rel_rows) + 1, dtype=np.int64)
    fused_carrier_id = np.concatenate([base_carrier_id, pair_carrier_ids]).astype(np.int64, copy=False)
    base_rel = base.get("carrier_reliability")
    base_broad = base.get("carrier_broad_risk")
    rel_base_np = (
        base_rel.cpu().numpy().astype(np.float32)
        if torch.is_tensor(base_rel) and int(base_rel.numel()) == int(base_carrier_id.shape[0])
        else np.ones((base_carrier_id.shape[0],), dtype=np.float32)
    )
    broad_base_np = (
        base_broad.cpu().numpy().astype(np.float32)
        if torch.is_tensor(base_broad) and int(base_broad.numel()) == int(base_carrier_id.shape[0])
        else np.zeros((base_carrier_id.shape[0],), dtype=np.float32)
    )
    fused_rel = np.concatenate([rel_base_np, np.asarray(rel_rows, dtype=np.float32)]).astype(np.float32, copy=False)
    fused_broad = np.concatenate([broad_base_np, np.asarray(broad_rows, dtype=np.float32)]).astype(np.float32, copy=False)

    feature = _countsketch(
        carrier_idx=fused_incidence[:, 0].astype(np.int64),
        mask_idx=fused_incidence[:, 1].astype(np.int64),
        b_ia=fused_incidence[:, 4].astype(np.float32),
        mask_weight=mask_weight,
        carrier_count=int(fused_carrier_id.shape[0]),
        sketch_dim=int(args.sketch_dim),
        device=device,
    )
    norm = np.linalg.norm(feature, axis=1)
    incidence_path = scene_out / "primitive_incidence_sparse.pt"
    feature_path = scene_out / "primitive_affinity_feature.pt"
    provider = ["d4rt_positive_core" for _ in range(base_carrier_id.shape[0])] + ["da3_bridge_pair_primitive" for _ in rel_rows]
    torch.save(
        {
            "schema_version": "stream4d_v103_phase9n_fused_primitive_incidence_sparse_v1",
            "phase_id": PHASE_ID,
            "scene_id": scene_id,
            "carrier_id": torch.as_tensor(fused_carrier_id, dtype=torch.int64),
            "carrier_local_index": torch.as_tensor(fused_incidence[:, 0].astype(np.int64), dtype=torch.int64),
            "mask_observation_index": torch.as_tensor(fused_incidence[:, 1].astype(np.int64), dtype=torch.int64),
            "frame_local_index": torch.as_tensor(fused_incidence[:, 2].astype(np.int64), dtype=torch.int64),
            "mask_id": torch.as_tensor(fused_incidence[:, 3].astype(np.int64), dtype=torch.int64),
            "B_ia": torch.as_tensor(fused_incidence[:, 4].astype(np.float32), dtype=torch.float32),
            "mask_frame": torch.as_tensor(mask_frame, dtype=torch.int64),
            "mask_label": torch.as_tensor(mask_label, dtype=torch.int64),
            "mask_is_object_like": torch.as_tensor(mask_is_object, dtype=torch.bool),
            "mask_is_broad": torch.as_tensor(mask_is_broad, dtype=torch.bool),
            "mask_weight": torch.as_tensor(mask_weight, dtype=torch.float32),
            "carrier_reliability": torch.as_tensor(fused_rel, dtype=torch.float32),
            "carrier_broad_risk": torch.as_tensor(fused_broad, dtype=torch.float32),
            "primitive_provider": provider,
            "base_phase4_root": _rel(phase4_root),
            "phase9e_root": _rel(phase9e_root),
            "phase2_root": _rel(phase2_root),
            "pair_source": pair_source,
            "pair_filter": pair_filter,
            "B_ia_formula": "For each accepted DA3 bridge pair, create one synthetic pair primitive with B_ia=clip(final_bridge_score * semantic_residual_cosine * (1 - broad_contamination_score), 0, 1) on both endpoint mask observations.",
            "uses_gt": False,
            "uses_gt_for_diagnostic": True,
            "uses_future": False,
        },
        incidence_path,
    )
    torch.save(
        {
            "schema_version": "stream4d_v103_phase9n_fused_primitive_affinity_feature_v1",
            "phase_id": PHASE_ID,
            "scene_id": scene_id,
            "carrier_id": torch.as_tensor(fused_carrier_id, dtype=torch.int64),
            "feature": torch.as_tensor(feature, dtype=torch.float16),
            "feature_norm_source_dtype": "float32",
            "sketch_dim": int(args.sketch_dim),
            "sketch_seed": SKETCH_SEED,
            "base_phase4_root": _rel(phase4_root),
            "phase9e_root": _rel(phase9e_root),
            "phase2_root": _rel(phase2_root),
            "uses_gt": False,
            "uses_future": False,
        },
        feature_path,
    )
    _write_csv(scene_out / "da3_bridge_pair_primitive_rows.csv", pair_rows)

    pair_mask_idx = pair_incidence[:, 1].astype(np.int64) if pair_incidence.size else np.asarray([], dtype=np.int64)
    base_support = np.bincount(base_mask_idx, minlength=int(mask_frame.shape[0])).astype(np.int64)
    pair_support = np.bincount(pair_mask_idx, minlength=int(mask_frame.shape[0])).astype(np.int64) if pair_mask_idx.size else np.zeros_like(base_support)
    newly_supported = (base_support == 0) & (pair_support > 0)
    diag_false_rate = float(np.mean([bool(r["diagnostic_different_gt"]) for r in pair_rows])) if pair_rows else 0.0
    metric = {
        "schema_version": "stream4d_v103_phase9n_metric_row_v1",
        "phase_id": PHASE_ID,
        "scene_id": scene_id,
        "phase2_root": _rel(phase2_root),
        "pair_source": pair_source,
        "pair_filter": pair_filter,
        "min_pair_reliability": float(args.min_pair_reliability),
        "input_pair_count": int(len(pairs_all)),
        "filtered_pair_count": int(len(pairs)),
        "included_pair_primitive_count": int(len(pair_rows)),
        "pair_missing_endpoint_count": int(missing),
        "base_d4rt_primitive_count": int(base_carrier_id.shape[0]),
        "fused_primitive_count": int(fused_carrier_id.shape[0]),
        "pair_incidence_row_count": int(pair_incidence.shape[0]),
        "pair_supported_mask_observation_count": int(np.count_nonzero(pair_support > 0)),
        "newly_supported_mask_observation_count": int(np.count_nonzero(newly_supported)),
        "newly_supported_object_like_count": int(np.count_nonzero(newly_supported & mask_is_object)),
        "newly_supported_broad_count": int(np.count_nonzero(newly_supported & mask_is_broad)),
        "pair_reliability_mean": float(np.mean(rel_rows)) if rel_rows else 0.0,
        "pair_reliability_p05": float(np.percentile(rel_rows, 5)) if rel_rows else 0.0,
        "pair_reliability_p95": float(np.percentile(rel_rows, 95)) if rel_rows else 0.0,
        "diagnostic_different_gt_pair_rate": diag_false_rate,
        "feature_valid_rate": float(np.mean(norm > 0.0)) if norm.size else 0.0,
        "uses_gt_for_feature": False,
        "uses_gt_for_diagnostic": True,
        "uses_future": False,
    }
    gates = [
        ("has_included_pair_primitives", metric["included_pair_primitive_count"] > 0, metric["included_pair_primitive_count"], ">0"),
        ("feature_valid_rate_ge_0p95", metric["feature_valid_rate"] >= 0.95, metric["feature_valid_rate"], 0.95),
    ]
    gate_rows = [
        {
            "schema_version": "stream4d_v103_phase9n_gate_row_v1",
            "phase_id": PHASE_ID,
            "scene_id": scene_id,
            "gate_name": name,
            "pass": bool(ok),
            "observed": observed,
            "required": required,
        }
        for name, ok, observed, required in gates
    ]
    failure_rows = [
        {
            "schema_version": "stream4d_v103_phase9n_failure_row_v1",
            "phase_id": PHASE_ID,
            "scene_id": scene_id,
            "failure_id": row["gate_name"],
            "severity": "blocking",
            "evidence": f"observed={row['observed']} required={row['required']}",
            "uses_gt_for_prediction": False,
        }
        for row in gate_rows
        if not bool(row["pass"])
    ]
    artifact_rows = [
        {
            "schema_version": "stream4d_v103_phase9n_artifact_row_v1",
            "phase_id": PHASE_ID,
            "scene_id": scene_id,
            "role": "primitive_incidence_sparse",
            "path": _rel(incidence_path),
            "exists": incidence_path.exists(),
            "size_bytes": incidence_path.stat().st_size if incidence_path.exists() else 0,
        },
        {
            "schema_version": "stream4d_v103_phase9n_artifact_row_v1",
            "phase_id": PHASE_ID,
            "scene_id": scene_id,
            "role": "da3_bridge_pair_primitive_rows",
            "path": _rel(scene_out / "da3_bridge_pair_primitive_rows.csv"),
            "exists": (scene_out / "da3_bridge_pair_primitive_rows.csv").exists(),
            "size_bytes": (scene_out / "da3_bridge_pair_primitive_rows.csv").stat().st_size if (scene_out / "da3_bridge_pair_primitive_rows.csv").exists() else 0,
        },
    ]
    _write_json(scene_out / "scene_summary.json", {"metric": metric, "failure_count": len(failure_rows)})
    return metric, gate_rows, failure_rows, artifact_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fuse Phase9e DA3 bridge-pair primitives into the v103 Phase4 affinity interface.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--phase4-root", default=str(DEFAULT_PHASE4_ROOT))
    parser.add_argument("--phase9e-root", default=str(DEFAULT_PHASE9E_ROOT))
    parser.add_argument(
        "--pair-source",
        default="best_clean_variant",
        help="Accepted pair source from Phase9e: best_clean_variant, best_variant, or variant:<phase9e_variant_id>.",
    )
    parser.add_argument(
        "--pair-filter",
        choices=[
            "both_object_nonbroad",
            "no_broad",
            "high_reliability",
            "no_broad_or_high_reliability",
            "all_clean_pairs",
        ],
        default="both_object_nonbroad",
    )
    parser.add_argument("--min-pair-reliability", type=float, default=0.70)
    parser.add_argument("--scene", choices=["all", "scene0011_00", "scene0050_00"], default="all")
    parser.add_argument("--sketch-dim", type=int, default=2048)
    parser.add_argument("--scene0011-phase2-root", default=None)
    parser.add_argument("--scene0050-phase2-root", default=None)
    parser.add_argument("--scene0011-pair-source", default=None)
    parser.add_argument("--scene0050-pair-source", default=None)
    parser.add_argument("--scene0011-pair-filter", choices=[
        "both_object_nonbroad",
        "no_broad",
        "high_reliability",
        "no_broad_or_high_reliability",
        "all_clean_pairs",
    ], default=None)
    parser.add_argument("--scene0050-pair-filter", choices=[
        "both_object_nonbroad",
        "no_broad",
        "high_reliability",
        "no_broad_or_high_reliability",
        "all_clean_pairs",
    ], default=None)
    return parser


def main() -> int:
    t0 = time.time()
    args = build_parser().parse_args()
    out = _project_phase(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    (out / "last_command.txt").write_text(" ".join([sys.executable, str(Path(__file__)), *sys.argv[1:]]) + "\n", encoding="utf-8")
    phase4_summary = _read_json(_project(args.phase4_root) / "summary.json")
    if not bool(phase4_summary.get("phase4_pass", False)):
        raise RuntimeError(f"base Phase4 root did not pass: {args.phase4_root}")
    scenes = list(SCENE_SPECS.keys()) if args.scene == "all" else [str(args.scene)]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metric_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    artifact_rows: list[dict[str, Any]] = []
    for scene in scenes:
        metric, gates, failures, artifacts = _run_scene(scene, args, device)
        metric_rows.append(metric)
        gate_rows.extend(gates)
        failure_rows.extend(failures)
        artifact_rows.extend(artifacts)
    _write_csv(out / "phase9n_metric_rows.csv", metric_rows)
    _write_csv(out / "gate_rows.csv", gate_rows)
    _write_csv(out / "failure_rows.csv", failure_rows)
    _write_csv(out / "artifact_rows.csv", artifact_rows)
    decision = "PASS_PHASE9N_DA3_BRIDGE_PAIR_FUSED_PHASE4_ARTIFACT_READY" if not failure_rows else "NO_GO_PHASE9N_DA3_BRIDGE_PAIR_FUSED_PHASE4"
    phase4_ready = not failure_rows
    summary = {
        "schema_version": "stream4d_v103_phase9n_summary_v1",
        "phase_id": PHASE_ID,
        "created_unix_time": time.time(),
        "runtime_sec": time.time() - t0,
        "decision": decision,
        "phase4_pass": bool(phase4_ready),
        "phase9n_artifact_ready": bool(phase4_ready),
        "failure_count": len(failure_rows),
        "scene_ids": scenes,
        "phase4_root": _rel(_project(args.phase4_root)),
        "phase9e_root": _rel(_project_phase(args.phase9e_root)),
        "pair_source": str(args.pair_source),
        "pair_filter": str(args.pair_filter),
        "scene_phase2_roots": {scene: _rel(_scene_phase2_root(scene, args)) for scene in scenes},
        "scene_pair_sources": {scene: _scene_pair_source(scene, args) for scene in scenes},
        "scene_pair_filters": {scene: _scene_pair_filter(scene, args) for scene in scenes},
        "min_pair_reliability": float(args.min_pair_reliability),
        "uses_gt_for_feature": False,
        "uses_gt_for_diagnostic": True,
        "uses_future": False,
        "truthfulness_note": "Phase9n turns GT-free Phase9e accepted DA3 bridge pairs into synthetic pair primitives under the same B_ia -> z_i -> Phi_a interface. Diagnostic GT labels from Phase9e are copied only for audit summaries and are not used for feature construction or filtering.",
        "outputs": {
            "summary": _rel(out / "summary.json"),
            "phase9n_metric_rows": _rel(out / "phase9n_metric_rows.csv"),
            "gate_rows": _rel(out / "gate_rows.csv"),
            "failure_rows": _rel(out / "failure_rows.csv"),
            "artifact_rows": _rel(out / "artifact_rows.csv"),
            "last_command": _rel(out / "last_command.txt"),
        },
    }
    _write_json(out / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if not failure_rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
