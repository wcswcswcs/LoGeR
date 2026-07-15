#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
AUDIT_ROOT = STREAM3D_ROOT / "outputs" / "audit"

PHASE_ID = "v103_r2_phase1_semantic_soft_candidate_universe"
DEFAULT_OUT = AUDIT_ROOT / PHASE_ID

DEFAULT_PHASE0_ROOT = AUDIT_ROOT / "v103_r2_phase0_fact_lock"
DEFAULT_PHASE9B_ROOT = AUDIT_ROOT / "v103_phase9b_da3_c0001_provider_readiness_all_r1"
DEFAULT_PHASE9E_ROOT = AUDIT_ROOT / "v103_phase9e_d4rt_anchor_da3_induced_carriers_suppS1_d4rt48mix_s5repair_r1"
DEFAULT_SCENE0011_PHASE2 = (
    AUDIT_ROOT / "v103_phase2_stratified_q5c_objlike16384_scene0011_c0001_cap24576_qchunk16384_d4rt48mix_maskbalanced8"
)
DEFAULT_SCENE0050_PHASE2 = (
    AUDIT_ROOT / "v103_phase2_stratified_q5c_objlike16384_scene0050_c0001_cap24576_qchunk16384_d4rt48mix_maskbalanced8"
)

SEMANTIC_ROWS = {
    "scene0011_00": AUDIT_ROOT / "v91_radio_mask_features_npz_scene0011" / "mask_feature_rows.csv",
    "scene0050_00": AUDIT_ROOT / "v91_radio_mask_features_npz_scene0050" / "mask_feature_rows.csv",
}

OBJECT_LIKE_AREA_MIN = 0.001
SEM_HARD_BROAD_AREA_RATIO = 0.12
SEM_SOFT_AREA_MAX = 0.20
SEM_SOFT_RISK_MAX = 0.55


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


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


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


def _truth(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _floatish(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _mask(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img, dtype=np.int32)


def _load_phase2_masks(root: Path) -> tuple[dict[str, Any], list[int], np.ndarray]:
    summary = _read_json(root / "summary.json")
    frame_ids = [int(v) for v in summary["frame_ids"]]
    mask_root = _project(summary["mask_root"])
    masks = np.stack([_mask(mask_root / f"{frame_id}.png") for frame_id in frame_ids], axis=0)
    return summary, frame_ids, masks


def _semantic_meta(scene_id: str) -> dict[tuple[int, int], dict[str, Any]]:
    path = SEMANTIC_ROWS[scene_id]
    df = pd.read_csv(path)
    out: dict[tuple[int, int], dict[str, Any]] = {}
    for row in df.to_dict("records"):
        out[(int(row["frame_id"]), int(row["mask_id"]))] = {
            "semantic_entropy": _floatish(row.get("semantic_entropy")),
            "semantic_boundary_variance": _floatish(row.get("semantic_boundary_variance")),
            "semantic_background_score_proxy": _truth(row.get("semantic_background_score_proxy")),
            "broad_background_risk": _truth(row.get("broad_background_risk")),
            "feature_available": _truth(row.get("feature_available")),
        }
    return out


def _support_map(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if df.empty:
        return {}
    count_col = "d4rt_anchor_support_count"
    if count_col not in df.columns:
        numeric = [col for col in df.columns if col.endswith("_count")]
        count_col = numeric[0] if numeric else ""
    out: dict[str, int] = {}
    for row in df.to_dict("records"):
        obs = str(row.get("mask_observation_id", ""))
        if not obs:
            continue
        out[obs] = int(_floatish(row.get(count_col), 1.0))
    return out


def _da3_mask_hits(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    out: dict[str, dict[str, Any]] = {}
    for row in df.to_dict("records"):
        obs = str(row.get("mask_observation_id", ""))
        if not obs:
            continue
        out[obs] = {
            "da3_participating_primitive_count": int(_floatish(row.get("participating_primitive_count"))),
            "da3_mask_area_ratio": _floatish(row.get("mask_area_ratio")),
            "uses_gt_for_prediction": _truth(row.get("uses_gt_for_prediction")),
            "uses_gt_for_diagnostic_labels": _truth(row.get("uses_gt_for_diagnostic_labels")),
        }
    return out


def _area_stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"p50": 0.0, "p90": 0.0, "p95": 0.0, "max": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "p50": float(np.quantile(arr, 0.50)),
        "p90": float(np.quantile(arr, 0.90)),
        "p95": float(np.quantile(arr, 0.95)),
        "max": float(np.max(arr)),
    }


def _risk(area_ratio: float, semantic_broad: bool, boundary_variance: float, has_veto: bool, has_positive: bool) -> dict[str, float]:
    area_risk = float(np.clip((area_ratio - SEM_HARD_BROAD_AREA_RATIO) / max(SEM_SOFT_AREA_MAX - SEM_HARD_BROAD_AREA_RATIO, 1e-6), 0.0, 1.0))
    sem_risk = 1.0 if semantic_broad else 0.0
    edge_risk = float(np.clip(boundary_variance / 0.001, 0.0, 1.0))
    support_risk = 0.50 if has_veto and not has_positive else (0.25 if has_veto else 0.0)
    score = 0.25 * area_risk + 0.40 * sem_risk + 0.15 * edge_risk + 0.20 * support_risk
    return {
        "area_risk": area_risk,
        "semantic_broad_risk_score": sem_risk,
        "edge_risk": edge_risk,
        "support_risk": support_risk,
        "risk_score": float(score),
    }


def _scene_phase2_roots(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "scene0011_00": _project(args.scene0011_phase2_root),
        "scene0050_00": _project(args.scene0050_phase2_root),
    }


def _build_scene(
    *,
    scene_id: str,
    phase2_root: Path,
    phase9b_root: Path,
    phase9e_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    phase2_summary, frame_ids, masks = _load_phase2_masks(phase2_root)
    sem = _semantic_meta(scene_id)
    h, w = masks.shape[1:]
    denom = float(max(h * w, 1))

    scene9e = phase9e_root / scene_id
    a_support = _support_map(scene9e / "d4rt_positive_anchor_support_rows.csv")
    s_support = _support_map(scene9e / "d4rt_support_coverage_rows.csv")
    v_support = _support_map(scene9e / "d4rt_veto_support_rows.csv")
    da3_hits = _da3_mask_hits(phase9b_root / scene_id / "chunk32_mask_primitive_summary_rows.csv")

    all_rows: list[dict[str, Any]] = []
    hard_obs: set[str] = set()
    soft_obs: set[str] = set()
    for fi, frame_id in enumerate(frame_ids):
        labels, counts = np.unique(masks[fi], return_counts=True)
        for label, count in zip(labels.tolist(), counts.tolist()):
            label = int(label)
            if label <= 0:
                continue
            area_ratio = float(count) / denom
            meta = sem.get((int(frame_id), label), {})
            semantic_broad = bool(meta.get("broad_background_risk")) or bool(meta.get("semantic_background_score_proxy"))
            obs = f"{scene_id}:{int(frame_id)}:{label}"
            is_area_object_scale = OBJECT_LIKE_AREA_MIN <= area_ratio <= SEM_SOFT_AREA_MAX
            is_hard_broad = (area_ratio >= SEM_HARD_BROAD_AREA_RATIO) or semantic_broad
            is_semhard = is_area_object_scale and not is_hard_broad
            has_a = obs in a_support
            has_s = obs in s_support
            has_v = obs in v_support
            risk = _risk(
                area_ratio=area_ratio,
                semantic_broad=semantic_broad,
                boundary_variance=_floatish(meta.get("semantic_boundary_variance")),
                has_veto=has_v,
                has_positive=has_a or has_s,
            )
            risk_exception = bool(has_a or has_s)
            is_semsoft = bool(is_area_object_scale and (risk["risk_score"] <= SEM_SOFT_RISK_MAX or risk_exception))
            if is_semhard:
                hard_obs.add(obs)
            if is_semsoft:
                soft_obs.add(obs)
                da3 = da3_hits.get(obs, {})
                all_rows.append(
                    {
                        "schema_version": "stream4d_v103_r2_phase1_candidate_universe_row_v1",
                        "phase_id": PHASE_ID,
                        "scene_id": scene_id,
                        "chunk_id": "c0001",
                        "frame_id": int(frame_id),
                        "frame_local_index": int(fi),
                        "mask_id": label,
                        "mask_observation_id": obs,
                        "pixel_count": int(count),
                        "area_ratio": area_ratio,
                        "candidate_policy_id": "semsoft_area020_semantic_broad_as_risk_anchor_exception",
                        "semantic_soft_risk_max": SEM_SOFT_RISK_MAX,
                        "risk_exception_policy": "A_anchor_or_S_support_exception",
                        "risk_exception_used": bool(risk["risk_score"] > SEM_SOFT_RISK_MAX and risk_exception),
                        "in_semantic_hard_universe": bool(is_semhard),
                        "in_semantic_soft_universe": bool(is_semsoft),
                        "candidate_delta_type": "existing_semhard" if is_semhard else "extra_over_semhard",
                        "semantic_broad_flag": semantic_broad,
                        "broad_background_risk": bool(meta.get("broad_background_risk")),
                        "semantic_background_score_proxy": bool(meta.get("semantic_background_score_proxy")),
                        "semantic_entropy": _floatish(meta.get("semantic_entropy")),
                        "semantic_boundary_variance": _floatish(meta.get("semantic_boundary_variance")),
                        "A_anchor_support_count": int(a_support.get(obs, 0)),
                        "S_support_count": int(s_support.get(obs, 0)),
                        "V_veto_support_count": int(v_support.get(obs, 0)),
                        "A_anchor_hit": has_a,
                        "S_support_hit": has_s,
                        "V_veto_hit": has_v,
                        "DA3_primitive_hit": int(da3.get("da3_participating_primitive_count", 0)) > 0,
                        "DA3_participating_primitive_count": int(da3.get("da3_participating_primitive_count", 0)),
                        "DA3_hit_source": _rel(phase9b_root / scene_id / "chunk32_mask_primitive_summary_rows.csv"),
                        **risk,
                        "uses_gt_for_selection": False,
                        "uses_future": False,
                    }
                )

    candidate_rows = all_rows
    extra_rows = [row for row in candidate_rows if row["candidate_delta_type"] == "extra_over_semhard"]
    semantic_broad_rows = [row for row in candidate_rows if row["semantic_broad_flag"]]
    da3_rows = [
        {
            "schema_version": "stream4d_v103_r2_phase1_candidate_da3_hit_row_v1",
            "phase_id": PHASE_ID,
            "scene_id": row["scene_id"],
            "chunk_id": row["chunk_id"],
            "frame_id": row["frame_id"],
            "frame_local_index": row["frame_local_index"],
            "mask_id": row["mask_id"],
            "mask_observation_id": row["mask_observation_id"],
            "candidate_delta_type": row["candidate_delta_type"],
            "DA3_primitive_hit": row["DA3_primitive_hit"],
            "DA3_participating_primitive_count": row["DA3_participating_primitive_count"],
            "DA3_hit_source": row["DA3_hit_source"],
            "uses_gt_for_selection": False,
            "uses_future": False,
        }
        for row in candidate_rows
    ]
    role_rows = [
        {
            "schema_version": "stream4d_v103_r2_phase1_candidate_role_support_row_v1",
            "phase_id": PHASE_ID,
            "scene_id": row["scene_id"],
            "chunk_id": row["chunk_id"],
            "frame_id": row["frame_id"],
            "frame_local_index": row["frame_local_index"],
            "mask_id": row["mask_id"],
            "mask_observation_id": row["mask_observation_id"],
            "candidate_delta_type": row["candidate_delta_type"],
            "A_anchor_support_count": row["A_anchor_support_count"],
            "S_support_count": row["S_support_count"],
            "V_veto_support_count": row["V_veto_support_count"],
            "A_anchor_hit": row["A_anchor_hit"],
            "S_support_hit": row["S_support_hit"],
            "V_veto_hit": row["V_veto_hit"],
            "role_support_source": _rel(scene9e),
            "uses_gt_for_selection": False,
            "uses_future": False,
        }
        for row in candidate_rows
    ]

    extra_areas = [float(row["area_ratio"]) for row in extra_rows]
    extra_non_anchor = [row for row in extra_rows if not row["A_anchor_hit"]]
    extra_da3_sum = int(sum(int(row["DA3_participating_primitive_count"]) for row in extra_rows))
    extra_da3_hit_count = int(sum(bool(row["DA3_primitive_hit"]) for row in extra_rows))
    policy = {
        "schema_version": "stream4d_v103_r2_phase1_candidate_policy_row_v1",
        "phase_id": PHASE_ID,
        "scene_id": scene_id,
        "chunk_id": "c0001",
        "phase2_root": _rel(phase2_root),
        "phase9b_root": _rel(phase9b_root),
        "phase9e_root": _rel(phase9e_root),
        "frame_min": min(frame_ids),
        "frame_max": max(frame_ids),
        "frame_count": len(frame_ids),
        "policy_id": "semsoft_area020_semantic_broad_as_risk_anchor_exception",
        "object_like_area_min": OBJECT_LIKE_AREA_MIN,
        "semhard_broad_area_ratio": SEM_HARD_BROAD_AREA_RATIO,
        "semsoft_area_max": SEM_SOFT_AREA_MAX,
        "semsoft_risk_max": SEM_SOFT_RISK_MAX,
        "risk_exception_policy": "A_anchor_or_S_support_exception",
        "risk_exception_count": int(sum(bool(row.get("risk_exception_used", False)) for row in candidate_rows)),
        "semantic_broad_hard_veto": False,
        "candidate_count": len(candidate_rows),
        "semantic_hard_candidate_count": len(hard_obs),
        "extra_candidate_count_over_semhard": len(extra_rows),
        "extra_non_anchor_candidate_count": len(extra_non_anchor),
        "extra_area_p50": _area_stats(extra_areas)["p50"],
        "extra_area_p90": _area_stats(extra_areas)["p90"],
        "semantic_broad_rate": float(np.mean([row["semantic_broad_flag"] for row in candidate_rows])) if candidate_rows else 0.0,
        "DA3_extra_mask_hit_count": extra_da3_hit_count,
        "DA3_extra_gaussian_hit_count": extra_da3_sum,
        "A_anchor_hit_rate": float(np.mean([row["A_anchor_hit"] for row in extra_rows])) if extra_rows else 0.0,
        "S_support_hit_rate": float(np.mean([row["S_support_hit"] for row in extra_rows])) if extra_rows else 0.0,
        "V_veto_hit_rate": float(np.mean([row["V_veto_hit"] for row in extra_rows])) if extra_rows else 0.0,
        "risk_score_mean": float(np.mean([row["risk_score"] for row in extra_rows])) if extra_rows else 0.0,
        "risk_score_p90": float(np.quantile([row["risk_score"] for row in extra_rows], 0.90)) if extra_rows else 0.0,
        "uses_gt_for_selection": False,
        "uses_future": False,
        "truthfulness_note": "R2-1 reconstructs c0001 semantic-soft candidates from masks/semantic metadata; GT diagnostic columns in Phase9B are ignored.",
    }
    return candidate_rows, semantic_broad_rows, da3_rows, role_rows, policy


def _gate(gate_id: str, passed: bool, observed: Any, required: Any, repair_direction: str = "") -> dict[str, Any]:
    return {
        "schema_version": "stream4d_v103_r2_phase1_gate_row_v1",
        "phase_id": PHASE_ID,
        "gate_id": gate_id,
        "pass": bool(passed),
        "observed": observed,
        "required": required,
        "repair_direction": repair_direction,
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.time()
    out = _project(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    phase0_root = _project(args.phase0_root)
    phase9b_root = _project(args.phase9b_root)
    phase9e_root = _project(args.phase9e_root)

    all_candidates: list[dict[str, Any]] = []
    semantic_broad_rows: list[dict[str, Any]] = []
    da3_rows: list[dict[str, Any]] = []
    role_rows: list[dict[str, Any]] = []
    policy_rows: list[dict[str, Any]] = []
    for scene, phase2_root in _scene_phase2_roots(args).items():
        c_rows, b_rows, d_rows, r_rows, policy = _build_scene(
            scene_id=scene,
            phase2_root=phase2_root,
            phase9b_root=phase9b_root,
            phase9e_root=phase9e_root,
        )
        all_candidates.extend(c_rows)
        semantic_broad_rows.extend(b_rows)
        da3_rows.extend(d_rows)
        role_rows.extend(r_rows)
        policy_rows.append(policy)

    gate_rows: list[dict[str, Any]] = []
    for row in policy_rows:
        scene = row["scene_id"]
        gate_rows.append(
            _gate(
                f"{scene}_extra_non_anchor_candidate_count_ge_30",
                int(row["extra_non_anchor_candidate_count"]) >= 30,
                row["extra_non_anchor_candidate_count"],
                ">= 30",
                "Try R2-1 variants: area threshold, anchor-hit exception, or semantic baseline calibration.",
            )
        )
        gate_rows.append(
            _gate(
                f"{scene}_da3_extra_gaussian_hit_count_gt_0",
                int(row["DA3_extra_gaussian_hit_count"]) > 0,
                row["DA3_extra_gaussian_hit_count"],
                "> 0",
                "If zero, check Phase9B c0001 DA3 provider root and mask-observation ids.",
            )
        )
        gate_rows.append(
            _gate(
                f"{scene}_extra_area_p50_le_0p08",
                float(row["extra_area_p50"]) <= 0.08,
                row["extra_area_p50"],
                "<= 0.08",
                "Keep semantic broad as risk; do not widen into broad dominated masks.",
            )
        )
    gate_rows.append(
        _gate(
            "semantic_broad_risk_rows_available",
            len(semantic_broad_rows) > 0,
            len(semantic_broad_rows),
            "> 0",
            "If absent, inspect semantic feature rows/proxy construction.",
        )
    )
    gate_rows.append(_gate("uses_gt_for_selection_false", True, False, "False"))
    gate_rows.append(_gate("uses_future_false", True, False, "False"))

    failure_rows = [
        {
            "schema_version": "stream4d_v103_r2_phase1_failure_row_v1",
            "phase_id": PHASE_ID,
            "failure_id": row["gate_id"],
            "severity": "blocker",
            "observed": row["observed"],
            "expected": row["required"],
            "repair_direction": row["repair_direction"],
        }
        for row in gate_rows
        if not row["pass"]
    ]
    pass_all = not failure_rows

    _write_csv(out / "candidate_universe_rows.csv", all_candidates)
    _write_csv(out / "candidate_policy_rows.csv", policy_rows)
    _write_csv(out / "semantic_broad_risk_rows.csv", semantic_broad_rows)
    _write_csv(out / "candidate_da3_hit_rows.csv", da3_rows)
    _write_csv(out / "candidate_role_support_rows.csv", role_rows)
    _write_csv(out / "gate_rows.csv", gate_rows)
    _write_csv(out / "failure_rows.csv", failure_rows)
    (out / "last_command.txt").write_text(" ".join([sys.executable, *sys.argv]) + "\n", encoding="utf-8")

    summary = {
        "schema_version": "stream4d_v103_r2_phase1_summary_v1",
        "phase_id": PHASE_ID,
        "created_unix_time": time.time(),
        "runtime_sec": time.time() - t0,
        "decision": "PASS_R2_1_SEMANTIC_SOFT_CANDIDATE_UNIVERSE_READY"
        if pass_all
        else "NO_GO_SEMANTIC_SOFT_CANDIDATE_BLOCKER",
        "phase0_root": _rel(phase0_root),
        "phase9b_root": _rel(phase9b_root),
        "phase9e_root": _rel(phase9e_root),
        "scene_count": len(policy_rows),
        "candidate_count": int(sum(int(row["candidate_count"]) for row in policy_rows)),
        "extra_candidate_count_over_semhard": int(sum(int(row["extra_candidate_count_over_semhard"]) for row in policy_rows)),
        "extra_non_anchor_candidate_count_min": int(
            min([int(row["extra_non_anchor_candidate_count"]) for row in policy_rows] or [0])
        ),
        "DA3_extra_gaussian_hit_count_min": int(min([int(row["DA3_extra_gaussian_hit_count"]) for row in policy_rows] or [0])),
        "extra_area_p50_max": float(max([float(row["extra_area_p50"]) for row in policy_rows] or [0.0])),
        "uses_gt_for_selection": False,
        "uses_future": False,
        "truthfulness_note": "R2-1 is a c0001 candidate-universe artifact. It does not claim provider-ready DA3 components or AP improvement.",
        "outputs": {
            "candidate_universe_rows": _rel(out / "candidate_universe_rows.csv"),
            "candidate_policy_rows": _rel(out / "candidate_policy_rows.csv"),
            "semantic_broad_risk_rows": _rel(out / "semantic_broad_risk_rows.csv"),
            "candidate_da3_hit_rows": _rel(out / "candidate_da3_hit_rows.csv"),
            "candidate_role_support_rows": _rel(out / "candidate_role_support_rows.csv"),
            "gate_rows": _rel(out / "gate_rows.csv"),
            "failure_rows": _rel(out / "failure_rows.csv"),
            "summary": _rel(out / "summary.json"),
        },
    }
    _write_json(out / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build v103 R2-1 c0001 semantic-soft candidate universe.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--phase0-root", default=str(DEFAULT_PHASE0_ROOT))
    parser.add_argument("--phase9b-root", default=str(DEFAULT_PHASE9B_ROOT))
    parser.add_argument("--phase9e-root", default=str(DEFAULT_PHASE9E_ROOT))
    parser.add_argument("--scene0011-phase2-root", default=str(DEFAULT_SCENE0011_PHASE2))
    parser.add_argument("--scene0050-phase2-root", default=str(DEFAULT_SCENE0050_PHASE2))
    return parser.parse_args()


def main() -> int:
    summary = build(parse_args())
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if summary["decision"].startswith("PASS_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
