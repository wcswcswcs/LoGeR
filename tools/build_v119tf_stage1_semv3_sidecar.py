#!/usr/bin/env python3
"""Build ACL2 v119-TF Stage1 SEM-V3 causal object sidecar artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
V118_ROOT = ROOT / "results/acl2_v118tf_operation_specific_semantic_carrier_calibration"
V119_ROOT = ROOT / "results/acl2_v119tf_semantic_addressable_geometry_carrier_routing_representation_repair"
V118_STAGE1 = V118_ROOT / "stage1_causal_object_track_sidecar"
OUT = V119_ROOT / "stage1_semv3_sidecar"
IMAGE_AREA = 218.0 * 720.0
SEQ_IDS = ("00", "02")
LEGACY_FORMULA_COLUMNS = (
    "shape_stability_prefix",
    "area_ratio_stability_prefix",
    "boundary_stability_prefix",
    "boundary_stability_mode",
    "semantic_persistence_prefix",
)


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_json(v) for v in value]
    if isinstance(value, Path):
        return rel(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return clean_json(value.item())
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(clean_json(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def finite(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def median_abs_deviation(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    med = float(np.median(arr))
    return float(np.median(np.abs(arr - med)))


def add_semv3_fields(group: pd.DataFrame, *, lambda_p: float, tau_reobs: float) -> pd.DataFrame:
    group = group.sort_values("frame_id").copy()
    log_area_values: list[float] = []
    compactness_values: list[float] = []
    shape_scores: list[float] = []
    mad_log_area_values: list[float] = []
    mad_compactness_values: list[float] = []
    area_pixels_values: list[float] = []
    compactness_current_values: list[float] = []

    for row in group.itertuples(index=False):
        area_ratio = max(finite(getattr(row, "current_area_ratio")), 1.0 / IMAGE_AREA)
        area_pixels = max(area_ratio * IMAGE_AREA, 1.0)
        perimeter = max(finite(getattr(row, "current_perimeter")), 0.0)
        log_area = math.log(area_pixels)
        compactness = perimeter / math.sqrt(area_pixels)
        log_area_values.append(log_area)
        compactness_values.append(compactness)
        mad_log_area = median_abs_deviation(log_area_values)
        mad_compactness = median_abs_deviation(compactness_values)
        mad_log_area_values.append(mad_log_area)
        mad_compactness_values.append(mad_compactness)
        shape_scores.append(float(math.exp(-mad_log_area - lambda_p * mad_compactness)))
        area_pixels_values.append(area_pixels)
        compactness_current_values.append(compactness)

    visible = group["visible_count_prefix"].astype(float).clip(lower=0.0)
    age = group["track_age_prefix"].astype(float).clip(lower=1.0)
    reobs = group["reobservation_count_prefix"].astype(float).clip(lower=0.0)
    out = group.copy()
    out["semv3_visibility_prefix"] = visible / age
    out["semv3_role_prior_prefix"] = group["role_consistency_prefix"].astype(float).clip(lower=0.0, upper=1.0)
    out["semv3_shape_mad_log_area_prefix"] = mad_log_area_values
    out["semv3_shape_mad_compactness_prefix"] = mad_compactness_values
    out["semv3_shape_score_prefix"] = shape_scores
    out["semv3_reobs_score_prefix"] = 1.0 - np.exp(-reobs / max(tau_reobs, 1e-9))
    out["semv3_motion_residual_prefix"] = group["motion_residual_prefix"].astype(float).clip(lower=0.0)
    out["semv3_motion_compensation_available_prefix"] = group[
        "motion_compensation_available_prefix"
    ].astype(float).clip(lower=0.0, upper=1.0)
    out["semv3_identity_key"] = (
        group["seq"].astype(str) + ":" + group["track_id"].astype(int).astype(str)
    )
    out["semv3_area_pixels_current"] = area_pixels_values
    out["semv3_compactness_current"] = compactness_current_values
    out["semv3_formula_version"] = "v119_semv3_visibility_role_shape_mad_reobs_motion_v1"
    out["semv3_output_scope"] = "role_prior_and_identity_cues_only_no_geometry_utility"
    return out


def build_semv3_rows(df: pd.DataFrame, *, lambda_p: float, tau_reobs: float) -> pd.DataFrame:
    rows = [
        add_semv3_fields(group, lambda_p=lambda_p, tau_reobs=tau_reobs)
        for _, group in df.groupby(["seq", "track_id"], sort=True)
    ]
    return pd.concat(rows, ignore_index=True)


def prefix_parity(
    source_df: pd.DataFrame,
    full_df: pd.DataFrame,
    *,
    lambda_p: float,
    tau_reobs: float,
    sample_limit: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fields = [
        "semv3_visibility_prefix",
        "semv3_role_prior_prefix",
        "semv3_shape_score_prefix",
        "semv3_shape_mad_log_area_prefix",
        "semv3_shape_mad_compactness_prefix",
        "semv3_reobs_score_prefix",
        "semv3_motion_residual_prefix",
        "semv3_motion_compensation_available_prefix",
    ]
    full_by_key = {
        (str(row.seq), int(row.track_id), int(row.frame_id)): row
        for row in full_df[["seq", "track_id", "frame_id", *fields]].itertuples(index=False)
    }
    parity_rows: list[dict[str, Any]] = []
    violations: list[dict[str, Any]] = []
    rng = random.Random("acl2_v119tf_stage1_semv3_prefix_parity")

    for seq, seq_df in source_df.groupby("seq", sort=True):
        track_ids = sorted(int(value) for value in seq_df["track_id"].unique())
        sampled = sorted(rng.sample(track_ids, min(sample_limit, len(track_ids))))
        seq_frame_max = int(seq_df["frame_id"].max())
        for track_id in sampled:
            track = seq_df[seq_df["track_id"].astype(int) == track_id]
            for ratio in (0.25, 0.5, 0.75, 1.0):
                limit = int(math.floor(seq_frame_max * ratio))
                prefix_source = track[track["frame_id"].astype(int) <= limit]
                prefix_full = add_semv3_fields(prefix_source, lambda_p=lambda_p, tau_reobs=tau_reobs)
                compared = 0
                max_abs_diff = 0.0
                for row in prefix_full.itertuples(index=False):
                    full_row = full_by_key.get((str(seq), track_id, int(row.frame_id)))
                    if full_row is None:
                        continue
                    compared += 1
                    for field in fields:
                        prefix_value = finite(getattr(row, field), float("nan"))
                        full_value = finite(getattr(full_row, field), float("nan"))
                        diff = abs(prefix_value - full_value)
                        if math.isfinite(diff):
                            max_abs_diff = max(max_abs_diff, diff)
                        if math.isfinite(diff) and diff > 1e-6:
                            violations.append(
                                {
                                    "seq": seq,
                                    "track_id": track_id,
                                    "prefix_ratio": ratio,
                                    "prefix_frame_limit": limit,
                                    "frame_id": int(row.frame_id),
                                    "field": field,
                                    "prefix_value": prefix_value,
                                    "full_value": full_value,
                                    "abs_diff": diff,
                                }
                            )
                parity_rows.append(
                    {
                        "schema": "acl2_v119tf_stage1_semv3_prefix_parity_row_v1",
                        "seq": seq,
                        "track_id": track_id,
                        "prefix_ratio": ratio,
                        "prefix_frame_limit": limit,
                        "compared_rows": compared,
                        "max_abs_diff": max_abs_diff,
                        "pass": bool(max_abs_diff <= 1e-6),
                    }
                )
    return parity_rows, violations


def final_rows(semv3: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (seq, track_id), group in semv3.groupby(["seq", "track_id"], sort=True):
        final = group.sort_values("frame_id").iloc[-1]
        rows.append(
            {
                "schema": "acl2_v119tf_stage1_semv3_running_summary_row_v1",
                "seq": str(seq),
                "track_id": int(track_id),
                "first_frame": int(group["frame_id"].min()),
                "last_frame": int(group["frame_id"].max()),
                "visible_count": int(final["visible_count_prefix"]),
                "track_age": int(final["track_age_prefix"]),
                "identity_key": str(final["semv3_identity_key"]),
                "dominant_label": str(final["dominant_label_prefix"]),
                "dominant_role": str(final["dominant_role_prefix"]),
                "visibility_score": float(final["semv3_visibility_prefix"]),
                "role_prior": float(final["semv3_role_prior_prefix"]),
                "shape_score": float(final["semv3_shape_score_prefix"]),
                "shape_mad_log_area": float(final["semv3_shape_mad_log_area_prefix"]),
                "shape_mad_compactness": float(final["semv3_shape_mad_compactness_prefix"]),
                "reobs_score": float(final["semv3_reobs_score_prefix"]),
                "reobservation_count": int(final["reobservation_count_prefix"]),
                "motion_residual": float(final["semv3_motion_residual_prefix"]),
                "motion_compensation_available": float(
                    final["semv3_motion_compensation_available_prefix"]
                ),
                "output_scope": str(final["semv3_output_scope"]),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lambda-p", type=float, default=1.0)
    parser.add_argument("--tau-reobs", type=float, default=3.0)
    parser.add_argument("--sample-limit", type=int, default=120)
    args = parser.parse_args()

    source_path = V118_STAGE1 / "object_track_prefix_rows.parquet"
    v118_summary_path = V118_STAGE1 / "stage1_semantic_track_v2_summary.json"
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    source = pd.read_parquet(source_path)
    source = source[source["seq"].astype(str).isin(SEQ_IDS)].copy()
    source["seq"] = source["seq"].astype(str)
    source["track_id"] = source["track_id"].astype(int)
    source["frame_id"] = source["frame_id"].astype(int)
    source = source.sort_values(["seq", "track_id", "frame_id"]).reset_index(drop=True)

    semv3_full = build_semv3_rows(source, lambda_p=args.lambda_p, tau_reobs=args.tau_reobs)
    parity_rows, violations = prefix_parity(
        source,
        semv3_full,
        lambda_p=args.lambda_p,
        tau_reobs=args.tau_reobs,
        sample_limit=args.sample_limit,
    )
    semv3 = semv3_full.drop(
        columns=[col for col in LEGACY_FORMULA_COLUMNS if col in semv3_full.columns]
    )
    running_rows = final_rows(semv3)

    OUT.mkdir(parents=True, exist_ok=True)
    prefix_path = OUT / "semv3_prefix_rows.parquet"
    semv3.to_parquet(prefix_path, index=False)
    write_csv(OUT / "semv3_running_summary.csv", running_rows)
    write_csv(OUT / "semv3_prefix_parity_rows.csv", parity_rows)
    write_csv(OUT / "semv3_future_leakage_violation_rows.csv", violations)

    v118_summary = read_json(v118_summary_path)
    finite_shape = bool(np.isfinite(semv3["semv3_shape_score_prefix"].to_numpy(dtype=np.float64)).all())
    role_basis_values = sorted(set(source["role_consistency_basis"].astype(str)))
    motion_modes = sorted(set(source["motion_compensation_mode"].astype(str)))
    seq_summaries = []
    for seq, group in semv3.groupby("seq", sort=True):
        seq_summaries.append(
            {
                "seq": str(seq),
                "prefix_row_count": int(len(group)),
                "track_count": int(group["track_id"].nunique()),
                "visibility_mean": float(group["semv3_visibility_prefix"].mean()),
                "role_prior_mean": float(group["semv3_role_prior_prefix"].mean()),
                "shape_score_mean": float(group["semv3_shape_score_prefix"].mean()),
                "reobs_score_mean": float(group["semv3_reobs_score_prefix"].mean()),
                "motion_residual_mean": float(group["semv3_motion_residual_prefix"].mean()),
            }
        )

    prefix_leakage_gate = len(violations) == 0 and all(row["pass"] for row in parity_rows)
    role_basis_gate = role_basis_values == ["running_label_histogram_before_path_role_mapping"]
    motion_compensation_gate = motion_modes == ["causal_static_background_median_centroid_proxy"]
    output_columns = set(semv3.columns)
    legacy_formula_output_gate = not any(col in output_columns for col in LEGACY_FORMULA_COLUMNS)
    output_scope_gate = "semantic_geometry_utility" not in output_columns and legacy_formula_output_gate
    semv3_ready = bool(
        len(semv3) > 0
        and prefix_leakage_gate
        and role_basis_gate
        and finite_shape
        and motion_compensation_gate
        and output_scope_gate
        and bool(v118_summary.get("future_leakage_gate"))
        and int(v118_summary.get("future_leakage_violation_count", -1)) == 0
    )
    blockers: list[str] = []
    if not prefix_leakage_gate:
        blockers.append("semv3_prefix_leakage_failed")
    if not role_basis_gate:
        blockers.append("role_prior_not_label_entropy_basis")
    if not finite_shape:
        blockers.append("shape_mad_nonfinite")
    if not motion_compensation_gate:
        blockers.append("motion_compensation_mode_not_allowed")
    if not output_scope_gate:
        blockers.append("geometry_utility_or_legacy_persistence_output_detected")
    if not bool(v118_summary.get("future_leakage_gate")):
        blockers.append("source_prefix_sidecar_future_leakage_gate_false")

    summary = {
        "schema": "acl2_v119tf_stage1_semv3_sidecar_summary_v1",
        "semv3_ready": semv3_ready,
        "stage1_dependency": "SEM-V3",
        "source_prefix_rows": rel(source_path),
        "source_v118_summary": rel(v118_summary_path),
        "source_reuse_boundary": (
            "Uses v118 Stage-C cache-derived prefix observations as input, but recomputes v119 SEM-V3 "
            "visibility, role-prior, shape-MAD, reobservation, and motion cues. It does not reuse "
            "v118 semantic_persistence_prefix as the v119 cue."
        ),
        "lambda_p": float(args.lambda_p),
        "tau_reobs": float(args.tau_reobs),
        "image_area_pixels": IMAGE_AREA,
        "prefix_row_count": int(len(semv3)),
        "track_count": int(semv3.groupby(["seq", "track_id"]).ngroups),
        "seq_summaries": seq_summaries,
        "prefix_parity_row_count": len(parity_rows),
        "future_leakage_violation_count": len(violations),
        "prefix_leakage_gate": prefix_leakage_gate,
        "role_basis_gate": role_basis_gate,
        "role_basis_values": role_basis_values,
        "shape_mad_finite_gate": finite_shape,
        "motion_compensation_gate": motion_compensation_gate,
        "motion_compensation_modes": motion_modes,
        "output_scope_gate": output_scope_gate,
        "no_geometry_utility_output": output_scope_gate,
        "legacy_formula_output_gate": legacy_formula_output_gate,
        "dropped_legacy_formula_columns": [
            col for col in LEGACY_FORMULA_COLUMNS if col in semv3_full.columns
        ],
        "blockers": blockers,
        "outputs": {
            "prefix_rows": rel(prefix_path),
            "running_summary": rel(OUT / "semv3_running_summary.csv"),
            "prefix_parity_rows": rel(OUT / "semv3_prefix_parity_rows.csv"),
            "future_leakage_violation_rows": rel(OUT / "semv3_future_leakage_violation_rows.csv"),
            "summary": rel(OUT / "semv3_summary.json"),
            "report": rel(OUT / "SEM_V3_REPORT.md"),
        },
    }
    write_json(OUT / "semv3_summary.json", summary)
    report = [
        "# ACL2 v119-TF Stage1 SEM-V3 Sidecar",
        "",
        f"- semv3_ready: `{summary['semv3_ready']}`",
        f"- prefix_leakage_gate: `{summary['prefix_leakage_gate']}`",
        f"- future_leakage_violation_count: `{summary['future_leakage_violation_count']}`",
        f"- role_basis_gate: `{summary['role_basis_gate']}`",
        f"- shape_mad_finite_gate: `{summary['shape_mad_finite_gate']}`",
        f"- motion_compensation_gate: `{summary['motion_compensation_gate']}`",
        f"- no_geometry_utility_output: `{summary['no_geometry_utility_output']}`",
        "",
        "Boundary: this is a Stage1 dependency artifact. It does not claim runtime LB-AI-FIX success or v119 global success.",
    ]
    (OUT / "SEM_V3_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(clean_json(summary), indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
