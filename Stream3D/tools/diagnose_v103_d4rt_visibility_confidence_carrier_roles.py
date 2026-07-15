#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
AUDIT_ROOT = STREAM3D_ROOT / "outputs" / "audit"

DEFAULT_PHASES1_ROOT = AUDIT_ROOT / "v103_supp_phaseS1_multirole_carriers_objectanchor_r3_r1"
DEFAULT_OUTPUT_ROOT = AUDIT_ROOT / "v103_r2_d4rt_visibility_confidence_carrier_role_diagnostic_r1"

PHASE_ID = "v103_r2_d4rt_visibility_confidence_carrier_role_diagnostic"

ROLE_MASKS = [
    ("A_anchor", "is_A_anchor"),
    ("S_support", "is_S_support"),
    ("V_veto", "is_V_veto"),
    ("S_and_V_overlap", "__support_and_veto__"),
    ("U_uncertain", "is_U_uncertain"),
]

STAT_COLUMNS = [
    "visibility_rate",
    "in_image_rate",
    "confidence_mean_in_image",
    "r_geo",
    "reliability_s2",
    "normalized_jitter",
    "broad_mask_participation_rate",
    "object_like_mask_rate",
    "competing_mask_conflict_rate",
    "semantic_contradiction_rate",
]

QUANTILES = [0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0]

LOW_FRACTION_TESTS = [
    ("visibility_rate", "<", 0.10),
    ("visibility_rate", "<", 0.15),
    ("visibility_rate", "<", 0.20),
    ("in_image_rate", "<", 0.10),
    ("in_image_rate", "<", 0.15),
    ("confidence_mean_in_image", "<", 0.50),
    ("confidence_mean_in_image", "<", 0.90),
    ("confidence_mean_in_image", "<", 0.9990),
    ("confidence_mean_in_image", "<", 0.9999),
]

SUPPORT_GATE_PROFILES = [
    ("baseline_no_extra_gate", 0.0, 0.0),
    ("support_visibility_ge_0p08", 0.08, 0.0),
    ("support_visibility_ge_0p10", 0.10, 0.0),
    ("support_visibility_ge_0p15", 0.15, 0.0),
    ("support_visibility_ge_0p20", 0.20, 0.0),
    ("support_visibility_ge_0p40_core_diagnostic", 0.40, 0.0),
    ("support_visibility_ge_0p10_conf_ge_0p9990", 0.10, 0.9990),
    ("support_visibility_ge_0p10_conf_ge_0p9999", 0.10, 0.9999),
    ("support_visibility_ge_0p15_conf_ge_0p9999", 0.15, 0.9999),
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
    if isinstance(value, (list, tuple)):
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


def _role_mask(df: pd.DataFrame, role_column: str) -> np.ndarray:
    if role_column == "__support_and_veto__":
        return df["is_S_support"].astype(bool).to_numpy() & df["is_V_veto"].astype(bool).to_numpy()
    return df[role_column].astype(bool).to_numpy()


def _safe_quantiles(values: pd.Series) -> dict[str, float]:
    arr = values.astype(float).to_numpy()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {f"p{int(q * 100):02d}": 0.0 for q in QUANTILES}
    quantiles = np.quantile(arr, QUANTILES)
    return {f"p{int(q * 100):02d}": float(v) for q, v in zip(QUANTILES, quantiles)}


def _safe_mean(values: pd.Series) -> float:
    arr = values.astype(float).to_numpy()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    return float(np.mean(arr))


def _safe_median(values: pd.Series) -> float:
    arr = values.astype(float).to_numpy()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    return float(np.median(arr))


def _role_stat_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scene in sorted(df["scene_id"].unique().tolist()):
        sdf = df[df["scene_id"] == scene]
        total = int(len(sdf))
        for role_name, role_column in ROLE_MASKS:
            sub = sdf[_role_mask(sdf, role_column)]
            for metric in STAT_COLUMNS:
                row = {
                    "schema_version": "stream4d_v103_r2_d4rt_vis_conf_role_stat_row_v1",
                    "phase_id": PHASE_ID,
                    "scene_id": scene,
                    "role_mask": role_name,
                    "metric_name": metric,
                    "carrier_count": int(len(sub)),
                    "scene_total_carrier_count": total,
                    "role_rate": float(len(sub) / max(total, 1)),
                    "uses_gt": False,
                    "uses_future": False,
                }
                row.update(_safe_quantiles(sub[metric]))
                rows.append(row)
    return rows


def _low_fraction_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scene in sorted(df["scene_id"].unique().tolist()):
        sdf = df[df["scene_id"] == scene]
        for role_name, role_column in ROLE_MASKS:
            sub = sdf[_role_mask(sdf, role_column)]
            denom = int(len(sub))
            for metric, op, threshold in LOW_FRACTION_TESTS:
                vals = sub[metric].astype(float).to_numpy()
                if op != "<":
                    raise ValueError(op)
                hit = np.isfinite(vals) & (vals < float(threshold))
                rows.append(
                    {
                        "schema_version": "stream4d_v103_r2_d4rt_vis_conf_low_fraction_row_v1",
                        "phase_id": PHASE_ID,
                        "scene_id": scene,
                        "role_mask": role_name,
                        "metric_name": metric,
                        "operator": op,
                        "threshold": float(threshold),
                        "hit_count": int(np.count_nonzero(hit)),
                        "carrier_count": denom,
                        "hit_rate": float(np.count_nonzero(hit) / max(denom, 1)),
                        "uses_gt": False,
                        "uses_future": False,
                    }
                )
    return rows


def _support_gate_simulation_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scene in sorted(df["scene_id"].unique().tolist()):
        sdf = df[df["scene_id"] == scene]
        support = sdf[sdf["is_S_support"].astype(bool)]
        baseline_count = int(len(support))
        for profile_name, min_visibility, min_confidence in SUPPORT_GATE_PROFILES:
            keep = (
                (support["visibility_rate"].astype(float).to_numpy() >= float(min_visibility))
                & (support["confidence_mean_in_image"].astype(float).to_numpy() >= float(min_confidence))
            )
            kept = support[keep]
            removed = support[~keep]
            rows.append(
                {
                    "schema_version": "stream4d_v103_r2_d4rt_vis_conf_support_gate_sim_row_v1",
                    "phase_id": PHASE_ID,
                    "scene_id": scene,
                    "profile_name": profile_name,
                    "min_visibility_rate": float(min_visibility),
                    "min_confidence_mean_in_image": float(min_confidence),
                    "baseline_support_count": baseline_count,
                    "kept_support_count": int(len(kept)),
                    "removed_support_count": int(len(removed)),
                    "kept_support_rate": float(len(kept) / max(baseline_count, 1)),
                    "removed_support_rate": float(len(removed) / max(baseline_count, 1)),
                    "kept_broad_mean": _safe_mean(kept["broad_mask_participation_rate"]),
                    "removed_broad_mean": _safe_mean(removed["broad_mask_participation_rate"]),
                    "kept_object_like_mean": _safe_mean(kept["object_like_mask_rate"]),
                    "removed_object_like_mean": _safe_mean(removed["object_like_mask_rate"]),
                    "kept_reliability_s2_median": _safe_median(kept["reliability_s2"]),
                    "removed_reliability_s2_median": _safe_median(removed["reliability_s2"]),
                    "uses_gt": False,
                    "uses_future": False,
                }
            )
    return rows


def build(args: argparse.Namespace) -> dict[str, Any]:
    phase_s1_root = _project(args.phase_s1_root)
    output_root = _project(args.output_root)
    carrier_role_path = phase_s1_root / "carrier_role_rows.parquet"
    if not carrier_role_path.exists():
        raise FileNotFoundError(carrier_role_path)

    df = pd.read_parquet(carrier_role_path)
    missing = [c for c in ["scene_id", "is_S_support", *STAT_COLUMNS] if c not in df.columns]
    if missing:
        raise RuntimeError(f"carrier_role_rows missing columns: {missing}")

    role_stat_rows = _role_stat_rows(df)
    low_fraction_rows = _low_fraction_rows(df)
    support_gate_rows = _support_gate_simulation_rows(df)

    _write_csv(output_root / "role_visibility_confidence_quantile_rows.csv", role_stat_rows)
    _write_csv(output_root / "role_low_visibility_confidence_fraction_rows.csv", low_fraction_rows)
    _write_csv(output_root / "support_visibility_confidence_gate_simulation_rows.csv", support_gate_rows)

    support_low_visibility: dict[str, dict[str, float]] = {}
    confidence_low_rates: dict[str, dict[str, float]] = {}
    gate_summary: dict[str, dict[str, Any]] = {}
    for scene in sorted(df["scene_id"].unique().tolist()):
        sdf = df[df["scene_id"] == scene]
        support = sdf[sdf["is_S_support"].astype(bool)]
        anchor = sdf[sdf["is_A_anchor"].astype(bool)]
        support_low_visibility[scene] = {
            "support_visibility_lt_0p10_rate": float((support["visibility_rate"].astype(float) < 0.10).mean()) if len(support) else 0.0,
            "support_visibility_lt_0p15_rate": float((support["visibility_rate"].astype(float) < 0.15).mean()) if len(support) else 0.0,
            "anchor_visibility_lt_0p10_rate": float((anchor["visibility_rate"].astype(float) < 0.10).mean()) if len(anchor) else 0.0,
            "anchor_visibility_lt_0p15_rate": float((anchor["visibility_rate"].astype(float) < 0.15).mean()) if len(anchor) else 0.0,
        }
        confidence_low_rates[scene] = {
            "support_confidence_lt_0p9_rate": float((support["confidence_mean_in_image"].astype(float) < 0.90).mean()) if len(support) else 0.0,
            "support_confidence_lt_0p999_rate": float((support["confidence_mean_in_image"].astype(float) < 0.999).mean()) if len(support) else 0.0,
            "support_confidence_p01": float(np.quantile(support["confidence_mean_in_image"].astype(float).to_numpy(), 0.01)) if len(support) else 0.0,
            "anchor_confidence_p01": float(np.quantile(anchor["confidence_mean_in_image"].astype(float).to_numpy(), 0.01)) if len(anchor) else 0.0,
        }
        scene_gate_rows = [
            r
            for r in support_gate_rows
            if r["scene_id"] == scene
            and r["profile_name"] in {"support_visibility_ge_0p10", "support_visibility_ge_0p15", "support_visibility_ge_0p40_core_diagnostic"}
        ]
        gate_summary[scene] = {str(r["profile_name"]): {"kept_support_rate": r["kept_support_rate"], "removed_support_rate": r["removed_support_rate"]} for r in scene_gate_rows}

    summary = {
        "schema_version": "stream4d_v103_r2_d4rt_visibility_confidence_diagnostic_summary_v1",
        "phase_id": PHASE_ID,
        "decision": "DIAG_D4RT_VISIBILITY_CONFIDENCE_PARTIALLY_USED_CONFIDENCE_SATURATED_SUPPORT_VISIBILITY_UNDER_HARD_GATED",
        "phase_s1_root": phase_s1_root,
        "carrier_role_rows": carrier_role_path,
        "output_root": output_root,
        "row_count": int(len(df)),
        "scene_ids": sorted(df["scene_id"].unique().tolist()),
        "support_low_visibility": support_low_visibility,
        "confidence_low_rates": confidence_low_rates,
        "support_gate_summary": gate_summary,
        "interpretation": (
            "D4RT visibility/confidence are already in reliability_s0/s2 through r_geo and later incidence weights, "
            "but current S1 support role does not hard-gate visibility or confidence. Confidence is saturated in the current provider output, "
            "so it is weak as a denoising signal; visibility remains useful for a conservative support diagnostic gate."
        ),
        "recommended_next_probe": (
            "Run a support-visibility-gated S1/Phase9e branch before AP claims: keep A_anchor semantics unchanged, "
            "add support_min_visibility around 0.10-0.15 as an ablation, and do not rely on confidence alone unless a new D4RT checkpoint "
            "produces a less saturated confidence distribution."
        ),
        "uses_gt": False,
        "uses_future": False,
    }
    _write_json(output_root / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-s1-root", type=Path, default=DEFAULT_PHASES1_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> int:
    summary = build(parse_args())
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
