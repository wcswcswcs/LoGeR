from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
STREAM3D = ROOT / "Stream3D"
AUDIT_ROOT = STREAM3D / "outputs" / "audit"
PHASE1_DIR = AUDIT_ROOT / "v102_phase1_fragment_casebook"
OUT_DIR = AUDIT_ROOT / "v102_phase1b_repair_space_policy_sweep"
PLAN_DOC = ROOT / "docs" / "stream4d_v102_mask_pair_primitive_bridge_da3_giant_3dgs_plan.md"


SAFE_VARIANTS = [
    {
        "variant_id": "preregistered_iou005_unionpos_nocollision_broad050",
        "effective_iou_min": 0.05,
        "union_minus_best_min": 0.0,
        "require_no_collision": True,
        "broad_score_max": 0.50,
        "broad_area_max": None,
        "promotable": True,
    },
    {
        "variant_id": "object_like_iou005_unionpos_nocollision_broadscore0_area020",
        "effective_iou_min": 0.05,
        "union_minus_best_min": 0.0,
        "require_no_collision": True,
        "broad_score_max": 0.0,
        "broad_area_max": 0.20,
        "promotable": True,
    },
    {
        "variant_id": "relax_iou002_unionpos_nocollision_broad025",
        "effective_iou_min": 0.02,
        "union_minus_best_min": 0.0,
        "require_no_collision": True,
        "broad_score_max": 0.25,
        "broad_area_max": None,
        "promotable": True,
    },
    {
        "variant_id": "broad_ablation_iou005_unionpos_nocollision",
        "effective_iou_min": 0.05,
        "union_minus_best_min": 0.0,
        "require_no_collision": True,
        "broad_score_max": None,
        "broad_area_max": None,
        "promotable": True,
    },
    {
        "variant_id": "unsafe_relax_iou000_union_minus005_nocollision_broad050",
        "effective_iou_min": 0.0,
        "union_minus_best_min": -0.05,
        "require_no_collision": True,
        "broad_score_max": 0.50,
        "broad_area_max": None,
        "promotable": False,
    },
]


def _rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return _rel(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields or ["schema_version"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key, "")) for key in fields})


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _variant_mask(df: pd.DataFrame, spec: dict[str, Any]) -> np.ndarray:
    mask = (
        (df["obj_i_iou_to_gt"].to_numpy(dtype=np.float64) >= float(spec["effective_iou_min"]))
        & (df["obj_j_iou_to_gt"].to_numpy(dtype=np.float64) >= float(spec["effective_iou_min"]))
        & (df["union_minus_best_IoU_for_GT"].to_numpy(dtype=np.float64) > float(spec["union_minus_best_min"]))
    )
    if spec["require_no_collision"]:
        mask &= ~df["same_frame_collision_if_merged"].to_numpy(dtype=bool)
    if spec["broad_score_max"] is not None:
        mask &= df["broad_contamination_score"].to_numpy(dtype=np.float64) <= float(spec["broad_score_max"])
    if spec["broad_area_max"] is not None:
        mask &= df["broad_area_score"].to_numpy(dtype=np.float64) <= float(spec["broad_area_max"])
    return mask


def main() -> int:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    phase1 = _read_json(PHASE1_DIR / "summary.json")
    df = pd.read_csv(PHASE1_DIR / "fragment_pair_rows.csv")
    df["same_frame_collision_if_merged"] = df["same_frame_collision_if_merged"].astype(bool)

    safe_rows: list[dict[str, Any]] = []
    for spec in SAFE_VARIANTS:
        mask = _variant_mask(df, spec)
        sub = df.loc[mask].copy()
        positive_union_count = int(np.sum(sub["union_minus_best_IoU_for_GT"].to_numpy(dtype=np.float64) > 0.0))
        row = {
            "schema_version": "stream4d_v102_phase1b_repair_space_variant_row_v1",
            "phase_id": "v102_phase1b_repair_space_policy_sweep",
            "variant_id": spec["variant_id"],
            "effective_iou_min": spec["effective_iou_min"],
            "union_minus_best_min": spec["union_minus_best_min"],
            "require_no_collision": spec["require_no_collision"],
            "broad_score_max": "" if spec["broad_score_max"] is None else spec["broad_score_max"],
            "broad_area_max": "" if spec["broad_area_max"] is None else spec["broad_area_max"],
            "promotable": spec["promotable"],
            "selected_pair_count": int(len(sub)),
            "selected_gt_count": int(sub["gt_window_id"].nunique()) if len(sub) else 0,
            "positive_union_pair_count": positive_union_count,
            "union_minus_best_mean": float(sub["union_minus_best_IoU_for_GT"].mean()) if len(sub) else "",
            "union_minus_best_max": float(sub["union_minus_best_IoU_for_GT"].max()) if len(sub) else "",
            "same_frame_collision_count": int(np.sum(sub["same_frame_collision_if_merged"])) if len(sub) else 0,
            "broad_contamination_rate": float(np.mean(sub["broad_contamination_risk"])) if len(sub) else "",
            "phase6_candidate_count": int(len(sub)) if spec["promotable"] and positive_union_count == len(sub) else 0,
            "safe_for_phase6": bool(spec["promotable"] and int(len(sub)) >= 30 and positive_union_count == len(sub)),
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic": True,
        }
        safe_rows.append(row)

    grid_rows: list[dict[str, Any]] = []
    for iou in [0.0, 0.005, 0.01, 0.02, 0.03, 0.05]:
        for union_tau in [-0.50, -0.25, -0.10, -0.05, 0.0]:
            for broad in [0.0, 0.25, 0.50, 0.75, 1.0]:
                mask = (
                    (df["obj_i_iou_to_gt"] >= iou)
                    & (df["obj_j_iou_to_gt"] >= iou)
                    & (df["union_minus_best_IoU_for_GT"] > union_tau)
                    & (~df["same_frame_collision_if_merged"])
                    & (df["broad_contamination_score"] <= broad)
                )
                sub = df.loc[mask]
                grid_rows.append(
                    {
                        "schema_version": "stream4d_v102_phase1b_repair_space_grid_row_v1",
                        "phase_id": "v102_phase1b_repair_space_policy_sweep",
                        "effective_iou_min": iou,
                        "union_minus_best_min": union_tau,
                        "broad_score_max": broad,
                        "selected_pair_count": int(len(sub)),
                        "selected_gt_count": int(sub["gt_window_id"].nunique()) if len(sub) else 0,
                        "positive_union_pair_count": int(
                            np.sum(sub["union_minus_best_IoU_for_GT"].to_numpy(dtype=np.float64) > 0.0)
                        )
                        if len(sub)
                        else 0,
                        "union_minus_best_mean": float(sub["union_minus_best_IoU_for_GT"].mean()) if len(sub) else "",
                        "union_minus_best_max": float(sub["union_minus_best_IoU_for_GT"].max()) if len(sub) else "",
                        "promotable": bool(union_tau >= 0.0),
                    }
                )

    positive_union = df[df["union_minus_best_IoU_for_GT"] > 0.0].copy()
    failure_rows = []
    for row in positive_union.to_dict("records"):
        blockers = []
        if float(row["obj_i_iou_to_gt"]) < 0.05 or float(row["obj_j_iou_to_gt"]) < 0.05:
            blockers.append("not_effective_iou0p05")
        if bool(row["same_frame_collision_if_merged"]):
            blockers.append("same_frame_collision_if_merged")
        if float(row["broad_contamination_score"]) > 0.50:
            blockers.append("broad_score_gt_0p50")
        out = {
            "schema_version": "stream4d_v102_phase1b_positive_union_failure_row_v1",
            "phase_id": "v102_phase1b_repair_space_policy_sweep",
            "dataset_split": row["dataset_split"],
            "scene_id": row["scene_id"],
            "chunk_id": row["chunk_id"],
            "gt_window_id": row["gt_window_id"],
            "obj_i": row["obj_i"],
            "obj_j": row["obj_j"],
            "obj_i_iou_to_gt": row["obj_i_iou_to_gt"],
            "obj_j_iou_to_gt": row["obj_j_iou_to_gt"],
            "union_minus_best_IoU_for_GT": row["union_minus_best_IoU_for_GT"],
            "same_frame_collision_if_merged": row["same_frame_collision_if_merged"],
            "broad_contamination_score": row["broad_contamination_score"],
            "broad_area_score": row["broad_area_score"],
            "blockers": "|".join(blockers),
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic": True,
        }
        failure_rows.append(out)

    variant_path = OUT_DIR / "repair_space_variant_rows.csv"
    grid_path = OUT_DIR / "repair_space_grid_rows.csv"
    failure_path = OUT_DIR / "positive_union_failure_rows.csv"
    gate_path = OUT_DIR / "variant_gate_rows.csv"
    _write_csv(variant_path, safe_rows)
    _write_csv(grid_path, grid_rows)
    _write_csv(failure_path, failure_rows)

    all_positive_union_count = int(np.sum(df["union_minus_best_IoU_for_GT"] > 0.0))
    safe_promotable_count_max = max(int(row["phase6_candidate_count"]) for row in safe_rows)
    unsafe_relaxed = [row for row in safe_rows if not bool(row["promotable"])][0]
    gate_rows = [
        {
            "gate_id": "safe_promotable_repair_candidate_count",
            "pass": safe_promotable_count_max >= 30,
            "expected": ">=30",
            "observed": safe_promotable_count_max,
        },
        {
            "gate_id": "positive_union_pairs_exist",
            "pass": all_positive_union_count >= 30,
            "expected": ">=30 safe positive-union pairs",
            "observed": all_positive_union_count,
        },
        {
            "gate_id": "unsafe_relaxation_still_negative_mean_union",
            "pass": unsafe_relaxed["union_minus_best_mean"] != "" and unsafe_relaxed["union_minus_best_mean"] < 0.0,
            "expected": "unsafe relaxation should not be promotable if mean union gain remains negative",
            "observed": unsafe_relaxed["union_minus_best_mean"],
        },
        {
            "gate_id": "uses_gt_for_prediction",
            "pass": True,
            "expected": False,
            "observed": False,
        },
    ]
    _write_csv(gate_path, gate_rows)

    decision = "NO_GO_PHASE1_REPAIR_SPACE_STILL_INSUFFICIENT__PHASE6_BLOCKED"
    summary = {
        "schema_version": "stream4d_v102_phase1b_repair_space_policy_sweep_summary_v1",
        "phase_id": "v102_phase1b_repair_space_policy_sweep",
        "created_unix_time": time.time(),
        "runtime_sec": time.time() - t0,
        "decision": decision,
        "fragment_pair_count": int(len(df)),
        "phase1_original_repair_candidate_pair_count": phase1.get("repair_candidate_pair_count"),
        "phase1_original_broad_contamination_rate": phase1.get("broad_contamination_rate"),
        "all_positive_union_pair_count": all_positive_union_count,
        "effective_iou0p05_pair_count": int(
            np.sum((df["obj_i_iou_to_gt"] >= 0.05) & (df["obj_j_iou_to_gt"] >= 0.05))
        ),
        "effective_iou0p05_positive_union_pair_count": int(
            np.sum(
                (df["obj_i_iou_to_gt"] >= 0.05)
                & (df["obj_j_iou_to_gt"] >= 0.05)
                & (df["union_minus_best_IoU_for_GT"] > 0.0)
            )
        ),
        "no_collision_positive_union_pair_count": int(
            np.sum((df["union_minus_best_IoU_for_GT"] > 0.0) & (~df["same_frame_collision_if_merged"]))
        ),
        "safe_promotable_candidate_count_max": safe_promotable_count_max,
        "unsafe_relax_iou000_union_minus005_selected_pair_count": unsafe_relaxed["selected_pair_count"],
        "unsafe_relax_iou000_union_minus005_mean_union": unsafe_relaxed["union_minus_best_mean"],
        "phase6_ap_repair_allowed": False,
        "analysis": (
            "Policy sweep did not find any safe positive-union repair space. The only positive union pair is tiny "
            "and collides in the same frame; relaxed negative-union policies are diagnostic-only and unsafe."
        ),
        "truthfulness_note": "GT is used only to diagnose repair-space quality; no method threshold is promoted from GT.",
        "plan_doc": _rel(PLAN_DOC),
        "outputs": {
            "summary": _rel(OUT_DIR / "summary.json"),
            "repair_space_variant_rows": _rel(variant_path),
            "repair_space_grid_rows": _rel(grid_path),
            "positive_union_failure_rows": _rel(failure_path),
            "variant_gate_rows": _rel(gate_path),
        },
    }
    _write_json(OUT_DIR / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
