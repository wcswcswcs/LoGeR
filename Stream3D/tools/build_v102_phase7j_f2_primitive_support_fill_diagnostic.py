#!/usr/bin/env python3
"""Use Phase7h primitive-support rows as conservative missing-frame fill for F2 objects."""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
if str(STREAM3D_ROOT) not in sys.path:
    sys.path.insert(0, str(STREAM3D_ROOT))

from tools.run_v65_scene_multiview_ap import _load_gt_2d  # noqa: E402
from tools import build_v102_phase7d_phase7c_materialized_ap_diagnostic as p7d  # noqa: E402
from tools import build_v102_phase7i_same_chunk_f2_vs_phase7h_comparison as p7i  # noqa: E402


AUDIT_ROOT = STREAM3D_ROOT / "outputs" / "audit"
OUT_DIR = AUDIT_ROOT / "v102_phase7j_f2_primitive_support_fill_diagnostic"
PLAN_DOC = REPO_ROOT / "docs" / "stream4d_v102_mask_pair_primitive_bridge_da3_giant_3dgs_plan.md"
F2_ROWS = AUDIT_ROOT / "v100_phase2c_overlap3_local_repair" / "mv_object_frame_mask_rows.parquet"
PHASE7H_SUMMARY = AUDIT_ROOT / "v102_phase7h_chunk32_primitive_support_shape_diagnostic" / "summary.json"
PHASE7H_ROWS = AUDIT_ROOT / "v102_phase7h_chunk32_primitive_support_shape_diagnostic" / "materialized_expanded_rows.csv"

PHASE_ID = "v102_phase7j_f2_primitive_support_fill_diagnostic"
F2_VARIANT = "F2_v100_chunk32_overlap3_surfel_maskview_thr018_p2d2"
VARIANT_PREFIX = "F2P_v102_phase7j"

VARIANTS = [
    {
        "variant_id": "J0_f2_same_chunk_baseline",
        "expand": False,
        "min_shared_mask_count": 99,
        "min_shared_ratio": 2.0,
        "min_support_fraction": 2.0,
        "min_mask_coverage": 2.0,
        "drop_broad": False,
        "allow_replace_existing_object_frame": False,
    },
    {
        "variant_id": "J1_map1_ratio050_support080_cover010",
        "expand": True,
        "min_shared_mask_count": 1,
        "min_shared_ratio": 0.50,
        "min_support_fraction": 0.80,
        "min_mask_coverage": 0.010,
        "drop_broad": False,
        "allow_replace_existing_object_frame": False,
    },
    {
        "variant_id": "J2_map2_ratio040_support080_cover010",
        "expand": True,
        "min_shared_mask_count": 2,
        "min_shared_ratio": 0.40,
        "min_support_fraction": 0.80,
        "min_mask_coverage": 0.010,
        "drop_broad": False,
        "allow_replace_existing_object_frame": False,
    },
    {
        "variant_id": "J3_map1_ratio030_support090_cover015",
        "expand": True,
        "min_shared_mask_count": 1,
        "min_shared_ratio": 0.30,
        "min_support_fraction": 0.90,
        "min_mask_coverage": 0.015,
        "drop_broad": False,
        "allow_replace_existing_object_frame": False,
    },
    {
        "variant_id": "J4_map1_ratio030_support070_cover005",
        "expand": True,
        "min_shared_mask_count": 1,
        "min_shared_ratio": 0.30,
        "min_support_fraction": 0.70,
        "min_mask_coverage": 0.005,
        "drop_broad": False,
        "allow_replace_existing_object_frame": False,
    },
    {
        "variant_id": "J5_map1_ratio030_support070_cover005_dropbroad",
        "expand": True,
        "min_shared_mask_count": 1,
        "min_shared_ratio": 0.30,
        "min_support_fraction": 0.70,
        "min_mask_coverage": 0.005,
        "drop_broad": True,
        "allow_replace_existing_object_frame": False,
    },
    {
        "variant_id": "J6_replace_map1_ratio030_support090_cover015",
        "expand": True,
        "min_shared_mask_count": 1,
        "min_shared_ratio": 0.30,
        "min_support_fraction": 0.90,
        "min_mask_coverage": 0.015,
        "drop_broad": False,
        "allow_replace_existing_object_frame": True,
    },
    {
        "variant_id": "J7_replace_map1_ratio030_support080_cover010",
        "expand": True,
        "min_shared_mask_count": 1,
        "min_shared_ratio": 0.30,
        "min_support_fraction": 0.80,
        "min_mask_coverage": 0.010,
        "drop_broad": False,
        "allow_replace_existing_object_frame": True,
    },
    {
        "variant_id": "J8_replace_map2_ratio040_support080_cover010",
        "expand": True,
        "min_shared_mask_count": 2,
        "min_shared_ratio": 0.40,
        "min_support_fraction": 0.80,
        "min_mask_coverage": 0.010,
        "drop_broad": False,
        "allow_replace_existing_object_frame": True,
    },
]


def _rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return p.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return p.as_posix()


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
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


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


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return float(default)
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _same_chunk_rows(path: Path, variant_id: str) -> pd.DataFrame:
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    return p7i._filter_same_chunk(df, variant_id)


def _phase7h_best_rows() -> tuple[str, pd.DataFrame]:
    summary = _read_json(PHASE7H_SUMMARY)
    variant = f"P2_v102_phase7h_{summary.get('best_variant_id')}"
    return variant, p7i._filter_same_chunk(pd.read_csv(PHASE7H_ROWS), variant)


def _object_scores(rows: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = defaultdict(float)
    for row in rows.to_dict(orient="records"):
        oid = str(row["mv_object_id"])
        out[oid] = max(float(out[oid]), _num(row.get("score"), 1.0))
    return out


def _dominant_gt_by_object(rows: pd.DataFrame) -> dict[str, str]:
    mask_path_by_frame, _mask_source = p7d._mask_path_lookup()
    label_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    counts_by_object: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows.to_dict(orient="records"):
        frame = int(row["frame_id"])
        mask_id = int(_num(row.get("selected_mask_id")))
        if frame not in label_cache:
            mask_path = mask_path_by_frame.get((p7d.SCENE_ID, frame))
            if mask_path is None or not mask_path.exists():
                continue
            label = p7d._read_label(mask_path)
            gt = _load_gt_2d(p7d.SCENE_ID, frame, label.shape)
            label_cache[frame] = (label, gt)
        label, gt = label_cache[frame]
        vals = gt[label == mask_id]
        vals = vals[vals > 0]
        if vals.size == 0:
            continue
        ids, cnt = np.unique(vals, return_counts=True)
        counts_by_object[str(row["mv_object_id"])][str(int(ids[int(np.argmax(cnt))]))] += int(np.max(cnt))
    out: dict[str, str] = {}
    for oid, counts in counts_by_object.items():
        out[oid] = counts.most_common(1)[0][0] if counts else ""
    return out


def _mask_gt(label: np.ndarray, gt: np.ndarray, mask_id: int) -> str:
    vals = gt[label == int(mask_id)]
    vals = vals[vals > 0]
    if vals.size == 0:
        return ""
    ids, cnt = np.unique(vals, return_counts=True)
    return str(int(ids[int(np.argmax(cnt))]))


def _build_mapping(f2_rows: pd.DataFrame, p7h_rows: pd.DataFrame) -> list[dict[str, Any]]:
    f2_key_to_oid = {
        (int(row["frame_id"]), int(row["selected_mask_id"])): str(row["mv_object_id"])
        for row in f2_rows.to_dict(orient="records")
    }
    f2_object_frame_count = f2_rows.groupby("mv_object_id")["frame_id"].nunique().astype(int).to_dict()
    mapping_rows: list[dict[str, Any]] = []
    base = p7h_rows[p7h_rows["phase7h_row_source"].astype(str) == "base_phase7d"].copy()
    for p7h_oid, rows in base.groupby("mv_object_id"):
        hits = Counter()
        for row in rows.to_dict(orient="records"):
            key = (int(row["frame_id"]), int(row["selected_mask_id"]))
            if key in f2_key_to_oid:
                hits[f2_key_to_oid[key]] += 1
        if not hits:
            mapping_rows.append(
                {
                    "schema_version": "stream4d_v102_phase7j_mapping_row_v1",
                    "phase_id": PHASE_ID,
                    "phase7h_mv_object_id": str(p7h_oid),
                    "mapped_f2_mv_object_id": "",
                    "shared_mask_count": 0,
                    "phase7h_base_mask_count": int(len(rows)),
                    "f2_object_frame_count": 0,
                    "shared_ratio_vs_phase7h": 0.0,
                    "ambiguous_tie_count": 0,
                    "uses_gt_for_prediction": False,
                }
            )
            continue
        best_count = hits.most_common(1)[0][1]
        tied = sorted([oid for oid, count in hits.items() if count == best_count])
        f2_oid = tied[0]
        mapping_rows.append(
            {
                "schema_version": "stream4d_v102_phase7j_mapping_row_v1",
                "phase_id": PHASE_ID,
                "phase7h_mv_object_id": str(p7h_oid),
                "mapped_f2_mv_object_id": f2_oid,
                "shared_mask_count": int(best_count),
                "phase7h_base_mask_count": int(len(rows)),
                "f2_object_frame_count": int(f2_object_frame_count.get(f2_oid, 0)),
                "shared_ratio_vs_phase7h": float(best_count / max(1, len(rows))),
                "ambiguous_tie_count": int(len(tied)),
                "uses_gt_for_prediction": False,
            }
        )
    return mapping_rows


def _expand_f2(
    f2_rows: pd.DataFrame,
    p7h_rows: pd.DataFrame,
    mapping_rows: list[dict[str, Any]],
    spec: dict[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    base_records = [dict(row) | {"phase7j_row_source": "base_f2"} for row in f2_rows.to_dict(orient="records")]
    if not bool(spec["expand"]):
        return pd.DataFrame(base_records), [], {
            "mapping_available_count": sum(1 for row in mapping_rows if row["mapped_f2_mv_object_id"]),
            "candidate_examined_count": 0,
            "proposed_fill_count": 0,
            "accepted_fill_count": 0,
            "wta_drop_count": 0,
            "rejected_mapping_count": 0,
            "rejected_existing_object_frame_count": 0,
            "rejected_occupied_mask_count": 0,
            "rejected_threshold_count": 0,
            "rejected_broad_count": 0,
            "accepted_replacement_count": 0,
            "accepted_missing_fill_count": 0,
        }

    mapping = {
        str(row["phase7h_mv_object_id"]): row
        for row in mapping_rows
        if str(row.get("mapped_f2_mv_object_id", ""))
        and int(_num(row.get("shared_mask_count"))) >= int(spec["min_shared_mask_count"])
        and _num(row.get("shared_ratio_vs_phase7h")) >= float(spec["min_shared_ratio"])
    }
    f2_scores = _object_scores(f2_rows)
    existing_object_frame = {(str(row["mv_object_id"]), int(row["frame_id"])) for row in base_records}
    mask_to_object_frame = {
        (int(row["frame_id"]), int(row["selected_mask_id"])): (str(row["mv_object_id"]), int(row["frame_id"]))
        for row in base_records
    }
    occupied_mask = set(mask_to_object_frame)
    proposed: list[dict[str, Any]] = []
    candidate_examined_count = 0
    rejected_mapping_count = 0
    rejected_existing_object_frame_count = 0
    rejected_occupied_mask_count = 0
    rejected_threshold_count = 0
    rejected_broad_count = 0

    fill_rows = p7h_rows[p7h_rows["phase7h_row_source"].astype(str) == "primitive_support_missing_frame_fill"]
    for row in fill_rows.to_dict(orient="records"):
        candidate_examined_count += 1
        p7h_oid = str(row["mv_object_id"])
        map_row = mapping.get(p7h_oid)
        if map_row is None:
            rejected_mapping_count += 1
            continue
        f2_oid = str(map_row["mapped_f2_mv_object_id"])
        frame = int(row["frame_id"])
        mask_id = int(row["selected_mask_id"])
        replace_key = None
        if (f2_oid, frame) in existing_object_frame:
            if not bool(spec.get("allow_replace_existing_object_frame", False)):
                rejected_existing_object_frame_count += 1
                continue
            replace_key = (f2_oid, frame)
        occupied_by = mask_to_object_frame.get((frame, mask_id))
        if occupied_by is not None and occupied_by != replace_key:
            rejected_occupied_mask_count += 1
            continue
        if replace_key is None and (f2_oid, frame) in existing_object_frame:
            rejected_existing_object_frame_count += 1
            continue
        if bool(spec["drop_broad"]) and _bool(row.get("broad_background_risk")):
            rejected_broad_count += 1
            continue
        if _num(row.get("primitive_support_fraction")) < float(spec["min_support_fraction"]):
            rejected_threshold_count += 1
            continue
        if _num(row.get("primitive_support_mask_coverage")) < float(spec["min_mask_coverage"]):
            rejected_threshold_count += 1
            continue
        new = dict(row)
        new["schema_version"] = "stream4d_v102_phase7j_f2_primitive_support_fill_row_v1"
        new["phase_id"] = PHASE_ID
        new["dataset_split"] = "dev"
        new["variant_id"] = f"{VARIANT_PREFIX}_{spec['variant_id']}"
        new["mv_object_id"] = f2_oid
        new["object_id"] = f2_oid
        new["source_phase7h_mv_object_id"] = p7h_oid
        new["source_phase7h_variant_id"] = row.get("variant_id", "")
        new["score"] = f2_scores.get(f2_oid, 1.0)
        new["object_score"] = f2_scores.get(f2_oid, 1.0)
        new["score_policy"] = "keep_mapped_f2_object_score_after_phase7h_primitive_support_fill"
        new["object_id_policy"] = "v100_f2_identity_with_phase7h_primitive_support_missing_frame_fill"
        new["object_birth_scope"] = "phase7j_f2_backbone_primitive_support_fill"
        new["phase7j_row_source"] = "phase7h_primitive_support_fill_mapped_to_f2"
        new["phase7j_update_mode"] = (
            "replace_existing_object_frame" if replace_key is not None else "fill_missing_object_frame"
        )
        new["phase7j_mapping_shared_mask_count"] = int(map_row["shared_mask_count"])
        new["phase7j_mapping_shared_ratio_vs_phase7h"] = float(map_row["shared_ratio_vs_phase7h"])
        proposed.append(new)

    by_object_frame: dict[tuple[str, int], dict[str, Any]] = {}
    wta_drop_count = 0
    for row in proposed:
        key = (str(row["mv_object_id"]), int(row["frame_id"]))
        current = by_object_frame.get(key)
        if current is None or _num(row.get("primitive_support_expansion_score")) > _num(
            current.get("primitive_support_expansion_score")
        ):
            if current is not None:
                wta_drop_count += 1
            by_object_frame[key] = row
        else:
            wta_drop_count += 1
    by_mask: dict[tuple[int, int], dict[str, Any]] = {}
    for row in by_object_frame.values():
        key = (int(row["frame_id"]), int(row["selected_mask_id"]))
        current = by_mask.get(key)
        if current is None or _num(row.get("primitive_support_expansion_score")) > _num(
            current.get("primitive_support_expansion_score")
        ):
            if current is not None:
                wta_drop_count += 1
            by_mask[key] = row
        else:
            wta_drop_count += 1
    accepted = list(by_mask.values())
    replacement_keys = {
        (str(row["mv_object_id"]), int(row["frame_id"]))
        for row in accepted
        if row.get("phase7j_update_mode") == "replace_existing_object_frame"
    }
    kept_base = [
        row
        for row in base_records
        if (str(row["mv_object_id"]), int(row["frame_id"])) not in replacement_keys
    ]
    out = pd.DataFrame(kept_base + accepted)
    diag = {
        "mapping_available_count": int(len(mapping)),
        "candidate_examined_count": int(candidate_examined_count),
        "proposed_fill_count": int(len(proposed)),
        "accepted_fill_count": int(len(accepted)),
        "accepted_replacement_count": int(len(replacement_keys)),
        "accepted_missing_fill_count": int(
            sum(1 for row in accepted if row.get("phase7j_update_mode") == "fill_missing_object_frame")
        ),
        "wta_drop_count": int(wta_drop_count),
        "rejected_mapping_count": int(rejected_mapping_count),
        "rejected_existing_object_frame_count": int(rejected_existing_object_frame_count),
        "rejected_occupied_mask_count": int(rejected_occupied_mask_count),
        "rejected_threshold_count": int(rejected_threshold_count),
        "rejected_broad_count": int(rejected_broad_count),
    }
    return out, accepted, diag


def _fill_same_gt_rate(accepted_rows: list[dict[str, Any]], f2_dominant_gt: dict[str, str]) -> dict[str, Any]:
    if not accepted_rows:
        return {"accepted_fill_gt_checked": 0, "accepted_fill_same_gt": 0, "accepted_fill_same_gt_rate": 0.0}
    mask_path_by_frame, _mask_source = p7d._mask_path_lookup()
    label_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    checked = 0
    same = 0
    for row in accepted_rows:
        frame = int(row["frame_id"])
        mask_id = int(row["selected_mask_id"])
        if frame not in label_cache:
            mask_path = mask_path_by_frame.get((p7d.SCENE_ID, frame))
            if mask_path is None or not mask_path.exists():
                continue
            label = p7d._read_label(mask_path)
            gt = _load_gt_2d(p7d.SCENE_ID, frame, label.shape)
            label_cache[frame] = (label, gt)
        label, gt = label_cache[frame]
        checked += 1
        same += int(_mask_gt(label, gt, mask_id) == f2_dominant_gt.get(str(row["mv_object_id"]), ""))
    return {
        "accepted_fill_gt_checked": int(checked),
        "accepted_fill_same_gt": int(same),
        "accepted_fill_same_gt_rate": float(same / max(1, checked)),
    }


def main() -> int:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    f2_rows = _same_chunk_rows(F2_ROWS, F2_VARIANT)
    p7h_variant, p7h_rows = _phase7h_best_rows()
    mapping_rows = _build_mapping(f2_rows, p7h_rows)
    f2_dominant_gt = _dominant_gt_by_object(f2_rows)

    variant_rows: list[dict[str, Any]] = []
    accepted_fill_rows_all: list[dict[str, Any]] = []
    materialized_rows_all: list[dict[str, Any]] = []
    for spec in VARIANTS:
        rows, accepted_rows, expand_diag = _expand_f2(f2_rows, p7h_rows, mapping_rows, spec)
        if len(rows):
            rows["variant_id"] = f"{VARIANT_PREFIX}_{spec['variant_id']}"
        accepted_fill_rows_all.extend(accepted_rows)
        materialized_rows_all.extend(rows.to_dict(orient="records"))
        metric = p7i._evaluate(rows, f"{VARIANT_PREFIX}_{spec['variant_id']}", f"phase7j_{spec['variant_id']}")
        same_gt_diag = _fill_same_gt_rate(accepted_rows, f2_dominant_gt)
        variant_rows.append(
            {
                "schema_version": "stream4d_v102_phase7j_variant_metric_v1",
                "phase_id": PHASE_ID,
                "variant_id": spec["variant_id"],
                "metric_scope": "scene0050_00_c0000_frames_0_155_stride5_local_diagnostic",
                "phase7h_source_variant_id": p7h_variant,
                "expand": bool(spec["expand"]),
                "min_shared_mask_count": spec["min_shared_mask_count"],
                "min_shared_ratio": spec["min_shared_ratio"],
                "min_support_fraction": spec["min_support_fraction"],
                "min_mask_coverage": spec["min_mask_coverage"],
                "drop_broad": spec["drop_broad"],
                "allow_replace_existing_object_frame": spec["allow_replace_existing_object_frame"],
                **expand_diag,
                **same_gt_diag,
                "MV_AP_window": metric.get("MV_AP_window"),
                "MV_AP50_window": metric.get("MV_AP50_window"),
                "MV_AP25_window": metric.get("MV_AP25_window"),
                "MV_AP_scene": None,
                "MV_AP50_scene": None,
                "scene_metric_computed": False,
                "scene_metric_not_computed_reason": "Phase7j evaluates only scene0050_00/c0000 chunk32 frames 0..155 stride5; full-scene/local2history scene metric is not computed.",
                "ScoreFreeMatch50_window": metric.get("ScoreFreeMatch50_window"),
                "ScoreFreeMatch25_window": metric.get("ScoreFreeMatch25_window"),
                "object_count": metric.get("object_count"),
                "frame_mask_count": metric.get("frame_mask_count"),
                "same_frame_duplicate_mask_count": metric.get("same_frame_duplicate_mask_count"),
                "object_frame_duplicate_count": metric.get("object_frame_duplicate_count"),
                "pixel_collision_count": metric.get("pixel_collision_count"),
                "uses_gt_for_prediction_any": metric.get("uses_gt_for_prediction_any"),
                "uses_future_any": metric.get("uses_future_any"),
            }
        )

    base = next(row for row in variant_rows if row["variant_id"] == "J0_f2_same_chunk_baseline")
    best = max(
        variant_rows,
        key=lambda row: (
            _num(row.get("ScoreFreeMatch50_window")),
            _num(row.get("MV_AP50_window")),
            _num(row.get("MV_AP_window")),
            _num(row.get("accepted_fill_same_gt_rate")),
        ),
    )
    delta_ap = _num(best.get("MV_AP_window")) - _num(base.get("MV_AP_window"))
    delta_ap50 = _num(best.get("MV_AP50_window")) - _num(base.get("MV_AP50_window"))
    delta_sf50 = _num(best.get("ScoreFreeMatch50_window")) - _num(base.get("ScoreFreeMatch50_window"))
    safe_fill = _num(best.get("accepted_fill_same_gt_rate")) >= 0.80 or int(_num(best.get("accepted_fill_count"))) == 0
    integrity_pass = bool(
        int(_num(best.get("same_frame_duplicate_mask_count"))) == 0
        and int(_num(best.get("object_frame_duplicate_count"))) == 0
        and int(_num(best.get("pixel_collision_count"))) == 0
        and not bool(best.get("uses_gt_for_prediction_any"))
        and not bool(best.get("uses_future_any"))
    )
    improves = bool((delta_ap50 > 1e-12 or delta_sf50 > 1e-12) and delta_ap >= -1e-12)
    decision = (
        "PASS_PHASE7J_F2_PRIMITIVE_SUPPORT_FILL_LOCAL_IMPROVES__FORMAL_TARGET_NOT_CLAIMED"
        if improves and safe_fill and integrity_pass
        else "NO_GO_PHASE7J_F2_PRIMITIVE_SUPPORT_FILL_NO_SAFE_LOCAL_GAIN"
    )
    gate_rows = [
        {
            "schema_version": "stream4d_v102_phase7j_gate_v1",
            "phase_id": PHASE_ID,
            "gate": "best_fill_improves_over_same_chunk_f2",
            "pass": bool(improves),
            "observed": f"delta_ap={delta_ap}; delta_ap50={delta_ap50}; delta_sf50={delta_sf50}",
            "required": "AP50 or ScoreFreeMatch50 improves and AP does not regress",
        },
        {
            "schema_version": "stream4d_v102_phase7j_gate_v1",
            "phase_id": PHASE_ID,
            "gate": "best_fill_same_gt_rate_ge_0p80_diagnostic",
            "pass": bool(safe_fill),
            "observed": best.get("accepted_fill_same_gt_rate"),
            "required": ">=0.80 diagnostic same-GT for accepted F2 fill rows",
        },
        {
            "schema_version": "stream4d_v102_phase7j_gate_v1",
            "phase_id": PHASE_ID,
            "gate": "best_fill_integrity",
            "pass": bool(integrity_pass),
            "observed": (
                f"dup_mask={best.get('same_frame_duplicate_mask_count')}; "
                f"object_frame_dup={best.get('object_frame_duplicate_count')}; "
                f"pixel_collision={best.get('pixel_collision_count')}; "
                f"uses_gt={best.get('uses_gt_for_prediction_any')}; uses_future={best.get('uses_future_any')}"
            ),
            "required": "no duplicates/collisions and no GT/future prediction use",
        },
        {
            "schema_version": "stream4d_v102_phase7j_gate_v1",
            "phase_id": PHASE_ID,
            "gate": "formal_v102_target_achieved",
            "pass": False,
            "observed": "same-chunk local diagnostic only; no full-dev/holdout Phase6 AP repair",
            "required": "full-dev/holdout formal AP repair gate",
        },
    ]

    _write_csv(OUT_DIR / "phase7h_to_f2_mapping_rows.csv", mapping_rows)
    _write_csv(OUT_DIR / "accepted_fill_rows.csv", accepted_fill_rows_all)
    _write_csv(OUT_DIR / "materialized_f2_fill_rows.csv", materialized_rows_all)
    _write_csv(OUT_DIR / "variant_metric_rows.csv", variant_rows)
    _write_csv(OUT_DIR / "variant_gate_rows.csv", gate_rows)
    summary = {
        "schema_version": "stream4d_v102_phase7j_f2_primitive_support_fill_summary_v1",
        "phase_id": PHASE_ID,
        "created_unix_time": time.time(),
        "runtime_sec": time.time() - t0,
        "decision": decision,
        "metric_scope": "scene0050_00_c0000_frames_0_155_stride5_local_diagnostic",
        "variant_count": len(VARIANTS),
        "f2_variant_id": F2_VARIANT,
        "phase7h_source_variant_id": p7h_variant,
        "f2_base_MV_AP_window": base.get("MV_AP_window"),
        "f2_base_MV_AP50_window": base.get("MV_AP50_window"),
        "f2_base_MV_AP_scene": None,
        "f2_base_MV_AP50_scene": None,
        "f2_base_ScoreFreeMatch50_window": base.get("ScoreFreeMatch50_window"),
        "best_variant_id": best.get("variant_id"),
        "best_MV_AP_window": best.get("MV_AP_window"),
        "best_MV_AP50_window": best.get("MV_AP50_window"),
        "best_MV_AP_scene": None,
        "best_MV_AP50_scene": None,
        "best_ScoreFreeMatch50_window": best.get("ScoreFreeMatch50_window"),
        "scene_metric_computed": False,
        "scene_metric_not_computed_reason": "Phase7j evaluates only scene0050_00/c0000 frames; MV_AP_scene/MV_AP50_scene are not computed.",
        "best_delta_MV_AP_window_vs_f2": delta_ap,
        "best_delta_MV_AP50_window_vs_f2": delta_ap50,
        "best_delta_ScoreFreeMatch50_window_vs_f2": delta_sf50,
        "best_accepted_fill_count": best.get("accepted_fill_count"),
        "best_accepted_fill_same_gt_rate": best.get("accepted_fill_same_gt_rate"),
        "local_diagnostic_improves": improves,
        "best_fill_diagnostic_safe": safe_fill,
        "integrity_pass": integrity_pass,
        "formal_v102_target_achieved": False,
        "formal_target_blocker": "Phase7j is same-chunk local F2-fill diagnostic only; Phase6 full AP repair remains blocked by Phase1b repair-space.",
        "uses_gt_for_prediction": False,
        "uses_gt_for_eval": True,
        "uses_future": False,
        "truthfulness_note": (
            "Phase7j maps Phase7h objects to fixed v100 F2 objects by shared frame/mask observations, then uses "
            "only Phase7h primitive-support fill rows to fill missing F2 object frames. GT is used only after rows "
            "are fixed for SparseSceneIoU/AP and same-GT diagnostics."
        ),
        "plan_doc": _rel(PLAN_DOC),
        "inputs": {
            "f2_rows": _rel(F2_ROWS),
            "phase7h_summary": _rel(PHASE7H_SUMMARY),
            "phase7h_rows": _rel(PHASE7H_ROWS),
        },
        "outputs": {
            "summary": _rel(OUT_DIR / "summary.json"),
            "phase7h_to_f2_mapping_rows": _rel(OUT_DIR / "phase7h_to_f2_mapping_rows.csv"),
            "accepted_fill_rows": _rel(OUT_DIR / "accepted_fill_rows.csv"),
            "materialized_f2_fill_rows": _rel(OUT_DIR / "materialized_f2_fill_rows.csv"),
            "variant_metric_rows": _rel(OUT_DIR / "variant_metric_rows.csv"),
            "variant_gate_rows": _rel(OUT_DIR / "variant_gate_rows.csv"),
        },
    }
    _write_json(OUT_DIR / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
