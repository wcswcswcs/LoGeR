#!/usr/bin/env python3
"""GT-free semantic missing-frame expansion for v102 Phase7 components."""

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

from tools.run_v65_scene_multiview_ap import SparseSceneIoU, _load_gt_2d, _summarize_iou  # noqa: E402
from tools import build_v102_phase7d_phase7c_materialized_ap_diagnostic as p7d  # noqa: E402


AUDIT_ROOT = STREAM3D_ROOT / "outputs" / "audit"
OUT_DIR = AUDIT_ROOT / "v102_phase7g_semantic_missing_frame_expansion"
PHASE7E_SUMMARY = AUDIT_ROOT / "v102_phase7e_gtfree_score_calibration_diagnostic" / "summary.json"
NODE_ROWS = AUDIT_ROOT / "v102_phase7d_phase7c_materialized_ap_diagnostic" / "mv_object_frame_mask_rows.parquet"
COMPONENT_ROWS = AUDIT_ROOT / "v102_phase7d_phase7c_materialized_ap_diagnostic" / "materialized_component_rows.csv"
FEATURE_STORE = AUDIT_ROOT / "v91_radio_mask_features_npz_scene0050" / "mask_features.npz"
FEATURE_ROWS = AUDIT_ROOT / "v91_radio_mask_features_npz_scene0050" / "mask_feature_rows.csv"

PHASE_ID = "v102_phase7g_semantic_missing_frame_expansion"
VARIANT_PREFIX = "P2_v102_phase7g"

VARIANTS = [
    {
        "variant_id": "G0_no_expand_s8_score",
        "cosine_min": 2.0,
        "entropy_max": 0.0,
        "require_prototype_match": True,
        "drop_broad": True,
    },
    {
        "variant_id": "G1_proto_cos098_entropy045",
        "cosine_min": 0.98,
        "entropy_max": 0.45,
        "require_prototype_match": True,
        "drop_broad": False,
    },
    {
        "variant_id": "G2_proto_cos096_entropy045",
        "cosine_min": 0.96,
        "entropy_max": 0.45,
        "require_prototype_match": True,
        "drop_broad": False,
    },
    {
        "variant_id": "G3_proto_cos094_entropy040",
        "cosine_min": 0.94,
        "entropy_max": 0.40,
        "require_prototype_match": True,
        "drop_broad": False,
    },
    {
        "variant_id": "G4_proto_cos092_entropy035",
        "cosine_min": 0.92,
        "entropy_max": 0.35,
        "require_prototype_match": True,
        "drop_broad": True,
    },
    {
        "variant_id": "G5_no_proto_cos096_entropy035",
        "cosine_min": 0.96,
        "entropy_max": 0.35,
        "require_prototype_match": False,
        "drop_broad": True,
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


def _norm(values: dict[str, float]) -> dict[str, float]:
    finite = [float(v) for v in values.values() if math.isfinite(float(v))]
    if not finite:
        return {key: 0.0 for key in values}
    lo = min(finite)
    hi = max(finite)
    if hi - lo <= 1e-12:
        return {key: 0.5 for key in values}
    return {key: (float(value) - lo) / (hi - lo) for key, value in values.items()}


def _load_features() -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    store = np.load(FEATURE_STORE)
    feats = np.asarray(store["features"], dtype=np.float32)
    feats = feats / np.maximum(np.linalg.norm(feats, axis=1, keepdims=True), 1e-12)
    ids = [str(x) for x in store["mask_observation_id"]]
    return {node_id: feats[i] for i, node_id in enumerate(ids)}, pd.read_csv(FEATURE_ROWS)


def _object_meta(node_rows: pd.DataFrame, component_rows: pd.DataFrame, fmap: dict[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    diag = component_rows.set_index("mv_object_id").to_dict(orient="index")
    out: dict[str, dict[str, Any]] = {}
    for oid, rows in node_rows.groupby("mv_object_id"):
        oid = str(oid)
        node_ids = [str(v) for v in rows["source_mask_observation_id"].tolist()]
        feats = [fmap[node_id] for node_id in node_ids if node_id in fmap]
        centroid = np.mean(np.stack(feats), axis=0) if feats else np.zeros((768,), dtype=np.float32)
        centroid = centroid / max(float(np.linalg.norm(centroid)), 1e-12)
        prototypes = [str(v) for v in rows["semantic_prototype_margin"].tolist()]
        feature_prototypes = []
        # Feature rows store the actual prototype id; node rows keep margin only, so use FEATURE_ROWS later for candidates.
        frame_ids = sorted({int(v) for v in rows["frame_id"].tolist()})
        out[oid] = {
            "mv_object_id": oid,
            "centroid": centroid.astype(np.float32, copy=False),
            "frames": frame_ids,
            "diagnostic_gt_dominant": str(diag.get(oid, {}).get("diagnostic_gt_dominant", "")),
            "node_count": len(node_ids),
            "area_sum": float(pd.to_numeric(rows["used_pixel_count"], errors="coerce").fillna(0.0).sum()),
            "semantic_entropy_mean": float(pd.to_numeric(rows["semantic_entropy"], errors="coerce").fillna(0.0).mean()),
            "semantic_margin_mean": float(pd.to_numeric(rows["semantic_prototype_margin"], errors="coerce").fillna(0.0).mean()),
            "dominant_prototype": "",
        }
    return out


def _fill_dominant_prototypes(meta: dict[str, dict[str, Any]], node_rows: pd.DataFrame, feature_rows: pd.DataFrame) -> None:
    proto_by_node = feature_rows.set_index("mask_observation_id")["semantic_prototype_id"].astype(str).to_dict()
    for oid, rows in node_rows.groupby("mv_object_id"):
        prototypes = [proto_by_node.get(str(v), "") for v in rows["source_mask_observation_id"].tolist()]
        counts = Counter([p for p in prototypes if p])
        meta[str(oid)]["dominant_prototype"] = counts.most_common(1)[0][0] if counts else ""


def _candidate_rows_by_frame(feature_rows: pd.DataFrame, fmap: dict[str, np.ndarray]) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = defaultdict(list)
    frames = set(p7d._frame_universe())
    for row in feature_rows.to_dict(orient="records"):
        scene = str(row.get("scene_id", ""))
        frame = int(_num(row.get("frame_id")))
        node_id = str(row.get("mask_observation_id", ""))
        if scene != p7d.SCENE_ID or frame not in frames or node_id not in fmap:
            continue
        out[frame].append(row)
    return out


def _expand_rows(
    base_rows: pd.DataFrame,
    meta: dict[str, dict[str, Any]],
    candidates_by_frame: dict[int, list[dict[str, Any]]],
    fmap: dict[str, np.ndarray],
    spec: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = [dict(row) | {"phase7g_row_source": "base_phase7d"} for row in base_rows.to_dict(orient="records")]
    existing_object_frame = {(str(row["mv_object_id"]), int(row["frame_id"])) for row in rows}
    occupied_mask = {(int(row["frame_id"]), int(row["selected_mask_id"])) for row in rows}
    proposed: list[dict[str, Any]] = []
    candidate_examined = 0
    for oid, obj in meta.items():
        for frame in p7d._frame_universe():
            if (oid, int(frame)) in existing_object_frame:
                continue
            best: dict[str, Any] | None = None
            best_score = -1e9
            for cand in candidates_by_frame.get(int(frame), []):
                candidate_examined += 1
                mask_id = int(_num(cand.get("mask_id")))
                if (int(frame), mask_id) in occupied_mask:
                    continue
                if _num(cand.get("semantic_entropy"), 999.0) > float(spec["entropy_max"]):
                    continue
                if spec["drop_broad"] and _bool(cand.get("broad_background_risk")):
                    continue
                if spec["require_prototype_match"] and obj["dominant_prototype"] != str(cand.get("semantic_prototype_id", "")):
                    continue
                node_id = str(cand.get("mask_observation_id", ""))
                cos = float(np.dot(obj["centroid"], fmap[node_id]))
                if cos < float(spec["cosine_min"]):
                    continue
                score = cos + 0.01 * _num(cand.get("semantic_prototype_margin")) - 0.001 * _num(cand.get("semantic_entropy"))
                if score > best_score:
                    best_score = score
                    best = cand | {"expansion_cosine": cos, "expansion_score": score}
            if best is None:
                continue
            proposed.append(
                {
                    "schema_version": "stream4d_v102_phase7g_mv_object_frame_mask_row_v1",
                    "phase_id": PHASE_ID,
                    "dataset_split": "dev",
                    "variant_id": f"{VARIANT_PREFIX}_{spec['variant_id']}",
                    "source_phase7d_variant_id": str(base_rows.iloc[0].get("variant_id", "")),
                    "mv_object_id": oid,
                    "object_id": oid,
                    "source_component_id": str(base_rows[base_rows["mv_object_id"].astype(str) == oid].iloc[0]["source_component_id"]),
                    "scene_id": p7d.SCENE_ID,
                    "chunk_id": p7d.CHUNK_ID,
                    "window_id": p7d.CHUNK_ID,
                    "frame_id": int(frame),
                    "selected_mask_id": int(_num(best.get("mask_id"))),
                    "mask_id_or_generated_id": int(_num(best.get("mask_id"))),
                    "source_mask_observation_id": str(best.get("mask_observation_id")),
                    "readout_mode": "phase7g_semantic_centroid_missing_frame_fill",
                    "score": 0.0,
                    "object_score": 0.0,
                    "score_scope": "current_chunk32_diagnostic",
                    "score_policy": "phase7g_s8_node_area_semantic_after_expansion",
                    "component_node_count": int(meta[oid]["node_count"]),
                    "component_frame_count": len(meta[oid]["frames"]),
                    "method_chunk_size": p7d.CHUNK_SIZE,
                    "method_chunk_overlap": 0,
                    "frame_stride": p7d.FRAME_STRIDE,
                    "object_id_policy": "phase7c_component_identity_with_phase7g_missing_frame_fill",
                    "object_birth_scope": "phase7g_semantic_missing_frame_expansion",
                    "semantic_entropy": best.get("semantic_entropy", ""),
                    "semantic_prototype_margin": best.get("semantic_prototype_margin", ""),
                    "broad_background_risk": best.get("broad_background_risk", ""),
                    "used_pixel_count": best.get("used_pixel_count", ""),
                    "expansion_cosine": best["expansion_cosine"],
                    "expansion_score": best["expansion_score"],
                    "phase7g_row_source": "semantic_missing_frame_fill",
                    "uses_gt_for_prediction": False,
                    "uses_gt_for_eval": False,
                    "uses_future": False,
                }
            )
    by_mask: dict[tuple[int, int], dict[str, Any]] = {}
    wta_drop_count = 0
    for row in proposed:
        key = (int(row["frame_id"]), int(row["selected_mask_id"]))
        current = by_mask.get(key)
        if current is None or float(row["expansion_score"]) > float(current["expansion_score"]):
            if current is not None:
                wta_drop_count += 1
            by_mask[key] = row
        else:
            wta_drop_count += 1
    accepted = list(by_mask.values())
    rows.extend(accepted)
    diag = {
        "candidate_examined_count": int(candidate_examined),
        "proposed_expansion_count": int(len(proposed)),
        "accepted_expansion_count": int(len(accepted)),
        "wta_drop_count": int(wta_drop_count),
    }
    return rows, diag


def _scores(rows: list[dict[str, Any]]) -> dict[str, float]:
    by_object: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_object[str(row["mv_object_id"])].append(row)
    raw_node = {oid: float(len(vals)) for oid, vals in by_object.items()}
    raw_area = {
        oid: math.log1p(sum(_num(row.get("used_pixel_count")) for row in vals)) for oid, vals in by_object.items()
    }
    raw_entropy = {
        oid: float(np.mean([_num(row.get("semantic_entropy")) for row in vals])) for oid, vals in by_object.items()
    }
    raw_margin = {
        oid: float(np.mean([_num(row.get("semantic_prototype_margin")) for row in vals])) for oid, vals in by_object.items()
    }
    node = _norm(raw_node)
    area = _norm(raw_area)
    entropy_good = {oid: 1.0 - val for oid, val in _norm(raw_entropy).items()}
    margin = _norm(raw_margin)
    sem = {oid: 0.60 * margin[oid] + 0.40 * entropy_good[oid] for oid in by_object}
    return {oid: 0.45 * node[oid] + 0.35 * area[oid] + 0.20 * sem[oid] for oid in by_object}


def _mask_diagnostic_gt(label: np.ndarray, gt: np.ndarray, mask_id: int) -> str:
    mask = label == int(mask_id)
    vals = gt[mask]
    vals = vals[vals > 0]
    if vals.size == 0:
        return ""
    ids, counts = np.unique(vals, return_counts=True)
    return str(int(ids[int(np.argmax(counts))]))


def _evaluate(rows: list[dict[str, Any]], scores: dict[str, float], meta: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    object_ids = sorted({str(row["mv_object_id"]) for row in rows})
    object_index = {oid: idx + 1 for idx, oid in enumerate(object_ids)}
    rows_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_frame[int(row["frame_id"])].append(row)
    mask_path_by_frame, mask_source = p7d._mask_path_lookup()
    acc = SparseSceneIoU()
    pixel_collision_count = 0
    missing_mask_frame_count = 0
    selected_mask_missing_count = 0
    accepted_expansion_gt_checked = 0
    accepted_expansion_same_gt = 0
    for frame in p7d._frame_universe():
        mask_path = mask_path_by_frame.get((p7d.SCENE_ID, int(frame)))
        if mask_path is None or not mask_path.exists():
            missing_mask_frame_count += 1
            continue
        label = p7d._read_label(mask_path)
        gt = _load_gt_2d(p7d.SCENE_ID, int(frame), label.shape)
        pred = np.zeros(label.shape, dtype=np.int64)
        for row in sorted(rows_by_frame.get(int(frame), []), key=lambda r: (-scores[str(r["mv_object_id"])], str(r["mv_object_id"]))):
            mask = label == int(row["selected_mask_id"])
            if int(np.count_nonzero(mask)) <= 0:
                selected_mask_missing_count += 1
                continue
            if row.get("phase7g_row_source") == "semantic_missing_frame_fill":
                accepted_expansion_gt_checked += 1
                diag_gt = _mask_diagnostic_gt(label, gt, int(row["selected_mask_id"]))
                accepted_expansion_same_gt += int(diag_gt != "" and diag_gt == meta[str(row["mv_object_id"])]["diagnostic_gt_dominant"])
            occupied = (pred > 0) & mask
            pixel_collision_count += int(np.count_nonzero(occupied))
            pred[(pred == 0) & mask] = object_index[str(row["mv_object_id"])]
        acc.add(pred, gt)
    input_scores = np.ones((len(object_ids),), dtype=np.float32)
    for oid, idx in object_index.items():
        input_scores[idx - 1] = float(scores.get(oid, 0.0))
    summary, _iou, _pred_ids, _gt_ids = _summarize_iou(
        accumulator=acc,
        min_pred_pixels=64,
        min_gt_pixels=64,
        score_mode="input",
        input_scores=input_scores,
    )
    diag = {
        "object_count": len(object_ids),
        "frame_mask_count": len(rows),
        "eval_frame_count": int(acc.frame_count),
        "missing_mask_frame_count": int(missing_mask_frame_count),
        "selected_mask_missing_count": int(selected_mask_missing_count),
        "pixel_collision_count": int(pixel_collision_count),
        "accepted_expansion_gt_checked": int(accepted_expansion_gt_checked),
        "accepted_expansion_same_gt": int(accepted_expansion_same_gt),
        "accepted_expansion_same_gt_rate": float(accepted_expansion_same_gt / max(1, accepted_expansion_gt_checked)),
        "mask_source": mask_source,
    }
    return summary, diag


def main() -> int:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    phase7e = _read_json(PHASE7E_SUMMARY)
    base_rows = pd.read_parquet(NODE_ROWS)
    component_rows = pd.read_csv(COMPONENT_ROWS)
    fmap, feature_rows = _load_features()
    meta = _object_meta(base_rows, component_rows, fmap)
    _fill_dominant_prototypes(meta, base_rows, feature_rows)
    candidates_by_frame = _candidate_rows_by_frame(feature_rows, fmap)

    variant_rows: list[dict[str, Any]] = []
    expansion_rows_all: list[dict[str, Any]] = []
    materialized_rows_all: list[dict[str, Any]] = []
    for spec in VARIANTS:
        rows, expand_diag = _expand_rows(base_rows, meta, candidates_by_frame, fmap, spec)
        scores = _scores(rows)
        for row in rows:
            row = dict(row)
            row["variant_id"] = f"{VARIANT_PREFIX}_{spec['variant_id']}"
            row["score"] = scores[str(row["mv_object_id"])]
            row["object_score"] = scores[str(row["mv_object_id"])]
            materialized_rows_all.append(row)
            if row.get("phase7g_row_source") == "semantic_missing_frame_fill":
                expansion_rows_all.append(row)
        summary, eval_diag = _evaluate(rows, scores, meta)
        variant_rows.append(
            {
                "schema_version": "stream4d_v102_phase7g_variant_metric_v1",
                "phase_id": PHASE_ID,
                "variant_id": spec["variant_id"],
                "metric_scope": "chunk32_scene0050_local_diagnostic_not_full_dev",
                "cosine_min": spec["cosine_min"],
                "entropy_max": spec["entropy_max"],
                "require_prototype_match": spec["require_prototype_match"],
                "drop_broad": spec["drop_broad"],
                **expand_diag,
                **eval_diag,
                "MV_AP_window": summary.get("ap"),
                "MV_AP50_window": summary.get("ap50"),
                "MV_AP25_window": summary.get("ap25"),
                "MV_AP_scene": summary.get("ap"),
                "MV_AP50_scene": summary.get("ap50"),
                "ScoreFreeMatch50_window": (summary.get("score_free_match_at_050") or {}).get("recall"),
                "ScoreFreeMatch25_window": (summary.get("score_free_match_at_025") or {}).get("recall"),
                "uses_gt_for_prediction": False,
                "uses_gt_for_eval": True,
                "uses_future": False,
            }
        )

    phase7e_ap50 = _num(phase7e.get("best_MV_AP50_window"))
    phase7e_sf50 = _num(phase7e.get("best_ScoreFreeMatch50_window"))
    best = max(
        variant_rows,
        key=lambda row: (
            _num(row.get("ScoreFreeMatch50_window")),
            _num(row.get("MV_AP50_window")),
            _num(row.get("MV_AP_window")),
            _num(row.get("accepted_expansion_same_gt_rate")),
        ),
    )
    best_delta_ap50 = _num(best.get("MV_AP50_window")) - phase7e_ap50
    best_delta_sf50 = _num(best.get("ScoreFreeMatch50_window")) - phase7e_sf50
    improves = bool(best_delta_ap50 > 1e-12 or best_delta_sf50 > 1e-12)
    safe_fill = _num(best.get("accepted_expansion_same_gt_rate")) >= 0.80 or int(_num(best.get("accepted_expansion_count"))) == 0
    decision = (
        "PASS_PHASE7G_SEMANTIC_MISSING_FRAME_EXPANSION_LOCAL_IMPROVES__FORMAL_TARGET_NOT_CLAIMED"
        if improves and safe_fill
        else "NO_GO_PHASE7G_SEMANTIC_MISSING_FRAME_EXPANSION_NO_SAFE_LOCAL_GAIN"
    )
    gate_rows = [
        {
            "schema_version": "stream4d_v102_phase7g_gate_v1",
            "phase_id": PHASE_ID,
            "gate": "best_local_diagnostic_improves_over_phase7e",
            "pass": bool(improves),
            "observed": f"delta_ap50={best_delta_ap50}; delta_sf50={best_delta_sf50}",
            "required": ">0 local AP50 or ScoreFreeMatch50 delta",
        },
        {
            "schema_version": "stream4d_v102_phase7g_gate_v1",
            "phase_id": PHASE_ID,
            "gate": "best_expansion_same_gt_rate_ge_0p80_diagnostic",
            "pass": bool(safe_fill),
            "observed": best.get("accepted_expansion_same_gt_rate"),
            "required": ">=0.80 diagnostic same-GT for accepted fill rows",
        },
        {
            "schema_version": "stream4d_v102_phase7g_gate_v1",
            "phase_id": PHASE_ID,
            "gate": "formal_v102_target_achieved",
            "pass": False,
            "observed": "not claimed from local missing-frame expansion diagnostic",
            "required": "full-dev/holdout formal AP repair gate",
        },
    ]
    _write_csv(OUT_DIR / "expansion_candidate_rows.csv", expansion_rows_all)
    _write_csv(OUT_DIR / "materialized_expanded_rows.csv", materialized_rows_all)
    _write_csv(OUT_DIR / "expansion_variant_metric_rows.csv", variant_rows)
    _write_csv(OUT_DIR / "expansion_gate_rows.csv", gate_rows)
    summary = {
        "schema_version": "stream4d_v102_phase7g_semantic_missing_frame_expansion_summary_v1",
        "phase_id": PHASE_ID,
        "created_unix_time": time.time(),
        "runtime_sec": time.time() - t0,
        "decision": decision,
        "metric_scope": "chunk32_scene0050_local_diagnostic_not_full_dev",
        "variant_count": len(VARIANTS),
        "phase7e_best_MV_AP50_window": phase7e_ap50,
        "phase7e_best_ScoreFreeMatch50_window": phase7e_sf50,
        "best_variant_id": best["variant_id"],
        "best_MV_AP_window": best.get("MV_AP_window"),
        "best_MV_AP50_window": best.get("MV_AP50_window"),
        "best_MV_AP25_window": best.get("MV_AP25_window"),
        "best_ScoreFreeMatch50_window": best.get("ScoreFreeMatch50_window"),
        "best_ScoreFreeMatch25_window": best.get("ScoreFreeMatch25_window"),
        "best_delta_MV_AP50_window_vs_phase7e": best_delta_ap50,
        "best_delta_ScoreFreeMatch50_window_vs_phase7e": best_delta_sf50,
        "best_accepted_expansion_count": best.get("accepted_expansion_count"),
        "best_wta_drop_count": best.get("wta_drop_count"),
        "best_frame_mask_count": best.get("frame_mask_count"),
        "best_accepted_expansion_same_gt_rate": best.get("accepted_expansion_same_gt_rate"),
        "best_pixel_collision_count": best.get("pixel_collision_count"),
        "local_diagnostic_improves": improves,
        "best_expansion_diagnostic_safe": safe_fill,
        "formal_v102_target_achieved": False,
        "formal_target_blocker": "Phase7g is local chunk32 missing-frame support expansion; Phase6 full repair remains blocked by Phase1b.",
        "uses_gt_for_prediction": False,
        "uses_gt_for_eval": True,
        "uses_future": False,
        "truthfulness_note": (
            "Expansion candidates use RADIO feature similarity/prototype/entropy only. Diagnostic same-GT rate and AP use GT only after prediction rows are fixed."
        ),
        "outputs": {
            "summary": _rel(OUT_DIR / "summary.json"),
            "expansion_candidate_rows": _rel(OUT_DIR / "expansion_candidate_rows.csv"),
            "materialized_expanded_rows": _rel(OUT_DIR / "materialized_expanded_rows.csv"),
            "expansion_variant_metric_rows": _rel(OUT_DIR / "expansion_variant_metric_rows.csv"),
            "expansion_gate_rows": _rel(OUT_DIR / "expansion_gate_rows.csv"),
        },
    }
    _write_json(OUT_DIR / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
