#!/usr/bin/env python3
"""GT-free component stitching diagnostic for v102 Phase7d/7e."""

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
OUT_DIR = AUDIT_ROOT / "v102_phase7f_gtfree_component_stitch_diagnostic"
PHASE7D_DIR = AUDIT_ROOT / "v102_phase7d_phase7c_materialized_ap_diagnostic"
PHASE7E_SUMMARY = AUDIT_ROOT / "v102_phase7e_gtfree_score_calibration_diagnostic" / "summary.json"
NODE_ROWS = PHASE7D_DIR / "mv_object_frame_mask_rows.parquet"
MATERIALIZED_COMPONENT_ROWS = PHASE7D_DIR / "materialized_component_rows.csv"
FEATURE_STORE = AUDIT_ROOT / "v91_radio_mask_features_npz_scene0050" / "mask_features.npz"
FEATURE_ROWS = AUDIT_ROOT / "v91_radio_mask_features_npz_scene0050" / "mask_feature_rows.csv"

PHASE_ID = "v102_phase7f_gtfree_component_stitch_diagnostic"
VARIANT_PREFIX = "P2_v102_phase7f"

VARIANTS = [
    {
        "variant_id": "F0_no_stitch_s8_score",
        "centroid_cosine_min": 2.0,
        "max_frame_gap": None,
        "require_dominant_prototype_match": True,
    },
    {
        "variant_id": "F1_proto_centroid098_gap20",
        "centroid_cosine_min": 0.98,
        "max_frame_gap": 20,
        "require_dominant_prototype_match": True,
    },
    {
        "variant_id": "F2_proto_centroid096_gap20",
        "centroid_cosine_min": 0.96,
        "max_frame_gap": 20,
        "require_dominant_prototype_match": True,
    },
    {
        "variant_id": "F3_proto_centroid094_gap20",
        "centroid_cosine_min": 0.94,
        "max_frame_gap": 20,
        "require_dominant_prototype_match": True,
    },
    {
        "variant_id": "F4_proto_centroid096_anygap",
        "centroid_cosine_min": 0.96,
        "max_frame_gap": None,
        "require_dominant_prototype_match": True,
    },
    {
        "variant_id": "F5_centroid098_gap20_no_proto",
        "centroid_cosine_min": 0.98,
        "max_frame_gap": 20,
        "require_dominant_prototype_match": False,
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


def _norm(values: dict[str, float]) -> dict[str, float]:
    finite = [float(v) for v in values.values() if math.isfinite(float(v))]
    if not finite:
        return {key: 0.0 for key in values}
    lo = min(finite)
    hi = max(finite)
    if hi - lo <= 1e-12:
        return {key: 0.5 for key in values}
    return {key: (float(value) - lo) / (hi - lo) for key, value in values.items()}


def _load_feature_map() -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    store = np.load(FEATURE_STORE)
    feats = np.asarray(store["features"], dtype=np.float32)
    feats = feats / np.maximum(np.linalg.norm(feats, axis=1, keepdims=True), 1e-12)
    ids = [str(x) for x in store["mask_observation_id"]]
    fmap = {node_id: feats[i] for i, node_id in enumerate(ids)}
    frows = pd.read_csv(FEATURE_ROWS).set_index("mask_observation_id").to_dict(orient="index")
    return fmap, {str(k): v for k, v in frows.items()}


class Forest:
    def __init__(self, nodes: list[str]) -> None:
        self.parent = {node: node for node in nodes}
        self.size = {node: 1 for node in nodes}

    def find(self, x: str) -> str:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]

    def groups(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = defaultdict(list)
        for node in self.parent:
            out[self.find(node)].append(node)
        return {root: sorted(vals) for root, vals in out.items()}


def _component_meta(node_rows: pd.DataFrame, component_rows: pd.DataFrame) -> dict[str, dict[str, Any]]:
    fmap, feature_meta = _load_feature_map()
    diagnostic = component_rows.set_index("mv_object_id").to_dict(orient="index")
    out: dict[str, dict[str, Any]] = {}
    for oid, rows in node_rows.groupby("mv_object_id"):
        oid = str(oid)
        node_ids = [str(v) for v in rows["source_mask_observation_id"].tolist()]
        feats = [fmap[node_id] for node_id in node_ids if node_id in fmap]
        centroid = np.mean(np.stack(feats), axis=0) if feats else np.zeros((768,), dtype=np.float32)
        centroid = centroid / max(float(np.linalg.norm(centroid)), 1e-12)
        prototypes = [
            str(feature_meta.get(node_id, {}).get("semantic_prototype_id", ""))
            for node_id in node_ids
            if feature_meta.get(node_id, {}).get("semantic_prototype_id", "")
        ]
        proto_counts = Counter(prototypes)
        dominant_proto = proto_counts.most_common(1)[0][0] if proto_counts else ""
        frames = sorted({int(v) for v in rows["frame_id"].tolist()})
        used = pd.to_numeric(rows["used_pixel_count"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        entropy = pd.to_numeric(rows["semantic_entropy"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        margin = pd.to_numeric(rows["semantic_prototype_margin"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        broad = rows["broad_background_risk"].astype(str).str.lower().isin(["1", "true", "yes", "y"]).to_numpy()
        diag = diagnostic.get(oid, {})
        out[oid] = {
            "mv_object_id": oid,
            "node_ids": node_ids,
            "frames": frames,
            "centroid": centroid.astype(np.float32, copy=False),
            "dominant_prototype": dominant_proto,
            "prototype_count": len(proto_counts),
            "node_count": len(node_ids),
            "area_sum": float(np.sum(used)) if used.size else 0.0,
            "area_mean": float(np.mean(used)) if used.size else 0.0,
            "semantic_entropy_mean": float(np.mean(entropy)) if entropy.size else 0.0,
            "semantic_margin_mean": float(np.mean(margin)) if margin.size else 0.0,
            "broad_fraction": float(np.mean(broad)) if broad.size else 0.0,
            "diagnostic_gt_dominant": str(diag.get("diagnostic_gt_dominant", "")),
        }
    return out


def _frame_gap(a: list[int], b: list[int]) -> int:
    return min(abs(int(x) - int(y)) for x in a for y in b)


def _build_groups(meta: dict[str, dict[str, Any]], spec: dict[str, Any]) -> tuple[dict[str, list[str]], list[dict[str, Any]]]:
    ids = sorted(meta)
    forest = Forest(ids)
    edge_rows: list[dict[str, Any]] = []
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            ma, mb = meta[a], meta[b]
            same_frame = bool(set(ma["frames"]) & set(mb["frames"]))
            if same_frame:
                continue
            gap = _frame_gap(ma["frames"], mb["frames"])
            max_gap = spec.get("max_frame_gap")
            if max_gap is not None and gap > int(max_gap):
                continue
            proto_match = ma["dominant_prototype"] and ma["dominant_prototype"] == mb["dominant_prototype"]
            if spec["require_dominant_prototype_match"] and not proto_match:
                continue
            centroid_cos = float(np.dot(ma["centroid"], mb["centroid"]))
            if centroid_cos < float(spec["centroid_cosine_min"]):
                continue
            forest.union(a, b)
            edge_rows.append(
                {
                    "schema_version": "stream4d_v102_phase7f_stitch_edge_v1",
                    "phase_id": PHASE_ID,
                    "variant_id": spec["variant_id"],
                    "component_a": a,
                    "component_b": b,
                    "frame_gap": int(gap),
                    "centroid_cosine": centroid_cos,
                    "dominant_prototype_match": bool(proto_match),
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )
    return forest.groups(), edge_rows


def _group_scores(groups: dict[str, list[str]], meta: dict[str, dict[str, Any]]) -> dict[str, float]:
    raw_frame = {gid: float(len({f for oid in members for f in meta[oid]["frames"]})) for gid, members in groups.items()}
    raw_node = {gid: float(sum(meta[oid]["node_count"] for oid in members)) for gid, members in groups.items()}
    raw_area = {gid: math.log1p(sum(meta[oid]["area_sum"] for oid in members)) for gid, members in groups.items()}
    raw_entropy = {
        gid: float(np.mean([meta[oid]["semantic_entropy_mean"] for oid in members])) for gid, members in groups.items()
    }
    raw_margin = {
        gid: float(np.mean([meta[oid]["semantic_margin_mean"] for oid in members])) for gid, members in groups.items()
    }
    frame = _norm(raw_frame)
    node = _norm(raw_node)
    area = _norm(raw_area)
    entropy_good = {gid: 1.0 - val for gid, val in _norm(raw_entropy).items()}
    margin = _norm(raw_margin)
    sem = {gid: 0.60 * margin[gid] + 0.40 * entropy_good[gid] for gid in groups}
    return {gid: 0.45 * node[gid] + 0.35 * area[gid] + 0.20 * sem[gid] for gid in groups}


def _evaluate_variant(
    node_rows: pd.DataFrame,
    groups: dict[str, list[str]],
    scores: dict[str, float],
    variant_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    component_to_group: dict[str, str] = {}
    group_ids = []
    for group_idx, (_root, members) in enumerate(sorted(groups.items()), start=1):
        group_id = f"{VARIANT_PREFIX}:{variant_id}:{p7d.SCENE_ID}:{p7d.CHUNK_ID}:stitched_{group_idx:04d}"
        group_ids.append(group_id)
        for oid in members:
            component_to_group[oid] = group_id
    group_index = {gid: idx + 1 for idx, gid in enumerate(group_ids)}
    rows_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in node_rows.to_dict(orient="records"):
        row = dict(row)
        row["stitched_group_id"] = component_to_group[str(row["mv_object_id"])]
        rows_by_frame[int(row["frame_id"])].append(row)
    mask_path_by_frame, mask_source = p7d._mask_path_lookup()
    acc = SparseSceneIoU()
    missing_mask_frame_count = 0
    selected_mask_missing_count = 0
    pixel_collision_count = 0
    same_frame_duplicate_count = 0
    for frame_id in p7d._frame_universe():
        mask_path = mask_path_by_frame.get((p7d.SCENE_ID, int(frame_id)))
        if mask_path is None or not mask_path.exists():
            missing_mask_frame_count += 1
            continue
        label = p7d._read_label(mask_path)
        pred = np.zeros(label.shape, dtype=np.int64)
        seen_groups = Counter(str(row["stitched_group_id"]) for row in rows_by_frame.get(int(frame_id), []))
        same_frame_duplicate_count += sum(max(0, count - 1) for count in seen_groups.values())
        for row in sorted(rows_by_frame.get(int(frame_id), []), key=lambda r: (-scores[str(r["stitched_group_id"])], str(r["stitched_group_id"]))):
            mask = label == int(row["selected_mask_id"])
            if int(np.count_nonzero(mask)) <= 0:
                selected_mask_missing_count += 1
                continue
            pred_id = group_index[str(row["stitched_group_id"])]
            occupied = (pred > 0) & mask
            pixel_collision_count += int(np.count_nonzero(occupied))
            pred[(pred == 0) & mask] = pred_id
        gt = _load_gt_2d(p7d.SCENE_ID, int(frame_id), label.shape)
        acc.add(pred, gt)
    input_scores = np.ones((len(group_ids),), dtype=np.float32)
    for gid, idx in group_index.items():
        input_scores[idx - 1] = float(scores.get(gid, 0.0))
    summary, _iou, _pred_ids, _gt_ids = _summarize_iou(
        accumulator=acc,
        min_pred_pixels=64,
        min_gt_pixels=64,
        score_mode="input",
        input_scores=input_scores,
    )
    diag = {
        "group_count": len(group_ids),
        "eval_frame_count": int(acc.frame_count),
        "missing_mask_frame_count": int(missing_mask_frame_count),
        "selected_mask_missing_count": int(selected_mask_missing_count),
        "pixel_collision_count": int(pixel_collision_count),
        "same_frame_duplicate_group_count": int(same_frame_duplicate_count),
        "mask_source": mask_source,
    }
    return summary, diag


def main() -> int:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    phase7e = _read_json(PHASE7E_SUMMARY)
    node_rows = pd.read_parquet(NODE_ROWS)
    component_rows = pd.read_csv(MATERIALIZED_COMPONENT_ROWS)
    meta = _component_meta(node_rows, component_rows)

    variant_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    edge_rows_all: list[dict[str, Any]] = []
    for spec in VARIANTS:
        groups, edge_rows = _build_groups(meta, spec)
        edge_rows_all.extend(edge_rows)
        group_scores_by_root = _group_scores(groups, meta)
        scores: dict[str, float] = {}
        group_diag_multi_gt_count = 0
        largest_group_size = 0
        for group_idx, (root, members) in enumerate(sorted(groups.items()), start=1):
            group_id = f"{VARIANT_PREFIX}:{spec['variant_id']}:{p7d.SCENE_ID}:{p7d.CHUNK_ID}:stitched_{group_idx:04d}"
            scores[group_id] = group_scores_by_root[root]
            gt_vals = [meta[oid]["diagnostic_gt_dominant"] for oid in members if meta[oid]["diagnostic_gt_dominant"]]
            unique_gt = sorted(set(gt_vals))
            group_diag_multi_gt_count += int(len(unique_gt) > 1)
            largest_group_size = max(largest_group_size, len(members))
            group_rows.append(
                {
                    "schema_version": "stream4d_v102_phase7f_stitch_group_v1",
                    "phase_id": PHASE_ID,
                    "variant_id": spec["variant_id"],
                    "stitched_group_id": group_id,
                    "source_component_count": len(members),
                    "source_components_joined": "|".join(members),
                    "frame_count": len({f for oid in members for f in meta[oid]["frames"]}),
                    "node_count": sum(meta[oid]["node_count"] for oid in members),
                    "score": scores[group_id],
                    "diagnostic_gt_unique_count": len(unique_gt),
                    "uses_gt_for_prediction": False,
                    "uses_gt_for_diagnostic_labels": True,
                    "uses_future": False,
                }
            )
        summary, render_diag = _evaluate_variant(node_rows, groups, scores, spec["variant_id"])
        variant_rows.append(
            {
                "schema_version": "stream4d_v102_phase7f_variant_metric_v1",
                "phase_id": PHASE_ID,
                "variant_id": spec["variant_id"],
                "metric_scope": "chunk32_scene0050_local_diagnostic_not_full_dev",
                "centroid_cosine_min": spec["centroid_cosine_min"],
                "max_frame_gap": spec["max_frame_gap"],
                "require_dominant_prototype_match": spec["require_dominant_prototype_match"],
                "stitch_edge_count": sum(1 for row in edge_rows if row["variant_id"] == spec["variant_id"]),
                "group_count": render_diag["group_count"],
                "largest_group_source_component_count": largest_group_size,
                "diagnostic_multi_gt_group_count": group_diag_multi_gt_count,
                "MV_AP_window": summary.get("ap"),
                "MV_AP50_window": summary.get("ap50"),
                "MV_AP25_window": summary.get("ap25"),
                "MV_AP_scene": summary.get("ap"),
                "MV_AP50_scene": summary.get("ap50"),
                "ScoreFreeMatch50_window": (summary.get("score_free_match_at_050") or {}).get("recall"),
                "ScoreFreeMatch25_window": (summary.get("score_free_match_at_025") or {}).get("recall"),
                **render_diag,
                "uses_gt_for_prediction": False,
                "uses_gt_for_eval": True,
                "uses_future": False,
            }
        )

    base = next(row for row in variant_rows if row["variant_id"] == "F0_no_stitch_s8_score")
    best = max(
        variant_rows,
        key=lambda row: (
            _num(row.get("ScoreFreeMatch50_window")),
            _num(row.get("MV_AP50_window")),
            _num(row.get("MV_AP_window")),
            -_num(row.get("diagnostic_multi_gt_group_count")),
        ),
    )
    phase7e_best_ap50 = _num(phase7e.get("best_MV_AP50_window"))
    best_delta_ap50_vs_7e = _num(best.get("MV_AP50_window")) - phase7e_best_ap50
    best_delta_sf50_vs_base = _num(best.get("ScoreFreeMatch50_window")) - _num(base.get("ScoreFreeMatch50_window"))
    local_improves = bool(best_delta_ap50_vs_7e > 1e-12 or best_delta_sf50_vs_base > 1e-12)
    decision = (
        "PASS_PHASE7F_GT_FREE_STITCH_LOCAL_DIAGNOSTIC_IMPROVES__FORMAL_TARGET_NOT_CLAIMED"
        if local_improves
        else "NO_GO_PHASE7F_GT_FREE_STITCH_NO_LOCAL_GAIN"
    )
    gate_rows = [
        {
            "schema_version": "stream4d_v102_phase7f_gate_v1",
            "phase_id": PHASE_ID,
            "gate": "best_local_diagnostic_improves_over_phase7e_or_base",
            "pass": local_improves,
            "observed": f"delta_ap50_vs_phase7e={best_delta_ap50_vs_7e}; delta_sf50_vs_base={best_delta_sf50_vs_base}",
            "required": ">0 local AP50 delta vs Phase7e or >0 ScoreFreeMatch50 delta vs no-stitch baseline",
        },
        {
            "schema_version": "stream4d_v102_phase7f_gate_v1",
            "phase_id": PHASE_ID,
            "gate": "formal_v102_target_achieved",
            "pass": False,
            "observed": "not claimed from local stitching diagnostic",
            "required": "full-dev/holdout formal AP repair gate",
        },
    ]
    _write_csv(OUT_DIR / "stitch_edge_rows.csv", edge_rows_all)
    _write_csv(OUT_DIR / "stitch_group_rows.csv", group_rows)
    _write_csv(OUT_DIR / "stitch_variant_metric_rows.csv", variant_rows)
    _write_csv(OUT_DIR / "stitch_gate_rows.csv", gate_rows)
    summary = {
        "schema_version": "stream4d_v102_phase7f_component_stitch_summary_v1",
        "phase_id": PHASE_ID,
        "created_unix_time": time.time(),
        "runtime_sec": time.time() - t0,
        "decision": decision,
        "metric_scope": "chunk32_scene0050_local_diagnostic_not_full_dev",
        "variant_count": len(VARIANTS),
        "base_variant_id": base["variant_id"],
        "base_MV_AP_window": base.get("MV_AP_window"),
        "base_MV_AP50_window": base.get("MV_AP50_window"),
        "base_ScoreFreeMatch50_window": base.get("ScoreFreeMatch50_window"),
        "phase7e_best_MV_AP50_window": phase7e_best_ap50,
        "best_variant_id": best["variant_id"],
        "best_MV_AP_window": best.get("MV_AP_window"),
        "best_MV_AP50_window": best.get("MV_AP50_window"),
        "best_MV_AP25_window": best.get("MV_AP25_window"),
        "best_ScoreFreeMatch50_window": best.get("ScoreFreeMatch50_window"),
        "best_ScoreFreeMatch25_window": best.get("ScoreFreeMatch25_window"),
        "best_delta_MV_AP50_window_vs_phase7e": best_delta_ap50_vs_7e,
        "best_delta_ScoreFreeMatch50_window_vs_base": best_delta_sf50_vs_base,
        "best_group_count": best.get("group_count"),
        "best_stitch_edge_count": best.get("stitch_edge_count"),
        "best_largest_group_source_component_count": best.get("largest_group_source_component_count"),
        "best_diagnostic_multi_gt_group_count": best.get("diagnostic_multi_gt_group_count"),
        "local_diagnostic_improves": local_improves,
        "formal_v102_target_achieved": False,
        "formal_target_blocker": "Phase7f is a local chunk32 component-stitch diagnostic; Phase6 full repair remains blocked by Phase1b.",
        "uses_gt_for_prediction": False,
        "uses_gt_for_eval": True,
        "uses_future": False,
        "truthfulness_note": (
            "Stitch edges use only same-frame cannot-link, RADIO centroid cosine, frame gap, and RADIO prototype agreement. "
            "Diagnostic GT labels are used only after grouping to audit contamination and AP."
        ),
        "outputs": {
            "summary": _rel(OUT_DIR / "summary.json"),
            "stitch_edge_rows": _rel(OUT_DIR / "stitch_edge_rows.csv"),
            "stitch_group_rows": _rel(OUT_DIR / "stitch_group_rows.csv"),
            "stitch_variant_metric_rows": _rel(OUT_DIR / "stitch_variant_metric_rows.csv"),
            "stitch_gate_rows": _rel(OUT_DIR / "stitch_gate_rows.csv"),
        },
    }
    _write_json(OUT_DIR / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
