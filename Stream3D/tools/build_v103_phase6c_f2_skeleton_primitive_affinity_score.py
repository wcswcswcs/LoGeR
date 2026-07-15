#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
if str(STREAM3D_ROOT) not in sys.path:
    sys.path.insert(0, str(STREAM3D_ROOT))

from tools.build_v103_phase6_mask_clustering_local_object_birth import (  # noqa: E402
    DEFAULT_PHASE2_SCENE0011,
    DEFAULT_PHASE2_SCENE0050,
    _evaluate_variant,
    _jsonable,
    _project,
    _read_json,
    _rel,
    _write_csv,
    _write_json,
)


PHASE_ID = "v103_phase6c_f2_skeleton_primitive_affinity_score"
DEFAULT_F2_ROOT = STREAM3D_ROOT / "outputs/audit/v100_phase2c_overlap3_local_repair"
DEFAULT_PHASE5_ROOT = STREAM3D_ROOT / "outputs/audit/v103_phase5_positive_core_pooling_q5c_repair5_r1"
DEFAULT_OUT = STREAM3D_ROOT / "outputs/audit/v103_phase6c_f2_skeleton_primitive_score_r1"
DEFAULT_SUBSET_BASELINE = STREAM3D_ROOT / "outputs/audit/v103_phase6_baseline_subset_contract_r1/baseline_subset_metric_rows.csv"


VARIANTS = [
    {
        "variant_id": "S0_f2_original_score_replay",
        "score_policy": "f2_original_score",
    },
    {
        "variant_id": "S1_f2_original_plus_primitive_coherence025",
        "score_policy": "original_plus_primitive_coherence",
        "coherence_weight": 0.25,
    },
    {
        "variant_id": "S2_primitive_coherence_frame_score",
        "score_policy": "primitive_coherence_frame_score",
    },
    {
        "variant_id": "S3_f2_original_times_primitive_coherence",
        "score_policy": "original_times_primitive_coherence",
    },
    {
        "variant_id": "S4_f2_original_minus_broad_plus_coherence",
        "score_policy": "original_minus_broad_plus_coherence",
        "coherence_weight": 0.15,
        "broad_penalty": 0.35,
    },
]


def _load_subset_baseline(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if df.empty:
        return {}
    row = df.iloc[0]
    return {
        "MV_AP_window": float(row["MV_AP_window"]),
        "MV_AP50_window": float(row["MV_AP50_window"]),
        "MV_AP25_window": float(row["MV_AP25_window"]),
        "ScoreFreeMatch50_window": float(row["ScoreFreeMatch50_window"]),
    }


def _load_phase5_scene(phase5_root: Path, scene: str) -> dict[str, Any]:
    payload = torch.load(phase5_root / scene / "mask_level_feature.pt", map_location="cpu")
    feature = payload["feature"].to(torch.float32).cpu().numpy().astype(np.float32, copy=False)
    norm = np.linalg.norm(feature, axis=1, keepdims=True)
    feature = feature / np.maximum(norm, 1e-12)
    feature[~np.isfinite(feature)] = 0.0
    return {
        "feature": feature,
        "mask_frame": payload["mask_frame"].cpu().numpy().astype(np.int64),
        "mask_label": payload["mask_label"].cpu().numpy().astype(np.int64),
        "mask_is_broad": payload["mask_is_broad"].cpu().numpy().astype(bool),
        "mask_is_object_like": payload["mask_is_object_like"].cpu().numpy().astype(bool),
        "support_count": payload["support_count"].cpu().numpy().astype(np.int64),
    }


def _phase5_index(payload: dict[str, Any]) -> dict[tuple[int, int], int]:
    return {
        (int(frame), int(label)): int(idx)
        for idx, (frame, label) in enumerate(zip(payload["mask_frame"].tolist(), payload["mask_label"].tolist()))
    }


def _object_affinity_stats(group: pd.DataFrame, phase5_payload: dict[str, Any], mask_index: dict[tuple[int, int], int]) -> dict[str, float]:
    idxs: list[int] = []
    missing = 0
    for row in group.to_dict("records"):
        key = (int(row["frame_local_index"]), int(row["selected_mask_id"]))
        idx = mask_index.get(key)
        if idx is None:
            missing += 1
        else:
            idxs.append(int(idx))
    unique = sorted(set(idxs))
    if len(unique) >= 2:
        feat = phase5_payload["feature"][np.asarray(unique, dtype=np.int64)]
        sim = feat @ feat.T
        tri = sim[np.triu_indices(sim.shape[0], k=1)]
        coherence = float(np.mean(tri)) if tri.size else 0.0
        coherence_p10 = float(np.percentile(tri, 10)) if tri.size else 0.0
    else:
        coherence = 0.0
        coherence_p10 = 0.0
    if unique:
        broad_rate = float(np.mean(phase5_payload["mask_is_broad"][unique]))
        object_like_rate = float(np.mean(phase5_payload["mask_is_object_like"][unique]))
        support_mean = float(np.mean(phase5_payload["support_count"][unique]))
    else:
        broad_rate = 1.0
        object_like_rate = 0.0
        support_mean = 0.0
    return {
        "primitive_coherence_mean": coherence,
        "primitive_coherence_p10": coherence_p10,
        "primitive_coherence_norm": float(np.clip((coherence + 1.0) * 0.5, 0.0, 1.0)),
        "selected_broad_rate": broad_rate,
        "selected_object_like_rate": object_like_rate,
        "selected_support_mean": support_mean,
        "phase5_mask_match_rate": float(len(idxs) / max(1, len(group))),
        "phase5_missing_mask_count": float(missing),
    }


def _score_object(*, original_score: float, frame_count: int, stats: dict[str, float], variant: dict[str, Any]) -> float:
    coherence = float(stats["primitive_coherence_norm"])
    broad_rate = float(stats["selected_broad_rate"])
    frame_score = float(frame_count / 32.0)
    policy = str(variant["score_policy"])
    if policy == "f2_original_score":
        return float(original_score)
    if policy == "original_plus_primitive_coherence":
        w = float(variant.get("coherence_weight", 0.25))
        return float((1.0 - w) * original_score + w * coherence)
    if policy == "primitive_coherence_frame_score":
        return float(frame_score * max(0.05, coherence) * max(0.05, 1.0 - 0.25 * broad_rate))
    if policy == "original_times_primitive_coherence":
        return float(original_score * max(0.05, coherence))
    if policy == "original_minus_broad_plus_coherence":
        w = float(variant.get("coherence_weight", 0.15))
        penalty = float(variant.get("broad_penalty", 0.35))
        return float(max(0.0, original_score * (1.0 - penalty * broad_rate)) + w * coherence)
    raise ValueError(f"unsupported score_policy={policy}")


def _load_f2_rows(
    *,
    f2_root: Path,
    phase2_summaries: dict[str, dict[str, Any]],
    phase5_payloads: dict[str, dict[str, Any]],
    dataset_split: str,
    chunk_id: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows = pd.read_parquet(f2_root / "mv_object_frame_mask_rows.parquet")
    rows = rows[(rows["dataset_split"].astype(str) == dataset_split) & (rows["chunk_id"].astype(str) == chunk_id)].copy()
    frame_to_local = {
        scene: {int(frame_id): idx for idx, frame_id in enumerate(summary["frame_ids"])}
        for scene, summary in phase2_summaries.items()
    }
    adapted: list[dict[str, Any]] = []
    for row in rows.to_dict("records"):
        scene = str(row["scene_id"])
        frame_id = int(row["frame_id"])
        if scene not in frame_to_local or frame_id not in frame_to_local[scene]:
            continue
        new = dict(row)
        new["frame_local_index"] = int(frame_to_local[scene][frame_id])
        adapted.append(new)
    if not adapted:
        raise RuntimeError(f"no F2 rows match v103 first32 subset for dataset_split={dataset_split} chunk_id={chunk_id}")
    base = pd.DataFrame(adapted)

    stat_rows: list[dict[str, Any]] = []
    indexes = {scene: _phase5_index(payload) for scene, payload in phase5_payloads.items()}
    object_stats: dict[str, dict[str, float]] = {}
    for (scene, oid), group in base.groupby(["scene_id", "mv_object_id"], sort=False):
        stats = _object_affinity_stats(group, phase5_payloads[str(scene)], indexes[str(scene)])
        object_stats[str(oid)] = stats
        stat_rows.append(
            {
                "schema_version": "stream4d_v103_phase6c_f2_object_affinity_stat_row_v1",
                "phase_id": PHASE_ID,
                "scene_id": str(scene),
                "mv_object_id": str(oid),
                "frame_count": int(group["frame_id"].nunique()),
                "original_score": float(group.get("score", pd.Series([0.0])).max()),
                **stats,
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic": False,
            }
        )
    base["_object_stats_key"] = base["mv_object_id"].astype(str)
    return base, stat_rows


def _variant_scene_rows(base: pd.DataFrame, stat_by_object: dict[str, dict[str, Any]], variant: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    scene_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    frame_counts = base.groupby("mv_object_id")["frame_id"].nunique().to_dict()
    original_scores = base.groupby("mv_object_id")["score"].max().to_dict()
    object_scores: dict[str, float] = {}
    for oid in base["mv_object_id"].astype(str).unique().tolist():
        stats = stat_by_object[str(oid)]
        object_scores[str(oid)] = _score_object(
            original_score=float(original_scores.get(oid, 0.0)),
            frame_count=int(frame_counts.get(oid, 0)),
            stats=stats,
            variant=variant,
        )
    for row in base.to_dict("records"):
        oid = str(row["mv_object_id"])
        scene = str(row["scene_id"])
        score = float(object_scores[oid])
        scene_rows[scene].append(
            {
                "schema_version": "stream4d_v103_phase6c_f2_skeleton_frame_mask_row_v1",
                "phase_id": PHASE_ID,
                "variant_id": str(variant["variant_id"]),
                "mv_object_id": oid,
                "object_id": oid,
                "scene_id": scene,
                "chunk_id": str(row.get("chunk_id", "c0000")),
                "window_id": str(row.get("window_id", "c0000")),
                "frame_local_index": int(row["frame_local_index"]),
                "frame_id": int(row["frame_id"]),
                "selected_mask_id": int(row["selected_mask_id"]),
                "mask_id_or_generated_id": int(row["mask_id_or_generated_id"]),
                "object_score": score,
                "score": score,
                "support_count": int(row.get("support_surfel_count", 0) or 0),
                "node_policy": "f2_overlap3_skeleton",
                "emit_policy": "f2_primary_emit",
                "score_policy": str(variant["score_policy"]),
                "readout_mode": "f2_skeleton_primitive_affinity_score",
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
    return scene_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score the locked F2 skeleton with v103 primitive affinity features.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--f2-root", default=str(DEFAULT_F2_ROOT))
    parser.add_argument("--phase5-root", default=str(DEFAULT_PHASE5_ROOT))
    parser.add_argument("--scene0011-phase2-root", default=str(DEFAULT_PHASE2_SCENE0011))
    parser.add_argument("--scene0050-phase2-root", default=str(DEFAULT_PHASE2_SCENE0050))
    parser.add_argument("--subset-baseline-rows", default=str(DEFAULT_SUBSET_BASELINE))
    parser.add_argument("--dataset-split", default="dev")
    parser.add_argument("--chunk-id", default="c0000")
    parser.add_argument("--min-pred-pixels", type=int, default=20)
    parser.add_argument("--min-gt-pixels", type=int, default=20)
    parser.add_argument("--cupy-device-id", type=int, default=0)
    parser.add_argument("--disable-cupy-iou", action="store_true")
    return parser


def main() -> int:
    t0 = time.time()
    args = build_parser().parse_args()
    out = _project(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    (out / "last_command.txt").write_text(" ".join([sys.executable, str(Path(__file__)), *sys.argv[1:]]) + "\n", encoding="utf-8")
    phase2_summaries = {
        "scene0011_00": _read_json(_project(args.scene0011_phase2_root) / "summary.json"),
        "scene0050_00": _read_json(_project(args.scene0050_phase2_root) / "summary.json"),
    }
    phase5_root = _project(args.phase5_root)
    phase5_payloads = {scene: _load_phase5_scene(phase5_root, scene) for scene in phase2_summaries}
    base, stat_rows = _load_f2_rows(
        f2_root=_project(args.f2_root),
        phase2_summaries=phase2_summaries,
        phase5_payloads=phase5_payloads,
        dataset_split=str(args.dataset_split),
        chunk_id=str(args.chunk_id),
    )
    stat_by_object = {str(row["mv_object_id"]): row for row in stat_rows}
    all_metric_rows: list[dict[str, Any]] = []
    all_window_rows: list[dict[str, Any]] = []
    all_selected_rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        scene_rows = _variant_scene_rows(base, stat_by_object, variant)
        window_rows, aggregate, selected_rows, _pixel_collision_count, _missing_count, _frame_count = _evaluate_variant(
            variant_id=str(variant["variant_id"]),
            scene_rows=scene_rows,
            phase2_summaries=phase2_summaries,
            min_pred_pixels=int(args.min_pred_pixels),
            min_gt_pixels=int(args.min_gt_pixels),
            use_cupy_iou=not bool(args.disable_cupy_iou),
            cupy_device_id=int(args.cupy_device_id),
        )
        aggregate.update(
            {
                "phase_id": PHASE_ID,
                "score_policy": variant["score_policy"],
                "dataset_split": str(args.dataset_split),
                "chunk_id": str(args.chunk_id),
                "metric_scope": "same_subset_as_v103_phase6_first32_c0000",
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
        all_metric_rows.append(aggregate)
        all_window_rows.extend(window_rows)
        all_selected_rows.extend(selected_rows)

    subset_baseline = _load_subset_baseline(_project(args.subset_baseline_rows))
    best = max(all_metric_rows, key=lambda r: (float(r.get("MV_AP_window", 0.0)), float(r.get("MV_AP50_window", 0.0))), default={})
    _write_csv(out / "object_affinity_stat_rows.csv", stat_rows)
    _write_csv(out / "skeleton_score_metric_rows.csv", all_metric_rows)
    _write_csv(out / "skeleton_score_window_rows.csv", all_window_rows)
    _write_csv(out / "skeleton_score_selected_rows.csv", all_selected_rows)
    summary = {
        "schema_version": "stream4d_v103_phase6c_f2_skeleton_primitive_score_summary_v1",
        "phase_id": PHASE_ID,
        "created_unix_time": time.time(),
        "runtime_sec": time.time() - t0,
        "variant_count": len(VARIANTS),
        "best_variant_id": best.get("variant_id", ""),
        "best_MV_AP_window": best.get("MV_AP_window", ""),
        "best_MV_AP50_window": best.get("MV_AP50_window", ""),
        "f2_subset_baseline": subset_baseline,
        "best_minus_f2_subset_MV_AP_window": float(best.get("MV_AP_window", 0.0)) - float(subset_baseline.get("MV_AP_window", 0.0))
        if subset_baseline
        else "",
        "uses_gt_for_prediction": False,
        "uses_gt_for_eval": True,
        "truthfulness_note": "Object-frame masks are the locked F2 overlap3 skeleton; v103 primitive affinity features affect only GT-free object scores in non-S0 variants.",
        "outputs": {
            "summary": _rel(out / "summary.json"),
            "object_affinity_stat_rows": _rel(out / "object_affinity_stat_rows.csv"),
            "skeleton_score_metric_rows": _rel(out / "skeleton_score_metric_rows.csv"),
            "skeleton_score_window_rows": _rel(out / "skeleton_score_window_rows.csv"),
            "skeleton_score_selected_rows": _rel(out / "skeleton_score_selected_rows.csv"),
            "last_command": _rel(out / "last_command.txt"),
        },
    }
    _write_json(out / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
