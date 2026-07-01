from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stream4d_native.v91_mask_feature_store import load_mask_feature_store  # noqa: E402
from tools import diagnose_v91_source_mask_oracle_upper_bound as oracle  # noqa: E402
from tools import run_v90_geo_semantic_witness_cover as phase4  # noqa: E402
from tools import run_v91_phase4_adaptive_uncertainty_materialization as adaptive  # noqa: E402
from tools.run_v65_scene_multiview_ap import _load_gt_2d  # noqa: E402


SUPPORT_ROWS = ROOT / "outputs/audit/v90_phase3_v82_full_support/native_carrier_support_rows.csv"
SCENES = {"scene0011_00", "scene0050_00"}


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(adaptive._jsonable(row))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(adaptive._jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _resolve_path(text: str) -> Path:
    path = Path(str(text))
    return path if path.is_absolute() else ROOT / path


def _resolve_csv_paths(text: str) -> list[Path]:
    return [_resolve_path(part.strip()) for part in str(text).split(",") if part.strip()]


def _mean(values: list[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else 0.0


def _auc(scores: list[float], labels: list[int]) -> float | None:
    pos = sum(1 for label in labels if int(label) == 1)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return None
    ordered = sorted(zip(scores, labels), key=lambda item: item[0])
    rank_sum_pos = 0.0
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j][0] == ordered[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        rank_sum_pos += avg_rank * sum(1 for _score, label in ordered[i:j] if int(label) == 1)
        i = j
    return float((rank_sum_pos - pos * (pos + 1) / 2.0) / (pos * neg))


def _cosine(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    a = np.asarray(vec_a, dtype=np.float32).reshape(-1)
    b = np.asarray(vec_b, dtype=np.float32).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 1e-12 else 0.0


def _gt_at_uv(gt: np.ndarray, uv_x: float, uv_y: float) -> int:
    h, w = gt.shape
    x = min(w - 1, max(0, int(round(float(uv_x) * (w - 1)))))
    y = min(h - 1, max(0, int(round(float(uv_y) * (h - 1)))))
    return int(gt[y, x])


def _hist_stats(counter: Counter[int]) -> dict[str, Any]:
    total = int(sum(counter.values()))
    fg_total = int(sum(v for k, v in counter.items() if int(k) > 0))
    dominant_gt = 0
    dominant_count = 0
    if fg_total > 0:
        dominant_gt, dominant_count = max(
            ((int(k), int(v)) for k, v in counter.items() if int(k) > 0),
            key=lambda item: item[1],
        )
    return {
        "support_point_count": total,
        "foreground_support_point_count": fg_total,
        "background_support_point_count": int(counter.get(0, 0)),
        "dominant_support_gt_id": int(dominant_gt),
        "dominant_support_gt_count": int(dominant_count),
        "dominant_support_gt_purity": float(dominant_count / max(1, fg_total)),
        "background_support_rate": float(counter.get(0, 0) / max(1, total)),
        "unique_foreground_gt_count": int(len([k for k in counter if int(k) > 0])),
    }


def _load_semantic_meta(paths: list[Path]) -> dict[tuple[str, int, int], dict[str, Any]]:
    out: dict[tuple[str, int, int], dict[str, Any]] = {}
    for path in paths:
        for row in _read_csv(path):
            if adaptive._bool(row.get("uses_gt_for_prediction")):
                continue
            key = (
                str(row.get("scene_id", "")),
                adaptive._int(row.get("frame_id"), -1),
                adaptive._int(row.get("mask_id"), -1),
            )
            if key[0] and key[1] >= 0 and key[2] > 0:
                out[key] = {
                    "semantic_prototype_id": row.get("semantic_prototype_id", ""),
                    "semantic_prototype_margin": adaptive._num(row.get("semantic_prototype_margin"), 0.0),
                    "semantic_entropy": adaptive._num(row.get("semantic_entropy"), 1.0),
                    "broad_background_risk": adaptive._bool(row.get("broad_background_risk")),
                    "feature_source": adaptive._rel(path),
                }
    return out


def _source_mask_gt_index(mask_dirs: dict[str, Path]) -> dict[tuple[str, int, int], dict[str, Any]]:
    frame_keys = sorted(
        {
            (str(row.get("scene_id", "")), adaptive._int(row.get("frame_id"), -1))
            for row in _read_csv(SUPPORT_ROWS)
            if str(row.get("scene_id", "")) in SCENES and adaptive._int(row.get("frame_id"), -1) >= 0
        }
    )
    out: dict[tuple[str, int, int], dict[str, Any]] = {}
    for scene, frame_id in frame_keys:
        source_path = mask_dirs[scene] / f"{int(frame_id)}.png"
        if not source_path.exists():
            continue
        source = phase4._read_label(source_path)
        gt = _load_gt_2d(scene, int(frame_id), source.shape)
        gt_area, src_area, inter = oracle._gt_mask_pair_stats(gt, source)
        by_src: dict[int, dict[str, Any]] = {}
        for (gt_id, src_id), intersection in inter.items():
            union = int(gt_area.get(gt_id, 0)) + int(src_area.get(src_id, 0)) - int(intersection)
            iou = float(intersection / union) if union > 0 else 0.0
            precision = float(intersection / max(1, int(src_area.get(src_id, 0))))
            coverage = float(intersection / max(1, int(gt_area.get(gt_id, 0))))
            prev = by_src.get(int(src_id))
            if prev is None or iou > float(prev.get("source_best_gt_iou", -1.0)):
                by_src[int(src_id)] = {
                    "source_best_gt_id": int(gt_id),
                    "source_best_gt_iou": iou,
                    "source_best_gt_precision": precision,
                    "source_best_gt_coverage": coverage,
                    "source_mask_area": int(src_area.get(src_id, 0)),
                    "source_gt_intersection_pixels": int(intersection),
                    "source_gt_pixels": int(gt_area.get(gt_id, 0)),
                }
        for src_id, stats in by_src.items():
            out[(scene, int(frame_id), int(src_id))] = stats
    return out


def _build_group_rows(
    *,
    vectors: dict[tuple[str, int, int], np.ndarray],
    semantic_meta: dict[tuple[str, int, int], dict[str, Any]],
    source_gt: dict[tuple[str, int, int], dict[str, Any]],
    mask_dirs: dict[str, Path],
) -> tuple[list[dict[str, Any]], dict[tuple[str, int, int], dict[str, Any]]]:
    gt_cache: dict[tuple[str, int], np.ndarray] = {}
    group_hist: dict[tuple[str, str, int, int], Counter[int]] = defaultdict(Counter)
    group_meta: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    for row in _read_csv(SUPPORT_ROWS):
        if adaptive._bool(row.get("uses_gt_for_prediction")) or adaptive._bool(row.get("uses_future")):
            continue
        if not adaptive._bool(row.get("native_support_allowed", "True")):
            continue
        scene = str(row.get("scene_id", ""))
        if scene not in SCENES:
            continue
        slot = str(row.get("local_slot_id", ""))
        frame_id = adaptive._int(row.get("frame_id"), -1)
        mask_id = adaptive._int(row.get("mask_id"), -1)
        if not slot or frame_id < 0 or mask_id <= 0:
            continue
        frame_key = (scene, int(frame_id))
        if frame_key not in gt_cache:
            source_path = mask_dirs[scene] / f"{int(frame_id)}.png"
            if not source_path.exists():
                continue
            source = phase4._read_label(source_path)
            gt_cache[frame_key] = _load_gt_2d(scene, int(frame_id), source.shape)
        gt_id = _gt_at_uv(gt_cache[frame_key], adaptive._num(row.get("carrier_uv_x")), adaptive._num(row.get("carrier_uv_y")))
        group_key = (scene, slot, int(frame_id), int(mask_id))
        group_hist[group_key][gt_id] += 1
        group_meta.setdefault(
            group_key,
            {
                "scene_id": scene,
                "local_slot_id": slot,
                "chunk_id": row.get("chunk_id", ""),
                "cluster_id": row.get("cluster_id", ""),
                "frame_id": int(frame_id),
                "mask_id": int(mask_id),
            },
        )
    rows: list[dict[str, Any]] = []
    unique_masks: dict[tuple[str, int, int], dict[str, Any]] = {}
    for key, hist in sorted(group_hist.items()):
        scene, slot, frame_id, mask_id = key
        mask_key = (scene, int(frame_id), int(mask_id))
        vec = vectors.get(mask_key)
        src = source_gt.get(mask_key, {})
        sem = semantic_meta.get(mask_key, {})
        stats = _hist_stats(hist)
        source_gt_id = adaptive._int(src.get("source_best_gt_id"), 0)
        row = {
            **group_meta[key],
            **stats,
            **src,
            **sem,
            "feature_available": vec is not None,
            "support_gt_matches_source_best_gt": bool(stats["dominant_support_gt_id"] > 0 and stats["dominant_support_gt_id"] == source_gt_id),
            "source_iou_ge_025": adaptive._num(src.get("source_best_gt_iou")) >= 0.25,
            "source_iou_ge_050": adaptive._num(src.get("source_best_gt_iou")) >= 0.50,
            "diagnostic_only_uses_gt": True,
            "uses_gt_for_prediction": False,
            "uses_future": False,
        }
        rows.append(row)
        if vec is not None and mask_key not in unique_masks:
            unique_masks[mask_key] = {
                "scene_id": scene,
                "frame_id": int(frame_id),
                "mask_id": int(mask_id),
                **src,
                **sem,
                "feature_available": True,
                "diagnostic_only_uses_gt": True,
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
    return rows, unique_masks


def _pair_auc(rows: list[dict[str, Any]], vectors: dict[tuple[str, int, int], np.ndarray], label_key: str, seed: int, max_pairs: int) -> dict[str, Any]:
    usable = [
        row for row in rows
        if adaptive._int(row.get(label_key), 0) > 0
        and (str(row.get("scene_id", "")), adaptive._int(row.get("frame_id"), -1), adaptive._int(row.get("mask_id"), -1)) in vectors
    ]
    by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_scene_label: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in usable:
        scene = str(row.get("scene_id", ""))
        label = adaptive._int(row.get(label_key), 0)
        by_scene[scene].append(row)
        by_scene_label[(scene, label)].append(row)
    rng = np.random.default_rng(seed)
    scores: list[float] = []
    labels: list[int] = []
    pos_labels = [key for key, vals in by_scene_label.items() if len(vals) >= 2]
    scenes = [scene for scene, vals in by_scene.items() if len({adaptive._int(r.get(label_key), 0) for r in vals}) >= 2]
    half = max_pairs // 2
    for _ in range(half):
        if not pos_labels:
            break
        scene, label = pos_labels[int(rng.integers(0, len(pos_labels)))]
        vals = by_scene_label[(scene, label)]
        a, b = rng.choice(len(vals), size=2, replace=False)
        ra, rb = vals[int(a)], vals[int(b)]
        va = vectors[(str(ra["scene_id"]), adaptive._int(ra["frame_id"], -1), adaptive._int(ra["mask_id"], -1))]
        vb = vectors[(str(rb["scene_id"]), adaptive._int(rb["frame_id"], -1), adaptive._int(rb["mask_id"], -1))]
        scores.append(_cosine(va, vb))
        labels.append(1)
    for _ in range(half):
        if not scenes:
            break
        scene = scenes[int(rng.integers(0, len(scenes)))]
        vals = by_scene[scene]
        for _try in range(20):
            a, b = rng.choice(len(vals), size=2, replace=False)
            ra, rb = vals[int(a)], vals[int(b)]
            if adaptive._int(ra.get(label_key), 0) != adaptive._int(rb.get(label_key), 0):
                va = vectors[(str(ra["scene_id"]), adaptive._int(ra["frame_id"], -1), adaptive._int(ra["mask_id"], -1))]
                vb = vectors[(str(rb["scene_id"]), adaptive._int(rb["frame_id"], -1), adaptive._int(rb["mask_id"], -1))]
                scores.append(_cosine(va, vb))
                labels.append(0)
                break
    auc = _auc(scores, labels)
    return {
        "label_key": label_key,
        "sampled_pairs": len(scores),
        "positive_pairs": int(sum(labels)),
        "negative_pairs": int(len(labels) - sum(labels)),
        "same_label_auc": auc if auc is not None else "",
        "score_mean_positive": _mean([score for score, label in zip(scores, labels) if label == 1]),
        "score_mean_negative": _mean([score for score, label in zip(scores, labels) if label == 0]),
        "diagnostic_only_uses_gt": True,
        "uses_gt_for_prediction": False,
        "uses_future": False,
    }


def _slot_proto_rows(group_rows: list[dict[str, Any]], vectors: dict[tuple[str, int, int], np.ndarray]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    accum: dict[tuple[str, str], np.ndarray] = {}
    weights: dict[tuple[str, str], float] = defaultdict(float)
    for row in group_rows:
        key = (str(row.get("scene_id", "")), adaptive._int(row.get("frame_id"), -1), adaptive._int(row.get("mask_id"), -1))
        vec = vectors.get(key)
        if vec is None:
            continue
        slot_key = (str(row.get("scene_id", "")), str(row.get("local_slot_id", "")))
        weight = max(1.0, adaptive._num(row.get("foreground_support_point_count"), 1.0))
        if adaptive._bool(row.get("broad_background_risk")):
            weight *= 0.25
        accum[slot_key] = accum.get(slot_key, np.zeros_like(vec)) + vec * weight
        weights[slot_key] += weight
    proto: dict[tuple[str, str], np.ndarray] = {}
    for key, vec in accum.items():
        out = vec / max(1e-8, weights[key])
        norm = float(np.linalg.norm(out))
        if norm > 1e-8:
            proto[key] = out / norm
    rows: list[dict[str, Any]] = []
    scores: list[float] = []
    labels: list[int] = []
    for row in group_rows:
        key = (str(row.get("scene_id", "")), adaptive._int(row.get("frame_id"), -1), adaptive._int(row.get("mask_id"), -1))
        vec = vectors.get(key)
        slot_vec = proto.get((str(row.get("scene_id", "")), str(row.get("local_slot_id", ""))))
        if vec is None or slot_vec is None:
            continue
        score = _cosine(vec, slot_vec)
        label = 1 if adaptive._bool(row.get("support_gt_matches_source_best_gt")) else 0
        scores.append(score)
        labels.append(label)
        rows.append(
            {
                "scene_id": row.get("scene_id", ""),
                "local_slot_id": row.get("local_slot_id", ""),
                "frame_id": row.get("frame_id", ""),
                "mask_id": row.get("mask_id", ""),
                "slot_proto_cosine": score,
                "support_gt_matches_source_best_gt": bool(label),
                "source_best_gt_iou": row.get("source_best_gt_iou", ""),
                "dominant_support_gt_purity": row.get("dominant_support_gt_purity", ""),
                "broad_background_risk": row.get("broad_background_risk", ""),
                "diagnostic_only_uses_gt": True,
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
    auc = _auc(scores, labels)
    summary = {
        "slot_proto_count": len(proto),
        "slot_proto_scored_rows": len(rows),
        "slot_proto_auc_for_support_source_match": auc if auc is not None else "",
        "slot_proto_cosine_mean_match": _mean([score for score, label in zip(scores, labels) if label == 1]),
        "slot_proto_cosine_mean_mismatch": _mean([score for score, label in zip(scores, labels) if label == 0]),
    }
    return rows, summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    out = _resolve_path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    feature_store_path = _resolve_path(args.feature_store)
    semantic_paths = _resolve_csv_paths(args.semantic_feature_rows)
    store = load_mask_feature_store(feature_store_path)
    vectors = store.as_keyed_dict()
    semantic_meta = _load_semantic_meta(semantic_paths)
    mask_dirs = phase4._mask_dir_by_scene()
    source_gt = _source_mask_gt_index(mask_dirs)
    group_rows, unique_masks = _build_group_rows(vectors=vectors, semantic_meta=semantic_meta, source_gt=source_gt, mask_dirs=mask_dirs)
    unique_rows = list(unique_masks.values())
    unique_vectors = {key: vec for key, vec in vectors.items() if key in unique_masks}
    pair_rows = [
        {"pair_scope": "unique_source_mask", **_pair_auc(unique_rows, unique_vectors, "source_best_gt_id", seed=31001, max_pairs=int(args.max_pairs))},
        {"pair_scope": "slot_group_support", **_pair_auc(group_rows, vectors, "dominant_support_gt_id", seed=31002, max_pairs=int(args.max_pairs))},
    ]
    slot_proto_rows, slot_proto_summary = _slot_proto_rows(group_rows, vectors)
    scene_rows: list[dict[str, Any]] = []
    for scene in sorted(SCENES):
        scene_groups = [row for row in group_rows if row.get("scene_id") == scene]
        scene_unique = [row for row in unique_rows if row.get("scene_id") == scene]
        scene_rows.append(
            {
                "scene_id": scene,
                "slot_group_rows": len(scene_groups),
                "unique_source_mask_rows": len(scene_unique),
                "feature_available_group_rate": _mean([1.0 if adaptive._bool(row.get("feature_available")) else 0.0 for row in scene_groups]),
                "support_gt_source_gt_match_rate": _mean([1.0 if adaptive._bool(row.get("support_gt_matches_source_best_gt")) else 0.0 for row in scene_groups]),
                "source_iou_ge_025_rate": _mean([1.0 if adaptive._bool(row.get("source_iou_ge_025")) else 0.0 for row in scene_groups]),
                "source_iou_ge_050_rate": _mean([1.0 if adaptive._bool(row.get("source_iou_ge_050")) else 0.0 for row in scene_groups]),
                "dominant_support_gt_purity_mean": _mean([adaptive._num(row.get("dominant_support_gt_purity")) for row in scene_groups]),
                "background_support_rate_mean": _mean([adaptive._num(row.get("background_support_rate")) for row in scene_groups]),
                "broad_background_risk_rate": _mean([1.0 if adaptive._bool(row.get("broad_background_risk")) else 0.0 for row in scene_groups]),
            }
        )
    summary = {
        "phase": "v91_radio_feature_store_quality",
        "schema": "stream4d_v91_radio_feature_store_quality_v1",
        "feature_store": adaptive._rel(feature_store_path),
        "feature_store_backend": store.backend,
        "feature_store_layer": store.layer,
        "feature_store_row_count": int(store.features.shape[0]),
        "feature_store_dim": int(store.features.shape[1]),
        "semantic_feature_rows": [adaptive._rel(path) for path in semantic_paths],
        "vector_count": len(vectors),
        "slot_group_rows": len(group_rows),
        "unique_source_mask_rows": len(unique_rows),
        "source_mask_gt_pair_auc": next((row.get("same_label_auc") for row in pair_rows if row.get("pair_scope") == "unique_source_mask"), ""),
        "slot_group_support_gt_pair_auc": next((row.get("same_label_auc") for row in pair_rows if row.get("pair_scope") == "slot_group_support"), ""),
        **slot_proto_summary,
        "support_gt_source_gt_match_rate": _mean([1.0 if adaptive._bool(row.get("support_gt_matches_source_best_gt")) else 0.0 for row in group_rows]),
        "source_iou_ge_025_rate": _mean([1.0 if adaptive._bool(row.get("source_iou_ge_025")) else 0.0 for row in group_rows]),
        "source_iou_ge_050_rate": _mean([1.0 if adaptive._bool(row.get("source_iou_ge_050")) else 0.0 for row in group_rows]),
        "diagnostic_gt_usage": "GT is used only for post-hoc separability labels; no GT label is used by prediction/readout artifacts.",
        "uses_gt_for_prediction": False,
        "uses_future": False,
        "runtime_sec": time.time() - started,
    }
    _write_csv(out / "radio_slot_group_quality_rows.csv", group_rows)
    _write_csv(out / "radio_unique_mask_quality_rows.csv", unique_rows)
    _write_csv(out / "radio_pair_auc_rows.csv", pair_rows)
    _write_csv(out / "radio_slot_proto_rows.csv", slot_proto_rows)
    _write_csv(out / "scene_quality_rows.csv", scene_rows)
    _write_json(out / "summary.json", summary)
    outputs = [
        out / "radio_slot_group_quality_rows.csv",
        out / "radio_unique_mask_quality_rows.csv",
        out / "radio_pair_auc_rows.csv",
        out / "radio_slot_proto_rows.csv",
        out / "scene_quality_rows.csv",
        out / "summary.json",
    ]
    _write_json(out / "SHA256SUMS.json", {adaptive._rel(path): adaptive._sha256(path) for path in outputs if path.exists()})
    print(json.dumps(adaptive._jsonable(summary), indent=2, sort_keys=True), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose v91 RADIO NPZ feature-store separability and slot readout usage.")
    parser.add_argument("--feature-store", default="outputs/audit/v91_radio_mask_features_npz")
    parser.add_argument("--semantic-feature-rows", default="outputs/audit/v91_radio_mask_features_npz_scene0011/mask_feature_rows.csv,outputs/audit/v91_radio_mask_features_npz_scene0050/mask_feature_rows.csv")
    parser.add_argument("--output-root", default="outputs/audit/v91_radio_feature_store_quality")
    parser.add_argument("--max-pairs", type=int, default=200000)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
