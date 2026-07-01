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

from tools import diagnose_v91_source_mask_oracle_upper_bound as oracle  # noqa: E402
from tools import run_v90_geo_semantic_witness_cover as phase4  # noqa: E402
from tools import run_v91_phase4_adaptive_uncertainty_materialization as adaptive  # noqa: E402
from tools.run_v65_scene_multiview_ap import _load_gt_2d  # noqa: E402


OUT = ROOT / "outputs/audit/v91_affinity_semantic_feature_quality"
SUPPORT_ROWS = ROOT / "outputs/audit/v90_phase3_v82_full_support/native_carrier_support_rows.csv"
V80_PHASE1 = ROOT / "outputs/audit/v80_phase1_streaming_affinity_features_dev_r79_semguard013125_parentmin30_incl090_signed030_ownerhard"
V80_PHASE2 = ROOT / "outputs/audit/v80_phase2_signed_affinity_dev_r79_semguard013125_parentmin30_incl090_signed030_ownerhard"
V80_PHASE4 = ROOT / "outputs/audit/v80_phase4_scale_clustering_dev_r79_semguard013125_parentmin30_incl090_signed030_ownerhard"
V79_R2 = ROOT / "outputs/audit/v79_phase1_affinity_features_r2_idf15_tk6_thr045"
V85_PHASE1 = ROOT / "outputs/audit/v85_phase1_local_affinity_feature"
SEMANTIC_ROWS = ROOT / "outputs/audit/v81_dino_feature_json_scene0011_scene0050/mask_feature_rows.csv"
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


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _auc(scores: list[float], labels: list[int]) -> float | None:
    pos = sum(1 for label in labels if label == 1)
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
        rank_sum_pos += avg_rank * sum(1 for _score, label in ordered[i:j] if label == 1)
        i = j
    return float((rank_sum_pos - pos * (pos + 1) / 2.0) / (pos * neg))


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
        "dominant_gt_id": int(dominant_gt),
        "dominant_gt_support_count": int(dominant_count),
        "dominant_gt_purity": float(dominant_count / max(1, fg_total)),
        "background_support_rate": float(counter.get(0, 0) / max(1, total)),
        "unique_foreground_gt_count": int(len([k for k in counter if int(k) > 0])),
    }


def _load_semantic_index() -> dict[tuple[str, int, int], dict[str, Any]]:
    out: dict[tuple[str, int, int], dict[str, Any]] = {}
    if not SEMANTIC_ROWS.exists():
        return out
    with SEMANTIC_ROWS.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if adaptive._bool(row.get("uses_gt_for_prediction")) or not adaptive._bool(row.get("feature_available", "True")):
                continue
            scene = str(row.get("scene_id", ""))
            if scene not in SCENES:
                continue
            key = (scene, adaptive._int(row.get("frame_id"), -1), adaptive._int(row.get("mask_id"), -1))
            if key[1] < 0 or key[2] <= 0:
                continue
            out[key] = {
                "semantic_prototype_id": row.get("semantic_prototype_id", ""),
                "semantic_prototype_margin": adaptive._num(row.get("semantic_prototype_margin"), 0.0),
                "semantic_entropy": adaptive._num(row.get("semantic_entropy"), 1.0),
                "broad_background_risk": adaptive._bool(row.get("broad_background_risk")),
                "semantic_background_score_proxy": adaptive._bool(row.get("semantic_background_score_proxy")),
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


def _build_support_diagnostic(
    mask_dirs: dict[str, Path],
    semantic_index: dict[tuple[str, int, int], dict[str, Any]],
    source_gt: dict[tuple[str, int, int], dict[str, Any]],
) -> tuple[
    dict[str, dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[tuple[str, str], str],
]:
    gt_cache: dict[tuple[str, int], np.ndarray] = {}
    carrier_hist: dict[str, Counter[int]] = defaultdict(Counter)
    slot_hist: dict[tuple[str, str], Counter[int]] = defaultdict(Counter)
    slot_proto_hist: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
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
        carrier = str(row.get("native_carrier_global_id", ""))
        if not slot or not carrier or frame_id < 0 or mask_id <= 0:
            continue
        frame_key = (scene, int(frame_id))
        if frame_key not in gt_cache:
            source_path = mask_dirs[scene] / f"{int(frame_id)}.png"
            if not source_path.exists():
                continue
            source = phase4._read_label(source_path)
            gt_cache[frame_key] = _load_gt_2d(scene, int(frame_id), source.shape)
        gt_id = _gt_at_uv(gt_cache[frame_key], adaptive._num(row.get("carrier_uv_x")), adaptive._num(row.get("carrier_uv_y")))
        carrier_hist[carrier][gt_id] += 1
        slot_hist[(scene, slot)][gt_id] += 1
        group_key = (scene, slot, int(frame_id), int(mask_id))
        group_hist[group_key][gt_id] += 1
        feat = semantic_index.get((scene, int(frame_id), int(mask_id)), {})
        proto = str(feat.get("semantic_prototype_id", ""))
        if proto:
            slot_proto_hist[(scene, slot)][proto] += 1
        if group_key not in group_meta:
            group_meta[group_key] = {
                "scene_id": scene,
                "local_slot_id": slot,
                "cluster_id": row.get("cluster_id", ""),
                "chunk_id": row.get("chunk_id", ""),
                "frame_id": int(frame_id),
                "mask_id": int(mask_id),
            }

    carrier_profile: dict[str, dict[str, Any]] = {
        carrier: {"carrier_id": carrier, **_hist_stats(hist)}
        for carrier, hist in carrier_hist.items()
    }
    slot_proto: dict[tuple[str, str], str] = {
        key: str(counter.most_common(1)[0][0])
        for key, counter in slot_proto_hist.items()
        if counter
    }
    cluster_rows: list[dict[str, Any]] = []
    for (scene, slot), hist in sorted(slot_hist.items()):
        stats = _hist_stats(hist)
        cluster_rows.append(
            {
                "scene_id": scene,
                "local_slot_id": slot,
                "dominant_semantic_prototype_id": slot_proto.get((scene, slot), ""),
                **stats,
                "diagnostic_only_uses_gt": True,
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )

    readout_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    for key, hist in sorted(group_hist.items()):
        scene, slot, frame_id, mask_id = key
        stats = _hist_stats(hist)
        src = source_gt.get((scene, int(frame_id), int(mask_id)), {})
        sem = semantic_index.get((scene, int(frame_id), int(mask_id)), {})
        proto = str(sem.get("semantic_prototype_id", ""))
        row = {
            **group_meta[key],
            **stats,
            **src,
            **sem,
            "slot_dominant_semantic_prototype_id": slot_proto.get((scene, slot), ""),
            "semantic_proto_match_slot": bool(proto and proto == slot_proto.get((scene, slot), "")),
            "support_gt_matches_source_best_gt": bool(
                stats["dominant_gt_id"] > 0 and int(stats["dominant_gt_id"]) == adaptive._int(src.get("source_best_gt_id"), -1)
            ),
            "source_iou_ge_025": adaptive._num(src.get("source_best_gt_iou")) >= 0.25,
            "source_iou_ge_050": adaptive._num(src.get("source_best_gt_iou")) >= 0.50,
            "diagnostic_only_uses_gt": True,
            "uses_gt_for_prediction": False,
            "uses_future": False,
        }
        readout_rows.append(row)
        if not row["support_gt_matches_source_best_gt"] or adaptive._num(src.get("source_best_gt_iou")) < 0.25 or stats["dominant_gt_purity"] < 0.75:
            failure_rows.append({**row, "failure_type": "weak_feature_to_mask_readout_alignment"})
    return carrier_profile, cluster_rows, readout_rows, failure_rows, slot_proto


def _signed_pair_diagnostics(carrier_profile: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    specs = [
        ("positive_candidate", V80_PHASE2 / "positive_candidate_rows.csv", "positive_affinity"),
        ("signed_neighbor", V80_PHASE2 / "signed_neighbor_rows.csv", "signed_affinity"),
        ("negative_candidate", V80_PHASE2 / "negative_candidate_rows.csv", "positive_affinity"),
    ]
    out: list[dict[str, Any]] = []
    for name, path, score_key in specs:
        scores: list[float] = []
        labels: list[int] = []
        same_gt = 0
        diff_gt = 0
        bg_or_unknown = 0
        rows_used = 0
        same_frame_rows = 0
        for row in _read_csv(path):
            left = carrier_profile.get(str(row.get("carrier_i", "")), {})
            right = carrier_profile.get(str(row.get("carrier_j", "")), {})
            lgt = adaptive._int(left.get("dominant_gt_id"), 0)
            rgt = adaptive._int(right.get("dominant_gt_id"), 0)
            if lgt <= 0 or rgt <= 0:
                bg_or_unknown += 1
                continue
            label = 1 if lgt == rgt else 0
            labels.append(label)
            score = adaptive._num(row.get(score_key), adaptive._num(row.get("positive_affinity"), 0.0))
            scores.append(score)
            same_gt += int(label == 1)
            diff_gt += int(label == 0)
            same_frame_rows += int(adaptive._bool(row.get("same_frame_flag")))
            rows_used += 1
        auc = _auc(scores, labels)
        out.append(
            {
                "pair_source": name,
                "path": adaptive._rel(path),
                "score_key": score_key,
                "rows_used": rows_used,
                "same_dominant_gt_pairs": same_gt,
                "different_dominant_gt_pairs": diff_gt,
                "background_or_unknown_pairs": bg_or_unknown,
                "same_dominant_gt_rate": float(same_gt / max(1, rows_used)),
                "same_frame_pair_rate": float(same_frame_rows / max(1, rows_used)),
                "score_auc_for_same_dominant_gt": auc if auc is not None else "",
                "score_mean_same_gt": _mean([score for score, label in zip(scores, labels) if label == 1]),
                "score_mean_diff_gt": _mean([score for score, label in zip(scores, labels) if label == 0]),
                "diagnostic_only_uses_gt": True,
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
    return out


def _aggregate_readout(readout_rows: list[dict[str, Any]], cluster_rows: list[dict[str, Any]], pair_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in readout_rows:
        by_scene[str(row.get("scene_id", ""))].append(row)
    scene_rows: list[dict[str, Any]] = []
    for scene, rows in sorted(by_scene.items()):
        scene_rows.append(
            {
                "scene_id": scene,
                "readout_rows": len(rows),
                "support_gt_source_gt_match_rate": _mean([1.0 if adaptive._bool(r.get("support_gt_matches_source_best_gt")) else 0.0 for r in rows]),
                "source_best_gt_iou_mean": _mean([adaptive._num(r.get("source_best_gt_iou")) for r in rows]),
                "source_iou_ge_025_rate": _mean([1.0 if adaptive._bool(r.get("source_iou_ge_025")) else 0.0 for r in rows]),
                "source_iou_ge_050_rate": _mean([1.0 if adaptive._bool(r.get("source_iou_ge_050")) else 0.0 for r in rows]),
                "dominant_gt_purity_mean": _mean([adaptive._num(r.get("dominant_gt_purity")) for r in rows]),
                "background_support_rate_mean": _mean([adaptive._num(r.get("background_support_rate")) for r in rows]),
                "semantic_proto_match_slot_rate": _mean([1.0 if adaptive._bool(r.get("semantic_proto_match_slot")) else 0.0 for r in rows]),
                "broad_background_risk_rate": _mean([1.0 if adaptive._bool(r.get("broad_background_risk")) else 0.0 for r in rows]),
            }
        )
    cluster_purity = _mean([adaptive._num(row.get("dominant_gt_purity")) for row in cluster_rows])
    cluster_bg = _mean([adaptive._num(row.get("background_support_rate")) for row in cluster_rows])
    readout_match = _mean([1.0 if adaptive._bool(row.get("support_gt_matches_source_best_gt")) else 0.0 for row in readout_rows])
    iou25 = _mean([1.0 if adaptive._bool(row.get("source_iou_ge_025")) else 0.0 for row in readout_rows])
    iou50 = _mean([1.0 if adaptive._bool(row.get("source_iou_ge_050")) else 0.0 for row in readout_rows])
    signed = next((row for row in pair_rows if row.get("pair_source") == "signed_neighbor"), {})
    pair_same_rate = adaptive._num(signed.get("same_dominant_gt_rate"), 0.0)
    pair_auc = signed.get("score_auc_for_same_dominant_gt", "")
    return {
        "scene_rows": scene_rows,
        "summary": {
            "phase": "v91_affinity_semantic_feature_quality",
            "schema": "stream4d_v91_affinity_semantic_feature_quality_v1",
            "feature_sources": {
                "v80_phase1": adaptive._rel(V80_PHASE1),
                "v80_phase2": adaptive._rel(V80_PHASE2),
                "v80_phase4": adaptive._rel(V80_PHASE4),
                "v79_r2_counterexample": adaptive._rel(V79_R2),
                "semantic_rows": adaptive._rel(SEMANTIC_ROWS),
            },
            "v80_phase1_decision": _load_json(V80_PHASE1 / "summary.json").get("decision", ""),
            "v80_phase2_decision": _load_json(V80_PHASE2 / "summary.json").get("decision", ""),
            "v85_phase1_decision": _load_json(V85_PHASE1 / "feature_summary.json").get("decision", ""),
            "v79_r2_decision": _load_json(V79_R2 / "summary.json").get("decision", ""),
            "carrier_cluster_rows": len(cluster_rows),
            "mask_readout_rows": len(readout_rows),
            "cluster_dominant_gt_purity_mean": cluster_purity,
            "cluster_background_support_rate_mean": cluster_bg,
            "readout_support_gt_source_gt_match_rate": readout_match,
            "readout_source_iou_ge_025_rate": iou25,
            "readout_source_iou_ge_050_rate": iou50,
            "signed_neighbor_same_dominant_gt_rate": pair_same_rate,
            "signed_neighbor_auc_for_same_dominant_gt": pair_auc,
            "diagnostic_gt_usage": "GT is used only to audit feature/readout quality after artifacts are produced; no row is a legal method candidate.",
            "uses_gt_for_prediction": False,
            "uses_future": False,
        },
    }


def run(_args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    mask_dirs = phase4._mask_dir_by_scene()
    semantic_index = _load_semantic_index()
    source_gt = _source_mask_gt_index(mask_dirs)
    carrier_profile, cluster_rows, readout_rows, failure_rows, _slot_proto = _build_support_diagnostic(mask_dirs, semantic_index, source_gt)
    carrier_rows = list(carrier_profile.values())
    pair_rows = _signed_pair_diagnostics(carrier_profile)
    aggregate = _aggregate_readout(readout_rows, cluster_rows, pair_rows)
    scene_rows = aggregate["scene_rows"]
    summary = {
        **aggregate["summary"],
        "carrier_profile_rows": len(carrier_rows),
        "pair_diagnostic_rows": len(pair_rows),
        "scene_summary_rows": len(scene_rows),
        "failure_case_rows": len(failure_rows),
        "runtime_sec": time.time() - started,
    }

    _write_csv(OUT / "carrier_gt_profile_rows.csv", carrier_rows)
    _write_csv(OUT / "cluster_gt_quality_rows.csv", cluster_rows)
    _write_csv(OUT / "mask_readout_quality_rows.csv", readout_rows)
    _write_csv(OUT / "signed_pair_quality_rows.csv", pair_rows)
    _write_csv(OUT / "scene_quality_rows.csv", scene_rows)
    _write_csv(OUT / "failure_case_rows.csv", failure_rows)
    _write_json(OUT / "summary.json", summary)
    outputs = [
        OUT / "carrier_gt_profile_rows.csv",
        OUT / "cluster_gt_quality_rows.csv",
        OUT / "mask_readout_quality_rows.csv",
        OUT / "signed_pair_quality_rows.csv",
        OUT / "scene_quality_rows.csv",
        OUT / "failure_case_rows.csv",
        OUT / "summary.json",
    ]
    _write_json(OUT / "SHA256SUMS.json", {adaptive._rel(path): adaptive._sha256(path) for path in outputs if path.exists()})
    print(json.dumps(adaptive._jsonable(summary), indent=2, sort_keys=True), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose v91 affinity/semantic feature quality before readout repair.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
