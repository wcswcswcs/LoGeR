from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stream4d.scannet_stream import ScanNetStream  # noqa: E402
from stream4d_native.v71_representative_setcover import _diagnostic_mask_stats, _load_pipeline_roots  # noqa: E402
from tools.run_v66_local_chunk_eval import _chunk_rows, _frame_data  # noqa: E402
from tools.run_v66_scene_mv_ap_probe5 import DEFAULT_SCENES, _mask_dir_from_pipeline, _parse_csv_list  # noqa: E402


def _rooted(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _float(value: Any, default: float = 0.0) -> float:
    if value is None or str(value).strip() == "":
        return float(default)
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return float(out) if math.isfinite(out) else float(default)


def _int(value: Any, default: int = 0) -> int:
    try:
        if value is None or str(value).strip() == "":
            return int(default)
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _mean(values: list[float | None]) -> float | None:
    valid = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(sum(valid) / len(valid)) if valid else None


def _p10(values: list[float]) -> float | None:
    if not values:
        return None
    vals = sorted(values)
    idx = max(0, min(len(vals) - 1, int(round((len(vals) - 1) * 0.10))))
    return float(vals[idx])


def _auc(pairs: list[tuple[float, bool]]) -> float | None:
    clean = [(float(score), bool(label)) for score, label in pairs if math.isfinite(float(score))]
    pos = sum(1 for _, label in clean if label)
    neg = len(clean) - pos
    if pos == 0 or neg == 0:
        return None
    clean.sort(key=lambda item: item[0])
    rank_sum_pos = 0.0
    i = 0
    while i < len(clean):
        j = i + 1
        while j < len(clean) and clean[j][0] == clean[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        rank_sum_pos += avg_rank * sum(1 for _, label in clean[i:j] if label)
        i = j
    return float((rank_sum_pos - pos * (pos + 1) / 2.0) / (pos * neg))


def _area_objectness(area: float) -> float:
    if area <= 0.0:
        return -1.0
    center = 0.055
    return float(max(0.0, 1.0 - abs(math.log(max(area, 1e-6) / center)) / math.log(8.0)))


def _semantic_scores(row: dict[str, Any]) -> dict[str, float]:
    area = _float(row.get("area_ratio"))
    entropy = _float(row.get("semantic_entropy"), 1.0)
    intra = _float(row.get("semantic_intra_variance"), 0.0)
    boundary = _float(row.get("semantic_boundary_variance"), 0.0)
    margin = _float(row.get("semantic_prototype_margin"), 0.0)
    overlap = _float(row.get("same_frame_overlap_count")) + _float(row.get("same_frame_competing_mask_count"))
    broad = 1.0 if _bool(row.get("broad_large_risk")) else 0.0
    underseg = 1.0 if _bool(row.get("underseg_proxy")) else 0.0
    return {
        "AREA0_area_only_control": math.log1p(100.0 * area),
        "SEM4_entropy_only_cov_trace": entropy,
        "SEM0_margin_low_entropy": 2.5 * margin + 0.8 * _area_objectness(area) - 0.8 * entropy - 0.04 * overlap - 0.7 * broad - 0.5 * underseg,
        "SEM1_entropy_inverse_compact": -1.0 * entropy - 80.0 * intra + 0.5 * _area_objectness(area) - 0.5 * broad - 0.4 * underseg,
        "SEM2_boundary_interior_divergence": 2.0 * margin + 50.0 * boundary - 0.5 * entropy + 0.4 * _area_objectness(area) - 0.05 * overlap,
        "SEM3_broad_decomposition_potential": entropy + 2.0 * margin + 0.5 * broad + 0.3 * underseg - 0.03 * overlap,
    }


def _d4rt_score(row: dict[str, Any]) -> float:
    reliability = _float(row.get("D4RT_carrier_reliability_mean"), 0.0)
    confidence = _float(row.get("D4RT_carrier_confidence_mean"), 0.0)
    carrier_count = math.log1p(_float(row.get("D4RT_visible_carrier_count"), 0.0))
    traj_entropy = _float(row.get("D4RT_carrier_trajectory_entropy"), 0.0)
    return float(0.75 * reliability + 0.20 * confidence + 0.05 * carrier_count - 0.20 * traj_entropy)


def _load_feature_rows(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return out
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            out[str(row.get("mask_observation_id") or "")] = row
    return out


def _load_candidate_rows(path: Path, scenes: set[str]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            scene = str(row.get("scene_id") or "")
            if scenes and scene not in scenes:
                continue
            row["frame_id"] = _int(row.get("frame_id"), -1)
            row["mask_id"] = _int(row.get("mask_id"), -1)
            row["area_ratio"] = _float(row.get("area_ratio"), 0.0)
            row["semantic_entropy"] = _float(row.get("semantic_entropy"), 1.0)
            row["semantic_prototype_margin"] = _float(row.get("semantic_prototype_margin"), 0.0)
            row["semantic_intra_variance"] = _float(row.get("semantic_intra_mask_variance"), 0.0)
            row["same_frame_overlap_count"] = _float(row.get("same_frame_overlap_count"), 0.0)
            row["same_frame_competing_mask_count"] = _float(row.get("same_frame_competing_mask_count"), 0.0)
            row["broad_large_risk"] = _bool(row.get("broad_background_risk")) or _bool(row.get("large_mask_risk")) or _float(row.get("area_ratio"), 0.0) >= 0.30
            row["underseg_proxy"] = _float(row.get("underseg_proxy_score"), 0.0) >= 0.75
            out[str(row.get("chunk_id") or "")].append(row)
    return out


def _topk_rate(rows: list[dict[str, Any]], score_key: str, k: int, label_key: str = "diagnostic_iou50") -> float | None:
    if not rows:
        return None
    top = sorted(rows, key=lambda row: _float(row.get(score_key), float("-inf")), reverse=True)[: min(k, len(rows))]
    return _mean([1.0 if _bool(row.get(label_key)) else 0.0 for row in top])


def _topk_rate_per_chunk(rows: list[dict[str, Any]], score_key: str, k: int, label_key: str = "diagnostic_iou50") -> float | None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("chunk_id") or "")].append(row)
    vals = [_topk_rate(subset, score_key, k, label_key) for subset in grouped.values() if subset]
    return _mean(vals)


def _rank_rows(rows: list[dict[str, Any]], score_key: str, rank_key: str) -> None:
    for rank, row in enumerate(sorted(rows, key=lambda item: _float(item.get(score_key), float("-inf")), reverse=True)):
        row[rank_key] = rank


def _summarize_variant(rows: list[dict[str, Any]], score_key: str, variant: str) -> dict[str, Any]:
    pairs = [(float(row[score_key]), _bool(row.get("diagnostic_iou50"))) for row in rows]
    bg_pairs = [(float(row[score_key]), _bool(row.get("diagnostic_background_fp"))) for row in rows]
    return {
        "variant": variant,
        "candidate_count": len(rows),
        "AUC_iou50": _auc(pairs),
        "AUC_background_fp": _auc(bg_pairs),
        "top64_iou50_rate": _topk_rate_per_chunk(rows, score_key, 64),
        "top128_iou50_rate": _topk_rate_per_chunk(rows, score_key, 128),
        "top64_background_fp_rate": _topk_rate_per_chunk(rows, score_key, 64, "diagnostic_background_fp"),
        "topk_scope": "per_chunk_mean",
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = _rooted(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    scenes = _parse_csv_list(args.scenes)
    candidate_rows_by_chunk = _load_candidate_rows(_rooted(args.candidate_rows), set(scenes))
    feature_rows = _load_feature_rows(_rooted(args.mask_feature_rows))
    pipeline_roots = _load_pipeline_roots(_rooted(args.witness_summary), scenes)
    semantic_summary = _load_json(_rooted(args.semantic_summary))
    d4rt_summary = _load_json(_rooted(args.d4rt_summary))

    semantic_rows: list[dict[str, Any]] = []
    fusion_rows: list[dict[str, Any]] = []
    chunk_count = 0
    for scene in scenes:
        pipeline_root = pipeline_roots.get(scene)
        if pipeline_root is None:
            continue
        mask_dir = _mask_dir_from_pipeline(pipeline_root)
        stream = ScanNetStream(scene, root=ROOT / "data/scannet/processed")
        all_frames = stream.frame_ids(stride=int(args.stride), max_frames=None)
        for chunk in _chunk_rows(pipeline_root, scene):
            if int(args.max_chunks) > 0 and chunk_count >= int(args.max_chunks):
                break
            chunk_id = str(chunk.get("chunk_id"))
            candidates = candidate_rows_by_chunk.get(chunk_id, [])
            if not candidates:
                continue
            chunk_count += 1
            print(f"[v72-signal] chunk {chunk_count}: {chunk_id}", file=sys.stderr, flush=True)
            raw_start = int(float(chunk.get("raw_frame_start") or 0))
            raw_end = int(float(chunk.get("raw_frame_end") or raw_start))
            frame_ids = [int(frame) for frame in all_frames if raw_start <= int(frame) <= raw_end]
            if not frame_ids:
                continue
            frame_data = _frame_data(scene, frame_ids, mask_dir)
            pairs = {(int(row["frame_id"]), int(row["mask_id"])) for row in candidates}
            diagnostic = _diagnostic_mask_stats(frame_data, pairs)
            chunk_rows: list[dict[str, Any]] = []
            for cand in candidates:
                obs = str(cand.get("mask_observation_id") or "")
                feat = feature_rows.get(obs, {})
                if feat:
                    cand["semantic_boundary_variance"] = _float(feat.get("semantic_boundary_variance"), 0.0)
                    cand["semantic_background_score_proxy"] = _bool(feat.get("semantic_background_score_proxy"))
                else:
                    cand["semantic_boundary_variance"] = 0.0
                    cand["semantic_background_score_proxy"] = False
                stats = diagnostic.get((int(cand["frame_id"]), int(cand["mask_id"])), {})
                scores = _semantic_scores(cand)
                d4rt_score = _d4rt_score(cand)
                best_semantic_score = max(value for key, value in scores.items() if key.startswith("SEM"))
                fusion_score = best_semantic_score + 0.35 * d4rt_score
                majority_iou = _float(stats.get("majority_iou"), 0.0)
                out = {
                    "scene_id": cand.get("scene_id"),
                    "chunk_id": chunk_id,
                    "frame_id": cand.get("frame_id"),
                    "mask_id": cand.get("mask_id"),
                    "mask_observation_id": obs,
                    "semantic_backend": cand.get("semantic_backend"),
                    "DINO_feature_available": cand.get("DINO_feature_available"),
                    "RADIO_feature_available": cand.get("RADIO_feature_available"),
                    "semantic_entropy": cand.get("semantic_entropy"),
                    "semantic_intra_variance": cand.get("semantic_intra_variance"),
                    "semantic_boundary_variance": cand.get("semantic_boundary_variance"),
                    "semantic_prototype_id": cand.get("semantic_prototype_id"),
                    "semantic_prototype_margin": cand.get("semantic_prototype_margin"),
                    "area_ratio": cand.get("area_ratio"),
                    "broad_large_risk": cand.get("broad_large_risk"),
                    "underseg_proxy": cand.get("underseg_proxy"),
                    "same_frame_overlap_count": cand.get("same_frame_overlap_count"),
                    "same_frame_competing_mask_count": cand.get("same_frame_competing_mask_count"),
                    "majority_iou_diagnostic": majority_iou,
                    "majority_gt_id_diagnostic": stats.get("majority_gt"),
                    "majority_purity_diagnostic": stats.get("majority_purity"),
                    "diagnostic_iou50": majority_iou >= 0.50,
                    "diagnostic_background_fp": majority_iou < 0.25 and (_bool(cand.get("broad_large_risk")) or _bool(cand.get("semantic_background_score_proxy"))),
                    "semantic_score_best": best_semantic_score,
                    "d4rt_score": d4rt_score,
                    "fusion_score": fusion_score,
                    "uses_gt_for_prediction": False,
                    "diagnostic_only": False,
                    "forbidden_for_method_table": False,
                }
                out.update(scores)
                semantic_rows.append(out)
                chunk_rows.append(out)
            for score_key, rank_key in [
                ("semantic_score_best", "semantic_rank"),
                ("d4rt_score", "D4RT_rank"),
                ("fusion_score", "fusion_rank"),
            ]:
                _rank_rows(chunk_rows, score_key, rank_key)
            fusion_rows.extend(chunk_rows)
        if int(args.max_chunks) > 0 and chunk_count >= int(args.max_chunks):
            break

    semantic_variants = [
        "SEM0_margin_low_entropy",
        "SEM1_entropy_inverse_compact",
        "SEM2_boundary_interior_divergence",
        "SEM3_broad_decomposition_potential",
        "SEM4_entropy_only_cov_trace",
        "semantic_score_best",
    ]
    control_variants = ["AREA0_area_only_control"]
    metric_rows: list[dict[str, Any]] = []
    semantic_variant_rows = [_summarize_variant(semantic_rows, key, key) for key in semantic_variants]
    control_variant_rows = [_summarize_variant(semantic_rows, key, key) for key in control_variants]
    d4rt_variant = _summarize_variant(fusion_rows, "d4rt_score", "D4RT_candidate_score")
    fusion_variant = _summarize_variant(fusion_rows, "fusion_score", "fusion_semantic_plus_D4RT")
    metric_rows.extend(semantic_variant_rows)
    metric_rows.extend(control_variant_rows)
    metric_rows.extend([d4rt_variant, fusion_variant])

    clean_entropy = _mean([_float(row.get("semantic_entropy")) for row in semantic_rows if not _bool(row.get("broad_large_risk")) and not _bool(row.get("underseg_proxy"))])
    broad_entropy = _mean([_float(row.get("semantic_entropy")) for row in semantic_rows if _bool(row.get("broad_large_risk")) or _bool(row.get("underseg_proxy"))])
    best_semantic = max(semantic_variant_rows, key=lambda row: _float(row.get("AUC_iou50"), -1.0), default={})

    d4rt_rows, d4rt_chunk_rows = _d4rt_signal_rows(_rooted(args.d4rt_atom_rows), set(scenes), int(args.max_chunks))
    d4rt_reliability_values = [_float(row.get("non_gt_reliability_score")) for row in d4rt_rows]
    d4rt_semantic_success = _mean([1.0 if _bool(row.get("semantic_feature_available")) else 0.0 for row in d4rt_rows])

    semantic_metrics = semantic_summary.get("key_metrics") or {}
    d4rt_metrics = d4rt_summary.get("key_metrics") or {}
    feature_success = semantic_metrics.get("semantic_feature_success_rate")
    radio_unavailable = bool(semantic_metrics.get("RADIO_unavailable"))
    semantic_pass = (
        _float(feature_success, 0.0) >= 0.95
        and _float(best_semantic.get("AUC_iou50"), 0.0) >= 0.70
        and (broad_entropy is not None and clean_entropy is not None and broad_entropy - clean_entropy >= 0.10)
        and _float(best_semantic.get("top64_iou50_rate"), 0.0) >= 0.50
    )
    d4rt_pass = (
        _float(d4rt_metrics.get("atom_count_per_chunk_mean"), 0.0) >= 5.0 * _float(d4rt_metrics.get("diagnostic_GT_count_per_chunk_mean"), 0.0)
        and _float(d4rt_metrics.get("atom_reliability_mean"), 0.0) >= 0.30
        and _float(d4rt_semantic_success, 0.0) >= 0.90
    )
    fusion_pass = (
        _float(fusion_variant.get("top64_iou50_rate"), 0.0)
        >= max(_float(best_semantic.get("top64_iou50_rate"), 0.0), _float(d4rt_variant.get("top64_iou50_rate"), 0.0)) + 0.05
    )
    can_enter_phase2_semantic_proposals = bool(semantic_pass)
    if semantic_pass and d4rt_pass and fusion_pass:
        decision = "PASS_V72_PHASE1_SIGNAL_ADEQUACY"
    elif semantic_pass:
        decision = "PARTIAL_GO_PHASE1_SEMANTIC_SIGNAL_D4RT_FUSION_NOT_PROVEN"
    else:
        decision = "NO_GO_PHASE1_SIGNAL_ADEQUACY"
    summary = {
        "phase": "v72_phase1_signal_adequacy",
        "decision": decision,
        "processed_chunk_count": chunk_count,
        "semantic_candidate_count": len(semantic_rows),
        "d4rt_atom_row_count": len(d4rt_rows),
        "RADIO_unavailable": radio_unavailable,
        "key_metrics": {
            "semantic_feature_success_rate": feature_success,
            "best_semantic_variant": best_semantic.get("variant"),
            "best_semantic_AUC_iou50": best_semantic.get("AUC_iou50"),
            "best_semantic_top64_iou50_rate": best_semantic.get("top64_iou50_rate"),
            "best_semantic_top128_iou50_rate": best_semantic.get("top128_iou50_rate"),
            "broad_mask_semantic_entropy_mean": broad_entropy,
            "clean_mask_semantic_entropy_mean": clean_entropy,
            "broad_minus_clean_semantic_entropy": None if broad_entropy is None or clean_entropy is None else broad_entropy - clean_entropy,
            "D4RT_candidate_top64_iou50_rate": d4rt_variant.get("top64_iou50_rate"),
            "fusion_top64_iou50_rate": fusion_variant.get("top64_iou50_rate"),
            "area_only_control_AUC_iou50": control_variant_rows[0].get("AUC_iou50") if control_variant_rows else None,
            "area_only_control_top64_iou50_rate": control_variant_rows[0].get("top64_iou50_rate") if control_variant_rows else None,
            "fusion_minus_best_single_top64_iou50_rate": _float(fusion_variant.get("top64_iou50_rate"), 0.0)
            - max(_float(best_semantic.get("top64_iou50_rate"), 0.0), _float(d4rt_variant.get("top64_iou50_rate"), 0.0)),
            "atom_count_per_chunk_mean_upstream": d4rt_metrics.get("atom_count_per_chunk_mean"),
            "diagnostic_GT_count_per_chunk_mean_upstream": d4rt_metrics.get("diagnostic_GT_count_per_chunk_mean"),
            "atom_reliability_mean_upstream": d4rt_metrics.get("atom_reliability_mean"),
            "atom_reliability_mean_rows": _mean(d4rt_reliability_values),
            "atom_reliability_p10_rows": _p10(d4rt_reliability_values),
            "atom_semantic_feature_success_rate_rows": d4rt_semantic_success,
            "real_minus_shuffled_inside_consistency": None,
            "real_minus_no_temporal_inside_consistency": None,
        },
        "gate": {
            "semantic_pass": semantic_pass,
            "d4rt_pass": d4rt_pass,
            "fusion_pass": fusion_pass,
            "can_enter_phase2_semantic_proposals": can_enter_phase2_semantic_proposals,
            "pass": decision == "PASS_V72_PHASE1_SIGNAL_ADEQUACY",
            "real_vs_shuffled_deferred_to_phase3": True,
        },
        "notes": [
            "Scores use only candidate, semantic feature, and D4RT fields. GT appears only in diagnostic labels after ranking.",
            "RADIO is not required when unavailable; DINO-only variants are evaluated.",
            "real-minus-shuffled D4RT controls are recorded as deferred to Phase3 proposal verification, per plan fallback.",
        ],
    }

    _write_csv(output_root / "semantic_signal_rows.csv", semantic_rows)
    _write_csv(output_root / "d4rt_signal_rows.csv", d4rt_rows)
    _write_csv(output_root / "d4rt_chunk_signal_rows.csv", d4rt_chunk_rows)
    _write_csv(output_root / "fusion_signal_rows.csv", fusion_rows)
    _write_csv(output_root / "metric_rows.csv", metric_rows)
    _write_json(output_root / "signal_adequacy_summary.json", summary)
    _write_sha_rows(output_root, [
        _rooted(args.candidate_rows),
        _rooted(args.mask_feature_rows),
        _rooted(args.d4rt_atom_rows),
        _rooted(args.key_atom_rows),
        _rooted(args.semantic_summary),
        _rooted(args.d4rt_summary),
    ])
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def _d4rt_signal_rows(path: Path, scenes: set[str], max_chunks: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    chunks_seen: set[str] = set()
    chunk_values: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if not path.exists():
        return rows, []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            scene = str(row.get("scene_id") or "")
            if scenes and scene not in scenes:
                continue
            chunk_id = str(row.get("chunk_id") or "")
            if chunk_id not in chunks_seen:
                if max_chunks > 0 and len(chunks_seen) >= max_chunks:
                    continue
                chunks_seen.add(chunk_id)
            out = {
                "scene_id": scene,
                "chunk_id": chunk_id,
                "atom_id": row.get("atom_id"),
                "atom_type": row.get("atom_type"),
                "carrier_count": row.get("cluster_size") or 1,
                "visible_frame_count": _float(row.get("visible_frame_count")),
                "frame_span": _float(row.get("frame_span")),
                "confidence_mean": _float(row.get("confidence_mean")),
                "valid_uv_ratio": _float(row.get("uv_valid_count")) / max(1.0, _float(row.get("visible_frame_count"))),
                "visibility_mean": _float(row.get("visibility_mean")),
                "non_gt_reliability_score": _float(row.get("non_gt_reliability_score")),
                "mask_membership_entropy": _float(row.get("mask_membership_entropy")),
                "semantic_available_ratio": 1.0 if _bool(row.get("semantic_feature_available")) else 0.0,
                "semantic_feature_available": row.get("semantic_feature_available"),
                "semantic_prototype_id_mode": row.get("semantic_prototype_id"),
                "trajectory_smoothness_uv": _float(row.get("D4RT_temporal_smoothness")),
                "observed_mask_switch_count": max(0.0, _float(row.get("mask_membership_count")) - 1.0),
                "D4RT_3D_position_available": False,
                "uv_projection_only": True,
                "uses_gt_for_prediction": False,
                "diagnostic_only": False,
                "forbidden_for_method_table": False,
            }
            rows.append(out)
            chunk_values[chunk_id].append(out)
    chunk_rows = []
    for chunk_id, subset in sorted(chunk_values.items()):
        chunk_rows.append(
            {
                "chunk_id": chunk_id,
                "atom_count": len(subset),
                "visible_frame_count_mean": _mean([_float(row.get("visible_frame_count")) for row in subset]),
                "frame_span_mean": _mean([_float(row.get("frame_span")) for row in subset]),
                "confidence_mean": _mean([_float(row.get("confidence_mean")) for row in subset]),
                "visibility_mean": _mean([_float(row.get("visibility_mean")) for row in subset]),
                "non_gt_reliability_mean": _mean([_float(row.get("non_gt_reliability_score")) for row in subset]),
                "mask_membership_entropy_mean": _mean([_float(row.get("mask_membership_entropy")) for row in subset]),
                "semantic_available_ratio": _mean([_float(row.get("semantic_available_ratio")) for row in subset]),
            }
        )
    return rows, chunk_rows


def _write_sha_rows(output_root: Path, input_paths: list[Path]) -> None:
    rows: list[dict[str, Any]] = []
    for path in input_paths:
        if path.exists():
            rows.append({"path": _rel(path), "bytes": path.stat().st_size, "sha256": _sha256(path), "kind": "input"})
    for path in sorted(output_root.glob("*")):
        if path.is_file() and path.name != "sha256_rows.csv":
            rows.append({"path": _rel(path), "bytes": path.stat().st_size, "sha256": _sha256(path), "kind": "output"})
    _write_csv(output_root / "sha256_rows.csv", rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="v72 Phase1 geometry/semantic signal adequacy audit.")
    parser.add_argument("--scenes", default=",".join(DEFAULT_SCENES))
    parser.add_argument("--candidate-rows", default="outputs/audit/v71_candidate_bank/candidate_mask_rows.csv")
    parser.add_argument("--mask-feature-rows", default="outputs/audit/v71_semantic_features/mask_feature_rows.csv")
    parser.add_argument("--d4rt-atom-rows", default="outputs/audit/v71_d4rt_atoms/atom_rows.csv")
    parser.add_argument("--key-atom-rows", default="outputs/audit/v71_key_atoms/key_atom_rows.csv")
    parser.add_argument("--semantic-summary", default="outputs/audit/v71_semantic_features/semantic_summary.json")
    parser.add_argument("--d4rt-summary", default="outputs/audit/v71_d4rt_atoms/atom_summary.json")
    parser.add_argument("--witness-summary", default="outputs/audit/v70_carrier_witness/witness_summary.json")
    parser.add_argument("--output-root", default="outputs/audit/v72_phase1_signal_adequacy")
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--max-chunks", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
