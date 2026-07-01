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

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _float(value: Any, default: float = 0.0) -> float:
    if value is None or str(value).strip() == "":
        return float(default)
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return float(out) if math.isfinite(out) else float(default)


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _mean(values: list[float | None]) -> float | None:
    valid = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.mean(valid)) if valid else None


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _auc(scores: list[float], labels: list[bool]) -> float | None:
    if not scores or len(scores) != len(labels):
        return None
    pos = [float(score) for score, label in zip(scores, labels) if label]
    neg = [float(score) for score, label in zip(scores, labels) if not label]
    if not pos or not neg:
        return None
    wins = 0.0
    total = 0.0
    for p in pos:
        for n in neg:
            total += 1.0
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return float(wins / max(1.0, total))


def _area_bin(area: float) -> str:
    if area < 0.005:
        return "a00_000_005"
    if area < 0.02:
        return "a01_005_020"
    if area < 0.05:
        return "a02_020_050"
    if area < 0.10:
        return "a03_050_100"
    if area < 0.30:
        return "a04_100_300"
    return "a05_300_plus"


def _load_verification_scores(path: Path) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    if not path.exists():
        return out
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            pid = str(row.get("proposal_id") or "")
            if not pid:
                continue
            out[pid] = {
                "d4rt_score": _float(row.get("proposal_D4RT_score"), 0.0),
                "d4rt_no_temporal_score": _float(row.get("proposal_D4RT_no_temporal_score"), 0.0),
                "d4rt_reliability": _float(row.get("proposal_D4RT_reliability_mean"), 0.0),
                "d4rt_membership_entropy": _float(row.get("proposal_D4RT_membership_entropy_mean"), 1.0),
                "d4rt_atom_coverage": _float(row.get("proposal_D4RT_atom_coverage_score"), 0.0),
            }
    return out


def _load_proposals(path: Path, verification_rows: Path, target_dense_variant: str) -> list[dict[str, Any]]:
    d4rt = _load_verification_scores(verification_rows)
    target = {"SP0_existing_masks_baseline", str(target_dense_variant)}
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            variant = str(row.get("variant") or "")
            if variant not in target:
                continue
            pid = str(row.get("proposal_id") or "")
            aux = d4rt.get(pid, {})
            area = _float(row.get("proposal_area_ratio"), 0.0)
            iou = _float(row.get("majority_iou_diagnostic"), 0.0)
            entropy = _float(row.get("semantic_entropy"), 1.0)
            broad = _bool(row.get("source_broad_large_risk"))
            under = _bool(row.get("source_underseg_proxy"))
            item = {
                "proposal_id": pid,
                "scene_id": str(row.get("scene_id") or ""),
                "chunk_id": str(row.get("chunk_id") or ""),
                "frame_id": int(float(row.get("frame_id") or 0)),
                "source_variant": variant,
                "area_ratio": area,
                "area_bin": _area_bin(area),
                "semantic_score": _float(row.get("proposal_compactness_score"), 0.0),
                "semantic_entropy": entropy,
                "boundary_score": _float(row.get("semantic_boundary_divergence"), 0.0),
                "semantic_margin": _float(row.get("semantic_prototype_margin"), 0.0),
                "background_proxy_score": _float(row.get("proposal_background_proxy_score"), 0.0),
                "background_proxy_label": _float(row.get("proposal_background_proxy_score"), 0.0) >= 0.75,
                "source_broad_large_risk": broad,
                "source_underseg_proxy": under,
                "majority_iou": iou,
                "iou50_label": iou >= 0.50,
                "background_label": iou < 0.25 or str(row.get("majority_gt_id_diagnostic") or "0") in {"", "0"},
                "d4rt_score": aux.get("d4rt_score", 0.0),
                "d4rt_no_temporal_score": aux.get("d4rt_no_temporal_score", 0.0),
                "d4rt_reliability": aux.get("d4rt_reliability", 0.0),
                "d4rt_membership_entropy": aux.get("d4rt_membership_entropy", 1.0),
                "d4rt_atom_coverage": aux.get("d4rt_atom_coverage", 0.0),
            }
            rows.append(item)
    _add_frame_local_margins(rows)
    return rows


def _add_frame_local_margins(rows: list[dict[str, Any]]) -> None:
    by_frame: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_frame[(str(row["scene_id"]), int(row["frame_id"]))].append(row)
    for subset in by_frame.values():
        values = np.asarray([float(row["semantic_score"]) for row in subset], dtype=np.float64)
        mean = float(np.mean(values)) if values.size else 0.0
        std = float(np.std(values)) if values.size else 0.0
        for row in subset:
            row["frame_local_semantic_z"] = 0.0 if std <= 1e-12 else float((float(row["semantic_score"]) - mean) / std)


def _score(row: dict[str, Any], variant: str) -> tuple[float, dict[str, float]]:
    area = float(row["area_ratio"])
    extent = math.sqrt(max(0.0, area))
    semantic = float(row["semantic_score"])
    frame_sem = float(row.get("frame_local_semantic_z", 0.0))
    boundary = float(row["boundary_score"])
    d4rt = float(row["d4rt_score"])
    d4rt_entropy = float(row["d4rt_membership_entropy"])
    entropy = float(row["semantic_entropy"])
    risk = (1.0 if row["source_broad_large_risk"] else 0.0) + (1.0 if row["source_underseg_proxy"] else 0.0)
    if variant == "OR0_area_only_baseline":
        value = area
        parts = {"area": value}
    elif variant == "OR1_semantic_compactness_only":
        value = semantic
        parts = {"semantic": semantic}
    elif variant == "OR2_D4RT_verification_only":
        value = d4rt
        parts = {"d4rt": d4rt}
    elif variant == "OR3_semantic_plus_boundary_extent":
        value = frame_sem + 0.75 * boundary + 1.5 * extent - 0.25 * entropy
        parts = {"semantic": frame_sem, "boundary": 0.75 * boundary, "extent": 1.5 * extent, "entropy_penalty": -0.25 * entropy}
    elif variant == "OR4_semantic_plus_D4RT_soft":
        value = frame_sem + 0.75 * d4rt - 0.25 * d4rt_entropy
        parts = {"semantic": frame_sem, "d4rt": 0.75 * d4rt, "d4rt_entropy_penalty": -0.25 * d4rt_entropy}
    elif variant == "OR6_background_suppressed_fusion":
        value = frame_sem + 0.50 * boundary + 0.50 * d4rt + 1.0 * extent - 0.40 * entropy - 0.25 * d4rt_entropy - 0.35 * risk
        parts = {
            "semantic": frame_sem,
            "boundary": 0.50 * boundary,
            "d4rt": 0.50 * d4rt,
            "extent": 1.0 * extent,
            "entropy_penalty": -0.40 * entropy,
            "d4rt_entropy_penalty": -0.25 * d4rt_entropy,
            "risk_penalty": -0.35 * risk,
        }
    elif variant == "OR8_area_bin_balanced_fusion_repair":
        value = frame_sem + 0.50 * boundary + 0.35 * d4rt - 0.35 * entropy - 0.20 * risk
        parts = {"semantic": frame_sem, "boundary": 0.50 * boundary, "d4rt": 0.35 * d4rt, "entropy_penalty": -0.35 * entropy, "risk_penalty": -0.20 * risk}
    elif variant == "OR9_risk_capped_area_repair":
        value = area
        parts = {"area": area, "risk_capped_top64_policy": -risk}
    else:
        value = semantic
        parts = {"semantic": semantic}
    return float(value), parts


def _ranked_rows(rows: list[dict[str, Any]], variant: str) -> list[dict[str, Any]]:
    scored = []
    for row in rows:
        score, parts = _score(row, variant)
        item = dict(row)
        item["objectness_variant"] = variant
        item["objectness_score"] = score
        item["objectness_parts_json"] = json.dumps(parts, sort_keys=True)
        scored.append(item)
    if variant == "OR8_area_bin_balanced_fusion_repair":
        return _area_bin_round_robin(scored)
    if variant == "OR9_risk_capped_area_repair":
        return _risk_capped_per_chunk(scored, top_k=64, max_risk_fraction=0.35)
    return sorted(scored, key=lambda row: float(row["objectness_score"]), reverse=True)


def _area_bin_round_robin(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_chunk_bin: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_chunk_bin[(str(row["chunk_id"]), str(row["area_bin"]))].append(row)
    for key in by_chunk_bin:
        by_chunk_bin[key] = sorted(by_chunk_bin[key], key=lambda row: float(row["objectness_score"]), reverse=True)
    out: list[dict[str, Any]] = []
    for chunk_id in sorted({str(row["chunk_id"]) for row in rows}):
        bins = sorted({bin_name for (cid, bin_name) in by_chunk_bin if cid == chunk_id})
        indexes = {bin_name: 0 for bin_name in bins}
        while True:
            added = False
            for bin_name in bins:
                subset = by_chunk_bin.get((chunk_id, bin_name), [])
                idx = indexes[bin_name]
                if idx < len(subset):
                    out.append(subset[idx])
                    indexes[bin_name] += 1
                    added = True
            if not added:
                break
    return out


def _risk_capped_per_chunk(rows: list[dict[str, Any]], *, top_k: int, max_risk_fraction: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    by_chunk: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_chunk[str(row["chunk_id"])].append(row)
    for chunk_id in sorted(by_chunk):
        subset = sorted(by_chunk[chunk_id], key=lambda row: float(row["objectness_score"]), reverse=True)
        risky = [row for row in subset if bool(row["source_broad_large_risk"]) or bool(row["source_underseg_proxy"])]
        clean = [row for row in subset if not (bool(row["source_broad_large_risk"]) or bool(row["source_underseg_proxy"]))]
        risk_cap = int(math.floor(int(top_k) * float(max_risk_fraction)))
        prefix = risky[:risk_cap] + clean[: max(0, int(top_k) - risk_cap)]
        used = {id(row) for row in prefix}
        rest = [row for row in subset if id(row) not in used]
        ordered = prefix + rest
        for local_rank, row in enumerate(ordered):
            item = dict(row)
            item["objectness_score"] = float(len(ordered) - local_rank)
            out.append(item)
    return out


def _topk_rate_per_chunk(ranked: list[dict[str, Any]], key: str, k: int) -> float | None:
    by_chunk: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ranked:
        by_chunk[str(row["chunk_id"])].append(row)
    rates = []
    for subset in by_chunk.values():
        top = subset[: min(int(k), len(subset))]
        if not top:
            continue
        rates.append(float(np.mean([1.0 if row[key] else 0.0 for row in top])))
    return _mean(rates)


def _topk_mean_numeric_per_chunk(ranked: list[dict[str, Any]], key: str, k: int) -> float | None:
    by_chunk: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ranked:
        by_chunk[str(row["chunk_id"])].append(row)
    vals = []
    for subset in by_chunk.values():
        top = subset[: min(int(k), len(subset))]
        if top:
            vals.append(float(np.mean([float(row[key]) for row in top])))
    return _mean(vals)


def _summarize_variant(rows: list[dict[str, Any]], variant: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ranked = _ranked_rows(rows, variant)
    scores = [float(row["objectness_score"]) for row in ranked]
    iou50 = [bool(row["iou50_label"]) for row in ranked]
    foreground = [not bool(row["background_label"]) for row in ranked]
    area = [float(row["area_ratio"]) for row in ranked]
    entropy = [float(row["semantic_entropy"]) for row in ranked]
    d4rt_rel = [float(row["d4rt_reliability"]) for row in ranked]
    top64 = []
    by_chunk: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ranked:
        by_chunk[str(row["chunk_id"])].append(row)
    for subset in by_chunk.values():
        top64.extend(subset[: min(64, len(subset))])
    metric = {
        "variant": variant,
        "proposal_count": len(rows),
        "chunk_count": len(by_chunk),
        "proposal_objectness_AUC_iou50_diagnostic": _auc(scores, iou50),
        "proposal_objectness_AUC_foreground_vs_background_diagnostic": _auc(scores, foreground),
        "top32_iou50_rate_per_chunk": _topk_rate_per_chunk(ranked, "iou50_label", 32),
        "top64_iou50_rate_per_chunk": _topk_rate_per_chunk(ranked, "iou50_label", 64),
        "top128_iou50_rate_per_chunk": _topk_rate_per_chunk(ranked, "iou50_label", 128),
        "selected_top64_broad_rate": _mean([1.0 if row["source_broad_large_risk"] else 0.0 for row in top64]),
        "selected_top64_underseg_rate": _mean([1.0 if row["source_underseg_proxy"] else 0.0 for row in top64]),
        "selected_top64_broad_underseg_rate": _mean([1.0 if (row["source_broad_large_risk"] or row["source_underseg_proxy"]) else 0.0 for row in top64]),
        "selected_top64_background_proxy_rate": _mean([1.0 if row["background_proxy_label"] else 0.0 for row in top64]),
        "selected_top64_background_diagnostic_rate": _mean([1.0 if row["background_label"] else 0.0 for row in top64]),
        "selected_temporal_span_mean": None,
        "D4RT_score_contribution_mean": _mean([float(row["d4rt_score"]) for row in top64]),
        "semantic_score_contribution_mean": _mean([float(row["semantic_score"]) for row in top64]),
        "score_area_correlation": _pearson(scores, area),
        "score_entropy_correlation": _pearson(scores, entropy),
        "score_d4rt_reliability_correlation": _pearson(scores, d4rt_rel),
        "uses_gt_for_prediction": False,
        "uses_gt_for_evaluation": True,
        "diagnostic_only": True,
        "forbidden_for_method_table": True,
    }
    rank_rows = []
    for rank, row in enumerate(ranked):
        out = {
            "rank": rank,
            "objectness_variant": variant,
            "proposal_id": row["proposal_id"],
            "scene_id": row["scene_id"],
            "chunk_id": row["chunk_id"],
            "frame_id": row["frame_id"],
            "source_variant": row["source_variant"],
            "objectness_score": row["objectness_score"],
            "objectness_parts_json": row["objectness_parts_json"],
            "area_ratio": row["area_ratio"],
            "area_bin": row["area_bin"],
            "semantic_entropy": row["semantic_entropy"],
            "semantic_score": row["semantic_score"],
            "d4rt_score": row["d4rt_score"],
            "d4rt_reliability": row["d4rt_reliability"],
            "majority_iou_diagnostic": row["majority_iou"],
            "iou50_label_diagnostic": row["iou50_label"],
            "background_label_diagnostic": row["background_label"],
            "background_proxy_label_non_gt": row["background_proxy_label"],
            "background_proxy_score_non_gt": row["background_proxy_score"],
            "source_broad_large_risk": row["source_broad_large_risk"],
            "source_underseg_proxy": row["source_underseg_proxy"],
            "uses_gt_for_prediction": False,
            "uses_gt_for_evaluation": True,
            "diagnostic_only": True,
            "forbidden_for_method_table": True,
        }
        rank_rows.append(out)
    return metric, rank_rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = _rooted(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    proposal_rows = _rooted(args.proposal_rows)
    verification_rows = _rooted(args.verification_rows)
    missing = []
    for name, path in [("proposal_rows", proposal_rows), ("verification_rows", verification_rows)]:
        if not path.exists():
            missing.append({"name": name, "path": _rel(path)})
    if missing:
        _write_csv(output_root / "missing_input_rows.csv", missing)
        summary = {"phase": "v72_phase4_objectness_ranking", "decision": "FAIL_MISSING_INPUTS", "missing_input_count": len(missing), "gate": {"pass": False}}
        _write_json(output_root / "objectness_summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return summary
    rows = _load_proposals(proposal_rows, verification_rows, str(args.target_dense_variant))
    variants = [
        "OR0_area_only_baseline",
        "OR1_semantic_compactness_only",
        "OR2_D4RT_verification_only",
        "OR3_semantic_plus_boundary_extent",
        "OR4_semantic_plus_D4RT_soft",
        "OR6_background_suppressed_fusion",
        "OR8_area_bin_balanced_fusion_repair",
        "OR9_risk_capped_area_repair",
    ]
    metric_rows = []
    rank_rows_all = []
    for variant in variants:
        metric, rank_rows = _summarize_variant(rows, variant)
        metric_rows.append(metric)
        rank_rows_all.extend(rank_rows)
    area_row = next(row for row in metric_rows if row["variant"] == "OR0_area_only_baseline")
    semantic_row = next(row for row in metric_rows if row["variant"] == "OR1_semantic_compactness_only")
    d4rt_row = next(row for row in metric_rows if row["variant"] == "OR2_D4RT_verification_only")
    fusion_rows = [row for row in metric_rows if row["variant"] not in {"OR0_area_only_baseline", "OR1_semantic_compactness_only", "OR2_D4RT_verification_only"}]
    deployable_rows = [row for row in metric_rows if row["variant"] != "OR0_area_only_baseline"]
    best_top64 = max(metric_rows, key=lambda row: _float(row.get("top64_iou50_rate_per_chunk"), -1.0), default={})
    best_fusion = max(fusion_rows, key=lambda row: _float(row.get("proposal_objectness_AUC_iou50_diagnostic"), -1.0), default={})
    best_auc_single = max(_float(semantic_row.get("proposal_objectness_AUC_iou50_diagnostic")), _float(d4rt_row.get("proposal_objectness_AUC_iou50_diagnostic")))
    fusion_auc_gain = _float(best_fusion.get("proposal_objectness_AUC_iou50_diagnostic")) - best_auc_single
    area_background = _float(area_row.get("selected_top64_background_proxy_rate"))

    def _row_background_drop(row: dict[str, Any]) -> float:
        return area_background - _float(row.get("selected_top64_background_proxy_rate"))

    def _row_passes_objectness_gate(row: dict[str, Any]) -> bool:
        return (
            _float(row.get("top64_iou50_rate_per_chunk")) >= 0.50
            and _float(row.get("proposal_objectness_AUC_iou50_diagnostic")) >= 0.75
            and _row_background_drop(row) >= 0.10
            and _float(row.get("selected_top64_broad_underseg_rate"), 1.0) <= 0.35
        )

    gate_candidates = [row for row in deployable_rows if _row_passes_objectness_gate(row)]
    if gate_candidates:
        best_non_oracle = max(
            gate_candidates,
            key=lambda row: (
                _float(row.get("top64_iou50_rate_per_chunk"), -1.0),
                _float(row.get("proposal_objectness_AUC_iou50_diagnostic"), -1.0),
            ),
        )
    else:
        best_non_oracle = max(
            deployable_rows,
            key=lambda row: (
                _float(row.get("top64_iou50_rate_per_chunk"), -1.0),
                _row_background_drop(row),
                -_float(row.get("selected_top64_broad_underseg_rate"), 1.0),
            ),
            default={},
        )
    background_drop = _row_background_drop(best_non_oracle)
    phase4_pass = (
        _float(best_non_oracle.get("top64_iou50_rate_per_chunk")) >= 0.50
        and _float(best_non_oracle.get("proposal_objectness_AUC_iou50_diagnostic")) >= 0.75
        and background_drop >= 0.10
        and _float(best_non_oracle.get("selected_top64_broad_underseg_rate"), 1.0) <= 0.35
        and fusion_auc_gain >= 0.03
    )
    summary = {
        "phase": "v72_phase4_objectness_ranking",
        "decision": "PASS_V72_PHASE4_OBJECTNESS_RANKING" if phase4_pass else "NO_GO_PHASE4_OBJECTNESS_RANKING",
        "proposal_rows": _rel(proposal_rows),
        "verification_rows": _rel(verification_rows),
        "target_dense_variant": str(args.target_dense_variant),
        "proposal_count": len(rows),
        "best_top64_variant": best_top64.get("variant"),
        "best_top64_rate": best_top64.get("top64_iou50_rate_per_chunk"),
        "best_gate_candidate_variant": best_non_oracle.get("variant") if gate_candidates else None,
        "gate_candidate_count": len(gate_candidates),
        "best_non_oracle_variant": best_non_oracle.get("variant"),
        "best_non_oracle_top64_iou50_rate": best_non_oracle.get("top64_iou50_rate_per_chunk"),
        "best_non_oracle_AUC_iou50": best_non_oracle.get("proposal_objectness_AUC_iou50_diagnostic"),
        "best_non_oracle_background_proxy_rate": best_non_oracle.get("selected_top64_background_proxy_rate"),
        "area_only_background_proxy_rate": area_row.get("selected_top64_background_proxy_rate"),
        "best_non_oracle_background_drop_vs_area": background_drop,
        "best_non_oracle_broad_underseg_rate": best_non_oracle.get("selected_top64_broad_underseg_rate"),
        "best_fusion_variant": best_fusion.get("variant"),
        "fusion_objectness_AUC_gain_vs_best_semantic_or_D4RT": fusion_auc_gain,
        "gate": {
            "best_non_oracle_top64_iou50_rate_ge_0p50": _float(best_non_oracle.get("top64_iou50_rate_per_chunk")) >= 0.50,
            "best_non_oracle_AUC_iou50_ge_0p75": _float(best_non_oracle.get("proposal_objectness_AUC_iou50_diagnostic")) >= 0.75,
            "best_non_oracle_background_proxy_rate_le_area_minus_0p10": background_drop >= 0.10,
            "best_non_oracle_broad_underseg_rate_le_0p35": _float(best_non_oracle.get("selected_top64_broad_underseg_rate"), 1.0) <= 0.35,
            "fusion_AUC_ge_best_single_plus_0p03": fusion_auc_gain >= 0.03,
            "uses_gt_for_prediction_false": True,
            "pass": phase4_pass,
        },
        "method_boundary": {
            "uses_gt_for_prediction": False,
            "uses_gt_for_evaluation": True,
            "diagnostic_only": True,
            "forbidden_for_method_table": True,
            "score_variants_are_rule_based_no_training": True,
        },
    }
    _write_csv(output_root / "objectness_rank_rows.csv", rank_rows_all)
    _write_csv(output_root / "objectness_metric_rows.csv", metric_rows)
    _write_csv(output_root / "variant_summary_rows.csv", metric_rows)
    _write_csv(output_root / "missing_input_rows.csv", missing)
    _write_json(output_root / "objectness_summary.json", summary)
    sha_rows = []
    for path in sorted(output_root.glob("*")):
        if path.is_file() and path.name != "sha256_rows.csv":
            sha_rows.append({"path": _rel(path), "bytes": int(path.stat().st_size), "sha256": _sha256(path), "kind": "output"})
    for path in [proposal_rows, verification_rows]:
        if path.exists():
            sha_rows.append({"path": _rel(path), "bytes": int(path.stat().st_size), "sha256": _sha256(path), "kind": "input"})
    _write_csv(output_root / "sha256_rows.csv", sha_rows)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="v72 Phase4 proposal objectness ranking diagnostic.")
    parser.add_argument("--proposal-rows", default="outputs/audit/v72_phase2_dense_token_proposals_smoke10_area_bin1/proposal_rows.csv")
    parser.add_argument("--verification-rows", default="outputs/audit/v72_phase3_d4rt_proposal_verification_area_bin1/proposal_verification_rows.csv")
    parser.add_argument("--output-root", default="outputs/audit/v72_phase4_objectness_ranking")
    parser.add_argument("--target-dense-variant", default="SP2_DINO_affinity_connected_components")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
