from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stream4d.scannet_stream import ScanNetStream  # noqa: E402
from stream4d_native.v71_representative_setcover import (  # noqa: E402
    CandidateMask,
    _diagnostic_mask_stats,
    _load_candidates,
    _load_pipeline_roots,
    _mean,
    _oracle_eval_for_selected,
    _rel,
)
from tools.run_v66_local_chunk_eval import _chunk_rows, _frame_data  # noqa: E402
from tools.run_v66_scene_mv_ap_probe5 import DEFAULT_SCENES, _mask_dir_from_pipeline, _parse_csv_list  # noqa: E402


@dataclass(frozen=True)
class DiverseConfig:
    name: str
    allow_broad: bool
    allow_underseg: bool
    min_area: float
    max_area: float
    max_per_frame: int
    max_per_proto: int
    base_objectness: str
    frame_bonus: float
    proto_bonus: float
    area_bin_bonus: float
    risk_penalty: float
    overlap_penalty: float


CONFIGS = [
    DiverseConfig("ODR0_clean_objectness_diverse", False, False, 0.006, 0.24, 6, 10, "clean", 0.80, 0.70, 0.35, 1.5, 0.06),
    DiverseConfig("ODR1_clean_strong_diversity", False, False, 0.006, 0.24, 4, 6, "clean", 1.10, 1.00, 0.45, 1.5, 0.08),
    DiverseConfig("ODR2_allow_underseg_diverse", False, True, 0.004, 0.28, 6, 10, "clean", 0.90, 0.80, 0.40, 1.0, 0.05),
    DiverseConfig("ODR3_allow_large_risky_diverse", True, True, 0.004, 0.40, 8, 14, "risky", 0.80, 0.70, 0.35, 0.8, 0.04),
]


def _rooted(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _area_objectness(area: float) -> float:
    if area <= 0.0:
        return -1.0
    center = 0.055
    log_dist = abs(math.log(max(1e-6, area) / center))
    return float(max(0.0, 1.0 - log_dist / math.log(8.0)))


def _area_bin(area: float) -> str:
    if area < 0.01:
        return "xs"
    if area < 0.03:
        return "s"
    if area < 0.08:
        return "m"
    if area < 0.18:
        return "l"
    return "xl"


def _eligible(cand: CandidateMask, cfg: DiverseConfig) -> bool:
    if cand.area_ratio < cfg.min_area or cand.area_ratio > cfg.max_area:
        return False
    if cand.broad_large_risk and not cfg.allow_broad:
        return False
    if cand.underseg_proxy and not cfg.allow_underseg:
        return False
    if cand.small_mask_risk and cand.area_ratio < cfg.min_area * 1.25:
        return False
    return True


def _base_score(cand: CandidateMask, cfg: DiverseConfig) -> float:
    rel = cand.d4rt_reliability if cand.d4rt_reliability is not None else 0.0
    overlap = cand.same_frame_overlap_count + cand.same_frame_competing_mask_count
    if cfg.base_objectness == "risky":
        score = (
            0.85 * _area_objectness(cand.area_ratio)
            + 0.55 * cand.semantic_entropy
            + 1.45 * cand.semantic_prototype_margin
            + 0.35 * rel
        )
    else:
        score = (
            1.00 * _area_objectness(cand.area_ratio)
            + 0.30 * cand.semantic_entropy
            + 2.20 * cand.semantic_prototype_margin
            + 0.15 * rel
        )
    score -= cfg.overlap_penalty * overlap
    score -= cfg.risk_penalty if cand.broad_large_risk else 0.0
    score -= cfg.risk_penalty if cand.underseg_proxy else 0.0
    return float(score)


def _diverse_select(candidates: list[CandidateMask], cfg: DiverseConfig, budget: int) -> list[dict[str, Any]]:
    pool = [cand for cand in candidates if _eligible(cand, cfg)]
    base = {cand.obs_id: _base_score(cand, cfg) for cand in pool}
    pool = [cand for cand in pool if math.isfinite(base[cand.obs_id])]
    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    frame_counts: dict[int, int] = {}
    proto_counts: dict[str, int] = {}
    area_counts: dict[str, int] = {}
    while len(selected) < budget:
        best: tuple[float, CandidateMask, dict[str, float]] | None = None
        for cand in pool:
            if cand.obs_id in used:
                continue
            if frame_counts.get(cand.frame_id, 0) >= cfg.max_per_frame:
                continue
            proto = cand.semantic_prototype_id or "none"
            if proto_counts.get(proto, 0) >= cfg.max_per_proto:
                continue
            abin = _area_bin(cand.area_ratio)
            frame_gain = cfg.frame_bonus / math.sqrt(frame_counts.get(cand.frame_id, 0) + 1.0)
            proto_gain = cfg.proto_bonus / math.sqrt(proto_counts.get(proto, 0) + 1.0)
            area_gain = cfg.area_bin_bonus / math.sqrt(area_counts.get(abin, 0) + 1.0)
            total = base[cand.obs_id] + frame_gain + proto_gain + area_gain
            detail = {
                "base_objectness_score": base[cand.obs_id],
                "frame_diversity_gain": frame_gain,
                "prototype_diversity_gain": proto_gain,
                "area_bin_diversity_gain": area_gain,
            }
            if best is None or total > best[0]:
                best = (float(total), cand, detail)
        if best is None:
            break
        total, cand, detail = best
        selected.append(
            {
                "candidate": cand,
                "rank": len(selected),
                "score_total": total,
                "score_new_atom_coverage": 0.0,
                "score_d4rt_coverage": 0.0,
                "score_semantic_coverage": 0.0,
                "new_key_atom_count": 0,
                "new_key_atom_weight": 0.0,
                "covered_key_atom_count_after_selection": 0,
                "covered_key_atom_weight_ratio_after_selection": 0.0,
                "covered_D4RT_atom_weight_ratio_after_selection": 0.0,
                "covered_semantic_atom_weight_ratio_after_selection": 0.0,
                **detail,
            }
        )
        used.add(cand.obs_id)
        frame_counts[cand.frame_id] = frame_counts.get(cand.frame_id, 0) + 1
        proto = cand.semantic_prototype_id or "none"
        proto_counts[proto] = proto_counts.get(proto, 0) + 1
        abin = _area_bin(cand.area_ratio)
        area_counts[abin] = area_counts.get(abin, 0) + 1
    return selected


def _metric_row(
    *,
    variant: str,
    budget: int,
    scene: str,
    chunk_id: str,
    selected: list[dict[str, Any]],
    diagnostic_stats: dict[tuple[int, int], dict[str, Any]],
    frame_data: list[dict[str, Any]],
) -> dict[str, Any]:
    cands = [row["candidate"] for row in selected]
    stats = [diagnostic_stats.get((cand.frame_id, cand.mask_id), {}) for cand in cands]
    oracle_eval = _oracle_eval_for_selected(
        frame_data=frame_data,
        selected=selected,
        diagnostic_stats=diagnostic_stats,
        variant=variant,
    )
    return {
        "scene_id": scene,
        "chunk_id": chunk_id,
        "variant": variant,
        "budget": int(budget),
        "selected_mask_count": len(cands),
        "selected_frame_count": len({cand.frame_id for cand in cands}),
        "selected_proto_count": len({cand.semantic_prototype_id for cand in cands if cand.semantic_prototype_id}),
        "top_selected_iou50_rate": _mean([1.0 if float(row.get("majority_iou") or 0.0) >= 0.50 else 0.0 for row in stats]),
        "top_selected_iou25_rate": _mean([1.0 if float(row.get("majority_iou") or 0.0) >= 0.25 else 0.0 for row in stats]),
        "top_selected_majority_iou_mean": _mean([float(row.get("majority_iou") or 0.0) for row in stats]),
        "top_selected_majority_purity_mean": _mean([float(row.get("majority_purity") or 0.0) for row in stats]),
        "top_selected_positive_gt_count_mean": _mean([float(row.get("positive_gt_count") or 0.0) for row in stats]),
        "selected_mask_area_ratio_mean": _mean([cand.area_ratio for cand in cands]),
        "selected_semantic_entropy_mean": _mean([cand.semantic_entropy for cand in cands]),
        "selected_semantic_prototype_margin_mean": _mean([cand.semantic_prototype_margin for cand in cands]),
        "broad_large_selected_rate": _mean([1.0 if cand.broad_large_risk else 0.0 for cand in cands]),
        "underseg_proxy_selected_rate": _mean([1.0 if cand.underseg_proxy else 0.0 for cand in cands]),
        "uses_gt_for_prediction": False,
        "diagnostic_only": False,
        "forbidden_for_method_table": False,
        **oracle_eval,
    }


def _selected_rows(
    *,
    variant: str,
    budget: int,
    scene: str,
    chunk_id: str,
    selected: list[dict[str, Any]],
    diagnostic_stats: dict[tuple[int, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for entry in selected:
        cand: CandidateMask = entry["candidate"]
        stats = diagnostic_stats.get((cand.frame_id, cand.mask_id), {})
        rows.append(
            {
                "scene_id": scene,
                "chunk_id": chunk_id,
                "variant": variant,
                "budget": budget,
                "selection_rank": entry["rank"],
                "frame_id": cand.frame_id,
                "mask_id": cand.mask_id,
                "mask_observation_id": cand.obs_id,
                "score_total": entry["score_total"],
                "base_objectness_score": entry.get("base_objectness_score"),
                "frame_diversity_gain": entry.get("frame_diversity_gain"),
                "prototype_diversity_gain": entry.get("prototype_diversity_gain"),
                "area_bin_diversity_gain": entry.get("area_bin_diversity_gain"),
                "area_ratio": cand.area_ratio,
                "semantic_entropy": cand.semantic_entropy,
                "semantic_prototype_margin": cand.semantic_prototype_margin,
                "semantic_prototype_id": cand.semantic_prototype_id,
                "broad_large_risk": cand.broad_large_risk,
                "underseg_proxy": cand.underseg_proxy,
                "diagnostic_majority_gt": stats.get("majority_gt"),
                "diagnostic_majority_iou": stats.get("majority_iou"),
                "diagnostic_majority_purity": stats.get("majority_purity"),
                "diagnostic_positive_gt_count": stats.get("positive_gt_count"),
                "uses_gt_for_prediction": False,
                "diagnostic_only": False,
                "forbidden_for_method_table": False,
            }
        )
    return rows


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    keys = sorted({(row["variant"], int(row["budget"])) for row in rows})
    metrics = [
        "selected_mask_count",
        "selected_frame_count",
        "selected_proto_count",
        "top_selected_iou50_rate",
        "top_selected_iou25_rate",
        "top_selected_majority_iou_mean",
        "top_selected_majority_purity_mean",
        "top_selected_positive_gt_count_mean",
        "selected_mask_area_ratio_mean",
        "selected_semantic_entropy_mean",
        "selected_semantic_prototype_margin_mean",
        "broad_large_selected_rate",
        "underseg_proxy_selected_rate",
        "representative_oracle_SF50",
        "representative_oracle_AP50",
        "representative_oracle_AP25",
        "representative_GT_best_IoU_mean",
        "representative_pred_best_IoU_median",
    ]
    for variant, budget in keys:
        subset = [row for row in rows if row["variant"] == variant and int(row["budget"]) == budget]
        item: dict[str, Any] = {"variant": variant, "budget": budget, "chunk_count": len(subset)}
        for metric in metrics:
            item[f"{metric}_mean"] = _mean([float(row[metric]) for row in subset if row.get(metric) not in ("", None)])
        item["uses_gt_for_prediction"] = False
        item["diagnostic_only"] = False
        item["forbidden_for_method_table"] = False
        out.append(item)
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    scenes = _parse_csv_list(args.scenes)
    budgets = [int(item) for item in str(args.budgets).split(",") if item.strip()]
    output_root = _rooted(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    candidates_by_chunk = _load_candidates(_rooted(args.candidate_rows), set(scenes))
    pipeline_roots = _load_pipeline_roots(_rooted(args.witness_summary), scenes)
    configs = [cfg for cfg in CONFIGS if cfg.name in set(_parse_csv_list(args.variants))]
    metric_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    processed = 0
    for scene in scenes:
        pipeline_root = pipeline_roots.get(scene)
        if pipeline_root is None:
            continue
        mask_dir = _mask_dir_from_pipeline(pipeline_root)
        stream = ScanNetStream(scene, root=ROOT / "data/scannet/processed")
        all_frames = stream.frame_ids(stride=int(args.stride), max_frames=None)
        for chunk in _chunk_rows(pipeline_root, scene):
            if int(args.max_chunks) > 0 and processed >= int(args.max_chunks):
                break
            chunk_id = str(chunk.get("chunk_id"))
            candidates = candidates_by_chunk.get(chunk_id, [])
            if not candidates:
                continue
            processed += 1
            print(f"[v71-objectness-diverse] chunk {processed}: {chunk_id}", file=sys.stderr, flush=True)
            raw_start = int(float(chunk.get("raw_frame_start") or 0))
            raw_end = int(float(chunk.get("raw_frame_end") or raw_start))
            frame_ids = [int(frame) for frame in all_frames if raw_start <= int(frame) <= raw_end]
            if not frame_ids:
                continue
            frame_data = _frame_data(scene, frame_ids, mask_dir)
            diagnostic_stats = _diagnostic_mask_stats(frame_data, {(cand.frame_id, cand.mask_id) for cand in candidates})
            for budget in budgets:
                for cfg in configs:
                    selected = _diverse_select(candidates, cfg, budget)
                    metric_rows.append(
                        _metric_row(
                            variant=cfg.name,
                            budget=budget,
                            scene=scene,
                            chunk_id=chunk_id,
                            selected=selected,
                            diagnostic_stats=diagnostic_stats,
                            frame_data=frame_data,
                        )
                    )
                    selected_rows.extend(
                        _selected_rows(
                            variant=cfg.name,
                            budget=budget,
                            scene=scene,
                            chunk_id=chunk_id,
                            selected=selected,
                            diagnostic_stats=diagnostic_stats,
                        )
                    )
        if int(args.max_chunks) > 0 and processed >= int(args.max_chunks):
            break
    summary_rows = _summarize(metric_rows)
    best = max(summary_rows, key=lambda row: float(row.get("representative_oracle_SF50_mean") or 0.0), default={})
    summary = {
        "decision": "OBJECTNESS_DIVERSE_SELECTION_DIAGNOSTIC_DONE",
        "processed_chunk_count": processed,
        "budgets": budgets,
        "variants": [cfg.name for cfg in configs],
        "best_variant": best.get("variant"),
        "best_budget": best.get("budget"),
        "best_representative_oracle_SF50": best.get("representative_oracle_SF50_mean"),
        "best_GT_best_IoU_mean": best.get("representative_GT_best_IoU_mean_mean"),
        "best_broad_large_selected_rate": best.get("broad_large_selected_rate_mean"),
        "best_underseg_proxy_selected_rate": best.get("underseg_proxy_selected_rate_mean"),
        "summary_rows": summary_rows,
    }
    _write_csv(output_root / "objectness_diverse_metric_rows.csv", metric_rows)
    _write_csv(output_root / "objectness_diverse_selected_rows.csv", selected_rows)
    _write_csv(output_root / "objectness_diverse_variant_summary_rows.csv", summary_rows)
    (output_root / "objectness_diverse_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sha_rows = [
        {"path": _rel(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in sorted(output_root.glob("*"))
        if path.is_file()
    ]
    _write_csv(output_root / "sha256_rows.csv", sha_rows)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", default=",".join(DEFAULT_SCENES))
    parser.add_argument("--candidate-rows", default="outputs/audit/v71_candidate_bank/candidate_mask_rows.csv")
    parser.add_argument("--witness-summary", default="outputs/audit/v70_carrier_witness/witness_summary.json")
    parser.add_argument("--output-root", default="outputs/audit/v71_objectness_diverse_selection")
    parser.add_argument("--variants", default=",".join(cfg.name for cfg in CONFIGS))
    parser.add_argument("--budgets", default="64,128,192")
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--max-chunks", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
