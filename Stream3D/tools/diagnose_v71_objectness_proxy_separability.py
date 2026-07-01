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
class ProxyConfig:
    name: str
    allow_broad: bool
    allow_underseg: bool
    min_area: float
    max_area: float
    frame_cap: int
    proto_cap: int
    d4rt_weight: float
    area_weight: float
    entropy_weight: float
    margin_weight: float
    risk_penalty: float
    overlap_penalty: float


CONFIGS = [
    ProxyConfig("OP0_clean_mid_margin_entropy", False, False, 0.006, 0.24, 8, 16, 0.05, 1.00, 0.30, 2.00, 1.5, 0.06),
    ProxyConfig("OP1_clean_d4rt_reliability", False, False, 0.004, 0.26, 8, 16, 0.60, 0.65, 0.20, 1.25, 1.4, 0.06),
    ProxyConfig("OP2_clean_proto_frame_balanced", False, False, 0.006, 0.24, 4, 6, 0.10, 0.80, 0.20, 2.25, 1.5, 0.08),
    ProxyConfig("OP3_allow_large_entropy_d4rt", True, True, 0.004, 0.45, 8, 16, 0.45, 0.80, 0.50, 1.25, 0.8, 0.04),
    ProxyConfig("OP4_strict_low_overlap_margin", False, False, 0.008, 0.18, 4, 6, 0.05, 0.80, 0.20, 3.00, 1.7, 0.20),
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


def _auc_score(pairs: list[tuple[float, bool]]) -> float | None:
    pos = [score for score, label in pairs if label]
    neg = [score for score, label in pairs if not label]
    if not pos or not neg:
        return None
    wins = 0.0
    total = 0
    for ps in pos:
        for ns in neg:
            total += 1
            if ps > ns:
                wins += 1.0
            elif ps == ns:
                wins += 0.5
    return float(wins / max(1, total))


def _area_objectness(area: float) -> float:
    if area <= 0.0:
        return -1.0
    # Smooth preference for mid-sized regions without forbidding small object parts.
    center = 0.055
    log_dist = abs(math.log(max(1e-6, area) / center))
    return float(max(0.0, 1.0 - log_dist / math.log(8.0)))


def _score_candidate(cand: CandidateMask, cfg: ProxyConfig) -> float:
    broad = cand.broad_large_risk
    under = cand.underseg_proxy
    if cand.area_ratio < cfg.min_area or cand.area_ratio > cfg.max_area:
        return float("-inf")
    if broad and not cfg.allow_broad:
        return float("-inf")
    if under and not cfg.allow_underseg:
        return float("-inf")
    d4rt = cand.d4rt_reliability if cand.d4rt_reliability is not None else 0.0
    overlap = cand.same_frame_overlap_count + cand.same_frame_competing_mask_count
    score = (
        cfg.area_weight * _area_objectness(cand.area_ratio)
        + cfg.entropy_weight * cand.semantic_entropy
        + cfg.margin_weight * cand.semantic_prototype_margin
        + cfg.d4rt_weight * d4rt
        - cfg.overlap_penalty * overlap
    )
    score -= cfg.risk_penalty if broad else 0.0
    score -= cfg.risk_penalty if under else 0.0
    score -= 0.35 if cand.small_mask_risk else 0.0
    return float(score)


def _select_ranked(
    candidates: list[CandidateMask],
    cfg: ProxyConfig,
    budget: int,
    diagnostic_stats: dict[tuple[int, int], dict[str, Any]],
    oracle: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    scored: list[tuple[float, CandidateMask]] = []
    for cand in candidates:
        if oracle:
            stats = diagnostic_stats.get((cand.frame_id, cand.mask_id), {})
            score = float(stats.get("majority_iou") or 0.0) + 0.01 * float(stats.get("majority_purity") or 0.0)
        else:
            score = _score_candidate(cand, cfg)
        if math.isfinite(score):
            scored.append((score, cand))
    scored.sort(key=lambda item: (item[0], item[1].semantic_prototype_margin, -item[1].area_ratio), reverse=True)

    selected: list[dict[str, Any]] = []
    selected_obs: set[str] = set()
    frame_counts: dict[int, int] = {}
    proto_counts: dict[str, int] = {}
    rank_rows: list[dict[str, Any]] = []
    for rank, (score, cand) in enumerate(scored):
        stats = diagnostic_stats.get((cand.frame_id, cand.mask_id), {})
        rank_rows.append(
            {
                "rank": rank,
                "frame_id": cand.frame_id,
                "mask_id": cand.mask_id,
                "mask_observation_id": cand.obs_id,
                "score": score,
                "area_ratio": cand.area_ratio,
                "semantic_entropy": cand.semantic_entropy,
                "semantic_prototype_margin": cand.semantic_prototype_margin,
                "d4rt_reliability": cand.d4rt_reliability,
                "broad_large_risk": cand.broad_large_risk,
                "underseg_proxy": cand.underseg_proxy,
                "majority_gt": stats.get("majority_gt"),
                "majority_iou": stats.get("majority_iou"),
                "majority_purity": stats.get("majority_purity"),
                "positive_gt_count": stats.get("positive_gt_count"),
                "uses_gt_for_prediction": bool(oracle),
                "diagnostic_only": bool(oracle),
                "forbidden_for_method_table": bool(oracle),
            }
        )
        if len(selected) >= budget:
            continue
        if cand.obs_id in selected_obs:
            continue
        if not oracle:
            if frame_counts.get(cand.frame_id, 0) >= cfg.frame_cap:
                continue
            if cand.semantic_prototype_id and proto_counts.get(cand.semantic_prototype_id, 0) >= cfg.proto_cap:
                continue
        selected.append(
            {
                "candidate": cand,
                "rank": len(selected),
                "score_total": score,
                "score_new_atom_coverage": 0.0,
                "score_d4rt_coverage": 0.0,
                "score_semantic_coverage": 0.0,
                "new_key_atom_count": 0,
                "new_key_atom_weight": 0.0,
                "covered_key_atom_count_after_selection": 0,
                "covered_key_atom_weight_ratio_after_selection": 0.0,
                "covered_D4RT_atom_weight_ratio_after_selection": 0.0,
                "covered_semantic_atom_weight_ratio_after_selection": 0.0,
            }
        )
        selected_obs.add(cand.obs_id)
        frame_counts[cand.frame_id] = frame_counts.get(cand.frame_id, 0) + 1
        proto_counts[cand.semantic_prototype_id] = proto_counts.get(cand.semantic_prototype_id, 0) + 1
    diag = {
        "ranked_candidate_count": len(scored),
        "selected_mask_count": len(selected),
        "selected_frame_count": len({row["candidate"].frame_id for row in selected}),
        "selected_proto_count": len({row["candidate"].semantic_prototype_id for row in selected if row["candidate"].semantic_prototype_id}),
    }
    return selected, rank_rows, diag


def _selection_metric_row(
    *,
    variant: str,
    budget: int,
    scene: str,
    chunk_id: str,
    selected: list[dict[str, Any]],
    rank_rows: list[dict[str, Any]],
    diagnostic_stats: dict[tuple[int, int], dict[str, Any]],
    frame_data: list[dict[str, Any]],
    oracle: bool,
) -> dict[str, Any]:
    cands = [row["candidate"] for row in selected]
    pairs = []
    for row in rank_rows:
        iou = float(row.get("majority_iou") or 0.0)
        pairs.append((float(row["score"]), iou >= 0.50))
    oracle_eval = _oracle_eval_for_selected(
        frame_data=frame_data,
        selected=selected,
        diagnostic_stats=diagnostic_stats,
        variant=variant,
    )
    selected_stats = [diagnostic_stats.get((cand.frame_id, cand.mask_id), {}) for cand in cands]
    return {
        "scene_id": scene,
        "chunk_id": chunk_id,
        "variant": variant,
        "budget": int(budget),
        "selected_mask_count": len(cands),
        "ranked_candidate_count": len(rank_rows),
        "objectness_auc_iou50": _auc_score(pairs),
        "ranked_positive_iou50_rate": _mean([1.0 if float(row.get("majority_iou") or 0.0) >= 0.50 else 0.0 for row in rank_rows]),
        "top_selected_iou50_rate": _mean([1.0 if float(stats.get("majority_iou") or 0.0) >= 0.50 else 0.0 for stats in selected_stats]),
        "top_selected_iou25_rate": _mean([1.0 if float(stats.get("majority_iou") or 0.0) >= 0.25 else 0.0 for stats in selected_stats]),
        "top_selected_majority_iou_mean": _mean([float(stats.get("majority_iou") or 0.0) for stats in selected_stats]),
        "top_selected_majority_purity_mean": _mean([float(stats.get("majority_purity") or 0.0) for stats in selected_stats]),
        "top_selected_positive_gt_count_mean": _mean([float(stats.get("positive_gt_count") or 0.0) for stats in selected_stats]),
        "selected_mask_area_ratio_mean": _mean([cand.area_ratio for cand in cands]),
        "selected_semantic_entropy_mean": _mean([cand.semantic_entropy for cand in cands]),
        "selected_semantic_prototype_margin_mean": _mean([cand.semantic_prototype_margin for cand in cands]),
        "selected_d4rt_reliability_mean": _mean([cand.d4rt_reliability for cand in cands if cand.d4rt_reliability is not None]),
        "broad_large_selected_rate": _mean([1.0 if cand.broad_large_risk else 0.0 for cand in cands]),
        "underseg_proxy_selected_rate": _mean([1.0 if cand.underseg_proxy else 0.0 for cand in cands]),
        "small_mask_selected_rate": _mean([1.0 if cand.small_mask_risk else 0.0 for cand in cands]),
        "uses_gt_for_prediction": bool(oracle),
        "diagnostic_only": bool(oracle),
        "forbidden_for_method_table": bool(oracle),
        **oracle_eval,
    }


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = sorted({(row["variant"], int(row["budget"])) for row in rows})
    out = []
    metric_keys = [
        "selected_mask_count",
        "ranked_candidate_count",
        "objectness_auc_iou50",
        "ranked_positive_iou50_rate",
        "top_selected_iou50_rate",
        "top_selected_iou25_rate",
        "top_selected_majority_iou_mean",
        "top_selected_majority_purity_mean",
        "top_selected_positive_gt_count_mean",
        "selected_mask_area_ratio_mean",
        "selected_semantic_entropy_mean",
        "selected_semantic_prototype_margin_mean",
        "selected_d4rt_reliability_mean",
        "broad_large_selected_rate",
        "underseg_proxy_selected_rate",
        "small_mask_selected_rate",
        "representative_oracle_SF50",
        "representative_oracle_AP50",
        "representative_oracle_AP25",
        "representative_GT_best_IoU_mean",
        "representative_pred_best_IoU_median",
        "representative_oracle_mapping_count",
    ]
    for variant, budget in keys:
        subset = [row for row in rows if row["variant"] == variant and int(row["budget"]) == budget]
        item: dict[str, Any] = {
            "variant": variant,
            "budget": int(budget),
            "chunk_count": len(subset),
            "uses_gt_for_prediction": any(bool(row.get("uses_gt_for_prediction")) for row in subset),
            "diagnostic_only": any(bool(row.get("diagnostic_only")) for row in subset),
            "forbidden_for_method_table": any(bool(row.get("forbidden_for_method_table")) for row in subset),
        }
        for key in metric_keys:
            item[f"{key}_mean"] = _mean([float(row[key]) for row in subset if row.get(key) not in ("", None)])
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
    rank_rows_out: list[dict[str, Any]] = []
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
            print(f"[v71-objectness-proxy] chunk {processed}: {chunk_id}", file=sys.stderr, flush=True)
            raw_start = int(float(chunk.get("raw_frame_start") or 0))
            raw_end = int(float(chunk.get("raw_frame_end") or raw_start))
            frame_ids = [int(frame) for frame in all_frames if raw_start <= int(frame) <= raw_end]
            if not frame_ids:
                continue
            frame_data = _frame_data(scene, frame_ids, mask_dir)
            diagnostic_stats = _diagnostic_mask_stats(frame_data, {(cand.frame_id, cand.mask_id) for cand in candidates})
            for budget in budgets:
                for cfg in configs:
                    selected, rank_rows, _diag = _select_ranked(candidates, cfg, budget, diagnostic_stats, oracle=False)
                    metric_rows.append(
                        _selection_metric_row(
                            variant=cfg.name,
                            budget=budget,
                            scene=scene,
                            chunk_id=chunk_id,
                            selected=selected,
                            rank_rows=rank_rows,
                            diagnostic_stats=diagnostic_stats,
                            frame_data=frame_data,
                            oracle=False,
                        )
                    )
                    for row in rank_rows[: int(args.keep_rank_rows)]:
                        row.update({"scene_id": scene, "chunk_id": chunk_id, "variant": cfg.name, "budget": budget})
                        rank_rows_out.append(row)
                if args.include_oracle:
                    selected, rank_rows, _diag = _select_ranked(candidates, configs[0], budget, diagnostic_stats, oracle=True)
                    metric_rows.append(
                        _selection_metric_row(
                            variant="OP9_gt_iou_oracle_diagnostic",
                            budget=budget,
                            scene=scene,
                            chunk_id=chunk_id,
                            selected=selected,
                            rank_rows=rank_rows,
                            diagnostic_stats=diagnostic_stats,
                            frame_data=frame_data,
                            oracle=True,
                        )
                    )
                    for row in rank_rows[: int(args.keep_rank_rows)]:
                        row.update({"scene_id": scene, "chunk_id": chunk_id, "variant": "OP9_gt_iou_oracle_diagnostic", "budget": budget})
                        rank_rows_out.append(row)
        if int(args.max_chunks) > 0 and processed >= int(args.max_chunks):
            break

    summary_rows = _summarize(metric_rows)
    non_oracle = [row for row in summary_rows if not bool(row.get("uses_gt_for_prediction"))]
    best = max(non_oracle, key=lambda row: float(row.get("representative_oracle_SF50_mean") or 0.0), default={})
    summary = {
        "decision": "OBJECTNESS_PROXY_SEPARABILITY_DIAGNOSTIC_DONE",
        "processed_chunk_count": processed,
        "budgets": budgets,
        "variants": [cfg.name for cfg in configs],
        "include_oracle": bool(args.include_oracle),
        "best_non_oracle_variant": best.get("variant"),
        "best_non_oracle_budget": best.get("budget"),
        "best_non_oracle_representative_oracle_SF50": best.get("representative_oracle_SF50_mean"),
        "best_non_oracle_GT_best_IoU_mean": best.get("representative_GT_best_IoU_mean_mean"),
        "best_non_oracle_broad_large_selected_rate": best.get("broad_large_selected_rate_mean"),
        "best_non_oracle_underseg_proxy_selected_rate": best.get("underseg_proxy_selected_rate_mean"),
        "summary_rows": summary_rows,
    }
    _write_csv(output_root / "objectness_proxy_metric_rows.csv", metric_rows)
    _write_csv(output_root / "objectness_proxy_variant_summary_rows.csv", summary_rows)
    _write_csv(output_root / "objectness_proxy_rank_rows_top.csv", rank_rows_out)
    (output_root / "objectness_proxy_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
    parser.add_argument("--output-root", default="outputs/audit/v71_objectness_proxy_separability")
    parser.add_argument("--variants", default=",".join(cfg.name for cfg in CONFIGS))
    parser.add_argument("--budgets", default="64,128,192")
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--max-chunks", type=int, default=0)
    parser.add_argument("--include-oracle", action="store_true")
    parser.add_argument("--keep-rank-rows", type=int, default=64)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
