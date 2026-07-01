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
from stream4d_native.v71_representative_setcover import (  # noqa: E402
    CandidateMask,
    _diagnostic_mask_stats,
    _load_pipeline_roots,
    _oracle_eval_for_selected,
)
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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


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
            row["semantic_intra_variance"] = _float(row.get("semantic_intra_mask_variance"), 0.0)
            row["semantic_prototype_margin"] = _float(row.get("semantic_prototype_margin"), 0.0)
            row["bbox_x0"] = _float(row.get("bbox_x0"), 0.0)
            row["bbox_y0"] = _float(row.get("bbox_y0"), 0.0)
            row["bbox_x1"] = _float(row.get("bbox_x1"), 0.0)
            row["bbox_y1"] = _float(row.get("bbox_y1"), 0.0)
            row["broad_large_risk"] = _bool(row.get("broad_background_risk")) or _bool(row.get("large_mask_risk")) or row["area_ratio"] >= 0.30
            row["underseg_proxy"] = _float(row.get("underseg_proxy_score"), 0.0) >= 0.75
            row["small_mask_risk"] = _bool(row.get("small_mask_risk"))
            out[str(row.get("chunk_id") or "")].append(row)
    return out


def _candidate_obj(row: dict[str, Any]) -> CandidateMask:
    rel_raw = row.get("D4RT_carrier_reliability_mean")
    return CandidateMask(
        scene=str(row.get("scene_id") or ""),
        chunk_id=str(row.get("chunk_id") or ""),
        frame_id=int(row.get("frame_id") or -1),
        mask_id=int(row.get("mask_id") or -1),
        obs_id=str(row.get("mask_observation_id") or ""),
        area_ratio=_float(row.get("area_ratio"), 0.0),
        semantic_entropy=_float(row.get("semantic_entropy"), 1.0),
        semantic_prototype_margin=_float(row.get("semantic_prototype_margin"), 0.0),
        trajectory_entropy=_float(row.get("D4RT_carrier_trajectory_entropy"), 0.0),
        d4rt_reliability=None if rel_raw in (None, "") else _float(rel_raw, 0.0),
        semantic_prototype_id=str(row.get("semantic_prototype_id") or ""),
        source_flags=str(row.get("candidate_source_flags") or ""),
        representative_available=_bool(row.get("representative_available")),
        high_quality_raw_available=_bool(row.get("raw_cropformer_available")),
        small_mask_risk=_bool(row.get("small_mask_risk")),
        broad_large_risk=_bool(row.get("broad_large_risk")),
        underseg_proxy=_bool(row.get("underseg_proxy")),
        same_frame_overlap_count=_float(row.get("same_frame_overlap_count"), 0.0),
        same_frame_competing_mask_count=_float(row.get("same_frame_competing_mask_count"), 0.0),
    )


def _bbox_inside(inner: dict[str, Any], outer: dict[str, Any]) -> bool:
    return (
        _float(inner.get("bbox_x0")) >= _float(outer.get("bbox_x0")) - 2
        and _float(inner.get("bbox_y0")) >= _float(outer.get("bbox_y0")) - 2
        and _float(inner.get("bbox_x1")) <= _float(outer.get("bbox_x1")) + 2
        and _float(inner.get("bbox_y1")) <= _float(outer.get("bbox_y1")) + 2
    )


def _proposal_row(variant: str, source_type: str, target: dict[str, Any], source_ids: list[str], source_broad: bool, source_under: bool) -> dict[str, Any]:
    proposal_id = f"{variant}:{target.get('mask_observation_id')}:{len(source_ids)}"
    entropy = _float(target.get("semantic_entropy"), 1.0)
    area = _float(target.get("area_ratio"), 0.0)
    overlap = _float(target.get("same_frame_overlap_count")) + _float(target.get("same_frame_competing_mask_count"))
    compactness = max(0.0, 1.0 - entropy) + _float(target.get("semantic_prototype_margin"), 0.0)
    background = (1.0 if _bool(target.get("broad_large_risk")) else 0.0) + 0.25 * overlap + 0.5 * entropy
    return {
        "proposal_id": proposal_id,
        "scene_id": target.get("scene_id"),
        "chunk_id": target.get("chunk_id"),
        "frame_id": target.get("frame_id"),
        "target_mask_id": target.get("mask_id"),
        "target_mask_observation_id": target.get("mask_observation_id"),
        "source_mask_ids": "|".join(source_ids),
        "source_type": source_type,
        "proposal_area_ratio": area,
        "proposal_bbox": json.dumps(
            {
                "x0": target.get("bbox_x0"),
                "y0": target.get("bbox_y0"),
                "x1": target.get("bbox_x1"),
                "y1": target.get("bbox_y1"),
            },
            sort_keys=True,
        ),
        "semantic_backend": target.get("semantic_backend"),
        "semantic_entropy": entropy,
        "semantic_intra_variance": target.get("semantic_intra_variance"),
        "semantic_boundary_divergence": "",
        "semantic_prototype_id": target.get("semantic_prototype_id"),
        "semantic_prototype_margin": target.get("semantic_prototype_margin"),
        "source_broad_large_risk": source_broad,
        "source_underseg_proxy": source_under,
        "proposal_broad_large_risk": target.get("broad_large_risk"),
        "proposal_underseg_proxy": target.get("underseg_proxy"),
        "proposal_compactness_score": compactness,
        "proposal_background_proxy_score": background,
        "proposal_same_frame_overlap_count": overlap,
        "uses_gt_for_prediction": False,
        "diagnostic_only": False,
        "forbidden_for_method_table": False,
        "variant": variant,
    }


def _generate_proposals(candidates: list[dict[str, Any]], max_per_frame: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for cand in candidates:
        by_frame[int(cand.get("frame_id") or -1)].append(cand)
        rows.append(_proposal_row("SP0_existing_masks_baseline", "existing_mask", cand, [str(cand.get("mask_observation_id") or "")], _bool(cand.get("broad_large_risk")), _bool(cand.get("underseg_proxy"))))
    for frame_id, frame_rows in by_frame.items():
        sorted_entropy = sorted(
            [
                row
                for row in frame_rows
                if 0.003 <= _float(row.get("area_ratio")) <= 0.35 and not _bool(row.get("small_mask_risk"))
            ],
            key=lambda row: (_float(row.get("semantic_entropy")), _float(row.get("semantic_prototype_margin")), _float(row.get("area_ratio"))),
            reverse=True,
        )[:max_per_frame]
        for cand in sorted_entropy:
            rows.append(_proposal_row("SP1_DINO_entropy_ranked_mask_proposals", "entropy_ranked_existing_mask", cand, [str(cand.get("mask_observation_id") or "")], _bool(cand.get("broad_large_risk")), _bool(cand.get("underseg_proxy"))))
        clean = [
            row
            for row in frame_rows
            if 0.004 <= _float(row.get("area_ratio")) <= 0.18
            and not _bool(row.get("broad_large_risk"))
            and not _bool(row.get("underseg_proxy"))
            and not _bool(row.get("small_mask_risk"))
        ]
        for broad in [row for row in frame_rows if _bool(row.get("broad_large_risk")) or _bool(row.get("underseg_proxy"))]:
            subs = [
                row
                for row in clean
                if row.get("mask_observation_id") != broad.get("mask_observation_id")
                and _float(row.get("area_ratio")) < 0.75 * _float(broad.get("area_ratio"))
                and _bbox_inside(row, broad)
                and (
                    row.get("semantic_prototype_id") == broad.get("semantic_prototype_id")
                    or _float(row.get("semantic_entropy")) <= _float(broad.get("semantic_entropy"))
                )
            ]
            subs = sorted(subs, key=lambda row: (_float(row.get("semantic_entropy")), _float(row.get("semantic_prototype_margin"))), reverse=True)[:3]
            for sub in subs:
                rows.append(
                    _proposal_row(
                        "SP4_same_frame_mask_constrained_cut",
                        "broad_source_clean_submask_proxy",
                        sub,
                        [str(broad.get("mask_observation_id") or ""), str(sub.get("mask_observation_id") or "")],
                        True,
                        _bool(broad.get("underseg_proxy")),
                    )
                )
    return rows


def _evaluate_variant(
    *,
    variant: str,
    proposals: list[dict[str, Any]],
    candidate_lookup: dict[str, dict[str, Any]],
    diagnostic_stats: dict[tuple[int, int], dict[str, Any]],
    frame_data: list[dict[str, Any]],
) -> dict[str, Any]:
    selected = []
    for prop in proposals:
        cand = candidate_lookup.get(str(prop.get("target_mask_observation_id") or ""))
        if cand is None:
            continue
        selected.append(
            {
                "candidate": _candidate_obj(cand),
                "rank": len(selected),
                "score_total": _float(prop.get("proposal_compactness_score")),
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
    oracle = _oracle_eval_for_selected(frame_data=frame_data, selected=selected, diagnostic_stats=diagnostic_stats, variant=variant)
    stats = [diagnostic_stats.get((int(prop.get("frame_id") or -1), int(prop.get("target_mask_id") or -1)), {}) for prop in proposals]
    return {
        "variant": variant,
        "proposal_count": len(proposals),
        "proposal_count_per_frame": len(proposals) / max(1, len({int(prop.get("frame_id") or -1) for prop in proposals})),
        "proposal_area_ratio_mean": _mean([_float(prop.get("proposal_area_ratio")) for prop in proposals]),
        "proposal_semantic_entropy_mean": _mean([_float(prop.get("semantic_entropy")) for prop in proposals]),
        "proposal_background_proxy_rate": _mean([1.0 if _float(prop.get("proposal_background_proxy_score")) >= 0.75 else 0.0 for prop in proposals]),
        "proposal_source_broad_rate": _mean([1.0 if _bool(prop.get("source_broad_large_risk")) else 0.0 for prop in proposals]),
        "proposal_from_broad_count": sum(1 for prop in proposals if _bool(prop.get("source_broad_large_risk"))),
        "proposal_GT_best_IoU_mean_diagnostic": _mean([_float(stat.get("majority_iou")) for stat in stats]),
        "proposal_IoU50_rate_diagnostic": _mean([1.0 if _float(stat.get("majority_iou")) >= 0.50 else 0.0 for stat in stats]),
        "proposal_same_frame_conflict_rate": _mean([1.0 if _float(prop.get("proposal_same_frame_overlap_count")) > 0 else 0.0 for prop in proposals]),
        "uses_gt_for_prediction": False,
        **oracle,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = _rooted(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    scenes = _parse_csv_list(args.scenes)
    candidates_by_chunk = _load_candidate_rows(_rooted(args.candidate_rows), set(scenes))
    pipeline_roots = _load_pipeline_roots(_rooted(args.witness_summary), scenes)

    proposal_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
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
            print(f"[v72-semantic-proposals] chunk {processed}: {chunk_id}", file=sys.stderr, flush=True)
            raw_start = int(float(chunk.get("raw_frame_start") or 0))
            raw_end = int(float(chunk.get("raw_frame_end") or raw_start))
            frame_ids = [int(frame) for frame in all_frames if raw_start <= int(frame) <= raw_end]
            frame_data = _frame_data(scene, frame_ids, mask_dir)
            candidate_lookup = {str(row.get("mask_observation_id") or ""): row for row in candidates}
            proposals = _generate_proposals(candidates, int(args.max_proposals_per_frame))
            pairs = {(int(prop.get("frame_id") or -1), int(prop.get("target_mask_id") or -1)) for prop in proposals}
            diagnostic = _diagnostic_mask_stats(frame_data, pairs)
            for prop in proposals:
                stat = diagnostic.get((int(prop.get("frame_id") or -1), int(prop.get("target_mask_id") or -1)), {})
                prop["majority_iou_diagnostic"] = _float(stat.get("majority_iou"))
                prop["majority_gt_id_diagnostic"] = stat.get("majority_gt")
                proposal_rows.append(prop)
            by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for prop in proposals:
                by_variant[str(prop.get("variant") or "")].append(prop)
            for variant, subset in by_variant.items():
                row = _evaluate_variant(
                    variant=variant,
                    proposals=subset,
                    candidate_lookup=candidate_lookup,
                    diagnostic_stats=diagnostic,
                    frame_data=frame_data,
                )
                row.update({"scene_id": scene, "chunk_id": chunk_id})
                metric_rows.append(row)
        if int(args.max_chunks) > 0 and processed >= int(args.max_chunks):
            break

    variant_summary_rows = []
    for variant in sorted({row["variant"] for row in metric_rows}):
        subset = [row for row in metric_rows if row["variant"] == variant]
        item = {
            "variant": variant,
            "chunk_count": len(subset),
            "proposal_count_per_chunk_mean": _mean([_float(row.get("proposal_count")) for row in subset]),
            "proposal_count_per_frame_mean": _mean([_float(row.get("proposal_count_per_frame")) for row in subset]),
            "proposal_oracle_SF50_mean": _mean([_float(row.get("representative_oracle_SF50")) for row in subset]),
            "proposal_oracle_AP50_mean": _mean([_float(row.get("representative_oracle_AP50")) for row in subset]),
            "proposal_GT_best_IoU_mean": _mean([_float(row.get("proposal_GT_best_IoU_mean_diagnostic")) for row in subset]),
            "proposal_IoU50_rate_mean": _mean([_float(row.get("proposal_IoU50_rate_diagnostic")) for row in subset]),
            "proposal_background_proxy_rate_mean": _mean([_float(row.get("proposal_background_proxy_rate")) for row in subset]),
            "proposal_source_broad_rate_mean": _mean([_float(row.get("proposal_source_broad_rate")) for row in subset]),
            "proposal_same_frame_conflict_rate_mean": _mean([_float(row.get("proposal_same_frame_conflict_rate")) for row in subset]),
            "uses_gt_for_prediction": False,
        }
        variant_summary_rows.append(item)

    baseline = next((row for row in variant_summary_rows if row["variant"] == "SP0_existing_masks_baseline"), {})
    best = max(variant_summary_rows, key=lambda row: _float(row.get("proposal_oracle_SF50_mean"), -1.0), default={})
    sf50_gain = _float(best.get("proposal_oracle_SF50_mean"), 0.0) - _float(baseline.get("proposal_oracle_SF50_mean"), 0.0)
    gt_gain = _float(best.get("proposal_GT_best_IoU_mean"), 0.0) - _float(baseline.get("proposal_GT_best_IoU_mean"), 0.0)
    phase2_pass = (
        sf50_gain >= 0.10
        and gt_gain >= 0.08
        and _float(best.get("proposal_count_per_frame_mean"), 9999.0) <= 60.0
        and _float(best.get("proposal_count_per_chunk_mean"), 9999.0) <= 1200.0
    )
    summary = {
        "phase": "v72_phase2_semantic_proposals",
        "decision": "PASS_V72_PHASE2_SEMANTIC_PROPOSALS" if phase2_pass else "NO_GO_PHASE2_MASK_LEVEL_PROPOSAL_FALLBACK",
        "processed_chunk_count": processed,
        "proposal_row_count": len(proposal_rows),
        "best_variant": best.get("variant"),
        "baseline_SP0_oracle_SF50": baseline.get("proposal_oracle_SF50_mean"),
        "best_proposal_oracle_SF50": best.get("proposal_oracle_SF50_mean"),
        "best_minus_SP0_oracle_SF50": sf50_gain,
        "best_minus_SP0_GT_best_IoU": gt_gain,
        "gate": {
            "proposal_oracle_SF50_gain_ge_0p10": sf50_gain >= 0.10,
            "proposal_GT_best_IoU_gain_ge_0p08": gt_gain >= 0.08,
            "proposal_count_per_frame_mean_le_60": _float(best.get("proposal_count_per_frame_mean"), 9999.0) <= 60.0,
            "proposal_count_per_chunk_mean_le_1200": _float(best.get("proposal_count_per_chunk_mean"), 9999.0) <= 1200.0,
            "uses_gt_for_prediction_false": True,
            "pass": phase2_pass,
        },
        "notes": [
            "This implementation is a mask-level semantic fallback because current v71 artifacts expose mask-level DINO summaries, not dense token maps.",
            "SP4 uses same-frame clean masks as non-GT subproposal proxies for broad/underseg sources; it does not invent new pixel masks.",
        ],
    }
    _write_csv(output_root / "proposal_rows.csv", proposal_rows)
    _write_csv(output_root / "proposal_metric_rows.csv", metric_rows)
    _write_csv(output_root / "proposal_variant_summary_rows.csv", variant_summary_rows)
    _write_json(output_root / "semantic_proposal_summary.json", summary)
    _write_sha_rows(output_root, [_rooted(args.candidate_rows)])
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def _write_sha_rows(output_root: Path, inputs: list[Path]) -> None:
    rows = []
    for path in inputs:
        if path.exists():
            rows.append({"path": _rel(path), "bytes": path.stat().st_size, "sha256": _sha256(path), "kind": "input"})
    for path in sorted(output_root.glob("*")):
        if path.is_file() and path.name != "sha256_rows.csv":
            rows.append({"path": _rel(path), "bytes": path.stat().st_size, "sha256": _sha256(path), "kind": "output"})
    _write_csv(output_root / "sha256_rows.csv", rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="v72 Phase2 semantic proposal generation.")
    parser.add_argument("--scenes", default=",".join(DEFAULT_SCENES))
    parser.add_argument("--candidate-rows", default="outputs/audit/v71_candidate_bank/candidate_mask_rows.csv")
    parser.add_argument("--witness-summary", default="outputs/audit/v70_carrier_witness/witness_summary.json")
    parser.add_argument("--output-root", default="outputs/audit/v72_phase2_semantic_proposals")
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--max-chunks", type=int, default=0)
    parser.add_argument("--max-proposals-per-frame", type=int, default=40)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
