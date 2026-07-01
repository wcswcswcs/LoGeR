from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stream4d_native.v67_local_baselines import _row_from_eval, _summarize_variant_all  # noqa: E402
from tools.run_v65_scene_multiview_ap import _sha256, _write_csv, _write_json  # noqa: E402
from tools.run_v66_local_chunk_eval import (  # noqa: E402
    _best_variant,
    _chunk_rows,
    _float_or_none,
    _frame_data,
    _load_csv_rows,
    _mapping_from_candidates,
    _mean,
    _parse_json_list,
    _rel,
)
from tools.run_v66_scene_mv_ap_probe5 import (  # noqa: E402
    DEFAULT_SCENES,
    _discover_pipeline_root,
    _mask_dir_from_pipeline,
    _parse_csv_list,
)
from stream4d.scannet_stream import ScanNetStream  # noqa: E402


@dataclass(frozen=True)
class SetCoverConfig:
    name: str
    component_weight: float
    frame_bin_weight: float
    area_bin_weight: float
    source_frame_weight: float
    repeat_weight: float
    base_penalty: float
    underseg_penalty: float
    large_area_penalty: float
    same_frame_violation_penalty: float
    duplicate_source_frame_penalty: float
    reject_underseg: bool
    reject_large_area: bool
    wta: bool


CONFIGS = [
    SetCoverConfig("K0_component_cover_penalty0", 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False, False, False),
    SetCoverConfig("K1_object_balanced_component_frame_area", 0.45, 2.0, 2.0, 0.25, 0.0, 0.20, 0.0, 0.0, 0.0, 0.10, False, False, False),
    SetCoverConfig("K2_area_bin_underseg_penalty", 0.35, 2.0, 4.0, 0.25, 0.0, 0.35, 2.0, 3.0, 0.0, 0.20, False, False, False),
    SetCoverConfig("K3_frame_area_balanced", 0.30, 4.0, 3.0, 0.50, 0.0, 0.35, 2.0, 3.0, 0.0, 0.30, False, False, False),
    SetCoverConfig("K4_same_frame_violation_penalty", 0.30, 4.0, 3.0, 0.50, 0.0, 0.35, 2.0, 3.0, 8.0, 0.60, False, False, False),
    SetCoverConfig("K5_underseg_large_quarantine", 0.30, 4.0, 3.0, 0.50, 0.0, 0.35, 0.0, 0.0, 8.0, 0.60, True, True, False),
    SetCoverConfig("K6_repeated_signature_priority", 0.30, 4.0, 3.0, 0.50, 0.60, 0.35, 0.0, 0.0, 8.0, 0.60, True, True, False),
    SetCoverConfig("K7_repeated_signature_priority_WTA", 0.30, 4.0, 3.0, 0.50, 0.60, 0.35, 0.0, 0.0, 8.0, 0.60, True, True, True),
]


def _parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _float(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return float(default)
    try:
        return float(value)
    except Exception:
        return float(default)


def _area_bin(area_ratio: float) -> str:
    if area_ratio < 0.0025:
        return "tiny"
    if area_ratio < 0.02:
        return "small"
    if area_ratio < 0.12:
        return "medium"
    if area_ratio < 0.30:
        return "large"
    return "xlarge"


def _frame_bin(frame_id: int, frame_ids: list[int]) -> str:
    if not frame_ids:
        return "none"
    lo = int(frame_ids[0])
    hi = int(frame_ids[-1])
    if hi <= lo:
        return "q0"
    rel = (int(frame_id) - lo) / max(1, hi - lo)
    return f"q{min(3, max(0, int(rel * 4.0)))}"


def _mask_area_lookup(frame_data: list[dict[str, Any]]) -> dict[tuple[int, int], float]:
    out: dict[tuple[int, int], float] = {}
    for item in frame_data:
        frame_id = int(item["frame_id"])
        mask = item["mask"]
        if mask is None:
            continue
        total = int(mask.size)
        ids, counts = np.unique(mask[mask > 0], return_counts=True) if np.any(mask > 0) else ([], [])
        for mask_id, count in zip(ids, counts):
            out[(frame_id, int(mask_id))] = float(int(count) / max(1, total))
    return out


def _candidate_features(
    *,
    candidate_rows: list[dict[str, Any]],
    scene: str,
    chunk_id: str,
    frame_ids: list[int],
    area_lookup: dict[tuple[int, int], float],
) -> list[dict[str, Any]]:
    features: list[dict[str, Any]] = []
    for row in candidate_rows:
        if row.get("scene") != scene or row.get("chunk_id") != chunk_id:
            continue
        candidate_id = str(row.get("candidate_id") or "").strip()
        if not candidate_id:
            continue
        frame_id = int(_float(row.get("source_frame_id"), -1))
        mask_id = int(_float(row.get("source_mask_id"), -1))
        area_ratio = area_lookup.get((frame_id, mask_id), 0.0)
        components = set(_parse_json_list(row.get("component_ids")))
        features.append(
            {
                "candidate_id": candidate_id,
                "components": components,
                "frame_id": frame_id,
                "mask_id": mask_id,
                "area_ratio": float(area_ratio),
                "area_bin": _area_bin(float(area_ratio)),
                "frame_bin": _frame_bin(frame_id, frame_ids),
                "underseg_proxy": _parse_bool(row.get("underseg_proxy")),
                "same_frame_violation_rate": _float(row.get("same_frame_exclusion_violation_rate"), 0.0),
                "repeated_signature_len": _float(row.get("repeated_support_signature_len"), 0.0),
                "repeated_support_total": _float(row.get("repeated_support_total_support"), 0.0),
                "component_count": _float(row.get("component_count"), 0.0),
            }
        )
    return features


def _select_candidates(
    features: list[dict[str, Any]],
    *,
    config: SetCoverConfig,
    max_selected: int,
) -> tuple[list[str], dict[str, Any]]:
    total_components: set[str] = set()
    total_area_bins: set[str] = set()
    total_frame_bins: set[str] = set()
    for item in features:
        total_components |= set(item["components"])
        total_area_bins.add(str(item["area_bin"]))
        total_frame_bins.add(str(item["frame_bin"]))
    covered_components: set[str] = set()
    covered_area_bins: set[str] = set()
    covered_frame_bins: set[str] = set()
    selected_frames: set[int] = set()
    selected_ids: list[str] = []
    by_id = {str(item["candidate_id"]): item for item in features}
    while len(selected_ids) < int(max_selected):
        best_id = ""
        best_score = float("-inf")
        best_new_components = 0
        for item in features:
            candidate_id = str(item["candidate_id"])
            if candidate_id in selected_ids:
                continue
            if config.reject_underseg and bool(item["underseg_proxy"]):
                continue
            if config.reject_large_area and float(item["area_ratio"]) >= 0.30:
                continue
            new_components = set(item["components"]) - covered_components
            score = 0.0
            score += config.component_weight * math.sqrt(float(len(new_components)))
            score += config.frame_bin_weight if str(item["frame_bin"]) not in covered_frame_bins else 0.0
            score += config.area_bin_weight if str(item["area_bin"]) not in covered_area_bins else 0.0
            score += config.source_frame_weight if int(item["frame_id"]) not in selected_frames else 0.0
            score += config.repeat_weight * math.log1p(float(item["repeated_support_total"]))
            score -= config.base_penalty
            if bool(item["underseg_proxy"]):
                score -= config.underseg_penalty
            if float(item["area_ratio"]) >= 0.30:
                score -= config.large_area_penalty
            score -= config.same_frame_violation_penalty * float(item["same_frame_violation_rate"])
            if int(item["frame_id"]) in selected_frames:
                score -= config.duplicate_source_frame_penalty
            if score > best_score or (score == best_score and len(new_components) > best_new_components):
                best_id = candidate_id
                best_score = float(score)
                best_new_components = int(len(new_components))
        if not best_id:
            break
        if best_score <= 0.0 and selected_ids:
            break
        item = by_id[best_id]
        selected_ids.append(best_id)
        covered_components |= set(item["components"])
        covered_area_bins.add(str(item["area_bin"]))
        covered_frame_bins.add(str(item["frame_bin"]))
        selected_frames.add(int(item["frame_id"]))
        if covered_components == total_components and covered_area_bins == total_area_bins and covered_frame_bins == total_frame_bins:
            break
    selected = [by_id[cid] for cid in selected_ids if cid in by_id]
    underseg_count = sum(1 for item in selected if bool(item["underseg_proxy"]))
    same_frame_conflict_rate = _mean([float(item["same_frame_violation_rate"]) for item in selected])
    return selected_ids, {
        "selected_mask_count": int(len(selected_ids)),
        "coverage_component_ratio": float(len(covered_components) / max(1, len(total_components))),
        "coverage_area_bin_ratio": float(len(covered_area_bins) / max(1, len(total_area_bins))),
        "coverage_frame_bin_ratio": float(len(covered_frame_bins) / max(1, len(total_frame_bins))),
        "coverage_semantic_mode_ratio": "",
        "underseg_selected_rate": float(underseg_count / max(1, len(selected))),
        "same_frame_conflict_rate": same_frame_conflict_rate,
        "candidate_count": int(len(features)),
        "total_component_count": int(len(total_components)),
        "selected_component_count": int(len(covered_components)),
    }


def _mapping_from_selected_ids(
    *,
    selected_ids: list[str],
    ledger_rows: list[dict[str, Any]],
    scene: str,
    chunk_id: str,
    wta: bool,
) -> tuple[dict[tuple[int, int], int], dict[str, Any]]:
    candidate_to_object_idx = {candidate_id: idx + 1 for idx, candidate_id in enumerate(selected_ids)}
    mapping, diag = _mapping_from_candidates(
        ledger_rows=ledger_rows,
        scene=scene,
        chunk_id=chunk_id,
        candidate_to_object_idx=candidate_to_object_idx,
        wta=bool(wta),
    )
    diag["selected_mask_count"] = int(len(selected_ids))
    return mapping, diag


def _summarize_setcover_all(rows: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    base = _summarize_variant_all(rows, variant)
    subset = [row for row in rows if row["variant"] == variant]
    base.update(
        {
            "coverage_component_ratio_mean": _mean([_float_or_none(row.get("coverage_component_ratio")) for row in subset]),
            "coverage_area_bin_ratio_mean": _mean([_float_or_none(row.get("coverage_area_bin_ratio")) for row in subset]),
            "coverage_frame_bin_ratio_mean": _mean([_float_or_none(row.get("coverage_frame_bin_ratio")) for row in subset]),
            "underseg_selected_rate_mean": _mean([_float_or_none(row.get("underseg_selected_rate")) for row in subset]),
            "same_frame_conflict_rate_mean": _mean([_float_or_none(row.get("same_frame_conflict_rate")) for row in subset]),
            "selected_mask_count_mean": _mean([_float_or_none(row.get("selected_mask_count")) for row in subset]),
            "runtime_sec_mean": _mean([_float_or_none(row.get("runtime_sec")) for row in subset]),
        }
    )
    return base


def _best_variant_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return max(rows, key=lambda row: float(row.get("local_score_free_match50_recall_mean") or 0.0))


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)
    scenes = _parse_csv_list(args.scenes)
    rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    pipeline_roots: dict[str, str] = {}
    stride = int(args.stride)
    max_selected = int(args.max_selected)
    for scene in scenes:
        print(f"[v67-setcover] scene={scene}", file=sys.stderr, flush=True)
        pipeline_root = _discover_pipeline_root(scene)
        if pipeline_root is None:
            missing_rows.append({"scene_id": scene, "missing": "soma_fullscene_pipeline_root"})
            continue
        pipeline_roots[scene] = _rel(pipeline_root)
        _best_variant(pipeline_root)
        mask_dir = _mask_dir_from_pipeline(pipeline_root)
        stream = ScanNetStream(scene, root=ROOT / "data/scannet/processed")
        all_stride_frames = stream.frame_ids(stride=stride, max_frames=None)
        ledger_rows = _load_csv_rows(pipeline_root / "reprojection_ledger/reprojection_ledger_rows.csv")
        candidate_rows = _load_csv_rows(pipeline_root / "reprojection_ledger/candidate_rows.csv")
        for chunk in _chunk_rows(pipeline_root, scene):
            chunk_id = str(chunk.get("chunk_id"))
            raw_start = int(float(chunk.get("raw_frame_start") or 0))
            raw_end = int(float(chunk.get("raw_frame_end") or raw_start))
            frame_ids = [int(frame) for frame in all_stride_frames if raw_start <= int(frame) <= raw_end]
            if not frame_ids:
                continue
            frame_data = _frame_data(scene, frame_ids, mask_dir)
            area_lookup = _mask_area_lookup(frame_data)
            features = _candidate_features(
                candidate_rows=candidate_rows,
                scene=scene,
                chunk_id=chunk_id,
                frame_ids=frame_ids,
                area_lookup=area_lookup,
            )
            for config in CONFIGS:
                t0 = time.time()
                selected_ids, cover_diag = _select_candidates(features, config=config, max_selected=max_selected)
                mapping, map_diag = _mapping_from_selected_ids(
                    selected_ids=selected_ids,
                    ledger_rows=ledger_rows,
                    scene=scene,
                    chunk_id=chunk_id,
                    wta=config.wta,
                )
                row = _row_from_eval(
                    scene=scene,
                    chunk_id=chunk_id,
                    variant=config.name,
                    frame_ids=frame_ids,
                    chunk=chunk,
                    frame_data=frame_data,
                    mapping=mapping,
                    raw_per_frame_masks=False,
                    diag={**map_diag, **cover_diag},
                    uses_gt_for_prediction=False,
                    forbidden_for_method_table=False,
                    pipeline_root=pipeline_root,
                )
                row.update(cover_diag)
                row["wta"] = bool(config.wta)
                row["runtime_sec"] = float(time.time() - t0)
                rows.append(row)
    variant_summary_rows = [_summarize_setcover_all(rows, variant) for variant in sorted({row["variant"] for row in rows})]
    best = _best_variant_row(variant_summary_rows) or {}
    best_sf50 = _float_or_none(best.get("local_score_free_match50_recall_mean"))
    best_gt_best = _float_or_none(best.get("local_GT_best_IoU_mean_mean"))
    best_ap50 = _float_or_none(best.get("local_AP50_mean"))
    best_dup = _float_or_none(best.get("local_duplicate_frame_mask_conflict_rate_mean"))
    gate = {
        "all_pipeline_roots_available": len(pipeline_roots) == len(scenes),
        "best_K_local_SF50_ge_0p30": best_sf50 is not None and best_sf50 >= 0.30,
        "best_K_GT_best_IoU_ge_0p25": best_gt_best is not None and best_gt_best >= 0.25,
        "best_K_AP50_ge_0p05": best_ap50 is not None and best_ap50 >= 0.05,
        "best_K_duplicate_rate_le_0p02": best_dup is not None and best_dup <= 0.02,
    }
    gate["best_K_local_gate_pass"] = (
        gate["best_K_local_SF50_ge_0p30"]
        and gate["best_K_GT_best_IoU_ge_0p25"]
        and gate["best_K_AP50_ge_0p05"]
        and gate["best_K_duplicate_rate_le_0p02"]
    )
    decision = "PASS_OBJECT_BALANCED_SETCOVER" if gate["best_K_local_gate_pass"] else "OBJECT_BALANCED_SETCOVER_FAILS_LOCAL_GATE"
    _write_csv(output_root / "setcover_rows.csv", rows)
    _write_csv(output_root / "setcover_variant_summary_rows.csv", variant_summary_rows)
    _write_csv(output_root / "missing_input_rows.csv", missing_rows)
    summary = {
        "phase": "v67_object_balanced_key_mask_setcover",
        "decision": decision,
        "diagnostic_only": True,
        "uses_gt_for_prediction": False,
        "scenes": scenes,
        "stride": stride,
        "max_selected": max_selected,
        "pipeline_roots": pipeline_roots,
        "gate": gate,
        "best_K": best,
        "rows": {
            "setcover_rows_csv": _rel(output_root / "setcover_rows.csv"),
            "setcover_variant_summary_rows_csv": _rel(output_root / "setcover_variant_summary_rows.csv"),
            "missing_input_rows_csv": _rel(output_root / "missing_input_rows.csv"),
        },
        "notes": [
            "Selection uses candidate component IDs, source frame/mask IDs, mask area bins, frame bins, underseg_proxy, same-frame violation proxy, and repeated signature fields.",
            "The selector does not read GT labels; GT is used only by the downstream diagnostic evaluator, matching earlier local-baseline practice.",
            "No semantic mode atom is available in the current candidate table, so coverage_semantic_mode_ratio is left blank.",
        ],
    }
    _write_json(output_root / "setcover_summary.json", summary)
    sha_rows = []
    for path in [
        output_root / "setcover_summary.json",
        output_root / "setcover_rows.csv",
        output_root / "setcover_variant_summary_rows.csv",
        output_root / "missing_input_rows.csv",
    ]:
        if path.exists():
            sha_rows.append({"path": _rel(path), "sha256": _sha256(path), "bytes": int(path.stat().st_size)})
    _write_csv(output_root / "sha256_rows.csv", sha_rows)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run v67 object-balanced key-mask set cover diagnostics.")
    parser.add_argument("--scenes", default=",".join(DEFAULT_SCENES))
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--output-root", default="outputs/audit/v67_object_balanced_setcover")
    parser.add_argument("--max-selected", type=int, default=128)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
