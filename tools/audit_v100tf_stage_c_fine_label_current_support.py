#!/usr/bin/env python3
"""Audit v100 Stage-C current support in the stable fine-label ID space.

The earlier quick prototype joined R2 source labels to Stage-C label_maps by
raw numeric ID.  Stage-C raw IDs are local to each cache label_names list, while
R2 source_label_mode comes from SemanticPriorGenerator's stable fine-label IDs.
This script repairs that join by mapping current Stage-C raw label IDs through
the same fine-label metadata before scoring current support.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from loger.pipeline.semantic_prior_generator import (
    SEMANTIC_FINE_MOVABLE_IDS,
    SEMANTIC_FINE_SKY_IDS,
    SEMANTIC_FINE_STRUCTURE_IDS,
    SEMANTIC_FINE_VEGETATION_IDS,
    _dense_label_metadata,
    _mode_pool_dense_semantic_patches,
    _normalize_dense_semantic_confidence,
)
from tools.build_v100tf_same_space_semantic_anchor_latent_state_multiroute_memory_control import (
    evaluate_pattern,
    f,
    pearson,
    write_json,
    write_rows,
)


ROOT = Path("results/acl2_v100tf_same_space_semantic_anchor_latent_state_memory_control")
L2_DIR = ROOT / "trackL2_anchor_scale_observability"
R2_DIR = ROOT / "trackR2_anchor_edge_identity_control_audit"
STAGE_C_ROOT = Path("results/kitti_preprocess")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def clean(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, torch.Tensor):
        return clean(value.detach().cpu().tolist())
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def fine_group(label_id: int) -> str:
    lid = int(label_id)
    if lid == 0:
        return "unknown"
    if lid in SEMANTIC_FINE_STRUCTURE_IDS:
        return "structure"
    if lid in SEMANTIC_FINE_SKY_IDS:
        return "sky"
    if lid in SEMANTIC_FINE_VEGETATION_IDS:
        return "vegetation"
    if lid in SEMANTIC_FINE_MOVABLE_IDS:
        return "movable"
    return "other"


def build_anchor_semantic_map(anchor_rows: list[dict[str, str]]) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    counts: dict[tuple[str, str], Counter[int]] = defaultdict(Counter)
    row_counts = 0
    missing = 0
    for row in anchor_rows:
        case_id = str(row.get("case_id", ""))
        anchor_id = str(row.get("anchor_id", ""))
        semantic = row.get("semantic_class", "")
        if not case_id or not anchor_id:
            continue
        val = f(semantic)
        if not math.isfinite(val):
            missing += 1
            continue
        counts[(case_id, anchor_id)][int(val)] += 1
        row_counts += 1
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for key, counter in counts.items():
        label_id, count = counter.most_common(1)[0]
        total = sum(counter.values())
        out[key] = {
            "anchor_fine_label_id": int(label_id),
            "anchor_fine_label_frac": float(count) / float(total) if total else math.nan,
            "anchor_fine_label_observations": int(total),
            "anchor_fine_group": fine_group(int(label_id)),
        }
    return out, {
        "anchor_semantic_key_count": len(out),
        "anchor_semantic_observation_rows": row_counts,
        "anchor_semantic_missing_rows": missing,
    }


def stage_c_masklet_path(seq: str, chunk_idx: int) -> Path | None:
    cache = STAGE_C_ROOT / str(seq) / "stage_c_cache_semantic_chunks"
    index = cache / "cache_index.jsonl"
    if index.is_file():
        with index.open("r", encoding="utf-8") as handle:
            for raw in handle:
                raw = raw.strip()
                if not raw:
                    continue
                row = json.loads(raw)
                if int(row.get("chunk_idx", -1)) == int(chunk_idx):
                    path = cache / str(row.get("chunk", "")) / "masklet.pt"
                    return path if path.is_file() else None
    matches = sorted(cache.glob(f"chunk_{int(chunk_idx):03d}_*/masklet.pt"))
    return matches[0] if matches else None


def load_stage_c_patch_labels(seq: str, chunk_idx: int, patch_grid: tuple[int, int]) -> dict[str, Any]:
    path = stage_c_masklet_path(seq, chunk_idx)
    if path is None:
        raise FileNotFoundError(f"missing Stage-C masklet for seq={seq} chunk={chunk_idx}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    sem = payload.get("semantic_segmentation") if isinstance(payload, dict) else getattr(payload, "semantic_segmentation", None)
    if not isinstance(sem, dict) or "label_maps" not in sem:
        raise KeyError(f"missing semantic_segmentation.label_maps: {path}")
    labels = sem["label_maps"]
    if not isinstance(labels, torch.Tensor):
        labels = torch.as_tensor(labels)
    labels = labels.detach().cpu().long()
    confidence, _ = _normalize_dense_semantic_confidence(
        sem.get("confidence_maps"),
        target_shape=(int(labels.shape[0]), int(labels.shape[1]), int(labels.shape[2])),
    )
    raw_patch, purity_patch, conf_patch = _mode_pool_dense_semantic_patches(
        labels,
        confidence,
        patch_grid=patch_grid,
    )
    meta = _dense_label_metadata(sem.get("label_names", []))
    max_label_id = int(raw_patch.max().item()) if raw_patch.numel() else 0
    lut_len = max(max_label_id + 1, int(meta["fine_ids"].numel()), 1)
    fine_lut = torch.zeros((lut_len,), dtype=torch.long)
    n_meta = int(meta["fine_ids"].numel())
    if n_meta > 0:
        fine_lut[: min(n_meta, lut_len)] = meta["fine_ids"][: min(n_meta, lut_len)]
    safe_raw = raw_patch.clamp(min=0, max=lut_len - 1)
    fine_patch = fine_lut[safe_raw]
    group_patch = torch.empty_like(fine_patch)
    for lid in torch.unique(fine_patch).tolist():
        group_name = fine_group(int(lid))
        group_id = {"unknown": 0, "structure": 1, "sky": 2, "vegetation": 3, "movable": 4, "other": 5}[group_name]
        group_patch[fine_patch == int(lid)] = int(group_id)
    return {
        "path": str(path),
        "raw_patch": raw_patch,
        "fine_patch": fine_patch,
        "group_patch": group_patch,
        "purity_patch": purity_patch,
        "conf_patch": conf_patch,
        "label_names": [str(x) for x in sem.get("label_names", [])],
        "canonical_names": list(meta.get("canonical_names", [])),
    }


def finite_mean(values: list[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return sum(vals) / len(vals) if vals else math.nan


def aggregate_case_rows(edge_rows: list[dict[str, str]], anchor_sem: dict[tuple[str, str], dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    patch_cache: dict[tuple[str, int], dict[str, Any]] = {}
    case_parts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    read_errors: list[dict[str, Any]] = []
    edge_count = 0
    source_semantic_count = 0
    fine_match_count = 0
    group_match_count = 0
    stage_c_chunks_loaded: set[tuple[str, int]] = set()
    source_label_counter: Counter[int] = Counter()
    current_label_counter: Counter[int] = Counter()

    for row in edge_rows:
        edge_count += 1
        case_id = str(row.get("case_id", ""))
        anchor_id = str(row.get("anchor_id", ""))
        seq = str(row.get("seq", ""))
        curr_chunk = int(f(row.get("curr_chunk"), -1))
        query_frame = int(f(row.get("query_frame"), -1))
        query_pr = int(f(row.get("query_patch_row"), -1))
        query_pc = int(f(row.get("query_patch_col"), -1))
        sem = anchor_sem.get((case_id, anchor_id))
        if sem is None:
            continue
        source_semantic_count += 1
        source_label = int(sem["anchor_fine_label_id"])
        if source_label <= 0:
            continue
        key = (seq, curr_chunk)
        if key not in patch_cache:
            try:
                patch_cache[key] = load_stage_c_patch_labels(seq, curr_chunk, (19, 66))
                stage_c_chunks_loaded.add(key)
            except Exception as exc:
                read_errors.append({"seq": seq, "curr_chunk": curr_chunk, "error": f"{type(exc).__name__}:{exc}"})
                patch_cache[key] = {}
        patch = patch_cache.get(key) or {}
        fine_patch = patch.get("fine_patch")
        group_patch = patch.get("group_patch")
        purity_patch = patch.get("purity_patch")
        conf_patch = patch.get("conf_patch")
        if not (
            torch.is_tensor(fine_patch)
            and torch.is_tensor(group_patch)
            and torch.is_tensor(purity_patch)
            and torch.is_tensor(conf_patch)
            and 0 <= query_frame < int(fine_patch.shape[0])
            and 0 <= query_pr < int(fine_patch.shape[1])
            and 0 <= query_pc < int(fine_patch.shape[2])
        ):
            continue
        current_label = int(fine_patch[query_frame, query_pr, query_pc].item())
        current_group = int(group_patch[query_frame, query_pr, query_pc].item())
        source_group = {"unknown": 0, "structure": 1, "sky": 2, "vegetation": 3, "movable": 4, "other": 5}[fine_group(source_label)]
        conf = max(0.0, min(1.0, f(conf_patch[query_frame, query_pr, query_pc].item(), 0.0)))
        purity = max(0.0, min(1.0, f(purity_patch[query_frame, query_pr, query_pc].item(), 0.0)))
        fine_match = int(current_label == source_label)
        group_match = int(current_group == source_group and source_group != 0)
        fine_support = float(fine_match) * conf * purity
        group_support = float(group_match) * conf * purity
        baseline = f(row.get("camera_translation_baseline"))
        depth_ratio = f(row.get("abs_log_depth_ratio"))
        source_label_counter[source_label] += 1
        current_label_counter[current_label] += 1
        fine_match_count += fine_match
        group_match_count += group_match
        case_parts[case_id].append({
            "case_id": case_id,
            "seq": seq,
            "case_label": row.get("case_label", ""),
            "L3_handoff_transfer_penalty_proxy": f(row.get("L3_handoff_transfer_penalty_proxy")),
            "source_fine_label": source_label,
            "current_fine_label": current_label,
            "source_group": source_group,
            "current_group": current_group,
            "fine_match": fine_match,
            "group_match": group_match,
            "conf": conf,
            "purity": purity,
            "fine_support": fine_support,
            "group_support": group_support,
            "fine_parallax_support": fine_support * baseline if math.isfinite(baseline) else math.nan,
            "group_parallax_support": group_support * baseline if math.isfinite(baseline) else math.nan,
            "fine_parallax_depth_support": fine_support * baseline * depth_ratio if math.isfinite(baseline) and math.isfinite(depth_ratio) else math.nan,
            "group_parallax_depth_support": group_support * baseline * depth_ratio if math.isfinite(baseline) and math.isfinite(depth_ratio) else math.nan,
            "depth_ratio": depth_ratio,
        })

    case_rows: list[dict[str, Any]] = []
    for case_id, parts in sorted(case_parts.items()):
        base = parts[0]
        fine_support = [f(part.get("fine_support")) for part in parts]
        group_support = [f(part.get("group_support")) for part in parts]
        fine_match = [f(part.get("fine_match")) for part in parts]
        group_match = [f(part.get("group_match")) for part in parts]
        confs = [f(part.get("conf")) for part in parts]
        purities = [f(part.get("purity")) for part in parts]
        fine_parallax = [f(part.get("fine_parallax_support")) for part in parts]
        group_parallax = [f(part.get("group_parallax_support")) for part in parts]
        fine_depth = [f(part.get("fine_parallax_depth_support")) for part in parts]
        group_depth = [f(part.get("group_parallax_depth_support")) for part in parts]
        depth_ratio = [f(part.get("depth_ratio")) for part in parts if int(f(part.get("fine_match"), 0)) == 1]
        case_rows.append({
            "case_id": case_id,
            "seq": base.get("seq", ""),
            "case_label": base.get("case_label", ""),
            "L3_handoff_transfer_penalty_proxy": base.get("L3_handoff_transfer_penalty_proxy"),
            "fine_label_edge_count": len(parts),
            "fine_semantic_match_count": int(sum(fine_match)),
            "fine_semantic_match_frac": finite_mean(fine_match),
            "fine_semantic_current_support": finite_mean(fine_support),
            "fine_low_semantic_current_support_risk": 1.0 - finite_mean(fine_support) if math.isfinite(finite_mean(fine_support)) else math.nan,
            "fine_semantic_match_conf_mean": finite_mean([a * b for a, b in zip(fine_match, confs)]),
            "fine_semantic_match_purity_mean": finite_mean([a * b for a, b in zip(fine_match, purities)]),
            "fine_semantic_parallax_current": finite_mean(fine_parallax),
            "fine_semantic_parallax_depth_current": finite_mean(fine_depth),
            "fine_semantic_depth_ratio_on_match": finite_mean(depth_ratio),
            "group_semantic_match_count": int(sum(group_match)),
            "group_semantic_match_frac": finite_mean(group_match),
            "group_semantic_current_support": finite_mean(group_support),
            "group_low_semantic_current_support_risk": 1.0 - finite_mean(group_support) if math.isfinite(finite_mean(group_support)) else math.nan,
            "group_semantic_match_conf_mean": finite_mean([a * b for a, b in zip(group_match, confs)]),
            "group_semantic_parallax_current": finite_mean(group_parallax),
            "group_semantic_parallax_depth_current": finite_mean(group_depth),
        })

    meta = {
        "edge_count": edge_count,
        "source_semantic_edge_count": source_semantic_count,
        "fine_semantic_matched_edge_count": fine_match_count,
        "group_semantic_matched_edge_count": group_match_count,
        "stage_c_chunks_loaded": len(stage_c_chunks_loaded),
        "read_error_count": len(read_errors),
        "source_fine_label_top10": source_label_counter.most_common(10),
        "current_fine_label_top10": current_label_counter.most_common(10),
    }
    return case_rows, read_errors, meta


def score_case_rows(case_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    specs = [
        ("fine_semantic_match_frac", "higher_bad"),
        ("fine_semantic_match_frac", "lower_bad"),
        ("fine_semantic_current_support", "higher_bad"),
        ("fine_semantic_current_support", "lower_bad"),
        ("fine_low_semantic_current_support_risk", "higher_bad"),
        ("fine_semantic_match_conf_mean", "higher_bad"),
        ("fine_semantic_parallax_current", "lower_bad"),
        ("fine_semantic_parallax_current", "higher_bad"),
        ("fine_semantic_parallax_depth_current", "lower_bad"),
        ("fine_semantic_depth_ratio_on_match", "higher_bad"),
        ("fine_semantic_depth_ratio_on_match", "lower_bad"),
        ("group_semantic_match_frac", "higher_bad"),
        ("group_semantic_match_frac", "lower_bad"),
        ("group_semantic_current_support", "higher_bad"),
        ("group_semantic_current_support", "lower_bad"),
        ("group_low_semantic_current_support_risk", "higher_bad"),
        ("group_semantic_match_conf_mean", "higher_bad"),
        ("group_semantic_parallax_current", "lower_bad"),
        ("group_semantic_parallax_current", "higher_bad"),
        ("group_semantic_parallax_depth_current", "lower_bad"),
    ]
    metrics: list[dict[str, Any]] = []
    for field, direction in specs:
        metric = evaluate_pattern(case_rows, f"{field}_{direction}", field, direction)
        metric["field"] = field
        metric["direction"] = direction
        metric["gate_like"] = bool(
            f(metric.get("bad_recall")) >= 0.65
            and f(metric.get("good_FPR"), 1.0) <= 0.25
            and f(metric.get("abs_corr_L3")) >= 0.50
            and bool(metric.get("corr_direction_correct"))
            and int(f(metric.get("sequence_coverage"), 0)) >= 4
            and f(metric.get("selected_positive_sequence_max_frac"), 1.0) <= 0.60
        )
        metrics.append(metric)
    metrics.sort(
        key=lambda row: (
            bool(row.get("gate_like")),
            f(row.get("balanced_accuracy")),
            f(row.get("bad_recall")),
            -f(row.get("good_FPR"), 1.0),
        ),
        reverse=True,
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry-rows", type=Path, default=L2_DIR / "geometry_edge_rows.csv")
    parser.add_argument("--anchor-rows", type=Path, default=R2_DIR / "anchor_edge_rows.csv")
    parser.add_argument("--out-dir", type=Path, default=L2_DIR)
    args = parser.parse_args()

    edge_rows = read_rows(args.geometry_rows)
    anchor_rows = read_rows(args.anchor_rows)
    anchor_sem, anchor_meta = build_anchor_semantic_map(anchor_rows)
    case_rows, read_errors, meta = aggregate_case_rows(edge_rows, anchor_sem)
    metrics = score_case_rows(case_rows)
    best = metrics[0] if metrics else {}
    summary = {
        **anchor_meta,
        **meta,
        "case_count": len(case_rows),
        "metrics": len(metrics),
        "gate_like_count": sum(1 for row in metrics if row.get("gate_like")),
        "best": best,
        "note": "Current Stage-C raw labels are mapped to SemanticPriorGenerator stable fine-label IDs before matching R2 source_label_mode.",
    }
    out = args.out_dir
    write_rows(out / "stage_c_fine_label_current_support_rows.csv", case_rows)
    write_rows(out / "stage_c_fine_label_current_support_metrics.csv", metrics)
    write_rows(out / "stage_c_fine_label_current_support_read_errors.csv", read_errors)
    write_json(out / "stage_c_fine_label_current_support_summary.json", summary)
    report = [
        "# Stage-C Fine-Label Current Support Audit",
        "",
        "This audit repairs the earlier raw-ID join by mapping Stage-C label_maps through SemanticPriorGenerator stable fine-label IDs before matching R2 source_label_mode.",
        "",
        f"- case_count: `{len(case_rows)}`",
        f"- edge_count: `{meta['edge_count']}`",
        f"- source_semantic_edge_count: `{meta['source_semantic_edge_count']}`",
        f"- fine_semantic_matched_edge_count: `{meta['fine_semantic_matched_edge_count']}`",
        f"- group_semantic_matched_edge_count: `{meta['group_semantic_matched_edge_count']}`",
        f"- stage_c_chunks_loaded: `{meta['stage_c_chunks_loaded']}`",
        f"- read_error_count: `{meta['read_error_count']}`",
        f"- gate_like_count: `{summary['gate_like_count']}`",
        "",
        "## Best Metric",
        "",
        "```json",
        json.dumps(clean(best), ensure_ascii=False, indent=2, sort_keys=True),
        "```",
        "",
        "Gate-like here is diagnostic-only and still lacks true anchor-id, semantic-label, and query-head rerun controls.",
    ]
    (out / "stage_c_fine_label_current_support_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(clean(summary), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
