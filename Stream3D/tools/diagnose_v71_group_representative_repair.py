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
from stream4d_native.v67_local_baselines import _row_from_eval, _summarize_variant_all  # noqa: E402
from stream4d_native.v71_representative_setcover import _load_json, _load_pipeline_roots, _mean, _rel  # noqa: E402
from tools.run_v66_local_chunk_eval import _chunk_rows, _frame_data  # noqa: E402
from tools.run_v66_scene_mv_ap_probe5 import DEFAULT_SCENES, _mask_dir_from_pipeline, _parse_csv_list  # noqa: E402


def _rooted(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_candidates(path: Path, scenes: set[str]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if scenes and row.get("scene_id") not in scenes:
                continue
            out[str(row.get("chunk_id") or "")].append(row)
    return out


def _f(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    if value in ("", None):
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _b(row: dict[str, Any], key: str) -> bool:
    return str(row.get(key)).strip().lower() in {"1", "true", "yes", "y"}


def _candidate_score(row: dict[str, Any], variant: str) -> float:
    area = _f(row, "area_ratio")
    entropy = _f(row, "semantic_entropy", 1.0)
    margin = _f(row, "semantic_prototype_margin")
    broad = _b(row, "broad_background_risk") or _b(row, "large_mask_risk") or area >= 0.30
    underseg = _f(row, "underseg_proxy_score") >= 0.75
    small = _b(row, "small_mask_risk")
    clean = not broad and not underseg
    mid = 0.015 <= area <= 0.22
    score = 0.0
    if variant == "GR0_clean_mid_proto_signature":
        score += 3.0 if clean and mid else 0.0
        score += 1.2 if clean and 0.008 <= area < 0.015 else 0.0
    elif variant == "GR1_clean_mid_entropy_objectness":
        score += 2.5 if clean and mid else 0.0
        score += 0.7 * entropy
    elif variant == "GR2_clean_any_temporal":
        score += 1.5 if clean else 0.0
        score += 0.5 if 0.008 <= area <= 0.22 else 0.0
    elif variant == "GR3_area_temporal_signature_risky":
        score += 1.5 if 0.015 <= area <= 0.30 else 0.0
        score += 1.0 * min(0.30, area)
        score += 0.5 * entropy
    score += 2.0 * margin
    score -= 1.5 if broad else 0.0
    score -= 1.3 if underseg else 0.0
    score -= 0.5 if small else 0.0
    return float(score)


def _group_key(row: dict[str, Any], variant: str) -> tuple[str, str]:
    proto = str(row.get("semantic_prototype_id") or "")
    signature = str(row.get("repeated_signature_id") or "")
    if variant == "GR2_clean_any_temporal":
        size_bucket = signature.split("|", 1)[0] if signature else ""
        return proto, size_bucket
    if variant == "GR3_area_temporal_signature_risky":
        return proto, signature
    return proto, signature


def _select_group_mapping(rows: list[dict[str, Any]], variant: str, max_groups: int) -> tuple[dict[tuple[int, int], int], list[dict[str, Any]], dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[_group_key(row, variant)].append(row)
    group_infos = []
    for key, members in groups.items():
        clean_members = [
            row
            for row in members
            if not (_b(row, "broad_background_risk") or _b(row, "large_mask_risk") or _f(row, "area_ratio") >= 0.30)
            and _f(row, "underseg_proxy_score") < 0.75
        ]
        if variant in {"GR0_clean_mid_proto_signature", "GR1_clean_mid_entropy_objectness"}:
            usable = [row for row in clean_members if 0.015 <= _f(row, "area_ratio") <= 0.22]
        elif variant == "GR3_area_temporal_signature_risky":
            usable = [row for row in members if 0.015 <= _f(row, "area_ratio") <= 0.30]
        else:
            usable = [row for row in clean_members if 0.008 <= _f(row, "area_ratio") <= 0.22]
        if not usable:
            continue
        by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in usable:
            by_frame[int(float(row.get("frame_id") or -1))].append(row)
        best_per_frame = [max(frame_rows, key=lambda row: _candidate_score(row, variant)) for frame_rows in by_frame.values()]
        frame_count = len(best_per_frame)
        member_count = len(best_per_frame)
        score = (
            sum(_candidate_score(row, variant) for row in best_per_frame)
            + 0.80 * frame_count
            + 0.20 * min(10, len(usable))
            - 0.15 * _mean([_f(row, "same_frame_overlap_count") + _f(row, "same_frame_competing_mask_count") for row in best_per_frame])
        )
        group_infos.append(
            {
                "group_key": key,
                "score": float(score),
                "frame_count": int(frame_count),
                "member_count": int(member_count),
                "usable_count": int(len(usable)),
                "area_mean": _mean([_f(row, "area_ratio") for row in best_per_frame]),
                "entropy_mean": _mean([_f(row, "semantic_entropy") for row in best_per_frame]),
                "margin_mean": _mean([_f(row, "semantic_prototype_margin") for row in best_per_frame]),
                "members": best_per_frame,
            }
        )
    group_infos.sort(key=lambda item: (item["score"], item["frame_count"], item["usable_count"], item["group_key"]), reverse=True)
    mapping: dict[tuple[int, int], int] = {}
    object_rows = []
    for object_id, group in enumerate(group_infos[:max_groups], start=1):
        members = group["members"]
        for row in members:
            mapping[(int(float(row.get("frame_id") or -1)), int(float(row.get("mask_id") or -1)))] = object_id
        object_rows.append(
            {
                "variant": variant,
                "local_object_id": object_id,
                "group_key_semantic_prototype": group["group_key"][0],
                "group_key_signature": group["group_key"][1],
                "group_score": group["score"],
                "member_mask_count": group["member_count"],
                "member_frame_count": group["frame_count"],
                "usable_candidate_count": group["usable_count"],
                "mean_member_mask_area": group["area_mean"],
                "semantic_entropy_mean": group["entropy_mean"],
                "semantic_prototype_margin_mean": group["margin_mean"],
                "broad_large_member_rate": sum(
                    1
                    for row in members
                    if _b(row, "broad_background_risk") or _b(row, "large_mask_risk") or _f(row, "area_ratio") >= 0.30
                )
                / max(1, len(members)),
                "underseg_proxy_member_rate": sum(1 for row in members if _f(row, "underseg_proxy_score") >= 0.75) / max(1, len(members)),
                "uses_gt_for_prediction": False,
                "diagnostic_only": False,
                "forbidden_for_method_table": False,
            }
        )
    diag = {
        "selected_group_count": len(object_rows),
        "support_pair_count": len(mapping),
        "selected_mask_count": len(mapping),
        "mean_masks_per_group": float(len(mapping) / max(1, len(object_rows))),
        "duplicate_frame_mask_conflict_pairs": 0,
        "duplicate_frame_mask_conflict_rate": 0.0,
    }
    return mapping, object_rows, diag


def run(args: argparse.Namespace) -> dict[str, Any]:
    scenes = _parse_csv_list(args.scenes)
    output_root = _rooted(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    candidates_by_chunk = _read_candidates(_rooted(args.candidate_rows), set(scenes))
    pipeline_roots = _load_pipeline_roots(_rooted(args.witness_summary), scenes)
    atom_summary = _load_json(_rooted(args.atom_root) / "atom_summary.json")
    atom_metrics = atom_summary.get("key_metrics") if isinstance(atom_summary.get("key_metrics"), dict) else atom_summary
    diagnostic_gt_mean = float(atom_metrics.get("diagnostic_GT_count_per_chunk_mean") or 21.515923566878982)
    max_groups = int(args.max_groups_per_chunk or max(1, math.floor(3.0 * diagnostic_gt_mean)))
    variants = [item.strip() for item in str(args.variants).split(",") if item.strip()]
    object_rows_all: list[dict[str, Any]] = []
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
            rows = candidates_by_chunk.get(chunk_id, [])
            if not rows:
                continue
            processed += 1
            print(f"[v71-group-repair] chunk {processed}: {chunk_id}", file=sys.stderr, flush=True)
            raw_start = int(float(chunk.get("raw_frame_start") or 0))
            raw_end = int(float(chunk.get("raw_frame_end") or raw_start))
            frame_ids = [int(frame) for frame in all_frames if raw_start <= int(frame) <= raw_end]
            if not frame_ids:
                continue
            frame_data = _frame_data(scene, frame_ids, mask_dir)
            for variant in variants:
                mapping, object_rows, diag = _select_group_mapping(rows, variant, max_groups=max_groups)
                for row in object_rows:
                    row.update({"scene_id": scene, "chunk_id": chunk_id})
                object_rows_all.extend(object_rows)
                metric_rows.append(
                    _row_from_eval(
                        scene=scene,
                        chunk_id=chunk_id,
                        variant=variant,
                        frame_ids=frame_ids,
                        chunk=chunk,
                        frame_data=frame_data,
                        mapping=mapping,
                        raw_per_frame_masks=False,
                        diag=diag,
                        uses_gt_for_prediction=False,
                        forbidden_for_method_table=False,
                        pipeline_root=pipeline_root,
                    )
                )
        if int(args.max_chunks) > 0 and processed >= int(args.max_chunks):
            break
    summary_rows = [_summarize_variant_all(metric_rows, variant) for variant in variants]
    best = max(summary_rows, key=lambda row: float(row.get("local_SF50_mean") or row.get("local_score_free_match50_recall_mean") or 0.0), default={})
    summary = {
        "decision": "GROUP_REPRESENTATIVE_REPAIR_DIAGNOSTIC_DONE",
        "processed_chunk_count": processed,
        "max_groups_per_chunk": max_groups,
        "variants": variants,
        "best_variant": best.get("variant"),
        "best_variant_local_SF50": best.get("local_SF50_mean") or best.get("local_score_free_match50_recall_mean"),
        "best_variant_GT_best_IoU_mean": best.get("local_GT_best_IoU_mean_mean"),
        "best_variant_duplicate_frame_mask_conflict_rate": best.get("local_duplicate_frame_mask_conflict_rate_mean"),
        "summary_rows": summary_rows,
    }
    _write_csv(output_root / "group_object_rows.csv", object_rows_all)
    _write_csv(output_root / "group_metric_rows.csv", metric_rows)
    _write_csv(output_root / "group_variant_summary_rows.csv", summary_rows)
    (output_root / "group_repair_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
    parser.add_argument("--atom-root", default="outputs/audit/v71_d4rt_atoms")
    parser.add_argument("--output-root", default="outputs/audit/v71_group_representative_repair")
    parser.add_argument("--variants", default="GR0_clean_mid_proto_signature,GR1_clean_mid_entropy_objectness,GR2_clean_any_temporal,GR3_area_temporal_signature_risky")
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--max-chunks", type=int, default=0)
    parser.add_argument("--max-groups-per-chunk", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
