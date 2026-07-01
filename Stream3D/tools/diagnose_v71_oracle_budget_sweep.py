from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stream4d.scannet_stream import ScanNetStream  # noqa: E402
from stream4d_native.v71_representative_setcover import (  # noqa: E402
    _diagnostic_mask_stats,
    _load_candidates,
    _load_pipeline_roots,
    _mean,
    _rel,
)
from tools.run_v66_local_chunk_eval import _chunk_rows, _evaluate_frame_data, _frame_data, _score_free  # noqa: E402
from tools.run_v66_scene_mv_ap_probe5 import DEFAULT_SCENES, _mask_dir_from_pipeline, _parse_csv_list  # noqa: E402


def _rooted(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args: argparse.Namespace) -> dict[str, Any]:
    scenes = _parse_csv_list(args.scenes)
    budgets = [int(x) for x in str(args.budgets).split(",") if str(x).strip()]
    output_root = _rooted(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    candidates_by_chunk = _load_candidates(_rooted(args.candidate_rows), set(scenes))
    pipeline_roots = _load_pipeline_roots(_rooted(args.witness_summary), scenes)
    rows: list[dict[str, Any]] = []
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
            print(f"[v71-oracle-budget] chunk {processed}: {chunk_id}", file=sys.stderr, flush=True)
            raw_start = int(float(chunk.get("raw_frame_start") or 0))
            raw_end = int(float(chunk.get("raw_frame_end") or raw_start))
            frame_ids = [int(frame) for frame in all_frames if raw_start <= int(frame) <= raw_end]
            if not frame_ids:
                continue
            data = _frame_data(scene, frame_ids, mask_dir)
            stats = _diagnostic_mask_stats(data, {(cand.frame_id, cand.mask_id) for cand in candidates})
            oracle_items = []
            best_by_gt: dict[int, tuple[float, float, Any, int]] = {}
            for cand in candidates:
                item_stats = stats.get((cand.frame_id, cand.mask_id), {})
                gid = int(item_stats.get("majority_gt") or 0)
                iou = float(item_stats.get("majority_iou") or 0.0)
                purity = float(item_stats.get("majority_purity") or 0.0)
                item = (iou, purity, cand, gid)
                oracle_items.append(item)
                if gid > 0 and (gid not in best_by_gt or item[:2] > best_by_gt[gid][:2]):
                    best_by_gt[gid] = item
            best_items = sorted(best_by_gt.values(), key=lambda x: (x[0], x[1]), reverse=True)
            fill_items = sorted(oracle_items, key=lambda x: (x[0], x[1]), reverse=True)
            for budget in budgets:
                selected = []
                used: set[str] = set()
                for item in best_items:
                    if len(selected) >= budget:
                        break
                    cand = item[2]
                    if cand.obs_id in used:
                        continue
                    selected.append(item)
                    used.add(cand.obs_id)
                for item in fill_items:
                    if len(selected) >= budget:
                        break
                    cand = item[2]
                    if cand.obs_id in used:
                        continue
                    selected.append(item)
                    used.add(cand.obs_id)
                mapping = {(item[2].frame_id, item[2].mask_id): item[3] for item in selected if item[3] > 0}
                summary, _iou, _pred_ids, _gt_ids = _evaluate_frame_data(
                    frame_data=data,
                    variant=f"oracle_budget_{budget}",
                    mapping=mapping,
                    raw_per_frame_masks=False,
                )
                rows.append(
                    {
                        "budget": int(budget),
                        "scene_id": scene,
                        "chunk_id": chunk_id,
                        "candidate_pair_count": len(candidates),
                        "selected_mask_count": len(selected),
                        "local_gt_count": summary.get("evaluated_gt_count"),
                        "oracle_SF50": _score_free(summary),
                        "oracle_AP50": summary.get("ap50"),
                        "oracle_AP25": summary.get("ap25"),
                        "oracle_AP": summary.get("ap"),
                        "oracle_GT_best_IoU_mean": summary.get("gt_best_iou_mean"),
                        "oracle_pred_best_IoU_median": summary.get("pred_best_iou_median"),
                        "broad_large_selected_rate": sum(1 for item in selected if item[2].broad_large_risk) / max(1, len(selected)),
                        "underseg_proxy_selected_rate": sum(1 for item in selected if item[2].underseg_proxy) / max(1, len(selected)),
                        "selected_mask_area_ratio_mean": _mean([item[2].area_ratio for item in selected]),
                        "uses_gt_for_prediction": True,
                        "forbidden_for_method_table": True,
                        "diagnostic_only": True,
                    }
                )
        if int(args.max_chunks) > 0 and processed >= int(args.max_chunks):
            break
    _write_csv(output_root / "oracle_budget_sweep_chunk_rows.csv", rows)
    summaries = []
    for budget in budgets:
        subset = [row for row in rows if int(row["budget"]) == int(budget)]
        summary = {
            "budget": int(budget),
            "chunk_count": len(subset),
            "diagnostic_only": True,
            "forbidden_for_method_table": True,
            "uses_gt_for_prediction": True,
            "means": {},
        }
        for key in [
            "candidate_pair_count",
            "selected_mask_count",
            "local_gt_count",
            "oracle_SF50",
            "oracle_AP50",
            "oracle_AP25",
            "oracle_AP",
            "oracle_GT_best_IoU_mean",
            "oracle_pred_best_IoU_median",
            "broad_large_selected_rate",
            "underseg_proxy_selected_rate",
            "selected_mask_area_ratio_mean",
        ]:
            summary["means"][key] = _mean([row.get(key) for row in subset])
        summaries.append(summary)
        (output_root / f"oracle_budget_{budget}_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    final = {
        "decision": "ORACLE_BUDGET_SWEEP_DIAGNOSTIC_DONE",
        "processed_chunk_count": processed,
        "budgets": budgets,
        "summaries": summaries,
    }
    (output_root / "oracle_budget_sweep_summary.json").write_text(json.dumps(final, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sha_rows = [
        {"path": _rel(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in sorted(output_root.glob("*"))
        if path.is_file()
    ]
    _write_csv(output_root / "sha256_rows.csv", sha_rows)
    return final


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", default=",".join(DEFAULT_SCENES))
    parser.add_argument("--budgets", default="64,80,96,112,128")
    parser.add_argument("--candidate-rows", default="outputs/audit/v71_candidate_bank/candidate_mask_rows.csv")
    parser.add_argument("--witness-summary", default="outputs/audit/v70_carrier_witness/witness_summary.json")
    parser.add_argument("--output-root", default="outputs/audit/v71_representative_setcover_oracle_budget_sweep")
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--max-chunks", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
