from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from stream4d_native.v55_history_update import (
    ROOT,
    _dominant,
    _load_list,
    _semantic_mask_feature,
    _support_component_gt,
)
from stream4d_native.v47_common import parse_int, read_csv, write_csv, write_json


def _project(path: str | Path) -> Path:
    path_obj = Path(path)
    return path_obj if path_obj.is_absolute() else ROOT / path_obj


def _auc_lower_scores_for_positive(rows: list[dict[str, Any]], *, label_key: str, score_key: str) -> float | None:
    positives = [float(row[score_key]) for row in rows if row.get(label_key) is True and row.get(score_key) is not None]
    negatives = [float(row[score_key]) for row in rows if row.get(label_key) is False and row.get(score_key) is not None]
    if not positives or not negatives:
        return None
    wins = 0.0
    total = 0.0
    for pos_score in positives:
        for neg_score in negatives:
            total += 1.0
            if pos_score < neg_score:
                wins += 1.0
            elif pos_score == neg_score:
                wins += 0.5
    return float(wins / max(total, 1.0))


def _history_dominant_gt(
    anchor_birth_rows: list[dict[str, Any]],
    support_rows_path: Path,
    *,
    support_variant: str,
) -> dict[str, str]:
    scenes = {str(row.get("scene")) for row in anchor_birth_rows}
    component_gt = _support_component_gt(support_rows_path, support_variant=support_variant, scenes=scenes)
    history_gt: dict[str, str] = {}
    for row in anchor_birth_rows:
        if str(row.get("accepted_birth")).lower() != "true":
            continue
        scene = str(row.get("scene"))
        gt_counter: Counter[str] = Counter()
        for component_id in _load_list(row.get("component_ids")):
            gt_counter.update(component_gt.get((scene, component_id), Counter()))
        history_gt[str(row.get("birth_object_id"))] = _dominant(gt_counter) or ""
    return history_gt


def _mask_dominant_gt(
    support_rows: list[dict[str, Any]],
    *,
    support_variant: str,
) -> dict[str, str]:
    counters: dict[str, Counter[str]] = {}
    for row in support_rows:
        if str(row.get("variant")) != support_variant:
            continue
        mask_observation_id = str(row.get("mask_observation_id") or "")
        gt = str(row.get("diagnostic_gt_instance") or "")
        if not mask_observation_id or not gt or gt == "0":
            continue
        counters.setdefault(mask_observation_id, Counter())[gt] += max(parse_int(row.get("support_count")), 1)
    return {mask_id: (_dominant(counter) or "") for mask_id, counter in counters.items()}


def run(args: argparse.Namespace) -> dict[str, Any]:
    anchor_birth_rows = read_csv(_project(args.anchor_birth_rows))
    history_update_rows = read_csv(_project(args.history_update_rows))
    support_rows = read_csv(_project(args.support_rows))
    history_source_mask = {
        str(row.get("birth_object_id")): str(row.get("source_mask_observation_id") or "")
        for row in anchor_birth_rows
        if str(row.get("accepted_birth")).lower() == "true"
    }
    history_gt = _history_dominant_gt(anchor_birth_rows, _project(args.support_rows), support_variant=args.support_variant)
    mask_gt = _mask_dominant_gt(support_rows, support_variant=args.support_variant)

    adapter_cache: dict[str, Any] = {}
    feature_map_cache: dict[tuple[str, int, str], Any] = {}
    feature_cache: dict[tuple[str, str, str, int], tuple[list[float], dict[str, Any]]] = {}
    rows: list[dict[str, Any]] = []
    for update_row in history_update_rows:
        if str(update_row.get("update_source")) != "native_history_mask_projection":
            continue
        history_id = str(update_row.get("history_id") or "")
        mask_observation_id = str(update_row.get("candidate_id") or "")
        source_mask_observation_id = history_source_mask.get(history_id, "")
        current_feature, current_diag = _semantic_mask_feature(
            mask_observation_id,
            backend=args.backend,
            device=args.device,
            checkpoint=args.checkpoint,
            short_side=args.short_side,
            adapter_cache=adapter_cache,
            feature_map_cache=feature_map_cache,
            feature_cache=feature_cache,
        )
        source_feature, source_diag = _semantic_mask_feature(
            source_mask_observation_id,
            backend=args.backend,
            device=args.device,
            checkpoint=args.checkpoint,
            short_side=args.short_side,
            adapter_cache=adapter_cache,
            feature_map_cache=feature_map_cache,
            feature_cache=feature_cache,
        )
        feature_success = bool(current_feature and source_feature)
        cosine_to_anchor = None
        if feature_success:
            import numpy as np

            left = np.asarray(current_feature, dtype=np.float32)
            right = np.asarray(source_feature, dtype=np.float32)
            denom = float(np.linalg.norm(left) * np.linalg.norm(right))
            cosine_to_anchor = float(np.dot(left, right) / denom) if denom > 1e-12 else 0.0
        hist_gt = history_gt.get(history_id, "")
        cand_gt = mask_gt.get(mask_observation_id, "")
        confirmed_update = str(update_row.get("update_state")) == "confirmed_update"
        false_update_diagnostic = bool(confirmed_update and hist_gt and cand_gt and hist_gt != cand_gt)
        rows.append(
            {
                "scene": update_row.get("scene"),
                "chunk_id": update_row.get("chunk_id"),
                "history_id": history_id,
                "mask_observation_id": mask_observation_id,
                "source_mask_observation_id": source_mask_observation_id,
                "update_state": update_row.get("update_state"),
                "accepted_component_count": parse_int(update_row.get("accepted_component_count")),
                "candidate_component_count": parse_int(update_row.get("candidate_component_count")),
                "history_dominant_gt_diagnostic": hist_gt,
                "mask_dominant_gt_diagnostic": cand_gt,
                "false_update_diagnostic": false_update_diagnostic,
                "feature_success": feature_success,
                "cosine_to_anchor_source": cosine_to_anchor,
                "current_missing_reason": current_diag.get("semantic_feature_missing_reason"),
                "source_missing_reason": source_diag.get("semantic_feature_missing_reason"),
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic_labels": True,
            }
        )

    threshold_rows: list[dict[str, Any]] = []
    for threshold in args.thresholds:
        kept = [row for row in rows if row["feature_success"] and float(row["cosine_to_anchor_source"]) >= threshold]
        rejected = [row for row in rows if row["feature_success"] and float(row["cosine_to_anchor_source"]) < threshold]
        threshold_rows.append(
            {
                "threshold": float(threshold),
                "kept_count": len(kept),
                "rejected_count": len(rejected),
                "kept_false_update_count_diagnostic": sum(bool(row["false_update_diagnostic"]) for row in kept),
                "rejected_false_update_count_diagnostic": sum(bool(row["false_update_diagnostic"]) for row in rejected),
                "kept_accepted_component_count": sum(int(row["accepted_component_count"]) for row in kept),
                "rejected_accepted_component_count": sum(int(row["accepted_component_count"]) for row in rejected),
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic_labels": True,
            }
        )

    confirmed_rows = [row for row in rows if row["update_state"] == "confirmed_update"]
    summary = {
        "phase": "v55_semantic_memory_diagnostic",
        "backend": args.backend,
        "checkpoint": args.checkpoint,
        "device": args.device,
        "short_side": int(args.short_side),
        "history_update_rows": str(_project(args.history_update_rows).relative_to(ROOT)),
        "u6_row_count": len(rows),
        "feature_success_count": sum(bool(row["feature_success"]) for row in rows),
        "feature_success_rate": sum(bool(row["feature_success"]) for row in rows) / max(len(rows), 1),
        "confirmed_u6_row_count": len(confirmed_rows),
        "confirmed_feature_success_count": sum(bool(row["feature_success"]) for row in confirmed_rows),
        "confirmed_feature_success_rate": sum(bool(row["feature_success"]) for row in confirmed_rows)
        / max(len(confirmed_rows), 1),
        "false_update_count_diagnostic": sum(bool(row["false_update_diagnostic"]) for row in rows),
        "confirmed_false_update_count_diagnostic": sum(bool(row["false_update_diagnostic"]) for row in confirmed_rows),
        "semantic_drift_detection_AUC_diagnostic": _auc_lower_scores_for_positive(
            [row for row in rows if row["feature_success"]],
            label_key="false_update_diagnostic",
            score_key="cosine_to_anchor_source",
        ),
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }
    out = _project(args.output_root)
    write_json(out / "semantic_memory_summary.json", summary)
    write_csv(out / "semantic_drift_rows.csv", rows)
    write_csv(out / "semantic_threshold_rows.csv", threshold_rows)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnostic-only DINO semantic drift audit for v55 U6 rows.")
    parser.add_argument("--output-root", default="outputs/audit/v55_semantic_memory_diagnostic_dinov2_scripted")
    parser.add_argument(
        "--history-update-rows",
        default=(
            "outputs/audit/v55_history_update_repair_native_boundary_uv_historymask_s100_r09_d15_plus_"
            "cosupport_seed038_fixed/history_update_rows.csv"
        ),
    )
    parser.add_argument("--anchor-birth-rows", default="outputs/audit/v55_anchor_birth/anchor_birth_rows.csv")
    parser.add_argument(
        "--support-rows",
        default="outputs/audit/v54_mask_component_support_tau005_stride1_probe5_q4096_notopup/mask_component_support_rows.csv",
    )
    parser.add_argument("--support-variant", default="R0_visible_tau0.05")
    parser.add_argument("--backend", default="dinov2_timm", choices=["dinov2_timm", "colorhist"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--short-side", type=int, default=518)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.90, 0.92, 0.94, 0.96, 0.97, 0.98, 0.985, 0.99, 0.995],
    )
    args = parser.parse_args()
    summary = run(args)
    print(summary)


if __name__ == "__main__":
    main()
