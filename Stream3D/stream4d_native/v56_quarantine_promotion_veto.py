from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from .v47_common import ROOT, parse_bool, parse_float, read_csv, utc_now, write_csv, write_json


def _project(path: str | Path) -> Path:
    path_obj = Path(path)
    return path_obj if path_obj.is_absolute() else ROOT / path_obj


def _rel(path: str | Path) -> str:
    path_obj = _project(path)
    try:
        return str(path_obj.relative_to(ROOT))
    except ValueError:
        return str(path_obj)


def _is_false_promotion(row: dict[str, Any]) -> bool:
    history_gt = str(row.get("history_dominant_gt_diagnostic") or "")
    component_gt = str(row.get("component_dominant_gt_diagnostic") or "")
    return bool(history_gt and component_gt and history_gt != component_gt)


def _promoted_rows(path: Path) -> list[dict[str, Any]]:
    return [row for row in read_csv(path) if str(row.get("promotion_state")) == "promoted"]


def _history_quarantine_scores(
    quarantine_rows: list[dict[str, Any]],
    *,
    same_anchor_only: bool,
    min_shared_min_ratio: float,
) -> dict[str, float]:
    scores: dict[str, float] = defaultdict(float)
    for row in quarantine_rows:
        if same_anchor_only and not parse_bool(row.get("same_anchor_chunk")):
            continue
        ratio = parse_float(row.get("shared_min_ratio"))
        if ratio < float(min_shared_min_ratio):
            continue
        for key in ("left_history_id", "right_history_id"):
            history_id = str(row.get(key) or "")
            if history_id:
                scores[history_id] = max(scores[history_id], ratio)
    return dict(scores)


def build_v56_quarantine_promotion_veto(
    *,
    promotion_rows_path: str | Path,
    quarantine_rows_path: str | Path = "outputs/audit/v56_quarantine/quarantine_rows.csv",
    ratio_thresholds: tuple[float, ...] = (0.0, 0.02, 0.04, 0.08, 0.12),
    same_anchor_only: bool = True,
) -> dict[str, Any]:
    promoted_rows = _promoted_rows(_project(promotion_rows_path))
    quarantine_rows = read_csv(_project(quarantine_rows_path))
    total_promoted = len(promoted_rows)
    false_promoted = sum(1 for row in promoted_rows if _is_false_promotion(row))
    true_promoted = total_promoted - false_promoted
    rows: list[dict[str, Any]] = []
    best_safe_row: dict[str, Any] | None = None
    for threshold in ratio_thresholds:
        quarantine_score_by_history = _history_quarantine_scores(
            quarantine_rows,
            same_anchor_only=same_anchor_only,
            min_shared_min_ratio=float(threshold),
        )
        vetoed = [row for row in promoted_rows if str(row.get("history_id")) in quarantine_score_by_history]
        false_vetoed = sum(1 for row in vetoed if _is_false_promotion(row))
        true_vetoed = len(vetoed) - false_vetoed
        remaining = total_promoted - len(vetoed)
        remaining_false = false_promoted - false_vetoed
        remaining_true = true_promoted - true_vetoed
        precision_after_veto = (remaining_true / remaining) if remaining else None
        false_promotion_reduction = (false_vetoed / false_promoted) if false_promoted else None
        expanded_completeness_drop_proxy = (len(vetoed) / total_promoted) if total_promoted else None
        row = {
            "variant": f"same_anchor_min_shared_ratio_{threshold:g}" if same_anchor_only else f"any_min_shared_ratio_{threshold:g}",
            "same_anchor_only": bool(same_anchor_only),
            "min_shared_min_ratio": float(threshold),
            "promoted_component_count": total_promoted,
            "false_promotion_count": false_promoted,
            "vetoed_promotion_count": len(vetoed),
            "false_vetoed_count": false_vetoed,
            "true_vetoed_count": true_vetoed,
            "remaining_promotion_count": remaining,
            "remaining_false_promotion_count": remaining_false,
            "precision_after_veto_diagnostic": precision_after_veto,
            "false_promotion_reduction": false_promotion_reduction,
            "expanded_completeness_drop_proxy": expanded_completeness_drop_proxy,
            "gate_false_promotion_reduction_ge_0.10": false_promotion_reduction is not None
            and false_promotion_reduction >= 0.10,
            "gate_expanded_completeness_drop_le_0.04": expanded_completeness_drop_proxy is not None
            and expanded_completeness_drop_proxy <= 0.04,
            "gate_pass": bool(
                false_promotion_reduction is not None
                and expanded_completeness_drop_proxy is not None
                and false_promotion_reduction >= 0.10
                and expanded_completeness_drop_proxy <= 0.04
            ),
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic_labels": True,
        }
        if row["gate_pass"] and best_safe_row is None:
            best_safe_row = row
        rows.append(row)

    best_reduction_row = max(
        rows,
        key=lambda row: (
            float(row["false_promotion_reduction"] or 0.0),
            -float(row["expanded_completeness_drop_proxy"] or 0.0),
        ),
        default=None,
    )
    summary = {
        "phase": "v56_quarantine_promotion_veto",
        "created_at": utc_now(),
        "input_paths": {
            "promotion_rows_path": _rel(promotion_rows_path),
            "quarantine_rows_path": _rel(quarantine_rows_path),
        },
        "same_anchor_only": bool(same_anchor_only),
        "ratio_thresholds": [float(value) for value in ratio_thresholds],
        "promoted_component_count": total_promoted,
        "false_promotion_count": false_promoted,
        "true_promotion_count": true_promoted,
        "quarantine_pair_row_count": len(quarantine_rows),
        "best_safe_variant": best_safe_row["variant"] if best_safe_row else None,
        "best_reduction_variant": best_reduction_row["variant"] if best_reduction_row else None,
        "best_reduction_false_promotion_reduction": best_reduction_row["false_promotion_reduction"]
        if best_reduction_row
        else None,
        "best_reduction_expanded_completeness_drop_proxy": best_reduction_row["expanded_completeness_drop_proxy"]
        if best_reduction_row
        else None,
        "gate": {
            "false_promotion_reduction_ge_0.10_and_drop_le_0.04": bool(best_safe_row),
            "pass": bool(best_safe_row),
        },
        "diagnostic_status": (
            "history-pair quarantine veto is too coarse for safe promotion"
            if not best_safe_row
            else "history-pair quarantine veto has a safe diagnostic operating point"
        ),
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }
    return {"summary": summary, "veto_metric_rows": rows}


def write_v56_quarantine_promotion_veto(output_root: str | Path, payload: dict[str, Any]) -> None:
    out = _project(output_root)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "quarantine_promotion_veto_summary.json", payload["summary"])
    write_csv(out / "quarantine_promotion_veto_rows.csv", payload["veto_metric_rows"])

