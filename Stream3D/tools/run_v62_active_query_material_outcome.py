from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(val) for val in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(_json_safe(row.get(key)), sort_keys=True)
                    if isinstance(row.get(key), (dict, list))
                    else row.get(key, "")
                    for key in keys
                }
            )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _bool_str(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def _mean_bool(values: list[bool]) -> float | None:
    if not values:
        return None
    return float(sum(1 for value in values if value) / len(values))


def _safe_rate(num: int, den: int) -> float | None:
    if den <= 0:
        return None
    return float(num / den)


def _material_rows_for_run(root: Path, args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary = _read_json(root / "v62_d4rt_smoke_summary.json")
    query_rows = _read_csv(root / "v62_d4rt_smoke_query_rows.csv")
    carrier = np.load(root / "carrier_batch_smoke.npz")
    valid = np.asarray(carrier["valid"], dtype=bool)
    uv_pred = np.asarray(carrier["uv_pred"], dtype=np.float32)
    visibility = np.asarray(carrier["visibility_prob"], dtype=np.float32)
    confidence = np.asarray(carrier["confidence_prob"], dtype=np.float32)
    accepted = valid & (visibility >= float(args.min_visibility)) & (confidence >= float(args.min_confidence))
    in_bounds = valid & (uv_pred[..., 0] >= 0.0) & (uv_pred[..., 0] <= 1.0) & (uv_pred[..., 1] >= 0.0) & (uv_pred[..., 1] <= 1.0)
    frame_count = int(valid.shape[0]) if valid.ndim == 2 else 0
    rows: list[dict[str, Any]] = []
    for idx, query in enumerate(query_rows):
        query_index = int(query.get("d4rt_query_index") or idx)
        accepted_count = int(np.count_nonzero(accepted[:, query_index])) if accepted.ndim == 2 else 0
        valid_count = int(np.count_nonzero(valid[:, query_index])) if valid.ndim == 2 else 0
        in_bounds_count = int(np.count_nonzero(in_bounds[:, query_index])) if in_bounds.ndim == 2 else 0
        accepted_ratio = _safe_rate(accepted_count, frame_count) or 0.0
        valid_ratio = _safe_rate(valid_count, frame_count) or 0.0
        in_bounds_ratio = _safe_rate(in_bounds_count, max(valid_count, 1)) or 0.0
        outside_residual_rate = 1.0 - in_bounds_ratio
        has_stable_track = accepted_count >= int(args.min_accepted_frames) and accepted_ratio >= float(args.min_accepted_ratio)
        valid_material = bool(has_stable_track and in_bounds_ratio >= float(args.min_in_bounds_ratio))
        is_shared_shortcut = query.get("candidate_source") == "shared_shortcut_boundary" or query.get("state") == "shared"
        is_confirmable = (
            query.get("candidate_source") in {"bridge_low_support", "update_new_low_support"}
            and query.get("state") == "confirmed"
            and _bool_str(query.get("has_K_mat"))
        )
        query_to_confirm = bool(valid_material and is_confirmable and accepted_ratio >= float(args.confirm_min_accepted_ratio))
        query_to_quarantine = bool(
            valid_material
            and not query_to_confirm
            and (
                is_shared_shortcut
                or outside_residual_rate >= float(args.quarantine_outside_residual_rate)
            )
        )
        same_frame_conflict_proxy = bool(is_shared_shortcut)
        rows.append(
            {
                **query,
                "run_name": root.name,
                "frame_count": frame_count,
                "valid_track_count": valid_count,
                "accepted_track_count": accepted_count,
                "accepted_track_ratio": accepted_ratio,
                "in_bounds_track_count": in_bounds_count,
                "in_bounds_valid_ratio": in_bounds_ratio,
                "outside_residual_rate": outside_residual_rate,
                "same_frame_conflict_proxy": same_frame_conflict_proxy,
                "valid_material_evidence": valid_material,
                "query_to_confirm": query_to_confirm,
                "query_to_quarantine": query_to_quarantine,
                "confirm_or_quarantine_outcome": bool(query_to_confirm or query_to_quarantine),
                "material_outcome_label": "confirm"
                if query_to_confirm
                else "quarantine"
                if query_to_quarantine
                else "unresolved",
                "outcome_rule": _rule_label(valid_material, query_to_confirm, query_to_quarantine, is_confirmable, is_shared_shortcut),
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic_labels": False,
            }
        )
    run_summary = _summarize_rows(rows)
    run_summary.update(
        {
            "run_name": root.name,
            "scene": summary.get("scene"),
            "query_budget": summary.get("query_budget"),
            "source_smoke_summary": str(root / "v62_d4rt_smoke_summary.json"),
            "carrier_batch_npz": str(root / "carrier_batch_smoke.npz"),
            "min_visibility": float(args.min_visibility),
            "min_confidence": float(args.min_confidence),
            "min_accepted_frames": int(args.min_accepted_frames),
            "min_accepted_ratio": float(args.min_accepted_ratio),
            "confirm_min_accepted_ratio": float(args.confirm_min_accepted_ratio),
            "min_in_bounds_ratio": float(args.min_in_bounds_ratio),
            "quarantine_outside_residual_rate": float(args.quarantine_outside_residual_rate),
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic_labels": False,
        }
    )
    return run_summary, rows


def _rule_label(
    valid_material: bool,
    query_to_confirm: bool,
    query_to_quarantine: bool,
    is_confirmable: bool,
    is_shared_shortcut: bool,
) -> str:
    if not valid_material:
        return "insufficient_stable_track"
    if query_to_confirm:
        return "confirmed_bridge_or_update_with_K_mat_and_stable_track"
    if query_to_quarantine:
        if is_shared_shortcut:
            return "stable_shared_shortcut_track_quarantine"
        return "stable_track_with_high_outside_residual_quarantine"
    if not is_confirmable and not is_shared_shortcut:
        return "valid_track_but_not_state_eligible"
    return "valid_track_unresolved_by_conservative_rule"


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    query_count = len(rows)
    valid = [bool(row["valid_material_evidence"]) for row in rows]
    confirm = [bool(row["query_to_confirm"]) for row in rows]
    quarantine = [bool(row["query_to_quarantine"]) for row in rows]
    confirm_or_quarantine = [bool(row["confirm_or_quarantine_outcome"]) for row in rows]
    outside = [float(row["outside_residual_rate"]) for row in rows]
    same_frame_proxy = [bool(row["same_frame_conflict_proxy"]) for row in rows]
    labels: dict[str, int] = {}
    sources: dict[str, int] = {}
    for row in rows:
        labels[row["material_outcome_label"]] = labels.get(row["material_outcome_label"], 0) + 1
        source = str(row.get("candidate_source", ""))
        sources[source] = sources.get(source, 0) + 1
    return {
        "query_count": query_count,
        "valid_material_evidence_rate": _mean_bool(valid),
        "query_to_confirm_rate": _mean_bool(confirm),
        "query_to_quarantine_rate": _mean_bool(quarantine),
        "query_to_confirm_or_quarantine_rate": _mean_bool(confirm_or_quarantine),
        "outside_residual_rate": float(np.mean(outside)) if outside else None,
        "same_frame_conflict_rate": None,
        "same_frame_conflict_proxy_rate": _mean_bool(same_frame_proxy),
        "real_minus_shuffled_query_AUC": None,
        "real_minus_no_temporal_query_AUC": None,
        "material_outcome_counts": labels,
        "candidate_source_counts": sources,
        "claim_status": "diagnostic_material_outcome_only_no_active_query_claim",
        "gate": {
            "valid_material_evidence_rate_ge_0_50": (_mean_bool(valid) or 0.0) >= 0.50,
            "confirm_or_quarantine_rate_available": _mean_bool(confirm_or_quarantine) is not None,
            "real_minus_shuffled_query_AUC_available": False,
            "real_minus_no_temporal_query_AUC_available": False,
            "pass": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-process v62 D4RT smoke outputs into material outcome diagnostics.")
    parser.add_argument("--smoke-root", action="append", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--min-visibility", type=float, default=0.5)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--min-accepted-frames", type=int, default=2)
    parser.add_argument("--min-accepted-ratio", type=float, default=0.0)
    parser.add_argument("--confirm-min-accepted-ratio", type=float, default=0.25)
    parser.add_argument("--min-in-bounds-ratio", type=float, default=0.80)
    parser.add_argument("--quarantine-outside-residual-rate", type=float, default=0.20)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    all_rows: list[dict[str, Any]] = []
    run_summaries: list[dict[str, Any]] = []
    for smoke_root in args.smoke_root:
        run_summary, rows = _material_rows_for_run(Path(smoke_root), args)
        run_summaries.append(run_summary)
        all_rows.extend(rows)
    combined_summary = _summarize_rows(all_rows)
    combined_summary.update(
        {
            "phase": "v62_active_query_material_outcome_diagnostic",
            "run_count": len(run_summaries),
            "run_summaries": run_summaries,
            "input_smoke_roots": [str(Path(root)) for root in args.smoke_root],
            "method_note": (
                "Conservative post-processing of real v62 D4RT carrier smoke. "
                "Confirm requires a stable track, confirmed bridge/update candidate, and K_mat. "
                "Quarantine requires a stable shared-shortcut track or high out-of-bounds residual. "
                "AUC and native AP are not computed here."
            ),
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic_labels": False,
            "ap_status": "not_run",
        }
    )
    _write_csv(output_root / "material_outcome_rows.csv", all_rows)
    _write_json(output_root / "material_outcome_summary.json", combined_summary)
    _write_csv(output_root / "material_outcome_run_summaries.csv", run_summaries)
    print(json.dumps(_json_safe(combined_summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
