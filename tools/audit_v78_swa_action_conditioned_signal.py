#!/usr/bin/env python3
"""Audit action-conditioned SWA signals for v78 Phase9 head/selectors.

This is diagnostic-only.  It joins already-landed Phase9 artifacts:
metrics CSVs, hook attention-mass summaries, boundary-residual audits, and
SWA overlap feature dumps.  The goal is to separate three effects that were
previously conflated:

* scene-level geometry/visibility risk,
* source-quality of the selected top-q tokens,
* action conditioning such as head coverage and attention-mass lift.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import torch


REPORT_ROOT = Path(
    "results/kitti01_hmc_v2/acl2_v78tf_pca_grounded_memory_control/report_final"
)
REPORT_ROOT_KITTI02 = Path(
    "results/kitti02_hmc_v2/acl2_v78tf_pca_grounded_memory_control/report_final"
)
PHASE9_ROOT = REPORT_ROOT / "phase9_swa_cache_value_carryover"
PHASE9_ROOT_KITTI02 = REPORT_ROOT_KITTI02 / "phase9_swa_cache_value_carryover"
DEFAULT_SCENE_SCORE_CSV = (
    REPORT_ROOT
    / "bad_good_case_contrast/v2_unique_scenes_top5/boundary_local_score_audit/boundary_local_score_rows.csv"
)


SUITES: list[dict[str, Any]] = [
    {
        "suite": "KITTI01_chunk06_P9_34_all_heads_topq80",
        "sequence": "01",
        "chunk": 6,
        "action": "P9_34",
        "action_label": "weak_positive_boundary",
        "window_key": "01:145:174:206",
        "head_indices": "all",
        "metrics_csv": PHASE9_ROOT
        / "smoke_chunk06_context2_v20_topq80_bias_per_head_summary/phase9_swa_cache_value_metrics.csv",
        "boundary_json": PHASE9_ROOT / "boundary_residual_audit_v1/p9_34_boundary_residuals.json",
        "candidate_run": "P9_34_ATTENTION_BIAS_STABLE_AGREEMENT_TOPQ80_MASS_AUDIT_LAST",
        "control_run": "P9_35_ATTENTION_BIAS_STABLE_AGREEMENT_TOPQ80_RANDOM_SAME_MASS_MASS_AUDIT_LAST",
        "baseline_run": "P9_0_NATIVE",
        "feature_dump": PHASE9_ROOT
        / "smoke_chunk06_context2_v20_topq80_bias_per_head_summary/chunk06/"
        / "P9_34_ATTENTION_BIAS_STABLE_AGREEMENT_TOPQ80_MASS_AUDIT_LAST/"
        / "swa_overlap_feature_maps/chunk_006_swa_overlap_source_bias_geometric_layer_03.pt",
    },
    {
        "suite": "KITTI01_chunk06_P9_36_head6_topq80",
        "sequence": "01",
        "chunk": 6,
        "action": "P9_36",
        "action_label": "weak_negative_default",
        "window_key": "01:145:174:206",
        "head_indices": "6",
        "metrics_csv": PHASE9_ROOT / "smoke_chunk06_context2_v21_head6_bias/phase9_swa_cache_value_metrics.csv",
        "boundary_json": PHASE9_ROOT / "boundary_residual_audit_v1/p9_36_chunk06_head6_boundary_residuals.json",
        "candidate_run": "P9_36_ATTENTION_BIAS_STABLE_AGREEMENT_TOPQ80_HEAD6_LAST",
        "control_run": "P9_37_ATTENTION_BIAS_STABLE_AGREEMENT_TOPQ80_RANDOM_SAME_MASS_HEAD6_LAST",
        "baseline_run": "P9_0_NATIVE",
        "feature_dump": PHASE9_ROOT
        / "smoke_chunk06_context2_v21_head6_bias/chunk06/"
        / "P9_36_ATTENTION_BIAS_STABLE_AGREEMENT_TOPQ80_HEAD6_LAST/"
        / "swa_overlap_feature_maps/chunk_006_swa_overlap_source_bias_geometric_layer_03.pt",
    },
    {
        "suite": "KITTI01_chunk06_P9_38_heads0_6_8_topq80",
        "sequence": "01",
        "chunk": 6,
        "action": "P9_38",
        "action_label": "weak_negative_overlap",
        "window_key": "01:145:174:206",
        "head_indices": "0,6,8",
        "metrics_csv": PHASE9_ROOT / "smoke_chunk06_context2_v22_heads0_6_8_bias/phase9_swa_cache_value_metrics.csv",
        "boundary_json": PHASE9_ROOT / "boundary_residual_audit_v1/p9_38_chunk06_heads0_6_8_boundary_residuals.json",
        "candidate_run": "P9_38_ATTENTION_BIAS_STABLE_AGREEMENT_TOPQ80_HEADS0_6_8_LAST",
        "control_run": "P9_39_ATTENTION_BIAS_STABLE_AGREEMENT_TOPQ80_RANDOM_SAME_MASS_HEADS0_6_8_LAST",
        "baseline_run": "P9_0_NATIVE",
        "feature_dump": PHASE9_ROOT
        / "smoke_chunk06_context2_v22_heads0_6_8_bias/chunk06/"
        / "P9_38_ATTENTION_BIAS_STABLE_AGREEMENT_TOPQ80_HEADS0_6_8_LAST/"
        / "swa_overlap_feature_maps/chunk_006_swa_overlap_source_bias_geometric_layer_03.pt",
    },
    {
        "suite": "KITTI02_chunk14_P9_34_all_heads_topq80",
        "sequence": "02",
        "chunk": 14,
        "action": "P9_34",
        "action_label": "weak_negative_boundary",
        "window_key": "02:377:406:438",
        "head_indices": "all",
        "metrics_csv": PHASE9_ROOT_KITTI02
        / "smoke_chunk14_context2_topbadpair13_14_p9_34_v3_per_head_summary/"
        / "phase9_swa_cache_value_metrics.csv",
        "boundary_json": PHASE9_ROOT_KITTI02 / "boundary_residual_audit_v1/p9_34_chunk14_boundary_residuals.json",
        "candidate_run": "P9_34_ATTENTION_BIAS_STABLE_AGREEMENT_TOPQ80_MASS_AUDIT_LAST",
        "control_run": "P9_35_ATTENTION_BIAS_STABLE_AGREEMENT_TOPQ80_RANDOM_SAME_MASS_MASS_AUDIT_LAST",
        "baseline_run": "P9_0_NATIVE",
        "feature_dump": PHASE9_ROOT_KITTI02
        / "smoke_chunk14_context2_topbadpair13_14_p9_34_v3_per_head_summary/chunk14/"
        / "P9_34_ATTENTION_BIAS_STABLE_AGREEMENT_TOPQ80_MASS_AUDIT_LAST/"
        / "swa_overlap_feature_maps/chunk_014_swa_overlap_source_bias_geometric_layer_03.pt",
    },
    {
        "suite": "KITTI02_chunk14_P9_36_head6_topq80",
        "sequence": "02",
        "chunk": 14,
        "action": "P9_36",
        "action_label": "weak_positive_default",
        "window_key": "02:377:406:438",
        "head_indices": "6",
        "metrics_csv": PHASE9_ROOT_KITTI02
        / "smoke_chunk14_context2_topbadpair13_14_p9_36_head6_v4/phase9_swa_cache_value_metrics.csv",
        "boundary_json": PHASE9_ROOT_KITTI02
        / "boundary_residual_audit_v1/p9_36_chunk14_head6_boundary_residuals.json",
        "candidate_run": "P9_36_ATTENTION_BIAS_STABLE_AGREEMENT_TOPQ80_HEAD6_LAST",
        "control_run": "P9_37_ATTENTION_BIAS_STABLE_AGREEMENT_TOPQ80_RANDOM_SAME_MASS_HEAD6_LAST",
        "baseline_run": "P9_0_NATIVE",
        "feature_dump": PHASE9_ROOT_KITTI02
        / "smoke_chunk14_context2_topbadpair13_14_p9_36_head6_v4/chunk14/"
        / "P9_36_ATTENTION_BIAS_STABLE_AGREEMENT_TOPQ80_HEAD6_LAST/"
        / "swa_overlap_feature_maps/chunk_014_swa_overlap_source_bias_geometric_layer_03.pt",
    },
    {
        "suite": "KITTI02_chunk14_P9_38_heads0_6_8_topq80",
        "sequence": "02",
        "chunk": 14,
        "action": "P9_38",
        "action_label": "weak_negative_overlap",
        "window_key": "02:377:406:438",
        "head_indices": "0,6,8",
        "metrics_csv": PHASE9_ROOT_KITTI02
        / "smoke_chunk14_context2_topbadpair13_14_p9_38_heads0_6_8_v5/"
        / "phase9_swa_cache_value_metrics.csv",
        "boundary_json": PHASE9_ROOT_KITTI02
        / "boundary_residual_audit_v1/p9_38_chunk14_heads0_6_8_boundary_residuals.json",
        "candidate_run": "P9_38_ATTENTION_BIAS_STABLE_AGREEMENT_TOPQ80_HEADS0_6_8_LAST",
        "control_run": "P9_39_ATTENTION_BIAS_STABLE_AGREEMENT_TOPQ80_RANDOM_SAME_MASS_HEADS0_6_8_LAST",
        "baseline_run": "P9_0_NATIVE",
        "feature_dump": PHASE9_ROOT_KITTI02
        / "smoke_chunk14_context2_topbadpair13_14_p9_38_heads0_6_8_v5/chunk14/"
        / "P9_38_ATTENTION_BIAS_STABLE_AGREEMENT_TOPQ80_HEADS0_6_8_LAST/"
        / "swa_overlap_feature_maps/chunk_014_swa_overlap_source_bias_geometric_layer_03.pt",
    },
    {
        "suite": "KITTI02_chunk18_P9_34_all_heads_topq80",
        "sequence": "02",
        "chunk": 18,
        "action": "P9_34",
        "action_label": "heldout_weak_boundary_only",
        "window_key": "02:493:522:554",
        "head_indices": "all",
        "metrics_csv": PHASE9_ROOT_KITTI02
        / "smoke_chunk18_context2_topbadpair17_18_p9_34_36_38_v1/phase9_swa_cache_value_metrics.csv",
        "boundary_json": PHASE9_ROOT_KITTI02 / "boundary_residual_audit_v1/p9_34_chunk18_boundary_residuals.json",
        "candidate_run": "P9_34_ATTENTION_BIAS_STABLE_AGREEMENT_TOPQ80_MASS_AUDIT_LAST",
        "control_run": "P9_35_ATTENTION_BIAS_STABLE_AGREEMENT_TOPQ80_RANDOM_SAME_MASS_MASS_AUDIT_LAST",
        "baseline_run": "P9_0_NATIVE",
        "feature_dump": PHASE9_ROOT_KITTI02
        / "smoke_chunk18_context2_topbadpair17_18_p9_34_36_38_v1/chunk18/"
        / "P9_34_ATTENTION_BIAS_STABLE_AGREEMENT_TOPQ80_MASS_AUDIT_LAST/"
        / "swa_overlap_feature_maps/chunk_018_swa_overlap_source_bias_geometric_layer_03.pt",
    },
    {
        "suite": "KITTI02_chunk18_P9_36_head6_topq80",
        "sequence": "02",
        "chunk": 18,
        "action": "P9_36",
        "action_label": "heldout_weak_negative_head6",
        "window_key": "02:493:522:554",
        "head_indices": "6",
        "metrics_csv": PHASE9_ROOT_KITTI02
        / "smoke_chunk18_context2_topbadpair17_18_p9_34_36_38_v1/phase9_swa_cache_value_metrics.csv",
        "boundary_json": PHASE9_ROOT_KITTI02
        / "boundary_residual_audit_v1/p9_36_chunk18_head6_boundary_residuals.json",
        "candidate_run": "P9_36_ATTENTION_BIAS_STABLE_AGREEMENT_TOPQ80_HEAD6_LAST",
        "control_run": "P9_37_ATTENTION_BIAS_STABLE_AGREEMENT_TOPQ80_RANDOM_SAME_MASS_HEAD6_LAST",
        "baseline_run": "P9_0_NATIVE",
        "feature_dump": PHASE9_ROOT_KITTI02
        / "smoke_chunk18_context2_topbadpair17_18_p9_34_36_38_v1/chunk18/"
        / "P9_36_ATTENTION_BIAS_STABLE_AGREEMENT_TOPQ80_HEAD6_LAST/"
        / "swa_overlap_feature_maps/chunk_018_swa_overlap_source_bias_geometric_layer_03.pt",
    },
    {
        "suite": "KITTI02_chunk18_P9_38_heads0_6_8_topq80",
        "sequence": "02",
        "chunk": 18,
        "action": "P9_38",
        "action_label": "heldout_weak_negative_overlap",
        "window_key": "02:493:522:554",
        "head_indices": "0,6,8",
        "metrics_csv": PHASE9_ROOT_KITTI02
        / "smoke_chunk18_context2_topbadpair17_18_p9_34_36_38_v1/phase9_swa_cache_value_metrics.csv",
        "boundary_json": PHASE9_ROOT_KITTI02
        / "boundary_residual_audit_v1/p9_38_chunk18_heads0_6_8_boundary_residuals.json",
        "candidate_run": "P9_38_ATTENTION_BIAS_STABLE_AGREEMENT_TOPQ80_HEADS0_6_8_LAST",
        "control_run": "P9_39_ATTENTION_BIAS_STABLE_AGREEMENT_TOPQ80_RANDOM_SAME_MASS_HEADS0_6_8_LAST",
        "baseline_run": "P9_0_NATIVE",
        "feature_dump": PHASE9_ROOT_KITTI02
        / "smoke_chunk18_context2_topbadpair17_18_p9_34_36_38_v1/chunk18/"
        / "P9_38_ATTENTION_BIAS_STABLE_AGREEMENT_TOPQ80_HEADS0_6_8_LAST/"
        / "swa_overlap_feature_maps/chunk_018_swa_overlap_source_bias_geometric_layer_03.pt",
    },
]


LOWER_IS_BETTER_METRICS = [
    "local_sim3_ate_rmse_m",
    "head10_to_tail10_pose_sim3_rmse_m",
    "overlap3_to_future_pose_sim3_rmse_m",
    "scale_cv_head_mid_tail_pose_sim3",
]


BOUNDARY_METRICS = [
    "global_query_head_rmse_m",
    "global_future_from_boundary_rmse_m",
    "global_after_head_future_rmse_m",
    "boundary_step_error_m",
    "tail3_to_head3_sim3_rmse_m",
    "tail3_to_future_from_boundary_sim3_rmse_m",
    "tail3_to_after_head_future_sim3_rmse_m",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--scene-score-csv", type=Path, default=DEFAULT_SCENE_SCORE_CSV)
    return parser.parse_args()


def _finite(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(row.get(key), ensure_ascii=True)
                if isinstance(row.get(key), (list, dict))
                else row.get(key)
                for key in fieldnames
            })


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def _metric_rows(path: Path) -> dict[str, dict[str, str]]:
    rows = _read_csv(path)
    return {row.get("run", ""): row for row in rows}


def _scene_scores(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    for row in _read_csv(path):
        key = row.get("window_key") or ""
        score = _finite(row.get("boundary_local_score"))
        if score is None:
            score = _finite(row.get("score"))
        if key and score is not None:
            out[key] = score
    return out


def _head_count(text: str) -> int:
    if text == "all":
        return 16
    if not text:
        return 0
    return len([part for part in text.split(",") if part.strip() != ""])


def _load_boundary(path: Path, action: str) -> dict[str, Any]:
    if not path.is_file():
        return {"available": False}
    payload = json.loads(path.read_text(encoding="utf-8"))
    comp = (payload.get("comparisons") or {}).get(action) or {}
    beats = []
    rows: dict[str, Any] = {}
    for metric in BOUNDARY_METRICS:
        item = comp.get(metric) or {}
        rows[metric] = item
        if item.get("beats_controls") is True:
            beats.append(metric)
    return {
        "available": True,
        "beats_control_count": len(beats),
        "metrics_count": len([m for m in BOUNDARY_METRICS if m in comp]),
        "beats_control_metrics": beats,
        "metrics": rows,
    }


def _load_feature_quality(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"available": False}
    obj = torch.load(path, map_location="cpu")
    score = obj.get("score_overlap")
    if not torch.is_tensor(score) or score.numel() == 0:
        return {"available": False, "reason": "missing_score_overlap"}
    score_f = score.detach().cpu().float().reshape(-1)
    threshold = torch.quantile(score_f, 0.8)
    selected = score_f >= threshold
    selected_count = int(selected.sum().item())
    selected_values = score_f[selected]
    unselected_values = score_f[~selected]

    frame_rows = []
    score_by_frame = obj.get("score_overlap")
    if torch.is_tensor(score_by_frame) and score_by_frame.ndim == 3:
        per_frame = score_by_frame.detach().cpu().float()[0]
        for frame_idx in range(int(per_frame.shape[0])):
            vals = per_frame[frame_idx].reshape(-1)
            mask = vals >= threshold
            frame_rows.append({
                "overlap_frame_index": frame_idx,
                "selected_fraction": float(mask.float().mean().item()),
                "selected_count": int(mask.sum().item()),
                "score_mean": float(vals.mean().item()),
                "selected_score_mean": float(vals[mask].mean().item()) if bool(mask.any()) else None,
            })

    mean_all = float(score_f.mean().item())
    selected_mean = float(selected_values.mean().item()) if selected_count else None
    unselected_mean = float(unselected_values.mean().item()) if int((~selected).sum().item()) else None
    return {
        "available": True,
        "feature_dump": str(path),
        "schema": obj.get("schema"),
        "mode": obj.get("mode"),
        "runtime_swa_overlap_feature_not_qk_proxy": bool(obj.get("runtime_swa_overlap_feature_not_qk_proxy", False)),
        "score_mean": mean_all,
        "score_q80_threshold": float(threshold.item()),
        "score_q90": float(torch.quantile(score_f, 0.9).item()),
        "selected_count_topq80_reconstructed": selected_count,
        "selected_fraction_topq80_reconstructed": float(selected.float().mean().item()),
        "selected_score_mean_topq80": selected_mean,
        "unselected_score_mean": unselected_mean,
        "selected_quality_lift_vs_all_mean": selected_mean - mean_all if selected_mean is not None else None,
        "random_same_mass_expected_score_mean": mean_all,
        "per_overlap_frame": frame_rows,
    }


def _delta(cand: dict[str, str], other: dict[str, str], metric: str) -> float | None:
    c = _finite(cand.get(metric))
    o = _finite(other.get(metric))
    return c - o if c is not None and o is not None else None


def _beats(cand: dict[str, str], other: dict[str, str], metric: str) -> bool | None:
    diff = _delta(cand, other, metric)
    return diff < 0.0 if diff is not None else None


def _extract_attention(row: dict[str, str]) -> dict[str, Any]:
    keys = [
        "phase9_swa_attention_mass_available_frac",
        "phase9_swa_attention_mass_selected_before",
        "phase9_swa_attention_mass_selected_after",
        "phase9_swa_attention_mass_selected_lift",
        "phase9_swa_attention_mass_source_before",
        "phase9_swa_attention_mass_source_after",
        "phase9_swa_attention_mass_source_lift",
        "phase9_swa_attention_mass_selected_head_max_before",
        "phase9_swa_attention_mass_selected_head_max_after",
        "phase9_swa_attention_mass_selected_head_max_lift",
        "phase9_swa_overlap_bias_applied_sum",
        "phase9_swa_overlap_bias_mean_abs",
        "phase9_swa_overlap_bias_max_abs",
    ]
    return {key: _finite(row.get(key)) for key in keys}


def main() -> None:
    args = parse_args()
    scene_scores = _scene_scores(args.scene_score_csv)
    row_out: list[dict[str, Any]] = []
    pair_out: list[dict[str, Any]] = []
    limitations: list[str] = []

    for suite in SUITES:
        metrics = _metric_rows(Path(suite["metrics_csv"]))
        cand = metrics.get(str(suite["candidate_run"]), {})
        ctrl = metrics.get(str(suite["control_run"]), {})
        base = metrics.get(str(suite["baseline_run"]), {})
        if not cand or not ctrl or not base:
            limitations.append(f"{suite['suite']}: missing metrics rows")

        quality = _load_feature_quality(Path(suite["feature_dump"]))
        if quality.get("runtime_swa_overlap_feature_not_qk_proxy"):
            limitations.append(
                f"{suite['suite']}: feature dump is runtime SWA overlap score, not direct Q/K/V tensor alignment"
            )

        boundary = _load_boundary(Path(suite["boundary_json"]), str(suite["action"]))
        official_beats_control = [
            metric for metric in LOWER_IS_BETTER_METRICS if _beats(cand, ctrl, metric) is True
        ]
        official_beats_baseline = [
            metric for metric in LOWER_IS_BETTER_METRICS if _beats(cand, base, metric) is True
        ]
        attention_cand = _extract_attention(cand)
        attention_ctrl = _extract_attention(ctrl)

        flat = {
            "suite": suite["suite"],
            "sequence": suite["sequence"],
            "chunk": suite["chunk"],
            "action": suite["action"],
            "action_label": suite["action_label"],
            "window_key": suite["window_key"],
            "scene_geometry_score": scene_scores.get(str(suite["window_key"])),
            "head_indices": suite["head_indices"],
            "head_count": _head_count(str(suite["head_indices"])),
            "candidate_run": suite["candidate_run"],
            "control_run": suite["control_run"],
            "metrics_csv": str(suite["metrics_csv"]),
            "boundary_json": str(suite["boundary_json"]),
            "feature_dump_available": bool(quality.get("available")),
            "feature_runtime_not_qk_proxy": bool(quality.get("runtime_swa_overlap_feature_not_qk_proxy", False)),
            "selected_score_mean_topq80": quality.get("selected_score_mean_topq80"),
            "random_same_mass_expected_score_mean": quality.get("random_same_mass_expected_score_mean"),
            "selected_quality_lift_vs_all_mean": quality.get("selected_quality_lift_vs_all_mean"),
            "selected_fraction_topq80_reconstructed": quality.get("selected_fraction_topq80_reconstructed"),
            "attention_selected_lift_candidate": attention_cand.get("phase9_swa_attention_mass_selected_lift"),
            "attention_selected_lift_control": attention_ctrl.get("phase9_swa_attention_mass_selected_lift"),
            "attention_selected_lift_candidate_minus_control": (
                attention_cand.get("phase9_swa_attention_mass_selected_lift")
                - attention_ctrl.get("phase9_swa_attention_mass_selected_lift")
                if attention_cand.get("phase9_swa_attention_mass_selected_lift") is not None
                and attention_ctrl.get("phase9_swa_attention_mass_selected_lift") is not None
                else None
            ),
            "attention_source_lift_candidate": attention_cand.get("phase9_swa_attention_mass_source_lift"),
            "attention_source_lift_control": attention_ctrl.get("phase9_swa_attention_mass_source_lift"),
            "attention_source_lift_candidate_minus_control": (
                attention_cand.get("phase9_swa_attention_mass_source_lift")
                - attention_ctrl.get("phase9_swa_attention_mass_source_lift")
                if attention_cand.get("phase9_swa_attention_mass_source_lift") is not None
                and attention_ctrl.get("phase9_swa_attention_mass_source_lift") is not None
                else None
            ),
            "official_metrics_beats_control_count": len(official_beats_control),
            "official_metrics_count": len(LOWER_IS_BETTER_METRICS),
            "official_metrics_beats_control": official_beats_control,
            "official_metrics_beats_baseline_count": len(official_beats_baseline),
            "official_metrics_beats_baseline": official_beats_baseline,
            "boundary_beats_control_count": boundary.get("beats_control_count"),
            "boundary_metrics_count": boundary.get("metrics_count"),
            "boundary_beats_control_metrics": boundary.get("beats_control_metrics"),
        }
        for metric in LOWER_IS_BETTER_METRICS:
            flat[f"{metric}_candidate"] = _finite(cand.get(metric))
            flat[f"{metric}_control"] = _finite(ctrl.get(metric))
            flat[f"{metric}_baseline"] = _finite(base.get(metric))
            flat[f"{metric}_cand_minus_control"] = _delta(cand, ctrl, metric)
            flat[f"{metric}_cand_minus_baseline"] = _delta(cand, base, metric)
        row_out.append(flat)

        pair_out.append({
            "suite": suite["suite"],
            "sequence": suite["sequence"],
            "chunk": suite["chunk"],
            "action": suite["action"],
            "action_label": suite["action_label"],
            "head_count": flat["head_count"],
            "scene_geometry_score": flat["scene_geometry_score"],
            "selected_quality_lift_vs_all_mean": flat["selected_quality_lift_vs_all_mean"],
            "attention_selected_lift_candidate_minus_control": flat["attention_selected_lift_candidate_minus_control"],
            "official_metrics_beats_control_count": len(official_beats_control),
            "official_metrics_count": len(LOWER_IS_BETTER_METRICS),
            "boundary_beats_control_count": boundary.get("beats_control_count"),
            "boundary_metrics_count": boundary.get("metrics_count"),
            "interpretation": "",
        })

    if any(row.get("attention_selected_lift_candidate_minus_control", 0) is not None and row.get("attention_selected_lift_candidate_minus_control", 0) < 0 for row in row_out):
        limitations.append(
            "raw attention-mass lift is not sufficient: a candidate can beat control while lifting less selected mass"
        )
    limitations.append(
        "random same-mass masks were not materialized; random source quality is estimated by score-map mean"
    )

    summary = {
        "schema": "acl2_v78_swa_action_conditioned_signal_audit_v2",
        "diagnostic_only": True,
        "method_gate_claimed": False,
        "scene_score_csv": str(args.scene_score_csv),
        "out_dir": str(args.out_dir),
        "num_suites": len(SUITES),
        "rows": row_out,
        "key_findings": {
            "scene_score_by_window_key": {
                key: sorted({row.get("scene_geometry_score") for row in row_out if row.get("window_key") == key})
                for key in sorted({str(row.get("window_key")) for row in row_out})
            },
            "official_beats_control_by_suite": {
                row["suite"]: row["official_metrics_beats_control_count"] for row in row_out
            },
            "boundary_beats_control_by_suite": {
                row["suite"]: row["boundary_beats_control_count"] for row in row_out
            },
            "attention_selected_lift_candidate_minus_control_by_suite": {
                row["suite"]: row["attention_selected_lift_candidate_minus_control"] for row in row_out
            },
            "selected_quality_lift_vs_all_mean_by_suite": {
                row["suite"]: row["selected_quality_lift_vs_all_mean"] for row in row_out
            },
        },
        "interpretation": [
            "Scene-level geometry score is constant within a fixed window and cannot choose among P9 actions by itself.",
            "Reconstructed topq80 source quality is high relative to random expectation, but this alone is also not sufficient.",
            "KITTI01 chunk06 favors all-head P9_34; KITTI02 chunk14 favors head6 P9_36 on official-like control comparison.",
            "The action-conditioned signal is therefore not a global head rule; it depends on window regime plus head/action coverage.",
            "The next runtime selector should gate by scene risk, then choose action by source-quality, head coverage, and per-window evidence rather than raw mass lift.",
        ],
        "limitations": sorted(set(limitations)),
    }

    _write_csv(args.out_dir / "swa_action_conditioned_signal_rows.csv", row_out)
    _write_csv(args.out_dir / "swa_action_conditioned_signal_pairwise.csv", pair_out)
    _write_json(args.out_dir / "swa_action_conditioned_signal_summary.json", _jsonable(summary))
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
