from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from stream4d_native.matching_significance import V43_MINIMUM_GATE, as_float
from stream4d_native.v37_object_field_adapter import read_csv_rows, read_json


def _sum(values: list[Any]) -> float:
    out = 0.0
    for value in values:
        parsed = as_float(value)
        if parsed is not None:
            out += float(parsed)
    return out


def _mean(values: list[Any]) -> float | None:
    nums = [as_float(value) for value in values]
    nums = [float(value) for value in nums if value is not None]
    return float(sum(nums) / len(nums)) if nums else None


def _find_row(rows: list[dict[str, Any]], key: str, value: str) -> dict[str, Any]:
    return next((row for row in rows if str(row.get(key)) == value), {})


def build_error_profile(stream3d_root: Path, *, v37_4d_root: str, adapter_summary: dict[str, Any]) -> dict[str, Any]:
    root = Path(stream3d_root)
    v37_root = root / v37_4d_root
    v37_decision = read_json(v37_root / "4d_memory_decision.json") or {}
    best = dict(v37_decision.get("best_metrics") or {})
    scene_rows = read_csv_rows(v37_root / "4d_memory_scene_rows.csv")
    best_variant = str(v37_decision.get("best_variant") or best.get("variant") or "")
    best_scene_rows = [row for row in scene_rows if str(row.get("variant")) == best_variant]
    delta_pair_rows = read_json(root / "outputs/audit/v37_adaptive_density_final_probe5/v37_delta_t_pair_attribution/delta_t_pair_summary.json") or []
    learned_rows = read_json(root / "outputs/audit/v37_adaptive_density_final_probe5/v37_learned_diagnostic/learned_diagnostic_summary.json") or []
    hard_root_cause = read_csv_rows(
        root / "outputs/audit/v42_structure_affinity_twohop_backfill8_max480_r1/root_cause/hard_scene_root_cause_rows.csv"
    )

    labeled_total = _sum([row.get("labeled_tube_count") for row in best_scene_rows])
    loss_rows = [
        {
            "error_type": "oversplit_id_switch_fragmentation",
            "loss_proxy": float(best.get("ID_switches") or 0.0) + float(best.get("fragmentation") or 0.0) * max(len(best_scene_rows), 1),
            "source": "v37_4d_memory_decision.best_metrics.ID_switches_fragmentation",
        },
        {
            "error_type": "overmerge",
            "loss_proxy": float(best.get("merge_errors") or 0.0),
            "source": "v37_4d_memory_decision.best_metrics.merge_errors",
        },
        {
            "error_type": "low_support_unknown",
            "loss_proxy": float(best.get("unknown_tube_ratio") or 0.0) * max(labeled_total, 1.0),
            "source": "v37_4d_scene_rows.unknown_tube_ratio*labeled_tube_count",
        },
        {
            "error_type": "hard_scene_scene0081_gap",
            "loss_proxy": max(0.0, V43_MINIMUM_GATE["scene0081_ARI"] - float(best.get("scene0081_ARI") or 0.0)) * 10000.0,
            "source": "v43_gate.scene0081_ARI - v37_scene0081_ARI",
        },
    ]
    total_loss = sum(float(row["loss_proxy"]) for row in loss_rows)
    for row in loss_rows:
        row["loss_share"] = float(row["loss_proxy"] / total_loss) if total_loss else 0.0
    loss_rows = sorted(loss_rows, key=lambda row: float(row["loss_proxy"]), reverse=True)
    top_two_share = sum(float(row.get("loss_share") or 0.0) for row in loss_rows[:2])

    same_frame = _find_row(delta_pair_rows, "delta_bin", "B0_dt0_same_frame")
    short_rows = [row for row in delta_pair_rows if str(row.get("delta_bin")) in {"B1_dt1_adjacent", "B2_dt2_near", "B3_dt3_4_short"}]
    learned_mean = _find_row(learned_rows, "fold", "MEAN")
    hard_semantic_auc = _mean([row.get("semantic_affinity_AUC") for row in hard_root_cause])
    hard_coverage = _mean([row.get("coverage@0.10") for row in hard_root_cause])
    precision = {
        "overmerge_suspect_precision": as_float(same_frame.get("diff_GT_ratio")),
        "oversplit_suspect_precision_proxy": as_float(learned_mean.get("F1")),
        "low_support_suspect_precision": None,
        "hard_scene_semantic_auc_mean": hard_semantic_auc,
        "hard_scene_coverage010_mean": hard_coverage,
        "near_temporal_merge_auc_mean": _mean([row.get("merge_AUC") for row in short_rows]),
    }
    gate = {
        "top_two_error_types_explain": top_two_share,
        "top_two_error_types_explain_pass": top_two_share >= 0.60,
        "overmerge_suspect_precision_pass": precision["overmerge_suspect_precision"] is not None
        and float(precision["overmerge_suspect_precision"]) >= 0.70,
        "oversplit_suspect_precision_pass": precision["oversplit_suspect_precision_proxy"] is not None
        and float(precision["oversplit_suspect_precision_proxy"]) >= 0.70,
        "low_support_suspect_precision_pass": False,
        "low_support_precision_status": "missing_prediction_only_precision_evidence",
    }
    gate["profiler_gate_pass"] = bool(
        gate["top_two_error_types_explain_pass"]
        and gate["overmerge_suspect_precision_pass"]
        and gate["oversplit_suspect_precision_pass"]
        and gate["low_support_suspect_precision_pass"]
    )
    repair_attempts = [
        {
            "attempt": "combine_v37_same_frame_cannot_link_with_v37_adjacent_pair_diagnostic",
            "result": "overmerge_and_oversplit_have_high_proxy_precision",
            "evidence": {
                "overmerge_precision": precision["overmerge_suspect_precision"],
                "oversplit_precision_proxy": precision["oversplit_suspect_precision_proxy"],
            },
        },
        {
            "attempt": "import_v42_hard_scene_root_cause_for_low_support",
            "result": "coverage/AUC diagnostics available but no low_support precision label for accepted correction",
            "evidence": {
                "hard_scene_semantic_auc_mean": hard_semantic_auc,
                "hard_scene_coverage010_mean": hard_coverage,
            },
        },
    ]
    return {
        "phase": "v43_2_matching_error_profiler",
        "status": "PASS_PROFILER" if gate["profiler_gate_pass"] else "NO_GO_NO_REPAIRABLE_ERRORS",
        "best_variant": best_variant,
        "adapter_status": adapter_summary.get("status"),
        "diagnostic_loss_rows": loss_rows,
        "suspect_precision": precision,
        "gate": gate,
        "repair_attempts": repair_attempts,
        "hard_scene_root_cause_rows": hard_root_cause,
        "notes": [
            "oversplit_suspect_precision_proxy uses v37 learned diagnostic F1 because the artifact does not expose precision directly.",
            "low_support_suspect_precision is deliberately not fabricated; current artifacts do not provide a prediction-only precision estimate.",
        ],
    }
