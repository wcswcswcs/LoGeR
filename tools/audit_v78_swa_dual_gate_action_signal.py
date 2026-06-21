#!/usr/bin/env python3
"""Join v78 SWA window-level Q/K/V signal with action-level head-route signal.

This is diagnostic-only.  It does not choose a runtime action by chunk id and
does not use GT to build a policy.  It consolidates already-produced artifacts
so the next SWA experiment can be chosen from auditable evidence:

* selected-mask source quality and K/V alignment are window-level signals;
* per-head attention-mass lift and candidate-vs-control metrics are
  action-level signals.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


PHASE9_ROOT_01 = Path(
    "results/kitti01_hmc_v2/acl2_v78tf_pca_grounded_memory_control/report_final/"
    "phase9_swa_cache_value_carryover"
)
PHASE9_ROOT_02 = Path(
    "results/kitti02_hmc_v2/acl2_v78tf_pca_grounded_memory_control/report_final/"
    "phase9_swa_cache_value_carryover"
)
DEFAULT_QKV_ROWS = (
    PHASE9_ROOT_01
    / "selected_mask_qkv_alignment_v2_all_actions_beta070/selected_mask_qkv_alignment_rows.csv"
)
DEFAULT_OUT_DIR = PHASE9_ROOT_01 / "dual_gate_action_signal_v1"


CASES: list[dict[str, Any]] = [
    {
        "suite": "KITTI01_chunk06_P9_34_all_heads_topq80",
        "sequence": "01",
        "chunk": 6,
        "action": "P9_34",
        "action_label": "weak_positive_boundary",
        "requested_heads": list(range(16)),
        "route_label": "all_heads",
        "per_head_comparison_csv": PHASE9_ROOT_01
        / "per_head_attention_mass/p9_34_chunk06_v20_per_head_comparison.csv",
        "decision_json": PHASE9_ROOT_01
        / "smoke_chunk06_context2_v20_topq80_bias_per_head_summary/phase9_swa_cache_value_decision.json",
        "candidate": "P9_34_ATTENTION_BIAS_STABLE_AGREEMENT_TOPQ80_MASS_AUDIT_LAST",
        "control": "P9_35_ATTENTION_BIAS_STABLE_AGREEMENT_TOPQ80_RANDOM_SAME_MASS_MASS_AUDIT_LAST",
        "scene_regime_note": "low scene-score / stable-corridor-like weak-positive boundary",
    },
    {
        "suite": "KITTI01_chunk06_P9_36_head6_topq80",
        "sequence": "01",
        "chunk": 6,
        "action": "P9_36",
        "action_label": "weak_negative_default",
        "requested_heads": [6],
        "route_label": "head6",
        "per_head_comparison_csv": PHASE9_ROOT_01
        / "per_head_attention_mass/p9_36_chunk06_v21_head6_per_head_comparison.csv",
        "decision_json": PHASE9_ROOT_01
        / "smoke_chunk06_context2_v21_head6_bias/phase9_swa_cache_value_decision.json",
        "candidate": "P9_36_ATTENTION_BIAS_STABLE_AGREEMENT_TOPQ80_HEAD6_LAST",
        "control": "P9_37_ATTENTION_BIAS_STABLE_AGREEMENT_TOPQ80_RANDOM_SAME_MASS_HEAD6_LAST",
        "scene_regime_note": "low scene-score / stable-corridor-like head6-only weak-negative",
    },
    {
        "suite": "KITTI01_chunk06_P9_38_heads0_6_8_topq80",
        "sequence": "01",
        "chunk": 6,
        "action": "P9_38",
        "action_label": "weak_negative_overlap",
        "requested_heads": [0, 6, 8],
        "route_label": "heads0_6_8",
        "per_head_comparison_csv": PHASE9_ROOT_01
        / "per_head_attention_mass/p9_38_chunk06_v22_heads0_6_8_per_head_comparison.csv",
        "decision_json": PHASE9_ROOT_01
        / "smoke_chunk06_context2_v22_heads0_6_8_bias/phase9_swa_cache_value_decision.json",
        "candidate": "P9_38_ATTENTION_BIAS_STABLE_AGREEMENT_TOPQ80_HEADS0_6_8_LAST",
        "control": "P9_39_ATTENTION_BIAS_STABLE_AGREEMENT_TOPQ80_RANDOM_SAME_MASS_HEADS0_6_8_LAST",
        "scene_regime_note": "low scene-score / stable-corridor-like heads0,6,8 weak-negative",
    },
    {
        "suite": "KITTI02_chunk14_P9_34_all_heads_topq80",
        "sequence": "02",
        "chunk": 14,
        "action": "P9_34",
        "action_label": "weak_negative_boundary",
        "requested_heads": list(range(16)),
        "route_label": "all_heads",
        "per_head_comparison_csv": PHASE9_ROOT_02
        / "per_head_attention_mass/p9_34_chunk14_v3_per_head_comparison.csv",
        "decision_json": PHASE9_ROOT_02
        / "smoke_chunk14_context2_topbadpair13_14_p9_34_v3_per_head_summary/phase9_swa_cache_value_decision.json",
        "candidate": "P9_34_ATTENTION_BIAS_STABLE_AGREEMENT_TOPQ80_MASS_AUDIT_LAST",
        "control": "P9_35_ATTENTION_BIAS_STABLE_AGREEMENT_TOPQ80_RANDOM_SAME_MASS_MASS_AUDIT_LAST",
        "scene_regime_note": "higher scene-risk / corridor-shift-like all-head weak-negative",
    },
    {
        "suite": "KITTI02_chunk14_P9_36_head6_topq80",
        "sequence": "02",
        "chunk": 14,
        "action": "P9_36",
        "action_label": "weak_positive_default",
        "requested_heads": [6],
        "route_label": "head6",
        "per_head_comparison_csv": PHASE9_ROOT_02
        / "per_head_attention_mass/p9_36_chunk14_v4_head6_per_head_comparison.csv",
        "decision_json": PHASE9_ROOT_02
        / "smoke_chunk14_context2_topbadpair13_14_p9_36_head6_v4/phase9_swa_cache_value_decision.json",
        "candidate": "P9_36_ATTENTION_BIAS_STABLE_AGREEMENT_TOPQ80_HEAD6_LAST",
        "control": "P9_37_ATTENTION_BIAS_STABLE_AGREEMENT_TOPQ80_RANDOM_SAME_MASS_HEAD6_LAST",
        "scene_regime_note": "higher scene-risk / corridor-shift-like head6 weak-positive",
    },
    {
        "suite": "KITTI02_chunk14_P9_38_heads0_6_8_topq80",
        "sequence": "02",
        "chunk": 14,
        "action": "P9_38",
        "action_label": "weak_negative_overlap",
        "requested_heads": [0, 6, 8],
        "route_label": "heads0_6_8",
        "per_head_comparison_csv": PHASE9_ROOT_02
        / "per_head_attention_mass/p9_38_chunk14_v5_heads0_6_8_per_head_comparison.csv",
        "decision_json": PHASE9_ROOT_02
        / "smoke_chunk14_context2_topbadpair13_14_p9_38_heads0_6_8_v5/phase9_swa_cache_value_decision.json",
        "candidate": "P9_38_ATTENTION_BIAS_STABLE_AGREEMENT_TOPQ80_HEADS0_6_8_LAST",
        "control": "P9_39_ATTENTION_BIAS_STABLE_AGREEMENT_TOPQ80_RANDOM_SAME_MASS_HEADS0_6_8_LAST",
        "scene_regime_note": "higher scene-risk / corridor-shift-like heads0,6,8 weak-negative",
    },
]


MECHANISM_KEYS = [
    "head10_to_tail10_pose_sim3_rmse_m",
    "overlap3_to_future_pose_sim3_rmse_m",
    "scale_cv_head_mid_tail_pose_sim3",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qkv-rows", type=Path, default=DEFAULT_QKV_ROWS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
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
            writer.writerow(row)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _qkv_index(rows: list[dict[str, str]]) -> dict[tuple[str, str, int], dict[str, str]]:
    index: dict[tuple[str, str, int], dict[str, str]] = {}
    for row in rows:
        if row.get("source_pair") != "prev_current__cur_current":
            continue
        kind = str(row.get("kind") or "")
        layer = row.get("layer_id")
        try:
            layer_i = int(str(layer))
        except ValueError:
            continue
        index[(str(row.get("suite")), kind, layer_i)] = row
    return index


def _extract_qkv(case: dict[str, Any], qkv: dict[tuple[str, str, int], dict[str, str]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for kind in ("k", "v"):
        for layer in (18, 26):
            row = qkv.get((case["suite"], kind, layer), {})
            prefix = f"{kind}_l{layer}"
            out[f"{prefix}_selected_minus_random_cosine"] = _finite(
                row.get("selected_minus_random_patch_same_mass_cosine_mean")
            )
            out[f"{prefix}_selected_cosine"] = _finite(row.get("selected_cosine_mean"))
            out[f"{prefix}_random_same_mass_cosine"] = _finite(
                row.get("random_patch_same_mass_cosine_mean")
            )
    meta = qkv.get((case["suite"], "k", 18), {})
    out["score_patch_selected_mean"] = _finite(meta.get("score_patch_selected_mean"))
    out["score_patch_random_same_mass_mean"] = _finite(meta.get("score_patch_random_same_mass_mean"))
    out["score_patch_selected_minus_random_same_mass_mean"] = _finite(
        meta.get("score_patch_selected_minus_random_same_mass_mean")
    )
    out["selected_patch_count"] = _finite(meta.get("selected_patch_count"))
    out["selected_special_count_dropped"] = _finite(meta.get("selected_special_count_dropped"))
    return out


def _extract_head_signal(case: dict[str, Any]) -> dict[str, Any]:
    rows = _read_csv(Path(case["per_head_comparison_csv"]))
    heads = [int(head) for head in case["requested_heads"]]
    out: dict[str, Any] = {
        "per_head_comparison_csv": str(case["per_head_comparison_csv"]),
        "per_head_rows": len(rows),
    }
    parsed: list[dict[str, Any]] = []
    for row in rows:
        try:
            head = int(str(row.get("head")))
        except (TypeError, ValueError):
            continue
        parsed.append(
            {
                "head": head,
                "selected_delta": _finite(row.get("candidate_minus_control_selected_lift")),
                "source_delta": _finite(row.get("candidate_minus_control_source_lift")),
                "candidate_selected_lift": _finite(row.get("candidate_selected_lift")),
                "control_selected_lift": _finite(row.get("control_selected_lift")),
                "candidate_source_lift": _finite(row.get("candidate_source_lift")),
                "control_source_lift": _finite(row.get("control_source_lift")),
            }
        )
    requested = [row for row in parsed if int(row["head"]) in heads]
    positive = [row for row in parsed if (row.get("selected_delta") or 0.0) > 0.0]
    requested_positive = [row for row in requested if (row.get("selected_delta") or 0.0) > 0.0]
    top = max(parsed, key=lambda row: row.get("selected_delta") or -math.inf, default={})
    out.update(
        {
            "requested_heads": ",".join(str(head) for head in heads),
            "requested_head_count": len(heads),
            "requested_positive_head_count": len(requested_positive),
            "positive_head_count_all": len(positive),
            "top_selected_delta_head": top.get("head"),
            "top_selected_delta": top.get("selected_delta"),
            "top_source_delta": top.get("source_delta"),
            "requested_selected_delta_sum": sum(
                float(row.get("selected_delta") or 0.0) for row in requested
            ),
            "requested_selected_delta_sum_negative": sum(
                float(row.get("selected_delta") or 0.0) for row in requested
            )
            < 0.0,
            "requested_selected_delta_max": max(
                [float(row.get("selected_delta") or 0.0) for row in requested],
                default=None,
            ),
            "requested_source_delta_sum": sum(float(row.get("source_delta") or 0.0) for row in requested),
            "requested_source_delta_max": max(
                [float(row.get("source_delta") or 0.0) for row in requested],
                default=None,
            ),
            "requested_candidate_selected_lift_sum": sum(
                float(row.get("candidate_selected_lift") or 0.0) for row in requested
            ),
            "requested_control_selected_lift_sum": sum(
                float(row.get("control_selected_lift") or 0.0) for row in requested
            ),
        }
    )
    return out


def _extract_decision(case: dict[str, Any]) -> dict[str, Any]:
    path = Path(case["decision_json"])
    out: dict[str, Any] = {"decision_json": str(path), "decision_available": False}
    if not path.is_file():
        return out
    payload = json.loads(path.read_text(encoding="utf-8"))
    decision = (payload.get("decisions") or {}).get(case["candidate"]) or {}
    out.update(
        {
            "decision_available": True,
            "phase9_gate_pass": bool(decision.get("phase9_gate_pass", False)),
            "action_fidelity_pass": bool(decision.get("action_fidelity_pass", False)),
            "metric_passes": ",".join(str(x) for x in (decision.get("metric_passes") or [])),
        }
    )
    comparisons = decision.get("comparisons") or {}
    for key in MECHANISM_KEYS:
        comp = comparisons.get(key) or {}
        prefix = {
            "head10_to_tail10_pose_sim3_rmse_m": "head_tail",
            "overlap3_to_future_pose_sim3_rmse_m": "future",
            "scale_cv_head_mid_tail_pose_sim3": "scale_cv",
        }[key]
        out[f"{prefix}_candidate"] = _finite(comp.get("candidate"))
        out[f"{prefix}_baseline"] = _finite(comp.get("baseline"))
        out[f"{prefix}_best_control"] = _finite(comp.get("best_control"))
        out[f"{prefix}_improvement_vs_baseline_ratio"] = _finite(
            comp.get("improvement_vs_baseline_ratio")
        )
        out[f"{prefix}_beats_control"] = bool(comp.get("beats_controls", False))
        out[f"{prefix}_metric_key_pass"] = bool(comp.get("phase9_metric_key_pass", False))
    return out


def _classify(row: dict[str, Any]) -> dict[str, Any]:
    score_delta = row.get("score_patch_selected_minus_random_same_mass_mean")
    v_l26 = row.get("v_l26_selected_minus_random_cosine")
    k_l26 = row.get("k_l26_selected_minus_random_cosine")
    positive_requested_heads = int(row.get("requested_positive_head_count") or 0)
    out = {
        "window_source_quality_positive": bool(score_delta is not None and score_delta > 0.0),
        "window_v_l26_positive": bool(v_l26 is not None and v_l26 > 0.0),
        "window_k_l26_risk_negative": bool(k_l26 is not None and k_l26 < 0.0),
        "action_requested_route_positive": bool(positive_requested_heads > 0),
    }
    out["dual_gate_offline_signal"] = bool(
        out["window_source_quality_positive"]
        and out["window_v_l26_positive"]
        and out["window_k_l26_risk_negative"]
        and out["action_requested_route_positive"]
    )
    return out


def main() -> None:
    args = parse_args()
    qkv = _qkv_index(_read_csv(args.qkv_rows))
    rows: list[dict[str, Any]] = []
    for case in CASES:
        row: dict[str, Any] = {
            "suite": case["suite"],
            "sequence": case["sequence"],
            "chunk": case["chunk"],
            "action": case["action"],
            "action_label": case["action_label"],
            "route_label": case["route_label"],
            "candidate": case["candidate"],
            "control": case["control"],
            "scene_regime_note": case["scene_regime_note"],
        }
        row.update(_extract_qkv(case, qkv))
        row.update(_extract_head_signal(case))
        row.update(_extract_decision(case))
        row.update(_classify(row))
        rows.append(row)

    out_csv = args.out_dir / "dual_gate_action_signal_rows.csv"
    out_json = args.out_dir / "dual_gate_action_signal_summary.json"
    _write_csv(out_csv, rows)
    summary = {
        "schema": "acl2_v78_swa_dual_gate_action_signal_v1",
        "qkv_rows": str(args.qkv_rows),
        "out_csv": str(out_csv),
        "num_cases": len(rows),
        "dual_gate_offline_signal_count": sum(1 for row in rows if row["dual_gate_offline_signal"]),
        "phase9_gate_pass_count": sum(1 for row in rows if row.get("phase9_gate_pass")),
        "rows": rows,
        "interpretation_limits": [
            "This is offline diagnostic aggregation only.",
            "It does not prove runtime success or choose a chunk-specific policy.",
            "Q/K/V selected alignment is treated as window-level signal.",
            "Head-route attention mass and candidate-vs-random metrics are treated as action-level signal.",
        ],
    }
    _write_json(out_json, summary)
    print(f"wrote_csv={out_csv} rows={len(rows)}")
    print(f"wrote_json={out_json}")


if __name__ == "__main__":
    main()
