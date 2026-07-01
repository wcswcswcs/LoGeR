#!/usr/bin/env python3
"""Post-final v99 Phase10I one-shot same-schema support-weight holdout test.

Phase10H fixed the holdout support_surfel_count schema and found that
parent60/semantic20/support20 was the best measured repair, but it still missed
the strict holdout gate. This script performs one bounded follow-up variant:
shift a little more weight to same-schema support while reducing semantic
weight. It replays the Phase10H best row as a reference and does not overwrite
Phase10H artifacts.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
if str(STREAM3D_ROOT) not in sys.path:
    sys.path.insert(0, str(STREAM3D_ROOT))

from tools import build_v98_1_canonical_holdout_metrics as holdout  # noqa: E402
from tools import build_v99_phase10h_same_schema_support_quality_holdout as p10h  # noqa: E402


AUDIT_ROOT = STREAM3D_ROOT / "outputs/audit"
OUT_DIR = AUDIT_ROOT / "v99_phase10i_same_schema_support_final_weight_holdout"
PHASE0_DIR = AUDIT_ROOT / "v99_phase0_fact_lock"


def _score(config: dict[str, Any], feature: dict[str, float]) -> float:
    sem = 0.5 * feature["radio_norm"] + 0.5 * feature["dino_norm"]
    mode = str(config["mode"])
    if mode == "parent":
        return feature["parent_max"]
    if mode == "h3_reference_parent60_sem20_support20":
        return 0.6 * feature["parent_norm"] + 0.2 * sem + 0.2 * feature["support_surfel_count_norm"]
    if mode == "final_parent55_sem15_support30":
        return 0.55 * feature["parent_norm"] + 0.15 * sem + 0.30 * feature["support_surfel_count_norm"]
    raise ValueError(f"unknown mode {mode}")


def _make_rows(
    parent_rows: list[dict[str, Any]],
    features: dict[str, dict[str, float]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    variant = f"V99P10I_holdout_{config['name']}"
    out: list[dict[str, Any]] = []
    for row in parent_rows:
        oid = str(row["mv_object_id"])
        feature = features[oid]
        new = dict(row)
        new["variant_id"] = variant
        new["variant"] = variant
        new["score"] = float(_score(config, feature))
        new["score_policy"] = config["score_policy"]
        new["phase10i_parent_variant_id"] = row.get("variant_id", "")
        new["phase10i_mode"] = config["mode"]
        new["phase10i_support_surfel_count_mean"] = feature["support_surfel_count_mean"]
        new["phase10i_support_surfel_count_norm"] = feature["support_surfel_count_norm"]
        new["phase10i_radio_coherence"] = feature["radio_coherence"]
        new["phase10i_dino_coherence"] = feature["dino_coherence"]
        new["uses_gt_for_prediction"] = False
        new["uses_future"] = False
        out.append(new)
    return out


def main() -> int:
    started = datetime.now()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    phase0 = json.loads((PHASE0_DIR / "summary.json").read_text(encoding="utf-8"))
    scope = holdout._load_source_scope(p10h.HOLDOUT_SOURCE_ROWS)
    parent_rows = [dict(row) for row in p10h._read_csv(p10h.HOLDOUT_FIXED_ROWS)]
    joined_rows, join_diag = p10h._join_support_surfel_count(parent_rows)
    semantic = p10h._semantic_features(joined_rows, scope)
    features = p10h._feature_table(joined_rows, semantic)

    configs = [
        {
            "name": "I0_parent_reference",
            "mode": "parent",
            "score_policy": "phase10i_parent_score_replay",
            "is_new_repair_variant": False,
        },
        {
            "name": "Iref_H3_reference_parent60_sem20_support20",
            "mode": "h3_reference_parent60_sem20_support20",
            "score_policy": "phase10i_replay_phase10h_best_0p60_parent_0p20_semantic_0p20_support_surfel_count",
            "is_new_repair_variant": False,
        },
        {
            "name": "I1_final_parent55_sem15_support30",
            "mode": "final_parent55_sem15_support30",
            "score_policy": "phase10i_final_one_shot_0p55_parent_0p15_semantic_0p30_support_surfel_count",
            "is_new_repair_variant": True,
        },
    ]

    all_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    top_rows: list[dict[str, Any]] = []
    config_rows: list[dict[str, Any]] = []
    for config in configs:
        rows = _make_rows(joined_rows, features, config)
        metrics, cases, tops = holdout._evaluate_variant(f"V99P10I_holdout_{config['name']}", rows, scope)
        all_rows.extend(rows)
        metric_rows.extend(metrics)
        case_rows.extend(cases)
        top_rows.extend(tops)
        config_rows.append(
            {
                "schema_version": "stream4d_v99_phase10i_variant_config_v1",
                "phase_id": "v99_phase10i_same_schema_support_final_weight_holdout",
                "name": config["name"],
                "mode": config["mode"],
                "score_policy": config["score_policy"],
                "is_new_repair_variant": config["is_new_repair_variant"],
                "support_source": p10h._rel(p10h.HOLDOUT_SUPPORT_ROWS),
                "support_schema": "support_surfel_count joined by object_id/scene/frame/mask",
                "variant_selection_rationale": "one bounded follow-up after Phase10H best favored support20; test support30 with lower semantic weight",
                "uses_gt_for_prediction": False,
                "uses_future": False,
                "formal_claim_allowed": False,
            }
        )

    aggregate_rows = holdout._aggregate(
        metric_rows,
        family="v99_phase10i_same_schema_support_final_weight_holdout",
    )
    f2_hold_window = float(phase0["F2_base_holdout_MV_AP_window"])
    f2_hold_ap50 = float(phase0["F2_base_holdout_MV_AP50_window"])
    parent = next(row for row in aggregate_rows if str(row["variant_id"]).endswith("I0_parent_reference"))
    paired: list[dict[str, Any]] = []
    for row in aggregate_rows:
        name = str(row["variant_id"]).replace("V99P10I_holdout_", "")
        window = p10h._num(row.get("mean_MV_AP_window"))
        ap50 = p10h._num(row.get("mean_MV_AP50_window"))
        gate = window >= f2_hold_window + 0.005 and ap50 >= f2_hold_ap50 + 0.010
        is_new = name == "I1_final_parent55_sem15_support30"
        paired.append(
            {
                "schema_version": "stream4d_v99_phase10i_holdout_metric_v1",
                "phase_id": "v99_phase10i_same_schema_support_final_weight_holdout",
                "name": name,
                "holdout_variant_id": row["variant_id"],
                "is_new_repair_variant": is_new,
                "holdout_MV_AP_window": row.get("mean_MV_AP_window"),
                "holdout_MV_AP50_window": row.get("mean_MV_AP50_window"),
                "holdout_MV_AP25_window": row.get("mean_MV_AP25_window"),
                "delta_vs_parent_window": window - p10h._num(parent.get("mean_MV_AP_window")),
                "delta_vs_parent_AP50_window": ap50 - p10h._num(parent.get("mean_MV_AP50_window")),
                "holdout_delta_vs_F2_base_window": window - f2_hold_window,
                "holdout_gate_pass": gate,
                "uses_gt_for_prediction": False,
                "uses_future": False,
                "formal_claim_allowed": False,
            }
        )

    reference_rows = [row for row in paired if row["name"] != "I0_parent_reference"]
    new_rows = [row for row in paired if row["is_new_repair_variant"]]
    best_any = max(reference_rows, key=lambda row: (p10h._num(row["holdout_MV_AP_window"]), p10h._num(row["holdout_MV_AP50_window"])))
    best_new = max(new_rows, key=lambda row: (p10h._num(row["holdout_MV_AP_window"]), p10h._num(row["holdout_MV_AP50_window"])))
    any_pass = any(bool(row["holdout_gate_pass"]) for row in reference_rows)
    new_pass = any(bool(row["holdout_gate_pass"]) for row in new_rows)

    gate_rows = [
        {
            "gate_id": "same_schema_support_join_complete",
            "pass": int(join_diag["missing_parent_row_count"]) == 0,
            "expected": "0 missing fixed holdout rows after support_surfel_count join",
            "observed": join_diag["missing_parent_row_count"],
            "severity": "required_data_contract",
        },
        {
            "gate_id": "final_support_weight_holdout_gate",
            "pass": any_pass,
            "expected": f"MV_AP_window>={f2_hold_window + 0.005} and MV_AP50_window>={f2_hold_ap50 + 0.010}",
            "observed": f"best_any={best_any['name']} MV_AP_window={best_any['holdout_MV_AP_window']} MV_AP50_window={best_any['holdout_MV_AP50_window']}; best_new={best_new['name']} MV_AP_window={best_new['holdout_MV_AP_window']} MV_AP50_window={best_new['holdout_MV_AP50_window']}",
            "severity": "method_gate",
        },
        {
            "gate_id": "new_final_weight_variant_gate",
            "pass": new_pass,
            "expected": f"new final-weight variant passes MV_AP_window>={f2_hold_window + 0.005} and MV_AP50_window>={f2_hold_ap50 + 0.010}",
            "observed": f"{best_new['name']} MV_AP_window={best_new['holdout_MV_AP_window']} MV_AP50_window={best_new['holdout_MV_AP50_window']}",
            "severity": "method_gate",
        },
        {
            "gate_id": "formal_claim_allowed_after_post_final_diagnostic",
            "pass": False,
            "expected": "fresh frozen holdout",
            "observed": "post-final diagnostic after prior holdout feedback",
            "severity": "formal_claim_blocker",
        },
    ]
    failure_rows = [
        {
            "failure_id": row["gate_id"],
            "severity": row["severity"],
            "expected": row["expected"],
            "observed": row["observed"],
            "repair_direction": "Stop local score-blend repair on this fixed object universe; continue only with a new object-birth/candidate-generation plan or a fresh frozen holdout protocol.",
        }
        for row in gate_rows
        if not bool(row["pass"])
    ]
    feature_rows = [
        {
            "schema_version": "stream4d_v99_phase10i_object_feature_v1",
            "phase_id": "v99_phase10i_same_schema_support_final_weight_holdout",
            "mv_object_id": oid,
            **values,
            "uses_gt_for_prediction": False,
            "uses_future": False,
        }
        for oid, values in sorted(features.items())
    ]
    summary = {
        "schema_version": "stream4d_v99_phase10i_same_schema_support_final_weight_holdout_summary_v1",
        "phase_id": "v99_phase10i_same_schema_support_final_weight_holdout",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "runtime_sec": (datetime.now() - started).total_seconds(),
        "decision": "DIAGNOSTIC_FINAL_SUPPORT_WEIGHT_HOLDOUT_PASS_REQUIRES_FRESH_HOLDOUT" if any_pass else "NO_GO_FINAL_SUPPORT_WEIGHT_HOLDOUT",
        "formal_claim_allowed": False,
        "uses_gt_for_prediction": False,
        "uses_future": False,
        "variant_count": len(configs),
        "new_repair_variant_count": len(new_rows),
        "object_count": len(features),
        "join_diag": join_diag,
        "best_any_name": best_any["name"],
        "best_any_MV_AP_window": float(p10h._num(best_any["holdout_MV_AP_window"])),
        "best_any_MV_AP50_window": float(p10h._num(best_any["holdout_MV_AP50_window"])),
        "best_new_name": best_new["name"],
        "best_new_MV_AP_window": float(p10h._num(best_new["holdout_MV_AP_window"])),
        "best_new_MV_AP50_window": float(p10h._num(best_new["holdout_MV_AP50_window"])),
        "best_new_delta_vs_F2_base_window": float(p10h._num(best_new["holdout_delta_vs_F2_base_window"])),
        "any_holdout_variant_passes_strict_gate": any_pass,
        "new_final_weight_variant_passes_strict_gate": new_pass,
        "outputs": {
            "summary": p10h._rel(OUT_DIR / "summary.json"),
            "variant_config_rows": p10h._rel(OUT_DIR / "variant_config_rows.csv"),
            "paired_metric_rows": p10h._rel(OUT_DIR / "paired_metric_rows.csv"),
            "holdout_metric_rows": p10h._rel(OUT_DIR / "holdout_metric_rows.csv"),
            "holdout_metric_aggregate_rows": p10h._rel(OUT_DIR / "holdout_metric_aggregate_rows.csv"),
            "object_feature_rows": p10h._rel(OUT_DIR / "object_feature_rows.csv"),
            "variant_gate_rows": p10h._rel(OUT_DIR / "variant_gate_rows.csv"),
            "variant_failure_rows": p10h._rel(OUT_DIR / "variant_failure_rows.csv"),
        },
    }

    p10h._write_csv(OUT_DIR / "variant_config_rows.csv", config_rows)
    p10h._write_csv(OUT_DIR / "paired_metric_rows.csv", paired)
    p10h._write_csv(OUT_DIR / "holdout_metric_rows.csv", metric_rows)
    p10h._write_csv(OUT_DIR / "holdout_metric_aggregate_rows.csv", aggregate_rows)
    p10h._write_csv(OUT_DIR / "holdout_case_rows.csv", case_rows)
    p10h._write_csv(OUT_DIR / "holdout_top_iou_rows.csv", top_rows)
    p10h._write_csv(OUT_DIR / "holdout_mv_object_frame_mask_rows.csv", all_rows)
    p10h._write_csv(OUT_DIR / "object_feature_rows.csv", feature_rows)
    p10h._write_csv(OUT_DIR / "variant_gate_rows.csv", gate_rows)
    p10h._write_csv(OUT_DIR / "variant_failure_rows.csv", failure_rows)
    p10h._write_json(OUT_DIR / "summary.json", summary)
    print(json.dumps(p10h._jsonable(summary), indent=2, sort_keys=True))
    return 0 if any_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
