from __future__ import annotations

import argparse
from statistics import mean
from typing import Any

from stream4d_native.v47_common import write_csv, write_json
from stream4d_native.v48_data_contract import load_optional_json, project_path, rel, utc_now


def _avg(values: list[Any]) -> float | None:
    nums: list[float] = []
    for value in values:
        if value is None or value == "":
            continue
        try:
            nums.append(float(value))
        except (TypeError, ValueError):
            continue
    return float(mean(nums)) if nums else None


def _rows_for(payload: dict[str, Any], variants: set[str]) -> list[dict[str, Any]]:
    return [row for row in payload.get("positive_summary_rows", []) if str(row.get("variant")) in variants]


def _best_auc(rows: list[dict[str, Any]]) -> tuple[float | None, str | None]:
    best: tuple[float | None, str | None] = (None, None)
    for row in rows:
        value = row.get("edge_same_gt_AUC")
        if value is None:
            continue
        score = float(value)
        if best[0] is None or score > best[0]:
            best = (score, str(row.get("variant")))
    return best


def _feature_success_rate(payload: dict[str, Any]) -> float | None:
    values = [scene.get("descriptor_coverage") for scene in payload.get("diag", {}).values() if isinstance(scene, dict)]
    return _avg(values)


def _mean_field(rows: list[dict[str, Any]], key: str) -> float | None:
    return _avg([row.get(key) for row in rows])


def _summarize_backend(name: str, source: str, payload: dict[str, Any]) -> dict[str, Any]:
    if name == "colorhist":
        feature_variants = {"P6_colorhist_feature_only"}
        merge_variants = {
            "P5_p4_colorhist_boost_capped",
            "P5_p4_colorhist_linear_capped",
            "P5_p4_colorhist_product_rescore_capped",
        }
        shuffled_variants = {
            "P5_shuffled_colorhist_boost_capped",
            "P5_shuffled_colorhist_product_rescore_capped",
        }
    else:
        feature_variants = {"P6_feature_only"}
        merge_variants = {
            "P5_p4_semantic_boost_capped",
            "P5_p4_semantic_linear_capped",
            "P5_p4_semantic_product_rescore_capped",
        }
        shuffled_variants = {
            "P5_shuffled_semantic_boost_capped",
            "P5_shuffled_semantic_product_rescore_capped",
        }

    feature_rows = _rows_for(payload, feature_variants)
    merge_rows = _rows_for(payload, merge_variants)
    shuffled_rows = _rows_for(payload, shuffled_variants)
    feature_auc, feature_variant = _best_auc(feature_rows)
    merge_auc, merge_variant = _best_auc(merge_rows)
    shuffled_auc, shuffled_variant = _best_auc(shuffled_rows)
    real_minus_shuffled = _mean_field(merge_rows, "real_minus_shuffled_edge_AUC")
    if real_minus_shuffled is None and merge_auc is not None and shuffled_auc is not None:
        real_minus_shuffled = float(merge_auc - shuffled_auc)
    negative_rows = payload.get("negative_summary_rows", [])
    negative_gate_all = bool(negative_rows) and all(bool(row.get("gate_pass")) for row in negative_rows)
    negative_gate_any = bool(negative_rows) and any(bool(row.get("gate_pass")) for row in negative_rows)
    return {
        "backend_id": name,
        "feature_backend": payload.get("feature_backend") or ("colorhist" if name == "colorhist" else name),
        "source": rel(source),
        "feature_success_rate": _feature_success_rate(payload),
        "component_pair_AUC_proxy": feature_auc,
        "component_pair_AUC_proxy_variant": feature_variant,
        "semantic_merge_AUC_proxy": merge_auc,
        "semantic_merge_AUC_proxy_variant": merge_variant,
        "shuffled_semantic_AUC_proxy": shuffled_auc,
        "shuffled_semantic_AUC_proxy_variant": shuffled_variant,
        "real_minus_shuffled_edge_AUC": real_minus_shuffled,
        "positive_gate_any": any(bool(row.get("gate_pass")) for row in merge_rows),
        "negative_guard_all_scene_pass": negative_gate_all,
        "negative_guard_any_scene_pass": negative_gate_any,
        "uses_frozen_dense_features": bool(payload.get("uses_frozen_dense_features") or any(scene.get("uses_frozen_dense_features") for scene in payload.get("diag", {}).values() if isinstance(scene, dict))),
        "uses_gt_for_prediction": bool(payload.get("gate", {}).get("uses_gt_for_prediction", payload.get("uses_gt_for_prediction", False))),
        "uses_gt_for_diagnostic_labels": bool(payload.get("gate", {}).get("uses_gt_for_diagnostic_labels", payload.get("uses_gt_for_diagnostic_labels", True))),
        "metric_scope": "hard_scene_edge_proxy_not_component_merge",
    }


def build_audit(args: argparse.Namespace) -> dict[str, Any]:
    sources = {
        "colorhist": args.colorhist_root + "/raw_visual_semantic_repair.json",
        "dinov2": args.dinov2_root + "/raw_visual_semantic_repair.json",
        "radio": args.radio_root + "/raw_visual_semantic_repair.json",
    }
    rows = [_summarize_backend(name, source, load_optional_json(source)) for name, source in sources.items()]
    color = next(row for row in rows if row["backend_id"] == "colorhist")
    color_feature_auc = color.get("component_pair_AUC_proxy")
    color_merge_auc = color.get("semantic_merge_AUC_proxy")

    for row in rows:
        feature_auc = row.get("component_pair_AUC_proxy")
        merge_auc = row.get("semantic_merge_AUC_proxy")
        row["feature_AUC_delta_vs_colorhist"] = None if feature_auc is None or color_feature_auc is None else float(feature_auc - color_feature_auc)
        row["merge_AUC_delta_vs_colorhist"] = None if merge_auc is None or color_merge_auc is None else float(merge_auc - color_merge_auc)
        row["beats_colorhist_feature_gate"] = row["feature_AUC_delta_vs_colorhist"] is not None and row["feature_AUC_delta_vs_colorhist"] >= 0.08
        row["beats_colorhist_merge_gate"] = row["merge_AUC_delta_vs_colorhist"] is not None and row["merge_AUC_delta_vs_colorhist"] >= 0.02
        row["shuffled_control_gate"] = row.get("real_minus_shuffled_edge_AUC") is not None and float(row["real_minus_shuffled_edge_AUC"]) >= 0.02
        row["phase2_positive_guard_pass"] = bool((row["beats_colorhist_feature_gate"] or row["beats_colorhist_merge_gate"]) and row["shuffled_control_gate"])

    strong_rows = [row for row in rows if row["backend_id"] != "colorhist" and row["phase2_positive_guard_pass"]]
    contradiction_rows = [row for row in rows if row["backend_id"] != "colorhist" and row["negative_guard_all_scene_pass"]]
    gate = {
        "strong_backend_beats_colorhist": bool(strong_rows),
        "best_positive_backend": strong_rows[0]["backend_id"] if strong_rows else None,
        "semantic_contradiction_guard_available": bool(contradiction_rows),
        "recommended_contradiction_backend": contradiction_rows[0]["backend_id"] if contradiction_rows else None,
        "feature_shuffled_control_any_pass": any(row["backend_id"] != "colorhist" and row["shuffled_control_gate"] for row in rows),
        "no_gt_for_prediction": not any(row["uses_gt_for_prediction"] for row in rows),
    }
    gate["pass"] = bool(gate["strong_backend_beats_colorhist"] and gate["no_gt_for_prediction"])
    failure_label = None if gate["pass"] else "NO_GO_SEMANTIC_GUARD"
    recommendation = (
        "Use strong frozen features as positive component merge guard."
        if gate["pass"]
        else "Do not claim positive semantic co-inference; keep RADIO/DINO as availability/runtime evidence and use passing RADIO contradiction guard only."
    )
    return {
        "phase": "v48_semantic_feature_audit",
        "created_at": utc_now(),
        "backend_rows": rows,
        "gate": gate,
        "failure_label": failure_label,
        "recommendation": recommendation,
        "thresholds": {
            "feature_AUC_delta_vs_colorhist": 0.08,
            "merge_AUC_delta_vs_colorhist": 0.02,
            "real_minus_shuffled_edge_AUC": 0.02,
        },
        "metric_scope_note": "Existing artifacts are v46 hard-scene raw visual semantic edge proxies; they verify runtime and edge-level behavior but do not prove v48 component-level merge success.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stream4D v48 Phase 2 frozen semantic feature audit.")
    parser.add_argument("--colorhist-root", default="outputs/audit/v46_raw_visual_semantic_repair_v26_d2r4_n120_colorhist_gap4_product")
    parser.add_argument("--dinov2-root", default="outputs/audit/v46_raw_visual_semantic_repair_v26_d2r4_n120_dinov2")
    parser.add_argument("--radio-root", default="outputs/audit/v46_raw_visual_semantic_repair_v26_d2r4_n120_radio_q6densitysoft008c500_gap4_highthr")
    parser.add_argument("--output-root", default="outputs/audit/v48_semantic_features")
    args = parser.parse_args()

    payload = build_audit(args)
    out = project_path(args.output_root)
    write_json(out / "semantic_feature_audit_summary.json", payload)
    write_csv(out / "semantic_feature_audit_rows.csv", payload["backend_rows"])
    print({"summary": str(out / "semantic_feature_audit_summary.json"), "gate": payload["gate"], "failure_label": payload["failure_label"]})


if __name__ == "__main__":
    main()

