from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from stream4d_native.v47_common import write_csv, write_json
from stream4d_native.v48_data_contract import (
    DEFAULT_ROOTS,
    ROOT,
    first_present,
    load_optional_json,
    metric_row,
    nested,
    project_path,
    rel,
    utc_now,
)


def _source_available(source_payload: dict[str, Any], needle: str) -> bool:
    for row in source_payload.get("sources", []):
        if needle.lower() in str(row.get("source", "")).lower():
            return bool(row.get("source_available"))
    return False


def _best_v47_selected(stage1: dict[str, Any], final: dict[str, Any]) -> dict[str, Any]:
    return first_present(
        stage1.get("method_row"),
        nested(stage1, "carrier_mdl_context", "row"),
        nested(final, "stage1_final_summary", "method_row"),
        nested(final, "stage1_final_summary", "carrier_mdl_context", "row"),
        {},
    )


def build_fact_lock(args: argparse.Namespace) -> dict[str, Any]:
    roots = dict(DEFAULT_ROOTS)
    for key in roots:
        override = getattr(args, key, None)
        if override:
            roots[key] = override

    v37_v44 = load_optional_json(roots["v37_v44_summary"])
    v46_fact = load_optional_json(roots["v46_fact"])
    v46_final = load_optional_json(roots["v46_final"])
    v47_fact = load_optional_json(roots["v47_fact"])
    v47_obs = load_optional_json(roots["v47_observation"])
    v47_final = load_optional_json(roots["v47_final"])
    v47_stage1 = load_optional_json(roots["v47_stage1"])
    v47_tracklet = load_optional_json(roots["v47_tracklet"])
    v47_carrier = load_optional_json(roots["v47_carrier_supertrack"])
    v47_carrier_mdl = load_optional_json(roots["v47_carrier_mdl"])
    v47_matching_flow = load_optional_json(roots["v47_matching_flow"])
    semantic_source = load_optional_json(roots["semantic_source"])
    radio_vipe = load_optional_json(roots["radio_vipe"])

    selected = _best_v47_selected(v47_stage1, v47_final)
    carrier_best = first_present(
        nested(v47_final, "carrier_component_context", "row"),
        {
            "ARI": v47_carrier.get("object_from_component_ARI"),
            "purity": v47_carrier.get("object_from_component_purity"),
            "completeness": v47_carrier.get("object_from_component_completeness"),
            "variant": "Z8_carrier_supertrack_component_union32",
        },
    )

    rows = [
        metric_row(key="v37_4D_ARI", value=nested(v37_v44, "baseline", "v37_best_metrics", default={}).get("4D_ARI") or _fact_value(v47_fact, "v37_4D_ARI"), source=roots["v47_fact"]),
        metric_row(key="v37_purity", value=nested(v37_v44, "baseline", "v37_best_metrics", default={}).get("4D_purity") or _fact_value(v47_fact, "v37_4D_purity"), source=roots["v47_fact"]),
        metric_row(key="v37_completeness", value=nested(v37_v44, "baseline", "v37_best_metrics", default={}).get("4D_completeness") or _fact_value(v47_fact, "v37_4D_completeness"), source=roots["v47_fact"]),
        metric_row(key="v37_temporal_span", value=nested(v37_v44, "baseline", "v37_best_metrics", default={}).get("temporal_span_mean") or _fact_value(v47_fact, "v37_temporal_span_mean"), source=roots["v47_fact"]),
        metric_row(key="v44_best_ARI", value=_fact_value(v47_fact, "v44_best_ARI"), source=roots["v47_fact"]),
        metric_row(key="v46_final_label", value=v46_final.get("final_label"), source=roots["v46_final"]),
        metric_row(key="v47_final_label", value=v47_final.get("final_label"), source=roots["v47_final"]),
        metric_row(key="v47_selected_variant", value=selected.get("variant"), source=roots["v47_stage1"]),
        metric_row(key="v47_selected_ARI", value=selected.get("ARI"), source=roots["v47_stage1"]),
        metric_row(key="v47_selected_purity", value=selected.get("purity"), source=roots["v47_stage1"]),
        metric_row(key="v47_selected_completeness", value=selected.get("completeness"), source=roots["v47_stage1"]),
        metric_row(key="v47_carrier_component_best_ARI", value=carrier_best.get("ARI"), source=roots["v47_carrier_supertrack"]),
        metric_row(key="v47_carrier_component_best_purity", value=carrier_best.get("purity"), source=roots["v47_carrier_supertrack"]),
        metric_row(key="v47_carrier_component_best_completeness", value=carrier_best.get("completeness"), source=roots["v47_carrier_supertrack"]),
        metric_row(key="v47_min_cost_flow_solver_note", value=v47_matching_flow.get("solver_note"), source=roots["v47_matching_flow"]),
        metric_row(key="D4RT_encoder_stride", value=v47_obs.get("D4RT_encoder_stride") or _fact_value(v47_fact, "D4RT_encoder_stride"), source=roots["v47_observation"]),
        metric_row(key="scale_guard_pass", value=nested(v47_fact, "gate", "scale_guard_pass") or nested(v46_fact, "gate", "scale_guard_pass"), source=roots["v47_fact"]),
        metric_row(key="RADIO_available", value=bool(radio_vipe.get("radio_available") or _source_available(semantic_source, "RADIO")), source=roots["radio_vipe"]),
        metric_row(key="DINO_available", value=bool(radio_vipe.get("dinov2_fallback_available") or _source_available(semantic_source, "DINO")), source=roots["radio_vipe"]),
        metric_row(key="feature_backend_used_in_v47_final", value=v47_obs.get("feature_backend"), source=roots["v47_observation"]),
        metric_row(key="carrier_observation_table_built", value=bool(v47_obs.get("carrier_observation_table_exists")), source=roots["v47_observation"]),
        metric_row(key="mask_observation_table_built", value=bool(v47_obs.get("mask_observation_table_exists")), source=roots["v47_observation"]),
        metric_row(key="carrier_row_count", value=v47_obs.get("carrier_row_count"), source=roots["v47_observation"], required=False),
        metric_row(key="mask_count", value=v47_obs.get("mask_count"), source=roots["v47_observation"], required=False),
        metric_row(key="tracklet_ARI", value=v47_tracklet.get("tracklet_ARI"), source=roots["v47_tracklet"], required=False),
        metric_row(key="carrier_mdl_ARI", value=nested(v47_carrier_mdl, "best_real_row", "ARI") or selected.get("ARI"), source=roots["v47_carrier_mdl"], required=False),
    ]

    missing_required = [row["key"] for row in rows if row["required"] and not row["available"]]
    final_label = v47_final.get("final_label")
    gate = {
        "required_fields_available": not missing_required,
        "missing_required_fields": missing_required,
        "v47_final_label_partial": final_label in {"PARTIAL_CARRIER_COMPONENT_SIGNAL", "PARTIAL_COMPONENT_SIGNAL"},
        "v47_not_stage1_success": not bool(nested(v47_stage1, "gate", "pass")),
        "d4rt_encoder_stride_eq_1": (v47_obs.get("D4RT_encoder_stride") or _fact_value(v47_fact, "D4RT_encoder_stride")) == 1,
        "scale_guard_pass": bool(nested(v47_fact, "gate", "scale_guard_pass") or nested(v46_fact, "gate", "scale_guard_pass")),
        "observation_tables_ready": bool(v47_obs.get("carrier_observation_table_exists") and v47_obs.get("mask_observation_table_exists")),
        "no_gt_for_prediction_in_v47_observation": v47_obs.get("uses_gt_for_prediction") is False,
        "radio_available": bool(radio_vipe.get("radio_available") or _source_available(semantic_source, "RADIO")),
        "dino_available": bool(radio_vipe.get("dinov2_fallback_available") or _source_available(semantic_source, "DINO")),
        "v47_flow_marked_proxy": "proxy" in str(v47_matching_flow.get("solver_note", "")).lower(),
    }
    gate["pass"] = bool(
        gate["required_fields_available"]
        and gate["v47_final_label_partial"]
        and gate["v47_not_stage1_success"]
        and gate["d4rt_encoder_stride_eq_1"]
        and gate["scale_guard_pass"]
        and gate["observation_tables_ready"]
        and gate["no_gt_for_prediction_in_v47_observation"]
        and gate["v47_flow_marked_proxy"]
    )

    return {
        "phase": "v48_fact_lock",
        "created_at": utc_now(),
        "plan_doc": rel(args.plan_doc),
        "source_roots": {key: rel(value) for key, value in roots.items()},
        "fact_rows": rows,
        "selected_v47_row": selected,
        "carrier_component_best_row": carrier_best,
        "gate": gate,
    }


def _fact_value(payload: dict[str, Any], key: str) -> Any:
    for row in payload.get("fact_rows", []):
        if row.get("key") == key:
            return row.get("value")
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Stream4D v48 Phase 0 fact lock from prior artifacts.")
    parser.add_argument("--plan-doc", default="docs/stream4d_v48_multi_branch_breakthrough_plan.md")
    parser.add_argument("--output-root", default="outputs/audit/v48_fact_lock")
    for key, default in DEFAULT_ROOTS.items():
        parser.add_argument(f"--{key.replace('_', '-')}", default=default)
    args = parser.parse_args()

    payload = build_fact_lock(args)
    out = project_path(args.output_root)
    write_json(out / "fact_lock.json", payload)
    write_csv(out / "fact_lock_rows.csv", payload["fact_rows"])
    print({"summary": str(out / "fact_lock.json"), "gate": payload["gate"]})


if __name__ == "__main__":
    main()
