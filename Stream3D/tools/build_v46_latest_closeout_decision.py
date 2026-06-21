from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _artifact(path: str) -> str:
    return str(Path(path))


def _rows_by_scene_variant(rows: list[dict[str, Any]], variant: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        if str(row.get("variant")) == variant:
            out[str(row.get("scene"))] = row
    return out


def _best_scene_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        scene = str(row.get("scene"))
        current = out.get(scene)
        if current is None or float(row.get("diagnostic_subset_ari") or -1.0) > float(
            current.get("diagnostic_subset_ari") or -1.0
        ):
            out[scene] = row
    return out


def _select_fields(row: dict[str, Any], fields: list[str]) -> dict[str, Any]:
    return {field: row.get(field) for field in fields}


def _raw_positive_summary(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    rows = payload.get("positive_summary_rows") or []
    p5 = _rows_by_scene_variant(rows, "P5_p4_semantic_boost_capped")
    shared = _rows_by_scene_variant(rows, "shared_carrier_jaccard")
    out: dict[str, Any] = {
        "artifact": _artifact(path.relative_to(ROOT)),
        "gate": payload.get("gate"),
        "feature_backend": payload.get("feature_backend"),
        "quality_variant": payload.get("quality_variant"),
        "uses_gt_for_prediction": (payload.get("gate") or {}).get("uses_gt_for_prediction"),
        "uses_gt_for_diagnostic_labels": (payload.get("gate") or {}).get("uses_gt_for_diagnostic_labels"),
        "p5_boost_by_scene": {},
    }
    for scene, row in p5.items():
        shared_row = shared.get(scene, {})
        out["p5_boost_by_scene"][scene] = {
            **_select_fields(
                row,
                [
                    "edge_same_gt_AUC",
                    "edge_precision@top5k",
                    "real_minus_shared_edge_AUC",
                    "real_minus_no_temporal_edge_AUC",
                    "edge_recall@threshold",
                    "positive_edge_density@threshold",
                ],
            ),
            "shared_carrier_jaccard_AUC": shared_row.get("edge_same_gt_AUC"),
            "shared_precision@top5k": shared_row.get("edge_precision@top5k"),
            "gate_pass": row.get("gate_pass"),
        }
    return out


def _solver_summary(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    rows = _best_scene_rows(payload.get("best_rows") or [])
    return {
        "artifact": _artifact(path.relative_to(ROOT)),
        "gate": payload.get("gate"),
        "best_by_scene": {
            scene: _select_fields(
                row,
                [
                    "solver_variant",
                    "positive_key",
                    "positive_threshold",
                    "negative_key",
                    "diagnostic_gate_pass",
                    "diagnostic_subset_purity",
                    "diagnostic_subset_completeness",
                    "diagnostic_subset_ari",
                    "diagnostic_pairwise_precision",
                    "diagnostic_pairwise_recall",
                    "hard_negative_violation_rate",
                    "positive_candidate_count",
                    "cluster_count",
                ],
            )
            for scene, row in rows.items()
        },
    }


def _local_solver_summary(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    rows = _best_scene_rows(payload.get("best_rows") or [])
    scene_rows = payload.get("scene_rows") or []
    scene0591_signed = [
        row
        for row in scene_rows
        if str(row.get("scene")) == "scene0591_00" and str(row.get("negative_key")) != "none"
    ]

    def best_scene0591_under_violation(limit: float) -> dict[str, Any] | None:
        candidates = [
            row
            for row in scene0591_signed
            if float(row.get("hard_negative_violation_rate") or 0.0) <= float(limit)
        ]
        if not candidates:
            return None
        candidates.sort(
            key=lambda row: (
                float(row.get("diagnostic_subset_completeness") or -1.0),
                float(row.get("diagnostic_subset_purity") or -1.0),
                float(row.get("diagnostic_subset_ari") or -1.0),
            ),
            reverse=True,
        )
        return _select_fields(
            candidates[0],
            [
                "input_root_name",
                "solver_variant",
                "positive_threshold",
                "negative_key",
                "negative_mode",
                "negative_weight",
                "local_mode",
                "local_topk",
                "diagnostic_gate_pass",
                "diagnostic_subset_purity",
                "diagnostic_subset_completeness",
                "diagnostic_subset_ari",
                "hard_negative_violation_rate",
                "cluster_count",
            ],
        )

    joint_rows = payload.get("joint_config_rows") or []
    return {
        "artifact": _artifact(path.relative_to(ROOT)),
        "gate": payload.get("gate"),
        "best_by_scene": {
            scene: _select_fields(
                row,
                [
                    "solver_variant",
                    "positive_key",
                    "positive_threshold",
                    "negative_key",
                    "negative_mode",
                    "negative_weight",
                    "local_mode",
                    "local_topk",
                    "diagnostic_gate_pass",
                    "diagnostic_subset_purity",
                    "diagnostic_subset_completeness",
                    "diagnostic_subset_ari",
                    "hard_negative_violation_rate",
                    "positive_candidate_count",
                    "local_candidate_same_gt_precision",
                    "cluster_count",
                ],
            )
            for scene, row in rows.items()
        },
        "top_joint_configs": [
            _select_fields(
                row,
                [
                    "input_root_name",
                    "positive_threshold",
                    "negative_key",
                    "negative_mode",
                    "negative_weight",
                    "local_mode",
                    "local_topk",
                    "all_scene_diagnostic_gate_pass",
                    "mean_purity_minus_s0",
                    "min_completeness",
                    "max_hard_negative_violation_rate",
                ],
            )
            for row in joint_rows[:5]
        ],
        "scene0591_pareto": {
            "best_under_violation_0p01": best_scene0591_under_violation(0.01),
            "best_under_violation_0p05": best_scene0591_under_violation(0.05),
            "best_under_violation_0p10": best_scene0591_under_violation(0.10),
        },
    }


def _correlation_solver_summary(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    rows = _best_scene_rows(payload.get("best_rows") or [])
    joint_rows = payload.get("joint_config_rows") or []
    return {
        "artifact": _artifact(path.relative_to(ROOT)),
        "gate": payload.get("gate"),
        "best_by_scene": {
            scene: _select_fields(
                row,
                [
                    "solver_variant",
                    "positive_key",
                    "positive_threshold",
                    "negative_key",
                    "negative_lambda",
                    "cluster_lambda",
                    "init_mode",
                    "local_mode",
                    "local_topk",
                    "phase5_solver_gate_pass",
                    "strict_local_gate_pass",
                    "diagnostic_subset_purity",
                    "diagnostic_subset_completeness",
                    "diagnostic_subset_ari",
                    "hard_negative_violation_rate",
                    "cluster_count",
                    "positive_cut_cost",
                    "negative_inside_cost",
                    "energy_total",
                ],
            )
            for scene, row in rows.items()
        },
        "top_joint_configs": [
            _select_fields(
                row,
                [
                    "input_root_name",
                    "positive_threshold",
                    "negative_key",
                    "negative_lambda",
                    "cluster_lambda",
                    "init_mode",
                    "local_mode",
                    "local_topk",
                    "all_scene_phase5_solver_gate_pass",
                    "all_scene_strict_local_gate_pass",
                    "mean_purity_minus_s0",
                    "min_completeness",
                    "mean_ari",
                    "max_hard_negative_violation_rate",
                ],
            )
            for row in joint_rows[:5]
        ],
    }


def _exact_micro_summary(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    rows = payload.get("best_rows") or []
    return {
        "artifact": _artifact(path.relative_to(ROOT)),
        "gate": payload.get("gate"),
        "full_scene_exact": payload.get("gate", {}).get("full_scene_exact"),
        "best_rows": [
            _select_fields(
                row,
                [
                    "solver_variant",
                    "positive_threshold",
                    "negative_key",
                    "negative_lambda",
                    "cluster_lambda",
                    "local_mode",
                    "local_topk",
                    "micro_size",
                    "micro_phase5_gate_pass",
                    "diagnostic_subset_purity",
                    "diagnostic_subset_completeness",
                    "diagnostic_subset_ari",
                    "purity_minus_s0",
                    "ari_minus_s0",
                    "hard_negative_violation_rate",
                    "cluster_count",
                    "partition_count_visited",
                    "partition_search_truncated",
                    "energy_total",
                    "micro_node_ids",
                ],
            )
            for row in rows[:5]
        ],
    }


def _s7_carrier_summary(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    summary_rows = payload.get("summary_rows") or []
    return {
        "artifact": _artifact(path.relative_to(ROOT)),
        "gate": payload.get("gate"),
        "diagnostic_only": payload.get("diagnostic_only"),
        "uses_gt_for_prediction": payload.get("uses_gt_for_prediction"),
        "uses_gt_for_diagnostic_labels": payload.get("uses_gt_for_diagnostic_labels"),
        "carrier_cache_root": payload.get("carrier_cache_root"),
        "max_carriers": payload.get("max_carriers"),
        "positive_thresholds": payload.get("positive_thresholds"),
        "negative_thresholds": payload.get("negative_thresholds"),
        "soft_negative_lambdas": payload.get("soft_negative_lambdas"),
        "merge_margins": payload.get("merge_margins"),
        "summary_by_scene": {
            str(row.get("scene")): {
                **_select_fields(
                    row,
                    [
                        "carrier_node_count",
                        "carrier_edge_count",
                        "diagnostic_labeled_carrier_count",
                        "positive_score_same_gt_auc",
                        "negative_score_diff_gt_auc",
                        "positive_precision@top1k",
                    ],
                ),
                "best_signed": _select_fields(
                    row.get("best_signed") or {},
                    [
                        "solver_variant",
                        "positive_threshold",
                        "negative_threshold",
                        "soft_negative_lambda",
                        "merge_margin",
                        "carrier_phase5_gate_pass",
                        "diagnostic_subset_purity",
                        "diagnostic_subset_completeness",
                        "diagnostic_subset_ari",
                        "diagnostic_pairwise_precision",
                        "diagnostic_pairwise_recall",
                        "hard_negative_violation_rate",
                        "positive_candidate_count",
                        "signed_merge_candidate_count",
                        "hard_negative_count",
                        "accepted_merge_count",
                        "rejected_negative_veto_count",
                        "rejected_soft_margin_count",
                        "cluster_count",
                        "purity_minus_s0",
                        "ari_minus_s0",
                        "completeness_minus_s0",
                    ],
                ),
            }
            for row in summary_rows
        },
        "best_rows": [
            _select_fields(
                row,
                [
                    "scene",
                    "solver_variant",
                    "positive_threshold",
                    "negative_threshold",
                    "soft_negative_lambda",
                    "merge_margin",
                    "carrier_phase5_gate_pass",
                    "diagnostic_subset_purity",
                    "diagnostic_subset_completeness",
                    "diagnostic_subset_ari",
                    "hard_negative_violation_rate",
                    "cluster_count",
                ],
            )
            for row in payload.get("best_rows") or []
        ],
    }


def _supporter_proxy_summary(path: Path, interesting_variants: list[str]) -> dict[str, Any]:
    payload = _read_json(path)
    rows = payload.get("summary_rows") or []
    by_variant = {variant: _rows_by_scene_variant(rows, variant) for variant in interesting_variants}
    compact: dict[str, Any] = {}
    for variant, scene_rows in by_variant.items():
        compact[variant] = {
            scene: _select_fields(
                row,
                [
                    "edge_same_gt_AUC",
                    "edge_precision@top5k",
                    "real_minus_shared_edge_AUC",
                    "precision_top5k_minus_shared",
                    "real_minus_shuffled_edge_AUC",
                    "supporter_hub_candidate_threshold",
                    "supporter_hub_fanout_cap",
                    "supporter_hub_fanout_mean",
                    "supporter_hub_fanout_p90",
                    "supporter_hub_fanout_max",
                    "supporter_hub_weight_mean",
                    "same_frame_false_merge_supporter_rate",
                    "gate_pass",
                ],
            )
            for scene, row in scene_rows.items()
        }
    return {
        "artifact": _artifact(path.relative_to(ROOT)),
        "gate": payload.get("gate"),
        "variants": compact,
    }


def build_payload() -> dict[str, Any]:
    base_final = _read_json(ROOT / "outputs/audit/v46_final_decision/v46_final_decision.json")
    radio_availability_path = ROOT / "outputs/audit/v46_loger_env_radio_radseg_availability_recheck_20260619/radio_vipe_availability.json"
    radio_smoke_path = ROOT / "outputs/audit/v46_loger_env_radio_radseg_adapter_smoke_recheck_20260619/feature_smoke.json"
    radseg_probe_path = (
        ROOT
        / "outputs/audit/v46_radseg_prediction_probe_scene0591_top20_pooled_prompts_amp_side448/radseg_prediction_probe.json"
    )
    raw_radio_path = (
        ROOT / "outputs/audit/v46_raw_visual_semantic_repair_v21_3_d2r4_n120_radio_gap4_highthr/raw_visual_semantic_repair.json"
    )
    q6_hard_path = (
        ROOT
        / "outputs/audit/v46_raw_visual_semantic_repair_v21_3_d2r4_n120_radio_q6densityhard006c300_gap4_highthr/raw_visual_semantic_repair.json"
    )
    solver_radio_path = ROOT / "outputs/audit/v46_raw_signed_solver_diagnostic_radio_gap4_highthr/raw_signed_solver_diagnostic.json"
    solver_q6_hard_path = (
        ROOT / "outputs/audit/v46_raw_signed_solver_diagnostic_radio_q6densityhard006c300_gap4_highthr/raw_signed_solver_diagnostic.json"
    )
    local_radio_path = (
        ROOT / "outputs/audit/v46_local_candidate_signed_solver_radio_gap4_highthr/local_candidate_signed_solver.json"
    )
    local_q6_hard_path = (
        ROOT
        / "outputs/audit/v46_local_candidate_signed_solver_radio_q6densityhard006c300_gap4_highthr/local_candidate_signed_solver.json"
    )
    local_q6_soft_path = (
        ROOT
        / "outputs/audit/v46_local_candidate_signed_solver_radio_q6densitysoft008c500_gap4_highthr/local_candidate_signed_solver.json"
    )
    local_q6_expand_path = (
        ROOT
        / "outputs/audit/v46_local_candidate_signed_solver_radio_q6densityhard006c300_expand_gap4_highthr/local_candidate_signed_solver.json"
    )
    local_q6_expand_highconf_path = (
        ROOT
        / "outputs/audit/v46_local_candidate_signed_solver_radio_q6densityhard006c300_expand_highconf_gap4_highthr/local_candidate_signed_solver.json"
    )
    correlation_s5_sanity_path = (
        ROOT
        / "outputs/audit/v46_correlation_local_search_q6densityhard_sanity_gap4_highthr/correlation_local_search_solver.json"
    )
    exact_micro_top10_path = (
        ROOT
        / "outputs/audit/v46_exact_micro_partition_scene0591_q6densityhard_top10_gap4_highthr/exact_micro_partition.json"
    )
    exact_micro_top11_path = (
        ROOT
        / "outputs/audit/v46_exact_micro_partition_scene0591_q6densityhard_top11_single_gap4_highthr/exact_micro_partition.json"
    )
    s7_carrier_path = (
        ROOT
        / "outputs/audit/v46_s7_carrier_level_diagnostic_v21_3_d2r4_subset160_softneg_margin_sweep/s7_carrier_level_diagnostic.json"
    )
    hubsoft_path = ROOT / "outputs/audit/v46_supporter_quality_hubsoft_v21_3_gap4_proxy/supporter_quality_raw_repair.json"
    hubcombo_path = (
        ROOT / "outputs/audit/v46_supporter_quality_hubsoft_density_combo_v21_3_gap4_proxy/supporter_quality_raw_repair.json"
    )
    hubcap_path = ROOT / "outputs/audit/v46_supporter_quality_hubcap_v21_3_gap4_proxy/supporter_quality_raw_repair.json"

    radio_availability = _read_json(radio_availability_path)
    radio_smoke = _read_json(radio_smoke_path)
    radseg_probe = _read_json(radseg_probe_path)
    radio_smoke_row = (radio_smoke.get("rows") or [{}])[0]

    evidence = {
        "base_final_decision": {
            "artifact": "outputs/audit/v46_final_decision/v46_final_decision.json",
            "final_label": base_final.get("final_label"),
            "answers": base_final.get("answers"),
            "created_at": base_final.get("created_at"),
        },
        "loger_radio_radseg": {
            "availability_artifact": _artifact(radio_availability_path.relative_to(ROOT)),
            "adapter_smoke_artifact": _artifact(radio_smoke_path.relative_to(ROOT)),
            "radio_available": radio_availability.get("radio_available"),
            "radio_or_radseg_import_available": radio_availability.get("radio_or_radseg_import_available"),
            "radio_feature_smoke_ok": (radio_availability.get("radio_feature_smoke") or {}).get("ok"),
            "adapter_gate_pass": radio_smoke.get("gate_pass"),
            "adapter_backend": radio_smoke.get("backend"),
            "feature_shape": [
                radio_smoke_row.get("feature_h"),
                radio_smoke_row.get("feature_w"),
                radio_smoke_row.get("feature_c"),
            ],
            "feature_smoke_row": radio_smoke_row,
        },
        "radseg_probe": {
            "artifact": _artifact(radseg_probe_path.relative_to(ROOT)),
            "phase": radseg_probe.get("phase"),
            "diagnostic_only": radseg_probe.get("diagnostic_only"),
            "probe_mode": radseg_probe.get("probe_mode"),
            "max_image_side": radseg_probe.get("max_image_side"),
            "summary_rows": radseg_probe.get("summary_rows"),
        },
        "raw_radio_positive": _raw_positive_summary(raw_radio_path),
        "raw_radio_q6_density_hard_positive": _raw_positive_summary(q6_hard_path),
        "raw_radio_signed_solver": _solver_summary(solver_radio_path),
        "raw_radio_q6_density_hard_signed_solver": _solver_summary(solver_q6_hard_path),
        "local_radio_signed_solver": _local_solver_summary(local_radio_path),
        "local_radio_q6_density_hard_signed_solver": _local_solver_summary(local_q6_hard_path),
        "local_radio_q6_density_soft_signed_solver": _local_solver_summary(local_q6_soft_path),
        "local_radio_q6_density_hard_expand_signed_solver": _local_solver_summary(local_q6_expand_path),
        "local_radio_q6_density_hard_expand_highconf_signed_solver": _local_solver_summary(
            local_q6_expand_highconf_path
        ),
        "correlation_local_search_q6_density_hard_sanity": _correlation_solver_summary(correlation_s5_sanity_path),
        "exact_micro_partition_scene0591_top10": _exact_micro_summary(exact_micro_top10_path),
        "exact_micro_partition_scene0591_top11": _exact_micro_summary(exact_micro_top11_path),
        "s7_carrier_level_diagnostic": _s7_carrier_summary(s7_carrier_path),
        "s7_carrier_level_feasibility": {
            "artifact": _artifact(s7_carrier_path.relative_to(ROOT)),
            "can_run_from_existing_artifacts": True,
            "reason": (
                "S7 can be reconstructed for diagnostics by reusing the raw carrier incidence repair loader, "
                "which samples carrier uv_pred against prepared masks into labels_by_frame. The resulting "
                "carrier-level graph is diagnostic-only/subset160 and is not a full Stage-1 method export."
            ),
            "uses_gt_for_prediction": False,
        },
        "hubsoft_proxy": _supporter_proxy_summary(
            hubsoft_path,
            ["Q5_split_outside_fragment_soft", "Q5_split_outside_fragment_soft_hubsoft_q020"],
        ),
        "hubsoft_density_combo_proxy": _supporter_proxy_summary(
            hubcombo_path,
            [
                "Q6_density_hard_0p006_c300",
                "Q6_density_hard_0p006_c300_hubsoft_q020",
                "Q6_density_soft_0p008_c500",
                "Q6_density_soft_0p008_c500_hubsoft_q020",
            ],
        ),
        "hubcap_proxy": _supporter_proxy_summary(
            hubcap_path,
            ["Q5_split_outside_fragment_soft", "Q5_split_outside_fragment_soft_hubcap32_q020"],
        ),
    }

    answers = {
        "1_built_mask_carrier_incidence": {
            "answer": True,
            "qualification": "base v46 incidence gate passed, but incidence_uses_raw_uv_containment=false; later raw-cache proxies were used for repairs.",
            "evidence": evidence["base_final_decision"]["answers"],
        },
        "2_view_consensus_beats_shared_tube": {
            "answer": False,
            "evidence": evidence["raw_radio_positive"]["p5_boost_by_scene"],
        },
        "3_supporter_reliability_reduces_false_positive_edges": {
            "answer": False,
            "evidence": {
                "base_supporter_answer": (base_final.get("answers") or {}).get(
                    "supporter_reliability_reduces_false_positive_edges"
                ),
                "hubsoft_gate": evidence["hubsoft_proxy"]["gate"],
                "hubcap_gate": evidence["hubcap_proxy"]["gate"],
                "hubsoft_density_combo_gate": evidence["hubsoft_density_combo_proxy"]["gate"],
            },
        },
        "4_negative_edges_reduce_false_merge": {
            "answer": False,
            "qualification": "N4 negatives can be precise, but raw/local/soft/expansion solver gates remain false and scene0591 trades completeness against purity or hard-negative violations.",
            "evidence": {
                "raw": evidence["raw_radio_signed_solver"]["best_by_scene"],
                "local_expand_pareto": evidence["local_radio_q6_density_hard_expand_highconf_signed_solver"][
                    "scene0591_pareto"
                ],
            },
        },
        "5_signed_solver_beats_positive_only": {
            "answer": False,
            "qualification": "scene0081 passes the diagnostic, scene0591 does not; raw/local/soft/expand joint gates are false.",
            "evidence": {
                "raw_gate": evidence["raw_radio_signed_solver"]["gate"],
                "local_gate": evidence["local_radio_signed_solver"]["gate"],
                "local_q6_hard_gate": evidence["local_radio_q6_density_hard_signed_solver"]["gate"],
                "local_q6_expand_gate": evidence["local_radio_q6_density_hard_expand_signed_solver"]["gate"],
                "local_q6_expand_highconf_gate": evidence[
                    "local_radio_q6_density_hard_expand_highconf_signed_solver"
                ]["gate"],
                "correlation_s5_sanity_gate": evidence["correlation_local_search_q6_density_hard_sanity"][
                    "gate"
                ],
                "exact_micro_top10_gate": evidence["exact_micro_partition_scene0591_top10"]["gate"],
                "exact_micro_top11_gate": evidence["exact_micro_partition_scene0591_top11"]["gate"],
                "s7_carrier_diagnostic_gate": evidence["s7_carrier_level_diagnostic"]["gate"],
            },
        },
        "6_full_stage1_significant": {
            "answer": False,
            "evidence": base_final.get("stage1_gate"),
        },
        "7_d4rt_real_beats_controls": {
            "answer": False,
            "evidence": (base_final.get("answers") or {}).get("d4rt_real_beats_controls"),
        },
        "8_no_d4rt_tube_birth_no_gt_method_path": {
            "answer": True,
            "qualification": "latest diagnostic artifacts preserve uses_gt_for_prediction=false; solver best rows report birth_from_d4rt_tube_count=0.",
            "evidence": {
                "raw_radio_gate": evidence["raw_radio_positive"]["gate"],
                "solver_scene_rows": evidence["raw_radio_signed_solver"]["best_by_scene"],
                "local_solver_scene_rows": evidence["local_radio_q6_density_hard_expand_signed_solver"][
                    "best_by_scene"
                ],
                "s7_carrier_uses_gt_for_prediction": evidence["s7_carrier_level_diagnostic"][
                    "uses_gt_for_prediction"
                ],
            },
        },
        "9_scale_guard_pass_weak_chunks_blocked": {
            "answer": bool((base_final.get("answers") or {}).get("scale_guard_pass")),
            "evidence": base_final.get("fact_gate"),
        },
        "10_ap_eval_aligned_only": {
            "answer": True,
            "qualification": "AP is not promoted because Stage-1 is false; base AP answer remains eval-only.",
            "evidence": {
                "ap_eval_aligned_only": (base_final.get("answers") or {}).get("ap_eval_aligned_only"),
                "ap_gate": base_final.get("ap_gate"),
            },
        },
        "11_stage2_allowed": {
            "answer": False,
            "evidence": base_final.get("stage2_gate"),
        },
        "12_failure_location": {
            "answer": "NO_GO_VIEW_CONSENSUS_EDGE",
            "secondary": "NO_GO_SUPPORTER_RELIABILITY",
            "additional": ["NO_GO_NEGATIVE_EDGE", "NO_GO_SOLVER"],
            "evidence": {
                "base_final_label": base_final.get("final_label"),
                "latest_supporter_gates": {
                    "hubsoft": evidence["hubsoft_proxy"]["gate"],
                    "hubsoft_density_combo": evidence["hubsoft_density_combo_proxy"]["gate"],
                    "hubcap": evidence["hubcap_proxy"]["gate"],
                },
                "latest_local_solver_gates": {
                    "local_radio": evidence["local_radio_signed_solver"]["gate"],
                    "local_q6_hard": evidence["local_radio_q6_density_hard_signed_solver"]["gate"],
                    "local_q6_soft": evidence["local_radio_q6_density_soft_signed_solver"]["gate"],
                    "local_q6_expand": evidence["local_radio_q6_density_hard_expand_signed_solver"]["gate"],
                    "local_q6_expand_highconf": evidence[
                        "local_radio_q6_density_hard_expand_highconf_signed_solver"
                    ]["gate"],
                    "correlation_s5_sanity": evidence["correlation_local_search_q6_density_hard_sanity"][
                        "gate"
                    ],
                    "exact_micro_top10": evidence["exact_micro_partition_scene0591_top10"]["gate"],
                    "exact_micro_top11": evidence["exact_micro_partition_scene0591_top11"]["gate"],
                    "s7_carrier_diagnostic": evidence["s7_carrier_level_diagnostic"]["gate"],
                },
                "s7_feasibility": evidence["s7_carrier_level_feasibility"],
                "s7_scene_summary": evidence["s7_carrier_level_diagnostic"]["summary_by_scene"],
            },
        },
    }

    return {
        "phase": "v46_latest_closeout_decision",
        "created_at": _utc_now(),
        "final_label": "NO_GO_VIEW_CONSENSUS_EDGE",
        "secondary_label": "NO_GO_SUPPORTER_RELIABILITY",
        "additional_labels": ["NO_GO_NEGATIVE_EDGE", "NO_GO_SOLVER"],
        "stage2_allowed": False,
        "ap_promoted": False,
        "uses_gt_for_prediction": False,
        "diagnostic_uses_gt_labels": True,
        "evidence": evidence,
        "answers": answers,
        "decision": {
            "summary": (
                "Latest RADIO/RADSeg, supporter hub/fanout, local candidate retrieval, soft-negative, and "
                "cluster-expansion plus S7 carrier-level diagnostic experiments do not overturn the base v46 "
                "NO_GO_VIEW_CONSENSUS_EDGE decision. The strongest scene0591 completeness gains require "
                "large purity drops or hard-negative violations, so Stage-2/AP remain blocked."
            ),
            "why_not_go": [
                "P5/RADIO positive edge gate remains false.",
                "scene0591 signed solver completeness remains low under acceptable hard-negative violation.",
                "Increasing expansion enough to recover scene0591 completeness overmerges and violates hard negatives.",
                "S5 correlation local search sanity sweep has no all-scene joint pass and leaves scene0591 low-completeness.",
                "S6 exact micro-partition diagnostics are exhaustive on 10/11-node scene0591 subgraphs but do not improve over S0.",
                "S7 carrier-level diagnostic runs from reconstructed carrier-to-mask containment, but all-scene gate remains false.",
                "S7 scene0591 hard-veto variants overcut completeness; soft-negative variants recover completeness only with purity loss or hard-negative violations.",
                "hub/fanout supporter repair has no all-scene proxy gate pass.",
                "density+hubsoft helps scene0591 locally but destroys scene0081.",
            ],
        },
    }


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# v46 Latest Closeout Decision",
        "",
        f"created_at: {payload['created_at']}",
        f"final_label: {payload['final_label']}",
        f"secondary_label: {payload['secondary_label']}",
        f"additional_labels: {payload.get('additional_labels')}",
        f"stage2_allowed: {payload['stage2_allowed']}",
        f"ap_promoted: {payload['ap_promoted']}",
        "",
        "## Summary",
        "",
        payload["decision"]["summary"],
        "",
        "## Why Not GO",
        "",
    ]
    for item in payload["decision"]["why_not_go"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Plan Questions", ""])
    for key, answer in payload["answers"].items():
        lines.append(f"- {key}: {answer.get('answer')}")
        if answer.get("secondary") is not None:
            lines.append(f"  secondary: {answer.get('secondary')}")
        if answer.get("qualification"):
            lines.append(f"  note: {answer.get('qualification')}")
    lines.extend(["", "## Evidence Artifacts", ""])
    for key, value in payload["evidence"].items():
        artifact = value.get("artifact") if isinstance(value, dict) else None
        if artifact:
            lines.append(f"- {key}: {artifact}")
        elif isinstance(value, dict):
            artifacts = [v for k, v in value.items() if k.endswith("artifact") and isinstance(v, str)]
            if artifacts:
                lines.append(f"- {key}: {', '.join(artifacts)}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build latest v46 closeout decision including follow-up repairs.")
    parser.add_argument("--output-root", default="outputs/audit/v46_latest_closeout_decision_20260619")
    args = parser.parse_args()
    payload = build_payload()
    out = ROOT / str(args.output_root)
    _write_json(out / "v46_latest_closeout_decision.json", payload)
    _write_text(out / "v46_latest_closeout_decision.md", _markdown(payload))
    print(
        json.dumps(
            {
                "summary": str(out / "v46_latest_closeout_decision.json"),
                "markdown": str(out / "v46_latest_closeout_decision.md"),
                "final_label": payload["final_label"],
                "secondary_label": payload["secondary_label"],
                "stage2_allowed": payload["stage2_allowed"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
