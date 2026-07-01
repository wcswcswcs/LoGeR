from __future__ import annotations

from pathlib import Path
from typing import Any

from .v47_common import ROOT, read_json, utc_now, write_csv, write_json


REQUIRED_ARTIFACTS = {
    "v58_fact_lock": "outputs/audit/v58_phase0_fact_lock/fact_lock.json",
    "v58_semantic_dino": "outputs/audit/v58_semantic_memory_dino_full_repair2/semantic_memory_summary.json",
    "v58_explanation": "outputs/audit/v58_counterfactual_explanation_dino_full_repair6/explanation_summary.json",
    "v58_strict_query": "outputs/audit/v58_active_material_query_q128_repair3/query_summary.json",
    "v58_expanded_query": "outputs/audit/v58_active_material_query_q128_repair5_expanded_all_minvis1/query_summary.json",
    "v58_expanded_reprojection": (
        "outputs/audit/v58_active_query_reprojection_ledger_deferred_max1600_noveto_minvis1/"
        "reprojection_summary.json"
    ),
    "v56_core": "outputs/audit/v56_core_update/core_update_summary.json",
    "v56_tentative": "outputs/audit/v56_tentative_support/tentative_support_summary.json",
}


def _project(path: str | Path) -> Path:
    path_obj = Path(path)
    return path_obj if path_obj.is_absolute() else ROOT / path_obj


def _rel(path: str | Path) -> str:
    path_obj = _project(path)
    try:
        return str(path_obj.relative_to(ROOT))
    except ValueError:
        return str(path_obj)


def _load_required(paths: dict[str, str | Path]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    payloads: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for key, path in paths.items():
        path_obj = _project(path)
        if not path_obj.exists():
            missing.append(_rel(path_obj))
            continue
        payloads[key] = read_json(path_obj)
    return payloads, missing


def _num(value: Any, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: Any, default: int = 0) -> int:
    num = _num(value)
    return default if num is None else int(num)


def _gate_pass(payload: dict[str, Any] | None) -> bool | None:
    if not payload:
        return None
    gate = payload.get("gate") or {}
    if "pass" not in gate:
        return None
    return bool(gate.get("pass"))


def build_v59_fact_lock(artifact_paths: dict[str, str | Path] | None = None) -> dict[str, Any]:
    paths = {**REQUIRED_ARTIFACTS, **(artifact_paths or {})}
    artifacts, missing = _load_required(paths)

    v58_fact = artifacts.get("v58_fact_lock", {})
    sem = artifacts.get("v58_semantic_dino", {})
    exp = artifacts.get("v58_explanation", {})
    strict = artifacts.get("v58_strict_query", {})
    expanded = artifacts.get("v58_expanded_query", {})
    reproj = artifacts.get("v58_expanded_reprojection", {})
    v56_core = artifacts.get("v56_core", {})
    v56_tentative = artifacts.get("v56_tentative", {})

    strict_valid = _num(strict.get("Q6_valid_material_evidence_rate"))
    expanded_valid = _num(expanded.get("Q6_valid_material_evidence_rate"))
    strict_auc = _num(strict.get("Q6_real_minus_shuffled_query_AUC"))
    expanded_auc = _num(expanded.get("Q6_real_minus_shuffled_query_AUC"))
    strict_no_temporal = _num(strict.get("Q6_real_minus_no_temporal_query_AUC"))
    expanded_no_temporal = _num(expanded.get("Q6_real_minus_no_temporal_query_AUC"))
    reproj_success = _num(reproj.get("reprojection_success_rate"))
    reproj_conflict = _num(reproj.get("same_frame_exclusion_violation_rate"))

    validity_drop = (
        strict_valid is not None and expanded_valid is not None and expanded_valid < strict_valid
    )
    shuffled_control_drop = (
        strict_auc is not None and expanded_auc is not None and expanded_auc < strict_auc
    )
    no_temporal_control_drop = (
        strict_no_temporal is not None
        and expanded_no_temporal is not None
        and expanded_no_temporal < strict_no_temporal
    )
    expanded_reprojection_failed = (
        reproj_success is not None
        and reproj_conflict is not None
        and (reproj_success < 0.60 or reproj_conflict > 0.05)
    )
    expanded_quality_drop_observed = bool(
        validity_drop and shuffled_control_drop and no_temporal_control_drop and expanded_reprojection_failed
    )

    fact_lock = {
        "phase": "v59_phase0_fact_lock",
        "created_at": utc_now(),
        "method_note": (
            "Phase0 locks v58/v56 evidence only. It does not create a v59 object-field prediction. "
            "GT-bearing source fields are copied only as diagnostic metadata from prior summaries."
        ),
        "input_paths": {key: _rel(path) for key, path in paths.items()},
        "missing_artifacts": missing,
        "v58_phase0_gate_pass": _gate_pass(v58_fact),
        "v58_phase1_dino_recall@3": sem.get("history_shortlist_recall@3"),
        "v58_phase1_underseg_AUC": sem.get("underseg_detection_AUC"),
        "v58_phase1_gate_pass": _gate_pass(sem),
        "v58_phase2_actionable_count": exp.get("actionable_count"),
        "v58_phase2_deferred_count": exp.get("deferred_count"),
        "v58_phase2_assign_precision": exp.get("assign_precision_diagnostic"),
        "v58_phase2_underseg_precision": exp.get("underseg_precision_diagnostic"),
        "v58_phase2_projection_completeness": exp.get("completeness"),
        "v58_phase2_false_history_update_rate": exp.get("false_history_update_rate"),
        "v58_phase2_gate_pass": _gate_pass(exp),
        "v58_phase3_strict_gate_pass": _gate_pass(strict),
        "v58_phase3_Q6_entropy_reduction": strict.get("Q6_entropy_reduction"),
        "v58_phase3_Q6_entropy_required": strict.get("Q6_entropy_reduction_required_for_gate"),
        "v58_phase3_Q6_valid_rate": strict_valid,
        "v58_phase3_Q6_query_to_confirm_rate": strict.get("Q6_query_to_confirm_rate"),
        "v58_phase3_Q6_real_minus_shuffled_AUC": strict_auc,
        "v58_phase3_Q6_real_minus_no_temporal_AUC": strict_no_temporal,
        "v58_phase3_expanded_gate_pass": _gate_pass(expanded),
        "v58_phase3_expanded_Q6_entropy_reduction": expanded.get("Q6_entropy_reduction"),
        "v58_phase3_expanded_Q6_valid_rate": expanded_valid,
        "v58_phase3_expanded_Q6_real_minus_shuffled_AUC": expanded_auc,
        "v58_phase3_expanded_Q6_real_minus_no_temporal_AUC": expanded_no_temporal,
        "v58_phase3_expanded_candidate_success": reproj_success,
        "v58_phase3_expanded_candidate_conflict": reproj_conflict,
        "expanded_validity_drop_observed": validity_drop,
        "expanded_shuffled_control_drop_observed": shuffled_control_drop,
        "expanded_no_temporal_control_drop_observed": no_temporal_control_drop,
        "expanded_reprojection_failure_observed": expanded_reprojection_failed,
        "expanded_candidate_quality_drop_observed": expanded_quality_drop_observed,
        "v56_core_ARI": v56_core.get("core_ARI"),
        "v56_core_purity": v56_core.get("core_purity"),
        "v56_core_completeness": v56_core.get("core_completeness"),
        "v56_core_real_minus_shuffled_ARI": v56_core.get("real_minus_shuffled_ARI"),
        "v56_core_real_minus_no_temporal_ARI": v56_core.get("real_minus_no_temporal_ARI"),
        "v56_update_precision_diagnostic": v56_core.get("update_precision_diagnostic"),
        "v56_core_gate_pass": _gate_pass(v56_core),
        "v56_expanded_ARI": v56_tentative.get("expanded_ARI"),
        "v56_expanded_purity": v56_tentative.get("expanded_purity"),
        "v56_expanded_completeness": v56_tentative.get("expanded_completeness"),
        "v56_tentative_precision_diagnostic": v56_tentative.get("tentative_precision_diagnostic"),
        "v56_tentative_gate_pass": _gate_pass(v56_tentative),
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
        "diagnostic_label_sources": [
            "prior v58/v56 summary diagnostic metrics only",
            "no GT labels are used to build a v59 prediction in Phase0",
        ],
    }

    gate = {
        "no_missing_required_artifacts": not missing,
        "v58_phase1_dino_recall_at_3_ge_0_85": _num(fact_lock["v58_phase1_dino_recall@3"], 0.0) >= 0.85,
        "v58_phase2_deferred_count_gt_actionable_count": _int(fact_lock["v58_phase2_deferred_count"])
        > _int(fact_lock["v58_phase2_actionable_count"]),
        "v58_phase3_H3_gate_pass_is_false": fact_lock["v58_phase3_strict_gate_pass"] is False,
        "expanded_candidate_quality_drop_observed": expanded_quality_drop_observed,
    }
    gate["pass"] = bool(all(gate.values()))
    fact_lock["gate"] = gate

    metric_rows = [
        {
            "row": "v58_phase1_dino_semantic_memory",
            "history_shortlist_recall@3": fact_lock["v58_phase1_dino_recall@3"],
            "underseg_detection_AUC": fact_lock["v58_phase1_underseg_AUC"],
            "gate_pass": fact_lock["v58_phase1_gate_pass"],
        },
        {
            "row": "v58_phase2_counterfactual_explanation",
            "actionable_count": fact_lock["v58_phase2_actionable_count"],
            "deferred_count": fact_lock["v58_phase2_deferred_count"],
            "assign_precision": fact_lock["v58_phase2_assign_precision"],
            "underseg_precision": fact_lock["v58_phase2_underseg_precision"],
            "projection_completeness": fact_lock["v58_phase2_projection_completeness"],
            "false_history_update_rate": fact_lock["v58_phase2_false_history_update_rate"],
            "gate_pass": fact_lock["v58_phase2_gate_pass"],
        },
        {
            "row": "v58_phase3_strict_active_query",
            "Q6_entropy_reduction": fact_lock["v58_phase3_Q6_entropy_reduction"],
            "Q6_entropy_required": fact_lock["v58_phase3_Q6_entropy_required"],
            "Q6_valid_rate": fact_lock["v58_phase3_Q6_valid_rate"],
            "Q6_real_minus_shuffled_AUC": fact_lock["v58_phase3_Q6_real_minus_shuffled_AUC"],
            "Q6_real_minus_no_temporal_AUC": fact_lock["v58_phase3_Q6_real_minus_no_temporal_AUC"],
            "gate_pass": fact_lock["v58_phase3_strict_gate_pass"],
        },
        {
            "row": "v58_phase3_expanded_active_query",
            "Q6_entropy_reduction": fact_lock["v58_phase3_expanded_Q6_entropy_reduction"],
            "Q6_valid_rate": fact_lock["v58_phase3_expanded_Q6_valid_rate"],
            "Q6_real_minus_shuffled_AUC": fact_lock["v58_phase3_expanded_Q6_real_minus_shuffled_AUC"],
            "Q6_real_minus_no_temporal_AUC": fact_lock["v58_phase3_expanded_Q6_real_minus_no_temporal_AUC"],
            "candidate_success": fact_lock["v58_phase3_expanded_candidate_success"],
            "candidate_conflict": fact_lock["v58_phase3_expanded_candidate_conflict"],
            "gate_pass": fact_lock["v58_phase3_expanded_gate_pass"],
        },
        {
            "row": "v56_core_tentative_baseline",
            "core_ARI": fact_lock["v56_core_ARI"],
            "core_purity": fact_lock["v56_core_purity"],
            "core_completeness": fact_lock["v56_core_completeness"],
            "core_real_minus_shuffled_ARI": fact_lock["v56_core_real_minus_shuffled_ARI"],
            "core_real_minus_no_temporal_ARI": fact_lock["v56_core_real_minus_no_temporal_ARI"],
            "expanded_completeness": fact_lock["v56_expanded_completeness"],
        },
    ]
    failure_chain_rows = [
        {
            "step": "semantic_memory",
            "evidence": "DINO recall@3 clears 0.85 and underseg AUC is high",
            "metric": fact_lock["v58_phase1_dino_recall@3"],
            "interpretation": "semantic signal exists but phase gate remains partial because same-category labels are unavailable",
        },
        {
            "step": "counterfactual_explanation",
            "evidence": "deferred observations exceed actionable observations",
            "metric": f"{fact_lock['v58_phase2_deferred_count']}/{fact_lock['v58_phase2_actionable_count']}",
            "interpretation": "precision-first explanation is not a complete object-field update",
        },
        {
            "step": "strict_active_query",
            "evidence": "strict material validity is high but entropy gate fails",
            "metric": fact_lock["v58_phase3_Q6_valid_rate"],
            "interpretation": "clean material evidence exists in a narrow candidate pool",
        },
        {
            "step": "expanded_active_query",
            "evidence": "expanded candidate quality drops and reprojection conflicts rise",
            "metric": fact_lock["v58_phase3_expanded_Q6_valid_rate"],
            "interpretation": "coverage-first expansion needs manifold boundary and shortcut reasoning",
        },
    ]
    return {
        "fact_lock": fact_lock,
        "v59_phase0_metric_rows": metric_rows,
        "v59_phase0_failure_chain_rows": failure_chain_rows,
    }


def write_v59_fact_lock(
    output_root: str | Path,
    payload: dict[str, Any],
    visualization_root: str | Path = "outputs/audit/v59_visualizations/phase0",
) -> dict[str, str]:
    out = _project(output_root)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "fact_lock": out / "fact_lock.json",
        "metric_rows": out / "v59_phase0_metric_rows.csv",
        "failure_chain_rows": out / "v59_phase0_failure_chain_rows.csv",
    }
    write_json(paths["fact_lock"], payload["fact_lock"])
    write_csv(paths["metric_rows"], payload["v59_phase0_metric_rows"])
    write_csv(paths["failure_chain_rows"], payload["v59_phase0_failure_chain_rows"])
    visual_paths = write_v59_phase0_visualizations(payload, visualization_root)
    return {
        **{name: _rel(path) for name, path in paths.items()},
        **visual_paths,
    }


def write_v59_phase0_visualizations(payload: dict[str, Any], visualization_root: str | Path) -> dict[str, str]:
    root = _project(visualization_root)
    root.mkdir(parents=True, exist_ok=True)
    fact = payload["fact_lock"]
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        dashboard = root / "v59_phase0_failure_to_manifold_dashboard.png"
        fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
        axes[0].bar(
            ["actionable", "deferred"],
            [_num(fact.get("v58_phase2_actionable_count"), 0.0), _num(fact.get("v58_phase2_deferred_count"), 0.0)],
            color=["#52796F", "#B56576"],
        )
        axes[0].set_title("v58 explanation split")
        axes[0].set_ylabel("observations")
        axes[1].bar(
            ["strict valid", "expanded valid", "strict AUC", "expanded AUC"],
            [
                _num(fact.get("v58_phase3_Q6_valid_rate"), 0.0),
                _num(fact.get("v58_phase3_expanded_Q6_valid_rate"), 0.0),
                _num(fact.get("v58_phase3_Q6_real_minus_shuffled_AUC"), 0.0),
                _num(fact.get("v58_phase3_expanded_Q6_real_minus_shuffled_AUC"), 0.0),
            ],
            color=["#355070", "#6D597A", "#2A9D8F", "#E76F51"],
        )
        axes[1].set_title("strict vs expanded query quality")
        axes[1].set_ylim(0.0, 1.05)
        fig.suptitle("v59 Phase0: why manifold reasoning is needed")
        fig.tight_layout()
        fig.savefig(dashboard, dpi=160)
        plt.close(fig)

        coverage = root / "v58_deferred_coverage_map.png"
        fig, ax = plt.subplots(figsize=(7.6, 4.2))
        labels = ["strict eligible", "expanded candidates", "expanded success", "expanded conflict"]
        values = [
            _num(fact.get("v58_phase3_Q6_query_to_confirm_rate"), 0.0),
            _num(fact.get("v58_phase3_expanded_Q6_valid_rate"), 0.0),
            _num(fact.get("v58_phase3_expanded_candidate_success"), 0.0),
            _num(fact.get("v58_phase3_expanded_candidate_conflict"), 0.0),
        ]
        ax.plot(labels, values, marker="o", color="#264653")
        ax.fill_between(range(len(values)), values, color="#A8DADC", alpha=0.35)
        ax.set_ylim(0.0, 1.05)
        ax.set_title("v58 deferred coverage and conflict signals")
        ax.set_ylabel("rate")
        ax.tick_params(axis="x", labelrotation=15)
        fig.tight_layout()
        fig.savefig(coverage, dpi=160)
        plt.close(fig)
        return {
            "dashboard": _rel(dashboard),
            "deferred_coverage_map": _rel(coverage),
            "visualization_status": "created",
        }
    except Exception as exc:  # pragma: no cover - optional visualization backend
        error_path = root / "v59_phase0_visualization_error.txt"
        error_path.write_text(f"visualization unavailable: {type(exc).__name__}: {exc}\n", encoding="utf-8")
        return {
            "visualization_error": _rel(error_path),
            "visualization_status": "unavailable",
        }
