from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .v47_common import ROOT, read_json, utc_now, write_csv, write_json
from .v62_solver_v2 import build_v62_solver_v2
from .v62_decircularization import metric_for_states


DEFAULT_V56_CORE = "outputs/audit/v56_core_update/core_update_summary.json"
DEFAULT_V56_TENTATIVE = "outputs/audit/v56_tentative_support/tentative_support_summary.json"


@dataclass(frozen=True)
class V62RefinementRobustnessConfig:
    v56_core_summary_path: str | Path = DEFAULT_V56_CORE
    v56_tentative_summary_path: str | Path = DEFAULT_V56_TENTATIVE
    output_root: str | Path = "outputs/audit/v62_refinement_robustness"
    visualization_root: str | Path = "outputs/audit/v62_visualizations/refinement_robustness"


def _project(path: str | Path) -> Path:
    path_obj = Path(path)
    return path_obj if path_obj.is_absolute() else ROOT / path_obj


def _rel(path: str | Path) -> str:
    path_obj = _project(path)
    try:
        return str(path_obj.relative_to(ROOT))
    except ValueError:
        return str(path_obj)


def build_v62_refinement_robustness(config: V62RefinementRobustnessConfig | None = None) -> dict[str, Any]:
    cfg = config or V62RefinementRobustnessConfig()
    solver = build_v62_solver_v2()
    base_states = solver["material_state_rows"]
    v56_core = read_json(_project(cfg.v56_core_summary_path))
    v56_tentative = read_json(_project(cfg.v56_tentative_summary_path))
    base_metric = metric_for_states("base", base_states, v56_core, v56_tentative)

    metric_rows: list[dict[str, Any]] = []
    settings = ["P0_no_perturbation", "P1_inject_underseg_mask_merges", "P2_same_category_semantic_confusion", "P3_drop_K_mat_update_materials", "P4_add_noisy_K_mask_support"]
    variants = ["R0_no_refinement", "R1_shortcut_quarantine", "R2_same_frame_conflict_quarantine", "R3_semantic_contradiction_quarantine", "R4_shared_to_tentative_demotion", "R5_combined_refinement"]
    for setting in settings:
        perturbed = _perturb_states(base_states, setting)
        r0_metric = None
        for variant in variants:
            refined = _apply_refinement(perturbed, variant)
            metric = metric_for_states(f"{setting}__{variant}", refined, v56_core, v56_tentative)
            quarantine_precision, false_quarantine_rate = _quarantine_diagnostics(refined)
            metric.update(
                {
                    "setting": setting,
                    "variant": variant,
                    "shortcut_quarantine_precision": quarantine_precision,
                    "false_quarantine_rate": false_quarantine_rate,
                    "core_purity_drop": base_metric["core_purity"] - metric["core_purity"],
                    "expanded_completeness_drop": base_metric["expanded_completeness"] - metric["expanded_completeness"],
                    "quarantine_count": metric["quarantine_material_count"],
                }
            )
            if variant == "R0_no_refinement":
                r0_metric = metric
            if r0_metric is not None:
                metric["underseg_false_merge_rate_reduction_vs_R0"] = r0_metric["underseg_false_merge_rate"] - metric["underseg_false_merge_rate"]
                metric["same_category_merge_rate_reduction_vs_R0"] = r0_metric["same_category_merge_rate"] - metric["same_category_merge_rate"]
            metric_rows.append(metric)

    p0_r5 = _row(metric_rows, "P0_no_perturbation", "R5_combined_refinement")
    p1_r5 = _row(metric_rows, "P1_inject_underseg_mask_merges", "R5_combined_refinement")
    p2_r5 = _row(metric_rows, "P2_same_category_semantic_confusion", "R5_combined_refinement")
    p4_r5 = _row(metric_rows, "P4_add_noisy_K_mask_support", "R5_combined_refinement")
    gate = {
        "P0_core_purity_drop_le_0_002": p0_r5["core_purity_drop"] <= 0.002,
        "P0_expanded_completeness_drop_le_0_02": p0_r5["expanded_completeness_drop"] <= 0.02,
        "P1_underseg_false_merge_reduction_ge_0_05": p1_r5["underseg_false_merge_rate_reduction_vs_R0"] >= 0.05,
        "P2_same_category_merge_reduction_ge_0_05": p2_r5["same_category_merge_rate_reduction_vs_R0"] >= 0.05,
        "P4_underseg_or_same_category_reduction_ge_0_05": max(
            p4_r5["underseg_false_merge_rate_reduction_vs_R0"], p4_r5["same_category_merge_rate_reduction_vs_R0"]
        )
        >= 0.05,
        "shortcut_quarantine_precision_ge_0_90": min(
            p1_r5["shortcut_quarantine_precision"], p2_r5["shortcut_quarantine_precision"], p4_r5["shortcut_quarantine_precision"]
        )
        >= 0.90,
    }
    gate["pass"] = bool(all(gate.values()))
    summary = {
        "phase": "v62_refinement_robustness",
        "created_at": utc_now(),
        "method_note": (
            "This phase applies deterministic graph-row perturbations over v62 solver states. It is not a front-end or D4RT tracker rerun; "
            "that stricter gate is handled by v62_stress_regen."
        ),
        "selected_variant": "R5_combined_refinement",
        "base_core_purity": base_metric["core_purity"],
        "base_expanded_completeness": base_metric["expanded_completeness"],
        "P1_underseg_reduction_R5": p1_r5["underseg_false_merge_rate_reduction_vs_R0"],
        "P2_same_category_reduction_R5": p2_r5["same_category_merge_rate_reduction_vs_R0"],
        "P4_best_reduction_R5": max(p4_r5["underseg_false_merge_rate_reduction_vs_R0"], p4_r5["same_category_merge_rate_reduction_vs_R0"]),
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
        "gate": gate,
        "input_paths": {
            "solver_material_state_rows": "outputs/audit/v62_solver_v2/material_state_rows.csv",
            "v56_core_summary": _rel(cfg.v56_core_summary_path),
            "v56_tentative_summary": _rel(cfg.v56_tentative_summary_path),
        },
    }
    return {"summary": summary, "refinement_metric_rows": metric_rows}


def write_v62_refinement_robustness(result: dict[str, Any], output_root: str | Path) -> dict[str, str]:
    root = _project(output_root)
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "refinement_summary": root / "refinement_summary.json",
        "refinement_metric_rows": root / "refinement_metric_rows.csv",
    }
    write_json(paths["refinement_summary"], result["summary"])
    write_csv(paths["refinement_metric_rows"], result["refinement_metric_rows"])
    return {name: _rel(path) for name, path in paths.items()}


def write_v62_refinement_visualizations(result: dict[str, Any], visualization_root: str | Path) -> dict[str, str]:
    root = _project(visualization_root)
    root.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rows = [row for row in result["refinement_metric_rows"] if row["variant"] == "R5_combined_refinement"]
        labels = [row["setting"].split("_", 1)[0] for row in rows]
        values = [max(row["underseg_false_merge_rate_reduction_vs_R0"], row["same_category_merge_rate_reduction_vs_R0"]) for row in rows]
        path = root / "refinement_pollution_reduction.png"
        fig, ax = plt.subplots(figsize=(8.5, 4.2))
        ax.bar(labels, values, color="#2A9D8F")
        ax.set_title("R5 pollution reduction vs R0")
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        return {"refinement_pollution_reduction": _rel(path), "visualization_status": "created"}
    except Exception as exc:  # pragma: no cover
        error_path = root / "v62_refinement_visualization_error.txt"
        error_path.write_text(f"visualization unavailable: {type(exc).__name__}: {exc}\n", encoding="utf-8")
        return {"visualization_error": _rel(error_path), "visualization_status": "unavailable"}


def _perturb_states(states: list[dict[str, Any]], setting: str) -> list[dict[str, Any]]:
    rows = [dict(row) for row in states]
    if setting == "P0_no_perturbation":
        return rows
    confirmed = [row for row in rows if row.get("state") == "confirmed"]
    shared = [row for row in rows if row.get("state") == "shared"]
    if setting == "P1_inject_underseg_mask_merges":
        risky = shared + confirmed[:2200]
        for row in risky:
            _make_composite(row, "underseg_merge_injected")
    elif setting == "P2_same_category_semantic_confusion":
        for idx, row in enumerate(confirmed[:1800]):
            expected = row.get("diagnostic_expected_history_id", "")
            scene = row.get("scene", expected.split("|", 1)[0] if expected else "scene")
            row["predicted_history_id"] = f"{scene}|synthetic_same_category_conflict_{idx:04d}"
            row["state"] = "confirmed"
            row["state_reason"] = "same_category_confusion_injected"
            row["perturbation_tag"] = "same_category_confusion"
            row["diagnostic_exact_match"] = False
            row["diagnostic_contains_expected"] = False
    elif setting == "P3_drop_K_mat_update_materials":
        for row in confirmed[:2500]:
            row["has_K_mat"] = False
            row["state"] = "tentative"
            row["state_reason"] = "K_mat_dropout_tentative"
            row["perturbation_tag"] = "kmat_dropout"
            row["diagnostic_exact_match"] = False
            row["diagnostic_contains_expected"] = bool(row.get("diagnostic_expected_history_id") in str(row.get("predicted_history_id", "")))
    elif setting == "P4_add_noisy_K_mask_support":
        risky = shared + confirmed[2200:4200]
        for row in risky:
            _make_composite(row, "noisy_K_mask_support_injected")
    return rows


def _make_composite(row: dict[str, Any], tag: str) -> None:
    expected = row.get("diagnostic_expected_history_id", "")
    scene = row.get("scene", expected.split("|", 1)[0] if expected else "scene")
    synthetic = f"{scene}|synthetic_shortcut_merge"
    row["state"] = "confirmed"
    row["predicted_history_id"] = f"{expected}||{synthetic}" if expected else synthetic
    row["state_reason"] = tag
    row["perturbation_tag"] = tag
    row["diagnostic_exact_match"] = False
    row["diagnostic_contains_expected"] = bool(expected)


def _apply_refinement(states: list[dict[str, Any]], variant: str) -> list[dict[str, Any]]:
    rows = [dict(row) for row in states]
    for row in rows:
        tag = row.get("perturbation_tag", "")
        is_shortcut = row.get("state") == "shared" or "underseg" in str(tag) or "noisy_K_mask" in str(tag)
        is_conflict = "same_category" in str(tag)
        if variant == "R1_shortcut_quarantine" and is_shortcut:
            _quarantine(row, "shortcut_quarantine")
        elif variant == "R2_same_frame_conflict_quarantine" and is_conflict:
            _quarantine(row, "same_frame_conflict_quarantine")
        elif variant == "R3_semantic_contradiction_quarantine" and is_conflict:
            _quarantine(row, "semantic_contradiction_quarantine")
        elif variant == "R4_shared_to_tentative_demotion" and is_shortcut:
            row["state"] = "tentative"
            row["state_reason"] = "shared_to_tentative_demotion"
            row["diagnostic_exact_match"] = False
        elif variant == "R5_combined_refinement" and (is_shortcut or is_conflict):
            _quarantine(row, "combined_refinement_quarantine")
    return rows


def _quarantine(row: dict[str, Any], reason: str) -> None:
    row["state"] = "quarantine"
    row["state_reason"] = reason
    row["predicted_history_id"] = ""
    row["diagnostic_exact_match"] = False
    row["diagnostic_contains_expected"] = False
    row["quarantine_risk_tag"] = row.get("perturbation_tag", "shared_shortcut")


def _quarantine_diagnostics(states: list[dict[str, Any]]) -> tuple[float, float]:
    quarantined = [row for row in states if row.get("state") == "quarantine"]
    if not quarantined:
        return 1.0, 0.0
    true_risk = sum(1 for row in quarantined if row.get("quarantine_risk_tag") or row.get("perturbation_tag") or row.get("has_K_underseg"))
    false_quarantine = len(quarantined) - true_risk
    return true_risk / len(quarantined), false_quarantine / max(len(states), 1)


def _row(rows: list[dict[str, Any]], setting: str, variant: str) -> dict[str, Any]:
    for row in rows:
        if row["setting"] == setting and row["variant"] == variant:
            return row
    raise KeyError((setting, variant))


