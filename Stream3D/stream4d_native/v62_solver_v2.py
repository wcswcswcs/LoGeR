from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .v47_common import ROOT, read_json, utc_now, write_csv, write_json
from .v61_global_embedding import _observation_explanations
from .v62_decircularization import (
    DEFAULT_CANDIDATES,
    V62DecircularizationConfig,
    build_states_from_candidates,
    load_candidate_rows,
    metric_for_states,
    no_kmat_candidate,
    typed_non_diagnostic_candidate,
)


DEFAULT_V56_CORE = "outputs/audit/v56_core_update/core_update_summary.json"
DEFAULT_V56_TENTATIVE = "outputs/audit/v56_tentative_support/tentative_support_summary.json"
DEFAULT_V61_EMBEDDING = "outputs/audit/v61_global_embedding/embedding_summary.json"


@dataclass(frozen=True)
class V62SolverV2Config:
    material_candidate_rows_path: str | Path = DEFAULT_CANDIDATES
    v56_core_summary_path: str | Path = DEFAULT_V56_CORE
    v56_tentative_summary_path: str | Path = DEFAULT_V56_TENTATIVE
    v61_embedding_summary_path: str | Path = DEFAULT_V61_EMBEDDING
    output_root: str | Path = "outputs/audit/v62_solver_v2"
    visualization_root: str | Path = "outputs/audit/v62_visualizations/solver_v2"


def _project(path: str | Path) -> Path:
    path_obj = Path(path)
    return path_obj if path_obj.is_absolute() else ROOT / path_obj


def _rel(path: str | Path) -> str:
    path_obj = _project(path)
    try:
        return str(path_obj.relative_to(ROOT))
    except ValueError:
        return str(path_obj)


def build_v62_solver_v2(config: V62SolverV2Config | None = None) -> dict[str, Any]:
    cfg = config or V62SolverV2Config()
    candidates = load_candidate_rows(cfg.material_candidate_rows_path)
    material_ids = sorted({row["material_node_id"] for row in candidates})
    v56_core = read_json(_project(cfg.v56_core_summary_path))
    v56_tentative = read_json(_project(cfg.v56_tentative_summary_path))
    v61_embedding = read_json(_project(cfg.v61_embedding_summary_path))
    v61_variants = {row["variant"]: row for row in v61_embedding.get("variant_rows", [])}

    rows: list[dict[str, Any]] = []
    rows.append(_v56_row("S0_v56_core_baseline", v56_core))
    if "M1_material_unary_only" in v61_variants:
        rows.append({"variant": "S1_v61_original_M1_material_unary_only", **_copy_metric(v61_variants["M1_material_unary_only"])})

    s2_states = build_states_from_candidates(
        candidates,
        variant="S2_K_mat_only_decircularized",
        mode="material_only",
        filter_fn=lambda row: typed_non_diagnostic_candidate(row) and row.get("has_K_mat") == "True",
        material_ids=material_ids,
    )
    s3_states = build_states_from_candidates(
        candidates,
        variant="S3_K_mat_plus_K_mask_tentative",
        mode="weak_tentative",
        filter_fn=lambda row: typed_non_diagnostic_candidate(row) and (row.get("has_K_mat") == "True" or row.get("has_K_mask") == "True"),
        material_ids=material_ids,
    )
    s4_states = build_states_from_candidates(
        candidates,
        variant="S4_K_mat_plus_K_sem_shortlist",
        mode="weak_tentative",
        filter_fn=lambda row: typed_non_diagnostic_candidate(row) and (row.get("has_K_mat") == "True" or row.get("has_K_sem") == "True"),
        material_ids=material_ids,
    )
    s5_states = build_states_from_candidates(
        candidates,
        variant="S5_K_mat_plus_shortcut_shared_modeling",
        mode="shared_modeling",
        filter_fn=typed_non_diagnostic_candidate,
        material_ids=material_ids,
    )
    s6_states = build_states_from_candidates(
        candidates,
        variant="S6_full_SOMA_Manifold_v2_solver",
        mode="shared_modeling",
        filter_fn=typed_non_diagnostic_candidate,
        material_ids=material_ids,
    )
    s7_states = [_quarantine_shared(row, "S7_full_solver_with_stress_aware_refinement") for row in s6_states]
    semantic_only_states = build_states_from_candidates(
        candidates,
        variant="semantic_only_control",
        mode="weak_tentative",
        filter_fn=lambda row: typed_non_diagnostic_candidate(row) and row.get("has_K_sem") == "True" and row.get("has_K_mat") != "True",
        material_ids=material_ids,
    )
    mask_only_states = build_states_from_candidates(
        candidates,
        variant="mask_only_control",
        mode="weak_tentative",
        filter_fn=no_kmat_candidate,
        material_ids=material_ids,
    )

    for variant, states in [
        ("S2_K_mat_only_decircularized", s2_states),
        ("S3_K_mat_plus_K_mask_tentative", s3_states),
        ("S4_K_mat_plus_K_sem_shortlist", s4_states),
        ("S5_K_mat_plus_shortcut_shared_modeling", s5_states),
        ("S6_full_SOMA_Manifold_v2_solver", s6_states),
        ("S7_full_solver_with_stress_aware_refinement", s7_states),
        ("semantic_only_control", semantic_only_states),
        ("mask_only_control", mask_only_states),
    ]:
        rows.append(metric_for_states(variant, states, v56_core, v56_tentative))

    full = next(row for row in rows if row["variant"] == "S6_full_SOMA_Manifold_v2_solver")
    s1 = next(row for row in rows if row["variant"] == "S1_v61_original_M1_material_unary_only")
    gate = {
        "full_solver_core_purity_ge_0_95": full["core_purity"] >= 0.95,
        "full_solver_core_completeness_ge_0_90": full["core_completeness"] >= 0.90,
        "full_solver_real_minus_shuffled_ARI_ge_0_30": full["real_minus_shuffled_ARI"] >= 0.30,
        "full_solver_real_minus_no_temporal_ARI_ge_0_25": full["real_minus_no_temporal_ARI"] >= 0.25,
        "full_solver_same_category_merge_rate_le_0_05": full["same_category_merge_rate"] <= 0.05,
        "full_solver_underseg_false_merge_rate_le_0_02": full["underseg_false_merge_rate"] <= 0.02,
    }
    gate["pass"] = bool(all(gate.values()))
    summary = {
        "phase": "v62_solver_v2",
        "created_at": utc_now(),
        "selected_variant": "S6_full_SOMA_Manifold_v2_solver",
        "method_note": (
            "S6 is the de-circularized typed ownership solver. S1 and S6 are intentionally reported together; "
            "if their metrics are nearly identical, the paper claim should simplify to K_mat ownership field plus typed shared/quarantine ledger."
        ),
        "full_solver_core_purity": full["core_purity"],
        "full_solver_core_completeness": full["core_completeness"],
        "full_solver_real_minus_shuffled_ARI": full["real_minus_shuffled_ARI"],
        "full_solver_real_minus_no_temporal_ARI": full["real_minus_no_temporal_ARI"],
        "S1_minus_S6_core_ARI": s1["core_ARI"] - full["core_ARI"],
        "solver_complexity_claim": "simplify_to_decircularized_K_mat_field_plus_typed_ledger"
        if abs(s1["core_ARI"] - full["core_ARI"]) <= 0.005
        else "full_solver_adds_measurable_value",
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
        "gate": gate,
        "input_paths": {
            "candidate_rows": _rel(cfg.material_candidate_rows_path),
            "v56_core_summary": _rel(cfg.v56_core_summary_path),
            "v56_tentative_summary": _rel(cfg.v56_tentative_summary_path),
            "v61_embedding_summary": _rel(cfg.v61_embedding_summary_path),
        },
    }
    return {
        "summary": summary,
        "material_state_rows": s6_states,
        "observation_explanation_rows": _observation_explanations(s6_states),
        "energy_rows": rows,
        "semantic_only_state_rows": semantic_only_states,
        "mask_only_state_rows": mask_only_states,
    }


def write_v62_solver_v2(result: dict[str, Any], output_root: str | Path) -> dict[str, str]:
    root = _project(output_root)
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "solver_summary": root / "solver_summary.json",
        "material_state_rows": root / "material_state_rows.csv",
        "observation_explanation_rows": root / "observation_explanation_rows.csv",
        "energy_rows": root / "energy_rows.csv",
    }
    write_json(paths["solver_summary"], result["summary"])
    write_csv(paths["material_state_rows"], result["material_state_rows"])
    write_csv(paths["observation_explanation_rows"], result["observation_explanation_rows"])
    write_csv(paths["energy_rows"], result["energy_rows"])
    return {name: _rel(path) for name, path in paths.items()}


def write_v62_solver_visualizations(result: dict[str, Any], visualization_root: str | Path) -> dict[str, str]:
    root = _project(visualization_root)
    root.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rows = result["energy_rows"]
        labels = [str(row["variant"]).split("_", 1)[0] for row in rows if str(row["variant"]).startswith("S")]
        values = [row["core_ARI"] for row in rows if str(row["variant"]).startswith("S")]
        path = root / "solver_variant_core_ari.png"
        fig, ax = plt.subplots(figsize=(8.5, 4.2))
        ax.bar(labels, values, color="#457B9D")
        ax.set_ylim(0, 1.05)
        ax.set_title("v62 solver variants")
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        return {"solver_variant_core_ari": _rel(path), "visualization_status": "created"}
    except Exception as exc:  # pragma: no cover
        error_path = root / "v62_solver_visualization_error.txt"
        error_path.write_text(f"visualization unavailable: {type(exc).__name__}: {exc}\n", encoding="utf-8")
        return {"visualization_error": _rel(error_path), "visualization_status": "unavailable"}


def _v56_row(variant: str, summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "variant": variant,
        "material_node_count": None,
        "assigned_material_count": summary.get("confirmed_added_component_count"),
        "confirmed_material_count": summary.get("confirmed_added_component_count"),
        "tentative_material_count": 0,
        "shared_material_count": 0,
        "quarantine_material_count": 0,
        "unknown_material_count": None,
        "core_ARI": summary.get("core_ARI"),
        "core_purity": summary.get("core_purity"),
        "core_completeness": summary.get("core_completeness"),
        "expanded_ARI": summary.get("core_ARI"),
        "expanded_purity": summary.get("core_purity"),
        "expanded_completeness": summary.get("core_completeness"),
        "real_minus_shuffled_ARI": summary.get("real_minus_shuffled_ARI"),
        "real_minus_no_temporal_ARI": summary.get("real_minus_no_temporal_ARI"),
        "same_category_merge_rate": None,
        "underseg_false_merge_rate": None,
        "conflict_rate": None,
        "duplicate_rate": None,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }


def _copy_metric(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "variant"}


def _quarantine_shared(row: dict[str, Any], variant: str) -> dict[str, Any]:
    next_row = dict(row)
    next_row["variant"] = variant
    if next_row.get("state") == "shared":
        next_row["state"] = "quarantine"
        next_row["state_reason"] = "stress_aware_shortcut_quarantine"
        next_row["predicted_history_id"] = ""
        next_row["diagnostic_exact_match"] = False
        next_row["diagnostic_contains_expected"] = False
    return next_row
