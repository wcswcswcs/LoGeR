from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .v47_common import ROOT, read_json, utc_now, write_csv, write_json
from .v62_increment_attribution import build_v62_increment_attribution


DEFAULT_V61_QUERY = "outputs/audit/v61_manifold_query/query_summary.json"


@dataclass(frozen=True)
class V62ActiveQueryRefreshConfig:
    v61_query_summary_path: str | Path = DEFAULT_V61_QUERY
    output_root: str | Path = "outputs/audit/v62_active_query_refresh"
    visualization_root: str | Path = "outputs/audit/v62_visualizations/query_refresh"


def _project(path: str | Path) -> Path:
    path_obj = Path(path)
    return path_obj if path_obj.is_absolute() else ROOT / path_obj


def _rel(path: str | Path) -> str:
    path_obj = _project(path)
    try:
        return str(path_obj.relative_to(ROOT))
    except ValueError:
        return str(path_obj)


def build_v62_active_query_refresh(config: V62ActiveQueryRefreshConfig | None = None) -> dict[str, Any]:
    cfg = config or V62ActiveQueryRefreshConfig()
    increment = build_v62_increment_attribution()
    material_rows = increment["novelty_material_rows"]
    v61_query = read_json(_project(cfg.v61_query_summary_path))
    existing_pool_count = int(v61_query.get("candidate_pool_count") or 440)
    candidates: list[dict[str, Any]] = []
    for row in material_rows:
        source = _query_source(row)
        if not source:
            continue
        candidates.append(
            {
                "query_candidate_id": f"v62q_{len(candidates):08d}",
                "material_node_id": row["material_node_id"],
                "scene": row.get("scene", ""),
                "component_id": row.get("component_id", ""),
                "candidate_source": source,
                "state": row.get("state", ""),
                "novelty_type": row.get("novelty_type", ""),
                "support_observation_count": row.get("support_observation_count", 0),
                "has_material_boundary_source": source in {"shared_shortcut_boundary", "update_new_low_support", "bridge_low_support"},
                "has_existing_query_outcome": False,
                "valid_material_evidence": None,
                "confirm_or_quarantine_outcome": None,
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic_labels": False,
            }
        )

    source_counts = Counter(row["candidate_source"] for row in candidates)
    candidate_pool_count = len(candidates)
    measured_rows = [row for row in candidates if row["has_existing_query_outcome"]]
    valid_upper = None
    confirm_upper = None
    if measured_rows:
        valid_upper = sum(1 for row in measured_rows if row["valid_material_evidence"]) / len(measured_rows)
        confirm_upper = sum(1 for row in measured_rows if row["confirm_or_quarantine_outcome"]) / len(measured_rows)

    gate = {
        "candidate_pool_count_ge_3x_existing": candidate_pool_count >= 3 * existing_pool_count,
        "valid_candidate_upper_bound_ge_0_50": valid_upper is not None and valid_upper >= 0.50,
        "confirm_or_quarantine_upper_bound_ge_0_45": confirm_upper is not None and confirm_upper >= 0.45,
        "has_material_boundary_or_shortcut_source": any(row["has_material_boundary_source"] for row in candidates),
    }
    gate["pass"] = bool(all(gate.values()))
    summary = {
        "phase": "v62_active_query_refresh",
        "created_at": utc_now(),
        "method_note": (
            "Phase 5A expands a material-node query candidate pool from v62 ownership states. It does not execute new D4RT tracking. "
            "Because these new candidates have no query outcomes, upper-bound validity cannot be measured without Phase 5B."
        ),
        "existing_pool_count": existing_pool_count,
        "candidate_pool_count": candidate_pool_count,
        "candidate_source_breakdown": dict(source_counts),
        "valid_candidate_upper_bound": valid_upper,
        "confirm_or_quarantine_upper_bound": confirm_upper,
        "real_minus_shuffled_query_AUC": None,
        "real_minus_no_temporal_query_AUC": None,
        "phase5b_real_D4RT_query_run": False,
        "blocker": "new_material_query_outcomes_missing_without_real_D4RT_query",
        "repair_attempt": "constructed expanded non-mask-only candidate pool with material boundary, shared shortcut, update_new, and low-support sources",
        "claim_status": "diagnostic_candidate_pool_only_no_active_query_claim",
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": False,
        "gate": gate,
        "input_paths": {
            "increment_material_rows": "outputs/audit/v62_increment_attribution/novelty_material_rows.csv",
            "v61_query_summary": _rel(cfg.v61_query_summary_path),
        },
    }
    query_result_rows: list[dict[str, Any]] = []
    return {"summary": summary, "query_candidate_rows": candidates, "query_result_rows": query_result_rows}


def write_v62_active_query_refresh(result: dict[str, Any], output_root: str | Path) -> dict[str, str]:
    root = _project(output_root)
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "query_refresh_summary": root / "query_refresh_summary.json",
        "query_candidate_rows": root / "query_candidate_rows.csv",
        "query_result_rows": root / "query_result_rows.csv",
    }
    write_json(paths["query_refresh_summary"], result["summary"])
    write_csv(paths["query_candidate_rows"], result["query_candidate_rows"])
    write_csv(paths["query_result_rows"], result["query_result_rows"])
    return {name: _rel(path) for name, path in paths.items()}


def write_v62_query_visualizations(result: dict[str, Any], visualization_root: str | Path) -> dict[str, str]:
    root = _project(visualization_root)
    root.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        counts = result["summary"]["candidate_source_breakdown"]
        labels = list(counts)
        values = [counts[label] for label in labels]
        path = root / "query_candidate_source_breakdown.png"
        fig, ax = plt.subplots(figsize=(8.5, 4.2))
        ax.bar(labels, values, color="#F4A261")
        ax.set_title("v62 query refresh candidate sources")
        ax.tick_params(axis="x", labelrotation=25)
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        return {"query_candidate_source_breakdown": _rel(path), "visualization_status": "created"}
    except Exception as exc:  # pragma: no cover
        error_path = root / "v62_query_visualization_error.txt"
        error_path.write_text(f"visualization unavailable: {type(exc).__name__}: {exc}\n", encoding="utf-8")
        return {"visualization_error": _rel(error_path), "visualization_status": "unavailable"}


def _query_source(row: dict[str, Any]) -> str:
    novelty = row.get("novelty_type")
    state = row.get("state")
    support = int(row.get("support_observation_count") or 0)
    if novelty == "shortcut_shared" or state == "shared":
        return "shared_shortcut_boundary"
    if novelty == "update_new" and support <= 3:
        return "update_new_low_support"
    if novelty == "bridge_overlap" and support <= 3:
        return "bridge_low_support"
    if state == "tentative":
        return "state_tentative"
    if novelty == "update_new" and support <= 8:
        return "update_new_uncertain"
    return ""


