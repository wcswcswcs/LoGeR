from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .v47_common import ROOT, parse_bool, parse_float, read_json, utc_now, write_csv, write_json


DEFAULT_EMBEDDING_ROOT = "outputs/audit/v60_manifold_embedding"


@dataclass(frozen=True)
class V60RefinementConfig:
    embedding_root: str | Path = DEFAULT_EMBEDDING_ROOT
    output_root: str | Path = "outputs/audit/v60_manifold_refinement"
    visualization_root: str | Path = "outputs/audit/v60_visualizations/manifold_refinement"


def _project(path: str | Path) -> Path:
    path_obj = Path(path)
    return path_obj if path_obj.is_absolute() else ROOT / path_obj


def _rel(path: str | Path) -> str:
    path_obj = _project(path)
    try:
        return str(path_obj.relative_to(ROOT))
    except ValueError:
        return str(path_obj)


def _iter_csv(path: str | Path) -> Iterable[dict[str, str]]:
    with _project(path).open(newline="", encoding="utf-8") as handle:
        yield from csv.DictReader(handle)


def build_v60_manifold_refinement(config: V60RefinementConfig | None = None) -> dict[str, Any]:
    cfg = config or V60RefinementConfig()
    embedding_root = _project(cfg.embedding_root)
    embedding_summary = read_json(embedding_root / "embedding_summary.json")
    node_rows = list(_iter_csv(embedding_root / "node_state_rows.csv"))
    baseline = _row_from_embedding("R0_no_refinement", embedding_summary, node_rows, promoted=0)
    promotion_rows = [_promotion_attempt(node_rows, threshold) for threshold in (0.50, 0.52, 0.55, 0.58)]
    baseline.update(
        {
            "selected": True,
            "selection_reason": "tentative promotion false rate exceeds 0.15 and quarantine overcuts; keep support as tentative/shared evidence",
        }
    )
    selected_row = dict(baseline)
    rows = [baseline, *promotion_rows]
    for row in rows:
        row.setdefault("selected", False)
        row.setdefault("selection_reason", "")

    gate = {
        "quarantine_precision_ge_0_80": selected_row["quarantine_precision_diagnostic"] is not None and selected_row["quarantine_precision_diagnostic"] >= 0.80,
        "false_promotion_rate_le_0_15": selected_row["false_promotion_rate"] is not None and selected_row["false_promotion_rate"] <= 0.15,
        "core_purity_gain_ge_0_005": selected_row["core_purity_gain"] is not None and selected_row["core_purity_gain"] >= 0.005,
        "expanded_completeness_drop_le_0_04": selected_row["expanded_completeness_drop"] is not None and selected_row["expanded_completeness_drop"] <= 0.04,
        "real_minus_shuffled_ARI_drop_le_0_01": selected_row["real_minus_shuffled_change"] is not None and selected_row["real_minus_shuffled_change"] >= -0.01,
    }
    gate["pass"] = bool(all(gate.values()))
    summary = {
        "phase": "v60_manifold_refinement",
        "created_at": utc_now(),
        "diagnostic_only_bypass": True,
        "bypass_reason": "Phase3 embedding gate failed; Phase4 is run only to localize repair options, not to promote v60 as method success.",
        "selected_variant": "R0_no_refinement_keep_tentative_shared",
        "gate": gate,
        "pruned_node_count": selected_row["pruned_node_count"],
        "promoted_node_count": selected_row["promoted_node_count"],
        "quarantined_node_count": selected_row["quarantined_node_count"],
        "split_core_count": selected_row["split_core_count"],
        "duplicate_merge_count": selected_row["duplicate_merge_count"],
        "false_promotion_rate": selected_row["false_promotion_rate"],
        "quarantine_precision_diagnostic": selected_row["quarantine_precision_diagnostic"],
        "core_purity_gain": selected_row["core_purity_gain"],
        "expanded_completeness_change": selected_row["expanded_completeness_change"],
        "real_minus_shuffled_change": selected_row["real_minus_shuffled_change"],
        "real_minus_no_temporal_change": selected_row["real_minus_no_temporal_change"],
        "analysis": (
            "Phase3 repair already moved low-margin accepted paths to tentative. Non-GT margin-only promotion "
            "attempts have diagnostic false-promotion rates above the 0.15 gate, and quarantine overcuts many "
            "diagnostically correct observations. The plan-directed fix is to require active material-query evidence."
        ),
        "promotion_attempts": promotion_rows,
        "input_paths": {
            "embedding_summary": _rel(embedding_root / "embedding_summary.json"),
            "node_state_rows": _rel(embedding_root / "node_state_rows.csv"),
        },
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }
    return {"summary": summary, "refinement_rows": rows}


def write_v60_manifold_refinement(result: dict[str, Any], output_root: str | Path) -> dict[str, str]:
    root = _project(output_root)
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "refinement_summary": root / "refinement_summary.json",
        "refinement_rows": root / "refinement_rows.csv",
    }
    write_json(paths["refinement_summary"], result["summary"])
    write_csv(paths["refinement_rows"], result["refinement_rows"])
    return {name: _rel(path) for name, path in paths.items()}


def write_v60_refinement_visualizations(result: dict[str, Any], visualization_root: str | Path) -> dict[str, str]:
    root = _project(visualization_root)
    root.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rows = [row for row in result["refinement_rows"] if row["variant"].startswith("R4")]
        path = root / "manifold_refinement_promotion_false_rate.png"
        fig, ax = plt.subplots(figsize=(7.0, 4.0))
        ax.plot([row["promotion_margin_threshold"] for row in rows], [row["false_promotion_rate"] for row in rows], marker="o", color="#B56576")
        ax.axhline(0.15, color="#333333", linestyle="--", linewidth=1.0)
        ax.set_xlabel("promotion margin threshold")
        ax.set_ylabel("false promotion rate")
        ax.set_title("v60 diagnostic promotion attempts")
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        return {"promotion_false_rate_plot": _rel(path), "visualization_status": "created"}
    except Exception as exc:  # pragma: no cover
        error_path = root / "v60_refinement_visualization_error.txt"
        error_path.write_text(f"visualization unavailable: {type(exc).__name__}: {exc}\n", encoding="utf-8")
        return {"visualization_error": _rel(error_path), "visualization_status": "unavailable"}


def _row_from_embedding(variant: str, summary: dict[str, Any], node_rows: list[dict[str, str]], promoted: int) -> dict[str, Any]:
    quarantine = [row for row in node_rows if row.get("state") == "quarantine"]
    correct_quarantine = sum(1 for row in quarantine if not parse_bool(row.get("diagnostic_correct")))
    return {
        "variant": variant,
        "promotion_margin_threshold": "",
        "pruned_node_count": 0,
        "promoted_node_count": promoted,
        "quarantined_node_count": len(quarantine),
        "split_core_count": 0,
        "duplicate_merge_count": 0,
        "false_promotion_rate": 0.0 if promoted == 0 else None,
        "quarantine_precision_diagnostic": _safe_div(correct_quarantine, len(quarantine)),
        "core_purity": summary.get("core_purity"),
        "core_purity_gain": 0.0,
        "expanded_completeness": summary.get("expanded_completeness"),
        "expanded_completeness_change": 0.0,
        "expanded_completeness_drop": 0.0,
        "real_minus_shuffled_ARI": summary.get("real_minus_shuffled_ARI"),
        "real_minus_shuffled_change": 0.0,
        "real_minus_no_temporal_ARI": summary.get("real_minus_no_temporal_ARI"),
        "real_minus_no_temporal_change": 0.0,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }


def _promotion_attempt(node_rows: list[dict[str, str]], threshold: float) -> dict[str, Any]:
    candidates = [
        row
        for row in node_rows
        if row.get("state") == "tentative"
        and parse_float(row.get("posterior_top1_margin"), 0.0) >= threshold
        and not parse_bool(row.get("crosses_shortcut_or_exclusion"))
        and int(parse_float(row.get("independent_path_count"), 0.0)) >= 2
    ]
    false_count = sum(1 for row in candidates if not parse_bool(row.get("diagnostic_correct")))
    false_rate = _safe_div(false_count, len(candidates))
    return {
        "variant": f"R4_promote_tentative_margin_{threshold:.2f}",
        "promotion_margin_threshold": threshold,
        "pruned_node_count": 0,
        "promoted_node_count": len(candidates),
        "quarantined_node_count": sum(1 for row in node_rows if row.get("state") == "quarantine"),
        "split_core_count": 0,
        "duplicate_merge_count": 0,
        "false_promotion_rate": false_rate,
        "quarantine_precision_diagnostic": None,
        "core_purity": None,
        "core_purity_gain": None,
        "expanded_completeness": None,
        "expanded_completeness_change": None,
        "expanded_completeness_drop": None,
        "real_minus_shuffled_ARI": None,
        "real_minus_shuffled_change": None,
        "real_minus_no_temporal_ARI": None,
        "real_minus_no_temporal_change": None,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }


def _safe_div(num: float, denom: float) -> float | None:
    return None if denom == 0 else float(num) / float(denom)
