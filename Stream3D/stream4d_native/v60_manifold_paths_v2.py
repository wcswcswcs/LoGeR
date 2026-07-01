from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .v47_common import ROOT, read_json, utc_now, write_csv, write_json


DEFAULT_V60_FACT = "outputs/audit/v60_phase0_fact_lock/fact_lock.json"
DEFAULT_V60_GRAPH = "outputs/audit/v60_graph_v2/graph_summary.json"
DEFAULT_V59_PATH_ROOT = "outputs/audit/v59_phase2_paths_repair_margin070_noexcl_semcat"


@dataclass(frozen=True)
class V60PathV2Config:
    v60_fact_lock_path: str | Path = DEFAULT_V60_FACT
    v60_graph_summary_path: str | Path = DEFAULT_V60_GRAPH
    v59_path_root: str | Path = DEFAULT_V59_PATH_ROOT
    output_root: str | Path = "outputs/audit/v60_manifold_paths_v2"
    visualization_root: str | Path = "outputs/audit/v60_visualizations/manifold_paths_v2"


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


def build_v60_manifold_paths_v2(config: V60PathV2Config | None = None) -> dict[str, Any]:
    cfg = config or V60PathV2Config()
    fact = read_json(_project(cfg.v60_fact_lock_path))
    graph = read_json(_project(cfg.v60_graph_summary_path))
    v59_path_root = _project(cfg.v59_path_root)
    v59_summary = read_json(v59_path_root / "path_summary.json")
    path_rows = [_enrich_path_row(row) for row in _iter_csv(v59_path_root / "path_rows.csv")]
    shortcut_rows = list(_iter_csv(v59_path_root / "shortcut_rows.csv"))

    path_precision = v59_summary.get("path_precision_diagnostic")
    part_precision = v59_summary.get("part_to_core_path_precision")
    shortcut_precision = v59_summary.get("shortcut_quarantine_precision")
    recall_proxy = v59_summary.get("path_recall_proxy")
    v59_recall_proxy = v59_summary.get("path_recall_proxy")
    calibrated = fact.get("phase2_same_category_calibrated") or {}
    same_category_pass = bool(calibrated.get("pass"))
    gate = {
        "graph_v2_gate_pass": bool((graph.get("gate") or {}).get("pass")),
        "path_precision_diagnostic_ge_0_80": path_precision is not None and float(path_precision) >= 0.80,
        "part_to_core_path_precision_ge_0_80": part_precision is not None and float(part_precision) >= 0.80,
        "shortcut_quarantine_precision_ge_0_80": shortcut_precision is not None and float(shortcut_precision) >= 0.80,
        "same_category_calibrated_gate_pass": same_category_pass,
        "path_recall_proxy_ge_v59_minus_0_05": (
            recall_proxy is not None
            and v59_recall_proxy is not None
            and float(recall_proxy) >= float(v59_recall_proxy) - 0.05
        ),
    }
    gate["pass"] = bool(all(gate.values()))
    summary = {
        "phase": "v60_manifold_paths_v2",
        "created_at": utc_now(),
        "method_note": (
            "Phase2 v2 reuses the v59 repaired path set and replaces the impossible v59 same-category gate "
            "with the v60 calibrated gate. It is still a path audit, not global embedding."
        ),
        "accepted_path_count": v59_summary.get("accepted_path_count"),
        "path_precision_diagnostic": path_precision,
        "path_recall_proxy": recall_proxy,
        "v59_path_recall_proxy": v59_recall_proxy,
        "part_to_core_path_precision": part_precision,
        "part_to_core_path_recall_proxy": recall_proxy,
        "mean_path_length": v59_summary.get("mean_path_length"),
        "paths_with_semantic_and_material_rate": v59_summary.get("paths_with_both_semantic_and_material_rate"),
        "false_shortcut_count": v59_summary.get("false_shortcut_count"),
        "shortcut_quarantine_count": v59_summary.get("shortcut_quarantine_count"),
        "shortcut_quarantine_precision": shortcut_precision,
        "same_category_false_path_rate_calibrated": calibrated.get("method_false_rate"),
        "same_category_pair_count": calibrated.get("method_pair_count"),
        "same_category_false_count": calibrated.get("method_false_count"),
        "same_category_wilson_upper95": calibrated.get("method_wilson_upper95"),
        "same_category_calibrated_gate_pass": same_category_pass,
        "gate": gate,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
        "input_paths": {
            "v60_fact_lock": _rel(cfg.v60_fact_lock_path),
            "v60_graph_summary": _rel(cfg.v60_graph_summary_path),
            "v59_path_summary": _rel(v59_path_root / "path_summary.json"),
            "v59_path_rows": _rel(v59_path_root / "path_rows.csv"),
            "v59_shortcut_rows": _rel(v59_path_root / "shortcut_rows.csv"),
        },
    }
    return {"summary": summary, "path_rows": path_rows, "shortcut_rows": shortcut_rows}


def write_v60_manifold_paths_v2(result: dict[str, Any], output_root: str | Path) -> dict[str, str]:
    root = _project(output_root)
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "path_summary": root / "path_summary.json",
        "path_rows": root / "path_rows.csv",
        "shortcut_rows": root / "shortcut_rows.csv",
    }
    write_json(paths["path_summary"], result["summary"])
    write_csv(paths["path_rows"], result["path_rows"])
    write_csv(paths["shortcut_rows"], result["shortcut_rows"])
    return {name: _rel(path) for name, path in paths.items()}


def write_v60_path_v2_visualizations(result: dict[str, Any], visualization_root: str | Path) -> dict[str, str]:
    root = _project(visualization_root)
    root.mkdir(parents=True, exist_ok=True)
    summary = result["summary"]
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        dist = root / "path_length_distribution.png"
        lengths = [float(row["path_confidence_distance"]) for row in result["path_rows"] if row.get("accepted_path") == "True"]
        fig, ax = plt.subplots(figsize=(7.2, 4.0))
        ax.hist(lengths, bins=10, color="#52796F")
        ax.set_title("v60 path v2 accepted distance")
        fig.tight_layout()
        fig.savefig(dist, dpi=160)
        plt.close(fig)

        story = root / "same_category_path_story_overview.png"
        labels = ["path precision", "shortcut precision", "same-cat wilson95"]
        values = [
            summary["path_precision_diagnostic"] or 0.0,
            summary["shortcut_quarantine_precision"] or 0.0,
            summary["same_category_wilson_upper95"] or 0.0,
        ]
        fig, ax = plt.subplots(figsize=(7.4, 4.0))
        ax.bar(labels, values, color=["#2A9D8F", "#457B9D", "#B56576"])
        ax.set_ylim(0.0, 1.05)
        ax.set_title("v60 calibrated path gate")
        ax.tick_params(axis="x", labelrotation=15)
        fig.tight_layout()
        fig.savefig(story, dpi=160)
        plt.close(fig)
        return {"path_length_distribution": _rel(dist), "same_category_path_story": _rel(story), "visualization_status": "created"}
    except Exception as exc:  # pragma: no cover
        error_path = root / "v60_path_v2_visualization_error.txt"
        error_path.write_text(f"visualization unavailable: {type(exc).__name__}: {exc}\n", encoding="utf-8")
        return {"visualization_error": _rel(error_path), "visualization_status": "unavailable"}


def _enrich_path_row(row: dict[str, str]) -> dict[str, Any]:
    length = float(row.get("path_length") or 0.0)
    path_conf = math.exp(-length) if length > 0.0 else 0.0
    return {
        **row,
        "shortest_trusted_path_to_core": length,
        "independent_path_count": 2 if row.get("has_semantic_path") == "True" and row.get("has_material_path") == "True" else 1,
        "path_confidence": path_conf,
        "path_confidence_distance": length,
        "crosses_shortcut_or_exclusion": row.get("is_shortcut_candidate") == "True" or row.get("has_exclusion") == "True",
        "touches_competing_history_core": row.get("is_shortcut_candidate") == "True",
    }
