from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from .v47_common import ROOT, parse_bool, read_json, utc_now, write_csv, write_json
from .v62_decircularization import (
    V62DecircularizationConfig,
    build_v62_decircularization,
    metric_for_states,
)


DEFAULT_COMPONENT_ATOMS = "outputs/audit/v55_atoms/component_atom_rows.csv"
DEFAULT_ANCHOR_BIRTH = "outputs/audit/v55_anchor_birth/anchor_birth_rows.csv"
DEFAULT_V56_CORE = "outputs/audit/v56_core_update/core_update_summary.json"
DEFAULT_V56_TENTATIVE = "outputs/audit/v56_tentative_support/tentative_support_summary.json"


@dataclass(frozen=True)
class V62IncrementAttributionConfig:
    component_atom_rows_path: str | Path = DEFAULT_COMPONENT_ATOMS
    anchor_birth_rows_path: str | Path = DEFAULT_ANCHOR_BIRTH
    v56_core_summary_path: str | Path = DEFAULT_V56_CORE
    v56_tentative_summary_path: str | Path = DEFAULT_V56_TENTATIVE
    output_root: str | Path = "outputs/audit/v62_increment_attribution"
    visualization_root: str | Path = "outputs/audit/v62_visualizations/increment"


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


def build_v62_increment_attribution(config: V62IncrementAttributionConfig | None = None) -> dict[str, Any]:
    cfg = config or V62IncrementAttributionConfig()
    decirc = build_v62_decircularization(V62DecircularizationConfig())
    states = decirc["decircularized_material_state_rows"]
    comp_chunks = _component_chunks(cfg.component_atom_rows_path)
    anchor_components = _anchor_components(cfg.anchor_birth_rows_path)
    v56_core = read_json(_project(cfg.v56_core_summary_path))
    v56_tentative = read_json(_project(cfg.v56_tentative_summary_path))

    rows: list[dict[str, Any]] = []
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for state in states:
        novelty, evidence = _classify_novelty(state, comp_chunks, anchor_components)
        support_count = len(state.get("support_observation_ids_json") or [])
        row = {
            **state,
            "novelty_type": novelty,
            "novelty_evidence": evidence,
            "support_observation_count": support_count,
        }
        rows.append(row)
        by_type[novelty].append(row)

    metric_rows: list[dict[str, Any]] = []
    for novelty in ["anchor_known", "bridge_overlap", "update_new", "mask_only_new", "shortcut_shared", "unknown_or_low_evidence"]:
        type_states = by_type.get(novelty, [])
        if type_states:
            metric = metric_for_states(novelty, type_states, v56_core, v56_tentative)
        else:
            metric = _empty_metric(novelty)
        metric.update(
            {
                "novelty_type": novelty,
                "material_count": len(type_states),
                "support_observation_count_mean": mean([row["support_observation_count"] for row in type_states]) if type_states else None,
            }
        )
        metric_rows.append(metric)

    counts = Counter(row["novelty_type"] for row in rows)
    state_counts = Counter((row["novelty_type"], row["state"]) for row in rows)
    update_count = counts.get("update_new", 0)
    bridge_count = counts.get("bridge_overlap", 0)
    mask_only_count = counts.get("mask_only_new", 0)
    shortcut_count = counts.get("shortcut_shared", 0)
    anchor_confirmed = state_counts[("anchor_known", "confirmed")]
    update_confirmed = state_counts[("update_new", "confirmed")]
    bridge_confirmed = state_counts[("bridge_overlap", "confirmed")]
    mask_only_confirmed = state_counts[("mask_only_new", "confirmed")]
    shortcut_shared_or_quarantine = state_counts[("shortcut_shared", "shared")] + state_counts[("shortcut_shared", "quarantine")]
    summary = {
        "phase": "v62_increment_attribution",
        "created_at": utc_now(),
        "method_note": (
            "Novelty labels are derived from available artifacts: v55 component_atom chunk indices and accepted "
            "v55 anchor_birth component ids. Material rows do not carry native chunk ids, so each row records the "
            "derivation evidence."
        ),
        "material_count": len(rows),
        "novelty_counts": dict(counts),
        "new_material_gain_vs_anchor_only": _safe_div(update_confirmed + bridge_confirmed, max(anchor_confirmed, 1)),
        "update_new_material_count": update_count,
        "update_new_confirmed_rate": _safe_div(update_confirmed, update_count),
        "bridge_overlap_confirmed_rate": _safe_div(bridge_confirmed, bridge_count),
        "mask_only_new_confirmed_rate": _safe_div(mask_only_confirmed, mask_only_count) if mask_only_count else 0.0,
        "shortcut_shared_quarantine_rate": _safe_div(shortcut_shared_or_quarantine, shortcut_count),
        "shortcut_literal_quarantine_rate": _safe_div(state_counts[("shortcut_shared", "quarantine")], shortcut_count),
        "shortcut_shared_rate": _safe_div(state_counts[("shortcut_shared", "shared")], shortcut_count),
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
        "gate": {},
        "input_paths": {
            "decircularized_material_state_rows": "outputs/audit/v62_decircularization/decircularized_material_state_rows.csv",
            "component_atom_rows": _rel(cfg.component_atom_rows_path),
            "anchor_birth_rows": _rel(cfg.anchor_birth_rows_path),
        },
    }
    gate = {
        "update_new_material_count_gt_0": update_count > 0,
        "update_new_confirmed_rate_ge_0_50": summary["update_new_confirmed_rate"] >= 0.50,
        "update_new_core_purity_ge_0_90": _metric_value(metric_rows, "update_new", "core_purity") >= 0.90,
        "bridge_overlap_confirmed_rate_ge_0_70": summary["bridge_overlap_confirmed_rate"] >= 0.70,
        "mask_only_new_confirmed_rate_le_update_new": summary["mask_only_new_confirmed_rate"] <= summary["update_new_confirmed_rate"],
        "shortcut_shared_or_quarantine_rate_ge_0_80": summary["shortcut_shared_quarantine_rate"] >= 0.80,
    }
    gate["pass"] = bool(all(gate.values()))
    summary["gate"] = gate
    return {"summary": summary, "novelty_material_rows": rows, "novelty_metric_rows": metric_rows}


def write_v62_increment_attribution(result: dict[str, Any], output_root: str | Path) -> dict[str, str]:
    root = _project(output_root)
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "increment_summary": root / "increment_summary.json",
        "novelty_material_rows": root / "novelty_material_rows.csv",
        "novelty_metric_rows": root / "novelty_metric_rows.csv",
    }
    write_json(paths["increment_summary"], result["summary"])
    write_csv(paths["novelty_material_rows"], result["novelty_material_rows"])
    write_csv(paths["novelty_metric_rows"], result["novelty_metric_rows"])
    return {name: _rel(path) for name, path in paths.items()}


def write_v62_increment_visualizations(result: dict[str, Any], visualization_root: str | Path) -> dict[str, str]:
    root = _project(visualization_root)
    root.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        counts = result["summary"]["novelty_counts"]
        labels = list(counts)
        values = [counts[label] for label in labels]
        path = root / "ownership_novelty_breakdown.png"
        fig, ax = plt.subplots(figsize=(8.5, 4.2))
        ax.bar(labels, values, color="#2A9D8F")
        ax.set_title("v62 material novelty attribution")
        ax.tick_params(axis="x", labelrotation=20)
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)

        gain_path = root / "new_material_gain_by_scene.png"
        scene_counts = Counter(row["scene"] for row in result["novelty_material_rows"] if row["novelty_type"] in {"bridge_overlap", "update_new"} and row["state"] == "confirmed")
        scenes = list(scene_counts)[:20]
        fig, ax = plt.subplots(figsize=(8.5, 4.2))
        ax.bar(scenes, [scene_counts[scene] for scene in scenes], color="#E9C46A")
        ax.set_title("confirmed bridge/update materials by scene")
        ax.tick_params(axis="x", labelrotation=45)
        fig.tight_layout()
        fig.savefig(gain_path, dpi=160)
        plt.close(fig)
        return {"ownership_novelty_breakdown": _rel(path), "new_material_gain_by_scene": _rel(gain_path), "visualization_status": "created"}
    except Exception as exc:  # pragma: no cover
        error_path = root / "v62_increment_visualization_error.txt"
        error_path.write_text(f"visualization unavailable: {type(exc).__name__}: {exc}\n", encoding="utf-8")
        return {"visualization_error": _rel(error_path), "visualization_status": "unavailable"}


def _component_chunks(path: str | Path) -> dict[str, set[str]]:
    chunks: dict[str, set[str]] = defaultdict(set)
    for row in _iter_csv(path):
        component_id = row.get("component_id", "")
        if component_id:
            chunks[component_id].add(row.get("chunk_index", ""))
    return chunks


def _anchor_components(path: str | Path) -> set[str]:
    components: set[str] = set()
    for row in _iter_csv(path):
        if not parse_bool(row.get("accepted_birth")):
            continue
        try:
            parsed = json.loads(row.get("component_ids", "[]"))
        except json.JSONDecodeError:
            parsed = []
        components.update(str(item) for item in parsed)
    return components


def _classify_novelty(state: dict[str, Any], comp_chunks: dict[str, set[str]], anchor_components: set[str]) -> tuple[str, str]:
    component_id = state.get("component_id", "")
    chunks = comp_chunks.get(component_id, set())
    has_kmat = bool(state.get("has_K_mat"))
    has_kmask = bool(state.get("has_K_mask"))
    has_underseg = bool(state.get("has_K_underseg"))
    if state.get("state") == "shared" or (has_underseg and not has_kmat):
        return "shortcut_shared", f"state={state.get('state')}; has_K_underseg={has_underseg}; has_K_mat={has_kmat}"
    if component_id in anchor_components and chunks <= {"000"}:
        return "anchor_known", f"component in accepted anchor_birth; chunks={sorted(chunks)}"
    if "000" in chunks and any(chunk != "000" for chunk in chunks):
        return "bridge_overlap", f"component visible in anchor and update chunks; chunks={sorted(chunks)}"
    if chunks and "000" not in chunks:
        return "update_new", f"component only visible in non-anchor chunks; chunks={sorted(chunks)}"
    if has_kmask and not has_kmat:
        return "mask_only_new", "K_mask evidence without K_mat"
    return "unknown_or_low_evidence", f"component chunks={sorted(chunks)}; has_K_mat={has_kmat}; has_K_mask={has_kmask}"


def _empty_metric(novelty: str) -> dict[str, Any]:
    return {
        "variant": novelty,
        "assigned_material_count": 0,
        "confirmed_material_count": 0,
        "tentative_material_count": 0,
        "shared_material_count": 0,
        "quarantine_material_count": 0,
        "unknown_material_count": 0,
        "core_ARI": 0.0,
        "core_purity": 0.0,
        "core_completeness": 0.0,
        "real_minus_shuffled_ARI": 0.0,
        "same_category_merge_rate": 0.0,
        "underseg_false_merge_rate": 0.0,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }


def _safe_div(num: int | float, denom: int | float) -> float:
    return 0.0 if float(denom) == 0.0 else float(num) / float(denom)


def _metric_value(rows: list[dict[str, Any]], novelty: str, key: str) -> float:
    for row in rows:
        if row.get("novelty_type") == novelty:
            return float(row.get(key) or 0.0)
    return 0.0


