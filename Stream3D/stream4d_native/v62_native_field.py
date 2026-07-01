from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .v47_common import ROOT, read_json, utc_now, write_csv, write_json
from .v62_solver_v2 import (
    DEFAULT_V56_CORE,
    DEFAULT_V56_TENTATIVE,
    DEFAULT_V61_EMBEDDING,
    V62SolverV2Config,
    build_v62_solver_v2,
)
from .v62_decircularization import DEFAULT_CANDIDATES


DEFAULT_V61_NATIVE = "outputs/audit/v61_native_field/native_field_summary.json"


@dataclass(frozen=True)
class V62NativeFieldConfig:
    v61_native_summary_path: str | Path = DEFAULT_V61_NATIVE
    material_candidate_rows_path: str | Path = DEFAULT_CANDIDATES
    v56_core_summary_path: str | Path = DEFAULT_V56_CORE
    v56_tentative_summary_path: str | Path = DEFAULT_V56_TENTATIVE
    v61_embedding_summary_path: str | Path = DEFAULT_V61_EMBEDDING
    output_root: str | Path = "outputs/audit/v62_native_field"
    visualization_root: str | Path = "outputs/audit/v62_visualizations/native_field"


def _project(path: str | Path) -> Path:
    path_obj = Path(path)
    return path_obj if path_obj.is_absolute() else ROOT / path_obj


def _rel(path: str | Path) -> str:
    path_obj = _project(path)
    try:
        return str(path_obj.relative_to(ROOT))
    except ValueError:
        return str(path_obj)


def build_v62_native_field(config: V62NativeFieldConfig | None = None) -> dict[str, Any]:
    cfg = config or V62NativeFieldConfig()
    v61_native = read_json(_project(cfg.v61_native_summary_path))
    solver = build_v62_solver_v2(
        V62SolverV2Config(
            material_candidate_rows_path=cfg.material_candidate_rows_path,
            v56_core_summary_path=cfg.v56_core_summary_path,
            v56_tentative_summary_path=cfg.v56_tentative_summary_path,
            v61_embedding_summary_path=cfg.v61_embedding_summary_path,
        )
    )
    states = solver["material_state_rows"]
    component_rows = []
    carrier_rows = []
    for row in states:
        support_ids = _json_list(row.get("support_observation_ids_json") or row.get("support_observation_ids"))
        component_rows.append(
            {
                "scene": row.get("scene", ""),
                "component_id": row.get("component_id", ""),
                "material_node_id": row.get("material_node_id", ""),
                "state": row.get("state", ""),
                "predicted_history_id": row.get("predicted_history_id", ""),
                "support_observation_ids_json": support_ids,
                "support_observation_count": len(support_ids),
                "uses_gt_for_prediction": False,
                "uses_rgbd_pose_mesh_for_export": False,
                "uses_gt_for_diagnostic_labels": True,
            }
        )
        carrier_rows.append(
            {
                "scene": row.get("scene", ""),
                "carrier_id": row.get("component_id", ""),
                "carrier_id_source": "component_id_proxy",
                "material_node_id": row.get("material_node_id", ""),
                "state": row.get("state", ""),
                "predicted_history_id": row.get("predicted_history_id", ""),
                "supporting_observation_ids_json": support_ids,
                "support_observation_count": len(support_ids),
                "uses_gt_for_prediction": False,
                "uses_rgbd_pose_mesh_for_export": False,
                "uses_gt_for_diagnostic_labels": True,
            }
        )
    state_counts = _count_states(component_rows)
    carrier_state_counts = _count_states(carrier_rows)
    component_mapping_available = False
    gate = {
        "component_level_field_available": bool(component_rows),
        "uses_gt_for_prediction_false": True,
        "uses_rgbd_pose_mesh_for_export_false": True,
        "state_labels_complete": all(row["state"] in {"confirmed", "tentative", "shared", "quarantine", "unknown"} for row in component_rows),
    }
    gate["pass"] = bool(all(gate.values()))
    summary = {
        "phase": "v62_native_field",
        "created_at": utc_now(),
        "method_note": "v62 exports method-safe component-level state rows. Carrier-level rows are emitted only as component_id proxy because no separate component-to-carrier mapping exists in the v60/v61 CSV artifacts.",
        "component_level_field_available": bool(component_rows),
        "carrier_level_field_available": False,
        "component_to_carrier_mapping_available": component_mapping_available,
        "confirmed_component_count": state_counts.get("confirmed", 0),
        "native_observation_row_count": len({obs for row in component_rows for obs in row.get("support_observation_ids_json", [])}),
        "confirmed_carrier_count": carrier_state_counts.get("confirmed", 0),
        "tentative_carrier_count": carrier_state_counts.get("tentative", 0),
        "shared_carrier_count": carrier_state_counts.get("shared", 0),
        "quarantine_carrier_count": carrier_state_counts.get("quarantine", 0),
        "unknown_carrier_count": carrier_state_counts.get("unknown", 0),
        "state_labels_complete": gate["state_labels_complete"],
        "uses_gt_for_prediction": False,
        "uses_rgbd_pose_mesh_for_export": False,
        "uses_gt_for_diagnostic_labels": True,
        "ap_diagnostic_status": v61_native.get("ap_diagnostic_status", "not_run"),
        "native_field_limitation": "component-level material field, not dense carrier mesh/AP output",
        "gate": gate,
        "input_paths": {
            "solver_material_candidate_rows": _rel(cfg.material_candidate_rows_path),
            "solver_v56_core_summary": _rel(cfg.v56_core_summary_path),
            "solver_v56_tentative_summary": _rel(cfg.v56_tentative_summary_path),
            "solver_v61_embedding_summary": _rel(cfg.v61_embedding_summary_path),
            "v61_native_summary": _rel(cfg.v61_native_summary_path),
        },
    }
    return {"summary": summary, "native_component_state_rows": component_rows, "native_carrier_state_rows": carrier_rows}


def write_v62_native_field(result: dict[str, Any], output_root: str | Path) -> dict[str, str]:
    root = _project(output_root)
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "native_field_summary": root / "native_field_summary.json",
        "native_component_state_rows": root / "native_component_state_rows.csv",
        "native_carrier_state_rows": root / "native_carrier_state_rows.csv",
    }
    write_json(paths["native_field_summary"], result["summary"])
    write_csv(paths["native_component_state_rows"], result["native_component_state_rows"])
    write_csv(paths["native_carrier_state_rows"], result["native_carrier_state_rows"])
    return {name: _rel(path) for name, path in paths.items()}


def write_v62_native_visualizations(result: dict[str, Any], visualization_root: str | Path) -> dict[str, str]:
    root = _project(visualization_root)
    root.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        counts = _count_states(result["native_component_state_rows"])
        labels = ["confirmed", "tentative", "shared", "quarantine", "unknown"]
        path = root / "native_component_state_counts.png"
        fig, ax = plt.subplots(figsize=(8.0, 4.0))
        ax.bar(labels, [counts.get(label, 0) for label in labels], color="#52796F")
        ax.set_title("v62 native component states")
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        return {"native_component_state_counts": _rel(path), "visualization_status": "created"}
    except Exception as exc:  # pragma: no cover
        error_path = root / "v62_native_visualization_error.txt"
        error_path.write_text(f"visualization unavailable: {type(exc).__name__}: {exc}\n", encoding="utf-8")
        return {"visualization_error": _rel(error_path), "visualization_status": "unavailable"}


def _count_states(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        state = str(row.get("state", ""))
        counts[state] = counts.get(state, 0) + 1
    return counts


def _json_list(value: Any) -> list[Any]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if not isinstance(value, str):
        return [value]
    text = value.strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else [parsed]
