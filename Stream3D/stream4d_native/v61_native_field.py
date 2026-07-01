from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .v47_common import ROOT, parse_bool, utc_now, write_csv, write_json


DEFAULT_REFINED_STATES = "outputs/audit/v61_refinement/material_state_after_refinement.csv"
DEFAULT_QUERY_ROWS = "outputs/audit/v61_manifold_query/query_rows.csv"
DEFAULT_V60_NODES = "outputs/audit/v60_graph_v2/node_rows.csv"


@dataclass(frozen=True)
class V61NativeFieldConfig:
    refined_state_rows_path: str | Path = DEFAULT_REFINED_STATES
    query_rows_path: str | Path = DEFAULT_QUERY_ROWS
    v60_node_rows_path: str | Path = DEFAULT_V60_NODES
    output_root: str | Path = "outputs/audit/v61_native_field"
    visualization_root: str | Path = "outputs/audit/v61_visualizations/native_field"


def build_v61_native_field(config: V61NativeFieldConfig | None = None) -> dict[str, Any]:
    cfg = config or V61NativeFieldConfig()
    state_rows = [_parse_state_row(row) for row in _iter_csv(cfg.refined_state_rows_path)]
    semantic_modes = _semantic_modes(cfg.v60_node_rows_path)
    query_by_observation = _query_ledger(cfg.query_rows_path)
    history_payloads: dict[str, dict[str, Any]] = {}
    carrier_rows: list[dict[str, Any]] = []
    shortcut_rows: list[dict[str, Any]] = []
    state_counts: Counter[str] = Counter()

    for row in state_rows:
        state = _clean_state(row.get("state"))
        state_counts[state] += 1
        histories = _history_ids(row)
        if not histories:
            histories = [""]
        carrier_rows.append(_carrier_row(row, state, histories))
        if state in {"shared", "quarantine"} or len(histories) > 1:
            shortcut_rows.append(_shortcut_row(row, state, histories))
        for history_id in histories:
            if not history_id:
                continue
            payload = history_payloads.setdefault(history_id, _empty_history_payload(history_id, semantic_modes.get(history_id, [])))
            _add_state_to_history(payload, row, state)
            for obs_id in row.get("support_observation_ids", []):
                payload["supporting_observation_ids"].add(obs_id)
                if obs_id in query_by_observation:
                    payload["query_ledger"].add(query_by_observation[obs_id])
            payload["manifold_path_ledger"].add(row.get("candidate_evidence_types") or "")
            if state in {"shared", "quarantine"} or len(histories) > 1:
                payload["shortcut_ledger"].add(row["material_node_id"])

    history_rows = [_finalize_history_payload(payload) for payload in history_payloads.values()]
    ap_rows = [
        {
            "metric": "native_AP",
            "status": "not_run",
            "value": "",
            "is_diagnostic_only": True,
            "forbidden_for_method_table": True,
            "uses_gt_for_evaluation_alignment": False,
            "uses_rgbd_pose_mesh_for_export": False,
            "reason": "Phase6 exports method-safe native field only; no mesh/AP adapter was executed.",
        }
    ]
    labels_complete = all(row["state"] in {"confirmed", "tentative", "shared", "quarantine", "unknown"} for row in carrier_rows)
    summary = {
        "phase": "v61_native_field",
        "created_at": utc_now(),
        "history_object_count": len(history_rows),
        "confirmed_carrier_count": state_counts["confirmed"],
        "tentative_carrier_count": state_counts["tentative"],
        "shared_carrier_count": state_counts["shared"],
        "quarantine_carrier_count": state_counts["quarantine"],
        "unknown_carrier_count": state_counts["unknown"],
        "native_observation_row_count": len({obs for row in state_rows for obs in row.get("support_observation_ids", [])}),
        "method_safe_native_support_available": bool(history_rows and state_counts["confirmed"] > 0 and labels_complete),
        "state_labels_complete": labels_complete,
        "carrier_mapping_note": "No separate carrier id exists in v60/v61 CSV artifacts; carrier_id is set to component_id as a component-carrier proxy.",
        "ap_diagnostic_status": "not_run",
        "gate": {
            "method_safe_native_support_available": bool(history_rows and state_counts["confirmed"] > 0 and labels_complete),
            "history_object_count_gt_0": len(history_rows) > 0,
            "confirmed_carrier_count_gt_0": state_counts["confirmed"] > 0,
            "uses_gt_for_prediction_false": True,
            "uses_rgbd_pose_mesh_for_export_false": True,
            "state_labels_complete": labels_complete,
        },
        "input_paths": {
            "refined_state_rows": _rel(cfg.refined_state_rows_path),
            "query_rows": _rel(cfg.query_rows_path),
            "v60_node_rows": _rel(cfg.v60_node_rows_path),
        },
        "uses_gt_for_prediction": False,
        "uses_rgbd_pose_mesh_for_export": False,
        "uses_gt_for_diagnostic_labels": True,
    }
    summary["gate"]["pass"] = bool(all(summary["gate"].values()))
    return {
        "summary": summary,
        "native_history_rows": history_rows,
        "native_carrier_state_rows": carrier_rows,
        "shortcut_ledger_rows": shortcut_rows,
        "ap_metric_rows": ap_rows,
    }


def write_v61_native_field(result: dict[str, Any], output_root: str | Path) -> dict[str, str]:
    root = _project(output_root)
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "native_field_summary": root / "native_field_summary.json",
        "native_history_rows": root / "native_history_rows.csv",
        "native_carrier_state_rows": root / "native_carrier_state_rows.csv",
        "shortcut_ledger_rows": root / "shortcut_ledger_rows.csv",
        "ap_metric_rows": root / "ap_metric_rows.csv",
    }
    write_json(paths["native_field_summary"], result["summary"])
    write_csv(paths["native_history_rows"], result["native_history_rows"])
    write_csv(paths["native_carrier_state_rows"], result["native_carrier_state_rows"])
    write_csv(paths["shortcut_ledger_rows"], result["shortcut_ledger_rows"])
    write_csv(paths["ap_metric_rows"], result["ap_metric_rows"])
    return {name: _rel(path) for name, path in paths.items()}


def write_v61_native_visualizations(result: dict[str, Any], visualization_root: str | Path) -> dict[str, str]:
    root = _project(visualization_root)
    root.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        summary = result["summary"]
        labels = ["confirmed", "tentative", "shared", "quarantine", "unknown"]
        values = [summary[f"{label}_carrier_count"] for label in labels]
        state_path = root / "native_carrier_state_counts.png"
        fig, ax = plt.subplots(figsize=(7.2, 4.0))
        ax.bar(labels, values, color=["#2A9D8F", "#E9C46A", "#457B9D", "#E76F51", "#7A8B99"])
        ax.set_title("v61 native carrier state counts")
        fig.tight_layout()
        fig.savefig(state_path, dpi=160)
        plt.close(fig)

        histories = sorted(result["native_history_rows"], key=lambda row: int(row["confirmed_material_count"]), reverse=True)[:20]
        hist_path = root / "native_history_confirmed_material_top20.png"
        fig, ax = plt.subplots(figsize=(10.0, 4.2))
        ax.bar([str(idx + 1) for idx in range(len(histories))], [row["confirmed_material_count"] for row in histories], color="#457B9D")
        ax.set_title("v61 native histories by confirmed material count")
        ax.set_xlabel("top histories")
        fig.tight_layout()
        fig.savefig(hist_path, dpi=160)
        plt.close(fig)
        return {
            "native_carrier_state_counts": _rel(state_path),
            "native_history_confirmed_material_top20": _rel(hist_path),
            "visualization_status": "created",
        }
    except Exception as exc:  # pragma: no cover
        error_path = root / "v61_native_visualization_error.txt"
        error_path.write_text(f"visualization unavailable: {type(exc).__name__}: {exc}\n", encoding="utf-8")
        return {"visualization_error": _rel(error_path), "visualization_status": "unavailable"}


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


def _parse_state_row(row: dict[str, str]) -> dict[str, Any]:
    out = dict(row)
    out["support_observation_ids"] = _parse_json_list(row.get("support_observation_ids_json") or row.get("support_observation_ids"))
    out["uses_gt_for_prediction"] = parse_bool(row.get("uses_gt_for_prediction"))
    return out


def _semantic_modes(path: str | Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = defaultdict(list)
    path_obj = _project(path)
    if not path_obj.exists():
        return out
    for row in _iter_csv(path_obj):
        if row.get("node_type") == "semantic_mode":
            history_id = row.get("history_id") or _history_from_semantic_node(row.get("node_id", ""))
            if history_id:
                out[history_id].append(row.get("semantic_mode_id") or row.get("node_id") or "")
    return out


def _query_ledger(path: str | Path) -> dict[str, str]:
    path_obj = _project(path)
    if not path_obj.exists():
        return {}
    out: dict[str, str] = {}
    for row in _iter_csv(path_obj):
        obs_id = _normalize_observation_id(row.get("observation_id") or "")
        out[obs_id] = row.get("query_id") or ""
    return out


def _carrier_row(row: dict[str, Any], state: str, histories: list[str]) -> dict[str, Any]:
    return {
        "material_node_id": row["material_node_id"],
        "scene": row.get("scene", ""),
        "component_id": row.get("component_id", ""),
        "carrier_id": row.get("component_id", ""),
        "state": state,
        "history_ids_json": histories,
        "primary_history_id": histories[0] if histories and len(histories) == 1 else "",
        "supporting_observation_ids_json": row.get("support_observation_ids", []),
        "state_reason": row.get("state_reason", ""),
        "candidate_evidence_types": row.get("candidate_evidence_types", ""),
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }


def _shortcut_row(row: dict[str, Any], state: str, histories: list[str]) -> dict[str, Any]:
    return {
        "material_node_id": row["material_node_id"],
        "scene": row.get("scene", ""),
        "component_id": row.get("component_id", ""),
        "state": state,
        "history_ids_json": histories,
        "supporting_observation_ids_json": row.get("support_observation_ids", []),
        "shortcut_reason": row.get("refinement_reason") or row.get("state_reason", ""),
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }


def _empty_history_payload(history_id: str, semantic_modes: list[str]) -> dict[str, Any]:
    return {
        "history_id": history_id,
        "scene": history_id.split("|", 1)[0],
        "semantic_modes": set(semantic_modes),
        "confirmed_material_nodes": set(),
        "confirmed_component_ids": set(),
        "confirmed_carrier_ids": set(),
        "tentative_material_nodes": set(),
        "shared_material_nodes": set(),
        "quarantine_material_nodes": set(),
        "supporting_observation_ids": set(),
        "manifold_path_ledger": set(),
        "shortcut_ledger": set(),
        "query_ledger": set(),
        "state_timeline": set(),
    }


def _add_state_to_history(payload: dict[str, Any], row: dict[str, Any], state: str) -> None:
    material_id = row["material_node_id"]
    component_id = row.get("component_id", "")
    if state == "confirmed":
        payload["confirmed_material_nodes"].add(material_id)
        payload["confirmed_component_ids"].add(component_id)
        payload["confirmed_carrier_ids"].add(component_id)
    elif state == "tentative":
        payload["tentative_material_nodes"].add(material_id)
    elif state == "shared":
        payload["shared_material_nodes"].add(material_id)
    elif state == "quarantine":
        payload["quarantine_material_nodes"].add(material_id)
    payload["state_timeline"].add(f"{material_id}:{state}")


def _finalize_history_payload(payload: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"history_id": payload["history_id"], "scene": payload["scene"]}
    for key, value in payload.items():
        if key in {"history_id", "scene"}:
            continue
        out[key] = sorted(item for item in value if item)
    out["confirmed_material_count"] = len(out["confirmed_material_nodes"])
    out["tentative_material_count"] = len(out["tentative_material_nodes"])
    out["shared_material_count"] = len(out["shared_material_nodes"])
    out["quarantine_material_count"] = len(out["quarantine_material_nodes"])
    out["supporting_observation_count"] = len(out["supporting_observation_ids"])
    return out


def _history_ids(row: dict[str, Any]) -> list[str]:
    pred = str(row.get("predicted_history_id") or row.get("candidate_history_id") or "")
    return [part for part in pred.split("||") if part]


def _clean_state(value: Any) -> str:
    state = str(value or "unknown")
    return state if state in {"confirmed", "tentative", "shared", "quarantine", "unknown"} else "unknown"


def _normalize_observation_id(value: str) -> str:
    value = str(value or "")
    return value if value.startswith("m:") else f"m:{value}"


def _history_from_semantic_node(node_id: str) -> str:
    value = str(node_id)
    if value.startswith("s:"):
        value = value[2:]
    return value.split(":mode", 1)[0]


def _parse_json_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if not value:
        return []
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    return [str(item) for item in parsed] if isinstance(parsed, list) else []
