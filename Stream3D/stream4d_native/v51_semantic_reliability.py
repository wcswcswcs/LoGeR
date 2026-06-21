from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from stream4d_native.v47_common import ROOT, utc_now, write_csv, write_json


PLAN_PATH = "docs/stream4d_v51_r2_mosaic_remask_lift_codex_plan.md"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _rel(path: str | Path) -> str:
    path_obj = Path(path)
    if not path_obj.is_absolute():
        path_obj = ROOT / path_obj
    try:
        return str(path_obj.relative_to(ROOT))
    except ValueError:
        return str(path_obj)


def _component_set(row: dict[str, Any]) -> list[str]:
    try:
        value = json.loads(str(row.get("component_set") or "[]"))
    except json.JSONDecodeError:
        value = []
    return [str(item) for item in value]


def _cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom <= 0.0:
        return 1.0
    return 1.0 - float(np.dot(left, right) / denom)


def build_v51_semantic_reliability(
    keymask_root: str | Path,
    mask_observation_table: str | Path = "outputs/audit/v47_observation_tables_metricfix/mask_observation_table.csv",
    vote_rows_path: str | Path = "outputs/audit/v47_carrier_supertrack_union_32_fine_metricfix/carrier_supertrack_mask_vote_rows.csv",
    contradiction_threshold: float = 0.80,
) -> dict[str, Any]:
    root = ROOT / keymask_root if not Path(keymask_root).is_absolute() else Path(keymask_root)
    mask_table_path = ROOT / mask_observation_table if not Path(mask_observation_table).is_absolute() else Path(mask_observation_table)
    vote_path = ROOT / vote_rows_path if not Path(vote_rows_path).is_absolute() else Path(vote_rows_path)
    keymask_rows = _read_csv(root / "keymask_rows.csv")
    mask_features: dict[str, np.ndarray] = {}
    for row in _read_csv(mask_table_path):
        try:
            feature = np.asarray(json.loads(row.get("core_feature") or "[]"), dtype=np.float64)
        except json.JSONDecodeError:
            continue
        if feature.size:
            mask_features[str(row.get("mask_observation_id") or "")] = feature
    component_features: dict[str, list[np.ndarray]] = defaultdict(list)
    for row in _read_csv(vote_path):
        component = str(row.get("predicted_component_object_id") or "")
        if not component or component.startswith("uncovered:"):
            continue
        feature = mask_features.get(str(row.get("mask_observation_id") or ""))
        if feature is None:
            continue
        component_features[f"{row.get('scene')}|{component}"].append(feature)
    component_mean = {
        component: np.mean(features, axis=0)
        for component, features in component_features.items()
        if features
    }
    rows: list[dict[str, Any]] = []
    contradiction_values: list[float] = []
    total_components = 0
    matched_components = 0
    for row in keymask_rows:
        if row.get("selected_role") != "merge_keymask":
            continue
        components = _component_set(row)
        total_components += len(components)
        features = [component_mean[component] for component in components if component in component_mean]
        matched_components += len(features)
        max_distance = 0.0
        mean_distance = 0.0
        pair_count = 0
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                distance = _cosine_distance(features[i], features[j])
                max_distance = max(max_distance, distance)
                mean_distance += distance
                pair_count += 1
        if pair_count:
            mean_distance /= pair_count
        contradiction_values.append(max_distance)
        semantic_keep = max_distance <= float(contradiction_threshold)
        rows.append(
            {
                "proposal_id": row.get("proposal_id"),
                "scene": row.get("scene"),
                "frame_id": row.get("frame_id"),
                "component_set_size": len(components),
                "matched_component_feature_count": len(features),
                "semantic_contradiction": max_distance,
                "semantic_contradiction_mean_pairwise": mean_distance,
                "semantic_keep": semantic_keep,
                "semantic_backend": "colorhist_fallback",
                "uses_gt_for_prediction": False,
            }
        )
    contradiction_values_sorted = sorted(contradiction_values)
    p90 = contradiction_values_sorted[int(0.90 * (len(contradiction_values_sorted) - 1))] if contradiction_values_sorted else 0.0
    high_count = sum(1 for value in contradiction_values if value > float(contradiction_threshold))
    summary = {
        "semantic_backend": "colorhist_fallback",
        "keymask_count": len(rows),
        "feature_success_rate": matched_components / max(total_components, 1),
        "component_feature_success_rate": matched_components / max(total_components, 1),
        "semantic_contradiction_mean": sum(contradiction_values) / max(len(contradiction_values), 1),
        "semantic_contradiction_p90": p90,
        "semantic_contradiction_max": max(contradiction_values) if contradiction_values else 0.0,
        "contradiction_threshold": float(contradiction_threshold),
        "high_contradiction_keymask_count": high_count,
        "high_contradiction_keymask_rate": high_count / max(len(rows), 1),
        "semantic_keep_count": len(rows) - high_count,
        "uses_gt_for_prediction": False,
    }
    gate = {
        "feature_success_rate_pass": summary["feature_success_rate"] >= 0.95,
        "high_contradiction_rate_pass": summary["high_contradiction_keymask_rate"] <= 0.25,
        "uses_gt_for_prediction": False,
    }
    gate["pass"] = bool(gate["feature_success_rate_pass"] and gate["high_contradiction_rate_pass"])
    return {
        "phase": "v51_r2_semantic_reliability",
        "created_at": utc_now(),
        "plan": PLAN_PATH,
        "keymask_root": _rel(root),
        "mask_observation_table": _rel(mask_table_path),
        "vote_rows_path": _rel(vote_path),
        "summary": summary,
        "gate": gate,
        "semantic_rows": rows,
    }


def write_v51_semantic_reliability(output_root: str | Path, payload: dict[str, Any]) -> None:
    out = ROOT / output_root if not Path(output_root).is_absolute() else Path(output_root)
    write_json(out / "semantic_reliability_summary.json", {key: value for key, value in payload.items() if key != "semantic_rows"})
    write_csv(out / "semantic_reliability_rows.csv", payload["semantic_rows"])
