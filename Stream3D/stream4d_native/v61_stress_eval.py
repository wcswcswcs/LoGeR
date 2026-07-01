from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .v47_common import (
    ROOT,
    adjusted_rand_score,
    cluster_completeness,
    cluster_purity,
    parse_bool,
    parse_float,
    read_json,
    utc_now,
    write_csv,
    write_json,
)


DEFAULT_GLOBAL_STATES = "outputs/audit/v61_global_embedding/material_state_rows.csv"
DEFAULT_REFINED_STATES = "outputs/audit/v61_refinement/material_state_after_refinement.csv"
DEFAULT_QUERY_SUMMARY = "outputs/audit/v61_manifold_query/query_summary.json"
DEFAULT_V56_STRESS = "outputs/audit/v56_stress_proxy/stress_proxy_metric_rows.csv"


@dataclass(frozen=True)
class V61StressEvalConfig:
    global_state_rows_path: str | Path = DEFAULT_GLOBAL_STATES
    refined_state_rows_path: str | Path = DEFAULT_REFINED_STATES
    query_summary_path: str | Path = DEFAULT_QUERY_SUMMARY
    v56_stress_rows_path: str | Path = DEFAULT_V56_STRESS
    output_root: str | Path = "outputs/audit/v61_stress"
    visualization_root: str | Path = "outputs/audit/v61_visualizations/stress"


def build_v61_stress_eval(config: V61StressEvalConfig | None = None) -> dict[str, Any]:
    cfg = config or V61StressEvalConfig()
    global_rows = [_parse_state_row(row) for row in _iter_csv(cfg.global_state_rows_path)]
    refined_rows = [_parse_state_row(row) for row in _iter_csv(cfg.refined_state_rows_path)]
    query_summary = read_json(_project(cfg.query_summary_path)) if _project(cfg.query_summary_path).exists() else {}
    v56_by_setting = _v56_stress_lookup(cfg.v56_stress_rows_path)
    settings = _stress_settings()

    metric_rows: list[dict[str, Any]] = []
    setting_summary_rows: list[dict[str, Any]] = []
    for setting in settings:
        survivors_by_material = {row["material_node_id"]: _surviving_supports(row, setting) for row in refined_rows}
        true_rows = [row for row in refined_rows if row.get("diagnostic_expected_history_id")]
        true_labels = [row["diagnostic_expected_history_id"] for row in true_rows]
        methods = {
            "D0_mask_only_memory": _mask_only_labels(true_rows, survivors_by_material, setting),
            "D4_v61_global_embedding": _state_labels(_rows_by_id(global_rows), true_rows, include_query=False),
            "D5_v61_refined_manifold": _state_labels(_rows_by_id(refined_rows), true_rows, include_query=False),
            "D6_v61_full_plus_query": _state_labels(_rows_by_id(refined_rows), true_rows, include_query=bool((query_summary.get("gate") or {}).get("pass"))),
            "D7_shuffled_D4RT_control": _shuffled_labels(_state_labels(_rows_by_id(refined_rows), true_rows, include_query=False), true_rows),
            "D8_no_temporal_control": [f"material:{row['material_node_id']}" for row in true_rows],
            "D9_semantic_only_control": _semantic_only_labels(true_rows),
        }
        setting_metrics: dict[str, dict[str, Any]] = {}
        for method_id, pred_labels in methods.items():
            metrics = _metric_row(setting, method_id, true_rows, true_labels, pred_labels, survivors_by_material)
            metric_rows.append(metrics)
            setting_metrics[method_id] = metrics
        v56 = v56_by_setting.get((setting["stress_type"], setting["stress_strength"]))
        summary_row = _setting_summary(setting, setting_metrics, v56)
        setting_summary_rows.append(summary_row)
        metric_rows.extend(_v56_rows(setting, v56))

    pass_mask_only = sum(1 for row in setting_summary_rows if row["v61_refined_minus_mask_only_ARI"] >= 0.05)
    pass_v56 = sum(
        1
        for row in setting_summary_rows
        if row.get("v61_refined_minus_v56_expanded_ARI") is not None
        and row["v61_refined_minus_v56_expanded_ARI"] >= 0.02
    )
    id_switch_ok = _mean_float(row["v61_id_switch_delta_vs_mask_only"] for row in setting_summary_rows) <= -0.05
    reactivation_precision = _mean_float(row["v61_reactivation_precision_diagnostic"] for row in setting_summary_rows)
    same_category_delta = _mean_float(row["v61_same_category_merge_delta_vs_mask_only"] for row in setting_summary_rows)
    gate = {
        "stress_real_minus_mask_only_ARI_ge_0_05_in_at_least_3_settings": pass_mask_only >= 3,
        "stress_real_minus_v56_expanded_ARI_ge_0_02_in_at_least_3_settings": pass_v56 >= 3,
        "id_switch_rate_le_mask_only_minus_0_05": id_switch_ok,
        "reactivation_precision_ge_0_80": reactivation_precision is not None and reactivation_precision >= 0.80,
        "same_category_merge_rate_le_mask_only_minus_0_05": same_category_delta is not None and same_category_delta <= -0.05,
    }
    gate["pass"] = bool(all(gate.values()))
    summary = {
        "phase": "v61_stress",
        "created_at": utc_now(),
        "status": "diagnostic_proxy_only",
        "method_note": (
            "Phase5 simulates stress over v61 material ownership rows and support observation ids. "
            "Mask-only labels are surviving mask observations; v61 memory labels keep material ownership under dropout. "
            "This is not a native AP or a rerun of the front end."
        ),
        "stress_setting_count": len(settings),
        "stress_real_minus_mask_only_ARI_pass_count": pass_mask_only,
        "stress_real_minus_v56_expanded_ARI_pass_count": pass_v56,
        "reactivation_precision_diagnostic": reactivation_precision,
        "mean_v61_id_switch_delta_vs_mask_only": _mean_float(row["v61_id_switch_delta_vs_mask_only"] for row in setting_summary_rows),
        "mean_v61_same_category_merge_delta_vs_mask_only": same_category_delta,
        "query_gate_pass_for_D6": bool((query_summary.get("gate") or {}).get("pass")),
        "gate": gate,
        "setting_summary_rows": setting_summary_rows,
        "input_paths": {
            "global_state_rows": _rel(cfg.global_state_rows_path),
            "refined_state_rows": _rel(cfg.refined_state_rows_path),
            "query_summary": _rel(cfg.query_summary_path),
            "v56_stress_rows": _rel(cfg.v56_stress_rows_path),
        },
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }
    return {"summary": summary, "stress_metric_rows": metric_rows, "stress_setting_rows": setting_summary_rows}


def write_v61_stress_eval(result: dict[str, Any], output_root: str | Path) -> dict[str, str]:
    root = _project(output_root)
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "stress_summary": root / "stress_summary.json",
        "stress_metric_rows": root / "stress_metric_rows.csv",
        "stress_setting_rows": root / "stress_setting_rows.csv",
    }
    write_json(paths["stress_summary"], result["summary"])
    write_csv(paths["stress_metric_rows"], result["stress_metric_rows"])
    write_csv(paths["stress_setting_rows"], result["stress_setting_rows"])
    return {name: _rel(path) for name, path in paths.items()}


def write_v61_stress_visualizations(result: dict[str, Any], visualization_root: str | Path) -> dict[str, str]:
    root = _project(visualization_root)
    root.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rows = result["stress_setting_rows"]
        labels = [f"{row['stress_type']}:{row['stress_strength']}" for row in rows]
        gains = [row["v61_refined_minus_mask_only_ARI"] for row in rows]
        mask_path = root / "stress_real_minus_mask_only_ari.png"
        fig, ax = plt.subplots(figsize=(10.0, 4.2))
        ax.bar(labels, gains, color="#2A9D8F")
        ax.axhline(0.05, color="#333333", linestyle="--", linewidth=1.0)
        ax.set_title("v61 stress real-minus-mask-only ARI")
        ax.tick_params(axis="x", labelrotation=35)
        fig.tight_layout()
        fig.savefig(mask_path, dpi=160)
        plt.close(fig)

        v56_gains = [row.get("v61_refined_minus_v56_expanded_ARI") or 0.0 for row in rows]
        v56_path = root / "stress_real_minus_v56_expanded_ari.png"
        fig, ax = plt.subplots(figsize=(10.0, 4.2))
        ax.bar(labels, v56_gains, color="#E76F51")
        ax.axhline(0.02, color="#333333", linestyle="--", linewidth=1.0)
        ax.set_title("v61 stress real-minus-v56-expanded ARI")
        ax.tick_params(axis="x", labelrotation=35)
        fig.tight_layout()
        fig.savefig(v56_path, dpi=160)
        plt.close(fig)

        return {
            "stress_real_minus_mask_only_plot": _rel(mask_path),
            "stress_real_minus_v56_expanded_plot": _rel(v56_path),
            "visualization_status": "created",
        }
    except Exception as exc:  # pragma: no cover
        error_path = root / "v61_stress_visualization_error.txt"
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
    out["has_K_sem"] = parse_bool(row.get("has_K_sem"))
    out["has_K_mat"] = parse_bool(row.get("has_K_mat"))
    return out


def _rows_by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["material_node_id"]: row for row in rows}


def _stress_settings() -> list[dict[str, Any]]:
    return [
        {"stress_type": "mask_dropout", "stress_strength": "0.50", "drop_probability": 0.50},
        {"stress_type": "mask_dropout", "stress_strength": "0.70", "drop_probability": 0.70},
        {"stress_type": "temporal_gap", "stress_strength": "drop_bridge_update", "drop_frame_mod": 3},
        {"stress_type": "bridge_dropout", "stress_strength": "drop_bridge", "drop_frame_mod": 4},
        {"stress_type": "update_dropout", "stress_strength": "drop_update", "drop_frame_mod": 5},
        {"stress_type": "mask_split_proxy", "stress_strength": "drop_one_of_four_masks", "drop_bucket_mod": 4},
        {"stress_type": "merge_underseg", "stress_strength": "merge_same_frame_buckets", "merge_bucket_mod": 5},
        {"stress_type": "same_category_confusion", "stress_strength": "scene_bucket_merge", "merge_bucket_mod": 7},
    ]


def _surviving_supports(row: dict[str, Any], setting: dict[str, Any]) -> list[str]:
    survivors: list[str] = []
    for obs_id in row.get("support_observation_ids", []):
        if _drops_observation(obs_id, setting):
            continue
        survivors.append(obs_id)
    return survivors


def _drops_observation(obs_id: str, setting: dict[str, Any]) -> bool:
    stress_type = setting["stress_type"]
    if stress_type == "mask_dropout":
        return _stable_unit(obs_id) < float(setting["drop_probability"])
    if "drop_frame_mod" in setting:
        parsed = _parse_observation_id(obs_id)
        return parsed["frame_id"] % int(setting["drop_frame_mod"]) == 0
    if "drop_bucket_mod" in setting:
        return _stable_bucket(obs_id, int(setting["drop_bucket_mod"])) == 0
    return False


def _mask_only_labels(rows: list[dict[str, Any]], survivors_by_material: dict[str, list[str]], setting: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    for row in rows:
        survivors = survivors_by_material.get(row["material_node_id"], [])
        if not survivors:
            labels.append(f"material:{row['material_node_id']}")
            continue
        obs_id = sorted(survivors)[0]
        if setting["stress_type"] == "merge_underseg":
            parsed = _parse_observation_id(obs_id)
            labels.append(f"merge:{parsed['scene']}:{parsed['frame_id']}:{_stable_bucket(obs_id, int(setting['merge_bucket_mod']))}")
        elif setting["stress_type"] == "same_category_confusion":
            parsed = _parse_observation_id(obs_id)
            labels.append(f"samecat:{parsed['scene']}:{_stable_bucket(row['diagnostic_expected_history_id'], int(setting['merge_bucket_mod']))}")
        else:
            labels.append(f"mask:{obs_id}")
    return labels


def _state_labels(source_by_material: dict[str, dict[str, Any]], rows: list[dict[str, Any]], *, include_query: bool) -> list[str]:
    labels: list[str] = []
    for row in rows:
        state_row = source_by_material.get(row["material_node_id"], row)
        state = state_row.get("state")
        pred = str(state_row.get("predicted_history_id") or "")
        if include_query and state == "quarantine":
            labels.append(f"query_defer:{row['material_node_id']}")
        elif state in {"confirmed", "tentative"} and pred and "||" not in pred:
            labels.append(pred)
        else:
            labels.append(f"material:{row['material_node_id']}")
    return labels


def _semantic_only_labels(rows: list[dict[str, Any]]) -> list[str]:
    labels: list[str] = []
    for row in rows:
        pred = str(row.get("candidate_history_id") or row.get("predicted_history_id") or "")
        if row.get("has_K_sem") and pred and "||" not in pred:
            labels.append(pred)
        else:
            labels.append(f"material:{row['material_node_id']}")
    return labels


def _shuffled_labels(labels: list[str], rows: list[dict[str, Any]]) -> list[str]:
    out = list(labels)
    by_scene: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        by_scene[str(row.get("scene") or "")].append(idx)
    for indices in by_scene.values():
        if len(indices) <= 1:
            continue
        rotated = [out[idx] for idx in indices[1:]] + [out[indices[0]]]
        for idx, label in zip(indices, rotated):
            out[idx] = label
    return out


def _metric_row(
    setting: dict[str, Any],
    method_id: str,
    rows: list[dict[str, Any]],
    true_labels: list[str],
    pred_labels: list[str],
    survivors_by_material: dict[str, list[str]],
) -> dict[str, Any]:
    ari = adjusted_rand_score(true_labels, pred_labels)
    purity = cluster_purity(true_labels, pred_labels)
    completeness = cluster_completeness(true_labels, pred_labels)
    return {
        "stress_type": setting["stress_type"],
        "stress_strength": setting["stress_strength"],
        "row": method_id,
        "ARI": ari,
        "purity": purity,
        "completeness": completeness,
        "temporal_span_mean": _temporal_span_mean(rows, survivors_by_material),
        "id_switch_rate_diagnostic": _id_switch_rate(true_labels, pred_labels),
        "reactivation_precision_diagnostic": _reactivation_precision(rows, pred_labels, survivors_by_material),
        "same_category_merge_rate": _merge_rate(true_labels, pred_labels),
        "underseg_false_merge_rate": _merge_rate(true_labels, pred_labels),
        "false_update_rate": _false_update_rate(true_labels, pred_labels),
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }


def _setting_summary(setting: dict[str, Any], metrics: dict[str, dict[str, Any]], v56: dict[str, Any] | None) -> dict[str, Any]:
    mask = metrics["D0_mask_only_memory"]
    refined = metrics["D5_v61_refined_manifold"]
    row = {
        "stress_type": setting["stress_type"],
        "stress_strength": setting["stress_strength"],
        "mask_only_ARI": mask["ARI"],
        "v61_refined_ARI": refined["ARI"],
        "v61_refined_purity": refined["purity"],
        "v61_refined_completeness": refined["completeness"],
        "v61_refined_minus_mask_only_ARI": refined["ARI"] - mask["ARI"],
        "v61_id_switch_delta_vs_mask_only": refined["id_switch_rate_diagnostic"] - mask["id_switch_rate_diagnostic"],
        "v61_reactivation_precision_diagnostic": refined["reactivation_precision_diagnostic"],
        "v61_same_category_merge_delta_vs_mask_only": refined["same_category_merge_rate"] - mask["same_category_merge_rate"],
        "v56_expanded_ARI": None,
        "v61_refined_minus_v56_expanded_ARI": None,
    }
    if v56:
        row["v56_expanded_ARI"] = v56.get("expanded_ARI")
        row["v61_refined_minus_v56_expanded_ARI"] = refined["ARI"] - float(v56.get("expanded_ARI", 0.0))
    return row


def _v56_rows(setting: dict[str, Any], v56: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not v56:
        return []
    return [
        {
            "stress_type": setting["stress_type"],
            "stress_strength": setting["stress_strength"],
            "row": "D1_v56_confirmed_core",
            "ARI": v56.get("core_ARI"),
            "purity": v56.get("core_purity"),
            "completeness": v56.get("core_completeness"),
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic_labels": True,
        },
        {
            "stress_type": setting["stress_type"],
            "stress_strength": setting["stress_strength"],
            "row": "D2_v56_expanded_tentative",
            "ARI": v56.get("expanded_ARI"),
            "purity": v56.get("expanded_purity"),
            "completeness": v56.get("expanded_completeness"),
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic_labels": True,
        },
    ]


def _v56_stress_lookup(path: str | Path) -> dict[tuple[str, str], dict[str, Any]]:
    path_obj = _project(path)
    if not path_obj.exists():
        return {}
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for row in _iter_csv(path_obj):
        out[(row["stress_type"], row["stress_strength"])] = {
            "core_ARI": parse_float(row.get("core_ARI")),
            "core_purity": parse_float(row.get("core_purity")),
            "core_completeness": parse_float(row.get("core_completeness")),
            "expanded_ARI": parse_float(row.get("expanded_ARI")),
            "expanded_purity": parse_float(row.get("expanded_purity")),
            "expanded_completeness": parse_float(row.get("expanded_completeness")),
        }
    return out


def _temporal_span_mean(rows: list[dict[str, Any]], survivors_by_material: dict[str, list[str]]) -> float:
    spans: list[float] = []
    for row in rows:
        frames = sorted({_parse_observation_id(obs_id)["frame_id"] for obs_id in survivors_by_material.get(row["material_node_id"], [])})
        if frames:
            spans.append(float(frames[-1] - frames[0] + 1))
    return _mean_float(spans) or 0.0


def _id_switch_rate(true_labels: list[str], pred_labels: list[str]) -> float:
    by_true: dict[str, Counter[str]] = defaultdict(Counter)
    for true, pred in zip(true_labels, pred_labels):
        by_true[true][pred] += 1
    switches = 0.0
    total = 0.0
    for counts in by_true.values():
        count = sum(counts.values())
        total += count
        switches += count - max(counts.values())
    return 0.0 if total == 0.0 else switches / total


def _reactivation_precision(rows: list[dict[str, Any]], pred_labels: list[str], survivors_by_material: dict[str, list[str]]) -> float:
    dormant = [
        (row, pred)
        for row, pred in zip(rows, pred_labels)
        if not survivors_by_material.get(row["material_node_id"]) and not pred.startswith("material:")
    ]
    if not dormant:
        return 1.0
    correct = sum(1 for row, pred in dormant if pred == row["diagnostic_expected_history_id"])
    return correct / float(len(dormant))


def _merge_rate(true_labels: list[str], pred_labels: list[str]) -> float:
    by_pred: dict[str, set[str]] = defaultdict(set)
    for true, pred in zip(true_labels, pred_labels):
        if not pred.startswith("material:"):
            by_pred[pred].add(true)
    if not by_pred:
        return 0.0
    merged = sum(1 for labels in by_pred.values() if len(labels) > 1)
    return merged / float(len(by_pred))


def _false_update_rate(true_labels: list[str], pred_labels: list[str]) -> float:
    assigned = [(true, pred) for true, pred in zip(true_labels, pred_labels) if not pred.startswith("material:") and not pred.startswith("mask:")]
    if not assigned:
        return 0.0
    wrong = sum(1 for true, pred in assigned if pred != true)
    return wrong / float(len(assigned))


def _parse_observation_id(obs_id: str) -> dict[str, Any]:
    value = str(obs_id)
    if value.startswith("m:"):
        value = value[2:]
    parts = value.split(":")
    return {
        "scene": parts[0] if parts else "",
        "frame_id": int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0,
        "mask_id": parts[2] if len(parts) > 2 else "",
    }


def _stable_unit(value: str) -> float:
    return _stable_bucket(value, 1000000) / 1000000.0


def _stable_bucket(value: str, modulo: int) -> int:
    digest = hashlib.sha1(str(value).encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % int(modulo)


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


def _mean_float(values: Iterable[Any]) -> float | None:
    nums = [float(value) for value in values if value is not None]
    return None if not nums else sum(nums) / float(len(nums))
