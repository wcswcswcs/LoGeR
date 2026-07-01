from __future__ import annotations

import csv
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from .v47_common import ROOT, parse_int, read_csv, utc_now, write_csv, write_json
from .v55_history_update import _build_assignment_map, _metrics_from_support
from .v56_full_eval import SUPPORT_ROWS, SUPPORT_VARIANT


CORE_COMPONENT_ROWS = "outputs/audit/v56_reuse_core_update_C3_e1_boundary_uv/history_component_rows.csv"
SELECTED_COMPONENT_ROWS = (
    "outputs/audit/v55_history_update_repair_native_boundary_uv_historymask_s100_r09_d15_plus_cosupport_seed038_componentrows_probe/"
    "history_component_rows.csv"
)
CHUNK_ROWS = "outputs/audit/v54_chunk_universe_stride1_probe5_q4096_notopup/chunk_rows.csv"
CHUNK_ROLE_ROWS = "outputs/audit/v55_chunk_roles/chunk_role_rows.csv"


def _project(path: str | Path) -> Path:
    path_obj = Path(path)
    return path_obj if path_obj.is_absolute() else ROOT / path_obj


def _rel(path: str | Path) -> str:
    path_obj = _project(path)
    try:
        return str(path_obj.relative_to(ROOT))
    except ValueError:
        return str(path_obj)


def _stable_bucket(value: str, modulo: int = 1000) -> int:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % modulo


def _history_components(path: Path) -> dict[str, dict[str, Any]]:
    histories: dict[str, dict[str, Any]] = {}
    for row in read_csv(path):
        history_id = str(row.get("history_id"))
        history = histories.setdefault(
            history_id,
            {"history_id": history_id, "scene": str(row.get("scene")), "components": set()},
        )
        history["components"].add(str(row.get("component_id")))
    return histories


def _component_history_keys(histories: dict[str, dict[str, Any]]) -> set[tuple[str, str, str]]:
    return {
        (history_id, str(history["scene"]), str(component_id))
        for history_id, history in histories.items()
        for component_id in history["components"]
    }


def _multi_history_component_rows(
    component_rows: set[tuple[str, str, str]],
) -> set[tuple[str, str, str]]:
    histories_by_component: dict[tuple[str, str], set[str]] = defaultdict(set)
    for history_id, scene, component_id in component_rows:
        histories_by_component[(scene, component_id)].add(history_id)
    shared_components = {
        component_key for component_key, history_ids in histories_by_component.items() if len(history_ids) > 1
    }
    return {row for row in component_rows if (row[1], row[2]) in shared_components}


def _chunk_lookup(chunk_rows: list[dict[str, Any]], role_rows: list[dict[str, Any]]) -> dict[tuple[str, int], tuple[str, str]]:
    role_by_chunk = {str(row.get("chunk_id")): str(row.get("role")) for row in role_rows}
    lookup: dict[tuple[str, int], tuple[str, str]] = {}
    for row in chunk_rows:
        chunk_id = str(row.get("chunk_id"))
        role = role_by_chunk.get(chunk_id, "")
        scene = str(row.get("scene"))
        for frame_id in range(parse_int(row.get("raw_frame_start")), parse_int(row.get("raw_frame_end")) + 1):
            lookup[(scene, frame_id)] = (chunk_id, role)
    return lookup


def _support_masks_by_component(
    support_rows_path: Path,
    *,
    support_variant: str,
    scenes: set[str],
    chunk_by_scene_frame: dict[tuple[str, int], tuple[str, str]],
) -> dict[tuple[str, str], set[tuple[str, str, str, str]]]:
    masks: dict[tuple[str, str], set[tuple[str, str, str, str]]] = defaultdict(set)
    with support_rows_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("variant")) != support_variant:
                continue
            scene = str(row.get("scene"))
            if scene not in scenes:
                continue
            frame_id = parse_int(row.get("frame_id"))
            chunk_id, role = chunk_by_scene_frame.get((scene, frame_id), ("", ""))
            masks[(scene, str(row.get("component_id")))].add(
                (scene, chunk_id, role, str(row.get("mask_observation_id")))
            )
    return masks


def _filter_histories(
    histories: dict[str, dict[str, Any]],
    keep_component: Callable[[str, str], bool],
) -> dict[str, dict[str, Any]]:
    filtered: dict[str, dict[str, Any]] = {}
    for history_id, history in histories.items():
        scene = str(history["scene"])
        components = {component_id for component_id in history["components"] if keep_component(scene, component_id)}
        if components:
            filtered[history_id] = {"history_id": history_id, "scene": scene, "components": components}
    return filtered


def _with_components(
    base_histories: dict[str, dict[str, Any]],
    component_rows: set[tuple[str, str, str]],
    keep_component: Callable[[str, str], bool],
) -> dict[str, dict[str, Any]]:
    out = {
        history_id: {
            "history_id": history_id,
            "scene": str(history["scene"]),
            "components": set(history["components"]),
        }
        for history_id, history in base_histories.items()
    }
    for history_id, scene, component_id in component_rows:
        if not keep_component(scene, component_id):
            continue
        history = out.setdefault(history_id, {"history_id": history_id, "scene": scene, "components": set()})
        history["components"].add(component_id)
    return out


def _metrics(
    histories: dict[str, dict[str, Any]],
    *,
    support_rows_path: Path,
    support_variant: str,
    scenes: set[str],
) -> tuple[dict[str, float], int]:
    assignment, duplicate_count = _build_assignment_map(histories, "components")
    metrics = _metrics_from_support(
        support_rows_path,
        support_variant=support_variant,
        scenes=scenes,
        component_to_history=assignment,
    )
    return metrics, duplicate_count


def build_v56_stress_proxy(
    *,
    core_component_rows_path: str | Path = CORE_COMPONENT_ROWS,
    selected_component_rows_path: str | Path = SELECTED_COMPONENT_ROWS,
    support_rows_path: str | Path = SUPPORT_ROWS,
    support_variant: str = SUPPORT_VARIANT,
    chunk_rows_path: str | Path = CHUNK_ROWS,
    chunk_role_rows_path: str | Path = CHUNK_ROLE_ROWS,
) -> dict[str, Any]:
    support_path = _project(support_rows_path)
    core_histories = _history_components(_project(core_component_rows_path))
    selected_histories = _history_components(_project(selected_component_rows_path))
    scenes = {str(history["scene"]) for history in selected_histories.values()} | {
        str(history["scene"]) for history in core_histories.values()
    }
    selected_rows = _component_history_keys(selected_histories)
    core_rows = _component_history_keys(core_histories)
    raw_tentative_rows = selected_rows - core_rows
    quarantine_rows = _multi_history_component_rows(selected_rows) & raw_tentative_rows
    tentative_rows = raw_tentative_rows - quarantine_rows

    chunk_by_scene_frame = _chunk_lookup(read_csv(_project(chunk_rows_path)), read_csv(_project(chunk_role_rows_path)))
    masks_by_component = _support_masks_by_component(
        support_path,
        support_variant=support_variant,
        scenes=scenes,
        chunk_by_scene_frame=chunk_by_scene_frame,
    )

    def has_surviving_mask(scene: str, component_id: str, setting: dict[str, Any]) -> bool:
        masks = masks_by_component.get((scene, component_id), set())
        if not masks:
            return False
        stress_type = str(setting["stress_type"])
        strength = float(setting.get("strength", 0.0))
        drop_roles = set(setting.get("drop_roles", []))
        for _scene, _chunk_id, role, mask_observation_id in masks:
            if role in drop_roles:
                continue
            if stress_type == "mask_dropout" and _stable_bucket(mask_observation_id) < int(strength * 1000):
                continue
            if stress_type == "mask_split_proxy" and _stable_bucket(mask_observation_id, 4) == 0:
                continue
            return True
        return False

    settings = [
        {"stress_type": "mask_dropout", "stress_strength": "0.50", "strength": 0.50, "drop_roles": []},
        {"stress_type": "mask_dropout", "stress_strength": "0.70", "strength": 0.70, "drop_roles": []},
        {"stress_type": "temporal_gap", "stress_strength": "drop_bridge_update", "strength": 0.0, "drop_roles": ["bridge", "update"]},
        {"stress_type": "bridge_dropout", "stress_strength": "drop_bridge", "strength": 0.0, "drop_roles": ["bridge"]},
        {"stress_type": "update_dropout", "stress_strength": "drop_update", "strength": 0.0, "drop_roles": ["update"]},
        {"stress_type": "mask_split_proxy", "stress_strength": "drop_one_of_four_masks", "strength": 0.25, "drop_roles": []},
    ]

    metric_rows: list[dict[str, Any]] = []
    pass_count = 0
    best_gain = None
    for setting in settings:
        keep = lambda scene, component_id, setting=setting: has_surviving_mask(scene, component_id, setting)
        mask_only_histories = _filter_histories(selected_histories, keep)
        core_memory_histories = core_histories
        expanded_memory_histories = _with_components(core_histories, tentative_rows, keep)
        mask_metrics, mask_duplicates = _metrics(
            mask_only_histories, support_rows_path=support_path, support_variant=support_variant, scenes=scenes
        )
        core_metrics, core_duplicates = _metrics(
            core_memory_histories, support_rows_path=support_path, support_variant=support_variant, scenes=scenes
        )
        expanded_metrics, expanded_duplicates = _metrics(
            expanded_memory_histories, support_rows_path=support_path, support_variant=support_variant, scenes=scenes
        )
        core_gain = float(core_metrics["ARI"] - mask_metrics["ARI"])
        expanded_gain = float(expanded_metrics["ARI"] - mask_metrics["ARI"])
        best_setting_gain = max(core_gain, expanded_gain)
        best_gain = best_setting_gain if best_gain is None else max(best_gain, best_setting_gain)
        if expanded_gain >= 0.05:
            pass_count += 1
        metric_rows.append(
            {
                "stress_type": setting["stress_type"],
                "stress_strength": setting["stress_strength"],
                "row": "stress_proxy_mask_only_vs_v56_memory",
                "mask_only_ARI": mask_metrics["ARI"],
                "mask_only_purity": mask_metrics["purity"],
                "mask_only_completeness": mask_metrics["completeness"],
                "core_ARI": core_metrics["ARI"],
                "core_purity": core_metrics["purity"],
                "core_completeness": core_metrics["completeness"],
                "expanded_ARI": expanded_metrics["ARI"],
                "expanded_purity": expanded_metrics["purity"],
                "expanded_completeness": expanded_metrics["completeness"],
                "core_real_minus_mask_only_ARI": core_gain,
                "expanded_real_minus_mask_only_ARI": expanded_gain,
                "mask_only_duplicate_component_count": mask_duplicates,
                "core_duplicate_component_count": core_duplicates,
                "expanded_duplicate_component_count": expanded_duplicates,
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic_labels": True,
            }
        )

    summary = {
        "phase": "v56_stress_proxy",
        "created_at": utc_now(),
        "status": "diagnostic_proxy_only",
        "input_paths": {
            "core_component_rows_path": _rel(core_component_rows_path),
            "selected_component_rows_path": _rel(selected_component_rows_path),
            "support_rows_path": _rel(support_rows_path),
            "chunk_rows_path": _rel(chunk_rows_path),
            "chunk_role_rows_path": _rel(chunk_role_rows_path),
        },
        "stress_setting_count": len(settings),
        "stress_real_minus_mask_only_ARI_pass_count": pass_count,
        "best_real_minus_mask_only_ARI_proxy": best_gain,
        "reactivation_precision_diagnostic": None,
        "false_promotion_rate": None,
        "core_purity": metric_rows[0]["core_purity"] if metric_rows else None,
        "gate": {
            "stress_real_minus_mask_only_ARI_ge_0.05_in_at_least_3_settings": pass_count >= 3,
            "stress_temporal_span_gain_vs_mask_only_ge_0.30": False,
            "reactivation_precision_diagnostic_ge_0.80": False,
            "false_promotion_rate_le_0.15": False,
            "core_purity_ge_0.88": bool(metric_rows and metric_rows[0]["core_purity"] >= 0.88),
        },
        "diagnostic_status": "proxy does not satisfy full Phase 8 dynamic-ready gate",
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }
    summary["gate"]["pass"] = bool(all(summary["gate"].values()))
    return {"summary": summary, "stress_metric_rows": metric_rows}


def write_v56_stress_proxy(output_root: str | Path, payload: dict[str, Any]) -> None:
    out = _project(output_root)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "stress_proxy_summary.json", payload["summary"])
    write_csv(out / "stress_proxy_metric_rows.csv", payload["stress_metric_rows"])

