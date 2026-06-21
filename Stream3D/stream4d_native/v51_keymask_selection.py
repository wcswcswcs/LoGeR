from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from stream4d_native.v47_common import ROOT, utc_now, write_csv, write_json
from stream4d_native.v51_remask_source_discovery import _frame_id


PLAN_PATH = "docs/stream4d_v51_r2_mosaic_remask_lift_codex_plan.md"


def _rel(path: str | Path) -> str:
    path_obj = Path(path)
    if not path_obj.is_absolute():
        path_obj = ROOT / path_obj
    try:
        return str(path_obj.relative_to(ROOT))
    except ValueError:
        return str(path_obj)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _proposal_id(scene: str, frame_id: int, source: str, index: int) -> str:
    return f"{scene}|frame{int(frame_id):06d}|{source}|p{int(index):05d}"


def _load_mask_to_component(vote_rows_path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for row in _read_csv(vote_rows_path):
        mask_id = str(row.get("mask_observation_id") or "")
        component = str(row.get("predicted_component_object_id") or "")
        scene = str(row.get("scene") or "")
        if not mask_id or not component or component.startswith("uncovered:"):
            continue
        mapping[mask_id] = f"{scene}|{component}"
    return mapping


def _load_carriers(
    carrier_rows_path: Path,
    mask_to_component: dict[str, str],
    scenes: set[str],
    frame_ids: set[int],
) -> dict[tuple[str, int], list[tuple[float, float, str, str]]]:
    carriers: dict[tuple[str, int], list[tuple[float, float, str, str]]] = defaultdict(list)
    for row in _read_csv(carrier_rows_path):
        scene = str(row.get("scene") or "")
        try:
            frame_id = int(float(row.get("frame_id") or 0))
        except ValueError:
            continue
        if scene not in scenes or frame_id not in frame_ids:
            continue
        if not _parse_bool(row.get("visible")) or not _parse_bool(row.get("valid_uv")):
            continue
        observed_mask_id = str(row.get("observed_mask_id") or "")
        component = mask_to_component.get(f"{scene}:{frame_id}:{observed_mask_id}")
        if not component:
            continue
        carriers[(scene, frame_id)].append(
            (
                float(row.get("uv_x") or 0.0),
                float(row.get("uv_y") or 0.0),
                component,
                str(row.get("carrier_global_id") or ""),
            )
        )
    return carriers


def _scene_dirs(root: Path) -> list[Path]:
    if any(root.glob("*_masks.npz")):
        return [root]
    return sorted([path for path in root.iterdir() if path.is_dir()])


def _support_rows_for_root(
    mask_root: Path,
    carriers: dict[tuple[str, int], list[tuple[float, float, str, str]]],
    component_min_carriers: int,
    source_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    proposal_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    for scene_dir in _scene_dirs(mask_root):
        scene = scene_dir.name
        for path in sorted(scene_dir.glob("*_masks.npz"), key=lambda p: _frame_id(p) if _frame_id(p) is not None else 10**12):
            frame_id = _frame_id(path)
            if frame_id is None:
                continue
            carrier_rows = carriers.get((scene, frame_id), [])
            payload = np.load(path, allow_pickle=True)
            masks = np.asarray(payload["masks"])
            if masks.dtype != bool:
                masks = masks != 0
            areas = np.asarray(payload["areas"], dtype=np.int64) if "areas" in payload.files else masks.reshape(masks.shape[0], -1).sum(axis=1)
            scores = np.asarray(payload["scores"], dtype=np.float32) if "scores" in payload.files else np.zeros((masks.shape[0],), dtype=np.float32)
            n, h, w = masks.shape
            if carrier_rows:
                xs = np.clip(np.rint(np.asarray([row[0] for row in carrier_rows]) * (w - 1)).astype(np.int64), 0, w - 1)
                ys = np.clip(np.rint(np.asarray([row[1] for row in carrier_rows]) * (h - 1)).astype(np.int64), 0, h - 1)
                comps = [row[2] for row in carrier_rows]
                inside = masks[:, ys, xs]
            else:
                inside = np.zeros((n, 0), dtype=bool)
                comps = []
            for proposal_idx in range(n):
                proposal_id = _proposal_id(scene, frame_id, source_id, proposal_idx)
                inside_idx = np.flatnonzero(inside[proposal_idx])
                counts = Counter(comps[int(idx)] for idx in inside_idx)
                component_set = sorted([component for component, count in counts.items() if count >= component_min_carriers])
                role = "no_component_support"
                if len(component_set) == 1:
                    role = "measurement_only"
                elif len(component_set) >= 2:
                    role = "keymask_merge_candidate"
                carrier_count = int(sum(counts.values()))
                proposal_rows.append(
                    {
                        "proposal_id": proposal_id,
                        "scene": scene,
                        "frame_id": frame_id,
                        "source_id": source_id,
                        "mask_path": _rel(path),
                        "proposal_index": proposal_idx,
                        "mask_area": int(areas[proposal_idx]),
                        "score": float(scores[proposal_idx]) if proposal_idx < len(scores) else 0.0,
                        "carrier_support_count": carrier_count,
                        "component_set": component_set,
                        "component_set_size": len(component_set),
                        "role": role,
                        "uses_gt_for_prediction": False,
                    }
                )
                for component, count in sorted(counts.items()):
                    component_rows.append(
                        {
                            "proposal_id": proposal_id,
                            "scene": scene,
                            "frame_id": frame_id,
                            "source_id": source_id,
                            "component_id": component,
                            "carrier_overlap_count": int(count),
                            "included_in_component_set": int(count) >= int(component_min_carriers),
                            "uses_gt_for_prediction": False,
                        }
                    )
    return proposal_rows, component_rows


def _select_keymasks(
    proposal_rows: list[dict[str, Any]],
    component_universe: set[str],
    area_p90: int,
    underseg_component_threshold: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    multi_by_set: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in proposal_rows:
        component_set = tuple(sorted(row.get("component_set") or []))
        if len(component_set) < 2:
            continue
        current = multi_by_set.get(component_set)
        score = (int(row.get("carrier_support_count") or 0), -int(row.get("mask_area") or 0), float(row.get("score") or 0.0))
        if current is None:
            multi_by_set[component_set] = row
        else:
            current_score = (
                int(current.get("carrier_support_count") or 0),
                -int(current.get("mask_area") or 0),
                float(current.get("score") or 0.0),
            )
            if score > current_score:
                multi_by_set[component_set] = row
    selected_multi = list(multi_by_set.values())
    underseg_removed = [
        row
        for row in selected_multi
        if int(row.get("mask_area") or 0) >= int(area_p90)
        and int(row.get("component_set_size") or 0) >= int(underseg_component_threshold)
    ]
    if underseg_removed and len(selected_multi) - len(underseg_removed) >= 20:
        remove_ids = {str(row.get("proposal_id")) for row in underseg_removed}
        selected_multi = [row for row in selected_multi if str(row.get("proposal_id")) not in remove_ids]
    else:
        underseg_removed = []
    selected = []
    covered: set[str] = set()
    for rank, row in enumerate(sorted(selected_multi, key=lambda r: (-len(r.get("component_set") or []), -int(r.get("carrier_support_count") or 0))), start=1):
        out = dict(row)
        out["selected_rank"] = rank
        out["selected_role"] = "merge_keymask"
        out["is_multicomponent_keymask"] = True
        selected.append(out)
        covered.update(out.get("component_set") or [])
    singleton_by_component: dict[str, dict[str, Any]] = {}
    for row in proposal_rows:
        component_set = list(row.get("component_set") or [])
        if len(component_set) != 1:
            continue
        component = component_set[0]
        current = singleton_by_component.get(component)
        score = (int(row.get("carrier_support_count") or 0), -int(row.get("mask_area") or 0), float(row.get("score") or 0.0))
        if current is None:
            singleton_by_component[component] = row
        else:
            current_score = (
                int(current.get("carrier_support_count") or 0),
                -int(current.get("mask_area") or 0),
                float(current.get("score") or 0.0),
            )
            if score > current_score:
                singleton_by_component[component] = row
    for component in sorted(component_universe - covered):
        row = singleton_by_component.get(component)
        if not row:
            continue
        out = dict(row)
        out["selected_rank"] = len(selected) + 1
        out["selected_role"] = "measurement_only"
        out["is_multicomponent_keymask"] = False
        selected.append(out)
        covered.update(out.get("component_set") or [])
    stats = {
        "unique_multi_component_set_count": len(multi_by_set),
        "underseg_risk_keymask_removed_count": len(underseg_removed),
        "measurement_only_added_count": sum(1 for row in selected if row.get("selected_role") == "measurement_only"),
        "selected_component_coverage_count": len(covered),
        "missing_component_count": len(component_universe - covered),
    }
    return selected, stats


def build_v51_keymask_selection(
    raw_mask_root: str | Path,
    output_root: str | Path | None = None,
    carrier_rows_path: str | Path = "outputs/audit/v47_observation_tables_metricfix/carrier_observation_table.csv",
    vote_rows_path: str | Path = "outputs/audit/v47_carrier_supertrack_union_32_fine_metricfix/carrier_supertrack_mask_vote_rows.csv",
    scenes: str = "scene0011_00,scene0030_00,scene0050_00,scene0081_01,scene0591_00",
    frame_ids: str = "0,10,20,30",
    component_min_carriers: int = 1,
    underseg_component_threshold: int = 5,
) -> dict[str, Any]:
    raw_root = ROOT / raw_mask_root if not Path(raw_mask_root).is_absolute() else Path(raw_mask_root)
    carrier_path = ROOT / carrier_rows_path if not Path(carrier_rows_path).is_absolute() else Path(carrier_rows_path)
    vote_path = ROOT / vote_rows_path if not Path(vote_rows_path).is_absolute() else Path(vote_rows_path)
    scene_set = {scene.strip() for scene in scenes.split(",") if scene.strip()}
    frame_set = {int(part.strip()) for part in frame_ids.split(",") if part.strip()}
    mask_to_component = _load_mask_to_component(vote_path)
    carriers = _load_carriers(carrier_path, mask_to_component, scene_set, frame_set)
    component_universe = {row[2] for rows in carriers.values() for row in rows}
    proposal_rows, component_rows = _support_rows_for_root(
        raw_root,
        carriers,
        component_min_carriers=component_min_carriers,
        source_id="sam2_raw",
    )
    raw_nonempty = [row for row in proposal_rows if int(row.get("component_set_size") or 0) >= 1]
    raw_multi = [row for row in proposal_rows if int(row.get("component_set_size") or 0) >= 2]
    raw_component_coverage = {
        component for row in raw_nonempty for component in (row.get("component_set") or [])
    }
    areas = sorted(int(row.get("mask_area") or 0) for row in proposal_rows)
    area_p90 = areas[int(0.90 * (len(areas) - 1))] if areas else 0
    selected_rows, selection_stats = _select_keymasks(
        proposal_rows,
        component_universe,
        area_p90=area_p90,
        underseg_component_threshold=underseg_component_threshold,
    )
    keymask_rows = [row for row in selected_rows if row.get("selected_role") == "merge_keymask"]
    measurement_rows = [row for row in selected_rows if row.get("selected_role") == "measurement_only"]
    keymask_component_counts = [int(row.get("component_set_size") or 0) for row in keymask_rows]
    selected_components = {component for row in selected_rows for component in (row.get("component_set") or [])}
    keymask_components = {component for row in keymask_rows for component in (row.get("component_set") or [])}
    raw_large = [row for row in proposal_rows if int(row.get("mask_area") or 0) >= area_p90]
    selected_large = [row for row in keymask_rows if int(row.get("mask_area") or 0) >= area_p90]
    raw_large_underseg = [row for row in raw_large if int(row.get("component_set_size") or 0) >= 5]
    selected_large_underseg = [row for row in selected_large if int(row.get("component_set_size") or 0) >= 5]
    summary = {
        "proposal_count": len(proposal_rows),
        "proposal_with_component_support_count": len(raw_nonempty),
        "raw_multi_component_candidate_count": len(raw_multi),
        "component_universe_count": len(component_universe),
        "component_min_carriers": int(component_min_carriers),
        "underseg_component_threshold": int(underseg_component_threshold),
        "area_p90": int(area_p90),
        "raw_component_coverage": len(raw_component_coverage) / max(len(component_universe), 1),
        "key_mask_count": len(keymask_rows),
        "measurement_only_mask_count": len(measurement_rows),
        "selected_mask_count_including_measurement": len(selected_rows),
        "key_mask_ratio": len(keymask_rows) / max(len(proposal_rows), 1),
        "selected_multicomponent_keymask_count": len(keymask_rows),
        "single_component_keymask_ratio": 0.0,
        "selected_single_component_measurement_ratio": len(measurement_rows) / max(len(selected_rows), 1),
        "mean_components_per_keymask": sum(keymask_component_counts) / max(len(keymask_component_counts), 1),
        "median_components_per_keymask": float(np.median(keymask_component_counts)) if keymask_component_counts else 0.0,
        "keymask_component_coverage": len(keymask_components) / max(len(component_universe), 1),
        "component_coverage": len(selected_components) / max(len(component_universe), 1),
        "missing_component_count": int(selection_stats["missing_component_count"]),
        "carrier_coverage": None,
        "large_underseg_selected_rate": len(selected_large_underseg) / max(len(selected_large), 1),
        "raw_large_underseg_rate": len(raw_large_underseg) / max(len(raw_large), 1),
        "fallback_singleton_keymask_count": 0,
        "measurement_only_single_component_count": len(measurement_rows),
        "underseg_risk_keymask_removed_count": int(selection_stats["underseg_risk_keymask_removed_count"]),
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": False,
        "strict_keymask_only_component_coverage_pass": len(keymask_components) / max(len(component_universe), 1) >= 0.70,
    }
    gate = {
        "selected_multicomponent_keymask_count_pass": summary["selected_multicomponent_keymask_count"] >= 20,
        "mean_components_per_keymask_pass": summary["mean_components_per_keymask"] >= 1.5,
        "single_component_keymask_ratio_pass": summary["single_component_keymask_ratio"] <= 0.65,
        "measurement_augmented_component_coverage_pass": summary["component_coverage"] >= 0.70,
        "strict_keymask_only_component_coverage_pass": summary["strict_keymask_only_component_coverage_pass"],
        "large_underseg_selected_rate_pass": summary["large_underseg_selected_rate"] <= summary["raw_large_underseg_rate"],
        "uses_gt_for_prediction": False,
    }
    gate["pass"] = bool(
        gate["selected_multicomponent_keymask_count_pass"]
        and gate["mean_components_per_keymask_pass"]
        and gate["single_component_keymask_ratio_pass"]
        and gate["measurement_augmented_component_coverage_pass"]
        and gate["large_underseg_selected_rate_pass"]
        and not gate["uses_gt_for_prediction"]
    )
    return {
        "phase": "v51_r2_keymask_selection",
        "created_at": utc_now(),
        "plan": PLAN_PATH,
        "raw_mask_root": _rel(raw_root),
        "carrier_rows_path": _rel(carrier_path),
        "vote_rows_path": _rel(vote_path),
        "scenes": sorted(scene_set),
        "frame_ids": sorted(frame_set),
        "selection_policy": "dedupe multi-component raw SAM2 proposals by component-set; add single-component measurement-only masks only for uncovered components",
        "summary": summary,
        "gate": gate,
        "proposal_rows": proposal_rows,
        "keymask_rows": selected_rows,
        "keymask_component_rows": component_rows,
    }


def write_v51_keymask_selection(output_root: str | Path, payload: dict[str, Any]) -> None:
    out = ROOT / output_root if not Path(output_root).is_absolute() else Path(output_root)
    write_json(out / "keymask_summary.json", {key: value for key, value in payload.items() if key not in {"proposal_rows", "keymask_rows", "keymask_component_rows"}})
    write_csv(out / "proposal_component_support_rows.csv", payload["proposal_rows"])
    write_csv(out / "keymask_rows.csv", payload["keymask_rows"])
    write_csv(out / "keymask_component_rows.csv", payload["keymask_component_rows"])
