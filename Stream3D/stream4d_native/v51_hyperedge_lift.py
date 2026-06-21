from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

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


def _component_set(row: dict[str, Any]) -> tuple[str, ...]:
    text = row.get("component_set") or "[]"
    if isinstance(text, list):
        return tuple(sorted(str(item) for item in text))
    try:
        value = json.loads(str(text))
    except json.JSONDecodeError:
        value = []
    return tuple(sorted(str(item) for item in value))


def _score_frame_count(frame_count: float, max_frames_per_scene: int) -> float:
    return frame_count / max(float(max_frames_per_scene), 1.0)


def _shuffled_set(component_set: tuple[str, ...], frame_id: int, scene_components: dict[str, list[str]]) -> tuple[str, ...]:
    out: list[str] = []
    for component in component_set:
        scene = component.split("|", 1)[0] if "|" in component else ""
        comps = scene_components.get(scene, [])
        if component not in comps or not comps:
            out.append(component)
            continue
        idx = comps.index(component)
        offset = int(frame_id // 10) + 1
        out.append(comps[(idx + offset) % len(comps)])
    return tuple(sorted(out))


def build_v51_hyperedge_lift(
    keymask_root: str | Path,
    output_root: str | Path | None = None,
    max_frames_per_scene: int = 4,
) -> dict[str, Any]:
    root = ROOT / keymask_root if not Path(keymask_root).is_absolute() else Path(keymask_root)
    keymask_rows = _read_csv(root / "keymask_rows.csv")
    proposal_rows = _read_csv(root / "proposal_component_support_rows.csv")
    selected_sets = {
        _component_set(row)
        for row in keymask_rows
        if row.get("selected_role") == "merge_keymask" and len(_component_set(row)) >= 2
    }
    selected_keymask_by_set: dict[tuple[str, ...], dict[str, str]] = {}
    for row in keymask_rows:
        cset = _component_set(row)
        if row.get("selected_role") == "merge_keymask" and len(cset) >= 2:
            selected_keymask_by_set[cset] = row
    support_frames: dict[tuple[str, ...], set[tuple[str, int]]] = defaultdict(set)
    support_counts: Counter[tuple[str, ...]] = Counter()
    scene_components: dict[str, set[str]] = defaultdict(set)
    selected_support_rows: list[dict[str, str]] = []
    for row in proposal_rows:
        cset = _component_set(row)
        for component in cset:
            scene = component.split("|", 1)[0] if "|" in component else str(row.get("scene") or "")
            scene_components[scene].add(component)
        if cset not in selected_sets:
            continue
        frame_id = int(float(row.get("frame_id") or 0))
        support_frames[cset].add((str(row.get("scene") or ""), frame_id))
        support_counts[cset] += 1
        selected_support_rows.append(row)
    scene_component_lists = {scene: sorted(values) for scene, values in scene_components.items()}
    shuffled_frames: dict[tuple[str, ...], set[tuple[str, int]]] = defaultdict(set)
    for row in selected_support_rows:
        cset = _component_set(row)
        frame_id = int(float(row.get("frame_id") or 0))
        shuffled = _shuffled_set(cset, frame_id, scene_component_lists)
        shuffled_frames[shuffled].add((str(row.get("scene") or ""), frame_id))
    global_rows: list[dict[str, Any]] = []
    support_frame_counts: list[int] = []
    support_scores: list[float] = []
    for idx, cset in enumerate(sorted(selected_sets), start=1):
        frames = support_frames.get(cset, set())
        frame_count = len(frames)
        support_frame_counts.append(frame_count)
        score = _score_frame_count(frame_count, max_frames_per_scene=max_frames_per_scene)
        support_scores.append(score)
        keymask_row = selected_keymask_by_set.get(cset, {})
        global_rows.append(
            {
                "global_hyperedge_id": f"v51_he_{idx:05d}",
                "scene": str(keymask_row.get("scene") or ""),
                "component_set": list(cset),
                "component_set_size": len(cset),
                "support_frame_count": frame_count,
                "support_keymask_count": int(support_counts.get(cset, 0)),
                "support_chunk_count": 1 if frame_count else 0,
                "mean_support_score": score,
                "source_keymask_proposal_id": keymask_row.get("proposal_id"),
                "uses_gt_for_prediction": False,
            }
        )
    shuffled_counts = [len(frames) for frames in shuffled_frames.values()]
    real_mean_frame_count = sum(support_frame_counts) / max(len(support_frame_counts), 1)
    shuffled_mean_frame_count = sum(shuffled_counts) / max(len(shuffled_counts), 1)
    real_score = _score_frame_count(real_mean_frame_count, max_frames_per_scene=max_frames_per_scene)
    shuffled_score = _score_frame_count(shuffled_mean_frame_count, max_frames_per_scene=max_frames_per_scene)
    no_temporal_score = _score_frame_count(1.0, max_frames_per_scene=max_frames_per_scene)
    control_rows = [
        {
            "control": "real_d4rt_component_set_support",
            "mean_support_frame_count": real_mean_frame_count,
            "mean_support_score": real_score,
            "uses_gt_for_prediction": False,
        },
        {
            "control": "deterministic_framewise_component_shuffle",
            "mean_support_frame_count": shuffled_mean_frame_count,
            "mean_support_score": shuffled_score,
            "uses_gt_for_prediction": False,
        },
        {
            "control": "no_temporal_single_frame_baseline",
            "mean_support_frame_count": 1.0,
            "mean_support_score": no_temporal_score,
            "uses_gt_for_prediction": False,
        },
    ]
    summary = {
        "frame_hyperedge_count": len(selected_sets),
        "chunk_hyperedge_count": len(selected_sets),
        "global_hyperedge_count": len(global_rows),
        "mean_components_per_frame_hyperedge": sum(len(cset) for cset in selected_sets) / max(len(selected_sets), 1),
        "mean_components_per_global_hyperedge": sum(len(row["component_set"]) for row in global_rows) / max(len(global_rows), 1),
        "support_frame_count_mean": real_mean_frame_count,
        "support_frame_count_histogram": dict(sorted(Counter(support_frame_counts).items())),
        "support_chunk_count_mean": 1.0 if global_rows else 0.0,
        "support_keymask_count_mean": sum(int(row["support_keymask_count"]) for row in global_rows) / max(len(global_rows), 1),
        "mean_underseg_risk": None,
        "mean_semantic_contradiction": None,
        "mean_visible_outside_residual": None,
        "real_support_score": real_score,
        "shuffled_support_score": shuffled_score,
        "no_temporal_support_score": no_temporal_score,
        "real_minus_shuffled_support": real_score - shuffled_score,
        "real_minus_no_temporal_support": real_score - no_temporal_score,
        "uses_gt_for_prediction": False,
    }
    gate = {
        "frame_hyperedge_count_pass": len(selected_sets) >= 20,
        "mean_components_per_frame_hyperedge_pass": summary["mean_components_per_frame_hyperedge"] >= 1.5,
        "real_minus_shuffled_support_pass": summary["real_minus_shuffled_support"] >= 0.05,
        "real_minus_no_temporal_support_pass": summary["real_minus_no_temporal_support"] >= 0.03,
        "gt_diagnostics_not_evaluated": True,
        "uses_gt_for_prediction": False,
    }
    gate["pass"] = bool(
        gate["frame_hyperedge_count_pass"]
        and gate["mean_components_per_frame_hyperedge_pass"]
        and gate["real_minus_shuffled_support_pass"]
        and gate["real_minus_no_temporal_support_pass"]
        and not gate["uses_gt_for_prediction"]
    )
    return {
        "phase": "v51_r2_hyperedge_lift",
        "created_at": utc_now(),
        "plan": PLAN_PATH,
        "keymask_root": _rel(root),
        "max_frames_per_scene": int(max_frames_per_scene),
        "summary": summary,
        "gate": gate,
        "frame_hyperedge_rows": [
            {
                "frame_hyperedge_id": f"frame::{row['global_hyperedge_id']}",
                "global_hyperedge_id": row["global_hyperedge_id"],
                "scene": row["scene"],
                "component_set": row["component_set"],
                "component_set_size": row["component_set_size"],
                "support_frame_count": row["support_frame_count"],
                "uses_gt_for_prediction": False,
            }
            for row in global_rows
        ],
        "chunk_hyperedge_rows": global_rows,
        "global_hyperedge_rows": global_rows,
        "control_rows": control_rows,
    }


def write_v51_hyperedge_lift(output_root: str | Path, payload: dict[str, Any]) -> None:
    out = ROOT / output_root if not Path(output_root).is_absolute() else Path(output_root)
    write_json(out / "hyperedge_lift_summary.json", {key: value for key, value in payload.items() if not key.endswith("_rows")})
    write_csv(out / "frame_hyperedge_rows.csv", payload["frame_hyperedge_rows"])
    write_csv(out / "chunk_hyperedge_rows.csv", payload["chunk_hyperedge_rows"])
    write_csv(out / "global_hyperedge_rows.csv", payload["global_hyperedge_rows"])
    write_csv(out / "control_rows.csv", payload["control_rows"])
