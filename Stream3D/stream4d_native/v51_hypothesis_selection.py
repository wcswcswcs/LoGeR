from __future__ import annotations

import csv
import json
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


def _components(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    try:
        parsed = json.loads(str(value or "[]"))
    except json.JSONDecodeError:
        parsed = []
    return [str(item) for item in parsed]


def build_v51_hypothesis_selection(
    hyperedge_root: str | Path,
    semantic_root: str | Path,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    hyp_root = ROOT / hyperedge_root if not Path(hyperedge_root).is_absolute() else Path(hyperedge_root)
    sem_root = ROOT / semantic_root if not Path(semantic_root).is_absolute() else Path(semantic_root)
    global_rows = _read_csv(hyp_root / "global_hyperedge_rows.csv")
    semantic_rows = _read_csv(sem_root / "semantic_reliability_rows.csv")
    semantic_by_proposal = {str(row.get("proposal_id") or ""): row for row in semantic_rows}
    candidate_rows: list[dict[str, Any]] = []
    for row in global_rows:
        proposal_id = str(row.get("source_keymask_proposal_id") or "")
        sem = semantic_by_proposal.get(proposal_id, {})
        components = _components(row.get("component_set"))
        contradiction = float(sem.get("semantic_contradiction") or 0.0)
        semantic_keep = str(sem.get("semantic_keep", "True")).lower() == "true"
        support_score = float(row.get("mean_support_score") or 0.0)
        score = support_score + 0.05 * len(components) - 0.25 * contradiction
        candidate_rows.append(
            {
                "hypothesis_id": row.get("global_hyperedge_id"),
                "scene": row.get("scene"),
                "component_set": components,
                "component_set_size": len(components),
                "source_keymask_proposal_id": proposal_id,
                "support_frame_count": int(float(row.get("support_frame_count") or 0)),
                "support_keymask_count": int(float(row.get("support_keymask_count") or 0)),
                "mean_support_score": support_score,
                "semantic_contradiction": contradiction,
                "semantic_keep": semantic_keep,
                "selection_score": score,
                "candidate_status": "eligible" if semantic_keep else "semantic_guard_rejected",
                "uses_gt_for_prediction": False,
            }
        )
    selected_rows: list[dict[str, Any]] = []
    used_components: set[str] = set()
    for row in sorted(candidate_rows, key=lambda item: (item["candidate_status"] == "eligible", item["selection_score"]), reverse=True):
        if row["candidate_status"] != "eligible":
            continue
        components = set(row["component_set"])
        if components & used_components:
            continue
        out = dict(row)
        out["selected_rank"] = len(selected_rows) + 1
        out["selected"] = True
        selected_rows.append(out)
        used_components.update(components)
    candidate_components = {component for row in candidate_rows for component in row["component_set"]}
    selected_components = {component for row in selected_rows for component in row["component_set"]}
    scene_count = len({str(row.get("scene") or "") for row in candidate_rows if row.get("scene")})
    duplicate_rate = 0.0
    conflict_rate = sum(1 for row in selected_rows if float(row.get("semantic_contradiction") or 0.0) > 0.80) / max(len(selected_rows), 1)
    summary = {
        "candidate_hypothesis_count": len(candidate_rows),
        "semantic_rejected_candidate_count": sum(1 for row in candidate_rows if row["candidate_status"] != "eligible"),
        "selected_object_count": len(selected_rows),
        "mean_predictions_per_scene": len(selected_rows) / max(scene_count, 1),
        "selected_component_count": len(selected_components),
        "candidate_component_count": len(candidate_components),
        "selected_component_coverage": len(selected_components) / max(len(candidate_components), 1),
        "duplicate_rate": duplicate_rate,
        "conflict_rate": conflict_rate,
        "maskless_object_count": 0,
        "birth_from_d4rt_tube_count": 0,
        "native_3d_materialization_available": False,
        "uses_gt_for_prediction": False,
    }
    gate = {
        "selected_object_count_pass": len(selected_rows) > 0,
        "mean_predictions_per_scene_pass": summary["mean_predictions_per_scene"] <= 150,
        "duplicate_rate_pass": duplicate_rate <= 0.05,
        "conflict_rate_pass": conflict_rate <= 0.10,
        "birth_from_d4rt_tube_count_pass": True,
        "maskless_object_count_pass": True,
        "native_3d_materialization_available": False,
        "uses_gt_for_prediction": False,
    }
    gate["pass"] = bool(
        gate["selected_object_count_pass"]
        and gate["mean_predictions_per_scene_pass"]
        and gate["duplicate_rate_pass"]
        and gate["conflict_rate_pass"]
        and gate["birth_from_d4rt_tube_count_pass"]
        and gate["maskless_object_count_pass"]
        and not gate["uses_gt_for_prediction"]
    )
    return {
        "phase": "v51_r2_hypothesis_selection",
        "created_at": utc_now(),
        "plan": PLAN_PATH,
        "hyperedge_root": _rel(hyp_root),
        "semantic_root": _rel(sem_root),
        "summary": summary,
        "gate": gate,
        "candidate_rows": candidate_rows,
        "selected_rows": selected_rows,
    }


def write_v51_hypothesis_selection(output_root: str | Path, payload: dict[str, Any]) -> None:
    out = ROOT / output_root if not Path(output_root).is_absolute() else Path(output_root)
    write_json(out / "hypothesis_selection_summary.json", {key: value for key, value in payload.items() if key not in {"candidate_rows", "selected_rows"}})
    write_csv(out / "hypothesis_candidate_rows.csv", payload["candidate_rows"])
    write_csv(out / "selected_object_rows.csv", payload["selected_rows"])
