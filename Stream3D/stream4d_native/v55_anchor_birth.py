from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .v47_common import ROOT, parse_bool, parse_float, parse_int, read_csv, read_json, utc_now, write_csv, write_json


def _project(path: str | Path) -> Path:
    path_obj = Path(path)
    return path_obj if path_obj.is_absolute() else ROOT / path_obj


def _rel(path: str | Path) -> str:
    path_obj = _project(path)
    try:
        return str(path_obj.relative_to(ROOT))
    except ValueError:
        return str(path_obj)


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _load_list(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    try:
        payload = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    return [str(item) for item in payload]


def _load_component_atom_map(rows: list[dict[str, Any]]) -> dict[tuple[str, str], str]:
    out: dict[tuple[str, str], str] = {}
    for row in rows:
        out[(str(row.get("scene")), str(row.get("component_id")))] = str(row.get("atom_id"))
    return out


def _diagnostic_gt_count_for_anchor_chunks(
    support_rows_path: Path,
    *,
    support_variant: str,
    anchor_chunks: set[str],
) -> dict[str, int]:
    gt_by_scene: dict[str, set[str]] = defaultdict(set)
    with support_rows_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("variant")) != support_variant:
                continue
            gt = str(row.get("diagnostic_gt_instance") or "")
            if not gt or gt == "0":
                continue
            # Support rows do not carry chunk_id, so this diagnostic is scene-level for anchor scenes.
            scene = str(row.get("scene"))
            gt_by_scene[scene].add(gt)
    return {scene: len(labels) for scene, labels in gt_by_scene.items()}


def build_v55_anchor_birth(
    *,
    chunk_role_rows_path: str | Path = "outputs/audit/v55_chunk_roles/chunk_role_rows.csv",
    objectlet_rows_path: str | Path = "outputs/audit/v54_local_objectlets_k0all_conflict_veto018_skip_repeated_sig_l12_stride1_probe5_q4096_notopup_max4000_skip/objectlet_rows.csv",
    local_metric_rows_path: str | Path = "outputs/audit/v54_local_reproduction_stride1_probe5_q4096_notopup_k0all_max4000_veto018_w123_fullchunk/local_metric_rows.csv",
    local_summary_path: str | Path = "outputs/audit/v54_local_reproduction_stride1_probe5_q4096_notopup_k0all_max4000_veto018_w123_fullchunk/local_reproduction_summary.json",
    component_atom_rows_path: str | Path = "outputs/audit/v55_atoms/component_atom_rows.csv",
    support_rows_path: str | Path = "outputs/audit/v54_mask_component_support_tau005_stride1_probe5_q4096_notopup/mask_component_support_rows.csv",
    support_variant: str = "R0_visible_tau0.05",
) -> dict[str, Any]:
    role_rows = read_csv(_project(chunk_role_rows_path))
    objectlet_rows = read_csv(_project(objectlet_rows_path))
    local_metric_rows = read_csv(_project(local_metric_rows_path))
    local_summary = read_json(_project(local_summary_path))
    component_atom_rows = read_csv(_project(component_atom_rows_path))
    component_to_atom = _load_component_atom_map(component_atom_rows)
    anchor_chunks = {str(row.get("chunk_id")) for row in role_rows if str(row.get("role")) == "anchor"}
    anchor_scenes = {str(row.get("scene")) for row in role_rows if str(row.get("role")) == "anchor"}
    best_variant = str(local_summary.get("best_method_variant") or "")

    anchor_metric_rows = [
        row for row in local_metric_rows if str(row.get("variant")) == best_variant and str(row.get("chunk_id")) in anchor_chunks
    ]
    candidate_rows = [
        row for row in objectlet_rows if str(row.get("variant")) == best_variant and str(row.get("chunk_id")) in anchor_chunks
    ]
    birth_rows: list[dict[str, Any]] = []
    accepted_birth_count_by_scene: Counter[str] = Counter()
    for row in candidate_rows:
        component_ids = _load_list(row.get("component_ids"))
        atom_ids = sorted({component_to_atom.get((str(row.get("scene")), component_id), "") for component_id in component_ids})
        atom_ids = [atom_id for atom_id in atom_ids if atom_id]
        has_mask_evidence = bool(str(row.get("source_mask_observation_id") or ""))
        conflict_rate = parse_float(row.get("same_frame_exclusion_violation_rate"))
        outside_residual = parse_float(row.get("outside_all_related_masks_ratio_mean"))
        accepted = bool(has_mask_evidence and component_ids and conflict_rate <= 0.08 and outside_residual <= 0.35)
        if accepted:
            accepted_birth_count_by_scene[str(row.get("scene"))] += 1
        birth_rows.append(
            {
                "scene": row.get("scene"),
                "anchor_chunk_id": row.get("chunk_id"),
                "birth_object_id": row.get("objectlet_id"),
                "birth_variant": best_variant,
                "source_mask_observation_id": row.get("source_mask_observation_id"),
                "has_mask_evidence": has_mask_evidence,
                "accepted_birth": accepted,
                "atom_count": len(atom_ids),
                "component_count": len(component_ids),
                "birth_mask_support_count": 1 if has_mask_evidence else 0,
                "candidate_success_rate": parse_float(row.get("candidate_success_rate")),
                "outside_all_related_masks_ratio_mean": outside_residual,
                "same_frame_exclusion_violation_rate": conflict_rate,
                "underseg_proxy": parse_bool(row.get("underseg_proxy")),
                "birth_from_d4rt_only": bool(not has_mask_evidence),
                "atom_ids": atom_ids,
                "component_ids": component_ids,
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic_labels": True,
            }
        )

    gt_count_by_scene = _diagnostic_gt_count_for_anchor_chunks(
        _project(support_rows_path),
        support_variant=support_variant,
        anchor_chunks=anchor_chunks,
    )
    total_gt_objects = sum(gt_count_by_scene.get(scene, 0) for scene in anchor_scenes)
    accepted_rows = [row for row in birth_rows if row["accepted_birth"]]
    birth_from_d4rt_only_count = sum(1 for row in birth_rows if row["birth_from_d4rt_only"])
    accepted_birth_count = len(accepted_rows)
    anchor_birth_purity = _mean([parse_float(row.get("local_purity")) for row in anchor_metric_rows])
    anchor_birth_completeness = _mean([parse_float(row.get("local_completeness")) for row in anchor_metric_rows])
    birth_conflict_rate = _mean([float(row["same_frame_exclusion_violation_rate"]) for row in accepted_rows])
    summary = {
        "phase": "v55_anchor_birth",
        "created_at": utc_now(),
        "input_paths": {
            "chunk_role_rows_path": _rel(chunk_role_rows_path),
            "objectlet_rows_path": _rel(objectlet_rows_path),
            "local_metric_rows_path": _rel(local_metric_rows_path),
            "local_summary_path": _rel(local_summary_path),
            "component_atom_rows_path": _rel(component_atom_rows_path),
            "support_rows_path": _rel(support_rows_path),
        },
        "birth_variant": best_variant,
        "anchor_chunk_count": len(anchor_chunks),
        "anchor_scenes": sorted(anchor_scenes),
        "birth_candidate_count": len(candidate_rows),
        "accepted_birth_count": accepted_birth_count,
        "mean_atoms_per_birth": _mean([float(row["atom_count"]) for row in accepted_rows]),
        "mean_components_per_birth": _mean([float(row["component_count"]) for row in accepted_rows]),
        "birth_mask_support_count": int(sum(int(row["birth_mask_support_count"]) for row in accepted_rows)),
        "birth_purity_diagnostic": anchor_birth_purity,
        "birth_completeness_diagnostic": anchor_birth_completeness,
        "birth_conflict_rate": birth_conflict_rate,
        "birth_underseg_rate": _mean([1.0 if row["underseg_proxy"] else 0.0 for row in accepted_rows]),
        "birth_from_d4rt_only_count": birth_from_d4rt_only_count,
        "GT_object_count_diagnostic": total_gt_objects,
        "accepted_birth_to_GT_object_ratio_diagnostic": float(accepted_birth_count / max(total_gt_objects, 1)),
        "accepted_birth_count_by_scene": dict(accepted_birth_count_by_scene),
        "GT_object_count_by_scene_diagnostic": gt_count_by_scene,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }
    gate = {
        "birth_from_d4rt_only_count_eq_0": birth_from_d4rt_only_count == 0,
        "anchor_birth_purity_ge_0.88": (anchor_birth_purity or 0.0) >= 0.88,
        "anchor_birth_completeness_ge_0.60": (anchor_birth_completeness or 0.0) >= 0.60,
        "birth_conflict_rate_le_0.08": (birth_conflict_rate if birth_conflict_rate is not None else 999.0) <= 0.08,
        "accepted_birth_count_ge_GT_object_count_x0.50_diagnostic": accepted_birth_count >= total_gt_objects * 0.50,
    }
    gate["pass"] = bool(all(gate.values()))
    summary["gate"] = gate
    return {"summary": summary, "anchor_birth_rows": birth_rows}


def _write_visualizations(out: Path, vis_root: Path, payload: dict[str, Any]) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        write_json(out / "visualization_error.json", {"error": repr(exc)})
        return [{"path": str(out / "visualization_error.json"), "status": "matplotlib_unavailable"}]

    vis_root.mkdir(parents=True, exist_ok=True)
    rows_by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in payload["anchor_birth_rows"]:
        rows_by_scene[str(row["scene"])].append(row)
    for scene, rows in sorted(rows_by_scene.items()):
        accepted = [row for row in rows if row["accepted_birth"]]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(["candidates", "accepted", "d4rt_only"], [len(rows), len(accepted), sum(1 for row in rows if row["birth_from_d4rt_only"])])
        ax.set_title(f"{scene} anchor birth panel")
        path = vis_root / f"anchor_birth_panel_{scene}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        manifest.append({"path": str(path), "kind": "anchor_birth_panel", "scene": scene})

        casebook = vis_root / f"birth_failure_casebook_{scene}.md"
        rejected = [row for row in rows if not row["accepted_birth"]]
        lines = [
            f"# Birth Failure Casebook {scene}",
            "",
            f"candidate_count={len(rows)}",
            f"accepted_birth_count={len(accepted)}",
            f"rejected_birth_count={len(rejected)}",
            "",
            "Rejected examples:",
        ]
        for row in rejected[:20]:
            lines.append(
                f"- {row['birth_object_id']}: mask={row['has_mask_evidence']} "
                f"components={row['component_count']} conflict={row['same_frame_exclusion_violation_rate']} "
                f"outside={row['outside_all_related_masks_ratio_mean']}"
            )
        casebook.write_text("\n".join(lines) + "\n", encoding="utf-8")
        manifest.append({"path": str(casebook), "kind": "birth_failure_casebook", "scene": scene})
    return manifest


def write_v55_anchor_birth(
    output_root: str | Path,
    payload: dict[str, Any],
    visualization_root: str | Path = "outputs/audit/v55_visualizations/anchor_birth",
) -> None:
    out = _project(output_root)
    vis = _project(visualization_root)
    write_json(out / "anchor_birth_summary.json", payload["summary"])
    write_csv(out / "anchor_birth_rows.csv", payload["anchor_birth_rows"])
    manifest = _write_visualizations(out, vis, payload)
    write_json(out / "visualization_manifest.json", {"phase": "v55_anchor_birth", "files": manifest})


__all__ = ["build_v55_anchor_birth", "write_v55_anchor_birth"]
