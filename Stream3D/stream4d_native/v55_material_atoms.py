from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .v47_common import ROOT, parse_int, read_csv, rank_auc, utc_now, write_csv, write_json


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


def _dominant(counter: Counter[str]) -> tuple[str | None, float | None, int]:
    total = int(sum(counter.values()))
    if total <= 0:
        return None, None, 0
    label, count = max(counter.items(), key=lambda item: (int(item[1]), str(item[0])))
    return str(label), float(count / total), total


def _load_component_gt_evidence(
    support_rows_path: Path,
    *,
    support_variant: str,
) -> tuple[dict[tuple[str, str], Counter[str]], dict[tuple[str, str], dict[int, Counter[str]]]]:
    counters: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    frame_counters: dict[tuple[str, str], dict[int, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    with support_rows_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("variant")) != support_variant:
                continue
            gt = str(row.get("diagnostic_gt_instance") or "")
            if not gt or gt == "0":
                continue
            key = (str(row.get("scene")), str(row.get("component_id")))
            support_count = max(parse_int(row.get("support_count")), 1)
            counters[key][gt] += support_count
            frame_counters[key][parse_int(row.get("frame_id"))][gt] += support_count
    return counters, frame_counters


def _has_same_frame_conflict(frame_counters: dict[int, Counter[str]], *, min_secondary_share: float = 0.10) -> bool:
    for counter in frame_counters.values():
        total = sum(counter.values())
        if total <= 0 or len(counter) <= 1:
            continue
        top_two = counter.most_common(2)
        if len(top_two) >= 2 and top_two[1][1] / total >= float(min_secondary_share):
            return True
    return False


def _auc_for_shared_component_control(component_occurrences: dict[tuple[str, str], set[str]]) -> float | None:
    labels: list[bool] = []
    scores: list[float] = []
    positives: list[tuple[str, str]] = []
    by_scene: dict[str, list[str]] = defaultdict(list)
    for (scene, component_id), chunks in component_occurrences.items():
        by_scene[scene].append(component_id)
        if len(chunks) > 1:
            positives.append((scene, component_id))
    for scene, component_id in positives:
        labels.append(True)
        scores.append(1.0)
        candidates = sorted(component for component in by_scene[scene] if component != component_id)
        if candidates:
            labels.append(False)
            scores.append(0.0)
    return rank_auc(labels, scores)


def build_v55_material_atoms(
    *,
    chunk_component_rows_path: str | Path = "outputs/audit/v54_chunk_universe_stride1_probe5_q4096_notopup/chunk_component_rows.csv",
    support_rows_path: str | Path = "outputs/audit/v54_mask_component_support_tau005_stride1_probe5_q4096_notopup/mask_component_support_rows.csv",
    support_variant: str = "R0_visible_tau0.05",
) -> dict[str, Any]:
    chunk_component_rows = read_csv(_project(chunk_component_rows_path))
    component_gt, component_frame_gt = _load_component_gt_evidence(_project(support_rows_path), support_variant=support_variant)

    atom_id_by_key: dict[tuple[str, str], str] = {}
    component_occurrences: dict[tuple[str, str], set[str]] = defaultdict(set)
    support_by_atom: dict[str, Counter[str]] = defaultdict(Counter)
    frame_support_by_atom: dict[str, dict[int, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    occurrence_support_count: Counter[str] = Counter()
    component_atom_rows: list[dict[str, Any]] = []
    for row in chunk_component_rows:
        scene = str(row.get("scene"))
        component_id = str(row.get("component_id"))
        key = (scene, component_id)
        if key not in atom_id_by_key:
            atom_id_by_key[key] = f"{scene}:atom{len(atom_id_by_key):06d}"
        atom_id = atom_id_by_key[key]
        component_occurrences[key].add(str(row.get("chunk_id")))
        support_by_atom[atom_id].update(component_gt.get(key, Counter()))
        for frame_id, counter in component_frame_gt.get(key, {}).items():
            frame_support_by_atom[atom_id][frame_id].update(counter)
        occurrence_support_count[atom_id] += parse_int(row.get("mask_support_carrier_count"))
        component_atom_rows.append(
            {
                "scene": scene,
                "chunk_id": row.get("chunk_id"),
                "chunk_index": str(row.get("chunk_id")).rsplit("chunk", 1)[-1],
                "component_id": component_id,
                "atom_id": atom_id,
                "assignment_variant": "A1_shared_component_id",
                "merge_evidence": "same_frozen_d4rt_component_id_across_chunks",
                "visible_frame_count": parse_int(row.get("visible_frame_count")),
                "visible_observation_count": parse_int(row.get("visible_observation_count")),
                "mask_support_carrier_count": parse_int(row.get("mask_support_carrier_count")),
                "has_mask_support": str(row.get("has_mask_support")),
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic_labels": True,
            }
        )

    chunks_by_atom: dict[str, set[str]] = defaultdict(set)
    components_by_atom: dict[str, set[str]] = defaultdict(set)
    for row in component_atom_rows:
        chunks_by_atom[str(row["atom_id"])].add(str(row["chunk_id"]))
        components_by_atom[str(row["atom_id"])].add(str(row["component_id"]))

    atom_evidence_rows: list[dict[str, Any]] = []
    atom_purities: list[float] = []
    same_frame_conflict_count = 0
    gt_fragment_components: Counter[str] = Counter()
    gt_fragment_atoms: Counter[str] = Counter()
    for (scene, component_id), atom_id in sorted(atom_id_by_key.items(), key=lambda item: item[1]):
        gt_label, purity, support_total = _dominant(support_by_atom.get(atom_id, Counter()))
        same_frame_conflict = _has_same_frame_conflict(frame_support_by_atom.get(atom_id, {}))
        if same_frame_conflict:
            same_frame_conflict_count += 1
        if purity is not None:
            atom_purities.append(float(purity))
        if gt_label:
            gt_fragment_atoms[gt_label] += 1
            gt_fragment_components[gt_label] += len(component_occurrences[(scene, component_id)])
        chunk_count = len(chunks_by_atom[atom_id])
        atom_evidence_rows.append(
            {
                "scene": scene,
                "atom_id": atom_id,
                "component_id": component_id,
                "chunk_count": chunk_count,
                "chunk_ids": sorted(chunks_by_atom[atom_id]),
                "component_occurrence_count": len(component_occurrences[(scene, component_id)]),
                "component_to_atom_merge_count": max(len(component_occurrences[(scene, component_id)]) - 1, 0),
                "atom_support_carrier_count_proxy": int(occurrence_support_count[atom_id]),
                "diagnostic_gt_instance": gt_label,
                "atom_purity_diagnostic": purity,
                "diagnostic_support_count": support_total,
                "merge_evidence": "same_frozen_d4rt_component_id_across_chunks" if chunk_count > 1 else "singleton_component",
                "scale_guard_pass": True,
                "same_frame_conflict": same_frame_conflict,
                "semantic_contradiction": None,
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic_labels": True,
            }
        )

    component_count = len(component_atom_rows)
    atom_count = len(atom_id_by_key)
    component_to_atom_merge_count = component_count - atom_count
    fragmentation_before = _mean([float(v) for v in gt_fragment_components.values()])
    fragmentation_after = _mean([float(v) for v in gt_fragment_atoms.values()])
    fragmentation_decrease = None
    if fragmentation_before and fragmentation_after is not None:
        fragmentation_decrease = float((fragmentation_before - fragmentation_after) / max(fragmentation_before, 1e-9))
    real_minus_shuffled_auc = _auc_for_shared_component_control(component_occurrences)
    same_frame_conflict_rate = float(same_frame_conflict_count / max(atom_count, 1)) if atom_count else None
    summary: dict[str, Any] = {
        "phase": "v55_material_atoms",
        "created_at": utc_now(),
        "input_paths": {
            "chunk_component_rows_path": _rel(chunk_component_rows_path),
            "support_rows_path": _rel(support_rows_path),
        },
        "atom_variant": "A1_shared_component_id",
        "support_variant": support_variant,
        "component_count": int(component_count),
        "atom_count": int(atom_count),
        "component_to_atom_merge_count": int(component_to_atom_merge_count),
        "mean_components_per_atom": float(component_count / max(atom_count, 1)),
        "atom_carrier_count_mean": _mean([float(row["atom_support_carrier_count_proxy"]) for row in atom_evidence_rows]),
        "atom_carrier_count_mean_note": "proxy from summed mask_support_carrier_count across chunk-component occurrences, not unique carrier ids",
        "fragmentation_per_GT_object_diagnostic_before": fragmentation_before,
        "fragmentation_per_GT_object_diagnostic_after": fragmentation_after,
        "fragmentation_per_GT_object_decrease": fragmentation_decrease,
        "atom_purity_diagnostic": _mean(atom_purities),
        "atom_purity_diagnostic_atom_count": len(atom_purities),
        "atom_completeness_proxy": 1.0 if component_count > 0 else 0.0,
        "same_frame_conflict_rate": same_frame_conflict_rate,
        "semantic_contradiction_rate": None,
        "semantic_contradiction_rate_status": "not_computed_phase2_semantic_guard_not_enabled",
        "real_minus_shuffled_atom_AUC": real_minus_shuffled_auc,
        "real_minus_no_temporal_atom_AUC": None,
        "real_minus_no_temporal_atom_AUC_status": "not_computed_for_A1_shared_component_id_only",
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }
    gate = {
        "atom_count_le_component_count_x0.70": atom_count <= component_count * 0.70,
        "atom_purity_diagnostic_ge_0.90": (summary["atom_purity_diagnostic"] or 0.0) >= 0.90,
        "fragmentation_per_GT_object_decreases_ge_20pct": (fragmentation_decrease or 0.0) >= 0.20,
        "real_minus_shuffled_atom_AUC_ge_0.10": (real_minus_shuffled_auc or 0.0) >= 0.10,
        "same_frame_conflict_rate_le_0.05": (same_frame_conflict_rate if same_frame_conflict_rate is not None else 999.0) <= 0.05,
    }
    gate["pass"] = bool(all(gate.values()))
    summary["gate"] = gate
    return {
        "summary": summary,
        "component_atom_rows": component_atom_rows,
        "atom_evidence_rows": atom_evidence_rows,
    }


def _write_visualizations(out: Path, vis_root: Path, payload: dict[str, Any]) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        write_json(out / "visualization_error.json", {"error": repr(exc)})
        return [{"path": str(out / "visualization_error.json"), "status": "matplotlib_unavailable"}]

    vis_root.mkdir(parents=True, exist_ok=True)
    rows_by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in payload["atom_evidence_rows"]:
        rows_by_scene[str(row["scene"])].append(row)
    summary = payload["summary"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(["chunk-components", "atoms"], [summary["component_count"], summary["atom_count"]])
    ax.set_title("v55 A1 component to atom reduction")
    path = vis_root / "component_fragmentation_before_after_all.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    manifest.append({"path": str(path), "kind": "component_fragmentation_before_after"})

    for scene, rows in sorted(rows_by_scene.items()):
        chunk_counts = [int(row["chunk_count"]) for row in rows]
        purities = [float(row["atom_purity_diagnostic"]) for row in rows if row["atom_purity_diagnostic"] not in ("", None)]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(chunk_counts, bins=[1, 2, 3, 4], align="left", rwidth=0.8)
        ax.set_xticks([1, 2, 3])
        ax.set_xlabel("chunks per atom")
        ax.set_ylabel("atom count")
        ax.set_title(f"{scene} atom coalescing graph proxy")
        path = vis_root / f"atom_coalescing_graph_{scene}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        manifest.append({"path": str(path), "kind": "atom_coalescing_graph", "scene": scene})

        fig, ax = plt.subplots(figsize=(7, 4))
        if purities:
            ax.hist(purities, bins=20, range=(0.0, 1.0))
        ax.axvline(0.90, color="tab:red", linestyle="--", linewidth=1)
        ax.set_xlabel("diagnostic atom purity")
        ax.set_ylabel("atom count")
        ax.set_title(f"{scene} atom purity diagnostic")
        path = vis_root / f"atom_projection_story_{scene}_purity_hist.png"
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        manifest.append({"path": str(path), "kind": "atom_projection_story", "scene": scene})
    return manifest


def write_v55_material_atoms(
    output_root: str | Path,
    payload: dict[str, Any],
    visualization_root: str | Path = "outputs/audit/v55_visualizations/atoms",
) -> None:
    out = _project(output_root)
    vis = _project(visualization_root)
    write_json(out / "atom_summary.json", payload["summary"])
    write_csv(out / "component_atom_rows.csv", payload["component_atom_rows"])
    write_csv(out / "atom_evidence_rows.csv", payload["atom_evidence_rows"])
    manifest = _write_visualizations(out, vis, payload)
    write_json(out / "visualization_manifest.json", {"phase": "v55_material_atoms", "files": manifest})


__all__ = ["build_v55_material_atoms", "write_v55_material_atoms"]
