from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .v47_common import ROOT, load_mask_label, parse_float, parse_int, read_csv, utc_now, write_csv, write_json


def _project(path: str | Path) -> Path:
    path_obj = Path(path)
    return path_obj if path_obj.is_absolute() else ROOT / path_obj


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _quantile(values: list[float], q: float) -> float | None:
    return float(np.quantile(values, q)) if values else None


def _stable_score(value: str, seed: str = "v53_representative") -> float:
    digest = hashlib.sha1(f"{seed}:{value}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12)


def _load_support_sets(support_rows_path: Path, support_variant: str) -> tuple[dict[str, set[str]], dict[str, Counter[str]]]:
    components_by_mask: dict[str, set[str]] = defaultdict(set)
    counts_by_mask: dict[str, Counter[str]] = defaultdict(Counter)
    with support_rows_path.open(newline="", encoding="utf-8") as handle:
        import csv

        for row in csv.DictReader(handle):
            if str(row.get("variant")) != support_variant:
                continue
            mask_observation_id = str(row.get("mask_observation_id"))
            component_id = str(row.get("component_id"))
            components_by_mask[mask_observation_id].add(component_id)
            counts_by_mask[mask_observation_id][component_id] += parse_int(row.get("support_count"))
    return components_by_mask, counts_by_mask


def _mask_meta(mask_summary_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("mask_observation_id")): row for row in mask_summary_rows}


def _underseg_proxy(row: dict[str, Any], component_count_key: str) -> bool:
    component_count = parse_int(row.get(component_count_key))
    entropy = parse_float(row.get("raw_component_entropy"), 0.0)
    return bool(component_count >= 20 or entropy >= 1.5)


def _select_greedy(
    mask_ids: list[str],
    universe: set[str],
    components_by_mask: dict[str, set[str]],
    meta_by_mask: dict[str, dict[str, Any]],
    max_selected: int,
    component_count_key: str,
    redundancy_penalty: float = 0.0,
    underseg_penalty: float = 0.0,
    cannot_link_penalty: float = 0.0,
    underseg_rate_cap: float | None = None,
) -> tuple[list[str], list[dict[str, Any]]]:
    selected: list[str] = []
    covered: set[str] = set()
    selected_frame_components: dict[tuple[str, int], set[str]] = defaultdict(set)
    progress: list[dict[str, Any]] = []
    remaining = set(mask_ids)
    while remaining and len(selected) < max_selected and (len(covered) / max(len(universe), 1)) < 0.75:
        best_id = ""
        best_score = -1e18
        best_gain: set[str] = set()
        for mask_id in remaining:
            comps = components_by_mask.get(mask_id, set()) & universe
            gain = comps - covered
            if not gain:
                continue
            row = meta_by_mask.get(mask_id, {})
            frame_key = (str(row.get("scene")), parse_int(row.get("frame_id")))
            redundancy = len(comps & covered)
            underseg = 1 if _underseg_proxy(row, component_count_key) else 0
            if underseg_rate_cap is not None and underseg:
                current_underseg = sum(
                    1 for selected_id in selected if _underseg_proxy(meta_by_mask.get(selected_id, {}), component_count_key)
                )
                if (current_underseg + 1) / max(len(selected) + 1, 1) > float(underseg_rate_cap):
                    continue
            same_frame_overlap = len(comps & selected_frame_components[frame_key])
            score = float(len(gain)) - redundancy_penalty * redundancy - underseg_penalty * underseg - cannot_link_penalty * same_frame_overlap
            if score > best_score:
                best_score = score
                best_id = mask_id
                best_gain = gain
        if not best_id:
            break
        selected.append(best_id)
        remaining.remove(best_id)
        row = meta_by_mask.get(best_id, {})
        comps = components_by_mask.get(best_id, set()) & universe
        covered |= comps
        selected_frame_components[(str(row.get("scene")), parse_int(row.get("frame_id")))] |= comps
        progress.append(
            {
                "step": len(selected),
                "selected_mask_observation_id": best_id,
                "marginal_component_gain": len(best_gain),
                "component_coverage": len(covered) / max(len(universe), 1),
                "underseg_proxy": _underseg_proxy(row, component_count_key),
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic_labels": True,
            }
        )
    return selected, progress


def _same_frame_conflict_rate(mask_ids: list[str], components_by_mask: dict[str, set[str]], meta_by_mask: dict[str, dict[str, Any]]) -> float:
    frame_component_owner: dict[tuple[str, int, str], str] = {}
    conflict_count = 0
    checked = 0
    for mask_id in mask_ids:
        row = meta_by_mask.get(mask_id, {})
        scene = str(row.get("scene"))
        frame_id = parse_int(row.get("frame_id"))
        for component_id in components_by_mask.get(mask_id, set()):
            checked += 1
            key = (scene, frame_id, component_id)
            owner = frame_component_owner.get(key)
            if owner is not None and owner != mask_id:
                conflict_count += 1
            frame_component_owner[key] = mask_id
    return float(conflict_count / max(checked, 1))


def _diagnostic_gt_representative(
    selected_mask_ids: list[str],
    candidate_mask_ids: list[str],
    meta_by_mask: dict[str, dict[str, Any]],
    threshold: float,
) -> float | None:
    all_gt = {
        str(meta_by_mask.get(mask_id, {}).get("diagnostic_gt_instance"))
        for mask_id in candidate_mask_ids
        if str(meta_by_mask.get(mask_id, {}).get("diagnostic_gt_instance") or "")
    }
    if not all_gt:
        return None
    represented: set[str] = set()
    for mask_id in selected_mask_ids:
        row = meta_by_mask.get(mask_id, {})
        gt = str(row.get("diagnostic_gt_instance") or "")
        if gt and parse_float(row.get("diagnostic_gt_purity"), 0.0) >= threshold:
            represented.add(gt)
    return float(len(represented) / max(len(all_gt), 1))


def _summarize_variant(
    variant: str,
    selected: list[str],
    mask_ids: list[str],
    universe: set[str],
    components_by_mask: dict[str, set[str]],
    meta_by_mask: dict[str, dict[str, Any]],
    component_count_key: str,
    progress: list[dict[str, Any]],
) -> dict[str, Any]:
    selected_components: set[str] = set()
    raw_underseg = [_underseg_proxy(meta_by_mask.get(mask_id, {}), component_count_key) for mask_id in mask_ids]
    selected_underseg = [_underseg_proxy(meta_by_mask.get(mask_id, {}), component_count_key) for mask_id in selected]
    for mask_id in selected:
        selected_components |= components_by_mask.get(mask_id, set()) & universe
    component_counts = [len(components_by_mask.get(mask_id, set()) & universe) for mask_id in selected]
    multi = sum(1 for value in component_counts if value > 1)
    single = sum(1 for value in component_counts if value == 1)
    unknown = sum(1 for value in component_counts if value == 0)
    selected_count = len(selected)
    return {
        "variant": variant,
        "selected_observation_count": int(selected_count),
        "selected_observation_ratio": float(selected_count / max(len(mask_ids), 1)),
        "component_coverage": float(len(selected_components) / max(len(universe), 1)),
        "carrier_coverage": float(len(selected_components) / max(len(universe), 1)),
        "mean_components_per_selected_mask": _mean([float(value) for value in component_counts]),
        "multi_component_selected_ratio": float(multi / max(selected_count, 1)),
        "single_component_selected_ratio": float(single / max(selected_count, 1)),
        "redundancy_rate": float(
            (sum(component_counts) - len(selected_components)) / max(sum(component_counts), 1)
        ),
        "underseg_selected_rate": float(sum(selected_underseg) / max(len(selected_underseg), 1)),
        "raw_underseg_rate": float(sum(raw_underseg) / max(len(raw_underseg), 1)),
        "same_frame_conflict_rate": _same_frame_conflict_rate(selected, components_by_mask, meta_by_mask),
        "unknown_or_outlier_observation_rate": float(unknown / max(selected_count, 1)),
        "GT_object_has_representative@0.25": _diagnostic_gt_representative(selected, mask_ids, meta_by_mask, 0.25),
        "GT_object_has_representative@0.50": _diagnostic_gt_representative(selected, mask_ids, meta_by_mask, 0.50),
        "representative_observation_purity": _mean(
            [
                parse_float(meta_by_mask.get(mask_id, {}).get("diagnostic_gt_purity"), 0.0)
                for mask_id in selected
                if str(meta_by_mask.get(mask_id, {}).get("diagnostic_gt_purity", "")) != ""
            ]
        ),
        "representative_observation_completeness_proxy": float(len(selected_components) / max(len(universe), 1)),
        "false_representative_rate": float(
            sum(1 for mask_id in selected if parse_float(meta_by_mask.get(mask_id, {}).get("diagnostic_gt_purity"), 1.0) < 0.25)
            / max(selected_count, 1)
        ),
        "coverage_progress_final": progress[-1]["component_coverage"] if progress else 0.0,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }


def build_representative_observations(
    support_rows_path: str | Path = "outputs/audit/v53_mask_component_support_tau005/mask_component_support_rows.csv",
    mask_summary_path: str | Path = "outputs/audit/v53_mask_component_support_tau005/mask_summary_rows.csv",
    chunk_component_rows_path: str | Path = "outputs/audit/v53_chunk_universe/chunk_component_rows.csv",
    chunk_mask_rows_path: str | Path = "outputs/audit/v53_chunk_universe/chunk_mask_rows.csv",
    support_variant: str = "R0_visible_tau0.05",
    max_selected_ratio: float = 0.60,
    gate_variant: str = "K5_coverage_redundancy_cannot_link",
) -> dict[str, Any]:
    support_rows_path = _project(support_rows_path)
    components_by_mask, counts_by_mask = _load_support_sets(support_rows_path, support_variant)
    mask_summary_rows = read_csv(_project(mask_summary_path))
    chunk_component_rows = read_csv(_project(chunk_component_rows_path))
    chunk_mask_rows = read_csv(_project(chunk_mask_rows_path))
    meta_by_mask = _mask_meta(mask_summary_rows)
    component_count_key = f"{support_variant}_component_count"

    universe_by_chunk: dict[str, set[str]] = defaultdict(set)
    for row in chunk_component_rows:
        universe_by_chunk[str(row.get("chunk_id"))].add(str(row.get("component_id")))
    masks_by_chunk: dict[str, list[str]] = defaultdict(list)
    for row in chunk_mask_rows:
        masks_by_chunk[str(row.get("chunk_id"))].append(str(row.get("mask_observation_id")))

    variant_settings = [
        ("K0_all_masks_no_selection", "all", {}),
        ("K1_random_masks_control", "random", {}),
        ("K2_area_largest_control", "area", {}),
        ("K3_component_coverage_set_cover", "greedy", {}),
        ("K4_coverage_redundancy_penalty", "greedy", {"redundancy_penalty": 0.05}),
        ("K5_coverage_redundancy_cannot_link", "greedy", {"redundancy_penalty": 0.05, "cannot_link_penalty": 0.50}),
        ("K6_K5_semantic_underseg_guard_proxy", "greedy", {"redundancy_penalty": 0.05, "cannot_link_penalty": 0.50, "underseg_penalty": 100.0}),
        ("K7_K6_reprojection_precheck_proxy", "greedy", {"redundancy_penalty": 0.05, "cannot_link_penalty": 0.50, "underseg_penalty": 100.0}),
        (
            "K8_underseg_capped_partial_repair",
            "greedy",
            {"redundancy_penalty": 0.05, "cannot_link_penalty": 0.50, "underseg_penalty": 100.0, "underseg_rate_cap": "raw"},
        ),
    ]
    representative_rows: list[dict[str, Any]] = []
    progress_rows: list[dict[str, Any]] = []
    per_chunk_variant_rows: list[dict[str, Any]] = []

    for chunk_id, mask_ids in sorted(masks_by_chunk.items()):
        universe = universe_by_chunk.get(chunk_id, set())
        max_selected = max(1, int(max_selected_ratio * len(mask_ids)))
        for variant, mode, kwargs in variant_settings:
            progress: list[dict[str, Any]] = []
            if mode == "all":
                selected = list(mask_ids)
            elif mode == "random":
                selected = sorted(mask_ids, key=lambda mask_id: _stable_score(mask_id))[:max_selected]
            elif mode == "area":
                selected = sorted(
                    mask_ids,
                    key=lambda mask_id: -parse_int(meta_by_mask.get(mask_id, {}).get("mask_area")),
                )[:max_selected]
            else:
                raw_underseg_rate = sum(
                    1 for mask_id in mask_ids if _underseg_proxy(meta_by_mask.get(mask_id, {}), component_count_key)
                ) / max(len(mask_ids), 1)
                cap_value = raw_underseg_rate if kwargs.get("underseg_rate_cap") == "raw" else kwargs.get("underseg_rate_cap")
                selected, progress = _select_greedy(
                    mask_ids=mask_ids,
                    universe=universe,
                    components_by_mask=components_by_mask,
                    meta_by_mask=meta_by_mask,
                    max_selected=max_selected,
                    component_count_key=component_count_key,
                    redundancy_penalty=float(kwargs.get("redundancy_penalty", 0.0)),
                    underseg_penalty=float(kwargs.get("underseg_penalty", 0.0)),
                    cannot_link_penalty=float(kwargs.get("cannot_link_penalty", 0.0)),
                    underseg_rate_cap=None if cap_value is None else float(cap_value),
                )
            summary = _summarize_variant(
                variant=variant,
                selected=selected,
                mask_ids=mask_ids,
                universe=universe,
                components_by_mask=components_by_mask,
                meta_by_mask=meta_by_mask,
                component_count_key=component_count_key,
                progress=progress,
            )
            summary["chunk_id"] = chunk_id
            summary["scene"] = chunk_id.split(":")[0]
            per_chunk_variant_rows.append(summary)
            for rank, mask_id in enumerate(selected, start=1):
                row = meta_by_mask.get(mask_id, {})
                representative_rows.append(
                    {
                        "variant": variant,
                        "chunk_id": chunk_id,
                        "scene": row.get("scene"),
                        "frame_id": row.get("frame_id"),
                        "mask_id": row.get("mask_id"),
                        "mask_observation_id": mask_id,
                        "selected_rank": int(rank),
                        "component_count": int(len(components_by_mask.get(mask_id, set()) & universe)),
                        "support_carrier_count": int(sum(counts_by_mask.get(mask_id, Counter()).values())),
                        "underseg_proxy": _underseg_proxy(row, component_count_key),
                        "diagnostic_gt_instance": row.get("diagnostic_gt_instance", ""),
                        "diagnostic_gt_purity": row.get("diagnostic_gt_purity", ""),
                        "uses_gt_for_prediction": False,
                        "uses_gt_for_diagnostic_labels": True,
                    }
                )
            for progress_row in progress:
                progress_rows.append({"variant": variant, "chunk_id": chunk_id, **progress_row})

    variant_summaries: list[dict[str, Any]] = []
    for variant, _mode, _kwargs in variant_settings:
        rows = [row for row in per_chunk_variant_rows if row["variant"] == variant]
        variant_summaries.append(
            {
                "variant": variant,
                "selected_observation_count": int(sum(parse_int(row.get("selected_observation_count")) for row in rows)),
                "selected_observation_ratio": _mean([parse_float(row.get("selected_observation_ratio")) for row in rows]),
                "component_coverage": _mean([parse_float(row.get("component_coverage")) for row in rows]),
                "carrier_coverage": _mean([parse_float(row.get("carrier_coverage")) for row in rows]),
                "mean_components_per_selected_mask": _mean(
                    [
                        parse_float(row.get("mean_components_per_selected_mask"))
                        for row in rows
                        if str(row.get("mean_components_per_selected_mask", "")) != ""
                    ]
                ),
                "multi_component_selected_ratio": _mean([parse_float(row.get("multi_component_selected_ratio")) for row in rows]),
                "single_component_selected_ratio": _mean([parse_float(row.get("single_component_selected_ratio")) for row in rows]),
                "redundancy_rate": _mean([parse_float(row.get("redundancy_rate")) for row in rows]),
                "underseg_selected_rate": _mean([parse_float(row.get("underseg_selected_rate")) for row in rows]),
                "raw_underseg_rate": _mean([parse_float(row.get("raw_underseg_rate")) for row in rows]),
                "same_frame_conflict_rate": _mean([parse_float(row.get("same_frame_conflict_rate")) for row in rows]),
                "unknown_or_outlier_observation_rate": _mean([parse_float(row.get("unknown_or_outlier_observation_rate")) for row in rows]),
                "GT_object_has_representative@0.25": _mean(
                    [
                        parse_float(row.get("GT_object_has_representative@0.25"))
                        for row in rows
                        if str(row.get("GT_object_has_representative@0.25", "")) != ""
                    ]
                ),
                "GT_object_has_representative@0.50": _mean(
                    [
                        parse_float(row.get("GT_object_has_representative@0.50"))
                        for row in rows
                        if str(row.get("GT_object_has_representative@0.50", "")) != ""
                    ]
                ),
                "representative_observation_purity": _mean(
                    [
                        parse_float(row.get("representative_observation_purity"))
                        for row in rows
                        if str(row.get("representative_observation_purity", "")) != ""
                    ]
                ),
                "representative_observation_completeness_proxy": _mean(
                    [parse_float(row.get("representative_observation_completeness_proxy")) for row in rows]
                ),
                "false_representative_rate": _mean([parse_float(row.get("false_representative_rate")) for row in rows]),
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic_labels": True,
            }
        )

    summary_by_variant = {row["variant"]: row for row in variant_summaries}
    gate_row = summary_by_variant[gate_variant]
    gate = {
        "component_coverage_ge_0.75": float(gate_row["component_coverage"] or 0.0) >= 0.75,
        "selected_observation_ratio_le_0.60": float(gate_row["selected_observation_ratio"] or 1.0) <= 0.60,
        "same_frame_conflict_rate_le_0.10": float(gate_row["same_frame_conflict_rate"] or 0.0) <= 0.10,
        "underseg_selected_rate_le_raw_underseg_rate": float(gate_row["underseg_selected_rate"] or 0.0)
        <= float(gate_row["raw_underseg_rate"] or 0.0),
        "GT_object_has_representative_0.25_ge_0.60_diagnostic": float(gate_row["GT_object_has_representative@0.25"] or 0.0)
        >= 0.60,
    }
    gate["pass"] = bool(all(gate.values()))
    representative_mode = "partial-heavy"
    if float(gate_row["multi_component_selected_ratio"] or 0.0) >= 0.50:
        representative_mode = "multi-component-heavy"

    summary = {
        "phase": "v53_representative_observations",
        "created_at": utc_now(),
        "support_rows_path": str(support_rows_path),
        "support_variant": support_variant,
        "gate_variant": gate_variant,
        "representative_mode": representative_mode,
        "variant_summaries": variant_summaries,
        "gate": gate,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }
    return {
        "summary": summary,
        "representative_mask_rows": representative_rows,
        "coverage_progress_rows": progress_rows,
        "per_chunk_variant_rows": per_chunk_variant_rows,
    }


def _write_visualizations(output_root: Path, vis_root: Path, payload: dict[str, Any]) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        write_json(output_root / "visualization_error.json", {"error": repr(exc)})
        return [{"path": str(output_root / "visualization_error.json"), "status": "matplotlib_unavailable"}]
    vis_root.mkdir(parents=True, exist_ok=True)
    gate_variant = payload["summary"]["gate_variant"]
    selected_by_chunk: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in payload["representative_mask_rows"]:
        if row["variant"] == gate_variant:
            selected_by_chunk[str(row["chunk_id"])].append(row)
    for chunk_id, rows in sorted(selected_by_chunk.items()):
        scene = chunk_id.split(":")[0]
        selected = rows[:6]
        if selected:
            fig, axes = plt.subplots(1, len(selected), figsize=(3 * len(selected), 3))
            if len(selected) == 1:
                axes = [axes]
            for ax, row in zip(axes, selected):
                frame_id = parse_int(row.get("frame_id"))
                mask_id = parse_int(row.get("mask_id"))
                label = load_mask_label(scene, frame_id)
                image = np.zeros((64, 64), dtype=np.uint8) if label is None else (label == mask_id).astype(np.uint8)
                ax.imshow(image, cmap="gray", interpolation="nearest")
                ax.set_title(f"f{frame_id} m{mask_id} c{row.get('component_count')}")
                ax.axis("off")
            path = vis_root / f"representative_masks_panel_{scene}_{chunk_id.split(':')[-1]}.png"
            fig.tight_layout()
            fig.savefig(path, dpi=140)
            plt.close(fig)
            manifest.append({"path": str(path), "kind": "representative_masks_panel", "scene": scene, "chunk_id": chunk_id})
        progress = [row for row in payload["coverage_progress_rows"] if row["variant"] == gate_variant and row["chunk_id"] == chunk_id]
        if progress:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot([parse_int(row["step"]) for row in progress], [parse_float(row["component_coverage"]) for row in progress], marker="o")
            ax.set_xlabel("selected mask count")
            ax.set_ylabel("component coverage")
            ax.set_ylim(0.0, 1.0)
            ax.set_title(f"{scene} coverage progress {chunk_id.split(':')[-1]}")
            path = vis_root / f"coverage_progress_curve_{scene}_{chunk_id.split(':')[-1]}.png"
            fig.tight_layout()
            fig.savefig(path, dpi=140)
            plt.close(fig)
            manifest.append({"path": str(path), "kind": "coverage_progress_curve", "scene": scene, "chunk_id": chunk_id})

        fig, ax = plt.subplots(figsize=(7, 4))
        counts = [parse_int(row.get("component_count")) for row in rows]
        ax.bar(range(len(counts)), counts)
        ax.set_xlabel("selected rank")
        ax.set_ylabel("component count")
        ax.set_title(f"{scene} component coverage map proxy")
        path = vis_root / f"component_coverage_map_{scene}_{chunk_id.split(':')[-1]}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        manifest.append({"path": str(path), "kind": "component_coverage_map", "scene": scene, "chunk_id": chunk_id})
    return manifest


def write_representative_observations(
    output_root: str | Path,
    payload: dict[str, Any],
    visualization_root: str | Path = "outputs/audit/v53_visualizations/local_objectlets",
) -> None:
    out = _project(output_root)
    vis = _project(visualization_root)
    write_json(out / "representative_summary.json", payload["summary"])
    write_csv(out / "representative_mask_rows.csv", payload["representative_mask_rows"])
    write_csv(out / "coverage_progress_rows.csv", payload["coverage_progress_rows"])
    write_csv(out / "representative_variant_rows.csv", payload["per_chunk_variant_rows"])
    manifest = _write_visualizations(out, vis, payload)
    write_json(out / "visualization_manifest.json", {"phase": "v53_representative_observations", "files": manifest})


__all__ = ["build_representative_observations", "write_representative_observations"]
