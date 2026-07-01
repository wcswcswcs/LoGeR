from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .v47_common import ROOT, load_mask_label_from_root, parse_float, parse_int, read_csv, utc_now, write_csv, write_json
from .v53_mask_component_support import _build_components, _carrier_global_id, _collect_support, _is_visible_row


def _project(path: str | Path) -> Path:
    path_obj = Path(path)
    return path_obj if path_obj.is_absolute() else ROOT / path_obj


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _quantile(values: list[float], q: float) -> float | None:
    return float(np.quantile(values, q)) if values else None


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


def _mask_key(scene: str, frame_id: int, mask_id: int) -> tuple[str, int, int]:
    return scene, int(frame_id), int(mask_id)


def _repeated_support_signature_candidates(
    *,
    support_rows: list[dict[str, Any]],
    representative_by_mask: dict[str, dict[str, Any]],
    seen_component_sets: set[tuple[str, tuple[str, ...]]],
    start_index: int,
    min_frames: int,
    min_components: int,
    min_w_visible: float,
    max_components: int,
    max_groups_per_scene: int,
) -> list[dict[str, Any]]:
    best_by_component_frame: dict[tuple[str, str, int], dict[str, Any]] = {}
    support_totals: Counter[str] = Counter()
    for row in support_rows:
        mask_observation_id = str(row.get("mask_observation_id"))
        if mask_observation_id not in representative_by_mask:
            continue
        w_visible = parse_float(row.get("W_visible"))
        if w_visible < float(min_w_visible):
            continue
        component_id = str(row.get("component_id"))
        scene = str(row.get("scene"))
        frame_id = parse_int(row.get("frame_id"))
        support_count = parse_int(row.get("support_count"))
        support_totals[component_id] += support_count
        key = (scene, component_id, frame_id)
        old = best_by_component_frame.get(key)
        if old is None:
            best_by_component_frame[key] = row
            continue
        old_key = (parse_float(old.get("W_visible")), parse_int(old.get("support_count")), str(old.get("mask_observation_id")))
        new_key = (w_visible, support_count, mask_observation_id)
        if new_key > old_key:
            best_by_component_frame[key] = row

    frames_by_scene_component: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for (scene, component_id, _frame_id), row in best_by_component_frame.items():
        frames_by_scene_component[(scene, component_id)].append(row)

    components_by_signature: dict[tuple[str, tuple[str, ...]], list[str]] = defaultdict(list)
    for (scene, component_id), rows in frames_by_scene_component.items():
        rows = sorted(rows, key=lambda row: parse_int(row.get("frame_id")))
        signature = tuple(str(row.get("mask_observation_id")) for row in rows)
        if len(signature) < int(min_frames):
            continue
        components_by_signature[(scene, signature)].append(component_id)

    groups_by_scene: dict[str, list[tuple[tuple[str, ...], list[str]]]] = defaultdict(list)
    for (scene, signature), components in components_by_signature.items():
        unique_components = sorted(set(components), key=lambda component: (-support_totals[component], component))
        if len(unique_components) < int(min_components):
            continue
        if int(max_components) > 0:
            unique_components = unique_components[: int(max_components)]
        if len(unique_components) < int(min_components):
            continue
        groups_by_scene[scene].append((signature, unique_components))

    candidates: list[dict[str, Any]] = []
    next_index = int(start_index)
    for scene, groups in sorted(groups_by_scene.items()):
        groups = sorted(
            groups,
            key=lambda item: (
                -len(item[0]),
                -len(item[1]),
                -sum(support_totals[component] for component in item[1]),
                item[0][0],
            ),
        )[: int(max_groups_per_scene)]
        for signature, components in groups:
            key = (scene, tuple(components))
            if key in seen_component_sets:
                continue
            source_mask = signature[0]
            representative = representative_by_mask.get(source_mask, {})
            if not representative:
                continue
            seen_component_sets.add(key)
            candidates.append(
                {
                    "candidate_id": f"cand{next_index:05d}",
                    "candidate_source": "R5_repeated_support_signature",
                    "scene": scene,
                    "chunk_id": representative.get("chunk_id"),
                    "source_mask_observation_id": source_mask,
                    "source_frame_id": parse_int(representative.get("frame_id")),
                    "source_mask_id": parse_int(representative.get("mask_id")),
                    "component_ids": components,
                    "component_count": len(components),
                    "source_diagnostic_gt_instance": representative.get("diagnostic_gt_instance", ""),
                    "repeated_support_signature_len": len(signature),
                    "repeated_support_total_support": int(sum(support_totals[component] for component in components)),
                    "uses_gt_for_prediction": False,
                    "uses_gt_for_diagnostic_labels": True,
                }
            )
            next_index += 1
    return candidates


def build_reprojection_ledger(
    carrier_table_path: str | Path = "outputs/audit/v47_observation_tables_metricfix/carrier_observation_table.csv",
    mask_table_path: str | Path = "outputs/audit/v47_observation_tables_metricfix/mask_observation_table.csv",
    support_rows_path: str | Path = "outputs/audit/v53_mask_component_support_tau005/mask_component_support_rows.csv",
    representative_rows_path: str | Path = "outputs/audit/v53_representative_observations_k8_underseg_cap_fixed/representative_mask_rows.csv",
    support_variant: str = "R0_visible_tau0.05",
    representative_variant: str = "K8_underseg_capped_partial_repair",
    max_union_unique_carriers: int = 32,
    min_visibility_prob: float = 0.5,
    min_confidence: float = 0.5,
    min_visible_carriers: int = 3,
    max_candidates: int = 240,
    max_components_per_candidate: int = 0,
    skip_no_related_measurement: bool = False,
    max_candidate_conflict_rate: float | None = None,
    include_repeated_support_candidates: bool = False,
    repeated_support_min_frames: int = 4,
    repeated_support_min_components: int = 2,
    repeated_support_min_w_visible: float = 0.50,
    repeated_support_max_components: int = 128,
    repeated_support_max_groups_per_scene: int = 80,
    deduplicate_component_sets: bool = True,
) -> dict[str, Any]:
    carrier_rows = read_csv(_project(carrier_table_path))
    mask_rows = read_csv(_project(mask_table_path))
    representative_rows = [
        row for row in read_csv(_project(representative_rows_path)) if str(row.get("variant")) == representative_variant
    ]
    support_rows = [row for row in read_csv(_project(support_rows_path)) if str(row.get("variant")) == support_variant]
    components_by_mask, counts_by_mask = _load_support_sets(_project(support_rows_path), support_variant)
    component_payload = _build_components(
        carrier_rows=carrier_rows,
        mask_rows=mask_rows,
        max_union_unique_carriers=max_union_unique_carriers,
        min_visibility_prob=min_visibility_prob,
        min_confidence=min_confidence,
    )
    support_payload = _collect_support(
        visible_rows=component_payload["visible_rows"],
        mask_rows=mask_rows,
        component_by_carrier=component_payload["component_by_carrier"],
    )
    component_by_carrier: dict[str, str] = component_payload["component_by_carrier"]
    support_by_mask: dict[str, Counter[str]] = support_payload["support_by_mask"]
    mask_by_key = {
        _mask_key(str(row.get("scene")), parse_int(row.get("frame_id")), parse_int(row.get("mask_id"))): row
        for row in mask_rows
    }
    mask_gt = {str(row.get("mask_observation_id")): str(row.get("diagnostic_gt_instance") or "") for row in mask_rows}
    representative_by_mask = {str(row.get("mask_observation_id")): row for row in representative_rows}

    visible_by_scene_frame_component: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    frames_by_scene: dict[str, set[int]] = defaultdict(set)
    for row in carrier_rows:
        if not _is_visible_row(row, min_visibility_prob=min_visibility_prob, min_confidence=min_confidence):
            continue
        scene = str(row.get("scene"))
        frame_id = parse_int(row.get("frame_id"))
        component_id = component_by_carrier.get(_carrier_global_id(row))
        if not component_id:
            continue
        frames_by_scene[scene].add(frame_id)
        visible_by_scene_frame_component[(scene, frame_id, component_id)].append(row)

    candidates: list[dict[str, Any]] = []
    seen_component_sets: set[tuple[str, tuple[str, ...]]] = set()
    for row in representative_rows:
        mask_id = str(row.get("mask_observation_id"))
        scene = str(row.get("scene"))
        if int(max_components_per_candidate) > 0:
            components = [
                component
                for component, _count in counts_by_mask.get(mask_id, Counter()).most_common(int(max_components_per_candidate))
            ]
        else:
            components = sorted(components_by_mask.get(mask_id, set()))
        if not components:
            continue
        key = (scene, tuple(components))
        if deduplicate_component_sets and key in seen_component_sets:
            continue
        if deduplicate_component_sets:
            seen_component_sets.add(key)
        candidates.append(
            {
                "candidate_id": f"cand{len(candidates):05d}",
                "candidate_source": "R2_partial_top_component_subset"
                if int(max_components_per_candidate) > 0
                else "R0_single_representative_mask",
                "scene": scene,
                "chunk_id": row.get("chunk_id"),
                "source_mask_observation_id": mask_id,
                "source_frame_id": parse_int(row.get("frame_id")),
                "source_mask_id": parse_int(row.get("mask_id")),
                "component_ids": components,
                "component_count": len(components),
                "source_diagnostic_gt_instance": row.get("diagnostic_gt_instance", ""),
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic_labels": True,
            }
        )
        if len(candidates) >= int(max_candidates):
            break

    repeated_support_candidate_count = 0
    if include_repeated_support_candidates:
        extra_candidates = _repeated_support_signature_candidates(
            support_rows=support_rows,
            representative_by_mask=representative_by_mask,
            seen_component_sets=seen_component_sets,
            start_index=len(candidates),
            min_frames=repeated_support_min_frames,
            min_components=repeated_support_min_components,
            min_w_visible=repeated_support_min_w_visible,
            max_components=repeated_support_max_components,
            max_groups_per_scene=repeated_support_max_groups_per_scene,
        )
        repeated_support_candidate_count = len(extra_candidates)
        candidates.extend(extra_candidates)

    ledger_rows: list[dict[str, Any]] = []
    sparse_no_related_frame_count = 0
    for candidate in candidates:
        scene = str(candidate["scene"])
        component_ids = set(candidate["component_ids"])
        for frame_id in sorted(frames_by_scene.get(scene, set())):
            visible_rows: list[dict[str, Any]] = []
            for component_id in component_ids:
                visible_rows.extend(visible_by_scene_frame_component.get((scene, frame_id, component_id), []))
            visible_count = len(visible_rows)
            if visible_count < int(min_visible_carriers):
                continue
            observed_mask_counts: Counter[int] = Counter()
            related_mask_counts: Counter[str] = Counter()
            inside_any_count = 0
            related_count = 0
            for row in visible_rows:
                observed_mask_id = parse_int(row.get("observed_mask_id"))
                if observed_mask_id <= 0:
                    continue
                inside_any_count += 1
                observed_mask_counts[observed_mask_id] += 1
                mask_row = mask_by_key.get(_mask_key(scene, frame_id, observed_mask_id))
                if not mask_row:
                    continue
                mask_observation_id = str(mask_row.get("mask_observation_id"))
                support_components = set(support_by_mask.get(mask_observation_id, Counter()))
                if support_components & component_ids:
                    related_mask_counts[mask_observation_id] += 1
                    related_count += 1
            best_mask_observation_id = ""
            best_count = 0
            if related_mask_counts:
                best_mask_observation_id, best_count = related_mask_counts.most_common(1)[0]
            if skip_no_related_measurement and not related_mask_counts:
                sparse_no_related_frame_count += 1
                continue
            best_mask_total = int(sum(support_by_mask.get(best_mask_observation_id, Counter()).values())) if best_mask_observation_id else 0
            inside_best_mask_ratio = float(best_count / max(visible_count, 1))
            inside_any_mask_ratio = float(inside_any_count / max(visible_count, 1))
            outside_all_related_masks_ratio = float((visible_count - related_count) / max(visible_count, 1))
            mask_explained_ratio = float(best_count / max(best_mask_total, 1)) if best_mask_observation_id else 0.0
            competing_related = sum(1 for count in related_mask_counts.values() if count / max(visible_count, 1) >= 0.10)
            same_frame_exclusion_violation = bool(competing_related > 1 and inside_best_mask_ratio < 0.75)
            success = bool(inside_best_mask_ratio >= 0.50 and outside_all_related_masks_ratio <= 0.35 and not same_frame_exclusion_violation)
            src_gt = str(candidate.get("source_diagnostic_gt_instance") or "")
            best_gt = mask_gt.get(best_mask_observation_id, "")
            ledger_rows.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "candidate_source": candidate["candidate_source"],
                    "scene": scene,
                    "chunk_id": candidate["chunk_id"],
                    "target_frame_id": int(frame_id),
                    "visible_carrier_count": int(visible_count),
                    "inside_best_mask_ratio": inside_best_mask_ratio,
                    "inside_any_mask_ratio": inside_any_mask_ratio,
                    "outside_all_related_masks_ratio": outside_all_related_masks_ratio,
                    "mask_explained_ratio": mask_explained_ratio,
                    "same_frame_exclusion_violation": same_frame_exclusion_violation,
                    "occlusion_uncertain_count": 0,
                    "best_mask_observation_id": best_mask_observation_id,
                    "best_mask_related_carrier_count": int(best_count),
                    "related_mask_count": int(len(related_mask_counts)),
                    "reprojection_success": success,
                    "diagnostic_source_gt": src_gt,
                    "diagnostic_best_gt": best_gt,
                    "diagnostic_success_same_gt": bool(success and src_gt and best_gt and src_gt == best_gt),
                    "uses_gt_for_prediction": False,
                    "uses_gt_for_diagnostic_labels": True,
                }
            )

    candidate_success: Counter[str] = Counter()
    candidate_rows_out: list[dict[str, Any]] = []
    rows_by_candidate: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ledger_rows:
        rows_by_candidate[str(row["candidate_id"])].append(row)
        if row["reprojection_success"]:
            candidate_success[str(row["candidate_id"])] += 1
    for candidate in candidates:
        rows = rows_by_candidate.get(str(candidate["candidate_id"]), [])
        conflict_rate = _mean([1.0 if row.get("same_frame_exclusion_violation") else 0.0 for row in rows])
        candidate_rows_out.append(
            {
                **{key: value for key, value in candidate.items() if key != "component_ids"},
                "component_ids": candidate["component_ids"],
                "ledger_row_count": len(rows),
                "candidate_success_rate": float(candidate_success[candidate["candidate_id"]] / max(len(rows), 1)),
                "outside_all_related_masks_ratio_mean": _mean(
                    [parse_float(row.get("outside_all_related_masks_ratio")) for row in rows]
                ),
                "same_frame_exclusion_violation_rate": conflict_rate,
                "candidate_conflict_veto_pass": _passes_candidate_conflict_veto(
                    conflict_rate=conflict_rate,
                    ledger_row_count=len(rows),
                    max_candidate_conflict_rate=max_candidate_conflict_rate,
                ),
            }
        )

    pre_veto_candidate_count = len(candidate_rows_out)
    pre_veto_ledger_row_count = len(ledger_rows)
    vetoed_candidate_count = 0
    if max_candidate_conflict_rate is not None and float(max_candidate_conflict_rate) >= 0.0:
        keep_candidate_ids = {
            str(row["candidate_id"]) for row in candidate_rows_out if row.get("candidate_conflict_veto_pass")
        }
        vetoed_candidate_count = len(candidate_rows_out) - len(keep_candidate_ids)
        candidate_rows_out = [row for row in candidate_rows_out if str(row["candidate_id"]) in keep_candidate_ids]
        ledger_rows = [row for row in ledger_rows if str(row["candidate_id"]) in keep_candidate_ids]
        candidates = [candidate for candidate in candidates if str(candidate["candidate_id"]) in keep_candidate_ids]

    outside_values = [parse_float(row.get("outside_all_related_masks_ratio")) for row in ledger_rows]
    inside_values = [parse_float(row.get("inside_best_mask_ratio")) for row in ledger_rows]
    mask_explained_values = [parse_float(row.get("mask_explained_ratio")) for row in ledger_rows]
    success_rows = [row for row in ledger_rows if row.get("reprojection_success")]
    success_with_gt = [
        row
        for row in success_rows
        if str(row.get("diagnostic_source_gt") or "") and str(row.get("diagnostic_best_gt") or "")
    ]
    same_gt_precision = (
        sum(1 for row in success_with_gt if row.get("diagnostic_success_same_gt")) / len(success_with_gt)
        if success_with_gt
        else None
    )
    summary = {
        "phase": "v53_reprojection_ledger",
        "created_at": utc_now(),
        "support_variant": support_variant,
        "representative_variant": representative_variant,
        "max_components_per_candidate": int(max_components_per_candidate),
        "include_repeated_support_candidates": bool(include_repeated_support_candidates),
        "repeated_support_candidate_count": int(repeated_support_candidate_count),
        "repeated_support_min_frames": int(repeated_support_min_frames),
        "repeated_support_min_components": int(repeated_support_min_components),
        "repeated_support_min_w_visible": float(repeated_support_min_w_visible),
        "repeated_support_max_components": int(repeated_support_max_components),
        "repeated_support_max_groups_per_scene": int(repeated_support_max_groups_per_scene),
        "skip_no_related_measurement": bool(skip_no_related_measurement),
        "max_candidate_conflict_rate": max_candidate_conflict_rate,
        "sparse_no_related_frame_count": int(sparse_no_related_frame_count),
        "pre_veto_candidate_count": int(pre_veto_candidate_count),
        "pre_veto_ledger_row_count": int(pre_veto_ledger_row_count),
        "vetoed_candidate_count": int(vetoed_candidate_count),
        "candidate_count": len(candidates),
        "ledger_row_count": len(ledger_rows),
        "inside_best_mask_ratio_mean": _mean(inside_values),
        "inside_best_mask_ratio_p10": _quantile(inside_values, 0.10),
        "outside_all_related_masks_ratio_mean": _mean(outside_values),
        "outside_all_related_masks_ratio_p90": _quantile(outside_values, 0.90),
        "mask_explained_ratio_mean": _mean(mask_explained_values),
        "reprojection_success_rate": float(len(success_rows) / max(len(ledger_rows), 1)),
        "same_frame_exclusion_violation_rate": float(
            sum(1 for row in ledger_rows if row.get("same_frame_exclusion_violation")) / max(len(ledger_rows), 1)
        ),
        "underseg_signal_count": int(sum(1 for row in ledger_rows if parse_int(row.get("related_mask_count")) > 1)),
        "candidate_source_breakdown": dict(Counter(str(row["candidate_source"]) for row in candidates)),
        "reprojection_success_same_GT_precision": same_gt_precision,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }
    gate = {
        "reprojection_success_rate_ge_0.60": float(summary["reprojection_success_rate"]) >= 0.60,
        "outside_all_related_masks_ratio_mean_le_0.35": float(summary["outside_all_related_masks_ratio_mean"] or 1.0)
        <= 0.35,
        "same_frame_exclusion_violation_rate_le_0.05": float(summary["same_frame_exclusion_violation_rate"]) <= 0.05,
        "reprojection_success_same_GT_precision_ge_0.80_diagnostic": float(same_gt_precision or 0.0) >= 0.80,
    }
    gate["pass"] = bool(all(gate.values()))
    summary["gate"] = gate
    return {"summary": summary, "reprojection_ledger_rows": ledger_rows, "candidate_rows": candidate_rows_out}


def _passes_candidate_conflict_veto(
    conflict_rate: float | None,
    ledger_row_count: int,
    max_candidate_conflict_rate: float | None,
) -> bool:
    if ledger_row_count <= 0:
        return False
    if max_candidate_conflict_rate is None or float(max_candidate_conflict_rate) < 0.0:
        return True
    return float(conflict_rate or 0.0) <= float(max_candidate_conflict_rate)


def _write_visualizations(output_root: Path, vis_root: Path, payload: dict[str, Any], mask_root: str | Path | None = None) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        write_json(output_root / "visualization_error.json", {"error": repr(exc)})
        return [{"path": str(output_root / "visualization_error.json"), "status": "matplotlib_unavailable"}]
    vis_root.mkdir(parents=True, exist_ok=True)
    rows_by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in payload["reprojection_ledger_rows"]:
        rows_by_scene[str(row["scene"])].append(row)
    for scene, rows in sorted(rows_by_scene.items()):
        sample = rows[:8]
        fig, axes = plt.subplots(1, max(1, len(sample)), figsize=(3 * max(1, len(sample)), 3))
        if len(sample) == 1:
            axes = [axes]
        for ax, row in zip(axes, sample):
            frame_id = parse_int(row.get("target_frame_id"))
            best = str(row.get("best_mask_observation_id") or "")
            mask_id = parse_int(best.split(":")[-1]) if best else 0
            label = load_mask_label_from_root(scene, frame_id, mask_root)
            image = np.zeros((64, 64), dtype=np.uint8) if label is None or mask_id <= 0 else (label == mask_id).astype(np.uint8)
            ax.imshow(image, cmap="gray", interpolation="nearest")
            ax.set_title(f"f{frame_id} in{parse_float(row.get('inside_best_mask_ratio')):.2f} out{parse_float(row.get('outside_all_related_masks_ratio')):.2f}")
            ax.axis("off")
        path = vis_root / f"reprojection_panel_{scene}_sample.png"
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        manifest.append({"path": str(path), "kind": "reprojection_panel", "scene": scene})

        pivot: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            pivot[str(row["candidate_id"])].append(parse_float(row.get("outside_all_related_masks_ratio")))
        values = [np.mean(vals) for vals in pivot.values()]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(values, bins=min(40, max(5, len(values))), color="#ef4444")
        ax.set_xlabel("candidate outside residual mean")
        ax.set_ylabel("candidate count")
        ax.set_title(f"{scene} outside residual heatmap proxy")
        path = vis_root / f"outside_residual_heatmap_{scene}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        manifest.append({"path": str(path), "kind": "outside_residual_heatmap", "scene": scene})

        conflicts = [row for row in rows if row.get("same_frame_exclusion_violation")]
        if conflicts:
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.bar(range(len(conflicts[:20])), [parse_float(row.get("inside_best_mask_ratio")) for row in conflicts[:20]])
            ax.set_title(f"{scene} conflict frame gallery proxy")
            ax.set_xlabel("conflict row")
            ax.set_ylabel("inside best ratio")
            path = vis_root / f"conflict_frame_gallery_{scene}.png"
            fig.tight_layout()
            fig.savefig(path, dpi=140)
            plt.close(fig)
            manifest.append({"path": str(path), "kind": "conflict_frame_gallery", "scene": scene})
    return manifest


def write_reprojection_ledger(
    output_root: str | Path,
    payload: dict[str, Any],
    visualization_root: str | Path = "outputs/audit/v53_visualizations/reprojection",
    mask_root: str | Path | None = None,
) -> None:
    out = _project(output_root)
    vis = _project(visualization_root)
    write_json(out / "reprojection_summary.json", payload["summary"])
    write_csv(out / "reprojection_ledger_rows.csv", payload["reprojection_ledger_rows"])
    write_csv(out / "candidate_rows.csv", payload["candidate_rows"])
    manifest = _write_visualizations(out, vis, payload, mask_root=mask_root)
    write_json(out / "visualization_manifest.json", {"phase": "v53_reprojection_ledger", "files": manifest})


__all__ = ["build_reprojection_ledger", "write_reprojection_ledger"]
