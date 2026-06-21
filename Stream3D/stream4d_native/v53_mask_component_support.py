from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .v47_common import ROOT, UnionFind, load_mask_label, parse_bool, parse_float, parse_int, read_csv, utc_now, write_csv, write_json


MaskKey = tuple[str, int, int]
FrameComponentKey = tuple[str, int, str]


def _mask_key(row: dict[str, Any]) -> MaskKey:
    return str(row.get("scene")), parse_int(row.get("frame_id")), parse_int(row.get("mask_id"))


def _carrier_global_id(row: dict[str, Any]) -> str:
    return str(row.get("carrier_global_id") or f"{row.get('scene')}:{parse_int(row.get('carrier_id'))}")


def _project(path: str | Path) -> Path:
    path_obj = Path(path)
    return path_obj if path_obj.is_absolute() else ROOT / path_obj


def _is_visible_row(row: dict[str, Any], min_visibility_prob: float, min_confidence: float) -> bool:
    if not parse_bool(row.get("valid", True)) or not parse_bool(row.get("valid_uv", True)):
        return False
    if parse_float(row.get("confidence"), 0.0) < float(min_confidence):
        return False
    if "visibility_prob" in row and str(row.get("visibility_prob")) != "":
        return parse_float(row.get("visibility_prob"), 0.0) >= float(min_visibility_prob)
    return parse_bool(row.get("visible"))


def _component_entropy(counts: list[int]) -> float:
    total = float(sum(counts))
    if total <= 0.0:
        return 0.0
    probs = [count / total for count in counts if count > 0]
    return float(-sum(prob * math.log(prob) for prob in probs))


def _quantile(values: list[float], q: float) -> float | None:
    return float(np.quantile(values, q)) if values else None


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _build_components(
    carrier_rows: list[dict[str, Any]],
    mask_rows: list[dict[str, Any]],
    max_union_unique_carriers: int,
    min_visibility_prob: float,
    min_confidence: float,
) -> dict[str, Any]:
    mask_by_key = {_mask_key(row): row for row in mask_rows}
    lengths: Counter[str] = Counter()
    mask_carrier_counts: dict[MaskKey, Counter[str]] = defaultdict(Counter)
    visible_rows: list[dict[str, Any]] = []

    for row in carrier_rows:
        if not _is_visible_row(row, min_visibility_prob=min_visibility_prob, min_confidence=min_confidence):
            continue
        carrier_id = _carrier_global_id(row)
        lengths[carrier_id] += 1
        visible_rows.append(row)
        observed_mask_id = parse_int(row.get("observed_mask_id"))
        if observed_mask_id <= 0:
            continue
        key = (str(row.get("scene")), parse_int(row.get("frame_id")), observed_mask_id)
        if key in mask_by_key:
            mask_carrier_counts[key][carrier_id] += 1

    carrier_ids = sorted(lengths)
    carrier_index = {carrier_id: idx for idx, carrier_id in enumerate(carrier_ids)}
    uf = UnionFind(carrier_index.values())
    union_mask_count = 0
    skipped_large_union_mask_count = 0
    for counts in mask_carrier_counts.values():
        ids = list(counts)
        if len(ids) <= 1:
            continue
        if max_union_unique_carriers >= 0 and len(ids) > int(max_union_unique_carriers):
            skipped_large_union_mask_count += 1
            continue
        first = carrier_index[ids[0]]
        for other in ids[1:]:
            uf.union(first, carrier_index[other])
        union_mask_count += 1

    root_by_carrier = {carrier_id: uf.find(idx) for carrier_id, idx in carrier_index.items()}
    roots = sorted(set(root_by_carrier.values()))
    component_by_root = {root: f"c{idx:05d}" for idx, root in enumerate(roots)}
    component_by_carrier = {carrier_id: component_by_root[root_by_carrier[carrier_id]] for carrier_id in carrier_ids}
    return {
        "visible_rows": visible_rows,
        "component_by_carrier": component_by_carrier,
        "component_ids": sorted(component_by_root.values()),
        "mask_carrier_counts": mask_carrier_counts,
        "union_mask_count": union_mask_count,
        "skipped_large_union_mask_count": skipped_large_union_mask_count,
    }


def _collect_support(
    visible_rows: list[dict[str, Any]],
    mask_rows: list[dict[str, Any]],
    component_by_carrier: dict[str, str],
) -> dict[str, Any]:
    mask_by_key = {_mask_key(row): row for row in mask_rows}
    visible_denominator: Counter[FrameComponentKey] = Counter()
    component_total_visible: Counter[str] = Counter()
    support_by_mask: dict[str, Counter[str]] = defaultdict(Counter)
    mask_carrier_total: Counter[str] = Counter()
    support_visible_row_count = 0

    for row in visible_rows:
        carrier_id = _carrier_global_id(row)
        component_id = component_by_carrier.get(carrier_id)
        if not component_id:
            continue
        scene = str(row.get("scene"))
        frame_id = parse_int(row.get("frame_id"))
        visible_denominator[(scene, frame_id, component_id)] += 1
        component_total_visible[component_id] += 1
        observed_mask_id = parse_int(row.get("observed_mask_id"))
        if observed_mask_id <= 0:
            continue
        key = (scene, frame_id, observed_mask_id)
        mask_row = mask_by_key.get(key)
        if not mask_row:
            continue
        mask_observation_id = str(mask_row.get("mask_observation_id"))
        support_by_mask[mask_observation_id][component_id] += 1
        mask_carrier_total[mask_observation_id] += 1
        support_visible_row_count += 1

    return {
        "visible_denominator": visible_denominator,
        "component_total_visible": component_total_visible,
        "support_by_mask": support_by_mask,
        "mask_carrier_total": mask_carrier_total,
        "support_visible_row_count": support_visible_row_count,
        "visible_row_count": len(visible_rows),
    }


def _support_rows_for_variant(
    variant: str,
    denominator_mode: str,
    tau: float,
    mask_rows: list[dict[str, Any]],
    support_by_mask: dict[str, Counter[str]],
    visible_denominator: Counter[FrameComponentKey],
    component_total_visible: Counter[str],
    mask_carrier_total: Counter[str],
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, float]]:
    rows: list[dict[str, Any]] = []
    components_per_mask: dict[str, int] = {}
    entropy_by_mask: dict[str, float] = {}
    for mask_row in mask_rows:
        scene = str(mask_row.get("scene"))
        frame_id = parse_int(mask_row.get("frame_id"))
        mask_id = parse_int(mask_row.get("mask_id"))
        mask_observation_id = str(mask_row.get("mask_observation_id"))
        component_counts = support_by_mask.get(mask_observation_id, Counter())
        entropy_by_mask[mask_observation_id] = _component_entropy(list(component_counts.values()))
        candidates: list[dict[str, Any]] = []
        for component_id, support_count in component_counts.items():
            visible_count = int(visible_denominator[(scene, frame_id, component_id)])
            total_visible = int(component_total_visible[component_id])
            mask_total = int(mask_carrier_total[mask_observation_id])
            w_visible = float(support_count / max(visible_count, 1))
            r_mask = float(support_count / max(mask_total, 1))
            w_all = float(support_count / max(total_visible, 1))
            if denominator_mode == "visible":
                score = w_visible
            elif denominator_mode == "all_carrier":
                score = w_all
            elif denominator_mode == "mask_support":
                score = r_mask
            elif denominator_mode == "dominant_only":
                score = w_visible
            else:
                raise ValueError(f"unknown denominator mode: {denominator_mode}")
            candidates.append(
                {
                    "variant": variant,
                    "denominator_mode": denominator_mode,
                    "tau": float(tau),
                    "mask_observation_id": mask_observation_id,
                    "scene": scene,
                    "frame_id": frame_id,
                    "mask_id": mask_id,
                    "component_id": component_id,
                    "support_count": int(support_count),
                    "component_visible_count_in_frame": visible_count,
                    "component_total_visible_count": total_visible,
                    "mask_carrier_count": mask_total,
                    "W_visible": w_visible,
                    "R_mask": r_mask,
                    "W_all_carrier": w_all,
                    "selection_score": score,
                    "diagnostic_gt_instance": mask_row.get("diagnostic_gt_instance", ""),
                    "uses_gt_for_prediction": False,
                    "uses_gt_for_diagnostic_labels": True,
                }
            )
        candidates.sort(key=lambda row: (-float(row["selection_score"]), -int(row["support_count"]), str(row["component_id"])))
        selected: list[dict[str, Any]]
        if denominator_mode == "dominant_only":
            selected = candidates[:1]
        else:
            selected = [row for row in candidates if float(row["selection_score"]) >= float(tau)]
        for rank, row in enumerate(selected, start=1):
            out = dict(row)
            out["selected_rank"] = rank
            out["is_dominant_component"] = rank == 1
            rows.append(out)
        components_per_mask[mask_observation_id] = len(selected)
    return rows, components_per_mask, entropy_by_mask


def _variant_summary(
    variant: str,
    rows: list[dict[str, Any]],
    components_per_mask: dict[str, int],
    entropy_by_mask: dict[str, float],
    mask_count: int,
    component_count: int,
    support_visible_row_count: int,
    visible_row_count: int,
) -> dict[str, Any]:
    counts = list(components_per_mask.values())
    supported_components = {str(row["component_id"]) for row in rows}
    incidence_row_count = len(rows)
    zero_count = sum(1 for value in counts if value == 0)
    single_count = sum(1 for value in counts if value == 1)
    multi_count = sum(1 for value in counts if value > 1)
    return {
        "variant": variant,
        "mask_count": int(mask_count),
        "component_count": int(component_count),
        "incidence_row_count": int(incidence_row_count),
        "components_per_mask_mean": _mean([float(value) for value in counts]),
        "components_per_mask_p50": _quantile([float(value) for value in counts], 0.50),
        "components_per_mask_p75": _quantile([float(value) for value in counts], 0.75),
        "components_per_mask_p90": _quantile([float(value) for value in counts], 0.90),
        "multi_component_mask_ratio": float(multi_count / max(mask_count, 1)),
        "single_component_mask_ratio": float(single_count / max(mask_count, 1)),
        "zero_component_mask_ratio": float(zero_count / max(mask_count, 1)),
        "component_coverage_by_masks": float(len(supported_components) / max(component_count, 1)),
        "carrier_coverage_by_masks": float(support_visible_row_count / max(visible_row_count, 1)),
        "dominant_component_collapse_detected": bool(incidence_row_count <= mask_count),
        "component_support_distribution_entropy": _mean(list(entropy_by_mask.values())),
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }


def _mask_summary_rows(
    mask_rows: list[dict[str, Any]],
    support_by_mask: dict[str, Counter[str]],
    components_per_mask_by_variant: dict[str, dict[str, int]],
    entropy_by_mask: dict[str, float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mask_row in mask_rows:
        mask_observation_id = str(mask_row.get("mask_observation_id"))
        component_counts = support_by_mask.get(mask_observation_id, Counter())
        variant_counts = {
            f"{variant}_component_count": components.get(mask_observation_id, 0)
            for variant, components in components_per_mask_by_variant.items()
        }
        rows.append(
            {
                "mask_observation_id": mask_observation_id,
                "scene": mask_row.get("scene"),
                "frame_id": mask_row.get("frame_id"),
                "mask_id": mask_row.get("mask_id"),
                "mask_area": mask_row.get("mask_area"),
                "raw_supported_component_count": int(len(component_counts)),
                "raw_support_carrier_count": int(sum(component_counts.values())),
                "raw_component_entropy": entropy_by_mask.get(mask_observation_id, 0.0),
                "diagnostic_gt_instance": mask_row.get("diagnostic_gt_instance", ""),
                "diagnostic_gt_purity": mask_row.get("diagnostic_gt_purity", ""),
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic_labels": True,
                **variant_counts,
            }
        )
    return rows


def _scene_summary_rows(mask_summary_rows: list[dict[str, Any]], variant: str = "I0_visible_tau0.10") -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    key = f"{variant}_component_count"
    for row in mask_summary_rows:
        groups[str(row["scene"])].append(row)
    out: list[dict[str, Any]] = []
    for scene, rows in sorted(groups.items()):
        counts = [parse_int(row.get(key)) for row in rows]
        raw_counts = [parse_int(row.get("raw_supported_component_count")) for row in rows]
        out.append(
            {
                "scene": scene,
                "mask_count": len(rows),
                "raw_components_per_mask_mean": _mean([float(value) for value in raw_counts]),
                "components_per_mask_mean": _mean([float(value) for value in counts]),
                "components_per_mask_p50": _quantile([float(value) for value in counts], 0.50),
                "zero_component_mask_ratio": float(sum(1 for value in counts if value == 0) / max(len(counts), 1)),
                "multi_component_mask_ratio": float(sum(1 for value in counts if value > 1) / max(len(counts), 1)),
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic_labels": True,
            }
        )
    return out


def _write_visualizations(
    output_root: Path,
    vis_root: Path,
    support_rows: list[dict[str, Any]],
    mask_summary: list[dict[str, Any]],
    variant_summaries: list[dict[str, Any]],
    max_masks_per_scene: int = 40,
) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - environment dependent
        write_json(output_root / "visualization_error.json", {"error": repr(exc)})
        return [{"path": str(output_root / "visualization_error.json"), "status": "matplotlib_unavailable"}]

    vis_root.mkdir(parents=True, exist_ok=True)
    main_rows = [row for row in support_rows if row.get("variant") == "I0_visible_tau0.10"]
    rows_by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in main_rows:
        rows_by_scene[str(row["scene"])].append(row)
    mask_by_id = {str(row["mask_observation_id"]): row for row in mask_summary}

    for scene, rows in sorted(rows_by_scene.items()):
        mask_ids = sorted({str(row["mask_observation_id"]) for row in rows}, key=lambda mo: -parse_int(mask_by_id.get(mo, {}).get("I0_visible_tau0.10_component_count")))
        component_ids = [item for item, _count in Counter(str(row["component_id"]) for row in rows).most_common(60)]
        heat = np.zeros((min(len(mask_ids), max_masks_per_scene), len(component_ids)), dtype=np.float32)
        mask_index = {mask_id: idx for idx, mask_id in enumerate(mask_ids[:max_masks_per_scene])}
        component_index = {component_id: idx for idx, component_id in enumerate(component_ids)}
        for row in rows:
            mi = mask_index.get(str(row["mask_observation_id"]))
            ci = component_index.get(str(row["component_id"]))
            if mi is None or ci is None:
                continue
            heat[mi, ci] = max(heat[mi, ci], float(row["W_visible"]))
        fig, ax = plt.subplots(figsize=(12, max(3, heat.shape[0] * 0.16)))
        ax.imshow(heat, aspect="auto", interpolation="nearest", cmap="magma")
        ax.set_title(f"{scene} mask-component W visible heatmap")
        ax.set_xlabel("top components")
        ax.set_ylabel("top masks by component count")
        heat_path = vis_root / f"mask_component_heatmap_{scene}.png"
        fig.tight_layout()
        fig.savefig(heat_path, dpi=140)
        plt.close(fig)
        manifest.append({"path": str(heat_path), "kind": "mask_component_heatmap", "scene": scene})

        scene_masks = [row for row in mask_summary if str(row["scene"]) == scene]
        counts = [parse_int(row.get("I0_visible_tau0.10_component_count")) for row in scene_masks]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(counts, bins=min(60, max(5, len(set(counts)))), color="#3b82f6")
        ax.set_title(f"{scene} components per mask")
        ax.set_xlabel("components per mask")
        ax.set_ylabel("mask count")
        hist_path = vis_root / f"components_per_mask_hist_{scene}.png"
        fig.tight_layout()
        fig.savefig(hist_path, dpi=140)
        plt.close(fig)
        manifest.append({"path": str(hist_path), "kind": "components_per_mask_hist", "scene": scene})

        for panel_kind, selected in [
            ("top_multicomponent_masks_panel", sorted(scene_masks, key=lambda row: -parse_int(row.get("I0_visible_tau0.10_component_count")))[:6]),
            ("zero_component_masks_panel", [row for row in scene_masks if parse_int(row.get("I0_visible_tau0.10_component_count")) == 0][:6]),
        ]:
            if not selected:
                continue
            fig, axes = plt.subplots(1, len(selected), figsize=(3 * len(selected), 3))
            if len(selected) == 1:
                axes = [axes]
            for ax, mask_row in zip(axes, selected):
                frame_id = parse_int(mask_row.get("frame_id"))
                mask_id = parse_int(mask_row.get("mask_id"))
                label = load_mask_label(scene, frame_id)
                if label is None:
                    image = np.zeros((64, 64), dtype=np.uint8)
                else:
                    image = (label == mask_id).astype(np.uint8)
                ax.imshow(image, cmap="gray", interpolation="nearest")
                ax.set_title(f"f{frame_id} m{mask_id} c{mask_row.get('I0_visible_tau0.10_component_count')}")
                ax.axis("off")
            panel_path = vis_root / f"{panel_kind}_{scene}.png"
            fig.tight_layout()
            fig.savefig(panel_path, dpi=140)
            plt.close(fig)
            manifest.append({"path": str(panel_path), "kind": panel_kind, "scene": scene})

    labels = [row["variant"] for row in variant_summaries]
    incidence = [row["incidence_row_count"] for row in variant_summaries]
    mask_counts = [row["mask_count"] for row in variant_summaries]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(9, len(labels) * 1.6), 4.5))
    ax.bar(x - 0.18, incidence, width=0.36, label="incidence rows")
    ax.bar(x + 0.18, mask_counts, width=0.36, label="mask count")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yscale("log")
    ax.set_title("Dominant collapse check")
    ax.legend()
    collapse_path = vis_root / "dominant_collapse_check.png"
    fig.tight_layout()
    fig.savefig(collapse_path, dpi=140)
    plt.close(fig)
    manifest.append({"path": str(collapse_path), "kind": "dominant_collapse_check", "scene": "all"})
    return manifest


def build_mask_component_support(
    carrier_table_path: str | Path = "outputs/audit/v47_observation_tables_metricfix/carrier_observation_table.csv",
    mask_table_path: str | Path = "outputs/audit/v47_observation_tables_metricfix/mask_observation_table.csv",
    max_union_unique_carriers: int = 32,
    min_visibility_prob: float = 0.5,
    min_confidence: float = 0.5,
    extra_visible_taus: list[float] | None = None,
    gate_variant: str = "I0_visible_tau0.10",
) -> dict[str, Any]:
    carrier_rows = read_csv(_project(carrier_table_path))
    mask_rows = read_csv(_project(mask_table_path))
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
    variants = [
        ("I0_visible_tau0.10", "visible", 0.10),
        ("I1_visible_tau0.20", "visible", 0.20),
        ("I2_visible_tau0.30", "visible", 0.30),
        ("I3_all_carrier_den_tau0.10", "all_carrier", 0.10),
        ("I4_dominant_component_only", "dominant_only", 0.0),
        ("I5_mask_support_den_tau0.10", "mask_support", 0.10),
    ]
    existing_variant_names = {name for name, _mode, _tau in variants}
    for idx, tau in enumerate(extra_visible_taus or []):
        variant_name = f"R{idx}_visible_tau{float(tau):.2f}"
        if variant_name not in existing_variant_names:
            variants.append((variant_name, "visible", float(tau)))
            existing_variant_names.add(variant_name)
    all_support_rows: list[dict[str, Any]] = []
    variant_summaries: list[dict[str, Any]] = []
    components_per_mask_by_variant: dict[str, dict[str, int]] = {}
    entropy_for_summary: dict[str, float] = {}
    for variant, denominator_mode, tau in variants:
        rows, components_per_mask, entropy_by_mask = _support_rows_for_variant(
            variant=variant,
            denominator_mode=denominator_mode,
            tau=tau,
            mask_rows=mask_rows,
            support_by_mask=support_payload["support_by_mask"],
            visible_denominator=support_payload["visible_denominator"],
            component_total_visible=support_payload["component_total_visible"],
            mask_carrier_total=support_payload["mask_carrier_total"],
        )
        all_support_rows.extend(rows)
        components_per_mask_by_variant[variant] = components_per_mask
        entropy_for_summary = entropy_by_mask
        summary = _variant_summary(
            variant=variant,
            rows=rows,
            components_per_mask=components_per_mask,
            entropy_by_mask=entropy_by_mask,
            mask_count=len(mask_rows),
            component_count=len(component_payload["component_ids"]),
            support_visible_row_count=support_payload["support_visible_row_count"],
            visible_row_count=support_payload["visible_row_count"],
        )
        variant_summaries.append(summary)

    mask_summary = _mask_summary_rows(mask_rows, support_payload["support_by_mask"], components_per_mask_by_variant, entropy_for_summary)
    scene_summary = _scene_summary_rows(mask_summary, variant="I0_visible_tau0.10")
    summary_by_variant = {str(row["variant"]): row for row in variant_summaries}
    if gate_variant not in summary_by_variant:
        raise ValueError(f"gate_variant={gate_variant} not found in variants: {sorted(summary_by_variant)}")
    main_summary = summary_by_variant[gate_variant]
    gate = {
        "dominant_component_collapse_detected_false": not bool(main_summary["dominant_component_collapse_detected"]),
        "incidence_row_count_gt_mask_count": int(main_summary["incidence_row_count"]) > int(main_summary["mask_count"]),
        "component_coverage_by_masks_ge_0.80": float(main_summary["component_coverage_by_masks"]) >= 0.80,
        "zero_component_mask_ratio_le_0.25": float(main_summary["zero_component_mask_ratio"]) <= 0.25,
    }
    gate["pass"] = bool(all(gate.values()))
    summary = {
        "phase": "v53_mask_component_support",
        "created_at": utc_now(),
        "carrier_table_path": str(carrier_table_path),
        "mask_table_path": str(mask_table_path),
        "max_union_unique_carriers": int(max_union_unique_carriers),
        "min_visibility_prob": float(min_visibility_prob),
        "min_confidence": float(min_confidence),
        "visible_row_count": support_payload["visible_row_count"],
        "support_visible_row_count": support_payload["support_visible_row_count"],
        "component_count": len(component_payload["component_ids"]),
        "union_mask_count": component_payload["union_mask_count"],
        "skipped_large_union_mask_count": component_payload["skipped_large_union_mask_count"],
        "default_main_variant": "I0_visible_tau0.10",
        "gate_variant": gate_variant,
        "main_summary": main_summary,
        "variant_summaries": variant_summaries,
        "gate": gate,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }
    return {
        "summary": summary,
        "support_rows": all_support_rows,
        "mask_summary_rows": mask_summary,
        "scene_summary_rows": scene_summary,
        "component_ids": component_payload["component_ids"],
    }


def write_mask_component_support(
    output_root: str | Path,
    payload: dict[str, Any],
    visualization_root: str | Path = "outputs/audit/v53_visualizations/local_objectlets",
) -> None:
    out = _project(output_root)
    vis = _project(visualization_root)
    write_json(out / "support_summary.json", payload["summary"])
    write_csv(out / "mask_component_support_rows.csv", payload["support_rows"])
    write_csv(out / "mask_summary_rows.csv", payload["mask_summary_rows"])
    write_csv(out / "scene_summary_rows.csv", payload["scene_summary_rows"])
    visual_manifest = _write_visualizations(
        output_root=out,
        vis_root=vis,
        support_rows=payload["support_rows"],
        mask_summary=payload["mask_summary_rows"],
        variant_summaries=payload["summary"]["variant_summaries"],
    )
    write_json(out / "visualization_manifest.json", {"phase": "v53_mask_component_support", "files": visual_manifest})


__all__ = ["build_mask_component_support", "write_mask_component_support"]
