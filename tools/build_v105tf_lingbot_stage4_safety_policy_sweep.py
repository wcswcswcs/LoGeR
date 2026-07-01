#!/usr/bin/env python3
"""Search stronger non-target semantic safety filters for v105 LingBot Stage 4."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/acl2_v105tf_dualtrack_lingbot_loger_evidence_eligibility"
STAGE3 = RESULT_ROOT / "stage3_lingbot_oracle"
STAGE4 = RESULT_ROOT / "stage4_lingbot_action_pilot_or_blocked"

BASE_SEMANTIC_THR = 0.07418
BASE_LOCAL_WINDOW_THR = 0.7209
MIN_SAMPLE_POSITION = 8
FEATURE_COLUMNS = [
    "scale_reference_patch_frac",
    "local_registration_patch_frac",
    "context_only_patch_frac",
    "reject_unreliable_patch_frac",
    "semantic_confidence_mean",
    "semantic_confidence_p10",
    "trace_topk_attention_sum",
    "scale_reference_context_attention_frac",
    "local_window_context_attention_frac",
    "current_or_latest_frame_attention_frac",
    "semantic_scale_reference_attention_frac",
    "semantic_local_registration_attention_frac",
    "semantic_context_only_attention_frac",
    "semantic_reject_unreliable_attention_frac",
    "scale_context_reject_attention_frac",
    "scale_context_structure_attention_frac",
    "local_context_reject_attention_frac",
]
LABEL_FEATURES = [
    "label_road_frac",
    "label_tree_frac",
    "label_house_frac",
    "label_sky_frac",
    "label_car_frac",
    "label_path_frac",
    "label_grass_frac",
    "label_void_frac",
    "label_building_frac",
    "label_other_plant_frac",
    "label_handrail_or_fence_frac",
    "label_ground_frac",
    "label_wall_frac",
    "label_pole_frac",
    "label_vegetation_frac",
    "label_structure_frac",
    "label_dynamic_frac",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def as_float(row: dict[str, Any], key: str) -> float:
    raw = row.get(key, 0.0)
    if raw in ("", None):
        return 0.0
    return float(raw)


def as_int(row: dict[str, Any], key: str) -> int:
    return int(float(row.get(key, 0) or 0))


def as_bool(row: dict[str, Any], key: str) -> bool:
    return str(row.get(key, "")).strip().lower() in {"1", "true", "yes"}


def seq_name(row: dict[str, Any]) -> str:
    return f"{as_int(row, 'seq'):02d}"


def parse_top_labels(raw: str) -> dict[str, float]:
    counts: dict[str, float] = {}
    for part in str(raw or "").split(";"):
        if ":" not in part:
            continue
        name, value = part.rsplit(":", 1)
        try:
            counts[name.strip()] = float(value)
        except ValueError:
            continue
    return counts


def add_label_features(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    counts = parse_top_labels(str(row.get("top_labels", "")))
    total = sum(counts.values()) or as_float(row, "patch_count") or 1.0

    def frac(*names: str) -> float:
        return sum(counts.get(name, 0.0) for name in names) / total

    out["label_road_frac"] = frac("road")
    out["label_tree_frac"] = frac("tree")
    out["label_house_frac"] = frac("house")
    out["label_sky_frac"] = frac("sky")
    out["label_car_frac"] = frac("car")
    out["label_path_frac"] = frac("path")
    out["label_grass_frac"] = frac("grass")
    out["label_void_frac"] = frac("void")
    out["label_building_frac"] = frac("building")
    out["label_other_plant_frac"] = frac("other_plant")
    out["label_handrail_or_fence_frac"] = frac("handrail_or_fence")
    out["label_ground_frac"] = frac("ground")
    out["label_wall_frac"] = frac("wall")
    out["label_pole_frac"] = frac("pole")
    out["label_vegetation_frac"] = frac("tree", "grass", "other_plant")
    out["label_structure_frac"] = frac("house", "building", "wall", "handrail_or_fence", "pillar")
    out["label_dynamic_frac"] = frac("car", "person", "traffic sign", "traffic light")
    return out


def bit(indices: list[int]) -> int:
    mask = 0
    for idx in indices:
        mask |= 1 << idx
    return mask


def iter_indices(mask: int, n: int) -> list[int]:
    return [idx for idx in range(n) if mask & (1 << idx)]


def count_mask(mask: int) -> int:
    return mask.bit_count()


def thresholds(values: list[float]) -> list[float]:
    uniq = sorted({round(v, 12) for v in values if math.isfinite(v)})
    return uniq


def build_predicates(rows: list[dict[str, Any]], base_mask: int) -> list[dict[str, Any]]:
    features = FEATURE_COLUMNS + LABEL_FEATURES
    predicates: list[dict[str, Any]] = []
    seen_masks: dict[tuple[str, str, int], str] = {}
    n = len(rows)
    for feature in features:
        values = [as_float(row, feature) for row in rows]
        for threshold in thresholds(values):
            for op in (">=", "<="):
                if op == ">=":
                    pred_mask = bit([idx for idx, row in enumerate(rows) if as_float(row, feature) >= threshold])
                else:
                    pred_mask = bit([idx for idx, row in enumerate(rows) if as_float(row, feature) <= threshold])
                refined = base_mask & pred_mask
                selected = count_mask(refined)
                if selected == 0 or selected == count_mask(base_mask):
                    continue
                key = (feature, op, refined)
                expr = f"{feature}_{'ge' if op == '>=' else 'le'}_{threshold:.6g}"
                if key in seen_masks:
                    continue
                seen_masks[key] = expr
                predicates.append(
                    {
                        "feature": feature,
                        "op": op,
                        "threshold": threshold,
                        "mask": pred_mask,
                        "base_refined_mask": refined,
                        "expr": expr,
                    }
                )
    # Prefer lower-complexity, higher-impact predicates first for pair search determinism.
    predicates.sort(key=lambda item: (item["feature"], item["op"], item["threshold"]))
    return predicates


def metrics(rows: list[dict[str, Any]], mask: int, clauses: list[str]) -> dict[str, Any]:
    n = len(rows)
    selected_indices = iter_indices(mask, n)
    bad_total = sum(1 for row in rows if as_bool(row, "bad_label"))
    good_total = sum(1 for row in rows if as_bool(row, "good_label"))
    bad_selected = [idx for idx in selected_indices if as_bool(rows[idx], "bad_label")]
    good_selected = [idx for idx in selected_indices if as_bool(rows[idx], "good_label")]
    neutral_selected = [
        idx
        for idx in selected_indices
        if not as_bool(rows[idx], "bad_label") and not as_bool(rows[idx], "good_label")
    ]
    selected_seqs = sorted({seq_name(rows[idx]) for idx in selected_indices})
    bad_seqs = sorted({seq_name(rows[idx]) for idx in bad_selected})
    good_seqs = sorted({seq_name(rows[idx]) for idx in good_selected})
    labeled_selected = len(bad_selected) + len(good_selected)
    bad_recall = len(bad_selected) / bad_total if bad_total else 0.0
    good_fpr = len(good_selected) / good_total if good_total else 0.0
    precision_vs_good = len(bad_selected) / labeled_selected if labeled_selected else 0.0
    bad_to_good_margin = len(bad_selected) - 3 * len(good_selected)
    deployable = len(bad_selected) >= 4 and len(bad_seqs) >= 2 and len(good_selected) <= 2
    strict_deployable = len(bad_selected) >= 4 and len(bad_seqs) >= 2 and len(good_selected) <= 1
    return {
        "schema": "acl2_v105tf_lingbot_stage4_safety_sweep_candidate_v1",
        "policy": " AND ".join(clauses),
        "clause_count": len(clauses),
        "selected_count": len(selected_indices),
        "bad_selected": len(bad_selected),
        "good_selected": len(good_selected),
        "neutral_selected": len(neutral_selected),
        "bad_total": bad_total,
        "good_total": good_total,
        "bad_recall": bad_recall,
        "good_fpr": good_fpr,
        "precision_vs_good": precision_vs_good,
        "bad_to_good_margin": bad_to_good_margin,
        "selected_seq_coverage": len(selected_seqs),
        "bad_seq_coverage": len(bad_seqs),
        "good_seq_coverage": len(good_seqs),
        "selected_seqs": ";".join(selected_seqs),
        "bad_seqs": ";".join(bad_seqs),
        "good_seqs": ";".join(good_seqs),
        "selected_positions": ";".join(f"{seq_name(rows[idx])}:{as_int(rows[idx], 'sample_position')}" for idx in selected_indices),
        "bad_positions": ";".join(f"{seq_name(rows[idx])}:{as_int(rows[idx], 'sample_position')}" for idx in bad_selected),
        "good_positions": ";".join(f"{seq_name(rows[idx])}:{as_int(rows[idx], 'sample_position')}" for idx in good_selected),
        "deployable_candidate": deployable,
        "strict_deployable_candidate": strict_deployable,
        "_mask": mask,
    }


def candidate_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        bool(row["strict_deployable_candidate"]),
        bool(row["deployable_candidate"]),
        int(row["bad_to_good_margin"]),
        float(row["precision_vs_good"]),
        float(row["bad_recall"]),
        -float(row["good_fpr"]),
        -int(row["clause_count"]),
        -int(row["selected_count"]),
    )


def public_row(row: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in row.items() if not k.startswith("_")}


def build() -> dict[str, Any]:
    STAGE4.mkdir(parents=True, exist_ok=True)
    rows = [add_label_features(row) for row in read_csv(STAGE3 / "frame_semantic_geometry_rows.csv")]
    n = len(rows)
    eligible_mask = bit([idx for idx, row in enumerate(rows) if as_int(row, "sample_position") >= MIN_SAMPLE_POSITION])
    base_mask = eligible_mask
    base_mask &= bit(
        [
            idx
            for idx, row in enumerate(rows)
            if as_float(row, "semantic_reject_unreliable_attention_frac") >= BASE_SEMANTIC_THR
        ]
    )
    base_mask &= bit(
        [
            idx
            for idx, row in enumerate(rows)
            if as_float(row, "local_window_context_attention_frac") >= BASE_LOCAL_WINDOW_THR
        ]
    )
    base_clauses = [
        f"semantic_reject_unreliable_attention_frac_ge_{BASE_SEMANTIC_THR}",
        f"local_window_context_attention_frac_ge_{BASE_LOCAL_WINDOW_THR}",
    ]
    base_metrics = metrics(rows, base_mask, base_clauses)
    predicates = build_predicates(rows, base_mask)

    candidates: list[dict[str, Any]] = [base_metrics]
    seen_masks = {base_mask}
    for pred in predicates:
        mask = base_mask & int(pred["mask"])
        if mask in seen_masks:
            continue
        seen_masks.add(mask)
        candidates.append(metrics(rows, mask, base_clauses + [str(pred["expr"])]))

    for left_idx, left in enumerate(predicates):
        left_mask = base_mask & int(left["mask"])
        if count_mask(left_mask) == 0:
            continue
        for right in predicates[left_idx + 1 :]:
            mask = left_mask & int(right["mask"])
            if mask == 0 or mask in seen_masks:
                continue
            seen_masks.add(mask)
            candidates.append(metrics(rows, mask, base_clauses + [str(left["expr"]), str(right["expr"])]))

    candidates.sort(key=candidate_sort_key, reverse=True)
    top_rows = [public_row(row) for row in candidates[:200]]
    deployable = [row for row in candidates if row["deployable_candidate"]]
    selected = deployable[0] if deployable else None
    selected_rows: list[dict[str, Any]] = []
    if selected:
        for idx in iter_indices(int(selected["_mask"]), n):
            row = rows[idx]
            selected_rows.append(
                {
                    "schema": "acl2_v105tf_lingbot_stage4_safety_selected_frame_v1",
                    "policy": selected["policy"],
                    "seq": seq_name(row),
                    "sample_position": as_int(row, "sample_position"),
                    "original_frame": as_int(row, "original_frame"),
                    "bad_label": as_bool(row, "bad_label"),
                    "good_label": as_bool(row, "good_label"),
                    "sim3_residual_m": as_float(row, "sim3_residual_m"),
                    "semantic_reject_unreliable_attention_frac": as_float(
                        row, "semantic_reject_unreliable_attention_frac"
                    ),
                    "local_window_context_attention_frac": as_float(row, "local_window_context_attention_frac"),
                    "reject_unreliable_patch_frac": as_float(row, "reject_unreliable_patch_frac"),
                    "semantic_confidence_p10": as_float(row, "semantic_confidence_p10"),
                    "top_labels": row.get("top_labels", ""),
                }
            )

    write_csv(STAGE4 / "semantic_safety_filter_sweep_top_candidates.csv", top_rows)
    write_csv(STAGE4 / "semantic_safety_filter_selected_rows.csv", selected_rows)
    summary = {
        "schema": "acl2_v105tf_lingbot_stage4_safety_sweep_summary_v1",
        "candidate_count": len(candidates),
        "predicate_count": len(predicates),
        "base_policy": base_metrics["policy"],
        "base_selected_count": base_metrics["selected_count"],
        "base_bad_selected": base_metrics["bad_selected"],
        "base_good_selected": base_metrics["good_selected"],
        "base_bad_recall": base_metrics["bad_recall"],
        "base_good_fpr": base_metrics["good_fpr"],
        "deployable_candidate_count": len(deployable),
        "strict_deployable_candidate_count": sum(1 for row in candidates if row["strict_deployable_candidate"]),
        "selected_candidate": public_row(selected) if selected else None,
        "outputs": {
            "top_candidates": str(STAGE4 / "semantic_safety_filter_sweep_top_candidates.csv"),
            "selected_rows": str(STAGE4 / "semantic_safety_filter_selected_rows.csv"),
        },
    }
    (STAGE4 / "semantic_safety_filter_sweep_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":
    build()
