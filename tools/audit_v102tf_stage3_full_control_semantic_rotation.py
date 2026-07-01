#!/usr/bin/env python3
"""Audit v102 Stage3 semantic oracle controls on the v101 target universe.

This is a diagnostic-only fail-forward audit.  It uses existing v101
per-anchor geometry observability and Stage-C seed geometry rows to test
whether semantic/observability selectors survive same-count random,
semantic-label rotation, and anchor-geometry rotation controls.

No runtime action is authorized here.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("results/acl2_v102tf_drift_source_autopsy_semantic_oracle_memory_control")
STAGE3 = ROOT / "stage3_semantic_oracle_upper_bound"

V101 = Path("results/acl2_v101tf_semantic_anchor_state_estimation_scale_update_admission")
PER_ANCHOR_ROWS = V101 / "trackV_anchor_scale_observability/per_anchor_geometry_observability_rows.csv"
LIFECYCLE_JOIN_ROWS = V101 / "final_decision/anchor_seed_lifecycle_geometry_observability_join_rows.csv"
CASE_ROWS = V101 / "final_decision/anchor_seed_lifecycle_geometry_observability_case_rows.csv"
STAGE_C_CASE_ROWS = V101 / "final_decision/stage_c_seed_geometry_smoke_target28_case_rows.csv"

CASE_OUT = STAGE3 / "stage3_full_control_semantic_rotation_case_rows.csv"
POLICY_OUT = STAGE3 / "stage3_full_control_semantic_rotation_policy_rows.csv"
CONTROL_OUT = STAGE3 / "stage3_full_control_semantic_rotation_control_rows.csv"
SUMMARY_OUT = STAGE3 / "stage3_full_control_semantic_rotation_summary.json"
REPORT_OUT = STAGE3 / "stage3_full_control_semantic_rotation_report.md"

POS_HANDOFF = "HANDOFF_SCALE_GAUGE_TARGET"
SAFE_GOOD = "SAFE_GOOD"

# The numeric semantic ids in Stage-C caches are sequence-local.  These names
# were read from the corresponding sequence chunk_000 masklet payloads; keeping
# them here avoids a hard torch dependency for this audit script.
SEQ_LABEL_NAMES = {
    "00": [
        "void",
        "parasol_or_umbrella",
        "truck",
        "bicycle",
        "motorcycle",
        "person",
        "bench",
        "flower_pot_or_vase",
        "handrail_or_fence",
        "pillar",
        "pole",
        "ground",
        "grass",
        "road",
        "path",
        "building",
        "house",
        "other_construction",
        "sky",
        "billboard_or_bulletin_board",
        "wheeled_machine",
        "tree",
        "flower",
        "other_plant",
        "trash_can",
        "car",
        "traffic sign",
    ],
    "01": [
        "void",
        "person",
        "wall",
        "handrail_or_fence",
        "pole",
        "ground",
        "grass",
        "road",
        "building",
        "house",
        "bridge",
        "other_construction",
        "sky",
        "mountain",
        "billboard_or_bulletin_board",
        "tree",
        "car",
        "traffic sign",
    ],
    "02": [
        "void",
        "motorcycle",
        "person",
        "bench",
        "flower_pot_or_vase",
        "wall",
        "stair",
        "handrail_or_fence",
        "pillar",
        "pole",
        "ground",
        "grass",
        "road",
        "path",
        "house",
        "other_construction",
        "sky",
        "mountain",
        "stone",
        "billboard_or_bulletin_board",
        "wheeled_machine",
        "tree",
        "flower",
        "other_plant",
        "trash_can",
        "car",
        "traffic sign",
    ],
    "05": [
        "void",
        "roadblock",
        "bus",
        "truck",
        "bicycle",
        "person",
        "bench",
        "flower_pot_or_vase",
        "stair",
        "handrail_or_fence",
        "pillar",
        "pole",
        "ground",
        "grass",
        "road",
        "path",
        "crosswalk",
        "house",
        "sky",
        "billboard_or_bulletin_board",
        "tree",
        "flower",
        "other_plant",
        "trash_can",
        "other_machine",
        "car",
        "traffic sign",
    ],
}

STABLE_WORDS = {
    "wall",
    "building",
    "house",
    "other_construction",
    "bridge",
    "handrail_or_fence",
    "pillar",
    "pole",
    "ground",
    "road",
    "path",
    "stair",
    "stone",
    "traffic sign",
}
UNRELIABLE_WORDS = {
    "void",
    "sky",
    "grass",
    "tree",
    "flower",
    "other_plant",
    "person",
    "car",
    "truck",
    "bus",
    "bicycle",
    "motorcycle",
    "wheeled_machine",
    "other_machine",
}

FEATURES = [
    ("semantic_unreliable_anchor_frac", "high", "B2 unreliable semantic mass"),
    ("unreliable_or_low_observable_frac", "high", "B2/B5 unreliable or low-O_scale evidence"),
    ("stable_observable_frac_q50", "low", "B3 low stable+observable anchor fraction"),
    ("stable_observable_topk_mean", "low", "B3 low stable-anchor O_scale top-k mean"),
    ("O_scale_mean", "low", "B3 low case mean O_scale"),
    ("O_scale_top25_mean", "low", "B3 low top-quartile O_scale"),
    ("lifecycle_geometry_unique_coverage", "low", "B6 low lifecycle geometry coverage"),
    ("stage_c_same_payload_join_coverage", "low", "B6 low Stage-C same-payload geometry join coverage"),
    ("local_scale_mode_entropy", "high", "B6 high local scale-mode entropy"),
    ("abs_log_depth_ratio_mean", "high", "B3 high query/cache depth-ratio mismatch"),
    ("dominant_semantic_name_frac", "high", "B1 dominant semantic concentration"),
    ("semantic_unique_name_count", "low", "B1 low semantic diversity"),
]
SEMANTIC_FEATURES = {
    "semantic_unreliable_anchor_frac",
    "unreliable_or_low_observable_frac",
    "stable_observable_frac_q50",
    "stable_observable_topk_mean",
    "dominant_semantic_name_frac",
    "semantic_unique_name_count",
}
ANCHOR_GEOMETRY_FEATURES = {
    "unreliable_or_low_observable_frac",
    "stable_observable_frac_q50",
    "stable_observable_topk_mean",
    "O_scale_mean",
    "O_scale_top25_mean",
}


def f(value: Any, default: float = math.nan) -> float:
    try:
        if value is None or value == "":
            return default
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def b(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    if not keys:
        keys = ["empty"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key, "")) for key in keys})


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(jsonable(value), sort_keys=True)
    return value


def jsonable(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def mean(values: list[Any]) -> float:
    xs = [f(v) for v in values if math.isfinite(f(v))]
    return sum(xs) / len(xs) if xs else math.nan


def quantile(values: list[Any], q: float) -> float:
    xs = sorted(f(v) for v in values if math.isfinite(f(v)))
    if not xs:
        return math.nan
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def deterministic_rng(*parts: str) -> random.Random:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def seq_of(case_id: str) -> str:
    return str(case_id).split("_", 1)[0]


def semantic_name(case_id: str, label: Any) -> str:
    names = SEQ_LABEL_NAMES.get(seq_of(case_id), [])
    idx = int(f(label, -1))
    if 0 <= idx < len(names):
        return names[idx]
    return f"unknown_label_{label}"


def is_stable(name: str) -> bool:
    return name in STABLE_WORDS


def is_unreliable(name: str) -> bool:
    return name in UNRELIABLE_WORDS


def grouped(rows: list[dict[str, str]], key: str) -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get(key):
            out[row[key]].append(row)
    return out


def first_by_case(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        case_id = row.get("case_id", "")
        if case_id and case_id not in out:
            out[case_id] = row
    return out


def shuffle_field_by_seq(
    anchor_rows: list[dict[str, Any]],
    field: str,
    *,
    tag: str,
    trial: int,
) -> dict[int, Any]:
    idx_by_seq: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(anchor_rows):
        idx_by_seq[seq_of(str(row.get("case_id", "")))].append(idx)
    out: dict[int, Any] = {}
    for seq, indices in idx_by_seq.items():
        values = [anchor_rows[idx].get(field) for idx in indices]
        rng = deterministic_rng(tag, str(trial), seq, field)
        rng.shuffle(values)
        for idx, value in zip(indices, values):
            out[idx] = value
    return out


def top_fraction_mean(values: list[float], frac: float) -> float:
    xs = sorted([value for value in values if math.isfinite(value)], reverse=True)
    if not xs:
        return math.nan
    keep = max(1, int(math.ceil(len(xs) * frac)))
    return sum(xs[:keep]) / keep


def build_case_features(
    *,
    anchor_rows: list[dict[str, str]],
    lifecycle_rows: list[dict[str, str]],
    case_rows: list[dict[str, str]],
    stagec_case_rows: list[dict[str, str]],
    semantic_rotation_trial: int | None = None,
    anchor_geometry_rotation_trial: int | None = None,
) -> list[dict[str, Any]]:
    all_o = [f(row.get("O_scale_repaired")) for row in anchor_rows if math.isfinite(f(row.get("O_scale_repaired")))]
    o_q25 = quantile(all_o, 0.25)
    o_q50 = quantile(all_o, 0.50)
    anchor_rows_any: list[dict[str, Any]] = [dict(row) for row in anchor_rows]
    if semantic_rotation_trial is not None:
        rotated = shuffle_field_by_seq(anchor_rows_any, "semantic_name", tag="semantic_rotation", trial=semantic_rotation_trial)
        for idx, value in rotated.items():
            anchor_rows_any[idx]["semantic_name"] = value
    if anchor_geometry_rotation_trial is not None:
        for field in ["O_scale_repaired", "anchor_point_count", "P_parallax_score", "D_spread_score", "semantic_risk_penalty"]:
            rotated = shuffle_field_by_seq(anchor_rows_any, field, tag="anchor_geometry_rotation", trial=anchor_geometry_rotation_trial)
            for idx, value in rotated.items():
                anchor_rows_any[idx][field] = value

    anchors_by_case = grouped(anchor_rows_any, "case_id")
    lifecycle_by_case = grouped(lifecycle_rows, "case_id")
    case_meta = first_by_case(case_rows)
    stagec_by_case = first_by_case(stagec_case_rows)
    all_cases = sorted(set(case_meta) | set(anchors_by_case) | set(stagec_by_case))
    out: list[dict[str, Any]] = []
    for case_id in all_cases:
        anchors = anchors_by_case.get(case_id, [])
        lifecycle = lifecycle_by_case.get(case_id, [])
        meta = case_meta.get(case_id, {})
        stagec = stagec_by_case.get(case_id, {})
        tax = meta.get("target_taxonomy") or stagec.get("target_taxonomy") or (anchors[0].get("target_taxonomy") if anchors else "")
        o_vals = [f(row.get("O_scale_repaired")) for row in anchors if math.isfinite(f(row.get("O_scale_repaired")))]
        stable_o = [
            f(row.get("O_scale_repaired"))
            for row in anchors
            if is_stable(str(row.get("semantic_name", ""))) and math.isfinite(f(row.get("O_scale_repaired")))
        ]
        names = [str(row.get("semantic_name", "")) for row in anchors]
        name_counts: dict[str, int] = defaultdict(int)
        for name in names:
            name_counts[name] += 1
        n = len(anchors)
        stable_count = sum(1 for name in names if is_stable(name))
        unreliable_count = sum(
            1
            for row in anchors
            if is_unreliable(str(row.get("semantic_name", ""))) or f(row.get("semantic_risk_penalty"), 0.0) > 0.0
        )
        stable_observable = sum(
            1
            for row in anchors
            if is_stable(str(row.get("semantic_name", ""))) and f(row.get("O_scale_repaired")) >= o_q50
        )
        unreliable_or_low_o = sum(
            1
            for row in anchors
            if is_unreliable(str(row.get("semantic_name", ""))) or f(row.get("O_scale_repaired")) <= o_q25
        )
        raw_join = sum(1 for row in lifecycle if b(row.get("raw_geometry_edge_joined")))
        geometry_join = sum(1 for row in lifecycle if b(row.get("geometry_joined")))
        out.append(
            {
                "case_id": case_id,
                "seq": seq_of(case_id),
                "target_taxonomy": tax,
                "L3_handoff_transfer_penalty_proxy": meta.get("L3_handoff_transfer_penalty_proxy")
                or stagec.get("L3_handoff_transfer_penalty_proxy")
                or (anchors[0].get("L3_handoff_transfer_penalty_proxy") if anchors else ""),
                "anchor_row_count": n,
                "semantic_stable_anchor_frac": stable_count / n if n else math.nan,
                "semantic_unreliable_anchor_frac": unreliable_count / n if n else math.nan,
                "stable_observable_frac_q50": stable_observable / n if n else math.nan,
                "stable_observable_topk_mean": top_fraction_mean(stable_o, 0.25),
                "unreliable_or_low_observable_frac": unreliable_or_low_o / n if n else math.nan,
                "O_scale_mean": mean(o_vals),
                "O_scale_top25_mean": top_fraction_mean(o_vals, 0.25),
                "anchor_point_count_mean": mean([row.get("anchor_point_count") for row in anchors]),
                "P_parallax_score_mean": mean([row.get("P_parallax_score") for row in anchors]),
                "D_spread_score_mean": mean([row.get("D_spread_score") for row in anchors]),
                "true_geometry_source_frac": mean([1.0 if b(row.get("true_geometry_source_available")) else 0.0 for row in anchors]),
                "semantic_unique_name_count": len({name for name in names if name}),
                "dominant_semantic_name_frac": (max(name_counts.values()) / n) if n and name_counts else math.nan,
                "lifecycle_anchor_count": len(lifecycle),
                "lifecycle_geometry_unique_coverage": meta.get("combined_geometry_unique_coverage", ""),
                "lifecycle_raw_geometry_join_frac": raw_join / len(lifecycle) if lifecycle else math.nan,
                "lifecycle_geometry_join_frac": geometry_join / len(lifecycle) if lifecycle else math.nan,
                "source_stage_c_seed_count_mean": mean([row.get("source_stage_c_seed_count") for row in lifecycle]),
                "stage_c_same_payload_join_coverage": stagec.get("lifecycle_geometry_same_payload_join_coverage", ""),
                "local_scale_mode_entropy": stagec.get("local_scale_mode_entropy", ""),
                "abs_log_depth_ratio_mean": stagec.get("abs_log_depth_ratio_mean", ""),
                "claim_level": "v102_full_control_semantic_rotation_case_diagnostic_no_action",
            }
        )
    return out


def select(values: list[float], threshold: float, direction: str) -> list[bool]:
    if direction == "high":
        return [math.isfinite(value) and value >= threshold for value in values]
    return [math.isfinite(value) and value <= threshold for value in values]


def score(pred: list[bool], positives: list[bool], good_controls: list[bool], seqs: list[str]) -> tuple[float, float, int, int, str, str, str]:
    pos_n = sum(positives)
    good_n = sum(good_controls)
    tp_cases = []
    fp_cases = []
    selected = []
    for chosen, pos, good, seq in zip(pred, positives, good_controls, seqs):
        _ = seq
        if chosen:
            selected.append("")
        if chosen and pos:
            tp_cases.append("")
        if chosen and good:
            fp_cases.append("")
    tp = sum(1 for chosen, pos in zip(pred, positives) if chosen and pos)
    fp_good = sum(1 for chosen, good in zip(pred, good_controls) if chosen and good)
    recall = tp / pos_n if pos_n else math.nan
    fpr = fp_good / good_n if good_n else math.nan
    seq_cov = len({seq for chosen, pos, seq in zip(pred, positives, seqs) if chosen and pos and seq})
    return recall, fpr, seq_cov, sum(pred), "", "", ""


def balanced_accuracy(recall: float, good_fpr: float) -> float:
    if not math.isfinite(recall) or not math.isfinite(good_fpr):
        return math.nan
    return 0.5 * (recall + (1.0 - good_fpr))


def scope_labels(rows: list[dict[str, Any]], scope: str) -> tuple[list[dict[str, Any]], list[bool], list[bool], str]:
    if scope == "clean_handoff_vs_safe":
        scoped = [row for row in rows if row.get("target_taxonomy") in {POS_HANDOFF, SAFE_GOOD}]
        positives = [row.get("target_taxonomy") == POS_HANDOFF for row in scoped]
        good = [row.get("target_taxonomy") == SAFE_GOOD for row in scoped]
        return scoped, positives, good, "strict clean handoff target vs SAFE_GOOD controls"
    if scope == "expanded_high_l3_non_good_vs_safe":
        vals = [f(row.get("L3_handoff_transfer_penalty_proxy")) for row in rows if row.get("target_taxonomy") != SAFE_GOOD]
        q75 = quantile(vals, 0.75)
        scoped = [row for row in rows if row.get("target_taxonomy") == SAFE_GOOD or f(row.get("L3_handoff_transfer_penalty_proxy")) >= q75]
        positives = [row.get("target_taxonomy") != SAFE_GOOD and f(row.get("L3_handoff_transfer_penalty_proxy")) >= q75 for row in scoped]
        good = [row.get("target_taxonomy") == SAFE_GOOD for row in scoped]
        return scoped, positives, good, f"exploratory high-L3 non-good q75={q75}"
    if scope == "handoff_multimode_contaminated_vs_safe":
        pos_tax = {POS_HANDOFF, "MULTIMODE_LOWOBS_ABSTAIN", "GOOD_HIGH_L3_CONTAMINATED"}
        scoped = [row for row in rows if row.get("target_taxonomy") in pos_tax | {SAFE_GOOD}]
        positives = [row.get("target_taxonomy") in pos_tax for row in scoped]
        good = [row.get("target_taxonomy") == SAFE_GOOD for row in scoped]
        return scoped, positives, good, "exploratory handoff/multimode/contaminated risk vs SAFE_GOOD"
    if scope == "local_bad_vs_safe":
        scoped = [row for row in rows if row.get("target_taxonomy") in {"LOCAL_BAD_NOT_HANDOFF", SAFE_GOOD}]
        positives = [row.get("target_taxonomy") == "LOCAL_BAD_NOT_HANDOFF" for row in scoped]
        good = [row.get("target_taxonomy") == SAFE_GOOD for row in scoped]
        return scoped, positives, good, "exploratory local-bad vs SAFE_GOOD"
    raise ValueError(scope)


def score_policy_on_rows(policy: dict[str, Any], rows: list[dict[str, Any]], scope: str) -> tuple[float, float, float, int]:
    scoped, positives, good, _ = scope_labels(rows, scope)
    scoped_order = [dict(row, _positive=pos, _good_control=ctrl) for row, pos, ctrl in zip(scoped, positives, good)]
    by_case = {row["case_id"]: row for row in scoped_order}
    selected_cases = [case for case in str(policy.get("scoped_case_order", "")).split(";") if case]
    ordered = [by_case[case] for case in selected_cases if case in by_case]
    if not ordered:
        ordered = scoped_order
    values = [f(row.get(str(policy["score_field"]))) for row in ordered]
    pred = select(values, f(policy["threshold"]), str(policy["direction"]))
    positives = [row.get("_positive", False) for row in ordered]
    good = [row.get("_good_control", False) for row in ordered]
    seqs = [str(row.get("seq", "")) for row in ordered]
    recall, good_fpr, seq_cov, _selected_count, _, _, _ = score(pred, positives, good, seqs)
    return balanced_accuracy(recall, good_fpr), recall, good_fpr, seq_cov


def control_margin(
    *,
    policy: dict[str, Any],
    observed_ba: float,
    rows: list[dict[str, Any]],
    anchor_rows: list[dict[str, str]],
    lifecycle_rows: list[dict[str, str]],
    case_rows: list[dict[str, str]],
    stagec_case_rows: list[dict[str, str]],
    control: str,
    semantic_rotations: list[list[dict[str, Any]]] | None = None,
    anchor_geometry_rotations: list[list[dict[str, Any]]] | None = None,
    trials: int = 128,
) -> tuple[float, float]:
    if not math.isfinite(observed_ba):
        return math.nan, math.nan
    score_field = str(policy.get("score_field", ""))
    bas: list[float] = []
    if control == "same_count_random":
        scoped, positives, good, _ = scope_labels(rows, str(policy["eval_scope"]))
        selected_count = int(f(policy.get("selected_count"), 0))
        n = len(scoped)
        if selected_count <= 0 or selected_count > n:
            return math.nan, math.nan
        rng = deterministic_rng(control, str(policy["eval_scope"]), score_field, str(selected_count))
        indices = list(range(n))
        seqs = [str(row.get("seq", "")) for row in scoped]
        for _ in range(trials):
            chosen = set(rng.sample(indices, selected_count))
            pred = [idx in chosen for idx in indices]
            recall, good_fpr, _, _, _, _, _ = score(pred, positives, good, seqs)
            bas.append(balanced_accuracy(recall, good_fpr))
    elif control == "semantic_label_rotation":
        if score_field not in SEMANTIC_FEATURES:
            return math.nan, math.nan
        rotations = semantic_rotations or [
            build_case_features(
                anchor_rows=anchor_rows,
                lifecycle_rows=lifecycle_rows,
                case_rows=case_rows,
                stagec_case_rows=stagec_case_rows,
                semantic_rotation_trial=trial,
            )
            for trial in range(trials)
        ]
        for rotated_rows in rotations:
            ba, _, _, _ = score_policy_on_rows(policy, rotated_rows, str(policy["eval_scope"]))
            bas.append(ba)
    elif control == "anchor_id_geometry_rotation":
        if score_field not in ANCHOR_GEOMETRY_FEATURES:
            return math.nan, math.nan
        rotations = anchor_geometry_rotations or [
            build_case_features(
                anchor_rows=anchor_rows,
                lifecycle_rows=lifecycle_rows,
                case_rows=case_rows,
                stagec_case_rows=stagec_case_rows,
                anchor_geometry_rotation_trial=trial,
            )
            for trial in range(trials)
        ]
        for rotated_rows in rotations:
            ba, _, _, _ = score_policy_on_rows(policy, rotated_rows, str(policy["eval_scope"]))
            bas.append(ba)
    else:
        raise ValueError(control)
    mean_ba = mean(bas)
    return observed_ba - mean_ba if math.isfinite(mean_ba) else math.nan, mean_ba


def evaluate_policies(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for scope in [
        "clean_handoff_vs_safe",
        "expanded_high_l3_non_good_vs_safe",
        "handoff_multimode_contaminated_vs_safe",
        "local_bad_vs_safe",
    ]:
        scoped, positives, good_controls, desc = scope_labels(rows, scope)
        scoped_order = [dict(row, _positive=pos, _good_control=good) for row, pos, good in zip(scoped, positives, good_controls)]
        seqs = [str(row.get("seq", "")) for row in scoped_order]
        for feature, direction, feature_desc in FEATURES:
            values = [f(row.get(feature)) for row in scoped_order]
            thresholds = sorted({value for value in values if math.isfinite(value)})
            best: dict[str, Any] | None = None
            for threshold in thresholds:
                pred = select(values, threshold, direction)
                recall, good_fpr, seq_cov, selected_count, _, _, _ = score(pred, positives, good_controls, seqs)
                ba = balanced_accuracy(recall, good_fpr)
                row = {
                    "eval_scope": scope,
                    "eval_scope_description": desc,
                    "score_field": feature,
                    "feature_description": feature_desc,
                    "direction": direction,
                    "threshold": threshold,
                    "eval_case_count": len(scoped_order),
                    "positive_count": sum(positives),
                    "good_control_count": sum(good_controls),
                    "selected_count": selected_count,
                    "selected_cases": ";".join(row["case_id"] for row, chosen in zip(scoped_order, pred) if chosen),
                    "true_positive_cases": ";".join(row["case_id"] for row, chosen, pos in zip(scoped_order, pred, positives) if chosen and pos),
                    "good_false_positive_cases": ";".join(row["case_id"] for row, chosen, good in zip(scoped_order, pred, good_controls) if chosen and good),
                    "bad_recall": recall,
                    "good_FPR": good_fpr,
                    "sequence_coverage": seq_cov,
                    "balanced_accuracy": ba,
                    "scoped_case_order": ";".join(row["case_id"] for row in scoped_order),
                }
                if best is None:
                    best = row
                else:
                    key = (f(row["balanced_accuracy"]), f(row["bad_recall"]), -f(row["good_FPR"]), f(row["sequence_coverage"]))
                    best_key = (f(best["balanced_accuracy"]), f(best["bad_recall"]), -f(best["good_FPR"]), f(best["sequence_coverage"]))
                    if key > best_key:
                        best = row
            if best is not None:
                out.append(best)
    return out


def add_controls(
    policy_rows: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    anchor_rows: list[dict[str, str]],
    lifecycle_rows: list[dict[str, str]],
    case_rows: list[dict[str, str]],
    stagec_case_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    controls: list[dict[str, Any]] = []
    trials = 128
    needs_semantic = any(str(row.get("score_field", "")) in SEMANTIC_FEATURES for row in policy_rows)
    needs_anchor_geometry = any(str(row.get("score_field", "")) in ANCHOR_GEOMETRY_FEATURES for row in policy_rows)
    semantic_rotations = (
        [
            build_case_features(
                anchor_rows=anchor_rows,
                lifecycle_rows=lifecycle_rows,
                case_rows=case_rows,
                stagec_case_rows=stagec_case_rows,
                semantic_rotation_trial=trial,
            )
            for trial in range(trials)
        ]
        if needs_semantic
        else []
    )
    anchor_geometry_rotations = (
        [
            build_case_features(
                anchor_rows=anchor_rows,
                lifecycle_rows=lifecycle_rows,
                case_rows=case_rows,
                stagec_case_rows=stagec_case_rows,
                anchor_geometry_rotation_trial=trial,
            )
            for trial in range(trials)
        ]
        if needs_anchor_geometry
        else []
    )
    for row in policy_rows:
        observed = f(row.get("balanced_accuracy"))
        for control in ["same_count_random", "semantic_label_rotation", "anchor_id_geometry_rotation"]:
            margin, control_ba = control_margin(
                policy=row,
                observed_ba=observed,
                rows=rows,
                anchor_rows=anchor_rows,
                lifecycle_rows=lifecycle_rows,
                case_rows=case_rows,
                stagec_case_rows=stagec_case_rows,
                control=control,
                semantic_rotations=semantic_rotations,
                anchor_geometry_rotations=anchor_geometry_rotations,
                trials=trials,
            )
            row[f"{control}_balanced_accuracy"] = control_ba
            row[f"{control}_margin"] = margin
            controls.append(
                {
                    "eval_scope": row["eval_scope"],
                    "score_field": row["score_field"],
                    "control": control,
                    "observed_balanced_accuracy": observed,
                    "control_balanced_accuracy": control_ba,
                    "control_margin": margin,
                    "control_applicable": math.isfinite(control_ba),
                    "claim_level": "v102_full_control_rotation_diagnostic_no_action",
                }
            )
        same_margin = f(row.get("same_count_random_margin"))
        sem_margin = f(row.get("semantic_label_rotation_margin"))
        anchor_margin = f(row.get("anchor_id_geometry_rotation_margin"))
        sem_ok = (not row["score_field"] in SEMANTIC_FEATURES) or (math.isfinite(sem_margin) and sem_margin >= 0.05)
        anchor_ok = (not row["score_field"] in ANCHOR_GEOMETRY_FEATURES) or (math.isfinite(anchor_margin) and anchor_margin >= 0.05)
        row["full_control_oracle_pass"] = (
            f(row.get("positive_count")) >= 3
            and f(row.get("bad_recall")) >= 0.65
            and f(row.get("good_FPR")) <= 0.25
            and f(row.get("sequence_coverage")) >= 2
            and math.isfinite(same_margin)
            and same_margin >= 0.05
            and sem_ok
            and anchor_ok
        )
        row["strict_promotion_pass"] = bool(row["full_control_oracle_pass"] and row["eval_scope"] == "clean_handoff_vs_safe" and f(row.get("sequence_coverage")) >= 3)
        row["claim_level"] = "v102_full_control_semantic_rotation_policy_diagnostic_no_action"
    return controls


def main() -> int:
    raw_anchor_rows = read_rows(PER_ANCHOR_ROWS)
    anchor_rows: list[dict[str, str]] = []
    for row in raw_anchor_rows:
        rec = dict(row)
        rec["semantic_name"] = semantic_name(str(row.get("case_id", "")), row.get("semantic_label", ""))
        anchor_rows.append(rec)
    lifecycle_rows = read_rows(LIFECYCLE_JOIN_ROWS)
    case_rows = read_rows(CASE_ROWS)
    stagec_case_rows = read_rows(STAGE_C_CASE_ROWS)
    case_feature_rows = build_case_features(
        anchor_rows=anchor_rows,
        lifecycle_rows=lifecycle_rows,
        case_rows=case_rows,
        stagec_case_rows=stagec_case_rows,
    )
    policy_rows = evaluate_policies(case_feature_rows)
    control_rows = add_controls(policy_rows, case_feature_rows, anchor_rows, lifecycle_rows, case_rows, stagec_case_rows)
    policy_rows = sorted(
        policy_rows,
        key=lambda row: (
            not bool(row.get("strict_promotion_pass")),
            not bool(row.get("full_control_oracle_pass")),
            -f(row.get("balanced_accuracy")),
            str(row.get("eval_scope")),
            str(row.get("score_field")),
        ),
    )
    best = policy_rows[0] if policy_rows else {}
    strict_passes = [row for row in policy_rows if row.get("strict_promotion_pass")]
    exploratory_passes = [row for row in policy_rows if row.get("full_control_oracle_pass")]
    clean_scope_rows, clean_pos, clean_good, _ = scope_labels(case_feature_rows, "clean_handoff_vs_safe")
    summary = {
        "schema": "acl2_v102_stage3_full_control_semantic_rotation_v1",
        "diagnostic_only": True,
        "runtime_action_allowed": False,
        "input_per_anchor_row_count": len(anchor_rows),
        "input_case_count": len(case_feature_rows),
        "clean_handoff_eval_case_count": len(clean_scope_rows),
        "clean_handoff_positive_count": sum(clean_pos),
        "clean_handoff_safe_good_count": sum(clean_good),
        "policy_count": len(policy_rows),
        "control_row_count": len(control_rows),
        "full_control_oracle_pass_count": len(exploratory_passes),
        "strict_promotion_pass_count": len(strict_passes),
        "strict_semantic_oracle_pass": False,
        "exploratory_control_signal_present": bool(exploratory_passes),
        "best_policy": best,
        "blockers": [
            "strict clean handoff-vs-safe universe has fewer than 3 positive handoff cases",
            "this audit is offline diagnostic-only and does not measure L2/L3 improvement from a memory action surface",
            "Stage4/5/6/7 remain unauthorized unless Stage3 strict and Stage4 action-surface gates pass",
        ],
        "outputs": {
            "case_rows": CASE_OUT.as_posix(),
            "policy_rows": POLICY_OUT.as_posix(),
            "control_rows": CONTROL_OUT.as_posix(),
            "report": REPORT_OUT.as_posix(),
        },
    }
    write_rows(CASE_OUT, case_feature_rows)
    write_rows(POLICY_OUT, policy_rows)
    write_rows(CONTROL_OUT, control_rows)
    write_json(SUMMARY_OUT, summary)
    write_text(
        REPORT_OUT,
        "\n".join(
            [
                "# Stage3 Full-Control Semantic Rotation Audit",
                "",
                f"- input_per_anchor_row_count: {summary['input_per_anchor_row_count']}",
                f"- input_case_count: {summary['input_case_count']}",
                f"- clean_handoff_positive_count: {summary['clean_handoff_positive_count']}",
                f"- clean_handoff_safe_good_count: {summary['clean_handoff_safe_good_count']}",
                f"- full_control_oracle_pass_count: {summary['full_control_oracle_pass_count']}",
                f"- strict_promotion_pass_count: {summary['strict_promotion_pass_count']}",
                f"- strict_semantic_oracle_pass: {summary['strict_semantic_oracle_pass']}",
                f"- runtime_action_allowed: {summary['runtime_action_allowed']}",
                "",
                "Best policy:",
                "",
                "```json",
                json.dumps(jsonable(best), indent=2, sort_keys=True),
                "```",
                "",
                "Blockers:",
                "",
                "\n".join(f"- {item}" for item in summary["blockers"]),
            ]
        ),
    )
    print(json.dumps(jsonable(summary), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
