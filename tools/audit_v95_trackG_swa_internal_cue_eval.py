#!/usr/bin/env python3
"""Evaluate v83/v85 internal SWA cues on the v95 Track G labelled universe.

This script is a cue-bank audit only. It does not authorize runtime action.
The method-safe scope uses numeric internal features from v83 and v85 Q/K
artifacts. Support-class/context-tag cues are reported separately because their
provenance includes diagnostic/semantic construction choices.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd
import torch


ROOT = Path("results/acl2_v95tf_multiroute_semantic_memory_evidence_control")
DEFAULT_EVAL_ROWS = ROOT / "trackG_swa_handoff_cue_bank_v2/evaluation_rows.csv"
DEFAULT_V83_CLUE_MATRIX = (
    Path("results/acl2_v83tf_clue_sufficiency_vs_action_misuse")
    / "phase1_unified_clue_matrix/unified_clue_matrix.csv"
)
DEFAULT_V85_QK_INDEX = (
    Path("results/acl2_v85tf_latent_anchor_alignment_pairwise_memory_ruler")
    / "phase2_qk_feature_bank/qk_anchor_feature_index.csv"
)
DEFAULT_V85_QK_FEATURES = DEFAULT_V85_QK_INDEX.with_name("qk_anchor_features.pt")
DEFAULT_OUT_DIR = ROOT / "trackG_swa_internal_cue_eval_v1"

V83_SCOPE_RANK = {
    "v82_swa_adjacent_pair": 0,
    "v80_mid_adjacent_pair": 1,
    "v80_long_window": 2,
    "v80_short_read_case": 3,
}

V83_NUMERIC_COLUMNS = [
    "QK_pair_compatibility",
    "query_risk_mass",
    "read_entropy",
    "current_Q_alignment",
    "cache_K_alignment",
    "cache_V_alignment",
    "K_risk_delta",
    "V_protect_delta",
    "route_mass",
    "head_layer_sensitivity",
    "actual_vs_random_route_delta",
    "hmc_route_mass",
    "G5_route_mass",
]

V85_NUMERIC_PREFIXES = [
    "v85_anchor_row_count",
    "v85_q_norm_",
    "v85_k_norm_",
    "v85_qk_cosine_",
    "v85_qk_l2_",
    "v85_qk_dot_",
    "v85_qk_absdiff_",
    "v85_qk_norm_absdiff_",
    "v85_qk_norm_ratio_",
    "v85_route_mass_available_frac",
]

CONTROL_THRESHOLDS = {
    "bad_recall": 0.60,
    "good_FPR": 0.25,
    "positive_sequence_coverage": 2,
    "control_margin": 0.05,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-rows", type=Path, default=DEFAULT_EVAL_ROWS)
    parser.add_argument("--v83-clue-matrix", type=Path, default=DEFAULT_V83_CLUE_MATRIX)
    parser.add_argument("--v85-qk-index", type=Path, default=DEFAULT_V85_QK_INDEX)
    parser.add_argument("--v85-qk-features", type=Path, default=DEFAULT_V85_QK_FEATURES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--random-seeds", type=int, default=128)
    parser.add_argument("--max-method-control-candidates", type=int, default=120)
    parser.add_argument("--max-combined-control-candidates", type=int, default=80)
    return parser.parse_args()


def serialize(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.12g}"
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: serialize(row.get(key, "")) for key in fields})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def f(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def safe_int(value: Any) -> int | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def pair_key(seq: Any, prev_chunk: Any, curr_chunk: Any) -> str:
    seq_i = safe_int(seq)
    prev_i = safe_int(prev_chunk)
    curr_i = safe_int(curr_chunk)
    if seq_i is None or prev_i is None or curr_i is None:
        return ""
    return f"{seq_i:02d}_{prev_i:03d}_{curr_i:03d}"


def safe_name(value: Any) -> str:
    text = str(value).strip()
    out = []
    for ch in text:
        out.append(ch if ch.isalnum() else "_")
    return "".join(out).strip("_") or "EMPTY"


def stable_unit(*parts: Any) -> float:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12)


def percentile(values: Iterable[float], pct: float) -> float | None:
    finite = sorted(v for v in values if math.isfinite(v))
    if not finite:
        return None
    pos = (len(finite) - 1) * pct
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(finite[lo])
    return float(finite[lo] * (hi - pos) + finite[hi] * (pos - lo))


def mean(values: list[float]) -> float | None:
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return None
    return float(sum(finite) / len(finite))


def std(values: list[float]) -> float | None:
    finite = [v for v in values if math.isfinite(v)]
    if len(finite) < 2:
        return 0.0 if finite else None
    m = sum(finite) / len(finite)
    return float(math.sqrt(sum((v - m) ** 2 for v in finite) / len(finite)))


def stat_bundle(prefix: str, values: list[float]) -> dict[str, Any]:
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return {
            f"{prefix}_count": 0,
            f"{prefix}_mean": None,
            f"{prefix}_median": None,
            f"{prefix}_std": None,
            f"{prefix}_p25": None,
            f"{prefix}_p75": None,
            f"{prefix}_p90": None,
            f"{prefix}_min": None,
            f"{prefix}_max": None,
        }
    return {
        f"{prefix}_count": len(finite),
        f"{prefix}_mean": mean(finite),
        f"{prefix}_median": percentile(finite, 0.50),
        f"{prefix}_std": std(finite),
        f"{prefix}_p25": percentile(finite, 0.25),
        f"{prefix}_p75": percentile(finite, 0.75),
        f"{prefix}_p90": percentile(finite, 0.90),
        f"{prefix}_min": min(finite),
        f"{prefix}_max": max(finite),
    }


def torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_qk_vectors(path: Path, expected_rows: int) -> tuple[list[dict[str, float]], dict[str, Any]]:
    meta: dict[str, Any] = {
        "v85_qk_features_path": str(path),
        "v85_qk_feature_vector_loaded": False,
        "v85_qk_feature_vector_error": "",
    }
    if not path.exists():
        meta["v85_qk_feature_vector_error"] = "file_missing"
        return [], meta
    try:
        payload = torch_load(path)
    except Exception as exc:  # noqa: BLE001
        meta["v85_qk_feature_vector_error"] = f"{type(exc).__name__}: {exc}"
        return [], meta
    if not isinstance(payload, dict) or "q_features" not in payload or "k_features" not in payload:
        meta["v85_qk_feature_vector_error"] = "schema_missing_q_features_or_k_features"
        return [], meta
    q = payload["q_features"].detach().cpu().float()
    k = payload["k_features"].detach().cpu().float()
    meta["v85_qk_feature_vector_loaded"] = True
    meta["v85_qk_feature_schema"] = payload.get("schema", "")
    meta["v85_q_feature_shape"] = list(q.shape)
    meta["v85_k_feature_shape"] = list(k.shape)
    meta["v85_qk_index_row_count_expected"] = expected_rows
    if q.shape[0] != expected_rows or k.shape[0] != expected_rows:
        meta["v85_qk_feature_vector_error"] = "row_count_mismatch"
        return [], meta
    q_norm = torch.linalg.norm(q, dim=1)
    k_norm = torch.linalg.norm(k, dim=1)
    dot = (q * k).sum(dim=1)
    cosine = dot / torch.clamp(q_norm * k_norm, min=1e-12)
    l2 = torch.linalg.norm(q - k, dim=1)
    absdiff = torch.mean(torch.abs(q - k), dim=1)
    rows: list[dict[str, float]] = []
    for idx in range(expected_rows):
        kn = float(k_norm[idx].item())
        qn = float(q_norm[idx].item())
        rows.append(
            {
                "qk_cosine": float(cosine[idx].item()),
                "qk_l2": float(l2[idx].item()),
                "qk_dot": float(dot[idx].item()),
                "qk_absdiff": float(absdiff[idx].item()),
                "qk_norm_absdiff": abs(qn - kn),
                "qk_norm_ratio": qn / kn if kn > 1e-12 else float("nan"),
            }
        )
    return rows, meta


def load_v83_pair_rows(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    frame = pd.read_csv(path)
    records = frame.to_dict(orient="records")
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        key = pair_key(row.get("seq"), row.get("prev_chunk"), row.get("curr_chunk"))
        if key:
            by_pair[key].append(row)
    selected: dict[str, dict[str, Any]] = {}
    for key, rows in by_pair.items():
        rows = sorted(
            rows,
            key=lambda row: (
                V83_SCOPE_RANK.get(str(row.get("row_scope") or ""), 99),
                str(row.get("row_id") or ""),
            ),
        )
        row = rows[0]
        out: dict[str, Any] = {
            "v83_row_id": row.get("row_id"),
            "v83_row_scope": row.get("row_scope"),
            "v83_case_type": row.get("case_type"),
            "v83_target_label": row.get("target_label"),
            "v83_duplicate_scope_count": len(rows),
            "v83_selected_scope_rank": V83_SCOPE_RANK.get(str(row.get("row_scope") or ""), 99),
        }
        for column in V83_NUMERIC_COLUMNS:
            value = f(row.get(column))
            out[f"v83_{column}"] = value if math.isfinite(value) else None
            out[f"v83_{column}_available"] = math.isfinite(value)
        selected[key] = out
    return selected, {
        "v83_source_path": str(path),
        "v83_source_row_count": len(records),
        "v83_source_pair_count": len(by_pair),
    }


def aggregate_v85(path: Path, feature_path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    frame = pd.read_csv(path)
    records = frame.to_dict(orient="records")
    vector_rows, vector_meta = load_qk_vectors(feature_path, len(records))
    by_pair: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(records):
        key = pair_key(row.get("seq"), row.get("prev_chunk"), row.get("curr_chunk"))
        if key:
            by_pair[key].append(idx)
    support_classes = sorted(
        {str(row.get("anchor_support_class") or "") for row in records if str(row.get("anchor_support_class") or "")}
    )
    aggregated: dict[str, dict[str, Any]] = {}
    for key, indices in by_pair.items():
        class_counts = Counter(str(records[idx].get("anchor_support_class") or "") for idx in indices)
        out: dict[str, Any] = {
            "v85_anchor_row_count": len(indices),
            "v85_anchor_support_class_count": len([klass for klass, count in class_counts.items() if klass and count]),
            "v85_route_mass_available_frac": mean(
                [1.0 if str(records[idx].get("route_mass_available") or "").lower() == "true" else 0.0 for idx in indices]
            ),
        }
        out.update(stat_bundle("v85_q_norm", [f(records[idx].get("q_norm")) for idx in indices]))
        out.update(stat_bundle("v85_k_norm", [f(records[idx].get("k_norm")) for idx in indices]))
        if vector_rows:
            for column in ["qk_cosine", "qk_l2", "qk_dot", "qk_absdiff", "qk_norm_absdiff", "qk_norm_ratio"]:
                out.update(stat_bundle(f"v85_{column}", [vector_rows[idx][column] for idx in indices]))
        for klass in support_classes:
            count = class_counts.get(klass, 0)
            out[f"v85_frac_{safe_name(klass)}"] = count / len(indices) if indices else None
            out[f"v85_count_{safe_name(klass)}"] = count
        aggregated[key] = out
    meta = {
        "v85_qk_index_path": str(path),
        "v85_qk_index_row_count": len(records),
        "v85_qk_index_pair_count": len(by_pair),
        "v85_anchor_support_classes": support_classes,
        **vector_meta,
    }
    return aggregated, meta


def build_eval_rows(
    eval_path: Path,
    v83_rows: Mapping[str, Mapping[str, Any]],
    v85_rows: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    frame = pd.read_csv(eval_path)
    rows: list[dict[str, Any]] = []
    v83_matches: set[str] = set()
    v85_matches: set[str] = set()
    for row in frame.to_dict(orient="records"):
        label = str(row.get("cue_eval_label") or "")
        if label not in {"positive", "negative"}:
            continue
        key = pair_key(row.get("seq"), row.get("prev_chunk"), row.get("curr_chunk"))
        out = dict(row)
        out["canonical_pair_key"] = key
        if key in v83_rows:
            out.update(v83_rows[key])
            v83_matches.add(key)
        if key in v85_rows:
            out.update(v85_rows[key])
            v85_matches.add(key)
        rows.append(out)
    meta = {
        "evaluation_rows_path": str(eval_path),
        "labelled_universe_count": len(rows),
        "positive_count": sum(1 for row in rows if row.get("cue_eval_label") == "positive"),
        "negative_count": sum(1 for row in rows if row.get("cue_eval_label") == "negative"),
        "v83_matched_pair_count": len(v83_matches),
        "v85_matched_pair_count": len(v85_matches),
    }
    return rows, meta


def numeric_feature_columns(rows: list[Mapping[str, Any]]) -> tuple[list[str], list[str]]:
    method_cols: list[str] = []
    combined_cols: list[str] = []
    v83_method_names = {f"v83_{column}" for column in V83_NUMERIC_COLUMNS}
    if not rows:
        return method_cols, combined_cols
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    for column in fields:
        values = [f(row.get(column)) for row in rows]
        if not any(math.isfinite(value) for value in values):
            continue
        is_method = column in v83_method_names
        is_method = is_method or any(column.startswith(prefix) for prefix in V85_NUMERIC_PREFIXES)
        is_support = column.startswith("v85_frac_") or column.startswith("v85_count_")
        if is_method and not is_support:
            method_cols.append(column)
        elif is_support:
            combined_cols.append(column)
    return sorted(method_cols), sorted(combined_cols)


def atom_thresholds(rows: list[Mapping[str, Any]], columns: list[str]) -> tuple[dict[str, list[bool]], dict[str, float]]:
    atoms: dict[str, list[bool]] = {}
    thresholds: dict[str, float] = {}
    for column in columns:
        values = [f(row.get(column)) for row in rows]
        finite = [value for value in values if math.isfinite(value)]
        if not finite:
            continue
        for quantile in [0.25, 0.50, 0.60, 0.75, 0.85]:
            suffix = int(quantile * 100)
            threshold = percentile(finite, quantile)
            if threshold is None:
                continue
            key = f"{column}_q{suffix}"
            thresholds[key] = threshold
            atoms[f"{column.upper()}_GE_Q{suffix}"] = [
                math.isfinite(value) and value >= threshold for value in values
            ]
            atoms[f"{column.upper()}_LE_Q{suffix}"] = [
                math.isfinite(value) and value <= threshold for value in values
            ]
    return atoms, thresholds


def bool_tag_atoms(rows: list[Mapping[str, Any]], columns: list[str]) -> dict[str, list[bool]]:
    atoms: dict[str, list[bool]] = {}
    for column in columns:
        values = sorted({str(row.get(column) or "") for row in rows if str(row.get(column) or "")})
        for value in values:
            atoms[f"{column.upper()}_EQ_{safe_name(value)}"] = [str(row.get(column) or "") == value for row in rows]
    return atoms


def and_mask(left: list[bool], right: list[bool]) -> list[bool]:
    return [a and b for a, b in zip(left, right)]


def mask_key(mask: list[bool]) -> str:
    return "".join("1" if value else "0" for value in mask)


def candidate_masks(atoms: Mapping[str, list[bool]], include_pairs: bool = True) -> dict[str, list[bool]]:
    out: dict[str, list[bool]] = {}
    seen_masks: set[str] = set()
    for name, mask in sorted(atoms.items()):
        if not any(mask):
            continue
        key = mask_key(mask)
        if key in seen_masks:
            continue
        out[name] = mask
        seen_masks.add(key)
    if include_pairs:
        singles = sorted(out.items())
        for (left_name, left_mask), (right_name, right_mask) in combinations(singles, 2):
            mask = and_mask(left_mask, right_mask)
            if not any(mask):
                continue
            key = mask_key(mask)
            if key in seen_masks:
                continue
            out[f"{left_name}__AND__{right_name}"] = mask
            seen_masks.add(key)
    return out


def balanced_metrics(rows: list[Mapping[str, Any]], mask: list[bool], include_hits: bool = True) -> dict[str, Any]:
    positives = [str(row.get("cue_eval_label")) == "positive" for row in rows]
    negatives = [str(row.get("cue_eval_label")) == "negative" for row in rows]
    selected = [idx for idx, value in enumerate(mask) if value]
    positive_selected = [idx for idx in selected if positives[idx]]
    negative_selected = [idx for idx in selected if negatives[idx]]
    positive_total = sum(positives)
    negative_total = sum(negatives)
    recall = len(positive_selected) / max(positive_total, 1)
    fpr = len(negative_selected) / max(negative_total, 1)
    out: dict[str, Any] = {
        "selected_count": len(selected),
        "selected_positive_count": len(positive_selected),
        "selected_negative_count": len(negative_selected),
        "positive_total": positive_total,
        "negative_total": negative_total,
        "bad_recall": recall,
        "good_FPR": fpr,
        "balanced_accuracy": (recall + (1.0 - fpr)) / 2.0,
        "positive_sequence_coverage": len({str(rows[idx].get("seq")) for idx in positive_selected}),
        "selected_sequence_coverage": len({str(rows[idx].get("seq")) for idx in selected}) if selected else 0,
    }
    if include_hits:
        out.update(
            {
                "selected_pair_ids": ",".join(str(rows[idx].get("pair_id")) for idx in selected),
                "selected_positive_pair_ids": ",".join(str(rows[idx].get("pair_id")) for idx in positive_selected),
                "selected_negative_pair_ids": ",".join(str(rows[idx].get("pair_id")) for idx in negative_selected),
            }
        )
    return out


def random_mask(rows: list[Mapping[str, Any]], count: int, seed: int) -> list[bool]:
    ordered = sorted(range(len(rows)), key=lambda idx: stable_unit("global", seed, rows[idx].get("pair_id")))
    selected = set(ordered[: min(count, len(ordered))])
    return [idx in selected for idx in range(len(rows))]


def seq_count_random_mask(rows: list[Mapping[str, Any]], mask: list[bool], seed: int) -> list[bool]:
    by_seq: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        by_seq[str(row.get("seq"))].append(idx)
    selected: set[int] = set()
    for seq, indices in by_seq.items():
        count = sum(mask[idx] for idx in indices)
        ordered = sorted(indices, key=lambda idx: stable_unit("seq", seed, seq, rows[idx].get("pair_id")))
        selected.update(ordered[: min(count, len(indices))])
    return [idx in selected for idx in range(len(rows))]


def rotate(values: list[bool], amount: int) -> list[bool]:
    if not values:
        return []
    amount %= len(values)
    return values[amount:] + values[:amount]


def internal_rotation_masks(rows: list[Mapping[str, Any]], mask: list[bool]) -> list[list[bool]]:
    ordered = sorted(
        range(len(rows)),
        key=lambda idx: (
            str(rows[idx].get("v83_row_scope") or ""),
            f(rows[idx].get("v85_anchor_row_count"), -1.0),
            f(rows[idx].get("v85_qk_cosine_median"), -999.0),
            str(rows[idx].get("pair_id") or ""),
        ),
    )
    values = [mask[idx] for idx in ordered]
    controls: list[list[bool]] = []
    for amount in range(1, len(values)):
        rotated = rotate(values, amount)
        out = [False] * len(mask)
        for idx, value in zip(ordered, rotated):
            out[idx] = value
        controls.append(out)
    return controls


def evaluate_candidate(
    rows: list[Mapping[str, Any]],
    mask: list[bool],
    cue_id: str,
    scope: str,
    random_seeds: int,
    include_controls: bool = True,
) -> dict[str, Any]:
    actual = balanced_metrics(rows, mask, include_hits=True)
    selected_count = int(actual["selected_count"])
    out = {"cue_id": cue_id, "scope": scope, **actual}
    if include_controls:
        global_bas = [
            balanced_metrics(rows, random_mask(rows, selected_count, seed), include_hits=False)["balanced_accuracy"]
            for seed in range(random_seeds)
        ]
        seq_bas = [
            balanced_metrics(rows, seq_count_random_mask(rows, mask, seed), include_hits=False)["balanced_accuracy"]
            for seed in range(random_seeds)
        ]
        rotation_bas = [
            balanced_metrics(rows, control, include_hits=False)["balanced_accuracy"]
            for control in internal_rotation_masks(rows, mask)
        ]
        global_p95 = percentile(global_bas, 0.95)
        seq_p95 = percentile(seq_bas, 0.95)
        rotation_p95 = percentile(rotation_bas, 0.95)
    else:
        global_p95 = seq_p95 = rotation_p95 = None
    actual_ba = float(out["balanced_accuracy"])
    out.update(
        {
            "control_evaluated": include_controls,
            "global_same_count_random_ba_p95": global_p95,
            "seq_count_random_ba_p95": seq_p95,
            "internal_rotation_ba_p95": rotation_p95,
            "global_same_count_margin": actual_ba - global_p95 if global_p95 is not None else None,
            "seq_count_margin": actual_ba - seq_p95 if seq_p95 is not None else None,
            "internal_rotation_margin": actual_ba - rotation_p95 if rotation_p95 is not None else None,
        }
    )
    gates = {
        "bad_recall_gate": out["bad_recall"] >= CONTROL_THRESHOLDS["bad_recall"],
        "good_FPR_gate": out["good_FPR"] <= CONTROL_THRESHOLDS["good_FPR"],
        "positive_sequence_coverage_gate": int(out["positive_sequence_coverage"])
        >= CONTROL_THRESHOLDS["positive_sequence_coverage"],
        "global_same_count_margin_gate": out["global_same_count_margin"] is not None
        and out["global_same_count_margin"] >= CONTROL_THRESHOLDS["control_margin"],
        "seq_count_margin_gate": out["seq_count_margin"] is not None
        and out["seq_count_margin"] >= CONTROL_THRESHOLDS["control_margin"],
        "internal_rotation_margin_gate": out["internal_rotation_margin"] is not None
        and out["internal_rotation_margin"] >= CONTROL_THRESHOLDS["control_margin"],
    }
    out.update(gates)
    out["candidate_gate_pass"] = bool(include_controls and all(gates.values()))
    return out


def bool_text(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def basic_rank_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        bool_text(row.get("bad_recall_gate"))
        and bool_text(row.get("good_FPR_gate"))
        and bool_text(row.get("positive_sequence_coverage_gate")),
        f(row.get("balanced_accuracy"), -1.0),
        f(row.get("bad_recall"), -1.0),
        -f(row.get("good_FPR"), 999.0),
        int(row.get("positive_sequence_coverage") or 0),
        -int(row.get("selected_negative_count") or 0),
        int(row.get("selected_positive_count") or 0),
    )


def final_rank_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        bool_text(row.get("candidate_gate_pass")),
        f(row.get("balanced_accuracy"), -1.0),
        f(row.get("global_same_count_margin"), -999.0),
        f(row.get("seq_count_margin"), -999.0),
        f(row.get("internal_rotation_margin"), -999.0),
        f(row.get("bad_recall"), -1.0),
        -f(row.get("good_FPR"), 999.0),
    )


def select_control_candidates(
    basics: list[dict[str, Any]],
    candidates: Mapping[str, list[bool]],
    limit: int,
) -> list[tuple[str, list[bool]]]:
    ranked = sorted(basics, key=basic_rank_key, reverse=True)
    selected: list[tuple[str, list[bool]]] = []
    seen: set[str] = set()
    for pass_pre_gates in [True, False]:
        for row in ranked:
            raw_pass = (
                bool_text(row.get("bad_recall_gate"))
                and bool_text(row.get("good_FPR_gate"))
                and bool_text(row.get("positive_sequence_coverage_gate"))
            )
            if raw_pass != pass_pre_gates:
                continue
            cue_id = str(row["cue_id"])
            if cue_id not in candidates or cue_id in seen:
                continue
            selected.append((cue_id, candidates[cue_id]))
            seen.add(cue_id)
            if len(selected) >= limit:
                return selected
    return selected


def evaluate_candidate_set(
    rows: list[Mapping[str, Any]],
    atoms: Mapping[str, list[bool]],
    scope: str,
    random_seeds: int,
    control_limit: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, list[bool]]]:
    candidates = candidate_masks(atoms, include_pairs=True)
    basics = [
        evaluate_candidate(rows, mask, cue_id, scope, random_seeds, include_controls=False)
        for cue_id, mask in candidates.items()
    ]
    selected = select_control_candidates(basics, candidates, control_limit)
    metrics = [
        evaluate_candidate(rows, mask, cue_id, scope, random_seeds, include_controls=True)
        for cue_id, mask in selected
    ]
    basics.sort(key=basic_rank_key, reverse=True)
    metrics.sort(key=final_rank_key, reverse=True)
    return basics, metrics, candidates


def selected_good_false_positives(rows: list[Mapping[str, Any]], metric: Mapping[str, Any]) -> list[dict[str, Any]]:
    selected = {part for part in str(metric.get("selected_negative_pair_ids") or "").split(",") if part}
    out = []
    for row in rows:
        if str(row.get("pair_id")) in selected:
            out.append(dict(row))
    return out


def make_report(summary: Mapping[str, Any], best_method: Mapping[str, Any], best_combined: Mapping[str, Any]) -> str:
    return f"""
# Track G SWA Internal Cue Eval v1

- Labelled universe: `{summary['labelled_universe_count']}` (`{summary['positive_count']}` positive, `{summary['negative_count']}` negative).
- v83 matched pairs: `{summary['v83_matched_pair_count']}`.
- v85 matched pairs: `{summary['v85_matched_pair_count']}`.
- v85 Q/K vector loaded: `{summary['v85_qk_feature_vector_loaded']}`.
- Method-safe internal basic/control/pass: `{summary['method_safe_internal_basic_count']}` / `{summary['method_safe_internal_control_evaluated_count']}` / `{summary['method_safe_internal_passing_count']}`.
- Combined-context diagnostic basic/control/pass: `{summary['combined_context_basic_count']}` / `{summary['combined_context_control_evaluated_count']}` / `{summary['combined_context_passing_count']}`.

## Best Method-Safe Internal

- cue: `{best_method.get('cue_id')}`
- bad_recall / good_FPR / BA: `{best_method.get('bad_recall')}` / `{best_method.get('good_FPR')}` / `{best_method.get('balanced_accuracy')}`
- global / seq / internal-rotation margins: `{best_method.get('global_same_count_margin')}` / `{best_method.get('seq_count_margin')}` / `{best_method.get('internal_rotation_margin')}`
- candidate_gate_pass: `{best_method.get('candidate_gate_pass')}`

## Best Combined-Context Diagnostic

- cue: `{best_combined.get('cue_id')}`
- bad_recall / good_FPR / BA: `{best_combined.get('bad_recall')}` / `{best_combined.get('good_FPR')}` / `{best_combined.get('balanced_accuracy')}`
- global / seq / internal-rotation margins: `{best_combined.get('global_same_count_margin')}` / `{best_combined.get('seq_count_margin')}` / `{best_combined.get('internal_rotation_margin')}`
- candidate_gate_pass: `{best_combined.get('candidate_gate_pass')}`

## Interpretation

The method-safe gate is based only on numeric internal v83/v85 cue columns. The
combined-context diagnostic scope includes support-class/context-derived columns
and is therefore not used to authorize Track E runtime action. Even if a cue
passes here, runtime action remains blocked until Track E has a strict handoff
mechanism that beats measured controls and protects good cases.
"""


def make_conflict_report(
    summary: Mapping[str, Any],
    best_method: Mapping[str, Any],
    best_combined: Mapping[str, Any],
) -> str:
    method_pass = bool_text(best_method.get("candidate_gate_pass"))
    combined_pass = bool_text(best_combined.get("candidate_gate_pass"))
    lines = [
        "# Cue Conflict Report",
        "",
        f"- method-safe internal gate pass: `{method_pass}`",
        f"- combined-context diagnostic gate pass: `{combined_pass}`",
        f"- method blocker: `{summary.get('blocker')}`",
        "",
        "## Evidence Chain",
        "",
        f"- best method cue: `{best_method.get('cue_id')}`",
        f"- best method selected positives: `{best_method.get('selected_positive_pair_ids')}`",
        f"- best method selected negatives: `{best_method.get('selected_negative_pair_ids')}`",
        f"- best combined cue: `{best_combined.get('cue_id')}`",
        f"- best combined selected positives: `{best_combined.get('selected_positive_pair_ids')}`",
        f"- best combined selected negatives: `{best_combined.get('selected_negative_pair_ids')}`",
    ]
    if (not method_pass) and combined_pass:
        lines.extend(
            [
                "",
                "## Conflict",
                "",
                "Combined/context tags separate the labelled universe better than pure numeric internal cues.",
                "This is not action-ready evidence because support-class tags include diagnostic construction choices.",
            ]
        )
    elif not method_pass:
        lines.extend(
            [
                "",
                "## Conflict",
                "",
                "No tested internal numeric cue passed G5 controls. Continue Track G cue mining before action.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "## Conflict",
                "",
                "A method-safe internal cue passed this cue audit, but Track E action-surface validation is still required.",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    v83_rows, v83_meta = load_v83_pair_rows(args.v83_clue_matrix)
    v85_rows, v85_meta = aggregate_v85(args.v85_qk_index, args.v85_qk_features)
    rows, eval_meta = build_eval_rows(args.evaluation_rows, v83_rows, v85_rows)
    write_csv(args.out_dir / "internal_eval_rows.csv", rows)

    if not rows:
        summary = {
            "stage": "TrackG_G5_SWA_internal_cue_eval_v1",
            **eval_meta,
            **v83_meta,
            **v85_meta,
            "gate_pass": False,
            "runtime_action_allowed": False,
            "blocker": "empty_labelled_evaluation_universe",
        }
        write_json(args.out_dir / "summary.json", summary)
        print("blocker=empty_labelled_evaluation_universe")
        return

    method_cols, support_cols = numeric_feature_columns(rows)
    method_atoms, method_thresholds = atom_thresholds(rows, method_cols)
    support_atoms, support_thresholds = atom_thresholds(rows, support_cols)
    context_atoms = bool_tag_atoms(rows, ["v83_row_scope", "v83_case_type"])
    combined_atoms = {**method_atoms, **support_atoms, **context_atoms}

    method_basic, method_metrics, _ = evaluate_candidate_set(
        rows,
        method_atoms,
        "method_safe_internal_numeric_v83_v85",
        args.random_seeds,
        args.max_method_control_candidates,
    )
    combined_basic, combined_metrics, _ = evaluate_candidate_set(
        rows,
        combined_atoms,
        "diagnostic_combined_context_support_tags",
        args.random_seeds,
        args.max_combined_control_candidates,
    )

    write_csv(args.out_dir / "method_safe_internal_candidate_basic_metrics.csv", method_basic)
    write_csv(args.out_dir / "method_safe_internal_candidate_metrics.csv", method_metrics)
    write_csv(args.out_dir / "combined_context_candidate_basic_metrics.csv", combined_basic)
    write_csv(args.out_dir / "combined_context_candidate_metrics.csv", combined_metrics)
    write_json(args.out_dir / "thresholds.json", {"method": method_thresholds, "combined_support": support_thresholds})
    write_json(args.out_dir / "feature_columns.json", {"method_safe_internal": method_cols, "combined_support": support_cols})

    method_passing = [row for row in method_metrics if bool_text(row.get("candidate_gate_pass"))]
    combined_passing = [row for row in combined_metrics if bool_text(row.get("candidate_gate_pass"))]
    best_method = method_metrics[0] if method_metrics else {}
    best_combined = combined_metrics[0] if combined_metrics else {}
    blocker = "" if method_passing else "no_method_safe_internal_swa_cue_passes_g5_controls"
    recommendation = (
        "trackG_internal_swa_cue_service_ready_but_trackE_action_surface_still_required"
        if method_passing
        else "continue_trackG_internal_cue_mining_before_trackE_action"
    )
    summary = {
        "stage": "TrackG_G5_SWA_internal_cue_eval_v1",
        **eval_meta,
        **v83_meta,
        **v85_meta,
        "random_seeds": args.random_seeds,
        "method_safe_internal_feature_count": len(method_cols),
        "combined_context_support_feature_count": len(support_cols),
        "method_safe_internal_basic_count": len(method_basic),
        "method_safe_internal_control_evaluated_count": len(method_metrics),
        "method_safe_internal_passing_count": len(method_passing),
        "combined_context_basic_count": len(combined_basic),
        "combined_context_control_evaluated_count": len(combined_metrics),
        "combined_context_passing_count": len(combined_passing),
        "best_method_safe_internal": best_method,
        "best_combined_context": best_combined,
        "gate_pass": bool(method_passing),
        "runtime_action_allowed": False,
        "blocker": blocker,
        "next_route_recommendation": recommendation,
        "action_note": "cue audit only; Track E strict handoff action-surface validation remains required",
    }
    write_json(args.out_dir / "summary.json", summary)
    write_csv(args.out_dir / "good_false_positive_rows_best_method.csv", selected_good_false_positives(rows, best_method))
    write_csv(args.out_dir / "good_false_positive_rows_best_combined.csv", selected_good_false_positives(rows, best_combined))
    write_text(args.out_dir / "analysis.md", make_report(summary, best_method, best_combined))
    write_text(args.out_dir / "cue_conflict_report.md", make_conflict_report(summary, best_method, best_combined))
    write_text(args.out_dir / "next_route_recommendation.md", recommendation)

    print(f"labelled_universe_count={summary['labelled_universe_count']}")
    print(f"positive_count={summary['positive_count']}")
    print(f"negative_count={summary['negative_count']}")
    print(f"v83_matched_pair_count={summary['v83_matched_pair_count']}")
    print(f"v85_matched_pair_count={summary['v85_matched_pair_count']}")
    print(f"v85_qk_feature_vector_loaded={summary['v85_qk_feature_vector_loaded']}")
    print(f"method_safe_internal_basic_count={summary['method_safe_internal_basic_count']}")
    print(f"method_safe_internal_control_evaluated_count={summary['method_safe_internal_control_evaluated_count']}")
    print(f"method_safe_internal_passing_count={summary['method_safe_internal_passing_count']}")
    print(f"combined_context_basic_count={summary['combined_context_basic_count']}")
    print(f"combined_context_control_evaluated_count={summary['combined_context_control_evaluated_count']}")
    print(f"combined_context_passing_count={summary['combined_context_passing_count']}")
    print(f"best_method_safe_internal={best_method.get('cue_id')}")
    print(f"best_method_safe_bad_recall={best_method.get('bad_recall')}")
    print(f"best_method_safe_good_FPR={best_method.get('good_FPR')}")
    print(f"best_method_safe_global_margin={best_method.get('global_same_count_margin')}")
    print(f"best_method_safe_seq_margin={best_method.get('seq_count_margin')}")
    print(f"best_method_safe_internal_rotation_margin={best_method.get('internal_rotation_margin')}")
    print(f"gate_pass={summary['gate_pass']}")
    print(f"runtime_action_allowed={summary['runtime_action_allowed']}")
    print(f"blocker={summary['blocker']}")
    print(f"next_route_recommendation={summary['next_route_recommendation']}")


if __name__ == "__main__":
    main()
