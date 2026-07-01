#!/usr/bin/env python3
"""Audit READ patch-dump tensor proxies as v95 Track D cue sources.

This is diagnostic-only. It reads existing `read_cue_patch.pt` payloads and
tests whether patch-level proxy tensors such as `qk_var_patch`,
`uncertainty_patch`, and `dyn_patch` can separate READ_LOCAL_BAD cases from
good controls. These tensors are not raw READ Q/K tensors, so a passing cue
would still not authorize a READ QK action by itself.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch


ROOT = Path("results/acl2_v95tf_multiroute_semantic_memory_evidence_control")
DEFAULT_READ_ROWS = ROOT / "trackD_read_eligibility/rows.csv"
DEFAULT_CASE_ROWS = ROOT / "trackA_base_case_bank/rows.csv"
DEFAULT_OUT_DIR = ROOT / "trackD_read_patch_tensor_cue_source_v1"

TENSOR_KEYS = [
    "read_patch_final",
    "qk_var_patch",
    "key_avg_patch",
    "dyn_patch",
    "uncertainty_patch",
    "occlusion_patch",
    "confidence_patch",
    "anchor_patch",
    "read_active_gt050_patch",
    "read_active_q90_patch",
    "qq_patch",
    "kk_patch",
    "dyn4d_patch",
    "query_avg_patch",
    "query_shallow_patch",
    "query_deep_patch",
    "fa_query_shallow_patch",
    "fa_key_l0_patch",
    "fa_key_l4_patch",
    "fa_key_shallow_patch",
    "fa_key_middle_patch",
    "fa_key_deep_patch",
    "fa_key_all_patch",
    "fa_key_layer_var_patch",
    "fa_key_decay_patch",
    "fa_key_l0_deep_decay_patch",
    "fa_key_l4_deep_decay_patch",
    "fa_key_deep_low_patch",
    "gg_qk_middle_patch",
    "gg_qk_shallow_patch",
    "gg_qk_deep_patch",
    "gg_qq_middle_patch",
    "gg_kk_middle_patch",
    "gg_qq_shallow_patch",
    "gg_deep_static_patch",
    "gg_smd_a1b1g1_patch",
    "gg_smd_a0b1g1_patch",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--read-rows", type=Path, default=DEFAULT_READ_ROWS)
    parser.add_argument("--case-rows", type=Path, default=DEFAULT_CASE_ROWS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--override-dump-root",
        type=Path,
        default=None,
        help="Optional root containing seqXX/chunkNNN/read_cue_patch_dumps/chunk_NNN_read_cue_patch.pt.",
    )
    parser.add_argument("--seq-filter", default="", help="Optional comma-separated zero-padded sequence ids to evaluate.")
    parser.add_argument("--random-seeds", type=int, default=256)
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=5000,
        help="Maximum threshold candidates to evaluate after expanding tensor proxy keys.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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


def finite(values: Iterable[Any]) -> list[float]:
    return [value for value in (f(item) for item in values) if math.isfinite(value)]


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    vals = sorted(values)
    pos = (len(vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def stable_unit(*parts: Any) -> float:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12)


def stable_random_mask(rows: list[Mapping[str, Any]], count: int, seed: int) -> list[bool]:
    order = sorted(range(len(rows)), key=lambda idx: stable_unit("v95_trackD_patch_tensor_random", seed, rows[idx].get("pair_id"), idx))
    selected = set(order[: min(count, len(order))])
    return [idx in selected for idx in range(len(rows))]


def seq_count_random_mask(rows: list[Mapping[str, Any]], mask: list[bool], seed: int) -> list[bool]:
    by_seq: dict[str, list[int]] = {}
    for idx, row in enumerate(rows):
        by_seq.setdefault(str(row.get("seq")), []).append(idx)
    selected: set[int] = set()
    for seq, indices in by_seq.items():
        count = sum(1 for idx in indices if mask[idx])
        order = sorted(indices, key=lambda idx: stable_unit("v95_trackD_patch_tensor_seq_random", seed, seq, rows[idx].get("pair_id"), idx))
        selected.update(order[: min(count, len(order))])
    return [idx in selected for idx in range(len(rows))]


def wanted_seqs(text: str) -> set[str]:
    return {item.strip().zfill(2) for item in str(text or "").replace(";", ",").split(",") if item.strip()}


def join_rows(
    read_rows: list[dict[str, str]],
    case_rows: list[dict[str, str]],
    *,
    seq_filter: set[str],
    override_dump_root: Path | None,
) -> list[dict[str, Any]]:
    by_id = {str(row.get("case_id")): row for row in case_rows}
    rows: list[dict[str, Any]] = []
    for row in read_rows:
        seq = str(row.get("seq", "")).zfill(2)
        if seq_filter and seq not in seq_filter:
            continue
        pair_id = str(row.get("pair_id"))
        case = by_id.get(pair_id, {})
        merged = dict(row)
        if override_dump_root is not None:
            curr = int(float(row.get("curr_chunk")))
            replacement = override_dump_root / f"seq{seq}" / f"chunk{curr:03d}" / "read_cue_patch_dumps" / f"chunk_{curr:03d}_read_cue_patch.pt"
            merged["read_trace_path_original"] = merged.get("read_trace_path", "")
            merged["read_trace_path"] = str(replacement)
            merged["read_trace_path_override_exists"] = replacement.exists()
        merged.update(
            {
                "case_id": pair_id,
                "v95_case_bucket": case.get("v95_case_bucket", ""),
                "case_label_offline_only": case.get("case_label_offline_only", ""),
                "failure_type_primary": case.get("failure_type_primary", ""),
                "L2_intra_scale_cv": case.get("L2_intra_scale_cv", ""),
                "L2_head_tail_proxy_error": case.get("L2_head_tail_proxy_error", ""),
                "L3_J_handoff": case.get("L3_J_handoff", ""),
            }
        )
        rows.append(merged)
    return rows


def tensor_stats(tensor: Any, prefix: str) -> dict[str, Any]:
    if tensor is None:
        return {}
    try:
        arr = torch.as_tensor(tensor).detach().float().cpu().flatten()
    except Exception as exc:  # noqa: BLE001
        return {f"{prefix}_load_error": f"{type(exc).__name__}:{exc}"}
    arr = arr[torch.isfinite(arr)]
    if arr.numel() == 0:
        return {f"{prefix}_numel": 0}
    return {
        f"{prefix}_numel": int(arr.numel()),
        f"{prefix}_mean": float(arr.mean().item()),
        f"{prefix}_std": float(arr.std(unbiased=False).item()) if arr.numel() > 1 else 0.0,
        f"{prefix}_p10": float(torch.quantile(arr, 0.10).item()),
        f"{prefix}_p50": float(torch.quantile(arr, 0.50).item()),
        f"{prefix}_p90": float(torch.quantile(arr, 0.90).item()),
        f"{prefix}_gt050_mass": float((arr > 0.50).float().mean().item()),
        f"{prefix}_gt075_mass": float((arr > 0.75).float().mean().item()),
    }


def scan_keys(obj: Any, prefix: str = "") -> list[str]:
    keys: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            text = f"{prefix}.{key}" if prefix else str(key)
            keys.append(text)
            keys.extend(scan_keys(value, text))
    return keys


def has_raw_read_qk(keys: Iterable[str]) -> bool:
    lowered = [key.lower() for key in keys]
    raw_query = any("query" in key or key.endswith(".q") or "_q_" in key for key in lowered)
    raw_key = any("key_tensor" in key or key.endswith(".k") or "_k_" in key for key in lowered)
    return raw_query and raw_key


def add_patch_tensor_features(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        path = Path(str(row.get("read_trace_path") or ""))
        row["patch_tensor_path_exists"] = path.exists()
        if not path.exists():
            row["patch_tensor_load_error"] = "missing_read_trace_path"
            continue
        try:
            payload = torch.load(path, map_location="cpu")
        except Exception as exc:  # noqa: BLE001
            row["patch_tensor_load_error"] = f"{type(exc).__name__}:{exc}"
            continue
        keys = scan_keys(payload)
        row["patch_dump_schema"] = payload.get("schema", "") if isinstance(payload, dict) else ""
        row["raw_read_qk_tensor_available"] = has_raw_read_qk(keys)
        tensors = payload.get("tensors", {}) if isinstance(payload, dict) else {}
        row["patch_tensor_keys"] = ",".join(sorted(str(key) for key in tensors.keys())) if isinstance(tensors, dict) else ""
        if not isinstance(tensors, dict):
            row["patch_tensor_load_error"] = "payload_tensors_not_dict"
            continue
        for key in TENSOR_KEYS:
            if key in tensors:
                row.update(tensor_stats(tensors[key], key))
            else:
                row[f"{key}_missing"] = True
        add_combination_features(row)


def add_combination_features(row: dict[str, Any]) -> None:
    qk = f(row.get("qk_var_patch_mean"))
    unc = f(row.get("uncertainty_patch_mean"))
    dyn = f(row.get("dyn_patch_mean"))
    occ = f(row.get("occlusion_patch_mean"))
    conf = f(row.get("confidence_patch_mean"))
    read = f(row.get("read_patch_final_mean"))
    keyavg = f(row.get("key_avg_patch_mean"))
    anchor = f(row.get("anchor_patch_mean"))
    row["qk_uncertainty_mean_sum"] = qk + unc
    row["qk_dyn_mean_sum"] = qk + dyn
    row["read_qk_proxy_mean_product"] = read * qk
    row["read_uncertainty_mean_product"] = read * unc
    row["read_dyn_mean_product"] = read * dyn
    row["risk_minus_confidence_mean"] = (qk + unc + dyn + occ) / 4.0 - conf
    row["keyavg_minus_qk_proxy_mean"] = keyavg - qk
    row["anchor_minus_uncertainty_mean"] = anchor - unc


def feature_columns(rows: list[Mapping[str, Any]]) -> list[str]:
    candidates: list[str] = []
    excluded_suffixes = ("_numel",)
    for row in rows:
        for key, value in row.items():
            if key.endswith(excluded_suffixes):
                continue
            vals = finite([value])
            if vals and key not in candidates:
                candidates.append(key)
    return candidates


def build_candidates(rows: list[Mapping[str, Any]], max_candidates: int) -> list[dict[str, Any]]:
    features = [
        key
        for key in feature_columns(rows)
        if any(
            token in key
            for token in (
                "_mean",
                "_std",
                "_p10",
                "_p50",
                "_p90",
                "_gt050_mass",
                "_gt075_mass",
            )
        )
        or key
        in {
            "qk_uncertainty_mean_sum",
            "qk_dyn_mean_sum",
            "read_qk_proxy_mean_product",
            "read_uncertainty_mean_product",
            "read_dyn_mean_product",
            "risk_minus_confidence_mean",
            "keyavg_minus_qk_proxy_mean",
            "anchor_minus_uncertainty_mean",
        }
    ]
    candidates: list[dict[str, Any]] = []
    for feature in features:
        vals = finite(row.get(feature) for row in rows)
        if len(set(vals)) < 2:
            continue
        for direction, qs in (("ge", (0.50, 0.60, 0.70, 0.75, 0.80, 0.90)), ("le", (0.10, 0.20, 0.25, 0.30, 0.40, 0.50))):
            for q in qs:
                threshold = quantile(vals, q)
                if threshold is None:
                    continue
                candidates.append(
                    {
                        "cue_id": f"{feature.upper()}_{direction.upper()}_Q{int(q * 100)}",
                        "feature": feature,
                        "direction": direction,
                        "threshold": threshold,
                    }
                )
    return candidates[:max_candidates]


def selected_mask(rows: list[Mapping[str, Any]], candidate: Mapping[str, Any]) -> list[bool]:
    feature = str(candidate["feature"])
    direction = str(candidate["direction"])
    threshold = f(candidate["threshold"])
    mask = []
    for row in rows:
        value = f(row.get(feature))
        hit = value >= threshold if direction == "ge" else value <= threshold
        mask.append(bool(hit and math.isfinite(value)))
    return mask


def balanced_metrics(rows: list[Mapping[str, Any]], mask: list[bool], include_ids: bool = True) -> dict[str, Any]:
    positives = [idx for idx, row in enumerate(rows) if row.get("v95_case_bucket") == "READ_LOCAL_BAD"]
    negatives = [idx for idx, row in enumerate(rows) if row.get("case_label_offline_only") == "good"]
    pos_hits = [idx for idx in positives if mask[idx]]
    neg_hits = [idx for idx in negatives if mask[idx]]
    bad_recall = len(pos_hits) / max(len(positives), 1)
    good_fpr = len(neg_hits) / max(len(negatives), 1)
    out: dict[str, Any] = {
        "selected_count": int(sum(mask)),
        "positive_total": len(positives),
        "negative_total": len(negatives),
        "selected_positive_count": len(pos_hits),
        "selected_negative_count": len(neg_hits),
        "bad_recall": bad_recall,
        "good_FPR": good_fpr,
        "balanced_accuracy": 0.5 * (bad_recall + (1.0 - good_fpr)),
        "positive_sequence_coverage": len({str(rows[idx].get("seq")) for idx in pos_hits}),
        "selected_sequence_coverage": len({str(row.get("seq")) for idx, row in enumerate(rows) if mask[idx]}),
    }
    if include_ids:
        out.update(
            {
                "selected_pair_ids": ",".join(str(row.get("pair_id")) for idx, row in enumerate(rows) if mask[idx]),
                "selected_positive_pair_ids": ",".join(str(rows[idx].get("pair_id")) for idx in pos_hits),
                "selected_negative_pair_ids": ",".join(str(rows[idx].get("pair_id")) for idx in neg_hits),
            }
        )
    return out


def evaluate_candidate(rows: list[Mapping[str, Any]], candidate: Mapping[str, Any], random_seeds: int) -> dict[str, Any]:
    mask = selected_mask(rows, candidate)
    actual = balanced_metrics(rows, mask)
    selected_count = int(actual["selected_count"])
    global_random = [
        balanced_metrics(rows, stable_random_mask(rows, selected_count, seed), include_ids=False)["balanced_accuracy"]
        for seed in range(random_seeds)
    ]
    seq_random = [
        balanced_metrics(rows, seq_count_random_mask(rows, mask, seed), include_ids=False)["balanced_accuracy"]
        for seed in range(random_seeds)
    ]
    global_p95 = quantile(global_random, 0.95)
    seq_p95 = quantile(seq_random, 0.95)
    gates = {
        "bad_recall_gate": actual["bad_recall"] >= 0.60,
        "good_FPR_gate": actual["good_FPR"] <= 0.25,
        "positive_sequence_coverage_gate": actual["positive_sequence_coverage"] >= 3,
        "global_same_count_margin_gate": global_p95 is not None and actual["balanced_accuracy"] > global_p95,
        "seq_count_margin_gate": seq_p95 is not None and actual["balanced_accuracy"] > seq_p95,
    }
    return {
        **candidate,
        **actual,
        "global_same_count_random_ba_p95": global_p95,
        "seq_count_random_ba_p95": seq_p95,
        "global_same_count_margin": None if global_p95 is None else actual["balanced_accuracy"] - global_p95,
        "seq_count_margin": None if seq_p95 is None else actual["balanced_accuracy"] - seq_p95,
        **gates,
        "candidate_gate_pass": all(gates.values()),
    }


def rank_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        bool(row.get("candidate_gate_pass")),
        f(row.get("bad_recall"), -1.0),
        -f(row.get("good_FPR"), 2.0),
        f(row.get("balanced_accuracy"), -1.0),
        f(row.get("global_same_count_margin"), -999.0),
    )


def main() -> None:
    args = parse_args()
    seq_filter = wanted_seqs(args.seq_filter)
    rows = join_rows(
        read_csv(args.read_rows),
        read_csv(args.case_rows),
        seq_filter=seq_filter,
        override_dump_root=args.override_dump_root,
    )
    add_patch_tensor_features(rows)
    candidates = build_candidates(rows, args.max_candidates)
    metrics = [evaluate_candidate(rows, candidate, args.random_seeds) for candidate in candidates]
    metrics.sort(key=rank_key, reverse=True)
    passing = [row for row in metrics if row.get("candidate_gate_pass")]
    best = metrics[0] if metrics else {}
    raw_qk_rows = sum(1 for row in rows if row.get("raw_read_qk_tensor_available"))
    loaded_rows = sum(1 for row in rows if row.get("patch_tensor_path_exists") and not row.get("patch_tensor_load_error"))
    summary = {
        "phase": "v95_trackD_read_patch_tensor_cue_source_v1",
        "read_rows": str(args.read_rows),
        "case_rows": str(args.case_rows),
        "override_dump_root": str(args.override_dump_root or ""),
        "seq_filter": sorted(seq_filter),
        "row_count": len(rows),
        "loaded_patch_dump_rows": loaded_rows,
        "raw_read_qk_tensor_available_rows": raw_qk_rows,
        "candidate_count": len(metrics),
        "max_candidates_requested": int(args.max_candidates),
        "candidate_passing_count": len(passing),
        "gate_pass": bool(passing),
        "runtime_action_allowed": False,
        "read_local_positive_count": sum(1 for row in rows if row.get("v95_case_bucket") == "READ_LOCAL_BAD"),
        "good_control_count": sum(1 for row in rows if row.get("case_label_offline_only") == "good"),
        "best_candidate": best,
        "passing_candidates": passing[:20],
        "blocker": (
            "raw_READ_QK_tensor_unavailable"
            if passing and raw_qk_rows == 0
            else ""
            if passing
            else "no_patch_tensor_proxy_passes_bad_good_random_controls;raw_READ_QK_tensor_unavailable"
        ),
        "interpretation_boundary": (
            "Patch tensors are proxy cue sources only. qk_var_patch is not raw READ Q/K compatibility, "
            "and this audit does not measure a READ action surface."
        ),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "patch_tensor_feature_rows.csv", rows)
    write_csv(args.out_dir / "patch_tensor_candidate_metrics.csv", metrics)
    if best:
        best_mask = selected_mask(rows, best)
        write_csv(args.out_dir / "best_candidate_selected_rows.csv", [row for idx, row in enumerate(rows) if best_mask[idx]])
    write_json(args.out_dir / "summary.json", summary)
    write_text(
        args.out_dir / "analysis.md",
        f"""
# Track D READ Patch Tensor Cue-Source Audit

- row_count: `{summary['row_count']}`
- loaded_patch_dump_rows: `{summary['loaded_patch_dump_rows']}`
- raw_read_qk_tensor_available_rows: `{summary['raw_read_qk_tensor_available_rows']}`
- read_local_positive_count: `{summary['read_local_positive_count']}`
- good_control_count: `{summary['good_control_count']}`
- candidate_count: `{summary['candidate_count']}`
- candidate_passing_count: `{summary['candidate_passing_count']}`
- best_candidate: `{best.get('cue_id')}`
- best_bad_recall: `{best.get('bad_recall')}`
- best_good_FPR: `{best.get('good_FPR')}`
- best_balanced_accuracy: `{best.get('balanced_accuracy')}`
- best_global_same_count_margin: `{best.get('global_same_count_margin')}`
- best_seq_count_margin: `{best.get('seq_count_margin')}`
- gate_pass: `{summary['gate_pass']}`
- runtime_action_allowed: `{summary['runtime_action_allowed']}`
- blocker: `{summary['blocker']}`

Interpretation: this audit follows the Track G fail-forward direction by
checking internal patch proxy tensors after the aggregate READ mass cue failed.
It does not turn `qk_var_patch` into raw READ Q/K compatibility and it does not
authorize a READ action.
""",
    )
    write_text(
        args.out_dir / "cue_source_failure_report.md",
        """
If this audit fails, the existing READ patch-dump proxy tensors are not enough
to pass the READ cue gate. The next auditable repair direction is to produce
true READ per-layer/per-head Q/K or attention dumps for the fixed v95 READ_LOCAL
bad/good case bank, then re-run bad/good/random cue validation before any
READ action pilot.
""",
    )
    print(f"row_count={summary['row_count']}")
    print(f"loaded_patch_dump_rows={summary['loaded_patch_dump_rows']}")
    print(f"raw_read_qk_tensor_available_rows={summary['raw_read_qk_tensor_available_rows']}")
    print(f"read_local_positive_count={summary['read_local_positive_count']}")
    print(f"good_control_count={summary['good_control_count']}")
    print(f"candidate_count={summary['candidate_count']}")
    print(f"candidate_passing_count={summary['candidate_passing_count']}")
    print(f"best_candidate={best.get('cue_id')}")
    print(f"best_bad_recall={best.get('bad_recall')}")
    print(f"best_good_FPR={best.get('good_FPR')}")
    print(f"best_balanced_accuracy={best.get('balanced_accuracy')}")
    print(f"best_global_same_count_margin={best.get('global_same_count_margin')}")
    print(f"best_seq_count_margin={best.get('seq_count_margin')}")
    print(f"gate_pass={summary['gate_pass']}")
    print(f"runtime_action_allowed={summary['runtime_action_allowed']}")
    print(f"blocker={summary['blocker']}")


if __name__ == "__main__":
    main()
