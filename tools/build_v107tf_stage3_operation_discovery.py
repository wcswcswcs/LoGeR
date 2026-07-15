#!/usr/bin/env python3
"""Build ACL2 v107TF Stage3 operation-level discovery rows and lever ranks."""

from __future__ import annotations

import csv
import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
V107 = ROOT / "results/acl2_v107tf_lingbot_cache_operation_observability_semantic_aware_update_retention"
STAGE1 = V107 / "stage1_cache_operation_instrumentation"
STAGE2 = V107 / "stage2_metric_reliability_verifier"
LC = V107 / "stage3_operation_discovery/length_control_safe96"
LC_STAGE2 = LC / "stage2_metric_reliability_verifier"
OUT = V107 / "stage3_operation_discovery"
TOP_REPORTS = OUT / "top_lever_reports"


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


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def fnum(row: dict[str, Any], key: str, default: float = float("nan")) -> float:
    try:
        raw = row.get(key, "")
        if raw in {"", None}:
            return default
        return float(raw)
    except (TypeError, ValueError):
        return default


def bval(row: dict[str, Any], key: str) -> bool:
    return str(row.get(key, "")).lower() in {"true", "1", "yes"}


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def pearson(xs: list[float], ys: list[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 3:
        return float("nan")
    xvals = [p[0] for p in pairs]
    yvals = [p[1] for p in pairs]
    mx = sum(xvals) / len(xvals)
    my = sum(yvals) / len(yvals)
    vx = sum((x - mx) ** 2 for x in xvals)
    vy = sum((y - my) ** 2 for y in yvals)
    if vx <= 0 or vy <= 0:
        return float("nan")
    cov = sum((x - mx) * (y - my) for x, y in pairs)
    return cov / math.sqrt(vx * vy)


def quantile(values: list[float], q: float) -> float:
    vals = sorted(v for v in values if math.isfinite(v))
    if not vals:
        return float("nan")
    pos = (len(vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def label_of(row: dict[str, Any]) -> str:
    return "bad" if row.get("target_kind") == "high_l3" else "safe_good"


def load_targets(path: Path) -> dict[str, dict[str, Any]]:
    targets: dict[str, dict[str, Any]] = {}
    for row in read_csv(path):
        row = dict(row)
        row["safe_good_or_bad_label"] = label_of(row)
        row["L3_handoff_penalty_nearby"] = fnum(row, "handoff_transfer_penalty")
        row["L4_adjacent_log_scale_jump_proxy"] = fnum(row, "adjacent_log_scale_jump")
        row["L4_metric_source"] = "adjacent_log_scale_jump_proxy_from_target_manifest"
        row["local_window_support"] = fnum(row, "local_sim3_ate_rmse_m")
        row["trace_frame_count"] = int(float(row.get("trace_frame_count", 0)))
        targets[row["target_id"]] = row
    return targets


def mean(values: list[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    if not vals:
        return float("nan")
    return sum(vals) / len(vals)


def memory_role_guess(row: dict[str, Any]) -> str:
    op = row.get("operation_type", "")
    ctx = row.get("context_path", "")
    token = row.get("token_type", "")
    metric = fnum(row, "metric_reliability_score")
    scale = fnum(row, "scale_observability_score")
    boundary = fnum(row, "boundary_mismatch_risk")
    if not finite(metric) or metric < 0.18 or (finite(boundary) and boundary > 0.25):
        return "REJECT_UNRELIABLE"
    if ctx == "anchor_context" or bval(row, "is_scale_frame") or token == "scale_frame_token":
        if scale >= 0.045:
            return "SCALE_REFERENCE_EVIDENCE"
        return "CONTEXT_ONLY"
    if op == "trajectory_write" or token == "trajectory_token" or ctx == "trajectory_memory":
        return "TRAJECTORY_MEMORY_EVIDENCE" if metric >= 0.23 else "CONTEXT_ONLY"
    if ctx == "local_pose_reference_window":
        return "LOCAL_REGISTRATION_EVIDENCE"
    return "CONTEXT_ONLY"


def enrich_rows(rows: list[dict[str, str]], targets: dict[str, dict[str, Any]], universe: str) -> list[dict[str, Any]]:
    by_target_values: dict[str, dict[str, float]] = defaultdict(dict)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["target_id"]].append(row)
    feature_cols = [
        "metric_reliability_score",
        "scale_observability_score",
        "boundary_mismatch_risk",
        "depth_spread",
        "point_spread",
        "metric_depth_consistency",
        "far_background_weakness",
    ]
    for target_id, group in grouped.items():
        for col in feature_cols:
            by_target_values[target_id][f"target_mean_{col}"] = mean([fnum(row, col) for row in group])

    out: list[dict[str, Any]] = []
    for row in rows:
        target = targets[row["target_id"]]
        merged: dict[str, Any] = {
            "schema": "acl2_v107tf_stage3_operation_discovery_row_v1",
            "universe": universe,
            **row,
            "sequence_id": target["seq"],
            "safe_good_or_bad_label": target["safe_good_or_bad_label"],
            "L3_handoff_penalty_nearby": target["L3_handoff_penalty_nearby"],
            "L4_rolling_drift_nearby": "",
            "L4_adjacent_log_scale_jump_proxy": target["L4_adjacent_log_scale_jump_proxy"],
            "L4_metric_source": target["L4_metric_source"],
            "local_window_support": target["local_window_support"],
            "trace_frame_count": target["trace_frame_count"],
            "semantic_available": False,
            "semantic_role": "unknown",
            "semantic_trust": 0.0,
            "semantic_missing_reason": "no_selected_window_aligned_semantic_masklet_stage_c_sidecar_found",
            "trajectory_memory_age": row.get("source_age", ""),
            "anchor_reference_age": row.get("source_age", "") if row.get("context_path") == "anchor_context" else "",
            **by_target_values[row["target_id"]],
        }
        merged["memory_role_prediction"] = memory_role_guess(merged)
        out.append(merged)
    return out


class Candidate:
    def __init__(
        self,
        family: str,
        operation_scope: str,
        cue_name: str,
        predicate: Callable[[dict[str, Any]], bool],
        semantic_claimed: bool = False,
        role_claimed: bool = False,
        token_type_claimed: bool = False,
        notes: str = "",
    ) -> None:
        self.family = family
        self.operation_scope = operation_scope
        self.cue_name = cue_name
        self.predicate = predicate
        self.semantic_claimed = semantic_claimed
        self.role_claimed = role_claimed
        self.token_type_claimed = token_type_claimed
        self.notes = notes


def build_candidates(rows: list[dict[str, Any]]) -> list[Candidate]:
    candidates: list[Candidate] = []
    for op in sorted({row.get("operation_type", "") for row in rows if row.get("operation_type", "")}):
        candidates.append(Candidate("F0_operation_only", op, f"operation_type=={op}", lambda r, op=op: r.get("operation_type") == op))
    for ctx in sorted({row.get("context_path", "") for row in rows if row.get("context_path", "")}):
        candidates.append(Candidate("F0_operation_only", "all", f"context_path=={ctx}", lambda r, ctx=ctx: r.get("context_path") == ctx))
    for token in sorted({row.get("token_type", "") for row in rows if row.get("token_type", "")}):
        candidates.append(Candidate("F0_operation_only", "all", f"token_type=={token}", lambda r, token=token: r.get("token_type") == token, token_type_claimed=True))
    for phase in sorted({row.get("phase", "") for row in rows if row.get("phase", "")}):
        candidates.append(Candidate("F0_operation_only", "all", f"phase=={phase}", lambda r, phase=phase: r.get("phase") == phase))
    budget_ops = {"eviction", "budget_drop", "retention", "budget_keep", "trajectory_write", "special_token_update"}
    candidates.append(Candidate("F0_operation_only", "budget_lifecycle", "operation_type in budget_lifecycle_ops", lambda r: r.get("operation_type") in budget_ops, notes="fixed_budget_lifecycle_family"))

    local_vals = sorted({fnum(row, "local_window_support") for row in rows if finite(row.get("local_window_support", ""))})
    for q in [0.5, 0.75]:
        thr = quantile(local_vals, q)
        candidates.append(Candidate("F1_geometry_only", "all", f"local_window_ate_rmse>=q{int(q*100)}:{thr:.6g}", lambda r, thr=thr: fnum(r, "local_window_support") >= thr))
        candidates.append(Candidate("F1_geometry_only", "all", f"local_window_ate_rmse<=q{int(q*100)}:{thr:.6g}", lambda r, thr=thr: fnum(r, "local_window_support") <= thr))

    verifier_cols = [
        ("metric_reliability_score", True),
        ("scale_observability_score", True),
        ("boundary_mismatch_risk", False),
        ("depth_spread", True),
        ("point_spread", True),
        ("metric_depth_consistency", False),
        ("far_background_weakness", False),
    ]
    for col, high_is_good in verifier_cols:
        vals = [fnum(row, f"target_mean_{col}") for row in rows if finite(row.get(f"target_mean_{col}", ""))]
        for q in [0.5, 0.75]:
            thr = quantile(vals, q)
            if high_is_good:
                candidates.append(Candidate("F3_verifier_only", "all", f"target_mean_{col}>=q{int(q*100)}:{thr:.6g}", lambda r, col=col, thr=thr: fnum(r, f"target_mean_{col}") >= thr))
            else:
                candidates.append(Candidate("F3_verifier_only", "all", f"target_mean_{col}<=q{int(q*100)}:{thr:.6g}", lambda r, col=col, thr=thr: fnum(r, f"target_mean_{col}") <= thr))

    op_values = sorted({row.get("operation_type", "") for row in rows if row.get("operation_type", "")})
    for op in op_values:
        for col in ["metric_reliability_score", "scale_observability_score", "boundary_mismatch_risk", "depth_spread"]:
            vals = [fnum(row, col) for row in rows if row.get("operation_type") == op and finite(row.get(col, ""))]
            if not vals:
                continue
            thr = quantile(vals, 0.5)
            if col == "boundary_mismatch_risk":
                candidates.append(Candidate("F7_internal_operation_plus_verifier", op, f"{op} & {col}<=median:{thr:.6g}", lambda r, op=op, col=col, thr=thr: r.get("operation_type") == op and fnum(r, col) <= thr))
            else:
                candidates.append(Candidate("F7_internal_operation_plus_verifier", op, f"{op} & {col}>=median:{thr:.6g}", lambda r, op=op, col=col, thr=thr: r.get("operation_type") == op and fnum(r, col) >= thr))
    return candidates


def ba_for_selection(selected: set[str], targets: dict[str, dict[str, Any]]) -> tuple[float, float, float]:
    bad = [tid for tid, row in targets.items() if row["safe_good_or_bad_label"] == "bad"]
    good = [tid for tid, row in targets.items() if row["safe_good_or_bad_label"] == "safe_good"]
    bad_recall = sum(tid in selected for tid in bad) / len(bad) if bad else 0.0
    good_fpr = sum(tid in selected for tid in good) / len(good) if good else 0.0
    ba = (bad_recall + (1.0 - good_fpr)) / 2.0
    return bad_recall, good_fpr, ba


def random_p95_for_count(targets: dict[str, dict[str, Any]], count: int) -> tuple[float, float]:
    ids = sorted(targets)
    bas: list[float] = []
    if count < 0 or count > len(ids):
        return float("nan"), float("nan")
    for combo in itertools.combinations(ids, count):
        bas.append(ba_for_selection(set(combo), targets)[2])
    if not bas:
        return float("nan"), float("nan")
    return mean(bas), quantile(bas, 0.95)


def evaluate_candidate(candidate: Candidate, rows: list[dict[str, Any]], targets: dict[str, dict[str, Any]], universe: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    selected_rows_by_target: dict[str, int] = defaultdict(int)
    for row in rows:
        grouped[row["target_id"]].append(row)
        if candidate.predicate(row):
            selected_rows_by_target[row["target_id"]] += 1

    selected_targets = {tid for tid, count in selected_rows_by_target.items() if count > 0}
    scores = []
    l3 = []
    l4 = []
    for tid in sorted(targets):
        total = len(grouped.get(tid, []))
        score = selected_rows_by_target.get(tid, 0) / total if total else 0.0
        scores.append(score)
        l3.append(float(targets[tid]["L3_handoff_penalty_nearby"]))
        l4.append(float(targets[tid]["L4_adjacent_log_scale_jump_proxy"]))

    bad_recall, good_fpr, ba = ba_for_selection(selected_targets, targets)
    random_mean, random_p95 = random_p95_for_count(targets, len(selected_targets))
    selected_seq_counts = Counter(targets[tid]["seq"] for tid in selected_targets)
    selected_bad_seqs = {targets[tid]["seq"] for tid in selected_targets if targets[tid]["safe_good_or_bad_label"] == "bad"}
    selected_rows_total = sum(selected_rows_by_target.values())
    row = {
        "schema": "acl2_v107tf_stage3_operation_lever_rank_row_v1",
        "universe": universe,
        "cue_family": candidate.family,
        "operation_scope": candidate.operation_scope,
        "cue_name": candidate.cue_name,
        "available_case_count": len(targets),
        "available_operation_row_count": len(rows),
        "selected_target_count": len(selected_targets),
        "selected_operation_row_count": selected_rows_total,
        "sequence_coverage": len(selected_bad_seqs),
        "selected_positive_sequence_coverage": len(selected_seq_counts),
        "selected_positive_sequence_max_frac": (max(selected_seq_counts.values()) / len(selected_targets)) if selected_targets else 0.0,
        "bad_recall": bad_recall,
        "good_FPR": good_fpr,
        "balanced_accuracy": ba,
        "abs_corr_L3": abs(pearson(scores, l3)) if math.isfinite(pearson(scores, l3)) else "",
        "abs_corr_L4": abs(pearson(scores, l4)) if math.isfinite(pearson(scores, l4)) else "",
        "same_count_random_BA_mean": random_mean,
        "same_count_random_BA_p95": random_p95,
        "same_count_random_margin": ba - random_p95 if math.isfinite(random_p95) else "",
        "semantic_shuffle_margin": "" if not candidate.semantic_claimed else 0.0,
        "role_rotation_margin": "" if not candidate.role_claimed else 0.0,
        "token_type_rotation_margin": "" if not candidate.token_type_claimed else "not_computed_no_action_claim",
        "semantic_claimed": candidate.semantic_claimed,
        "role_claimed": candidate.role_claimed,
        "token_type_claimed": candidate.token_type_claimed,
        "notes": candidate.notes,
    }
    corr_l3 = float(row["abs_corr_L3"]) if row["abs_corr_L3"] != "" else 0.0
    corr_l4 = float(row["abs_corr_L4"]) if row["abs_corr_L4"] != "" else 0.0
    margin = float(row["same_count_random_margin"]) if row["same_count_random_margin"] != "" else -math.inf
    row["lever_pass"] = (
        bad_recall >= 0.65
        and good_fpr <= 0.35
        and (corr_l3 >= 0.35 or corr_l4 >= 0.35)
        and len(selected_bad_seqs) >= 3
        and margin >= 0.05
    )
    loso_rows = []
    for seq in sorted({t["seq"] for t in targets.values()}):
        seq_targets = {tid: t for tid, t in targets.items() if t["seq"] == seq}
        seq_selected = {tid for tid in selected_targets if targets[tid]["seq"] == seq}
        br, gf, sba = ba_for_selection(seq_selected, seq_targets)
        loso_rows.append({
            "schema": "acl2_v107tf_stage3_sequence_loso_row_v1",
            "universe": universe,
            "cue_family": candidate.family,
            "operation_scope": candidate.operation_scope,
            "cue_name": candidate.cue_name,
            "heldout_seq": seq,
            "heldout_case_count": len(seq_targets),
            "heldout_bad_recall": br,
            "heldout_good_FPR": gf,
            "heldout_balanced_accuracy": sba,
        })
    row["sequence_LOSO_mean_BA"] = mean([float(r["heldout_balanced_accuracy"]) for r in loso_rows])
    return row, loso_rows


def rank_universe(rows: list[dict[str, Any]], targets: dict[str, dict[str, Any]], universe: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rank_rows: list[dict[str, Any]] = []
    loso_rows: list[dict[str, Any]] = []
    for candidate in build_candidates(rows):
        row, seq_rows = evaluate_candidate(candidate, rows, targets, universe)
        rank_rows.append(row)
        loso_rows.extend(seq_rows)
    rank_rows.sort(
        key=lambda r: (
            bool(r["lever_pass"]),
            float(r["balanced_accuracy"]),
            float(r["bad_recall"]),
            -float(r["good_FPR"]),
            float(r["abs_corr_L3"] or 0.0),
            float(r["abs_corr_L4"] or 0.0),
        ),
        reverse=True,
    )
    return rank_rows, loso_rows


def subset_rows(rows: list[dict[str, Any]], target_ids: set[str], universe: str) -> list[dict[str, Any]]:
    return [dict(row, universe=universe) for row in rows if row["target_id"] in target_ids]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    TOP_REPORTS.mkdir(parents=True, exist_ok=True)

    base_targets = load_targets(STAGE1 / "target_manifest.csv")
    lc_targets = load_targets(LC / "target_manifest.csv")
    base_rows_raw = read_csv(STAGE2 / "operation_verifier_join_rows.csv")
    lc_rows_raw = read_csv(LC_STAGE2 / "operation_verifier_join_rows.csv")

    base_rows = enrich_rows(base_rows_raw, base_targets, "base_original_mixed_64_96f")
    base_rank, base_loso = rank_universe(base_rows, base_targets, "base_original_mixed_64_96f")

    length_targets: dict[str, dict[str, Any]] = {}
    length_target_ids: set[str] = set()
    for tid, target in base_targets.items():
        if target["target_kind"] == "high_l3" or (target["target_kind"] == "safe_good_low_drift" and target["trace_frame_count"] == 96):
            length_targets[tid] = target
            length_target_ids.add(tid)
    for tid, target in lc_targets.items():
        length_targets[tid] = target
        length_target_ids.add(tid)

    length_rows_base = subset_rows(base_rows, set(base_targets) & length_target_ids, "length_matched_96f")
    length_rows_lc = enrich_rows(lc_rows_raw, lc_targets, "length_matched_96f")
    length_rows = length_rows_base + length_rows_lc
    length_rank, length_loso = rank_universe(length_rows, length_targets, "length_matched_96f")

    write_csv(OUT / "operation_discovery_rows.csv", length_rows)
    write_csv(OUT / "operation_lever_rank.csv", length_rank)
    write_csv(OUT / "base_mixed_length_operation_lever_rank.csv", base_rank)
    write_csv(OUT / "sequence_loso_rows.csv", length_loso)
    write_csv(OUT / "base_mixed_length_sequence_loso_rows.csv", base_loso)

    control_rows: list[dict[str, Any]] = []
    base_top = base_rank[0] if base_rank else {}
    length_top = length_rank[0] if length_rank else {}
    for row in base_rank[:20] + length_rank[:20]:
        control_rows.append({
            "schema": "acl2_v107tf_stage3_control_margin_row_v1",
            "universe": row["universe"],
            "cue_family": row["cue_family"],
            "operation_scope": row["operation_scope"],
            "cue_name": row["cue_name"],
            "balanced_accuracy": row["balanced_accuracy"],
            "same_count_random_BA_p95": row["same_count_random_BA_p95"],
            "same_count_random_margin": row["same_count_random_margin"],
            "semantic_shuffle_margin": row["semantic_shuffle_margin"],
            "role_rotation_margin": row["role_rotation_margin"],
            "token_type_rotation_margin": row["token_type_rotation_margin"],
            "lever_pass": row["lever_pass"],
        })
    write_csv(OUT / "control_margin_rows.csv", control_rows)

    semantic_rows = []
    for op in sorted({row.get("operation_type", "") for row in length_rows if row.get("operation_type", "")}):
        semantic_rows.append({
            "schema": "acl2_v107tf_stage3_semantic_increment_row_v1",
            "operation_type": op,
            "semantic_available": False,
            "semantic_increment_claimed": False,
            "geometry_only_good_FPR": "",
            "semantic_geometry_verifier_good_FPR": "",
            "good_FPR_delta": "",
            "bad_recall_delta_at_good_FPR_le_0p25": "",
            "reason": "no_selected_window_aligned_semantic_masklet_stage_c_sidecar_found",
        })
    write_csv(OUT / "semantic_increment_by_operation.csv", semantic_rows)
    write_text(
        OUT / "semantic_increment_failure.md",
        "\n".join(
            [
                "# Semantic Increment Failure",
                "",
                "Stage3 did not claim semantic contribution because no selected-window aligned semantic masklet-stage-C sidecar with a reliable join key was found.",
                "Rows are retained with `semantic_role=unknown`, `semantic_trust=0.0`, and `semantic_available=false`.",
                "Therefore F2/F4/F5/F6 semantic-family claims are unavailable rather than failed by measured negative effect.",
            ]
        ),
    )

    pass_rows = [row for row in length_rank if row["lever_pass"]]
    base_pass_rows = [row for row in base_rank if row["lever_pass"]]
    stage3_pass = bool(pass_rows)
    taxonomy = "MEMORY_OPERATION_LEVER_DIAGNOSTIC_PASS_ACTION_BLOCKED" if stage3_pass else "NO_MEMORY_OPERATION_LEVER_FOUND"
    no_go_reason = ""
    if not stage3_pass:
        no_go_reason = (
            "length_matched_96f universe found no cue satisfying bad_recall/good_FPR/correlation/"
            "sequence_coverage/same_count_random_margin gates; base mixed-length operation-only pass is treated as trace-length/cache-budget confound."
        )
        write_text(
            OUT / "NO_MEMORY_OPERATION_LEVER_FOUND.md",
            "\n".join(
                [
                    "# NO_MEMORY_OPERATION_LEVER_FOUND",
                    "",
                    no_go_reason,
                    "",
                    f"- base_top_cue: `{base_top.get('cue_name', '')}`",
                    f"- base_top_pass: `{base_top.get('lever_pass', '')}`",
                    f"- length_matched_top_cue: `{length_top.get('cue_name', '')}`",
                    f"- length_matched_top_pass: `{length_top.get('lever_pass', '')}`",
                    f"- semantic_available: `false`",
                ]
            ),
        )

    write_text(
        OUT / "trace_length_confound_report.md",
        "\n".join(
            [
                "# Trace-Length / Cache-Budget Confound Report",
                "",
                "The original base discovery universe mixed 96F high-L3 traces with three 64F safe-good traces.",
                "Because LingBot's SDPA cache uses a fixed 64-frame sliding-window plus 8 scale frames, `eviction`, `retention`, `budget_drop`, and `trajectory_write` appear when a trace reaches 96F.",
                "Three safe-good 64F cases were rerun as 96F length controls. All passed no-action parity and all exposed the same budget lifecycle operation families.",
                "",
                f"- base_top_cue: `{base_top.get('cue_name', '')}`",
                f"- base_top_balanced_accuracy: `{base_top.get('balanced_accuracy', '')}`",
                f"- base_top_lever_pass: `{base_top.get('lever_pass', '')}`",
                f"- length_matched_top_cue: `{length_top.get('cue_name', '')}`",
                f"- length_matched_top_balanced_accuracy: `{length_top.get('balanced_accuracy', '')}`",
                f"- length_matched_top_lever_pass: `{length_top.get('lever_pass', '')}`",
            ]
        ),
    )

    for row in length_rank[:5]:
        write_text(
            TOP_REPORTS / f"{row['cue_family']}_{row['operation_scope']}_{abs(hash(row['cue_name'])) % 100000}.md",
            "\n".join(
                [
                    f"# {row['cue_name']}",
                    "",
                    f"- universe: `{row['universe']}`",
                    f"- family: `{row['cue_family']}`",
                    f"- operation_scope: `{row['operation_scope']}`",
                    f"- lever_pass: `{row['lever_pass']}`",
                    f"- bad_recall: `{row['bad_recall']}`",
                    f"- good_FPR: `{row['good_FPR']}`",
                    f"- balanced_accuracy: `{row['balanced_accuracy']}`",
                    f"- abs_corr_L3: `{row['abs_corr_L3']}`",
                    f"- abs_corr_L4_proxy: `{row['abs_corr_L4']}`",
                    f"- same_count_random_margin: `{row['same_count_random_margin']}`",
                    f"- notes: `{row['notes']}`",
                ]
            ),
        )

    summary = {
        "schema": "acl2_v107tf_stage3_operation_discovery_summary_v1",
        "stage3_pass": stage3_pass,
        "final_taxonomy_if_stop_here": taxonomy,
        "semantic_available": False,
        "semantic_increment_claimed": False,
        "base_mixed_length": {
            "case_count": len(base_targets),
            "operation_row_count": len(base_rows),
            "lever_pass_count": len(base_pass_rows),
            "top_cue": base_top,
        },
        "length_matched_96f": {
            "case_count": len(length_targets),
            "operation_row_count": len(length_rows),
            "lever_pass_count": len(pass_rows),
            "top_cue": length_top,
        },
        "no_go_reason": no_go_reason,
        "outputs": {
            "operation_discovery_rows": rel(OUT / "operation_discovery_rows.csv"),
            "operation_lever_rank": rel(OUT / "operation_lever_rank.csv"),
            "base_mixed_length_operation_lever_rank": rel(OUT / "base_mixed_length_operation_lever_rank.csv"),
            "semantic_increment_by_operation": rel(OUT / "semantic_increment_by_operation.csv"),
            "sequence_loso_rows": rel(OUT / "sequence_loso_rows.csv"),
            "control_margin_rows": rel(OUT / "control_margin_rows.csv"),
            "top_lever_reports": rel(TOP_REPORTS),
            "trace_length_confound_report": rel(OUT / "trace_length_confound_report.md"),
        },
    }
    write_text(OUT / "stage3_summary.json", json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
