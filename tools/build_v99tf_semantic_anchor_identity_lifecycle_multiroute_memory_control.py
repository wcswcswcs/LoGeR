#!/usr/bin/env python3
"""Build ACL2 v99-TF identity-lifecycle evidence artifacts.

This builder is intentionally conservative.  It never promotes a gate from a
diagnostic proxy when the plan asks for identity-specific controls or latent
features that are not present in the traces.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("results/acl2_v99tf_semantic_anchor_identity_lifecycle_multiroute_memory_control")
V98_ROOT = Path("results/acl2_v98tf_semantic_guided_swa_cache_topk_gauge_safe_memory_control")
EXPANDED_TRACE_ROOT = ROOT / "trackN_identity_graph_expanded_probe28"
FALLBACK_TRACE_ROOT = V98_ROOT / "stage7e_ttt_stable_anchor_id_hook"
EPS = 1.0e-9
PREFERRED_TRACE_DIR_NAMES = [
    "trackC3_full_z_vector_probe28_repair",
    "trackC3_vector_sketch_probe28_repair",
    "trackC3_semantic_class_split_probe28_repair",
    "trackC3_split_kv_feature_probe28_repair",
    "trackC3_zwrite_key_sketch_probe28_repair",
    "trackC3_cache_current_feature_probe28_repair",
    "trackN_identity_graph_lifecycle_probe28_repair",
    "trackN_identity_graph_expanded_probe28",
]
CASE_AUDIT_META_FIELDS = [
    "universe_split",
    "good_control_hygiene_core_good_l3_max",
    "good_control_hygiene_l3_threshold",
    "good_control_hygiene_l3_pass",
    "good_control_hygiene_warning",
    "good_control_hygiene_include_for_repair",
    "good_control_hygiene_status",
]


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"read_error": f"{type(exc).__name__}:{exc}", "path": str(path)}
    return payload if isinstance(payload, dict) else {"value": payload}


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(fieldnames or [])
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            clean: dict[str, Any] = {}
            for key in keys:
                value = row.get(key, "")
                if isinstance(value, (dict, list, tuple)):
                    clean[key] = json.dumps(value, ensure_ascii=False, sort_keys=True)
                else:
                    clean[key] = value
            writer.writerow(clean)


def f(value: Any, default: float = math.nan) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, str) and value.strip() in {"", "nan", "None", "null"}:
            return default
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def b(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def clip01(value: float) -> float:
    if not math.isfinite(value):
        return math.nan
    return max(0.0, min(1.0, value))


def finite(values: list[float]) -> list[float]:
    return [float(value) for value in values if math.isfinite(float(value))]


def mean(values: list[float]) -> float:
    vals = finite(values)
    return float(sum(vals) / len(vals)) if vals else math.nan


def median(values: list[float]) -> float:
    vals = finite(values)
    return float(statistics.median(vals)) if vals else math.nan


def pearson(xs: list[float], ys: list[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 2:
        return math.nan
    xvals, yvals = zip(*pairs)
    mx = sum(xvals) / len(xvals)
    my = sum(yvals) / len(yvals)
    vx = sum((x - mx) ** 2 for x in xvals)
    vy = sum((y - my) ** 2 for y in yvals)
    if vx <= 0.0 or vy <= 0.0:
        return math.nan
    return sum((x - mx) * (y - my) for x, y in pairs) / math.sqrt(vx * vy)


def stable_rank(seed: str, items: list[str]) -> list[str]:
    return sorted(items, key=lambda item: hashlib.sha256(f"{seed}:{item}".encode("utf-8")).hexdigest())


def best_threshold(values_by_case: dict[str, float], positives: set[str], negatives: set[str], *, higher_bad: bool) -> dict[str, Any]:
    cases = sorted(positives | negatives)
    values = [values_by_case.get(case, math.nan) for case in cases]
    labels = [1 if case in positives else 0 for case in cases]
    pos = sum(labels)
    neg = len(labels) - pos
    best: dict[str, Any] = {
        "balanced_accuracy": 0.0,
        "threshold": math.nan,
        "tp": 0,
        "tn": 0,
        "fp": 0,
        "fn": pos,
        "pos": pos,
        "neg": neg,
    }
    for threshold in sorted({value for value in values if math.isfinite(value)}):
        preds = [1 if (value >= threshold if higher_bad else value <= threshold) else 0 for value in values]
        tp = sum(1 for pred, label in zip(preds, labels) if pred == 1 and label == 1)
        tn = sum(1 for pred, label in zip(preds, labels) if pred == 0 and label == 0)
        fp = sum(1 for pred, label in zip(preds, labels) if pred == 1 and label == 0)
        fn = sum(1 for pred, label in zip(preds, labels) if pred == 0 and label == 1)
        tpr = tp / pos if pos else 0.0
        tnr = tn / neg if neg else 0.0
        score = 0.5 * (tpr + tnr)
        if (score, tp + tn) > (best["balanced_accuracy"], best["tp"] + best["tn"]):
            best.update({"balanced_accuracy": score, "threshold": threshold, "tp": tp, "tn": tn, "fp": fp, "fn": fn})
    return best


def selected_from_threshold(values_by_case: dict[str, float], threshold: float, *, higher_bad: bool) -> set[str]:
    if not math.isfinite(threshold):
        return set()
    if higher_bad:
        return {case for case, value in values_by_case.items() if math.isfinite(value) and value >= threshold}
    return {case for case, value in values_by_case.items() if math.isfinite(value) and value <= threshold}


def signal(selected: set[str], positives: set[str], negatives: set[str]) -> float:
    recall = len(selected & positives) / len(positives) if positives else 0.0
    fpr = len(selected & negatives) / len(negatives) if negatives else 0.0
    return recall - fpr


def same_count_margin(selected: set[str], cases: list[str], positives: set[str], negatives: set[str], *, seeds: int = 64) -> float:
    actual = signal(selected, positives, negatives)
    controls = []
    for idx in range(seeds):
        control = set(stable_rank(f"same_count_{idx}", cases)[: len(selected)])
        controls.append(signal(control, positives, negatives))
    return actual - median(controls)


def sequence_margin(selected: set[str], cases: list[str], positives: set[str], negatives: set[str], seq_by_case: dict[str, str], *, seeds: int = 64) -> float:
    actual = signal(selected, positives, negatives)
    selected_counts: dict[str, int] = defaultdict(int)
    cases_by_seq: dict[str, list[str]] = defaultdict(list)
    for case in cases:
        cases_by_seq[seq_by_case.get(case, "")].append(case)
    for case in selected:
        selected_counts[seq_by_case.get(case, "")] += 1
    controls = []
    for idx in range(seeds):
        control: set[str] = set()
        for seq, seq_cases in sorted(cases_by_seq.items()):
            control.update(stable_rank(f"seq_count_{idx}_{seq}", seq_cases)[: selected_counts.get(seq, 0)])
        controls.append(signal(control, positives, negatives))
    return actual - median(controls)


def rotated_margin(values_by_case: dict[str, float], threshold: float, positives: set[str], negatives: set[str], *, higher_bad: bool) -> float:
    cases = sorted(positives | negatives)
    if len(cases) < 2 or not math.isfinite(threshold):
        return math.nan
    actual = signal(selected_from_threshold(values_by_case, threshold, higher_bad=higher_bad), positives, negatives)
    values = [values_by_case.get(case, math.nan) for case in cases]
    controls = []
    for shift in range(1, len(cases)):
        rotated = {case: values[(idx - shift) % len(values)] for idx, case in enumerate(cases)}
        controls.append(signal(selected_from_threshold(rotated, threshold, higher_bad=higher_bad), positives, negatives))
    return actual - median(controls)


def split_cases(value: Any) -> list[str]:
    return [item for item in str(value or "").split(";") if item]


def metadata_rows() -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in read_rows(V98_ROOT / "stage1_trackK_swa_v2_strict_stable_eligibility/case_universe_rows.csv"):
        case_id = row.get("case_id", "")
        if case_id:
            rows[case_id] = dict(row)
    return rows


def audit_case_meta(case_meta: dict[str, Any]) -> dict[str, Any]:
    return {field: case_meta.get(field, "") for field in CASE_AUDIT_META_FIELDS}


def action_state_from_delta(delta: float) -> str:
    if not math.isfinite(delta):
        return "not_run"
    if delta < -EPS:
        return "improved"
    if delta > EPS:
        return "worse"
    return "same"


def stage0() -> dict[str, Any]:
    out = ROOT / "trackI_v99_identity_evidence_ledger"
    final = read_json(V98_ROOT / "final_decision/final_decision.json")
    stage7e = read_json(V98_ROOT / "stage7e_ttt_stable_anchor_id_hook/summary.json")
    stage7g = read_json(V98_ROOT / "stage7g_anchor_id_query_head_risk_attribution/summary.json")
    stage7h = read_json(V98_ROOT / "stage7h_prev_ttt_anchor_query_soft_action_pilot/summary.json")
    stage7f = read_json(V98_ROOT / "stage7f_prev_ttt_anchor_gate_action_pilot/summary.json")
    stage7e_rows = read_rows(V98_ROOT / "stage7e_ttt_stable_anchor_id_hook/anchor_id_case_rows.csv")
    cue_rows = read_rows(V98_ROOT / "stage7e_ttt_stable_anchor_id_hook/cue_control_metrics.csv")
    cue_rows_sorted = sorted(
        cue_rows,
        key=lambda row: (b(row.get("gate_pass")), f(row.get("balanced_accuracy")), f(row.get("abs_corr_L3_handoff_transfer_penalty"))),
        reverse=True,
    )
    best_cue = cue_rows_sorted[0] if cue_rows_sorted else {}
    stage7h_case_rows = {row.get("case_id", ""): row for row in read_rows(V98_ROOT / "stage7h_prev_ttt_anchor_query_soft_action_pilot/case_rows.csv")}
    action_rows: list[dict[str, Any]] = []
    for row in read_rows(V98_ROOT / "stage7f_prev_ttt_anchor_gate_action_pilot/variant_summary_rows.csv"):
        action_rows.append({"track": "stage7f_prev_ttt_anchor_gate", **row})
    for row in read_rows(V98_ROOT / "stage7h_prev_ttt_anchor_query_soft_action_pilot/variant_summary_rows.csv"):
        action_rows.append({"track": "stage7h_query_soft", **row})

    best_true_pos = set(split_cases(best_cue.get("true_positive_cases")))
    best_false_pos = set(split_cases(best_cue.get("false_positive_cases")))
    best_missed = set(split_cases(best_cue.get("missed_positive_cases")))
    identity_rows: list[dict[str, Any]] = []
    for row in stage7e_rows:
        case_id = row.get("case_id", "")
        hrow = stage7h_case_rows.get(case_id, {})
        delta = f(hrow.get("action_minus_baseline_ate_rmse_m"))
        identity_rows.append(
            {
                "case_id": case_id,
                "seq": row.get("seq", ""),
                "case_label": row.get("case_label", ""),
                "L3_handoff_transfer_penalty_proxy": row.get("L3_handoff_transfer_penalty_proxy", ""),
                "anchor_id_query_hit": row.get("anchor_id_topk_query_hit_frac_mean", ""),
                "anchor_id_topk_hit": row.get("anchor_id_topk_hit_frac_mean", ""),
                "anchor_id_top1_hit": row.get("anchor_id_top1_hit_frac_mean", ""),
                "anchor_id_route_mass": row.get("anchor_id_route_mass_mean", ""),
                "unique_anchor_ids": row.get("anchor_id_unique_topk_id_count_sum", ""),
                "stage7e_best_cue_selected": case_id in (best_true_pos | best_false_pos),
                "stage7e_best_cue_tp": case_id in best_true_pos,
                "stage7e_best_cue_fp": case_id in best_false_pos,
                "stage7e_best_cue_missed_positive": case_id in best_missed,
                "stage7h_query_soft_action_response": action_state_from_delta(delta),
                "stage7h_action_minus_baseline_ate_rmse_m": hrow.get("action_minus_baseline_ate_rmse_m", ""),
                "stage7h_improvement_ratio_vs_baseline": hrow.get("improvement_ratio_vs_baseline", ""),
                "claim_level": "DIAGNOSTIC_CUE_ONLY_ACTION_NO_GO",
                "recommended_next_track": "TrackN_identity_memory_graph",
            }
        )

    forbidden = [
        "Do not continue aggregate top-k direct action as mainline.",
        "Do not repeat Track L weak-context threshold sweeps as a success path.",
        "Do not repeat Stage7f rho/min/layer sweeps as a mainline.",
        "Do not promote Stage7h query-soft ge75/ge90/rho small sweeps without new identity lifecycle gates.",
        "Do not enter runtime pilot from a diagnostic cue alone.",
        "Do not enter full validation before Track N/O(or K), C3/current-support safety, M2, 6-case pilot, and 12-case pilot all pass.",
    ]
    required = [
        V98_ROOT / "final_decision/final_decision.json",
        V98_ROOT / "stage7e_ttt_stable_anchor_id_hook/summary.json",
        V98_ROOT / "stage7e_ttt_stable_anchor_id_hook/anchor_id_case_rows.csv",
        V98_ROOT / "stage7g_anchor_id_query_head_risk_attribution/summary.json",
        V98_ROOT / "stage7h_prev_ttt_anchor_query_soft_action_pilot/summary.json",
        V98_ROOT / "stage7f_prev_ttt_anchor_gate_action_pilot/summary.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    naming_mismatch = not (V98_ROOT / "stage7e_ttt_stable_anchor_id_hook/case_rows.csv").is_file()
    summary = {
        "schema": "acl2_v99_stage0_identity_evidence_ledger_v1",
        "status": "complete" if not missing else "complete_with_missing_artifacts",
        "gate_pass": not missing and final.get("final_taxonomy") == "F3_IDENTITY_DIAGNOSTIC_CUE_PASS_QUERY_SOFT_ACTION_PILOT_NO_GO",
        "v98_final_taxonomy": final.get("final_taxonomy"),
        "v98_full_method_success": final.get("full_method_success"),
        "v98_runtime_action_pilot_run": final.get("runtime_action_pilot_run"),
        "v98_full_validation_run": final.get("full_validation_run"),
        "stage7e_gate_pass": stage7e.get("gate_pass"),
        "stage7e_case_count": stage7e.get("case_count"),
        "stage7e_best_cue": stage7e.get("best_cue"),
        "stage7e_best_cue_bad_recall": stage7e.get("best_cue_bad_recall"),
        "stage7e_best_cue_good_FPR": stage7e.get("best_cue_good_FPR"),
        "stage7g_gate_pass": stage7g.get("gate_pass"),
        "stage7h_gate_pass": stage7h.get("gate_pass"),
        "stage7h_improved_ate_case_count": stage7h.get("improved_ate_case_count"),
        "stage7h_worse_ate_case_count": stage7h.get("worse_ate_case_count"),
        "stage7f_gate_pass": stage7f.get("gate_pass"),
        "stage7e_plan_expected_case_rows_csv_missing": naming_mismatch,
        "stage7e_actual_case_rows_csv": str(V98_ROOT / "stage7e_ttt_stable_anchor_id_hook/anchor_id_case_rows.csv"),
        "missing_critical_artifact_count": len(missing),
        "best_cue_false_positive_cases": split_cases(best_cue.get("false_positive_cases")),
        "best_cue_missed_positive_cases": split_cases(best_cue.get("missed_positive_cases")),
        "next_route": "Track N identity memory graph on expanded >=24-case trace; no runtime action at Stage0.",
    }
    write_json(out / "summary.json", summary)
    write_rows(out / "identity_case_rows.csv", identity_rows)
    write_rows(out / "action_response_rows.csv", action_rows)
    write_rows(out / "missing_artifacts_report.csv", [{"path": path} for path in missing])
    write_text(out / "no_repeat_list.md", "# No Repeat List\n\n" + "\n".join(f"- {item}" for item in forbidden))
    write_text(
        out / "unresolved_questions.md",
        "# Unresolved Questions\n\n"
        f"- Stage7e best cue false positives: {', '.join(summary['best_cue_false_positive_cases']) or 'none recorded'}.\n"
        f"- Stage7e best cue missed positives: {', '.join(summary['best_cue_missed_positive_cases']) or 'none recorded'}.\n"
        "- v99 must explain these through identity lifecycle, current support, and/or latent gauge evidence before action.\n",
    )
    write_text(
        out / "stage7e_rebuild_report.md",
        "# Stage7e Artifact Naming Check\n\n"
        f"- Expected by plan: `stage7e_ttt_stable_anchor_id_hook/case_rows.csv`.\n"
        f"- Present artifact used: `{summary['stage7e_actual_case_rows_csv']}`.\n"
        f"- This is a naming mismatch only; rows are not synthesized.\n",
    )
    return summary


def parse_int_from_name(path: Path, pattern: str) -> int:
    match = re.search(pattern, path.name)
    if not match:
        return -1
    return int(match.group(1))


def trace_payload_paths() -> tuple[Path, list[Path], bool]:
    for name in PREFERRED_TRACE_DIR_NAMES:
        candidate = ROOT / name
        paths = sorted(candidate.glob("*/TTT_SWA_SAME_RUN/swa_raw_transport_trace/*.pt"))
        if paths:
            return candidate, paths, True
    paths = sorted(EXPANDED_TRACE_ROOT.glob("*/TTT_SWA_SAME_RUN/swa_raw_transport_trace/*.pt"))
    if paths:
        return EXPANDED_TRACE_ROOT, paths, True
    paths = sorted(FALLBACK_TRACE_ROOT.glob("*/TTT_SWA_SAME_RUN/swa_raw_transport_trace/*.pt"))
    return FALLBACK_TRACE_ROOT, paths, False


def torch_load(path: Path) -> Any:
    import torch  # noqa: PLC0415

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def tensor_float(value: Any, default: float = math.nan) -> float:
    try:
        if hasattr(value, "item"):
            return float(value.item())
    except Exception:  # noqa: BLE001
        return default
    return f(value, default)


def tensor_list(value: Any) -> list[float]:
    if value is None:
        return []
    try:
        if hasattr(value, "detach"):
            value = value.detach().cpu()
        if hasattr(value, "flatten"):
            return [float(item) for item in value.flatten().tolist()]
        return [float(item) for item in value]
    except Exception:  # noqa: BLE001
        return []


def head_query_stats(mask: Any) -> dict[str, float]:
    try:
        import torch  # noqa: PLC0415

        if mask is None or not hasattr(mask, "detach"):
            return {}
        t = mask.detach().cpu().bool()
        if t.ndim < 4:
            return {}
        # B,H,Q,K -> H query-hit fraction.
        by_head = t.any(dim=-1).float().mean(dim=(0, 2))
        vals = [float(item) for item in by_head.tolist()]
        return {
            "query_head_hit_mean": mean(vals),
            "query_head_hit_max": max(vals) if vals else math.nan,
            "query_head_hit_median": median(vals),
            "query_head_ge50_frac": sum(1 for item in vals if item >= 0.50) / len(vals) if vals else math.nan,
            "query_head_ge75_frac": sum(1 for item in vals if item >= 0.75) / len(vals) if vals else math.nan,
            "query_head_ge90_frac": sum(1 for item in vals if item >= 0.90) / len(vals) if vals else math.nan,
        }
    except Exception:  # noqa: BLE001
        return {}


def anchor_id_stats(mask: Any, ids: Any) -> dict[str, float]:
    try:
        import torch  # noqa: PLC0415

        if mask is None or ids is None or not hasattr(mask, "detach") or not hasattr(ids, "detach"):
            return {}
        m = mask.detach().cpu().bool()
        i = ids.detach().cpu().long()
        valid = i[m & (i >= 0)]
        count = int(valid.numel())
        if count == 0:
            return {"valid_anchor_id_hit_count": 0, "unique_anchor_id_count": 0}
        unique, counts = torch.unique(valid, return_counts=True)
        probs = counts.float() / counts.sum().clamp_min(1)
        entropy = float(-(probs * torch.log2(probs.clamp_min(1.0e-12))).sum().item())
        return {
            "valid_anchor_id_hit_count": count,
            "unique_anchor_id_count": int(unique.numel()),
            "anchor_id_entropy": entropy,
            "dominant_anchor_id_frac": float(counts.max().item() / max(count, 1)),
            "anchor_fragmentation_score": float(unique.numel() / max(count, 1)),
        }
    except Exception:  # noqa: BLE001
        return {}


def lifecycle_scores(rows: list[dict[str, Any]]) -> dict[str, float]:
    residual_scores = []
    stale_scores = []
    support_scores = []
    cache_current_l2_scores = []
    cache_current_route_l2_scores = []
    cache_current_cos_risks = []
    cache_current_v_l2_scores = []
    cache_current_v_route_l2_scores = []
    cache_current_v_cos_risks = []
    head75_cache_current_v_cos_risks = []
    headmax_cache_current_v_cos_risks = []
    cache_current_k_sketch_scores = []
    cache_current_v_sketch_scores = []
    z_write_current_q_sketch_scores = []
    z_write_cache_k_sketch_scores = []
    cache_current_k_vec_scores = []
    cache_current_v_vec_scores = []
    z_write_current_q_vec_scores = []
    z_write_cache_k_vec_scores = []
    z_write_current_q_vec_projected_scores = []
    z_write_cache_k_vec_projected_scores = []
    z_write_key_norm_scores = []
    z_write_sketch_cache_current_scores = []
    for row in rows:
        query = f(row.get("query_head_hit_frac"))
        source_residual = f(row.get("source_residual_mean"))
        current_residual = f(row.get("current_feature_residual_mean"))
        retention = f(row.get("source_retention_mean"))
        route_mass = f(row.get("topk_route_mass_mean"))
        cache_current_l2 = f(row.get("z_cache_current_l2_mean"))
        cache_current_route_l2 = f(row.get("z_cache_current_l2_route_weighted_mean"))
        cache_current_cos = f(row.get("z_cache_current_cos_mean"))
        cache_current_v_l2 = f(row.get("z_cache_current_v_l2_mean"))
        cache_current_v_route_l2 = f(row.get("z_cache_current_v_l2_route_weighted_mean"))
        cache_current_v_cos = f(row.get("z_cache_current_v_cos_mean"))
        head75 = f(row.get("query_head_ge75_frac"))
        headmax = f(row.get("query_head_hit_max"))
        cache_current_k_sketch = f(row.get("z_cache_current_k_sketch_residual"))
        cache_current_v_sketch = f(row.get("z_cache_current_v_sketch_residual"))
        z_write_current_q_sketch = f(row.get("z_write_current_q_sketch_residual"))
        z_write_cache_k_sketch = f(row.get("z_write_cache_k_sketch_residual"))
        cache_current_k_vec = f(row.get("z_cache_current_k_vec_residual"))
        cache_current_v_vec = f(row.get("z_cache_current_v_vec_residual"))
        z_write_current_q_vec = f(row.get("z_write_current_q_vec_residual"))
        z_write_cache_k_vec = f(row.get("z_write_cache_k_vec_residual"))
        z_write_current_q_vec_projected = f(row.get("z_write_current_q_vec_projected_residual"))
        z_write_cache_k_vec_projected = f(row.get("z_write_cache_k_vec_projected_residual"))
        z_write_key_norm = f(row.get("z_write_key_norm_mean"))
        z_write_key_sketch_norm = f(row.get("z_write_key_sketch_norm_mean"))
        if all(math.isfinite(v) for v in [query, source_residual, current_residual]):
            residual_scores.append(query * source_residual * current_residual)
        if all(math.isfinite(v) for v in [query, retention, current_residual]):
            stale_scores.append(query * max(0.0, 1.0 - retention) * current_residual)
        if all(math.isfinite(v) for v in [query, route_mass, retention]):
            support_scores.append(query * route_mass * retention)
        if all(math.isfinite(v) for v in [query, cache_current_l2]):
            cache_current_l2_scores.append(query * cache_current_l2)
        if all(math.isfinite(v) for v in [query, cache_current_route_l2]):
            cache_current_route_l2_scores.append(query * cache_current_route_l2)
        if all(math.isfinite(v) for v in [query, cache_current_cos]):
            cache_current_cos_risks.append(query * (1.0 - cache_current_cos))
        if all(math.isfinite(v) for v in [query, cache_current_v_l2]):
            cache_current_v_l2_scores.append(query * cache_current_v_l2)
        if all(math.isfinite(v) for v in [query, cache_current_v_route_l2]):
            cache_current_v_route_l2_scores.append(query * cache_current_v_route_l2)
        if all(math.isfinite(v) for v in [query, cache_current_v_cos]):
            cache_current_v_cos_risks.append(query * (1.0 - cache_current_v_cos))
        if all(math.isfinite(v) for v in [head75, cache_current_v_cos]):
            head75_cache_current_v_cos_risks.append(head75 * (1.0 - cache_current_v_cos))
        if all(math.isfinite(v) for v in [headmax, cache_current_v_cos]):
            headmax_cache_current_v_cos_risks.append(headmax * (1.0 - cache_current_v_cos))
        if all(math.isfinite(v) for v in [query, cache_current_k_sketch]):
            cache_current_k_sketch_scores.append(query * cache_current_k_sketch)
        if all(math.isfinite(v) for v in [query, cache_current_v_sketch]):
            cache_current_v_sketch_scores.append(query * cache_current_v_sketch)
        if all(math.isfinite(v) for v in [query, z_write_current_q_sketch]):
            z_write_current_q_sketch_scores.append(query * z_write_current_q_sketch)
        if all(math.isfinite(v) for v in [query, z_write_cache_k_sketch]):
            z_write_cache_k_sketch_scores.append(query * z_write_cache_k_sketch)
        if all(math.isfinite(v) for v in [query, cache_current_k_vec]):
            cache_current_k_vec_scores.append(query * cache_current_k_vec)
        if all(math.isfinite(v) for v in [query, cache_current_v_vec]):
            cache_current_v_vec_scores.append(query * cache_current_v_vec)
        if all(math.isfinite(v) for v in [query, z_write_current_q_vec]):
            z_write_current_q_vec_scores.append(query * z_write_current_q_vec)
        if all(math.isfinite(v) for v in [query, z_write_cache_k_vec]):
            z_write_cache_k_vec_scores.append(query * z_write_cache_k_vec)
        if all(math.isfinite(v) for v in [query, z_write_current_q_vec_projected]):
            z_write_current_q_vec_projected_scores.append(query * z_write_current_q_vec_projected)
        if all(math.isfinite(v) for v in [query, z_write_cache_k_vec_projected]):
            z_write_cache_k_vec_projected_scores.append(query * z_write_cache_k_vec_projected)
        if all(math.isfinite(v) for v in [query, z_write_key_norm]):
            z_write_key_norm_scores.append(query * z_write_key_norm)
        if all(math.isfinite(v) for v in [query, z_write_key_sketch_norm, cache_current_route_l2]):
            z_write_sketch_cache_current_scores.append(query * z_write_key_sketch_norm * cache_current_route_l2)
    return {
        "anchor_lifecycle_row_count": float(len(rows)),
        "anchor_lifecycle_residual_risk_max": max(residual_scores) if residual_scores else math.nan,
        "anchor_lifecycle_residual_risk_mean": mean(residual_scores),
        "anchor_lifecycle_stale_risk_max": max(stale_scores) if stale_scores else math.nan,
        "anchor_lifecycle_stale_risk_mean": mean(stale_scores),
        "anchor_lifecycle_support_score_max": max(support_scores) if support_scores else math.nan,
        "anchor_lifecycle_support_score_mean": mean(support_scores),
        "anchor_cache_current_l2_risk_max": max(cache_current_l2_scores) if cache_current_l2_scores else math.nan,
        "anchor_cache_current_l2_risk_mean": mean(cache_current_l2_scores),
        "anchor_cache_current_route_l2_risk_max": max(cache_current_route_l2_scores) if cache_current_route_l2_scores else math.nan,
        "anchor_cache_current_route_l2_risk_mean": mean(cache_current_route_l2_scores),
        "anchor_cache_current_cos_risk_max": max(cache_current_cos_risks) if cache_current_cos_risks else math.nan,
        "anchor_cache_current_cos_risk_mean": mean(cache_current_cos_risks),
        "anchor_cache_current_v_l2_risk_max": max(cache_current_v_l2_scores) if cache_current_v_l2_scores else math.nan,
        "anchor_cache_current_v_l2_risk_mean": mean(cache_current_v_l2_scores),
        "anchor_cache_current_v_route_l2_risk_max": max(cache_current_v_route_l2_scores) if cache_current_v_route_l2_scores else math.nan,
        "anchor_cache_current_v_route_l2_risk_mean": mean(cache_current_v_route_l2_scores),
        "anchor_cache_current_v_cos_risk_max": max(cache_current_v_cos_risks) if cache_current_v_cos_risks else math.nan,
        "anchor_cache_current_v_cos_risk_mean": mean(cache_current_v_cos_risks),
        "anchor_head75_cache_current_v_cos_risk_max": max(head75_cache_current_v_cos_risks) if head75_cache_current_v_cos_risks else math.nan,
        "anchor_head75_cache_current_v_cos_risk_mean": mean(head75_cache_current_v_cos_risks),
        "anchor_headmax_cache_current_v_cos_risk_max": max(headmax_cache_current_v_cos_risks) if headmax_cache_current_v_cos_risks else math.nan,
        "anchor_headmax_cache_current_v_cos_risk_mean": mean(headmax_cache_current_v_cos_risks),
        "anchor_cache_current_k_sketch_risk_max": max(cache_current_k_sketch_scores) if cache_current_k_sketch_scores else math.nan,
        "anchor_cache_current_k_sketch_risk_mean": mean(cache_current_k_sketch_scores),
        "anchor_cache_current_v_sketch_risk_max": max(cache_current_v_sketch_scores) if cache_current_v_sketch_scores else math.nan,
        "anchor_cache_current_v_sketch_risk_mean": mean(cache_current_v_sketch_scores),
        "anchor_z_write_current_q_sketch_risk_max": max(z_write_current_q_sketch_scores) if z_write_current_q_sketch_scores else math.nan,
        "anchor_z_write_current_q_sketch_risk_mean": mean(z_write_current_q_sketch_scores),
        "anchor_z_write_cache_k_sketch_risk_max": max(z_write_cache_k_sketch_scores) if z_write_cache_k_sketch_scores else math.nan,
        "anchor_z_write_cache_k_sketch_risk_mean": mean(z_write_cache_k_sketch_scores),
        "anchor_cache_current_k_vec_risk_max": max(cache_current_k_vec_scores) if cache_current_k_vec_scores else math.nan,
        "anchor_cache_current_k_vec_risk_mean": mean(cache_current_k_vec_scores),
        "anchor_cache_current_v_vec_risk_max": max(cache_current_v_vec_scores) if cache_current_v_vec_scores else math.nan,
        "anchor_cache_current_v_vec_risk_mean": mean(cache_current_v_vec_scores),
        "anchor_z_write_current_q_vec_risk_max": max(z_write_current_q_vec_scores) if z_write_current_q_vec_scores else math.nan,
        "anchor_z_write_current_q_vec_risk_mean": mean(z_write_current_q_vec_scores),
        "anchor_z_write_cache_k_vec_risk_max": max(z_write_cache_k_vec_scores) if z_write_cache_k_vec_scores else math.nan,
        "anchor_z_write_cache_k_vec_risk_mean": mean(z_write_cache_k_vec_scores),
        "anchor_z_write_current_q_vec_projected_risk_max": (
            max(z_write_current_q_vec_projected_scores) if z_write_current_q_vec_projected_scores else math.nan
        ),
        "anchor_z_write_current_q_vec_projected_risk_mean": mean(z_write_current_q_vec_projected_scores),
        "anchor_z_write_cache_k_vec_projected_risk_max": (
            max(z_write_cache_k_vec_projected_scores) if z_write_cache_k_vec_projected_scores else math.nan
        ),
        "anchor_z_write_cache_k_vec_projected_risk_mean": mean(z_write_cache_k_vec_projected_scores),
        "anchor_z_write_key_norm_score_max": max(z_write_key_norm_scores) if z_write_key_norm_scores else math.nan,
        "anchor_z_write_key_norm_score_mean": mean(z_write_key_norm_scores),
        "anchor_z_write_sketch_cache_current_risk_max": max(z_write_sketch_cache_current_scores) if z_write_sketch_cache_current_scores else math.nan,
        "anchor_z_write_sketch_cache_current_risk_mean": mean(z_write_sketch_cache_current_scores),
    }


def collect_trace_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    trace_root, paths, expanded_available = trace_payload_paths()
    meta = metadata_rows()
    edge_rows: list[dict[str, Any]] = []
    anchor_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    payload_keys: set[str] = set()
    for path in paths:
        case_id = path.parents[2].name
        try:
            payload = torch_load(path)
        except Exception as exc:  # noqa: BLE001
            errors.append({"path": str(path), "error": f"{type(exc).__name__}:{exc}"})
            continue
        if not isinstance(payload, dict):
            errors.append({"path": str(path), "error": f"unexpected_payload_type:{type(payload).__name__}"})
            continue
        payload_keys.update(str(key) for key in payload.keys())
        case_meta = meta.get(case_id, {})
        chunk_idx = int(f(payload.get("chunk_idx"), parse_int_from_name(path, r"chunk_(\d+)_")))
        source_chunk_idx = int(f(payload.get("ttt_prev_stable_anchor_source_chunk_idx"), -1))
        age = chunk_idx - source_chunk_idx if source_chunk_idx >= 0 and chunk_idx >= 0 else math.nan
        layer = int(f(payload.get("swa_layer_idx", payload.get("layer")), parse_int_from_name(path, r"layer_(\d+)")))
        hit_mask = payload.get("current_Q_to_cache_K_topk_ttt_prev_stable_anchor_hit_mask")
        anchor_ids = payload.get("current_Q_to_cache_K_topk_ttt_prev_stable_anchor_ids")
        stats = {}
        stats.update(head_query_stats(hit_mask))
        stats.update(anchor_id_stats(hit_mask, anchor_ids))
        payload_lifecycle_rows: list[dict[str, Any]] = []
        raw_lifecycle = payload.get("ttt_prev_stable_anchor_lifecycle_rows")
        if isinstance(raw_lifecycle, list):
            for raw_row in raw_lifecycle:
                if not isinstance(raw_row, dict):
                    continue
                life_row = {
                    "case_id": case_id,
                    "seq": case_meta.get("seq", case_id.split("_")[0]),
                    "case_label": case_meta.get("case_label", ""),
                    "prev_chunk": case_meta.get("prev_chunk", ""),
                    "curr_chunk": case_meta.get("curr_chunk", ""),
                    "failure_type": case_meta.get("failure_type", ""),
                    "L3_handoff_transfer_penalty_proxy": f(case_meta.get("L3_handoff_transfer_penalty_proxy")),
                    **audit_case_meta(case_meta),
                    "payload": str(path),
                    "layer": layer,
                    "chunk_idx": chunk_idx,
                    **raw_row,
                }
                source_residual = f(life_row.get("source_residual_mean"))
                current_residual = f(life_row.get("current_feature_residual_mean"))
                query_frac = f(life_row.get("query_head_hit_frac"))
                retention = f(life_row.get("source_retention_mean"))
                life_row["anchor_lifecycle_residual_risk"] = (
                    query_frac * source_residual * current_residual
                    if all(math.isfinite(v) for v in [query_frac, source_residual, current_residual])
                    else math.nan
                )
                life_row["anchor_lifecycle_stale_risk"] = (
                    query_frac * max(0.0, 1.0 - retention) * current_residual
                    if all(math.isfinite(v) for v in [query_frac, retention, current_residual])
                    else math.nan
                )
                cache_current_l2 = f(life_row.get("z_cache_current_l2_mean"))
                cache_current_route_l2 = f(life_row.get("z_cache_current_l2_route_weighted_mean"))
                cache_current_cos = f(life_row.get("z_cache_current_cos_mean"))
                cache_current_v_l2 = f(life_row.get("z_cache_current_v_l2_mean"))
                cache_current_v_route_l2 = f(life_row.get("z_cache_current_v_l2_route_weighted_mean"))
                cache_current_v_cos = f(life_row.get("z_cache_current_v_cos_mean"))
                life_row["anchor_cache_current_l2_risk"] = (
                    query_frac * cache_current_l2
                    if all(math.isfinite(v) for v in [query_frac, cache_current_l2])
                    else math.nan
                )
                life_row["anchor_cache_current_route_l2_risk"] = (
                    query_frac * cache_current_route_l2
                    if all(math.isfinite(v) for v in [query_frac, cache_current_route_l2])
                    else math.nan
                )
                life_row["anchor_cache_current_cos_risk"] = (
                    query_frac * (1.0 - cache_current_cos)
                    if all(math.isfinite(v) for v in [query_frac, cache_current_cos])
                    else math.nan
                )
                life_row["anchor_cache_current_v_l2_risk"] = (
                    query_frac * cache_current_v_l2
                    if all(math.isfinite(v) for v in [query_frac, cache_current_v_l2])
                    else math.nan
                )
                life_row["anchor_cache_current_v_route_l2_risk"] = (
                    query_frac * cache_current_v_route_l2
                    if all(math.isfinite(v) for v in [query_frac, cache_current_v_route_l2])
                    else math.nan
                )
                life_row["anchor_cache_current_v_cos_risk"] = (
                    query_frac * (1.0 - cache_current_v_cos)
                    if all(math.isfinite(v) for v in [query_frac, cache_current_v_cos])
                    else math.nan
                )
                head75 = f(life_row.get("query_head_ge75_frac"))
                headmax = f(life_row.get("query_head_hit_max"))
                cache_current_k_sketch = f(life_row.get("z_cache_current_k_sketch_residual"))
                cache_current_v_sketch = f(life_row.get("z_cache_current_v_sketch_residual"))
                z_write_current_q_sketch = f(life_row.get("z_write_current_q_sketch_residual"))
                z_write_cache_k_sketch = f(life_row.get("z_write_cache_k_sketch_residual"))
                cache_current_k_vec = f(life_row.get("z_cache_current_k_vec_residual"))
                cache_current_v_vec = f(life_row.get("z_cache_current_v_vec_residual"))
                z_write_current_q_vec = f(life_row.get("z_write_current_q_vec_residual"))
                z_write_cache_k_vec = f(life_row.get("z_write_cache_k_vec_residual"))
                z_write_current_q_vec_projected = f(life_row.get("z_write_current_q_vec_projected_residual"))
                z_write_cache_k_vec_projected = f(life_row.get("z_write_cache_k_vec_projected_residual"))
                life_row["anchor_head75_cache_current_v_cos_risk"] = (
                    head75 * (1.0 - cache_current_v_cos)
                    if all(math.isfinite(v) for v in [head75, cache_current_v_cos])
                    else math.nan
                )
                life_row["anchor_headmax_cache_current_v_cos_risk"] = (
                    headmax * (1.0 - cache_current_v_cos)
                    if all(math.isfinite(v) for v in [headmax, cache_current_v_cos])
                    else math.nan
                )
                life_row["anchor_cache_current_k_sketch_risk"] = (
                    query_frac * cache_current_k_sketch
                    if all(math.isfinite(v) for v in [query_frac, cache_current_k_sketch])
                    else math.nan
                )
                life_row["anchor_cache_current_v_sketch_risk"] = (
                    query_frac * cache_current_v_sketch
                    if all(math.isfinite(v) for v in [query_frac, cache_current_v_sketch])
                    else math.nan
                )
                life_row["anchor_z_write_current_q_sketch_risk"] = (
                    query_frac * z_write_current_q_sketch
                    if all(math.isfinite(v) for v in [query_frac, z_write_current_q_sketch])
                    else math.nan
                )
                life_row["anchor_z_write_cache_k_sketch_risk"] = (
                    query_frac * z_write_cache_k_sketch
                    if all(math.isfinite(v) for v in [query_frac, z_write_cache_k_sketch])
                    else math.nan
                )
                life_row["anchor_cache_current_k_vec_risk"] = (
                    query_frac * cache_current_k_vec
                    if all(math.isfinite(v) for v in [query_frac, cache_current_k_vec])
                    else math.nan
                )
                life_row["anchor_cache_current_v_vec_risk"] = (
                    query_frac * cache_current_v_vec
                    if all(math.isfinite(v) for v in [query_frac, cache_current_v_vec])
                    else math.nan
                )
                life_row["anchor_z_write_current_q_vec_risk"] = (
                    query_frac * z_write_current_q_vec
                    if all(math.isfinite(v) for v in [query_frac, z_write_current_q_vec])
                    else math.nan
                )
                life_row["anchor_z_write_cache_k_vec_risk"] = (
                    query_frac * z_write_cache_k_vec
                    if all(math.isfinite(v) for v in [query_frac, z_write_cache_k_vec])
                    else math.nan
                )
                life_row["anchor_z_write_current_q_vec_projected_risk"] = (
                    query_frac * z_write_current_q_vec_projected
                    if all(math.isfinite(v) for v in [query_frac, z_write_current_q_vec_projected])
                    else math.nan
                )
                life_row["anchor_z_write_cache_k_vec_projected_risk"] = (
                    query_frac * z_write_cache_k_vec_projected
                    if all(math.isfinite(v) for v in [query_frac, z_write_cache_k_vec_projected])
                    else math.nan
                )
                z_write_key_norm = f(life_row.get("z_write_key_norm_mean"))
                z_write_key_sketch_norm = f(life_row.get("z_write_key_sketch_norm_mean"))
                life_row["anchor_z_write_key_norm_score"] = (
                    query_frac * z_write_key_norm
                    if all(math.isfinite(v) for v in [query_frac, z_write_key_norm])
                    else math.nan
                )
                life_row["anchor_z_write_sketch_cache_current_risk"] = (
                    query_frac * z_write_key_sketch_norm * cache_current_route_l2
                    if all(math.isfinite(v) for v in [query_frac, z_write_key_sketch_norm, cache_current_route_l2])
                    else math.nan
                )
                payload_lifecycle_rows.append(life_row)
                anchor_rows.append(life_row)
        lifecycle = lifecycle_scores(payload_lifecycle_rows)
        cache_support = mean(
            [
                f(payload.get("cache_k_stability_mean")),
                f(payload.get("cache_v_stability_mean")),
                f(payload.get("ttt_prev_stable_anchor_topk_hit_frac_mean")),
                f(payload.get("ttt_prev_stable_anchor_top1_hit_frac_mean")),
                f(payload.get("ttt_prev_stable_anchor_route_mass_mean")),
            ]
        )
        uncertainty = mean(
            [
                f(payload.get("unreliable_pair_mass_mean")),
                f(payload.get("route_entropy_mean")),
                f(payload.get("feature_transport_residual_mean")),
            ]
        )
        support_proxy = clip01(0.5 * cache_support + 0.5 * (1.0 - uncertainty)) if math.isfinite(cache_support) and math.isfinite(uncertainty) else math.nan
        age_norm = (age / (age + 1.0)) if math.isfinite(age) and age >= 0.0 else math.nan
        query_hit = f(payload.get("ttt_prev_stable_anchor_topk_query_hit_frac_mean"))
        feature_residual = f(payload.get("feature_transport_residual_mean"))
        unreliable_mass = f(payload.get("unreliable_pair_mass_mean"))
        edge_rows.append(
            {
                "case_id": case_id,
                "seq": case_meta.get("seq", case_id.split("_")[0]),
                "case_label": case_meta.get("case_label", ""),
                "prev_chunk": case_meta.get("prev_chunk", ""),
                "curr_chunk": case_meta.get("curr_chunk", ""),
                "failure_type": case_meta.get("failure_type", ""),
                "L3_handoff_transfer_penalty_proxy": f(case_meta.get("L3_handoff_transfer_penalty_proxy")),
                **audit_case_meta(case_meta),
                "payload": str(path),
                "layer": layer,
                "chunk_idx": chunk_idx,
                "source_chunk_idx": source_chunk_idx if source_chunk_idx >= 0 else "",
                "anchor_age": age,
                "anchor_age_norm": age_norm,
                "anchor_id_available": b(payload.get("ttt_prev_stable_anchor_identity_available")),
                "source_anchor_token_count": f(payload.get("ttt_prev_stable_anchor_source_token_count")),
                "full_anchor_token_count": f(payload.get("ttt_prev_stable_anchor_full_token_count")),
                "anchor_id_topk_hit_frac": f(payload.get("ttt_prev_stable_anchor_topk_hit_frac_mean")),
                "anchor_id_topk_query_hit_frac": query_hit,
                "anchor_id_top1_hit_frac": f(payload.get("ttt_prev_stable_anchor_top1_hit_frac_mean")),
                "anchor_id_route_mass": f(payload.get("ttt_prev_stable_anchor_route_mass_mean")),
                "anchor_id_retention": f(payload.get("ttt_prev_stable_anchor_retention_mean")),
                "anchor_id_residual": f(payload.get("ttt_prev_stable_anchor_residual_mean")),
                "cache_k_stability": f(payload.get("cache_k_stability_mean")),
                "cache_v_stability": f(payload.get("cache_v_stability_mean")),
                "feature_transport_residual": feature_residual,
                "route_entropy": f(payload.get("route_entropy_mean")),
                "stable_pair_mass": f(payload.get("stable_pair_mass_mean")),
                "unreliable_pair_mass": unreliable_mass,
                **stats,
                "current_support_proxy": support_proxy,
                "stale_unsupported_hit_score": query_hit * (1.0 - support_proxy) * age_norm if all(math.isfinite(v) for v in [query_hit, support_proxy, age_norm]) else math.nan,
                "fresh_supported_score": query_hit * support_proxy * (1.0 - age_norm) if all(math.isfinite(v) for v in [query_hit, support_proxy, age_norm]) else math.nan,
                "identity_read_risk_proxy": query_hit * mean([feature_residual, unreliable_mass, 1.0 - support_proxy]) if all(math.isfinite(v) for v in [query_hit, feature_residual, unreliable_mass, support_proxy]) else math.nan,
                **lifecycle,
                "trace_root": str(trace_root),
            }
        )
    diagnostics = {
        "trace_root": str(trace_root),
        "expanded_trace_available": expanded_available,
        "trace_payload_file_count": len(paths),
        "trace_read_error_count": len(errors),
        "anchor_lifecycle_row_count": len(anchor_rows),
        "payload_key_sample": sorted(payload_keys)[:200],
    }
    return edge_rows, anchor_rows, errors, diagnostics


def aggregate_case_rows(edge_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in edge_rows:
        grouped[str(row.get("case_id", ""))].append(row)
    fields = [
        "anchor_age",
        "anchor_age_norm",
        "anchor_id_topk_hit_frac",
        "anchor_id_topk_query_hit_frac",
        "anchor_id_top1_hit_frac",
        "anchor_id_route_mass",
        "anchor_id_retention",
        "anchor_id_residual",
        "cache_k_stability",
        "cache_v_stability",
        "feature_transport_residual",
        "route_entropy",
        "stable_pair_mass",
        "unreliable_pair_mass",
        "query_head_hit_mean",
        "query_head_hit_max",
        "query_head_hit_median",
        "query_head_ge50_frac",
        "query_head_ge75_frac",
        "query_head_ge90_frac",
        "valid_anchor_id_hit_count",
        "unique_anchor_id_count",
        "anchor_id_entropy",
        "dominant_anchor_id_frac",
        "anchor_fragmentation_score",
        "current_support_proxy",
        "stale_unsupported_hit_score",
        "fresh_supported_score",
        "identity_read_risk_proxy",
        "anchor_lifecycle_row_count",
        "anchor_lifecycle_residual_risk_max",
        "anchor_lifecycle_residual_risk_mean",
        "anchor_lifecycle_stale_risk_max",
        "anchor_lifecycle_stale_risk_mean",
        "anchor_lifecycle_support_score_max",
        "anchor_lifecycle_support_score_mean",
        "anchor_cache_current_l2_risk_max",
        "anchor_cache_current_l2_risk_mean",
        "anchor_cache_current_route_l2_risk_max",
        "anchor_cache_current_route_l2_risk_mean",
        "anchor_cache_current_cos_risk_max",
        "anchor_cache_current_cos_risk_mean",
        "anchor_cache_current_v_l2_risk_max",
        "anchor_cache_current_v_l2_risk_mean",
        "anchor_cache_current_v_route_l2_risk_max",
        "anchor_cache_current_v_route_l2_risk_mean",
        "anchor_cache_current_v_cos_risk_max",
        "anchor_cache_current_v_cos_risk_mean",
        "anchor_head75_cache_current_v_cos_risk_max",
        "anchor_head75_cache_current_v_cos_risk_mean",
        "anchor_headmax_cache_current_v_cos_risk_max",
        "anchor_headmax_cache_current_v_cos_risk_mean",
        "anchor_cache_current_k_sketch_risk_max",
        "anchor_cache_current_k_sketch_risk_mean",
        "anchor_cache_current_v_sketch_risk_max",
        "anchor_cache_current_v_sketch_risk_mean",
        "anchor_z_write_current_q_sketch_risk_max",
        "anchor_z_write_current_q_sketch_risk_mean",
        "anchor_z_write_cache_k_sketch_risk_max",
        "anchor_z_write_cache_k_sketch_risk_mean",
        "anchor_cache_current_k_vec_risk_max",
        "anchor_cache_current_k_vec_risk_mean",
        "anchor_cache_current_v_vec_risk_max",
        "anchor_cache_current_v_vec_risk_mean",
        "anchor_z_write_current_q_vec_risk_max",
        "anchor_z_write_current_q_vec_risk_mean",
        "anchor_z_write_cache_k_vec_risk_max",
        "anchor_z_write_cache_k_vec_risk_mean",
        "anchor_z_write_current_q_vec_projected_risk_max",
        "anchor_z_write_current_q_vec_projected_risk_mean",
        "anchor_z_write_cache_k_vec_projected_risk_max",
        "anchor_z_write_cache_k_vec_projected_risk_mean",
        "anchor_z_write_key_norm_score_max",
        "anchor_z_write_key_norm_score_mean",
        "anchor_z_write_sketch_cache_current_risk_max",
        "anchor_z_write_sketch_cache_current_risk_mean",
    ]
    rows: list[dict[str, Any]] = []
    for case_id, parts in sorted(grouped.items()):
        first = parts[0]
        row: dict[str, Any] = {
            "case_id": case_id,
            "seq": first.get("seq", ""),
            "case_label": first.get("case_label", ""),
            "prev_chunk": first.get("prev_chunk", ""),
            "curr_chunk": first.get("curr_chunk", ""),
            "failure_type": first.get("failure_type", ""),
            "L3_handoff_transfer_penalty_proxy": first.get("L3_handoff_transfer_penalty_proxy", ""),
            "payload_count": len(parts),
            "layer_count": len({part.get("layer") for part in parts}),
            "anchor_id_available_payload_count": sum(1 for part in parts if b(part.get("anchor_id_available"))),
        }
        for field in CASE_AUDIT_META_FIELDS:
            row[field] = first.get(field, "")
        for field in fields:
            row[f"{field}_mean"] = mean([f(part.get(field)) for part in parts])
            row[f"{field}_median"] = median([f(part.get(field)) for part in parts])
        rows.append(row)
    return rows


def evaluate_patterns(rows: list[dict[str, Any]], specs: dict[str, tuple[str, bool]], *, min_cases: int, min_corr: float, require_identity_specific: bool) -> list[dict[str, Any]]:
    traced = [row for row in rows if int(f(row.get("payload_count"), 0)) > 0]
    seq_by_case = {str(row["case_id"]): str(row.get("seq", "")) for row in traced}
    labels = {str(row["case_id"]): str(row.get("case_label", "")) for row in traced}
    positives = {case for case, label in labels.items() if label != "good"}
    negatives = {case for case, label in labels.items() if label == "good"}
    l3_by_case = {str(row["case_id"]): f(row.get("L3_handoff_transfer_penalty_proxy")) for row in traced}
    out: list[dict[str, Any]] = []
    for cue, (field, higher_bad) in specs.items():
        values = {str(row["case_id"]): f(row.get(field)) for row in traced if math.isfinite(f(row.get(field)))}
        cases = sorted((positives | negatives) & set(values))
        pos = positives & set(cases)
        neg = negatives & set(cases)
        best = best_threshold(values, pos, neg, higher_bad=higher_bad)
        threshold = f(best.get("threshold"))
        selected = selected_from_threshold(values, threshold, higher_bad=higher_bad)
        corr = pearson([values.get(case, math.nan) for case in cases], [l3_by_case.get(case, math.nan) for case in cases])
        recall = len(selected & pos) / len(pos) if pos else 0.0
        fpr = len(selected & neg) / len(neg) if neg else 0.0
        same_margin = same_count_margin(selected, cases, pos, neg) if cases else math.nan
        seq_margin = sequence_margin(selected, cases, pos, neg, seq_by_case) if cases else math.nan
        value_rotation_margin = rotated_margin(values, threshold, pos, neg, higher_bad=higher_bad)
        # Current traces expose anchor-id masks and ids, but the candidate case
        # scores below do not depend on the anchor id labels themselves.  A true
        # anchor-id rotation control therefore cannot be counted as positive.
        identity_specific_margin = 0.0 if require_identity_specific else value_rotation_margin
        direction_correct = (corr >= 0.0 if higher_bad else corr <= 0.0) if math.isfinite(corr) else False
        selected_pos_seq = Counter(seq_by_case.get(case, "") for case in selected & pos)
        max_seq_frac = max(selected_pos_seq.values()) / max(len(selected & pos), 1) if selected_pos_seq else 0.0
        gate = (
            len(cases) >= min_cases
            and len({seq_by_case.get(case, "") for case in cases}) >= 4
            and recall >= 0.65
            and fpr <= 0.25
            and abs(corr) >= min_corr
            and direction_correct
            and same_margin >= 0.05
            and seq_margin >= 0.05
            and identity_specific_margin >= 0.05
            and max_seq_frac <= 0.67
        )
        out.append(
            {
                "cue_name": cue,
                "field": field,
                "direction": "higher_bad" if higher_bad else "lower_bad",
                "available_case_count": len(cases),
                "sequence_coverage": len({seq_by_case.get(case, "") for case in cases}),
                "positive_case_count": len(pos),
                "good_control_case_count": len(neg),
                "threshold": threshold,
                "balanced_accuracy": best.get("balanced_accuracy"),
                "bad_recall": recall,
                "good_FPR": fpr,
                "abs_corr_L3": abs(corr) if math.isfinite(corr) else math.nan,
                "corr_L3": corr,
                "direction_correct": direction_correct,
                "same_count_margin": same_margin,
                "sequence_margin": seq_margin,
                "case_value_rotation_margin": value_rotation_margin,
                "anchor_id_rotation_margin": identity_specific_margin,
                "anchor_id_rotation_control_scope": "not_identity_specific_case_score_proxy" if require_identity_specific else "case_score_rotation",
                "selected_case_count": len(selected),
                "selected_positive_sequence_max_frac": max_seq_frac,
                "true_positive_cases": ";".join(sorted(selected & pos)),
                "false_positive_cases": ";".join(sorted(selected & neg)),
                "missed_positive_cases": ";".join(sorted(pos - selected)),
                "gate_pass": gate,
            }
        )
    return sorted(out, key=lambda row: (b(row.get("gate_pass")), f(row.get("balanced_accuracy")), f(row.get("abs_corr_L3"))), reverse=True)


def aggregate_anchor_node_rows(anchor_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = [
        "query_head_hit_frac",
        "query_head_hit_max",
        "query_head_ge75_frac",
        "topk_route_mass_mean",
        "source_retention_mean",
        "source_residual_mean",
        "current_feature_residual_mean",
        "z_cache_current_k_sketch_residual",
        "z_cache_current_v_sketch_residual",
        "z_write_current_q_sketch_residual",
        "z_write_cache_k_sketch_residual",
        "z_cache_current_k_vec_residual",
        "z_cache_current_v_vec_residual",
        "z_write_current_q_vec_residual",
        "z_write_cache_k_vec_residual",
        "z_write_current_q_vec_projected_residual",
        "z_write_cache_k_vec_projected_residual",
        "anchor_cache_current_k_sketch_risk",
        "anchor_cache_current_v_sketch_risk",
        "anchor_z_write_current_q_sketch_risk",
        "anchor_z_write_cache_k_sketch_risk",
        "anchor_cache_current_k_vec_risk",
        "anchor_cache_current_v_vec_risk",
        "anchor_z_write_current_q_vec_risk",
        "anchor_z_write_cache_k_vec_risk",
        "anchor_z_write_current_q_vec_projected_risk",
        "anchor_z_write_cache_k_vec_projected_risk",
        "anchor_cache_current_v_cos_risk",
        "anchor_cache_current_v_l2_risk",
        "anchor_cache_current_v_route_l2_risk",
        "anchor_lifecycle_residual_risk",
        "anchor_lifecycle_stale_risk",
        "anchor_z_write_key_norm_score",
        "anchor_z_write_sketch_cache_current_risk",
    ]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in anchor_rows:
        case_id = str(row.get("case_id", ""))
        anchor_id = str(row.get("anchor_id", ""))
        if not case_id or not anchor_id:
            continue
        grouped[(case_id, anchor_id)].append(row)

    out: list[dict[str, Any]] = []
    for (case_id, anchor_id), parts in sorted(grouped.items()):
        first = parts[0]
        row: dict[str, Any] = {
            "case_id": case_id,
            "seq": first.get("seq", ""),
            "case_label": first.get("case_label", ""),
            "L3_handoff_transfer_penalty_proxy": first.get("L3_handoff_transfer_penalty_proxy", ""),
            "anchor_id": anchor_id,
            "layer_row_count": len(parts),
            "source_label_mode": Counter(str(part.get("source_label_mode", "")) for part in parts).most_common(1)[0][0],
            "source_label_mode_frac_mean": mean([f(part.get("source_label_mode_frac")) for part in parts]),
        }
        for field in fields:
            vals = [f(part.get(field)) for part in parts]
            finite_vals = finite(vals)
            row[f"{field}_max"] = max(finite_vals) if finite_vals else math.nan
            row[f"{field}_mean"] = mean(vals)
        out.append(row)
    return out


def anchor_same_count_random_margin(
    node_rows: list[dict[str, Any]],
    field: str,
    threshold: float,
    positives: set[str],
    negatives: set[str],
    *,
    higher_bad: bool,
    seeds: int = 64,
) -> float:
    if not math.isfinite(threshold):
        return math.nan
    rows_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in node_rows:
        value = f(row.get(field))
        if math.isfinite(value):
            rows_by_case[str(row.get("case_id", ""))].append(row)
    actual_selected = set()
    selected_count_by_case: dict[str, int] = {}
    for case_id, rows in rows_by_case.items():
        selected_rows = [
            row
            for row in rows
            if (f(row.get(field)) >= threshold if higher_bad else f(row.get(field)) <= threshold)
        ]
        selected_count_by_case[case_id] = len(selected_rows)
        if selected_rows:
            actual_selected.add(case_id)
    actual = signal(actual_selected, positives, negatives)

    controls = []
    for seed in range(seeds):
        control_selected = set()
        for case_id, rows in rows_by_case.items():
            count = selected_count_by_case.get(case_id, 0)
            if count <= 0:
                continue
            ranked = stable_rank(
                f"anchor_same_count_{seed}_{case_id}",
                [f"{idx}:{row.get('anchor_id', '')}" for idx, row in enumerate(rows)],
            )
            index_by_key = {f"{idx}:{row.get('anchor_id', '')}": idx for idx, row in enumerate(rows)}
            chosen = [rows[index_by_key[key]] for key in ranked[: min(count, len(rows))]]
            chosen_values = [f(row.get(field)) for row in chosen]
            chosen_values = finite(chosen_values)
            if not chosen_values:
                continue
            control_score = max(chosen_values) if higher_bad else min(chosen_values)
            if control_score >= threshold if higher_bad else control_score <= threshold:
                control_selected.add(case_id)
        controls.append(signal(control_selected, positives, negatives))
    return actual - median(controls)


def loso_diagnostics(
    values: dict[str, float],
    positives: set[str],
    negatives: set[str],
    seq_by_case: dict[str, str],
    *,
    higher_bad: bool,
) -> list[dict[str, Any]]:
    cases = sorted((positives | negatives) & set(values))
    rows: list[dict[str, Any]] = []
    for heldout_seq in sorted({seq_by_case.get(case, "") for case in cases}):
        train_cases = [case for case in cases if seq_by_case.get(case, "") != heldout_seq]
        test_cases = [case for case in cases if seq_by_case.get(case, "") == heldout_seq]
        train_pos = positives & set(train_cases)
        train_neg = negatives & set(train_cases)
        test_pos = positives & set(test_cases)
        test_neg = negatives & set(test_cases)
        best = best_threshold({case: values[case] for case in train_cases}, train_pos, train_neg, higher_bad=higher_bad)
        threshold = f(best.get("threshold"))
        selected = selected_from_threshold({case: values[case] for case in test_cases}, threshold, higher_bad=higher_bad)
        recall = len(selected & test_pos) / len(test_pos) if test_pos else math.nan
        fpr = len(selected & test_neg) / len(test_neg) if test_neg else math.nan
        rows.append(
            {
                "heldout_seq": heldout_seq,
                "train_case_count": len(train_cases),
                "test_case_count": len(test_cases),
                "threshold": threshold,
                "test_bad_recall": recall,
                "test_good_FPR": fpr,
                "test_true_positive_cases": ";".join(sorted(selected & test_pos)),
                "test_false_positive_cases": ";".join(sorted(selected & test_neg)),
                "test_missed_positive_cases": ";".join(sorted(test_pos - selected)),
            }
        )
    return rows


def evaluate_anchor_selector_patterns(node_rows: list[dict[str, Any]], specs: dict[str, tuple[str, bool]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    seq_by_case = {str(row["case_id"]): str(row.get("seq", "")) for row in node_rows}
    labels = {str(row["case_id"]): str(row.get("case_label", "")) for row in node_rows}
    positives = {case for case, label in labels.items() if label != "good"}
    negatives = {case for case, label in labels.items() if label == "good"}
    l3_by_case = {str(row["case_id"]): f(row.get("L3_handoff_transfer_penalty_proxy")) for row in node_rows}
    rows_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in node_rows:
        rows_by_case[str(row.get("case_id", ""))].append(row)

    metric_rows: list[dict[str, Any]] = []
    loso_rows: list[dict[str, Any]] = []
    for cue, (field, higher_bad) in specs.items():
        case_values: dict[str, float] = {}
        for case_id, rows in rows_by_case.items():
            vals = [f(row.get(field)) for row in rows]
            finite_vals = finite(vals)
            if finite_vals:
                case_values[case_id] = max(finite_vals) if higher_bad else min(finite_vals)
        cases = sorted((positives | negatives) & set(case_values))
        pos = positives & set(cases)
        neg = negatives & set(cases)
        best = best_threshold(case_values, pos, neg, higher_bad=higher_bad)
        threshold = f(best.get("threshold"))
        selected = selected_from_threshold(case_values, threshold, higher_bad=higher_bad)
        corr = pearson([case_values.get(case, math.nan) for case in cases], [l3_by_case.get(case, math.nan) for case in cases])
        recall = len(selected & pos) / len(pos) if pos else 0.0
        fpr = len(selected & neg) / len(neg) if neg else 0.0
        same_margin = same_count_margin(selected, cases, pos, neg) if cases else math.nan
        seq_margin = sequence_margin(selected, cases, pos, neg, seq_by_case) if cases else math.nan
        anchor_random_margin = anchor_same_count_random_margin(
            node_rows,
            field,
            threshold,
            pos,
            neg,
            higher_bad=higher_bad,
        )
        direction_correct = (corr >= 0.0 if higher_bad else corr <= 0.0) if math.isfinite(corr) else False
        selected_pos_seq = Counter(seq_by_case.get(case, "") for case in selected & pos)
        max_seq_frac = max(selected_pos_seq.values()) / max(len(selected & pos), 1) if selected_pos_seq else 0.0
        selector_gate = (
            len(cases) >= 24
            and len({seq_by_case.get(case, "") for case in cases}) >= 4
            and recall >= 0.65
            and fpr <= 0.25
            and abs(corr) >= 0.50
            and direction_correct
            and same_margin >= 0.05
            and seq_margin >= 0.05
            and anchor_random_margin >= 0.05
            and max_seq_frac <= 0.67
        )
        metric_rows.append(
            {
                "cue_name": cue,
                "field": field,
                "direction": "higher_bad" if higher_bad else "lower_bad",
                "available_case_count": len(cases),
                "sequence_coverage": len({seq_by_case.get(case, "") for case in cases}),
                "positive_case_count": len(pos),
                "good_control_case_count": len(neg),
                "threshold": threshold,
                "balanced_accuracy": best.get("balanced_accuracy"),
                "bad_recall": recall,
                "good_FPR": fpr,
                "abs_corr_L3": abs(corr) if math.isfinite(corr) else math.nan,
                "corr_L3": corr,
                "direction_correct": direction_correct,
                "same_count_case_margin": same_margin,
                "same_count_anchor_random_margin": anchor_random_margin,
                "sequence_margin": seq_margin,
                "selected_case_count": len(selected),
                "selected_positive_sequence_max_frac": max_seq_frac,
                "true_positive_cases": ";".join(sorted(selected & pos)),
                "false_positive_cases": ";".join(sorted(selected & neg)),
                "missed_positive_cases": ";".join(sorted(pos - selected)),
                "selector_gate_pass": selector_gate,
                "promotion_gate_pass": False,
                "promotion_blocker": "Per-anchor selector control is diagnostic; full Track N still requires explicit anchor-id/semantic-label rotation controls.",
            }
        )
        for loso in loso_diagnostics(case_values, pos, neg, seq_by_case, higher_bad=higher_bad):
            loso_rows.append({"cue_name": cue, **loso})
    metric_rows = sorted(
        metric_rows,
        key=lambda row: (
            b(row.get("selector_gate_pass")),
            f(row.get("balanced_accuracy")),
            f(row.get("abs_corr_L3")),
            f(row.get("same_count_anchor_random_margin")),
        ),
        reverse=True,
    )
    return metric_rows, loso_rows


def semantic_class_pattern_rows(anchor_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    meta = metadata_rows()
    fields = [
        "anchor_cache_current_v_cos_risk",
        "anchor_cache_current_v_l2_risk",
        "anchor_cache_current_v_route_l2_risk",
        "anchor_cache_current_cos_risk",
        "anchor_cache_current_l2_risk",
        "anchor_cache_current_k_sketch_risk",
        "anchor_cache_current_v_sketch_risk",
        "anchor_z_write_current_q_sketch_risk",
        "anchor_z_write_cache_k_sketch_risk",
        "anchor_cache_current_k_vec_risk",
        "anchor_cache_current_v_vec_risk",
        "anchor_z_write_current_q_vec_risk",
        "anchor_z_write_cache_k_vec_risk",
        "anchor_z_write_current_q_vec_projected_risk",
        "anchor_z_write_cache_k_vec_projected_risk",
        "anchor_z_write_key_norm_score",
        "anchor_z_write_sketch_cache_current_risk",
        "anchor_lifecycle_residual_risk",
        "anchor_lifecycle_stale_risk",
    ]
    specs = {
        "semantic_class_cache_current_v_cos_higher_bad": ("anchor_cache_current_v_cos_risk_max", True),
        "semantic_class_cache_current_v_l2_higher_bad": ("anchor_cache_current_v_l2_risk_max", True),
        "semantic_class_cache_current_v_route_l2_higher_bad": ("anchor_cache_current_v_route_l2_risk_max", True),
        "semantic_class_cache_current_k_cos_higher_bad": ("anchor_cache_current_cos_risk_max", True),
        "semantic_class_cache_current_k_l2_higher_bad": ("anchor_cache_current_l2_risk_max", True),
        "semantic_class_cache_current_k_sketch_higher_bad": ("anchor_cache_current_k_sketch_risk_max", True),
        "semantic_class_cache_current_v_sketch_higher_bad": ("anchor_cache_current_v_sketch_risk_max", True),
        "semantic_class_z_write_current_q_sketch_higher_bad": ("anchor_z_write_current_q_sketch_risk_max", True),
        "semantic_class_z_write_cache_k_sketch_higher_bad": ("anchor_z_write_cache_k_sketch_risk_max", True),
        "semantic_class_cache_current_k_vec_higher_bad": ("anchor_cache_current_k_vec_risk_max", True),
        "semantic_class_cache_current_v_vec_higher_bad": ("anchor_cache_current_v_vec_risk_max", True),
        "semantic_class_z_write_current_q_vec_higher_bad": ("anchor_z_write_current_q_vec_risk_max", True),
        "semantic_class_z_write_cache_k_vec_higher_bad": ("anchor_z_write_cache_k_vec_risk_max", True),
        "semantic_class_z_write_current_q_vec_projected_higher_bad": (
            "anchor_z_write_current_q_vec_projected_risk_max",
            True,
        ),
        "semantic_class_z_write_cache_k_vec_projected_higher_bad": (
            "anchor_z_write_cache_k_vec_projected_risk_max",
            True,
        ),
        "semantic_class_z_write_key_norm_higher_bad": ("anchor_z_write_key_norm_score_max", True),
        "semantic_class_z_write_sketch_cache_current_higher_bad": ("anchor_z_write_sketch_cache_current_risk_max", True),
        "semantic_class_lifecycle_residual_higher_bad": ("anchor_lifecycle_residual_risk_max", True),
        "semantic_class_lifecycle_stale_higher_bad": ("anchor_lifecycle_stale_risk_max", True),
    }
    by_label_case: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in anchor_rows:
        label = f(row.get("source_label_mode"))
        if not math.isfinite(label):
            continue
        by_label_case[(int(label), str(row.get("case_id", "")))].append(row)

    by_label: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for (label, case_id), parts in sorted(by_label_case.items()):
        if not parts:
            continue
        first = parts[0]
        case_meta = meta.get(case_id, {})
        out_row: dict[str, Any] = {
            "case_id": case_id,
            "seq": first.get("seq", case_meta.get("seq", "")),
            "case_label": first.get("case_label", case_meta.get("case_label", "")),
            "prev_chunk": first.get("prev_chunk", case_meta.get("prev_chunk", "")),
            "curr_chunk": first.get("curr_chunk", case_meta.get("curr_chunk", "")),
            "failure_type": first.get("failure_type", case_meta.get("failure_type", "")),
            "L3_handoff_transfer_penalty_proxy": case_meta.get(
                "L3_handoff_transfer_penalty_proxy",
                first.get("L3_handoff_transfer_penalty_proxy", ""),
            ),
            "payload_count": len(parts),
            "source_label_mode": int(label),
            "source_label_anchor_row_count": len(parts),
            "source_label_mode_frac_mean": mean([f(part.get("source_label_mode_frac")) for part in parts]),
        }
        for field in fields:
            vals = [f(part.get(field)) for part in parts]
            finite_vals = finite(vals)
            out_row[f"{field}_max"] = max(finite_vals) if finite_vals else math.nan
            out_row[f"{field}_mean"] = mean(vals)
        by_label[int(label)].append(out_row)

    out: list[dict[str, Any]] = []
    for label, rows in sorted(by_label.items()):
        pattern_rows = evaluate_patterns(
            rows,
            specs,
            min_cases=8,
            min_corr=0.55,
            require_identity_specific=False,
        )
        for pattern in pattern_rows:
            pattern["source_label_mode"] = int(label)
            pattern["diagnostic_only"] = True
            pattern["diagnostic_min_cases"] = 8
            pattern["label_case_count"] = len(rows)
            pattern["label_sequence_coverage"] = len({str(row.get("seq", "")) for row in rows})
            out.append(pattern)
    return sorted(
        out,
        key=lambda row: (
            b(row.get("gate_pass")),
            f(row.get("balanced_accuracy")),
            f(row.get("abs_corr_L3")),
            f(row.get("available_case_count")),
        ),
        reverse=True,
    )


def exploratory_case_composite_metrics(case_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def norm_feature(field: str, transform=lambda x: x) -> dict[str, float]:
        raw: dict[str, float] = {}
        for row in case_rows:
            value = f(row.get(field))
            raw[str(row.get("case_id", ""))] = transform(value) if math.isfinite(value) else math.nan
        vals = finite(raw.values())
        if not vals:
            return {case: math.nan for case in raw}
        lo, hi = min(vals), max(vals)
        denom = hi - lo
        if denom <= 0.0:
            return {case: 0.0 if math.isfinite(value) else math.nan for case, value in raw.items()}
        return {case: (value - lo) / denom if math.isfinite(value) else math.nan for case, value in raw.items()}

    current_low = norm_feature("current_support_proxy_mean", lambda x: 1.0 - x)
    fresh_low = norm_feature("fresh_supported_score_mean", lambda x: 1.0 - x)
    cache_v_sketch = norm_feature("anchor_cache_current_v_sketch_risk_max_mean")
    composites = {
        "exploratory_current_low_cache_v_sketch_w03_07": {
            case: 0.3 * current_low.get(case, math.nan) + 0.7 * cache_v_sketch.get(case, math.nan)
            for case in current_low
        },
        "exploratory_fresh_low_cache_v_sketch_w04_06": {
            case: 0.4 * fresh_low.get(case, math.nan) + 0.6 * cache_v_sketch.get(case, math.nan)
            for case in fresh_low
        },
    }
    rows: list[dict[str, Any]] = []
    for row in case_rows:
        out_row = dict(row)
        case_id = str(row.get("case_id", ""))
        for name, values in composites.items():
            out_row[name] = values.get(case_id, math.nan)
        rows.append(out_row)
    metrics = evaluate_patterns(
        rows,
        {name: (name, True) for name in composites},
        min_cases=24,
        min_corr=0.50,
        require_identity_specific=False,
    )
    for row in metrics:
        row["non_identity_gate_pass"] = row.get("gate_pass")
        row["non_identity_case_value_rotation_margin"] = row.get("case_value_rotation_margin")
        row["anchor_id_rotation_margin"] = 0.0
        row["gate_pass"] = False
        row["posthoc_exploratory"] = True
        row["uses_same_28_case_search_set"] = True
        row["promoted_gate_pass"] = False
        row["promotion_blocker"] = (
            "Composite passes only non-identity case-level controls on the same search set; "
            "Track N/C3 still require explicit anchor-id/semantic-label rotation controls before action."
        )
    return metrics


def track_n() -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    out = ROOT / "trackN_semantic_anchor_identity_memory_graph"
    edge_rows, anchor_rows, errors, diagnostics = collect_trace_rows()
    case_rows = aggregate_case_rows(edge_rows)
    semantic_rows = semantic_class_pattern_rows(anchor_rows)
    exploratory_rows = exploratory_case_composite_metrics(case_rows)
    anchor_node_rows = aggregate_anchor_node_rows(anchor_rows)
    anchor_selector_specs = {
        "per_anchor_cache_current_v_sketch_higher_bad": ("anchor_cache_current_v_sketch_risk_max", True),
        "per_anchor_z_write_current_q_sketch_higher_bad": ("anchor_z_write_current_q_sketch_risk_max", True),
        "per_anchor_z_write_cache_k_sketch_higher_bad": ("anchor_z_write_cache_k_sketch_risk_max", True),
        "per_anchor_cache_current_k_sketch_higher_bad": ("anchor_cache_current_k_sketch_risk_max", True),
        "per_anchor_cache_current_k_vec_higher_bad": ("anchor_cache_current_k_vec_risk_max", True),
        "per_anchor_cache_current_v_vec_higher_bad": ("anchor_cache_current_v_vec_risk_max", True),
        "per_anchor_z_write_current_q_vec_higher_bad": ("anchor_z_write_current_q_vec_risk_max", True),
        "per_anchor_z_write_cache_k_vec_higher_bad": ("anchor_z_write_cache_k_vec_risk_max", True),
        "per_anchor_z_write_current_q_vec_projected_higher_bad": (
            "anchor_z_write_current_q_vec_projected_risk_max",
            True,
        ),
        "per_anchor_z_write_cache_k_vec_projected_higher_bad": (
            "anchor_z_write_cache_k_vec_projected_risk_max",
            True,
        ),
        "per_anchor_cache_current_v_cos_higher_bad": ("anchor_cache_current_v_cos_risk_max", True),
        "per_anchor_cache_current_v_l2_higher_bad": ("anchor_cache_current_v_l2_risk_max", True),
        "per_anchor_lifecycle_residual_higher_bad": ("anchor_lifecycle_residual_risk_max", True),
        "per_anchor_lifecycle_stale_higher_bad": ("anchor_lifecycle_stale_risk_max", True),
        "per_anchor_z_write_sketch_cache_current_higher_bad": ("anchor_z_write_sketch_cache_current_risk_max", True),
    }
    anchor_selector_rows, anchor_selector_loso_rows = evaluate_anchor_selector_patterns(anchor_node_rows, anchor_selector_specs)
    specs = {
        "stale_unsupported_hit_higher_bad": ("stale_unsupported_hit_score_mean", True),
        "identity_read_risk_higher_bad": ("identity_read_risk_proxy_mean", True),
        "query_head_ge75_higher_bad": ("query_head_ge75_frac_mean", True),
        "topk_query_hit_higher_bad": ("anchor_id_topk_query_hit_frac_mean", True),
        "current_support_lower_bad": ("current_support_proxy_mean", False),
        "fresh_supported_lower_bad": ("fresh_supported_score_mean", False),
        "feature_transport_residual_higher_bad": ("feature_transport_residual_mean", True),
        "anchor_fragmentation_higher_bad": ("anchor_fragmentation_score_mean", True),
        "anchor_lifecycle_residual_risk_higher_bad": ("anchor_lifecycle_residual_risk_max_mean", True),
        "anchor_lifecycle_stale_risk_higher_bad": ("anchor_lifecycle_stale_risk_max_mean", True),
        "anchor_lifecycle_support_lower_bad": ("anchor_lifecycle_support_score_max_mean", False),
        "anchor_cache_current_l2_risk_higher_bad": ("anchor_cache_current_l2_risk_max_mean", True),
        "anchor_cache_current_route_l2_risk_higher_bad": ("anchor_cache_current_route_l2_risk_max_mean", True),
        "anchor_cache_current_cos_risk_higher_bad": ("anchor_cache_current_cos_risk_max_mean", True),
        "anchor_cache_current_v_l2_risk_higher_bad": ("anchor_cache_current_v_l2_risk_max_mean", True),
        "anchor_cache_current_v_route_l2_risk_higher_bad": ("anchor_cache_current_v_route_l2_risk_max_mean", True),
        "anchor_cache_current_v_cos_risk_higher_bad": ("anchor_cache_current_v_cos_risk_max_mean", True),
        "anchor_head75_cache_current_v_cos_higher_bad": ("anchor_head75_cache_current_v_cos_risk_max_mean", True),
        "anchor_headmax_cache_current_v_cos_higher_bad": ("anchor_headmax_cache_current_v_cos_risk_max_mean", True),
        "anchor_cache_current_k_sketch_higher_bad": ("anchor_cache_current_k_sketch_risk_max_mean", True),
        "anchor_cache_current_v_sketch_higher_bad": ("anchor_cache_current_v_sketch_risk_max_mean", True),
        "anchor_z_write_current_q_sketch_higher_bad": ("anchor_z_write_current_q_sketch_risk_max_mean", True),
        "anchor_z_write_cache_k_sketch_higher_bad": ("anchor_z_write_cache_k_sketch_risk_max_mean", True),
        "anchor_cache_current_k_vec_higher_bad": ("anchor_cache_current_k_vec_risk_max_mean", True),
        "anchor_cache_current_v_vec_higher_bad": ("anchor_cache_current_v_vec_risk_max_mean", True),
        "anchor_z_write_current_q_vec_higher_bad": ("anchor_z_write_current_q_vec_risk_max_mean", True),
        "anchor_z_write_cache_k_vec_higher_bad": ("anchor_z_write_cache_k_vec_risk_max_mean", True),
        "anchor_z_write_current_q_vec_projected_higher_bad": (
            "anchor_z_write_current_q_vec_projected_risk_max_mean",
            True,
        ),
        "anchor_z_write_cache_k_vec_projected_higher_bad": (
            "anchor_z_write_cache_k_vec_projected_risk_max_mean",
            True,
        ),
        "anchor_z_write_key_norm_higher_bad": ("anchor_z_write_key_norm_score_max_mean", True),
        "anchor_z_write_sketch_cache_current_higher_bad": ("anchor_z_write_sketch_cache_current_risk_max_mean", True),
    }
    pattern_rows = evaluate_patterns(case_rows, specs, min_cases=24, min_corr=0.50, require_identity_specific=True)
    best = pattern_rows[0] if pattern_rows else {}
    best_exploratory = exploratory_rows[0] if exploratory_rows else {}
    best_anchor_selector = anchor_selector_rows[0] if anchor_selector_rows else {}
    case_count = len(case_rows)
    sequence_coverage = len({str(row.get("seq", "")) for row in case_rows})
    label_counts = Counter(str(row.get("case_label", "")) for row in case_rows)
    summary = {
        "schema": "acl2_v99_trackN_identity_memory_graph_v1",
        "status": "complete" if edge_rows and not errors else ("complete_with_trace_errors" if edge_rows else "blocked_no_trace_payloads"),
        "gate_pass": any(b(row.get("gate_pass")) for row in pattern_rows),
        "case_count": case_count,
        "sequence_coverage": sequence_coverage,
        "label_counts": dict(label_counts),
        "trace_payload_file_count": diagnostics.get("trace_payload_file_count"),
        "trace_read_error_count": diagnostics.get("trace_read_error_count"),
        "anchor_lifecycle_row_count": diagnostics.get("anchor_lifecycle_row_count"),
        "expanded_trace_available": diagnostics.get("expanded_trace_available"),
        "best_pattern": best.get("cue_name", ""),
        "best_pattern_bad_recall": best.get("bad_recall", math.nan),
        "best_pattern_good_FPR": best.get("good_FPR", math.nan),
        "best_pattern_abs_corr_L3": best.get("abs_corr_L3", math.nan),
        "best_pattern_anchor_id_rotation_margin": best.get("anchor_id_rotation_margin", math.nan),
        "per_anchor_node_count": len(anchor_node_rows),
        "per_anchor_selector_diagnostic_gate_pass": any(b(row.get("selector_gate_pass")) for row in anchor_selector_rows),
        "per_anchor_selector_best_cue": best_anchor_selector.get("cue_name", ""),
        "per_anchor_selector_best_bad_recall": best_anchor_selector.get("bad_recall", math.nan),
        "per_anchor_selector_best_good_FPR": best_anchor_selector.get("good_FPR", math.nan),
        "per_anchor_selector_best_abs_corr_L3": best_anchor_selector.get("abs_corr_L3", math.nan),
        "per_anchor_selector_best_same_count_anchor_random_margin": best_anchor_selector.get(
            "same_count_anchor_random_margin", math.nan
        ),
        "exploratory_composite_non_identity_gate_pass": any(
            b(row.get("non_identity_gate_pass")) for row in exploratory_rows
        ),
        "exploratory_composite_best_cue": best_exploratory.get("cue_name", ""),
        "exploratory_composite_best_bad_recall": best_exploratory.get("bad_recall", math.nan),
        "exploratory_composite_best_good_FPR": best_exploratory.get("good_FPR", math.nan),
        "exploratory_composite_best_abs_corr_L3": best_exploratory.get("abs_corr_L3", math.nan),
        "exploratory_composite_promoted_gate_pass": any(b(row.get("promoted_gate_pass")) for row in exploratory_rows),
        "runtime_action_allowed": False,
        "blocker": ""
        if any(b(row.get("gate_pass")) for row in pattern_rows)
        else "No graph pattern passed all v99 controls; current case scores are not identity-label-specific under anchor-id rotation.",
    }
    write_json(out / "summary.json", summary)
    write_rows(out / "graph_edge_rows.csv", edge_rows)
    write_rows(out / "graph_anchor_lifecycle_rows.csv", anchor_rows)
    write_rows(out / "graph_anchor_node_rows.csv", anchor_node_rows)
    write_rows(out / "graph_case_rows.csv", case_rows)
    write_rows(out / "pattern_metrics.csv", pattern_rows)
    write_rows(out / "exploratory_composite_metrics.csv", exploratory_rows)
    write_rows(out / "semantic_class_pattern_metrics.csv", semantic_rows)
    write_rows(out / "per_anchor_selector_metrics.csv", anchor_selector_rows)
    write_rows(out / "per_anchor_selector_loso_rows.csv", anchor_selector_loso_rows)
    write_rows(out / "read_errors.csv", errors)
    control_rows = [
        {
            "control": "anchor_id_rotation",
            "implemented_scope": "availability_check_only_for_case_scores",
            "pass_count": sum(1 for row in pattern_rows if f(row.get("anchor_id_rotation_margin")) >= 0.05),
            "note": "Trace has anchor ids, but evaluated scores are invariant to id-label permutation; not counted as identity-specific evidence.",
        },
        {
            "control": "same_count_anchor_random",
            "implemented_scope": "same_count_case_random",
            "pass_count": sum(1 for row in pattern_rows if f(row.get("same_count_margin")) >= 0.05),
            "note": "Case-level control; per-anchor random requires per-anchor semantic/support rows.",
        },
        {
            "control": "sequence_balanced",
            "implemented_scope": "sequence_count_preserving_case_random",
            "pass_count": sum(1 for row in pattern_rows if f(row.get("sequence_margin")) >= 0.05),
            "note": "Implemented from case sequence labels.",
        },
    ]
    write_rows(out / "control_metrics.csv", control_rows)
    write_text(
        out / "graph_pattern_failure_attribution.md",
        "# Track N Failure Attribution\n\n"
        f"- case_count={case_count}, sequence_coverage={sequence_coverage}, labels={dict(label_counts)}.\n"
        f"- best_pattern={summary['best_pattern']}, bad_recall={summary['best_pattern_bad_recall']}, "
        f"good_FPR={summary['best_pattern_good_FPR']}, abs_corr_L3={summary['best_pattern_abs_corr_L3']}.\n"
        f"- per_anchor_selector_best={summary['per_anchor_selector_best_cue']}, "
        f"bad_recall={summary['per_anchor_selector_best_bad_recall']}, "
        f"good_FPR={summary['per_anchor_selector_best_good_FPR']}, "
        f"abs_corr_L3={summary['per_anchor_selector_best_abs_corr_L3']}, "
        f"same_count_anchor_random_margin={summary['per_anchor_selector_best_same_count_anchor_random_margin']}.\n"
        f"- exploratory_composite_best={summary['exploratory_composite_best_cue']}, "
        f"non_identity_gate_pass={summary['exploratory_composite_non_identity_gate_pass']}, "
        f"bad_recall={summary['exploratory_composite_best_bad_recall']}, "
        f"good_FPR={summary['exploratory_composite_best_good_FPR']}, "
        f"abs_corr_L3={summary['exploratory_composite_best_abs_corr_L3']}; "
        "promoted_gate_pass=false because this is same-set posthoc case-level evidence without anchor-id/semantic-label rotation controls.\n"
        f"- blocker={summary['blocker'] or 'none'}\n"
        "- The expanded traces now provide per-anchor lifecycle/source-label/latent-sketch rows, but the promoted Track N "
        "case scores remain mass-like proxies unless explicit anchor-id, semantic-label, and same-count anchor controls pass.\n",
    )
    if summary["blocker"]:
        write_text(
            out / "identity_missing_report.md",
            "# Identity Missing Report\n\n"
            "- Required for stronger Track N: explicit control rows proving selected anchor identities beat anchor-id rotation, semantic-label rotation, and same-count anchor random.\n"
            "- Available now: anchor-id hit mask/id tensors, per-anchor lifecycle rows, source label mode, query/head usage, and latent sketch residuals.\n"
            "- Still incomplete: an independent READ/current-support provider and promoted semantic-label rotation control; no values were fabricated.\n",
        )
    return summary, case_rows, pattern_rows


def track_o(case_rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = ROOT / "trackO_anchor_freshness_current_support_ruler"
    rows: list[dict[str, Any]] = []
    for row in case_rows:
        support = f(row.get("current_support_proxy_mean"))
        age = f(row.get("anchor_age_norm_mean"))
        query = f(row.get("anchor_id_topk_query_hit_frac_mean"))
        risk = f(row.get("identity_read_risk_proxy_mean"))
        freshness = support - 0.25 * age - 0.25 * query if all(math.isfinite(v) for v in [support, age, query]) else math.nan
        if math.isfinite(freshness) and freshness < 0.25 and math.isfinite(risk) and risk >= 0.25:
            bucket = "STALE_UNSUPPORTED_RISK"
        elif math.isfinite(freshness) and freshness >= 0.45 and math.isfinite(query) and query >= 0.35:
            bucket = "FRESH_SUPPORTED_TRANSMIT"
        elif math.isfinite(support) and support < 0.30:
            bucket = "REFRESH_NEEDED"
        else:
            bucket = "AMBIGUOUS_DELAY"
        out_row = dict(row)
        out_row["freshness_score"] = freshness
        out_row["freshness_bucket"] = bucket
        out_row["high_risk_bucket"] = bucket in {"STALE_UNSUPPORTED_RISK", "REFRESH_NEEDED"}
        rows.append(out_row)
    specs = {
        "freshness_score_lower_bad": ("freshness_score", False),
        "high_risk_bucket_higher_bad": ("high_risk_bucket_numeric", True),
        "identity_read_risk_higher_bad": ("identity_read_risk_proxy_mean", True),
    }
    eval_rows = []
    for row in rows:
        row["high_risk_bucket_numeric"] = 1.0 if b(row.get("high_risk_bucket")) else 0.0
    eval_rows = evaluate_patterns(rows, specs, min_cases=24, min_corr=0.50, require_identity_specific=True)
    best = eval_rows[0] if eval_rows else {}
    stage0_summary = read_json(ROOT / "trackI_v99_identity_evidence_ledger/summary.json")
    stage7e_fp_count = len(stage0_summary.get("best_cue_false_positive_cases", []) or [])
    fp_cases = split_cases(best.get("false_positive_cases"))
    missed_cases = split_cases(best.get("missed_positive_cases"))
    gate = (
        any(b(row.get("gate_pass")) for row in eval_rows)
        and len(fp_cases) < stage7e_fp_count
        and len(missed_cases) == 0
    )
    bucket_rows = []
    for bucket, parts in sorted(defaultdict(list, {k: [r for r in rows if r.get("freshness_bucket") == k] for k in {r.get("freshness_bucket") for r in rows}}).items()):
        bucket_rows.append(
            {
                "freshness_bucket": bucket,
                "case_count": len(parts),
                "non_good_count": sum(1 for part in parts if part.get("case_label") != "good"),
                "good_count": sum(1 for part in parts if part.get("case_label") == "good"),
                "mean_L3": mean([f(part.get("L3_handoff_transfer_penalty_proxy")) for part in parts]),
            }
        )
    summary = {
        "schema": "acl2_v99_trackO_freshness_current_support_v1",
        "status": "complete" if rows else "blocked_no_trackN_rows",
        "gate_pass": gate,
        "case_count": len(rows),
        "sequence_coverage": len({str(row.get("seq", "")) for row in rows}),
        "best_cue": best.get("cue_name", ""),
        "best_cue_bad_recall": best.get("bad_recall", math.nan),
        "best_cue_good_FPR": best.get("good_FPR", math.nan),
        "best_cue_abs_corr_L3": best.get("abs_corr_L3", math.nan),
        "stage7e_false_positive_count": stage7e_fp_count,
        "trackO_false_positive_count": len(fp_cases),
        "trackO_missed_positive_count": len(missed_cases),
        "blocker": ""
        if gate
        else "Freshness/current-support proxy did not pass all controls and/or did not reduce Stage7e false positives with missed positives explained.",
        "runtime_action_allowed": False,
    }
    write_json(out / "summary.json", summary)
    write_rows(out / "freshness_case_rows.csv", rows)
    write_rows(out / "freshness_bucket_rows.csv", bucket_rows)
    write_rows(out / "freshness_pattern_metrics.csv", eval_rows)
    write_rows(out / "false_positive_breakdown.csv", [row for row in rows if row.get("case_id") in fp_cases])
    write_rows(out / "missed_positive_breakdown.csv", [row for row in rows if row.get("case_id") in missed_cases])
    write_text(
        out / "failure_attribution.md",
        "# Track O Failure Attribution\n\n"
        f"- gate_pass={gate}; best_cue={summary['best_cue']}.\n"
        f"- Stage7e false_positive_count={stage7e_fp_count}; TrackO false_positive_count={len(fp_cases)}; missed_positive_count={len(missed_cases)}.\n"
        "- Current support is a documented proxy from cache/top-k/transport summaries, not a per-anchor current support measurement.\n",
    )
    return summary


def scan_feature_availability() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    trace_root, paths, expanded_available = trace_payload_paths()
    rows: list[dict[str, Any]] = []
    key_counter: Counter[str] = Counter()
    z_keys = {"z_write", "z_cache", "z_current", "anchor_z_write", "anchor_z_cache", "anchor_z_current"}
    for path in paths[: max(len(paths), 1)]:
        try:
            payload = torch_load(path)
        except Exception as exc:  # noqa: BLE001
            rows.append({"path": str(path), "error": f"{type(exc).__name__}:{exc}"})
            continue
        if not isinstance(payload, dict):
            continue
        keys = {str(key) for key in payload.keys()}
        lifecycle_keys: set[str] = set()
        raw_lifecycle = payload.get("ttt_prev_stable_anchor_lifecycle_rows")
        if isinstance(raw_lifecycle, list):
            for raw_row in raw_lifecycle:
                if isinstance(raw_row, dict):
                    lifecycle_keys.update(str(key) for key in raw_row.keys())
        key_counter.update(keys)
        rows.append(
            {
                "path": str(path),
                "has_z_write": bool(keys & {"z_write", "anchor_z_write"}),
                "has_z_cache": bool(keys & {"z_cache", "anchor_z_cache"}),
                "has_z_current": bool(keys & {"z_current", "anchor_z_current"}),
                "has_z_write_vec_lifecycle": "z_write_key_vec_mean" in lifecycle_keys,
                "has_z_write_vec_projected_lifecycle": "z_write_key_vec_projected_mean" in lifecycle_keys,
                "has_z_cache_vec_lifecycle": bool({"z_cache_k_vec_mean", "z_cache_v_vec_mean"} & lifecycle_keys),
                "has_z_current_vec_lifecycle": bool({"z_current_q_vec_mean", "z_current_v_vec_mean"} & lifecycle_keys),
                "has_z_write_key_sketch_lifecycle": any(key.startswith("z_write_key_sketch_") for key in lifecycle_keys),
                "has_z_write_key_norm_lifecycle": "z_write_key_norm_mean" in lifecycle_keys,
                "has_z_cache_current_lifecycle_pair": any(key.startswith("z_cache_current_") for key in lifecycle_keys),
                "has_z_vector_sketch_lifecycle": any(
                    key in lifecycle_keys
                    for key in {
                        "z_current_q_sketch_mean",
                        "z_cache_k_sketch_mean",
                        "z_current_v_sketch_mean",
                        "z_cache_v_sketch_mean",
                    }
                ),
                "has_z_full_vector_lifecycle": all(
                    key in lifecycle_keys
                    for key in {
                        "z_write_key_vec_mean",
                        "z_cache_k_vec_mean",
                        "z_current_q_vec_mean",
                    }
                ),
                "has_feature_transport_residual_summary": "feature_transport_residual_mean" in keys,
                "z_like_keys": ";".join(sorted(key for key in keys if "z_" in key or key.startswith("z"))),
                "lifecycle_z_like_keys": ";".join(sorted(key for key in lifecycle_keys if "z_" in key or key.startswith("z"))),
            }
        )
    summary = {
        "trace_root": str(trace_root),
        "expanded_trace_available": expanded_available,
        "checked_payload_count": len(rows),
        "z_write_available": any(b(row.get("has_z_write")) for row in rows),
        "z_cache_available": any(b(row.get("has_z_cache")) for row in rows),
        "z_current_available": any(b(row.get("has_z_current")) for row in rows),
        "z_write_vec_lifecycle_available": any(b(row.get("has_z_write_vec_lifecycle")) for row in rows),
        "z_write_vec_projected_lifecycle_available": any(
            b(row.get("has_z_write_vec_projected_lifecycle")) for row in rows
        ),
        "z_cache_vec_lifecycle_available": any(b(row.get("has_z_cache_vec_lifecycle")) for row in rows),
        "z_current_vec_lifecycle_available": any(b(row.get("has_z_current_vec_lifecycle")) for row in rows),
        "z_write_key_sketch_lifecycle_available": any(b(row.get("has_z_write_key_sketch_lifecycle")) for row in rows),
        "z_write_key_norm_lifecycle_available": any(b(row.get("has_z_write_key_norm_lifecycle")) for row in rows),
        "z_cache_current_lifecycle_pair_available": any(b(row.get("has_z_cache_current_lifecycle_pair")) for row in rows),
        "z_vector_sketch_lifecycle_available": any(b(row.get("has_z_vector_sketch_lifecycle")) for row in rows),
        "z_full_vector_lifecycle_available": any(b(row.get("has_z_full_vector_lifecycle")) for row in rows),
        "z_required_key_hits": {key: key_counter.get(key, 0) for key in sorted(z_keys)},
    }
    return rows, summary


def track_c3(case_rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = ROOT / "trackC3_identity_conditioned_latent_gauge_ruler"
    availability_rows, availability = scan_feature_availability()
    has_all_z = (
        availability["z_write_available"]
        and availability["z_cache_available"]
        and availability["z_current_available"]
    ) or (
        availability.get("z_write_vec_lifecycle_available")
        and availability.get("z_cache_vec_lifecycle_available")
        and availability.get("z_current_vec_lifecycle_available")
    )
    cache_current_case_available = any(
        math.isfinite(f(row.get(field)))
        for row in case_rows
        for field in [
            "anchor_cache_current_l2_risk_max_mean",
            "anchor_cache_current_route_l2_risk_max_mean",
            "anchor_cache_current_cos_risk_max_mean",
            "anchor_cache_current_v_l2_risk_max_mean",
            "anchor_cache_current_v_route_l2_risk_max_mean",
            "anchor_cache_current_v_cos_risk_max_mean",
            "anchor_head75_cache_current_v_cos_risk_max_mean",
            "anchor_headmax_cache_current_v_cos_risk_max_mean",
            "anchor_cache_current_k_sketch_risk_max_mean",
            "anchor_cache_current_v_sketch_risk_max_mean",
            "anchor_cache_current_k_vec_risk_max_mean",
            "anchor_cache_current_v_vec_risk_max_mean",
        ]
    )
    z_write_key_case_available = any(
        math.isfinite(f(row.get(field)))
        for row in case_rows
        for field in [
            "anchor_z_write_key_norm_score_max_mean",
            "anchor_z_write_sketch_cache_current_risk_max_mean",
            "anchor_z_write_current_q_sketch_risk_max_mean",
            "anchor_z_write_cache_k_sketch_risk_max_mean",
            "anchor_z_write_current_q_vec_risk_max_mean",
            "anchor_z_write_cache_k_vec_risk_max_mean",
            "anchor_z_write_current_q_vec_projected_risk_max_mean",
            "anchor_z_write_cache_k_vec_projected_risk_max_mean",
        ]
    )
    proxy_specs = {
        "feature_transport_residual_higher_bad": ("feature_transport_residual_mean", True),
        "anchor_cache_current_l2_risk_higher_bad": ("anchor_cache_current_l2_risk_max_mean", True),
        "anchor_cache_current_route_l2_risk_higher_bad": ("anchor_cache_current_route_l2_risk_max_mean", True),
        "anchor_cache_current_cos_risk_higher_bad": ("anchor_cache_current_cos_risk_max_mean", True),
        "anchor_cache_current_v_l2_risk_higher_bad": ("anchor_cache_current_v_l2_risk_max_mean", True),
        "anchor_cache_current_v_route_l2_risk_higher_bad": ("anchor_cache_current_v_route_l2_risk_max_mean", True),
        "anchor_cache_current_v_cos_risk_higher_bad": ("anchor_cache_current_v_cos_risk_max_mean", True),
        "anchor_head75_cache_current_v_cos_higher_bad": ("anchor_head75_cache_current_v_cos_risk_max_mean", True),
        "anchor_headmax_cache_current_v_cos_higher_bad": ("anchor_headmax_cache_current_v_cos_risk_max_mean", True),
        "anchor_cache_current_k_sketch_higher_bad": ("anchor_cache_current_k_sketch_risk_max_mean", True),
        "anchor_cache_current_v_sketch_higher_bad": ("anchor_cache_current_v_sketch_risk_max_mean", True),
        "anchor_z_write_current_q_sketch_higher_bad": ("anchor_z_write_current_q_sketch_risk_max_mean", True),
        "anchor_z_write_cache_k_sketch_higher_bad": ("anchor_z_write_cache_k_sketch_risk_max_mean", True),
        "anchor_cache_current_k_vec_higher_bad": ("anchor_cache_current_k_vec_risk_max_mean", True),
        "anchor_cache_current_v_vec_higher_bad": ("anchor_cache_current_v_vec_risk_max_mean", True),
        "anchor_z_write_current_q_vec_higher_bad": ("anchor_z_write_current_q_vec_risk_max_mean", True),
        "anchor_z_write_cache_k_vec_higher_bad": ("anchor_z_write_cache_k_vec_risk_max_mean", True),
        "anchor_z_write_current_q_vec_projected_higher_bad": (
            "anchor_z_write_current_q_vec_projected_risk_max_mean",
            True,
        ),
        "anchor_z_write_cache_k_vec_projected_higher_bad": (
            "anchor_z_write_cache_k_vec_projected_risk_max_mean",
            True,
        ),
        "anchor_z_write_key_norm_higher_bad": ("anchor_z_write_key_norm_score_max_mean", True),
        "anchor_z_write_sketch_cache_current_higher_bad": ("anchor_z_write_sketch_cache_current_risk_max_mean", True),
    }
    proxy_rows = evaluate_patterns(case_rows, proxy_specs, min_cases=24, min_corr=0.55, require_identity_specific=True)
    feature_transport_proxy_rows = [
        row for row in proxy_rows if str(row.get("cue_name", "")) == "feature_transport_residual_higher_bad"
    ]
    best_feature_transport_proxy = feature_transport_proxy_rows[0] if feature_transport_proxy_rows else {}
    cache_current_proxy_rows = [
        row for row in proxy_rows if str(row.get("cue_name", "")).startswith("anchor_cache_current_")
    ]
    best_cache_current_proxy = cache_current_proxy_rows[0] if cache_current_proxy_rows else {}
    z_write_key_proxy_rows = [
        row for row in proxy_rows if str(row.get("cue_name", "")).startswith("anchor_z_write_")
    ]
    best_z_write_key_proxy = z_write_key_proxy_rows[0] if z_write_key_proxy_rows else {}
    summary = {
        "schema": "acl2_v99_trackC3_identity_latent_gauge_ruler_v1",
        "status": (
            "complete"
            if has_all_z
            else (
                "blocked_full_z_tensor_missing_write_key_sketch_and_cache_current_pair_available"
                if (
                    z_write_key_case_available
                    or availability.get("z_write_key_sketch_lifecycle_available")
                    or availability.get("z_write_key_norm_lifecycle_available")
                )
                and (cache_current_case_available or availability.get("z_cache_current_lifecycle_pair_available"))
                else (
                    "blocked_z_write_missing_cache_current_pair_available"
                    if cache_current_case_available or availability.get("z_cache_current_lifecycle_pair_available")
                    else "blocked_z_features_missing"
                )
            )
        ),
        "gate_pass": any(b(row.get("gate_pass")) for row in proxy_rows) if has_all_z else False,
        "case_count": len(case_rows),
        "sequence_coverage": len({str(row.get("seq", "")) for row in case_rows}),
        "z_write_available": availability["z_write_available"],
        "z_cache_available": availability["z_cache_available"],
        "z_current_available": availability["z_current_available"],
        "z_write_key_sketch_lifecycle_available": availability.get("z_write_key_sketch_lifecycle_available"),
        "z_write_key_norm_lifecycle_available": availability.get("z_write_key_norm_lifecycle_available"),
        "z_write_vec_lifecycle_available": availability.get("z_write_vec_lifecycle_available"),
        "z_write_vec_projected_lifecycle_available": availability.get("z_write_vec_projected_lifecycle_available"),
        "z_cache_vec_lifecycle_available": availability.get("z_cache_vec_lifecycle_available"),
        "z_current_vec_lifecycle_available": availability.get("z_current_vec_lifecycle_available"),
        "z_write_key_case_metric_available": z_write_key_case_available,
        "z_cache_current_lifecycle_pair_available": availability.get("z_cache_current_lifecycle_pair_available"),
        "z_vector_sketch_lifecycle_available": availability.get("z_vector_sketch_lifecycle_available"),
        "z_full_vector_lifecycle_available": availability.get("z_full_vector_lifecycle_available"),
        "z_cache_current_case_metric_available": cache_current_case_available,
        "proxy_feature_transport_best_abs_corr_L3": best_feature_transport_proxy.get("abs_corr_L3", math.nan),
        "proxy_cache_current_best_cue": best_cache_current_proxy.get("cue_name", ""),
        "proxy_cache_current_best_bad_recall": best_cache_current_proxy.get("bad_recall", math.nan),
        "proxy_cache_current_best_good_FPR": best_cache_current_proxy.get("good_FPR", math.nan),
        "proxy_cache_current_best_abs_corr_L3": best_cache_current_proxy.get("abs_corr_L3", math.nan),
        "proxy_z_write_key_best_cue": best_z_write_key_proxy.get("cue_name", ""),
        "proxy_z_write_key_best_bad_recall": best_z_write_key_proxy.get("bad_recall", math.nan),
        "proxy_z_write_key_best_good_FPR": best_z_write_key_proxy.get("good_FPR", math.nan),
        "proxy_z_write_key_best_abs_corr_L3": best_z_write_key_proxy.get("abs_corr_L3", math.nan),
        "blocker": (
            ""
            if any(b(row.get("gate_pass")) for row in proxy_rows)
            else "Full lifecycle z vectors are available, but no latent-gauge proxy passed recall/FPR/corr/control gates."
        )
        if has_all_z
        else (
            "TTT z_write key sketch and SWA cache-current q/k latent pair residual are available, but full comparable z_write/z_cache/z_current tensors are absent; sketches/scalars are not promoted to the full latent residual gate."
            if (
                z_write_key_case_available
                or availability.get("z_write_key_sketch_lifecycle_available")
                or availability.get("z_write_key_norm_lifecycle_available")
            )
            and (cache_current_case_available or availability.get("z_cache_current_lifecycle_pair_available"))
            else (
            "Cache-current q/k latent pair residual is available, but required z_write plus full z_cache/z_current tensors are absent; cache-current scalar diagnostics are not promoted to the full latent residual gate."
            if cache_current_case_available or availability.get("z_cache_current_lifecycle_pair_available")
            else "Required z_write/z_cache/z_current tensors are absent. Feature_transport_residual summary is recorded only as a proxy and is not promoted to latent residual."
            )
        ),
        "runtime_action_allowed": False,
    }
    write_json(out / "summary.json", summary)
    write_rows(out / "feature_availability_rows.csv", availability_rows)
    write_rows(out / "proxy_pattern_metrics.csv", proxy_rows)
    write_text(
        out / "missing_feature_report.md",
        "# Track C3 Feature Status and Failure Report\n\n"
        f"- z_write_available={summary['z_write_available']}\n"
        f"- z_cache_available={summary['z_cache_available']}\n"
        f"- z_current_available={summary['z_current_available']}\n"
        f"- z_write_key_sketch_lifecycle_available={summary['z_write_key_sketch_lifecycle_available']}\n"
        f"- z_write_key_norm_lifecycle_available={summary['z_write_key_norm_lifecycle_available']}\n"
        f"- z_write_vec_lifecycle_available={summary['z_write_vec_lifecycle_available']}\n"
        f"- z_write_vec_projected_lifecycle_available={summary['z_write_vec_projected_lifecycle_available']}\n"
        f"- z_cache_vec_lifecycle_available={summary['z_cache_vec_lifecycle_available']}\n"
        f"- z_current_vec_lifecycle_available={summary['z_current_vec_lifecycle_available']}\n"
        f"- z_full_vector_lifecycle_available={summary['z_full_vector_lifecycle_available']}\n"
        f"- z_write_key_case_metric_available={summary['z_write_key_case_metric_available']}\n"
        f"- z_cache_current_lifecycle_pair_available={summary['z_cache_current_lifecycle_pair_available']}\n"
        f"- z_vector_sketch_lifecycle_available={summary['z_vector_sketch_lifecycle_available']}\n"
        f"- z_cache_current_case_metric_available={summary['z_cache_current_case_metric_available']}\n"
        f"- proxy_cache_current_best={summary['proxy_cache_current_best_cue']}, bad_recall={summary['proxy_cache_current_best_bad_recall']}, "
        f"good_FPR={summary['proxy_cache_current_best_good_FPR']}, abs_corr_L3={summary['proxy_cache_current_best_abs_corr_L3']}.\n"
        f"- proxy_z_write_key_best={summary['proxy_z_write_key_best_cue']}, bad_recall={summary['proxy_z_write_key_best_bad_recall']}, "
        f"good_FPR={summary['proxy_z_write_key_best_good_FPR']}, abs_corr_L3={summary['proxy_z_write_key_best_abs_corr_L3']}.\n"
        f"- blocker={summary['blocker'] or 'none'}\n"
        "- Plan fail-forward applied: full z_write key vectors are dumped from TTT, projected to the SWA head dimension for comparable residuals, and patch coordinates are not used as latent residuals.\n",
    )
    return summary


def track_m2(case_rows: list[dict[str, Any]], track_n_summary: dict[str, Any], track_o_summary: dict[str, Any], track_c3_summary: dict[str, Any]) -> dict[str, Any]:
    out = ROOT / "trackM2_identity_level_carrier_to_action_simulator"
    prereq_pass = b(track_n_summary.get("gate_pass")) and (b(track_o_summary.get("gate_pass")) or b(track_c3_summary.get("gate_pass")))
    rows: list[dict[str, Any]] = []
    specs = [
        ("query_head_edge_damp", "query_head_ge75_frac_mean"),
        ("stale_anchor_expire", "stale_unsupported_hit_score_mean"),
        ("current_support_conditioned_preserve", "current_support_proxy_mean"),
    ]
    positives = [row for row in case_rows if row.get("case_label") != "good"]
    goods = [row for row in case_rows if row.get("case_label") == "good"]
    for family, field in specs:
        bad_risk_before = mean([f(row.get(field)) for row in positives])
        good_risk_before = mean([f(row.get(field)) for row in goods])
        if family == "current_support_conditioned_preserve":
            bad_after = bad_risk_before * 0.98 if math.isfinite(bad_risk_before) else math.nan
            good_after = good_risk_before * 0.99 if math.isfinite(good_risk_before) else math.nan
            safe_preservation = 0.99
        else:
            bad_after = bad_risk_before * 0.90 if math.isfinite(bad_risk_before) else math.nan
            good_after = good_risk_before * 0.95 if math.isfinite(good_risk_before) else math.nan
            safe_preservation = 0.95
        bad_reduction = bad_risk_before - bad_after if all(math.isfinite(v) for v in [bad_risk_before, bad_after]) else math.nan
        good_harm = max(0.0, good_after - good_risk_before) if all(math.isfinite(v) for v in [good_risk_before, good_after]) else math.nan
        rows.append(
            {
                "action_family": family,
                "simulated_field": field,
                "prereq_pass": prereq_pass,
                "bad_simulated_risk_before": bad_risk_before,
                "bad_simulated_risk_after": bad_after,
                "bad_simulated_risk_reduction": bad_reduction,
                "good_simulated_risk_before": good_risk_before,
                "good_simulated_risk_after": good_after,
                "good_simulated_harm": good_harm,
                "safe_preservation": safe_preservation,
                "actual_minus_random_margin": 0.0,
                "anchor_id_rotation_control_fails": False,
                "query_head_random_control_fails": False,
                "gate_pass": False,
                "note": "Simulator kept blocked because Track N/O/C3 prerequisites did not pass identity-specific controls." if not prereq_pass else "Proxy simulation only; controls not passed.",
            }
        )
    summary = {
        "schema": "acl2_v99_trackM2_identity_action_simulator_v1",
        "status": "blocked_prereq_no_action_sim" if not prereq_pass else "complete_no_passing_action_family",
        "gate_pass": False,
        "case_count": len(case_rows),
        "prereq_trackN_gate_pass": track_n_summary.get("gate_pass"),
        "prereq_trackO_gate_pass": track_o_summary.get("gate_pass"),
        "prereq_trackC3_gate_pass": track_c3_summary.get("gate_pass"),
        "passing_action_family_count": 0,
        "runtime_action_allowed": False,
        "blocker": "No action family may enter runtime pilot because identity/current-support/latent gates did not pass.",
    }
    write_json(out / "summary.json", summary)
    write_rows(out / "simulator_rows.csv", rows)
    write_text(
        out / "simulator_failure_attribution.md",
        "# Track M2 Failure Attribution\n\n"
        f"- prereq_trackN_gate_pass={summary['prereq_trackN_gate_pass']}\n"
        f"- prereq_trackO_gate_pass={summary['prereq_trackO_gate_pass']}\n"
        f"- prereq_trackC3_gate_pass={summary['prereq_trackC3_gate_pass']}\n"
        "- v99 plan only permits runtime pilots after passing M2; no runtime pilot was launched from this blocked simulator.\n",
    )
    return summary


def final_decision(stage0_summary: dict[str, Any], track_n_summary: dict[str, Any], track_o_summary: dict[str, Any], track_c3_summary: dict[str, Any], track_m2_summary: dict[str, Any]) -> dict[str, Any]:
    out = ROOT / "final_decision"
    full_method_success = False
    runtime_allowed = b(track_m2_summary.get("gate_pass"))
    if runtime_allowed:
        taxonomy = "IDENTITY_GATES_PASS_RUNTIME_PILOT_REQUIRED"
    elif b(track_n_summary.get("gate_pass")):
        taxonomy = "IDENTITY_DIAGNOSTIC_PASS_DOWNSTREAM_SAFETY_NO_GO"
    else:
        taxonomy = "IDENTITY_LIFECYCLE_CONTROL_NO_GO"
    decision = {
        "schema": "acl2_v99_final_decision_v1",
        "final_taxonomy": taxonomy,
        "full_method_success": full_method_success,
        "runtime_action_pilot_run": False,
        "full_validation_run": False,
        "runtime_action_allowed": runtime_allowed,
        "stage0_gate_pass": stage0_summary.get("gate_pass"),
        "trackN_gate_pass": track_n_summary.get("gate_pass"),
        "trackO_gate_pass": track_o_summary.get("gate_pass"),
        "trackC3_gate_pass": track_c3_summary.get("gate_pass"),
        "trackM2_gate_pass": track_m2_summary.get("gate_pass"),
        "case_count": track_n_summary.get("case_count"),
        "sequence_coverage": track_n_summary.get("sequence_coverage"),
        "primary_blocker": track_m2_summary.get("blocker") or track_c3_summary.get("blocker") or track_o_summary.get("blocker") or track_n_summary.get("blocker"),
        "claim": "No runtime or full validation success is claimed.",
    }
    write_json(out / "final_decision.json", decision)
    write_json(out / "summary.json", decision)
    write_text(
        out / "final_report.md",
        "# ACL2 v99 Final Report\n\n"
        f"- final_taxonomy: {taxonomy}\n"
        f"- full_method_success: {full_method_success}\n"
        f"- runtime_action_pilot_run: false\n"
        f"- full_validation_run: false\n"
        f"- TrackN gate: {track_n_summary.get('gate_pass')} (case_count={track_n_summary.get('case_count')}, sequence_coverage={track_n_summary.get('sequence_coverage')})\n"
        f"- TrackO gate: {track_o_summary.get('gate_pass')}\n"
        f"- TrackC3 gate: {track_c3_summary.get('gate_pass')}\n"
        f"- TrackM2 gate: {track_m2_summary.get('gate_pass')}\n"
        f"- primary_blocker: {decision['primary_blocker']}\n",
    )
    write_text(
        out / "failure_report.md",
        "# ACL2 v99 Failure Report\n\n"
        "- The run produced an expanded identity trace and built Track I/N/O/C3/M2 artifacts.\n"
        "- Runtime action and full validation were not run because prerequisite gates did not pass.\n"
        "- Key repair direction: add true per-anchor lifecycle/current-support rows and z_write/z_cache/z_current feature dumps, then rerun Track N/O/C3 before any action pilot.\n",
    )
    return decision


def main() -> None:
    global ROOT, EXPANDED_TRACE_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    ROOT = args.root
    EXPANDED_TRACE_ROOT = ROOT / "trackN_identity_graph_expanded_probe28"
    stage0_summary = stage0()
    track_n_summary, case_rows, _ = track_n()
    track_o_summary = track_o(case_rows)
    track_c3_summary = track_c3(case_rows)
    track_m2_summary = track_m2(case_rows, track_n_summary, track_o_summary, track_c3_summary)
    decision = final_decision(stage0_summary, track_n_summary, track_o_summary, track_c3_summary, track_m2_summary)
    print(json.dumps(decision, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
