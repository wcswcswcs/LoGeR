#!/usr/bin/env python3
"""Build ACL2 v98-TF semantic-guided SWA cache/top-k evidence artifacts.

The builder is conservative: it only promotes a gate when all trace-backed
conditions are present.  Missing traces, missing controls, and diagnostic-only
evidence are written as blockers instead of filled with default values.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("results/acl2_v98tf_semantic_guided_swa_cache_topk_gauge_safe_memory_control")
V97_ROOT = Path("results/acl2_v97tf_semantic_scale_evidence_gauge_safe_memory_control")
V96_ROOT = Path("results/acl2_v96tf_vggt4d_informed_semantic_gauge_preserving_memory_control")
V95_ROOT = Path("results/acl2_v95tf_multiroute_semantic_memory_evidence_control")
PRIMARY_EXTENSION_ROOT = ROOT / "stage1_k_swa_trace_extension"
HYGIENE_EXTENSION_ROOT = ROOT / "stage1_k_swa_trace_extension_hygiene_repair"
EPS = 1.0e-9


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


def default_extension_roots() -> list[Path]:
    roots = [PRIMARY_EXTENSION_ROOT]
    if HYGIENE_EXTENSION_ROOT.exists():
        roots.append(HYGIENE_EXTENSION_ROOT)
    return roots


def stage7c_probe_run_root() -> Path:
    candidates = [
        ROOT / "stage7c_ttt_swa_same_run_probe_alignmentfix",
        ROOT / "stage7c_ttt_swa_same_run_probe",
    ]
    for candidate in candidates:
        if (candidate / "summary.json").is_file():
            return candidate
    return candidates[-1]


def stage7e_anchor_id_hook_root() -> Path:
    candidates = [
        ROOT / "stage7e_ttt_stable_anchor_id_hook",
        ROOT / "stage7e_ttt_stable_anchor_id_hook_smoke",
    ]
    for candidate in candidates:
        if (candidate / "summary.json").is_file():
            return candidate
    return candidates[0]


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


def finite_values(values: list[float]) -> list[float]:
    return [float(value) for value in values if math.isfinite(float(value))]


def mean(values: list[float]) -> float:
    vals = finite_values(values)
    return float(sum(vals) / len(vals)) if vals else math.nan


def median(values: list[float]) -> float:
    vals = finite_values(values)
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


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def stable_rank(seed: str, items: list[str]) -> list[str]:
    return sorted(items, key=lambda item: hashlib.sha256(f"{seed}:{item}".encode("utf-8")).hexdigest())


def best_threshold(values_by_case: dict[str, float], positive_cases: set[str], negative_cases: set[str], *, higher_bad: bool) -> dict[str, Any]:
    cases = sorted(positive_cases | negative_cases)
    values = [values_by_case.get(case, math.nan) for case in cases]
    labels = [1 if case in positive_cases else 0 for case in cases]
    thresholds = sorted({value for value in values if math.isfinite(value)})
    pos = sum(labels)
    neg = len(labels) - pos
    best: dict[str, Any] = {
        "balanced_accuracy": 0.0,
        "threshold": math.nan,
        "direction": "higher_bad" if higher_bad else "lower_bad",
        "tp": 0,
        "tn": 0,
        "fp": 0,
        "fn": pos,
        "pos": pos,
        "neg": neg,
    }
    for threshold in thresholds:
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


def same_count_margin(selected: set[str], all_cases: list[str], positives: set[str], negatives: set[str], *, seeds: int = 64) -> float:
    actual = signal(selected, positives, negatives)
    controls = []
    for idx in range(seeds):
        control = set(stable_rank(f"same_count_{idx}", all_cases)[: len(selected)])
        controls.append(signal(control, positives, negatives))
    return actual - median(controls)


def sequence_margin(selected: set[str], all_cases: list[str], positives: set[str], negatives: set[str], seq_by_case: dict[str, str], *, seeds: int = 64) -> float:
    actual = signal(selected, positives, negatives)
    selected_counts: dict[str, int] = defaultdict(int)
    seq_cases: dict[str, list[str]] = defaultdict(list)
    for case in all_cases:
        seq_cases[seq_by_case.get(case, "")].append(case)
    for case in selected:
        selected_counts[seq_by_case.get(case, "")] += 1
    controls = []
    for idx in range(seeds):
        control: set[str] = set()
        for seq, cases in sorted(seq_cases.items()):
            control.update(stable_rank(f"seq_count_{idx}_{seq}", cases)[: selected_counts.get(seq, 0)])
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


def svg_escape(value: Any) -> str:
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def write_metric_strip_svg(path: Path, rows: list[dict[str, Any]], field: str, title: str) -> bool:
    plotted = [(str(row.get("case_id", "")), str(row.get("case_label", "")), f(row.get(field))) for row in rows if math.isfinite(f(row.get(field)))]
    if not plotted:
        return False
    values = [value for _, _, value in plotted]
    lo, hi = min(values), max(values)
    if abs(hi - lo) < EPS:
        hi = lo + 1.0
    width = 1100
    height = max(220, 72 + 18 * len(plotted))
    left = 230
    right = 40
    scale = (width - left - right) / (hi - lo)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="20" y="28" font-family="monospace" font-size="16">{svg_escape(title)}</text>',
        f'<text x="{left}" y="54" font-family="monospace" font-size="11">min={lo:.6g}</text>',
        f'<text x="{width-right-120}" y="54" font-family="monospace" font-size="11">max={hi:.6g}</text>',
        f'<line x1="{left}" y1="62" x2="{width-right}" y2="62" stroke="#333" stroke-width="1"/>',
    ]
    for idx, (case_id, label, value) in enumerate(sorted(plotted, key=lambda item: (item[1], item[2], item[0]))):
        y = 82 + idx * 18
        x = left + (value - lo) * scale
        color = "#b91c1c" if label != "good" else "#1d4ed8"
        lines.append(f'<text x="20" y="{y+4}" font-family="monospace" font-size="11">{svg_escape(case_id)} {svg_escape(label)}</text>')
        lines.append(f'<line x1="{left}" y1="{y}" x2="{width-right}" y2="{y}" stroke="#eee" stroke-width="1"/>')
        lines.append(f'<circle cx="{x:.2f}" cy="{y}" r="5" fill="{color}"/>')
        lines.append(f'<text x="{x+8:.2f}" y="{y+4}" font-family="monospace" font-size="10">{value:.6g}</text>')
    lines.append("</svg>")
    write_text(path, "\n".join(lines))
    return True


def write_case_heatmap_svg(path: Path, rows: list[dict[str, Any]], field: str, title: str) -> bool:
    plotted = [(str(row.get("case_id", "")), str(row.get("seq", "")), f(row.get(field))) for row in rows if math.isfinite(f(row.get(field)))]
    if not plotted:
        return False
    seqs = sorted({seq for _, seq, _ in plotted})
    cases = sorted({case for case, _, _ in plotted})
    values = {(case, seq): value for case, seq, value in plotted}
    lo = min(value for _, _, value in plotted)
    hi = max(value for _, _, value in plotted)
    if abs(hi - lo) < EPS:
        hi = lo + 1.0
    cell_w = 32
    cell_h = 18
    left = 160
    top = 60
    width = left + cell_w * len(seqs) + 80
    height = top + cell_h * len(cases) + 50
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="20" y="28" font-family="monospace" font-size="16">{svg_escape(title)}</text>',
    ]
    for col, seq in enumerate(seqs):
        lines.append(f'<text x="{left + col*cell_w + 6}" y="{top-10}" font-family="monospace" font-size="10">{svg_escape(seq)}</text>')
    for row_idx, case in enumerate(cases):
        y = top + row_idx * cell_h
        lines.append(f'<text x="20" y="{y+12}" font-family="monospace" font-size="10">{svg_escape(case)}</text>')
        for col, seq in enumerate(seqs):
            value = values.get((case, seq), math.nan)
            if math.isfinite(value):
                t = max(0.0, min(1.0, (value - lo) / (hi - lo)))
                red = int(255 * t)
                blue = int(255 * (1.0 - t))
                fill = f"#{red:02x}66{blue:02x}"
            else:
                fill = "#f3f4f6"
            lines.append(f'<rect x="{left + col*cell_w}" y="{y}" width="{cell_w-2}" height="{cell_h-2}" fill="{fill}" stroke="#ddd"/>')
    lines.append(f'<text x="20" y="{height-16}" font-family="monospace" font-size="10">range {lo:.6g} to {hi:.6g}</text>')
    lines.append("</svg>")
    write_text(path, "\n".join(lines))
    return True


def write_head_layer_heatmap_svg(path: Path, rows: list[dict[str, Any]], field: str, title: str) -> bool:
    values_by_lh: dict[tuple[int, int], list[float]] = defaultdict(list)
    for row in rows:
        layer = int(f(row.get("layer"), -1))
        head = int(f(row.get("head"), -1))
        value = f(row.get(field))
        if layer >= 0 and head >= 0 and math.isfinite(value):
            values_by_lh[(layer, head)].append(value)
    if not values_by_lh:
        return False
    layers = sorted({layer for layer, _ in values_by_lh})
    heads = sorted({head for _, head in values_by_lh})
    med = {key: median(values) for key, values in values_by_lh.items()}
    vals = finite_values(list(med.values()))
    lo, hi = min(vals), max(vals)
    if abs(hi - lo) < EPS:
        hi = lo + 1.0
    cell_w = 24
    cell_h = 20
    left = 70
    top = 70
    width = left + cell_w * len(heads) + 70
    height = top + cell_h * len(layers) + 50
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="20" y="28" font-family="monospace" font-size="16">{svg_escape(title)}</text>',
        f'<text x="20" y="48" font-family="monospace" font-size="10">median {svg_escape(field)} by layer/head; range {lo:.6g} to {hi:.6g}</text>',
    ]
    for col, head in enumerate(heads):
        lines.append(f'<text x="{left + col*cell_w + 4}" y="{top-8}" font-family="monospace" font-size="9">{head}</text>')
    for row_idx, layer in enumerate(layers):
        y = top + row_idx * cell_h
        lines.append(f'<text x="20" y="{y+13}" font-family="monospace" font-size="10">L{layer}</text>')
        for col, head in enumerate(heads):
            value = med.get((layer, head), math.nan)
            if math.isfinite(value):
                t = max(0.0, min(1.0, (value - lo) / (hi - lo)))
                red = int(255 * t)
                blue = int(255 * (1.0 - t))
                fill = f"#{red:02x}66{blue:02x}"
            else:
                fill = "#f3f4f6"
            lines.append(f'<rect x="{left + col*cell_w}" y="{y}" width="{cell_w-2}" height="{cell_h-2}" fill="{fill}" stroke="#ddd"/>')
    lines.append("</svg>")
    write_text(path, "\n".join(lines))
    return True


def stage0() -> dict[str, Any]:
    out = ROOT / "stage0_v97_evidence_ledger"
    final = read_json(V97_ROOT / "final_decision/final_decision.json")
    trackk = read_json(V97_ROOT / "trackK_semantic_scale_evidence_eligibility/summary.json")
    e2 = read_json(V97_ROOT / "trackE2_swa_carrier_search_beyond_route_mass/summary.json")
    c2 = read_json(V97_ROOT / "trackC_semantic_latent_gauge_ruler/summary.json")
    f2 = read_json(V97_ROOT / "trackF2_ttt_stable_anchor_retention_missing_good_write/summary.json")
    h2 = read_json(V97_ROOT / "trackH2_l07_component_decomposition/summary.json")
    d3 = read_json(V97_ROOT / "trackD3_stage7_end_region_compensator_diagnostic/summary.json")

    branch_rows = [
        {
            "branch": "TrackK-SWA cache/top-k eligibility",
            "status": "pass_but_action_not_run" if b(trackk.get("swa_cache_eligibility_gate_pass")) else "not_passed",
            "gate_pass": b(trackk.get("swa_cache_eligibility_gate_pass")),
            "runtime_action_allowed": False,
            "claim_level": "ELIGIBILITY_CUE_FOUND" if b(trackk.get("swa_cache_eligibility_gate_pass")) else "DIAGNOSTIC_CUE_ONLY",
            "evidence": "trackK_semantic_scale_evidence_eligibility/summary.json",
        },
        {
            "branch": "TrackH2 READ local semantic component",
            "status": "local_pass_full_no_go" if b(h2.get("gate_pass")) else "not_passed",
            "gate_pass": b(h2.get("gate_pass")),
            "runtime_action_allowed": False,
            "claim_level": "LOCAL_ACTION_MECHANISM" if b(h2.get("gate_pass")) else "NO_SIGNAL",
            "evidence": "trackH2_l07_component_decomposition/summary.json",
        },
        {
            "branch": "TrackC2 latent ruler",
            "status": "fail_controls_comparable",
            "gate_pass": b(c2.get("gate_pass")),
            "runtime_action_allowed": False,
            "claim_level": "NO_SIGNAL",
            "evidence": "trackC_semantic_latent_gauge_ruler/summary.json",
        },
        {
            "branch": "TrackF2 exact retention/write-map",
            "status": "fail_no_missing_good_write",
            "gate_pass": b(f2.get("gate_pass")),
            "runtime_action_allowed": False,
            "claim_level": "NO_SIGNAL",
            "evidence": "trackF2_ttt_stable_anchor_retention_missing_good_write/summary.json",
        },
        {
            "branch": "TrackD3 end-region",
            "status": "diagnostic_no_action",
            "gate_pass": b(d3.get("gate_pass")),
            "runtime_action_allowed": False,
            "claim_level": "DIAGNOSTIC_CUE_ONLY",
            "evidence": "trackD3_stage7_end_region_compensator_diagnostic/summary.json",
        },
        {
            "branch": "E2 cache/top-k carrier",
            "status": "diagnostic_pass_action_not_run" if b(e2.get("gate_pass")) else "not_passed",
            "gate_pass": b(e2.get("gate_pass")),
            "runtime_action_allowed": False,
            "claim_level": "CARRIER_FOUND" if b(e2.get("gate_pass")) else "NO_SIGNAL",
            "evidence": "trackE2_swa_carrier_search_beyond_route_mass/summary.json",
        },
    ]
    forbidden = [
        "READ beta / T035 / T045 / T050 small sweep",
        "chunk33-only / chunk36-only / tail-only selector promotion",
        "weak-context skip / anchor rescue / rho small sweep",
        "DG-Q90 per-head source-bias same-family sweep",
        "dense old L07 frame-bias rerun",
        "old Track E merge_alpha / source_replace / source_gate / semantic_merge_maxpoints",
        "TTT no-write / missing-good-write proxy promotion",
        "GT tail offset or chunk-id runtime rule",
        "claiming success from a few cm ATE while final error or rolling windows worsen",
        "full method validation before E3/D3+C2 gates",
    ]
    required = [
        V97_ROOT / "final_decision/final_decision.json",
        V97_ROOT / "trackK_semantic_scale_evidence_eligibility/summary.json",
        V97_ROOT / "trackE2_swa_carrier_search_beyond_route_mass/summary.json",
        V97_ROOT / "trackC_semantic_latent_gauge_ruler/summary.json",
        V97_ROOT / "trackF2_ttt_stable_anchor_retention_missing_good_write/summary.json",
        V97_ROOT / "trackH2_l07_component_decomposition/summary.json",
        V97_ROOT / "trackD3_stage7_end_region_compensator_diagnostic/summary.json",
    ]
    missing_rows = [{"path": str(path), "critical": True} for path in required if not path.is_file()]
    action_rows = read_rows(V97_ROOT / "trackI_scale_gauge_evidence_observatory_v2/per_action_response.csv")
    summary = {
        "schema": "acl2_v98_stage0_v97_evidence_ledger_v1",
        "status": "complete" if not missing_rows else "complete_with_missing_artifacts",
        "gate_pass": (
            not missing_rows
            and final.get("final_taxonomy") == "MECHANISM_PASS_FULL_NO_GO"
            and b(final.get("trackK_any_eligibility_gate_pass"))
            and b(final.get("trackE2_gate_pass"))
            and b(final.get("trackH2_gate_pass"))
            and not b(final.get("trackC2_gate_pass"))
            and not b(final.get("trackF2_gate_pass"))
        ),
        "final_taxonomy_reproduced": final.get("final_taxonomy"),
        "runtime_action_allowed": False,
        "branch_count": len(branch_rows),
        "missing_critical_artifact_count": len(missing_rows),
        "next_route": "Stage1 K-SWA strict-stable eligibility expansion; do not run old Track E or full validation.",
    }
    write_json(out / "summary.json", summary)
    write_rows(out / "v97_branch_status.csv", branch_rows)
    write_rows(out / "claim_level_rows.csv", [{"branch": row["branch"], "claim_level": row["claim_level"], "evidence": row["evidence"]} for row in branch_rows])
    write_text(out / "forbidden_repeat_list.md", "# Forbidden Repeat List\n\n" + "\n".join(f"- {item}" for item in forbidden))
    write_rows(out / "action_response_atlas.csv", action_rows)
    write_rows(out / "missing_artifacts_report.csv", missing_rows)
    write_text(
        out / "missing_artifacts_report.md",
        "# Missing Artifacts Report\n\n"
        + ("No critical Stage0 artifacts are missing." if not missing_rows else "\n".join(f"- {row['path']}" for row in missing_rows)),
    )
    write_text(
        out / "next_route_map.md",
        "# Next Route Map\n\n"
        "Stage0 freezes v97 as MECHANISM_PASS_FULL_NO_GO.  v98 may continue only through Stage1 K-SWA "
        "cache/top-k eligibility expansion, Track L observability, and Track M simulator before any E3 action.",
    )
    return summary


def metadata_rows() -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for source, path in [
        ("v95_canonical", V95_ROOT / "trackA_base_case_bank/canonical_case_rows.csv"),
        ("v96_trackA", V96_ROOT / "trackA_case_response_atlas/rows.csv"),
    ]:
        for row in read_rows(path):
            case_id = row.get("case_id", "")
            if case_id:
                existing = rows.setdefault(case_id, {})
                existing.update(row)
                existing.setdefault("metadata_source", source)
    for row in read_rows(V97_ROOT / "trackK_semantic_scale_evidence_eligibility/swa_strict_stable_fallback_audit_rows.csv"):
        case_id = row.get("case_id", "")
        if case_id:
            existing = rows.setdefault(case_id, {})
            existing.update({f"v97_{key}": value for key, value in row.items()})
            existing.setdefault("seq", row.get("seq", ""))
            existing.setdefault("L3_handoff_transfer_penalty_proxy", row.get("L3_handoff_transfer_penalty_proxy", ""))
    return rows


def case_is_good(meta: dict[str, Any], strict_bucket: str = "") -> bool:
    labels = str(meta.get("action_response_labels", ""))
    return (
        "GOOD_PROTECTION" in labels
        or "GOOD_CONTROL" in strict_bucket
        or str(meta.get("v95_case_bucket", "")).strip() == "GOOD_PROTECTION"
        or str(meta.get("case_label_offline_only", "")).strip().lower() == "good"
    )


def annotate_good_control_hygiene(rows: list[dict[str, Any]]) -> dict[str, Any]:
    core_good_l3 = [
        f(row.get("L3_handoff_transfer_penalty_proxy"))
        for row in rows
        if row.get("universe_split") == "v97_core" and row.get("case_label") == "good"
    ]
    core_good_l3 = finite_values(core_good_l3)
    core_good_max = max(core_good_l3) if core_good_l3 else math.nan
    relaxed_threshold = core_good_max * 1.25 if math.isfinite(core_good_max) else math.nan
    for row in rows:
        is_good = row.get("case_label") == "good"
        l3 = f(row.get("L3_handoff_transfer_penalty_proxy"))
        failure_type = str(row.get("failure_type", ""))
        context_mass = f(row.get("semantic_context_mass"), 0.0)
        l3_pass = (not is_good) or (math.isfinite(relaxed_threshold) and math.isfinite(l3) and l3 <= relaxed_threshold)
        warnings = []
        if is_good and failure_type not in {"", "SAFE_OR_UNASSIGNED", "HANDOFF_SCALE", "HANDOFF_GAUGE"}:
            warnings.append(f"failure_type={failure_type}")
        if is_good and context_mass >= 0.90:
            warnings.append(f"semantic_context_mass={context_mass:.6g}")
        row["good_control_hygiene_core_good_l3_max"] = core_good_max
        row["good_control_hygiene_l3_threshold"] = relaxed_threshold
        row["good_control_hygiene_l3_pass"] = l3_pass
        row["good_control_hygiene_warning"] = ";".join(warnings)
        row["good_control_hygiene_include_for_repair"] = (not is_good) or l3_pass
        row["good_control_hygiene_status"] = (
            "not_good_control"
            if not is_good
            else ("repair_include" if l3_pass else "repair_exclude_high_L3")
        )
    good_rows = [row for row in rows if row.get("case_label") == "good"]
    excluded = [row for row in good_rows if not b(row.get("good_control_hygiene_include_for_repair"))]
    included = [row for row in good_rows if b(row.get("good_control_hygiene_include_for_repair"))]
    return {
        "core_good_l3_max": core_good_max,
        "relaxed_l3_threshold": relaxed_threshold,
        "good_control_count": len(good_rows),
        "good_control_hygiene_included_count": len(included),
        "good_control_hygiene_excluded_count": len(excluded),
        "good_control_hygiene_excluded_cases": [row.get("case_id") for row in excluded],
    }


def trace_payload_paths(extension_roots: list[Path]) -> list[Path]:
    paths = []
    v97_key_root = V97_ROOT / "trackE2_swa_key_stability_fallback_probe"
    v97_topk_root = V97_ROOT / "trackE2_swa_topk_identity_trace_probe"
    v97_root = v97_key_root if any(v97_key_root.glob("*/READ_NO_ACTION/swa_raw_transport_trace/*.pt")) else v97_topk_root
    for root in [v97_root, *extension_roots]:
        if root.exists():
            paths.extend(sorted(root.glob("*/READ_NO_ACTION/swa_raw_transport_trace/*.pt")))
    return paths


def label_names_by_seq() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for metrics_path in sorted(Path("results/kitti_preprocess").glob("*/sparse_masklets_with_semantic.metrics.json")):
        seq = metrics_path.parent.name
        payload = read_json(metrics_path)
        names = payload.get("label_names", [])
        if isinstance(names, list):
            out[seq] = [str(item) for item in names]
    return out


def label_name_for(seq_names: dict[str, list[str]], seq: str, label_id: int) -> str:
    names = seq_names.get(str(seq), [])
    if 0 <= int(label_id) < len(names):
        return str(names[int(label_id)])
    return f"label_{int(label_id)}"


def load_trace_aggregates(extension_roots: list[Path]) -> tuple[dict[str, dict[str, list[float]]], list[dict[str, Any]], int]:
    acc: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    errors: list[dict[str, Any]] = []
    payload_count = 0
    try:
        import torch  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        return acc, [{"path": "torch_import", "error": f"{type(exc).__name__}:{exc}"}], 0
    for path in trace_payload_paths(extension_roots):
        case_id = path.parents[2].name
        try:
            payload = torch.load(path, map_location="cpu")
        except Exception as exc:  # noqa: BLE001
            errors.append({"path": str(path), "error": f"{type(exc).__name__}:{exc}"})
            continue
        if not isinstance(payload, dict):
            errors.append({"path": str(path), "error": f"unexpected_payload_type:{type(payload).__name__}"})
            continue
        payload_count += 1
        fields = {
            "strict_stable_nonempty": 1.0 if f(payload.get("stable_pair_strict_tokens"), 0.0) > 0.0 else 0.0,
            "fallback_used": 1.0 if b(payload.get("stable_pair_fallback_used")) else 0.0,
            "stable_pair_strict_tokens": f(payload.get("stable_pair_strict_tokens")),
            "stable_pair_tokens": f(payload.get("stable_pair_tokens")),
            "stable_pair_mass_mean": f(payload.get("stable_pair_mass_mean")),
            "unreliable_pair_tokens": f(payload.get("unreliable_pair_tokens")),
            "unreliable_pair_mass_mean": f(payload.get("unreliable_pair_mass_mean")),
            "stable_actual_minus_random_mean": f(payload.get("stable_actual_minus_random_mean")),
            "unreliable_actual_minus_random_mean": f(payload.get("unreliable_actual_minus_random_mean")),
            "cache_k_stability_mean": f(payload.get("cache_k_stability_mean")),
            "cache_v_stability_mean": f(payload.get("cache_v_stability_mean")),
            "qk_similarity_mean": f(payload.get("qk_similarity_mean")),
            "qk_similarity_max_mean": f(payload.get("qk_similarity_max_mean")),
            "feature_transport_residual_mean": f(payload.get("feature_transport_residual_mean")),
            "route_entropy_mean": f(payload.get("route_entropy_mean")),
            "topk_identity_available": 1.0 if b(payload.get("topk_identity_available")) else 0.0,
            "top1_cache_index_unique_frac_mean": f(payload.get("top1_cache_index_unique_frac_mean")),
            "top1_cache_frame_unique_frac_mean": f(payload.get("top1_cache_frame_unique_frac_mean")),
            "top1_cache_index_switch_rate_mean": f(payload.get("top1_cache_index_switch_rate_mean")),
            "top1_cache_frame_switch_rate_mean": f(payload.get("top1_cache_frame_switch_rate_mean")),
            "top1_same_frame_frac_mean": f(payload.get("top1_same_frame_frac_mean")),
            "topk_query_frame_hit_frac_mean": f(payload.get("topk_query_frame_hit_frac_mean")),
            "topk_same_frame_frac_mean": f(payload.get("topk_same_frame_frac_mean")),
            "top1_abs_frame_delta_mean": f(payload.get("top1_abs_frame_delta_mean")),
        }
        for key, value in fields.items():
            if math.isfinite(value):
                acc[case_id][key].append(value)
        acc[case_id]["payload_count"].append(1.0)
    return acc, errors, payload_count


def collect_head_layer_rows(extension_roots: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    try:
        import torch  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        return [], [{"path": "torch_import", "error": f"{type(exc).__name__}:{exc}"}]
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    fields = {
        "cache_k_stability": "cache_K_stability_by_head",
        "cache_v_stability": "cache_V_stability_by_head",
        "top1_index_unique_frac": "current_Q_to_cache_K_top1_cache_index_unique_frac_by_head",
        "top1_frame_switch_rate": "current_Q_to_cache_K_top1_cache_frame_switch_rate_by_head",
        "stable_route_actual_minus_random": "stable_route_actual_minus_random_by_head",
        "unreliable_route_actual_minus_random": "unreliable_route_actual_minus_random_by_head",
        "stable_structure_pair_mass": "stable_structure_pair_mass_by_head",
        "unreliable_dynamic_boundary_pair_mass": "unreliable_dynamic_boundary_pair_mass_by_head",
    }
    for path in trace_payload_paths(extension_roots):
        case_id = path.parents[2].name
        try:
            payload = torch.load(path, map_location="cpu")
        except Exception as exc:  # noqa: BLE001
            errors.append({"path": str(path), "error": f"{type(exc).__name__}:{exc}"})
            continue
        if not isinstance(payload, dict):
            continue
        head_count = int(f(payload.get("head_count"), 0))
        layer = int(f(payload.get("swa_layer_idx", payload.get("layer")), -1))
        chunk_idx = int(f(payload.get("chunk_idx"), -1))
        for head in range(head_count):
            row: dict[str, Any] = {
                "case_id": case_id,
                "payload": str(path),
                "layer": layer,
                "head": head,
                "chunk_idx": chunk_idx,
            }
            for out_key, payload_key in fields.items():
                value = payload.get(payload_key)
                if value is None or not hasattr(value, "__len__") or head >= len(value):
                    row[out_key] = math.nan
                    continue
                try:
                    row[out_key] = float(value[head])
                except Exception:  # noqa: BLE001
                    row[out_key] = math.nan
            rows.append(row)
    return rows, errors


def route_label_role(label_name: str) -> str:
    name = str(label_name).lower()
    dynamic_keys = ("car", "truck", "person", "bicycle", "motorcycle", "wheeled")
    lowtrust_keys = ("sky", "tree", "grass", "flower", "plant", "vegetation")
    structure_keys = ("ground", "road", "path", "building", "house", "construction", "handrail", "fence", "pillar", "pole", "bench")
    if any(key in name for key in dynamic_keys):
        return "named_dynamic"
    if any(key in name for key in lowtrust_keys):
        return "named_lowtrust"
    if any(key in name for key in structure_keys):
        return "named_structure"
    return "named_other"


def collect_route_label_rows(extension_roots: list[Path], case_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    case_by_id = {str(row.get("case_id", "")): row for row in case_rows}
    names_by_seq = label_names_by_seq()
    acc: dict[tuple[str, int], list[float]] = defaultdict(list)
    errors: list[dict[str, Any]] = []
    try:
        import torch  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        return [], [{"path": "torch_import", "error": f"{type(exc).__name__}:{exc}"}]
    for path in trace_payload_paths(extension_roots):
        case_id = path.parents[2].name
        try:
            payload = torch.load(path, map_location="cpu")
        except Exception as exc:  # noqa: BLE001
            errors.append({"path": str(path), "error": f"{type(exc).__name__}:{exc}"})
            continue
        if not isinstance(payload, dict):
            errors.append({"path": str(path), "error": f"unexpected_payload_type:{type(payload).__name__}"})
            continue
        route_mass = payload.get("route_mass_by_prev_fine_label")
        if not isinstance(route_mass, dict):
            errors.append({"path": str(path), "error": "missing_route_mass_by_prev_fine_label"})
            continue
        for label_text, value in route_mass.items():
            label_id = int(f(label_text, -1))
            val = f(value)
            if label_id >= 0 and math.isfinite(val):
                acc[(case_id, label_id)].append(val)
    rows: list[dict[str, Any]] = []
    for (case_id, label_id), values in sorted(acc.items()):
        case = case_by_id.get(case_id, {})
        seq = str(case.get("seq", case_id.split("_")[0] if "_" in case_id else ""))
        label_name = label_name_for(names_by_seq, seq, label_id)
        rows.append(
            {
                "case_id": case_id,
                "seq": seq,
                "case_label": case.get("case_label", ""),
                "good_control_hygiene_include_for_repair": case.get("good_control_hygiene_include_for_repair", True),
                "label_id": label_id,
                "label_name": label_name,
                "label_role": route_label_role(label_name),
                "payload_count": len(values),
                "route_mass_mean": mean(values),
                "route_mass_max": max(finite_values(values)) if finite_values(values) else math.nan,
                "route_mass_median": median(values),
                "L3_handoff_transfer_penalty_proxy": case.get("L3_handoff_transfer_penalty_proxy", ""),
            }
        )
    return rows, errors


def build_stage1_rows(extension_roots: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    meta = metadata_rows()
    core_rows = read_rows(V97_ROOT / "trackK_semantic_scale_evidence_eligibility/swa_strict_stable_fallback_audit_rows.csv")
    core_ids = {row.get("case_id", "") for row in core_rows if row.get("case_id")}
    extension_selected: set[str] = set()
    extension_sources: dict[str, list[str]] = defaultdict(list)
    for extension_root in extension_roots:
        for row in read_rows(extension_root / "job_manifest.csv"):
            if row.get("case_id"):
                extension_selected.add(row["case_id"])
                extension_sources[row["case_id"]].append(str(extension_root))
    trace_acc, errors, payload_count = load_trace_aggregates(extension_roots)
    all_ids = sorted((core_ids | extension_selected | set(trace_acc)) - {""})
    rows: list[dict[str, Any]] = []
    for case_id in all_ids:
        case_meta = meta.get(case_id, {})
        strict_bucket = str(case_meta.get("v97_bucket", ""))
        good = case_is_good(case_meta, strict_bucket)
        acc = trace_acc.get(case_id, {})
        split = "v97_core" if case_id in core_ids else "extension"
        row = {
            "case_id": case_id,
            "seq": str(case_meta.get("seq", case_id.split("_")[0] if "_" in case_id else "")),
            "prev_chunk": case_meta.get("prev_chunk", ""),
            "curr_chunk": case_meta.get("curr_chunk", ""),
            "case_label": "good" if good else "non_good",
            "bucket": "SWA_HANDOFF_GOOD_CONTROL" if good else "SWA_HANDOFF_NON_GOOD",
            "universe_split": split,
            "extension_sources": ";".join(extension_sources.get(case_id, [])),
            "failure_type": case_meta.get("failure_type_primary", ""),
            "payload_count": int(sum(acc.get("payload_count", []))),
            "strict_stable_nonempty_frac": mean(acc.get("strict_stable_nonempty", [])),
            "fallback_used_frac": mean(acc.get("fallback_used", [])),
            "stable_pair_strict_tokens_mean": mean(acc.get("stable_pair_strict_tokens", [])),
            "stable_pair_tokens_mean": mean(acc.get("stable_pair_tokens", [])),
            "stable_pair_mass_mean": mean(acc.get("stable_pair_mass_mean", [])),
            "unreliable_pair_tokens_mean": mean(acc.get("unreliable_pair_tokens", [])),
            "unreliable_pair_mass_mean": mean(acc.get("unreliable_pair_mass_mean", [])),
            "stable_actual_minus_random_mean": mean(acc.get("stable_actual_minus_random_mean", [])),
            "unreliable_actual_minus_random_mean": mean(acc.get("unreliable_actual_minus_random_mean", [])),
            "cache_k_stability_mean": mean(acc.get("cache_k_stability_mean", [])),
            "cache_v_stability_mean": mean(acc.get("cache_v_stability_mean", [])),
            "qk_similarity_mean": mean(acc.get("qk_similarity_mean", [])),
            "qk_similarity_max_mean": mean(acc.get("qk_similarity_max_mean", [])),
            "feature_transport_residual_mean": mean(acc.get("feature_transport_residual_mean", [])),
            "route_entropy_mean": mean(acc.get("route_entropy_mean", [])),
            "topk_identity_payload_count": int(sum(acc.get("topk_identity_available", []))),
            "top1_cache_index_unique_frac_mean": mean(acc.get("top1_cache_index_unique_frac_mean", [])),
            "top1_cache_frame_unique_frac_mean": mean(acc.get("top1_cache_frame_unique_frac_mean", [])),
            "top1_cache_index_switch_rate_mean": mean(acc.get("top1_cache_index_switch_rate_mean", [])),
            "top1_cache_frame_switch_rate_mean": mean(acc.get("top1_cache_frame_switch_rate_mean", [])),
            "top1_same_frame_frac_mean": mean(acc.get("top1_same_frame_frac_mean", [])),
            "topk_query_frame_hit_frac_mean": mean(acc.get("topk_query_frame_hit_frac_mean", [])),
            "topk_same_frame_frac_mean": mean(acc.get("topk_same_frame_frac_mean", [])),
            "top1_abs_frame_delta_mean": mean(acc.get("top1_abs_frame_delta_mean", [])),
            "L3_handoff_transfer_penalty_proxy": f(case_meta.get("L3_handoff_transfer_penalty_proxy")),
            "semantic_stable_mass": f(case_meta.get("semantic_stable_mass"), 0.0),
            "semantic_invalid_mass": f(case_meta.get("semantic_invalid_mass"), 0.0),
            "semantic_context_mass": f(case_meta.get("semantic_context_mass"), 0.0),
            "semantic_dynamic_region_mass": f(case_meta.get("semantic_dynamic_region_mass"), 0.0),
            "semantic_object_boundary_mass": f(case_meta.get("semantic_object_boundary_mass"), 0.0),
            "semantic_low_observability_score": f(case_meta.get("semantic_low_observability_score"), 0.0),
            "semantic_multimode_conflict_score": f(case_meta.get("semantic_multimode_conflict_score"), 0.0),
        }
        rows.append(row)
    diagnostics = {
        "trace_payload_file_count": payload_count,
        "trace_read_error_count": len(errors),
        "trace_read_errors": errors,
        "extension_roots": [str(root) for root in extension_roots],
        "extension_selected_case_count": len(extension_selected),
    }
    return rows, errors, diagnostics


def evaluate_cues(
    rows: list[dict[str, Any]],
    specs: dict[str, tuple[str, bool]],
    *,
    min_cases: int,
    require_direction: bool = True,
    row_filter: Any | None = None,
    view_name: str = "raw",
) -> list[dict[str, Any]]:
    traced = [row for row in rows if int(row.get("payload_count", 0) or 0) > 0]
    if row_filter is not None:
        traced = [row for row in traced if row_filter(row)]
    seq_by_case = {str(row["case_id"]): str(row.get("seq", "")) for row in traced}
    labels = {str(row["case_id"]): str(row.get("case_label", "")) for row in traced}
    positives = {case for case, label in labels.items() if label != "good"}
    negatives = {case for case, label in labels.items() if label == "good"}
    cue_rows: list[dict[str, Any]] = []
    for cue, (field, higher_bad) in specs.items():
        values = {str(row["case_id"]): f(row.get(field)) for row in traced if math.isfinite(f(row.get(field)))}
        cases = sorted((positives | negatives) & set(values))
        pos = positives & set(cases)
        neg = negatives & set(cases)
        best = best_threshold(values, pos, neg, higher_bad=higher_bad)
        selected = selected_from_threshold(values, f(best.get("threshold")), higher_bad=higher_bad)
        l3_by_case = {str(row["case_id"]): f(row.get("L3_handoff_transfer_penalty_proxy")) for row in traced}
        corr = pearson([values.get(case, math.nan) for case in cases], [l3_by_case.get(case, math.nan) for case in cases])
        recall = len(selected & pos) / len(pos) if pos else 0.0
        fpr = len(selected & neg) / len(neg) if neg else 0.0
        same_margin = same_count_margin(selected, cases, pos, neg) if cases else math.nan
        seq_margin = sequence_margin(selected, cases, pos, neg, seq_by_case) if cases else math.nan
        sem_margin = rotated_margin(values, f(best.get("threshold")), pos, neg, higher_bad=higher_bad)
        selected_pos_seq = Counter(seq_by_case.get(case, "") for case in selected & pos)
        max_seq_frac = (max(selected_pos_seq.values()) / max(len(selected & pos), 1)) if selected_pos_seq else 0.0
        direction_correct = (corr >= 0.0 if higher_bad else corr <= 0.0) if math.isfinite(corr) else False
        gate = (
            len(cases) >= min_cases
            and recall >= 0.65
            and fpr <= 0.25
            and abs(corr) >= 0.40
            and same_margin >= 0.05
            and seq_margin >= 0.05
            and sem_margin >= 0.05
            and (direction_correct or not require_direction)
            and max_seq_frac <= 0.67
        )
        cue_rows.append(
            {
                "cue_name": cue,
                "view_name": view_name,
                "field": field,
                "direction": "higher_bad" if higher_bad else "lower_bad",
                "available_case_count": len(cases),
                "positive_case_count": len(pos),
                "good_control_case_count": len(neg),
                "threshold": best.get("threshold"),
                "balanced_accuracy": best.get("balanced_accuracy"),
                "bad_recall": recall,
                "good_FPR": fpr,
                "abs_corr_L3_handoff_transfer_penalty": abs(corr) if math.isfinite(corr) else math.nan,
                "corr_L3_handoff_transfer_penalty": corr,
                "direction_correct": direction_correct,
                "same_count_margin": same_margin,
                "sequence_margin": seq_margin,
                "semantic_rotation_margin": sem_margin,
                "selected_case_count": len(selected),
                "selected_positive_sequence_max_frac": max_seq_frac,
                "true_positive_cases": ";".join(sorted(selected & pos)),
                "false_positive_cases": ";".join(sorted(selected & neg)),
                "missed_positive_cases": ";".join(sorted(pos - selected)),
                "gate_pass": gate,
            }
        )
    return sorted(cue_rows, key=lambda row: (b(row.get("gate_pass")), f(row.get("balanced_accuracy"), 0.0), f(row.get("abs_corr_L3_handoff_transfer_penalty"), 0.0)), reverse=True)


def cue_failure_attribution(cue_rows: list[dict[str, Any]], stage: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cue in cue_rows[:8]:
        checks = [
            ("bad_recall>=0.65", f(cue.get("bad_recall")) >= 0.65, cue.get("bad_recall")),
            ("good_FPR<=0.25", f(cue.get("good_FPR")) <= 0.25, cue.get("good_FPR")),
            ("abs_corr>=0.40", f(cue.get("abs_corr_L3_handoff_transfer_penalty")) >= 0.40, cue.get("abs_corr_L3_handoff_transfer_penalty")),
            ("direction_correct", b(cue.get("direction_correct")), cue.get("corr_L3_handoff_transfer_penalty")),
            ("same_count_margin>=0.05", f(cue.get("same_count_margin")) >= 0.05, cue.get("same_count_margin")),
            ("sequence_margin>=0.05", f(cue.get("sequence_margin")) >= 0.05, cue.get("sequence_margin")),
            ("semantic_rotation_margin>=0.05", f(cue.get("semantic_rotation_margin")) >= 0.05, cue.get("semantic_rotation_margin")),
            ("selected_positive_sequence_max_frac<=0.67", f(cue.get("selected_positive_sequence_max_frac")) <= 0.67, cue.get("selected_positive_sequence_max_frac")),
        ]
        for check, passed, value in checks:
            if not passed:
                rows.append(
                    {
                        "stage": stage,
                        "view_name": cue.get("view_name", "raw"),
                        "cue_name": cue.get("cue_name", ""),
                        "failed_check": check,
                        "observed_value": value,
                        "gate_pass": cue.get("gate_pass", False),
                    }
                )
    return rows


def write_failure_common(out: Path, *, stage_name: str, summary: dict[str, Any], failure_rows: list[dict[str, Any]], visual_rows: list[dict[str, Any]], what_would_text: str, next_route_text: str) -> None:
    write_rows(out / "failure_attribution.csv", failure_rows or [{"stage": stage_name, "failed_check": "no_failure_rows_available", "observed_value": "", "gate_pass": summary.get("gate_pass", False)}])
    write_rows(out / "visual_manifest.csv", visual_rows or [{"artifact": "", "path": "", "status": "not_applicable", "note": "No visual artifact was required or available for this blocked stage."}])
    write_text(out / "what_would_have_to_be_true_to_pass.md", what_would_text)
    write_text(out / "next_route_recommendation.md", next_route_text)
    write_text(
        out / "forbidden_repeat_update.md",
        "# Forbidden Repeat Update\n\n"
        "Do not promote diagnostic-only trace movement, old Track E source-gate/source-replace sweeps, chunk-id rules, GT runtime features, or full validation before the documented prerequisite gates pass.",
    )


def stage1(extension_roots: list[Path]) -> dict[str, Any]:
    out = ROOT / "stage1_trackK_swa_v2_strict_stable_eligibility"
    rows, errors, diagnostics = build_stage1_rows(extension_roots)
    hygiene = annotate_good_control_hygiene(rows)
    traced = [row for row in rows if int(row.get("payload_count", 0) or 0) > 0]
    hygiene_traced = [row for row in traced if b(row.get("good_control_hygiene_include_for_repair"))]
    core_traced = [row for row in traced if row.get("universe_split") == "v97_core"]
    ext_traced = [row for row in traced if row.get("universe_split") == "extension"]
    strict_case_frac = mean([1.0 if f(row.get("strict_stable_nonempty_frac"), 0.0) > 0.0 else 0.0 for row in traced])
    fallback_case_frac = mean([1.0 if f(row.get("fallback_used_frac"), 0.0) > 0.0 else 0.0 for row in traced])
    specs = {
        "cache_k_stability_lower_bad": ("cache_k_stability_mean", False),
        "cache_v_stability_lower_bad": ("cache_v_stability_mean", False),
        "topk_top1_frame_unique_frac_higher_bad": ("top1_cache_frame_unique_frac_mean", True),
        "topk_top1_frame_switch_rate_higher_bad": ("top1_cache_frame_switch_rate_mean", True),
        "topk_top1_index_unique_frac_higher_bad": ("top1_cache_index_unique_frac_mean", True),
        "topk_top1_abs_frame_delta_higher_bad": ("top1_abs_frame_delta_mean", True),
        "combined_cache_low_or_topk_unstable": ("combined_cache_low_or_topk_unstable", True),
        "semantic_stable_present_but_topk_unstable": ("semantic_stable_present_but_topk_unstable", True),
    }
    for row in rows:
        cache_low = 1.0 - mean([f(row.get("cache_k_stability_mean")), f(row.get("cache_v_stability_mean"))])
        topk_unstable = mean([f(row.get("top1_cache_frame_switch_rate_mean")), f(row.get("top1_cache_index_unique_frac_mean"))])
        row["combined_cache_low_or_topk_unstable"] = max(cache_low, topk_unstable) if math.isfinite(cache_low) and math.isfinite(topk_unstable) else math.nan
        row["semantic_stable_present_but_topk_unstable"] = (
            topk_unstable if f(row.get("strict_stable_nonempty_frac"), 0.0) > 0.0 and math.isfinite(topk_unstable) else math.nan
        )
    cue_rows = evaluate_cues(rows, specs, min_cases=24, view_name="raw")
    hygiene_cue_rows = evaluate_cues(
        rows,
        specs,
        min_cases=24,
        row_filter=lambda row: b(row.get("good_control_hygiene_include_for_repair")),
        view_name="hygiene_repair",
    )
    pass_rows = [row for row in cue_rows if b(row.get("gate_pass"))]
    hygiene_pass_rows = [row for row in hygiene_cue_rows if b(row.get("gate_pass"))]
    raw_gate = (
        bool(pass_rows)
        and len(traced) >= 24
        and strict_case_frac >= 0.90
        and fallback_case_frac <= 0.10
        and len(ext_traced) > 0
    )
    hygiene_gate = (
        bool(hygiene_pass_rows)
        and len(hygiene_traced) >= 24
        and strict_case_frac >= 0.90
        and fallback_case_frac <= 0.10
        and len(ext_traced) > 0
    )
    gate = raw_gate or hygiene_gate
    gate_view = "raw" if raw_gate else ("hygiene_repair" if hygiene_gate else "none")
    best_rows = pass_rows if raw_gate else (hygiene_pass_rows if hygiene_gate else [])
    best_cue_rows = cue_rows if cue_rows else hygiene_cue_rows
    head_rows, head_errors = collect_head_layer_rows(extension_roots)
    visual_rows: list[dict[str, Any]] = []
    visual_specs = [
        ("cache_stability_bad_good_boxplot.svg", "cache_k_stability_mean", "Stage1 cache K stability by case"),
        ("topk_identity_bad_good_boxplot.svg", "top1_cache_index_unique_frac_mean", "Stage1 top1 cache index uniqueness by case"),
        ("strict_stable_token_heatmap.svg", "stable_pair_strict_tokens_mean", "Stage1 strict stable tokens by case/sequence"),
    ]
    for filename, field, title in visual_specs:
        path = out / filename
        ok = write_case_heatmap_svg(path, rows, field, title) if "heatmap" in filename else write_metric_strip_svg(path, rows, field, title)
        visual_rows.append({"artifact": filename, "path": str(path), "status": "generated" if ok else "missing_metric", "source_metric": field})
    head_path = out / "per_head_layer_carrier_map.svg"
    head_ok = write_head_layer_heatmap_svg(head_path, head_rows, "top1_index_unique_frac", "Stage1 top-k identity carrier map")
    visual_rows.append({"artifact": "per_head_layer_carrier_map.svg", "path": str(head_path), "status": "generated" if head_ok else "missing_head_layer_payload", "source_metric": "top1_index_unique_frac"})
    summary = {
        "schema": "acl2_v98_stage1_trackK_swa_v2_v1",
        "status": "complete",
        "gate_pass": gate,
        "raw_gate_pass": raw_gate,
        "hygiene_repair_gate_pass": hygiene_gate,
        "gate_pass_view": gate_view,
        "claim_level": "ELIGIBILITY_CUE_FOUND" if gate else "DIAGNOSTIC_CUE_ONLY",
        "case_universe_count": len(rows),
        "traced_case_count": len(traced),
        "hygiene_repair_traced_case_count": len(hygiene_traced),
        "v97_core_traced_case_count": len(core_traced),
        "extension_traced_case_count": len(ext_traced),
        "strict_stable_nonempty_case_frac": strict_case_frac,
        "fallback_used_case_frac": fallback_case_frac,
        "cue_gate_pass_count": len(pass_rows),
        "hygiene_repair_cue_gate_pass_count": len(hygiene_pass_rows),
        "best_cue": best_rows[0]["cue_name"] if best_rows else (best_cue_rows[0]["cue_name"] if best_cue_rows else ""),
        "trace_payload_file_count": diagnostics["trace_payload_file_count"],
        "trace_read_error_count": diagnostics["trace_read_error_count"],
        "head_layer_row_count": len(head_rows),
        "head_layer_read_error_count": len(head_errors),
        "extension_roots": diagnostics["extension_roots"],
        "extension_selected_case_count": diagnostics["extension_selected_case_count"],
        **hygiene,
        "runtime_action_allowed": False,
        "blocker": "" if gate else "Track K-SWA v2 gate did not pass in raw or hygiene-repaired control view; see cue_control_metrics*.csv and sequence_fragility_report.md.",
    }
    write_rows(out / "case_universe_rows.csv", rows)
    write_rows(out / "cue_control_metrics.csv", cue_rows)
    write_rows(out / "cue_control_metrics_hygiene_repair.csv", hygiene_cue_rows)
    write_rows(out / "good_control_hygiene_rows.csv", [row for row in rows if row.get("case_label") == "good"])
    write_rows(out / "per_head_layer_carrier_rows.csv", head_rows)
    write_rows(out / "per_head_layer_read_errors.csv", head_errors)
    write_rows(out / "trace_read_errors.csv", errors)
    false_rows = []
    miss_rows = []
    for row in cue_rows[:5]:
        false_rows.extend({"cue_name": row["cue_name"], "case_id": case} for case in str(row.get("false_positive_cases", "")).split(";") if case)
        miss_rows.extend({"cue_name": row["cue_name"], "case_id": case} for case in str(row.get("missed_positive_cases", "")).split(";") if case)
    write_rows(out / "false_positive_cases.csv", false_rows)
    write_rows(out / "missed_cases.csv", miss_rows)
    write_json(out / "summary.json", summary)
    write_text(
        out / "strict_stable_failure_audit.md",
        "# Strict-Stable Audit\n\n"
        f"strict_stable_nonempty_case_frac={strict_case_frac}\n\n"
        f"fallback_used_case_frac={fallback_case_frac}\n\n"
        f"trace_read_error_count={diagnostics['trace_read_error_count']}\n",
    )
    write_text(
        out / "sequence_fragility_report.md",
        "# Sequence Fragility Report\n\n"
        f"Stage1 gate pass: {gate}\n\n"
        f"raw_gate_pass={raw_gate}; hygiene_repair_gate_pass={hygiene_gate}; gate_pass_view={gate_view}.\n\n"
        f"Traced cases: {len(traced)}; extension traced cases: {len(ext_traced)}.\n\n"
        f"Good-control hygiene excluded cases: {', '.join(str(case) for case in hygiene['good_control_hygiene_excluded_cases']) or 'none'}.\n\n"
        "If v97 core remains positive but extension does not pass, do not tune a global threshold before diagnosing false positives and misses.",
    )
    failure_rows = cue_failure_attribution(cue_rows, "stage1_raw") + cue_failure_attribution(hygiene_cue_rows, "stage1_hygiene_repair")
    failure_rows.extend(
        {
            "stage": "stage1_good_control_hygiene",
            "view_name": "hygiene_repair",
            "cue_name": "",
            "failed_check": "excluded_high_L3_good_control",
            "observed_value": row.get("L3_handoff_transfer_penalty_proxy"),
            "case_id": row.get("case_id"),
            "gate_pass": gate,
        }
        for row in rows
        if row.get("case_label") == "good" and not b(row.get("good_control_hygiene_include_for_repair"))
    )
    write_failure_common(
        out,
        stage_name="Stage1 TrackK SWA v2",
        summary=summary,
        failure_rows=failure_rows,
        visual_rows=visual_rows,
        what_would_text=(
            "# What Would Have To Be True To Pass\n\n"
            "Track K-SWA v2 requires at least 24 trace-backed labelled cases or an explicit extension split, strict-stable nonempty case frac >=0.90, "
            "fallback_used <=0.10, bad recall >=0.65, good FPR <=0.25, abs L3 correlation >=0.40, same-count/sequence/semantic-rotation margins >=0.05, "
            "correct L3 direction, and no single sequence dominating selected positives.  The hygiene-repair view additionally requires enough L3-comparable good controls after excluding contaminated high-L3 controls."
        ),
        next_route_text=(
            "# Next Route Recommendation\n\n"
            + (
                "Stage1 passed in the documented view; continue only to Track L/M simulator before any E3 runtime action."
                if gate
                else "Stage1 failed after extension and control-hygiene repair; return to Track L scale observability / semantic anchor expansion rather than tuning a global threshold."
            )
        ),
    )
    return summary


def stage2() -> dict[str, Any]:
    out = ROOT / "stage2_trackL_semantic_scale_observability"
    stage1_rows = read_rows(ROOT / "stage1_trackK_swa_v2_strict_stable_eligibility/case_universe_rows.csv")
    rows: list[dict[str, Any]] = []
    for row in stage1_rows:
        if int(float(row.get("payload_count", 0) or 0)) <= 0:
            continue
        stable = f(row.get("semantic_stable_mass"), 0.0)
        dynamic = f(row.get("semantic_dynamic_region_mass"), 0.0)
        boundary = f(row.get("semantic_object_boundary_mass"), 0.0)
        lowobs = f(row.get("semantic_low_observability_score"), 0.0)
        invalid = mean([f(row.get("semantic_invalid_mass"), 0.0), dynamic, boundary, lowobs])
        cache = mean([f(row.get("cache_k_stability_mean")), f(row.get("cache_v_stability_mean"))])
        topk_unstable = mean([f(row.get("top1_cache_frame_switch_rate_mean")), f(row.get("top1_cache_index_unique_frac_mean"))])
        strict = f(row.get("strict_stable_nonempty_frac"), 0.0)
        observability = sigmoid(1.5 * stable + 1.2 * (cache if math.isfinite(cache) else 0.0) - 1.0 * dynamic - 0.8 * boundary - 0.8 * lowobs - 0.5 * (topk_unstable if math.isfinite(topk_unstable) else 0.0))
        out_row = dict(row)
        out_row.update(
            {
                "scale_anchor_score": max(stable, strict),
                "scale_observability_score": observability,
                "scale_invalid_score": invalid,
                "delay_score": max(0.0, strict * (topk_unstable if math.isfinite(topk_unstable) else 0.0) - 0.25 * (cache if math.isfinite(cache) else 0.0)),
                "weak_context_baseline": f(row.get("semantic_context_mass"), 0.0),
            }
        )
        rows.append(out_row)
    specs = {
        "scale_observability_lower_bad": ("scale_observability_score", False),
        "scale_invalid_higher_bad": ("scale_invalid_score", True),
        "delay_score_higher_bad": ("delay_score", True),
        "weak_context_baseline_higher_bad": ("weak_context_baseline", True),
    }
    cue_rows = evaluate_cues(rows, specs, min_cases=12, require_direction=True, view_name="raw")
    hygiene_cue_rows = evaluate_cues(
        rows,
        specs,
        min_cases=12,
        require_direction=True,
        row_filter=lambda row: b(row.get("good_control_hygiene_include_for_repair", True)),
        view_name="hygiene_repair",
    )
    pass_rows = [row for row in cue_rows if b(row.get("gate_pass")) and row.get("cue_name") != "weak_context_baseline_higher_bad"]
    hygiene_pass_rows = [row for row in hygiene_cue_rows if b(row.get("gate_pass")) and row.get("cue_name") != "weak_context_baseline_higher_bad"]
    seq_coverage = len({row.get("seq", "") for row in rows})
    raw_gate = bool(pass_rows) and seq_coverage >= 3
    hygiene_gate = bool(hygiene_pass_rows) and seq_coverage >= 3
    gate = raw_gate or hygiene_gate
    gate_view = "raw" if raw_gate else ("hygiene_repair" if hygiene_gate else "none")
    best_rows = pass_rows if raw_gate else (hygiene_pass_rows if hygiene_gate else [])
    weak_best = cue_rows and cue_rows[0].get("cue_name") == "weak_context_baseline_higher_bad"
    summary = {
        "schema": "acl2_v98_stage2_trackL_semantic_scale_observability_v1",
        "status": "complete",
        "gate_pass": gate,
        "raw_gate_pass": raw_gate,
        "hygiene_repair_gate_pass": hygiene_gate,
        "gate_pass_view": gate_view,
        "claim_level": "ELIGIBILITY_CUE_FOUND" if gate else "DIAGNOSTIC_CUE_ONLY",
        "available_case_count": len(rows),
        "sequence_coverage": seq_coverage,
        "best_cue": best_rows[0]["cue_name"] if best_rows else (cue_rows[0]["cue_name"] if cue_rows else ""),
        "weak_context_collapse": bool(weak_best),
        "runtime_action_allowed": False,
        "blocker": "" if gate else "Track L did not produce a non-weak-context memory-specific pass.",
    }
    write_rows(out / "observability_rows.csv", rows)
    write_rows(out / "cue_control_metrics.csv", cue_rows)
    write_rows(out / "cue_control_metrics_hygiene_repair.csv", hygiene_cue_rows)
    write_json(out / "summary.json", summary)
    write_text(
        out / "saliency_not_scale_observability.md",
        "# Weak-Context Saliency Check\n\n"
        f"weak_context_collapse={bool(weak_best)}\n\n"
        "Weak-context-only scores are not accepted as SWA scale evidence action candidates.",
    )
    write_failure_common(
        out,
        stage_name="Stage2 TrackL semantic scale observability",
        summary=summary,
        failure_rows=cue_failure_attribution(cue_rows, "stage2_raw") + cue_failure_attribution(hygiene_cue_rows, "stage2_hygiene_repair"),
        visual_rows=[{"artifact": "observability_rows.csv", "path": str(out / "observability_rows.csv"), "status": "generated_table", "source_metric": "scale_observability_score"}],
        what_would_text=(
            "# What Would Have To Be True To Pass\n\n"
            "Track L needs a non-weak-context memory-specific cue with bad_recall >=0.65, good_FPR <=0.25, sequence coverage >=3, "
            "same-count and semantic-rotation margins >=0.05, and correct L3 correlation direction."
        ),
        next_route_text=(
            "# Next Route Recommendation\n\n"
            + (
                "Track L passed; continue to Track M simulator and keep weak-context-only scores out of action candidates."
                if gate
                else "Track L collapsed to weak-context or failed controls; expand scale-anchor taxonomy and do not use weak-context saliency as action evidence."
            )
        ),
    )
    return summary


def stage2b_trace_semantic_anchor_expansion(extension_roots: list[Path]) -> dict[str, Any]:
    out = ROOT / "stage2b_trackL_trace_semantic_anchor_expansion"
    stage1_rows = read_rows(ROOT / "stage1_trackK_swa_v2_strict_stable_eligibility/case_universe_rows.csv")
    rows: list[dict[str, Any]] = []
    label_rows, label_errors = collect_route_label_rows(extension_roots, stage1_rows)
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for label_row in label_rows:
        by_case[str(label_row.get("case_id", ""))].append(label_row)
    for row in stage1_rows:
        if int(float(row.get("payload_count", 0) or 0)) <= 0:
            continue
        case_id = str(row.get("case_id", ""))
        labels = by_case.get(case_id, [])
        total = sum(f(item.get("route_mass_mean"), 0.0) for item in labels)
        if total <= 0.0:
            total = 1.0
        top = max(labels, key=lambda item: f(item.get("route_mass_mean"), 0.0), default={})
        entropy = 0.0
        for item in labels:
            p = max(f(item.get("route_mass_mean"), 0.0), 0.0) / total
            if p > 0.0:
                entropy -= p * math.log(p)
        entropy_norm = entropy / math.log(max(len(labels), 2)) if labels else math.nan
        role_mass: dict[str, float] = defaultdict(float)
        label_mass: dict[int, float] = defaultdict(float)
        for item in labels:
            mass = f(item.get("route_mass_mean"), 0.0)
            role_mass[str(item.get("label_role", "named_other"))] += mass
            label_mass[int(f(item.get("label_id"), -1))] += mass
        static_contract_ids = {1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15}
        static_contract_mass = sum(label_mass[label_id] for label_id in static_contract_ids)
        out_row = dict(row)
        out_row.update(
            {
                "route_label_count": len(labels),
                "route_label_entropy_norm": entropy_norm,
                "route_label_top_mass": f(top.get("route_mass_mean")),
                "route_label_top_id": top.get("label_id", ""),
                "route_label_top_name": top.get("label_name", ""),
                "route_label_top_role": top.get("label_role", ""),
                "route_label_named_structure_mass": role_mass.get("named_structure", 0.0),
                "route_label_named_dynamic_mass": role_mass.get("named_dynamic", 0.0),
                "route_label_named_lowtrust_mass": role_mass.get("named_lowtrust", 0.0),
                "route_label_named_other_mass": role_mass.get("named_other", 0.0),
                "route_label_contract_static_mass": static_contract_mass,
                "route_label_contract_nonstatic_mass": max(0.0, total - static_contract_mass),
                "route_label_building_mass": label_mass.get(15, 0.0),
                "route_label_road_ground_path_mass": label_mass.get(11, 0.0) + label_mass.get(13, 0.0) + label_mass.get(14, 0.0),
                "trace_anchor_mass_balance": f(row.get("unreliable_pair_mass_mean"), 0.0) - f(row.get("stable_pair_mass_mean"), 0.0),
                "trace_stable_pair_density": f(row.get("stable_pair_mass_mean"), 0.0) / max(f(row.get("stable_pair_strict_tokens_mean"), 0.0), 1.0),
                "trace_unreliable_pair_density": f(row.get("unreliable_pair_mass_mean"), 0.0) / max(f(row.get("unreliable_pair_tokens_mean"), 0.0), 1.0),
                "trace_stable_vs_unreliable_random_delta": f(row.get("stable_actual_minus_random_mean"), 0.0) - f(row.get("unreliable_actual_minus_random_mean"), 0.0),
            }
        )
        rows.append(out_row)
    specs = {
        "trace_label_entropy_higher_bad": ("route_label_entropy_norm", True),
        "trace_top_label_mass_lower_bad": ("route_label_top_mass", False),
        "trace_named_structure_mass_lower_bad": ("route_label_named_structure_mass", False),
        "trace_named_dynamic_mass_higher_bad": ("route_label_named_dynamic_mass", True),
        "trace_named_lowtrust_mass_higher_bad": ("route_label_named_lowtrust_mass", True),
        "trace_contract_static_mass_lower_bad": ("route_label_contract_static_mass", False),
        "trace_contract_nonstatic_mass_higher_bad": ("route_label_contract_nonstatic_mass", True),
        "trace_building_mass_lower_bad": ("route_label_building_mass", False),
        "trace_road_ground_path_mass_lower_bad": ("route_label_road_ground_path_mass", False),
        "trace_anchor_mass_balance_higher_bad": ("trace_anchor_mass_balance", True),
        "trace_stable_pair_density_lower_bad": ("trace_stable_pair_density", False),
        "trace_unreliable_pair_density_higher_bad": ("trace_unreliable_pair_density", True),
        "trace_stable_vs_unreliable_random_delta_lower_bad": ("trace_stable_vs_unreliable_random_delta", False),
    }
    cue_rows = evaluate_cues(rows, specs, min_cases=24, require_direction=True, view_name="raw")
    hygiene_cue_rows = evaluate_cues(
        rows,
        specs,
        min_cases=24,
        require_direction=True,
        row_filter=lambda row: b(row.get("good_control_hygiene_include_for_repair", True)),
        view_name="hygiene_repair",
    )
    raw_pass = [row for row in cue_rows if b(row.get("gate_pass"))]
    hygiene_pass = [row for row in hygiene_cue_rows if b(row.get("gate_pass"))]
    raw_gate = bool(raw_pass)
    hygiene_gate = bool(hygiene_pass)
    gate = raw_gate or hygiene_gate
    gate_view = "raw" if raw_gate else ("hygiene_repair" if hygiene_gate else "none")
    best_rows = raw_pass if raw_gate else (hygiene_pass if hygiene_gate else [])
    visible_rows = cue_rows if cue_rows else hygiene_cue_rows
    summary = {
        "schema": "acl2_v98_stage2b_trace_semantic_anchor_expansion_v1",
        "status": "complete",
        "gate_pass": gate,
        "raw_gate_pass": raw_gate,
        "hygiene_repair_gate_pass": hygiene_gate,
        "gate_pass_view": gate_view,
        "claim_level": "ELIGIBILITY_CUE_FOUND" if gate else "DIAGNOSTIC_CUE_ONLY",
        "available_case_count": len(rows),
        "route_label_row_count": len(label_rows),
        "route_label_error_count": len(label_errors),
        "sequence_coverage": len({row.get("seq", "") for row in rows}),
        "best_cue": best_rows[0]["cue_name"] if best_rows else (visible_rows[0]["cue_name"] if visible_rows else ""),
        "runtime_action_allowed": False,
        "blocker": "" if gate else "Trace-semantic anchor expansion did not find a gate-passing non-action cue.",
    }
    write_rows(out / "trace_semantic_anchor_rows.csv", rows)
    write_rows(out / "route_label_rows.csv", label_rows)
    write_rows(out / "route_label_read_errors.csv", label_errors)
    write_rows(out / "cue_control_metrics.csv", cue_rows)
    write_rows(out / "cue_control_metrics_hygiene_repair.csv", hygiene_cue_rows)
    write_json(out / "summary.json", summary)
    failure_rows = cue_failure_attribution(cue_rows, "stage2b_raw") + cue_failure_attribution(hygiene_cue_rows, "stage2b_hygiene_repair")
    write_failure_common(
        out,
        stage_name="Stage2b TrackL trace-semantic anchor expansion",
        summary=summary,
        failure_rows=failure_rows,
        visual_rows=[
            {"artifact": "trace_semantic_anchor_rows.csv", "path": str(out / "trace_semantic_anchor_rows.csv"), "status": "generated_table", "source_metric": "route_label_features"},
            {"artifact": "route_label_rows.csv", "path": str(out / "route_label_rows.csv"), "status": "generated_table", "source_metric": "route_mass_by_prev_fine_label"},
        ],
        what_would_text=(
            "# What Would Have To Be True To Pass\n\n"
            "Trace-semantic anchor expansion would need a route-label or trace-anchor cue with bad_recall >=0.65, good_FPR <=0.25, "
            "abs L3 correlation >=0.40, correct metric direction, sequence/semantic/random margins >=0.05, and sequence coverage >=3."
        ),
        next_route_text=(
            "# Next Route Recommendation\n\n"
            + (
                "A trace-semantic anchor cue passed; run Track M against this cue before considering any E3 action."
                if gate
                else "Trace-semantic anchor expansion still failed; proceed to deeper instrumentation, especially F3 stable-anchor write-to-use identity or an actual SWA before/after scale-evidence hook."
            )
        ),
    )
    return summary


def stage3() -> dict[str, Any]:
    out = ROOT / "stage3_trackM_carrier_to_action_simulator"
    s1 = read_json(ROOT / "stage1_trackK_swa_v2_strict_stable_eligibility/summary.json")
    s2 = read_json(ROOT / "stage2_trackL_semantic_scale_observability/summary.json")
    s2b = read_json(ROOT / "stage2b_trackL_trace_semantic_anchor_expansion/summary.json")
    rows = read_rows(ROOT / "stage1_trackK_swa_v2_strict_stable_eligibility/case_universe_rows.csv")
    if b(s1.get("gate_pass")):
        cue_file = (
            ROOT / "stage1_trackK_swa_v2_strict_stable_eligibility/cue_control_metrics_hygiene_repair.csv"
            if s1.get("gate_pass_view") == "hygiene_repair"
            else ROOT / "stage1_trackK_swa_v2_strict_stable_eligibility/cue_control_metrics.csv"
        )
        cue_source = "stage1_trackK"
        cue_gate_view = s1.get("gate_pass_view", "raw")
    elif b(s2b.get("gate_pass")):
        cue_file = (
            ROOT / "stage2b_trackL_trace_semantic_anchor_expansion/cue_control_metrics_hygiene_repair.csv"
            if s2b.get("gate_pass_view") == "hygiene_repair"
            else ROOT / "stage2b_trackL_trace_semantic_anchor_expansion/cue_control_metrics.csv"
        )
        cue_source = "stage2b_trace_semantic_anchor"
        cue_gate_view = s2b.get("gate_pass_view", "raw")
    else:
        cue_file = (
            ROOT / "stage1_trackK_swa_v2_strict_stable_eligibility/cue_control_metrics_hygiene_repair.csv"
            if s1.get("gate_pass_view") == "hygiene_repair"
            else ROOT / "stage1_trackK_swa_v2_strict_stable_eligibility/cue_control_metrics.csv"
        )
        cue_source = "stage1_trackK_best_diagnostic"
        cue_gate_view = s1.get("gate_pass_view", "none")
    cue_rows = read_rows(cue_file)
    cue = next((row for row in cue_rows if b(row.get("gate_pass"))), cue_rows[0] if cue_rows else {})
    can_simulate = b(s1.get("gate_pass")) or b(s2.get("gate_pass")) or b(s2b.get("gate_pass"))
    selected = set(str(cue.get("true_positive_cases", "")).split(";") + str(cue.get("false_positive_cases", "")).split(";")) - {""}
    sim_rows: list[dict[str, Any]] = []
    for row in rows:
        if int(float(row.get("payload_count", 0) or 0)) <= 0:
            continue
        if cue_gate_view == "hygiene_repair" and not b(row.get("good_control_hygiene_include_for_repair", True)):
            continue
        case_id = str(row.get("case_id", ""))
        strict = f(row.get("strict_stable_nonempty_frac"), 0.0)
        cache = mean([f(row.get("cache_k_stability_mean")), f(row.get("cache_v_stability_mean"))])
        topk_unstable = mean([f(row.get("top1_cache_frame_switch_rate_mean")), f(row.get("top1_cache_index_unique_frac_mean"))])
        risk_mass = f(row.get("unreliable_pair_mass_mean"), 0.0)
        stable_mass = f(row.get("stable_pair_mass_mean"), 0.0)
        is_selected = case_id in selected
        if not is_selected:
            decision = "NO_ACTION"
        elif strict <= 0.0 or risk_mass > 0.55:
            decision = "REJECT"
        elif math.isfinite(cache) and cache < 0.78 and math.isfinite(topk_unstable) and topk_unstable > 0.80:
            decision = "PROTECT_PREV"
        elif math.isfinite(topk_unstable) and topk_unstable > 0.80:
            decision = "DELAY"
        else:
            decision = "TRANSMIT"
        sim_rows.append(
            {
                "case_id": case_id,
                "seq": row.get("seq", ""),
                "case_label": row.get("case_label", ""),
                "gate_view": cue_gate_view,
                "selected_by_cue": is_selected,
                "action_decision": decision,
                "stable_anchor_transport_mass_delta": 0.0 if decision in {"NO_ACTION", "REJECT"} else stable_mass,
                "risk_transport_mass_delta": risk_mass if decision in {"REJECT", "DELAY", "PROTECT_PREV"} else 0.0,
                "topk_frame_switch_rate_delta": -0.10 * topk_unstable if decision in {"DELAY", "PROTECT_PREV"} and math.isfinite(topk_unstable) else 0.0,
                "cache_KV_stability_delta": max(0.0, 0.80 - cache) if decision == "PROTECT_PREV" and math.isfinite(cache) else 0.0,
                "stable_anchor_collapse": decision == "REJECT" and strict > 0.0 and stable_mass > 0.02,
            }
        )
    bad_selected = [row for row in sim_rows if row["case_label"] != "good" and row["selected_by_cue"]]
    good_selected = [row for row in sim_rows if row["case_label"] == "good" and row["selected_by_cue"]]
    target_delta = median([f(row.get("risk_transport_mass_delta")) for row in bad_selected])
    good_harm = mean([f(row.get("risk_transport_mass_delta")) for row in good_selected]) if good_selected else 0.0
    random_margin = f(cue.get("same_count_margin"), 0.0)
    stable_collapse = any(b(row.get("stable_anchor_collapse")) for row in sim_rows)
    gate = can_simulate and target_delta >= 0.05 and random_margin >= 0.05 and good_harm <= 0.02 and not stable_collapse
    summary = {
        "schema": "acl2_v98_stage3_trackM_carrier_to_action_simulator_v1",
        "status": "complete" if can_simulate else "skipped_prerequisite_gate_not_passed",
        "gate_pass": gate,
        "simulator_prerequisite_gate_available": can_simulate,
        "cue_metric_file": str(cue_file),
        "cue_source": cue_source,
        "gate_pass_view": cue_gate_view,
        "best_cue": cue.get("cue_name", ""),
        "trace_target_improvement_proxy": target_delta,
        "actual_vs_random_margin": random_margin,
        "good_simulated_harm": good_harm,
        "stable_anchor_collapse": stable_collapse,
        "runtime_action_allowed": gate,
        "blocker": "" if gate else "Track M did not authorize runtime action; see action_headroom_report.md.",
    }
    write_rows(out / "simulator_rows.csv", sim_rows)
    write_json(out / "summary.json", summary)
    write_text(
        out / "action_headroom_report.md",
        "# Action Headroom Report\n\n"
        f"gate_pass={gate}\n\n"
        f"trace_target_improvement_proxy={target_delta}\n\n"
        f"actual_vs_random_margin={random_margin}\n\n"
        f"good_simulated_harm={good_harm}\n\n"
        "This is a diagnostic simulator.  It authorizes a runtime pilot only if the gate is true; it is not itself an L3 action result.",
    )
    failure_rows = [
        {"stage": "stage3", "failed_check": "simulator_prerequisite_gate_available", "observed_value": can_simulate, "gate_pass": gate},
        {"stage": "stage3", "failed_check": "trace_target_improvement_proxy>=0.05", "observed_value": target_delta, "gate_pass": gate},
        {"stage": "stage3", "failed_check": "actual_vs_random_margin>=0.05", "observed_value": random_margin, "gate_pass": gate},
        {"stage": "stage3", "failed_check": "good_simulated_harm<=0.02", "observed_value": good_harm, "gate_pass": gate},
        {"stage": "stage3", "failed_check": "stable_anchor_collapse_false", "observed_value": stable_collapse, "gate_pass": gate},
    ]
    write_failure_common(
        out,
        stage_name="Stage3 TrackM carrier-to-action simulator",
        summary=summary,
        failure_rows=[row for row in failure_rows if not (row["failed_check"].endswith(">=0.05") and f(row["observed_value"]) >= 0.05) and not (row["failed_check"] == "simulator_prerequisite_gate_available" and row["observed_value"]) and not (row["failed_check"] == "good_simulated_harm<=0.02" and f(row["observed_value"]) <= 0.02) and not (row["failed_check"] == "stable_anchor_collapse_false" and not row["observed_value"])],
        visual_rows=[{"artifact": "simulator_rows.csv", "path": str(out / "simulator_rows.csv"), "status": "generated_table", "source_metric": "risk_transport_mass_delta"}],
        what_would_text=(
            "# What Would Have To Be True To Pass\n\n"
            "Track M can authorize runtime only when a prior K/L gate passes, trace target improvement proxy >=0.05, actual-vs-random margin >=0.05, "
            "good simulated harm <=0.02, and stable anchors do not collapse."
        ),
        next_route_text=(
            "# Next Route Recommendation\n\n"
            + (
                "Track M passed; verify E3 hook availability before any runtime pilot."
                if gate
                else "Track M did not authorize runtime; return to K/L carrier diagnosis or implement a new hook only after the carrier passes."
            )
        ),
    )
    return summary


def downstream_stages() -> dict[str, Any]:
    out = ROOT / "stage4_to_stage10_decision"
    s1 = read_json(ROOT / "stage1_trackK_swa_v2_strict_stable_eligibility/summary.json")
    s2 = read_json(ROOT / "stage2_trackL_semantic_scale_observability/summary.json")
    s2b = read_json(ROOT / "stage2b_trackL_trace_semantic_anchor_expansion/summary.json")
    s3 = read_json(ROOT / "stage3_trackM_carrier_to_action_simulator/summary.json")
    s7b = read_json(ROOT / "stage7b_trackF3_ttt_write_to_swa_usage_proxy/summary.json")
    s7c = read_json(ROOT / "stage7c_ttt_swa_same_run_alignment/summary.json")
    s7d = read_json(ROOT / "stage7d_ttt_swa_spatial_token_proxy_identity/summary.json")
    s7e = read_json(ROOT / "stage7e_ttt_stable_anchor_id_hook/summary.json")
    s7f = read_json(ROOT / "stage7f_prev_ttt_anchor_gate_action_pilot/summary.json")
    s7g = read_json(ROOT / "stage7g_anchor_id_query_head_risk_attribution/summary.json")
    s7h = read_json(ROOT / "stage7h_prev_ttt_anchor_query_soft_action_pilot/summary.json")
    s7h = read_json(ROOT / "stage7h_prev_ttt_anchor_query_soft_action_pilot/summary.json")
    s7h = read_json(ROOT / "stage7h_prev_ttt_anchor_query_soft_action_pilot/summary.json")
    s7h = read_json(ROOT / "stage7h_prev_ttt_anchor_query_soft_action_pilot/summary.json")
    hmc_text = (Path("loger/pipeline/hybrid_memory_controller.py").read_text(encoding="utf-8", errors="replace")
                + Path("run_pipeline_abc_v2.py").read_text(encoding="utf-8", errors="replace"))
    has_e3_hook = "semantic_guided_swa_cache_topk" in hmc_text or "swa_cache_topk" in hmc_text
    e3_allowed = b(s3.get("gate_pass")) and has_e3_hook
    c2 = read_json(V97_ROOT / "trackC_semantic_latent_gauge_ruler/summary.json")
    f2 = read_json(V97_ROOT / "trackF2_ttt_stable_anchor_retention_missing_good_write/summary.json")
    h2 = read_json(V97_ROOT / "trackH2_l07_component_decomposition/summary.json")
    d3 = read_json(V97_ROOT / "trackD3_stage7_end_region_compensator_diagnostic/summary.json")
    branch_rows = [
        {"stage": "Stage4_E3", "status": "blocked_missing_e3_hook" if b(s3.get("gate_pass")) and not has_e3_hook else ("not_allowed_by_stageM" if not b(s3.get("gate_pass")) else "allowed_not_run_by_builder"), "gate_pass": False, "runtime_action_allowed": e3_allowed},
        {"stage": "Stage5_C2_v2", "status": "not_run_no_E3_candidate; v97 C2 failed controls", "gate_pass": False, "source_gate_pass": b(c2.get("gate_pass"))},
        {"stage": "Stage6_H2_D3", "status": "H2 local component retained; pairwise READ trace/action not opened by main SWA gate", "gate_pass": False, "h2_gate_pass": b(h2.get("gate_pass")), "d3_gate_pass": b(d3.get("gate_pass"))},
        {"stage": "Stage7_F3", "status": "not_opened; F2 exact retention/write-map failed and no write-to-use chain was traced", "gate_pass": False, "f2_gate_pass": b(f2.get("gate_pass"))},
        {"stage": "Stage7b_F3_proxy", "status": "diagnostic_proxy_only; true stable-anchor write-to-use identity unavailable", "gate_pass": False, "proxy_case_count": s7b.get("proxy_case_count", ""), "sequence_coverage": s7b.get("sequence_coverage", ""), "true_anchor_identity_available": b(s7b.get("true_anchor_identity_available"))},
        {"stage": "Stage7c_same_run", "status": "same-run TTT/SWA traces available but no persistent identity and strict-stable SWA pairs fallback", "gate_pass": False, "case_with_both_count": s7c.get("case_with_both_count", ""), "swa_strict_stable_nonempty_case_count": s7c.get("swa_strict_stable_nonempty_case_count", ""), "true_anchor_identity_available": b(s7c.get("true_anchor_identity_available"))},
        {"stage": "Stage7d_spatial_token_proxy", "status": "spatial-token proxy only; no persistent stable-anchor id or promotable F3 chain", "gate_pass": False, "case_count": s7d.get("case_count", ""), "prev_topk_hit_frac_mean": s7d.get("prev_topk_hit_frac_mean", ""), "true_anchor_identity_available": b(s7d.get("true_anchor_identity_available"))},
        {"stage": "Stage7e_anchor_id_hook", "status": "state-carried anchor ids observable but F3 failure/action gate still not passed", "gate_pass": False, "case_count": s7e.get("case_count", ""), "identity_case_count": s7e.get("identity_case_count", ""), "write_to_swa_topk_chain_case_count": s7e.get("write_to_swa_topk_chain_case_count", ""), "persistent_anchor_id_available": b(s7e.get("persistent_anchor_id_available"))},
        {"stage": "Stage7f_prev_ttt_anchor_gate_action_pilot", "status": "action_pilot_no_go" if b(s7f.get("runtime_action_pilot_run")) and not b(s7f.get("gate_pass")) else ("not_run" if not b(s7f.get("runtime_action_pilot_run")) else "action_pilot_gate_pass"), "gate_pass": b(s7f.get("gate_pass")), "runtime_action_pilot_run": b(s7f.get("runtime_action_pilot_run")), "best_variant": s7f.get("best_variant", ""), "best_variant_improved_ate_case_count": s7f.get("best_variant_improved_ate_case_count", ""), "best_variant_worse_ate_case_count": s7f.get("best_variant_worse_ate_case_count", ""), "best_variant_median_improvement_ratio_vs_baseline": s7f.get("best_variant_median_improvement_ratio_vs_baseline", "")},
        {"stage": "Stage7g_query_head_anchor_id_attribution", "status": "selective_query_head_cue_found_action_not_run" if b(s7g.get("gate_pass")) else ("not_run" if not s7g else "selective_attribution_no_go"), "gate_pass": b(s7g.get("gate_pass")), "runtime_action_allowed": b(s7g.get("runtime_action_allowed")), "best_cue": s7g.get("best_cue", ""), "selected_case_candidate_action_query_ge75_frac_mean": s7g.get("selected_case_candidate_action_query_ge75_frac_mean", ""), "selected_case_candidate_action_query_ge75_frac_median": s7g.get("selected_case_candidate_action_query_ge75_frac_median", "")},
        {"stage": "Stage7h_prev_ttt_anchor_query_soft_action_pilot", "status": "query_soft_action_pilot_no_go" if b(s7h.get("runtime_action_pilot_run")) and not b(s7h.get("gate_pass")) else ("not_run" if not b(s7h.get("runtime_action_pilot_run")) else "query_soft_action_pilot_gate_pass"), "gate_pass": b(s7h.get("gate_pass")), "runtime_action_pilot_run": b(s7h.get("runtime_action_pilot_run")), "variant": s7h.get("variant", ""), "improved_ate_case_count": s7h.get("improved_ate_case_count", ""), "worse_ate_case_count": s7h.get("worse_ate_case_count", ""), "median_improvement_ratio_vs_baseline": s7h.get("median_improvement_ratio_vs_baseline", "")},
        {"stage": "Stage8_J", "status": "diagnostic atlas only; no causal memory-specific region effect promoted", "gate_pass": False},
        {"stage": "Stage9_full_validation", "status": "blocked_by_prerequisite_gates", "gate_pass": False, "full_validation_allowed": False},
    ]
    final_taxonomy = (
        "L3_HANDOFF_MECHANISM_PASS_FULL_PENDING" if e3_allowed else
        "ELIGIBILITY_CUE_PASS_ACTION_NOT_RUN" if b(s1.get("gate_pass")) or b(s2.get("gate_pass")) or b(s2b.get("gate_pass")) else
        "F3_IDENTITY_DIAGNOSTIC_CUE_PASS_QUERY_SOFT_ACTION_PILOT_NO_GO" if b(s7h.get("runtime_action_pilot_run")) and not b(s7h.get("gate_pass")) else
        "F3_IDENTITY_DIAGNOSTIC_CUE_PASS_QUERY_SOFT_ACTION_PILOT_PASS_FULL_PENDING" if b(s7h.get("gate_pass")) else
        "F3_IDENTITY_DIAGNOSTIC_CUE_PASS_ACTION_PILOT_NO_GO_SELECTIVE_QUERY_HEAD_CUE_FOUND" if b(s7f.get("runtime_action_pilot_run")) and not b(s7f.get("gate_pass")) and b(s7g.get("gate_pass")) else
        "F3_IDENTITY_DIAGNOSTIC_CUE_PASS_ACTION_PILOT_NO_GO_SELECTIVE_ATTRIBUTION_NO_GO" if b(s7f.get("runtime_action_pilot_run")) and not b(s7f.get("gate_pass")) and s7g else
        "F3_IDENTITY_DIAGNOSTIC_CUE_PASS_ACTION_PILOT_NO_GO" if b(s7f.get("runtime_action_pilot_run")) and not b(s7f.get("gate_pass")) else
        "F3_IDENTITY_DIAGNOSTIC_CUE_PASS_ACTION_PILOT_PASS_FULL_PENDING" if b(s7f.get("gate_pass")) else
        "F3_IDENTITY_DIAGNOSTIC_CUE_PASS_ACTION_NOT_RUN" if b(s7e.get("gate_pass")) else
        "PARSER_OR_TRACE_BLOCKED" if int(s1.get("traced_case_count", 0) or 0) < 24 else
        "NO_SIGNAL"
    )
    summary = {
        "schema": "acl2_v98_stage4_to_stage10_decision_v1",
        "status": "complete_blocked_or_no_go",
        "full_method_success": False,
        "method_success": False,
        "runtime_action_run": False,
        "runtime_action_pilot_run": b(s7f.get("runtime_action_pilot_run")) or b(s7h.get("runtime_action_pilot_run")),
        "full_validation_run": False,
        "final_taxonomy": final_taxonomy,
        "stage1_gate_pass": b(s1.get("gate_pass")),
        "stage2_gate_pass": b(s2.get("gate_pass")),
        "stage2b_trace_semantic_anchor_gate_pass": b(s2b.get("gate_pass")),
        "stage3_gate_pass": b(s3.get("gate_pass")),
        "stage7b_ttt_write_to_swa_usage_proxy_gate_pass": b(s7b.get("gate_pass")),
        "stage7b_true_anchor_identity_available": b(s7b.get("true_anchor_identity_available")),
        "stage7b_proxy_case_count": s7b.get("proxy_case_count", ""),
        "stage7c_ttt_swa_same_run_gate_pass": b(s7c.get("gate_pass")),
        "stage7c_case_with_both_count": s7c.get("case_with_both_count", ""),
        "stage7c_swa_strict_stable_nonempty_case_count": s7c.get("swa_strict_stable_nonempty_case_count", ""),
        "stage7d_ttt_swa_spatial_token_proxy_gate_pass": b(s7d.get("gate_pass")),
        "stage7d_spatial_token_proxy_available": b(s7d.get("spatial_token_proxy_available")),
        "stage7d_case_count": s7d.get("case_count", ""),
        "stage7d_prev_topk_hit_frac_mean": s7d.get("prev_topk_hit_frac_mean", ""),
        "stage7e_ttt_stable_anchor_id_hook_gate_pass": b(s7e.get("gate_pass")),
        "stage7e_f3_write_to_use_risk_cue_shown": b(s7e.get("f3_write_to_use_risk_cue_shown")),
        "stage7e_persistent_anchor_id_available": b(s7e.get("persistent_anchor_id_available")),
        "stage7e_write_to_use_chain_available": b(s7e.get("write_to_use_chain_available")),
        "stage7e_case_count": s7e.get("case_count", ""),
        "stage7e_anchor_id_topk_hit_frac_mean": s7e.get("anchor_id_topk_hit_frac_mean", ""),
        "stage7f_prev_ttt_anchor_gate_action_pilot_run": b(s7f.get("runtime_action_pilot_run")),
        "stage7f_prev_ttt_anchor_gate_action_pilot_gate_pass": b(s7f.get("gate_pass")),
        "stage7f_best_variant": s7f.get("best_variant", ""),
        "stage7f_best_variant_improved_ate_case_count": s7f.get("best_variant_improved_ate_case_count", ""),
        "stage7f_best_variant_worse_ate_case_count": s7f.get("best_variant_worse_ate_case_count", ""),
        "stage7f_best_variant_median_improvement_ratio_vs_baseline": s7f.get("best_variant_median_improvement_ratio_vs_baseline", ""),
        "stage7g_anchor_id_query_head_risk_attribution_gate_pass": b(s7g.get("gate_pass")),
        "stage7g_query_head_gate_pass": b(s7g.get("query_head_gate_pass")),
        "stage7g_id_specific_risk_cue_gate_pass": b(s7g.get("id_specific_risk_cue_gate_pass")),
        "stage7g_selective_action_mass_gate_pass": b(s7g.get("selective_action_mass_gate_pass")),
        "stage7g_best_cue": s7g.get("best_cue", ""),
        "stage7g_selected_case_candidate_action_query_ge75_frac_mean": s7g.get("selected_case_candidate_action_query_ge75_frac_mean", ""),
        "stage7g_selected_case_candidate_action_query_ge75_frac_median": s7g.get("selected_case_candidate_action_query_ge75_frac_median", ""),
        "stage7h_prev_ttt_anchor_query_soft_action_pilot_run": b(s7h.get("runtime_action_pilot_run")),
        "stage7h_prev_ttt_anchor_query_soft_action_pilot_gate_pass": b(s7h.get("gate_pass")),
        "stage7h_variant": s7h.get("variant", ""),
        "stage7h_improved_ate_case_count": s7h.get("improved_ate_case_count", ""),
        "stage7h_worse_ate_case_count": s7h.get("worse_ate_case_count", ""),
        "stage7h_median_improvement_ratio_vs_baseline": s7h.get("median_improvement_ratio_vs_baseline", ""),
        "e3_hook_available": has_e3_hook,
        "e3_runtime_allowed": e3_allowed,
        "primary_blocker": "" if e3_allowed else (
            s7h.get("primary_blocker")
            if b(s7h.get("runtime_action_pilot_run")) and not b(s7h.get("gate_pass"))
            else s7g.get("primary_blocker")
            if b(s7f.get("runtime_action_pilot_run")) and not b(s7f.get("gate_pass")) and s7g
            else s7f.get("primary_blocker")
            if b(s7f.get("runtime_action_pilot_run")) and not b(s7f.get("gate_pass"))
            else "E3 runtime action not executed because prerequisite gates or implementation availability did not authorize it."
        ),
    }
    write_rows(out / "branch_decision_rows.csv", branch_rows)
    write_json(out / "summary.json", summary)
    write_text(
        out / "full_validation_blocker.md",
        "# Full Validation Blocker\n\n"
        f"full_validation_allowed=False\n\nfinal_taxonomy={final_taxonomy}\n\n"
        "Stage9 full validation requires E3 L3 handoff action pass, or D3 READ pairwise action pass plus C2 gauge-safety predictor pass. "
        "Those prerequisites are not satisfied in this builder run.",
    )
    write_text(
        out / "forbidden_repeat_update.md",
        "# Forbidden Repeat Update\n\n"
        "Do not use old Track E source gate/source replace/merge alpha as a substitute for E3 cache/top-k transmit/delay/reject/protect_prev.",
    )
    write_failure_common(
        out,
        stage_name="Stage4-10 downstream decision",
        summary=summary,
        failure_rows=[
            {"stage": row["stage"], "failed_check": row["status"], "observed_value": row.get("gate_pass"), "gate_pass": row.get("gate_pass")}
            for row in branch_rows
            if not b(row.get("gate_pass"))
        ],
        visual_rows=[{"artifact": "branch_decision_rows.csv", "path": str(out / "branch_decision_rows.csv"), "status": "generated_table", "source_metric": "branch_gate_status"}],
        what_would_text=(
            "# What Would Have To Be True To Pass\n\n"
            "Downstream runtime/full validation needs E3 L3 handoff action pass, or D3 READ pairwise action pass plus C2 gauge-safety predictor pass. "
            "Diagnostic replay alone is insufficient."
        ),
        next_route_text=(
            "# Next Route Recommendation\n\n"
            + (
                "E3 is allowed; implement/verify the hook before any full validation."
                if e3_allowed
                else "Do not run full validation.  The next productive route is deeper carrier/hook instrumentation, not old Track E or chunk selector repeats."
            )
        ),
    )
    return summary


def stage5_c2_diagnostic_replay_ledger() -> dict[str, Any]:
    out = ROOT / "stage5_trackC2_diagnostic_full_sequence_latent_replay"
    c2 = read_json(V97_ROOT / "trackC_semantic_latent_gauge_ruler/summary.json")
    summary = {
        "schema": "acl2_v98_stage5_c2_diagnostic_replay_ledger_v1",
        "status": "complete_existing_v97_diagnostic_reused",
        "diagnostic_only": True,
        "gate_pass": False,
        "runtime_action_allowed": False,
        "v97_c2_gate_pass": b(c2.get("gate_pass")),
        "read_latent_residual_available": b(c2.get("read_latent_residual_available")),
        "stage7_full_latent_dump_available_any": b(c2.get("stage7_full_latent_dump_available_any")),
        "downstream_control_comparable_or_stronger": b(c2.get("downstream_control_comparable_or_stronger")),
        "classification": c2.get("classification", ""),
        "primary_blocker": "C2 remains diagnostic/control-comparable and cannot authorize runtime or full validation without E3/D3 mechanism pass.",
    }
    write_json(out / "summary.json", summary)
    write_text(
        out / "diagnostic_replay_report.md",
        "# Stage5 C2 Diagnostic Replay Ledger\n\n"
        f"v97_c2_gate_pass={summary['v97_c2_gate_pass']}\n\n"
        f"read_latent_residual_available={summary['read_latent_residual_available']}\n\n"
        f"stage7_full_latent_dump_available_any={summary['stage7_full_latent_dump_available_any']}\n\n"
        f"classification={summary['classification']}\n\n"
        "No v98 full-sequence latent replay is promoted as validation.",
    )
    write_failure_common(
        out,
        stage_name="Stage5 C2 diagnostic replay",
        summary=summary,
        failure_rows=[
            {"stage": "stage5_c2", "failed_check": "v97_c2_gate_pass", "observed_value": summary["v97_c2_gate_pass"], "gate_pass": False},
            {"stage": "stage5_c2", "failed_check": "stage7_full_latent_dump_available_any", "observed_value": summary["stage7_full_latent_dump_available_any"], "gate_pass": False},
            {"stage": "stage5_c2", "failed_check": "controls_not_comparable", "observed_value": not summary["downstream_control_comparable_or_stronger"], "gate_pass": False},
        ],
        visual_rows=[{"artifact": "v97_trackC_visual_manifest", "path": str(V97_ROOT / "trackC_semantic_latent_gauge_ruler/read_latent_dump_smoke_h2_anchorcomp_t050_all8/visual_manifest.csv"), "status": "existing_artifact" if (V97_ROOT / "trackC_semantic_latent_gauge_ruler/read_latent_dump_smoke_h2_anchorcomp_t050_all8/visual_manifest.csv").is_file() else "missing", "source_metric": "latent_residual"}],
        what_would_text=(
            "# What Would Have To Be True To Pass\n\n"
            "C2-v2 would need full latent dumps for baseline and candidate, stable-anchor residual abs corr >=0.50 with final/yaw/scale shift, "
            "correlation stronger than controls by >=0.05, visible active/inactive tradeoff, and no chunk-selector dependence."
        ),
        next_route_text="# Next Route Recommendation\n\nDo not use C2 as a runtime action.  Revisit only after E3 or D3 produces a real mechanism candidate.",
    )
    return summary


def stage6_h2_d3_read_ledger() -> dict[str, Any]:
    out = ROOT / "stage6_trackH2_D3_read_component_pairwise"
    h2 = read_json(V97_ROOT / "trackH2_l07_component_decomposition/summary.json")
    d3 = read_json(V97_ROOT / "trackD3_stage7_end_region_compensator_diagnostic/summary.json")
    summary = {
        "schema": "acl2_v98_stage6_h2_d3_read_ledger_v1",
        "status": "complete_existing_v97_diagnostic_reused",
        "gate_pass": False,
        "runtime_action_allowed": False,
        "h2_local_component_gate_pass": b(h2.get("gate_pass")),
        "h2_best_candidate": (h2.get("best_passing_component") or {}).get("candidate", ""),
        "h2_bad_L2_improvement": (h2.get("best_passing_component") or {}).get("bad_L2_improvement", math.nan),
        "d3_gate_pass": b(d3.get("gate_pass")),
        "d3_classification": d3.get("classification", ""),
        "read_pairwise_action_run": False,
        "primary_blocker": "H2 is local-only and D3 remains diagnostic/no-action; no pairwise READ source-target action gate is available.",
    }
    write_json(out / "summary.json", summary)
    write_rows(
        out / "read_component_status_rows.csv",
        [
            {"branch": "H2_component", "gate_pass": summary["h2_local_component_gate_pass"], "runtime_action_allowed": False, "evidence": str(V97_ROOT / "trackH2_l07_component_decomposition/summary.json")},
            {"branch": "D3_end_region", "gate_pass": summary["d3_gate_pass"], "runtime_action_allowed": False, "evidence": str(V97_ROOT / "trackD3_stage7_end_region_compensator_diagnostic/summary.json")},
            {"branch": "D3_pairwise_action", "gate_pass": False, "runtime_action_allowed": False, "evidence": "not opened by v98 K/L/M gates"},
        ],
    )
    write_failure_common(
        out,
        stage_name="Stage6 H2/D3 READ component and pairwise action",
        summary=summary,
        failure_rows=[
            {"stage": "stage6_h2_d3", "failed_check": "d3_pairwise_action_gate_available", "observed_value": False, "gate_pass": False},
            {"stage": "stage6_h2_d3", "failed_check": "d3_gate_pass", "observed_value": summary["d3_gate_pass"], "gate_pass": False},
        ],
        visual_rows=[{"artifact": "read_component_status_rows.csv", "path": str(out / "read_component_status_rows.csv"), "status": "generated_table", "source_metric": "component_gate_status"}],
        what_would_text=(
            "# What Would Have To Be True To Pass\n\n"
            "Stage6 would need sampled query-source pairwise READ traces, risk-to-weak/dynamic mass separation, stable-to-stable preservation, good controls not flagged, sequence coverage >=3, and control margins >=0.05."
        ),
        next_route_text="# Next Route Recommendation\n\nKeep H2 as local mechanism evidence only; do not rerun dense L07 without a new pairwise source-target gate.",
    )
    return summary


def stage7_f3_ttt_ledger() -> dict[str, Any]:
    out = ROOT / "stage7_trackF3_ttt_write_to_use"
    f2 = read_json(V97_ROOT / "trackF2_ttt_stable_anchor_retention_missing_good_write/summary.json")
    retention_rows = read_rows(V97_ROOT / "trackF2_ttt_stable_anchor_retention_missing_good_write/exact_retention_rows.csv")
    write_rows_in = read_rows(V97_ROOT / "trackF2_ttt_stable_anchor_retention_missing_good_write/exact_write_map_audit_rows.csv")
    summary = {
        "schema": "acl2_v98_stage7_f3_ttt_write_to_use_ledger_v1",
        "status": "complete_existing_v97_f2_diagnostic_checked",
        "gate_pass": False,
        "runtime_action_allowed": False,
        "v97_f2_gate_pass": b(f2.get("gate_pass")),
        "exact_retention_row_count": len(retention_rows),
        "exact_write_map_row_count": len(write_rows_in),
        "stable_anchor_retention_available": b(f2.get("stable_anchor_retention_available")),
        "write_to_use_chain_available": False,
        "f3_write_to_use_failure_shown": False,
        "primary_blocker": "Existing artifacts expose TTT retention/write-map diagnostics but no stable-anchor write-to-READ/SWA-use identity chain.",
    }
    write_json(out / "summary.json", summary)
    write_rows(
        out / "write_to_use_availability_rows.csv",
        [
            {"artifact": "exact_retention_rows.csv", "available": bool(retention_rows), "row_count": len(retention_rows), "supports_f3_write_to_use": False},
            {"artifact": "exact_write_map_audit_rows.csv", "available": bool(write_rows_in), "row_count": len(write_rows_in), "supports_f3_write_to_use": False},
            {"artifact": "stable_anchor_write_to_read_usage", "available": False, "row_count": 0, "supports_f3_write_to_use": True},
            {"artifact": "stable_anchor_write_to_swa_cache_usage", "available": False, "row_count": 0, "supports_f3_write_to_use": True},
        ],
    )
    write_failure_common(
        out,
        stage_name="Stage7 F3 TTT write-to-use",
        summary=summary,
        failure_rows=[
            {"stage": "stage7_f3", "failed_check": "write_to_use_chain_available", "observed_value": False, "gate_pass": False},
            {"stage": "stage7_f3", "failed_check": "v97_f2_gate_pass", "observed_value": summary["v97_f2_gate_pass"], "gate_pass": False},
        ],
        visual_rows=[{"artifact": "write_to_use_availability_rows.csv", "path": str(out / "write_to_use_availability_rows.csv"), "status": "generated_table", "source_metric": "availability"}],
        what_would_text=(
            "# What Would Have To Be True To Pass\n\n"
            "F3 would need stable anchor ids written at chunk c, later READ/SWA usage at c+1/c+2/c+3, future top-k identity stability, and correlation with future L3/L4 drift not explained by write energy alone."
        ),
        next_route_text="# Next Route Recommendation\n\nPause TTT action.  Add stable-anchor identity propagation instrumentation before any F3 action or no-write claim.",
    )
    return summary


def _parse_case_chunks(case_id: str) -> tuple[int, int] | None:
    parts = str(case_id).split("_")
    if len(parts) < 3:
        return None
    try:
        return int(parts[-2]), int(parts[-1])
    except ValueError:
        return None


def stage7b_ttt_write_to_swa_usage_proxy() -> dict[str, Any]:
    out = ROOT / "stage7b_trackF3_ttt_write_to_swa_usage_proxy"
    stage1_rows = read_rows(ROOT / "stage1_trackK_swa_v2_strict_stable_eligibility/case_universe_rows.csv")
    stage2b_rows = read_rows(ROOT / "stage2b_trackL_trace_semantic_anchor_expansion/trace_semantic_anchor_rows.csv")
    meta_by_case = {str(row.get("case_id", "")): row for row in stage1_rows if row.get("case_id")}
    stage2b_by_case = {str(row.get("case_id", "")): row for row in stage2b_rows if row.get("case_id")}
    exact_rows = read_rows(V97_ROOT / "trackF2_ttt_stable_anchor_retention_missing_good_write/exact_retention_rows.csv")
    map_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []

    def tensor_mean(tensor: Any, mask: Any | None = None, invert_mask: bool = False) -> float:
        try:
            import torch

            if not torch.is_tensor(tensor):
                return math.nan
            x = tensor.detach().cpu().float()
            if mask is not None and torch.is_tensor(mask):
                m = mask.detach().cpu().bool()
                if tuple(m.shape) != tuple(x.shape):
                    return math.nan
                if invert_mask:
                    m = ~m
                x = x[m]
            x = x[torch.isfinite(x)]
            return float(x.mean().item()) if int(x.numel()) > 0 else math.nan
        except Exception:  # noqa: BLE001
            return math.nan

    def tensor_mass(mask: Any) -> float:
        try:
            import torch

            if not torch.is_tensor(mask):
                return math.nan
            m = mask.detach().cpu().bool()
            return float(m.float().mean().item()) if int(m.numel()) > 0 else math.nan
        except Exception:  # noqa: BLE001
            return math.nan

    for row in exact_rows:
        case_id = str(row.get("case_id", ""))
        path = Path(str(row.get("path", "")))
        if not path.is_file():
            error_rows.append({"case_id": case_id, "path": str(path), "error": "missing_ttt_spatial_map"})
            continue
        try:
            import torch

            payload = torch.load(path, map_location="cpu")
        except Exception as exc:  # noqa: BLE001
            error_rows.append({"case_id": case_id, "path": str(path), "error": f"{type(exc).__name__}:{exc}"})
            continue
        mask = payload.get("stable_anchor_mask_patch")
        prior = payload.get("ttt_write_prior_patch")
        energy = payload.get("U_ttt_write_replay_contribution_patch")
        retention = payload.get("stable_anchor_retention_patch")
        residual = payload.get("stable_anchor_residual_patch")
        chunk_idx = int(f(row.get("chunk_idx"), -1))
        chunks = _parse_case_chunks(case_id)
        chunk_role = "unknown"
        if chunks is not None:
            prev_chunk, curr_chunk = chunks
            if chunk_idx == prev_chunk:
                chunk_role = "prev_chunk"
            elif chunk_idx == curr_chunk:
                chunk_role = "curr_chunk"
            else:
                chunk_role = "other_chunk"
        stable_prior = tensor_mean(prior, mask)
        nonstable_prior = tensor_mean(prior, mask, invert_mask=True)
        stable_energy = tensor_mean(energy, mask)
        nonstable_energy = tensor_mean(energy, mask, invert_mask=True)
        map_rows.append(
            {
                "case_id": case_id,
                "seq": case_id.split("_")[0] if "_" in case_id else "",
                "chunk_idx": chunk_idx,
                "chunk_role": chunk_role,
                "path": str(path),
                "schema": payload.get("schema", ""),
                "spatial_token_aligned": payload.get("spatial_token_aligned", ""),
                "projection_not_raw_per_token_fast_weight_delta": payload.get("projection_not_raw_per_token_fast_weight_delta", ""),
                "stable_anchor_mask_frac": tensor_mass(mask),
                "stable_anchor_token_count": row.get("stable_anchor_token_count", ""),
                "stable_anchor_write_prior_mean": stable_prior,
                "nonstable_write_prior_mean": nonstable_prior,
                "stable_prior_minus_nonstable": stable_prior - nonstable_prior if math.isfinite(stable_prior) and math.isfinite(nonstable_prior) else math.nan,
                "stable_anchor_write_energy_mean": stable_energy,
                "nonstable_write_energy_mean": nonstable_energy,
                "stable_energy_minus_nonstable": stable_energy - nonstable_energy if math.isfinite(stable_energy) and math.isfinite(nonstable_energy) else math.nan,
                "stable_anchor_retention_mean": tensor_mean(retention, mask),
                "stable_anchor_residual_mean": tensor_mean(residual, mask),
                "payload_source": "v97_trackF2_exact_retention_probe_v1",
                "true_anchor_identity_available": False,
                "later_read_or_swa_identity_available": False,
            }
        )

    map_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in map_rows:
        map_by_case[str(row.get("case_id", ""))].append(row)

    proxy_rows: list[dict[str, Any]] = []
    for case_id, rows_for_case in sorted(map_by_case.items()):
        meta = meta_by_case.get(case_id, {})
        if not meta:
            continue
        s2b = stage2b_by_case.get(case_id, {})
        stable_prior_mean = mean([f(row.get("stable_anchor_write_prior_mean")) for row in rows_for_case])
        stable_energy_mean = mean([f(row.get("stable_anchor_write_energy_mean")) for row in rows_for_case])
        retention_mean = mean([f(row.get("stable_anchor_retention_mean")) for row in rows_for_case])
        residual_mean = mean([f(row.get("stable_anchor_residual_mean")) for row in rows_for_case])
        mask_frac_mean = mean([f(row.get("stable_anchor_mask_frac")) for row in rows_for_case])
        prior_minus_nonstable = mean([f(row.get("stable_prior_minus_nonstable")) for row in rows_for_case])
        energy_minus_nonstable = mean([f(row.get("stable_energy_minus_nonstable")) for row in rows_for_case])
        topk_same = f(meta.get("topk_same_frame_frac_mean"))
        topk_query_hit = f(meta.get("topk_query_frame_hit_frac_mean"))
        top1_same = f(meta.get("top1_same_frame_frac_mean"))
        top1_frame_switch = f(meta.get("top1_cache_frame_switch_rate_mean"))
        top1_index_unique = f(meta.get("top1_cache_index_unique_frac_mean"))
        stable_pair_mass = f(meta.get("stable_pair_mass_mean"))
        unreliable_pair_mass = f(meta.get("unreliable_pair_mass_mean"))
        route_dynamic_mass = f(s2b.get("route_label_named_dynamic_mass"))
        route_structure_mass = f(s2b.get("route_label_named_structure_mass"))
        proxy_rows.append(
            {
                "case_id": case_id,
                "seq": meta.get("seq", case_id.split("_")[0]),
                "case_label": meta.get("case_label", ""),
                "good_control_hygiene_include_for_repair": meta.get("good_control_hygiene_include_for_repair", True),
                "payload_count": 1,
                "ttt_map_chunk_count": len(rows_for_case),
                "L3_handoff_transfer_penalty_proxy": f(meta.get("L3_handoff_transfer_penalty_proxy")),
                "stable_anchor_mask_frac_mean": mask_frac_mean,
                "stable_anchor_write_prior_mean": stable_prior_mean,
                "stable_prior_minus_nonstable_mean": prior_minus_nonstable,
                "stable_anchor_write_energy_mean": stable_energy_mean,
                "stable_energy_minus_nonstable_mean": energy_minus_nonstable,
                "stable_anchor_retention_mean": retention_mean,
                "stable_anchor_residual_mean": residual_mean,
                "swa_stable_pair_mass_mean": stable_pair_mass,
                "swa_unreliable_pair_mass_mean": unreliable_pair_mass,
                "swa_topk_same_frame_frac_mean": topk_same,
                "swa_topk_query_frame_hit_frac_mean": topk_query_hit,
                "swa_top1_same_frame_frac_mean": top1_same,
                "swa_top1_frame_switch_rate_mean": top1_frame_switch,
                "swa_top1_index_unique_frac_mean": top1_index_unique,
                "route_label_named_dynamic_mass": route_dynamic_mass,
                "route_label_named_structure_mass": route_structure_mass,
                "proxy_stable_write_not_topk_used": stable_prior_mean * (1.0 - topk_same) if math.isfinite(stable_prior_mean) and math.isfinite(topk_same) else math.nan,
                "proxy_energy_not_stable_swa_used": stable_energy_mean * (1.0 - stable_pair_mass) if math.isfinite(stable_energy_mean) and math.isfinite(stable_pair_mass) else math.nan,
                "proxy_retention_but_topk_unstable": retention_mean * top1_frame_switch if math.isfinite(retention_mean) and math.isfinite(top1_frame_switch) else math.nan,
                "proxy_write_energy_x_dynamic_route": stable_energy_mean * route_dynamic_mass if math.isfinite(stable_energy_mean) and math.isfinite(route_dynamic_mass) else math.nan,
                "proxy_write_prior_x_low_structure_route": stable_prior_mean * (1.0 - route_structure_mass) if math.isfinite(stable_prior_mean) and math.isfinite(route_structure_mass) else math.nan,
                "proxy_usage_support": stable_energy_mean * topk_same if math.isfinite(stable_energy_mean) and math.isfinite(topk_same) else math.nan,
                "true_anchor_identity_available": False,
                "later_read_or_swa_identity_available": False,
                "claim_scope": "proxy_semantic_anchor_topk_identity_only",
            }
        )

    specs = {
        "proxy_stable_write_not_topk_used_higher_bad": ("proxy_stable_write_not_topk_used", True),
        "proxy_energy_not_stable_swa_used_higher_bad": ("proxy_energy_not_stable_swa_used", True),
        "proxy_retention_but_topk_unstable_higher_bad": ("proxy_retention_but_topk_unstable", True),
        "proxy_write_energy_x_dynamic_route_higher_bad": ("proxy_write_energy_x_dynamic_route", True),
        "proxy_write_prior_x_low_structure_route_higher_bad": ("proxy_write_prior_x_low_structure_route", True),
        "proxy_usage_support_lower_bad": ("proxy_usage_support", False),
        "stable_anchor_mask_frac_higher_bad": ("stable_anchor_mask_frac_mean", True),
        "stable_energy_minus_nonstable_higher_bad": ("stable_energy_minus_nonstable_mean", True),
    }
    cue_rows = evaluate_cues(proxy_rows, specs, min_cases=12, require_direction=True, view_name="proxy")
    hygiene_cue_rows = evaluate_cues(
        proxy_rows,
        specs,
        min_cases=12,
        require_direction=True,
        row_filter=lambda row: b(row.get("good_control_hygiene_include_for_repair", True)),
        view_name="proxy_hygiene_repair",
    )
    sequence_coverage = len({str(row.get("seq", "")) for row in proxy_rows})
    true_identity_available = False
    diagnostic_proxy_metric_pass = any(b(row.get("gate_pass")) for row in cue_rows + hygiene_cue_rows)
    gate = False
    best_pool = cue_rows if cue_rows else hygiene_cue_rows
    summary = {
        "schema": "acl2_v98_stage7b_ttt_write_to_swa_usage_proxy_v1",
        "status": "complete_proxy_only",
        "gate_pass": gate,
        "runtime_action_allowed": False,
        "claim_level": "DIAGNOSTIC_CUE_ONLY",
        "true_anchor_identity_available": true_identity_available,
        "later_read_or_swa_identity_available": False,
        "write_to_use_chain_available": False,
        "f3_write_to_use_failure_shown": False,
        "proxy_case_count": len(proxy_rows),
        "ttt_map_row_count": len(map_rows),
        "ttt_map_read_error_count": len(error_rows),
        "sequence_coverage": sequence_coverage,
        "proxy_min_cases_required": 12,
        "diagnostic_proxy_metric_gate_pass": diagnostic_proxy_metric_pass,
        "best_cue": best_pool[0]["cue_name"] if best_pool else "",
        "primary_blocker": "Only proxy semantic-anchor/top-k alignment is available; true stable-anchor write-to-READ/SWA-use identity chain is not traced, and proxy coverage is underpowered.",
    }
    write_rows(out / "ttt_spatial_map_proxy_rows.csv", map_rows)
    write_rows(out / "ttt_spatial_map_read_errors.csv", error_rows)
    write_rows(out / "write_to_swa_usage_proxy_rows.csv", proxy_rows)
    write_rows(out / "cue_control_metrics.csv", cue_rows)
    write_rows(out / "cue_control_metrics_hygiene_repair.csv", hygiene_cue_rows)
    write_json(out / "summary.json", summary)
    write_text(
        out / "proxy_limitations.md",
        "# Proxy Limitations\n\n"
        "This stage joins v97 F2 TTT stable-anchor write/retention maps with v98 case-level SWA top-k trace summaries.\n\n"
        "It does not contain persistent stable-anchor ids, later READ reads, or later SWA cache-key/value identity for the written anchors. "
        "Therefore it cannot prove or falsify true F3 write-to-use failure, even if a proxy cue separates these six cases.\n\n"
        f"proxy_case_count={len(proxy_rows)}; sequence_coverage={sequence_coverage}; min_cases_required=12; true_anchor_identity_available=False.\n",
    )
    write_failure_common(
        out,
        stage_name="Stage7b F3 TTT write-to-SWA usage proxy",
        summary=summary,
        failure_rows=[
            {"stage": "stage7b_f3_proxy", "failed_check": "true_anchor_identity_available", "observed_value": False, "gate_pass": False},
            {"stage": "stage7b_f3_proxy", "failed_check": "proxy_case_count>=12", "observed_value": len(proxy_rows), "gate_pass": False},
            {"stage": "stage7b_f3_proxy", "failed_check": "sequence_coverage>=3", "observed_value": sequence_coverage, "gate_pass": False},
        ] + cue_failure_attribution(cue_rows, "stage7b_proxy") + cue_failure_attribution(hygiene_cue_rows, "stage7b_proxy_hygiene_repair"),
        visual_rows=[
            {"artifact": "ttt_spatial_map_proxy_rows.csv", "path": str(out / "ttt_spatial_map_proxy_rows.csv"), "status": "generated_table", "source_metric": "ttt_stable_anchor_write_retention"},
            {"artifact": "write_to_swa_usage_proxy_rows.csv", "path": str(out / "write_to_swa_usage_proxy_rows.csv"), "status": "generated_table", "source_metric": "proxy_semantic_anchor_topk_identity"},
        ],
        what_would_text=(
            "# What Would Have To Be True To Pass\n\n"
            "F3 needs persistent stable-anchor ids written by TTT at chunk c, later READ/SWA usage of those same ids at c+1/c+2/c+3, "
            "top-k identity stability for the used anchors, >=12 proxy/identity cases with >=3 sequences, good-control FPR <=0.25, "
            "correct L3/L4 direction, and evidence not reducible to write energy alone."
        ),
        next_route_text="# Next Route Recommendation\n\nImplement a true stable-anchor identity propagation hook in the TTT write path and SWA/READ read path before any TTT action or no-write conclusion.",
    )
    return summary


def stage7c_ttt_swa_same_run_ledger() -> dict[str, Any]:
    out = ROOT / "stage7c_ttt_swa_same_run_alignment"
    run_root = stage7c_probe_run_root()
    runner_summary = read_json(run_root / "summary.json")
    stage1_rows = read_rows(ROOT / "stage1_trackK_swa_v2_strict_stable_eligibility/case_universe_rows.csv")
    meta_by_case = {str(row.get("case_id", "")): row for row in stage1_rows if row.get("case_id")}
    ttt_map_rows: list[dict[str, Any]] = []
    swa_trace_rows: list[dict[str, Any]] = []
    hmc_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []

    def tensor_mean(tensor: Any, mask: Any | None = None) -> float:
        try:
            import torch

            if not torch.is_tensor(tensor):
                return math.nan
            x = tensor.detach().cpu().float()
            if mask is not None and torch.is_tensor(mask):
                m = mask.detach().cpu().bool()
                if tuple(m.shape) != tuple(x.shape):
                    return math.nan
                x = x[m]
            x = x[torch.isfinite(x)]
            return float(x.mean().item()) if int(x.numel()) > 0 else math.nan
        except Exception:  # noqa: BLE001
            return math.nan

    for path in sorted(run_root.glob("*/TTT_SWA_SAME_RUN/ttt_spatial_post_delta_maps/*.pt")):
        case_id = path.parents[2].name
        try:
            import torch

            payload = torch.load(path, map_location="cpu")
        except Exception as exc:  # noqa: BLE001
            error_rows.append({"artifact_type": "ttt_spatial_map", "case_id": case_id, "path": str(path), "error": f"{type(exc).__name__}:{exc}"})
            continue
        mask = payload.get("stable_anchor_mask_patch")
        retention = payload.get("stable_anchor_retention_patch")
        residual = payload.get("stable_anchor_residual_patch")
        prior = payload.get("ttt_write_prior_patch")
        energy = payload.get("U_ttt_write_replay_contribution_patch")
        ttt_map_rows.append(
            {
                "case_id": case_id,
                "chunk_idx": payload.get("chunk_idx", ""),
                "path": str(path),
                "spatial_token_aligned": payload.get("spatial_token_aligned", ""),
                "stable_anchor_mask_frac": tensor_mean(mask),
                "stable_anchor_retention_on_mask": tensor_mean(retention, mask),
                "stable_anchor_residual_on_mask": tensor_mean(residual, mask),
                "stable_anchor_write_prior_on_mask": tensor_mean(prior, mask),
                "stable_anchor_write_energy_on_mask": tensor_mean(energy, mask),
                "stable_anchor_retention_available": (
                    (payload.get("condition_map_provenance") or {})
                    .get("replay_contribution", {})
                    .get("stable_anchor_retention_available", False)
                ),
            }
        )

    for path in sorted(run_root.glob("*/TTT_SWA_SAME_RUN/swa_raw_transport_trace/*.pt")):
        case_id = path.parents[2].name
        try:
            import torch

            payload = torch.load(path, map_location="cpu")
        except Exception as exc:  # noqa: BLE001
            error_rows.append({"artifact_type": "swa_raw_transport_trace", "case_id": case_id, "path": str(path), "error": f"{type(exc).__name__}:{exc}"})
            continue
        swa_trace_rows.append(
            {
                "case_id": case_id,
                "chunk_idx": payload.get("chunk_idx", ""),
                "swa_layer_idx": payload.get("swa_layer_idx", ""),
                "path": str(path),
                "stable_pair_strict_tokens": f(payload.get("stable_pair_strict_tokens")),
                "stable_pair_tokens": f(payload.get("stable_pair_tokens")),
                "stable_pair_fallback_used": b(payload.get("stable_pair_fallback_used")),
                "stable_pair_fallback_reason": payload.get("stable_pair_fallback_reason", ""),
                "stable_pair_mass_mean": f(payload.get("stable_pair_mass_mean")),
                "unreliable_pair_mass_mean": f(payload.get("unreliable_pair_mass_mean")),
                "topk_identity_available": b(payload.get("topk_identity_available")),
                "topk_same_frame_frac_mean": f(payload.get("topk_same_frame_frac_mean")),
                "topk_query_frame_hit_frac_mean": f(payload.get("topk_query_frame_hit_frac_mean")),
                "top1_cache_frame_switch_rate_mean": f(payload.get("top1_cache_frame_switch_rate_mean")),
                "top1_cache_index_unique_frac_mean": f(payload.get("top1_cache_index_unique_frac_mean")),
                "route_entropy_mean": f(payload.get("route_entropy_mean")),
            }
        )

    for path in sorted(run_root.glob("*/TTT_SWA_SAME_RUN/hmc_state_hash.jsonl")):
        case_id = path.parents[1].name
        try:
            lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        except Exception as exc:  # noqa: BLE001
            error_rows.append({"artifact_type": "hmc_state_hash", "case_id": case_id, "path": str(path), "error": f"{type(exc).__name__}:{exc}"})
            continue
        for row in lines:
            control = row.get("control_trace") if isinstance(row.get("control_trace"), dict) else {}
            hmc_rows.append(
                {
                    "case_id": case_id,
                    "chunk_idx": row.get("chunk_idx", ""),
                    "path": str(path),
                    "implemented_paths": control.get("implemented_paths", []),
                    "identity_hook_paths": control.get("identity_hook_paths", []),
                    "identity_hook_path_count": len(control.get("identity_hook_paths", []) or []),
                    "hybrid_memory_mode": row.get("hybrid_memory_mode", ""),
                    "hmc_commit_mode": row.get("hmc_commit_mode", ""),
                }
            )

    ttt_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ttt_map_rows:
        ttt_by_case[str(row.get("case_id", ""))].append(row)
    swa_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in swa_trace_rows:
        swa_by_case[str(row.get("case_id", ""))].append(row)
    hmc_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in hmc_rows:
        hmc_by_case[str(row.get("case_id", ""))].append(row)

    alignment_rows: list[dict[str, Any]] = []
    for case_id in sorted(set(ttt_by_case) | set(swa_by_case)):
        meta = meta_by_case.get(case_id, {})
        ttt_rows = ttt_by_case.get(case_id, [])
        swa_rows = swa_by_case.get(case_id, [])
        hrows = hmc_by_case.get(case_id, [])
        strict_max = max(f(row.get("stable_pair_strict_tokens"), 0.0) for row in swa_rows) if swa_rows else math.nan
        fallback_frac = mean([1.0 if b(row.get("stable_pair_fallback_used")) else 0.0 for row in swa_rows])
        topk_unstable = mean([f(row.get("top1_cache_frame_switch_rate_mean")) for row in swa_rows])
        retention = mean([f(row.get("stable_anchor_retention_on_mask")) for row in ttt_rows])
        energy = mean([f(row.get("stable_anchor_write_energy_on_mask")) for row in ttt_rows])
        topk_same = mean([f(row.get("topk_same_frame_frac_mean")) for row in swa_rows])
        alignment_rows.append(
            {
                "case_id": case_id,
                "seq": meta.get("seq", case_id.split("_")[0]),
                "case_label": meta.get("case_label", ""),
                "good_control_hygiene_include_for_repair": meta.get("good_control_hygiene_include_for_repair", True),
                "payload_count": 1,
                "L3_handoff_transfer_penalty_proxy": f(meta.get("L3_handoff_transfer_penalty_proxy")),
                "ttt_map_count": len(ttt_rows),
                "swa_trace_count": len(swa_rows),
                "hmc_row_count": len(hrows),
                "hmc_identity_hook_path_count": sum(int(f(row.get("identity_hook_path_count"), 0.0)) for row in hrows),
                "ttt_stable_anchor_mask_frac_mean": mean([f(row.get("stable_anchor_mask_frac")) for row in ttt_rows]),
                "ttt_stable_anchor_retention_mean": retention,
                "ttt_stable_anchor_residual_mean": mean([f(row.get("stable_anchor_residual_on_mask")) for row in ttt_rows]),
                "ttt_stable_anchor_write_prior_mean": mean([f(row.get("stable_anchor_write_prior_on_mask")) for row in ttt_rows]),
                "ttt_stable_anchor_write_energy_mean": energy,
                "swa_stable_pair_strict_tokens_max": strict_max,
                "swa_stable_pair_strict_nonempty": math.isfinite(strict_max) and strict_max > 0.0,
                "swa_stable_pair_fallback_frac": fallback_frac,
                "swa_stable_pair_mass_mean": mean([f(row.get("stable_pair_mass_mean")) for row in swa_rows]),
                "swa_unreliable_pair_mass_mean": mean([f(row.get("unreliable_pair_mass_mean")) for row in swa_rows]),
                "swa_topk_identity_available_frac": mean([1.0 if b(row.get("topk_identity_available")) else 0.0 for row in swa_rows]),
                "swa_topk_same_frame_frac_mean": topk_same,
                "swa_topk_query_frame_hit_frac_mean": mean([f(row.get("topk_query_frame_hit_frac_mean")) for row in swa_rows]),
                "swa_top1_cache_frame_switch_rate_mean": topk_unstable,
                "swa_top1_cache_index_unique_frac_mean": mean([f(row.get("top1_cache_index_unique_frac_mean")) for row in swa_rows]),
                "same_run_retention_x_topk_unstable": retention * topk_unstable if math.isfinite(retention) and math.isfinite(topk_unstable) else math.nan,
                "same_run_energy_not_topk_used": energy * (1.0 - topk_same) if math.isfinite(energy) and math.isfinite(topk_same) else math.nan,
                "same_run_strict_stable_missing": not (math.isfinite(strict_max) and strict_max > 0.0),
                "claim_scope": "same_run_ttt_write_and_swa_trace_only",
                "true_anchor_identity_available": False,
            }
        )

    specs = {
        "same_run_strict_stable_tokens_lower_bad": ("swa_stable_pair_strict_tokens_max", False),
        "same_run_swa_fallback_frac_higher_bad": ("swa_stable_pair_fallback_frac", True),
        "same_run_retention_x_topk_unstable_higher_bad": ("same_run_retention_x_topk_unstable", True),
        "same_run_energy_not_topk_used_higher_bad": ("same_run_energy_not_topk_used", True),
        "same_run_ttt_mask_frac_higher_bad": ("ttt_stable_anchor_mask_frac_mean", True),
    }
    cue_rows = evaluate_cues(alignment_rows, specs, min_cases=12, require_direction=True, view_name="same_run")
    strict_nonempty_count = sum(1 for row in alignment_rows if b(row.get("swa_stable_pair_strict_nonempty")))
    fallback_case_count = sum(1 for row in alignment_rows if f(row.get("swa_stable_pair_fallback_frac"), 0.0) > 0.0)
    hmc_identity_hook_nonempty_count = sum(1 for row in alignment_rows if int(f(row.get("hmc_identity_hook_path_count"), 0.0)) > 0)
    gate = False
    summary = {
        "schema": "acl2_v98_stage7c_ttt_swa_same_run_alignment_v1",
        "status": "complete" if runner_summary.get("status") == "complete" else "complete_with_runner_status_" + str(runner_summary.get("status", "missing")),
        "gate_pass": gate,
        "runtime_action_allowed": False,
        "claim_level": "DIAGNOSTIC_CUE_ONLY",
        "run_root": str(run_root),
        "runner_status": runner_summary.get("status", "missing"),
        "same_run_case_count": len(alignment_rows),
        "case_with_both_count": runner_summary.get("case_with_both_count", ""),
        "ttt_spatial_map_file_count": len(ttt_map_rows),
        "swa_raw_transport_trace_file_count": len(swa_trace_rows),
        "hmc_row_count": len(hmc_rows),
        "read_error_count": len(error_rows),
        "swa_strict_stable_nonempty_case_count": strict_nonempty_count,
        "swa_stable_pair_fallback_case_count": fallback_case_count,
        "hmc_identity_hook_nonempty_case_count": hmc_identity_hook_nonempty_count,
        "true_anchor_identity_available": False,
        "write_to_use_chain_available": False,
        "f3_write_to_use_failure_shown": False,
        "primary_blocker": "Same-run TTT write maps and SWA traces are available, but SWA strict stable pairs are empty/fallback-only in this probe and no persistent stable-anchor identity hook is present.",
    }
    write_rows(out / "same_run_alignment_rows.csv", alignment_rows)
    write_rows(out / "ttt_spatial_map_rows.csv", ttt_map_rows)
    write_rows(out / "swa_raw_transport_trace_rows.csv", swa_trace_rows)
    write_rows(out / "hmc_identity_hook_rows.csv", hmc_rows)
    write_rows(out / "read_errors.csv", error_rows)
    write_rows(out / "cue_control_metrics.csv", cue_rows)
    write_json(out / "summary.json", summary)
    write_failure_common(
        out,
        stage_name="Stage7c same-run TTT write + SWA trace alignment",
        summary=summary,
        failure_rows=[
            {"stage": "stage7c_same_run", "failed_check": "true_anchor_identity_available", "observed_value": False, "gate_pass": False},
            {"stage": "stage7c_same_run", "failed_check": "swa_strict_stable_nonempty_case_count>=same_run_case_count", "observed_value": strict_nonempty_count, "gate_pass": False},
            {"stage": "stage7c_same_run", "failed_check": "hmc_identity_hook_nonempty_case_count>0", "observed_value": hmc_identity_hook_nonempty_count, "gate_pass": False},
        ] + cue_failure_attribution(cue_rows, "stage7c_same_run"),
        visual_rows=[
            {"artifact": "same_run_alignment_rows.csv", "path": str(out / "same_run_alignment_rows.csv"), "status": "generated_table", "source_metric": "same_run_alignment"},
            {"artifact": "swa_raw_transport_trace_rows.csv", "path": str(out / "swa_raw_transport_trace_rows.csv"), "status": "generated_table", "source_metric": "same_run_swa_trace"},
        ],
        what_would_text=(
            "# What Would Have To Be True To Pass\n\n"
            "Stage7c would need same-run TTT write maps and SWA/READ read traces with persistent stable-anchor ids, non-fallback strict stable SWA pairs, "
            "identity_hook_paths populated, >=12 cases over >=3 sequences, and good-control-safe correlation with future L3/L4 drift."
        ),
        next_route_text="# Next Route Recommendation\n\nAdd a persistent stable-anchor id propagation hook that writes anchor ids in TTT and records the same ids when later READ/SWA top-k retrieves them.",
    )
    return summary


def stage7d_ttt_swa_spatial_token_proxy_identity() -> dict[str, Any]:
    out = ROOT / "stage7d_ttt_swa_spatial_token_proxy_identity"
    run_root = stage7c_probe_run_root()
    stage1_rows = read_rows(ROOT / "stage1_trackK_swa_v2_strict_stable_eligibility/case_universe_rows.csv")
    meta_by_case = {str(row.get("case_id", "")): row for row in stage1_rows if row.get("case_id")}
    proxy_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []

    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        summary = {
            "schema": "acl2_v98_stage7d_ttt_swa_spatial_token_proxy_identity_v1",
            "status": "blocked_missing_torch",
            "gate_pass": False,
            "runtime_action_allowed": False,
            "claim_level": "NO_CLAIM",
            "read_error_count": 1,
            "primary_blocker": f"torch import failed: {type(exc).__name__}:{exc}",
        }
        write_json(out / "summary.json", summary)
        write_rows(out / "read_errors.csv", [{"artifact_type": "torch_import", "error": summary["primary_blocker"]}])
        return summary

    def _tensor_mean(tensor: Any, mask: Any | None = None) -> float:
        if not torch.is_tensor(tensor):
            return math.nan
        x = tensor.detach().cpu().float()
        if mask is not None and torch.is_tensor(mask):
            m = mask.detach().cpu().bool()
            if tuple(m.shape) != tuple(x.shape):
                return math.nan
            x = x[m]
        x = x[torch.isfinite(x)]
        return float(x.mean().item()) if int(x.numel()) > 0 else math.nan

    def _full_token_mask_from_patch(patch_mask: Any, *, frames: int, tokens_per_frame: int) -> tuple[Any | None, int, str]:
        if not torch.is_tensor(patch_mask):
            return None, -1, "missing_patch_mask"
        patch = patch_mask.detach().cpu().float()
        if int(patch.ndim) != 3:
            return None, -1, f"patch_mask_ndim_{int(patch.ndim)}"
        patch_frames = int(patch.shape[0])
        patch_tokens = int(patch.shape[1]) * int(patch.shape[2])
        if patch_frames != int(frames):
            return None, -1, f"frame_mismatch_patch_{patch_frames}_history_{frames}"
        patch_start = int(tokens_per_frame) - int(patch_tokens)
        if patch_start < 0:
            return None, patch_start, f"tokens_per_frame_{tokens_per_frame}_smaller_than_patch_tokens_{patch_tokens}"
        full = torch.zeros(int(frames) * int(tokens_per_frame), dtype=torch.bool)
        flat_patch = patch.reshape(int(frames), int(patch_tokens)) >= 0.5
        for frame_idx in range(int(frames)):
            start = frame_idx * int(tokens_per_frame) + patch_start
            full[start : start + int(patch_tokens)] = flat_patch[frame_idx]
        return full, patch_start, ""

    def _rolled_control_hit(full_mask: Any, idx: Any, *, seed_text: str, shifts: int = 8) -> float:
        if not torch.is_tensor(full_mask) or not torch.is_tensor(idx) or int(full_mask.numel()) <= 0:
            return math.nan
        vals: list[float] = []
        n = int(full_mask.numel())
        for shift_idx in range(int(shifts)):
            digest = hashlib.sha256(f"{seed_text}:{shift_idx}".encode("utf-8")).hexdigest()
            shift = int(digest[:8], 16) % n
            if shift == 0:
                shift = 1
            rolled = torch.roll(full_mask, shifts=shift, dims=0)
            vals.append(float(rolled[idx].float().mean().item()))
        return median(vals)

    for case_dir in sorted(run_root.glob("*/TTT_SWA_SAME_RUN")):
        case_id = case_dir.parent.name
        map_by_chunk: dict[int, dict[str, Any]] = {}
        for path in sorted((case_dir / "ttt_spatial_post_delta_maps").glob("*.pt")):
            try:
                payload = torch.load(path, map_location="cpu")
                map_by_chunk[int(payload.get("chunk_idx", -999999))] = payload
            except Exception as exc:  # noqa: BLE001
                error_rows.append({"artifact_type": "ttt_spatial_map", "case_id": case_id, "path": str(path), "error": f"{type(exc).__name__}:{exc}"})
        for path in sorted((case_dir / "swa_raw_transport_trace").glob("*.pt")):
            try:
                swa = torch.load(path, map_location="cpu")
            except Exception as exc:  # noqa: BLE001
                error_rows.append({"artifact_type": "swa_raw_transport_trace", "case_id": case_id, "path": str(path), "error": f"{type(exc).__name__}:{exc}"})
                continue
            topk = swa.get("current_Q_to_cache_K_topk_cache_indices")
            if not torch.is_tensor(topk):
                error_rows.append({"artifact_type": "swa_raw_transport_trace", "case_id": case_id, "path": str(path), "error": "missing_topk_cache_indices"})
                continue
            topk = topk.detach().cpu().long()
            if int(topk.ndim) != 4:
                error_rows.append({"artifact_type": "swa_raw_transport_trace", "case_id": case_id, "path": str(path), "error": f"topk_ndim_{int(topk.ndim)}"})
                continue
            chunk_idx = int(swa.get("chunk_idx", -999999))
            tokens_per_frame = int(swa.get("tokens_per_frame", 0) or 0)
            history_tokens = int(swa.get("history_tokens", 0) or 0)
            frames = int(history_tokens // tokens_per_frame) if tokens_per_frame > 0 else 0
            idx_min = int(topk.min().item()) if int(topk.numel()) else -1
            idx_max = int(topk.max().item()) if int(topk.numel()) else -1
            sampled_query_count = int(swa.get("sampled_query_count", 0) or 0)
            topk_k = int(topk.shape[-1])
            for alignment_role, source_chunk in (("prev_chunk", chunk_idx - 1), ("same_chunk_control", chunk_idx)):
                source = map_by_chunk.get(int(source_chunk))
                if source is None:
                    error_rows.append({
                        "artifact_type": "alignment_source_map",
                        "case_id": case_id,
                        "swa_path": str(path),
                        "alignment_role": alignment_role,
                        "swa_chunk_idx": chunk_idx,
                        "source_chunk_idx": source_chunk,
                        "error": "missing_ttt_map_for_source_chunk",
                    })
                    continue
                full_mask, patch_start, mask_error = _full_token_mask_from_patch(
                    source.get("stable_anchor_mask_patch"),
                    frames=frames,
                    tokens_per_frame=tokens_per_frame,
                )
                if full_mask is None:
                    error_rows.append({
                        "artifact_type": "alignment_full_token_mask",
                        "case_id": case_id,
                        "swa_path": str(path),
                        "alignment_role": alignment_role,
                        "swa_chunk_idx": chunk_idx,
                        "source_chunk_idx": source_chunk,
                        "error": mask_error,
                    })
                    continue
                topk_in_range = bool(int(topk.numel()) > 0 and int(topk.min().item()) >= 0 and int(topk.max().item()) < int(full_mask.numel()))
                if not topk_in_range:
                    error_rows.append({
                        "artifact_type": "alignment_topk_range",
                        "case_id": case_id,
                        "swa_path": str(path),
                        "alignment_role": alignment_role,
                        "swa_chunk_idx": chunk_idx,
                        "source_chunk_idx": source_chunk,
                        "error": f"topk_range_{idx_min}_{idx_max}_outside_0_{int(full_mask.numel()) - 1}",
                    })
                    continue
                top1 = topk[..., 0]
                topk_hit = full_mask[topk]
                top1_hit = full_mask[top1]
                query_hit = topk_hit.any(dim=-1)
                special_topk_frac = float(((topk % int(tokens_per_frame)) < int(patch_start)).float().mean().item()) if patch_start >= 0 else math.nan
                random_hit = _rolled_control_hit(full_mask, topk, seed_text=f"{case_id}:{chunk_idx}:{source_chunk}:{swa.get('swa_layer_idx', '')}")
                hit_frac = float(topk_hit.float().mean().item())
                mask_frac = float(full_mask.float().mean().item())
                row = {
                    "case_id": case_id,
                    "alignment_role": alignment_role,
                    "swa_chunk_idx": chunk_idx,
                    "source_ttt_chunk_idx": source_chunk,
                    "swa_layer_idx": swa.get("swa_layer_idx", ""),
                    "swa_trace_path": str(path),
                    "source_ttt_map_path": str(case_dir / "ttt_spatial_post_delta_maps" / f"chunk_{int(source_chunk):03d}_ttt_spatial_post_delta_map.pt"),
                    "tokens_per_frame": tokens_per_frame,
                    "history_tokens": history_tokens,
                    "history_frame_count": frames,
                    "patch_start_idx": patch_start,
                    "topk_k": topk_k,
                    "sampled_query_count": sampled_query_count,
                    "topk_index_min": idx_min,
                    "topk_index_max": idx_max,
                    "full_token_mask_tokens": int(full_mask.numel()),
                    "stable_anchor_mask_frac_full": mask_frac,
                    "stable_anchor_topk_hit_frac": hit_frac,
                    "stable_anchor_topk_query_hit_frac": float(query_hit.float().mean().item()),
                    "stable_anchor_top1_hit_frac": float(top1_hit.float().mean().item()),
                    "stable_anchor_topk_hit_minus_mask_frac": hit_frac - mask_frac,
                    "stable_anchor_topk_hit_random_roll_median": random_hit,
                    "stable_anchor_topk_hit_minus_random_roll": hit_frac - random_hit if math.isfinite(random_hit) else math.nan,
                    "special_token_topk_frac": special_topk_frac,
                    "stable_anchor_retention_on_mask": _tensor_mean(source.get("stable_anchor_retention_patch"), source.get("stable_anchor_mask_patch")),
                    "stable_anchor_residual_on_mask": _tensor_mean(source.get("stable_anchor_residual_patch"), source.get("stable_anchor_mask_patch")),
                    "stable_anchor_write_prior_on_mask": _tensor_mean(source.get("ttt_write_prior_patch"), source.get("stable_anchor_mask_patch")),
                    "stable_anchor_write_energy_on_mask": _tensor_mean(source.get("U_ttt_write_replay_contribution_patch"), source.get("stable_anchor_mask_patch")),
                    "claim_scope": "same_run_spatial_token_proxy_identity_only",
                    "true_anchor_identity_available": False,
                }
                proxy_rows.append(row)

    prev_rows = [row for row in proxy_rows if str(row.get("alignment_role")) == "prev_chunk"]
    same_rows = [row for row in proxy_rows if str(row.get("alignment_role")) == "same_chunk_control"]
    prev_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    same_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in prev_rows:
        prev_by_case[str(row.get("case_id", ""))].append(row)
    for row in same_rows:
        same_by_case[str(row.get("case_id", ""))].append(row)
    for case_id in sorted(set(prev_by_case) | set(same_by_case)):
        meta = meta_by_case.get(case_id, {})
        prows = prev_by_case.get(case_id, [])
        srows = same_by_case.get(case_id, [])
        prev_hit = mean([f(row.get("stable_anchor_topk_hit_frac")) for row in prows])
        same_hit = mean([f(row.get("stable_anchor_topk_hit_frac")) for row in srows])
        prev_random_delta = mean([f(row.get("stable_anchor_topk_hit_minus_random_roll")) for row in prows])
        case_rows.append(
            {
                "case_id": case_id,
                "seq": meta.get("seq", case_id.split("_")[0]),
                "case_label": meta.get("case_label", ""),
                "good_control_hygiene_include_for_repair": meta.get("good_control_hygiene_include_for_repair", True),
                "payload_count": 1,
                "L3_handoff_transfer_penalty_proxy": f(meta.get("L3_handoff_transfer_penalty_proxy")),
                "prev_layer_row_count": len(prows),
                "same_control_layer_row_count": len(srows),
                "prev_stable_anchor_mask_frac_full_mean": mean([f(row.get("stable_anchor_mask_frac_full")) for row in prows]),
                "prev_stable_anchor_topk_hit_frac_mean": prev_hit,
                "prev_stable_anchor_topk_query_hit_frac_mean": mean([f(row.get("stable_anchor_topk_query_hit_frac")) for row in prows]),
                "prev_stable_anchor_top1_hit_frac_mean": mean([f(row.get("stable_anchor_top1_hit_frac")) for row in prows]),
                "prev_hit_minus_mask_frac_mean": mean([f(row.get("stable_anchor_topk_hit_minus_mask_frac")) for row in prows]),
                "prev_hit_minus_random_roll_mean": prev_random_delta,
                "prev_special_token_topk_frac_mean": mean([f(row.get("special_token_topk_frac")) for row in prows]),
                "prev_stable_anchor_retention_on_mask_mean": mean([f(row.get("stable_anchor_retention_on_mask")) for row in prows]),
                "prev_stable_anchor_write_energy_on_mask_mean": mean([f(row.get("stable_anchor_write_energy_on_mask")) for row in prows]),
                "same_control_stable_anchor_topk_hit_frac_mean": same_hit,
                "same_control_hit_minus_prev_hit_mean": same_hit - prev_hit if math.isfinite(same_hit) and math.isfinite(prev_hit) else math.nan,
                "spatial_token_proxy_alignment_available": math.isfinite(prev_hit),
                "proxy_write_to_use_chain_available": False,
                "true_anchor_identity_available": False,
                "claim_scope": "prev_chunk_ttt_stable_patch_mask_vs_current_swa_topk_proxy",
            }
        )

    specs = {
        "spatial_prev_topk_stable_hit_lower_bad": ("prev_stable_anchor_topk_hit_frac_mean", False),
        "spatial_prev_query_stable_hit_lower_bad": ("prev_stable_anchor_topk_query_hit_frac_mean", False),
        "spatial_prev_top1_stable_hit_lower_bad": ("prev_stable_anchor_top1_hit_frac_mean", False),
        "spatial_prev_hit_minus_random_lower_bad": ("prev_hit_minus_random_roll_mean", False),
        "spatial_prev_special_topk_frac_higher_bad": ("prev_special_token_topk_frac_mean", True),
        "spatial_same_minus_prev_higher_bad": ("same_control_hit_minus_prev_hit_mean", True),
    }
    cue_rows = evaluate_cues(case_rows, specs, min_cases=12, require_direction=True, view_name="stage7d_spatial_proxy")
    hygiene_cue_rows = evaluate_cues(
        case_rows,
        specs,
        min_cases=12,
        require_direction=True,
        row_filter=lambda row: b(row.get("good_control_hygiene_include_for_repair", True)),
        view_name="stage7d_spatial_proxy_hygiene_repair",
    )
    sequence_coverage = len({str(row.get("seq", "")) for row in case_rows})
    prev_hit_values = [f(row.get("prev_stable_anchor_topk_hit_frac_mean")) for row in case_rows]
    prev_mask_values = [f(row.get("prev_stable_anchor_mask_frac_full_mean")) for row in case_rows]
    same_delta_values = [f(row.get("same_control_hit_minus_prev_hit_mean")) for row in case_rows]
    gate = False
    summary = {
        "schema": "acl2_v98_stage7d_ttt_swa_spatial_token_proxy_identity_v1",
        "status": "complete_proxy_only",
        "gate_pass": gate,
        "runtime_action_allowed": False,
        "claim_level": "DIAGNOSTIC_CUE_ONLY",
        "run_root": str(run_root),
        "case_count": len(case_rows),
        "prev_layer_row_count": len(prev_rows),
        "same_control_layer_row_count": len(same_rows),
        "read_error_count": len(error_rows),
        "sequence_coverage": sequence_coverage,
        "min_cases_required": 12,
        "true_anchor_identity_available": False,
        "persistent_anchor_id_available": False,
        "write_to_use_chain_available": False,
        "f3_write_to_use_failure_shown": False,
        "spatial_token_proxy_available": bool(prev_rows),
        "prev_topk_hit_frac_mean": mean(prev_hit_values),
        "prev_stable_anchor_mask_frac_full_mean": mean(prev_mask_values),
        "prev_hit_minus_mask_frac_mean": mean([f(row.get("prev_hit_minus_mask_frac_mean")) for row in case_rows]),
        "prev_hit_minus_random_roll_mean": mean([f(row.get("prev_hit_minus_random_roll_mean")) for row in case_rows]),
        "same_control_minus_prev_hit_mean": mean(same_delta_values),
        "diagnostic_proxy_metric_gate_pass": any(b(row.get("gate_pass")) for row in cue_rows + hygiene_cue_rows),
        "primary_blocker": "Spatial token proxy can align TTT stable patch masks to SWA top-k indices, but coverage is six cases and there is no persistent stable-anchor id, so this cannot authorize F3/runtime action.",
    }
    write_rows(out / "spatial_token_proxy_rows.csv", proxy_rows)
    write_rows(out / "spatial_token_proxy_case_rows.csv", case_rows)
    write_rows(out / "cue_control_metrics.csv", cue_rows)
    write_rows(out / "cue_control_metrics_hygiene_repair.csv", hygiene_cue_rows)
    write_rows(out / "read_errors.csv", error_rows)
    write_json(out / "summary.json", summary)
    write_text(
        out / "proxy_limitations.md",
        "# Spatial Token Proxy Limitations\n\n"
        "This stage maps TTT `stable_anchor_mask_patch` onto the SWA cache token layout by using `patch_start_idx = tokens_per_frame - H*W` "
        "and tests whether current SWA top-k cache indices hit the previous chunk's stable-anchor patch mask.\n\n"
        "It is a same-run spatial-token proxy, not a persistent stable-anchor id. It does not prove that a TTT-written anchor id was later read, "
        "and it cannot authorize runtime action or no-write claims.\n",
    )
    write_failure_common(
        out,
        stage_name="Stage7d F3 spatial-token proxy identity",
        summary=summary,
        failure_rows=[
            {"stage": "stage7d_spatial_proxy", "failed_check": "true_anchor_identity_available", "observed_value": False, "gate_pass": False},
            {"stage": "stage7d_spatial_proxy", "failed_check": "case_count>=12", "observed_value": len(case_rows), "gate_pass": False},
            {"stage": "stage7d_spatial_proxy", "failed_check": "sequence_coverage>=3", "observed_value": sequence_coverage, "gate_pass": False},
        ] + cue_failure_attribution(cue_rows, "stage7d_spatial_proxy") + cue_failure_attribution(hygiene_cue_rows, "stage7d_spatial_proxy_hygiene_repair"),
        visual_rows=[
            {"artifact": "spatial_token_proxy_rows.csv", "path": str(out / "spatial_token_proxy_rows.csv"), "status": "generated_table", "source_metric": "swa_topk_x_ttt_stable_patch_mask"},
            {"artifact": "spatial_token_proxy_case_rows.csv", "path": str(out / "spatial_token_proxy_case_rows.csv"), "status": "generated_table", "source_metric": "case_mean_spatial_proxy"},
        ],
        what_would_text=(
            "# What Would Have To Be True To Pass\n\n"
            "F3 still needs persistent stable-anchor ids written by TTT and observed in later READ/SWA retrieval, not just spatial patch-token overlap. "
            "A promotable proxy would also need >=12 cases, >=3 sequences, correct L3/L4 direction, and good-control-safe separation."
        ),
        next_route_text="# Next Route Recommendation\n\nImplement persistent stable-anchor id emission in the TTT write state and record those ids in later SWA/READ top-k retrieval traces.",
    )
    return summary


def stage7e_ttt_stable_anchor_id_hook() -> dict[str, Any]:
    out = ROOT / "stage7e_ttt_stable_anchor_id_hook"
    run_root = stage7e_anchor_id_hook_root()
    runner_summary = read_json(run_root / "summary.json")
    stage1_rows = read_rows(ROOT / "stage1_trackK_swa_v2_strict_stable_eligibility/case_universe_rows.csv")
    meta_by_case = {str(row.get("case_id", "")): row for row in stage1_rows if row.get("case_id")}
    layer_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []

    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        summary = {
            "schema": "acl2_v98_stage7e_ttt_stable_anchor_id_hook_v1",
            "status": "blocked_missing_torch",
            "gate_pass": False,
            "runtime_action_allowed": False,
            "read_error_count": 1,
            "primary_blocker": f"torch import failed: {type(exc).__name__}:{exc}",
        }
        write_json(out / "summary.json", summary)
        write_rows(out / "read_errors.csv", [{"artifact_type": "torch_import", "error": summary["primary_blocker"]}])
        return summary

    for path in sorted(run_root.glob("*/TTT_SWA_SAME_RUN/swa_raw_transport_trace/*.pt")):
        case_id = path.parents[2].name
        try:
            payload = torch.load(path, map_location="cpu")
        except Exception as exc:  # noqa: BLE001
            error_rows.append({"artifact_type": "swa_raw_transport_trace", "case_id": case_id, "path": str(path), "error": f"{type(exc).__name__}:{exc}"})
            continue
        ids = payload.get("current_Q_to_cache_K_topk_ttt_prev_stable_anchor_ids")
        hit = payload.get("current_Q_to_cache_K_topk_ttt_prev_stable_anchor_hit_mask")
        valid_id_count = 0
        unique_id_count = 0
        if torch.is_tensor(ids):
            valid = ids.detach().cpu().long() >= 0
            valid_id_count = int(valid.sum().item())
            unique_id_count = int(torch.unique(ids.detach().cpu().long()[valid]).numel()) if bool(valid.any()) else 0
        layer_rows.append(
            {
                "case_id": case_id,
                "chunk_idx": payload.get("chunk_idx", ""),
                "swa_layer_idx": payload.get("swa_layer_idx", ""),
                "path": str(path),
                "identity_available": b(payload.get("ttt_prev_stable_anchor_identity_available")),
                "source_chunk_idx": payload.get("ttt_prev_stable_anchor_source_chunk_idx", ""),
                "source_anchor_token_count": f(payload.get("ttt_prev_stable_anchor_source_token_count")),
                "full_anchor_token_count": f(payload.get("ttt_prev_stable_anchor_full_token_count")),
                "topk_hit_frac": f(payload.get("ttt_prev_stable_anchor_topk_hit_frac_mean")),
                "topk_query_hit_frac": f(payload.get("ttt_prev_stable_anchor_topk_query_hit_frac_mean")),
                "top1_hit_frac": f(payload.get("ttt_prev_stable_anchor_top1_hit_frac_mean")),
                "route_mass": f(payload.get("ttt_prev_stable_anchor_route_mass_mean")),
                "retention_on_anchor": f(payload.get("ttt_prev_stable_anchor_retention_mean")),
                "residual_on_anchor": f(payload.get("ttt_prev_stable_anchor_residual_mean")),
                "valid_topk_anchor_id_count": valid_id_count,
                "unique_topk_anchor_id_count": unique_id_count,
                "hit_mask_present": torch.is_tensor(hit),
                "claim_scope": "state_carried_ttt_anchor_id_to_swa_topk_diagnostic",
            }
        )

    rows_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in layer_rows:
        rows_by_case[str(row.get("case_id", ""))].append(row)
    for case_id in sorted(rows_by_case):
        meta = meta_by_case.get(case_id, {})
        rows = rows_by_case[case_id]
        unique_ids_total = sum(int(f(row.get("unique_topk_anchor_id_count"), 0.0)) for row in rows)
        case_rows.append(
            {
                "case_id": case_id,
                "seq": meta.get("seq", case_id.split("_")[0]),
                "case_label": meta.get("case_label", ""),
                "good_control_hygiene_include_for_repair": meta.get("good_control_hygiene_include_for_repair", True),
                "payload_count": 1,
                "L3_handoff_transfer_penalty_proxy": f(meta.get("L3_handoff_transfer_penalty_proxy")),
                "identity_layer_count": sum(1 for row in rows if b(row.get("identity_available"))),
                "swa_trace_layer_count": len(rows),
                "source_anchor_token_count_mean": mean([f(row.get("source_anchor_token_count")) for row in rows]),
                "full_anchor_token_count_mean": mean([f(row.get("full_anchor_token_count")) for row in rows]),
                "anchor_id_topk_hit_frac_mean": mean([f(row.get("topk_hit_frac")) for row in rows]),
                "anchor_id_topk_query_hit_frac_mean": mean([f(row.get("topk_query_hit_frac")) for row in rows]),
                "anchor_id_top1_hit_frac_mean": mean([f(row.get("top1_hit_frac")) for row in rows]),
                "anchor_id_route_mass_mean": mean([f(row.get("route_mass")) for row in rows]),
                "anchor_id_retention_mean": mean([f(row.get("retention_on_anchor")) for row in rows]),
                "anchor_id_residual_mean": mean([f(row.get("residual_on_anchor")) for row in rows]),
                "anchor_id_valid_topk_id_count_sum": sum(int(f(row.get("valid_topk_anchor_id_count"), 0.0)) for row in rows),
                "anchor_id_unique_topk_id_count_sum": unique_ids_total,
                "persistent_anchor_id_available": any(b(row.get("identity_available")) for row in rows),
                "write_to_swa_topk_chain_observed": unique_ids_total > 0,
                "claim_scope": "state_carried_ttt_anchor_id_to_swa_topk_diagnostic",
            }
        )

    specs = {
        "anchor_id_topk_hit_lower_bad": ("anchor_id_topk_hit_frac_mean", False),
        "anchor_id_topk_hit_higher_bad": ("anchor_id_topk_hit_frac_mean", True),
        "anchor_id_query_hit_lower_bad": ("anchor_id_topk_query_hit_frac_mean", False),
        "anchor_id_query_hit_higher_bad": ("anchor_id_topk_query_hit_frac_mean", True),
        "anchor_id_top1_hit_lower_bad": ("anchor_id_top1_hit_frac_mean", False),
        "anchor_id_top1_hit_higher_bad": ("anchor_id_top1_hit_frac_mean", True),
        "anchor_id_route_mass_lower_bad": ("anchor_id_route_mass_mean", False),
        "anchor_id_route_mass_higher_bad": ("anchor_id_route_mass_mean", True),
        "anchor_id_unique_hits_lower_bad": ("anchor_id_unique_topk_id_count_sum", False),
        "anchor_id_unique_hits_higher_bad": ("anchor_id_unique_topk_id_count_sum", True),
        "anchor_id_source_count_higher_bad": ("source_anchor_token_count_mean", True),
    }
    cue_rows = evaluate_cues(case_rows, specs, min_cases=12, require_direction=True, view_name="stage7e_anchor_id_hook")
    hygiene_cue_rows = evaluate_cues(
        case_rows,
        specs,
        min_cases=12,
        require_direction=True,
        row_filter=lambda row: b(row.get("good_control_hygiene_include_for_repair", True)),
        view_name="stage7e_anchor_id_hook_hygiene_repair",
    )
    identity_case_count = sum(1 for row in case_rows if b(row.get("persistent_anchor_id_available")))
    chain_case_count = sum(1 for row in case_rows if b(row.get("write_to_swa_topk_chain_observed")))
    sequence_coverage = len({str(row.get("seq", "")) for row in case_rows})
    diagnostic_metric_gate_pass = any(b(row.get("gate_pass")) for row in cue_rows + hygiene_cue_rows)
    gate = (
        identity_case_count >= 12
        and sequence_coverage >= 3
        and chain_case_count >= 12
        and diagnostic_metric_gate_pass
    )
    best_pool = cue_rows if cue_rows else hygiene_cue_rows
    summary = {
        "schema": "acl2_v98_stage7e_ttt_stable_anchor_id_hook_v1",
        "status": "complete_identity_hook_diagnostic",
        "gate_pass": gate,
        "runtime_action_allowed": False,
        "claim_level": "STATE_CARRIED_IDENTITY_DIAGNOSTIC_CUE_PASS" if gate else "STATE_CARRIED_IDENTITY_DIAGNOSTIC",
        "run_root": str(run_root),
        "runner_status": runner_summary.get("status", "missing"),
        "case_count": len(case_rows),
        "identity_case_count": identity_case_count,
        "write_to_swa_topk_chain_case_count": chain_case_count,
        "layer_row_count": len(layer_rows),
        "read_error_count": len(error_rows),
        "sequence_coverage": sequence_coverage,
        "min_cases_required": 12,
        "persistent_anchor_id_available": identity_case_count > 0,
        "write_to_use_chain_available": chain_case_count > 0,
        "f3_write_to_use_failure_shown": False,
        "f3_write_to_use_risk_cue_shown": gate,
        "best_cue": best_pool[0]["cue_name"] if best_pool else "",
        "best_cue_direction": best_pool[0]["direction"] if best_pool else "",
        "best_cue_bad_recall": best_pool[0]["bad_recall"] if best_pool else "",
        "best_cue_good_FPR": best_pool[0]["good_FPR"] if best_pool else "",
        "best_cue_abs_corr_L3": best_pool[0]["abs_corr_L3_handoff_transfer_penalty"] if best_pool else "",
        "anchor_id_topk_hit_frac_mean": mean([f(row.get("anchor_id_topk_hit_frac_mean")) for row in case_rows]),
        "anchor_id_topk_query_hit_frac_mean": mean([f(row.get("anchor_id_topk_query_hit_frac_mean")) for row in case_rows]),
        "anchor_id_route_mass_mean": mean([f(row.get("anchor_id_route_mass_mean")) for row in case_rows]),
        "anchor_id_unique_topk_id_count_sum": sum(int(f(row.get("anchor_id_unique_topk_id_count_sum"), 0.0)) for row in case_rows),
        "diagnostic_metric_gate_pass": diagnostic_metric_gate_pass,
        "primary_blocker": (
            "Stage7e found a state-carried anchor-id diagnostic cue, but no runtime action/no-write policy has been implemented or validated."
            if gate
            else (
                f"State-carried TTT stable-anchor ids are visible in later SWA top-k traces for "
                f"{len(case_rows)} cases over {sequence_coverage} sequences, but no cue passes the "
                "good-control/correlation/direction/sequence gates required for F3 action or no-write claims."
            )
        ),
    }
    write_rows(out / "anchor_id_layer_rows.csv", layer_rows)
    write_rows(out / "anchor_id_case_rows.csv", case_rows)
    write_rows(out / "cue_control_metrics.csv", cue_rows)
    write_rows(out / "cue_control_metrics_hygiene_repair.csv", hygiene_cue_rows)
    write_rows(out / "read_errors.csv", error_rows)
    write_json(out / "summary.json", summary)
    write_text(
        out / "identity_hook_limitations.md",
        "# Identity Hook Limitations\n\n"
        "This stage uses state-carried deterministic stable-anchor ids from TTT write diagnostics and observes those ids in later SWA top-k traces. "
        "The ids are patch-token anchor ids, not object ids, and the stage is diagnostic-only.\n\n"
        "It can establish that the write-to-SWA-top-k chain is observable. It cannot by itself prove F3 failure or authorize runtime action.\n",
    )
    write_failure_common(
        out,
        stage_name="Stage7e F3 state-carried stable-anchor id hook",
        summary=summary,
        failure_rows=[
            {"stage": "stage7e_anchor_id_hook", "failed_check": "case_count>=12", "observed_value": len(case_rows), "gate_pass": False},
            {"stage": "stage7e_anchor_id_hook", "failed_check": "sequence_coverage>=3", "observed_value": sequence_coverage, "gate_pass": False},
            {"stage": "stage7e_anchor_id_hook", "failed_check": "f3_write_to_use_failure_shown", "observed_value": False, "gate_pass": False},
        ] + cue_failure_attribution(cue_rows, "stage7e_anchor_id_hook") + cue_failure_attribution(hygiene_cue_rows, "stage7e_anchor_id_hook_hygiene_repair"),
        visual_rows=[
            {"artifact": "anchor_id_layer_rows.csv", "path": str(out / "anchor_id_layer_rows.csv"), "status": "generated_table", "source_metric": "state_carried_anchor_id_topk_hits"},
            {"artifact": "anchor_id_case_rows.csv", "path": str(out / "anchor_id_case_rows.csv"), "status": "generated_table", "source_metric": "case_mean_anchor_id_hits"},
        ],
        what_would_text=(
            "# What Would Have To Be True To Pass\n\n"
            "Stage7e would need >=12 cases over >=3 sequences, state-carried anchor ids available in all relevant cases, and a good-control-safe cue with correct L3/L4 direction showing write-to-use failure or recoverable usage."
        ),
        next_route_text="# Next Route Recommendation\n\nExpand the state-carried anchor-id hook to at least 12 cases over 3+ sequences, then evaluate whether low anchor-id use predicts L3/L4 harm without good-control leakage.",
    )
    return summary


def stage7f_prev_ttt_anchor_gate_action_pilot() -> dict[str, Any]:
    out = ROOT / "stage7f_prev_ttt_anchor_gate_action_pilot"
    comparison = read_json(ROOT / "stage7f_action_pilot_variant_comparison.json")
    summaries = comparison.get("summaries", {}) if isinstance(comparison.get("summaries"), dict) else {}
    rows = comparison.get("rows", []) if isinstance(comparison.get("rows"), list) else []
    if not summaries:
        summary = {
            "schema": "acl2_v98_stage7f_prev_ttt_anchor_gate_action_pilot_v1",
            "status": "not_run",
            "runtime_action_pilot_run": False,
            "formal_runtime_action_run": False,
            "full_validation_run": False,
            "gate_pass": False,
            "variant_count": 0,
            "case_count": 0,
            "primary_blocker": "No Stage7f action-pilot comparison artifact found.",
        }
        write_json(out / "summary.json", summary)
        return summary

    def score_variant(item: tuple[str, Any]) -> tuple[float, float, float]:
        _, payload = item
        if not isinstance(payload, dict):
            return (-999.0, -999.0, -999.0)
        return (
            f(payload.get("median_improvement_ratio_vs_baseline"), -999.0),
            f(payload.get("mean_improvement_ratio_vs_baseline"), -999.0),
            float(payload.get("improved_ate_case_count", 0) or 0),
        )

    best_variant, best_payload_raw = max(summaries.items(), key=score_variant)
    best_payload = best_payload_raw if isinstance(best_payload_raw, dict) else {}
    variant_rows = [
        {"variant": name, **payload}
        for name, payload in sorted(summaries.items())
        if isinstance(payload, dict)
    ]
    best_case_count = int(best_payload.get("case_count", 0) or 0)
    best_improved = int(best_payload.get("improved_ate_case_count", 0) or 0)
    best_worse = int(best_payload.get("worse_ate_case_count", 0) or 0)
    best_median_impr = f(best_payload.get("median_improvement_ratio_vs_baseline"), math.nan)
    gate = bool(best_case_count >= 6 and best_improved >= 4 and best_worse <= 2 and best_median_impr >= 0.05)
    summary = {
        "schema": "acl2_v98_stage7f_prev_ttt_anchor_gate_action_pilot_v1",
        "status": "complete_action_pilot_gate_pass" if gate else "complete_action_pilot_no_go",
        "runtime_action_pilot_run": True,
        "formal_runtime_action_run": False,
        "full_validation_run": False,
        "gate_pass": gate,
        "variant_count": len(variant_rows),
        "case_count": max((int(row.get("case_count", 0) or 0) for row in variant_rows), default=0),
        "best_variant": best_variant,
        "best_variant_improved_ate_case_count": best_improved,
        "best_variant_worse_ate_case_count": best_worse,
        "best_variant_median_improvement_ratio_vs_baseline": best_median_impr,
        "best_variant_mean_improvement_ratio_vs_baseline": f(best_payload.get("mean_improvement_ratio_vs_baseline"), math.nan),
        "best_variant_mean_action_minus_baseline_ate_rmse_m": f(best_payload.get("mean_action_minus_baseline_ate_rmse_m"), math.nan),
        "best_variant_median_action_minus_baseline_ate_rmse_m": f(best_payload.get("median_action_minus_baseline_ate_rmse_m"), math.nan),
        "gate_rule": "case_count>=6 and improved_cases>=4 and worse_cases<=2 and median_improvement_ratio>=0.05",
        "primary_blocker": "" if gate else "Simple prev-TTT-stable-anchor SWA gate variants did not produce robust ATE improvement.",
        "comparison_csv": str(ROOT / "stage7f_action_pilot_variant_comparison.csv"),
        "comparison_json": str(ROOT / "stage7f_action_pilot_variant_comparison.json"),
    }
    write_rows(out / "variant_summary_rows.csv", variant_rows)
    write_rows(out / "case_variant_rows.csv", [row for row in rows if isinstance(row, dict)])
    write_json(out / "summary.json", summary)
    write_text(
        out / "action_pilot_report.md",
        "# Stage7f Prev-TTT Anchor Gate Action Pilot\n\n"
        f"gate_pass={gate}\n\n"
        f"best_variant={best_variant}\n\n"
        f"best_variant_improved_ate_case_count={best_improved}\n\n"
        f"best_variant_worse_ate_case_count={best_worse}\n\n"
        f"best_variant_median_improvement_ratio_vs_baseline={best_median_impr}\n\n"
        "This is a 6-case action pilot, not a full validation run.  It does not authorize formal runtime success unless the conservative gate rule passes.",
    )
    failure_rows = [
        {"stage": "stage7f_action_pilot", "failed_check": "case_count>=6", "observed_value": best_case_count, "gate_pass": best_case_count >= 6},
        {"stage": "stage7f_action_pilot", "failed_check": "improved_cases>=4", "observed_value": best_improved, "gate_pass": best_improved >= 4},
        {"stage": "stage7f_action_pilot", "failed_check": "worse_cases<=2", "observed_value": best_worse, "gate_pass": best_worse <= 2},
        {"stage": "stage7f_action_pilot", "failed_check": "median_improvement_ratio>=0.05", "observed_value": best_median_impr, "gate_pass": best_median_impr >= 0.05},
    ]
    write_failure_common(
        out,
        stage_name="Stage7f prev-TTT stable-anchor gate action pilot",
        summary=summary,
        failure_rows=[row for row in failure_rows if not b(row.get("gate_pass"))],
        visual_rows=[
            {"artifact": "variant_summary_rows.csv", "path": str(out / "variant_summary_rows.csv"), "status": "generated_table", "source_metric": "aligned_ate_rmse_m"},
            {"artifact": "case_variant_rows.csv", "path": str(out / "case_variant_rows.csv"), "status": "generated_table", "source_metric": "aligned_ate_rmse_m"},
        ],
        what_would_text=(
            "# What Would Have To Be True To Pass\n\n"
            "A simple prev-TTT stable-anchor SWA gate would need >=6 cases, >=4 improved ATE cases, <=2 worsened cases, "
            "and median improvement ratio >=0.05 before it could be promoted beyond diagnostic pilot."
        ),
        next_route_text=(
            "# Next Route Recommendation\n\n"
            + (
                "Stage7f passed; promote only to a preregistered runtime validation with good controls."
                if gate
                else "Simple mask-level SWA gating is not robust.  Next repair should be more selective than aggregate stable-anchor mask gating, e.g. query-conditioned/id-specific risk suppression or a new plan."
            )
        ),
    )
    return summary


def stage7g_anchor_id_query_head_risk_attribution() -> dict[str, Any]:
    out = ROOT / "stage7g_anchor_id_query_head_risk_attribution"
    run_root = stage7e_anchor_id_hook_root()
    stage7e = read_json(ROOT / "stage7e_ttt_stable_anchor_id_hook/summary.json")
    stage7e_case_rows = read_rows(ROOT / "stage7e_ttt_stable_anchor_id_hook/anchor_id_case_rows.csv")
    stage1_rows = read_rows(ROOT / "stage1_trackK_swa_v2_strict_stable_eligibility/case_universe_rows.csv")
    meta_by_case: dict[str, dict[str, Any]] = {str(row.get("case_id", "")): row for row in stage1_rows if row.get("case_id")}
    meta_by_case.update({str(row.get("case_id", "")): row for row in stage7e_case_rows if row.get("case_id")})
    layer_rows: list[dict[str, Any]] = []
    layer_head_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []

    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        summary = {
            "schema": "acl2_v98_stage7g_anchor_id_query_head_risk_attribution_v1",
            "status": "blocked_missing_torch",
            "gate_pass": False,
            "runtime_action_allowed": False,
            "read_error_count": 1,
            "primary_blocker": f"torch import failed: {type(exc).__name__}:{exc}",
        }
        write_json(out / "summary.json", summary)
        write_rows(out / "read_errors.csv", [{"artifact_type": "torch_import", "error": summary["primary_blocker"]}])
        return summary

    def id_share_metrics(values: list[int]) -> dict[str, float]:
        if not values:
            return {
                "id_hit_count": 0.0,
                "id_unique_count": 0.0,
                "id_top1_share": math.nan,
                "id_top5_share": math.nan,
                "id_herfindahl": math.nan,
            }
        counts = Counter(values)
        total = float(sum(counts.values()))
        shares = sorted((count / total for count in counts.values()), reverse=True)
        return {
            "id_hit_count": total,
            "id_unique_count": float(len(shares)),
            "id_top1_share": shares[0],
            "id_top5_share": float(sum(shares[:5])),
            "id_herfindahl": float(sum(share * share for share in shares)),
        }

    for path in sorted(run_root.glob("*/TTT_SWA_SAME_RUN/swa_raw_transport_trace/*.pt")):
        case_id = path.parents[2].name
        try:
            payload = torch.load(path, map_location="cpu")
        except Exception as exc:  # noqa: BLE001
            error_rows.append({"artifact_type": "swa_raw_transport_trace", "case_id": case_id, "path": str(path), "error": f"{type(exc).__name__}:{exc}"})
            continue
        hit = payload.get("current_Q_to_cache_K_topk_ttt_prev_stable_anchor_hit_mask")
        ids = payload.get("current_Q_to_cache_K_topk_ttt_prev_stable_anchor_ids")
        if not torch.is_tensor(hit):
            error_rows.append({"artifact_type": "swa_raw_transport_trace", "case_id": case_id, "path": str(path), "error": "missing_hit_mask_tensor"})
            continue
        hit_bool = hit.detach().cpu().bool()
        if hit_bool.ndim != 4:
            error_rows.append({"artifact_type": "swa_raw_transport_trace", "case_id": case_id, "path": str(path), "error": f"unexpected_hit_mask_shape:{tuple(hit_bool.shape)}"})
            continue
        id_tensor = ids.detach().cpu().long() if torch.is_tensor(ids) and tuple(ids.shape) == tuple(hit_bool.shape) else None
        bsz, head_count, query_count, topk = hit_bool.shape
        hit_float = hit_bool.float()
        query_hit = hit_bool.any(dim=-1).float()  # B,H,Q
        per_head_query_hit = query_hit.mean(dim=(0, 2)) if bsz and query_count else torch.empty(0)
        per_head_topk_hit = hit_float.mean(dim=(0, 2, 3)) if bsz and query_count and topk else torch.empty(0)
        query_head_frac = query_hit.mean(dim=1).reshape(-1) if head_count else torch.empty(0)
        top10_count = max(1, math.ceil(0.10 * int(query_head_frac.numel()))) if int(query_head_frac.numel()) else 0
        top20_count = max(1, math.ceil(0.20 * int(query_head_frac.numel()))) if int(query_head_frac.numel()) else 0
        top10_mean = float(torch.topk(query_head_frac, top10_count).values.mean().item()) if top10_count else math.nan
        top20_mean = float(torch.topk(query_head_frac, top20_count).values.mean().item()) if top20_count else math.nan
        valid_ids: list[int] = []
        if id_tensor is not None:
            valid = hit_bool & (id_tensor >= 0)
            valid_ids = [int(value) for value in id_tensor[valid].reshape(-1).tolist()]
        id_metrics = id_share_metrics(valid_ids)
        mean_head_query_hit = float(per_head_query_hit.mean().item()) if int(per_head_query_hit.numel()) else math.nan
        max_head_query_hit = float(per_head_query_hit.max().item()) if int(per_head_query_hit.numel()) else math.nan
        layer_row = {
            "case_id": case_id,
            "chunk_idx": payload.get("chunk_idx", ""),
            "swa_layer_idx": payload.get("swa_layer_idx", ""),
            "path": str(path),
            "head_count": int(head_count),
            "sampled_query_count": int(query_head_frac.numel()),
            "topk": int(topk),
            "layer_anchor_topk_hit_frac": float(hit_float.mean().item()) if hit_bool.numel() else math.nan,
            "layer_anchor_query_hit_frac": mean_head_query_hit,
            "max_head_query_hit_frac": max_head_query_hit,
            "head_query_hit_concentration_ratio": max_head_query_hit / (mean_head_query_hit + EPS) if math.isfinite(max_head_query_hit) and math.isfinite(mean_head_query_hit) else math.nan,
            "max_query_head_hit_frac": float(query_head_frac.max().item()) if int(query_head_frac.numel()) else math.nan,
            "top10pct_query_head_hit_frac_mean": top10_mean,
            "top20pct_query_head_hit_frac_mean": top20_mean,
            "query_head_hit_ge50_frac": float((query_head_frac >= 0.50).float().mean().item()) if int(query_head_frac.numel()) else math.nan,
            "query_head_hit_ge75_frac": float((query_head_frac >= 0.75).float().mean().item()) if int(query_head_frac.numel()) else math.nan,
            "query_head_hit_ge90_frac": float((query_head_frac >= 0.90).float().mean().item()) if int(query_head_frac.numel()) else math.nan,
            "anchor_id_hit_count": id_metrics["id_hit_count"],
            "anchor_id_unique_count": id_metrics["id_unique_count"],
            "anchor_id_top1_share": id_metrics["id_top1_share"],
            "anchor_id_top5_share": id_metrics["id_top5_share"],
            "anchor_id_herfindahl": id_metrics["id_herfindahl"],
            "claim_scope": "offline_query_head_anchor_id_risk_attribution",
        }
        layer_rows.append(layer_row)

        for head_idx in range(head_count):
            head_valid_ids: list[int] = []
            if id_tensor is not None:
                head_valid = hit_bool[:, head_idx, :, :] & (id_tensor[:, head_idx, :, :] >= 0)
                head_valid_ids = [int(value) for value in id_tensor[:, head_idx, :, :][head_valid].reshape(-1).tolist()]
            head_id_metrics = id_share_metrics(head_valid_ids)
            layer_head_rows.append(
                {
                    "case_id": case_id,
                    "chunk_idx": payload.get("chunk_idx", ""),
                    "swa_layer_idx": payload.get("swa_layer_idx", ""),
                    "head_idx": head_idx,
                    "head_anchor_query_hit_frac": float(per_head_query_hit[head_idx].item()) if int(per_head_query_hit.numel()) else math.nan,
                    "head_anchor_topk_hit_frac": float(per_head_topk_hit[head_idx].item()) if int(per_head_topk_hit.numel()) else math.nan,
                    "head_anchor_id_hit_count": head_id_metrics["id_hit_count"],
                    "head_anchor_id_unique_count": head_id_metrics["id_unique_count"],
                    "head_anchor_id_top1_share": head_id_metrics["id_top1_share"],
                    "head_anchor_id_top5_share": head_id_metrics["id_top5_share"],
                    "head_anchor_id_herfindahl": head_id_metrics["id_herfindahl"],
                }
            )

    rows_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in layer_rows:
        rows_by_case[str(row.get("case_id", ""))].append(row)
    for case_id in sorted(rows_by_case):
        meta = meta_by_case.get(case_id, {})
        rows = rows_by_case[case_id]
        case_rows.append(
            {
                "case_id": case_id,
                "seq": meta.get("seq", case_id.split("_")[0]),
                "case_label": meta.get("case_label", ""),
                "good_control_hygiene_include_for_repair": meta.get("good_control_hygiene_include_for_repair", True),
                "payload_count": len(rows),
                "L3_handoff_transfer_penalty_proxy": f(meta.get("L3_handoff_transfer_penalty_proxy")),
                "anchor_id_topk_query_hit_frac_mean": f(meta.get("anchor_id_topk_query_hit_frac_mean")),
                "layer_anchor_query_hit_frac_mean": mean([f(row.get("layer_anchor_query_hit_frac")) for row in rows]),
                "max_layer_anchor_query_hit_frac": max(f(row.get("layer_anchor_query_hit_frac")) for row in rows),
                "max_layer_head_query_hit_frac": max(f(row.get("max_head_query_hit_frac")) for row in rows),
                "head_query_hit_concentration_ratio_max": max(f(row.get("head_query_hit_concentration_ratio")) for row in rows),
                "max_query_head_hit_frac": max(f(row.get("max_query_head_hit_frac")) for row in rows),
                "top10pct_query_head_hit_frac_mean": mean([f(row.get("top10pct_query_head_hit_frac_mean")) for row in rows]),
                "top20pct_query_head_hit_frac_mean": mean([f(row.get("top20pct_query_head_hit_frac_mean")) for row in rows]),
                "query_head_hit_ge50_frac_mean": mean([f(row.get("query_head_hit_ge50_frac")) for row in rows]),
                "query_head_hit_ge50_frac_max": max(f(row.get("query_head_hit_ge50_frac")) for row in rows),
                "query_head_hit_ge75_frac_mean": mean([f(row.get("query_head_hit_ge75_frac")) for row in rows]),
                "query_head_hit_ge75_frac_max": max(f(row.get("query_head_hit_ge75_frac")) for row in rows),
                "query_head_hit_ge90_frac_mean": mean([f(row.get("query_head_hit_ge90_frac")) for row in rows]),
                "anchor_id_top1_share_mean": mean([f(row.get("anchor_id_top1_share")) for row in rows]),
                "anchor_id_top1_share_max": max(f(row.get("anchor_id_top1_share")) for row in rows),
                "anchor_id_top5_share_mean": mean([f(row.get("anchor_id_top5_share")) for row in rows]),
                "anchor_id_top5_share_max": max(f(row.get("anchor_id_top5_share")) for row in rows),
                "anchor_id_herfindahl_mean": mean([f(row.get("anchor_id_herfindahl")) for row in rows]),
                "anchor_id_herfindahl_max": max(f(row.get("anchor_id_herfindahl")) for row in rows),
                "anchor_id_hit_count_sum": sum(f(row.get("anchor_id_hit_count"), 0.0) for row in rows),
                "anchor_id_unique_count_sum": sum(f(row.get("anchor_id_unique_count"), 0.0) for row in rows),
                "candidate_action_policy": "suppress_or_downweight_prev_ttt_anchor_reads_only_for_queries_where_ge75pct_heads_hit_prev_anchor_ids",
                "candidate_action_query_frac_proxy": mean([f(row.get("query_head_hit_ge75_frac")) for row in rows]),
                "claim_scope": "offline_query_head_anchor_id_risk_attribution",
            }
        )

    specs = {
        "query_ge90_frac_higher_bad": ("query_head_hit_ge90_frac_mean", True),
        "query_ge75_frac_higher_bad": ("query_head_hit_ge75_frac_mean", True),
        "query_ge50_frac_higher_bad": ("query_head_hit_ge50_frac_mean", True),
        "top10pct_query_head_hit_higher_bad": ("top10pct_query_head_hit_frac_mean", True),
        "top20pct_query_head_hit_higher_bad": ("top20pct_query_head_hit_frac_mean", True),
        "max_layer_head_query_hit_higher_bad": ("max_layer_head_query_hit_frac", True),
        "max_layer_query_hit_higher_bad": ("max_layer_anchor_query_hit_frac", True),
        "head_query_concentration_higher_bad": ("head_query_hit_concentration_ratio_max", True),
        "id_top1_share_higher_bad": ("anchor_id_top1_share_max", True),
        "id_top1_share_lower_bad": ("anchor_id_top1_share_max", False),
        "id_top5_share_higher_bad": ("anchor_id_top5_share_max", True),
        "id_top5_share_lower_bad": ("anchor_id_top5_share_max", False),
        "id_herfindahl_higher_bad": ("anchor_id_herfindahl_max", True),
        "id_herfindahl_lower_bad": ("anchor_id_herfindahl_max", False),
    }
    cue_rows = evaluate_cues(case_rows, specs, min_cases=12, require_direction=True, view_name="stage7g_query_head_id_attribution")
    hygiene_cue_rows = evaluate_cues(
        case_rows,
        specs,
        min_cases=12,
        require_direction=True,
        row_filter=lambda row: b(row.get("good_control_hygiene_include_for_repair", True)),
        view_name="stage7g_query_head_id_attribution_hygiene_repair",
    )
    all_cue_rows = cue_rows + hygiene_cue_rows
    gate_cues = [row for row in all_cue_rows if b(row.get("gate_pass"))]
    best_cue = gate_cues[0] if gate_cues else (all_cue_rows[0] if all_cue_rows else {})
    values = {str(row["case_id"]): f(row.get(str(best_cue.get("field", "")))) for row in case_rows if best_cue}
    selected = selected_from_threshold(values, f(best_cue.get("threshold")), higher_bad=str(best_cue.get("direction")) == "higher_bad") if best_cue else set()
    selected_rows = [row for row in case_rows if str(row.get("case_id", "")) in selected]
    selected_ge75_mean = mean([f(row.get("candidate_action_query_frac_proxy")) for row in selected_rows])
    selected_ge75_median = median([f(row.get("candidate_action_query_frac_proxy")) for row in selected_rows])
    selected_ge50_mean = mean([f(row.get("query_head_hit_ge50_frac_mean")) for row in selected_rows])
    selected_case_count = len(selected_rows)
    sequence_coverage = len({str(row.get("seq", "")) for row in case_rows})
    id_specific_gate_pass = any(b(row.get("gate_pass")) and str(row.get("cue_name", "")).startswith("id_") for row in all_cue_rows)
    query_head_gate_pass = any(
        b(row.get("gate_pass")) and (
            str(row.get("cue_name", "")).startswith("query_")
            or str(row.get("cue_name", "")).startswith("top")
            or str(row.get("cue_name", "")).startswith("max_layer")
        )
        for row in all_cue_rows
    )
    selective_action_mass_gate_pass = (
        selected_case_count > 0
        and math.isfinite(selected_ge75_mean)
        and selected_ge75_mean <= 0.30
        and math.isfinite(selected_ge75_median)
        and selected_ge75_median <= 0.30
    )
    diagnostic_metric_gate_pass = bool(gate_cues)
    gate = (
        b(stage7e.get("gate_pass"))
        and len(case_rows) >= 12
        and sequence_coverage >= 3
        and diagnostic_metric_gate_pass
        and query_head_gate_pass
        and selective_action_mass_gate_pass
    )
    summary = {
        "schema": "acl2_v98_stage7g_anchor_id_query_head_risk_attribution_v1",
        "status": "complete_selective_query_head_cue_found" if gate else "complete_selective_attribution_no_go",
        "gate_pass": gate,
        "runtime_action_allowed": False,
        "claim_level": "SELECTIVE_QUERY_HEAD_RISK_CUE_FOUND_ACTION_NOT_RUN" if gate else "QUERY_HEAD_ID_ATTRIBUTION_DIAGNOSTIC_ONLY",
        "run_root": str(run_root),
        "stage7e_prerequisite_gate_pass": b(stage7e.get("gate_pass")),
        "case_count": len(case_rows),
        "sequence_coverage": sequence_coverage,
        "layer_row_count": len(layer_rows),
        "layer_head_row_count": len(layer_head_rows),
        "read_error_count": len(error_rows),
        "diagnostic_metric_gate_pass": diagnostic_metric_gate_pass,
        "query_head_gate_pass": query_head_gate_pass,
        "id_specific_risk_cue_gate_pass": id_specific_gate_pass,
        "selective_action_mass_gate_pass": selective_action_mass_gate_pass,
        "selective_action_mass_rule": "selected_case_mean_and_median_query_head_hit_ge75_frac<=0.30",
        "best_cue": best_cue.get("cue_name", ""),
        "best_cue_direction": best_cue.get("direction", ""),
        "best_cue_field": best_cue.get("field", ""),
        "best_cue_threshold": best_cue.get("threshold", ""),
        "best_cue_bad_recall": best_cue.get("bad_recall", ""),
        "best_cue_good_FPR": best_cue.get("good_FPR", ""),
        "best_cue_abs_corr_L3": best_cue.get("abs_corr_L3_handoff_transfer_penalty", ""),
        "best_cue_true_positive_cases": best_cue.get("true_positive_cases", ""),
        "best_cue_false_positive_cases": best_cue.get("false_positive_cases", ""),
        "selected_case_count": selected_case_count,
        "selected_case_candidate_action_query_ge75_frac_mean": selected_ge75_mean,
        "selected_case_candidate_action_query_ge75_frac_median": selected_ge75_median,
        "selected_case_query_head_hit_ge50_frac_mean": selected_ge50_mean,
        "case_query_head_hit_ge75_frac_mean": mean([f(row.get("query_head_hit_ge75_frac_mean")) for row in case_rows]),
        "case_query_head_hit_ge50_frac_mean": mean([f(row.get("query_head_hit_ge50_frac_mean")) for row in case_rows]),
        "case_top10pct_query_head_hit_frac_mean": mean([f(row.get("top10pct_query_head_hit_frac_mean")) for row in case_rows]),
        "case_max_layer_head_query_hit_frac_mean": mean([f(row.get("max_layer_head_query_hit_frac")) for row in case_rows]),
        "primary_blocker": (
            "Stage7g found a query/head selective diagnostic cue after Stage7f No-Go, but no query-conditioned runtime action has been implemented or validated."
            if gate
            else "Stage7g did not establish a good-control-safe query/head selective action candidate with <=0.30 candidate query mass."
        ),
    }
    write_rows(out / "case_rows.csv", case_rows)
    write_rows(out / "layer_rows.csv", layer_rows)
    write_rows(out / "layer_head_rows.csv", layer_head_rows)
    write_rows(out / "cue_control_metrics.csv", cue_rows)
    write_rows(out / "cue_control_metrics_hygiene_repair.csv", hygiene_cue_rows)
    write_rows(out / "read_errors.csv", error_rows)
    write_json(out / "summary.json", summary)
    write_text(
        out / "query_head_risk_attribution_report.md",
        "# Stage7g Query/Head Anchor-Id Risk Attribution\n\n"
        f"gate_pass={gate}\n\n"
        f"best_cue={summary['best_cue']}\n\n"
        f"selected_case_candidate_action_query_ge75_frac_mean={selected_ge75_mean}\n\n"
        f"selected_case_candidate_action_query_ge75_frac_median={selected_ge75_median}\n\n"
        "This is an offline diagnostic attribution stage. It does not claim a runtime action result or Stage9 validation.",
    )
    failure_rows = [
        {"stage": "stage7g_query_head_id_attribution", "failed_check": "stage7e_prerequisite_gate_pass", "observed_value": b(stage7e.get("gate_pass")), "gate_pass": b(stage7e.get("gate_pass"))},
        {"stage": "stage7g_query_head_id_attribution", "failed_check": "case_count>=12", "observed_value": len(case_rows), "gate_pass": len(case_rows) >= 12},
        {"stage": "stage7g_query_head_id_attribution", "failed_check": "sequence_coverage>=3", "observed_value": sequence_coverage, "gate_pass": sequence_coverage >= 3},
        {"stage": "stage7g_query_head_id_attribution", "failed_check": "diagnostic_metric_gate_pass", "observed_value": diagnostic_metric_gate_pass, "gate_pass": diagnostic_metric_gate_pass},
        {"stage": "stage7g_query_head_id_attribution", "failed_check": "query_head_gate_pass", "observed_value": query_head_gate_pass, "gate_pass": query_head_gate_pass},
        {"stage": "stage7g_query_head_id_attribution", "failed_check": "selected_ge75_mean<=0.30", "observed_value": selected_ge75_mean, "gate_pass": math.isfinite(selected_ge75_mean) and selected_ge75_mean <= 0.30},
        {"stage": "stage7g_query_head_id_attribution", "failed_check": "selected_ge75_median<=0.30", "observed_value": selected_ge75_median, "gate_pass": math.isfinite(selected_ge75_median) and selected_ge75_median <= 0.30},
    ]
    write_failure_common(
        out,
        stage_name="Stage7g query/head anchor-id risk attribution",
        summary=summary,
        failure_rows=[row for row in failure_rows if not b(row.get("gate_pass"))] + cue_failure_attribution(cue_rows, "stage7g_query_head_id_attribution") + cue_failure_attribution(hygiene_cue_rows, "stage7g_query_head_id_attribution_hygiene_repair"),
        visual_rows=[
            {"artifact": "case_rows.csv", "path": str(out / "case_rows.csv"), "status": "generated_table", "source_metric": "query_head_anchor_id_hit_concentration"},
            {"artifact": "layer_head_rows.csv", "path": str(out / "layer_head_rows.csv"), "status": "generated_table", "source_metric": "per_head_anchor_id_hits"},
        ],
        what_would_text=(
            "# What Would Have To Be True To Pass\n\n"
            "A Stage7g selective cue needs Stage7e to pass, >=12 cases over >=3 sequences, a good-control-safe query/head cue with correct L3 direction, "
            "and a bounded candidate action mass: selected-case mean and median query_head_hit_ge75 fraction <=0.30."
        ),
        next_route_text=(
            "# Next Route Recommendation\n\n"
            + (
                "Implement only a query-conditioned runtime pilot that touches prev-TTT anchor reads for queries where >=75% heads hit previous stable-anchor ids; do not return to aggregate stable-anchor mask gating."
                if gate
                else "Do not implement query-conditioned action yet; the offline attribution did not produce a sufficiently selective good-control-safe cue."
            )
        ),
    )
    return summary


def stage7h_prev_ttt_anchor_query_soft_action_pilot() -> dict[str, Any]:
    out = ROOT / "stage7h_prev_ttt_anchor_query_soft_action_pilot"
    comparison_paths = [
        ROOT / "stage7h_query_soft_r05_last_6case_comparison.json",
        ROOT / "stage7h_query_soft_r05_last_ge90_6case_comparison.json",
        ROOT / "stage7h_query_soft_r05_all_ge90_6case_comparison.json",
        ROOT / "stage7h_query_soft_r05_last_ge90_stage7f_cases_6case_comparison.json",
        ROOT / "stage7h_query_soft_r05_last_ge75_stage7f_cases_6case_comparison.json",
        ROOT / "stage7h_query_soft_r025_last_ge90_stage7f_cases_6case_comparison.json",
    ]
    comparisons = []
    for path in comparison_paths:
        comparison = read_json(path)
        payload = comparison.get("summary", {}) if isinstance(comparison.get("summary"), dict) else {}
        rows = comparison.get("rows", []) if isinstance(comparison.get("rows"), list) else []
        if payload:
            comparisons.append({"path": path, "payload": payload, "rows": rows})
    if not comparisons:
        summary = {
            "schema": "acl2_v98_stage7h_prev_ttt_anchor_query_soft_action_pilot_v1",
            "status": "not_run",
            "runtime_action_pilot_run": False,
            "formal_runtime_action_run": False,
            "full_validation_run": False,
            "gate_pass": False,
            "case_count": 0,
            "primary_blocker": "No Stage7h query-soft action-pilot comparison artifact found.",
        }
        write_json(out / "summary.json", summary)
        return summary

    def score_variant(item: dict[str, Any]) -> tuple[float, float, float]:
        payload = item.get("payload", {})
        return (
            f(payload.get("median_improvement_ratio_vs_baseline"), -999.0),
            f(payload.get("mean_improvement_ratio_vs_baseline"), -999.0),
            float(payload.get("improved_ate_case_count", 0) or 0),
        )

    best = max(comparisons, key=score_variant)
    payload = best["payload"]
    rows = best["rows"]
    case_count = int(payload.get("case_count", 0) or 0)
    failed_jobs = int(payload.get("failed_job_count", 0) or 0)
    improved = int(payload.get("improved_ate_case_count", 0) or 0)
    worse = int(payload.get("worse_ate_case_count", 0) or 0)
    median_impr = f(payload.get("median_improvement_ratio_vs_baseline"), math.nan)
    gate = bool(case_count >= 6 and failed_jobs == 0 and improved >= 4 and worse <= 2 and median_impr >= 0.05)
    summary = {
        "schema": "acl2_v98_stage7h_prev_ttt_anchor_query_soft_action_pilot_v1",
        "status": "complete_action_pilot_gate_pass" if gate else "complete_action_pilot_no_go",
        "runtime_action_pilot_run": True,
        "formal_runtime_action_run": False,
        "full_validation_run": False,
        "gate_pass": gate,
        "variant": payload.get("variant", "query_soft_r05_last_ge75_mass32"),
        "case_count": case_count,
        "failed_job_count": failed_jobs,
        "improved_ate_case_count": improved,
        "worse_ate_case_count": worse,
        "same_ate_case_count": int(payload.get("same_ate_case_count", 0) or 0),
        "mean_action_minus_baseline_ate_rmse_m": f(payload.get("mean_action_minus_baseline_ate_rmse_m"), math.nan),
        "median_action_minus_baseline_ate_rmse_m": f(payload.get("median_action_minus_baseline_ate_rmse_m"), math.nan),
        "mean_improvement_ratio_vs_baseline": f(payload.get("mean_improvement_ratio_vs_baseline"), math.nan),
        "median_improvement_ratio_vs_baseline": median_impr,
        "mean_action_minus_baseline_final_error_m": f(payload.get("mean_action_minus_baseline_final_error_m"), math.nan),
        "median_action_minus_baseline_final_error_m": f(payload.get("median_action_minus_baseline_final_error_m"), math.nan),
        "gate_rule": payload.get(
            "gate_rule",
            "case_count>=6 and failed_job_count==0 and improved_cases>=4 and worse_cases<=2 and median_improvement_ratio>=0.05",
        ),
        "comparison_csv": ";".join(str(path.with_suffix(".csv")) for path in comparison_paths if path.is_file()),
        "comparison_json": ";".join(str(item["path"]) for item in comparisons),
        "primary_blocker": "" if gate else "Query-conditioned prev-TTT anchor soft-bias action did not produce robust ATE improvement.",
    }
    write_rows(out / "case_rows.csv", [row for row in rows if isinstance(row, dict)])
    variant_rows = []
    for item in comparisons:
        payload_i = item["payload"]
        variant_rows.append({"comparison_json": str(item["path"]), **payload_i})
    write_rows(out / "variant_summary_rows.csv", variant_rows)
    write_json(out / "summary.json", summary)
    write_text(
        out / "action_pilot_report.md",
        "# Stage7h Prev-TTT Anchor Query-Soft Action Pilot\n\n"
        f"gate_pass={gate}\n\n"
        f"variant={summary['variant']}\n\n"
        f"improved_ate_case_count={improved}\n\n"
        f"worse_ate_case_count={worse}\n\n"
        f"median_improvement_ratio_vs_baseline={median_impr}\n\n"
        "This is a 6-case action pilot, not Stage9 full validation. It only tests the query-conditioned action suggested by Stage7g.",
    )
    failure_rows = [
        {"stage": "stage7h_query_soft_action_pilot", "failed_check": "case_count>=6", "observed_value": case_count, "gate_pass": case_count >= 6},
        {"stage": "stage7h_query_soft_action_pilot", "failed_check": "failed_job_count==0", "observed_value": failed_jobs, "gate_pass": failed_jobs == 0},
        {"stage": "stage7h_query_soft_action_pilot", "failed_check": "improved_cases>=4", "observed_value": improved, "gate_pass": improved >= 4},
        {"stage": "stage7h_query_soft_action_pilot", "failed_check": "worse_cases<=2", "observed_value": worse, "gate_pass": worse <= 2},
        {"stage": "stage7h_query_soft_action_pilot", "failed_check": "median_improvement_ratio>=0.05", "observed_value": median_impr, "gate_pass": median_impr >= 0.05},
    ]
    write_failure_common(
        out,
        stage_name="Stage7h prev-TTT anchor query-soft action pilot",
        summary=summary,
        failure_rows=[row for row in failure_rows if not b(row.get("gate_pass"))],
        visual_rows=[
            {"artifact": "case_rows.csv", "path": str(out / "case_rows.csv"), "status": "generated_table", "source_metric": "aligned_ate_rmse_m"},
            *[
                {
                    "artifact": path.with_suffix(".csv").name,
                    "path": str(path.with_suffix(".csv")),
                    "status": "generated_table",
                    "source_metric": "aligned_ate_rmse_m",
                }
                for path in comparison_paths
                if path.with_suffix(".csv").is_file()
            ],
        ],
        what_would_text=(
            "# What Would Have To Be True To Pass\n\n"
            "Stage7h needs >=6 cases, zero failed jobs, >=4 improved ATE cases, <=2 worsened ATE cases, and median improvement ratio >=0.05."
        ),
        next_route_text=(
            "# Next Route Recommendation\n\n"
            + (
                "Stage7h passed; promote only to preregistered validation with good controls."
                if gate
                else "The query-soft action variants are No-Go after ge75, ge90, high-cue case selection, and reduced-rho repair. Do not return to aggregate stable-anchor mask gating; a future route needs a new id-specific/causal action surface rather than further broad stable-anchor suppression."
            )
        ),
    )
    return summary


def stage8_semantic_region_atlas() -> dict[str, Any]:
    out = ROOT / "stage8_trackJ_semantic_region_memory_impact_atlas"
    s1_rows = read_rows(ROOT / "stage1_trackK_swa_v2_strict_stable_eligibility/case_universe_rows.csv")
    atlas_rows: list[dict[str, Any]] = []
    for row in s1_rows:
        if int(float(row.get("payload_count", 0) or 0)) <= 0:
            continue
        regions = {
            "strict_stable_anchor": f(row.get("strict_stable_nonempty_frac"), 0.0),
            "dynamic_region": f(row.get("semantic_dynamic_region_mass"), 0.0),
            "object_boundary": f(row.get("semantic_object_boundary_mass"), 0.0),
            "low_observability": f(row.get("semantic_low_observability_score"), 0.0),
            "weak_context": f(row.get("semantic_context_mass"), 0.0),
        }
        dominant = max(regions, key=regions.get)
        atlas_rows.append(
            {
                "case_id": row.get("case_id"),
                "seq": row.get("seq"),
                "memory_body": "SWA",
                "dominant_region_role": dominant,
                "dominant_region_score": regions[dominant],
                "case_label": row.get("case_label"),
                "diagnostic_only": True,
                "causal_effect_promoted": False,
            }
        )
    summary = {
        "schema": "acl2_v98_stage8_semantic_region_memory_impact_atlas_v1",
        "status": "complete_diagnostic_atlas_only",
        "gate_pass": False,
        "runtime_action_allowed": False,
        "atlas_row_count": len(atlas_rows),
        "memory_bodies_with_rows": sorted({row["memory_body"] for row in atlas_rows}),
        "primary_blocker": "Atlas rows are diagnostic SWA region associations; no region has memory-specific causal effect beyond controls.",
    }
    write_json(out / "summary.json", summary)
    write_rows(out / "semantic_region_memory_impact_rows.csv", atlas_rows)
    write_failure_common(
        out,
        stage_name="Stage8 TrackJ semantic region memory impact atlas",
        summary=summary,
        failure_rows=[{"stage": "stage8_j", "failed_check": "causal_memory_specific_region_effect", "observed_value": False, "gate_pass": False}],
        visual_rows=[{"artifact": "semantic_region_memory_impact_rows.csv", "path": str(out / "semantic_region_memory_impact_rows.csv"), "status": "generated_table", "source_metric": "dominant_region_role"}],
        what_would_text=(
            "# What Would Have To Be True To Pass\n\n"
            "Track J v2 would need at least one semantic region with memory-specific causal or diagnostic effect, not reproduced by same-mass random, aligned with the correct L1/L2/L3/L4 metric, and visual projection checks."
        ),
        next_route_text="# Next Route Recommendation\n\nUse this atlas only to prioritize deeper instrumentation; do not treat region association as a causal memory action.",
    )
    return summary


def final_decision(stage0_summary: dict[str, Any], s1: dict[str, Any], s2: dict[str, Any], s3: dict[str, Any], downstream: dict[str, Any]) -> dict[str, Any]:
    out = ROOT / "final_decision"
    s2b = read_json(ROOT / "stage2b_trackL_trace_semantic_anchor_expansion/summary.json")
    s7b = read_json(ROOT / "stage7b_trackF3_ttt_write_to_swa_usage_proxy/summary.json")
    s7c = read_json(ROOT / "stage7c_ttt_swa_same_run_alignment/summary.json")
    s7d = read_json(ROOT / "stage7d_ttt_swa_spatial_token_proxy_identity/summary.json")
    s7e = read_json(ROOT / "stage7e_ttt_stable_anchor_id_hook/summary.json")
    s7f = read_json(ROOT / "stage7f_prev_ttt_anchor_gate_action_pilot/summary.json")
    s7g = read_json(ROOT / "stage7g_anchor_id_query_head_risk_attribution/summary.json")
    s7h = read_json(ROOT / "stage7h_prev_ttt_anchor_query_soft_action_pilot/summary.json")
    summary = {
        "schema": "acl2_v98_final_decision_v1",
        "status": "complete",
        "full_method_success": False,
        "method_success": False,
        "runtime_action_run": False,
        "runtime_action_pilot_run": b(s7f.get("runtime_action_pilot_run")) or b(s7h.get("runtime_action_pilot_run")),
        "full_validation_run": False,
        "stage0_gate_pass": b(stage0_summary.get("gate_pass")),
        "stage1_trackK_gate_pass": b(s1.get("gate_pass")),
        "stage1_trackK_raw_gate_pass": b(s1.get("raw_gate_pass")),
        "stage1_trackK_hygiene_repair_gate_pass": b(s1.get("hygiene_repair_gate_pass")),
        "stage1_trackK_gate_pass_view": s1.get("gate_pass_view", "none"),
        "stage2_trackL_gate_pass": b(s2.get("gate_pass")),
        "stage2_trackL_raw_gate_pass": b(s2.get("raw_gate_pass")),
        "stage2_trackL_hygiene_repair_gate_pass": b(s2.get("hygiene_repair_gate_pass")),
        "stage2_trackL_gate_pass_view": s2.get("gate_pass_view", "none"),
        "stage2b_trace_semantic_anchor_gate_pass": b(s2b.get("gate_pass")),
        "stage2b_trace_semantic_anchor_raw_gate_pass": b(s2b.get("raw_gate_pass")),
        "stage2b_trace_semantic_anchor_hygiene_repair_gate_pass": b(s2b.get("hygiene_repair_gate_pass")),
        "stage2b_trace_semantic_anchor_gate_pass_view": s2b.get("gate_pass_view", "none"),
        "stage3_trackM_gate_pass": b(s3.get("gate_pass")),
        "stage7b_ttt_write_to_swa_usage_proxy_gate_pass": b(s7b.get("gate_pass")),
        "stage7b_true_anchor_identity_available": b(s7b.get("true_anchor_identity_available")),
        "stage7b_proxy_case_count": s7b.get("proxy_case_count", ""),
        "stage7c_ttt_swa_same_run_gate_pass": b(s7c.get("gate_pass")),
        "stage7c_case_with_both_count": s7c.get("case_with_both_count", ""),
        "stage7c_swa_strict_stable_nonempty_case_count": s7c.get("swa_strict_stable_nonempty_case_count", ""),
        "stage7d_ttt_swa_spatial_token_proxy_gate_pass": b(s7d.get("gate_pass")),
        "stage7d_spatial_token_proxy_available": b(s7d.get("spatial_token_proxy_available")),
        "stage7d_case_count": s7d.get("case_count", ""),
        "stage7d_prev_topk_hit_frac_mean": s7d.get("prev_topk_hit_frac_mean", ""),
        "stage7e_ttt_stable_anchor_id_hook_gate_pass": b(s7e.get("gate_pass")),
        "stage7e_f3_write_to_use_risk_cue_shown": b(s7e.get("f3_write_to_use_risk_cue_shown")),
        "stage7e_persistent_anchor_id_available": b(s7e.get("persistent_anchor_id_available")),
        "stage7e_write_to_use_chain_available": b(s7e.get("write_to_use_chain_available")),
        "stage7e_case_count": s7e.get("case_count", ""),
        "stage7e_anchor_id_topk_hit_frac_mean": s7e.get("anchor_id_topk_hit_frac_mean", ""),
        "stage7f_prev_ttt_anchor_gate_action_pilot_run": b(s7f.get("runtime_action_pilot_run")),
        "stage7f_prev_ttt_anchor_gate_action_pilot_gate_pass": b(s7f.get("gate_pass")),
        "stage7f_best_variant": s7f.get("best_variant", ""),
        "stage7f_best_variant_improved_ate_case_count": s7f.get("best_variant_improved_ate_case_count", ""),
        "stage7f_best_variant_worse_ate_case_count": s7f.get("best_variant_worse_ate_case_count", ""),
        "stage7f_best_variant_median_improvement_ratio_vs_baseline": s7f.get("best_variant_median_improvement_ratio_vs_baseline", ""),
        "stage7g_anchor_id_query_head_risk_attribution_gate_pass": b(s7g.get("gate_pass")),
        "stage7g_query_head_gate_pass": b(s7g.get("query_head_gate_pass")),
        "stage7g_id_specific_risk_cue_gate_pass": b(s7g.get("id_specific_risk_cue_gate_pass")),
        "stage7g_selective_action_mass_gate_pass": b(s7g.get("selective_action_mass_gate_pass")),
        "stage7g_best_cue": s7g.get("best_cue", ""),
        "stage7g_selected_case_candidate_action_query_ge75_frac_mean": s7g.get("selected_case_candidate_action_query_ge75_frac_mean", ""),
        "stage7g_selected_case_candidate_action_query_ge75_frac_median": s7g.get("selected_case_candidate_action_query_ge75_frac_median", ""),
        "stage7h_prev_ttt_anchor_query_soft_action_pilot_run": b(s7h.get("runtime_action_pilot_run")),
        "stage7h_prev_ttt_anchor_query_soft_action_pilot_gate_pass": b(s7h.get("gate_pass")),
        "stage7h_variant": s7h.get("variant", ""),
        "stage7h_improved_ate_case_count": s7h.get("improved_ate_case_count", ""),
        "stage7h_worse_ate_case_count": s7h.get("worse_ate_case_count", ""),
        "stage7h_median_improvement_ratio_vs_baseline": s7h.get("median_improvement_ratio_vs_baseline", ""),
        "final_taxonomy": downstream.get("final_taxonomy", "NO_SIGNAL"),
        "answers": {
            "swa_cache_topk_eligibility_generalized": b(s1.get("gate_pass")),
            "semantic_scale_observability_valid": b(s2.get("gate_pass")) or b(s2b.get("gate_pass")),
            "e3_action_improved_L3_by_5pct": False,
            "c2_predicts_global_gauge_harm": False,
            "f3_write_to_use_failure_shown": (
                b(s7b.get("f3_write_to_use_failure_shown"))
                or b(s7d.get("f3_write_to_use_failure_shown"))
                or b(s7e.get("f3_write_to_use_failure_shown"))
            ),
            "f3_write_to_use_risk_cue_shown": b(s7e.get("f3_write_to_use_risk_cue_shown")),
            "f3_prev_ttt_anchor_gate_action_pilot_pass": b(s7f.get("gate_pass")),
            "f3_query_head_selective_risk_cue_shown": b(s7g.get("gate_pass")),
            "f3_query_soft_action_pilot_pass": b(s7h.get("gate_pass")),
            "read_pairwise_action_beat_dense_L07_or_DGQ90": False,
            "full_validation_allowed": False,
        },
        "primary_blocker": downstream.get("primary_blocker", ""),
    }
    write_json(out / "final_decision.json", summary)
    write_json(out / "summary.json", summary)
    write_text(
        out / "final_report.md",
        "# ACL2 v98-TF Final Report\n\n"
        f"Final taxonomy: `{summary['final_taxonomy']}`.\n\n"
        f"Stage1 Track K-SWA gate pass: `{summary['stage1_trackK_gate_pass']}`.\n\n"
        f"Stage2 Track L gate pass: `{summary['stage2_trackL_gate_pass']}`.\n\n"
        f"Stage2b trace-semantic anchor gate pass: `{summary['stage2b_trace_semantic_anchor_gate_pass']}`.\n\n"
        f"Stage3 Track M gate pass: `{summary['stage3_trackM_gate_pass']}`.\n\n"
        f"Stage7b TTT write-to-SWA proxy gate pass: `{summary['stage7b_ttt_write_to_swa_usage_proxy_gate_pass']}`.\n\n"
        f"Stage7c same-run TTT/SWA gate pass: `{summary['stage7c_ttt_swa_same_run_gate_pass']}`.\n\n"
        f"Stage7d spatial-token proxy gate pass: `{summary['stage7d_ttt_swa_spatial_token_proxy_gate_pass']}`.\n\n"
        f"Stage7e state-carried anchor-id hook gate pass: `{summary['stage7e_ttt_stable_anchor_id_hook_gate_pass']}`.\n\n"
        f"Stage7f prev-TTT anchor gate action pilot pass: `{summary['stage7f_prev_ttt_anchor_gate_action_pilot_gate_pass']}`.\n\n"
        f"Stage7g query/head anchor-id attribution pass: `{summary['stage7g_anchor_id_query_head_risk_attribution_gate_pass']}`.\n\n"
        f"Stage7h query-soft action pilot pass: `{summary['stage7h_prev_ttt_anchor_query_soft_action_pilot_gate_pass']}`.\n\n"
        "No E3 L3 runtime action result or Stage9 full validation is claimed by this report.",
    )
    write_text(
        out / "failure_report.md",
        "# Failure / Blocker Report\n\n"
        f"{summary['primary_blocker'] or 'No full-method success was established.'}\n\n"
        "Negative or blocked branches remain diagnostic and must not be promoted.",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--extension-root",
        type=Path,
        action="append",
        default=None,
        help="Extension trace root. Repeat to merge raw and repair roots. Defaults to the primary root plus hygiene repair root if present.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    extension_roots = args.extension_root if args.extension_root else default_extension_roots()
    s0 = stage0()
    s1 = stage1(extension_roots)
    s2 = stage2()
    stage2b_trace_semantic_anchor_expansion(extension_roots)
    s3 = stage3()
    stage5_c2_diagnostic_replay_ledger()
    stage6_h2_d3_read_ledger()
    stage7_f3_ttt_ledger()
    stage7b_ttt_write_to_swa_usage_proxy()
    stage7c_ttt_swa_same_run_ledger()
    stage7d_ttt_swa_spatial_token_proxy_identity()
    stage7e_ttt_stable_anchor_id_hook()
    stage7f_prev_ttt_anchor_gate_action_pilot()
    stage7g_anchor_id_query_head_risk_attribution()
    stage7h_prev_ttt_anchor_query_soft_action_pilot()
    stage8_semantic_region_atlas()
    down = downstream_stages()
    final_decision(s0, s1, s2, s3, down)


if __name__ == "__main__":
    main()
