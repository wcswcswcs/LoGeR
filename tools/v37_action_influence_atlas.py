#!/usr/bin/env python3
"""Build v37 Track 0 action/influence atlas from landed smoke rollouts.

The script only aggregates landed artifacts.  It does not reconstruct missing
attention tensors or fabricate per-label/masklet action masses.  Missing fields
are written explicitly as explainability_missing so downstream decisions do not
quietly overclaim what the run recorded.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


PATHS = ["frame_attention", "chunk_attention", "swa_read", "ttt_apply"]
ROLE_FIELDS = {
    "frame_attention": "R_frame_tok",
    "chunk_attention": "R_global_tok",
    "swa_read": "R_swa_tok",
    "ttt_apply": "R_ttt_tok",
}


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _clean(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _status_done(run_dir: Path, run_name: str) -> bool:
    status = run_dir / "run_status.txt"
    if not status.exists():
        return False
    return f"DONE {run_name}" in status.read_text(encoding="utf-8", errors="replace")


def _sum_role_counts(rows: Sequence[Dict[str, Any]], role_field: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for row in rows:
        path_counts = row.get("path_role_counts")
        if not isinstance(path_counts, dict):
            continue
        counts = path_counts.get(role_field)
        if not isinstance(counts, dict):
            continue
        for key, value in counts.items():
            out[str(key)] = out.get(str(key), 0.0) + float(value or 0.0)
    return out


def _role_frac(counts: Dict[str, float], role: str) -> float:
    total = sum(float(v) for v in counts.values())
    if total <= 0:
        return 0.0
    return float(counts.get(role, 0.0)) / total


def _role_dist(counts: Dict[str, float]) -> Dict[str, float]:
    total = sum(float(v) for v in counts.values())
    if total <= 0:
        return {}
    return {k: float(v) / total for k, v in counts.items()}


def _last_by_path(rows: Sequence[Dict[str, Any]], path_name: str) -> Dict[str, Any]:
    selected = [row for row in rows if row.get("path") == path_name]
    return selected[-1] if selected else {}


def _mean(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return float("nan")
    return sum(vals) / len(vals)


def _max(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return float("nan")
    return max(vals)


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(out):
        return default
    return out


def _latest_hmc(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return rows[-1] if rows else {}


def _hook_values(rows: Sequence[Dict[str, Any]], path_name: str, key: str) -> List[float]:
    vals: List[float] = []
    for row in rows:
        summary = row.get("hook_effect_summary")
        if not isinstance(summary, dict):
            summary = row.get("control_trace", {}).get("hook_effect_summary") if isinstance(row.get("control_trace"), dict) else {}
        if not isinstance(summary, dict):
            continue
        path = summary.get(path_name)
        if isinstance(path, dict) and path.get(key) is not None:
            try:
                vals.append(float(path.get(key)))
            except (TypeError, ValueError):
                pass
    return vals


def _weighted_jaccard(a: Sequence[float], b: Sequence[float]) -> float:
    num = 0.0
    den = 0.0
    for x, y in zip(a, b):
        x = max(0.0, float(x))
        y = max(0.0, float(y))
        num += min(x, y)
        den += max(x, y)
    if den <= 0:
        return 1.0
    return num / den


def _feature_vector(row: Dict[str, Any]) -> List[float]:
    return [
        float(row.get("frame_skip_ratio") or 0.0),
        float(row.get("global_skip_ratio") or 0.0),
        float(row.get("frame_influence_mass") or 0.0),
        float(row.get("global_influence_mass") or 0.0),
        float(row.get("swa_source_gate_applied_ratio_proxy") or 0.0),
        float(row.get("swa_overlap_gate_delta_abs") or 0.0),
        float(row.get("ttt_positive_role_mass") or 0.0),
        float(row.get("ttt_negative_role_mass") or 0.0),
        float(row.get("ttt_state_mean_rel_diff") or 0.0),
        float(row.get("ttt_w0_mean_rel_diff") or 0.0),
    ]


def _run_name(prefix: str, parent: str, candidate: str, chunk: int, horizon: int) -> str:
    return f"{prefix}_{parent}_{candidate}_chunk{chunk}_h{horizon}_globalgate_H9parent_SWKS3"


def _parse_csv_ints(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def _parse_csv_strs(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _write_heatmap(path: Path, row_labels: Sequence[str], col_labels: Sequence[str], values: Sequence[Sequence[float]], title: str) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    fig_w = max(6.0, 0.5 * len(col_labels) + 2.0)
    fig_h = max(4.0, 0.35 * len(row_labels) + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-root", required=True, type=Path)
    parser.add_argument("--run-prefix", default="V37_TRACK0_SMOKE_R1")
    parser.add_argument("--parents", default="H9,C9")
    parser.add_argument("--chunks", default="6,10,16")
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    parents = _parse_csv_strs(args.parents)
    chunks = _parse_csv_ints(args.chunks)
    candidates = _parse_csv_strs(args.candidates)
    out_dir = args.out_dir
    rollout_root = args.rollout_root

    atlas_rows: List[Dict[str, Any]] = []
    missing_rows: List[Dict[str, Any]] = []
    per_path_rows: List[Dict[str, Any]] = []
    keep_rows: List[Dict[str, Any]] = []
    empty_rows: List[Dict[str, Any]] = []
    protected_rows: List[Dict[str, Any]] = []

    for parent in parents:
        for chunk in chunks:
            for candidate in candidates:
                run_name = _run_name(args.run_prefix, parent, candidate, chunk, args.horizon)
                run_dir = rollout_root / run_name
                if not _status_done(run_dir, run_name):
                    missing_rows.append({
                        "parent": parent,
                        "chunk": chunk,
                        "candidate": candidate,
                        "run_name": run_name,
                        "run_dir": str(run_dir),
                        "reason": "missing_or_not_done",
                    })
                    continue

                context_rows = _read_jsonl(run_dir / "context_skip_summary.jsonl")
                role_rows = _read_jsonl(run_dir / "semantic_role_summary.jsonl")
                hook_rows = _read_jsonl(run_dir / "hook_effect_summary.jsonl")
                hmc_rows = _read_jsonl(run_dir / "hmc_state_hash.jsonl")
                sem_rows = _read_jsonl(run_dir / "semantic_group_summary.jsonl")
                latest_hmc = _latest_hmc(hmc_rows)
                latest_role = role_rows[-1] if role_rows else {}
                latest_sem = sem_rows[-1] if sem_rows else {}

                role_by_path = {path: _sum_role_counts(role_rows, ROLE_FIELDS[path]) for path in PATHS}
                path_metrics: Dict[str, Dict[str, Any]] = {}
                for path in PATHS:
                    ctx = _last_by_path(context_rows, path)
                    role_counts = role_by_path[path]
                    negative_mass = _role_frac(role_counts, "3")
                    positive_mass = _role_frac(role_counts, "1")
                    no_write_mass = _role_frac(role_counts, "4")
                    neutral_mass = _role_frac(role_counts, "2")
                    attention_available = bool(ctx.get("attention_mass_available", False))
                    removed_before = ctx.get("mean_attention_mass_removed_before")
                    influence = float(removed_before) if removed_before is not None else float("nan")
                    if path == "swa_read" and not math.isfinite(influence):
                        swa_gate = _max(_hook_values(hook_rows, "swa_read", "mean_swa_overlap_source_gate_delta"))
                        swa_score = _max(_hook_values(hook_rows, "swa_read", "mean_swa_overlap_source_score"))
                        vals = [v for v in [swa_gate, swa_score] if math.isfinite(v)]
                        influence = max(vals) if vals else float("nan")
                    if path == "ttt_apply" and not math.isfinite(influence):
                        influence = float(latest_hmc.get("memory_ttt_w0_mean_rel_diff") or 0.0)
                    path_metrics[path] = {
                        "path": path,
                        "role_counts": role_counts,
                        "positive_mass": positive_mass,
                        "neutral_mass": neutral_mass,
                        "negative_mass": negative_mass,
                        "no_write_mass": no_write_mass,
                        "keep_ratio": ctx.get("mean_context_source_keep_ratio"),
                        "skip_tokens": ctx.get("max_context_source_skip_tokens"),
                        "empty_source_events": ctx.get("num_context_empty_source_events"),
                        "attention_mass_available": attention_available,
                        "attention_mass_removed_before": removed_before,
                        "attention_mass_removed_after": ctx.get("mean_attention_mass_removed_after"),
                        "attention_mass_retained_before": ctx.get("mean_attention_mass_retained_before"),
                        "attention_mass_retained_after": ctx.get("mean_attention_mass_retained_after"),
                        "influence_mass": influence,
                        "explainability_status": (
                            "sampled-qk-softmax-mass" if attention_available
                            else "proxy_or_explainability_missing"
                        ),
                    }
                    per_path_rows.append({
                        "parent": parent,
                        "chunk": chunk,
                        "horizon": args.horizon,
                        "candidate": candidate,
                        "path": path,
                        "run_name": run_name,
                        "keep_ratio": ctx.get("mean_context_source_keep_ratio"),
                        "skip_tokens": ctx.get("max_context_source_skip_tokens"),
                        "empty_source_events": ctx.get("num_context_empty_source_events"),
                        "attention_mass_available": attention_available,
                        "attention_mass_removed_before": removed_before,
                        "attention_mass_removed_after": ctx.get("mean_attention_mass_removed_after"),
                        "attention_mass_retained_before": ctx.get("mean_attention_mass_retained_before"),
                        "attention_mass_retained_after": ctx.get("mean_attention_mass_retained_after"),
                        "positive_role_mass": positive_mass,
                        "neutral_role_mass": neutral_mass,
                        "negative_role_mass": negative_mass,
                        "no_write_role_mass": no_write_mass,
                        "influence_mass": influence,
                        "role_distribution": json.dumps(_role_dist(role_counts), sort_keys=True),
                        "explainability_status": path_metrics[path]["explainability_status"],
                    })

                frame_keep = path_metrics["frame_attention"].get("keep_ratio")
                global_keep = path_metrics["chunk_attention"].get("keep_ratio")
                frame_skip_ratio = 1.0 - float(frame_keep) if frame_keep is not None else 0.0
                global_skip_ratio = 1.0 - float(global_keep) if global_keep is not None else 0.0
                frame_empty = int(path_metrics["frame_attention"].get("empty_source_events") or 0)
                global_empty = int(path_metrics["chunk_attention"].get("empty_source_events") or 0)
                frame_skip_tokens = float(path_metrics["frame_attention"].get("skip_tokens") or 0.0)
                global_skip_tokens = float(path_metrics["chunk_attention"].get("skip_tokens") or 0.0)
                swa_source_gate_count = _max(_hook_values(hook_rows, "swa_read", "num_source_gate_applied"))
                swa_gate_delta = _max([abs(v) for v in _hook_values(hook_rows, "swa_read", "mean_swa_gate")])
                swa_overlap_delta = _max([abs(v) for v in _hook_values(hook_rows, "swa_read", "mean_swa_overlap_source_gate_delta")])
                ttt_mean_rel = float(latest_hmc.get("memory_ttt_mean_rel_diff") or 0.0)
                ttt_w0_mean_rel = float(latest_hmc.get("memory_ttt_w0_mean_rel_diff") or 0.0)
                protected = latest_hmc.get("prior_protected_token_count")
                if protected is None:
                    protected = latest_hmc.get("prior_protect_patch_count")
                atlas_rows.append({
                    "parent": parent,
                    "chunk": chunk,
                    "horizon": args.horizon,
                    "candidate": candidate,
                    "run_name": run_name,
                    "run_dir": str(run_dir),
                    "semantic_role_policy": latest_role.get("semantic_role_policy"),
                    "semantic_memory_paths": latest_role.get("semantic_memory_paths"),
                    "fine_label_available": latest_sem.get("fine_label_available"),
                    "fine_label_name_counts": json.dumps(latest_sem.get("fine_label_name_counts", {}), sort_keys=True),
                    "group_counts": json.dumps(latest_sem.get("group_counts", {}), sort_keys=True),
                    "token_count": latest_role.get("token_count"),
                    "protected_token_count": protected,
                    "frame_skip_tokens": frame_skip_tokens,
                    "global_skip_tokens": global_skip_tokens,
                    "frame_keep_ratio": frame_keep,
                    "global_keep_ratio": global_keep,
                    "frame_skip_ratio": frame_skip_ratio,
                    "global_skip_ratio": global_skip_ratio,
                    "context_empty_source_events": frame_empty + global_empty,
                    "frame_influence_mass": path_metrics["frame_attention"]["influence_mass"],
                    "global_influence_mass": path_metrics["chunk_attention"]["influence_mass"],
                    "frame_attention_mass_available": path_metrics["frame_attention"]["attention_mass_available"],
                    "global_attention_mass_available": path_metrics["chunk_attention"]["attention_mass_available"],
                    "swa_source_gate_applied": swa_source_gate_count if math.isfinite(swa_source_gate_count) else 0.0,
                    "swa_source_gate_applied_ratio_proxy": 1.0 if math.isfinite(swa_source_gate_count) and swa_source_gate_count > 0 else 0.0,
                    "swa_gate_abs": swa_gate_delta if math.isfinite(swa_gate_delta) else 0.0,
                    "swa_overlap_gate_delta_abs": swa_overlap_delta if math.isfinite(swa_overlap_delta) else 0.0,
                    "swa_influence_mass": path_metrics["swa_read"]["influence_mass"],
                    "ttt_positive_role_mass": path_metrics["ttt_apply"]["positive_mass"],
                    "ttt_negative_role_mass": path_metrics["ttt_apply"]["negative_mass"],
                    "ttt_no_long_write_mass": path_metrics["ttt_apply"]["no_write_mass"],
                    "ttt_state_mean_rel_diff": ttt_mean_rel,
                    "ttt_w0_mean_rel_diff": ttt_w0_mean_rel,
                    "ttt_influence_proxy": path_metrics["ttt_apply"]["influence_mass"],
                    "influence_max": _max([
                        _finite_float(path_metrics["frame_attention"]["influence_mass"]),
                        _finite_float(path_metrics["chunk_attention"]["influence_mass"]),
                        _finite_float(path_metrics["swa_read"]["influence_mass"]),
                        _finite_float(path_metrics["ttt_apply"]["influence_mass"]),
                    ]),
                    "explainability_missing_fields": ",".join(
                        path for path in PATHS
                        if path_metrics[path]["explainability_status"] == "proxy_or_explainability_missing"
                    ),
                })
                keep_rows.append({
                    "parent": parent,
                    "chunk": chunk,
                    "candidate": candidate,
                    "frame_keep_ratio": frame_keep,
                    "global_keep_ratio": global_keep,
                    "frame_skip_ratio": frame_skip_ratio,
                    "global_skip_ratio": global_skip_ratio,
                })
                empty_rows.append({
                    "parent": parent,
                    "chunk": chunk,
                    "candidate": candidate,
                    "context_empty_source_events": frame_empty + global_empty,
                })
                protected_rows.append({
                    "parent": parent,
                    "chunk": chunk,
                    "candidate": candidate,
                    "protected_token_count": protected,
                })

    # Per-policy action vectors aggregated across chunks/parents.
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in atlas_rows:
        grouped.setdefault(str(row["candidate"]), []).append(row)
    action_vectors: Dict[str, List[float]] = {}
    action_summary_rows: List[Dict[str, Any]] = []
    for candidate, rows in grouped.items():
        agg = {
            "candidate": candidate,
            "rows": len(rows),
            "mean_frame_skip_ratio": _mean([_finite_float(r.get("frame_skip_ratio")) for r in rows]),
            "mean_global_skip_ratio": _mean([_finite_float(r.get("global_skip_ratio")) for r in rows]),
            "mean_frame_influence_mass": _mean([_finite_float(r.get("frame_influence_mass")) for r in rows]),
            "mean_global_influence_mass": _mean([_finite_float(r.get("global_influence_mass")) for r in rows]),
            "max_influence_mass": _max([_finite_float(r.get("influence_max")) for r in rows]),
            "mean_swa_gate_proxy": _mean([_finite_float(r.get("swa_source_gate_applied_ratio_proxy")) for r in rows]),
            "mean_ttt_positive_role_mass": _mean([_finite_float(r.get("ttt_positive_role_mass")) for r in rows]),
            "mean_ttt_negative_role_mass": _mean([_finite_float(r.get("ttt_negative_role_mass")) for r in rows]),
            "mean_ttt_state_mean_rel_diff": _mean([_finite_float(r.get("ttt_state_mean_rel_diff")) for r in rows]),
        }
        action_summary_rows.append(agg)
        action_vectors[candidate] = [
            float(agg["mean_frame_skip_ratio"] or 0.0),
            float(agg["mean_global_skip_ratio"] or 0.0),
            float(agg["mean_frame_influence_mass"] or 0.0),
            float(agg["mean_global_influence_mass"] or 0.0),
            float(agg["mean_swa_gate_proxy"] or 0.0),
            float(agg["mean_ttt_positive_role_mass"] or 0.0),
            float(agg["mean_ttt_negative_role_mass"] or 0.0),
            float(agg["mean_ttt_state_mean_rel_diff"] or 0.0),
        ]

    jaccard_rows: List[Dict[str, Any]] = []
    candidates_present = sorted(action_vectors)
    for left in candidates_present:
        for right in candidates_present:
            j = _weighted_jaccard(action_vectors[left], action_vectors[right])
            max_abs_diff = max(abs(a - b) for a, b in zip(action_vectors[left], action_vectors[right]))
            jaccard_rows.append({
                "left": left,
                "right": right,
                "weighted_jaccard": j,
                "max_abs_feature_diff": max_abs_diff,
                "distinguishable_by_v37_h0b": bool(j <= 0.85 or max_abs_diff >= 0.05),
            })

    # Minimal per-label/masklet action files are explicit about unavailable granularity.
    per_label_rows: List[Dict[str, Any]] = []
    for row in atlas_rows:
        per_label_rows.append({
            "parent": row["parent"],
            "chunk": row["chunk"],
            "candidate": row["candidate"],
            "fine_label_name_counts": row["fine_label_name_counts"],
            "action_mass_status": "per-label action tensor not landed; only aggregate role/source summaries available",
        })
    per_masklet_rows: List[Dict[str, Any]] = [{
        "status": "explainability_missing",
        "reason": "per-masklet action masks were not landed in rollout artifacts",
    }]

    attention_mass_rows = [
        r for r in atlas_rows
        if bool(r.get("frame_attention_mass_available")) or bool(r.get("global_attention_mass_available"))
    ]
    context_empty_total = sum(int(r.get("context_empty_source_events") or 0) for r in atlas_rows)
    source_effect_rows = [
        r for r in atlas_rows
        if float(r.get("frame_skip_tokens") or 0.0) > 0.0 or float(r.get("global_skip_tokens") or 0.0) > 0.0
    ]
    swa_effect_rows = [r for r in atlas_rows if float(r.get("swa_source_gate_applied") or 0.0) > 0.0]
    ttt_effect_rows = [r for r in atlas_rows if float(r.get("ttt_state_mean_rel_diff") or 0.0) > 0.0]
    non_base_pairs = [
        r for r in jaccard_rows
        if r["left"] != r["right"] and r["left"] != "V31_BASE_H9_REFERENCE" and r["right"] != "V31_BASE_H9_REFERENCE"
    ]
    distinguishable_pairs = [r for r in non_base_pairs if bool(r["distinguishable_by_v37_h0b"])]
    max_influence = _max([_finite_float(r.get("influence_max")) for r in atlas_rows])
    source_influence = _max([
        max(_finite_float(r.get("frame_influence_mass")), _finite_float(r.get("global_influence_mass")))
        for r in atlas_rows
        if _finite_float(r.get("frame_skip_tokens")) > 0.0 or _finite_float(r.get("global_skip_tokens")) > 0.0
    ])
    h0a = bool(source_effect_rows and context_empty_total == 0 and (swa_effect_rows or ttt_effect_rows))
    h0b = bool(distinguishable_pairs)
    h0c = bool(math.isfinite(source_influence) and source_influence >= 0.03 and math.isfinite(max_influence) and max_influence >= 0.05)
    summary = {
        "rows_expected": len(parents) * len(chunks) * len(candidates),
        "rows_done": len(atlas_rows),
        "missing_rows": len(missing_rows),
        "parents": parents,
        "chunks": chunks,
        "horizon": args.horizon,
        "candidates": candidates,
        "context_empty_source_events_total": context_empty_total,
        "source_effect_rows": len(source_effect_rows),
        "swa_effect_rows": len(swa_effect_rows),
        "ttt_effect_rows": len(ttt_effect_rows),
        "attention_mass_rows": len(attention_mass_rows),
        "max_influence_mass": max_influence,
        "max_skipped_source_influence_mass": source_influence,
        "h0a_hook_reachability_pass": h0a,
        "h0b_action_distinguishability_pass": h0b,
        "h0c_influence_nontriviality_pass": h0c,
        "track0_gate_pass": bool(len(missing_rows) == 0 and h0a and h0b and h0c),
        "boundary": "Track 0 is action/influence audit only; it is not ATE or deployable online evidence.",
        "missing_explainability_boundary": (
            "Fields not landed at per-label/per-masklet/map granularity are marked explainability_missing; "
            "numeric influence uses only landed source attention mass or explicit state-diff proxies."
        ),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "semantic_influence_atlas.csv", atlas_rows)
    _write_csv(out_dir / "semantic_path_action_influence.csv", per_path_rows)
    _write_csv(out_dir / "action_summary_by_candidate.csv", action_summary_rows)
    _write_csv(out_dir / "action_jaccard_matrix.csv", jaccard_rows)
    _write_csv(out_dir / "policy_jaccard_matrix.csv", jaccard_rows)
    _write_csv(out_dir / "action_keep_ratio_by_path.csv", keep_rows)
    _write_csv(out_dir / "context_empty_source_events.csv", empty_rows)
    _write_csv(out_dir / "protected_token_count.csv", protected_rows)
    _write_csv(out_dir / "per_label_action_mass.csv", per_label_rows)
    _write_csv(out_dir / "per_masklet_action_mass.csv", per_masklet_rows)
    _write_csv(out_dir / "missing_rows.csv", missing_rows)
    _write_json(out_dir / "phase0_action_influence_summary.json", summary)

    # Heatmaps.
    heatmap_rows = sorted(action_summary_rows, key=lambda r: str(r["candidate"]))
    heat_cols = [
        "mean_frame_skip_ratio",
        "mean_global_skip_ratio",
        "mean_frame_influence_mass",
        "mean_global_influence_mass",
        "mean_swa_gate_proxy",
        "mean_ttt_positive_role_mass",
        "mean_ttt_negative_role_mass",
        "mean_ttt_state_mean_rel_diff",
    ]
    heat_values = [[float(row.get(col) or 0.0) for col in heat_cols] for row in heatmap_rows]
    heat_ok = _write_heatmap(
        out_dir / "semantic_group_memory_path_heatmap.png",
        [str(row["candidate"]) for row in heatmap_rows],
        heat_cols,
        heat_values,
        "v37 Track0 candidate x memory-path action/influence",
    )
    matrix_values: List[List[float]] = []
    for left in candidates_present:
        matrix_values.append([
            next(float(r["weighted_jaccard"]) for r in jaccard_rows if r["left"] == left and r["right"] == right)
            for right in candidates_present
        ])
    matrix_ok = _write_heatmap(
        out_dir / "action_jaccard_heatmap.png",
        candidates_present,
        candidates_present,
        matrix_values,
        "v37 Track0 weighted action Jaccard",
    )
    summary["heatmaps_generated"] = bool(heat_ok and matrix_ok)
    summary["semantic_group_memory_path_heatmap"] = str(out_dir / "semantic_group_memory_path_heatmap.png") if heat_ok else "generation_failed"
    summary["action_jaccard_heatmap"] = str(out_dir / "action_jaccard_heatmap.png") if matrix_ok else "generation_failed"
    _write_json(out_dir / "phase0_action_influence_summary.json", summary)

    md = [
        "# v37 Track 0 Action / Influence Audit",
        "",
        "## Scope",
        "",
        f"- Parents: `{','.join(parents)}`",
        f"- Chunks: `{','.join(str(c) for c in chunks)}`",
        f"- Horizon: `{args.horizon}`",
        f"- Rows expected: `{summary['rows_expected']}`",
        f"- Rows done: `{summary['rows_done']}`",
        f"- Missing rows: `{summary['missing_rows']}`",
        "",
        "## Gates",
        "",
        f"- H0A hook reachability: `{summary['h0a_hook_reachability_pass']}`",
        f"- H0B action distinguishability: `{summary['h0b_action_distinguishability_pass']}`",
        f"- H0C influence nontriviality: `{summary['h0c_influence_nontriviality_pass']}`",
        f"- Track 0 gate pass: `{summary['track0_gate_pass']}`",
        "",
        "## Key Metrics",
        "",
        f"- Context empty source events total: `{summary['context_empty_source_events_total']}`",
        f"- Source-effect rows: `{summary['source_effect_rows']}`",
        f"- SWA-effect rows: `{summary['swa_effect_rows']}`",
        f"- TTT-effect rows: `{summary['ttt_effect_rows']}`",
        f"- Rows with sampled attention mass: `{summary['attention_mass_rows']}`",
        f"- Max influence mass: `{summary['max_influence_mass']}`",
        f"- Max skipped source influence mass: `{summary['max_skipped_source_influence_mass']}`",
        "",
        "## Boundary",
        "",
        summary["boundary"],
        "",
        summary["missing_explainability_boundary"],
        "",
        "## Outputs",
        "",
        "- `semantic_influence_atlas.csv`",
        "- `semantic_path_action_influence.csv`",
        "- `action_summary_by_candidate.csv`",
        "- `action_jaccard_matrix.csv`",
        "- `semantic_group_memory_path_heatmap.png`",
        "- `action_jaccard_heatmap.png`",
        "- `per_label_action_mass.csv`",
        "- `per_masklet_action_mass.csv`",
        "",
    ]
    if missing_rows:
        md.extend(["## Missing Rows", ""])
        for row in missing_rows[:20]:
            md.append(f"- `{row['run_name']}`: {row['reason']}")
        if len(missing_rows) > 20:
            md.append(f"- ... plus {len(missing_rows) - 20} more")
        md.append("")
    (out_dir / "phase0_action_influence_report.md").write_text("\n".join(md), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary["track0_gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
