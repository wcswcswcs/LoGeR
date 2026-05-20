#!/usr/bin/env python3
"""Select automatic tri-replay body/exit windows from TTT self-cue traces."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Sequence, Tuple


MANUAL_BODY = {5, 6, 7, 8, 9}
MANUAL_EXIT = {10, 11, 12}


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: Any, default: int = -1) -> int:
    try:
        if value in (None, ""):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _robust_z(values: Dict[int, float]) -> Dict[int, float]:
    vals = [v for v in values.values() if math.isfinite(v)]
    if not vals:
        return {k: 0.0 for k in values}
    med = median(vals)
    mad = median([abs(v - med) for v in vals]) or 1e-9
    scale = 1.4826 * mad
    return {k: (v - med) / scale for k, v in values.items()}


def _phase_residual(values: Dict[int, float], reset_every: int) -> Dict[int, float]:
    if reset_every <= 0:
        return dict(values)
    by_phase: Dict[int, List[float]] = defaultdict(list)
    for ch, value in values.items():
        by_phase[ch % reset_every].append(value)
    med_by_phase = {phase: median(vals) for phase, vals in by_phase.items()}
    return {ch: value - med_by_phase.get(ch % reset_every, 0.0) for ch, value in values.items()}


def _moving_average(values: Dict[int, float], radius: int) -> Dict[int, float]:
    if radius <= 0:
        return dict(values)
    out: Dict[int, float] = {}
    for ch in sorted(values):
        vals = [values[k] for k in range(ch - radius, ch + radius + 1) if k in values]
        out[ch] = sum(vals) / max(len(vals), 1)
    return out


def _best_contiguous_window(chunks: Sequence[int], score: Dict[int, float], length: int) -> List[int]:
    allowed = set(int(ch) for ch in chunks)
    best: List[int] = []
    best_score: float | None = None
    for start in sorted(allowed):
        window = list(range(start, start + length))
        if not all(ch in allowed for ch in window):
            continue
        value = sum(score.get(ch, -999.0) for ch in window)
        if best_score is None or value > best_score:
            best_score = value
            best = window
    return best


def _score_f1(selected: Sequence[int], target: set[int]) -> Tuple[float, float, float]:
    sel = set(int(x) for x in selected)
    inter = len(sel & target)
    prec = inter / max(len(sel), 1)
    rec = inter / max(len(target), 1)
    f1 = 2.0 * prec * rec / max(prec + rec, 1e-9)
    return prec, rec, f1


def _contiguous_tail(body: Sequence[int], count: int = 3, *, max_chunk: int | None = None) -> List[int]:
    if not body:
        return []
    start = max(body) + 1
    tail = list(range(start, start + count))
    if max_chunk is not None:
        tail = [ch for ch in tail if ch <= max_chunk]
    return tail


def _chunk_gamma_map(body: Sequence[int], exit_chunks: Sequence[int], body_gamma: float, exit_gamma: float) -> str:
    parts: List[str] = []
    for ch in sorted(set(int(x) for x in body)):
        parts.append(f"{ch}:{body_gamma:.4f}".rstrip("0").rstrip("."))
    for ch in sorted(set(int(x) for x in exit_chunks) - set(int(x) for x in body)):
        if exit_gamma > 0:
            parts.append(f"{ch}:{exit_gamma:.4f}".rstrip("0").rstrip("."))
    return ",".join(parts)


def _chunk_params(chunks: Sequence[int], pos: float, neg: float, neu: float) -> str:
    return ",".join(f"{int(ch)}:{pos:.2f}/{neg:.2f}/{neu:.2f}" for ch in sorted(set(chunks)))


def _fmt_chunks(chunks: Sequence[int]) -> str:
    return ",".join(str(int(ch)) for ch in sorted(set(chunks)))


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers_csv", required=True)
    parser.add_argument("--heads_csv", default="")
    parser.add_argument("--hmc_jsonl", default="")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--body_gamma", type=float, default=0.005)
    parser.add_argument("--exit_gamma", type=float, default=0.003)
    parser.add_argument("--pos_frac", type=float, default=0.35)
    parser.add_argument("--neg_frac", type=float, default=0.12)
    parser.add_argument("--neutral_lambda", type=float, default=0.85)
    parser.add_argument("--reset_every", type=int, default=5)
    parser.add_argument("--local_start", type=int, default=4)
    parser.add_argument("--local_end", type=int, default=13)
    parser.add_argument("--smooth_radius", type=int, default=1)
    args = parser.parse_args()

    layer_rows = _load_csv(Path(args.layers_csv))
    chunks = sorted({ _int(r.get("chunk_idx")) for r in layer_rows if _int(r.get("chunk_idx")) >= 0 })
    layer_risk: Dict[int, Dict[int, float]] = defaultdict(dict)
    layer_energy: Dict[int, Dict[int, float]] = defaultdict(dict)
    for row in layer_rows:
        ch = _int(row.get("chunk_idx"))
        ly = _int(row.get("layer_idx"))
        if ch < 0 or ly < 0:
            continue
        layer_risk[ch][ly] = _float(row.get("risk_mean"))
        layer_energy[ch][ly] = _float(row.get("energy_mean"))

    hmc_w0: Dict[int, float] = {}
    dg_mass: Dict[int, float] = {}
    starts: Dict[int, int] = {}
    ends: Dict[int, int] = {}
    if args.hmc_jsonl:
        for row in _load_jsonl(Path(args.hmc_jsonl)):
            ch = _int(row.get("chunk_idx"))
            if ch < 0:
                continue
            hmc_w0[ch] = _float(row.get("memory_ttt_w0_mean_rel_diff"))
            dg_mass[ch] = _float(row.get("prior_dynamic_mass_D_gt_050"))
            starts[ch] = _int(row.get("start_frame"))
            ends[ch] = _int(row.get("end_frame"))

    head12_0: Dict[int, float] = {}
    if args.heads_csv:
        for row in _load_csv(Path(args.heads_csv)):
            ch = _int(row.get("chunk_idx"))
            if _int(row.get("layer_idx")) == 12 and _int(row.get("head_idx")) == 0:
                head12_0[ch] = _float(row.get("risk_mean"))

    mean_risk = {ch: sum(layer_risk[ch].values()) / max(len(layer_risk[ch]), 1) for ch in chunks}
    peak_risk = {ch: max(layer_risk[ch].values()) if layer_risk[ch] else 0.0 for ch in chunks}
    layer12 = {ch: layer_risk[ch].get(12, 0.0) for ch in chunks}
    layer5 = {ch: layer_risk[ch].get(5, 0.0) for ch in chunks}
    layer9 = {ch: layer_risk[ch].get(9, 0.0) for ch in chunks}
    energy12 = {ch: layer_energy[ch].get(12, 0.0) for ch in chunks}
    z_mean = _robust_z(mean_risk)
    z_peak = _robust_z(peak_risk)
    z_l12 = _robust_z(layer12)
    z_l5 = _robust_z(layer5)
    z_l9 = _robust_z(layer9)
    z_e12 = _robust_z(energy12)
    z_w0 = _robust_z({ch: hmc_w0.get(ch, 0.0) for ch in chunks})
    z_dg = _robust_z({ch: dg_mass.get(ch, 0.0) for ch in chunks})
    z_h120 = _robust_z({ch: head12_0.get(ch, 0.0) for ch in chunks})

    body_score: Dict[int, float] = {}
    exit_score: Dict[int, float] = {}
    for ch in chunks:
        body_score[ch] = (
            1.0 * z_mean[ch]
            + 1.0 * z_peak[ch]
            + 1.0 * z_l12[ch]
            + 0.6 * z_l5[ch]
            + 0.4 * z_l9[ch]
            + 0.4 * z_e12[ch]
            + 0.4 * z_w0[ch]
            + 0.3 * z_dg[ch]
        )
        exit_score[ch] = 0.7 * z_mean[ch] + 0.7 * z_l12[ch] + 0.4 * z_w0[ch]

    z_body_phase = _robust_z(_phase_residual(body_score, args.reset_every))
    z_exit_phase = _robust_z(_phase_residual(exit_score, args.reset_every))
    body_v2_raw: Dict[int, float] = {}
    exit_v2_raw: Dict[int, float] = {}
    for ch in chunks:
        body_v2_raw[ch] = (
            1.10 * z_body_phase[ch]
            + 0.60 * z_l12[ch]
            + 0.35 * z_h120[ch]
            + 0.25 * z_w0[ch]
            + 0.20 * z_dg[ch]
        )
        exit_v2_raw[ch] = (
            0.90 * z_exit_phase[ch]
            + 0.45 * z_peak[ch]
            + 0.35 * z_w0[ch]
            + 0.20 * z_dg[ch]
        )
    body_v2_score = _moving_average(body_v2_raw, args.smooth_radius)
    exit_v2_score = _moving_average(exit_v2_raw, args.smooth_radius)

    top_body = [ch for ch, _ in sorted(body_score.items(), key=lambda kv: kv[1], reverse=True)[: args.top_k]]
    threshold_body = [ch for ch in chunks if body_score[ch] > 0.5]
    if not threshold_body:
        threshold_body = top_body
    head_body = [ch for ch, _ in sorted(z_h120.items(), key=lambda kv: kv[1], reverse=True)[: args.top_k]]
    manual_body = sorted(MANUAL_BODY)
    manual_exit = sorted(MANUAL_EXIT)

    max_chunk = max(chunks) if chunks else None
    auto_exit_after_top = _contiguous_tail(top_body, 3, max_chunk=max_chunk)
    auto_exit_after_threshold = _contiguous_tail(threshold_body, 3, max_chunk=max_chunk)
    auto_exit_after_head = _contiguous_tail(head_body, 3, max_chunk=max_chunk)
    exit_candidates = [ch for ch in chunks if ch > max(manual_body)]
    auto_exit_for_manual = [ch for ch, _ in sorted(((ch, exit_score[ch]) for ch in exit_candidates), key=lambda kv: kv[1], reverse=True)[:3]]
    local_chunks = [ch for ch in chunks if args.local_start <= ch <= args.local_end]
    contig_body = _best_contiguous_window(local_chunks, body_v2_score, 5)
    contig_exit_candidates = [ch for ch in local_chunks if not contig_body or ch > max(contig_body)]
    contig_exit = _best_contiguous_window(contig_exit_candidates, exit_v2_score, 3)
    tail_exit = _contiguous_tail(contig_body, 3, max_chunk=max_chunk)
    tail_exit = [ch for ch in tail_exit if ch in set(local_chunks)]
    local_top_body = [
        ch for ch, _ in sorted(
            ((ch, body_v2_score[ch]) for ch in local_chunks),
            key=lambda kv: kv[1],
            reverse=True,
        )[: args.top_k]
    ]
    local_top_exit = [
        ch for ch, _ in sorted(
            ((ch, exit_v2_score[ch]) for ch in local_chunks if ch not in set(local_top_body)),
            key=lambda kv: kv[1],
            reverse=True,
        )[:3]
    ]

    strategies = [
        ("AUTO_WIN_01", "top5_R_body_plus_tail3", top_body, auto_exit_after_top),
        ("AUTO_WIN_02", "threshold_R_body_hysteresis_tail", threshold_body, auto_exit_after_threshold),
        ("AUTO_WIN_03", "top5_layer12_head0_plus_tail3", head_body, auto_exit_after_head),
        ("AUTO_WIN_04", "top5_R_body_no_exit", top_body, []),
        ("AUTO_WIN_05", "manual_body_auto_exit", manual_body, auto_exit_for_manual),
        ("AUTO_WIN_06", "auto_body_manual_exit", top_body, manual_exit),
        ("AUTO2_WIN_01", "resetaware_contig_body_tail3", contig_body, tail_exit),
        ("AUTO2_WIN_02", "resetaware_contig_body_best_exit3", contig_body, contig_exit or tail_exit),
        ("AUTO2_WIN_03", "resetaware_local_top_body_tail3", local_top_body, _contiguous_tail(local_top_body, 3, max_chunk=max_chunk)),
        ("AUTO2_WIN_04", "resetaware_local_top_body_no_exit", local_top_body, []),
        ("AUTO2_WIN_05", "resetaware_contig_body_manualexit", contig_body, manual_exit),
        ("AUTO2_WIN_06", "manualbody_resetaware_best_exit", manual_body, contig_exit or local_top_exit),
    ]

    chunk_rows: List[Dict[str, Any]] = []
    for ch in chunks:
        chunk_rows.append({
            "chunk": ch,
            "start": starts.get(ch, ""),
            "end": ends.get(ch, ""),
            "R_body": body_score[ch],
            "R_exit": exit_score[ch],
            "R_body_phase": z_body_phase[ch],
            "R_exit_phase": z_exit_phase[ch],
            "R_body_v2": body_v2_score[ch],
            "R_exit_v2": exit_v2_score[ch],
            "mean_risk": mean_risk[ch],
            "peak_risk": peak_risk[ch],
            "layer12_risk": layer12[ch],
            "layer5_risk": layer5[ch],
            "layer9_risk": layer9[ch],
            "layer12_energy": energy12[ch],
            "head12_0_risk": head12_0.get(ch, 0.0),
            "w0_rel_diff": hmc_w0.get(ch, 0.0),
            "Dg_mass_gt_050": dg_mass.get(ch, 0.0),
            "manual_body": int(ch in MANUAL_BODY),
            "manual_exit": int(ch in MANUAL_EXIT),
        })

    strategy_rows: List[Dict[str, Any]] = []
    shell_lines: List[str] = []
    for name, desc, body, exit_chunks in strategies:
        active = sorted(set(body) | set(exit_chunks))
        bp, br, bf = _score_f1(body, MANUAL_BODY)
        ep, er, ef = _score_f1(exit_chunks, MANUAL_EXIT)
        gamma_map = _chunk_gamma_map(body, exit_chunks, args.body_gamma, args.exit_gamma)
        params = _chunk_params(active, args.pos_frac, args.neg_frac, args.neutral_lambda)
        strategy_rows.append({
            "strategy": name,
            "description": desc,
            "body_chunks": _fmt_chunks(body),
            "exit_chunks": _fmt_chunks(exit_chunks),
            "active_chunks": _fmt_chunks(active),
            "body_precision": bp,
            "body_recall": br,
            "body_f1": bf,
            "exit_precision": ep,
            "exit_recall": er,
            "exit_f1": ef,
            "chunk_gammas": gamma_map,
            "chunk_params": params,
        })
        shell_lines.append(f"# {name}: {desc}")
        shell_lines.append(f"TTT_WRITE_GRADIENT_REVERSAL_CHUNK_GAMMAS='{gamma_map}'")
        shell_lines.append(f"TTT_WRITE_TRI_REPLAY_CHUNK_PARAMS='{params}'")
        shell_lines.append("")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "auto_window_chunk_scores.csv", chunk_rows)
    _write_csv(out_dir / "auto_window_strategies.csv", strategy_rows)
    (out_dir / "auto_window_env_snippets.sh").write_text("\n".join(shell_lines), encoding="utf-8")

    summary = ["# TTT Auto Window Selector", ""]
    summary.append("| Strategy | Body | Exit | Body F1 | Exit F1 |")
    summary.append("|---|---|---|---:|---:|")
    for row in strategy_rows:
        summary.append(
            f"| `{row['strategy']}` | `{row['body_chunks']}` | `{row['exit_chunks']}` | "
            f"{float(row['body_f1']):.3f} | {float(row['exit_f1']):.3f} |"
        )
    summary.append("")
    summary.append("## Top R_body chunks")
    summary.append("")
    summary.append("| Chunk | R_body | R_exit | layer12 risk | head12/0 risk |")
    summary.append("|---:|---:|---:|---:|---:|")
    for row in sorted(chunk_rows, key=lambda r: float(r["R_body"]), reverse=True)[:12]:
        summary.append(
            f"| {row['chunk']} | {float(row['R_body']):.3f} | {float(row['R_exit']):.3f} | "
            f"{float(row['layer12_risk']):.6f} | {float(row['head12_0_risk']):.6f} |"
        )
    summary.append("")
    summary.append("## Top reset-aware local chunks")
    summary.append("")
    summary.append("| Chunk | R_body_v2 | R_exit_v2 | R_body_phase | R_exit_phase |")
    summary.append("|---:|---:|---:|---:|---:|")
    for row in sorted(chunk_rows, key=lambda r: float(r["R_body_v2"]), reverse=True)[:12]:
        summary.append(
            f"| {row['chunk']} | {float(row['R_body_v2']):.3f} | {float(row['R_exit_v2']):.3f} | "
            f"{float(row['R_body_phase']):.3f} | {float(row['R_exit_phase']):.3f} |"
        )
    (out_dir / "auto_window_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
