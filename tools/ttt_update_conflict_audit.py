#!/usr/bin/env python3
"""Summarize TTT update-conflict self-cue diagnostics from prior_debug.jsonl."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Tuple


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    return float(mean(vals)) if vals else 0.0


def _load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _find_prior_debug(run_dir: Path) -> Path:
    direct = run_dir / "prior_debug.jsonl"
    if direct.exists():
        return direct
    matches = sorted(run_dir.glob("**/prior_debug.jsonl"))
    if matches:
        return matches[0]
    log = run_dir / "01.log"
    if log.exists():
        return log
    matches = sorted(run_dir.glob("**/01.log"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No prior_debug.jsonl or 01.log found under {run_dir}")


def _run_id_from_path(path: Path) -> str:
    if path.name == "prior_debug.jsonl":
        return path.parent.name
    if path.name == "01.log":
        return path.parent.name
    return path.stem


def _iter_log_layer_records(path: Path, *, focus_chunk: int | None) -> Iterable[Dict[str, Any]]:
    chunk_idx = -1
    start_frame = -1
    end_frame = -1
    header_re = re.compile(r"# V2 Chunk\s+(\d+)/\d+\s+frames\s+\[(\d+),\s*(\d+)\)")
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = header_re.search(line)
            if match:
                chunk_idx = int(match.group(1))
                start_frame = int(match.group(2))
                end_frame = int(match.group(3))
                continue
            if "debug                       :" not in line:
                continue
            if focus_chunk is not None and chunk_idx != int(focus_chunk):
                continue
            payload_text = line.split("debug                       :", 1)[1].strip()
            if not payload_text.startswith("{"):
                continue
            try:
                payload = ast.literal_eval(payload_text)
            except (SyntaxError, ValueError):
                continue
            if not isinstance(payload, dict):
                continue
            for key, value in payload.items():
                if not key.startswith("layer_") or not isinstance(value, dict):
                    continue
                rec = dict(value)
                rec.update({
                    "run_id": _run_id_from_path(path),
                    "chunk_idx": chunk_idx,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "layer_idx": int(key.split("_", 1)[1]),
                })
                yield rec


def _collect(
    source_paths: List[Path],
    *,
    focus_chunk: int | None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    layer_rows: List[Dict[str, Any]] = []
    head_rows: List[Dict[str, Any]] = []
    for source_path in source_paths:
        run_id = _run_id_from_path(source_path)
        records = (
            _iter_log_layer_records(source_path, focus_chunk=focus_chunk)
            if source_path.name.endswith(".log")
            else _load_jsonl(source_path)
        )
        for rec in records:
            if "ttt_update_conflict_energy_mean" not in rec:
                continue
            chunk_idx = _as_int(rec.get("chunk_idx"), -1)
            if focus_chunk is not None and chunk_idx != int(focus_chunk):
                continue
            layer_idx = _as_int(rec.get("layer_idx"), -1)
            layer_rows.append({
                "run_id": rec.get("run_id") or run_id,
                "chunk_idx": chunk_idx,
                "start_frame": _as_int(rec.get("start_frame"), -1),
                "end_frame": _as_int(rec.get("end_frame"), -1),
                "layer_idx": layer_idx,
                "risk_source": rec.get("ttt_gradient_reversal_risk_source"),
                "conflict_mode": rec.get("ttt_update_conflict_mode"),
                "risk_mean": _as_float(rec.get("ttt_update_conflict_risk_mean")),
                "risk_p90": _as_float(rec.get("ttt_update_conflict_risk_p90")),
                "energy_mean": _as_float(rec.get("ttt_update_conflict_energy_mean")),
                "energy_p90": _as_float(rec.get("ttt_update_conflict_energy_p90")),
                "cos_mean": _as_float(rec.get("ttt_update_conflict_cos_mean")),
                "cos_p10": _as_float(rec.get("ttt_update_conflict_cos_p10")),
                "negative_cos_mass": _as_float(rec.get("ttt_update_conflict_negative_cos_mass")),
                "tri_pos_mass": _as_float(rec.get("ttt_tri_replay_pos_mass")),
                "tri_neu_mass": _as_float(rec.get("ttt_tri_replay_neu_mass")),
                "tri_neg_mass": _as_float(rec.get("ttt_tri_replay_neg_mass")),
                "tri_gamma_w0": _as_float(rec.get("ttt_tri_replay_w0_gamma")),
            })

            risk_heads = rec.get("ttt_update_conflict_risk_head_mean") or []
            energy_heads = rec.get("ttt_update_conflict_energy_head_mean") or []
            cos_heads = rec.get("ttt_update_conflict_cos_head_mean") or []
            n_heads = max(len(risk_heads), len(energy_heads), len(cos_heads))
            for head_idx in range(n_heads):
                head_rows.append({
                    "run_id": rec.get("run_id") or run_id,
                    "chunk_idx": chunk_idx,
                    "layer_idx": layer_idx,
                    "head_idx": head_idx,
                    "risk_mean": _as_float(risk_heads[head_idx] if head_idx < len(risk_heads) else None),
                    "energy_mean": _as_float(energy_heads[head_idx] if head_idx < len(energy_heads) else None),
                    "cos_mean": _as_float(cos_heads[head_idx] if head_idx < len(cos_heads) else None),
                })
    return layer_rows, head_rows


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _summarize_layer(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["run_id"]), int(row["layer_idx"]))].append(row)
    out: List[Dict[str, Any]] = []
    for (run_id, layer_idx), vals in sorted(grouped.items()):
        out.append({
            "run_id": run_id,
            "layer_idx": layer_idx,
            "risk_mean": _mean(v["risk_mean"] for v in vals),
            "risk_p90": _mean(v["risk_p90"] for v in vals),
            "energy_mean": _mean(v["energy_mean"] for v in vals),
            "energy_p90": _mean(v["energy_p90"] for v in vals),
            "cos_mean": _mean(v["cos_mean"] for v in vals),
            "negative_cos_mass": _mean(v["negative_cos_mass"] for v in vals),
            "tri_neg_mass": _mean(v["tri_neg_mass"] for v in vals),
            "records": len(vals),
        })
    return out


def _summarize_head(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, int, int], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["run_id"]), int(row["layer_idx"]), int(row["head_idx"]))].append(row)
    out: List[Dict[str, Any]] = []
    for (run_id, layer_idx, head_idx), vals in sorted(grouped.items()):
        out.append({
            "run_id": run_id,
            "layer_idx": layer_idx,
            "head_idx": head_idx,
            "risk_mean": _mean(v["risk_mean"] for v in vals),
            "energy_mean": _mean(v["energy_mean"] for v in vals),
            "cos_mean": _mean(v["cos_mean"] for v in vals),
            "records": len(vals),
        })
    return out


def _write_summary(path: Path, layer_summary: List[Dict[str, Any]], head_summary: List[Dict[str, Any]]) -> None:
    lines: List[str] = ["# TTT Update-Conflict Cue Audit", ""]
    runs = sorted({str(row["run_id"]) for row in layer_summary})
    lines.append(f"- Runs: {', '.join(runs) if runs else 'none'}")
    lines.append(f"- Layer records: {len(layer_summary)}")
    lines.append(f"- Head records: {len(head_summary)}")
    lines.append("")
    lines.append("## Top Layers By Risk")
    lines.append("")
    lines.append("| Run | Layer | Risk mean | Energy mean | Cos mean | Negative cos mass |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in sorted(layer_summary, key=lambda x: x["risk_mean"], reverse=True)[:20]:
        lines.append(
            f"| `{row['run_id']}` | {row['layer_idx']} | {row['risk_mean']:.6f} | "
            f"{row['energy_mean']:.6f} | {row['cos_mean']:.6f} | {row['negative_cos_mass']:.6f} |"
        )
    lines.append("")
    lines.append("## Top Heads By Risk")
    lines.append("")
    lines.append("| Run | Layer | Head | Risk mean | Energy mean | Cos mean |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in sorted(head_summary, key=lambda x: x["risk_mean"], reverse=True)[:30]:
        lines.append(
            f"| `{row['run_id']}` | {row['layer_idx']} | {row['head_idx']} | "
            f"{row['risk_mean']:.6f} | {row['energy_mean']:.6f} | {row['cos_mean']:.6f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", action="append", default=[], help="Run directory containing prior_debug.jsonl.")
    parser.add_argument("--prior_debug", action="append", default=[], help="Direct prior_debug.jsonl path.")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--focus_chunk", type=int, default=5, help="Chunk to summarize. Use -1 for all chunks.")
    args = parser.parse_args()

    prior_paths = [Path(p) for p in args.prior_debug]
    prior_paths.extend(_find_prior_debug(Path(p)) for p in args.run_dir)
    if not prior_paths:
        raise SystemExit("Provide at least one --run_dir or --prior_debug")

    focus_chunk = None if int(args.focus_chunk) < 0 else int(args.focus_chunk)
    layer_rows, head_rows = _collect(prior_paths, focus_chunk=focus_chunk)
    layer_summary = _summarize_layer(layer_rows)
    head_summary = _summarize_head(head_rows)

    out_dir = Path(args.out_dir)
    _write_csv(out_dir / "update_conflict_layers_raw.csv", layer_rows)
    _write_csv(out_dir / "update_conflict_heads_raw.csv", head_rows)
    _write_csv(out_dir / "update_conflict_layers_summary.csv", layer_summary)
    _write_csv(out_dir / "update_conflict_heads_summary.csv", head_summary)
    _write_summary(out_dir / "update_conflict_summary.md", layer_summary, head_summary)


if __name__ == "__main__":
    main()
