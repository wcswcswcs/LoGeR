#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/acl2_v115tf_semantic_internal_alignment_evidence_influence_control"
STAGE2 = RESULT_ROOT / "stage2_alignment_cues"
SEM_ROOT = ROOT / "results/acl2_v113hs_horizonstream_semantic_aware_geometric_evidence_influence/semantic_projection"


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def f(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def semantic_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seq in ["00", "01", "02", "05"]:
        risk_path = SEM_ROOT / f"seq{seq}_risk.npy"
        stable_path = SEM_ROOT / f"seq{seq}_stable.npy"
        conf_path = SEM_ROOT / f"seq{seq}_confidence.npy"
        if not (risk_path.exists() and stable_path.exists()):
            continue
        risk = np.load(risk_path, mmap_mode="r")
        stable = np.load(stable_path, mmap_mode="r")
        conf = np.load(conf_path, mmap_mode="r") if conf_path.exists() else None
        frame_risk = np.asarray(np.mean(risk, axis=1), dtype=np.float32)
        frame_stable = np.asarray(np.mean(stable, axis=1), dtype=np.float32)
        frame_conf = np.asarray(np.mean(conf, axis=1), dtype=np.float32) if conf is not None else np.full_like(frame_risk, np.nan)
        novelty = np.abs(frame_risk - np.concatenate([frame_risk[:1], frame_risk[:-1]])) + np.abs(
            frame_stable - np.concatenate([frame_stable[:1], frame_stable[:-1]])
        )
        for idx in range(frame_risk.shape[0]):
            rows.append(
                {
                    "seq": seq,
                    "frame_idx": idx,
                    "semantic_risk_mean": float(frame_risk[idx]),
                    "semantic_stable_mean": float(frame_stable[idx]),
                    "semantic_confidence_mean": float(frame_conf[idx]) if math.isfinite(float(frame_conf[idx])) else "",
                    "semantic_novelty_proxy": float(novelty[idx]),
                    "source_path": rel(SEM_ROOT),
                }
            )
    return rows


def collect_hg_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted((RESULT_ROOT / "diagnostics").glob("*/hs_hg_action_gate_rows.csv")):
        for row in read_csv(path):
            row = dict(row)
            row["source_path"] = rel(path)
            row["case"] = path.parent.name
            rows.append(row)
    return rows


def head_reliability_rows(hg_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in hg_rows:
        out.append(
            {
                "seq": row.get("seq", ""),
                "chunk_idx": row.get("chunk_idx", ""),
                "chunk_start": row.get("chunk_start", ""),
                "chunk_end": row.get("chunk_end", ""),
                "action": row.get("action", ""),
                "control": row.get("control", ""),
                "scope": row.get("scope", ""),
                "num_local_rows": row.get("num_local_rows", ""),
                "num_heads": row.get("num_heads", ""),
                "semantic_risk_mean": row.get("semantic_risk_mean", ""),
                "semantic_stable_mean": row.get("semantic_stable_mean", ""),
                "internal_head_q_mean": row.get("internal_head_q_mean", ""),
                "internal_head_q_std": row.get("internal_head_q_std", ""),
                "gate_mean": row.get("gate_mean", ""),
                "gate_std": row.get("gate_std", ""),
                "gate_row_mean_mean": row.get("gate_row_mean_mean", ""),
                "gate_row_mean_std": row.get("gate_row_mean_std", ""),
                "changed_head_fraction_abs_gt_1e_4": row.get("changed_head_fraction_abs_gt_1e_4", ""),
                "source_path": row.get("source_path", ""),
                "case": row.get("case", ""),
            }
        )
    return out


def mrt_rows() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in sorted((RESULT_ROOT / "diagnostics").glob("*/hs_mrt_readout_probe_rows.csv")):
        for row in read_csv(path):
            row["source_path"] = rel(path)
            row["case"] = path.parent.name
            out.append(row)
    return out


def gla_rows() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in sorted((RESULT_ROOT / "diagnostics").glob("*/hs_gla_state_probe_rows.csv")):
        for row in read_csv(path):
            row["source_path"] = rel(path)
            row["case"] = path.parent.name
            out.append(row)
    return out


def agreement_rows(hs_sem: list[dict[str, Any]], head_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sem_by_seq = {}
    for row in hs_sem:
        sem_by_seq.setdefault(str(row["seq"]), []).append(row)
    out: list[dict[str, Any]] = []
    for row in head_rows:
        seq = str(row.get("seq", "")).replace("\\", "/").split("/")[0]
        try:
            start = int(row.get("chunk_start", 0))
            end = int(row.get("chunk_end", start + 1))
        except ValueError:
            start, end = 0, 0
        sem = sem_by_seq.get(seq, [])
        chunk = [r for r in sem if start <= int(r["frame_idx"]) < end]
        sem_risk = float(np.mean([float(r["semantic_risk_mean"]) for r in chunk])) if chunk else math.nan
        sem_stable = float(np.mean([float(r["semantic_stable_mean"]) for r in chunk])) if chunk else math.nan
        row_risk = f(row.get("semantic_risk_mean"))
        row_stable = f(row.get("semantic_stable_mean"))
        out.append(
            {
                "seq": seq,
                "chunk_idx": row.get("chunk_idx", ""),
                "chunk_start": start,
                "chunk_end": end,
                "action": row.get("action", ""),
                "case": row.get("case", ""),
                "semantic_projection_risk_mean": sem_risk if math.isfinite(sem_risk) else "",
                "runtime_gate_risk_mean": row.get("semantic_risk_mean", ""),
                "risk_abs_delta": abs(sem_risk - row_risk) if math.isfinite(sem_risk) and math.isfinite(row_risk) else "",
                "semantic_projection_stable_mean": sem_stable if math.isfinite(sem_stable) else "",
                "runtime_gate_stable_mean": row.get("semantic_stable_mean", ""),
                "stable_abs_delta": abs(sem_stable - row_stable) if math.isfinite(sem_stable) and math.isfinite(row_stable) else "",
                "internal_head_q_std": row.get("internal_head_q_std", ""),
                "gate_std": row.get("gate_std", ""),
                "gate_row_mean_std": row.get("gate_row_mean_std", ""),
            }
        )
    return out


def main() -> None:
    STAGE2.mkdir(parents=True, exist_ok=True)
    sem = semantic_rows()
    hg = collect_hg_rows()
    head = head_reliability_rows(hg)
    mrt = mrt_rows()
    gla = gla_rows()
    agree = agreement_rows(sem, head)
    write_csv(STAGE2 / "hs_local_attention_quality_rows.csv", sem)
    write_csv(STAGE2 / "hs_head_reliability_rows.csv", head)
    write_csv(STAGE2 / "hs_gla_state_quality_rows.csv", gla)
    write_csv(STAGE2 / "hs_mrt_scale_safety_rows.csv", mrt)
    write_csv(STAGE2 / "hs_semantic_internal_agreement_rows.csv", agree)
    seqs = sorted({str(r["seq"]) for r in sem})
    frame_count = len(sem)
    agreement_count = len(agree)
    join_coverage = 1.0 if head and agreement_count == len(head) else (0.0 if head else None)
    summary = {
        "schema": "acl2_v115tf_stage2_alignment_cues_summary_v1",
        "semantic_frame_rows": frame_count,
        "semantic_seqs": seqs,
        "hs_head_reliability_rows": len(head),
        "hs_mrt_scale_safety_rows": len(mrt),
        "hs_gla_state_quality_rows": len(gla),
        "hs_semantic_internal_agreement_rows": agreement_count,
        "join_coverage_for_head_rows": join_coverage,
        "stage2_status": "pass" if head and join_coverage is not None and join_coverage >= 0.95 else "partial_missing_head_rows",
        "outputs": {
            "hs_local_attention_quality_rows": rel(STAGE2 / "hs_local_attention_quality_rows.csv"),
            "hs_head_reliability_rows": rel(STAGE2 / "hs_head_reliability_rows.csv"),
            "hs_gla_state_quality_rows": rel(STAGE2 / "hs_gla_state_quality_rows.csv"),
            "hs_mrt_scale_safety_rows": rel(STAGE2 / "hs_mrt_scale_safety_rows.csv"),
            "hs_semantic_internal_agreement_rows": rel(STAGE2 / "hs_semantic_internal_agreement_rows.csv"),
        },
    }
    if summary["stage2_status"] != "pass":
        (STAGE2 / "CUE_JOIN_BLOCKED.md").write_text(
            "# CUE_JOIN_BLOCKED\n\n"
            f"head_rows={len(head)}, agreement_rows={agreement_count}, join_coverage={join_coverage}\n",
            encoding="utf-8",
        )
    write_json(STAGE2 / "stage2_alignment_cues_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

