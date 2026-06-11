#!/usr/bin/env python3
"""Offline v29C action distinguishability gate from masklet-3D alignment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Set


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _f(row: Dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        text = str(row.get(key, "")).strip()
        return float(text) if text else default
    except Exception:
        return default


def _masklet_key(row: Dict[str, str]) -> str:
    return f"{row['chunk_idx']}:{row['masklet_id']}"


def _actions(rows: List[Dict[str, str]], policy: str) -> Dict[str, Set[str]]:
    out = {
        "source_keep": set(),
        "source_skip": set(),
        "swa_anchor_keep": set(),
        "ttt_positive": set(),
        "ttt_neutral": set(),
        "ttt_negative": set(),
        "ttt_no_write": set(),
    }
    for row in rows:
        key = _masklet_key(row)
        group = str(row.get("video_group_pred", "UNKNOWN"))
        support = int(float(row.get("projected_pixel_support_count") or 0))
        agreement = str(row.get("agreement_pred_vs_projected", "unknown")).lower()
        t_mask = _f(row, "t_mask", 0.0)
        q3d_known = str(row.get("q_3d", "")).strip() != ""

        sem_positive = group in {"GROUND_STRUCTURE", "STRUCTURE_ANCHOR"}
        sem_lowstuff = group in {"VEGETATION_STUFF", "SKY_STUFF"}
        trusted = t_mask >= 0.65
        supported_agree = q3d_known and agreement == "true"
        supported_disagree = q3d_known and agreement == "false"

        if policy == "SEM_ONLY":
            if sem_positive:
                out["source_keep"].add(key)
                out["swa_anchor_keep"].add(key)
                out["ttt_positive"].add(key)
            elif sem_lowstuff:
                out["source_skip"].add(key)
                out["ttt_neutral"].add(key)
            else:
                out["ttt_no_write"].add(key)
        elif policy == "QUALITY_AWARE_SEM":
            if sem_positive and trusted and not supported_disagree:
                out["source_keep"].add(key)
                out["swa_anchor_keep"].add(key)
                out["ttt_positive"].add(key)
            elif supported_disagree or (support >= 50 and not trusted):
                out["source_skip"].add(key)
                out["ttt_no_write"].add(key)
            elif sem_lowstuff and trusted and supported_agree:
                out["source_keep"].add(key)
                out["ttt_neutral"].add(key)
            else:
                out["source_skip"].add(key)
                out["ttt_neutral"].add(key)
        elif policy == "RISK_ONLY":
            if support >= 50 and t_mask >= 0.75:
                out["source_keep"].add(key)
                out["ttt_positive"].add(key)
            elif support >= 50 and t_mask < 0.55:
                out["source_skip"].add(key)
                out["ttt_no_write"].add(key)
            else:
                out["ttt_neutral"].add(key)
        elif policy == "QUALITY_AWARE_SEM_RISK":
            if sem_positive and trusted and supported_agree:
                out["source_keep"].add(key)
                out["swa_anchor_keep"].add(key)
                out["ttt_positive"].add(key)
            elif supported_disagree:
                out["source_skip"].add(key)
                out["ttt_negative"].add(key)
            elif trusted:
                out["source_keep"].add(key)
                out["ttt_neutral"].add(key)
            else:
                out["source_skip"].add(key)
                out["ttt_no_write"].add(key)
        else:
            raise ValueError(policy)
    return out


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def _ratio_diff(a: Set[str], b: Set[str], total: int) -> float:
    return abs(len(a) / max(1, total) - len(b) / max(1, total))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alignment-csv", default="results/kitti01_hmc_v2/acl2_v29c_semantickitti_download_projection_videomasklet/masklet_3d_alignment/masklet_alignment.csv")
    parser.add_argument("--results-root", default="results/kitti01_hmc_v2/acl2_v29c_semantickitti_download_projection_videomasklet")
    args = parser.parse_args()

    rows = _read_csv(Path(args.alignment_csv))
    policies = {
        name: _actions(rows, name)
        for name in ("SEM_ONLY", "QUALITY_AWARE_SEM", "RISK_ONLY", "QUALITY_AWARE_SEM_RISK")
    }
    pairs = [
        ("SEM_ONLY", "QUALITY_AWARE_SEM"),
        ("QUALITY_AWARE_SEM", "RISK_ONLY"),
        ("RISK_ONLY", "QUALITY_AWARE_SEM_RISK"),
    ]
    total = len(rows)
    out_rows: List[Dict[str, object]] = []
    pass_any = False
    for left, right in pairs:
        for action in sorted(policies[left]):
            jac = _jaccard(policies[left][action], policies[right][action])
            diff = _ratio_diff(policies[left][action], policies[right][action], total)
            row = {
                "left_policy": left,
                "right_policy": right,
                "action": action,
                "left_count": len(policies[left][action]),
                "right_count": len(policies[right][action]),
                "jaccard": jac,
                "ratio_difference": diff,
                "action_gate_pass": bool(jac <= 0.85 or diff >= 0.05),
            }
            pass_any = pass_any or bool(row["action_gate_pass"])
            out_rows.append(row)

    out_dir = Path(args.results_root) / "phase4_action_distinguishability"
    _write_csv(out_dir / "action_distinguishability.csv", out_rows)
    summary = {
        "phase": "v29c_phase4_action_distinguishability",
        "masklets": total,
        "action_distinguishability_gate_pass": pass_any,
        "gate_rule": "Jaccard <= 0.85 or ratio_difference >= 0.05 for at least one compared action",
        "note": "Offline masklet-level action preview; no trajectory rollout is claimed.",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "action_distinguishability_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0 if pass_any else 1


if __name__ == "__main__":
    raise SystemExit(main())
