#!/usr/bin/env python3
"""Phase-0 C9 locked repeat gate for ACL2 v43."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any, Dict

HISTORICAL_C9_ATE = 33.7629421029
EXPECTED = {
    "beta_frame": "4.75",
    "beta_swa": "4.75",
    "hybrid_memory_mode": "hybrid",
    "hmc_commit_mode": "probe_ttt_write",
    "hmc_write_score_source": "stage_d_x_dg_inv_sqrt",
    "read_cue_source": "acl2.gg.qq.low.g2_3.past_only.headmean.robustq",
    "read_beta_frame_chunks": "5:4.85,6:4.85,7:4.85,8:4.85,9:4.85,10:4.25,11:4.25,12:4.25,16:4.25",
    "stage_c_mode": "none",
    "stage_c_cache_mode": "'off'",
    "enable_context_source_skip": "0",
    "semantic_role_policy": "none",
    "semantic_memory_paths": "''",
    "semantic_action_active_chunks": "''",
    "semantic_action_inactive_read_cue_source": "''",
    "enable_swa_overlap_source_replace": "1",
    "swa_overlap_source_replace_alpha": "0.5",
    "swa_overlap_source_replace_mode": "source",
    "swa_overlap_source_replace_target": "kv",
    "ttt_write_gradient_reversal_mode": "tri_replay",
    "ttt_write_gradient_reversal_chunk_gammas": "5:0.005,6:0.005,7:0.005,8:0.005,9:0.005,10:0.003,11:0.003,12:0.003,16:0.0003",
    "ttt_write_gradient_reversal_gamma": "0.0",
    "ttt_write_gradient_reversal_branch_mask": "'0'",
    "ttt_write_gradient_reversal_risk_source": "update_conflict_energy",
    "ttt_write_commit_ema_alpha": "0.5",
    "ttt_write_commit_ema_branch_mask": "'0'",
    "ttt_write_commit_ema_chunks": "5,6",
    "ttt_write_native_mix_scales": "1.10,1.00,1.00",
}


def _read_config(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" not in line or line.startswith(" "):
            continue
        key, value = line.split(":", 1)
        out[key.strip()] = value.strip()
    return out


def _read_ate(path: Path) -> float:
    if not path.exists():
        return float("nan")
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.strip().split()
        if len(parts) >= 2 and parts[0] == "01":
            return float(parts[1])
    return float("nan")


def _jsonl_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip())


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--historical-c9-ate", type=float, default=HISTORICAL_C9_ATE)
    parser.add_argument("--ate-tol", type=float, default=0.03)
    args = parser.parse_args()

    cfg_path = args.run_dir / "hmc_config.yaml"
    config = _read_config(cfg_path)
    diffs = []
    for key, expected in EXPECTED.items():
        actual = config.get(key)
        if actual != expected:
            diffs.append({"key": key, "expected": expected, "actual": actual})

    ate = _read_ate(args.run_dir / "results_sim3" / "results_ate.txt")
    hmc_rows = _jsonl_count(args.run_dir / "hmc_state_hash.jsonl")
    abs_delta = abs(float(ate) - float(args.historical_c9_ate)) if math.isfinite(float(ate)) else None
    config_pass = len(diffs) == 0
    summary = {
        "run_dir": str(args.run_dir),
        "ATE": ate,
        "historical_c9_ate": float(args.historical_c9_ate),
        "abs_delta_vs_historical_c9": abs_delta,
        "ate_gate_pass": bool(abs_delta is not None and abs_delta <= float(args.ate_tol)),
        "hmc_rows": hmc_rows,
        "hmc_rows_gate_pass": hmc_rows == 38,
        "effective_config_unexpected_diff_count": len(diffs),
        "effective_config_gate_pass": config_pass,
        "stage_c_disabled": config.get("stage_c_mode") == "none",
        "semantic_action_disabled": config.get("semantic_role_policy") == "none"
        and config.get("semantic_memory_paths") == "''"
        and config.get("semantic_action_active_chunks") == "''",
    }
    summary["phase0_noop_gate_pass"] = bool(
        summary["ate_gate_pass"]
        and summary["hmc_rows_gate_pass"]
        and summary["effective_config_gate_pass"]
        and summary["stage_c_disabled"]
        and summary["semantic_action_disabled"]
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if cfg_path.exists():
        shutil.copyfile(cfg_path, args.out_dir / "effective_config.yaml")
    _write_json(args.out_dir / "effective_config_diff_vs_C9.json", {"diffs": diffs, "expected": EXPECTED})
    _write_json(args.out_dir / "noop_gate_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
