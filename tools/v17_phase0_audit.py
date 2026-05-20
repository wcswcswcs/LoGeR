#!/usr/bin/env python3
"""Audit ACL2 v17 Phase 0 boundary from locked v16 artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, List, Mapping


RUNS = {
    "H9_P0_R2": "phase0_boundary/V16_P0_R2_H9_locked_exact_merge_input_SWKS3",
    "C9_P0_R2": "phase0_boundary/V16_P0_R2_C9_locked_exact_merge_input_SWKS3",
    "WINGAM_P0_R3": "phase0_boundary/V16_P0_R3_WINGAM_locked_exact_merge_input_SWKS3",
}

EXPECTED = {
    "H9_P0_R2": {
        "ate_rmse": 34.125776940054486,
        "seg_200_300": 74.40992731884393,
        "seg_400_600": 44.3536379244608,
        "mp_alpha": "0.125",
        "read_beta_frame_chunks": "5:4.85,6:4.85,7:4.85,8:4.85,9:4.85,10:4.25,11:4.25,12:4.25,16:4.25",
        "ttt_write_gradient_reversal_chunk_gammas": "5:0.005,6:0.005,7:0.005,8:0.005,9:0.005,10:0.003,11:0.003,12:0.003,16:0.0003",
    },
    "C9_P0_R2": {
        "ate_rmse": 33.76294210291885,
        "seg_200_300": 76.10213555431245,
        "seg_400_600": 41.896364212570404,
        "mp_alpha": "0.1",
        "read_beta_frame_chunks": "5:4.85,6:4.85,7:4.85,8:4.85,9:4.85,10:4.25,11:4.25,12:4.25,16:4.25",
        "ttt_write_gradient_reversal_chunk_gammas": "5:0.005,6:0.005,7:0.005,8:0.005,9:0.005,10:0.003,11:0.003,12:0.003,16:0.0003",
    },
    "WINGAM_P0_R3": {
        "ate_rmse": 34.19027827324245,
        "seg_200_300": 75.57602117933182,
        "seg_400_600": 42.28048494410868,
        "mp_alpha": "0.125",
        "read_beta_frame_chunks": "",
        "ttt_write_gradient_reversal_chunk_gammas": "5:0.005,6:0.005,7:0.005,8:0.005,9:0.005,10:0.0030,11:0.0030,12:0.0030",
    },
}

KEYS = [
    "mp_alpha",
    "reset_every",
    "read_beta_frame_chunks",
    "ttt_write_gradient_reversal_mode",
    "ttt_write_gradient_reversal_risk_source",
    "ttt_write_gradient_reversal_chunk_gammas",
    "ttt_write_tri_replay_positive_frac",
    "ttt_write_tri_replay_negative_frac",
    "ttt_write_tri_replay_neutral_lambda",
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_simple_yaml(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" not in line or line.lstrip() != line:
            continue
        key, value = line.split(":", 1)
        out[key.strip()] = value.strip().strip("'\"")
    return out


def _read_registry(path: Path) -> Dict[str, Mapping[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["run"]: row for row in csv.DictReader(handle)}


def _line_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _float_close(actual: str, expected: float, tol: float = 0.03) -> bool:
    try:
        return abs(float(actual) - expected) <= tol
    except ValueError:
        return False


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
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


def _write_json(path: Path, rows: List[Dict[str, object]]) -> None:
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v16-root", default="results/kitti01_hmc_v2/acl2_v16_ttt_causalfork_candidatebank_target25")
    parser.add_argument("--out-dir", default="results/kitti01_hmc_v2/acl2_v17_ttt_write_causalstate_reboot_target25/phase0_boundary")
    args = parser.parse_args()

    v16_root = Path(args.v16_root)
    out_dir = Path(args.out_dir)
    registry = _read_registry(v16_root / "registry_v16_phase0_boundary_R3.csv")
    parity_rows = list(csv.DictReader((v16_root / "phase1_causalfork/phase1_causalfork_gate_summary_R3.csv").open("r", encoding="utf-8")))

    config_rows: List[Dict[str, object]] = []
    metric_rows: List[Dict[str, object]] = []
    snapshot_rows: List[Dict[str, object]] = []

    for run, rel in RUNS.items():
        run_dir = v16_root / rel
        config = run_dir / "hmc_config.yaml"
        values = _parse_simple_yaml(config)
        expected = EXPECTED[run]
        row: Dict[str, object] = {
            "run": run,
            "run_dir": str(run_dir),
            "hmc_config_sha256": _sha256(config),
            "config_exists": config.exists(),
            "config_gate_pass": True,
        }
        for key in KEYS:
            actual = values.get(key, "")
            row[key] = actual
            if key in expected and actual != expected[key]:
                row["config_gate_pass"] = False
        if values.get("reset_every", "") != "5":
            row["config_gate_pass"] = False
        config_rows.append(row)

        reg = registry[run]
        metric_gate = (
            _float_close(reg["ate_rmse"], float(expected["ate_rmse"]))
            and _float_close(reg["seg_200_300"], float(expected["seg_200_300"]))
            and _float_close(reg["seg_400_600"], float(expected["seg_400_600"]))
            and reg.get("counts_as_ttt_write") == "True"
            and reg.get("no_gt_runtime_action") == "True"
            and reg.get("no_external_trajectory_rewrite") == "True"
            and reg.get("hmc_state_rows") == "38"
        )
        metric_rows.append({
            "run": run,
            "ate_rmse": reg["ate_rmse"],
            "seg_200_300": reg["seg_200_300"],
            "seg_400_600": reg["seg_400_600"],
            "hmc_rows": reg["hmc_state_rows"],
            "metric_gate_pass": metric_gate,
        })

    for chunk in [5, 6, 10, 16]:
        snap = f"{chunk:03d}"
        hmc = v16_root / f"phase1_causalfork/state_snapshots/H9_P0_V16_R2/chunk_{snap}_input.pt"
        merge = v16_root / f"phase1_causalfork/merge_state_snapshots/H9_P0_V16_R2/chunk_{snap}_input.pt"
        snapshot_rows.append({
            "chunk_id": chunk,
            "hmc_snapshot": str(hmc),
            "hmc_exists": hmc.exists(),
            "hmc_size_bytes": hmc.stat().st_size if hmc.exists() else 0,
            "merge_snapshot": str(merge),
            "merge_exists": merge.exists(),
            "merge_size_bytes": merge.stat().st_size if merge.exists() else 0,
            "snapshot_gate_pass": hmc.exists() and merge.exists(),
        })

    parity_pass = all(str(row.get("gate_pass", "")).lower() == "true" for row in parity_rows)
    config_pass = all(bool(row["config_gate_pass"]) for row in config_rows)
    metric_pass = all(bool(row["metric_gate_pass"]) for row in metric_rows)
    snapshot_pass = all(bool(row["snapshot_gate_pass"]) for row in snapshot_rows)
    summary = [{
        "phase": "Phase 0 boundary audit",
        "config_gate_pass": config_pass,
        "metric_gate_pass": metric_pass,
        "snapshot_gate_pass": snapshot_pass,
        "phase1_v16_parity_gate_pass": parity_pass,
        "phase0_gate_pass": config_pass and metric_pass and snapshot_pass and parity_pass,
        "phase1_v16_parity_rows": len(parity_rows),
        "phase1_v16_parity_failures": sum(str(row.get("gate_pass", "")).lower() != "true" for row in parity_rows),
    }]

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "v17_phase0_config_audit.csv", config_rows)
    _write_json(out_dir / "v17_phase0_config_audit.json", config_rows)
    _write_csv(out_dir / "v17_phase0_metric_audit.csv", metric_rows)
    _write_json(out_dir / "v17_phase0_metric_audit.json", metric_rows)
    _write_csv(out_dir / "v17_phase0_snapshot_audit.csv", snapshot_rows)
    _write_json(out_dir / "v17_phase0_snapshot_audit.json", snapshot_rows)
    _write_csv(out_dir / "v17_phase0_gate_summary.csv", summary)
    _write_json(out_dir / "v17_phase0_gate_summary.json", summary)

    md = [
        "# ACL2 v17 Phase 0 Boundary Audit",
        "",
        f"Phase 0 gate pass: `{summary[0]['phase0_gate_pass']}`",
        "",
        "Inputs are locked v16 boundary artifacts. No new full run is counted here.",
        "",
        "## Summary",
        "",
        f"- config gate: `{config_pass}`",
        f"- metric gate: `{metric_pass}`",
        f"- snapshot gate: `{snapshot_pass}`",
        f"- v16 causal-fork parity gate: `{parity_pass}`",
        "",
    ]
    out_dir.joinpath("v17_phase0_boundary_audit.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
