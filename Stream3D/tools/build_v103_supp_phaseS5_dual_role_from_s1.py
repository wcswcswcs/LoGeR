#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
AUDIT_ROOT = STREAM3D_ROOT / "outputs" / "audit"

PHASE_ID = "v103_supp_phaseS5_dual_role_from_s1"
DEFAULT_OUT = AUDIT_ROOT / "v103_supp_phaseS5_dual_role_from_s1_d4rt48mix_r1"
DEFAULT_PHASES1_ROOT = AUDIT_ROOT / "v103_supp_phaseS1_multirole_carriers"
DEFAULT_SCENE0011_PHASE2 = (
    AUDIT_ROOT / "v103_phase2_stratified_q5c_objlike16384_scene0011_c0001_cap24576_qchunk16384_d4rt48mix_maskbalanced8"
)
DEFAULT_SCENE0050_PHASE2 = (
    AUDIT_ROOT / "v103_phase2_stratified_q5c_objlike16384_scene0050_c0001_cap24576_qchunk16384_d4rt48mix_maskbalanced8"
)


def _project(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if p.parts and p.parts[0] == "Stream3D":
        return REPO_ROOT / p
    return STREAM3D_ROOT / p


def _rel(path: str | Path) -> str:
    p = Path(path)
    try:
        return p.resolve().relative_to(REPO_ROOT).as_posix()
    except Exception:
        return p.as_posix()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return _rel(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields or ["schema_version"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key, "")) for key in fields})


def _scene_phase2_roots(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "scene0011_00": _project(args.scene0011_phase2_root),
        "scene0050_00": _project(args.scene0050_phase2_root),
    }


def _load_carrier_ids(phase2_root: Path) -> np.ndarray:
    mmap_path = phase2_root / "carrier_batch_mmap_cache" / "carrier_id.npy"
    if mmap_path.exists():
        return np.load(mmap_path).astype(np.int64, copy=False)
    npz_path = phase2_root / "carrier_batch.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"missing carrier batch under {phase2_root}")
    with np.load(npz_path) as pack:
        return np.asarray(pack["carrier_id"], dtype=np.int64)


def _build_scene(
    *,
    scene_id: str,
    phase2_root: Path,
    s1_rows: pd.DataFrame,
    out: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    carrier_ids = _load_carrier_ids(phase2_root)
    scene_rows = s1_rows[s1_rows["scene_id"].astype(str) == scene_id].copy()
    if scene_rows.empty:
        raise RuntimeError(f"no Phase S1 role rows for {scene_id}")
    scene_rows["carrier_id"] = scene_rows["carrier_id"].astype(np.int64)
    if scene_rows["carrier_id"].duplicated().any():
        dup_count = int(scene_rows["carrier_id"].duplicated().sum())
        raise RuntimeError(f"Phase S1 role rows have duplicated carrier_id for {scene_id}: {dup_count}")

    role_by_id = scene_rows.set_index("carrier_id", drop=False)
    phase2_id_index = pd.Index(carrier_ids)
    missing_in_phase2 = int((~pd.Index(role_by_id.index).isin(phase2_id_index)).sum())
    missing_in_s1 = int((~phase2_id_index.isin(role_by_id.index)).sum())
    if missing_in_phase2 or missing_in_s1:
        raise RuntimeError(
            f"carrier_id mismatch for {scene_id}: missing_s1_in_phase2={missing_in_phase2} phase2_not_in_s1={missing_in_s1}"
        )

    ordered = role_by_id.loc[carrier_ids]
    positive_mask = ordered["is_A_anchor"].astype(bool).to_numpy()
    veto_mask = ordered["is_V_veto"].astype(bool).to_numpy()
    support_mask = ordered["is_S_support"].astype(bool).to_numpy()
    uncertain_mask = ordered["is_U_uncertain"].astype(bool).to_numpy()

    scene_out = out / scene_id
    scene_out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        scene_out / "dual_role_carrier_sets.npz",
        carrier_id=carrier_ids,
        positive_mask=positive_mask,
        veto_mask=veto_mask,
        support_mask=support_mask,
        uncertain_mask=uncertain_mask,
    )

    n = int(carrier_ids.shape[0])
    role_rows = [
        ("positive_core", positive_mask),
        ("veto_support", veto_mask),
        ("support_diagnostic", support_mask),
        ("uncertain_diagnostic", uncertain_mask),
        ("union_positive_veto", positive_mask | veto_mask),
    ]
    metric_rows: list[dict[str, Any]] = []
    for role, mask in role_rows:
        metric_rows.append(
            {
                "schema_version": "stream4d_v103_supp_phaseS5_dual_role_from_s1_metric_row_v1",
                "phase_id": PHASE_ID,
                "scene_id": scene_id,
                "role": role,
                "carrier_count": int(np.count_nonzero(mask)),
                "total_carrier_count": n,
                "retained_carrier_rate": float(np.count_nonzero(mask) / max(n, 1)),
                "uses_gt_for_prediction": False,
                "uses_gt_for_selection": False,
            }
        )

    artifact_row = {
        "schema_version": "stream4d_v103_supp_phaseS5_dual_role_from_s1_artifact_row_v1",
        "phase_id": PHASE_ID,
        "scene_id": scene_id,
        "artifact_id": "dual_role_carrier_sets",
        "path": _rel(scene_out / "dual_role_carrier_sets.npz"),
        "phase2_root": _rel(phase2_root),
        "positive_count": int(np.count_nonzero(positive_mask)),
        "veto_count": int(np.count_nonzero(veto_mask)),
        "support_count": int(np.count_nonzero(support_mask)),
        "uncertain_count": int(np.count_nonzero(uncertain_mask)),
        "overlap_positive_veto_count": int(np.count_nonzero(positive_mask & veto_mask)),
        "carrier_id_join_missing_count": 0,
        "uses_gt_for_prediction": False,
        "uses_gt_for_selection": False,
    }
    return artifact_row, metric_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Phase9e dual-role masks from v103 supplement S1 role rows.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--phaseS1-root", default=str(DEFAULT_PHASES1_ROOT))
    parser.add_argument("--scene0011-phase2-root", default=str(DEFAULT_SCENE0011_PHASE2))
    parser.add_argument("--scene0050-phase2-root", default=str(DEFAULT_SCENE0050_PHASE2))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    started = time.time()
    out = _project(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    (out / "last_command.txt").write_text(" ".join([sys.executable, *sys.argv]) + "\n", encoding="utf-8")

    phaseS1_root = _project(args.phaseS1_root)
    s1_rows = pd.read_parquet(phaseS1_root / "carrier_role_rows.parquet")
    phase2_roots = _scene_phase2_roots(args)
    artifact_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    for scene_id, phase2_root in phase2_roots.items():
        try:
            artifact_row, rows = _build_scene(scene_id=scene_id, phase2_root=phase2_root, s1_rows=s1_rows, out=out)
            artifact_rows.append(artifact_row)
            metric_rows.extend(rows)
        except Exception as exc:
            failure_rows.append(
                {
                    "schema_version": "stream4d_v103_supp_phaseS5_dual_role_from_s1_failure_row_v1",
                    "phase_id": PHASE_ID,
                    "scene_id": scene_id,
                    "failure_id": "dual_role_from_s1_build_failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "uses_gt_for_prediction": False,
                }
            )

    gate_rows = [
        {
            "schema_version": "stream4d_v103_supp_phaseS5_dual_role_from_s1_gate_row_v1",
            "phase_id": PHASE_ID,
            "gate_name": "all_scene_carrier_id_join_exact",
            "pass": len(failure_rows) == 0,
            "observed": f"failure_count={len(failure_rows)}",
            "required": "0 failures",
            "uses_gt": False,
        }
    ]
    _write_csv(out / "artifact_rows.csv", artifact_rows)
    _write_csv(out / "dual_role_metric_rows.csv", metric_rows)
    _write_csv(out / "gate_rows.csv", gate_rows)
    _write_csv(out / "failure_rows.csv", failure_rows)
    selected_roles = {
        scene_id: {"positive_core": "S1_A_anchor", "veto_support": "S1_V_veto"}
        for scene_id in phase2_roots
        if scene_id not in {str(row.get("scene_id")) for row in failure_rows}
    }
    summary = {
        "schema_version": "stream4d_v103_supp_phaseS5_dual_role_from_s1_summary_v1",
        "phase_id": PHASE_ID,
        "created_unix_time": time.time(),
        "runtime_sec": time.time() - started,
        "decision": "PASS_PHASES5_DUAL_ROLE_FROM_S1" if not failure_rows else "NO_GO_PHASES5_DUAL_ROLE_FROM_S1",
        "phaseS1_root": _rel(phaseS1_root),
        "selected_roles_by_scene": selected_roles,
        "scene_count": len(phase2_roots),
        "failure_count": len(failure_rows),
        "truthfulness_note": "This artifact converts supplement Phase S1 role rows into Phase9e-compatible boolean masks using exact carrier_id joins. It does not use GT and does not claim DA3 or object AP success.",
        "outputs": {
            "summary": _rel(out / "summary.json"),
            "artifact_rows": _rel(out / "artifact_rows.csv"),
            "dual_role_metric_rows": _rel(out / "dual_role_metric_rows.csv"),
            "gate_rows": _rel(out / "gate_rows.csv"),
            "failure_rows": _rel(out / "failure_rows.csv"),
            "last_command": _rel(out / "last_command.txt"),
        },
        "uses_gt_for_prediction": False,
        "uses_gt_for_selection": False,
    }
    _write_json(out / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if not failure_rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
