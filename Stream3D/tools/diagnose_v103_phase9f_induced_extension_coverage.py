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
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_v103_phase3_fast_carrier_reliability_filter import _compute_scene_arrays, _project  # noqa: E402


PHASE_ID = "v103_phase9f_induced_extension_coverage"
AUDIT_ROOT = STREAM3D_ROOT / "outputs/audit"
PLAN_DOC = REPO_ROOT / "docs/stream4d_v103_training_free_primitive_affinity_field_experiment_plan.md"

DEFAULT_PHASE9E_ROOT = AUDIT_ROOT / "v103_phase9e_d4rt_anchor_da3_induced_carriers_r4_all_save_clean"
DEFAULT_OUT = AUDIT_ROOT / "v103_phase9f_induced_extension_coverage_r1"
DEFAULT_SCENE0011_PHASE2 = AUDIT_ROOT / "v103_phase2_stratified_q5c_objlike16384_scene0011_first32"
DEFAULT_SCENE0050_PHASE2 = AUDIT_ROOT / "v103_phase2_stratified_q5c_objlike16384_scene0050_first32"

SCENE_SPECS = {
    "scene0011_00": {
        "phase2_root": DEFAULT_SCENE0011_PHASE2,
        "semantic_npz": AUDIT_ROOT / "v91_radio_mask_features_npz_scene0011/mask_features.npz",
        "semantic_rows": AUDIT_ROOT / "v91_radio_mask_features_npz_scene0011/mask_feature_rows.csv",
    },
    "scene0050_00": {
        "phase2_root": DEFAULT_SCENE0050_PHASE2,
        "semantic_npz": AUDIT_ROOT / "v91_radio_mask_features_npz_scene0050/mask_features.npz",
        "semantic_rows": AUDIT_ROOT / "v91_radio_mask_features_npz_scene0050/mask_feature_rows.csv",
    },
}


def _rel(path: Path | str) -> str:
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


def _project_phase(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if p.parts and p.parts[0] == "Stream3D":
        return REPO_ROOT / p
    return STREAM3D_ROOT / p


def _parse_obs(obs: str) -> tuple[str, int, int]:
    scene, frame, mask_id = str(obs).split(":")
    return scene, int(frame), int(mask_id)


def _obs_sets_from_clean_rows(rows: pd.DataFrame) -> tuple[set[str], set[str]]:
    accepted_obs: set[str] = set()
    induced_obs: set[str] = set()
    for row in rows.itertuples(index=False):
        obs_a = str(row.mask_a_observation_id)
        obs_b = str(row.mask_b_observation_id)
        pos_a = int(getattr(row, "d4rt_selected_seed_support_a", row.d4rt_positive_support_a))
        pos_b = int(getattr(row, "d4rt_selected_seed_support_b", row.d4rt_positive_support_b))
        accepted_obs.add(obs_a)
        accepted_obs.add(obs_b)
        if pos_a == 0 and pos_b > 0:
            induced_obs.add(obs_a)
        if pos_b == 0 and pos_a > 0:
            induced_obs.add(obs_b)
    return accepted_obs, induced_obs


def _obs_meta(diag: dict[str, Any]) -> dict[str, dict[str, Any]]:
    scene = str(diag["scene_id"])
    frame_ids = [int(v) for v in diag["frame_ids"]]
    object_like_by_frame = {int(k): np.asarray(v, dtype=np.int32) for k, v in dict(diag["object_like_by_frame"]).items()}
    broad_map = np.asarray(diag["broad_map"], dtype=bool)
    object_map = np.asarray(diag["object_map"], dtype=bool)
    meta: dict[str, dict[str, Any]] = {}
    for fi, frame_id in enumerate(frame_ids):
        object_like_labels = set(int(v) for v in object_like_by_frame.get(fi, np.asarray([], dtype=np.int32)).tolist())
        for label in np.unique(diag["masks"][fi]).astype(int).tolist():
            if label <= 0:
                continue
            obs = f"{scene}:{frame_id}:{int(label)}"
            safe = min(int(label), broad_map.shape[1] - 1)
            meta[obs] = {
                "frame_id": int(frame_id),
                "mask_id": int(label),
                "is_object_like": bool(int(label) in object_like_labels or object_map[fi, safe]),
                "is_broad": bool(broad_map[fi, safe]),
            }
    return meta


def _summarize_obs(prefix: str, obs_set: set[str], meta: dict[str, dict[str, Any]]) -> dict[str, Any]:
    known = [obs for obs in obs_set if obs in meta]
    object_like = [obs for obs in known if bool(meta[obs]["is_object_like"])]
    broad = [obs for obs in known if bool(meta[obs]["is_broad"])]
    return {
        f"{prefix}_obs_count": int(len(obs_set)),
        f"{prefix}_known_obs_count": int(len(known)),
        f"{prefix}_object_like_obs_count": int(len(object_like)),
        f"{prefix}_object_like_obs_rate": float(len(object_like) / max(len(known), 1)),
        f"{prefix}_broad_obs_count": int(len(broad)),
        f"{prefix}_broad_obs_rate": float(len(broad) / max(len(known), 1)),
    }


def _process_scene(
    scene_id: str,
    phase9e_root: Path,
    out: Path,
    device_id: int,
    phase2_root_by_scene: dict[str, Path] | None = None,
) -> dict[str, Any]:
    scene_dir = phase9e_root / scene_id
    support_path = scene_dir / "d4rt_positive_anchor_support_rows.csv"
    clean_path = scene_dir / "best_clean_variant_accepted_pair_rows.parquet"
    variant_path = scene_dir / "induced_variant_rows.csv"
    if not support_path.exists() or not clean_path.exists():
        return {
            "schema_version": "stream4d_v103_phase9f_failure_row_v1",
            "phase_id": PHASE_ID,
            "scene_id": scene_id,
            "blocker": "phase9e_clean_inputs_missing",
            "support_path": _rel(support_path),
            "clean_path": _rel(clean_path),
            "uses_gt_for_prediction": False,
        }

    spec = dict(SCENE_SPECS[scene_id])
    if phase2_root_by_scene and scene_id in phase2_root_by_scene:
        spec["phase2_root"] = phase2_root_by_scene[scene_id]
    spec["phase2_root"] = _project(spec["phase2_root"])
    diag, _unused_a, _unused_b, _arrays = _compute_scene_arrays(scene_id, spec, out / scene_id, int(device_id))
    meta = _obs_meta(diag)

    support_df = pd.read_csv(support_path)
    anchor_obs = set(str(v) for v in support_df["mask_observation_id"].tolist())
    clean_df = pd.read_parquet(clean_path)
    accepted_obs, induced_obs = _obs_sets_from_clean_rows(clean_df)
    union_obs = anchor_obs | induced_obs
    new_object_like_obs = {obs for obs in induced_obs - anchor_obs if obs in meta and bool(meta[obs]["is_object_like"])}
    new_broad_obs = {obs for obs in induced_obs - anchor_obs if obs in meta and bool(meta[obs]["is_broad"])}

    variant_df = pd.read_csv(variant_path)
    clean_variant_id = str(clean_df["phase9e_variant_id"].iloc[0]) if len(clean_df) else ""
    vrow = variant_df.loc[variant_df["variant_id"].astype(str) == clean_variant_id]
    vbits = vrow.iloc[0].to_dict() if len(vrow) else {}

    row: dict[str, Any] = {
        "schema_version": "stream4d_v103_phase9f_scene_summary_row_v1",
        "phase_id": PHASE_ID,
        "scene_id": scene_id,
        "phase9e_root": _rel(phase9e_root),
        "clean_variant_id": clean_variant_id,
        "clean_variant_global_recall": vbits.get("same_object_bridge_recall_global", ""),
        "clean_variant_anchor_reachable_recall": vbits.get("same_object_bridge_recall_anchor_reachable", ""),
        "clean_variant_different_gt_false_bridge_among_accepted": vbits.get("different_gt_false_bridge_among_accepted", ""),
        "clean_variant_uses_gt_for_diagnostic_labels": True,
        "new_induced_obs_count": int(len(induced_obs - anchor_obs)),
        "new_induced_object_like_obs_count": int(len(new_object_like_obs)),
        "new_induced_broad_obs_count": int(len(new_broad_obs)),
        "extension_coverage_gate_pass": bool(len(new_object_like_obs) >= 30 and len(new_broad_obs) <= max(5, len(new_object_like_obs))),
        "uses_gt_for_prediction": False,
        "uses_gt_for_coverage_gate": False,
    }
    row.update(_summarize_obs("base_anchor", anchor_obs, meta))
    row.update(_summarize_obs("clean_accepted", accepted_obs, meta))
    row.update(_summarize_obs("clean_induced", induced_obs, meta))
    row.update(_summarize_obs("base_plus_induced", union_obs, meta))
    return row


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose v103 Phase9e clean induced extension coverage before Phase4/5 promotion.")
    parser.add_argument("--phase9e-root", default=str(DEFAULT_PHASE9E_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--scene", choices=["all", "scene0011_00", "scene0050_00"], default="all")
    parser.add_argument("--scene0011-phase2-root", default="")
    parser.add_argument("--scene0050-phase2-root", default="")
    parser.add_argument("--cupy-device-id", type=int, default=0)
    return parser


def main() -> int:
    t0 = time.time()
    args = build_parser().parse_args()
    phase9e_root = _project_phase(args.phase9e_root)
    out = _project_phase(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    (out / "last_command.txt").write_text(" ".join([sys.executable, str(Path(__file__)), *sys.argv[1:]]) + "\n", encoding="utf-8")
    scenes = list(SCENE_SPECS.keys()) if args.scene == "all" else [str(args.scene)]
    phase2_root_by_scene = {
        "scene0011_00": _project_phase(args.scene0011_phase2_root) if str(args.scene0011_phase2_root).strip() else _project_phase(DEFAULT_SCENE0011_PHASE2),
        "scene0050_00": _project_phase(args.scene0050_phase2_root) if str(args.scene0050_phase2_root).strip() else _project_phase(DEFAULT_SCENE0050_PHASE2),
    }
    scene_rows = [
        _process_scene(scene, phase9e_root, out, int(args.cupy_device_id), phase2_root_by_scene=phase2_root_by_scene)
        for scene in scenes
    ]
    failure_rows = [row for row in scene_rows if str(row.get("blocker", ""))]
    pass_count = sum(bool(row.get("extension_coverage_gate_pass", False)) for row in scene_rows)
    _write_csv(out / "scene_summary_rows.csv", scene_rows)
    _write_csv(out / "failure_rows.csv", failure_rows)
    decision = (
        "PASS_PHASE9F_INDUCED_EXTENSION_COVERAGE_GATE"
        if pass_count == len(scenes) and not failure_rows
        else "PARTIAL_PHASE9F_INDUCED_EXTENSION_COVERAGE_GATE"
        if pass_count > 0 and not failure_rows
        else "NO_GO_PHASE9F_INDUCED_EXTENSION_COVERAGE_GATE"
    )
    summary = {
        "schema_version": "stream4d_v103_phase9f_induced_extension_coverage_summary_v1",
        "phase_id": PHASE_ID,
        "created_unix_time": time.time(),
        "runtime_sec": time.time() - t0,
        "decision": decision,
        "scene_count": len(scenes),
        "pass_scene_count": pass_count,
        "failure_count": len(failure_rows),
        "plan_doc": _rel(PLAN_DOC),
        "truthfulness_note": (
            "Coverage gates use GT-free object-like/broad mask metadata from the v103 carrier-filtering data model. "
            "Clean variant recall/false fields are copied from Phase9e diagnostic rows and remain diagnostic-only."
        ),
        "outputs": {
            "summary": _rel(out / "summary.json"),
            "scene_summary_rows": _rel(out / "scene_summary_rows.csv"),
            "failure_rows": _rel(out / "failure_rows.csv"),
            "last_command": _rel(out / "last_command.txt"),
        },
    }
    _write_json(out / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if decision.startswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
