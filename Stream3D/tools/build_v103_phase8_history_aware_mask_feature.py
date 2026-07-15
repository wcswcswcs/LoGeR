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
import torch


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
PHASE_ID = "v103_phase8_history_aware_mask_feature"
DEFAULT_SCENES = ("scene0011_00", "scene0050_00")


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


def _norm_rows(arr: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    norm = np.linalg.norm(arr, axis=1, keepdims=True)
    out = arr / np.maximum(norm, float(eps))
    out[~np.isfinite(out)] = 0.0
    return out.astype(np.float32, copy=False)


def _history_projection(history_count: int, dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed) + 1009 * int(history_count) + 9173 * int(dim))
    proj = rng.standard_normal((int(history_count), int(dim))).astype(np.float32)
    return _norm_rows(proj)


def _load_phase7_payload(phase7_root: Path) -> dict[str, Any]:
    path = phase7_root / "history_token_feature_rows.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu")
    if "payload_by_scene" not in payload:
        raise KeyError(f"missing payload_by_scene in {path}")
    return payload


def _scene_history_matrix(
    *,
    scene: str,
    local_payload: dict[str, Any],
    incidence_payload: dict[str, Any],
    token_payload: dict[str, Any],
    control_mode: str,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    carrier_ids_phase4 = incidence_payload["carrier_id"].cpu().numpy().astype(np.int64)
    carrier_local = incidence_payload["carrier_local_index"].cpu().numpy().astype(np.int64)
    mask_idx = incidence_payload["mask_observation_index"].cpu().numpy().astype(np.int64)
    b_ia = incidence_payload["B_ia"].cpu().numpy().astype(np.float32)
    mask_count = int(local_payload["feature"].shape[0])

    token_carrier_ids = token_payload["carrier_ids"].cpu().numpy().astype(np.int64)
    assigned = token_payload["h2_assigned"].cpu().numpy().astype(bool)
    top1 = token_payload["h2_top1_history_index"].cpu().numpy().astype(np.int64)
    top1_score = token_payload["h2_top1_score"].cpu().numpy().astype(np.float32)
    margin = token_payload["h2_margin"].cpu().numpy().astype(np.float32)
    entropy = token_payload["h2_entropy"].cpu().numpy().astype(np.float32)
    history_count = len(token_payload.get("history_ids", []))
    if history_count <= 0:
        raise ValueError(f"{scene}: no history ids in token payload")

    id_to_token = {int(cid): int(i) for i, cid in enumerate(token_carrier_ids.tolist())}
    token_index_for_phase4 = np.full((carrier_ids_phase4.shape[0],), -1, dtype=np.int64)
    for i, cid in enumerate(carrier_ids_phase4.tolist()):
        token_index_for_phase4[i] = id_to_token.get(int(cid), -1)
    valid_carrier = token_index_for_phase4 >= 0
    valid_token_count = int(np.count_nonzero(valid_carrier))

    token_top1 = top1.copy()
    if control_mode == "shuffle_history":
        rng = np.random.default_rng(int(seed) + sum(ord(c) for c in scene))
        assigned_indices = np.flatnonzero(assigned)
        shuffled = token_top1[assigned_indices].copy()
        rng.shuffle(shuffled)
        token_top1[assigned_indices] = shuffled
    elif control_mode != "real":
        raise ValueError(f"unsupported control_mode={control_mode}")

    row_token = token_index_for_phase4[carrier_local]
    row_ok = row_token >= 0
    row_assigned = np.zeros((row_token.shape[0],), dtype=bool)
    row_hist = np.zeros((row_token.shape[0],), dtype=np.int64)
    row_score = np.zeros((row_token.shape[0],), dtype=np.float32)
    if np.any(row_ok):
        rt = row_token[row_ok]
        row_assigned[row_ok] = assigned[rt]
        row_hist[row_ok] = token_top1[rt]
        row_score[row_ok] = top1_score[rt]
    row_ok &= row_assigned
    row_ok &= (row_hist >= 0) & (row_hist < history_count)
    row_weight = b_ia * np.clip(row_score, 0.0, 1.0)

    hist = np.zeros((mask_count, history_count), dtype=np.float32)
    denom = np.zeros((mask_count,), dtype=np.float32)
    if np.any(row_ok):
        np.add.at(hist, (mask_idx[row_ok], row_hist[row_ok]), row_weight[row_ok])
        np.add.at(denom, mask_idx[row_ok], row_weight[row_ok])
    nonzero = denom > 0
    hist[nonzero] = hist[nonzero] / np.maximum(denom[nonzero, None], 1e-12)
    hist = _norm_rows(hist)

    assigned_token_count = int(np.count_nonzero(assigned))
    meta = {
        "scene_id": scene,
        "history_object_count": int(history_count),
        "phase7_token_carrier_count": int(token_carrier_ids.shape[0]),
        "phase4_carrier_count": int(carrier_ids_phase4.shape[0]),
        "phase4_carrier_with_phase7_token_count": valid_token_count,
        "phase4_carrier_with_phase7_token_rate": float(valid_token_count / max(1, carrier_ids_phase4.shape[0])),
        "assigned_token_count": assigned_token_count,
        "assigned_token_rate": float(assigned_token_count / max(1, token_carrier_ids.shape[0])),
        "mask_count": int(mask_count),
        "mask_with_history_token_count": int(np.count_nonzero(nonzero)),
        "mask_with_history_token_rate": float(np.mean(nonzero)) if nonzero.size else 0.0,
        "history_token_top1_score_mean_assigned": float(np.mean(top1_score[assigned])) if np.any(assigned) else 0.0,
        "history_token_margin_mean_assigned": float(np.mean(margin[assigned])) if np.any(assigned) else 0.0,
        "history_token_entropy_mean_assigned": float(np.mean(entropy[assigned])) if np.any(assigned) else 1.0,
        "control_mode": control_mode,
    }
    return hist, meta


def _run_scene(scene: str, args: argparse.Namespace, phase7_payload: dict[str, Any]) -> dict[str, Any]:
    phase5_root = _project(args.phase5_root)
    phase4_root = _project(args.phase4_root)
    out_scene = _project(args.output_root) / scene
    out_scene.mkdir(parents=True, exist_ok=True)
    local_path = phase5_root / scene / "mask_level_feature.pt"
    incidence_path = phase4_root / scene / "primitive_incidence_sparse.pt"
    if not local_path.exists():
        raise FileNotFoundError(local_path)
    if not incidence_path.exists():
        raise FileNotFoundError(incidence_path)
    local_payload = torch.load(local_path, map_location="cpu")
    incidence_payload = torch.load(incidence_path, map_location="cpu")
    token_payload = phase7_payload["payload_by_scene"][scene]

    local_feature = local_payload["feature"].to(torch.float32).cpu().numpy().astype(np.float32)
    local_feature = _norm_rows(local_feature)
    hist_matrix, meta = _scene_history_matrix(
        scene=scene,
        local_payload=local_payload,
        incidence_payload=incidence_payload,
        token_payload=token_payload,
        control_mode=str(args.control_mode),
        seed=int(args.seed),
    )
    proj = _history_projection(hist_matrix.shape[1], local_feature.shape[1], int(args.seed))
    hist_feature = _norm_rows(hist_matrix @ proj)
    fused = _norm_rows(local_feature + float(args.history_weight) * hist_feature)
    variant_id = (
        f"H8_{args.control_mode}_lam{str(float(args.history_weight)).replace('.', 'p')}_"
        f"from_{local_payload.get('variant_id', 'local')}"
    )

    out_payload = dict(local_payload)
    out_payload.update(
        {
            "schema_version": "stream4d_v103_phase8_history_aware_mask_feature_v1",
            "phase_id": PHASE_ID,
            "variant_id": variant_id,
            "local_phase5_variant_id": str(local_payload.get("variant_id", "")),
            "history_weight": float(args.history_weight),
            "control_mode": str(args.control_mode),
            "history_object_count": int(hist_matrix.shape[1]),
            "history_projection_seed": int(args.seed),
            "feature": torch.as_tensor(fused, dtype=torch.float16),
            "history_mask_feature": torch.as_tensor(hist_matrix, dtype=torch.float16),
            "uses_gt": False,
            "uses_future": False,
        }
    )
    torch.save(out_payload, out_scene / "mask_level_feature.pt")
    np.savez_compressed(
        out_scene / "history_mask_feature_debug.npz",
        history_mask_feature=hist_matrix.astype(np.float32),
        projected_history_feature=hist_feature.astype(np.float32),
        local_feature=local_feature.astype(np.float32),
        fused_feature=fused.astype(np.float32),
    )
    row = {
        "schema_version": "stream4d_v103_phase8_history_feature_metric_row_v1",
        "phase_id": PHASE_ID,
        "variant_id": variant_id,
        "local_phase5_variant_id": str(local_payload.get("variant_id", "")),
        "history_weight": float(args.history_weight),
        "uses_gt_for_feature": False,
        "uses_future": False,
        **meta,
    }
    return row


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build v103 Phase8 history-aware mask features for Phase6 reuse.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--phase5-root", required=True)
    parser.add_argument("--phase4-root", required=True)
    parser.add_argument("--phase7-root", required=True)
    parser.add_argument("--history-weight", type=float, default=0.50)
    parser.add_argument("--control-mode", choices=["real", "shuffle_history"], default="real")
    parser.add_argument("--seed", type=int, default=10381)
    parser.add_argument("--scene", choices=["all", *DEFAULT_SCENES], default="all")
    return parser


def main() -> int:
    t0 = time.time()
    args = build_parser().parse_args()
    out = _project(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    (out / "last_command.txt").write_text(" ".join([sys.executable, str(Path(__file__)), *sys.argv[1:]]) + "\n", encoding="utf-8")
    phase7_payload = _load_phase7_payload(_project(args.phase7_root))
    scenes = list(DEFAULT_SCENES) if args.scene == "all" else [str(args.scene)]
    rows = [_run_scene(scene, args, phase7_payload) for scene in scenes]
    _write_csv(out / "history_feature_metric_rows.csv", rows)
    _write_csv(
        out / "gate_rows.csv",
        [
            {
                "schema_version": "stream4d_v103_phase8_history_feature_gate_row_v1",
                "phase_id": PHASE_ID,
                "scene_id": row["scene_id"],
                "variant_id": row["variant_id"],
                "gate_name": "mask_with_history_token_rate_gt_0",
                "pass": float(row["mask_with_history_token_rate"]) > 0.0,
                "observed": row["mask_with_history_token_rate"],
                "required": ">0",
            }
            for row in rows
        ],
    )
    failure_rows = [
        {
            "schema_version": "stream4d_v103_phase8_history_feature_failure_row_v1",
            "phase_id": PHASE_ID,
            "scene_id": row["scene_id"],
            "variant_id": row["variant_id"],
            "failure_id": "no_history_token_masks",
            "severity": "blocking",
            "evidence": f"mask_with_history_token_rate={row['mask_with_history_token_rate']}",
        }
        for row in rows
        if float(row["mask_with_history_token_rate"]) <= 0.0
    ]
    _write_csv(out / "failure_rows.csv", failure_rows)
    summary = {
        "schema_version": "stream4d_v103_phase8_history_aware_mask_feature_summary_v1",
        "phase_id": PHASE_ID,
        "created_unix_time": time.time(),
        "runtime_sec": time.time() - t0,
        "phase5_pass": len(failure_rows) == 0,
        "phase8_feature_pass": len(failure_rows) == 0,
        "decision": "PASS_READY_FOR_PHASE6_HISTORY_AWARE_CLUSTERING" if not failure_rows else "NO_GO_PHASE8_HISTORY_FEATURE",
        "failure_count": len(failure_rows),
        "phase5_root": _rel(_project(args.phase5_root)),
        "phase4_root": _rel(_project(args.phase4_root)),
        "phase7_root": _rel(_project(args.phase7_root)),
        "history_weight": float(args.history_weight),
        "control_mode": str(args.control_mode),
        "scene_ids": scenes,
        "truthfulness_note": "This script injects pre-update Phase7 history tokens into mask features and writes a Phase5-like root for existing Phase6 clustering. It does not update history memory or compute scene AP by itself.",
        "uses_gt_for_feature": False,
        "uses_future": False,
        "outputs": {
            "summary": _rel(out / "summary.json"),
            "history_feature_metric_rows": _rel(out / "history_feature_metric_rows.csv"),
            "gate_rows": _rel(out / "gate_rows.csv"),
            "failure_rows": _rel(out / "failure_rows.csv"),
            "last_command": _rel(out / "last_command.txt"),
        },
    }
    _write_json(out / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if not failure_rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
