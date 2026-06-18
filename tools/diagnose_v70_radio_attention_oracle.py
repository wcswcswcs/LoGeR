#!/usr/bin/env python3
"""Offline v70 RADIO attention correction oracle entrypoint.

This tool intentionally refuses to synthesize LoGeR attention maps. It can only
promote an oracle when real path taps are present. If path taps are missing or
their schema is unknown, it writes an auditable blocker report.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from v70_radio_sidecar_common import parse_chunks, resize_linear, resize_nearest, torch_load, utc_now, write_json, write_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radio-sidecar-dir", type=Path, required=True)
    parser.add_argument("--path-taps-dir", type=Path, required=True)
    parser.add_argument("--target-chunks", required=True)
    parser.add_argument("--modes", default="object_interior_floor,cross_object_veto,postsoftmax_mix")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def _sidecar_paths(root: Path, chunks: list[int]) -> list[Path]:
    paths: list[Path] = []
    for chunk_id in chunks:
        hits = sorted(root.glob(f"chunk_{chunk_id:03d}_*/radio_sidecar.pt"))
        paths.extend(hits)
    return paths


def _tap_candidates(root: Path) -> list[dict[str, Any]]:
    if not root.exists():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".pt", ".pth", ".json", ".jsonl", ".csv"}:
            continue
        rows.append({"path": str(path), "bytes": path.stat().st_size, "suffix": path.suffix.lower()})
    return rows


def _known_attention_payload(path: Path) -> dict[str, Any]:
    if path.suffix.lower() not in {".pt", ".pth"}:
        return {"path": str(path), "loadable": False, "reason": "not_torch"}
    try:
        payload = torch_load(path)
    except Exception as exc:
        return {"path": str(path), "loadable": False, "reason": type(exc).__name__, "message": str(exc)}
    if not isinstance(payload, dict):
        return {"path": str(path), "loadable": True, "known": False, "reason": f"payload_type={type(payload)}"}
    keys = sorted(str(k) for k in payload.keys())
    attention_keys = [k for k in keys if "attn" in k.lower() or "attention" in k.lower()]
    return {
        "path": str(path),
        "loadable": True,
        "known": bool(attention_keys),
        "keys": keys[:50],
        "attention_like_keys": attention_keys[:50],
    }


def _tensor_np(value: Any, *, integer: bool = False) -> np.ndarray:
    if hasattr(value, "detach"):
        arr = value.detach().cpu().numpy()
    else:
        arr = np.asarray(value)
    return arr.astype(np.int64 if integer else np.float32)


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    z = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(z).astype(np.float32)
    return exp / np.maximum(exp.sum(axis=axis, keepdims=True), 1e-12)


def _entropy(attn: np.ndarray) -> float:
    a = np.clip(attn, 1e-12, 1.0)
    return float(np.mean(-np.sum(a * np.log(a), axis=-1)))


def _kl(before: np.ndarray, after: np.ndarray) -> float:
    b = np.clip(before, 1e-12, 1.0)
    a = np.clip(after, 1e-12, 1.0)
    return float(np.mean(np.sum(a * (np.log(a) - np.log(b)), axis=-1)))


def _mass(attn: np.ndarray, mask: np.ndarray) -> float:
    return float(np.mean(np.sum(attn * mask.astype(np.float32), axis=-1)))


def _proxy_attention_metrics_for_chunk(sidecar_path: Path, tap_entry: dict[str, Any], modes: list[str]) -> list[dict[str, Any]]:
    sidecar = torch_load(sidecar_path)
    feature_payload = torch_load(Path(tap_entry["feature_path"]))
    tensor = feature_payload[tap_entry["tensor_key"]].detach().cpu().float().numpy()
    pca = _tensor_np(sidecar["radio_feat_pca"])
    comp = _tensor_np(sidecar["object_component_id"], integer=True)
    risk = np.maximum.reduce(
        [
            _tensor_np(sidecar["object_boundary_score"]),
            _tensor_np(sidecar["radio_dynamic_score"]),
            _tensor_np(sidecar["radio_sky_context_score"]),
            _tensor_np(sidecar["radio_lowtrust_score"]),
        ]
    ).astype(np.float32)
    rows: list[dict[str, Any]] = []
    for layer_row in tap_entry.get("layer_positions", []):
        layer = int(layer_row["layer"])
        pos = int(layer_row["position"])
        loger_feat = tensor[:, pos]
        t, h, w, d = loger_feat.shape
        radio_feat = np.zeros((t, h, w, pca.shape[-1]), dtype=np.float32)
        comp_small = np.zeros((t, h, w), dtype=np.int64)
        risk_small = np.zeros((t, h, w), dtype=np.float32)
        for frame_idx in range(t):
            radio_feat[frame_idx] = resize_linear(pca[frame_idx], (h, w))
            comp_small[frame_idx] = resize_nearest(comp[frame_idx].astype(np.int32), (h, w)).astype(np.int64)
            risk_small[frame_idx] = resize_linear(risk[frame_idx], (h, w))
        lf = loger_feat.reshape(t, h * w, d)
        lf = lf / np.maximum(np.linalg.norm(lf, axis=-1, keepdims=True), 1e-6)
        rf = radio_feat.reshape(t, h * w, radio_feat.shape[-1])
        rf = rf / np.maximum(np.linalg.norm(rf, axis=-1, keepdims=True), 1e-6)
        comp_flat = comp_small.reshape(t, h * w)
        risk_flat = np.clip(risk_small.reshape(t, h * w), 0.0, 1.0)
        for mode in modes:
            frame_metrics_by_control: dict[str, list[dict[str, float]]] = {
                "radio": [],
                "radio_component_shuffle": [],
                "radio_feature_shuffle": [],
                "radio_risk_shuffle": [],
                "same_count_random_components": [],
            }
            for frame_idx in range(t):
                sim = lf[frame_idx] @ lf[frame_idx].T
                before = _softmax(sim / 0.10, axis=-1)
                same = comp_flat[frame_idx][:, None] == comp_flat[frame_idx][None, :]
                cross = ~same
                pair_risk = np.maximum(risk_flat[frame_idx][:, None], risk_flat[frame_idx][None, :])
                rng = np.random.default_rng(7100 + int(sidecar["chunk_id"]) * 10000 + int(layer) * 100 + frame_idx)
                perm_comp = rng.permutation(comp_flat[frame_idx])
                perm_feat_idx = rng.permutation(rf.shape[1])
                perm_risk = risk_flat[frame_idx][rng.permutation(risk_flat.shape[1])]
                variants = {
                    "radio": (comp_flat[frame_idx], risk_flat[frame_idx], rf[frame_idx]),
                    "radio_component_shuffle": (perm_comp, risk_flat[frame_idx], rf[frame_idx]),
                    "radio_feature_shuffle": (comp_flat[frame_idx], risk_flat[frame_idx], rf[frame_idx][perm_feat_idx]),
                    "radio_risk_shuffle": (comp_flat[frame_idx], perm_risk, rf[frame_idx]),
                    "same_count_random_components": (rng.permutation(comp_flat[frame_idx]), risk_flat[frame_idx], rf[frame_idx]),
                }
                for control_type, (corr_comp, corr_risk, corr_rf) in variants.items():
                    corr_same = corr_comp[:, None] == corr_comp[None, :]
                    corr_cross = ~corr_same
                    corr_pair_risk = np.maximum(corr_risk[:, None], corr_risk[None, :])
                    radio_sim = corr_rf @ corr_rf.T
                    radio_logits = (
                        radio_sim
                        + 0.35 * corr_same.astype(np.float32)
                        - 0.50 * corr_cross.astype(np.float32) * corr_pair_risk
                    )
                    proxy = _softmax(radio_logits / 0.12, axis=-1)
                    if mode == "object_interior_floor":
                        after = before + 0.05 * corr_same.astype(np.float32)
                        after = after / np.maximum(after.sum(axis=-1, keepdims=True), 1e-12)
                    elif mode == "cross_object_veto":
                        after = before * (1.0 - 0.30 * corr_cross.astype(np.float32) * corr_pair_risk)
                        after = after / np.maximum(after.sum(axis=-1, keepdims=True), 1e-12)
                    elif mode == "postsoftmax_mix":
                        after = 0.90 * before + 0.10 * proxy
                        after = after / np.maximum(after.sum(axis=-1, keepdims=True), 1e-12)
                    else:
                        continue
                    same_before = _mass(before, same)
                    same_after = _mass(after, same)
                    cross_before = _mass(before, cross)
                    cross_after = _mass(after, cross)
                    risky_before = _mass(before, pair_risk > 0.50)
                    risky_after = _mass(after, pair_risk > 0.50)
                    ent_before = _entropy(before)
                    ent_after = _entropy(after)
                    frame_metrics_by_control[control_type].append(
                        {
                            "same_object_attention_mass_before": same_before,
                            "same_object_attention_mass_after": same_after,
                            "cross_object_attention_mass_before": cross_before,
                            "cross_object_attention_mass_after": cross_after,
                            "risky_attention_mass_before": risky_before,
                            "risky_attention_mass_after": risky_after,
                            "attention_entropy_before": ent_before,
                            "attention_entropy_after": ent_after,
                            "attention_kl_after_before": _kl(before, after),
                            "empty_attention_rows": float(np.sum(after.sum(axis=-1) <= 1e-8)),
                        }
                    )
            def avg(frame_metrics: list[dict[str, float]], key: str) -> float:
                vals = [row[key] for row in frame_metrics if math.isfinite(float(row[key]))]
                return float(sum(vals) / len(vals)) if vals else float("nan")

            for control_type, frame_metrics in frame_metrics_by_control.items():
                if not frame_metrics:
                    continue
                same_before = avg(frame_metrics, "same_object_attention_mass_before")
                same_after = avg(frame_metrics, "same_object_attention_mass_after")
                cross_before = avg(frame_metrics, "cross_object_attention_mass_before")
                cross_after = avg(frame_metrics, "cross_object_attention_mass_after")
                risky_before = avg(frame_metrics, "risky_attention_mass_before")
                risky_after = avg(frame_metrics, "risky_attention_mass_after")
                ent_before = avg(frame_metrics, "attention_entropy_before")
                ent_after = avg(frame_metrics, "attention_entropy_after")
                fidelity_pass = bool(
                    same_after >= same_before * 1.10
                    and cross_after < cross_before
                    and risky_after <= risky_before
                    and avg(frame_metrics, "empty_attention_rows") == 0.0
                    and ent_after >= 0.50 * ent_before
                )
                rows.append(
                    {
                        "chunk_id": int(sidecar["chunk_id"]),
                        "layer": layer,
                        "mode": mode,
                        "control_type": control_type,
                        "tap_type": tap_entry.get("tap_type"),
                        "frames": int(t),
                        "patch_grid": [int(h), int(w)],
                        "attention_KL_before_after": avg(frame_metrics, "attention_kl_after_before"),
                        "attention_entropy_before": ent_before,
                        "attention_entropy_after": ent_after,
                        "same_object_attention_mass_before": same_before,
                        "same_object_attention_mass_after": same_after,
                        "same_object_attention_mass_ratio": same_after / max(same_before, 1e-12),
                        "cross_object_attention_mass_before": cross_before,
                        "cross_object_attention_mass_after": cross_after,
                        "dynamic_sky_attention_mass_before": risky_before,
                        "dynamic_sky_attention_mass_after": risky_after,
                        "empty_attention_rows": avg(frame_metrics, "empty_attention_rows"),
                        "attention_fidelity_pass": fidelity_pass,
                        "mechanism_metrics_available": False,
                        "mechanism_note": "Feature-proxy attention only; no J_v70, future/head-tail/scale, or ATE metric computed.",
                    }
                )
    return rows


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    chunks = parse_chunks(args.target_chunks)
    modes = [x.strip() for x in str(args.modes).split(",") if x.strip()]
    sidecars = _sidecar_paths(args.radio_sidecar_dir, chunks)
    tap_candidates = _tap_candidates(args.path_taps_dir)
    inspected = [_known_attention_payload(Path(row["path"])) for row in tap_candidates[:20]]
    manifest_path = args.path_taps_dir / "path_taps_manifest.json"
    proxy_rows: list[dict[str, Any]] = []
    proxy_manifest: dict[str, Any] | None = None
    if manifest_path.exists():
        proxy_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = [row for row in proxy_manifest.get("entries", []) if row.get("available") and int(row.get("chunk_id", -1)) in chunks]
        for entry in entries:
            chunk_sidecars = [p for p in sidecars if f"chunk_{int(entry['chunk_id']):03d}_" in str(p)]
            for sidecar_path in chunk_sidecars:
                proxy_rows.extend(_proxy_attention_metrics_for_chunk(sidecar_path, entry, modes))

    blocker = ""
    gate_pass = False
    radio_rows = [row for row in proxy_rows if row.get("control_type", "radio") == "radio"]
    control_rows = [row for row in proxy_rows if row.get("control_type", "radio") != "radio"]
    if not sidecars:
        blocker = "blocked_missing_radio_sidecar"
    elif not args.path_taps_dir.exists():
        blocker = "blocked_missing_v70_path_taps"
    elif not tap_candidates:
        blocker = "blocked_empty_v70_path_taps"
    elif proxy_rows:
        gate_pass = any(bool(row.get("attention_fidelity_pass")) for row in radio_rows)
        blocker = "" if gate_pass else "blocked_proxy_attention_fidelity_failed"
    elif not any(row.get("known") for row in inspected):
        blocker = "blocked_unknown_or_missing_attention_tap_schema"
    else:
        blocker = "blocked_attention_oracle_not_implemented_for_detected_schema"

    summary = {
        "created_at": utc_now(),
        "phase": "R3_R5_radio_attention_oracle_smoke",
        "radio_sidecar_dir": str(args.radio_sidecar_dir),
        "path_taps_dir": str(args.path_taps_dir),
        "target_chunks": chunks,
        "modes": modes,
        "sidecar_paths": [str(p) for p in sidecars],
        "tap_candidates_count": len(tap_candidates),
        "tap_candidates_sample": tap_candidates[:50],
        "inspected_torch_payloads": inspected,
        "proxy_manifest": proxy_manifest,
        "proxy_attention_rows": proxy_rows,
        "radio_proxy_rows_count": len(radio_rows),
        "control_proxy_rows_count": len(control_rows),
        "radio_proxy_rows_pass": sum(1 for row in radio_rows if bool(row.get("attention_fidelity_pass"))),
        "control_proxy_rows_pass": sum(1 for row in control_rows if bool(row.get("attention_fidelity_pass"))),
        "gate_pass": gate_pass,
        "blocker": blocker,
        "mechanism_metrics_available": False,
        "note": "No online HMC attention/action metric is fabricated. global_k_feature_proxy rows measure attention-fidelity proxy only.",
    }
    write_json(args.out_dir / "radio_attention_oracle_summary.json", summary)
    if proxy_rows:
        from v70_radio_sidecar_common import write_csv

        write_csv(args.out_dir / "radio_attention_oracle_proxy_rows.csv", proxy_rows)
    report = [
        "# v70 RADIO Attention Oracle Smoke",
        "",
        f"- gate_pass: `{gate_pass}`",
        f"- blocker: `{blocker}`",
        f"- sidecars: `{len(sidecars)}`",
        f"- tap_candidates_count: `{len(tap_candidates)}`",
        f"- proxy_rows: `{len(proxy_rows)}`",
        f"- radio_proxy_rows_pass: `{sum(1 for row in radio_rows if bool(row.get('attention_fidelity_pass')))}` / `{len(radio_rows)}`",
        f"- control_proxy_rows_pass: `{sum(1 for row in control_rows if bool(row.get('attention_fidelity_pass')))}` / `{len(control_rows)}`",
        "",
        "This tool does not compute J_v70, future/head-tail/scale, ATE, or online method metrics. When `global_k_feature_proxy` taps are present, it computes only proxy attention-fidelity metrics from real LoGeR feature dumps.",
    ]
    write_text(args.out_dir / "radio_attention_oracle_report.md", "\n".join(report) + "\n")
    print(json.dumps({"out_dir": str(args.out_dir), "gate_pass": gate_pass, "blocker": blocker}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
