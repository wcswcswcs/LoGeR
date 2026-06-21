#!/usr/bin/env python3
"""ACL2 v70 RADIO sidecar READ offline attention oracle.

This tool evaluates READ-style attention correction candidates on available
LoGeR global_k feature-proxy taps. It deliberately separates feature-proxy
attention fidelity from the real R5 READ gate, which would require a trajectory
or local-window READ intervention.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

try:
    from diagnose_v70_radio_merge_oracle import _finite_mean, _finite_median, _safe_tag, _write_csv
    from v70_radio_sidecar_common import (
        find_stage_chunk,
        load_stage_semantic,
        parse_chunks,
        pool_semantic_to_grid,
        resize_linear,
        resize_nearest,
        torch_load,
        utc_now,
    )
except ImportError:  # pragma: no cover
    from tools.diagnose_v70_radio_merge_oracle import _finite_mean, _finite_median, _safe_tag, _write_csv
    from tools.v70_radio_sidecar_common import (
        find_stage_chunk,
        load_stage_semantic,
        parse_chunks,
        pool_semantic_to_grid,
        resize_linear,
        resize_nearest,
        torch_load,
        utc_now,
    )


RADIO_CANDIDATES = {"READ_R3_object_interior_floor", "READ_R4_cross_object_risk_veto"}
CONTROL_CANDIDATES = {
    "READ_R1_label_only",
    "READ_label_shuffle",
    "READ_confidence_shuffle",
    "READ_radio_component_shuffle",
    "READ_radio_feature_shuffle",
    "READ_radio_risk_shuffle",
    "READ_same_entropy_random_proxy",
}
BASELINE_CANDIDATES = {"READ_R0_native"}


def _float(value: Any, default: float = float("nan")) -> float:
    if value is None or value == "":
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _family(candidate_type: str) -> str:
    if candidate_type in RADIO_CANDIDATES:
        return "radio"
    if candidate_type in CONTROL_CANDIDATES:
        return "control"
    if candidate_type in BASELINE_CANDIDATES:
        return "baseline"
    return "unknown"


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
    return float(np.mean(np.sum(attn * mask, axis=-1)))


def _normalise_feat(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-6)


def _sidecar_path(root: Path, chunk_id: int) -> Path:
    hits = sorted(root.glob(f"chunk_{int(chunk_id):03d}_*/radio_sidecar.pt"))
    if not hits:
        raise FileNotFoundError(f"missing sidecar for chunk {chunk_id}: {root}")
    return hits[0]


def _load_manifest(path: Path) -> Dict[int, Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        int(row["chunk_id"]): row
        for row in payload.get("entries", [])
        if row.get("available") and row.get("feature_path")
    }


def _semantic_for_chunk(stage_c_cache: Path, chunk_id: int, grid_hw: Tuple[int, int], frames: int) -> Dict[str, np.ndarray]:
    chunk = find_stage_chunk(stage_c_cache, chunk_id)
    semantic = load_stage_semantic(chunk.masklet_path)
    labels = semantic["label_maps"][:frames]
    conf = semantic.get("confidence_maps")
    conf = conf[:frames] if conf is not None else None
    return pool_semantic_to_grid(labels, conf, [str(x) for x in semantic.get("label_names", [])], grid_hw)


def _random_mask_like(score: np.ndarray, base: np.ndarray, keep_mass: float, rng: np.random.Generator) -> np.ndarray:
    out = np.zeros_like(score, dtype=np.float32)
    flat = np.where(base.reshape(-1))[0]
    if flat.size == 0:
        return out
    keep = max(1, min(flat.size, int(round(float(keep_mass) * flat.size))))
    chosen = rng.choice(flat, size=keep, replace=False)
    out.reshape(-1)[chosen] = 1.0
    return out


def _attention_after(
    before: np.ndarray,
    candidate_type: str,
    *,
    same: np.ndarray,
    cross: np.ndarray,
    risk: np.ndarray,
    label_same: np.ndarray,
    label_conf_pair: np.ndarray,
    radio_proxy: np.ndarray,
    lam: float,
) -> np.ndarray:
    if candidate_type == "READ_R0_native":
        after = before.copy()
    elif candidate_type in {"READ_R3_object_interior_floor", "READ_radio_component_shuffle"}:
        after = before + float(lam) * same.astype(np.float32) * (1.0 - risk)
    elif candidate_type in {"READ_R4_cross_object_risk_veto", "READ_radio_risk_shuffle"}:
        after = before * (1.0 - float(lam) * cross.astype(np.float32) * risk)
    elif candidate_type in {"READ_R1_label_only", "READ_label_shuffle", "READ_confidence_shuffle"}:
        after = before + float(lam) * label_same.astype(np.float32) * label_conf_pair
    elif candidate_type in {"READ_radio_feature_shuffle", "READ_same_entropy_random_proxy"}:
        after = (1.0 - float(lam)) * before + float(lam) * radio_proxy
    else:
        raise ValueError(f"unknown candidate_type={candidate_type}")
    after = np.clip(after, 1e-12, None)
    return after / np.maximum(after.sum(axis=-1, keepdims=True), 1e-12)


def _metrics(
    before: np.ndarray,
    after: np.ndarray,
    *,
    same_eval: np.ndarray,
    cross_eval: np.ndarray,
    risk_eval: np.ndarray,
) -> Dict[str, float]:
    same_before = _mass(before, same_eval)
    same_after = _mass(after, same_eval)
    cross_before = _mass(before, cross_eval)
    cross_after = _mass(after, cross_eval)
    risky_before = _mass(before, risk_eval > 0.50)
    risky_after = _mass(after, risk_eval > 0.50)
    ent_before = _entropy(before)
    ent_after = _entropy(after)
    return {
        "attention_KL_before_after": _kl(before, after),
        "attention_entropy_before": ent_before,
        "attention_entropy_after": ent_after,
        "attention_entropy_ratio": ent_after / max(ent_before, 1e-12),
        "same_object_attention_mass_before": same_before,
        "same_object_attention_mass_after": same_after,
        "same_object_attention_mass_ratio": same_after / max(same_before, 1e-12),
        "cross_object_attention_mass_before": cross_before,
        "cross_object_attention_mass_after": cross_after,
        "cross_object_attention_mass_delta": cross_after - cross_before,
        "dynamic_sky_attention_mass_before": risky_before,
        "dynamic_sky_attention_mass_after": risky_after,
        "dynamic_sky_attention_mass_delta": risky_after - risky_before,
        "empty_attention_rows": float(np.sum(after.sum(axis=-1) <= 1e-8)),
    }


def _gate(row: Mapping[str, Any], min_same_ratio: float, min_entropy_ratio: float, max_kl: float) -> bool:
    candidate = str(row.get("candidate_type"))
    if candidate == "READ_R0_native":
        return False
    return bool(
        _float(row.get("same_object_attention_mass_ratio")) >= float(min_same_ratio)
        and _float(row.get("cross_object_attention_mass_delta")) < 0.0
        and _float(row.get("dynamic_sky_attention_mass_delta")) <= 0.0
        and _float(row.get("attention_entropy_ratio")) >= float(min_entropy_ratio)
        and _float(row.get("attention_KL_before_after")) <= float(max_kl)
        and int(_float(row.get("empty_attention_rows"), 1.0)) == 0
    )


def _best_by_chunk(rows: Sequence[Dict[str, Any]], family: str) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        if row.get("candidate_family") != family:
            continue
        chunk = int(row["chunk_id"])
        score = _float(row.get("read_proxy_score"), -1e9)
        cur = out.get(chunk)
        if cur is None or score > _float(cur.get("read_proxy_score"), -1e9):
            out[chunk] = row
    return out


def _candidate_counts(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    for row in rows:
        key = str(row.get("candidate_type"))
        out.setdefault(key, {"rows": 0, "proxy_gate_pass": 0})
        out[key]["rows"] += 1
        out[key]["proxy_gate_pass"] += int(bool(row.get("read_attention_proxy_gate_pass")))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radio-sidecar-dir", type=Path, required=True)
    parser.add_argument("--path-taps-dir", type=Path, required=True)
    parser.add_argument("--stage-c-cache", type=Path, required=True)
    parser.add_argument("--target-chunks", required=True)
    parser.add_argument("--lambda-read", type=float, action="append", default=None)
    parser.add_argument("--candidate-type", action="append", default=None)
    parser.add_argument("--layer", type=int, action="append", default=None)
    parser.add_argument("--max-frames-per-chunk", type=int, default=0)
    parser.add_argument("--max-tokens-per-frame", type=int, default=0)
    parser.add_argument("--min-same-ratio", type=float, default=1.10)
    parser.add_argument("--min-entropy-ratio", type=float, default=0.50)
    parser.add_argument("--max-kl", type=float, default=3.00)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    chunks = parse_chunks(args.target_chunks)
    manifest = _load_manifest(args.path_taps_dir / "path_taps_manifest.json")
    lambdas = args.lambda_read if args.lambda_read is not None else [0.05, 0.10, 0.20]
    candidate_filter = set(args.candidate_type or [])
    layer_filter = set(int(x) for x in (args.layer or []))
    rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    for chunk_id in chunks:
        tap = manifest.get(int(chunk_id))
        if not tap:
            failures.append({"chunk_id": int(chunk_id), "failure": "missing_path_tap"})
            continue
        try:
            sidecar = torch_load(_sidecar_path(args.radio_sidecar_dir, chunk_id))
            payload = torch_load(Path(tap["feature_path"]))
            tensor = payload[tap["tensor_key"]].detach().cpu().float().numpy()
        except Exception as exc:  # noqa: BLE001
            failures.append({"chunk_id": int(chunk_id), "failure": f"input_error:{type(exc).__name__}:{exc}"})
            continue
        pca = _tensor_np(sidecar["radio_feat_pca"])
        comp = _tensor_np(sidecar["object_component_id"], integer=True)
        interior = _tensor_np(sidecar["object_interior_score"])
        risk = np.maximum.reduce([
            _tensor_np(sidecar["object_boundary_score"]),
            _tensor_np(sidecar["radio_dynamic_score"]),
            _tensor_np(sidecar["radio_sky_context_score"]),
            _tensor_np(sidecar["radio_lowtrust_score"]),
        ]).astype(np.float32)

        for layer_row in tap.get("layer_positions", []):
            layer = int(layer_row["layer"])
            if layer_filter and layer not in layer_filter:
                continue
            pos = int(layer_row["position"])
            loger = tensor[:, pos]
            t, h, w, d = loger.shape
            semantic = _semantic_for_chunk(args.stage_c_cache, chunk_id, (h, w), t)
            labels = semantic["label"].astype(np.int64)
            label_conf = semantic["confidence"].astype(np.float32)
            radio_feat = np.zeros((t, h, w, pca.shape[-1]), dtype=np.float32)
            comp_small = np.zeros((t, h, w), dtype=np.int64)
            risk_small = np.zeros((t, h, w), dtype=np.float32)
            interior_small = np.zeros((t, h, w), dtype=np.float32)
            if int(args.max_frames_per_chunk) > 0 and t > int(args.max_frames_per_chunk):
                frame_indices = sorted(set(np.linspace(0, t - 1, int(args.max_frames_per_chunk)).round().astype(int).tolist()))
            else:
                frame_indices = list(range(t))
            for frame_idx in frame_indices:
                radio_feat[frame_idx] = resize_linear(pca[frame_idx], (h, w))
                comp_small[frame_idx] = resize_nearest(comp[frame_idx].astype(np.int32), (h, w)).astype(np.int64)
                risk_small[frame_idx] = resize_linear(risk[frame_idx], (h, w))
                interior_small[frame_idx] = resize_linear(interior[frame_idx], (h, w))
            lf = _normalise_feat(loger.reshape(t, h * w, d))
            rf = _normalise_feat(radio_feat.reshape(t, h * w, radio_feat.shape[-1]))
            comp_flat = comp_small.reshape(t, h * w)
            labels_flat = labels.reshape(t, h * w)
            conf_flat = label_conf.reshape(t, h * w)
            risk_flat = np.clip(risk_small.reshape(t, h * w), 0.0, 1.0)
            interior_flat = np.clip(interior_small.reshape(t, h * w), 0.0, 1.0)
            max_tokens = int(args.max_tokens_per_frame)
            token_subsampled = False
            if max_tokens > 0 and lf.shape[1] > max_tokens:
                token_indices = sorted(set(np.linspace(0, lf.shape[1] - 1, max_tokens).round().astype(int).tolist()))
                lf = lf[:, token_indices]
                rf = rf[:, token_indices]
                comp_flat = comp_flat[:, token_indices]
                labels_flat = labels_flat[:, token_indices]
                conf_flat = conf_flat[:, token_indices]
                risk_flat = risk_flat[:, token_indices]
                interior_flat = interior_flat[:, token_indices]
                token_subsampled = True

            for frame_idx in frame_indices:
                sim = lf[frame_idx] @ lf[frame_idx].T
                before = _softmax(sim / 0.10, axis=-1)
                same = (comp_flat[frame_idx][:, None] == comp_flat[frame_idx][None, :]).astype(np.float32)
                cross = 1.0 - same
                pair_risk = np.maximum(risk_flat[frame_idx][:, None], risk_flat[frame_idx][None, :])
                pair_interior = np.minimum(interior_flat[frame_idx][:, None], interior_flat[frame_idx][None, :])
                label_same = (labels_flat[frame_idx][:, None] == labels_flat[frame_idx][None, :]).astype(np.float32)
                label_conf_pair = np.minimum(conf_flat[frame_idx][:, None], conf_flat[frame_idx][None, :])
                radio_sim = rf[frame_idx] @ rf[frame_idx].T
                radio_logits = radio_sim + 0.35 * same + 0.25 * pair_interior - 0.50 * cross * pair_risk
                radio_proxy = _softmax(radio_logits / 0.12, axis=-1)
                rng = np.random.default_rng(7200 + int(chunk_id) * 10000 + layer * 100 + frame_idx)
                comp_perm = rng.permutation(comp_flat[frame_idx])
                same_comp_shuffle = (comp_perm[:, None] == comp_perm[None, :]).astype(np.float32)
                feat_perm = rng.permutation(rf.shape[1])
                rf_shuf = rf[frame_idx][feat_perm]
                radio_proxy_shuf = _softmax((rf_shuf @ rf_shuf.T) / 0.12, axis=-1)
                risk_perm = risk_flat[frame_idx][rng.permutation(risk_flat.shape[1])]
                pair_risk_shuffle = np.maximum(risk_perm[:, None], risk_perm[None, :])
                label_perm = rng.permutation(labels_flat[frame_idx])
                label_shuffle_same = (labels_flat[frame_idx][:, None] == label_perm[None, :]).astype(np.float32)
                conf_perm = conf_flat[frame_idx][rng.permutation(conf_flat.shape[1])]
                label_conf_shuffle = np.minimum(conf_flat[frame_idx][:, None], conf_perm[None, :])
                random_proxy = _random_mask_like(radio_proxy, np.ones_like(radio_proxy, dtype=bool), 0.20, rng)
                random_proxy = random_proxy / np.maximum(random_proxy.sum(axis=-1, keepdims=True), 1e-12)

                variants = {
                    "READ_R0_native": (same, cross, pair_risk, label_same, label_conf_pair, radio_proxy),
                    "READ_R1_label_only": (same, cross, pair_risk, label_same, label_conf_pair, radio_proxy),
                    "READ_label_shuffle": (same, cross, pair_risk, label_shuffle_same, label_conf_pair, radio_proxy),
                    "READ_confidence_shuffle": (same, cross, pair_risk, label_same, label_conf_shuffle, radio_proxy),
                    "READ_R3_object_interior_floor": (same, cross, pair_risk, label_same, label_conf_pair, radio_proxy),
                    "READ_R4_cross_object_risk_veto": (same, cross, pair_risk, label_same, label_conf_pair, radio_proxy),
                    "READ_radio_component_shuffle": (same_comp_shuffle, 1.0 - same_comp_shuffle, pair_risk, label_same, label_conf_pair, radio_proxy),
                    "READ_radio_feature_shuffle": (same, cross, pair_risk, label_same, label_conf_pair, radio_proxy_shuf),
                    "READ_radio_risk_shuffle": (same, cross, pair_risk_shuffle, label_same, label_conf_pair, radio_proxy),
                    "READ_same_entropy_random_proxy": (same, cross, pair_risk, label_same, label_conf_pair, random_proxy),
                }
                if candidate_filter:
                    variants = {k: v for k, v in variants.items() if k in candidate_filter}
                for lam in lambdas:
                    for candidate_type, (cand_same, cand_cross, cand_risk, cand_label_same, cand_label_conf, cand_proxy) in variants.items():
                        after = _attention_after(
                            before,
                            candidate_type,
                            same=cand_same,
                            cross=cand_cross,
                            risk=cand_risk,
                            label_same=cand_label_same,
                            label_conf_pair=cand_label_conf,
                            radio_proxy=cand_proxy,
                            lam=float(lam),
                        )
                        row = {
                            "chunk_id": int(chunk_id),
                            "frame_index": int(frame_idx),
                            "layer": int(layer),
                            "candidate_type": candidate_type,
                            "candidate_family": _family(candidate_type),
                            "lambda_read": float(lam),
                            "tap_type": tap.get("tap_type"),
                            "patch_grid": [int(h), int(w)],
                            "eval_tokens": int(lf.shape[1]),
                            "token_subsampled": bool(token_subsampled),
                            "local_window_improvement_m": None,
                            "overlap_to_future_improvement": None,
                            "mechanism_metric_available": False,
                            "mechanism_metric_source": "feature_proxy_no_online_read",
                            **_metrics(before, after, same_eval=same, cross_eval=cross, risk_eval=pair_risk),
                        }
                        row["read_proxy_score"] = (
                            max(0.0, _float(row.get("same_object_attention_mass_ratio")) - 1.0)
                            + max(0.0, -_float(row.get("cross_object_attention_mass_delta")))
                            + max(0.0, -_float(row.get("dynamic_sky_attention_mass_delta")))
                            - max(0.0, 0.50 - _float(row.get("attention_entropy_ratio")))
                        )
                        row["read_attention_proxy_gate_pass"] = _gate(
                            row,
                            float(args.min_same_ratio),
                            float(args.min_entropy_ratio),
                            float(args.max_kl),
                        )
                        row["r5_read_oracle_gate_pass"] = False
                        rows.append(row)

    best_radio = _best_by_chunk(rows, "radio")
    best_control = _best_by_chunk(rows, "control")
    best_baseline = _best_by_chunk(rows, "baseline")
    radio_beats_controls_chunks = []
    for chunk, row in best_radio.items():
        radio_score = _float(row.get("read_proxy_score"), -1e9)
        control_score = _float(best_control.get(chunk, {}).get("read_proxy_score"), -1e9)
        baseline_score = _float(best_baseline.get(chunk, {}).get("read_proxy_score"), -1e9)
        if bool(row.get("read_attention_proxy_gate_pass")) and radio_score > max(control_score, baseline_score):
            radio_beats_controls_chunks.append(chunk)
    best_radio_proxy = [
        row for row in best_radio.values()
        if bool(row.get("read_attention_proxy_gate_pass"))
    ]
    best_radio_scores = [row.get("read_proxy_score") for row in best_radio_proxy]
    proxy_gate = bool(len(radio_beats_controls_chunks) >= 4 and (_finite_median(best_radio_scores) or float("-inf")) > 0.0)

    rows.sort(key=lambda row: (
        not bool(row.get("read_attention_proxy_gate_pass")),
        -_float(row.get("read_proxy_score"), -1e9),
        str(row.get("candidate_type")),
    ))
    radio_rows = [row for row in rows if row.get("candidate_family") == "radio"]
    control_rows = [row for row in rows if row.get("candidate_family") == "control"]
    summary = {
        "schema": "acl2_v70_radio_read_oracle_summary_v1",
        "created_at": utc_now(),
        "radio_sidecar_dir": str(args.radio_sidecar_dir),
        "path_taps_dir": str(args.path_taps_dir),
        "stage_c_cache": str(args.stage_c_cache),
        "target_chunks": chunks,
        "rows": len(rows),
        "failures": failures,
        "candidate_counts": _candidate_counts(rows),
        "radio_proxy_gate_rows": sum(bool(row.get("read_attention_proxy_gate_pass")) for row in radio_rows),
        "control_proxy_gate_rows": sum(bool(row.get("read_attention_proxy_gate_pass")) for row in control_rows),
        "radio_proxy_gate_chunks": sorted({
            int(row.get("chunk_id")) for row in radio_rows if bool(row.get("read_attention_proxy_gate_pass"))
        }),
        "radio_beats_controls_chunks": sorted(radio_beats_controls_chunks),
        "median_best_radio_proxy_score": _finite_median(best_radio_scores),
        "mean_radio_proxy_score": _finite_mean(row.get("read_proxy_score") for row in radio_rows),
        "mean_control_proxy_score": _finite_mean(row.get("read_proxy_score") for row in control_rows),
        "read_attention_proxy_gate_pass": proxy_gate,
        "r5_read_oracle_gate_pass": False,
        "r6_online_allowed_by_this_oracle": False,
        "decision": "diagnostic_only_proxy_pass_no_online" if proxy_gate else "no_go_r6_continue_r5_repair",
        "gate_rule": {
            "min_same_ratio": float(args.min_same_ratio),
            "min_entropy_ratio": float(args.min_entropy_ratio),
            "max_kl": float(args.max_kl),
            "lambdas": [float(x) for x in lambdas],
            "layers": sorted(layer_filter) if layer_filter else "all_manifest_layers",
            "candidate_types": sorted(candidate_filter) if candidate_filter else "all",
            "max_frames_per_chunk": int(args.max_frames_per_chunk),
            "max_tokens_per_frame": int(args.max_tokens_per_frame),
            "proxy_pass_rule": ">=4 chunks where best RADIO READ proxy beats best control/baseline and median best RADIO proxy score > 0",
            "r5_gate_note": "False because no local-window or overlap-to-future trajectory READ intervention is available.",
        },
        "best_row": rows[0] if rows else {},
        "best_radio_row": next((row for row in rows if row.get("candidate_family") == "radio"), {}),
        "best_control_row": next((row for row in rows if row.get("candidate_family") == "control"), {}),
        "note": "Offline READ attention-proxy oracle; no online READ/local-window/ATE improvement is claimed.",
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / "radio_read_oracle_results.csv", rows)
    (args.out_dir / "radio_read_oracle_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    report = [
        "# v70 RADIO READ Oracle",
        "",
        f"- rows: `{summary['rows']}`",
        f"- read_attention_proxy_gate_pass: `{summary['read_attention_proxy_gate_pass']}`",
        f"- r5_read_oracle_gate_pass: `{summary['r5_read_oracle_gate_pass']}`",
        f"- radio_proxy_gate_rows: `{summary['radio_proxy_gate_rows']}`",
        f"- control_proxy_gate_rows: `{summary['control_proxy_gate_rows']}`",
        f"- radio_beats_controls_chunks: `{summary['radio_beats_controls_chunks']}`",
        "",
        "This is not an online READ result; local-window and overlap-to-future metrics are unavailable.",
    ]
    (args.out_dir / "radio_read_oracle_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=True))


if __name__ == "__main__":
    main()
