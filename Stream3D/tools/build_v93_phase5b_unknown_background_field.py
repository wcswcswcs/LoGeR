#!/usr/bin/env python3
"""Run a v93 Phase5b unknown/background field readout from Phase5 field shards."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import build_v92_phase5_source_container_field as v92field  # noqa: E402
from tools import build_v93_phase5_boundary_affinity_field as phase5  # noqa: E402
from tools import run_v90_dev_extent_score_cross_audit as phase7d  # noqa: E402
from tools import run_v91_phase4_radius_sweep as radius_sweep  # noqa: E402


PHASE_ID = "v93_phase5b_unknown_background_field"
RUN_ID = "v93_phase5b_unknown_background_field_gpu"
OUT = ROOT / "outputs/audit/v93_phase5b_unknown_background_field"
DEFAULT_FIELD_ROOT = ROOT / "outputs/audit/v93_phase5_boundary_affinity_field"
V93_PHASE0 = ROOT / "outputs/audit/v93_phase0_contract"
V93_PHASE1 = ROOT / "outputs/audit/v93_phase1_source_edge_registry"
V93_PHASE3 = ROOT / "outputs/audit/v93_phase3_region_edge_graph"


def _resolve(raw: str | Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == ROOT.name:
        return WORKSPACE_ROOT / path
    return ROOT / path


def _rel(path: Path | str) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        try:
            return str(path.relative_to(WORKSPACE_ROOT))
        except ValueError:
            return str(path)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key, "")) for key in fieldnames})


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return float(default)
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def _mean(values: list[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else 0.0


def _created_at() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _stable_seed(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16) % (2**32)


def _cuda_devices() -> list[str]:
    if torch is None or not torch.cuda.is_available():
        return []
    return [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]


def _device_for(index: int) -> Any:
    if torch is None or not torch.cuda.is_available():
        return torch.device("cpu") if torch is not None else None
    return torch.device(f"cuda:{index % max(1, torch.cuda.device_count())}")


def _variant_specs() -> list[dict[str, Any]]:
    return [
        {
            "variant_id": "U0_whole_source_replay",
            "family": "baseline",
            "mode": "whole",
            "base_variant": "F0_whole_source_baseline",
            "score_threshold": -999.0,
            "area_cap": 1.0,
            "description": "Replay all region nodes from the Phase5 shard to verify materialization/evaluator wiring.",
        },
        {
            "variant_id": "U1_source_preserve_conflict_unknown",
            "family": "real",
            "mode": "unknown_background",
            "base_variant": "F2_D4RT_RADIO_pairwise",
            "score_threshold": 0.28,
            "unknown_margin": 0.08,
            "area_cap": 0.96,
            "min_area_fraction": 0.70,
            "description": "Preserve source extent but route high-conflict or unsupported boundary regions to unknown/background.",
        },
        {
            "variant_id": "U2_boundary_cut_unknown",
            "family": "real",
            "mode": "unknown_background",
            "base_variant": "F2_D4RT_RADIO_pairwise",
            "score_threshold": 0.34,
            "unknown_margin": 0.12,
            "area_cap": 0.86,
            "min_area_fraction": 0.52,
            "description": "Stronger unknown/background label near source/nested/competing boundaries.",
        },
        {
            "variant_id": "U3_seed_expand_object_specific",
            "family": "real",
            "mode": "seed_expand",
            "base_variant": "F2_D4RT_RADIO_pairwise",
            "score_threshold": 0.38,
            "area_cap": 0.78,
            "min_area_fraction": 0.42,
            "expand_iters": 3,
            "description": "Start from object-support seeds and expand over high-affinity edges with an unknown/background outside option.",
        },
        {
            "variant_id": "U4_compact_graphcut_unknown",
            "family": "real",
            "mode": "unknown_background",
            "base_variant": "F4_RADIO_edge_barrier",
            "score_threshold": 0.46,
            "unknown_margin": 0.16,
            "area_cap": 0.68,
            "min_area_fraction": 0.25,
            "description": "Compact object label; high uncertainty and boundary conflict become unknown/background.",
        },
        {
            "variant_id": "C0_unknown_random_area_control",
            "family": "control",
            "mode": "random_area",
            "base_variant": "F2_D4RT_RADIO_pairwise",
            "reference_variant": "U1_source_preserve_conflict_unknown",
            "score_threshold": 0.0,
            "area_cap": 0.96,
            "description": "Random region control with U1-selected area target.",
        },
        {
            "variant_id": "C1_unknown_shuffled_score_control",
            "family": "control",
            "mode": "shuffled_score",
            "base_variant": "F2_D4RT_RADIO_pairwise",
            "reference_variant": "U1_source_preserve_conflict_unknown",
            "score_threshold": 0.28,
            "area_cap": 0.96,
            "description": "Shuffled object-score control with the same source-local score distribution.",
        },
    ]


def _phase8_gate_audit(metric_paths: list[Path]) -> dict[str, Any]:
    rows: list[dict[str, str]] = []
    for path in metric_paths:
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                row = dict(row)
                row["_artifact"] = _rel(path)
                rows.append(row)
    best_gap = -999.0
    best_row: dict[str, str] = {}
    for row in rows:
        sf50 = _num(row.get("mean_score_free_Match50_window"))
        ap50 = _num(row.get("mean_MV_AP50_window"))
        gap = sf50 - ap50
        if gap > best_gap:
            best_gap = gap
            best_row = row
    return {
        "phase8_enter_gate_pass": bool(best_gap >= 0.10),
        "best_scorefree_minus_ap50_gap": best_gap if rows else "",
        "best_gap_variant_id": best_row.get("variant_id", ""),
        "best_gap_artifact": best_row.get("_artifact", ""),
        "interpretation": "Phase8 score tuning is not warranted unless score-free Match50 is at least AP50 + 0.10.",
    }


def _load_phase5_score_map(field_root: Path) -> dict[tuple[str, str, int, int, str], float]:
    generated_path = field_root / "generated_mask_rows.csv"
    mv_path = field_root / "mv_object_frame_mask_rows.csv"
    if not generated_path.exists() or not mv_path.exists():
        return {}
    mv_score: dict[tuple[str, str, int, int, str], float] = {}
    with mv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            variant_id = str(row.get("variant", ""))
            object_id = str(row.get("mv_object_id", ""))
            prefix = f"{variant_id}:"
            if object_id.startswith(prefix):
                object_id = object_id[len(prefix) :]
            key = (
                variant_id,
                str(row.get("scene_id", "")),
                _int(row.get("frame_id"), -1),
                _int(row.get("mask_id"), -1),
                object_id,
            )
            mv_score[key] = _num(row.get("object_score"), _num(row.get("frame_mask_score"), 0.0))

    score_map: dict[tuple[str, str, int, int, str], float] = {}
    with generated_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            variant_id = str(row.get("variant_id", ""))
            scene = str(row.get("scene_id", ""))
            frame_id = _int(row.get("frame_id"), -1)
            source_mask_id = _int(row.get("source_mask_id"), -1)
            new_mask_id = _int(row.get("new_mask_id"), -1)
            object_id = str(row.get("object_hypothesis_id", ""))
            score = mv_score.get((variant_id, scene, frame_id, new_mask_id, object_id))
            if score is not None:
                score_map[(variant_id, scene, frame_id, source_mask_id, object_id)] = score
    return score_map


def _collect_source_keys(shard_paths: list[Path]) -> set[tuple[str, int, int]]:
    keys: set[tuple[str, int, int]] = set()
    for shard in shard_paths:
        with np.load(shard, allow_pickle=False) as data:
            for raw in data["source_keys"]:
                scene, frame, mask = str(raw).split("|")
                keys.add((scene, int(frame), int(mask)))
    return keys


def _load_region_nodes(path: Path, keys: set[tuple[str, int, int]]) -> dict[tuple[str, int, int], dict[int, dict[str, Any]]]:
    out: dict[tuple[str, int, int], dict[int, dict[str, Any]]] = defaultdict(dict)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (str(row.get("scene_id", "")), _int(row.get("frame_id"), -1), _int(row.get("source_mask_id"), -1))
            if key not in keys:
                continue
            idx = _int(row.get("region_index"), -1)
            if idx >= 0:
                out[key][idx] = row
    return out


def _node_mask(nodes: list[dict[str, Any]], selected: set[int], source_mask: np.ndarray) -> np.ndarray:
    return v92field._node_mask(nodes, selected, source_mask)


def _edge_expand(selected: Any, edge_u: Any, edge_v: Any, edge_affinity: Any, score: Any, threshold: float, iters: int) -> Any:
    if edge_u.numel() == 0 or selected.numel() == 0:
        return selected
    selected = selected.clone()
    for _ in range(max(0, int(iters))):
        src_selected = selected[edge_u] | selected[edge_v]
        strong = edge_affinity >= 0.62
        good_u = score[edge_u] >= threshold
        good_v = score[edge_v] >= threshold
        add_v = src_selected & strong & good_v
        add_u = src_selected & strong & good_u
        new_selected = selected.clone()
        if add_v.any():
            new_selected[edge_v[add_v]] = True
        if add_u.any():
            new_selected[edge_u[add_u]] = True
        if bool(torch.equal(new_selected, selected)):
            break
        selected = new_selected
    return selected


def _cap_by_area(selected: set[int], nodes: list[dict[str, Any]], cap: float, scores: np.ndarray, must_keep: set[int]) -> set[int]:
    return phase5._cap_by_area(selected, nodes, float(cap), [float(v) for v in scores.tolist()], must_keep)


def _enforce_min_area(selected: set[int], nodes: list[dict[str, Any]], min_fraction: float, scores: np.ndarray) -> set[int]:
    if not selected and len(nodes) > 0:
        selected = {int(np.argmax(scores))}
    total = sum(max(1, _int(row.get("pixel_count", row.get("area_px")), 1)) for row in nodes)
    target = float(min_fraction) * max(1, total)
    area = sum(max(1, _int(nodes[i].get("pixel_count", nodes[i].get("area_px")), 1)) for i in selected)
    if area >= target:
        return selected
    for idx in np.argsort(scores)[::-1].tolist():
        selected.add(int(idx))
        area += max(1, _int(nodes[int(idx)].get("pixel_count", nodes[int(idx)].get("area_px")), 1))
        if area >= target:
            break
    return selected


def _select_regions(
    *,
    spec: dict[str, Any],
    key: tuple[str, int, int],
    nodes: list[dict[str, Any]],
    features: np.ndarray,
    prob: np.ndarray,
    edge_u: np.ndarray,
    edge_v: np.ndarray,
    edge_weight: np.ndarray,
    base_variant_idx: int,
    reference_selected: set[int],
    device: Any,
) -> tuple[set[int], np.ndarray, dict[str, float]]:
    n = len(nodes)
    if n == 0:
        return set(), np.zeros(0, dtype=np.float32), {}
    sem = torch.as_tensor(features[:, 0], dtype=torch.float32, device=device)
    d4rt = torch.as_tensor(features[:, 1], dtype=torch.float32, device=device)
    inside = torch.as_tensor(features[:, 2], dtype=torch.float32, device=device)
    source_bar = torch.as_tensor(features[:, 3], dtype=torch.float32, device=device)
    nested_bar = torch.as_tensor(features[:, 4], dtype=torch.float32, device=device)
    competing_bar = torch.as_tensor(features[:, 5], dtype=torch.float32, device=device)
    negative = torch.as_tensor(features[:, 6], dtype=torch.float32, device=device)
    p = torch.as_tensor(prob, dtype=torch.float32, device=device)
    boundary = torch.maximum(source_bar, torch.maximum(nested_bar, competing_bar))
    unknown = 0.42 * boundary + 0.28 * (1.0 - torch.clamp(d4rt, 0.0, 1.0)) + 0.18 * torch.abs(p - 0.5) + 0.12 * negative
    background = 0.48 * negative + 0.24 * competing_bar + 0.16 * source_bar + 0.12 * (1.0 - torch.clamp(sem, 0.0, 1.0))
    object_score = 0.52 * p + 0.18 * d4rt + 0.17 * sem + 0.13 * inside - 0.20 * negative - 0.16 * competing_bar
    mode = str(spec["mode"])
    if mode == "whole":
        selected_t = torch.ones(n, dtype=torch.bool, device=device)
    elif mode == "random_area":
        target = len(reference_selected) if reference_selected else max(1, int(0.75 * n))
        rng = np.random.default_rng(_stable_seed(f"{key}:{spec['variant_id']}"))
        chosen = rng.choice(np.arange(n), size=min(n, target), replace=False)
        selected = {int(i) for i in chosen.tolist()}
        score_np = object_score.detach().cpu().numpy()
        selected = _cap_by_area(selected, nodes, float(spec.get("area_cap", 1.0)), score_np, set())
        return selected, score_np, {"target_region_count": float(target)}
    elif mode == "shuffled_score":
        score_np = object_score.detach().cpu().numpy().copy()
        rng = np.random.default_rng(_stable_seed(f"{key}:{spec['variant_id']}"))
        rng.shuffle(score_np)
        selected = {int(i) for i, value in enumerate(score_np) if value >= float(spec.get("score_threshold", 0.0))}
        selected = _enforce_min_area(selected, nodes, float(spec.get("min_area_fraction", 0.0)), score_np)
        selected = _cap_by_area(selected, nodes, float(spec.get("area_cap", 1.0)), score_np, set())
        return selected, score_np, {"target_region_count": float(len(reference_selected))}
    elif mode == "seed_expand":
        seed_score = object_score + 0.16 * d4rt
        seed_threshold = max(float(spec.get("score_threshold", 0.0)), float(torch.quantile(seed_score, 0.78).detach().cpu()))
        selected_t = seed_score >= seed_threshold
        if edge_u.size:
            e_u = torch.as_tensor(edge_u, dtype=torch.long, device=device)
            e_v = torch.as_tensor(edge_v, dtype=torch.long, device=device)
            e_w = torch.as_tensor(edge_weight[:, base_variant_idx], dtype=torch.float32, device=device)
            selected_t = _edge_expand(selected_t, e_u, e_v, e_w, object_score, float(spec.get("score_threshold", 0.0)), int(spec.get("expand_iters", 0)))
        selected_t = selected_t & ((object_score - unknown) >= -0.08) & ((object_score - background) >= -0.02)
    else:
        margin = float(spec.get("unknown_margin", 0.0))
        selected_t = (object_score >= float(spec.get("score_threshold", 0.0))) & ((object_score - unknown) >= -margin) & (
            (object_score - background) >= -margin
        )
    score_np = object_score.detach().cpu().numpy()
    selected = {int(i) for i in torch.nonzero(selected_t, as_tuple=False).flatten().detach().cpu().tolist()}
    seed_like = {int(i) for i, value in enumerate((features[:, 1] + 0.5 * features[:, 0]).tolist()) if value >= 0.80}
    selected |= seed_like
    selected = _enforce_min_area(selected, nodes, float(spec.get("min_area_fraction", 0.0)), score_np)
    selected = _cap_by_area(selected, nodes, float(spec.get("area_cap", 1.0)), score_np, seed_like)
    diagnostics = {
        "object_score_mean": float(torch.mean(object_score).detach().cpu()),
        "unknown_score_mean": float(torch.mean(unknown).detach().cpu()),
        "background_score_mean": float(torch.mean(background).detach().cpu()),
        "seed_like_count": float(len(seed_like)),
    }
    return selected, score_np, diagnostics


def _phase5b_gate_rows(variant_metric_rows: list[dict[str, Any]], specs: list[dict[str, Any]], phase0: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    spec_by_id = {spec["variant_id"]: spec for spec in specs}
    controls = [row for row in variant_metric_rows if spec_by_id.get(str(row.get("variant_id", "")), {}).get("family") == "control"]
    current_best_control = max(controls, key=lambda row: _num(row.get("mean_MV_AP_window"), -999.0), default={})
    locked_control_ap = _num(phase0.get("best_control_MV_AP_window"))
    locked_control_ap50 = _num(phase0.get("best_control_MV_AP50_window"))
    observed_control_ap = max(locked_control_ap, _num(current_best_control.get("mean_MV_AP_window"), locked_control_ap))
    observed_control_ap50 = max(locked_control_ap50, _num(current_best_control.get("mean_MV_AP50_window"), locked_control_ap50))
    required_ap = max(_num(phase0.get("v91_best_MV_AP_window")) + 0.006, observed_control_ap + 0.008)
    required_ap50 = max(_num(phase0.get("v91_best_MV_AP50_window")) + 0.012, observed_control_ap50 + 0.012)
    gate_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    any_pass = False
    for row in variant_metric_rows:
        variant_id = str(row.get("variant_id", ""))
        spec = spec_by_id.get(variant_id, {})
        mv_ap = _num(row.get("mean_MV_AP_window"))
        mv_ap50 = _num(row.get("mean_MV_AP50_window"))
        is_real = spec.get("family") == "real"
        v91_gate = mv_ap >= _num(phase0.get("v91_best_MV_AP_window")) + 0.006 and mv_ap50 >= _num(phase0.get("v91_best_MV_AP50_window")) + 0.012
        control_gate = mv_ap >= observed_control_ap + 0.008 and mv_ap50 >= observed_control_ap50 + 0.012
        provenance_gate = not row.get("uses_gt_for_prediction") and not row.get("uses_future")
        pass_gate = bool(is_real and v91_gate and control_gate and provenance_gate)
        any_pass = any_pass or pass_gate
        gate_rows.append(
            {
                "schema_version": "stream4d_v93_phase5b_variant_gate_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "variant_id": variant_id,
                "family": spec.get("family", ""),
                "MV_AP_window": mv_ap,
                "MV_AP50_window": mv_ap50,
                "required_MV_AP_window": required_ap,
                "required_MV_AP50_window": required_ap50,
                "v91_progress_gate_pass": v91_gate,
                "control_gate_pass": control_gate,
                "provenance_gate_pass": provenance_gate,
                "phase5b_dev_gate_pass": pass_gate,
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
        if is_real and not pass_gate:
            failure_rows.append(
                {
                    "schema_version": "stream4d_v93_phase5b_failure_v1",
                    "phase_id": PHASE_ID,
                    "run_id": RUN_ID,
                    "variant_id": variant_id,
                    "failure_type": "UNKNOWN_BACKGROUND_FIELD_NO_DEV_GATE",
                    "repair_direction": "Do not keep sweeping unknown/background thresholds unless one real variant exceeds controls; next needs a different object-specific field model.",
                    "MV_AP_window": mv_ap,
                    "MV_AP50_window": mv_ap50,
                    "required_MV_AP_window": required_ap,
                    "required_MV_AP50_window": required_ap50,
                    "best_control_for_gate_MV_AP_window": observed_control_ap,
                    "best_control_for_gate_MV_AP50_window": observed_control_ap50,
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                    "created_at": _created_at(),
                }
            )
    return gate_rows, failure_rows, any_pass


def run(args: argparse.Namespace) -> dict[str, Any]:
    if torch is None:
        raise RuntimeError("torch is required for Phase5b GPU/torch field readout")
    started = time.time()
    out = _resolve(args.output_root)
    if out.exists() and args.clean:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    created_at = _created_at()
    devices = _cuda_devices()
    field_root = _resolve(args.field_root)
    shard_paths = sorted((field_root / "field_shards").glob("field_shard_*.npz"))
    if args.max_shards > 0:
        shard_paths = shard_paths[: int(args.max_shards)]
    if not shard_paths:
        raise RuntimeError(f"No field shards found under {field_root / 'field_shards'}")
    source_keys = _collect_source_keys(shard_paths)
    source_meta = v92field._load_source_meta(_resolve(args.source_container_rows))
    nodes_by_key = _load_region_nodes(_resolve(args.region_node_rows), source_keys)
    phase0 = _read_json(V93_PHASE0 / "summary.json")
    specs = _variant_specs()
    variant_ids = [spec["variant_id"] for spec in specs]
    phase5_score_map = _load_phase5_score_map(field_root)
    config_rows = [
        {
            "schema_version": "stream4d_v93_phase5b_variant_config_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "variant_id": spec["variant_id"],
            "family": spec["family"],
            "mode": spec["mode"],
            "base_variant": spec.get("base_variant", ""),
            "score_threshold": spec.get("score_threshold", ""),
            "unknown_margin": spec.get("unknown_margin", ""),
            "area_cap": spec.get("area_cap", ""),
            "min_area_fraction": spec.get("min_area_fraction", ""),
            "description": spec.get("description", ""),
            "uses_gt_for_prediction": False,
            "uses_future": False,
            "created_at": created_at,
        }
        for spec in specs
    ]

    writer = phase5.ScoreWTAFrameWriter(out, variant_ids)
    generated_rows: list[dict[str, Any]] = []
    mv_rows: list[dict[str, Any]] = []
    assignment_summary_rows: list[dict[str, Any]] = []
    source_failure_rows: list[dict[str, Any]] = []
    label_cache: dict[tuple[str, int], np.ndarray] = {}
    device_counts: Counter[str] = Counter()
    score_protocol_counts: Counter[str] = Counter()
    processed_source_count = 0

    for shard_i, shard_path in enumerate(shard_paths):
        with np.load(shard_path, allow_pickle=False) as data:
            shard_variant_ids = [str(v) for v in data["variant_ids"].tolist()]
            source_key_raw = [str(v) for v in data["source_keys"].tolist()]
            object_ids = [str(v) for v in data["source_object_ids"].tolist()]
            region_source_idx = data["region_source_idx"]
            region_index_all = data["region_index"]
            region_feature_all = data["region_feature"]
            unary_source_idx = data["unary_source_idx"]
            unary_variant_idx = data["unary_variant_idx"]
            unary_region_index = data["unary_region_index"]
            unary_prob = data["unary_prob"]
            edge_source_idx = data["edge_source_idx"]
            edge_u_all = data["edge_u"]
            edge_v_all = data["edge_v"]
            edge_weight_all = data["edge_weight"]

            for local_source_idx, raw_key in enumerate(source_key_raw):
                scene, frame_raw, mask_raw = raw_key.split("|")
                key = (scene, int(frame_raw), int(mask_raw))
                meta = source_meta.get(key)
                node_map = nodes_by_key.get(key, {})
                if not meta or not node_map:
                    source_failure_rows.append(
                        {
                            "schema_version": "stream4d_v93_phase5b_source_failure_v1",
                            "phase_id": PHASE_ID,
                            "run_id": RUN_ID,
                            "source_key": raw_key,
                            "failure_type": "missing_source_meta_or_region_nodes",
                            "uses_gt_for_prediction": False,
                            "uses_future": False,
                        }
                    )
                    continue
                region_mask = region_source_idx == local_source_idx
                region_indices = region_index_all[region_mask].astype(np.int32)
                features = region_feature_all[region_mask].astype(np.float32)
                nodes = [node_map.get(int(region_idx)) for region_idx in region_indices.tolist()]
                if any(node is None for node in nodes):
                    source_failure_rows.append(
                        {
                            "schema_version": "stream4d_v93_phase5b_source_failure_v1",
                            "phase_id": PHASE_ID,
                            "run_id": RUN_ID,
                            "source_key": raw_key,
                            "failure_type": "missing_region_node_for_shard_index",
                            "missing_count": sum(1 for node in nodes if node is None),
                            "uses_gt_for_prediction": False,
                            "uses_future": False,
                        }
                    )
                    continue
                nodes_typed: list[dict[str, Any]] = [node for node in nodes if node is not None]
                n = len(nodes_typed)
                probs_by_variant: dict[str, np.ndarray] = {}
                for variant_idx, variant_id in enumerate(shard_variant_ids):
                    unary_mask = (unary_source_idx == local_source_idx) & (unary_variant_idx == variant_idx)
                    if not np.any(unary_mask):
                        continue
                    order = np.argsort(unary_region_index[unary_mask])
                    vals = unary_prob[unary_mask][order].astype(np.float32)
                    if vals.shape[0] == n:
                        probs_by_variant[variant_id] = vals
                if not probs_by_variant:
                    continue
                edge_mask = edge_source_idx == local_source_idx
                edge_u = edge_u_all[edge_mask].astype(np.int64)
                edge_v = edge_v_all[edge_mask].astype(np.int64)
                edge_weight = edge_weight_all[edge_mask].astype(np.float32)
                mask_path = _resolve(meta.get("mask_path", ""))
                frame_key = (scene, int(frame_raw))
                if frame_key not in label_cache:
                    label_cache[frame_key] = v92field._read_label(mask_path)
                label = label_cache[frame_key]
                source_mask = label == int(mask_raw)
                if not np.any(source_mask):
                    continue
                writer.ensure_frame(scene, int(frame_raw), label.shape)
                device = _device_for(processed_source_count)
                device_name = "cpu"
                if torch.cuda.is_available() and getattr(device, "type", "") == "cuda":
                    torch.cuda.set_device(device)
                    device_name = f"cuda:{device.index}:{torch.cuda.get_device_name(device.index)}"
                device_counts[device_name] += 1
                selections: dict[str, set[int]] = {}
                score_cache: dict[str, np.ndarray] = {}
                diagnostics_by_variant: dict[str, dict[str, float]] = {}
                for spec in specs:
                    base_variant = str(spec.get("base_variant", "F2_D4RT_RADIO_pairwise"))
                    prob = probs_by_variant.get(base_variant)
                    if prob is None:
                        prob = next(iter(probs_by_variant.values()))
                    base_variant_idx = shard_variant_ids.index(base_variant) if base_variant in shard_variant_ids else 0
                    selected, score, diag = _select_regions(
                        spec=spec,
                        key=key,
                        nodes=nodes_typed,
                        features=features,
                        prob=prob,
                        edge_u=edge_u,
                        edge_v=edge_v,
                        edge_weight=edge_weight,
                        base_variant_idx=base_variant_idx,
                        reference_selected=selections.get(str(spec.get("reference_variant", "")), set()),
                        device=device,
                    )
                    selections[spec["variant_id"]] = selected
                    score_cache[spec["variant_id"]] = score
                    diagnostics_by_variant[spec["variant_id"]] = diag

                source_area = int(np.count_nonzero(source_mask))
                object_id = object_ids[local_source_idx]
                for spec in specs:
                    variant_id = str(spec["variant_id"])
                    selected = selections[variant_id]
                    score = score_cache[variant_id]
                    mask = _node_mask(nodes_typed, selected, source_mask)
                    if not np.any(mask):
                        continue
                    area = int(np.count_nonzero(mask))
                    area_ratio = float(area / max(1, source_area))
                    selected_scores = [float(score[i]) for i in selected] if selected else [0.0]
                    selected_negative = [float(features[i, 6]) for i in selected] if selected else [0.0]
                    selected_d4rt = [float(features[i, 1]) for i in selected] if selected else [0.0]
                    score_variant = str(spec.get("base_variant", "F2_D4RT_RADIO_pairwise"))
                    mapped_score = phase5_score_map.get((score_variant, scene, int(frame_raw), int(mask_raw), object_id))
                    if mapped_score is None:
                        object_score = float(0.60 * _mean(selected_scores) + 0.25 * _mean(selected_d4rt) + 0.15 * (1.0 - _mean(selected_negative)))
                        score_protocol = "fallback_phase5b_feature_score"
                    else:
                        object_score = float(mapped_score)
                        score_protocol = f"reused_phase5_{score_variant}_object_score"
                    score_protocol_counts[score_protocol] += 1
                    gen_path = out / "generated_masks" / variant_id / scene / "mask" / f"{int(frame_raw)}.png"
                    generated_row = {
                        "schema_version": "stream4d_v93_phase5b_generated_mask_v1",
                        "phase_id": PHASE_ID,
                        "run_id": RUN_ID,
                        "variant_id": variant_id,
                        "scene_id": scene,
                        "split": "dev",
                        "window_id": nodes_typed[0].get("window_id", ""),
                        "frame_id": int(frame_raw),
                        "source_mask_id": int(mask_raw),
                        "new_mask_id": "",
                        "object_hypothesis_id": object_id,
                        "generated_mask_path": _rel(gen_path),
                        "source_mask_area": source_area,
                        "generated_mask_area_before_frame_wta": area,
                        "generated_mask_area": area,
                        "generated_mask_area_ratio": area_ratio,
                        "selected_region_count": len(selected),
                        "total_region_count": n,
                        "mean_selected_score": _mean(selected_scores),
                        "mean_selected_d4rt": _mean(selected_d4rt),
                        "mean_selected_hard_negative": _mean(selected_negative),
                        "score_protocol": score_protocol,
                        "solver_backend": "torch_cuda" if device_name.startswith("cuda:") else "torch_cpu",
                        "gpu_device": device_name,
                        "uses_gt_for_prediction": False,
                        "uses_future": False,
                    }
                    mv_row = {
                        "split": "dev",
                        "scene_id": scene,
                        "source_variant": variant_id,
                        "variant": variant_id,
                        "mv_object_id": f"{variant_id}:{object_id}",
                        "frame_id": int(frame_raw),
                        "mask_id": "",
                        "frame_mask_score": object_score,
                        "object_score": object_score,
                        "uses_gt_for_prediction": False,
                        "uses_future": False,
                        "uses_rgbd_pose_mesh": False,
                        "materializable": True,
                        "selection_reason": f"v93_phase5b_{spec['mode']}_{score_protocol}",
                    }
                    writer.add(variant_id, mask, object_score, generated_row, mv_row)
                    assignment_summary_rows.append(
                        {
                            "schema_version": "stream4d_v93_phase5b_assignment_summary_v1",
                            "phase_id": PHASE_ID,
                            "run_id": RUN_ID,
                            "variant_id": variant_id,
                            "scene_id": scene,
                            "frame_id": int(frame_raw),
                            "source_mask_id": int(mask_raw),
                            "selected_region_count": len(selected),
                            "total_region_count": n,
                            "selected_region_fraction": float(len(selected) / max(1, n)),
                            "generated_mask_area_ratio": area_ratio,
                            **diagnostics_by_variant.get(variant_id, {}),
                            "uses_gt_for_prediction": False,
                            "uses_future": False,
                        }
                    )
                processed_source_count += 1
        if (shard_i + 1) % max(1, int(args.progress_every_shards)) == 0:
            print(
                json.dumps(
                    {
                        "phase": PHASE_ID,
                        "processed_shards": shard_i + 1,
                        "processed_source_count": processed_source_count,
                        "elapsed_sec": time.time() - started,
                        "device_counts": dict(device_counts),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    writer.flush()
    generated_rows = writer.generated_rows
    mv_rows = writer.mv_rows
    object_rows = phase5._object_rows_from_mv(mv_rows)
    radius_sweep.OUT = out
    metric_rows: list[dict[str, Any]] = []
    casebook_rows: list[dict[str, Any]] = []
    for variant_id in variant_ids:
        rows = [row for row in mv_rows if row.get("variant") == variant_id]
        metrics, cases = radius_sweep._evaluate_variant(variant_id, rows)
        metric_rows.extend(metrics)
        casebook_rows.extend({**row, "variant_id": variant_id} for row in cases)
    aggregate_rows = phase7d._aggregate(metric_rows)
    area_by_variant: dict[str, dict[str, float]] = {}
    grouped_assign: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in assignment_summary_rows:
        grouped_assign[str(row.get("variant_id", ""))].append(row)
    for variant_id, rows in grouped_assign.items():
        area_by_variant[variant_id] = {
            "mean_generated_area_ratio": _mean([_num(row.get("generated_mask_area_ratio")) for row in rows]),
            "object_region_count_mean": _mean([_num(row.get("selected_region_count")) for row in rows]),
            "selected_region_fraction_mean": _mean([_num(row.get("selected_region_fraction")) for row in rows]),
            "mean_object_score_mean": _mean([_num(row.get("object_score_mean")) for row in rows if row.get("object_score_mean", "") != ""]),
            "mean_unknown_score_mean": _mean([_num(row.get("unknown_score_mean")) for row in rows if row.get("unknown_score_mean", "") != ""]),
            "mean_background_score_mean": _mean([_num(row.get("background_score_mean")) for row in rows if row.get("background_score_mean", "") != ""]),
        }
    variant_metric_rows: list[dict[str, Any]] = []
    for row in aggregate_rows:
        variant_id = str(row.get("variant_id", ""))
        variant_metric_rows.append(
            {
                "schema_version": "stream4d_v93_phase5b_variant_metric_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "variant_id": variant_id,
                "scene_id": "ALL_DEV",
                "split": "dev",
                "source_artifact": _rel(out / "mv_metric_aggregate_rows.csv"),
                "created_at": created_at,
                **row,
                **area_by_variant.get(variant_id, {}),
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
    gate_rows, failure_rows, any_pass = _phase5b_gate_rows(variant_metric_rows, specs, phase0)
    phase8_gate = _phase8_gate_audit(
        [
            field_root / "variant_metric_rows.csv",
            ROOT / "outputs/audit/v93_phase5_boundary_affinity_field_A512/variant_metric_rows.csv",
        ]
    )
    best_real = max(
        [row for row in variant_metric_rows if next((spec for spec in specs if spec["variant_id"] == row.get("variant_id")), {}).get("family") == "real"],
        key=lambda row: (_num(row.get("mean_MV_AP_window"), -999.0), _num(row.get("mean_MV_AP50_window"), -999.0)),
        default={},
    )
    best_control = max(
        [row for row in variant_metric_rows if next((spec for spec in specs if spec["variant_id"] == row.get("variant_id")), {}).get("family") == "control"],
        key=lambda row: (_num(row.get("mean_MV_AP_window"), -999.0), _num(row.get("mean_MV_AP50_window"), -999.0)),
        default={},
    )
    _write_csv(out / "variant_config_rows.csv", config_rows)
    _write_csv(out / "generated_mask_rows.csv", generated_rows)
    _write_csv(out / "mv_object_rows.csv", object_rows)
    _write_csv(out / "mv_object_frame_mask_rows.csv", mv_rows)
    _write_csv(out / "assignment_summary_rows.csv", assignment_summary_rows)
    _write_csv(out / "source_failure_rows.csv", source_failure_rows)
    _write_csv(out / "mv_metric_rows.csv", metric_rows)
    _write_csv(out / "mv_metric_aggregate_rows.csv", aggregate_rows)
    _write_csv(out / "variant_metric_rows.csv", variant_metric_rows)
    _write_csv(out / "variant_gate_rows.csv", gate_rows)
    _write_csv(out / "variant_failure_rows.csv", failure_rows)
    _write_csv(out / "casebook_rows.csv", casebook_rows)
    _write_json(out / "phase8_gate_audit.json", phase8_gate)
    summary = {
        "schema": "stream4d_v93_phase5b_unknown_background_field_summary_v1",
        "phase_id": PHASE_ID,
        "run_id": RUN_ID,
        "decision": "PASS_V93_PHASE5B_DEV_GATE" if any_pass else "NO_GO_V93_PHASE5B_UNKNOWN_BACKGROUND_FIELD_NO_GAIN",
        "created_at": created_at,
        "duration_sec": time.time() - started,
        "field_root": _rel(field_root),
        "field_shard_count": len(shard_paths),
        "processed_source_count": processed_source_count,
        "gpu_devices_visible": devices,
        "gpu_device_source_counts": dict(device_counts),
        "score_protocol_counts": dict(score_protocol_counts),
        "variant_count": len(specs),
        "best_real_variant_id": best_real.get("variant_id", ""),
        "best_real_MV_AP_window": best_real.get("mean_MV_AP_window", ""),
        "best_real_MV_AP50_window": best_real.get("mean_MV_AP50_window", ""),
        "best_control_variant_id": best_control.get("variant_id", ""),
        "best_control_MV_AP_window": best_control.get("mean_MV_AP_window", ""),
        "best_control_MV_AP50_window": best_control.get("mean_MV_AP50_window", ""),
        "any_phase5b_dev_gate_pass": bool(any_pass),
        "phase8_gate_audit": phase8_gate,
        "row_counts": {
            "variant_config_rows": len(config_rows),
            "generated_mask_rows": len(generated_rows),
            "mv_object_rows": len(object_rows),
            "mv_object_frame_mask_rows": len(mv_rows),
            "assignment_summary_rows": len(assignment_summary_rows),
            "source_failure_rows": len(source_failure_rows),
            "mv_metric_rows": len(metric_rows),
            "variant_metric_rows": len(variant_metric_rows),
            "variant_gate_rows": len(gate_rows),
            "variant_failure_rows": len(failure_rows),
            "casebook_rows": len(casebook_rows),
        },
        "input_artifacts": {
            _rel(path): _sha256(path)
            for path in [
                field_root / "field_artifact_manifest.json",
                field_root / "summary.json",
                V93_PHASE0 / "summary.json",
                V93_PHASE1 / "source_container_rows.csv",
                V93_PHASE3 / "region_node_rows.csv",
            ]
            if path.exists()
        },
        "uses_gt_for_prediction": False,
        "uses_future": False,
    }
    _write_json(out / "summary.json", summary)
    outputs = [
        out / "summary.json",
        out / "variant_config_rows.csv",
        out / "generated_mask_rows.csv",
        out / "mv_object_rows.csv",
        out / "mv_object_frame_mask_rows.csv",
        out / "assignment_summary_rows.csv",
        out / "source_failure_rows.csv",
        out / "mv_metric_rows.csv",
        out / "mv_metric_aggregate_rows.csv",
        out / "variant_metric_rows.csv",
        out / "variant_gate_rows.csv",
        out / "variant_failure_rows.csv",
        out / "casebook_rows.csv",
        out / "phase8_gate_audit.json",
    ]
    _write_json(out / "SHA256SUMS.json", {_rel(path): _sha256(path) for path in outputs if path.exists()})
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(OUT))
    parser.add_argument("--field-root", default=str(DEFAULT_FIELD_ROOT))
    parser.add_argument("--source-container-rows", default=str(V93_PHASE1 / "source_container_rows.csv"))
    parser.add_argument("--region-node-rows", default=str(V93_PHASE3 / "region_node_rows.csv"))
    parser.add_argument("--max-shards", type=int, default=0, help="Debug cap; 0 means all Phase5 shards.")
    parser.add_argument("--progress-every-shards", type=int, default=1)
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
