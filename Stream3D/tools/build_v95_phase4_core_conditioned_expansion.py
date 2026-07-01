#!/usr/bin/env python3
"""Materialize v95 core-conditioned expansion variants and evaluate MV_AP_window."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import run_v90_dev_extent_score_cross_audit as phase7d  # noqa: E402
from tools import run_v91_phase4_radius_sweep as radius_sweep  # noqa: E402


PHASE_ID = "v95_phase4_core_conditioned_expansion"
RUN_ID = "v95_phase4_familyA_core_conditioned_expansion"
OUT = ROOT / "outputs/audit/v95_phase4_familyA_core_conditioned_expansion"
PHASE0 = ROOT / "outputs/audit/v95_phase0_fact_lock/summary.json"
PHASE1 = ROOT / "outputs/audit/v95_phase1_physical_source_registry"
PHASE3 = ROOT / "outputs/audit/v95_phase3_object_query"


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
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        try:
            return path.resolve().relative_to(WORKSPACE_ROOT.resolve()).as_posix()
        except ValueError:
            return path.as_posix()


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
        return value.as_posix()
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key, "")) for key in fields})


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _num(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _int(value: Any, default: int = 0) -> int:
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _mean(values: list[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else 0.0


def _percentile(values: list[float], q: float) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.percentile(np.asarray(vals, dtype=np.float64), q)) if vals else 0.0


def _created_at() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _variant_specs(family: str) -> list[dict[str, Any]]:
    if family == "B":
        return [
            {
                "variant_id": "B0_core_radio_graph",
                "family": "real",
                "mode": "graph_propagation",
                "steps": 2,
                "min_edge_radio": 0.84,
                "score_quantile": 0.62,
                "area_cap": 0.58,
                "description": "Core seeds propagated over high RADIO-continuity graph edges.",
            },
            {
                "variant_id": "B1_core_radio_edge_barrier",
                "family": "real",
                "mode": "graph_propagation",
                "steps": 2,
                "min_edge_radio": 0.86,
                "min_edge_weight": 0.78,
                "score_quantile": 0.58,
                "risk_cap": 0.32,
                "area_cap": 0.62,
                "description": "Core seeds with RADIO plus edge-weight barrier.",
            },
            {
                "variant_id": "B2_core_radio_d4rt_conflict",
                "family": "real",
                "mode": "graph_propagation",
                "steps": 3,
                "min_edge_radio": 0.82,
                "score_quantile": 0.56,
                "d4rt_bonus": 0.22,
                "risk_cap": 0.38,
                "area_cap": 0.68,
                "description": "Core seeds with D4RT-supported graph propagation.",
            },
            {
                "variant_id": "B3_core_all_barriers_strict",
                "family": "real",
                "mode": "graph_propagation",
                "steps": 3,
                "min_edge_radio": 0.87,
                "min_edge_weight": 0.82,
                "score_quantile": 0.60,
                "risk_cap": 0.24,
                "area_cap": 0.56,
                "description": "Strict all-barrier graph propagation.",
            },
            {
                "variant_id": "B4_high_recall_graph_source_fallback",
                "family": "real",
                "mode": "graph_propagation",
                "steps": 4,
                "min_edge_radio": 0.78,
                "score_quantile": 0.45,
                "risk_cap": 0.55,
                "area_cap": 0.82,
                "description": "Higher-recall graph propagation with source-contained fallback.",
            },
        ]
    return [
        {
            "variant_id": "A0_core_only",
            "family": "real",
            "mode": "core_only",
            "description": "Confirmed query core only; lower-bound precision check.",
        },
        {
            "variant_id": "A1_core_top_unary_p18",
            "family": "real",
            "mode": "top_fraction",
            "top_fraction": 0.18,
            "area_cap": 0.58,
            "description": "Core plus top object-unary regions.",
        },
        {
            "variant_id": "A2_core_d4rt_supported",
            "family": "real",
            "mode": "d4rt_supported",
            "top_fraction": 0.14,
            "d4rt_threshold": 0.78,
            "area_cap": 0.62,
            "description": "Core plus D4RT-supported high-unary regions.",
        },
        {
            "variant_id": "A3_core_radio_threshold",
            "family": "real",
            "mode": "proto_threshold",
            "proto_quantile": 0.76,
            "area_cap": 0.66,
            "description": "Core plus source-local high residual-query similarity regions.",
        },
        {
            "variant_id": "A4_core_risk_capped_expansion",
            "family": "real",
            "mode": "risk_capped",
            "unary_quantile": 0.70,
            "risk_cap": 0.18,
            "area_cap": 0.64,
            "description": "Core plus high-unary regions under a risk cap.",
        },
    ]


class FrameWriter:
    def __init__(self, out: Path, variant_ids: list[str]) -> None:
        self.out = out
        self.variant_ids = variant_ids
        self.current: tuple[str, int] | None = None
        self.labels: dict[str, np.ndarray] = {}
        self.next_ids: dict[str, int] = {}

    def flush(self) -> None:
        if self.current is None:
            return
        scene, frame_id = self.current
        for variant_id, label in self.labels.items():
            out_dir = self.out / "generated_masks" / variant_id / scene / "mask"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{int(frame_id)}.png"
            if not cv2.imwrite(str(out_path), label.astype(np.uint16)):
                raise RuntimeError(f"failed to write {out_path}")
        self.current = None
        self.labels = {}
        self.next_ids = {}

    def ensure_frame(self, scene: str, frame_id: int, shape: tuple[int, int]) -> None:
        key = (scene, int(frame_id))
        if self.current != key:
            self.flush()
            self.current = key
            self.labels = {variant_id: np.zeros(shape, dtype=np.uint16) for variant_id in self.variant_ids}
            self.next_ids = {variant_id: 1 for variant_id in self.variant_ids}

    def add_mask(self, variant_id: str, mask: np.ndarray) -> int:
        label = self.labels[variant_id]
        write = np.asarray(mask, dtype=bool) & (label == 0)
        if not np.any(write):
            return -1
        new_id = int(self.next_ids[variant_id])
        label[write] = new_id
        self.next_ids[variant_id] = new_id + 1
        return new_id


def _load_source_meta(path: Path) -> dict[tuple[str, str, int, int], dict[str, str]]:
    out: dict[tuple[str, str, int, int], dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if _bool(row.get("uses_gt_for_prediction")) or _bool(row.get("uses_future")):
                continue
            key = (row["scene_id"], row["window_id"], _int(row["frame_id"]), _int(row["source_mask_id"]))
            out[key] = dict(row)
    return out


def _load_region_nodes(path: Path, allowed_sources: set[tuple[str, str, int, int]] | None) -> dict[tuple[str, str, int, int], dict[int, dict[str, str]]]:
    out: dict[tuple[str, str, int, int], dict[int, dict[str, str]]] = defaultdict(dict)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["scene_id"], row["window_id"], _int(row["frame_id"]), _int(row["source_mask_id"]))
            if allowed_sources is not None and key not in allowed_sources:
                continue
            if _bool(row.get("uses_gt_for_prediction")) or _bool(row.get("uses_future")):
                continue
            out[key][_int(row.get("region_index"))] = dict(row)
    return dict(out)


def _load_edge_adjacency(
    path: Path,
    allowed_sources: set[tuple[str, str, int, int]] | None,
    region_nodes: dict[tuple[str, str, int, int], dict[int, dict[str, str]]],
) -> dict[tuple[str, str, int, int], dict[int, list[tuple[int, float, float]]]]:
    id_to_index: dict[tuple[str, str, int, int], dict[str, int]] = {}
    for key, nodes in region_nodes.items():
        id_to_index[key] = {str(row.get("region_id", "")): int(region_index) for region_index, row in nodes.items()}
    adjacency: dict[tuple[str, str, int, int], dict[int, list[tuple[int, float, float]]]] = defaultdict(lambda: defaultdict(list))
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["scene_id"], row["window_id"], _int(row["frame_id"]), _int(row["source_mask_id"]))
            if allowed_sources is not None and key not in allowed_sources:
                continue
            if key not in id_to_index:
                continue
            if _bool(row.get("uses_gt_for_prediction")) or _bool(row.get("uses_future")):
                continue
            if not _bool(row.get("is_adjacent")):
                continue
            lookup = id_to_index[key]
            u_raw = str(row.get("region_u") or row.get("region_id_a") or "")
            v_raw = str(row.get("region_v") or row.get("region_id_b") or "")
            if u_raw not in lookup or v_raw not in lookup:
                continue
            u = int(lookup[u_raw])
            v = int(lookup[v_raw])
            if u == v:
                continue
            radio = max(0.0, _num(row.get("radio_cosine"), 0.0))
            edge_weight = max(0.0, _num(row.get("edge_weight"), radio))
            adjacency[key][u].append((v, radio, edge_weight))
            adjacency[key][v].append((u, radio, edge_weight))
    return {key: dict(value) for key, value in adjacency.items()}


def _read_label(mask_path: Path) -> np.ndarray:
    label = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if label is None:
        raise RuntimeError(f"failed to read mask label image: {mask_path}")
    if label.ndim == 3:
        label = label[:, :, 0]
    return label


def _node_mask(nodes: dict[int, dict[str, str]], selected: set[int], source_mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(source_mask, dtype=bool)
    for idx in selected:
        node = nodes.get(int(idx))
        if not node:
            continue
        x0 = _int(node.get("bbox_x0"), 0)
        x1 = _int(node.get("bbox_x1"), 0)
        y0 = _int(node.get("bbox_y0"), 0)
        y1 = _int(node.get("bbox_y1"), 0)
        out[y0 : y1 + 1, x0 : x1 + 1] |= source_mask[y0 : y1 + 1, x0 : x1 + 1]
    return out & source_mask


def _selected_area_ratio(rows: list[dict[str, Any]], selected: set[int]) -> float:
    return float(sum(_num(row.get("region_area_ratio")) for row in rows if int(row["region_index"]) in selected))


def _cap_by_area(rows: list[dict[str, Any]], selected: set[int], cap: float) -> set[int]:
    if cap >= 1.0:
        return set(selected)
    core = {int(row["region_index"]) for row in rows if _bool(row.get("is_core_region"))}
    ranked = sorted(
        [row for row in rows if int(row["region_index"]) in selected],
        key=lambda row: (_bool(row.get("is_core_region")), _num(row.get("object_unary_score"))),
        reverse=True,
    )
    out: set[int] = set()
    area = 0.0
    for row in ranked:
        idx = int(row["region_index"])
        ratio = _num(row.get("region_area_ratio"))
        if idx not in core and area + ratio > cap and out:
            continue
        out.add(idx)
        area += ratio
    return out | core


def _select_regions(rows: list[dict[str, Any]], spec: dict[str, Any], adjacency: dict[int, list[tuple[int, float, float]]] | None = None) -> set[int]:
    if not rows:
        return set()
    indices = [int(row["region_index"]) for row in rows]
    core = {int(row["region_index"]) for row in rows if _bool(row.get("is_core_region"))}
    if not core:
        best = max(rows, key=lambda row: _num(row.get("object_unary_score")))
        core = {int(best["region_index"])}
    mode = str(spec["mode"])
    selected = set(core)
    unary = np.asarray([_num(row.get("object_unary_score")) for row in rows], dtype=np.float32)
    proto = np.asarray([_num(row.get("proto_similarity")) for row in rows], dtype=np.float32)
    d4rt = np.asarray([_num(row.get("D4RT_witness_score")) for row in rows], dtype=np.float32)
    risk = np.asarray([_num(row.get("risk_score")) for row in rows], dtype=np.float32)
    if mode == "top_fraction":
        k = max(len(core), int(math.ceil(float(spec.get("top_fraction", 0.18)) * len(rows))))
        for pos in np.argsort(unary)[::-1][:k].tolist():
            selected.add(indices[int(pos)])
    elif mode == "d4rt_supported":
        k = max(len(core), int(math.ceil(float(spec.get("top_fraction", 0.14)) * len(rows))))
        for pos in np.argsort(0.70 * unary + 0.30 * d4rt)[::-1][:k].tolist():
            selected.add(indices[int(pos)])
        for pos, value in enumerate(d4rt.tolist()):
            if value >= float(spec.get("d4rt_threshold", 0.78)) and unary[pos] >= float(np.percentile(unary, 60)):
                selected.add(indices[pos])
    elif mode == "proto_threshold":
        threshold = float(np.percentile(proto, 100.0 * float(spec.get("proto_quantile", 0.76))))
        for pos, value in enumerate(proto.tolist()):
            if value >= threshold:
                selected.add(indices[pos])
    elif mode == "risk_capped":
        threshold = float(np.percentile(unary, 100.0 * float(spec.get("unary_quantile", 0.70))))
        for pos, value in enumerate(unary.tolist()):
            if value >= threshold and risk[pos] <= float(spec.get("risk_cap", 0.18)):
                selected.add(indices[pos])
    elif mode == "graph_propagation":
        adjacency = adjacency or {}
        row_by_index = {int(row["region_index"]): row for row in rows}
        score_by_index = {
            int(row["region_index"]): (
                _num(row.get("object_unary_score"))
                + float(spec.get("d4rt_bonus", 0.0)) * _num(row.get("D4RT_witness_score"))
                - 0.12 * _num(row.get("risk_score"))
            )
            for row in rows
        }
        threshold = float(np.percentile(np.asarray(list(score_by_index.values()), dtype=np.float32), 100.0 * float(spec.get("score_quantile", 0.58))))
        min_edge_radio = float(spec.get("min_edge_radio", 0.82))
        min_edge_weight = float(spec.get("min_edge_weight", 0.0))
        risk_cap = float(spec.get("risk_cap", 1.0))
        frontier = set(core)
        for _ in range(max(1, int(spec.get("steps", 2)))):
            new_frontier: set[int] = set()
            for src in frontier:
                for dst, edge_radio, edge_weight in adjacency.get(src, []):
                    if dst in selected or dst not in row_by_index:
                        continue
                    row = row_by_index[dst]
                    if float(edge_radio) < min_edge_radio:
                        continue
                    if float(edge_weight) < min_edge_weight:
                        continue
                    if _num(row.get("risk_score")) > risk_cap:
                        continue
                    edge_adjusted = score_by_index[dst] + 0.08 * float(edge_radio) + 0.06 * float(edge_weight)
                    if edge_adjusted >= threshold:
                        selected.add(dst)
                        new_frontier.add(dst)
            if not new_frontier:
                break
            frontier = new_frontier
        if len(selected) <= len(core) and str(spec.get("variant_id")) == "B4_high_recall_graph_source_fallback":
            fallback_k = max(len(core), int(math.ceil(0.35 * len(rows))))
            for pos in np.argsort(unary)[::-1][:fallback_k].tolist():
                if risk[int(pos)] <= risk_cap:
                    selected.add(indices[int(pos)])
    return _cap_by_area(rows, selected, float(spec.get("area_cap", 1.0)))


def _iter_unary_groups(path: Path, max_sources: int = 0):
    current_key: tuple[str, str, int, int, str] | None = None
    current_rows: list[dict[str, Any]] = []
    seen_sources: set[tuple[str, str, int, int]] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            source = (row["scene_id"], row["window_id"], _int(row["frame_id"]), _int(row["source_mask_id"]))
            if max_sources > 0 and source not in seen_sources and len(seen_sources) >= max_sources:
                continue
            seen_sources.add(source)
            key = (*source, row["object_id"])
            if current_key is not None and key != current_key:
                yield current_key, current_rows
                current_rows = []
            current_key = key
            current_rows.append(dict(row))
    if current_key is not None:
        yield current_key, current_rows


def _prepare_rows(rows: list[dict[str, Any]], source_area: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        row = dict(row)
        row["region_area_ratio"] = 0.0
        out.append(row)
    return out


def _gate_rows(variant_metric_rows: list[dict[str, Any]], specs: list[dict[str, Any]], phase0: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool, dict[str, Any]]:
    spec_by_id = {spec["variant_id"]: spec for spec in specs}
    required_ap = _num(phase0.get("required_MV_AP_window"))
    required_ap50 = _num(phase0.get("required_MV_AP50_window"))
    v91_ap = _num(phase0.get("v91_best_MV_AP_window"))
    v91_ap50 = _num(phase0.get("v91_best_MV_AP50_window"))
    control_ap = _num(phase0.get("best_control_MV_AP_window"))
    control_ap50 = _num(phase0.get("best_control_MV_AP50_window"))
    candidate_required_ap = max(v91_ap + 0.002, control_ap + 0.005)
    candidate_required_ap50 = max(v91_ap50 + 0.004, control_ap50 + 0.010)
    gate_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    any_pass = False
    best_real: dict[str, Any] = {}
    for row in variant_metric_rows:
        variant_id = str(row.get("variant_id", ""))
        spec = spec_by_id.get(variant_id, {})
        mv_ap = _num(row.get("mean_MV_AP_window"))
        mv_ap50 = _num(row.get("mean_MV_AP50_window"))
        collision = _int(row.get("same_frame_collision_count"))
        missing = _int(row.get("missing_mask_raster_count"))
        is_real = spec.get("family") == "real"
        progress_gate = bool(mv_ap >= candidate_required_ap and mv_ap50 >= candidate_required_ap50)
        final_threshold_gate = bool(mv_ap >= required_ap and mv_ap50 >= required_ap50)
        provenance_gate = bool(collision == 0 and missing == 0 and not _bool(row.get("uses_gt_for_prediction")) and not _bool(row.get("uses_future")))
        pass_gate = bool(is_real and progress_gate and provenance_gate)
        any_pass = any_pass or pass_gate
        if is_real and (not best_real or (mv_ap, mv_ap50) > (_num(best_real.get("mean_MV_AP_window"), -999.0), _num(best_real.get("mean_MV_AP50_window"), -999.0))):
            best_real = dict(row)
        gate_rows.append(
            {
                "schema_version": "stream4d_v95_phase4_variant_gate_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "variant_id": variant_id,
                "family": spec.get("family", ""),
                "MV_AP_window": mv_ap,
                "MV_AP50_window": mv_ap50,
                "MV_AP25_window": row.get("mean_MV_AP25_window", ""),
                "ScoreFreeMatch50_window": row.get("mean_score_free_Match50_window", ""),
                "candidate_required_MV_AP_window": candidate_required_ap,
                "candidate_required_MV_AP50_window": candidate_required_ap50,
                "final_required_MV_AP_window": required_ap,
                "final_required_MV_AP50_window": required_ap50,
                "phase4_candidate_gate_pass": pass_gate,
                "phase4_final_threshold_gate_pass": final_threshold_gate,
                "progress_gate_pass": progress_gate,
                "provenance_gate_pass": provenance_gate,
                "same_frame_collision_count": collision,
                "missing_mask_raster_count": missing,
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
        if is_real and not pass_gate:
            failure_rows.append(
                {
                    "schema_version": "stream4d_v95_phase4_failure_v1",
                    "phase_id": PHASE_ID,
                    "run_id": RUN_ID,
                    "variant_id": variant_id,
                    "failure_type": "PHASE4_FAMILY_A_NO_CANDIDATE_GATE",
                    "MV_AP_window": mv_ap,
                    "MV_AP50_window": mv_ap50,
                    "repair_direction": "If family A fails, switch to seeded random-walker / graph propagation per v95 plan; do not continue threshold sweep.",
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )
    return gate_rows, failure_rows, any_pass, best_real


def run(args: argparse.Namespace) -> dict[str, Any]:
    global RUN_ID
    RUN_ID = f"v95_phase4_family{args.family}_core_conditioned_expansion"
    started = time.time()
    out = _resolve(args.output_root)
    if out.exists() and args.clean:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    phase0 = json.loads(PHASE0.read_text(encoding="utf-8"))
    phase3 = json.loads((_resolve(args.query_root) / "summary.json").read_text(encoding="utf-8"))
    if phase3.get("decision") != "PASS_V95_PHASE3_OBJECT_QUERY_READY":
        raise RuntimeError("v95 Phase3 must pass before Phase4")
    specs = _variant_specs(str(args.family))
    variant_ids = [spec["variant_id"] for spec in specs]
    source_meta = _load_source_meta(_resolve(args.source_container_rows))
    allowed_sources = None
    if int(args.max_sources) > 0:
        allowed_sources = set()
        with (_resolve(args.query_root) / "region_object_unary_rows.csv").open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                allowed_sources.add((row["scene_id"], row["window_id"], _int(row["frame_id"]), _int(row["source_mask_id"])))
                if len(allowed_sources) >= int(args.max_sources):
                    break
    region_nodes = _load_region_nodes(_resolve(args.region_node_rows), allowed_sources)
    edge_adjacency = (
        _load_edge_adjacency(_resolve(args.region_edge_rows), allowed_sources, region_nodes)
        if str(args.family) == "B"
        else {}
    )
    frame_writer = FrameWriter(out, variant_ids)
    label_cache: dict[tuple[str, int], np.ndarray] = {}
    generated_rows: list[dict[str, Any]] = []
    mv_rows: list[dict[str, Any]] = []
    ownership_rows: list[dict[str, Any]] = []
    casebook_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    area_ratios_by_variant: dict[str, list[float]] = defaultdict(list)
    core_retention_by_variant: dict[str, list[float]] = defaultdict(list)
    processed_groups = 0

    config_rows = [
        {
            "schema_version": "stream4d_v95_phase4_variant_config_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "created_at": _created_at(),
            **spec,
            "query_root": _rel(_resolve(args.query_root)),
            "selected_query_family": phase3.get("selected_query_family", ""),
            "uses_gt_for_prediction": False,
            "uses_future": False,
        }
        for spec in specs
    ]

    for group_key, raw_rows in _iter_unary_groups(_resolve(args.query_root) / "region_object_unary_rows.csv", int(args.max_sources)):
        scene, window, frame_id, mask_id, object_id = group_key
        source_key = (scene, window, frame_id, mask_id)
        meta = source_meta.get(source_key)
        nodes = region_nodes.get(source_key, {})
        if not meta or not nodes:
            failure_rows.append(
                {
                    "schema_version": "stream4d_v95_phase4_failure_v1",
                    "phase_id": PHASE_ID,
                    "run_id": RUN_ID,
                    "failure_type": "missing_source_meta_or_region_nodes",
                    "scene_id": scene,
                    "window_id": window,
                    "frame_id": frame_id,
                    "source_mask_id": mask_id,
                    "object_id": object_id,
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )
            continue
        mask_path = _resolve(meta.get("mask_path", ""))
        frame_key = (scene, int(frame_id))
        if frame_key not in label_cache:
            label_cache[frame_key] = _read_label(mask_path)
        label = label_cache[frame_key]
        source_mask = label == int(mask_id)
        if not np.any(source_mask):
            continue
        frame_writer.ensure_frame(scene, int(frame_id), source_mask.shape)
        rows: list[dict[str, Any]] = []
        source_area = float(max(1, int(np.count_nonzero(source_mask))))
        for row in raw_rows:
            idx = _int(row.get("region_index"))
            node = nodes.get(idx)
            if not node:
                continue
            area = max(1.0, _num(node.get("pixel_count"), _num(node.get("area_px"), 1.0)))
            row = dict(row)
            row["region_area_ratio"] = float(area / source_area)
            rows.append(row)
        if not rows:
            continue
        processed_groups += 1
        core_indices = {int(row["region_index"]) for row in rows if _bool(row.get("is_core_region"))}
        for spec in specs:
            variant_id = spec["variant_id"]
            selected = _select_regions(rows, spec, edge_adjacency.get(source_key, {}))
            if not selected:
                continue
            mask = _node_mask(nodes, selected, source_mask)
            new_id = frame_writer.add_mask(variant_id, mask)
            if new_id <= 0:
                continue
            selected_area = int(np.count_nonzero(mask))
            if selected_area <= 0:
                continue
            selected_scores = [_num(row.get("object_unary_score")) for row in rows if int(row["region_index"]) in selected]
            selected_proto = [_num(row.get("proto_similarity")) for row in rows if int(row["region_index"]) in selected]
            selected_risk = [_num(row.get("risk_score")) for row in rows if int(row["region_index"]) in selected]
            retained_core = len(core_indices & selected) / max(1, len(core_indices))
            area_ratio = selected_area / source_area
            area_ratios_by_variant[variant_id].append(float(area_ratio))
            core_retention_by_variant[variant_id].append(float(retained_core))
            object_score = float(
                np.clip(
                    0.50
                    + 0.42 * _mean(selected_scores)
                    + 0.10 * _mean(selected_proto)
                    - 0.18 * _mean(selected_risk)
                    + 0.10 * retained_core,
                    0.0,
                    1.0,
                )
            )
            gen_path = out / "generated_masks" / variant_id / scene / "mask" / f"{int(frame_id)}.png"
            generated_rows.append(
                {
                    "schema_version": "stream4d_v95_phase4_generated_mask_v1",
                    "phase_id": PHASE_ID,
                    "run_id": RUN_ID,
                    "variant_id": variant_id,
                    "scene_id": scene,
                    "window_id": window,
                    "frame_id": int(frame_id),
                    "source_mask_id": int(mask_id),
                    "object_id": object_id,
                    "new_mask_id": int(new_id),
                    "generated_mask_path": _rel(gen_path),
                    "source_mask_area": int(source_area),
                    "generated_mask_area": int(selected_area),
                    "generated_area_ratio": float(area_ratio),
                    "selected_region_count": int(len(selected)),
                    "core_region_count": int(len(core_indices)),
                    "core_retention_rate": float(retained_core),
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )
            mv_object_id = f"{variant_id}:{object_id}"
            mv_rows.append(
                {
                    "split": "dev",
                    "scene_id": scene,
                    "source_variant": variant_id,
                    "variant": variant_id,
                    "mv_object_id": mv_object_id,
                    "frame_id": int(frame_id),
                    "mask_id": int(new_id),
                    "frame_mask_score": object_score,
                    "object_score": object_score,
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                    "uses_rgbd_pose_mesh": False,
                    "materializable": True,
                    "selection_reason": f"v95_phase4_{spec['mode']}_from_{phase3.get('selected_query_family', '')}",
                }
            )
            ownership_rows.append(
                {
                    "schema_version": "stream4d_v95_phase4_ownership_audit_v1",
                    "phase_id": PHASE_ID,
                    "run_id": RUN_ID,
                    "variant_id": variant_id,
                    "scene_id": scene,
                    "window_id": window,
                    "frame_id": int(frame_id),
                    "source_mask_id": int(mask_id),
                    "object_id": object_id,
                    "selected_region_count": int(len(selected)),
                    "total_region_count": int(len(rows)),
                    "generated_area_ratio": float(area_ratio),
                    "core_retention_rate": float(retained_core),
                    "object_score": object_score,
                    "unknown_area_ratio": 0.0,
                    "background_area_ratio": float(max(0.0, 1.0 - area_ratio)),
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )
        if int(args.progress_every_groups) > 0 and processed_groups % int(args.progress_every_groups) == 0:
            print(
                json.dumps(
                    {
                        "phase": PHASE_ID,
                        "processed_source_object_groups": processed_groups,
                        "generated_mask_rows": len(generated_rows),
                        "elapsed_sec": time.time() - started,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    frame_writer.flush()
    _write_csv(out / "variant_config_rows.csv", config_rows)
    _write_csv(out / "generated_mask_rows.csv", generated_rows)
    _write_csv(out / "mv_object_rows.csv", [{"variant_id": row["variant"], "mv_object_id": row["mv_object_id"], "uses_gt_for_prediction": False, "uses_future": False} for row in mv_rows])
    _write_csv(out / "mv_object_frame_mask_rows.csv", mv_rows)
    _write_csv(out / "ownership_audit_rows.csv", ownership_rows)
    _write_csv(out / "variant_failure_rows_pre_eval.csv", failure_rows)

    metric_rows: list[dict[str, Any]] = []
    eval_case_rows: list[dict[str, Any]] = []
    if not bool(args.skip_eval):
        radius_sweep.OUT = out
        for spec in specs:
            variant_id = spec["variant_id"]
            rows = [row for row in mv_rows if row.get("variant") == variant_id]
            metrics, cases = radius_sweep._evaluate_variant(variant_id, rows)
            metric_rows.extend(metrics)
            eval_case_rows.extend({**case, "variant_id": variant_id} for case in cases)
    aggregate_rows = phase7d._aggregate(metric_rows) if metric_rows else []
    variant_metric_rows: list[dict[str, Any]] = []
    aggregate_by_variant = {row.get("variant_id", ""): row for row in aggregate_rows}
    for spec in specs:
        variant_id = spec["variant_id"]
        agg = aggregate_by_variant.get(variant_id, {})
        variant_metric_rows.append(
            {
                "schema_version": "stream4d_v95_phase4_variant_metric_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "variant_id": variant_id,
                "family": spec["family"],
                "mean_MV_AP_window": agg.get("mean_MV_AP_window", ""),
                "mean_MV_AP50_window": agg.get("mean_MV_AP50_window", ""),
                "mean_MV_AP25_window": agg.get("mean_MV_AP25_window", ""),
                "mean_score_free_Match50_window": agg.get("mean_score_free_Match50_window", ""),
                "mean_generated_area_ratio": _mean(area_ratios_by_variant.get(variant_id, [])),
                "generated_area_ratio_p10": _percentile(area_ratios_by_variant.get(variant_id, []), 10),
                "generated_area_ratio_p90": _percentile(area_ratios_by_variant.get(variant_id, []), 90),
                "core_retention_rate": _mean(core_retention_by_variant.get(variant_id, [])),
                "same_frame_collision_count": agg.get("same_frame_collision_count", 0),
                "missing_mask_raster_count": agg.get("missing_mask_raster_count", 0),
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
    gate_rows, gate_failure_rows, any_pass, best_real = _gate_rows(variant_metric_rows, specs, phase0) if metric_rows else ([], [], False, {})
    failure_rows.extend(gate_failure_rows)
    _write_csv(out / "mv_metric_rows.csv", metric_rows)
    _write_csv(out / "mv_metric_aggregate_rows.csv", aggregate_rows)
    _write_csv(out / "variant_metric_rows.csv", variant_metric_rows)
    _write_csv(out / "variant_gate_rows.csv", gate_rows)
    _write_csv(out / "variant_failure_rows.csv", failure_rows)
    _write_csv(out / "casebook_rows.csv", eval_case_rows + casebook_rows)
    _write_csv(out / "mv_iou_matrix_rows.csv", [])
    _write_csv(out / "scorefree_match_rows.csv", [])

    summary = {
        "schema": "stream4d_v95_phase4_core_conditioned_expansion_summary_v1",
        "phase_id": PHASE_ID,
        "run_id": RUN_ID,
        "created_at": _created_at(),
        "decision": (
            f"PASS_V95_PHASE4_FAMILY_{args.family}_CANDIDATE_GATE"
            if any_pass
            else ("SMOKE_V95_PHASE4_MATERIALIZATION_ONLY" if bool(args.skip_eval) else f"NO_GO_V95_PHASE4_FAMILY_{args.family}_NO_CANDIDATE_GATE")
        ),
        "family": "A_core_preserving_threshold_expansion" if str(args.family) == "A" else "B_seeded_graph_propagation",
        "query_root": _rel(_resolve(args.query_root)),
        "selected_query_family": phase3.get("selected_query_family", ""),
        "processed_source_object_groups": processed_groups,
        "best_real_variant_id": best_real.get("variant_id", ""),
        "best_real_MV_AP_window": best_real.get("mean_MV_AP_window", ""),
        "best_real_MV_AP50_window": best_real.get("mean_MV_AP50_window", ""),
        "best_real_MV_AP25_window": best_real.get("mean_MV_AP25_window", ""),
        "best_real_ScoreFreeMatch50_window": best_real.get("mean_score_free_Match50_window", ""),
        "phase4_candidate_gate_pass": any_pass,
        "uses_gt_for_prediction_count": 0,
        "uses_future_count": 0,
        "uses_gt_for_prediction": False,
        "uses_future": False,
        "duration_sec": time.time() - started,
        "row_counts": {
            "variant_config_rows": len(config_rows),
            "generated_mask_rows": len(generated_rows),
            "mv_object_rows": len(mv_rows),
            "mv_object_frame_mask_rows": len(mv_rows),
            "ownership_audit_rows": len(ownership_rows),
            "mv_metric_rows": len(metric_rows),
            "mv_metric_aggregate_rows": len(aggregate_rows),
            "variant_metric_rows": len(variant_metric_rows),
            "variant_gate_rows": len(gate_rows),
            "variant_failure_rows": len(failure_rows),
            "casebook_rows": len(eval_case_rows + casebook_rows),
        },
    }
    _write_json(out / "summary.json", summary)
    outputs = [
        out / "variant_config_rows.csv",
        out / "generated_mask_rows.csv",
        out / "mv_object_rows.csv",
        out / "mv_object_frame_mask_rows.csv",
        out / "ownership_audit_rows.csv",
        out / "mv_metric_rows.csv",
        out / "mv_metric_aggregate_rows.csv",
        out / "variant_metric_rows.csv",
        out / "variant_gate_rows.csv",
        out / "variant_failure_rows.csv",
        out / "casebook_rows.csv",
        out / "summary.json",
    ]
    _write_json(out / "SHA256SUMS.json", {_rel(path): _sha256(path) for path in outputs if path.exists()})
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-root", default=str(PHASE3))
    parser.add_argument("--output-root", default=str(OUT))
    parser.add_argument("--source-container-rows", default=str(PHASE1 / "source_container_rows.csv"))
    parser.add_argument("--region-node-rows", default=str(PHASE1 / "region_node_rows.csv"))
    parser.add_argument("--region-edge-rows", default=str(PHASE1 / "region_edge_rows.csv"))
    parser.add_argument("--family", choices=["A", "B"], default="A")
    parser.add_argument("--max-sources", type=int, default=0)
    parser.add_argument("--progress-every-groups", type=int, default=512)
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
