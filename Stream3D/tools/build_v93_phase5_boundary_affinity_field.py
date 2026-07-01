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
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - audited in summary.json
    torch = None  # type: ignore[assignment]

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - audited in summary.json
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import build_v92_phase5_source_container_field as v92field  # noqa: E402
from tools import run_v90_dev_extent_score_cross_audit as phase7d  # noqa: E402
from tools import run_v91_phase4_radius_sweep as radius_sweep  # noqa: E402


PHASE_ID = "v93_phase5_boundary_affinity_field"
RUN_ID = "v93_phase5_boundary_affinity_field_gpu_triton"
OUT = ROOT / "outputs/audit/v93_phase5_boundary_affinity_field"

V93_PHASE0 = ROOT / "outputs/audit/v93_phase0_contract"
V93_PHASE1 = ROOT / "outputs/audit/v93_phase1_source_edge_registry"
V93_PHASE3 = ROOT / "outputs/audit/v93_phase3_region_edge_graph"
V93_PHASE4 = ROOT / "outputs/audit/v93_phase4_cue_isolation"
LOWRES_SUPPORT = ROOT / "outputs/audit/v90_phase3_v82_full_support/native_carrier_support_rows.csv"
HR2_SUPPORT = ROOT / "outputs/audit/v92_phase3_d4rt_highres_hr2_grid16/highres_native_carrier_support_rows.csv"
BASE_SOURCE_VARIANT = "V91_AD4_sr2_adapt_sig8_b05_j075_r12_source"

FIELD_UNARY_FIELDS = [
    "schema_version",
    "phase_id",
    "run_id",
    "scene_id",
    "split",
    "window_id",
    "frame_id",
    "source_mask_id",
    "object_hypothesis_id",
    "variant_id",
    "region_id",
    "region_index",
    "p_object",
    "unary_total_logit",
    "unary_d4rt",
    "unary_semantic",
    "edge_inside_score",
    "source_edge_barrier_score",
    "nested_edge_barrier_score",
    "competing_edge_barrier_score",
    "hard_negative_penalty",
    "pairwise_message_score",
    "selected_as_object_before_frame_wta",
    "solver_backend",
    "gpu_device",
    "uses_gt_for_prediction",
    "uses_future",
]

REGION_ASSIGNMENT_FIELDS = [
    "schema_version",
    "phase_id",
    "run_id",
    "scene_id",
    "split",
    "window_id",
    "frame_id",
    "source_mask_id",
    "object_hypothesis_id",
    "variant_id",
    "region_id",
    "region_index",
    "assigned_label",
    "assigned_object_id",
    "p_object",
    "selected_as_object_before_frame_wta",
    "multilabel_source_object_count",
    "solver_backend",
    "gpu_device",
    "uses_gt_for_prediction",
    "uses_future",
]

BASE_FIELD_EDGE_FIELDS = [
    "schema_version",
    "phase_id",
    "run_id",
    "scene_id",
    "split",
    "window_id",
    "frame_id",
    "source_mask_id",
    "region_u",
    "region_v",
    "region_index_u",
    "region_index_v",
    "is_adjacent",
    "spatial_distance",
    "radio_cosine",
    "dino_cosine",
    "mask_edge_barrier",
    "nested_edge_barrier",
    "competing_edge_barrier",
    "semantic_gradient_barrier",
    "rgb_gradient_barrier",
    "d4rt_conflict_barrier",
    "base_edge_weight",
    "solver_backend",
    "gpu_device",
    "uses_gt_for_prediction",
    "uses_future",
]


if triton is not None and tl is not None:

    @triton.jit
    def _unary_kernel(
        sem,
        d4rt,
        inside,
        source_bar,
        nested_bar,
        competing_bar,
        negative,
        w_d4rt,
        w_sem,
        w_inside,
        w_source_bar,
        w_nested_bar,
        w_competing_bar,
        w_negative,
        bias,
        out,
        n_nodes: tl.constexpr,
        block: tl.constexpr,
    ):
        variant = tl.program_id(0)
        block_id = tl.program_id(1)
        offsets = block_id * block + tl.arange(0, block)
        mask = offsets < n_nodes
        value = (
            tl.load(w_d4rt + variant) * tl.load(d4rt + offsets, mask=mask, other=0.0)
            + tl.load(w_sem + variant) * tl.load(sem + offsets, mask=mask, other=0.0)
            + tl.load(w_inside + variant) * tl.load(inside + offsets, mask=mask, other=0.0)
            - tl.load(w_source_bar + variant) * tl.load(source_bar + offsets, mask=mask, other=0.0)
            - tl.load(w_nested_bar + variant) * tl.load(nested_bar + offsets, mask=mask, other=0.0)
            - tl.load(w_competing_bar + variant) * tl.load(competing_bar + offsets, mask=mask, other=0.0)
            - tl.load(w_negative + variant) * tl.load(negative + offsets, mask=mask, other=0.0)
            + tl.load(bias + variant)
        )
        tl.store(out + variant * n_nodes + offsets, value, mask=mask)

    @triton.jit
    def _edge_message_kernel(
        prob,
        edge_u,
        edge_v,
        edge_weight,
        accum,
        denom,
        n_nodes: tl.constexpr,
        n_edges: tl.constexpr,
        block: tl.constexpr,
    ):
        variant = tl.program_id(0)
        block_id = tl.program_id(1)
        offsets = block_id * block + tl.arange(0, block)
        mask = offsets < n_edges
        u = tl.load(edge_u + offsets, mask=mask, other=0)
        v = tl.load(edge_v + offsets, mask=mask, other=0)
        w = tl.load(edge_weight + variant * n_edges + offsets, mask=mask, other=0.0)
        pu = tl.load(prob + variant * n_nodes + u, mask=mask, other=0.0)
        pv = tl.load(prob + variant * n_nodes + v, mask=mask, other=0.0)
        tl.atomic_add(accum + variant * n_nodes + u, w * pv, sem="relaxed", mask=mask)
        tl.atomic_add(accum + variant * n_nodes + v, w * pu, sem="relaxed", mask=mask)
        tl.atomic_add(denom + variant * n_nodes + u, w, sem="relaxed", mask=mask)
        tl.atomic_add(denom + variant * n_nodes + v, w, sem="relaxed", mask=mask)


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


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = []
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


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "allow"}


def _int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return float(default)
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _mean(values: list[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else 0.0


def _created_at() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _safe_variant(variant_id: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in variant_id)


def _stable_seed(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16) % (2**32)


def _region_index(region_id: str) -> int:
    tail = str(region_id).split(":")[-1]
    if tail.startswith("r") and tail[1:].isdigit():
        return int(tail[1:])
    return -1


def _variant_specs() -> list[dict[str, Any]]:
    return [
        {
            "variant_id": "F0_whole_source_baseline",
            "family": "baseline",
            "mode": "whole",
            "threshold": 0.0,
            "area_cap": 1.0,
            "smooth": 0.0,
            "weights": dict(d4rt=0.0, sem=0.0, inside=0.0, source_bar=0.0, nested_bar=0.0, competing_bar=0.0, negative=0.0, bias=4.0),
            "edge": dict(semantic=0.0, source=0.0, nested=0.0, competing=0.0, d4rt=0.0, scale=0.0),
        },
        {
            "variant_id": "F1_D4RT_RADIO_unary",
            "family": "real",
            "mode": "binary",
            "threshold": 0.54,
            "area_cap": 0.90,
            "smooth": 0.0,
            "weights": dict(d4rt=1.05, sem=0.85, inside=0.15, source_bar=0.05, nested_bar=0.05, competing_bar=0.08, negative=0.45, bias=-0.55),
            "edge": dict(semantic=0.0, source=0.0, nested=0.0, competing=0.0, d4rt=0.0, scale=0.0),
        },
        {
            "variant_id": "F2_D4RT_RADIO_pairwise",
            "family": "real",
            "mode": "binary_pairwise",
            "threshold": 0.54,
            "area_cap": 0.88,
            "smooth": 1.10,
            "weights": dict(d4rt=1.00, sem=0.95, inside=0.12, source_bar=0.05, nested_bar=0.06, competing_bar=0.08, negative=0.45, bias=-0.55),
            "edge": dict(semantic=1.45, source=0.15, nested=0.20, competing=0.25, d4rt=0.15, scale=1.0),
        },
        {
            "variant_id": "F3_D4RT_edge_barrier",
            "family": "real",
            "mode": "binary_pairwise",
            "threshold": 0.55,
            "area_cap": 0.84,
            "smooth": 1.05,
            "weights": dict(d4rt=1.25, sem=0.20, inside=0.24, source_bar=0.10, nested_bar=0.18, competing_bar=0.38, negative=0.65, bias=-0.52),
            "edge": dict(semantic=0.55, source=0.35, nested=0.75, competing=1.15, d4rt=0.70, scale=1.0),
        },
        {
            "variant_id": "F4_RADIO_edge_barrier",
            "family": "real",
            "mode": "binary_pairwise",
            "threshold": 0.55,
            "area_cap": 0.84,
            "smooth": 1.10,
            "weights": dict(d4rt=0.25, sem=1.30, inside=0.20, source_bar=0.08, nested_bar=0.22, competing_bar=0.34, negative=0.55, bias=-0.58),
            "edge": dict(semantic=1.35, source=0.25, nested=0.70, competing=1.00, d4rt=0.25, scale=1.0),
        },
        {
            "variant_id": "F5_D4RT_RADIO_edge_binary",
            "family": "real",
            "mode": "binary_pairwise",
            "threshold": 0.56,
            "area_cap": 0.78,
            "smooth": 1.25,
            "weights": dict(d4rt=1.05, sem=1.05, inside=0.18, source_bar=0.10, nested_bar=0.25, competing_bar=0.42, negative=0.75, bias=-0.62),
            "edge": dict(semantic=1.10, source=0.30, nested=0.95, competing=1.35, d4rt=0.55, scale=1.05),
        },
        {
            "variant_id": "F6_D4RT_RADIO_edge_multilabel_competition",
            "family": "real",
            "mode": "multilabel",
            "threshold": 0.57,
            "area_cap": 0.74,
            "smooth": 1.30,
            "weights": dict(d4rt=1.05, sem=1.05, inside=0.20, source_bar=0.10, nested_bar=0.30, competing_bar=0.55, negative=0.82, bias=-0.66),
            "edge": dict(semantic=1.05, source=0.32, nested=1.10, competing=1.55, d4rt=0.65, scale=1.12),
        },
        {
            "variant_id": "F7_F6_plus_hard_negative_veto",
            "family": "real",
            "mode": "multilabel_hard_negative",
            "threshold": 0.58,
            "area_cap": 0.70,
            "smooth": 1.32,
            "weights": dict(d4rt=1.05, sem=1.00, inside=0.20, source_bar=0.12, nested_bar=0.30, competing_bar=0.62, negative=1.05, bias=-0.70),
            "edge": dict(semantic=1.00, source=0.35, nested=1.10, competing=1.70, d4rt=0.75, scale=1.18),
            "hard_negative_veto": True,
        },
        {
            "variant_id": "F8_F7_plus_partwhole_rescue",
            "family": "real",
            "mode": "multilabel_partwhole",
            "threshold": 0.57,
            "area_cap": 0.76,
            "smooth": 1.25,
            "weights": dict(d4rt=1.00, sem=1.10, inside=0.22, source_bar=0.10, nested_bar=0.12, competing_bar=0.60, negative=0.95, bias=-0.66),
            "edge": dict(semantic=1.02, source=0.32, nested=0.45, competing=1.62, d4rt=0.70, scale=1.12),
            "hard_negative_veto": True,
            "partwhole_rescue": True,
        },
        {
            "variant_id": "C0_random_edge_density_control",
            "family": "control",
            "mode": "random_control",
            "threshold": 0.56,
            "area_cap": 0.78,
            "smooth": 1.25,
            "weights": dict(d4rt=1.05, sem=1.05, inside=0.18, source_bar=0.10, nested_bar=0.25, competing_bar=0.42, negative=0.75, bias=-0.62),
            "edge": dict(semantic=1.10, source=0.30, nested=0.95, competing=1.35, d4rt=0.55, scale=1.05),
        },
        {
            "variant_id": "C1_shuffled_edge_density_control",
            "family": "control",
            "mode": "shuffled_control",
            "threshold": 0.56,
            "area_cap": 0.78,
            "smooth": 1.25,
            "weights": dict(d4rt=1.05, sem=1.05, inside=0.18, source_bar=0.10, nested_bar=0.25, competing_bar=0.42, negative=0.75, bias=-0.62),
            "edge": dict(semantic=1.10, source=0.30, nested=0.95, competing=1.35, d4rt=0.55, scale=1.05),
        },
    ]


def _cuda_devices() -> list[str]:
    if torch is None or not torch.cuda.is_available():
        return []
    return [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]


def _choose_backend(requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    if torch is None:
        return "cpu"
    if not torch.cuda.is_available():
        return "cpu"
    if requested == "torch":
        return "torch_cuda"
    if requested == "triton" and triton is not None:
        return "triton_cuda"
    if triton is not None:
        return "triton_cuda"
    return "torch_cuda"


def _torch_device(device_index: int, backend: str) -> Any:
    if torch is None or backend == "cpu":
        return torch.device("cpu") if torch is not None else None
    count = max(1, torch.cuda.device_count())
    return torch.device(f"cuda:{device_index % count}")


def _tensor(data: Any, *, device: Any, dtype: Any) -> Any:
    tensor = torch.as_tensor(data, dtype=dtype)
    if device is not None:
        tensor = tensor.to(device)
    return tensor.contiguous()


def _compute_unary_triton(features: dict[str, Any], specs: list[dict[str, Any]], device: Any, backend: str) -> Any:
    n = int(features["sem"].numel())
    weights = {name: [] for name in ["d4rt", "sem", "inside", "source_bar", "nested_bar", "competing_bar", "negative", "bias"]}
    for spec in specs:
        raw = spec["weights"]
        for name in weights:
            weights[name].append(float(raw[name]))
    w = {name: _tensor(values, device=device, dtype=torch.float32) for name, values in weights.items()}
    out = torch.empty((len(specs), n), device=device, dtype=torch.float32)
    if backend == "triton_cuda" and triton is not None and n > 0 and out.is_cuda:
        block = 256
        grid = (len(specs), triton.cdiv(n, block))
        _unary_kernel[grid](
            features["sem"],
            features["d4rt"],
            features["inside"],
            features["source_bar"],
            features["nested_bar"],
            features["competing_bar"],
            features["negative"],
            w["d4rt"],
            w["sem"],
            w["inside"],
            w["source_bar"],
            w["nested_bar"],
            w["competing_bar"],
            w["negative"],
            w["bias"],
            out,
            n,
            block,
        )
        return out
    return (
        w["d4rt"][:, None] * features["d4rt"][None, :]
        + w["sem"][:, None] * features["sem"][None, :]
        + w["inside"][:, None] * features["inside"][None, :]
        - w["source_bar"][:, None] * features["source_bar"][None, :]
        - w["nested_bar"][:, None] * features["nested_bar"][None, :]
        - w["competing_bar"][:, None] * features["competing_bar"][None, :]
        - w["negative"][:, None] * features["negative"][None, :]
        + w["bias"][:, None]
    )


def _edge_message(prob: Any, edge_u: Any, edge_v: Any, edge_weight: Any, *, backend: str) -> tuple[Any, Any]:
    n_variants, n_nodes = prob.shape
    n_edges = int(edge_u.numel())
    if n_edges <= 0:
        neutral = torch.full_like(prob, 0.5)
        return neutral, torch.zeros_like(prob)
    accum = torch.zeros_like(prob)
    denom = torch.zeros_like(prob)
    if backend == "triton_cuda" and triton is not None and prob.is_cuda:
        block = 256
        grid = (int(n_variants), triton.cdiv(n_edges, block))
        _edge_message_kernel[grid](prob, edge_u, edge_v, edge_weight, accum, denom, int(n_nodes), n_edges, block)
    else:
        accum.index_add_(1, edge_u, edge_weight * prob[:, edge_v])
        accum.index_add_(1, edge_v, edge_weight * prob[:, edge_u])
        denom.index_add_(1, edge_u, edge_weight)
        denom.index_add_(1, edge_v, edge_weight)
    neutral = torch.full_like(prob, 0.5)
    message = torch.where(denom > 1e-6, accum / denom.clamp_min(1e-6), neutral)
    return message, denom


def _support_counts_gpu(nodes: list[dict[str, Any]], points: list[tuple[float, float]], shape: tuple[int, int], device: Any) -> list[int]:
    if not points:
        return [0 for _ in nodes]
    if torch is None or device is None:
        return v92field._support_counts(nodes, points, shape)
    h, w = shape
    x0 = _tensor([_int(node.get("bbox_x0"), 0) for node in nodes], device=device, dtype=torch.float32)
    x1 = _tensor([_int(node.get("bbox_x1"), 0) for node in nodes], device=device, dtype=torch.float32)
    y0 = _tensor([_int(node.get("bbox_y0"), 0) for node in nodes], device=device, dtype=torch.float32)
    y1 = _tensor([_int(node.get("bbox_y1"), 0) for node in nodes], device=device, dtype=torch.float32)
    pts = _tensor(points, device=device, dtype=torch.float32)
    px = pts[:, 0] * float(w - 1)
    py = pts[:, 1] * float(h - 1)
    inside = (px[None, :] >= x0[:, None]) & (px[None, :] <= x1[:, None]) & (py[None, :] >= y0[:, None]) & (py[None, :] <= y1[:, None])
    return [int(v) for v in inside.sum(dim=1).detach().cpu().tolist()]


def _load_edges(path: Path, selected_keys: set[tuple[str, int, int]]) -> dict[tuple[str, int, int], list[dict[str, Any]]]:
    edges: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (str(row.get("scene_id", "")), _int(row.get("frame_id"), -1), _int(row.get("source_mask_id"), -1))
            if key not in selected_keys:
                continue
            u = _region_index(str(row.get("region_u", row.get("region_id_a", ""))))
            v = _region_index(str(row.get("region_v", row.get("region_id_b", ""))))
            if u < 0 or v < 0:
                continue
            edges[key].append(
                {
                    "row": row,
                    "u": int(u),
                    "v": int(v),
                    "radio_cosine": _num(row.get("radio_cosine"), 0.0),
                    "dino_cosine": _num(row.get("dino_cosine"), 0.0),
                    "semantic_gradient": _num(row.get("semantic_gradient_barrier"), max(0.0, 1.0 - _num(row.get("radio_cosine"), 0.0))),
                    "rgb_gradient": _num(row.get("rgb_gradient_barrier"), 0.0),
                    "mask_edge": _num(row.get("mask_edge_barrier"), 0.0),
                    "nested_edge": _num(row.get("nested_edge_barrier"), 0.0),
                    "competing_edge": _num(row.get("competing_edge_barrier"), 0.0),
                    "d4rt_conflict": _num(row.get("d4rt_conflict_barrier"), 0.0),
                }
            )
    return dict(edges)


class ScoreWTAFrameWriter:
    def __init__(self, out: Path, variant_ids: list[str]) -> None:
        self.out = out
        self.variant_ids = variant_ids
        self.current: tuple[str, int] | None = None
        self.shape: tuple[int, int] | None = None
        self.pending: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.generated_rows: list[dict[str, Any]] = []
        self.mv_rows: list[dict[str, Any]] = []

    def ensure_frame(self, scene: str, frame_id: int, shape: tuple[int, int]) -> None:
        key = (scene, int(frame_id))
        if self.current != key:
            self.flush()
            self.current = key
            self.shape = shape
            self.pending = defaultdict(list)

    def add(self, variant_id: str, mask: np.ndarray, score: float, generated_row: dict[str, Any], mv_row: dict[str, Any]) -> None:
        if self.current is None:
            raise RuntimeError("ensure_frame must be called before add")
        if np.any(mask):
            self.pending[variant_id].append({"mask": np.asarray(mask, dtype=bool), "score": float(score), "generated_row": generated_row, "mv_row": mv_row})

    def flush(self) -> None:
        if self.current is None or self.shape is None:
            return
        scene, frame_id = self.current
        for variant_id in self.variant_ids:
            label = np.zeros(self.shape, dtype=np.uint16)
            next_id = 1
            for item in sorted(self.pending.get(variant_id, []), key=lambda row: row["score"], reverse=True):
                write = item["mask"] & (label == 0)
                if not np.any(write):
                    continue
                new_id = next_id
                next_id += 1
                label[write] = int(new_id)
                gen_row = dict(item["generated_row"])
                mv_row = dict(item["mv_row"])
                gen_row["new_mask_id"] = int(new_id)
                gen_row["generated_mask_area_after_frame_wta"] = int(np.count_nonzero(write))
                gen_row["generated_mask_area_ratio_after_frame_wta"] = float(
                    np.count_nonzero(write) / max(1, _int(gen_row.get("source_mask_area"), 1))
                )
                mv_row["mask_id"] = int(new_id)
                self.generated_rows.append(gen_row)
                self.mv_rows.append(mv_row)
            out_dir = self.out / "generated_masks" / variant_id / scene / "mask"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{int(frame_id)}.png"
            if not cv2.imwrite(str(out_path), label):
                raise RuntimeError(f"failed to write {out_path}")
        self.current = None
        self.shape = None
        self.pending = defaultdict(list)


class FieldShardWriter:
    def __init__(self, out: Path, specs: list[dict[str, Any]], shard_sources: int) -> None:
        self.root = out / "field_shards"
        self.root.mkdir(parents=True, exist_ok=True)
        self.specs = specs
        self.shard_sources = max(1, int(shard_sources))
        self.shard_index = 0
        self.records: list[dict[str, Any]] = []
        self.paths: list[dict[str, Any]] = []
        self.row_counts = Counter()
        self._reset()

    def _reset(self) -> None:
        self.source_keys: list[str] = []
        self.source_object_ids: list[str] = []
        self.region_source_idx: list[np.ndarray] = []
        self.region_index: list[np.ndarray] = []
        self.region_feature: list[np.ndarray] = []
        self.unary_source_idx: list[np.ndarray] = []
        self.unary_variant_idx: list[np.ndarray] = []
        self.unary_region_index: list[np.ndarray] = []
        self.unary_logit: list[np.ndarray] = []
        self.unary_prob: list[np.ndarray] = []
        self.unary_message: list[np.ndarray] = []
        self.assignment_label_code: list[np.ndarray] = []
        self.edge_source_idx: list[np.ndarray] = []
        self.edge_u: list[np.ndarray] = []
        self.edge_v: list[np.ndarray] = []
        self.edge_feature: list[np.ndarray] = []
        self.edge_weight: list[np.ndarray] = []

    def add_source(
        self,
        *,
        key: tuple[str, int, int],
        object_id: str,
        nodes: list[dict[str, Any]],
        features_np: dict[str, np.ndarray],
        edge_u_np: np.ndarray,
        edge_v_np: np.ndarray,
        edge_features_np: np.ndarray,
        edge_weight_np: np.ndarray,
        probs: np.ndarray,
        unary_np: np.ndarray,
        message_np: np.ndarray,
        selections: dict[str, set[int]],
    ) -> None:
        local_source_idx = len(self.source_keys)
        n = len(nodes)
        v = len(self.specs)
        region_idx = np.asarray([_int(node.get("region_index"), i) for i, node in enumerate(nodes)], dtype=np.int32)
        region_feature = np.stack(
            [
                features_np["sem"],
                features_np["d4rt"],
                features_np["inside"],
                features_np["source_bar"],
                features_np["nested_bar"],
                features_np["competing_bar"],
                features_np["negative"],
                features_np["pixel_area"],
            ],
            axis=1,
        ).astype(np.float32)
        selected = np.zeros((v, n), dtype=np.uint8)
        for spec_idx, spec in enumerate(self.specs):
            if spec["variant_id"] in selections:
                idx = [int(i) for i in selections[spec["variant_id"]] if 0 <= int(i) < n]
                if idx:
                    selected[spec_idx, np.asarray(idx, dtype=np.int64)] = 1
        label_code = np.where(selected > 0, 1, np.where(probs >= 0.40, 2, 0)).astype(np.uint8)

        self.source_keys.append(f"{key[0]}|{int(key[1])}|{int(key[2])}")
        self.source_object_ids.append(object_id)
        self.region_source_idx.append(np.full(n, local_source_idx, dtype=np.int32))
        self.region_index.append(region_idx)
        self.region_feature.append(region_feature)
        self.unary_source_idx.append(np.full(v * n, local_source_idx, dtype=np.int32))
        self.unary_variant_idx.append(np.repeat(np.arange(v, dtype=np.int16), n))
        self.unary_region_index.append(np.tile(region_idx, v).astype(np.int32))
        self.unary_logit.append(unary_np.astype(np.float32, copy=False).reshape(-1))
        self.unary_prob.append(probs.astype(np.float32, copy=False).reshape(-1))
        self.unary_message.append(message_np.astype(np.float32, copy=False).reshape(-1))
        self.assignment_label_code.append(label_code.reshape(-1))
        e = int(edge_u_np.shape[0])
        self.edge_source_idx.append(np.full(e, local_source_idx, dtype=np.int32))
        self.edge_u.append(edge_u_np.astype(np.int32, copy=False))
        self.edge_v.append(edge_v_np.astype(np.int32, copy=False))
        self.edge_feature.append(edge_features_np.astype(np.float32, copy=False))
        self.edge_weight.append(edge_weight_np.T.astype(np.float32, copy=False))

        self.row_counts["field_unary_rows"] += int(v * n)
        self.row_counts["region_assignment_rows"] += int(v * n)
        self.row_counts["field_edge_rows"] += int(e)
        if len(self.source_keys) >= self.shard_sources:
            self.flush()

    def flush(self) -> None:
        if not self.source_keys:
            return
        path = self.root / f"field_shard_{self.shard_index:04d}.npz"
        payload = {
            "source_keys": np.asarray(self.source_keys),
            "source_object_ids": np.asarray(self.source_object_ids),
            "variant_ids": np.asarray([spec["variant_id"] for spec in self.specs]),
            "region_source_idx": np.concatenate(self.region_source_idx) if self.region_source_idx else np.zeros(0, dtype=np.int32),
            "region_index": np.concatenate(self.region_index) if self.region_index else np.zeros(0, dtype=np.int32),
            "region_feature_columns": np.asarray(["sem", "d4rt", "inside", "source_bar", "nested_bar", "competing_bar", "negative", "pixel_area"]),
            "region_feature": np.concatenate(self.region_feature, axis=0) if self.region_feature else np.zeros((0, 8), dtype=np.float32),
            "unary_source_idx": np.concatenate(self.unary_source_idx) if self.unary_source_idx else np.zeros(0, dtype=np.int32),
            "unary_variant_idx": np.concatenate(self.unary_variant_idx) if self.unary_variant_idx else np.zeros(0, dtype=np.int16),
            "unary_region_index": np.concatenate(self.unary_region_index) if self.unary_region_index else np.zeros(0, dtype=np.int32),
            "unary_logit": np.concatenate(self.unary_logit) if self.unary_logit else np.zeros(0, dtype=np.float32),
            "unary_prob": np.concatenate(self.unary_prob) if self.unary_prob else np.zeros(0, dtype=np.float32),
            "unary_message": np.concatenate(self.unary_message) if self.unary_message else np.zeros(0, dtype=np.float32),
            "assignment_label_code": np.concatenate(self.assignment_label_code) if self.assignment_label_code else np.zeros(0, dtype=np.uint8),
            "edge_source_idx": np.concatenate(self.edge_source_idx) if self.edge_source_idx else np.zeros(0, dtype=np.int32),
            "edge_u": np.concatenate(self.edge_u) if self.edge_u else np.zeros(0, dtype=np.int32),
            "edge_v": np.concatenate(self.edge_v) if self.edge_v else np.zeros(0, dtype=np.int32),
            "edge_feature_columns": np.asarray(
                ["semantic_gradient", "source_bar", "nested_bar", "competing_bar", "mask_bar", "d4rt_conflict", "base_barrier", "base_edge_weight"]
            ),
            "edge_feature": np.concatenate(self.edge_feature, axis=0) if self.edge_feature else np.zeros((0, 8), dtype=np.float32),
            "edge_weight": np.concatenate(self.edge_weight, axis=0) if self.edge_weight else np.zeros((0, len(self.specs)), dtype=np.float32),
        }
        np.savez(path, **payload)
        self.paths.append(
            {
                "path": _rel(path),
                "source_count": len(self.source_keys),
                "field_unary_rows": int(payload["unary_prob"].shape[0]),
                "region_assignment_rows": int(payload["assignment_label_code"].shape[0]),
                "field_edge_rows": int(payload["edge_u"].shape[0]),
                "sha256": _sha256(path),
            }
        )
        self.shard_index += 1
        self._reset()

    def manifest(self) -> dict[str, Any]:
        self.flush()
        return {
            "schema": "stream4d_v93_phase5_binary_field_shards_v1",
            "artifact_policy": "method_field_rows_are_npz_shards; evaluator_and_metric_tables_remain_csv",
            "shard_count": len(self.paths),
            "row_counts": dict(self.row_counts),
            "shards": self.paths,
        }


def _feature_arrays(nodes: list[dict[str, Any]], low_counts: list[int], hr_counts: list[int]) -> dict[str, np.ndarray]:
    sem_raw = np.asarray([_num(row.get("source_mean_cosine"), 0.0) for row in nodes], dtype=np.float32)
    sem_span = float(np.max(sem_raw) - np.min(sem_raw)) if sem_raw.size else 0.0
    sem = (sem_raw - float(np.min(sem_raw))) / max(1e-6, sem_span) if sem_raw.size else sem_raw
    d4rt_raw = np.asarray(
        [max(0, int(low_counts[i])) + 0.75 * max(0, int(hr_counts[i])) for i, _row in enumerate(nodes)],
        dtype=np.float32,
    )
    d4rt = np.log1p(d4rt_raw)
    d4rt = d4rt / max(1e-6, float(np.max(d4rt))) if d4rt.size else d4rt
    source_dist = np.asarray([_num(row.get("source_edge_distance"), 0.0) for row in nodes], dtype=np.float32)
    nested_dist = np.asarray([_num(row.get("nested_edge_distance"), -1.0) for row in nodes], dtype=np.float32)
    competing_dist = np.asarray([_num(row.get("competing_edge_distance"), -1.0) for row in nodes], dtype=np.float32)
    source_bar = 1.0 / (1.0 + np.maximum(source_dist, 0.0))
    nested_bar = np.zeros_like(nested_dist, dtype=np.float32)
    nested_valid = nested_dist >= 0.0
    nested_bar[nested_valid] = 1.0 / (1.0 + nested_dist[nested_valid])
    competing_bar = np.zeros_like(competing_dist, dtype=np.float32)
    competing_valid = competing_dist >= 0.0
    competing_bar[competing_valid] = 1.0 / (1.0 + competing_dist[competing_valid])
    inside = 1.0 - source_bar
    hard_neg = np.asarray([max(0.0, _num(row.get("hard_negative_witness_mass"), 0.0)) for row in nodes], dtype=np.float32)
    hard_neg = hard_neg / max(1e-6, float(np.max(hard_neg))) if hard_neg.size and float(np.max(hard_neg)) > 0 else hard_neg
    boundary_without_support = np.asarray(
        [
            1.0 if _bool(row.get("boundary_token")) and int(low_counts[i]) == 0 and int(hr_counts[i]) == 0 else 0.0
            for i, row in enumerate(nodes)
        ],
        dtype=np.float32,
    )
    negative = np.maximum(hard_neg, boundary_without_support * 0.35).astype(np.float32)
    pixel_area = np.asarray([max(1, _int(row.get("pixel_count", row.get("area_px")), 1)) for row in nodes], dtype=np.float32)
    return {
        "sem_raw": sem_raw,
        "sem": sem.astype(np.float32),
        "d4rt": d4rt.astype(np.float32),
        "inside": inside.astype(np.float32),
        "source_bar": source_bar.astype(np.float32),
        "nested_bar": nested_bar.astype(np.float32),
        "competing_bar": competing_bar.astype(np.float32),
        "negative": negative.astype(np.float32),
        "pixel_area": pixel_area,
    }


def _edge_arrays(
    edges: list[dict[str, Any]],
    region_to_pos: dict[int, int],
    features_np: dict[str, np.ndarray],
    specs: list[dict[str, Any]],
    source_key: tuple[str, int, int],
    *,
    make_rows: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]], np.ndarray]:
    u_list: list[int] = []
    v_list: list[int] = []
    base_rows: list[dict[str, Any]] = []
    barrier_parts: list[tuple[float, float, float, float, float, float, float]] = []
    for edge in edges:
        if int(edge["u"]) not in region_to_pos or int(edge["v"]) not in region_to_pos:
            continue
        u = region_to_pos[int(edge["u"])]
        v = region_to_pos[int(edge["v"])]
        sem_grad = max(0.0, float(edge["semantic_gradient"]))
        source_bar = float(0.5 * (features_np["source_bar"][u] + features_np["source_bar"][v]))
        nested_bar = max(float(edge["nested_edge"]), float(0.5 * (features_np["nested_bar"][u] + features_np["nested_bar"][v])))
        competing_bar = max(float(edge["competing_edge"]), float(0.5 * (features_np["competing_bar"][u] + features_np["competing_bar"][v])))
        mask_bar = max(float(edge["mask_edge"]), source_bar, nested_bar, competing_bar)
        d4rt_conflict = max(
            float(edge["d4rt_conflict"]),
            abs(float(features_np["d4rt"][u]) - float(features_np["d4rt"][v])),
            float(0.5 * (features_np["negative"][u] + features_np["negative"][v])),
        )
        base_barrier = sem_grad + 0.25 * mask_bar + 0.25 * d4rt_conflict
        u_list.append(u)
        v_list.append(v)
        barrier_parts.append((sem_grad, source_bar, nested_bar, competing_bar, mask_bar, d4rt_conflict, base_barrier))
        base_rows.append(edge["row"])
    if not u_list:
        return (
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros((len(specs), 0), dtype=np.float32),
            [],
            np.zeros((0, 8), dtype=np.float32),
        )
    parts = np.asarray(barrier_parts, dtype=np.float32)
    edge_weight = np.zeros((len(specs), len(u_list)), dtype=np.float32)
    for spec_idx, spec in enumerate(specs):
        edge_cfg = spec["edge"]
        if float(spec.get("smooth", 0.0)) <= 0.0 or float(edge_cfg.get("scale", 0.0)) <= 0.0:
            continue
        barrier = (
            float(edge_cfg["semantic"]) * parts[:, 0]
            + float(edge_cfg["source"]) * parts[:, 1]
            + float(edge_cfg["nested"]) * parts[:, 2]
            + float(edge_cfg["competing"]) * parts[:, 3]
            + float(edge_cfg["d4rt"]) * parts[:, 5]
        )
        if spec["mode"] == "random_control":
            rng = np.random.default_rng(_stable_seed(f"edge-random:{source_key}:{spec['variant_id']}"))
            barrier = rng.permutation(barrier)
        elif spec["mode"] == "shuffled_control":
            rng = np.random.default_rng(_stable_seed(f"edge-shuffle:{source_key}:{spec['variant_id']}"))
            order = np.arange(len(barrier))
            rng.shuffle(order)
            barrier = barrier[order]
        edge_weight[spec_idx, :] = np.exp(-np.clip(float(edge_cfg["scale"]) * barrier, 0.0, 20.0)).astype(np.float32)
    edge_features = np.column_stack(
        [
            parts[:, 0],
            parts[:, 1],
            parts[:, 2],
            parts[:, 3],
            parts[:, 4],
            parts[:, 5],
            parts[:, 6],
            np.exp(-parts[:, 6]),
        ]
    ).astype(np.float32)
    field_edge_rows: list[dict[str, Any]] = []
    if make_rows:
        for row_idx, row in enumerate(base_rows):
            field_row = {
                "schema_version": "stream4d_v93_phase5_field_edge_v2_wide_variant_weights",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "scene_id": row.get("scene_id", ""),
                "split": row.get("split", "dev"),
                "window_id": row.get("window_id", ""),
                "frame_id": row.get("frame_id", ""),
                "source_mask_id": row.get("source_mask_id", ""),
                "region_u": row.get("region_u", row.get("region_id_a", "")),
                "region_v": row.get("region_v", row.get("region_id_b", "")),
                "region_index_u": int(u_list[row_idx]),
                "region_index_v": int(v_list[row_idx]),
                "is_adjacent": row.get("is_adjacent", True),
                "spatial_distance": row.get("spatial_distance", ""),
                "radio_cosine": row.get("radio_cosine", ""),
                "dino_cosine": row.get("dino_cosine", ""),
                "mask_edge_barrier": float(parts[row_idx, 4]),
                "nested_edge_barrier": float(parts[row_idx, 2]),
                "competing_edge_barrier": float(parts[row_idx, 3]),
                "semantic_gradient_barrier": float(parts[row_idx, 0]),
                "rgb_gradient_barrier": row.get("rgb_gradient_barrier", ""),
                "d4rt_conflict_barrier": float(parts[row_idx, 5]),
                "base_edge_weight": float(math.exp(-float(parts[row_idx, 6]))),
            }
            for spec_idx, spec in enumerate(specs):
                field_row[f"edge_weight_{_safe_variant(spec['variant_id'])}"] = float(edge_weight[spec_idx, row_idx])
            field_edge_rows.append(field_row)
    return np.asarray(u_list, dtype=np.int64), np.asarray(v_list, dtype=np.int64), edge_weight, field_edge_rows, edge_features


def _cap_by_area(selected: set[int], nodes: list[dict[str, Any]], max_fraction: float, scores: list[float], must_keep: set[int]) -> set[int]:
    return v92field._cap_selected_by_area(
        selected,
        nodes=nodes,
        max_fraction=float(max_fraction),
        scores=scores,
        must_keep=set(must_keep),
    )


def _select_from_prob(
    *,
    spec: dict[str, Any],
    variant_idx: int,
    probs: np.ndarray,
    nodes: list[dict[str, Any]],
    features_np: dict[str, np.ndarray],
    active_seed: set[int],
    selected_reference: dict[str, set[int]],
    source_key: tuple[str, int, int],
) -> set[int]:
    n = len(nodes)
    if n == 0:
        return set()
    if spec["mode"] == "whole":
        return set(range(n))
    if spec["mode"] == "random_control":
        ref = selected_reference.get("F5_D4RT_RADIO_edge_binary", set())
        k = max(1, len(ref))
        rng = np.random.default_rng(_stable_seed(f"select-random:{source_key}:{spec['variant_id']}"))
        selected = set(int(i) for i in rng.choice(np.arange(n), size=min(n, k), replace=False).tolist())
    elif spec["mode"] == "shuffled_control":
        p = np.asarray(probs[variant_idx], dtype=np.float32).copy()
        rng = np.random.default_rng(_stable_seed(f"select-shuffle:{source_key}:{spec['variant_id']}"))
        rng.shuffle(p)
        selected = {int(i) for i, value in enumerate(p) if value >= float(spec["threshold"])}
    else:
        selected = {int(i) for i, value in enumerate(probs[variant_idx]) if value >= float(spec["threshold"])}
        selected |= active_seed
    if spec.get("hard_negative_veto"):
        selected = {
            i
            for i in selected
            if i in active_seed or not (features_np["negative"][i] >= 0.60 and features_np["d4rt"][i] < 0.10)
        }
    if spec.get("partwhole_rescue"):
        sem = features_np["sem"]
        q80 = float(np.quantile(sem, 0.80)) if len(sem) else 1.0
        selected |= {int(i) for i, value in enumerate(features_np["nested_bar"]) if value > 0.05 and sem[i] >= q80}
    if not selected:
        selected = {int(np.argmax(probs[variant_idx]))}
    selected = _cap_by_area(
        selected,
        nodes,
        float(spec.get("area_cap", 1.0)),
        [float(v) for v in probs[variant_idx].tolist()],
        active_seed,
    )
    return selected or {int(np.argmax(probs[variant_idx]))}


def _node_mask(nodes: list[dict[str, Any]], selected: set[int], source_mask: np.ndarray) -> np.ndarray:
    return v92field._node_mask(nodes, selected, source_mask)


def _object_rows_from_mv(mv_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    out: list[dict[str, Any]] = []
    for row in mv_rows:
        key = (str(row.get("variant", "")), str(row.get("mv_object_id", "")))
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "variant_id": key[0],
                "mv_object_id": key[1],
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
    return out


def _aggregate_area(generated_rows: list[dict[str, Any]], assignment_stats: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in generated_rows:
        grouped[str(row.get("variant_id", ""))].append(row)
    out: dict[str, dict[str, float]] = {}
    for variant_id, group in grouped.items():
        ratios = [_num(row.get("generated_mask_area_ratio_after_frame_wta"), _num(row.get("generated_mask_area_ratio"))) for row in group]
        before = [_num(row.get("generated_mask_area_ratio")) for row in group]
        out[variant_id] = {
            "mean_generated_area_ratio": _mean(ratios),
            "source_area_ratio_after_readout": _mean(ratios),
            "whole_source_similarity": _mean(before),
            "object_region_count_mean": _mean(assignment_stats.get(f"{variant_id}:selected_region_count", [])),
            "region_assignment_entropy": _mean(assignment_stats.get(f"{variant_id}:entropy", [])),
            "seed_coverage_rate": _mean(assignment_stats.get(f"{variant_id}:seed_coverage", [])),
            "hard_negative_inclusion_rate": _mean(assignment_stats.get(f"{variant_id}:hard_negative_inclusion", [])),
        }
    return out


def _phase5_gate_rows(variant_metric_rows: list[dict[str, Any]], specs: list[dict[str, Any]], phase0: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    spec_by_id = {spec["variant_id"]: spec for spec in specs}
    current_controls = [row for row in variant_metric_rows if spec_by_id.get(str(row.get("variant_id", "")), {}).get("family") == "control"]
    current_best_control = max(current_controls, key=lambda row: _num(row.get("mean_MV_AP_window")), default={})
    locked_control_ap = _num(phase0.get("best_control_MV_AP_window"))
    locked_control_ap50 = _num(phase0.get("best_control_MV_AP50_window"))
    observed_control_ap = max(locked_control_ap, _num(current_best_control.get("mean_MV_AP_window"), locked_control_ap))
    observed_control_ap50 = max(locked_control_ap50, _num(current_best_control.get("mean_MV_AP50_window"), locked_control_ap50))
    gate_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    any_pass = False
    for row in variant_metric_rows:
        variant_id = str(row.get("variant_id", ""))
        spec = spec_by_id.get(variant_id, {})
        mv_ap = _num(row.get("mean_MV_AP_window"))
        mv_ap50 = _num(row.get("mean_MV_AP50_window"))
        mv_ap25 = _num(row.get("mean_MV_AP25_window"))
        is_real = spec.get("family") == "real"
        v91_gate = mv_ap >= _num(phase0.get("v91_best_MV_AP_window")) + 0.006 and mv_ap50 >= _num(phase0.get("v91_best_MV_AP50_window")) + 0.012
        control_gate = mv_ap >= observed_control_ap + 0.008 and mv_ap50 >= observed_control_ap50 + 0.012
        provenance_gate = not _bool(row.get("uses_gt_for_prediction")) and not _bool(row.get("uses_future"))
        pass_gate = bool(is_real and v91_gate and control_gate and provenance_gate)
        any_pass = any_pass or pass_gate
        gate = {
            "schema_version": "stream4d_v93_phase5_variant_gate_v2",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "variant_id": variant_id,
            "family": spec.get("family", ""),
            "MV_AP_window": mv_ap,
            "MV_AP50_window": mv_ap50,
            "MV_AP25_window": mv_ap25,
            "v91_best_MV_AP_window": _num(phase0.get("v91_best_MV_AP_window")),
            "v91_best_MV_AP50_window": _num(phase0.get("v91_best_MV_AP50_window")),
            "locked_best_control_MV_AP_window": locked_control_ap,
            "locked_best_control_MV_AP50_window": locked_control_ap50,
            "current_best_control_variant": current_best_control.get("variant_id", phase0.get("best_control_variant", "")),
            "best_control_for_gate_MV_AP_window": observed_control_ap,
            "best_control_for_gate_MV_AP50_window": observed_control_ap50,
            "required_MV_AP_window": max(_num(phase0.get("v91_best_MV_AP_window")) + 0.006, observed_control_ap + 0.008),
            "required_MV_AP50_window": max(_num(phase0.get("v91_best_MV_AP50_window")) + 0.012, observed_control_ap50 + 0.012),
            "v91_progress_gate_pass": v91_gate,
            "control_gate_pass": control_gate,
            "provenance_gate_pass": provenance_gate,
            "phase5_dev_gate_pass": pass_gate,
            "uses_gt_for_prediction": _bool(row.get("uses_gt_for_prediction")),
            "uses_future": _bool(row.get("uses_future")),
        }
        gate_rows.append(gate)
        if is_real and not pass_gate:
            area = _num(row.get("mean_generated_area_ratio"), 0.0)
            if observed_control_ap > mv_ap:
                failure_type = "CONTROL_BIAS_OR_CONTROL_STRONGER"
                repair = "Do not claim method progress; add control-resistant D4RT/RADIO residual diagnostics or move to Phase7 adaptive D4RT if support is insufficient."
            elif mv_ap25 > _num(phase0.get("v91_best_MV_AP25_window")) and not (mv_ap50 >= _num(phase0.get("v91_best_MV_AP50_window")) + 0.012):
                failure_type = "EXTENT_TOO_LOOSE_AP25_ONLY"
                repair = "Increase edge barriers, add competing-edge veto, or reduce whole-source fallback."
            elif area < 0.20:
                failure_type = "MASKS_TOO_SMALL"
                repair = "Relax seed threshold, add semantic expansion, or allow partwhole rescue."
            elif area > 0.88:
                failure_type = "WHOLE_SOURCE_LIKE_MASKS"
                repair = "Increase nested/competing edge barrier or use multi-label competition."
            else:
                failure_type = "NO_PHASE5_CONTROL_GATE_GAIN"
                repair = "Use Phase6 if multi-object competition is unsupported by binary readout; use Phase7 if D4RT witness footprint remains insufficient."
            failure_rows.append(
                {
                    "schema_version": "stream4d_v93_phase5_failure_v2",
                    "phase_id": PHASE_ID,
                    "run_id": RUN_ID,
                    "variant_id": variant_id,
                    "failure_type": failure_type,
                    "repair_direction": repair,
                    "MV_AP_window": mv_ap,
                    "MV_AP50_window": mv_ap50,
                    "MV_AP25_window": mv_ap25,
                    "mean_generated_area_ratio": area,
                    "best_control_for_gate_MV_AP_window": observed_control_ap,
                    "best_control_for_gate_MV_AP50_window": observed_control_ap50,
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                    "created_at": _created_at(),
                }
            )
    return gate_rows, failure_rows, any_pass


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    out = _resolve(args.output_root)
    if out.exists() and args.clean:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    created_at = _created_at()
    backend = _choose_backend(str(args.backend))
    devices = _cuda_devices()
    specs = _variant_specs()
    variant_ids = [spec["variant_id"] for spec in specs]
    field_edge_fields = BASE_FIELD_EDGE_FIELDS + [f"edge_weight_{_safe_variant(variant_id)}" for variant_id in variant_ids]

    source_meta = v92field._load_source_meta(_resolve(args.source_container_rows))
    links = v92field._load_links(_resolve(args.object_container_link_rows), str(args.base_source_variant))
    if args.max_sources > 0:
        kept = set(sorted(links)[: int(args.max_sources)])
        links = {key: value for key, value in links.items() if key in kept}
    selected_keys = set(links)
    low_support = v92field._load_support_points(_resolve(args.lowres_support_rows), selected_keys)
    hr_support = v92field._load_support_points(_resolve(args.highres_support_rows), selected_keys)
    edge_rows_by_key = _load_edges(_resolve(args.region_edge_rows), selected_keys)
    object_count_hist = Counter(len(value) for value in links.values())
    multi_object_source_count = sum(1 for value in links.values() if len(value) > 1)

    radius_sweep.OUT = out
    writer = ScoreWTAFrameWriter(out, variant_ids)
    assignment_stats: dict[str, list[float]] = defaultdict(list)
    failure_rows_extra: list[dict[str, Any]] = []
    device_counts: Counter[str] = Counter()
    processed_source_count = 0
    field_unary_count = 0
    region_assignment_count = 0
    field_edge_count = 0
    triton_kernel_source_count = 0

    config_rows = [
        {
            "schema_version": "stream4d_v93_phase5_variant_config_v2",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "variant_id": spec["variant_id"],
            "family": spec["family"],
            "mode": spec["mode"],
            "threshold": spec["threshold"],
            "area_cap": spec["area_cap"],
            "smooth": spec["smooth"],
            "weights_json": json.dumps(spec["weights"], sort_keys=True),
            "edge_weights_json": json.dumps(spec["edge"], sort_keys=True),
            "solver_backend_requested": str(args.backend),
            "solver_backend_actual": backend,
            "gpu_devices_visible": ";".join(devices),
            "base_source_variant": str(args.base_source_variant),
            "notes": "F6/F7/F8 run through the multi-label-capable path; current dev source universe determines whether multi-object competition is actually exercised.",
            "uses_gt_for_prediction": False,
            "uses_future": False,
            "created_at": created_at,
        }
        for spec in specs
    ]

    export_full_field_csv = bool(args.export_full_field_csv)
    field_shard_writer = FieldShardWriter(out, specs, int(args.field_shard_sources))
    field_artifact_manifest: dict[str, Any] = {}
    with ExitStack() as stack:
        if export_full_field_csv:
            field_unary_path = out / "field_unary_rows.csv"
            field_edge_path = out / "field_edge_rows.csv"
            assignment_path = out / "region_assignment_rows.csv"
            unary_handle = stack.enter_context(field_unary_path.open("w", newline="", encoding="utf-8"))
            edge_handle = stack.enter_context(field_edge_path.open("w", newline="", encoding="utf-8"))
            assignment_handle = stack.enter_context(assignment_path.open("w", newline="", encoding="utf-8"))
            unary_writer = csv.DictWriter(unary_handle, fieldnames=FIELD_UNARY_FIELDS)
            edge_writer = csv.DictWriter(edge_handle, fieldnames=field_edge_fields)
            assignment_writer = csv.DictWriter(assignment_handle, fieldnames=REGION_ASSIGNMENT_FIELDS)
            unary_writer.writeheader()
            edge_writer.writeheader()
            assignment_writer.writeheader()
        else:
            unary_writer = None
            edge_writer = None
            assignment_writer = None

        current_key: tuple[str, int, int] | None = None
        nodes_by_key: list[dict[str, Any]] = []
        label_cache: dict[tuple[str, int], np.ndarray] = {}

        def process_group(key: tuple[str, int, int] | None, nodes: list[dict[str, Any]]) -> None:
            nonlocal processed_source_count, field_unary_count, region_assignment_count, field_edge_count, triton_kernel_source_count
            if key is None or not nodes or key not in links:
                return
            scene, frame_id, source_mask_id = key
            meta = source_meta.get(key, {})
            if not meta:
                return
            mask_path = _resolve(meta.get("mask_path", ""))
            frame_key = (scene, int(frame_id))
            if frame_key not in label_cache:
                label_cache[frame_key] = v92field._read_label(mask_path)
            label = label_cache[frame_key]
            source_mask = label == int(source_mask_id)
            if not np.any(source_mask):
                return
            writer.ensure_frame(scene, int(frame_id), label.shape)
            device_index = processed_source_count % max(1, len(devices))
            device = _torch_device(device_index, backend)
            if torch is not None and backend != "cpu" and getattr(device, "type", "") == "cuda":
                torch.cuda.set_device(device)
            device_name = "cpu" if backend == "cpu" or not devices else f"cuda:{device_index % max(1, len(devices))}:{devices[device_index % max(1, len(devices))]}"
            device_counts[device_name] += 1
            low_counts = _support_counts_gpu(nodes, low_support.get(key, []), label.shape, device)
            hr_counts = _support_counts_gpu(nodes, hr_support.get(key, []), label.shape, device)
            region_to_pos = {_int(row.get("region_index"), idx): idx for idx, row in enumerate(nodes)}
            features_np = _feature_arrays(nodes, low_counts, hr_counts)
            edge_u_np, edge_v_np, edge_weight_np, field_edge_rows, edge_features_np = _edge_arrays(
                edge_rows_by_key.get(key, []),
                region_to_pos,
                features_np,
                specs,
                key,
                make_rows=export_full_field_csv,
            )
            if edge_writer is not None:
                for edge_row in field_edge_rows:
                    edge_row["solver_backend"] = backend
                    edge_row["gpu_device"] = device_name
                    edge_row["uses_gt_for_prediction"] = False
                    edge_row["uses_future"] = False
                    edge_writer.writerow({field: _jsonable(edge_row.get(field, "")) for field in field_edge_fields})
            field_edge_count += int(edge_u_np.shape[0])

            if torch is None:
                raise RuntimeError("torch is required for v93 Phase5 solver")
            features_t = {
                name: _tensor(value, device=device, dtype=torch.float32)
                for name, value in features_np.items()
                if name in {"sem", "d4rt", "inside", "source_bar", "nested_bar", "competing_bar", "negative"}
            }
            unary = _compute_unary_triton(features_t, specs, device, backend)
            edge_u = _tensor(edge_u_np, device=device, dtype=torch.long)
            edge_v = _tensor(edge_v_np, device=device, dtype=torch.long)
            edge_weight = _tensor(edge_weight_np, device=device, dtype=torch.float32)
            smooth = _tensor([float(spec["smooth"]) for spec in specs], device=device, dtype=torch.float32)[:, None]
            logits = unary.clone()
            pairwise_message = torch.full_like(logits, 0.5)
            for _ in range(int(args.label_prop_iters)):
                prob_iter = torch.sigmoid(logits)
                pairwise_message, _denom = _edge_message(prob_iter, edge_u, edge_v, edge_weight, backend=backend)
                logits = unary + smooth * (pairwise_message - 0.5)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            unary_np = unary.detach().cpu().numpy()
            message_np = pairwise_message.detach().cpu().numpy()
            if backend == "triton_cuda":
                triton_kernel_source_count += 1

            active_seed = {i for i in range(len(nodes)) if low_counts[i] > 0 or hr_counts[i] > 0}
            if not active_seed and len(nodes) > 0:
                fallback_score = features_np["sem"] + 0.25 * features_np["inside"]
                k = max(1, min(len(nodes), int(math.ceil(0.06 * len(nodes)))))
                top = np.argsort(fallback_score)[-k:]
                active_seed = {int(i) for i in top.tolist()}
            selected_reference: dict[str, set[int]] = {}
            selections: dict[str, set[int]] = {}
            for spec_idx, spec in enumerate(specs):
                selected = _select_from_prob(
                    spec=spec,
                    variant_idx=spec_idx,
                    probs=probs,
                    nodes=nodes,
                    features_np=features_np,
                    active_seed=active_seed,
                    selected_reference=selected_reference,
                    source_key=key,
                )
                selections[spec["variant_id"]] = selected
                selected_reference[spec["variant_id"]] = selected

            source_area = int(np.count_nonzero(source_mask))
            link_rows = links[key]
            object_count = len(link_rows)
            first_object = str(link_rows[0].get("object_hypothesis_id", "")) if link_rows else ""
            if object_count > 1:
                failure_rows_extra.append(
                    {
                        "schema_version": "stream4d_v93_phase5_failure_v2",
                        "phase_id": PHASE_ID,
                        "run_id": RUN_ID,
                        "variant_id": "F6_D4RT_RADIO_edge_multilabel_competition",
                        "failure_type": "MULTIOBJECT_SOURCE_WITHOUT_OBJECT_SPECIFIC_UNARY",
                        "repair_direction": "Add object-specific support region prototypes before claiming multi-label competition quality.",
                        "scene_id": scene,
                        "frame_id": int(frame_id),
                        "source_mask_id": int(source_mask_id),
                        "object_count": object_count,
                        "uses_gt_for_prediction": False,
                        "uses_future": False,
                        "created_at": created_at,
                    }
                )
            field_shard_writer.add_source(
                key=key,
                object_id=first_object,
                nodes=nodes,
                features_np=features_np,
                edge_u_np=edge_u_np,
                edge_v_np=edge_v_np,
                edge_features_np=edge_features_np,
                edge_weight_np=edge_weight_np,
                probs=probs,
                unary_np=unary_np,
                message_np=message_np,
                selections=selections,
            )
            if not export_full_field_csv:
                field_unary_count += int(len(specs) * len(nodes))
                region_assignment_count += int(len(specs) * len(nodes))

            for spec_idx, spec in enumerate(specs):
                variant_id = str(spec["variant_id"])
                selected = selections[variant_id]
                selected_prob = [float(probs[spec_idx, i]) for i in selected] if selected else [0.0]
                selected_negative = [float(features_np["negative"][i]) for i in selected] if selected else [0.0]
                selected_region_count = len(selected)
                seed_hits = len(selected & active_seed)
                entropy_p = max(1e-6, min(1.0 - 1e-6, selected_region_count / max(1, len(nodes))))
                assignment_stats[f"{variant_id}:selected_region_count"].append(float(selected_region_count))
                assignment_stats[f"{variant_id}:entropy"].append(float(-(entropy_p * math.log(entropy_p) + (1.0 - entropy_p) * math.log(1.0 - entropy_p))))
                assignment_stats[f"{variant_id}:seed_coverage"].append(float(seed_hits / max(1, len(active_seed))))
                assignment_stats[f"{variant_id}:hard_negative_inclusion"].append(float(_mean(selected_negative)))
                if unary_writer is not None and assignment_writer is not None:
                    for node_idx, node in enumerate(nodes):
                        selected_flag = node_idx in selected
                        common = {
                            "phase_id": PHASE_ID,
                            "run_id": RUN_ID,
                            "scene_id": scene,
                            "split": "dev",
                            "window_id": node.get("window_id", ""),
                            "frame_id": int(frame_id),
                            "source_mask_id": int(source_mask_id),
                            "variant_id": variant_id,
                            "region_id": node.get("region_id", ""),
                            "region_index": _int(node.get("region_index"), node_idx),
                            "p_object": float(probs[spec_idx, node_idx]),
                            "solver_backend": backend,
                            "gpu_device": device_name,
                            "uses_gt_for_prediction": False,
                            "uses_future": False,
                        }
                        unary_writer.writerow(
                            {
                                **common,
                                "schema_version": "stream4d_v93_phase5_field_unary_v2",
                                "object_hypothesis_id": first_object,
                                "unary_total_logit": float(unary_np[spec_idx, node_idx]),
                                "unary_d4rt": float(features_np["d4rt"][node_idx]),
                                "unary_semantic": float(features_np["sem"][node_idx]),
                                "edge_inside_score": float(features_np["inside"][node_idx]),
                                "source_edge_barrier_score": float(features_np["source_bar"][node_idx]),
                                "nested_edge_barrier_score": float(features_np["nested_bar"][node_idx]),
                                "competing_edge_barrier_score": float(features_np["competing_bar"][node_idx]),
                                "hard_negative_penalty": float(features_np["negative"][node_idx]),
                                "pairwise_message_score": float(message_np[spec_idx, node_idx]),
                                "selected_as_object_before_frame_wta": selected_flag,
                            }
                        )
                        field_unary_count += 1
                        assignment_writer.writerow(
                            {
                                **common,
                                "schema_version": "stream4d_v93_phase5_region_assignment_v2",
                                "object_hypothesis_id": first_object,
                                "assigned_label": "object" if selected_flag else ("unknown" if probs[spec_idx, node_idx] >= 0.40 else "background"),
                                "assigned_object_id": first_object if selected_flag else "",
                                "selected_as_object_before_frame_wta": selected_flag,
                                "multilabel_source_object_count": object_count,
                            }
                        )
                        region_assignment_count += 1

                for link in link_rows:
                    object_id = str(link.get("object_hypothesis_id", ""))
                    selected_for_object = set(selected)
                    if spec["mode"].startswith("multilabel") and object_count > 1:
                        priors = np.asarray([_num(row.get("mask_selected_score"), _num(row.get("adapter_score_raw"), 0.0)) for row in link_rows], dtype=np.float32)
                        winner = int(np.argmax(priors)) if len(priors) else 0
                        this_idx = link_rows.index(link)
                        selected_for_object = selected if this_idx == winner else set()
                    mask = _node_mask(nodes, selected_for_object, source_mask)
                    if not np.any(mask):
                        continue
                    selected_area = int(np.count_nonzero(mask))
                    area_ratio = float(selected_area / max(1, source_area))
                    base_score = _num(link.get("mask_selected_score"), _num(link.get("adapter_score_raw"), 1.0))
                    mean_prob = _mean([float(probs[spec_idx, i]) for i in selected_for_object]) if selected_for_object else 0.0
                    neg_mean = _mean([float(features_np["negative"][i]) for i in selected_for_object]) if selected_for_object else 0.0
                    score = float(base_score * (0.45 + 0.55 * mean_prob) * max(0.05, 1.0 - 0.35 * neg_mean))
                    gen_path = out / "generated_masks" / variant_id / scene / "mask" / f"{int(frame_id)}.png"
                    mv_object_id = f"{variant_id}:{object_id}"
                    generated_row = {
                        "schema_version": "stream4d_v93_phase5_generated_mask_v2",
                        "phase_id": PHASE_ID,
                        "run_id": RUN_ID,
                        "variant_id": variant_id,
                        "scene_id": scene,
                        "split": "dev",
                        "window_id": link.get("window_id", ""),
                        "frame_id": int(frame_id),
                        "source_mask_id": int(source_mask_id),
                        "new_mask_id": "",
                        "object_hypothesis_id": object_id,
                        "generated_mask_path": _rel(gen_path),
                        "source_mask_area": int(source_area),
                        "generated_mask_area_before_frame_wta": int(selected_area),
                        "generated_mask_area": int(selected_area),
                        "generated_mask_area_ratio": area_ratio,
                        "selected_region_count": int(len(selected_for_object)),
                        "total_region_count": int(len(nodes)),
                        "lowres_support_count": int(sum(low_counts)),
                        "hr2_support_count": int(sum(hr_counts)),
                        "mean_selected_probability": mean_prob,
                        "mean_selected_hard_negative": neg_mean,
                        "frame_wta_policy": "score_descending_non_overlap_final_stage",
                        "solver_backend": backend,
                        "gpu_device": device_name,
                        "uses_gt_for_prediction": False,
                        "uses_future": False,
                    }
                    mv_row = {
                        "split": "dev",
                        "scene_id": scene,
                        "source_variant": variant_id,
                        "variant": variant_id,
                        "mv_object_id": mv_object_id,
                        "frame_id": int(frame_id),
                        "mask_id": "",
                        "frame_mask_score": score,
                        "object_score": score,
                        "uses_gt_for_prediction": False,
                        "uses_future": False,
                        "uses_rgbd_pose_mesh": False,
                        "materializable": True,
                        "selection_reason": f"v93_phase5_{spec['mode']}_{backend}",
                    }
                    writer.add(variant_id, mask, score, generated_row, mv_row)
            processed_source_count += 1
            if processed_source_count % max(1, int(args.progress_every_sources)) == 0:
                print(
                    json.dumps(
                        {
                            "phase": PHASE_ID,
                            "processed_source_count": processed_source_count,
                            "field_unary_rows": field_unary_count,
                            "field_edge_rows": field_edge_count,
                            "backend": backend,
                            "device_counts": dict(device_counts),
                            "elapsed_sec": time.time() - started,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

        region_rows_path = _resolve(args.region_root) / "region_node_rows.csv"
        with region_rows_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                key = (str(row.get("scene_id", "")), _int(row.get("frame_id"), -1), _int(row.get("source_mask_id"), -1))
                if key not in selected_keys:
                    continue
                if current_key is not None and key != current_key:
                    process_group(current_key, nodes_by_key)
                    nodes_by_key = []
                current_key = key
                nodes_by_key.append(row)
        process_group(current_key, nodes_by_key)
        writer.flush()
        field_artifact_manifest = field_shard_writer.manifest()
        _write_json(out / "field_artifact_manifest.json", field_artifact_manifest)

    generated_rows = writer.generated_rows
    mv_rows = writer.mv_rows
    object_rows = _object_rows_from_mv(mv_rows)
    metric_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    for variant_id in variant_ids:
        rows = [row for row in mv_rows if row.get("variant") == variant_id]
        metrics, cases = radius_sweep._evaluate_variant(variant_id, rows)
        metric_rows.extend(metrics)
        case_rows.extend({**row, "variant_id": variant_id} for row in cases)
    aggregate_rows = phase7d._aggregate(metric_rows)
    area_by_variant = _aggregate_area(generated_rows, assignment_stats)
    variant_metric_rows: list[dict[str, Any]] = []
    for row in aggregate_rows:
        variant_id = str(row.get("variant_id", ""))
        variant_metric_rows.append(
            {
                "schema_version": "stream4d_v93_phase5_variant_metric_v2",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "variant_id": variant_id,
                "scene_id": "ALL_DEV",
                "split": "dev",
                "window_id": "ALL_WINDOWS",
                "source_artifact": _rel(out / "mv_metric_aggregate_rows.csv"),
                "created_at": created_at,
                **row,
                **area_by_variant.get(variant_id, {}),
                "uses_gt_for_prediction": _bool(row.get("uses_gt_for_prediction")),
                "uses_future": _bool(row.get("uses_future")),
            }
        )
    phase0 = _read_json(V93_PHASE0 / "summary.json")
    gate_rows, failure_rows, any_pass = _phase5_gate_rows(variant_metric_rows, specs, phase0)
    failure_rows.extend(failure_rows_extra)

    _write_csv(out / "variant_config_rows.csv", config_rows)
    _write_csv(out / "generated_mask_rows.csv", generated_rows)
    _write_csv(out / "mv_object_rows.csv", object_rows)
    _write_csv(out / "mv_object_frame_mask_rows.csv", mv_rows)
    _write_csv(out / "mv_metric_rows.csv", metric_rows)
    _write_csv(out / "mv_metric_aggregate_rows.csv", aggregate_rows)
    _write_csv(out / "variant_metric_rows.csv", variant_metric_rows)
    _write_csv(out / "variant_gate_rows.csv", gate_rows)
    _write_csv(out / "variant_failure_rows.csv", failure_rows)
    _write_csv(out / "casebook_rows.csv", case_rows)

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
    outputs = [
        out / "variant_config_rows.csv",
        out / "field_artifact_manifest.json",
        out / "generated_mask_rows.csv",
        out / "mv_object_rows.csv",
        out / "mv_object_frame_mask_rows.csv",
        out / "mv_metric_rows.csv",
        out / "mv_metric_aggregate_rows.csv",
        out / "variant_metric_rows.csv",
        out / "variant_gate_rows.csv",
        out / "variant_failure_rows.csv",
        out / "casebook_rows.csv",
    ]
    if export_full_field_csv:
        outputs.extend([out / "field_unary_rows.csv", out / "field_edge_rows.csv", out / "region_assignment_rows.csv"])
    for shard in field_artifact_manifest.get("shards", []):
        outputs.append(_resolve(str(shard.get("path", ""))))
    summary = {
        "schema": "stream4d_v93_phase5_boundary_affinity_field_summary_v2",
        "phase_id": PHASE_ID,
        "run_id": RUN_ID,
        "decision": "PASS_V93_PHASE5_DEV_GATE" if any_pass else "NO_GO_V93_PHASE5_BOUNDARY_AFFINITY_NO_CONTROL_GAIN",
        "created_at": created_at,
        "duration_sec": time.time() - started,
        "base_source_variant": str(args.base_source_variant),
        "solver_backend_requested": str(args.backend),
        "solver_backend_actual": backend,
        "method_field_artifact_mode": "full_csv_plus_npz_shards" if export_full_field_csv else "npz_shards_no_full_field_csv",
        "export_full_field_csv": bool(export_full_field_csv),
        "field_artifact_manifest": _rel(out / "field_artifact_manifest.json"),
        "field_shard_count": int(field_artifact_manifest.get("shard_count", 0)),
        "triton_available": triton is not None,
        "torch_available": torch is not None,
        "cuda_available": bool(torch is not None and torch.cuda.is_available()),
        "gpu_devices_visible": devices,
        "gpu_device_source_counts": dict(device_counts),
        "triton_kernel_source_count": triton_kernel_source_count,
        "label_prop_iters": int(args.label_prop_iters),
        "variant_count": len(specs),
        "processed_source_count": processed_source_count,
        "source_object_count_hist": dict(sorted(object_count_hist.items())),
        "multi_object_source_count": int(multi_object_source_count),
        "multi_label_exercised": bool(multi_object_source_count > 0),
        "multi_label_note": "F6/F7/F8 use the multi-label-capable path, but current dev links have one object per source so multi-object competition is not stress-tested."
        if multi_object_source_count == 0
        else "Multi-object source rows exist; see variant_failure_rows for object-specific unary coverage warnings.",
        "best_real_variant_id": best_real.get("variant_id", ""),
        "best_real_MV_AP_window": best_real.get("mean_MV_AP_window", ""),
        "best_real_MV_AP50_window": best_real.get("mean_MV_AP50_window", ""),
        "best_control_variant_id": best_control.get("variant_id", ""),
        "best_control_MV_AP_window": best_control.get("mean_MV_AP_window", ""),
        "best_control_MV_AP50_window": best_control.get("mean_MV_AP50_window", ""),
        "any_phase5_dev_gate_pass": bool(any_pass),
        "row_counts": {
            "variant_config_rows": len(config_rows),
            "field_unary_rows": field_unary_count,
            "field_edge_rows": field_edge_count,
            "region_assignment_rows": region_assignment_count,
            "generated_mask_rows": len(generated_rows),
            "mv_object_rows": len(object_rows),
            "mv_object_frame_mask_rows": len(mv_rows),
            "mv_metric_rows": len(metric_rows),
            "variant_metric_rows": len(variant_metric_rows),
            "variant_gate_rows": len(gate_rows),
            "variant_failure_rows": len(failure_rows),
            "casebook_rows": len(case_rows),
        },
        "input_artifacts": {
            _rel(path): _sha256(path)
            for path in [
                V93_PHASE0 / "summary.json",
                V93_PHASE1 / "source_container_rows.csv",
                V93_PHASE1 / "object_container_link_rows.csv",
                V93_PHASE1 / "mask_edge_hypothesis_rows.csv",
                V93_PHASE3 / "region_node_rows.csv",
                V93_PHASE3 / "region_edge_rows.csv",
                V93_PHASE4 / "summary.json",
            ]
            if path.exists()
        },
        "uses_gt_for_prediction": False,
        "uses_future": False,
    }
    _write_json(out / "summary.json", summary)
    outputs.append(out / "summary.json")
    _write_json(out / "SHA256SUMS.json", {_rel(path): _sha256(path) for path in outputs if path.exists()})
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run v93 Phase5 boundary-aware affinity field readout with CUDA/Triton graph propagation.")
    parser.add_argument("--output-root", default=str(OUT))
    parser.add_argument("--region-root", default=str(V93_PHASE3))
    parser.add_argument("--object-container-link-rows", default=str(V93_PHASE1 / "object_container_link_rows.csv"))
    parser.add_argument("--source-container-rows", default=str(V93_PHASE1 / "source_container_rows.csv"))
    parser.add_argument("--lowres-support-rows", default=str(LOWRES_SUPPORT))
    parser.add_argument("--highres-support-rows", default=str(HR2_SUPPORT))
    parser.add_argument("--region-edge-rows", default=str(V93_PHASE3 / "region_edge_rows.csv"))
    parser.add_argument("--base-source-variant", default=BASE_SOURCE_VARIANT)
    parser.add_argument("--backend", choices=["auto", "triton", "torch", "cpu"], default="triton")
    parser.add_argument("--label-prop-iters", type=int, default=5)
    parser.add_argument("--field-shard-sources", type=int, default=128)
    parser.add_argument("--export-full-field-csv", action="store_true", help="Slow audit mode: also export full region field CSV row tables.")
    parser.add_argument("--progress-every-sources", type=int, default=100)
    parser.add_argument("--max-sources", type=int, default=0, help="Debug-only cap; 0 means full dev source universe.")
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
