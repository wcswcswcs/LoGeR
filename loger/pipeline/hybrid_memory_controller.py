"""
Hybrid Memory Controller for LoGeR pipeline v2.

This module is the v2 replacement shell for the old TTT-only write
controller.  The first implementation intentionally keeps the working
control surface conservative:

* Pass 1: probe forward from committed state, collecting geometry and TTT
  write primitives without committing state.
* Pass 2: controlled forward from the same committed state.  The frame
  attention, SWA read, TTT apply, and chunk/global-attention identity hooks are
  wired through real Pi3 model sites; non-identity read controllers are still
  intentionally conservative.

That gives us the correct two-pass protocol and preserves the old BL-01 style
TTT branch/layer experiments while leaving clear extension points for stronger
read-path controllers.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .dynamic_cue_extractor import CUE_ANCHOR, CUE_DYN, CUE_OCC, CUE_UNC, CueOutput
from .geometry_backbone import (
    GeometryOutput,
    LoGeRGeometryBackbone,
    TOKEN_TYPE_PATCH,
    WriteCacheOutput,
)
from .semantic_prior_generator import PriorOutput
from .ttt_write_controller import TTTWriteController, WriteResult


@dataclass
class HybridMemoryState:
    """Committed hybrid memory state read by a chunk.

    Phase-1 stores the existing LoGeR adaptive state in ``ttt_state``.  The
    ``history`` entry inside that dict is the current SWA/local-memory carrier
    used by the model.  Explicit ``swa_state`` / ``ref_state`` fields are kept
    so callers can migrate without changing the public protocol again.
    """

    ttt_state: Optional[Dict[str, Any]] = None
    swa_state: Optional[Dict[str, Any]] = None
    ref_state: Optional[Dict[str, Any]] = None
    prev_control_summary: Optional[Dict[str, Any]] = None
    debug: Dict[str, Any] = field(default_factory=dict)

    def to_ttt_input(self) -> Optional[Dict[str, Any]]:
        return self.ttt_state

    @classmethod
    def from_ttt_state(cls, ttt_state: Optional[Dict[str, Any]]) -> "HybridMemoryState":
        return cls(ttt_state=ttt_state)


@dataclass
class HybridMemoryCacheOutput:
    """Probe/controlled memory cache.

    ``ttt_cache`` stores replay primitives, while the read-path fields hold
    compact trace records collected from real identity hook sites.
    """

    token_meta: Dict[str, Any]
    ttt_cache: Optional[WriteCacheOutput] = None
    swa_cache: Optional[Dict[str, Any]] = None
    frame_attn_cache: Optional[Dict[str, Any]] = None
    chunk_attn_cache: Optional[Dict[str, Any]] = None
    state_refs: Dict[str, Any] = field(default_factory=dict)
    update_meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProbeOutput:
    geometry: GeometryOutput
    token_meta: Dict[str, Any]
    hybrid_cache: HybridMemoryCacheOutput
    probe_trace: Dict[str, Any]
    native_provisional_state: HybridMemoryState
    debug: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HybridMemoryControlPrior:
    """Control bundle produced by Stage D for v2.

    ``P_ttt_write`` is the mature controlled path today.  Read-path fields are
    passed to real hook sites, but only identity/small single-path controllers
    should be trusted until Phase C validates them.
    """

    D_tok: Optional[torch.Tensor] = None
    R_tok: Optional[torch.Tensor] = None
    U_tok: Optional[torch.Tensor] = None
    P_ref: Optional[torch.Tensor] = None
    P_ttt_write: Optional[torch.Tensor] = None
    P_ttt_read: Optional[torch.Tensor] = None
    P_swa_read_prev: Optional[torch.Tensor] = None
    D_swa_write_tok: Optional[torch.Tensor] = None
    frame_bias_spec: Optional[Dict[str, Any]] = None
    chunk_bias_spec: Optional[Dict[str, Any]] = None
    sparse_route_mask: Optional[torch.Tensor] = None
    B_chunk_geo: Optional[float] = None
    debug: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HybridMemoryResult:
    geometry_output: GeometryOutput
    state_next: HybridMemoryState
    ttt_state_next: Optional[Dict[str, Any]]
    swa_state_next: Optional[Dict[str, Any]]
    write_result: Optional[WriteResult] = None
    control_trace: Dict[str, Any] = field(default_factory=dict)
    debug: Dict[str, Any] = field(default_factory=dict)


def _tensor_fingerprint(t: torch.Tensor) -> str:
    t_cpu = t.detach().cpu()
    sample = t_cpu.reshape(-1)[:16].float()
    payload = {
        "shape": list(t_cpu.shape),
        "dtype": str(t_cpu.dtype),
        "sum16": float(sample.sum().item()) if sample.numel() else 0.0,
        "mean16": float(sample.mean().item()) if sample.numel() else 0.0,
    }
    return json.dumps(payload, sort_keys=True)


def hybrid_state_fingerprint(state: Optional[HybridMemoryState]) -> str:
    """Small debug fingerprint for double-write checks.

    This is not a cryptographic tensor hash.  It is intentionally cheap and
    stable enough to catch "controlled pass accidentally consumed probe state"
    type bugs in logs.
    """

    if state is None or state.ttt_state is None:
        return "empty"
    h = hashlib.sha1()
    ttt = state.ttt_state
    for key in ("w0", "w1", "w2"):
        values = ttt.get(key, []) or []
        h.update(key.encode("utf-8"))
        h.update(str(len(values)).encode("utf-8"))
        for value in values:
            if value is None:
                h.update(b"none")
            elif torch.is_tensor(value):
                h.update(_tensor_fingerprint(value).encode("utf-8"))
            else:
                h.update(str(type(value)).encode("utf-8"))
    history = ttt.get("history")
    h.update(b"history")
    h.update(str(0 if history is None else len(history)).encode("utf-8"))
    return h.hexdigest()[:16]


def _build_token_meta(geo: GeometryOutput) -> Dict[str, Any]:
    token_type = geo.token_type.detach().cpu().long()
    L_tok = int(token_type.shape[0])
    T = max(int(geo.num_frames), 1)
    tokens_per_frame = max(L_tok // T, 1)
    token_frame_id = (torch.arange(L_tok, dtype=torch.long) // tokens_per_frame).clamp_max(T - 1)
    token_local_index = torch.arange(L_tok, dtype=torch.long) % tokens_per_frame
    token_order_id = torch.arange(L_tok, dtype=torch.long)
    token_valid = torch.ones(L_tok, dtype=torch.bool)
    protected_default_mask = token_type != TOKEN_TYPE_PATCH
    return {
        "patch_meta": geo.patch_meta.detach().cpu().long(),
        "token_type": token_type,
        "token_frame_id": token_frame_id,
        "token_local_index": token_local_index,
        "token_order_id": token_order_id,
        "token_valid": token_valid,
        "special_token_meta": {
            "num_special_tokens": int((token_type != TOKEN_TYPE_PATCH).sum().item()),
            "num_patch_tokens": int((token_type == TOKEN_TYPE_PATCH).sum().item()),
        },
        "protected_default_mask": protected_default_mask,
    }


def _token_from_patch_values(
    geo: GeometryOutput,
    patch_values: torch.Tensor,
    *,
    special_value: float,
) -> torch.Tensor:
    token_type = geo.token_type.detach().cpu().long()
    out = torch.full((int(token_type.shape[0]),), float(special_value), dtype=torch.float32)
    patch_mask = token_type == TOKEN_TYPE_PATCH
    n_patch = int(patch_mask.sum().item())
    vals = patch_values.detach().cpu().float().reshape(-1)
    if vals.numel() < n_patch:
        padded = torch.full((n_patch,), float(special_value), dtype=torch.float32)
        padded[: vals.numel()] = vals
        vals = padded
    out[patch_mask] = vals[:n_patch]
    return out


def _normalize01(values: torch.Tensor) -> torch.Tensor:
    values = values.detach().cpu().float()
    finite = torch.isfinite(values)
    if not bool(finite.any()):
        return torch.zeros_like(values, dtype=torch.float32)
    safe = torch.where(finite, values, torch.zeros_like(values))
    vmin = safe[finite].min()
    vmax = safe[finite].max()
    if float((vmax - vmin).abs().item()) < 1e-8:
        return torch.zeros_like(safe, dtype=torch.float32)
    return ((safe - vmin) / (vmax - vmin)).clamp(0.0, 1.0)


def _robust_quantile01(
    values: torch.Tensor,
    *,
    num_frames: int,
    qlo: float = 0.50,
    qhi: float = 0.95,
) -> torch.Tensor:
    vals = torch.nan_to_num(values.detach().cpu().float().reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    if vals.numel() == 0:
        return vals
    qlo = min(max(float(qlo), 0.0), 1.0)
    qhi = min(max(float(qhi), qlo + 1e-4), 1.0)
    if num_frames <= 0 or vals.numel() % num_frames != 0:
        lo = torch.quantile(vals, qlo)
        hi = torch.quantile(vals, qhi)
        return ((vals - lo) / (hi - lo).clamp_min(1e-6)).clamp(0.0, 1.0)
    per_frame = vals.reshape(num_frames, -1)
    lo = torch.quantile(per_frame, qlo, dim=1, keepdim=True)
    hi = torch.quantile(per_frame, qhi, dim=1, keepdim=True)
    return ((per_frame - lo) / (hi - lo).clamp_min(1e-6)).reshape(-1).clamp(0.0, 1.0)


def _centered_percentile_rank(values: torch.Tensor) -> torch.Tensor:
    vals = torch.nan_to_num(values.detach().cpu().float(), nan=0.0, posinf=1.0, neginf=0.0)
    n = int(vals.numel())
    if n <= 1:
        return torch.zeros_like(vals)
    if float((vals.max() - vals.min()).abs().item()) < 1e-8:
        return torch.zeros_like(vals)
    order = torch.argsort(vals, stable=True)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(n, dtype=torch.float32)
    return 2.0 * (ranks / float(n - 1)) - 1.0


def _frame_patch_mask(num_patch: int, num_frames: int, *, first: int = 0, last: int = 0) -> torch.Tensor:
    if num_patch <= 0 or num_frames <= 0 or num_patch % num_frames != 0:
        return torch.zeros(num_patch, dtype=torch.bool)
    patches_per_frame = num_patch // num_frames
    frame_id = (torch.arange(num_patch, dtype=torch.long) // patches_per_frame).clamp_max(num_frames - 1)
    mask = torch.zeros(num_patch, dtype=torch.bool)
    if first > 0:
        mask |= frame_id < int(first)
    if last > 0:
        mask |= frame_id >= max(num_frames - int(last), 0)
    return mask


def _high_quantile_mask(values: torch.Tensor, q: float) -> torch.Tensor:
    vals = values.detach().cpu().float().reshape(-1)
    if vals.numel() == 0:
        return torch.zeros_like(vals, dtype=torch.bool)
    q = min(max(float(q), 0.0), 1.0)
    thr = torch.quantile(vals, q)
    return vals >= thr


def _topk_per_frame(values: torch.Tensor, *, num_frames: int, frac: float) -> torch.Tensor:
    vals = values.detach().cpu().float().reshape(-1)
    if frac <= 0.0 or frac >= 1.0 or num_frames <= 0 or vals.numel() == 0:
        return vals
    if vals.numel() % num_frames != 0:
        k = max(1, int(round(vals.numel() * frac)))
        if k >= vals.numel():
            return vals
        idx = torch.topk(vals, k=k, largest=True).indices
        out = torch.zeros_like(vals)
        out[idx] = vals[idx]
        return out
    per_frame = vals.reshape(num_frames, -1)
    k = max(1, int(round(per_frame.shape[1] * frac)))
    if k >= per_frame.shape[1]:
        return vals
    out = torch.zeros_like(per_frame)
    idx = torch.topk(per_frame, k=k, dim=1, largest=True).indices
    out.scatter_(1, idx, per_frame.gather(1, idx))
    return out.reshape(-1)


def _adaptive_quantile_calibrate(
    values: torch.Tensor,
    *,
    num_frames: int,
    target_mass: float,
    tau: float,
) -> torch.Tensor:
    vals = values.detach().cpu().float().reshape(-1)
    q = float(target_mass)
    if q <= 0.0 or q >= 1.0 or vals.numel() == 0:
        return _normalize01(vals)
    tau = max(float(tau), 1e-4)
    if num_frames <= 0 or vals.numel() % num_frames != 0:
        threshold = torch.quantile(vals, max(0.0, min(1.0, 1.0 - q)))
        return torch.sigmoid((vals - threshold) / tau).clamp(0.0, 1.0)
    per_frame = vals.reshape(num_frames, -1)
    threshold = torch.quantile(per_frame, max(0.0, min(1.0, 1.0 - q)), dim=1, keepdim=True)
    return torch.sigmoid((per_frame - threshold) / tau).reshape(-1).clamp(0.0, 1.0)


def _connected_components_count(mask_2d: torch.Tensor) -> int:
    mask = mask_2d.detach().cpu().bool()
    if mask.ndim != 2 or not bool(mask.any()):
        return 0
    h, w = int(mask.shape[0]), int(mask.shape[1])
    seen = torch.zeros((h, w), dtype=torch.bool)
    count = 0
    for y in range(h):
        for x in range(w):
            if not bool(mask[y, x]) or bool(seen[y, x]):
                continue
            count += 1
            stack = [(y, x)]
            seen[y, x] = True
            while stack:
                cy, cx = stack.pop()
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and bool(mask[ny, nx]) and not bool(seen[ny, nx]):
                        seen[ny, nx] = True
                        stack.append((ny, nx))
    return count


def _fragmentation(values: torch.Tensor, *, num_frames: int, patch_grid: Tuple[int, int], thr: float = 0.5) -> float:
    vals = values.detach().cpu().float().reshape(-1)
    h, w = int(patch_grid[0]), int(patch_grid[1])
    if num_frames <= 0 or h <= 0 or w <= 0 or vals.numel() < num_frames * h * w:
        return 0.0
    cube = vals[: num_frames * h * w].reshape(num_frames, h, w)
    frags: List[float] = []
    for frame in cube:
        mask = frame > float(thr)
        active = int(mask.sum().item())
        if active <= 0:
            frags.append(0.0)
            continue
        frags.append(float(_connected_components_count(mask)) / float(active))
    return float(torch.tensor(frags, dtype=torch.float32).mean().item()) if frags else 0.0


def _frame_bias_energy(
    D_tok: torch.Tensor,
    P_ref: torch.Tensor,
    *,
    num_frames: int,
    mode: str,
) -> float:
    D = D_tok.detach().cpu().float().reshape(-1)
    ref = P_ref.detach().cpu().float().reshape(-1) if P_ref is not None else torch.zeros_like(D)
    if num_frames <= 0 or D.numel() == 0 or D.numel() % num_frames != 0:
        return 0.0
    tokens_per_frame = D.numel() // num_frames
    D = (D * (1.0 - ref.clamp(0.0, 1.0))).reshape(num_frames, tokens_per_frame)
    Dq = D[:, :, None]
    Dk = D[:, None, :]
    if mode == "key":
        keep = (1.0 - Dk).expand(-1, tokens_per_frame, -1)
    elif mode == "query":
        return 0.0
    else:
        keep = 1.0 - (1.0 - Dq) * Dk
    bias = torch.log(keep.clamp_min(1e-4))
    return float(bias.abs().mean().item()) if bias.numel() else 0.0


def _flatten_optional_patch_map(map_tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if map_tensor is None:
        return None
    return map_tensor.detach().cpu().float().reshape(-1)


def _layer_indices(layer_ids: Optional[torch.Tensor], n_layers: int, group: str) -> torch.Tensor:
    if n_layers <= 0:
        return torch.zeros((0,), dtype=torch.long)
    idx_all = torch.arange(n_layers, dtype=torch.long)
    if layer_ids is not None and int(layer_ids.numel()) == n_layers:
        ids = layer_ids.detach().cpu().long().reshape(-1)
    else:
        ids = idx_all

    if group == "l0":
        hit = torch.nonzero(ids == 0, as_tuple=False).reshape(-1)
        return hit[:1] if hit.numel() else idx_all[:1]
    if group == "l4":
        hit = torch.nonzero(ids == 4, as_tuple=False).reshape(-1)
        return hit[:1] if hit.numel() else idx_all[min(1, n_layers - 1) : min(2, n_layers)]
    if group == "all":
        return idx_all

    third = max(n_layers // 3, 1)
    if group == "shallow":
        return idx_all[:third]
    if group == "middle":
        start = third
        end = max(2 * third, start + 1)
        return idx_all[start:min(end, n_layers)]
    if group == "deep":
        return idx_all[min(2 * third, n_layers - 1):]
    return idx_all


def _flatten_layer_group_map(
    layer_maps: Optional[torch.Tensor],
    layer_ids: Optional[torch.Tensor],
    default: torch.Tensor,
    *,
    group: str,
    stat: str = "mean",
) -> torch.Tensor:
    if layer_maps is None:
        return torch.zeros_like(default)
    maps = layer_maps.detach().cpu().float()
    if maps.ndim != 4:
        return _flatten_or_default(maps, default)
    # Expected shape: [T, L, H_tok, W_tok].
    n_layers = int(maps.shape[1])
    idx = _layer_indices(layer_ids, n_layers, group)
    if idx.numel() == 0:
        return torch.zeros_like(default)
    selected = maps.index_select(1, idx)
    if stat == "var":
        reduced = selected.var(dim=1, unbiased=False)
    else:
        reduced = selected.mean(dim=1)
    return _flatten_or_default(reduced, default)


def _flatten_or_default(map_tensor: Optional[torch.Tensor], default: torch.Tensor) -> torch.Tensor:
    vals = _flatten_optional_patch_map(map_tensor)
    if vals is None or vals.numel() == 0:
        return torch.zeros_like(default)
    if vals.numel() == default.numel():
        return vals
    out = torch.zeros_like(default)
    n = min(int(vals.numel()), int(default.numel()))
    out[:n] = vals[:n]
    return out


def _flatten_cue_field_or_default(
    map_tensor: Optional[torch.Tensor],
    default: torch.Tensor,
    *,
    num_frames: int,
    patch_grid: Tuple[int, int],
) -> torch.Tensor:
    """Flatten a CueOutput pixel/grid field to the patch-token layout."""
    if map_tensor is None:
        return torch.zeros_like(default)
    vals = map_tensor.detach().cpu().float()
    if vals.numel() == default.numel():
        return vals.reshape(-1)
    h_tok, w_tok = int(patch_grid[0]), int(patch_grid[1])
    if vals.ndim == 3 and num_frames > 0 and h_tok > 0 and w_tok > 0:
        pooled = F.interpolate(
            vals.unsqueeze(1),
            size=(h_tok, w_tok),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        if pooled.numel() == default.numel():
            return pooled.reshape(-1).clamp(0.0, 1.0)
    return _flatten_or_default(vals, default).clamp(0.0, 1.0)


def _pearson_corr(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().cpu().float().reshape(-1)
    b = b.detach().cpu().float().reshape(-1)
    n = min(int(a.numel()), int(b.numel()))
    if n < 2:
        return 0.0
    a = a[:n]
    b = b[:n]
    mask = torch.isfinite(a) & torch.isfinite(b)
    if int(mask.sum().item()) < 2:
        return 0.0
    a = a[mask] - a[mask].mean()
    b = b[mask] - b[mask].mean()
    denom = (a.square().sum().sqrt() * b.square().sum().sqrt()).clamp_min(1e-8)
    return float((a * b).sum().div(denom).clamp(-1.0, 1.0).item())


def _global_centroid_metric(
    q_layers: Optional[torch.Tensor],
    k_layers: Optional[torch.Tensor],
    layer_ids: Optional[torch.Tensor],
    default: torch.Tensor,
    *,
    group: str,
    kind: str,
) -> torch.Tensor:
    if q_layers is None or k_layers is None:
        return torch.zeros_like(default)
    q_raw = q_layers.detach().cpu().float()
    k_raw = k_layers.detach().cpu().float()
    if q_raw.ndim != 5 or k_raw.ndim != 5:
        return torch.zeros_like(default)
    # Expected shape: [T, L, H_tok, W_tok, D].
    T, L, H, W, D = q_raw.shape
    if T <= 0 or L <= 0 or H <= 0 or W <= 0 or D <= 0:
        return torch.zeros_like(default)
    idx = _layer_indices(layer_ids, int(L), group)
    if idx.numel() == 0:
        return torch.zeros_like(default)
    q = F.normalize(q_raw.index_select(1, idx).reshape(T, idx.numel(), H * W, D), dim=-1)
    k = F.normalize(k_raw.index_select(1, idx).reshape(T, idx.numel(), H * W, D), dim=-1)
    q_cent = F.normalize(q.mean(dim=2), dim=-1)  # [T, Lg, D]
    k_cent = F.normalize(k.mean(dim=2), dim=-1)

    per_frame: List[torch.Tensor] = []
    for t in range(T):
        support = [s for s in range(T) if s != t]
        if not support:
            per_frame.append(torch.zeros((H * W,), dtype=torch.float32))
            continue
        if kind == "qq_low":
            cent = q_cent[support].permute(1, 0, 2)  # [Lg, S, D]
            sims = torch.einsum("lpd,lsd->lps", q[t], cent)
            raw = 1.0 - ((sims.mean(dim=(0, 2)) + 1.0) * 0.5)
        elif kind == "kk_low":
            cent = k_cent[support].permute(1, 0, 2)
            sims = torch.einsum("lpd,lsd->lps", k[t], cent)
            raw = 1.0 - ((sims.mean(dim=(0, 2)) + 1.0) * 0.5)
        elif kind == "deep_static":
            q_sim = torch.einsum("lpd,lsd->lps", q[t], q_cent[support].permute(1, 0, 2))
            k_sim = torch.einsum("lpd,lsd->lps", k[t], k_cent[support].permute(1, 0, 2))
            raw = (((q_sim.mean(dim=(0, 2)) + 1.0) * 0.5) + ((k_sim.mean(dim=(0, 2)) + 1.0) * 0.5)) * 0.5
        else:
            cent = k_cent[support].permute(1, 0, 2)
            sims = torch.einsum("lpd,lsd->lps", q[t], cent)
            raw = sims.var(dim=2, unbiased=False).mean(dim=0)
        per_frame.append(raw.reshape(-1))
    out = torch.stack(per_frame, dim=0).reshape(-1)
    return _flatten_or_default(out, default)


def _acl2_support_indices(num_frames: int, t: int, support: str) -> List[int]:
    if num_frames <= 1:
        return []
    support = support.lower()
    if support == "off246":
        offsets = (-6, -4, -2, 2, 4, 6)
        return [t + d for d in offsets if 0 <= t + d < num_frames]
    if support == "near12":
        offsets = (-2, -1, 1, 2)
        return [t + d for d in offsets if 0 <= t + d < num_frames]
    if support == "near24":
        offsets = (-4, -2, 2, 4)
        return [t + d for d in offsets if 0 <= t + d < num_frames]
    if support == "far612":
        offsets = (-12, -8, -6, 6, 8, 12)
        return [t + d for d in offsets if 0 <= t + d < num_frames]
    if support in {"past", "past_only"}:
        return list(range(0, t))
    if support in {"future", "future_only"}:
        return list(range(t + 1, num_frames))
    if support in {"full", "noovlp", "ovlp_only", "overlap_excluded"}:
        # The current HMC cue builder does not receive chunk-boundary metadata,
        # so noovlp/ovlp_only fall back to full support for this first ACL2 pass.
        return [s for s in range(num_frames) if s != t]
    return [s for s in range(num_frames) if s != t]


def _acl2_layer_indices(n_layers: int, layerwin: str) -> torch.Tensor:
    if n_layers <= 0:
        return torch.zeros((0,), dtype=torch.long)
    aliases = {
        "middle": "g6_11",
        "currentmiddle": "g6_11",
        "v4d_shallow": "g0",
        "v4d_mid": "g2_6",
        "v4d_deep": "g13_17",
        "v4d_deepvar": "g13_15",
    }
    spec = aliases.get(layerwin.lower(), layerwin.lower())
    if spec.startswith("g"):
        body = spec[1:]
        try:
            if "_" in body:
                a_str, b_str = body.split("_", 1)
                start = int(a_str)
                end = int(b_str)
            else:
                start = end = int(body)
        except ValueError:
            return torch.arange(n_layers, dtype=torch.long)
        start = max(0, min(start, n_layers - 1))
        end = max(0, min(end, n_layers - 1))
        if end < start:
            start, end = end, start
        return torch.arange(start, end + 1, dtype=torch.long)
    return torch.arange(n_layers, dtype=torch.long)


def _global_acl2_centroid_metric(
    q_layers: Optional[torch.Tensor],
    k_layers: Optional[torch.Tensor],
    default: torch.Tensor,
    *,
    basis: str,
    stat: str,
    layerwin: str,
    support: str,
) -> torch.Tensor:
    """Head-mean ACL2 global q/k cue.

    This first ACL2 implementation keeps the memory footprint of the existing
    head-mean export path.  It compares patch vectors with support-frame
    centroids for arbitrary layer windows and temporal supports; strict
    per-head/full attention-map statistics are left to the heavier cache path.
    """
    if q_layers is None or k_layers is None:
        return torch.zeros_like(default)
    q_raw = q_layers.detach().float()
    k_raw = k_layers.detach().float()
    if q_raw.ndim != 5 or k_raw.ndim != 5:
        return torch.zeros_like(default)
    T, L, H, W, D = q_raw.shape
    if T <= 0 or L <= 0 or H <= 0 or W <= 0 or D <= 0:
        return torch.zeros_like(default)
    idx = _acl2_layer_indices(int(L), layerwin)
    if idx.numel() == 0:
        return torch.zeros_like(default)

    q = F.normalize(q_raw.index_select(1, idx).reshape(T, idx.numel(), H * W, D), dim=-1)
    k = F.normalize(k_raw.index_select(1, idx).reshape(T, idx.numel(), H * W, D), dim=-1)
    q_cent = q.mean(dim=2)  # [T, Lg, D], intentionally not renormalized.
    k_cent = k.mean(dim=2)

    basis = basis.lower()
    stat = stat.lower()
    support_key = support.lower()
    full_support = support_key in {"full", "noovlp", "ovlp_only", "overlap_excluded"}

    if full_support and stat != "var" and T > 1:
        if basis == "kk":
            tgt_all = k
            cent_all = k_cent
        elif basis == "qk":
            tgt_all = q
            cent_all = k_cent
        else:
            tgt_all = q
            cent_all = q_cent
        sum_cent = cent_all.sum(dim=0, keepdim=True)
        supp_cent = (sum_cent - cent_all) / float(T - 1)  # [T, Lg, D]
        sim = (tgt_all * supp_cent.unsqueeze(2)).sum(dim=-1)  # [T, Lg, P]
        sim01 = ((sim + 1.0) * 0.5).clamp(0.0, 1.0)
        raw = sim01.mean(dim=1)
        if stat not in {"high", "mean"}:
            raw = 1.0 - raw
        return _flatten_or_default(raw.reshape(-1), default)

    per_frame: List[torch.Tensor] = []
    for t in range(T):
        supp = _acl2_support_indices(T, t, support)
        if not supp:
            per_frame.append(torch.zeros((H * W,), dtype=torch.float32))
            continue
        if basis == "kk":
            tgt = k[t]
            cent = k_cent[supp].permute(1, 0, 2)
        elif basis == "qk":
            tgt = q[t]
            cent = k_cent[supp].permute(1, 0, 2)
        else:
            tgt = q[t]
            cent = q_cent[supp].permute(1, 0, 2)
        sim = torch.einsum("lpd,lsd->lps", tgt, cent)
        sim01 = ((sim + 1.0) * 0.5).clamp(0.0, 1.0)
        if stat == "var":
            raw = sim01.var(dim=2, unbiased=False).mean(dim=0)
        elif stat in {"high", "mean"}:
            raw = sim01.mean(dim=(0, 2))
        else:
            raw = 1.0 - sim01.mean(dim=(0, 2))
        per_frame.append(raw.reshape(-1))
    out = torch.stack(per_frame, dim=0).reshape(-1)
    return _flatten_or_default(out, default)


def _acl2_global_patch(
    q_layers: Optional[torch.Tensor],
    k_layers: Optional[torch.Tensor],
    default: torch.Tensor,
    *,
    num_frames: int,
    basis: str,
    stat: str,
    layerwin: str,
    support: str,
) -> torch.Tensor:
    raw = _global_acl2_centroid_metric(
        q_layers,
        k_layers,
        default,
        basis=basis,
        stat=stat,
        layerwin=layerwin,
        support=support,
    )
    return _robust_quantile01(raw, num_frames=num_frames)


def _acl2_read_patch_from_source(
    source: str,
    q_layers: Optional[torch.Tensor],
    k_layers: Optional[torch.Tensor],
    default: torch.Tensor,
    *,
    num_frames: int,
) -> Optional[torch.Tensor]:
    src = source
    if src.startswith("acl2."):
        src = src[len("acl2."):]

    if src.startswith("gg."):
        parts = src.split(".")
        # gg.qq.low.g13_15.off246.headmean.robustq
        if len(parts) >= 7 and parts[0] == "gg" and parts[5].startswith("head") and parts[6] == "robustq":
            basis, stat, layerwin, support = parts[1], parts[2], parts[3], parts[4]
            return _acl2_global_patch(
                q_layers,
                k_layers,
                default,
                num_frames=num_frames,
                basis=basis,
                stat=stat,
                layerwin=layerwin,
                support=support,
            )

    if not src.startswith("v4d."):
        return None

    mean1 = _acl2_global_patch(
        q_layers, k_layers, default,
        num_frames=num_frames, basis="qq", stat="high", layerwin="g0", support="off246",
    )
    var1 = _acl2_global_patch(
        q_layers, k_layers, default,
        num_frames=num_frames, basis="qq", stat="var", layerwin="g0_2", support="off246",
    )
    mean2_low = _acl2_global_patch(
        q_layers, k_layers, default,
        num_frames=num_frames, basis="qq", stat="low", layerwin="g2_6", support="off246",
    )
    mean3_high = _acl2_global_patch(
        q_layers, k_layers, default,
        num_frames=num_frames, basis="kk", stat="high", layerwin="g13_17", support="off246",
    )
    var3_high = _acl2_global_patch(
        q_layers, k_layers, default,
        num_frames=num_frames, basis="qk", stat="var", layerwin="g13_15", support="off246",
    )
    var3_low = (1.0 - var3_high).clamp(0.0, 1.0)

    if src == "v4d.mean1":
        return mean1
    if src == "v4d.var1":
        return var1
    if src in {"v4d.mean2", "v4d.mean2.low"}:
        return mean2_low
    if src in {"v4d.mean3", "v4d.mean3.high", "v4d.mean3.high_as_rescue"}:
        return mean3_high
    if src in {"v4d.var3", "v4d.var3.low", "v4d.var3.low_as_rescue_conf"}:
        return var3_low
    if src == "v4d.product.m1_m2low_m3":
        raw = mean1 * mean2_low * mean3_high
    elif src == "v4d.product.v1_m2low_m3":
        raw = var1 * mean2_low * mean3_high
    elif src == "v4d.product.m2low_m3":
        raw = mean2_low * mean3_high
    elif src == "v4d.product.m1_m2low_m3_v3low":
        raw = mean1 * mean2_low * mean3_high * var3_low
    elif src == "v4d.product.m2low_rescue":
        rescue = (mean3_high * var3_low).clamp(0.0, 1.0)
        raw = mean2_low * (1.0 - 0.5 * rescue).clamp(0.0, 1.0)
    else:
        return None
    return _robust_quantile01(raw, num_frames=num_frames)


class HybridMemoryController:
    """v2 controller coordinating probe and controlled passes."""

    def __init__(
        self,
        *,
        device: str = "cuda",
        lambda_min: float = 1.0,
        lambda_max: float = 1.0,
        eta_mean_preserve: bool = False,
        eta_norm_eps: float = 1e-8,
        prior_branch_mask: str = "0",
        prior_layer_mode: str = "all",
        prior_single_layer: int = -1,
        prior_layer_branch_policy: Optional[str] = None,
        ttt_write_delta_scale: float = 1.0,
        ttt_write_delta_scales: Optional[str] = None,
        ttt_write_native_mix_scales: Optional[str] = None,
        ttt_write_prior_transform_mode: str = "none",
        ttt_write_prior_anti_scale: float = 0.0,
        ttt_write_prior_gamma: float = 1.0,
        ttt_write_replay_feature_gate_mode: str = "none",
        ttt_write_replay_feature_gate_rho: float = 0.0,
        ttt_write_replay_feature_gate_min: float = 0.5,
        ttt_write_replay_feature_gate_branch_mask: str = "all",
        ttt_write_replay_token_filter_mode: str = "none",
        ttt_write_replay_token_filter_ratio: float = 1.0,
        ttt_write_replay_token_filter_threshold: float = 1.0,
        ttt_write_replay_token_filter_scope: str = "all",
        ttt_write_replay_token_filter_branch_mask: str = "all",
        ttt_write_replay_token_filter_blend: float = 1.0,
        ttt_write_replay_token_filter_blend_mode: str = "linear",
        ttt_write_transient_delta_subtract_scale: float = 0.0,
        ttt_write_transient_delta_branch_mask: str = "0",
        ttt_write_commit_ema_alpha: float = 1.0,
        ttt_write_commit_ema_branch_mask: str = "all",
        ttt_write_native_delta_gate_mode: str = "none",
        ttt_write_native_delta_gate_min_cos: float = 0.0,
        ttt_write_native_delta_gate_fallback: float = 0.0,
        ttt_write_native_delta_gate_cap_ratio: float = 1.0,
        ttt_write_native_delta_gate_branch_mask: str = "all",
        ttt_write_commit_filter_mode: str = "none",
        ttt_write_commit_filter_risk_source: str = "d_tok",
        ttt_write_commit_filter_scope: str = "tail_overlap",
        ttt_write_commit_filter_stat: str = "mean",
        ttt_write_commit_filter_base: float = 0.0,
        ttt_write_commit_filter_gain: float = 1.0,
        ttt_write_commit_filter_min: float = 0.0,
        ttt_write_commit_filter_max: float = 1.0,
        ttt_write_commit_filter_branch_mask: str = "0",
        enable_frame_read_control: bool = False,
        enable_swa_read_control: bool = False,
        enable_ttt_apply_control: bool = False,
        enable_chunk_read_control: bool = False,
        beta_frame: float = 0.0,
        beta_swa: float = 0.0,
        beta_chunk: float = 0.0,
        swa_gate_min: float = 0.85,
        enable_swa_overlap_bias: bool = False,
        swa_overlap_bias_beta: float = 0.0,
        swa_overlap_bias_min_keep: float = 1e-4,
        swa_overlap_bias_mode: str = "pair",
        swa_overlap_bias_layer_mode: str = "last",
        swa_overlap_bias_single_layer: int = -1,
        enable_swa_overlap_source_gate: bool = False,
        swa_overlap_source_gate_rho: float = 0.0,
        swa_overlap_source_gate_min: float = 0.85,
        swa_overlap_source_gate_mode: str = "source",
        swa_overlap_source_gate_target: str = "v",
        swa_overlap_source_gate_layer_mode: str = "last",
        swa_overlap_source_gate_single_layer: int = -1,
        enable_swa_overlap_source_replace: bool = False,
        swa_overlap_source_replace_alpha: float = 0.0,
        swa_overlap_source_replace_mode: str = "union",
        swa_overlap_source_replace_target: str = "kv",
        swa_overlap_source_replace_layer_mode: str = "last",
        swa_overlap_source_replace_single_layer: int = -1,
        rho_ttt_apply: float = 0.0,
        read_layer_mode: str = "all",
        read_single_layer: int = -1,
        read_cue_source: str = "dyn",
        read_topk_frac: float = 0.0,
        frame_bias_mode: str = "pair",
        chunk_bias_mode: str = "key",
        swa_bias_mode: str = "prev_key",
        flow_model: str = "none",
        flow_pair_stride: int = 1,
        flow_fb_thr: float = 1.5,
        flow_residual_thr: float = 0.15,
        gram_layer_groups: str = "shallow,middle,deep",
        read_calib_mode: str = "none",
        read_target_mass: float = 0.06,
        read_calib_tau: float = 0.05,
        read_blend_lambda: float = 0.25,
        read_quality_mass_min: float = 0.03,
        read_quality_mass_max: float = 0.20,
        read_quality_anchor_max: float = 0.35,
        read_quality_frag_max: float = 0.15,
        beta_policy: str = "fixed",
        beta_energy_target: float = 0.0,
        beta_min: float = 0.5,
        beta_max: float = 1.5,
        read_protect_ref: bool = True,
        read_protect_static: bool = False,
        read_protection_mode: str = "none",
        read_ref_strength: float = 1.0,
        read_overlap_frames: int = 0,
        read_reset_frames: int = 1,
        read_attention_q: float = 0.90,
        read_static_anchor_thr: float = 0.6,
        read_static_dyn_thr: float = 0.3,
        hmc_write_score_source: str = "stage_d",
        hmc_write_sparse_ratio: float = 1.0,
        hmc_write_sparse_mode: str = "none",
        hmc_write_alpha: float = 0.1,
        hmc_write_min: float = 0.8,
        hmc_write_max: float = 1.2,
        ttt_apply_min_gate: float = 0.0,
        enable_swa_write_control: bool = False,
        swa_write_mode: str = "none",
        swa_write_rho: float = 0.0,
        swa_write_min_gate: float = 0.0,
        swa_write_sparse_ratio: float = 1.0,
        swa_write_layer_mode: str = "all",
        swa_write_single_layer: int = -1,
        swa_write_scope: str = "all",
        swa_write_keep_scope: str = "all",
        swa_write_score_source: str = "read",
        swa_write_cache_blend_alpha: float = 0.0,
        swa_write_cache_blend_mode: str = "dynamic",
        swa_write_cache_blend_target: str = "v",
        ttt_write_token_scope: str = "all",
        ttt_write_token_scope_floor: float = 0.0,
        fast_cue_eval: bool = False,
    ):
        self.device = device
        self.enable_frame_read_control = bool(enable_frame_read_control)
        self.enable_swa_read_control = bool(enable_swa_read_control)
        self.enable_ttt_apply_control = bool(enable_ttt_apply_control)
        self.enable_chunk_read_control = bool(enable_chunk_read_control)
        self.beta_frame = float(beta_frame)
        self.beta_swa = float(beta_swa)
        self.beta_chunk = float(beta_chunk)
        self.swa_gate_min = float(swa_gate_min)
        self.enable_swa_overlap_bias = bool(enable_swa_overlap_bias)
        self.swa_overlap_bias_beta = float(swa_overlap_bias_beta)
        self.swa_overlap_bias_min_keep = float(swa_overlap_bias_min_keep)
        self.swa_overlap_bias_mode = str(swa_overlap_bias_mode)
        self.swa_overlap_bias_layer_mode = str(swa_overlap_bias_layer_mode)
        self.swa_overlap_bias_single_layer = int(swa_overlap_bias_single_layer)
        self.enable_swa_overlap_source_gate = bool(enable_swa_overlap_source_gate)
        self.swa_overlap_source_gate_rho = float(swa_overlap_source_gate_rho)
        self.swa_overlap_source_gate_min = float(swa_overlap_source_gate_min)
        self.swa_overlap_source_gate_mode = str(swa_overlap_source_gate_mode)
        self.swa_overlap_source_gate_target = str(swa_overlap_source_gate_target)
        self.swa_overlap_source_gate_layer_mode = str(swa_overlap_source_gate_layer_mode)
        self.swa_overlap_source_gate_single_layer = int(swa_overlap_source_gate_single_layer)
        self.enable_swa_overlap_source_replace = bool(enable_swa_overlap_source_replace)
        self.swa_overlap_source_replace_alpha = float(swa_overlap_source_replace_alpha)
        self.swa_overlap_source_replace_mode = str(swa_overlap_source_replace_mode)
        self.swa_overlap_source_replace_target = str(swa_overlap_source_replace_target)
        self.swa_overlap_source_replace_layer_mode = str(swa_overlap_source_replace_layer_mode)
        self.swa_overlap_source_replace_single_layer = int(swa_overlap_source_replace_single_layer)
        self.rho_ttt_apply = float(rho_ttt_apply)
        self.read_layer_mode = str(read_layer_mode)
        self.read_single_layer = int(read_single_layer)
        self.read_cue_source = str(read_cue_source)
        self.read_topk_frac = float(read_topk_frac)
        self.frame_bias_mode = str(frame_bias_mode)
        self.chunk_bias_mode = str(chunk_bias_mode)
        self.swa_bias_mode = str(swa_bias_mode)
        self.flow_model = str(flow_model)
        self.flow_pair_stride = max(int(flow_pair_stride), 1)
        self.flow_fb_thr = float(flow_fb_thr)
        self.flow_residual_thr = float(flow_residual_thr)
        self.gram_layer_groups = str(gram_layer_groups)
        self.read_calib_mode = str(read_calib_mode)
        self.read_target_mass = float(read_target_mass)
        self.read_calib_tau = float(read_calib_tau)
        self.read_blend_lambda = float(read_blend_lambda)
        self.read_quality_mass_min = float(read_quality_mass_min)
        self.read_quality_mass_max = float(read_quality_mass_max)
        self.read_quality_anchor_max = float(read_quality_anchor_max)
        self.read_quality_frag_max = float(read_quality_frag_max)
        self.beta_policy = str(beta_policy)
        self.beta_energy_target = float(beta_energy_target)
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)
        self.read_protect_ref = bool(read_protect_ref)
        self.read_protect_static = bool(read_protect_static)
        self.read_protection_mode = str(read_protection_mode)
        self.read_ref_strength = float(read_ref_strength)
        self.read_overlap_frames = max(int(read_overlap_frames), 0)
        self.read_reset_frames = max(int(read_reset_frames), 0)
        self.read_attention_q = float(read_attention_q)
        self.read_static_anchor_thr = float(read_static_anchor_thr)
        self.read_static_dyn_thr = float(read_static_dyn_thr)
        self.hmc_write_score_source = str(hmc_write_score_source)
        self.hmc_write_sparse_ratio = float(hmc_write_sparse_ratio)
        self.hmc_write_sparse_mode = str(hmc_write_sparse_mode)
        self.hmc_write_alpha = float(hmc_write_alpha)
        self.hmc_write_min = float(hmc_write_min)
        self.hmc_write_max = float(hmc_write_max)
        self.ttt_apply_min_gate = float(ttt_apply_min_gate)
        self.enable_swa_write_control = bool(enable_swa_write_control)
        self.swa_write_mode = str(swa_write_mode)
        self.swa_write_rho = float(swa_write_rho)
        self.swa_write_min_gate = float(swa_write_min_gate)
        self.swa_write_sparse_ratio = float(swa_write_sparse_ratio)
        self.swa_write_layer_mode = str(swa_write_layer_mode)
        self.swa_write_single_layer = int(swa_write_single_layer)
        self.swa_write_scope = str(swa_write_scope)
        self.swa_write_keep_scope = str(swa_write_keep_scope)
        self.swa_write_score_source = str(swa_write_score_source)
        self.swa_write_cache_blend_alpha = float(swa_write_cache_blend_alpha)
        self.swa_write_cache_blend_mode = str(swa_write_cache_blend_mode)
        self.swa_write_cache_blend_target = str(swa_write_cache_blend_target)
        self.ttt_write_token_scope = str(ttt_write_token_scope)
        self.ttt_write_token_scope_floor = float(ttt_write_token_scope_floor)
        self.fast_cue_eval = bool(fast_cue_eval)
        self.ttt_update_controller = TTTWriteController(
            lambda_min=lambda_min,
            lambda_max=lambda_max,
            device=device,
            write_mode="semantic",
            eta_mean_preserve=eta_mean_preserve,
            eta_norm_eps=eta_norm_eps,
            prior_branch_mask=prior_branch_mask,
            prior_layer_mode=prior_layer_mode,
            prior_single_layer=prior_single_layer,
            prior_layer_branch_policy=prior_layer_branch_policy,
            update_delta_scale=ttt_write_delta_scale,
            update_delta_scales=ttt_write_delta_scales,
            update_native_mix_scales=ttt_write_native_mix_scales,
            prior_transform_mode=ttt_write_prior_transform_mode,
            prior_anti_scale=ttt_write_prior_anti_scale,
            prior_gamma=ttt_write_prior_gamma,
            update_token_scope=ttt_write_token_scope,
            update_token_scope_floor=ttt_write_token_scope_floor,
            replay_feature_gate_mode=ttt_write_replay_feature_gate_mode,
            replay_feature_gate_rho=ttt_write_replay_feature_gate_rho,
            replay_feature_gate_min=ttt_write_replay_feature_gate_min,
            replay_feature_gate_branch_mask=ttt_write_replay_feature_gate_branch_mask,
            replay_token_filter_mode=ttt_write_replay_token_filter_mode,
            replay_token_filter_ratio=ttt_write_replay_token_filter_ratio,
            replay_token_filter_threshold=ttt_write_replay_token_filter_threshold,
            replay_token_filter_scope=ttt_write_replay_token_filter_scope,
            replay_token_filter_branch_mask=ttt_write_replay_token_filter_branch_mask,
            replay_token_filter_blend=ttt_write_replay_token_filter_blend,
            replay_token_filter_blend_mode=ttt_write_replay_token_filter_blend_mode,
            transient_delta_subtract_scale=ttt_write_transient_delta_subtract_scale,
            transient_delta_branch_mask=ttt_write_transient_delta_branch_mask,
            commit_ema_alpha=ttt_write_commit_ema_alpha,
            commit_ema_branch_mask=ttt_write_commit_ema_branch_mask,
            native_delta_gate_mode=ttt_write_native_delta_gate_mode,
            native_delta_gate_min_cos=ttt_write_native_delta_gate_min_cos,
            native_delta_gate_fallback=ttt_write_native_delta_gate_fallback,
            native_delta_gate_cap_ratio=ttt_write_native_delta_gate_cap_ratio,
            native_delta_gate_branch_mask=ttt_write_native_delta_gate_branch_mask,
            commit_filter_mode=ttt_write_commit_filter_mode,
            commit_filter_risk_source=ttt_write_commit_filter_risk_source,
            commit_filter_scope=ttt_write_commit_filter_scope,
            commit_filter_stat=ttt_write_commit_filter_stat,
            commit_filter_base=ttt_write_commit_filter_base,
            commit_filter_gain=ttt_write_commit_filter_gain,
            commit_filter_min=ttt_write_commit_filter_min,
            commit_filter_max=ttt_write_commit_filter_max,
            commit_filter_branch_mask=ttt_write_commit_filter_branch_mask,
        )

    def run_probe(
        self,
        backbone: LoGeRGeometryBackbone,
        images: torch.Tensor,
        state_m: Optional[HybridMemoryState],
        *,
        collect_hybrid_trace: bool = True,
    ) -> ProbeOutput:
        committed_hash = hybrid_state_fingerprint(state_m)
        geo, write_cache = backbone.run(
            images,
            ttt_state=state_m.to_ttt_input() if state_m is not None else None,
            cache_ttt_primitives=True,
            hmc_control=self._build_model_hmc_control(None, mode="probe", identity_hooks=True),
        )
        token_meta = _build_token_meta(geo)
        provisional_ttt = self._state_from_write_cache(write_cache)
        provisional = HybridMemoryState.from_ttt_state(provisional_ttt)
        cache = HybridMemoryCacheOutput(
            token_meta=token_meta,
            ttt_cache=write_cache,
            state_refs={
                "committed_state_hash": committed_hash,
                "native_provisional_state_hash": hybrid_state_fingerprint(provisional),
            },
            update_meta={
                "collect_hybrid_trace": bool(collect_hybrid_trace),
                "implemented_trace_paths": [
                    "frame_attn_hook",
                    "swa_read_hook",
                    "ttt_apply_hook",
                    "chunk_attn_hook",
                    "ttt_cache",
                ],
                "missing_trace_paths": ["swa_dense_importance_map", "chunk_dense_dynamic_mass_map"],
            },
        )
        trace_dict = geo.hmc_trace or {}
        cache.frame_attn_cache = {"records": trace_dict.get("frame_attention", [])}
        cache.swa_cache = {"records": trace_dict.get("swa_read", [])}
        cache.chunk_attn_cache = {"records": trace_dict.get("chunk_attention", [])}
        trace = self._build_probe_trace(geo, write_cache)
        return ProbeOutput(
            geometry=geo,
            token_meta=token_meta,
            hybrid_cache=cache,
            probe_trace=trace,
            native_provisional_state=provisional,
            debug={
                "mode": "probe",
                "committed_state_hash": committed_hash,
                "native_provisional_state_hash": hybrid_state_fingerprint(provisional),
                "ttt_cache_layers": int(write_cache.num_ttt_layers),
                "token_count": int(geo.token_type.numel()),
                "read_path_traces": "real_identity_hook_sites",
                "hmc_hook_trace_counts": {
                    "frame_attention": len(trace_dict.get("frame_attention", [])),
                    "swa_read": len(trace_dict.get("swa_read", [])),
                    "ttt_apply": len(trace_dict.get("ttt_apply", [])),
                    "chunk_attention": len(trace_dict.get("chunk_attention", [])),
                },
            },
        )

    def _build_phase_e_protection_patch(
        self,
        *,
        mode: str,
        num_patch: int,
        num_frames: int,
        safe_patch: torch.Tensor,
        key_avg_patch: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        mode = str(mode or "none")
        zero = torch.zeros(num_patch, dtype=torch.bool)
        if num_patch <= 0:
            return zero.float(), {"read_protection_mode": mode}

        overlap_mask = _frame_patch_mask(
            num_patch,
            num_frames,
            first=self.read_overlap_frames,
            last=self.read_overlap_frames,
        )
        reset_mask = _frame_patch_mask(num_patch, num_frames, first=self.read_reset_frames, last=0)
        anchor_mask = safe_patch.detach().cpu().bool().reshape(-1)
        attn_mask = _high_quantile_mask(key_avg_patch, self.read_attention_q)

        if mode in {"none", "off"}:
            protect = zero
        elif mode in {"static", "anchor", "high_anchor"}:
            protect = anchor_mask
        elif mode == "overlap":
            protect = overlap_mask
        elif mode in {"reset", "ref"}:
            protect = reset_mask
        elif mode in {"attention", "attn"}:
            protect = attn_mask
        elif mode == "combined_light":
            protect = overlap_mask | anchor_mask | attn_mask
        elif mode == "combined_strong":
            protect = overlap_mask | reset_mask | anchor_mask | attn_mask
        else:
            protect = zero

        debug = {
            "read_protection_mode": mode,
            "read_ref_strength": float(self.read_ref_strength),
            "read_overlap_frames": int(self.read_overlap_frames),
            "read_reset_frames": int(self.read_reset_frames),
            "read_attention_q": float(self.read_attention_q),
            "protect_overlap_count": int(overlap_mask.sum().item()),
            "protect_reset_count": int(reset_mask.sum().item()),
            "protect_anchor_count": int(anchor_mask.sum().item()),
            "protect_attention_count": int(attn_mask.sum().item()),
            "protect_patch_count": int(protect.sum().item()),
            "protect_patch_mass": float(protect.float().mean().item()) if protect.numel() else 0.0,
        }
        return protect.float(), debug

    def _ttt_residual_token_score(self, probe: ProbeOutput, L_tok: int) -> torch.Tensor:
        write_cache = probe.hybrid_cache.ttt_cache
        if write_cache is None or not write_cache.layer_caches:
            return torch.zeros(L_tok, dtype=torch.float32)

        accum = torch.zeros(L_tok, dtype=torch.float32)
        count = torch.zeros(L_tok, dtype=torch.float32)
        for layer_cache in write_cache.layer_caches:
            y = getattr(layer_cache, "apply_output_raw", None)
            v = getattr(layer_cache, "v", None)
            if y is None or v is None:
                continue
            y_cpu = y.detach().cpu().float()
            v_cpu = v.detach().cpu().float()
            if y_cpu.shape != v_cpu.shape or y_cpu.ndim < 2:
                continue
            res = (y_cpu - v_cpu).norm(dim=-1) / v_cpu.norm(dim=-1).clamp_min(1e-6)
            if res.ndim == 2:
                per_tok = res.mean(dim=0)
            else:
                per_tok = res.reshape(-1, res.shape[-1]).mean(dim=0)
            n = min(int(per_tok.numel()), L_tok)
            if n <= 0:
                continue
            accum[:n] += per_tok[:n]
            count[:n] += 1.0
        return torch.where(count > 0, accum / count.clamp_min(1.0), torch.zeros_like(accum))

    def _phase_e_write_prior(
        self,
        *,
        probe: ProbeOutput,
        token_type: torch.Tensor,
        dyn_patch: Optional[torch.Tensor],
        explicit_dyn_patch: Optional[torch.Tensor],
        occ_patch: Optional[torch.Tensor],
        unc_patch: Optional[torch.Tensor],
        read_patch: Optional[torch.Tensor],
        P_ref: torch.Tensor,
        base_write_prior: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
        source = str(self.hmc_write_score_source)
        if source in {"stage_d", "prior", "bl01"}:
            return None, {"hmc_write_score_source": source, "hmc_write_override": False}

        L_tok = int(token_type.numel())
        patch_mask = token_type == TOKEN_TYPE_PATCH
        num_patch = int(patch_mask.sum().item())
        if num_patch <= 0:
            return torch.ones(L_tok, dtype=torch.float32), {"hmc_write_score_source": source, "hmc_write_override": True}

        needs_residual = source in {
            "ttt_residual",
            "residual",
            "ttt_residual_reliable",
            "residual_reliable",
            "residual_reliability",
            "alignment_confidence",
            "alignment",
        }
        if needs_residual:
            residual_tok = self._ttt_residual_token_score(probe, L_tok)
            residual_patch = residual_tok[patch_mask].detach().cpu().float()
        else:
            residual_patch = torch.zeros(num_patch, dtype=torch.float32)

        def _patch_or(default: float, tensor: Optional[torch.Tensor]) -> torch.Tensor:
            if tensor is None:
                return torch.full((num_patch,), float(default), dtype=torch.float32)
            vals = tensor.detach().cpu().float().reshape(-1)
            if vals.numel() < num_patch:
                padded = torch.full((num_patch,), float(default), dtype=torch.float32)
                padded[: vals.numel()] = vals
                return padded
            return vals[:num_patch]

        dyn = _patch_or(0.0, dyn_patch).clamp(0.0, 1.0)
        exp_dyn = _patch_or(0.0, explicit_dyn_patch).clamp(0.0, 1.0)
        occ = _patch_or(0.0, occ_patch).clamp(0.0, 1.0)
        unc = _patch_or(0.0, unc_patch).clamp(0.0, 1.0)
        read = _patch_or(0.0, read_patch).clamp(0.0, 1.0)
        pref = P_ref[patch_mask].detach().cpu().float().clamp(0.0, 1.0)
        reliability = ((1.0 - read) * (1.0 - occ) * (1.0 - unc) * (1.0 - pref)).clamp(0.0, 1.0)
        if base_write_prior is not None:
            base_write = base_write_prior.detach().cpu().float().reshape(-1)
            if base_write.numel() >= L_tok:
                base_patch = base_write[:L_tok][patch_mask].clamp(0.0, 2.0)
            else:
                padded = torch.ones(L_tok, dtype=torch.float32)
                padded[: base_write.numel()] = base_write
                base_patch = padded[patch_mask].clamp(0.0, 2.0)
        else:
            base_patch = torch.ones(num_patch, dtype=torch.float32)
        base_score = _normalize01(base_patch)

        if source in {"dyn", "old_dyn"}:
            raw_score = (1.0 - dyn).clamp(0.0, 1.0)
        elif source in {"explicit_dyn_inv", "exp_dyn_inv", "v5_exp_dyn_inv"}:
            raw_score = (1.0 - exp_dyn).clamp(0.0, 1.0)
        elif source in {"dg_inv", "dg_locked_inv", "read_inv", "read_inverse", "v5_dg_inv"}:
            raw_score = (1.0 - read).clamp(0.0, 1.0)
        elif source in {"dg_inv_sqrt", "v5_dg_inv_sqrt"}:
            raw_score = torch.sqrt((1.0 - read).clamp(0.0, 1.0))
        elif source in {"dg_inv_sq", "v5_dg_inv_sq"}:
            raw_score = (1.0 - read).clamp(0.0, 1.0).square()
        elif source in {"stage_d_x_dg_inv", "v5_stage_d_x_dg_inv"}:
            raw_score = (base_score * (1.0 - read).clamp(0.0, 1.0)).clamp(0.0, 1.0)
        elif source in {"stage_d_x_dg_inv_sqrt", "v5_stage_d_x_dg_inv_sqrt"}:
            raw_score = (base_score * torch.sqrt((1.0 - read).clamp(0.0, 1.0))).clamp(0.0, 1.0)
        elif source in {"stage_d_x_dg_high_inv", "v5_stage_d_x_dg_high_inv"}:
            high_read = ((read - 0.5) * 2.0).clamp(0.0, 1.0)
            raw_score = (base_score * (1.0 - high_read)).clamp(0.0, 1.0)
        elif source in {"stage_d_x_dg_high_inv_sqrt", "v5_stage_d_x_dg_high_inv_sqrt"}:
            high_read = ((read - 0.5) * 2.0).clamp(0.0, 1.0)
            raw_score = (base_score * torch.sqrt((1.0 - high_read).clamp(0.0, 1.0))).clamp(0.0, 1.0)
        elif source in {"stage_d_x_exp_inv", "v5_stage_d_x_exp_inv"}:
            raw_score = (base_score * (1.0 - exp_dyn).clamp(0.0, 1.0)).clamp(0.0, 1.0)
        elif source in {"stage_d_x_exp_inv_sqrt", "v5_stage_d_x_exp_inv_sqrt"}:
            raw_score = (base_score * torch.sqrt((1.0 - exp_dyn).clamp(0.0, 1.0))).clamp(0.0, 1.0)
        elif source in {"stage_d_x_exp_inv_sq", "v5_stage_d_x_exp_inv_sq", "stage_d_x_exp_focal2"}:
            exp_static = (1.0 - exp_dyn).clamp(0.0, 1.0)
            raw_score = (base_score * exp_static.square()).clamp(0.0, 1.0)
        elif source in {"stage_d_x_dg_exp_inv_sqrt", "v5_stage_d_x_dg_exp_inv_sqrt"}:
            dg_static = torch.sqrt((1.0 - read).clamp(0.0, 1.0))
            exp_static = torch.sqrt((1.0 - exp_dyn).clamp(0.0, 1.0))
            raw_score = (base_score * dg_static * exp_static).clamp(0.0, 1.0)
        elif source in {"stage_d_x_dg_exp_inter_inv", "v5_stage_d_x_dg_exp_inter_inv"}:
            consensus_dyn = (read * exp_dyn).clamp(0.0, 1.0)
            raw_score = (base_score * (1.0 - consensus_dyn)).clamp(0.0, 1.0)
        elif source in {"stage_d_x_dg_exp_inter_inv_sqrt", "v5_stage_d_x_dg_exp_inter_inv_sqrt"}:
            consensus_dyn = (read * exp_dyn).clamp(0.0, 1.0)
            raw_score = (base_score * torch.sqrt((1.0 - consensus_dyn).clamp(0.0, 1.0))).clamp(0.0, 1.0)
        elif source in {"stage_d_x_union_dyn_inv", "v5_stage_d_x_union_dyn_inv"}:
            union_dyn = torch.maximum(read, exp_dyn).clamp(0.0, 1.0)
            raw_score = (base_score * (1.0 - union_dyn)).clamp(0.0, 1.0)
        elif source in {"stage_d_x_dg_inv_sq", "v5_stage_d_x_dg_inv_sq", "stage_d_x_static_focal2"}:
            static = (1.0 - read).clamp(0.0, 1.0)
            raw_score = (base_score * static.square()).clamp(0.0, 1.0)
        elif source in {"stage_d_x_dg_inv_pow4", "v5_stage_d_x_dg_inv_pow4", "stage_d_x_static_focal4"}:
            static = (1.0 - read).clamp(0.0, 1.0)
            raw_score = (base_score * static.square().square()).clamp(0.0, 1.0)
        elif source in {"stage_d_x_dg_boundary_focal2", "v5_stage_d_x_dg_boundary_focal2"}:
            static = (1.0 - read).clamp(0.0, 1.0)
            dynamic = read.clamp(0.0, 1.0)
            raw_score = (base_score * static * dynamic.square()).clamp(0.0, 1.0)
        elif source in {"ttt_residual", "residual"}:
            raw_score = _normalize01(residual_patch)
        elif source in {"ttt_residual_reliable", "residual_reliable", "residual_reliability"}:
            raw_score = _normalize01(residual_patch) * reliability
        elif source in {"alignment_confidence", "alignment"}:
            raw_score = _normalize01(1.0 - _normalize01(residual_patch)) * reliability
        else:
            raw_score = (1.0 - dyn).clamp(0.0, 1.0)

        raw_score = torch.nan_to_num(raw_score.float(), nan=0.0, posinf=1.0, neginf=0.0)
        lo = min(float(self.hmc_write_min), float(self.hmc_write_max))
        hi = max(float(self.hmc_write_min), float(self.hmc_write_max))
        centered = _centered_percentile_rank(raw_score)
        p_patch = (1.0 + float(self.hmc_write_alpha) * centered).clamp(lo, hi)

        sparse_ratio = min(max(float(self.hmc_write_sparse_ratio), 0.0), 1.0)
        sparse_selected = torch.ones(num_patch, dtype=torch.bool)
        if sparse_ratio < 1.0:
            k = max(1, int(round(num_patch * sparse_ratio)))
            k = min(k, num_patch)
            sparse_selected = torch.zeros(num_patch, dtype=torch.bool)
            if k > 0:
                idx = torch.topk(raw_score, k=k, largest=True).indices
                sparse_selected[idx] = True
            if self.hmc_write_sparse_mode in {"hard", "exact", "exact_preserve", "none"}:
                p_patch = sparse_selected.float()
            else:
                p_patch = p_patch * sparse_selected.float()

        A_tok = torch.ones(L_tok, dtype=torch.float32)
        A_tok[patch_mask] = p_patch

        debug = {
            "hmc_write_score_source": source,
            "hmc_write_override": True,
            "hmc_write_alpha": float(self.hmc_write_alpha),
            "hmc_write_min": lo,
            "hmc_write_max": hi,
            "hmc_write_sparse_ratio": sparse_ratio,
            "hmc_write_sparse_mode": self.hmc_write_sparse_mode,
            "hmc_write_selected_mass": float(sparse_selected.float().mean().item()) if sparse_selected.numel() else 1.0,
            "hmc_write_score_mean": float(raw_score.mean().item()) if raw_score.numel() else 0.0,
            "hmc_write_score_q90": float(torch.quantile(raw_score, 0.9).item()) if raw_score.numel() else 0.0,
            "hmc_write_prior_mean_patch": float(p_patch.mean().item()) if p_patch.numel() else 1.0,
            "hmc_write_residual_mean_patch": float(residual_patch.mean().item()) if residual_patch.numel() else 0.0,
            "hmc_write_residual_q90_patch": float(torch.quantile(residual_patch, 0.9).item()) if residual_patch.numel() else 0.0,
            "hmc_write_reliability_mean_patch": float(reliability.mean().item()) if reliability.numel() else 1.0,
            "hmc_write_corr_score_dyn": _pearson_corr(raw_score, dyn),
            "hmc_write_corr_score_exp_dyn": _pearson_corr(raw_score, exp_dyn),
            "hmc_write_corr_score_unc": _pearson_corr(raw_score, unc),
            "hmc_write_corr_score_residual": _pearson_corr(raw_score, residual_patch),
        }
        return A_tok, debug

    def _apply_swa_history_write_gate(
        self,
        history: Optional[List[Optional[Dict[str, torch.Tensor]]]],
        control_prior: Optional[HybridMemoryControlPrior],
        geo: GeometryOutput,
    ) -> Tuple[Optional[List[Optional[Dict[str, torch.Tensor]]]], Dict[str, Any]]:
        """Gate SWA KV history before it is committed for the next chunk."""
        mode = str(self.swa_write_mode)
        keep_scope = str(self.swa_write_keep_scope or "all")
        wants_gate = mode not in {"", "none", "off"}
        wants_keep = keep_scope not in {"", "all", "full"}
        cache_blend_alpha = max(float(self.swa_write_cache_blend_alpha), 0.0)
        wants_cache_blend = cache_blend_alpha > 0.0
        debug: Dict[str, Any] = {
            "swa_write_enabled": bool(self.enable_swa_write_control),
            "swa_write_mode": mode,
            "swa_write_score_source": str(self.swa_write_score_source),
            "swa_write_rho": float(self.swa_write_rho),
            "swa_write_min_gate": float(self.swa_write_min_gate),
            "swa_write_sparse_ratio": float(self.swa_write_sparse_ratio),
            "swa_write_layer_mode": str(self.swa_write_layer_mode),
            "swa_write_single_layer": int(self.swa_write_single_layer),
            "swa_write_scope": str(self.swa_write_scope),
            "swa_write_keep_scope": keep_scope,
            "swa_write_cache_blend_alpha": cache_blend_alpha,
            "swa_write_cache_blend_mode": str(self.swa_write_cache_blend_mode),
            "swa_write_cache_blend_target": str(self.swa_write_cache_blend_target),
            "swa_write_cache_blend_applied_layers": 0,
            "swa_write_applied_layers": 0,
            "swa_write_history_tokens_before": 0,
            "swa_write_history_tokens_after": 0,
        }
        if (
            not self.enable_swa_write_control
            or (not wants_gate and not wants_keep and not wants_cache_blend)
            or history is None
            or control_prior is None
            or control_prior.D_tok is None
        ):
            return history, debug

        token_type = geo.token_type.detach().cpu().long()
        patch_mask = token_type == TOKEN_TYPE_PATCH
        source_tok = control_prior.D_swa_write_tok if control_prior.D_swa_write_tok is not None else control_prior.D_tok
        D_patch = source_tok.detach().cpu().float().reshape(-1)[patch_mask].clamp(0.0, 1.0)
        if D_patch.numel() == 0:
            return history, debug

        rho = max(float(self.swa_write_rho), 0.0)
        min_gate = min(max(float(self.swa_write_min_gate), 0.0), 1.0)
        per_frame = 0
        overlap_frames = max(int(self.read_overlap_frames), 0)
        if int(geo.num_frames) > 0 and D_patch.numel() % int(geo.num_frames) == 0:
            per_frame = int(D_patch.numel() // int(geo.num_frames))

        def _scope_mask(scope_name: str) -> torch.Tensor:
            scope_norm = str(scope_name or "all")
            exclude = False
            if scope_norm.startswith("exclude_"):
                exclude = True
                scope_norm = scope_norm[len("exclude_"):]
            mask = torch.ones_like(D_patch, dtype=torch.bool)
            if scope_norm in {"all", "full", ""}:
                return ~mask if exclude else mask
            if per_frame <= 0 or overlap_frames <= 0:
                empty = torch.zeros_like(D_patch, dtype=torch.bool)
                return ~empty if exclude else empty
            n = min(int(D_patch.numel()), overlap_frames * per_frame)
            mask = torch.zeros_like(D_patch, dtype=torch.bool)
            if scope_norm in {"tail_overlap", "overlap_tail", "tail"}:
                mask[-n:] = True
            elif scope_norm in {"head_overlap", "overlap_head", "head"}:
                mask[:n] = True
            elif scope_norm in {"both_overlap", "overlap_both"}:
                mask[:n] = True
                mask[-n:] = True
            else:
                raise ValueError(f"Unsupported SWA write scope: {scope_name}")
            return ~mask if exclude else mask

        scope_mask = _scope_mask(str(self.swa_write_scope or "all"))
        keep_mask = _scope_mask(keep_scope)
        D_scope = D_patch[scope_mask]
        if wants_gate and D_scope.numel() == 0:
            return history, debug
        if wants_gate:
            if mode in {"v_centered", "kv_centered", "v_resid_centered", "kv_resid_centered"}:
                static_rank = _centered_percentile_rank((1.0 - D_scope).clamp(0.0, 1.0))
                max_gate = max(1.0, 2.0 - min_gate)
                gate_scope = (1.0 + rho * static_rank).clamp(min_gate, max_gate)
            else:
                gate_scope = (1.0 - rho * D_scope).clamp(min_gate, 1.0)
        else:
            gate_scope = torch.ones_like(D_scope)
        blend_mode = str(self.swa_write_cache_blend_mode or "dynamic").strip().lower()
        if blend_mode in {"dynamic", "dyn", "score", "d"}:
            blend_score_scope = D_scope.clamp(0.0, 1.0)
        elif blend_mode in {"static", "inv", "inverse"}:
            blend_score_scope = (1.0 - D_scope).clamp(0.0, 1.0)
        elif blend_mode in {"all", "flat", "constant"}:
            blend_score_scope = torch.ones_like(D_scope)
        else:
            raise ValueError(f"Unsupported SWA cache blend mode: {self.swa_write_cache_blend_mode}")
        sparse_ratio = min(max(float(self.swa_write_sparse_ratio), 0.0), 1.0)
        selected_mass = 1.0
        if wants_gate and sparse_ratio < 1.0:
            k = max(1, int(round(D_scope.numel() * sparse_ratio)))
            k = min(k, int(D_scope.numel()))
            static_score = (1.0 - D_scope).clamp(0.0, 1.0)
            selected = torch.zeros_like(static_score, dtype=torch.bool)
            selected[torch.topk(static_score, k=k, largest=True).indices] = True
            selected_mass = float(selected.float().mean().item())
            gate_scope = torch.where(selected, gate_scope, torch.full_like(gate_scope, min_gate))
        base_gate = torch.ones_like(D_patch, dtype=torch.float32)
        base_gate[scope_mask] = gate_scope
        base_blend = torch.zeros_like(D_patch, dtype=torch.float32)
        if wants_cache_blend:
            base_blend[scope_mask] = (cache_blend_alpha * blend_score_scope).clamp(0.0, min(cache_blend_alpha, 1.0))

        def _aligned_gate(num_tokens: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
            gate = base_gate
            if gate.numel() < num_tokens:
                prefix = torch.ones(num_tokens - gate.numel(), dtype=torch.float32)
                gate = torch.cat([prefix, gate], dim=0)
            elif gate.numel() > num_tokens:
                gate = gate[-num_tokens:]
            return gate.reshape(1, 1, num_tokens, 1).to(device=device, dtype=dtype)

        def _aligned_scope_mask(num_tokens: int, *, device: torch.device) -> torch.Tensor:
            mask = scope_mask
            if mask.numel() < num_tokens:
                prefix = torch.zeros(num_tokens - mask.numel(), dtype=torch.bool)
                mask = torch.cat([prefix, mask], dim=0)
            elif mask.numel() > num_tokens:
                mask = mask[-num_tokens:]
            return mask.to(device=device, dtype=torch.bool)

        def _aligned_blend(num_tokens: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
            blend = base_blend
            if blend.numel() < num_tokens:
                prefix = torch.zeros(num_tokens - blend.numel(), dtype=torch.float32)
                blend = torch.cat([prefix, blend], dim=0)
            elif blend.numel() > num_tokens:
                blend = blend[-num_tokens:]
            return blend.reshape(1, 1, num_tokens, 1).to(device=device, dtype=dtype)

        def _aligned_keep_indices(num_tokens: int, *, device: torch.device) -> Optional[torch.Tensor]:
            if not wants_keep:
                return None
            mask = keep_mask
            if mask.numel() < num_tokens:
                prefix = torch.ones(num_tokens - mask.numel(), dtype=torch.bool)
                mask = torch.cat([prefix, mask], dim=0)
            elif mask.numel() > num_tokens:
                mask = mask[-num_tokens:]
            idx = torch.nonzero(mask, as_tuple=False).reshape(-1)
            if idx.numel() == 0:
                idx = torch.arange(num_tokens, dtype=torch.long)
            return idx.to(device=device, dtype=torch.long)

        gated_history: List[Optional[Dict[str, torch.Tensor]]] = []
        gate_means: List[float] = []
        gate_p10: List[float] = []
        gate_p50: List[float] = []
        gate_p90: List[float] = []
        applied = 0
        cache_blend_applied = 0
        tokens_before: List[int] = []
        tokens_after: List[int] = []
        n_layers = len(history)
        for layer_idx, item in enumerate(history):
            if not self._swa_write_layer_enabled(layer_idx, n_layers):
                gated_history.append(item)
                continue
            if not isinstance(item, dict) or "v" not in item:
                gated_history.append(item)
                continue
            v = item["v"]
            if not torch.is_tensor(v) or v.ndim < 3:
                gated_history.append(item)
                continue
            num_tokens = int(v.shape[2])
            tokens_before.append(num_tokens)
            new_item = dict(item)
            keep_idx = _aligned_keep_indices(num_tokens, device=v.device)
            if keep_idx is not None:
                new_item["v"] = v.index_select(2, keep_idx)
                if torch.is_tensor(item.get("k")):
                    new_item["k"] = item["k"].index_select(2, keep_idx.to(device=item["k"].device))
                if torch.is_tensor(item.get("v_post")):
                    new_item["v_post"] = item["v_post"].index_select(2, keep_idx.to(device=item["v_post"].device))
                if torch.is_tensor(item.get("k_post")):
                    new_item["k_post"] = item["k_post"].index_select(2, keep_idx.to(device=item["k_post"].device))
                v = new_item["v"]
                num_tokens = int(v.shape[2])
            gate = _aligned_gate(num_tokens, device=v.device, dtype=v.dtype)
            scope_aligned = _aligned_scope_mask(num_tokens, device=v.device)
            blend = _aligned_blend(num_tokens, device=v.device, dtype=v.dtype)

            def _residual_center(tensor: torch.Tensor, gate_tensor: torch.Tensor) -> torch.Tensor:
                idx = torch.nonzero(scope_aligned, as_tuple=False).reshape(-1)
                if idx.numel() == 0:
                    return tensor
                src = tensor.index_select(2, idx)
                center = src.mean(dim=2, keepdim=True)
                g = gate_tensor.index_select(2, idx).to(device=tensor.device, dtype=tensor.dtype)
                updated = center + g * (src - center)
                out = tensor.clone()
                out.index_copy_(2, idx.to(device=tensor.device), updated)
                return out

            if mode in {"v_resid", "kv_resid", "v_resid_centered", "kv_resid_centered"}:
                new_item["v"] = _residual_center(v, gate)
            elif mode in {"v", "value", "kv", "both", "v_centered", "kv_centered"}:
                new_item["v"] = v * gate
            if (
                mode in {"k", "key", "kv", "both", "kv_centered", "kv_resid", "kv_resid_centered"}
                and torch.is_tensor(item.get("k"))
            ):
                k_tensor = new_item["k"] if keep_idx is not None else item["k"]
                if mode in {"kv_resid", "kv_resid_centered"}:
                    new_item["k"] = _residual_center(
                        k_tensor,
                        gate.to(device=k_tensor.device, dtype=k_tensor.dtype),
                    )
                else:
                    new_item["k"] = k_tensor * gate.to(device=k_tensor.device, dtype=k_tensor.dtype)
            if wants_cache_blend:
                target = str(self.swa_write_cache_blend_target or "v").strip().lower()
                if target in {"v", "value", "kv", "both"} and torch.is_tensor(new_item.get("v_post")):
                    post = new_item["v_post"].to(device=new_item["v"].device, dtype=new_item["v"].dtype)
                    new_item["v"] = new_item["v"] * (1.0 - blend) + post * blend
                    cache_blend_applied += 1
                if target in {"k", "key", "kv", "both"} and torch.is_tensor(new_item.get("k_post")) and torch.is_tensor(new_item.get("k")):
                    blend_k = blend.to(device=new_item["k"].device, dtype=new_item["k"].dtype)
                    post = new_item["k_post"].to(device=new_item["k"].device, dtype=new_item["k"].dtype)
                    new_item["k"] = new_item["k"] * (1.0 - blend_k) + post * blend_k
                    cache_blend_applied += 1
                new_item.pop("v_post", None)
                new_item.pop("k_post", None)
            gated_history.append(new_item)
            gate_f = gate.detach().cpu().float().reshape(-1)
            gate_means.append(float(gate_f.mean().item()))
            gate_p10.append(float(torch.quantile(gate_f, 0.10).item()))
            gate_p50.append(float(torch.quantile(gate_f, 0.50).item()))
            gate_p90.append(float(torch.quantile(gate_f, 0.90).item()))
            applied += 1
            tokens_after.append(num_tokens)

        debug.update({
            "swa_write_applied_layers": int(applied),
            "swa_write_cache_blend_applied_layers": int(cache_blend_applied),
            "swa_write_gate_mean": float(torch.tensor(gate_means).mean().item()) if gate_means else 1.0,
            "swa_write_gate_p10": float(torch.tensor(gate_p10).mean().item()) if gate_p10 else 1.0,
            "swa_write_gate_p50": float(torch.tensor(gate_p50).mean().item()) if gate_p50 else 1.0,
            "swa_write_gate_p90": float(torch.tensor(gate_p90).mean().item()) if gate_p90 else 1.0,
            "swa_write_selected_mass": float(selected_mass),
            "swa_write_scope_mass": float(scope_mask.float().mean().item()) if scope_mask.numel() else 1.0,
            "swa_write_scope_tokens": int(scope_mask.sum().item()),
            "swa_write_keep_mass": float(keep_mask.float().mean().item()) if keep_mask.numel() else 1.0,
            "swa_write_keep_tokens": int(keep_mask.sum().item()),
            "swa_write_overlap_frames": int(overlap_frames),
            "swa_write_history_tokens_before": max(tokens_before) if tokens_before else 0,
            "swa_write_history_tokens_after": max(tokens_after) if tokens_after else 0,
            "swa_write_D_patch_mean": float(D_patch.mean().item()),
            "swa_write_D_patch_q90": float(torch.quantile(D_patch, 0.90).item()),
            "swa_write_D_scope_mean": float(D_scope.mean().item()) if D_scope.numel() else 0.0,
            "swa_write_D_scope_q90": float(torch.quantile(D_scope, 0.90).item()) if D_scope.numel() else 0.0,
            "swa_write_D_keep_mean": float(D_patch[keep_mask].mean().item()) if keep_mask.any() else 0.0,
            "swa_write_D_keep_q90": float(torch.quantile(D_patch[keep_mask], 0.90).item()) if keep_mask.any() else 0.0,
            "swa_write_cache_blend_scope_mean": float(base_blend[scope_mask].mean().item()) if scope_mask.any() else 0.0,
            "swa_write_cache_blend_scope_q90": float(torch.quantile(base_blend[scope_mask], 0.90).item()) if scope_mask.any() else 0.0,
        })
        return gated_history, debug

    def _swa_write_layer_enabled(self, layer_idx: int, n_layers: int) -> bool:
        mode = str(self.swa_write_layer_mode or "all")
        if mode == "all":
            return True
        if mode == "first":
            return layer_idx == 0
        if mode == "last":
            return layer_idx == max(0, n_layers - 1)
        if mode == "early":
            return layer_idx < max(1, n_layers // 2)
        if mode == "late":
            return layer_idx >= n_layers // 2
        if mode == "middle":
            lo = n_layers // 3
            hi = max(lo + 1, (2 * n_layers) // 3)
            return lo <= layer_idx < hi
        if mode == "single":
            return layer_idx == int(self.swa_write_single_layer)
        raise ValueError(f"Unsupported swa_write_layer_mode: {mode}")

    def build_control_prior(
        self,
        *,
        probe: ProbeOutput,
        cue: Optional[CueOutput],
        prior_output: Optional[PriorOutput],
        mode: str,
    ) -> HybridMemoryControlPrior:
        geo = probe.geometry
        token_type = geo.token_type.detach().cpu().long()
        L_tok = int(token_type.shape[0])
        patch_mask = token_type == TOKEN_TYPE_PATCH
        num_frames = max(int(geo.num_frames), 1)

        if cue is not None and cue.E_cue_patch is not None:
            patch_cube = cue.E_cue_patch.detach().cpu().float()
            patch = patch_cube.reshape(-1, patch_cube.shape[-1])
            dyn_patch = patch[:, CUE_DYN].clamp(0.0, 1.0)
            occ_patch = patch[:, CUE_OCC].clamp(0.0, 1.0)
            unc_patch = patch[:, CUE_UNC].clamp(0.0, 1.0)
            anchor_patch = patch[:, CUE_ANCHOR].clamp(0.0, 1.0)
            conf_patch = _flatten_or_default(geo.confidence, dyn_patch).clamp(0.0, 1.0)
            fast_write_sources = {
                "stage_d",
                "prior",
                "bl01",
                "dg_inv",
                "dg_locked_inv",
                "read_inv",
                "read_inverse",
                "v5_dg_inv",
                "explicit_dyn_inv",
                "exp_dyn_inv",
                "v5_exp_dyn_inv",
                "dg_inv_sqrt",
                "v5_dg_inv_sqrt",
                "dg_inv_sq",
                "v5_dg_inv_sq",
                "stage_d_x_dg_inv",
                "v5_stage_d_x_dg_inv",
                "stage_d_x_dg_inv_sqrt",
                "v5_stage_d_x_dg_inv_sqrt",
                "stage_d_x_dg_high_inv",
                "v5_stage_d_x_dg_high_inv",
                "stage_d_x_dg_high_inv_sqrt",
                "v5_stage_d_x_dg_high_inv_sqrt",
                "stage_d_x_exp_inv",
                "v5_stage_d_x_exp_inv",
                "stage_d_x_exp_inv_sqrt",
                "v5_stage_d_x_exp_inv_sqrt",
                "stage_d_x_exp_inv_sq",
                "v5_stage_d_x_exp_inv_sq",
                "stage_d_x_exp_focal2",
                "stage_d_x_dg_exp_inv_sqrt",
                "v5_stage_d_x_dg_exp_inv_sqrt",
                "stage_d_x_dg_exp_inter_inv",
                "v5_stage_d_x_dg_exp_inter_inv",
                "stage_d_x_dg_exp_inter_inv_sqrt",
                "v5_stage_d_x_dg_exp_inter_inv_sqrt",
                "stage_d_x_union_dyn_inv",
                "v5_stage_d_x_union_dyn_inv",
                "stage_d_x_dg_inv_sq",
                "v5_stage_d_x_dg_inv_sq",
                "stage_d_x_static_focal2",
                "stage_d_x_dg_inv_pow4",
                "v5_stage_d_x_dg_inv_pow4",
                "stage_d_x_static_focal4",
                "stage_d_x_dg_boundary_focal2",
                "v5_stage_d_x_dg_boundary_focal2",
                "residual_reliability",
            }
            fast_acl2_only = (
                self.fast_cue_eval
                and self.read_cue_source.startswith("acl2.gg.")
                and self.read_cue_source.endswith(".headmean.robustq")
                and self.read_protection_mode == "none"
                and not self.read_protect_static
                and str(self.hmc_write_score_source) in fast_write_sources
            )
            if fast_acl2_only:
                explicit_patch = _flatten_cue_field_or_default(
                    cue.C_dyn_explicit,
                    dyn_patch,
                    num_frames=num_frames,
                    patch_grid=(int(geo.patch_grid[0]), int(geo.patch_grid[1])),
                ).clamp(0.0, 1.0)
                read_patch = _acl2_read_patch_from_source(
                    self.read_cue_source,
                    geo.global_q_raw_patchvec_layers,
                    geo.global_k_raw_patchvec_layers,
                    dyn_patch,
                    num_frames=num_frames,
                )
                if read_patch is None:
                    read_patch = dyn_patch
                read_patch = read_patch.clamp(0.0, 1.0)
                if self.read_calib_mode == "per_frame_quantile":
                    calibrated = _adaptive_quantile_calibrate(
                        read_patch,
                        num_frames=num_frames,
                        target_mass=self.read_target_mass,
                        tau=self.read_calib_tau,
                    ).to(read_patch.device)
                    blend = min(max(float(self.read_blend_lambda), 0.0), 1.0)
                    read_patch = ((1.0 - blend) * read_patch + blend * calibrated).clamp(0.0, 1.0)
                if self.read_topk_frac > 0.0:
                    read_patch = _topk_per_frame(read_patch, num_frames=num_frames, frac=self.read_topk_frac)

                swa_source = str(self.swa_write_score_source or "read").strip().lower()
                if swa_source in {"", "read", "dg", "dg_locked", "d_g"}:
                    swa_write_patch = read_patch
                elif swa_source in {"explicit", "explicit_dyn", "exp_dyn"}:
                    swa_write_patch = explicit_patch
                elif swa_source in {"old", "old_dyn", "dyn"}:
                    swa_write_patch = dyn_patch
                elif swa_source in {"union", "union_dyn", "dg_exp_union", "max_dg_exp"}:
                    swa_write_patch = torch.maximum(read_patch, explicit_patch).clamp(0.0, 1.0)
                elif swa_source in {"intersection", "inter", "dg_exp_inter", "min_dg_exp"}:
                    swa_write_patch = torch.minimum(read_patch, explicit_patch).clamp(0.0, 1.0)
                else:
                    raise ValueError(f"Unsupported SWA write score source: {self.swa_write_score_source}")

                phase_e_protect_patch = torch.zeros_like(read_patch)
                D_tok = _token_from_patch_values(geo, read_patch, special_value=0.0).clamp(0.0, 1.0)
                D_swa_write_tok = _token_from_patch_values(geo, swa_write_patch, special_value=0.0).clamp(0.0, 1.0)
                R_tok = _token_from_patch_values(
                    geo,
                    ((1.0 - occ_patch) * (1.0 - unc_patch)).clamp(0.0, 1.0),
                    special_value=1.0,
                )
                P_safe = _token_from_patch_values(geo, phase_e_protect_patch, special_value=0.0)

                old_mask = dyn_patch > 0.5
                new_mask = read_patch > 0.5
                inter = (old_mask & new_mask).float().sum()
                union = (old_mask | new_mask).float().sum().clamp_min(1.0)
                new_count = new_mask.float().sum().clamp_min(1.0)
                old_count = old_mask.float().sum().clamp_min(1.0)
                anchor_collision = (
                    float(((read_patch * anchor_patch).sum() / read_patch.sum().clamp_min(1e-8)).item())
                    if read_patch.numel()
                    else 0.0
                )
                fragmentation = _fragmentation(
                    read_patch,
                    num_frames=num_frames,
                    patch_grid=(int(geo.patch_grid[0]), int(geo.patch_grid[1])),
                    thr=0.5,
                )
                dynamic_mass_001 = float((read_patch > 0.01).float().mean().item()) if read_patch.numel() else 0.0
                dynamic_mass_050 = float((read_patch > 0.5).float().mean().item()) if read_patch.numel() else 0.0
                quality_mass_pass = self.read_quality_mass_min <= dynamic_mass_050 <= self.read_quality_mass_max
                quality_anchor_pass = anchor_collision <= self.read_quality_anchor_max
                quality_frag_pass = fragmentation <= self.read_quality_frag_max
                patch_debug = {
                    "fast_cue_eval": True,
                    "mean_D_patch": float(read_patch.mean().item()) if read_patch.numel() else 0.0,
                    "q10_D_patch": float(torch.quantile(read_patch, 0.1).item()) if read_patch.numel() else 0.0,
                    "q50_D_patch": float(torch.quantile(read_patch, 0.5).item()) if read_patch.numel() else 0.0,
                    "q90_D_patch": float(torch.quantile(read_patch, 0.9).item()) if read_patch.numel() else 0.0,
                    "dynamic_mass_D_gt_001": dynamic_mass_001,
                    "dynamic_mass_D_gt_050": dynamic_mass_050,
                    "dynamic_mass_D_gt_075": float((read_patch > 0.75).float().mean().item()) if read_patch.numel() else 0.0,
                    "anchor_collision": anchor_collision,
                    "fragmentation": fragmentation,
                    "old_dyn_iou": float((inter / union).item()),
                    "old_dyn_coverage": float((inter / new_count).item()),
                    "old_dyn_recall": float((inter / old_count).item()),
                    "cue_quality_mass_pass": bool(quality_mass_pass),
                    "cue_quality_anchor_pass": bool(quality_anchor_pass),
                    "cue_quality_frag_pass": bool(quality_frag_pass),
                    "cue_quality_pass": bool(quality_mass_pass and quality_anchor_pass and quality_frag_pass),
                    "cue_gate": 1.0,
                    "fallback_rate": 0.0,
                    "cue_source_effective": self.read_cue_source,
                    "corr_D_unc": _pearson_corr(read_patch, unc_patch),
                    "corr_D_occ": _pearson_corr(read_patch, occ_patch),
                    "corr_D_conf": _pearson_corr(read_patch, conf_patch),
                    "corr_D_inv_conf": _pearson_corr(read_patch, 1.0 - conf_patch),
                    "corr_D_old_dyn": _pearson_corr(read_patch, dyn_patch),
                    "swa_write_score_source": swa_source,
                    "swa_write_score_mean_patch": float(swa_write_patch.mean().item()) if swa_write_patch.numel() else 0.0,
                    "swa_write_score_q90_patch": float(torch.quantile(swa_write_patch, 0.9).item()) if swa_write_patch.numel() else 0.0,
                    "swa_write_corr_score_D": _pearson_corr(swa_write_patch, read_patch),
                    "swa_write_corr_score_exp_dyn": _pearson_corr(swa_write_patch, explicit_patch),
                    "swa_write_corr_score_old_dyn": _pearson_corr(swa_write_patch, dyn_patch),
                }

                U_tok = torch.ones(L_tok, dtype=torch.float32)
                P_ref = (token_type != TOKEN_TYPE_PATCH).float() if self.read_protect_ref else torch.zeros(L_tok, dtype=torch.float32)
                raw_frame_bias_energy = _frame_bias_energy(
                    D_tok,
                    P_ref,
                    num_frames=num_frames,
                    mode=self.frame_bias_mode,
                )
                beta_frame_effective = self.beta_frame
                beta_was_clipped = False
                if self.beta_policy == "bias_energy_norm" and raw_frame_bias_energy > 0.0 and self.beta_energy_target > 0.0:
                    unclipped = self.beta_frame * self.beta_energy_target / max(raw_frame_bias_energy, 1e-8)
                    beta_frame_effective = min(max(unclipped, self.beta_min), self.beta_max)
                    beta_was_clipped = abs(beta_frame_effective - unclipped) > 1e-8

                P_ttt_write = None
                B_chunk_geo = None
                if prior_output is not None:
                    P_ttt_write = prior_output.A_tok.detach().cpu().float()
                    B_chunk_geo = float(prior_output.B_chunk_geo)
                    override_write, write_debug = self._phase_e_write_prior(
                        probe=probe,
                        token_type=token_type,
                        dyn_patch=dyn_patch,
                        explicit_dyn_patch=explicit_patch,
                        occ_patch=occ_patch,
                        unc_patch=unc_patch,
                        read_patch=read_patch,
                        P_ref=P_ref,
                        base_write_prior=P_ttt_write,
                    )
                    if override_write is not None:
                        P_ttt_write = override_write
                        B_chunk_geo = 1.0
                    patch_debug.update(write_debug)
                else:
                    patch_debug.update({"hmc_write_score_source": self.hmc_write_score_source, "hmc_write_override": False})

                if mode in {"unity_replay", "native", "identity_hooks", "read_path_only", "probe_only"}:
                    P_ttt_write = None
                    B_chunk_geo = None

                return HybridMemoryControlPrior(
                    D_tok=D_tok,
                    R_tok=R_tok,
                    U_tok=U_tok,
                    P_ref=P_ref,
                    P_ttt_write=P_ttt_write,
                    P_ttt_read=None,
                    P_swa_read_prev=None,
                    D_swa_write_tok=D_swa_write_tok,
                    frame_bias_spec={
                        "enabled": self.enable_frame_read_control,
                        "beta": self.beta_frame,
                        "beta_effective": beta_frame_effective,
                        "mode": self.frame_bias_mode,
                        "cue_source": self.read_cue_source,
                        "topk_frac": self.read_topk_frac,
                        "calib_mode": self.read_calib_mode,
                        "target_mass": self.read_target_mass,
                        "calib_tau": self.read_calib_tau,
                        "blend_lambda": self.read_blend_lambda,
                        "protection_mode": self.read_protection_mode,
                        "ref_strength": self.read_ref_strength,
                    },
                    chunk_bias_spec={
                        "enabled": self.enable_chunk_read_control,
                        "beta": self.beta_chunk,
                        "mode": self.chunk_bias_mode,
                        "cue_source": self.read_cue_source,
                    },
                    B_chunk_geo=B_chunk_geo,
                    debug={
                        "mode": mode,
                        "mean_D_tok": float(D_tok.mean().item()) if D_tok.numel() else 0.0,
                        "max_D_tok": float(D_tok.max().item()) if D_tok.numel() else 0.0,
                        "q90_D_tok": float(torch.quantile(D_tok, 0.9).item()) if D_tok.numel() else 0.0,
                        "mean_R_tok": float(R_tok.mean().item()) if R_tok.numel() else 1.0,
                        "protected_token_count": int((P_ref > 0.5).sum().item()),
                        "safe_patch_token_count": int((P_safe > 0.5).sum().item()),
                        "patch_token_count": int(patch_mask.sum().item()),
                        "read_cue_source": self.read_cue_source,
                        "read_topk_frac": self.read_topk_frac,
                        "read_calib_mode": self.read_calib_mode,
                        "read_target_mass": self.read_target_mass,
                        "read_calib_tau": self.read_calib_tau,
                        "read_blend_lambda": self.read_blend_lambda,
                        "read_quality_mass_min": self.read_quality_mass_min,
                        "read_quality_mass_max": self.read_quality_mass_max,
                        "read_quality_anchor_max": self.read_quality_anchor_max,
                        "read_quality_frag_max": self.read_quality_frag_max,
                        "beta_policy": self.beta_policy,
                        "beta_frame_effective": beta_frame_effective,
                        "beta_energy_target": self.beta_energy_target,
                        "beta_raw_frame_bias_energy": raw_frame_bias_energy,
                        "beta_was_clipped": bool(beta_was_clipped),
                        "frame_bias_mode": self.frame_bias_mode,
                        "chunk_bias_mode": self.chunk_bias_mode,
                        "swa_bias_mode": self.swa_bias_mode,
                        "read_protect_ref": self.read_protect_ref,
                        "read_protect_static": self.read_protect_static,
                        "read_protection_mode": self.read_protection_mode,
                        "read_ref_strength": self.read_ref_strength,
                        "read_overlap_frames": self.read_overlap_frames,
                        "read_reset_frames": self.read_reset_frames,
                        "read_attention_q": self.read_attention_q,
                        "hmc_write_score_source": self.hmc_write_score_source,
                        "hmc_write_sparse_ratio": self.hmc_write_sparse_ratio,
                        "hmc_write_sparse_mode": self.hmc_write_sparse_mode,
                        "ttt_write_prior_present": P_ttt_write is not None,
                        "ttt_write_prior_mean": float(P_ttt_write.mean().item()) if P_ttt_write is not None else 1.0,
                        "read_path_control": "real_hook_identity_or_requested",
                        **patch_debug,
                    },
                )
            kk_patch = _normalize01(_flatten_or_default(geo.dyn4d_kk_mean_patch, dyn_patch))
            qq_patch = _normalize01(_flatten_or_default(geo.dyn4d_qq_mean_patch, dyn_patch))
            qk_var_patch = _normalize01(_flatten_or_default(geo.dyn4d_qk_var_patch, dyn_patch))
            dyn4d_patch = _normalize01(_flatten_or_default(geo.dyn4d_patch, dyn_patch))
            key_avg_patch = _normalize01(_flatten_or_default(geo.frame_attn_key_cosine_avg, dyn_patch))
            query_avg_patch = _normalize01(_flatten_or_default(geo.frame_attn_cosine_avg, dyn_patch))
            query_shallow_patch = _normalize01(_flatten_or_default(geo.frame_attn_cosine_shallow, dyn_patch))
            query_deep_patch = _normalize01(_flatten_or_default(geo.frame_attn_cosine_deep, dyn_patch))
            attn_patch = _normalize01(_flatten_or_default(geo.attn_dynamic_patch, dyn_patch))
            frame_layer_ids = geo.frame_attn_cosine_layer_ids
            global_layer_ids = geo.dyn4d_global_layer_ids
            fa_key_l0_raw = _flatten_or_default(geo.frame_attn_key_cosine_l0, dyn_patch)
            fa_key_l4_raw = _flatten_or_default(geo.frame_attn_key_cosine_l4, dyn_patch)
            if geo.frame_attn_cosine_key_layers is None:
                # Current LoGeR trace exports explicit l0/l4 maps but not the full
                # frame-attention layer stack; keep the plan's group names runnable
                # by mapping them onto the available shallow/deeper endpoints.
                fa_key_shallow_raw = fa_key_l0_raw
                fa_key_middle_raw = 0.5 * (fa_key_l0_raw + fa_key_l4_raw)
                fa_key_deep_raw = fa_key_l4_raw
                fa_key_all_raw = fa_key_middle_raw
                fa_key_layer_var_raw = torch.stack([fa_key_l0_raw, fa_key_l4_raw], dim=0).var(dim=0, unbiased=False)
            else:
                fa_key_shallow_raw = _flatten_layer_group_map(
                    geo.frame_attn_cosine_key_layers,
                    frame_layer_ids,
                    dyn_patch,
                    group="shallow",
                )
                fa_key_middle_raw = _flatten_layer_group_map(
                    geo.frame_attn_cosine_key_layers,
                    frame_layer_ids,
                    dyn_patch,
                    group="middle",
                )
                fa_key_deep_raw = _flatten_layer_group_map(
                    geo.frame_attn_cosine_key_layers,
                    frame_layer_ids,
                    dyn_patch,
                    group="deep",
                )
                fa_key_all_raw = _flatten_layer_group_map(
                    geo.frame_attn_cosine_key_layers,
                    frame_layer_ids,
                    dyn_patch,
                    group="all",
                )
                fa_key_layer_var_raw = _flatten_layer_group_map(
                    geo.frame_attn_cosine_key_layers,
                    frame_layer_ids,
                    dyn_patch,
                    group="all",
                    stat="var",
                )
            fa_query_shallow_raw = _flatten_layer_group_map(
                geo.frame_attn_cosine_query_layers,
                frame_layer_ids,
                dyn_patch,
                group="shallow",
            )
            fa_key_l0_patch = _robust_quantile01(fa_key_l0_raw, num_frames=num_frames)
            fa_key_l4_patch = _robust_quantile01(fa_key_l4_raw, num_frames=num_frames)
            fa_key_shallow_patch = _robust_quantile01(fa_key_shallow_raw, num_frames=num_frames)
            fa_key_middle_patch = _robust_quantile01(fa_key_middle_raw, num_frames=num_frames)
            fa_key_deep_patch = _robust_quantile01(fa_key_deep_raw, num_frames=num_frames)
            fa_key_all_patch = _robust_quantile01(fa_key_all_raw, num_frames=num_frames)
            fa_key_layer_var_patch = _robust_quantile01(
                fa_key_layer_var_raw,
                num_frames=num_frames,
            )
            fa_key_decay_patch = _robust_quantile01(
                fa_key_shallow_raw * (fa_key_shallow_raw - fa_key_deep_raw).clamp_min(0.0),
                num_frames=num_frames,
            )
            fa_key_l0_deep_decay_patch = _robust_quantile01(
                fa_key_l0_raw * (fa_key_l0_raw - fa_key_deep_raw).clamp_min(0.0),
                num_frames=num_frames,
            )
            fa_key_l4_deep_decay_patch = _robust_quantile01(
                fa_key_l4_raw * (fa_key_l4_raw - fa_key_deep_raw).clamp_min(0.0),
                num_frames=num_frames,
            )
            fa_key_deep_low_patch = _robust_quantile01(1.0 - fa_key_deep_raw, num_frames=num_frames)
            fa_query_shallow_patch = _robust_quantile01(fa_query_shallow_raw, num_frames=num_frames)
            gg_qk_middle_patch = _robust_quantile01(
                _global_centroid_metric(
                    geo.global_q_raw_patchvec_layers,
                    geo.global_k_raw_patchvec_layers,
                    global_layer_ids,
                    dyn_patch,
                    group="middle",
                    kind="qk_var",
                ),
                num_frames=num_frames,
            )
            gg_qk_shallow_patch = _robust_quantile01(
                _global_centroid_metric(
                    geo.global_q_raw_patchvec_layers,
                    geo.global_k_raw_patchvec_layers,
                    global_layer_ids,
                    dyn_patch,
                    group="shallow",
                    kind="qk_var",
                ),
                num_frames=num_frames,
            )
            gg_qk_deep_patch = _robust_quantile01(
                _global_centroid_metric(
                    geo.global_q_raw_patchvec_layers,
                    geo.global_k_raw_patchvec_layers,
                    global_layer_ids,
                    dyn_patch,
                    group="deep",
                    kind="qk_var",
                ),
                num_frames=num_frames,
            )
            gg_qq_middle_patch = _robust_quantile01(
                _global_centroid_metric(
                    geo.global_q_raw_patchvec_layers,
                    geo.global_k_raw_patchvec_layers,
                    global_layer_ids,
                    dyn_patch,
                    group="middle",
                    kind="qq_low",
                ),
                num_frames=num_frames,
            )
            gg_kk_middle_patch = _robust_quantile01(
                _global_centroid_metric(
                    geo.global_q_raw_patchvec_layers,
                    geo.global_k_raw_patchvec_layers,
                    global_layer_ids,
                    dyn_patch,
                    group="middle",
                    kind="kk_low",
                ),
                num_frames=num_frames,
            )
            gg_qq_shallow_patch = _robust_quantile01(
                _global_centroid_metric(
                    geo.global_q_raw_patchvec_layers,
                    geo.global_k_raw_patchvec_layers,
                    global_layer_ids,
                    dyn_patch,
                    group="shallow",
                    kind="qq_low",
                ),
                num_frames=num_frames,
            )
            gg_deep_static_patch = _robust_quantile01(
                _global_centroid_metric(
                    geo.global_q_raw_patchvec_layers,
                    geo.global_k_raw_patchvec_layers,
                    global_layer_ids,
                    dyn_patch,
                    group="deep",
                    kind="deep_static",
                ),
                num_frames=num_frames,
            )
            gg_smd_a1b1g1_patch = _robust_quantile01(
                gg_qq_shallow_patch * gg_qk_middle_patch * gg_deep_static_patch,
                num_frames=num_frames,
            )
            gg_smd_a0b1g1_patch = _robust_quantile01(
                gg_qk_middle_patch * gg_deep_static_patch,
                num_frames=num_frames,
            )
            explicit_patch = _flatten_cue_field_or_default(
                cue.C_dyn_explicit,
                dyn_patch,
                num_frames=num_frames,
                patch_grid=(int(geo.patch_grid[0]), int(geo.patch_grid[1])),
            ).clamp(0.0, 1.0)
            implicit_patch = _flatten_cue_field_or_default(
                cue.C_dyn_implicit,
                dyn_patch,
                num_frames=num_frames,
                patch_grid=(int(geo.patch_grid[0]), int(geo.patch_grid[1])),
            ).clamp(0.0, 1.0)
            fusion_max_patch = _flatten_cue_field_or_default(
                cue.C_dyn_fusion_max,
                dyn_patch,
                num_frames=num_frames,
                patch_grid=(int(geo.patch_grid[0]), int(geo.patch_grid[1])),
            ).clamp(0.0, 1.0)
            fusion_soft_or_patch = _flatten_cue_field_or_default(
                cue.C_dyn_fusion_soft_or,
                dyn_patch,
                num_frames=num_frames,
                patch_grid=(int(geo.patch_grid[0]), int(geo.patch_grid[1])),
            ).clamp(0.0, 1.0)
            fusion_avg_patch = _flatten_cue_field_or_default(
                cue.C_dyn_fusion_avg,
                dyn_patch,
                num_frames=num_frames,
                patch_grid=(int(geo.patch_grid[0]), int(geo.patch_grid[1])),
            ).clamp(0.0, 1.0)
            fusion_addclip_patch = _flatten_cue_field_or_default(
                cue.C_dyn_fusion_addclip,
                dyn_patch,
                num_frames=num_frames,
                patch_grid=(int(geo.patch_grid[0]), int(geo.patch_grid[1])),
            ).clamp(0.0, 1.0)
            flow_proxy_raw = (dyn_patch * (1.0 - occ_patch).clamp(0.0, 1.0) * conf_patch).clamp(0.0, 1.0)
            structure_like = (anchor_patch >= self.read_static_anchor_thr) & (dyn_patch <= self.read_static_dyn_thr)
            flow_sem_veto_raw = torch.where(structure_like, torch.zeros_like(flow_proxy_raw), flow_proxy_raw)
            flow_proxy_calib = _adaptive_quantile_calibrate(
                flow_proxy_raw,
                num_frames=num_frames,
                target_mass=self.read_target_mass,
                tau=self.read_calib_tau,
            )
            flow_sem_veto_calib = _adaptive_quantile_calibrate(
                flow_sem_veto_raw,
                num_frames=num_frames,
                target_mass=self.read_target_mass,
                tau=self.read_calib_tau,
            )
            cue_gate = 1.0
            fallback_rate = 0.0
            cue_source_effective = self.read_cue_source
            acl2_patch = _acl2_read_patch_from_source(
                self.read_cue_source,
                geo.global_q_raw_patchvec_layers,
                geo.global_k_raw_patchvec_layers,
                dyn_patch,
                num_frames=num_frames,
            )

            if acl2_patch is not None:
                read_patch = acl2_patch
            elif self.read_cue_source in {"dyn", "old_dyn", "old_dyn_calibrated_soft_or"}:
                read_patch = dyn_patch
            elif self.read_cue_source == "explicit_dyn_only":
                read_patch = explicit_patch
            elif self.read_cue_source in {"implicit_dyn_only", "manual_implicit_dyn"}:
                read_patch = implicit_patch
            elif self.read_cue_source == "old_dyn_max":
                read_patch = fusion_max_patch
            elif self.read_cue_source == "old_dyn_soft_or":
                read_patch = fusion_soft_or_patch
            elif self.read_cue_source == "old_dyn_avg":
                read_patch = fusion_avg_patch
            elif self.read_cue_source == "old_dyn_addclip":
                read_patch = fusion_addclip_patch
            elif self.read_cue_source == "inverted_dyn":
                read_patch = (1.0 - dyn_patch).clamp(0.0, 1.0)
            elif self.read_cue_source == "random":
                generator = torch.Generator(device="cpu")
                generator.manual_seed(1729 + int(dyn_patch.numel()))
                read_patch = torch.rand(dyn_patch.shape, generator=generator, dtype=torch.float32)
            elif self.read_cue_source == "dyn_reliable":
                read_patch = (dyn_patch * (1.0 - unc_patch) * (1.0 - occ_patch)).clamp(0.0, 1.0)
            elif self.read_cue_source == "gram_lite":
                raw = (1.0 - kk_patch).clamp(0.0, 1.0) * (1.0 - qq_patch).clamp(0.0, 1.0) * (0.5 + dyn4d_patch)
                read_patch = _normalize01(raw)
            elif self.read_cue_source == "gram4d":
                raw = (1.0 - kk_patch).clamp(0.0, 1.0) * qk_var_patch * (0.25 + qq_patch) * (0.5 + dyn4d_patch)
                read_patch = _normalize01(raw)
            elif self.read_cue_source == "dyn4d_patch":
                read_patch = dyn4d_patch
            elif self.read_cue_source == "qk_var":
                read_patch = qk_var_patch
            elif self.read_cue_source == "qqkk_disagree":
                raw = (1.0 - qq_patch).clamp(0.0, 1.0) * (1.0 - kk_patch).clamp(0.0, 1.0)
                read_patch = _normalize01(raw)
            elif self.read_cue_source == "entropy":
                # Dense attention rows are not exported; this is a conservative
                # dispersion proxy from available attention/dyn4d statistics.
                raw = 0.45 * qk_var_patch + 0.35 * attn_patch + 0.20 * (1.0 - key_avg_patch).clamp(0.0, 1.0)
                read_patch = _normalize01(raw)
            elif self.read_cue_source in {"flow", "flow_proxy"}:
                read_patch = _normalize01(flow_proxy_raw)
            elif self.read_cue_source in {"flow_sem_veto", "flow_sem_veto_proxy"}:
                # v3 patch-match flow proxy: use reprojection/dynamic residual if
                # Stage-B was configured for it, then suppress occluded/uncertain
                # low-confidence patches. RAFT/GMFlow integration is left explicit
                # via flow_model for later runs.
                read_patch = _normalize01(flow_sem_veto_raw)
            elif self.read_cue_source == "flow_proxy_calib":
                read_patch = flow_proxy_calib
            elif self.read_cue_source == "flow_sem_veto_calib":
                read_patch = flow_sem_veto_calib
            elif self.read_cue_source == "old_dyn_plus_flow_proxy":
                lam = min(max(self.read_blend_lambda, 0.0), 1.0)
                read_patch = _normalize01((1.0 - lam) * dyn_patch + lam * flow_proxy_calib)
            elif self.read_cue_source == "old_dyn_plus_flow_sem_veto":
                lam = min(max(self.read_blend_lambda, 0.0), 1.0)
                read_patch = _normalize01((1.0 - lam) * dyn_patch + lam * flow_sem_veto_calib)
            elif self.read_cue_source == "old_dyn_gram_lite_agree":
                lam = min(max(self.read_blend_lambda, 0.0), 1.0)
                gram = _normalize01((1.0 - kk_patch).clamp(0.0, 1.0) * (1.0 - qq_patch).clamp(0.0, 1.0) * (0.5 + dyn4d_patch))
                agree = (1.0 + lam * (gram - 0.5)).clamp(0.25, 1.75)
                read_patch = (dyn_patch * agree).clamp(0.0, 1.0)
            elif self.read_cue_source == "old_dyn_gram4d_agree":
                lam = min(max(self.read_blend_lambda, 0.0), 1.0)
                gram = _normalize01((1.0 - kk_patch).clamp(0.0, 1.0) * qk_var_patch * (0.25 + qq_patch) * (0.5 + dyn4d_patch))
                agree = (1.0 + lam * (gram - 0.5)).clamp(0.25, 1.75)
                read_patch = (dyn_patch * agree).clamp(0.0, 1.0)
            elif self.read_cue_source == "old_dyn_key_static_rescue":
                lam = min(max(self.read_blend_lambda, 0.0), 1.0)
                rescue = (key_avg_patch * anchor_patch * (1.0 - unc_patch).clamp(0.0, 1.0)).clamp(0.0, 1.0)
                read_patch = (dyn_patch * (1.0 - lam * rescue)).clamp(0.0, 1.0)
            elif self.read_cue_source in {
                "mix.c23_g4_soft_l010",
                "mix.c23_g4_soft_l025",
                "mix.c23_g4_soft_l050",
                "mix.c23_g4_soft_l075",
            }:
                D23 = _acl2_read_patch_from_source(
                    "acl2.gg.qq.low.g2_3.full.headmean.robustq",
                    geo.global_q_raw_patchvec_layers,
                    geo.global_k_raw_patchvec_layers,
                    dyn_patch,
                    num_frames=num_frames,
                )
                D4 = _acl2_read_patch_from_source(
                    "acl2.gg.qq.low.g4.full.headmean.robustq",
                    geo.global_q_raw_patchvec_layers,
                    geo.global_k_raw_patchvec_layers,
                    dyn_patch,
                    num_frames=num_frames,
                )
                if D23 is None:
                    D23 = dyn_patch
                if D4 is None:
                    D4 = dyn_patch
                lam_g4 = 0.10
                if self.read_cue_source.endswith("_l025"):
                    lam_g4 = 0.25
                elif self.read_cue_source.endswith("_l050"):
                    lam_g4 = 0.50
                elif self.read_cue_source.endswith("_l075"):
                    lam_g4 = 0.75
                read_patch = ((1.0 - lam_g4) * D23 + lam_g4 * D4).clamp(0.0, 1.0)
                cue_source_effective = self.read_cue_source
            elif self.read_cue_source in {
                "mix.c24past_old_route_lg100_lo025",
                "mix.c24past_old_route_lg100_lo050",
                "mix.c24past_exp_route_lg100_le050",
                "mix.c23past_old_route_lg100_lo025",
                "mix.c24past_static_rescue_a025",
                "mix.c23past_static_rescue_a025",
            }:
                if "c23past" in self.read_cue_source:
                    dg_source = "acl2.gg.qq.low.g2_3.past_only.headmean.robustq"
                else:
                    dg_source = "acl2.gg.qq.low.g2_4.past_only.headmean.robustq"
                D_g = _acl2_read_patch_from_source(
                    dg_source,
                    geo.global_q_raw_patchvec_layers,
                    geo.global_k_raw_patchvec_layers,
                    dyn_patch,
                    num_frames=num_frames,
                )
                if D_g is None:
                    D_g = dyn_patch
                D_g = D_g.clamp(0.0, 1.0)
                if "static_rescue" in self.read_cue_source:
                    alpha = 0.25
                    read_patch = (D_g * (1.0 - alpha * gg_deep_static_patch)).clamp(0.0, 1.0)
                else:
                    aux = explicit_patch if "_exp_" in self.read_cue_source else fusion_addclip_patch
                    lam_g = 1.0
                    lam_aux = 0.50 if ("lo050" in self.read_cue_source or "le050" in self.read_cue_source) else 0.25
                    g_on = D_g > 0.5
                    aux_on = aux > 0.5
                    read_patch = torch.zeros_like(D_g)
                    read_patch = torch.where(g_on & aux_on, torch.maximum(D_g, aux), read_patch)
                    read_patch = torch.where(g_on & (~aux_on), (lam_g * D_g).clamp(0.0, 1.0), read_patch)
                    read_patch = torch.where((~g_on) & aux_on, (lam_aux * aux).clamp(0.0, 1.0), read_patch)
                cue_source_effective = self.read_cue_source
            elif self.read_cue_source == "old_dyn_switch_flow_proxy":
                dyn_mass_flow = float((flow_proxy_calib > 0.5).float().mean().item()) if flow_proxy_calib.numel() else 0.0
                anchor_flow = float(
                    ((flow_proxy_calib * anchor_patch).sum() / flow_proxy_calib.sum().clamp_min(1e-8)).item()
                ) if flow_proxy_calib.numel() else 0.0
                mass_gate = min(max((dyn_mass_flow - 0.02) / 0.04, 0.0), 1.0)
                anchor_gate = min(max((self.read_quality_anchor_max - anchor_flow) / 0.20, 0.0), 1.0)
                cue_gate = float(mass_gate * anchor_gate)
                fallback_rate = 1.0 - cue_gate
                read_patch = _normalize01((1.0 - cue_gate) * dyn_patch + cue_gate * flow_proxy_calib)
            elif self.read_cue_source == "old_dyn_switch_flow_sem_veto":
                dyn_mass_flow = float((flow_sem_veto_calib > 0.5).float().mean().item()) if flow_sem_veto_calib.numel() else 0.0
                anchor_flow = float(
                    ((flow_sem_veto_calib * anchor_patch).sum() / flow_sem_veto_calib.sum().clamp_min(1e-8)).item()
                ) if flow_sem_veto_calib.numel() else 0.0
                mass_gate = min(max((dyn_mass_flow - 0.02) / 0.04, 0.0), 1.0)
                anchor_gate = min(max((self.read_quality_anchor_max - anchor_flow) / 0.20, 0.0), 1.0)
                cue_gate = float(mass_gate * anchor_gate)
                fallback_rate = 1.0 - cue_gate
                read_patch = _normalize01((1.0 - cue_gate) * dyn_patch + cue_gate * flow_sem_veto_calib)
            elif self.read_cue_source == "internal_attn":
                internal = _flatten_optional_patch_map(geo.attn_dynamic_patch)
                read_patch = internal.clamp(0.0, 1.0) if internal is not None and internal.numel() else dyn_patch
            elif self.read_cue_source == "key_cosine_avg":
                internal = _flatten_optional_patch_map(geo.frame_attn_key_cosine_avg)
                read_patch = _normalize01(internal) if internal is not None and internal.numel() else dyn_patch
            elif self.read_cue_source == "key_cosine_shallow":
                internal = _flatten_optional_patch_map(geo.frame_attn_key_cosine_shallow)
                read_patch = _normalize01(internal) if internal is not None and internal.numel() else dyn_patch
            elif self.read_cue_source == "key_cosine_deep":
                internal = _flatten_optional_patch_map(geo.frame_attn_key_cosine_deep)
                read_patch = _normalize01(internal) if internal is not None and internal.numel() else dyn_patch
            elif self.read_cue_source == "query_cosine_avg":
                read_patch = query_avg_patch
            elif self.read_cue_source == "query_cosine_shallow":
                read_patch = query_shallow_patch
            elif self.read_cue_source == "query_cosine_deep":
                read_patch = query_deep_patch
            elif self.read_cue_source == "shallow_deep_disagree":
                read_patch = _normalize01((query_shallow_patch - query_deep_patch).abs())
            elif self.read_cue_source == "fa.key.l0.high.robustq":
                read_patch = fa_key_l0_patch
            elif self.read_cue_source == "fa.key.l4.high.robustq":
                read_patch = fa_key_l4_patch
            elif self.read_cue_source == "fa.key.shallow.high.robustq":
                read_patch = fa_key_shallow_patch
            elif self.read_cue_source == "fa.key.middle.high.robustq":
                read_patch = fa_key_middle_patch
            elif self.read_cue_source == "fa.key.deep.high.robustq":
                read_patch = fa_key_deep_patch
            elif self.read_cue_source == "fa.key.all.high.robustq":
                read_patch = fa_key_all_patch
            elif self.read_cue_source == "fa.key.shallow_deep.decay.robustq":
                read_patch = fa_key_decay_patch
            elif self.read_cue_source == "fa.key.l0_deep.decay.robustq":
                read_patch = fa_key_l0_deep_decay_patch
            elif self.read_cue_source == "fa.key.l4_deep.decay.robustq":
                read_patch = fa_key_l4_deep_decay_patch
            elif self.read_cue_source == "fa.key.layerVar.robustq":
                read_patch = fa_key_layer_var_patch
            elif self.read_cue_source == "fa.key.deep.low.robustq":
                read_patch = fa_key_deep_low_patch
            elif self.read_cue_source == "fa.query.shallow.high.robustq":
                read_patch = fa_query_shallow_patch
            elif self.read_cue_source == "gg.qk.middle.var.robustq":
                read_patch = gg_qk_middle_patch
            elif self.read_cue_source == "gg.qk.shallow.var.robustq":
                read_patch = gg_qk_shallow_patch
            elif self.read_cue_source == "gg.qk.deep.var.robustq":
                read_patch = gg_qk_deep_patch
            elif self.read_cue_source == "gg.qq.middle.low.robustq":
                read_patch = gg_qq_middle_patch
            elif self.read_cue_source == "gg.kk.middle.low.robustq":
                read_patch = gg_kk_middle_patch
            elif self.read_cue_source == "gg.smd.product.a1b1g1.robustq":
                read_patch = gg_smd_a1b1g1_patch
            elif self.read_cue_source == "gg.smd.product.a0b1g1.robustq":
                read_patch = gg_smd_a0b1g1_patch
            elif self.read_cue_source == "mix.exp_add_fa_decay_l05":
                read_patch = (explicit_patch + 0.5 * fa_key_decay_patch).clamp(0.0, 1.0)
            elif self.read_cue_source == "mix.old_agree_fa_decay_l05":
                agree = (1.0 + 0.5 * (2.0 * fa_key_decay_patch - 1.0)).clamp(0.25, 1.75)
                read_patch = (fusion_addclip_patch * agree).clamp(0.0, 1.0)
            elif self.read_cue_source == "mix.old_veto_gg_deep_static_l05":
                read_patch = (fusion_addclip_patch * (1.0 - 0.5 * gg_deep_static_patch)).clamp(0.0, 1.0)
            elif self.read_cue_source == "mix.old_route_gg_smd":
                reliability = (1.0 - (gg_smd_a1b1g1_patch - fusion_addclip_patch).abs()).clamp(0.0, 1.0)
                read_patch = (reliability * fusion_addclip_patch + (1.0 - reliability) * explicit_patch).clamp(0.0, 1.0)
            elif self.read_cue_source == "mix.exp_add_gg_qkvar_l05":
                read_patch = (explicit_patch + 0.5 * gg_qk_middle_patch).clamp(0.0, 1.0)
            else:
                read_patch = dyn_patch

            if self.read_calib_mode == "per_frame_quantile":
                calibrated = _adaptive_quantile_calibrate(
                    read_patch,
                    num_frames=num_frames,
                    target_mass=self.read_target_mass,
                    tau=self.read_calib_tau,
                ).to(read_patch.device)
                blend = min(max(float(self.read_blend_lambda), 0.0), 1.0)
                read_patch = ((1.0 - blend) * read_patch + blend * calibrated).clamp(0.0, 1.0)

            if self.read_topk_frac > 0.0:
                read_patch = _topk_per_frame(read_patch, num_frames=num_frames, frac=self.read_topk_frac)

            safe_patch = (anchor_patch >= self.read_static_anchor_thr) & (dyn_patch <= self.read_static_dyn_thr)
            phase_e_protect_patch, protect_debug = self._build_phase_e_protection_patch(
                mode=self.read_protection_mode,
                num_patch=int(read_patch.numel()),
                num_frames=num_frames,
                safe_patch=safe_patch,
                key_avg_patch=key_avg_patch,
            )
            if self.read_protect_static:
                phase_e_protect_patch = torch.maximum(phase_e_protect_patch, safe_patch.float())
            if bool((phase_e_protect_patch > 0.0).any()):
                strength = min(max(float(self.read_ref_strength), 0.0), 1.0)
                read_patch = (read_patch * (1.0 - strength * phase_e_protect_patch)).clamp(0.0, 1.0)
                cue_source_effective = f"{cue_source_effective}+protect:{self.read_protection_mode}"

            D_tok = _token_from_patch_values(geo, read_patch, special_value=0.0).clamp(0.0, 1.0)
            R_tok = _token_from_patch_values(
                geo,
                ((1.0 - occ_patch) * (1.0 - unc_patch)).clamp(0.0, 1.0),
                special_value=1.0,
            )
            P_safe = _token_from_patch_values(geo, phase_e_protect_patch.float(), special_value=0.0)
            old_mask = dyn_patch > 0.5
            new_mask = read_patch > 0.5
            inter = (old_mask & new_mask).float().sum()
            union = (old_mask | new_mask).float().sum().clamp_min(1.0)
            new_count = new_mask.float().sum().clamp_min(1.0)
            old_count = old_mask.float().sum().clamp_min(1.0)
            anchor_collision = (
                float(((read_patch * anchor_patch).sum() / read_patch.sum().clamp_min(1e-8)).item())
                if read_patch.numel()
                else 0.0
            )
            fragmentation = _fragmentation(
                read_patch,
                num_frames=num_frames,
                patch_grid=(int(geo.patch_grid[0]), int(geo.patch_grid[1])),
                thr=0.5,
            )
            dynamic_mass_001 = float((read_patch > 0.01).float().mean().item()) if read_patch.numel() else 0.0
            dynamic_mass_050 = float((read_patch > 0.5).float().mean().item()) if read_patch.numel() else 0.0
            quality_mass_pass = self.read_quality_mass_min <= dynamic_mass_050 <= self.read_quality_mass_max
            quality_anchor_pass = anchor_collision <= self.read_quality_anchor_max
            quality_frag_pass = fragmentation <= self.read_quality_frag_max
            patch_debug = {
                "mean_D_patch": float(read_patch.mean().item()) if read_patch.numel() else 0.0,
                "q10_D_patch": float(torch.quantile(read_patch, 0.1).item()) if read_patch.numel() else 0.0,
                "q50_D_patch": float(torch.quantile(read_patch, 0.5).item()) if read_patch.numel() else 0.0,
                "q90_D_patch": float(torch.quantile(read_patch, 0.9).item()) if read_patch.numel() else 0.0,
                "dynamic_mass_D_gt_001": dynamic_mass_001,
                "dynamic_mass_D_gt_050": dynamic_mass_050,
                "dynamic_mass_D_gt_075": float((read_patch > 0.75).float().mean().item()) if read_patch.numel() else 0.0,
                "anchor_collision": anchor_collision,
                "fragmentation": fragmentation,
                "old_dyn_iou": float((inter / union).item()),
                "old_dyn_coverage": float((inter / new_count).item()),
                "old_dyn_recall": float((inter / old_count).item()),
                "cue_quality_mass_pass": bool(quality_mass_pass),
                "cue_quality_anchor_pass": bool(quality_anchor_pass),
                "cue_quality_frag_pass": bool(quality_frag_pass),
                "cue_quality_pass": bool(quality_mass_pass and quality_anchor_pass and quality_frag_pass),
                "cue_gate": float(cue_gate),
                "fallback_rate": float(fallback_rate),
                "cue_source_effective": cue_source_effective,
                "read_calib_mode": self.read_calib_mode,
                "read_target_mass": self.read_target_mass,
                "read_calib_tau": self.read_calib_tau,
                "read_blend_lambda": self.read_blend_lambda,
                "read_quality_mass_min": self.read_quality_mass_min,
                "read_quality_mass_max": self.read_quality_mass_max,
                "read_quality_anchor_max": self.read_quality_anchor_max,
                "read_quality_frag_max": self.read_quality_frag_max,
                "corr_D_unc": _pearson_corr(read_patch, unc_patch),
                "corr_D_occ": _pearson_corr(read_patch, occ_patch),
                "corr_D_conf": _pearson_corr(read_patch, conf_patch),
                "corr_D_inv_conf": _pearson_corr(read_patch, 1.0 - conf_patch),
                "corr_D_old_dyn": _pearson_corr(read_patch, dyn_patch),
                "flow_model": self.flow_model,
                "flow_pair_stride": self.flow_pair_stride,
                "flow_fb_thr": self.flow_fb_thr,
                "flow_residual_thr": self.flow_residual_thr,
                "gram_layer_groups": self.gram_layer_groups,
                **protect_debug,
            }
        else:
            D_tok = torch.zeros(L_tok, dtype=torch.float32)
            R_tok = torch.ones(L_tok, dtype=torch.float32)
            P_safe = torch.zeros(L_tok, dtype=torch.float32)
            patch_debug = {}

        U_tok = torch.ones(L_tok, dtype=torch.float32)
        P_ref = (token_type != TOKEN_TYPE_PATCH).float() if self.read_protect_ref else torch.zeros(L_tok, dtype=torch.float32)
        if bool((P_safe > 0.0).any()):
            P_ref = torch.maximum(P_ref, P_safe)
        raw_frame_bias_energy = _frame_bias_energy(
            D_tok,
            P_ref,
            num_frames=num_frames,
            mode=self.frame_bias_mode,
        )
        beta_frame_effective = self.beta_frame
        beta_was_clipped = False
        if self.beta_policy == "bias_energy_norm" and raw_frame_bias_energy > 0.0 and self.beta_energy_target > 0.0:
            unclipped = self.beta_frame * self.beta_energy_target / max(raw_frame_bias_energy, 1e-8)
            beta_frame_effective = min(max(unclipped, self.beta_min), self.beta_max)
            beta_was_clipped = abs(beta_frame_effective - unclipped) > 1e-8
        P_ttt_write = None
        B_chunk_geo = None
        if prior_output is not None:
            P_ttt_write = prior_output.A_tok.detach().cpu().float()
            B_chunk_geo = float(prior_output.B_chunk_geo)
            override_write, write_debug = self._phase_e_write_prior(
                probe=probe,
                token_type=token_type,
                dyn_patch=dyn_patch if cue is not None and cue.E_cue_patch is not None else None,
                explicit_dyn_patch=explicit_patch if cue is not None and cue.E_cue_patch is not None else None,
                occ_patch=occ_patch if cue is not None and cue.E_cue_patch is not None else None,
                unc_patch=unc_patch if cue is not None and cue.E_cue_patch is not None else None,
                read_patch=read_patch if cue is not None and cue.E_cue_patch is not None else None,
                P_ref=P_ref,
                base_write_prior=P_ttt_write,
            )
            if override_write is not None:
                P_ttt_write = override_write
                B_chunk_geo = 1.0
            patch_debug.update(write_debug)
        else:
            patch_debug.update({"hmc_write_score_source": self.hmc_write_score_source, "hmc_write_override": False})

        if mode in {"unity_replay", "native", "identity_hooks", "read_path_only", "probe_only"}:
            P_ttt_write = None
            B_chunk_geo = None

        return HybridMemoryControlPrior(
            D_tok=D_tok,
            R_tok=R_tok,
            U_tok=U_tok,
            P_ref=P_ref,
            P_ttt_write=P_ttt_write,
            P_ttt_read=None,
            P_swa_read_prev=None,
            frame_bias_spec={
                "enabled": self.enable_frame_read_control,
                "beta": self.beta_frame,
                "beta_effective": beta_frame_effective,
                "mode": self.frame_bias_mode,
                "cue_source": self.read_cue_source,
                "topk_frac": self.read_topk_frac,
                "calib_mode": self.read_calib_mode,
                "target_mass": self.read_target_mass,
                "calib_tau": self.read_calib_tau,
                "blend_lambda": self.read_blend_lambda,
                "protection_mode": self.read_protection_mode,
                "ref_strength": self.read_ref_strength,
            },
            chunk_bias_spec={
                "enabled": self.enable_chunk_read_control,
                "beta": self.beta_chunk,
                "mode": self.chunk_bias_mode,
                "cue_source": self.read_cue_source,
            },
            B_chunk_geo=B_chunk_geo,
            debug={
                "mode": mode,
                "mean_D_tok": float(D_tok.mean().item()) if D_tok.numel() else 0.0,
                "max_D_tok": float(D_tok.max().item()) if D_tok.numel() else 0.0,
                "q90_D_tok": float(torch.quantile(D_tok, 0.9).item()) if D_tok.numel() else 0.0,
                "mean_R_tok": float(R_tok.mean().item()) if R_tok.numel() else 1.0,
                "protected_token_count": int((P_ref > 0.5).sum().item()),
                "safe_patch_token_count": int((P_safe > 0.5).sum().item()),
                "patch_token_count": int(patch_mask.sum().item()),
                "read_cue_source": self.read_cue_source,
                "read_topk_frac": self.read_topk_frac,
                "read_calib_mode": self.read_calib_mode,
                "read_target_mass": self.read_target_mass,
                "read_calib_tau": self.read_calib_tau,
                "read_blend_lambda": self.read_blend_lambda,
                "read_quality_mass_min": self.read_quality_mass_min,
                "read_quality_mass_max": self.read_quality_mass_max,
                "read_quality_anchor_max": self.read_quality_anchor_max,
                "read_quality_frag_max": self.read_quality_frag_max,
                "beta_policy": self.beta_policy,
                "beta_frame_effective": beta_frame_effective,
                "beta_energy_target": self.beta_energy_target,
                "beta_raw_frame_bias_energy": raw_frame_bias_energy,
                "beta_was_clipped": bool(beta_was_clipped),
                "frame_bias_mode": self.frame_bias_mode,
                "chunk_bias_mode": self.chunk_bias_mode,
                "swa_bias_mode": self.swa_bias_mode,
                "read_protect_ref": self.read_protect_ref,
                "read_protect_static": self.read_protect_static,
                "read_protection_mode": self.read_protection_mode,
                "read_ref_strength": self.read_ref_strength,
                "read_overlap_frames": self.read_overlap_frames,
                "read_reset_frames": self.read_reset_frames,
                "read_attention_q": self.read_attention_q,
                "hmc_write_score_source": self.hmc_write_score_source,
                "hmc_write_sparse_ratio": self.hmc_write_sparse_ratio,
                "hmc_write_sparse_mode": self.hmc_write_sparse_mode,
                "ttt_write_prior_present": P_ttt_write is not None,
                "ttt_write_prior_mean": float(P_ttt_write.mean().item()) if P_ttt_write is not None else 1.0,
                "read_path_control": "real_hook_identity_or_requested",
                **patch_debug,
            },
        )

    def run_controlled(
        self,
        backbone: LoGeRGeometryBackbone,
        images: torch.Tensor,
        state_m: Optional[HybridMemoryState],
        control_prior: Optional[HybridMemoryControlPrior],
        *,
        mode: str,
        token_type: Optional[torch.Tensor] = None,
        skip_ttt_write_replay: bool = False,
    ) -> HybridMemoryResult:
        if mode == "probe_only":
            raise ValueError("probe_only should be handled by the caller after run_probe().")

        input_hash = hybrid_state_fingerprint(state_m)
        model_hmc_control = self._build_model_hmc_control(control_prior, mode=mode)
        if state_m is not None and state_m.prev_control_summary is not None:
            prev_D_patch = state_m.prev_control_summary.get("D_patch")
            if prev_D_patch is not None:
                model_hmc_control["D_prev_patch"] = prev_D_patch
        geo, write_cache = backbone.run(
            images,
            ttt_state=state_m.to_ttt_input() if state_m is not None else None,
            cache_ttt_primitives=True,
            hmc_control=model_hmc_control,
        )

        write_result: Optional[WriteResult] = None
        if mode in {"native", "identity_hooks"}:
            ttt_next = self._state_from_write_cache(write_cache)
            write_debug = {
                "mode": mode,
                "native_write_through": True,
                "ttt_layers_committed": int(write_cache.num_ttt_layers),
            }
        else:
            if mode in {"unity_replay", "read_path_only"}:
                write_mode = "unity_replay"
                A_tok = None
                B_chunk_geo = None
            elif mode in {"ttt_write_only", "hybrid"}:
                write_mode = "semantic"
                A_tok = control_prior.P_ttt_write if control_prior is not None else None
                B_chunk_geo = control_prior.B_chunk_geo if control_prior is not None else None
            else:
                raise ValueError(f"Unsupported hybrid memory mode: {mode}")

            if skip_ttt_write_replay:
                ttt_next = self._state_from_write_cache(write_cache)
                write_debug = {
                    "mode": write_mode,
                    "semantic_write_skipped": True,
                    "skip_reason": "probe_ttt_write_commit_replays_probe_cache",
                    "ttt_layers_committed": int(write_cache.num_ttt_layers),
                }
            else:
                old_mode = self.ttt_update_controller.write_mode
                self.ttt_update_controller.write_mode = write_mode
                write_result = self.ttt_update_controller.run(
                    write_cache,
                    A_tok,
                    B_chunk_geo=B_chunk_geo,
                    device=self.device,
                    token_type=token_type,
                    num_frames=int(geo.num_frames),
                    overlap_frames=int(self.read_overlap_frames),
                    risk_tok=control_prior.D_tok if control_prior is not None else None,
                    prev_transient_delta=(
                        state_m.ttt_state.get("transient_delta")
                        if state_m is not None and isinstance(state_m.ttt_state, dict)
                        else None
                    ),
                )
                self.ttt_update_controller.write_mode = old_mode
                gated_history, swa_write_debug = self._apply_swa_history_write_gate(
                    write_result.history,
                    control_prior,
                    geo,
                )
                write_result.history = gated_history
                write_result.debug.update(swa_write_debug)
                ttt_next = {"w0": write_result.w0, "w1": write_result.w1, "w2": write_result.w2}
                if write_result.history is not None:
                    ttt_next["history"] = write_result.history
                if write_result.transient_delta is not None:
                    ttt_next["transient_delta"] = write_result.transient_delta
                write_debug = write_result.debug

        state_next = HybridMemoryState(
            ttt_state=ttt_next,
            swa_state=None,
            ref_state=None,
            prev_control_summary=self._build_prev_control_summary(control_prior, geo),
        )
        output_hash = hybrid_state_fingerprint(state_next)
        trace_dict = geo.hmc_trace or {}
        hook_effect_summary: Dict[str, Dict[str, Any]] = {}
        for path_name, records in trace_dict.items():
            if not isinstance(records, list):
                continue
            mean_vals = [
                float(rec.get("mean_abs_bias", 0.0))
                for rec in records
                if isinstance(rec, dict) and rec.get("mean_abs_bias") is not None
            ]
            mean_vals.extend([
                float(rec.get("swa_overlap_bias_mean_abs", 0.0))
                for rec in records
                if isinstance(rec, dict) and rec.get("swa_overlap_bias_mean_abs") is not None
            ])
            max_vals = [
                float(rec.get("max_abs_bias", 0.0))
                for rec in records
                if isinstance(rec, dict) and rec.get("max_abs_bias") is not None
            ]
            max_vals.extend([
                float(rec.get("swa_overlap_bias_max_abs", 0.0))
                for rec in records
                if isinstance(rec, dict) and rec.get("swa_overlap_bias_max_abs") is not None
            ])
            gate_mean_vals = [
                float(rec.get("swa_gate_mean", 1.0))
                for rec in records
                if isinstance(rec, dict) and rec.get("swa_gate_mean") is not None
            ]
            overlap_source_gate_mean_vals = [
                float(rec.get("swa_overlap_source_gate_mean", 1.0))
                for rec in records
                if isinstance(rec, dict) and rec.get("swa_overlap_source_gate_mean") is not None
            ]
            gate_delta_vals = [
                float(rec.get("mean_abs_gate_delta", 0.0))
                for rec in records
                if isinstance(rec, dict) and rec.get("mean_abs_gate_delta") is not None
            ]
            overlap_source_gate_delta_vals = [
                float(rec.get("swa_overlap_source_gate_mean_abs_delta", 0.0))
                for rec in records
                if isinstance(rec, dict) and rec.get("swa_overlap_source_gate_mean_abs_delta") is not None
            ]
            max_gate_delta_vals = [
                float(rec.get("max_abs_gate_delta", 0.0))
                for rec in records
                if isinstance(rec, dict) and rec.get("max_abs_gate_delta") is not None
            ]
            overlap_source_max_gate_delta_vals = [
                float(rec.get("swa_overlap_source_gate_max_abs_delta", 0.0))
                for rec in records
                if isinstance(rec, dict) and rec.get("swa_overlap_source_gate_max_abs_delta") is not None
            ]
            overlap_source_score_vals = [
                float(rec.get("swa_overlap_source_gate_score_mean", 0.0))
                for rec in records
                if isinstance(rec, dict) and rec.get("swa_overlap_source_gate_score_mean") is not None
            ]
            overlap_source_score_q90_vals = [
                float(rec.get("swa_overlap_source_gate_score_q90", 0.0))
                for rec in records
                if isinstance(rec, dict) and rec.get("swa_overlap_source_gate_score_q90") is not None
            ]
            overlap_source_replace_alpha_vals = [
                float(rec.get("swa_overlap_source_replace_alpha_mean", 0.0))
                for rec in records
                if isinstance(rec, dict) and rec.get("swa_overlap_source_replace_alpha_mean") is not None
            ]
            overlap_source_replace_alpha_p90_vals = [
                float(rec.get("swa_overlap_source_replace_alpha_p90", 0.0))
                for rec in records
                if isinstance(rec, dict) and rec.get("swa_overlap_source_replace_alpha_p90") is not None
            ]
            overlap_source_replace_score_vals = [
                float(rec.get("swa_overlap_source_replace_score_mean", 0.0))
                for rec in records
                if isinstance(rec, dict) and rec.get("swa_overlap_source_replace_score_mean") is not None
            ]
            history_token_vals = [
                int(rec.get("history_tokens", 0) or 0)
                for rec in records
                if isinstance(rec, dict)
            ]
            d_prev_token_vals = [
                int(rec.get("d_prev_tokens", 0) or 0)
                for rec in records
                if isinstance(rec, dict)
            ]
            hook_effect_summary[path_name] = {
                "num_calls": len(records),
                "num_enabled_layers": sum(
                    1 for rec in records
                    if isinstance(rec, dict)
                    and (
                        bool(rec.get("layer_enabled", True))
                        or bool(rec.get("swa_overlap_bias_applied", False))
                        or bool(rec.get("swa_overlap_source_gate_applied", False))
                        or bool(rec.get("swa_overlap_source_replace_applied", False))
                    )
                ),
                "num_swa_overlap_bias_applied": sum(
                    1 for rec in records if isinstance(rec, dict) and bool(rec.get("swa_overlap_bias_applied", False))
                ),
                "num_swa_overlap_source_gate_applied": sum(
                    1 for rec in records if isinstance(rec, dict) and bool(rec.get("swa_overlap_source_gate_applied", False))
                ),
                "num_swa_overlap_source_replace_applied": sum(
                    1 for rec in records
                    if isinstance(rec, dict) and bool(rec.get("swa_overlap_source_replace_applied", False))
                ),
                "num_source_gate_applied": sum(
                    1 for rec in records if isinstance(rec, dict) and bool(rec.get("source_gate_applied", False))
                ),
                "mean_abs_bias": float(torch.tensor(mean_vals).mean().item()) if mean_vals else 0.0,
                "max_abs_bias": max(max_vals) if max_vals else 0.0,
                "mean_swa_gate": float(torch.tensor(gate_mean_vals).mean().item()) if gate_mean_vals else 1.0,
                "mean_abs_gate_delta": float(torch.tensor(gate_delta_vals).mean().item()) if gate_delta_vals else 0.0,
                "max_abs_gate_delta": max(max_gate_delta_vals) if max_gate_delta_vals else 0.0,
                "mean_swa_overlap_source_gate": float(torch.tensor(overlap_source_gate_mean_vals).mean().item())
                if overlap_source_gate_mean_vals else 1.0,
                "mean_swa_overlap_source_gate_delta": float(torch.tensor(overlap_source_gate_delta_vals).mean().item())
                if overlap_source_gate_delta_vals else 0.0,
                "max_swa_overlap_source_gate_delta": max(overlap_source_max_gate_delta_vals)
                if overlap_source_max_gate_delta_vals else 0.0,
                "mean_swa_overlap_source_score": float(torch.tensor(overlap_source_score_vals).mean().item())
                if overlap_source_score_vals else 0.0,
                "mean_swa_overlap_source_score_q90": float(torch.tensor(overlap_source_score_q90_vals).mean().item())
                if overlap_source_score_q90_vals else 0.0,
                "mean_swa_overlap_source_replace_alpha": float(torch.tensor(overlap_source_replace_alpha_vals).mean().item())
                if overlap_source_replace_alpha_vals else 0.0,
                "mean_swa_overlap_source_replace_alpha_p90": float(torch.tensor(overlap_source_replace_alpha_p90_vals).mean().item())
                if overlap_source_replace_alpha_p90_vals else 0.0,
                "mean_swa_overlap_source_replace_score": float(torch.tensor(overlap_source_replace_score_vals).mean().item())
                if overlap_source_replace_score_vals else 0.0,
                "max_history_tokens": max(history_token_vals) if history_token_vals else 0,
                "max_d_prev_tokens": max(d_prev_token_vals) if d_prev_token_vals else 0,
            }
        implemented_paths: List[str] = []
        if mode == "identity_hooks":
            implemented_paths = ["frame_attention", "swa_read", "ttt_apply", "chunk_attention"]
        elif mode == "ttt_write_only":
            implemented_paths = ["ttt_update"]
        elif mode == "hybrid":
            implemented_paths = ["ttt_update"]
            if self.enable_frame_read_control:
                implemented_paths.append("frame_attention")
            if self.enable_swa_read_control:
                implemented_paths.append("swa_read")
            elif self.enable_swa_overlap_bias:
                implemented_paths.append("swa_overlap_bias")
            elif self.enable_swa_overlap_source_gate:
                implemented_paths.append("swa_overlap_source_gate")
            elif self.enable_swa_overlap_source_replace:
                implemented_paths.append("swa_overlap_source_replace")
            if self.enable_ttt_apply_control:
                implemented_paths.append("ttt_apply")
            if self.enable_chunk_read_control:
                implemented_paths.append("chunk_attention")
        elif mode == "read_path_only":
            if self.enable_frame_read_control:
                implemented_paths.append("frame_attention")
            if self.enable_swa_read_control:
                implemented_paths.append("swa_read")
            elif self.enable_swa_overlap_bias:
                implemented_paths.append("swa_overlap_bias")
            elif self.enable_swa_overlap_source_gate:
                implemented_paths.append("swa_overlap_source_gate")
            elif self.enable_swa_overlap_source_replace:
                implemented_paths.append("swa_overlap_source_replace")
            if self.enable_ttt_apply_control:
                implemented_paths.append("ttt_apply")
            if self.enable_chunk_read_control:
                implemented_paths.append("chunk_attention")
        return HybridMemoryResult(
            geometry_output=geo,
            state_next=state_next,
            ttt_state_next=ttt_next,
            swa_state_next=None,
            write_result=write_result,
            control_trace={
                "implemented_paths": implemented_paths,
                "identity_hook_paths": ["frame_attention", "swa_read", "ttt_apply", "chunk_attention"]
                if mode == "identity_hooks" else [],
                "hook_trace_counts": {
                    "frame_attention": len(trace_dict.get("frame_attention", [])),
                    "swa_read": len(trace_dict.get("swa_read", [])),
                    "ttt_apply": len(trace_dict.get("ttt_apply", [])),
                    "chunk_attention": len(trace_dict.get("chunk_attention", [])),
                },
                "hook_effect_summary": hook_effect_summary,
                "mode": mode,
            },
            debug={
                "mode": mode,
                "controlled_input_state_hash": input_hash,
                "controlled_output_state_hash": output_hash,
                "ttt_cache_layers": int(write_cache.num_ttt_layers),
                "write_debug": write_debug,
                "read_path_control": "real_identity_hook_sites" if mode == "identity_hooks" else "available_hook_sites",
            },
        )

    def build_probe_ttt_write_state(
        self,
        probe: ProbeOutput,
        control_prior: Optional[HybridMemoryControlPrior],
        *,
        token_type: Optional[torch.Tensor] = None,
        prev_ttt_state: Optional[Dict[str, Any]] = None,
    ) -> Tuple[HybridMemoryState, Optional[WriteResult]]:
        """Commit TTT writes from the native probe cache, not controlled tokens.

        Phase D uses this to combine read-path controlled *outputs* with a
        separately decided TTT write memory update.  The controlled pass may
        improve the current chunk, but its altered tokens should not implicitly
        become the source of the next chunk's fast weights.
        """

        write_cache = probe.hybrid_cache.ttt_cache
        if write_cache is None:
            return probe.native_provisional_state, None
        A_tok = control_prior.P_ttt_write if control_prior is not None else None
        B_chunk_geo = control_prior.B_chunk_geo if control_prior is not None else None
        old_mode = self.ttt_update_controller.write_mode
        self.ttt_update_controller.write_mode = "semantic"
        write_result = self.ttt_update_controller.run(
            write_cache,
            A_tok,
            B_chunk_geo=B_chunk_geo,
            device=self.device,
            token_type=token_type,
            num_frames=int(probe.geometry.num_frames),
            overlap_frames=int(self.read_overlap_frames),
            risk_tok=control_prior.D_tok if control_prior is not None else None,
            prev_transient_delta=(
                prev_ttt_state.get("transient_delta")
                if isinstance(prev_ttt_state, dict)
                else None
            ),
        )
        self.ttt_update_controller.write_mode = old_mode
        gated_history, swa_write_debug = self._apply_swa_history_write_gate(
            write_result.history,
            control_prior,
            probe.geometry,
        )
        write_result.history = gated_history
        write_result.debug.update(swa_write_debug)
        ttt_next = {"w0": write_result.w0, "w1": write_result.w1, "w2": write_result.w2}
        if write_result.history is not None:
            ttt_next["history"] = write_result.history
        if write_result.transient_delta is not None:
            ttt_next["transient_delta"] = write_result.transient_delta
        state_next = HybridMemoryState(
            ttt_state=ttt_next,
            swa_state=None,
            ref_state=None,
            prev_control_summary=self._build_prev_control_summary(control_prior, probe.geometry),
            debug={
                "commit_source": "probe_ttt_write",
                "write_debug": write_result.debug,
                "probe_native_state_hash": hybrid_state_fingerprint(probe.native_provisional_state),
            },
        )
        return state_next, write_result

    def _build_model_hmc_control(
        self,
        control_prior: Optional[HybridMemoryControlPrior],
        *,
        mode: str,
        identity_hooks: bool = False,
    ) -> Dict[str, Any]:
        is_identity = bool(identity_hooks or mode == "identity_hooks")
        return {
            "identity_hooks": is_identity,
            "collect_trace": True,
            "D_tok": control_prior.D_tok if control_prior is not None else None,
            "P_ref": control_prior.P_ref if control_prior is not None else None,
            "enable_frame_read_control": False if is_identity else self.enable_frame_read_control,
            "enable_swa_read_control": False if is_identity else self.enable_swa_read_control,
            "enable_ttt_apply_control": False if is_identity else self.enable_ttt_apply_control,
            "enable_chunk_read_control": False if is_identity else self.enable_chunk_read_control,
            "beta_frame": 0.0 if is_identity else (
                float((control_prior.frame_bias_spec or {}).get("beta_effective", self.beta_frame))
                if control_prior is not None else self.beta_frame
            ),
            "beta_swa": 0.0 if is_identity else self.beta_swa,
            "swa_gate_min": 1.0 if is_identity else self.swa_gate_min,
            "swa_layer_mode": "first_swa_only",
            "enable_swa_overlap_bias": False if is_identity else self.enable_swa_overlap_bias,
            "swa_overlap_bias_beta": 0.0 if is_identity else self.swa_overlap_bias_beta,
            "swa_overlap_bias_min_keep": self.swa_overlap_bias_min_keep,
            "swa_overlap_bias_mode": self.swa_overlap_bias_mode,
            "swa_overlap_bias_layer_mode": self.swa_overlap_bias_layer_mode,
            "swa_overlap_bias_single_layer": self.swa_overlap_bias_single_layer,
            "enable_swa_overlap_source_gate": False if is_identity else self.enable_swa_overlap_source_gate,
            "swa_overlap_source_gate_rho": 0.0 if is_identity else self.swa_overlap_source_gate_rho,
            "swa_overlap_source_gate_min": self.swa_overlap_source_gate_min,
            "swa_overlap_source_gate_mode": self.swa_overlap_source_gate_mode,
            "swa_overlap_source_gate_target": self.swa_overlap_source_gate_target,
            "swa_overlap_source_gate_layer_mode": self.swa_overlap_source_gate_layer_mode,
            "swa_overlap_source_gate_single_layer": self.swa_overlap_source_gate_single_layer,
            "enable_swa_overlap_source_replace": False if is_identity else self.enable_swa_overlap_source_replace,
            "swa_overlap_source_replace_alpha": 0.0 if is_identity else self.swa_overlap_source_replace_alpha,
            "swa_overlap_source_replace_mode": self.swa_overlap_source_replace_mode,
            "swa_overlap_source_replace_target": self.swa_overlap_source_replace_target,
            "swa_overlap_source_replace_layer_mode": self.swa_overlap_source_replace_layer_mode,
            "swa_overlap_source_replace_single_layer": self.swa_overlap_source_replace_single_layer,
            "swa_write_cache_store_post": self.enable_swa_write_control and self.swa_write_cache_blend_alpha > 0.0,
            "swa_overlap_frames": int(self.read_overlap_frames),
            "beta_chunk": 0.0 if is_identity else self.beta_chunk,
            "rho_ttt_apply": 0.0 if is_identity else self.rho_ttt_apply,
            "read_layer_mode": "all" if is_identity else self.read_layer_mode,
            "read_single_layer": -1 if is_identity else self.read_single_layer,
            "read_cue_source": self.read_cue_source,
            "read_topk_frac": self.read_topk_frac,
            "beta_policy": self.beta_policy,
            "beta_energy_target": self.beta_energy_target,
            "frame_bias_mode": self.frame_bias_mode,
            "chunk_bias_mode": self.chunk_bias_mode,
            "swa_bias_mode": self.swa_bias_mode,
            "ttt_apply_min_gate": 0.0 if is_identity else self.ttt_apply_min_gate,
        }

    def _build_probe_trace(
        self,
        geo: GeometryOutput,
        write_cache: WriteCacheOutput,
    ) -> Dict[str, Any]:
        trace: Dict[str, Any] = {
            "internal_dynamic_patch": geo.attn_dynamic_patch,
            "ttt_residual_patch": None,
            "swa_prev_importance": None,
            "frame_attn_entropy_patch": None,
            "chunk_attn_dynamic_mass": None,
            "reference_importance": None,
            "debug": {
                "ttt_layers": int(write_cache.num_ttt_layers),
                "implemented": [
                    "internal_dynamic_patch_if_available",
                    "ttt_write_cache",
                    "real_frame_attn_hook_sites",
                    "real_swa_hook_sites",
                    "real_ttt_apply_hook_sites",
                    "real_chunk_attn_hook_sites",
                    "ttt_residual_summary",
                ],
                "missing": ["swa_dense_importance_map", "chunk_dense_dynamic_mass_map"],
            },
        }
        residual_means: List[float] = []
        residual_maxes: List[float] = []
        for lc in write_cache.layer_caches:
            if lc.apply_output_raw is None:
                continue
            pred = lc.apply_output_raw.float()
            target = lc.v.float()
            denom = torch.linalg.norm(target, dim=-1).clamp_min(1e-6)
            residual = torch.linalg.norm(pred - target, dim=-1) / denom
            residual_means.append(float(residual.mean().item()))
            residual_maxes.append(float(residual.max().item()))
        trace["ttt_residual_summary"] = {
            "layers": len(residual_means),
            "mean": float(sum(residual_means) / len(residual_means)) if residual_means else None,
            "max": float(max(residual_maxes)) if residual_maxes else None,
        }
        if geo.hmc_trace is not None:
            trace["hook_trace_counts"] = {
                "frame_attention": len(geo.hmc_trace.get("frame_attention", [])),
                "swa_read": len(geo.hmc_trace.get("swa_read", [])),
                "ttt_apply": len(geo.hmc_trace.get("ttt_apply", [])),
                "chunk_attention": len(geo.hmc_trace.get("chunk_attention", [])),
            }
        return trace

    @staticmethod
    def _state_from_write_cache(write_cache: WriteCacheOutput) -> Dict[str, Any]:
        state = {
            "w0": list(write_cache.w0_provisional),
            "w1": list(write_cache.w1_provisional),
            "w2": list(write_cache.w2_provisional),
        }
        if write_cache.history_provisional is not None:
            state["history"] = write_cache.history_provisional
        return state

    @staticmethod
    def _build_prev_control_summary(
        control_prior: Optional[HybridMemoryControlPrior],
        geo: GeometryOutput,
    ) -> Optional[Dict[str, Any]]:
        if control_prior is None or control_prior.D_tok is None:
            return None
        token_type = geo.token_type.detach().cpu().long()
        patch_mask = token_type == TOKEN_TYPE_PATCH
        return {
            "mean_D_tok": float(control_prior.D_tok.float().mean().item()),
            "mean_D_patch": float(control_prior.D_tok[patch_mask].float().mean().item()) if patch_mask.any() else 0.0,
            "mean_R_tok": float(control_prior.R_tok.float().mean().item()) if control_prior.R_tok is not None else 1.0,
            "num_frames": int(geo.num_frames),
            "patch_grid": tuple(int(x) for x in geo.patch_grid),
            "D_patch": control_prior.D_tok[patch_mask].detach().cpu().float() if patch_mask.any() else None,
        }


__all__ = [
    "HybridMemoryState",
    "HybridMemoryCacheOutput",
    "ProbeOutput",
    "HybridMemoryControlPrior",
    "HybridMemoryResult",
    "HybridMemoryController",
    "hybrid_state_fingerprint",
]
