"""
Stage A: LoGeR Geometry Backbone

Wraps the Pi3/LoGeR model and provides a clean interface for the
Semantic Prior Pipeline.  Each call to `run_chunk` processes one
chunk of RGB frames and returns structured geometry outputs that
downstream modules (Dynamic Cue Extractor, Semantic Prior Generator,
TTT Write Controller) expect.
"""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import yaml
from PIL import Image
from torchvision import transforms

from loger.models.pi3 import Pi3
from loger.utils.geometry import depth_edge, homogenize_points


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------
@dataclass
class GeometryOutput:
    """Structured output of Stage A for a single chunk.

    All tensors are on CPU and use float32 unless noted otherwise.
    Batch dimension is squeezed — shapes assume a *single* sequence.
    """

    # Camera-space pointmap: [T, H_p, W_p, 3]
    local_points: torch.Tensor

    # World-space pointmap: [T, H_p, W_p, 3]
    world_points: torch.Tensor

    # World-from-camera poses: [T, 4, 4]
    camera_poses: torch.Tensor

    # Geometry confidence (after sigmoid + edge mask): [T, H_p, W_p]
    confidence: torch.Tensor

    # Patch token metadata: [L_patch, 3]  — each row is (t, y_tok, x_tok)
    patch_meta: torch.Tensor

    # Token type ids: [L_tok] — 0 = register, 1 = role, 2 = patch
    token_type: torch.Tensor

    # Symmetric frame affinity prior derived from decoder attention: [T, T]
    frame_attention_prior: Optional[torch.Tensor] = None

    # Patch-level Stage-A attention feature used downstream. By default this is
    # the mean key-cosine map over frame-attention layers [0,2,4,6,8,10,12,14].
    attn_dynamic_patch: Optional[torch.Tensor] = None

    # VGGT4D-style global-attention dynamic saliency: [T, H_tok, W_tok]
    dyn4d_patch: Optional[torch.Tensor] = None
    dyn4d_qq_mean_patch: Optional[torch.Tensor] = None
    dyn4d_qk_var_patch: Optional[torch.Tensor] = None
    dyn4d_kk_mean_patch: Optional[torch.Tensor] = None
    global_q_raw_patchvec: Optional[torch.Tensor] = None
    global_k_raw_patchvec: Optional[torch.Tensor] = None
    global_q_raw_patchvec_layers: Optional[torch.Tensor] = None
    global_k_raw_patchvec_layers: Optional[torch.Tensor] = None
    dyn4d_global_layer_ids: Optional[torch.Tensor] = None

    # MUT3R-style frame-attention cosine response maps: [T, H_tok, W_tok]
    frame_attn_cosine_shallow: Optional[torch.Tensor] = None
    frame_attn_cosine_deep: Optional[torch.Tensor] = None
    frame_attn_cosine_avg: Optional[torch.Tensor] = None
    frame_attn_key_cosine_l0: Optional[torch.Tensor] = None
    frame_attn_key_cosine_l4: Optional[torch.Tensor] = None
    frame_attn_key_cosine_shallow: Optional[torch.Tensor] = None
    frame_attn_key_cosine_deep: Optional[torch.Tensor] = None
    frame_attn_key_cosine_avg: Optional[torch.Tensor] = None
    frame_attn_cosine_query_layers: Optional[torch.Tensor] = None
    frame_attn_cosine_key_layers: Optional[torch.Tensor] = None
    frame_attn_cosine_layer_ids: Optional[torch.Tensor] = None

    # Number of frames in this chunk
    num_frames: int = 0

    # Pointmap spatial resolution (H_p, W_p)
    pointmap_resolution: Tuple[int, int] = (0, 0)

    # Patch token grid (H_tok, W_tok)
    patch_grid: Tuple[int, int] = (0, 0)

    # Raw model prediction dict (kept for debug / downstream flexibility)
    raw_predictions: Dict[str, torch.Tensor] = field(default_factory=dict)


@dataclass
class TTTLayerCache:
    """Raw update primitives for one TTT layer — enough for delayed
    write-back with token-level prior weighting."""

    q: torch.Tensor            # [b*h, l, d_k]
    k: torch.Tensor            # [b*h, l, d_k]
    v: torch.Tensor            # [b*h, l, d_v]
    lr0: torch.Tensor          # [b*h, l, 1]   (η for branch 0)
    lr1: torch.Tensor          # [b*h, l, 1]   (η for branch 1)
    lr2: torch.Tensor          # [b*h, l, 1]   (η for branch 2)
    w0_old: torch.Tensor       # [b*h, d, dh]  (W_m branch 0)
    w1_old: torch.Tensor       # [b*h, dh, d]  (W_m branch 1)
    w2_old: torch.Tensor       # [b*h, d, dh]  (W_m branch 2)
    momentum: Optional[torch.Tensor]  # None or tensor
    muon_update_steps: int
    ttt_update_steps: int
    ttt_op_order: list


@dataclass
class WriteCacheOutput:
    """WriteCache_m — all information the TTT Write Controller needs to
    perform delayed write-back (W_m → W_{m+1})."""

    # Per-layer raw update primitives
    layer_caches: List[TTTLayerCache]

    # W_m reference: provisional updated weights (for debug / fallback)
    w0_provisional: List[Optional[torch.Tensor]]
    w1_provisional: List[Optional[torch.Tensor]]
    w2_provisional: List[Optional[torch.Tensor]]
    history_provisional: Optional[List[Optional[Dict[str, torch.Tensor]]]] = None

    # Token alignment info (duplicated from GeometryOutput for self-containment)
    num_frames: int = 0
    patch_grid: Tuple[int, int] = (0, 0)
    num_ttt_layers: int = 0


# ---------------------------------------------------------------------------
# Image loading helpers
# ---------------------------------------------------------------------------
_to_tensor = transforms.ToTensor()

PATCH_SIZE = 14
NUM_REGISTER_TOKENS = 5
NUM_ROLE_TOKENS = 1  # one pe_token chosen per frame
SPECIAL_TOKENS_PER_FRAME = NUM_REGISTER_TOKENS + NUM_ROLE_TOKENS

# Token type IDs (matches pipeline convention)
TOKEN_TYPE_REGISTER = 0
TOKEN_TYPE_ROLE = 1
TOKEN_TYPE_PATCH = 2


def compute_target_resolution(
    width: int,
    height: int,
    pixel_limit: int = 255_000,
) -> Tuple[int, int]:
    """Choose the largest 14-aligned resolution within *pixel_limit*."""
    scale = math.sqrt(pixel_limit / (width * height)) if width * height > 0 else 1.0
    k = round(width * scale / PATCH_SIZE)
    m = round(height * scale / PATCH_SIZE)
    while (k * PATCH_SIZE) * (m * PATCH_SIZE) > pixel_limit:
        if k / m > width / height:
            k -= 1
        else:
            m -= 1
    return max(1, k) * PATCH_SIZE, max(1, m) * PATCH_SIZE


def load_images(
    paths: Sequence[str],
    target_w: Optional[int] = None,
    target_h: Optional[int] = None,
    pixel_limit: int = 255_000,
) -> torch.Tensor:
    """Load a list of image paths into a ``[T, 3, H, W]`` float32 tensor.

    If *target_w* / *target_h* are ``None`` they are computed automatically
    from the first image so that the total pixel count stays below
    *pixel_limit* while remaining 14-aligned.
    """
    images: List[Image.Image] = []
    for p in paths:
        images.append(Image.open(p).convert("RGB"))
    if not images:
        raise ValueError("No images loaded")

    if target_w is None or target_h is None:
        w0, h0 = images[0].size
        target_w, target_h = compute_target_resolution(w0, h0, pixel_limit)

    tensors = [
        _to_tensor(img.resize((target_w, target_h), Image.Resampling.LANCZOS))
        for img in images
    ]
    return torch.stack(tensors, dim=0)  # [T, 3, H, W]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(
    checkpoint_path: str,
    config_path: Optional[str] = None,
) -> Pi3:
    """Instantiate a :class:`Pi3` model and load weights.

    Parameters
    ----------
    checkpoint_path:
        Path to the ``latest.pt`` checkpoint file.
    config_path:
        Path to the ``original_config.yaml`` that accompanies the checkpoint.
        When given, its ``model`` section is used for constructor kwargs.
    """
    model_kwargs: Dict = {}
    if config_path:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        model_section = config.get("model", {})

        sig = inspect.signature(Pi3.__init__)
        valid_keys = {
            name
            for name, param in sig.parameters.items()
            if name not in {"self", "args", "kwargs"}
            and param.kind
            in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        }
        for key in sorted(valid_keys):
            if key in model_section:
                val = model_section[key]
                if key in {"ttt_insert_after", "attn_insert_after"} and isinstance(val, str):
                    val = yaml.safe_load(val)
                model_kwargs[key] = val

    model = Pi3(**model_kwargs)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    state_dict = {
        (k[7:] if k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }
    model.load_state_dict(state_dict, strict=True)
    return model


# ---------------------------------------------------------------------------
# Core backbone class
# ---------------------------------------------------------------------------
class LoGeRGeometryBackbone:
    """Stage A of the Semantic Prior Pipeline.

    Thin stateless wrapper around a loaded :class:`Pi3` model.  It
    normalises the call convention, builds ``forward_kwargs`` from a
    simple config, and converts the raw output dict into a
    :class:`GeometryOutput` that the rest of the pipeline can consume.

    Usage::

        backbone = LoGeRGeometryBackbone.from_config(
            checkpoint="ckpts/LoGeR_star/latest.pt",
            config="ckpts/LoGeR_star/original_config.yaml",
            device="cuda",
        )
        images = load_images(paths, target_w=504, target_h=280)
        output = backbone.run(images)
    """

    def __init__(
        self,
        model: Pi3,
        *,
        device: str = "cuda",
        dtype: Optional[torch.dtype] = None,
        window_size: int = 32,
        overlap_size: int = 3,
        reset_every: int = 0,
        se3: bool = False,
        sim3: bool = False,
        sim3_scale_mode: str = "median",
        turn_off_ttt: bool = False,
        turn_off_swa: bool = False,
        edge_rtol: float = 0.03,
        update_ttt_weights: bool = False,
    ):
        self.model = model.to(device).eval()
        self.device = device
        if dtype is None:
            if torch.cuda.is_available():
                cap = torch.cuda.get_device_capability(device)
                dtype = torch.bfloat16 if cap[0] >= 8 else torch.float16
            else:
                dtype = torch.float32
        self.dtype = dtype

        self.forward_kwargs = dict(
            window_size=window_size,
            overlap_size=overlap_size,
            reset_every=reset_every,
            se3=se3,
            sim3=sim3,
            sim3_scale_mode=sim3_scale_mode,
            turn_off_ttt=turn_off_ttt,
            turn_off_swa=turn_off_swa,
        )
        self.edge_rtol = edge_rtol
        self.update_ttt_weights = bool(update_ttt_weights)
        self._ttt_state: Optional[Dict[str, Any]] = None

    # -- Factory -----------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        checkpoint: str,
        config: Optional[str] = None,
        **kwargs,
    ) -> "LoGeRGeometryBackbone":
        """Build from checkpoint + config paths.

        Extra *kwargs* are forwarded to ``__init__`` (e.g. ``device``,
        ``window_size``, ``se3``).  If *config* supplies ``se3: true``
        and the caller does not override it, it is picked up
        automatically.
        """
        se3_from_config = False
        if config:
            with open(config, "r") as f:
                cfg = yaml.safe_load(f)
            se3_from_config = cfg.get("model", {}).get("se3", False)

        if "se3" not in kwargs:
            kwargs["se3"] = se3_from_config

        model = load_model(checkpoint, config)
        return cls(model, **kwargs)

    # -- Inference ---------------------------------------------------------

    def reset_ttt_state(self) -> None:
        """Clear the internally tracked adaptive state (TTT + SWA)."""
        self._ttt_state = None

    def get_ttt_state(self) -> Optional[Dict[str, Any]]:
        """Return a shallow copy of the internal adaptive state, if any."""
        if self._ttt_state is None:
            return None
        state = {
            "w0": list(self._ttt_state["w0"]),
            "w1": list(self._ttt_state["w1"]),
            "w2": list(self._ttt_state["w2"]),
        }
        if "history" in self._ttt_state:
            state["history"] = self._copy_history_state(self._ttt_state["history"])
        return state

    def _copy_history_state(
        self,
        history: Optional[List[Optional[Dict[str, torch.Tensor]]]],
    ) -> Optional[List[Optional[Dict[str, torch.Tensor]]]]:
        if history is None:
            return None
        copied: List[Optional[Dict[str, torch.Tensor]]] = []
        for entry in history:
            if entry is None:
                copied.append(None)
            else:
                copied.append({
                    "k": entry["k"].clone(),
                    "v": entry["v"].clone(),
                })
        return copied

    def _move_ttt_state_to_device(
        self,
        ttt_state: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Move an adaptive state dict to the backbone device."""
        if ttt_state is None:
            return None
        moved = {
            "w0": [x.to(self.device) if x is not None else None for x in ttt_state["w0"]],
            "w1": [x.to(self.device) if x is not None else None for x in ttt_state["w1"]],
            "w2": [x.to(self.device) if x is not None else None for x in ttt_state["w2"]],
        }
        history = ttt_state.get("history")
        if history is not None:
            moved_history: List[Optional[Dict[str, torch.Tensor]]] = []
            for entry in history:
                if entry is None:
                    moved_history.append(None)
                else:
                    moved_history.append({
                        "k": entry["k"].to(self.device),
                        "v": entry["v"].to(self.device),
                    })
            moved["history"] = moved_history
        return moved

    def _extract_ttt_state(self, raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract provisional adaptive state from raw model output."""
        ttt_info = raw.get("ttt_output_info")
        if not ttt_info:
            return None

        w0 = ttt_info.get("w0")
        w1 = ttt_info.get("w1")
        w2 = ttt_info.get("w2")
        if w0 is None or w1 is None or w2 is None:
            return None

        state = {
            "w0": [x.detach().cpu() if x is not None else None for x in w0],
            "w1": [x.detach().cpu() if x is not None else None for x in w1],
            "w2": [x.detach().cpu() if x is not None else None for x in w2],
        }
        history = ttt_info.get("history")
        if history is not None:
            history_cpu: List[Optional[Dict[str, torch.Tensor]]] = []
            for entry in history:
                if entry is None:
                    history_cpu.append(None)
                else:
                    history_cpu.append({
                        "k": entry["k"].detach().cpu(),
                        "v": entry["v"].detach().cpu(),
                    })
            state["history"] = history_cpu
        return state

    @torch.no_grad()
    def run(
        self,
        images: torch.Tensor,
        ttt_state: Optional[Dict[str, Any]] = None,
        cache_ttt_primitives: bool = False,
        **override_kwargs,
    ) -> Union[GeometryOutput, Tuple[GeometryOutput, WriteCacheOutput]]:
        """Run geometry inference on a batch of images (one chunk).

        Parameters
        ----------
        images:
            ``[T, 3, H, W]`` float32 tensor (values in [0, 1]).
        ttt_state:
            External adaptive state to use. Dict with keys ``"w0"``,
            ``"w1"``, ``"w2"`` and optional ``"history"`` for the SWA KV
            cache. When ``None`` the model's default initial states are used.
        cache_ttt_primitives:
            If True, return a ``WriteCacheOutput`` alongside the
            ``GeometryOutput``.  The cache contains everything needed
            by the TTT Write Controller for delayed write-back.
        override_kwargs:
            Any key present in ``self.forward_kwargs`` can be overridden
            per-call (e.g. ``window_size=48``).

        Returns
        -------
        GeometryOutput
            When *cache_ttt_primitives* is False.
        (GeometryOutput, WriteCacheOutput)
            When *cache_ttt_primitives* is True.
        """
        if images.dim() == 3:
            images = images.unsqueeze(0)
        if images.dim() == 4:
            images = images.unsqueeze(0)

        T = images.shape[1]
        H, W = images.shape[3], images.shape[4]
        patch_h, patch_w = H // PATCH_SIZE, W // PATCH_SIZE

        fwd_kwargs = {**self.forward_kwargs, **override_kwargs}
        if cache_ttt_primitives:
            fwd_kwargs["cache_ttt_primitives"] = True
        elif self.update_ttt_weights:
            # Need only updated fast weights, not full replay caches (q/k/v/lr).
            fwd_kwargs["return_ttt_state"] = True

        ttt_state_input = ttt_state
        if ttt_state_input is None and self.update_ttt_weights:
            ttt_state_input = self._ttt_state
        if ttt_state_input is not None:
            fwd_kwargs["ttt_state_input"] = self._move_ttt_state_to_device(ttt_state_input)

        # Keep the full sequence on CPU by default. Pi3.forward() already
        # slices by internal windows and moves each window to the model device,
        # which dramatically lowers peak GPU memory while preserving the
        # original window-loop semantics.
        with torch.cuda.amp.autocast(
            enabled=torch.cuda.is_available(), dtype=self.dtype
        ):
            raw = self.model(images, **fwd_kwargs)

        if self.update_ttt_weights:
            self._ttt_state = self._extract_ttt_state(raw)

        geo = self._postprocess(raw, images, T, H, W, patch_h, patch_w)

        if not cache_ttt_primitives:
            return geo

        write_cache = self._build_write_cache(raw, T, patch_h, patch_w)
        return geo, write_cache

    # -- Write cache builder -----------------------------------------------

    def _build_write_cache(
        self,
        raw: Dict[str, Any],
        T: int,
        patch_h: int,
        patch_w: int,
    ) -> WriteCacheOutput:
        """Extract TTT layer caches from the raw model output."""
        ttt_info = raw.get("ttt_output_info")

        layer_caches: List[TTTLayerCache] = []
        w0_prov: List[Optional[torch.Tensor]] = []
        w1_prov: List[Optional[torch.Tensor]] = []
        w2_prov: List[Optional[torch.Tensor]] = []
        history_prov: Optional[List[Optional[Dict[str, torch.Tensor]]]] = None

        if ttt_info is not None:
            write_caches = ttt_info.get("write_cache", [])
            n_layers = len(write_caches) if write_caches else 0

            for li in range(n_layers):
                wc = write_caches[li]
                if wc is None:
                    continue
                layer_caches.append(TTTLayerCache(
                    q=wc["q"].cpu(),
                    k=wc["k"].cpu(),
                    v=wc["v"].cpu(),
                    lr0=wc["lr0"].cpu(),
                    lr1=wc["lr1"].cpu(),
                    lr2=wc["lr2"].cpu(),
                    w0_old=wc["w0_old"].cpu(),
                    w1_old=wc["w1_old"].cpu(),
                    w2_old=wc["w2_old"].cpu(),
                    momentum=wc["momentum"].cpu() if wc["momentum"] is not None else None,
                    muon_update_steps=wc["muon_update_steps"],
                    ttt_update_steps=wc["ttt_update_steps"],
                    ttt_op_order=wc["ttt_op_order"],
                ))

            for li in range(n_layers):
                w0_prov.append(
                    ttt_info["w0"][li].detach().cpu()
                    if ttt_info["w0"][li] is not None else None
                )
                w1_prov.append(
                    ttt_info["w1"][li].detach().cpu()
                    if ttt_info["w1"][li] is not None else None
                )
                w2_prov.append(
                    ttt_info["w2"][li].detach().cpu()
                    if ttt_info["w2"][li] is not None else None
                )

            history = ttt_info.get("history")
            if history is not None:
                history_prov = []
                for entry in history:
                    if entry is None:
                        history_prov.append(None)
                    else:
                        history_prov.append({
                            "k": entry["k"].detach().cpu(),
                            "v": entry["v"].detach().cpu(),
                        })

        return WriteCacheOutput(
            layer_caches=layer_caches,
            w0_provisional=w0_prov,
            w1_provisional=w1_prov,
            w2_provisional=w2_prov,
            history_provisional=history_prov,
            num_frames=T,
            patch_grid=(patch_h, patch_w),
            num_ttt_layers=len(layer_caches),
        )

    # -- Post-processing ---------------------------------------------------

    def _postprocess(
        self,
        raw: Dict[str, torch.Tensor],
        images: torch.Tensor,
        T: int,
        H: int,
        W: int,
        patch_h: int,
        patch_w: int,
    ) -> GeometryOutput:
        """Convert raw Pi3 output into :class:`GeometryOutput`."""

        # --- Confidence: sigmoid + edge suppression -----------------------
        conf = raw.get("conf")
        if conf is not None:
            conf = torch.sigmoid(conf)  # [B, T, H_p, W_p, 1]
            local_pts = raw.get("local_points")
            if local_pts is not None and self.edge_rtol > 0:
                edge = depth_edge(local_pts[..., 2], rtol=self.edge_rtol)
                conf[edge] = 0.0

        # --- Squeeze batch dim & move to CPU float32 ---------------------
        def _sq(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if t is None:
                return None
            return t.squeeze(0).detach().cpu().float()

        world_pts = _sq(raw.get("points"))        # [T, H_p, W_p, 3]
        local_pts = _sq(raw.get("local_points"))   # [T, H_p, W_p, 3]
        cam_poses = _sq(raw.get("camera_poses"))   # [T, 4, 4]
        conf_out = _sq(conf)                       # [T, H_p, W_p, 1]
        frame_attn = _sq(raw.get("frame_attention_prior"))  # [T, T]
        attn_dynamic_patch = _sq(raw.get("attn_dynamic_patch"))  # [T, H_tok, W_tok]
        dyn4d_patch = _sq(raw.get("dyn4d_patch"))  # [T, H_tok, W_tok]
        dyn4d_qq_mean_patch = _sq(raw.get("dyn4d_qq_mean_patch"))  # [T, H_tok, W_tok]
        dyn4d_qk_var_patch = _sq(raw.get("dyn4d_qk_var_patch"))  # [T, H_tok, W_tok]
        dyn4d_kk_mean_patch = _sq(raw.get("dyn4d_kk_mean_patch"))  # [T, H_tok, W_tok]
        global_q_raw_patchvec = _sq(raw.get("global_q_raw_patchvec"))  # [T, H_tok, W_tok, D]
        global_k_raw_patchvec = _sq(raw.get("global_k_raw_patchvec"))  # [T, H_tok, W_tok, D]
        global_q_raw_patchvec_layers = _sq(raw.get("global_q_raw_patchvec_layers"))  # [T, L, H_tok, W_tok, D]
        global_k_raw_patchvec_layers = _sq(raw.get("global_k_raw_patchvec_layers"))  # [T, L, H_tok, W_tok, D]
        dyn4d_global_layer_ids = raw.get("dyn4d_global_layer_ids")
        if dyn4d_global_layer_ids is not None:
            dyn4d_global_layer_ids = dyn4d_global_layer_ids.squeeze(0).detach().cpu().long()
        frame_attn_cosine_shallow = _sq(raw.get("frame_attn_cosine_shallow"))  # [T, H_tok, W_tok]
        frame_attn_cosine_deep = _sq(raw.get("frame_attn_cosine_deep"))        # [T, H_tok, W_tok]
        frame_attn_cosine_avg = _sq(raw.get("frame_attn_cosine_avg"))          # [T, H_tok, W_tok]
        frame_attn_key_cosine_l0 = _sq(raw.get("frame_attn_key_cosine_l0"))    # [T, H_tok, W_tok]
        frame_attn_key_cosine_l4 = _sq(raw.get("frame_attn_key_cosine_l4"))    # [T, H_tok, W_tok]
        frame_attn_key_cosine_shallow = _sq(raw.get("frame_attn_key_cosine_shallow"))  # [T, H_tok, W_tok]
        frame_attn_key_cosine_deep = _sq(raw.get("frame_attn_key_cosine_deep"))        # [T, H_tok, W_tok]
        frame_attn_key_cosine_avg = _sq(raw.get("frame_attn_key_cosine_avg"))          # [T, H_tok, W_tok]
        frame_attn_cosine_query_layers = _sq(raw.get("frame_attn_cosine_query_layers"))  # [T, L, H_tok, W_tok]
        frame_attn_cosine_key_layers = _sq(raw.get("frame_attn_cosine_key_layers"))      # [T, L, H_tok, W_tok]
        frame_attn_cosine_layer_ids = raw.get("frame_attn_cosine_layer_ids")
        if frame_attn_cosine_layer_ids is not None:
            frame_attn_cosine_layer_ids = frame_attn_cosine_layer_ids.squeeze(0).detach().cpu().long()

        if conf_out is not None and conf_out.dim() == 4:
            conf_out = conf_out.squeeze(-1)        # [T, H_p, W_p]

        # If local_points were discarded during raw post-processing and only
        # world_points + poses are available, reconstruct local_points.
        if local_pts is None and world_pts is not None and cam_poses is not None:
            T_cw = torch.inverse(cam_poses)  # [T, 4, 4]
            ones = torch.ones_like(world_pts[..., :1])
            world_h = torch.cat([world_pts, ones], dim=-1)  # [T, H, W, 4]
            local_pts = torch.einsum("tij, thwj -> thwi", T_cw, world_h)[..., :3]

        # --- Build PatchMeta: [L_patch, 3] --------------------------------
        # Each patch token is indexed by (frame_idx, y_tok, x_tok).
        frame_ids = torch.arange(T).unsqueeze(1).unsqueeze(2).expand(T, patch_h, patch_w)
        y_ids = torch.arange(patch_h).unsqueeze(0).unsqueeze(2).expand(T, patch_h, patch_w)
        x_ids = torch.arange(patch_w).unsqueeze(0).unsqueeze(1).expand(T, patch_h, patch_w)
        patch_meta = torch.stack(
            [frame_ids.reshape(-1), y_ids.reshape(-1), x_ids.reshape(-1)], dim=1
        ).long()  # [L_patch, 3]

        # --- Build TokenType: [L_tok] ------------------------------------
        # Per-frame layout: [reg*5, role*1, patch*patch_h*patch_w]
        patches_per_frame = patch_h * patch_w
        L_patch = T * patches_per_frame
        L_special = T * SPECIAL_TOKENS_PER_FRAME
        L_tok = L_patch + L_special

        token_type_per_frame = torch.cat([
            torch.full((NUM_REGISTER_TOKENS,), TOKEN_TYPE_REGISTER, dtype=torch.long),
            torch.full((NUM_ROLE_TOKENS,), TOKEN_TYPE_ROLE, dtype=torch.long),
            torch.full((patches_per_frame,), TOKEN_TYPE_PATCH, dtype=torch.long),
        ])  # [tokens_per_frame]
        token_type = token_type_per_frame.repeat(T)  # [L_tok]

        # --- Raw predictions dict for debug --------------------------------
        raw_cpu = {}
        for k, v in raw.items():
            if v is not None and torch.is_tensor(v):
                raw_cpu[k] = v.squeeze(0).detach().cpu().float()

        return GeometryOutput(
            local_points=local_pts,
            world_points=world_pts,
            camera_poses=cam_poses,
            confidence=conf_out,
            frame_attention_prior=frame_attn,
            attn_dynamic_patch=attn_dynamic_patch,
            dyn4d_patch=dyn4d_patch,
            dyn4d_qq_mean_patch=dyn4d_qq_mean_patch,
            dyn4d_qk_var_patch=dyn4d_qk_var_patch,
            dyn4d_kk_mean_patch=dyn4d_kk_mean_patch,
            global_q_raw_patchvec=global_q_raw_patchvec,
            global_k_raw_patchvec=global_k_raw_patchvec,
            global_q_raw_patchvec_layers=global_q_raw_patchvec_layers,
            global_k_raw_patchvec_layers=global_k_raw_patchvec_layers,
            dyn4d_global_layer_ids=dyn4d_global_layer_ids,
            frame_attn_cosine_shallow=frame_attn_cosine_shallow,
            frame_attn_cosine_deep=frame_attn_cosine_deep,
            frame_attn_cosine_avg=frame_attn_cosine_avg,
            frame_attn_key_cosine_l0=frame_attn_key_cosine_l0,
            frame_attn_key_cosine_l4=frame_attn_key_cosine_l4,
            frame_attn_key_cosine_shallow=frame_attn_key_cosine_shallow,
            frame_attn_key_cosine_deep=frame_attn_key_cosine_deep,
            frame_attn_key_cosine_avg=frame_attn_key_cosine_avg,
            frame_attn_cosine_query_layers=frame_attn_cosine_query_layers,
            frame_attn_cosine_key_layers=frame_attn_cosine_key_layers,
            frame_attn_cosine_layer_ids=frame_attn_cosine_layer_ids,
            patch_meta=patch_meta,
            token_type=token_type,
            num_frames=T,
            pointmap_resolution=(world_pts.shape[1], world_pts.shape[2]) if world_pts is not None else (0, 0),
            patch_grid=(patch_h, patch_w),
            raw_predictions=raw_cpu,
        )
