"""
Stage E: TTT Write Controller

Takes the token-level write prior ``A_tok`` (from Stage D) and the
``WriteCacheOutput`` (from Stage A) to perform a **delayed write-back**
of TTT fast weights:

    W_m  →  W_{m+1}

The controller replays the TTT update loop with modified per-token
learning rates (lr_new = A_tok · lr_original), so that low-prior tokens
contribute less to the fast-weight update.  It also applies an optional
block-level write gain that scales the entire update direction based on
the chunk-level suppression ratio.

Phase 1: deterministic replay with token-level prior weighting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

from loger.models.ttt import fast_weight_replay_update
from .geometry_backbone import WriteCacheOutput, TTTLayerCache


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------
@dataclass
class WriteResult:
    """Output of the TTT Write Controller — the committed W_{m+1}."""

    w0: List[Optional[torch.Tensor]]    # per-layer branch-0 weights
    w1: List[Optional[torch.Tensor]]    # per-layer branch-1 weights
    w2: List[Optional[torch.Tensor]]    # per-layer branch-2 weights
    history: Optional[List[Optional[Dict[str, torch.Tensor]]]] = None

    debug: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core controller
# ---------------------------------------------------------------------------
class TTTWriteController:
    """Stage E of the Semantic Prior Pipeline.

    Usage::

        ctrl = TTTWriteController()
        result = ctrl.run(write_cache, A_tok, device="cuda")
        # result.w0/w1/w2 is W_{m+1} — pass to next chunk's Stage A
    """

    def __init__(
        self,
        *,
        lambda_min: float = 0.0,
        lambda_max: float = 1.0,
        device: str = "cuda",
        write_mode: str = "semantic",
    ):
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.device = device
        self.write_mode = write_mode

    # -- public API --------------------------------------------------------

    def run(
        self,
        write_cache: WriteCacheOutput,
        A_tok: Optional[torch.Tensor],
        B_chunk_geo: Optional[float] = None,
        device: Optional[str] = None,
    ) -> WriteResult:
        """Perform delayed write-back: W_m → W_{m+1}.

        Parameters
        ----------
        write_cache :
            ``WriteCacheOutput`` from Stage A containing per-layer
            cached primitives and the old weights W_m.
        A_tok :
            Token-level write prior from Stage D, shape ``[L_tok]``.
            Values in [0, 1]; higher means more write-allowed.
        device :
            Device to run the replay on.  Defaults to ``self.device``.

        Returns
        -------
        WriteResult
            ``w0``, ``w1``, ``w2`` lists ready to be fed as
            ``ttt_state_input`` to the next chunk's Geometry Backbone.
        """
        mode = self.write_mode
        dev = device or self.device
        n_layers = write_cache.num_ttt_layers

        history = write_cache.history_provisional

        if mode == "native":
            debug_info = {
                "mode": mode,
                "native_write_through": True,
            }
            return WriteResult(
                w0=list(write_cache.w0_provisional),
                w1=list(write_cache.w1_provisional),
                w2=list(write_cache.w2_provisional),
                history=history,
                debug=debug_info,
            )

        if mode not in {"semantic", "unity_replay"}:
            raise ValueError(f"Unsupported write_mode: {mode}")

        w0_new: List[Optional[torch.Tensor]] = [None] * n_layers
        w1_new: List[Optional[torch.Tensor]] = [None] * n_layers
        w2_new: List[Optional[torch.Tensor]] = [None] * n_layers

        debug_info: Dict[str, Any] = {"mode": mode}

        for li, lc in enumerate(write_cache.layer_caches):
            if mode == "unity_replay":
                effective_prior = None
                effective_budget = 1.0
            else:
                effective_prior = A_tok
                effective_budget = B_chunk_geo
            w0_li, w1_li, w2_li, layer_debug = self._replay_layer(
                lc, effective_prior, effective_budget, dev,
            )
            w0_new[li] = w0_li
            w1_new[li] = w1_li
            w2_new[li] = w2_li
            debug_info[f"layer_{li}"] = layer_debug

        return WriteResult(
            w0=w0_new,
            w1=w1_new,
            w2=w2_new,
            history=history,
            debug=debug_info,
        )

    # -- per-layer replay --------------------------------------------------

    def _replay_layer(
        self,
        lc: TTTLayerCache,
        A_tok: Optional[torch.Tensor],
        B_chunk_geo: Optional[float],
        device: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Replay one TTT layer's update with prior-weighted lr."""
        # Move cached tensors to device.  The original forward mixes
        # bfloat16 (q/k/v inside autocast) and float32 (weights, lr).
        # We cast everything to bfloat16 for the matmul-heavy replay
        # (matching the original autocast context), except lr which
        # stays float32 as in the original code.
        compute_dtype = torch.bfloat16
        k = lc.k.to(device=device, dtype=compute_dtype)
        v = lc.v.to(device=device, dtype=compute_dtype)
        lr0 = lc.lr0.to(device=device, dtype=torch.float32)
        lr1 = lc.lr1.to(device=device, dtype=torch.float32)
        lr2 = lc.lr2.to(device=device, dtype=torch.float32)
        w0_old = lc.w0_old.to(device=device, dtype=compute_dtype)
        w1_old = lc.w1_old.to(device=device, dtype=compute_dtype)
        w2_old = lc.w2_old.to(device=device, dtype=compute_dtype)
        momentum = lc.momentum.to(device=device, dtype=torch.float32) if lc.momentum is not None else None

        # Build token_prior: shape [1, l, 1] to broadcast over batch*heads
        # A_tok is [L_tok] but cached k is [b*h, l, d] where l is only the
        # tokens that went through this TTT layer (patch tokens from the
        # window processed by decode).  We need to align.
        l = k.shape[1]
        if A_tok is None:
            prior_flat = torch.ones(l)
        elif A_tok.shape[0] >= l:
            prior_flat = A_tok[:l]
        else:
            prior_flat = torch.ones(l)
            prior_flat[:A_tok.shape[0]] = A_tok
        token_prior = prior_flat.to(device).unsqueeze(0).unsqueeze(-1)  # [1, l, 1]

        mean_prior = prior_flat.mean().item()
        budget_geo = float(B_chunk_geo) if B_chunk_geo is not None else mean_prior
        lam = self.lambda_min + (self.lambda_max - self.lambda_min) * budget_geo

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            w0_new, w1_new, w2_new = fast_weight_replay_update(
                w0_old, w1_old, w2_old,
                k, v,
                lr0 * lam, lr1 * lam, lr2 * lam,
                token_prior,
                lc.ttt_op_order,
                muon_update_steps=lc.muon_update_steps,
                momentum=momentum,
                ttt_update_steps=lc.ttt_update_steps,
            )

        debug = {
            "mean_prior": mean_prior,
            "budget_geo": budget_geo,
            "lambda_write": lam,
        }

        return w0_new.cpu(), w1_new.cpu(), w2_new.cpu(), debug
