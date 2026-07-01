import torch
import torch.nn as nn
import torch.nn.functional as F
import csv
import math
from functools import partial
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Union, List, Tuple

from .dinov2.layers import Mlp
from ..utils.geometry import homogenize_points, robust_scale_estimation
from .layers.pos_embed import RoPE2D, PositionGetter
from .layers.block import BlockRope
from .layers.attention import FlashAttentionRope
from .layers.transformer_head import TransformerDecoder, LinearPts3d, ContextOnlyTransformerDecoder
from .layers.camera_head import CameraHead
from .layers.conv_head import ConvHead
from .dinov2.hub.backbones import dinov2_vitl14, dinov2_vitl14_reg
from huggingface_hub import PyTorchModelHubMixin
from loger.models.ttt import FastWeightGluMLPMultihead, TTTOperator

_CONTEXT_SKY_FINE_LABEL_IDS = {20, 26}
_CONTEXT_VEGETATION_FINE_LABEL_IDS = {21, 22, 23, 24, 27, 28}
_CONTEXT_GROUND_FINE_LABEL_IDS = {1, 2, 8, 10, 11}
_CONTEXT_VERTICAL_STATIC_FINE_LABEL_IDS = {3, 4, 5, 6, 7, 12, 13, 15}
_CONTEXT_SEM_GROUP_STRUCTURE = 0
_CONTEXT_SEM_GROUP_STATIC = 1
_CONTEXT_SEM_GROUP_MOVABLE = 2
_CONTEXT_SEM_GROUP_LOWSTUFF = 3
_CONTEXT_SEM_GROUP_UNCERTAIN = 4
_SEMANTIC_ROLE_FALLBACK = 0
_SEMANTIC_ROLE_POSITIVE_LONG = 1
_SEMANTIC_ROLE_NEUTRAL_KEEP = 2
_SEMANTIC_ROLE_NEGATIVE_SHORT = 3
_SEMANTIC_ROLE_PROTECT_NEUTRAL = 4
_V102_STATE_MACHINE_ACTIONS = {
    "TRANSMIT_SUPPORTED_ANCHORS",
    "REJECT_UNRELIABLE_ANCHORS",
    "DELAY_UPDATE",
    "HOLD_PREV_REFERENCE",
    "CONTEXT_ONLY_DEMOTION",
    "WRITE_CONFIRMED_ANCHORS_ONLY",
    "EXPIRE_UNSUPPORTED_STALE_ANCHORS",
    "REFRESH_SUPPORTED_STALE_ANCHORS",
    "WRITE_CONTEXT_ONLY",
}

class Pi3(nn.Module, PyTorchModelHubMixin):
    def __init__(
            self,
            pos_type='rope100',
            decoder_size='large',
            ttt_insert_after: Union[int, List[int]] = None,
            ttt_head_dim: int = 512,
            ttt_inter_multi: int = 2,
            num_muon_update_steps: int = 5,
            use_momentum: bool = False,
            ttt_update_steps: int = 1,
            conf: bool = True,
            attn_insert_after: Union[int, List[int], None] = None,
            feature_frame_attn_layers: Union[int, List[int], None] = None,
            feature_global_attn_layers: Union[int, List[int], None] = None,
            dyn4d_window_radius: int = 2,
            export_attn_debug: bool = False,
            ttt_pre_norm: bool = False,
            pi3x: bool = False,
            pi3x_metric: bool = True,
        ):
        super().__init__()

        # ----------------------
        #        Encoder
        # ----------------------
        def _normalize_insert_positions(value: Union[int, List[int], None]) -> List[int]:
            if isinstance(value, (int, float)):
                return [int(value)]
            if isinstance(value, (list, tuple)):
                return [int(x) for x in value]
            return []

        parsed_ttt_insert_after = _normalize_insert_positions(ttt_insert_after)
        parsed_attn_insert_after = _normalize_insert_positions(attn_insert_after)
        parsed_feature_frame_attn_layers = _normalize_insert_positions(feature_frame_attn_layers)
        parsed_feature_global_attn_layers = _normalize_insert_positions(feature_global_attn_layers)

        if not parsed_attn_insert_after:
            parsed_attn_insert_after = parsed_ttt_insert_after.copy()

        self.ttt_insert_after = parsed_ttt_insert_after
        self.attn_insert_after = parsed_attn_insert_after
        self.export_attn_debug = bool(export_attn_debug)
        self.export_full_pca_debug = False
        self.pca_debug_max_feature_dim = 64
        self.detach_swa_history = False
        self.initialize_swa_from_global = True
        self.encoder = dinov2_vitl14_reg(pretrained=False)
        self.patch_size = 14
        self.num_muon_update_steps = int(num_muon_update_steps)
        self.num_pe_tokens = 3
        self.use_momentum = use_momentum
        self.ttt_update_steps = int(ttt_update_steps)
        self.use_conf = bool(conf)
        self.ttt_pre_norm = ttt_pre_norm
        self.pi3x = pi3x
        self.pi3x_metric = pi3x_metric
        del self.encoder.mask_token

        # ----------------------
        #  Positonal Encoding
        # ----------------------
        self.pos_type = pos_type if pos_type is not None else 'none'
        self.rope=None
        if self.pos_type.startswith('rope'): # eg rope100 
            if RoPE2D is None: raise ImportError("Cannot find cuRoPE2D, please install it following the README instructions")
            freq = float(self.pos_type[len('rope'):])
            self.rope = RoPE2D(freq=freq)
            self.position_getter = PositionGetter()
        else:
            raise NotImplementedError
        

        # ----------------------
        #        Decoder
        # ----------------------
        enc_embed_dim = self.encoder.blocks[0].attn.qkv.in_features        # 1024
        if decoder_size == 'small':
            dec_embed_dim = 384
            dec_num_heads = 6
            mlp_ratio = 4
            dec_depth = 24
        elif decoder_size == 'base':
            dec_embed_dim = 768
            dec_num_heads = 12
            mlp_ratio = 4
            dec_depth = 24
        elif decoder_size == 'large':
            dec_embed_dim = 1024
            dec_num_heads = 16
            mlp_ratio = 4
            dec_depth = 36
        else:
            raise NotImplementedError
        self.decoder = nn.ModuleList([
            BlockRope(
                dim=dec_embed_dim,
                num_heads=dec_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                drop_path=0.0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU,
                ffn_layer=Mlp,
                init_values=0.01,
                qk_norm=True,
                attn_class=FlashAttentionRope,
                rope=self.rope
            ) for _ in range(dec_depth)])
        self.dec_embed_dim = dec_embed_dim
        self.attn_prior_layers = self._resolve_attn_prior_layers(
            len(self.decoder), self.attn_insert_after,
        )
        self.frame_attn_map_layers = self._resolve_frame_attention_map_layers(
            len(self.decoder), self.attn_insert_after,
        )
        self.feature_frame_attn_layers = self._resolve_feature_frame_attention_layers(
            len(self.decoder), parsed_feature_frame_attn_layers,
        )
        self.feature_global_attn_layers = self._resolve_feature_global_attention_layers(
            len(self.decoder), parsed_feature_global_attn_layers,
        )
        self.dyn4d_window_radius = max(int(dyn4d_window_radius), 1)
        self.all_frame_attn_layers = [idx for idx in range(len(self.decoder)) if idx % 2 == 0]

        # ----------------------
        #     Register_token
        # ----------------------
        num_register_tokens = 5
        self.patch_start_idx = num_register_tokens
        self.register_token = nn.Parameter(torch.randn(1, 1, num_register_tokens, self.dec_embed_dim))
        nn.init.normal_(self.register_token, std=1e-6)

        for i in range(3):
            pe_token = nn.Parameter(torch.randn(1, 1, 1, self.dec_embed_dim))
            nn.init.normal_(pe_token, std=1e-6)
            self.register_parameter(f'pe_token_{i}', pe_token)
        self.patch_start_idx += 1

        # ----------------------
        #  Local Points Decoder
        # ----------------------
        self.point_decoder = TransformerDecoder(
            in_dim=2*self.dec_embed_dim, 
            dec_embed_dim=1024,
            dec_num_heads=16,
            out_dim=1024,
            rope=self.rope,
        )
        if self.pi3x:
            self.point_head = ConvHead(
                num_features=4, 
                dim_in=1024,
                projects=nn.Identity(),
                dim_out=[2, 1], 
                dim_proj=1024,
                dim_upsample=[256, 128, 64],
                dim_times_res_block_hidden=2,
                num_res_blocks=2,
                res_block_norm='group_norm',
                last_res_blocks=0,
                last_conv_channels=32,
                last_conv_size=1,
                using_uv=True
            )
        else:
            self.point_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=3)

        # ----------------------
        #     Conf Decoder
        # ----------------------
        if self.use_conf:
            self.conf_decoder = deepcopy(self.point_decoder)
            self.conf_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=1)
        else:
            self.conf_decoder = None
            self.conf_head = None

        # ----------------------
        #     Metric Decoder
        # ----------------------
        if self.pi3x and self.pi3x_metric:
            self.metric_token = nn.Parameter(torch.randn(1, 1, 2*self.dec_embed_dim))
            self.metric_decoder = ContextOnlyTransformerDecoder(
                in_dim=2*self.dec_embed_dim, 
                dec_embed_dim=512,
                dec_num_heads=8,                # 8
                out_dim=512,
                rope=self.rope,
            )
            self.metric_head = nn.Linear(512, 1)
            nn.init.normal_(self.metric_token, std=1e-6)
        else:
            self.metric_token = None
            self.metric_decoder = None
            self.metric_head = None

        # ----------------------
        #  Camera Pose Decoder
        # ----------------------
        self.camera_decoder = TransformerDecoder(
            in_dim=2*self.dec_embed_dim, 
            dec_embed_dim=1024,
            dec_num_heads=16,                # 8
            out_dim=512,
            rope=self.rope,
            use_checkpoint=False
        )
        self.camera_head = CameraHead(dim=512, output_quat=False)

        # For ImageNet Normalize
        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        self.register_buffer("image_mean", image_mean)
        self.register_buffer("image_std", image_std)

        # ----------------------
        #            TTT
        # ----------------------

        self.ttt_layers = None
        self.ttt_gate_projs = None
        self.ttt_op_order = None

        self.ttt_layers = nn.ModuleList([
            FastWeightGluMLPMultihead(
                dim=dec_embed_dim,
                head_dim=ttt_head_dim,
                inter_multi=ttt_inter_multi,
                bias=False,
                base_lr=0.01,
                muon_update_steps=self.num_muon_update_steps,
                use_momentum=self.use_momentum,
                ttt_update_steps=self.ttt_update_steps,
                ttt_pre_norm=self.ttt_pre_norm,
            )
            for _ in self.ttt_insert_after
        ])
        self.ttt_gate_projs = nn.ModuleList([
            nn.Linear(dec_embed_dim, 1)
            for _ in self.ttt_insert_after
        ])

        for gate_proj in self.ttt_gate_projs:
            torch.nn.init.zeros_(gate_proj.weight)
            if gate_proj.bias is not None:
                torch.nn.init.zeros_(gate_proj.bias)

        self.ttt_op_order = [
            TTTOperator(start=0, end=None, update=False, apply=True),
            TTTOperator(start=0, end=None, update=True, apply=False),
        ]

        # ----------------------
        #   Attention Adapters
        # ----------------------
        self.swa_layers = nn.ModuleList([
            BlockRope(
                dim=dec_embed_dim,
                num_heads=dec_num_heads,
                mlp_ratio=ttt_inter_multi,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                drop_path=0.0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU,
                ffn_layer=Mlp,
                init_values=0.01,
                qk_norm=True,
                attn_class=FlashAttentionRope,
                rope=self.rope,
            )
            for _ in self.attn_insert_after
        ])
        self.swa_gate_projs = nn.ModuleList([
            nn.Linear(dec_embed_dim, 1)
            for _ in self.attn_insert_after
        ])

        for gate_proj in self.swa_gate_projs:
            torch.nn.init.zeros_(gate_proj.weight)
            if gate_proj.bias is not None:
                torch.nn.init.zeros_(gate_proj.bias)
    
    def _initialize_ttt_layers_from_global(
        self,
        layers: Optional[nn.ModuleList],
        kind: str,
        insert_after: Optional[List[int]] = None,
    ) -> None:
        """Helper for initializing adapter layers from decoder global attention weights."""
        if layers is None or len(layers) == 0:
            print(f"{kind} initialization skipped: no target layers defined.")
            return

        insert_positions = insert_after if insert_after is not None else self.ttt_insert_after
        if not insert_positions:
            print(f"{kind} initialization skipped: no insert positions defined.")
            return

        num_decoder_layers = len(self.decoder)
        print(f"Initializing {len(layers)} {kind} layers from decoder attention blocks")
        print(f"  Insert positions: {insert_positions}")


        for layer_idx, insert_idx in enumerate(insert_positions):
            decoder_idx = int(insert_idx)
            if decoder_idx % 2 == 0:
                decoder_idx += 1  # move to the subsequent global-attention layer

            if decoder_idx >= num_decoder_layers:
                raise IndexError(
                    f"Decoder index {decoder_idx} out of range for {kind} initialization (decoder has {num_decoder_layers} layers)."
                )

            if decoder_idx % 2 == 0:
                raise AssertionError(
                    f"Decoder index {decoder_idx} is not a global-attention layer after adjustment."
                )

            source_layer = self.decoder[decoder_idx]
            target_layer = layers[layer_idx]
            target_layer.load_state_dict(source_layer.state_dict())

            print(f"  Initialized {kind}_layer[{layer_idx}] from decoder[{decoder_idx}]")

    def _initialize_swa_from_global(self):
        if self.swa_layers is None:
            return
        self._initialize_ttt_layers_from_global(self.swa_layers, "swa", self.attn_insert_after)

    @staticmethod
    def _resolve_attn_prior_layers(
        num_decoder_layers: int,
        insert_after: Optional[List[int]],
    ) -> List[int]:
        """Resolve decoder blocks used for attention-prior extraction.

        The config stores adapter insertion points, which may target frame
        attention blocks. For motion priors we prefer nearby global-attention
        layers, so even indices are shifted to the subsequent odd layer.
        """
        resolved: List[int] = []
        for insert_idx in insert_after or []:
            layer_idx = int(insert_idx)
            if layer_idx % 2 == 0:
                layer_idx += 1
            if 0 <= layer_idx < num_decoder_layers and layer_idx % 2 == 1:
                resolved.append(layer_idx)

        if resolved:
            return sorted(set(resolved))

        fallback = [idx for idx in range(num_decoder_layers) if idx % 2 == 1]
        return fallback[-4:] if len(fallback) > 4 else fallback

    @staticmethod
    def _resolve_frame_attention_map_layers(
        num_decoder_layers: int,
        insert_after: Optional[List[int]],
    ) -> List[int]:
        """Resolve frame-attention layers for MUT3R-style visualization."""
        resolved: List[int] = []
        for insert_idx in insert_after or []:
            layer_idx = int(insert_idx)
            if layer_idx % 2 == 1:
                layer_idx -= 1
            if 0 <= layer_idx < num_decoder_layers and layer_idx % 2 == 0:
                resolved.append(layer_idx)

        if resolved:
            return sorted(set(resolved))

        even_layers = [idx for idx in range(num_decoder_layers) if idx % 2 == 0]
        if len(even_layers) <= 4:
            return even_layers

        sample_ids = [
            0,
            len(even_layers) // 3,
            (2 * len(even_layers)) // 3,
            len(even_layers) - 1,
        ]
        return sorted({even_layers[idx] for idx in sample_ids})

    @staticmethod
    def _resolve_feature_frame_attention_layers(
        num_decoder_layers: int,
        layers: Optional[List[int]],
    ) -> List[int]:
        """Resolve default frame-attention layers used for Stage-A features."""
        default_layers = [0, 2, 4, 6, 8, 10, 12, 14]
        candidates = layers if layers else default_layers
        resolved: List[int] = []
        for layer_idx in candidates:
            layer_idx = int(layer_idx)
            if layer_idx % 2 == 1:
                layer_idx -= 1
            if 0 <= layer_idx < num_decoder_layers and layer_idx % 2 == 0:
                resolved.append(layer_idx)
        return sorted(set(resolved))

    @staticmethod
    def _resolve_feature_global_attention_layers(
        num_decoder_layers: int,
        layers: Optional[List[int]],
    ) -> List[int]:
        """Resolve global-attention layers used for VGGT4D-style 4D dynamic cues."""
        if layers:
            candidates = layers
        else:
            candidates = [idx for idx in range(num_decoder_layers) if idx % 2 == 1]

        resolved: List[int] = []
        for layer_idx in candidates:
            layer_idx = int(layer_idx)
            if layer_idx % 2 == 0:
                layer_idx += 1
            if 0 <= layer_idx < num_decoder_layers and layer_idx % 2 == 1:
                resolved.append(layer_idx)
        return sorted(set(resolved))

    @staticmethod
    def _split_dyn4d_global_layer_groups(
        layers: List[int],
    ) -> Tuple[List[int], List[int], List[int]]:
        """Split selected global-attention layers into shallow/middle/deep groups."""
        if not layers:
            return [], [], []

        layers = sorted(set(int(layer) for layer in layers))
        num_layers = len(layers)
        if num_layers == 1:
            return layers[:], layers[:], layers[:]
        if num_layers == 2:
            return [layers[0]], [layers[0]], [layers[1]]

        split_1 = max(1, num_layers // 3)
        split_2 = max(split_1 + 1, (2 * num_layers) // 3)
        split_2 = min(split_2, num_layers - 1)
        shallow = layers[:split_1]
        middle = layers[split_1:split_2]
        deep = layers[split_2:]
        if not middle:
            middle = shallow[-1:]
        if not deep:
            deep = middle[-1:]
        return shallow, middle, deep

    def _pca_debug_enabled(self) -> bool:
        return bool(getattr(self, "export_full_pca_debug", False))

    def _pca_feature_limit(self) -> int:
        try:
            return max(0, int(getattr(self, "pca_debug_max_feature_dim", 64)))
        except Exception:
            return 64

    def _pca_truncate_feature_dim(self, x: torch.Tensor) -> torch.Tensor:
        max_dim = self._pca_feature_limit()
        if max_dim > 0 and int(x.shape[-1]) > max_dim:
            x = x[..., :max_dim]
        return x.contiguous()

    def _pca_tokens_to_patchvec(
        self,
        tokens: Optional[torch.Tensor],
        *,
        batch_size: int,
        frame_num: int,
        patch_h: int,
        patch_w: int,
    ) -> Optional[torch.Tensor]:
        if tokens is None or tokens.ndim != 4:
            return None
        if int(tokens.shape[0]) != int(batch_size) or int(tokens.shape[1]) != int(frame_num):
            return None
        patch_tokens = int(patch_h) * int(patch_w)
        start = int(self.patch_start_idx)
        end = start + patch_tokens
        if int(tokens.shape[2]) < end:
            return None
        x = tokens[:, :, start:end, :]
        x = x.reshape(batch_size, frame_num, patch_h, patch_w, int(x.shape[-1]))
        return self._pca_truncate_feature_dim(x.detach())

    def _pca_heads_to_patchvec(
        self,
        heads_tokens: Optional[torch.Tensor],
        *,
        batch_size: int,
        frame_num: int,
        tokens_per_frame: int,
        patch_h: int,
        patch_w: int,
        layout: str,
    ) -> Optional[torch.Tensor]:
        if heads_tokens is None or heads_tokens.ndim != 4:
            return None
        if layout == "frame":
            expected_batch = int(batch_size) * int(frame_num)
            expected_tokens = int(tokens_per_frame)
            if int(heads_tokens.shape[0]) != expected_batch or int(heads_tokens.shape[2]) != expected_tokens:
                return None
            x = heads_tokens.transpose(1, 2).reshape(
                batch_size,
                frame_num,
                tokens_per_frame,
                -1,
            )
        else:
            expected_tokens = int(frame_num) * int(tokens_per_frame)
            if int(heads_tokens.shape[0]) != int(batch_size) or int(heads_tokens.shape[2]) != expected_tokens:
                return None
            x = heads_tokens.transpose(1, 2).reshape(
                batch_size,
                frame_num,
                tokens_per_frame,
                -1,
            )
        return self._pca_tokens_to_patchvec(
            x,
            batch_size=batch_size,
            frame_num=frame_num,
            patch_h=patch_h,
            patch_w=patch_w,
        )

    def _extract_pca_attention_qkv_patchvec(
        self,
        blk: nn.Module,
        x: torch.Tensor,
        xpos: Optional[torch.Tensor],
        *,
        batch_size: int,
        frame_num: int,
        tokens_per_frame: int,
        patch_h: int,
        patch_w: int,
        layout: str,
    ) -> Optional[Dict[str, torch.Tensor]]:
        if not self._pca_debug_enabled():
            return None
        try:
            with torch.no_grad():
                x_norm = blk.norm1(x)
                Bx, Nt, C = x_norm.shape
                attn = blk.attn
                qkv = attn.qkv(x_norm).reshape(
                    Bx,
                    Nt,
                    3,
                    attn.num_heads,
                    C // attn.num_heads,
                ).transpose(1, 3)
                q, k, v = [qkv[:, :, idx] for idx in range(3)]
                q = attn.q_norm(q).to(v.dtype)
                k = attn.k_norm(k).to(v.dtype)
                if attn.rope is not None and xpos is not None:
                    q = attn.rope(q, xpos)
                    k = attn.rope(k, xpos)
                return {
                    "q": self._pca_heads_to_patchvec(
                        q,
                        batch_size=batch_size,
                        frame_num=frame_num,
                        tokens_per_frame=tokens_per_frame,
                        patch_h=patch_h,
                        patch_w=patch_w,
                        layout=layout,
                    ),
                    "k": self._pca_heads_to_patchvec(
                        k,
                        batch_size=batch_size,
                        frame_num=frame_num,
                        tokens_per_frame=tokens_per_frame,
                        patch_h=patch_h,
                        patch_w=patch_w,
                        layout=layout,
                    ),
                    "v": self._pca_heads_to_patchvec(
                        v,
                        batch_size=batch_size,
                        frame_num=frame_num,
                        tokens_per_frame=tokens_per_frame,
                        patch_h=patch_h,
                        patch_w=patch_w,
                        layout=layout,
                    ),
                }
        except Exception:
            return None

    def _extract_pca_swa_current_qkv_patchvec(
        self,
        swa_layer: nn.Module,
        x_flat: torch.Tensor,
        xpos: Optional[torch.Tensor],
        *,
        batch_size: int,
        frame_num: int,
        tokens_per_frame: int,
        patch_h: int,
        patch_w: int,
    ) -> Optional[Dict[str, torch.Tensor]]:
        if not self._pca_debug_enabled():
            return None
        return self._extract_pca_attention_qkv_patchvec(
            swa_layer,
            x_flat,
            xpos,
            batch_size=batch_size,
            frame_num=frame_num,
            tokens_per_frame=tokens_per_frame,
            patch_h=patch_h,
            patch_w=patch_w,
            layout="global",
        )

    def _pca_ttt_heads_to_patchvec(
        self,
        heads_tokens: Optional[torch.Tensor],
        *,
        batch_size: int,
        frame_num: int,
        tokens_per_frame: int,
        patch_h: int,
        patch_w: int,
    ) -> Optional[torch.Tensor]:
        if heads_tokens is None or heads_tokens.ndim != 3:
            return None
        if int(heads_tokens.shape[0]) % int(batch_size) != 0:
            return None
        if int(heads_tokens.shape[1]) != int(frame_num) * int(tokens_per_frame):
            return None
        heads = int(heads_tokens.shape[0]) // int(batch_size)
        x = heads_tokens.reshape(
            batch_size,
            heads,
            frame_num,
            tokens_per_frame,
            int(heads_tokens.shape[-1]),
        ).permute(0, 2, 3, 1, 4).reshape(
            batch_size,
            frame_num,
            tokens_per_frame,
            heads * int(heads_tokens.shape[-1]),
        )
        return self._pca_tokens_to_patchvec(
            x,
            batch_size=batch_size,
            frame_num=frame_num,
            patch_h=patch_h,
            patch_w=patch_w,
        )

    def _extract_frame_attention_cosine_map(
        self,
        blk: nn.Module,
        x: torch.Tensor,
        xpos: Optional[torch.Tensor],
        batch_size: int,
        frame_num: int,
        patch_h: int,
        patch_w: int,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Extract a MUT3R-style patch response map from frame attention.

        We only keep patch tokens and replace the expensive attention matrix
        with the average q/k cosine similarity over keys:

            mean_j cos(q_i, k_j) = q_i · mean_j(k_j)

        and its key-side counterpart:

            mean_j cos(q_j, k_i) = mean_j(q_j) · k_i

        This keeps the patch-only frame-attention semantics while avoiding
        materializing the full [P, P] score matrix.
        """
        if frame_num <= 0:
            return None, None

        batch_frames, total_tokens, dim = x.shape
        if batch_frames != batch_size * frame_num:
            return None, None

        num_patch_tokens = total_tokens - self.patch_start_idx
        if num_patch_tokens <= 0 or num_patch_tokens != patch_h * patch_w:
            return None, None

        x_patch = x[:, self.patch_start_idx:, :]
        pos_patch = xpos[:, self.patch_start_idx:, :] if xpos is not None else None

        x_norm = blk.norm1(x_patch)
        qkv = blk.attn.qkv(x_norm).reshape(
            batch_frames,
            num_patch_tokens,
            3,
            blk.attn.num_heads,
            dim // blk.attn.num_heads,
        ).transpose(1, 3)
        q, k, v = [qkv[:, :, i] for i in range(3)]
        q = blk.attn.q_norm(q).to(v.dtype)
        k = blk.attn.k_norm(k).to(v.dtype)

        if blk.attn.rope is not None and pos_patch is not None:
            q = blk.attn.rope(q, pos_patch)
            k = blk.attn.rope(k, pos_patch)

        q = F.normalize(q.float(), dim=-1)
        k = F.normalize(k.float(), dim=-1)

        query_centroid = q.mean(dim=2)
        key_centroid = k.mean(dim=2)
        query_response = (q * key_centroid.unsqueeze(2)).sum(dim=-1).mean(dim=1)
        key_response = (k * query_centroid.unsqueeze(2)).sum(dim=-1).mean(dim=1)

        def _normalize_response(response: torch.Tensor) -> torch.Tensor:
            response_mean = response.mean(dim=-1, keepdim=True)
            response_std = response.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
            response = torch.sigmoid((response - response_mean) / (2.0 * response_std))
            return response.reshape(batch_size, frame_num, patch_h, patch_w).clamp(0.0, 1.0)

        return _normalize_response(query_response), _normalize_response(key_response)

    def _extract_attention_prior_from_block(
        self,
        blk: nn.Module,
        x: torch.Tensor,
        xpos: Optional[torch.Tensor],
        frame_num: int,
        tokens_per_frame: int,
        patch_h: int,
        patch_w: int,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Summarize one global-attention layer into frame/patch priors.

        Returns
        -------
        frame_affinity : [B, T, T] or None
            Symmetric chunk-internal frame affinity in [0, 1].
        token_dynamic : [B, T, H_tok, W_tok] or None
            Patch-level dynamicness prior in [0, 1], where larger means
            less attention support from other frames.
        """
        if frame_num <= 1:
            return None, None

        batch_size, total_tokens, dim = x.shape
        if frame_num * tokens_per_frame != total_tokens:
            return None, None

        num_patch_tokens = tokens_per_frame - self.patch_start_idx
        if num_patch_tokens <= 0 or num_patch_tokens != patch_h * patch_w:
            return None, None

        x_patch = x.reshape(batch_size, frame_num, tokens_per_frame, dim)
        x_patch = x_patch[:, :, self.patch_start_idx:, :].reshape(
            batch_size, frame_num * num_patch_tokens, dim,
        )

        pos_patch = None
        if xpos is not None:
            pos_patch = xpos.reshape(batch_size, frame_num, tokens_per_frame, -1)
            pos_patch = pos_patch[:, :, self.patch_start_idx:, :].reshape(
                batch_size, frame_num * num_patch_tokens, -1,
            )

        x_norm = blk.norm1(x_patch)
        qkv = blk.attn.qkv(x_norm).reshape(
            batch_size,
            frame_num * num_patch_tokens,
            3,
            blk.attn.num_heads,
            dim // blk.attn.num_heads,
        ).transpose(1, 3)
        q, k, v = [qkv[:, :, i] for i in range(3)]
        q = blk.attn.q_norm(q).to(v.dtype)
        k = blk.attn.k_norm(k).to(v.dtype)

        if blk.attn.rope is not None and pos_patch is not None:
            q = blk.attn.rope(q, pos_patch)
            k = blk.attn.rope(k, pos_patch)

        q = q.reshape(
            batch_size, blk.attn.num_heads, frame_num, num_patch_tokens, -1,
        )
        k = k.reshape(
            batch_size, blk.attn.num_heads, frame_num, num_patch_tokens, -1,
        )

        scale = float(blk.attn.scale)
        # Re-normalize frame centroids. Without this, patch-wise normalized
        # vectors can cancel out during averaging, making centroid norms very
        # small and collapsing the downstream cosine statistics toward 0.
        q_frame = F.normalize(q.mean(dim=3), dim=-1)
        k_frame = F.normalize(k.mean(dim=3), dim=-1)

        frame_logits = scale * torch.einsum("bhtd,bhsd->bhts", q_frame, k_frame)
        frame_probs = torch.softmax(frame_logits, dim=-1).mean(dim=1)
        frame_affinity = 0.5 * (frame_probs + frame_probs.transpose(-1, -2))
        eye = torch.eye(
            frame_num, device=frame_affinity.device, dtype=torch.bool,
        ).unsqueeze(0)
        frame_affinity = frame_affinity.masked_fill(eye, 0.0)
        frame_affinity = frame_affinity / frame_affinity.amax(
            dim=-1, keepdim=True,
        ).clamp_min(1e-6)

        token_logits = scale * torch.einsum("bhtpd,bhsd->bhtps", q, k_frame)
        token_similarity = torch.sigmoid(token_logits.mean(dim=1))
        frame_weights = frame_affinity / frame_affinity.sum(
            dim=-1, keepdim=True,
        ).clamp_min(1e-6)
        token_support = (token_similarity * frame_weights[:, :, None, :]).sum(dim=-1)

        token_support_mean = token_support.mean(dim=(1, 2), keepdim=True)
        token_support_std = token_support.std(
            dim=(1, 2), keepdim=True, unbiased=False,
        ).clamp_min(1e-6)
        token_static = torch.sigmoid(
            (token_support - token_support_mean) / (2.0 * token_support_std),
        )
        token_dynamic = 1.0 - token_static
        token_dynamic = token_dynamic.reshape(batch_size, frame_num, patch_h, patch_w)

        return frame_affinity.clamp(0.0, 1.0), token_dynamic.clamp(0.0, 1.0)

    def _extract_dyn4d_global_stats_from_block(
        self,
        blk: nn.Module,
        x: torch.Tensor,
        xpos: Optional[torch.Tensor],
        frame_num: int,
        tokens_per_frame: int,
        patch_h: int,
        patch_w: int,
        window_radius: int,
    ) -> Optional[dict]:
        """Export raw patch-level q/k/v vectors from one global-attention layer."""
        if frame_num <= 1:
            return None

        batch_size, total_tokens, dim = x.shape
        if frame_num * tokens_per_frame != total_tokens:
            return None

        num_patch_tokens = tokens_per_frame - self.patch_start_idx
        if num_patch_tokens <= 0 or num_patch_tokens != patch_h * patch_w:
            return None

        x_patch = x.reshape(batch_size, frame_num, tokens_per_frame, dim)
        x_patch = x_patch[:, :, self.patch_start_idx:, :].reshape(
            batch_size, frame_num * num_patch_tokens, dim,
        )

        pos_patch = None
        if xpos is not None:
            pos_patch = xpos.reshape(batch_size, frame_num, tokens_per_frame, -1)
            pos_patch = pos_patch[:, :, self.patch_start_idx:, :].reshape(
                batch_size, frame_num * num_patch_tokens, -1,
            )

        x_norm = blk.norm1(x_patch)
        qkv = blk.attn.qkv(x_norm).reshape(
            batch_size,
            frame_num * num_patch_tokens,
            3,
            blk.attn.num_heads,
            dim // blk.attn.num_heads,
        ).transpose(1, 3)
        q, k, v = [qkv[:, :, i] for i in range(3)]
        q = blk.attn.q_norm(q).to(v.dtype)
        k = blk.attn.k_norm(k).to(v.dtype)

        if blk.attn.rope is not None and pos_patch is not None:
            q = blk.attn.rope(q, pos_patch)
            k = blk.attn.rope(k, pos_patch)

        q_raw = q.reshape(
            batch_size, blk.attn.num_heads, frame_num, num_patch_tokens, -1,
        ).float()
        k_raw = k.reshape(
            batch_size, blk.attn.num_heads, frame_num, num_patch_tokens, -1,
        ).float()
        v_raw = v.reshape(
            batch_size, blk.attn.num_heads, frame_num, num_patch_tokens, -1,
        ).float()
        return {
            "q_raw_patchvec": q_raw.mean(dim=1).reshape(batch_size, frame_num, patch_h, patch_w, -1),
            "k_raw_patchvec": k_raw.mean(dim=1).reshape(batch_size, frame_num, patch_h, patch_w, -1),
            "v_raw_patchvec": v_raw.mean(dim=1).reshape(batch_size, frame_num, patch_h, patch_w, -1),
        }

    def _aggregate_dyn4d_from_global_stats(
        self,
        dyn4d_parts: List[Tuple[int, dict]],
    ) -> Optional[dict]:
        """Aggregate raw global q/k into token-level Gram statistics and 4D_dyn.

        Important: Gram statistics are computed per global-attention layer first,
        then averaged across layers. This keeps the computation path closer to
        4DVGGT than averaging q/k vectors across layers before forming Gram stats.
        """
        if not dyn4d_parts:
            return None

        available_layers = sorted(layer_id for layer_id, _ in dyn4d_parts)
        stats_by_layer = {layer_id: stats for layer_id, stats in dyn4d_parts}

        def _collect_stack(layer_ids: List[int], key: str) -> Optional[torch.Tensor]:
            parts = [
                stats_by_layer[layer_id][key]
                for layer_id in layer_ids
                if layer_id in stats_by_layer and key in stats_by_layer[layer_id]
            ]
            if not parts:
                return None
            return torch.stack(parts, dim=1)

        global_q_raw_layers = _collect_stack(available_layers, "q_raw_patchvec")
        global_k_raw_layers = _collect_stack(available_layers, "k_raw_patchvec")
        global_v_raw_layers = _collect_stack(available_layers, "v_raw_patchvec")
        if global_q_raw_layers is None or global_k_raw_layers is None:
            return None

        batch_size, num_layers, frame_num, patch_h, patch_w, dim = global_q_raw_layers.shape
        num_patches = patch_h * patch_w
        q = F.normalize(
            global_q_raw_layers.reshape(batch_size, num_layers, frame_num, num_patches, dim).float(),
            dim=-1,
        )
        k = F.normalize(
            global_k_raw_layers.reshape(batch_size, num_layers, frame_num, num_patches, dim).float(),
            dim=-1,
        )

        qq_sum = torch.zeros(batch_size, num_layers, frame_num, num_patches, device=q.device, dtype=q.dtype)
        kk_sum = torch.zeros_like(qq_sum)
        qk_sum = torch.zeros_like(qq_sum)
        qk_sumsq = torch.zeros_like(qq_sum)
        counts = torch.zeros(1, 1, frame_num, 1, device=q.device, dtype=q.dtype)

        for t in range(frame_num):
            start = max(0, t - self.dyn4d_window_radius)
            end = min(frame_num, t + self.dyn4d_window_radius + 1)
            q_t = q[:, :, t]
            k_t = k[:, :, t]
            for s in range(start, end):
                if s == t:
                    continue
                q_s = q[:, :, s]
                k_s = k[:, :, s]

                qq_scores = torch.matmul(q_t, q_s.transpose(-1, -2))
                qk_scores = torch.matmul(q_t, k_s.transpose(-1, -2))
                kk_scores = torch.matmul(k_t, k_s.transpose(-1, -2))

                qq_sum[:, :, t] += qq_scores.sum(dim=-1)
                kk_sum[:, :, t] += kk_scores.sum(dim=-1)
                qk_sum[:, :, t] += qk_scores.sum(dim=-1)
                qk_sumsq[:, :, t] += qk_scores.square().sum(dim=-1)
                counts[:, :, t] += num_patches

        counts = counts.clamp_min(1.0)
        qq_mean = ((qq_sum / counts) + 1.0) * 0.5
        kk_mean = ((kk_sum / counts) + 1.0) * 0.5
        qk_mean = qk_sum / counts
        qk_var = (qk_sumsq / counts) - qk_mean.square()
        qk_var = qk_var.clamp_min(0.0)

        qq_mean = qq_mean.reshape(batch_size, num_layers, frame_num, patch_h, patch_w).clamp(0.0, 1.0)
        kk_mean = kk_mean.reshape(batch_size, num_layers, frame_num, patch_h, patch_w).clamp(0.0, 1.0)
        qk_var = qk_var.reshape(batch_size, num_layers, frame_num, patch_h, patch_w)

        qq_mean = qq_mean.mean(dim=1)
        kk_mean = kk_mean.mean(dim=1)
        qk_var = qk_var.mean(dim=1)

        qk_var_flat = qk_var.reshape(batch_size, frame_num, -1)
        qk_var_min = qk_var_flat.amin(dim=-1, keepdim=True)
        qk_var_max = qk_var_flat.amax(dim=-1, keepdim=True)
        qk_var_norm = (
            (qk_var_flat - qk_var_min)
            / (qk_var_max - qk_var_min).clamp_min(1e-6)
        ).reshape_as(qk_var).clamp(0.0, 1.0)

        dyn4d_raw = (
            0.35 * (1.0 - qq_mean)
            + 0.40 * qk_var_norm
            + 0.25 * (1.0 - kk_mean)
        ).clamp(0.0, 1.0)
        dyn4d_flat = dyn4d_raw.reshape(batch_size, frame_num, -1)
        dyn4d_min = dyn4d_flat.amin(dim=-1, keepdim=True)
        dyn4d_max = dyn4d_flat.amax(dim=-1, keepdim=True)
        dyn4d_norm = (dyn4d_flat - dyn4d_min) / (dyn4d_max - dyn4d_min).clamp_min(1e-6)
        global_q_raw = global_q_raw_layers.mean(dim=1)
        global_k_raw = global_k_raw_layers.mean(dim=1)
        global_v_raw = global_v_raw_layers.mean(dim=1) if global_v_raw_layers is not None else None
        global_v_layers_out = (
            global_v_raw_layers.permute(0, 2, 1, 3, 4, 5).contiguous().float()
            if global_v_raw_layers is not None
            else None
        )
        return {
            "dyn4d_patch": dyn4d_norm.reshape_as(dyn4d_raw).clamp(0.0, 1.0),
            "dyn4d_qq_mean_patch": qq_mean,
            "dyn4d_qk_var_patch": qk_var_norm,
            "dyn4d_kk_mean_patch": kk_mean,
            "global_q_raw_patchvec": global_q_raw.float(),
            "global_k_raw_patchvec": global_k_raw.float(),
            "global_v_raw_patchvec": global_v_raw.float() if global_v_raw is not None else None,
            "global_q_raw_patchvec_layers": global_q_raw_layers.permute(0, 2, 1, 3, 4, 5).contiguous().float(),
            "global_k_raw_patchvec_layers": global_k_raw_layers.permute(0, 2, 1, 3, 4, 5).contiguous().float(),
            "global_v_raw_patchvec_layers": global_v_layers_out,
            "dyn4d_global_layer_ids": torch.tensor(
                available_layers,
                device=global_q_raw_layers.device,
                dtype=torch.long,
            ),
        }

    @staticmethod
    def _new_hmc_trace(hmc_control: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not hmc_control:
            return None
        return {
            "identity_hooks": bool(hmc_control.get("identity_hooks", False)),
            "collect_trace": bool(hmc_control.get("collect_trace", True)),
            "frame_attention": [],
            "swa_read": [],
            "ttt_apply": [],
            "chunk_attention": [],
        }

    @staticmethod
    def _hmc_hook_requested(hmc_control: Optional[Dict[str, Any]], key: str) -> bool:
        if not hmc_control:
            return False
        return bool(
            hmc_control.get("identity_hooks", False)
            or hmc_control.get("collect_trace", False)
            or hmc_control.get(key, False)
        )

    @staticmethod
    def _append_hmc_trace(trace: Optional[Dict[str, Any]], key: str, record: Dict[str, Any]) -> None:
        if trace is None:
            return
        if key not in trace:
            trace[key] = []
        trace[key].append(record)

    @staticmethod
    def _hmc_read_layer_enabled(
        hmc_control: Optional[Dict[str, Any]],
        *,
        layer: int,
        total_layers: int,
    ) -> bool:
        if not hmc_control or hmc_control.get("identity_hooks", False):
            return False
        mode = str(hmc_control.get("read_layer_mode", "all"))
        single_layer = int(hmc_control.get("read_single_layer", -1))
        return Pi3._hmc_layer_mode_enabled(mode, single_layer, layer=layer, total_layers=total_layers)

    @staticmethod
    def _hmc_layer_mode_enabled(
        mode: str,
        single_layer: int,
        *,
        layer: int,
        total_layers: int,
    ) -> bool:
        if mode == "all":
            return True
        if mode == "single":
            return int(layer) == int(single_layer)
        if mode == "early_quarter":
            return int(layer) < max(1, int(total_layers) // 4)
        if mode == "early_half":
            return int(layer) < max(1, int(total_layers) // 2)
        span = max(1, int(total_layers) // 3)
        if mode == "early":
            return int(layer) < span
        if mode == "late":
            return int(layer) >= int(total_layers) - span
        if mode == "middle":
            return span <= int(layer) < int(total_layers) - span
        return True

    @staticmethod
    def _hmc_context_source_skip_layer_enabled(
        hmc_control: Optional[Dict[str, Any]],
        *,
        layer: int,
        total_layers: int,
    ) -> bool:
        if not hmc_control or hmc_control.get("identity_hooks", False):
            return False
        if not hmc_control.get("enable_context_source_skip", False):
            return False
        mode = str(hmc_control.get("context_source_skip_layer_mode", "early"))
        single_layer = int(hmc_control.get("context_source_skip_single_layer", -1))
        return Pi3._hmc_layer_mode_enabled(mode, single_layer, layer=layer, total_layers=total_layers)

    def _make_frame_attention_bias(
        self,
        hmc_control: Optional[Dict[str, Any]],
        *,
        batch_size: int,
        frame_num: int,
        tokens_per_frame: int,
        num_heads: int = 0,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        """Build a real frame-attention bias tensor when non-identity control is requested.

        Identity hooks return ``None`` so the exact native kernel path is kept,
        while the hook call itself is still recorded at the real model site.
        """
        if not hmc_control or hmc_control.get("identity_hooks", False):
            return None
        if not hmc_control.get("enable_frame_read_control", False):
            return None
        beta = float(hmc_control.get("beta_frame", 0.0))
        if beta == 0.0:
            return None
        D_tok = hmc_control.get("D_tok")
        if D_tok is None:
            return None
        D = D_tok.to(device=device, dtype=torch.float32).reshape(batch_size, frame_num, tokens_per_frame)
        P_ref = hmc_control.get("P_ref")
        if P_ref is not None:
            ref = P_ref.to(device=device, dtype=torch.float32).reshape(batch_size, frame_num, tokens_per_frame)
            D = D * (1.0 - ref.clamp(0.0, 1.0))
        D = D.reshape(batch_size * frame_num, tokens_per_frame)
        Dq = D[:, :, None]
        Dk = D[:, None, :]

        def _scope_bias(pair_bias: torch.Tensor) -> torch.Tensor:
            query_region = str(hmc_control.get("frame_bias_query_region", "all")).strip().lower()
            if query_region not in {"all", "head", "mid_tail", "tail"}:
                query_region = "all"
            if query_region != "all":
                ov = max(int(hmc_control.get("read_overlap_frames", 0) or 0), 0)
                if ov <= 0:
                    ov = min(3, int(frame_num))
                ov = min(int(ov), int(frame_num))
                frame_ids = torch.arange(int(frame_num), device=device).reshape(1, int(frame_num), 1)
                if query_region == "head":
                    query_mask = frame_ids < ov
                elif query_region == "tail":
                    query_mask = frame_ids >= max(int(frame_num) - ov, 0)
                else:
                    query_mask = frame_ids >= ov
                query_mask = query_mask.expand(batch_size, int(frame_num), int(tokens_per_frame))
                query_mask = query_mask.reshape(batch_size * int(frame_num), int(tokens_per_frame))
                pair_bias = pair_bias * query_mask[:, :, None].to(dtype=pair_bias.dtype)

            raw_heads = str(hmc_control.get("frame_bias_head_indices", "") or "").strip()
            if raw_heads and int(num_heads) > 0:
                indices: list[int] = []
                for part in raw_heads.split(","):
                    part = part.strip()
                    if not part:
                        continue
                    try:
                        idx = int(part)
                    except ValueError:
                        continue
                    if 0 <= idx < int(num_heads):
                        indices.append(idx)
                if indices:
                    head_mask = torch.zeros((int(num_heads),), device=device, dtype=pair_bias.dtype)
                    head_mask[torch.tensor(sorted(set(indices)), device=device, dtype=torch.long)] = 1.0
                    return pair_bias.unsqueeze(1) * head_mask.reshape(1, int(num_heads), 1, 1)
            return pair_bias.unsqueeze(1)

        mode = str(hmc_control.get("frame_bias_mode", "pair"))
        if mode == "key":
            keep = (1.0 - Dk).expand(-1, D.shape[1], -1)
        elif mode == "query":
            # A uniform per-query attention-logit shift cancels under softmax;
            # query weakening is handled as an output gate below instead.
            return None
        elif mode in {"qk_pair_stable_harm", "read_qk_pair_bias", "qk_pair_key_stability"}:
            q_risk = Dq.clamp(0.0, 1.0)
            K_stable = hmc_control.get("K_stable_tok") if mode == "qk_pair_key_stability" else None
            if K_stable is not None:
                K = K_stable.to(device=device, dtype=torch.float32).reshape(batch_size, frame_num, tokens_per_frame)
                if P_ref is not None:
                    K = K * (1.0 - ref.clamp(0.0, 1.0))
                k_stable = K.reshape(batch_size * frame_num, tokens_per_frame).clamp(0.0, 1.0)[:, None, :]
                k_harm = (1.0 - k_stable).clamp(0.0, 1.0)
            else:
                k_harm = Dk.clamp(0.0, 1.0)
                k_stable = (1.0 - k_harm).clamp(0.0, 1.0)
            pair_score = q_risk * (k_stable - k_harm)
            return _scope_bias((beta * pair_score).to(dtype=dtype))
        elif mode in {"qk_pair_random_same_mass", "qk_pair_key_stability_random_same_mass"}:
            q_risk = Dq.clamp(0.0, 1.0)
            K_stable = hmc_control.get("K_stable_tok") if mode == "qk_pair_key_stability_random_same_mass" else None
            if K_stable is not None:
                K = K_stable.to(device=device, dtype=torch.float32).reshape(batch_size, frame_num, tokens_per_frame)
                if P_ref is not None:
                    K = K * (1.0 - ref.clamp(0.0, 1.0))
                k_stable = K.reshape(batch_size * frame_num, tokens_per_frame).clamp(0.0, 1.0)[:, None, :]
                k_harm = (1.0 - k_stable).clamp(0.0, 1.0)
            else:
                k_harm = Dk.clamp(0.0, 1.0)
                k_stable = (1.0 - k_harm).clamp(0.0, 1.0)
            pair_score = q_risk * (k_stable - k_harm)
            flat = pair_score.reshape(pair_score.shape[0], -1)
            shuffled = torch.empty_like(flat)
            base_seed = int(hmc_control.get("read_pair_random_seed", 1729))
            for row_idx in range(flat.shape[0]):
                gen = torch.Generator(device=flat.device)
                gen.manual_seed(base_seed + int(row_idx) + 1009 * int(flat.shape[1]))
                perm = torch.randperm(flat.shape[1], device=flat.device, generator=gen)
                shuffled[row_idx] = flat[row_idx, perm]
            return _scope_bias((beta * shuffled.reshape_as(pair_score)).to(dtype=dtype))
        else:
            keep = 1.0 - (1.0 - Dq) * Dk
        keep = keep.clamp_min(1e-4)
        return _scope_bias((beta * torch.log(keep)).to(dtype=dtype))

    def _sample_frame_bias_attention_mass_stats(
        self,
        blk: nn.Module,
        hidden_for_block: torch.Tensor,
        pos_for_block: Optional[torch.Tensor],
        attn_mask: Optional[torch.Tensor],
        *,
        max_queries: int = 64,
    ) -> Dict[str, Any]:
        """Audit-only raw-QK attention mass before/after a dense frame bias.

        The returned values are diagnostic summaries only. They are computed
        from sampled query rows and never feed back into the model output.
        """
        if attn_mask is None or not torch.is_tensor(attn_mask) or not torch.is_floating_point(attn_mask):
            return {}
        try:
            B, N, C = hidden_for_block.shape
            attn = getattr(blk, "attn", None)
            if attn is None or not hasattr(attn, "qkv"):
                return {"frame_bias_attention_mass_error": "missing_attn_qkv"}
            with torch.no_grad():
                qkv = attn.qkv(hidden_for_block).reshape(
                    B,
                    N,
                    3,
                    int(attn.num_heads),
                    C // int(attn.num_heads),
                ).transpose(1, 3)
                q, k, _v = [qkv[:, :, idx] for idx in range(3)]
                q = attn.q_norm(q)
                k = attn.k_norm(k).to(q.dtype)
                if getattr(attn, "rope", None) is not None:
                    q = attn.rope(q, pos_for_block)
                    k = attn.rope(k, pos_for_block)
                q = q.float()
                k = k.float()
                bias = attn_mask.detach().to(device=q.device, dtype=torch.float32)
                if bias.ndim != 4 or int(bias.shape[0]) != int(B) or int(bias.shape[-2]) != int(N) or int(bias.shape[-1]) != int(N):
                    return {
                        "frame_bias_attention_mass_error": (
                            f"bias_shape_mismatch bias={tuple(bias.shape)} hidden={(int(B), int(N), int(C))}"
                        )
                    }
                if int(bias.shape[1]) not in {1, int(q.shape[1])}:
                    return {
                        "frame_bias_attention_mass_error": (
                            f"bias_head_mismatch bias={tuple(bias.shape)} heads={int(q.shape[1])}"
                        )
                    }
                max_q = max(1, int(max_queries))
                if int(N) > max_q:
                    q_idx = torch.linspace(0, int(N) - 1, steps=max_q, device=q.device).round().long().unique()
                else:
                    q_idx = torch.arange(int(N), device=q.device)
                if int(q_idx.numel()) <= 0:
                    return {"frame_bias_attention_mass_error": "empty_query_sample"}
                qb = q[:, :, q_idx, :]
                head_dim = max(1, int(q.shape[-1]))
                scores = torch.matmul(qb, k.transpose(-2, -1)) * (float(head_dim) ** -0.5)
                bias_sample = bias[:, :, q_idx, :]
                if int(bias_sample.shape[1]) == 1 and int(q.shape[1]) != 1:
                    bias_sample = bias_sample.expand(-1, int(q.shape[1]), -1, -1)
                attn_before = torch.softmax(scores, dim=-1)
                attn_after = torch.softmax(scores + bias_sample, dim=-1)
                pos_mask = bias_sample > 1e-7
                neg_mask = bias_sample < -1e-7

                def _masked_mean_mass(attn_tensor: torch.Tensor, mask: torch.Tensor) -> Optional[float]:
                    if not bool(mask.any().item()):
                        return None
                    mass = (attn_tensor * mask.to(dtype=attn_tensor.dtype)).sum(dim=-1)
                    valid = mask.any(dim=-1)
                    if not bool(valid.any().item()):
                        return None
                    return float(mass[valid].mean().item())

                pos_before = _masked_mean_mass(attn_before, pos_mask)
                pos_after = _masked_mean_mass(attn_after, pos_mask)
                neg_before = _masked_mean_mass(attn_before, neg_mask)
                neg_after = _masked_mean_mass(attn_after, neg_mask)
                out: Dict[str, Any] = {
                    "attention_mass_metric": "frame_bias_pair_sign_sampled_qk",
                    "attention_mass_query_sample_tokens_mean": float(q_idx.numel()),
                    "frame_bias_attention_mass_sampled": True,
                    "frame_bias_attention_mass_query_count": int(q_idx.numel()),
                    "frame_bias_positive_pair_count_mean": float(pos_mask.sum(dim=-1).float().mean().item()),
                    "frame_bias_negative_pair_count_mean": float(neg_mask.sum(dim=-1).float().mean().item()),
                    "frame_bias_positive_pair_fraction": float(pos_mask.float().mean().item()),
                    "frame_bias_negative_pair_fraction": float(neg_mask.float().mean().item()),
                }
                if pos_before is not None and pos_after is not None:
                    out.update({
                        "frame_bias_positive_pair_mass_before": pos_before,
                        "frame_bias_positive_pair_mass_after": pos_after,
                        "frame_bias_positive_pair_mass_lift": pos_after - pos_before,
                        "attention_mass_retained_before": pos_before,
                        "attention_mass_retained_after": pos_after,
                    })
                if neg_before is not None and neg_after is not None:
                    out.update({
                        "frame_bias_negative_pair_mass_before": neg_before,
                        "frame_bias_negative_pair_mass_after": neg_after,
                        "frame_bias_negative_pair_mass_lift": neg_after - neg_before,
                        "attention_mass_removed_before": neg_before,
                        "attention_mass_removed_after": neg_after,
                        "attention_mass_actual_after": neg_after,
                    })
                if bool(pos_mask.any().item()):
                    out["frame_bias_positive_bias_mean"] = float(bias_sample[pos_mask].mean().item())
                if bool(neg_mask.any().item()):
                    out["frame_bias_negative_bias_mean"] = float(bias_sample[neg_mask].mean().item())
                return out
        except Exception as exc:  # pragma: no cover - diagnostic path must not break inference.
            return {"frame_bias_attention_mass_error": f"{type(exc).__name__}: {exc}"}

    def _make_frame_attention_query_gate(
        self,
        hmc_control: Optional[Dict[str, Any]],
        *,
        batch_size: int,
        frame_num: int,
        tokens_per_frame: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if not hmc_control or hmc_control.get("identity_hooks", False):
            return None
        if str(hmc_control.get("frame_bias_mode", "pair")) != "query":
            return None
        if not hmc_control.get("enable_frame_read_control", False):
            return None
        beta = float(hmc_control.get("beta_frame", 0.0))
        if beta == 0.0:
            return None
        D_tok = hmc_control.get("D_tok")
        if D_tok is None:
            return None
        D = D_tok.to(device=device, dtype=torch.float32).reshape(batch_size, frame_num, tokens_per_frame)
        P_ref = hmc_control.get("P_ref")
        if P_ref is not None:
            ref = P_ref.to(device=device, dtype=torch.float32).reshape(batch_size, frame_num, tokens_per_frame)
            D = D * (1.0 - ref.clamp(0.0, 1.0))
        gate = (1.0 - beta * D).clamp(0.0, 1.0)
        return gate.reshape(batch_size * frame_num, tokens_per_frame, 1).to(dtype=dtype)

    def _make_chunk_attention_source_bias(
        self,
        hmc_control: Optional[Dict[str, Any]],
        *,
        batch_size: int,
        frame_num: int,
        tokens_per_frame: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[Dict[str, Any]]:
        """Build a compact global/chunk attention source bias.

        Global decoder layers see all frames as one long token sequence. A
        dense pairwise bias would be prohibitively large for KITTI chunks, so
        this uses the attention layer's source_soft descriptor to apply a
        source-column logit bias without materializing a [Q,K] matrix.
        """
        if not hmc_control or hmc_control.get("identity_hooks", False):
            return None
        if not hmc_control.get("enable_chunk_read_control", False):
            return None
        beta = float(hmc_control.get("beta_chunk", hmc_control.get("beta_frame", 0.0)))
        if beta == 0.0:
            return None
        D_tok = hmc_control.get("D_tok")
        if D_tok is None:
            return None
        D = D_tok.to(device=device, dtype=torch.float32).reshape(batch_size, frame_num, tokens_per_frame)
        P_ref = hmc_control.get("P_ref")
        if P_ref is not None:
            ref = P_ref.to(device=device, dtype=torch.float32).reshape(batch_size, frame_num, tokens_per_frame)
            D = D * (1.0 - ref.clamp(0.0, 1.0))
        D = D.reshape(batch_size, frame_num * tokens_per_frame)
        mode = str(hmc_control.get("chunk_bias_mode", "key")).strip().lower()
        if mode in {"query", "none", "off"}:
            return None
        min_keep = min(max(float(hmc_control.get("chunk_bias_min_keep", 1e-4)), 1e-6), 1.0)
        if mode in {"inverse_key", "source_inverse"}:
            keep = D.clamp_min(min_keep)
        else:
            keep = (1.0 - D).clamp_min(min_keep)
        source_bias = beta * torch.log(keep)
        affected = D > 0.0
        return {
            "type": "source_soft",
            "affected_mask": affected,
            "source_bias_values": source_bias.to(dtype=dtype),
            "attention_mass_stats": [],
            "attention_mass_max_queries": int(hmc_control.get("attention_mass_max_queries", 512) or 512),
            "attention_mass_metric": "chunk_source_soft",
        }

    def _make_context_source_skip_bias(
        self,
        hmc_control: Optional[Dict[str, Any]],
        *,
        path: str,
        batch_size: int,
        frame_num: int,
        tokens_per_frame: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
        """Build a source-column mask for VGGT4D-style context-source skip.

        Query rows are preserved.  Selected patch tokens are prevented from
        acting as Key/Value sources by adding a large negative logit bias to
        their source columns.  This keeps tensor shapes unchanged while matching
        source-removal attention semantics.
        """

        empty_stats: Dict[str, Any] = {
            "context_source_skip_applied": False,
            "context_source_skip_impl": str(hmc_control.get("context_source_skip_impl", "bias")) if hmc_control else "bias",
            "source_skip_tokens": 0,
            "source_tokens_before": 0,
            "source_tokens_after": 0,
            "source_keep_ratio": 1.0,
            "special_token_keep_ratio": 1.0,
            "empty_source_events": 0,
        }
        if not hmc_control or hmc_control.get("identity_hooks", False):
            return None, empty_stats
        if not hmc_control.get("enable_context_source_skip", False):
            return None, empty_stats
        scope = str(hmc_control.get("context_source_skip_scope", "frame"))
        if path == "frame_attention" and scope not in {"frame", "both"}:
            return None, empty_stats
        if path == "chunk_attention" and scope not in {"chunk", "both"}:
            return None, empty_stats
        D_tok = hmc_control.get("D_tok")
        if D_tok is None:
            return None, empty_stats

        D = D_tok.to(device=device, dtype=torch.float32).reshape(batch_size, frame_num, tokens_per_frame)
        P_ref = hmc_control.get("P_ref")
        if P_ref is None:
            protected = torch.zeros_like(D, dtype=torch.bool)
        else:
            protected = (
                P_ref.to(device=device, dtype=torch.float32)
                .reshape(batch_size, frame_num, tokens_per_frame)
                .clamp(0.0, 1.0)
                > 0.5
            )
        eligible = ~protected
        frame_region = str(hmc_control.get("context_source_skip_frame_region", "all")).strip().lower()
        if frame_region not in {"all", "head", "mid_tail", "tail"}:
            frame_region = "all"
        frame_region_debug: Dict[str, Any] = {
            "context_source_skip_frame_region": frame_region,
            "context_source_skip_frame_region_overlap_frames": 0,
            "context_source_skip_frame_region_eligible_before": int(eligible.sum().item()),
            "context_source_skip_frame_region_eligible_after": int(eligible.sum().item()),
        }
        if frame_region != "all":
            ov = max(int(hmc_control.get("read_overlap_frames", 0) or 0), 0)
            if ov <= 0:
                ov = min(3, int(frame_num))
            ov = min(int(ov), int(frame_num))
            frame_ids = torch.arange(int(frame_num), device=device).reshape(1, int(frame_num), 1)
            if frame_region == "head":
                region_mask = frame_ids < ov
            elif frame_region == "tail":
                region_mask = frame_ids >= max(int(frame_num) - ov, 0)
            else:
                region_mask = frame_ids >= ov
            region_mask = region_mask.expand_as(eligible)
            before_region = int(eligible.sum().item())
            eligible = eligible & region_mask
            frame_region_debug.update({
                "context_source_skip_frame_region_overlap_frames": int(ov),
                "context_source_skip_frame_region_eligible_before": before_region,
                "context_source_skip_frame_region_eligible_after": int(eligible.sum().item()),
            })
        query_region = str(hmc_control.get("context_source_skip_query_region", "all")).strip().lower()
        if query_region not in {"all", "head", "mid_tail", "tail"}:
            query_region = "all"
        query_mask_out = None
        query_region_debug: Dict[str, Any] = {
            "context_source_skip_query_region": query_region,
            "context_source_skip_query_region_overlap_frames": 0,
            "context_source_skip_query_region_tokens": None,
        }
        if query_region != "all":
            ov = max(int(hmc_control.get("read_overlap_frames", 0) or 0), 0)
            if ov <= 0:
                ov = min(3, int(frame_num))
            ov = min(int(ov), int(frame_num))
            frame_ids = torch.arange(int(frame_num), device=device).reshape(1, int(frame_num), 1)
            if query_region == "head":
                query_region_mask = frame_ids < ov
            elif query_region == "tail":
                query_region_mask = frame_ids >= max(int(frame_num) - ov, 0)
            else:
                query_region_mask = frame_ids >= ov
            query_region_mask = query_region_mask.expand(batch_size, int(frame_num), int(tokens_per_frame))
            if path == "frame_attention":
                query_mask_out = query_region_mask.reshape(batch_size * int(frame_num), int(tokens_per_frame)).detach()
            else:
                query_mask_out = query_region_mask.reshape(batch_size, int(frame_num) * int(tokens_per_frame)).detach()
            query_region_debug.update({
                "context_source_skip_query_region_overlap_frames": int(ov),
                "context_source_skip_query_region_tokens": int(query_region_mask.sum().item()),
            })
        source_eligible_base = eligible.clone()
        mask_name = str(hmc_control.get("context_source_skip_mask", "dg_q90")).lower()
        anchor_boost_mask = mask_name in {
            "semantic_anchor",
            "semantic_anchor_boost",
            "semantic_geometry_anchor",
            "semantic_anchor_bank",
            "anchor_bank",
        }
        semantic_role_random_base_map = {
            "random_same_mass_semantic_role_negative": "semantic_role_negative",
            "random_same_mass_semrole_negative": "semantic_role_negative",
            "random_same_mass": "semantic_role_negative",
            "random_same_mass_semantic_role_positive": "semantic_role_positive",
            "random_same_mass_semrole_positive": "semantic_role_positive",
            "random_same_mass_semantic_role_protect": "semantic_role_protect",
            "random_same_mass_semantic_role_protected": "semantic_role_protect",
            "random_same_mass_semantic_role_stable": "semantic_role_stable",
            "random_same_mass_semantic_role_anchor": "semantic_role_stable",
        }
        random_same_mass_semantic_role = mask_name in semantic_role_random_base_map
        random_same_mass_high_influence = mask_name in {
            "random_same_mass_high_influence",
            "random_high_influence_same_mass",
            "v40_read_a2_high_influence_random_same_mass",
        }
        phase2_random_base_map = {
            "dg_q90_random_same_mass": "dg_q90",
            "dg_q90_anchor_rescue_random_same_mass": "dg_q90_anchor_rescue",
            "dg_high_strict_random_same_mass": "dg_high_strict",
            "highd_q90_random_same_mass": "highd_q90",
            "v67_carrier_highd_q80_random_same_mass": "v67_carrier_highd_q80",
            "v67_carrier_highd_q90_random_same_mass": "v67_carrier_highd_q90",
            "v67_source_attention_q90_random_same_mass": "v67_source_attention_q90",
            "v67_source_attention_q95_random_same_mass": "v67_source_attention_q95",
            "v67_carrier_dynamic_highd_random_same_mass": "v67_carrier_dynamic_highd",
            "v67_carrier_sky_highd_random_same_mass": "v67_carrier_sky_highd",
            "v67_carrier_vegetation_highd_random_same_mass": "v67_carrier_vegetation_highd",
            "v67_carrier_ground_highd_random_same_mass": "v67_carrier_ground_highd",
            "v67_carrier_vertical_static_highd_random_same_mass": "v67_carrier_vertical_static_highd",
            "v67_carrier_overlap_highd_random_same_mass": "v67_carrier_overlap_highd",
            "v67_carrier_tail_highd_random_same_mass": "v67_carrier_tail_highd",
        }
        phase2_source_attention_group_specs = {
            "v67_source_attention_sky_q90": ("sky", 0.90),
            "v67_source_attention_sky_q95": ("sky", 0.95),
            "v67_source_attention_lowstuff_q90": ("lowstuff", 0.90),
            "v67_source_attention_lowstuff_q95": ("lowstuff", 0.95),
            "v67_source_attention_structure_q90": ("structure", 0.90),
            "v67_source_attention_structure_q95": ("structure", 0.95),
            "v67_source_attention_movable_q90": ("movable", 0.90),
            "v67_source_attention_movable_q95": ("movable", 0.95),
            "v96_source_attention_lowstuff_q90_anchor_rescue": ("lowstuff", 0.90),
        }
        anchor_rescue_source_attention_masks = {
            "dg_q90_anchor_rescue",
            "v96_source_attention_lowstuff_q90_anchor_rescue",
        }
        for _source_attention_group_base in tuple(phase2_source_attention_group_specs.keys()):
            phase2_random_base_map[f"{_source_attention_group_base}_random_same_mass"] = _source_attention_group_base
        phase2_shuffled_base_map = {
            "v67_carrier_sky_highd_shuffled": "v67_carrier_sky_highd",
            "v67_carrier_vegetation_highd_shuffled": "v67_carrier_vegetation_highd",
            "v67_carrier_ground_highd_shuffled": "v67_carrier_ground_highd",
            "v67_carrier_vertical_static_highd_shuffled": "v67_carrier_vertical_static_highd",
        }
        for _source_attention_group_base in tuple(phase2_source_attention_group_specs.keys()):
            phase2_shuffled_base_map[f"{_source_attention_group_base}_shuffled"] = _source_attention_group_base
        random_same_mass_phase2 = mask_name in phase2_random_base_map
        random_same_mass = random_same_mass_semantic_role or random_same_mass_high_influence or random_same_mass_phase2
        sem_z_dg_soft = mask_name in {
            "sem_z_dg_soft_resid",
            "semantic_z_dg_soft_resid",
            "semantic_conditioned_dg_soft_resid",
        }
        explicit_source_attention_top_quantile = hmc_control.get("context_source_skip_source_attention_top_quantile", None)
        swa_redirection_source_masks = {
            "swa_redirection_source_positive",
            "semantic_swa_redirection_source_positive",
        }
        if random_same_mass_semantic_role:
            base_mask_name = semantic_role_random_base_map[mask_name]
        elif random_same_mass_high_influence:
            base_mask_name = "v40_read_a2_high_influence"
        elif random_same_mass_phase2:
            base_mask_name = phase2_random_base_map[mask_name]
        elif mask_name in phase2_shuffled_base_map:
            base_mask_name = phase2_shuffled_base_map[mask_name]
        else:
            base_mask_name = mask_name
        r2_sky_masks = {
            "v40_read_a4_sky_appanom",
            "v40_read_a4_sky_appanom_rho",
            "sky_appanom",
            "sky_highd_source_mass",
        }
        r2_sky_no_source_gate_masks = {
            "v40_read_a4_sky_appanom_no_source_mass_control",
            "sky_appanom_no_source_mass_control",
        }
        r2_sky_shuffled_masks = {
            "v40_read_a4_sky_appanom_shuffled",
            "sky_appanom_shuffled",
            "shuffled_sky_appanom",
        }
        r3_high_influence_masks = {
            "v40_read_a2_high_influence",
            "high_influence_anomaly",
            "source_influence_dg_q80",
        }
        phase2_carrier_masks = {
            "v67_carrier_highd_q80",
            "v67_carrier_highd_q90",
            "v67_carrier_dynamic_highd",
            "v67_carrier_sky_highd",
            "v67_carrier_vegetation_highd",
            "v67_carrier_ground_highd",
            "v67_carrier_vertical_static_highd",
            "v67_carrier_overlap_highd",
            "v67_carrier_tail_highd",
        }
        phase2_source_attention_masks = {
            "v67_source_attention_q90",
            "v67_source_attention_q95",
        }
        phase2_source_attention_masks.update(phase2_source_attention_group_specs.keys())
        semantic_extra_stats: Dict[str, Any] = {}
        source_attention_top_quantile = None
        source_attention_top_random_same_mass = False
        source_attention_anchor_boost_mask = None
        source_attention_anchor_boost_score = None
        if anchor_boost_mask:
            quantile = 0.0
        elif base_mask_name in {"dg_q80", "dg_high", "highd_q80"}:
            quantile = 0.80
        elif base_mask_name in {"dg_q85", "dg_high_q85", "highd_q85"}:
            quantile = 0.85
        elif base_mask_name in {"dg_q90", "dg_q90_anchor_rescue", "dg_high_strict", "highd_q90"}:
            quantile = 0.90
        elif base_mask_name in {"lowstuff_highd", "semantic_lowstuff_highd", "sem_lowstuff_highd"}:
            quantile = 0.90
        elif base_mask_name in {"sem_structure_rescue_dg_q80", "structure_rescue_dg_q80"}:
            quantile = 0.80
        elif base_mask_name in {
            "semantic_role_negative",
            "semantic_role_source_skip",
            "semrole_negative",
            "semantic_role_positive",
            "semantic_role_stable",
            "semantic_role_anchor",
            "semantic_role_protect",
            "semantic_role_protected",
            "swa_redirection_source_positive",
            "semantic_swa_redirection_source_positive",
            "v36_synthetic_role_negative",
            "sem_z_dg_soft_resid",
            "semantic_z_dg_soft_resid",
            "semantic_conditioned_dg_soft_resid",
        }:
            quantile = 0.80
        elif base_mask_name in r2_sky_masks or base_mask_name in r2_sky_no_source_gate_masks or base_mask_name in r2_sky_shuffled_masks:
            quantile = 0.80
        elif base_mask_name in r3_high_influence_masks:
            quantile = 0.80
        elif base_mask_name in phase2_carrier_masks:
            quantile = 0.90 if base_mask_name == "v67_carrier_highd_q90" else 0.80
        elif base_mask_name in phase2_source_attention_masks:
            if base_mask_name in phase2_source_attention_group_specs:
                quantile = float(phase2_source_attention_group_specs[base_mask_name][1])
            else:
                quantile = 0.95 if base_mask_name == "v67_source_attention_q95" else 0.90
        else:
            quantile = 0.90

        source_attention_group_name = None
        if base_mask_name in phase2_source_attention_group_specs:
            source_attention_group_name = phase2_source_attention_group_specs[base_mask_name][0]
            source_attention_group_shuffled = (
                mask_name in phase2_shuffled_base_map
                and phase2_shuffled_base_map.get(mask_name) == base_mask_name
            )

            def _shuffle_semantic_tensor(values: torch.Tensor) -> torch.Tensor:
                flat = values.reshape(-1)
                token_idx = torch.arange(int(flat.numel()), device=device, dtype=torch.float32)
                chunk = float(hmc_control.get("semantic_action_chunk_idx", -1) if hmc_control else -1)
                scores = torch.frac(torch.sin((token_idx + 1.0 + chunk * 131.0) * 12.9898) * 43758.5453)
                order = torch.argsort(scores)
                return flat[order].reshape_as(values)

            group_mask = torch.zeros_like(eligible, dtype=torch.bool)
            if source_attention_group_name == "sky":
                L_sem_tok = hmc_control.get("L_sem_tok")
                if L_sem_tok is not None:
                    L = L_sem_tok.to(device=device, dtype=torch.long).reshape(batch_size, frame_num, tokens_per_frame)
                    if source_attention_group_shuffled:
                        L = _shuffle_semantic_tensor(L)
                    for label_id in _CONTEXT_SKY_FINE_LABEL_IDS:
                        group_mask |= L == int(label_id)
            else:
                G_sem_tok = hmc_control.get("G_sem_tok")
                group_id_map = {
                    "structure": _CONTEXT_SEM_GROUP_STRUCTURE,
                    "movable": _CONTEXT_SEM_GROUP_MOVABLE,
                    "lowstuff": _CONTEXT_SEM_GROUP_LOWSTUFF,
                }
                if G_sem_tok is not None and source_attention_group_name in group_id_map:
                    G = G_sem_tok.to(device=device, dtype=torch.long).reshape(batch_size, frame_num, tokens_per_frame)
                    if source_attention_group_shuffled:
                        G = _shuffle_semantic_tensor(G)
                    group_mask = G == int(group_id_map[source_attention_group_name])
            eligible = eligible & group_mask
            semantic_extra_stats.update({
                "v67_source_attention_semantic_group_id": float({
                    "sky": 10,
                    "structure": _CONTEXT_SEM_GROUP_STRUCTURE,
                    "movable": _CONTEXT_SEM_GROUP_MOVABLE,
                    "lowstuff": _CONTEXT_SEM_GROUP_LOWSTUFF,
                }.get(source_attention_group_name, -1)),
                "v67_source_attention_group_eligible_tokens": float(eligible.sum().item()),
                "v67_source_attention_group_missing_semantic": bool(not group_mask.any().item()),
                "v67_source_attention_label_control": "shuffled" if source_attention_group_shuffled else "semantic",
            })

        eligible_scores = D[eligible]
        if eligible_scores.numel() == 0:
            stats = dict(empty_stats)
            stats.update({
                "context_source_skip_mask": mask_name,
                "context_source_skip_reason": "no_eligible_patch_tokens",
            })
            stats.update(frame_region_debug)
            stats.update(query_region_debug)
            return None, stats
        thr = torch.quantile(eligible_scores.float(), float(quantile))
        skip = (D > thr) & eligible

        source_control_score = None
        if anchor_boost_mask:
            A_anchor_tok = hmc_control.get("A_anchor_tok")
            M_anchor_tok = hmc_control.get("A_anchor_mask_tok")
            if A_anchor_tok is None:
                skip = torch.zeros_like(skip, dtype=torch.bool)
                source_control_score = torch.zeros_like(D, dtype=torch.float32)
                semantic_reason = "semantic_anchor_missing"
            else:
                A = A_anchor_tok.to(device=device, dtype=torch.float32).reshape(batch_size, frame_num, tokens_per_frame).clamp(0.0, 1.0)
                if M_anchor_tok is None:
                    M = A > 0.0
                else:
                    M = (
                        M_anchor_tok.to(device=device, dtype=torch.float32)
                        .reshape(batch_size, frame_num, tokens_per_frame)
                        .clamp(0.0, 1.0)
                        > 0.0
                    )
                skip = M & eligible
                source_control_score = torch.where(skip, A, torch.zeros_like(A))
                semantic_reason = f"semantic_anchor_bank:{hmc_control.get('semantic_anchor_mode', 'semantic')}"
        elif base_mask_name in {"lowstuff_highd", "semantic_lowstuff_highd"}:
            S_tok = hmc_control.get("S_tok")
            if S_tok is None:
                skip = torch.zeros_like(skip, dtype=torch.bool)
                semantic_reason = "semantic_value_missing"
            else:
                S = S_tok.to(device=device, dtype=torch.float32).reshape(batch_size, frame_num, tokens_per_frame)
                skip = skip & (S <= 0.45)
                semantic_reason = "semantic_value_le_045_and_highD"
        elif base_mask_name in {"sem_lowstuff_highd", "sem_structure_rescue_dg_q80", "structure_rescue_dg_q80"}:
            G_sem_tok = hmc_control.get("G_sem_tok")
            if G_sem_tok is None:
                skip = torch.zeros_like(skip, dtype=torch.bool)
                semantic_reason = "semantic_group_missing"
            else:
                G = G_sem_tok.to(device=device, dtype=torch.long).reshape(batch_size, frame_num, tokens_per_frame)
                lowstuff = G == 3
                structure = G == 0
                if base_mask_name in {"sem_lowstuff_highd"}:
                    skip = skip & lowstuff
                    semantic_reason = "exact_semantic_lowstuff_and_highD"
                else:
                    skip = skip & (~structure)
                    protected = protected | structure
                    semantic_reason = "exact_semantic_structure_rescue_and_highD"
        elif base_mask_name in r2_sky_masks or base_mask_name in r2_sky_no_source_gate_masks or base_mask_name in r2_sky_shuffled_masks:
            L_sem_tok = hmc_control.get("L_sem_tok")
            no_source_gate = base_mask_name in r2_sky_no_source_gate_masks
            shuffled_label_control = base_mask_name in r2_sky_shuffled_masks
            if L_sem_tok is None:
                skip = torch.zeros_like(skip, dtype=torch.bool)
                semantic_reason = "v40_read_a4_sky_appanom_label_missing"
            else:
                L = L_sem_tok.to(device=device, dtype=torch.long).reshape(batch_size, frame_num, tokens_per_frame)
                if shuffled_label_control:
                    flat_l = L.reshape(-1)
                    token_idx = torch.arange(int(flat_l.numel()), device=device, dtype=torch.float32)
                    chunk = float(hmc_control.get("semantic_action_chunk_idx", -1) if hmc_control else -1)
                    scores = torch.frac(torch.sin((token_idx + 1.0 + chunk * 131.0) * 12.9898) * 43758.5453)
                    order = torch.argsort(scores)
                    L = flat_l[order].reshape_as(L)
                sky = torch.zeros_like(eligible, dtype=torch.bool)
                for label_id in _CONTEXT_SKY_FINE_LABEL_IDS:
                    sky |= L == int(label_id)
                sky = sky & eligible
                sky_count = int(sky.sum().item())
                eligible_count = max(int(eligible.sum().item()), 1)
                if sky_count <= 0:
                    skip = torch.zeros_like(skip, dtype=torch.bool)
                    sky_thr = torch.tensor(float("nan"), device=device)
                    source_mass_proxy_ratio = 0.0
                    sky_highd_proxy_ratio = 0.0
                    proxy_gate_pass = bool(no_source_gate)
                    semantic_reason = "v40_read_a4_sky_appanom_no_sky_tokens"
                else:
                    sky_vals = D[sky].float()
                    sky_thr = torch.quantile(sky_vals, 0.80)
                    sky_risk = sky & (D >= sky_thr)
                    source_mass_proxy_ratio = float(sky_count / eligible_count)
                    sky_highd_proxy_ratio = float(int(sky_risk.sum().item()) / eligible_count)
                    proxy_gate_pass = bool(
                        no_source_gate
                        or (
                            source_mass_proxy_ratio >= 0.05
                            and sky_highd_proxy_ratio >= 0.01
                        )
                    )
                    skip = sky_risk if proxy_gate_pass else torch.zeros_like(skip, dtype=torch.bool)
                    semantic_reason = (
                        "v40_read_a4_sky_appanom_"
                        + ("shuffled_label_" if shuffled_label_control else "")
                        + ("no_source_gate" if no_source_gate else "proxy_source_gate")
                    )
                rho = float(hmc_control.get("context_source_skip_soft_rho", 0.0) or 0.0)
                semantic_extra_stats.update({
                    "v40_r2_label_control": "shuffled" if shuffled_label_control else "semantic",
                    "v40_r2_no_source_mass_control": bool(no_source_gate),
                    "v40_r2_sky_token_count": int(sky_count),
                    "v40_r2_sky_token_ratio": float(sky_count / eligible_count),
                    "v40_r2_sky_highd_token_ratio": float(sky_highd_proxy_ratio),
                    "v40_r2_source_mass_proxy_ratio": float(source_mass_proxy_ratio),
                    "v40_r2_source_mass_proxy_gate_pass": bool(proxy_gate_pass),
                    "v40_r2_sky_highd_threshold": (
                        float(sky_thr.item()) if torch.isfinite(sky_thr).item() else None
                    ),
                    "v40_r2_context_floor_sky_min_keep_target": 0.30,
                    "v40_r2_context_floor_global_min_keep_target": 0.95,
                    "v40_r2_global_keep_proxy_after": float(1.0 - rho * max(sky_highd_proxy_ratio, 0.0)),
                })
        elif base_mask_name in r3_high_influence_masks:
            semantic_reason = "v40_read_a2_high_influence_source_proxy_D_q80"
            source_control_score = None
            semantic_extra_stats.update({
                "v40_r3_source_influence_proxy": "D_tok_q80",
                "v40_r3_residual_available": False,
                "v40_r3_source_attention_mass_available_before_action": False,
                "v40_r3_highd_quantile": float(quantile),
                "v40_r3_highd_threshold": float(thr.item()),
            })
        elif base_mask_name in phase2_carrier_masks:
            semantic_reason = f"v67_phase2_carrier:{base_mask_name}"
            shuffled_label_control = mask_name in phase2_shuffled_base_map
            source_control_score = None

            def _fine_mask(label_ids: set[int]) -> torch.Tensor:
                L_sem_tok = hmc_control.get("L_sem_tok")
                if L_sem_tok is None:
                    return torch.zeros_like(eligible, dtype=torch.bool)
                L = L_sem_tok.to(device=device, dtype=torch.long).reshape(batch_size, frame_num, tokens_per_frame)
                if shuffled_label_control:
                    flat_l = L.reshape(-1)
                    token_idx = torch.arange(int(flat_l.numel()), device=device, dtype=torch.float32)
                    chunk = float(hmc_control.get("semantic_action_chunk_idx", -1) if hmc_control else -1)
                    scores = torch.frac(torch.sin((token_idx + 1.0 + chunk * 131.0) * 12.9898) * 43758.5453)
                    order = torch.argsort(scores)
                    L = flat_l[order].reshape_as(L)
                out = torch.zeros_like(eligible, dtype=torch.bool)
                for label_id in sorted(int(x) for x in label_ids):
                    out |= L == int(label_id)
                return out & eligible

            if base_mask_name in {"v67_carrier_highd_q80", "v67_carrier_highd_q90"}:
                skip = skip & eligible
            elif base_mask_name == "v67_carrier_dynamic_highd":
                G_sem_tok = hmc_control.get("G_sem_tok")
                if G_sem_tok is None:
                    skip = torch.zeros_like(skip, dtype=torch.bool)
                    semantic_reason = "v67_phase2_carrier:dynamic_group_missing"
                else:
                    G = G_sem_tok.to(device=device, dtype=torch.long).reshape(batch_size, frame_num, tokens_per_frame)
                    skip = skip & (G == _CONTEXT_SEM_GROUP_MOVABLE)
            elif base_mask_name == "v67_carrier_sky_highd":
                sem = _fine_mask(_CONTEXT_SKY_FINE_LABEL_IDS)
                if int(sem.sum().item()) > 0:
                    sem_thr = torch.quantile(D[sem].float(), 0.80)
                    skip = sem & (D >= sem_thr)
                else:
                    skip = torch.zeros_like(skip, dtype=torch.bool)
            elif base_mask_name == "v67_carrier_vegetation_highd":
                sem = _fine_mask(_CONTEXT_VEGETATION_FINE_LABEL_IDS)
                if int(sem.sum().item()) > 0:
                    sem_thr = torch.quantile(D[sem].float(), 0.80)
                    skip = sem & (D >= sem_thr)
                else:
                    skip = torch.zeros_like(skip, dtype=torch.bool)
            elif base_mask_name == "v67_carrier_ground_highd":
                skip = _fine_mask(_CONTEXT_GROUND_FINE_LABEL_IDS) & (D >= thr)
            elif base_mask_name == "v67_carrier_vertical_static_highd":
                skip = _fine_mask(_CONTEXT_VERTICAL_STATIC_FINE_LABEL_IDS) & (D >= thr)
            elif base_mask_name == "v67_carrier_overlap_highd":
                ov = max(int(hmc_control.get("read_overlap_frames", 0) or 0), 0)
                if ov <= 0:
                    ov = min(3, int(frame_num))
                frame_ids = torch.arange(int(frame_num), device=device).reshape(1, frame_num, 1)
                overlap_region = (frame_ids < ov) | (frame_ids >= max(int(frame_num) - ov, 0))
                skip = skip & overlap_region
            elif base_mask_name == "v67_carrier_tail_highd":
                ov = max(int(hmc_control.get("read_overlap_frames", 0) or 0), 0)
                if ov <= 0:
                    ov = min(3, int(frame_num))
                frame_ids = torch.arange(int(frame_num), device=device).reshape(1, frame_num, 1)
                skip = skip & (frame_ids >= max(int(frame_num) - ov, 0))
            else:
                skip = torch.zeros_like(skip, dtype=torch.bool)
            semantic_extra_stats.update({
                "v67_phase2_carrier_mask": base_mask_name,
                "v67_phase2_label_control": "shuffled" if shuffled_label_control else "semantic_or_geometry",
                "v67_phase2_highd_quantile": float(quantile),
                "v67_phase2_highd_threshold": float(thr.item()),
            })
        elif base_mask_name in phase2_source_attention_masks:
            semantic_reason = f"v67_phase2_source_attention:{base_mask_name}"
            source_attention_top_quantile = float(quantile)
            source_attention_top_random_same_mass = bool(mask_name in phase2_random_base_map)
            source_control_score = torch.where(eligible, torch.ones_like(D, dtype=torch.float32), torch.zeros_like(D, dtype=torch.float32))
            skip = eligible
            semantic_extra_stats.update({
                "v67_source_attention_top_quantile": float(source_attention_top_quantile),
                "v67_source_attention_random_same_mass": bool(source_attention_top_random_same_mass),
            })
        elif base_mask_name in swa_redirection_source_masks:
            source_mask_tok = hmc_control.get("swa_redirection_source_mask_tok")
            if source_mask_tok is None:
                skip = torch.zeros_like(skip, dtype=torch.bool)
                source_control_score = torch.zeros_like(D, dtype=torch.float32)
                semantic_reason = "swa_redirection_source_mask_missing"
            else:
                source_mask = (
                    source_mask_tok.to(device=device, dtype=torch.float32)
                    .reshape(batch_size, frame_num, tokens_per_frame)
                    > 0.5
                )
                skip = source_mask & eligible
                source_control_score = torch.where(
                    skip,
                    torch.ones_like(D, dtype=torch.float32),
                    torch.zeros_like(D, dtype=torch.float32),
                )
                semantic_reason = "swa_redirection_source_mask_tok"
            semantic_extra_stats.update({
                "swa_redirection_source_mask_requested": True,
                "swa_redirection_source_mask_available": source_mask_tok is not None,
            })
        elif base_mask_name in {
            "semantic_role_negative",
            "semantic_role_source_skip",
            "semrole_negative",
            "semantic_role_positive",
            "semantic_role_stable",
            "semantic_role_anchor",
            "semantic_role_protect",
            "semantic_role_protected",
            "v36_synthetic_role_negative",
            "sem_z_dg_soft_resid",
            "semantic_z_dg_soft_resid",
            "semantic_conditioned_dg_soft_resid",
        }:
            if path == "frame_attention":
                R_sem_tok = hmc_control.get("R_frame_tok")
                role_stream_name = "R_frame_tok"
                if R_sem_tok is None:
                    R_sem_tok = hmc_control.get("R_sem_tok")
                    role_stream_name = "R_sem_tok"
            elif path == "chunk_attention":
                R_sem_tok = hmc_control.get("R_global_tok")
                role_stream_name = "R_global_tok"
                if R_sem_tok is None:
                    R_sem_tok = hmc_control.get("R_sem_tok")
                    role_stream_name = "R_sem_tok"
            else:
                R_sem_tok = hmc_control.get("R_sem_tok")
                role_stream_name = "R_sem_tok"
            if R_sem_tok is None:
                skip = torch.zeros_like(skip, dtype=torch.bool)
                semantic_reason = "semantic_role_missing"
            else:
                R = R_sem_tok.to(device=device, dtype=torch.long).reshape(batch_size, frame_num, tokens_per_frame)
                negative_short = R == 3
                positive_long = R == 1
                protected_neutral = R == 4
                if base_mask_name == "v36_synthetic_role_negative":
                    skip = negative_short & eligible
                elif base_mask_name in {"semantic_role_positive", "semantic_role_anchor"}:
                    skip = positive_long & eligible
                    source_control_score = torch.where(skip, torch.ones_like(D, dtype=torch.float32), torch.zeros_like(D, dtype=torch.float32))
                    semantic_reason = f"semantic_role_positive_long:{role_stream_name}"
                elif base_mask_name in {"semantic_role_protect", "semantic_role_protected"}:
                    skip = protected_neutral & eligible
                    source_control_score = torch.where(skip, torch.ones_like(D, dtype=torch.float32), torch.zeros_like(D, dtype=torch.float32))
                    semantic_reason = f"semantic_role_protect_neutral:{role_stream_name}"
                elif base_mask_name == "semantic_role_stable":
                    skip = (positive_long | protected_neutral) & eligible
                    source_control_score = torch.where(skip, torch.ones_like(D, dtype=torch.float32), torch.zeros_like(D, dtype=torch.float32))
                    semantic_reason = f"semantic_role_stable_positive_or_protect:{role_stream_name}"
                elif sem_z_dg_soft:
                    G_sem_tok = hmc_control.get("G_sem_tok")
                    if G_sem_tok is None:
                        skip = torch.zeros_like(skip, dtype=torch.bool)
                        source_control_score = torch.zeros_like(D, dtype=torch.float32)
                        semantic_reason = f"semantic_z_dg_group_missing:{role_stream_name}"
                    else:
                        G = G_sem_tok.to(device=device, dtype=torch.long).reshape(batch_size, frame_num, tokens_per_frame)
                        structure = G == 0
                        protected = protected | structure | positive_long | protected_neutral
                        risk = torch.zeros_like(D, dtype=torch.float32)
                        group_ids = torch.unique(G[eligible])
                        for gid in group_ids:
                            gm = (G == gid) & eligible & (~protected)
                            if int(gm.sum().item()) <= 1:
                                continue
                            vals = D[gm].float()
                            center = vals.mean()
                            high = torch.quantile(vals, 0.90)
                            denom = (high - center).clamp_min(1e-4)
                            risk[gm] = ((D[gm] - center) / denom).clamp(0.0, 1.0)
                        risk = risk * negative_short.float() * eligible.float()
                        source_control_score = risk.clamp(0.0, 1.0)
                        skip = source_control_score > 0.05
                        semantic_reason = f"semantic_z_dg_soft_risk:{role_stream_name}"
                elif base_mask_name in {"semantic_role_negative", "semantic_role_source_skip", "semrole_negative"}:
                    skip = skip & negative_short
                else:
                    skip = torch.zeros_like(skip, dtype=torch.bool)
                protected = protected | positive_long | protected_neutral
                if not sem_z_dg_soft and base_mask_name in {"semantic_role_negative", "semantic_role_source_skip", "semrole_negative"}:
                    semantic_reason = f"semantic_role_negative_short_and_highD:{role_stream_name}"
                if sem_z_dg_soft and R_sem_tok is not None and hmc_control.get("G_sem_tok") is not None:
                    semantic_reason = f"semantic_z_dg_soft_risk:{role_stream_name}"
                if base_mask_name == "v36_synthetic_role_negative":
                    semantic_reason = f"v36_synthetic_role_negative_without_D_filter:{role_stream_name}"
        else:
            semantic_reason = "not_semantic_mask"

        if random_same_mass:
            base_selected = int(skip.sum().item())
            flat_eligible = eligible.reshape(-1)
            skip_flat = torch.zeros_like(flat_eligible, dtype=torch.bool)
            if base_selected > 0 and int(flat_eligible.sum().item()) > 0:
                idx = torch.nonzero(flat_eligible, as_tuple=False).reshape(-1)
                token_idx = torch.arange(int(flat_eligible.numel()), device=device, dtype=torch.float32)
                chunk = float(hmc_control.get("semantic_action_chunk_idx", -1) if hmc_control else -1)
                path_offset = 17.0 if path == "chunk_attention" else 3.0
                scores = torch.frac(torch.sin((token_idx + 1.0 + chunk * 97.0 + path_offset) * 12.9898) * 43758.5453)
                k_select = min(base_selected, int(idx.numel()))
                top = torch.topk(scores[idx], k_select).indices
                skip_flat[idx[top]] = True
            skip = skip_flat.reshape_as(skip)
            semantic_reason = f"random_same_mass_control_from_{base_mask_name}:n={base_selected}"

        if base_mask_name in anchor_rescue_source_attention_masks:
            A_anchor_tok = hmc_control.get("A_anchor_tok")
            M_anchor_tok = hmc_control.get("A_anchor_mask_tok")
            if A_anchor_tok is None:
                anchor_mask_local = torch.zeros_like(skip, dtype=torch.bool)
                anchor_score_local = torch.zeros_like(D, dtype=torch.float32)
                anchor_reason = "semantic_anchor_missing"
            else:
                A_local = (
                    A_anchor_tok.to(device=device, dtype=torch.float32)
                    .reshape(batch_size, frame_num, tokens_per_frame)
                    .clamp(0.0, 1.0)
                )
                if M_anchor_tok is None:
                    M_local = A_local > 0.0
                else:
                    M_local = (
                        M_anchor_tok.to(device=device, dtype=torch.float32)
                        .reshape(batch_size, frame_num, tokens_per_frame)
                        .clamp(0.0, 1.0)
                        > 0.0
                    )
                anchor_mask_local = M_local & source_eligible_base
                anchor_score_local = torch.where(anchor_mask_local, A_local, torch.zeros_like(A_local))
                anchor_reason = f"semantic_anchor_bank:{hmc_control.get('semantic_anchor_mode', 'semantic')}"
            skip = skip & (~anchor_mask_local)
            source_attention_anchor_boost_mask = anchor_mask_local
            source_attention_anchor_boost_score = anchor_score_local
            anchor_count = int(anchor_mask_local.sum().item())
            anchor_eligible_count = max(int(source_eligible_base.sum().item()), 1)
            semantic_extra_stats.update({
                "semantic_anchor_rescue_source_tokens": float(anchor_count),
                "semantic_anchor_rescue_source_ratio": float(anchor_count / anchor_eligible_count),
                "semantic_anchor_rescue_source_score_mean": (
                    float(anchor_score_local[anchor_mask_local].mean().item()) if anchor_count > 0 else 0.0
                ),
                "semantic_anchor_rescue_source_score_max": float(anchor_score_local.max().item())
                if anchor_score_local.numel() else 0.0,
                "semantic_anchor_rescue_source_available": bool(anchor_count > 0),
                "semantic_anchor_rescue_source_reason": anchor_reason,
            })

        if explicit_source_attention_top_quantile is not None:
            try:
                explicit_top_q = float(explicit_source_attention_top_quantile)
            except (TypeError, ValueError):
                explicit_top_q = -1.0
            if explicit_top_q >= 0.0:
                source_attention_top_quantile = min(max(explicit_top_q, 0.0), 1.0)
                source_attention_top_random_same_mass = bool(
                    hmc_control.get("context_source_skip_source_attention_top_random_same_mass", False)
                )
                semantic_extra_stats.update({
                    "context_source_skip_source_attention_top_quantile": float(source_attention_top_quantile),
                    "context_source_skip_source_attention_top_random_same_mass": bool(source_attention_top_random_same_mass),
                })

        selected_for_stats = skip & eligible
        selected_count = int(selected_for_stats.sum().item())
        eligible_count = max(int(eligible.sum().item()), 1)
        semantic_extra_stats.update({
            "context_source_selected_token_count": selected_count,
            "context_source_selected_token_ratio": float(selected_count / eligible_count),
        })
        G_sem_tok = hmc_control.get("G_sem_tok")
        if G_sem_tok is not None and selected_count > 0:
            G = G_sem_tok.to(device=device, dtype=torch.long).reshape(batch_size, frame_num, tokens_per_frame)
            denom = max(selected_count, 1)
            semantic_extra_stats.update({
                "context_source_selected_group_structure_frac": float(((G == _CONTEXT_SEM_GROUP_STRUCTURE) & selected_for_stats).sum().item() / denom),
                "context_source_selected_group_static_frac": float(((G == _CONTEXT_SEM_GROUP_STATIC) & selected_for_stats).sum().item() / denom),
                "context_source_selected_group_movable_frac": float(((G == _CONTEXT_SEM_GROUP_MOVABLE) & selected_for_stats).sum().item() / denom),
                "context_source_selected_group_lowstuff_frac": float(((G == _CONTEXT_SEM_GROUP_LOWSTUFF) & selected_for_stats).sum().item() / denom),
                "context_source_selected_group_uncertain_frac": float(((G == _CONTEXT_SEM_GROUP_UNCERTAIN) & selected_for_stats).sum().item() / denom),
            })
        L_sem_tok = hmc_control.get("L_sem_tok")
        if L_sem_tok is not None and selected_count > 0:
            L = L_sem_tok.to(device=device, dtype=torch.long).reshape(batch_size, frame_num, tokens_per_frame)
            sky_sel = torch.zeros_like(selected_for_stats, dtype=torch.bool)
            for label_id in _CONTEXT_SKY_FINE_LABEL_IDS:
                sky_sel |= L == int(label_id)
            semantic_extra_stats["context_source_selected_fine_sky_frac"] = float((sky_sel & selected_for_stats).sum().item() / max(selected_count, 1))

        if source_control_score is None:
            source_control_score = skip.float()

        source_tokens_before = int(eligible.sum().item())
        source_skip_tokens = int(skip.sum().item())
        total_special = int(protected.sum().item())
        kept_special = int((protected & ~skip).sum().item())
        special_keep = float(kept_special / max(total_special, 1))
        if source_skip_tokens <= 0:
            source_tokens_after = source_tokens_before
            stats = dict(empty_stats)
            stats.update({
                "context_source_skip_mask": mask_name,
                "context_source_skip_mode": str(hmc_control.get("context_source_skip_mode", "hard")),
                "context_source_skip_threshold": float(thr.item()),
                "context_source_skip_quantile": float(quantile),
                "context_source_skip_reason": "no_tokens_selected",
                "context_source_skip_semantic_reason": semantic_reason,
                "source_tokens_before": source_tokens_before,
                "source_tokens_after": source_tokens_after,
                "special_token_keep_ratio": special_keep,
            })
            stats.update(frame_region_debug)
            stats.update(query_region_debug)
            stats.update(semantic_extra_stats)
            return None, stats

        mode = str(hmc_control.get("context_source_skip_mode", "hard")).lower()
        impl = str(hmc_control.get("context_source_skip_impl", "bias")).lower()
        trace_only_impl = impl in {"trace_only", "raw_qk_trace", "attention_trace_only"}
        boost_action = mode in {"boost", "soft_boost", "anchor_boost"} or impl in {
            "boost",
            "bias_boost",
            "source_boost",
            "anchor_boost",
        }
        soft_action = (
            trace_only_impl
            or boost_action
            or mode == "soft"
            or impl in {"v_only", "vonly", "value", "value_only"}
        )
        source_keep_tensor = None
        source_bias_values = None
        source_weights = None
        attention_mass_metric = "anchor_boost_attention_mass" if boost_action else "soft_attention_bias_mass"
        if boost_action:
            rho = float(hmc_control.get("context_source_skip_soft_rho", 0.2))
            source_keep_tensor = torch.ones_like(D, dtype=torch.float32)
            source_bias_values = torch.log1p((rho * source_control_score.float()).clamp_min(0.0))
            source_weights = None
        elif soft_action and mode != "soft":
            rho = float(hmc_control.get("context_source_skip_soft_rho", 0.5))
            min_keep = float(hmc_control.get("context_source_skip_soft_min_keep", 0.5))
            source_keep_tensor = (1.0 - rho * source_control_score.float()).clamp(min_keep, 1.0)
            source_weights = source_keep_tensor
        if impl == "compact_kv":
            keep = ((~skip) | protected).detach()
            if path == "frame_attention":
                source_keep_mask = keep.reshape(batch_size * frame_num, tokens_per_frame)
            else:
                source_keep_mask = keep.reshape(batch_size, frame_num * tokens_per_frame)
            source_bias_values = None
            source_weights = None
        elif boost_action:
            pass
        elif mode == "soft":
            rho = float(hmc_control.get("context_source_skip_soft_rho", 0.5))
            min_keep = float(hmc_control.get("context_source_skip_soft_min_keep", 0.5))
            source_keep_tensor = (1.0 - rho * source_control_score.float()).clamp(min_keep, 1.0)
            source_bias_values = torch.log(source_keep_tensor.clamp_min(1e-4))
            source_weights = source_keep_tensor
        else:
            source_bias_values = torch.zeros_like(D, dtype=torch.float32)
            source_bias_values = source_bias_values.masked_fill(skip, -1.0e4)
            source_weights = None

        def _attach_source_attention_top(control: Dict[str, Any], eligible_out: torch.Tensor) -> None:
            if source_attention_anchor_boost_mask is not None:
                if path == "frame_attention":
                    anchor_mask_out = source_attention_anchor_boost_mask.reshape(
                        batch_size * frame_num,
                        tokens_per_frame,
                    )
                    anchor_score_out = source_attention_anchor_boost_score.reshape(
                        batch_size * frame_num,
                        tokens_per_frame,
                    )
                else:
                    anchor_mask_out = source_attention_anchor_boost_mask.reshape(
                        batch_size,
                        frame_num * tokens_per_frame,
                    )
                    anchor_score_out = source_attention_anchor_boost_score.reshape(
                        batch_size,
                        frame_num * tokens_per_frame,
                    )
                control["source_attention_top_anchor_boost_mask"] = anchor_mask_out.detach()
                control["source_attention_top_anchor_boost_score"] = anchor_score_out.detach()
                control["source_attention_top_anchor_boost_rho"] = float(
                    hmc_control.get("context_source_skip_soft_rho", 0.5) or 0.5
                )
                control["stable_anchor_mask"] = anchor_mask_out.detach()
            if source_attention_top_quantile is None:
                return
            control["source_attention_top_quantile"] = float(source_attention_top_quantile)
            control["source_attention_top_random_same_mass"] = bool(source_attention_top_random_same_mass)
            control["source_attention_top_random_salt"] = int(hmc_control.get("semantic_action_chunk_idx", 0) if hmc_control else 0)
            control["source_attention_top_rho"] = float(hmc_control.get("context_source_skip_soft_rho", 0.5) or 0.5)
            control["source_attention_top_min_keep"] = float(hmc_control.get("context_source_skip_soft_min_keep", 0.5) or 0.5)
            control["source_attention_top_eligible_mask"] = eligible_out.detach()
            control["source_attention_top_boost_action"] = bool(boost_action)

        if impl == "compact_kv":
            bias = {
                "type": "compact_kv",
                "source_keep_mask": source_keep_mask,
            }
            if bool(hmc_control.get("context_source_skip_record_attention_mass", False)):
                bias["attention_mass_stats"] = []
                bias["attention_mass_max_queries"] = int(
                    hmc_control.get("context_source_skip_attention_mass_max_queries", 512) or 512
                )
            per_frame_keep = ((~skip) | protected).float().mean(dim=-1)
            empty_events = int((per_frame_keep <= 0.0).sum().item())
        elif impl in {"v_only", "vonly", "value", "value_only"}:
            if path == "frame_attention":
                affected_mask = skip.reshape(batch_size * frame_num, tokens_per_frame)
                source_weights_out = source_weights.reshape(batch_size * frame_num, tokens_per_frame)
            else:
                affected_mask = skip.reshape(batch_size, frame_num * tokens_per_frame)
                source_weights_out = source_weights.reshape(batch_size, frame_num * tokens_per_frame)
            bias = {
                "type": "source_soft",
                "mode": "v_only",
                "affected_mask": affected_mask.detach(),
                "source_weights": source_weights_out.detach(),
                "attention_mass_metric": "v_only_effective_value_mass",
            }
            if anchor_boost_mask:
                bias["stable_anchor_mask"] = affected_mask.detach()
            if query_mask_out is not None:
                bias["query_mask"] = query_mask_out
            _attach_source_attention_top(bias, affected_mask)
            if bool(hmc_control.get("context_source_skip_record_attention_mass", False)):
                bias["attention_mass_stats"] = []
                bias["attention_mass_max_queries"] = int(
                    hmc_control.get("context_source_skip_attention_mass_max_queries", 512) or 512
                )
            per_frame_keep = torch.ones_like(D[..., 0])
            empty_events = 0
        else:
            if path == "frame_attention":
                source_bias = source_bias_values.reshape(batch_size * frame_num, tokens_per_frame)
                if soft_action and bool(hmc_control.get("context_source_skip_record_attention_mass", False)):
                    bias = {
                        "type": "source_soft",
                        "mode": "trace_only" if trace_only_impl else "bias",
                        "affected_mask": skip.reshape(batch_size * frame_num, tokens_per_frame).detach(),
                        "source_bias_values": source_bias.detach(),
                        "attention_mass_metric": attention_mass_metric,
                    }
                    if anchor_boost_mask:
                        bias["stable_anchor_mask"] = bias["affected_mask"]
                    if query_mask_out is not None:
                        bias["query_mask"] = query_mask_out
                    _attach_source_attention_top(bias, skip.reshape(batch_size * frame_num, tokens_per_frame))
                    bias["attention_mass_stats"] = []
                    bias["attention_mass_max_queries"] = int(
                        hmc_control.get("context_source_skip_attention_mass_max_queries", 512) or 512
                    )
                else:
                    bias = source_bias[:, None, None, :].expand(-1, 1, tokens_per_frame, -1)
                per_frame_keep = (
                    source_keep_tensor.mean(dim=-1)
                    if source_keep_tensor is not None else ((~skip) | protected).float().mean(dim=-1)
                )
                empty_events = 0 if soft_action else int((per_frame_keep <= 0.0).sum().item())
            else:
                source_bias = source_bias_values.reshape(batch_size, frame_num * tokens_per_frame)
                seq_len = int(frame_num * tokens_per_frame)
                if soft_action and bool(hmc_control.get("context_source_skip_record_attention_mass", False)):
                    bias = {
                        "type": "source_soft",
                        "mode": "trace_only" if trace_only_impl else "bias",
                        "affected_mask": skip.reshape(batch_size, frame_num * tokens_per_frame).detach(),
                        "source_bias_values": source_bias.detach(),
                        "attention_mass_metric": attention_mass_metric,
                    }
                    if anchor_boost_mask:
                        bias["stable_anchor_mask"] = bias["affected_mask"]
                    if query_mask_out is not None:
                        bias["query_mask"] = query_mask_out
                    _attach_source_attention_top(bias, skip.reshape(batch_size, frame_num * tokens_per_frame))
                    bias["attention_mass_stats"] = []
                    bias["attention_mass_max_queries"] = int(
                        hmc_control.get("context_source_skip_attention_mass_max_queries", 512) or 512
                    )
                else:
                    bias = source_bias[:, None, None, :].expand(-1, 1, seq_len, -1)
                per_frame_keep = (
                    source_keep_tensor.mean(dim=-1)
                    if source_keep_tensor is not None else ((~skip) | protected).float().mean(dim=-1)
                )
                empty_events = 0 if soft_action else int((((~skip) | protected).reshape(batch_size, seq_len).sum(dim=-1) <= 0).sum().item())

        if source_keep_tensor is not None and bool(eligible.any()):
            keep_ratio = float(source_keep_tensor[eligible].mean().item())
        else:
            keep_ratio = float((source_tokens_before - source_skip_tokens) / max(source_tokens_before, 1))
        source_tokens_after = source_tokens_before if soft_action else source_tokens_before - source_skip_tokens
        stats = {
            "context_source_skip_applied": True,
            "context_source_skip_path": path,
            "context_source_skip_scope": scope,
            "context_source_skip_mode": mode,
            "context_source_skip_impl": impl,
            "context_source_skip_mask": mask_name,
            **frame_region_debug,
            **query_region_debug,
            "context_source_skip_threshold": float(thr.item()),
            "context_source_skip_quantile": float(quantile),
            "context_source_skip_semantic_reason": semantic_reason,
            "context_source_skip_soft_action": bool(soft_action),
            "context_source_boost_action": bool(boost_action),
            "source_skip_tokens": 0 if boost_action else source_skip_tokens,
            "source_control_tokens": source_skip_tokens,
            "source_boost_tokens": source_skip_tokens if boost_action else 0,
            "source_tokens_before": source_tokens_before,
            "source_tokens_after": source_tokens_after,
            "source_keep_ratio": keep_ratio,
            "source_skip_fraction": 0.0 if boost_action else float(source_skip_tokens / max(source_tokens_before, 1)),
            "source_control_fraction": float(source_skip_tokens / max(source_tokens_before, 1)),
            "source_boost_fraction": float(source_skip_tokens / max(source_tokens_before, 1)) if boost_action else 0.0,
            "semantic_anchor_boost_applied": bool(boost_action),
            "semantic_anchor_rescue_applied": bool(source_attention_anchor_boost_mask is not None),
            "attention_mass_metric": attention_mass_metric if soft_action else None,
            "attention_mass_requested": bool(hmc_control.get("context_source_skip_record_attention_mass", False))
            and (impl == "compact_kv" or isinstance(bias, dict)),
            "source_control_score_mean": float(source_control_score[eligible].mean().item()) if bool(eligible.any()) else 0.0,
            "source_control_score_max": float(source_control_score.max().item()) if source_control_score.numel() else 0.0,
            "source_weight_mean": float(source_keep_tensor[eligible].mean().item())
            if source_keep_tensor is not None and bool(eligible.any()) else None,
            "source_weight_min": float(source_keep_tensor[eligible].min().item())
            if source_keep_tensor is not None and bool(eligible.any()) else None,
            "per_frame_source_keep_ratio_min": float(per_frame_keep.min().item()) if per_frame_keep.numel() else 1.0,
            "per_frame_source_keep_ratio_mean": float(per_frame_keep.mean().item()) if per_frame_keep.numel() else 1.0,
            "special_token_kept_count": kept_special,
            "special_token_total_count": total_special,
            "special_token_keep_ratio": special_keep,
            "empty_source_events": empty_events,
        }
        stats.update(semantic_extra_stats)
        if isinstance(bias, dict):
            if bias.get("type") == "source_soft":
                bias["source_attention_head_indices"] = str(
                    hmc_control.get("context_source_skip_head_indices", "") or ""
                )
            dump_dir = str(hmc_control.get("context_source_skip_attention_map_dump_dir", "") or "").strip()
            if dump_dir and bias.get("type") == "source_soft":
                bias["source_attention_map_dump_dir"] = dump_dir
                bias["source_attention_map_dump_chunk_idx"] = int(
                    hmc_control.get("semantic_action_chunk_idx", -1) if hmc_control else -1
                )
                bias["source_attention_map_dump_hook_path"] = str(path)
                bias["source_attention_map_dump_max_queries"] = int(
                    hmc_control.get("context_source_skip_attention_map_dump_max_queries", 64) or 64
                )
                bias["source_attention_map_dump_dtype"] = str(
                    hmc_control.get("context_source_skip_attention_map_dump_dtype", "float16") or "float16"
                )
                bias["source_attention_map_dump_full_query_marginal"] = bool(
                    hmc_control.get("context_source_skip_attention_map_dump_full_query_marginal", False)
                )
                bias["source_attention_map_dump_head_marginal"] = bool(
                    hmc_control.get("context_source_skip_attention_map_dump_head_marginal", False)
                )
                bias["source_attention_map_dump_query_block"] = int(
                    hmc_control.get("context_source_skip_attention_map_dump_query_block", 32) or 32
                )
            return bias, stats
        return bias.to(dtype=dtype), stats

    def _make_ttt_apply_gate(
        self,
        hmc_control: Optional[Dict[str, Any]],
        *,
        batch_size: int,
        frame_num: int,
        tokens_per_frame: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if not hmc_control or hmc_control.get("identity_hooks", False):
            return None
        if not hmc_control.get("enable_ttt_apply_control", False):
            return None
        rho = float(hmc_control.get("rho_ttt_apply", 0.0))
        if rho == 0.0:
            return None
        D_tok = hmc_control.get("D_tok")
        P_ref = hmc_control.get("P_ref")
        if D_tok is None:
            return None
        D = D_tok.to(device=device, dtype=torch.float32).reshape(batch_size, frame_num, tokens_per_frame)
        min_gate = float(hmc_control.get("ttt_apply_min_gate", 0.0))
        gate = (1.0 - rho * D).clamp(min_gate, 1.0)
        if P_ref is not None:
            ref = P_ref.to(device=device, dtype=torch.float32).reshape(batch_size, frame_num, tokens_per_frame)
            gate = torch.where(ref > 0.5, torch.ones_like(gate), gate)
        return gate.unsqueeze(-1).to(dtype=dtype)

    def _make_swa_prev_source_gate(
        self,
        hmc_control: Optional[Dict[str, Any]],
        *,
        history_tokens: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if not hmc_control or hmc_control.get("identity_hooks", False):
            return None
        if not hmc_control.get("enable_swa_read_control", False):
            return None
        rho = float(hmc_control.get("beta_swa", 0.0))
        if rho == 0.0 or history_tokens <= 0:
            return None
        D_prev = hmc_control.get("D_prev_patch")
        if D_prev is None:
            return None
        D = D_prev.to(device=device, dtype=torch.float32).reshape(-1)
        if D.numel() < history_tokens:
            # SWA history can span more than the immediately previous chunk.
            # Gate only the most recent previous source tokens and leave older
            # cached sources unchanged.
            prefix = torch.zeros(history_tokens - D.numel(), device=device, dtype=torch.float32)
            D = torch.cat([prefix, D], dim=0)
        elif D.numel() != history_tokens:
            # Keep the most recent source tokens if the persisted summary spans
            # more frames than the active SWA cache.
            D = D[-history_tokens:]
        min_gate = float(hmc_control.get("swa_gate_min", 0.85))
        gate = (1.0 - rho * D).clamp(min_gate, 1.0)
        return gate.reshape(1, 1, history_tokens, 1).to(dtype=dtype)

    @staticmethod
    def _swa_prev_ttt_stable_anchor_layer_enabled(
        hmc_control: Optional[Dict[str, Any]],
        *,
        layer_idx: int,
        n_layers: int,
    ) -> bool:
        if not hmc_control or hmc_control.get("identity_hooks", False):
            return False
        if not hmc_control.get("enable_swa_prev_ttt_stable_anchor_gate", False):
            return False
        mode = str(hmc_control.get("swa_prev_ttt_stable_anchor_gate_layer_mode", "last"))
        if mode == "all":
            return True
        if mode == "first":
            return int(layer_idx) == 0
        if mode == "last":
            return int(layer_idx) == max(0, int(n_layers) - 1)
        if mode == "single":
            return int(layer_idx) == int(hmc_control.get("swa_prev_ttt_stable_anchor_gate_single_layer", -1))
        return False

    def _make_swa_prev_ttt_stable_anchor_gate(
        self,
        hmc_control: Optional[Dict[str, Any]],
        *,
        batch_size: int,
        tokens_per_frame: int,
        history_tokens: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
        stats: Dict[str, Any] = {
            "swa_prev_ttt_stable_anchor_gate_applied": False,
            "swa_prev_ttt_stable_anchor_gate_tokens": 0,
            "swa_prev_ttt_stable_anchor_gate_available": False,
        }
        if not hmc_control or hmc_control.get("identity_hooks", False):
            return None, stats
        if not hmc_control.get("enable_swa_prev_ttt_stable_anchor_gate", False):
            return None, stats
        rho = float(hmc_control.get("swa_prev_ttt_stable_anchor_gate_rho", 0.0))
        if rho == 0.0 or history_tokens <= 0:
            return None, stats
        mask = Pi3._swa_trace_source_values(
            hmc_control,
            "prev_ttt_stable_anchor_mask_patch",
            batch_size=batch_size,
            history_tokens=history_tokens,
            tokens_per_frame=tokens_per_frame,
            device=device,
            dtype=torch.float32,
        )
        if mask is None:
            stats["swa_prev_ttt_stable_anchor_gate_reason"] = "missing_prev_ttt_stable_anchor_mask_patch"
            return None, stats
        score = (mask >= 0.5).to(device=device, dtype=torch.float32)
        selected = score > 0.5
        selected_count = int(selected.sum().item())
        stats.update({
            "swa_prev_ttt_stable_anchor_gate_available": True,
            "swa_prev_ttt_stable_anchor_gate_source_token_count": int(score.numel()),
            "swa_prev_ttt_stable_anchor_gate_selected_token_count": selected_count,
            "swa_prev_ttt_stable_anchor_gate_selected_frac": float(score.mean().item()) if score.numel() else 0.0,
        })
        if selected_count <= 0:
            stats["swa_prev_ttt_stable_anchor_gate_reason"] = "empty_prev_ttt_stable_anchor_mask"
            return None, stats
        min_gate = min(max(float(hmc_control.get("swa_prev_ttt_stable_anchor_gate_min", 0.85)), 0.0), 1.0)
        gate_values = (1.0 - rho * score).clamp(min_gate, 1.0).to(dtype=dtype)
        gate_delta = (1.0 - gate_values.detach().float()).abs()
        stats.update({
            "swa_prev_ttt_stable_anchor_gate_applied": True,
            "swa_prev_ttt_stable_anchor_gate_rho": rho,
            "swa_prev_ttt_stable_anchor_gate_min": min_gate,
            "swa_prev_ttt_stable_anchor_gate_tokens": selected_count,
            "swa_prev_ttt_stable_anchor_gate_mean": float(gate_values.detach().float().mean().item()),
            "swa_prev_ttt_stable_anchor_gate_p10": float(torch.quantile(gate_values.detach().float(), 0.10).item()),
            "swa_prev_ttt_stable_anchor_gate_p50": float(torch.quantile(gate_values.detach().float(), 0.50).item()),
            "swa_prev_ttt_stable_anchor_gate_p90": float(torch.quantile(gate_values.detach().float(), 0.90).item()),
            "swa_prev_ttt_stable_anchor_gate_mean_abs_delta": float(gate_delta.mean().item()),
            "swa_prev_ttt_stable_anchor_gate_max_abs_delta": float(gate_delta.max().item()),
        })
        return gate_values.reshape(int(batch_size), 1, int(history_tokens), 1).to(dtype=dtype), stats

    @staticmethod
    def _swa_prev_ttt_anchor_query_soft_layer_enabled(
        hmc_control: Optional[Dict[str, Any]],
        *,
        layer_idx: int,
        n_layers: int,
    ) -> bool:
        if not hmc_control or hmc_control.get("identity_hooks", False):
            return False
        if not hmc_control.get("enable_swa_prev_ttt_anchor_query_soft", False):
            return False
        mode = str(hmc_control.get("swa_prev_ttt_anchor_query_soft_layer_mode", "last"))
        if mode == "all":
            return True
        if mode == "first":
            return int(layer_idx) == 0
        if mode == "last":
            return int(layer_idx) == max(0, int(n_layers) - 1)
        if mode == "single":
            return int(layer_idx) == int(hmc_control.get("swa_prev_ttt_anchor_query_soft_single_layer", -1))
        return False

    def _make_swa_prev_ttt_anchor_query_soft_control(
        self,
        hmc_control: Optional[Dict[str, Any]],
        *,
        batch_size: int,
        tokens_per_frame: int,
        history_tokens: int,
        current_tokens: int,
        device: torch.device,
    ) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        stats: Dict[str, Any] = {
            "swa_prev_ttt_anchor_query_soft_available": False,
            "swa_prev_ttt_anchor_query_soft_applied": False,
            "swa_prev_ttt_anchor_query_soft_source_tokens": 0,
        }
        if not hmc_control or hmc_control.get("identity_hooks", False):
            return None, stats
        if not hmc_control.get("enable_swa_prev_ttt_anchor_query_soft", False):
            return None, stats
        rho = float(hmc_control.get("swa_prev_ttt_anchor_query_soft_rho", 0.0))
        if rho == 0.0 or history_tokens <= 0:
            stats["swa_prev_ttt_anchor_query_soft_reason"] = "rho_zero_or_empty_history"
            return None, stats
        mask = Pi3._swa_trace_source_values(
            hmc_control,
            "prev_ttt_stable_anchor_mask_patch",
            batch_size=batch_size,
            history_tokens=history_tokens,
            tokens_per_frame=tokens_per_frame,
            device=device,
            dtype=torch.float32,
        )
        if mask is None:
            stats["swa_prev_ttt_anchor_query_soft_reason"] = "missing_prev_ttt_stable_anchor_mask_patch"
            return None, stats
        stable_mask = (mask >= 0.5).reshape(int(batch_size), int(history_tokens))
        selected_count = int(stable_mask.sum().item())
        stats.update({
            "swa_prev_ttt_anchor_query_soft_available": True,
            "swa_prev_ttt_anchor_query_soft_source_token_count": int(stable_mask.numel()),
            "swa_prev_ttt_anchor_query_soft_source_tokens": selected_count,
            "swa_prev_ttt_anchor_query_soft_source_frac": float(stable_mask.float().mean().item())
            if stable_mask.numel() else 0.0,
            "swa_prev_ttt_anchor_query_soft_rho": float(rho),
            "swa_prev_ttt_anchor_query_soft_min_keep": float(
                hmc_control.get("swa_prev_ttt_anchor_query_soft_min_keep", 0.5)
            ),
            "swa_prev_ttt_anchor_query_soft_query_head_frac_threshold": float(
                hmc_control.get("swa_prev_ttt_anchor_query_soft_query_head_frac_threshold", 0.75)
            ),
            "swa_prev_ttt_anchor_query_soft_topk": int(
                hmc_control.get("swa_prev_ttt_anchor_query_soft_topk", 8)
            ),
            "swa_prev_ttt_anchor_query_soft_query_block_size": int(
                hmc_control.get("swa_prev_ttt_anchor_query_soft_query_block_size", 64)
            ),
        })
        if selected_count <= 0:
            stats["swa_prev_ttt_anchor_query_soft_reason"] = "empty_prev_ttt_stable_anchor_mask"
            return None, stats
        attention_mass_stats: List[Dict[str, Any]] = []
        control = {
            "type": "prev_ttt_anchor_query_soft",
            "stable_anchor_mask": stable_mask.to(device=device),
            "history_tokens": int(history_tokens),
            "rho": float(rho),
            "min_keep": float(hmc_control.get("swa_prev_ttt_anchor_query_soft_min_keep", 0.5)),
            "query_head_frac_threshold": float(
                hmc_control.get("swa_prev_ttt_anchor_query_soft_query_head_frac_threshold", 0.75)
            ),
            "topk": int(hmc_control.get("swa_prev_ttt_anchor_query_soft_topk", 8)),
            "query_block_size": int(hmc_control.get("swa_prev_ttt_anchor_query_soft_query_block_size", 64)),
            "attention_mass_stats": attention_mass_stats,
            "attention_mass_max_queries": int(
                hmc_control.get("swa_prev_ttt_anchor_query_soft_attention_mass_max_queries", 64)
            ),
            "attention_mass_metric": "prev_ttt_anchor_query_soft",
        }
        return control, stats

    @staticmethod
    def _swa_prev_ttt_tracked_instance_query_soft_layer_enabled(
        hmc_control: Optional[Dict[str, Any]],
        *,
        layer_idx: int,
        n_layers: int,
    ) -> bool:
        if not hmc_control or hmc_control.get("identity_hooks", False):
            return False
        if not (
            hmc_control.get("enable_swa_prev_ttt_tracked_instance_query_soft_trace", False)
            or hmc_control.get("enable_swa_prev_ttt_tracked_instance_query_soft_action", False)
        ):
            return False
        mode = str(hmc_control.get("swa_prev_ttt_tracked_instance_query_soft_layer_mode", "last"))
        if mode == "all":
            return True
        if mode == "first":
            return int(layer_idx) == 0
        if mode == "last":
            return int(layer_idx) == max(0, int(n_layers) - 1)
        if mode == "single":
            return int(layer_idx) == int(
                hmc_control.get("swa_prev_ttt_tracked_instance_query_soft_single_layer", -1)
            )
        return False

    def _make_swa_prev_ttt_tracked_instance_query_soft_trace_control(
        self,
        hmc_control: Optional[Dict[str, Any]],
        *,
        batch_size: int,
        tokens_per_frame: int,
        history_tokens: int,
        current_tokens: int,
        device: torch.device,
    ) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        stats: Dict[str, Any] = {
            "swa_prev_ttt_tracked_instance_query_soft_available": False,
            "swa_prev_ttt_tracked_instance_query_soft_applied": False,
            "swa_prev_ttt_tracked_instance_query_soft_trace_only": True,
            "swa_prev_ttt_tracked_instance_query_soft_action_requested": False,
            "swa_prev_ttt_tracked_instance_query_soft_runtime_action_allowed": False,
            "swa_prev_ttt_tracked_instance_query_soft_source_tokens": 0,
        }
        if not hmc_control or hmc_control.get("identity_hooks", False):
            return None, stats
        trace_requested = bool(
            hmc_control.get("enable_swa_prev_ttt_tracked_instance_query_soft_trace", False)
        )
        action_requested = bool(
            hmc_control.get("enable_swa_prev_ttt_tracked_instance_query_soft_action", False)
        )
        action_runtime_allowed = bool(
            action_requested
            and hmc_control.get(
                "swa_prev_ttt_tracked_instance_query_soft_action_runtime_authorized",
                False,
            )
        )
        stats.update({
            "swa_prev_ttt_tracked_instance_query_soft_trace_requested": trace_requested,
            "swa_prev_ttt_tracked_instance_query_soft_action_requested": action_requested,
            "swa_prev_ttt_tracked_instance_query_soft_runtime_action_allowed": action_runtime_allowed,
            "swa_prev_ttt_tracked_instance_query_soft_trace_only": not action_runtime_allowed,
        })
        if not (trace_requested or action_requested):
            return None, stats
        if history_tokens <= 0:
            stats["swa_prev_ttt_tracked_instance_query_soft_reason"] = "empty_history"
            return None, stats
        mask = Pi3._swa_trace_source_values(
            hmc_control,
            "prev_ttt_tracked_instance_anchor_mask_patch",
            batch_size=batch_size,
            history_tokens=history_tokens,
            tokens_per_frame=tokens_per_frame,
            device=device,
            dtype=torch.float32,
        )
        if mask is None:
            stats["swa_prev_ttt_tracked_instance_query_soft_reason"] = (
                "missing_prev_ttt_tracked_instance_anchor_mask_patch"
            )
            return None, stats
        seed_ids = Pi3._swa_trace_source_values(
            hmc_control,
            "prev_ttt_tracked_instance_anchor_seed_patch",
            batch_size=batch_size,
            history_tokens=history_tokens,
            tokens_per_frame=tokens_per_frame,
            device=device,
            dtype=torch.int64,
        )
        source_instance_ids = Pi3._swa_trace_source_values(
            hmc_control,
            "prev_ttt_tracked_instance_anchor_id_patch",
            batch_size=batch_size,
            history_tokens=history_tokens,
            tokens_per_frame=tokens_per_frame,
            device=device,
            dtype=torch.int64,
        )

        def _current_values(key: str) -> Optional[torch.Tensor]:
            raw = hmc_control.get(key)
            if not torch.is_tensor(raw) or int(current_tokens) <= 0:
                return None
            vals = raw.detach().to(device=device, dtype=torch.int64).reshape(-1)
            needed = int(batch_size) * int(current_tokens)
            if int(vals.numel()) < needed:
                pad = torch.full(
                    (needed - int(vals.numel()),),
                    -1,
                    device=device,
                    dtype=torch.int64,
                )
                vals = torch.cat([vals, pad], dim=0)
            return vals[:needed].reshape(int(batch_size), int(current_tokens))

        query_seed_ids = _current_values("stage_c_seed_global_track_idx_tok")
        query_instance_ids = _current_values("stage_c_masklet_instance_idx_tok")
        direct_match_mode = str(
            hmc_control.get("swa_prev_ttt_tracked_instance_query_soft_direct_match_mode", "any")
        ).strip().lower()
        if direct_match_mode not in {"any", "same_seed", "same_masklet"}:
            direct_match_mode = "any"
        tracked_mask = (mask >= 0.5).reshape(int(batch_size), int(history_tokens))
        selected_count = int(tracked_mask.sum().item())
        stats.update({
            "swa_prev_ttt_tracked_instance_query_soft_available": True,
            "swa_prev_ttt_tracked_instance_query_soft_source_token_count": int(tracked_mask.numel()),
            "swa_prev_ttt_tracked_instance_query_soft_source_tokens": selected_count,
            "swa_prev_ttt_tracked_instance_query_soft_source_frac": float(tracked_mask.float().mean().item())
            if tracked_mask.numel() else 0.0,
            "swa_prev_ttt_tracked_instance_query_soft_rho": float(
                hmc_control.get("swa_prev_ttt_tracked_instance_query_soft_rho", 0.0)
            ),
            "swa_prev_ttt_tracked_instance_query_soft_min_keep": float(
                hmc_control.get("swa_prev_ttt_tracked_instance_query_soft_min_keep", 0.5)
            ),
            "swa_prev_ttt_tracked_instance_query_soft_query_head_frac_threshold": float(
                hmc_control.get("swa_prev_ttt_tracked_instance_query_soft_query_head_frac_threshold", 0.75)
            ),
            "swa_prev_ttt_tracked_instance_query_soft_topk": int(
                hmc_control.get("swa_prev_ttt_tracked_instance_query_soft_topk", 4)
            ),
            "swa_prev_ttt_tracked_instance_query_soft_query_block_size": int(
                hmc_control.get("swa_prev_ttt_tracked_instance_query_soft_query_block_size", 64)
            ),
            "swa_prev_ttt_tracked_instance_query_soft_min_direct_witness_seeds": int(
                hmc_control.get("swa_prev_ttt_tracked_instance_query_soft_min_direct_witness_seeds", 4)
            ),
            "swa_prev_ttt_tracked_instance_query_soft_seed_ids_available": bool(seed_ids is not None),
            "swa_prev_ttt_tracked_instance_query_soft_source_instance_ids_available": bool(
                source_instance_ids is not None
            ),
            "swa_prev_ttt_tracked_instance_query_soft_query_seed_ids_available": bool(
                query_seed_ids is not None
            ),
            "swa_prev_ttt_tracked_instance_query_soft_query_instance_ids_available": bool(
                query_instance_ids is not None
            ),
            "swa_prev_ttt_tracked_instance_query_soft_direct_match_mode": direct_match_mode,
            "swa_prev_ttt_tracked_instance_query_soft_trace_only": not action_runtime_allowed,
            "swa_prev_ttt_tracked_instance_query_soft_runtime_action_allowed": action_runtime_allowed,
        })
        if action_requested and not action_runtime_allowed:
            stats["swa_prev_ttt_tracked_instance_query_soft_action_blocker"] = (
                "runtime_authorization_false"
            )
        if selected_count <= 0:
            stats["swa_prev_ttt_tracked_instance_query_soft_reason"] = (
                "empty_prev_ttt_tracked_instance_anchor_mask"
            )
            return None, stats
        attention_mass_stats: List[Dict[str, Any]] = []
        control = {
            "type": "prev_ttt_anchor_query_soft",
            "mode": "action" if action_runtime_allowed else "trace_only",
            "trace_only": not action_runtime_allowed,
            "diagnostic_only": not action_runtime_allowed,
            "stat_prefix": "prev_ttt_tracked_instance_query_soft",
            "stable_anchor_mask": tracked_mask.to(device=device),
            "source_seed_ids": seed_ids.reshape(int(batch_size), int(history_tokens)).to(device=device)
            if seed_ids is not None else None,
            "source_instance_ids": source_instance_ids.reshape(int(batch_size), int(history_tokens)).to(device=device)
            if source_instance_ids is not None else None,
            "query_seed_ids": query_seed_ids,
            "query_instance_ids": query_instance_ids,
            "direct_match_mode": direct_match_mode,
            "history_tokens": int(history_tokens),
            "rho": float(hmc_control.get("swa_prev_ttt_tracked_instance_query_soft_rho", 0.0)),
            "min_keep": float(hmc_control.get("swa_prev_ttt_tracked_instance_query_soft_min_keep", 0.5)),
            "query_head_frac_threshold": float(
                hmc_control.get("swa_prev_ttt_tracked_instance_query_soft_query_head_frac_threshold", 0.75)
            ),
            "topk": int(hmc_control.get("swa_prev_ttt_tracked_instance_query_soft_topk", 4)),
            "query_block_size": int(
                hmc_control.get("swa_prev_ttt_tracked_instance_query_soft_query_block_size", 64)
            ),
            "min_direct_witness_seeds": int(
                hmc_control.get("swa_prev_ttt_tracked_instance_query_soft_min_direct_witness_seeds", 4)
            ),
            "attention_mass_stats": attention_mass_stats,
            "attention_mass_max_queries": int(
                hmc_control.get("swa_prev_ttt_tracked_instance_query_soft_attention_mass_max_queries", 64)
            ),
            "attention_mass_metric": (
                "prev_ttt_tracked_instance_query_soft_action"
                if action_runtime_allowed
                else "prev_ttt_tracked_instance_query_soft_trace_only"
            ),
        }
        return control, stats

    @staticmethod
    def _swa_overlap_layer_enabled(
        hmc_control: Optional[Dict[str, Any]],
        *,
        layer_idx: int,
        n_layers: int,
    ) -> bool:
        if not hmc_control or hmc_control.get("identity_hooks", False):
            return False
        if not hmc_control.get("enable_swa_overlap_bias", False):
            return False
        mode = str(hmc_control.get("swa_overlap_bias_layer_mode", "last"))
        if mode == "all":
            return True
        if mode == "first":
            return int(layer_idx) == 0
        if mode == "last":
            return int(layer_idx) == max(0, int(n_layers) - 1)
        if mode == "single":
            return int(layer_idx) == int(hmc_control.get("swa_overlap_bias_single_layer", -1))
        return False

    def _make_external_v84_anchor_source_score(
        self,
        *,
        mask_csv: str,
        variant: str,
        seq_filter: str,
        curr_chunk: int,
        batch_size: int,
        source_tokens: int,
        overlap_frames: int,
        tokens_per_frame: int,
        device: torch.device,
    ) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
        stats: Dict[str, Any] = {
            "swa_overlap_bias_external_mask_available": False,
            "swa_overlap_bias_external_mask_reason": "",
            "swa_overlap_bias_external_mask_csv": str(mask_csv or ""),
            "swa_overlap_bias_external_mask_variant": str(variant or ""),
            "swa_overlap_bias_external_mask_seq": str(seq_filter or ""),
            "swa_overlap_bias_external_mask_curr_chunk": int(curr_chunk),
            "swa_overlap_bias_external_mask_rows_matching": 0,
            "swa_overlap_bias_external_mask_source_tokens_selected": 0,
        }
        if not mask_csv:
            stats["swa_overlap_bias_external_mask_reason"] = "missing_mask_csv"
            return None, stats
        path = Path(mask_csv)
        if not path.exists():
            stats["swa_overlap_bias_external_mask_reason"] = "mask_csv_not_found"
            return None, stats
        if source_tokens <= 0 or overlap_frames <= 0 or tokens_per_frame <= 0:
            stats["swa_overlap_bias_external_mask_reason"] = "invalid_window_shape"
            return None, stats
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
        except Exception as exc:  # noqa: BLE001
            stats["swa_overlap_bias_external_mask_reason"] = f"mask_csv_read_error:{type(exc).__name__}"
            return None, stats

        seq_norm = str(seq_filter or "").zfill(2) if str(seq_filter or "").strip() else ""
        selected_rows: List[Dict[str, str]] = []
        for row in rows:
            if str(row.get("side", "")).strip().lower() != "source":
                continue
            if str(row.get("variant", "")).strip() != str(variant):
                continue
            if seq_norm and str(row.get("seq", "")).zfill(2) != seq_norm:
                continue
            try:
                row_curr_chunk = int(float(str(row.get("curr_chunk", "")).strip()))
            except ValueError:
                continue
            if row_curr_chunk != int(curr_chunk):
                continue
            selected_rows.append(row)

        score = torch.zeros((int(batch_size), int(source_tokens)), device=device, dtype=torch.float32)
        stats["swa_overlap_bias_external_mask_rows_matching"] = int(len(selected_rows))
        if not selected_rows:
            stats.update({
                "swa_overlap_bias_external_mask_available": True,
                "swa_overlap_bias_external_mask_reason": "no_rows_for_current_chunk",
                "swa_overlap_bias_external_mask_source_tokens_selected": 0,
            })
            return score, stats

        local_frames: List[int] = []
        for row in selected_rows:
            try:
                local_frames.append(int(float(str(row.get("local_frame", "")).strip())))
            except ValueError:
                continue
        if not local_frames:
            stats["swa_overlap_bias_external_mask_reason"] = "no_valid_local_frames"
            return score, stats
        source_frame_base = max(0, max(local_frames) - int(overlap_frames) + 1)
        patch_start = int(getattr(self, "patch_start_idx", 0))
        patch_tokens_per_frame = max(0, int(tokens_per_frame) - patch_start)
        selected_token_indices: set[int] = set()
        skipped_out_of_window = 0
        skipped_bad_patch = 0
        for row in selected_rows:
            try:
                local_frame = int(float(str(row.get("local_frame", "")).strip()))
                patch_token_index = int(float(str(row.get("patch_token_index", "")).strip()))
            except ValueError:
                continue
            if patch_token_index < 0 or (patch_tokens_per_frame > 0 and patch_token_index >= patch_tokens_per_frame):
                skipped_bad_patch += 1
                continue
            window_frame = local_frame - source_frame_base
            if window_frame < 0 or window_frame >= int(overlap_frames):
                skipped_out_of_window += 1
                continue
            source_index = int(window_frame) * int(tokens_per_frame) + patch_start + patch_token_index
            if 0 <= source_index < int(source_tokens):
                selected_token_indices.add(source_index)
            else:
                skipped_out_of_window += 1
        if selected_token_indices:
            idx = torch.tensor(sorted(selected_token_indices), device=device, dtype=torch.long)
            score[:, idx] = 1.0
        stats.update({
            "swa_overlap_bias_external_mask_available": True,
            "swa_overlap_bias_external_mask_reason": "ok" if selected_token_indices else "empty_after_window_mapping",
            "swa_overlap_bias_external_mask_source_frame_base": int(source_frame_base),
            "swa_overlap_bias_external_mask_overlap_frames": int(overlap_frames),
            "swa_overlap_bias_external_mask_patch_start_idx": int(patch_start),
            "swa_overlap_bias_external_mask_patch_tokens_per_frame": int(patch_tokens_per_frame),
            "swa_overlap_bias_external_mask_source_tokens_selected": int(len(selected_token_indices)),
            "swa_overlap_bias_external_mask_rows_skipped_out_of_window": int(skipped_out_of_window),
            "swa_overlap_bias_external_mask_rows_skipped_bad_patch": int(skipped_bad_patch),
        })
        return score, stats

    def _make_external_v92_policy_side_score(
        self,
        *,
        mask_csv: str,
        variant: str,
        seq_filter: str,
        curr_chunk: int,
        side: str,
        batch_size: int,
        token_count: int,
        overlap_frames: int,
        tokens_per_frame: int,
        device: torch.device,
    ) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
        side_norm = str(side or "").strip().lower()
        prefix = f"swa_overlap_bias_external_v92_policy_{side_norm}"
        stats: Dict[str, Any] = {
            f"{prefix}_available": False,
            f"{prefix}_reason": "",
            f"{prefix}_mask_csv": str(mask_csv or ""),
            f"{prefix}_variant": str(variant or ""),
            f"{prefix}_seq": str(seq_filter or ""),
            f"{prefix}_curr_chunk": int(curr_chunk),
            f"{prefix}_rows_matching": 0,
            f"{prefix}_tokens_selected": 0,
        }
        if side_norm not in {"query", "source"}:
            stats[f"{prefix}_reason"] = "invalid_side"
            return None, stats
        if not mask_csv:
            stats[f"{prefix}_reason"] = "missing_mask_csv"
            return None, stats
        path = Path(mask_csv)
        if not path.exists():
            stats[f"{prefix}_reason"] = "mask_csv_not_found"
            return None, stats
        if token_count <= 0 or overlap_frames <= 0 or tokens_per_frame <= 0:
            stats[f"{prefix}_reason"] = "invalid_window_shape"
            return None, stats
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
        except Exception as exc:  # noqa: BLE001
            stats[f"{prefix}_reason"] = f"mask_csv_read_error:{type(exc).__name__}"
            return None, stats

        seq_norm = str(seq_filter or "").zfill(2) if str(seq_filter or "").strip() else ""
        selected_rows: List[Dict[str, str]] = []
        for row in rows:
            if str(row.get("side", "")).strip().lower() != side_norm:
                continue
            if str(row.get("variant", "")).strip() != str(variant):
                continue
            if seq_norm and str(row.get("seq", "")).zfill(2) != seq_norm:
                continue
            try:
                row_curr_chunk = int(float(str(row.get("curr_chunk", "")).strip()))
            except ValueError:
                continue
            if row_curr_chunk != int(curr_chunk):
                continue
            selected_rows.append(row)

        score = torch.zeros((int(batch_size), int(token_count)), device=device, dtype=torch.float32)
        stats[f"{prefix}_rows_matching"] = int(len(selected_rows))
        if not selected_rows:
            stats.update({
                f"{prefix}_available": True,
                f"{prefix}_reason": "no_rows_for_current_chunk",
                f"{prefix}_tokens_selected": 0,
            })
            return score, stats

        local_frames: List[int] = []
        for row in selected_rows:
            try:
                local_frames.append(int(float(str(row.get("local_frame", "")).strip())))
            except ValueError:
                continue
        if not local_frames:
            stats[f"{prefix}_reason"] = "no_valid_local_frames"
            return score, stats
        frame_base = max(0, max(local_frames) - int(overlap_frames) + 1)
        patch_start = int(getattr(self, "patch_start_idx", 0))
        patch_tokens_per_frame = max(0, int(tokens_per_frame) - patch_start)
        selected_token_indices: set[int] = set()
        skipped_out_of_window = 0
        skipped_bad_patch = 0
        for row in selected_rows:
            try:
                local_frame = int(float(str(row.get("local_frame", "")).strip()))
                patch_token_index = int(float(str(row.get("patch_token_index", "")).strip()))
            except ValueError:
                continue
            if patch_token_index < 0 or (patch_tokens_per_frame > 0 and patch_token_index >= patch_tokens_per_frame):
                skipped_bad_patch += 1
                continue
            window_frame = local_frame - frame_base
            if window_frame < 0 or window_frame >= int(overlap_frames):
                skipped_out_of_window += 1
                continue
            token_index = int(window_frame) * int(tokens_per_frame) + patch_start + patch_token_index
            if 0 <= token_index < int(token_count):
                selected_token_indices.add(token_index)
            else:
                skipped_out_of_window += 1
        if selected_token_indices:
            idx = torch.tensor(sorted(selected_token_indices), device=device, dtype=torch.long)
            score[:, idx] = 1.0
        stats.update({
            f"{prefix}_available": True,
            f"{prefix}_reason": "ok" if selected_token_indices else "empty_after_window_mapping",
            f"{prefix}_frame_base": int(frame_base),
            f"{prefix}_overlap_frames": int(overlap_frames),
            f"{prefix}_patch_start_idx": int(patch_start),
            f"{prefix}_patch_tokens_per_frame": int(patch_tokens_per_frame),
            f"{prefix}_tokens_selected": int(len(selected_token_indices)),
            f"{prefix}_rows_skipped_out_of_window": int(skipped_out_of_window),
            f"{prefix}_rows_skipped_bad_patch": int(skipped_bad_patch),
        })
        return score, stats

    def _make_swa_overlap_attention_bias(
        self,
        hmc_control: Optional[Dict[str, Any]],
        *,
        batch_size: int,
        frame_num: int,
        tokens_per_frame: int,
        history_tokens: int,
        current_tokens: int,
        device: torch.device,
        dtype: torch.dtype,
        swa_layer_idx: int = -1,
    ) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
        stats: Dict[str, Any] = {
            "swa_overlap_bias_applied": False,
            "swa_overlap_bias_query_tokens": 0,
            "swa_overlap_bias_source_tokens": 0,
        }
        if not hmc_control or hmc_control.get("identity_hooks", False):
            return None, stats
        beta = float(hmc_control.get("swa_overlap_bias_beta", 0.0))
        if beta == 0.0 or history_tokens <= 0 or current_tokens <= 0:
            return None, stats
        D_tok = hmc_control.get("D_tok")
        D_prev = hmc_control.get("D_prev_patch")
        if D_tok is None or D_prev is None:
            return None, stats
        if frame_num <= 0 or tokens_per_frame <= 0:
            return None, stats
        if current_tokens != frame_num * tokens_per_frame:
            return None, stats

        overlap_frames = max(int(hmc_control.get("swa_overlap_frames", 0)), 0)
        if overlap_frames <= 0:
            return None, stats

        D_cur = D_tok.to(device=device, dtype=torch.float32).reshape(batch_size, frame_num, tokens_per_frame)
        prev_flat = D_prev.to(device=device, dtype=torch.float32).reshape(-1)
        if prev_flat.numel() < tokens_per_frame:
            return None, stats
        prev_frames = int(prev_flat.numel() // tokens_per_frame)
        hist_frames = int(history_tokens // tokens_per_frame)
        if prev_frames <= 0 or hist_frames <= 0:
            return None, stats
        usable_frames = min(prev_frames, hist_frames)
        prev_flat = prev_flat[-usable_frames * tokens_per_frame:]
        D_src_frames = prev_flat.reshape(1, usable_frames, tokens_per_frame).expand(batch_size, -1, -1)

        ov = min(overlap_frames, frame_num, usable_frames)
        if ov <= 0:
            return None, stats
        qn = ov * tokens_per_frame
        sn = ov * tokens_per_frame
        source_end = history_tokens
        source_start = max(0, source_end - sn)
        sn = source_end - source_start
        if sn <= 0:
            return None, stats

        Dq = D_cur[:, :ov, :].reshape(batch_size, qn)
        Ds = D_src_frames[:, -ov:, :].reshape(batch_size, ov * tokens_per_frame)
        if Ds.shape[1] != sn:
            Ds = Ds[:, -sn:]
        if Dq.shape[1] != qn:
            Dq = Dq[:, :qn]

        mode = str(hmc_control.get("swa_overlap_bias_mode", "pair"))
        mode_l = mode.lower()
        Dq_c = Dq.clamp(0.0, 1.0)
        Ds_c = Ds.clamp(0.0, 1.0)
        role_bias_min_keep = min(max(float(hmc_control.get("swa_overlap_bias_min_keep", 1e-4)), 1e-6), 1.0)
        Dq_aligned = Dq_c[:, :sn] if Dq_c.shape[1] != sn else Dq_c
        Ds_aligned = Ds_c[:, :sn] if Ds_c.shape[1] != sn else Ds_c
        if mode_l == "semantic_role_stable_protect_minus_negative":
            stable_score, stable_stats = self._make_swa_overlap_source_role_score(
                hmc_control,
                mode="semantic_role_stable",
                batch_size=batch_size,
                frame_num=frame_num,
                tokens_per_frame=tokens_per_frame,
                source_tokens=sn,
                ov=ov,
                Dq=Dq_aligned,
                device=device,
            )
            negative_score, negative_stats = self._make_swa_overlap_source_role_score(
                hmc_control,
                mode="semantic_role_negative",
                batch_size=batch_size,
                frame_num=frame_num,
                tokens_per_frame=tokens_per_frame,
                source_tokens=sn,
                ov=ov,
                Dq=Dq_aligned,
                device=device,
            )
            if stable_score is None or negative_score is None:
                return None, stats
            alpha = max(abs(beta), 0.0)
            factor = (1.0 + alpha * stable_score - alpha * negative_score).clamp_min(role_bias_min_keep)
            keep = factor.unsqueeze(1).expand(batch_size, qn, sn)
            score_for_dump = (stable_score - negative_score).clamp(-1.0, 1.0)
            stats.update({
                "swa_overlap_bias_role_mode": mode_l,
                "swa_overlap_bias_role_base_mode": mode_l,
                "swa_overlap_bias_role_action": "stable_protect_boost_and_negative_damp",
                "swa_overlap_bias_mass_preserving_logit_reweight": True,
                "swa_overlap_bias_role_factor_mean": float(factor.detach().float().mean().item()),
                "swa_overlap_bias_role_factor_p10": float(torch.quantile(factor.detach().float(), 0.10).item()),
                "swa_overlap_bias_role_factor_p90": float(torch.quantile(factor.detach().float(), 0.90).item()),
                "swa_overlap_bias_stable_selected_tokens": int((stable_score > 0.0).sum().item()),
                "swa_overlap_bias_negative_selected_tokens": int((negative_score > 0.0).sum().item()),
                "swa_overlap_bias_stable_mode": stable_stats.get("swa_overlap_source_semantic_role_mode"),
                "swa_overlap_bias_negative_mode": negative_stats.get("swa_overlap_source_semantic_role_mode"),
            })
            stats.update(self._dump_swa_overlap_feature_map(
                hmc_control,
                kind="source_bias_role",
                mode=mode,
                swa_layer_idx=int(swa_layer_idx),
                batch_size=batch_size,
                frame_num=frame_num,
                tokens_per_frame=tokens_per_frame,
                history_tokens=history_tokens,
                source_start=source_start,
                source_end=source_end,
                overlap_frames=ov,
                Dq=Dq_aligned,
                Ds=Ds_aligned,
                score=score_for_dump,
                control=factor,
            ))
        elif mode_l in {
            "external_v84_anchor_mask",
            "external_v84_anchor_mask_random_same_mass",
            "v84_anchor_external_mask",
            "v84_anchor_external_mask_random_same_mass",
        }:
            random_same_mass = mode_l.endswith("_random_same_mass")
            mask_csv = str(hmc_control.get("swa_overlap_external_mask_csv", "") or "")
            variant = str(hmc_control.get("swa_overlap_external_mask_variant", "current_role_anchor") or "current_role_anchor")
            seq_filter = str(hmc_control.get("swa_overlap_external_mask_seq", "") or "")
            chunk_idx = int(hmc_control.get("semantic_action_chunk_idx", -1))
            score, mask_stats = self._make_external_v84_anchor_source_score(
                mask_csv=mask_csv,
                variant=variant,
                seq_filter=seq_filter,
                curr_chunk=chunk_idx,
                batch_size=batch_size,
                source_tokens=sn,
                overlap_frames=ov,
                tokens_per_frame=tokens_per_frame,
                device=device,
            )
            if score is None:
                stats.update(mask_stats)
                return None, stats
            selected_before_random = int((score > 0.0).sum().item())
            if random_same_mass and selected_before_random > 0:
                score = self._randomize_swa_overlap_score_same_distribution(
                    score,
                    hmc_control,
                    swa_layer_idx=int(swa_layer_idx),
                    salt_offset=8400.0,
                )
            factor = (1.0 + score).clamp_min(1.0)
            keep = factor.unsqueeze(1).expand(batch_size, qn, sn)
            selected_after_random = int((score > 0.0).sum().item())
            stats.update(mask_stats)
            stats.update({
                "swa_overlap_bias_external_mask_mode": mode_l,
                "swa_overlap_bias_external_mask_variant": variant,
                "swa_overlap_bias_external_mask_random_same_mass": bool(random_same_mass),
                "swa_overlap_bias_external_mask_selected_before_random": int(selected_before_random),
                "swa_overlap_bias_external_mask_selected_after_random": int(selected_after_random),
                "swa_overlap_bias_external_mask_selected_ratio": float(
                    selected_after_random / max(int(batch_size * sn), 1)
                ),
                "swa_overlap_bias_mass_preserving_logit_reweight": True,
                "swa_overlap_bias_external_mask_factor_mean": float(factor.detach().float().mean().item()),
                "swa_overlap_bias_external_mask_factor_p10": float(torch.quantile(factor.detach().float(), 0.10).item()),
                "swa_overlap_bias_external_mask_factor_p90": float(torch.quantile(factor.detach().float(), 0.90).item()),
            })
            stats.update(self._dump_swa_overlap_feature_map(
                hmc_control,
                kind="source_bias_external_v84_anchor",
                mode=mode,
                swa_layer_idx=int(swa_layer_idx),
                batch_size=batch_size,
                frame_num=frame_num,
                tokens_per_frame=tokens_per_frame,
                history_tokens=history_tokens,
                source_start=source_start,
                source_end=source_end,
                overlap_frames=ov,
                Dq=Dq_aligned,
                Ds=Ds_aligned,
                score=score,
                control=factor,
            ))
        elif mode_l in {
            "external_v92_policy_query_mask",
            "external_v92_policy_query_mask_random_same_mass",
            "external_v92_policy_pair_mask",
            "external_v92_policy_pair_mask_random_same_mass",
            "external_v92_policy_qk_pair_mask",
            "external_v92_policy_qk_pair_mask_random_same_mass",
        }:
            random_same_mass = mode_l.endswith("_random_same_mass")
            pair_mode = "pair_mask" in mode_l or "qk_pair_mask" in mode_l
            mask_csv = str(hmc_control.get("swa_overlap_external_mask_csv", "") or "")
            variant = str(hmc_control.get("swa_overlap_external_mask_variant", "v92_policy_risk_pair_mask") or "v92_policy_risk_pair_mask")
            seq_filter = str(hmc_control.get("swa_overlap_external_mask_seq", "") or "")
            chunk_idx = int(hmc_control.get("semantic_action_chunk_idx", -1))
            q_score, query_stats = self._make_external_v92_policy_side_score(
                mask_csv=mask_csv,
                variant=variant,
                seq_filter=seq_filter,
                curr_chunk=chunk_idx,
                side="query",
                batch_size=batch_size,
                token_count=qn,
                overlap_frames=ov,
                tokens_per_frame=tokens_per_frame,
                device=device,
            )
            if q_score is None:
                stats.update(query_stats)
                return None, stats
            s_score: Optional[torch.Tensor] = None
            source_stats: Dict[str, Any] = {}
            if pair_mode:
                s_score, source_stats = self._make_external_v92_policy_side_score(
                    mask_csv=mask_csv,
                    variant=variant,
                    seq_filter=seq_filter,
                    curr_chunk=chunk_idx,
                    side="source",
                    batch_size=batch_size,
                    token_count=sn,
                    overlap_frames=ov,
                    tokens_per_frame=tokens_per_frame,
                    device=device,
                )
                if s_score is None:
                    stats.update(query_stats)
                    stats.update(source_stats)
                    return None, stats
            q_selected_before_random = int((q_score > 0.0).sum().item())
            s_selected_before_random = int((s_score > 0.0).sum().item()) if s_score is not None else 0
            if random_same_mass and q_selected_before_random > 0:
                q_score = self._randomize_swa_overlap_score_same_distribution(
                    q_score,
                    hmc_control,
                    swa_layer_idx=int(swa_layer_idx),
                    salt_offset=9200.0,
                )
            if random_same_mass and s_score is not None and s_selected_before_random > 0:
                s_score = self._randomize_swa_overlap_score_same_distribution(
                    s_score,
                    hmc_control,
                    swa_layer_idx=int(swa_layer_idx),
                    salt_offset=9300.0,
                )
            q_selected_after_random = int((q_score > 0.0).sum().item())
            s_selected_after_random = int((s_score > 0.0).sum().item()) if s_score is not None else 0
            if pair_mode and s_score is not None:
                pair_score = q_score.unsqueeze(2) * s_score.unsqueeze(1)
                factor = (1.0 + pair_score).clamp_min(1.0)
                keep = factor
                factor_for_stats = factor.detach().float().reshape(-1)
                selected_cells = int((pair_score > 0.0).sum().item())
                route_kind = "pair"
            else:
                factor = (1.0 + q_score).clamp_min(1.0)
                keep = factor.unsqueeze(2).expand(batch_size, qn, sn)
                factor_for_stats = factor.detach().float().reshape(-1)
                selected_cells = int(q_selected_after_random * sn)
                route_kind = "query"
            stats.update(query_stats)
            stats.update(source_stats)
            stats.update({
                "swa_overlap_bias_external_v92_policy_mode": mode_l,
                "swa_overlap_bias_external_v92_policy_variant": variant,
                "swa_overlap_bias_external_v92_policy_route_kind": route_kind,
                "swa_overlap_bias_external_v92_policy_random_same_mass": bool(random_same_mass),
                "swa_overlap_bias_external_v92_policy_query_tokens_selected_before_random": int(q_selected_before_random),
                "swa_overlap_bias_external_v92_policy_query_tokens_selected_after_random": int(q_selected_after_random),
                "swa_overlap_bias_external_v92_policy_source_tokens_selected_before_random": int(s_selected_before_random),
                "swa_overlap_bias_external_v92_policy_source_tokens_selected_after_random": int(s_selected_after_random),
                "swa_overlap_bias_external_v92_policy_pair_cells_selected_after_random": int(selected_cells),
                "swa_overlap_bias_external_v92_policy_query_selected_ratio": float(
                    q_selected_after_random / max(int(batch_size * qn), 1)
                ),
                "swa_overlap_bias_external_v92_policy_source_selected_ratio": float(
                    s_selected_after_random / max(int(batch_size * sn), 1)
                ) if pair_mode else None,
                "swa_overlap_bias_external_v92_policy_pair_selected_ratio": float(
                    selected_cells / max(int(batch_size * qn * sn), 1)
                ),
                "swa_overlap_bias_mass_preserving_logit_reweight": True,
                "swa_overlap_bias_external_v92_policy_factor_mean": float(factor_for_stats.mean().item()),
                "swa_overlap_bias_external_v92_policy_factor_p10": float(torch.quantile(factor_for_stats, 0.10).item()),
                "swa_overlap_bias_external_v92_policy_factor_p90": float(torch.quantile(factor_for_stats, 0.90).item()),
            })
        elif mode_l in self._swa_overlap_source_role_modes():
            role_score, role_stats = self._make_swa_overlap_source_role_score(
                hmc_control,
                mode=mode,
                batch_size=batch_size,
                frame_num=frame_num,
                tokens_per_frame=tokens_per_frame,
                source_tokens=sn,
                ov=ov,
                Dq=Dq_aligned,
                device=device,
            )
            if role_score is None:
                return None, stats
            role_base = mode_l
            if role_base.startswith("random_same_mass_"):
                role_base = role_base[len("random_same_mass_"):]
            if role_base.endswith("_random_same_mass"):
                role_base = role_base[:-len("_random_same_mass")]
            alpha = max(abs(beta), 0.0)
            if role_base in {"semantic_role_negative", "semantic_role_source_skip", "semrole_negative"}:
                factor = (1.0 - alpha * role_score).clamp_min(role_bias_min_keep)
                role_action = "damp"
            else:
                factor = (1.0 + alpha * role_score).clamp_min(1e-6)
                role_action = "boost"
            keep = factor.unsqueeze(1).expand(batch_size, qn, sn)
            stats.update(role_stats)
            stats.update({
                "swa_overlap_bias_role_mode": mode_l,
                "swa_overlap_bias_role_base_mode": role_base,
                "swa_overlap_bias_role_action": role_action,
                "swa_overlap_bias_mass_preserving_logit_reweight": True,
                "swa_overlap_bias_role_factor_mean": float(factor.detach().float().mean().item()),
                "swa_overlap_bias_role_factor_p10": float(torch.quantile(factor.detach().float(), 0.10).item()),
                "swa_overlap_bias_role_factor_p90": float(torch.quantile(factor.detach().float(), 0.90).item()),
            })
            stats.update(self._dump_swa_overlap_feature_map(
                hmc_control,
                kind="source_bias_role",
                mode=mode,
                swa_layer_idx=int(swa_layer_idx),
                batch_size=batch_size,
                frame_num=frame_num,
                tokens_per_frame=tokens_per_frame,
                history_tokens=history_tokens,
                source_start=source_start,
                source_end=source_end,
                overlap_frames=ov,
                Dq=Dq_aligned,
                Ds=Ds_aligned,
                score=role_score,
                control=factor,
            ))
        elif mode_l in {
            "semantic_same_group_boost_stable_agreement",
            "semantic_same_group_boost_stable_agreement_random_same_mass",
            "semantic_same_group_boost_stable_agreement_shuffled_semantic",
        }:
            random_same_mass = mode_l.endswith("_random_same_mass")
            shuffled_semantic = mode_l.endswith("_shuffled_semantic")
            base_mode = mode_l
            if random_same_mass:
                base_mode = base_mode[:-len("_random_same_mass")]
            if shuffled_semantic:
                base_mode = base_mode[:-len("_shuffled_semantic")]

            def _prev_overlap_groups() -> Tuple[Optional[torch.Tensor], str]:
                raw = hmc_control.get("G_prev_patch") if hmc_control else None
                if raw is None:
                    return None, "missing_G_prev_patch"
                flat = raw.to(device=device, dtype=torch.long).reshape(-1)
                if int(flat.numel()) <= 0:
                    return None, "empty_G_prev_patch"
                if int(flat.numel()) % int(tokens_per_frame) == 0:
                    prev_frames_local = int(flat.numel() // int(tokens_per_frame))
                    labels_full = flat.reshape(prev_frames_local, int(tokens_per_frame))
                    layout = "full_token_G_prev_patch"
                else:
                    patch_tokens = int(tokens_per_frame) - int(getattr(self, "patch_start_idx", 0))
                    if patch_tokens <= 0 or int(flat.numel()) % int(patch_tokens) != 0:
                        return None, "shape_mismatch_G_prev_patch"
                    prev_frames_local = int(flat.numel() // int(patch_tokens))
                    labels_full = torch.full(
                        (prev_frames_local, int(tokens_per_frame)),
                        int(_CONTEXT_SEM_GROUP_UNCERTAIN),
                        device=device,
                        dtype=torch.long,
                    )
                    labels_full[:, int(getattr(self, "patch_start_idx", 0)):] = flat.reshape(
                        prev_frames_local, patch_tokens
                    )
                    layout = "patch_G_prev_patch_with_uncertain_special_tokens"
                hist_frames_local = int(history_tokens // tokens_per_frame)
                usable_local = min(prev_frames_local, hist_frames_local)
                if usable_local <= 0:
                    return None, "no_usable_prev_G_frames"
                label_ov = min(int(ov), usable_local)
                labels = (
                    labels_full[-usable_local:]
                    .reshape(1, usable_local, int(tokens_per_frame))
                    .expand(batch_size, -1, -1)[:, -label_ov:, :]
                    .reshape(batch_size, label_ov * int(tokens_per_frame))
                )
                if int(labels.shape[1]) < int(sn):
                    return None, "short_prev_overlap_G_labels"
                if int(labels.shape[1]) != int(sn):
                    labels = labels[:, -sn:]
                return labels, layout

            def _current_head_groups() -> Tuple[Optional[torch.Tensor], str]:
                raw = hmc_control.get("G_sem_tok") if hmc_control else None
                if raw is None:
                    return None, "missing_G_sem_tok"
                flat = raw.to(device=device, dtype=torch.long)
                if int(flat.numel()) != int(batch_size * frame_num * tokens_per_frame):
                    return None, "shape_mismatch_G_sem_tok"
                labels = flat.reshape(batch_size, frame_num, tokens_per_frame)[:, :ov, :].reshape(
                    batch_size, ov * tokens_per_frame
                )
                if int(labels.shape[1]) < int(sn):
                    return None, "short_current_head_G_labels"
                if int(labels.shape[1]) != int(sn):
                    labels = labels[:, :sn]
                return labels, "current_head_G_sem_tok"

            Gs, source_group_layout = _prev_overlap_groups()
            Gq, query_group_layout = _current_head_groups()
            if Gs is None or Gq is None:
                score = torch.zeros_like(Ds_aligned, dtype=torch.float32)
                semantic_mask = torch.zeros_like(Ds_aligned, dtype=torch.bool)
                missing_semantic_groups = True
            else:
                if shuffled_semantic:
                    shuffled = torch.empty_like(Gq)
                    base_idx = torch.arange(int(Gq.shape[1]), device=device, dtype=torch.float32)
                    chunk_idx = int((hmc_control or {}).get("semantic_action_chunk_idx", -1))
                    for b in range(int(batch_size)):
                        salt = 9000.0 + float(max(chunk_idx, 0)) * 101.0 + float(swa_layer_idx) * 17.0 + float(b) * 13.0
                        rand = torch.frac(torch.sin((base_idx + 1.0 + salt) * 12.9898) * 43758.5453)
                        perm = torch.argsort(rand, stable=True)
                        shuffled[b] = Gq[b, perm]
                    Gq = shuffled
                allowed_groups = (
                    (Gs == int(_CONTEXT_SEM_GROUP_STRUCTURE))
                    | (Gs == int(_CONTEXT_SEM_GROUP_STATIC))
                    | (Gs == int(_CONTEXT_SEM_GROUP_LOWSTUFF))
                )
                semantic_mask = (Gs == Gq) & allowed_groups
                score = (1.0 - torch.maximum(Dq_aligned, Ds_aligned)).clamp(0.0, 1.0)
                score = score * semantic_mask.float()
                missing_semantic_groups = False
            selected_before_random = int((score > 0.0).sum().item())
            if random_same_mass and selected_before_random > 0:
                score = self._randomize_swa_overlap_score_same_distribution(
                    score,
                    hmc_control,
                    swa_layer_idx=int(swa_layer_idx),
                    salt_offset=7000.0,
                )
            factor = (1.0 + score).clamp_min(1.0)
            keep = factor.unsqueeze(1).expand(batch_size, qn, sn)
            selected_after_random = int((score > 0.0).sum().item())
            stats.update({
                "swa_overlap_bias_geometric_mode": mode_l,
                "swa_overlap_bias_geometric_base_mode": base_mode,
                "swa_overlap_bias_geometric_action": "semantic_same_group_stable_agreement_boost",
                "swa_overlap_bias_geometric_random_same_mass": bool(random_same_mass),
                "swa_overlap_bias_semantic_group_shuffled_semantic": bool(shuffled_semantic),
                "swa_overlap_bias_mass_preserving_logit_reweight": True,
                "swa_overlap_bias_semantic_group_source": source_group_layout,
                "swa_overlap_bias_semantic_group_query": query_group_layout,
                "swa_overlap_bias_semantic_group_missing": bool(missing_semantic_groups),
                "swa_overlap_bias_semantic_group_selected_before_random": int(selected_before_random),
                "swa_overlap_bias_semantic_group_selected_after_random": int(selected_after_random),
                "swa_overlap_bias_semantic_group_selected_ratio": float(
                    selected_after_random / max(int(batch_size * sn), 1)
                ),
                "swa_overlap_bias_geometric_factor_mean": float(factor.detach().float().mean().item()),
                "swa_overlap_bias_geometric_factor_p10": float(torch.quantile(factor.detach().float(), 0.10).item()),
                "swa_overlap_bias_geometric_factor_p90": float(torch.quantile(factor.detach().float(), 0.90).item()),
            })
            stats.update(self._dump_swa_overlap_feature_map(
                hmc_control,
                kind="source_bias_geometric_semantic",
                mode=mode,
                swa_layer_idx=int(swa_layer_idx),
                batch_size=batch_size,
                frame_num=frame_num,
                tokens_per_frame=tokens_per_frame,
                history_tokens=history_tokens,
                source_start=source_start,
                source_end=source_end,
                overlap_frames=ov,
                Dq=Dq_aligned,
                Ds=Ds_aligned,
                score=score,
                control=factor,
            ))
        elif mode_l in {
            "boost_stable",
            "boost_stable_agreement",
            "boost_low_dyn_agreement",
            "boost_stable_random_same_mass",
            "boost_stable_agreement_random_same_mass",
            "boost_low_dyn_agreement_random_same_mass",
            "boost_stable_agreement_topq80",
            "boost_stable_agreement_topq80_random_same_mass",
            "boost_stable_agreement_topq80_aligned",
            "boost_stable_agreement_topq80_aligned_random_same_mass",
            "boost_stable_agreement_topq90",
            "boost_stable_agreement_topq90_random_same_mass",
        }:
            score_mode = mode_l[len("boost_"):]
            random_same_mass = score_mode.endswith("_random_same_mass")
            base_mode = score_mode[:-len("_random_same_mass")] if random_same_mass else score_mode
            aligned_route = False
            if base_mode.endswith("_aligned"):
                aligned_route = True
                base_mode = base_mode[:-len("_aligned")]
            top_quantile = None
            if base_mode.endswith("_topq80"):
                top_quantile = 0.80
                base_mode = base_mode[:-len("_topq80")]
            if base_mode.endswith("_topq90"):
                top_quantile = 0.90
                base_mode = base_mode[:-len("_topq90")]
            score = 1.0 - torch.maximum(Dq_aligned, Ds_aligned)
            score = score.clamp(0.0, 1.0)
            if top_quantile is not None:
                top_mask = torch.zeros_like(score, dtype=torch.bool)
                for b in range(int(batch_size)):
                    thr = torch.quantile(score[b].float(), float(top_quantile))
                    top_mask[b] = score[b] >= thr
                score = torch.where(top_mask, score, torch.zeros_like(score))
            if random_same_mass:
                score = self._randomize_swa_overlap_score_same_distribution(
                    score,
                    hmc_control,
                    swa_layer_idx=int(swa_layer_idx),
                    salt_offset=5000.0,
                )
            factor = (1.0 + score).clamp_min(1.0)
            if aligned_route:
                keep = torch.ones(batch_size, qn, sn, device=device, dtype=torch.float32)
                diag_n = min(int(qn), int(sn))
                if diag_n > 0:
                    diag_idx = torch.arange(diag_n, device=device)
                    keep[:, diag_idx, diag_idx] = factor[:, :diag_n]
            else:
                keep = factor.unsqueeze(1).expand(batch_size, qn, sn)
            stats.update({
                "swa_overlap_bias_geometric_mode": mode_l,
                "swa_overlap_bias_geometric_base_mode": base_mode,
                "swa_overlap_bias_geometric_action": "stable_agreement_boost",
                "swa_overlap_bias_geometric_random_same_mass": bool(random_same_mass),
                "swa_overlap_bias_geometric_aligned_route": bool(aligned_route),
                "swa_overlap_bias_geometric_top_quantile": (
                    float(top_quantile) if top_quantile is not None else None
                ),
                "swa_overlap_bias_geometric_selected_tokens": int((score > 0.0).sum().item()),
                "swa_overlap_bias_geometric_selected_ratio": float(
                    int((score > 0.0).sum().item()) / max(int(batch_size * sn), 1)
                ),
                "swa_overlap_bias_mass_preserving_logit_reweight": True,
                "swa_overlap_bias_geometric_factor_mean": float(factor.detach().float().mean().item()),
                "swa_overlap_bias_geometric_factor_p10": float(torch.quantile(factor.detach().float(), 0.10).item()),
                "swa_overlap_bias_geometric_factor_p90": float(torch.quantile(factor.detach().float(), 0.90).item()),
            })
            stats.update(self._dump_swa_overlap_feature_map(
                hmc_control,
                kind="source_bias_geometric",
                mode=mode,
                swa_layer_idx=int(swa_layer_idx),
                batch_size=batch_size,
                frame_num=frame_num,
                tokens_per_frame=tokens_per_frame,
                history_tokens=history_tokens,
                source_start=source_start,
                source_end=source_end,
                overlap_frames=ov,
                Dq=Dq_aligned,
                Ds=Ds_aligned,
                score=score,
                control=factor,
            ))
        elif mode_l == "source":
            keep = (1.0 - Ds_c).unsqueeze(1).expand(batch_size, qn, sn)
        elif mode_l == "union":
            keep = 1.0 - torch.maximum(Dq_c.unsqueeze(-1), Ds_c.unsqueeze(1))
        elif mode_l == "intersection":
            keep = 1.0 - torch.minimum(Dq_c.unsqueeze(-1), Ds_c.unsqueeze(1))
        else:
            keep = 1.0 - (1.0 - Dq_c).unsqueeze(-1) * Ds_c.unsqueeze(1)
        min_keep = min(max(float(hmc_control.get("swa_overlap_bias_min_keep", 1e-4)), 1e-6), 1.0)
        keep = keep.clamp_min(min_keep)

        # Do not materialize a full [current_tokens, history+current] bias matrix.
        # KITTI full chunks have ~40k current tokens; the dense mask would add
        # multi-GB allocations.  The attention layer understands this compact
        # descriptor and recomputes only the overlap query rows in small blocks.
        bias_values = beta * torch.log(keep)
        query_block = max(1, int(hmc_control.get("swa_overlap_bias_query_block", 128)))
        compact_bias = {
            "type": "overlap_bias",
            "query_tokens": int(qn),
            "source_start": int(source_start),
            "source_end": int(source_end),
            "bias_values": bias_values.to(dtype=dtype),
            "query_block_size": int(query_block),
        }
        head_indices: List[int] = []
        for raw_part in str(hmc_control.get("swa_overlap_bias_head_indices", "") or "").split(","):
            part = raw_part.strip()
            if not part:
                continue
            try:
                head_indices.append(int(part))
            except ValueError:
                continue
        if head_indices:
            compact_bias["head_indices"] = head_indices
        if bool(hmc_control.get("swa_overlap_bias_record_attention_mass", False)):
            compact_bias["attention_mass_stats"] = []
            compact_bias["attention_mass_max_queries"] = int(
                hmc_control.get("swa_overlap_bias_attention_mass_max_queries", 64) or 64
            )
            compact_bias["attention_mass_metric"] = "swa_overlap_bias_selected_mass"
        stats.update({
            "swa_overlap_bias_applied": True,
            "swa_overlap_bias_mode": mode,
            "swa_overlap_bias_beta": beta,
            "swa_overlap_bias_query_tokens": int(qn),
            "swa_overlap_bias_source_tokens": int(sn),
            "swa_overlap_bias_mean_keep": float(keep.mean().detach().cpu().item()),
            "swa_overlap_bias_min_keep_observed": float(keep.min().detach().cpu().item()),
            "swa_overlap_bias_query_block": int(query_block),
            "swa_overlap_bias_mean_abs": float(bias_values.abs().mean().detach().cpu().item()),
            "swa_overlap_bias_max_abs": float(bias_values.abs().max().detach().cpu().item()),
            "swa_overlap_bias_compact": True,
            "swa_overlap_bias_head_masked": bool(head_indices),
            "swa_overlap_bias_head_count_requested": int(len(head_indices)),
            "swa_overlap_bias_head_indices": ",".join(str(v) for v in head_indices),
        })
        return compact_bias, stats

    @staticmethod
    def _swa_overlap_source_layer_enabled(
        hmc_control: Optional[Dict[str, Any]],
        *,
        layer_idx: int,
        n_layers: int,
    ) -> bool:
        if not hmc_control or hmc_control.get("identity_hooks", False):
            return False
        if not hmc_control.get("enable_swa_overlap_source_gate", False):
            return False
        gate_mode = str(hmc_control.get("swa_overlap_source_gate_mode", "source")).lower()
        if Pi3._swa_overlap_source_mode_requires_semantic_action(gate_mode) and not bool(
            hmc_control.get("semantic_action_chunk_gate_active", True)
        ):
            return False
        mode = str(hmc_control.get("swa_overlap_source_gate_layer_mode", "last"))
        if mode == "all":
            return True
        if mode == "first":
            return int(layer_idx) == 0
        if mode == "last":
            return int(layer_idx) == max(0, int(n_layers) - 1)
        if mode == "single":
            return int(layer_idx) == int(hmc_control.get("swa_overlap_source_gate_single_layer", -1))
        return False

    @staticmethod
    def _swa_overlap_source_replace_layer_enabled(
        hmc_control: Optional[Dict[str, Any]],
        *,
        layer_idx: int,
        n_layers: int,
    ) -> bool:
        if not hmc_control or hmc_control.get("identity_hooks", False):
            return False
        if not hmc_control.get("enable_swa_overlap_source_replace", False):
            return False
        replace_mode = str(hmc_control.get("swa_overlap_source_replace_mode", "union")).lower()
        if Pi3._swa_overlap_source_mode_requires_semantic_action(replace_mode) and not bool(
            hmc_control.get("semantic_action_chunk_gate_active", True)
        ):
            return False
        mode = str(hmc_control.get("swa_overlap_source_replace_layer_mode", "last"))
        if mode == "all":
            return True
        if mode == "first":
            return int(layer_idx) == 0
        if mode == "last":
            return int(layer_idx) == max(0, int(n_layers) - 1)
        if mode == "single":
            return int(layer_idx) == int(hmc_control.get("swa_overlap_source_replace_single_layer", -1))
        return False

    @staticmethod
    def _v102_state_machine_trace_layer_enabled(
        hmc_control: Optional[Dict[str, Any]],
        *,
        layer_idx: int,
        n_layers: int,
    ) -> bool:
        if not hmc_control or hmc_control.get("identity_hooks", False):
            return False
        if not hmc_control.get("enable_v102_state_machine_trace", False):
            return False
        mode = str(hmc_control.get("v102_state_machine_layer_mode", "last"))
        if mode == "all":
            return True
        if mode == "first":
            return int(layer_idx) == 0
        if mode == "last":
            return int(layer_idx) == max(0, int(n_layers) - 1)
        if mode == "single":
            return int(layer_idx) == int(hmc_control.get("v102_state_machine_single_layer", -1))
        return False

    @staticmethod
    def _make_v102_state_machine_trace_record(
        hmc_control: Optional[Dict[str, Any]],
        *,
        history_tokens: int,
        current_tokens: int,
        swa_layer_idx: int,
    ) -> Dict[str, Any]:
        stats: Dict[str, Any] = {
            "v102_swa_state_machine_trace_available": False,
            "v102_swa_state_machine_trace_applied": False,
            "v102_swa_state_machine_action": "",
            "v102_swa_state_machine_reason": "disabled",
        }
        if not hmc_control or hmc_control.get("identity_hooks", False):
            stats["v102_swa_state_machine_reason"] = "missing_hmc_or_identity_hooks"
            return stats
        if not hmc_control.get("enable_v102_state_machine_trace", False):
            return stats
        action = str(hmc_control.get("v102_state_machine_action", "") or "").strip().upper()
        if action not in _V102_STATE_MACHINE_ACTIONS:
            stats.update({
                "v102_swa_state_machine_trace_available": True,
                "v102_swa_state_machine_reason": f"unsupported_action:{action or 'empty'}",
            })
            return stats
        strict_gate = bool(hmc_control.get("v102_state_machine_strict_gate_pass", False))
        true_l3_gate = bool(hmc_control.get("v102_state_machine_true_l3_gate_pass", False))
        stats.update({
            "v102_swa_state_machine_trace_available": True,
            "v102_swa_state_machine_trace_applied": False,
            "v102_swa_state_machine_action": action,
            "v102_swa_state_machine_scaffold_only": True,
            "v102_swa_state_machine_reason": "diagnostic_scaffold_only_no_kv_or_attention_change",
            "v102_swa_state_machine_swa_layer": int(swa_layer_idx),
            "v102_swa_state_machine_history_tokens": int(history_tokens),
            "v102_swa_state_machine_current_tokens": int(current_tokens),
            "v102_swa_state_machine_strict_gate_pass": strict_gate,
            "v102_swa_state_machine_true_l3_gate_pass": true_l3_gate,
            "v102_swa_state_machine_runtime_action_allowed": False,
            "v102_swa_state_machine_required_terms": (
                "anchor_identity,current_support,O_scale,R_same,query_head_controls,true_L3_L4_evaluator"
            ),
        })
        return stats

    @staticmethod
    def _make_v102_state_machine_action_probe(
        hmc_control: Optional[Dict[str, Any]],
        *,
        batch_size: int,
        tokens_per_frame: int,
        history_tokens: int,
        current_tokens: int,
        device: torch.device,
        swa_layer_idx: int,
    ) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        stats = Pi3._make_v102_state_machine_trace_record(
            hmc_control,
            history_tokens=history_tokens,
            current_tokens=current_tokens,
            swa_layer_idx=swa_layer_idx,
        )
        stats["v102_swa_state_machine_action_probe_enabled"] = bool(
            hmc_control and hmc_control.get("enable_v102_state_machine_action_probe", False)
        )
        stats["v102_swa_state_machine_probe_impl"] = str(
            (hmc_control or {}).get("v102_state_machine_probe_impl", "compact_kv_reject_unreliable")
        )
        if not hmc_control or hmc_control.get("identity_hooks", False):
            return None, stats
        if not hmc_control.get("enable_v102_state_machine_trace", False):
            return None, stats
        if not hmc_control.get("enable_v102_state_machine_action_probe", False):
            return None, stats

        action = str(hmc_control.get("v102_state_machine_action", "") or "").strip().upper()
        impl = str(hmc_control.get("v102_state_machine_probe_impl", "compact_kv_reject_unreliable") or "")
        soft_supported_impl = impl == "source_soft_transmit_supported"
        supported_impl = impl == "compact_kv_transmit_supported" or soft_supported_impl
        reject_impl = impl == "compact_kv_reject_unreliable"
        soft_hold_impl = impl == "source_soft_hold_prev_reference"
        hold_impl = impl == "compact_kv_hold_prev_reference" or soft_hold_impl
        soft_delay_impl = impl == "source_soft_delay_update"
        delay_impl = impl == "compact_kv_delay_update" or soft_delay_impl
        context_only_impl = impl == "source_soft_context_only_demotion"
        if action == "TRANSMIT_SUPPORTED_ANCHORS" and not supported_impl:
            stats.update({
                "v102_swa_state_machine_scaffold_only": True,
                "v102_swa_state_machine_reason": f"unsupported_probe_impl_for_transmit:{impl or 'empty'}",
            })
            return None, stats
        if action == "REJECT_UNRELIABLE_ANCHORS" and not reject_impl:
            stats.update({
                "v102_swa_state_machine_scaffold_only": True,
                "v102_swa_state_machine_reason": f"unsupported_probe_impl_for_reject:{impl or 'empty'}",
            })
            return None, stats
        if action == "HOLD_PREV_REFERENCE" and not hold_impl:
            stats.update({
                "v102_swa_state_machine_scaffold_only": True,
                "v102_swa_state_machine_reason": f"unsupported_probe_impl_for_hold:{impl or 'empty'}",
            })
            return None, stats
        if action == "DELAY_UPDATE" and not delay_impl:
            stats.update({
                "v102_swa_state_machine_scaffold_only": True,
                "v102_swa_state_machine_reason": f"unsupported_probe_impl_for_delay:{impl or 'empty'}",
            })
            return None, stats
        if action == "CONTEXT_ONLY_DEMOTION" and not context_only_impl:
            stats.update({
                "v102_swa_state_machine_scaffold_only": True,
                "v102_swa_state_machine_reason": f"unsupported_probe_impl_for_context_only:{impl or 'empty'}",
            })
            return None, stats
        if action not in {
            "TRANSMIT_SUPPORTED_ANCHORS",
            "REJECT_UNRELIABLE_ANCHORS",
            "HOLD_PREV_REFERENCE",
            "DELAY_UPDATE",
            "CONTEXT_ONLY_DEMOTION",
        }:
            stats.update({
                "v102_swa_state_machine_scaffold_only": True,
                "v102_swa_state_machine_reason": f"diagnostic_action_probe_not_implemented_for:{action or 'empty'}",
            })
            return None, stats

        d_min = float(hmc_control.get("v102_state_machine_unreliable_d_min", 0.50) or 0.50)
        g_min = float(hmc_control.get("v102_state_machine_unreliable_g_min", 0.50) or 0.50)
        supported_d_max = float(hmc_control.get("v102_state_machine_supported_d_max", 0.25) or 0.25)
        supported_k_min = float(hmc_control.get("v102_state_machine_supported_k_min", 0.0) or 0.0)
        require_static_semantic = bool(hmc_control.get("v102_state_machine_supported_require_static_semantic", True))
        soft_unsupported_min_keep = float(hmc_control.get("v102_state_machine_soft_unsupported_min_keep", 0.50) or 0.50)
        soft_unsupported_min_keep = min(1.0, max(1.0e-4, soft_unsupported_min_keep))
        hold_prev_frames = int(hmc_control.get("v102_state_machine_hold_prev_frames", 1) or 1)
        hold_prev_frames = max(1, hold_prev_frames)
        hold_soft_min_keep = float(hmc_control.get("v102_state_machine_hold_soft_min_keep", 0.50) or 0.50)
        hold_soft_min_keep = min(1.0, max(1.0e-4, hold_soft_min_keep))
        delay_current_soft_min_keep = float(hmc_control.get("v102_state_machine_delay_current_soft_min_keep", 0.50) or 0.50)
        delay_current_soft_min_keep = min(1.0, max(1.0e-4, delay_current_soft_min_keep))
        context_soft_min_keep = float(hmc_control.get("v102_state_machine_context_soft_min_keep", 0.50) or 0.50)
        context_soft_min_keep = min(1.0, max(1.0e-4, context_soft_min_keep))
        min_keep_frac = float(hmc_control.get("v102_state_machine_min_history_keep_frac", 0.05) or 0.05)
        min_keep_frac = min(1.0, max(0.0, min_keep_frac))
        d_prev = Pi3._swa_trace_source_values(
            hmc_control,
            "D_prev_patch",
            batch_size=batch_size,
            history_tokens=history_tokens,
            tokens_per_frame=tokens_per_frame,
            device=device,
            dtype=torch.float32,
        )
        g_prev = Pi3._swa_trace_source_values(
            hmc_control,
            "G_prev_patch",
            batch_size=batch_size,
            history_tokens=history_tokens,
            tokens_per_frame=tokens_per_frame,
            device=device,
            dtype=torch.float32,
        )
        g_prev_labels = Pi3._swa_trace_source_values(
            hmc_control,
            "G_prev_patch",
            batch_size=batch_size,
            history_tokens=history_tokens,
            tokens_per_frame=tokens_per_frame,
            device=device,
            dtype=torch.long,
        )
        l_prev_labels = Pi3._swa_trace_source_values(
            hmc_control,
            "L_prev_patch",
            batch_size=batch_size,
            history_tokens=history_tokens,
            tokens_per_frame=tokens_per_frame,
            device=device,
            dtype=torch.long,
        )
        k_stable = Pi3._swa_trace_source_values(
            hmc_control,
            "K_stable_tok",
            batch_size=batch_size,
            history_tokens=history_tokens,
            tokens_per_frame=tokens_per_frame,
            device=device,
            dtype=torch.float32,
        )

        semantic_static = None
        semantic_any = None
        if g_prev_labels is not None:
            semantic_static = (
                (g_prev_labels == int(_CONTEXT_SEM_GROUP_STRUCTURE))
                | (g_prev_labels == int(_CONTEXT_SEM_GROUP_STATIC))
            )
            semantic_any = (
                semantic_static
                | (g_prev_labels == int(_CONTEXT_SEM_GROUP_MOVABLE))
                | (g_prev_labels == int(_CONTEXT_SEM_GROUP_LOWSTUFF))
                | (g_prev_labels == int(_CONTEXT_SEM_GROUP_UNCERTAIN))
            )
        if l_prev_labels is not None:
            fine_static = torch.zeros_like(l_prev_labels, dtype=torch.bool)
            for label_id in set(_CONTEXT_GROUND_FINE_LABEL_IDS) | set(_CONTEXT_VERTICAL_STATIC_FINE_LABEL_IDS) | {15}:
                fine_static |= l_prev_labels == int(label_id)
            fine_context = torch.zeros_like(l_prev_labels, dtype=torch.bool)
            for label_id in set(_CONTEXT_SKY_FINE_LABEL_IDS) | set(_CONTEXT_VEGETATION_FINE_LABEL_IDS):
                fine_context |= l_prev_labels == int(label_id)
            semantic_static = fine_static if semantic_static is None else (semantic_static | fine_static)
            semantic_any = (
                (fine_static | fine_context)
                if semantic_any is None else (semantic_any | fine_static | fine_context)
            )

        if context_only_impl:
            if semantic_any is None:
                stats.update({
                    "v102_swa_state_machine_scaffold_only": True,
                    "v102_swa_state_machine_reason": "diagnostic_action_probe_context_only_missing_semantic_sources",
                    "v102_swa_state_machine_context_semantic_tokens": 0,
                    "v102_swa_state_machine_context_scale_observable_tokens": 0,
                    "v102_swa_state_machine_context_demoted_tokens": 0,
                    "v102_swa_state_machine_rejected_history_tokens": 0,
                    "v102_swa_state_machine_kept_history_tokens": int(history_tokens) * int(batch_size),
                })
                return None, stats
            if semantic_static is None:
                semantic_static = torch.zeros_like(semantic_any, dtype=torch.bool)
            scale_observable = semantic_static.to(device=device, dtype=torch.bool)
            d_low_count = 0
            if d_prev is not None:
                d_low = d_prev <= supported_d_max
                d_low_count = int(d_low.sum().item())
                scale_observable = scale_observable & d_low.to(device=device, dtype=torch.bool)
            if supported_k_min > 0.0 and k_stable is not None:
                scale_observable = scale_observable & (k_stable >= supported_k_min).to(device=device, dtype=torch.bool)

            semantic_any = semantic_any.to(device=device, dtype=torch.bool)
            scale_observable = scale_observable.to(device=device, dtype=torch.bool)
            context_only_mask = semantic_any & ~scale_observable
            context_semantic_tokens = int(semantic_any.sum().item())
            scale_observable_tokens = int(scale_observable.sum().item())
            context_demoted_tokens = int(context_only_mask.sum().item())
            if context_demoted_tokens <= 0:
                history_keep = torch.ones(int(batch_size), int(history_tokens), device=device, dtype=torch.bool)
                current_keep = torch.ones(int(batch_size), int(current_tokens), device=device, dtype=torch.bool)
                source_keep_mask = torch.cat([history_keep, current_keep], dim=1)
                stats.update({
                    "v102_swa_state_machine_trace_applied": False,
                    "v102_swa_state_machine_scaffold_only": True,
                    "v102_swa_state_machine_reason": "diagnostic_action_probe_context_only_no_demotable_semantic_tokens",
                    "v102_swa_state_machine_context_semantic_tokens": context_semantic_tokens,
                    "v102_swa_state_machine_context_scale_observable_tokens": scale_observable_tokens,
                    "v102_swa_state_machine_context_d_low_tokens": d_low_count,
                    "v102_swa_state_machine_context_demoted_tokens": 0,
                    "v102_swa_state_machine_context_soft_min_keep": context_soft_min_keep,
                    "v102_swa_state_machine_rejected_history_tokens": 0,
                    "v102_swa_state_machine_kept_history_tokens": int(history_keep.sum().item()),
                    "v102_swa_state_machine_source_keep_tokens": int(source_keep_mask.sum().item()),
                    "v102_swa_state_machine_source_total_tokens": int(source_keep_mask.numel()),
                    "v102_swa_state_machine_rejected_history_frac": 0.0,
                })
                return None, stats

            history_keep = ~context_only_mask
            current_keep = torch.ones(int(batch_size), int(current_tokens), device=device, dtype=torch.bool)
            source_keep_mask = torch.cat([history_keep, current_keep], dim=1)
            affected_mask = torch.cat([context_only_mask, torch.zeros_like(current_keep)], dim=1)
            source_bias_values = torch.zeros_like(source_keep_mask, dtype=torch.float32)
            source_bias_values = source_bias_values.masked_fill(
                affected_mask,
                math.log(float(context_soft_min_keep)),
            )
            attention_mass_stats: List[Dict[str, Any]] = []
            attn_mask = {
                "type": "source_soft",
                "affected_mask": affected_mask.detach(),
                "source_bias_values": source_bias_values.detach(),
                "stable_anchor_mask": source_keep_mask.detach(),
                "attention_mass_stats": attention_mass_stats,
                "attention_mass_max_queries": int(
                    hmc_control.get("v102_state_machine_attention_mass_max_queries", 64) or 64
                ),
                "attention_mass_metric": "v102_source_soft_context_only_demotion",
            }
            kept_history_tokens = int(history_keep.sum().item())
            stats.update({
                "v102_swa_state_machine_trace_applied": True,
                "v102_swa_state_machine_scaffold_only": False,
                "v102_swa_state_machine_reason": "diagnostic_action_probe_source_soft_context_only_demotion",
                "v102_swa_state_machine_runtime_action_allowed": False,
                "v102_swa_state_machine_context_semantic_tokens": context_semantic_tokens,
                "v102_swa_state_machine_context_scale_observable_tokens": scale_observable_tokens,
                "v102_swa_state_machine_context_d_low_tokens": d_low_count,
                "v102_swa_state_machine_context_demoted_tokens": context_demoted_tokens,
                "v102_swa_state_machine_context_soft_min_keep": context_soft_min_keep,
                "v102_swa_state_machine_rejected_history_tokens": context_demoted_tokens,
                "v102_swa_state_machine_kept_history_tokens": kept_history_tokens,
                "v102_swa_state_machine_source_keep_tokens": int(source_keep_mask.sum().item()),
                "v102_swa_state_machine_source_total_tokens": int(source_keep_mask.numel()),
                "v102_swa_state_machine_rejected_history_frac": float(
                    context_demoted_tokens / max(1, int(batch_size) * int(history_tokens))
                ),
            })
            return attn_mask, stats

        if delay_impl:
            history_keep = torch.ones(int(batch_size), int(history_tokens), device=device, dtype=torch.bool)
            current_keep = torch.zeros(int(batch_size), int(current_tokens), device=device, dtype=torch.bool)
            source_keep_mask = torch.cat([history_keep, current_keep], dim=1)
            delayed_current_tokens = int((~current_keep).sum().item())
            kept_history_tokens = int(history_keep.sum().item())
            if delayed_current_tokens <= 0:
                stats.update({
                    "v102_swa_state_machine_trace_applied": False,
                    "v102_swa_state_machine_scaffold_only": True,
                    "v102_swa_state_machine_reason": "diagnostic_action_probe_delay_no_current_tokens",
                    "v102_swa_state_machine_delay_current_tokens": 0,
                    "v102_swa_state_machine_delay_current_frac": 0.0,
                    "v102_swa_state_machine_rejected_history_tokens": 0,
                    "v102_swa_state_machine_kept_history_tokens": kept_history_tokens,
                    "v102_swa_state_machine_source_keep_tokens": int(source_keep_mask.sum().item()),
                    "v102_swa_state_machine_source_total_tokens": int(source_keep_mask.numel()),
                })
                return None, stats

            attention_mass_stats: List[Dict[str, Any]] = []
            if soft_delay_impl:
                affected_mask = ~source_keep_mask
                source_bias_values = torch.zeros_like(source_keep_mask, dtype=torch.float32)
                source_bias_values = source_bias_values.masked_fill(
                    affected_mask,
                    math.log(float(delay_current_soft_min_keep)),
                )
                attn_mask = {
                    "type": "source_soft",
                    "affected_mask": affected_mask.detach(),
                    "source_bias_values": source_bias_values.detach(),
                    "stable_anchor_mask": source_keep_mask.detach(),
                    "attention_mass_stats": attention_mass_stats,
                    "attention_mass_max_queries": int(
                        hmc_control.get("v102_state_machine_attention_mass_max_queries", 64) or 64
                    ),
                    "attention_mass_metric": "v102_source_soft_delay_update",
                }
                reason = "diagnostic_action_probe_source_soft_delay_update"
            else:
                attn_mask = {
                    "type": "compact_kv",
                    "source_keep_mask": source_keep_mask.detach(),
                    "attention_mass_stats": attention_mass_stats,
                    "attention_mass_max_queries": int(
                        hmc_control.get("v102_state_machine_attention_mass_max_queries", 64) or 64
                    ),
                }
                reason = "diagnostic_action_probe_compact_kv_delay_update"
            stats.update({
                "v102_swa_state_machine_trace_applied": True,
                "v102_swa_state_machine_scaffold_only": False,
                "v102_swa_state_machine_reason": reason,
                "v102_swa_state_machine_runtime_action_allowed": False,
                "v102_swa_state_machine_delay_current_tokens": delayed_current_tokens,
                "v102_swa_state_machine_delay_current_frac": float(
                    delayed_current_tokens / max(1, int(batch_size) * int(current_tokens))
                ),
                "v102_swa_state_machine_delay_current_soft_min_keep": delay_current_soft_min_keep if soft_delay_impl else 0.0,
                "v102_swa_state_machine_rejected_history_tokens": 0,
                "v102_swa_state_machine_kept_history_tokens": kept_history_tokens,
                "v102_swa_state_machine_source_keep_tokens": int(source_keep_mask.sum().item()),
                "v102_swa_state_machine_source_total_tokens": int(source_keep_mask.numel()),
                "v102_swa_state_machine_rejected_history_frac": 0.0,
            })
            return attn_mask, stats

        if hold_impl:
            if int(tokens_per_frame) <= 0 or int(history_tokens) < int(tokens_per_frame):
                stats.update({
                    "v102_swa_state_machine_scaffold_only": True,
                    "v102_swa_state_machine_reason": "diagnostic_action_probe_hold_missing_frame_geometry",
                    "v102_swa_state_machine_hold_prev_frames": hold_prev_frames,
                    "v102_swa_state_machine_hold_history_frames": 0,
                    "v102_swa_state_machine_hold_reference_tokens": 0,
                    "v102_swa_state_machine_rejected_history_tokens": 0,
                    "v102_swa_state_machine_kept_history_tokens": int(history_tokens) * int(batch_size),
                })
                return None, stats
            history_frames = int(history_tokens) // int(tokens_per_frame)
            hold_frames = min(int(hold_prev_frames), max(1, history_frames))
            reference_tokens_per_sample = int(hold_frames) * int(tokens_per_frame)
            history_keep = torch.zeros(int(batch_size), int(history_tokens), device=device, dtype=torch.bool)
            history_keep[:, int(history_tokens) - reference_tokens_per_sample :] = True

            reference_slice = slice(int(history_tokens) - reference_tokens_per_sample, int(history_tokens))
            hold_d_low_count = 0
            hold_semantic_static_count = 0
            hold_k_stable_count = 0
            if d_prev is not None:
                hold_d_low_count = int((d_prev[:, reference_slice] <= supported_d_max).sum().item())
            semantic_static = None
            if g_prev_labels is not None:
                semantic_static = (
                    (g_prev_labels == int(_CONTEXT_SEM_GROUP_STRUCTURE))
                    | (g_prev_labels == int(_CONTEXT_SEM_GROUP_STATIC))
                )
            if l_prev_labels is not None:
                fine_static = torch.zeros_like(l_prev_labels, dtype=torch.bool)
                for label_id in set(_CONTEXT_GROUND_FINE_LABEL_IDS) | set(_CONTEXT_VERTICAL_STATIC_FINE_LABEL_IDS) | {15}:
                    fine_static |= l_prev_labels == int(label_id)
                semantic_static = fine_static if semantic_static is None else (semantic_static | fine_static)
            if semantic_static is not None:
                hold_semantic_static_count = int(semantic_static[:, reference_slice].sum().item())
            if k_stable is not None:
                hold_k_stable_count = int((k_stable[:, reference_slice] >= supported_k_min).sum().item())

            current_keep = torch.ones(int(batch_size), int(current_tokens), device=device, dtype=torch.bool)
            source_keep_mask = torch.cat([history_keep, current_keep], dim=1)
            rejected_history_tokens = int((~history_keep).sum().item())
            kept_history_tokens = int(history_keep.sum().item())
            if rejected_history_tokens <= 0:
                stats.update({
                    "v102_swa_state_machine_trace_applied": False,
                    "v102_swa_state_machine_scaffold_only": True,
                    "v102_swa_state_machine_reason": "diagnostic_action_probe_hold_all_history_is_reference",
                    "v102_swa_state_machine_hold_prev_frames": hold_prev_frames,
                    "v102_swa_state_machine_hold_history_frames": history_frames,
                    "v102_swa_state_machine_hold_reference_tokens": kept_history_tokens,
                    "v102_swa_state_machine_hold_d_low_tokens": hold_d_low_count,
                    "v102_swa_state_machine_hold_semantic_static_tokens": hold_semantic_static_count,
                    "v102_swa_state_machine_hold_k_stable_tokens": hold_k_stable_count,
                    "v102_swa_state_machine_rejected_history_tokens": 0,
                    "v102_swa_state_machine_kept_history_tokens": kept_history_tokens,
                    "v102_swa_state_machine_source_keep_tokens": int(source_keep_mask.sum().item()),
                    "v102_swa_state_machine_source_total_tokens": int(source_keep_mask.numel()),
                    "v102_swa_state_machine_rejected_history_frac": 0.0,
                })
                return None, stats

            attention_mass_stats: List[Dict[str, Any]] = []
            if soft_hold_impl:
                affected_mask = ~source_keep_mask
                source_bias_values = torch.zeros_like(source_keep_mask, dtype=torch.float32)
                source_bias_values = source_bias_values.masked_fill(
                    affected_mask,
                    math.log(float(hold_soft_min_keep)),
                )
                attn_mask = {
                    "type": "source_soft",
                    "affected_mask": affected_mask.detach(),
                    "source_bias_values": source_bias_values.detach(),
                    "stable_anchor_mask": source_keep_mask.detach(),
                    "attention_mass_stats": attention_mass_stats,
                    "attention_mass_max_queries": int(
                        hmc_control.get("v102_state_machine_attention_mass_max_queries", 64) or 64
                    ),
                    "attention_mass_metric": "v102_source_soft_hold_prev_reference",
                }
                reason = "diagnostic_action_probe_source_soft_hold_prev_reference"
            else:
                attn_mask = {
                    "type": "compact_kv",
                    "source_keep_mask": source_keep_mask.detach(),
                    "attention_mass_stats": attention_mass_stats,
                    "attention_mass_max_queries": int(
                        hmc_control.get("v102_state_machine_attention_mass_max_queries", 64) or 64
                    ),
                }
                reason = "diagnostic_action_probe_compact_kv_hold_prev_reference"
            stats.update({
                "v102_swa_state_machine_trace_applied": True,
                "v102_swa_state_machine_scaffold_only": False,
                "v102_swa_state_machine_reason": reason,
                "v102_swa_state_machine_runtime_action_allowed": False,
                "v102_swa_state_machine_hold_prev_frames": hold_prev_frames,
                "v102_swa_state_machine_hold_history_frames": history_frames,
                "v102_swa_state_machine_hold_reference_tokens": kept_history_tokens,
                "v102_swa_state_machine_hold_d_low_tokens": hold_d_low_count,
                "v102_swa_state_machine_hold_semantic_static_tokens": hold_semantic_static_count,
                "v102_swa_state_machine_hold_k_stable_tokens": hold_k_stable_count,
                "v102_swa_state_machine_hold_soft_min_keep": hold_soft_min_keep if soft_hold_impl else 0.0,
                "v102_swa_state_machine_min_history_keep_frac": min_keep_frac,
                "v102_swa_state_machine_rejected_history_tokens": rejected_history_tokens,
                "v102_swa_state_machine_kept_history_tokens": kept_history_tokens,
                "v102_swa_state_machine_source_keep_tokens": int(source_keep_mask.sum().item()),
                "v102_swa_state_machine_source_total_tokens": int(source_keep_mask.numel()),
                "v102_swa_state_machine_rejected_history_frac": float(
                    rejected_history_tokens / max(1, int(batch_size) * int(history_tokens))
                ),
            })
            return attn_mask, stats

        if supported_impl:
            supported_mask = None
            d_low_count = 0
            semantic_static_count = 0
            k_stable_count = 0
            fallback_used = False
            if d_prev is not None:
                d_low = d_prev <= supported_d_max
                d_low_count = int(d_low.sum().item())
                supported_mask = d_low
            semantic_static = None
            if g_prev_labels is not None:
                semantic_static = (
                    (g_prev_labels == int(_CONTEXT_SEM_GROUP_STRUCTURE))
                    | (g_prev_labels == int(_CONTEXT_SEM_GROUP_STATIC))
                )
            if l_prev_labels is not None:
                fine_static = torch.zeros_like(l_prev_labels, dtype=torch.bool)
                for label_id in set(_CONTEXT_GROUND_FINE_LABEL_IDS) | set(_CONTEXT_VERTICAL_STATIC_FINE_LABEL_IDS) | {15}:
                    fine_static |= l_prev_labels == int(label_id)
                semantic_static = fine_static if semantic_static is None else (semantic_static | fine_static)
            if semantic_static is not None:
                semantic_static_count = int(semantic_static.sum().item())
                if require_static_semantic:
                    supported_mask = semantic_static if supported_mask is None else (supported_mask & semantic_static)
            if supported_k_min > 0.0 and k_stable is not None:
                k_mask = k_stable >= supported_k_min
                k_stable_count = int(k_mask.sum().item())
                supported_mask = k_mask if supported_mask is None else (supported_mask & k_mask)
            if supported_mask is None:
                stats.update({
                    "v102_swa_state_machine_scaffold_only": True,
                    "v102_swa_state_machine_reason": "diagnostic_action_probe_missing_supported_sources",
                    "v102_swa_state_machine_supported_d_low_tokens": 0,
                    "v102_swa_state_machine_supported_semantic_static_tokens": 0,
                    "v102_swa_state_machine_supported_k_stable_tokens": 0,
                    "v102_swa_state_machine_rejected_history_tokens": 0,
                    "v102_swa_state_machine_kept_history_tokens": int(history_tokens) * int(batch_size),
                })
                return None, stats
            if not bool(supported_mask.any()) and d_prev is not None:
                supported_mask = d_prev <= supported_d_max
                fallback_used = True

            history_keep = supported_mask.to(device=device, dtype=torch.bool)
            min_keep = int(math.ceil(float(history_tokens) * min_keep_frac))
            if min_keep > 0:
                for b in range(int(batch_size)):
                    if int(history_keep[b].sum().item()) >= min_keep:
                        continue
                    scores = torch.zeros(int(history_tokens), device=device, dtype=torch.float32)
                    if d_prev is not None:
                        scores = scores + d_prev[b].float()
                    if g_prev is not None:
                        scores = scores + g_prev[b].float()
                    if k_stable is not None:
                        scores = scores - k_stable[b].float()
                    keep_idx = torch.argsort(scores, descending=False)[:min_keep]
                    history_keep[b, keep_idx] = True

            current_keep = torch.ones(int(batch_size), int(current_tokens), device=device, dtype=torch.bool)
            source_keep_mask = torch.cat([history_keep, current_keep], dim=1)
            rejected_history_tokens = int((~history_keep).sum().item())
            kept_history_tokens = int(history_keep.sum().item())
            if rejected_history_tokens <= 0:
                stats.update({
                    "v102_swa_state_machine_trace_applied": False,
                    "v102_swa_state_machine_scaffold_only": True,
                    "v102_swa_state_machine_reason": "diagnostic_action_probe_all_history_supported",
                    "v102_swa_state_machine_supported_d_max": supported_d_max,
                    "v102_swa_state_machine_supported_k_min": supported_k_min,
                    "v102_swa_state_machine_supported_require_static_semantic": require_static_semantic,
                    "v102_swa_state_machine_supported_fallback_used": fallback_used,
                    "v102_swa_state_machine_supported_d_low_tokens": d_low_count,
                    "v102_swa_state_machine_supported_semantic_static_tokens": semantic_static_count,
                    "v102_swa_state_machine_supported_k_stable_tokens": k_stable_count,
                    "v102_swa_state_machine_supported_history_tokens": kept_history_tokens,
                    "v102_swa_state_machine_rejected_history_tokens": 0,
                    "v102_swa_state_machine_kept_history_tokens": kept_history_tokens,
                    "v102_swa_state_machine_source_keep_tokens": int(source_keep_mask.sum().item()),
                    "v102_swa_state_machine_source_total_tokens": int(source_keep_mask.numel()),
                    "v102_swa_state_machine_rejected_history_frac": 0.0,
                })
                return None, stats

            attention_mass_stats: List[Dict[str, Any]] = []
            if soft_supported_impl:
                affected_mask = ~source_keep_mask
                source_bias_values = torch.zeros_like(source_keep_mask, dtype=torch.float32)
                source_bias_values = source_bias_values.masked_fill(
                    affected_mask,
                    math.log(float(soft_unsupported_min_keep)),
                )
                attn_mask = {
                    "type": "source_soft",
                    "affected_mask": affected_mask.detach(),
                    "source_bias_values": source_bias_values.detach(),
                    "stable_anchor_mask": source_keep_mask.detach(),
                    "attention_mass_stats": attention_mass_stats,
                    "attention_mass_max_queries": int(
                        hmc_control.get("v102_state_machine_attention_mass_max_queries", 64) or 64
                    ),
                    "attention_mass_metric": "v102_source_soft_transmit_supported",
                }
                stats.update({
                    "v102_swa_state_machine_trace_applied": True,
                    "v102_swa_state_machine_scaffold_only": False,
                    "v102_swa_state_machine_reason": "diagnostic_action_probe_source_soft_transmit_supported",
                    "v102_swa_state_machine_runtime_action_allowed": False,
                    "v102_swa_state_machine_supported_d_max": supported_d_max,
                    "v102_swa_state_machine_supported_k_min": supported_k_min,
                    "v102_swa_state_machine_supported_require_static_semantic": require_static_semantic,
                    "v102_swa_state_machine_supported_fallback_used": fallback_used,
                    "v102_swa_state_machine_supported_d_low_tokens": d_low_count,
                    "v102_swa_state_machine_supported_semantic_static_tokens": semantic_static_count,
                    "v102_swa_state_machine_supported_k_stable_tokens": k_stable_count,
                    "v102_swa_state_machine_supported_history_tokens": kept_history_tokens,
                    "v102_swa_state_machine_soft_unsupported_min_keep": soft_unsupported_min_keep,
                    "v102_swa_state_machine_min_history_keep_frac": min_keep_frac,
                    "v102_swa_state_machine_rejected_history_tokens": rejected_history_tokens,
                    "v102_swa_state_machine_kept_history_tokens": kept_history_tokens,
                    "v102_swa_state_machine_source_keep_tokens": int(source_keep_mask.sum().item()),
                    "v102_swa_state_machine_source_total_tokens": int(source_keep_mask.numel()),
                    "v102_swa_state_machine_rejected_history_frac": float(
                        rejected_history_tokens / max(1, int(batch_size) * int(history_tokens))
                    ),
                })
                return attn_mask, stats
            attn_mask = {
                "type": "compact_kv",
                "source_keep_mask": source_keep_mask.detach(),
                "attention_mass_stats": attention_mass_stats,
                "attention_mass_max_queries": int(
                    hmc_control.get("v102_state_machine_attention_mass_max_queries", 64) or 64
                ),
            }
            stats.update({
                "v102_swa_state_machine_trace_applied": True,
                "v102_swa_state_machine_scaffold_only": False,
                "v102_swa_state_machine_reason": "diagnostic_action_probe_compact_kv_transmit_supported",
                "v102_swa_state_machine_runtime_action_allowed": False,
                "v102_swa_state_machine_supported_d_max": supported_d_max,
                "v102_swa_state_machine_supported_k_min": supported_k_min,
                "v102_swa_state_machine_supported_require_static_semantic": require_static_semantic,
                "v102_swa_state_machine_supported_fallback_used": fallback_used,
                "v102_swa_state_machine_supported_d_low_tokens": d_low_count,
                "v102_swa_state_machine_supported_semantic_static_tokens": semantic_static_count,
                "v102_swa_state_machine_supported_k_stable_tokens": k_stable_count,
                "v102_swa_state_machine_supported_history_tokens": kept_history_tokens,
                "v102_swa_state_machine_min_history_keep_frac": min_keep_frac,
                "v102_swa_state_machine_rejected_history_tokens": rejected_history_tokens,
                "v102_swa_state_machine_kept_history_tokens": kept_history_tokens,
                "v102_swa_state_machine_source_keep_tokens": int(source_keep_mask.sum().item()),
                "v102_swa_state_machine_source_total_tokens": int(source_keep_mask.numel()),
                "v102_swa_state_machine_rejected_history_frac": float(
                    rejected_history_tokens / max(1, int(batch_size) * int(history_tokens))
                ),
            })
            return attn_mask, stats

        unreliable_mask = None
        d_high_count = 0
        g_high_count = 0
        if d_prev is not None:
            d_high = d_prev >= d_min
            d_high_count = int(d_high.sum().item())
            unreliable_mask = d_high
        if g_prev is not None:
            g_high = g_prev >= g_min
            g_high_count = int(g_high.sum().item())
            unreliable_mask = g_high if unreliable_mask is None else (unreliable_mask | g_high)
        if unreliable_mask is None:
            stats.update({
                "v102_swa_state_machine_scaffold_only": True,
                "v102_swa_state_machine_reason": "diagnostic_action_probe_missing_D_prev_and_G_prev",
                "v102_swa_state_machine_unreliable_d_high_tokens": 0,
                "v102_swa_state_machine_unreliable_g_high_tokens": 0,
                "v102_swa_state_machine_rejected_history_tokens": 0,
                "v102_swa_state_machine_kept_history_tokens": int(history_tokens) * int(batch_size),
            })
            return None, stats

        history_keep = ~unreliable_mask.to(device=device, dtype=torch.bool)
        min_keep = int(math.ceil(float(history_tokens) * min_keep_frac))
        if min_keep > 0:
            for b in range(int(batch_size)):
                if int(history_keep[b].sum().item()) >= min_keep:
                    continue
                scores = torch.zeros(int(history_tokens), device=device, dtype=torch.float32)
                if d_prev is not None:
                    scores = scores + d_prev[b].float()
                if g_prev is not None:
                    scores = scores + g_prev[b].float()
                keep_idx = torch.argsort(scores, descending=False)[:min_keep]
                history_keep[b, keep_idx] = True

        current_keep = torch.ones(int(batch_size), int(current_tokens), device=device, dtype=torch.bool)
        source_keep_mask = torch.cat([history_keep, current_keep], dim=1)
        rejected_history_tokens = int((~history_keep).sum().item())
        kept_history_tokens = int(history_keep.sum().item())
        if rejected_history_tokens <= 0:
            stats.update({
                "v102_swa_state_machine_trace_applied": False,
                "v102_swa_state_machine_scaffold_only": True,
                "v102_swa_state_machine_reason": "diagnostic_action_probe_no_unreliable_source_tokens",
                "v102_swa_state_machine_unreliable_d_high_tokens": d_high_count,
                "v102_swa_state_machine_unreliable_g_high_tokens": g_high_count,
                "v102_swa_state_machine_rejected_history_tokens": 0,
                "v102_swa_state_machine_kept_history_tokens": kept_history_tokens,
                "v102_swa_state_machine_source_keep_tokens": int(source_keep_mask.sum().item()),
                "v102_swa_state_machine_source_total_tokens": int(source_keep_mask.numel()),
            })
            return None, stats

        attention_mass_stats: List[Dict[str, Any]] = []
        attn_mask = {
            "type": "compact_kv",
            "source_keep_mask": source_keep_mask.detach(),
            "attention_mass_stats": attention_mass_stats,
            "attention_mass_max_queries": int(
                hmc_control.get("v102_state_machine_attention_mass_max_queries", 64) or 64
            ),
        }
        stats.update({
            "v102_swa_state_machine_trace_applied": True,
            "v102_swa_state_machine_scaffold_only": False,
            "v102_swa_state_machine_reason": "diagnostic_action_probe_compact_kv_reject_unreliable",
            "v102_swa_state_machine_runtime_action_allowed": False,
            "v102_swa_state_machine_unreliable_d_min": d_min,
            "v102_swa_state_machine_unreliable_g_min": g_min,
            "v102_swa_state_machine_min_history_keep_frac": min_keep_frac,
            "v102_swa_state_machine_unreliable_d_high_tokens": d_high_count,
            "v102_swa_state_machine_unreliable_g_high_tokens": g_high_count,
            "v102_swa_state_machine_rejected_history_tokens": rejected_history_tokens,
            "v102_swa_state_machine_kept_history_tokens": kept_history_tokens,
            "v102_swa_state_machine_source_keep_tokens": int(source_keep_mask.sum().item()),
            "v102_swa_state_machine_source_total_tokens": int(source_keep_mask.numel()),
            "v102_swa_state_machine_rejected_history_frac": float(
                rejected_history_tokens / max(1, int(batch_size) * int(history_tokens))
            ),
        })
        return attn_mask, stats

    @staticmethod
    def _swa_overlap_source_semantic_modes() -> set:
        groups = (
            "structure",
            "static",
            "movable",
            "lowstuff",
            "uncertain",
            "sky",
            "vegetation",
            "ground",
            "vertical_static",
        )
        suffixes = ("", "_highd_q80", "_highd_q90", "_highd_q95")
        modes = set()
        for group in groups:
            for suffix in suffixes:
                base = f"semantic_{group}{suffix}"
                modes.add(base)
                modes.add(f"{base}_random_same_mass")
        return modes

    @staticmethod
    def _swa_overlap_source_role_modes() -> set:
        modes = {
            "semantic_role_negative",
            "semantic_role_source_skip",
            "semrole_negative",
            "semantic_role_positive",
            "semantic_role_stable",
            "semantic_role_anchor",
            "semantic_role_protect",
            "semantic_role_protected",
        }
        random_modes = {f"random_same_mass_{mode}" for mode in modes}
        random_modes.update({f"{mode}_random_same_mass" for mode in modes})
        return modes | random_modes

    @staticmethod
    def _swa_overlap_source_mode_requires_semantic_action(mode: str) -> bool:
        mode_l = str(mode or "").strip().lower()
        return (
            mode_l.startswith("semantic_")
            or mode_l.startswith("semrole_")
            or mode_l.startswith("random_same_mass_semantic_")
            or mode_l.startswith("random_same_mass_semrole_")
        )

    @staticmethod
    def _randomize_swa_overlap_score_same_distribution(
        score: torch.Tensor,
        hmc_control: Optional[Dict[str, Any]],
        *,
        swa_layer_idx: int,
        salt_offset: float = 0.0,
    ) -> torch.Tensor:
        if score.ndim != 2 or int(score.shape[-1]) <= 1:
            return score
        out = torch.empty_like(score)
        source_tokens = int(score.shape[-1])
        base_idx = torch.arange(source_tokens, device=score.device, dtype=torch.float32)
        chunk = float(hmc_control.get("semantic_action_chunk_idx", -1) if hmc_control else -1)
        for b in range(int(score.shape[0])):
            salt = chunk * 149.0 + float(swa_layer_idx) * 31.0 + float(b) * 17.0 + float(salt_offset)
            rand = torch.frac(torch.sin((base_idx + 1.0 + salt) * 12.9898) * 43758.5453)
            perm = torch.argsort(rand, stable=True)
            out[b] = score[b, perm]
        return out

    def _make_swa_overlap_source_semantic_score(
        self,
        hmc_control: Optional[Dict[str, Any]],
        *,
        mode: str,
        batch_size: int,
        frame_num: int,
        tokens_per_frame: int,
        history_tokens: int,
        source_tokens: int,
        ov: int,
        Ds: torch.Tensor,
        device: torch.device,
    ) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
        stats: Dict[str, Any] = {}
        mode_l = str(mode).lower()
        if mode_l not in self._swa_overlap_source_semantic_modes():
            return None, stats

        random_same_mass = mode_l.endswith("_random_same_mass")
        base_mode = mode_l[:-len("_random_same_mass")] if random_same_mass else mode_l
        highd_quantile = None
        for suffix, q in (("_highd_q80", 0.80), ("_highd_q90", 0.90), ("_highd_q95", 0.95)):
            if base_mode.endswith(suffix):
                highd_quantile = q
                base_mode = base_mode[:-len(suffix)]
                break
        group_name = base_mode[len("semantic_"):]

        coarse_groups = {
            "structure": _CONTEXT_SEM_GROUP_STRUCTURE,
            "static": _CONTEXT_SEM_GROUP_STATIC,
            "movable": _CONTEXT_SEM_GROUP_MOVABLE,
            "lowstuff": _CONTEXT_SEM_GROUP_LOWSTUFF,
            "uncertain": _CONTEXT_SEM_GROUP_UNCERTAIN,
        }
        fine_groups = {
            "sky": _CONTEXT_SKY_FINE_LABEL_IDS,
            "vegetation": _CONTEXT_VEGETATION_FINE_LABEL_IDS,
            "ground": _CONTEXT_GROUND_FINE_LABEL_IDS,
            "vertical_static": _CONTEXT_VERTICAL_STATIC_FINE_LABEL_IDS,
        }

        def _prev_overlap_labels(key: str) -> Optional[torch.Tensor]:
            raw = hmc_control.get(key) if hmc_control else None
            if raw is None:
                return None
            flat = raw.to(device=device, dtype=torch.long).reshape(-1)
            if int(flat.numel()) < int(tokens_per_frame):
                return None
            prev_frames = int(flat.numel() // tokens_per_frame)
            hist_frames = int(history_tokens // tokens_per_frame)
            usable_frames = min(prev_frames, hist_frames)
            if usable_frames <= 0:
                return None
            label_ov = min(int(ov), usable_frames)
            if label_ov <= 0:
                return None
            labels = (
                flat[-usable_frames * tokens_per_frame:]
                .reshape(1, usable_frames, tokens_per_frame)
                .expand(batch_size, -1, -1)[:, -label_ov:, :]
                .reshape(batch_size, label_ov * tokens_per_frame)
            )
            if int(labels.shape[1]) < int(source_tokens):
                return None
            if int(labels.shape[1]) != int(source_tokens):
                labels = labels[:, -source_tokens:]
            return labels

        def _current_head_labels(key: str) -> Optional[torch.Tensor]:
            raw = hmc_control.get(key) if hmc_control else None
            if raw is None:
                return None
            flat = raw.to(device=device, dtype=torch.long)
            if int(flat.numel()) != int(batch_size * frame_num * tokens_per_frame):
                return None
            labels = flat.reshape(batch_size, frame_num, tokens_per_frame)[:, :ov, :].reshape(
                batch_size, ov * tokens_per_frame
            )
            if int(labels.shape[1]) < int(source_tokens):
                return None
            if int(labels.shape[1]) != int(source_tokens):
                labels = labels[:, :source_tokens]
            return labels

        if group_name in coarse_groups:
            labels = _prev_overlap_labels("G_prev_patch")
            label_source = "prev_tail_G_patch"
            if labels is None:
                labels = _current_head_labels("G_sem_tok")
                label_source = "current_head_G_sem_tok_fallback" if labels is not None else "missing_G"
            if labels is None:
                score = torch.zeros(batch_size, source_tokens, device=device, dtype=torch.float32)
                missing = True
            else:
                score = (labels == int(coarse_groups[group_name])).float()
                missing = False
        elif group_name in fine_groups:
            labels = _prev_overlap_labels("L_prev_patch")
            label_source = "prev_tail_L_patch"
            if labels is None:
                labels = _current_head_labels("L_sem_tok")
                label_source = "current_head_L_sem_tok_fallback" if labels is not None else "missing_L"
            if labels is None:
                score = torch.zeros(batch_size, source_tokens, device=device, dtype=torch.float32)
                missing = True
            else:
                score = torch.zeros(batch_size, source_tokens, device=device, dtype=torch.float32)
                for label_id in sorted(int(x) for x in fine_groups[group_name]):
                    score = torch.maximum(score, (labels == int(label_id)).float())
                missing = False
        else:
            return None, stats

        if highd_quantile is not None and bool((score > 0.0).any().item()):
            highd = torch.zeros_like(score)
            for b in range(int(batch_size)):
                mask_b = score[b] > 0.0
                if int(mask_b.sum().item()) <= 0:
                    continue
                vals = Ds[b][mask_b].float()
                thr = torch.quantile(vals, float(highd_quantile))
                highd[b] = (mask_b & (Ds[b].float() >= thr)).float()
            score = highd

        semantic_count_before_random = int((score > 0.0).sum().item())
        if random_same_mass and semantic_count_before_random > 0:
            randomized = torch.zeros_like(score)
            chunk = float(hmc_control.get("semantic_action_chunk_idx", -1) if hmc_control else -1)
            base_idx = torch.arange(int(source_tokens), device=device, dtype=torch.float32)
            for b in range(int(batch_size)):
                k_select = min(int((score[b] > 0.0).sum().item()), int(source_tokens))
                if k_select <= 0:
                    continue
                salt = chunk * 149.0 + float(b) * 17.0
                rand = torch.frac(torch.sin((base_idx + 1.0 + salt) * 12.9898) * 43758.5453)
                top = torch.topk(rand, k_select).indices
                randomized[b, top] = 1.0
            score = randomized

        selected_tokens = int((score > 0.0).sum().item())
        selected_mask = score > 0.0
        if selected_tokens > 0:
            local_idx = torch.arange(int(source_tokens), device=device, dtype=torch.float32).reshape(1, source_tokens)
            selected_index_mean = float(local_idx.expand(batch_size, -1)[selected_mask].mean().item())
            selected_d_mean = float(Ds.float()[selected_mask].mean().item())
        else:
            selected_index_mean = 0.0
            selected_d_mean = 0.0
        stats.update({
            "swa_overlap_source_semantic_mode": mode_l,
            "swa_overlap_source_semantic_group": group_name,
            "swa_overlap_source_semantic_label_source": label_source,
            "swa_overlap_source_semantic_missing_labels": bool(missing),
            "swa_overlap_source_semantic_random_same_mass": bool(random_same_mass),
            "swa_overlap_source_semantic_highd_quantile": (
                float(highd_quantile) if highd_quantile is not None else None
            ),
            "swa_overlap_source_semantic_tokens_before_random": int(semantic_count_before_random),
            "swa_overlap_source_semantic_selected_tokens": int(selected_tokens),
            "swa_overlap_source_semantic_selected_ratio": float(selected_tokens / max(int(batch_size * source_tokens), 1)),
            "swa_overlap_source_semantic_selected_index_mean": selected_index_mean,
            "swa_overlap_source_semantic_selected_D_mean": selected_d_mean,
        })
        return score.clamp(0.0, 1.0), stats

    def _make_swa_overlap_source_role_score(
        self,
        hmc_control: Optional[Dict[str, Any]],
        *,
        mode: str,
        batch_size: int,
        frame_num: int,
        tokens_per_frame: int,
        source_tokens: int,
        ov: int,
        Dq: torch.Tensor,
        device: torch.device,
    ) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
        stats: Dict[str, Any] = {}
        mode_l = str(mode or "").strip().lower()
        if mode_l not in self._swa_overlap_source_role_modes():
            return None, stats

        random_same_mass = mode_l.startswith("random_same_mass_") or mode_l.endswith("_random_same_mass")
        base_mode = mode_l
        if base_mode.startswith("random_same_mass_"):
            base_mode = base_mode[len("random_same_mass_"):]
        if base_mode.endswith("_random_same_mass"):
            base_mode = base_mode[:-len("_random_same_mass")]

        role_values = {
            "semantic_role_negative": {_SEMANTIC_ROLE_NEGATIVE_SHORT},
            "semantic_role_source_skip": {_SEMANTIC_ROLE_NEGATIVE_SHORT},
            "semrole_negative": {_SEMANTIC_ROLE_NEGATIVE_SHORT},
            "semantic_role_positive": {_SEMANTIC_ROLE_POSITIVE_LONG},
            "semantic_role_anchor": {_SEMANTIC_ROLE_POSITIVE_LONG},
            "semantic_role_protect": {_SEMANTIC_ROLE_PROTECT_NEUTRAL},
            "semantic_role_protected": {_SEMANTIC_ROLE_PROTECT_NEUTRAL},
            "semantic_role_stable": {_SEMANTIC_ROLE_POSITIVE_LONG, _SEMANTIC_ROLE_PROTECT_NEUTRAL},
        }
        selected_roles = role_values.get(base_mode)
        if selected_roles is None:
            return None, stats

        raw = hmc_control.get("R_swa_tok") if hmc_control else None
        role_source = "current_head_R_swa_tok"
        missing = False
        if raw is None:
            score = torch.zeros(batch_size, source_tokens, device=device, dtype=torch.float32)
            missing = True
            role_source = "missing_R_swa_tok"
        else:
            flat = raw.to(device=device, dtype=torch.long)
            if int(flat.numel()) != int(batch_size * frame_num * tokens_per_frame):
                score = torch.zeros(batch_size, source_tokens, device=device, dtype=torch.float32)
                missing = True
                role_source = "shape_mismatch_R_swa_tok"
            else:
                roles = flat.reshape(batch_size, frame_num, tokens_per_frame)[:, :ov, :].reshape(
                    batch_size, ov * tokens_per_frame
                )
                if int(roles.shape[1]) < int(source_tokens):
                    score = torch.zeros(batch_size, source_tokens, device=device, dtype=torch.float32)
                    missing = True
                    role_source = "short_current_head_R_swa_tok"
                else:
                    if int(roles.shape[1]) != int(source_tokens):
                        roles = roles[:, :source_tokens]
                    score = torch.zeros(batch_size, source_tokens, device=device, dtype=torch.float32)
                    for role_id in sorted(int(v) for v in selected_roles):
                        score = torch.maximum(score, (roles == int(role_id)).float())

        selected_tokens_before_random = int((score > 0.0).sum().item())
        if random_same_mass and selected_tokens_before_random > 0:
            randomized = torch.zeros_like(score)
            chunk = float(hmc_control.get("semantic_action_chunk_idx", -1) if hmc_control else -1)
            base_idx = torch.arange(int(source_tokens), device=device, dtype=torch.float32)
            for b in range(int(batch_size)):
                k_select = min(int((score[b] > 0.0).sum().item()), int(source_tokens))
                if k_select <= 0:
                    continue
                salt = chunk * 149.0 + float(b) * 17.0 + 3000.0
                rand = torch.frac(torch.sin((base_idx + 1.0 + salt) * 12.9898) * 43758.5453)
                top = torch.topk(rand, k_select).indices
                randomized[b, top] = 1.0
            score = randomized

        selected_tokens = int((score > 0.0).sum().item())
        selected_mask = score > 0.0
        if selected_tokens > 0:
            local_idx = torch.arange(int(source_tokens), device=device, dtype=torch.float32).reshape(1, source_tokens)
            selected_index_mean = float(local_idx.expand(batch_size, -1)[selected_mask].mean().item())
            selected_d_mean = float(Dq.float()[selected_mask].mean().item())
        else:
            selected_index_mean = 0.0
            selected_d_mean = 0.0

        stats.update({
            "swa_overlap_source_semantic_role_mode": mode_l,
            "swa_overlap_source_semantic_role_base_mode": base_mode,
            "swa_overlap_source_semantic_role_label_source": role_source,
            "swa_overlap_source_semantic_role_missing_labels": bool(missing),
            "swa_overlap_source_semantic_role_random_same_mass": bool(random_same_mass),
            "swa_overlap_source_semantic_role_tokens_before_random": int(selected_tokens_before_random),
            "swa_overlap_source_semantic_role_selected_tokens": int(selected_tokens),
            "swa_overlap_source_semantic_role_selected_ratio": float(selected_tokens / max(int(batch_size * source_tokens), 1)),
            "swa_overlap_source_semantic_role_selected_index_mean": selected_index_mean,
            "swa_overlap_source_semantic_role_selected_D_mean": selected_d_mean,
            "swa_overlap_source_semantic_role_value_ids": sorted(int(v) for v in selected_roles),
        })
        return score.clamp(0.0, 1.0), stats

    @staticmethod
    def _dump_swa_overlap_feature_map(
        hmc_control: Optional[Dict[str, Any]],
        *,
        kind: str,
        mode: str,
        swa_layer_idx: int,
        batch_size: int,
        frame_num: int,
        tokens_per_frame: int,
        history_tokens: int,
        source_start: int,
        source_end: int,
        overlap_frames: int,
        Dq: torch.Tensor,
        Ds: torch.Tensor,
        score: torch.Tensor,
        control: torch.Tensor,
    ) -> Dict[str, Any]:
        dump_dir_text = str((hmc_control or {}).get("swa_overlap_feature_dump_dir", "") or "").strip()
        if not dump_dir_text:
            return {}
        try:
            dtype_name = str((hmc_control or {}).get("swa_overlap_feature_dump_dtype", "float16") or "float16").lower()
            dump_dtype = torch.float32 if dtype_name in {"float", "float32", "fp32"} else torch.float16
            dump_dir = Path(dump_dir_text)
            dump_dir.mkdir(parents=True, exist_ok=True)
            chunk_idx = int((hmc_control or {}).get("semantic_action_chunk_idx", -1))

            def _pack_map(x: torch.Tensor) -> torch.Tensor:
                flat = x.detach().cpu().to(dtype=dump_dtype)
                if tokens_per_frame > 0 and int(flat.shape[-1]) % int(tokens_per_frame) == 0:
                    frames = int(flat.shape[-1]) // int(tokens_per_frame)
                    return flat.reshape(int(batch_size), frames, int(tokens_per_frame))
                return flat

            out_path = dump_dir / (
                f"chunk_{chunk_idx:03d}_swa_overlap_{kind}_layer_{int(swa_layer_idx):02d}.pt"
            )
            payload = {
                "schema": "acl2_v68_swa_overlap_feature_map_v1",
                "artifact": "SAVE_V68_OVERLAP_FEATURES",
                "kind": str(kind),
                "mode": str(mode),
                "chunk_idx": int(chunk_idx),
                "swa_layer_idx": int(swa_layer_idx),
                "batch_size": int(batch_size),
                "frame_num": int(frame_num),
                "tokens_per_frame": int(tokens_per_frame),
                "history_tokens": int(history_tokens),
                "source_start": int(source_start),
                "source_end": int(source_end),
                "source_tokens": int(max(0, int(source_end) - int(source_start))),
                "overlap_frames_effective": int(overlap_frames),
                "runtime_swa_overlap_feature_not_qk_proxy": True,
                "Dq_overlap": _pack_map(Dq),
                "Ds_overlap": _pack_map(Ds),
                "score_overlap": _pack_map(score),
                "control_overlap": _pack_map(control),
                "score_mean": float(score.detach().float().mean().item()),
                "score_q90": float(torch.quantile(score.detach().float(), 0.90).item()),
                "control_mean": float(control.detach().float().mean().item()),
                "control_q90": float(torch.quantile(control.detach().float(), 0.90).item()),
            }
            torch.save(payload, out_path)
            return {
                "swa_overlap_feature_dump_path": str(out_path),
                "swa_overlap_feature_dump_schema": payload["schema"],
                "swa_overlap_feature_dump_kind": str(kind),
            }
        except Exception as exc:  # pragma: no cover - audit-only best effort.
            return {"swa_overlap_feature_dump_error": f"{type(exc).__name__}: {exc}"}

    @staticmethod
    def _swa_raw_transport_trace_layer_enabled(
        hmc_control: Optional[Dict[str, Any]],
        *,
        layer_idx: int,
        n_layers: int,
    ) -> bool:
        if not hmc_control or hmc_control.get("identity_hooks", False):
            return False
        if not str(hmc_control.get("swa_raw_transport_trace_dir", "") or "").strip():
            return False
        mode = str(hmc_control.get("swa_raw_transport_trace_layer_mode", "all") or "all")
        single = int(hmc_control.get("swa_raw_transport_trace_single_layer", -1) or -1)
        if mode == "first":
            return int(layer_idx) == 0
        if mode == "last":
            return int(layer_idx) == max(int(n_layers) - 1, 0)
        if mode == "single":
            return int(layer_idx) == int(single)
        return True

    @staticmethod
    def _swa_trace_source_values(
        hmc_control: Optional[Dict[str, Any]],
        key: str,
        *,
        batch_size: int,
        history_tokens: int,
        tokens_per_frame: int = 0,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        value = (hmc_control or {}).get(key)
        if value is None or not hasattr(value, "detach"):
            return None
        flat = value.detach().to(device=device).reshape(-1)

        def _interleave_patch_tokens(x: torch.Tensor, samples: int) -> Optional[torch.Tensor]:
            if int(tokens_per_frame) <= 0 or int(history_tokens) % int(tokens_per_frame) != 0:
                return None
            frames = int(history_tokens) // int(tokens_per_frame)
            if frames <= 0:
                return None
            if int(x.numel()) % int(samples) != 0:
                return None
            per_sample = int(x.numel()) // int(samples)
            if per_sample <= 0 or per_sample >= int(history_tokens):
                return None
            if per_sample % frames != 0:
                return None
            patch_tokens_per_frame = per_sample // frames
            patch_start = int(tokens_per_frame) - int(patch_tokens_per_frame)
            if patch_start < 0:
                return None
            shaped = x.reshape(int(samples), frames, patch_tokens_per_frame)
            out = torch.zeros(int(samples), frames, int(tokens_per_frame), device=device, dtype=x.dtype)
            out[:, :, patch_start : patch_start + patch_tokens_per_frame] = shaped
            return out.reshape(int(samples), int(history_tokens))

        if int(flat.numel()) == int(history_tokens):
            flat = flat.reshape(1, int(history_tokens)).repeat(int(batch_size), 1)
        elif int(flat.numel()) < int(history_tokens) and int(batch_size) == 1:
            interleaved = _interleave_patch_tokens(flat, 1)
            if interleaved is not None:
                flat = interleaved
            else:
                pad = int(history_tokens) - int(flat.numel())
                flat = torch.cat([torch.zeros(pad, device=device, dtype=flat.dtype), flat], dim=0)
                flat = flat.reshape(1, int(history_tokens))
        elif int(flat.numel()) == int(batch_size) * int(history_tokens):
            flat = flat.reshape(int(batch_size), int(history_tokens))
        elif int(batch_size) > 1 and int(flat.numel()) < int(batch_size) * int(history_tokens):
            per_sample = int(flat.numel()) // int(batch_size)
            if per_sample > 0 and per_sample * int(batch_size) == int(flat.numel()) and per_sample < int(history_tokens):
                interleaved = _interleave_patch_tokens(flat, int(batch_size))
                if interleaved is not None:
                    flat = interleaved
                else:
                    pad = int(history_tokens) - int(per_sample)
                    flat = flat.reshape(int(batch_size), per_sample)
                    flat = torch.cat(
                        [torch.zeros(int(batch_size), pad, device=device, dtype=flat.dtype), flat],
                        dim=1,
                    )
            else:
                return None
        else:
            return None
        return flat.to(dtype=dtype)

    @staticmethod
    def _swa_trace_source_matrix(
        hmc_control: Optional[Dict[str, Any]],
        key: str,
        *,
        batch_size: int,
        history_tokens: int,
        tokens_per_frame: int = 0,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        value = (hmc_control or {}).get(key)
        if value is None or not hasattr(value, "detach"):
            return None
        raw = value.detach().to(device=device)
        if raw.ndim <= 1:
            mat = raw.reshape(-1, 1)
        else:
            mat = raw.reshape(-1, int(raw.shape[-1]))
        feature_dim = int(mat.shape[-1])
        if feature_dim <= 0:
            return None

        def _interleave_patch_tokens(x: torch.Tensor, samples: int) -> Optional[torch.Tensor]:
            if int(tokens_per_frame) <= 0 or int(history_tokens) % int(tokens_per_frame) != 0:
                return None
            frames = int(history_tokens) // int(tokens_per_frame)
            if frames <= 0:
                return None
            if int(x.shape[0]) % int(samples) != 0:
                return None
            per_sample = int(x.shape[0]) // int(samples)
            if per_sample <= 0 or per_sample >= int(history_tokens):
                return None
            if per_sample % frames != 0:
                return None
            patch_tokens_per_frame = per_sample // frames
            patch_start = int(tokens_per_frame) - int(patch_tokens_per_frame)
            if patch_start < 0:
                return None
            shaped = x.reshape(int(samples), frames, patch_tokens_per_frame, feature_dim)
            out = torch.zeros(
                int(samples),
                frames,
                int(tokens_per_frame),
                feature_dim,
                device=device,
                dtype=x.dtype,
            )
            out[:, :, patch_start : patch_start + patch_tokens_per_frame, :] = shaped
            return out.reshape(int(samples), int(history_tokens), feature_dim)

        if int(mat.shape[0]) == int(history_tokens):
            mat = mat.reshape(1, int(history_tokens), feature_dim).repeat(int(batch_size), 1, 1)
        elif int(mat.shape[0]) < int(history_tokens) and int(batch_size) == 1:
            interleaved = _interleave_patch_tokens(mat, 1)
            if interleaved is not None:
                mat = interleaved
            else:
                pad = int(history_tokens) - int(mat.shape[0])
                mat = torch.cat(
                    [torch.zeros(pad, feature_dim, device=device, dtype=mat.dtype), mat],
                    dim=0,
                ).reshape(1, int(history_tokens), feature_dim)
        elif int(mat.shape[0]) == int(batch_size) * int(history_tokens):
            mat = mat.reshape(int(batch_size), int(history_tokens), feature_dim)
        elif int(batch_size) > 1 and int(mat.shape[0]) < int(batch_size) * int(history_tokens):
            per_sample = int(mat.shape[0]) // int(batch_size)
            if per_sample > 0 and per_sample * int(batch_size) == int(mat.shape[0]) and per_sample < int(history_tokens):
                interleaved = _interleave_patch_tokens(mat, int(batch_size))
                if interleaved is not None:
                    mat = interleaved
                else:
                    pad = int(history_tokens) - int(per_sample)
                    mat = mat.reshape(int(batch_size), per_sample, feature_dim)
                    mat = torch.cat(
                        [torch.zeros(int(batch_size), pad, feature_dim, device=device, dtype=mat.dtype), mat],
                        dim=1,
                    )
            else:
                return None
        else:
            return None
        return mat.to(dtype=dtype)

    @staticmethod
    def _swa_trace_same_count_control(mask: torch.Tensor, *, salt: float) -> torch.Tensor:
        mask_bool = mask.detach().bool()
        out = torch.zeros_like(mask_bool)
        token_idx = torch.arange(int(mask_bool.shape[1]), device=mask_bool.device, dtype=torch.float32)
        base_scores = torch.frac(torch.sin((token_idx + 1.0 + float(salt)) * 12.9898) * 43758.5453)
        for b in range(int(mask_bool.shape[0])):
            count = int(mask_bool[b].sum().item())
            if count <= 0:
                continue
            selected = torch.topk(base_scores + float(b) * 1e-6, k=min(count, int(mask_bool.shape[1]))).indices
            out[b, selected] = True
        return out

    @staticmethod
    def _dump_swa_raw_transport_trace(
        hmc_control: Optional[Dict[str, Any]],
        *,
        layer: int,
        swa_layer_idx: int,
        batch_size: int,
        frame_num: int,
        tokens_per_frame: int,
        q_current: torch.Tensor,
        k_current: torch.Tensor,
        v_current: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        hidden_current: Optional[torch.Tensor] = None,
        hidden_cache: Optional[torch.Tensor] = None,
        extra_trace_fields: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        dump_dir_text = str((hmc_control or {}).get("swa_raw_transport_trace_dir", "") or "").strip()
        if not dump_dir_text:
            return {}
        try:
            max_queries = int((hmc_control or {}).get("swa_raw_transport_trace_max_queries", 128) or 128)
            max_queries = max(1, max_queries)
            history_tokens = int(k_cache.shape[2])
            current_tokens = int(q_current.shape[2])
            head_count = int(q_current.shape[1])
            head_dim = int(q_current.shape[-1])
            q_idx = torch.linspace(
                0,
                max(current_tokens - 1, 0),
                steps=min(max_queries, current_tokens),
                device=q_current.device,
            ).round().long().unique(sorted=True)
            if int(q_idx.numel()) == 0:
                q_idx = torch.arange(min(1, current_tokens), device=q_current.device, dtype=torch.long)

            with torch.no_grad():
                direct_match_only = bool((hmc_control or {}).get("swa_raw_transport_trace_direct_match_only", False))
                q_sample = q_current.index_select(2, q_idx).detach().float()
                k_hist = k_cache.detach().float()
                if direct_match_only:
                    topk_count = int((hmc_control or {}).get("swa_raw_transport_trace_topk", 8) or 8)
                    topk_count = max(1, min(int(topk_count), int(history_tokens)))
                    query_block_size = int(
                        (hmc_control or {}).get("swa_raw_transport_trace_query_block_size", 128) or 128
                    )
                    query_block_size = max(1, int(query_block_size))
                    topk_scores_parts: List[torch.Tensor] = []
                    topk_indices_parts: List[torch.Tensor] = []
                    scale = max(float(head_dim) ** 0.5, 1.0)
                    for q_start in range(0, int(q_sample.shape[2]), int(query_block_size)):
                        q_end = min(q_start + int(query_block_size), int(q_sample.shape[2]))
                        q_block = q_sample[:, :, q_start:q_end, :]
                        scores_block = torch.matmul(q_block, k_hist.transpose(-2, -1)) / scale
                        block_scores, block_indices = torch.topk(scores_block, k=topk_count, dim=-1)
                        topk_scores_parts.append(block_scores.detach().cpu().to(torch.float16))
                        topk_indices_parts.append(block_indices.detach().cpu().to(torch.int32))
                        del scores_block, block_scores, block_indices, q_block
                    topk_scores_cpu = torch.cat(topk_scores_parts, dim=2) if topk_scores_parts else torch.empty(0)
                    topk_indices_cpu = torch.cat(topk_indices_parts, dim=2) if topk_indices_parts else torch.empty(0)

                    def _current_trace_values_direct(key: str, dtype: torch.dtype) -> Optional[torch.Tensor]:
                        raw = (hmc_control or {}).get(key)
                        if not torch.is_tensor(raw) or int(current_tokens) <= 0:
                            return None
                        vals = raw.detach().cpu().to(dtype=dtype)
                        needed = int(batch_size) * int(current_tokens)
                        if int(vals.numel()) < needed:
                            return None
                        return vals.reshape(int(batch_size), -1)[:, : int(current_tokens)]

                    def _source_values_direct(key: str, dtype: torch.dtype) -> Optional[torch.Tensor]:
                        vals = Pi3._swa_trace_source_values(
                            hmc_control,
                            key,
                            batch_size=batch_size,
                            history_tokens=history_tokens,
                            tokens_per_frame=tokens_per_frame,
                            device=q_current.device,
                            dtype=dtype,
                        )
                        if vals is None:
                            return None
                        return vals.detach().cpu().to(dtype=dtype)

                    def _topk_source_values_direct(
                        values: Optional[torch.Tensor],
                        dtype: torch.dtype,
                    ) -> Optional[torch.Tensor]:
                        if values is None or int(topk_indices_cpu.numel()) == 0:
                            return None
                        vals = values.detach().cpu().to(dtype=dtype)
                        if int(vals.numel()) < int(batch_size) * int(history_tokens):
                            return None
                        vals = vals.reshape(int(batch_size), -1)[:, : int(history_tokens)]
                        idx = topk_indices_cpu.long()
                        expanded = vals[:, None, None, :].expand(
                            int(batch_size),
                            int(head_count),
                            int(idx.shape[2]),
                            int(history_tokens),
                        )
                        return torch.gather(expanded, -1, idx)

                    def _bool_mean_direct(x: Optional[torch.Tensor]) -> Optional[float]:
                        if x is None:
                            return None
                        return float(x.detach().float().mean().item())

                    def _current_unique_hist(
                        values: Optional[torch.Tensor],
                    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], int, int]:
                        if values is None:
                            return None, None, 0, 0
                        flat = values.detach().long().reshape(-1)
                        nonnegative = flat[flat >= 0]
                        if int(nonnegative.numel()) == 0:
                            return None, None, 0, 0
                        unique, counts = torch.unique(nonnegative, sorted=True, return_counts=True)
                        return (
                            unique.detach().cpu().to(torch.int32),
                            counts.detach().cpu().to(torch.int32),
                            int(nonnegative.numel()),
                            int(unique.numel()),
                        )

                    l_current = _current_trace_values_direct("L_sem_tok", torch.int64)
                    g_current = _current_trace_values_direct("G_sem_tok", torch.int64)
                    stage_c_seed_current = _current_trace_values_direct(
                        "stage_c_seed_global_track_idx_tok",
                        torch.int64,
                    )
                    stage_c_masklet_instance_current = _current_trace_values_direct(
                        "stage_c_masklet_instance_idx_tok",
                        torch.int64,
                    )
                    q_idx_cpu = q_idx.detach().cpu().long()
                    sampled_query_fine_labels = (
                        l_current.index_select(1, q_idx_cpu) if l_current is not None and int(q_idx_cpu.numel()) else None
                    )
                    sampled_query_group_labels = (
                        g_current.index_select(1, q_idx_cpu) if g_current is not None and int(q_idx_cpu.numel()) else None
                    )
                    sampled_query_stage_c_seed_global_track_idx = (
                        stage_c_seed_current.index_select(1, q_idx_cpu)
                        if stage_c_seed_current is not None and int(q_idx_cpu.numel()) else None
                    )
                    sampled_query_stage_c_masklet_instance_idx = (
                        stage_c_masklet_instance_current.index_select(1, q_idx_cpu)
                        if stage_c_masklet_instance_current is not None and int(q_idx_cpu.numel()) else None
                    )
                    (
                        current_stage_c_seed_unique_ids,
                        current_stage_c_seed_unique_counts,
                        current_stage_c_seed_nonnegative_count,
                        current_stage_c_seed_unique_count,
                    ) = _current_unique_hist(stage_c_seed_current)
                    (
                        current_stage_c_masklet_instance_unique_ids,
                        current_stage_c_masklet_instance_unique_counts,
                        current_stage_c_masklet_instance_nonnegative_count,
                        current_stage_c_masklet_instance_unique_count,
                    ) = _current_unique_hist(stage_c_masklet_instance_current)

                    l_prev = _source_values_direct("L_prev_patch", torch.float32)
                    g_prev = _source_values_direct("G_prev_patch", torch.float32)
                    stage_c_seed_prev = _source_values_direct("stage_c_seed_global_track_idx_prev_patch", torch.int64)
                    stage_c_masklet_instance_prev = _source_values_direct(
                        "stage_c_masklet_instance_idx_prev_patch",
                        torch.int64,
                    )
                    ttt_prev_tracked_instance = _source_values_direct(
                        "prev_ttt_tracked_instance_anchor_mask_patch",
                        torch.float32,
                    )
                    ttt_prev_tracked_instance_anchor_ids = _source_values_direct(
                        "prev_ttt_tracked_instance_anchor_id_patch",
                        torch.int64,
                    )
                    ttt_prev_tracked_instance_anchor_seeds = _source_values_direct(
                        "prev_ttt_tracked_instance_anchor_seed_patch",
                        torch.int64,
                    )
                    topk_cache_fine_labels = _topk_source_values_direct(l_prev, torch.int64)
                    topk_cache_group_labels = _topk_source_values_direct(g_prev, torch.int64)
                    topk_cache_stage_c_seed_global_track_idx = _topk_source_values_direct(
                        stage_c_seed_prev,
                        torch.int64,
                    )
                    topk_cache_stage_c_masklet_instance_idx = _topk_source_values_direct(
                        stage_c_masklet_instance_prev,
                        torch.int64,
                    )
                    topk_tracked_instance_anchor_mask_values = _topk_source_values_direct(
                        ttt_prev_tracked_instance,
                        torch.float32,
                    )
                    topk_tracked_instance_anchor_hit_mask = (
                        topk_tracked_instance_anchor_mask_values >= 0.5
                        if topk_tracked_instance_anchor_mask_values is not None
                        else None
                    )
                    topk_tracked_instance_anchor_ids = _topk_source_values_direct(
                        ttt_prev_tracked_instance_anchor_ids,
                        torch.int64,
                    )
                    if topk_tracked_instance_anchor_hit_mask is not None and topk_tracked_instance_anchor_ids is not None:
                        topk_tracked_instance_anchor_ids = torch.where(
                            topk_tracked_instance_anchor_hit_mask,
                            topk_tracked_instance_anchor_ids,
                            torch.full_like(topk_tracked_instance_anchor_ids, -1),
                        )
                    topk_tracked_instance_anchor_seeds = _topk_source_values_direct(
                        ttt_prev_tracked_instance_anchor_seeds,
                        torch.int64,
                    )
                    if topk_tracked_instance_anchor_hit_mask is not None and topk_tracked_instance_anchor_seeds is not None:
                        topk_tracked_instance_anchor_seeds = torch.where(
                            topk_tracked_instance_anchor_hit_mask,
                            topk_tracked_instance_anchor_seeds,
                            torch.full_like(topk_tracked_instance_anchor_seeds, -1),
                        )
                    topk_same_fine_labels: Optional[torch.Tensor] = None
                    topk_same_group_labels: Optional[torch.Tensor] = None
                    topk_same_stage_c_seed_global_track_idx: Optional[torch.Tensor] = None
                    topk_same_stage_c_masklet_instance_idx: Optional[torch.Tensor] = None
                    if sampled_query_fine_labels is not None and topk_cache_fine_labels is not None:
                        q_fine = sampled_query_fine_labels[:, None, :, None].expand_as(topk_cache_fine_labels)
                        topk_same_fine_labels = (topk_cache_fine_labels == q_fine) & (topk_cache_fine_labels > 0)
                    if sampled_query_group_labels is not None and topk_cache_group_labels is not None:
                        q_group = sampled_query_group_labels[:, None, :, None].expand_as(topk_cache_group_labels)
                        topk_same_group_labels = (topk_cache_group_labels == q_group) & (topk_cache_group_labels > 0)
                    if (
                        sampled_query_stage_c_seed_global_track_idx is not None
                        and topk_cache_stage_c_seed_global_track_idx is not None
                    ):
                        q_seed = sampled_query_stage_c_seed_global_track_idx[:, None, :, None].expand_as(
                            topk_cache_stage_c_seed_global_track_idx
                        )
                        topk_same_stage_c_seed_global_track_idx = (
                            topk_cache_stage_c_seed_global_track_idx == q_seed
                        ) & (topk_cache_stage_c_seed_global_track_idx >= 0)
                    if (
                        sampled_query_stage_c_masklet_instance_idx is not None
                        and topk_cache_stage_c_masklet_instance_idx is not None
                    ):
                        q_inst = sampled_query_stage_c_masklet_instance_idx[:, None, :, None].expand_as(
                            topk_cache_stage_c_masklet_instance_idx
                        )
                        topk_same_stage_c_masklet_instance_idx = (
                            topk_cache_stage_c_masklet_instance_idx == q_inst
                        ) & (topk_cache_stage_c_masklet_instance_idx >= 0)
                    topk_tracked_instance_same_seed_hits = (
                        topk_tracked_instance_anchor_hit_mask & topk_same_stage_c_seed_global_track_idx
                        if topk_tracked_instance_anchor_hit_mask is not None
                        and topk_same_stage_c_seed_global_track_idx is not None
                        else None
                    )
                    topk_tracked_instance_same_masklet_hits = (
                        topk_tracked_instance_anchor_hit_mask & topk_same_stage_c_masklet_instance_idx
                        if topk_tracked_instance_anchor_hit_mask is not None
                        and topk_same_stage_c_masklet_instance_idx is not None
                        else None
                    )

                    def _tracked_instance_anchor_lifecycle_rows_direct() -> List[Dict[str, Any]]:
                        if (
                            ttt_prev_tracked_instance is None
                            or ttt_prev_tracked_instance_anchor_ids is None
                            or ttt_prev_tracked_instance_anchor_seeds is None
                            or topk_tracked_instance_anchor_ids is None
                            or topk_tracked_instance_anchor_hit_mask is None
                        ):
                            return []
                        valid_topk = topk_tracked_instance_anchor_ids[topk_tracked_instance_anchor_ids >= 0]
                        if int(valid_topk.numel()) == 0:
                            return []
                        unique_ids, counts = torch.unique(valid_topk, return_counts=True)
                        order = torch.argsort(counts, descending=True)
                        max_rows = int(
                            (hmc_control or {}).get(
                                "swa_raw_transport_trace_tracked_instance_lifecycle_max_ids",
                                (hmc_control or {}).get("swa_raw_transport_trace_anchor_lifecycle_max_ids", 128),
                            )
                            or 128
                        )
                        max_rows = max(1, min(max_rows, int(unique_ids.numel())))
                        denom_qh = max(int(batch_size) * int(head_count) * int(topk_indices_cpu.shape[2]), 1)
                        rows_out: List[Dict[str, Any]] = []
                        tracked_bool = ttt_prev_tracked_instance.to(dtype=torch.bool)
                        ids_hist = ttt_prev_tracked_instance_anchor_ids.to(dtype=torch.int64)
                        seed_hist = ttt_prev_tracked_instance_anchor_seeds.to(dtype=torch.int64)
                        topk_score_float = topk_scores_cpu.to(dtype=torch.float32)
                        for idx in order[:max_rows]:
                            anchor_id = int(unique_ids[idx].item())
                            hist_mask = tracked_bool & (ids_hist == int(anchor_id))
                            hit_pos = topk_tracked_instance_anchor_ids == int(anchor_id)
                            qh_hit = hit_pos.any(dim=-1)
                            score_sum_for_anchor = torch.where(
                                hit_pos,
                                topk_score_float,
                                torch.zeros_like(topk_score_float),
                            ).sum(dim=-1)
                            source_seed_mode: Optional[int] = None
                            source_seed_mode_frac: Optional[float] = None
                            if bool(hist_mask.any()):
                                seed_vals = seed_hist[hist_mask]
                                seed_vals = seed_vals[seed_vals >= 0]
                                if int(seed_vals.numel()) > 0:
                                    seed_ids, seed_counts = torch.unique(seed_vals, return_counts=True)
                                    seed_best = torch.argmax(seed_counts)
                                    source_seed_mode = int(seed_ids[seed_best].item())
                                    source_seed_mode_frac = float(
                                        seed_counts[seed_best].float().div(seed_counts.sum().clamp_min(1)).item()
                                    )
                            same_seed_count = 0
                            if topk_tracked_instance_same_seed_hits is not None:
                                same_seed_count = int((hit_pos & topk_tracked_instance_same_seed_hits).sum().item())
                            same_masklet_count = 0
                            if topk_tracked_instance_same_masklet_hits is not None:
                                same_masklet_count = int((hit_pos & topk_tracked_instance_same_masklet_hits).sum().item())
                            qh_by_head = (
                                qh_hit.float().mean(dim=(0, 2))
                                if qh_hit.ndim == 3
                                else torch.empty(0, device=topk_scores_cpu.device)
                            )
                            rows_out.append(
                                {
                                    "anchor_id": int(anchor_id),
                                    "source_type": "thing_tracked",
                                    "source_token_count": int(hist_mask.sum().item()),
                                    "topk_hit_position_count": int(hit_pos.sum().item()),
                                    "topk_same_seed_position_count": same_seed_count,
                                    "topk_same_masklet_position_count": same_masklet_count,
                                    "query_head_hit_frac": float(qh_hit.float().sum().item() / float(denom_qh)),
                                    "query_head_hit_max": float(qh_by_head.max().item())
                                    if int(qh_by_head.numel()) else None,
                                    "query_head_ge50_frac": float((qh_by_head >= 0.50).float().mean().item())
                                    if int(qh_by_head.numel()) else None,
                                    "query_head_ge75_frac": float((qh_by_head >= 0.75).float().mean().item())
                                    if int(qh_by_head.numel()) else None,
                                    "direct_match_topk_score_sum_mean": float(
                                        score_sum_for_anchor.float().mean().item()
                                    ),
                                    "direct_match_topk_score_sum_max": float(
                                        score_sum_for_anchor.float().max().item()
                                    ),
                                    "topk_route_mass_mean": None,
                                    "topk_route_mass_max": None,
                                    "source_stage_c_seed_global_track_idx_mode": source_seed_mode,
                                    "source_stage_c_seed_global_track_idx_mode_frac": source_seed_mode_frac,
                                    "source_chunk_idx": int(
                                        (hmc_control or {}).get(
                                            "prev_ttt_tracked_instance_anchor_source_chunk_idx",
                                            -1,
                                        )
                                        or -1
                                    ),
                                    "current_chunk_idx": int(
                                        (hmc_control or {}).get("semantic_action_chunk_idx", -1) or -1
                                    ),
                                    "runtime_action_allowed": False,
                                    "claim_level": "diagnostic_tracked_instance_anchor_lifecycle_no_runtime",
                                }
                            )
                        return rows_out

                    ttt_prev_tracked_instance_lifecycle_rows_direct = (
                        _tracked_instance_anchor_lifecycle_rows_direct()
                    )

                    topk_frames: Optional[torch.Tensor] = None
                    topk_identity_frame_missing_reason = ""
                    if int(tokens_per_frame) > 0 and int(history_tokens) % int(tokens_per_frame) == 0:
                        topk_frames = torch.div(topk_indices_cpu.long(), int(tokens_per_frame), rounding_mode="floor")
                    else:
                        topk_identity_frame_missing_reason = (
                            "tokens_per_frame<=0 or history_tokens not divisible by tokens_per_frame"
                        )
                    hidden_trace_debug: Dict[str, Any] = {
                        "hidden_current_input_shape": list(hidden_current.shape)
                        if torch.is_tensor(hidden_current) else None,
                        "hidden_cache_input_shape": list(hidden_cache.shape)
                        if torch.is_tensor(hidden_cache) else None,
                        "hidden_current_accepted": False,
                        "hidden_cache_accepted": False,
                        "hidden_trace_reason": "direct_match_only_skips_hidden_trace",
                    }
                    dump_dir = Path(dump_dir_text)
                    dump_dir.mkdir(parents=True, exist_ok=True)
                    chunk_idx = int((hmc_control or {}).get("semantic_action_chunk_idx", -1))
                    out_path = dump_dir / f"chunk_{chunk_idx:03d}_swa_raw_transport_layer_{int(swa_layer_idx):02d}.pt"
                    payload = {
                        "schema": "acl2_v103_swa_raw_transport_direct_match_trace_v2",
                        "artifact": "SAVE_V103_SWA_RAW_TRANSPORT_DIRECT_MATCH_TOPK_IDENTITY",
                        "diagnostic_only": True,
                        "direct_match_only": True,
                        "direct_match_query_block_size": int(query_block_size),
                        "chunk_idx": int(chunk_idx),
                        "layer": int(layer),
                        "swa_layer_idx": int(swa_layer_idx),
                        "batch_size": int(batch_size),
                        "frame_num": int(frame_num),
                        "tokens_per_frame": int(tokens_per_frame),
                        "head_count": int(head_count),
                        "current_tokens": int(current_tokens),
                        "history_tokens": int(history_tokens),
                        "current_semantic_fine_trace_available": bool(sampled_query_fine_labels is not None),
                        "current_semantic_group_trace_available": bool(sampled_query_group_labels is not None),
                        "current_stage_c_seed_global_track_idx_trace_available": bool(
                            sampled_query_stage_c_seed_global_track_idx is not None
                        ),
                        "cache_stage_c_seed_global_track_idx_trace_available": bool(
                            topk_cache_stage_c_seed_global_track_idx is not None
                        ),
                        "current_stage_c_masklet_instance_idx_trace_available": bool(
                            sampled_query_stage_c_masklet_instance_idx is not None
                        ),
                        "cache_stage_c_masklet_instance_idx_trace_available": bool(
                            topk_cache_stage_c_masklet_instance_idx is not None
                        ),
                        "ttt_prev_tracked_instance_anchor_identity_available": bool(
                            topk_tracked_instance_anchor_hit_mask is not None
                            and bool(topk_tracked_instance_anchor_hit_mask.any())
                        ),
                        "ttt_prev_tracked_instance_anchor_source_token_count": int(
                            (hmc_control or {}).get("prev_ttt_tracked_instance_anchor_token_count", 0) or 0
                        ),
                        "ttt_prev_tracked_instance_anchor_source_chunk_idx": int(
                            (hmc_control or {}).get("prev_ttt_tracked_instance_anchor_source_chunk_idx", -1) or -1
                        ),
                        "ttt_prev_tracked_instance_anchor_lifecycle_schema": (
                            "acl2_v103_tracked_instance_anchor_lifecycle_rows_v1"
                        ),
                        "ttt_prev_tracked_instance_anchor_lifecycle_rows": (
                            ttt_prev_tracked_instance_lifecycle_rows_direct
                        ),
                        "ttt_prev_tracked_instance_anchor_lifecycle_row_count": int(
                            len(ttt_prev_tracked_instance_lifecycle_rows_direct)
                        ),
                        "sampled_query_count": int(q_idx.numel()),
                        "sampled_query_indices": q_idx.detach().cpu(),
                        "sampled_query_fine_label_ids": (
                            sampled_query_fine_labels.detach().cpu().to(torch.int16)
                            if sampled_query_fine_labels is not None else None
                        ),
                        "sampled_query_group_ids": (
                            sampled_query_group_labels.detach().cpu().to(torch.int16)
                            if sampled_query_group_labels is not None else None
                        ),
                        "sampled_query_stage_c_seed_global_track_idx": (
                            sampled_query_stage_c_seed_global_track_idx.detach().cpu().to(torch.int32)
                            if sampled_query_stage_c_seed_global_track_idx is not None else None
                        ),
                        "sampled_query_stage_c_masklet_instance_idx": (
                            sampled_query_stage_c_masklet_instance_idx.detach().cpu().to(torch.int32)
                            if sampled_query_stage_c_masklet_instance_idx is not None else None
                        ),
                        "current_stage_c_seed_global_track_idx_unique_ids": current_stage_c_seed_unique_ids,
                        "current_stage_c_seed_global_track_idx_unique_counts": current_stage_c_seed_unique_counts,
                        "current_stage_c_seed_global_track_idx_nonnegative_count": int(
                            current_stage_c_seed_nonnegative_count
                        ),
                        "current_stage_c_seed_global_track_idx_unique_count": int(current_stage_c_seed_unique_count),
                        "current_stage_c_masklet_instance_idx_unique_ids": current_stage_c_masklet_instance_unique_ids,
                        "current_stage_c_masklet_instance_idx_unique_counts": (
                            current_stage_c_masklet_instance_unique_counts
                        ),
                        "current_stage_c_masklet_instance_idx_nonnegative_count": int(
                            current_stage_c_masklet_instance_nonnegative_count
                        ),
                        "current_stage_c_masklet_instance_idx_unique_count": int(
                            current_stage_c_masklet_instance_unique_count
                        ),
                        "topk_identity_available": True,
                        "topk_identity_topk": int(topk_count),
                        "topk_identity_missing_reason": "",
                        "topk_identity_frame_missing_reason": topk_identity_frame_missing_reason,
                        "q_current_shape": list(q_current.shape),
                        "k_current_shape": list(k_current.shape),
                        "v_current_shape": list(v_current.shape),
                        "k_cache_shape": list(k_cache.shape),
                        "v_cache_shape": list(v_cache.shape),
                        **hidden_trace_debug,
                        "current_Q_to_cache_K_topk_cache_indices": topk_indices_cpu.to(torch.int32),
                        "current_Q_to_cache_K_topk_cache_fine_label_ids": (
                            topk_cache_fine_labels.detach().cpu().to(torch.int16)
                            if topk_cache_fine_labels is not None else None
                        ),
                        "current_Q_to_cache_K_topk_cache_group_ids": (
                            topk_cache_group_labels.detach().cpu().to(torch.int16)
                            if topk_cache_group_labels is not None else None
                        ),
                        "current_Q_to_cache_K_topk_cache_stage_c_seed_global_track_idx": (
                            topk_cache_stage_c_seed_global_track_idx.detach().cpu().to(torch.int32)
                            if topk_cache_stage_c_seed_global_track_idx is not None else None
                        ),
                        "current_Q_to_cache_K_topk_cache_stage_c_masklet_instance_idx": (
                            topk_cache_stage_c_masklet_instance_idx.detach().cpu().to(torch.int32)
                            if topk_cache_stage_c_masklet_instance_idx is not None else None
                        ),
                        "current_Q_to_cache_K_topk_same_fine_label": (
                            topk_same_fine_labels.detach().cpu()
                            if topk_same_fine_labels is not None else None
                        ),
                        "current_Q_to_cache_K_topk_same_group": (
                            topk_same_group_labels.detach().cpu()
                            if topk_same_group_labels is not None else None
                        ),
                        "current_Q_to_cache_K_topk_same_stage_c_seed_global_track_idx": (
                            topk_same_stage_c_seed_global_track_idx.detach().cpu()
                            if topk_same_stage_c_seed_global_track_idx is not None else None
                        ),
                        "current_Q_to_cache_K_topk_same_stage_c_masklet_instance_idx": (
                            topk_same_stage_c_masklet_instance_idx.detach().cpu()
                            if topk_same_stage_c_masklet_instance_idx is not None else None
                        ),
                        "current_Q_to_cache_K_topk_ttt_prev_tracked_instance_anchor_hit_mask": (
                            topk_tracked_instance_anchor_hit_mask.detach().cpu()
                            if topk_tracked_instance_anchor_hit_mask is not None else None
                        ),
                        "current_Q_to_cache_K_topk_ttt_prev_tracked_instance_anchor_ids": (
                            topk_tracked_instance_anchor_ids.detach().cpu().to(torch.int64)
                            if topk_tracked_instance_anchor_ids is not None else None
                        ),
                        "current_Q_to_cache_K_topk_ttt_prev_tracked_instance_anchor_seeds": (
                            topk_tracked_instance_anchor_seeds.detach().cpu().to(torch.int64)
                            if topk_tracked_instance_anchor_seeds is not None else None
                        ),
                        "current_Q_to_cache_K_topk_ttt_prev_tracked_instance_anchor_same_seed": (
                            topk_tracked_instance_same_seed_hits.detach().cpu()
                            if topk_tracked_instance_same_seed_hits is not None else None
                        ),
                        "current_Q_to_cache_K_topk_ttt_prev_tracked_instance_anchor_same_masklet": (
                            topk_tracked_instance_same_masklet_hits.detach().cpu()
                            if topk_tracked_instance_same_masklet_hits is not None else None
                        ),
                        "ttt_prev_tracked_instance_anchor_topk_hit_frac_mean": _bool_mean_direct(
                            topk_tracked_instance_anchor_hit_mask
                        ),
                        "ttt_prev_tracked_instance_anchor_topk_same_seed_frac_mean": _bool_mean_direct(
                            topk_tracked_instance_same_seed_hits
                        ),
                        "ttt_prev_tracked_instance_anchor_topk_same_masklet_frac_mean": _bool_mean_direct(
                            topk_tracked_instance_same_masklet_hits
                        ),
                        "current_Q_to_cache_K_topk_cache_frames": (
                            topk_frames.detach().cpu().to(torch.int16) if topk_frames is not None else None
                        ),
                        "current_Q_to_cache_K_topk_scores": topk_scores_cpu.to(torch.float16),
                    }
                    torch.save(payload, out_path)
                    return {
                        "swa_raw_transport_trace_available": True,
                        "swa_raw_transport_trace_path": str(out_path),
                        "swa_raw_transport_trace_schema": payload["schema"],
                        "swa_raw_transport_trace_sampled_query_count": int(q_idx.numel()),
                        "swa_raw_transport_trace_head_count": int(head_count),
                        "swa_raw_transport_trace_direct_match_only": True,
                        "swa_raw_transport_trace_query_block_size": int(query_block_size),
                        "swa_raw_transport_topk_identity_available": True,
                        "swa_raw_transport_topk_identity_topk": int(topk_count),
                        "swa_raw_transport_current_tokens": int(current_tokens),
                        "swa_raw_transport_history_tokens": int(history_tokens),
                    }

                k_sample = k_current.index_select(2, q_idx).detach().float()
                v_sample = v_current.index_select(2, q_idx).detach().float()
                v_hist = v_cache.detach().float()
                hidden_sample: Optional[torch.Tensor] = None
                hidden_hist: Optional[torch.Tensor] = None
                hidden_trace_debug: Dict[str, Any] = {
                    "hidden_current_input_shape": list(hidden_current.shape)
                    if torch.is_tensor(hidden_current) else None,
                    "hidden_cache_input_shape": list(hidden_cache.shape)
                    if torch.is_tensor(hidden_cache) else None,
                    "hidden_current_accepted": False,
                    "hidden_cache_accepted": False,
                    "hidden_trace_reason": "",
                }
                if (
                    torch.is_tensor(hidden_current)
                    and hidden_current.ndim == 3
                    and int(hidden_current.shape[0]) == int(batch_size)
                    and int(hidden_current.shape[1]) >= int(current_tokens)
                ):
                    hidden_sample = hidden_current.index_select(1, q_idx).detach().float()
                    hidden_trace_debug["hidden_current_accepted"] = True
                if (
                    torch.is_tensor(hidden_cache)
                    and hidden_cache.ndim == 3
                    and int(hidden_cache.shape[0]) == int(batch_size)
                    and int(hidden_cache.shape[1]) >= int(history_tokens)
                ):
                    hidden_hist = hidden_cache[:, : int(history_tokens), :].detach().float()
                    hidden_trace_debug["hidden_cache_accepted"] = True
                if hidden_sample is None or hidden_hist is None:
                    missing = []
                    if hidden_sample is None:
                        missing.append("hidden_current")
                    if hidden_hist is None:
                        missing.append("hidden_cache")
                    hidden_trace_debug["hidden_trace_reason"] = "missing_or_shape_mismatch:" + ",".join(missing)
                q_norm = F.normalize(q_sample, dim=-1)
                k_norm = F.normalize(k_hist, dim=-1)
                cosine = torch.matmul(q_norm, k_norm.transpose(-2, -1))
                scores = torch.matmul(q_sample, k_hist.transpose(-2, -1)) / max(float(head_dim) ** 0.5, 1.0)
                route = torch.softmax(scores, dim=-1)
                entropy = -(route.clamp_min(1e-12) * route.clamp_min(1e-12).log()).sum(dim=-1)
                entropy = entropy / torch.log(
                    torch.tensor(float(max(history_tokens, 2)), device=route.device, dtype=entropy.dtype)
                ).clamp_min(1e-12)
                transported_v = torch.matmul(route, v_hist)
                residual = (transported_v - v_sample).norm(dim=-1) / max(float(head_dim) ** 0.5, 1.0)

                topk_count = int((hmc_control or {}).get("swa_raw_transport_trace_topk", 8) or 8)
                topk_count = max(1, min(int(topk_count), int(history_tokens)))
                topk_scores, topk_indices = torch.topk(scores, k=topk_count, dim=-1)
                top1_indices = topk_indices[..., 0]

                def _unique_frac_by_head(index_tensor: torch.Tensor) -> torch.Tensor:
                    vals = index_tensor.detach().reshape(int(batch_size), int(head_count), -1)
                    out_vals: List[float] = []
                    for head_idx in range(int(head_count)):
                        head_vals = vals[:, head_idx, :].reshape(-1)
                        denom = max(int(head_vals.numel()), 1)
                        out_vals.append(float(torch.unique(head_vals).numel()) / float(denom))
                    return torch.tensor(out_vals, device=index_tensor.device, dtype=torch.float32)

                def _switch_rate_by_head(index_tensor: torch.Tensor) -> torch.Tensor:
                    vals = index_tensor.detach().reshape(int(batch_size), int(head_count), -1)
                    if int(vals.shape[-1]) < 2:
                        return torch.zeros(int(head_count), device=index_tensor.device, dtype=torch.float32)
                    switches = (vals[..., 1:] != vals[..., :-1]).float()
                    return switches.mean(dim=(0, 2)).float()

                top1_index_unique_frac = _unique_frac_by_head(top1_indices)
                top1_index_switch_rate = _switch_rate_by_head(top1_indices)
                topk_frames: Optional[torch.Tensor] = None
                top1_frame_unique_frac: Optional[torch.Tensor] = None
                top1_frame_switch_rate: Optional[torch.Tensor] = None
                top1_same_frame_frac: Optional[torch.Tensor] = None
                topk_query_frame_hit_frac: Optional[torch.Tensor] = None
                topk_same_frame_frac: Optional[torch.Tensor] = None
                top1_abs_frame_delta_mean: Optional[torch.Tensor] = None
                topk_identity_frame_missing_reason = ""
                if int(tokens_per_frame) > 0 and int(history_tokens) % int(tokens_per_frame) == 0:
                    topk_frames = torch.div(topk_indices, int(tokens_per_frame), rounding_mode="floor")
                    q_frames = torch.div(q_idx, int(tokens_per_frame), rounding_mode="floor").view(1, 1, -1)
                    q_frames = q_frames.to(device=topk_frames.device, dtype=topk_frames.dtype)
                    top1_frames = topk_frames[..., 0]
                    top1_frame_unique_frac = _unique_frac_by_head(top1_frames)
                    top1_frame_switch_rate = _switch_rate_by_head(top1_frames)
                    same_top1 = (top1_frames == q_frames).float()
                    same_topk = (topk_frames == q_frames.unsqueeze(-1)).float()
                    top1_same_frame_frac = same_top1.mean(dim=(0, 2)).float()
                    topk_query_frame_hit_frac = (same_topk.sum(dim=-1) > 0).float().mean(dim=(0, 2)).float()
                    topk_same_frame_frac = same_topk.mean(dim=(0, 2, 3)).float()
                    top1_abs_frame_delta_mean = (top1_frames.float() - q_frames.float()).abs().mean(dim=(0, 2)).float()
                else:
                    topk_identity_frame_missing_reason = (
                        "tokens_per_frame<=0 or history_tokens not divisible by tokens_per_frame"
                    )

                def _mean_by_head(x: torch.Tensor) -> List[float]:
                    vals = x.float()
                    while vals.ndim > 2:
                        vals = vals.mean(dim=-1)
                    if vals.ndim == 2:
                        vals = vals.mean(dim=0)
                    return [float(v) for v in vals.detach().cpu().reshape(-1).tolist()]

                def _scalar_mean(x: Optional[torch.Tensor]) -> Optional[float]:
                    if x is None:
                        return None
                    return float(x.detach().float().mean().item())

                def _route_mass(mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
                    if mask is None:
                        return None
                    mask_bool = mask.to(device=route.device, dtype=torch.bool)
                    if not bool(mask_bool.any()):
                        return None
                    return (route * mask_bool[:, None, None, :].float()).sum(dim=-1)

                def _topk_mask_hits(
                    mask: Optional[torch.Tensor],
                ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
                    if mask is None:
                        return None, None, None
                    mask_bool = mask.to(device=topk_indices.device, dtype=torch.bool)
                    if not bool(mask_bool.any()):
                        return None, None, None
                    expanded = mask_bool[:, None, None, :].expand(
                        int(batch_size),
                        int(head_count),
                        int(topk_indices.shape[2]),
                        int(history_tokens),
                    )
                    hits = torch.gather(expanded, -1, topk_indices.long())
                    return hits, hits.any(dim=-1).float(), hits[..., 0].float()

                def _topk_anchor_ids(
                    ids: Optional[torch.Tensor],
                    hit_mask: Optional[torch.Tensor],
                ) -> Optional[torch.Tensor]:
                    if ids is None or hit_mask is None:
                        return None
                    ids_long = ids.to(device=topk_indices.device, dtype=torch.int64)
                    expanded = ids_long[:, None, None, :].expand(
                        int(batch_size),
                        int(head_count),
                        int(topk_indices.shape[2]),
                        int(history_tokens),
                    )
                    gathered = torch.gather(expanded, -1, topk_indices.long())
                    return torch.where(hit_mask.to(dtype=torch.bool), gathered, torch.full_like(gathered, -1))

                def _current_trace_values(key: str, dtype: torch.dtype) -> Optional[torch.Tensor]:
                    raw = (hmc_control or {}).get(key)
                    if not torch.is_tensor(raw) or int(current_tokens) <= 0:
                        return None
                    vals = raw.to(device=route.device, dtype=dtype)
                    needed = int(batch_size) * int(current_tokens)
                    if int(vals.numel()) < needed:
                        return None
                    vals = vals.reshape(int(batch_size), -1)[:, : int(current_tokens)]
                    return vals

                def _topk_source_values(
                    values: Optional[torch.Tensor],
                    dtype: torch.dtype,
                    missing_value: int = -1,
                ) -> Optional[torch.Tensor]:
                    if values is None:
                        return None
                    vals = values.to(device=topk_indices.device, dtype=dtype)
                    if int(vals.numel()) < int(batch_size) * int(history_tokens):
                        return None
                    vals = vals.reshape(int(batch_size), -1)[:, : int(history_tokens)]
                    expanded = vals[:, None, None, :].expand(
                        int(batch_size),
                        int(head_count),
                        int(topk_indices.shape[2]),
                        int(history_tokens),
                    )
                    gathered = torch.gather(expanded, -1, topk_indices.long())
                    if missing_value is None:
                        return gathered
                    return torch.where(
                        torch.ones_like(gathered, dtype=torch.bool),
                        gathered,
                        torch.full_like(gathered, int(missing_value)),
                    )

                def _adjacent_frame_stability(x: torch.Tensor) -> Optional[torch.Tensor]:
                    if int(tokens_per_frame) <= 0 or history_tokens % int(tokens_per_frame) != 0:
                        return None
                    hist_frames = history_tokens // int(tokens_per_frame)
                    if hist_frames < 2:
                        return None
                    shaped = F.normalize(
                        x.detach().float().reshape(
                            int(batch_size),
                            int(x.shape[1]),
                            int(hist_frames),
                            int(tokens_per_frame),
                            int(x.shape[-1]),
                        ),
                        dim=-1,
                    )
                    cos = (shaped[:, :, 1:] * shaped[:, :, :-1]).sum(dim=-1)
                    return cos.mean(dim=(0, 2, 3))

                d_prev = Pi3._swa_trace_source_values(
                    hmc_control,
                    "D_prev_patch",
                    batch_size=batch_size,
                    history_tokens=history_tokens,
                    tokens_per_frame=tokens_per_frame,
                    device=route.device,
                    dtype=torch.float32,
                )
                g_prev = Pi3._swa_trace_source_values(
                    hmc_control,
                    "G_prev_patch",
                    batch_size=batch_size,
                    history_tokens=history_tokens,
                    tokens_per_frame=tokens_per_frame,
                    device=route.device,
                    dtype=torch.float32,
                )
                l_prev = Pi3._swa_trace_source_values(
                    hmc_control,
                    "L_prev_patch",
                    batch_size=batch_size,
                    history_tokens=history_tokens,
                    tokens_per_frame=tokens_per_frame,
                    device=route.device,
                    dtype=torch.float32,
                )
                l_current = _current_trace_values("L_sem_tok", torch.int64)
                g_current = _current_trace_values("G_sem_tok", torch.int64)
                stage_c_seed_current = _current_trace_values("stage_c_seed_global_track_idx_tok", torch.int64)
                stage_c_masklet_instance_current = _current_trace_values(
                    "stage_c_masklet_instance_idx_tok",
                    torch.int64,
                )
                sampled_query_fine_labels = (
                    l_current.index_select(1, q_idx.to(device=l_current.device, dtype=torch.long))
                    if l_current is not None and int(q_idx.numel()) > 0
                    else None
                )
                sampled_query_group_labels = (
                    g_current.index_select(1, q_idx.to(device=g_current.device, dtype=torch.long))
                    if g_current is not None and int(q_idx.numel()) > 0
                    else None
                )
                sampled_query_stage_c_seed_global_track_idx = (
                    stage_c_seed_current.index_select(
                        1,
                        q_idx.to(device=stage_c_seed_current.device, dtype=torch.long),
                    )
                    if stage_c_seed_current is not None and int(q_idx.numel()) > 0
                    else None
                )
                sampled_query_stage_c_masklet_instance_idx = (
                    stage_c_masklet_instance_current.index_select(
                        1,
                        q_idx.to(device=stage_c_masklet_instance_current.device, dtype=torch.long),
                    )
                    if stage_c_masklet_instance_current is not None and int(q_idx.numel()) > 0
                    else None
                )

                def _current_unique_hist(
                    values: Optional[torch.Tensor],
                ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], int, int]:
                    if values is None:
                        return None, None, 0, 0
                    flat = values.detach().long().reshape(-1)
                    nonnegative = flat[flat >= 0]
                    if int(nonnegative.numel()) == 0:
                        return None, None, 0, 0
                    unique, counts = torch.unique(nonnegative, sorted=True, return_counts=True)
                    return (
                        unique.detach().cpu().to(torch.int32),
                        counts.detach().cpu().to(torch.int32),
                        int(nonnegative.numel()),
                        int(unique.numel()),
                    )

                (
                    current_stage_c_seed_unique_ids,
                    current_stage_c_seed_unique_counts,
                    current_stage_c_seed_nonnegative_count,
                    current_stage_c_seed_unique_count,
                ) = _current_unique_hist(stage_c_seed_current)
                (
                    current_stage_c_masklet_instance_unique_ids,
                    current_stage_c_masklet_instance_unique_counts,
                    current_stage_c_masklet_instance_nonnegative_count,
                    current_stage_c_masklet_instance_unique_count,
                ) = _current_unique_hist(stage_c_masklet_instance_current)
                topk_cache_fine_labels = _topk_source_values(l_prev, torch.int64)
                topk_cache_group_labels = _topk_source_values(g_prev, torch.int64)
                stage_c_seed_prev = Pi3._swa_trace_source_values(
                    hmc_control,
                    "stage_c_seed_global_track_idx_prev_patch",
                    batch_size=batch_size,
                    history_tokens=history_tokens,
                    tokens_per_frame=tokens_per_frame,
                    device=route.device,
                    dtype=torch.int64,
                )
                topk_cache_stage_c_seed_global_track_idx = _topk_source_values(stage_c_seed_prev, torch.int64)
                stage_c_masklet_instance_prev = Pi3._swa_trace_source_values(
                    hmc_control,
                    "stage_c_masklet_instance_idx_prev_patch",
                    batch_size=batch_size,
                    history_tokens=history_tokens,
                    tokens_per_frame=tokens_per_frame,
                    device=route.device,
                    dtype=torch.int64,
                )
                topk_cache_stage_c_masklet_instance_idx = _topk_source_values(
                    stage_c_masklet_instance_prev,
                    torch.int64,
                )
                topk_same_fine_labels: Optional[torch.Tensor] = None
                topk_same_group_labels: Optional[torch.Tensor] = None
                topk_same_stage_c_seed_global_track_idx: Optional[torch.Tensor] = None
                topk_same_stage_c_masklet_instance_idx: Optional[torch.Tensor] = None
                if sampled_query_fine_labels is not None and topk_cache_fine_labels is not None:
                    q_fine = sampled_query_fine_labels[:, None, :, None].expand_as(topk_cache_fine_labels)
                    topk_same_fine_labels = (topk_cache_fine_labels == q_fine) & (topk_cache_fine_labels > 0)
                if sampled_query_group_labels is not None and topk_cache_group_labels is not None:
                    q_group = sampled_query_group_labels[:, None, :, None].expand_as(topk_cache_group_labels)
                    topk_same_group_labels = (topk_cache_group_labels == q_group) & (topk_cache_group_labels > 0)
                if (
                    sampled_query_stage_c_seed_global_track_idx is not None
                    and topk_cache_stage_c_seed_global_track_idx is not None
                ):
                    q_seed = sampled_query_stage_c_seed_global_track_idx[:, None, :, None].expand_as(
                        topk_cache_stage_c_seed_global_track_idx
                    )
                    topk_same_stage_c_seed_global_track_idx = (
                        topk_cache_stage_c_seed_global_track_idx == q_seed
                    ) & (topk_cache_stage_c_seed_global_track_idx >= 0)
                if (
                    sampled_query_stage_c_masklet_instance_idx is not None
                    and topk_cache_stage_c_masklet_instance_idx is not None
                ):
                    q_inst = sampled_query_stage_c_masklet_instance_idx[:, None, :, None].expand_as(
                        topk_cache_stage_c_masklet_instance_idx
                    )
                    topk_same_stage_c_masklet_instance_idx = (
                        topk_cache_stage_c_masklet_instance_idx == q_inst
                    ) & (topk_cache_stage_c_masklet_instance_idx >= 0)
                k_stable = Pi3._swa_trace_source_values(
                    hmc_control,
                    "K_stable_tok",
                    batch_size=batch_size,
                    history_tokens=history_tokens,
                    tokens_per_frame=tokens_per_frame,
                    device=route.device,
                    dtype=torch.float32,
                )
                ttt_prev_stable = Pi3._swa_trace_source_values(
                    hmc_control,
                    "prev_ttt_stable_anchor_mask_patch",
                    batch_size=batch_size,
                    history_tokens=history_tokens,
                    tokens_per_frame=tokens_per_frame,
                    device=route.device,
                    dtype=torch.float32,
                )
                ttt_prev_anchor_ids = Pi3._swa_trace_source_values(
                    hmc_control,
                    "prev_ttt_stable_anchor_id_patch",
                    batch_size=batch_size,
                    history_tokens=history_tokens,
                    tokens_per_frame=tokens_per_frame,
                    device=route.device,
                    dtype=torch.int64,
                )
                ttt_prev_retention = Pi3._swa_trace_source_values(
                    hmc_control,
                    "prev_ttt_stable_anchor_retention_patch",
                    batch_size=batch_size,
                    history_tokens=history_tokens,
                    tokens_per_frame=tokens_per_frame,
                    device=route.device,
                    dtype=torch.float32,
                )
                ttt_prev_residual = Pi3._swa_trace_source_values(
                    hmc_control,
                    "prev_ttt_stable_anchor_residual_patch",
                    batch_size=batch_size,
                    history_tokens=history_tokens,
                    tokens_per_frame=tokens_per_frame,
                    device=route.device,
                    dtype=torch.float32,
                )
                ttt_prev_z_write_key_norm = Pi3._swa_trace_source_values(
                    hmc_control,
                    "prev_ttt_stable_anchor_z_write_key_norm_patch",
                    batch_size=batch_size,
                    history_tokens=history_tokens,
                    tokens_per_frame=tokens_per_frame,
                    device=route.device,
                    dtype=torch.float32,
                )
                ttt_prev_z_write_key_sketch = Pi3._swa_trace_source_matrix(
                    hmc_control,
                    "prev_ttt_stable_anchor_z_write_key_sketch_patch",
                    batch_size=batch_size,
                    history_tokens=history_tokens,
                    tokens_per_frame=tokens_per_frame,
                    device=route.device,
                    dtype=torch.float32,
                )
                ttt_prev_z_write_key_vec = Pi3._swa_trace_source_matrix(
                    hmc_control,
                    "prev_ttt_stable_anchor_z_write_key_vec_patch",
                    batch_size=batch_size,
                    history_tokens=history_tokens,
                    tokens_per_frame=tokens_per_frame,
                    device=route.device,
                    dtype=torch.float32,
                )
                ttt_prev_z_write_hidden_vec = Pi3._swa_trace_source_matrix(
                    hmc_control,
                    "prev_ttt_stable_anchor_z_write_hidden_vec_patch",
                    batch_size=batch_size,
                    history_tokens=history_tokens,
                    tokens_per_frame=tokens_per_frame,
                    device=route.device,
                    dtype=torch.float32,
                )
                ttt_prev_tracked_instance = Pi3._swa_trace_source_values(
                    hmc_control,
                    "prev_ttt_tracked_instance_anchor_mask_patch",
                    batch_size=batch_size,
                    history_tokens=history_tokens,
                    tokens_per_frame=tokens_per_frame,
                    device=route.device,
                    dtype=torch.float32,
                )
                ttt_prev_tracked_instance_anchor_ids = Pi3._swa_trace_source_values(
                    hmc_control,
                    "prev_ttt_tracked_instance_anchor_id_patch",
                    batch_size=batch_size,
                    history_tokens=history_tokens,
                    tokens_per_frame=tokens_per_frame,
                    device=route.device,
                    dtype=torch.int64,
                )
                ttt_prev_tracked_instance_anchor_seeds = Pi3._swa_trace_source_values(
                    hmc_control,
                    "prev_ttt_tracked_instance_anchor_seed_patch",
                    batch_size=batch_size,
                    history_tokens=history_tokens,
                    tokens_per_frame=tokens_per_frame,
                    device=route.device,
                    dtype=torch.int64,
                )

                stable_d_max = float((hmc_control or {}).get("swa_raw_transport_trace_stable_d_max", 0.25) or 0.25)
                unreliable_d_min = float((hmc_control or {}).get("swa_raw_transport_trace_unreliable_d_min", 0.50) or 0.50)
                d_low_mask = None
                d_high_mask = None
                g_high_mask = None
                k_stable_mask = None
                label_static_mask = None
                stable_mask = None
                unreliable_mask = None
                stable_reason = "missing_prev_dynamic_and_label_proxy"
                unreliable_reason = "missing_prev_dynamic_proxy"
                if d_prev is not None:
                    d_low_mask = d_prev <= stable_d_max
                    d_high_mask = d_prev >= unreliable_d_min
                    stable_mask = d_low_mask
                    unreliable_mask = d_high_mask
                    stable_reason = "D_prev_low"
                    unreliable_reason = "D_prev_high"
                if k_stable is not None:
                    k_stable_mask = k_stable >= 0.5
                    stable_mask = k_stable_mask if stable_mask is None else (stable_mask & k_stable_mask)
                    stable_reason += "+K_stable_tok"
                if l_prev is not None:
                    stable_labels = (
                        set(_CONTEXT_GROUND_FINE_LABEL_IDS)
                        | set(_CONTEXT_VERTICAL_STATIC_FINE_LABEL_IDS)
                        | {15}
                    )
                    label_static_mask = torch.zeros_like(l_prev, dtype=torch.bool)
                    l_long = l_prev.long()
                    for label_id in stable_labels:
                        label_static_mask |= l_long == int(label_id)
                    stable_mask = label_static_mask if stable_mask is None else (stable_mask & label_static_mask)
                    stable_reason += "+L_prev_static_structure"
                if g_prev is not None:
                    g_high_mask = g_prev >= unreliable_d_min
                    unreliable_mask = g_high_mask if unreliable_mask is None else (unreliable_mask | g_high_mask)
                    unreliable_reason += "+G_prev_high"

                strict_stable_mask = stable_mask
                semantic_lowd_mask = None
                lowd_nonunreliable_mask = None
                stable_fallback_used = False
                stable_fallback_reason = ""
                if d_low_mask is not None and label_static_mask is not None:
                    semantic_lowd_mask = d_low_mask & label_static_mask
                if d_low_mask is not None:
                    if unreliable_mask is None:
                        lowd_nonunreliable_mask = d_low_mask
                    else:
                        lowd_nonunreliable_mask = d_low_mask & (~unreliable_mask)
                if stable_mask is not None and not bool(stable_mask.any()):
                    if semantic_lowd_mask is not None and bool(semantic_lowd_mask.any()):
                        stable_mask = semantic_lowd_mask
                        stable_fallback_used = True
                        stable_fallback_reason = "strict_empty_use_D_prev_low+L_prev_static_structure"
                    elif lowd_nonunreliable_mask is not None and bool(lowd_nonunreliable_mask.any()):
                        stable_mask = lowd_nonunreliable_mask
                        stable_fallback_used = True
                        stable_fallback_reason = "strict_empty_use_D_prev_low+not_unreliable"
                    if stable_fallback_used:
                        stable_reason += f"+fallback:{stable_fallback_reason}"

                def _mask_count(mask: Optional[torch.Tensor]) -> int:
                    if mask is None:
                        return 0
                    return int(mask.sum().item())

                stable_mass = _route_mass(stable_mask)
                unreliable_mass = _route_mass(unreliable_mask)
                ttt_prev_stable_mask = ttt_prev_stable >= 0.5 if ttt_prev_stable is not None else None
                ttt_prev_stable_mass = _route_mass(ttt_prev_stable_mask)
                ttt_prev_topk_hits, ttt_prev_query_hits, ttt_prev_top1_hits = _topk_mask_hits(
                    ttt_prev_stable_mask
                )
                ttt_prev_topk_anchor_ids = _topk_anchor_ids(ttt_prev_anchor_ids, ttt_prev_topk_hits)
                ttt_prev_tracked_instance_mask = (
                    ttt_prev_tracked_instance >= 0.5
                    if ttt_prev_tracked_instance is not None
                    else None
                )
                ttt_prev_tracked_instance_mass = _route_mass(ttt_prev_tracked_instance_mask)
                (
                    ttt_prev_tracked_instance_topk_hits,
                    ttt_prev_tracked_instance_query_hits,
                    ttt_prev_tracked_instance_top1_hits,
                ) = _topk_mask_hits(ttt_prev_tracked_instance_mask)
                ttt_prev_tracked_instance_topk_anchor_ids = _topk_anchor_ids(
                    ttt_prev_tracked_instance_anchor_ids,
                    ttt_prev_tracked_instance_topk_hits,
                )
                ttt_prev_tracked_instance_topk_anchor_seeds = _topk_anchor_ids(
                    ttt_prev_tracked_instance_anchor_seeds,
                    ttt_prev_tracked_instance_topk_hits,
                )
                ttt_prev_same_fine_topk_hits = (
                    ttt_prev_topk_hits.to(dtype=torch.bool) & topk_same_fine_labels
                    if ttt_prev_topk_hits is not None and topk_same_fine_labels is not None
                    else None
                )
                ttt_prev_same_group_topk_hits = (
                    ttt_prev_topk_hits.to(dtype=torch.bool) & topk_same_group_labels
                    if ttt_prev_topk_hits is not None and topk_same_group_labels is not None
                    else None
                )
                ttt_prev_same_stage_c_seed_topk_hits = (
                    ttt_prev_topk_hits.to(dtype=torch.bool) & topk_same_stage_c_seed_global_track_idx
                    if ttt_prev_topk_hits is not None and topk_same_stage_c_seed_global_track_idx is not None
                    else None
                )
                ttt_prev_tracked_instance_same_seed_topk_hits = (
                    ttt_prev_tracked_instance_topk_hits.to(dtype=torch.bool)
                    & topk_same_stage_c_seed_global_track_idx
                    if ttt_prev_tracked_instance_topk_hits is not None
                    and topk_same_stage_c_seed_global_track_idx is not None
                    else None
                )
                ttt_prev_tracked_instance_same_masklet_topk_hits = (
                    ttt_prev_tracked_instance_topk_hits.to(dtype=torch.bool)
                    & topk_same_stage_c_masklet_instance_idx
                    if ttt_prev_tracked_instance_topk_hits is not None
                    and topk_same_stage_c_masklet_instance_idx is not None
                    else None
                )

                def _anchor_lifecycle_rows() -> List[Dict[str, Any]]:
                    if (
                        ttt_prev_anchor_ids is None
                        or ttt_prev_stable_mask is None
                        or ttt_prev_topk_anchor_ids is None
                        or ttt_prev_topk_hits is None
                    ):
                        return []
                    ids_hist = ttt_prev_anchor_ids.to(device=topk_indices.device, dtype=torch.int64)
                    stable_bool = ttt_prev_stable_mask.to(device=topk_indices.device, dtype=torch.bool)
                    valid_topk = ttt_prev_topk_anchor_ids.to(device=topk_indices.device, dtype=torch.int64)
                    valid_topk = valid_topk[valid_topk >= 0]
                    if int(valid_topk.numel()) == 0:
                        return []
                    unique_ids, counts = torch.unique(valid_topk, return_counts=True)
                    order = torch.argsort(counts, descending=True)
                    max_rows = int((hmc_control or {}).get("swa_raw_transport_trace_anchor_lifecycle_max_ids", 128) or 128)
                    max_rows = max(1, min(max_rows, int(unique_ids.numel())))
                    topk_route = torch.gather(route, -1, topk_indices.long())
                    topk_cosine = torch.gather(cosine, -1, topk_indices.long()).float()
                    topk_norm_l2 = torch.sqrt(torch.clamp(2.0 - 2.0 * topk_cosine, min=0.0))
                    v_norm = F.normalize(v_sample, dim=-1)
                    v_hist_norm = F.normalize(v_hist, dim=-1)
                    v_cosine = torch.matmul(v_norm, v_hist_norm.transpose(-2, -1))
                    topk_v_cosine = torch.gather(v_cosine, -1, topk_indices.long()).float()
                    topk_v_norm_l2 = torch.sqrt(torch.clamp(2.0 - 2.0 * topk_v_cosine, min=0.0))
                    gather_index = topk_indices.long().unsqueeze(-1).expand(
                        int(batch_size),
                        int(head_count),
                        int(topk_indices.shape[2]),
                        int(topk_indices.shape[3]),
                        int(head_dim),
                    )
                    value_dim = int(v_hist.shape[-1])
                    v_gather_index = topk_indices.long().unsqueeze(-1).expand(
                        int(batch_size),
                        int(head_count),
                        int(topk_indices.shape[2]),
                        int(topk_indices.shape[3]),
                        int(value_dim),
                    )
                    k_topk_norm = torch.gather(
                        k_norm[:, :, None, :, :].expand(
                            int(batch_size),
                            int(head_count),
                            int(topk_indices.shape[2]),
                            int(history_tokens),
                            int(value_dim),
                        ),
                        3,
                        v_gather_index,
                    )
                    v_topk_norm = torch.gather(
                        v_hist_norm[:, :, None, :, :].expand(
                            int(batch_size),
                            int(head_count),
                            int(topk_indices.shape[2]),
                            int(history_tokens),
                            int(head_dim),
                        ),
                        3,
                        gather_index,
                    )
                    q_topk_norm = q_norm.unsqueeze(-2).expand_as(k_topk_norm)
                    k_sample_norm = F.normalize(k_sample, dim=-1)
                    k_current_topk_norm = k_sample_norm.unsqueeze(-2).expand_as(k_topk_norm)
                    v_current_topk_norm = v_norm.unsqueeze(-2).expand_as(v_topk_norm)
                    hidden_hist_norm: Optional[torch.Tensor] = None
                    hidden_sample_norm: Optional[torch.Tensor] = None
                    hidden_topk_norm: Optional[torch.Tensor] = None
                    hidden_current_topk_norm: Optional[torch.Tensor] = None
                    if hidden_hist is not None and hidden_sample is not None:
                        hidden_hist_norm = F.normalize(hidden_hist, dim=-1)
                        hidden_sample_norm = F.normalize(hidden_sample, dim=-1)
                        hidden_dim = int(hidden_hist_norm.shape[-1])
                        hidden_gather_index = topk_indices.long().unsqueeze(-1).expand(
                            int(batch_size),
                            int(head_count),
                            int(topk_indices.shape[2]),
                            int(topk_indices.shape[3]),
                            hidden_dim,
                        )
                        hidden_topk_norm = torch.gather(
                            hidden_hist_norm[:, None, None, :, :].expand(
                                int(batch_size),
                                int(head_count),
                                int(topk_indices.shape[2]),
                                int(history_tokens),
                                hidden_dim,
                            ),
                            3,
                            hidden_gather_index,
                        )
                        hidden_current_topk_norm = hidden_sample_norm[:, None, :, None, :].expand_as(
                            hidden_topk_norm
                        )
                    z_sketch_dim = int((hmc_control or {}).get("swa_raw_transport_trace_z_sketch_dim", 16) or 16)
                    z_sketch_dim = max(1, min(int(z_sketch_dim), 32))

                    def _chunk_sketch(vec: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
                        if vec is None or int(vec.numel()) == 0:
                            return None
                        flat_vec = vec.detach().float().reshape(-1)
                        dim = min(int(z_sketch_dim), int(flat_vec.numel()))
                        if dim <= 0:
                            return None
                        return torch.stack([part.mean() for part in torch.chunk(flat_vec, chunks=dim, dim=0)], dim=0)

                    def _project_vec(vec: Optional[torch.Tensor], target_dim: int) -> Optional[torch.Tensor]:
                        if vec is None or int(vec.numel()) == 0 or int(target_dim) <= 0:
                            return None
                        flat_vec = vec.detach().float().reshape(-1)
                        if int(flat_vec.numel()) == int(target_dim):
                            return flat_vec
                        chunks = torch.chunk(flat_vec, chunks=min(int(target_dim), int(flat_vec.numel())), dim=0)
                        projected = torch.stack([part.mean() for part in chunks], dim=0)
                        if int(projected.numel()) != int(target_dim):
                            return None
                        return projected

                    def _weighted_feature_mean(
                        values: torch.Tensor,
                        weights: torch.Tensor,
                    ) -> Optional[torch.Tensor]:
                        if int(values.numel()) == 0 or int(weights.numel()) == 0:
                            return None
                        vals = values.detach().float().reshape(-1, int(values.shape[-1]))
                        w = weights.detach().float().reshape(-1, 1)
                        denom = w.sum().clamp_min(1.0e-12)
                        return (vals * w).sum(dim=0) / denom

                    def _vec_list(vec: Optional[torch.Tensor]) -> Optional[List[float]]:
                        if vec is None:
                            return None
                        return [float(v) for v in vec.detach().float().cpu().reshape(-1).tolist()]

                    def _cosine_residual(
                        lhs: Optional[torch.Tensor],
                        rhs: Optional[torch.Tensor],
                    ) -> Optional[float]:
                        if lhs is None or rhs is None or int(lhs.numel()) != int(rhs.numel()):
                            return None
                        lhs_f = lhs.detach().float().reshape(-1)
                        rhs_f = rhs.detach().float().reshape(-1)
                        denom = lhs_f.norm().mul(rhs_f.norm()).clamp_min(1.0e-12)
                        cos_val = torch.dot(lhs_f, rhs_f).div(denom).clamp(-1.0, 1.0)
                        return float((1.0 - cos_val).item())
                    rows_out: List[Dict[str, Any]] = []
                    denom_qh = max(int(batch_size) * int(head_count) * int(topk_indices.shape[2]), 1)
                    for idx in order[:max_rows]:
                        anchor_id = int(unique_ids[idx].item())
                        hist_mask = stable_bool & (ids_hist == int(anchor_id))
                        hit_pos = ttt_prev_topk_anchor_ids.to(device=topk_indices.device, dtype=torch.int64) == int(anchor_id)
                        qh_hit = hit_pos.any(dim=-1)
                        route_mass_for_anchor = torch.where(
                            hit_pos,
                            topk_route,
                            torch.zeros_like(topk_route),
                        ).sum(dim=-1)
                        retention_mean: Optional[float] = None
                        if ttt_prev_retention is not None and bool(hist_mask.any()):
                            retention_mean = float(
                                ttt_prev_retention.to(device=hist_mask.device, dtype=torch.float32)[hist_mask]
                                .detach()
                                .float()
                                .mean()
                                .item()
                            )
                        source_residual_mean: Optional[float] = None
                        if ttt_prev_residual is not None and bool(hist_mask.any()):
                            source_residual_mean = float(
                                ttt_prev_residual.to(device=hist_mask.device, dtype=torch.float32)[hist_mask]
                                .detach()
                                .float()
                                .mean()
                                .item()
                            )
                        source_label_mode: Optional[int] = None
                        source_label_mode_frac: Optional[float] = None
                        if l_prev is not None and bool(hist_mask.any()):
                            label_vals = l_prev.to(device=hist_mask.device, dtype=torch.int64)[hist_mask]
                            if int(label_vals.numel()) > 0:
                                label_ids, label_counts = torch.unique(label_vals, return_counts=True)
                                label_best = torch.argmax(label_counts)
                                source_label_mode = int(label_ids[label_best].item())
                                source_label_mode_frac = float(
                                    label_counts[label_best].float().div(label_counts.sum().clamp_min(1)).item()
                                )
                        source_stage_c_seed_global_track_idx_mode: Optional[int] = None
                        source_stage_c_seed_global_track_idx_mode_frac: Optional[float] = None
                        if stage_c_seed_prev is not None and bool(hist_mask.any()):
                            seed_vals = stage_c_seed_prev.to(device=hist_mask.device, dtype=torch.int64)[hist_mask]
                            seed_vals = seed_vals[seed_vals >= 0]
                            if int(seed_vals.numel()) > 0:
                                seed_ids, seed_counts = torch.unique(seed_vals, return_counts=True)
                                seed_best = torch.argmax(seed_counts)
                                source_stage_c_seed_global_track_idx_mode = int(seed_ids[seed_best].item())
                                source_stage_c_seed_global_track_idx_mode_frac = float(
                                    seed_counts[seed_best].float().div(seed_counts.sum().clamp_min(1)).item()
                                )
                        current_residual_mean: Optional[float] = None
                        if bool(qh_hit.any()):
                            current_residual_mean = float(residual[qh_hit].detach().float().mean().item())
                        z_write_key_norm_mean: Optional[float] = None
                        z_write_key_sketch_norm_mean: Optional[float] = None
                        z_write_key_sketch_abs_mean: Optional[float] = None
                        z_write_key_sketch_mean: Optional[List[float]] = None
                        z_write_hidden_vec_mean: Optional[List[float]] = None
                        z_write_key_sketch_dim = 0
                        z_write_key_vec_mean: Optional[List[float]] = None
                        z_write_key_vec_projected_mean: Optional[List[float]] = None
                        z_write_key_vec_dim = 0
                        z_write_key_vec_projected_dim = 0
                        z_write_key_vec: Optional[torch.Tensor] = None
                        z_write_key_vec_projected: Optional[torch.Tensor] = None
                        z_write_hidden_vec: Optional[torch.Tensor] = None
                        if ttt_prev_z_write_key_norm is not None and bool(hist_mask.any()):
                            z_write_key_norm_mean = float(
                                ttt_prev_z_write_key_norm.to(device=hist_mask.device, dtype=torch.float32)[hist_mask]
                                .detach()
                                .float()
                                .mean()
                                .item()
                            )
                        if ttt_prev_z_write_key_sketch is not None and bool(hist_mask.any()):
                            sketch_vals = ttt_prev_z_write_key_sketch.to(device=hist_mask.device, dtype=torch.float32)[hist_mask]
                            if int(sketch_vals.numel()) > 0 and sketch_vals.ndim == 2:
                                z_write_key_sketch_dim = int(sketch_vals.shape[-1])
                                z_write_key_sketch_vec = sketch_vals.detach().float().mean(dim=0)
                                z_write_key_sketch_mean = _vec_list(z_write_key_sketch_vec)
                                z_write_key_sketch_norm_mean = float(
                                    sketch_vals.detach().float().norm(dim=-1).mean().item()
                                )
                                z_write_key_sketch_abs_mean = float(
                                    sketch_vals.detach().float().abs().mean().item()
                                )
                        if ttt_prev_z_write_key_vec is not None and bool(hist_mask.any()):
                            vec_vals = ttt_prev_z_write_key_vec.to(device=hist_mask.device, dtype=torch.float32)[hist_mask]
                            if int(vec_vals.numel()) > 0 and vec_vals.ndim == 2:
                                z_write_key_vec_dim = int(vec_vals.shape[-1])
                                z_write_key_vec = F.normalize(vec_vals.detach().float().mean(dim=0), dim=0)
                                z_write_key_vec_mean = _vec_list(z_write_key_vec)
                        if ttt_prev_z_write_hidden_vec is not None and bool(hist_mask.any()):
                            hidden_vals = ttt_prev_z_write_hidden_vec.to(
                                device=hist_mask.device,
                                dtype=torch.float32,
                            )[hist_mask]
                            if int(hidden_vals.numel()) > 0 and hidden_vals.ndim == 2:
                                z_write_hidden_vec = F.normalize(
                                    hidden_vals.detach().float().mean(dim=0),
                                    dim=0,
                                )
                                z_write_hidden_vec_mean = _vec_list(z_write_hidden_vec)
                        z_cache_current_cos_mean: Optional[float] = None
                        z_cache_current_cos_route_weighted_mean: Optional[float] = None
                        z_cache_current_l2_mean: Optional[float] = None
                        z_cache_current_l2_route_weighted_mean: Optional[float] = None
                        z_cache_current_v_cos_mean: Optional[float] = None
                        z_cache_current_v_cos_route_weighted_mean: Optional[float] = None
                        z_cache_current_v_l2_mean: Optional[float] = None
                        z_cache_current_v_l2_route_weighted_mean: Optional[float] = None
                        z_current_q_sketch_mean: Optional[List[float]] = None
                        z_cache_k_sketch_mean: Optional[List[float]] = None
                        z_current_v_sketch_mean: Optional[List[float]] = None
                        z_cache_v_sketch_mean: Optional[List[float]] = None
                        z_current_q_vec_mean: Optional[List[float]] = None
                        z_current_k_vec_mean: Optional[List[float]] = None
                        z_cache_k_vec_mean: Optional[List[float]] = None
                        z_ref_cache_k_vec_mean: Optional[List[float]] = None
                        z_current_v_vec_mean: Optional[List[float]] = None
                        z_cache_v_vec_mean: Optional[List[float]] = None
                        z_ref_cache_v_vec_mean: Optional[List[float]] = None
                        z_ref_hidden_vec_mean: Optional[List[float]] = None
                        z_cache_hidden_vec_mean: Optional[List[float]] = None
                        z_current_hidden_vec_mean: Optional[List[float]] = None
                        z_cache_current_k_sketch_residual: Optional[float] = None
                        z_cache_current_v_sketch_residual: Optional[float] = None
                        z_write_current_q_sketch_residual: Optional[float] = None
                        z_write_cache_k_sketch_residual: Optional[float] = None
                        z_cache_current_k_vec_residual: Optional[float] = None
                        z_cache_current_k_native_vec_residual: Optional[float] = None
                        z_ref_current_k_native_vec_residual: Optional[float] = None
                        z_ref_cache_k_vec_residual: Optional[float] = None
                        z_cache_current_v_vec_residual: Optional[float] = None
                        z_ref_current_v_vec_residual: Optional[float] = None
                        z_ref_cache_v_vec_residual: Optional[float] = None
                        z_write_cache_hidden_vec_residual: Optional[float] = None
                        z_write_current_hidden_vec_residual: Optional[float] = None
                        z_ref_current_hidden_vec_residual: Optional[float] = None
                        z_ref_cache_hidden_vec_residual: Optional[float] = None
                        z_write_current_q_vec_residual: Optional[float] = None
                        z_write_current_k_vec_residual: Optional[float] = None
                        z_write_cache_k_vec_residual: Optional[float] = None
                        z_write_current_q_vec_projected_residual: Optional[float] = None
                        z_write_current_k_vec_projected_residual: Optional[float] = None
                        z_write_cache_k_vec_projected_residual: Optional[float] = None
                        z_write_ref_cache_k_vec_projected_residual: Optional[float] = None
                        k_ref_vec: Optional[torch.Tensor] = None
                        v_ref_vec: Optional[torch.Tensor] = None
                        hidden_ref_vec: Optional[torch.Tensor] = None
                        if bool(hist_mask.any()):
                            hist_head_mask = hist_mask[:, None, :].expand(
                                int(batch_size),
                                int(head_count),
                                int(history_tokens),
                            )
                            if bool(hist_head_mask.any()):
                                k_ref_values = k_norm[hist_head_mask]
                                v_ref_values = v_hist_norm[hist_head_mask]
                                ref_weights = torch.ones(
                                    int(k_ref_values.reshape(-1, int(k_ref_values.shape[-1])).shape[0]),
                                    device=k_ref_values.device,
                                    dtype=torch.float32,
                                )
                                k_ref_vec = _weighted_feature_mean(k_ref_values, ref_weights)
                                v_ref_vec = _weighted_feature_mean(v_ref_values, ref_weights)
                                z_ref_cache_k_vec_mean = _vec_list(k_ref_vec)
                                z_ref_cache_v_vec_mean = _vec_list(v_ref_vec)
                            if hidden_hist_norm is not None:
                                hidden_ref_values = hidden_hist_norm[hist_mask]
                                hidden_ref_weights = torch.ones(
                                    int(hidden_ref_values.reshape(-1, int(hidden_ref_values.shape[-1])).shape[0]),
                                    device=hidden_ref_values.device,
                                    dtype=torch.float32,
                                )
                                hidden_ref_vec = _weighted_feature_mean(hidden_ref_values, hidden_ref_weights)
                                z_ref_hidden_vec_mean = _vec_list(hidden_ref_vec)
                        if bool(hit_pos.any()):
                            cos_hits = topk_cosine[hit_pos].detach().float()
                            l2_hits = topk_norm_l2[hit_pos].detach().float()
                            v_cos_hits = topk_v_cosine[hit_pos].detach().float()
                            v_l2_hits = topk_v_norm_l2[hit_pos].detach().float()
                            route_hits = topk_route[hit_pos].detach().float()
                            z_cache_current_cos_mean = float(cos_hits.mean().item())
                            z_cache_current_l2_mean = float(l2_hits.mean().item())
                            z_cache_current_v_cos_mean = float(v_cos_hits.mean().item())
                            z_cache_current_v_l2_mean = float(v_l2_hits.mean().item())
                            route_denom = route_hits.sum().clamp_min(1.0e-12)
                            z_cache_current_cos_route_weighted_mean = float((cos_hits * route_hits).sum().div(route_denom).item())
                            z_cache_current_l2_route_weighted_mean = float((l2_hits * route_hits).sum().div(route_denom).item())
                            z_cache_current_v_cos_route_weighted_mean = float((v_cos_hits * route_hits).sum().div(route_denom).item())
                            z_cache_current_v_l2_route_weighted_mean = float((v_l2_hits * route_hits).sum().div(route_denom).item())
                            q_vec = _weighted_feature_mean(q_topk_norm[hit_pos], route_hits)
                            k_current_vec = _weighted_feature_mean(k_current_topk_norm[hit_pos], route_hits)
                            k_vec = _weighted_feature_mean(k_topk_norm[hit_pos], route_hits)
                            v_current_vec = _weighted_feature_mean(v_current_topk_norm[hit_pos], route_hits)
                            v_cache_vec = _weighted_feature_mean(v_topk_norm[hit_pos], route_hits)
                            hidden_current_vec: Optional[torch.Tensor] = None
                            hidden_cache_vec: Optional[torch.Tensor] = None
                            if hidden_topk_norm is not None and hidden_current_topk_norm is not None:
                                hidden_cache_vec = _weighted_feature_mean(hidden_topk_norm[hit_pos], route_hits)
                                hidden_current_vec = _weighted_feature_mean(
                                    hidden_current_topk_norm[hit_pos],
                                    route_hits,
                                )
                            z_current_q_vec_mean = _vec_list(q_vec)
                            z_current_k_vec_mean = _vec_list(k_current_vec)
                            z_cache_k_vec_mean = _vec_list(k_vec)
                            z_current_v_vec_mean = _vec_list(v_current_vec)
                            z_cache_v_vec_mean = _vec_list(v_cache_vec)
                            z_cache_hidden_vec_mean = _vec_list(hidden_cache_vec)
                            z_current_hidden_vec_mean = _vec_list(hidden_current_vec)
                            z_cache_current_k_vec_residual = _cosine_residual(k_vec, q_vec)
                            z_cache_current_k_native_vec_residual = _cosine_residual(k_vec, k_current_vec)
                            z_ref_current_k_native_vec_residual = _cosine_residual(k_ref_vec, k_current_vec)
                            z_ref_cache_k_vec_residual = _cosine_residual(k_ref_vec, k_vec)
                            z_cache_current_v_vec_residual = _cosine_residual(v_cache_vec, v_current_vec)
                            z_ref_current_v_vec_residual = _cosine_residual(v_ref_vec, v_current_vec)
                            z_ref_cache_v_vec_residual = _cosine_residual(v_ref_vec, v_cache_vec)
                            z_write_cache_hidden_vec_residual = _cosine_residual(z_write_hidden_vec, hidden_cache_vec)
                            z_write_current_hidden_vec_residual = _cosine_residual(
                                z_write_hidden_vec,
                                hidden_current_vec,
                            )
                            z_ref_current_hidden_vec_residual = _cosine_residual(hidden_ref_vec, hidden_current_vec)
                            z_ref_cache_hidden_vec_residual = _cosine_residual(hidden_ref_vec, hidden_cache_vec)
                            if q_vec is not None and z_write_key_vec is not None:
                                z_write_key_vec_projected = _project_vec(z_write_key_vec, int(q_vec.numel()))
                                if z_write_key_vec_projected is not None:
                                    z_write_key_vec_projected_dim = int(z_write_key_vec_projected.numel())
                                    z_write_key_vec_projected_mean = _vec_list(z_write_key_vec_projected)
                            q_sketch = _chunk_sketch(q_vec)
                            k_sketch = _chunk_sketch(k_vec)
                            v_current_sketch = _chunk_sketch(v_current_vec)
                            v_cache_sketch = _chunk_sketch(v_cache_vec)
                            z_current_q_sketch_mean = _vec_list(q_sketch)
                            z_cache_k_sketch_mean = _vec_list(k_sketch)
                            z_current_v_sketch_mean = _vec_list(v_current_sketch)
                            z_cache_v_sketch_mean = _vec_list(v_cache_sketch)
                            z_cache_current_k_sketch_residual = _cosine_residual(k_sketch, q_sketch)
                            z_cache_current_v_sketch_residual = _cosine_residual(v_cache_sketch, v_current_sketch)
                            if z_write_key_sketch_mean is not None:
                                z_write_sketch_tensor = torch.tensor(
                                    z_write_key_sketch_mean,
                                    device=route.device,
                                    dtype=torch.float32,
                                )
                                z_write_current_q_sketch_residual = _cosine_residual(
                                    z_write_sketch_tensor,
                                    q_sketch,
                                )
                                z_write_cache_k_sketch_residual = _cosine_residual(
                                    z_write_sketch_tensor,
                                    k_sketch,
                                )
                            if z_write_key_vec is not None:
                                z_write_current_q_vec_residual = _cosine_residual(z_write_key_vec, q_vec)
                                z_write_current_k_vec_residual = _cosine_residual(z_write_key_vec, k_current_vec)
                                z_write_cache_k_vec_residual = _cosine_residual(z_write_key_vec, k_vec)
                            if z_write_key_vec_projected is not None:
                                z_write_current_q_vec_projected_residual = _cosine_residual(
                                    z_write_key_vec_projected,
                                    q_vec,
                                )
                                z_write_current_k_vec_projected_residual = _cosine_residual(
                                    z_write_key_vec_projected,
                                    k_current_vec,
                                )
                                z_write_cache_k_vec_projected_residual = _cosine_residual(
                                    z_write_key_vec_projected,
                                    k_vec,
                                )
                                z_write_ref_cache_k_vec_projected_residual = _cosine_residual(
                                    z_write_key_vec_projected,
                                    k_ref_vec,
                                )
                        qh_by_head = qh_hit.float().mean(dim=(0, 2)) if qh_hit.ndim == 3 else torch.empty(0, device=route.device)
                        rows_out.append(
                            {
                                "anchor_id": int(anchor_id),
                                "source_token_count": int(hist_mask.sum().item()),
                                "topk_hit_position_count": int(hit_pos.sum().item()),
                                "query_head_hit_frac": float(qh_hit.float().sum().item() / float(denom_qh)),
                                "query_head_hit_max": float(qh_by_head.max().item()) if int(qh_by_head.numel()) else None,
                                "query_head_ge50_frac": (
                                    float((qh_by_head >= 0.50).float().mean().item())
                                    if int(qh_by_head.numel()) else None
                                ),
                                "query_head_ge75_frac": (
                                    float((qh_by_head >= 0.75).float().mean().item())
                                    if int(qh_by_head.numel()) else None
                                ),
                                "topk_route_mass_mean": float(route_mass_for_anchor.detach().float().mean().item()),
                                "topk_route_mass_max": float(route_mass_for_anchor.detach().float().max().item()),
                                "source_retention_mean": retention_mean,
                                "source_residual_mean": source_residual_mean,
                                "source_label_mode": source_label_mode,
                                "source_label_mode_frac": source_label_mode_frac,
                                "source_stage_c_seed_global_track_idx_mode": (
                                    source_stage_c_seed_global_track_idx_mode
                                ),
                                "source_stage_c_seed_global_track_idx_mode_frac": (
                                    source_stage_c_seed_global_track_idx_mode_frac
                                ),
                                "current_feature_residual_mean": current_residual_mean,
                                "z_write_key_norm_mean": z_write_key_norm_mean,
                                "z_write_key_sketch_norm_mean": z_write_key_sketch_norm_mean,
                                "z_write_key_sketch_abs_mean": z_write_key_sketch_abs_mean,
                                "z_write_key_sketch_dim": int(z_write_key_sketch_dim),
                                "z_write_key_sketch_mean": z_write_key_sketch_mean,
                                "z_write_key_sketch_source": "ttt_pre_zp_replay_key_mean_normalized_chunk_mean",
                                "z_write_hidden_vec_mean": z_write_hidden_vec_mean,
                                "z_write_hidden_vec_source": "ttt_write_tokens_out_hidden_mean_normalized",
                                "z_write_key_vec_dim": int(z_write_key_vec_dim),
                                "z_write_key_vec_mean": z_write_key_vec_mean,
                                "z_write_key_vec_source": "ttt_pre_zp_replay_key_mean_normalized_full",
                                "z_write_key_vec_projected_dim": int(z_write_key_vec_projected_dim),
                                "z_write_key_vec_projected_mean": z_write_key_vec_projected_mean,
                                "z_write_key_vec_projected_source": "chunk_mean_projection_to_swa_head_dim",
                                "z_sketch_dim": int(z_sketch_dim),
                                "z_current_q_sketch_mean": z_current_q_sketch_mean,
                                "z_cache_k_sketch_mean": z_cache_k_sketch_mean,
                                "z_current_v_sketch_mean": z_current_v_sketch_mean,
                                "z_cache_v_sketch_mean": z_cache_v_sketch_mean,
                                "z_current_q_vec_mean": z_current_q_vec_mean,
                                "z_current_k_vec_mean": z_current_k_vec_mean,
                                "z_cache_k_vec_mean": z_cache_k_vec_mean,
                                "z_ref_cache_k_vec_mean": z_ref_cache_k_vec_mean,
                                "z_current_v_vec_mean": z_current_v_vec_mean,
                                "z_cache_v_vec_mean": z_cache_v_vec_mean,
                                "z_ref_cache_v_vec_mean": z_ref_cache_v_vec_mean,
                                "z_ref_hidden_vec_mean": z_ref_hidden_vec_mean,
                                "z_cache_hidden_vec_mean": z_cache_hidden_vec_mean,
                                "z_current_hidden_vec_mean": z_current_hidden_vec_mean,
                                "z_cache_current_k_sketch_residual": z_cache_current_k_sketch_residual,
                                "z_cache_current_v_sketch_residual": z_cache_current_v_sketch_residual,
                                "z_write_current_q_sketch_residual": z_write_current_q_sketch_residual,
                                "z_write_cache_k_sketch_residual": z_write_cache_k_sketch_residual,
                                "z_cache_current_k_vec_residual": z_cache_current_k_vec_residual,
                                "z_cache_current_k_native_vec_residual": z_cache_current_k_native_vec_residual,
                                "z_ref_current_k_native_vec_residual": z_ref_current_k_native_vec_residual,
                                "z_ref_cache_k_vec_residual": z_ref_cache_k_vec_residual,
                                "z_cache_current_v_vec_residual": z_cache_current_v_vec_residual,
                                "z_ref_current_v_vec_residual": z_ref_current_v_vec_residual,
                                "z_ref_cache_v_vec_residual": z_ref_cache_v_vec_residual,
                                "z_write_cache_hidden_vec_residual": z_write_cache_hidden_vec_residual,
                                "z_write_current_hidden_vec_residual": z_write_current_hidden_vec_residual,
                                "z_ref_current_hidden_vec_residual": z_ref_current_hidden_vec_residual,
                                "z_ref_cache_hidden_vec_residual": z_ref_cache_hidden_vec_residual,
                                "z_write_current_q_vec_residual": z_write_current_q_vec_residual,
                                "z_write_current_k_vec_residual": z_write_current_k_vec_residual,
                                "z_write_cache_k_vec_residual": z_write_cache_k_vec_residual,
                                "z_write_current_q_vec_projected_residual": z_write_current_q_vec_projected_residual,
                                "z_write_current_k_vec_projected_residual": z_write_current_k_vec_projected_residual,
                                "z_write_cache_k_vec_projected_residual": z_write_cache_k_vec_projected_residual,
                                "z_write_ref_cache_k_vec_projected_residual": z_write_ref_cache_k_vec_projected_residual,
                                "z_cache_current_pair_count": int(hit_pos.sum().item()),
                                "z_cache_current_cos_mean": z_cache_current_cos_mean,
                                "z_cache_current_cos_route_weighted_mean": z_cache_current_cos_route_weighted_mean,
                                "z_cache_current_l2_mean": z_cache_current_l2_mean,
                                "z_cache_current_l2_route_weighted_mean": z_cache_current_l2_route_weighted_mean,
                                "z_cache_current_v_cos_mean": z_cache_current_v_cos_mean,
                                "z_cache_current_v_cos_route_weighted_mean": z_cache_current_v_cos_route_weighted_mean,
                                "z_cache_current_v_l2_mean": z_cache_current_v_l2_mean,
                                "z_cache_current_v_l2_route_weighted_mean": z_cache_current_v_l2_route_weighted_mean,
                                "source_chunk_idx": int((hmc_control or {}).get("prev_ttt_stable_anchor_source_chunk_idx", -1) or -1),
                                "current_chunk_idx": int((hmc_control or {}).get("semantic_action_chunk_idx", -1) or -1),
                            }
                        )
                    return rows_out

                def _tracked_instance_anchor_lifecycle_rows() -> List[Dict[str, Any]]:
                    if (
                        ttt_prev_tracked_instance_anchor_ids is None
                        or ttt_prev_tracked_instance_anchor_seeds is None
                        or ttt_prev_tracked_instance_mask is None
                        or ttt_prev_tracked_instance_topk_anchor_ids is None
                        or ttt_prev_tracked_instance_topk_hits is None
                    ):
                        return []
                    ids_hist = ttt_prev_tracked_instance_anchor_ids.to(
                        device=topk_indices.device,
                        dtype=torch.int64,
                    )
                    seed_hist = ttt_prev_tracked_instance_anchor_seeds.to(
                        device=topk_indices.device,
                        dtype=torch.int64,
                    )
                    tracked_bool = ttt_prev_tracked_instance_mask.to(
                        device=topk_indices.device,
                        dtype=torch.bool,
                    )
                    topk_ids = ttt_prev_tracked_instance_topk_anchor_ids.to(
                        device=topk_indices.device,
                        dtype=torch.int64,
                    )
                    valid_topk = topk_ids[topk_ids >= 0]
                    if int(valid_topk.numel()) == 0:
                        return []
                    unique_ids, counts = torch.unique(valid_topk, return_counts=True)
                    order = torch.argsort(counts, descending=True)
                    max_rows = int(
                        (hmc_control or {}).get(
                            "swa_raw_transport_trace_tracked_instance_lifecycle_max_ids",
                            (hmc_control or {}).get("swa_raw_transport_trace_anchor_lifecycle_max_ids", 128),
                        )
                        or 128
                    )
                    max_rows = max(1, min(max_rows, int(unique_ids.numel())))
                    topk_route = torch.gather(route, -1, topk_indices.long())
                    denom_qh = max(int(batch_size) * int(head_count) * int(topk_indices.shape[2]), 1)
                    rows_out: List[Dict[str, Any]] = []
                    for idx in order[:max_rows]:
                        anchor_id = int(unique_ids[idx].item())
                        hist_mask = tracked_bool & (ids_hist == int(anchor_id))
                        hit_pos = topk_ids == int(anchor_id)
                        qh_hit = hit_pos.any(dim=-1)
                        route_mass_for_anchor = torch.where(
                            hit_pos,
                            topk_route,
                            torch.zeros_like(topk_route),
                        ).sum(dim=-1)
                        source_seed_mode: Optional[int] = None
                        source_seed_mode_frac: Optional[float] = None
                        if bool(hist_mask.any()):
                            seed_vals = seed_hist[hist_mask]
                            seed_vals = seed_vals[seed_vals >= 0]
                            if int(seed_vals.numel()) > 0:
                                seed_ids, seed_counts = torch.unique(seed_vals, return_counts=True)
                                seed_best = torch.argmax(seed_counts)
                                source_seed_mode = int(seed_ids[seed_best].item())
                                source_seed_mode_frac = float(
                                    seed_counts[seed_best].float().div(seed_counts.sum().clamp_min(1)).item()
                                )
                        same_seed_count = 0
                        if ttt_prev_tracked_instance_same_seed_topk_hits is not None:
                            same_seed_count = int(
                                (
                                    hit_pos
                                    & ttt_prev_tracked_instance_same_seed_topk_hits.to(
                                        device=hit_pos.device,
                                        dtype=torch.bool,
                                    )
                                ).sum().item()
                            )
                        same_masklet_count = 0
                        if ttt_prev_tracked_instance_same_masklet_topk_hits is not None:
                            same_masklet_count = int(
                                (
                                    hit_pos
                                    & ttt_prev_tracked_instance_same_masklet_topk_hits.to(
                                        device=hit_pos.device,
                                        dtype=torch.bool,
                                    )
                                ).sum().item()
                            )
                        qh_by_head = (
                            qh_hit.float().mean(dim=(0, 2))
                            if qh_hit.ndim == 3
                            else torch.empty(0, device=route.device)
                        )
                        rows_out.append(
                            {
                                "anchor_id": int(anchor_id),
                                "source_type": "thing_tracked",
                                "source_token_count": int(hist_mask.sum().item()),
                                "topk_hit_position_count": int(hit_pos.sum().item()),
                                "topk_same_seed_position_count": same_seed_count,
                                "topk_same_masklet_position_count": same_masklet_count,
                                "query_head_hit_frac": float(qh_hit.float().sum().item() / float(denom_qh)),
                                "query_head_hit_max": float(qh_by_head.max().item()) if int(qh_by_head.numel()) else None,
                                "query_head_ge50_frac": (
                                    float((qh_by_head >= 0.50).float().mean().item())
                                    if int(qh_by_head.numel()) else None
                                ),
                                "query_head_ge75_frac": (
                                    float((qh_by_head >= 0.75).float().mean().item())
                                    if int(qh_by_head.numel()) else None
                                ),
                                "topk_route_mass_mean": float(route_mass_for_anchor.detach().float().mean().item()),
                                "topk_route_mass_max": float(route_mass_for_anchor.detach().float().max().item()),
                                "source_stage_c_seed_global_track_idx_mode": source_seed_mode,
                                "source_stage_c_seed_global_track_idx_mode_frac": source_seed_mode_frac,
                                "source_chunk_idx": int(
                                    (hmc_control or {}).get(
                                        "prev_ttt_tracked_instance_anchor_source_chunk_idx",
                                        -1,
                                    )
                                    or -1
                                ),
                                "current_chunk_idx": int(
                                    (hmc_control or {}).get("semantic_action_chunk_idx", -1) or -1
                                ),
                                "runtime_action_allowed": False,
                                "claim_level": "diagnostic_tracked_instance_anchor_lifecycle_no_runtime",
                            }
                        )
                    return rows_out

                ttt_prev_anchor_lifecycle_rows = _anchor_lifecycle_rows()
                ttt_prev_tracked_instance_lifecycle_rows = _tracked_instance_anchor_lifecycle_rows()
                stable_random_mass = None
                unreliable_random_mass = None
                if stable_mask is not None and bool(stable_mask.any()):
                    stable_random_mass = _route_mass(
                        Pi3._swa_trace_same_count_control(stable_mask, salt=17031.0 + float(swa_layer_idx))
                    )
                if unreliable_mask is not None and bool(unreliable_mask.any()):
                    unreliable_random_mass = _route_mass(
                        Pi3._swa_trace_same_count_control(unreliable_mask, salt=27031.0 + float(swa_layer_idx))
                    )

                route_mass_by_prev_label: Dict[str, float] = {}
                if l_prev is not None:
                    l_long = l_prev.long()
                    for label_id in sorted(int(v) for v in torch.unique(l_long.detach()).cpu().tolist()):
                        label_mask = l_long == int(label_id)
                        label_mass = _route_mass(label_mask)
                        if label_mass is not None:
                            route_mass_by_prev_label[str(label_id)] = float(label_mass.mean().item())

                cache_k_stability = _adjacent_frame_stability(k_hist)
                cache_v_stability = _adjacent_frame_stability(v_hist)

                stable_delta = None
                if stable_mass is not None and stable_random_mass is not None:
                    stable_delta = stable_mass - stable_random_mass
                unreliable_delta = None
                if unreliable_mass is not None and unreliable_random_mass is not None:
                    unreliable_delta = unreliable_mass - unreliable_random_mass

                dump_dir = Path(dump_dir_text)
                dump_dir.mkdir(parents=True, exist_ok=True)
                chunk_idx = int((hmc_control or {}).get("semantic_action_chunk_idx", -1))
                out_path = dump_dir / f"chunk_{chunk_idx:03d}_swa_raw_transport_layer_{int(swa_layer_idx):02d}.pt"
                payload = {
                    "schema": "acl2_v97_swa_raw_transport_trace_v2",
                    "artifact": "SAVE_V97_SWA_RAW_TRANSPORT_TRACE_TOPK_IDENTITY",
                    "diagnostic_only": True,
                    "chunk_idx": int(chunk_idx),
                    "layer": int(layer),
                    "swa_layer_idx": int(swa_layer_idx),
                    "batch_size": int(batch_size),
                    "frame_num": int(frame_num),
                    "tokens_per_frame": int(tokens_per_frame),
                    "head_count": int(head_count),
                    "current_tokens": int(current_tokens),
                    "history_tokens": int(history_tokens),
                    "d_prev_patch_tokens": int(d_prev.shape[-1]) if d_prev is not None else 0,
                    "label_prev_patch_tokens": int(l_prev.shape[-1]) if l_prev is not None else 0,
                    "current_semantic_fine_trace_available": bool(sampled_query_fine_labels is not None),
                    "current_semantic_group_trace_available": bool(sampled_query_group_labels is not None),
                    "current_stage_c_seed_global_track_idx_trace_available": bool(
                        sampled_query_stage_c_seed_global_track_idx is not None
                    ),
                    "cache_stage_c_seed_global_track_idx_trace_available": bool(
                        topk_cache_stage_c_seed_global_track_idx is not None
                    ),
                    "current_stage_c_masklet_instance_idx_trace_available": bool(
                        sampled_query_stage_c_masklet_instance_idx is not None
                    ),
                    "cache_stage_c_masklet_instance_idx_trace_available": bool(
                        topk_cache_stage_c_masklet_instance_idx is not None
                    ),
                    "sampled_query_count": int(q_idx.numel()),
                    "sampled_query_indices": q_idx.detach().cpu(),
                    "sampled_query_fine_label_ids": (
                        sampled_query_fine_labels.detach().cpu().to(torch.int16)
                        if sampled_query_fine_labels is not None else None
                    ),
                    "sampled_query_group_ids": (
                        sampled_query_group_labels.detach().cpu().to(torch.int16)
                        if sampled_query_group_labels is not None else None
                    ),
                    "sampled_query_stage_c_seed_global_track_idx": (
                        sampled_query_stage_c_seed_global_track_idx.detach().cpu().to(torch.int32)
                        if sampled_query_stage_c_seed_global_track_idx is not None else None
                    ),
                    "sampled_query_stage_c_masklet_instance_idx": (
                        sampled_query_stage_c_masklet_instance_idx.detach().cpu().to(torch.int32)
                        if sampled_query_stage_c_masklet_instance_idx is not None else None
                    ),
                    "current_stage_c_seed_global_track_idx_unique_ids": current_stage_c_seed_unique_ids,
                    "current_stage_c_seed_global_track_idx_unique_counts": current_stage_c_seed_unique_counts,
                    "current_stage_c_seed_global_track_idx_nonnegative_count": int(
                        current_stage_c_seed_nonnegative_count
                    ),
                    "current_stage_c_seed_global_track_idx_unique_count": int(current_stage_c_seed_unique_count),
                    "current_stage_c_masklet_instance_idx_unique_ids": current_stage_c_masklet_instance_unique_ids,
                    "current_stage_c_masklet_instance_idx_unique_counts": (
                        current_stage_c_masklet_instance_unique_counts
                    ),
                    "current_stage_c_masklet_instance_idx_nonnegative_count": int(
                        current_stage_c_masklet_instance_nonnegative_count
                    ),
                    "current_stage_c_masklet_instance_idx_unique_count": int(
                        current_stage_c_masklet_instance_unique_count
                    ),
                    "topk_identity_available": True,
                    "topk_identity_topk": int(topk_count),
                    "topk_identity_missing_reason": "",
                    "topk_identity_frame_missing_reason": topk_identity_frame_missing_reason,
                    "q_current_shape": list(q_current.shape),
                    "k_current_shape": list(k_current.shape),
                    "v_current_shape": list(v_current.shape),
                    "k_cache_shape": list(k_cache.shape),
                    "v_cache_shape": list(v_cache.shape),
                    **hidden_trace_debug,
                    "current_Q_to_cache_K_topk_cache_indices": topk_indices.detach().cpu().to(torch.int32),
                    "current_Q_to_cache_K_topk_cache_fine_label_ids": (
                        topk_cache_fine_labels.detach().cpu().to(torch.int16)
                        if topk_cache_fine_labels is not None else None
                    ),
                    "current_Q_to_cache_K_topk_cache_group_ids": (
                        topk_cache_group_labels.detach().cpu().to(torch.int16)
                        if topk_cache_group_labels is not None else None
                    ),
                    "current_Q_to_cache_K_topk_cache_stage_c_seed_global_track_idx": (
                        topk_cache_stage_c_seed_global_track_idx.detach().cpu().to(torch.int32)
                        if topk_cache_stage_c_seed_global_track_idx is not None else None
                    ),
                    "current_Q_to_cache_K_topk_cache_stage_c_masklet_instance_idx": (
                        topk_cache_stage_c_masklet_instance_idx.detach().cpu().to(torch.int32)
                        if topk_cache_stage_c_masklet_instance_idx is not None else None
                    ),
                    "current_Q_to_cache_K_topk_same_fine_label": (
                        topk_same_fine_labels.detach().cpu()
                        if topk_same_fine_labels is not None else None
                    ),
                    "current_Q_to_cache_K_topk_same_group": (
                        topk_same_group_labels.detach().cpu()
                        if topk_same_group_labels is not None else None
                    ),
                    "current_Q_to_cache_K_topk_same_stage_c_seed_global_track_idx": (
                        topk_same_stage_c_seed_global_track_idx.detach().cpu()
                        if topk_same_stage_c_seed_global_track_idx is not None else None
                    ),
                    "current_Q_to_cache_K_topk_same_stage_c_masklet_instance_idx": (
                        topk_same_stage_c_masklet_instance_idx.detach().cpu()
                        if topk_same_stage_c_masklet_instance_idx is not None else None
                    ),
                    "current_Q_to_cache_K_topk_same_fine_label_frac_mean": _scalar_mean(
                        topk_same_fine_labels.float() if topk_same_fine_labels is not None else None
                    ),
                    "current_Q_to_cache_K_topk_same_group_frac_mean": _scalar_mean(
                        topk_same_group_labels.float() if topk_same_group_labels is not None else None
                    ),
                    "current_Q_to_cache_K_topk_same_stage_c_seed_global_track_idx_frac_mean": _scalar_mean(
                        topk_same_stage_c_seed_global_track_idx.float()
                        if topk_same_stage_c_seed_global_track_idx is not None else None
                    ),
                    "current_Q_to_cache_K_topk_same_stage_c_masklet_instance_idx_frac_mean": _scalar_mean(
                        topk_same_stage_c_masklet_instance_idx.float()
                        if topk_same_stage_c_masklet_instance_idx is not None else None
                    ),
                    "current_Q_to_cache_K_topk_cache_frames": (
                        topk_frames.detach().cpu().to(torch.int16) if topk_frames is not None else None
                    ),
                    "current_Q_to_cache_K_topk_scores": topk_scores.detach().cpu().to(torch.float16),
                    "current_Q_to_cache_K_similarity_mean_by_head": torch.tensor(_mean_by_head(cosine), dtype=torch.float32),
                    "current_Q_to_cache_K_similarity_max_by_head": torch.tensor(_mean_by_head(cosine.max(dim=-1).values), dtype=torch.float32),
                    "current_Q_to_cache_K_top1_cache_index_unique_frac_by_head": (
                        top1_index_unique_frac.detach().cpu().float()
                    ),
                    "current_Q_to_cache_K_top1_cache_frame_unique_frac_by_head": (
                        top1_frame_unique_frac.detach().cpu().float()
                        if top1_frame_unique_frac is not None else None
                    ),
                    "current_Q_to_cache_K_top1_cache_index_switch_rate_by_head": (
                        top1_index_switch_rate.detach().cpu().float()
                    ),
                    "current_Q_to_cache_K_top1_cache_frame_switch_rate_by_head": (
                        top1_frame_switch_rate.detach().cpu().float()
                        if top1_frame_switch_rate is not None else None
                    ),
                    "current_Q_to_cache_K_top1_same_frame_frac_by_head": (
                        top1_same_frame_frac.detach().cpu().float()
                        if top1_same_frame_frac is not None else None
                    ),
                    "current_Q_to_cache_K_topk_query_frame_hit_frac_by_head": (
                        topk_query_frame_hit_frac.detach().cpu().float()
                        if topk_query_frame_hit_frac is not None else None
                    ),
                    "current_Q_to_cache_K_topk_same_frame_frac_by_head": (
                        topk_same_frame_frac.detach().cpu().float()
                        if topk_same_frame_frac is not None else None
                    ),
                    "current_Q_to_cache_K_top1_abs_frame_delta_mean_by_head": (
                        top1_abs_frame_delta_mean.detach().cpu().float()
                        if top1_abs_frame_delta_mean is not None else None
                    ),
                    "route_entropy_mean_by_head": torch.tensor(_mean_by_head(entropy), dtype=torch.float32),
                    "feature_transport_residual_by_head": torch.tensor(_mean_by_head(residual), dtype=torch.float32),
                    "cache_K_stability_by_head": (
                        cache_k_stability.detach().cpu().float() if cache_k_stability is not None else None
                    ),
                    "cache_V_stability_by_head": (
                        cache_v_stability.detach().cpu().float() if cache_v_stability is not None else None
                    ),
                    "stable_structure_pair_mass_by_head": (
                        torch.tensor(_mean_by_head(stable_mass), dtype=torch.float32) if stable_mass is not None else None
                    ),
                    "unreliable_dynamic_boundary_pair_mass_by_head": (
                        torch.tensor(_mean_by_head(unreliable_mass), dtype=torch.float32)
                        if unreliable_mass is not None else None
                    ),
                    "stable_route_actual_minus_random_by_head": (
                        torch.tensor(_mean_by_head(stable_delta), dtype=torch.float32) if stable_delta is not None else None
                    ),
                    "unreliable_route_actual_minus_random_by_head": (
                        torch.tensor(_mean_by_head(unreliable_delta), dtype=torch.float32)
                        if unreliable_delta is not None else None
                    ),
                    "route_mass_by_prev_fine_label": route_mass_by_prev_label,
                    "stable_pair_reason": stable_reason,
                    "unreliable_pair_reason": unreliable_reason,
                    "stable_pair_fallback_used": bool(stable_fallback_used),
                    "stable_pair_fallback_reason": stable_fallback_reason,
                    "d_prev_low_pair_tokens": _mask_count(d_low_mask),
                    "d_prev_high_pair_tokens": _mask_count(d_high_mask),
                    "g_prev_high_pair_tokens": _mask_count(g_high_mask),
                    "k_stable_pair_tokens": _mask_count(k_stable_mask),
                    "label_static_structure_pair_tokens": _mask_count(label_static_mask),
                    "stable_pair_strict_tokens": _mask_count(strict_stable_mask),
                    "stable_pair_semantic_lowd_tokens": _mask_count(semantic_lowd_mask),
                    "stable_pair_lowd_nonunreliable_tokens": _mask_count(lowd_nonunreliable_mask),
                    "stable_pair_tokens": int(stable_mask.sum().item()) if stable_mask is not None else 0,
                    "unreliable_pair_tokens": int(unreliable_mask.sum().item()) if unreliable_mask is not None else 0,
                    "ttt_prev_stable_anchor_identity_available": bool(
                        ttt_prev_stable_mask is not None and bool(ttt_prev_stable_mask.any())
                    ),
                    "ttt_prev_stable_anchor_lifecycle_schema": "acl2_v99_anchor_lifecycle_rows_v1",
                    "ttt_prev_stable_anchor_lifecycle_rows": ttt_prev_anchor_lifecycle_rows,
                    "ttt_prev_stable_anchor_lifecycle_row_count": int(len(ttt_prev_anchor_lifecycle_rows)),
                    "ttt_prev_tracked_instance_anchor_lifecycle_schema": (
                        "acl2_v103_tracked_instance_anchor_lifecycle_rows_v1"
                    ),
                    "ttt_prev_tracked_instance_anchor_lifecycle_rows": (
                        ttt_prev_tracked_instance_lifecycle_rows
                    ),
                    "ttt_prev_tracked_instance_anchor_lifecycle_row_count": int(
                        len(ttt_prev_tracked_instance_lifecycle_rows)
                    ),
                    "ttt_prev_stable_anchor_source_chunk_idx": int(
                        (hmc_control or {}).get("prev_ttt_stable_anchor_source_chunk_idx", -1) or -1
                    ),
                    "ttt_prev_stable_anchor_source_token_count": int(
                        (hmc_control or {}).get("prev_ttt_stable_anchor_token_count", 0) or 0
                    ),
                    "ttt_prev_stable_anchor_full_token_count": _mask_count(ttt_prev_stable_mask),
                    "ttt_prev_stable_anchor_route_mass_mean": _scalar_mean(ttt_prev_stable_mass),
                    "ttt_prev_stable_anchor_topk_hit_frac_mean": _scalar_mean(
                        ttt_prev_topk_hits.float() if ttt_prev_topk_hits is not None else None
                    ),
                    "ttt_prev_stable_anchor_topk_query_hit_frac_mean": _scalar_mean(ttt_prev_query_hits),
                    "ttt_prev_stable_anchor_top1_hit_frac_mean": _scalar_mean(ttt_prev_top1_hits),
                    "ttt_prev_tracked_instance_anchor_identity_available": bool(
                        ttt_prev_tracked_instance_mask is not None
                        and bool(ttt_prev_tracked_instance_mask.any())
                    ),
                    "ttt_prev_tracked_instance_anchor_source_chunk_idx": int(
                        (hmc_control or {}).get("prev_ttt_tracked_instance_anchor_source_chunk_idx", -1) or -1
                    ),
                    "ttt_prev_tracked_instance_anchor_source_token_count": int(
                        (hmc_control or {}).get("prev_ttt_tracked_instance_anchor_token_count", 0) or 0
                    ),
                    "ttt_prev_tracked_instance_anchor_full_token_count": _mask_count(
                        ttt_prev_tracked_instance_mask
                    ),
                    "ttt_prev_tracked_instance_anchor_route_mass_mean": _scalar_mean(
                        ttt_prev_tracked_instance_mass
                    ),
                    "ttt_prev_tracked_instance_anchor_topk_hit_frac_mean": _scalar_mean(
                        ttt_prev_tracked_instance_topk_hits.float()
                        if ttt_prev_tracked_instance_topk_hits is not None else None
                    ),
                    "ttt_prev_tracked_instance_anchor_topk_query_hit_frac_mean": _scalar_mean(
                        ttt_prev_tracked_instance_query_hits
                    ),
                    "ttt_prev_tracked_instance_anchor_top1_hit_frac_mean": _scalar_mean(
                        ttt_prev_tracked_instance_top1_hits
                    ),
                    "ttt_prev_tracked_instance_anchor_topk_same_seed_frac_mean": _scalar_mean(
                        ttt_prev_tracked_instance_same_seed_topk_hits.float()
                        if ttt_prev_tracked_instance_same_seed_topk_hits is not None else None
                    ),
                    "ttt_prev_tracked_instance_anchor_topk_same_masklet_frac_mean": _scalar_mean(
                        ttt_prev_tracked_instance_same_masklet_topk_hits.float()
                        if ttt_prev_tracked_instance_same_masklet_topk_hits is not None else None
                    ),
                    "ttt_prev_stable_anchor_topk_same_query_fine_label_frac_mean": _scalar_mean(
                        ttt_prev_same_fine_topk_hits.float()
                        if ttt_prev_same_fine_topk_hits is not None else None
                    ),
                    "ttt_prev_stable_anchor_topk_same_query_group_frac_mean": _scalar_mean(
                        ttt_prev_same_group_topk_hits.float()
                        if ttt_prev_same_group_topk_hits is not None else None
                    ),
                    "ttt_prev_stable_anchor_retention_mean": _scalar_mean(
                        ttt_prev_retention[ttt_prev_stable_mask]
                        if ttt_prev_retention is not None
                        and ttt_prev_stable_mask is not None
                        and bool(ttt_prev_stable_mask.any())
                        else None
                    ),
                    "ttt_prev_stable_anchor_residual_mean": _scalar_mean(
                        ttt_prev_residual[ttt_prev_stable_mask]
                        if ttt_prev_residual is not None
                        and ttt_prev_stable_mask is not None
                        and bool(ttt_prev_stable_mask.any())
                        else None
                    ),
                    "current_Q_to_cache_K_topk_ttt_prev_stable_anchor_hit_mask": (
                        ttt_prev_topk_hits.detach().cpu() if ttt_prev_topk_hits is not None else None
                    ),
                    "current_Q_to_cache_K_topk_ttt_prev_stable_anchor_ids": (
                        ttt_prev_topk_anchor_ids.detach().cpu().to(torch.int64)
                        if ttt_prev_topk_anchor_ids is not None else None
                    ),
                    "current_Q_to_cache_K_topk_ttt_prev_stable_anchor_same_query_fine_label": (
                        ttt_prev_same_fine_topk_hits.detach().cpu()
                        if ttt_prev_same_fine_topk_hits is not None else None
                    ),
                    "current_Q_to_cache_K_topk_ttt_prev_stable_anchor_same_query_group": (
                        ttt_prev_same_group_topk_hits.detach().cpu()
                        if ttt_prev_same_group_topk_hits is not None else None
                    ),
                    "current_Q_to_cache_K_topk_ttt_prev_tracked_instance_anchor_hit_mask": (
                        ttt_prev_tracked_instance_topk_hits.detach().cpu()
                        if ttt_prev_tracked_instance_topk_hits is not None else None
                    ),
                    "current_Q_to_cache_K_topk_ttt_prev_tracked_instance_anchor_ids": (
                        ttt_prev_tracked_instance_topk_anchor_ids.detach().cpu().to(torch.int64)
                        if ttt_prev_tracked_instance_topk_anchor_ids is not None else None
                    ),
                    "current_Q_to_cache_K_topk_ttt_prev_tracked_instance_anchor_seeds": (
                        ttt_prev_tracked_instance_topk_anchor_seeds.detach().cpu().to(torch.int64)
                        if ttt_prev_tracked_instance_topk_anchor_seeds is not None else None
                    ),
                    "current_Q_to_cache_K_topk_ttt_prev_tracked_instance_anchor_same_seed": (
                        ttt_prev_tracked_instance_same_seed_topk_hits.detach().cpu()
                        if ttt_prev_tracked_instance_same_seed_topk_hits is not None else None
                    ),
                    "current_Q_to_cache_K_topk_ttt_prev_tracked_instance_anchor_same_masklet": (
                        ttt_prev_tracked_instance_same_masklet_topk_hits.detach().cpu()
                        if ttt_prev_tracked_instance_same_masklet_topk_hits is not None else None
                    ),
                    "stable_pair_groups_nonempty": bool(stable_mask is not None and bool(stable_mask.any())),
                    "unreliable_pair_groups_nonempty": bool(unreliable_mask is not None and bool(unreliable_mask.any())),
                    "qk_similarity_mean": _scalar_mean(cosine),
                    "qk_similarity_max_mean": _scalar_mean(cosine.max(dim=-1).values),
                    "route_entropy_mean": _scalar_mean(entropy),
                    "feature_transport_residual_mean": _scalar_mean(residual),
                    "cache_k_stability_mean": _scalar_mean(cache_k_stability),
                    "cache_v_stability_mean": _scalar_mean(cache_v_stability),
                    "top1_cache_index_unique_frac_mean": _scalar_mean(top1_index_unique_frac),
                    "top1_cache_frame_unique_frac_mean": _scalar_mean(top1_frame_unique_frac),
                    "top1_cache_index_switch_rate_mean": _scalar_mean(top1_index_switch_rate),
                    "top1_cache_frame_switch_rate_mean": _scalar_mean(top1_frame_switch_rate),
                    "top1_same_frame_frac_mean": _scalar_mean(top1_same_frame_frac),
                    "topk_query_frame_hit_frac_mean": _scalar_mean(topk_query_frame_hit_frac),
                    "topk_same_frame_frac_mean": _scalar_mean(topk_same_frame_frac),
                    "top1_abs_frame_delta_mean": _scalar_mean(top1_abs_frame_delta_mean),
                    "stable_pair_mass_mean": _scalar_mean(stable_mass),
                    "unreliable_pair_mass_mean": _scalar_mean(unreliable_mass),
                    "stable_actual_minus_random_mean": _scalar_mean(stable_delta),
                    "unreliable_actual_minus_random_mean": _scalar_mean(unreliable_delta),
                }
                if isinstance(extra_trace_fields, dict):
                    for key, value in extra_trace_fields.items():
                        key_text = str(key)
                        if not key_text.startswith("v102_swa_state_machine_"):
                            continue
                        if torch.is_tensor(value):
                            if int(value.numel()) != 1:
                                continue
                            value = value.detach().cpu().item()
                        elif isinstance(value, tuple):
                            value = list(value)
                        if isinstance(value, (str, bool, int, float)) or value is None:
                            payload[key_text] = value
                        elif isinstance(value, list) and all(
                            isinstance(item, (str, bool, int, float)) or item is None
                            for item in value
                        ):
                            payload[key_text] = value
                torch.save(payload, out_path)

                return {
                    "swa_raw_transport_trace_available": True,
                    "swa_raw_transport_trace_path": str(out_path),
                    "swa_raw_transport_trace_schema": payload["schema"],
                    "swa_raw_transport_trace_sampled_query_count": int(q_idx.numel()),
                    "swa_raw_transport_trace_head_count": int(head_count),
                    "swa_raw_transport_topk_identity_available": True,
                    "swa_raw_transport_topk_identity_topk": int(topk_count),
                    "swa_raw_transport_current_tokens": int(current_tokens),
                    "swa_raw_transport_history_tokens": int(history_tokens),
                    "swa_raw_transport_d_prev_low_pair_tokens": int(payload["d_prev_low_pair_tokens"]),
                    "swa_raw_transport_d_prev_high_pair_tokens": int(payload["d_prev_high_pair_tokens"]),
                    "swa_raw_transport_g_prev_high_pair_tokens": int(payload["g_prev_high_pair_tokens"]),
                    "swa_raw_transport_k_stable_pair_tokens": int(payload["k_stable_pair_tokens"]),
                    "swa_raw_transport_label_static_structure_pair_tokens": int(payload["label_static_structure_pair_tokens"]),
                    "swa_raw_transport_stable_pair_strict_tokens": int(payload["stable_pair_strict_tokens"]),
                    "swa_raw_transport_stable_pair_semantic_lowd_tokens": int(payload["stable_pair_semantic_lowd_tokens"]),
                    "swa_raw_transport_stable_pair_lowd_nonunreliable_tokens": int(payload["stable_pair_lowd_nonunreliable_tokens"]),
                    "swa_raw_transport_stable_pair_tokens": int(payload["stable_pair_tokens"]),
                    "swa_raw_transport_unreliable_pair_tokens": int(payload["unreliable_pair_tokens"]),
                    "swa_raw_transport_ttt_prev_stable_anchor_identity_available": bool(
                        payload["ttt_prev_stable_anchor_identity_available"]
                    ),
                    "swa_raw_transport_ttt_prev_stable_anchor_full_token_count": int(
                        payload["ttt_prev_stable_anchor_full_token_count"]
                    ),
                    "swa_raw_transport_ttt_prev_stable_anchor_topk_hit_frac_mean": payload[
                        "ttt_prev_stable_anchor_topk_hit_frac_mean"
                    ],
                    "swa_raw_transport_ttt_prev_stable_anchor_topk_query_hit_frac_mean": payload[
                        "ttt_prev_stable_anchor_topk_query_hit_frac_mean"
                    ],
                    "swa_raw_transport_stable_fallback_used": bool(stable_fallback_used),
                    "swa_raw_transport_stable_groups_nonempty": bool(payload["stable_pair_groups_nonempty"]),
                    "swa_raw_transport_unreliable_groups_nonempty": bool(payload["unreliable_pair_groups_nonempty"]),
                    "swa_raw_transport_qk_similarity_mean": _scalar_mean(cosine),
                    "swa_raw_transport_qk_similarity_max_mean": _scalar_mean(cosine.max(dim=-1).values),
                    "swa_raw_transport_route_entropy_mean": _scalar_mean(entropy),
                    "swa_raw_transport_feature_residual_mean": _scalar_mean(residual),
                    "swa_raw_transport_cache_k_stability_mean": _scalar_mean(cache_k_stability),
                    "swa_raw_transport_cache_v_stability_mean": _scalar_mean(cache_v_stability),
                    "swa_raw_transport_top1_cache_index_unique_frac_mean": _scalar_mean(top1_index_unique_frac),
                    "swa_raw_transport_top1_cache_frame_unique_frac_mean": _scalar_mean(top1_frame_unique_frac),
                    "swa_raw_transport_top1_cache_index_switch_rate_mean": _scalar_mean(top1_index_switch_rate),
                    "swa_raw_transport_top1_cache_frame_switch_rate_mean": _scalar_mean(top1_frame_switch_rate),
                    "swa_raw_transport_top1_same_frame_frac_mean": _scalar_mean(top1_same_frame_frac),
                    "swa_raw_transport_topk_query_frame_hit_frac_mean": _scalar_mean(topk_query_frame_hit_frac),
                    "swa_raw_transport_topk_same_frame_frac_mean": _scalar_mean(topk_same_frame_frac),
                    "swa_raw_transport_top1_abs_frame_delta_mean": _scalar_mean(top1_abs_frame_delta_mean),
                    "swa_raw_transport_stable_pair_mass_mean": _scalar_mean(stable_mass),
                    "swa_raw_transport_unreliable_pair_mass_mean": _scalar_mean(unreliable_mass),
                    "swa_raw_transport_stable_actual_minus_random_mean": _scalar_mean(stable_delta),
                    "swa_raw_transport_unreliable_actual_minus_random_mean": _scalar_mean(unreliable_delta),
                }
        except Exception as exc:  # pragma: no cover - diagnostic-only best effort.
            return {
                "swa_raw_transport_trace_available": False,
                "swa_raw_transport_trace_error": f"{type(exc).__name__}: {exc}",
            }

    def _make_swa_overlap_source_gate(
        self,
        hmc_control: Optional[Dict[str, Any]],
        *,
        batch_size: int,
        frame_num: int,
        tokens_per_frame: int,
        history_tokens: int,
        current_tokens: int,
        device: torch.device,
        dtype: torch.dtype,
        swa_layer_idx: int,
    ) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
        stats: Dict[str, Any] = {
            "swa_overlap_source_gate_applied": False,
            "swa_overlap_source_gate_tokens": 0,
        }
        if not hmc_control or hmc_control.get("identity_hooks", False):
            return None, stats
        rho = float(hmc_control.get("swa_overlap_source_gate_rho", 0.0))
        if rho == 0.0 or history_tokens <= 0 or current_tokens <= 0:
            return None, stats
        D_tok = hmc_control.get("D_tok")
        D_prev = hmc_control.get("D_prev_patch")
        if D_tok is None or D_prev is None:
            return None, stats
        if frame_num <= 0 or tokens_per_frame <= 0:
            return None, stats
        if current_tokens != frame_num * tokens_per_frame:
            return None, stats

        overlap_frames = max(int(hmc_control.get("swa_overlap_frames", 0)), 0)
        if overlap_frames <= 0:
            return None, stats

        D_cur = D_tok.to(device=device, dtype=torch.float32).reshape(batch_size, frame_num, tokens_per_frame)
        prev_flat = D_prev.to(device=device, dtype=torch.float32).reshape(-1)
        if prev_flat.numel() < tokens_per_frame:
            return None, stats
        prev_frames = int(prev_flat.numel() // tokens_per_frame)
        hist_frames = int(history_tokens // tokens_per_frame)
        usable_frames = min(prev_frames, hist_frames)
        if usable_frames <= 0:
            return None, stats
        ov = min(overlap_frames, frame_num, usable_frames)
        if ov <= 0:
            return None, stats

        source_tokens = ov * tokens_per_frame
        source_end = history_tokens
        source_start = max(0, source_end - source_tokens)
        source_tokens = source_end - source_start
        if source_tokens <= 0:
            return None, stats

        prev_flat = prev_flat[-usable_frames * tokens_per_frame:]
        D_src_frames = prev_flat.reshape(1, usable_frames, tokens_per_frame).expand(batch_size, -1, -1)
        Ds = D_src_frames[:, -ov:, :].reshape(batch_size, ov * tokens_per_frame)
        Dq = D_cur[:, :ov, :].reshape(batch_size, ov * tokens_per_frame)
        if Ds.shape[1] != source_tokens:
            Ds = Ds[:, -source_tokens:]
        if Dq.shape[1] != source_tokens:
            Dq = Dq[:, :source_tokens]

        mode = str(hmc_control.get("swa_overlap_source_gate_mode", "source"))
        mode_l = mode.lower()
        boost_mode = mode_l.startswith("boost_")
        score_mode = mode_l[len("boost_"):] if boost_mode else mode_l
        random_same_mass = score_mode.endswith("_random_same_mass")
        base_mode = score_mode[:-len("_random_same_mass")] if random_same_mass else score_mode
        top_quantile = None
        for suffix, quantile in (("_topq80", 0.80), ("_topq90", 0.90)):
            if base_mode.endswith(suffix):
                top_quantile = quantile
                base_mode = base_mode[: -len(suffix)]
                break
        Dq = Dq.clamp(0.0, 1.0)
        Ds = Ds.clamp(0.0, 1.0)
        if base_mode in {"source", "prev", "previous"}:
            score = Ds
        elif base_mode in {"current", "query"}:
            score = Dq
        elif base_mode == "union":
            score = torch.maximum(Dq, Ds)
        elif base_mode in {"intersection", "inter"}:
            score = torch.minimum(Dq, Ds)
        elif base_mode in {"disagreement", "mismatch"}:
            score = (Dq - Ds).abs()
        elif base_mode in {"agree_dyn", "product"}:
            score = Dq * Ds
        elif base_mode in {"stable", "stable_agreement", "low_dyn_agreement"}:
            score = 1.0 - torch.maximum(Dq, Ds)
        elif score_mode in self._swa_overlap_source_semantic_modes():
            semantic_score, semantic_stats = self._make_swa_overlap_source_semantic_score(
                hmc_control,
                mode=score_mode,
                batch_size=batch_size,
                frame_num=frame_num,
                tokens_per_frame=tokens_per_frame,
                history_tokens=history_tokens,
                source_tokens=source_tokens,
                ov=ov,
                Ds=Ds,
                device=device,
            )
            if semantic_score is None:
                return None, stats
            score = semantic_score
            stats.update(semantic_stats)
        elif score_mode in self._swa_overlap_source_role_modes():
            semantic_role_score, semantic_role_stats = self._make_swa_overlap_source_role_score(
                hmc_control,
                mode=score_mode,
                batch_size=batch_size,
                frame_num=frame_num,
                tokens_per_frame=tokens_per_frame,
                source_tokens=source_tokens,
                ov=ov,
                Dq=Dq,
                device=device,
            )
            if semantic_role_score is None:
                return None, stats
            score = semantic_role_score
            stats.update(semantic_role_stats)
        else:
            raise ValueError(f"Unsupported SWA overlap source gate mode: {mode}")
        score = score.clamp(0.0, 1.0)
        if top_quantile is not None:
            cutoff = torch.quantile(score.detach().float(), float(top_quantile), dim=1, keepdim=True)
            selected = score.detach().float() >= cutoff
            score = torch.where(selected, score, torch.zeros_like(score))
            stats.update({
                "swa_overlap_source_geometric_topq": float(top_quantile),
                "swa_overlap_source_geometric_topq_base_mode": base_mode,
                "swa_overlap_source_geometric_topq_selected_tokens": int(selected.sum().item()),
            })
        if (
            random_same_mass
            and score_mode not in self._swa_overlap_source_semantic_modes()
            and score_mode not in self._swa_overlap_source_role_modes()
        ):
            score = self._randomize_swa_overlap_score_same_distribution(
                score,
                hmc_control,
                swa_layer_idx=swa_layer_idx,
                salt_offset=1000.0,
            )
            stats.update({
                "swa_overlap_source_geometric_random_same_mass": True,
                "swa_overlap_source_geometric_random_base_mode": base_mode,
            })

        min_gate = min(max(float(hmc_control.get("swa_overlap_source_gate_min", 0.85)), 0.0), 1.0)
        if boost_mode:
            gate_max = max(1.0, 1.0 + abs(rho))
            gate_slice = (1.0 + abs(rho) * score).clamp(1.0, gate_max).to(dtype=dtype)
        else:
            gate_max = 1.0
            gate_slice = (1.0 - rho * score).clamp(min_gate, 1.0).to(dtype=dtype)
        gate = torch.ones(batch_size, 1, history_tokens, 1, device=device, dtype=dtype)
        gate[:, :, source_start:source_end, :] = gate_slice.reshape(batch_size, 1, source_tokens, 1)
        gate_delta = (1.0 - gate_slice.detach().float()).abs()
        stats.update(self._dump_swa_overlap_feature_map(
            hmc_control,
            kind="source_gate",
            mode=mode,
            swa_layer_idx=swa_layer_idx,
            batch_size=batch_size,
            frame_num=frame_num,
            tokens_per_frame=tokens_per_frame,
            history_tokens=history_tokens,
            source_start=source_start,
            source_end=source_end,
            overlap_frames=ov,
            Dq=Dq,
            Ds=Ds,
            score=score,
            control=gate_slice,
        ))
        stats.update({
            "swa_overlap_source_gate_applied": True,
            "swa_overlap_source_gate_mode": mode,
            "swa_overlap_source_gate_score_mode": score_mode,
            "swa_overlap_source_gate_boost": bool(boost_mode),
            "swa_overlap_source_gate_rho": rho,
            "swa_overlap_source_gate_min": min_gate,
            "swa_overlap_source_gate_max": float(gate_max),
            "swa_overlap_source_gate_tokens": int(source_tokens),
            "swa_overlap_source_gate_source_start": int(source_start),
            "swa_overlap_source_gate_source_end": int(source_end),
            "swa_overlap_source_gate_mean": float(gate_slice.detach().float().mean().item()),
            "swa_overlap_source_gate_p10": float(torch.quantile(gate_slice.detach().float(), 0.10).item()),
            "swa_overlap_source_gate_p50": float(torch.quantile(gate_slice.detach().float(), 0.50).item()),
            "swa_overlap_source_gate_p90": float(torch.quantile(gate_slice.detach().float(), 0.90).item()),
            "swa_overlap_source_gate_mean_abs_delta": float(gate_delta.mean().item()),
            "swa_overlap_source_gate_max_abs_delta": float(gate_delta.max().item()),
            "swa_overlap_source_gate_score_mean": float(score.detach().float().mean().item()),
            "swa_overlap_source_gate_score_q90": float(torch.quantile(score.detach().float(), 0.90).item()),
        })
        return gate, stats

    def _make_swa_overlap_source_replace(
        self,
        hmc_control: Optional[Dict[str, Any]],
        *,
        batch_size: int,
        frame_num: int,
        tokens_per_frame: int,
        history_tokens: int,
        current_tokens: int,
        device: torch.device,
        dtype: torch.dtype,
        swa_layer_idx: int,
    ) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        stats: Dict[str, Any] = {
            "swa_overlap_source_replace_applied": False,
            "swa_overlap_source_replace_tokens": 0,
        }
        if not hmc_control or hmc_control.get("identity_hooks", False):
            return None, stats
        alpha_max = max(float(hmc_control.get("swa_overlap_source_replace_alpha", 0.0)), 0.0)
        if alpha_max <= 0.0 or history_tokens <= 0 or current_tokens <= 0:
            return None, stats
        D_tok = hmc_control.get("D_tok")
        D_prev = hmc_control.get("D_prev_patch")
        if D_tok is None or D_prev is None:
            return None, stats
        if frame_num <= 0 or tokens_per_frame <= 0:
            return None, stats
        if current_tokens != frame_num * tokens_per_frame:
            return None, stats

        overlap_frames = max(int(hmc_control.get("swa_overlap_frames", 0)), 0)
        if overlap_frames <= 0:
            return None, stats

        D_cur = D_tok.to(device=device, dtype=torch.float32).reshape(batch_size, frame_num, tokens_per_frame)
        prev_flat = D_prev.to(device=device, dtype=torch.float32).reshape(-1)
        if prev_flat.numel() < tokens_per_frame:
            return None, stats
        prev_frames = int(prev_flat.numel() // tokens_per_frame)
        hist_frames = int(history_tokens // tokens_per_frame)
        usable_frames = min(prev_frames, hist_frames)
        if usable_frames <= 0:
            return None, stats
        ov = min(overlap_frames, frame_num, usable_frames)
        if ov <= 0:
            return None, stats

        source_tokens = ov * tokens_per_frame
        source_end = history_tokens
        source_start = max(0, source_end - source_tokens)
        source_tokens = source_end - source_start
        if source_tokens <= 0:
            return None, stats

        prev_flat = prev_flat[-usable_frames * tokens_per_frame:]
        D_src_frames = prev_flat.reshape(1, usable_frames, tokens_per_frame).expand(batch_size, -1, -1)
        Ds = D_src_frames[:, -ov:, :].reshape(batch_size, ov * tokens_per_frame)
        Dq = D_cur[:, :ov, :].reshape(batch_size, ov * tokens_per_frame)
        if Ds.shape[1] != source_tokens:
            Ds = Ds[:, -source_tokens:]
        if Dq.shape[1] != source_tokens:
            Dq = Dq[:, :source_tokens]

        mode = str(hmc_control.get("swa_overlap_source_replace_mode", "union"))
        mode_l = mode.lower()
        random_same_mass = mode_l.endswith("_random_same_mass")
        base_mode = mode_l[:-len("_random_same_mass")] if random_same_mass else mode_l
        top_quantile = None
        for suffix, quantile in (("_topq80", 0.80), ("_topq90", 0.90)):
            if base_mode.endswith(suffix):
                top_quantile = quantile
                base_mode = base_mode[: -len(suffix)]
                break
        Dq = Dq.clamp(0.0, 1.0)
        Ds = Ds.clamp(0.0, 1.0)
        if base_mode in {"source", "prev", "previous"}:
            score = Ds
        elif base_mode in {"current", "query"}:
            score = Dq
        elif base_mode == "union":
            score = torch.maximum(Dq, Ds)
        elif base_mode in {"intersection", "inter"}:
            score = torch.minimum(Dq, Ds)
        elif base_mode in {"disagreement", "mismatch"}:
            score = (Dq - Ds).abs()
        elif base_mode in {"agree_dyn", "product"}:
            score = Dq * Ds
        elif base_mode in {"stable", "stable_agreement", "low_dyn_agreement"}:
            score = 1.0 - torch.maximum(Dq, Ds)
        elif mode_l in self._swa_overlap_source_semantic_modes():
            semantic_score, semantic_stats = self._make_swa_overlap_source_semantic_score(
                hmc_control,
                mode=mode,
                batch_size=batch_size,
                frame_num=frame_num,
                tokens_per_frame=tokens_per_frame,
                history_tokens=history_tokens,
                source_tokens=source_tokens,
                ov=ov,
                Ds=Ds,
                device=device,
            )
            if semantic_score is None:
                return None, stats
            score = semantic_score
            stats.update(semantic_stats)
        elif mode_l in self._swa_overlap_source_role_modes():
            semantic_role_score, semantic_role_stats = self._make_swa_overlap_source_role_score(
                hmc_control,
                mode=mode,
                batch_size=batch_size,
                frame_num=frame_num,
                tokens_per_frame=tokens_per_frame,
                source_tokens=source_tokens,
                ov=ov,
                Dq=Dq,
                device=device,
            )
            if semantic_role_score is None:
                return None, stats
            score = semantic_role_score
            stats.update(semantic_role_stats)
        else:
            raise ValueError(f"Unsupported SWA overlap source replace mode: {mode}")
        score = score.clamp(0.0, 1.0)
        if top_quantile is not None:
            cutoff = torch.quantile(score.detach().float(), float(top_quantile), dim=1, keepdim=True)
            selected = score.detach().float() >= cutoff
            score = torch.where(selected, score, torch.zeros_like(score))
            stats.update({
                "swa_overlap_source_geometric_topq": float(top_quantile),
                "swa_overlap_source_geometric_topq_base_mode": base_mode,
                "swa_overlap_source_geometric_topq_selected_tokens": int(selected.sum().item()),
            })
        if random_same_mass and mode_l not in self._swa_overlap_source_semantic_modes() and mode_l not in self._swa_overlap_source_role_modes():
            score = self._randomize_swa_overlap_score_same_distribution(
                score,
                hmc_control,
                swa_layer_idx=swa_layer_idx,
                salt_offset=2000.0,
            )
            stats.update({
                "swa_overlap_source_geometric_random_same_mass": True,
                "swa_overlap_source_geometric_random_base_mode": base_mode,
            })

        alpha = (alpha_max * score).clamp(0.0, min(alpha_max, 1.0)).to(dtype=dtype)
        alpha_delta = alpha.detach().float()
        stats.update(self._dump_swa_overlap_feature_map(
            hmc_control,
            kind="source_replace",
            mode=mode,
            swa_layer_idx=swa_layer_idx,
            batch_size=batch_size,
            frame_num=frame_num,
            tokens_per_frame=tokens_per_frame,
            history_tokens=history_tokens,
            source_start=source_start,
            source_end=source_end,
            overlap_frames=ov,
            Dq=Dq,
            Ds=Ds,
            score=score,
            control=alpha,
        ))
        desc = {
            "source_start": int(source_start),
            "source_end": int(source_end),
            "source_tokens": int(source_tokens),
            "alpha": alpha.reshape(batch_size, 1, source_tokens, 1),
        }
        stats.update({
            "swa_overlap_source_replace_applied": True,
            "swa_overlap_source_replace_mode": mode,
            "swa_overlap_source_replace_alpha_max": float(alpha_max),
            "swa_overlap_source_replace_tokens": int(source_tokens),
            "swa_overlap_source_replace_source_start": int(source_start),
            "swa_overlap_source_replace_source_end": int(source_end),
            "swa_overlap_source_replace_alpha_mean": float(alpha_delta.mean().item()),
            "swa_overlap_source_replace_alpha_p90": float(torch.quantile(alpha_delta, 0.90).item()),
            "swa_overlap_source_replace_score_mean": float(score.detach().float().mean().item()),
            "swa_overlap_source_replace_score_q90": float(torch.quantile(score.detach().float(), 0.90).item()),
        })
        return desc, stats

    def decode(self, hidden, N, H, W, ttt_dict: Optional[dict] = None, window_size: Optional[int] = None, overlap_size: Optional[int] = None, is_first_window: bool = False,
               turn_off_ttt=False, turn_off_swa=False, cache_ttt_primitives: bool = False,
               hmc_control: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        BN, hw, _ = hidden.shape
        B = BN // N

        final_output = []
        hmc_trace = self._new_hmc_trace(hmc_control)
        total_decoder_layers = len(self.decoder)
        attn_prior_frame_parts: List[torch.Tensor] = []
        feature_key_parts: List[Tuple[int, torch.Tensor]] = []
        dyn4d_global_parts: List[Tuple[int, dict]] = []
        frame_attn_cosine_query_parts: List[Tuple[int, torch.Tensor]] = []
        frame_attn_cosine_key_parts: List[Tuple[int, torch.Tensor]] = []
        pca_attn_frame_q_parts: List[Tuple[int, torch.Tensor]] = []
        pca_attn_frame_k_parts: List[Tuple[int, torch.Tensor]] = []
        pca_attn_frame_v_parts: List[Tuple[int, torch.Tensor]] = []
        pca_attn_global_q_parts: List[Tuple[int, torch.Tensor]] = []
        pca_attn_global_k_parts: List[Tuple[int, torch.Tensor]] = []
        pca_attn_global_v_parts: List[Tuple[int, torch.Tensor]] = []
        pca_swa_current_q_parts: List[Tuple[int, torch.Tensor]] = []
        pca_swa_current_k_parts: List[Tuple[int, torch.Tensor]] = []
        pca_swa_current_v_parts: List[Tuple[int, torch.Tensor]] = []
        pca_swa_cache_k_parts: List[Tuple[int, torch.Tensor]] = []
        pca_swa_cache_v_parts: List[Tuple[int, torch.Tensor]] = []
        pca_ttt_q_parts: List[Tuple[int, torch.Tensor]] = []
        pca_ttt_k_parts: List[Tuple[int, torch.Tensor]] = []
        pca_ttt_v_parts: List[Tuple[int, torch.Tensor]] = []
        pca_ttt_input_parts: List[Tuple[int, torch.Tensor]] = []
        pca_ttt_apply_raw_parts: List[Tuple[int, torch.Tensor]] = []
        pca_ttt_operator_output_parts: List[Tuple[int, torch.Tensor]] = []
        pca_ttt_update_term_parts: List[Tuple[int, torch.Tensor]] = []
        pca_ttt_final_output_parts: List[Tuple[int, torch.Tensor]] = []
        frame_attn_key_cosine_l0 = None
        frame_attn_key_cosine_l4 = None
        
        hidden = hidden.reshape(B*N, hw, -1)

        register_token = self.register_token.repeat(B, N, 1, 1).reshape(B*N, *self.register_token.shape[-2:])

        pe_token_0 = getattr(self, 'pe_token_0')  # (1, 1, 1, dim)
        pe_token_1 = getattr(self, 'pe_token_1')  # (1, 1, 1, dim)
        pe_token_2 = getattr(self, 'pe_token_2')  # (1, 1, 1, dim)
        if overlap_size is None or window_size is None:
            raise ValueError("overlap_size and window_size must be provided when num_pe_tokens > 0")
        num_overlap_with_previous = min(overlap_size, N)
        num_other_frames = min(max(window_size - 2 * overlap_size, 0), N - num_overlap_with_previous)
        num_overlap_with_later = max(min(overlap_size, N, N - num_overlap_with_previous - num_other_frames), 0)
        pe_tokens = torch.cat([
            pe_token_0.repeat(B, num_overlap_with_previous, 1, 1),
            pe_token_1.repeat(B, num_other_frames, 1, 1),
            pe_token_2.repeat(B, num_overlap_with_later, 1, 1)
        ], dim=1).to(hidden.device).to(hidden.dtype).reshape(B*N, *pe_token_0.shape[-2:])  # (B*N, 1, dim)
        hidden = torch.cat([pe_tokens, hidden], dim=1)

        # Concatenate special tokens with patch tokens
        hidden = torch.cat([register_token, hidden], dim=1)
        hw = hidden.shape[1]

        if self.pos_type.startswith('rope'):
            pos = self.position_getter(B * N, H//self.patch_size, W//self.patch_size, hidden.device)

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + torch.ones_like(pos)
            pos_special = torch.zeros(B * N, self.patch_start_idx, 2).to(hidden.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)
       
        ttt_output_info = None
        ttt_state = ttt_dict.get("ttt") if ttt_dict is not None else None
        attn_state = ttt_dict.get("attn") if ttt_dict is not None else None
        gate_scales: List[torch.Tensor] = []
        attn_gate_scales: List[torch.Tensor] = []
        for i in range(len(self.decoder)):
            blk = self.decoder[i]

            if i % 2 == 0:
                # frame attention
                pos_reshaped = pos.reshape(B*N, hw, -1) if pos is not None else None
                hidden = hidden.reshape(B*N, hw, -1)
                hidden_for_block = hidden
                pos_for_block = pos_reshaped
                hmc_attn_path = "frame_attention"
            else:
                # global attention
                pos_reshaped = pos.reshape(B, N*hw, -1) if pos is not None else None
                hidden = hidden.reshape(B, N*hw, -1)
                hidden_for_block = hidden
                pos_for_block = pos_reshaped
                hmc_attn_path = "chunk_attention"

            if self._pca_debug_enabled():
                pca_qkv = self._extract_pca_attention_qkv_patchvec(
                    blk,
                    hidden_for_block,
                    pos_for_block,
                    batch_size=B,
                    frame_num=N,
                    tokens_per_frame=hw,
                    patch_h=H // self.patch_size,
                    patch_w=W // self.patch_size,
                    layout=("frame" if i % 2 == 0 else "global"),
                )
                if pca_qkv is not None:
                    if i % 2 == 0:
                        if pca_qkv.get("q") is not None:
                            pca_attn_frame_q_parts.append((i, pca_qkv["q"]))
                        if pca_qkv.get("k") is not None:
                            pca_attn_frame_k_parts.append((i, pca_qkv["k"]))
                        if pca_qkv.get("v") is not None:
                            pca_attn_frame_v_parts.append((i, pca_qkv["v"]))
                    else:
                        if pca_qkv.get("q") is not None:
                            pca_attn_global_q_parts.append((i, pca_qkv["q"]))
                        if pca_qkv.get("k") is not None:
                            pca_attn_global_k_parts.append((i, pca_qkv["k"]))
                        if pca_qkv.get("v") is not None:
                            pca_attn_global_v_parts.append((i, pca_qkv["v"]))

            # Save pre-block hidden for the fixed no-skip-residual path.
            # With skip0 config removed, default behavior is skip0=False.
            layer_skip0 = (
                len(self.ttt_insert_after) == 36
                and i in self.ttt_insert_after
                and self.ttt_insert_after.index(i) % 2 == 0
            )
            
            if i % 2 == 1 and not layer_skip0:
                hidden_before_block = hidden_for_block
            elif i % 2 == 0 and layer_skip0:
                hidden_before_block = hidden_for_block
            else:
                hidden_before_block = hidden_for_block # dummy

            need_feature_key = i in self.feature_frame_attn_layers and i % 2 == 0
            need_layer0_key = i == 0
            need_layer4_key = i == 4
            need_debug_maps = self.export_attn_debug and i in self.all_frame_attn_layers and i % 2 == 0
            if need_feature_key or need_layer0_key or need_layer4_key or need_debug_maps:
                frame_attn_cosine_query, frame_attn_cosine_key = self._extract_frame_attention_cosine_map(
                    blk,
                    hidden_for_block,
                    pos_for_block,
                    B,
                    N,
                    H // self.patch_size,
                    W // self.patch_size,
                )
                if need_debug_maps and frame_attn_cosine_query is not None:
                    frame_attn_cosine_query_parts.append((i, frame_attn_cosine_query))
                if need_debug_maps and frame_attn_cosine_key is not None:
                    frame_attn_cosine_key_parts.append((i, frame_attn_cosine_key))
                if need_layer0_key and frame_attn_cosine_key is not None:
                    frame_attn_key_cosine_l0 = frame_attn_cosine_key
                if need_layer4_key and frame_attn_cosine_key is not None:
                    frame_attn_key_cosine_l4 = frame_attn_cosine_key
                if need_feature_key and frame_attn_cosine_key is not None:
                    feature_key_parts.append((i, frame_attn_cosine_key))

            if i in self.attn_prior_layers and i % 2 == 1:
                frame_prior, _dynamic_prior = self._extract_attention_prior_from_block(
                    blk,
                    hidden_for_block,
                    pos_for_block,
                    N,
                    hw,
                    H // self.patch_size,
                    W // self.patch_size,
                )
                if frame_prior is not None:
                    attn_prior_frame_parts.append(frame_prior)

            if i in self.feature_global_attn_layers and i % 2 == 1:
                dyn4d_stats = self._extract_dyn4d_global_stats_from_block(
                    blk,
                    hidden_for_block,
                    pos_for_block,
                    N,
                    hw,
                    H // self.patch_size,
                    W // self.patch_size,
                    self.dyn4d_window_radius,
                )
                if dyn4d_stats is not None:
                    dyn4d_global_parts.append((i, dyn4d_stats))

            attn_mask = None
            frame_query_gate = None
            context_skip_stats: Dict[str, Any] = {
                "context_source_skip_applied": False,
                "source_keep_ratio": 1.0,
                "source_skip_tokens": 0,
                "empty_source_events": 0,
            }
            if hmc_attn_path == "frame_attention":
                layer_enabled = self._hmc_read_layer_enabled(hmc_control, layer=i, total_layers=total_decoder_layers)
                if layer_enabled:
                    attn_mask = self._make_frame_attention_bias(
                        hmc_control,
                        batch_size=B,
                        frame_num=N,
                        tokens_per_frame=hw,
                        num_heads=int(getattr(getattr(blk, "attn", None), "num_heads", 0) or 0),
                        device=hidden_for_block.device,
                        dtype=hidden_for_block.dtype,
                    )
                    frame_query_gate = self._make_frame_attention_query_gate(
                        hmc_control,
                        batch_size=B,
                        frame_num=N,
                        tokens_per_frame=hw,
                        device=hidden_for_block.device,
                        dtype=hidden_for_block.dtype,
                    )
                hook_key = "enable_frame_read_control"
            else:
                layer_enabled = self._hmc_read_layer_enabled(hmc_control, layer=i, total_layers=total_decoder_layers)
                if layer_enabled:
                    attn_mask = self._make_chunk_attention_source_bias(
                        hmc_control,
                        batch_size=B,
                        frame_num=N,
                        tokens_per_frame=hw,
                        device=hidden_for_block.device,
                        dtype=hidden_for_block.dtype,
                    )
                hook_key = "enable_chunk_read_control"

            context_skip_layer_enabled = self._hmc_context_source_skip_layer_enabled(
                hmc_control,
                layer=i,
                total_layers=total_decoder_layers,
            )
            if context_skip_layer_enabled:
                context_skip_bias, context_skip_stats = self._make_context_source_skip_bias(
                    hmc_control,
                    path=hmc_attn_path,
                    batch_size=B,
                    frame_num=N,
                    tokens_per_frame=hw,
                    device=hidden_for_block.device,
                    dtype=hidden_for_block.dtype,
                )
                if context_skip_bias is not None:
                    if isinstance(context_skip_bias, dict):
                        context_skip_bias["source_attention_map_dump_layer"] = int(i)
                        if attn_mask is None:
                            attn_mask = context_skip_bias
                        elif (
                            torch.is_tensor(attn_mask)
                            and context_skip_bias.get("type") == "source_soft"
                        ):
                            context_skip_bias = dict(context_skip_bias)
                            context_skip_bias["base_attn_mask"] = attn_mask
                            attn_mask = context_skip_bias
                        else:
                            context_skip_stats["context_source_skip_compact_blocked_by_existing_mask"] = True
                    else:
                        attn_mask = context_skip_bias if attn_mask is None else attn_mask + context_skip_bias

            if self._hmc_hook_requested(hmc_control, hook_key):
                mean_abs_bias = 0.0
                max_abs_bias = 0.0
                frame_bias_attention_mass_stats: Dict[str, Any] = {}
                if attn_mask is not None and torch.is_tensor(attn_mask):
                    bias_abs = attn_mask.detach().float().abs()
                    mean_abs_bias = float(bias_abs.mean().item()) if bias_abs.numel() else 0.0
                    max_abs_bias = float(bias_abs.max().item()) if bias_abs.numel() else 0.0
                    if hmc_attn_path == "frame_attention" and bool(hmc_control.get("frame_attention_record_bias_mass", False)):
                        frame_bias_attention_mass_stats = self._sample_frame_bias_attention_mass_stats(
                            blk,
                            hidden_for_block,
                            pos_for_block,
                            attn_mask,
                            max_queries=int(hmc_control.get("frame_attention_bias_mass_max_queries", 64) or 64),
                        )
                elif isinstance(attn_mask, dict) and attn_mask.get("type") == "compact_kv":
                    mean_abs_bias = 0.0
                    max_abs_bias = 0.0
                elif isinstance(attn_mask, dict) and attn_mask.get("type") == "source_soft":
                    bias_values = attn_mask.get("source_bias_values")
                    weights = attn_mask.get("source_weights")
                    if torch.is_tensor(bias_values):
                        bias_abs = bias_values.detach().float().abs()
                        mean_abs_bias = float(bias_abs.mean().item()) if bias_abs.numel() else 0.0
                        max_abs_bias = float(bias_abs.max().item()) if bias_abs.numel() else 0.0
                    elif torch.is_tensor(weights):
                        gate_delta = (1.0 - weights.detach().float()).abs()
                        mean_abs_bias = float(gate_delta.mean().item()) if gate_delta.numel() else 0.0
                        max_abs_bias = float(gate_delta.max().item()) if gate_delta.numel() else 0.0
                mean_abs_query_gate_delta = 0.0
                max_abs_query_gate_delta = 0.0
                if frame_query_gate is not None:
                    gate_delta = (1.0 - frame_query_gate.detach().float()).abs()
                    mean_abs_query_gate_delta = float(gate_delta.mean().item()) if gate_delta.numel() else 0.0
                    max_abs_query_gate_delta = float(gate_delta.max().item()) if gate_delta.numel() else 0.0
                trace_record = {
                    "layer": int(i),
                    "identity": bool(hmc_control.get("identity_hooks", False)) if hmc_control else False,
                    "layer_enabled": bool(layer_enabled),
                    "shape": [int(x) for x in hidden_for_block.shape],
                    "attn_mask_applied": attn_mask is not None,
                    "query_gate_applied": frame_query_gate is not None,
                    "context_source_skip_layer_enabled": bool(context_skip_layer_enabled),
                    "mean_abs_bias": mean_abs_bias,
                    "max_abs_bias": max_abs_bias,
                    "mean_abs_query_gate_delta": mean_abs_query_gate_delta,
                    "max_abs_query_gate_delta": max_abs_query_gate_delta,
                    "hook_site": "decoder_block_attn",
                    **context_skip_stats,
                    **frame_bias_attention_mass_stats,
                }
                self._append_hmc_trace(hmc_trace, hmc_attn_path, trace_record)
            else:
                trace_record = None

            hidden = blk(hidden_for_block, xpos=pos_for_block, attn_mask=attn_mask)
            if (
                trace_record is not None
                and isinstance(attn_mask, dict)
                and isinstance(attn_mask.get("attention_mass_stats"), list)
                and attn_mask["attention_mass_stats"]
            ):
                trace_record.update(attn_mask["attention_mass_stats"][-1])
            if frame_query_gate is not None:
                hidden = hidden_before_block + (hidden - hidden_before_block) * frame_query_gate

            if ttt_state is not None and i in ttt_state.get("insert_after", []):
                assert self.ttt_gate_projs is not None and self.ttt_layers is not None
                insert_after_list = ttt_state.get("insert_after", [])
                layer_idx = insert_after_list.index(i)

                x_for_residual = hidden.view(B, N, hw, -1)
                tokens_post = x_for_residual
                tokens_in = tokens_post
                if self._pca_debug_enabled():
                    pca_ttt_input = self._pca_tokens_to_patchvec(
                        tokens_in,
                        batch_size=B,
                        frame_num=N,
                        patch_h=H // self.patch_size,
                        patch_w=W // self.patch_size,
                    )
                    if pca_ttt_input is not None:
                        pca_ttt_input_parts.append((i, pca_ttt_input))

                gate_scale = torch.nn.functional.silu(self.ttt_gate_projs[layer_idx](tokens_in))
                if turn_off_ttt: gate_scale = torch.zeros_like(gate_scale)
                gate_scales.append(gate_scale)
                info = {
                    "ttt_op_order": ttt_state.get("ttt_op_order", []),
                    "w0": ttt_state["w0"][layer_idx],
                    "w1": ttt_state["w1"][layer_idx],
                    "w2": ttt_state["w2"][layer_idx],
                }
                cache_ttt_for_pca = bool(cache_ttt_primitives or self._pca_debug_enabled())
                ttt_output, output = self.ttt_layers[layer_idx](
                    tokens_in, info, cache_primitives=cache_ttt_for_pca,
                )
                if self._pca_debug_enabled():
                    for key, parts in (
                        ("q", pca_ttt_q_parts),
                        ("k", pca_ttt_k_parts),
                        ("v", pca_ttt_v_parts),
                    ):
                        pca_ttt = self._pca_ttt_heads_to_patchvec(
                            output.get(key),
                            batch_size=B,
                            frame_num=N,
                            tokens_per_frame=hw,
                            patch_h=H // self.patch_size,
                            patch_w=W // self.patch_size,
                        )
                        if pca_ttt is not None:
                            parts.append((i, pca_ttt))
                    pca_ttt_apply_raw = self._pca_ttt_heads_to_patchvec(
                        output.get("apply_output_raw"),
                        batch_size=B,
                        frame_num=N,
                        tokens_per_frame=hw,
                        patch_h=H // self.patch_size,
                        patch_w=W // self.patch_size,
                    )
                    if pca_ttt_apply_raw is not None:
                        pca_ttt_apply_raw_parts.append((i, pca_ttt_apply_raw))
                    pca_ttt_operator = self._pca_tokens_to_patchvec(
                        ttt_output,
                        batch_size=B,
                        frame_num=N,
                        patch_h=H // self.patch_size,
                        patch_w=W // self.patch_size,
                    )
                    if pca_ttt_operator is not None:
                        pca_ttt_operator_output_parts.append((i, pca_ttt_operator))

                ttt_apply_gate = self._make_ttt_apply_gate(
                    hmc_control,
                    batch_size=B,
                    frame_num=N,
                    tokens_per_frame=hw,
                    device=ttt_output.device,
                    dtype=ttt_output.dtype,
                ) if self._hmc_read_layer_enabled(hmc_control, layer=i, total_layers=total_decoder_layers) else None
                if self._hmc_hook_requested(hmc_control, "enable_ttt_apply_control"):
                    self._append_hmc_trace(hmc_trace, "ttt_apply", {
                        "layer": int(i),
                        "ttt_layer": int(layer_idx),
                        "identity": bool(hmc_control.get("identity_hooks", False)) if hmc_control else False,
                        "layer_enabled": ttt_apply_gate is not None,
                        "shape": [int(x) for x in ttt_output.shape],
                        "gate_applied": ttt_apply_gate is not None,
                        "hook_site": "ttt_apply_residual",
                    })

                update_term = ttt_output * gate_scale
                if ttt_apply_gate is not None:
                    update_term = update_term * ttt_apply_gate

                tokens_out = update_term + tokens_post
                if self._pca_debug_enabled():
                    pca_ttt_update = self._pca_tokens_to_patchvec(
                        update_term,
                        batch_size=B,
                        frame_num=N,
                        patch_h=H // self.patch_size,
                        patch_w=W // self.patch_size,
                    )
                    if pca_ttt_update is not None:
                        pca_ttt_update_term_parts.append((i, pca_ttt_update))
                    pca_ttt_final = self._pca_tokens_to_patchvec(
                        tokens_out,
                        batch_size=B,
                        frame_num=N,
                        patch_h=H // self.patch_size,
                        patch_w=W // self.patch_size,
                    )
                    if pca_ttt_final is not None:
                        pca_ttt_final_output_parts.append((i, pca_ttt_final))

                hidden = tokens_out

                if ttt_output_info is None:
                    ttt_output_info = {
                        "w0": [None] * len(insert_after_list),
                        "w1": [None] * len(insert_after_list),
                        "w2": [None] * len(insert_after_list),
                    }
                    if cache_ttt_primitives:
                        ttt_output_info["write_cache"] = [None] * len(insert_after_list)
                ttt_output_info["w0"][layer_idx] = output["w0"]
                ttt_output_info["w1"][layer_idx] = output["w1"]
                ttt_output_info["w2"][layer_idx] = output["w2"]

                if cache_ttt_primitives:
                    if "write_cache" not in ttt_output_info:
                        ttt_output_info["write_cache"] = [None] * len(insert_after_list)
                    ttt_output_info["write_cache"][layer_idx] = {
                        "q": output["q"].detach().cpu(),
                        "k": output["k"].detach().cpu(),
                        "v": output["v"].detach().cpu(),
                        "lr0": output["lr0"].detach().cpu(),
                        "lr1": output["lr1"].detach().cpu(),
                        "lr2": output["lr2"].detach().cpu(),
                        "w0_old": output["w0_old"].detach().cpu(),
                        "w1_old": output["w1_old"].detach().cpu(),
                        "w2_old": output["w2_old"].detach().cpu(),
                        "apply_output_raw": output["apply_output_raw"].detach().cpu()
                        if output.get("apply_output_raw") is not None else None,
                        "write_hidden": tokens_out.reshape(B, N * hw, -1).detach().cpu(),
                        "momentum": output["momentum"].detach().cpu() if output.get("momentum") is not None else None,
                        "muon_update_steps": output.get("muon_update_steps", 0),
                        "ttt_update_steps": output.get("ttt_update_steps", 1),
                        "ttt_op_order": info["ttt_op_order"],
                    }

            # Sliding Window Attention (SWA)
            if attn_state is not None and i in attn_state.get("insert_after", []):
                assert self.swa_gate_projs is not None and self.swa_layers is not None
                insert_after_list = attn_state.get("insert_after", [])
                layer_idx = insert_after_list.index(i)

                patch_tokens_post_block = hidden
                x_for_residual = patch_tokens_post_block.view(B, N, hw, -1)
                x_in = x_for_residual

                history_list = attn_state.get("history", [None] * len(insert_after_list))
                history = history_list[layer_idx]
                x_in_for_layer = x_in

                # Prepare position embeddings for current tokens
                if pos is not None:
                    pos_current = pos.reshape(B, N, hw, -1).reshape(B, N * hw, -1)
                else:
                    pos_current = None

                if self._pca_debug_enabled():
                    swa_qkv = self._extract_pca_swa_current_qkv_patchvec(
                        self.swa_layers[layer_idx],
                        x_in_for_layer.reshape(B, N * hw, -1),
                        pos_current,
                        batch_size=B,
                        frame_num=N,
                        tokens_per_frame=hw,
                        patch_h=H // self.patch_size,
                        patch_w=W // self.patch_size,
                    )
                    if swa_qkv is not None:
                        if swa_qkv.get("q") is not None:
                            pca_swa_current_q_parts.append((i, swa_qkv["q"]))
                        if swa_qkv.get("k") is not None:
                            pca_swa_current_k_parts.append((i, swa_qkv["k"]))
                        if swa_qkv.get("v") is not None:
                            pca_swa_current_v_parts.append((i, swa_qkv["v"]))

                # Check if we have KV cache from history
                use_kv_cache = (
                    history is not None 
                    and isinstance(history, dict) 
                    and "k" in history
                )

                if use_kv_cache:
                    # Use KV cache path
                    k_cache = history["k"]  # [B, num_heads, N_hist * hw, head_dim]
                    v_cache = history["v"]  # [B, num_heads, N_hist * hw, head_dim]
                    # Forward with KV cache
                    x_curr_flat = x_in_for_layer.reshape(B, N * hw, -1)
                    history_tokens = int(k_cache.shape[2])
                    swa_attn_mask = None
                    swa_overlap_bias_stats: Dict[str, Any] = {
                        "swa_overlap_bias_applied": False,
                        "swa_overlap_bias_query_tokens": 0,
                        "swa_overlap_bias_source_tokens": 0,
                    }
                    swa_source_gate = None
                    swa_layer_enabled = False
                    if hmc_control and hmc_control.get("enable_swa_read_control", False):
                        swa_layer_mode = str(hmc_control.get("swa_layer_mode", "first"))
                        if swa_layer_mode in {"first", "first_swa_only"}:
                            swa_layer_enabled = layer_idx == 0
                        elif swa_layer_mode == "all":
                            swa_layer_enabled = True
                        elif swa_layer_mode == "single":
                            swa_layer_enabled = layer_idx == int(hmc_control.get("swa_single_layer", 0))
                    if swa_layer_enabled:
                        swa_source_gate = self._make_swa_prev_source_gate(
                            hmc_control,
                            history_tokens=history_tokens,
                            device=v_cache.device,
                            dtype=v_cache.dtype,
                        )
                    swa_overlap_layer_enabled = self._swa_overlap_layer_enabled(
                        hmc_control,
                        layer_idx=layer_idx,
                        n_layers=len(insert_after_list),
                    )
                    if swa_overlap_layer_enabled:
                        swa_attn_mask, swa_overlap_bias_stats = self._make_swa_overlap_attention_bias(
                            hmc_control,
                            batch_size=B,
                            frame_num=N,
                            tokens_per_frame=hw,
                            history_tokens=history_tokens,
                            current_tokens=int(N * hw),
                            device=x_curr_flat.device,
                            dtype=x_curr_flat.dtype,
                            swa_layer_idx=layer_idx,
                        )
                    d_prev = hmc_control.get("D_prev_patch") if hmc_control else None
                    d_prev_tokens = int(d_prev.numel()) if hasattr(d_prev, "numel") else 0
                    k_cache_controlled = k_cache
                    v_cache_controlled = v_cache
                    if swa_source_gate is not None:
                        v_cache_controlled = v_cache * swa_source_gate
                    swa_prev_ttt_anchor_gate = None
                    swa_prev_ttt_anchor_gate_stats: Dict[str, Any] = {
                        "swa_prev_ttt_stable_anchor_gate_applied": False,
                        "swa_prev_ttt_stable_anchor_gate_tokens": 0,
                        "swa_prev_ttt_stable_anchor_gate_available": False,
                    }
                    if self._swa_prev_ttt_stable_anchor_layer_enabled(
                        hmc_control,
                        layer_idx=layer_idx,
                        n_layers=len(insert_after_list),
                    ):
                        swa_prev_ttt_anchor_gate, swa_prev_ttt_anchor_gate_stats = (
                            self._make_swa_prev_ttt_stable_anchor_gate(
                                hmc_control,
                                batch_size=B,
                                tokens_per_frame=hw,
                                history_tokens=history_tokens,
                                device=v_cache.device,
                                dtype=v_cache.dtype,
                            )
                        )
                        if swa_prev_ttt_anchor_gate is not None:
                            target = str(hmc_control.get("swa_prev_ttt_stable_anchor_gate_target", "v"))
                            if target in {"v", "value", "kv", "both"}:
                                v_cache_controlled = v_cache_controlled * swa_prev_ttt_anchor_gate
                            if target in {"k", "key", "kv", "both"}:
                                k_cache_controlled = k_cache_controlled * swa_prev_ttt_anchor_gate.to(
                                    device=k_cache.device,
                                    dtype=k_cache.dtype,
                                )
                    swa_prev_ttt_anchor_query_soft_stats: Dict[str, Any] = {
                        "swa_prev_ttt_anchor_query_soft_available": False,
                        "swa_prev_ttt_anchor_query_soft_applied": False,
                        "swa_prev_ttt_anchor_query_soft_source_tokens": 0,
                    }
                    if self._swa_prev_ttt_anchor_query_soft_layer_enabled(
                        hmc_control,
                        layer_idx=layer_idx,
                        n_layers=len(insert_after_list),
                    ):
                        swa_prev_ttt_anchor_query_soft, swa_prev_ttt_anchor_query_soft_stats = (
                            self._make_swa_prev_ttt_anchor_query_soft_control(
                                hmc_control,
                                batch_size=B,
                                tokens_per_frame=hw,
                                history_tokens=history_tokens,
                                device=x_curr_flat.device,
                            )
                        )
                        if swa_prev_ttt_anchor_query_soft is not None:
                            if swa_attn_mask is None:
                                swa_attn_mask = swa_prev_ttt_anchor_query_soft
                            else:
                                swa_prev_ttt_anchor_query_soft_stats[
                                    "swa_prev_ttt_anchor_query_soft_reason"
                                ] = "incompatible_existing_swa_attn_mask"
                    swa_prev_ttt_tracked_instance_query_soft_stats: Dict[str, Any] = {
                        "swa_prev_ttt_tracked_instance_query_soft_available": False,
                        "swa_prev_ttt_tracked_instance_query_soft_applied": False,
                        "swa_prev_ttt_tracked_instance_query_soft_trace_only": True,
                        "swa_prev_ttt_tracked_instance_query_soft_action_requested": False,
                        "swa_prev_ttt_tracked_instance_query_soft_runtime_action_allowed": False,
                        "swa_prev_ttt_tracked_instance_query_soft_source_tokens": 0,
                    }
                    if self._swa_prev_ttt_tracked_instance_query_soft_layer_enabled(
                        hmc_control,
                        layer_idx=layer_idx,
                        n_layers=len(insert_after_list),
                    ):
                        (
                            swa_prev_ttt_tracked_instance_query_soft,
                            swa_prev_ttt_tracked_instance_query_soft_stats,
                        ) = self._make_swa_prev_ttt_tracked_instance_query_soft_trace_control(
                            hmc_control,
                            batch_size=B,
                            tokens_per_frame=hw,
                            history_tokens=history_tokens,
                            current_tokens=int(x_curr_flat.shape[1]),
                            device=x_curr_flat.device,
                        )
                        if swa_prev_ttt_tracked_instance_query_soft is not None:
                            if swa_attn_mask is None:
                                swa_attn_mask = swa_prev_ttt_tracked_instance_query_soft
                            else:
                                swa_prev_ttt_tracked_instance_query_soft_stats[
                                    "swa_prev_ttt_tracked_instance_query_soft_reason"
                                ] = "incompatible_existing_swa_attn_mask"
                    swa_overlap_source_gate = None
                    swa_overlap_source_gate_stats: Dict[str, Any] = {
                        "swa_overlap_source_gate_applied": False,
                        "swa_overlap_source_gate_tokens": 0,
                    }
                    if self._swa_overlap_source_layer_enabled(
                        hmc_control,
                        layer_idx=layer_idx,
                        n_layers=len(insert_after_list),
                    ):
                        swa_overlap_source_gate, swa_overlap_source_gate_stats = self._make_swa_overlap_source_gate(
                            hmc_control,
                            batch_size=B,
                            frame_num=N,
                            tokens_per_frame=hw,
                            history_tokens=history_tokens,
                            current_tokens=int(N * hw),
                            device=v_cache.device,
                            dtype=v_cache.dtype,
                            swa_layer_idx=layer_idx,
                        )
                        if swa_overlap_source_gate is not None:
                            target = str(hmc_control.get("swa_overlap_source_gate_target", "v"))
                            if target in {"v", "value", "kv", "both"}:
                                v_cache_controlled = v_cache_controlled * swa_overlap_source_gate
                            if target in {"k", "key", "kv", "both"}:
                                k_cache_controlled = k_cache * swa_overlap_source_gate.to(
                                    device=k_cache.device,
                                    dtype=k_cache.dtype,
                                )
                    swa_overlap_source_replace_stats: Dict[str, Any] = {
                        "swa_overlap_source_replace_applied": False,
                        "swa_overlap_source_replace_tokens": 0,
                    }
                    if self._swa_overlap_source_replace_layer_enabled(
                        hmc_control,
                        layer_idx=layer_idx,
                        n_layers=len(insert_after_list),
                    ):
                        source_replace, swa_overlap_source_replace_stats = self._make_swa_overlap_source_replace(
                            hmc_control,
                            batch_size=B,
                            frame_num=N,
                            tokens_per_frame=hw,
                            history_tokens=history_tokens,
                            current_tokens=int(N * hw),
                            device=v_cache.device,
                            dtype=v_cache.dtype,
                            swa_layer_idx=layer_idx,
                        )
                        if source_replace is not None:
                            source_start = int(source_replace["source_start"])
                            source_end = int(source_replace["source_end"])
                            source_tokens = int(source_replace["source_tokens"])
                            alpha = source_replace["alpha"]
                            if pos is not None:
                                pos_for_replace = pos.reshape(B, N, hw, -1)[:, :1].repeat(
                                    1, N, 1, 1
                                ).reshape(B, N * hw, -1)
                            else:
                                pos_for_replace = None
                            k_cur_cache, v_cur_cache = self.swa_layers[layer_idx].compute_kv_cache(
                                x_curr_flat,
                                xpos=pos_for_replace,
                            )
                            target = str(hmc_control.get("swa_overlap_source_replace_target", "kv"))

                            def _blend_source(cache_tensor: torch.Tensor, cur_tensor: torch.Tensor) -> torch.Tensor:
                                if source_end <= source_start or cur_tensor.shape[2] < source_tokens:
                                    return cache_tensor
                                out = cache_tensor.clone()
                                old = out[:, :, source_start:source_end, :]
                                cur = cur_tensor[:, :, :source_tokens, :].to(device=old.device, dtype=old.dtype)
                                a = alpha.to(device=old.device, dtype=old.dtype)
                                out[:, :, source_start:source_end, :] = old * (1.0 - a) + cur * a
                                return out

                            if target in {"v", "value", "kv", "both"}:
                                v_cache_controlled = _blend_source(v_cache_controlled, v_cur_cache)
                            if target in {"k", "key", "kv", "both"}:
                                k_cache_controlled = _blend_source(k_cache_controlled, k_cur_cache)
                    v102_state_machine_stats: Dict[str, Any] = {
                        "v102_swa_state_machine_trace_available": False,
                        "v102_swa_state_machine_trace_applied": False,
                    }
                    if self._v102_state_machine_trace_layer_enabled(
                        hmc_control,
                        layer_idx=layer_idx,
                        n_layers=len(insert_after_list),
                    ):
                        v102_attn_mask, v102_state_machine_stats = self._make_v102_state_machine_action_probe(
                            hmc_control,
                            batch_size=B,
                            tokens_per_frame=hw,
                            history_tokens=history_tokens,
                            current_tokens=int(N * hw),
                            device=x_curr_flat.device,
                            swa_layer_idx=layer_idx,
                        )
                        if v102_attn_mask is not None:
                            if swa_attn_mask is None:
                                swa_attn_mask = v102_attn_mask
                            else:
                                v102_state_machine_stats.update({
                                    "v102_swa_state_machine_trace_applied": False,
                                    "v102_swa_state_machine_scaffold_only": True,
                                    "v102_swa_state_machine_reason": "incompatible_existing_swa_attn_mask",
                                })
                    swa_raw_transport_stats: Dict[str, Any] = {}
                    if self._swa_raw_transport_trace_layer_enabled(
                        hmc_control,
                        layer_idx=layer_idx,
                        n_layers=len(insert_after_list),
                    ):
                        q_cur_trace, k_cur_trace, v_cur_trace = self.swa_layers[layer_idx].compute_qkv_cache(
                            x_curr_flat,
                            xpos=pos_current,
                        )
                        swa_raw_transport_stats = self._dump_swa_raw_transport_trace(
                            hmc_control,
                            layer=int(i),
                            swa_layer_idx=layer_idx,
                            batch_size=B,
                            frame_num=N,
                            tokens_per_frame=hw,
                            q_current=q_cur_trace,
                            k_current=k_cur_trace,
                            v_current=v_cur_trace,
                            k_cache=k_cache,
                            v_cache=v_cache,
                            hidden_current=x_curr_flat,
                            hidden_cache=history.get("hidden_pre") if isinstance(history, dict) else None,
                            extra_trace_fields=v102_state_machine_stats,
                        )
                    swa_trace_record: Optional[Dict[str, Any]] = None
                    if (
                        self._hmc_hook_requested(hmc_control, "enable_swa_read_control")
                        or self._hmc_hook_requested(hmc_control, "enable_swa_overlap_bias")
                        or self._hmc_hook_requested(hmc_control, "enable_swa_prev_ttt_stable_anchor_gate")
                        or self._hmc_hook_requested(hmc_control, "enable_swa_prev_ttt_anchor_query_soft")
                        or self._hmc_hook_requested(
                            hmc_control,
                            "enable_swa_prev_ttt_tracked_instance_query_soft_trace",
                        )
                        or self._hmc_hook_requested(
                            hmc_control,
                            "enable_swa_prev_ttt_tracked_instance_query_soft_action",
                        )
                        or self._hmc_hook_requested(hmc_control, "enable_swa_overlap_source_gate")
                        or self._hmc_hook_requested(hmc_control, "enable_swa_overlap_source_replace")
                        or self._hmc_hook_requested(hmc_control, "enable_v102_state_machine_trace")
                        or bool(swa_raw_transport_stats)
                    ):
                        gate_stats = {}
                        if swa_source_gate is not None:
                            gate_f = swa_source_gate.detach().float()
                            gate_stats = {
                                "source_gate_applied": True,
                                "d_prev_tokens": d_prev_tokens,
                                "source_pad_tokens": int(max(0, history_tokens - d_prev_tokens)),
                                "source_trim_tokens": int(max(0, d_prev_tokens - history_tokens)),
                                "swa_gate_mean": float(gate_f.mean().item()),
                                "swa_gate_p10": float(torch.quantile(gate_f, 0.10).item()),
                                "swa_gate_p50": float(torch.quantile(gate_f, 0.50).item()),
                                "swa_gate_p90": float(torch.quantile(gate_f, 0.90).item()),
                                "mean_abs_gate_delta": float((1.0 - gate_f).abs().mean().item()),
                                "max_abs_gate_delta": float((1.0 - gate_f).abs().max().item()),
                            }
                        else:
                            gate_stats = {
                                "source_gate_applied": False,
                                "d_prev_tokens": d_prev_tokens,
                                "source_pad_tokens": 0,
                                "source_trim_tokens": 0,
                                "swa_gate_mean": 1.0,
                                "swa_gate_p10": 1.0,
                                "swa_gate_p50": 1.0,
                                "swa_gate_p90": 1.0,
                                "mean_abs_gate_delta": 0.0,
                                "max_abs_gate_delta": 0.0,
                            }
                        swa_trace_record = {
                            "layer": int(i),
                            "swa_layer": int(layer_idx),
                            "identity": bool(hmc_control.get("identity_hooks", False)) if hmc_control else False,
                            "layer_enabled": bool(swa_layer_enabled),
                            "used_kv_cache": True,
                            "current_tokens": int(N * hw),
                            "history_tokens": history_tokens,
                            "attn_mask_applied": swa_attn_mask is not None,
                            "hook_site": "swa_kv_cache_read",
                            **gate_stats,
                            **swa_prev_ttt_anchor_gate_stats,
                            **swa_prev_ttt_anchor_query_soft_stats,
                            **swa_prev_ttt_tracked_instance_query_soft_stats,
                            **swa_overlap_bias_stats,
                            **swa_overlap_source_gate_stats,
                            **swa_overlap_source_replace_stats,
                            **v102_state_machine_stats,
                            **swa_raw_transport_stats,
                        }
                    swa_output_flat = self.swa_layers[layer_idx].forward_with_kv_cache(
                        x_curr_flat, k_cache_controlled, v_cache_controlled,
                        xpos=pos_current,
                        attn_mask=swa_attn_mask,
                    )
                    if (
                        swa_trace_record is not None
                        and isinstance(swa_attn_mask, dict)
                        and isinstance(swa_attn_mask.get("attention_mass_stats"), list)
                        and swa_attn_mask["attention_mass_stats"]
                    ):
                        swa_trace_record.update(swa_attn_mask["attention_mass_stats"][-1])
                    if swa_trace_record is not None:
                        self._append_hmc_trace(hmc_trace, "swa_read", swa_trace_record)
                    swa_output = swa_output_flat.reshape(B, N, hw, -1)
                else:
                    # Original path (no history or legacy format)
                    # Handle legacy history format (raw tensor instead of dict)
                    history_raw = history if history is not None and not isinstance(history, dict) else None

                    if history_raw is not None:
                        x_with_history = torch.cat([history_raw, x_in_for_layer], dim=1)
                    else:
                        x_with_history = x_in_for_layer

                    N_total = x_with_history.shape[1]
                    x_swa = x_with_history.reshape(B, N_total * hw, -1)

                    if pos is not None:
                        pos_swa = pos.reshape(B, N, hw, -1)
                        if history_raw is not None:
                            N_hist = history_raw.shape[1]
                            pos_hist = pos_swa[:, :1].repeat(1, N_hist, 1, 1)
                            pos_swa = torch.cat([pos_hist, pos_swa], dim=1)
                        pos_swa = pos_swa.reshape(B, N_total * hw, -1)
                    else:
                        pos_swa = None

                    swa_attn_mask = None
                    if (
                        self._hmc_hook_requested(hmc_control, "enable_swa_read_control")
                        or self._hmc_hook_requested(hmc_control, "enable_swa_overlap_bias")
                    ):
                        self._append_hmc_trace(hmc_trace, "swa_read", {
                            "layer": int(i),
                            "swa_layer": int(layer_idx),
                            "identity": bool(hmc_control.get("identity_hooks", False)) if hmc_control else False,
                            "layer_enabled": False,
                            "used_kv_cache": False,
                            "current_tokens": int(N * hw),
                            "history_tokens": int((N_total - N) * hw),
                            "attn_mask_applied": False,
                            "source_gate_applied": False,
                            "swa_gate_mean": 1.0,
                            "swa_gate_p10": 1.0,
                            "swa_gate_p50": 1.0,
                            "swa_gate_p90": 1.0,
                            "mean_abs_gate_delta": 0.0,
                            "max_abs_gate_delta": 0.0,
                            "swa_overlap_bias_applied": False,
                            "swa_overlap_bias_query_tokens": 0,
                            "swa_overlap_bias_source_tokens": 0,
                            "hook_site": "swa_full_read",
                        })
                    swa_output_full = self.swa_layers[layer_idx](
                        x_swa, 
                        xpos=pos_swa,
                        attn_mask=swa_attn_mask,
                    )
                    swa_output_full = swa_output_full.reshape(B, N_total, hw, x_in.shape[-1])
                    if history_raw is not None:
                        N_hist = history_raw.shape[1]
                        swa_output = swa_output_full[:, N_hist:, :, :]
                    else:
                        swa_output = swa_output_full

                gate_scale = torch.nn.functional.silu(self.swa_gate_projs[layer_idx](swa_output))
                if turn_off_swa: gate_scale = torch.zeros_like(gate_scale)
                attn_gate_scales.append(gate_scale)

                update_term = swa_output * gate_scale
                x_out_patch = update_term + x_for_residual
                x_out_patch_flat = x_out_patch.reshape(B, N * hw, -1)
                hidden = x_out_patch_flat.reshape(B * N, hw, -1)

                # Store KV cache for next window
                # Compute KV for current x_in with history_pe (since it will be history next time)
                if ttt_output_info is None:
                    ttt_output_info = {"history": [None] * len(insert_after_list)}
                elif "history" not in ttt_output_info:
                    ttt_output_info["history"] = [None] * len(insert_after_list)

                x_for_cache = x_in
                x_for_cache_flat = x_for_cache.reshape(B, N * hw, -1)
                
                # Position for cache: use first frame's position repeated (same as original logic)
                if pos is not None:
                    pos_for_cache = pos.reshape(B, N, hw, -1)[:, :1].repeat(1, N, 1, 1).reshape(B, N * hw, -1)
                else:
                    pos_for_cache = None

                k_new, v_new = self.swa_layers[layer_idx].compute_kv_cache(x_for_cache_flat, xpos=pos_for_cache)
                if self._pca_debug_enabled():
                    pca_swa_cache_k = self._pca_heads_to_patchvec(
                        k_new,
                        batch_size=B,
                        frame_num=N,
                        tokens_per_frame=hw,
                        patch_h=H // self.patch_size,
                        patch_w=W // self.patch_size,
                        layout="global",
                    )
                    pca_swa_cache_v = self._pca_heads_to_patchvec(
                        v_new,
                        batch_size=B,
                        frame_num=N,
                        tokens_per_frame=hw,
                        patch_h=H // self.patch_size,
                        patch_w=W // self.patch_size,
                        layout="global",
                    )
                    if pca_swa_cache_k is not None:
                        pca_swa_cache_k_parts.append((i, pca_swa_cache_k))
                    if pca_swa_cache_v is not None:
                        pca_swa_cache_v_parts.append((i, pca_swa_cache_v))

                history_entry = {"k": k_new, "v": v_new}
                if cache_ttt_primitives or (
                    hmc_control and str(hmc_control.get("swa_raw_transport_trace_dir", "") or "").strip()
                ):
                    history_entry["hidden_pre"] = x_for_cache_flat.detach()
                if hmc_control and hmc_control.get("swa_write_cache_store_post", False):
                    x_post_cache_flat = x_out_patch.reshape(B, N * hw, -1)
                    k_post, v_post = self.swa_layers[layer_idx].compute_kv_cache(
                        x_post_cache_flat,
                        xpos=pos_for_cache,
                    )
                    history_entry["k_post"] = k_post
                    history_entry["v_post"] = v_post
                
                if getattr(self, "detach_swa_history", False):
                    history_entry = {
                        key: value.detach() if torch.is_tensor(value) else value
                        for key, value in history_entry.items()
                    }
                
                ttt_output_info["history"][layer_idx] = history_entry

            if i+1 in [len(self.decoder)-1, len(self.decoder)]:
                final_output.append(hidden.reshape(B*N, hw, -1))

        avg_gate_scale = torch.tensor(0.0, device=hidden.device, dtype=torch.float32)
        avg_attn_gate_scale: Optional[torch.Tensor] = None
        if gate_scales:
            all_gate_scales = torch.cat([g.flatten() for g in gate_scales])
            if all_gate_scales.numel() > 0:
                avg_gate_scale = all_gate_scales.abs().mean()
        if attn_gate_scales:
            all_attn_gate_scales = torch.cat([g.flatten() for g in attn_gate_scales])
            if all_attn_gate_scales.numel() > 0:
                avg_attn_gate_scale = all_attn_gate_scales.abs().mean()

        if len(final_output) < 2:
            raise RuntimeError(
                f"Decoder expected to collect two final outputs but got {len(final_output)}."
            )

        avg_frame_prior = None
        if attn_prior_frame_parts:
            avg_frame_prior = torch.stack(attn_prior_frame_parts, dim=0).mean(dim=0)
            eye = torch.eye(
                avg_frame_prior.shape[-1],
                device=avg_frame_prior.device,
                dtype=torch.bool,
            ).unsqueeze(0)
            avg_frame_prior = avg_frame_prior.masked_fill(eye, 0.0)

        avg_dynamic_prior = None
        if feature_key_parts:
            avg_dynamic_prior = torch.stack(
                [part for _, part in feature_key_parts], dim=0,
            ).mean(dim=0)

        dyn4d_outputs = self._aggregate_dyn4d_from_global_stats(dyn4d_global_parts)
        dyn4d_patch = None
        dyn4d_qq_mean_patch = None
        dyn4d_qk_var_patch = None
        dyn4d_kk_mean_patch = None
        global_q_raw_patchvec = None
        global_k_raw_patchvec = None
        global_v_raw_patchvec = None
        global_q_raw_patchvec_layers = None
        global_k_raw_patchvec_layers = None
        global_v_raw_patchvec_layers = None
        dyn4d_global_layer_ids = None
        if dyn4d_outputs is not None:
            dyn4d_patch = dyn4d_outputs.get("dyn4d_patch")
            dyn4d_qq_mean_patch = dyn4d_outputs.get("dyn4d_qq_mean_patch")
            dyn4d_qk_var_patch = dyn4d_outputs.get("dyn4d_qk_var_patch")
            dyn4d_kk_mean_patch = dyn4d_outputs.get("dyn4d_kk_mean_patch")
            global_q_raw_patchvec = dyn4d_outputs.get("global_q_raw_patchvec")
            global_k_raw_patchvec = dyn4d_outputs.get("global_k_raw_patchvec")
            global_v_raw_patchvec = dyn4d_outputs.get("global_v_raw_patchvec")
            global_q_raw_patchvec_layers = dyn4d_outputs.get("global_q_raw_patchvec_layers")
            global_k_raw_patchvec_layers = dyn4d_outputs.get("global_k_raw_patchvec_layers")
            global_v_raw_patchvec_layers = dyn4d_outputs.get("global_v_raw_patchvec_layers")
            dyn4d_global_layer_ids = dyn4d_outputs.get("dyn4d_global_layer_ids")

        frame_attn_cosine_layer_ids = None
        frame_attn_cosine_query_layers = None
        frame_attn_cosine_key_layers = None
        if self.export_attn_debug and frame_attn_cosine_query_parts and frame_attn_cosine_key_parts:
            frame_attn_cosine_layer_ids = torch.tensor(
                [layer_id for layer_id, _ in frame_attn_cosine_query_parts],
                device=hidden.device,
                dtype=torch.long,
            )
            frame_attn_cosine_query_layers = torch.stack(
                [part for _, part in frame_attn_cosine_query_parts], dim=2,
            )
            frame_attn_cosine_key_layers = torch.stack(
                [part for _, part in frame_attn_cosine_key_parts], dim=2,
            )

        frame_attn_cosine_shallow = None
        frame_attn_cosine_deep = None
        frame_attn_cosine_avg = None
        frame_attn_key_cosine_shallow = None
        frame_attn_key_cosine_deep = None
        frame_attn_key_cosine_avg = None
        if self.export_attn_debug and frame_attn_cosine_query_parts and frame_attn_cosine_key_parts:
            query_by_layer = {layer_id: part for layer_id, part in frame_attn_cosine_query_parts}
            key_by_layer = {layer_id: part for layer_id, part in frame_attn_cosine_key_parts}
            selected_query_parts = [
                query_by_layer[layer_id]
                for layer_id in self.frame_attn_map_layers
                if layer_id in query_by_layer
            ]
            selected_key_parts = [
                key_by_layer[layer_id]
                for layer_id in self.frame_attn_map_layers
                if layer_id in key_by_layer
            ]
            if selected_query_parts:
                frame_attn_cosine_shallow = selected_query_parts[0]
                frame_attn_cosine_deep = selected_query_parts[-1]
                frame_attn_cosine_avg = torch.stack(selected_query_parts, dim=0).mean(dim=0)
            if selected_key_parts:
                frame_attn_key_cosine_shallow = selected_key_parts[0]
                frame_attn_key_cosine_deep = selected_key_parts[-1]
                frame_attn_key_cosine_avg = torch.stack(selected_key_parts, dim=0).mean(dim=0)

        def _stack_pca_parts(parts: List[Tuple[int, torch.Tensor]]) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
            if not parts:
                return None, None
            return (
                torch.stack([part for _, part in parts], dim=2),
                torch.tensor([int(layer_id) for layer_id, _ in parts], device=hidden.device, dtype=torch.long),
            )

        pca_debug_outputs: Dict[str, Any] = {}

        def _put_pca(name: str, parts: List[Tuple[int, torch.Tensor]], layer_key: Optional[str] = None) -> None:
            tensor, ids = _stack_pca_parts(parts)
            if tensor is None:
                return
            pca_debug_outputs[name] = tensor
            if layer_key and ids is not None and layer_key not in pca_debug_outputs:
                pca_debug_outputs[layer_key] = ids

        if self._pca_debug_enabled():
            pca_debug_outputs["pca_debug_schema"] = "loger_full_layer_qkv_pca_debug_v1"
            _put_pca("pca_attn_frame_q_layers", pca_attn_frame_q_parts, "pca_attn_frame_layer_ids")
            _put_pca("pca_attn_frame_k_layers", pca_attn_frame_k_parts, "pca_attn_frame_layer_ids")
            _put_pca("pca_attn_frame_v_layers", pca_attn_frame_v_parts, "pca_attn_frame_layer_ids")
            _put_pca("pca_attn_global_q_layers", pca_attn_global_q_parts, "pca_attn_global_layer_ids")
            _put_pca("pca_attn_global_k_layers", pca_attn_global_k_parts, "pca_attn_global_layer_ids")
            _put_pca("pca_attn_global_v_layers", pca_attn_global_v_parts, "pca_attn_global_layer_ids")
            _put_pca("pca_swa_current_q_layers", pca_swa_current_q_parts, "pca_swa_layer_ids")
            _put_pca("pca_swa_current_k_layers", pca_swa_current_k_parts, "pca_swa_layer_ids")
            _put_pca("pca_swa_current_v_layers", pca_swa_current_v_parts, "pca_swa_layer_ids")
            _put_pca("pca_swa_cache_k_layers", pca_swa_cache_k_parts, "pca_swa_layer_ids")
            _put_pca("pca_swa_cache_v_layers", pca_swa_cache_v_parts, "pca_swa_layer_ids")
            _put_pca("pca_ttt_q_layers", pca_ttt_q_parts, "pca_ttt_layer_ids")
            _put_pca("pca_ttt_k_layers", pca_ttt_k_parts, "pca_ttt_layer_ids")
            _put_pca("pca_ttt_v_layers", pca_ttt_v_parts, "pca_ttt_layer_ids")
            _put_pca("pca_ttt_input_layers", pca_ttt_input_parts, "pca_ttt_layer_ids")
            _put_pca("pca_ttt_apply_raw_layers", pca_ttt_apply_raw_parts, "pca_ttt_layer_ids")
            _put_pca("pca_ttt_operator_output_layers", pca_ttt_operator_output_parts, "pca_ttt_layer_ids")
            _put_pca("pca_ttt_update_term_layers", pca_ttt_update_term_parts, "pca_ttt_layer_ids")
            _put_pca("pca_ttt_final_output_layers", pca_ttt_final_output_parts, "pca_ttt_layer_ids")

        return (
            torch.cat([final_output[0], final_output[1]], dim=-1),
            (pos.reshape(B*N, hw, -1) if pos is not None else None),
            ttt_output_info,
            avg_gate_scale,
            avg_attn_gate_scale,
            gate_scales,
            avg_frame_prior,
            avg_dynamic_prior,
            dyn4d_patch,
            dyn4d_qq_mean_patch,
            dyn4d_qk_var_patch,
            dyn4d_kk_mean_patch,
            global_q_raw_patchvec,
            global_k_raw_patchvec,
            global_v_raw_patchvec,
            global_q_raw_patchvec_layers,
            global_k_raw_patchvec_layers,
            global_v_raw_patchvec_layers,
            dyn4d_global_layer_ids,
            frame_attn_cosine_shallow,
            frame_attn_cosine_deep,
            frame_attn_cosine_avg,
            frame_attn_key_cosine_l0,
            frame_attn_key_cosine_l4,
            frame_attn_key_cosine_shallow,
            frame_attn_key_cosine_deep,
            frame_attn_key_cosine_avg,
            frame_attn_cosine_query_layers,
            frame_attn_cosine_key_layers,
            frame_attn_cosine_layer_ids,
            hmc_trace,
            pca_debug_outputs,
        )
    
    def forward(self, imgs, *args, **kwargs):
        # Windowing controls (optional)
        window_size = kwargs.pop('window_size', -1)
        overlap_size = kwargs.pop('overlap_size', 1)
        num_iterations = kwargs.pop('num_iterations', 1)
        no_detach = kwargs.pop('no_detach', False)
        sim3 = kwargs.pop('sim3', False)
        se3 = kwargs.pop('se3', False)
        reset_every = kwargs.pop('reset_every', 0)
        turn_off_ttt = kwargs.pop('turn_off_ttt', False)
        turn_off_swa = kwargs.pop('turn_off_swa', False)
        sim3_scale_mode = kwargs.pop('sim3_scale_mode', 'median')
        sim3_reuse_reset_block = kwargs.pop('sim3_reuse_reset_block', False)
        cache_ttt_primitives = kwargs.pop('cache_ttt_primitives', False)
        return_ttt_state = kwargs.pop('return_ttt_state', False)
        offload_adaptive_state_to_cpu = kwargs.pop('offload_adaptive_state_to_cpu', False)
        hmc_control = kwargs.pop('hmc_control', None)
        ttt_state_input = kwargs.pop('ttt_state_input', None)
        swa_state_input = None
        if isinstance(ttt_state_input, dict):
            swa_state_input = ttt_state_input.get("history")

        if sim3 and se3:
            raise ValueError("'sim3' and 'se3' alignments are mutually exclusive; enable only one.")

        # Ensure at least one decode iteration so that 'hidden' is always defined
        try:
            num_iterations = int(num_iterations)
        except Exception:
            num_iterations = 1
        if num_iterations < 1:
            num_iterations = 1
        try:
            reset_every = int(reset_every)
        except Exception:
            reset_every = 0
        if reset_every < 0:
            reset_every = 0

        # Ensure batch dimension
        if imgs.dim() == 4:
            imgs = imgs.unsqueeze(0)

        # Normalize
        # imgs = (imgs - self.image_mean) / self.image_std

        B, N, C, H, W = imgs.shape
        patch_h, patch_w = H // 14, W // 14

        # --- Unified Windowed Inference ---
        if window_size <= 0 or window_size >= N:
            windows = [(0, N)]
            eff_overlap = 0
            eff_window_size = N
        else:
            windows = []
            step = max(window_size - overlap_size, 1)
            for start_idx in range(0, N, step):
                end_idx = min(start_idx + window_size, N)
                if end_idx - start_idx >= overlap_size or (end_idx == N and start_idx < N):
                    windows.append((start_idx, end_idx))
                if end_idx == N:
                    break
            eff_overlap = overlap_size
            eff_window_size = window_size

        # Cache the effective window and overlap sizes for downstream merging utilities
        self._last_window_size = eff_window_size
        self._last_overlap_size = eff_overlap

        # Prepare TTT states across windows — accept external W_m
        if self.ttt_layers is not None:
            if ttt_state_input is not None:
                w0 = ttt_state_input["w0"]
                w1 = ttt_state_input["w1"]
                w2 = ttt_state_input["w2"]
            else:
                w0 = [None] * len(self.ttt_insert_after)
                w1 = [None] * len(self.ttt_insert_after)
                w2 = [None] * len(self.ttt_insert_after)
        else:
            w0 = w1 = w2 = None

        # Prepare SWA history states across windows. When chunked inference is
        # driven externally, we can resume the KV cache from the previous chunk
        # through ``ttt_state_input['history']`` so the behavior matches the
        # original single-call window loop more closely.
        if self.swa_layers is not None:
            if swa_state_input is not None:
                swa_history = swa_state_input
            else:
                swa_history = [None] * len(self.attn_insert_after)
        else:
            swa_history = None

        def reset_adaptive_states():
            """Reset fast-weight TTT states only; SWA history is preserved across resets."""
            nonlocal w0, w1, w2
            if self.ttt_layers is not None:
                w0 = [None] * len(self.ttt_insert_after)
                w1 = [None] * len(self.ttt_insert_after)
                w2 = [None] * len(self.ttt_insert_after)

        def _move_weight_state(state_list, device):
            if state_list is None:
                return None
            moved = []
            for item in state_list:
                if item is None:
                    moved.append(None)
                else:
                    moved.append(item.to(device))
            return moved

        def _move_history_state(history_list, device):
            if history_list is None:
                return None
            moved = []
            for entry in history_list:
                if entry is None:
                    moved.append(None)
                else:
                    moved_entry = {
                        "k": entry["k"].to(device),
                        "v": entry["v"].to(device),
                    }
                    for key in ("k_post", "v_post", "hidden_pre"):
                        value = entry.get(key)
                        if value is not None:
                            moved_entry[key] = value.to(device)
                    moved.append(moved_entry)
            return moved

        all_predictions = []
        all_gate_scales: List[torch.Tensor] = []
        all_attn_gate_scales: List[torch.Tensor] = []
        
        windows_iter = windows
        for window_idx, (start_idx, end_idx) in enumerate(windows_iter):
            if reset_every > 0 and window_idx > 0 and window_idx % reset_every == 0:
                reset_adaptive_states()
            imgs_w = imgs[:, start_idx:end_idx]  # (B, Nw, C, H, W)
            imgs_w = imgs_w.to(self.image_mean.device)
            imgs_w = (imgs_w - self.image_mean) / self.image_std
            Nw = imgs_w.shape[1]

            # Initialize to satisfy static analyzers; will be set inside decode loop
            hidden = None  # type: ignore[assignment]
            pos = None     # type: ignore[assignment]
            ttt_output_info = None
            decode_avg_gate_scale = None
            decode_avg_attn_gate_scale = None
            _decode_gate_scales = None
            frame_attention_prior = None
            attn_dynamic_patch = None
            dyn4d_patch = None
            dyn4d_qq_mean_patch = None
            dyn4d_qk_var_patch = None
            dyn4d_kk_mean_patch = None
            global_q_raw_patchvec = None
            global_k_raw_patchvec = None
            global_q_raw_patchvec_layers = None
            global_k_raw_patchvec_layers = None
            dyn4d_global_layer_ids = None
            frame_attn_cosine_shallow = None
            frame_attn_cosine_deep = None
            frame_attn_cosine_avg = None
            frame_attn_key_cosine_l0 = None
            frame_attn_key_cosine_l4 = None
            frame_attn_key_cosine_shallow = None
            frame_attn_key_cosine_deep = None
            frame_attn_key_cosine_avg = None
            frame_attn_cosine_query_layers = None
            frame_attn_cosine_key_layers = None
            frame_attn_cosine_layer_ids = None
            hmc_trace = None
            pca_debug_outputs = None

            for _ in range(num_iterations):
                if self.ttt_layers is not None and w0 is None:
                    w0 = [None] * len(self.ttt_insert_after)
                    w1 = [None] * len(self.ttt_insert_after)
                    w2 = [None] * len(self.ttt_insert_after)

                if self.swa_layers is not None and swa_history is None:
                    swa_history = [None] * len(self.attn_insert_after)

                if offload_adaptive_state_to_cpu:
                    if self.ttt_layers is not None:
                        w0 = _move_weight_state(w0, self.image_mean.device)
                        w1 = _move_weight_state(w1, self.image_mean.device)
                        w2 = _move_weight_state(w2, self.image_mean.device)
                    if self.swa_layers is not None:
                        swa_history = _move_history_state(swa_history, self.image_mean.device)

                imgs_flat = imgs_w.reshape(B * Nw, C, H, W)
                hidden_input = self.encoder(imgs_flat, is_training=True)
                if isinstance(hidden_input, dict):
                    hidden_input = hidden_input["x_norm_patchtokens"]

                # Prepare adapter control dictionaries for decode
                ttt_state = None
                attn_state = None

                if self.ttt_layers is not None:
                    ttt_state = {
                        "ttt_op_order": self.ttt_op_order if self.ttt_op_order is not None else [],
                        "insert_after": self.ttt_insert_after,
                        "w0": w0,
                        "w1": w1,
                        "w2": w2,
                    }

                if self.swa_layers is not None:
                    attn_state = {
                        "insert_after": self.attn_insert_after,
                        "history": swa_history,
                    }

                if ttt_state is None and attn_state is None:
                    ttt_dict = None
                else:
                    ttt_dict = {
                        "ttt": ttt_state,
                        "attn": attn_state,
                    }
                hidden, pos, ttt_output_info, decode_avg_gate_scale, decode_avg_attn_gate_scale, _decode_gate_scales, frame_attention_prior, attn_dynamic_patch, dyn4d_patch, dyn4d_qq_mean_patch, dyn4d_qk_var_patch, dyn4d_kk_mean_patch, global_q_raw_patchvec, global_k_raw_patchvec, global_v_raw_patchvec, global_q_raw_patchvec_layers, global_k_raw_patchvec_layers, global_v_raw_patchvec_layers, dyn4d_global_layer_ids, frame_attn_cosine_shallow, frame_attn_cosine_deep, frame_attn_cosine_avg, frame_attn_key_cosine_l0, frame_attn_key_cosine_l4, frame_attn_key_cosine_shallow, frame_attn_key_cosine_deep, frame_attn_key_cosine_avg, frame_attn_cosine_query_layers, frame_attn_cosine_key_layers, frame_attn_cosine_layer_ids, hmc_trace, pca_debug_outputs = self.decode(
                    hidden_input, Nw, H, W,
                    ttt_dict=ttt_dict,
                    window_size=window_size,
                    overlap_size=overlap_size,
                    is_first_window=(start_idx == 0),
                    turn_off_ttt=turn_off_ttt,
                    turn_off_swa=turn_off_swa,
                    cache_ttt_primitives=cache_ttt_primitives,
                    hmc_control=hmc_control,
                )
                if decode_avg_gate_scale is not None:
                    all_gate_scales.append(decode_avg_gate_scale.detach().cpu())
                if decode_avg_attn_gate_scale is not None:
                    all_attn_gate_scales.append(decode_avg_attn_gate_scale.detach().cpu())

                # TODO: get the updated state from the ttt layer
                if self.ttt_layers is not None and ttt_output_info is not None:
                    w0, w1, w2 = ttt_output_info["w0"], ttt_output_info["w1"], ttt_output_info["w2"]
                
                # TODO: get the updated history from the swa layer
                if ttt_output_info is not None:
                    swa_history = ttt_output_info.get("history", swa_history)

                if offload_adaptive_state_to_cpu:
                    if self.ttt_layers is not None:
                        w0 = _move_weight_state(w0, "cpu")
                        w1 = _move_weight_state(w1, "cpu")
                        w2 = _move_weight_state(w2, "cpu")
                    if self.swa_layers is not None:
                        swa_history = _move_history_state(swa_history, "cpu")

            # If for some reason decoding didn't produce hidden (e.g., empty window), skip this window
            if hidden is None:
                continue

            point_hidden = self.point_decoder(hidden, xpos=pos)
            if self.use_conf and self.conf_decoder is not None:
                conf_hidden = self.conf_decoder(hidden, xpos=pos)
            else:
                conf_hidden = None
            
            if self.pi3x and self.pi3x_metric:
                hw = hidden.shape[1]
                pos_hw = pos.reshape(B, Nw*hw, -1)
                metric_hidden = self.metric_decoder(self.metric_token.repeat(B, 1, 1), hidden.reshape(B, Nw*hw, -1), xpos=pos_hw[:, 0:1], ypos=pos_hw)
            else:
                metric_hidden = None

            camera_hidden = self.camera_decoder(hidden, xpos=pos)

            global_camera_hidden = camera_hidden

            with torch.autocast(device_type='cuda', enabled=False):
                # local points
                point_hidden = point_hidden.float()
                if self.pi3x:
                    xy, z = self.point_head(point_hidden[:, self.patch_start_idx:], patch_h=patch_h, patch_w=patch_w)
                    xy = xy.permute(0, 2, 3, 1).reshape(B, Nw, H, W, -1)
                    z = z.permute(0, 2, 3, 1).reshape(B, Nw, H, W, -1)
                    z = torch.exp(z.clamp(max=15.0))
                    local_points = torch.cat([xy * z, z], dim=-1)
                else:
                    ret = self.point_head([point_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, Nw, H, W, -1)
                    xy, z = ret.split([2, 1], dim=-1)
                    z = torch.exp(z)
                    local_points = torch.cat([xy * z, z], dim=-1)

                # confidence
                if conf_hidden is not None and self.conf_head is not None:
                    conf_hidden = conf_hidden.float()
                    conf = self.conf_head([conf_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, Nw, H, W, -1)
                else:
                    conf = None

                # camera
                global_camera_hidden = global_camera_hidden.float()
                camera_poses = self.camera_head(global_camera_hidden[:, self.patch_start_idx:], patch_h, patch_w).reshape(B, Nw, 4, 4)
                camera_qvec = None
                local_camera_poses = None
                local_camera_qvec = None

                # metric
                if self.pi3x and self.pi3x_metric and metric_hidden is not None:
                    metric = self.metric_head(metric_hidden.float()).reshape(B).exp()
                    
                    # apply metric to points and camera poses
                    # points = torch.einsum('bnij, bnhwj -> bnhwi', camera_poses, homogenize_points(local_points))[..., :3] * metric.view(B, 1, 1, 1, 1)
                    camera_poses[..., :3, 3] = camera_poses[..., :3, 3] * metric.view(B, 1, 1)
                    local_points = local_points * metric.view(B, 1, 1, 1, 1)
                    if local_camera_poses is not None:
                        local_camera_poses[..., :3, 3] = local_camera_poses[..., :3, 3] * metric.view(B, 1, 1)
                else:
                    metric = None


            # unproject local points using camera poses
            with torch.autocast(device_type='cuda', enabled=False):
                points = torch.einsum('bnij, bnhwj -> bnhwi', camera_poses, homogenize_points(local_points))[..., :3]


            def maybe_detach(t, no_detach=no_detach):
                if t is None:
                    return None
                return t if self.training or no_detach else t.detach().cpu()

            pred_dict = dict(
                points=maybe_detach(points, no_detach=no_detach),
                local_points=maybe_detach(local_points, no_detach=no_detach),
                conf=maybe_detach(conf, no_detach=no_detach),
                camera_poses=maybe_detach(camera_poses, no_detach=no_detach),
                local_camera_poses=maybe_detach(local_camera_poses, no_detach=no_detach),
                camera_qvec=maybe_detach(camera_qvec, no_detach=no_detach),
                local_camera_qvec=maybe_detach(local_camera_qvec, no_detach=no_detach),
                metric=maybe_detach(metric, no_detach=no_detach),
                frame_attention_prior=maybe_detach(frame_attention_prior, no_detach=no_detach),
                attn_dynamic_patch=maybe_detach(attn_dynamic_patch, no_detach=no_detach),
                dyn4d_patch=maybe_detach(dyn4d_patch, no_detach=no_detach),
                dyn4d_qq_mean_patch=maybe_detach(dyn4d_qq_mean_patch, no_detach=no_detach),
                dyn4d_qk_var_patch=maybe_detach(dyn4d_qk_var_patch, no_detach=no_detach),
                dyn4d_kk_mean_patch=maybe_detach(dyn4d_kk_mean_patch, no_detach=no_detach),
                global_q_raw_patchvec=maybe_detach(global_q_raw_patchvec, no_detach=no_detach),
                global_k_raw_patchvec=maybe_detach(global_k_raw_patchvec, no_detach=no_detach),
                global_v_raw_patchvec=maybe_detach(global_v_raw_patchvec, no_detach=no_detach),
                global_q_raw_patchvec_layers=maybe_detach(global_q_raw_patchvec_layers, no_detach=no_detach),
                global_k_raw_patchvec_layers=maybe_detach(global_k_raw_patchvec_layers, no_detach=no_detach),
                global_v_raw_patchvec_layers=maybe_detach(global_v_raw_patchvec_layers, no_detach=no_detach),
                dyn4d_global_layer_ids=maybe_detach(dyn4d_global_layer_ids, no_detach=no_detach),
                frame_attn_cosine_shallow=maybe_detach(frame_attn_cosine_shallow, no_detach=no_detach),
                frame_attn_cosine_deep=maybe_detach(frame_attn_cosine_deep, no_detach=no_detach),
                frame_attn_cosine_avg=maybe_detach(frame_attn_cosine_avg, no_detach=no_detach),
                frame_attn_key_cosine_l0=maybe_detach(frame_attn_key_cosine_l0, no_detach=no_detach),
                frame_attn_key_cosine_l4=maybe_detach(frame_attn_key_cosine_l4, no_detach=no_detach),
                frame_attn_key_cosine_shallow=maybe_detach(frame_attn_key_cosine_shallow, no_detach=no_detach),
                frame_attn_key_cosine_deep=maybe_detach(frame_attn_key_cosine_deep, no_detach=no_detach),
                frame_attn_key_cosine_avg=maybe_detach(frame_attn_key_cosine_avg, no_detach=no_detach),
                frame_attn_cosine_query_layers=maybe_detach(frame_attn_cosine_query_layers, no_detach=no_detach),
                frame_attn_cosine_key_layers=maybe_detach(frame_attn_cosine_key_layers, no_detach=no_detach),
                frame_attn_cosine_layer_ids=maybe_detach(frame_attn_cosine_layer_ids, no_detach=no_detach),
                hmc_trace=hmc_trace,
                _window_start=start_idx,
                _window_end=end_idx,
            )
            if isinstance(pca_debug_outputs, dict):
                for key, value in pca_debug_outputs.items():
                    pred_dict[key] = maybe_detach(value, no_detach=no_detach) if torch.is_tensor(value) else value
            all_predictions.append(pred_dict)

            if not self.training:
                del imgs_w, imgs_flat, hidden_input, hidden, pos
                del point_hidden, conf_hidden, camera_hidden, global_camera_hidden
                del local_points, conf, camera_poses, points
                del decode_avg_gate_scale, decode_avg_attn_gate_scale, _decode_gate_scales
                del frame_attention_prior, attn_dynamic_patch
                del dyn4d_patch, dyn4d_qq_mean_patch, dyn4d_qk_var_patch, dyn4d_kk_mean_patch
                del global_q_raw_patchvec, global_k_raw_patchvec
                del global_q_raw_patchvec_layers, global_k_raw_patchvec_layers
                del dyn4d_global_layer_ids
                del frame_attn_cosine_shallow, frame_attn_cosine_deep, frame_attn_cosine_avg
                del frame_attn_key_cosine_l0, frame_attn_key_cosine_l4
                del frame_attn_key_cosine_shallow, frame_attn_key_cosine_deep, frame_attn_key_cosine_avg
                del frame_attn_cosine_query_layers, frame_attn_cosine_key_layers, frame_attn_cosine_layer_ids
                del hmc_trace
                del pca_debug_outputs
                if metric_hidden is not None:
                    del metric_hidden
                if camera_qvec is not None:
                    del camera_qvec
                if local_camera_poses is not None:
                    del local_camera_poses
                if local_camera_qvec is not None:
                    del local_camera_qvec
                if metric is not None:
                    del metric
                if offload_adaptive_state_to_cpu:
                    torch.cuda.empty_cache()

        # Merge windowed predictions
        # When reset is enabled but explicit Sim3/SE3 alignment is off, keep each reset block
        # in a stable rigid frame by applying one estimated transform per block.
        align_on_resets_without_explicit_pose = reset_every > 0 and not sim3 and not se3
        if sim3:
            merged = self._merge_windowed_predictions_sim3(
                all_predictions, 
                allow_scale=True, 
                scale_mode=sim3_scale_mode,
                reset_every=reset_every,
                reuse_transform_within_reset_block=bool(sim3_reuse_reset_block),
            )
        elif se3 or align_on_resets_without_explicit_pose:
            merged = self._merge_windowed_predictions_sim3(
                all_predictions, 
                allow_scale=False,
                reset_every=reset_every,
                reuse_transform_within_reset_block=align_on_resets_without_explicit_pose,
            )
        else:
            merged = self._merge_windowed_predictions(all_predictions, eff_window_size, eff_overlap)
        if all_gate_scales:
            merged["avg_gate_scale"] = torch.stack(all_gate_scales).mean()
        if all_attn_gate_scales:
            merged["attn_gate_scale"] = torch.stack(all_attn_gate_scales).mean()

        if (cache_ttt_primitives or return_ttt_state) and ttt_output_info is not None:
            merged["ttt_output_info"] = ttt_output_info

        return merged

    def _merge_windowed_predictions(self, all_predictions, window_size, overlap_size):
        """
        Merge predictions from multiple windows by concatenating along the time dimension
        while removing overlapping frames.
        """
        if not all_predictions:
            return {}
        if len(all_predictions) == 1:
            return all_predictions[0]

        merged_predictions = {}
        keys = list(all_predictions[0].keys())
        sequence_keys = {
            "points",
            "local_points",
            "conf",
            "camera_poses",
            "local_camera_poses",
            "camera_qvec",
            "local_camera_qvec",
            "attn_dynamic_patch",
            "dyn4d_patch",
            "dyn4d_qq_mean_patch",
            "dyn4d_qk_var_patch",
            "dyn4d_kk_mean_patch",
            "global_q_raw_patchvec",
            "global_k_raw_patchvec",
            "global_v_raw_patchvec",
            "global_q_raw_patchvec_layers",
            "global_k_raw_patchvec_layers",
            "global_v_raw_patchvec_layers",
            "frame_attn_cosine_shallow",
            "frame_attn_cosine_deep",
            "frame_attn_cosine_avg",
            "frame_attn_key_cosine_l0",
            "frame_attn_key_cosine_l4",
            "frame_attn_key_cosine_shallow",
            "frame_attn_key_cosine_deep",
            "frame_attn_key_cosine_avg",
            "frame_attn_cosine_query_layers",
            "frame_attn_cosine_key_layers",
        }
        for key in keys:
            # Collect window tensors
            window_tensors = [pred.get(key, None) for pred in all_predictions]

            # Skip if all windows have None for this key
            if all(t is None for t in window_tensors):
                continue

            # Only perform overlap-aware concatenation for known sequence-shaped tensors
            if key == "frame_attention_prior":
                merged_prior = self._merge_windowed_frame_priors(all_predictions, key)
                if merged_prior is not None:
                    merged_predictions[key] = merged_prior
            elif key in sequence_keys or (str(key).startswith("pca_") and str(key).endswith("_layers")):
                # Filter out None windows safely while preserving positions for slicing
                result_parts = []

                # First window: drop last overlap_size frames
                first = window_tensors[0]
                if first is not None:
                    if overlap_size > 0 and first.shape[1] > overlap_size:
                        result_parts.append(first[:, :-overlap_size])
                    elif overlap_size > 0 and first.shape[1] <= overlap_size:
                        # If window shorter or equal to overlap, drop completely
                        pass
                    else:
                        result_parts.append(first)

                # Middle windows: drop last overlap_size frames
                for tensor in window_tensors[1:-1]:
                    if tensor is None:
                        continue
                    if overlap_size > 0 and tensor.shape[1] > overlap_size:
                        result_parts.append(tensor[:, :-overlap_size])
                    elif overlap_size > 0 and tensor.shape[1] <= overlap_size:
                        # If window shorter or equal to overlap, drop completely
                        continue
                    else:
                        result_parts.append(tensor)

                # Last window: keep all frames
                last_tensor = window_tensors[-1]
                if last_tensor is not None:
                    result_parts.append(last_tensor)

                if result_parts:
                    merged_predictions[key] = torch.cat(result_parts, dim=1)
                else:
                    # Fallback: if everything was dropped due to tiny windows, keep last non-None
                    for t in reversed(window_tensors):
                        if t is not None:
                            merged_predictions[key] = t
                            break
            else:
                # Non-sequence keys: keep the last non-None
                for t in reversed(window_tensors):
                    if t is not None:
                        merged_predictions[key] = t
                        break

        # Instead of computing overlap losses here, export overlap prev/next tensors for trainer-side chunk losses
        if overlap_size > 0 and len(all_predictions) > 1:
            prev_cam_chunks = []
            next_cam_chunks = []
            prev_pcd_chunks = []
            next_pcd_chunks = []
            next_conf_chunks = []

            for i in range(len(all_predictions) - 1):
                pred_a = all_predictions[i]
                pred_b = all_predictions[i + 1]

                cam_a = pred_a.get("camera_poses", None)
                cam_b = pred_b.get("camera_poses", None)
                lpts_a = pred_a.get("local_points", None)
                lpts_b = pred_b.get("local_points", None)
                conf_a = pred_a.get("conf", None)
                conf_b = pred_b.get("conf", None)

                # Only collect when both sides have enough frames for a full overlap window
                if cam_a is not None and cam_b is not None and cam_a.shape[1] >= overlap_size and cam_b.shape[1] >= overlap_size:
                    S_a = cam_a.shape[1]
                    # Take last overlap_size from A and first overlap_size from B
                    prev_cam_chunks.append(cam_a[:, S_a - overlap_size: S_a])  # (B, O, 4, 4)
                    next_cam_chunks.append(cam_b[:, 0: overlap_size])         # (B, O, 4, 4)

                if lpts_a is not None and lpts_b is not None and lpts_a.shape[1] >= overlap_size and lpts_b.shape[1] >= overlap_size:
                    S_a = lpts_a.shape[1]
                    prev_pcd_chunks.append(lpts_a[:, S_a - overlap_size: S_a])  # (B, O, H, W, 3)
                    next_pcd_chunks.append(lpts_b[:, 0: overlap_size])          # (B, O, H, W, 3)
                    if conf_b is not None and conf_b.shape[1] >= overlap_size:
                        next_conf_chunks.append(conf_b[:, 0: overlap_size].squeeze(-1))  # (B, O, H, W)

            # Stack along a new chunk dimension if any collected
            if prev_cam_chunks and next_cam_chunks:
                merged_predictions["overlap_prev_cam"] = torch.stack(prev_cam_chunks, dim=1)  # (B, K, O, 4, 4)
                merged_predictions["overlap_next_cam"] = torch.stack(next_cam_chunks, dim=1)  # (B, K, O, 4, 4)
            if prev_pcd_chunks and next_pcd_chunks:
                merged_predictions["overlap_prev_pcd"] = torch.stack(prev_pcd_chunks, dim=1)  # (B, K, O, H, W, 3)
                merged_predictions["overlap_next_pcd"] = torch.stack(next_pcd_chunks, dim=1)  # (B, K, O, H, W, 3)
                if next_conf_chunks:
                    merged_predictions["overlap_next_conf"] = torch.stack(next_conf_chunks, dim=1)  # (B, K, O, H, W)

        return merged_predictions

    def _merge_windowed_frame_priors(self, all_predictions, key: str) -> Optional[torch.Tensor]:
        """Merge per-window [B, T_w, T_w] frame priors into [B, T, T]."""
        priors = []
        starts = []
        ends = []
        for pred in all_predictions:
            prior = pred.get(key, None)
            start = pred.get("_window_start", None)
            end = pred.get("_window_end", None)
            if prior is None or start is None or end is None:
                continue
            priors.append(prior)
            starts.append(int(start))
            ends.append(int(end))

        if not priors:
            return None

        batch_size = priors[0].shape[0]
        total_frames = max(ends)
        device = priors[0].device
        dtype = priors[0].dtype

        merged = torch.zeros(batch_size, total_frames, total_frames, device=device, dtype=dtype)
        counts = torch.zeros(1, total_frames, total_frames, device=device, dtype=dtype)

        for prior, start, end in zip(priors, starts, ends):
            length = end - start
            if prior.shape[-2:] != (length, length):
                continue
            merged[:, start:end, start:end] += prior
            counts[:, start:end, start:end] += 1.0

        valid = counts > 0
        merged = torch.where(valid, merged / counts.clamp_min(1.0), merged)
        eye = torch.eye(total_frames, device=device, dtype=torch.bool).unsqueeze(0)
        merged = merged.masked_fill(eye, 0.0)
        return merged

    def _merge_windowed_predictions_sim3(
        self,
        all_predictions,
        allow_scale: bool = True,
        scale_mode: str = 'median',
        reset_every: int = 0,
        reuse_transform_within_reset_block: bool = False,
    ):
        """
        Merge windowed predictions by estimating relative poses between overlaps.
        When ``allow_scale`` is True this performs Sim(3) alignment (scale+SE(3));
        when False it reduces to SE(3) alignment by keeping the scale fixed to 1.
        If ``reuse_transform_within_reset_block`` is enabled with ``reset_every > 0``,
        one transform is estimated at each reset boundary and reused for the rest of
        that reset block.
        """
        # print("allow_scale -----------------------------", allow_scale)
        if not all_predictions:
            return {}
        if len(all_predictions) == 1:
            return all_predictions[0]

        # Locate a reference tensor to determine batch/device/dtype information
        sample_tensor = None
        for pred in all_predictions:
            for key in ("points", "camera_poses", "local_points", "conf"):
                tensor = pred.get(key, None)
                if tensor is not None:
                    sample_tensor = tensor
                    break
            if sample_tensor is not None:
                break
        if sample_tensor is None:
            raise ValueError("Sim3 merge requires at least one tensor prediction")

        device = sample_tensor.device
        dtype = sample_tensor.dtype
        batch_size = sample_tensor.shape[0]

        identity_rot = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).repeat(batch_size, 1, 1)
        zero_trans = torch.zeros(batch_size, 3, device=device, dtype=dtype)
        one_scale = torch.ones(batch_size, device=device, dtype=dtype)

        aligned_predictions: List[dict] = []
        sim3_scales: Optional[List[torch.Tensor]] = [] if allow_scale else None
        sim3_poses: List[torch.Tensor] = []

        window_size = getattr(self, "_last_window_size", -1)
        overlap_size = getattr(self, "_last_overlap_size", 0)

        def _estimate_relative_sim3(prev_aligned: dict, curr_raw: dict, overlap: int, current_allow_scale: bool, forced_scale: Optional[torch.Tensor] = None):
            if overlap <= 0:
                return torch.ones_like(one_scale), identity_rot, zero_trans

            prev_cam = prev_aligned.get("camera_poses", None)
            curr_cam = curr_raw.get("camera_poses", None)
            if prev_cam is None or curr_cam is None or prev_cam.shape[1] == 0 or curr_cam.shape[1] == 0:
                return torch.ones_like(one_scale), identity_rot, zero_trans

            prev_frames = prev_cam.shape[1]
            prev_idx = max(prev_frames - overlap, 0)

            prev_pose = prev_cam[:, prev_idx]
            curr_pose = curr_cam[:, 0]

            R_prev = prev_pose[:, :3, :3]
            t_prev = prev_pose[:, :3, 3]
            R_curr = curr_pose[:, :3, :3]
            t_curr = curr_pose[:, :3, 3]

            relative_rot = torch.matmul(R_prev, R_curr.transpose(-1, -2))

            relative_scale = torch.ones_like(one_scale)
            if forced_scale is not None:
                relative_scale = forced_scale
            elif current_allow_scale:
                prev_local_raw = prev_aligned.get("local_points", None)
                if prev_local_raw is None:
                    prev_local_raw = prev_aligned.get("_local_points_raw", None)
                curr_local_raw = curr_raw.get("local_points", None)

                if (
                    prev_local_raw is not None
                    and curr_local_raw is not None
                    and prev_local_raw.shape[1] > prev_idx
                    and curr_local_raw.shape[1] > 0
                ):
                    if scale_mode in ['median_all', 'trimmed_mean_all']:
                        # Use all overlapping frames
                        actual_overlap = min(overlap, prev_local_raw.shape[1] - prev_idx, curr_local_raw.shape[1])
                        if actual_overlap > 0:
                            prev_depth = prev_local_raw[:, prev_idx : prev_idx + actual_overlap, ..., 2]
                            curr_depth = curr_local_raw[:, :actual_overlap, ..., 2]
                        else:
                            # Fallback to single frame if overlap calculation fails (should not happen given checks above)
                            prev_depth = prev_local_raw[:, prev_idx, ..., 2]
                            curr_depth = curr_local_raw[:, 0, ..., 2]
                    else:
                        # Use only the first overlapping frame (standard behavior)
                        prev_depth = prev_local_raw[:, prev_idx, ..., 2]
                        curr_depth = curr_local_raw[:, 0, ..., 2]

                    prev_depth_f32 = prev_depth.to(torch.float32)
                    curr_depth_f32 = curr_depth.to(torch.float32)
                    eps_depth = torch.finfo(torch.float32).eps
                    valid = (
                        torch.isfinite(prev_depth_f32)
                        & torch.isfinite(curr_depth_f32)
                        & (curr_depth_f32.abs() > eps_depth)
                    )

                    prev_depth_flat = prev_depth_f32.reshape(batch_size, -1)
                    curr_depth_flat = curr_depth_f32.reshape(batch_size, -1)
                    valid_flat = valid.reshape(batch_size, -1)
                    
                    if scale_mode in ['median', 'median_all']:
                        scale_values = []
                        for b in range(batch_size):
                            valid_idx = valid_flat[b]
                            if valid_idx.any():
                                ratios = prev_depth_flat[b, valid_idx] / curr_depth_flat[b, valid_idx]
                                scale_values.append(ratios.median())
                            else:
                                scale_values.append(torch.tensor(1.0, device=device, dtype=torch.float32))
                        relative_scale = torch.stack(scale_values).to(dtype)
                    elif scale_mode in ['trimmed_mean', 'trimmed_mean_all']:
                        # Vectorized implementation for trimmed mean
                        # Mask invalid entries with NaN or filter before passing?
                        # robust_scale_estimation expects (B, N)
                        # Since N varies per batch due to validity, we might still need a loop or careful padding.
                        # However, valid_flat is (B, N_pixels).
                        
                        # To keep it simple and consistent with the median loop structure for now (which handles varying valid counts per batch):
                        scale_values = []
                        for b in range(batch_size):
                            valid_idx = valid_flat[b]
                            if valid_idx.any():
                                ratios = prev_depth_flat[b, valid_idx] / curr_depth_flat[b, valid_idx]
                                # ratios is 1D tensor of valid pixels
                                # We need to pass (1, N) to robust_scale_estimation to reuse it, or just use it directly if we modify it to handle 1D
                                # robust_scale_estimation expects (B, N). Let's reshape.
                                scale_val = robust_scale_estimation(ratios.unsqueeze(0), trim_ratio=0.25).squeeze(0)
                                scale_values.append(scale_val)
                            else:
                                scale_values.append(torch.tensor(1.0, device=device, dtype=torch.float32))
                        relative_scale = torch.stack(scale_values).to(dtype)
                    elif scale_mode in ['sim3_avg1']:
                        scale_values = []
                        for b in range(batch_size):
                            valid_idx = valid_flat[b]
                            if valid_idx.any():
                                ratios = prev_depth_flat[b, valid_idx] / curr_depth_flat[b, valid_idx]
                                scale_values.append(ratios.median())
                            else:
                                scale_values.append(torch.tensor(1.0, device=device, dtype=torch.float32))
                        relative_scale = torch.stack(scale_values).to(dtype)
                        relative_scale = (relative_scale + 1.0) / 2.0
                    else:
                        raise ValueError(f"Unknown scale_mode: {scale_mode}")

                    relative_scale = torch.clamp(relative_scale, min=1e-3, max=1e3)

            rotated_curr_centers = torch.matmul(relative_rot, t_curr.unsqueeze(-1)).squeeze(-1)
            relative_trans = t_prev - relative_scale.unsqueeze(-1) * rotated_curr_centers

            return relative_scale, relative_rot.to(dtype), relative_trans.to(dtype)

        block_scale: Optional[torch.Tensor] = None
        block_rot: Optional[torch.Tensor] = None
        block_trans: Optional[torch.Tensor] = None

        for window_idx, pred in enumerate(all_predictions):
            if window_idx == 0:
                current_scale = torch.ones_like(one_scale)
                current_rot = identity_rot.clone()
                current_trans = zero_trans.clone()
                if reuse_transform_within_reset_block and reset_every > 0:
                    block_scale = current_scale.clone()
                    block_rot = current_rot.clone()
                    block_trans = current_trans.clone()
            else:
                prev_aligned = aligned_predictions[-1]
                reuse_block_transform = (
                    reuse_transform_within_reset_block
                    and reset_every > 0
                    and window_idx % reset_every != 0
                    and block_rot is not None
                    and block_trans is not None
                )
                if reuse_block_transform:
                    current_rot = block_rot.clone()
                    current_trans = block_trans.clone()
                    if allow_scale and block_scale is not None:
                        current_scale = block_scale.clone()
                    else:
                        current_scale = torch.ones_like(one_scale)
                else:
                    current_scale, current_rot, current_trans = _estimate_relative_sim3(
                        prev_aligned, pred, overlap_size, allow_scale
                    )
                    if reuse_transform_within_reset_block and reset_every > 0:
                        block_scale = current_scale.clone()
                        block_rot = current_rot.clone()
                        block_trans = current_trans.clone()

            if allow_scale and sim3_scales is not None:
                sim3_scales.append(current_scale.clone())
                # print(current_scale, 'current_scale-----------------')
            pose_mat = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(batch_size, 1, 1)
            pose_mat[:, :3, :3] = current_rot
            pose_mat[:, :3, 3] = current_trans
            sim3_poses.append(pose_mat)

            aligned_pred: dict = {}

            original_local_points = pred.get("local_points", None)
            aligned_pred["_local_points_raw"] = original_local_points

            if original_local_points is not None:
                if allow_scale: # Keep using global allow_scale for applying scale if we have it, or maybe we should track per-window scale application?
                    # Actually, current_scale will be 1.0 if current_allow_scale was False.
                    # So we can just always apply current_scale.
                    scale_factor = current_scale.view(batch_size, 1, 1, 1, 1)
                    aligned_local_points = original_local_points * scale_factor
                else:
                    aligned_local_points = original_local_points
            else:
                aligned_local_points = None
            aligned_pred["local_points"] = aligned_local_points

            def _transform_camera(cam_tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
                if cam_tensor is None:
                    return None
                frames = cam_tensor.shape[1]
                rot_local = cam_tensor[..., :3, :3]
                trans_local = cam_tensor[..., :3, 3]
                rot_global = torch.matmul(
                    current_rot.unsqueeze(1).expand(-1, frames, -1, -1),
                    rot_local
                )
                rotated_trans = torch.matmul(
                    current_rot.unsqueeze(1).expand(-1, frames, -1, -1),
                    trans_local.unsqueeze(-1)
                ).squeeze(-1)
                if allow_scale:
                    rotated_trans = rotated_trans * current_scale.view(batch_size, 1, 1)
                trans_global = rotated_trans + current_trans.unsqueeze(1)
                cam_out = cam_tensor.clone()
                cam_out[..., :3, :3] = rot_global
                cam_out[..., :3, 3] = trans_global
                return cam_out

            camera_global = _transform_camera(pred.get("camera_poses", None))
            aligned_pred["camera_poses"] = camera_global

            local_camera_global = _transform_camera(pred.get("local_camera_poses", None))
            aligned_pred["local_camera_poses"] = local_camera_global

            if camera_global is not None and aligned_local_points is not None:
                aligned_points = torch.einsum(
                    'bnij, bnhwj -> bnhwi',
                    camera_global,
                    homogenize_points(aligned_local_points)
                )[..., :3]
            else:
                points = pred.get("points", None)
                if points is not None:
                    rotated_points = torch.einsum('bij, bnhwj -> bnhwi', current_rot, points)
                    if allow_scale:
                        rotated_points = rotated_points * current_scale.view(batch_size, 1, 1, 1, 1)
                    aligned_points = rotated_points + current_trans.view(batch_size, 1, 1, 1, 3)
                else:
                    aligned_points = None
            aligned_pred["points"] = aligned_points

            aligned_pred["conf"] = pred.get("conf", None)

            for key, value in pred.items():
                if key in aligned_pred:
                    continue
                aligned_pred[key] = value

            aligned_predictions.append(aligned_pred)

        aligned_predictions_clean = []
        for pred in aligned_predictions:
            cleaned = pred.copy()
            cleaned.pop("_local_points_raw", None)
            aligned_predictions_clean.append(cleaned)

        merged = self._merge_windowed_predictions(aligned_predictions_clean, window_size, overlap_size)

        pose_key = "chunk_sim3_poses" if allow_scale else "chunk_se3_poses"
        if allow_scale and sim3_scales:
            merged["chunk_sim3_scales"] = torch.stack(sim3_scales, dim=1)
        if sim3_poses:
            merged[pose_key] = torch.stack(sim3_poses, dim=1)
        merged["alignment_mode"] = "sim3" if allow_scale else "se3"

        return merged
