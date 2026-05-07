import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from copy import deepcopy
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
        """Export raw patch-level q/k vectors from one global-attention layer."""
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
        return {
            "q_raw_patchvec": q_raw.mean(dim=1).reshape(batch_size, frame_num, patch_h, patch_w, -1),
            "k_raw_patchvec": k_raw.mean(dim=1).reshape(batch_size, frame_num, patch_h, patch_w, -1),
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
        return {
            "dyn4d_patch": dyn4d_norm.reshape_as(dyn4d_raw).clamp(0.0, 1.0),
            "dyn4d_qq_mean_patch": qq_mean,
            "dyn4d_qk_var_patch": qk_var_norm,
            "dyn4d_kk_mean_patch": kk_mean,
            "global_q_raw_patchvec": global_q_raw.float(),
            "global_k_raw_patchvec": global_k_raw.float(),
            "global_q_raw_patchvec_layers": global_q_raw_layers.permute(0, 2, 1, 3, 4, 5).contiguous().float(),
            "global_k_raw_patchvec_layers": global_k_raw_layers.permute(0, 2, 1, 3, 4, 5).contiguous().float(),
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
        if mode == "all":
            return True
        if mode == "single":
            return int(layer) == int(hmc_control.get("read_single_layer", -1))
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

    def _make_frame_attention_bias(
        self,
        hmc_control: Optional[Dict[str, Any]],
        *,
        batch_size: int,
        frame_num: int,
        tokens_per_frame: int,
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
        mode = str(hmc_control.get("frame_bias_mode", "pair"))
        if mode == "key":
            keep = (1.0 - Dk).expand(-1, D.shape[1], -1)
        elif mode == "query":
            # A uniform per-query attention-logit shift cancels under softmax;
            # query weakening is handled as an output gate below instead.
            return None
        else:
            keep = 1.0 - (1.0 - Dq) * Dk
        keep = keep.clamp_min(1e-4)
        return (beta * torch.log(keep)).to(dtype=dtype).unsqueeze(1)

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
        Dq_c = Dq.clamp(0.0, 1.0)
        Ds_c = Ds.clamp(0.0, 1.0)
        if mode == "source":
            keep = (1.0 - Ds_c).unsqueeze(1).expand(batch_size, qn, sn)
        elif mode == "union":
            keep = 1.0 - torch.maximum(Dq_c.unsqueeze(-1), Ds_c.unsqueeze(1))
        elif mode == "intersection":
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
        Dq = Dq.clamp(0.0, 1.0)
        Ds = Ds.clamp(0.0, 1.0)
        if mode in {"source", "prev", "previous"}:
            score = Ds
        elif mode in {"current", "query"}:
            score = Dq
        elif mode == "union":
            score = torch.maximum(Dq, Ds)
        elif mode in {"intersection", "inter"}:
            score = torch.minimum(Dq, Ds)
        elif mode in {"disagreement", "mismatch"}:
            score = (Dq - Ds).abs()
        elif mode in {"agree_dyn", "product"}:
            score = Dq * Ds
        else:
            raise ValueError(f"Unsupported SWA overlap source gate mode: {mode}")

        min_gate = min(max(float(hmc_control.get("swa_overlap_source_gate_min", 0.85)), 0.0), 1.0)
        gate_slice = (1.0 - rho * score).clamp(min_gate, 1.0).to(dtype=dtype)
        gate = torch.ones(batch_size, 1, history_tokens, 1, device=device, dtype=dtype)
        gate[:, :, source_start:source_end, :] = gate_slice.reshape(batch_size, 1, source_tokens, 1)
        gate_delta = (1.0 - gate_slice.detach().float()).abs()
        stats.update({
            "swa_overlap_source_gate_applied": True,
            "swa_overlap_source_gate_mode": mode,
            "swa_overlap_source_gate_rho": rho,
            "swa_overlap_source_gate_min": min_gate,
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
        Dq = Dq.clamp(0.0, 1.0)
        Ds = Ds.clamp(0.0, 1.0)
        if mode in {"source", "prev", "previous"}:
            score = Ds
        elif mode in {"current", "query"}:
            score = Dq
        elif mode == "union":
            score = torch.maximum(Dq, Ds)
        elif mode in {"intersection", "inter"}:
            score = torch.minimum(Dq, Ds)
        elif mode in {"disagreement", "mismatch"}:
            score = (Dq - Ds).abs()
        elif mode in {"agree_dyn", "product"}:
            score = Dq * Ds
        else:
            raise ValueError(f"Unsupported SWA overlap source replace mode: {mode}")

        alpha = (alpha_max * score).clamp(0.0, min(alpha_max, 1.0)).to(dtype=dtype)
        alpha_delta = alpha.detach().float()
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
            if hmc_attn_path == "frame_attention":
                layer_enabled = self._hmc_read_layer_enabled(hmc_control, layer=i, total_layers=total_decoder_layers)
                if layer_enabled:
                    attn_mask = self._make_frame_attention_bias(
                        hmc_control,
                        batch_size=B,
                        frame_num=N,
                        tokens_per_frame=hw,
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
                # Dense chunk-attention bias is intentionally not materialized
                # for identity/G2.  Non-identity chunk control should use a
                # sparse/block implementation before Phase C.
                layer_enabled = self._hmc_read_layer_enabled(hmc_control, layer=i, total_layers=total_decoder_layers)
                hook_key = "enable_chunk_read_control"

            if self._hmc_hook_requested(hmc_control, hook_key):
                mean_abs_bias = 0.0
                max_abs_bias = 0.0
                if attn_mask is not None and torch.is_tensor(attn_mask):
                    bias_abs = attn_mask.detach().float().abs()
                    mean_abs_bias = float(bias_abs.mean().item()) if bias_abs.numel() else 0.0
                    max_abs_bias = float(bias_abs.max().item()) if bias_abs.numel() else 0.0
                mean_abs_query_gate_delta = 0.0
                max_abs_query_gate_delta = 0.0
                if frame_query_gate is not None:
                    gate_delta = (1.0 - frame_query_gate.detach().float()).abs()
                    mean_abs_query_gate_delta = float(gate_delta.mean().item()) if gate_delta.numel() else 0.0
                    max_abs_query_gate_delta = float(gate_delta.max().item()) if gate_delta.numel() else 0.0
                self._append_hmc_trace(hmc_trace, hmc_attn_path, {
                    "layer": int(i),
                    "identity": bool(hmc_control.get("identity_hooks", False)) if hmc_control else False,
                    "layer_enabled": bool(layer_enabled),
                    "shape": [int(x) for x in hidden_for_block.shape],
                    "attn_mask_applied": attn_mask is not None,
                    "query_gate_applied": frame_query_gate is not None,
                    "mean_abs_bias": mean_abs_bias,
                    "max_abs_bias": max_abs_bias,
                    "mean_abs_query_gate_delta": mean_abs_query_gate_delta,
                    "max_abs_query_gate_delta": max_abs_query_gate_delta,
                    "hook_site": "decoder_block_attn",
                })

            hidden = blk(hidden_for_block, xpos=pos_for_block, attn_mask=attn_mask)
            if frame_query_gate is not None:
                hidden = hidden_before_block + (hidden - hidden_before_block) * frame_query_gate

            if ttt_state is not None and i in ttt_state.get("insert_after", []):
                assert self.ttt_gate_projs is not None and self.ttt_layers is not None
                insert_after_list = ttt_state.get("insert_after", [])
                layer_idx = insert_after_list.index(i)

                x_for_residual = hidden.view(B, N, hw, -1)
                tokens_post = x_for_residual
                tokens_in = tokens_post

                gate_scale = torch.nn.functional.silu(self.ttt_gate_projs[layer_idx](tokens_in))
                if turn_off_ttt: gate_scale = torch.zeros_like(gate_scale)
                gate_scales.append(gate_scale)
                info = {
                    "ttt_op_order": ttt_state.get("ttt_op_order", []),
                    "w0": ttt_state["w0"][layer_idx],
                    "w1": ttt_state["w1"][layer_idx],
                    "w2": ttt_state["w2"][layer_idx],
                }
                ttt_output, output = self.ttt_layers[layer_idx](
                    tokens_in, info, cache_primitives=cache_ttt_primitives,
                )

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
                        )
                    d_prev = hmc_control.get("D_prev_patch") if hmc_control else None
                    d_prev_tokens = int(d_prev.numel()) if hasattr(d_prev, "numel") else 0
                    k_cache_controlled = k_cache
                    v_cache_controlled = v_cache
                    if swa_source_gate is not None:
                        v_cache_controlled = v_cache * swa_source_gate
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
                    if (
                        self._hmc_hook_requested(hmc_control, "enable_swa_read_control")
                        or self._hmc_hook_requested(hmc_control, "enable_swa_overlap_bias")
                        or self._hmc_hook_requested(hmc_control, "enable_swa_overlap_source_gate")
                        or self._hmc_hook_requested(hmc_control, "enable_swa_overlap_source_replace")
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
                        self._append_hmc_trace(hmc_trace, "swa_read", {
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
                            **swa_overlap_bias_stats,
                            **swa_overlap_source_gate_stats,
                            **swa_overlap_source_replace_stats,
                        })
                    swa_output_flat = self.swa_layers[layer_idx].forward_with_kv_cache(
                        x_curr_flat, k_cache_controlled, v_cache_controlled,
                        xpos=pos_current,
                        attn_mask=swa_attn_mask,
                    )
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

                history_entry = {"k": k_new, "v": v_new}
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
        global_q_raw_patchvec_layers = None
        global_k_raw_patchvec_layers = None
        dyn4d_global_layer_ids = None
        if dyn4d_outputs is not None:
            dyn4d_patch = dyn4d_outputs.get("dyn4d_patch")
            dyn4d_qq_mean_patch = dyn4d_outputs.get("dyn4d_qq_mean_patch")
            dyn4d_qk_var_patch = dyn4d_outputs.get("dyn4d_qk_var_patch")
            dyn4d_kk_mean_patch = dyn4d_outputs.get("dyn4d_kk_mean_patch")
            global_q_raw_patchvec = dyn4d_outputs.get("global_q_raw_patchvec")
            global_k_raw_patchvec = dyn4d_outputs.get("global_k_raw_patchvec")
            global_q_raw_patchvec_layers = dyn4d_outputs.get("global_q_raw_patchvec_layers")
            global_k_raw_patchvec_layers = dyn4d_outputs.get("global_k_raw_patchvec_layers")
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
            global_q_raw_patchvec_layers,
            global_k_raw_patchvec_layers,
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
                    moved.append({
                        "k": entry["k"].to(device),
                        "v": entry["v"].to(device),
                    })
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
                hidden, pos, ttt_output_info, decode_avg_gate_scale, decode_avg_attn_gate_scale, _decode_gate_scales, frame_attention_prior, attn_dynamic_patch, dyn4d_patch, dyn4d_qq_mean_patch, dyn4d_qk_var_patch, dyn4d_kk_mean_patch, global_q_raw_patchvec, global_k_raw_patchvec, global_q_raw_patchvec_layers, global_k_raw_patchvec_layers, dyn4d_global_layer_ids, frame_attn_cosine_shallow, frame_attn_cosine_deep, frame_attn_cosine_avg, frame_attn_key_cosine_l0, frame_attn_key_cosine_l4, frame_attn_key_cosine_shallow, frame_attn_key_cosine_deep, frame_attn_key_cosine_avg, frame_attn_cosine_query_layers, frame_attn_cosine_key_layers, frame_attn_cosine_layer_ids, hmc_trace = self.decode(
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
                global_q_raw_patchvec_layers=maybe_detach(global_q_raw_patchvec_layers, no_detach=no_detach),
                global_k_raw_patchvec_layers=maybe_detach(global_k_raw_patchvec_layers, no_detach=no_detach),
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
            "global_q_raw_patchvec_layers",
            "global_k_raw_patchvec_layers",
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
            elif key in sequence_keys:
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
