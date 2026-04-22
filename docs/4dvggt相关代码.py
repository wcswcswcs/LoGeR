from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange
from torch.utils.checkpoint import checkpoint

from vggt4d.layers.block import BlockFor4D
from vggt.models.aggregator import Aggregator, slice_expand_and_flatten


class AggregatorFor4D(Aggregator):
    def __init__(self, **kwargs):
        kwargs["block_fn"] = BlockFor4D
        super().__init__(**kwargs)

    def forward(self, images: torch.Tensor,
                dyn_masks: Optional[torch.Tensor] = None,
                enable_memory_saving: bool = True) -> Tuple[List[torch.Tensor], int]:
        """
        Args:
            images (torch.Tensor): Input images with shape [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
            dyn_masks (torch.Tensor): Dynamic masks with shape [B, S, H, W], in range [0, 1].

        Returns:
            (list[torch.Tensor], int):
                The list of outputs from the attention blocks,
                and the patch_start_idx indicating where patch tokens begin.
        """
        B, S, C_in, H, W = images.shape

        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")

        # Normalize images and reshape for patch embed
        images = (images - self._resnet_mean) / self._resnet_std

        # Reshape to [B*S, C, H, W] for patch embedding
        images = images.view(B * S, C_in, H, W)
        patch_tokens = self.patch_embed(images)

        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        _, P, C = patch_tokens.shape

        if dyn_masks is not None:
            dyn_masks = F.max_pool2d(
                dyn_masks.float(), kernel_size=self.patch_size, stride=self.patch_size)
            dyn_masks = rearrange(dyn_masks, "b s h w -> b s (h w)") > 0.5
            # dyn_masks[:, 0] = False
            # set patch tokens to 0 if dyn_masks is true
            # bad effect
            # print("Masking patch tokens")
            # patch_tokens[rearrange(dyn_masks, "b s n -> (b s) n")] = 0

        # Expand camera and register tokens to match batch size and sequence length
        camera_token = slice_expand_and_flatten(self.camera_token, B, S)
        register_token = slice_expand_and_flatten(self.register_token, B, S)

        # Concatenate special tokens with patch tokens
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)

        pos = None
        if self.rope is not None:
            pos = self.position_getter(
                B * S, H // self.patch_size, W // self.patch_size, device=images.device)

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = torch.zeros(
                B * S, self.patch_start_idx, 2).to(images.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        # update P because we added special tokens
        _, P, C = tokens.shape

        frame_idx = 0
        global_idx = 0
        output_list = [None] * (self.aa_block_num * B)
        global_q_list = []
        frame_q_list = []
        global_k_list = []
        frame_k_list = []
        preserve_layer_idx = [4, 11, 17, 23]

        for i in range(self.aa_block_num):
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, frame_intermediates, frame_q, frame_k = self._process_frame_attention(
                        tokens, B, S, P, C, frame_idx, pos=pos, dyn_masks=dyn_masks,
                    )
                    frame_q_list.append(frame_q.detach().cpu())
                    frame_k_list.append(frame_k.detach().cpu())
                    del frame_q, frame_k
                elif attn_type == "global":
                    tokens, global_idx, global_intermediates, global_q, global_k = self._process_global_attention(
                        tokens, B, S, P, C, global_idx, pos=pos, dyn_masks=dyn_masks,
                    )
                    global_q_list.append(global_q.detach().cpu())
                    global_k_list.append(global_k.detach().cpu())
                    del global_q, global_k
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            for j in range(len(frame_intermediates)):
                # concat frame and global intermediates, [B x S x P x 2C]
                concat_inter = torch.cat(
                    [frame_intermediates[j], global_intermediates[j]], dim=-1)
                output_list[i * B + j] = concat_inter

            if enable_memory_saving:
                if i not in preserve_layer_idx:
                    for j in range(B):
                        tmp = output_list[i * B + j]
                        output_list[i * B + j] = None
                        del tmp
                del concat_inter, frame_intermediates, global_intermediates

        global_q = torch.stack(global_q_list, dim=0)
        global_k = torch.stack(global_k_list, dim=0)
        frame_q = torch.stack(frame_q_list, dim=0)
        frame_k = torch.stack(frame_k_list, dim=0)

        if enable_memory_saving:
            del tokens

        qk_dict = {
            "global_q": global_q,
            "global_k": global_k,
            "frame_q": frame_q,
            "frame_k": frame_k
        }

        if "concat_inter" in locals():
            del concat_inter
        if "frame_intermediates" in locals():
            del frame_intermediates
        if "global_intermediates" in locals():
            del global_intermediates

        if enable_memory_saving:
            self.clear_inference_cache()

        return output_list, self.patch_start_idx, qk_dict, patch_tokens

    def clear_inference_cache(self):
        if hasattr(self, "rope") and self.rope is not None:
            if hasattr(self.rope, "frequency_cache"):
                self.rope.frequency_cache.clear()
        if hasattr(self, "position_getter") and self.position_getter is not None:
            if hasattr(self.position_getter, "position_cache"):
                self.position_getter.position_cache.clear()
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    def _process_frame_attention(self, tokens, B, S, P, C, frame_idx, pos=None, dyn_masks: Optional[torch.Tensor] = None):
        """
        Process frame attention blocks. We keep tokens in shape (B*S, P, C).
        """
        # If needed, reshape tokens or positions:
        if tokens.shape != (B * S, P, C):
            tokens = tokens.view(B, S, P, C).view(B * S, P, C)

        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.view(B, S, P, 2).view(B * S, P, 2)

        intermediates = []
        attn_q = []
        attn_k = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            if self.training:
                tokens, q, k = checkpoint(
                    self.frame_blocks[frame_idx], tokens, pos, use_reentrant=self.use_reentrant)
            else:
                tokens, q, k = self.frame_blocks[frame_idx](
                    tokens, pos=pos, is_frame_attn=True, layer_id=frame_idx, dyn_masks=dyn_masks)
            frame_idx += 1
            intermediates.append(tokens.view(B, S, P, C))
            attn_q.append(q)
            attn_k.append(k)

        attn_q = torch.stack(attn_q, dim=0)
        attn_k = torch.stack(attn_k, dim=0)
        return tokens, frame_idx, intermediates, attn_q, attn_k

    def _process_global_attention(self, tokens, B, S, P, C, global_idx, pos=None, dyn_masks: Optional[torch.Tensor] = None):
        """
        Process global attention blocks. We keep tokens in shape (B, S*P, C).
        """
        if tokens.shape != (B, S * P, C):
            tokens = tokens.view(B, S, P, C).view(B, S * P, C)

        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.view(B, S, P, 2).view(B, S * P, 2)

        intermediates = []
        attn_q = []
        attn_k = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            if self.training:
                tokens, q, k = checkpoint(
                    self.global_blocks[global_idx], tokens, pos, use_reentrant=self.use_reentrant)
            else:
                tokens, q, k = self.global_blocks[global_idx](
                    tokens, pos=pos, is_frame_attn=False, layer_id=global_idx, dyn_masks=dyn_masks)
            global_idx += 1
            intermediates.append(tokens.view(B, S, P, C))
            attn_q.append(q)
            attn_k.append(k)

        attn_q = torch.stack(attn_q, dim=0)
        attn_k = torch.stack(attn_k, dim=0)
        return tokens, global_idx, intermediates, attn_q, attn_k



import numpy as np
import torch
from einops import rearrange
from skimage.filters import threshold_multiotsu
from sklearn.cluster import KMeans
from tqdm import tqdm


def extract_mean1_map(ref_id, global_q: torch.Tensor, global_k: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
    # mean q_ref_q_src 3-8
    window = torch.tensor([-6, -4, -2, 2, 4, 6])
    n_img = global_q.shape[0]
    img_h, img_w = images.shape[-2:]
    n_h, n_w = img_h // 14, img_w // 14

    src_ids = ref_id + window

    src_ids = src_ids[src_ids >= 0]
    src_ids = src_ids[src_ids < n_img]
    # print(src_ids)

    layer_ids = torch.arange(3, 8)

    q_ref = global_q[ref_id]
    k_ref = global_k[ref_id]
    # print(q_ref.shape)
    # print(k_ref.shape)
    q_ref = q_ref.unsqueeze(0)[:, layer_ids]
    k_ref = k_ref.unsqueeze(0)[:, layer_ids]

    q_src = global_q[src_ids]
    k_src = global_k[src_ids]
    # print(q_src.shape)
    # print(k_src.shape)
    q_src = q_src[:, layer_ids]
    k_src = k_src[:, layer_ids]
    # print(q_src.shape)
    # print(k_src.shape)

    attn_map = q_ref @ q_src.transpose(-2, -1)
    # print(attn_map.shape)
    attn_map = rearrange(
        attn_map, "n_img n_layer n_head (n_h n_w) n_tok -> n_h n_w (n_layer n_head) n_img n_tok", n_h=n_h, n_w=n_w)
    # print(attn_map.shape)
    attn_map = attn_map.mean(dim=(2, 3, 4))
    attn_min = attn_map.min()
    attn_max = attn_map.max()

    attn_map = (attn_map - attn_min) / (attn_max - attn_min + 1e-6)
    # print(attn_map.shape)

    return attn_map


def extract_spacial_var1_map(ref_id, global_q: torch.Tensor, global_k: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
    # spacial std q_ref_q_src 18-20
    window = torch.tensor([-6, -4, -2, 2, 4, 6])
    n_img = global_q.shape[0]
    img_h, img_w = images.shape[-2:]
    n_h, n_w = img_h // 14, img_w // 14

    src_ids = ref_id + window

    src_ids = src_ids[src_ids >= 0]
    src_ids = src_ids[src_ids < n_img]
    # print(src_ids)

    layer_ids = torch.arange(18, 20)

    q_ref = global_q[ref_id]
    k_ref = global_k[ref_id]
    # print(q_ref.shape)
    # print(k_ref.shape)
    q_ref = q_ref.unsqueeze(0)[:, layer_ids]
    k_ref = k_ref.unsqueeze(0)[:, layer_ids]

    q_src = global_q[src_ids]
    k_src = global_k[src_ids]
    # print(q_src.shape)
    # print(k_src.shape)
    q_src = q_src[:, layer_ids]
    k_src = k_src[:, layer_ids]
    # print(q_src.shape)
    # print(k_src.shape)

    attn_map = q_ref @ q_src.transpose(-2, -1)
    # print(attn_map.shape)
    attn_map = rearrange(
        attn_map, "n_img n_layer n_head (n_h n_w) n_tok -> n_h n_w (n_layer n_head) n_img n_tok", n_h=n_h, n_w=n_w)
    # print(attn_map.shape)
    attn_map = attn_map.mean(dim=(2, 3)).std(dim=-1)
    attn_min = attn_map.min()
    attn_max = attn_map.max()

    attn_map = (attn_map - attn_min) / (attn_max - attn_min + 1e-6)
    # print(attn_map.shape)

    return attn_map


def extract_mean2_map(ref_id, global_q: torch.Tensor, global_k: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
    # mean q_ref_q_src 17-22
    window = torch.tensor([-6, -4, -2, 2, 4, 6])
    n_img = global_q.shape[0]
    img_h, img_w = images.shape[-2:]
    n_h, n_w = img_h // 14, img_w // 14

    src_ids = ref_id + window

    src_ids = src_ids[src_ids >= 0]
    src_ids = src_ids[src_ids < n_img]
    # print(src_ids)

    layer_ids = torch.arange(17, 22)

    q_ref = global_q[ref_id]
    k_ref = global_k[ref_id]
    # print(q_ref.shape)
    # print(k_ref.shape)
    q_ref = q_ref.unsqueeze(0)[:, layer_ids]
    k_ref = k_ref.unsqueeze(0)[:, layer_ids]

    q_src = global_q[src_ids]
    k_src = global_k[src_ids]
    # print(q_src.shape)
    # print(k_src.shape)
    q_src = q_src[:, layer_ids]
    k_src = k_src[:, layer_ids]
    # print(q_src.shape)
    # print(k_src.shape)

    attn_map = q_ref @ q_src.transpose(-2, -1)
    # print(attn_map.shape)
    attn_map = rearrange(
        attn_map, "n_img n_layer n_head (n_h n_w) n_tok -> n_h n_w (n_layer n_head) n_img n_tok", n_h=n_h, n_w=n_w)
    # print(attn_map.shape)
    attn_map = attn_map.mean(dim=(2, 3, 4))
    attn_min = attn_map.min()
    attn_max = attn_map.max()

    attn_map = (attn_map - attn_min) / (attn_max - attn_min + 1e-6)
    # print(attn_map.shape)

    return attn_map


def extract_mean3_map(ref_id, global_q: torch.Tensor, global_k: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
    # mean k_ref_k_src 0-1
    window = torch.tensor([-6, -4, -2, 2, 4, 6])
    n_img = global_q.shape[0]
    img_h, img_w = images.shape[-2:]
    n_h, n_w = img_h // 14, img_w // 14

    src_ids = ref_id + window

    src_ids = src_ids[src_ids >= 0]
    src_ids = src_ids[src_ids < n_img]
    # print(src_ids)

    layer_ids = torch.arange(0, 1)

    q_ref = global_q[ref_id]
    k_ref = global_k[ref_id]
    # print(q_ref.shape)
    # print(k_ref.shape)
    q_ref = q_ref.unsqueeze(0)[:, layer_ids]
    k_ref = k_ref.unsqueeze(0)[:, layer_ids]

    q_src = global_q[src_ids]
    k_src = global_k[src_ids]
    # print(q_src.shape)
    # print(k_src.shape)
    q_src = q_src[:, layer_ids]
    k_src = k_src[:, layer_ids]
    # print(q_src.shape)
    # print(k_src.shape)

    attn_map = k_ref @ k_src.transpose(-2, -1)
    # print(attn_map.shape)
    attn_map = rearrange(
        attn_map, "n_img n_layer n_head (n_h n_w) n_tok -> n_h n_w (n_layer n_head) n_img n_tok", n_h=n_h, n_w=n_w)
    # print(attn_map.shape)
    attn_map = attn_map.mean(dim=(2, 3, 4))
    attn_min = attn_map.min()
    attn_max = attn_map.max()

    attn_map = (attn_map - attn_min) / (attn_max - attn_min + 1e-6)
    # print(attn_map.shape)

    return attn_map


def extract_spacial_var3_map(ref_id, global_q: torch.Tensor, global_k: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
    # spacial std q_ref_k_src 0-1
    window = torch.tensor([-6, -4, -2, 2, 4, 6])
    n_img = global_q.shape[0]
    img_h, img_w = images.shape[-2:]
    n_h, n_w = img_h // 14, img_w // 14

    src_ids = ref_id + window

    src_ids = src_ids[src_ids >= 0]
    src_ids = src_ids[src_ids < n_img]
    # print(src_ids)

    layer_ids = torch.arange(0, 1)

    q_ref = global_q[ref_id]
    k_ref = global_k[ref_id]
    # print(q_ref.shape)
    # print(k_ref.shape)
    q_ref = q_ref.unsqueeze(0)[:, layer_ids]
    k_ref = k_ref.unsqueeze(0)[:, layer_ids]

    q_src = global_q[src_ids]
    k_src = global_k[src_ids]
    # print(q_src.shape)
    # print(k_src.shape)
    q_src = q_src[:, layer_ids]
    k_src = k_src[:, layer_ids]
    # print(q_src.shape)
    # print(k_src.shape)

    attn_map = q_ref @ k_src.transpose(-2, -1)
    # print(attn_map.shape)
    attn_map = rearrange(
        attn_map, "n_img n_layer n_head (n_h n_w) n_tok -> n_h n_w (n_layer n_head) n_img n_tok", n_h=n_h, n_w=n_w)
    # print(attn_map.shape)
    attn_map = attn_map.mean(dim=(2, 3)).std(dim=-1)
    attn_min = attn_map.min()
    attn_max = attn_map.max()

    attn_map = (attn_map - attn_min) / (attn_max - attn_min + 1e-6)
    # print(attn_map.shape)

    return attn_map


@torch.no_grad()
def extract_dyn_map(qk_dict: dict, images: torch.Tensor) -> torch.Tensor:
    dyn_maps = []
    n_img = images.shape[0]
    print(f"Extracting dynamic maps for {n_img} images")
    global_q = qk_dict["global_tok_q"].to("cuda")
    global_k = qk_dict["global_tok_k"].to("cuda")
    global_cam_q = qk_dict["global_cam_q"].to("cuda")
    for ref_id in tqdm(range(n_img)):
        mean1_map = extract_mean1_map(ref_id, global_q, global_k, images)
        mean2_map = extract_mean2_map(ref_id, global_q, global_k, images)
        mean3_map = extract_mean3_map(ref_id, global_q, global_k, images)
        var1_map = extract_spacial_var1_map(ref_id, global_q, global_k, images)
        var3_map = extract_spacial_var3_map(ref_id, global_q, global_k, images)

        dyn_map = (1 - mean1_map) * (1 - var1_map) * \
            (mean2_map) * (1 - mean3_map) * (var3_map)

        dyn_map_min = dyn_map.min()
        dyn_map_max = dyn_map.max()

        dyn_map = (dyn_map - dyn_map_min) / (dyn_map_max - dyn_map_min + 1e-6)
        dyn_maps.append(dyn_map)

    dyn_maps = torch.stack(dyn_maps)
    return dyn_maps.detach().cpu()


@torch.no_grad()
def batch_extract_dyn_map(qk_dict: dict, images: torch.Tensor) -> torch.Tensor:
    n_img, _, h_img, w_img = images.shape

    global_tok_q = qk_dict["global_tok_q"]
    global_tok_k = qk_dict["global_tok_k"]
    global_cam_q = qk_dict["global_cam_q"]

    n_batch = 50
    n_pad = 8
    dyn_maps = []
    for start_idx in range(0, n_img, n_batch):
        end_idx = min(start_idx + n_batch, n_img)
        b_start_idx = max(start_idx - n_pad, 0)
        b_end_idx = min(end_idx + n_pad, n_img)
        b_images = images[b_start_idx:b_end_idx]
        b_global_tok_q = global_tok_q[b_start_idx:b_end_idx]
        b_global_tok_k = global_tok_k[b_start_idx:b_end_idx]
        b_global_cam_q = global_cam_q[b_start_idx:b_end_idx]

        b_qk_dict = {
            "global_tok_q": b_global_tok_q,
            "global_tok_k": b_global_tok_k,
            "global_cam_q": b_global_cam_q,
        }
        b_dyn_maps = extract_dyn_map(b_qk_dict, b_images)
        b_dyn_mask_idx = torch.arange(
            start_idx - b_start_idx, end_idx - b_start_idx)
        dyn_map = b_dyn_maps[b_dyn_mask_idx]
        dyn_maps.append(dyn_map)

    dyn_maps = torch.cat(dyn_maps, dim=0)
    return dyn_maps


@torch.no_grad()
def cluster_attention_maps(feature, dynamic_map, n_clusters=64):
    """use KMeans to cluster the attention maps using feature

    Args:
        feature: encoder feature [B,H,W,C]
        dynamic_map: dynamic_map feature [B,H,W]
        n_clusters: number of clusters

    Returns:
        normalized_map: normalized cluster map [B,H,W]
        cluster_labels: reshaped cluster labels [B,H,W]
    """
    # data preprocessing
    B, H, W, C = feature.shape
    feature_np = feature.cpu().numpy()
    flattened_feature = feature_np.reshape(-1, C)

    # KMeans clustering
    clusterer = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = clusterer.fit_predict(flattened_feature)

    # calculate the average dynamic score for each cluster
    dynamic_map_np = dynamic_map.cpu().numpy()
    flattened_dynamic = dynamic_map_np.reshape(-1)
    cluster_dynamic_scores = np.zeros(n_clusters)
    for i in range(n_clusters):
        cluster_mask = (cluster_labels == i)
        cluster_dynamic_scores[i] = np.mean(flattened_dynamic[cluster_mask])

    # map the cluster labels to the dynamic score
    cluster_map = cluster_dynamic_scores[cluster_labels]
    normalized_map = cluster_map.reshape(B, H, W)

    # reshape cluster_labels
    reshaped_labels = cluster_labels.reshape(B, H, W)

    # convert to torch tensor
    normalized_map = torch.from_numpy(normalized_map).float()
    cluster_labels = torch.from_numpy(reshaped_labels).long()

    normalized_map_min = normalized_map.min(dim=1, keepdim=True)[
        0].min(dim=2, keepdim=True)[0]
    normalized_map_max = normalized_map.max(dim=1, keepdim=True)[
        0].max(dim=2, keepdim=True)[0]
    normalized_map = (normalized_map - normalized_map_min) / \
        (normalized_map_max - normalized_map_min + 1e-6)

    return normalized_map, cluster_labels


def adaptive_multiotsu_variance(img, verbose=False):
    """adaptive multi-threshold Otsu algorithm based on inter-class variance maximization

    Args:
        img: input image array
        verbose: whether to print detailed information

    Returns:
        tuple: (best threshold, best number of classes)
    """
    max_classes = 4
    best_score = -float('inf')
    best_threshold = None
    best_n_classes = None
    scores = {}

    for n_classes in range(2, max_classes + 1):
        thresholds = threshold_multiotsu(img, classes=n_classes)

        regions = np.digitize(img, bins=thresholds)
        var_between = np.var([img[regions == i].mean()
                             for i in range(n_classes)])

        score = var_between / np.sqrt(n_classes)
        scores[n_classes] = score

        if score > best_score:
            best_score = score
            best_threshold = thresholds[-1]
            best_n_classes = n_classes

    if verbose:
        print("number of classes score:")
        for n_classes, score in scores.items():
            print(f"number of classes {n_classes}: score {score:.4f}" +
                  (" (best)" if n_classes == best_n_classes else ""))
        print(f"final selected number of classes: {best_n_classes}")

    return best_threshold
#--------------------------------------------------
import cv2
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
from einops import einsum, rearrange, repeat
from sklearn.cluster import KMeans
from tqdm import tqdm


def inverse_project(depth: torch.Tensor,
                    intrinsics: torch.Tensor,
                    cam2world: torch.Tensor):
    """
    depth: [n_img, h_img, w_img]
    intrinsics: [n_img, 3, 3]
    cam2world: [n_img, 4, 4]
    return: [n_img, h_img, w_img, 3]
    """
    n_img, h_img, w_img = depth.shape
    y, x = torch.meshgrid(torch.arange(
        h_img), torch.arange(w_img), indexing="ij")
    y = y.to(depth.device) + 0.5
    x = x.to(depth.device) + 0.5
    y = y.unsqueeze(0).expand(n_img, -1, -1)
    x = x.unsqueeze(0).expand(n_img, -1, -1)
    xyz = torch.stack([x, y, torch.ones_like(x)], dim=-1).float()
    xyz = xyz * depth.unsqueeze(-1)
    xyz = rearrange(xyz, "n_img h w xyz -> h w n_img xyz 1")
    xyz = torch.inverse(intrinsics) @ xyz
    xyz = cam2world[..., :3, :3] @ xyz + cam2world[..., :3, 3, None]
    xyz = rearrange(xyz, "h w n_img xyz 1 -> n_img h w xyz")
    return xyz


def grid_sample_depth(depths: torch.Tensor, uv: torch.Tensor):
    """
    depths: [n_img, 1, h_img, w_img]
    uv: [n_img, 1, n_pick, 2]
    """
    h, w = depths.shape[-2:]
    uv = uv[..., :2].clone()
    uv[..., 0] = uv[..., 0] / (w - 1)
    uv[..., 1] = uv[..., 1] / (h - 1)
    uv[..., 0] = uv[..., 0] * 2 - 1
    uv[..., 1] = uv[..., 1] * 2 - 1
    sample_depth = F.grid_sample(
        depths, uv, mode="nearest", align_corners=True)
    return sample_depth


def grid_sample_mask(masks: torch.Tensor, uv: torch.Tensor):
    """
    masks: [n_img, 1, h_img, w_img]
    uv: [n_img, 1, n_pick, 2]
    """
    masks = masks.float()
    h, w = masks.shape[-2:]
    uv = uv[..., :2].clone()
    uv[..., 0] = uv[..., 0] / (w - 1)
    uv[..., 1] = uv[..., 1] / (h - 1)
    uv[..., 0] = uv[..., 0] * 2 - 1
    uv[..., 1] = uv[..., 1] * 2 - 1
    sample_mask = F.grid_sample(
        masks, uv, mode="bilinear", align_corners=True)
    sample_mask = sample_mask > 0.5
    return sample_mask


def grid_sample_rgb(rgb: torch.Tensor, uv: torch.Tensor):
    """
    rgb: [n_img, 3, h_img, w_img]
    uv: [n_img, 1, n_pick, 2]
    """
    rgb = rgb.float()
    h, w = rgb.shape[-2:]
    uv = uv[..., :2].clone()
    uv[..., 0] = uv[..., 0] / (w - 1)
    uv[..., 1] = uv[..., 1] / (h - 1)
    uv[..., 0] = uv[..., 0] * 2 - 1
    uv[..., 1] = uv[..., 1] * 2 - 1
    sample_rgb = F.grid_sample(
        rgb, uv, mode="bilinear", align_corners=True)
    sample_rgb = sample_rgb
    return sample_rgb


class RefineDynMask:

    def __init__(self, images: torch.Tensor,
                 depths: torch.Tensor,
                 coarse_masks: torch.Tensor,
                 cam2world: torch.Tensor,
                 intrinsics: torch.Tensor,
                 device: torch.device):
        self.images = images
        self.coarse_masks = coarse_masks
        self.depths = depths
        self.cam2world = cam2world
        self.intrinsics = intrinsics
        self.device = device
        pts = inverse_project(self.depths, self.intrinsics, self.cam2world)
        self.pts = pts

    def _compute_dyn_loss(self, cam_id: int,
                          pts: torch.Tensor,
                          rgb: torch.Tensor,
                          labels: torch.Tensor,
                          dyn_labels: torch.Tensor):
        n_img, _, h_img, w_img = self.images.shape
        label_losses = []
        for label in dyn_labels:
            pick_mask = labels == label
            pick_pts = pts[pick_mask]
            pick_rgb = rgb[pick_mask]
            other_cam_id = torch.tensor(
                [i for i in range(n_img) if i != cam_id], dtype=torch.long)
            other_cam2world = self.cam2world[other_cam_id]
            other_world2cam = torch.inverse(other_cam2world)

            pick_pts = rearrange(pick_pts, "n_pick xyz -> n_pick xyz 1")
            pick_pts_cam = other_world2cam[:, None, :3, :3] @ pick_pts \
                + other_world2cam[:, None, :3, 3:4]
            other_K = self.intrinsics[other_cam_id]
            pick_pts_proj = other_K[:, None, ...] @ pick_pts_cam

            pick_pts_proj = pick_pts_proj[..., 0]
            pick_pts_proj[..., 0:2] = pick_pts_proj[..., 0:2] / \
                pick_pts_proj[..., 2:3]
            valid_width = (pick_pts_proj[..., 0] > 0) & (
                pick_pts_proj[..., 0] < w_img)
            valid_height = (pick_pts_proj[..., 1] > 0) & (
                pick_pts_proj[..., 1] < h_img)
            valid_depth = pick_pts_proj[..., 2] > 0
            valid_proj = valid_width & valid_height & valid_depth

            other_depths = self.depths[other_cam_id][:, None, ...]
            pick_pts_proj = rearrange(
                pick_pts_proj, "n_cam n_pick xyz -> n_cam 1 n_pick xyz")

            sample_depths = grid_sample_depth(other_depths, pick_pts_proj)

            other_dyn_masks = self.coarse_masks[other_cam_id][:, None, ...]
            sample_dyn_masks = grid_sample_mask(other_dyn_masks, pick_pts_proj)

            other_rgbs = self.images[other_cam_id]
            sample_rgbs = grid_sample_rgb(other_rgbs, pick_pts_proj)

            sample_depths = rearrange(
                sample_depths, "n_cam 1 1 n_pick -> n_cam n_pick")
            pick_pts_proj = rearrange(
                pick_pts_proj, "n_cam 1 n_pick xyz -> n_cam n_pick xyz")
            sample_dyn_masks = rearrange(
                sample_dyn_masks, "n_cam 1 1 n_pick -> n_cam n_pick")
            sample_rgbs = rearrange(
                sample_rgbs, "n_cam c 1 n_pick -> n_cam n_pick c")

            # 屏蔽不可见的点
            visible_mask = pick_pts_proj[..., 2] - 0.01 < sample_depths
            # visible and project to static area
            loss_mask = visible_mask & (~sample_dyn_masks)
            loss_mask = loss_mask & valid_proj

            num_loss_points = loss_mask.sum()
            total_sample_points = (n_img - 1) * pick_pts.shape[0]

            # 如果损失点太少，则认为这个label是动态的
            if (num_loss_points / (total_sample_points + 1e-6)) < 0.05:
                label_losses.append((label, 1e10, 1e10, 1e10))
                continue

            depth_diff = pick_pts_proj[..., 2] - sample_depths
            rgb_diff = pick_rgb.unsqueeze(0) - sample_rgbs
            valid_depth_diff = depth_diff[loss_mask]
            valid_rgb_diff = rgb_diff[loss_mask]
            valid_depth_diff = torch.abs(valid_depth_diff)
            valid_rgb_diff = torch.abs(valid_rgb_diff)
            valid_depth_diff = valid_depth_diff.sum()
            valid_rgb_diff = valid_rgb_diff.sum()
            depth_loss = valid_depth_diff / loss_mask.sum()
            rgb_loss = valid_rgb_diff / loss_mask.sum()
            total_loss = depth_loss + rgb_loss / 3

            label_losses.append((label, depth_loss, rgb_loss, total_loss))

        return label_losses

    @torch.no_grad()
    def _refine_mask(self, cam_id: int):
        n_img, _, h_img, w_img = self.images.shape
        pts = self.pts[cam_id]
        rgb = self.images[cam_id]
        pts = rearrange(pts, "h w xyz -> (h w) xyz")
        rgb = rearrange(rgb, "c h w -> (h w) c")

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.cpu().numpy())
        pcd.colors = o3d.utility.Vector3dVector(rgb.cpu().numpy())
        _, select_idx = pcd.remove_statistical_outlier(
            nb_neighbors=20, std_ratio=2.5)
        # print(
        #     f"remove {pts.shape[0] - len(select_idx)} statistical outlier points")

        selected_mask = torch.zeros(pts.shape[0], dtype=torch.bool)
        selected_mask[select_idx] = True

        coarse_mask = self.coarse_masks[cam_id].cpu()
        coarse_mask = rearrange(coarse_mask, "h w -> (h w)")
        dyn_pts = pts[selected_mask & coarse_mask]

        n_clusters = 30
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        dyn_pts_labels = kmeans.fit_predict(dyn_pts.cpu().numpy())
        dyn_labels = np.unique(dyn_pts_labels)
        dyn_labels = torch.tensor(dyn_labels, dtype=torch.long)
        dyn_pts_labels = torch.tensor(dyn_pts_labels, dtype=torch.long)
        pts_labels = torch.zeros(pts.shape[0], dtype=torch.long)
        # -1 是静态，-2是离群点，>=0是动态
        pts_labels[selected_mask & (~coarse_mask)] = -1
        pts_labels[~selected_mask] = -2
        pts_labels[selected_mask & coarse_mask] = dyn_pts_labels

        label_losses = self._compute_dyn_loss(
            cam_id, pts, rgb, pts_labels, dyn_labels)

        thres = 0.1
        selected_labels = torch.tensor(
            [label for label, _, _, loss in label_losses if loss > thres])
        refine_dyn_mask = torch.isin(pts_labels, selected_labels)
        refine_dyn_mask = rearrange(refine_dyn_mask, "(h w) -> h w",
                                    h=h_img, w=w_img)
        return refine_dyn_mask

    def refine_masks(self):
        n_img = self.images.shape[0]
        refined_masks = []
        for i in tqdm(range(n_img)):
            mask = self._refine_mask(i)
            mask = mask.to(torch.uint8).cpu().numpy()
            mask = mask * 255
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(
                mask, cv2.MORPH_CLOSE, kernel, iterations=1)
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
            mask = mask > 0
            mask = torch.tensor(mask, dtype=torch.bool).to(self.device)
            refined_masks.append(mask)
        refined_masks = torch.stack(refined_masks, dim=0)
        return refined_masks




#--------------------------------------------------
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from einops import rearrange

from vggt4d.masks.refine_dyn_mask import RefineDynMask
from vggt4d.models.vggt4d import VGGTFor4D
from vggt4d.masks.dynamic_mask import (adaptive_multiotsu_variance,
                                             cluster_attention_maps,
                                             extract_dyn_map)
from vggt4d.utils.model_utils import inference, organize_qk_dict
from vggt4d.utils.store import (save_depth, save_depth_conf,
                                save_dynamic_masks, save_intrinsic_txt,
                                save_rgb, save_tum_poses)
from vggt.utils.load_fn import load_and_preprocess_images

device = torch.device("cuda") \
    if torch.cuda.is_available() \
    else torch.device("cpu")

model = VGGTFor4D()
model.load_state_dict(torch.load(
    "./ckpts/model_tracker_fixed_e20.pt", weights_only=True))
model.eval()
model = model.to(device)


def process_scene(scene_dir: Path, output_dir: Path):
    """
    Process a single scene

    Args:
        scene_dir: Scene input directory path
        output_dir: Scene output directory path
    """
    image_paths = list(scene_dir.glob("*.jpg")) + list(scene_dir.glob("*.png"))
    image_paths = sorted(image_paths)

    if len(image_paths) == 0:
        print(f"Warning: No images found in {scene_dir}, skipping this scene")
        return

    print(f"Processing scene: {scene_dir.name} ({len(image_paths)} images)")

    images = load_and_preprocess_images(
        [str(image_path) for image_path in image_paths]).to(device)
    n_img, _, h_img, w_img = images.shape

    output_dir.mkdir(parents=True, exist_ok=True)

    # stage 1 predict depth map and dynamic map
    print("  Stage 1: predict depth map and dynamic map")
    predictions1, qk_dict, enc_feat, agg_tokens_list = inference(
        model, images)
    del agg_tokens_list
    qk_dict = organize_qk_dict(qk_dict, images.shape[0])

    dyn_maps = extract_dyn_map(qk_dict, images)
    # save memory usage
    # dyn_maps = batch_extract_dyn_map(qk_dict, images)

    n_img, _, h_img, w_img = images.shape

    h_tok, w_tok = h_img // 14, w_img // 14

    feat_map = rearrange(
        enc_feat, "n_img (h w) c -> n_img h w c", h=h_tok, w=w_tok)

    norm_dyn_map, _ = cluster_attention_maps(
        feat_map, dyn_maps)

    upsampled_map = F.interpolate(rearrange(
        norm_dyn_map, "n_img h w -> n_img 1 h w"), size=(h_img, w_img), mode='bilinear', align_corners=False)
    upsampled_map = rearrange(
        upsampled_map, "n_img 1 h w -> n_img h w")

    thres = adaptive_multiotsu_variance(upsampled_map.cpu().numpy())
    dyn_masks = upsampled_map > thres

    # stage 2 refine extrinsics by dynamic map
    print("  Stage 2: refine extrinsics by dynamic map")
    if "enc_feat" in locals():
        del enc_feat
    if "feat_map" in locals():
        del feat_map

    torch.cuda.empty_cache()
    predictions2, _, _, _ = inference(model, images, dyn_masks.to(device))

    pred_intrinsic = predictions1["intrinsic"]
    pred_cam2world2 = predictions2["cam2world"]

    pred_depths = predictions1["depth"]
    pred_conf = predictions1["depth_conf"]

    # save predictions
    final_prediction = {**predictions1}
    final_prediction["extrinsic"] = predictions2["extrinsic"]
    final_prediction["cam2world"] = pred_cam2world2

    # stage 3 refine dynamic map
    print("  Stage 3: refine dynamic map")
    if "feat_map" in locals():
        del feat_map
    torch.cuda.empty_cache()

    pred_intrinsic = final_prediction["intrinsic"]
    pred_cam2world = final_prediction["cam2world"]

    pred_depths = final_prediction["depth"]
    pred_conf = final_prediction["depth_conf"]

    refiner = RefineDynMask(images, torch.tensor(pred_depths).to(device),
                            dyn_masks.to(device),
                            torch.tensor(
                                pred_cam2world).float().to(device),
                            torch.tensor(pred_intrinsic).to(device),
                            device)

    refined_mask = refiner.refine_masks()
    del refiner

    print(f"  Saving predictions to {output_dir}\n")
    save_intrinsic_txt(output_dir, pred_intrinsic)
    save_rgb(output_dir, images)
    save_depth(output_dir, pred_depths)
    save_depth_conf(output_dir, pred_conf)
    save_tum_poses(output_dir, pred_cam2world2)
    save_dynamic_masks(output_dir, refined_mask)


def main(input_dir: str, output_dir: str):
    """
    Main function

    Args:
        input_dir: Input data directory path
        output_dir: Output result directory path
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    scene_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
    scene_dirs = sorted(scene_dirs)

    if len(scene_dirs) == 0:
        raise ValueError(f"No scene directories found in {input_dir}")

    print(f"Found {len(scene_dirs)} scenes, starting processing...\n")

    for scene_dir in scene_dirs:
        scene_name = scene_dir.name
        scene_output_dir = output_dir / scene_name
        process_scene(scene_dir, scene_output_dir)

    print(f"All scenes processed! Results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VGGT4D demo script")
    parser.add_argument("--input_dir", type=str, default=None, help="Input data directory path")
    parser.add_argument("--output_dir", type=str,
                        default=None, help="Output result directory path")
    args = parser.parse_args()
    main(input_dir=args.input_dir, output_dir=args.output_dir)

