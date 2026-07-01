#!/usr/bin/env python3
"""Build a narrow v80 selected-write low-support TTT carrier map.

This is a bridge from the v80 visual TTT diagnostic grid to the runtime
support-map grid.  It intentionally writes a sparse support payload that can be
loaded by the existing semantic_ttt_overlap_support runtime hook.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.visualize_v78_phase4_ttt_output_separated import (  # noqa: E402
    _delta_map,
    _load_semantic,
    _mask_from_ids,
    _same_mass_random,
    _semantic_patch,
    _torch_load,
)


DEFAULT_VISUAL_ROOT = Path(
    "results/kitti01_hmc_v2/acl2_v80tf_multiseq_goodbad_semantic_three_memory_control/"
    "report_final/phase9_seq01_chunk08_manual_ttt_visual_probe_frame232"
)
DEFAULT_SUPPORT_MAP = Path(
    "results/kitti01_hmc_v2/acl2_v80tf_multiseq_goodbad_semantic_three_memory_control/"
    "report_final/phase9_seq01_ref055_v80_error_semantic_support_maps/"
    "chunk_008_swa_overlap_source_gate_layer_18.pt"
)
DEFAULT_OUT_DIR = Path(
    "results/kitti01_hmc_v2/acl2_v80tf_multiseq_goodbad_semantic_three_memory_control/"
    "report_final/phase9_seq01_ref055_v80_selected_write_support_maps"
)


def _clean(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if torch.is_tensor(value):
        return _clean(value.detach().cpu().tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _find_post_delta(visual_root: Path, case: str, chunk: int) -> Path:
    matches = sorted(
        visual_root.glob(
            f"targets/*/{case}/pipeline/ttt_spatial_post_delta_maps/"
            f"chunk_{int(chunk):03d}_ttt_spatial_post_delta_map.pt"
        )
    )
    if not matches:
        raise FileNotFoundError(
            f"missing post-delta map under {visual_root}/targets/*/{case}/pipeline/"
            f"ttt_spatial_post_delta_maps/chunk_{int(chunk):03d}_ttt_spatial_post_delta_map.pt"
        )
    return matches[0]


def _load_support(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError(f"unsupported support payload: {path}")
    score = payload.get("score_overlap")
    if not torch.is_tensor(score):
        raise KeyError(f"support payload missing score_overlap: {path}")
    return payload


def _support_score(payload: dict[str, Any]) -> torch.Tensor:
    score = payload["score_overlap"].detach().cpu().float()
    if score.ndim == 3:
        score = score[0]
    if score.ndim != 2:
        raise ValueError(f"expected support score [overlap,tokens], got {tuple(score.shape)}")
    return score.clamp(0.0, 1.0)


def _same_mass_from_pool(pool: torch.Tensor, count: int, seed: int, exclude: torch.Tensor | None = None) -> torch.Tensor:
    flat_pool = pool.detach().cpu().bool().reshape(-1).clone()
    if exclude is not None and int(exclude.numel()) == int(flat_pool.numel()):
        flat_pool &= ~exclude.detach().cpu().bool().reshape(-1)
    available = torch.nonzero(flat_pool, as_tuple=False).reshape(-1)
    if int(available.numel()) < int(count):
        flat_pool = pool.detach().cpu().bool().reshape(-1)
        available = torch.nonzero(flat_pool, as_tuple=False).reshape(-1)
    count = max(0, min(int(count), int(available.numel())))
    out = torch.zeros_like(flat_pool, dtype=torch.bool)
    if count <= 0:
        return out.reshape_as(pool)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    perm = torch.randperm(int(available.numel()), generator=gen)
    out[available[perm[:count]]] = True
    return out.reshape_as(pool)


def build(args: argparse.Namespace) -> dict[str, Any]:
    support_payload = _load_support(Path(args.source_support_map))
    support_score = _support_score(support_payload)
    overlap = min(int(args.overlap), int(support_score.shape[0]))
    runtime_grid = (int(args.runtime_grid[0]), int(args.runtime_grid[1]))
    runtime_tokens = int(runtime_grid[0]) * int(runtime_grid[1])
    if int(support_score.shape[-1]) != runtime_tokens:
        raise ValueError(
            f"runtime grid {runtime_grid} has {runtime_tokens} tokens but support has "
            f"{int(support_score.shape[-1])}"
        )

    post_delta_path = Path(args.post_delta_pt) if args.post_delta_pt else _find_post_delta(
        Path(args.visual_root), str(args.case), int(args.chunk)
    )
    post_delta = _torch_load(post_delta_path)
    start_frame = int(post_delta.get("start_frame", 0))
    local_frame = int(args.global_frame) - start_frame
    if not (0 <= local_frame < int(post_delta.get("num_frames", 0))):
        raise ValueError(f"global frame {args.global_frame} is outside post-delta chunk starting at {start_frame}")
    if local_frame >= overlap:
        raise ValueError(f"local frame {local_frame} is outside requested support overlap {overlap}")

    stage_c_masklet = Path(args.stage_c_masklet or support_payload.get("source_stage_c_masklet") or "")
    if not stage_c_masklet.is_file():
        raise FileNotFoundError(f"missing stage-C masklet: {stage_c_masklet}")
    semantic = _load_semantic(stage_c_masklet)

    d_geo = _delta_map(post_delta, "D_tok_patch", local_frame)
    if not torch.is_tensor(d_geo):
        raise KeyError(f"missing D_tok_patch in {post_delta_path}")
    d_geo = d_geo.detach().cpu().float()
    visual_h, visual_w = int(d_geo.shape[0]), int(d_geo.shape[1])
    labels, _sem_img, conf_patch = _semantic_patch(semantic, local_frame, (visual_w, visual_h))
    dyn_mask = _mask_from_ids(labels, semantic["dynamic_ids"])
    high_d = d_geo > torch.quantile(d_geo.reshape(-1), float(args.d_tok_quantile))
    low_conf = conf_patch < float(args.low_conf_threshold)
    selected_visual = dyn_mask | high_d | low_conf
    visual_random = _same_mass_random(selected_visual, int(args.seed) + int(args.global_frame))

    selected_runtime = F.interpolate(
        selected_visual.float()[None, None],
        size=runtime_grid,
        mode="nearest",
    ).squeeze(0).squeeze(0).bool()
    random_runtime_from_visual = F.interpolate(
        visual_random.float()[None, None],
        size=runtime_grid,
        mode="nearest",
    ).squeeze(0).squeeze(0).bool()
    support_runtime = support_score[int(local_frame)].reshape(runtime_grid)
    low_support = support_runtime <= float(args.support_threshold)
    selected_low = selected_runtime & low_support

    control_pool_name = str(args.control_pool or "low_support").strip().lower()
    if control_pool_name == "low_support":
        control_pool = low_support
    elif control_pool_name in {"not_low_support", "high_support"}:
        control_pool = ~low_support
    elif control_pool_name in {"not_selected", "outside_selected"}:
        control_pool = ~selected_runtime
    elif control_pool_name in {"all", "all_runtime"}:
        control_pool = torch.ones_like(low_support, dtype=torch.bool)
    else:
        raise ValueError(f"unsupported control pool: {args.control_pool}")
    control_exclude = selected_low if bool(args.exclude_selected_from_control) else None
    control_low = _same_mass_from_pool(
        control_pool,
        int(selected_low.sum().item()),
        int(args.control_seed),
        exclude=control_exclude,
    )

    score_overlap = torch.ones((overlap, runtime_tokens), dtype=torch.float32)
    control_overlap = torch.ones((overlap, runtime_tokens), dtype=torch.float32)
    score_overlap[int(local_frame)] = torch.where(
        selected_low.reshape(-1),
        torch.zeros(runtime_tokens, dtype=torch.float32),
        score_overlap[int(local_frame)],
    )
    control_overlap[int(local_frame)] = torch.where(
        control_low.reshape(-1),
        torch.zeros(runtime_tokens, dtype=torch.float32),
        control_overlap[int(local_frame)],
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"chunk_{int(args.chunk):03d}_swa_overlap_{args.kind}_layer_{int(args.layer_idx):02d}.pt"
    payload = {
        "schema": "acl2_v68_swa_overlap_feature_map_v1",
        "artifact": "ACL2_V80_SELECTED_WRITE_LOW_SUPPORT",
        "kind": str(args.kind),
        "mode": "selected_write_low_support_runtime_veto_candidate",
        "chunk_idx": int(args.chunk),
        "swa_layer_idx": int(args.layer_idx),
        "batch_size": 1,
        "frame_num": int(post_delta.get("num_frames", support_payload.get("frame_num", 0)) or 0),
        "tokens_per_frame": int(runtime_tokens),
        "history_tokens": 0,
        "source_start": 0,
        "source_end": int(score_overlap.numel()),
        "source_tokens": int(score_overlap.numel()),
        "overlap_frames_effective": int(overlap),
        "runtime_swa_overlap_feature_not_qk_proxy": True,
        "Dq_overlap": score_overlap.unsqueeze(0),
        "Ds_overlap": (1.0 - score_overlap).unsqueeze(0),
        "score_overlap": score_overlap.unsqueeze(0),
        "control_overlap": control_overlap.unsqueeze(0),
        "score_mean": float(score_overlap.mean().item()),
        "score_q10": float(torch.quantile(score_overlap.reshape(-1), 0.10).item()),
        "score_q50": float(torch.quantile(score_overlap.reshape(-1), 0.50).item()),
        "score_q90": float(torch.quantile(score_overlap.reshape(-1), 0.90).item()),
        "control_mean": float(control_overlap.mean().item()),
        "control_q90": float(torch.quantile(control_overlap.reshape(-1), 0.90).item()),
        "source_support_map": str(args.source_support_map),
        "source_support_artifact": support_payload.get("artifact"),
        "source_support_mode": support_payload.get("mode"),
        "source_support_bad_delta_key": support_payload.get("bad_delta_key"),
        "source_visual_root": str(args.visual_root),
        "source_post_delta_pt": str(post_delta_path),
        "source_stage_c_masklet": str(stage_c_masklet),
        "source_case": str(args.case),
        "source_global_frame": int(args.global_frame),
        "selection_rule": "dynamic_semantic OR D_tok_q75 OR semantic_conf_lt_0.55, then nearest resize to runtime grid, then intersect v80 low-support",
        "control_rule": f"same-mass random sample inside {control_pool_name} pool",
        "control_pool": control_pool_name,
    }
    torch.save(payload, out_path)

    summary = {
        "schema": "acl2_v80_selected_write_support_map_summary_v1",
        "support_path": str(out_path),
        "source_support_map": str(args.source_support_map),
        "source_support_artifact": support_payload.get("artifact"),
        "source_support_mode": support_payload.get("mode"),
        "source_support_bad_delta_key": support_payload.get("bad_delta_key"),
        "source_visual_root": str(args.visual_root),
        "source_post_delta_pt": str(post_delta_path),
        "source_stage_c_masklet": str(stage_c_masklet),
        "chunk": int(args.chunk),
        "global_frame": int(args.global_frame),
        "start_frame": int(start_frame),
        "local_frame": int(local_frame),
        "overlap_frames_effective": int(overlap),
        "visual_grid": [visual_h, visual_w],
        "runtime_grid": [int(runtime_grid[0]), int(runtime_grid[1])],
        "grid_bridge": "nearest_resize_visual_mask_to_runtime_grid",
        "support_threshold": float(args.support_threshold),
        "d_tok_quantile": float(args.d_tok_quantile),
        "low_conf_threshold": float(args.low_conf_threshold),
        "selected_visual_mass": int(selected_visual.sum().item()),
        "selected_visual_ratio": float(selected_visual.float().mean().item()),
        "visual_random_mass": int(visual_random.sum().item()),
        "selected_runtime_mass": int(selected_runtime.sum().item()),
        "random_runtime_from_visual_mass": int(random_runtime_from_visual.sum().item()),
        "runtime_low_support_mass": int(low_support.sum().item()),
        "runtime_low_support_ratio": float(low_support.float().mean().item()),
        "selected_low_support_mass": int(selected_low.sum().item()),
        "selected_low_support_given_selected_runtime": float(
            selected_low.float().sum().item() / max(1, int(selected_runtime.sum().item()))
        ),
        "selected_low_support_given_low_support": float(
            selected_low.float().sum().item() / max(1, int(low_support.sum().item()))
        ),
        "control_low_support_mass": int(control_low.sum().item()),
        "control_pool": control_pool_name,
        "control_pool_mass": int(control_pool.sum().item()),
        "control_seed": int(args.control_seed),
        "exclude_selected_from_control": bool(args.exclude_selected_from_control),
        "score_mean": float(score_overlap.mean().item()),
        "score_q50": float(torch.quantile(score_overlap.reshape(-1), 0.50).item()),
        "control_mean": float(control_overlap.mean().item()),
        "active_score_tokens": int((score_overlap < 0.999999).sum().item()),
        "active_control_tokens": int((control_overlap < 0.999999).sum().item()),
        "method_gate_claimed": False,
    }
    _write_json(out_dir / f"chunk_{int(args.chunk):03d}_selected_write_support_map_summary.json", summary)
    _write_json(out_dir / "selected_write_support_map_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visual-root", type=Path, default=DEFAULT_VISUAL_ROOT)
    parser.add_argument("--case", default="LW1_TTT_SEMANTIC_BASE")
    parser.add_argument("--post-delta-pt", type=Path, default=None)
    parser.add_argument("--source-support-map", type=Path, default=DEFAULT_SUPPORT_MAP)
    parser.add_argument("--stage-c-masklet", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--chunk", type=int, default=8)
    parser.add_argument("--global-frame", type=int, default=232)
    parser.add_argument("--overlap", type=int, default=3)
    parser.add_argument("--runtime-grid", nargs=2, type=int, default=(22, 57))
    parser.add_argument("--kind", default="source_gate")
    parser.add_argument("--layer-idx", type=int, default=18)
    parser.add_argument("--support-threshold", type=float, default=0.5)
    parser.add_argument("--d-tok-quantile", type=float, default=0.75)
    parser.add_argument("--low-conf-threshold", type=float, default=0.55)
    parser.add_argument("--seed", type=int, default=78)
    parser.add_argument("--control-seed", type=int, default=80642)
    parser.add_argument("--exclude-selected-from-control", type=int, default=1)
    parser.add_argument(
        "--control-pool",
        choices=("low_support", "not_low_support", "high_support", "not_selected", "outside_selected", "all", "all_runtime"),
        default="low_support",
        help="Pool used for same-mass random control tokens.",
    )
    return parser.parse_args()


def main() -> None:
    summary = build(parse_args())
    print(json.dumps(_clean(summary), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
