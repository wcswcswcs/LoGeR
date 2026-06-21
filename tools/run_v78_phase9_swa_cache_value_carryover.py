#!/usr/bin/env python3
"""Run ACL2 v78 Phase9 SWA cache/value carry-over smokes.

This follows the Phase8 PCA rediscovery clue that SWA cache-V L18 carries the
clearest road/corridor structure.  The runner keeps the experiment H35-clean:
no C9-informed chunk-wise replay parameters are used.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.evaluate_v68_phaseD_read_smoke import (  # noqa: E402
    LOWER_IS_BETTER_KEYS,
    _eval_run,
    _finite,
    _load_kitti_gt,
    _safe_ratio_improvement,
    _write_csv,
)


DEFAULT_OUTPUT_ROOT = Path(
    "results/kitti01_hmc_v2/acl2_v78tf_pca_grounded_memory_control/"
    "report_final/phase9_swa_cache_value_carryover/smoke_chunk06_context2_v1"
)
DEFAULT_INPUT = Path("/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/sequences/01/image_2")
DEFAULT_GT = Path("/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/poses/01.txt")
DEFAULT_CHECKPOINT = Path("/mnt/data/users/chengshun.wang/pjs/LoGeR/ckpts/LoGeR/latest.pt")
DEFAULT_CONFIG = Path("/mnt/data/users/chengshun.wang/pjs/LoGeR/ckpts/LoGeR/original_config.yaml")
DEFAULT_CONDA = Path("/mnt/data/users/chengshun.wang/miniconda3/bin/conda")
DEFAULT_STAGE_C_CACHE = Path("results/kitti_preprocess/01/stage_c_cache_semantic_chunks")
DEFAULT_CUDA_ALLOC_CONF = "expandable_segments:True"

MECHANISM_KEYS = [
    "head10_to_tail10_pose_sim3_rmse_m",
    "overlap3_to_future_pose_sim3_rmse_m",
    "scale_cv_head_mid_tail_pose_sim3",
]

PHASE9_CASES: Dict[str, Dict[str, Any]] = {
    "P9_0_NATIVE": {
        "family": "baseline",
        "semantic_role_policy": "none",
        "semantic_memory_paths": "",
    },
    "P9_1_CACHEV_STATIC_BLEND_L18": {
        "family": "swa_write_cache_blend",
        "enable_swa_write_control": 1,
        "swa_write_cache_blend_mode": "static",
        "swa_write_cache_blend_target": "v",
        "semantic_role_policy": "fine_ttt_lowstuff_highd_short",
        "semantic_memory_paths": "swa",
    },
    "P9_2_CACHEKV_STATIC_BLEND_L18": {
        "family": "swa_write_cache_blend",
        "enable_swa_write_control": 1,
        "swa_write_cache_blend_mode": "static",
        "swa_write_cache_blend_target": "kv",
        "semantic_role_policy": "fine_ttt_lowstuff_highd_short",
        "semantic_memory_paths": "swa",
    },
    "P9_3_CACHEV_ALL_BLEND_L18_CONTROL": {
        "family": "swa_write_cache_blend_control",
        "enable_swa_write_control": 1,
        "swa_write_cache_blend_mode": "all",
        "swa_write_cache_blend_target": "v",
        "semantic_role_policy": "fine_ttt_lowstuff_highd_short",
        "semantic_memory_paths": "swa",
    },
    "P9_4_SOURCE_REPLACE_GROUND_V_L18": {
        "family": "swa_source_replace",
        "enable_swa_overlap_source_replace": 1,
        "swa_overlap_source_replace_mode": "semantic_ground",
        "swa_overlap_source_replace_target": "v",
        "semantic_role_policy": "fine_ttt_lowstuff_highd_short",
        "semantic_memory_paths": "swa",
    },
    "P9_5_SOURCE_REPLACE_GROUND_RANDOM_SAME_MASS_V_L18": {
        "family": "swa_source_replace_control",
        "enable_swa_overlap_source_replace": 1,
        "swa_overlap_source_replace_mode": "semantic_ground_random_same_mass",
        "swa_overlap_source_replace_target": "v",
        "semantic_role_policy": "fine_ttt_lowstuff_highd_short",
        "semantic_memory_paths": "swa",
    },
    "P9_6_SOURCE_GATE_ROLE_NEGATIVE_V_LAST": {
        "family": "swa_source_gate",
        "enable_swa_overlap_source_gate": 1,
        "swa_overlap_source_gate_mode": "semantic_role_negative",
        "swa_overlap_source_gate_target": "v",
        "semantic_role_policy": "fine_ttt_lowstuff_highd_short",
        "semantic_memory_paths": "swa",
    },
    "P9_7_SOURCE_GATE_ROLE_NEGATIVE_RANDOM_SAME_MASS_V_LAST": {
        "family": "swa_source_gate_control",
        "enable_swa_overlap_source_gate": 1,
        "swa_overlap_source_gate_mode": "random_same_mass_semantic_role_negative",
        "swa_overlap_source_gate_target": "v",
        "semantic_role_policy": "fine_ttt_lowstuff_highd_short",
        "semantic_memory_paths": "swa",
    },
    "P9_8_SOURCE_GATE_DISAGREEMENT_V_LAST": {
        "family": "swa_source_gate",
        "enable_swa_overlap_source_gate": 1,
        "swa_overlap_source_gate_mode": "disagreement",
        "swa_overlap_source_gate_target": "v",
        "semantic_role_policy": "none",
        "semantic_memory_paths": "",
    },
    "P9_9_SOURCE_GATE_DISAGREEMENT_RANDOM_SAME_MASS_V_LAST": {
        "family": "swa_source_gate_control",
        "enable_swa_overlap_source_gate": 1,
        "swa_overlap_source_gate_mode": "disagreement_random_same_mass",
        "swa_overlap_source_gate_target": "v",
        "semantic_role_policy": "none",
        "semantic_memory_paths": "",
    },
    "P9_10_SOURCE_GATE_DISAGREEMENT_K_LAST": {
        "family": "swa_source_gate",
        "enable_swa_overlap_source_gate": 1,
        "swa_overlap_source_gate_mode": "disagreement",
        "swa_overlap_source_gate_target": "k",
        "semantic_role_policy": "none",
        "semantic_memory_paths": "",
    },
    "P9_11_SOURCE_GATE_DISAGREEMENT_RANDOM_SAME_MASS_K_LAST": {
        "family": "swa_source_gate_control",
        "enable_swa_overlap_source_gate": 1,
        "swa_overlap_source_gate_mode": "disagreement_random_same_mass",
        "swa_overlap_source_gate_target": "k",
        "semantic_role_policy": "none",
        "semantic_memory_paths": "",
    },
    "P9_12_SOURCE_GATE_DISAGREEMENT_KV_LAST": {
        "family": "swa_source_gate",
        "enable_swa_overlap_source_gate": 1,
        "swa_overlap_source_gate_mode": "disagreement",
        "swa_overlap_source_gate_target": "kv",
        "semantic_role_policy": "none",
        "semantic_memory_paths": "",
    },
    "P9_13_SOURCE_GATE_DISAGREEMENT_RANDOM_SAME_MASS_KV_LAST": {
        "family": "swa_source_gate_control",
        "enable_swa_overlap_source_gate": 1,
        "swa_overlap_source_gate_mode": "disagreement_random_same_mass",
        "swa_overlap_source_gate_target": "kv",
        "semantic_role_policy": "none",
        "semantic_memory_paths": "",
    },
    "P9_14_SOURCE_REPLACE_DISAGREEMENT_K_LAST": {
        "family": "swa_source_replace",
        "enable_swa_overlap_source_replace": 1,
        "swa_overlap_source_replace_mode": "disagreement",
        "swa_overlap_source_replace_target": "k",
        "semantic_role_policy": "none",
        "semantic_memory_paths": "",
    },
    "P9_15_SOURCE_REPLACE_DISAGREEMENT_RANDOM_SAME_MASS_K_LAST": {
        "family": "swa_source_replace_control",
        "enable_swa_overlap_source_replace": 1,
        "swa_overlap_source_replace_mode": "disagreement_random_same_mass",
        "swa_overlap_source_replace_target": "k",
        "semantic_role_policy": "none",
        "semantic_memory_paths": "",
    },
    "P9_16_SOURCE_REPLACE_DISAGREEMENT_KV_LAST": {
        "family": "swa_source_replace",
        "enable_swa_overlap_source_replace": 1,
        "swa_overlap_source_replace_mode": "disagreement",
        "swa_overlap_source_replace_target": "kv",
        "semantic_role_policy": "none",
        "semantic_memory_paths": "",
    },
    "P9_17_SOURCE_REPLACE_DISAGREEMENT_RANDOM_SAME_MASS_KV_LAST": {
        "family": "swa_source_replace_control",
        "enable_swa_overlap_source_replace": 1,
        "swa_overlap_source_replace_mode": "disagreement_random_same_mass",
        "swa_overlap_source_replace_target": "kv",
        "semantic_role_policy": "none",
        "semantic_memory_paths": "",
    },
    "P9_18_SOURCE_BOOST_STABLE_AGREEMENT_K_LAST": {
        "family": "swa_source_gate",
        "enable_swa_overlap_source_gate": 1,
        "swa_overlap_source_gate_mode": "boost_stable_agreement",
        "swa_overlap_source_gate_target": "k",
        "semantic_role_policy": "none",
        "semantic_memory_paths": "",
    },
    "P9_19_SOURCE_BOOST_STABLE_AGREEMENT_RANDOM_SAME_MASS_K_LAST": {
        "family": "swa_source_gate_control",
        "enable_swa_overlap_source_gate": 1,
        "swa_overlap_source_gate_mode": "boost_stable_agreement_random_same_mass",
        "swa_overlap_source_gate_target": "k",
        "semantic_role_policy": "none",
        "semantic_memory_paths": "",
    },
    "P9_20_ATTENTION_BIAS_STABLE_AGREEMENT_LAST": {
        "family": "swa_overlap_bias",
        "enable_swa_overlap_bias": 1,
        "swa_overlap_bias_mode": "boost_stable_agreement",
        "semantic_role_policy": "none",
        "semantic_memory_paths": "",
    },
    "P9_21_ATTENTION_BIAS_STABLE_AGREEMENT_RANDOM_SAME_MASS_LAST": {
        "family": "swa_overlap_bias_control",
        "enable_swa_overlap_bias": 1,
        "swa_overlap_bias_mode": "boost_stable_agreement_random_same_mass",
        "semantic_role_policy": "none",
        "semantic_memory_paths": "",
    },
    "P9_22_ATTENTION_BIAS_SEMANTIC_SAMEGROUP_STABLE_AGREEMENT_LAST": {
        "family": "swa_overlap_bias",
        "enable_swa_overlap_bias": 1,
        "swa_overlap_bias_mode": "semantic_same_group_boost_stable_agreement",
        "semantic_role_policy": "fine_ttt_lowstuff_highd_short",
        "semantic_memory_paths": "swa",
    },
    "P9_23_ATTENTION_BIAS_SEMANTIC_SAMEGROUP_STABLE_AGREEMENT_RANDOM_SAME_MASS_LAST": {
        "family": "swa_overlap_bias_control",
        "enable_swa_overlap_bias": 1,
        "swa_overlap_bias_mode": "semantic_same_group_boost_stable_agreement_random_same_mass",
        "semantic_role_policy": "fine_ttt_lowstuff_highd_short",
        "semantic_memory_paths": "swa",
    },
    "P9_24_ATTENTION_BIAS_STABLE_AGREEMENT_TOPQ90_LAST": {
        "family": "swa_overlap_bias",
        "enable_swa_overlap_bias": 1,
        "swa_overlap_bias_mode": "boost_stable_agreement_topq90",
        "semantic_role_policy": "none",
        "semantic_memory_paths": "",
    },
    "P9_25_ATTENTION_BIAS_STABLE_AGREEMENT_TOPQ90_RANDOM_SAME_MASS_LAST": {
        "family": "swa_overlap_bias_control",
        "enable_swa_overlap_bias": 1,
        "swa_overlap_bias_mode": "boost_stable_agreement_topq90_random_same_mass",
        "semantic_role_policy": "none",
        "semantic_memory_paths": "",
    },
    "P9_26_ATTENTION_BIAS_STABLE_AGREEMENT_TOPQ80_LAST": {
        "family": "swa_overlap_bias",
        "enable_swa_overlap_bias": 1,
        "swa_overlap_bias_mode": "boost_stable_agreement_topq80",
        "semantic_role_policy": "none",
        "semantic_memory_paths": "",
    },
    "P9_27_ATTENTION_BIAS_STABLE_AGREEMENT_TOPQ80_RANDOM_SAME_MASS_LAST": {
        "family": "swa_overlap_bias_control",
        "enable_swa_overlap_bias": 1,
        "swa_overlap_bias_mode": "boost_stable_agreement_topq80_random_same_mass",
        "semantic_role_policy": "none",
        "semantic_memory_paths": "",
    },
    "P9_28_ATTENTION_BIAS_STABLE_AGREEMENT_TOPQ80_ALIGNED_LAST": {
        "family": "swa_overlap_bias",
        "enable_swa_overlap_bias": 1,
        "swa_overlap_bias_mode": "boost_stable_agreement_topq80_aligned",
        "semantic_role_policy": "none",
        "semantic_memory_paths": "",
    },
    "P9_29_ATTENTION_BIAS_STABLE_AGREEMENT_TOPQ80_ALIGNED_RANDOM_SAME_MASS_LAST": {
        "family": "swa_overlap_bias_control",
        "enable_swa_overlap_bias": 1,
        "swa_overlap_bias_mode": "boost_stable_agreement_topq80_aligned_random_same_mass",
        "semantic_role_policy": "none",
        "semantic_memory_paths": "",
    },
    "P9_30_SOURCE_REPLACE_STABLE_AGREEMENT_TOPQ80_KV_LAST": {
        "family": "swa_source_replace",
        "enable_swa_overlap_source_replace": 1,
        "swa_overlap_source_replace_mode": "stable_agreement_topq80",
        "swa_overlap_source_replace_target": "kv",
        "semantic_role_policy": "none",
        "semantic_memory_paths": "",
    },
    "P9_31_SOURCE_REPLACE_STABLE_AGREEMENT_TOPQ80_RANDOM_SAME_MASS_KV_LAST": {
        "family": "swa_source_replace_control",
        "enable_swa_overlap_source_replace": 1,
        "swa_overlap_source_replace_mode": "stable_agreement_topq80_random_same_mass",
        "swa_overlap_source_replace_target": "kv",
        "semantic_role_policy": "none",
        "semantic_memory_paths": "",
    },
    "P9_32_SOURCE_REPLACE_STABLE_AGREEMENT_TOPQ80_K_LIGHT_LAST": {
        "family": "swa_source_replace",
        "enable_swa_overlap_source_replace": 1,
        "swa_overlap_source_replace_mode": "stable_agreement_topq80",
        "swa_overlap_source_replace_target": "k",
        "semantic_role_policy": "none",
        "semantic_memory_paths": "",
    },
    "P9_33_SOURCE_REPLACE_STABLE_AGREEMENT_TOPQ80_RANDOM_SAME_MASS_K_LIGHT_LAST": {
        "family": "swa_source_replace_control",
        "enable_swa_overlap_source_replace": 1,
        "swa_overlap_source_replace_mode": "stable_agreement_topq80_random_same_mass",
        "swa_overlap_source_replace_target": "k",
        "semantic_role_policy": "none",
        "semantic_memory_paths": "",
    },
    "P9_34_ATTENTION_BIAS_STABLE_AGREEMENT_TOPQ80_MASS_AUDIT_LAST": {
        "family": "swa_overlap_bias",
        "enable_swa_overlap_bias": 1,
        "swa_overlap_bias_mode": "boost_stable_agreement_topq80",
        "swa_overlap_bias_record_attention_mass": 1,
        "semantic_role_policy": "none",
        "semantic_memory_paths": "",
    },
    "P9_35_ATTENTION_BIAS_STABLE_AGREEMENT_TOPQ80_RANDOM_SAME_MASS_MASS_AUDIT_LAST": {
        "family": "swa_overlap_bias_control",
        "enable_swa_overlap_bias": 1,
        "swa_overlap_bias_mode": "boost_stable_agreement_topq80_random_same_mass",
        "swa_overlap_bias_record_attention_mass": 1,
        "semantic_role_policy": "none",
        "semantic_memory_paths": "",
    },
    "P9_36_ATTENTION_BIAS_STABLE_AGREEMENT_TOPQ80_HEAD6_LAST": {
        "family": "swa_overlap_bias",
        "enable_swa_overlap_bias": 1,
        "swa_overlap_bias_mode": "boost_stable_agreement_topq80",
        "swa_overlap_bias_head_indices": "6",
        "swa_overlap_bias_record_attention_mass": 1,
        "semantic_role_policy": "none",
        "semantic_memory_paths": "",
    },
    "P9_37_ATTENTION_BIAS_STABLE_AGREEMENT_TOPQ80_RANDOM_SAME_MASS_HEAD6_LAST": {
        "family": "swa_overlap_bias_control",
        "enable_swa_overlap_bias": 1,
        "swa_overlap_bias_mode": "boost_stable_agreement_topq80_random_same_mass",
        "swa_overlap_bias_head_indices": "6",
        "swa_overlap_bias_record_attention_mass": 1,
        "semantic_role_policy": "none",
        "semantic_memory_paths": "",
    },
    "P9_38_ATTENTION_BIAS_STABLE_AGREEMENT_TOPQ80_HEADS0_6_8_LAST": {
        "family": "swa_overlap_bias",
        "enable_swa_overlap_bias": 1,
        "swa_overlap_bias_mode": "boost_stable_agreement_topq80",
        "swa_overlap_bias_head_indices": "0,6,8",
        "swa_overlap_bias_record_attention_mass": 1,
        "semantic_role_policy": "none",
        "semantic_memory_paths": "",
    },
    "P9_39_ATTENTION_BIAS_STABLE_AGREEMENT_TOPQ80_RANDOM_SAME_MASS_HEADS0_6_8_LAST": {
        "family": "swa_overlap_bias_control",
        "enable_swa_overlap_bias": 1,
        "swa_overlap_bias_mode": "boost_stable_agreement_topq80_random_same_mass",
        "swa_overlap_bias_head_indices": "0,6,8",
        "swa_overlap_bias_record_attention_mass": 1,
        "semantic_role_policy": "none",
        "semantic_memory_paths": "",
    },
    "P9_40_SOURCE_REPLACE_STABLE_AGREEMENT_TOPQ80_V_LAST": {
        "family": "swa_source_replace",
        "enable_swa_overlap_source_replace": 1,
        "swa_overlap_source_replace_mode": "stable_agreement_topq80",
        "swa_overlap_source_replace_target": "v",
        "semantic_role_policy": "none",
        "semantic_memory_paths": "",
    },
    "P9_41_SOURCE_REPLACE_STABLE_AGREEMENT_TOPQ80_RANDOM_SAME_MASS_V_LAST": {
        "family": "swa_source_replace_control",
        "enable_swa_overlap_source_replace": 1,
        "swa_overlap_source_replace_mode": "stable_agreement_topq80_random_same_mass",
        "swa_overlap_source_replace_target": "v",
        "semantic_role_policy": "none",
        "semantic_memory_paths": "",
    },
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    return value


def _parse_csv_ints(text: str) -> List[int]:
    return [int(part.strip()) for part in str(text or "").split(",") if part.strip()]


def _read_jsonl_all(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _finite_mean(values: Iterable[Any]) -> Optional[float]:
    xs = [_finite(v) for v in values]
    xs = [v for v in xs if v is not None]
    return float(np.mean(xs)) if xs else None


def _finite_max(values: Iterable[Any]) -> Optional[float]:
    xs = [_finite(v) for v in values]
    xs = [v for v in xs if v is not None]
    return float(np.max(xs)) if xs else None


def _finite_sum(values: Iterable[Any]) -> Optional[float]:
    xs = [_finite(v) for v in values]
    xs = [v for v in xs if v is not None]
    return float(np.sum(xs)) if xs else None


def _context_window(args: argparse.Namespace, chunk: int) -> Dict[str, int]:
    stride = int(args.chunk_size) - int(args.chunk_overlap)
    context_chunks = max(int(args.context_chunks), 1)
    first_chunk = max(int(chunk) - context_chunks + 1, 0)
    actual_chunks = int(chunk) - first_chunk + 1
    start = first_chunk * stride
    end = start + int(args.chunk_size) + (actual_chunks - 1) * stride
    target_start = int(chunk) * stride
    target_end = target_start + int(args.chunk_size)
    return {
        "context_start_chunk": int(first_chunk),
        "context_chunks": int(actual_chunks),
        "start_frame": int(start),
        "end_frame": int(end),
        "target_start_frame": int(target_start),
        "target_end_frame": int(target_end),
    }


def _case_value(case_cfg: Dict[str, Any], key: str, default: Any) -> Any:
    return case_cfg[key] if key in case_cfg else default


def _build_command(args: argparse.Namespace, *, chunk: int, case: str, out_dir: Path) -> List[str]:
    cfg = PHASE9_CASES[case]
    window = _context_window(args, int(chunk))
    cmd = [
        str(args.conda),
        "run",
        "--no-capture-output",
        "-n",
        args.conda_env,
        "python",
        "run_pipeline_abc_v2.py",
        "--input",
        str(args.input),
        "--output_video",
        "",
        "--output_txt",
        str(out_dir / "01.txt"),
        "--checkpoint",
        str(args.checkpoint),
        "--config",
        str(args.config),
        "--chunk_size",
        str(args.chunk_size),
        "--chunk_overlap",
        str(args.chunk_overlap),
        "--start_frame",
        str(window["start_frame"]),
        "--end_frame",
        str(window["end_frame"]),
        "--global_chunk_offset",
        str(window["context_start_chunk"]),
        "--device",
        "cuda",
        "--hybrid_memory_mode",
        "read_path_only",
        "--hmc_commit_mode",
        "controlled",
        "--semantic_prior_mode",
        "spg_v2",
        "--stage_c_cache_mode",
        "read",
        "--stage_c_cache_dir",
        str(args.stage_c_cache_dir),
        "--stage_c_cache_require_hit",
        "1",
        "--read_path",
        "none",
        "--read_cue_source",
        "dyn",
        "--read_overlap_frames",
        str(args.chunk_overlap),
        "--read_layer_mode",
        "single",
        "--read_single_layer",
        str(args.read_single_layer),
        "--semantic_role_policy",
        str(_case_value(cfg, "semantic_role_policy", "none")),
        "--semantic_memory_paths",
        str(_case_value(cfg, "semantic_memory_paths", "")),
        "--semantic_role_highd_quantile",
        str(args.semantic_role_highd_quantile),
        "--semantic_role_low_trust",
        str(args.semantic_role_low_trust),
        "--semantic_role_swa_protect_scale",
        str(args.semantic_role_swa_protect_scale),
        "--enable_swa_write_control",
        str(int(_case_value(cfg, "enable_swa_write_control", 0))),
        "--swa_write_mode",
        "none",
        "--swa_write_layer_mode",
        "single",
        "--swa_write_single_layer",
        str(args.swa_layer),
        "--swa_write_scope",
        str(args.swa_write_scope),
        "--swa_write_score_source",
        "read",
        "--swa_write_cache_blend_alpha",
        str(args.swa_write_alpha if int(_case_value(cfg, "enable_swa_write_control", 0)) else 0.0),
        "--swa_write_cache_blend_mode",
        str(_case_value(cfg, "swa_write_cache_blend_mode", "static")),
        "--swa_write_cache_blend_target",
        str(_case_value(cfg, "swa_write_cache_blend_target", "v")),
        "--enable_swa_overlap_bias",
        str(int(_case_value(cfg, "enable_swa_overlap_bias", 0))),
        "--swa_overlap_bias_beta",
        str(args.swa_bias_beta if int(_case_value(cfg, "enable_swa_overlap_bias", 0)) else 0.0),
        "--swa_overlap_bias_min_keep",
        str(args.swa_bias_min_keep),
        "--swa_overlap_bias_mode",
        str(_case_value(cfg, "swa_overlap_bias_mode", "pair")),
        "--swa_overlap_bias_layer_mode",
        str(args.swa_source_layer_mode),
        "--swa_overlap_bias_single_layer",
        str(args.swa_source_single_layer),
        "--swa_overlap_bias_head_indices",
        str(_case_value(cfg, "swa_overlap_bias_head_indices", "")),
        "--swa_overlap_bias_record_attention_mass",
        str(int(_case_value(cfg, "swa_overlap_bias_record_attention_mass", 0))),
        "--swa_overlap_bias_attention_mass_max_queries",
        str(args.swa_bias_attention_mass_max_queries),
        "--enable_swa_overlap_source_gate",
        str(int(_case_value(cfg, "enable_swa_overlap_source_gate", 0))),
        "--swa_overlap_source_gate_rho",
        str(args.swa_gate_rho if int(_case_value(cfg, "enable_swa_overlap_source_gate", 0)) else 0.0),
        "--swa_overlap_source_gate_min",
        str(args.swa_gate_min),
        "--swa_overlap_source_gate_mode",
        str(_case_value(cfg, "swa_overlap_source_gate_mode", "source")),
        "--swa_overlap_source_gate_target",
        str(_case_value(cfg, "swa_overlap_source_gate_target", "v")),
        "--swa_overlap_source_gate_layer_mode",
        str(args.swa_source_layer_mode),
        "--swa_overlap_source_gate_single_layer",
        str(args.swa_source_single_layer),
        "--enable_swa_overlap_source_replace",
        str(int(_case_value(cfg, "enable_swa_overlap_source_replace", 0))),
        "--swa_overlap_source_replace_alpha",
        str(args.swa_replace_alpha if int(_case_value(cfg, "enable_swa_overlap_source_replace", 0)) else 0.0),
        "--swa_overlap_source_replace_mode",
        str(_case_value(cfg, "swa_overlap_source_replace_mode", "union")),
        "--swa_overlap_source_replace_target",
        str(_case_value(cfg, "swa_overlap_source_replace_target", "v")),
        "--swa_overlap_source_replace_layer_mode",
        str(args.swa_source_layer_mode),
        "--swa_overlap_source_replace_single_layer",
        str(args.swa_source_single_layer),
        "--swa_overlap_feature_dump_dir",
        str(out_dir / "swa_overlap_feature_maps"),
        "--swa_overlap_feature_dump_dtype",
        str(args.swa_overlap_feature_dump_dtype),
        "--fast_cue_eval",
        "1",
        "--empty_cuda_cache_each_chunk",
        "1",
        "--hybrid_debug_jsonl",
        str(out_dir / "hmc_state_hash.jsonl"),
    ]
    if int(args.v68_export_full_pca_debug):
        cmd.extend(
            [
                "--v68_export_full_pca_debug",
                "1",
                "--v68_layer_pca_feature_dir",
                str(out_dir / "v68_layer_pca_features"),
                "--v68_pca_taps",
                str(args.v68_pca_taps),
                "--v68_pca_layers",
                str(args.v68_pca_layers),
                "--v68_pca_max_feature_dim",
                str(args.v68_pca_max_feature_dim),
            ]
        )
    return cmd


def _run_job(job: Dict[str, Any]) -> Dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    env["CUDA_VISIBLE_DEVICES"] = str(job["gpu"])
    if str(job.get("cuda_alloc_conf") or "").strip():
        env["PYTORCH_CUDA_ALLOC_CONF"] = str(job["cuda_alloc_conf"]).strip()
    out_dir = Path(job["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    run_log = out_dir / "run.log"
    start_t = time.time()
    with run_log.open("w", encoding="utf-8") as handle:
        proc = subprocess.run(
            job["cmd"],
            cwd=job["workdir"],
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    end_t = time.time()
    job.update(
        {
            "returncode": int(proc.returncode),
            "duration_sec": float(end_t - start_t),
            "run_log": str(run_log),
            "trajectory": str(out_dir / "01.txt"),
            "hmc_state_hash": str(out_dir / "hmc_state_hash.jsonl"),
        }
    )
    return job


def _aggregate_phase9_hmc(run_dir: Path) -> Dict[str, Any]:
    rows = _read_jsonl_all(run_dir / "hmc_state_hash.jsonl")
    if not rows:
        return {"phase9_hmc_rows": 0}
    swa_rows = [
        row.get("control_trace", {}).get("hook_effect_summary", {}).get("swa_read", {})
        for row in rows
        if isinstance(row.get("control_trace", {}).get("hook_effect_summary", {}).get("swa_read", {}), dict)
    ]
    implemented = sorted(
        {
            str(path)
            for row in rows
            for path in (row.get("control_trace", {}).get("implemented_paths") or [])
        }
    )
    return {
        "phase9_hmc_rows": int(len(rows)),
        "phase9_implemented_paths_all": implemented,
        "phase9_swa_num_calls_sum": int(sum(int(row.get("num_calls", 0) or 0) for row in swa_rows)),
        "phase9_swa_overlap_bias_applied_sum": int(
            sum(int(row.get("num_swa_overlap_bias_applied", 0) or 0) for row in swa_rows)
        ),
        "phase9_swa_overlap_bias_mean_abs": _finite_mean(row.get("mean_abs_bias") for row in swa_rows),
        "phase9_swa_overlap_bias_max_abs": _finite_max(row.get("max_abs_bias") for row in swa_rows),
        "phase9_swa_overlap_source_replace_applied_sum": int(
            sum(int(row.get("num_swa_overlap_source_replace_applied", 0) or 0) for row in swa_rows)
        ),
        "phase9_swa_overlap_source_gate_applied_sum": int(
            sum(int(row.get("num_swa_overlap_source_gate_applied", 0) or 0) for row in swa_rows)
        ),
        "phase9_swa_mean_overlap_source_gate_delta": _finite_mean(
            row.get("mean_swa_overlap_source_gate_delta") for row in swa_rows
        ),
        "phase9_swa_max_overlap_source_gate_delta": _finite_max(
            row.get("max_swa_overlap_source_gate_delta") for row in swa_rows
        ),
        "phase9_swa_mean_overlap_source_gate": _finite_mean(
            row.get("mean_swa_overlap_source_gate") for row in swa_rows
        ),
        "phase9_swa_mean_overlap_source_score": _finite_mean(
            row.get("mean_swa_overlap_source_score") for row in swa_rows
        ),
        "phase9_swa_mean_overlap_source_score_q90": _finite_mean(
            row.get("mean_swa_overlap_source_score_q90") for row in swa_rows
        ),
        "phase9_swa_mean_overlap_source_replace_alpha": _finite_mean(
            row.get("mean_swa_overlap_source_replace_alpha") for row in swa_rows
        ),
        "phase9_swa_mean_overlap_source_replace_alpha_p90": _finite_mean(
            row.get("mean_swa_overlap_source_replace_alpha_p90") for row in swa_rows
        ),
        "phase9_swa_mean_overlap_source_replace_score": _finite_mean(
            row.get("mean_swa_overlap_source_replace_score") for row in swa_rows
        ),
        "phase9_swa_mean_semantic_selected_ratio": _finite_mean(
            row.get("mean_swa_overlap_source_semantic_selected_ratio") for row in swa_rows
        ),
        "phase9_swa_max_semantic_selected_tokens": _finite_max(
            row.get("max_swa_overlap_source_semantic_selected_tokens") for row in swa_rows
        ),
        "phase9_swa_frac_semantic_random_same_mass": _finite_mean(
            row.get("frac_swa_overlap_source_semantic_random_same_mass") for row in swa_rows
        ),
        "phase9_swa_frac_semantic_missing_labels": _finite_mean(
            row.get("frac_swa_overlap_source_semantic_missing_labels") for row in swa_rows
        ),
        "phase9_swa_attention_mass_available_frac": _finite_mean(
            1.0 if row.get("attention_mass_available") else 0.0 for row in swa_rows
        ),
        "phase9_swa_attention_mass_selected_before": _finite_mean(
            row.get("mean_swa_overlap_attention_mass_selected_before") for row in swa_rows
        ),
        "phase9_swa_attention_mass_selected_after": _finite_mean(
            row.get("mean_swa_overlap_attention_mass_selected_after") for row in swa_rows
        ),
        "phase9_swa_attention_mass_selected_lift": _finite_mean(
            row.get("mean_swa_overlap_attention_mass_selected_lift") for row in swa_rows
        ),
        "phase9_swa_attention_mass_source_before": _finite_mean(
            row.get("mean_swa_overlap_attention_mass_source_before") for row in swa_rows
        ),
        "phase9_swa_attention_mass_source_after": _finite_mean(
            row.get("mean_swa_overlap_attention_mass_source_after") for row in swa_rows
        ),
        "phase9_swa_attention_mass_source_lift": _finite_mean(
            row.get("mean_swa_overlap_attention_mass_source_lift") for row in swa_rows
        ),
        "phase9_swa_attention_mass_selected_head_max_before": _finite_mean(
            row.get("mean_swa_overlap_attention_mass_selected_head_max_before") for row in swa_rows
        ),
        "phase9_swa_attention_mass_selected_head_max_after": _finite_mean(
            row.get("mean_swa_overlap_attention_mass_selected_head_max_after") for row in swa_rows
        ),
        "phase9_swa_attention_mass_selected_head_max_lift": _finite_mean(
            row.get("mean_swa_overlap_attention_mass_selected_head_max_lift") for row in swa_rows
        ),
        "phase9_swa_write_cache_blend_applied_layers_sum": _finite_sum(
            row.get("swa_write_cache_blend_applied_layers") for row in rows
        ),
        "phase9_swa_write_cache_blend_applied_layers_max": _finite_max(
            row.get("swa_write_cache_blend_applied_layers") for row in rows
        ),
        "phase9_swa_write_applied_layers_sum": _finite_sum(row.get("swa_write_applied_layers") for row in rows),
        "phase9_swa_write_scope_tokens_max": _finite_max(row.get("swa_write_scope_tokens") for row in rows),
        "phase9_swa_write_scope_tokens_mean": _finite_mean(row.get("swa_write_scope_tokens") for row in rows),
        "phase9_swa_write_cache_blend_scope_mean": _finite_mean(
            row.get("swa_write_cache_blend_scope_mean") for row in rows
        ),
        "phase9_swa_write_cache_blend_scope_q90": _finite_mean(
            row.get("swa_write_cache_blend_scope_q90") for row in rows
        ),
        "phase9_swa_write_history_tokens_before_max": _finite_max(
            row.get("swa_write_history_tokens_before") for row in rows
        ),
        "phase9_swa_write_history_tokens_after_max": _finite_max(
            row.get("swa_write_history_tokens_after") for row in rows
        ),
        "phase9_prior_semantic_role_consumed_count": int(
            sum(1 for row in rows if bool(row.get("prior_semantic_role_consumed_any")))
        ),
        "phase9_prior_semantic_swa_adjusted_count": int(
            sum(1 for row in rows if bool(row.get("prior_semantic_role_swa_adjusted")))
        ),
    }


def _best_control_value(rows_by_name: Dict[str, Dict[str, Any]], controls: Sequence[str], key: str) -> Optional[float]:
    vals = [_finite(rows_by_name.get(name, {}).get(key)) for name in controls]
    vals = [value for value in vals if value is not None]
    return min(vals) if vals else None


def _action_fidelity(row: Dict[str, Any], *, family: str) -> bool:
    if family == "swa_write_cache_blend":
        return bool(
            float(row.get("phase9_swa_write_cache_blend_applied_layers_sum") or 0.0) > 0.0
            and float(row.get("phase9_swa_write_cache_blend_scope_mean") or 0.0) > 0.0
            and float(row.get("phase9_swa_write_history_tokens_before_max") or 0.0) > 0.0
        )
    if family == "swa_source_replace":
        return bool(
            float(row.get("phase9_swa_overlap_source_replace_applied_sum") or 0.0) > 0.0
            and float(row.get("phase9_swa_mean_overlap_source_replace_alpha") or 0.0) > 0.0
        )
    if family == "swa_source_gate":
        return bool(
            float(row.get("phase9_swa_overlap_source_gate_applied_sum") or 0.0) > 0.0
            and float(row.get("phase9_swa_mean_overlap_source_gate_delta") or 0.0) > 0.0
        )
    if family == "swa_overlap_bias":
        return bool(
            float(row.get("phase9_swa_overlap_bias_applied_sum") or 0.0) > 0.0
            and float(row.get("phase9_swa_overlap_bias_mean_abs") or 0.0) > 0.0
        )
    return False


def _build_phase9_decision(
    rows: Sequence[Dict[str, Any]],
    *,
    candidate: str,
    baseline: str,
    controls: Sequence[str],
) -> Dict[str, Any]:
    rows_by_name = {str(row["run"]): row for row in rows}
    cand = rows_by_name.get(candidate)
    base = rows_by_name.get(baseline)
    if cand is None:
        return {"phase9_gate_pass": False, "reason": f"missing_candidate:{candidate}"}
    if base is None:
        return {"phase9_gate_pass": False, "reason": f"missing_baseline:{baseline}"}
    family = str(PHASE9_CASES.get(candidate, {}).get("family", "unknown"))
    if family == "swa_write_cache_blend_control":
        family = "swa_write_cache_blend"
    if family == "swa_source_replace_control":
        family = "swa_source_replace"
    if family == "swa_source_gate_control":
        family = "swa_source_gate"
    if family == "swa_overlap_bias_control":
        family = "swa_overlap_bias"
    action_fidelity = _action_fidelity(cand, family=family)

    comparisons: Dict[str, Dict[str, Any]] = {}
    metric_passes: List[str] = []
    for key in LOWER_IS_BETTER_KEYS:
        cand_v = _finite(cand.get(key))
        base_v = _finite(base.get(key))
        best_control = _best_control_value(rows_by_name, controls, key)
        beats_controls = cand_v is not None and best_control is not None and cand_v < best_control
        ratio_improvement = _safe_ratio_improvement(base_v, cand_v)
        key_pass = False
        future_worse = False
        if key in MECHANISM_KEYS:
            if key in {"head10_to_tail10_pose_sim3_rmse_m", "scale_cv_head_mid_tail_pose_sim3"}:
                future_ratio = _safe_ratio_improvement(
                    base.get("overlap3_to_future_pose_sim3_rmse_m"),
                    cand.get("overlap3_to_future_pose_sim3_rmse_m"),
                )
                future_worse = bool(future_ratio is not None and future_ratio < -0.01)
            key_pass = bool(
                beats_controls
                and ratio_improvement is not None
                and ratio_improvement >= 0.10
                and not future_worse
            )
        if key_pass:
            metric_passes.append(key)
        comparisons[key] = {
            "candidate": cand_v,
            "baseline": base_v,
            "best_control": best_control,
            "candidate_minus_baseline": (cand_v - base_v) if cand_v is not None and base_v is not None else None,
            "candidate_minus_best_control": (
                cand_v - best_control if cand_v is not None and best_control is not None else None
            ),
            "improvement_vs_baseline_ratio": ratio_improvement,
            "beats_controls": beats_controls,
            "future_worse_gt1pct": future_worse,
            "phase9_metric_key_pass": key_pass,
        }

    return {
        "phase9_gate_pass": bool(action_fidelity and metric_passes),
        "candidate": candidate,
        "baseline": baseline,
        "controls": list(controls),
        "family": family,
        "action_fidelity_pass": action_fidelity,
        "metric_passes": metric_passes,
        "comparisons": comparisons,
        "rule": (
            "Phase9 requires action fidelity and at least one mechanism metric "
            "(head_tail, future_after_overlap, scale_cv) improving >=10% vs baseline "
            "while beating the listed controls; head_tail/scale_cv also require future "
            "not worsening by >1%."
        ),
    }


def _evaluate(args: argparse.Namespace, jobs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    _, gt_poses_all, gt_pos_all = _load_kitti_gt(args.gt)
    rows: List[Dict[str, Any]] = []
    for job in jobs:
        if int(job.get("returncode") or 0) != 0:
            continue
        run_dir = Path(job["out_dir"])
        row = _eval_run(str(job["case"]), run_dir, gt_poses_all, gt_pos_all)
        row.update(_aggregate_phase9_hmc(run_dir))
        row["phase9_family"] = PHASE9_CASES[str(job["case"])]["family"]
        rows.append(row)

    candidates = [
        "P9_1_CACHEV_STATIC_BLEND_L18",
        "P9_2_CACHEKV_STATIC_BLEND_L18",
        "P9_4_SOURCE_REPLACE_GROUND_V_L18",
        "P9_6_SOURCE_GATE_ROLE_NEGATIVE_V_LAST",
        "P9_8_SOURCE_GATE_DISAGREEMENT_V_LAST",
        "P9_10_SOURCE_GATE_DISAGREEMENT_K_LAST",
        "P9_12_SOURCE_GATE_DISAGREEMENT_KV_LAST",
        "P9_14_SOURCE_REPLACE_DISAGREEMENT_K_LAST",
        "P9_16_SOURCE_REPLACE_DISAGREEMENT_KV_LAST",
        "P9_18_SOURCE_BOOST_STABLE_AGREEMENT_K_LAST",
        "P9_20_ATTENTION_BIAS_STABLE_AGREEMENT_LAST",
        "P9_22_ATTENTION_BIAS_SEMANTIC_SAMEGROUP_STABLE_AGREEMENT_LAST",
        "P9_24_ATTENTION_BIAS_STABLE_AGREEMENT_TOPQ90_LAST",
        "P9_26_ATTENTION_BIAS_STABLE_AGREEMENT_TOPQ80_LAST",
        "P9_28_ATTENTION_BIAS_STABLE_AGREEMENT_TOPQ80_ALIGNED_LAST",
        "P9_30_SOURCE_REPLACE_STABLE_AGREEMENT_TOPQ80_KV_LAST",
        "P9_32_SOURCE_REPLACE_STABLE_AGREEMENT_TOPQ80_K_LIGHT_LAST",
        "P9_34_ATTENTION_BIAS_STABLE_AGREEMENT_TOPQ80_MASS_AUDIT_LAST",
        "P9_36_ATTENTION_BIAS_STABLE_AGREEMENT_TOPQ80_HEAD6_LAST",
        "P9_38_ATTENTION_BIAS_STABLE_AGREEMENT_TOPQ80_HEADS0_6_8_LAST",
        "P9_40_SOURCE_REPLACE_STABLE_AGREEMENT_TOPQ80_V_LAST",
    ]
    controls_by_candidate = {
        "P9_1_CACHEV_STATIC_BLEND_L18": ["P9_3_CACHEV_ALL_BLEND_L18_CONTROL"],
        "P9_2_CACHEKV_STATIC_BLEND_L18": ["P9_3_CACHEV_ALL_BLEND_L18_CONTROL"],
        "P9_4_SOURCE_REPLACE_GROUND_V_L18": ["P9_5_SOURCE_REPLACE_GROUND_RANDOM_SAME_MASS_V_L18"],
        "P9_6_SOURCE_GATE_ROLE_NEGATIVE_V_LAST": [
            "P9_7_SOURCE_GATE_ROLE_NEGATIVE_RANDOM_SAME_MASS_V_LAST"
        ],
        "P9_8_SOURCE_GATE_DISAGREEMENT_V_LAST": [
            "P9_9_SOURCE_GATE_DISAGREEMENT_RANDOM_SAME_MASS_V_LAST"
        ],
        "P9_10_SOURCE_GATE_DISAGREEMENT_K_LAST": [
            "P9_11_SOURCE_GATE_DISAGREEMENT_RANDOM_SAME_MASS_K_LAST"
        ],
        "P9_12_SOURCE_GATE_DISAGREEMENT_KV_LAST": [
            "P9_13_SOURCE_GATE_DISAGREEMENT_RANDOM_SAME_MASS_KV_LAST"
        ],
        "P9_14_SOURCE_REPLACE_DISAGREEMENT_K_LAST": [
            "P9_15_SOURCE_REPLACE_DISAGREEMENT_RANDOM_SAME_MASS_K_LAST"
        ],
        "P9_16_SOURCE_REPLACE_DISAGREEMENT_KV_LAST": [
            "P9_17_SOURCE_REPLACE_DISAGREEMENT_RANDOM_SAME_MASS_KV_LAST"
        ],
        "P9_18_SOURCE_BOOST_STABLE_AGREEMENT_K_LAST": [
            "P9_19_SOURCE_BOOST_STABLE_AGREEMENT_RANDOM_SAME_MASS_K_LAST"
        ],
        "P9_20_ATTENTION_BIAS_STABLE_AGREEMENT_LAST": [
            "P9_21_ATTENTION_BIAS_STABLE_AGREEMENT_RANDOM_SAME_MASS_LAST"
        ],
        "P9_22_ATTENTION_BIAS_SEMANTIC_SAMEGROUP_STABLE_AGREEMENT_LAST": [
            "P9_23_ATTENTION_BIAS_SEMANTIC_SAMEGROUP_STABLE_AGREEMENT_RANDOM_SAME_MASS_LAST"
        ],
        "P9_24_ATTENTION_BIAS_STABLE_AGREEMENT_TOPQ90_LAST": [
            "P9_25_ATTENTION_BIAS_STABLE_AGREEMENT_TOPQ90_RANDOM_SAME_MASS_LAST"
        ],
        "P9_26_ATTENTION_BIAS_STABLE_AGREEMENT_TOPQ80_LAST": [
            "P9_27_ATTENTION_BIAS_STABLE_AGREEMENT_TOPQ80_RANDOM_SAME_MASS_LAST"
        ],
        "P9_28_ATTENTION_BIAS_STABLE_AGREEMENT_TOPQ80_ALIGNED_LAST": [
            "P9_29_ATTENTION_BIAS_STABLE_AGREEMENT_TOPQ80_ALIGNED_RANDOM_SAME_MASS_LAST"
        ],
        "P9_30_SOURCE_REPLACE_STABLE_AGREEMENT_TOPQ80_KV_LAST": [
            "P9_31_SOURCE_REPLACE_STABLE_AGREEMENT_TOPQ80_RANDOM_SAME_MASS_KV_LAST"
        ],
        "P9_32_SOURCE_REPLACE_STABLE_AGREEMENT_TOPQ80_K_LIGHT_LAST": [
            "P9_33_SOURCE_REPLACE_STABLE_AGREEMENT_TOPQ80_RANDOM_SAME_MASS_K_LIGHT_LAST"
        ],
        "P9_34_ATTENTION_BIAS_STABLE_AGREEMENT_TOPQ80_MASS_AUDIT_LAST": [
            "P9_35_ATTENTION_BIAS_STABLE_AGREEMENT_TOPQ80_RANDOM_SAME_MASS_MASS_AUDIT_LAST"
        ],
        "P9_36_ATTENTION_BIAS_STABLE_AGREEMENT_TOPQ80_HEAD6_LAST": [
            "P9_37_ATTENTION_BIAS_STABLE_AGREEMENT_TOPQ80_RANDOM_SAME_MASS_HEAD6_LAST"
        ],
        "P9_38_ATTENTION_BIAS_STABLE_AGREEMENT_TOPQ80_HEADS0_6_8_LAST": [
            "P9_39_ATTENTION_BIAS_STABLE_AGREEMENT_TOPQ80_RANDOM_SAME_MASS_HEADS0_6_8_LAST"
        ],
        "P9_40_SOURCE_REPLACE_STABLE_AGREEMENT_TOPQ80_V_LAST": [
            "P9_41_SOURCE_REPLACE_STABLE_AGREEMENT_TOPQ80_RANDOM_SAME_MASS_V_LAST"
        ],
    }
    decisions = {
        candidate: _build_phase9_decision(
            rows,
            candidate=candidate,
            baseline="P9_0_NATIVE",
            controls=controls_by_candidate[candidate],
        )
        for candidate in candidates
    }
    payload = {
        "schema": "acl2_v78_phase9_swa_cache_value_carryover_v1",
        "output_root": str(args.output_root),
        "runs": rows,
        "baseline": "P9_0_NATIVE",
        "candidates": candidates,
        "controls_by_candidate": controls_by_candidate,
        "decisions": decisions,
        "phase9_any_gate_pass": bool(any(dec.get("phase9_gate_pass") for dec in decisions.values())),
        "actuator_boundary": (
            "P9_1/P9_2 modify committed SWA history cache K/V via write-side post-cache blend; "
            "P9_4 modifies SWA overlap source V during read; P9_6/P9_8 gate SWA overlap source V during read; "
            "P9_10/P9_12 gate SWA overlap source K/KV during read; "
            "P9_14/P9_16 replace SWA overlap source K/KV toward current overlap during read; "
            "P9_18 boosts low-dynamic current/source-agreeing SWA source K during read; "
            "P9_20 applies stable-agreement SWA overlap attention-bias route reweighting during read; "
            "P9_22 applies the same attention-bias route only on semantic same-group "
            "structure/static/lowstuff overlap tokens; "
            "P9_24/P9_26 apply the stable-agreement attention-bias only on q90/q80 source tokens; "
            "P9_28 applies q80 stable-agreement only on aligned current-head to previous-tail token pairs; "
            "P9_30 replaces q80 stable-agreement SWA overlap source K/V toward aligned current overlap K/V; "
            "P9_32 lightly replaces only q80 stable-agreement SWA overlap source K toward aligned current overlap K; "
            "P9_34 repeats P9_26 with sampled selected-token attention-mass diagnostics; "
            "P9_40 replaces only q80 stable-agreement SWA overlap source V after selected-mask-conditioned "
            "Q/K/V alignment indicated stronger value-side than key-side carry-over. "
            "No C9 chunk-wise replay parameters are used."
        ),
        "control_gap_note": (
            "This runner includes an all-blend nonsemantic cache control and semantic_ground_random_same_mass "
            "source-replace control, plus same-mass random source-gate controls for role-negative and "
            "runtime-disagreement V/K/KV gates/replacements, stable-agreement K boost, and "
            "stable-agreement overlap-bias route reweighting. P9_22/P9_23 add a "
            "semantic same-group route-bias pair after the P9_20 group audit; "
            "P9_24/P9_25 and P9_26/P9_27 add q90/q80 sparse stable-route pairs; "
            "P9_28/P9_29 add an aligned q80 route pair; "
            "P9_30/P9_31 test q80 stable-route K/V content replacement inspired by KV alignment/merging work; "
            "P9_32/P9_33 narrow the P9_30 failure to low-alpha K-only alignment; "
            "P9_34/P9_35 add heavy-hitter-style attention-mass observability without changing the P9_26 actuator; "
            "P9_40/P9_41 test the follow-up V-only content-replace pair after offline selected-mask "
            "Q/K/V audit showed V L26 selected alignment stronger than patch-random while K L26 stayed risky."
        ),
    }
    metrics_json = args.output_root / "phase9_swa_cache_value_metrics.json"
    metrics_csv = args.output_root / "phase9_swa_cache_value_metrics.csv"
    decision_json = args.output_root / "phase9_swa_cache_value_decision.json"
    metrics_json.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(metrics_csv, rows)
    decision_json.write_text(
        json.dumps(
            {
                "phase9_any_gate_pass": payload["phase9_any_gate_pass"],
                "decisions": decisions,
                "actuator_boundary": payload["actuator_boundary"],
                "control_gap_note": payload["control_gap_note"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            _jsonable({"phase9_any_gate_pass": payload["phase9_any_gate_pass"], "decisions": decisions}),
            indent=2,
            sort_keys=True,
        )
    )
    print(f"wrote_json={metrics_json}")
    print(f"wrote_csv={metrics_csv}")
    print(f"wrote_decision={decision_json}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--chunks", default="6")
    parser.add_argument("--cases", default=",".join(PHASE9_CASES))
    parser.add_argument("--gpus", default="0,1,2,3,4,5")
    parser.add_argument("--workdir", type=Path, default=Path.cwd())
    parser.add_argument("--conda", type=Path, default=DEFAULT_CONDA)
    parser.add_argument("--conda-env", default="loger")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--gt", type=Path, default=DEFAULT_GT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage-c-cache-dir", type=Path, default=DEFAULT_STAGE_C_CACHE)
    parser.add_argument("--cuda-alloc-conf", default=DEFAULT_CUDA_ALLOC_CONF)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--chunk-overlap", type=int, default=3)
    parser.add_argument("--context-chunks", type=int, default=2)
    parser.add_argument("--read-single-layer", type=int, default=18)
    parser.add_argument("--swa-layer", type=int, default=18)
    parser.add_argument(
        "--swa-source-layer-mode",
        choices=("all", "first", "last", "single"),
        default="last",
        help=(
            "Layer selector for SWA source replace/gate hook calls.  The Phase8 visual clue is named "
            "SWA cache-V L18, but the runtime SWA hook exposes compact SWA call indices; last is the "
            "closest auditable default and avoids the previous single=18 no-op."
        ),
    )
    parser.add_argument(
        "--swa-source-single-layer",
        type=int,
        default=-1,
        help="SWA source hook index used only when --swa-source-layer-mode=single.",
    )
    parser.add_argument("--swa-write-alpha", type=float, default=0.35)
    parser.add_argument("--swa-replace-alpha", type=float, default=0.35)
    parser.add_argument("--swa-gate-rho", type=float, default=0.35)
    parser.add_argument("--swa-gate-min", type=float, default=0.65)
    parser.add_argument("--swa-bias-beta", type=float, default=0.35)
    parser.add_argument("--swa-bias-min-keep", type=float, default=1e-4)
    parser.add_argument("--swa-bias-attention-mass-max-queries", type=int, default=64)
    parser.add_argument("--swa-write-scope", default="tail_overlap", choices=("all", "tail_overlap", "head_overlap", "both_overlap"))
    parser.add_argument("--semantic-role-highd-quantile", type=float, default=0.75)
    parser.add_argument("--semantic-role-low-trust", type=float, default=0.55)
    parser.add_argument("--semantic-role-swa-protect-scale", type=float, default=0.35)
    parser.add_argument("--swa-overlap-feature-dump-dtype", default="float16")
    parser.add_argument(
        "--v68-export-full-pca-debug",
        type=int,
        default=0,
        help="Forward --v68_export_full_pca_debug=1 to run_pipeline_abc_v2.py.",
    )
    parser.add_argument(
        "--v68-pca-taps",
        default="swa_k,swa_v,swa_cache_k,swa_cache_v",
        help="Comma-separated v68 PCA taps/aliases to dump when full PCA debug is enabled.",
    )
    parser.add_argument(
        "--v68-pca-layers",
        default="18,26",
        help="Comma-separated layer ids for compact v68 PCA feature dumps.",
    )
    parser.add_argument(
        "--v68-pca-max-feature-dim",
        type=int,
        default=16,
        help="Maximum trailing feature dimension saved by v68 compact PCA feature dumps.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    chunks = _parse_csv_ints(args.chunks)
    gpus = _parse_csv_ints(args.gpus)
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU id")
    cases = [case.strip() for case in str(args.cases).split(",") if case.strip()]
    unknown = [case for case in cases if case not in PHASE9_CASES]
    if unknown:
        raise ValueError(f"unknown cases: {unknown}; known={sorted(PHASE9_CASES)}")

    jobs: List[Dict[str, Any]] = []
    gpu_cursor = 0
    for chunk in chunks:
        window = _context_window(args, int(chunk))
        for case in cases:
            out_dir = args.output_root / f"chunk{chunk:02d}" / case
            cmd = _build_command(args, chunk=chunk, case=case, out_dir=out_dir)
            skipped = bool(args.skip_existing and (out_dir / "01.txt").exists() and (out_dir / "hmc_state_hash.jsonl").exists())
            cfg = PHASE9_CASES[case]
            jobs.append(
                {
                    "chunk": int(chunk),
                    "context_start_chunk": int(window["context_start_chunk"]),
                    "context_chunks": int(window["context_chunks"]),
                    "start_frame": int(window["start_frame"]),
                    "end_frame": int(window["end_frame"]),
                    "target_start_frame": int(window["target_start_frame"]),
                    "target_end_frame": int(window["target_end_frame"]),
                    "case": case,
                    "family": cfg["family"],
                    "gpu": int(gpus[gpu_cursor % len(gpus)]),
                    "out_dir": str(out_dir),
                    "cmd": cmd,
                    "cmd_shell": shlex.join(cmd),
                    "workdir": str(args.workdir),
                    "read_cue_source_effective": "dyn",
                    "semantic_role_policy_effective": cfg.get("semantic_role_policy", "none"),
                    "semantic_memory_paths_effective": cfg.get("semantic_memory_paths", ""),
                    "swa_layer_effective": int(args.swa_layer),
                    "swa_source_layer_mode_effective": str(args.swa_source_layer_mode),
                    "swa_source_single_layer_effective": int(args.swa_source_single_layer),
                    "swa_write_alpha_effective": float(args.swa_write_alpha),
                    "swa_replace_alpha_effective": float(args.swa_replace_alpha),
                    "swa_gate_rho_effective": float(args.swa_gate_rho),
                    "swa_gate_min_effective": float(args.swa_gate_min),
                    "swa_bias_beta_effective": float(
                        args.swa_bias_beta if int(_case_value(cfg, "enable_swa_overlap_bias", 0)) else 0.0
                    ),
                    "swa_bias_min_keep_effective": float(args.swa_bias_min_keep),
                    "swa_bias_record_attention_mass_effective": bool(
                        int(_case_value(cfg, "swa_overlap_bias_record_attention_mass", 0))
                    ),
                    "swa_bias_attention_mass_max_queries_effective": int(args.swa_bias_attention_mass_max_queries),
                    "swa_bias_head_indices_effective": str(_case_value(cfg, "swa_overlap_bias_head_indices", "")),
                    "cuda_alloc_conf": str(args.cuda_alloc_conf),
                    "skipped": skipped,
                    "returncode": 0 if skipped else None,
                }
            )
            gpu_cursor += 1

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "phase9_swa_cache_value_run_manifest.json"
    manifest: Dict[str, Any] = {"args": vars(args), "jobs": jobs}
    manifest_path.write_text(json.dumps(_jsonable(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"planned_jobs={len(jobs)} manifest={manifest_path}")
    if args.dry_run:
        return

    run_jobs = [job for job in jobs if not job["skipped"]]
    completed: List[Dict[str, Any]] = [job for job in jobs if job["skipped"]]
    completed_lock = threading.Lock()
    jobs_by_gpu: Dict[int, List[Dict[str, Any]]] = {int(gpu): [] for gpu in gpus}
    for job in run_jobs:
        jobs_by_gpu[int(job["gpu"])].append(job)

    def _run_gpu_queue(gpu: int, queue: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        gpu_results: List[Dict[str, Any]] = []
        for job in queue:
            result = _run_job(job)
            gpu_results.append(result)
            with completed_lock:
                completed.append(result)
                manifest["jobs"] = completed + [item for item in jobs if item not in completed]
                manifest_path.write_text(json.dumps(_jsonable(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(
                f"[gpu{gpu}] chunk={result['chunk']} case={result['case']} "
                f"returncode={result['returncode']} duration={result['duration_sec']:.1f}s"
            )
        return gpu_results

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [pool.submit(_run_gpu_queue, gpu, queue) for gpu, queue in jobs_by_gpu.items() if queue]
        for fut in concurrent.futures.as_completed(futures):
            fut.result()

    completed_by_key = {(int(job["chunk"]), str(job["case"])): job for job in completed}
    ordered = [completed_by_key.get((int(job["chunk"]), str(job["case"])), job) for job in jobs]
    failed = [job for job in ordered if int(job.get("returncode") or 0) != 0]
    manifest["jobs"] = ordered
    manifest["completed_count"] = int(len([job for job in ordered if job.get("returncode") is not None]))
    manifest["failed_jobs"] = [
        {"chunk": int(job["chunk"]), "case": str(job["case"]), "returncode": int(job.get("returncode") or -1)}
        for job in failed
    ]
    manifest_path.write_text(json.dumps(_jsonable(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"completed={manifest['completed_count']} failed={len(failed)} manifest={manifest_path}")
    if failed:
        return
    _evaluate(args, ordered)


if __name__ == "__main__":
    main()
