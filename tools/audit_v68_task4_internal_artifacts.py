#!/usr/bin/env python3
"""Audit existing ACL2 v68 Task 4 internal artifact readiness.

This script is intentionally read-only.  It inspects saved feature dumps and
debug traces, then reports which Task 4 artifacts are truly present versus only
available as scalar/branch summaries.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


try:
    import torch
except Exception:  # pragma: no cover - reported in output when it happens.
    torch = None  # type: ignore[assignment]


DEFAULT_ROOT = "results/kitti01_hmc_v2/acl2_v68_integrated_cueconstruction"
DEFAULT_CHUNKS = "6,7,8,10,12,19,20,29,30,31,32"
DEFAULT_LAYERS = "5,7"


def _parse_int_csv(value: str) -> List[int]:
    out: List[int] = []
    for part in str(value or "").split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


def _safe_rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _read_jsonl(path: Path, max_rows: Optional[int] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
            if max_rows is not None and len(rows) >= max_rows:
                break
    return rows


def _walk_json(obj: Any, prefix: str = "") -> Iterable[Tuple[str, Any]]:
    if isinstance(obj, dict):
        for key, value in obj.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            yield child, value
            yield from _walk_json(value, child)
    elif isinstance(obj, list):
        for idx, value in enumerate(obj):
            child = f"{prefix}[{idx}]"
            yield child, value
            yield from _walk_json(value, child)


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "shape"):
        return {"shape": list(obj.shape), "dtype": str(getattr(obj, "dtype", ""))}
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _status_from_counts(pass_count: int, partial_count: int = 0) -> str:
    if pass_count > 0:
        return "pass"
    if partial_count > 0:
        return "partial"
    return "missing"


def _load_torch_file(path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if torch is None:
        return None, "torch_unavailable"
    try:
        payload = torch.load(path, map_location="cpu")
    except TypeError:
        payload = torch.load(path, map_location="cpu")  # type: ignore[misc]
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if not isinstance(payload, dict):
        return None, f"unexpected_payload_type={type(payload).__name__}"
    return payload, None


def audit_qk_features(root: Path, chunks: Sequence[int], required_layers: Sequence[int]) -> Dict[str, Any]:
    feature_dir = root / "phaseC_target_feature_dumps" / "features"
    required_taps = ["global_q_raw_patchvec_layers", "global_k_raw_patchvec_layers"]
    chunk_reports: List[Dict[str, Any]] = []
    ok_count = 0
    missing_count = 0
    bad_count = 0

    for chunk in chunks:
        path = feature_dir / f"chunk_{chunk:03d}.pt"
        report: Dict[str, Any] = {
            "chunk": int(chunk),
            "path": str(path),
            "exists": path.exists(),
            "schema": None,
            "tap_status": {},
            "ok": False,
            "error": None,
        }
        if not path.exists():
            missing_count += 1
            report["error"] = "missing_feature_file"
            chunk_reports.append(report)
            continue

        payload, error = _load_torch_file(path)
        if error:
            bad_count += 1
            report["error"] = error
            chunk_reports.append(report)
            continue

        assert payload is not None
        report["schema"] = payload.get("schema")
        taps = payload.get("taps", {})
        if not isinstance(taps, dict):
            taps = {}

        tap_ok = True
        for tap in required_taps:
            meta = taps.get(tap, {})
            saved = payload.get(f"tap::{tap}")
            selected_layers = meta.get("selected_layers") if isinstance(meta, dict) else None
            if not isinstance(selected_layers, list):
                selected_layers = []
            required_layers_present = set(int(v) for v in required_layers).issubset(
                set(int(v) for v in selected_layers)
            )
            available = bool(meta.get("available", False)) if isinstance(meta, dict) else False
            tensor_present = torch is not None and bool(torch.is_tensor(saved))
            tap_report = {
                "available": available,
                "tensor_present": tensor_present,
                "selected_layers": selected_layers,
                "required_layers_present": required_layers_present,
                "saved_shape": meta.get("saved_shape") if isinstance(meta, dict) else None,
                "reason": meta.get("reason") if isinstance(meta, dict) else "tap_meta_missing",
            }
            report["tap_status"][tap] = tap_report
            tap_ok = tap_ok and available and tensor_present and required_layers_present

        report["ok"] = bool(
            report["schema"] == "acl2_v68_layer_pca_feature_dump_v1" and tap_ok
        )
        if report["ok"]:
            ok_count += 1
        else:
            bad_count += 1
        chunk_reports.append(report)

    if ok_count == len(chunks):
        status = "pass"
    elif ok_count > 0:
        status = "partial"
    else:
        status = "missing"

    return {
        "artifact": "SAVE_V68_QK_FEATURES",
        "status": status,
        "feature_dir": str(feature_dir),
        "required_chunks": [int(v) for v in chunks],
        "required_layers": [int(v) for v in required_layers],
        "required_taps": required_taps,
        "ok_count": int(ok_count),
        "missing_count": int(missing_count),
        "bad_count": int(bad_count),
        "chunk_reports": chunk_reports,
    }


def audit_source_attention(root: Path, chunks: Sequence[int]) -> Dict[str, Any]:
    hmc_paths = sorted(
        set(
            list((root / "phaseD_read_online_smoke").glob("**/hmc_state_hash.jsonl"))
            + list((root / "phaseE_read_multichunk").glob("**/candidate/hmc_state_hash.jsonl"))
            + list((root / "task4_source_attention_mass_smoke").glob("**/hmc_state_hash.jsonl"))
        )
    )
    scanned_rows = 0
    source_attention_map_key_count = 0
    attention_mass_available_true = 0
    attention_mass_requested_true = 0
    source_top_quantile_key_count = 0
    sample_paths: List[str] = []
    qk_proxy_paths = sorted((root / "task4_qk_source_attention_proxy_maps" / "maps").glob("chunk_*_qk_source_attention_proxy.pt"))
    qk_proxy_ok_count = 0
    qk_proxy_shape_sample: Optional[List[int]] = None
    qk_proxy_not_raw = None
    sampled_map_paths = sorted(root.glob("**/source_attention_maps/chunk_*_source_attention_sample.pt"))
    sampled_map_ok_count = 0
    sampled_map_shape_sample: Optional[List[int]] = None
    sampled_map_not_full: Optional[bool] = None
    full_marginal_paths = sorted(
        root.glob("**/source_attention_maps/chunk_*_source_attention_fullquery_marginal.pt")
    )
    full_marginal_ok_count = 0
    full_marginal_bad_count = 0
    full_marginal_chunks: List[int] = []
    full_marginal_layers: List[int] = []
    full_marginal_shape_sample: Optional[List[int]] = None
    full_marginal_pairwise_stored: Optional[bool] = None

    for path in qk_proxy_paths:
        payload, error = _load_torch_file(path)
        if error or payload is None:
            continue
        if payload.get("schema") != "acl2_v68_qk_pooled_source_attention_proxy_v1":
            continue
        tensor = payload.get("source_attention_proxy")
        if torch is not None and torch.is_tensor(tensor):
            qk_proxy_ok_count += 1
            if qk_proxy_shape_sample is None:
                qk_proxy_shape_sample = [int(v) for v in tensor.shape]
            qk_proxy_not_raw = bool(payload.get("not_raw_model_sdpa_attention", True))

    for path in sampled_map_paths:
        payload, error = _load_torch_file(path)
        if error or payload is None:
            continue
        if payload.get("schema") != "acl2_v68_sampled_source_attention_map_v1":
            continue
        tensor = payload.get("attention_before_control")
        if torch is not None and torch.is_tensor(tensor):
            sampled_map_ok_count += 1
            if sampled_map_shape_sample is None:
                sampled_map_shape_sample = [int(v) for v in tensor.shape]
            sampled_map_not_full = bool(payload.get("sampled_not_full_attention_map", True))

    for path in full_marginal_paths:
        payload, error = _load_torch_file(path)
        if error or payload is None:
            full_marginal_bad_count += 1
            continue
        if payload.get("schema") != "acl2_v68_fullquery_source_attention_marginal_v1":
            full_marginal_bad_count += 1
            continue
        tensor = payload.get("source_attention_before_marginal")
        if torch is None or not torch.is_tensor(tensor):
            full_marginal_bad_count += 1
            continue
        full_marginal_ok_count += 1
        chunk_idx = int(payload.get("chunk_idx", -1))
        layer_idx = int(payload.get("layer", -1))
        if chunk_idx >= 0:
            full_marginal_chunks.append(chunk_idx)
        if layer_idx >= 0:
            full_marginal_layers.append(layer_idx)
        if full_marginal_shape_sample is None:
            full_marginal_shape_sample = [int(v) for v in tensor.shape]
        full_marginal_pairwise_stored = bool(payload.get("pairwise_attention_matrix_stored", False))

    for path in hmc_paths:
        rows = _read_jsonl(path)
        scanned_rows += len(rows)
        if len(sample_paths) < 8:
            sample_paths.append(str(path))
        for row in rows:
            for key, value in _walk_json(row):
                low = key.lower()
                if "source_attention_map" in low:
                    source_attention_map_key_count += 1
                if key.endswith("attention_mass_available") and value is True:
                    attention_mass_available_true += 1
                if key.endswith("attention_mass_requested") and value is True:
                    attention_mass_requested_true += 1
                if "source_attention_top_quantile" in low:
                    source_top_quantile_key_count += 1

    code_evidence = [
        "run_pipeline_abc_v2.py uses reason=source_attention_map_unavailable_in_control_prior for several source-attention groups",
        "loger/pipeline/hybrid_memory_controller.py records attention_mass_available only when attention mass metrics are collected",
    ]
    full_marginal_unique_chunks = sorted(set(full_marginal_chunks))
    full_marginal_unique_layers = sorted(set(full_marginal_layers))
    required_chunk_set = {int(v) for v in chunks}
    full_marginal_covers_required = (
        bool(required_chunk_set) and required_chunk_set.issubset(set(full_marginal_unique_chunks))
    )
    if full_marginal_ok_count > 0 and full_marginal_covers_required:
        status = "pass"
    elif (
        full_marginal_ok_count > 0
        or source_attention_map_key_count > 0
        or
        sampled_map_ok_count > 0
        or qk_proxy_ok_count > 0
        or attention_mass_available_true > 0
        or source_top_quantile_key_count > 0
    ):
        status = "partial"
    else:
        status = "blocked"

    return {
        "artifact": "SAVE_V68_SOURCE_ATTENTION_MAP",
        "status": status,
        "hmc_files_scanned": len(hmc_paths),
        "rows_scanned": scanned_rows,
        "source_attention_map_key_count": source_attention_map_key_count,
        "attention_mass_available_true": attention_mass_available_true,
        "attention_mass_requested_true": attention_mass_requested_true,
        "source_attention_top_quantile_key_count": source_top_quantile_key_count,
        "sampled_source_attention_map_files": len(sampled_map_paths),
        "sampled_source_attention_map_ok_count": sampled_map_ok_count,
        "sampled_source_attention_map_shape_sample": sampled_map_shape_sample,
        "sampled_source_attention_map_not_full": sampled_map_not_full,
        "fullquery_source_attention_marginal_files": len(full_marginal_paths),
        "fullquery_source_attention_marginal_ok_count": int(full_marginal_ok_count),
        "fullquery_source_attention_marginal_bad_count": int(full_marginal_bad_count),
        "fullquery_source_attention_marginal_chunks": full_marginal_unique_chunks,
        "fullquery_source_attention_marginal_layers": full_marginal_unique_layers,
        "fullquery_source_attention_marginal_covers_required_chunks": full_marginal_covers_required,
        "fullquery_source_attention_marginal_shape_sample": full_marginal_shape_sample,
        "fullquery_source_attention_marginal_pairwise_matrix_stored": full_marginal_pairwise_stored,
        "qk_pooled_source_attention_proxy_files": len(qk_proxy_paths),
        "qk_pooled_source_attention_proxy_ok_count": qk_proxy_ok_count,
        "qk_pooled_source_attention_proxy_shape_sample": qk_proxy_shape_sample,
        "qk_pooled_source_attention_proxy_not_raw_model_attention": qk_proxy_not_raw,
        "sample_evidence_paths": sample_paths,
        "code_evidence": code_evidence,
        "interpretation": (
            "full-query raw source-attention marginal maps cover all required chunks"
            if status == "pass"
            else "full-query raw source-attention marginal maps exist but do not cover all required chunks"
            if full_marginal_ok_count > 0
            else "source-attention map keys found in traces, but full-query source marginal artifacts not found"
            if source_attention_map_key_count > 0
            else "sampled raw QK source-attention maps found, but full source-attention map not found"
            if sampled_map_ok_count > 0
            else "qk pooled source-attention proxy maps found, but full raw model source-attention map not found"
            if qk_proxy_ok_count > 0
            else "full source-attention map not found in existing traces"
        ),
    }


def audit_ttt_delta(root: Path, chunks: Sequence[int]) -> Dict[str, Any]:
    hmc_paths = sorted((root / "phaseD_ttt_online_smoke").glob("**/hmc_state_hash.jsonl"))
    dump_paths = sorted(
        (root / "task4_ttt_post_delta_dump_smoke").glob(
            "**/delta_dumps/chunk_*_ttt_post_zp_delta.pt"
        )
    )
    spatial_paths = sorted(
        set(root.glob("**/ttt_spatial_post_delta_maps/chunk_*_ttt_spatial_post_delta_map.pt"))
        | set((root / "task4_ttt_spatial_post_delta_maps").glob("**/*.pt"))
    )
    scanned_rows = 0
    debug_rows = 0
    post_delta_count_sum = 0
    action_delta_means: List[float] = []
    native_delta_means: List[float] = []
    post_delta_means: List[float] = []
    branch_summary_rows = 0
    branch_summary_entry_count = 0
    sample_paths: List[str] = []
    dump_ok_count = 0
    dump_rows_total = 0
    dump_tensor_groups_total = 0
    dump_shape_sample: Optional[Dict[str, Dict[str, List[int]]]] = None
    dump_fast_weight_not_spatial: Optional[bool] = None
    dump_sample_paths: List[str] = []
    spatial_ok_count = 0
    spatial_bad_count = 0
    spatial_chunks: List[int] = []
    spatial_layer_branch_rows_total = 0
    spatial_shape_sample: Optional[Dict[str, List[int]]] = None
    spatial_projection_flags: Dict[str, int] = {}
    spatial_stats_sample: Optional[Dict[str, Any]] = None

    for path in dump_paths:
        if len(dump_sample_paths) < 8:
            dump_sample_paths.append(str(path))
        payload, error = _load_torch_file(path)
        if error or payload is None:
            continue
        if payload.get("schema") != "acl2_v68_ttt_post_zp_delta_dump_v1":
            continue
        rows = payload.get("rows")
        deltas = payload.get("deltas")
        if not isinstance(rows, list) or not isinstance(deltas, dict):
            continue
        dump_ok_count += 1
        dump_rows_total += len(rows)
        dump_tensor_groups_total += len(deltas)
        dump_fast_weight_not_spatial = bool(
            payload.get("tensors_are_fast_weight_deltas_not_spatial_token_maps", True)
        )
        if dump_shape_sample is None:
            dump_shape_sample = {}
            for group_name, tensors in list(deltas.items())[:2]:
                if not isinstance(tensors, dict):
                    continue
                dump_shape_sample[str(group_name)] = {}
                for tensor_name, tensor in tensors.items():
                    if torch is not None and torch.is_tensor(tensor):
                        dump_shape_sample[str(group_name)][str(tensor_name)] = [
                            int(v) for v in tensor.shape
                        ]
        del payload

    for path in spatial_paths:
        if len(dump_sample_paths) < 12:
            dump_sample_paths.append(str(path))
        payload, error = _load_torch_file(path)
        if error or payload is None:
            spatial_bad_count += 1
            continue
        if payload.get("schema") != "acl2_v68_ttt_spatial_post_delta_map_v1":
            spatial_bad_count += 1
            continue
        prior = payload.get("ttt_write_prior_patch")
        committed = payload.get("committed_post_delta_norm_projection_patch")
        native = payload.get("native_delta_norm_projection_patch")
        action = payload.get("action_delta_norm_projection_patch")
        rows = payload.get("layer_branch_rows")
        if not (
            torch is not None
            and torch.is_tensor(prior)
            and torch.is_tensor(committed)
            and torch.is_tensor(native)
            and torch.is_tensor(action)
            and isinstance(rows, list)
            and rows
        ):
            spatial_bad_count += 1
            continue
        if not bool(payload.get("spatial_token_aligned", False)):
            spatial_bad_count += 1
            continue
        spatial_ok_count += 1
        spatial_layer_branch_rows_total += len(rows)
        chunk_idx = int(payload.get("chunk_idx", -1))
        if chunk_idx >= 0:
            spatial_chunks.append(chunk_idx)
        for flag in (
            "projection_not_raw_per_token_fast_weight_delta",
            "tensors_are_fast_weight_deltas_not_spatial_token_maps",
            "measures_output_delta_not_internal_fast_weight_delta",
        ):
            key = f"{flag}={bool(payload.get(flag, False))}"
            spatial_projection_flags[key] = spatial_projection_flags.get(key, 0) + 1
        if spatial_shape_sample is None:
            spatial_shape_sample = {
                "ttt_write_prior_patch": [int(v) for v in prior.shape],
                "committed_post_delta_norm_projection_patch": [int(v) for v in committed.shape],
                "native_delta_norm_projection_patch": [int(v) for v in native.shape],
                "action_delta_norm_projection_patch": [int(v) for v in action.shape],
            }
            spatial_stats_sample = payload.get("stats") if isinstance(payload.get("stats"), dict) else None
        del payload

    for path in hmc_paths:
        rows = _read_jsonl(path)
        scanned_rows += len(rows)
        if len(sample_paths) < 8:
            sample_paths.append(str(path))
        for row in rows:
            if bool(row.get("probe_ttt_write_debug_available", False)):
                debug_rows += 1
            count = row.get("probe_ttt_write_post_delta_norm_count")
            if isinstance(count, (int, float)):
                post_delta_count_sum += int(count)
            for key, target in [
                ("probe_ttt_write_action_delta_norm_mean", action_delta_means),
                ("probe_ttt_write_native_delta_norm_mean", native_delta_means),
                ("probe_ttt_write_post_delta_norm_mean", post_delta_means),
            ]:
                value = row.get(key)
                if isinstance(value, (int, float)):
                    target.append(float(value))
            branch_summary = row.get("probe_ttt_write_layer_branch_summary")
            if isinstance(branch_summary, dict) and branch_summary:
                branch_summary_rows += 1
                branch_summary_entry_count += len(branch_summary)
            elif isinstance(branch_summary, list) and branch_summary:
                branch_summary_rows += 1
                branch_summary_entry_count += len(branch_summary)

    scalar_summary_available = bool(
        debug_rows > 0
        and post_delta_count_sum > 0
        and action_delta_means
        and native_delta_means
    )
    branch_summary_available = bool(
        scalar_summary_available
        and branch_summary_rows > 0
    )
    dump_available = dump_ok_count > 0
    required_chunk_set = {int(v) for v in chunks}
    spatial_unique_chunks = sorted(set(spatial_chunks))
    spatial_covers_required = bool(required_chunk_set) and required_chunk_set.issubset(set(spatial_unique_chunks))
    status = (
        "pass"
        if spatial_ok_count > 0 and spatial_covers_required
        else "partial"
        if scalar_summary_available or dump_available or spatial_ok_count > 0
        else "missing"
    )
    return {
        "artifact": "SAVE_V68_TTT_POST_DELTA_MAP_AND_NATIVE_ACTION_DELTA",
        "status": status,
        "hmc_files_scanned": len(hmc_paths),
        "ttt_post_delta_dump_files": len(dump_paths),
        "ttt_post_delta_dump_ok_count": int(dump_ok_count),
        "ttt_post_delta_dump_rows_total": int(dump_rows_total),
        "ttt_post_delta_dump_tensor_groups_total": int(dump_tensor_groups_total),
        "ttt_post_delta_dump_shape_sample": dump_shape_sample,
        "ttt_post_delta_dump_fast_weight_not_spatial": dump_fast_weight_not_spatial,
        "ttt_spatial_post_delta_map_files": len(spatial_paths),
        "ttt_spatial_post_delta_map_ok_count": int(spatial_ok_count),
        "ttt_spatial_post_delta_map_bad_count": int(spatial_bad_count),
        "ttt_spatial_post_delta_map_chunks": spatial_unique_chunks,
        "ttt_spatial_post_delta_map_covers_required_chunks": spatial_covers_required,
        "ttt_spatial_post_delta_map_layer_branch_rows_total": int(spatial_layer_branch_rows_total),
        "ttt_spatial_post_delta_map_shape_sample": spatial_shape_sample,
        "ttt_spatial_post_delta_map_projection_flags": spatial_projection_flags,
        "ttt_spatial_post_delta_map_stats_sample": spatial_stats_sample,
        "rows_scanned": scanned_rows,
        "debug_rows": debug_rows,
        "post_delta_norm_count_sum": int(post_delta_count_sum),
        "action_delta_norm_mean_values": action_delta_means,
        "native_delta_norm_mean_values": native_delta_means,
        "post_delta_norm_mean_values": post_delta_means,
        "layer_branch_summary_rows": branch_summary_rows,
        "layer_branch_summary_entry_count": branch_summary_entry_count,
        "sample_evidence_paths": sample_paths + dump_sample_paths,
        "interpretation": (
            "TTT spatial/token-aligned post-delta projection maps cover all required chunks"
            if status == "pass"
            else "TTT spatial/token-aligned post-delta projection maps exist but do not cover all required chunks"
            if spatial_ok_count > 0
            else
            "TTT fast-weight post-ZP delta dumps exist, but they are not spatial/token-aligned post-delta maps"
            if dump_available and dump_fast_weight_not_spatial
            else "TTT dense post-delta dumps exist"
            if dump_available
            else
            "TTT scalar and layer/branch delta summaries exist, but no dense post-delta map artifact was found"
            if branch_summary_available
            else "TTT scalar delta summaries exist, but no dense post-delta map artifact was found"
            if scalar_summary_available
            else "TTT delta evidence not found in scanned traces"
        ),
    }


def audit_overlap_features(root: Path, chunks: Sequence[int]) -> Dict[str, Any]:
    trace_paths = sorted(
        set(
            list((root / "phaseD_merge_online_smoke").glob("**/merge_state_trace.jsonl"))
            + list((root / "phaseC_target_feature_dumps").glob("chunk_*/merge_state_trace.jsonl"))
        )
    )
    scanned_rows = 0
    residual_rows = 0
    fit_success_rows = 0
    overlap_key_rows = 0
    overlap_keys: Dict[str, int] = {}
    sample_paths: List[str] = []
    qk_proxy_paths = sorted((root / "task4_qk_source_attention_proxy_maps" / "maps").glob("chunk_*_qk_source_attention_proxy.pt"))
    qk_overlap_proxy_ok_count = 0
    qk_overlap_map_shape_sample: Optional[Dict[str, List[int]]] = None
    runtime_dump_paths = sorted(
        set((root / "task4_swa_overlap_feature_dumps").glob("**/*.pt"))
        | set(root.glob("**/swa_overlap_feature_maps/*.pt"))
    )
    runtime_dump_ok_count = 0
    runtime_dump_bad_count = 0
    runtime_dump_chunks: List[int] = []
    runtime_dump_kinds: Dict[str, int] = {}
    runtime_dump_shape_sample: Optional[Dict[str, List[int]]] = None

    for path in qk_proxy_paths:
        payload, error = _load_torch_file(path)
        if error or payload is None:
            continue
        tail_head = payload.get("tail_query_to_head_source_map")
        head_tail = payload.get("head_query_to_tail_source_map")
        if torch is not None and torch.is_tensor(tail_head) and torch.is_tensor(head_tail):
            qk_overlap_proxy_ok_count += 1
            if qk_overlap_map_shape_sample is None:
                qk_overlap_map_shape_sample = {
                    "tail_query_to_head_source_map": [int(v) for v in tail_head.shape],
                    "head_query_to_tail_source_map": [int(v) for v in head_tail.shape],
                }

    for path in runtime_dump_paths:
        payload, error = _load_torch_file(path)
        if error or payload is None:
            runtime_dump_bad_count += 1
            continue
        if payload.get("schema") != "acl2_v68_swa_overlap_feature_map_v1":
            runtime_dump_bad_count += 1
            continue
        Dq = payload.get("Dq_overlap")
        Ds = payload.get("Ds_overlap")
        score = payload.get("score_overlap")
        control = payload.get("control_overlap")
        if not (
            torch is not None
            and torch.is_tensor(Dq)
            and torch.is_tensor(Ds)
            and torch.is_tensor(score)
            and torch.is_tensor(control)
        ):
            runtime_dump_bad_count += 1
            continue
        runtime_dump_ok_count += 1
        chunk_idx = int(payload.get("chunk_idx", -1))
        if chunk_idx >= 0:
            runtime_dump_chunks.append(chunk_idx)
        kind = str(payload.get("kind", "unknown"))
        runtime_dump_kinds[kind] = runtime_dump_kinds.get(kind, 0) + 1
        if runtime_dump_shape_sample is None:
            runtime_dump_shape_sample = {
                "Dq_overlap": [int(v) for v in Dq.shape],
                "Ds_overlap": [int(v) for v in Ds.shape],
                "score_overlap": [int(v) for v in score.shape],
                "control_overlap": [int(v) for v in control.shape],
            }
        if len(sample_paths) < 8:
            sample_paths.append(str(path))

    for path in trace_paths:
        rows = _read_jsonl(path)
        scanned_rows += len(rows)
        if len(sample_paths) < 8:
            sample_paths.append(str(path))
        for row in rows:
            row_has_overlap_key = False
            for key, value in _walk_json(row):
                if "overlap" in key.lower():
                    overlap_keys[key] = overlap_keys.get(key, 0) + 1
                    row_has_overlap_key = True
                if key.endswith("semantic_merge_overlap_residual") and isinstance(value, (int, float)):
                    residual_rows += 1
                if key.endswith("semantic_merge_fit_success") and value is True:
                    fit_success_rows += 1
            if row_has_overlap_key:
                overlap_key_rows += 1

    runtime_unique_chunks = sorted(set(runtime_dump_chunks))
    required_chunk_set = {int(v) for v in chunks}
    runtime_covers_required = bool(required_chunk_set) and required_chunk_set.issubset(set(runtime_unique_chunks))
    status = (
        "pass"
        if runtime_dump_ok_count > 0 and runtime_covers_required
        else "partial"
        if (
            runtime_dump_ok_count > 0
            or residual_rows > 0
            or overlap_key_rows > 0
            or qk_overlap_proxy_ok_count > 0
        )
        else "missing"
    )
    top_overlap_keys = sorted(overlap_keys.items(), key=lambda kv: (-kv[1], kv[0]))[:20]
    return {
        "artifact": "SAVE_V68_OVERLAP_FEATURES",
        "status": status,
        "trace_files_scanned": len(trace_paths),
        "rows_scanned": scanned_rows,
        "semantic_merge_overlap_residual_rows": residual_rows,
        "semantic_merge_fit_success_rows": fit_success_rows,
        "rows_with_any_overlap_key": overlap_key_rows,
        "qk_overlap_proxy_files": len(qk_proxy_paths),
        "qk_overlap_proxy_ok_count": qk_overlap_proxy_ok_count,
        "qk_overlap_proxy_map_shape_sample": qk_overlap_map_shape_sample,
        "runtime_swa_overlap_feature_dump_files": len(runtime_dump_paths),
        "runtime_swa_overlap_feature_dump_ok_count": int(runtime_dump_ok_count),
        "runtime_swa_overlap_feature_dump_bad_count": int(runtime_dump_bad_count),
        "runtime_swa_overlap_feature_dump_chunks": runtime_unique_chunks,
        "runtime_swa_overlap_feature_dump_covers_required_chunks": runtime_covers_required,
        "runtime_swa_overlap_feature_dump_kinds": runtime_dump_kinds,
        "runtime_swa_overlap_feature_dump_shape_sample": runtime_dump_shape_sample,
        "top_overlap_keys": top_overlap_keys,
        "sample_evidence_paths": sample_paths,
        "interpretation": (
            "runtime SWA overlap feature maps cover all required chunks"
            if status == "pass"
            else
            "runtime SWA overlap feature maps exist but do not cover all required chunks"
            if runtime_dump_ok_count > 0
            else
            "qk dense overlap proxy maps and overlap residual/debug summaries exist, but no raw overlap feature map artifact was found"
            if qk_overlap_proxy_ok_count > 0 and status == "partial"
            else
            "overlap residual/debug summaries exist, but no dense overlap feature map artifact was found"
            if status == "partial"
            else "overlap feature evidence not found in scanned traces"
        ),
    }


def build_rows(summary: Dict[str, Any]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for item in summary["artifacts"]:
        artifact = str(item.get("artifact"))
        evidence_paths = item.get("sample_evidence_paths") or [item.get("feature_dir", "")]
        if isinstance(evidence_paths, str):
            evidence_paths = [evidence_paths]
        details = {k: v for k, v in item.items() if k not in {"chunk_reports"}}
        rows.append(
            {
                "artifact": artifact,
                "status": str(item.get("status")),
                "evidence_path": ";".join(str(p) for p in evidence_paths if p),
                "details_json": json.dumps(details, ensure_ascii=True, sort_keys=True, default=_json_default),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--chunks", default=DEFAULT_CHUNKS)
    parser.add_argument("--required-layers", default=DEFAULT_LAYERS)
    parser.add_argument(
        "--out-dir",
        default=str(Path(DEFAULT_ROOT) / "task4_internal_artifact_readiness"),
    )
    parser.add_argument("--out-json", default="")
    parser.add_argument("--out-csv", default="")
    args = parser.parse_args()

    root = Path(args.root)
    chunks = _parse_int_csv(args.chunks)
    required_layers = _parse_int_csv(args.required_layers)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = Path(args.out_json) if args.out_json else out_dir / "task4_readiness_summary.json"
    out_csv = Path(args.out_csv) if args.out_csv else out_dir / "task4_readiness_rows.csv"

    artifacts = [
        audit_qk_features(root, chunks, required_layers),
        audit_source_attention(root, chunks),
        audit_ttt_delta(root, chunks),
        audit_overlap_features(root, chunks),
    ]
    source_status = artifacts[1]["status"]
    ttt_status = artifacts[2]["status"]
    overlap_status = artifacts[3]["status"]
    qk_status = artifacts[0]["status"]
    source_proxy_count = int(artifacts[1].get("qk_pooled_source_attention_proxy_ok_count") or 0)
    sampled_source_count = int(artifacts[1].get("sampled_source_attention_map_ok_count") or 0)
    fullquery_source_count = int(artifacts[1].get("fullquery_source_attention_marginal_ok_count") or 0)
    overlap_proxy_count = int(artifacts[3].get("qk_overlap_proxy_ok_count") or 0)
    overlap_runtime_count = int(artifacts[3].get("runtime_swa_overlap_feature_dump_ok_count") or 0)
    ttt_dump_count = int(artifacts[2].get("ttt_post_delta_dump_ok_count") or 0)
    ttt_spatial_count = int(artifacts[2].get("ttt_spatial_post_delta_map_ok_count") or 0)
    ready = bool(
        qk_status == "pass"
        and source_status == "pass"
        and ttt_status == "pass"
        and overlap_status == "pass"
    )
    blocking_reasons: List[str] = []
    if source_status != "pass":
        blocking_reasons.append(
            "raw_source_attention_fullquery_marginal_incomplete"
            if fullquery_source_count > 0
            else "raw_source_attention_map_missing_sampled_available"
            if sampled_source_count > 0
            else "raw_source_attention_map_missing_proxy_available"
            if source_proxy_count > 0
            else "source_attention_map_missing"
        )
    if overlap_status != "pass":
        blocking_reasons.append(
            "raw_overlap_feature_map_incomplete_runtime_dump_available"
            if overlap_runtime_count > 0
            else "raw_overlap_feature_map_missing_proxy_available"
            if overlap_proxy_count > 0
            else "dense_overlap_feature_map_missing"
        )
    if qk_status != "pass":
        blocking_reasons.append("qk_feature_dump_incomplete")
    if ttt_status != "pass":
        blocking_reasons.append(
            "ttt_spatial_post_delta_map_incomplete"
            if ttt_spatial_count > 0
            else "ttt_spatial_post_delta_map_missing_fast_weight_dump_available"
            if ttt_dump_count > 0
            else "ttt_dense_post_delta_map_missing"
        )
    summary: Dict[str, Any] = {
        "schema": "acl2_v68_task4_internal_artifact_readiness_v1",
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "root": str(root),
        "chunks": [int(v) for v in chunks],
        "required_layers": [int(v) for v in required_layers],
        "artifacts": artifacts,
        "decision": {
            "ready_for_path_specific_cue_construction": ready,
            "overall_status": "ready" if ready else "partial_blocked",
            "blocking_reasons": blocking_reasons,
            "recommended_next_repair": (
                "Implement raw/per-layer model source-attention export and spatial/token-aligned TTT post-delta "
                "artifact dumps before another READ-only parameter sweep; sampled/QK source-attention maps and "
                "fast-weight TTT dumps are useful audit carriers but not full source-attention or spatial delta maps."
            ),
        },
    }

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=True, sort_keys=True, default=_json_default)
        f.write("\n")

    rows = build_rows(summary)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["artifact", "status", "evidence_path", "details_json"])
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(summary["decision"], indent=2, ensure_ascii=True, sort_keys=True))
    print(f"wrote {out_json}")
    print(f"wrote {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
