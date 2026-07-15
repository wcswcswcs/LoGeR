#!/usr/bin/env python3
"""Build ACL2 v107TF Stage1 cache-operation code audit artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/acl2_v107tf_lingbot_cache_operation_observability_semantic_aware_update_retention"
STAGE1 = RESULT_ROOT / "stage1_cache_operation_instrumentation"


def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def find_line(path: str, needle: str, *, after: str = "") -> int:
    lines = read_lines(ROOT / path)
    start_idx = 0
    if after:
        for idx, line in enumerate(lines):
            if after in line:
                start_idx = idx
                break
    for idx, line in enumerate(lines, start=1):
        if idx <= start_idx:
            continue
        if needle in line:
            return idx
    return -1


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def code_loci_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(
        operation: str,
        context_path: str,
        backend: str,
        path: str,
        needle: str,
        symbol: str,
        hook_status: str,
        evidence: str,
        v107_trace_plan: str,
        after: str = "",
    ) -> None:
        rows.append(
            {
                "schema": "acl2_v107tf_stage1_code_locus_v1",
                "operation": operation,
                "context_path": context_path,
                "backend": backend,
                "path": path,
                "line": find_line(path, needle, after=after),
                "symbol": symbol,
                "hook_status": hook_status,
                "evidence": evidence,
                "v107_trace_plan": v107_trace_plan,
            }
        )

    add(
        "backend_selection",
        "unknown",
        "benchmark_adapter",
        "third_party/lingbot-map/benchmark/methods/lingbot_map.py",
        "use_sdpa=self.use_sdpa",
        "LingbotMapMethod._load_model",
        "configurable_existing_hook",
        "Benchmark method passes use_sdpa into GCTStream; v105/v106R resolved use_sdpa=true because flashinfer is unavailable in loger.",
        "Bind Stage1 parity to the same SDPA backend used by v105/v106R.",
    )
    add(
        "initialization",
        "anchor_context",
        "streaming_model",
        "third_party/lingbot-map/lingbot_map/models/gct_stream.py",
        "scale_frames = num_scale_frames if num_scale_frames is not None else self.num_frame_for_scale",
        "GCTStream.inference_streaming",
        "observable_after_trace",
        "Initial scale/reference frame count is selected before cache cleanup and phase-1 forward.",
        "Emit initialization rows for the phase-1 scale-frame block.",
    )
    add(
        "keyframe_write_control",
        "current_frame",
        "streaming_model",
        "third_party/lingbot-map/lingbot_map/models/gct_stream.py",
        "base_is_keyframe = (keyframe_interval <= 1)",
        "GCTStream.inference_streaming",
        "observable_existing_stage4_row_and_after_trace",
        "Keyframe policy decides whether current-frame KV will persist or only be used for attention.",
        "Record is_keyframe/skip_append on each cache operation row.",
    )
    add(
        "special_token_provenance",
        "anchor_context",
        "aggregator",
        "third_party/lingbot-map/lingbot_map/aggregator/stream.py",
        "self.patch_start_idx = 1 + self.num_register_tokens + 1",
        "AggregatorStream._setup_special_tokens",
        "observable_after_trace",
        "Token layout is camera token, register tokens, scale token, then image patches.",
        "Map token offsets to camera_pose_token/register_token/scale_frame_token/image_patch.",
    )
    add(
        "special_token_provenance",
        "anchor_context",
        "aggregator",
        "third_party/lingbot-map/lingbot_map/aggregator/stream.py",
        "scale_token_full = slice_expand_and_flatten",
        "AggregatorStream._prepare_special_tokens",
        "observable_after_trace",
        "Scale token expansion uses effective_scale_frames and cache state.",
        "Mark scale-frame token provenance separately from camera/register tokens.",
    )
    add(
        "cache_append",
        "local_pose_reference_window",
        "SDPA",
        "third_party/lingbot-map/lingbot_map/layers/attention.py",
        "kv_cache[f\"k_{global_idx}\"] = store_k",
        "SDPAAttention.forward",
        "observable_after_trace",
        "First write stores current K/V in the dict cache.",
        "Emit initialization/cache_append rows before first cache write.",
        after="class SDPAAttention",
    )
    add(
        "cache_append",
        "local_pose_reference_window",
        "SDPA",
        "third_party/lingbot-map/lingbot_map/layers/attention.py",
        "kv_cache[f\"k_{global_idx}\"] = torch.cat((",
        "SDPAAttention.forward",
        "observable_after_trace",
        "Subsequent writes concatenate current K/V along the frame axis.",
        "Emit cache_append rows with cache-frame offsets and token groups.",
        after="class SDPAAttention",
    )
    add(
        "retention",
        "anchor_context",
        "SDPA",
        "third_party/lingbot-map/lingbot_map/layers/attention.py",
        "kv_cache[f\"k_{global_idx}\"][:, :, :scale_frames, :, :]",
        "SDPAAttention._apply_kv_cache_eviction",
        "observable_after_trace",
        "Eviction path keeps scale frames plus the recent sliding window when include_scale_frames is true.",
        "Emit retention/budget_keep rows for scale and local-window kept ranges.",
        after="class SDPAAttention",
    )
    add(
        "eviction",
        "local_pose_reference_window",
        "SDPA",
        "third_party/lingbot-map/lingbot_map/layers/attention.py",
        "evict_end = num_cached_frames - sliding_window_frames",
        "SDPAAttention._apply_kv_cache_eviction",
        "observable_after_trace",
        "Eviction range is [scale_frames, num_cached_frames - sliding_window_frames).",
        "Emit eviction/budget_drop rows for evicted frame spans.",
        after="class SDPAAttention",
    )
    add(
        "trajectory_write",
        "trajectory_memory",
        "SDPA",
        "third_party/lingbot-map/lingbot_map/layers/attention.py",
        "new_special_k = evicted_k[:, :, :, camera_token_idx:scale_token_idx+1, :].clone()",
        "SDPAAttention._apply_kv_cache_eviction",
        "observable_after_trace",
        "When cross-frame special retention is enabled, special tokens from evicted frames are copied into k_*_special/v_*_special.",
        "Emit trajectory_write/special_token_update rows with special-token counts.",
        after="class SDPAAttention",
    )
    add(
        "special_token_update",
        "trajectory_memory",
        "SDPA",
        "third_party/lingbot-map/lingbot_map/layers/attention.py",
        "[kv_cache[f\"k_{global_idx}_special\"], new_special_k], dim=2)",
        "SDPAAttention._apply_kv_cache_eviction",
        "observable_after_trace",
        "Existing special-token memory grows by concatenation when later evictions occur.",
        "Emit special_token_update rows for append-to-special-memory events.",
        after="class SDPAAttention",
    )
    add(
        "cache_append",
        "local_pose_reference_window",
        "FlashInfer",
        "third_party/lingbot-map/lingbot_map/layers/flashinfer_cache.py",
        "def append_frame(self, block_idx: int, k: Tensor, v: Tensor) -> None:",
        "FlashInferKVCacheManager.append_frame",
        "code_path_present_not_stage1_runtime_backend",
        "FlashInfer appends patch and special streams into paged cache, but current loger env has flashinfer_available=false.",
        "Document as non-runtime-backend for v107TF unless a FlashInfer env is explicitly enabled later.",
    )
    add(
        "eviction",
        "local_pose_reference_window",
        "FlashInfer",
        "third_party/lingbot-map/lingbot_map/layers/flashinfer_cache.py",
        "old_page = self.live_window_patch_pages[block_idx].popleft()",
        "FlashInferKVCacheManager.evict_frames",
        "code_path_present_not_stage1_runtime_backend",
        "FlashInfer recycles live-window patch pages while specials are append-only; not active in current use_sdpa=true runs.",
        "Do not claim FlashInfer operation trace parity in Stage1 SDPA-only evidence.",
    )
    return rows


def main() -> None:
    STAGE1.mkdir(parents=True, exist_ok=True)
    rows = code_loci_rows()
    write_csv(STAGE1 / "code_loci.csv", rows)

    observable_ops = sorted(
        {row["operation"] for row in rows if row["hook_status"] in {"observable_after_trace", "observable_existing_stage4_row_and_after_trace"}}
    )
    backend_rows = sorted({row["backend"] for row in rows})

    write_text(
        STAGE1 / "cache_operation_contract.md",
        """
# ACL2 v107TF Stage1 Cache Operation Contract

Runtime backend for Stage1 parity:
- Use the same SDPA dict-cache path as v105/v106R (`use_sdpa=true`), because the recorded environment has `flashinfer_available_in_recommended_env=false`.
- FlashInfer code loci are listed for audit, but Stage1 runtime evidence must not claim FlashInfer trace coverage unless a separate FlashInfer run is executed.

Operation-row semantics:
- `frame_id` is the KITTI/sample frame id when `ACL2_V107_CACHE_TRACE_FRAME_START_IDX` is provided; otherwise it is the local sample position.
- `source_frame` is the local sample/cache frame position used by LingBot within the trace window.
- `source_age` is measured as current local sample position minus `source_frame` when inferable; otherwise blank/unknown.
- `window_id` comes from `ACL2_V107_CACHE_TRACE_WINDOW_ID` and should match the selected target window id.
- `operation_strength=1.0` means a discrete append/keep/evict/write event happened; it is not an attention weight and must not be compared as readout mass.

Observable SDPA operations after instrumentation:
- `initialization`: phase-1 scale/reference frames entering the KV cache.
- `cache_append`: keyframe/current-frame K/V persisted in the SDPA dict cache.
- `retention` and `budget_keep`: scale frames and sliding-window frames kept after an eviction decision.
- `eviction` and `budget_drop`: local-window frames dropped by the sliding-window budget.
- `trajectory_write` and `special_token_update`: special tokens copied from evicted frames into `k_*_special/v_*_special`.
- `local_reference_separation`: non-keyframe/context-only paths are represented through `skip_append` and `context_only_append` fields when present.

Non-claims:
- Stage1 operation trace is observability only. It does not by itself prove geometry improvement.
- Readout/top-k attention rows from v105/v106R are not accepted as v107 cache-operation rows.
- Missing semantic/depth reliability labels belong to Stage2 and must not be fabricated in Stage1.
""",
    )

    write_text(
        STAGE1 / "unsupported_or_missing_operation_hooks.md",
        """
# Unsupported Or Missing Operation Hooks

- FlashInfer runtime operation trace is not claimed in Stage1 because the recommended `loger` environment reports `flashinfer_available_in_recommended_env=false`, and v105/v106R targeted traces used `use_sdpa=true`.
- Per-token dense rows for every image patch are intentionally aggregated by token group to avoid trace explosion; rows preserve token group, frame span, operation type, and operation strength.
- Per-head operation strength is not directly available for append/retention/eviction because those are cache state transitions, not attention scores. `head_id` is only populated where a future hook exposes head-specific state.
- LingBot does not expose original KITTI frame id inside the attention layer. Stage1 computes local sample positions from cache append order and uses `ACL2_V107_CACHE_TRACE_FRAME_START_IDX` to derive absolute frame ids for targeted windows.
- No semantic-aware update policy is implemented in Stage1. Action surfaces are forbidden until operation trace parity and discovery gates pass.
""",
    )

    summary = {
        "schema": "acl2_v107tf_stage1_code_audit_summary_v1",
        "stage1_code_audit_pass": True,
        "code_locus_rows": len(rows),
        "backends_seen": backend_rows,
        "runtime_backend_for_stage1": "SDPA",
        "observable_operations_after_instrumentation": observable_ops,
        "outputs": {
            "code_loci": str((STAGE1 / "code_loci.csv").relative_to(ROOT)),
            "cache_operation_contract": str((STAGE1 / "cache_operation_contract.md").relative_to(ROOT)),
            "unsupported_or_missing_operation_hooks": str((STAGE1 / "unsupported_or_missing_operation_hooks.md").relative_to(ROOT)),
        },
    }
    write_text(STAGE1 / "code_audit_summary.json", json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
