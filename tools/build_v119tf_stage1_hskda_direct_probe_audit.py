#!/usr/bin/env python3
"""Synthetic audit for v119TF HorizonStream direct KDA write instrumentation."""

from __future__ import annotations

import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
HS_ROOT = REPO_ROOT / "third_party" / "HorizonStream"
OUT_ROOT = (
    REPO_ROOT
    / "results"
    / "acl2_v119tf_semantic_addressable_geometry_carrier_routing_carrier_aware_augmented"
    / "stage1_hskda"
)


def _set_import_path() -> None:
    hs_root = str(HS_ROOT)
    if hs_root not in sys.path:
        sys.path.insert(0, hs_root)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _status_values(rows: list[dict[str, str]], key: str) -> list[str]:
    return sorted({str(row.get(key, "")) for row in rows if str(row.get(key, ""))})


def main() -> int:
    _set_import_path()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    trace_file = OUT_ROOT / "hs_gla_direct_kda_probe_rows.csv"
    state_file = OUT_ROOT / "hs_gla_state_probe_rows.csv"
    summary_path = OUT_ROOT / "hskda_direct_probe_audit_summary.json"
    for path in (trace_file, state_file):
        if path.exists():
            path.unlink()

    os.environ["HS_V113_TRACE_ENABLE"] = "1"
    os.environ["HS_V113_TRACE_GLA_ENABLE"] = "1"
    os.environ["HS_V113_TRACE_ROOT"] = str(OUT_ROOT)
    os.environ["HS_V119_KDA_DIRECT_PROBE_MAX_TOKENS"] = "32"

    summary: dict[str, object] = {
        "schema": "acl2_v119tf_hskda_direct_probe_audit_summary_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "trace_root": str(OUT_ROOT),
        "direct_probe_rows": str(trace_file),
        "state_probe_rows": str(state_file),
        "audit_pass": False,
    }

    try:
        import torch
        from horizonstream.runtime.layers.attention import GLAAttention
        from horizonstream.runtime.models.horizonstream import GLACache
        from horizonstream.runtime.semantic_runtime import clear_context, set_context

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(119)
        attn = GLAAttention(
            dim=32,
            num_heads=4,
            head_dim=8,
            layer_idx=0,
            expand_v=1.0,
            mode="chunk",
            use_short_conv=False,
            conv_size=4,
        ).to(device)
        attn.eval()
        cache = GLACache(1)

        for step in range(2):
            set_context(seq="synthetic_hskda", chunk_idx=step, chunk_start=step * 8, chunk_end=step * 8 + 8, window_size=8)
            x = torch.randn(1, 16, 32, device=device)
            with torch.no_grad():
                y = attn(x, kv_cache=cache)
            if tuple(y.shape) != (1, 16, 32):
                raise RuntimeError(f"unexpected GLAAttention output shape: {tuple(y.shape)}")
            cache.advance(is_last_chunk=False)
        clear_context()

        rows = _read_csv_rows(trace_file)
        state_rows = _read_csv_rows(state_file)
        errors = [row.get("trace_error", "") for row in rows if row.get("trace_error", "")]
        qkv_statuses = _status_values(rows, "direct_qkv_status")
        decay_statuses = _status_values(rows, "direct_decay_status")
        gamma_statuses = _status_values(rows, "direct_gamma_status")
        q_shapes = [row.get("q_proj_shape", "") for row in rows]
        k_shapes = [row.get("k_proj_shape", "") for row in rows]
        v_shapes = [row.get("v_proj_shape", "") for row in rows]
        update_norms = [row.get("update_contribution_norm", "") for row in rows]
        audit_pass = (
            len(rows) >= 2
            and not errors
            and "DIRECT_QKV_EXACT_SILU_PROJECTION_INPUT" in qkv_statuses
            and "DIRECT_DECAY_READ_FROM_F_PROJ_A_LOG_DT_BIAS" in decay_statuses
            and gamma_statuses == ["DIRECT_GAMMA_UNAVAILABLE"]
            and all(q_shapes)
            and all(k_shapes)
            and all(v_shapes)
            and any(value not in {"", "None"} for value in update_norms)
        )
        summary.update(
            {
                "device": str(device),
                "torch_cuda_available": bool(torch.cuda.is_available()),
                "direct_probe_row_count": len(rows),
                "state_probe_row_count": len(state_rows),
                "direct_qkv_statuses": qkv_statuses,
                "direct_decay_statuses": decay_statuses,
                "direct_gamma_statuses": gamma_statuses,
                "trace_errors": errors,
                "q_proj_shapes": q_shapes,
                "k_proj_shapes": k_shapes,
                "v_proj_shapes": v_shapes,
                "update_contribution_norms": update_norms,
                "audit_pass": bool(audit_pass),
            }
        )
    except Exception as exc:
        summary.update({"audit_error": repr(exc), "audit_pass": False})

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary.get("audit_pass") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
