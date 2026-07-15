#!/usr/bin/env python3
"""Synthetic audit for v119TF LingBot logical special-entry gathered K/V."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LB_ROOT = REPO_ROOT / "third_party" / "lingbot-map"
OUT_ROOT = (
    REPO_ROOT
    / "results"
    / "acl2_v119tf_semantic_addressable_geometry_carrier_routing_carrier_aware_augmented"
    / "stage1_lblogical"
)


def _set_import_path() -> None:
    lb_root = str(LB_ROOT)
    if lb_root not in sys.path:
        sys.path.insert(0, lb_root)


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> int:
    _set_import_path()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    trace_path = OUT_ROOT / "lblogical_gathered_kv_trace.jsonl"
    summary_path = OUT_ROOT / "lblogical_gathered_kv_audit_summary.json"
    table_path = OUT_ROOT / "lblogical_entry_table.json"
    if trace_path.exists():
        trace_path.unlink()

    os.environ["ACL2_V118_LB_FI_PROVENANCE_FILE"] = str(trace_path)
    os.environ["ACL2_V118_LB_PROVENANCE_SEQ"] = "synthetic_lblogical"
    os.environ["ACL2_V118_LB_FI_PROVENANCE_MAX_ROWS"] = "10000"

    summary: dict[str, object] = {
        "schema": "acl2_v119tf_lblogical_gathered_kv_audit_summary_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "trace_path": str(trace_path),
        "entry_table_path": str(table_path),
        "audit_pass": False,
    }

    try:
        import torch
        from lingbot_map.layers.flashinfer_cache import FlashInferKVCacheManager

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(119)
        tokens_per_frame = 22
        num_special = 6
        patches_per_frame = tokens_per_frame - num_special
        num_heads = 2
        head_dim = 64
        mgr = FlashInferKVCacheManager(
            num_blocks=1,
            max_num_frames=8,
            tokens_per_frame=tokens_per_frame,
            num_heads=num_heads,
            head_dim=head_dim,
            dtype=torch.float16,
            device=device,
            num_special_tokens=num_special,
            scale_frames=1,
            sliding_window=4,
            max_total_frames=12,
            force_fp32=False,
        )

        for frame in range(5):
            k = torch.zeros(tokens_per_frame, num_heads, head_dim, dtype=torch.float32, device=device)
            v = torch.zeros_like(k)
            for tok in range(num_special):
                k[tok].fill_(float(frame + 1))
                v[tok].fill_(float(frame * 10 + tok))
            k[num_special:].normal_(mean=float(frame), std=0.01)
            v[num_special:].normal_(mean=float(frame), std=0.01)
            mgr.append_frame(0, k, v)

        table = mgr.build_logical_special_entry_table(0)
        selected_k, selected_v, selected_rows = mgr.gather_logical_special_kv(0, source_frames=[2])
        multi_k, multi_v, multi_rows = mgr.gather_logical_special_kv(0, source_frames=[1, 3])
        q = torch.randn(tokens_per_frame, num_heads, head_dim, dtype=torch.float32, device=device)
        form_a_out, form_a_rows = mgr.compute_logical_special_subrange_mask_attention(0, q, source_frames=[2])
        gather_out, read_rows = mgr.compute_logical_special_attention(0, q, source_frames=[2])
        one_page_out, one_page_rows = mgr.compute_one_frame_per_page_special_attention(0, q, source_frames=[2])
        os.environ["ACL2_V118_LB_STAGE4_POLICY"] = "TR_LOGICAL_QK_TOPK"
        os.environ["ACL2_V119_LB_LOGICAL_TOPK_ENTRIES"] = "2"
        runtime_out = mgr.compute_attention(0, q)
        os.environ.pop("ACL2_V118_LB_STAGE4_POLICY", None)
        os.environ.pop("ACL2_V119_LB_LOGICAL_TOPK_ENTRIES", None)
        trace_rows = _read_jsonl(trace_path)

        with table_path.open("w", encoding="utf-8") as handle:
            json.dump(table, handle, indent=2, sort_keys=True)
            handle.write("\n")

        selected_values = selected_v.detach().float().cpu().reshape(selected_v.shape[0], -1).mean(dim=1).tolist()
        selected_token_count = int(selected_k.shape[0])
        multi_token_count = int(multi_k.shape[0])
        selected_physical_page_ids = selected_rows[0]["physical_page_id"] if selected_rows else ""
        form_a_trace_rows = [row for row in trace_rows if row.get("operation_type") == "read_subrange_mask_logical_special_entry"]
        read_trace_rows = [row for row in trace_rows if row.get("operation_type") == "read_logical_special_entry"]
        one_page_trace_rows = [row for row in trace_rows if row.get("operation_type") == "read_one_frame_per_page_logical_special_entry"]
        runtime_action_rows = [row for row in trace_rows if row.get("operation_type") == "stage4_logical_retrieval_policy_select"]
        runtime_read_rows = [row for row in trace_rows if row.get("operation_type") == "runtime_read_subrange_mask_logical_special_entry"]
        append_logical_rows = [row for row in trace_rows if row.get("operation_type") == "append_special_tokens"]
        max_abs_diff = float((gather_out.detach().float() - one_page_out.detach().float()).abs().max().cpu().item())
        form_a_form_b_max_abs_diff = float((form_a_out.detach().float() - gather_out.detach().float()).abs().max().cpu().item())
        audit_pass = (
            len(table) == 5
            and len(selected_rows) == 1
            and int(selected_rows[0]["source_frame"]) == 2
            and selected_token_count == num_special
            and multi_token_count == 2 * num_special
            and "|" in selected_physical_page_ids
            and tuple(form_a_out.shape) == (tokens_per_frame, num_heads, head_dim)
            and tuple(gather_out.shape) == (tokens_per_frame, num_heads, head_dim)
            and tuple(one_page_out.shape) == (tokens_per_frame, num_heads, head_dim)
            and tuple(runtime_out.shape) == (tokens_per_frame, num_heads, head_dim)
            and max_abs_diff <= 1e-6
            and form_a_form_b_max_abs_diff <= 5e-2
            and len(form_a_rows) == 1
            and len(read_rows) == 1
            and len(one_page_rows) == 1
            and form_a_rows[0].get("logical_backend") == "physical_page_logical_subrange_mask_flashinfer_custom_mask"
            and read_rows[0].get("logical_backend") == "gathered_logical_kv_exact_sdpa"
            and one_page_rows[0].get("logical_backend") == "one_frame_per_page_exact_sdpa_reference"
            and int(form_a_rows[0].get("entry_read_count", 0)) == 1
            and int(read_rows[0].get("entry_read_count", 0)) == 2
            and int(one_page_rows[0].get("entry_read_count", 0)) == 3
            and len(form_a_trace_rows) == 1
            and len(read_trace_rows) == 1
            and len(one_page_trace_rows) == 1
            and len(runtime_action_rows) >= 1
            and len(runtime_read_rows) >= 1
            and len(append_logical_rows) >= 5
        )
        summary.update(
            {
                "device": str(device),
                "torch_cuda_available": bool(torch.cuda.is_available()),
                "logical_entry_count": len(table),
                "selected_source_frames": [row.get("source_frame") for row in selected_rows],
                "selected_token_count": selected_token_count,
                "selected_value_token_means": selected_values,
                "selected_physical_page_ids": selected_physical_page_ids,
                "multi_selected_token_count": multi_token_count,
                "form_a_output_shape": str(tuple(form_a_out.shape)),
                "gather_output_shape": str(tuple(gather_out.shape)),
                "one_frame_page_output_shape": str(tuple(one_page_out.shape)),
                "runtime_logical_policy_output_shape": str(tuple(runtime_out.shape)),
                "form_b_form_c_max_abs_diff": max_abs_diff,
                "form_a_form_b_max_abs_diff": form_a_form_b_max_abs_diff,
                "form_a_read_row_count": len(form_a_rows),
                "read_row_count": len(read_rows),
                "one_frame_page_read_row_count": len(one_page_rows),
                "trace_row_count": len(trace_rows),
                "form_a_trace_row_count": len(form_a_trace_rows),
                "read_trace_row_count": len(read_trace_rows),
                "one_frame_page_trace_row_count": len(one_page_trace_rows),
                "runtime_logical_action_trace_row_count": len(runtime_action_rows),
                "runtime_logical_read_trace_row_count": len(runtime_read_rows),
                "runtime_logical_selected_entry_counts": [
                    row.get("selected_logical_entry_count") for row in runtime_action_rows
                ],
                "runtime_logical_custom_mask_true_counts": [
                    row.get("custom_mask_true_count") for row in runtime_action_rows
                ],
                "append_special_trace_row_count": len(append_logical_rows),
                "form_a_backend": form_a_rows[0].get("logical_backend") if form_a_rows else "",
                "logical_backend": read_rows[0].get("logical_backend") if read_rows else "",
                "one_frame_page_backend": one_page_rows[0].get("logical_backend") if one_page_rows else "",
                "form_a_entry_read_count": form_a_rows[0].get("entry_read_count") if form_a_rows else "",
                "entry_read_count": read_rows[0].get("entry_read_count") if read_rows else "",
                "one_frame_page_entry_read_count": one_page_rows[0].get("entry_read_count") if one_page_rows else "",
                "entry_qk_score": read_rows[0].get("entry_qk_score") if read_rows else "",
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
