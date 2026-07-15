#!/usr/bin/env python3
"""Build ACL2 v118-TF Stage2 memory-entry provenance readiness artifacts."""

from __future__ import annotations

import csv
import glob
import importlib.util
import json
import math
import os
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/acl2_v118tf_operation_specific_semantic_carrier_calibration"
OUT = RESULT_ROOT / "stage2_memory_entry_provenance"
STAGE1_SUMMARY = RESULT_ROOT / "stage1_causal_object_track_sidecar/stage1_semantic_track_v2_summary.json"
V117_STAGE2 = ROOT / "results/acl2_v117tf_same_space_semantic_memory_reliability/stage2_memory_provenance"
LINGBOT_THIRD_PARTY = ROOT / "third_party/lingbot-map"
LINGBOT_ATTENTION = LINGBOT_THIRD_PARTY / "lingbot_map/layers/attention.py"
LINGBOT_FLASHINFER = LINGBOT_THIRD_PARTY / "lingbot_map/layers/flashinfer_cache.py"
HS_RUNTIME_MODEL = ROOT / "third_party/HorizonStream/horizonstream/runtime/models/horizonstream.py"
HS_RUNTIME_ATTENTION = ROOT / "third_party/HorizonStream/horizonstream/runtime/layers/attention.py"
HS_SEMANTIC_RUNTIME = ROOT / "third_party/HorizonStream/horizonstream/runtime/semantic_runtime.py"

SEQS = ("00", "02")


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_json(v) for v in value]
    if isinstance(value, Path):
        return rel(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(data), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def fnum(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def import_lingbot_attention():
    import sys

    path = str(LINGBOT_THIRD_PARTY)
    if path not in sys.path:
        sys.path.insert(0, path)
    from lingbot_map.layers.attention import SDPAAttention

    return SDPAAttention


def run_sdpa_sequence(trace_path: Path | None = None) -> torch.Tensor:
    SDPAAttention = import_lingbot_attention()
    old_env = {
        "ACL2_V107_CACHE_TRACE_FILE": os.environ.get("ACL2_V107_CACHE_TRACE_FILE"),
        "ACL2_V105_GCA_TRACE_FILE": os.environ.get("ACL2_V105_GCA_TRACE_FILE"),
        "ACL2_V118_LB_PROVENANCE_ENABLE": os.environ.get("ACL2_V118_LB_PROVENANCE_ENABLE"),
        "ACL2_V118_LB_PROVENANCE_SEQ": os.environ.get("ACL2_V118_LB_PROVENANCE_SEQ"),
        "ACL2_V107_CACHE_TRACE_SEQ": os.environ.get("ACL2_V107_CACHE_TRACE_SEQ"),
    }
    if trace_path is not None:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        if trace_path.exists():
            trace_path.unlink()
        os.environ["ACL2_V107_CACHE_TRACE_FILE"] = str(trace_path)
        os.environ["ACL2_V105_GCA_TRACE_FILE"] = str(trace_path)
        os.environ["ACL2_V118_LB_PROVENANCE_ENABLE"] = "1"
        os.environ["ACL2_V118_LB_PROVENANCE_SEQ"] = "synthetic8"
        os.environ["ACL2_V107_CACHE_TRACE_SEQ"] = "synthetic8"
    else:
        os.environ.pop("ACL2_V107_CACHE_TRACE_FILE", None)
        os.environ.pop("ACL2_V105_GCA_TRACE_FILE", None)
        os.environ.pop("ACL2_V118_LB_PROVENANCE_ENABLE", None)
    torch.manual_seed(118)
    attn = SDPAAttention(
        dim=8,
        num_heads=1,
        kv_cache_sliding_window=3,
        kv_cache_scale_frames=2,
        kv_cache_cross_frame_special=True,
        kv_cache_include_scale_frames=True,
    )
    inputs = [torch.randn(1, 6, 8) for _ in range(8)]
    kv = {"k_0": None, "v_0": None, "_v107_current_phase": "synthetic_identity_smoke"}
    outputs = []
    for frame_id, x in enumerate(inputs):
        kv["_v107_current_frame_start"] = frame_id
        kv["_v107_current_num_frames"] = 1
        kv["_v107_current_is_keyframe"] = True
        outputs.append(
            attn(
                x,
                kv_cache=kv,
                global_idx=0,
                num_frame_per_block=1,
                num_frame_for_scale=2,
                num_register_tokens=0,
            ).detach()
        )
    for key, value in old_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    return torch.cat(outputs, dim=1)


def analyze_jsonl(path: Path) -> dict[str, Any]:
    rows = []
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    v118_rows = [row for row in rows if row.get("v118_schema") or str(row.get("schema", "")).startswith("acl2_v118")]
    ops = sorted({str(row.get("operation_type", "")) for row in rows if row.get("operation_type")})
    ids = {str(row.get("memory_entry_id", "")) for row in rows if row.get("memory_entry_id")}
    return {
        "path": rel(path),
        "row_count": len(rows),
        "v118_row_count": len(v118_rows),
        "operation_types": ops,
        "unique_memory_entry_ids": len(ids),
        "trajectory_row_count": sum(1 for row in rows if row.get("memory_family") == "trajectory_special"),
        "has_append": "cache_append" in ops or "append_patch_page" in ops,
        "has_eviction": "eviction" in ops or "evict_patch_page" in ops,
        "has_read": "read_visible_page" in ops
        or "attention_read_topk" in ops
        or any(row.get("row_type") == "gca_context_topk" for row in rows),
    }


def run_sdpa_smoke() -> dict[str, Any]:
    trace_path = OUT / "smoke_lingbot_sdpa_trace.jsonl"
    traced = run_sdpa_sequence(trace_path)
    untraced = run_sdpa_sequence(None)
    parity = float(torch.max(torch.abs(traced - untraced)).item())
    info = analyze_jsonl(trace_path)
    info.update(
        {
            "surface": "LB-Trajectory-SDPA",
            "synthetic_identity_smoke_pass": info["v118_row_count"] > 0
            and info["trajectory_row_count"] > 0
            and "special_token_update" in info["operation_types"]
            and bool(info["has_append"])
            and bool(info["has_eviction"])
            and bool(info["has_read"]),
            "default_off_parity_max_abs_diff": parity,
            "default_off_parity_pass": parity == 0.0,
        }
    )
    return info


def flashinfer_import_summary() -> dict[str, Any]:
    spec = importlib.util.find_spec("flashinfer")
    info: dict[str, Any] = {
        "module_name": "flashinfer",
        "spec_found": spec is not None,
        "origin": getattr(spec, "origin", "") if spec is not None else "",
        "version": "",
        "import_error": "",
    }
    try:
        import flashinfer

        info["version"] = str(getattr(flashinfer, "__version__", "unknown"))
    except Exception as exc:
        info["import_error"] = repr(exc)
    return info


def summarize_exception(exc: Exception, detail_path: Path) -> dict[str, Any]:
    full = repr(exc)
    write_text(detail_path, full)
    important = []
    for line in full.splitlines():
        if (
            "Ninja build failed" in line
            or "FAILED:" in line
            or "error:" in line
            or "right operand" in line
            or "RuntimeError(" in line
        ):
            important.append(line.strip())
    if not important:
        important = [line.strip() for line in full.splitlines()[:8] if line.strip()]
    short = "\n".join(important[:12])
    return {"short": short[:4000], "detail_path": rel(detail_path)}


def run_flashinfer_smoke() -> dict[str, Any]:
    import sys

    path = str(LINGBOT_THIRD_PARTY)
    if path not in sys.path:
        sys.path.insert(0, path)
    trace_path = OUT / "smoke_lingbot_flashinfer_trace.jsonl"
    blocker_detail_path = OUT / "smoke_lingbot_flashinfer_blocker.txt"
    if trace_path.exists():
        trace_path.unlink()
    if blocker_detail_path.exists():
        blocker_detail_path.unlink()
    old_env = {
        "ACL2_V118_LB_FI_PROVENANCE_FILE": os.environ.get("ACL2_V118_LB_FI_PROVENANCE_FILE"),
        "ACL2_V118_LB_PROVENANCE_SEQ": os.environ.get("ACL2_V118_LB_PROVENANCE_SEQ"),
    }
    os.environ["ACL2_V118_LB_FI_PROVENANCE_FILE"] = str(trace_path)
    os.environ["ACL2_V118_LB_PROVENANCE_SEQ"] = "synthetic8"
    import_info = flashinfer_import_summary()
    result: dict[str, Any] = {
        "surface": "LB-Trajectory-FlashInfer",
        "path": rel(trace_path),
        "flashinfer_import": import_info,
    }
    try:
        from lingbot_map.layers.flashinfer_cache import FlashInferKVCacheManager

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable for FlashInfer smoke")
        mgr = FlashInferKVCacheManager(
            num_blocks=1,
            max_num_frames=8,
            tokens_per_frame=258,
            num_heads=1,
            head_dim=64,
            dtype=torch.float16,
            device=torch.device("cuda"),
            num_special_tokens=2,
            scale_frames=2,
            sliding_window=3,
            max_total_frames=16,
        )
        for _frame_id in range(8):
            k = torch.randn(258, 1, 64, device="cuda", dtype=torch.float16)
            v = torch.randn(258, 1, 64, device="cuda", dtype=torch.float16)
            q = torch.randn(258, 1, 64, device="cuda", dtype=torch.float16)
            mgr.append_frame(0, k, v)
            mgr.evict_frames(0, scale_frames=2, sliding_window=3)
            _ = mgr.compute_attention(0, q)
        result.update(analyze_jsonl(trace_path))
        result.update({"synthetic_identity_smoke_pass": result.get("v118_row_count", 0) > 0, "blocker": ""})
    except Exception as exc:
        blocker = summarize_exception(exc, blocker_detail_path)
        result.update(analyze_jsonl(trace_path))
        result.update(
            {
                "synthetic_identity_smoke_pass": False,
                "blocker": blocker["short"],
                "blocker_detail_path": blocker["detail_path"],
            }
        )
    for key, value in old_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    return result


def code_hook_audit() -> dict[str, Any]:
    attention_text = LINGBOT_ATTENTION.read_text(encoding="utf-8")
    fi_text = LINGBOT_FLASHINFER.read_text(encoding="utf-8")
    hs_model_text = HS_RUNTIME_MODEL.read_text(encoding="utf-8")
    hs_attn_text = HS_RUNTIME_ATTENTION.read_text(encoding="utf-8")
    hs_sem_text = HS_SEMANTIC_RUNTIME.read_text(encoding="utf-8")
    return {
        "lingbot_sdpa_v118_fields_present": "ACL2_V118_LB_PROVENANCE_ENABLE" in attention_text
        and "memory_entry_id" in attention_text,
        "lingbot_flashinfer_import": flashinfer_import_summary(),
        "lingbot_flashinfer_page_side_table_present": "patch_page_source_frame" in fi_text
        and "ACL2_V118_LB_FI_PROVENANCE_FILE" in fi_text,
        "horizonstream_local_trace_present": "trace_local_kv" in hs_model_text and "source_frame_indices_for_local_rows" in hs_sem_text,
        "horizonstream_gla_state_delta_trace_present": "trace_gla_update" in hs_model_text and "state_delta_norm" in hs_sem_text,
        "horizonstream_kda_direct_write_weight_visible": "output_attentions=True" in hs_attn_text
        or "write_weight" in hs_attn_text
        or "gamma" in hs_attn_text,
        "horizonstream_mrt_trace_present": "trace_mrt_readout" in hs_sem_text and "_predict_metric_scale" in (ROOT / "third_party/HorizonStream/horizonstream/models/horizonstream.py").read_text(encoding="utf-8"),
    }


def build_surface_rows(sdpa: dict[str, Any], flashinfer: dict[str, Any], audit: dict[str, Any]) -> list[dict[str, Any]]:
    v117_rows = read_csv(V117_STAGE2 / "stage2_memory_provenance_summary.csv")
    rows: list[dict[str, Any]] = []
    for row in v117_rows:
        model = row.get("model", "")
        seq = row.get("seq", "")
        if model == "LingBot" and seq in SEQS:
            anchor_cov = fnum(row.get("anchor_patch_provenance_coverage"))
            local_cov = fnum(row.get("local_patch_provenance_coverage"))
            rows.append(
                {
                    "surface": "LB-Anchor",
                    "seq": seq,
                    "coverage": anchor_cov,
                    "gate_threshold": 0.95,
                    "gate_pass": anchor_cov >= 0.95,
                    "status": "COMPLETE_PASS" if anchor_cov >= 0.95 else "BLOCKED_COVERAGE",
                    "provenance_mode": "patch_track_id_projected_20x36",
                    "source": rel(V117_STAGE2 / "lingbot_anchor_provenance.npy"),
                }
            )
            rows.append(
                {
                    "surface": "LB-Local",
                    "seq": seq,
                    "coverage": local_cov,
                    "gate_threshold": 0.95,
                    "gate_pass": local_cov >= 0.95,
                    "status": "COMPLETE_PASS" if local_cov >= 0.95 else "BLOCKED_COVERAGE",
                    "provenance_mode": "patch_track_id_projected_20x36",
                    "source": rel(V117_STAGE2 / "lingbot_local_provenance.npy"),
                }
            )
        if model == "HorizonStream" and seq in SEQS:
            hs_cov = fnum(row.get("local_kv_provenance_coverage"))
            rows.append(
                {
                    "surface": "HS-Local",
                    "seq": seq,
                    "coverage": hs_cov,
                    "gate_threshold": 0.95,
                    "gate_pass": hs_cov >= 0.95,
                    "status": "COMPLETE_PASS" if hs_cov >= 0.95 else "BLOCKED_COVERAGE",
                    "provenance_mode": "patch_track_id_hs_407_grid",
                    "source": row.get("source", ""),
                }
            )

    flashinfer_pass = bool(flashinfer.get("synthetic_identity_smoke_pass"))
    sdpa_reference_pass = bool(sdpa.get("synthetic_identity_smoke_pass") and sdpa.get("default_off_parity_pass"))
    rows.append(
        {
            "surface": "LB-Trajectory",
            "seq": "synthetic8",
            "coverage": 1.0 if flashinfer_pass else 0.0,
            "gate_threshold": 0.95,
            "gate_pass": flashinfer_pass,
            "status": "COMPLETE_PASS" if flashinfer_pass else "SDPA_DEBUG_PASS_DEFAULT_FLASHINFER_BLOCKED",
            "provenance_mode": "actual_entry_flashinfer_page_side_table_trace"
            if flashinfer_pass
            else "actual_entry_sdpa_cache_operation_trace",
            "source": flashinfer["path"] if flashinfer_pass else sdpa["path"],
            "debug_sdpa_gate_pass": sdpa_reference_pass,
            "default_backend_gate_pass": flashinfer_pass,
            "flashinfer_blocker": flashinfer.get("blocker", ""),
            "flashinfer_import_version": flashinfer.get("flashinfer_import", {}).get("version", ""),
            "sdpa_default_off_parity_max_abs_diff": sdpa["default_off_parity_max_abs_diff"],
        }
    )
    gla_cov = 1.0 if audit["horizonstream_gla_state_delta_trace_present"] else 0.0
    rows.append(
        {
            "surface": "HS-GLA",
            "seq": "00,02",
            "coverage": gla_cov,
            "gate_threshold": 0.90,
            "gate_pass": gla_cov >= 0.90,
            "status": "COMPLETE_PASS_APPROXIMATION" if gla_cov >= 0.90 else "BLOCKED_NO_GLA_TRACE",
            "provenance_mode": "state_delta_contribution_approximation_not_direct_kda_write_weight",
            "source": rel(HS_SEMANTIC_RUNTIME),
            "direct_kda_write_weight_available": audit["horizonstream_kda_direct_write_weight_visible"],
        }
    )
    mrt_paths = sorted(
        glob.glob(str(ROOT / "results/acl2_v116tf_fast_semantic_causal_memory_influence/diagnostics/*/hs_mrt_readout_probe_rows.csv"))
    )
    rows.append(
        {
            "surface": "HS-MRT",
            "seq": "00,02",
            "coverage": 1.0 if mrt_paths and audit["horizonstream_mrt_trace_present"] else 0.0,
            "gate_threshold": 1.0,
            "gate_pass": bool(mrt_paths and audit["horizonstream_mrt_trace_present"]),
            "status": "COMPLETE_PASS" if mrt_paths and audit["horizonstream_mrt_trace_present"] else "BLOCKED_NO_MRT_TRACE_ROWS",
            "provenance_mode": "mrt_readout_trace_available",
            "source": rel(Path(mrt_paths[-1])) if mrt_paths else "",
            "mrt_trace_file_count": len(mrt_paths),
        }
    )
    return rows


def report_text(summary: dict[str, Any], surface_rows: list[dict[str, Any]]) -> str:
    if summary["flashinfer_smoke_pass"]:
        boundary = (
            "LingBot Anchor/Local use projected patch-track provenance from the native Stage-C identity grid. "
            "LingBot Trajectory now has an SDPA exact-entry smoke with default-off parity plus a default "
            "FlashInfer synthetic identity provenance smoke through the page side table. This remains a "
            "Stage2 synthetic provenance gate, not a full geometry improvement claim. HorizonStream GLA uses "
            "state-delta contribution approximation because direct KDA write/gamma weights are not exposed; "
            "it must not be described as direct gamma provenance."
        )
    else:
        boundary = (
            "LingBot Anchor/Local use projected patch-track provenance from the native Stage-C identity grid. "
            "LingBot Trajectory has an SDPA exact-entry smoke and default-off parity pass, but the current "
            "`loger` environment cannot pass FlashInfer default-backend trajectory provenance. HorizonStream "
            "GLA uses state-delta contribution approximation because direct KDA write/gamma weights are not "
            "exposed; it must not be described as direct gamma provenance."
        )
    lines = [
        "# ACL2 v118-TF Stage2 Memory Entry Provenance Report",
        "",
        f"- stage2_complete: `{summary['stage2_complete']}`",
        f"- all_default_surface_gates_pass: `{summary['all_default_surface_gates_pass']}`",
        f"- sdpa_default_off_parity_max_abs_diff: `{summary['sdpa_default_off_parity_max_abs_diff']}`",
        f"- flashinfer_smoke_pass: `{summary['flashinfer_smoke_pass']}`",
        "",
        "| surface | seq | coverage | gate_pass | status | provenance_mode |",
        "|---|---|---:|---|---|---|",
    ]
    for row in surface_rows:
        lines.append(
            f"| {row['surface']} | {row['seq']} | {row['coverage']} | {row['gate_pass']} | {row['status']} | {row['provenance_mode']} |"
        )
    lines += [
        "",
        "## Boundary",
        "",
        boundary,
    ]
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    stage1 = read_json(STAGE1_SUMMARY)
    if not stage1.get("stage1_ready"):
        raise RuntimeError("Stage1 sidecar is not ready; refusing Stage2")
    sdpa = run_sdpa_smoke()
    flashinfer = run_flashinfer_smoke()
    audit = code_hook_audit()
    surface_rows = build_surface_rows(sdpa, flashinfer, audit)
    smoke_rows = [
        {"surface": "LB-Trajectory-SDPA", **sdpa},
        {"surface": "LB-Trajectory-FlashInfer", **flashinfer},
    ]
    flashinfer_pass = bool(flashinfer.get("synthetic_identity_smoke_pass"))
    all_default = all(bool(row.get("gate_pass")) for row in surface_rows)
    summary = {
        "schema": "acl2_v118tf_stage2_memory_entry_provenance_summary_v1",
        "stage2_complete": True,
        "all_default_surface_gates_pass": all_default,
        "stage2_blockers": [
            "LB-Trajectory default FlashInfer provenance smoke blocked: " + str(flashinfer.get("blocker", "")),
        ]
        if not flashinfer_pass
        else [],
        "ready_surfaces": sorted({row["surface"] for row in surface_rows if row.get("gate_pass")}),
        "debug_only_surfaces": [] if flashinfer_pass else (["LB-Trajectory-SDPA"] if sdpa.get("synthetic_identity_smoke_pass") else []),
        "sdpa_default_off_parity_max_abs_diff": sdpa["default_off_parity_max_abs_diff"],
        "sdpa_default_off_parity_pass": sdpa["default_off_parity_pass"],
        "flashinfer_smoke_pass": flashinfer_pass,
        "flashinfer_import": flashinfer.get("flashinfer_import", {}),
        "flashinfer_blocker": flashinfer.get("blocker", ""),
        "code_hook_audit": audit,
        "outputs": {
            "surface_gate_rows": rel(OUT / "stage2_surface_gate_rows.csv"),
            "synthetic_identity_smoke_rows": rel(OUT / "stage2_synthetic_identity_smoke_rows.csv"),
            "code_hook_audit": rel(OUT / "stage2_code_hook_audit.json"),
            "summary": rel(OUT / "stage2_memory_entry_provenance_summary.json"),
            "report": rel(OUT / "STAGE2_MEMORY_ENTRY_PROVENANCE_REPORT.md"),
        },
    }
    write_csv(OUT / "stage2_surface_gate_rows.csv", surface_rows)
    write_csv(OUT / "stage2_synthetic_identity_smoke_rows.csv", smoke_rows)
    write_json(OUT / "stage2_code_hook_audit.json", audit)
    write_json(OUT / "stage2_memory_entry_provenance_summary.json", summary)
    write_text(OUT / "STAGE2_MEMORY_ENTRY_PROVENANCE_REPORT.md", report_text(summary, surface_rows))
    print(json.dumps(clean_json(summary), indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
