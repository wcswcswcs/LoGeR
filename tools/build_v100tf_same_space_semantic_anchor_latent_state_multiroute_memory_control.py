#!/usr/bin/env python3
"""Build ACL2 v100 same-space semantic-anchor latent-state diagnostics.

This builder is evidence-first: it never promotes projected/scalar proxies to
same-space latent state.  Runtime action remains blocked unless Track S passes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch


ROOT = Path("results/acl2_v100tf_same_space_semantic_anchor_latent_state_memory_control")
V99_ROOT = Path("results/acl2_v99tf_semantic_anchor_identity_lifecycle_multiroute_memory_control")
V98_ROOT = Path("results/acl2_v98tf_semantic_guided_swa_cache_topk_gauge_safe_memory_control")
V97_ROOT = Path("results/acl2_v97tf_semantic_scale_evidence_gauge_safe_memory_control")
V96_ROOT = Path("results/acl2_v96tf_vggt4d_informed_semantic_gauge_preserving_memory_control")
CASE_AUDIT_META_FIELDS = [
    "universe_split",
    "good_control_hygiene_core_good_l3_max",
    "good_control_hygiene_l3_threshold",
    "good_control_hygiene_l3_pass",
    "good_control_hygiene_warning",
    "good_control_hygiene_include_for_repair",
    "good_control_hygiene_status",
]
CASE_BASE_FIELDS = [
    "case_id",
    "seq",
    "case_label",
    "failure_type",
    "L3_handoff_transfer_penalty_proxy",
    *CASE_AUDIT_META_FIELDS,
]


def f(value: Any, default: float = math.nan) -> float:
    if value is None or value == "":
        return default
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def b(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def case_base(row: dict[str, Any]) -> dict[str, Any]:
    return {field: row.get(field, "") for field in CASE_BASE_FIELDS}


def finite(values: Any) -> list[float]:
    return [float(v) for v in values if math.isfinite(f(v))]


def mean(values: Any) -> float:
    vals = finite(values)
    return sum(vals) / len(vals) if vals else math.nan


def safe_product(*values: Any) -> float:
    out = 1.0
    for value in values:
        fv = f(value)
        if not math.isfinite(fv):
            return math.nan
        out *= fv
    return out


def pearson(xs: Any, ys: Any) -> float:
    pairs: list[tuple[float, float]] = []
    for x, y in zip(xs, ys):
        fx = f(x)
        fy = f(y)
        if math.isfinite(fx) and math.isfinite(fy):
            pairs.append((fx, fy))
    if len(pairs) < 2:
        return math.nan
    mx = sum(x for x, _ in pairs) / len(pairs)
    my = sum(y for _, y in pairs) / len(pairs)
    vx = sum((x - mx) ** 2 for x, _ in pairs)
    vy = sum((y - my) ** 2 for _, y in pairs)
    if vx <= 1.0e-12 or vy <= 1.0e-12:
        return math.nan
    cov = sum((x - mx) * (y - my) for x, y in pairs)
    return cov / math.sqrt(vx * vy)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    if not keys:
        keys = ["empty"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            clean = {
                key: json.dumps(value, ensure_ascii=False, sort_keys=True)
                if isinstance(value, (dict, list, tuple))
                else value
                for key, value in row.items()
            }
            writer.writerow(clean)


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except Exception:
        return []


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_case_id(case_id: str) -> tuple[str, int | None, int | None]:
    parts = str(case_id).split("_")
    if len(parts) != 3:
        return str(case_id)[:2], None, None
    try:
        return parts[0], int(parts[1]), int(parts[2])
    except Exception:
        return parts[0], None, None


def case_id_from_trace(path: Path) -> str:
    parts = list(path.parts)
    if "TTT_SWA_SAME_RUN" in parts:
        idx = parts.index("TTT_SWA_SAME_RUN")
        if idx > 0:
            return parts[idx - 1]
    for part in reversed(parts):
        bits = part.split("_")
        if len(bits) == 3 and all(bit.isdigit() for bit in bits):
            return part
    return ""


def torch_load(path: Path) -> Any:
    return torch.load(path, map_location="cpu")


def vec(value: Any) -> list[float] | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        flat = value.detach().cpu().float().reshape(-1).tolist()
    elif isinstance(value, (list, tuple)):
        flat = list(value)
    else:
        return None
    out: list[float] = []
    for item in flat:
        fv = f(item)
        if not math.isfinite(fv):
            return None
        out.append(fv)
    return out if out else None


def norm(value: list[float] | None) -> float:
    if not value:
        return math.nan
    return math.sqrt(sum(float(v) * float(v) for v in value))


def cosine_residual(lhs: list[float] | None, rhs: list[float] | None) -> float:
    if lhs is None or rhs is None or len(lhs) != len(rhs) or not lhs:
        return math.nan
    ln = norm(lhs)
    rn = norm(rhs)
    if not math.isfinite(ln) or not math.isfinite(rn) or ln <= 1.0e-12 or rn <= 1.0e-12:
        return math.nan
    dot = sum(float(a) * float(b) for a, b in zip(lhs, rhs))
    cos = max(-1.0, min(1.0, dot / max(ln * rn, 1.0e-12)))
    return 1.0 - cos


def stage0() -> dict[str, Any]:
    out = ROOT / "stage0_evidence_ledger"
    expected = [
        ("v99_final_decision", V99_ROOT / "final_decision/final_decision.json", True),
        ("v99_trackN_summary", V99_ROOT / "trackN_semantic_anchor_identity_memory_graph/summary.json", True),
        ("v99_trackN_case_rows", V99_ROOT / "trackN_semantic_anchor_identity_memory_graph/graph_case_rows.csv", True),
        ("v99_trackN_anchor_rows", V99_ROOT / "trackN_semantic_anchor_identity_memory_graph/graph_anchor_lifecycle_rows.csv", True),
        ("v99_trackC3_summary", V99_ROOT / "trackC3_identity_conditioned_latent_gauge_ruler/summary.json", True),
        ("v99_trackC3_proxy_rows", V99_ROOT / "trackC3_identity_conditioned_latent_gauge_ruler/proxy_pattern_metrics.csv", True),
        ("v99_trackO_summary", V99_ROOT / "trackO_anchor_freshness_current_support_ruler/summary.json", True),
        ("v99_trackM2_summary", V99_ROOT / "trackM2_identity_level_carrier_to_action_simulator/summary.json", True),
        ("v98_final_decision", V98_ROOT / "final_decision/final_decision.json", True),
        ("v98_stage7e_summary", V98_ROOT / "stage7e_ttt_stable_anchor_id_hook/summary.json", True),
        ("v98_stage7e_rows", V98_ROOT / "stage7e_ttt_stable_anchor_id_hook/anchor_id_case_rows.csv", True),
        ("v98_stage7f_summary", V98_ROOT / "stage7f_prev_ttt_anchor_gate_action_pilot/summary.json", False),
        ("v98_stage7g_summary", V98_ROOT / "stage7g_anchor_id_query_head_risk_attribution/summary.json", False),
        ("v98_stage7h_summary", V98_ROOT / "stage7h_prev_ttt_anchor_query_soft_action_pilot/summary.json", False),
        ("v97_H2_summary", V97_ROOT / "trackH2_l07_component_decomposition/summary.json", False),
        ("v97_E2_summary", V97_ROOT / "trackE2_swa_carrier_search_beyond_route_mass/summary.json", False),
        ("v97_C2_summary", V97_ROOT / "trackC_semantic_latent_gauge_ruler/summary.json", False),
        ("v97_F2_summary", V97_ROOT / "trackF2_ttt_stable_anchor_retention_missing_good_write/summary.json", False),
        ("v96_final_or_build_summary", V96_ROOT / "final_decision/final_decision.json", False),
    ]
    evidence_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    for name, path, critical in expected:
        exists = path.is_file()
        row = {
            "artifact": name,
            "path": str(path),
            "exists": exists,
            "critical": critical,
            "sha256": sha256_file(path) if exists and path.is_file() else "",
        }
        evidence_rows.append(row)
        if not exists:
            missing_rows.append(row)

    v99_final = read_json(V99_ROOT / "final_decision/final_decision.json")
    v99_track_summaries = {
        "trackN": read_json(V99_ROOT / "trackN_semantic_anchor_identity_memory_graph/summary.json"),
        "trackO": read_json(V99_ROOT / "trackO_anchor_freshness_current_support_ruler/summary.json"),
        "trackC3": read_json(V99_ROOT / "trackC3_identity_conditioned_latent_gauge_ruler/summary.json"),
        "trackM2": read_json(V99_ROOT / "trackM2_identity_level_carrier_to_action_simulator/summary.json"),
    }
    claim_rows = [
        {
            "track": "v99_final",
            "claim_level": "IDENTITY_LIFECYCLE_CONTROL_NO_GO",
            "gate_pass": False,
            "evidence": str(V99_ROOT / "final_decision/final_decision.json"),
            "summary": v99_final.get("final_taxonomy", ""),
        }
    ]
    for name, summary in v99_track_summaries.items():
        claim_rows.append(
            {
                "track": f"v99_{name}",
                "claim_level": "DIAGNOSTIC_OR_BLOCKED_NO_ACTION",
                "gate_pass": b(summary.get("gate_pass")),
                "evidence": str(V99_ROOT),
                "summary": summary.get("blocker", summary.get("status", "")),
            }
        )
    critical_missing = [row for row in missing_rows if row["critical"]]
    summary = {
        "schema": "acl2_v100_stage0_evidence_ledger_v1",
        "gate_pass": not critical_missing,
        "critical_missing_count": len(critical_missing),
        "missing_count": len(missing_rows),
        "v99_final_taxonomy": v99_final.get("final_taxonomy", ""),
        "v99_runtime_action_allowed": v99_final.get("runtime_action_allowed", False),
        "blocker": "" if not critical_missing else "Critical prerequisite artifacts are missing; see missing_artifacts_report.md.",
    }
    write_json(out / "summary.json", summary)
    write_rows(out / "evidence_ledger.csv", evidence_rows)
    write_rows(out / "claim_level_by_track.csv", claim_rows)
    write_rows(out / "missing_artifacts_report.csv", missing_rows)
    write_text(
        out / "missing_artifacts_report.md",
        "# Missing Artifacts Report\n\n"
        + ("\n".join(f"- {row['artifact']}: {row['path']} critical={row['critical']}" for row in missing_rows) or "- none\n"),
    )
    write_text(
        out / "do_not_repeat.md",
        "# Do Not Repeat\n\n"
        "- Do not repeat v96 weak-context READ skip as a full repair.\n"
        "- Do not repeat v97 centimeter-level full ATE claim with final-error harm.\n"
        "- Do not repeat v98 Stage7f aggregate prev-anchor gate or Stage7h query-soft sweep without same-space state.\n"
        "- Do not promote v99 scalar/sketch/projected proxy to same-space latent state.\n",
    )
    write_rows(out / "rows.csv", evidence_rows)
    write_rows(out / "gate_checks.csv", [{"gate": "critical_artifacts_present", "pass": not critical_missing}])
    write_text(out / "failure_report.md", summary["blocker"] or "Stage0 prerequisites loaded.\n")
    write_text(out / "what_would_have_to_be_true_to_pass.md", "All critical v99/v98 prerequisite rows must exist.\n")
    write_text(out / "visual_manifest.csv", "path,description\n")
    return summary


def trace_payload_paths() -> tuple[Path, list[Path], str]:
    candidates = [
        (ROOT / "trackS_same_space_latent_state/probe28_geometry_sidecar", "v100_probe28_geometry_sidecar"),
        (ROOT / "trackS_same_space_latent_state/probe28_sb_hidden_repair", "v100_probe28_sb_hidden_repair"),
        (ROOT / "trackS_same_space_latent_state/smoke2_sb_hidden_repair4", "v100_smoke2_sb_hidden_repair4"),
        (ROOT / "trackS_same_space_latent_state/smoke2_sb_hidden_repair3", "v100_smoke2_sb_hidden_repair3"),
        (ROOT / "trackS_same_space_latent_state/smoke2_sb_hidden_repair2", "v100_smoke2_sb_hidden_repair2"),
        (ROOT / "trackS_same_space_latent_state/smoke2_sb_hidden_repair", "v100_smoke2_sb_hidden_repair"),
        (ROOT / "trackS_same_space_latent_state/probe28_refk_repair", "v100_probe28_refk_repair"),
        (ROOT / "trackS_same_space_latent_state/probe28_same_run_probe", "v100_probe28"),
        (ROOT / "trackS_same_space_latent_state/smoke2_refk_repair", "v100_smoke2_refk_repair"),
        (ROOT / "trackS_same_space_latent_state/smoke2_same_run_probe", "v100_smoke2"),
        (V99_ROOT / "trackC3_full_z_vector_probe28_repair", "v99_reference_probe28"),
        (V99_ROOT / "trackC3_full_z_vector_projected_smoke2_repair", "v99_reference_smoke2"),
    ]
    for root, source in candidates:
        paths = sorted(root.glob("*/TTT_SWA_SAME_RUN/swa_raw_transport_trace/*.pt"))
        if paths:
            return root, paths, source
    return candidates[0][0], [], "missing"


def no_action_parity(trace_root: Path) -> dict[str, Any]:
    summaries = sorted(trace_root.glob("*/TTT_SWA_SAME_RUN/hmc_correctness_summary.json"))
    rows: list[dict[str, Any]] = []
    for path in summaries:
        payload = read_json(path)
        case_id = path.parents[1].name if len(path.parents) >= 2 else ""
        parity = (
            b(payload.get("probe_no_commit_hash_equal_all"))
            and b(payload.get("state_double_write_safe_all"))
            and f(payload.get("max_pass1_pass2_pose_matrix_abs_max"), math.inf) == 0.0
            and f(payload.get("max_pass1_pass2_pose_t_max"), math.inf) == 0.0
        )
        rows.append({
            "case_id": case_id,
            "summary_path": str(path),
            "probe_no_commit_hash_equal_all": b(payload.get("probe_no_commit_hash_equal_all")),
            "state_double_write_safe_all": b(payload.get("state_double_write_safe_all")),
            "max_pass1_pass2_pose_matrix_abs_max": payload.get("max_pass1_pass2_pose_matrix_abs_max"),
            "max_pass1_pass2_pose_t_max": payload.get("max_pass1_pass2_pose_t_max"),
            "no_action_parity_pass": parity,
        })
    return {
        "available": bool(rows),
        "all_pass": bool(rows) and all(b(row.get("no_action_parity_pass")) for row in rows),
        "case_count": len({row.get("case_id") for row in rows if row.get("case_id")}),
        "rows": rows,
    }


def same_space_row(
    *,
    case_id: str,
    seq: str,
    payload_path: Path,
    payload: dict[str, Any],
    life: dict[str, Any],
    canonical_space_name: str,
    z_ref: list[float] | None,
    z_write: list[float] | None,
    z_cache: list[float] | None,
    z_current: list[float] | None,
    projection_path: str,
    projection_is_native: bool,
    same_projection_family: bool,
    anchor_alignment_source: str,
) -> dict[str, Any]:
    dims = {
        "z_ref_dim": len(z_ref) if z_ref is not None else 0,
        "z_write_dim": len(z_write) if z_write is not None else 0,
        "z_cache_dim": len(z_cache) if z_cache is not None else 0,
        "z_current_dim": len(z_current) if z_current is not None else 0,
    }
    positive_dims = [dim for dim in dims.values() if dim > 0]
    same_dim = bool(positive_dims) and len(set(positive_dims)) == 1
    r_wc = cosine_residual(z_write, z_cache)
    r_cc = cosine_residual(z_cache, z_current)
    r_rc = cosine_residual(z_ref, z_current)
    rsame_parts = [value for value in [r_wc, r_cc, r_rc] if math.isfinite(value)]
    all_available = all(dim > 0 for dim in dims.values())
    assertion_pass = all_available and same_dim and same_projection_family and projection_is_native
    source_chunk = life.get("source_chunk_idx", payload.get("ttt_prev_stable_anchor_source_chunk_idx", ""))
    current_chunk = life.get("current_chunk_idx", payload.get("chunk_idx", ""))
    _, prev_chunk, curr_chunk = parse_case_id(case_id)
    return {
        "case_id": case_id,
        "seq": seq,
        "source_chunk": source_chunk if source_chunk != "" else prev_chunk,
        "current_chunk": current_chunk if current_chunk != "" else curr_chunk,
        "anchor_id": life.get("anchor_id", ""),
        "semantic_class": life.get("source_label_mode", ""),
        "anchor_role": "unknown",
        "canonical_space_name": canonical_space_name,
        "layer": payload.get("layer", ""),
        "head": "route_weighted_all_heads",
        **dims,
        "z_ref_norm": norm(z_ref),
        "z_write_norm": norm(z_write),
        "z_cache_norm": norm(z_cache),
        "z_current_norm": norm(z_current),
        "R_write_cache": r_wc,
        "R_cache_current": r_cc,
        "R_ref_current": r_rc,
        "R_same": mean(rsame_parts),
        "query_hit_frac": life.get("query_head_hit_frac", ""),
        "query_hit_max": life.get("query_head_hit_max", ""),
        "query_head_ge50_frac": life.get("query_head_ge50_frac", ""),
        "query_head_ge75_frac": life.get("query_head_ge75_frac", ""),
        "topk_hit_frac": life.get("topk_hit_position_count", ""),
        "topk_route_mass_mean": life.get("topk_route_mass_mean", ""),
        "topk_route_mass_max": life.get("topk_route_mass_max", ""),
        "top1_hit_frac": "",
        "query_head_risk": life.get("query_head_ge75_frac", ""),
        "source_token_count": life.get("source_token_count", ""),
        "source_retention_mean": life.get("source_retention_mean", ""),
        "source_residual_mean": life.get("source_residual_mean", ""),
        "source_label_mode_frac": life.get("source_label_mode_frac", ""),
        "current_feature_residual_mean": life.get("current_feature_residual_mean", ""),
        "z_cache_current_pair_count": life.get("z_cache_current_pair_count", ""),
        "z_cache_current_cos_mean": life.get("z_cache_current_cos_mean", ""),
        "z_cache_current_cos_route_weighted_mean": life.get("z_cache_current_cos_route_weighted_mean", ""),
        "z_cache_current_l2_mean": life.get("z_cache_current_l2_mean", ""),
        "z_cache_current_l2_route_weighted_mean": life.get("z_cache_current_l2_route_weighted_mean", ""),
        "latent_inconsistent_high_hit": math.isfinite(mean(rsame_parts))
        and mean(rsame_parts) >= 0.5
        and f(life.get("query_head_hit_frac")) >= 0.1,
        "safe_high_hit": math.isfinite(mean(rsame_parts))
        and mean(rsame_parts) < 0.25
        and f(life.get("query_head_hit_frac")) >= 0.1,
        "same_dim": same_dim,
        "same_space_dim": positive_dims[0] if same_dim and positive_dims else 0,
        "same_projection_family": same_projection_family,
        "projection_path": projection_path,
        "projection_is_native": projection_is_native,
        "same_space_assertion_pass": assertion_pass,
        "anchor_alignment_source": anchor_alignment_source,
        "trace_payload": str(payload_path),
    }


def collect_same_space_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    trace_root, paths, trace_source = trace_payload_paths()
    rows: list[dict[str, Any]] = []
    read_errors: list[dict[str, Any]] = []
    for path in paths:
        case_id = case_id_from_trace(path)
        seq, _, _ = parse_case_id(case_id)
        try:
            payload = torch_load(path)
        except Exception as exc:
            read_errors.append({"path": str(path), "error": f"{type(exc).__name__}:{exc}"})
            continue
        if not isinstance(payload, dict):
            continue
        lifecycle = payload.get("ttt_prev_stable_anchor_lifecycle_rows")
        if not isinstance(lifecycle, list):
            continue
        for life in lifecycle:
            if not isinstance(life, dict):
                continue
            z_write_512 = vec(life.get("z_write_key_vec_mean"))
            z_write_projected = vec(life.get("z_write_key_vec_projected_mean"))
            z_ref_cache_k = vec(life.get("z_ref_cache_k_vec_mean"))
            z_cache_k = vec(life.get("z_cache_k_vec_mean"))
            z_current_k = vec(life.get("z_current_k_vec_mean"))
            z_current_q = vec(life.get("z_current_q_vec_mean"))
            z_ref_cache_v = vec(life.get("z_ref_cache_v_vec_mean"))
            z_cache_v = vec(life.get("z_cache_v_vec_mean"))
            z_current_v = vec(life.get("z_current_v_vec_mean"))
            z_write_hidden = vec(life.get("z_write_hidden_vec_mean"))
            z_ref_hidden = vec(life.get("z_ref_hidden_vec_mean"))
            z_cache_hidden = vec(life.get("z_cache_hidden_vec_mean"))
            z_current_hidden = vec(life.get("z_current_hidden_vec_mean"))
            alignment_source = "ttt_prev_stable_anchor_id_patch_to_swa_topk_ids"
            rows.append(
                same_space_row(
                    case_id=case_id,
                    seq=seq,
                    payload_path=path,
                    payload=payload,
                    life=life,
                    canonical_space_name="S-A_swa_cache_k_native_current_k",
                    z_ref=z_ref_cache_k,
                    z_write=None,
                    z_cache=z_cache_k,
                    z_current=z_current_k,
                    projection_path="native_swa_ref_cache_k_cache_k_current_k;z_write_native_missing",
                    projection_is_native=True,
                    same_projection_family=z_cache_k is not None and z_current_k is not None,
                    anchor_alignment_source=alignment_source,
                )
            )
            rows.append(
                same_space_row(
                    case_id=case_id,
                    seq=seq,
                    payload_path=path,
                    payload=payload,
                    life=life,
                    canonical_space_name="S-A_swa_cache_k_projected_write_current_k",
                    z_ref=z_ref_cache_k,
                    z_write=z_write_projected,
                    z_cache=z_cache_k,
                    z_current=z_current_k,
                    projection_path="z_write_chunk_mean_projection_to_swa_head_dim;ref_cache_current_native_swa_k",
                    projection_is_native=False,
                    same_projection_family=z_write_projected is not None and z_cache_k is not None and z_current_k is not None,
                    anchor_alignment_source=alignment_source,
                )
            )
            rows.append(
                same_space_row(
                    case_id=case_id,
                    seq=seq,
                    payload_path=path,
                    payload=payload,
                    life=life,
                    canonical_space_name="S-A_legacy_projected_write_current_q",
                    z_ref=z_ref_cache_k,
                    z_write=z_write_projected,
                    z_cache=z_cache_k,
                    z_current=z_current_q,
                    projection_path="v99_legacy_current_q_not_current_k;z_write_chunk_mean_projection",
                    projection_is_native=False,
                    same_projection_family=z_write_projected is not None and z_cache_k is not None and z_current_q is not None,
                    anchor_alignment_source=alignment_source,
                )
            )
            rows.append(
                same_space_row(
                    case_id=case_id,
                    seq=seq,
                    payload_path=path,
                    payload=payload,
                    life=life,
                    canonical_space_name="S-B_preprojection_hidden",
                    z_ref=z_ref_hidden,
                    z_write=z_write_hidden,
                    z_cache=z_cache_hidden,
                    z_current=z_current_hidden,
                    projection_path="ttt_write_tokens_out_hidden;swa_history_hidden_pre;current_hidden_pre",
                    projection_is_native=True,
                    same_projection_family=all(
                        value is not None for value in [z_ref_hidden, z_write_hidden, z_cache_hidden, z_current_hidden]
                    ),
                    anchor_alignment_source=alignment_source,
                )
            )
            rows.append(
                same_space_row(
                    case_id=case_id,
                    seq=seq,
                    payload_path=path,
                    payload=payload,
                    life=life,
                    canonical_space_name="S-C_global_read_k_space",
                    z_ref=z_ref_cache_v,
                    z_write=None,
                    z_cache=z_cache_v,
                    z_current=z_current_v,
                    projection_path="global_read_k_not_dumped;using_swa_v_only_as_negative_control",
                    projection_is_native=False,
                    same_projection_family=False,
                    anchor_alignment_source=alignment_source,
                )
            )
    diagnostics = {
        "trace_root": str(trace_root),
        "trace_source": trace_source,
        "trace_payload_count": len(paths),
        "trace_read_error_count": len(read_errors),
        "read_errors": read_errors,
    }
    return rows, diagnostics


def track_s() -> dict[str, Any]:
    out = ROOT / "trackS_same_space_latent_state"
    rows, diagnostics = collect_same_space_rows()
    by_space: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_space[str(row.get("canonical_space_name", ""))].append(row)
    coverage_rows: list[dict[str, Any]] = []
    for space, parts in sorted(by_space.items()):
        cases = {str(row.get("case_id", "")) for row in parts if row.get("case_id")}
        seqs = {str(row.get("seq", "")) for row in parts if row.get("seq")}
        row_count = len(parts)
        coverage_rows.append(
            {
                "canonical_space_name": space,
                "row_count": row_count,
                "case_count": len(cases),
                "sequence_coverage": len(seqs),
                "z_ref_coverage": mean([1.0 if int(f(row.get("z_ref_dim"), 0)) > 0 else 0.0 for row in parts]),
                "z_write_coverage": mean([1.0 if int(f(row.get("z_write_dim"), 0)) > 0 else 0.0 for row in parts]),
                "z_cache_coverage": mean([1.0 if int(f(row.get("z_cache_dim"), 0)) > 0 else 0.0 for row in parts]),
                "z_current_coverage": mean([1.0 if int(f(row.get("z_current_dim"), 0)) > 0 else 0.0 for row in parts]),
                "same_dim_fraction": mean([1.0 if b(row.get("same_dim")) else 0.0 for row in parts]),
                "same_projection_family_fraction": mean(
                    [1.0 if b(row.get("same_projection_family")) else 0.0 for row in parts]
                ),
                "projection_native_fraction": mean([1.0 if b(row.get("projection_is_native")) else 0.0 for row in parts]),
                "same_space_assertion_pass_fraction": mean(
                    [1.0 if b(row.get("same_space_assertion_pass")) else 0.0 for row in parts]
                ),
                "R_same_mean": mean([f(row.get("R_same")) for row in parts]),
                "R_cache_current_mean": mean([f(row.get("R_cache_current")) for row in parts]),
            }
        )
    best = sorted(
        coverage_rows,
        key=lambda row: (
            f(row.get("same_space_assertion_pass_fraction")),
            f(row.get("z_write_coverage")),
            f(row.get("z_cache_coverage")),
            f(row.get("z_current_coverage")),
            f(row.get("row_count")),
        ),
        reverse=True,
    )
    best_row = best[0] if best else {}
    trace_source = diagnostics.get("trace_source", "")
    parity = no_action_parity(Path(str(diagnostics.get("trace_root", ""))))
    no_action_parity_available = b(parity.get("available"))
    no_action_parity_all_smoke_cases = b(parity.get("all_pass"))
    case_count = max([int(f(row.get("case_count"), 0)) for row in coverage_rows] or [0])
    sequence_coverage = max([int(f(row.get("sequence_coverage"), 0)) for row in coverage_rows] or [0])
    anchor_rows = max([int(f(row.get("row_count"), 0)) for row in coverage_rows] or [0])
    gate = (
        str(trace_source).startswith("v100")
        and case_count >= 28
        and sequence_coverage >= 4
        and anchor_rows >= 10000
        and f(best_row.get("same_space_assertion_pass_fraction")) >= 0.95
        and f(best_row.get("z_write_coverage")) >= 0.80
        and f(best_row.get("z_cache_coverage")) >= 0.80
        and f(best_row.get("z_current_coverage")) >= 0.80
        and no_action_parity_available
        and no_action_parity_all_smoke_cases
    )
    summary = {
        "schema": "acl2_v100_trackS_same_space_latent_state_v1",
        "status": "complete" if rows else "blocked_no_trace_payloads",
        "gate_pass": gate,
        "runtime_action_allowed": False,
        "trace_root": diagnostics.get("trace_root"),
        "trace_source": trace_source,
        "trace_payload_count": diagnostics.get("trace_payload_count"),
        "trace_read_error_count": diagnostics.get("trace_read_error_count"),
        "case_count": case_count,
        "sequence_coverage": sequence_coverage,
        "anchor_id_rows": anchor_rows,
        "best_canonical_space": best_row.get("canonical_space_name", ""),
        "best_same_space_assertion_pass_fraction": best_row.get("same_space_assertion_pass_fraction", math.nan),
        "best_z_write_coverage": best_row.get("z_write_coverage", math.nan),
        "best_z_cache_coverage": best_row.get("z_cache_coverage", math.nan),
        "best_z_current_coverage": best_row.get("z_current_coverage", math.nan),
        "no_action_parity_available": no_action_parity_available,
        "no_action_parity_all_smoke_cases": no_action_parity_all_smoke_cases,
        "blocker": ""
        if gate
        else "Track S same-space gate failed: z_ref/native z_write/current support/no-action parity are not all available in a v100 28-case trace.",
    }
    write_json(out / "summary.json", summary)
    write_rows(out / "same_space_anchor_rows.csv", rows)
    write_rows(out / "rows.csv", rows)
    write_rows(out / "same_space_coverage_rows.csv", coverage_rows)
    write_rows(out / "no_action_parity_rows.csv", parity.get("rows", []))
    write_rows(out / "gate_checks.csv", [
        {"gate": "trace_source_v100", "pass": str(trace_source).startswith("v100"), "value": trace_source},
        {"gate": "case_count_ge28", "pass": case_count >= 28, "value": case_count},
        {"gate": "sequence_coverage_ge4", "pass": sequence_coverage >= 4, "value": sequence_coverage},
        {"gate": "anchor_rows_ge10000", "pass": anchor_rows >= 10000, "value": anchor_rows},
        {
            "gate": "same_space_assertion_pass_fraction_ge095",
            "pass": f(best_row.get("same_space_assertion_pass_fraction")) >= 0.95,
            "value": best_row.get("same_space_assertion_pass_fraction", math.nan),
        },
        {
            "gate": "z_write_cache_current_coverage_ge080",
            "pass": (
                f(best_row.get("z_write_coverage")) >= 0.80
                and f(best_row.get("z_cache_coverage")) >= 0.80
                and f(best_row.get("z_current_coverage")) >= 0.80
            ),
            "value": "see same_space_coverage_rows.csv",
        },
        {"gate": "no_action_parity_available", "pass": no_action_parity_available, "value": no_action_parity_available},
        {"gate": "no_action_parity_all_cases", "pass": no_action_parity_all_smoke_cases, "value": no_action_parity_all_smoke_cases},
    ])
    write_text(
        out / "same_space_coverage_summary.json",
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    write_text(
        out / "same_space_failure_report.md",
        "# Track S Same-Space Failure Report\n\n"
        f"- gate_pass={gate}\n"
        f"- best_canonical_space={summary['best_canonical_space']}\n"
        f"- blocker={summary['blocker'] or 'none'}\n"
        "- Projected z_write rows are retained as diagnostic evidence only; projection_is_native=false.\n"
        "- S-B preprojection hidden is accepted only when z_write/z_ref/z_cache/z_current hidden vectors are all dumped in the new trace.\n"
        "- z_ref for S-A remains the first committed cache K proxy, not a separate semantic-anchor bank state.\n",
    )
    write_text(
        out / "projection_path_mismatch_report.md",
        "# Projection Path Mismatch Report\n\n"
        "- S-A native cache/current K-space can compare cache K with current K when z_current_k_vec_mean is dumped.\n"
        "- TTT z_write_key_vec_mean is replay-key state, not native SWA K-space.\n"
        "- z_write_key_vec_projected_mean is chunk-mean projected to 64 dimensions and cannot satisfy native same-space assertion.\n"
        "- S-B requires ttt_write_tokens_out_hidden + swa_history_hidden_pre + current_hidden_pre in a fresh trace.\n"
        "- S-C still requires deeper global READ diagnostic hooks.\n",
    )
    write_text(out / "failure_report.md", summary["blocker"] + "\n")
    write_text(
        out / "what_would_have_to_be_true_to_pass.md",
        "A v100 28-case trace must contain z_ref, native z_write, z_cache and z_current in the same canonical projection family, plus no-action parity evidence.\n",
    )
    write_text(out / "visual_manifest.csv", "path,description\n")
    return summary


def load_v99_case_labels() -> dict[str, dict[str, Any]]:
    cases: dict[str, dict[str, Any]] = {}
    v98_meta = {
        row.get("case_id", ""): row
        for row in read_rows(V98_ROOT / "stage1_trackK_swa_v2_strict_stable_eligibility/case_universe_rows.csv")
        if row.get("case_id")
    }
    for row in read_rows(V99_ROOT / "trackN_semantic_anchor_identity_memory_graph/graph_case_rows.csv"):
        case_id = str(row.get("case_id", ""))
        if not case_id:
            continue
        fallback = v98_meta.get(case_id, {})
        cases[case_id] = {
            "case_id": case_id,
            "seq": row.get("seq", ""),
            "case_label": row.get("case_label", ""),
            "L3_handoff_transfer_penalty_proxy": f(row.get("L3_handoff_transfer_penalty_proxy")),
            "prev_chunk": row.get("prev_chunk", ""),
            "curr_chunk": row.get("curr_chunk", ""),
            "failure_type": row.get("failure_type", ""),
            "case_audit_metadata_source": "v99_graph_rows+v98_stage1_fallback" if fallback else "v99_graph_rows",
        }
        for field in CASE_AUDIT_META_FIELDS:
            cases[case_id][field] = row.get(field, fallback.get(field, ""))
    return cases


def stage3_case_rows(canonical_space: str = "S-B_preprojection_hidden") -> list[dict[str, Any]]:
    labels = load_v99_case_labels()
    source_rows = read_rows(ROOT / "trackS_same_space_latent_state/same_space_anchor_rows.csv")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in source_rows:
        if row.get("canonical_space_name") != canonical_space:
            continue
        case_id = str(row.get("case_id", ""))
        if case_id and case_id in labels:
            grouped[case_id].append(row)

    rows: list[dict[str, Any]] = []
    for case_id, parts in sorted(grouped.items()):
        label = labels[case_id]
        q_vals = [f(row.get("query_hit_frac")) for row in parts]
        risk_vals = [f(row.get("query_head_risk")) for row in parts]
        r_same = [f(row.get("R_same")) for row in parts]
        r_wc = [f(row.get("R_write_cache")) for row in parts]
        r_cc = [f(row.get("R_cache_current")) for row in parts]
        r_rc = [f(row.get("R_ref_current")) for row in parts]
        qmax_vals = [f(row.get("query_hit_max")) for row in parts]
        qge50_vals = [f(row.get("query_head_ge50_frac")) for row in parts]
        qge75_vals = [f(row.get("query_head_ge75_frac")) for row in parts]
        route_mass_mean_vals = [f(row.get("topk_route_mass_mean")) for row in parts]
        route_mass_max_vals = [f(row.get("topk_route_mass_max")) for row in parts]
        current_residual_vals = [f(row.get("current_feature_residual_mean")) for row in parts]
        source_retention_vals = [f(row.get("source_retention_mean")) for row in parts]
        source_residual_vals = [f(row.get("source_residual_mean")) for row in parts]
        source_label_frac_vals = [f(row.get("source_label_mode_frac")) for row in parts]
        semantic_counts = Counter(str(row.get("semantic_class", "")) for row in parts if row.get("semantic_class") not in {"", None})
        dominant_semantic_class = semantic_counts.most_common(1)[0][0] if semantic_counts else ""
        dominant_semantic_frac = (
            semantic_counts.most_common(1)[0][1] / len(parts) if semantic_counts and parts else math.nan
        )
        high_hit_high_r = [1.0 if b(row.get("latent_inconsistent_high_hit")) else 0.0 for row in parts]
        safe_high_hit = [1.0 if b(row.get("safe_high_hit")) else 0.0 for row in parts]
        write_to_use_visible = [
            1.0
            if math.isfinite(f(row.get("R_write_cache"))) and math.isfinite(f(row.get("R_cache_current")))
            else 0.0
            for row in parts
        ]
        rows.append({
            **label,
            "canonical_space_name": canonical_space,
            "anchor_row_count": len(parts),
            "dominant_semantic_class": dominant_semantic_class,
            "dominant_semantic_frac": dominant_semantic_frac,
            "semantic_unique_count": len(semantic_counts),
            "same_space_assertion_pass_fraction": mean(
                [1.0 if b(row.get("same_space_assertion_pass")) else 0.0 for row in parts]
            ),
            "query_hit_mean": mean(q_vals),
            "query_hit_max": max(finite(q_vals), default=math.nan),
            "query_hit_max_mean": mean(qmax_vals),
            "query_hit_max_max": max(finite(qmax_vals), default=math.nan),
            "query_head_ge50_frac_mean": mean(qge50_vals),
            "query_head_ge75_frac_mean": mean(qge75_vals),
            "query_head_risk_mean": mean(risk_vals),
            "topk_route_mass_mean": mean(route_mass_mean_vals),
            "topk_route_mass_max": max(finite(route_mass_max_vals), default=math.nan),
            "source_retention_mean": mean(source_retention_vals),
            "source_residual_mean": mean(source_residual_vals),
            "source_label_mode_frac_mean": mean(source_label_frac_vals),
            "current_feature_residual_mean": mean(current_residual_vals),
            "current_feature_residual_max": max(finite(current_residual_vals), default=math.nan),
            "R_same_mean": mean(r_same),
            "R_same_max": max(finite(r_same), default=math.nan),
            "R_write_cache_mean": mean(r_wc),
            "R_cache_current_mean": mean(r_cc),
            "R_ref_current_mean": mean(r_rc),
            "high_hit_high_R_frac": mean(high_hit_high_r),
            "safe_high_hit_frac": mean(safe_high_hit),
            "query_hit_x_R_same": mean(q_vals) * mean(r_same)
            if math.isfinite(mean(q_vals)) and math.isfinite(mean(r_same))
            else math.nan,
            "query_hit_max_x_R_same": safe_product(mean(qmax_vals), mean(r_same)),
            "query_hit_max_x_current_residual": safe_product(mean(qmax_vals), mean(current_residual_vals)),
            "query_hit_max_x_R_same_x_current_residual": safe_product(
                mean(qmax_vals), mean(r_same), mean(current_residual_vals)
            ),
            "query_head_ge50_x_R_same": safe_product(mean(qge50_vals), mean(r_same)),
            "query_head_ge75_x_R_same": safe_product(mean(qge75_vals), mean(r_same)),
            "query_risk_x_R_same": mean(risk_vals) * mean(r_same)
            if math.isfinite(mean(risk_vals)) and math.isfinite(mean(r_same))
            else math.nan,
            "current_residual_x_R_same": safe_product(mean(current_residual_vals), mean(r_same)),
            "query_hit_x_current_residual": safe_product(mean(q_vals), mean(current_residual_vals)),
            "query_hit_x_R_same_x_current_residual": safe_product(mean(q_vals), mean(r_same), mean(current_residual_vals)),
            "route_mass_x_R_same": safe_product(mean(route_mass_mean_vals), mean(r_same)),
            "route_mass_max_x_R_same": safe_product(max(finite(route_mass_max_vals), default=math.nan), mean(r_same)),
            "fresh_supported_score_proxy": (
                mean(q_vals) / ((1.0 + mean(r_same)) * (1.0 + mean(current_residual_vals)))
                if math.isfinite(mean(q_vals)) and math.isfinite(mean(r_same)) and math.isfinite(mean(current_residual_vals))
                else math.nan
            ),
            "write_to_use_chain_coverage": mean(write_to_use_visible),
            "write_cache_current_risk": (
                mean(q_vals) * mean([value for value in [mean(r_wc), mean(r_cc)] if math.isfinite(value)])
                if math.isfinite(mean(q_vals)) and math.isfinite(mean([value for value in [mean(r_wc), mean(r_cc)] if math.isfinite(value)]))
                else math.nan
            ),
            "write_cache_current_risk_x_current_residual": safe_product(
                mean(q_vals),
                mean([value for value in [mean(r_wc), mean(r_cc)] if math.isfinite(value)]),
                mean(current_residual_vals),
            ),
        })
    return rows


def evaluate_pattern(case_rows: list[dict[str, Any]], cue_name: str, field: str, direction: str) -> dict[str, Any]:
    available = [
        row for row in case_rows
        if math.isfinite(f(row.get(field))) and row.get("case_label") in {"good", "non_good"}
    ]
    bad = [row for row in available if row.get("case_label") == "non_good"]
    good = [row for row in available if row.get("case_label") == "good"]
    values = sorted({f(row.get(field)) for row in available if math.isfinite(f(row.get(field)))})
    best: dict[str, Any] | None = None
    for threshold in values:
        if direction == "lower_bad":
            selected = [row for row in available if f(row.get(field)) <= threshold]
        else:
            selected = [row for row in available if f(row.get(field)) >= threshold]
        tp = [row for row in selected if row.get("case_label") == "non_good"]
        fp = [row for row in selected if row.get("case_label") == "good"]
        recall = len(tp) / len(bad) if bad else math.nan
        fpr = len(fp) / len(good) if good else math.nan
        ba = (recall + (1.0 - fpr)) / 2.0 if math.isfinite(recall) and math.isfinite(fpr) else math.nan
        candidate = {
            "threshold": threshold,
            "selected": selected,
            "tp": tp,
            "fp": fp,
            "recall": recall,
            "fpr": fpr,
            "balanced_accuracy": ba,
        }
        if best is None or (
            f(candidate["balanced_accuracy"]),
            f(candidate["recall"]),
            -f(candidate["fpr"], 1.0),
        ) > (
            f(best["balanced_accuracy"]),
            f(best["recall"]),
            -f(best["fpr"], 1.0),
        ):
            best = candidate
    best = best or {
        "threshold": math.nan,
        "selected": [],
        "tp": [],
        "fp": [],
        "recall": math.nan,
        "fpr": math.nan,
        "balanced_accuracy": math.nan,
    }
    selected = best["selected"]
    tp = best["tp"]
    fp = best["fp"]
    missed = [row for row in bad if row not in tp]
    seq_counts = Counter(str(row.get("seq", "")) for row in tp if row.get("seq"))
    selected_positive_sequence_max_frac = (
        max(seq_counts.values()) / len(tp) if tp and seq_counts else math.nan
    )
    corr = pearson([row.get(field) for row in available], [row.get("L3_handoff_transfer_penalty_proxy") for row in available])
    direction_correct = (
        corr > 0.0 if direction == "higher_bad" else corr < 0.0
    ) if math.isfinite(corr) else False
    return {
        "cue_name": cue_name,
        "field": field,
        "direction": direction,
        "available_case_count": len(available),
        "sequence_coverage": len({row.get("seq") for row in available if row.get("seq")}),
        "positive_case_count": len(bad),
        "good_control_case_count": len(good),
        "threshold": best["threshold"],
        "balanced_accuracy": best["balanced_accuracy"],
        "bad_recall": best["recall"],
        "good_FPR": best["fpr"],
        "abs_corr_L3": abs(corr) if math.isfinite(corr) else math.nan,
        "corr_L3": corr,
        "corr_direction_correct": direction_correct,
        "selected_case_count": len(selected),
        "selected_positive_sequence_max_frac": selected_positive_sequence_max_frac,
        "true_positive_cases": ";".join(str(row.get("case_id")) for row in tp),
        "false_positive_cases": ";".join(str(row.get("case_id")) for row in fp),
        "missed_positive_cases": ";".join(str(row.get("case_id")) for row in missed),
        "anchor_id_rotation_margin": "",
        "semantic_label_rotation_margin": "",
        "query_head_random_margin": "",
        "same_count_random_margin": "",
        "control_margins_available": False,
        "gate_pass": False,
    }


def with_gate(metric: dict[str, Any], *, require_ba_gain: float | None = None, ba_gain: float = math.nan) -> dict[str, Any]:
    pass_without_controls = (
        f(metric.get("bad_recall")) >= 0.65
        and f(metric.get("good_FPR"), 1.0) <= 0.25
        and f(metric.get("abs_corr_L3")) >= 0.50
        and b(metric.get("corr_direction_correct"))
        and int(f(metric.get("sequence_coverage"), 0)) >= 4
        and f(metric.get("selected_positive_sequence_max_frac"), 1.0) <= 0.60
    )
    if require_ba_gain is not None:
        pass_without_controls = pass_without_controls and math.isfinite(ba_gain) and ba_gain >= require_ba_gain
        metric["BA_gain_over_query_hit_only"] = ba_gain
    control_pass = (
        f(metric.get("anchor_id_rotation_margin")) >= 0.05
        and f(metric.get("semantic_label_rotation_margin")) >= 0.05
        and f(metric.get("query_head_random_margin")) >= 0.05
    )
    metric["gate_without_controls_pass"] = pass_without_controls
    metric["gate_pass"] = pass_without_controls and control_pass
    return metric


def threshold_feasibility_rows(case_rows: list[dict[str, Any]], pattern_specs: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for spec in pattern_specs:
        field = spec["field"]
        direction = spec.get("direction", "higher_bad")
        available = [
            row for row in case_rows
            if math.isfinite(f(row.get(field))) and row.get("case_label") in {"good", "non_good"}
        ]
        bad = [row for row in available if row.get("case_label") == "non_good"]
        good = [row for row in available if row.get("case_label") == "good"]
        thresholds = sorted({f(row.get(field)) for row in available if math.isfinite(f(row.get(field)))})
        feasible: list[dict[str, Any]] = []
        best: dict[str, Any] | None = None
        for threshold in thresholds:
            selected = [
                row for row in available
                if (f(row.get(field)) <= threshold if direction == "lower_bad" else f(row.get(field)) >= threshold)
            ]
            tp = [row for row in selected if row.get("case_label") == "non_good"]
            fp = [row for row in selected if row.get("case_label") == "good"]
            recall = len(tp) / len(bad) if bad else math.nan
            fpr = len(fp) / len(good) if good else math.nan
            ba = (recall + (1.0 - fpr)) / 2.0 if math.isfinite(recall) and math.isfinite(fpr) else math.nan
            seq_counts = Counter(str(row.get("seq", "")) for row in tp if row.get("seq"))
            selected_positive_sequence_max_frac = (
                max(seq_counts.values()) / len(tp) if tp and seq_counts else math.nan
            )
            selected_positive_sequence_coverage = len(seq_counts)
            row = {
                "cue_name": spec.get("cue_name", field),
                "field": field,
                "direction": direction,
                "threshold": threshold,
                "balanced_accuracy": ba,
                "bad_recall": recall,
                "good_FPR": fpr,
                "selected_case_count": len(selected),
                "selected_positive_sequence_coverage": selected_positive_sequence_coverage,
                "selected_positive_sequence_max_frac": selected_positive_sequence_max_frac,
                "true_positive_cases": ";".join(str(item.get("case_id")) for item in tp),
                "false_positive_cases": ";".join(str(item.get("case_id")) for item in fp),
            }
            if (
                recall >= 0.65
                and fpr <= 0.25
                and selected_positive_sequence_coverage >= 4
                and math.isfinite(selected_positive_sequence_max_frac)
                and selected_positive_sequence_max_frac <= 0.60
            ):
                feasible.append(row)
            if best is None or (
                f(row.get("balanced_accuracy")),
                f(row.get("bad_recall")),
                -f(row.get("good_FPR"), 1.0),
            ) > (
                f(best.get("balanced_accuracy")),
                f(best.get("bad_recall")),
                -f(best.get("good_FPR"), 1.0),
            ):
                best = row
        if best:
            out.append({
                **best,
                "row_kind": "best_balanced_accuracy",
                "feasible_gate_threshold_count": len(feasible),
            })
        for row in feasible:
            out.append({**row, "row_kind": "feasible_gate_threshold", "feasible_gate_threshold_count": len(feasible)})
    return out


def best_threshold_metric(case_rows: list[dict[str, Any]], field: str, direction: str) -> dict[str, Any]:
    return evaluate_pattern(case_rows, f"{field}_{direction}", field, direction)


def apply_threshold_metric(case_rows: list[dict[str, Any]], field: str, direction: str, threshold: float) -> dict[str, Any]:
    available = [
        row for row in case_rows
        if math.isfinite(f(row.get(field))) and row.get("case_label") in {"good", "non_good"}
    ]
    bad = [row for row in available if row.get("case_label") == "non_good"]
    good = [row for row in available if row.get("case_label") == "good"]
    selected = [
        row for row in available
        if (f(row.get(field)) <= threshold if direction == "lower_bad" else f(row.get(field)) >= threshold)
    ]
    tp = [row for row in selected if row.get("case_label") == "non_good"]
    fp = [row for row in selected if row.get("case_label") == "good"]
    recall = len(tp) / len(bad) if bad else math.nan
    fpr = len(fp) / len(good) if good else math.nan
    ba = (recall + (1.0 - fpr)) / 2.0 if math.isfinite(recall) and math.isfinite(fpr) else math.nan
    seq_counts = Counter(str(row.get("seq", "")) for row in tp if row.get("seq"))
    selected_positive_sequence_max_frac = max(seq_counts.values()) / len(tp) if tp and seq_counts else math.nan
    return {
        "available_case_count": len(available),
        "positive_case_count": len(bad),
        "good_control_case_count": len(good),
        "selected_case_count": len(selected),
        "bad_recall": recall,
        "good_FPR": fpr,
        "balanced_accuracy": ba,
        "selected_positive_sequence_coverage": len(seq_counts),
        "selected_positive_sequence_max_frac": selected_positive_sequence_max_frac,
        "true_positive_cases": ";".join(str(row.get("case_id")) for row in tp),
        "false_positive_cases": ";".join(str(row.get("case_id")) for row in fp),
        "missed_positive_cases": ";".join(str(row.get("case_id")) for row in bad if row not in tp),
    }


def quantile(values: list[float], q: float) -> float:
    vals = sorted(v for v in values if math.isfinite(v))
    if not vals:
        return math.nan
    if len(vals) == 1:
        return vals[0]
    pos = max(0.0, min(1.0, q)) * (len(vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def metric_from_selected(available: list[dict[str, Any]], selected_indexes: set[int]) -> dict[str, Any]:
    bad = [row for row in available if row.get("case_label") == "non_good"]
    good = [row for row in available if row.get("case_label") == "good"]
    selected = [row for idx, row in enumerate(available) if idx in selected_indexes]
    tp = [row for row in selected if row.get("case_label") == "non_good"]
    fp = [row for row in selected if row.get("case_label") == "good"]
    recall = len(tp) / len(bad) if bad else math.nan
    fpr = len(fp) / len(good) if good else math.nan
    ba = (recall + (1.0 - fpr)) / 2.0 if math.isfinite(recall) and math.isfinite(fpr) else math.nan
    return {"bad_recall": recall, "good_FPR": fpr, "balanced_accuracy": ba}


def attach_control_audit(metric: dict[str, Any], case_rows: list[dict[str, Any]]) -> dict[str, Any]:
    available = [
        row for row in case_rows
        if math.isfinite(f(row.get(metric.get("field", "")))) and row.get("case_label") in {"good", "non_good"}
    ]
    selected_count = int(f(metric.get("selected_case_count"), 0))
    selected_count = max(0, min(selected_count, len(available)))
    seed_payload = f"{metric.get('cue_name','')}|{metric.get('field','')}|{metric.get('direction','')}|{selected_count}"
    seed = int(hashlib.sha256(seed_payload.encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(seed)
    random_ba: list[float] = []
    repeats = 512 if selected_count > 0 else 0
    for _ in range(repeats):
        selected = set(rng.sample(range(len(available)), selected_count))
        random_ba.append(f(metric_from_selected(available, selected).get("balanced_accuracy")))
    random_mean = mean(random_ba)
    random_p95 = quantile(random_ba, 0.95)
    random_max = max(finite(random_ba), default=math.nan)
    actual_ba = f(metric.get("balanced_accuracy"))
    metric["same_count_random_available"] = repeats > 0
    metric["same_count_random_repeats"] = repeats
    metric["same_count_random_seed"] = seed
    metric["same_count_random_BA_mean"] = random_mean
    metric["same_count_random_BA_p95"] = random_p95
    metric["same_count_random_BA_max"] = random_max
    metric["same_count_random_margin"] = actual_ba - random_p95 if math.isfinite(actual_ba) and math.isfinite(random_p95) else math.nan
    metric["anchor_id_rotation_margin"] = 0.0
    metric["anchor_id_rotation_applicable"] = False
    metric["anchor_id_rotation_note"] = "case_level_selector_does_not_condition_on_anchor_id; edge_level_anchor_rows_required"
    metric["semantic_label_rotation_margin"] = 0.0
    metric["semantic_label_rotation_applicable"] = False
    metric["semantic_label_rotation_note"] = "case_level_selector_does_not_condition_on_per_anchor_semantic_label; edge_level_permutation_required"
    metric["query_head_random_margin"] = 0.0
    metric["query_head_random_applicable"] = False
    metric["query_head_random_note"] = "rows_are_route_weighted_all_heads; per_head_rows_required"
    metric["control_margins_available"] = False
    return metric


def control_audit_rows(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric in metrics:
        rows.append({
            "cue_name": metric.get("cue_name", ""),
            "field": metric.get("field", ""),
            "direction": metric.get("direction", ""),
            "selected_case_count": metric.get("selected_case_count", ""),
            "balanced_accuracy": metric.get("balanced_accuracy", math.nan),
            "same_count_random_available": metric.get("same_count_random_available", False),
            "same_count_random_repeats": metric.get("same_count_random_repeats", 0),
            "same_count_random_seed": metric.get("same_count_random_seed", ""),
            "same_count_random_BA_mean": metric.get("same_count_random_BA_mean", math.nan),
            "same_count_random_BA_p95": metric.get("same_count_random_BA_p95", math.nan),
            "same_count_random_BA_max": metric.get("same_count_random_BA_max", math.nan),
            "same_count_random_margin": metric.get("same_count_random_margin", math.nan),
            "anchor_id_rotation_margin": metric.get("anchor_id_rotation_margin", ""),
            "anchor_id_rotation_applicable": metric.get("anchor_id_rotation_applicable", ""),
            "anchor_id_rotation_note": metric.get("anchor_id_rotation_note", ""),
            "semantic_label_rotation_margin": metric.get("semantic_label_rotation_margin", ""),
            "semantic_label_rotation_applicable": metric.get("semantic_label_rotation_applicable", ""),
            "semantic_label_rotation_note": metric.get("semantic_label_rotation_note", ""),
            "query_head_random_margin": metric.get("query_head_random_margin", ""),
            "query_head_random_applicable": metric.get("query_head_random_applicable", ""),
            "query_head_random_available": metric.get("query_head_random_available", False),
            "query_head_random_repeats": metric.get("query_head_random_repeats", 0),
            "query_head_random_seed": metric.get("query_head_random_seed", ""),
            "query_head_random_BA_mean": metric.get("query_head_random_BA_mean", ""),
            "query_head_random_BA_p95": metric.get("query_head_random_BA_p95", ""),
            "query_head_random_BA_max": metric.get("query_head_random_BA_max", ""),
            "query_head_random_note": metric.get("query_head_random_note", ""),
            "anchor_identity_random_available": metric.get("anchor_identity_random_available", False),
            "anchor_identity_random_repeats": metric.get("anchor_identity_random_repeats", 0),
            "anchor_identity_random_seed": metric.get("anchor_identity_random_seed", ""),
            "anchor_identity_random_BA_mean": metric.get("anchor_identity_random_BA_mean", ""),
            "anchor_identity_random_BA_p95": metric.get("anchor_identity_random_BA_p95", ""),
            "anchor_identity_random_BA_max": metric.get("anchor_identity_random_BA_max", ""),
            "anchor_identity_random_margin": metric.get("anchor_identity_random_margin", ""),
            "control_margins_available": metric.get("control_margins_available", False),
        })
    return rows


def tensor_head_value(payload: dict[str, Any], key: str, head_idx: int) -> float:
    value = payload.get(key)
    if torch.is_tensor(value) and value.ndim >= 1 and int(value.shape[0]) > int(head_idx):
        return f(value[int(head_idx)].item())
    return math.nan


def collect_edge_head_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    labels = load_v99_case_labels()
    trace_root, paths, trace_source = trace_payload_paths()
    rows: list[dict[str, Any]] = []
    read_errors: list[dict[str, Any]] = []
    for path in paths:
        case_id = case_id_from_trace(path)
        if case_id not in labels:
            continue
        try:
            payload = torch_load(path)
        except Exception as exc:
            read_errors.append({"path": str(path), "error": f"{type(exc).__name__}:{exc}"})
            continue
        if not isinstance(payload, dict):
            continue
        hit = payload.get("current_Q_to_cache_K_topk_ttt_prev_stable_anchor_hit_mask")
        anchor_ids = payload.get("current_Q_to_cache_K_topk_ttt_prev_stable_anchor_ids")
        scores = payload.get("current_Q_to_cache_K_topk_scores")
        if not (torch.is_tensor(hit) and torch.is_tensor(anchor_ids)):
            read_errors.append({"path": str(path), "error": "missing_hit_mask_or_anchor_ids"})
            continue
        hit = hit.detach().cpu().bool()
        anchor_ids = anchor_ids.detach().cpu().long()
        scores = scores.detach().cpu().float() if torch.is_tensor(scores) else None
        if hit.ndim != 4 or anchor_ids.shape != hit.shape:
            read_errors.append({"path": str(path), "error": f"bad_shape hit={tuple(hit.shape)} ids={tuple(anchor_ids.shape)}"})
            continue
        batch, head_count, query_count, topk_count = [int(v) for v in hit.shape]
        label = labels[case_id]
        for batch_idx in range(batch):
            for head_idx in range(head_count):
                hit_h = hit[batch_idx, head_idx]
                ids_h = anchor_ids[batch_idx, head_idx]
                valid = hit_h & (ids_h >= 0)
                valid_ids = ids_h[valid]
                if int(valid_ids.numel()) > 0:
                    unique_ids, counts = torch.unique(valid_ids, return_counts=True)
                    dominant_count = int(counts.max().item())
                    unique_anchor_count = int(unique_ids.numel())
                    dominant_anchor_frac = dominant_count / max(int(valid_ids.numel()), 1)
                else:
                    dominant_count = 0
                    unique_anchor_count = 0
                    dominant_anchor_frac = math.nan
                stable_score_mean = math.nan
                all_score_mean = math.nan
                if scores is not None and scores.shape == hit.shape:
                    scores_h = scores[batch_idx, head_idx]
                    all_score_mean = f(scores_h.float().mean().item())
                    if bool(valid.any()):
                        stable_score_mean = f(scores_h[valid].float().mean().item())
                rows.append({
                    **label,
                    "trace_source": trace_source,
                    "trace_payload": str(path),
                    "layer": payload.get("layer", ""),
                    "swa_layer_idx": payload.get("swa_layer_idx", ""),
                    "batch_idx": batch_idx,
                    "head_idx": head_idx,
                    "query_count": query_count,
                    "topk_count": topk_count,
                    "stable_query_hit_frac": f(hit_h.any(dim=-1).float().mean().item()),
                    "stable_topk_hit_frac": f(hit_h.float().mean().item()),
                    "top1_stable_hit_frac": f(hit_h[:, 0].float().mean().item()) if topk_count > 0 else math.nan,
                    "stable_hit_count": int(valid.sum().item()),
                    "unique_stable_anchor_count": unique_anchor_count,
                    "dominant_stable_anchor_count": dominant_count,
                    "dominant_stable_anchor_frac": dominant_anchor_frac,
                    "stable_score_mean": stable_score_mean,
                    "all_topk_score_mean": all_score_mean,
                    "topk_query_frame_hit_frac": tensor_head_value(payload, "current_Q_to_cache_K_topk_query_frame_hit_frac_by_head", head_idx),
                    "topk_same_frame_frac": tensor_head_value(payload, "current_Q_to_cache_K_topk_same_frame_frac_by_head", head_idx),
                    "top1_abs_frame_delta_mean": tensor_head_value(payload, "current_Q_to_cache_K_top1_abs_frame_delta_mean_by_head", head_idx),
                    "top1_cache_frame_switch_rate": tensor_head_value(payload, "current_Q_to_cache_K_top1_cache_frame_switch_rate_by_head", head_idx),
                    "top1_cache_frame_unique_frac": tensor_head_value(payload, "current_Q_to_cache_K_top1_cache_frame_unique_frac_by_head", head_idx),
                    "top1_cache_index_switch_rate": tensor_head_value(payload, "current_Q_to_cache_K_top1_cache_index_switch_rate_by_head", head_idx),
                    "top1_cache_index_unique_frac": tensor_head_value(payload, "current_Q_to_cache_K_top1_cache_index_unique_frac_by_head", head_idx),
                    "route_entropy_mean": tensor_head_value(payload, "route_entropy_mean_by_head", head_idx),
                    "feature_transport_residual": tensor_head_value(payload, "feature_transport_residual_by_head", head_idx),
                    "cache_K_stability": tensor_head_value(payload, "cache_K_stability_by_head", head_idx),
                    "cache_V_stability": tensor_head_value(payload, "cache_V_stability_by_head", head_idx),
                    "stable_pair_mass": tensor_head_value(payload, "stable_structure_pair_mass_by_head", head_idx),
                    "unreliable_pair_mass": tensor_head_value(payload, "unreliable_dynamic_boundary_pair_mass_by_head", head_idx),
                    "stable_actual_minus_random": tensor_head_value(payload, "stable_route_actual_minus_random_by_head", head_idx),
                    "unreliable_actual_minus_random": tensor_head_value(payload, "unreliable_route_actual_minus_random_by_head", head_idx),
                })
    return rows, read_errors


def edge_head_case_rows(edge_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in edge_rows:
        grouped[str(row.get("case_id", ""))].append(row)
    rows: list[dict[str, Any]] = []
    for case_id, parts in sorted(grouped.items()):
        if not parts:
            continue
        base = case_base(parts[0])
        def vals(field: str) -> list[float]:
            return [f(row.get(field)) for row in parts]
        rows.append({
            **base,
            "edge_head_row_count": len(parts),
            "head_stable_query_hit_max": max(finite(vals("stable_query_hit_frac")), default=math.nan),
            "head_stable_query_hit_mean": mean(vals("stable_query_hit_frac")),
            "head_stable_topk_hit_max": max(finite(vals("stable_topk_hit_frac")), default=math.nan),
            "head_top1_stable_hit_max": max(finite(vals("top1_stable_hit_frac")), default=math.nan),
            "head_dominant_anchor_frac_max": max(finite(vals("dominant_stable_anchor_frac")), default=math.nan),
            "head_unique_anchor_count_max": max(finite(vals("unique_stable_anchor_count")), default=math.nan),
            "head_stable_score_mean_max": max(finite(vals("stable_score_mean")), default=math.nan),
            "head_topk_query_frame_hit_max": max(finite(vals("topk_query_frame_hit_frac")), default=math.nan),
            "head_topk_same_frame_frac_max": max(finite(vals("topk_same_frame_frac")), default=math.nan),
            "head_top1_abs_frame_delta_mean_max": max(finite(vals("top1_abs_frame_delta_mean")), default=math.nan),
            "head_top1_abs_frame_delta_mean_mean": mean(vals("top1_abs_frame_delta_mean")),
            "head_top1_cache_frame_switch_rate_max": max(finite(vals("top1_cache_frame_switch_rate")), default=math.nan),
            "head_top1_cache_frame_unique_frac_max": max(finite(vals("top1_cache_frame_unique_frac")), default=math.nan),
            "head_top1_cache_index_switch_rate_max": max(finite(vals("top1_cache_index_switch_rate")), default=math.nan),
            "head_top1_cache_index_unique_frac_max": max(finite(vals("top1_cache_index_unique_frac")), default=math.nan),
            "head_route_entropy_mean_max": max(finite(vals("route_entropy_mean")), default=math.nan),
            "head_feature_transport_residual_max": max(finite(vals("feature_transport_residual")), default=math.nan),
            "head_unreliable_pair_mass_max": max(finite(vals("unreliable_pair_mass")), default=math.nan),
            "head_stable_pair_mass_max": max(finite(vals("stable_pair_mass")), default=math.nan),
            "head_stable_actual_minus_random_max": max(finite(vals("stable_actual_minus_random")), default=math.nan),
            "head_unreliable_actual_minus_random_max": max(finite(vals("unreliable_actual_minus_random")), default=math.nan),
        })
    return rows


def query_head_random_audit(
    metric: dict[str, Any],
    case_rows: list[dict[str, Any]],
    edge_rows: list[dict[str, Any]],
    head_field: str,
    *,
    repeats: int = 512,
) -> dict[str, Any]:
    field = str(metric.get("field", ""))
    direction = str(metric.get("direction", "higher_bad"))
    threshold = f(metric.get("threshold"))
    actual_ba = f(metric.get("balanced_accuracy"))
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    label_by_case = {str(row.get("case_id", "")): row for row in case_rows}
    for row in edge_rows:
        case_id = str(row.get("case_id", ""))
        if case_id in label_by_case and math.isfinite(f(row.get(head_field))):
            by_case[case_id].append(row)
    seed_payload = f"query_head_random|{metric.get('cue_name','')}|{field}|{head_field}|{threshold}"
    seed = int(hashlib.sha256(seed_payload.encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(seed)
    ba_vals: list[float] = []
    if not math.isfinite(threshold) or not by_case:
        metric["query_head_random_available"] = False
        return metric
    ordered_cases = sorted(by_case)
    for _ in range(repeats):
        selected_indexes: set[int] = set()
        available_rows: list[dict[str, Any]] = []
        for idx, case_id in enumerate(ordered_cases):
            label_row = label_by_case[case_id]
            head_row = rng.choice(by_case[case_id])
            score = f(head_row.get(head_field))
            available_rows.append(label_row)
            selected = score <= threshold if direction == "lower_bad" else score >= threshold
            if selected:
                selected_indexes.add(idx)
        ba_vals.append(f(metric_from_selected(available_rows, selected_indexes).get("balanced_accuracy")))
    p95 = quantile(ba_vals, 0.95)
    metric["query_head_random_available"] = True
    metric["query_head_random_applicable"] = True
    metric["query_head_random_repeats"] = repeats
    metric["query_head_random_seed"] = seed
    metric["query_head_random_BA_mean"] = mean(ba_vals)
    metric["query_head_random_BA_p95"] = p95
    metric["query_head_random_BA_max"] = max(finite(ba_vals), default=math.nan)
    metric["query_head_random_margin"] = actual_ba - p95 if math.isfinite(actual_ba) and math.isfinite(p95) else math.nan
    return metric


def build_edge_head_control_audit() -> dict[str, Any]:
    out = ROOT / "trackR_edge_head_control_audit"
    edge_rows, read_errors = collect_edge_head_rows()
    case_rows = edge_head_case_rows(edge_rows)
    pattern_specs = [
        {"cue_name": "R_head_stable_query_hit_max", "field": "head_stable_query_hit_max", "head_field": "stable_query_hit_frac", "direction": "higher_bad"},
        {"cue_name": "R_head_top1_stable_hit_max", "field": "head_top1_stable_hit_max", "head_field": "top1_stable_hit_frac", "direction": "higher_bad"},
        {"cue_name": "R_head_dominant_anchor_frac_max", "field": "head_dominant_anchor_frac_max", "head_field": "dominant_stable_anchor_frac", "direction": "higher_bad"},
        {"cue_name": "R_head_unreliable_pair_mass_max", "field": "head_unreliable_pair_mass_max", "head_field": "unreliable_pair_mass", "direction": "higher_bad"},
        {"cue_name": "R_head_feature_transport_residual_max", "field": "head_feature_transport_residual_max", "head_field": "feature_transport_residual", "direction": "higher_bad"},
        {"cue_name": "R_head_stable_actual_minus_random_max", "field": "head_stable_actual_minus_random_max", "head_field": "stable_actual_minus_random", "direction": "higher_bad"},
    ]
    metrics: list[dict[str, Any]] = []
    for spec in pattern_specs:
        metric = evaluate_pattern(case_rows, spec["cue_name"], spec["field"], spec["direction"])
        metric = attach_control_audit(metric, case_rows)
        metric = query_head_random_audit(metric, case_rows, edge_rows, spec["head_field"])
        metrics.append(with_gate(metric))
    best = sorted(
        metrics,
        key=lambda row: (
            b(row.get("gate_pass")),
            b(row.get("gate_without_controls_pass")),
            f(row.get("balanced_accuracy")),
            f(row.get("bad_recall")),
            -f(row.get("good_FPR"), 1.0),
        ),
        reverse=True,
    )
    best_row = best[0] if best else {}
    gate = any(b(row.get("gate_pass")) for row in metrics)
    sequence_coverage = len({row.get("seq") for row in case_rows if row.get("seq")})
    label_counts = Counter(str(row.get("case_label", "")) for row in case_rows)
    summary = {
        "schema": "acl2_v100_trackR_edge_head_control_audit_v1",
        "status": "complete_diagnostic_no_go" if case_rows and not gate else ("complete" if gate else "blocked_missing_inputs"),
        "gate_pass": gate,
        "runtime_action_allowed": False,
        "case_count": len(case_rows),
        "sequence_coverage": sequence_coverage,
        "edge_head_row_count": len(edge_rows),
        "trace_read_error_count": len(read_errors),
        "label_counts": dict(label_counts),
        "best_pattern": best_row.get("cue_name", ""),
        "best_balanced_accuracy": best_row.get("balanced_accuracy", math.nan),
        "best_bad_recall": best_row.get("bad_recall", math.nan),
        "best_good_FPR": best_row.get("good_FPR", math.nan),
        "best_abs_corr_L3": best_row.get("abs_corr_L3", math.nan),
        "best_same_count_random_margin": best_row.get("same_count_random_margin", math.nan),
        "best_query_head_random_margin": best_row.get("query_head_random_margin", math.nan),
        "blocker": "" if gate else "Edge/head-level audit did not pass recall/FPR/correlation/control gates; runtime action remains blocked.",
    }
    write_json(out / "summary.json", summary)
    write_rows(out / "edge_head_rows.csv", edge_rows)
    write_rows(out / "case_rows.csv", case_rows)
    write_rows(out / "pattern_metrics.csv", metrics)
    write_rows(out / "control_audit_rows.csv", control_audit_rows(metrics))
    write_rows(out / "read_errors.csv", read_errors)
    write_rows(out / "gate_checks.csv", [
        {"gate": "case_count_ge28", "pass": len(case_rows) >= 28, "value": len(case_rows)},
        {"gate": "sequence_coverage_ge4", "pass": sequence_coverage >= 4, "value": sequence_coverage},
        {"gate": "edge_head_rows_available", "pass": bool(edge_rows), "value": len(edge_rows)},
        {"gate": "any_pattern_without_controls_pass", "pass": any(b(row.get("gate_without_controls_pass")) for row in metrics)},
        {"gate": "any_complete_gate_pass", "pass": gate},
    ])
    write_text(
        out / "failure_report.md",
        "# Edge/Head Control Audit Failure Report\n\n"
        f"- gate_pass: {gate}\n"
        f"- edge_head_row_count: {len(edge_rows)}\n"
        f"- best_pattern: {summary['best_pattern']}\n"
        f"- best_balanced_accuracy: {summary['best_balanced_accuracy']}\n"
        f"- best_bad_recall: {summary['best_bad_recall']}\n"
        f"- best_good_FPR: {summary['best_good_FPR']}\n"
        f"- best_query_head_random_margin: {summary['best_query_head_random_margin']}\n",
    )
    write_text(
        out / "what_would_have_to_be_true_to_pass.md",
        "A head/edge pattern must pass recall/FPR/correlation/sequence gates and beat same-count/query-head controls before it can feed M3.\n",
    )
    write_text(out / "visual_manifest.csv", "path,description\n")
    return summary


def lifecycle_hidden_r_same(life: dict[str, Any]) -> float:
    vals = [
        f(life.get("z_write_cache_hidden_vec_residual")),
        f(life.get("z_write_current_hidden_vec_residual")),
        f(life.get("z_ref_current_hidden_vec_residual")),
    ]
    return mean(vals)


def collect_anchor_edge_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    labels = load_v99_case_labels()
    trace_root, paths, trace_source = trace_payload_paths()
    rows: list[dict[str, Any]] = []
    read_errors: list[dict[str, Any]] = []
    for path in paths:
        case_id = case_id_from_trace(path)
        if case_id not in labels:
            continue
        try:
            payload = torch_load(path)
        except Exception as exc:
            read_errors.append({"path": str(path), "error": f"{type(exc).__name__}:{exc}"})
            continue
        if not isinstance(payload, dict):
            continue
        hit = payload.get("current_Q_to_cache_K_topk_ttt_prev_stable_anchor_hit_mask")
        anchor_ids = payload.get("current_Q_to_cache_K_topk_ttt_prev_stable_anchor_ids")
        scores = payload.get("current_Q_to_cache_K_topk_scores")
        lifecycle = payload.get("ttt_prev_stable_anchor_lifecycle_rows")
        if not (torch.is_tensor(hit) and torch.is_tensor(anchor_ids) and isinstance(lifecycle, list)):
            read_errors.append({"path": str(path), "error": "missing_hit_ids_or_lifecycle"})
            continue
        hit = hit.detach().cpu().bool()
        anchor_ids = anchor_ids.detach().cpu().long()
        scores = scores.detach().cpu().float() if torch.is_tensor(scores) else None
        if hit.ndim != 4 or anchor_ids.shape != hit.shape:
            read_errors.append({"path": str(path), "error": f"bad_shape hit={tuple(hit.shape)} ids={tuple(anchor_ids.shape)}"})
            continue
        life_by_anchor: dict[int, dict[str, Any]] = {}
        for life in lifecycle:
            if not isinstance(life, dict):
                continue
            anchor_id = int(f(life.get("anchor_id"), -1))
            if anchor_id >= 0:
                life_by_anchor[anchor_id] = life
        batch, head_count, query_count, topk_count = [int(v) for v in hit.shape]
        label = labels[case_id]
        for batch_idx in range(batch):
            for head_idx in range(head_count):
                hit_h = hit[batch_idx, head_idx]
                ids_h = anchor_ids[batch_idx, head_idx]
                valid_ids = ids_h[hit_h & (ids_h >= 0)]
                if int(valid_ids.numel()) == 0:
                    continue
                for anchor_id_tensor in torch.unique(valid_ids):
                    anchor_id = int(anchor_id_tensor.item())
                    mask = hit_h & (ids_h == anchor_id)
                    if not bool(mask.any()):
                        continue
                    life = life_by_anchor.get(anchor_id, {})
                    score_mean = math.nan
                    if scores is not None and scores.shape == hit.shape:
                        score_mean = f(scores[batch_idx, head_idx][mask].float().mean().item())
                    query_hit_frac = f(mask.any(dim=-1).float().mean().item())
                    topk_hit_frac = f(mask.float().mean().item())
                    top1_hit_frac = f(mask[:, 0].float().mean().item()) if topk_count > 0 else math.nan
                    hidden_r_same = lifecycle_hidden_r_same(life)
                    current_residual = f(life.get("current_feature_residual_mean"))
                    rows.append({
                        **label,
                        "trace_source": trace_source,
                        "trace_payload": str(path),
                        "layer": payload.get("layer", ""),
                        "swa_layer_idx": payload.get("swa_layer_idx", ""),
                        "batch_idx": batch_idx,
                        "head_idx": head_idx,
                        "anchor_id": anchor_id,
                        "semantic_class": life.get("source_label_mode", ""),
                        "source_label_mode_frac": f(life.get("source_label_mode_frac")),
                        "query_count": query_count,
                        "topk_count": topk_count,
                        "anchor_topk_hit_count": int(mask.sum().item()),
                        "anchor_query_hit_frac": query_hit_frac,
                        "anchor_topk_hit_frac": topk_hit_frac,
                        "anchor_top1_hit_frac": top1_hit_frac,
                        "anchor_score_mean": score_mean,
                        "anchor_hidden_R_same": hidden_r_same,
                        "anchor_current_feature_residual": current_residual,
                        "anchor_query_x_R_same": safe_product(query_hit_frac, hidden_r_same),
                        "anchor_query_x_R_same_x_current_residual": safe_product(query_hit_frac, hidden_r_same, current_residual),
                        "source_retention_mean": f(life.get("source_retention_mean")),
                        "source_residual_mean": f(life.get("source_residual_mean")),
                        "topk_route_mass_mean": f(life.get("topk_route_mass_mean")),
                        "topk_route_mass_max": f(life.get("topk_route_mass_max")),
                    })
    return rows, read_errors


def anchor_edge_case_rows(anchor_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in anchor_rows:
        grouped[str(row.get("case_id", ""))].append(row)
    rows: list[dict[str, Any]] = []
    for case_id, parts in sorted(grouped.items()):
        if not parts:
            continue
        base = case_base(parts[0])
        def vals(field: str) -> list[float]:
            return [f(row.get(field)) for row in parts]
        rows.append({
            **base,
            "anchor_edge_row_count": len(parts),
            "anchor_query_hit_max": max(finite(vals("anchor_query_hit_frac")), default=math.nan),
            "anchor_topk_hit_max": max(finite(vals("anchor_topk_hit_frac")), default=math.nan),
            "anchor_top1_hit_max": max(finite(vals("anchor_top1_hit_frac")), default=math.nan),
            "anchor_hidden_R_same_max": max(finite(vals("anchor_hidden_R_same")), default=math.nan),
            "anchor_query_x_R_same_max": max(finite(vals("anchor_query_x_R_same")), default=math.nan),
            "anchor_query_x_R_same_x_current_residual_max": max(
                finite(vals("anchor_query_x_R_same_x_current_residual")),
                default=math.nan,
            ),
            "anchor_current_feature_residual_max": max(finite(vals("anchor_current_feature_residual")), default=math.nan),
            "anchor_score_mean_max": max(finite(vals("anchor_score_mean")), default=math.nan),
            "anchor_route_mass_max": max(finite(vals("topk_route_mass_max")), default=math.nan),
        })
    return rows


def random_edge_row_audit(
    metric: dict[str, Any],
    case_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    candidate_field: str,
    *,
    prefix: str,
    repeats: int = 512,
) -> dict[str, Any]:
    field = str(metric.get("field", ""))
    direction = str(metric.get("direction", "higher_bad"))
    threshold = f(metric.get("threshold"))
    actual_ba = f(metric.get("balanced_accuracy"))
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    label_by_case = {str(row.get("case_id", "")): row for row in case_rows}
    for row in candidate_rows:
        case_id = str(row.get("case_id", ""))
        if case_id in label_by_case and math.isfinite(f(row.get(candidate_field))):
            by_case[case_id].append(row)
    seed_payload = f"{prefix}|{metric.get('cue_name','')}|{field}|{candidate_field}|{threshold}"
    seed = int(hashlib.sha256(seed_payload.encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(seed)
    ba_vals: list[float] = []
    if not math.isfinite(threshold) or not by_case:
        metric[f"{prefix}_available"] = False
        return metric
    ordered_cases = sorted(by_case)
    for _ in range(repeats):
        selected_indexes: set[int] = set()
        available_rows: list[dict[str, Any]] = []
        for idx, case_id in enumerate(ordered_cases):
            label_row = label_by_case[case_id]
            cand = rng.choice(by_case[case_id])
            score = f(cand.get(candidate_field))
            available_rows.append(label_row)
            selected = score <= threshold if direction == "lower_bad" else score >= threshold
            if selected:
                selected_indexes.add(idx)
        ba_vals.append(f(metric_from_selected(available_rows, selected_indexes).get("balanced_accuracy")))
    p95 = quantile(ba_vals, 0.95)
    metric[f"{prefix}_available"] = True
    metric[f"{prefix}_repeats"] = repeats
    metric[f"{prefix}_seed"] = seed
    metric[f"{prefix}_BA_mean"] = mean(ba_vals)
    metric[f"{prefix}_BA_p95"] = p95
    metric[f"{prefix}_BA_max"] = max(finite(ba_vals), default=math.nan)
    metric[f"{prefix}_margin"] = actual_ba - p95 if math.isfinite(actual_ba) and math.isfinite(p95) else math.nan
    return metric


def build_anchor_edge_identity_control_audit() -> dict[str, Any]:
    out = ROOT / "trackR2_anchor_edge_identity_control_audit"
    anchor_rows, read_errors = collect_anchor_edge_rows()
    case_rows = anchor_edge_case_rows(anchor_rows)
    pattern_specs = [
        {"cue_name": "R2_anchor_query_hit_max", "field": "anchor_query_hit_max", "candidate_field": "anchor_query_hit_frac", "direction": "higher_bad"},
        {"cue_name": "R2_anchor_top1_hit_max", "field": "anchor_top1_hit_max", "candidate_field": "anchor_top1_hit_frac", "direction": "higher_bad"},
        {"cue_name": "R2_anchor_query_R_same_max", "field": "anchor_query_x_R_same_max", "candidate_field": "anchor_query_x_R_same", "direction": "higher_bad"},
        {"cue_name": "R2_anchor_query_R_current_max", "field": "anchor_query_x_R_same_x_current_residual_max", "candidate_field": "anchor_query_x_R_same_x_current_residual", "direction": "higher_bad"},
        {"cue_name": "R2_anchor_route_mass_max", "field": "anchor_route_mass_max", "candidate_field": "topk_route_mass_max", "direction": "higher_bad"},
    ]
    metrics: list[dict[str, Any]] = []
    for spec in pattern_specs:
        metric = evaluate_pattern(case_rows, spec["cue_name"], spec["field"], spec["direction"])
        metric = attach_control_audit(metric, case_rows)
        metric = random_edge_row_audit(metric, case_rows, anchor_rows, spec["candidate_field"], prefix="anchor_identity_random")
        metrics.append(with_gate(metric))
    best = sorted(
        metrics,
        key=lambda row: (
            b(row.get("gate_pass")),
            b(row.get("gate_without_controls_pass")),
            f(row.get("balanced_accuracy")),
            f(row.get("bad_recall")),
            -f(row.get("good_FPR"), 1.0),
        ),
        reverse=True,
    )
    best_row = best[0] if best else {}
    gate = any(b(row.get("gate_pass")) for row in metrics)
    sequence_coverage = len({row.get("seq") for row in case_rows if row.get("seq")})
    label_counts = Counter(str(row.get("case_label", "")) for row in case_rows)
    summary = {
        "schema": "acl2_v100_trackR2_anchor_edge_identity_control_audit_v1",
        "status": "complete_diagnostic_no_go" if case_rows and not gate else ("complete" if gate else "blocked_missing_inputs"),
        "gate_pass": gate,
        "runtime_action_allowed": False,
        "case_count": len(case_rows),
        "sequence_coverage": sequence_coverage,
        "anchor_edge_row_count": len(anchor_rows),
        "trace_read_error_count": len(read_errors),
        "label_counts": dict(label_counts),
        "best_pattern": best_row.get("cue_name", ""),
        "best_balanced_accuracy": best_row.get("balanced_accuracy", math.nan),
        "best_bad_recall": best_row.get("bad_recall", math.nan),
        "best_good_FPR": best_row.get("good_FPR", math.nan),
        "best_abs_corr_L3": best_row.get("abs_corr_L3", math.nan),
        "best_same_count_random_margin": best_row.get("same_count_random_margin", math.nan),
        "best_anchor_identity_random_margin": best_row.get("anchor_identity_random_margin", math.nan),
        "blocker": "" if gate else "Anchor-conditioned edge audit did not pass diagnostic/control gates; runtime action remains blocked.",
    }
    write_json(out / "summary.json", summary)
    write_rows(out / "anchor_edge_rows.csv", anchor_rows)
    write_rows(out / "case_rows.csv", case_rows)
    write_rows(out / "pattern_metrics.csv", metrics)
    write_rows(out / "control_audit_rows.csv", control_audit_rows(metrics))
    write_rows(out / "read_errors.csv", read_errors)
    write_rows(out / "gate_checks.csv", [
        {"gate": "case_count_ge28", "pass": len(case_rows) >= 28, "value": len(case_rows)},
        {"gate": "sequence_coverage_ge4", "pass": sequence_coverage >= 4, "value": sequence_coverage},
        {"gate": "anchor_edge_rows_available", "pass": bool(anchor_rows), "value": len(anchor_rows)},
        {"gate": "any_pattern_without_controls_pass", "pass": any(b(row.get("gate_without_controls_pass")) for row in metrics)},
        {"gate": "any_complete_gate_pass", "pass": gate},
    ])
    write_text(
        out / "failure_report.md",
        "# Anchor-Edge Identity Control Audit Failure Report\n\n"
        f"- gate_pass: {gate}\n"
        f"- anchor_edge_row_count: {len(anchor_rows)}\n"
        f"- best_pattern: {summary['best_pattern']}\n"
        f"- best_balanced_accuracy: {summary['best_balanced_accuracy']}\n"
        f"- best_bad_recall: {summary['best_bad_recall']}\n"
        f"- best_good_FPR: {summary['best_good_FPR']}\n"
        f"- best_anchor_identity_random_margin: {summary['best_anchor_identity_random_margin']}\n",
    )
    write_text(
        out / "what_would_have_to_be_true_to_pass.md",
        "An anchor-conditioned edge pattern must pass recall/FPR/correlation/sequence gates and beat same-count/anchor-random controls before M3.\n",
    )
    write_text(out / "visual_manifest.csv", "path,description\n")
    return summary


def clamp01(value: Any) -> float:
    fv = f(value)
    if not math.isfinite(fv):
        return math.nan
    return max(0.0, min(1.0, fv))


def minmax_stats(rows: list[dict[str, Any]], fields: list[str]) -> dict[str, tuple[float, float]]:
    stats: dict[str, tuple[float, float]] = {}
    for field in fields:
        vals = finite([row.get(field) for row in rows])
        stats[field] = (min(vals), max(vals)) if vals else (math.nan, math.nan)
    return stats


def norm01_from_stats(row: dict[str, Any], field: str, stats: dict[str, tuple[float, float]], *, invert: bool = False) -> float:
    lo, hi = stats.get(field, (math.nan, math.nan))
    value = f(row.get(field))
    if not (math.isfinite(value) and math.isfinite(lo) and math.isfinite(hi)):
        return math.nan
    if abs(hi - lo) <= 1.0e-12:
        out = 0.5
    else:
        out = (value - lo) / (hi - lo)
    out = clamp01(out)
    return 1.0 - out if invert and math.isfinite(out) else out


def merge_case_sources() -> list[dict[str, Any]]:
    base = {str(row.get("case_id", "")): dict(row) for row in stage3_case_rows()}
    for path in [
        ROOT / "trackR_edge_head_control_audit/case_rows.csv",
        ROOT / "trackR2_anchor_edge_identity_control_audit/case_rows.csv",
    ]:
        for row in read_rows(path):
            case_id = str(row.get("case_id", ""))
            if not case_id or case_id not in base:
                continue
            for key, value in row.items():
                if key in {"case_id", "seq", "case_label", "L3_handoff_transfer_penalty_proxy"}:
                    continue
                base[case_id][key] = value
    return [base[key] for key in sorted(base)]


def token_patch_coords(token_index: int, tokens_per_frame: int, patch_h: int, patch_w: int) -> tuple[int, int, int] | None:
    if tokens_per_frame <= 0 or patch_h <= 0 or patch_w <= 0:
        return None
    frame_idx = int(token_index) // int(tokens_per_frame)
    in_frame = int(token_index) % int(tokens_per_frame)
    patch_count = int(patch_h) * int(patch_w)
    special_count = int(tokens_per_frame) - patch_count
    if special_count < 0 or in_frame < special_count:
        return None
    patch_idx = in_frame - special_count
    if patch_idx < 0 or patch_idx >= patch_count:
        return None
    return frame_idx, patch_idx // int(patch_w), patch_idx % int(patch_w)


def point_at_patch(points: torch.Tensor, frame_idx: int, patch_row: int, patch_col: int, patch_h: int, patch_w: int) -> torch.Tensor | None:
    if not torch.is_tensor(points) or points.ndim != 4 or points.shape[-1] < 3:
        return None
    if not (0 <= frame_idx < int(points.shape[0])):
        return None
    height = int(points.shape[1])
    width = int(points.shape[2])
    y = min(height - 1, max(0, int((float(patch_row) + 0.5) * float(height) / max(float(patch_h), 1.0))))
    x = min(width - 1, max(0, int((float(patch_col) + 0.5) * float(width) / max(float(patch_w), 1.0))))
    return points[int(frame_idx), y, x, :3].detach().float()


def scalar_at_patch(values: torch.Tensor | None, frame_idx: int, patch_row: int, patch_col: int, patch_h: int, patch_w: int) -> float:
    if not torch.is_tensor(values) or values.ndim < 3:
        return math.nan
    if not (0 <= frame_idx < int(values.shape[0])):
        return math.nan
    height = int(values.shape[1])
    width = int(values.shape[2])
    y = min(height - 1, max(0, int((float(patch_row) + 0.5) * float(height) / max(float(patch_h), 1.0))))
    x = min(width - 1, max(0, int((float(patch_col) + 0.5) * float(width) / max(float(patch_w), 1.0))))
    return f(values[int(frame_idx), y, x].detach().float().item())


def geometry_sidecar_paths_for_trace(payload_path: Path, case_id: str) -> tuple[Path | None, Path | None, str]:
    run_dir = payload_path.parent.parent
    sidecar_dir = run_dir / "per_chunk_geometry"
    if not sidecar_dir.is_dir():
        return None, None, "missing_per_chunk_geometry_dir"
    seq, prev_chunk, curr_chunk = parse_case_id(case_id)
    local_delta = int(curr_chunk - prev_chunk) if prev_chunk is not None and curr_chunk is not None else 1
    prev_path = sidecar_dir / "chunk_000.pt"
    curr_path = sidecar_dir / f"chunk_{max(local_delta, 1):03d}.pt"
    if not prev_path.is_file() or not curr_path.is_file():
        sidecars = sorted(sidecar_dir.glob("chunk_*.pt"))
        if len(sidecars) >= 2:
            return sidecars[0], sidecars[-1], "fallback_sorted_sidecars"
        return None, None, "missing_expected_prev_current_sidecars"
    return prev_path, curr_path, "matched_case_delta"


def collect_geometry_sidecar_edge_rows(max_rows_per_payload: int = 8192) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    labels = load_v99_case_labels()
    trace_root, paths, trace_source = trace_payload_paths()
    rows: list[dict[str, Any]] = []
    read_errors: list[dict[str, Any]] = []
    geometry_cache: dict[str, dict[str, Any]] = {}

    def load_sidecar(path: Path) -> dict[str, Any] | None:
        key = str(path)
        if key not in geometry_cache:
            try:
                payload = torch_load(path)
                geometry_cache[key] = payload if isinstance(payload, dict) else {}
            except Exception as exc:
                read_errors.append({"path": str(path), "error": f"{type(exc).__name__}:{exc}"})
                geometry_cache[key] = {}
        return geometry_cache.get(key) or None

    for path in paths:
        case_id = case_id_from_trace(path)
        if case_id not in labels:
            continue
        prev_path, curr_path, sidecar_match = geometry_sidecar_paths_for_trace(path, case_id)
        if prev_path is None or curr_path is None:
            read_errors.append({"path": str(path), "error": sidecar_match})
            continue
        prev_geo = load_sidecar(prev_path)
        curr_geo = load_sidecar(curr_path)
        if not prev_geo or not curr_geo:
            continue
        try:
            payload = torch_load(path)
        except Exception as exc:
            read_errors.append({"path": str(path), "error": f"{type(exc).__name__}:{exc}"})
            continue
        if not isinstance(payload, dict):
            continue
        hit = payload.get("current_Q_to_cache_K_topk_ttt_prev_stable_anchor_hit_mask")
        anchor_ids = payload.get("current_Q_to_cache_K_topk_ttt_prev_stable_anchor_ids")
        cache_indices = payload.get("current_Q_to_cache_K_topk_cache_indices")
        sampled_query_indices = payload.get("sampled_query_indices")
        if not (
            torch.is_tensor(hit)
            and torch.is_tensor(anchor_ids)
            and torch.is_tensor(cache_indices)
            and torch.is_tensor(sampled_query_indices)
        ):
            read_errors.append({"path": str(path), "error": "missing_stable_hit_anchor_cache_or_query_indices"})
            continue
        hit = hit.detach().cpu().bool()
        anchor_ids = anchor_ids.detach().cpu().long()
        cache_indices = cache_indices.detach().cpu().long()
        sampled_query_indices = sampled_query_indices.detach().cpu().long()
        if hit.ndim != 4 or anchor_ids.shape != hit.shape or cache_indices.shape != hit.shape:
            read_errors.append({"path": str(path), "error": "bad_topk_tensor_shape"})
            continue
        prev_local = prev_geo.get("local_points")
        curr_local = curr_geo.get("local_points")
        prev_world = prev_geo.get("points")
        curr_world = curr_geo.get("points")
        prev_conf = prev_geo.get("conf")
        curr_conf = curr_geo.get("conf")
        prev_pose = prev_geo.get("camera_poses")
        curr_pose = curr_geo.get("camera_poses")
        if not (torch.is_tensor(prev_local) and torch.is_tensor(curr_local) and torch.is_tensor(prev_pose) and torch.is_tensor(curr_pose)):
            read_errors.append({"path": str(path), "error": "missing_geometry_tensors_in_sidecar"})
            continue
        patch_h = int(curr_local.shape[1]) // 14 if int(curr_local.shape[1]) % 14 == 0 else 19
        patch_w = int(curr_local.shape[2]) // 14 if int(curr_local.shape[2]) % 14 == 0 else 66
        # Use token_type if present to recover the exact patch count.
        token_type = curr_geo.get("token_type")
        tokens_per_frame = int(payload.get("tokens_per_frame", 0) or 0)
        if torch.is_tensor(token_type) and tokens_per_frame > 0:
            per_frame = token_type[:tokens_per_frame]
            patch_count = int((per_frame == TOKEN_TYPE_PATCH).sum().item()) if "TOKEN_TYPE_PATCH" in globals() else int((per_frame == 2).sum().item())
            if patch_count > 0:
                # KITTI LoGeR runs use a 19x66 patch grid; keep square-ish fallback for other shapes.
                if patch_count == 19 * 66:
                    patch_h, patch_w = 19, 66
        label = labels[case_id]
        batch, head_count, query_count, topk_count = [int(v) for v in hit.shape]
        emitted = 0
        for batch_idx in range(batch):
            for head_idx in range(head_count):
                for query_pos in range(query_count):
                    q_token = int(sampled_query_indices[min(query_pos, int(sampled_query_indices.numel()) - 1)].item())
                    q_coords = token_patch_coords(q_token, tokens_per_frame, patch_h, patch_w)
                    if q_coords is None:
                        continue
                    q_frame, q_pr, q_pc = q_coords
                    q_local = point_at_patch(curr_local, q_frame, q_pr, q_pc, patch_h, patch_w)
                    q_world = point_at_patch(curr_world, q_frame, q_pr, q_pc, patch_h, patch_w) if torch.is_tensor(curr_world) else None
                    if q_local is None:
                        continue
                    for topk_idx in range(topk_count):
                        if not bool(hit[batch_idx, head_idx, query_pos, topk_idx].item()):
                            continue
                        anchor_id = int(anchor_ids[batch_idx, head_idx, query_pos, topk_idx].item())
                        if anchor_id < 0:
                            continue
                        cache_token = int(cache_indices[batch_idx, head_idx, query_pos, topk_idx].item())
                        c_coords = token_patch_coords(cache_token, tokens_per_frame, patch_h, patch_w)
                        if c_coords is None:
                            continue
                        c_frame, c_pr, c_pc = c_coords
                        c_local = point_at_patch(prev_local, c_frame, c_pr, c_pc, patch_h, patch_w)
                        c_world = point_at_patch(prev_world, c_frame, c_pr, c_pc, patch_h, patch_w) if torch.is_tensor(prev_world) else None
                        if c_local is None:
                            continue
                        baseline = math.nan
                        if 0 <= q_frame < int(curr_pose.shape[0]) and 0 <= c_frame < int(prev_pose.shape[0]):
                            baseline = f((curr_pose[q_frame, :3, 3].float() - prev_pose[c_frame, :3, 3].float()).norm().item())
                        world_distance = (
                            f((q_world.float() - c_world.float()).norm().item())
                            if q_world is not None and c_world is not None
                            else math.nan
                        )
                        q_depth = f(q_local[2].item())
                        c_depth = f(c_local[2].item())
                        rows.append({
                            **label,
                            "trace_source": trace_source,
                            "trace_payload": str(path),
                            "geometry_prev_sidecar": str(prev_path),
                            "geometry_curr_sidecar": str(curr_path),
                            "geometry_sidecar_match": sidecar_match,
                            "layer": payload.get("layer", ""),
                            "swa_layer_idx": payload.get("swa_layer_idx", ""),
                            "head_idx": head_idx,
                            "anchor_id": anchor_id,
                            "query_token": q_token,
                            "cache_token": cache_token,
                            "query_frame": q_frame,
                            "cache_frame": c_frame,
                            "query_patch_row": q_pr,
                            "query_patch_col": q_pc,
                            "cache_patch_row": c_pr,
                            "cache_patch_col": c_pc,
                            "query_depth": q_depth,
                            "cache_depth": c_depth,
                            "query_conf": scalar_at_patch(curr_conf, q_frame, q_pr, q_pc, patch_h, patch_w),
                            "cache_conf": scalar_at_patch(prev_conf, c_frame, c_pr, c_pc, patch_h, patch_w),
                            "camera_translation_baseline": baseline,
                            "world_pair_distance": world_distance,
                            "abs_log_depth_ratio": abs(math.log(max(q_depth, 1.0e-6) / max(c_depth, 1.0e-6)))
                            if q_depth > 0.0 and c_depth > 0.0 else math.nan,
                            "abs_depth_diff": abs(q_depth - c_depth) if math.isfinite(q_depth) and math.isfinite(c_depth) else math.nan,
                        })
                        emitted += 1
                        if emitted >= max_rows_per_payload:
                            break
                    if emitted >= max_rows_per_payload:
                        break
                if emitted >= max_rows_per_payload:
                    break
            if emitted >= max_rows_per_payload:
                break
    metadata = {
        "trace_root": str(trace_root),
        "trace_source": trace_source,
        "geometry_edge_row_count": len(rows),
        "geometry_read_error_count": len(read_errors),
        "geometry_sidecar_available": bool(rows),
    }
    return rows, read_errors, metadata


def geometry_sidecar_case_rows(edge_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in edge_rows:
        grouped[str(row.get("case_id", ""))].append(row)
    rows: list[dict[str, Any]] = []
    for case_id, parts in sorted(grouped.items()):
        if not parts:
            continue
        base = case_base(parts[0])
        def vals(field: str) -> list[float]:
            return [f(row.get(field)) for row in parts]
        rows.append({
            **base,
            "geometry_edge_row_count": len(parts),
            "geometry_layer_count": len({str(row.get("swa_layer_idx", "")) for row in parts}),
            "geometry_anchor_count": len({str(row.get("anchor_id", "")) for row in parts}),
            "geometry_camera_baseline_mean": mean(vals("camera_translation_baseline")),
            "geometry_camera_baseline_max": max(finite(vals("camera_translation_baseline")), default=math.nan),
            "geometry_world_pair_distance_mean": mean(vals("world_pair_distance")),
            "geometry_world_pair_distance_max": max(finite(vals("world_pair_distance")), default=math.nan),
            "geometry_cache_depth_mean": mean(vals("cache_depth")),
            "geometry_current_depth_mean": mean(vals("query_depth")),
            "geometry_cache_depth_spread": quantile(vals("cache_depth"), 0.90) - quantile(vals("cache_depth"), 0.10),
            "geometry_current_depth_spread": quantile(vals("query_depth"), 0.90) - quantile(vals("query_depth"), 0.10),
            "geometry_abs_log_depth_ratio_mean": mean(vals("abs_log_depth_ratio")),
            "geometry_abs_log_depth_ratio_max": max(finite(vals("abs_log_depth_ratio")), default=math.nan),
            "geometry_abs_depth_diff_mean": mean(vals("abs_depth_diff")),
            "geometry_query_conf_mean": mean(vals("query_conf")),
            "geometry_cache_conf_mean": mean(vals("cache_conf")),
            "geometry_current_support_proxy": mean([mean(vals("query_conf")), mean(vals("cache_conf"))]),
            "geometry_parallax_depth_score": safe_product(
                mean(vals("camera_translation_baseline")),
                quantile(vals("cache_depth"), 0.90) - quantile(vals("cache_depth"), 0.10),
            ),
        })
    return rows


def anchor_role_proxy_rows(anchor_rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if not anchor_rows:
        return [], [], {"anchor_role_proxy_available": False}
    thresholds = {
        "query_q25": quantile([f(row.get("anchor_query_hit_frac")) for row in anchor_rows], 0.25),
        "query_q75": quantile([f(row.get("anchor_query_hit_frac")) for row in anchor_rows], 0.75),
        "r_median": quantile([f(row.get("anchor_hidden_R_same")) for row in anchor_rows], 0.50),
        "r_q75": quantile([f(row.get("anchor_hidden_R_same")) for row in anchor_rows], 0.75),
        "current_median": quantile([f(row.get("anchor_current_feature_residual")) for row in anchor_rows], 0.50),
        "current_q75": quantile([f(row.get("anchor_current_feature_residual")) for row in anchor_rows], 0.75),
        "route_median": quantile([f(row.get("topk_route_mass_max")) for row in anchor_rows], 0.50),
        "route_q75": quantile([f(row.get("topk_route_mass_max")) for row in anchor_rows], 0.75),
        "label_frac_q75": quantile([f(row.get("source_label_mode_frac")) for row in anchor_rows], 0.75),
        "retention_median": quantile([f(row.get("source_retention_mean")) for row in anchor_rows], 0.50),
        "source_residual_q75": quantile([f(row.get("source_residual_mean")) for row in anchor_rows], 0.75),
    }
    role_rows: list[dict[str, Any]] = []
    grouped: dict[str, list[str]] = defaultdict(list)
    for row in anchor_rows:
        qhit = f(row.get("anchor_query_hit_frac"))
        rsame = f(row.get("anchor_hidden_R_same"))
        current = f(row.get("anchor_current_feature_residual"))
        route = f(row.get("topk_route_mass_max"))
        label_frac = f(row.get("source_label_mode_frac"))
        retention = f(row.get("source_retention_mean"))
        source_residual = f(row.get("source_residual_mean"))
        if (
            math.isfinite(qhit)
            and qhit >= thresholds["query_q75"]
            and math.isfinite(route)
            and route >= thresholds["route_q75"]
            and math.isfinite(label_frac)
            and label_frac >= thresholds["label_frac_q75"]
            and math.isfinite(current)
            and current <= thresholds["current_median"]
        ):
            role = "landmark_proxy"
        elif (
            math.isfinite(qhit)
            and qhit >= thresholds["query_q75"]
            and (
                (math.isfinite(rsame) and rsame >= thresholds["r_q75"])
                or (math.isfinite(current) and current >= thresholds["current_q75"])
            )
        ):
            role = "stale_candidate_proxy"
        elif math.isfinite(qhit) and qhit >= thresholds["query_q75"] and math.isfinite(current) and current < thresholds["current_q75"]:
            role = "local_recent_proxy"
        elif (
            math.isfinite(qhit)
            and qhit <= thresholds["query_q25"]
            and math.isfinite(retention)
            and retention >= thresholds["retention_median"]
        ):
            role = "global_reference_proxy"
        elif (
            math.isfinite(route)
            and route >= thresholds["route_median"]
            and math.isfinite(qhit)
            and thresholds["query_q25"] < qhit < thresholds["query_q75"]
        ):
            role = "camera_gauge_proxy"
        elif (
            (math.isfinite(current) and current >= thresholds["current_q75"])
            or (math.isfinite(source_residual) and source_residual >= thresholds["source_residual_q75"])
        ):
            role = "unstable_context_proxy"
        else:
            role = "unstable_context_proxy"
        case_id = str(row.get("case_id", ""))
        grouped[case_id].append(role)
        role_rows.append({
            "case_id": case_id,
            "seq": row.get("seq", ""),
            "case_label": row.get("case_label", ""),
            "anchor_id": row.get("anchor_id", ""),
            "head_idx": row.get("head_idx", ""),
            "semantic_class": row.get("semantic_class", ""),
            "anchor_role_proxy": role,
            "anchor_query_hit_frac": row.get("anchor_query_hit_frac", ""),
            "anchor_hidden_R_same": row.get("anchor_hidden_R_same", ""),
            "anchor_current_feature_residual": row.get("anchor_current_feature_residual", ""),
            "topk_route_mass_max": row.get("topk_route_mass_max", ""),
            "source_label_mode_frac": row.get("source_label_mode_frac", ""),
            "note": "role is assigned on anchor-edge rows by quantile proxy; not a physical unique-anchor role proof",
        })
    case_rows: list[dict[str, Any]] = []
    role_names = [
        "global_reference_proxy",
        "local_recent_proxy",
        "landmark_proxy",
        "camera_gauge_proxy",
        "unstable_context_proxy",
        "stale_candidate_proxy",
    ]
    for case_id, roles in sorted(grouped.items()):
        counts = Counter(roles)
        total = len(roles)
        row: dict[str, Any] = {"case_id": case_id, "anchor_role_proxy_row_count": total}
        for role in role_names:
            row[f"{role}_frac"] = counts.get(role, 0) / total if total else math.nan
        case_rows.append(row)
    metadata = {
        "anchor_role_proxy_available": True,
        "anchor_role_proxy_thresholds": thresholds,
        "anchor_role_proxy_row_count": len(role_rows),
        "anchor_role_proxy_unit": "anchor-edge row, not unique physical anchor",
    }
    return role_rows, case_rows, metadata


def v98_region_track_l_baseline() -> dict[str, Any]:
    path = V98_ROOT / "stage2_trackL_semantic_scale_observability/cue_control_metrics.csv"
    rows = read_rows(path)
    best: dict[str, Any] = {}
    for row in rows:
        if str(row.get("view_name", "raw")) not in {"raw", ""}:
            continue
        if not best or f(row.get("balanced_accuracy")) > f(best.get("balanced_accuracy")):
            best = row
    return {
        "baseline_path": str(path),
        "baseline_available": bool(best),
        "baseline_best_cue": best.get("cue_name", ""),
        "baseline_best_balanced_accuracy": f(best.get("balanced_accuracy")),
        "baseline_gate_pass": b(read_json(V98_ROOT / "stage2_trackL_semantic_scale_observability/summary.json").get("gate_pass")),
    }


def build_l2_proxy_case_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    rows = merge_case_sources()
    anchor_rows = read_rows(ROOT / "trackR2_anchor_edge_identity_control_audit/anchor_edge_rows.csv")
    role_rows, role_case_rows, role_meta = anchor_role_proxy_rows(anchor_rows)
    roles_by_case = {str(row.get("case_id", "")): row for row in role_case_rows}
    for row in rows:
        role_row = roles_by_case.get(str(row.get("case_id", "")), {})
        for key, value in role_row.items():
            if key != "case_id":
                row[key] = value
    geometry_edge_rows, geometry_read_errors, geometry_meta = collect_geometry_sidecar_edge_rows()
    geometry_case = {str(row.get("case_id", "")): row for row in geometry_sidecar_case_rows(geometry_edge_rows)}
    for row in rows:
        geom_row = geometry_case.get(str(row.get("case_id", "")), {})
        for key, value in geom_row.items():
            if key != "case_id":
                row[key] = value
    stats = minmax_stats(rows, [
        "dominant_semantic_frac",
        "head_stable_pair_mass_max",
        "anchor_route_mass_max",
        "anchor_query_hit_max",
        "head_topk_query_frame_hit_max",
        "head_top1_abs_frame_delta_mean_max",
        "head_top1_cache_frame_switch_rate_max",
        "head_top1_cache_frame_unique_frac_max",
        "current_feature_residual_mean",
        "anchor_current_feature_residual_max",
        "head_unreliable_pair_mass_max",
        "head_feature_transport_residual_max",
        "R_same_mean",
        "anchor_hidden_R_same_max",
        "anchor_query_x_R_same_x_current_residual_max",
        "head_stable_actual_minus_random_max",
        "geometry_camera_baseline_mean",
        "geometry_camera_baseline_max",
        "geometry_world_pair_distance_mean",
        "geometry_cache_depth_spread",
        "geometry_current_depth_spread",
        "geometry_abs_log_depth_ratio_mean",
        "geometry_abs_log_depth_ratio_max",
        "geometry_current_support_proxy",
        "geometry_parallax_depth_score",
    ])
    for row in rows:
        structure_support = mean([
            norm01_from_stats(row, "dominant_semantic_frac", stats),
            norm01_from_stats(row, "head_stable_pair_mass_max", stats),
            norm01_from_stats(row, "anchor_route_mass_max", stats),
        ])
        current_support = mean([
            norm01_from_stats(row, "anchor_query_hit_max", stats),
            norm01_from_stats(row, "head_topk_query_frame_hit_max", stats),
            norm01_from_stats(row, "current_feature_residual_mean", stats, invert=True),
            norm01_from_stats(row, "anchor_current_feature_residual_max", stats, invert=True),
        ])
        temporal_parallax_proxy = mean([
            norm01_from_stats(row, "head_top1_abs_frame_delta_mean_max", stats),
            norm01_from_stats(row, "head_top1_cache_frame_switch_rate_max", stats),
            norm01_from_stats(row, "head_top1_cache_frame_unique_frac_max", stats),
        ])
        geometry_available = math.isfinite(f(row.get("geometry_camera_baseline_mean"))) and math.isfinite(
            f(row.get("geometry_cache_depth_spread"))
        )
        geometry_parallax_proxy = mean([
            norm01_from_stats(row, "geometry_camera_baseline_mean", stats),
            norm01_from_stats(row, "geometry_camera_baseline_max", stats),
            norm01_from_stats(row, "geometry_world_pair_distance_mean", stats),
        ]) if geometry_available else math.nan
        geometry_depth_spread_proxy = mean([
            norm01_from_stats(row, "geometry_cache_depth_spread", stats),
            norm01_from_stats(row, "geometry_current_depth_spread", stats),
        ]) if geometry_available else math.nan
        geometry_support_proxy = norm01_from_stats(row, "geometry_current_support_proxy", stats) if geometry_available else math.nan
        if geometry_available:
            current_support = mean([current_support, geometry_support_proxy])
            effective_parallax_proxy = geometry_parallax_proxy
            effective_depth_spread_proxy = geometry_depth_spread_proxy
        else:
            effective_parallax_proxy = temporal_parallax_proxy
            effective_depth_spread_proxy = math.nan
        dynamic_risk = norm01_from_stats(row, "head_unreliable_pair_mass_max", stats)
        boundary_risk = norm01_from_stats(row, "head_feature_transport_residual_max", stats)
        same_space_inconsistency_risk = mean([
            norm01_from_stats(row, "R_same_mean", stats),
            norm01_from_stats(row, "anchor_hidden_R_same_max", stats),
            norm01_from_stats(row, "anchor_query_x_R_same_x_current_residual_max", stats),
        ])
        stable_route_advantage = norm01_from_stats(row, "head_stable_actual_minus_random_max", stats)
        low_observability_terms = [1.0 - current_support, 1.0 - effective_parallax_proxy, dynamic_risk, boundary_risk]
        if geometry_available and math.isfinite(effective_depth_spread_proxy):
            low_observability_terms.append(1.0 - effective_depth_spread_proxy)
        low_observability_risk = mean(low_observability_terms)
        stale_role = f(row.get("stale_candidate_proxy_frac"), 0.0)
        unstable_role = f(row.get("unstable_context_proxy_frac"), 0.0)
        scale_observability_score = (
            0.30 * structure_support
            + 0.30 * current_support
            + 0.15 * effective_parallax_proxy
            + (0.10 * effective_depth_spread_proxy if geometry_available and math.isfinite(effective_depth_spread_proxy) else 0.0)
            + 0.15 * stable_route_advantage
            - 0.20 * low_observability_risk
            - 0.15 * dynamic_risk
            - 0.15 * boundary_risk
            - 0.15 * same_space_inconsistency_risk
            - 0.10 * stale_role
        )
        row.update({
            "structure_support": structure_support,
            "parallax_proxy": effective_parallax_proxy,
            "parallax_proxy_available": True,
            "parallax_proxy_is_temporal_frame_delta_only": not geometry_available,
            "geometry_sidecar_terms_available": geometry_available,
            "geometry_parallax_proxy": geometry_parallax_proxy,
            "geometry_depth_spread_proxy": geometry_depth_spread_proxy,
            "geometry_support_proxy_norm": geometry_support_proxy,
            "depth_spread_proxy": effective_depth_spread_proxy,
            "depth_spread_proxy_available": geometry_available and math.isfinite(effective_depth_spread_proxy),
            "current_support": current_support,
            "current_support_is_proxy": True,
            "low_observability_risk": low_observability_risk,
            "low_temporal_parallax_risk": 1.0 - temporal_parallax_proxy,
            "low_geometry_parallax_risk": 1.0 - geometry_parallax_proxy if geometry_available else math.nan,
            "low_geometry_depth_spread_risk": 1.0 - geometry_depth_spread_proxy if geometry_available else math.nan,
            "dynamic_risk": dynamic_risk,
            "boundary_risk": boundary_risk,
            "same_space_inconsistency_risk_proxy": same_space_inconsistency_risk,
            "stable_route_advantage_proxy": stable_route_advantage,
            "temporal_parallax_x_current_support": safe_product(temporal_parallax_proxy, current_support),
            "temporal_parallax_x_same_space_risk": safe_product(temporal_parallax_proxy, same_space_inconsistency_risk),
            "geometry_parallax_x_current_support": safe_product(geometry_parallax_proxy, current_support),
            "geometry_parallax_depth_x_current_support": safe_product(
                geometry_parallax_proxy,
                geometry_depth_spread_proxy,
                current_support,
            ),
            "geometry_depth_ratio_risk": norm01_from_stats(row, "geometry_abs_log_depth_ratio_mean", stats) if geometry_available else math.nan,
            "scale_observability_score": scale_observability_score,
            "no_scale_evidence_proxy": low_observability_risk + same_space_inconsistency_risk + stale_role + unstable_role - scale_observability_score,
            "stale_or_unstable_role_frac": stale_role + unstable_role,
            "scale_observability_score_is_proxy": True,
        })
    metadata = {
        **role_meta,
        **geometry_meta,
        "geometry_case_count": len(geometry_case),
        "proxy_terms_note": (
            "L2 score uses same-space/current/role proxies. If per-chunk geometry sidecars are available, "
            "camera baseline and point-depth spread are used as geometry parallax/depth proxies; otherwise "
            "the builder falls back to temporal frame-delta proxy."
        ),
        "normalization": "per-run minmax over 28 case rows for bounded proxy fields",
    }
    return rows, role_rows, metadata, geometry_edge_rows, geometry_read_errors


def build_q_proxy_case_rows(l2_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = [dict(row) for row in l2_rows]
    stats = minmax_stats(rows, ["scale_observability_score", "no_scale_evidence_proxy"])
    current_median = quantile([f(row.get("current_support")) for row in rows], 0.50)
    lowobs_median = quantile([f(row.get("low_observability_risk")) for row in rows], 0.50)
    same_median = quantile([f(row.get("same_space_inconsistency_risk_proxy")) for row in rows], 0.50)
    stale_median = quantile([f(row.get("stale_candidate_proxy_frac")) for row in rows], 0.50)
    block_scores: list[float] = []
    allow_scores: list[float] = []
    for row in rows:
        fresh_supported = mean([
            f(row.get("current_support")),
            f(row.get("structure_support")),
            f(row.get("landmark_proxy_frac"), 0.0) + f(row.get("local_recent_proxy_frac"), 0.0),
        ])
        stale = f(row.get("stale_candidate_proxy_frac"), 0.0)
        block_score = (
            0.35 * f(row.get("low_observability_risk"))
            + 0.25 * f(row.get("same_space_inconsistency_risk_proxy"))
            + 0.20 * stale
            + 0.10 * f(row.get("dynamic_risk"))
            + 0.10 * f(row.get("boundary_risk"))
            - 0.20 * fresh_supported
        )
        allow_score = (
            0.35 * norm01_from_stats(row, "scale_observability_score", stats)
            + 0.30 * fresh_supported
            - 0.20 * f(row.get("same_space_inconsistency_risk_proxy"))
            - 0.15 * f(row.get("low_observability_risk"))
        )
        row["fresh_supported_anchor_count_proxy"] = f(row.get("anchor_role_proxy_row_count"), 0.0) * max(
            f(row.get("landmark_proxy_frac"), 0.0) + f(row.get("local_recent_proxy_frac"), 0.0),
            0.0,
        )
        row["fresh_supported_score_proxy"] = fresh_supported
        row["stale_anchor_score_proxy"] = stale
        row["admission_block_score_proxy"] = block_score
        row["admission_allow_score_proxy"] = allow_score
        block_scores.append(block_score)
        allow_scores.append(allow_score)
    block_q75 = quantile(block_scores, 0.75)
    block_median = quantile(block_scores, 0.50)
    allow_q75 = quantile(allow_scores, 0.75)
    for row in rows:
        if f(row.get("current_support")) < current_median and f(row.get("low_observability_risk")) >= lowobs_median:
            decision = "NO_SCALE_EVIDENCE"
        elif (
            f(row.get("admission_block_score_proxy")) >= block_q75
            or (
                f(row.get("stale_candidate_proxy_frac"), 0.0) >= stale_median
                and f(row.get("same_space_inconsistency_risk_proxy")) >= same_median
            )
        ):
            decision = "DELAY_UPDATE"
        elif f(row.get("admission_allow_score_proxy")) >= allow_q75 and f(row.get("admission_block_score_proxy")) <= block_median:
            decision = "ALLOW_UPDATE"
        else:
            decision = "CONTEXT_ONLY"
        row["admission_decision_proxy"] = decision
        row["delay_or_no_scale_proxy"] = decision in {"DELAY_UPDATE", "NO_SCALE_EVIDENCE"}
    metadata = {
        "decision_thresholds": {
            "current_support_median": current_median,
            "low_observability_risk_median": lowobs_median,
            "same_space_inconsistency_median": same_median,
            "stale_candidate_frac_median": stale_median,
            "admission_block_score_q75": block_q75,
            "admission_block_score_median": block_median,
            "admission_allow_score_q75": allow_q75,
        },
        "proxy_terms_note": "Q admission decisions are proxy categories from L2/R/R2/C4 diagnostics; no runtime update was executed.",
    }
    return rows, metadata


def or_rule_selected(row: dict[str, Any], rule: dict[str, Any]) -> bool:
    lhs = f(row.get(str(rule.get("field_a", ""))))
    rhs = f(row.get(str(rule.get("field_b", ""))))
    ta = f(rule.get("threshold_a"))
    tb = f(rule.get("threshold_b"))
    if not (math.isfinite(lhs) and math.isfinite(rhs) and math.isfinite(ta) and math.isfinite(tb)):
        return False
    left = lhs <= ta if rule.get("direction_a") == "lower_bad" else lhs >= ta
    right = rhs <= tb if rule.get("direction_b") == "lower_bad" else rhs >= tb
    return bool(left or right)


def evaluate_or_rule(
    case_rows: list[dict[str, Any]],
    field_a: str,
    direction_a: str,
    threshold_a: float,
    field_b: str,
    direction_b: str,
    threshold_b: float,
) -> dict[str, Any]:
    available = [
        row for row in case_rows
        if row.get("case_label") in {"good", "non_good"}
        and math.isfinite(f(row.get(field_a)))
        and math.isfinite(f(row.get(field_b)))
    ]
    rule = {
        "field_a": field_a,
        "direction_a": direction_a,
        "threshold_a": threshold_a,
        "field_b": field_b,
        "direction_b": direction_b,
        "threshold_b": threshold_b,
    }
    selected = [row for row in available if or_rule_selected(row, rule)]
    bad = [row for row in available if row.get("case_label") == "non_good"]
    good = [row for row in available if row.get("case_label") == "good"]
    tp = [row for row in selected if row.get("case_label") == "non_good"]
    fp = [row for row in selected if row.get("case_label") == "good"]
    missed = [row for row in bad if row not in tp]
    recall = len(tp) / len(bad) if bad else math.nan
    fpr = len(fp) / len(good) if good else math.nan
    ba = (recall + (1.0 - fpr)) / 2.0 if math.isfinite(recall) and math.isfinite(fpr) else math.nan
    seq_counts = Counter(str(row.get("seq", "")) for row in tp if row.get("seq"))
    selected_binary = [1.0 if row in selected else 0.0 for row in available]
    corr = pearson(selected_binary, [row.get("L3_handoff_transfer_penalty_proxy") for row in available])
    return {
        **rule,
        "cue_name": f"OR({field_a}_{direction_a},{field_b}_{direction_b})",
        "available_case_count": len(available),
        "positive_case_count": len(bad),
        "good_control_case_count": len(good),
        "balanced_accuracy": ba,
        "bad_recall": recall,
        "good_FPR": fpr,
        "abs_corr_L3": abs(corr) if math.isfinite(corr) else math.nan,
        "corr_L3": corr,
        "selected_case_count": len(selected),
        "selected_positive_sequence_coverage": len(seq_counts),
        "selected_positive_sequence_max_frac": max(seq_counts.values()) / len(tp) if tp and seq_counts else math.nan,
        "true_positive_cases": ";".join(str(row.get("case_id")) for row in tp),
        "false_positive_cases": ";".join(str(row.get("case_id")) for row in fp),
        "missed_positive_cases": ";".join(str(row.get("case_id")) for row in missed),
        "q_gate_without_true_terms_pass": (
            f(recall) >= 0.65
            and f(fpr, 1.0) <= 0.25
            and len(seq_counts) >= 4
        ),
    }


def q_or_rule_search_rows(case_rows: list[dict[str, Any]], specs: list[dict[str, str]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    all_rows: list[dict[str, Any]] = []
    best_overall: dict[str, Any] = {}
    for idx, lhs_spec in enumerate(specs):
        for rhs_spec in specs[idx + 1:]:
            field_a = lhs_spec["field"]
            field_b = rhs_spec["field"]
            direction_a = lhs_spec["direction"]
            direction_b = rhs_spec["direction"]
            vals_a = sorted({f(row.get(field_a)) for row in case_rows if math.isfinite(f(row.get(field_a)))})
            vals_b = sorted({f(row.get(field_b)) for row in case_rows if math.isfinite(f(row.get(field_b)))})
            best_pair: dict[str, Any] = {}
            feasible_pair_rows: list[dict[str, Any]] = []
            for threshold_a in vals_a:
                for threshold_b in vals_b:
                    metric = evaluate_or_rule(
                        case_rows,
                        field_a,
                        direction_a,
                        threshold_a,
                        field_b,
                        direction_b,
                        threshold_b,
                    )
                    if b(metric.get("q_gate_without_true_terms_pass")):
                        feasible_pair_rows.append({**metric, "row_kind": "q_gate_feasible_or_rule"})
                    if not best_pair or (
                        b(metric.get("q_gate_without_true_terms_pass")),
                        f(metric.get("balanced_accuracy")),
                        f(metric.get("bad_recall")),
                        -f(metric.get("good_FPR"), 1.0),
                    ) > (
                        b(best_pair.get("q_gate_without_true_terms_pass")),
                        f(best_pair.get("balanced_accuracy")),
                        f(best_pair.get("bad_recall")),
                        -f(best_pair.get("good_FPR"), 1.0),
                    ):
                        best_pair = metric
            if best_pair:
                all_rows.append({**best_pair, "row_kind": "best_pair_or_rule", "feasible_or_rule_count": len(feasible_pair_rows)})
            all_rows.extend(feasible_pair_rows)
            if best_pair and (
                not best_overall
                or (
                    b(best_pair.get("q_gate_without_true_terms_pass")),
                    f(best_pair.get("balanced_accuracy")),
                    f(best_pair.get("bad_recall")),
                    -f(best_pair.get("good_FPR"), 1.0),
                ) > (
                    b(best_overall.get("q_gate_without_true_terms_pass")),
                    f(best_overall.get("balanced_accuracy")),
                    f(best_overall.get("bad_recall")),
                    -f(best_overall.get("good_FPR"), 1.0),
                )
            ):
                best_overall = best_pair
    return best_overall, all_rows


def build_case_metric_track(
    track_dir: str,
    label: str,
    case_rows: list[dict[str, Any]],
    patterns: list[dict[str, str]],
    *,
    schema_suffix: str,
    proxy_note: str,
    missing_true_terms: list[str],
    extra_summary: dict[str, Any] | None = None,
    extra_rows: dict[str, list[dict[str, Any]]] | None = None,
    failure_filename: str = "failure_report.md",
) -> dict[str, Any]:
    out = ROOT / track_dir
    metrics: list[dict[str, Any]] = []
    for pattern in patterns:
        metric = evaluate_pattern(case_rows, pattern["cue_name"], pattern["field"], pattern["direction"])
        metric = attach_control_audit(metric, case_rows)
        metric = with_gate(metric)
        metric["proxy_only"] = True
        metric["missing_true_terms"] = ";".join(missing_true_terms)
        metric["gate_pass"] = False
        metrics.append(metric)
    best = sorted(
        metrics,
        key=lambda row: (
            b(row.get("gate_without_controls_pass")),
            f(row.get("balanced_accuracy")),
            f(row.get("bad_recall")),
            -f(row.get("good_FPR"), 1.0),
        ),
        reverse=True,
    )
    best_row = best[0] if best else {}
    label_counts = Counter(str(row.get("case_label", "")) for row in case_rows)
    sequence_coverage = len({row.get("seq") for row in case_rows if row.get("seq")})
    summary = {
        "schema": f"acl2_v100_{track_dir}_{schema_suffix}",
        "status": "complete_proxy_diagnostic_no_go" if case_rows else "blocked_missing_inputs",
        "gate_pass": False,
        "runtime_action_allowed": False,
        "proxy_only": True,
        "case_count": len(case_rows),
        "sequence_coverage": sequence_coverage,
        "label_counts": dict(label_counts),
        "best_pattern": best_row.get("cue_name", ""),
        "best_balanced_accuracy": best_row.get("balanced_accuracy", math.nan),
        "best_bad_recall": best_row.get("bad_recall", math.nan),
        "best_good_FPR": best_row.get("good_FPR", math.nan),
        "best_abs_corr_L3": best_row.get("abs_corr_L3", math.nan),
        "best_same_count_random_margin": best_row.get("same_count_random_margin", math.nan),
        "missing_true_terms": missing_true_terms,
        "blocker": (
            f"{label} generated proxy diagnostics but cannot pass: "
            f"missing true terms/controls = {', '.join(missing_true_terms)}."
        ),
        "proxy_note": proxy_note,
    }
    if extra_summary:
        summary.update(extra_summary)
    write_json(out / "summary.json", summary)
    write_rows(out / "rows.csv", case_rows)
    write_rows(out / "pattern_metrics.csv", metrics)
    write_rows(out / "control_audit_rows.csv", control_audit_rows(metrics))
    write_rows(out / "threshold_feasibility.csv", threshold_feasibility_rows(case_rows, patterns))
    write_rows(out / "sequence_loso_metrics.csv", sequence_loso_rows(case_rows, patterns))
    write_rows(out / "sequence_split_rows.csv", sequence_split_rows(case_rows, [spec["field"] for spec in patterns]))
    if any(row.get("dominant_semantic_class") for row in case_rows):
        write_rows(out / "semantic_split_rows.csv", grouped_split_rows(case_rows, [spec["field"] for spec in patterns], "dominant_semantic_class"))
        write_rows(out / "semantic_regime_metrics.csv", semantic_regime_metric_rows(case_rows, patterns))
    write_rows(out / "false_positive_missed_case_rows.csv", selection_audit_rows(case_rows, best_row))
    write_rows(out / "gate_checks.csv", [
        {"gate": "case_count_ge28", "pass": len(case_rows) >= 28, "value": len(case_rows)},
        {"gate": "sequence_coverage_ge4", "pass": sequence_coverage >= 4, "value": sequence_coverage},
        {"gate": "any_threshold_gate_without_controls_pass", "pass": any(b(row.get("gate_without_controls_pass")) for row in metrics)},
        {"gate": "same_count_controls_available", "pass": True},
        {"gate": "required_true_terms_available", "pass": False, "value": ";".join(missing_true_terms)},
        {"gate": "complete_gate_pass", "pass": False},
    ])
    if extra_rows:
        for filename, rows in extra_rows.items():
            write_rows(out / filename, rows)
    write_json(out / "missing_prereq.json", {
        "missing_true_terms": missing_true_terms,
        "proxy_note": proxy_note,
    })
    write_text(
        out / failure_filename,
        f"# {label} Failure Report\n\n"
        f"- gate_pass: false\n"
        f"- proxy_only: true\n"
        f"- case_count: {len(case_rows)}\n"
        f"- sequence_coverage: {sequence_coverage}\n"
        f"- best_pattern: {summary['best_pattern']}\n"
        f"- best_balanced_accuracy: {summary['best_balanced_accuracy']}\n"
        f"- best_bad_recall: {summary['best_bad_recall']}\n"
        f"- best_good_FPR: {summary['best_good_FPR']}\n"
        f"- best_abs_corr_L3: {summary['best_abs_corr_L3']}\n"
        f"- missing_true_terms: {', '.join(missing_true_terms)}\n"
        f"- note: {proxy_note}\n",
    )
    write_text(
        out / "control_gap_report.md",
        "# Control Gap Report\n\n"
        "- These rows are case-level proxy aggregations over existing C4/R/R2 diagnostics.\n"
        "- same_count_random_margin is materialized, but anchor-id, semantic-label, query-head, parallax and depth-spread controls are not true reruns.\n"
        "- Runtime action remains blocked even if a scalar threshold looks promising.\n",
    )
    write_text(
        out / "what_would_have_to_be_true_to_pass.md",
        "The proxy cue must be replaced or confirmed by true current_support, parallax/depth-spread and required control reruns before M3/E4 action.\n",
    )
    write_text(out / "visual_manifest.csv", "path,description\n")
    return summary


def build_l2_proxy_track() -> dict[str, Any]:
    case_rows, role_rows, metadata, geometry_edge_rows, geometry_read_errors = build_l2_proxy_case_rows()
    baseline = v98_region_track_l_baseline()
    patterns = [
        {"cue_name": "L2_no_scale_evidence_proxy", "field": "no_scale_evidence_proxy", "direction": "higher_bad"},
        {"cue_name": "L2_low_observability_risk", "field": "low_observability_risk", "direction": "higher_bad"},
        {"cue_name": "L2_low_temporal_parallax_risk", "field": "low_temporal_parallax_risk", "direction": "higher_bad"},
        {"cue_name": "L2_temporal_parallax_current_low_bad", "field": "temporal_parallax_x_current_support", "direction": "lower_bad"},
        {"cue_name": "L2_temporal_parallax_same_risk", "field": "temporal_parallax_x_same_space_risk", "direction": "higher_bad"},
        {"cue_name": "L2_low_geometry_parallax_risk", "field": "low_geometry_parallax_risk", "direction": "higher_bad"},
        {"cue_name": "L2_low_geometry_depth_spread_risk", "field": "low_geometry_depth_spread_risk", "direction": "higher_bad"},
        {"cue_name": "L2_geometry_parallax_current_low_bad", "field": "geometry_parallax_x_current_support", "direction": "lower_bad"},
        {"cue_name": "L2_geometry_parallax_depth_current_low_bad", "field": "geometry_parallax_depth_x_current_support", "direction": "lower_bad"},
        {"cue_name": "L2_geometry_depth_ratio_risk", "field": "geometry_depth_ratio_risk", "direction": "higher_bad"},
        {"cue_name": "L2_scale_observability_score_low_bad", "field": "scale_observability_score", "direction": "lower_bad"},
        {"cue_name": "L2_same_space_inconsistency_risk", "field": "same_space_inconsistency_risk_proxy", "direction": "higher_bad"},
        {"cue_name": "L2_stale_or_unstable_role_frac", "field": "stale_or_unstable_role_frac", "direction": "higher_bad"},
    ]
    provisional_metrics = [evaluate_pattern(case_rows, spec["cue_name"], spec["field"], spec["direction"]) for spec in patterns]
    best_ba = max(finite([metric.get("balanced_accuracy") for metric in provisional_metrics]), default=math.nan)
    baseline_ba = f(baseline.get("baseline_best_balanced_accuracy"))
    baseline_gain = best_ba - baseline_ba if math.isfinite(best_ba) and math.isfinite(baseline_ba) else math.nan
    missing_true_terms = [
        "full_semantic_geometric_current_support",
        "anchor_id_semantic_query_head_control_reruns",
    ]
    if int(f(metadata.get("geometry_case_count"), 0)) < len(case_rows):
        missing_true_terms = [
            "true_pose_depth_parallax_full_28case",
            "true_depth_spread_proxy_full_28case",
            *missing_true_terms,
        ]
    return build_case_metric_track(
        "trackL2_anchor_scale_observability",
        "Track L2 Anchor Scale Observability Proxy",
        case_rows,
        patterns,
        schema_suffix="anchor_scale_observability_proxy_v1",
        proxy_note=(
            "Uses C4 same-space rows plus R/R2 head/anchor-edge diagnostics. "
            "When per-chunk geometry sidecars are present, camera-baseline/depth-spread terms are joined; "
            "otherwise temporal frame-delta is used as a fallback proxy. Runtime action still requires full current_support and control reruns."
        ),
        missing_true_terms=missing_true_terms,
        extra_summary={
            **metadata,
            **baseline,
            "best_BA_gain_over_v98_region_trackL": baseline_gain,
            "beats_v98_region_trackL_by_0p05": bool(math.isfinite(baseline_gain) and baseline_gain >= 0.05),
        },
        extra_rows={
            "anchor_role_rows.csv": role_rows,
            "geometry_edge_rows.csv": geometry_edge_rows,
            "geometry_read_errors.csv": geometry_read_errors,
        },
        failure_filename="anchor_observability_failure_report.md",
    )


def build_q_proxy_track() -> dict[str, Any]:
    l2_rows = read_rows(ROOT / "trackL2_anchor_scale_observability/rows.csv")
    case_rows, metadata = build_q_proxy_case_rows(l2_rows)
    or_specs = [
        {"field": "stale_anchor_score_proxy", "direction": "higher_bad"},
        {"field": "no_scale_evidence_proxy", "direction": "higher_bad"},
        {"field": "admission_block_score_proxy", "direction": "higher_bad"},
        {"field": "low_observability_risk", "direction": "higher_bad"},
        {"field": "low_temporal_parallax_risk", "direction": "higher_bad"},
        {"field": "low_geometry_parallax_risk", "direction": "higher_bad"},
        {"field": "low_geometry_depth_spread_risk", "direction": "higher_bad"},
        {"field": "same_space_inconsistency_risk_proxy", "direction": "higher_bad"},
        {"field": "temporal_parallax_x_current_support", "direction": "lower_bad"},
        {"field": "temporal_parallax_x_same_space_risk", "direction": "higher_bad"},
        {"field": "geometry_parallax_x_current_support", "direction": "lower_bad"},
        {"field": "geometry_parallax_depth_x_current_support", "direction": "lower_bad"},
        {"field": "geometry_depth_ratio_risk", "direction": "higher_bad"},
        {"field": "fresh_supported_score_proxy", "direction": "lower_bad"},
        {"field": "fresh_supported_anchor_count_proxy", "direction": "lower_bad"},
        {"field": "scale_observability_score", "direction": "lower_bad"},
    ]
    best_or_rule, or_search_rows = q_or_rule_search_rows(case_rows, or_specs)
    if best_or_rule:
        for row in case_rows:
            selected = or_rule_selected(row, best_or_rule)
            row["q_composite_delay_or_no_scale_proxy"] = 1.0 if selected else 0.0
            if not selected:
                row["admission_decision_proxy_initial"] = row.get("admission_decision_proxy", "")
                row["admission_decision_proxy"] = (
                    "ALLOW_UPDATE"
                    if row.get("admission_decision_proxy") == "ALLOW_UPDATE"
                    else "CONTEXT_ONLY"
                )
                continue
            row["admission_decision_proxy_initial"] = row.get("admission_decision_proxy", "")
            lhs_bad = (
                f(row.get(best_or_rule.get("field_a"))) <= f(best_or_rule.get("threshold_a"))
                if best_or_rule.get("direction_a") == "lower_bad"
                else f(row.get(best_or_rule.get("field_a"))) >= f(best_or_rule.get("threshold_a"))
            )
            rhs_bad = (
                f(row.get(best_or_rule.get("field_b"))) <= f(best_or_rule.get("threshold_b"))
                if best_or_rule.get("direction_b") == "lower_bad"
                else f(row.get(best_or_rule.get("field_b"))) >= f(best_or_rule.get("threshold_b"))
            )
            if lhs_bad and "stale" in str(best_or_rule.get("field_a")):
                row["admission_decision_proxy"] = "DELAY_UPDATE"
            elif rhs_bad and "stale" in str(best_or_rule.get("field_b")):
                row["admission_decision_proxy"] = "DELAY_UPDATE"
            else:
                row["admission_decision_proxy"] = "NO_SCALE_EVIDENCE"
    else:
        for row in case_rows:
            row["q_composite_delay_or_no_scale_proxy"] = math.nan
            row["admission_decision_proxy_initial"] = row.get("admission_decision_proxy", "")
    patterns = [
        {"cue_name": "Q_composite_delay_or_no_scale_proxy", "field": "q_composite_delay_or_no_scale_proxy", "direction": "higher_bad"},
        {"cue_name": "Q_admission_block_score_proxy", "field": "admission_block_score_proxy", "direction": "higher_bad"},
        {"cue_name": "Q_no_scale_evidence_proxy", "field": "no_scale_evidence_proxy", "direction": "higher_bad"},
        {"cue_name": "Q_low_temporal_parallax_risk", "field": "low_temporal_parallax_risk", "direction": "higher_bad"},
        {"cue_name": "Q_temporal_parallax_current_low_bad", "field": "temporal_parallax_x_current_support", "direction": "lower_bad"},
        {"cue_name": "Q_low_geometry_parallax_risk", "field": "low_geometry_parallax_risk", "direction": "higher_bad"},
        {"cue_name": "Q_low_geometry_depth_spread_risk", "field": "low_geometry_depth_spread_risk", "direction": "higher_bad"},
        {"cue_name": "Q_geometry_parallax_current_low_bad", "field": "geometry_parallax_x_current_support", "direction": "lower_bad"},
        {"cue_name": "Q_geometry_parallax_depth_current_low_bad", "field": "geometry_parallax_depth_x_current_support", "direction": "lower_bad"},
        {"cue_name": "Q_allow_score_low_bad", "field": "admission_allow_score_proxy", "direction": "lower_bad"},
        {"cue_name": "Q_stale_anchor_score_proxy", "field": "stale_anchor_score_proxy", "direction": "higher_bad"},
    ]
    delay_no_scale = [row for row in case_rows if row.get("admission_decision_proxy") in {"DELAY_UPDATE", "NO_SCALE_EVIDENCE"}]
    bad = [row for row in case_rows if row.get("case_label") == "non_good"]
    good = [row for row in case_rows if row.get("case_label") == "good"]
    allow = [row for row in case_rows if row.get("admission_decision_proxy") == "ALLOW_UPDATE"]
    q_extra = {
        **metadata,
        "best_composite_rule": best_or_rule,
        "best_composite_balanced_accuracy": best_or_rule.get("balanced_accuracy", math.nan),
        "best_composite_bad_recall": best_or_rule.get("bad_recall", math.nan),
        "best_composite_good_FPR": best_or_rule.get("good_FPR", math.nan),
        "best_composite_q_gate_without_true_terms_pass": best_or_rule.get("q_gate_without_true_terms_pass", False),
        "admission_decision_counts": dict(Counter(str(row.get("admission_decision_proxy", "")) for row in case_rows)),
        "initial_admission_decision_counts": dict(Counter(str(row.get("admission_decision_proxy_initial", "")) for row in case_rows)),
        "delay_no_scale_bad_recall": len([row for row in delay_no_scale if row.get("case_label") == "non_good"]) / len(bad) if bad else math.nan,
        "delay_no_scale_good_FPR": len([row for row in delay_no_scale if row.get("case_label") == "good"]) / len(good) if good else math.nan,
        "allow_update_case_count": len(allow),
        "allow_update_mean_L3": mean([row.get("L3_handoff_transfer_penalty_proxy") for row in allow]),
        "non_allow_update_mean_L3": mean([row.get("L3_handoff_transfer_penalty_proxy") for row in case_rows if row not in allow]),
    }
    return build_case_metric_track(
        "trackQ_chunk_update_admission",
        "Track Q Chunk Update Admission Proxy",
        case_rows,
        patterns,
        schema_suffix="chunk_update_admission_proxy_v1",
        proxy_note="Chunk-level admission categories are derived from L2 proxy rows only; no scale/gauge update was run or allowed.",
        missing_true_terms=[
            "true_anchor_scale_observability",
            "runtime_update_outcome",
            "full_current_support_provider",
            "parallax_depth_control",
        ],
        extra_summary=q_extra,
        extra_rows={
            "composite_rule_metrics.csv": [best_or_rule] if best_or_rule else [],
            "composite_rule_search.csv": or_search_rows,
            "q_admission_gate_checks.csv": [
                {"gate": "composite_delay_no_scale_bad_recall_ge065", "pass": f(best_or_rule.get("bad_recall")) >= 0.65, "value": best_or_rule.get("bad_recall", math.nan)},
                {"gate": "composite_delay_no_scale_good_FPR_le025", "pass": f(best_or_rule.get("good_FPR"), 1.0) <= 0.25, "value": best_or_rule.get("good_FPR", math.nan)},
                {"gate": "composite_selected_positive_sequence_coverage_ge4", "pass": int(f(best_or_rule.get("selected_positive_sequence_coverage"), 0)) >= 4, "value": best_or_rule.get("selected_positive_sequence_coverage", math.nan)},
                {"gate": "required_true_terms_available", "pass": False, "value": "true_anchor_scale_observability;runtime_update_outcome;full_current_support_provider;parallax_depth_control"},
                {"gate": "runtime_action_allowed", "pass": False, "value": False},
            ],
        },
        failure_filename="chunk_update_admission_failure_report.md",
    )


def sequence_loso_rows(case_rows: list[dict[str, Any]], pattern_specs: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seqs = sorted({str(row.get("seq", "")) for row in case_rows if row.get("seq")})
    for spec in pattern_specs:
        field = spec["field"]
        direction = spec.get("direction", "higher_bad")
        cue_name = spec.get("cue_name", field)
        for holdout_seq in seqs:
            train = [row for row in case_rows if str(row.get("seq", "")) != holdout_seq]
            test = [row for row in case_rows if str(row.get("seq", "")) == holdout_seq]
            train_metric = best_threshold_metric(train, field, direction)
            threshold = f(train_metric.get("threshold"))
            test_metric = apply_threshold_metric(test, field, direction, threshold) if math.isfinite(threshold) else {}
            rows.append({
                "cue_name": cue_name,
                "field": field,
                "direction": direction,
                "holdout_seq": holdout_seq,
                "train_case_count": len(train),
                "test_case_count": len(test),
                "train_threshold": threshold,
                "train_balanced_accuracy": train_metric.get("balanced_accuracy", math.nan),
                "train_bad_recall": train_metric.get("bad_recall", math.nan),
                "train_good_FPR": train_metric.get("good_FPR", math.nan),
                "test_balanced_accuracy": test_metric.get("balanced_accuracy", math.nan),
                "test_bad_recall": test_metric.get("bad_recall", math.nan),
                "test_good_FPR": test_metric.get("good_FPR", math.nan),
                "test_selected_case_count": test_metric.get("selected_case_count", 0),
                "test_true_positive_cases": test_metric.get("true_positive_cases", ""),
                "test_false_positive_cases": test_metric.get("false_positive_cases", ""),
                "test_missed_positive_cases": test_metric.get("missed_positive_cases", ""),
            })
    return rows


def semantic_regime_metric_rows(case_rows: list[dict[str, Any]], pattern_specs: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    classes = sorted({str(row.get("dominant_semantic_class", "")) for row in case_rows if row.get("dominant_semantic_class")})
    for semantic_class in classes:
        parts = [row for row in case_rows if str(row.get("dominant_semantic_class", "")) == semantic_class]
        seq_cov = len({row.get("seq") for row in parts if row.get("seq")})
        label_counts = Counter(str(row.get("case_label", "")) for row in parts)
        for spec in pattern_specs:
            metric = evaluate_pattern(parts, spec.get("cue_name", spec["field"]), spec["field"], spec.get("direction", "higher_bad"))
            rows.append({
                "dominant_semantic_class": semantic_class,
                "case_count": len(parts),
                "sequence_coverage": seq_cov,
                "label_counts": dict(label_counts),
                "cue_name": spec.get("cue_name", spec["field"]),
                "field": spec["field"],
                "direction": spec.get("direction", "higher_bad"),
                "balanced_accuracy": metric.get("balanced_accuracy", math.nan),
                "bad_recall": metric.get("bad_recall", math.nan),
                "good_FPR": metric.get("good_FPR", math.nan),
                "abs_corr_L3": metric.get("abs_corr_L3", math.nan),
                "selected_case_count": metric.get("selected_case_count", 0),
                "true_positive_cases": metric.get("true_positive_cases", ""),
                "false_positive_cases": metric.get("false_positive_cases", ""),
                "missed_positive_cases": metric.get("missed_positive_cases", ""),
                "full_gate_scope_possible": len(parts) >= 8 and seq_cov >= 4 and label_counts.get("good", 0) > 0 and label_counts.get("non_good", 0) > 0,
            })
    return rows


def sequence_split_rows(case_rows: list[dict[str, Any]], fields: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seqs = sorted({str(row.get("seq", "")) for row in case_rows if row.get("seq")})
    for seq in seqs:
        parts = [row for row in case_rows if str(row.get("seq", "")) == seq]
        for field in fields:
            bad = [f(row.get(field)) for row in parts if row.get("case_label") == "non_good"]
            good = [f(row.get(field)) for row in parts if row.get("case_label") == "good"]
            rows.append({
                "seq": seq,
                "field": field,
                "case_count": len(parts),
                "bad_count": len([row for row in parts if row.get("case_label") == "non_good"]),
                "good_count": len([row for row in parts if row.get("case_label") == "good"]),
                "bad_mean": mean(bad),
                "good_mean": mean(good),
                "bad_min": min(finite(bad), default=math.nan),
                "bad_max": max(finite(bad), default=math.nan),
                "good_min": min(finite(good), default=math.nan),
                "good_max": max(finite(good), default=math.nan),
            })
    return rows


def grouped_split_rows(case_rows: list[dict[str, Any]], fields: list[str], group_field: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups = sorted({str(row.get(group_field, "")) for row in case_rows if str(row.get(group_field, ""))})
    for group in groups:
        parts = [row for row in case_rows if str(row.get(group_field, "")) == group]
        for field in fields:
            bad = [f(row.get(field)) for row in parts if row.get("case_label") == "non_good"]
            good = [f(row.get(field)) for row in parts if row.get("case_label") == "good"]
            rows.append({
                "group_field": group_field,
                "group_value": group,
                "field": field,
                "case_count": len(parts),
                "bad_count": len([row for row in parts if row.get("case_label") == "non_good"]),
                "good_count": len([row for row in parts if row.get("case_label") == "good"]),
                "bad_mean": mean(bad),
                "good_mean": mean(good),
                "bad_min": min(finite(bad), default=math.nan),
                "bad_max": max(finite(bad), default=math.nan),
                "good_min": min(finite(good), default=math.nan),
                "good_max": max(finite(good), default=math.nan),
                "case_ids": ";".join(str(row.get("case_id")) for row in parts),
            })
    return rows


def selection_audit_rows(case_rows: list[dict[str, Any]], metric: dict[str, Any]) -> list[dict[str, Any]]:
    field = str(metric.get("field", ""))
    direction = str(metric.get("direction", "higher_bad"))
    threshold = f(metric.get("threshold"))
    if not field or not math.isfinite(threshold):
        return []
    selected: list[dict[str, Any]] = []
    missed: list[dict[str, Any]] = []
    for row in case_rows:
        value = f(row.get(field))
        if not math.isfinite(value):
            continue
        is_selected = value <= threshold if direction == "lower_bad" else value >= threshold
        if is_selected:
            selected.append(row)
        elif row.get("case_label") == "non_good":
            missed.append(row)
    out: list[dict[str, Any]] = []
    for row in selected:
        out.append({
            "row_kind": "false_positive" if row.get("case_label") == "good" else "true_positive",
            "cue_name": metric.get("cue_name", ""),
            "field": field,
            "direction": direction,
            "threshold": threshold,
            "case_id": row.get("case_id"),
            "seq": row.get("seq"),
            "case_label": row.get("case_label"),
            "field_value": row.get(field),
            "L3_handoff_transfer_penalty_proxy": row.get("L3_handoff_transfer_penalty_proxy"),
            "dominant_semantic_class": row.get("dominant_semantic_class"),
            "query_hit_mean": row.get("query_hit_mean"),
            "query_hit_max_mean": row.get("query_hit_max_mean"),
            "R_same_mean": row.get("R_same_mean"),
            "current_feature_residual_mean": row.get("current_feature_residual_mean"),
            "source_retention_mean": row.get("source_retention_mean"),
            "source_residual_mean": row.get("source_residual_mean"),
        })
    for row in missed:
        out.append({
            "row_kind": "missed_positive",
            "cue_name": metric.get("cue_name", ""),
            "field": field,
            "direction": direction,
            "threshold": threshold,
            "case_id": row.get("case_id"),
            "seq": row.get("seq"),
            "case_label": row.get("case_label"),
            "field_value": row.get(field),
            "L3_handoff_transfer_penalty_proxy": row.get("L3_handoff_transfer_penalty_proxy"),
            "dominant_semantic_class": row.get("dominant_semantic_class"),
            "query_hit_mean": row.get("query_hit_mean"),
            "query_hit_max_mean": row.get("query_hit_max_mean"),
            "R_same_mean": row.get("R_same_mean"),
            "current_feature_residual_mean": row.get("current_feature_residual_mean"),
            "source_retention_mean": row.get("source_retention_mean"),
            "source_residual_mean": row.get("source_residual_mean"),
        })
    return out


def build_stage3_track(track_dir: str, label: str, patterns: list[dict[str, str]], *, c4_gain_gate: bool = False) -> dict[str, Any]:
    out = ROOT / track_dir
    cases = stage3_case_rows()
    query_hit_only = evaluate_pattern(cases, "query_hit_only_higher_bad", "query_hit_mean", "higher_bad")
    threshold_specs = [{"cue_name": "query_hit_only_higher_bad", "field": "query_hit_mean", "direction": "higher_bad"}] + patterns
    metrics: list[dict[str, Any]] = []
    for pattern in patterns:
        metric = evaluate_pattern(cases, pattern["cue_name"], pattern["field"], pattern["direction"])
        ba_gain = f(metric.get("balanced_accuracy")) - f(query_hit_only.get("balanced_accuracy"))
        metric = attach_control_audit(metric, cases)
        metrics.append(with_gate(metric, require_ba_gain=0.05 if c4_gain_gate else None, ba_gain=ba_gain))
    best = sorted(
        metrics,
        key=lambda row: (
            b(row.get("gate_pass")),
            b(row.get("gate_without_controls_pass")),
            f(row.get("balanced_accuracy")),
            f(row.get("bad_recall")),
            -f(row.get("good_FPR"), 1.0),
        ),
        reverse=True,
    )
    best_row = best[0] if best else {}
    gate = any(b(row.get("gate_pass")) for row in metrics)
    label_counts = Counter(str(row.get("case_label")) for row in cases)
    sequence_coverage = len({row.get("seq") for row in cases if row.get("seq")})
    blocker = (
        ""
        if gate
        else (
            f"{label} did not pass the complete Stage3 diagnostic gate. "
            "The v100 same-space rows support offline case-level scoring, but required "
            "anchor-id rotation, semantic-label rotation, query-head random, and same-count random controls "
            "were not materialized in this builder; runtime action remains blocked."
        )
    )
    summary = {
        "schema": f"acl2_v100_{track_dir}_stage3_diagnostic_v1",
        "status": "complete_diagnostic_no_go" if cases and not gate else ("complete" if gate else "blocked_missing_inputs"),
        "gate_pass": gate,
        "runtime_action_allowed": False,
        "prereq_trackS_gate_pass": True,
        "case_count": len(cases),
        "sequence_coverage": sequence_coverage,
        "label_counts": dict(label_counts),
        "canonical_space_name": "S-B_preprojection_hidden",
        "best_pattern": best_row.get("cue_name", ""),
        "best_balanced_accuracy": best_row.get("balanced_accuracy", math.nan),
        "best_bad_recall": best_row.get("bad_recall", math.nan),
        "best_good_FPR": best_row.get("good_FPR", math.nan),
        "best_abs_corr_L3": best_row.get("abs_corr_L3", math.nan),
        "query_hit_only_balanced_accuracy": query_hit_only.get("balanced_accuracy", math.nan),
        "control_margins_available": False,
        "blocker": blocker,
    }
    write_json(out / "summary.json", summary)
    write_rows(out / "rows.csv", cases)
    write_rows(out / "pattern_metrics.csv", metrics)
    write_rows(out / "control_audit_rows.csv", control_audit_rows(metrics))
    write_rows(out / "query_hit_only_metrics.csv", [query_hit_only])
    write_rows(out / "threshold_feasibility.csv", threshold_feasibility_rows(cases, threshold_specs))
    write_rows(out / "sequence_loso_metrics.csv", sequence_loso_rows(cases, threshold_specs))
    write_rows(out / "sequence_split_rows.csv", sequence_split_rows(cases, [spec["field"] for spec in threshold_specs]))
    write_rows(out / "semantic_split_rows.csv", grouped_split_rows(cases, [spec["field"] for spec in threshold_specs], "dominant_semantic_class"))
    write_rows(out / "semantic_regime_metrics.csv", semantic_regime_metric_rows(cases, threshold_specs))
    write_rows(out / "false_positive_missed_case_rows.csv", selection_audit_rows(cases, best_row))
    write_rows(out / "gate_checks.csv", [
        {"gate": "trackS_gate_pass", "pass": True},
        {"gate": "case_count_ge28", "pass": len(cases) >= 28, "value": len(cases)},
        {"gate": "sequence_coverage_ge4", "pass": sequence_coverage >= 4, "value": sequence_coverage},
        {"gate": "any_pattern_without_controls_pass", "pass": any(b(row.get("gate_without_controls_pass")) for row in metrics)},
        {"gate": "required_control_margins_available", "pass": False},
        {"gate": "complete_diagnostic_gate_pass", "pass": gate},
    ])
    write_json(out / "missing_prereq.json", {
        "missing_controls": [
            "anchor_id_rotation_margin",
            "semantic_label_rotation_margin",
            "query_head_random_margin",
            "same_count_random_margin",
        ],
        "note": "Case-level same-space scoring was built from v100 S-B rows plus v99 case labels; control permutations were not run.",
    })
    write_text(out / "blocked_reason.md", blocker + "\n")
    write_text(
        out / "failure_report.md",
        "# Stage3 Diagnostic Failure Report\n\n"
        f"- track: {label}\n"
        f"- gate_pass: {gate}\n"
        f"- best_pattern: {summary['best_pattern']}\n"
        f"- best_balanced_accuracy: {summary['best_balanced_accuracy']}\n"
        f"- best_bad_recall: {summary['best_bad_recall']}\n"
        f"- best_good_FPR: {summary['best_good_FPR']}\n"
        f"- best_abs_corr_L3: {summary['best_abs_corr_L3']}\n"
        f"- blocker: {blocker or 'none'}\n",
    )
    write_text(
        out / "fail_forward_report.md",
        "# Stage3 Fail-Forward Audit\n\n"
        "- threshold_feasibility.csv records every best-threshold summary and any threshold that satisfies recall/FPR/sequence gates before controls.\n"
        "- sequence_loso_metrics.csv trains a threshold on three sequences and evaluates the held-out sequence for every tested field.\n"
        "- sequence_split_rows.csv records per-sequence bad/good means and ranges for the tested fields.\n"
        "- semantic_split_rows.csv records dominant-semantic-class split means and ranges for the tested fields.\n"
        "- semantic_regime_metrics.csv records per-dominant-semantic-class threshold metrics; full_gate_scope_possible marks whether a class has enough mixed-label 4-sequence support for a full gate.\n"
        "- false_positive_missed_case_rows.csv records true positives, false positives, and missed positives for the best pattern.\n"
        "- control_audit_rows.csv records deterministic same-count random controls and explains unavailable anchor-id/semantic-label/query-head rotations.\n"
        "- If feasible_gate_threshold_count is 0, no scalar threshold in the tested field met recall>=0.65, good_FPR<=0.25, four-sequence positive coverage, and selected-positive max-sequence fraction<=0.60.\n"
        "- current_feature_residual_mean is recorded as a current-support proxy only; it is not a validated semantic/geometric current_support gate.\n"
        "- Control margins were not materialized, so even a threshold-only pass would still require identity/semantic/query-head/same-count controls before action.\n",
    )
    write_text(
        out / "control_gap_report.md",
        "# Stage3 Control Gap Report\n\n"
        "- The tested rows are case-level aggregations over route_weighted_all_heads, so query-head random controls cannot be faithfully computed from these rows alone.\n"
        "- The current selector fields do not depend on a specific anchor-id identity permutation; anchor-id rotation margin would not establish identity specificity without edge-level rows.\n"
        "- dominant_semantic_class splits are diagnostic only; semantic-label rotation controls require anchor/edge-level semantic assignment reruns or explicit permutation rows.\n"
        "- current_feature_residual_mean is a latent-current residual proxy, not the full current_support definition from the plan.\n",
    )
    write_text(
        out / "what_would_have_to_be_true_to_pass.md",
        "A pattern must pass recall/FPR/correlation/sequence gates and the required identity/semantic/query-head/same-count controls before M3 or runtime action.\n",
    )
    write_text(out / "visual_manifest.csv", "path,description\n")
    return summary


def blocked_track(track_dir: str, prereq: dict[str, Any], label: str, reason: str | None = None) -> dict[str, Any]:
    out = ROOT / track_dir
    track_s_pass = b(prereq.get("gate_pass"))
    if track_s_pass:
        blocker = reason or f"{label} was not materialized after Track S passed; required track-specific diagnostics are missing."
        status = "blocked_track_specific_prereq"
        missing = "track-specific diagnostic/control artifacts"
    else:
        blocker = f"{label} blocked because Track S same-space latent state gate did not pass."
        status = "blocked_missing_prereq"
        missing = "Track S same-space gate"
    summary = {
        "schema": f"acl2_v100_{track_dir}_blocked_v1",
        "status": status,
        "gate_pass": False,
        "runtime_action_allowed": False,
        "missing_prereq": missing,
        "prereq_trackS_gate_pass": prereq.get("gate_pass"),
        "blocker": blocker,
    }
    write_json(out / "summary.json", summary)
    write_json(out / "missing_prereq.json", {"trackS_summary": prereq})
    write_rows(out / "rows.csv", [])
    write_rows(out / "gate_checks.csv", [
        {"gate": "trackS_gate_pass", "pass": track_s_pass},
        {"gate": "track_specific_prereq_available", "pass": False},
    ])
    write_text(out / "blocked_reason.md", summary["blocker"] + "\n")
    write_text(out / "failure_report.md", summary["blocker"] + "\n")
    write_text(
        out / "what_would_have_to_be_true_to_pass.md",
        "Track S must pass, then this track needs its own diagnostic rows, controls, and gate evidence before action.\n",
    )
    write_text(out / "visual_manifest.csv", "path,description\n")
    return summary


def final_decision(stage0_summary: dict[str, Any], track_s_summary: dict[str, Any], downstream: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out = ROOT / "final_decision"
    track_s_pass = b(track_s_summary.get("gate_pass"))
    diagnostic_track_names = {
        "trackN2_anchor_identity_graph",
        "trackO2_freshness_current_support",
        "trackL2_anchor_scale_observability",
        "trackC4_identity_latent_gauge_ruler",
        "trackF4_ttt_write_to_use_same_space",
    }
    diagnostics_pass = any(
        b(summary.get("gate_pass")) for name, summary in downstream.items() if name in diagnostic_track_names
    )
    taxonomy = (
        "SAME_SPACE_DIAGNOSTIC_READY_ACTION_BLOCKED"
        if track_s_pass and not diagnostics_pass
        else "SAME_SPACE_INSTRUMENTATION_NO_GO"
    )
    decision = {
        "schema": "acl2_v100_final_decision_v1",
        "final_taxonomy": taxonomy,
        "stage0_gate_pass": stage0_summary.get("gate_pass"),
        "trackS_gate_pass": track_s_summary.get("gate_pass"),
        "diagnostic_track_gate_pass": diagnostics_pass,
        "trackM3_gate_pass": downstream.get("trackM3_identity_action_simulator", {}).get("gate_pass", False),
        "runtime_action_allowed": False,
        "runtime_action_pilot_run": False,
        "full_validation_run": False,
        "full_method_success": False,
        "claim": "No runtime or full validation success is claimed.",
        "case_count": track_s_summary.get("case_count"),
        "sequence_coverage": track_s_summary.get("sequence_coverage"),
        "primary_blocker": track_s_summary.get("blocker") or (
            "Stage3 diagnostics did not pass complete gates; M3/E4 runtime action remains blocked."
            if not diagnostics_pass
            else "M3/E4 runtime action remains blocked."
        ),
    }
    write_json(out / "final_decision.json", decision)
    write_json(out / "summary.json", decision)
    write_text(
        out / "final_report.md",
        "# ACL2 v100 Final Report\n\n"
        f"- final_taxonomy: {taxonomy}\n"
        f"- TrackS gate: {track_s_summary.get('gate_pass')}\n"
        f"- runtime_action_allowed: false\n"
        f"- primary_blocker: {decision['primary_blocker']}\n",
    )
    write_text(
        out / "failure_report.md",
        "# ACL2 v100 Failure Report\n\n"
        f"- {decision['primary_blocker']}\n"
        "- Runtime action and full validation were not run.\n",
    )
    write_text(
        out / "what_would_have_to_be_true_to_pass.md",
        "Track S, at least one N2/O2/L2/C4/F4 diagnostic, M3 simulator, and then E4 runtime gates must pass before full validation.\n",
    )
    write_rows(out / "rows.csv", [decision])
    write_rows(out / "gate_checks.csv", [
        {"gate": "stage0_gate_pass", "pass": b(stage0_summary.get("gate_pass"))},
        {"gate": "trackS_gate_pass", "pass": track_s_pass},
        {"gate": "runtime_action_allowed", "pass": False},
    ])
    write_text(out / "visual_manifest.csv", "path,description\n")
    return decision


def main() -> None:
    global ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    ROOT = args.root

    stage0_summary = stage0()
    track_s_summary = track_s()
    downstream: dict[str, dict[str, Any]] = {}
    if b(track_s_summary.get("gate_pass")):
        downstream["trackN2_anchor_identity_graph"] = build_stage3_track(
            "trackN2_anchor_identity_graph",
            "Track N2",
            [
                {"cue_name": "N-A_same_space_high_hit_high_R", "field": "high_hit_high_R_frac", "direction": "higher_bad"},
                {"cue_name": "N-E_query_head_risk_x_R_same", "field": "query_risk_x_R_same", "direction": "higher_bad"},
                {"cue_name": "N-query_hit_x_R_same", "field": "query_hit_x_R_same", "direction": "higher_bad"},
                {"cue_name": "N-headmax_x_R_same", "field": "query_hit_max_x_R_same", "direction": "higher_bad"},
                {"cue_name": "N-head75_x_R_same", "field": "query_head_ge75_x_R_same", "direction": "higher_bad"},
                {"cue_name": "N-stale_unsupported_query_R_current_proxy", "field": "query_hit_x_R_same_x_current_residual", "direction": "higher_bad"},
                {"cue_name": "N-headmax_R_current_proxy", "field": "query_hit_max_x_R_same_x_current_residual", "direction": "higher_bad"},
            ],
        )
        downstream["trackO2_freshness_current_support"] = build_stage3_track(
            "trackO2_freshness_current_support",
            "Track O2",
            [
                {"cue_name": "O2_current_residual_x_R_same", "field": "current_residual_x_R_same", "direction": "higher_bad"},
                {"cue_name": "O2_query_current_residual", "field": "query_hit_x_current_residual", "direction": "higher_bad"},
                {"cue_name": "O2_query_R_current_proxy", "field": "query_hit_x_R_same_x_current_residual", "direction": "higher_bad"},
                {"cue_name": "O2_headmax_R_current_proxy", "field": "query_hit_max_x_R_same_x_current_residual", "direction": "higher_bad"},
                {"cue_name": "O2_fresh_supported_score_low_bad", "field": "fresh_supported_score_proxy", "direction": "lower_bad"},
            ],
        )
        downstream["trackC4_identity_latent_gauge_ruler"] = build_stage3_track(
            "trackC4_identity_latent_gauge_ruler",
            "Track C4",
            [
                {"cue_name": "C4_high_hit_high_R_same", "field": "high_hit_high_R_frac", "direction": "higher_bad"},
                {"cue_name": "C4_R_same_mean", "field": "R_same_mean", "direction": "higher_bad"},
                {"cue_name": "C4_query_hit_x_R_same", "field": "query_hit_x_R_same", "direction": "higher_bad"},
                {"cue_name": "C4_headmax_x_R_same", "field": "query_hit_max_x_R_same", "direction": "higher_bad"},
                {"cue_name": "C4_R_same_x_current_proxy", "field": "current_residual_x_R_same", "direction": "higher_bad"},
                {"cue_name": "C4_query_R_current_proxy", "field": "query_hit_x_R_same_x_current_residual", "direction": "higher_bad"},
                {"cue_name": "C4_headmax_R_current_proxy", "field": "query_hit_max_x_R_same_x_current_residual", "direction": "higher_bad"},
            ],
            c4_gain_gate=True,
        )
        downstream["trackF4_ttt_write_to_use_same_space"] = build_stage3_track(
            "trackF4_ttt_write_to_use_same_space",
            "Track F4",
            [
                {"cue_name": "F4_write_cache_current_risk", "field": "write_cache_current_risk", "direction": "higher_bad"},
                {"cue_name": "F4_R_write_cache_mean", "field": "R_write_cache_mean", "direction": "higher_bad"},
                {"cue_name": "F4_R_cache_current_mean", "field": "R_cache_current_mean", "direction": "higher_bad"},
                {"cue_name": "F4_write_cache_current_risk_x_current_proxy", "field": "write_cache_current_risk_x_current_residual", "direction": "higher_bad"},
                {"cue_name": "F4_headmax_R_current_proxy", "field": "query_hit_max_x_R_same_x_current_residual", "direction": "higher_bad"},
            ],
        )
        downstream["trackR_edge_head_control_audit"] = build_edge_head_control_audit()
        downstream["trackR2_anchor_edge_identity_control_audit"] = build_anchor_edge_identity_control_audit()
        downstream["trackL2_anchor_scale_observability"] = build_l2_proxy_track()
        downstream["trackQ_chunk_update_admission"] = build_q_proxy_track()
    else:
        for name, label in {
            "trackN2_anchor_identity_graph": "Track N2",
            "trackO2_freshness_current_support": "Track O2",
            "trackL2_anchor_scale_observability": "Track L2",
            "trackC4_identity_latent_gauge_ruler": "Track C4",
            "trackF4_ttt_write_to_use_same_space": "Track F4",
        }.items():
            downstream[name] = blocked_track(name, track_s_summary, label)
    any_stage3_gate = any(
        b(downstream.get(name, {}).get("gate_pass"))
        for name in [
            "trackN2_anchor_identity_graph",
            "trackO2_freshness_current_support",
            "trackL2_anchor_scale_observability",
            "trackC4_identity_latent_gauge_ruler",
            "trackF4_ttt_write_to_use_same_space",
        ]
    )
    downstream["trackM3_identity_action_simulator"] = blocked_track(
        "trackM3_identity_action_simulator",
        track_s_summary,
        "Track M3",
        None if not b(track_s_summary.get("gate_pass")) else (
            "Track M3 blocked: no N2/O2/L2/C4/F4 diagnostic gate passed, so there is no action family to simulate."
            if not any_stage3_gate
            else "Track M3 simulator was not run in this builder."
        ),
    )
    downstream["trackE4_swa_identity_handoff_control"] = blocked_track(
        "trackE4_swa_identity_handoff_control",
        track_s_summary,
        "Track E4",
        None if not b(track_s_summary.get("gate_pass")) else "Track E4 blocked: M3 simulator did not pass, so runtime pilot is not allowed.",
    )
    for name, label, reason in [
        (
            "trackD4_read_current_support_provider",
            "Track D4",
            "Track D4 blocked: READ current-support provider integration was not required/run because Stage3 action diagnostics did not pass.",
        ),
        (
            "trackJ_anchor_instance_atlas",
            "Track J",
            "Track J blocked: semantic anchor instance atlas was not materialized after Stage3 No-Go.",
        ),
    ]:
        downstream[name] = blocked_track(name, track_s_summary, label, None if not b(track_s_summary.get("gate_pass")) else reason)
    if "trackQ_chunk_update_admission" not in downstream:
        downstream["trackQ_chunk_update_admission"] = blocked_track(
            "trackQ_chunk_update_admission",
            track_s_summary,
            "Track Q",
            None
            if not b(track_s_summary.get("gate_pass"))
            else "Track Q blocked: chunk-level scale update admission gate was not materialized.",
        )
    decision = final_decision(stage0_summary, track_s_summary, downstream)
    print(json.dumps(decision, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
