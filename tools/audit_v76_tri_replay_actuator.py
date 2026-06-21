#!/usr/bin/env python3
"""Audit exact tri-replay actuator evidence for ACL2 v76-TF Phase 2."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.v76tf_common import (
    C9_REPEAT_DIR,
    REPO_ROOT,
    V45_LEDGER,
    V46B_REGISTRY,
    V76_ROOT,
    boolish,
    collect_numeric_by_key,
    count_nonzero_csv,
    ensure_dir,
    first_row,
    mean,
    read_csv,
    read_jsonl,
    rel,
    safe_float,
    safe_int,
    walk_json,
    write_csv,
    write_json,
    write_text,
)


PLAN_HOOK_MAPPING = [
    {
        "plan_hook": "TTT_TRI_REPLAY_ENABLE",
        "existing_surface": "TTT_WRITE_GRADIENT_REVERSAL_MODE=tri_replay",
        "status": "implemented_existing_surface",
    },
    {
        "plan_hook": "TTT_TRI_REPLAY_PRESET",
        "existing_surface": "TTT_WRITE_TRI_REPLAY_POSITIVE_FRAC / NEGATIVE_FRAC / NEUTRAL_LAMBDA / CHUNK_PARAMS",
        "status": "implemented_existing_surface",
    },
    {
        "plan_hook": "TTT_TRI_GAMMA",
        "existing_surface": "TTT_WRITE_GRADIENT_REVERSAL_GAMMA / TTT_WRITE_GRADIENT_REVERSAL_CHUNK_GAMMAS",
        "status": "implemented_existing_surface",
    },
    {
        "plan_hook": "TTT_COMMIT_EMA_ENABLE",
        "existing_surface": "TTT_WRITE_COMMIT_EMA_CHUNKS and TTT_WRITE_COMMIT_EMA_ALPHA",
        "status": "implemented_existing_surface",
    },
    {
        "plan_hook": "TTT_COMMIT_EMA_ALPHA",
        "existing_surface": "TTT_WRITE_COMMIT_EMA_ALPHA",
        "status": "implemented_existing_surface",
    },
    {
        "plan_hook": "TTT_TRI_REPLAY_DIAG_CHUNKS",
        "existing_surface": "TTT_WRITE_TRI_REPLAY_CHUNK_PARAMS",
        "status": "implemented_existing_surface",
    },
    {
        "plan_hook": "TTT_TRI_REPLAY_ROLE_SOURCE",
        "existing_surface": "TTT_WRITE_TRI_REPLAY_ROLE_MODE plus trace keys ttt_tri_replay_role_source where present",
        "status": "partial_existing_surface",
    },
    {
        "plan_hook": "TTT_TRI_REPLAY_TRIGGER_MODE",
        "existing_surface": "candidate wrapper/run script responsibility; no unified core env alias found",
        "status": "missing_v76_alias",
    },
    {
        "plan_hook": "READ_TTT_ROLE_ALIGNMENT_LOG",
        "existing_surface": "report-level alignment audit only in current tooling",
        "status": "missing_core_hook",
    },
    {
        "plan_hook": "SWA_ROUTE_PRESERVE_MASS",
        "existing_surface": "SWA overlap source replace summaries exist; preserve-mass handoff is not a dedicated alias",
        "status": "missing_v76_alias",
    },
]


def _path_from_registry(row: Mapping[str, Any]) -> Optional[Path]:
    raw = row.get("run_dir")
    if not raw:
        return None
    path = Path(str(raw))
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _registry_runs() -> List[Tuple[str, Path]]:
    rows = read_csv(V46B_REGISTRY)
    out: List[Tuple[str, Path]] = []
    for row_name in ("F010_ONLY_TTT", "F110_FRAME_ATTN_TTT", "F111_ALL_THREE"):
        row = first_row(rows, "row", row_name)
        path = _path_from_registry(row or {})
        if path is not None:
            out.append((f"v46b_{row_name}", path))
    return out


def _numeric_values_for_keys(rows: Iterable[Mapping[str, Any]], key_fragments: Iterable[str]) -> Dict[str, List[float]]:
    fragments = tuple(key_fragments)
    out: Dict[str, List[float]] = {}
    for row in rows:
        for path, value in walk_json(row):
            key = path[-1] if path else ""
            if not any(fragment in key for fragment in fragments):
                continue
            num = safe_float(value)
            if num is not None:
                out.setdefault(key, []).append(num)
            elif isinstance(value, bool):
                out.setdefault(key, []).append(1.0 if value else 0.0)
    return out


def _any_row_has_positive(rows: Iterable[Mapping[str, Any]], key_fragments: Iterable[str]) -> int:
    fragments = tuple(key_fragments)
    count = 0
    for row in rows:
        row_positive = False
        for path, value in walk_json(row):
            key = path[-1] if path else ""
            if not any(fragment in key for fragment in fragments):
                continue
            if isinstance(value, bool) and value:
                row_positive = True
                break
            num = safe_float(value)
            if num is not None and num > 0:
                row_positive = True
                break
        if row_positive:
            count += 1
    return count


def _trace_role_mass(path: Path) -> Dict[str, Any]:
    rows = read_jsonl(path)
    numeric = _numeric_values_for_keys(rows, ("pos_mass", "neu_mass", "neg_mass", "positive", "neutral", "negative"))
    return {
        "path": rel(path),
        "exists": path.exists(),
        "row_count": len(rows),
        "numeric_keys": sorted(numeric.keys()),
        "numeric_value_count": sum(len(values) for values in numeric.values()),
        "mean_by_key": {key: mean(values) for key, values in sorted(numeric.items())},
    }


def _hash_change_summary(rows: List[Mapping[str, Any]]) -> Dict[str, Any]:
    input_output_changes = 0
    output_hashes = set()
    next_hashes = set()
    controlled_hashes = set()
    state_diff_values: List[float] = []
    for row in rows:
        inp = row.get("controlled_input_state_hash")
        out = row.get("controlled_output_state_hash")
        nxt = row.get("hash_H_next")
        if inp is not None and out is not None and inp != out:
            input_output_changes += 1
        if out is not None:
            output_hashes.add(str(out))
        if nxt is not None:
            next_hashes.add(str(nxt))
        if row.get("controlled_state_hash") is not None:
            controlled_hashes.add(str(row.get("controlled_state_hash")))
        side = row.get("memory_side_effect")
        if isinstance(side, dict):
            diff = side.get("ttt", {}).get("state_diff") if isinstance(side.get("ttt"), dict) else None
            if isinstance(diff, dict):
                for key in ("max_abs", "mean_abs", "l2"):
                    num = safe_float(diff.get(key))
                    if num is not None:
                        state_diff_values.append(abs(num))
    return {
        "controlled_input_output_hash_change_rows": input_output_changes,
        "unique_controlled_output_state_hashes": len(output_hashes),
        "unique_hash_H_next": len(next_hashes),
        "unique_controlled_state_hashes": len(controlled_hashes),
        "memory_side_effect_numeric_count": len(state_diff_values),
        "memory_side_effect_mean_abs": mean(state_diff_values),
        "hash_change_available": bool(input_output_changes > 0 or len(output_hashes) > 1 or len(next_hashes) > 1 or any(value > 0 for value in state_diff_values)),
    }


def _run_summary(label: str, run_dir: Path) -> Dict[str, Any]:
    hmc_path = run_dir / "hmc_state_hash.jsonl"
    rows = read_jsonl(hmc_path)
    numeric = _numeric_values_for_keys(
        rows,
        (
            "tri_replay",
            "commit_ema",
            "post_zp",
            "projection_role_mass",
            "pos_mass",
            "neu_mass",
            "neg_mass",
        ),
    )
    applied_rows = _any_row_has_positive(rows, ("tri_replay_applied", "auxgeo_tri_replay_applied_layer_count"))
    layer_values = []
    for key, values in numeric.items():
        if "applied_layer_count" in key:
            layer_values.extend(values)
    pos_values = []
    neu_values = []
    neg_values = []
    for key, values in numeric.items():
        lower = key.lower()
        if "pos_mass" in lower or "positive_mass" in lower:
            pos_values.extend(values)
        elif "neu_mass" in lower or "neutral_mass" in lower:
            neu_values.extend(values)
        elif "neg_mass" in lower or "negative_mass" in lower:
            neg_values.extend(values)
    commit_ema_rows = _any_row_has_positive(rows, ("commit_ema_applied",))
    post_zp = count_nonzero_csv(run_dir / "post_zeropower_delta_norm.csv")
    trace_role = _trace_role_mass(run_dir / "v11_projection_trace/tri_replay_role_mass.jsonl")
    hash_summary = _hash_change_summary(rows)
    return {
        "label": label,
        "run_dir": rel(run_dir),
        "hmc_state_hash_exists": hmc_path.exists(),
        "hmc_rows": len(rows),
        "tri_replay_applied_rows_recursive": applied_rows,
        "tri_replay_applied_layer_count_sum_recursive": sum(layer_values) if layer_values else 0,
        "tri_replay_numeric_keys_seen": ",".join(sorted(k for k in numeric if "tri_replay" in k)[:40]),
        "tri_replay_pos_mass_mean_recursive": mean(pos_values),
        "tri_replay_neu_mass_mean_recursive": mean(neu_values),
        "tri_replay_neg_mass_mean_recursive": mean(neg_values),
        "role_mass_numeric_value_count_recursive": len(pos_values) + len(neu_values) + len(neg_values),
        "commit_ema_applied_rows_recursive": commit_ema_rows,
        "post_zeropower_delta_norm_rows": post_zp["row_count"],
        "post_zeropower_delta_nonzero_numeric_count": post_zp["nonzero_numeric_value_count"],
        "trace_role_mass_rows": trace_role["row_count"],
        "trace_role_mass_numeric_value_count": trace_role["numeric_value_count"],
        **hash_summary,
    }


def _ledger_direction() -> Dict[str, Any]:
    rows = read_csv(V45_LEDGER)
    by_component = {str(row.get("component")): row for row in rows}
    keys = ("tri_replay", "tri_gamma_chunk_map", "commit_ema")
    values = {key: safe_float(by_component.get(key, {}).get("effect_delta_vs_C9")) for key in keys}
    return {
        "source_artifact": rel(V45_LEDGER),
        "values": values,
        "all_required_positive": all(value is not None and value > 0 for value in values.values()),
    }


def _registry_crosscheck() -> Dict[str, Any]:
    rows = read_csv(V46B_REGISTRY)
    out: Dict[str, Any] = {"source_artifact": rel(V46B_REGISTRY)}
    for row_name in ("F010_ONLY_TTT", "F110_FRAME_ATTN_TTT", "F111_ALL_THREE"):
        row = first_row(rows, "row", row_name)
        if not row:
            out[row_name] = {"available": False}
            continue
        out[row_name] = {
            "available": True,
            "ttt_tri_replay_applied_count": safe_int(row.get("ttt_tri_replay_applied_count")),
            "ttt_tri_replay_applied_layer_count_sum": safe_int(row.get("ttt_tri_replay_applied_layer_count_sum")),
            "ttt_positive_mass_mean": safe_float(row.get("ttt_positive_mass_mean")),
            "ttt_neutral_mass_mean": safe_float(row.get("ttt_neutral_mass_mean")),
            "ttt_negative_mass_mean": safe_float(row.get("ttt_negative_mass_mean")),
            "row_valid": boolish(row.get("row_valid")),
            "no_chunk_policy_pass": boolish(row.get("no_chunk_policy_pass")),
        }
    return out


def _gate(run_rows: List[Mapping[str, Any]], registry: Mapping[str, Any], ledger: Mapping[str, Any]) -> Dict[str, Any]:
    applied_recursive = sum(safe_int(row.get("tri_replay_applied_rows_recursive")) or 0 for row in run_rows)
    applied_registry = sum(
        safe_int((registry.get(name) or {}).get("ttt_tri_replay_applied_count")) or 0
        for name in ("F010_ONLY_TTT", "F110_FRAME_ATTN_TTT", "F111_ALL_THREE")
    )
    role_mass_values = sum(safe_int(row.get("role_mass_numeric_value_count_recursive")) or 0 for row in run_rows)
    role_trace_values = sum(safe_int(row.get("trace_role_mass_numeric_value_count")) or 0 for row in run_rows)
    post_zp_nonzero = sum(safe_int(row.get("post_zeropower_delta_nonzero_numeric_count")) or 0 for row in run_rows)
    hash_available = any(bool(row.get("hash_change_available")) for row in run_rows)
    return {
        "tri_replay_applied_count_gt0": bool(applied_recursive > 0 or applied_registry > 0),
        "tri_replay_applied_rows_recursive_total": applied_recursive,
        "tri_replay_applied_count_registry_total": applied_registry,
        "role_mass_nonempty": bool(role_mass_values > 0 or role_trace_values > 0),
        "role_mass_numeric_value_count_recursive_total": role_mass_values,
        "trace_role_mass_numeric_value_count_total": role_trace_values,
        "post_zp_delta_available": bool(post_zp_nonzero > 0),
        "post_zp_nonzero_numeric_count_total": post_zp_nonzero,
        "next_probe_or_state_hash_changes": bool(hash_available),
        "c9_knockout_direction_reproduced": bool(ledger.get("all_required_positive")),
        "phase2_actuator_gate_pass": bool(
            (applied_recursive > 0 or applied_registry > 0)
            and (role_mass_values > 0 or role_trace_values > 0)
            and post_zp_nonzero > 0
            and hash_available
            and ledger.get("all_required_positive")
        ),
    }


def _write_report(out_dir: Path, gate: Mapping[str, Any], run_rows: List[Mapping[str, Any]], ledger: Mapping[str, Any]) -> None:
    lines = [
        "# v76 Phase 2 Tri-Replay Actuator Audit",
        "",
        "This audit reads existing C9/v46B traces and does not create new trajectory metrics.",
        "",
        "## Gate",
        "",
    ]
    for key, value in gate.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend([
        "",
        "## Run Trace Summary",
        "",
        "| label | hmc_rows | applied_rows | role_mass_values | post_zp_nonzero | hash_change | commit_ema_rows |",
        "|---|---:|---:|---:|---:|---|---:|",
    ])
    for row in run_rows:
        lines.append(
            f"| `{row.get('label')}` | {row.get('hmc_rows')} | {row.get('tri_replay_applied_rows_recursive')} | "
            f"{row.get('role_mass_numeric_value_count_recursive')}+trace {row.get('trace_role_mass_numeric_value_count')} | "
            f"{row.get('post_zeropower_delta_nonzero_numeric_count')} | {row.get('hash_change_available')} | "
            f"{row.get('commit_ema_applied_rows_recursive')} |"
        )
    lines.extend([
        "",
        "## C9 Direction Cross-Check",
        "",
        f"- Source: `{ledger.get('source_artifact')}`",
        f"- Values: `{ledger.get('values')}`",
        "",
        "## Hook Mapping",
        "",
        "| plan hook | existing surface | status |",
        "|---|---|---|",
    ])
    for row in PLAN_HOOK_MAPPING:
        lines.append(f"| `{row['plan_hook']}` | `{row['existing_surface']}` | `{row['status']}` |")
    write_text(out_dir / "tri_replay_actuator_report.md", "\n".join(lines) + "\n")


def run(out_dir: Path) -> Dict[str, Any]:
    ensure_dir(out_dir)
    run_specs = [("c9_repeat", C9_REPEAT_DIR)] + _registry_runs()
    run_rows = [_run_summary(label, path) for label, path in run_specs]
    registry = _registry_crosscheck()
    ledger = _ledger_direction()
    gate = _gate(run_rows, registry, ledger)
    write_csv(out_dir / "tri_replay_actuator_trace.csv", run_rows)
    write_csv(out_dir / "tri_replay_hook_mapping.csv", PLAN_HOOK_MAPPING)
    write_json(out_dir / "tri_replay_registry_crosscheck.json", registry)
    write_json(out_dir / "tri_replay_c9_knockout_direction.json", ledger)
    write_json(out_dir / "tri_replay_actuator_summary.json", gate)
    _write_report(out_dir, gate, run_rows, ledger)
    return {"out_dir": rel(out_dir), **gate}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(V76_ROOT / "phase2_tri_replay_actuator_audit"))
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    result = run(Path(args.out_dir))
    write_json(Path(args.out_dir) / "command_result.json", result)
    if args.strict and not result["phase2_actuator_gate_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
