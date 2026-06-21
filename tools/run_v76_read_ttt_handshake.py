#!/usr/bin/env python3
"""Audit ACL2 v76 Phase4 READ + TTT tri-replay handshake evidence."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, Mapping, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.v76tf_common import (  # noqa: E402
    V46B_REGISTRY,
    V76_ROOT,
    boolish,
    ensure_dir,
    first_row,
    read_csv,
    read_json,
    rel,
    safe_float,
    write_csv,
    write_json,
    write_text,
)


DEFAULT_SMOKE_REGISTRY = V76_ROOT / "phase4_c9clean_semantic_trireplay_smoke/report_R1/full_online_registry.csv"


def _gain(reference: Optional[float], candidate: Optional[float]) -> Optional[float]:
    if reference is None or candidate is None:
        return None
    return reference - candidate


def _chunk_policy_free(run_dir_text: Any) -> Optional[bool]:
    if not run_dir_text:
        return None
    audit = read_json(REPO_ROOT / str(run_dir_text) / "chunk_id_policy_audit.json")
    if not isinstance(audit, dict):
        return None
    return not any(
        boolish(audit.get(key))
        for key in (
            "has_read_beta_frame_chunks",
            "has_tri_gamma_chunk_map",
            "has_tri_replay_chunk_params",
            "has_commit_ema_chunks",
        )
    )


def _ate(row: Optional[Mapping[str, Any]]) -> Optional[float]:
    if not row:
        return None
    return safe_float(row.get("ATE_full"))


def _row_by_name(rows: Iterable[Mapping[str, Any]], *names: str) -> Optional[Mapping[str, Any]]:
    for name in names:
        row = first_row(rows, "name", name)
        if row is not None:
            return row
        row = first_row(rows, "row", name)
        if row is not None:
            return row
    return None


def _start_a_rows() -> Dict[str, Any]:
    rows = read_csv(V46B_REGISTRY)
    base = _row_by_name(rows, "F000_NONE")
    read = _row_by_name(rows, "F100_ONLY_FRAME_ATTN")
    ttt = _row_by_name(rows, "F010_ONLY_TTT")
    combo = _row_by_name(rows, "F110_FRAME_ATTN_TTT")
    all_three = _row_by_name(rows, "F111_ALL_THREE")
    base_ate = _ate(base)
    read_ate = _ate(read)
    ttt_ate = _ate(ttt)
    combo_ate = _ate(combo)
    all_ate = _ate(all_three)
    best_single = min([x for x in (read_ate, ttt_ate) if x is not None], default=None)
    combo_gain = _gain(base_ate, combo_ate)
    best_single_gain = _gain(base_ate, best_single)
    synergy = combo_gain - best_single_gain if combo_gain is not None and best_single_gain is not None else None
    return {
        "scope": "start_a_clean_h35_full_imported_v46b",
        "artifact": rel(V46B_REGISTRY),
        "baseline": "F000_NONE",
        "read_only": "F100_ONLY_FRAME_ATTN",
        "ttt_only": "F010_ONLY_TTT",
        "read_ttt": "F110_FRAME_ATTN_TTT",
        "read_ttt_swa": "F111_ALL_THREE",
        "baseline_ATE_full": base_ate,
        "read_only_ATE_full": read_ate,
        "ttt_only_ATE_full": ttt_ate,
        "read_ttt_ATE_full": combo_ate,
        "read_ttt_swa_ATE_full": all_ate,
        "read_gain_m": _gain(base_ate, read_ate),
        "ttt_gain_m": _gain(base_ate, ttt_ate),
        "read_ttt_gain_m": combo_gain,
        "read_ttt_synergy_over_best_single_m": synergy,
        "read_ttt_beats_read_and_ttt": bool(
            combo_ate is not None
            and read_ate is not None
            and ttt_ate is not None
            and combo_ate < read_ate
            and combo_ate < ttt_ate
        ),
        "diagnostic_scope": "imported full-run positive-only factorial; semantic-free; no shuffled controls/alignment metric",
    }


def _smoke_rows(registry: Path) -> Dict[str, Any]:
    rows = read_csv(registry)
    base = _row_by_name(rows, "D7", "V76_C9CLEAN_D7_BASE_256F")
    geom = _row_by_name(rows, "A4", "V76_C9CLEAN_A4_ADAPTIVE_TRI_256F")
    read = _row_by_name(rows, "SEM2", "V76_C9CLEAN_SEM2_READ_L050_256F")
    combo = _row_by_name(rows, "SEM4", "V76_C9CLEAN_SEM4_READ_ADAPTIVE_TRI_L050_256F")
    base_ate = _ate(base)
    geom_ate = _ate(geom)
    read_ate = _ate(read)
    combo_ate = _ate(combo)
    policies = {
        "baseline_chunk_policy_free": _chunk_policy_free(base.get("run_dir") if base else None),
        "geometry_tri_chunk_policy_free": _chunk_policy_free(geom.get("run_dir") if geom else None),
        "semantic_read_chunk_policy_free": _chunk_policy_free(read.get("run_dir") if read else None),
        "semantic_read_ttt_chunk_policy_free": _chunk_policy_free(combo.get("run_dir") if combo else None),
    }
    beats = bool(
        combo_ate is not None
        and base_ate is not None
        and geom_ate is not None
        and read_ate is not None
        and combo_ate < base_ate
        and combo_ate < geom_ate
        and combo_ate < read_ate
    )
    policy_free = all(value is True for value in policies.values())
    return {
        "scope": "phase4_c9clean_256f_current_smoke",
        "artifact": rel(registry),
        "registry_available": registry.exists(),
        "baseline_ATE_full": base_ate,
        "geometry_adaptive_tri_ATE_full": geom_ate,
        "semantic_read_ATE_full": read_ate,
        "semantic_read_ttt_ATE_full": combo_ate,
        "semantic_read_ttt_gain_vs_baseline_m": _gain(base_ate, combo_ate),
        "semantic_read_ttt_gain_vs_geometry_tri_m": _gain(geom_ate, combo_ate),
        "semantic_read_ttt_gain_vs_semantic_read_m": _gain(read_ate, combo_ate),
        "semantic_read_ttt_beats_smoke_tracks": beats,
        "all_smoke_runs_chunk_policy_free": policy_free,
        **policies,
        "counts_as_strict_v76_success": False,
        "diagnostic_scope": (
            "current 256F C9-clean smoke only; diagnostic reference, not H35-base; "
            "not 704F/full; no shuffled controls/alignment metric"
        ),
    }


def run(out_dir: Path, smoke_registry: Path) -> Dict[str, Any]:
    ensure_dir(out_dir)
    start_a = _start_a_rows()
    smoke = _smoke_rows(smoke_registry)
    rows = [start_a, smoke]
    summary = {
        "phase4_start_a_factorial_handshake_signal": bool(start_a.get("read_ttt_beats_read_and_ttt")),
        "phase4_start_a_read_ttt_synergy_m": start_a.get("read_ttt_synergy_over_best_single_m"),
        "phase4_current_c9clean_smoke_available": bool(smoke.get("registry_available")),
        "phase4_current_c9clean_smoke_direction_pass": bool(
            smoke.get("semantic_read_ttt_beats_smoke_tracks")
            and smoke.get("all_smoke_runs_chunk_policy_free")
        ),
        "phase4_current_c9clean_smoke_counts_as_strict_success": False,
        "phase4_required_success_base": "Clean H35/v53; C9/C9-clean/dechunk are diagnostic references only",
        "phase4_current_c9clean_smoke_gain_vs_baseline_m": smoke.get("semantic_read_ttt_gain_vs_baseline_m"),
        "phase4_full_gate_pass": False,
        "phase4_full_gate_reason": (
            "Phase4 has Start A full-run READ+TTT positive signal and current C9-clean 256F smoke if available, "
            "but strict success must be H35-base and still lacks 704F/full deployable semantic candidate with "
            "shuffled/random controls and READ->TTT alignment metric."
        ),
    }
    write_csv(out_dir / "phase4_read_ttt_handshake_rows.csv", rows)
    write_json(out_dir / "phase4_read_ttt_handshake_summary.json", summary)
    _write_report(out_dir, rows, summary)
    return {"out_dir": rel(out_dir), **summary}


def _write_report(out_dir: Path, rows: list[Mapping[str, Any]], summary: Mapping[str, Any]) -> None:
    lines = [
        "# v76 Phase4 READ-TTT Handshake Audit",
        "",
        "This audit separates full imported Start A factorial evidence from the current C9-clean 256F smoke.",
        "",
        "## Summary",
        "",
    ]
    for key, value in summary.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Rows", "", "| scope | artifact | key result | diagnostic scope |", "|---|---|---|---|"])
    for row in rows:
        if row["scope"].startswith("start_a"):
            key = f"READ+TTT synergy {row.get('read_ttt_synergy_over_best_single_m')}"
        else:
            key = f"smoke gain vs baseline {row.get('semantic_read_ttt_gain_vs_baseline_m')}"
        lines.append(f"| `{row.get('scope')}` | `{row.get('artifact')}` | `{key}` | {row.get('diagnostic_scope')} |")
    lines.append("")
    write_text(out_dir / "phase4_read_ttt_handshake_report.md", "\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(V76_ROOT / "phase4_read_ttt_handshake"))
    parser.add_argument("--smoke-registry", default=str(DEFAULT_SMOKE_REGISTRY))
    args = parser.parse_args()
    result = run(Path(args.out_dir), Path(args.smoke_registry))
    write_json(Path(args.out_dir) / "command_result.json", result)


if __name__ == "__main__":
    main()
