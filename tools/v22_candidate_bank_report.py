#!/usr/bin/env python3
"""Aggregate ACL2 v22 short-rollout metrics and durability gates."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import v18_true_action_report as impl  # noqa: E402


impl.FAMILY_BY_CANDIDATE.update(
    {
        "K1_H9": "baseline",
        "S0_C23_PAST_LOCKED": "support_true",
        "S1_C23_FULL_CHUNK_TRUE": "support_true",
        "S2_C23_FULL_CHUNK_NO_OVERLAP_TRUE": "support_true",
        "S3_C23_PAST_PLUS_NEAR_FUTURE12": "support_true",
        "S4_C23_PAST_PLUS_FUTURE_LIGHT_REAL": "support_true",
        "KVC_01_FRAME_EARLY_DG_Q80_COMPACT": "compact_kv",
        "KVC_02_FRAME_EARLY_DG_Q90_COMPACT": "compact_kv",
        "KVC_03_FRAME_EARLY_LOWSTUFF_HIGHD_COMPACT": "compact_kv_semantic_coarse",
        "KVC_04_GLOBAL_EARLY_DG_Q80_COMPACT": "compact_kv_global",
        "KVC_05_FRAME_GLOBAL_EARLY_DG_Q80_COMPACT": "compact_kv_both",
        "KVC_06_FRAME_EARLY_DG_Q80_BIAS_REPEAT": "bias_repeat",
        "KVC_08_FRAME_EARLY_DG_Q80_COMPACT_WITH_STATIC_RESCUE": "compact_kv_static_rescue",
        "SEMFA_04_LOWSTUFF_HIGHD_FRAME_EARLY_COMPACT": "semantic_coarse_source_skip",
        "SEMFA_05_STRUCTURE_RESCUE_DGQ80_FRAME_EARLY_COMPACT": "semantic_coarse_source_skip",
        "TTTSSP_01_SCALECOMMIT_DGQ80_COMPACT": "scale_state_compact_combo",
        "TTTSSP_02_SCALECOMMIT_DGQ80_STRUCTURE_RESCUE_COMPACT": "scale_state_compact_combo",
        "SUP_LOCKED_A": "support_setting_A",
        "SUP_FULL_TRUE_A": "support_setting_A",
        "SUP_NO_OVERLAP_A": "support_setting_A",
        "SUP_PAST_NEAR_FUTURE12_A": "support_setting_A",
        "SUP_LOCKED_B_DGQ80_COMPACT": "support_setting_B_compact",
        "SUP_FULL_TRUE_B_DGQ80_COMPACT": "support_setting_B_compact",
        "SUP_NO_OVERLAP_B_DGQ80_COMPACT": "support_setting_B_compact",
        "SUP_PAST_NEAR_FUTURE12_B_DGQ80_COMPACT": "support_setting_B_compact",
        "SUP_LOCKED_C_STRUCTURE_RESCUE_COMPACT": "support_setting_C_semantic_rescue",
        "SUP_FULL_TRUE_C_STRUCTURE_RESCUE_COMPACT": "support_setting_C_semantic_rescue",
        "SUP_NO_OVERLAP_C_STRUCTURE_RESCUE_COMPACT": "support_setting_C_semantic_rescue",
        "SUP_PAST_NEAR_FUTURE12_C_STRUCTURE_RESCUE_COMPACT": "support_setting_C_semantic_rescue",
        "KVC_READ_01": "read_only_compact_kv",
        "KVC_READ_02": "read_only_compact_kv",
        "KVC_READ_03": "read_only_compact_kv",
        "KVC_READ_04": "read_only_compact_kv",
        "SEM_ROLE_01_STRUCTURE_RESCUE": "semantic_coarse_role",
        "SEM_ROLE_02_LOWSTUFF_HIGHD_SKIP": "semantic_coarse_role",
        "SEM_ROLE_04_STRUCTURE_POSITIVE_TTT": "semantic_all_memory_role",
        "SEM_ROLE_05_ALL_MEMORY_ROLE": "semantic_all_memory_role",
        "KVC_TTT_01_NEUTRAL_COMMIT_FILTER": "skip_aware_ttt_write",
        "KVC_TTT_02_WEAK_NEGATIVE": "skip_aware_ttt_write",
        "KVC_TTT_03_STRUCTURE_KEPT_BOOST": "skip_aware_ttt_write",
        "KVC_TTT_04_SOURCE_KEEP_GATED_WRITE": "skip_aware_ttt_write",
        "KVC_MEM_01_SWA_COMPACT_OVERLAP_HISTORY": "skip_aware_swa_memory",
        "KVC_MEM_02_SWA_DOWNWEIGHT_SKIPPED": "skip_aware_swa_memory",
        "KVC_MEM_03_GLOBAL_CHUNK_SOURCE_SKIP": "skip_aware_global_source",
        "KVC_MEM_04_TTT_AND_SWA_DOWNWEIGHT": "skip_aware_swa_memory",
        "TTT_DUR_01_READ_COMPACT_ONLY": "ttt_durability_control",
        "TTT_DUR_02_SKIP_AWARE_COMMIT_FILTER": "ttt_durable_commit",
        "TTT_DUR_03_NATIVE_READ_SKIP_REPLAY_ONLY": "ttt_durable_commit",
        "TTT_DUR_04_POST_ZP_SKIP_BASIS_ROUTING": "ttt_durable_commit",
        "TTT_LIFE_01_SHORT_HIGHD_K2": "ttt_lifecycle_split",
        "TTT_LIFE_02_SHORT_HIGHD_K4": "ttt_lifecycle_split",
        "TTT_LIFE_03_LOWSTUFF_SHORT_STRUCTURE_LONG": "ttt_lifecycle_split",
        "TTT_LIFE_04_SCALE_LONG_HIGHD_SHORT": "ttt_lifecycle_split",
    }
)


def _arg_value(name: str) -> str | None:
    if name not in sys.argv:
        return None
    idx = sys.argv.index(name)
    if idx + 1 >= len(sys.argv):
        return None
    return sys.argv[idx + 1]


def _to_float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, rows: List[Dict[str, object]]) -> None:
    clean = []
    for row in rows:
        clean.append({
            key: (None if isinstance(value, float) and math.isnan(value) else value)
            for key, value in row.items()
        })
    path.write_text(json.dumps(clean, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _postprocess_outputs(out_dir: Path) -> None:
    md_old = out_dir / "acl2_v18_true_action_report.md"
    md_new = out_dir / "acl2_v22_candidate_bank_report.md"
    if md_old.exists():
        text = md_old.read_text(encoding="utf-8")
        text = text.replace(
            "# ACL2 v18 True Action Candidate Report",
            "# ACL2 v22 Durable ContextSkip / SemanticAllMemory Candidate Report",
        )
        text = text.replace("v18 true-action", "v22 short-rollout")
        md_new.write_text(text, encoding="utf-8")

    rows = _read_csv(out_dir / "candidate_vs_H9_delta_by_horizon.csv")
    grouped: Dict[tuple[str, int], Dict[int, Mapping[str, str]]] = {}
    for row in rows:
        candidate = row.get("candidate_id", "")
        if candidate == "K1_H9":
            continue
        try:
            chunk = int(row.get("chunk_id", ""))
            horizon = int(row.get("horizon", ""))
        except ValueError:
            continue
        grouped.setdefault((candidate, chunk), {})[horizon] = row

    durability_rows: List[Dict[str, object]] = []
    durable_gate_pass = False
    best_durable: Dict[str, object] | None = None
    for (candidate, chunk), by_h in sorted(grouped.items()):
        h10 = by_h.get(10)
        h15 = by_h.get(15)
        if h10 is None and h15 is None:
            continue
        h10_ate = _to_float(h10.get("ATE_delta_vs_H9")) if h10 else float("nan")
        h15_ate = _to_float(h15.get("ATE_delta_vs_H9")) if h15 else float("nan")
        h10_seg = _to_float(h10.get("intersection_200_300_delta_vs_H9")) if h10 else float("nan")
        h15_seg = _to_float(h15.get("intersection_200_300_delta_vs_H9")) if h15 else float("nan")
        h10_down = _to_float(h10.get("intersection_400_600_delta_vs_H9")) if h10 else float("nan")
        h15_down = _to_float(h15.get("intersection_400_600_delta_vs_H9")) if h15 else float("nan")
        durability = (
            abs(h15_ate) / (abs(h10_ate) + 1e-8)
            if math.isfinite(h10_ate) and math.isfinite(h15_ate)
            else float("nan")
        )
        local_gate = (
            (math.isfinite(h10_ate) and h10_ate <= -3.0)
            or (math.isfinite(h10_seg) and h10_seg <= -5.0)
            or (math.isfinite(h15_ate) and h15_ate <= -3.0)
            or (math.isfinite(h15_seg) and h15_seg <= -5.0)
        )
        downstream_ok = (
            (not math.isfinite(h10_down) or h10_down <= 1.0)
            and (not math.isfinite(h15_down) or h15_down <= 1.0)
        )
        durable_ok = math.isfinite(durability) and durability >= 0.45
        row_out = {
            "candidate_id": candidate,
            "family": impl.FAMILY_BY_CANDIDATE.get(candidate, "unknown"),
            "chunk_id": chunk,
            "h10_ATE_delta_vs_H9": h10_ate,
            "h15_ATE_delta_vs_H9": h15_ate,
            "h10_200_300_delta_vs_H9": h10_seg,
            "h15_200_300_delta_vs_H9": h15_seg,
            "h10_400_600_delta_vs_H9": h10_down,
            "h15_400_600_delta_vs_H9": h15_down,
            "durability_abs_h15_over_h10": durability,
            "durability_pass": durable_ok,
            "local_gate_pass": local_gate,
            "downstream_gate_pass": downstream_ok,
            "selector_allowed": bool(local_gate and downstream_ok and durable_ok),
            "counts_as_ttt_write_success": False,
        }
        if row_out["selector_allowed"]:
            durable_gate_pass = True
            if best_durable is None or _to_float(row_out["h15_ATE_delta_vs_H9"]) < _to_float(best_durable["h15_ATE_delta_vs_H9"]):
                best_durable = row_out
        durability_rows.append(row_out)

    _write_csv(out_dir / "durability_by_candidate_chunk.csv", durability_rows)
    _write_json(out_dir / "durability_by_candidate_chunk.json", durability_rows)

    summary_path = out_dir / "true_action_gate_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        summary = [{}]
    if summary:
        summary[0]["phase"] = "v22 durable contextskip semantic-all-memory candidate bank"
        summary[0]["durability_gate_rule"] = "|delta_h15| / (|delta_h10| + eps) >= 0.45"
        summary[0]["durability_gate_pass"] = durable_gate_pass
        summary[0]["selector_allowed_before_durability"] = bool(summary[0].get("selector_allowed", False))
        summary[0]["selector_allowed"] = bool(summary[0].get("selector_allowed", False) and durable_gate_pass)
        summary[0]["full_online_validation_allowed"] = False
        if best_durable is not None:
            summary[0]["best_durable_candidate"] = best_durable["candidate_id"]
            summary[0]["best_durable_chunk"] = best_durable["chunk_id"]
            summary[0]["best_durable_h10_ATE_delta_vs_H9"] = best_durable["h10_ATE_delta_vs_H9"]
            summary[0]["best_durable_h15_ATE_delta_vs_H9"] = best_durable["h15_ATE_delta_vs_H9"]
            summary[0]["best_durable_ratio"] = best_durable["durability_abs_h15_over_h10"]
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_csv(out_dir / "true_action_gate_summary.csv", summary)

    if md_new.exists():
        extra = [
            "",
            "## Durability Gate",
            "",
            "Selector/full validation is allowed only when the short-rollout gate and the h10->h15 durability gate both pass.",
            "",
            f"Durability gate pass: `{str(durable_gate_pass).lower()}`",
            "",
        ]
        md_new.write_text(md_new.read_text(encoding="utf-8") + "\n".join(extra) + "\n", encoding="utf-8")


def main() -> None:
    out_dir_text = _arg_value("--out-dir")
    impl.main()
    if out_dir_text:
        _postprocess_outputs(Path(out_dir_text))


if __name__ == "__main__":
    main()
