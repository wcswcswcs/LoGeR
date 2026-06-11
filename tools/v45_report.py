#!/usr/bin/env python3
"""Summarize landed ACL2 v45 artifacts without inventing missing results."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

HISTORICAL_C9_ATE = 33.7629421029
DEFAULT_V43_ROOT = Path("results/kitti01_hmc_v2/acl2_v43_c9_dechunk_attribution_semanticread_target30")


def _float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _clean(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    return value


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            out.append(value)
    return out


def _fmt(value: Any) -> str:
    val = _float(value)
    return "" if val is None else f"{val:.10f}"


def _registry(root: Path, phase: str) -> List[Dict[str, Any]]:
    candidates = [
        root / phase / "report_R1" / "full_online_registry.csv",
        root / phase / "report_R1" / "short_registry.csv",
        root / phase / "report_R1" / "full_metrics" / "full_online_registry.csv",
    ]
    for path in candidates:
        rows = _read_csv(path)
        if rows:
            return rows
    rows: List[Dict[str, Any]] = []
    report_root = root / phase / "report_R1"
    if report_root.exists():
        for path in sorted(report_root.glob("**/full_online_registry.csv")):
            for row in _read_csv(path):
                row["_registry_path"] = str(path)
                rows.append(row)
    return rows


def _registries(root: Path, phases: Iterable[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for phase in phases:
        for row in _registry(root, phase):
            row = dict(row)
            row.setdefault("_phase", phase)
            rows.append(row)
    return rows


def _row_by_name(rows: List[Dict[str, Any]], name: str) -> Optional[Dict[str, Any]]:
    return next((row for row in rows if row.get("name") == name), None)


def _done_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [row for row in rows if row.get("status") == "done"]


def _delta(row: Optional[Dict[str, Any]], ref_ate: float = HISTORICAL_C9_ATE) -> Optional[float]:
    ate = _float(row.get("ATE_full")) if row else None
    return ate - ref_ate if ate is not None else None


def _load_v43_effects(v43_root: Path) -> Dict[str, float]:
    rows = _read_csv(v43_root / "phase2_attribution" / "report_R1" / "full_online_registry.csv")
    effects: Dict[str, float] = {}
    mapping = {
        "ATTR_01": "read_beta_map",
        "ATTR_02": "tri_gamma_chunk_map",
        "ATTR_03": "commit_ema",
        "ATTR_04": "swa_overlap_replace",
        "ATTR_05": "tri_replay",
        "ATTR_06": "native_mix",
    }
    for row in rows:
        name = str(row.get("name", ""))
        for prefix, key in mapping.items():
            if name.startswith(prefix):
                val = _float(row.get("ATE_delta_vs_reference"))
                if val is None:
                    val = _delta(row)
                if val is not None:
                    effects[key] = val
    return effects


def _phase_summary(rows: List[Dict[str, Any]], reference_name: str = "F0") -> Dict[str, Any]:
    done = _done_rows(rows)
    ref = _row_by_name(done, reference_name)
    ref_ate = _float(ref.get("ATE_full")) if ref else None
    candidates = [row for row in done if row.get("name") != reference_name]
    best = min(candidates, key=lambda r: _float(r.get("ATE_full")) or float("inf"), default=None)
    best_ate = _float(best.get("ATE_full")) if best else None
    return {
        "rows": len(rows),
        "done_rows": len(done),
        "reference_name": reference_name,
        "reference_ATE_full": ref_ate,
        "best_candidate": best.get("name") if best else None,
        "best_ATE_full": best_ate,
        "best_delta_vs_reference": (best_ate - ref_ate) if best_ate is not None and ref_ate is not None else None,
        "best_delta_vs_historical_c9": (best_ate - HISTORICAL_C9_ATE) if best_ate is not None else None,
    }


def _write_table(lines: List[str], rows: List[Dict[str, Any]], columns: List[Tuple[str, str]]) -> None:
    lines.append("| " + " | ".join(title for title, _ in columns) + " |")
    lines.append("|" + "|".join("---" for _ in columns) + "|")
    for row in rows:
        values = []
        for _title, key in columns:
            value = row.get(key)
            values.append(_fmt(value) if isinstance(value, (float, int)) or _float(value) is not None else str(value if value is not None else ""))
        lines.append("| " + " | ".join(values) + " |")


def _write_c9_clean_report(out_dir: Path, phase1_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary = _phase_summary(phase1_rows)
    rows = _done_rows(phase1_rows)
    d7 = next((r for r in rows if str(r.get("name", "")).startswith("D7")), None)
    d7_ate = _float(d7.get("ATE_full")) if d7 else None
    d7_delta = (d7_ate - HISTORICAL_C9_ATE) if d7_ate is not None else None
    summary.update({
        "c9_clean_candidate": d7.get("name") if d7 else None,
        "c9_clean_ATE": d7_ate,
        "c9_clean_delta_vs_C9": d7_delta,
        "acceptable": bool(d7_delta is not None and d7_delta <= 0.30),
        "promising": bool(d7_delta is not None and d7_delta <= 0.10),
        "success": bool(d7_delta is not None and d7_delta <= 0.0),
    })
    _write_json(out_dir / "v45_c9_clean_summary.json", summary)
    lines = ["# v45 C9 Clean Report", "", "## Summary", ""]
    for key, value in summary.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Rows", ""])
    _write_table(
        lines,
        rows,
        [
            ("Name", "name"),
            ("ATE", "ATE_full"),
            ("Delta vs C9", "delta_vs_historical_c9_ATE"),
            ("[200,300)", "segment_200_300_ATE"),
            ("[400,600)", "segment_400_600_ATE"),
            ("hmc_rows", "hmc_rows"),
        ],
    )
    lines.append("")
    (out_dir / "v45_c9_clean_report.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def _write_interaction(out_dir: Path, phase1_rows: List[Dict[str, Any]], phase2_rows: List[Dict[str, Any]], v43_effects: Dict[str, float]) -> List[Dict[str, Any]]:
    phase1_by_prefix = {str(r.get("name", "")): r for r in _done_rows(phase1_rows)}
    fixed_gamma_candidates = [r for r in _done_rows(phase1_rows) if str(r.get("name", "")).startswith(("D2", "D3", "D4"))]
    fixed_gamma_best = min(fixed_gamma_candidates, key=lambda r: _float(r.get("ATE_full")) or float("inf"), default=None)
    effects = dict(v43_effects)
    if fixed_gamma_best is not None:
        effects["fixed_tri_gamma_best"] = _delta(fixed_gamma_best) or effects.get("tri_gamma_chunk_map", float("nan"))
    d1 = next((r for r in phase1_by_prefix.values() if str(r.get("name", "")).startswith("D1")), None)
    if d1 is not None:
        effects["fixed_read_beta"] = _delta(d1) or effects.get("read_beta_map", float("nan"))
    d5 = next((r for r in phase1_by_prefix.values() if str(r.get("name", "")).startswith("D5")), None)
    d6 = next((r for r in phase1_by_prefix.values() if str(r.get("name", "")).startswith("D6")), None)
    if d5 is not None:
        effects["fixed_ema_best"] = _delta(d5) or effects.get("commit_ema", float("nan"))
    if d6 is not None:
        d6_delta = _delta(d6)
        if d6_delta is not None and abs(d6_delta) < abs(effects.get("fixed_ema_best", float("inf"))):
            effects["fixed_ema_best"] = d6_delta

    specs = {
        "I1": ("tri_replay", "commit_ema"),
        "I2": ("tri_replay", "swa_overlap_replace"),
        "I3": ("tri_replay", "native_mix"),
        "I4": ("fixed_tri_gamma_best", "commit_ema"),
        "I5": ("fixed_tri_gamma_best", "swa_overlap_replace"),
        "I6": ("fixed_tri_gamma_best", "native_mix"),
        "I7": ("fixed_read_beta", "fixed_tri_gamma_best"),
        "I8": ("fixed_read_beta", "fixed_tri_gamma_best", "fixed_ema_best"),
    }
    out_rows: List[Dict[str, Any]] = []
    for row in _done_rows(phase2_rows):
        name = str(row.get("name", ""))
        key = name.split("_", 1)[0]
        if key not in specs:
            continue
        ate = _float(row.get("ATE_full"))
        delta = ate - HISTORICAL_C9_ATE if ate is not None else None
        comps = specs[key]
        expected = sum(effects.get(comp, float("nan")) for comp in comps)
        interaction = delta - expected if delta is not None and math.isfinite(expected) else None
        out_rows.append({
            "candidate": name,
            "components": "+".join(comps),
            "ATE_full": ate,
            "delta_vs_C9": delta,
            "expected_additive_delta": expected if math.isfinite(expected) else None,
            "interaction_residual": interaction,
            "non_additive_abs_gt_0p3": bool(interaction is not None and abs(interaction) > 0.3),
        })
    _write_csv(out_dir / "v45_component_interaction_matrix.csv", out_rows)
    ledger_rows = [{"component": key, "effect_delta_vs_C9": value} for key, value in sorted(effects.items())]
    _write_csv(out_dir / "v45_component_contribution_ledger.csv", ledger_rows)
    lines = ["# v45 Component Interaction", "", "## Ledger", ""]
    _write_table(lines, ledger_rows, [("Component", "component"), ("Effect delta", "effect_delta_vs_C9")])
    lines.extend(["", "## Interactions", ""])
    _write_table(lines, out_rows, [("Candidate", "candidate"), ("Components", "components"), ("ATE", "ATE_full"), ("Delta", "delta_vs_C9"), ("Additive", "expected_additive_delta"), ("Residual", "interaction_residual"), ("|I| > 0.3", "non_additive_abs_gt_0p3")])
    lines.append("")
    (out_dir / "v45_component_interaction_report.md").write_text("\n".join(lines), encoding="utf-8")
    return out_rows


def _role_mass_summary(run_dir: Path) -> Dict[str, Any]:
    rows = _read_jsonl(run_dir / "v11_projection_trace" / "tri_replay_role_mass.jsonl")
    if not rows:
        rows = _read_jsonl(run_dir / "tri_replay_role_mass.jsonl")
    out: Dict[str, Any] = {"role_rows": len(rows)}
    for key in ("positive_mass", "neutral_mass", "negative_mass"):
        vals = [_float(r.get(key)) for r in rows]
        vals = [v for v in vals if v is not None]
        out[f"{key}_mean"] = sum(vals) / len(vals) if vals else None
    out["fallback_rows"] = sum(1 for r in rows if r.get("role_fallback") or r.get("ttt_tri_replay_role_fallback"))
    return out


def _write_simple_phase_report(out_dir: Path, filename: str, title: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary = _phase_summary(rows)
    _write_json(out_dir / filename.replace(".md", "_summary.json"), summary)
    lines = [f"# {title}", "", "## Summary", ""]
    for key, value in summary.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Rows", ""])
    _write_table(lines, _done_rows(rows), [("Name", "name"), ("ATE", "ATE_full"), ("Delta vs ref", "ATE_delta_vs_reference"), ("Delta vs C9", "delta_vs_historical_c9_ATE"), ("[200,300)", "segment_200_300_ATE"), ("[400,600)", "segment_400_600_ATE")])
    lines.append("")
    (out_dir / filename).write_text("\n".join(lines), encoding="utf-8")
    return summary


def _maybe_plots(out_dir: Path, phase1_rows: List[Dict[str, Any]], phase2_interactions: List[Dict[str, Any]], phase3_rows: List[Dict[str, Any]], phase4_rows: List[Dict[str, Any]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return

    def bar(rows: List[Dict[str, Any]], path: str, title: str) -> None:
        done = _done_rows(rows)
        if not done:
            return
        names = [str(r.get("name")) for r in done]
        vals = [_float(r.get("ATE_full")) or float("nan") for r in done]
        fig, ax = plt.subplots(figsize=(max(8, len(done) * 0.8), 4))
        ax.bar(range(len(vals)), vals, color="#4c78a8")
        ax.axhline(HISTORICAL_C9_ATE, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(title)
        ax.set_ylabel("ATE full (m)")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right")
        fig.tight_layout()
        fig.savefig(out_dir / path, dpi=160)
        plt.close(fig)

    bar(phase1_rows, "c9_clean_metric_bar.png", "v45 C9 clean")
    bar(phase3_rows, "support_variant_ate_bar.png", "v45 C23 support")
    bar(phase4_rows, "adaptive_tri_ate_bar.png", "v45 adaptive tri")

    if phase2_interactions:
        names = [r["candidate"] for r in phase2_interactions]
        vals = [(_float(r.get("interaction_residual")) or 0.0) for r in phase2_interactions]
        fig, ax = plt.subplots(figsize=(max(8, len(vals) * 0.7), 3.6))
        ax.imshow(np.asarray([vals]), aspect="auto", cmap="coolwarm")
        ax.set_yticks([0])
        ax.set_yticklabels(["I"])
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right")
        fig.tight_layout()
        fig.savefig(out_dir / "component_interaction_heatmap.png", dpi=160)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument("--v43-root", default=DEFAULT_V43_ROOT, type=Path)
    args = parser.parse_args()

    root = args.result_root
    out_dir = root / "final_reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    phase1 = _registry(root, "phase1_c9_clean")
    phase2 = _registry(root, "phase2_interaction")
    phase3 = _registry(root, "phase3_c23_support")
    phase4 = _registry(root, "phase4_adaptive_trireplay")
    phase5 = _registries(
        root,
        [
            "phase5_semantic_read",
            "phase5_semantic_read_extra",
            "phase5_semantic_read_extra2",
            "phase5_semantic_read_extra3",
            "phase5_semantic_read_extra4",
        ],
    )

    c9_clean = _write_c9_clean_report(out_dir, phase1)
    interactions = _write_interaction(out_dir, phase1, phase2, _load_v43_effects(args.v43_root))
    support = _write_simple_phase_report(out_dir, "v45_c23_support_dashboard.md", "v45 C23 Support Dashboard", phase3)
    adaptive = _write_simple_phase_report(out_dir, "v45_adaptive_trireplay_report.md", "v45 Adaptive Tri Replay Report", phase4)
    semantic = _write_simple_phase_report(out_dir, "v45_semantic_read_minimal_report.md", "v45 Semantic READ Minimal/Continuation Report", phase5)

    role_rows: List[Dict[str, Any]] = []
    for row in _done_rows(phase4):
        run_dir = Path(str(row.get("run_dir", "")))
        if run_dir.exists():
            role_rows.append({"name": row.get("name"), **_role_mass_summary(run_dir)})
    _write_csv(out_dir / "v45_adaptive_role_mass_summary.csv", role_rows)

    code_audit = {
        "ttt_two_replay_dead_code_removed": True,
        "tri_replay_role_mode_added": True,
        "full_chunk_no_overlap_alias_confirmed": True,
        "stage_c_semantic_default_off_in_v45_launcher": True,
    }
    _write_json(out_dir / "v45_code_audit_update.json", code_audit)
    (out_dir / "v45_code_audit_update.md").write_text(
        "\n".join([
            "# v45 Code Audit Update",
            "",
            "- TTT non-tri replay now returns through an explicit two-replay branch.",
            "- The unreachable legacy block after tri_replay return was removed.",
            "- Added `ttt_write_tri_replay_role_mode`; default `fixed` preserves C9 behavior.",
            "- v45 launcher defaults Stage C semantic to `none` unless a semantic candidate enables it.",
            "",
        ]),
        encoding="utf-8",
    )

    phase_summaries = [support, adaptive, semantic]
    best_ates = [_float(s.get("best_ATE_full")) for s in phase_summaries]
    best_deltas = [_float(s.get("best_delta_vs_historical_c9")) for s in phase_summaries]
    final = {
        "c9_clean": c9_clean,
        "support": support,
        "adaptive": adaptive,
        "semantic": semantic,
        "target30_success": any(bool(s.get("target30_success_pass") or s.get("target30_success")) for s in [support, adaptive, semantic]),
        "stage_progress": any(v is not None and v <= 33.0 for v in best_ates),
        "minimum_progress": any(
            (v is not None and v <= 33.3) or (d is not None and d <= -0.5)
            for v, d in zip(best_ates, best_deltas)
        ),
        "mechanism_progress": bool(interactions or c9_clean.get("c9_clean_candidate")),
        "phase6_sanity_recommended": bool(
            (c9_clean.get("acceptable") and interactions)
            or any(v is not None and v <= 33.0 for v in best_ates)
            or any(d is not None and d <= -0.5 for d in best_deltas)
        ),
    }
    _write_json(out_dir / "v45_final_decision.json", final)
    lines = ["# v45 Final Decision", ""]
    for key, value in final.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend([
        "",
        "## Required Questions",
        "",
        f"1. C9 no-chunk-id close to original: `{c9_clean.get('acceptable')}`.",
        f"2. Component interaction rows available: `{len(interactions)}`.",
        f"3. C23 support best: `{support.get('best_candidate')}` / ATE `{support.get('best_ATE_full')}`.",
        f"4. Adaptive tri best: `{adaptive.get('best_candidate')}` / ATE `{adaptive.get('best_ATE_full')}`.",
        f"5. Semantic READ best: `{semantic.get('best_candidate')}` / ATE `{semantic.get('best_ATE_full')}`.",
        f"6. Cross-sequence sanity recommended: `{final.get('phase6_sanity_recommended')}`.",
        "",
    ])
    (out_dir / "v45_final_decision.md").write_text("\n".join(lines), encoding="utf-8")

    _maybe_plots(out_dir, phase1, interactions, phase3, phase4)
    print(json.dumps(_clean(final), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
