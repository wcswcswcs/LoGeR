#!/usr/bin/env python3
"""Build ACL2 v84 Phase11 rediscovery question and visual bundle."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


DEFAULT_CANDIDATE_DIR = Path("results/acl2_v84tf_memory_ruler_audit/phase10_support_expansion_candidates")
DEFAULT_AUDIT_DIR = Path("results/acl2_v84tf_memory_ruler_audit/phase10_support_expansion_audit")
DEFAULT_OUT_DIR = Path("results/acl2_v84tf_memory_ruler_audit/phase11_memory_ruler_rediscovery")
GRID = (19, 66)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_CANDIDATE_DIR)
    parser.add_argument("--phase10-audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-panels-per-category", type=int, default=4)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: serialize(row.get(key, "")) for key in fields})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def serialize(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.12g}"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def safe_float(value: Any) -> float | None:
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    try:
        out = float(text)
    except ValueError:
        return None
    return out if math.isfinite(out) else None


def pair_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return str(row.get("seq", "")), str(row.get("prev_chunk", "")), str(row.get("curr_chunk", ""))


def panel_name(row: Mapping[str, Any], category: str) -> str:
    return f"{category}_seq{row.get('seq')}_c{int(float(row.get('prev_chunk') or 0)):03d}_{int(float(row.get('curr_chunk') or 0)):03d}.png"


def select_cases(rows: list[dict[str, str]], max_per_category: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []

    def add(category: str, question: str, candidates: list[dict[str, str]]) -> None:
        for row in candidates[:max_per_category]:
            item = dict(row)
            item["visual_category"] = category
            item["visual_question"] = question
            selected.append(item)

    labelled_bad = [row for row in rows if row.get("base_case_type") == "bad"]
    labelled_good = [row for row in rows if row.get("base_case_type") == "good"]
    support = [row for row in rows if row.get("base_case_type") == "unlabelled_support"]

    add(
        "bad_ruler_absence",
        "Are bad cases missing ruler anchors, or are anchors present but not used by memory?",
        sorted(
            [row for row in labelled_bad if (safe_float(row.get("ruler_anchor_count")) or 0.0) <= 0],
            key=lambda row: (row.get("seq", ""), safe_float(row.get("zero_conf_ratio")) or 0.0),
            reverse=True,
        ),
    )
    add(
        "bad_ruler_contradiction",
        "When anchors exist, do they form conflicting distance-ratio clusters or just sparse support?",
        sorted(
            [
                row
                for row in rows
                if "contradiction_observable" in str(row.get("failure_flags") or "")
                and (safe_float(row.get("ruler_anchor_count")) or 0.0) >= 2
            ],
            key=lambda row: safe_float(row.get("ruler_anchor_count")) or 0.0,
            reverse=True,
        ),
    )
    add(
        "good_false_positive",
        "Which good-protection clue is missing when good/false-positive rows also contain anchors?",
        sorted(
            [row for row in labelled_good if (safe_float(row.get("ruler_anchor_count")) or 0.0) > 0],
            key=lambda row: safe_float(row.get("ruler_anchor_count")) or 0.0,
            reverse=True,
        ),
    )
    add(
        "low_observability",
        "Are low-confidence or zero-confidence rows being correctly blocked from positive anchor evidence?",
        sorted(
            [row for row in rows if "zero_conf_or_lowconf" in str(row.get("failure_flags") or "")],
            key=lambda row: safe_float(row.get("zero_conf_ratio")) or 0.0,
            reverse=True,
        ),
    )
    add(
        "unlabelled_anchor_support",
        "Do newly observed support rows show plausible anchors that need labels or controls before action?",
        sorted(
            [row for row in support if (safe_float(row.get("ruler_anchor_count")) or 0.0) > 0],
            key=lambda row: safe_float(row.get("ruler_anchor_count")) or 0.0,
            reverse=True,
        ),
    )
    return selected


def make_panel(tokens: list[dict[str, str]], row: Mapping[str, Any], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import ListedColormap

    role_code = {"RULER_CONTEXT": 0.0, "RULER_DEGENERATE": 1.0, "RULER_RISK": 2.0, "RULER_ANCHOR": 3.0}
    arrays = {
        "role": np.full(GRID, np.nan, dtype=float),
        "READ": np.full(GRID, np.nan, dtype=float),
        "SWA": np.full(GRID, np.nan, dtype=float),
        "geo": np.full(GRID, np.nan, dtype=float),
        "overlap": np.full(GRID, np.nan, dtype=float),
        "risk": np.full(GRID, np.nan, dtype=float),
    }
    for token in tokens:
        py = int(float(token.get("patch_y") or -1))
        px = int(float(token.get("patch_x") or -1))
        if py < 0 or py >= GRID[0] or px < 0 or px >= GRID[1]:
            continue
        arrays["role"][py, px] = role_code.get(token.get("ruler_role", ""), np.nan)
        arrays["READ"][py, px] = safe_float(token.get("READ_usage")) or np.nan
        arrays["SWA"][py, px] = safe_float(token.get("SWA_usage")) or np.nan
        arrays["geo"][py, px] = safe_float(token.get("geometry_leverage")) or np.nan
        arrays["overlap"][py, px] = safe_float(token.get("overlap_consistency")) or np.nan
        arrays["risk"][py, px] = safe_float(token.get("risk_score")) or np.nan

    fig, axes = plt.subplots(2, 3, figsize=(15, 7), constrained_layout=True)
    title = (
        f"seq{row.get('seq')} chunk {row.get('prev_chunk')}->{row.get('curr_chunk')} "
        f"{row.get('case_type')} anchors={row.get('ruler_anchor_count')} flags={row.get('failure_flags')}"
    )
    fig.suptitle(title, fontsize=10)
    role_cmap = ListedColormap(["#8aa2c8", "#b7b7b7", "#d66a59", "#2b9c6b"])
    panels = [
        ("role", "role context/degenerate/risk/anchor", role_cmap, 0.0, 3.0),
        ("READ", "READ usage", "viridis", 0.0, 1.0),
        ("SWA", "SWA QK proxy", "viridis", 0.0, 1.0),
        ("geo", "geometry leverage", "magma", 0.0, 1.0),
        ("overlap", "overlap consistency", "magma", 0.0, 1.0),
        ("risk", "risk score", "inferno", 0.0, 1.0),
    ]
    for ax, (key, label, cmap, vmin, vmax) in zip(axes.flat, panels):
        image = ax.imshow(arrays[key], interpolation="nearest", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("patch x")
        ax.set_ylabel("patch y")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def make_unavailable_panel(out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.axis("off")
    ax.text(
        0.02,
        0.60,
        "Per-head route mass is unavailable in the current v84 evidence.",
        fontsize=13,
        transform=ax.transAxes,
    )
    ax.text(
        0.02,
        0.35,
        "SWA_usage in Phase1/10 is a PCA QK-compatibility proxy, not true route attention mass.",
        fontsize=10,
        transform=ax.transAxes,
    )
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    tokens = read_csv(args.candidate_dir / "ruler_candidate_tokens.csv")
    audit_rows = read_csv(args.phase10_audit_dir / "support_expansion_audit_by_pair.csv")
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for token in tokens:
        grouped[pair_key(token)].append(token)

    selected = select_cases(audit_rows, args.max_panels_per_category)
    manifest_rows: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    question_rows: list[dict[str, Any]] = []
    out_dir = args.out_dir
    category_dirs = {
        "bad_ruler_absence": out_dir / "bad_ruler_absence_panels",
        "bad_ruler_contradiction": out_dir / "bad_ruler_contradiction_panels",
        "good_false_positive": out_dir / "good_false_positive_panels",
        "low_observability": out_dir / "low_observability_panels",
        "unlabelled_anchor_support": out_dir / "bad_ruler_contradiction_panels",
    }
    for row in selected:
        category = str(row["visual_category"])
        panel_path = category_dirs[category] / panel_name(row, category)
        make_panel(grouped.get(pair_key(row), []), row, panel_path)
        manifest_rows.append(
            {
                "visual_category": category,
                "seq": row.get("seq"),
                "prev_chunk": row.get("prev_chunk"),
                "curr_chunk": row.get("curr_chunk"),
                "case_type": row.get("case_type"),
                "base_case_type": row.get("base_case_type"),
                "panel_path": str(panel_path),
                "visual_question": row.get("visual_question"),
                "source_tokens": str(args.candidate_dir / "ruler_candidate_tokens.csv"),
                "source_audit": str(args.phase10_audit_dir / "support_expansion_audit_by_pair.csv"),
            }
        )
        evidence = (
            f"anchors={row.get('ruler_anchor_count')}; risk={row.get('ruler_risk_count')}; "
            f"degenerate={row.get('ruler_degenerate_count')}; zero_conf={row.get('zero_conf_ratio')}; "
            f"flags={row.get('failure_flags')}"
        )
        review_rows.append(
            {
                "panel_path": str(panel_path),
                "review_status": "auto_data_panel_reviewed",
                "observed_evidence": evidence,
                "claim_allowed": "visual_support_only_not_method_success",
            }
        )
        question_rows.append(
            {
                "seq": row.get("seq"),
                "prev_chunk": row.get("prev_chunk"),
                "curr_chunk": row.get("curr_chunk"),
                "question_type": category,
                "question": row.get("visual_question"),
                "evidence": evidence,
                "panel_path": str(panel_path),
            }
        )

    per_head_path = out_dir / "per_head_carrier_panels" / "per_head_route_mass_unavailable.png"
    make_unavailable_panel(per_head_path)
    manifest_rows.append(
        {
            "visual_category": "per_head_carrier_unavailable",
            "seq": "",
            "prev_chunk": "",
            "curr_chunk": "",
            "case_type": "diagnostic_unavailable",
            "base_case_type": "",
            "panel_path": str(per_head_path),
            "visual_question": "Is per-head route mass available for carrier localization?",
            "source_tokens": str(args.candidate_dir / "ruler_candidate_tokens.csv"),
            "source_audit": str(args.phase10_audit_dir / "support_expansion_audit_summary.json"),
        }
    )
    review_rows.append(
        {
            "panel_path": str(per_head_path),
            "review_status": "auto_data_panel_reviewed",
            "observed_evidence": "per_head_route_mass_unavailable; SWA_usage_source=qk_compatibility_proxy_no_route_mass",
            "claim_allowed": "carrier_localization_blocker",
        }
    )

    write_csv(out_dir / "failed_case_to_visual_question.csv", question_rows)
    write_csv(out_dir / "visual_manifest.csv", manifest_rows)
    write_csv(out_dir / "visual_review.csv", review_rows)

    missing = []
    zero_byte = []
    for row in manifest_rows:
        path = Path(str(row.get("panel_path")))
        if not path.is_file():
            missing.append(str(path))
        elif path.stat().st_size <= 0:
            zero_byte.append(str(path))
    integrity = {
        "schema": "acl2_v84_phase11_visual_integrity_audit_v1",
        "visual_integrity_pass": not missing and not zero_byte and bool(manifest_rows),
        "manifest_rows": len(manifest_rows),
        "missing_files": missing,
        "zero_byte_files": zero_byte,
        "notes": [
            "Panels are generated from token-level candidate CSV values.",
            "Per-head carrier panel is an explicit unavailable-evidence panel, not a route visualization.",
        ],
    }
    write_json(out_dir / "visual_integrity_audit.json", integrity)

    hypothesis_md = [
        "# Phase11 New Hypothesis Bank",
        "",
        "## H11.1 Candidate Definition Too Sparse On Labelled Bad Rows",
        "",
        "Phase10 expansion found more anchors overall, but labelled bad anchor recall stayed low. "
        "The current strict intersection may be selecting support-rich easy rows rather than the actual failure carrier.",
        "",
        "## H11.2 SWA Proxy Is Not The True Carrier",
        "",
        "SWA_usage is a PCA QK-compatibility proxy. True per-head route mass is still unavailable, so a semantic-shuffle/same-head control cannot be run from current artifacts.",
        "",
        "## H11.3 Geometry Degeneracy Dominates Anchor Candidates",
        "",
        "Many visible anchors co-occur with degenerate-dominant flags. The next route should separate structural edges from broad planar road/background support.",
        "",
        "## H11.4 Merge/Gauge May Be The Missing Interface",
        "",
        "If READ/SWA usage stays nonspecific after true route dumps, the next carrier search should move to merge/gauge boundary state rather than SWA alpha/action sweeps.",
        "",
        "## H11.5 Label Universe May Not Target Scale/Gauge Failure Directly",
        "",
        "Unlabelled adjacent rows show anchors but have no bad/good handoff labels. A future support set needs non-GT runtime clues plus held-out labels only for audit.",
        "",
    ]
    (out_dir / "new_hypothesis_bank.md").write_text("\n".join(hypothesis_md), encoding="utf-8")

    insight_md = [
        "# Phase11 Visual Insight",
        "",
        f"- Visual integrity pass: `{integrity['visual_integrity_pass']}` with {len(manifest_rows)} manifest rows.",
        "- Bad labelled rows mostly ask whether ruler anchors are absent or blocked by risk/low observability.",
        "- Labelled good rows with anchors are treated as good-protection false positives, not success evidence.",
        "- Unlabelled support rows with anchors are useful for rediscovery but cannot count as bad/good separation.",
        "- Per-head route carrier localization remains blocked because true route attention mass is unavailable.",
        "",
    ]
    (out_dir / "visual_insight.md").write_text("\n".join(insight_md), encoding="utf-8")

    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "visual_integrity_pass": integrity["visual_integrity_pass"],
                "manifest_rows": len(manifest_rows),
                "question_rows": len(question_rows),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
