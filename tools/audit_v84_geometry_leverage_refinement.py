#!/usr/bin/env python3
"""Audit ACL2 v84 geometry-leverage refinements for memory-ruler anchors.

This is an offline diagnostic only. It re-scores landed Phase10 token
candidates with fixed, no-training structural-edge rules and reports whether
any rule repairs bad/good separation enough to justify the next route-control
step. It does not change runtime behavior or claim v84 anchor-specific route
usage.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


DEFAULT_TOKENS = Path(
    "results/acl2_v84tf_memory_ruler_audit/phase10_support_expansion_candidates/ruler_candidate_tokens.csv"
)
DEFAULT_PAIR_AUDIT = Path(
    "results/acl2_v84tf_memory_ruler_audit/phase10_support_expansion_audit/support_expansion_audit_by_pair.csv"
)
DEFAULT_OUT_DIR = Path("results/acl2_v84tf_memory_ruler_audit/phase14_geometry_leverage_refinement")

BAD_RECALL_GATE = 0.60
GOOD_FPR_GATE = 0.25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=Path, default=DEFAULT_TOKENS)
    parser.add_argument("--pair-audit", type=Path, default=DEFAULT_PAIR_AUDIT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
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


def clamp01(value: float | None) -> float:
    if value is None or not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, float(value)))


def pair_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return str(row.get("seq", "")).zfill(2), str(row.get("prev_chunk", "")), str(row.get("curr_chunk", ""))


def quantile(values: list[float], q: float) -> float | None:
    vals = sorted(v for v in values if math.isfinite(v))
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    pos = max(0.0, min(1.0, q)) * (len(vals) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return vals[lo]
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def mean(values: Iterable[float]) -> float | None:
    vals = [v for v in values if math.isfinite(v)]
    return sum(vals) / len(vals) if vals else None


def token_features(row: Mapping[str, Any]) -> dict[str, float]:
    read = max(safe_float(row.get("READ_usage")) or 0.0, 0.0)
    swa = max(safe_float(row.get("SWA_usage")) or 0.0, 0.0)
    memory_factor = math.sqrt(read * swa) if read > 0.0 and swa > 0.0 else 0.0
    semantic = clamp01(safe_float(row.get("semantic_trust")))
    purity = clamp01(safe_float(row.get("patch_purity")))
    sem_conf = clamp01(safe_float(row.get("semantic_confidence")))
    parallax = clamp01(safe_float(row.get("parallax_proxy")))
    nondeg = clamp01(safe_float(row.get("nondegenerate_proxy")))
    far = clamp01(safe_float(row.get("far_context_proxy")))
    overlap = clamp01(safe_float(row.get("overlap_consistency")))
    risk = clamp01(safe_float(row.get("risk_score")))
    current_score = clamp01(safe_float(row.get("ruler_anchor_score")))
    edge_shape = math.sqrt(max(parallax, 0.0) * max(nondeg, 0.0))
    far_penalty = max(0.0, 1.0 - 0.5 * far)
    structural_edge_score = (
        math.sqrt(max(semantic, 0.0))
        * edge_shape
        * overlap
        * math.sqrt(max(memory_factor, 0.0))
        * max(0.0, 1.0 - risk)
        * far_penalty
    )
    geometry_memory_score = edge_shape * overlap * math.sqrt(max(memory_factor, 0.0)) * max(0.0, 1.0 - risk) * far_penalty
    return {
        "read": read,
        "swa": swa,
        "memory_factor": memory_factor,
        "semantic": semantic,
        "purity": purity,
        "sem_conf": sem_conf,
        "parallax": parallax,
        "nondeg": nondeg,
        "far": far,
        "overlap": overlap,
        "risk": risk,
        "current_score": current_score,
        "edge_shape": edge_shape,
        "structural_edge_score": structural_edge_score,
        "geometry_memory_score": geometry_memory_score,
    }


def weak_eligible(feat: Mapping[str, float]) -> bool:
    return (
        feat["semantic"] >= 0.20
        and feat["purity"] >= 0.35
        and feat["parallax"] >= 0.05
        and feat["nondeg"] >= 0.05
        and feat["overlap"] >= 0.10
        and feat["risk"] < 0.70
        and feat["far"] <= 0.90
        and feat["memory_factor"] > 0.0
    )


def variant_match(name: str, row: Mapping[str, Any], feat: Mapping[str, float], thresholds: Mapping[str, float]) -> bool:
    if name == "current_role_anchor":
        return row.get("ruler_role") == "RULER_ANCHOR"
    if name == "structural_edge_strict_fixed":
        return (
            feat["semantic"] >= 0.45
            and feat["purity"] >= 0.50
            and feat["sem_conf"] >= 0.45
            and feat["parallax"] >= 0.10
            and feat["nondeg"] >= 0.10
            and feat["far"] <= 0.90
            and feat["overlap"] >= 0.20
            and feat["risk"] < 0.65
            and feat["memory_factor"] > 0.0
        )
    if name == "structural_edge_overlap_strict_fixed":
        return (
            feat["semantic"] >= 0.35
            and feat["parallax"] >= 0.08
            and feat["nondeg"] >= 0.08
            and feat["overlap"] >= 0.35
            and feat["risk"] < 0.55
            and feat["memory_factor"] > 0.0
        )
    if name == "weak_medium_conf_nonzero_fixed":
        return weak_eligible(feat)
    if name.startswith("structural_edge_topq"):
        threshold = thresholds[name]
        return weak_eligible(feat) and feat["structural_edge_score"] >= threshold
    if name.startswith("geometry_memory_topq"):
        threshold = thresholds[name]
        return (
            feat["parallax"] >= 0.05
            and feat["nondeg"] >= 0.05
            and feat["overlap"] >= 0.10
            and feat["risk"] < 0.70
            and feat["far"] <= 0.90
            and feat["memory_factor"] > 0.0
            and feat["geometry_memory_score"] >= threshold
        )
    raise KeyError(name)


def counter_to_dict(counter: Counter[Any]) -> dict[str, int]:
    return {str(k): int(v) for k, v in sorted(counter.items(), key=lambda item: str(item[0]))}


def build_report(summary: Mapping[str, Any], variant_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Phase14 Geometry-Leverage Refinement Audit",
        "",
        f"- Refinement gate pass: `{summary['refinement_gate_pass']}`",
        f"- Runtime action allowed: `{summary['runtime_action_allowed']}`",
        f"- Best under good-FPR gate: `{summary.get('best_variant_under_good_fpr_gate') or 'none'}`",
        f"- Best overall labelled bad recall: `{summary.get('best_variant_by_bad_recall') or 'none'}`",
        "",
        "## Variant Metrics",
        "",
        "| variant | threshold | bad recall | good FPR | labelled gate | support pairs | selected tokens |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in variant_rows:
        lines.append(
            "| {variant} | {threshold} | {bad:.6f} | {good:.6f} | `{gate}` | {support} | {tokens} |".format(
                variant=row["variant"],
                threshold=serialize(row.get("threshold")),
                bad=row["labelled_bad_recall_count1"],
                good=row["labelled_good_fpr_count1"],
                gate=row["labelled_gate_pass_count1"],
                support=row["support_positive_pair_count_count1"],
                tokens=row["selected_token_count"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This audit only repairs candidate role definitions. A passing row here would still require v84 RULER_ANCHOR-specific route controls before runtime action.",
            "The geometry-only variants are diagnostic controls, not valid semantic-geometric memory-ruler definitions.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    tokens = read_csv(args.tokens)
    pair_rows = read_csv(args.pair_audit)
    pair_by_key = {pair_key(row): row for row in pair_rows}

    enriched: list[dict[str, Any]] = []
    weak_scores: list[float] = []
    geom_scores: list[float] = []
    for row in tokens:
        feat = token_features(row)
        item = {**row, **{f"feat_{key}": value for key, value in feat.items()}}
        enriched.append(item)
        if weak_eligible(feat):
            weak_scores.append(feat["structural_edge_score"])
        if (
            feat["parallax"] >= 0.05
            and feat["nondeg"] >= 0.05
            and feat["overlap"] >= 0.10
            and feat["risk"] < 0.70
            and feat["far"] <= 0.90
            and feat["memory_factor"] > 0.0
        ):
            geom_scores.append(feat["geometry_memory_score"])

    thresholds: dict[str, float] = {}
    variants = [
        "current_role_anchor",
        "structural_edge_strict_fixed",
        "structural_edge_overlap_strict_fixed",
        "weak_medium_conf_nonzero_fixed",
    ]
    for q in (0.80, 0.90, 0.95):
        name = f"structural_edge_topq{int(q * 100)}"
        thresholds[name] = quantile(weak_scores, q) or math.inf
        variants.append(name)
    for q in (0.90, 0.95):
        name = f"geometry_memory_topq{int(q * 100)}_no_semantic_control"
        thresholds[name] = quantile(geom_scores, q) or math.inf
        variants.append(name)

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    selected_rows: list[dict[str, Any]] = []
    for row in enriched:
        grouped[pair_key(row)].append(row)

    pair_variant_rows: list[dict[str, Any]] = []
    variant_rows: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {
        "weak_eligible_token_count": len(weak_scores),
        "geometry_memory_eligible_token_count": len(geom_scores),
        "weak_structural_edge_score_quantiles": {
            str(q): quantile(weak_scores, q) for q in (0.50, 0.80, 0.90, 0.95, 0.99)
        },
        "geometry_memory_score_quantiles": {
            str(q): quantile(geom_scores, q) for q in (0.50, 0.80, 0.90, 0.95, 0.99)
        },
    }

    for variant in variants:
        labelled_bad_total = 0
        labelled_good_total = 0
        labelled_bad_pos1 = 0
        labelled_good_pos1 = 0
        labelled_bad_pos2 = 0
        labelled_good_pos2 = 0
        support_pos1 = 0
        support_pos2 = 0
        selected_token_count = 0
        selected_labels: Counter[Any] = Counter()
        selected_roles: Counter[Any] = Counter()
        selected_case_types: Counter[Any] = Counter()
        selected_scores: list[float] = []

        for key, pair in pair_by_key.items():
            group = grouped.get(key, [])
            selected: list[dict[str, Any]] = []
            for row in group:
                feat = {field[5:]: safe_float(row.get(field)) or 0.0 for field in row if field.startswith("feat_")}
                if variant_match(variant, row, feat, thresholds):
                    selected.append(row)
            count = len(selected)
            score_sum = sum((safe_float(row.get("feat_structural_edge_score")) or 0.0) for row in selected)
            geom_score_sum = sum((safe_float(row.get("feat_geometry_memory_score")) or 0.0) for row in selected)
            mean_overlap = mean([safe_float(row.get("overlap_consistency")) or 0.0 for row in selected])
            mean_memory = mean([safe_float(row.get("feat_memory_factor")) or 0.0 for row in selected])
            base_case = str(pair.get("base_case_type") or "")
            is_labelled = str(pair.get("support_expansion_label_scope") or "") == "labelled_v82_main_pair"
            if is_labelled and base_case == "bad":
                labelled_bad_total += 1
                labelled_bad_pos1 += int(count >= 1)
                labelled_bad_pos2 += int(count >= 2)
            elif is_labelled:
                labelled_good_total += 1
                labelled_good_pos1 += int(count >= 1)
                labelled_good_pos2 += int(count >= 2)
            if count >= 1:
                support_pos1 += 1
            if count >= 2:
                support_pos2 += 1
            selected_token_count += count
            selected_case_types[pair.get("case_type")] += count
            for row in selected:
                selected_labels[row.get("semantic_label")] += 1
                selected_roles[row.get("ruler_role")] += 1
                selected_scores.append(safe_float(row.get("feat_structural_edge_score")) or 0.0)
                if len(selected_rows) < 1200:
                    selected_rows.append(
                        {
                            "variant": variant,
                            "seq": row.get("seq"),
                            "prev_chunk": row.get("prev_chunk"),
                            "curr_chunk": row.get("curr_chunk"),
                            "case_type": row.get("case_type"),
                            "base_case_type": row.get("base_case_type"),
                            "patch_y": row.get("patch_y"),
                            "patch_x": row.get("patch_x"),
                            "semantic_label": row.get("semantic_label"),
                            "current_role": row.get("ruler_role"),
                            "structural_edge_score": row.get("feat_structural_edge_score"),
                            "geometry_memory_score": row.get("feat_geometry_memory_score"),
                            "memory_factor": row.get("feat_memory_factor"),
                            "semantic_trust": row.get("semantic_trust"),
                            "parallax_proxy": row.get("parallax_proxy"),
                            "nondegenerate_proxy": row.get("nondegenerate_proxy"),
                            "far_context_proxy": row.get("far_context_proxy"),
                            "overlap_consistency": row.get("overlap_consistency"),
                            "risk_score": row.get("risk_score"),
                        }
                    )
            pair_variant_rows.append(
                {
                    "variant": variant,
                    "seq": pair.get("seq"),
                    "prev_chunk": pair.get("prev_chunk"),
                    "curr_chunk": pair.get("curr_chunk"),
                    "case_type": pair.get("case_type"),
                    "base_case_type": pair.get("base_case_type"),
                    "support_expansion_label_scope": pair.get("support_expansion_label_scope"),
                    "selected_anchor_count": count,
                    "selected_structural_edge_score_sum": score_sum,
                    "selected_geometry_memory_score_sum": geom_score_sum,
                    "selected_overlap_consistency_mean": mean_overlap,
                    "selected_memory_factor_mean": mean_memory,
                    "pair_positive_count1": count >= 1,
                    "pair_positive_count2": count >= 2,
                    "original_failure_flags": pair.get("failure_flags"),
                }
            )

        bad_recall1 = labelled_bad_pos1 / max(labelled_bad_total, 1)
        good_fpr1 = labelled_good_pos1 / max(labelled_good_total, 1)
        bad_recall2 = labelled_bad_pos2 / max(labelled_bad_total, 1)
        good_fpr2 = labelled_good_pos2 / max(labelled_good_total, 1)
        row = {
            "variant": variant,
            "threshold": thresholds.get(variant),
            "selected_token_count": selected_token_count,
            "support_positive_pair_count_count1": support_pos1,
            "support_positive_pair_count_count2": support_pos2,
            "labelled_bad_rows": labelled_bad_total,
            "labelled_good_rows": labelled_good_total,
            "labelled_bad_positive_count1": labelled_bad_pos1,
            "labelled_good_positive_count1": labelled_good_pos1,
            "labelled_bad_recall_count1": bad_recall1,
            "labelled_good_fpr_count1": good_fpr1,
            "labelled_gate_pass_count1": bad_recall1 >= BAD_RECALL_GATE and good_fpr1 <= GOOD_FPR_GATE,
            "labelled_bad_positive_count2": labelled_bad_pos2,
            "labelled_good_positive_count2": labelled_good_pos2,
            "labelled_bad_recall_count2": bad_recall2,
            "labelled_good_fpr_count2": good_fpr2,
            "labelled_gate_pass_count2": bad_recall2 >= BAD_RECALL_GATE and good_fpr2 <= GOOD_FPR_GATE,
            "selected_structural_edge_score_mean": mean(selected_scores),
            "selected_semantic_label_counts_top10": dict(selected_labels.most_common(10)),
            "selected_current_role_counts": counter_to_dict(selected_roles),
            "selected_case_type_counts": counter_to_dict(selected_case_types),
            "notes": "geometry-only/no-semantic control" if "no_semantic" in variant else "",
        }
        variant_rows.append(row)

    passing = [row for row in variant_rows if row["labelled_gate_pass_count1"]]
    under_good_fpr = [row for row in variant_rows if row["labelled_good_fpr_count1"] <= GOOD_FPR_GATE]
    best_under_good = sorted(
        under_good_fpr,
        key=lambda row: (row["labelled_bad_recall_count1"], row["support_positive_pair_count_count1"], -row["labelled_good_fpr_count1"]),
        reverse=True,
    )
    best_by_recall = sorted(
        variant_rows,
        key=lambda row: (row["labelled_bad_recall_count1"], -row["labelled_good_fpr_count1"], row["support_positive_pair_count_count1"]),
        reverse=True,
    )
    summary = {
        "schema": "acl2_v84_geometry_leverage_refinement_v1",
        "refinement_gate_pass": bool(passing),
        "runtime_action_allowed": False,
        "runtime_action_blocker": "v84 RULER_ANCHOR-specific true route controls and semantic/same-mass controls remain unavailable or failing",
        "bad_recall_gate": BAD_RECALL_GATE,
        "good_fpr_gate": GOOD_FPR_GATE,
        "token_rows": len(tokens),
        "pair_rows": len(pair_rows),
        "labelled_pairs": sum(
            1 for row in pair_rows if str(row.get("support_expansion_label_scope") or "") == "labelled_v82_main_pair"
        ),
        "variant_count": len(variant_rows),
        "best_variant_under_good_fpr_gate": best_under_good[0]["variant"] if best_under_good else None,
        "best_variant_under_good_fpr_gate_metrics": best_under_good[0] if best_under_good else None,
        "best_variant_by_bad_recall": best_by_recall[0]["variant"] if best_by_recall else None,
        "best_variant_by_bad_recall_metrics": best_by_recall[0] if best_by_recall else None,
        "passing_variants": [row["variant"] for row in passing],
        "diagnostics": diagnostics,
        "limitations": [
            "This is an offline role-refinement audit, not runtime action.",
            "Unsupervised top-q variants use global token score quantiles without labels, but they are still diagnostic thresholds.",
            "Geometry-only variants intentionally drop semantic trust and are controls, not allowed memory-ruler definitions.",
            "Even if labelled separation passed, v84 anchor-specific true route controls would still be required before action.",
        ],
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "geometry_leverage_refinement_by_pair.csv", pair_variant_rows)
    write_csv(args.out_dir / "geometry_leverage_refinement_variants.csv", variant_rows)
    write_csv(args.out_dir / "geometry_leverage_refinement_selected_tokens_sample.csv", selected_rows)
    write_json(args.out_dir / "geometry_leverage_refinement_summary.json", summary)
    (args.out_dir / "geometry_leverage_refinement_report.md").write_text(
        build_report(summary, variant_rows), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "out_dir": str(args.out_dir),
                "refinement_gate_pass": summary["refinement_gate_pass"],
                "runtime_action_allowed": summary["runtime_action_allowed"],
                "best_variant_under_good_fpr_gate": summary["best_variant_under_good_fpr_gate"],
                "best_variant_by_bad_recall": summary["best_variant_by_bad_recall"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
