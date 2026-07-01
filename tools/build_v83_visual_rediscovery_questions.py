#!/usr/bin/env python3
"""Build ACL2 v83 visual rediscovery bundle from audited source artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Iterable, Mapping


DEFAULT_OUT_DIR = Path(
    "results/acl2_v83tf_clue_sufficiency_vs_action_misuse/"
    "phase11_visual_rediscovery"
)
DEFAULT_PHASE3_ROOT = Path(
    "results/acl2_v83tf_clue_sufficiency_vs_action_misuse/"
    "phase3_carrier_alignment"
)
DEFAULT_SOURCE_REDISCOVERY = Path(
    "results/acl2_v82tf_swa_carrier_semantic_scale_handoff/"
    "phase12_rediscovery"
)


GROUP_MAP = {
    "qk_pair_panels": "new_qk_pair_panels",
    "swa_kv_panels": "new_swa_kv_panels",
    "merge_boundary_panels": "new_merge_boundary_panels",
    "ttt_write_panels": "new_ttt_write_less_panels",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--phase3-root", type=Path, default=DEFAULT_PHASE3_ROOT)
    parser.add_argument("--source-rediscovery", type=Path, default=DEFAULT_SOURCE_REDISCOVERY)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: serialize(row.get(key, "")) for key in keys})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def serialize(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def copy_group(source_root: Path, out_dir: Path, group: str, source_group: str) -> list[dict[str, Any]]:
    src_dir = source_root / source_group
    dst_dir = out_dir / group
    dst_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for src in sorted(src_dir.glob("*.png")):
        dst = dst_dir / src.name
        shutil.copy2(src, dst)
        rows.append(
            {
                "group": group,
                "path": str(dst),
                "source_path": str(src),
                "status": "copied_existing_v82_phase12_artifact",
                "bytes": dst.stat().st_size if dst.is_file() else 0,
                "sha256": sha256(dst) if dst.is_file() else "",
                "note": "Copied from audited v82 Phase12 rediscovery; no new visual evidence generated.",
            }
        )
    return rows


def build_questions(phase3_root: Path, source_rediscovery: Path) -> list[dict[str, Any]]:
    carrier_rows = read_csv(phase3_root / "carrier_alignment_rows.csv")
    summary = read_json(phase3_root / "carrier_alignment_summary.json")
    questions: list[dict[str, Any]] = []
    for row in carrier_rows:
        carrier = row.get("carrier_body", "")
        blocker = row.get("blocker", "")
        if carrier == "READ":
            question = "Add READ QK same-pair random and semantic-shuffle QK controls; check whether stable/harm usage remains specific."
        elif carrier == "SWA":
            question = "SWA actual-vs-random exists, but semantic-shuffle specificity failed; inspect whether semantic same-group structure adds anything shuffle cannot."
        elif carrier == "merge_gauge":
            question = "Merge/gauge residual separates bad/good, but lacks same-overlap random and semantic-shuffle weighting controls."
        elif carrier == "TTT":
            question = "TTT remains not eligible until SWA or merge/gauge carrier is confirmed with good-case protection."
        else:
            question = "Carrier failed Phase3; inspect missing carrier evidence."
        questions.append(
            {
                "source": "v83_phase3_carrier_alignment",
                "carrier_body": carrier,
                "phase3_gate_pass": row.get("carrier_gate_pass", ""),
                "auc": row.get("auc", ""),
                "bad_recall": row.get("bad_recall", ""),
                "good_false_positive_rate": row.get("good_false_positive_rate", ""),
                "actual_vs_random_gate_pass": row.get("actual_vs_random_gate_pass", ""),
                "semantic_shuffle_specificity_gate_pass": row.get("semantic_shuffle_specificity_gate_pass", ""),
                "blocker": blocker,
                "question": question,
            }
        )

    route_evidence = summary.get("route_specificity_evidence", {}) if isinstance(summary, dict) else {}
    for near in route_evidence.get("top_near_misses", [])[:5]:
        if isinstance(near, dict):
            questions.append(
                {
                    "source": "v82_phase12_top_near_miss",
                    "carrier_body": "SWA",
                    "route_group": near.get("route_group", ""),
                    "filter_name": near.get("filter_name", ""),
                    "bad_recall": near.get("bad_recall", ""),
                    "good_false_positive_rate": near.get("good_false_positive_rate", ""),
                    "same_mass_random_rule_gate_pass": near.get("same_mass_random_rule_gate_pass", ""),
                    "semantic_shuffled_available_for_route_group": near.get("semantic_shuffled_available_for_route_group", ""),
                    "semantic_shuffled_rule_gate_pass": near.get("semantic_shuffled_rule_gate_pass", ""),
                    "question": "Near-miss beats same-mass control but fails semantic-shuffle; inspect what visual structure semantic shuffle preserves.",
                }
            )

    for src_name, label in [
        ("failed_swa_pair_to_visual_question.csv", "v82_phase12_failed_swa_pair"),
        ("failed_merge_boundary_to_visual_question.csv", "v82_phase12_failed_merge_boundary"),
        ("failed_long_window_to_visual_question.csv", "v82_phase12_failed_long_window"),
    ]:
        for row in read_csv(source_rediscovery / src_name):
            copied = {"source": label, **row}
            copied.setdefault("question", "Imported v82 rediscovery question.")
            questions.append(copied)
    return questions


def render_hypotheses(phase3_summary: Mapping[str, Any], source_audit: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# ACL2 v83 Visual Rediscovery Hypothesis Bank",
            "",
            "1. READ QK controls are missing: add same-pair random and semantic-shuffle QK panels before claiming READ carrier specificity.",
            "2. SWA has observable runtime route and actual-vs-random evidence, but existing semantic-shuffle controls still have zero passing rows, so semantic same-group route mass is not specific enough for promotion.",
            "3. Merge/gauge residuals are stronger than SWA numerically, but the current evidence lacks same-overlap random and semantic-shuffle weighting controls; this should be the next non-SWA interface if continuing.",
            "4. TTT remains downstream-only; do not run TTT write until SWA or merge/gauge has a confirmed carrier with good-case protection.",
            "",
            "## Evidence Pointers",
            "",
            f"- v83_phase3_decision: `{phase3_summary.get('decision')}`",
            f"- v83_phase3_gate_pass: `{phase3_summary.get('phase3_gate_pass')}`",
            f"- v82_phase12_visual_audit_gate_pass: `{source_audit.get('visual_audit_gate_pass')}`",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    phase3_summary = read_json(args.phase3_root / "carrier_alignment_summary.json")
    source_audit = read_json(args.source_rediscovery / "visual_integrity_audit.json")

    questions = build_questions(args.phase3_root, args.source_rediscovery)
    write_csv(args.out_dir / "failed_case_to_visual_question.csv", questions)

    manifest_rows: list[dict[str, Any]] = []
    for group, source_group in GROUP_MAP.items():
        manifest_rows.extend(copy_group(args.source_rediscovery, args.out_dir, group, source_group))
    write_csv(args.out_dir / "visual_manifest.csv", manifest_rows)

    review_rows = [
        {
            **row,
            "review_status": "present_existing_v82_phase12_artifact" if row.get("bytes", 0) else "missing_artifact",
        }
        for row in manifest_rows
    ]
    write_csv(args.out_dir / "visual_review.csv", review_rows)

    group_counts: dict[str, dict[str, int]] = {}
    for group in GROUP_MAP:
        rows = [row for row in manifest_rows if row.get("group") == group]
        group_counts[group] = {
            "count": len(rows),
            "ok": sum(1 for row in rows if int(row.get("bytes") or 0) > 0 and Path(str(row.get("path", ""))).is_file()),
        }

    all_groups_ok = all(item["count"] > 0 and item["ok"] == item["count"] for item in group_counts.values())
    visual_audit_gate_pass = bool(source_audit.get("visual_audit_gate_pass") and all_groups_ok and questions)
    audit = {
        "schema": "acl2_v83_phase11_visual_rediscovery_v1",
        "root": str(args.out_dir),
        "phase3_summary": str(args.phase3_root / "carrier_alignment_summary.json"),
        "source_rediscovery": str(args.source_rediscovery),
        "failed_case_question_count": len(questions),
        "group_counts": group_counts,
        "visual_review_rows": len(review_rows),
        "source_visual_audit_gate_pass": bool(source_audit.get("visual_audit_gate_pass")),
        "visual_audit_gate_pass": visual_audit_gate_pass,
        "phase3_gate_pass": bool(phase3_summary.get("phase3_gate_pass")),
        "after_visual_rediscovery_decision": (
            "carrier_not_localized"
            if not phase3_summary.get("phase3_gate_pass")
            else "carrier_aligned"
        ),
        "runtime_action_allowed": bool(phase3_summary.get("runtime_action_allowed")),
        "note": "Panels are copied from existing audited v82 Phase12 rediscovery artifacts; no new visual evidence is fabricated.",
    }
    write_json(args.out_dir / "visual_integrity_audit.json", audit)
    (args.out_dir / "visual_insight.md").write_text(
        "# ACL2 v83 Visual Rediscovery Insight\n\n"
        f"- visual_audit_gate_pass: `{audit['visual_audit_gate_pass']}`\n"
        f"- phase3_gate_pass: `{audit['phase3_gate_pass']}`\n"
        f"- after_visual_rediscovery_decision: `{audit['after_visual_rediscovery_decision']}`\n"
        f"- failed_case_question_count: `{audit['failed_case_question_count']}`\n\n"
        "The rediscovery audit reuses v82 Phase12 panels and questions as audited source artifacts. "
        "It confirms the current v83 stop point: visual evidence is present, but no carrier passed "
        "actual-vs-random plus semantic-shuffle specificity with good-case protection.\n",
        encoding="utf-8",
    )
    (args.out_dir / "new_hypothesis_bank.md").write_text(render_hypotheses(phase3_summary, source_audit), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
