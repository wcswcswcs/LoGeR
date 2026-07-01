#!/usr/bin/env python3
"""Build v81S mandatory rediscovery bundle from existing audited artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any, Iterable, Mapping


V81S_ROOT = Path("results/kitti01_hmc_v2/acl2_v81s_swa_first_semantic_scale_memory_control/report_final")
V81TF_ROOT = Path("results/kitti01_hmc_v2/acl2_v81tf_longwindow_semantic_three_memory_control/report_final")
DEFAULT_OUT = V81S_ROOT / "phaseS14_rediscovery"
DEFAULT_S5_ROWS = V81S_ROOT / "phaseS5_swa_action_route_audit/swa_action_route_rows.csv"
DEFAULT_S5_SUMMARY = V81S_ROOT / "phaseS5_swa_action_route_audit/swa_action_route_audit_summary.json"
DEFAULT_V81TF_PHASE11 = V81TF_ROOT / "phase11_rediscovery"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = {}
            for key in fields:
                value = row.get(key)
                out[key] = json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list, tuple)) else value
            writer.writerow(out)


def _copy_some(files: list[Path], dst: Path, *, limit: int) -> list[dict[str, Any]]:
    dst.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for src in files[:limit]:
        target = dst / src.name
        shutil.copy2(src, target)
        rows.append({"source": str(src), "path": str(target), "exists": target.is_file(), "bytes": target.stat().st_size if target.is_file() else 0})
    return rows


def _pngs(root: Path, pattern: str = "*.png") -> list[Path]:
    return sorted(path for path in root.glob(pattern) if path.is_file())


def _build_failed_swa_questions(s5_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in s5_rows:
        if row.get("case_type") != "bad":
            continue
        out.append(
            {
                "seq": row.get("seq"),
                "chunk": row.get("chunk"),
                "candidate": row.get("candidate_short") or row.get("candidate"),
                "route_mask_present": row.get("route_mask_present"),
                "action_applied": row.get("action_applied"),
                "head_tail_improvement_vs_baseline_ratio": row.get("head_tail_improvement_vs_baseline_ratio"),
                "future_after_overlap_improvement_vs_baseline_ratio": row.get("future_after_overlap_improvement_vs_baseline_ratio"),
                "scale_cv_improvement_vs_baseline_ratio": row.get("scale_cv_improvement_vs_baseline_ratio"),
                "question": "Why does a true SWA route/action signal stay below v81S geometry thresholds and fail controls?",
            }
        )
    return out


def _build_failed_long_questions(v81tf_phase11: Path) -> list[dict[str, Any]]:
    existing = _read_csv(v81tf_phase11 / "failed_long_window_to_visual_question.csv")
    if existing:
        return existing
    return [
        {
            "question": "Which long-window Type-B merge/gauge carrier is still missing after SWA route action fails?",
            "source": str(v81tf_phase11),
        }
    ]


def _render_insight(s5_summary: Mapping[str, Any], audit: Mapping[str, Any]) -> str:
    candidates = s5_summary.get("candidate_summaries", [])
    lines = [
        "# ACL2 v81S Phase S14 Rediscovery Insight",
        "",
        "## What Changed",
        "",
        "- S3 visual/QKV artifacts are available, and S5 route-smoke produced true SWA route maps on a multi-sequence bad/good sample.",
        "- S5 action fidelity passed for the reused source-replace/source-gate route actions, but the geometry metric gate failed.",
        "",
        "## Candidate Signals",
        "",
    ]
    for cand in candidates:
        lines.append(
            "- `{short}`: bad_head_median={head}, bad_future_median={future}, bad_scale_median={scale}, metric_signal_rows={signal}".format(
                short=cand.get("candidate_short"),
                head=cand.get("bad_head_tail_median_improvement_vs_baseline_ratio"),
                future=cand.get("bad_future_after_overlap_median_improvement_vs_baseline_ratio"),
                scale=cand.get("bad_scale_cv_median_improvement_vs_baseline_ratio"),
                signal=cand.get("v81s_metric_signal_rows"),
            )
        )
    lines.extend(
        [
            "",
            "## Next Hypotheses",
            "",
            "- The blocker is not missing route hook wiring; the route signal is too weak or not directionally specific enough.",
            "- Merge/gauge fallback evidence from the paired v81TF Type-B work shows the same pattern: large action mass or safe projection can occur, but it does not beat controls.",
            "- The next valid mechanism would need a direct, control-aware merge/gauge state interface with good-case protection, not more alpha or threshold sweeps.",
            "",
            "## Gate",
            "",
            f"- visual_audit_gate_pass: `{audit.get('visual_audit_gate_pass')}`",
        ]
    )
    return "\n".join(lines) + "\n"


def _render_hypotheses() -> str:
    return "\n".join(
        [
            "# ACL2 v81S Rediscovery Hypothesis Bank",
            "",
            "1. Direct merge/gauge state interface: SWA route masks can localize candidate evidence, but the current source-replace/gate actuator does not move the merge state with enough directionality.",
            "2. Control-aware carrier selection: candidate and same-mass random often move together, so the next rule must explain why semantic evidence should beat same-mass controls before runtime promotion.",
            "3. Good-case first protection: any next boundary controller must prove no-worsen on low-error adjacent pairs before long-window TTT can be re-opened.",
            "4. Uncertainty is not enough by itself: latent/ThingStuff/dense support can produce large mass, but prior v81TF fallback shows mass without directional control is not a method signal.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--s5-rows", type=Path, default=DEFAULT_S5_ROWS)
    parser.add_argument("--s5-summary", type=Path, default=DEFAULT_S5_SUMMARY)
    parser.add_argument("--v81tf-phase11", type=Path, default=DEFAULT_V81TF_PHASE11)
    args = parser.parse_args()

    s5_rows = _read_csv(args.s5_rows)
    s5_summary = _read_json(args.s5_summary)
    args.out_root.mkdir(parents=True, exist_ok=True)

    failed_swa = _build_failed_swa_questions(s5_rows)
    failed_long = _build_failed_long_questions(args.v81tf_phase11)
    _write_csv(args.out_root / "failed_swa_pair_to_visual_question.csv", failed_swa)
    _write_csv(args.out_root / "failed_long_window_to_visual_question.csv", failed_long)

    qk_rows = _copy_some(
        _pngs(V81S_ROOT / "phaseS3_swa_visual_confirmation/qkv_alignment_panels"),
        args.out_root / "new_qk_pair_panels",
        limit=6,
    )
    swa_rows = _copy_some(
        _pngs(V81S_ROOT / "phaseS3_swa_visual_confirmation/route_vs_random_panels"),
        args.out_root / "new_swa_kv_panels",
        limit=6,
    )
    ttt_rows = _copy_some(
        _pngs(args.v81tf_phase11 / "ttt_write_less_panels"),
        args.out_root / "new_ttt_write_less_panels",
        limit=4,
    )
    merge_rows = _copy_some(
        _pngs(args.v81tf_phase11 / "merge_boundary_panels"),
        args.out_root / "new_merge_boundary_panels",
        limit=6,
    )
    review_rows = []
    for group, rows in [
        ("new_qk_pair_panels", qk_rows),
        ("new_swa_kv_panels", swa_rows),
        ("new_ttt_write_less_panels", ttt_rows),
        ("new_merge_boundary_panels", merge_rows),
    ]:
        for row in rows:
            review_rows.append({"group": group, **row, "review_status": "present_existing_artifact"})
    _write_csv(args.out_root / "visual_review.csv", review_rows)

    group_counts = {
        "new_qk_pair_panels": {"count": len(qk_rows), "ok": sum(1 for row in qk_rows if row["exists"] and row["bytes"] > 0)},
        "new_swa_kv_panels": {"count": len(swa_rows), "ok": sum(1 for row in swa_rows if row["exists"] and row["bytes"] > 0)},
        "new_ttt_write_less_panels": {"count": len(ttt_rows), "ok": sum(1 for row in ttt_rows if row["exists"] and row["bytes"] > 0)},
        "new_merge_boundary_panels": {"count": len(merge_rows), "ok": sum(1 for row in merge_rows if row["exists"] and row["bytes"] > 0)},
    }
    required_ok = all(value["count"] > 0 and value["ok"] == value["count"] for value in group_counts.values())
    audit = {
        "schema": "acl2_v81s_phaseS14_rediscovery_v1",
        "root": str(args.out_root),
        "s5_summary": str(args.s5_summary),
        "v81tf_phase11_source": str(args.v81tf_phase11),
        "failed_swa_question_count": len(failed_swa),
        "failed_long_question_count": len(failed_long),
        "group_counts": group_counts,
        "visual_review_rows": len(review_rows),
        "visual_audit_gate_pass": bool(required_ok and failed_swa and failed_long),
        "note": "Panels are copied from existing audited v81S/v81TF artifacts; no new visual evidence is fabricated.",
    }
    (args.out_root / "visual_integrity_audit.json").write_text(
        json.dumps(_jsonable(audit), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.out_root / "visual_insight.md").write_text(_render_insight(s5_summary, audit), encoding="utf-8")
    (args.out_root / "new_hypothesis_bank.md").write_text(_render_hypotheses(), encoding="utf-8")
    print(json.dumps(_jsonable(audit), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
