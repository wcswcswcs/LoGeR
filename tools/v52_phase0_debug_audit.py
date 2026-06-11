#!/usr/bin/env python3
"""Audit v52 tri replay debug fields and report semantics."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]


def _line_hits(path: Path, pattern: str) -> List[Dict[str, Any]]:
    regex = re.compile(pattern)
    hits: List[Dict[str, Any]] = []
    for idx, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        if regex.search(line):
            hits.append({"line": idx, "text": line.strip()})
    return hits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    controller = REPO_ROOT / "loger/pipeline/ttt_write_controller.py"
    runner = REPO_ROOT / "run_pipeline_abc_v2.py"
    report = REPO_ROOT / "tools/v47_adaptive_ttt_writer_report.py"

    controller_text = controller.read_text(encoding="utf-8", errors="replace")
    report_text = report.read_text(encoding="utf-8", errors="replace")
    runner_text = runner.read_text(encoding="utf-8", errors="replace")

    tri_note_hits = _line_hits(controller, r"ttt_two_replay_debug_note.*tri_replay_path")
    two_replay_true_hits = _line_hits(controller, r'ttt_two_replay_applied"\]\s*=\s*True')
    report_two_replay_hits = _line_hits(report, r"ttt_two_replay_applied")

    payload: Dict[str, Any] = {
        "controller": str(controller.relative_to(REPO_ROOT)),
        "runner": str(runner.relative_to(REPO_ROOT)),
        "report": str(report.relative_to(REPO_ROOT)),
        "tri_replay_debug_note_count": len(tri_note_hits),
        "tri_replay_debug_note_hits": tri_note_hits,
        "two_replay_true_hits": two_replay_true_hits,
        "report_two_replay_applied_hits": report_two_replay_hits,
        "runner_uses_tri_replay_applied": "ttt_tri_replay_applied" in runner_text,
        "runner_uses_two_replay_for_tri_replay_count": False,
        "report_uses_tri_replay_applied_count": "ttt_tri_replay_applied_count" in report_text,
        "report_counts_adaptive_writer_split": "adaptive_writer_split_debug_count" in report_text,
        "report_counts_adaptive_writer_fused": "adaptive_writer_fused_debug_count" in report_text,
        "report_uses_two_replay_applied": bool(report_two_replay_hits),
    }
    payload["debug_field_audit_pass"] = (
        payload["tri_replay_debug_note_count"] >= 2
        and payload["runner_uses_tri_replay_applied"]
        and payload["report_uses_tri_replay_applied_count"]
        and payload["report_counts_adaptive_writer_split"]
        and payload["report_counts_adaptive_writer_fused"]
        and not payload["report_uses_two_replay_applied"]
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not payload["debug_field_audit_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
