#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np


SEQ_FIELD_RE = re.compile(r"seq(?P<seq>\d+)_(?P<field>.+)\.npy$")


def summarize_array(path: Path) -> dict[str, Any]:
    arr = np.load(path, mmap_mode="r")
    return {
        "path": str(path),
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
    }


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# V116 Semantic Cue Redesign Readiness Audit",
        "",
        "Scope: determine whether the current v116/v113 semantic projection artifacts already contain object identity, temporal continuity, or internal alignment fields needed by the post-No-Go recommendation.",
        "",
        "## Current Projection Fields",
        "",
    ]
    for seq, fields in sorted(payload["semantic_projection_by_seq"].items()):
        field_names = ", ".join(sorted(fields))
        lines.append(f"- seq {seq}: {field_names}")
    lines.extend(["", "## Readiness", ""])
    for item in payload["readiness"]:
        lines.append(f"- {item['cue_family']}: ready={item['ready']}; reason={item['reason']}")
    lines.extend(["", "## Recommendation", "", payload["recommendation"]])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit whether current v116 artifacts support the next semantic cue redesign.")
    parser.add_argument("--results-root", default="results/acl2_v116tf_fast_semantic_causal_memory_influence")
    parser.add_argument(
        "--semantic-root",
        default="results/acl2_v113hs_horizonstream_semantic_aware_geometric_evidence_influence/semantic_projection",
    )
    args = parser.parse_args()

    results_root = Path(args.results_root).resolve()
    semantic_root = Path(args.semantic_root).resolve()
    fields_by_seq: dict[str, dict[str, Any]] = {}
    for path in sorted(semantic_root.glob("seq*_*.npy")):
        match = SEQ_FIELD_RE.match(path.name)
        if not match:
            continue
        seq = match.group("seq")
        field = match.group("field")
        fields_by_seq.setdefault(seq, {})[field] = summarize_array(path)

    all_fields = {field for fields in fields_by_seq.values() for field in fields}
    object_fields = sorted(f for f in all_fields if any(key in f.lower() for key in ["object", "instance", "track", "masklet"]))
    continuity_fields = sorted(f for f in all_fields if any(key in f.lower() for key in ["continuity", "persistent", "persistence", "track"]))
    alignment_fields = sorted(
        f for f in all_fields if any(key in f.lower() for key in ["alignment", "residual", "novelty", "uncertainty", "contradiction"])
    )

    readiness = [
        {
            "cue_family": "object_identity_or_persistence",
            "ready": bool(object_fields),
            "available_fields": object_fields,
            "reason": "No object/instance/track/masklet field is present in the semantic projection."
            if not object_fields
            else "Object/track-like fields exist and can be inspected before a contrast.",
        },
        {
            "cue_family": "temporal_continuity",
            "ready": bool(continuity_fields),
            "available_fields": continuity_fields,
            "reason": "No continuity/persistence/track field is present in the semantic projection."
            if not continuity_fields
            else "Continuity-like fields exist and can be inspected before a contrast.",
        },
        {
            "cue_family": "internal_alignment_or_residual",
            "ready": bool(alignment_fields),
            "available_fields": alignment_fields,
            "reason": "No alignment/residual/novelty/uncertainty field is present in the semantic projection."
            if not alignment_fields
            else "Alignment-like fields exist and can be inspected before a contrast.",
        },
    ]

    recommendation = (
        "Do not continue v116 by sweeping frame-level stable/risk role mass. The next runnable semantic-cue redesign needs new "
        "instrumentation or sidecar artifacts for object identity/track persistence and/or runtime internal alignment residuals. "
        "Use the tight rowmean+MRT scale-delta branch as a generic carrier/safety baseline control when those cues exist."
    )

    payload = {
        "results_root": str(results_root),
        "semantic_root": str(semantic_root),
        "semantic_projection_by_seq": fields_by_seq,
        "all_projection_fields": sorted(all_fields),
        "readiness": readiness,
        "ready_for_next_semantic_cue_runtime_action": all(item["ready"] for item in readiness),
        "recommendation": recommendation,
    }
    out_dir = results_root / "carrier_diagnosis"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "SEMANTIC_CUE_REDESIGN_READINESS.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_report(out_dir / "SEMANTIC_CUE_REDESIGN_READINESS.md", payload)
    print(
        json.dumps(
            {
                "ready_for_next_semantic_cue_runtime_action": payload["ready_for_next_semantic_cue_runtime_action"],
                "all_projection_fields": payload["all_projection_fields"],
                "recommendation": recommendation,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
