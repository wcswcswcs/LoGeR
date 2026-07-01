#!/usr/bin/env python3
"""Build a v92 sidecar manifest from available RADIO/object-tracklet candidates."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.v86_soft_latent_utils import write_csv, write_json  # noqa: E402
from tools.v92_semantic_policy_carrier_utils import ROOT  # noqa: E402


DEFAULT_OUT = ROOT / "phase7_data_source_expansion"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-candidate-rows", type=Path, default=DEFAULT_OUT / "semantic_source_expansion_candidate_rows.csv")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def _bool_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    return df[col].astype(str).str.lower().isin({"true", "1", "yes"})


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(args.source_candidate_rows)
    manifest_rows: list[dict[str, Any]] = []
    for _, row in rows.iterrows():
        has_radio = str(row.get("has_radio", "")).lower() in {"true", "1", "yes"}
        has_stage_c = str(row.get("stage_c_pair_manifest_available", "")).lower() in {"true", "1", "yes"}
        has_track_meta = str(row.get("sam31_track_metadata_available", "")).lower() in {"true", "1", "yes"}
        object_identity = str(row.get("object_identity_available", "")).lower() in {"true", "1", "yes"}
        component_proxy = str(row.get("component_tracklet_available", "")).lower() in {"true", "1", "yes"}
        if has_radio or has_stage_c or has_track_meta or component_proxy:
            manifest_rows.append(
                {
                    "seq": str(row.get("seq", "")).zfill(2),
                    "prev_chunk": int(row.get("prev_chunk")),
                    "curr_chunk": int(row.get("curr_chunk")),
                    "pair_id": str(row.get("pair_id")),
                    "radio_pair_sidecar_available": has_radio,
                    "stage_c_pair_manifest_available": has_stage_c,
                    "sam31_track_metadata_available": has_track_meta,
                    "object_identity_available": object_identity,
                    "component_tracklet_available": component_proxy,
                    "source_scope": str(row.get("source_scope", "")),
                    "radio_sidecar_paths": row.get("radio_sidecar_paths", ""),
                    "radio_r5_sidecar_paths": row.get("radio_r5_sidecar_paths", ""),
                    "stage_c_manifest_paths": row.get("stage_c_manifest_paths", ""),
                    "sam31_track_metadata_path": row.get("sam31_track_metadata_path", ""),
                    "object_identity_availability_reason": row.get("object_identity_availability_reason", ""),
                }
            )
    labelled = rows[rows["base_case_type"].astype(str).isin(["bad", "good"])] if "base_case_type" in rows.columns else pd.DataFrame()
    summary = {
        "phase": "Phase7_radio_tracklet_sidecar_manifest",
        "input_rows": int(len(rows)),
        "manifest_rows": int(len(manifest_rows)),
        "radio_pair_rows": int(_bool_series(rows, "has_radio").sum()),
        "stage_c_pair_rows": int(_bool_series(rows, "stage_c_pair_manifest_available").sum()),
        "sam31_metadata_rows": int(_bool_series(rows, "sam31_track_metadata_available").sum()),
        "object_identity_rows": int(_bool_series(rows, "object_identity_available").sum()),
        "component_tracklet_proxy_rows": int(_bool_series(rows, "component_tracklet_available").sum()),
        "labelled_rows": int(len(labelled)),
        "object_identity_labelled_rows": int(_bool_series(labelled, "object_identity_available").sum()) if len(labelled) else 0,
        "component_tracklet_labelled_rows": int(_bool_series(labelled, "component_tracklet_available").sum()) if len(labelled) else 0,
        "sidecar_build_status": "candidate_manifest_only_no_runtime_join",
        "source_scope": "component_tracklet_proxy_plus_unjoined_sidecar_candidates",
        "no_radio_success_claim": True,
        "no_object_identity_success_claim": True,
        "runtime_action_allowed": False,
        "ttt_allowed": False,
    }
    summary["phase7_sidecar_manifest_gate_pass"] = bool(summary["manifest_rows"] == summary["input_rows"] and summary["component_tracklet_proxy_rows"] >= 49)
    if not summary["phase7_sidecar_manifest_gate_pass"]:
        summary["blocker"] = "phase7_sidecar_manifest_incomplete"
    write_csv(args.out_dir / "radio_tracklet_sidecar_manifest_rows.csv", manifest_rows)
    write_json(args.out_dir / "radio_tracklet_sidecar_summary.json", summary)
    print(f"phase7_sidecar_manifest_gate_pass={summary['phase7_sidecar_manifest_gate_pass']}")
    print(f"input_rows={summary['input_rows']}")
    print(f"radio_pair_rows={summary['radio_pair_rows']}")
    print(f"object_identity_rows={summary['object_identity_rows']}")
    print(f"component_tracklet_proxy_rows={summary['component_tracklet_proxy_rows']}")
    print(f"sidecar_build_status={summary['sidecar_build_status']}")


if __name__ == "__main__":
    main()
