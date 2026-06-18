#!/usr/bin/env python3
"""Materialize v70 Phase1 path-tap manifest from existing v68 feature dumps.

This does not fabricate online HMC attention. It registers real LoGeR feature
taps so downstream smoke tools can compute a clearly labeled proxy attention.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from v70_radio_sidecar_common import parse_chunks, torch_load, utc_now, write_json, write_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v68-feature-dir", type=Path, required=True)
    parser.add_argument("--target-chunks", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tap", default="global_k_raw_patchvec_layers")
    parser.add_argument("--layers", default="5,7")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    chunks = parse_chunks(args.target_chunks)
    wanted_layers = parse_chunks(args.layers)
    entries: list[dict[str, Any]] = []
    for chunk_id in chunks:
        feature_path = args.v68_feature_dir / f"chunk_{chunk_id:03d}.pt"
        if not feature_path.exists():
            entries.append({"chunk_id": chunk_id, "available": False, "reason": f"missing {feature_path}"})
            continue
        payload = torch_load(feature_path)
        tensor_key = f"tap::{args.tap}"
        tensor = payload.get(tensor_key) if isinstance(payload, dict) else None
        if tensor is None or not hasattr(tensor, "shape"):
            entries.append({"chunk_id": chunk_id, "available": False, "reason": f"missing {tensor_key}", "feature_path": str(feature_path)})
            continue
        tap_meta = dict((payload.get("taps") or {}).get(args.tap) or {})
        selected_layers = [int(x) for x in tap_meta.get("selected_layers", [])]
        layer_positions = []
        for layer in wanted_layers:
            if selected_layers and layer in selected_layers:
                layer_positions.append({"layer": layer, "position": selected_layers.index(layer)})
            elif not selected_layers and 0 <= layer < int(tensor.shape[1]):
                layer_positions.append({"layer": layer, "position": layer})
        entries.append(
            {
                "chunk_id": int(chunk_id),
                "available": bool(layer_positions),
                "feature_path": str(feature_path),
                "tap": args.tap,
                "tensor_key": tensor_key,
                "tap_type": "global_k_feature_proxy",
                "selected_layers": selected_layers,
                "requested_layers": wanted_layers,
                "layer_positions": layer_positions,
                "feature_shape": list(tensor.shape),
                "start_frame": int(payload.get("start_frame", -1)),
                "end_frame": int(payload.get("end_frame", -1)),
                "patch_grid": list(payload.get("patch_grid", [])),
                "reason": "" if layer_positions else "requested_layers_not_available",
            }
        )
    manifest = {
        "format": "v70_phase1_path_taps_manifest_v1",
        "created_at": utc_now(),
        "source": "v68_phaseC_target_feature_dumps",
        "v68_feature_dir": str(args.v68_feature_dir),
        "tap_materialization_scope": "feature_proxy_only_no_online_attention_logits",
        "entries": entries,
    }
    write_json(args.out_dir / "path_taps_manifest.json", manifest)
    write_text(
        args.out_dir / "README.md",
        "\n".join(
            [
                "# v70 Phase1 Path Taps",
                "",
                "This directory registers existing v68 target feature dumps as v70 Phase1 feature-proxy taps.",
                "It does not contain online HMC attention logits and must not be used as evidence for J_v70, ATE, or online method success by itself.",
                "",
                f"available_entries={sum(1 for e in entries if e.get('available'))}/{len(entries)}",
            ]
        )
        + "\n",
    )
    print(json.dumps({"out_dir": str(args.out_dir), "available_entries": sum(1 for e in entries if e.get("available")), "entries": len(entries)}, indent=2))


if __name__ == "__main__":
    main()

