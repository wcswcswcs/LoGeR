#!/usr/bin/env python3
"""Build diagnostic READ/SWA confirmation maps for v81 long windows."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


DEFAULT_ROWS = Path(
    "results/kitti01_hmc_v2/"
    "acl2_v81tf_longwindow_semantic_three_memory_control/"
    "report_final/phase1_long_window_cluster_bank/long_window_cluster_rows.csv"
)
DEFAULT_V80_REPORT = Path(
    "results/kitti01_hmc_v2/"
    "acl2_v80tf_multiseq_goodbad_semantic_three_memory_control/report_final"
)
DEFAULT_OUT_DIR = Path(
    "results/kitti01_hmc_v2/"
    "acl2_v81tf_longwindow_semantic_three_memory_control/"
    "report_final/phase4_read_swa_confirmation"
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_json_list(text: str) -> list[str]:
    try:
        data = json.loads(text)
    except Exception:
        return []
    return [str(item) for item in data] if isinstance(data, list) else []


def scan_read_pts(root: Path) -> dict[tuple[str, int], Path]:
    out: dict[tuple[str, int], Path] = {}
    for path in root.glob("**/read_cue_patch_dumps/chunk_*_read_cue_patch.pt"):
        match = re.search(r"seq(\d\d).*?/chunk(\d{2,3})/", str(path))
        if not match:
            # Fall back to parent names such as read_swa_seq02...
            match_seq = re.search(r"read_swa_seq(\d\d)", str(path))
            match_chunk = re.search(r"chunk_(\d{2,3})_read_cue_patch", path.name)
            if not (match_seq and match_chunk):
                continue
            key = (match_seq.group(1), int(match_chunk.group(1)))
        else:
            key = (match.group(1), int(match.group(2)))
        out.setdefault(key, path)
    return out


def heat(arr: np.ndarray, path: Path) -> None:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        arr = np.zeros((19, 66), dtype=np.float32)
    amin = float(np.nanmin(arr))
    amax = float(np.nanmax(arr))
    norm = (arr - amin) / (amax - amin) if amax > amin else np.zeros_like(arr)
    rgb = np.stack([norm * 255, norm * 180, (1.0 - norm) * 255], axis=-1)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).resize((396, 114), resample=Image.Resampling.NEAREST).save(path)


def read_mass(path: Path, out_png: Path) -> tuple[float | None, str]:
    try:
        payload = torch.load(path, map_location="cpu")
        tensors = payload.get("tensors") if isinstance(payload, dict) else None
        tensor = tensors.get("read_patch_final") if isinstance(tensors, dict) else None
        if not hasattr(tensor, "detach"):
            return None, "missing_read_patch_final"
        arr = tensor.detach().float().cpu().numpy()
        grid = payload.get("patch_grid") or [19, 66]
        map2 = arr.reshape(-1, int(grid[0]), int(grid[1])).mean(axis=0)
        heat(map2, out_png)
        mass = float((arr > 0.5).mean())
        stats = payload.get("stats") if isinstance(payload, dict) else {}
        if isinstance(stats, dict):
            mass = float(stats.get("read_active_gt050_mass", mass))
        return mass, "ok"
    except Exception as exc:
        return None, type(exc).__name__


def support_path_from_summary(summary_path: Path) -> Path | None:
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    for key in ("support_path", "source_support_map"):
        value = data.get(key)
        if value and Path(str(value)).is_file():
            return Path(str(value))
    return None


def swa_mass(path: Path, out_png: Path) -> tuple[float | None, str]:
    try:
        payload = torch.load(path, map_location="cpu")
        score = payload.get("score_overlap") if isinstance(payload, dict) else None
        if not hasattr(score, "detach"):
            return None, "missing_score_overlap"
        arr = score.detach().float().cpu().numpy()
        tokens = int(payload.get("tokens_per_frame") or arr.shape[-1])
        map2 = arr.reshape(-1, tokens).mean(axis=0)
        grid_h = 19
        grid_w = max(tokens // grid_h, 1)
        map2 = map2[: grid_h * grid_w].reshape(grid_h, grid_w)
        heat(map2, out_png)
        # Higher score means stronger support in these v80 support maps.
        return float((arr >= 0.5).mean()), "ok"
    except Exception as exc:
        return None, type(exc).__name__


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, default=DEFAULT_ROWS)
    parser.add_argument("--v80-report-root", type=Path, default=DEFAULT_V80_REPORT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    rows = read_csv(args.rows)
    read_pts = scan_read_pts(args.v80_report_root)
    out_rows: list[dict[str, Any]] = []
    for row in rows:
        seq = row["seq"]
        chunks = list(range(int(row["chunk_start"]), int(row["chunk_end"]) + 1))
        read_values = []
        read_sources = []
        for chunk in chunks:
            path = read_pts.get((seq, chunk))
            if not path:
                continue
            mass, status = read_mass(path, args.out_dir / "read_confirmation_maps" / f"{row['window_id']}_chunk{chunk:03d}_read.png")
            if mass is not None:
                read_values.append(mass)
                read_sources.append(str(path))
        swa_values = []
        swa_sources = []
        for source in parse_json_list(row.get("selected_chunk_sources", "")):
            support = support_path_from_summary(Path(source))
            if support is None:
                continue
            chunk_match = re.search(r"chunk[_]?(\d{2,3})", support.name)
            chunk_label = chunk_match.group(1) if chunk_match else "unk"
            mass, status = swa_mass(support, args.out_dir / "swa_confirmation_maps" / f"{row['window_id']}_chunk{chunk_label}_swa.png")
            if mass is not None:
                swa_values.append(mass)
                swa_sources.append(str(support))
        read_mass_mean = float(np.mean(read_values)) if read_values else None
        swa_mass_mean = float(np.mean(swa_values)) if swa_values else None
        if read_mass_mean is not None and swa_mass_mean is not None and max(read_mass_mean, swa_mass_mean) > 0:
            alignment = min(read_mass_mean, swa_mass_mean) / max(read_mass_mean, swa_mass_mean)
        else:
            alignment = None
        out_rows.append(
            {
                "window_id": row["window_id"],
                "seq": seq,
                "case_type": row["case_type"],
                "read_confirmed_stable_mass": read_mass_mean,
                "swa_confirmed_stable_mass": swa_mass_mean,
                "read_swa_alignment": alignment,
                "random_confirmation_alignment": "",
                "read_source_count": len(read_sources),
                "swa_source_count": len(swa_sources),
                "read_sources": json.dumps(read_sources, ensure_ascii=False),
                "swa_sources": json.dumps(swa_sources, ensure_ascii=False),
                "confirmation_scope": "diagnostic_existing_v80_hook_proxy",
            }
        )
    write_csv(args.out_dir / "read_swa_confirmation_rows.csv", out_rows)
    print(json.dumps({"out_dir": str(args.out_dir), "rows": len(out_rows)}, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
