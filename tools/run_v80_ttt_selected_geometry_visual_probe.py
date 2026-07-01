#!/usr/bin/env python3
"""Run candidate-specific TTT visual probes selected by geometry error maps."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

try:
    from PIL import Image, ImageDraw
except Exception:  # pragma: no cover - visual combining is optional.
    Image = None
    ImageDraw = None


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.run_v78_ttt_long_window_regime_action_smoke import (  # noqa: E402
    CASES,
    DEFAULT_CHECKPOINT,
    DEFAULT_CONDA,
    DEFAULT_CONFIG,
    DEFAULT_CUDA_ALLOC_CONF,
    DEFAULT_DATA_ROOT,
    _build_command,
    _jsonable,
    _parse_ints,
    _window_bounds,
)


DEFAULT_SELECTED_TARGETS = Path(
    "results/kitti01_hmc_v2/acl2_v80tf_multiseq_goodbad_semantic_three_memory_control/"
    "report_final/phase5_ttt_geometry_error_visual_bridge/selected_visual_targets.csv"
)
DEFAULT_OUT_ROOT = Path(
    "results/kitti01_hmc_v2/acl2_v80tf_multiseq_goodbad_semantic_three_memory_control/"
    "report_final/phase5_ttt_geometry_selected_visual_probe"
)
PAIRED_RANDOM_CONTROLS = {
    "LW23_TTT_OVERLAP_ROLECAL_TTL_B0_PRIOR100": "LW24_TTT_OVERLAP_ROLECAL_TTL_RANDOM_ROLE_B0_PRIOR100",
    "LW25_TTT_OVERLAP_ROLECAL_TTL_B0_PRIOR108": "LW26_TTT_OVERLAP_ROLECAL_TTL_RANDOM_ROLE_B0_PRIOR108",
    "LW27_TTT_OVERLAP_ROLESTRONG_TTL_B0_PRIOR108": "LW28_TTT_OVERLAP_ROLESTRONG_TTL_RANDOM_ROLE_B0_PRIOR108",
    "LW34_TTT_OVERLAP_SEMANTIC_HARM_FILTERONLY_B0_PRIOR100": "LW35_TTT_OVERLAP_SEMANTIC_HARM_FILTERONLY_RANDOM_ROLE_B0_PRIOR100",
    "LW36_TTT_READHARM_LOCAL_VETO_B0": "LW37_TTT_READHARM_LOCAL_VETO_RANDOM_ROLE_B0",
    "LW38_TTT_READHARM_NEGATIVE_LOCAL_VETO_B0": "LW39_TTT_READHARM_NEGATIVE_LOCAL_VETO_RANDOM_ROLE_B0",
    "LW40_TTT_V80_HEAD_SUPPORT_VETO_B0": "LW41_TTT_V80_HEAD_SUPPORT_RANDOM_VETO_B0",
    "LW42_TTT_V80_SELECTED_WRITE_SUPPORT_VETO_B0": "LW43_TTT_V80_SELECTED_WRITE_SUPPORT_RANDOM_VETO_B0",
}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(row.get(key), ensure_ascii=False)
                    if isinstance(row.get(key), (dict, list))
                    else row.get(key)
                    for key in fieldnames
                }
            )


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _parse_cases(row: dict[str, Any], include_native: bool, include_control: bool) -> list[str]:
    cases: list[str] = []
    baseline = str(row.get("baseline") or "").strip()
    candidate = str(row.get("candidate") or "").strip()
    control = str(row.get("paired_random_control") or PAIRED_RANDOM_CONTROLS.get(candidate, "") or "").strip()
    native = str(row.get("native_baseline") or "LW0_READPATH_NATIVE").strip()
    for case in ([native] if include_native else []) + [baseline, candidate] + ([control] if include_control else []):
        if case and case not in cases:
            cases.append(case)
    return cases


def _frames_for_chunk(target_frame: int, chunk: int, chunk_size: int, overlap: int) -> list[int]:
    stride = int(chunk_size) - int(overlap)
    start = int(chunk) * stride
    end = start + int(chunk_size)
    candidates = [max(start, int(target_frame) - 3), int(target_frame), min(end - 1, int(target_frame) + 3)]
    return sorted({frame for frame in candidates if start <= frame < end})


def _stage_c_masklet(stage_c_root: Path, chunk: int) -> Path:
    matches = sorted(stage_c_root.glob(f"chunk_{int(chunk):03d}_*/masklet.pt"))
    if not matches:
        raise FileNotFoundError(f"missing stage-c masklet for chunk {chunk}: {stage_c_root}/chunk_{chunk:03d}_*/masklet.pt")
    return matches[0]


def _build_case_command(args: argparse.Namespace, row: dict[str, Any], case: str, out_dir: Path) -> list[str]:
    seq = str(row["seq"]).zfill(2)
    chunk = int(row["visual_probe_chunk"])
    run_chunks_text = str(row.get("run_chunks") or "").strip()
    chunks = _parse_ints(run_chunks_text) if run_chunks_text else [chunk]
    stage_dir = Path(str(row.get("stage_c_cache_dir") or f"results/kitti_preprocess/{seq}/stage_c_cache_semantic_chunks"))
    bounds = _window_bounds(chunks, int(args.chunk_size), int(args.chunk_overlap))
    build_args = argparse.Namespace(
        seq=seq,
        conda=args.conda,
        conda_env=args.conda_env,
        data_root=args.data_root,
        checkpoint=args.checkpoint,
        config=args.config,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        stage_c_cache_dir=stage_dir,
    )
    cmd = _build_command(build_args, case, out_dir, bounds)
    cmd.extend(
        [
            "--v68_export_full_pca_debug",
            "1",
            "--v68_pca_max_feature_dim",
            str(args.v68_pca_max_feature_dim),
            "--v68_layer_pca_feature_dir",
            str(out_dir / "pca_features"),
            "--v68_pca_taps",
            "ttt_operator_output,ttt_update_term,ttt_final_output",
            "--v68_pca_layers",
            str(args.v68_pca_layers),
            "--ttt_spatial_post_delta_map_dump_dir",
            str(out_dir / "ttt_spatial_post_delta_maps"),
            "--ttt_spatial_post_delta_map_dump_dtype",
            str(args.ttt_spatial_post_delta_map_dump_dtype),
        ]
    )
    return cmd


def _visual_cmd(args: argparse.Namespace, row: dict[str, Any], case_dir: Path, visual_dir: Path) -> list[str]:
    seq = str(row["seq"]).zfill(2)
    chunk = int(row["visual_probe_chunk"])
    target_frame = int(float(row["visual_probe_frames_hint"]))
    stage_dir = Path(str(row.get("stage_c_cache_dir") or f"results/kitti_preprocess/{seq}/stage_c_cache_semantic_chunks"))
    pca_pt = case_dir / "pca_features" / f"chunk_{chunk:03d}.pt"
    post_delta_pt = case_dir / "ttt_spatial_post_delta_maps" / f"chunk_{chunk:03d}_ttt_spatial_post_delta_map.pt"
    frames = ",".join(str(frame) for frame in _frames_for_chunk(target_frame, chunk, int(args.chunk_size), int(args.chunk_overlap)))
    cmd = [
        str(args.python),
        "tools/visualize_v78_phase4_ttt_output_separated.py",
        "--pca-pt",
        str(pca_pt),
        "--stage-c-masklet",
        str(_stage_c_masklet(stage_dir, chunk)),
        "--rgb-dir",
        str(args.data_root / "sequences" / seq / "image_2"),
        "--out-dir",
        str(visual_dir),
        "--frames",
        frames,
        "--operator-layers",
        str(args.operator_layers),
        "--single-layer",
        str(args.single_layer),
        "--panel-width",
        str(args.panel_width),
    ]
    cmd.extend(["--post-delta-pt", str(post_delta_pt)])
    return cmd


def _run_subprocess(cmd: list[str], cwd: Path, env: dict[str, str], log_path: Path) -> tuple[int, float]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with log_path.open("w", encoding="utf-8") as handle:
        proc = subprocess.run(cmd, cwd=str(cwd), env=env, stdout=handle, stderr=subprocess.STDOUT, check=False)
    return int(proc.returncode), float(time.time() - start)


def _run_job(job: dict[str, Any]) -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    env["CUDA_VISIBLE_DEVICES"] = str(job["gpu"])
    if str(job.get("cuda_alloc_conf") or "").strip():
        env["PYTORCH_CUDA_ALLOC_CONF"] = str(job["cuda_alloc_conf"]).strip()
    if bool(job.get("disable_ttt_compile", False)):
        env["LOGER_TTT_DISABLE_COMPILE"] = "1"
    case_dir = Path(job["case_dir"])
    visual_dir = Path(job["visual_dir"])
    expected_pca = Path(job["expected_pca"])
    if bool(job.get("skipped")):
        job["pipeline_returncode"] = 0
        job["pipeline_duration_sec"] = 0.0
    else:
        rc, dt = _run_subprocess(job["pipeline_cmd"], REPO_ROOT, env, case_dir / "pipeline.log")
        job["pipeline_returncode"] = rc
        job["pipeline_duration_sec"] = dt
    if int(job["pipeline_returncode"]) == 0 and expected_pca.exists():
        visual_cmd = job["visual_cmd"]
        rc, dt = _run_subprocess(visual_cmd, REPO_ROOT, env, visual_dir / "visualize.log")
        job["visual_returncode"] = rc
        job["visual_duration_sec"] = dt
    else:
        job["visual_returncode"] = -1
        job["visual_duration_sec"] = 0.0
    audit_path = visual_dir / "visual_integrity_audit.json"
    if audit_path.exists():
        try:
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            audit = {}
        job["visual_gate_pass"] = bool(audit.get("gate_pass"))
        job["visual_audit"] = str(audit_path)
    else:
        job["visual_gate_pass"] = False
        job["visual_audit"] = str(audit_path)
    return job


def _label_image(img: "Image.Image", text: str) -> "Image.Image":
    out = img.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    try:
        bbox = draw.textbbox((5, 4), text)
        label_height = max(24, int(bbox[3]) + 6)
    except Exception:
        # Older PIL builds can require a TrueType font for textbbox.
        label_height = 24
    draw.rectangle((0, 0, out.width, label_height), fill=(0, 0, 0))
    draw.text((5, 4), text, fill=(255, 255, 255))
    return out


def _combine_target_images(out_root: Path, jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if Image is None or ImageDraw is None:
        return []
    grouped: dict[str, list[dict[str, Any]]] = {}
    for job in jobs:
        if int(job.get("visual_returncode") or 0) != 0:
            continue
        grouped.setdefault(str(job["target_id"]), []).append(job)
    suffixes = {
        "post_zp_delta_overlay": "TTT_post_zp_delta_overlay.png",
        "write_role_mass_panel": "TTT_write_role_mass_panel.png",
        "update_term": "TTT_update_term_L",
        "final_output": "TTT_final_output_L",
    }
    rows: list[dict[str, Any]] = []
    for target_id, target_jobs in grouped.items():
        target_jobs = sorted(target_jobs, key=lambda j: str(j["case"]))
        for kind, needle in suffixes.items():
            images: list[Image.Image] = []
            sources: list[str] = []
            for job in target_jobs:
                visual_dir = Path(job["visual_dir"])
                matches = sorted(visual_dir.glob(f"*{needle}*")) if "L" in needle else sorted(visual_dir.glob(f"*{needle}"))
                if not matches:
                    continue
                img = Image.open(matches[0]).convert("RGB")
                images.append(_label_image(img, str(job["case"])))
                sources.append(str(matches[0]))
            if len(images) < 2:
                continue
            width = max(img.width for img in images)
            height = sum(img.height for img in images)
            canvas = Image.new("RGB", (width, height), (0, 0, 0))
            y = 0
            for img in images:
                canvas.paste(img, (0, y))
                y += img.height
            out_path = out_root / "combined" / target_id / f"{kind}_case_compare.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            canvas.save(out_path)
            rows.append(
                {
                    "target_id": target_id,
                    "visual_kind": kind,
                    "combined_png": str(out_path),
                    "source_images": sources,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-targets", type=Path, default=DEFAULT_SELECTED_TARGETS)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--target-limit", type=int, default=6)
    parser.add_argument("--gpus", default="0,1,2,3,4,5")
    parser.add_argument("--include-native", action="store_true")
    parser.add_argument("--no-control", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--conda", type=Path, default=DEFAULT_CONDA)
    parser.add_argument("--python", type=Path, default=Path("/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python"))
    parser.add_argument("--conda-env", default="loger")
    parser.add_argument("--cuda-alloc-conf", default=DEFAULT_CUDA_ALLOC_CONF)
    parser.add_argument("--disable-ttt-compile", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--chunk-overlap", type=int, default=3)
    parser.add_argument("--v68-pca-layers", default="6,14,18")
    parser.add_argument("--v68-pca-max-feature-dim", type=int, default=8)
    parser.add_argument("--operator-layers", default="6,14,18")
    parser.add_argument("--single-layer", type=int, default=18)
    parser.add_argument("--panel-width", type=int, default=420)
    parser.add_argument("--ttt-spatial-post-delta-map-dump-dtype", default="float16")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--manual-seq", default="")
    parser.add_argument("--manual-window-id", default="")
    parser.add_argument("--manual-case-type", default="bad")
    parser.add_argument("--manual-case-rank", default="0")
    parser.add_argument("--manual-chunk", default="")
    parser.add_argument("--manual-run-chunks", default="")
    parser.add_argument("--manual-frame", default="")
    parser.add_argument("--manual-candidate", default="")
    parser.add_argument("--manual-baseline", default="LW1_TTT_SEMANTIC_BASE")
    parser.add_argument("--manual-control", default="")
    args = parser.parse_args()

    rows = _read_csv(args.selected_targets)
    if str(args.manual_seq).strip() and str(args.manual_chunk).strip() and str(args.manual_frame).strip():
        seq = str(args.manual_seq).zfill(2)
        candidate = str(args.manual_candidate or "").strip()
        control = str(args.manual_control or PAIRED_RANDOM_CONTROLS.get(candidate, "") or "").strip()
        chunk = int(float(args.manual_chunk))
        frame = int(float(args.manual_frame))
        window_id = str(args.manual_window_id or f"seq{seq}_manual_chunk{chunk:03d}_{args.manual_case_type}_rank{args.manual_case_rank}")
        rows.append(
            {
                "window_id": window_id,
                "seq": seq,
                "case_type": str(args.manual_case_type),
                "case_rank": str(args.manual_case_rank),
                "candidate": candidate,
                "baseline": str(args.manual_baseline),
                "paired_random_control": control,
                "visual_probe_chunk": str(chunk),
                "run_chunks": str(args.manual_run_chunks or chunk),
                "visual_probe_frames_hint": str(frame),
                "selection_rank": "manual",
                "selection_reason": "manual_geometry_microprobe_seed",
                "stage_c_cache_dir": str(Path(f"results/kitti_preprocess/{seq}/stage_c_cache_semantic_chunks")),
                "rgb_dir": str(args.data_root / "sequences" / seq / "image_2"),
            }
        )
    rows = rows[: int(args.target_limit)] if int(args.target_limit) > 0 else rows
    gpus = _parse_ints(args.gpus)
    if not gpus:
        raise ValueError("--gpus is empty")
    args.out_root.mkdir(parents=True, exist_ok=True)
    jobs: list[dict[str, Any]] = []
    gpu_cursor = 0
    for row_idx, row in enumerate(rows):
        seq = str(row["seq"]).zfill(2)
        frame = int(float(row["visual_probe_frames_hint"]))
        chunk = int(float(row["visual_probe_chunk"]))
        target_id = f"{row['window_id']}_frame{frame:06d}_chunk{chunk:03d}_rank{row.get('selection_rank', row_idx + 1)}"
        for case in _parse_cases(row, include_native=bool(args.include_native), include_control=not bool(args.no_control)):
            if case not in CASES:
                raise ValueError(f"unknown case from selected target: {case}")
            case_dir = args.out_root / "targets" / target_id / case / "pipeline"
            visual_dir = args.out_root / "targets" / target_id / case / "visual"
            expected_pca = case_dir / "pca_features" / f"chunk_{chunk:03d}.pt"
            expected_trajectory = case_dir / f"{seq}.txt"
            pipeline_cmd = _build_case_command(args, row, case, case_dir)
            visual_cmd = _visual_cmd(args, row, case_dir, visual_dir)
            skipped = bool(args.skip_existing and expected_pca.exists() and expected_trajectory.exists())
            jobs.append(
                {
                    "target_id": target_id,
                    "target_row_index": row_idx,
                    "window_id": row["window_id"],
                    "seq": seq,
                    "frame": frame,
                    "chunk": chunk,
                    "case": case,
                    "gpu": int(gpus[gpu_cursor % len(gpus)]),
                    "case_dir": str(case_dir),
                    "visual_dir": str(visual_dir),
                    "expected_pca": str(expected_pca),
                    "expected_trajectory": str(expected_trajectory),
                    "pipeline_cmd": pipeline_cmd,
                    "pipeline_cmd_shell": shlex.join(pipeline_cmd),
                    "visual_cmd": visual_cmd,
                    "visual_cmd_shell": shlex.join(visual_cmd),
                    "cuda_alloc_conf": str(args.cuda_alloc_conf),
                    "disable_ttt_compile": bool(args.disable_ttt_compile),
                    "skipped": skipped,
                    "source_selected_target": row,
                }
            )
            gpu_cursor += 1

    manifest_path = args.out_root / "selected_geometry_visual_probe_manifest.json"
    _write_json(
        manifest_path,
        {
            "schema": "acl2_v80_ttt_selected_geometry_visual_probe_manifest_v1",
            "diagnostic_only": True,
            "method_gate_claimed": False,
            "args": vars(args),
            "jobs": jobs,
        },
    )
    print(json.dumps(_jsonable({"planned_jobs": len(jobs), "manifest": manifest_path, "dry_run": args.dry_run}), indent=2, sort_keys=True))
    if args.dry_run or not jobs:
        return

    completed: list[dict[str, Any]] = []
    jobs_by_gpu: dict[int, list[dict[str, Any]]] = {int(gpu): [] for gpu in gpus}
    for job in jobs:
        jobs_by_gpu[int(job["gpu"])].append(job)

    def run_gpu_queue(queue: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for job in queue:
            result = _run_job(job)
            print(
                f"finished target={result['target_id']} case={result['case']} gpu={result['gpu']} "
                f"pipeline_rc={result['pipeline_returncode']} visual_rc={result['visual_returncode']}"
            )
            out.append(result)
        return out

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(gpus))) as pool:
        futures = [pool.submit(run_gpu_queue, queue) for queue in jobs_by_gpu.values() if queue]
        for future in concurrent.futures.as_completed(futures):
            completed.extend(future.result())

    completed = sorted(completed, key=lambda row: (str(row["target_id"]), str(row["case"])))
    combined = _combine_target_images(args.out_root, completed)
    _write_csv(args.out_root / "selected_geometry_visual_probe_jobs.csv", completed)
    _write_csv(args.out_root / "selected_geometry_visual_probe_combined.csv", combined)
    summary = {
        "schema": "acl2_v80_ttt_selected_geometry_visual_probe_summary_v1",
        "diagnostic_only": True,
        "method_gate_claimed": False,
        "jobs": completed,
        "combined_visuals": combined,
        "all_pipeline_ok": bool(all(int(job.get("pipeline_returncode") or 0) == 0 for job in completed)),
        "all_visual_ok": bool(all(int(job.get("visual_returncode") or 0) == 0 for job in completed)),
        "all_visual_gates_pass": bool(completed and all(bool(job.get("visual_gate_pass")) for job in completed)),
    }
    summary_path = args.out_root / "selected_geometry_visual_probe_summary.json"
    _write_json(summary_path, summary)
    print(
        json.dumps(
            _jsonable(
                {
                    "all_pipeline_ok": summary["all_pipeline_ok"],
                    "all_visual_ok": summary["all_visual_ok"],
                    "all_visual_gates_pass": summary["all_visual_gates_pass"],
                    "combined_visual_count": len(combined),
                    "summary": summary_path,
                }
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
