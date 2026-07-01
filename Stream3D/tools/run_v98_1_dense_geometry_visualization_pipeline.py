#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
STREAM3D_ROOT = ROOT / "Stream3D"
TOOLS_DIR = STREAM3D_ROOT / "tools"
DEFAULT_D4RT_ROOT = ROOT / "Open-d4rt"
DEFAULT_D4RT_CONFIG = DEFAULT_D4RT_ROOT / "checkpoints" / "OpenD4RT_32CLIP_9Dataset_NoAUG" / "model.yaml"
DEFAULT_D4RT_CKPT = DEFAULT_D4RT_ROOT / "checkpoints" / "OpenD4RT_32CLIP_9Dataset_NoAUG" / "opend4rt.ckpt"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _quote(cmd: list[str]) -> str:
    import shlex

    return " ".join(shlex.quote(str(part)) for part in cmd)


def _run_step(name: str, cmd: list[str], *, cwd: Path, log_path: Path, env: dict[str, str] | None = None) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"$ {_quote(cmd)}\n\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        returncode = proc.wait()
    elapsed = float(time.time() - started)
    if returncode != 0:
        raise RuntimeError(f"step {name} failed with returncode={returncode}; see {log_path}")
    return {
        "name": name,
        "command": cmd,
        "command_text": _quote(cmd),
        "log_path": str(log_path),
        "returncode": int(returncode),
        "seconds": elapsed,
    }


def _stop_existing_viewers() -> list[int]:
    out = subprocess.run(
        ["pgrep", "-f", "serve_v98_1_da3_variant_geometry_viewer.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    pids = [int(line.strip()) for line in out.stdout.splitlines() if line.strip().isdigit()]
    for pid in pids:
        if pid == os.getpid():
            continue
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    if pids:
        time.sleep(1.0)
    return pids


def _start_viewer(
    args: argparse.Namespace,
    final_root: Path,
    summary_json: Path,
    viewer_npz: Path,
    *,
    default_variant: str,
) -> dict[str, Any]:
    if bool(args.stop_existing_viewer):
        stopped = _stop_existing_viewers()
    else:
        stopped = []
    status_json = final_root / "viewer_server_status.json"
    log_path = final_root / "viewer_server.log"
    cmd = [
        sys.executable,
        str(TOOLS_DIR / "serve_v98_1_da3_variant_geometry_viewer.py"),
        "--output-root",
        str(final_root),
        "--summary-json",
        str(summary_json),
        "--viewer-npz",
        str(viewer_npz),
        "--port",
        str(int(args.port)),
        "--status-json",
        str(status_json),
    ]
    if default_variant:
        cmd.extend(["--default-variant", str(default_variant)])
    cmd.extend(["--d4rt-frame-mode", str(args.d4rt_viewer_frame_mode)])
    cmd.extend(["--d4rt-initial-frame", str(int(args.d4rt_initial_frame))])
    log_handle = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
        text=True,
    )
    deadline = time.time() + float(args.viewer_start_timeout)
    while time.time() < deadline:
        if status_json.is_file() and status_json.stat().st_size > 0:
            break
        if proc.poll() is not None:
            log_handle.close()
            raise RuntimeError(f"viewer exited early with returncode={proc.returncode}; see {log_path}")
        time.sleep(0.5)
    log_handle.close()
    if not status_json.is_file():
        raise RuntimeError(f"viewer did not write status within {args.viewer_start_timeout}s; see {log_path}")

    http = {"checked": False}
    url = f"http://127.0.0.1:{int(args.port)}/"
    try:
        with urllib.request.urlopen(url, timeout=float(args.http_timeout)) as response:
            data = response.read()
        http = {"checked": True, "url": url, "http_code": 200, "bytes": int(len(data))}
    except Exception as exc:
        http = {"checked": True, "url": url, "error": repr(exc)}
    return {
        "command": cmd,
        "command_text": _quote(cmd),
        "pid": int(proc.pid),
        "stopped_existing_pids": stopped,
        "status_json": str(status_json),
        "log_path": str(log_path),
        "http_check": http,
    }


def _read_summary_metrics(summary_json: Path) -> dict[str, Any]:
    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    d4rt = payload.get("d4rt_geometry", {})
    d4rt_metrics = d4rt.get("geometry_metrics_against_input_visible_gt", {})
    da3_rows = []
    for row in payload.get("variants", []):
        surface = row.get("surface_refined_sim3", {}).get("geometry_metrics", {})
        da3_rows.append(
            {
                "variant_key": row.get("variant_key"),
                "display_name": row.get("display_name"),
                "surface_refined_chamfer_l2_mean_m": surface.get("chamfer_l2_mean_m"),
                "surface_refined_fscore_0p10m": surface.get("fscore", {}).get("0.10m", {}).get("fscore"),
            }
        )
    return {
        "scene_id": payload.get("scene_id"),
        "layers": {
            "da3": da3_rows,
            "d4rt": {
                "display_name": d4rt.get("display_name"),
                "queries_per_frame": d4rt.get("source", {}).get("sampling_contract", {}).get("queries_per_frame"),
                "meets_requested_density": d4rt.get("source", {}).get("sampling_contract", {}).get("meets_requested_density"),
                "self_stitch_all_pairs_pass": d4rt.get("overlap_self_stitch", {}).get("all_pairs_pass"),
                "self_stitch_weak_alignment_chunk_count": d4rt.get("overlap_self_stitch", {}).get("weak_alignment_chunk_count"),
                "chamfer_l2_mean_m": d4rt_metrics.get("chamfer_l2_mean_m"),
                "fscore_0p10m": d4rt_metrics.get("fscore", {}).get("0.10m", {}).get("fscore"),
                "accuracy_p90_m": d4rt_metrics.get("accuracy_da3_to_gt_m", {}).get("p90"),
                "completeness_p90_m": d4rt_metrics.get("completeness_gt_to_da3_m", {}).get("p90"),
            },
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root)
    has_existing_base = bool(args.base_summary_json or args.base_viewer_npz or args.base_output_root)
    if args.base_output_root:
        base_root = Path(args.base_output_root)
    elif args.base_summary_json:
        base_root = Path(args.base_summary_json).parent
    else:
        base_root = output_root / "da3_base"
    final_root = output_root / "dense_d4rt"
    output_root.mkdir(parents=True, exist_ok=True)
    commands_path = output_root / "pipeline_commands.sh"
    steps: list[dict[str, Any]] = []
    env = os.environ.copy()
    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)

    variant_key = str(args.variant_key or "da3_streaming_full")
    if has_existing_base:
        if not args.base_summary_json or not args.base_viewer_npz:
            raise ValueError("--base-summary-json and --base-viewer-npz are required when reusing an existing DA3 base")
        base_summary = Path(args.base_summary_json)
        base_viewer_npz = Path(args.base_viewer_npz)
        if not base_summary.is_file():
            raise FileNotFoundError(base_summary)
        if not base_viewer_npz.is_file():
            raise FileNotFoundError(base_viewer_npz)
        steps.append(
            {
                "name": "use_existing_da3_base",
                "mode": "existing-da3-base",
                "base_root": str(base_root),
                "base_summary_json": str(base_summary),
                "base_viewer_npz": str(base_viewer_npz),
                "returncode": 0,
                "seconds": 0.0,
            }
        )
        base_cmd: list[str] | None = None
    else:
        if not args.da3_root or not args.da3_log:
            raise ValueError("--da3-root and --da3-log are required when not using an existing DA3 base")
        base_cmd = [
            sys.executable,
            str(TOOLS_DIR / "build_v98_1_da3_single_scene_geometry_quality.py"),
            "--scene-id",
            str(args.scene_id),
            "--da3-root",
            str(args.da3_root),
            "--da3-manifest",
            str(args.da3_manifest),
            "--da3-log",
            str(args.da3_log),
            "--output-root",
            str(base_root),
            "--variant-key",
            variant_key,
            "--display-name",
            str(args.display_name),
            "--model-name",
            str(args.model_name),
            "--repo-id",
            str(args.repo_id),
            "--gt-filter",
            str(args.gt_filter),
            "--da3-dense-step",
            str(int(args.da3_dense_step)),
            "--viewer-da3-sample-count",
            str(int(args.viewer_da3_sample_count)),
            "--viewer-gt-sample-count",
            str(int(args.viewer_gt_sample_count)),
            "--surface-fit-sample-count",
            str(int(args.surface_fit_sample_count)),
            "--surface-fit-iterations",
            str(int(args.surface_fit_iterations)),
            "--surface-fit-keep-ratio",
            str(float(args.surface_fit_keep_ratio)),
        ]
        steps.append(_run_step("build_da3_base", base_cmd, cwd=ROOT, log_path=output_root / "logs" / "01_build_da3_base.log", env=env))
        base_summary = base_root / "geometry_quality_summary.json"
        base_viewer_npz = base_root / f"{args.scene_id}_da3_single_geometry_viewer_points.npz"

    d4rt_cmd = [
        sys.executable,
        str(TOOLS_DIR / "build_v98_1_d4rt_dense_geometry_comparison.py"),
        "--scene-id",
        str(args.scene_id),
        "--base-output-root",
        str(base_root),
        "--base-summary-json",
        str(base_summary),
        "--base-viewer-npz",
        str(base_viewer_npz),
        "--output-root",
        str(final_root),
        "--da3-manifest",
        str(args.da3_manifest),
        "--d4rt-root",
        str(args.d4rt_root),
        "--d4rt-config",
        str(args.d4rt_config),
        "--d4rt-ckpt",
        str(args.d4rt_ckpt),
        "--device",
        str(args.device),
        "--rows",
        str(int(args.rows)),
        "--cols",
        str(int(args.cols)),
        "--chunk-size",
        str(int(args.chunk_size)),
        "--overlap",
        str(int(args.overlap)),
        "--query-chunk-size",
        str(int(args.query_chunk_size)),
        "--aspect-source",
        str(args.aspect_source),
        "--camera-frame-mode",
        str(args.d4rt_camera_frame_mode),
        "--viewer-d4rt-sample-count",
        str(int(args.viewer_d4rt_sample_count)),
        "--max-metric-points",
        str(int(args.max_metric_points)),
        "--gt-filter",
        str(args.gt_filter),
    ]
    steps.append(_run_step("build_dense_d4rt", d4rt_cmd, cwd=ROOT, log_path=output_root / "logs" / "02_build_dense_d4rt.log", env=env))

    final_summary = final_root / "geometry_quality_summary_with_d4rt_dense.json"
    final_viewer_npz = final_root / f"{args.scene_id}_da3_d4rt_dense_geometry_viewer_points.npz"
    viewer_info: dict[str, Any] | None = None
    if bool(args.serve):
        default_variant = str(args.default_variant or variant_key)
        viewer_info = _start_viewer(
            args,
            final_root=final_root,
            summary_json=final_summary,
            viewer_npz=final_viewer_npz,
            default_variant=default_variant,
        )

    commands = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {_quote([str(ROOT)])}",
        _quote(d4rt_cmd),
    ]
    if args.cuda_visible_devices:
        commands.insert(3, f"export CUDA_VISIBLE_DEVICES={_quote([str(args.cuda_visible_devices)])}")
    if base_cmd is not None:
        commands.insert(4 if args.cuda_visible_devices else 3, _quote(base_cmd))
    if viewer_info is not None:
        commands.append(_quote([str(part) for part in viewer_info["command"]]))
    commands_path.write_text("\n".join(commands) + "\n", encoding="utf-8")

    summary = {
        "pipeline": "v98_1_dense_geometry_visualization_pipeline",
        "scene_id": str(args.scene_id),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "output_root": str(output_root),
        "base_root": str(base_root),
        "final_root": str(final_root),
        "steps": steps,
        "viewer": viewer_info,
        "config": {
            "mode": "existing-da3-base" if has_existing_base else "single-da3-base",
            "cuda_visible_devices": str(args.cuda_visible_devices),
            "gt_filter": str(args.gt_filter),
            "d4rt_rows": int(args.rows),
            "d4rt_cols": int(args.cols),
            "d4rt_chunk_size": int(args.chunk_size),
            "d4rt_overlap": int(args.overlap),
            "d4rt_aspect_source": str(args.aspect_source),
            "d4rt_camera_frame_mode": str(args.d4rt_camera_frame_mode),
            "d4rt_viewer_frame_mode": str(args.d4rt_viewer_frame_mode),
            "d4rt_initial_frame": int(args.d4rt_initial_frame),
            "default_variant": str(args.default_variant or variant_key),
        },
        "outputs": {
            "pipeline_summary_json": str(output_root / "pipeline_summary.json"),
            "pipeline_commands_sh": str(commands_path),
            "base_summary_json": str(base_summary),
            "final_summary_json": str(final_summary),
            "final_viewer_npz": str(final_viewer_npz),
            "final_metrics_csv": str(final_root / "geometry_quality_metrics_with_d4rt_dense.csv"),
        },
        "metric_snippet": _read_summary_metrics(final_summary),
    }
    _write_json(output_root / "pipeline_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True, default=_json_default), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end v98.1 DA3 + dense D4RT geometry Viser pipeline.")
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--da3-root", default="")
    parser.add_argument("--da3-manifest", required=True)
    parser.add_argument("--da3-log", default="")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--variant-key", default="da3_streaming_full")
    parser.add_argument("--default-variant", default="")
    parser.add_argument("--display-name", default="DA3-Streaming full")
    parser.add_argument("--model-name", default="DA3-SMALL")
    parser.add_argument("--repo-id", default="depth-anything/DA3-SMALL")
    parser.add_argument("--base-output-root", default="")
    parser.add_argument("--base-summary-json", default="")
    parser.add_argument("--base-viewer-npz", default="")
    parser.add_argument("--gt-filter", choices=["full", "input_visible"], default="input_visible")
    parser.add_argument("--da3-dense-step", type=int, default=8)
    parser.add_argument("--viewer-da3-sample-count", type=int, default=120000)
    parser.add_argument("--viewer-gt-sample-count", type=int, default=180000)
    parser.add_argument("--surface-fit-sample-count", type=int, default=60000)
    parser.add_argument("--surface-fit-iterations", type=int, default=8)
    parser.add_argument("--surface-fit-keep-ratio", type=float, default=0.90)
    parser.add_argument("--d4rt-root", default=str(DEFAULT_D4RT_ROOT))
    parser.add_argument("--d4rt-config", default=str(DEFAULT_D4RT_CONFIG))
    parser.add_argument("--d4rt-ckpt", default=str(DEFAULT_D4RT_CKPT))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cuda-visible-devices", default="")
    parser.add_argument("--rows", type=int, default=120)
    parser.add_argument("--cols", type=int, default=160)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--overlap", type=int, default=3)
    parser.add_argument("--query-chunk-size", type=int, default=4096)
    parser.add_argument("--aspect-source", choices=["model_input", "native_rgb"], default="model_input")
    parser.add_argument("--d4rt-camera-frame-mode", choices=["ref0", "target_local"], default="ref0")
    parser.add_argument("--viewer-d4rt-sample-count", type=int, default=180000)
    parser.add_argument("--max-metric-points", type=int, default=0)
    parser.add_argument("--serve", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stop-existing-viewer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--port", type=int, default=8099)
    parser.add_argument("--d4rt-viewer-frame-mode", choices=["aggregate", "slider", "slider_with_aggregate"], default="aggregate")
    parser.add_argument("--d4rt-initial-frame", type=int, default=0)
    parser.add_argument("--viewer-start-timeout", type=float, default=30.0)
    parser.add_argument("--http-timeout", type=float, default=10.0)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
