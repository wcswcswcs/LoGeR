#!/usr/bin/env python3
"""Build an auditable Stage7 replay command from a saved hmc_config.yaml.

This helper does not claim Stage7 promotion.  Its main purpose is to repair the
v97 C2 full-latent-dump reproducibility gap: translate a saved Stage7 full
rollout config back into the current run_pipeline_abc_v2.py CLI, redirect all
outputs into a v97 diagnostic directory, and enable v68 PCA latent dumps.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE = REPO_ROOT / "run_pipeline_abc_v2.py"
METADATA_ONLY_KEYS = {
    "commit_source",
    "read_path_hooks_status",
    "resolved_hybrid_memory_mode",
    "two_pass",
}
DEFAULT_V68_TAPS = "pca_attn_global_k_layers,pca_attn_global_v_layers,pca_attn_frame_v_layers"
DEFAULT_V68_LAYERS = "4,5,12,13,16,17"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a YAML mapping")
    return payload


def load_pipeline_parser() -> argparse.ArgumentParser:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from run_pipeline_abc_v2 import build_parser  # noqa: PLC0415

    return build_parser()


def parser_destinations(parser: argparse.ArgumentParser) -> dict[str, argparse.Action]:
    actions: dict[str, argparse.Action] = {}
    for action in parser._actions:  # noqa: SLF001
        if action.dest and action.dest != "help":
            actions[action.dest] = action
    return actions


def canonical_option(action: argparse.Action) -> str:
    expected = f"--{action.dest}"
    if expected in action.option_strings:
        return expected
    long_options = [option for option in action.option_strings if option.startswith("--")]
    if long_options:
        underscore_options = [option for option in long_options if "_" in option]
        return underscore_options[0] if underscore_options else long_options[0]
    return action.option_strings[0]


def is_store_true(action: argparse.Action) -> bool:
    return isinstance(action, argparse._StoreTrueAction)  # noqa: SLF001


def is_store_false(action: argparse.Action) -> bool:
    return isinstance(action, argparse._StoreFalseAction)  # noqa: SLF001


def value_to_args(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    if isinstance(value, dict):
        return [json.dumps(value, sort_keys=True, separators=(",", ":"))]
    return [str(value)]


def add_action_args(cmd: list[str], action: argparse.Action, value: Any) -> None:
    if value is None:
        return
    option = canonical_option(action)
    if is_store_true(action):
        if bool(value):
            cmd.append(option)
        return
    if is_store_false(action):
        if not bool(value):
            cmd.append(option)
        return
    cmd.append(option)
    cmd.extend(value_to_args(value))


def sequence_id_from_input(input_path: Any) -> str:
    path = Path(str(input_path or ""))
    parts = path.parts
    if "sequences" in parts:
        idx = parts.index("sequences")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return "sequence"


def parse_overrides(items: list[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--override expects KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        if not key:
            raise ValueError(f"--override has empty KEY in {item!r}")
        overrides[key] = value
    return overrides


def make_replay(
    *,
    config_path: Path,
    out_dir: Path,
    python: Path,
    pipeline: Path,
    device: str,
    pca_taps: str,
    pca_layers: str,
    pca_max_feature_dim: int,
    extra_overrides: dict[str, str],
    strict: bool,
) -> dict[str, Any]:
    parser = load_pipeline_parser()
    actions = parser_destinations(parser)
    saved = read_yaml(config_path)
    seq = sequence_id_from_input(saved.get("input"))
    out_dir.mkdir(parents=True, exist_ok=True)

    effective = dict(saved)
    replay_overrides: dict[str, Any] = {
        "device": device,
        "hybrid_debug_jsonl": str(out_dir / "hmc_state_hash.jsonl"),
        "output_pt": None,
        "output_txt": str(out_dir / f"{seq}.txt"),
        "output_video": "",
        "v68_export_full_pca_debug": 1,
        "v68_layer_pca_feature_dir": str(out_dir / "pca_features"),
        "v68_pca_layers": pca_layers,
        "v68_pca_max_feature_dim": pca_max_feature_dim,
        "v68_pca_taps": pca_taps,
    }
    replay_overrides.update(extra_overrides)
    effective.update(replay_overrides)

    skipped_metadata = sorted(key for key in effective if key in METADATA_ONLY_KEYS)
    unsupported = sorted(key for key in effective if key not in actions and key not in METADATA_ONLY_KEYS)
    if strict and unsupported:
        raise ValueError(f"unsupported config keys for current parser: {unsupported}")

    cmd = [str(python), str(pipeline)]
    emitted_keys: list[str] = []
    omitted_none_keys: list[str] = []
    for key in sorted(effective):
        if key in METADATA_ONLY_KEYS or key not in actions:
            continue
        if effective[key] is None:
            omitted_none_keys.append(key)
            continue
        add_action_args(cmd, actions[key], effective[key])
        emitted_keys.append(key)

    shell_command = " ".join(shlex.quote(part) for part in cmd)
    manifest = {
        "schema": "acl2_v97_stage7_replay_from_hmc_config_v1",
        "diagnostic_only": True,
        "stage7_promotion_claimed": False,
        "source_hmc_config": str(config_path),
        "source_hmc_config_sha256": sha256_file(config_path),
        "pipeline": str(pipeline),
        "pipeline_sha256": sha256_file(pipeline),
        "repo_root": str(REPO_ROOT),
        "out_dir": str(out_dir),
        "sequence_id": seq,
        "python": str(python),
        "device": device,
        "metadata_only_keys_skipped": skipped_metadata,
        "unsupported_config_keys": unsupported,
        "omitted_none_keys": omitted_none_keys,
        "replay_overrides": replay_overrides,
        "emitted_config_keys": emitted_keys,
        "command_arg_count": len(cmd),
        "command_sha256": hashlib.sha256("\0".join(cmd).encode("utf-8")).hexdigest(),
        "command": cmd,
        "shell_command": shell_command,
        "created_unix_time": time.time(),
    }
    return manifest


def write_manifest(out_dir: Path, manifest: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "replay_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    script = "\n".join([
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {shlex.quote(str(REPO_ROOT))}",
        f"exec {manifest['shell_command']}",
        "",
    ])
    run_script = out_dir / "run_replay.sh"
    run_script.write_text(script, encoding="utf-8")
    run_script.chmod(run_script.stat().st_mode | 0o111)


def run_replay(out_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    run_log = out_dir / "run.log"
    start = time.time()
    with run_log.open("w", encoding="utf-8") as log:
        log.write("$ " + manifest["shell_command"] + "\n")
        log.flush()
        proc = subprocess.run(
            manifest["command"],
            cwd=str(REPO_ROOT),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    end = time.time()
    manifest["run"] = {
        "returncode": proc.returncode,
        "start_unix_time": start,
        "end_unix_time": end,
        "elapsed_seconds": end - start,
        "run_log": str(run_log),
        "output_txt_exists": Path(str(manifest["replay_overrides"]["output_txt"])).exists(),
        "hybrid_debug_jsonl_exists": Path(str(manifest["replay_overrides"]["hybrid_debug_jsonl"])).exists(),
        "pca_feature_file_count": len(list(Path(str(manifest["replay_overrides"]["v68_layer_pca_feature_dir"])).glob("*.pt"))),
    }
    write_manifest(out_dir, manifest)
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create or run an auditable v97 diagnostic Stage7 replay command from saved hmc_config.yaml.",
    )
    parser.add_argument("--hmc-config", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--pipeline", type=Path, default=PIPELINE)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--v68-pca-taps", default=DEFAULT_V68_TAPS)
    parser.add_argument(
        "--v68-pca-layers",
        default=DEFAULT_V68_LAYERS,
        help=(
            "Layer ids for PCA dump. The default mixes frame-attention even ids "
            "with global-attention odd ids so the default taps all have support."
        ),
    )
    parser.add_argument("--v68-pca-max-feature-dim", type=int, default=8)
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Extra replay override as KEY=VALUE after the standard diagnostic overrides.",
    )
    parser.add_argument("--strict", action="store_true", help="Fail if saved config has current-parser unsupported keys.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Only write replay_manifest.json and run_replay.sh.")
    mode.add_argument("--run", action="store_true", help="Execute the generated command and update the manifest.")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    config_path = args.hmc_config.resolve()
    out_dir = args.out_dir.resolve()
    pipeline = args.pipeline.resolve()
    python = args.python.resolve()
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    if not pipeline.exists():
        raise FileNotFoundError(pipeline)
    overrides = parse_overrides(args.override)
    manifest = make_replay(
        config_path=config_path,
        out_dir=out_dir,
        python=python,
        pipeline=pipeline,
        device=str(args.device),
        pca_taps=str(args.v68_pca_taps),
        pca_layers=str(args.v68_pca_layers),
        pca_max_feature_dim=int(args.v68_pca_max_feature_dim),
        extra_overrides=overrides,
        strict=bool(args.strict),
    )
    manifest["dry_run"] = not bool(args.run)
    write_manifest(out_dir, manifest)
    if args.run:
        manifest = run_replay(out_dir, manifest)
    print(json.dumps({
        "out_dir": str(out_dir),
        "manifest": str(out_dir / "replay_manifest.json"),
        "run_script": str(out_dir / "run_replay.sh"),
        "dry_run": manifest.get("dry_run", True),
        "unsupported_config_keys": manifest.get("unsupported_config_keys", []),
        "command_arg_count": manifest.get("command_arg_count"),
        "run": manifest.get("run", {}),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
