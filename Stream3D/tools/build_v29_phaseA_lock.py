from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REAL_ROOT = "v28_proposal_oracle_repair_temporal_track_640_pruned_shared2_t20_guard5_probe5_r2"
SHUFFLE_ROOT = "v28_proposal_oracle_shuffle_d4rt_guard5_probe5"
STRICT_ROOT = "v28_proposal_selection_guard5_strict_score02_r3_with_p8_proxy"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _count_csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            next(reader)
        except StopIteration:
            return 0
        return sum(1 for _ in reader)


def _read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _as_float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key)
    if value is None or value == "":
        return None
    return float(value)


def _find_row(rows: list[dict[str, str]], *, pool: str | None = None, variant: str | None = None, scene: str = "ALL") -> dict[str, str]:
    for row in rows:
        if str(row.get("scene")) != scene:
            continue
        if pool is not None and str(row.get("pool")) != pool:
            continue
        if variant is not None and str(row.get("variant")) != variant:
            continue
        return row
    label = pool if pool is not None else variant
    raise ValueError(f"missing summary row for {label} scene={scene}")


def _gate(actual: float | None, op: str, expected: float) -> bool:
    if actual is None:
        return False
    if op == ">=":
        return float(actual) >= float(expected)
    if op == "<=":
        return float(actual) <= float(expected)
    raise ValueError(op)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def build_phase_a_lock(audit_root: Path, out_dir: Path) -> dict[str, Any]:
    real_dir = audit_root / REAL_ROOT
    shuffle_dir = audit_root / SHUFFLE_ROOT
    strict_dir = audit_root / STRICT_ROOT
    for path in (real_dir, shuffle_dir, strict_dir):
        if not path.exists():
            raise FileNotFoundError(path)

    real_prefix = real_dir / REAL_ROOT
    shuffle_prefix = shuffle_dir / SHUFFLE_ROOT
    strict_prefix = strict_dir / STRICT_ROOT

    real_rows_path = real_prefix.with_name(f"{REAL_ROOT}_proposal_rows.csv")
    real_summary_path = real_prefix.with_name(f"{REAL_ROOT}_oracle_summary.csv")
    shuffle_summary_path = shuffle_prefix.with_name(f"{SHUFFLE_ROOT}_oracle_summary.csv")
    strict_summary_path = strict_prefix.with_name(f"{STRICT_ROOT}_selection_summary.csv")

    real_summary = _read_csv_dicts(real_summary_path)
    shuffle_summary = _read_csv_dicts(shuffle_summary_path)
    strict_summary = _read_csv_dicts(strict_summary_path)

    real_o5 = _find_row(real_summary, pool="O5_hybrid")
    shuffle_o5 = _find_row(shuffle_summary, pool="O5_hybrid")
    strict_p4 = _find_row(strict_summary, variant="P4_greedy_set_packing")

    metrics: dict[str, Any] = {
        "proposal_pool_hash": _sha256(real_rows_path),
        "proposal_row_count": _count_csv_rows(real_rows_path),
        "real_O5_ARI": _as_float(real_o5, "oracle_ARI"),
        "real_O5_purity": _as_float(real_o5, "oracle_purity"),
        "real_O5_completeness": _as_float(real_o5, "oracle_completeness"),
        "real_scene0081_ARI": _as_float(real_o5, "scene0081_oracle_ARI"),
        "shuffled_O5_ARI": _as_float(shuffle_o5, "oracle_ARI"),
        "shuffled_O5_purity": _as_float(shuffle_o5, "oracle_purity"),
        "shuffled_O5_completeness": _as_float(shuffle_o5, "oracle_completeness"),
        "shuffled_scene0081_ARI": _as_float(shuffle_o5, "scene0081_oracle_ARI"),
        "best_strict_P4_ARI": _as_float(strict_p4, "local_ARI"),
        "best_strict_P4_purity": _as_float(strict_p4, "local_purity"),
        "best_strict_P4_completeness": _as_float(strict_p4, "local_completeness"),
        "best_strict_P4_scene0081_ARI": _as_float(strict_p4, "scene0081_local_ARI"),
    }
    gates = {
        "real_O5_ARI_ge_0_45": _gate(metrics["real_O5_ARI"], ">=", 0.45),
        "real_O5_purity_ge_0_85": _gate(metrics["real_O5_purity"], ">=", 0.85),
        "real_O5_completeness_ge_0_55": _gate(metrics["real_O5_completeness"], ">=", 0.55),
        "shuffled_O5_ARI_le_0_05": _gate(metrics["shuffled_O5_ARI"], "<=", 0.05),
        "strict_P4_ARI_baseline_lock_present": metrics["best_strict_P4_ARI"] is not None,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    table_rows = [
        {
            "item": "real_O5",
            "ARI": metrics["real_O5_ARI"],
            "purity": metrics["real_O5_purity"],
            "completeness": metrics["real_O5_completeness"],
            "scene0081": metrics["real_scene0081_ARI"],
            "note": "v28 final O5 proposal oracle",
        },
        {
            "item": "shuffled_O5",
            "ARI": metrics["shuffled_O5_ARI"],
            "purity": metrics["shuffled_O5_purity"],
            "completeness": metrics["shuffled_O5_completeness"],
            "scene0081": metrics["shuffled_scene0081_ARI"],
            "note": "v28 full shuffled-D4RT O5 proposal oracle control",
        },
        {
            "item": "strict_P4",
            "ARI": metrics["best_strict_P4_ARI"],
            "purity": metrics["best_strict_P4_purity"],
            "completeness": metrics["best_strict_P4_completeness"],
            "scene0081": metrics["best_strict_P4_scene0081_ARI"],
            "note": "v28 strict greedy set-packing selection baseline",
        },
    ]
    _write_csv(out_dir / "phaseA_lock_table.csv", table_rows)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "phase": "v29_phaseA_lock",
        "is_method_result": False,
        "is_diagnostic_only": True,
        "forbidden_for_method_table": True,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
        "uses_rgbd_for_prediction": False,
        "uses_pose_for_prediction": False,
        "uses_scannet_mesh_for_prediction": False,
        "uses_eval_sim3_for_prediction": False,
        "uses_d4rt_self_sim3": True,
        "geometry_field": "D4RT canonical proposal tube memberships plus image-space mask/proposal features inherited from v28 artifacts",
        "coordinate_frame": "d4rt_canonical for tube geometry; image space for mask/proposal features",
        "alignment_source": "D4RT self-Sim3 inherited from v28 final artifacts",
        "audit_root": str(audit_root),
        "input_roots": {
            "real_oracle": str(real_dir),
            "shuffled_oracle": str(shuffle_dir),
            "strict_selection": str(strict_dir),
        },
        "input_hashes": {
            "real_proposal_rows_csv": _sha256(real_rows_path),
            "real_oracle_summary_csv": _sha256(real_summary_path),
            "shuffle_oracle_summary_csv": _sha256(shuffle_summary_path),
            "strict_selection_summary_csv": _sha256(strict_summary_path),
        },
        "metrics": metrics,
        "gates": gates,
        "phaseA_artifact_lock_pass": all(gates.values()),
        "clean_py_compile_pass": None,
        "clean_unittest_pass": None,
    }
    (out_dir / "phaseA_lock_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", type=Path, default=Path("outputs/audit"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/audit/v29_phaseA_lock"))
    parser.add_argument("--clean-py-compile-pass", action="store_true")
    parser.add_argument("--clean-unittest-pass", action="store_true")
    parser.add_argument("--unittest-test-count", type=int, default=None)
    parser.add_argument("--unittest-skipped-count", type=int, default=None)
    args = parser.parse_args()
    manifest = build_phase_a_lock(args.audit_root, args.out_dir)
    if args.clean_py_compile_pass or args.clean_unittest_pass or args.unittest_test_count is not None:
        manifest["clean_py_compile_pass"] = bool(args.clean_py_compile_pass)
        manifest["clean_unittest_pass"] = bool(args.clean_unittest_pass)
        manifest["unittest_test_count"] = args.unittest_test_count
        manifest["unittest_skipped_count"] = args.unittest_skipped_count
        manifest["validation_logs"] = {
            "py_compile": str(args.out_dir / "py_compile.log"),
            "unittest": str(args.out_dir / "unittest.log"),
        }
        (args.out_dir / "phaseA_lock_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    print(json.dumps({"out_dir": str(args.out_dir), "metrics": manifest["metrics"], "gates": manifest["gates"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
