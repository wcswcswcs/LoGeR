from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REAL_ORACLE_ROOT = "v28_proposal_oracle_repair_temporal_track_640_pruned_shared2_t20_guard5_probe5_r2"
SHUFFLE_ORACLE_ROOT = "v28_proposal_oracle_shuffle_d4rt_guard5_probe5"
STRICT_SELECTION_ROOT = "v28_proposal_selection_guard5_strict_score02_r3_with_p8_proxy"
P11_SELECTION_ROOT = "v28_proposal_selection_guard5_p11_ownership_expansion_r1"
V29_CONTINUATION_ROOT = "v29_solver_continuation"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _count_csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            next(reader)
        except StopIteration:
            return 0
        return sum(1 for _ in reader)


def _float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key)
    if value is None or value == "":
        return None
    return float(value)


def _find_row(
    rows: list[dict[str, str]],
    *,
    scene: str = "ALL",
    pool: str | None = None,
    variant: str | None = None,
    solver: str | None = None,
    control_kind: str | None = None,
) -> dict[str, str]:
    for row in rows:
        if str(row.get("scene")) != scene:
            continue
        if pool is not None and str(row.get("pool")) != pool:
            continue
        if variant is not None and str(row.get("variant")) != variant:
            continue
        if solver is not None and str(row.get("solver")) != solver:
            continue
        if control_kind is not None and str(row.get("control_kind")) != control_kind:
            continue
        return row
    parts = [
        f"scene={scene}",
        f"pool={pool}" if pool is not None else "",
        f"variant={variant}" if variant is not None else "",
        f"solver={solver}" if solver is not None else "",
        f"control_kind={control_kind}" if control_kind is not None else "",
    ]
    raise ValueError("missing row: " + " ".join(p for p in parts if p))


def _best_continuation(rows: list[dict[str, str]], control_kind: str) -> dict[str, str]:
    candidates = [row for row in rows if row.get("scene") == "ALL" and row.get("control_kind") == control_kind]
    if not candidates:
        raise ValueError(f"missing continuation rows for {control_kind}")
    return max(candidates, key=lambda row: float(row["ARI"]))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
    real_dir = audit_root / REAL_ORACLE_ROOT
    shuffle_dir = audit_root / SHUFFLE_ORACLE_ROOT
    strict_dir = audit_root / STRICT_SELECTION_ROOT
    p11_dir = audit_root / P11_SELECTION_ROOT
    continuation_dir = audit_root / V29_CONTINUATION_ROOT
    for path in (real_dir, shuffle_dir, strict_dir, p11_dir, continuation_dir):
        if not path.exists():
            raise FileNotFoundError(path)

    real_rows_path = real_dir / f"{REAL_ORACLE_ROOT}_proposal_rows.csv"
    real_summary_path = real_dir / f"{REAL_ORACLE_ROOT}_oracle_summary.csv"
    shuffle_summary_path = shuffle_dir / f"{SHUFFLE_ORACLE_ROOT}_oracle_summary.csv"
    strict_summary_path = strict_dir / f"{STRICT_SELECTION_ROOT}_selection_summary.csv"
    p11_summary_path = p11_dir / f"{P11_SELECTION_ROOT}_selection_summary.csv"
    continuation_summary_path = continuation_dir / "continuation_solver_summary.csv"

    real_o5 = _find_row(_read_csv(real_summary_path), pool="O5_hybrid")
    shuffle_o5 = _find_row(_read_csv(shuffle_summary_path), pool="O5_hybrid")
    strict_p4 = _find_row(_read_csv(strict_summary_path), variant="P4_greedy_set_packing")
    p11 = _find_row(_read_csv(p11_summary_path), variant="P11_calibrated_ownership_expansion")
    continuation_rows = _read_csv(continuation_summary_path)
    best_real_continuation = _best_continuation(continuation_rows, "real")
    best_shuffled_continuation = _best_continuation(continuation_rows, "shuffled_d4rt")
    best_no_temporal = _best_continuation(continuation_rows, "no_temporal")
    best_mask_only = _best_continuation(continuation_rows, "mask_only")

    metrics: dict[str, Any] = {
        "proposal_pool_hash": _sha256(real_rows_path),
        "proposal_row_count": _count_csv_rows(real_rows_path),
        "real_O5_ARI": _float(real_o5, "oracle_ARI"),
        "real_O5_purity": _float(real_o5, "oracle_purity"),
        "real_O5_completeness": _float(real_o5, "oracle_completeness"),
        "real_O5_scene0081_ARI": _float(real_o5, "scene0081_oracle_ARI"),
        "shuffled_O5_ARI": _float(shuffle_o5, "oracle_ARI"),
        "shuffled_O5_purity": _float(shuffle_o5, "oracle_purity"),
        "shuffled_O5_completeness": _float(shuffle_o5, "oracle_completeness"),
        "shuffled_O5_scene0081_ARI": _float(shuffle_o5, "scene0081_oracle_ARI"),
        "strict_P4_ARI": _float(strict_p4, "local_ARI"),
        "strict_P4_purity": _float(strict_p4, "local_purity"),
        "strict_P4_completeness": _float(strict_p4, "local_completeness"),
        "strict_P4_scene0081_ARI": _float(strict_p4, "scene0081_local_ARI"),
        "P11_ARI": _float(p11, "local_ARI"),
        "P11_purity": _float(p11, "local_purity"),
        "P11_completeness": _float(p11, "local_completeness"),
        "P11_scene0081_ARI": _float(p11, "scene0081_local_ARI"),
        "best_continuation_solver": best_real_continuation["solver"],
        "best_continuation_ARI": _float(best_real_continuation, "ARI"),
        "best_continuation_purity": _float(best_real_continuation, "purity"),
        "best_continuation_completeness": _float(best_real_continuation, "completeness"),
        "best_continuation_scene0081_ARI": _float(best_real_continuation, "scene0081_ARI"),
        "shuffled_best_solver": best_shuffled_continuation["solver"],
        "shuffled_best_ARI": _float(best_shuffled_continuation, "ARI"),
        "no_temporal_best_solver": best_no_temporal["solver"],
        "no_temporal_best_ARI": _float(best_no_temporal, "ARI"),
        "mask_only_best_solver": best_mask_only["solver"],
        "mask_only_best_ARI": _float(best_mask_only, "ARI"),
    }

    table_rows = [
        {
            "item": "real_O5",
            "ARI": metrics["real_O5_ARI"],
            "purity": metrics["real_O5_purity"],
            "completeness": metrics["real_O5_completeness"],
            "scene0081_ARI": metrics["real_O5_scene0081_ARI"],
            "source": str(real_summary_path),
        },
        {
            "item": "strict_P4",
            "ARI": metrics["strict_P4_ARI"],
            "purity": metrics["strict_P4_purity"],
            "completeness": metrics["strict_P4_completeness"],
            "scene0081_ARI": metrics["strict_P4_scene0081_ARI"],
            "source": str(strict_summary_path),
        },
        {
            "item": "P11",
            "ARI": metrics["P11_ARI"],
            "purity": metrics["P11_purity"],
            "completeness": metrics["P11_completeness"],
            "scene0081_ARI": metrics["P11_scene0081_ARI"],
            "source": str(p11_summary_path),
        },
        {
            "item": "best_continuation",
            "ARI": metrics["best_continuation_ARI"],
            "purity": metrics["best_continuation_purity"],
            "completeness": metrics["best_continuation_completeness"],
            "scene0081_ARI": metrics["best_continuation_scene0081_ARI"],
            "source": str(continuation_summary_path),
            "solver": metrics["best_continuation_solver"],
        },
        {
            "item": "shuffled_best",
            "ARI": metrics["shuffled_best_ARI"],
            "source": str(continuation_summary_path),
            "solver": metrics["shuffled_best_solver"],
        },
        {
            "item": "no_temporal_best",
            "ARI": metrics["no_temporal_best_ARI"],
            "source": str(continuation_summary_path),
            "solver": metrics["no_temporal_best_solver"],
        },
        {
            "item": "mask_only_best",
            "ARI": metrics["mask_only_best_ARI"],
            "source": str(continuation_summary_path),
            "solver": metrics["mask_only_best_solver"],
        },
    ]
    _write_csv(out_dir / "phaseA_required_metrics.csv", table_rows)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "phase": "v30_phaseA_lock",
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
        "geometry_field": "D4RT canonical proposal tube memberships and inherited v28/v29 proposal diagnostics",
        "coordinate_frame": "d4rt_canonical tubes plus image-space mask/proposal features",
        "alignment_source": "D4RT self-Sim3 inherited from v28/v29 artifacts",
        "audit_root": str(audit_root),
        "input_roots": {
            "real_oracle": str(real_dir),
            "shuffled_oracle": str(shuffle_dir),
            "strict_selection": str(strict_dir),
            "p11_selection": str(p11_dir),
            "v29_continuation": str(continuation_dir),
        },
        "input_hashes": {
            "real_proposal_rows_csv": _sha256(real_rows_path),
            "real_oracle_summary_csv": _sha256(real_summary_path),
            "shuffle_oracle_summary_csv": _sha256(shuffle_summary_path),
            "strict_selection_summary_csv": _sha256(strict_summary_path),
            "p11_selection_summary_csv": _sha256(p11_summary_path),
            "v29_continuation_summary_csv": _sha256(continuation_summary_path),
        },
        "metrics": metrics,
        "phaseA_metrics_loaded": all(value is not None for key, value in metrics.items() if key.endswith(("ARI", "purity", "completeness"))),
        "clean_py_compile_pass": None,
        "clean_unittest_pass": None,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "phaseA_lock_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v30 Phase A locked baseline manifest.")
    parser.add_argument("--audit-root", type=Path, default=Path("outputs/audit"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/audit/v30_phaseA_lock"))
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
        manifest["phaseA_gate_pass"] = bool(manifest["phaseA_metrics_loaded"] and args.clean_py_compile_pass and args.clean_unittest_pass)
        (args.out_dir / "phaseA_lock_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps({"out_dir": str(args.out_dir), "metrics": manifest["metrics"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
