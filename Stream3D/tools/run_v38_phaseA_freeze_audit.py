from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


EXPECTED_F31 = {
    "ARI": 0.4261893143223615,
    "purity": 0.8684666831805796,
    "completeness": 0.5047064651969284,
    "unknown_tube_ratio": 0.2355709685409958,
    "scene0081_ARI": 0.20186794681675915,
}

EXPECTED_I4 = {
    "4D_ARI": 0.42599481039581194,
    "4D_purity": 0.8673519940549913,
    "4D_completeness": 0.5056972999752292,
    "temporal_span_mean": 1.702673104336451,
}

EXPECTED_AP = {
    "raw_AP": 0.003937456854837711,
    "raw_AP50": 0.012952410140378828,
    "raw_AP25": 0.1194303308283155,
    "best_postprocess_AP": 0.06267933061300239,
    "best_postprocess_AP50": 0.1763713984308037,
    "best_postprocess_AP25": 0.421656132465113,
    "same_support_stream3d_AP": 0.3992127932017927,
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _exit_code(path: Path) -> int | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    try:
        return int(text)
    except ValueError:
        return None


def _parse_unittest_log(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "test_count": None, "ok": False}
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"Ran\s+(\d+)\s+tests?", text)
    return {
        "exists": True,
        "test_count": int(match.group(1)) if match else None,
        "ok": bool(re.search(r"\bOK\b", text)),
    }


def _metric_diff(actual: dict[str, Any], expected: dict[str, float]) -> dict[str, dict[str, float | bool | None]]:
    out: dict[str, dict[str, float | bool | None]] = {}
    for key, exp in expected.items():
        raw = actual.get(key)
        val = float(raw) if raw is not None else None
        diff = abs(val - exp) if val is not None else None
        out[key] = {
            "actual": val,
            "expected": exp,
            "abs_diff": diff,
            "pass_1e_6": bool(diff is not None and diff <= 1e-6),
        }
    return out


def _all_pass(diff_rows: dict[str, dict[str, float | bool | None]]) -> bool:
    return all(bool(row.get("pass_1e_6")) for row in diff_rows.values())


def _manifest_eval_only_check(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    checks = {
        "exists": path.exists(),
        "is_method_result_false": payload.get("is_method_result") is False,
        "is_diagnostic_only_true": payload.get("is_diagnostic_only") is True,
        "forbidden_for_method_table_true": payload.get("forbidden_for_method_table") is True,
        "uses_rgbd_for_prediction_true": payload.get("uses_rgbd_for_prediction") is True,
        "uses_pose_for_prediction_true": payload.get("uses_pose_for_prediction") is True,
        "uses_scannet_mesh_for_prediction_true": payload.get("uses_scannet_mesh_for_prediction") is True,
    }
    return {
        "path": str(path),
        "checks": checks,
        "pass": all(bool(value) for value in checks.values()),
        "key_fields": {
            key: payload.get(key)
            for key in (
                "output_config",
                "is_method_result",
                "is_diagnostic_only",
                "forbidden_for_method_table",
                "base_score_mode",
                "score_feature",
                "tiebreaker_weight",
                "min_area",
                "alignment_source",
            )
        },
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Stream4D v38 Phase A Freeze Audit",
        "",
        f"phaseA_pass={summary['phaseA_pass']}",
        "",
        "## F31",
        "",
    ]
    for key, row in summary["f31_diff"].items():
        lines.append(f"- {key}: actual={row['actual']} expected={row['expected']} abs_diff={row['abs_diff']} pass={row['pass_1e_6']}")
    lines.extend(["", "## I4", ""])
    for key, row in summary["i4_diff"].items():
        lines.append(f"- {key}: actual={row['actual']} expected={row['expected']} abs_diff={row['abs_diff']} pass={row['pass_1e_6']}")
    lines.extend(["", "## AP References", ""])
    for key, row in summary["ap_diff"].items():
        lines.append(f"- {key}: actual={row['actual']} expected={row['expected']} abs_diff={row['abs_diff']} pass={row['pass_1e_6']}")
    lines.extend(
        [
            "",
            "## Regression Tests",
            "",
            f"- py_compile_exit_code={summary['validation']['py_compile_exit_code']}",
            f"- unittest_exit_code={summary['validation']['unittest_exit_code']}",
            f"- unittest_test_count={summary['validation']['unittest']['test_count']}",
            f"- unittest_ok={summary['validation']['unittest']['ok']}",
            "",
            "## Manifest / AP Scoring",
            "",
            f"- manifest_eval_only_pass={summary['manifest_eval_only_pass']}",
            f"- constant_score_control_pass={summary['constant_score_control_pass']}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream3d-root", default=".")
    parser.add_argument("--output-root", default="outputs/audit/v38_phaseA_freeze")
    parser.add_argument("--py-compile-exit-code", default="outputs/audit/v38_phaseA_freeze/py_compile_v38_phaseA.exit_code")
    parser.add_argument("--unittest-exit-code", default="outputs/audit/v38_phaseA_freeze/unittest_v38_phaseA.exit_code")
    parser.add_argument("--unittest-log", default="outputs/audit/v38_phaseA_freeze/unittest_v38_phaseA.log")
    args = parser.parse_args()

    root = Path(args.stream3d_root).resolve()
    out = root / args.output_root
    out.mkdir(parents=True, exist_ok=True)

    f31_path = root / "outputs/audit/v37_adaptive_density_final_probe5/v37_final_decision/decision_summary.json"
    i4_path = root / "outputs/audit/v37_4d_if_allowed_i4_sparse/4d_memory_decision.json"
    raw_ap_path = root / "outputs/audit/v37_ap_if_allowed_i4_sparse/ap_eval_summary.json"
    post_ap_path = root / "outputs/audit/v37_ap_if_allowed_i4_sparse/ap_postprocess_final_summary.json"

    f31 = _read_json(f31_path)
    i4 = _read_json(i4_path)
    raw_ap = _read_json(raw_ap_path)
    post_ap = _read_json(post_ap_path)

    f31_metrics = f31["best_metrics"]
    i4_metrics = i4["best_metrics"]
    ap_actual = {
        "raw_AP": raw_ap.get("AP"),
        "raw_AP50": raw_ap.get("AP50"),
        "raw_AP25": raw_ap.get("AP25"),
        "best_postprocess_AP": post_ap.get("best_postprocess_metrics", {}).get("AP"),
        "best_postprocess_AP50": post_ap.get("best_postprocess_metrics", {}).get("AP50"),
        "best_postprocess_AP25": post_ap.get("best_postprocess_metrics", {}).get("AP25"),
        "same_support_stream3d_AP": post_ap.get("same_support_stream3d_metrics", {}).get("AP"),
    }

    manifest_paths = [
        root / "data/prediction/v37_i4_sparse_ap_eval_probe5_class_agnostic/config_manifest.json",
        root / "data/prediction/v37_i4_sparse_ap_eval_probe5_rescore_const_none_min100_class_agnostic/config_manifest.json",
    ]
    manifest_checks = [_manifest_eval_only_check(path) for path in manifest_paths]
    best_manifest = manifest_checks[1]["key_fields"]
    constant_score_control_pass = bool(
        best_manifest.get("base_score_mode") == "constant"
        and best_manifest.get("score_feature") == "none"
        and float(best_manifest.get("tiebreaker_weight") or 0.0) == 0.0
        and int(best_manifest.get("min_area") or 0) == 100
        and post_ap.get("preferred_audit_best_postprocess_variant") == "rescore_const_none_min100"
    )

    artifact_paths = [
        f31_path,
        i4_path,
        raw_ap_path,
        post_ap_path,
        *manifest_paths,
        root / "../docs/stream4d_v38_object_materialization_plan.md",
        root / "../docs/stream4d_v37_temporal_curriculum_masklet_report.md",
        root / "../docs/stream4d_v37_实验结果复盘.md",
    ]
    hashes = {str(path.resolve().relative_to(root.parent.resolve())): _sha256(path) for path in artifact_paths if path.exists()}
    hash_text = "\n".join(f"{digest}  {path}" for path, digest in sorted(hashes.items())) + "\n"
    (out / "phaseA_artifact_hashes.sha256").write_text(hash_text, encoding="utf-8")

    f31_diff = _metric_diff(f31_metrics, EXPECTED_F31)
    i4_diff = _metric_diff(i4_metrics, EXPECTED_I4)
    ap_diff = _metric_diff(ap_actual, EXPECTED_AP)
    validation = {
        "py_compile_exit_code": _exit_code(root / args.py_compile_exit_code),
        "unittest_exit_code": _exit_code(root / args.unittest_exit_code),
        "unittest": _parse_unittest_log(root / args.unittest_log),
    }
    regression_tests_pass = bool(
        validation["py_compile_exit_code"] == 0
        and validation["unittest_exit_code"] == 0
        and validation["unittest"]["ok"]
        and (validation["unittest"]["test_count"] or 0) >= 153
    )

    summary = {
        "phase": "v38_phaseA_freeze",
        "phaseA_pass": bool(
            _all_pass(f31_diff)
            and _all_pass(i4_diff)
            and _all_pass(ap_diff)
            and bool(f31.get("pass_3D_gate"))
            and bool(f31.get("pass_controls"))
            and bool(i4_metrics.get("pass_4D_gate"))
            and regression_tests_pass
            and all(bool(item["pass"]) for item in manifest_checks)
            and constant_score_control_pass
        ),
        "f31_stage": f31.get("best_stage"),
        "i4_variant": i4.get("best_variant"),
        "f31_diff": f31_diff,
        "i4_diff": i4_diff,
        "ap_diff": ap_diff,
        "f31_status": f31.get("final_status"),
        "i4_status": i4.get("final_status"),
        "ap_status": raw_ap.get("final_status"),
        "postprocess_status": post_ap.get("final_status"),
        "validation": validation,
        "regression_tests_pass": regression_tests_pass,
        "manifest_checks": manifest_checks,
        "manifest_eval_only_pass": all(bool(item["pass"]) for item in manifest_checks),
        "constant_score_control_pass": constant_score_control_pass,
        "artifact_hashes_file": str((out / "phaseA_artifact_hashes.sha256").relative_to(root)),
        "notes": [
            "This audit freezes v37 F31/I4 by reading exact landed artifacts and hashes.",
            "It does not claim new v38 AP repair success.",
        ],
    }
    _write_json(out / "phaseA_freeze_summary.json", summary)
    _write_json(out / "phaseA_manifest_audit.json", manifest_checks)
    _write_markdown(out / "phaseA_freeze_summary.md", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["phaseA_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
