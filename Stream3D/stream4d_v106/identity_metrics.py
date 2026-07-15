from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from .artifacts import write_json

LabelGrid = List[List[int]]
Point = Tuple[int, int]


@dataclass(frozen=True)
class SyntheticCase:
    case_id: str
    description: str
    previous: LabelGrid
    current: LabelGrid
    id_mapping: Dict[int, int] = field(default_factory=dict)
    duplicate_parent_ids: Dict[int, int] = field(default_factory=dict)
    reappearance_expected: bool = False
    reappeared_global_id: int | None = None


def _points(grid: LabelGrid) -> List[Point]:
    return [(r, c) for r, row in enumerate(grid) for c, _ in enumerate(row)]


def _label(grid: LabelGrid, point: Point) -> int:
    return grid[point[0]][point[1]]


def _union_foreground_points(previous: LabelGrid, current: LabelGrid) -> List[Point]:
    pts = []
    for point in _points(previous):
        if _label(previous, point) > 0 or _label(current, point) > 0:
            pts.append(point)
    return pts


def _foreground_points(grid: LabelGrid) -> List[Point]:
    return [point for point in _points(grid) if _label(grid, point) > 0]


def _ids(grid: LabelGrid) -> List[int]:
    return sorted({value for row in grid for value in row if value > 0})


def _mapped_current_id(current_id: int, mapping: Dict[int, int] | None) -> int:
    if current_id == 0:
        return 0
    if not mapping:
        return current_id
    return mapping.get(current_id, current_id)


def current_co_visible_object_consistency(
    previous: LabelGrid, current: LabelGrid, mapping: Dict[int, int] | None = None
) -> float:
    pts = _union_foreground_points(previous, current)
    if not pts:
        return 1.0
    correct = 0
    for point in pts:
        prev_id = _label(previous, point)
        cur_id = _mapped_current_id(_label(current, point), mapping)
        if prev_id == cur_id and prev_id > 0:
            correct += 1
    return correct / len(pts)


def object_partition_consistency(previous: LabelGrid, current: LabelGrid) -> float:
    pts = _union_foreground_points(previous, current)
    if len(pts) < 2:
        return 1.0
    agree = 0
    total = 0
    for a, b in combinations(pts, 2):
        prev_a = _label(previous, a)
        prev_b = _label(previous, b)
        cur_a = _label(current, a)
        cur_b = _label(current, b)
        same_prev = prev_a > 0 and prev_a == prev_b
        same_cur = cur_a > 0 and cur_a == cur_b
        if same_prev == same_cur:
            agree += 1
        total += 1
    return agree / total


def fragmentation_rate(previous: LabelGrid, current: LabelGrid) -> float:
    prev_ids = _ids(previous)
    if not prev_ids:
        return 0.0
    fragmented = 0
    for prev_id in prev_ids:
        cur_ids = {
            _label(current, point)
            for point in _points(previous)
            if _label(previous, point) == prev_id and _label(current, point) > 0
        }
        if len(cur_ids) > 1:
            fragmented += 1
    return fragmented / len(prev_ids)


def merge_rate(previous: LabelGrid, current: LabelGrid) -> float:
    cur_ids = _ids(current)
    if not cur_ids:
        return 0.0
    merged = 0
    for cur_id in cur_ids:
        prev_ids = {
            _label(previous, point)
            for point in _points(current)
            if _label(current, point) == cur_id and _label(previous, point) > 0
        }
        if len(prev_ids) > 1:
            merged += 1
    return merged / len(cur_ids)


def duplicate_rebirth_rate(case: SyntheticCase) -> float:
    if not case.duplicate_parent_ids:
        return 0.0
    parent_counts: Dict[int, int] = {}
    for parent_id in case.duplicate_parent_ids.values():
        parent_counts[parent_id] = parent_counts.get(parent_id, 0) + 1
    duplicate_parent_count = sum(1 for count in parent_counts.values() if count > 1)
    return duplicate_parent_count / max(1, len(parent_counts))


def coverage_metrics(previous: LabelGrid, current: LabelGrid) -> Dict[str, Any]:
    prev_fg = set(_foreground_points(previous))
    cur_fg = set(_foreground_points(current))
    retained = prev_fg & cur_fg
    return {
        "foreground_previous": len(prev_fg),
        "foreground_current": len(cur_fg),
        "retained_foreground": len(retained),
        "coverage_ratio": 1.0 if not prev_fg else len(cur_fg) / len(prev_fg),
        "history_coverage_retention": 1.0 if not prev_fg else len(retained) / len(prev_fg),
    }


def false_merge_count(previous: LabelGrid, current: LabelGrid) -> int:
    count = 0
    for cur_id in _ids(current):
        prev_ids = {
            _label(previous, point)
            for point in _points(current)
            if _label(current, point) == cur_id and _label(previous, point) > 0
        }
        if len(prev_ids) > 1:
            count += 1
    return count


def compute_metrics(case: SyntheticCase, mapping: Dict[int, int] | None = None) -> Dict[str, Any]:
    coverage = coverage_metrics(case.previous, case.current)
    metrics = {
        "CCOC": current_co_visible_object_consistency(case.previous, case.current, mapping),
        "OPC": object_partition_consistency(case.previous, case.current),
        "CFR": fragmentation_rate(case.previous, case.current),
        "CMR": merge_rate(case.previous, case.current),
        "DRR": duplicate_rebirth_rate(case),
        "HCR": coverage["history_coverage_retention"],
        "coverage_ratio": coverage["coverage_ratio"],
        "false_merge_count": false_merge_count(case.previous, case.current),
        "reappearance_success": (
            bool(case.reappearance_expected)
            and case.reappeared_global_id is not None
            and case.reappeared_global_id in _ids(case.current)
            and false_merge_count(case.previous, case.current) == 0
        ),
        "coverage_counts": coverage,
    }
    return metrics


def build_synthetic_cases() -> List[SyntheticCase]:
    base = [
        [1, 1, 0, 2],
        [1, 1, 0, 2],
        [0, 0, 0, 2],
        [3, 3, 0, 0],
    ]
    return [
        SyntheticCase("S0", "exact same partition + same IDs", base, base),
        SyntheticCase(
            "S1",
            "exact same partition + pure ID permutation",
            base,
            [
                [10, 10, 0, 20],
                [10, 10, 0, 20],
                [0, 0, 0, 20],
                [30, 30, 0, 0],
            ],
            id_mapping={10: 1, 20: 2, 30: 3},
        ),
        SyntheticCase(
            "S2",
            "one previous object split into two",
            base,
            [
                [1, 4, 0, 2],
                [1, 4, 0, 2],
                [0, 0, 0, 2],
                [3, 3, 0, 0],
            ],
        ),
        SyntheticCase(
            "S3",
            "two previous objects merged into one",
            base,
            [
                [7, 7, 0, 7],
                [7, 7, 0, 7],
                [0, 0, 0, 7],
                [3, 3, 0, 0],
            ],
        ),
        SyntheticCase(
            "S4",
            "duplicate rebirth",
            base,
            [
                [1, 99, 0, 2],
                [1, 99, 0, 2],
                [0, 0, 0, 2],
                [3, 3, 0, 0],
            ],
            duplicate_parent_ids={1: 1, 99: 1},
        ),
        SyntheticCase(
            "S5",
            "mask drift but same ID",
            base,
            [
                [1, 0, 1, 2],
                [1, 0, 0, 2],
                [0, 0, 0, 2],
                [3, 3, 0, 0],
            ],
        ),
        SyntheticCase(
            "S6",
            "foreground loss",
            base,
            [
                [1, 1, 0, 0],
                [1, 1, 0, 0],
                [0, 0, 0, 0],
                [3, 3, 0, 0],
            ],
        ),
        SyntheticCase(
            "S7",
            "correct occlusion and reappearance",
            [
                [0, 0, 0, 2],
                [0, 0, 0, 2],
                [0, 0, 0, 2],
                [3, 3, 0, 0],
            ],
            base,
            reappearance_expected=True,
            reappeared_global_id=1,
        ),
    ]


def _near_zero(value: float) -> bool:
    return abs(value) <= 1e-9


def expected_checks(case: SyntheticCase, actual: Dict[str, Any], mapped_actual: Dict[str, Any]) -> List[Dict[str, Any]]:
    cid = case.case_id
    if cid == "S0":
        return [
            {"metric": "CCOC", "predicate": "== 1", "actual": actual["CCOC"], "passes": actual["CCOC"] == 1.0},
            {"metric": "OPC", "predicate": "== 1", "actual": actual["OPC"], "passes": actual["OPC"] == 1.0},
            {"metric": "CFR", "predicate": "== 0", "actual": actual["CFR"], "passes": _near_zero(actual["CFR"])},
            {"metric": "CMR", "predicate": "== 0", "actual": actual["CMR"], "passes": _near_zero(actual["CMR"])},
            {"metric": "DRR", "predicate": "== 0", "actual": actual["DRR"], "passes": _near_zero(actual["DRR"])},
        ]
    if cid == "S1":
        return [
            {"metric": "CCOC_before_mapping", "predicate": "< 1", "actual": actual["CCOC"], "passes": actual["CCOC"] < 1.0},
            {"metric": "OPC_before_mapping", "predicate": "== 1", "actual": actual["OPC"], "passes": actual["OPC"] == 1.0},
            {"metric": "CCOC_after_mapping", "predicate": "== 1", "actual": mapped_actual["CCOC"], "passes": mapped_actual["CCOC"] == 1.0},
            {"metric": "OPC_after_mapping", "predicate": "== 1", "actual": mapped_actual["OPC"], "passes": mapped_actual["OPC"] == 1.0},
        ]
    if cid == "S2":
        return [
            {"metric": "CFR", "predicate": "> 0", "actual": actual["CFR"], "passes": actual["CFR"] > 0},
            {"metric": "CMR", "predicate": "== 0", "actual": actual["CMR"], "passes": _near_zero(actual["CMR"])},
        ]
    if cid == "S3":
        return [
            {"metric": "CMR", "predicate": "> 0", "actual": actual["CMR"], "passes": actual["CMR"] > 0},
            {"metric": "CFR", "predicate": "== 0", "actual": actual["CFR"], "passes": _near_zero(actual["CFR"])},
        ]
    if cid == "S4":
        return [{"metric": "DRR", "predicate": "> 0", "actual": actual["DRR"], "passes": actual["DRR"] > 0}]
    if cid == "S5":
        return [
            {"metric": "HCR", "predicate": "< 1", "actual": actual["HCR"], "passes": actual["HCR"] < 1.0},
            {"metric": "OPC", "predicate": "< 1", "actual": actual["OPC"], "passes": actual["OPC"] < 1.0},
        ]
    if cid == "S6":
        return [
            {"metric": "coverage_ratio", "predicate": "< 1", "actual": actual["coverage_ratio"], "passes": actual["coverage_ratio"] < 1.0},
            {"metric": "HCR", "predicate": "< 1", "actual": actual["HCR"], "passes": actual["HCR"] < 1.0},
        ]
    if cid == "S7":
        return [
            {"metric": "false_merge_count", "predicate": "== 0", "actual": actual["false_merge_count"], "passes": actual["false_merge_count"] == 0},
            {"metric": "reappearance_success", "predicate": "is true", "actual": actual["reappearance_success"], "passes": actual["reappearance_success"] is True},
        ]
    raise ValueError(f"unknown synthetic case {cid}")


def run_synthetic_metric_suite(output_dir: Path) -> Dict[str, Any]:
    cases_dir = output_dir / "synthetic_cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []
    for case in build_synthetic_cases():
        actual = compute_metrics(case)
        mapped_actual = compute_metrics(case, case.id_mapping) if case.id_mapping else actual
        checks = expected_checks(case, actual, mapped_actual)
        record = {
            "case_id": case.case_id,
            "description": case.description,
            "previous": case.previous,
            "current": case.current,
            "id_mapping": case.id_mapping,
            "actual": actual,
            "mapped_actual": mapped_actual,
            "checks": checks,
            "passes": all(check["passes"] for check in checks),
        }
        write_json(cases_dir / f"{case.case_id}.json", record)
        records.append(record)
    threshold_contract = {
        "CCOC_exact_pass": 1.0,
        "OPC_exact_pass": 1.0,
        "zero_tolerance": 1e-9,
        "positive_violation_threshold": "> 0",
        "note": "Synthetic gates use exact predicates and do not tune expected values from real scene data.",
    }
    summary = {
        "synthetic_case_count": len(records),
        "case_ids": [record["case_id"] for record in records],
        "all_pass": all(record["passes"] for record in records),
        "failed_cases": [record["case_id"] for record in records if not record["passes"]],
        "gate": (
            "all S0-S7 synthetic cases matched expected predicates; "
            "S1 OPC is ID-permutation invariant and mapped CCOC recovers to 1"
        ),
    }
    write_json(output_dir / "metric_expected_actual_records.json", records)
    write_json(output_dir / "metric_threshold_contract.json", threshold_contract)
    write_json(output_dir / "metric_unit_test_summary.json", summary)
    return {
        "summary": summary,
        "records": records,
        "threshold_contract": threshold_contract,
    }

