from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REQUIRED_NATIVE_MODULES = [
    "stream4d_native/object_tube_io.py",
    "stream4d_native/d4rt_scene_builder.py",
    "stream4d_native/self_stitch.py",
    "stream4d_native/measurement_bank.py",
    "stream4d_native/tube_cover.py",
    "stream4d_native/signed_tube_graph.py",
    "stream4d_native/tube_partition.py",
    "stream4d_native/tube_memory.py",
]

GEOMETRY_PATTERNS = [
    "xyz_local",
    "xyz_ref0",
    "xyz_ref",
    "xyz_canonical",
    "get_geometry_for_measurement",
    "get_geometry_for_merge",
    "point_ids",
]

METRIC_PATTERNS = [
    "np.linalg.norm",
    "cKDTree",
    "KDTree",
    "distance",
    "metric_merge",
    "memory_match",
    "get_geometry_for_merge",
    "assert_merge_geometry_valid",
]


@dataclass
class Site:
    path: str
    line: int
    function: str
    kind: str
    symbol: str
    classification: str
    guard_status: str
    snippet: str


class FunctionLocator(ast.NodeVisitor):
    def __init__(self) -> None:
        self.functions: list[tuple[int, int, str]] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        end = int(getattr(node, "end_lineno", node.lineno))
        self.functions.append((int(node.lineno), end, str(node.name)))
        self.generic_visit(node)


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _function_for_line(functions: list[tuple[int, int, str]], line: int) -> str:
    for start, end, name in functions:
        if start <= line <= end:
            return name
    return "<module>"


def _scan_file(root: Path, rel_path: str) -> tuple[list[Site], list[Site]]:
    path = root / rel_path
    if not path.exists():
        return [], []
    text = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(text)
        locator = FunctionLocator()
        locator.visit(tree)
        functions = locator.functions
    except SyntaxError:
        functions = []
    geometry: list[Site] = []
    metric: list[Site] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        snippet = line.strip()
        fn = _function_for_line(functions, line_no)
        for pattern in GEOMETRY_PATTERNS:
            if re.search(r"\b" + re.escape(pattern) + r"\b", line):
                classification = classify_geometry_read(rel_path, fn, pattern, snippet)
                geometry.append(
                    Site(rel_path, line_no, fn, "geometry_read", pattern, classification, guard_status(classification), snippet)
                )
        for pattern in METRIC_PATTERNS:
            if pattern in line:
                classification = classify_metric_op(rel_path, fn, pattern, snippet)
                metric.append(
                    Site(rel_path, line_no, fn, "metric_operation", pattern, classification, guard_status(classification), snippet)
                )
    return geometry, metric


def classify_geometry_read(path: str, function: str, symbol: str, snippet: str) -> str:
    if "object_tube_io.py" in path and function in {"<module>", "validate", "to_jsonable", "from_jsonable"}:
        return "tube_schema_or_serialization"
    if "object_tube_io.py" in path and function == "get_geometry_for_measurement":
        return "image_space_measurement_read"
    if "object_tube_io.py" in path and function in {"_merge_geometry_array", "get_geometry_for_merge"}:
        return "guarded_method_geometry_read"
    if "signed_tube_graph.py" in path and "get_geometry_for_merge" in snippet:
        return "guarded_method_geometry_read"
    if "signed_tube_graph.py" in path and function == "_spacing_scale" and symbol == "xyz_canonical":
        return "spacing_scale_canonical_only"
    if "measurement_bank.py" in path and "get_geometry_for_measurement" in snippet:
        return "image_space_measurement_read"
    if "tube_memory.py" in path and "get_geometry_for_merge" in snippet:
        return "guarded_method_geometry_read"
    if "d4rt_scene_builder.py" in path:
        if "xyz_canonical" in symbol or "stitch" in function or "canonical" in function:
            return "d4rt_self_sim3_canonicalization"
        return "d4rt_cache_or_model_decode"
    if "self_stitch.py" in path:
        return "d4rt_self_stitch_overlap_read"
    if "tools/" in path:
        return "diagnostic_or_trace_read"
    if "stream4d/" in path:
        return "legacy_stream4d_path_not_v25_native_method"
    return "unknown"


def classify_metric_op(path: str, function: str, symbol: str, snippet: str) -> str:
    if "object_tube_io.py" in path and function in {"<module>", "validate"} and symbol == "metric_merge":
        return "guard_metadata"
    if "d4rt_scene_builder.py" in path and symbol == "metric_merge":
        return "alignment_gate_metadata"
    if "signed_tube_graph.py" in path and function == "<module>" and symbol == "distance":
        return "edge_schema_no_geometry_read"
    if "signed_tube_graph.py" in path:
        if "get_geometry_for_merge" in snippet or function in {"build_signed_tube_graph", "_spacing_scale"}:
            return "guarded_or_spacing_normalized_native_metric"
    if "object_tube_io.py" in path and function in {"assert_merge_geometry_valid", "get_geometry_for_merge"}:
        return "guard_definition"
    if "self_stitch.py" in path:
        return "d4rt_self_sim3_matching_or_fit"
    if "tube_partition.py" in path:
        return "graph_partition_no_geometry_read"
    if "tube_memory.py" in path:
        return "guarded_memory_match"
    if "tools/" in path:
        return "diagnostic_or_trace_metric"
    if "stream4d/" in path:
        return "legacy_stream4d_path_not_v25_native_method"
    return "unknown"


def guard_status(classification: str) -> str:
    if classification.startswith("guarded") or classification in {"guard_definition", "guarded_memory_match"}:
        return "guarded"
    if classification in {
        "image_space_measurement_read",
        "graph_partition_no_geometry_read",
        "tube_schema_or_serialization",
        "guard_metadata",
        "alignment_gate_metadata",
        "edge_schema_no_geometry_read",
    }:
        return "not_metric_geometry"
    if classification == "spacing_scale_canonical_only":
        return "spacing_normalized_canonical_only"
    if classification.startswith("d4rt_self") or classification == "d4rt_cache_or_model_decode":
        return "pre_merge_native_geometry"
    if classification.startswith("diagnostic") or classification.startswith("legacy"):
        return "not_v25_method_path"
    return "unknown"


def _site_dict(site: Site) -> dict[str, Any]:
    return {
        "path": site.path,
        "line": site.line,
        "function": site.function,
        "kind": site.kind,
        "symbol": site.symbol,
        "classification": site.classification,
        "guard_status": site.guard_status,
        "snippet": site.snippet,
    }


def _write_csv(path: Path, rows: list[Site]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["path", "line", "function", "kind", "symbol", "classification", "guard_status", "snippet"]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(_site_dict(row))


def build_callgraph_summary(root: Path, geometry: list[Site], metric: list[Site]) -> dict[str, Any]:
    required = {module: (root / module).exists() for module in REQUIRED_NATIVE_MODULES}
    edges = [
        {"caller": "d4rt_scene_builder.extract_local_tubes_with_occupancy", "callee": "D4RT model infer_carriers"},
        {"caller": "d4rt_scene_builder.stitch_to_canonical", "callee": "self_stitch.match_overlap_carriers"},
        {"caller": "d4rt_scene_builder.stitch_to_canonical", "callee": "TubeRecord xyz_canonical materialization"},
        {"caller": "measurement_bank.build_measurement_bank", "callee": "TubeRecord.get_geometry_for_measurement"},
        {"caller": "tube_cover.select_tube_cover", "callee": "measurement_bank.MaskMeasurement"},
        {"caller": "signed_tube_graph.build_signed_tube_graph", "callee": "TubeRecord.get_geometry_for_merge"},
        {"caller": "TubeRecord.get_geometry_for_merge", "callee": "assert_merge_geometry_valid"},
        {"caller": "tube_partition.partition_tube_graph", "callee": "signed_tube_graph.TubeGraphEdge"},
        {"caller": "tube_memory.TubeMemory.update", "callee": "TubeRecord.get_geometry_for_merge"},
    ]
    unknown_native_geometry = [
        _site_dict(site)
        for site in geometry
        if site.guard_status == "unknown" and site.path.startswith("stream4d_native/")
    ]
    unknown_native_metric = [
        _site_dict(site)
        for site in metric
        if site.guard_status == "unknown" and site.path.startswith("stream4d_native/")
    ]
    return {
        "required_modules": required,
        "missing_required_module_count": int(sum(1 for ok in required.values() if not ok)),
        "geometry_read_site_count": int(len(geometry)),
        "metric_operation_site_count": int(len(metric)),
        "unknown_native_geometry_site_count": int(len(unknown_native_geometry)),
        "unknown_native_metric_site_count": int(len(unknown_native_metric)),
        "native_metric_guard_entry_present": any(
            site.path.endswith("signed_tube_graph.py") and site.symbol == "get_geometry_for_merge" for site in metric + geometry
        ),
        "default_range_id_fallback_present": "np.arange(int(length)" in (root / "stream4d_native/self_stitch.py").read_text(encoding="utf-8"),
        "callgraph_edges": edges,
        "unknown_native_geometry_sites": unknown_native_geometry,
        "unknown_native_metric_sites": unknown_native_metric,
    }


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# v25 Merge Callgraph Static Audit",
        "",
        "Diagnostic-only static audit for the v25 native merge path.",
        "",
        "## Summary",
        "",
        f"- missing_required_module_count: `{summary['missing_required_module_count']}`",
        f"- geometry_read_site_count: `{summary['geometry_read_site_count']}`",
        f"- metric_operation_site_count: `{summary['metric_operation_site_count']}`",
        f"- unknown_native_geometry_site_count: `{summary['unknown_native_geometry_site_count']}`",
        f"- unknown_native_metric_site_count: `{summary['unknown_native_metric_site_count']}`",
        f"- native_metric_guard_entry_present: `{summary['native_metric_guard_entry_present']}`",
        f"- default_range_id_fallback_present: `{summary['default_range_id_fallback_present']}`",
        "",
        "## Required Modules",
        "",
        "| module | present |",
        "| --- | ---: |",
    ]
    for module, ok in summary["required_modules"].items():
        lines.append(f"| `{module}` | `{ok}` |")
    lines.extend(["", "## Callgraph Edges", "", "| caller | callee |", "| --- | --- |"])
    for edge in summary["callgraph_edges"]:
        lines.append(f"| `{edge['caller']}` | `{edge['callee']}` |")
    if summary["unknown_native_geometry_sites"] or summary["unknown_native_metric_sites"]:
        lines.extend(["", "## Unknown Native Sites", ""])
        for row in summary["unknown_native_geometry_sites"] + summary["unknown_native_metric_sites"]:
            lines.append(f"- `{row['path']}:{row['line']}` `{row['symbol']}` {row['snippet']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = project_root()
    out_dir = Path(args.output_dir) if args.output_dir else root / "outputs/audit/v25_callgraph"
    scan_paths = list(REQUIRED_NATIVE_MODULES) + [
        "stream4d/measurement_bank.py",
        "stream4d/evidence_graph.py",
        "stream4d/object_memory.py",
        "tools/trace_d4rt_geometry_flow.py",
    ]
    geometry: list[Site] = []
    metric: list[Site] = []
    for rel in scan_paths:
        g, m = _scan_file(root, rel)
        geometry.extend(g)
        metric.extend(m)
    summary = build_callgraph_summary(root, geometry, metric)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "geometry_read_sites.csv", geometry)
    _write_csv(out_dir / "metric_operation_sites.csv", metric)
    (out_dir / "merge_callgraph_static.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(out_dir / "merge_callgraph_static.md", summary)
    if args.strict:
        failures = []
        if summary["missing_required_module_count"]:
            failures.append("missing required native v25 modules")
        if not summary["native_metric_guard_entry_present"]:
            failures.append("native signed graph does not call get_geometry_for_merge")
        if summary["default_range_id_fallback_present"]:
            failures.append("self_stitch still has default np.arange carrier-id fallback")
        if summary["unknown_native_geometry_site_count"] or summary["unknown_native_metric_site_count"]:
            failures.append("unknown native geometry/metric sites remain")
        if failures:
            print("v25 callgraph strict audit failed: " + "; ".join(failures), file=sys.stderr)
            return 2
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
