from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import build_v93_phase5_boundary_affinity_field as phase5  # noqa: E402


OUT = ROOT / "outputs/audit/v93_phase5_triton_kernel_validation"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        try:
            return str(path.relative_to(WORKSPACE_ROOT))
        except ValueError:
            return str(path)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0 and b.numel() == 0:
        return 0.0
    return float((a.detach() - b.detach()).abs().max().item())


def _max_abs_and_rel_diff(a: torch.Tensor, b: torch.Tensor, *, rel_floor: float = 1.0) -> tuple[float, float]:
    if a.numel() == 0 and b.numel() == 0:
        return 0.0, 0.0
    abs_diff = (a.detach() - b.detach()).abs()
    scale = torch.maximum(a.detach().abs(), b.detach().abs()).clamp_min(float(rel_floor))
    return float(abs_diff.max().item()), float((abs_diff / scale).max().item())


def _resolve_output_root(raw: str | Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == ROOT.name:
        return WORKSPACE_ROOT / path
    return ROOT / path


def _make_features(n_nodes: int, device: torch.device, seed: int) -> dict[str, torch.Tensor]:
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    sem = torch.rand(n_nodes, device=device, generator=gen, dtype=torch.float32)
    d4rt = torch.rand(n_nodes, device=device, generator=gen, dtype=torch.float32)
    inside = torch.rand(n_nodes, device=device, generator=gen, dtype=torch.float32)
    source_bar = torch.rand(n_nodes, device=device, generator=gen, dtype=torch.float32)
    nested_bar = torch.zeros(n_nodes, device=device, dtype=torch.float32)
    competing_bar = torch.zeros(n_nodes, device=device, dtype=torch.float32)
    negative = torch.zeros(n_nodes, device=device, dtype=torch.float32)
    if n_nodes:
        nested_bar[::2] = torch.rand((n_nodes + 1) // 2, device=device, generator=gen, dtype=torch.float32)
        competing_bar[1::3] = torch.rand(competing_bar[1::3].numel(), device=device, generator=gen, dtype=torch.float32)
        negative[::5] = torch.rand((n_nodes + 4) // 5, device=device, generator=gen, dtype=torch.float32)
    return {
        "sem": sem.contiguous(),
        "d4rt": d4rt.contiguous(),
        "inside": inside.contiguous(),
        "source_bar": source_bar.contiguous(),
        "nested_bar": nested_bar.contiguous(),
        "competing_bar": competing_bar.contiguous(),
        "negative": negative.contiguous(),
    }


def _make_edges(
    *,
    n_nodes: int,
    n_edges: int,
    mode: str,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if n_edges <= 0 or n_nodes <= 1:
        empty = torch.zeros(0, device=device, dtype=torch.long)
        return empty, empty
    if mode == "chain":
        u = torch.arange(0, n_nodes - 1, device=device, dtype=torch.long)
        v = torch.arange(1, n_nodes, device=device, dtype=torch.long)
        if u.numel() >= n_edges:
            return u[:n_edges].contiguous(), v[:n_edges].contiguous()
        reps = (n_edges + int(u.numel()) - 1) // int(u.numel())
        return u.repeat(reps)[:n_edges].contiguous(), v.repeat(reps)[:n_edges].contiguous()
    if mode == "star":
        v = (torch.arange(n_edges, device=device, dtype=torch.long) % (n_nodes - 1)) + 1
        u = torch.zeros_like(v)
        return u.contiguous(), v.contiguous()
    if mode == "duplicate":
        u = torch.zeros(n_edges, device=device, dtype=torch.long)
        v = torch.ones(n_edges, device=device, dtype=torch.long)
        return u, v
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    u = torch.randint(0, n_nodes, (n_edges,), device=device, generator=gen, dtype=torch.long)
    offset = torch.randint(1, n_nodes, (n_edges,), device=device, generator=gen, dtype=torch.long)
    v = (u + offset) % n_nodes
    return u.contiguous(), v.contiguous()


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    out = _resolve_output_root(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    if phase5.triton is None:
        raise RuntimeError("triton is not importable in the current environment")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; Triton kernel validation requires CUDA")

    torch.manual_seed(int(args.seed))
    device = torch.device(f"cuda:{int(args.device_index)}")
    torch.cuda.set_device(device)
    specs = phase5._variant_specs()
    n_variants = len(specs)
    tolerances = {
        "unary_max_abs": float(args.unary_tol),
        "message_max_abs": float(args.message_tol),
        "denom_max_abs": float(args.denom_abs_tol),
        "denom_max_rel": float(args.denom_rel_tol),
        "propagated_logit_max_abs": float(args.propagation_tol),
        "propagated_prob_max_abs": float(args.propagation_tol),
    }

    cases: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    global_max: dict[str, float] = {key: 0.0 for key in tolerances}

    unary_sizes = [1, 2, 7, 255, 256, 257, 513, 4096]
    for case_idx, n_nodes in enumerate(unary_sizes):
        features = _make_features(n_nodes, device, int(args.seed) + case_idx)
        triton_unary = phase5._compute_unary_triton(features, specs, device, "triton_cuda")
        torch_unary = phase5._compute_unary_triton(features, specs, device, "torch_cuda")
        torch.cuda.synchronize(device)
        diff = _max_abs_diff(triton_unary, torch_unary)
        global_max["unary_max_abs"] = max(global_max["unary_max_abs"], diff)
        row = {
            "case_type": "unary",
            "n_nodes": n_nodes,
            "n_variants": n_variants,
            "max_abs_diff": diff,
            "pass": diff <= tolerances["unary_max_abs"],
        }
        cases.append(row)
        if not row["pass"]:
            failures.append(row)

    edge_cases = [
        (1, 0, "empty"),
        (2, 1, "chain"),
        (7, 31, "random"),
        (255, 257, "random"),
        (256, 256, "chain"),
        (257, 777, "duplicate"),
        (513, 1025, "star"),
        (4096, 8191, "random"),
    ]
    for case_idx, (n_nodes, n_edges, mode) in enumerate(edge_cases):
        features = _make_features(n_nodes, device, int(args.seed) + 100 + case_idx)
        unary = phase5._compute_unary_triton(features, specs, device, "torch_cuda")
        prob = torch.sigmoid(unary)
        edge_u, edge_v = _make_edges(
            n_nodes=n_nodes,
            n_edges=n_edges,
            mode=mode,
            device=device,
            seed=int(args.seed) + 200 + case_idx,
        )
        gen = torch.Generator(device=device)
        gen.manual_seed(int(args.seed) + 300 + case_idx)
        edge_weight = torch.rand(n_variants, edge_u.numel(), device=device, generator=gen, dtype=torch.float32).contiguous()
        triton_msg, triton_denom = phase5._edge_message(prob, edge_u, edge_v, edge_weight, backend="triton_cuda")
        torch_msg, torch_denom = phase5._edge_message(prob, edge_u, edge_v, edge_weight, backend="torch_cuda")
        torch.cuda.synchronize(device)
        msg_diff = _max_abs_diff(triton_msg, torch_msg)
        denom_diff, denom_rel_diff = _max_abs_and_rel_diff(triton_denom, torch_denom)
        global_max["message_max_abs"] = max(global_max["message_max_abs"], msg_diff)
        global_max["denom_max_abs"] = max(global_max["denom_max_abs"], denom_diff)
        global_max["denom_max_rel"] = max(global_max["denom_max_rel"], denom_rel_diff)
        denom_pass = denom_diff <= tolerances["denom_max_abs"] or denom_rel_diff <= tolerances["denom_max_rel"]
        row = {
            "case_type": "edge_message",
            "n_nodes": n_nodes,
            "n_edges": int(edge_u.numel()),
            "edge_mode": mode,
            "n_variants": n_variants,
            "message_max_abs_diff": msg_diff,
            "denom_max_abs_diff": denom_diff,
            "denom_max_rel_diff": denom_rel_diff,
            "denom_pass_rule": "abs_or_rel",
            "pass": msg_diff <= tolerances["message_max_abs"] and denom_pass,
        }
        cases.append(row)
        if not row["pass"]:
            failures.append(row)

        smooth = torch.as_tensor([float(spec["smooth"]) for spec in specs], device=device, dtype=torch.float32)[:, None]
        triton_logits = unary.clone()
        torch_logits = unary.clone()
        for _ in range(int(args.label_prop_iters)):
            triton_msg, _ = phase5._edge_message(torch.sigmoid(triton_logits), edge_u, edge_v, edge_weight, backend="triton_cuda")
            torch_msg, _ = phase5._edge_message(torch.sigmoid(torch_logits), edge_u, edge_v, edge_weight, backend="torch_cuda")
            triton_logits = unary + smooth * (triton_msg - 0.5)
            torch_logits = unary + smooth * (torch_msg - 0.5)
        torch.cuda.synchronize(device)
        logit_diff = _max_abs_diff(triton_logits, torch_logits)
        prob_diff = _max_abs_diff(torch.sigmoid(triton_logits), torch.sigmoid(torch_logits))
        global_max["propagated_logit_max_abs"] = max(global_max["propagated_logit_max_abs"], logit_diff)
        global_max["propagated_prob_max_abs"] = max(global_max["propagated_prob_max_abs"], prob_diff)
        row = {
            "case_type": "label_propagation",
            "n_nodes": n_nodes,
            "n_edges": int(edge_u.numel()),
            "edge_mode": mode,
            "label_prop_iters": int(args.label_prop_iters),
            "n_variants": n_variants,
            "propagated_logit_max_abs_diff": logit_diff,
            "propagated_prob_max_abs_diff": prob_diff,
            "pass": logit_diff <= tolerances["propagated_logit_max_abs"] and prob_diff <= tolerances["propagated_prob_max_abs"],
        }
        cases.append(row)
        if not row["pass"]:
            failures.append(row)

    report = {
        "schema": "stream4d_v93_phase5_triton_kernel_validation_v1",
        "phase_id": phase5.PHASE_ID,
        "decision": "PASS_V93_PHASE5_TRITON_KERNEL_REFERENCE_CHECK" if not failures else "FAIL_V93_PHASE5_TRITON_KERNEL_REFERENCE_CHECK",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "duration_sec": time.time() - started,
        "seed": int(args.seed),
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "triton_available": phase5.triton is not None,
        "triton_version": getattr(phase5.triton, "__version__", ""),
        "n_variants": n_variants,
        "label_prop_iters": int(args.label_prop_iters),
        "tolerances": tolerances,
        "global_max_abs_diff": global_max,
        "case_count": len(cases),
        "failure_count": len(failures),
        "failure_cases": failures,
        "cases": cases,
        "notes": "Synthetic Torch reference check for Phase5 unary, edge-message, and label-propagation kernels. This does not use GT labels or evaluator metrics.",
    }
    report_path = out / "report.json"
    _write_json(report_path, report)
    _write_json(out / "SHA256SUMS.json", {_rel(report_path): _sha256(report_path)})
    print(json.dumps(_jsonable(report), indent=2, sort_keys=True), flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate v93 Phase5 Triton kernels against Torch reference tensors.")
    parser.add_argument("--output-root", default=str(OUT))
    parser.add_argument("--seed", type=int, default=9305)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--label-prop-iters", type=int, default=5)
    parser.add_argument("--unary-tol", type=float, default=5e-7)
    parser.add_argument("--message-tol", type=float, default=2e-5)
    parser.add_argument("--denom-abs-tol", type=float, default=2e-5)
    parser.add_argument("--denom-rel-tol", type=float, default=2e-6)
    parser.add_argument("--propagation-tol", type=float, default=2e-5)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    raise SystemExit(0 if result.get("failure_count", 1) == 0 else 2)
