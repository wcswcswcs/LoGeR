from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvalAlignmentManifest:
    alignment_protocol: str
    uses_gt_for_prediction: bool
    uses_gt_for_evaluation_alignment: bool
    scale_aligned_eval_protocol: bool
    is_method_result: bool


def validate_eval_alignment_manifest(manifest: EvalAlignmentManifest | dict[str, object]) -> dict[str, object]:
    item = manifest if isinstance(manifest, dict) else manifest.__dict__
    prediction_ok = bool(item.get("uses_gt_for_prediction")) is False
    eval_marked = bool(item.get("uses_gt_for_evaluation_alignment")) is True
    method_ok = bool(item.get("is_method_result")) is False or bool(item.get("scale_aligned_eval_protocol")) is True
    return {
        "prediction_gt_leak_free": prediction_ok,
        "eval_alignment_marked": eval_marked,
        "method_protocol_marked": method_ok,
        "pass": bool(prediction_ok and eval_marked and method_ok),
    }

