#!/usr/bin/env python3
"""Audit simple no-GT value triggers for ACL2 v34.

This intentionally stays small and interpretable: it builds reset-group samples
from landed H1 oracle reports, trains threshold rules without absolute chunk id,
and evaluates leave-one-reset-group-out validation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SEMANTIC_Z = ("V31_A1B_SEM_Z_COARSE", "V31_A1_SEM_Z_FINE")
FEATURE_PREFIX = {
    "V31_A0_ORIG_C23": "orig",
    "V31_A1B_SEM_Z_COARSE": "coarse",
    "V31_A1_SEM_Z_FINE": "fine",
    "V31_A5B_SEM_RESID_COARSE_L025": "resid",
}


def _float(v: str | None) -> float | None:
    if v is None or v == "" or v.lower() == "nan":
        return None
    try:
        x = float(v)
    except Exception:
        return None
    if not math.isfinite(x):
        return None
    return x


def _bool(v: str | None) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes"}


@dataclass(frozen=True)
class Atom:
    feature: str
    op: str
    threshold: float

    def apply(self, sample: dict[str, Any]) -> bool:
        v = sample["features"].get(self.feature)
        if v is None or not math.isfinite(v):
            return False
        if self.op == "<=":
            return v <= self.threshold
        return v >= self.threshold

    def to_dict(self) -> dict[str, Any]:
        return {"feature": self.feature, "op": self.op, "threshold": self.threshold}


@dataclass(frozen=True)
class Rule:
    atoms: tuple[Atom, ...]

    def apply(self, sample: dict[str, Any]) -> bool:
        return all(atom.apply(sample) for atom in self.atoms)

    def to_dict(self) -> dict[str, Any]:
        return {"atoms": [a.to_dict() for a in self.atoms]}


def read_effects(path: Path, parent: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            row = dict(row)
            row["parent"] = parent
            rows.append(row)
    return rows


def build_samples(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault((row["parent"], int(row["chunk"])), []).append(row)

    samples: list[dict[str, Any]] = []
    for (parent, chunk), group_rows in sorted(grouped.items()):
        by_candidate = {r["candidate"]: r for r in group_rows}
        features: dict[str, float] = {}
        for candidate, prefix in FEATURE_PREFIX.items():
            row = by_candidate.get(candidate)
            if not row:
                continue
            for src, dst in [
                ("prior_mean_D_patch", "mean_d"),
                ("prior_q90_D_patch", "q90_d"),
                ("prior_v32_semantic_z_high_mass", "z_high_mass"),
                ("prior_v32_semantic_d_mean", "semantic_d_mean"),
                ("prior_v32_semantic_d_q90", "semantic_d_q90"),
                ("prior_v31_semantic_label_count", "label_count"),
                ("prior_v31_semantic_label_fallback_ratio", "fallback_ratio"),
            ]:
                val = _float(row.get(src))
                if val is not None:
                    features[f"{prefix}_{dst}"] = val

        for prefix in ("coarse", "fine", "resid"):
            for key in ("mean_d", "q90_d", "z_high_mass", "semantic_d_mean", "semantic_d_q90"):
                a = features.get(f"{prefix}_{key}")
                b = features.get(f"orig_{key}")
                if a is not None and b is not None:
                    features[f"{prefix}_minus_orig_{key}"] = a - b
        for key in ("mean_d", "q90_d", "z_high_mass", "semantic_d_mean", "semantic_d_q90"):
            a = features.get(f"coarse_{key}")
            b = features.get(f"fine_{key}")
            if a is not None and b is not None:
                features[f"coarse_minus_fine_{key}"] = a - b

        sem_rows = [by_candidate[c] for c in SEMANTIC_Z if c in by_candidate]
        gate_rows = [r for r in sem_rows if _bool(r.get("gate_pass"))]
        label = bool(gate_rows)
        best_ate = min((_float(r.get("ATE_delta_vs_base")) for r in sem_rows), default=None)
        seg_vals = [_float(r.get("intersection_200_300_delta_vs_base")) for r in sem_rows]
        seg_vals = [v for v in seg_vals if v is not None]
        best_seg = min(seg_vals) if seg_vals else None
        samples.append(
            {
                "parent": parent,
                "chunk": chunk,
                "label": label,
                "best_sem_ate_delta": best_ate,
                "best_sem_200_300_delta": best_seg,
                "features": features,
            }
        )
    return samples


def candidate_atoms(samples: list[dict[str, Any]]) -> list[Atom]:
    atoms: list[Atom] = []
    feature_names = sorted({k for s in samples for k in s["features"]})
    for feat in feature_names:
        values = sorted({s["features"].get(feat) for s in samples if s["features"].get(feat) is not None})
        values = [v for v in values if isinstance(v, float) and math.isfinite(v)]
        if not values:
            continue
        thresholds = set(values)
        for a, b in zip(values, values[1:]):
            thresholds.add((a + b) / 2.0)
        for t in sorted(thresholds):
            atoms.append(Atom(feat, "<=", t))
            atoms.append(Atom(feat, ">=", t))
    return atoms


def confusion(rule: Rule, samples: list[dict[str, Any]]) -> dict[str, int]:
    out = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    for sample in samples:
        pred = rule.apply(sample)
        truth = bool(sample["label"])
        if pred and truth:
            out["tp"] += 1
        elif pred and not truth:
            out["fp"] += 1
        elif not pred and truth:
            out["fn"] += 1
        else:
            out["tn"] += 1
    return out


def score_rule(rule: Rule, samples: list[dict[str, Any]]) -> tuple[float, float, float]:
    c = confusion(rule, samples)
    recall = c["tp"] / max(1, c["tp"] + c["fn"])
    fpr = c["fp"] / max(1, c["fp"] + c["tn"])
    precision = c["tp"] / max(1, c["tp"] + c["fp"])
    score = recall * 3.0 + precision - fpr * 2.0 - 0.01 * len(rule.atoms)
    return score, recall, fpr


def train_rule(samples: list[dict[str, Any]], max_atoms: int) -> Rule:
    atoms = candidate_atoms(samples)
    rules: list[Rule] = [Rule((a,)) for a in atoms]
    if max_atoms >= 2:
        # AND pairs only. OR rules can memorize disconnected positives too easily
        # in this tiny dataset, so keep the model conservative.
        for i, a in enumerate(atoms):
            for b in atoms[i + 1 :]:
                if a.feature == b.feature:
                    continue
                rules.append(Rule((a, b)))
    best = max(rules, key=lambda r: score_rule(r, samples))
    return best


def loo_validate(samples: list[dict[str, Any]], max_atoms: int) -> dict[str, Any]:
    chunks = sorted({s["chunk"] for s in samples})
    folds = []
    all_preds = []
    for chunk in chunks:
        train = [s for s in samples if s["chunk"] != chunk]
        test = [s for s in samples if s["chunk"] == chunk]
        rule = train_rule(train, max_atoms=max_atoms)
        fold_preds = []
        for sample in test:
            pred = rule.apply(sample)
            fold_preds.append({**{k: sample[k] for k in ("parent", "chunk", "label")}, "pred": pred})
            all_preds.append(fold_preds[-1])
        folds.append({"heldout_chunk": chunk, "rule": rule.to_dict(), "predictions": fold_preds, "train_confusion": confusion(rule, train)})
    tp = fp = tn = fn = 0
    for pred in all_preds:
        if pred["pred"] and pred["label"]:
            tp += 1
        elif pred["pred"] and not pred["label"]:
            fp += 1
        elif not pred["pred"] and pred["label"]:
            fn += 1
        else:
            tn += 1
    return {
        "folds": folds,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "heldout_positive_recall": tp / max(1, tp + fn),
        "false_positive_rate": fp / max(1, fp + tn),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h9-h15-effects", required=True)
    ap.add_argument("--c9-h15-effects", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-atoms", type=int, default=2)
    args = ap.parse_args()

    rows = read_effects(Path(args.h9_h15_effects), "H9") + read_effects(Path(args.c9_h15_effects), "C9")
    samples = build_samples(rows)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rule = train_rule(samples, max_atoms=args.max_atoms)
    train_conf = confusion(all_rule, samples)
    loo = loo_validate(samples, max_atoms=args.max_atoms)

    sample_rows = []
    feature_names = sorted({k for s in samples for k in s["features"]})
    for s in samples:
        row = {
            "parent": s["parent"],
            "chunk": s["chunk"],
            "oracle_label_h15": s["label"],
            "best_sem_ate_delta": s["best_sem_ate_delta"],
            "best_sem_200_300_delta": s["best_sem_200_300_delta"],
        }
        row.update({k: s["features"].get(k) for k in feature_names})
        sample_rows.append(row)
    write_csv(out_dir / "value_trigger_samples.csv", sample_rows, ["parent", "chunk", "oracle_label_h15", "best_sem_ate_delta", "best_sem_200_300_delta"] + feature_names)

    pred_rows = []
    for fold in loo["folds"]:
        rule_s = json.dumps(fold["rule"], sort_keys=True)
        for pred in fold["predictions"]:
            pred_rows.append({"heldout_chunk": fold["heldout_chunk"], "parent": pred["parent"], "chunk": pred["chunk"], "truth": pred["label"], "pred": pred["pred"], "rule": rule_s})
    write_csv(out_dir / "value_trigger_loo_predictions.csv", pred_rows, ["heldout_chunk", "parent", "chunk", "truth", "pred", "rule"])

    summary = {
        "samples": len(samples),
        "positive_samples": sum(1 for s in samples if s["label"]),
        "positive_chunks": sorted({s["chunk"] for s in samples if s["label"]}),
        "uses_absolute_chunk_id": False,
        "model_family": f"threshold_rule_AND_max_atoms_{args.max_atoms}",
        "trained_all_rule": all_rule.to_dict(),
        "trained_all_confusion": train_conf,
        "loo": loo,
        "gate_requirements": {
            "heldout_positive_recall_min": 0.5,
            "false_positive_rate_max": 0.35,
            "no_absolute_chunk_id": True,
        },
    }
    summary["trigger_gate_pass"] = (
        not summary["uses_absolute_chunk_id"]
        and loo["heldout_positive_recall"] >= 0.5
        and loo["false_positive_rate"] <= 0.35
    )
    with (out_dir / "value_trigger_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["trigger_gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
