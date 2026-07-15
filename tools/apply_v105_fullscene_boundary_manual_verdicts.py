#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REVIEW_DATE = "2026-07-11"

PASS = "PASS_NO_OBVIOUS_BOUNDARY_ID_SWITCH"
POTENTIAL = "REPAIR_POTENTIAL_BOUNDARY_ID_SWITCH"
HARD = "NO_GO_HARD_BOUNDARY_ID_SWITCH"
UNCERTAIN = "REVIEW_UNCERTAIN_LARGE_VIEW_CHANGE"


OBSERVATIONS: dict[tuple[str, int, int], dict[str, str]] = {
    ("scene0030_00", 0, 1): {
        "verdict": PASS,
        "observation": (
            "Main table and green/blue chairs remain visually continuous across the boundary; a right table/chair enters "
            "after the boundary. Chair/table legs are fragmented but no large random ID switch was observed."
        ),
    },
    ("scene0030_00", 1, 2): {
        "verdict": PASS,
        "observation": (
            "Blackboard/wall, table, shelf, blue chair, and paper/book stacks remain visually continuous; no obvious "
            "full-object random switch was observed. Small shelf/book objects remain fragmented."
        ),
    },
    ("scene0030_00", 2, 3): {
        "verdict": PASS,
        "observation": (
            "Purple chair, green chair, counter, and paper stacks remain visually continuous; no obvious random ID collapse "
            "was observed, though revealed/cut regions change with view."
        ),
    },
    ("scene0030_00", 3, 4): {
        "verdict": PASS,
        "observation": (
            "Green swivel chair, yellow table, and large shelf regions remain visually continuous; the chair partly leaves "
            "view. Bookshelf objectlets are fine-grained but no hard boundary switch was observed."
        ),
    },
    ("scene0030_00", 4, 5): {
        "verdict": PASS,
        "observation": (
            "Yellow table and green chair remain stable while the bookcase view becomes more frontal; no large object was "
            "observed switching to a random ID."
        ),
    },
    ("scene0030_00", 5, 6): {
        "verdict": PASS,
        "observation": (
            "Red table and nearby green chair remain visually continuous; table/chair legs are noisy but the main IDs look "
            "stable."
        ),
    },
    ("scene0030_00", 6, 7): {
        "verdict": PASS,
        "observation": (
            "Red table, shelf, blue-gray chair, and right desk region remain visually stable; no obvious random boundary "
            "switch was observed."
        ),
    },
    ("scene0030_00", 7, 8): {
        "verdict": POTENTIAL,
        "observation": (
            "Most table/center/right chair coverage is present, but a left/back chair appears to change color across the "
            "boundary under similar visibility. This is a potential boundary ID switch and cannot be counted as a pass."
        ),
    },
    ("scene0030_00", 8, 9): {
        "verdict": HARD,
        "observation": (
            "The same sofa/armchair is pink before the boundary and beige after the boundary under continuous visibility. "
            "This is hard evidence against continuous full-scene identity."
        ),
    },
    ("scene0030_00", 9, 10): {
        "verdict": PASS,
        "observation": (
            "Two tables, blackboard, shelf, and paper regions remain visually stable; no clear random boundary switch was "
            "observed."
        ),
    },
    ("scene0030_00", 10, 11): {
        "verdict": PASS,
        "observation": (
            "Pink table, rear objects, and chair group remain visually stable while a foreground chair enters; no instant "
            "random boundary switch was observed."
        ),
    },
    ("scene0030_00", 11, 12): {
        "verdict": UNCERTAIN,
        "observation": (
            "Large viewpoint movement occurs at the boundary. Blue chair, pink table, and shelf/paper coverage remain "
            "visible, but the sheet is only medium-strength evidence and cannot prove continuous identity."
        ),
    },
    ("scene0030_00", 12, 13): {
        "verdict": POTENTIAL,
        "observation": (
            "The red table remains stable, but a center chair appears dark/brown before the boundary and red after the "
            "boundary while still visible. This is a potential hard ID switch requiring repair."
        ),
    },
    ("scene0030_00", 13, 14): {
        "verdict": UNCERTAIN,
        "observation": (
            "Large viewpoint change makes same-chair identity hard to confirm. Table/chair coverage is present, but this "
            "boundary is not a strong identity witness."
        ),
    },
    ("scene0591_00", 0, 1): {
        "verdict": PASS,
        "observation": (
            "Two display/screen regions, tabletop edge, and rear partition remain visually continuous; no large object hard "
            "ID switch was observed."
        ),
    },
    ("scene0591_00", 1, 2): {
        "verdict": PASS,
        "observation": (
            "Board/poster, foreground green screen/partition, and right purple partition remain visually continuous; small "
            "objects remain fragmented but no large hard switch was observed."
        ),
    },
    ("scene0591_00", 2, 3): {
        "verdict": POTENTIAL,
        "observation": (
            "Large door/cabinet structures are mostly continuous, but the same trash-bin-like small object appears gray "
            "before the boundary and green after the boundary. This is recorded as a potential small-object ID switch."
        ),
    },
    ("scene0591_00", 3, 4): {
        "verdict": PASS,
        "observation": (
            "Sofa body, two pillows, nearby long brown object, door panel, and floor remain visually consistent across the "
            "boundary; no hard ID jump was observed."
        ),
    },
    ("scene0591_00", 4, 5): {
        "verdict": HARD,
        "observation": (
            "The same tall cabinet/drawer unit is a large green region before the boundary and becomes blue/tan/purple "
            "segments after the boundary. This is a clear cross-chunk ID reassignment/split."
        ),
    },
    ("scene0591_00", 5, 6): {
        "verdict": HARD,
        "observation": (
            "The same desk/cabinet/file-grid structure changes from gray/wood-colored regions before the boundary to a "
            "large red-brown ID after the boundary. The pink box is stable, but the furniture body is not."
        ),
    },
    ("scene0591_00", 6, 7): {
        "verdict": PASS,
        "observation": (
            "Left white shelf, pink box/cabinet, and right purple cabinet remain visually corresponding across the boundary; "
            "only edge and sticker-sized fragments remain noisy."
        ),
    },
    ("scene0591_00", 7, 8): {
        "verdict": HARD,
        "observation": (
            "The same file cabinet/desk front is a single green region before the boundary and becomes a beige frame with "
            "multi-color drawers/compartments after the boundary. This is a clear cross-chunk ID reassignment."
        ),
    },
    ("scene0591_00", 8, 9): {
        "verdict": HARD,
        "observation": (
            "The sofa body changes from blue before the boundary to dark green/dark after the boundary while pillows remain "
            "in the same area. This is hard evidence against continuous identity."
        ),
    },
}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_boundary_records(root: Path) -> list[dict[str, Any]]:
    summary_path = root / "fullscene_l2h_stitch_summary.json"
    summary = _read_json(summary_path)
    if not isinstance(summary, dict):
        raise SystemExit(f"Expected dict summary at {summary_path}")
    records_path = Path(str(summary.get("boundary_sheet_records_json", "")))
    if not records_path.exists():
        raise SystemExit(f"Missing boundary records: {records_path}")
    records = _read_json(records_path)
    if not isinstance(records, list):
        raise SystemExit(f"Expected list records at {records_path}")
    return records


def _review_record(record: dict[str, Any]) -> dict[str, Any]:
    scene_id = str(record.get("scene_id"))
    prev_idx = int(record.get("prev_chunk_index"))
    curr_idx = int(record.get("curr_chunk_index"))
    key = (scene_id, prev_idx, curr_idx)
    observation = OBSERVATIONS.get(key)
    if observation is None:
        raise SystemExit(f"No manual observation for boundary {key}")

    verdict = observation["verdict"]
    is_pass = verdict == PASS
    reviewed = dict(record)
    reviewed.update(
        {
            "schema_version": "stream4d_v105_fullscene_boundary_manual_review_record_v1",
            "manual_review_date": REVIEW_DATE,
            "manual_review_method": "Codex visual inspection of one high-resolution 4x2 boundary sheet at original detail.",
            "verdict": verdict,
            "manual_identity_pass": is_pass,
            "hard_identity_failure": verdict == HARD,
            "potential_identity_failure": verdict == POTENTIAL,
            "uncertain_identity_witness": verdict == UNCERTAIN,
            "continuous_identity_proof": False,
            "observation": observation["observation"],
        }
    )
    return reviewed


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply manual full-scene boundary visual verdicts for v105 L2H stitch audit.")
    parser.add_argument("--stitch-root", required=True)
    args = parser.parse_args()

    root = Path(args.stitch_root)
    records_path = root / "boundary_visual_assessment_records.json"
    summary_path = root / "boundary_visual_assessment_summary.json"

    source_records = _load_boundary_records(root)
    reviewed_records = [_review_record(row) for row in source_records if isinstance(row, dict)]
    if len(reviewed_records) != len(source_records):
        raise SystemExit("At least one boundary sheet record is not a JSON object.")

    scene_counts: dict[str, Counter[str]] = defaultdict(Counter)
    verdict_counts = Counter(str(row.get("verdict")) for row in reviewed_records)
    for row in reviewed_records:
        scene_counts[str(row.get("scene_id"))][str(row.get("verdict"))] += 1

    hard_rows = [row for row in reviewed_records if row.get("verdict") == HARD]
    potential_rows = [row for row in reviewed_records if row.get("verdict") == POTENTIAL]
    uncertain_rows = [row for row in reviewed_records if row.get("verdict") == UNCERTAIN]
    pass_rows = [row for row in reviewed_records if row.get("verdict") == PASS]

    _write_json(records_path, reviewed_records)
    records_sha = _sha256(records_path)

    summary = {
        "schema_version": "stream4d_v105_fullscene_boundary_manual_review_summary_v1",
        "stitch_root": str(root),
        "records_json": str(records_path),
        "records_sha256": records_sha,
        "manual_review_date": REVIEW_DATE,
        "manual_review_complete": len(reviewed_records) == len(source_records),
        "record_count": len(reviewed_records),
        "source_boundary_record_count": len(source_records),
        "pass_no_obvious_switch_count": len(pass_rows),
        "potential_identity_failure_count": len(potential_rows),
        "hard_identity_failure_count": len(hard_rows),
        "uncertain_identity_witness_count": len(uncertain_rows),
        "hard_or_potential_failure_count": len(hard_rows) + len(potential_rows),
        "failure_or_uncertain_count": len(hard_rows) + len(potential_rows) + len(uncertain_rows),
        "continuous_scene_level_id_claim": False,
        "manual_review_conclusion": (
            "Manual boundary review is complete, but continuous full-scene identity is No-Go: hard boundary ID switches "
            "and additional potential/uncertain witnesses remain."
        ),
        "claim_boundary": (
            "This manual review only inspects rendered overlay boundary sheets. It records visible ID continuity failures "
            "and does not modify mask geometry or repair IDs."
        ),
        "verdict_counts": dict(sorted(verdict_counts.items())),
        "scene_verdict_counts": {scene: dict(sorted(counts.items())) for scene, counts in sorted(scene_counts.items())},
        "hard_identity_failures": [
            {
                "scene_id": row.get("scene_id"),
                "prev_chunk_index": row.get("prev_chunk_index"),
                "curr_chunk_index": row.get("curr_chunk_index"),
                "path": row.get("path"),
                "observation": row.get("observation"),
            }
            for row in hard_rows
        ],
        "potential_identity_failures": [
            {
                "scene_id": row.get("scene_id"),
                "prev_chunk_index": row.get("prev_chunk_index"),
                "curr_chunk_index": row.get("curr_chunk_index"),
                "path": row.get("path"),
                "observation": row.get("observation"),
            }
            for row in potential_rows
        ],
        "uncertain_identity_witnesses": [
            {
                "scene_id": row.get("scene_id"),
                "prev_chunk_index": row.get("prev_chunk_index"),
                "curr_chunk_index": row.get("curr_chunk_index"),
                "path": row.get("path"),
                "observation": row.get("observation"),
            }
            for row in uncertain_rows
        ],
    }
    _write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
