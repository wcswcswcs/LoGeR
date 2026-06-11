#!/usr/bin/env python3
"""Small resumable HTTP range downloader for v29C official KITTI zip files."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import time
import urllib.request
from pathlib import Path
from typing import Dict, Tuple


def _head(url: str) -> Tuple[int, bool]:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=30) as resp:
        length = int(resp.headers.get("Content-Length", "0"))
        ranges = "bytes" in resp.headers.get("Accept-Ranges", "").lower()
    if length <= 0:
        raise RuntimeError(f"Could not determine content length for {url}")
    return length, ranges


def _download_range(url: str, path: Path, done_dir: Path, index: int, start: int, end: int) -> Dict[str, object]:
    marker = done_dir / f"{index:05d}.done"
    expected = end - start + 1
    if marker.exists():
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
            if int(payload.get("start", -1)) == start and int(payload.get("end", -1)) == end:
                return {"index": index, "skipped": True, "bytes": expected}
        except Exception:
            pass

    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    written = 0
    with urllib.request.urlopen(req, timeout=120) as resp:
        code = getattr(resp, "status", 0)
        if code not in (200, 206):
            raise RuntimeError(f"Range {index} returned HTTP {code}")
        with path.open("r+b", buffering=0) as handle:
            handle.seek(start)
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                written += len(chunk)
    if written != expected:
        raise RuntimeError(f"Range {index} wrote {written} bytes, expected {expected}")
    marker.write_text(
        json.dumps({"index": index, "start": start, "end": end, "bytes": written}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"index": index, "skipped": False, "bytes": written}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-mib", type=int, default=512)
    args = parser.parse_args()

    url = str(args.url)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    done_dir = output.with_suffix(output.suffix + ".range_done")
    done_dir.mkdir(parents=True, exist_ok=True)

    length, ranges = _head(url)
    if not ranges:
        raise RuntimeError(f"Server did not advertise byte ranges for {url}")

    with output.open("a+b") as handle:
        handle.truncate(length)

    chunk_size = max(1, int(args.chunk_mib)) * 1024 * 1024
    ranges_to_fetch = []
    count = int(math.ceil(length / chunk_size))
    for index in range(count):
        start = index * chunk_size
        end = min(length - 1, start + chunk_size - 1)
        ranges_to_fetch.append((index, start, end))

    print(
        json.dumps(
            {
                "url": url,
                "output": str(output),
                "bytes": length,
                "ranges": len(ranges_to_fetch),
                "workers": int(args.workers),
                "chunk_mib": int(args.chunk_mib),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    started = time.time()
    completed = 0
    total_bytes = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = [
            pool.submit(_download_range, url, output, done_dir, index, start, end)
            for index, start, end in ranges_to_fetch
        ]
        for fut in concurrent.futures.as_completed(futures):
            rec = fut.result()
            completed += 1
            total_bytes += int(rec["bytes"])
            elapsed = max(time.time() - started, 1e-6)
            if completed == len(futures) or completed % 4 == 0:
                print(
                    json.dumps(
                        {
                            "completed_ranges": completed,
                            "total_ranges": len(futures),
                            "logical_bytes": total_bytes,
                            "elapsed_seconds": round(elapsed, 3),
                            "logical_mib_per_second": round(total_bytes / 1024 / 1024 / elapsed, 3),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    os.sync()
    print(json.dumps({"status": "complete", "output": str(output), "bytes": output.stat().st_size}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
