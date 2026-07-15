from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VideoExportPlan:
    export_mp4: bool
    sheets_per_chunk: int
    frames_per_sheet: int
    export_boundary_casebook: bool

