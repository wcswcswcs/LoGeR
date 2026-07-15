from __future__ import annotations

from dataclasses import dataclass, field

from .geometry_capsule import GeometrySupport


@dataclass(frozen=True)
class GapSeed:
    seed_id: str
    frame_id: int
    component_id: str
    uv: tuple[float, float]
    distance_to_component_edge_px: float
    distance_to_image_edge_px: float

    @property
    def is_interior_seed(self) -> bool:
        return self.distance_to_component_edge_px > 0 and self.distance_to_image_edge_px > 0


@dataclass(frozen=True)
class GapCandidateMask:
    candidate_id: str
    frame_id: int
    seed_id: str
    area_px: int
    bbox_area_fraction: float
    touches_image_edge: bool
    sam2_multimask_index: int | None = None


@dataclass(frozen=True)
class GapHypothesis:
    hypothesis_id: str
    frame_id: int
    component_id: str
    candidate_ids: tuple[str, ...]
    geometry_support: GeometrySupport | None
    existing_anchor_conflict: bool
    output_allowed: bool
    memory_admission_allowed: bool
    reason: str


@dataclass
class GapHypothesisGraph:
    seeds: dict[str, GapSeed] = field(default_factory=dict)
    candidates: dict[str, GapCandidateMask] = field(default_factory=dict)
    hypotheses: dict[str, GapHypothesis] = field(default_factory=dict)

    def add_seed(self, seed: GapSeed) -> None:
        self.seeds[seed.seed_id] = seed

    def add_candidate(self, candidate: GapCandidateMask) -> None:
        if candidate.seed_id not in self.seeds:
            raise KeyError(f"candidate seed is missing: {candidate.seed_id}")
        self.candidates[candidate.candidate_id] = candidate

    def add_hypothesis(self, hypothesis: GapHypothesis) -> None:
        missing = [cid for cid in hypothesis.candidate_ids if cid not in self.candidates]
        if missing:
            raise KeyError(f"hypothesis candidates are missing: {missing}")
        if hypothesis.memory_admission_allowed and not hypothesis.output_allowed:
            raise ValueError("memory admission cannot be allowed when output is not allowed")
        self.hypotheses[hypothesis.hypothesis_id] = hypothesis
