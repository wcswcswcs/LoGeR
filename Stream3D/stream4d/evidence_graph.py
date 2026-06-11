from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .mask_evidence import MaskObservation
from .object_memory import Object4D


@dataclass
class EvidenceNode:
    node_id: int
    frame_id: int
    mask_id: int
    carrier_ids: set[int] = field(default_factory=set)
    coverage_sum: float = 0.0
    observation_count: int = 0
    area: int = 0

    @property
    def coverage(self) -> float:
        return float(self.coverage_sum / max(self.observation_count, 1))


@dataclass
class EvidenceGraphResult:
    objects: dict[int, Object4D]
    diagnostics: dict[str, float]
    components: list[list[int]]


class _ComponentIndex:
    def __init__(self, nodes: list[EvidenceNode]) -> None:
        self.parent = list(range(len(nodes)))
        self.members: dict[int, set[int]] = {idx: {idx} for idx in range(len(nodes))}
        self.frame_masks: dict[int, dict[int, int]] = {
            idx: {int(node.frame_id): int(node.mask_id)} for idx, node in enumerate(nodes)
        }

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def can_union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        frames_a = self.frame_masks[ra]
        frames_b = self.frame_masks[rb]
        for frame_id, mask_id in frames_a.items():
            other_mask = frames_b.get(frame_id)
            if other_mask is not None and int(other_mask) != int(mask_id):
                return False
        return True

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb or not self.can_union(ra, rb):
            return False
        if len(self.members[ra]) < len(self.members[rb]):
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.members[ra].update(self.members.pop(rb))
        self.frame_masks[ra].update(self.frame_masks.pop(rb))
        return True

    def components(self) -> list[list[int]]:
        roots = sorted(self.members)
        return [sorted(self.members[root]) for root in roots]


class EvidenceGraphBuilder:
    def __init__(
        self,
        min_shared_carriers: int = 2,
        min_carrier_ioc: float = 0.50,
        min_component_observations: int = 1,
        min_component_carriers: int = 1,
        min_node_carriers: int = 1,
        min_node_coverage: float = 0.0,
        edge_coverage_power: float = 0.0,
    ) -> None:
        self.min_shared_carriers = max(1, int(min_shared_carriers))
        self.min_carrier_ioc = float(min_carrier_ioc)
        self.min_component_observations = max(1, int(min_component_observations))
        self.min_component_carriers = max(1, int(min_component_carriers))
        self.min_node_carriers = max(1, int(min_node_carriers))
        self.min_node_coverage = float(min_node_coverage)
        self.edge_coverage_power = max(0.0, float(edge_coverage_power))

    @staticmethod
    def _aggregate_nodes(observations: list[MaskObservation]) -> list[EvidenceNode]:
        by_key: dict[tuple[int, int], EvidenceNode] = {}
        for obs in observations:
            key = (int(obs.frame_id), int(obs.mask_id))
            node = by_key.get(key)
            if node is None:
                node = EvidenceNode(
                    node_id=len(by_key),
                    frame_id=int(obs.frame_id),
                    mask_id=int(obs.mask_id),
                    area=int(obs.area),
                )
                by_key[key] = node
            carriers = obs.carrier_set()
            node.carrier_ids.update(carriers)
            node.coverage_sum += float(len(carriers) / max(int(obs.area), 1))
            node.observation_count += 1
            node.area = max(int(node.area), int(obs.area))
        return sorted(by_key.values(), key=lambda item: (item.frame_id, item.mask_id))

    def _edge_candidates(self, nodes: list[EvidenceNode]) -> list[tuple[float, float, int, int, int]]:
        carrier_to_nodes: dict[int, list[int]] = {}
        for idx, node in enumerate(nodes):
            for carrier_id in node.carrier_ids:
                carrier_to_nodes.setdefault(int(carrier_id), []).append(idx)

        shared_counts: dict[tuple[int, int], int] = {}
        for node_ids in carrier_to_nodes.values():
            if len(node_ids) < 2:
                continue
            node_ids = sorted(set(node_ids))
            for pos, a in enumerate(node_ids):
                for b in node_ids[pos + 1 :]:
                    key = (int(a), int(b))
                    shared_counts[key] = shared_counts.get(key, 0) + 1

        edges: list[tuple[float, float, int, int, int]] = []
        max_coverage = max((float(node.coverage) for node in nodes), default=0.0)
        for (a, b), shared in shared_counts.items():
            denom = max(1, min(len(nodes[a].carrier_ids), len(nodes[b].carrier_ids)))
            score = float(shared / denom)
            sort_score = score
            if self.edge_coverage_power > 0.0 and max_coverage > 0.0:
                rel_coverage = min(float(nodes[a].coverage), float(nodes[b].coverage)) / max_coverage
                rel_coverage = max(rel_coverage, 1e-6)
                sort_score *= float(rel_coverage ** self.edge_coverage_power)
            edges.append((sort_score, score, int(shared), int(a), int(b)))
        edges.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return edges

    def build(self, observations: list[MaskObservation]) -> EvidenceGraphResult:
        raw_nodes = self._aggregate_nodes(observations)
        nodes = [
            node
            for node in raw_nodes
            if len(node.carrier_ids) >= self.min_node_carriers
            and float(node.coverage) >= self.min_node_coverage
        ]
        components = _ComponentIndex(nodes)
        edges = self._edge_candidates(nodes)
        accepted_edges = 0
        rejected_conflict_edges = 0
        rejected_weak_edges = 0
        for _, score, shared, a, b in edges:
            if int(shared) < self.min_shared_carriers or float(score) < self.min_carrier_ioc:
                rejected_weak_edges += 1
                continue
            if not components.can_union(a, b):
                rejected_conflict_edges += 1
                continue
            if components.union(a, b):
                accepted_edges += 1

        kept_components: list[list[int]] = []
        objects: dict[int, Object4D] = {}
        dropped_components = 0
        for member_ids in components.components():
            carrier_ids: set[int] = set()
            frame_support: dict[int, set[int]] = {}
            mask_observations: list[tuple[int, int, float]] = []
            frame_ids: set[int] = set()
            coverage_values: list[float] = []
            for node_id in member_ids:
                node = nodes[node_id]
                carrier_ids.update(node.carrier_ids)
                frame_support.setdefault(int(node.frame_id), set()).update(node.carrier_ids)
                coverage = float(node.coverage)
                frame_ids.add(int(node.frame_id))
                coverage_values.append(coverage)
                mask_observations.append((int(node.frame_id), int(node.mask_id), coverage))
            if len(mask_observations) < self.min_component_observations:
                dropped_components += 1
                continue
            if len(carrier_ids) < self.min_component_carriers:
                dropped_components += 1
                continue
            object_id = len(objects)
            obj = Object4D(
                object_id=object_id,
                carrier_ids=carrier_ids,
                frame_support=frame_support,
                mask_observations=sorted(mask_observations, key=lambda item: (item[0], item[1])),
                last_seen=0,
                birth_frame=min(frame_support) if frame_support else 0,
                state="active",
            )
            mean_coverage = float(np.mean(coverage_values)) if coverage_values else 0.0
            evidence_quality = (
                float(len(mask_observations))
                * np.sqrt(max(float(len(carrier_ids)), 1.0))
                * np.sqrt(max(float(len(frame_ids)), 1.0))
                * max(mean_coverage, 1e-6)
            )
            obj.evidence_num_nodes = int(len(member_ids))
            obj.evidence_num_frames = int(len(frame_ids))
            obj.evidence_num_carriers = int(len(carrier_ids))
            obj.evidence_mean_coverage = float(mean_coverage)
            obj.evidence_quality = float(evidence_quality)
            objects[object_id] = obj
            kept_components.append(member_ids)

        component_sizes = [len(item) for item in kept_components]
        carrier_sizes = [len(obj.carrier_ids) for obj in objects.values()]
        diagnostics = {
            "evidence_graph_num_raw_observations": float(len(observations)),
            "evidence_graph_num_raw_nodes": float(len(raw_nodes)),
            "evidence_graph_num_dropped_nodes": float(len(raw_nodes) - len(nodes)),
            "evidence_graph_num_nodes": float(len(nodes)),
            "evidence_graph_num_edge_candidates": float(len(edges)),
            "evidence_graph_accepted_edges": float(accepted_edges),
            "evidence_graph_rejected_conflict_edges": float(rejected_conflict_edges),
            "evidence_graph_rejected_weak_edges": float(rejected_weak_edges),
            "evidence_graph_num_components": float(len(components.components())),
            "evidence_graph_num_kept_components": float(len(kept_components)),
            "evidence_graph_num_dropped_components": float(dropped_components),
            "evidence_graph_mean_component_nodes": float(np.mean(component_sizes)) if component_sizes else 0.0,
            "evidence_graph_max_component_nodes": float(np.max(component_sizes)) if component_sizes else 0.0,
            "evidence_graph_mean_component_carriers": float(np.mean(carrier_sizes)) if carrier_sizes else 0.0,
            "evidence_graph_min_shared_carriers": float(self.min_shared_carriers),
            "evidence_graph_min_carrier_ioc": float(self.min_carrier_ioc),
            "evidence_graph_min_component_observations": float(self.min_component_observations),
            "evidence_graph_min_component_carriers": float(self.min_component_carriers),
            "evidence_graph_min_node_carriers": float(self.min_node_carriers),
            "evidence_graph_min_node_coverage": float(self.min_node_coverage),
            "evidence_graph_edge_coverage_power": float(self.edge_coverage_power),
        }
        return EvidenceGraphResult(objects=objects, diagnostics=diagnostics, components=kept_components)


class EvidenceGraphMemory:
    def __init__(self, objects: dict[int, Object4D], diagnostics: dict[str, float]) -> None:
        self.objects = objects
        self.diagnostics = diagnostics

    def to_jsonable(self) -> dict:
        return {
            "memory_version": "evidence_graph",
            "diagnostics": self.diagnostics,
            "objects": [
                {
                    "object_id": int(obj.object_id),
                    "num_carriers": int(len(obj.carrier_ids)),
                    "frames": sorted(int(k) for k in obj.frame_support.keys()),
                    "num_mask_observations": int(len(obj.mask_observations)),
                    "evidence_num_nodes": int(getattr(obj, "evidence_num_nodes", len(obj.mask_observations))),
                    "evidence_num_frames": int(getattr(obj, "evidence_num_frames", len(obj.frame_support))),
                    "evidence_num_carriers": int(getattr(obj, "evidence_num_carriers", len(obj.carrier_ids))),
                    "evidence_mean_coverage": float(getattr(obj, "evidence_mean_coverage", 0.0)),
                    "evidence_quality": float(getattr(obj, "evidence_quality", 0.0)),
                    "birth_frame": int(obj.birth_frame),
                    "state": obj.state,
                }
                for obj in sorted(self.objects.values(), key=lambda item: item.object_id)
            ],
        }
