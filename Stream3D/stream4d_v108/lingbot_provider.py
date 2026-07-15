from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LingBotCapability:
    depth: bool
    pose_or_trajectory: bool
    intrinsics: bool
    confidence: bool
    source: str = "LingBot-Map"
    uses_scannet_pose_or_depth_for_projection: bool = False


@dataclass(frozen=True)
class LingBotPacket:
    frame_id: int
    depth_path: str | None
    pose_path: str | None
    intrinsics_path: str | None
    confidence_path: str | None
    metadata: dict[str, Any]


class LingBotProviderContract:
    """Contract wrapper for LingBot Map outputs used by v108."""

    capability = LingBotCapability(depth=True, pose_or_trajectory=True, intrinsics=True, confidence=True)

    def __init__(self, root: Path):
        self.root = Path(root)

    def describe(self) -> dict[str, Any]:
        return {
            "root": self.root.as_posix(),
            "source": self.capability.source,
            "depth": self.capability.depth,
            "pose_or_trajectory": self.capability.pose_or_trajectory,
            "intrinsics": self.capability.intrinsics,
            "confidence": self.capability.confidence,
            "uses_scannet_pose_or_depth_for_projection": False,
        }
