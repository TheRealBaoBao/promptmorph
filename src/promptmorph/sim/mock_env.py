"""Fast deterministic environment used by CI and the first vertical-slice demo.

This is not presented as physics. It proves compiler/planner/runtime contracts before
MuJoCo is introduced, which keeps failures attributable and tests fast.
"""

from __future__ import annotations

from promptmorph.models import EntityKind, EntityState, Pose, WorldFrame


def marker_cup_frame(
    timestamp_s: float,
    marker_xyz: tuple[float, float, float],
    cup_xyz: tuple[float, float, float],
    *,
    gripper_xyz: tuple[float, float, float] = (0.0, 0.0, 0.30),
    gripper_closed: bool = False,
    confidence: float = 1.0,
) -> WorldFrame:
    return WorldFrame(
        timestamp_s=timestamp_s,
        entities={
            "marker": EntityState(
                entity_id="marker",
                kind=EntityKind.OBJECT,
                pose=Pose(position=marker_xyz),
                size_xyz=(0.015, 0.015, 0.12),
                confidence=confidence,
            ),
            "cup": EntityState(
                entity_id="cup",
                kind=EntityKind.TARGET,
                pose=Pose(position=cup_xyz),
                size_xyz=(0.08, 0.08, 0.10),
                confidence=confidence,
            ),
        },
        gripper_pose=Pose(position=gripper_xyz),
        gripper_closed=gripper_closed,
    )

