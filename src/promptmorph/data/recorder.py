"""Atomic, append-in-memory recorder for simulator interaction episodes."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from promptmorph.models import ActionChunk, Demonstration, RuntimeEvent, WorldFrame

SCHEMA_VERSION = "promptmorph.episode.v1"


def _json_line(model: BaseModel) -> str:
    return model.model_dump_json() + "\n"


def _atomic_write(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


@dataclass
class EpisodeRecorder:
    root: Path
    episode_id: str
    seed: int
    config: dict[str, Any]
    _frames: list[WorldFrame] = field(default_factory=list)
    _actions: list[ActionChunk] = field(default_factory=list)
    _events: list[RuntimeEvent] = field(default_factory=list)

    @property
    def episode_dir(self) -> Path:
        return self.root / self.episode_id

    def record_frame(self, frame: WorldFrame) -> None:
        if self._frames and frame.timestamp_s <= self._frames[-1].timestamp_s:
            raise ValueError("recorded frame timestamps must be strictly increasing")
        self._frames.append(frame)

    def record_action(self, chunk: ActionChunk) -> None:
        self._actions.append(chunk)

    def record_event(self, event: RuntimeEvent) -> None:
        self._events.append(event)

    def demonstration(self) -> Demonstration:
        return Demonstration(demonstration_id=self.episode_id, frames=tuple(self._frames))

    def close(self, *, outcome: str, extra: dict[str, Any] | None = None) -> Path:
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "episode_id": self.episode_id,
            "seed": self.seed,
            "outcome": outcome,
            "frame_count": len(self._frames),
            "action_count": len(self._actions),
            "event_count": len(self._events),
            "config": self.config,
            **(extra or {}),
        }
        _atomic_write(self.episode_dir / "metadata.json", json.dumps(metadata, indent=2) + "\n")
        _atomic_write(
            self.episode_dir / "frames.jsonl", "".join(_json_line(item) for item in self._frames)
        )
        _atomic_write(
            self.episode_dir / "actions.jsonl", "".join(_json_line(item) for item in self._actions)
        )
        _atomic_write(
            self.episode_dir / "events.jsonl", "".join(_json_line(item) for item in self._events)
        )
        return self.episode_dir
