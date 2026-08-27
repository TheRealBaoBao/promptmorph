"""Simulator-independent runtime boundaries."""

from typing import Protocol

from promptmorph.models import ActionChunk, ExecutionResult, WorldFrame


class PerceptionBackend(Protocol):
    def observe(self) -> WorldFrame: ...


class ChunkExecutor(Protocol):
    def execute(self, chunk: ActionChunk) -> ExecutionResult: ...
