"""NVTX ranges for profiling one engine step. Off unless ENGINE_NVTX=1, and then `nvtx_range`
hands back one shared no-op context, so the hot path pays only a method call."""

from __future__ import annotations

import contextlib
import os
from typing import ContextManager

ENABLED = os.environ.get("ENGINE_NVTX") == "1"

_NOOP = contextlib.nullcontext()


class _Range:
    def __init__(self, name: str):
        self._name = name

    def __enter__(self) -> None:
        import torch

        torch.cuda.nvtx.range_push(self._name)

    def __exit__(self, *exc: object) -> None:
        import torch

        torch.cuda.nvtx.range_pop()


def nvtx_range(name: str) -> ContextManager[None]:
    return _Range(name) if ENABLED else _NOOP
