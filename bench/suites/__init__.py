"""One module per suite, each exposing `execute(run, engines)`."""

from __future__ import annotations

import importlib
from typing import Callable

from ..run import Run

SUITE_ORDER = ["tune", "correctness", "probe", "sweep", "single_stream", "kernels", "nsys",
               "memory", "scheduler", "ablation", "precision", "coldstart", "soak"]


def suite_function(name: str) -> Callable[[Run, list[str]], None]:
    return importlib.import_module(f"{__package__}.{name}").execute
