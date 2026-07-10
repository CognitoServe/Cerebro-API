"""
agent/tracer.py
Structured step-by-step logger for the agent loop.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from typing import Literal

Phase = Literal["PLAN", "ACTION", "OBSERVATION", "FINAL", "ERROR", "GUARDRAIL"]

_LOG_FILE: str | None = os.environ.get("AGENT_LOG_FILE")
_log_handle = None


def _get_log_handle():
    global _log_handle
    if _LOG_FILE and _log_handle is None:
        _log_handle = open(_LOG_FILE, "a", encoding="utf-8")  # noqa: SIM115
    return _log_handle


def print_trace(iteration: int, phase: Phase, content: str) -> None:
    """
    Emit a single structured trace line.

    Format:
        [HH:MM:SS] [ITER NN | PHASE    ] content
    """
    timestamp   = datetime.now(tz=timezone.utc).strftime("%H:%M:%S")
    iter_label  = f"ITER {iteration + 1:02d}"
    phase_pad   = phase.ljust(11)
    line        = f"[{timestamp}] [{iter_label} | {phase_pad}] {content}"

    print(line, flush=True)

    fh = _get_log_handle()
    if fh:
        fh.write(line + "\n")
        fh.flush()


def print_separator(label: str = "") -> None:
    """Print a visual divider."""
    width = 72
    if label:
        pad  = (width - len(label) - 2) // 2
        line = "─" * pad + f" {label} " + "─" * pad
    else:
        line = "─" * width
    print(line, flush=True)
