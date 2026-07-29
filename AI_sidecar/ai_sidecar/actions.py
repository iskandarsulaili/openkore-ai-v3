"""Standalone action types — importable without triggering full module chain."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass
class HeuristicAction:
    """A single action produced by the heuristic / domain assessment.
    
    Kinds:
      - command:  An OpenKore command to execute (e.g., 'move prontera', 'set attackAuto 3')
      - log:      A log message (debug/diagnostic, no execution)
    """
    kind: str = "command"
    command: str = ""
    confidence: float = 0.9
    reason: str = ""
    domain: str = "general"
    metadata: dict[str, Any] = field(default_factory=dict)
