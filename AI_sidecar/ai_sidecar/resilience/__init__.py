"""Resilience module — self-healing, edge-case detection, and recovery."""
from __future__ import annotations

from ai_sidecar.resilience.edge_case_handler import EdgeCaseHandler, create_edge_case_handler

__all__ = ["EdgeCaseHandler", "create_edge_case_handler"]