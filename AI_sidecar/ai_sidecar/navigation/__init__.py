"""
Navigation — Kafra, Fly Wing, Butterfly Wing, and Walk routing.

Provides the NavigationEngine for intelligent RO navigation:
  - Kafra warp NPC routing (data-driven from YAML)
  - Fly Wing usage decisions (danger-based)
  - Butterfly Wing return decisions (HP-based)
  - Walk route generation (fallback)
  - Best-route selection across all methods

Usage:
    from ai_sidecar.navigation.nav_engine import NavigationEngine, create_nav_engine

    engine = create_nav_engine("data/kafra_warp.yaml")
    route = engine.best_route("prontera", "payon", zeny=5000)
"""

from __future__ import annotations

from ai_sidecar.navigation.nav_engine import NavigationEngine, create_nav_engine

__all__ = [
    "NavigationEngine",
    "create_nav_engine",
]
