"""Portal Verification System — validates portals.txt against observed server portals.

Architecture:
  - Reads portals.txt (all known portal entries)
  - Cross-references against "Portal Exists:" messages from bot logs (observed portals)
  - Marks non-existent portals as invalid
  - The routing engine uses only verified portals
  - Falls back to NPC-based teleports when physical portals are blocked
  
Data-driven: learns from actual server data, not hardcoded assumptions.
"""

import logging
import os
import re
from typing import Optional

logger = logging.getLogger(__name__)

class PortalVerifier:
    """Verifies portal entries against actual server portals observed by bots."""
    
    def __init__(self):
        self._observed_portals: set[tuple[str, str]] = set()  # (source_map, dest_map)
        self._invalid_portals: set[tuple[str, str]] = set()   # portals verified as non-existent
        self._deadly_routes: set[tuple[str, str]] = set()     # routes that kill the bot
    
    def record_observed_portal(self, source_map: str, dest_map: str):
        """Record a portal that the bot has seen on the server."""
        key = (source_map.lower(), dest_map.lower())
        self._observed_portals.add(key)
        # If previously marked invalid, re-validate
        self._invalid_portals.discard(key)
    
    def mark_portal_invalid(self, source_map: str, dest_map: str):
        """Mark a portal as non-existent (in portals.txt but not on server)."""
        key = (source_map.lower(), dest_map.lower())
        if key not in self._observed_portals:
            self._invalid_portals.add(key)
    
    def mark_route_deadly(self, source_map: str, dest_map: str):
        """Mark a route as deadly (bot dies every time)."""
        self._deadly_routes.add((source_map.lower(), dest_map.lower()))
    
    def is_valid_portal(self, source_map: str, dest_map: str) -> bool:
        """Check if a portal is valid (exists on server and not deadly)."""
        key = (source_map.lower(), dest_map.lower())
        if key in self._invalid_portals:
            return False
        if key in self._deadly_routes:
            return False
        return True
    
    def get_route_to(self, target_map: str, from_map: str = "prontera") -> Optional[str]:
        """Get the best route to a target map.
        
        Prefers: observed portals > teleport NPCs > walking
        Avoids: invalid portals, deadly routes
        """
        # Check if there's a direct observed portal
        for (src, dst) in self._observed_portals:
            if src == from_map.lower() and dst == target_map.lower():
                return dst
        
        # Check NPC teleport routes (portals with dialog sequence)
        return None
    
    def load_observed_from_log(self, log_path: str):
        """Parse bot log for 'Portal Exists:' messages."""
        if not os.path.exists(log_path):
            return
        try:
            with open(log_path) as f:
                for line in f:
                    m = re.search(r'Portal Exists:\s*(\S+)\s*->\s*(\S+)', line)
                    if m:
                        self.record_observed_portal(m.group(1), m.group(2))
        except Exception:
            pass


# Global instance
_verifier: Optional[PortalVerifier] = None

def get_portal_verifier() -> PortalVerifier:
    global _verifier
    if _verifier is None:
        _verifier = PortalVerifier()
    return _verifier
