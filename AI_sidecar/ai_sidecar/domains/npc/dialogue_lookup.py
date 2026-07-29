"""NPC Dialogue Lookup — deterministic replacement for LLM-based NPC interaction.

The LLM should NOT be called for NPC dialogue. All RO NPC interactions
are deterministic — the responses never change. This module provides
fast, reliable lookup-table-based NPC dialogue handling.
"""
from __future__ import annotations
from typing import Any
import logging
from pathlib import Path

from ai_sidecar.actions import HeuristicAction

logger = logging.getLogger(__name__)

try:
    import yaml
except ImportError:
    yaml = None

_DATA_DIR = Path(__file__).parent.parent.parent / "data"
_NPCDB = None


def _load_npc_data() -> dict:
    global _NPCDB
    if _NPCDB is not None:
        return _NPCDB
    if yaml is None:
        _NPCDB = {}
        return _NPCDB
    path = _DATA_DIR / "npc_dialogues.yaml"
    if path.exists():
        with open(path) as f:
            _NPCDB = yaml.safe_load(f) or {}
    else:
        _NPCDB = {}
    return _NPCDB


class NPCDialogLookup:
    """Deterministic NPC dialogue lookup.
    
    Replaces LLM calls for NPC interaction. All RO NPC dialogues
    are known and fixed — no need for AI.
    
    Usage:
        lookup = NPCDialogLookup()
        commands = lookup.get_job_change_dialogue("novice", "swordman")
        # Returns: ["h", "r1", "c", "r0"] (talk sequence)
    """
    
    def __init__(self):
        self._data = _load_npc_data()
    
    def get_job_change_dialogue(self, current_job: str, target_job: str) -> dict | None:
        """Get the NPC dialogue sequence for job change.
        
        Args:
            current_job: Current job name (lowercase, e.g. 'novice')
            target_job: Target job name (lowercase, e.g. 'swordman')
            
        Returns:
            Dict with npc, dialogue sequence, cost, level_req, or None
        """
        dialogues = self._data.get("npc_dialogues", {})
        jc = dialogues.get("job_change", {})
        
        # Direct lookup
        key = f"{current_job}_to_{target_job}"
        if key in jc:
            return jc[key]
        
        # Try reverse (some might be swapped)
        key = f"{target_job}_to_{current_job}"
        if key in jc:
            return jc[key]
        
        return None
    
    def get_shop(self, npc_location: str) -> dict | None:
        """Get shop inventory for an NPC."""
        shops = self._data.get("npc_dialogues", {}).get("npc_shops", {})
        return shops.get(npc_location)
    
    def get_kafra_position(self, town: str) -> dict | None:
        """Get Kafra position in a town."""
        kafras = self._data.get("npc_dialogues", {}).get("kafra", {})
        return kafras.get(town)
    
    def get_talk_command(self, dialogue_data: dict) -> str:
        """Convert NPC dialogue data to a talk command."""
        npc_pos = dialogue_data.get("npc", "")
        dialogue_seq = dialogue_data.get("dialogue", ["h", "c", "r0"])
        
        if not npc_pos:
            return ""
        
        # Build command: talknpc x y seq1 seq2 ...
        parts = npc_pos.split()
        if len(parts) >= 3:
            from ai_sidecar.domains.navigation.portals import PortalDB
            cmd = f"talknpc {parts[0]} {parts[1]} {parts[2]}"
            for step in dialogue_seq:
                cmd += f" {step}"
            return cmd
        else:
            coords = npc_pos.split()
            if len(coords) == 2:
                cmd = f"talknpc {coords[0]} {coords[1]}"
                for step in dialogue_seq:
                    cmd += f" {step}"
                return cmd
        
        return ""


def get_npc_lookup() -> NPCDialogLookup:
    return NPCDialogLookup()
