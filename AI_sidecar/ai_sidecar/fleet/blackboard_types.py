from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


class FleetObjective(BaseModel):
    objective_id: str
    objective_type: str  # "farm", "quest", "level", "pvp", "gvg", "trade", "explore"
    target_map: Optional[str] = None
    target_mob: Optional[str] = None
    target_item: Optional[str] = None
    priority: int = 0
    assigned_bots: List[str] = Field(default_factory=list)
    min_bots: int = 1
    max_bots: int = 5
    start_time: Optional[datetime] = None
    deadline: Optional[datetime] = None
    status: str = "active"


class ZoneClaim(BaseModel):
    zone_id: str
    map_name: str
    claimed_by: str
    claimed_at: datetime
    expires_at: datetime
    purpose: str
    conflict_key: str


class TaskLease(BaseModel):
    lease_id: str
    task_type: str
    assigned_to: str
    params: Dict[str, Any] = Field(default_factory=dict)
    issued_at: datetime
    deadline: datetime
    status: str = "active"


class ThreatBulletin(BaseModel):
    threat_id: str
    threat_type: str
    map_name: str
    coordinates: Optional[Tuple[int, int]] = None
    severity: int = 1
    reported_by: str
    reported_at: datetime
    expires_at: datetime


class DoctrineDirective(BaseModel):
    doctrine_id: str
    version: int
    issued_at: datetime
    rules: List[str] = Field(default_factory=list)
    active: bool = True
