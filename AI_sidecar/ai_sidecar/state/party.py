"""PartyState — party membership, members, leader, share settings."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PartyMember(BaseModel):
    """A member of the character's party."""

    model_config = ConfigDict(extra="ignore")

    name: str = ""
    level: int | None = None
    job: str | None = None
    hp: int | None = None
    hp_max: int | None = None
    sp: int | None = None
    sp_max: int | None = None
    map: str | None = None
    x: int | None = None
    y: int | None = None
    online: bool = True


class PartyState(BaseModel):
    """Party membership information."""

    model_config = ConfigDict(extra="ignore")

    in_party: bool = False
    is_leader: bool = False
    party_name: str | None = None
    leader_name: str | None = None
    members: list[PartyMember] = Field(default_factory=list)
    member_names: list[str] = Field(default_factory=list)
    share_exp: bool = False
    share_item: bool = False
    raw: dict[str, Any] = Field(default_factory=dict)


def collect_party(signals: dict[str, Any]) -> PartyState:
    """Parse party information from the bridge signal dict.

    Handles:
      - ``signals['in_party']`` — boolean
      - ``signals['party_name']`` — party name
      - ``signals['party_leader']`` or ``signals['leader_name']``
      - ``signals['party_members']`` — list of member names or dicts
      - ``signals['party_member_names']`` — pre-built name list
      - ``signals['is_leader']`` — whether this character is party leader
    """
    in_party = bool(signals.get("in_party", False))

    # Parse party members
    raw_members: list[Any] = list(signals.get("party_members", signals.get("party", [])) or [])
    party_name = signals.get("party_name") or None

    members: list[PartyMember] = []
    member_names: list[str] = []

    for m in raw_members:
        if isinstance(m, str):
            members.append(PartyMember(name=m))
            member_names.append(m)
        elif isinstance(m, dict):
            member = PartyMember(**{k: v for k, v in m.items() if k in PartyMember.model_fields})
            members.append(member)
            member_names.append(member.name)

    # Also accept pre-built member name list
    if not member_names:
        member_names = [str(n) for n in (signals.get("party_member_names") or [])]

    leader_name = signals.get("party_leader") or signals.get("leader_name") or (member_names[0] if member_names else None)
    is_leader = bool(signals.get("is_leader", False))

    return PartyState(
        in_party=in_party,
        is_leader=is_leader,
        party_name=party_name,
        leader_name=leader_name,
        members=members,
        member_names=member_names,
        share_exp=bool(signals.get("party_share_exp", True)),
        share_item=bool(signals.get("party_share_item", False)),
    )
