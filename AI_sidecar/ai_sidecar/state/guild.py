"""GuildState — guild membership, position, skills."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class GuildMember(BaseModel):
    """A member of the character's guild."""

    model_config = ConfigDict(extra="ignore")

    name: str = ""
    position: str | None = None
    level: int | None = None
    online: bool = True
    contribution: int | None = None


class GuildSkill(BaseModel):
    """A guild skill (passive or active)."""

    model_config = ConfigDict(extra="ignore")

    skill_id: int = 0
    name: str = ""
    level: int = 0
    max_level: int = 0


class GuildState(BaseModel):
    """Guild membership information."""

    model_config = ConfigDict(extra="ignore")

    in_guild: bool = False
    guild_name: str | None = None
    guild_id: int | None = None
    position: str | None = None
    position_id: int | None = None
    members: list[GuildMember] = Field(default_factory=list)
    skills: list[GuildSkill] = Field(default_factory=list)
    contribution: int = 0
    guild_level: int | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


def collect_guild(signals: dict[str, Any]) -> GuildState:
    """Parse guild information from the bridge signal dict.

    Handles:
      - ``signals['guild_name']`` — guild name
      - ``signals['guild_id']`` — guild numeric ID
      - ``signals['guild_position']`` — character's position title
      - ``signals['guild_members']`` — list of member dicts/names
      - ``signals['guild_skills']`` — list of guild skill dicts
      - ``signals['guild_contribution']`` or ``signals['guild_exp``
    """
    guild_name = signals.get("guild_name") or None
    if guild_name is None:
        # Not in a guild
        return GuildState(in_guild=False)

    raw_members: list[dict] = list(signals.get("guild_members", signals.get("guildMembers", [])) or [])
    members: list[GuildMember] = []
    for m in raw_members:
        if isinstance(m, str):
            members.append(GuildMember(name=m))
        elif isinstance(m, dict):
            members.append(GuildMember(**{k: v for k, v in m.items() if k in GuildMember.model_fields}))

    raw_skills: list[dict] = list(signals.get("guild_skills", signals.get("guildSkills", [])) or [])
    skills: list[GuildSkill] = []
    for s in raw_skills:
        if isinstance(s, dict):
            skills.append(GuildSkill(**{k: v for k, v in s.items() if k in GuildSkill.model_fields}))

    return GuildState(
        in_guild=True,
        guild_name=guild_name,
        guild_id=int(signals.get("guild_id", 0)) or None,
        position=signals.get("guild_position") or None,
        position_id=int(signals.get("guild_position_id", 0)) or None,
        members=members,
        skills=skills,
        contribution=int(signals.get("guild_contribution", signals.get("guild_exp", 0))),
        guild_level=int(signals.get("guild_level", 0)) or None,
    )
