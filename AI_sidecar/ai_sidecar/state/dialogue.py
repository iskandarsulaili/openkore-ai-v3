"""DialogueState — NPC conversation state and active dialogue."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DialogueOption(BaseModel):
    """An available dialogue option / menu choice."""

    model_config = ConfigDict(extra="ignore")

    index: int = 0
    text: str = ""


class DialogueState(BaseModel):
    """NPC conversation state — active dialogue, menu, and responses."""

    model_config = ConfigDict(extra="ignore")

    in_dialogue: bool = False
    npc_name: str | None = None
    npc_id: int | None = None
    npc_x: int | None = None
    npc_y: int | None = None
    npc_map: str | None = None
    dialogue_text: str | None = None
    options: list[DialogueOption] = Field(default_factory=list)
    option_count: int = 0
    has_menu: bool = False
    is_trading: bool = False
    conversation_id: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


def collect_dialogue(signals: dict[str, Any]) -> DialogueState:
    """Parse NPC conversation state from the bridge signal dict.

    Handles:
      - ``signals['dialogue']`` — dict with dialogue info
      - ``signals['in_dialogue']``, ``signals['npc_talk']`` — flat keys
      - ``signals['npc_name']``, ``signals['npc_identity']``
      - ``signals['menu_options']`` — list of menu option dicts/strings
    """
    d_dict: dict[str, Any] = signals.get("dialogue") or {}

    in_dialogue = bool(
        signals.get("in_dialogue", False)
        or d_dict.get("in_dialogue", False)
        or signals.get("npc_talk", False)
    )
    if not in_dialogue:
        return DialogueState(in_dialogue=False)

    # Parse options
    options_raw: list[Any] = list(
        signals.get("menu_options", d_dict.get("options", signals.get("options", [])))
    )
    options: list[DialogueOption] = []
    option_count = 0
    for opt in options_raw:
        if isinstance(opt, str):
            options.append(DialogueOption(index=option_count, text=opt))
            option_count += 1
        elif isinstance(opt, dict):
            options.append(
                DialogueOption(
                    index=int(opt.get("index", option_count)),
                    text=str(opt.get("text", opt.get("name", ""))),
                )
            )
            option_count += 1

    npc_name = (
        signals.get("npc_name")
        or d_dict.get("npc_name")
        or signals.get("npc_identity")
    )

    return DialogueState(
        in_dialogue=in_dialogue,
        npc_name=str(npc_name) if npc_name else None,
        npc_id=int(signals.get("npc_id", d_dict.get("npc_id", 0))) or None,
        npc_x=int(signals.get("npc_x", d_dict.get("npc_x", 0))) or None,
        npc_y=int(signals.get("npc_y", d_dict.get("npc_y", 0))) or None,
        npc_map=str(signals.get("npc_map", d_dict.get("npc_map", ""))) or None,
        dialogue_text=str(
            d_dict.get("text", signals.get("last_npc_text", ""))
        ) or None,
        options=options,
        option_count=option_count,
        has_menu=option_count > 0,
        is_trading=bool(d_dict.get("is_trading", False)),
        conversation_id=str(d_dict.get("conversation_id", signals.get("conversation_id", ""))) or None,
    )
