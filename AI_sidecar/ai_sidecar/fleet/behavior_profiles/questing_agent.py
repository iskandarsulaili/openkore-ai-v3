"""QuestingAgent — auto-accept, complete, turn-in, track quests, job change quests."""

from __future__ import annotations

from typing import Any

from ai_sidecar.fleet.behavior_profiles import BehaviorProfile


_QUEST_PRIORITY = {
    "job_change": 100, "access_quest": 80, "repeatable_exp": 60,
    "story": 40, "side": 20, "fetch": 10,
}


class QuestingAgent(BehaviorProfile):
    """Handles RO quest lifecycle — accept, progress, complete, job change quests."""

    def assess_quests(self, available_quests: list[dict[str, Any]],
                      active_quests: list[dict[str, Any]]) -> list[dict[str, Any]]:
        active_ids = {q.get("quest_id") for q in active_quests}
        candidates = []
        for q in available_quests:
            qid = q.get("quest_id", "")
            if qid in active_ids:
                continue
            priority = _QUEST_PRIORITY.get(q.get("type", "side"), 10)
            candidates.append({"quest_id": qid, "name": q.get("name", ""),
                               "type": q.get("type", ""), "priority": priority,
                               "level_req": q.get("level_req", 1)})
        candidates.sort(key=lambda x: (-x["priority"], x["level_req"]))
        return candidates[:5]

    def auto_accept(self, quest: dict[str, Any]) -> dict[str, Any]:
        return {"action": "accept_quest", "quest_id": quest["quest_id"],
                "quest_name": quest.get("name", ""), "npc": quest.get("giver_npc", "")}

    def auto_complete(self, quest: dict[str, Any], inventory: dict[str, int]) -> dict[str, Any]:
        required_items = quest.get("required_items", [])
        missing = [ri for ri in required_items if inventory.get(ri.get("item", ""), 0) < ri.get("qty", 1)]
        if missing:
            return {"action": "farm_quest_items", "needed": missing,
                    "target_mobs": quest.get("target_mobs", []),
                    "quest_id": quest["quest_id"]}
        return {"action": "turn_in_quest", "quest_id": quest["quest_id"],
                "npc": quest.get("completion_npc", "")}

    def prioritize_active(self, active_quests: list[dict[str, Any]],
                          base_level: int) -> dict[str, Any]:
        best_type = max(active_quests, key=lambda q: _QUEST_PRIORITY.get(q.get("type", "side"), 0))
        return {"focus_quest": best_type.get("quest_id", ""),
                "focus_name": best_type.get("name", ""),
                "progress_pct": best_type.get("progress", 0)}

    def job_change_quest(self, current_job: str, target_job: str,
                         base_level: int, job_level: int) -> dict[str, Any]:
        job_change_map = {
            "swordsman": {"knight": {"lvl": 40, "npc": "knight_union", "map": "prontera"},
                          "crusader": {"lvl": 40, "npc": "crusader_guild", "map": "payon"}},
            "mage": {"wizard": {"lvl": 40, "npc": "wizard_guild", "map": "geffen"},
                     "sage": {"lvl": 40, "npc": "sage_guild", "map": "geffen"}},
            "archer": {"hunter": {"lvl": 40, "npc": "hunter_guild", "map": "payon"},
                       "bard": {"lvl": 40, "npc": "bard_guild", "map": "payon"}},
            "acolyte": {"priest": {"lvl": 40, "npc": "priest_guild", "map": "prontera"},
                        "monk": {"lvl": 40, "npc": "monk_guild", "map": "morocc"}},
            "merchant": {"blacksmith": {"lvl": 40, "npc": "blacksmith_guild", "map": "geffen"},
                         "alchemist": {"lvl": 40, "npc": "alchemist_guild", "map": "aldebaran"}},
            "thief": {"assassin": {"lvl": 40, "npc": "assassin_guild", "map": "morocc"},
                      "rogue": {"lvl": 40, "npc": "rogue_guild", "map": "alberta"}},
        }
        routes = job_change_map.get(current_job, {})
        route = routes.get(target_job)
        if not route:
            return {"action": "no_route", "current": current_job, "target": target_job}
        req_lvl = route["lvl"]
        if base_level >= req_lvl and job_level >= 40:
            return {"action": "start_job_change", "npc": route["npc"],
                    "map": route["map"], "target_job": target_job,
                    "steps": ["talk_npc", "complete_quest", "get_new_job"]}
        return {"action": "level_first", "needed_base": max(0, req_lvl - base_level),
                "needed_job": max(0, 40 - job_level)}

    def record_outcome(self, action: str, success: bool, reward_xp: float = 0.0) -> None:
        self._record_experience("quest", action, success, reward=reward_xp, xp_gained=reward_xp)
