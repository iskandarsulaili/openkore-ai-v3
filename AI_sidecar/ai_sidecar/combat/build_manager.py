"""
Build Manager — real Ragnarok Online build data for 6 major classes.

Provides stat progression per level, equipment goals, skill order,
and build-aware combat recommendations.  Thread-safe singleton.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class StatAllocation:
    """Stat points to add at a given level range."""
    level_min: int
    level_max: int
    str_: int = 0
    agi: int = 0
    vit: int = 0
    int_: int = 0
    dex: int = 0
    luk: int = 0
    description: str = ""


@dataclass
class SkillLearnOrder:
    """A skill to learn, in order."""
    skill_name: str
    level: int
    required_level: int
    description: str = ""


@dataclass
class EquipmentGoal:
    """A piece of equipment to aim for."""
    slot: str
    name: str
    level_requirement: int = 1
    priority: int = 50
    notes: str = ""


@dataclass
class Build:
    """Complete build definition."""
    name: str
    job_class: str
    display_name: str
    description: str = ""
    primary_stats: list[str] = field(default_factory=lambda: ["str"])
    secondary_stats: list[str] = field(default_factory=lambda: ["dex"])
    stat_progression: list[StatAllocation] = field(default_factory=list)
    skill_order: list[SkillLearnOrder] = field(default_factory=list)
    equipment_goals: list[EquipmentGoal] = field(default_factory=list)
    rotation_name: str = ""
    buff_priority: list[str] = field(default_factory=list)
    notes: str = ""


class BuildManager:
    """Manages real RO build data for all major classes."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._builds: dict[str, Build] = {}
        self._load_builds()

    def _load_builds(self) -> None:
        """Load real build data for 6 major classes."""

        # HUNTER (Dex/Agi)
        self._builds["hunter_dex_agi"] = Build(
            name="hunter_dex_agi",
            job_class="hunter",
            display_name="Hunter (Dex/Agi)",
            description="Bow-based DPS. High flee, high ASPD, elemental arrows for coverage.",
            primary_stats=["dex", "agi"],
            secondary_stats=["luk", "str"],
            stat_progression=[
                StatAllocation(1, 20, dex=9, agi=6, description="Early damage + flee"),
                StatAllocation(21, 40, dex=7, agi=5, luk=3, description="Balance DEX/AGI, start LUK"),
                StatAllocation(41, 60, dex=6, agi=5, luk=4, description="Crit build phase"),
                StatAllocation(61, 80, dex=5, agi=5, luk=3, str=2, description="Add STR for weight"),
                StatAllocation(81, 99, dex=4, agi=4, luk=4, str=3, description="Cap DEX/AGI, push LUK"),
            ],
            skill_order=[
                SkillLearnOrder("Double Strafe", 10, 1, "Core DPS skill"),
                SkillLearnOrder("Improve Concentration", 10, 1, "Buff: DEX + AGI"),
                SkillLearnOrder("Arrow Shower", 5, 1, "AoE knockback"),
                SkillLearnOrder("Owl's Eye", 10, 1, "Passive DEX boost"),
                SkillLearnOrder("Vulture's Eye", 10, 1, "Passive range + hit"),
                SkillLearnOrder("Anklesnare", 5, 1, "Trap: immobilize + fire damage"),
                SkillLearnOrder("Blast Arrow", 5, 1, "Fire-element AoE"),
                SkillLearnOrder("Falconry Mastery", 5, 1, "Falcon damage"),
                SkillLearnOrder("Steel Crow", 5, 1, "Falcon attack"),
                SkillLearnOrder("Blitz Beat", 5, 1, "Auto-falcon proc"),
                SkillLearnOrder("Detect", 4, 1, "Reveal hidden"),
            ],
            equipment_goals=[
                EquipmentGoal("weapon", "Composite Bow [4]", 1, 90, "Slot for elemental cards"),
                EquipmentGoal("weapon", "Gakkung Bow [2]", 40, 80, "Higher ATK, 2 slots"),
                EquipmentGoal("weapon", "Arbalest [2]", 70, 70, "Endgame bow"),
                EquipmentGoal("weapon", "Hunter Bow [1]", 85, 60, "Endgame + elemental arrow"),
                EquipmentGoal("armor", "Tights [1]", 40, 80, "DEX + 1, slot for Peco Peco card"),
                EquipmentGoal("garment", "Muffler [1]", 1, 70, "Slot for Whisper card (flee)"),
                EquipmentGoal("shoes", "Boots [1]", 1, 70, "Slot for Matyr card (AGI)"),
                EquipmentGoal("accessory", "Glove [1]", 30, 80, "DEX + 2, slot for Zerom card"),
                EquipmentGoal("headgear", "Apple of Archer", 30, 75, "DEX + 3"),
            ],
            rotation_name="hunter_rotation",
            buff_priority=["Improve Concentration", "Wind Walker"],
            notes="Carry 4+ elemental arrow quivers. Use Anklesnare before Double Strafe for trapped bonus.",
        )

        # WIZARD (Int/Dex)
        self._builds["wizard_int_dex"] = Build(
            name="wizard_int_dex",
            job_class="wizard",
            display_name="Wizard (Int/Dex)",
            description="Elemental spellcaster. High MATK, fast cast, full elemental coverage.",
            primary_stats=["int", "dex"],
            secondary_stats=["vit", "luk"],
            stat_progression=[
                StatAllocation(1, 20, int_=9, dex=6, description="MATK + cast speed"),
                StatAllocation(21, 40, int_=8, dex=5, vit=2, description="Add VIT for survival"),
                StatAllocation(41, 60, int_=7, dex=5, vit=3, description="Balance INT/DEX"),
                StatAllocation(61, 80, int_=6, dex=4, vit=3, luk=2, description="Start LUK for cast"),
                StatAllocation(81, 99, int_=5, dex=4, vit=3, luk=3, description="Cap INT, push DEX"),
            ],
            skill_order=[
                SkillLearnOrder("Fire Bolt", 10, 1, "Fire single-target"),
                SkillLearnOrder("Cold Bolt", 10, 1, "Water single-target"),
                SkillLearnOrder("Lightning Bolt", 10, 1, "Wind single-target"),
                SkillLearnOrder("Fire Ball", 5, 1, "Fire AoE"),
                SkillLearnOrder("Fire Wall", 5, 1, "Defensive fire wall"),
                SkillLearnOrder("Frost Diver", 5, 1, "Freeze + water damage"),
                SkillLearnOrder("Frost Nova", 5, 1, "AoE freeze"),
                SkillLearnOrder("Thunderstorm", 5, 1, "Wind AoE"),
                SkillLearnOrder("Safety Wall", 10, 1, "Defensive barrier"),
                SkillLearnOrder("Napalm Beat", 5, 1, "Ghost damage vs undead"),
                SkillLearnOrder("Soul Strike", 5, 1, "Ghost damage vs undead/ghost"),
                SkillLearnOrder("Heaven's Drive", 5, 1, "Neutral AoE"),
                SkillLearnOrder("Storm Gust", 10, 1, "Water AoE freeze"),
                SkillLearnOrder("Meteor Storm", 10, 1, "Fire AoE stun"),
                SkillLearnOrder("Lord of Vermilion", 10, 1, "Wind AoE"),
                SkillLearnOrder("Energy Coat", 5, 1, "Defensive buff"),
            ],
            equipment_goals=[
                EquipmentGoal("weapon", "Wand [2]", 1, 90, "Slot for Int/Matk cards"),
                EquipmentGoal("weapon", "Arc Wand [2]", 40, 80, "INT + 3, 2 slots"),
                EquipmentGoal("weapon", "Lich's Bone Wand [2]", 70, 70, "INT + 5, MATK + 15%"),
                EquipmentGoal("weapon", "Staff of Destruction [1]", 85, 60, "Endgame staff"),
                EquipmentGoal("armor", "Robe of Cast [1]", 40, 80, "INT + 2, slot for Marc card"),
                EquipmentGoal("shield", "Guard [1]", 1, 70, "Slot for Thara Frog card"),
                EquipmentGoal("shoes", "Shoes [1]", 1, 70, "Slot for Sohee card (SP regen)"),
                EquipmentGoal("accessory", "Earring [1]", 30, 80, "INT + 2, slot for Zerom card"),
                EquipmentGoal("headgear", "Pecopeco Hairband", 30, 75, "INT + 1, DEX + 1"),
            ],
            rotation_name="wizard_rotation",
            buff_priority=["Energy Coat"],
            notes="Always lead with Safety Wall. Match spell element to target element for 200% damage.",
        )

        # PRIEST (Int/Vit)
        self._builds["priest_int_vit"] = Build(
            name="priest_int_vit",
            job_class="priest",
            display_name="Priest (Int/Vit)",
            description="Support/healer. High INT for heal power, VIT for survival.",
            primary_stats=["int", "vit"],
            secondary_stats=["dex", "luk"],
            stat_progression=[
                StatAllocation(1, 20, int_=9, vit=6, description="Heal power + HP"),
                StatAllocation(21, 40, int_=8, vit=5, dex=2, description="Add DEX for cast speed"),
                StatAllocation(41, 60, int_=7, vit=5, dex=3, description="Balance INT/VIT"),
                StatAllocation(61, 80, int_=6, vit=4, dex=3, luk=2, description="Start LUK"),
                StatAllocation(81, 99, int_=5, vit=4, dex=3, luk=3, description="Cap INT, push VIT"),
            ],
            skill_order=[
                SkillLearnOrder("Heal", 10, 1, "Core heal + undead damage"),
                SkillLearnOrder("Blessing", 10, 1, "STR/DEX/INT buff"),
                SkillLearnOrder("Increase Agility", 10, 1, "AGI/DEX buff"),
                SkillLearnOrder("Teleport", 4, 1, "Escape skill"),
                SkillLearnOrder("Holy Light", 5, 1, "Holy damage vs undead/demon"),
                SkillLearnOrder("Turn Undead", 5, 1, "Instant-kill undead"),
                SkillLearnOrder("Magnificat", 5, 1, "SP regen buff"),
                SkillLearnOrder("Gloria", 5, 1, "LUK buff"),
                SkillLearnOrder("Kyrie Eleison", 10, 1, "Auto-guard buff"),
                SkillLearnOrder("Impositio Manus", 5, 1, "ATK buff"),
                SkillLearnOrder("Aspersio", 5, 1, "Holy weapon enchant"),
                SkillLearnOrder("Assumptio", 5, 1, "Damage reduction buff"),
                SkillLearnOrder("Lex Divina", 5, 1, "Silence target"),
                SkillLearnOrder("Lex Aeterna", 5, 1, "Double magic damage"),
                SkillLearnOrder("Resurrection", 4, 1, "Revive party members"),
            ],
            equipment_goals=[
                EquipmentGoal("weapon", "Mace [3]", 1, 90, "Slot for Vadon/Archer skeleton cards"),
                EquipmentGoal("weapon", "Chain [3]", 40, 80, "Higher ATK, 3 slots"),
                EquipmentGoal("weapon", "Sword Mace [2]", 70, 70, "Endgame mace"),
                EquipmentGoal("armor", "Holy Robe [1]", 40, 80, "MDEF + 5, slot for Marc card"),
                EquipmentGoal("shield", "Holy Guard [1]", 40, 75, "MDEF + 3, slot for Thara Frog"),
                EquipmentGoal("shoes", "Shoes [1]", 1, 70, "Slot for Sohee card"),
                EquipmentGoal("accessory", "Glove [1]", 30, 80, "DEX + 2"),
                EquipmentGoal("headgear", "Tiara [1]", 30, 75, "INT + 2"),
            ],
            rotation_name="priest_rotation",
            buff_priority=["Gloria", "Magnificat", "Blessing", "Increase Agility", "Kyrie Eleison", "Impositio Manus", "Aspersio", "Assumptio"],
            notes="Buff order: Gloria -> Magnificat -> Blessing -> Increase AGI. Heal damages undead. Turn Undead for instant kills.",
        )

        # KNIGHT (Str/Dex)
        self._builds["knight_str_dex"] = Build(
            name="knight_str_dex",
            job_class="knight",
            display_name="Knight (Str/Dex)",
            description="Melee DPS. High ATK, Bowling Bash for AoE, Magnum Break for fire coverage.",
            primary_stats=["str", "dex"],
            secondary_stats=["vit", "agi"],
            stat_progression=[
                StatAllocation(1, 20, str_=9, dex=6, description="ATK + hit rate"),
                StatAllocation(21, 40, str_=8, dex=5, vit=2, description="Add VIT for HP"),
                StatAllocation(41, 60, str_=7, dex=5, vit=3, description="Balance STR/DEX"),
                StatAllocation(61, 80, str_=6, dex=4, vit=3, agi=2, description="Add AGI for ASPD"),
                StatAllocation(81, 99, str_=5, dex=4, vit=3, agi=3, description="Cap STR, push DEX"),
            ],
            skill_order=[
                SkillLearnOrder("Bash", 10, 1, "Single-target stun"),
                SkillLearnOrder("Magnum Break", 5, 1, "Fire-element AoE"),
                SkillLearnOrder("Provoke", 5, 1, "Taunt + def reduction"),
                SkillLearnOrder("Endure", 5, 1, "Defensive buff"),
                SkillLearnOrder("Bowling Bash", 10, 1, "Core AoE DPS"),
                SkillLearnOrder("Two-Handed Sword Mastery", 10, 1, "ATK + 30"),
                SkillLearnOrder("Spear Mastery", 10, 1, "Spear ATK"),
                SkillLearnOrder("Spear Stab", 5, 1, "Spear single-target"),
                SkillLearnOrder("Spear Boomerang", 5, 1, "Spear ranged attack"),
                SkillLearnOrder("Brandish Spear", 5, 1, "Spear AoE"),
                SkillLearnOrder("Pierce", 5, 1, "Spear multi-hit vs large"),
                SkillLearnOrder("Cavalier Mastery", 5, 1, "Mounted combat"),
            ],
            equipment_goals=[
                EquipmentGoal("weapon", "Two-Handed Sword [2]", 1, 90, "Slot for Vadon/Drainliar cards"),
                EquipmentGoal("weapon", "Great Sword [2]", 40, 80, "Higher ATK"),
                EquipmentGoal("weapon", "Muramash [1]", 70, 70, "Cursed blade, high ATK"),
                EquipmentGoal("weapon", "Holy Avenger [1]", 85, 60, "Endgame holy sword"),
                EquipmentGoal("armor", "Chain Mail [1]", 40, 80, "VIT + 1, slot for Peco Peco card"),
                EquipmentGoal("garment", "Manteau [1]", 1, 70, "Slot for Whisper card"),
                EquipmentGoal("shoes", "Greaves [1]", 1, 70, "Slot for Matyr card"),
                EquipmentGoal("accessory", "Ring [1]", 30, 80, "STR + 2, slot for Mantis card"),
                EquipmentGoal("headgear", "Helm [1]", 30, 75, "STR + 1"),
            ],
            rotation_name="knight_rotation",
            buff_priority=["Endure"],
            notes="Bowling Bash is your main DPS. Magnum Break for fire-weak enemies. Use Bash to stun casters.",
        )

        # ASSASSIN (Str/Agi)
        self._builds["assassin_str_agi"] = Build(
            name="assassin_str_agi",
            job_class="assassin",
            display_name="Assassin (Str/Agi)",
            description="Dual-wield DPS. High ASPD, critical hits, poison damage.",
            primary_stats=["str", "agi"],
            secondary_stats=["luk", "dex"],
            stat_progression=[
                StatAllocation(1, 20, str_=6, agi=9, description="ASPD first"),
                StatAllocation(21, 40, str_=5, agi=7, luk=3, description="Start crit build"),
                StatAllocation(41, 60, str_=5, agi=5, luk=4, dex=1, description="Balance STR/AGI"),
                StatAllocation(61, 80, str_=5, agi=4, luk=4, dex=2, description="Add DEX for hit"),
                StatAllocation(81, 99, str_=4, agi=4, luk=4, dex=3, description="Cap AGI, push STR"),
            ],
            skill_order=[
                SkillLearnOrder("Double Attack", 10, 1, "Passive double hit"),
                SkillLearnOrder("Throw Sand", 5, 1, "Blind + interrupt"),
                SkillLearnOrder("Hide", 10, 1, "Stealth + escape"),
                SkillLearnOrder("Sonic Blow", 10, 1, "High-damage single-target"),
                SkillLearnOrder("Grimtooth", 5, 1, "AoE dagger throw"),
                SkillLearnOrder("Katar Mastery", 10, 1, "Katar ATK + crit"),
                SkillLearnOrder("Right-Hand Mastery", 10, 1, "Dual-wield damage"),
                SkillLearnOrder("Left-Hand Mastery", 10, 1, "Dual-wield damage"),
                SkillLearnOrder("Enchant Poison", 5, 1, "Poison weapon buff"),
                SkillLearnOrder("Enchant Deadly Poison", 5, 1, "Deadly poison buff"),
                SkillLearnOrder("Cloaking", 5, 1, "Improved stealth"),
                SkillLearnOrder("Soul Destroyer", 5, 1, "Ranged magic damage"),
            ],
            equipment_goals=[
                EquipmentGoal("weapon", "Katar [2]", 1, 90, "Slot for crit cards"),
                EquipmentGoal("weapon", "Jamadhar [2]", 40, 80, "High ATK katar"),
                EquipmentGoal("weapon", "Inverse Scale [2]", 70, 70, "Endgame katar"),
                EquipmentGoal("weapon", "Katar of Rogue [1]", 85, 60, "Endgame + crit"),
                EquipmentGoal("armor", "Thief Clothes [1]", 40, 80, "AGI + 1, slot for Peco Peco card"),
                EquipmentGoal("garment", "Muffler [1]", 1, 70, "Slot for Whisper card"),
                EquipmentGoal("shoes", "Boots [1]", 1, 70, "Slot for Matyr card"),
                EquipmentGoal("accessory", "Glove [1]", 30, 80, "DEX + 2"),
                EquipmentGoal("headgear", "Assassin Mask", 30, 75, "STR + 1, AGI + 1"),
            ],
            rotation_name="assassin_rotation",
            buff_priority=["Enchant Deadly Poison", "Enchant Poison", "Cloaking"],
            notes="Sonic Blow for burst, Grimtooth for groups. Keep Enchant Deadly Poison up. Hide to reset aggro.",
        )

        # BLACKSMITH (Str/Dex)
        self._builds["blacksmith_str_dex"] = Build(
            name="blacksmith_str_dex",
            job_class="blacksmith",
            display_name="Blacksmith (Str/Dex)",
            description="Melee DPS + crafting. High ATK, weapon perfection, hammer skills.",
            primary_stats=["str", "dex"],
            secondary_stats=["vit", "luk"],
            stat_progression=[
                StatAllocation(1, 20, str_=9, dex=6, description="ATK + hit rate"),
                StatAllocation(21, 40, str_=8, dex=5, vit=2, description="Add VIT for HP"),
                StatAllocation(41, 60, str_=7, dex=5, vit=3, description="Balance STR/DEX"),
                StatAllocation(61, 80, str_=6, dex=4, vit=3, luk=2, description="Start LUK for crafting"),
                StatAllocation(81, 99, str_=5, dex=4, vit=3, luk=3, description="Cap STR, push DEX"),
            ],
            skill_order=[
                SkillLearnOrder("Mammonite", 10, 1, "High-damage single-target (costs zeny)"),
                SkillLearnOrder("Cart Revolution", 5, 1, "AoE cart attack"),
                SkillLearnOrder("Overcharge", 10, 1, "Better buy/sell prices"),
                SkillLearnOrder("Discount", 10, 1, "Buy items cheaper"),
                SkillLearnOrder("Hammer Fall", 5, 1, "AoE stun + damage"),
                SkillLearnOrder("Power Thrust", 5, 1, "ATK buff + ignore def"),
                SkillLearnOrder("Weapon Perfection", 5, 1, "Ignore size penalty"),
                SkillLearnOrder("Weaponry Research", 10, 1, "ATK + weapon crafting"),
                SkillLearnOrder("Adrenaline Rush", 5, 1, "ASPD buff for party"),
                SkillLearnOrder("Skin Tempering", 5, 1, "VIT + DEF buff"),
                SkillLearnOrder("Enchanted Stone Craft", 5, 1, "Elemental weapon crafting"),
                SkillLearnOrder("Weapon Production", 10, 1, "Craft weapons"),
            ],
            equipment_goals=[
                EquipmentGoal("weapon", "Hammer [2]", 1, 90, "Slot for cards"),
                EquipmentGoal("weapon", "Battle Hammer [2]", 40, 80, "Higher ATK"),
                EquipmentGoal("weapon", "War Axe [2]", 70, 70, "Endgame axe"),
                EquipmentGoal("weapon", "Doom Axe [1]", 85, 60, "Endgame + fire element"),
                EquipmentGoal("armor", "Chain Mail [1]", 40, 80, "VIT + 1"),
                EquipmentGoal("garment", "Manteau [1]", 1, 70, "Slot for Whisper card"),
                EquipmentGoal("shoes", "Greaves [1]", 1, 70, "Slot for Matyr card"),
                EquipmentGoal("accessory", "Ring [1]", 30, 80, "STR + 2"),
                EquipmentGoal("headgear", "Helm [1]", 30, 75, "STR + 1"),
            ],
            rotation_name="blacksmith_rotation",
            buff_priority=["Weapon Perfection", "Power Thrust", "Adrenaline Rush", "Skin Tempering"],
            notes="Keep Weapon Perfection active to ignore size penalties. Power Thrust before big hits. Adrenaline Rush for party ASPD.",
        )

    def get_build(self, name: str) -> Build | None:
        with self._lock:
            return self._builds.get(name)

    def get_builds_for_class(self, job_class: str) -> list[Build]:
        with self._lock:
            return [b for b in self._builds.values() if b.job_class == job_class]

    def get_all_builds(self) -> list[Build]:
        with self._lock:
            return list(self._builds.values())

    def get_build_names(self) -> list[str]:
        with self._lock:
            return list(self._builds.keys())

    def get_recommended_build(self, job_class: str) -> Build | None:
        builds = self.get_builds_for_class(job_class)
        return builds[0] if builds else None

    def get_stat_allocation(self, build_name: str, level: int) -> StatAllocation | None:
        build = self.get_build(build_name)
        if not build:
            return None
        for alloc in build.stat_progression:
            if alloc.level_min <= level <= alloc.level_max:
                return alloc
        return None

    def get_skills_to_learn(self, build_name: str, current_level: int) -> list[SkillLearnOrder]:
        build = self.get_build(build_name)
        if not build:
            return []
        return [s for s in build.skill_order if s.required_level <= current_level]

    def get_equipment_targets(self, build_name: str, current_level: int) -> list[EquipmentGoal]:
        build = self.get_build(build_name)
        if not build:
            return []
        return [e for e in build.equipment_goals if e.level_requirement <= current_level]

    def get_buff_priority(self, build_name: str) -> list[str]:
        build = self.get_build(build_name)
        return list(build.buff_priority) if build else []

    def get_rotation_name(self, build_name: str) -> str:
        build = self.get_build(build_name)
        return build.rotation_name if build else ""

    def get_build_summary(self, build_name: str) -> str:
        build = self.get_build(build_name)
        if not build:
            return f"Build '{build_name}' not found."
        lines = [
            f"-- {build.display_name} --",
            f"  {build.description}",
            f"  Primary: {', '.join(build.primary_stats)}",
            f"  Secondary: {', '.join(build.secondary_stats)}",
            f"  Rotation: {build.rotation_name}",
            f"  Buff priority: {', '.join(build.buff_priority)}",
            f"  Skills: {len(build.skill_order)}",
            f"  Equipment goals: {len(build.equipment_goals)}",
        ]
        return "\n".join(lines)


_build_manager: BuildManager | None = None
_build_manager_lock = RLock()


def get_build_manager() -> BuildManager:
    global _build_manager
    with _build_manager_lock:
        if _build_manager is None:
            _build_manager = BuildManager()
        return _build_manager
# Elements: BuildManager
# Edit code below the markers, then merge explicitly.
# ============================================================

# WARNING: element not found: BuildManager
