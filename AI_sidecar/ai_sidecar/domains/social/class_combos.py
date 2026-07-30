"""Class Combo Definitions — specific class-vs-class synergy combos for RO.

A pro party doesn't just stand together — it chains skills for multiplicative damage:
- Sage + Wizard: Endow element matching monster weakness, Deluge + Storm Gust freeze
- Priest + Hunter: Assumptio for 50% damage reduction, Lex Aeterna for 2x damage
- Dancer + Bard: AoE stun (Hypnotist's Waltz), party EXP bonus (Experience Increase)
- Alchemist + anyone: Acid Demonstration bypasses defense, counters Paladins in WoE
- Monk + Priest: Lex Aeterna → Asura Strike = instakill
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class ComboCategory(StrEnum):
    """Category of class combo."""
    ELEMENTAL = "elemental"          # Element enchant + matching damage
    DEFENSIVE = "defensive"          # Damage reduction, shields
    OFFENSIVE = "offensive"          # Damage amplification
    UTILITY = "utility"              # Buffs, debuffs, mobility
    CC = "cc"                        # Crowd control chains
    INSTAKILL = "instakill"          # One-shot combos
    WOE = "woe"                      # War of Emperium specific
    EXP = "exp"                      # EXP/grinding efficiency


@dataclass
class ClassCombo:
    """A class-vs-class synergy combo definition."""
    name: str
    category: ComboCategory
    prep_class: str
    main_class: str
    prep_skill: str
    main_skill: str
    prep_time_s: float
    window_s: float
    range: int
    description: str
    requirements: dict[str, Any] = field(default_factory=dict)
    woe_only: bool = False
    min_level: int = 1
    target_required: bool = True
    latency_buffer: float = 0.5


# ── All class combos ──────────────────────────────────────────────────────

CLASS_COMBOS: list[ClassCombo] = [
    # ═════════════════════════════════════════════════════════════════════
    # Sage + Wizard combos
    # ═════════════════════════════════════════════════════════════════════
    ClassCombo(
        name="Endow → Elemental Nuke",
        category=ComboCategory.ELEMENTAL,
        prep_class="sage",
        main_class="wizard",
        prep_skill="endow_element",
        main_skill="elemental_bolt",
        prep_time_s=2.0,
        window_s=30.0,
        range=5,
        description="Sage enchants Wizard's weapon with element matching monster weakness, Wizard nukes for 1.5x+ damage",
        requirements={"element": "match_monster_weakness"},
    ),
    ClassCombo(
        name="Deluge → Storm Gust Freeze",
        category=ComboCategory.CC,
        prep_class="sage",
        main_class="wizard",
        prep_skill="deluge",
        main_skill="storm_gust",
        prep_time_s=1.5,
        window_s=5.0,
        range=7,
        description="Sage casts Deluge (water field), Wizard casts Storm Gust on it — 100% freeze rate on all targets in AoE",
        requirements={"element": "water", "aoe": True},
    ),
    ClassCombo(
        name="Volcano → Meteor Storm",
        category=ComboCategory.ELEMENTAL,
        prep_class="sage",
        main_class="wizard",
        prep_skill="volcano",
        main_skill="meteor_storm",
        prep_time_s=2.0,
        window_s=10.0,
        range=7,
        description="Sage casts Volcano (fire field), Wizard casts Meteor Storm — +50% fire damage in AoE",
        requirements={"element": "fire", "aoe": True},
    ),
    ClassCombo(
        name="Frost → Frost Nova Chain",
        category=ComboCategory.CC,
        prep_class="sage",
        main_class="wizard",
        prep_skill="frost",
        main_skill="frost_nova",
        prep_time_s=1.0,
        window_s=4.0,
        range=7,
        description="Sage casts Frost (water field), Wizard Frost Novas — extended freeze duration",
        requirements={"element": "water"},
    ),

    # ═════════════════════════════════════════════════════════════════════
    # Priest + Hunter combos
    # ═════════════════════════════════════════════════════════════════════
    ClassCombo(
        name="Assumptio → Aggressive Position",
        category=ComboCategory.DEFENSIVE,
        prep_class="priest",
        main_class="hunter",
        prep_skill="assumptio",
        main_skill="ranged_attack",
        prep_time_s=2.0,
        window_s=30.0,
        range=9,
        description="Priest casts Assumptio on Hunter (50% damage reduction) — Hunter can fight in dangerous positions",
        target_required=False,
    ),
    ClassCombo(
        name="Lex Aeterna → Sharp Shooting",
        category=ComboCategory.OFFENSIVE,
        prep_class="priest",
        main_class="hunter",
        prep_skill="lex_aeterna",
        main_skill="sharp_shooting",
        prep_time_s=1.5,
        window_s=3.0,
        range=5,
        description="Priest marks target with Lex Aeterna (2x damage), Hunter delivers with Sharp Shooting",
    ),
    ClassCombo(
        name="Lex Aeterna → Double Strafe",
        category=ComboCategory.OFFENSIVE,
        prep_class="priest",
        main_class="hunter",
        prep_skill="lex_aeterna",
        main_skill="double_strafe",
        prep_time_s=1.5,
        window_s=3.0,
        range=5,
        description="Priest marks target with Lex Aeterna (2x damage), Hunter delivers with Double Strafe",
    ),
    ClassCombo(
        name="Safety Wall → Sniper Position",
        category=ComboCategory.DEFENSIVE,
        prep_class="priest",
        main_class="hunter",
        prep_skill="safety_wall",
        main_skill="ranged_attack",
        prep_time_s=1.0,
        window_s=10.0,
        range=5,
        description="Priest drops Safety Wall under Hunter — 10 hits of melee immunity while sniping",
        target_required=False,
    ),

    # ═════════════════════════════════════════════════════════════════════
    # Dancer + Bard combos
    # ═════════════════════════════════════════════════════════════════════
    ClassCombo(
        name="Hypnotist's Waltz → Encore Stun",
        category=ComboCategory.CC,
        prep_class="dancer",
        main_class="bard",
        prep_skill="hypnotist_waltz",
        main_skill="encores",
        prep_time_s=3.0,
        window_s=8.0,
        range=7,
        description="Dancer starts Hypnotist's Waltz (AoE sleep), Bard follows with Encore — extended CC chain",
        requirements={"aoe": True},
    ),
    ClassCombo(
        name="Experience Increase → Party EXP",
        category=ComboCategory.EXP,
        prep_class="bard",
        main_class="dancer",
        prep_skill="experience_increase",
        main_skill="experience_increase",
        prep_time_s=3.0,
        window_s=300.0,
        range=9,
        description="Bard and Dancer both play Experience Increase — combined gives +50% party EXP bonus",
        target_required=False,
    ),
    ClassCombo(
        name="Lullaby → AoE Sleep",
        category=ComboCategory.CC,
        prep_class="dancer",
        main_class="bard",
        prep_skill="lullaby",
        main_skill="lullaby",
        prep_time_s=3.0,
        window_s=10.0,
        range=7,
        description="Dancer and Bard both play Lullaby — extended AoE sleep duration on all enemies",
        requirements={"aoe": True},
        target_required=False,
    ),
    ClassCombo(
        name="Assassin Cross of Sunset → Mental Sensing",
        category=ComboCategory.UTILITY,
        prep_class="bard",
        main_class="dancer",
        prep_skill="assassin_cross",
        main_skill="mental_sensing",
        prep_time_s=3.0,
        window_s=120.0,
        range=9,
        description="Bard and Dancer combo song — reveals hidden enemies in AoE",
        requirements={"aoe": True},
        target_required=False,
    ),

    # ═════════════════════════════════════════════════════════════════════
    # Alchemist combos
    # ═════════════════════════════════════════════════════════════════════
    ClassCombo(
        name="Acid Demonstration → Focus Fire",
        category=ComboCategory.WOE,
        prep_class="alchemist",
        main_class="any",
        prep_skill="acid_demonstration",
        main_skill="attack",
        prep_time_s=2.0,
        window_s=5.0,
        range=5,
        description="Alchemist uses Acid Demonstration (bypasses defense, hits hard on Paladins), party focuses target",
        requirements={"woe_target": "paladin"},
        woe_only=True,
    ),
    ClassCombo(
        name="Full Chemical Protection → Push",
        category=ComboCategory.DEFENSIVE,
        prep_class="alchemist",
        main_class="any",
        prep_skill="full_chemical_protection",
        main_skill="attack",
        prep_time_s=3.0,
        window_s=300.0,
        range=5,
        description="Alchemist gives Full Chemical Protection (immune to strip/break) — party pushes safely",
        target_required=False,
    ),
    ClassCombo(
        name="Acid Bomb → Elemental Nuke",
        category=ComboCategory.OFFENSIVE,
        prep_class="alchemist",
        main_class="wizard",
        prep_skill="acid_bomb",
        main_skill="elemental_bolt",
        prep_time_s=1.5,
        window_s=3.0,
        range=5,
        description="Alchemist throws Acid Bomb (lowers DEF), Wizard follows with elemental nuke for max damage",
    ),

    # ═════════════════════════════════════════════════════════════════════
    # Monk + Priest combos (instakill)
    # ═════════════════════════════════════════════════════════════════════
    ClassCombo(
        name="Lex Aeterna → Asura Strike",
        category=ComboCategory.INSTAKILL,
        prep_class="priest",
        main_class="monk",
        prep_skill="lex_aeterna",
        main_skill="asura_strike",
        prep_time_s=1.5,
        window_s=3.0,
        range=5,
        description="Priest casts Lex Aeterna (2x damage), Monk follows with Asura Strike — instakill any non-boss",
        requirements={"sp_cost": "high"},
    ),
    ClassCombo(
        name="Gloria → Asura Strike Crit",
        category=ComboCategory.OFFENSIVE,
        prep_class="priest",
        main_class="monk",
        prep_skill="gloria",
        main_skill="asura_strike",
        prep_time_s=2.0,
        window_s=120.0,
        range=9,
        description="Priest casts Gloria (+20% crit), Monk Asura Strike benefits from crit rate",
        target_required=False,
    ),
    ClassCombo(
        name="Impositio Manus → Asura Strike",
        category=ComboCategory.OFFENSIVE,
        prep_class="priest",
        main_class="monk",
        prep_skill="impositio_manus",
        main_skill="asura_strike",
        prep_time_s=1.0,
        window_s=60.0,
        range=5,
        description="Priest buffs Monk's INT with Impositio Manus, Monk delivers boosted Asura Strike",
        target_required=False,
    ),

    # ═════════════════════════════════════════════════════════════════════
    # Wizard + Knight combos
    # ═════════════════════════════════════════════════════════════════════
    ClassCombo(
        name="Storm Gust → Bowling Bash Push",
        category=ComboCategory.CC,
        prep_class="wizard",
        main_class="knight",
        prep_skill="storm_gust",
        main_skill="bowling_bash",
        prep_time_s=4.0,
        window_s=3.0,
        range=5,
        description="Wizard freezes/slows mobs in Storm Gust AoE, Knight Bowling Bash pushes them back through the storm",
        requirements={"aoe": True},
    ),
    ClassCombo(
        name="Fire Wall → Spear Stab",
        category=ComboCategory.CC,
        prep_class="wizard",
        main_class="knight",
        prep_skill="fire_wall",
        main_skill="spear_stab",
        prep_time_s=2.0,
        window_s=8.0,
        range=5,
        description="Wizard places Fire Wall, Knight pushes enemies into it with Spear Stab",
    ),

    # ═════════════════════════════════════════════════════════════════════
    # Assassin + Priest combos
    # ═════════════════════════════════════════════════════════════════════
    ClassCombo(
        name="Aspersio → Soul Destroyer",
        category=ComboCategory.ELEMENTAL,
        prep_class="priest",
        main_class="assassin",
        prep_skill="aspersio",
        main_skill="soul_destroyer",
        prep_time_s=2.0,
        window_s=30.0,
        range=5,
        description="Priest blesses Assassin's weapon with holy element, Assassin uses Soul Destroyer for massive holy damage",
    ),
    ClassCombo(
        name="Kyrie Eleison → Cloaking Assault",
        category=ComboCategory.DEFENSIVE,
        prep_class="priest",
        main_class="assassin",
        prep_skill="kyrie_eleison",
        main_skill="cloaking",
        prep_time_s=2.0,
        window_s=300.0,
        range=9,
        description="Priest casts Kyrie Eleison (reflects 30% damage), Assassin cloaks and backstabs safely",
        target_required=False,
    ),

    # ═════════════════════════════════════════════════════════════════════
    # Blacksmith + Party combos
    # ═════════════════════════════════════════════════════════════════════
    ClassCombo(
        name="Weapon Perfection → Party DPS",
        category=ComboCategory.UTILITY,
        prep_class="blacksmith",
        main_class="any",
        prep_skill="weapon_perfection",
        main_skill="attack",
        prep_time_s=2.0,
        window_s=300.0,
        range=9,
        description="Blacksmith casts Weapon Perfection (ignores size penalty) — all party members deal full damage",
        target_required=False,
    ),
    ClassCombo(
        name="Over Thrust → Burst DPS",
        category=ComboCategory.OFFENSIVE,
        prep_class="blacksmith",
        main_class="any",
        prep_skill="over_thrust",
        main_skill="attack",
        prep_time_s=2.0,
        window_s=120.0,
        range=9,
        description="Blacksmith casts Over Thrust (+25% ATK) — party burst damage window",
        target_required=False,
    ),

    # ═════════════════════════════════════════════════════════════════════
    # Soul Linker combos
    # ═════════════════════════════════════════════════════════════════════
    ClassCombo(
        name="Soul Link → Class Mastery",
        category=ComboCategory.UTILITY,
        prep_class="soul_linker",
        main_class="any",
        prep_skill="soul_link",
        main_skill="attack",
        prep_time_s=3.0,
        window_s=300.0,
        range=5,
        description="Soul Linker links with a class — grants special bonuses (e.g., Assassin gets +30% ASPD)",
        target_required=False,
    ),
    ClassCombo(
        name="Kaahi → Sustain Push",
        category=ComboCategory.DEFENSIVE,
        prep_class="soul_linker",
        main_class="any",
        prep_skill="kaahi",
        main_skill="attack",
        prep_time_s=2.0,
        window_s=60.0,
        range=5,
        description="Soul Linker casts Kaahi (auto-heal on attack) — party sustain during extended fights",
        target_required=False,
    ),

    # ═════════════════════════════════════════════════════════════════════
    # Rogue + Party combos
    # ═════════════════════════════════════════════════════════════════════
    ClassCombo(
        name="Strip Weapon → Focus Fire",
        category=ComboCategory.WOE,
        prep_class="rogue",
        main_class="any",
        prep_skill="strip_weapon",
        main_skill="attack",
        prep_time_s=2.0,
        window_s=10.0,
        range=5,
        description="Rogue strips enemy weapon (disables attacks), party focuses the defenseless target",
        woe_only=True,
    ),
    ClassCombo(
        name="Strip Armor → Magic Nuke",
        category=ComboCategory.WOE,
        prep_class="rogue",
        main_class="wizard",
        prep_skill="strip_armor",
        main_skill="soul_strike",
        prep_time_s=2.0,
        window_s=10.0,
        range=5,
        description="Rogue strips enemy armor (-50% DEF), Wizard follows with magic nuke (ignores remaining DEF)",
        woe_only=True,
    ),
]


# ── Combo lookup helpers ──────────────────────────────────────────────────

def get_combos_for_classes(prep_class: str, main_class: str) -> list[ClassCombo]:
    """Get all combos that work between two classes."""
    prep_lower = prep_class.lower()
    main_lower = main_class.lower()
    return [
        c for c in CLASS_COMBOS
        if c.prep_class == prep_lower and c.main_class == main_lower
    ]


def get_combos_for_prep_class(prep_class: str) -> list[ClassCombo]:
    """Get all combos where a class is the prep caster."""
    return [c for c in CLASS_COMBOS if c.prep_class == prep_class.lower()]


def get_combos_for_main_class(main_class: str) -> list[ClassCombo]:
    """Get all combos where a class is the main caster."""
    return [c for c in CLASS_COMBOS if c.main_class == main_class.lower()]


def get_combos_by_category(category: ComboCategory) -> list[ClassCombo]:
    """Get all combos in a specific category."""
    return [c for c in CLASS_COMBOS if c.category == category]


def get_woe_combos() -> list[ClassCombo]:
    """Get all WoE-specific combos."""
    return [c for c in CLASS_COMBOS if c.woe_only]


def get_instakill_combos() -> list[ClassCombo]:
    """Get all instakill combos."""
    return [c for c in CLASS_COMBOS if c.category == ComboCategory.INSTAKILL]


def find_combo_by_name(name: str) -> ClassCombo | None:
    """Find a combo by its exact name."""
    for c in CLASS_COMBOS:
        if c.name.lower() == name.lower():
            return c
    return None


def get_class_vs_class_counter(attacker_class: str, defender_class: str) -> str | None:
    """Get the counter strategy for attacker vs defender in WoE.

    Returns a tactic string or None if no specific counter exists.
    """
    counters: dict[str, dict[str, str]] = {
        "alchemist": {
            "paladin": "Acid Demonstration bypasses Paladin's high DEF and Guard skills",
            "crusader": "Acid Demonstration ignores defense — focus fire",
        },
        "wizard": {
            "champion": "Storm Gust freezes Champion before Asura Strike",
            "monk": "Frost Diver stops Monk charge",
            "paladin": "Heaven's Drive deals neutral damage through Guard",
        },
        "assassin": {
            "wizard": "Cloak + Backstab — Wizards can't see cloaked Assassins",
            "priest": "Soul Destroyer interrupts heal cast",
        },
        "hunter": {
            "wizard": "Ankle Snare stops Wizard movement, ranged attack from safety",
            "priest": "Sharp Shooting from outside dispel range",
        },
        "priest": {
            "assassin": "Kyrie Eleison reflects backstab damage",
            "champion": "Lex Aeterna + turn undead if applicable",
        },
        "paladin": {
            "assassin": "Shield Reflect counters backstab",
            "hunter": "Defending Aura reduces ranged damage",
        },
        "champion": {
            "paladin": "Asura Strike bypasses Guard (neutral property)",
            "wizard": "Mental Strength + charge through AoE",
        },
    }
    return counters.get(attacker_class.lower(), {}).get(defender_class.lower())
