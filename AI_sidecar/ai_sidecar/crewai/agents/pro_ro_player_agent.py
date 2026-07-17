"""Pro RO Player — 20-year Ragnarok Online veteran providing expert tactical/strategic advice."""

from __future__ import annotations

from typing import Any

from .base_agent import BehaviorProfile

# ── RO knowledge databases ───────────────────────────────────────────────────

# Elemental wheel: attacker_element -> { target_element: multiplier }
# Standard RO elemental modifiers (100% = 1.0)
ELEMENTAL_ADVANTAGE: dict[str, dict[str, float]] = {
    "neutral": {"neutral": 1.0, "water": 1.0, "earth": 1.0, "fire": 1.0, "wind": 1.0, "poison": 1.0, "holy": 1.0, "dark": 1.0, "ghost": 0.75, "undead": 1.0},
    "water": {"neutral": 1.0, "water": 0.25, "earth": 0.75, "fire": 1.5, "wind": 0.75, "poison": 1.0, "holy": 1.0, "dark": 1.0, "ghost": 0.75, "undead": 1.0},
    "earth": {"neutral": 1.0, "water": 1.5, "earth": 0.25, "fire": 0.75, "wind": 1.5, "poison": 1.0, "holy": 1.0, "dark": 1.0, "ghost": 0.75, "undead": 1.0},
    "fire": {"neutral": 1.0, "water": 0.75, "earth": 1.5, "fire": 0.25, "wind": 1.5, "poison": 1.0, "holy": 1.0, "dark": 1.0, "ghost": 0.75, "undead": 1.0},
    "wind": {"neutral": 1.0, "water": 1.5, "earth": 0.75, "fire": 0.75, "wind": 0.25, "poison": 1.0, "holy": 1.0, "dark": 1.0, "ghost": 0.75, "undead": 1.0},
    "poison": {"neutral": 1.0, "water": 1.0, "earth": 0.75, "fire": 1.0, "wind": 1.0, "poison": 0.25, "holy": 0.75, "dark": 0.75, "ghost": 0.75, "undead": 1.0},
    "holy": {"neutral": 1.0, "water": 1.0, "earth": 1.0, "fire": 1.0, "wind": 1.0, "poison": 1.0, "holy": 0.25, "dark": 1.75, "ghost": 1.0, "undead": 1.75},
    "dark": {"neutral": 1.0, "water": 1.0, "earth": 1.0, "fire": 1.0, "wind": 1.0, "poison": 1.0, "holy": 0.25, "dark": 0.25, "ghost": 1.0, "undead": 1.0},
    "ghost": {"neutral": 0.75, "water": 1.0, "earth": 1.0, "fire": 1.0, "wind": 1.0, "poison": 1.0, "holy": 1.0, "dark": 1.0, "ghost": 1.5, "undead": 1.0},
    "undead": {"neutral": 1.0, "water": 1.0, "earth": 1.0, "fire": 1.25, "wind": 1.0, "poison": 1.0, "holy": 1.5, "dark": 0.5, "ghost": 1.0, "undead": 0.25},
}

# Monster size penalties by weapon type
SIZE_PENALTY: dict[str, dict[str, float]] = {
    "dagger": {"small": 1.0, "medium": 0.75, "large": 0.5},
    "sword": {"small": 0.75, "medium": 1.0, "large": 0.75},
    "spear": {"small": 0.75, "medium": 0.75, "large": 1.0},
    "axe": {"small": 0.75, "medium": 1.0, "large": 0.75},
    "mace": {"small": 0.75, "medium": 1.0, "large": 0.75},
    "bow": {"small": 1.0, "medium": 1.0, "large": 0.75},
    "staff": {"small": 1.0, "medium": 1.0, "large": 1.0},
    "knuckle": {"small": 1.0, "medium": 0.75, "large": 0.5},
    "instrument": {"small": 0.75, "medium": 1.0, "large": 0.75},
    "whip": {"small": 0.75, "medium": 1.0, "large": 0.75},
    "book": {"small": 0.75, "medium": 1.0, "large": 0.75},
    "katar": {"small": 1.0, "medium": 1.0, "large": 0.75},
    "claw": {"small": 1.0, "medium": 1.0, "large": 0.75},
    "two_handed_sword": {"small": 0.75, "medium": 0.75, "large": 1.0},
    "two_handed_spear": {"small": 0.75, "medium": 0.75, "large": 1.0},
    "two_handed_axe": {"small": 0.75, "medium": 0.75, "large": 1.0},
}

# Race-based behavioral hints
RACE_HINTS: dict[str, str] = {
    "brute": "Brute monsters are melee-oriented with high HP. Kite them. Weak to Fire and Wind.",
    "demihuman": "Demihumans often drop valuable loot and have balanced stats. Watch for multi-aggro pulls.",
    "demon": "Demons are typically Dark or Fire element. Holy property skills deal massive bonus damage.",
    "dragon": "Dragons are large-size, high-HP monsters. Spear users get full damage. Bring elemental converters.",
    "fish": "Fish monsters are Water element, weak to Wind. Usually slow-moving.",
    "formless": "Formless monsters resist Neutral. Use elemental converters. Often found in dungeons.",
    "angel": "Angels are Holy element. Dark property attacks work best. High flee, use magic.",
    "insect": "Insects are often Poison or Earth element. Fire works well. Check size for weapon penalty.",
    "plant": "Plants are usually Earth element. Fire skills deal bonus damage. Low HP but high DEF.",
    "undead": "Undead monsters are Dark or Undead element. Holy Water and Heal skill damage them! Undead element is weak to Fire and Holy.",
}

# ── Class-specific early game advice ─────────────────────────────────────────

CLASS_EARLY_GAME: dict[str, dict[str, Any]] = {
    "novice": {
        "first_map": "prt_fild00",
        "level_range": (1, 15),
        "advice": "As a Novice, train on Porings, Lunatics, and Fabres in prt_fild00 (south of Prontera). Pump DEX to 20 first for hit rate, then STR. Do the novice training ground quest for free gear.",
        "stats": "STR 20 > DEX 20 > rest STR",
        "equipment": "Sword[3] + 3x Fabre Card, Cotton Shirt, Sandals",
    },
    "swordman": {
        "first_map": "payon",
        "level_range": (1, 15),
        "advice": "Hunt Pecopeco and Hornets on Payon fields. These give great job exp. Pump STR to 40 first, then VIT to 30 before touching DEX.",
        "stats": "STR 40 > VIT 30 > DEX 20 > rest STR",
        "equipment": "Blade[3] + 3x Pecopeco Card, Cotton Shirt[1] + Peco Card, Shoes[1] + Matyr Card",
    },
    "mage": {
        "first_map": "gef_fild04",
        "level_range": (1, 15),
        "advice": "Train on Rockers and Spores near Geffen. Rockers are Earth 1 — Fire Bolt one-shots them. Pump INT to 40, then DEX to 30. Always carry Blue Potions.",
        "stats": "INT 40 > DEX 30 > rest INT",
        "equipment": "Rod[4] + 4x Pecopeco Card, Cotton Shirt, Shoes[1] + Matyr Card",
    },
    "archer": {
        "first_map": "payon",
        "level_range": (1, 15),
        "advice": "Go to Payon Cave 1-2 until job level 10. Skeletons and Zombies are slow, easy to kite. DEX is everything — pump it to 50 before touching AGI.",
        "stats": "DEX 50 > AGI 30 > rest DEX/LUK",
        "equipment": "Bow[3] + 3x Pecopeco Card (double-straight), Cotton Shirt, Boots[1] + Matyr Card",
    },
    "acolyte": {
        "first_map": "izlude",
        "level_range": (1, 15),
        "advice": "Train on Willow and familiar near Izlude. Use Heal offensively against Undead in Payon Cave for instant kills. Pump INT first for Heal power, then DEX for cast time.",
        "stats": "INT 40 > DEX 30 > rest INT/STR",
        "equipment": "Mace[3] + 3x Pecopeco Card, Cotton Shirt, Shoes[1] + Matyr Card",
    },
    "merchant": {
        "first_map": "morocc",
        "level_range": (1, 15),
        "advice": "Merchants are tanky with high HP. Hunt Pecopeco and Savage on the plains near Morocc. Use Overcharge to maximize zeny from NPC vend. Pump STR first, then VIT.",
        "stats": "STR 40 > VIT 30 > DEX 20 > rest STR",
        "equipment": "Axe[3] + 3x Pecopeco Card, Cotton Shirt[1] + Peco Card, Boots[1] + Matyr Card",
    },
    "thief": {
        "first_map": "payon",
        "level_range": (1, 15),
        "advice": "Hunt Poison Spores and Wolves near Payon. AGI builds shine early. Max Double Attack ASAP. Pump AGI to 40 first, then DEX to 30 for hit rate.",
        "stats": "AGI 40 > DEX 30 > rest STR",
        "equipment": "Dagger[3] + 3x Pecopeco Card (for ASPD), Cotton Shirt, Boots[1] + Matyr Card",
    },
    "taekwon": {
        "first_map": "izlude",
        "level_range": (1, 15),
        "advice": "Train on Muka and Yoyo near Izlude. Taekwon kicks scale with STR. Pump STR and AGI evenly. Use your movement speed to position behind monsters for Kick damage bonus.",
        "stats": "STR 30 > AGI 30 > rest DEX",
        "equipment": "Shoes, Cotton Shirt, no weapon (kick damage)",
    },
    "gunslinger": {
        "first_map": "einbroch",
        "level_range": (1, 15),
        "advice": "Hunt Poring and Poporing near Einbroch fields. Use Single Action for burst. DEX is king. Bullets cost money — track your zeny carefully.",
        "stats": "DEX 50 > AGI 30 > rest DEX/LUK",
        "equipment": "Six Shooter + appropriate bullets, Leather Jacket, Shoes",
    },
    "ninja": {
        "first_map": "amatsu",
        "level_range": (1, 15),
        "advice": "Train on Muka and Savage in Amatsu fields. Use Kunai to tag mobs at range. Pump INT for magic or STR for physical build.",
        "stats": "INT 40 > DEX 30 (magic) or STR 40 > AGI 30 (physical)",
        "equipment": "Ninja Suit, Ninja Scroll, appropriate elemental Kunai",
    },
    "soul_linker": {
        "first_map": "lighthalzen",
        "level_range": (1, 15),
        "advice": "Estrun and Luciola Vespa near Lighthalzen. Use Soul Strike and spirit orbs. Pump INT then DEX. Best exp via party play — link allies for shared benefits.",
        "stats": "INT 50 > DEX 30 > rest INT",
        "equipment": "Rod[4] + 4x Int/Wis bonuses, Robe, Shoes",
    },
}

# ── Hunting ground recommendations by level range ────────────────────────────

HUNTING_GROUNDS: dict[str, dict[str, Any]] = {
    "payon_cave_1f": {"level": (15, 30), "danger": "low", "description": "Payon Cave 1F — Skeletons, Zombies, Familiar. Slow undead, easy to kite. Bring Holy Water for efficiency."},
    "payon_cave_2f": {"level": (25, 40), "danger": "low", "description": "Payon Cave 2F — Ghoul, Munak, Bongun. Watch for Munak's multi-hit attack. Dark property, use holy skills."},
    "payon_cave_3f": {"level": (35, 55), "danger": "medium", "description": "Payon Cave 3F — Mummy, Verit, Arclouze. Mummies have high DEF, Verit has high flee. Arclouze cast Stone Curse."},
    "gef_fild01": {"level": (10, 25), "danger": "low", "description": "Geffen Field 1 — Spore, Rocker, Peco Peco. Great for mages — Fire Bolt one-shots Spores and Rockers (Earth)."},
    "gef_fild04": {"level": (15, 30), "danger": "low", "description": "Geffen Field 4 — Hornet, Spore, Lunatic, Poring. Mixed spawn, good variety. Watch for Hornet's poison."},
    "moc_fild01": {"level": (20, 35), "danger": "low", "description": "Morocc Field 1 — Savage, Pecopeco, Andre. Savages hit hard for their level — keep HP up."},
    "moc_fild02": {"level": (10, 25), "danger": "low", "description": "Morocc Field 2 — Desert Wolf, Vitata, Poring, Fabre. Good early leveling for melee classes."},
    "mjolnir_01": {"level": (10, 20), "danger": "low", "description": "Mjolnir 1 — Fabre, Peco Peco, Condor. Safe training for level 10-20."},
    "mjolnir_02": {"level": (15, 25), "danger": "low", "description": "Mjolnir 2 — Spore, Kukre, Thief Bug. Kukres are Water 1 — use Wind or Earth attacks."},
    "mjolnir_03": {"level": (20, 30), "danger": "low", "description": "Mjolnir 3 — Kukre, Wolf, Argiope. Watch for Argioe's poison. Good medium-density map."},
    "anthell01": {"level": (30, 50), "danger": "medium", "description": "Anthell 1 — Andre, Hornet, Vitata, Deniro. Insect dungeon. Fire elemental converters destroy here."},
    "anthell02": {"level": (40, 60), "danger": "medium", "description": "Anthell 2 — Spider Chitin, Giant Hornet. Dense spawns — be careful of multi-aggro."},
    "ein_fild01": {"level": (35, 55), "danger": "medium", "description": "Einbroch Field 1 — Baphomet Jr, Rideword, Alarm. Bapho Jr is a common MVP target — good medium-level training."},
    "ein_fild02": {"level": (45, 65), "danger": "medium", "description": "Einbroch Field 2 — Nightmare, Incubus, Succubus. Dark property, bring Holy Water."},
    "mag_dun01": {"level": (50, 70), "danger": "medium", "description": "Magma Dungeon 1 — Kaho, Lava Golem, Salamander. Fire element — bring Ice/Fire armor. Watch out for Meteor Storm from Kaho."},
    "mag_dun02": {"level": (60, 85), "danger": "high", "description": "Magma Dungeon 2 — Nightmare Terror, Sky Petite, Kaho. High density. Extremely dangerous for Water-element armor users."},
    "cave_gef01": {"level": (40, 60), "danger": "medium", "description": "Geffen Cave 1 — Flora, Marionette, Mineral. Marionettes cast Mind Breaker (reduces INT to 1). Bring Green Potions."},
    "cave_gef02": {"level": (50, 70), "danger": "high", "description": "Geffen Cave 2 — Arclouze, Wind Ghost, Medusa. Medusa Stone Curses — bring Stone Curse immunity gear."},
    "glast_heim01": {"level": (55, 75), "danger": "high", "description": "Glast Heim 1 — Raydric, Wraith, Wraith Dead. Undead + Demon mixed. Watch for Raydric's Parrying. Dark property resist gear recommended."},
    "glast_heim02": {"level": (65, 90), "danger": "high", "description": "Glast Heim 2 — Bloody Murderer, Wraith Dead, Banshee. Heavy aggro area. Multi-aggro is deadly — use teleport escape."},
    "gon_dun01": {"level": (55, 75), "danger": "medium", "description": "Gonryun Dungeon 1 — Nine Tail, Sohee, Mi Gao. Sohee drops Ice Pick (rare). Nine Tail uses Fire elemental attacks."},
    "gon_dun02": {"level": (65, 85), "danger": "high", "description": "Gonryun Dungeon 2 — Zipper Bear, Dark Priest. High DEF monsters — bring elemental converters."},
    "beach_dun01": {"level": (35, 55), "danger": "medium", "description": "Beach Dungeon 1 — Vadon, Mermaid, Marc. Water element enemies — use Wind attacks. Marc drops accessory cards."},
    "beach_dun02": {"level": (45, 65), "danger": "medium", "description": "Beach Dungeon 2 — Strouf, Merman, Kukre. High HP fish monsters. Good for hit-lock melee builds."},
    "ice_dun01": {"level": (60, 80), "danger": "high", "description": "Ice Dungeon 1 — Snowier, Gazeti, Ice Titan. Resist Sleep gear needed. Earth attacks recommended."},
    "ice_dun02": {"level": (70, 95), "danger": "high", "description": "Ice Dungeon 2 — Freezer, Hatii Snowier, Ice Elemental. Bring Fire element armor. Level gap penalty applies heavily here."},
    "thor_v01": {"level": (75, 99), "danger": "very_high", "description": "Thor Volcano 1 — Obsidian, Injustice, Hellion. Heavy spawn, high damage. Full Meteor Storm field. GTFO positioning required."},
    "thor_v02": {"level": (85, 99), "danger": "very_high", "description": "Thor Volcano 2 — Ifrit MVP farm route. Only for well-geared 90+. Not recommended for botting without GTFO gear."},
    "abyss_01": {"level": (80, 99), "danger": "very_high", "description": "Abyss Lake 1 — Hydra, Strouf, Swordfish. MVP: Kraken. High HP fish. Bring Wind arrows/converters."},
    "abyss_02": {"level": (85, 99), "danger": "very_high", "description": "Abyss Lake 2 — Knight of Abyss, Abyss Chaser, Sea Lord. Endgame farming. Full party recommended."},
}

# ── Common death causes and mitigations ──────────────────────────────────────

DEATH_CAUSES: dict[str, dict[str, Any]] = {
    "aoe_spell": {
        "symptom": "Killed by magic AoE (Storm Gust, Meteor Storm, Lord of Vermilion, etc.)",
        "counter": "Increase MDEF with accessory cards (Angeling, Peco Peco Egg, Marc). Watch for casters and aggro range. Use Safety Wall or Ground magic as cover.",
        "gear_solution": "Angeling Card in headgear for Holy armor, Marc Card in armor for Freeze immunity.",
    },
    "multi_aggro": {
        "symptom": "Swarmed by 3+ monsters simultaneously",
        "counter": "Reduce aggro range settings. Use Teleport or Fly Wing at first sight of second monster. Equip aggro-reducing gear (Hood[1] + Raydric Card).",
        "gear_solution": "Raydric Card in shield (30% Neutral reduction), Thief Clothes[1] with Whisper Card.",
    },
    "elemental_disadvantage": {
        "symptom": "Took heavily increased damage from an element you're weak to",
        "counter": "Check monster element vs your armor element. Swap armor element for the area (e.g., Fire armor for Magma, Water armor for Ice, Wind armor for Beach). Use elemental resist potions.",
        "gear_solution": "Always carry a set of elemental armor (Fire, Water, Wind, Earth). Switch based on hunting ground.",
    },
    "level_gap_penalty": {
        "symptom": "Character is lower level than monsters, suffering damage penalty",
        "counter": "If monster is 15+ levels higher than you, the damage bonus they receive is extreme. Move to a lower-level hunting ground that's within your level range.",
        "gear_solution": "No gear solution — purely level-dependent. Move maps until level gap is <= 10.",
    },
    "stun_lock": {
        "symptom": "Killed while stunned, unable to move or act",
        "counter": "Stack VIT for stun resistance (50 VIT = 100% stun resist). Bring Green Potions set to auto-use. Vitata Card in headgear for stun immunity.",
        "gear_solution": "Vitata Card, Orc Hero Card (endgame). Green Potions auto-consumption.",
    },
    "poison": {
        "symptom": "Died to poison damage over time",
        "counter": "Use Poison Resist potions. Equip Poison element armor. Use Green Potions/Saylette for cure. Argiope Card in armor for Poison element.",
        "gear_solution": "Argiope Card, Green Potions set to 50% HP auto-use.",
    },
    "freeze": {
        "symptom": "Died while frozen, unable to move",
        "counter": "Marc Card in armor for complete freeze immunity. Otherwise carry Hwergelmir's Tonic. Watch for Storm Gust and Cold Bolt casters.",
        "gear_solution": "Marc Card (mandatory for Ice Dungeon).",
    },
    "stone_curse": {
        "symptom": "Turned to stone and killed",
        "counter": "Medusa and Arclouze cast Stone Curse. Use anti-Stone Curse gear (Medusa Card Shield). Carry and auto-use Stone Curse cure items.",
        "gear_solution": "Medusa Card shield, Hwergelmir's Tonic, or Yggdrasil Leaf (manual).",
    },
    "aspd_debuff": {
        "symptom": "Killed because attack speed was crippled by debuff (slow cast, slow attack)",
        "counter": "Watch for monsters that cast Decrease Agility (Isis, Drops, Poring MVP). Bring AGI-up potions or Berserk potions for emergency.",
        "gear_solution": "Status resist gear. Some MVP cards grant immunity to AGI decrease.",
    },
    "reflect_damage": {
        "symptom": "Killed by your own reflected damage",
        "counter": "Monsters with Reflect Shield (Raydric, certain MVPs). Stop attacking when reflect is up — switch to tank mode. Use ranged attacks or dispel.",
        "gear_solution": "Use Undead armor (turns damage into healing from Dark attacks). Or just stop attacking briefly.",
    },
}


# ── NPC dialog knowledge — NPC types, dialog patterns, shop sequences ─────────

# NPC type detection from name patterns (used by _handle_npc_dialog)
NPC_TYPE_PATTERNS: dict[str, list[str]] = {
    "kafra": ["kafra", "storage", "keeper", "warehouse"],
    "vendor": ["tool", "dealer", "shop", "item", "mart", "store", "merchant", "trade", "pawn"],
    "warp": ["warp", "portal", "gate", "kafra"],
    "healer": ["heal", "nun", "nurse", "priest", "monk", "sister", "recovery"],
    "quest": ["quest", "mission", "notice", "board", "guide", "eden"],
    "job_change": ["job", "class", "master", "guild", "association", "change"],
    "buyer": ["buyer", "purchase", "collect", "recycle"],
    "refiner": ["refine", "smith", "forge", "upgrade", "enchant"],
    "skill": ["skill", "trainer", "master"],
    "identify": ["identify", "appraise", "kara", "judgement"],
}

# NPC dialog sequences by type — the correct button/response order for common NPCs
# Key format: "sequence_name": [list of response steps]
# Steps: c = click/talk, rN = response option N, b = buy menu, s = sell menu, w = wait, e = end
NPC_DIALOG_SEQUENCES: dict[str, dict[str, list[str]]] = {
    "vendor_buy": {
        "description": "Buy items from a tool/weapon/potion dealer",
        "steps": ["c", "r1", "b", "w", "e"],
        "notes": "Talk -> Buy menu -> Select items -> Wait for transaction -> End",
        "alternative_steps": ["c", "w", "r1", "b", "e"],
    },
    "vendor_sell": {
        "description": "Sell items to a shop NPC",
        "steps": ["c", "r2", "s", "w", "e"],
        "notes": "Talk -> Sell option -> Auto-sell -> Wait -> End",
    },
    "kafra_storage": {
        "description": "Use Kafra storage service",
        "steps": ["c", "r1", "w", "e"],
        "notes": "Talk -> Storage menu -> Wait for operation -> End",
    },
    "warp_generic": {
        "description": "Use a warp NPC",
        "steps": ["c", "r1", "w", "r1", "w", "e"],
        "notes": "Talk -> Warp menu -> Select destination -> Confirm -> Wait -> End",
    },
    "healer": {
        "description": "Get healed by a healer NPC",
        "steps": ["c", "r1", "w", "e"],
        "notes": "Talk -> Heal -> Wait -> End",
    },
    "skill_reset": {
        "description": "Reset skills at a skill master",
        "steps": ["c", "r1", "w", "r1", "w", "e"],
        "notes": "Talk -> Reset menu -> Confirm -> Yes -> Wait -> End",
    },
    "identify": {
        "description": "Identify items at an identifier NPC",
        "steps": ["c", "r1", "w", "e"],
        "notes": "Talk -> Identify all -> Wait -> End",
    },
    "generic_npc": {
        "description": "Generic NPC interaction — try response options in order",
        "steps": ["c", "r1"],
        "notes": "Talk -> Select first option (fallback)",
    },
}

# Common NPC names mapped to their type for fast lookup
NPC_NAME_TO_TYPE: dict[str, str] = {
    "tool dealer": "vendor",
    "tool": "vendor",
    "weapon dealer": "vendor",
    "armor dealer": "vendor",
    "potion dealer": "vendor",
    "potion": "vendor",
    "item dealer": "vendor",
    "item seller": "vendor",
    "general goods": "vendor",
    "kafra": "kafra",
    "kafra employee": "kafra",
    "warp portal": "warp",
    "warp": "warp",
    "warp girl": "warp",
    "skill master": "skill",
    "skill trainer": "skill",
    "job master": "job_change",
    "guild master": "job_change",
    "class master": "job_change",
    "healer": "healer",
    "nurse": "healer",
    "monk": "healer",
    "sister": "healer",
    "quest npc": "quest",
    "quest": "quest",
    "eden guide": "quest",
    "identifier": "identify",
    "appraiser": "identify",
    "refiner": "refiner",
    "blacksmith": "refiner",
}

# Shop command sequences for common items (OpenKore buyAuto format)
# This helps the bot figure out npc_steps for auto-buy configs
SHOP_COMMAND_TEMPLATES: dict[str, dict[str, str]] = {
    "tool_dealer": {
        "buy": "c r1 c r1",
        "sell": "c r2 c r1",
        "description": "Old-style tool dealer in Prontera (prt_in 126 76) and other towns",
        "notes": "Uses c=click r1=response1 style — works for most basic tool dealers.",
        "fallback": "c r1 c r1",
    },
    "kafra": {
        "storage": "c r1 c w",
        "save": "c r1 c r2",
    },
    "general_shop": {
        "buy": "c r1 c r1",
        "sell": "c r1 c r2",
    },
}

# Town NPC service locations by map (fallback for discovery)
TOWN_SERVICE_NPCS: dict[str, dict[str, list[tuple[int, int]]]] = {
    "prontera": {
        "tool_dealer": [(126, 76)],
        "kafra": [(146, 121)],
        "healer": [(165, 89)],
        "skill": [(224, 125)],
        "warp": [(156, 191)],
    },
    "morocc": {
        "tool_dealer": [(130, 108)],
        "kafra": [(157, 93)],
        "healer": [(131, 57)],
    },
    "payon": {
        "tool_dealer": [(181, 104)],
        "kafra": [(187, 152)],
    },
    "geffen": {
        "tool_dealer": [(77, 127)],
        "kafra": [(57, 166)],
    },
    "aldebaran": {
        "tool_dealer": [(60, 140)],
        "kafra": [(66, 132)],
    },
    "izlude": {
        "tool_dealer": [(106, 140)],
        "kafra": [(108, 260)],
    },
}


# ── Build databases ──────────────────────────────────────────────────────────

BUILD_ADVICE: dict[str, dict[str, Any]] = {
    "swordman": {
        "class_evolves": ["knight", "lord_knight", "rune_knight"],
        "stat_priority": {"early": "STR > VIT > DEX", "mid": "STR > VIT > DEX > AGI", "late": "STR 120 > VIT 90 > DEX 60 > AGI 50"},
        "key_skills": ["Bash", "Provoke", "Increase HP Recovery", "Sword Mastery", "Two-Hand Quicken", "Magnum Break"],
        "playstyle": "Tanky melee. Use Bash for single target burst, Magnum Break for AoE. Keep Provoke on bosses to maintain aggro.",
    },
    "knight": {
        "class_evolves": ["lord_knight", "rune_knight"],
        "stat_priority": {"early": "STR > VIT > DEX", "mid": "STR 80 > VIT 60 > DEX 40", "late": "STR 120 > VIT 90 > DEX 60"},
        "key_skills": ["Bowling Bash", "Spear Boomerang", "Brandish Spear", "Cavalry Mastery", "Two-Hand Quicken"],
        "playstyle": "Bowling Bash for AoE mobbing. Spear Boomerang for ranged pull. Cavalry for speed.",
    },
    "mage": {
        "class_evolves": ["wizard", "high_wizard", "warlock"],
        "stat_priority": {"early": "INT > DEX", "mid": "INT 80 > DEX 50 > rest INT", "late": "INT 130 > DEX 90 > rest INT"},
        "key_skills": ["Fire Bolt", "Cold Bolt", "Lightning Bolt", "Fireball", "Sight", "Safety Wall"],
        "playstyle": "Elemental advantage is everything. Fire Bolt one-shots Earth monsters. Safety Wall blocks melee damage while casting.",
    },
    "wizard": {
        "class_evolves": ["high_wizard", "warlock"],
        "stat_priority": {"early": "INT > DEX", "mid": "INT 90 > DEX 60", "late": "INT 130 > DEX 90 > rest VIT"},
        "key_skills": ["Storm Gust", "Heaven's Drive", "Lord of Vermilion", "Meteor Storm", "Frost Diver"],
        "playstyle": "Storm Gust freezes + AoE. Meteor Storm for fire AoE. Use Frost Diver on dangerous targets to lock them. Watch out for cast interruption.",
    },
    "archer": {
        "class_evolves": ["hunter", "sniper", "ranger"],
        "stat_priority": {"early": "DEX > AGI", "mid": "DEX 80 > AGI 50 > rest LUK", "late": "DEX 120 > AGI 80 > LUK 50"},
        "key_skills": ["Double Strafe", "Improve Concentration", "Arrow Shower", "Owl's Eye", "Vulture's Eye"],
        "playstyle": "Kite everything. Double Strafe is your main DPS. Arrow Shower for knockback. Always keep distance — you're fragile.",
    },
    "hunter": {
        "class_evolves": ["sniper", "ranger"],
        "stat_priority": {"early": "DEX > AGI", "mid": "DEX 90 > AGI 60 > LUK 30", "late": "DEX 120 > AGI 80 > LUK 60"},
        "key_skills": ["Double Strafe", "Blitz Beat", "Ankle Snare", "Beast Bane", "Detecting"],
        "playstyle": "Blitz Beat scales with LUK for auto-proc damage. Ankle Snare for trapping dangerous monsters. Owl's Eye max range let you outrange most enemies.",
    },
    "acolyte": {
        "class_evolves": ["priest", "high_priest", "arch_bishop"],
        "stat_priority": {"early": "INT > DEX", "mid": "INT 80 > DEX 50", "late": "INT 120 > DEX 80 > rest VIT"},
        "key_skills": ["Heal", "Increase Spirit", "Angelus", "Blessing", "Magnificat"],
        "playstyle": "Offensive Heal on Undead = instant kill. Keep Blessing + Agi up permanently. Angelus for group defense. You're the party backbone.",
    },
    "merchant": {
        "class_evolves": ["blacksmith", "whitesmith", "mechanic"],
        "stat_priority": {"early": "STR > VIT > DEX", "mid": "STR 80 > VIT 50 > DEX 30", "late": "STR 120 > VIT 80 > DEX 50"},
        "key_skills": ["Mammonite", "Cart Revolution", "Push Cart", "Overcharge", "Discount"],
        "playstyle": "Mammonite is expensive but deals massive single-target burst. Cart Revolution for AoE. Use Discount/Overcharge for passive zeny gen.",
    },
    "thief": {
        "class_evolves": ["assassin", "assassin_cross", "guillotine_cross"],
        "stat_priority": {"early": "AGI > DEX > STR", "mid": "AGI 80 > DEX 40 > STR 50", "late": "AGI 120 > STR 80 > DEX 50"},
        "key_skills": ["Double Attack", "Hide", "Envenom", "Steal", "Detoxify"],
        "playstyle": "Max AGI for ASPD breakpoints. Double Attack procs on basic attacks. Hide for emergency escape. Envenom adds DoT.",
    },
    "assassin": {
        "class_evolves": ["assassin_cross", "guillotine_cross"],
        "stat_priority": {"early": "AGI > DEX > STR", "mid": "AGI 90 > STR 50 > DEX 30", "late": "AGI 120 > STR 80 > DEX 50"},
        "key_skills": ["Sonic Blow", "Katar Mastery", "Grimtooth", "Cloaking", "Soul Destroyer"],
        "playstyle": "Sonic Blow for burst. Soul Destroyer for ranged finisher. Cloaking for safe navigation past dangerous mobs.",
    },
    "taekwon": {
        "class_evolves": ["taekwon_knight", "soul_linker"],
        "stat_priority": {"early": "STR > AGI", "mid": "STR 70 > AGI 50 > DEX 30", "late": "STR 100 > AGI 80 > DEX 50"},
        "key_skills": ["Kick", "Counter Kick", "Running", "Jump Kick", "Wind Step"],
        "playstyle": "Position behind monsters for max kick damage. Running gives you best-in-class mobility. Wind Step for flee bonus.",
    },
    "gunslinger": {
        "class_evolves": ["rebellion"],
        "stat_priority": {"early": "DEX > AGI", "mid": "DEX 90 > AGI 50 > LUK 20", "late": "DEX 120 > AGI 80 > LUK 40"},
        "key_skills": ["Single Action", "Chain Action", "Tracking", "Bull's Eye", "Gatling Fever"],
        "playstyle": "Single Action for burst, Chain Action for sustained DPS. Tracking for guaranteed hit on high-flee targets. Gatling Fever for ASPD steroid.",
    },
    "ninja": {
        "class_evolves": ["kagerou", "oboro"],
        "stat_priority": {"early": "INT > DEX (magic) or STR > AGI (phys)", "mid": "INT 80 > DEX 50 or STR 70 > AGI 50", "late": "INT 120 > DEX 70 or STR 100 > AGI 80"},
        "key_skills": ["Throw Kunai", "Throw Shuriken", "Fire Ninjutsu", "Wind Ninjutsu", "Mijin", "Dokumon"],
        "playstyle": "Dual build paths. Magic ninja uses elemental ninjutsu (Fire, Water, Wind, Earth). Physical ninja uses Kunai/Shuriken with STR.",
    },
}


def _resolve_class(signals: dict[str, Any]) -> str:
    """Resolve RO class from signals, normalizing to lowercase."""
    klass = str(signals.get("class", signals.get("job", "novice"))).lower()
    # Map common class names
    class_map = {
        "soul linker": "soul_linker",
        "soul_linker": "soul_linker",
        "soul link": "soul_linker",
    }
    return class_map.get(klass, klass)


def _get_monster_attribute(signals: dict[str, Any], attr: str, default: Any = None) -> Any:
    """Get a monster attribute from signals, checking multiple key patterns."""
    target = signals.get("target", {})
    if not target:
        target = signals.get("monster", {})
    return target.get(attr, signals.get(f"monster_{attr}", default))


def _level_based_advice(level: int, player_class: str) -> list[dict[str, Any]]:
    """Generate hunting ground suggestions based on current level."""
    advice = []
    for map_name, info in HUNTING_GROUNDS.items():
        lvl_min, lvl_max = info["level"]
        if lvl_min <= level <= lvl_max:
            advice.append({
                "map": map_name,
                "description": info["description"],
                "danger": info["danger"],
                "confidence": 0.5 if info["danger"] in ("high", "very_high") else 0.8,
            })
    return advice


class ProRoPlayerProfile(BehaviorProfile):
    """Expert Ragnarok Online player with 20 years of experience providing tactical and strategic advice."""

    agent_id = "pro_ro_player"
    role = "Pro RO Player"
    goal = "Provide expert Ragnarok Online tactical and strategic advice based on 20 years of gameplay experience"

    backstory = (
        "With twenty years of Ragnarok Online under my belt — from the early iRO Chaos alpha "
        "through the pServer renaissance to modern renewal — I've played every class to 99/70 "
        "across multiple servers and watched the meta evolve through every patch. I've led "
        "guilds through WoE, hunted MVPs before anyone knew spawn timers, and theorycrafted "
        "builds that became server standards. I know the elemental wheel in my sleep, can tell "
        "you the exact flee rate you need for any map, and have an encyclopedic memory of monster "
        "spawns, item drops, and hidden mechanics that most players never discover. When a bot "
        "gets stuck, dies mysteriously, or needs to plan its next fifty levels, I can diagnose "
        "the problem the way a master mechanic hears an engine knock. This isn't book knowledge — "
        "it's scar tissue and muscle memory from two decades of RO."
    )

    def can_handle(self, signals: dict[str, Any]) -> float:
        """Score relevance based on game situation signals."""
        situation = signals.get("situation", signals.get("kind", ""))

        # Cold start — brand new or very low level
        if situation == "cold_start":
            return 1.0

        # Death analysis — something just killed the bot
        if situation == "death_analysis":
            return 0.9

        # Map change — evaluating a new hunting map
        if situation == "map_change":
            return 0.8

        # Unknown monster — first encounter with a mob
        if situation == "unknown_monster":
            return 0.8

        # Stuck — can't find a good place to level
        if situation == "stuck":
            return 0.7

        # Build planning — stat and skill allocation advice
        if situation == "build_planning":
            return 0.6

        # NPC dialog — talking to an NPC, needs correct sequence
        if situation == "npc_dialog":
            return 0.9

        # NPC dialog stuck — failed NPC interaction, wrong NPC, wrong sequence
        if situation == "npc_dialog_stuck":
            return 0.85

        # General advice — catch-all
        if situation == "general_advice":
            return 0.5

        # If the signals carry a target monster with unknown properties
        target = signals.get("target", signals.get("monster", {}))
        if target and not target.get("known", True):
            return 0.7

        # Level-based engagement — player needs route advice
        level = signals.get("level", 1)
        if level > 0 and situation in ("leveling", "grinding", "farming"):
            return 0.6

        # Combat tactics — skill rotation, flee, burst decisions
        if situation == "combat_tactics":
            return 0.9
        # Equipment — gear upgrade recommendations
        if situation == "equipment":
            return 0.8
        # Economy — buy, sell, price check
        if situation == "economy":
            return 0.7
        # Party — composition, roles, recruitment
        if situation == "party":
            return 0.8
        # MvP hunting — strategy for specific MvP
        if situation == "mvp_hunting":
            return 0.95
        # WoE — War of Emperium strategy
        if situation == "woe":
            return 0.9
        # Leveling route — optimal leveling path
        if situation == "leveling_route":
            return 0.85
        # If none of the above apply, low baseline
        return 0.1

    def get_action(self, signals: dict[str, Any]) -> dict[str, Any] | None:
        """Generate expert advice based on the current game situation."""
        situation = signals.get("situation", signals.get("kind", ""))
        player_class = _resolve_class(signals)
        level = signals.get("level", 1)

        # ── Dispatch by situation ──────────────────────────────────────

        if situation == "cold_start":
            return self._handle_cold_start(signals, player_class, level)
        if situation == "death_analysis":
            return self._handle_death_analysis(signals, player_class, level)
        if situation == "map_change":
            return self._handle_map_change(signals, player_class, level)
        if situation == "unknown_monster":
            return self._handle_unknown_monster(signals)
        if situation == "stuck":
            return self._handle_stuck(signals, player_class, level)
        if situation == "build_planning":
            return self._handle_build_planning(signals, player_class, level)
        if situation == "combat_tactics":
            return self._handle_combat_tactics(signals)
        if situation == "equipment":
            return self._handle_equipment(signals, player_class, level)
        if situation == "economy":
            return self._handle_economy(signals)
        if situation == "party":
            return self._handle_party(signals)
        if situation == "mvp_hunting":
            return self._handle_mvp_hunting(signals)
        if situation == "woe":
            return self._handle_woe(signals)
        if situation == "leveling_route":
            return self._handle_leveling_route(signals, player_class, level)
        if situation == "npc_dialog":
            return self._handle_npc_dialog(signals, player_class, level)
        if situation == "npc_dialog_stuck":
            return self._handle_npc_dialog_stuck(signals, player_class, level)
        if situation in ("general_advice", "leveling", "grinding", "farming"):
            return self._handle_general_advice(signals, player_class, level)

        return None

    # ── Situation handlers ─────────────────────────────────────────────

    def _handle_cold_start(self, signals: dict[str, Any], player_class: str, level: int) -> dict[str, Any]:
        """Provide early-game guidance — data-driven from knowledge DB."""
        from ai_sidecar.game_knowledge import game_knowledge
        gk = game_knowledge()
        early = CLASS_EARLY_GAME.get(player_class, CLASS_EARLY_GAME["novice"])
        rec_map, rec_desc = gk.recommended_map(level, player_class)
        dyn_stats = gk.starting_stats(player_class)
        dyn_equip = gk.equipment_by_level(level, player_class)
        hunting_grounds = []
        safe_maps = gk.safe_hunting_maps(level)
        seen_maps = set()
        for sm in safe_maps:
            if sm not in seen_maps and len(hunting_grounds) < 3:
                hunting_grounds.append({"map": sm, "description": f"Level {level} hunting zone"})
                seen_maps.add(sm)
        current_hp = signals.get("hp", 0) or 0
        current_max_hp = signals.get("max_hp", 1) or 1
        hp_ratio = int(current_hp) / max(int(current_max_hp), 1)
        job_weapons = ", ".join(gk.job_weapon_types(player_class)[:3]) or "Sword/Dagger"
        advice_parts = [
            f"**{player_class.title()} Early Game Guide**",
            "",
            f"🎯 **First Hunting Ground:** {rec_map} ({rec_desc})",
            f"📊 **Stat Build:** {dyn_stats}",
            f"🛡️ **Starting Equipment:** {dyn_equip}",
            f"⚔️ **Recommended Weapons:** {job_weapons}",
            f"💡 **Advice:** {early.get('advice', 'Kill monsters, level up, gear up.')}",
        ]
        if hunting_grounds:
            advice_parts.append("")
            advice_parts.append("**Also try these maps:**")
            for hg in hunting_grounds[:3]:
                advice_parts.append(f"  • `{hg['map']}` — {hg['description']}")
        advice_parts.append("")
        advice_parts.append("⚡ **Pro Tip:** Buy potions and Fly Wings before leaving town.")
        return {
            "kind": "command",
            "command": f"move {rec_map}",
            "confidence": 0.9,
            "reason": f"Data-driven: level {level} {player_class}, map={rec_map}, stats={dyn_stats}",
            "advice": "\n".join(advice_parts),
            "build": dyn_stats,
            "starting_map": rec_map,
            "next_milestone": f"Reach level {min(level + 30, 99)} on {rec_map}",
            "hunting_grounds": hunting_grounds[:3],
            "stats": dyn_stats,
            "equipment": dyn_equip,
            "npc_tips": [
                "tool_dealer: buy potions/flywings (npc_steps: c r1 c r1)",
                "kafra: storage/save (npc_steps: c r1)",
                "healer: free healing (npc_steps: c r1)",
            ],
            "sustain_advice": {
                "hp_warning": hp_ratio < 0.5,
                "buy_potions": "Red Potion, Fly Wing, Green Potion",
                "auto_use": "Set auto-use Red Potion at 50% HP",
                "teleport_escape": "Set escape at 20% HP",
            },
        }


        return {
            "kind": "command",
            "command": f"move {early['first_map']}",
            "confidence": 0.9,
            "reason": f"Early-game guidance for level {level} {player_class}: recommended_hunt={hunting_grounds[0] if hunting_grounds else 'prt_fild08'}, buy_potions=Red_Potion, sell_at_50pct_weight",
            "advice": "\n".join(advice_parts),
            "build": early.get("stats", "agi_dex"),
            "starting_map": early["first_map"],
            "next_milestone": f"Reach level {early['level_range'][1]} on {early['first_map']}",
            "hunting_grounds": hunting_grounds[:3] if hunting_grounds else [],
            "stats": early["stats"],
            "equipment": early["equipment"],
            "npc_tips": [
                "tool_dealer: buy potions/flywings (npc_steps: c r1 c r1)",
                "kafra: storage/save (npc_steps: c r1)",
                "healer: free healing (npc_steps: c r1)",
            ],
            "sustain_advice": {
                "hp_warning": hp_ratio < 0.5,
                "buy_potions": "Red Potion, Fly Wing, Green Potion",
                "auto_use": "Set auto-use Red Potion at 50% HP",
                "teleport_escape": "Set escape at 20% HP",
            },
        }

    def _handle_death_analysis(self, signals: dict[str, Any], player_class: str, level: int) -> dict[str, Any]:
        """Diagnose why the bot died and recommend behavioral changes."""
        death_report = signals.get("death_report", {})
        killer = death_report.get("killer", signals.get("killer", {}))
        killer_name = killer.get("name", signals.get("killer_name", "unknown"))
        killer_element = _get_monster_attribute(signals, "element", "neutral")
        killer_attack = _get_monster_attribute(signals, "attack_type", "physical")
        death_zone = death_report.get("map", signals.get("map", "unknown"))

        # Analyze death cause
        symptoms = []
        mitigations = []
        gear_tips = []

        # Check for AoE magic
        if killer_attack == "magical" and killer_element in ("fire", "water", "wind", "earth"):
            symptoms.append(DEATH_CAUSES["aoe_spell"]["symptom"])
            mitigations.append(DEATH_CAUSES["aoe_spell"]["counter"])
            gear_tips.append(DEATH_CAUSES["aoe_spell"]["gear_solution"])

        # Check elemental disadvantage
        player_armor_element = signals.get("armor_element", "neutral")
        ele_mult = ELEMENTAL_ADVANTAGE.get(killer_element, {}).get(player_armor_element, 1.0)
        if ele_mult >= 1.5:
            symptoms.append(DEATH_CAUSES["elemental_disadvantage"]["symptom"])
            mitigations.append(
                f"You took {ele_mult:.0%} damage from {killer_element} element attacks "
                f"while wearing {player_armor_element} armor. "
                + DEATH_CAUSES["elemental_disadvantage"]["counter"]
            )
            gear_tips.append(DEATH_CAUSES["elemental_disadvantage"]["gear_solution"])

        # Check multi-aggro
        if death_report.get("monsters_around", 0) >= 3 or signals.get("swarmed", False):
            symptoms.append(DEATH_CAUSES["multi_aggro"]["symptom"])
            mitigations.append(DEATH_CAUSES["multi_aggro"]["counter"])
            gear_tips.append(DEATH_CAUSES["multi_aggro"]["gear_solution"])

        # Check level gap
        killer_level = killer.get("level", death_report.get("killer_level", 0))
        if killer_level - level >= 15:
            symptoms.append(DEATH_CAUSES["level_gap_penalty"]["symptom"])
            mitigations.append(DEATH_CAUSES["level_gap_penalty"]["counter"])

        # Check for stun
        if death_report.get("status_effects", {}).get("stun", False):
            symptoms.append(DEATH_CAUSES["stun_lock"]["symptom"])
            mitigations.append(DEATH_CAUSES["stun_lock"]["counter"])
            gear_tips.append(DEATH_CAUSES["stun_lock"]["gear_solution"])

        # Check for freeze
        if death_report.get("status_effects", {}).get("freeze", False):
            symptoms.append(DEATH_CAUSES["freeze"]["symptom"])
            mitigations.append(DEATH_CAUSES["freeze"]["counter"])
            gear_tips.append(DEATH_CAUSES["freeze"]["gear_solution"])

        # Check for poison
        if death_report.get("status_effects", {}).get("poison", False):
            symptoms.append(DEATH_CAUSES["poison"]["symptom"])
            mitigations.append(DEATH_CAUSES["poison"]["counter"])
            gear_tips.append(DEATH_CAUSES["poison"]["gear_solution"])

        # Default fallback
        if not symptoms:
            symptoms.append(
                f"Killed by {killer_name} on {death_zone}. "
                "Without detailed death info, here are common failure points: "
                "insufficient VIT for stun resist, wrong armor element, or too many simultaneous attackers."
            )
            mitigations.append(
                "Check your HP threshold settings — set teleport escape at 30-40% HP. "
                "Ensure you're not fighting monsters 15+ levels above you. "
                "Verify armor element is appropriate for the zone."
            )

        advice_parts = ["**Death Analysis Report**", ""]
        advice_parts.append(f"💀 **Killed by:** {killer_name} on `{death_zone}`")
        advice_parts.append("")
        advice_parts.append("**🔍 Diagnosed Issues:**")
        for i, symptom in enumerate(symptoms, 1):
            advice_parts.append(f"  {i}. {symptom}")
        advice_parts.append("")
        advice_parts.append("**✅ Recommended Fixes:**")
        for i, mitigation in enumerate(mitigations, 1):
            advice_parts.append(f"  {i}. {mitigation}")
        if gear_tips:
            advice_parts.append("")
            advice_parts.append("**🛡️ Gear Recommendations:**")
            for tip in gear_tips:
                advice_parts.append(f"  • {tip}")

        return {
            "kind": "death_analysis",
            "command": "advise_death_analysis",
            "confidence": 0.85,
            "reason": f"Diagnosed death by {killer_name} with {len(symptoms)} contributing factors",
            "advice": "\n".join(advice_parts),
            "symptoms": symptoms,
            "mitigations": mitigations,
            "gear_tips": gear_tips,
        }

    def _handle_map_change(self, signals: dict[str, Any], player_class: str, level: int) -> dict[str, Any]:
        """Evaluate a map for safety and suitability."""
        target_map = signals.get("map", signals.get("target_map", "unknown"))
        monster_spawns = signals.get("monsters", signals.get("spawns", []))
        player_armor_element = signals.get("armor_element", "neutral")

        # Find map info
        map_info = HUNTING_GROUNDS.get(target_map, None)
        if not map_info:
            map_info = {"level": (1, 99), "danger": "unknown", "description": f"Unknown map: {target_map}"}

        lvl_min, lvl_max = map_info["level"]

        # Safety evaluation
        warnings = []
        suggestions = []

        if level < lvl_min:
            warnings.append(f"⚠️ You are below the recommended level range ({lvl_min}-{lvl_max}) for this map. Monsters will hit harder and you'll have reduced hit/flee rates.")
            suggestions.append(f"Consider leveling to at least {lvl_min} before hunting here, or stay near the entrance.")

        if level > lvl_max + 10:
            warnings.append(f"⚠️ You may be over-leveled for this map ({lvl_max} max). Experience gain will be reduced.")
            suggestions.append("Consider moving to a higher-level hunting ground for better exp/hour.")

        # Elemental hazard check
        element_hazards = {
            "mag_dun01": "fire", "mag_dun02": "fire",
            "ice_dun01": "water", "ice_dun02": "water",
            "beach_dun01": "water", "beach_dun02": "water",
            "gef_fild01": "earth",
        }
        hazard_element = element_hazards.get(target_map)
        if hazard_element:
            if player_armor_element == hazard_element:
                ele_mult = ELEMENTAL_ADVANTAGE.get(hazard_element, {}).get(player_armor_element, 1.0)
                if ele_mult < 0.5:
                    warnings.append(f"🔥 Your armor element ({player_armor_element}) matches the dominant element on this map. This is good — you're naturally resistant.")
                else:
                    warnings.append(f"🔥 Monsters here are mainly {hazard_element} element. Consider wearing {hazard_element} armor or carrying elemental resist potions.")
            else:
                warnings.append(f"🔥 Monsters here are mainly {hazard_element} element. Your current armor ({player_armor_element}) does not resist this.")

        # Evaluate monster spawns
        if monster_spawns:
            dangerous = [m for m in monster_spawns if m.get("danger_level", 1) >= 3]
            if dangerous:
                warnings.append(f"⚠️ Spotted {len(dangerous)} dangerous monster types: {', '.join(m.get('name', '?') for m in dangerous[:3])}.")

        # Density and aggro evaluation
        map_density = signals.get("density", "unknown")
        if str(map_density).lower() == "high":
            warnings.append("⚠️ High spawn density detected — risk of multi-aggro is elevated.")
            suggestions.append("Use a bow/ranged pull to single-target. Set teleport escape at 40% HP.")

        # Build advice
        advice_parts = [f"**Map Evaluation: {target_map}**", ""]
        advice_parts.append(f"📊 **Level Range:** {lvl_min}-{lvl_max} | **Danger:** {map_info['danger']}")
        advice_parts.append(f"📝 **Description:** {map_info['description']}")
        advice_parts.append("")

        if warnings:
            advice_parts.append("**⚠️ Warnings:**")
            for w in warnings:
                advice_parts.append(f"  • {w}")
            advice_parts.append("")

        if suggestions:
            advice_parts.append("**💡 Suggestions:**")
            for s in suggestions:
                advice_parts.append(f"  • {s}")
            advice_parts.append("")

        # Find alternative maps if this one doesn't fit
        alternatives = _level_based_advice(level, player_class)
        if alternatives:
            advice_parts.append("**🗺️ Alternative maps at your level:**")
            for alt in alternatives[:3]:
                if alt["map"] != target_map:
                    advice_parts.append(f"  • `{alt['map']}` — {alt['description']}")

        return {
            "kind": "map_evaluation",
            "command": f"evaluate_map {target_map}",
            "confidence": 0.8,
            "reason": f"Evaluated {target_map} for level {level} {player_class} — found {len(warnings)} issues",
            "advice": "\n".join(advice_parts),
            "safe": len(warnings) == 0,
            "warnings": warnings,
            "alternatives": alternatives[:3] if alternatives else [],
        }

    def _handle_unknown_monster(self, signals: dict[str, Any]) -> dict[str, Any]:
        """Infer behavior and give combat advice for an unknown monster."""
        target = signals.get("target", signals.get("monster", {}))
        monster_name = target.get("name", signals.get("monster_name", "Unknown"))
        monster_element = target.get("element", signals.get("element", "neutral"))
        monster_race = target.get("race", signals.get("race", "formless"))
        monster_size = target.get("size", signals.get("size", "medium"))
        monster_level = target.get("level", signals.get("level", 0))
        player_class = _resolve_class(signals)

        # Build profile from known data
        race_hint = RACE_HINTS.get(monster_race, "Unknown race. Approach with caution and observe its attack pattern.")

        # Elemental weaknesses
        weaknesses = []
        resistances = []
        for attack_ele, targets in ELEMENTAL_ADVANTAGE.items():
            if attack_ele in ("neutral", "holy", "dark", "ghost", "poison"):
                continue
            mult = targets.get(monster_element, 1.0)
            if mult >= 1.5:
                weaknesses.append(attack_ele)
            elif mult <= 0.5:
                resistances.append(attack_ele)

        # Size-based weapon advice
        weapon_warnings = []
        weapon_type = str(signals.get("weapon_type", "sword")).lower()
        size_pen = SIZE_PENALTY.get(weapon_type, {}).get(monster_size, 1.0)
        if size_pen < 1.0:
            weapon_warnings.append(
                f"Your {weapon_type} deals only {size_pen:.0%} damage to {monster_size}-size monsters. "
                f"Consider switching to a weapon with better size efficiency."
            )

        # Build advice text
        advice_parts = [f"**Monster Intel: {monster_name}**", ""]
        advice_parts.append(f"🏷️ **Element:** {monster_element.title()} | **Race:** {monster_race.title()} | **Size:** {monster_size.title()}")
        if monster_level:
            advice_parts.append(f"📊 **Level:** ~{monster_level}")

        advice_parts.append("")
        advice_parts.append(f"📖 **Race Intel:** {race_hint}")

        if weaknesses:
            advice_parts.append(f"💥 **Weak to:** {', '.join(w.title() for w in weaknesses)} element attacks")
        else:
            advice_parts.append("💥 **No standout elemental weakness detected — use Neutral attacks or check with different elements.**")

        if resistances:
            advice_parts.append(f"🛡️ **Resists:** {', '.join(r.title() for r in resistances)} element attacks")

        # Undead-specific advice
        if monster_race == "undead":
            advice_parts.append("")
            advice_parts.append("⚰️ **Undead Countermeasures:**")
            advice_parts.append("  • Holy Water deals massive damage (consumable)")
            advice_parts.append("  • Heal skill damages undead — use it offensively!")
            advice_parts.append("  • Turn Undead can instantly kill weaker undead")
            advice_parts.append("  • Holy element weapons deal 75% bonus damage")
            advice_parts.append("  • Bring Green Potions — undead often inflict status effects")
            if monster_element == "undead":
                advice_parts.append("  • Undead element is weak to Fire (125%) and Holy (150%)")

        # Demon-specific advice
        if monster_race == "demon":
            advice_parts.append("")
            advice_parts.append("😈 **Demon Countermeasures:**")
            advice_parts.append("  • Demons are often Dark element — Holy element demolishes them (175% bonus)")
            advice_parts.append("  • Devotion and Aspersio skills give you Holy property attacks")
            advice_parts.append("  • Shadow property resists Dark (50%)")

        if weapon_warnings:
            advice_parts.append("")
            advice_parts.append(f"⚔️ **Weapon Note:**")
            for w in weapon_warnings:
                advice_parts.append(f"  • {w}")

        # Class-specific advice
        if player_class in ("mage", "wizard", "high_wizard", "warlock"):
            if weaknesses:
                advice_parts.append("")
                advice_parts.append(f"🔮 **Wizard Tip:** Use {'/'.join(w.title() for w in weaknesses[:2])} Bolt spells for max damage against this monster.")

        if player_class in ("archer", "hunter", "sniper", "ranger"):
            advice_parts.append("")
            advice_parts.append(f"🏹 **Archer Tip:** Use elemental arrows ({', '.join(w.title() for w in weaknesses[:2])}) to exploit elemental weakness.")

        return {
            "kind": "monster_intel",
            "command": f"identify_monster {monster_name}",
            "confidence": 0.8,
            "reason": f"Provided combat intel for {monster_name} ({monster_element} element, {monster_race} race)",
            "advice": "\n".join(advice_parts),
            "element": monster_element,
            "race": monster_race,
            "size": monster_size,
            "weaknesses": weaknesses,
            "resistances": resistances,
        }

    def _handle_stuck(self, signals: dict[str, Any], player_class: str, level: int) -> dict[str, Any]:
        """Help when the bot is stuck and can't find a good place to level."""
        current_map = signals.get("map", "unknown")
        restock_items = signals.get("needed_items", signals.get("restock", []))
        party_needed = signals.get("need_party", False)

        # Level-appropriate hunting grounds
        alternatives = _level_based_advice(level, player_class)

        # Generate advice
        advice_parts = [f"**Stuck? Here's a plan for level {level} {player_class}.**", ""]

        if alternatives:
            advice_parts.append("**🗺️ Recommended Hunting Grounds:**")
            for alt in alternatives[:5]:
                advice_parts.append(f"  • `{alt['map']}` — {alt['description']}")
            advice_parts.append("")
        else:
            advice_parts.append(f"⚠️ No specific hunting grounds found for level {level}. Let me give you general advice:")
            if level < 30:
                advice_parts.append("  Low level: Stay on field maps near towns. Payon fields, Geffen fields, Morocc fields.")
            elif level < 60:
                advice_parts.append("  Mid level: Try dungeons like Payon Cave, Anthell, Geffen Cave.")
            elif level < 85:
                advice_parts.append("  High level: Magma Dungeon, Ice Dungeon, Glast Heim.")
            else:
                advice_parts.append("  Endgame: Thor Volcano, Abyss Lake, Biolabs.")
            advice_parts.append("")

        # Restocking advice
        if restock_items:
            advice_parts.append("**📦 Restocking Checklist:**")
            advice_parts.append(f"  Items needed: {', '.join(str(r) for r in restock_items[:5])}")
            advice_parts.append("")
        else:
            advice_parts.append("**📦 General Restock Checklist:**")
            advice_parts.append("  • 200+ Fly Wings (mandatory for any dungeon)")
            advice_parts.append("  • 100+ Blue Potions / SP recovery if caster")
            advice_parts.append("  • Elemental converters matching your hunting ground")
            advice_parts.append("  • 50+ Green Potions for status cure")
            advice_parts.append("  • 20+ Holy Water if hunting undead/demon areas")
            advice_parts.append("  • Resist potions matching the zone element")
            advice_parts.append("")

        # Party advice
        if party_needed:
            advice_parts.append("**👥 Party Recruitment:**")
            advice_parts.append("  At your level, consider partying with:")
            advice_parts.append("  • A Priest for Blessing + Heal support")
            advice_parts.append("  • A Wizard for AoE clearing")
            advice_parts.append("  • A Hunter for ranged DPS and trapping")
            advice_parts.append("  Or join a dedicated leveling party for your level range.")
            advice_parts.append("")

        # Generic stuck advice
        advice_parts.append("**💡 General Tips:**")
        advice_parts.append("  • If dying too much, move to an easier map and ensure your armor element counters the zone.")
        advice_parts.append("  • Check that your stat build matches your class (DEX for hit rate, AGI for ASPD/flee).")
        advice_parts.append("  • Make sure you have the correct weapon type for monster size on this map.")
        advice_parts.append("  • Set teleport escape at 30% HP to avoid dying before you can react.")
        advice_parts.append(f"  • If all else fails, try a different class or farming zeny for better gear on {current_map}.")

        return {
            "kind": "stuck_advice",
            "command": "advise_stuck",
            "confidence": 0.7,
            "reason": f"Provided stuck advice for level {level} {player_class} with {len(alternatives)} alternatives",
            "advice": "\n".join(advice_parts),
            "alternatives": alternatives[:5],
        }

    def _handle_build_planning(self, signals: dict[str, Any], player_class: str, level: int) -> dict[str, Any]:
        """Provide stat distribution and skill rotation advice."""
        build = BUILD_ADVICE.get(player_class)

        if not build:
            # Try parent class lookup
            parent_map = {
                "knight": "swordman", "lord_knight": "swordman", "rune_knight": "swordman",
                "wizard": "mage", "high_wizard": "mage", "warlock": "mage",
                "hunter": "archer", "sniper": "archer", "ranger": "archer",
                "priest": "acolyte", "high_priest": "acolyte", "arch_bishop": "acolyte",
                "monk": "acolyte", "champion": "acolyte", "sura": "acolyte",
                "blacksmith": "merchant", "whitesmith": "merchant", "mechanic": "merchant",
                "alchemist": "merchant", "creator": "merchant", "genetic": "merchant",
                "assassin": "thief", "assassin_cross": "thief", "guillotine_cross": "thief",
                "rogue": "thief", "stalker": "thief", "shadow_chaser": "thief",
                "bard": "archer", "clown": "archer", "minstrel": "archer",
                "dancer": "archer", "gypsy": "archer", "wanderer": "archer",
                "sage": "mage", "professor": "mage", "sorcerer": "mage",
                "crusader": "swordman", "paladin": "swordman", "royal_guard": "swordman",
                "taekwon_knight": "taekwon", "soul_linker": "taekwon",
                "kagerou": "ninja", "oboro": "ninja",
            }
            parent = parent_map.get(player_class)
            build = BUILD_ADVICE.get(parent) if parent else None

        if not build:
            # Fallback generic advice
            advice = (
                f"**Build Guide for {player_class.title()} (Level {level})**\n\n"
                f"No specific build data for {player_class}. Generic RO build rules:\n"
                f"• DEX = hit rate + attack speed for ranged. You need enough to never miss.\n"
                f"• STR = damage for melee, VIT = HP + stun resist (50 VIT = 100% stun immunity)\n"
                f"• AGI = flee + ASPD. Pure AGI builds (99) are very effective for leveling.\n"
                f"• INT = SP + MATK for casters.\n"
                f"• LUK = crit + status resist. Good for hunters (Blitz Beat) and crit builds.\n"
                f"\nGeneral rule: focus ONE primary stat to 80+ before diversifying."
            )
            return {
                "kind": "build_advice",
                "command": "advise_build",
                "confidence": 0.4,
                "reason": f"Generic build advice for {player_class} at level {level} (no specific build data)",
                "advice": advice,
            }

        # Determine level bracket
        if level <= 40:
            bracket = "early"
        elif level <= 70:
            bracket = "mid"
        else:
            bracket = "late"

        stat_advice = build["stat_priority"].get(bracket, build["stat_priority"]["early"])
        evo_path = " -> ".join(build["class_evolves"])

        advice_parts = [
            f"**Build Guide: {player_class.title()} → {evo_path}**",
            "",
            f"📊 **Stat Priority ({bracket}-game, level {level}):**",
            f"  {stat_advice}",
            "",
            f"🎮 **Playstyle:** {build['playstyle']}",
            "",
            "**⚡ Key Skills to Max:**",
        ]
        for skill in build["key_skills"]:
            advice_parts.append(f"  • {skill}")

        advice_parts.append("")
        advice_parts.append("**📈 Stat Breakpoints to Know:**")
        if "swordman" in player_class or "knight" in player_class:
            advice_parts.append("  • STR 100: +50% damage bonus (stat scaling)")
            advice_parts.append("  • VIT 50: 100% stun resistance")
            advice_parts.append("  • DEX = hit rate — enough to never miss your level range")
        elif "mage" in player_class or "wizard" in player_class:
            advice_parts.append("  • INT 120: +120% MATK bonus")
            advice_parts.append("  • DEX 70: 1-second cast time reduction breakpoint")
            advice_parts.append("  • INT breakpoints: 40, 80, 120 give big MATK spikes")
        elif "archer" in player_class or "hunter" in player_class:
            advice_parts.append("  • DEX 120: max hit rate for endgame bosses")
            advice_parts.append("  • AGI 75-85: ASPD breakpoint for 2-attacks-per-second")
            advice_parts.append("  • LUK 30-60: Blitz Beat proc rate + crit")
        elif "thief" in player_class or "assassin" in player_class:
            advice_parts.append("  • AGI 99: max flee for leveling")
            advice_parts.append("  • STR 80+: damage starts scaling well after AGI is capped")
            advice_parts.append("  • ASPD 175-190: key breakpoints for katar damage")

        advice_parts.append("")
        advice_parts.append(
            "⚡ **Pro Tip:** Don't spread stats — focus your primary stat to at least 80 before "
            "putting points into a secondary stat. Hybrid builds underperform until very high levels."
        )

        return {
            "kind": "build_advice",
            "command": "advise_build",
            "confidence": 0.8,
            "reason": f"{bracket}-game build advice for {player_class} at level {level}",
            "advice": "\n".join(advice_parts),
            "stat_priority": {bracket: stat_advice},
            "key_skills": build["key_skills"],
            "playstyle": build["playstyle"],
        }

    def _handle_general_advice(self, signals: dict[str, Any], player_class: str, level: int) -> dict[str, Any]:
        """Provide general RO advice for leveling, farming, and gameplay."""
        current_map = signals.get("map", "unknown")
        hunting_grounds = _level_based_advice(level, player_class)

        # RO wisdom for your level range
        tips = []
        if level < 30:
            tips = [
                "Always keep 10+ Fly Wings in inventory for emergency escape.",
                "Don't hoard money early — gear upgrades double your kill speed.",
                "Pecopeco Cards in weapons (+ATK) are the best budget DPS upgrade.",
                "Train in dungeons (Payon Cave, Geffen fields) for faster spawns, not field maps.",
                "If you're dying, check DEX for hit rate — missing = zero damage.",
            ]
        elif level < 60:
            tips = [
                "Elemental advantage is the single biggest DPS multiplier in RO. Always exploit it.",
                "Anthell is excellent exp from 30-50. Fire element destroys the insect monsters there.",
                "Start thinking about your build's stat breakpoints (every 10 STR = +damage).",
                "Farm Pecopeco Card weapons and sell them to fund gear upgrades.",
                "MVP hunting in parties starts being viable around level 50.",
            ]
        elif level < 80:
            tips = [
                "Glast Heim and Geffen Cave are excellent but dangerous — set escape thresholds.",
                "Raydric Card shield (-30% Neutral) is one of the best defensive investments.",
                "Party with a Priest for massive exp/hour gains — Blessing is +hit rate + damage.",
                "Watch your armor element! Changing armor for each zone can make you nearly invincible.",
                "Start collecting endgame gear pieces early (Orc Hero, Baphomet cards are server-dependent).",
            ]
        else:
            tips = [
                "Endgame: optimize stat builds to specific breakpoints (STR 120, INT 130, etc.).",
                "Elemental armor swapping becomes mandatory for survival in Thor/Ice/Abyss.",
                "MVP gear transforms builds — plan which MVPs to camp based on your class needs.",
                "WoE/PvP builds differ from PvE — build separate stat presets if possible.",
                "Level gap penalty (15+ levels below monster) is the run killer — always stay within range.",
            ]

        advice_parts = [
            f"**General RO Advice for Level {level} {player_class.title()}**",
            "",
            f"📍 **Current Location:** `{current_map}`",
            "",
        ]

        if hunting_grounds:
            advice_parts.append("**🗺️ Best Maps for Your Level:**")
            for hg in hunting_grounds[:3]:
                advice_parts.append(f"  • `{hg['map']}` — {hg['description']}")
            advice_parts.append("")

        advice_parts.append("**💡 Pro Tips:**")
        for i, tip in enumerate(tips, 1):
            advice_parts.append(f"  {i}. {tip}")

        advice_parts.append("")
        advice_parts.append("**⚔️ Equipment Priorities by Level:**")
        if level < 30:
            advice_parts.append("  • Weapon[3] + element/race cards > everything else")
            advice_parts.append("  • Cotton Shirt[1] + Peco Card for HP")
            advice_parts.append("  • Boots[1] + Matyr Card for ASPD")
        elif level < 60:
            advice_parts.append("  • Slot weapon with 2+ cards matching your hunting ground")
            advice_parts.append("  • Armor with element that counters the zone")
            advice_parts.append("  • Accessories with stat bonuses (STR/INT/DEX rings)")
        elif level < 80:
            advice_parts.append("  • +7 or better weapon with racial/elemental cards")
            advice_parts.append("  • Shield[1] with Raydric Card (30% neutral reduction)")
            advice_parts.append("  • Garment[1] with Whisper Card (20% flee + ghost resist)")
            advice_parts.append("  • Headgear with Marc Card (freeze immunity) for Ice Dungeon")
        else:
            advice_parts.append("  • +10/+12 weapon with MVP cards (or good racial/elemental combos)")
            advice_parts.append("  • Full elemental armor set (Fire, Water, Wind, Earth)")
            advice_parts.append("  • Endgame accessories (Vesper Core, Ring of Flame, etc.)")
            advice_parts.append("  • MVP card gear for your specific build")

        return {
            "kind": "general_advice",
            "command": "advise_general",
            "confidence": 0.6,
            "reason": f"General RO advice for level {level} {player_class}",
            "advice": "\n".join(advice_parts),
            "tips": tips,
            "hunting_grounds": hunting_grounds[:3] if hunting_grounds else [],
        }
    def _handle_combat_tactics(self, signals: dict[str, Any]) -> dict[str, Any]:
        """Provide combat tactics advice."""
        target = str(signals.get("target", "")).lower()
        monster_elem = str(signals.get("monster_element", "")).lower()
        weapon = str(signals.get("weapon", "dagger")).lower()
        aggro = int(signals.get("aggro_count", 0))
        hp_pct = float(signals.get("hp_pct", 1.0))
        sp_pct = float(signals.get("sp_pct", 1.0))
        advice_parts = [f"**Combat Tactics for {target or 'current fight'}**"]
        if target and monster_elem:
            try:
                from ai_sidecar.combat.elemental_matrix import get_elemental_matrix
                em = get_elemental_matrix()
                best_elem = ""
                best_mult = 0.0
                for e in ["water", "earth", "fire", "wind", "poison", "holy", "dark", "ghost", "undead", "neutral"]:
                    mult = em.get_elemental_multiplier(e, monster_elem, 4) * 100
                    if mult > best_mult:
                        best_mult = mult
                        best_elem = e
                advice_parts.append(f"  Best element: **{best_elem}** ({best_mult:.0f}% damage)")
            except Exception:
                pass
        if hp_pct < 0.3:
            advice_parts.append("  CRITICAL HP — flee immediately")
        elif hp_pct < 0.5:
            advice_parts.append("  Low HP — heal or kite")
        if sp_pct < 0.2:
            advice_parts.append("  Low SP — auto-attack only")
        if aggro > 3:
            advice_parts.append(f"  {aggro} enemies — use AoE or flee")
        return {
            "kind": "combat_tactics", "command": "advise_combat",
            "confidence": 0.85, "reason": f"Combat advice for {target}",
            "advice": "\n".join(advice_parts),
        }

    def _handle_equipment(self, signals: dict[str, Any], player_class: str, level: int) -> dict[str, Any]:
        """Recommend equipment upgrades."""
        zeny = int(signals.get("zeny", 0))
        job_name = str(signals.get("job_name", player_class))
        advice_parts = [f"**Equipment: {job_name.title()} Lv{level}**", f"Zeny: {zeny}"]
        if level < 40:
            advice_parts.extend(["  Weapon[3] + cards", "  Cotton Shirt[1]", "  Focus: cards > refinement"])
        elif level < 70:
            advice_parts.extend(["  +4~6 weapon, slotted armor", "  Elemental/racial cards", "  Focus: atk > def > refinement"])
        elif level < 90:
            advice_parts.extend(["  +7+ weapon, elemental armor", "  MVP-tier accessories", "  Focus: +7 weapon > card set"])
        else:
            advice_parts.extend(["  +10 weapon, full racial cards", "  Endgame armor (Val/Ori/Goib)", "  Focus: refine > enchant > costume"])
        return {
            "kind": "equipment_guide", "command": "advise_equipment",
            "confidence": 0.8, "reason": f"Gear for Lv{level} {job_name}",
            "advice": "\n".join(advice_parts),
        }

    def _handle_economy(self, signals: dict[str, Any]) -> dict[str, Any]:
        """Provide economy advice."""
        zeny = int(signals.get("zeny", 0))
        if zeny < 1000:
            tips = ["Save zeny — buy only pots and arrows"]
        elif zeny < 10000:
            tips = ["Invest in a slotted weapon for cards"]
        elif zeny < 100000:
            tips = ["Consider gear upgrades or rare cards"]
        else:
            tips = ["Look at MVP gear or rare cards"]
        return {
            "kind": "economy_advice", "command": "advise_economy",
            "confidence": 0.7, "reason": f"Economy: {zeny}z",
            "advice": "\n".join(tips),
        }

    def _handle_party(self, signals: dict[str, Any]) -> dict[str, Any]:
        """Recommend party composition."""
        pc = str(signals.get("class", "unknown")).lower()
        goal = str(signals.get("goal", "leveling"))
        synergy = {"mage": "Tank + Priest for safety", "archer": "Tank + Priest for buffs",
                   "swordman": "Priest for heal", "acolyte": "Any DPS",
                   "thief": "Priest + Tank", "merchant": "Any — you bring discounts"}
        advice = f"**Party for {pc} ({goal}):** {synergy.get(pc, 'Balanced party')}"
        if goal == "mvp":
            advice += " | 1 Tank + 2 Healers + 2 DPS for burst"
        elif goal == "woe":
            advice += " | Tank + Dispeller + 2 DPS + Healer"
        return {
            "kind": "party_advice", "command": "advise_party",
            "confidence": 0.85, "reason": f"Party for {pc}",
            "advice": advice,
        }

    def _handle_mvp_hunting(self, signals: dict[str, Any]) -> dict[str, Any]:
        """Provide MvP hunting strategy."""
        mvp = str(signals.get("target", signals.get("mvp_name", "unknown"))).lower()
        strategies = {
            "baphomet": "Demon/Dark 3 — Holy weapon. Dodge Hell's Judgment. Assumptio before pull.",
            "osiris": "Undead 4 — Holy (200%). Escape Teleport at 20% HP — burst when low.",
            "maya": "Insect/Earth 3 — Fire (200%). Bring Fire weapon. Magnum Break — watch AoE.",
            "eddga": "Brute/Earth 2 — Fire (175%). Charge Attack — maintain distance.",
            "doppelganger": "Demon/Dark 2 — Holy (175%). Reflect at 10% HP — stop attacking!",
            "orc lord": "DemiHuman/Dark 2 — Holy (175%). Grand Darkness AoE + curse.",
            "drake": "Undead/Undead 2 — Holy (175%). Charge Attack — kite.",
            "mistress": "Insect/Wind 2 — Earth (175%). Runs at low HP — trap or one-shot.",
            "phreeoni": "Brute/Neutral 3 — Ghost only. Wide Web — stay ranged.",
            "gloom": "Demon/Dark 3 — Holy (175%). Vampiric Gift — BURST HARD.",
            "thanatos": "Undead/Undead 4 — Holy (200%). 4 forms. Tank magic with MDEF.",
            "kiel": "DemiHuman/Neutral 3 — Dark (175%). Full-divest — bring spares.",
        }
        strategy = strategies.get(mvp, f"No specific data for {mvp}")
        return {
            "kind": "mvp_hunting_guide", "command": "advise_mvp",
            "confidence": 0.9, "reason": f"MvP: {mvp}", "advice": strategy,
        }

    def _handle_woe(self, signals: dict[str, Any]) -> dict[str, Any]:
        """War of Emperium strategy."""
        role = str(signals.get("role", "unknown")).lower()
        defense = bool(signals.get("is_defense", False))
        strat = []
        if defense:
            strat.append("Defense: wall casters, dispeller near Emp, Safety Wall chokepoints")
        else:
            strat.append("Offense: main gate rush, back entrance squad, Assassin infiltration")
        role_tips = {"assassin": "Cloak through, ignore fights, rush Emp",
                     "priest": "Safety Wall breaker + Lex Aeterna before kill",
                     "wizard": "AoE chokepoints, Dispel enemy buffs",
                     "knight": "Tank Emp + Pneuma for ranged block",
                     "stalker": "Strip defenders, Full Divest on Emp breaker"}
        if role in role_tips:
            strat.append(f"Your role ({role}): {role_tips[role]}")
        return {
            "kind": "woe_strategy", "command": "advise_woe",
            "confidence": 0.85, "reason": f"WoE ({'D' if defense else 'O'})",
            "advice": " | ".join(strat),
        }

    def _handle_leveling_route(self, signals: dict[str, Any], player_class: str, level: int) -> dict[str, Any]:
        """Optimal leveling path."""
        job = str(signals.get("job_name", player_class)).lower()
        if level < 15:
            maps = ["Town fields — Porings/Lunatics/Fabres"]
        elif level < 40:
            maps = ["Geffen Dungeon 1F — Drainliar/Familiar",
                    "Payon Cave 2F-3F — Bongun/Munak/Skel"] if job in ("mage","wizard","sage") else                    ["Orc Dungeon 1F — Orc Warrior",
                    "Byalan 1F — Vadon/Marina"]
        elif level < 70:
            maps = ["Magma 1F — Magmaring (Water)",
                    "Toy Factory 1F — Marionette (Ghost)"]
        elif level < 90:
            maps = ["Magma 2F — high exp, dangerous",
                    "Abyss Lake 1F-2F — great exp, GTB needed",
                    "Thanatos Tower 1F-3F — Undead, Holy needed"]
        else:
            maps = ["Biolabs 3F-4F — highest exp",
                    "Thanatos Tower 4F+ — Thanatos MVP",
                    "Nameless Island — dense undead"]
        return {
            "kind": "leveling_route", "command": "advise_leveling",
            "confidence": 0.85, "reason": f"Route: Lv{level} {job}",
            "advice": " | ".join(maps),
        }

    def _handle_npc_dialog(self, signals: dict[str, Any], player_class: str, level: int) -> dict[str, Any]:
        """Handle NPC dialog — figure out the correct sequence to talk to an NPC.

        Uses RO knowledge to determine the NPC type, then provides the correct
        dialog sequence as an actionable command. This is the dynamic alternative
        to hardcoded 'c r1 c r1' sequences.
        """
        npc_name = str(signals.get("npc_name", signals.get("target_name", ""))).lower().strip()
        npc_id = str(signals.get("npc_id", signals.get("target_id", "")))
        npc_map = str(signals.get("map", signals.get("current_map", "unknown"))).lower()
        goal = str(signals.get("goal", "buy")).lower()
        dialog_history = signals.get("dialog_history", [])
        previous_result = str(signals.get("previous_result", "")).lower()
        is_retry = signals.get("is_retry", False) or "wrong" in previous_result or "fail" in previous_result

        advice_parts = [f"**NPC Dialog: Resolving interaction with {npc_name or 'unknown NPC'}**", ""]

        # Detect NPC type from name
        npc_type = self._detect_npc_type(npc_name)

        if npc_type is None:
            advice_parts.append(f"⚠️ Unknown NPC '{npc_name}'. Trying generic approach.")
            advice_parts.append("  • Default sequence: 'c r1' — talk then select first option")
            advice_parts.append("  • If that fails, try 'c r1 c r1' for buy menus")
            sequence = "c r1"
            confidence = 0.4
        else:
            advice_parts.append(f"🏷️ **Detected NPC Type:** {npc_type.title()}")
            advice_parts.append(f"📍 **Location:** {npc_map}")

            # Get appropriate dialog sequence
            sequence_info = self._get_dialog_sequence(npc_type, npc_name, goal, is_retry)
            sequence = sequence_info["sequence"]
            advice_parts.append(f"💬 **Dialog Sequence:** {sequence}")
            advice_parts.append(f"📝 **Note:** {sequence_info['notes']}")

            # Alternative sequences for retry
            if is_retry and "alternative" in sequence_info:
                advice_parts.append(f"🔄 **Retry Alternative:** {sequence_info['alternative']}")
                sequence = sequence_info["alternative"]

            confidence = sequence_info["confidence"]

            # NPC-type specific advice
            if npc_type == "vendor" and goal in ("buy", "restock"):
                advice_parts.append("")
                advice_parts.append("**📦 Vendor Shopping Tips:**")
                advice_parts.append("  • Make sure you have enough zeny before talking")
                advice_parts.append("  • If weight > 70%, you may not be able to buy (overburdened)")
                advice_parts.append("  • Check that the NPC actually sells what you need")
                advice_parts.append("  • If 'Talking to wrong npc', the NPC ID may differ on this server")
                if is_retry:
                    advice_parts.append("  • RETRY: Try talking to a different NPC at the same position")
                    advice_parts.append("  • RETRY: The vendor might have moved or this is a different NPC now")

            elif npc_type == "kafra":
                advice_parts.append("")
                advice_parts.append("**📦 Kafra Storage Tips:**")
                advice_parts.append("  • Kafra storage is shared across all characters on the same account")
                advice_parts.append("  • You need at least 1 zeny to open storage")
                advice_parts.append("  • Some servers charge storage fees per item")

            elif npc_type == "warp" and goal == "travel":
                advice_parts.append("")
                advice_parts.append("**🚀 Warp Portal Tips:**")
                advice_parts.append("  • Warp costs vary by destination (typically 100-2000z)")
                advice_parts.append("  • You can also use Fly Wings for random teleport")
                advice_parts.append("  • Some maps require quest completion to warp to")

        # Build npc_steps config recommendation (for buyAuto config)
        if npc_type == "vendor":
            advice_parts.append("")
            advice_parts.append("**⚙️ Config Recommendation for auto-buy:**")
            advice_parts.append(f"  Set `npc_steps {sequence}` in your buyAuto config")
            advice_parts.append(f"  Set `npc {npc_map} <x> <y>` (NPC coordinates)")

        return {
            "kind": "npc_dialog_guide",
            "command": f"talknpc {npc_id}" if npc_id else "macro npc_dialog_resolve",
            "confidence": confidence,
            "reason": f"NPC dialog guide for {npc_name} ({npc_type or 'unknown'}) in {npc_map} — sequence: {sequence}",
            "advice": "\n".join(advice_parts),
            "npc_type": npc_type or "unknown",
            "npc_name": npc_name,
            "npc_sequence": sequence,
            "sequence_parts": sequence.split(),
        }

    def _handle_npc_dialog_stuck(self, signals: dict[str, Any], player_class: str, level: int) -> dict[str, Any]:
        """Handle stuck NPC interactions — dialog failed, wrong NPC, shop can't complete.

        Diagnoses why an NPC interaction failed and provides a recovery plan.
        This handles the 'Talking to wrong npc', 'Npc did not respond',
        and auto-buy loop failure scenarios.
        """
        npc_name = str(signals.get("npc_name", signals.get("target_name", ""))).lower().strip()
        npc_map = str(signals.get("map", signals.get("current_map", "unknown"))).lower()
        failure_type = str(signals.get("failure_type", "")).lower()
        error_message = str(signals.get("error_message", signals.get("failure_reason", "")))
        dialog_history = signals.get("dialog_history", [])
        weight_pct = float(signals.get("weight_pct", signals.get("weight", 0)))
        hp = int(signals.get("hp", 0))
        max_hp = int(signals.get("max_hp", 1))
        zeny = int(signals.get("zeny", 0))
        attempts = int(signals.get("attempt_count", signals.get("retry_count", 0)))

        advice_parts = [f"**NPC Dialog Stuck: {npc_name or 'unknown NPC'} — Recovery Plan**", ""]

        # Detect stuck pattern
        is_wrong_npc = "wrong" in error_message or "wrong npc" in failure_type
        is_no_response = "not respond" in error_message or "no response" in failure_type
        is_overweight = weight_pct >= 70 or signals.get("weight_over_70", False)
        is_low_hp = hp < 50 and max_hp > 0 and (hp / max_hp) < 0.3
        is_loop = attempts >= 3

        advice_parts.append(f"🔍 **Situation Analysis:**")
        if is_wrong_npc:
            advice_parts.append("  ❌ Talking to wrong NPC — the NPC ID at this position has changed or")
            advice_parts.append("     the coordinates in the config point to a different NPC than expected.")
            advice_parts.append("")
            advice_parts.append("  **⚡ Recovery:**")
            advice_parts.append("    1. Use NPCDiscoveryEngine to find the actual vendor NPC position")
            advice_parts.append("    2. The Tool Dealer may be at a different coordinate on this server")
            advice_parts.append("    3. Check if the NPC name matches what the config expects")
            advice_parts.append("    4. Try using dynamic NPC discovery instead of hardcoded coordinates")
        elif is_no_response:
            advice_parts.append("  ❌ NPC did not respond — the dialog sequence may be wrong,")
            advice_parts.append("     or the NPC is too far away / blocked by other characters.")
            advice_parts.append("")
            advice_parts.append("  **⚡ Recovery:**")
            advice_parts.append("    1. Make sure you're standing close enough to the NPC (within 3 cells)")
            advice_parts.append("    2. The NPC might be busy with another player — wait and retry")
            advice_parts.append("    3. Try a different dialog sequence (some servers use different menus)")
        elif is_overweight:
            advice_parts.append("  ⚠️ Weight over 70% — you're overburdened and may not be able to")
            advice_parts.append("     move effectively or complete transactions.")
            advice_parts.append("")
            advice_parts.append("  **⚡ Recovery:**")
            advice_parts.append("    1. Sell junk items to lighten your load")
            advice_parts.append("    2. Deposit items in Kafra storage")
            advice_parts.append("    3. Use a Butterfly Wing to return to town if stuck")
            advice_parts.append("    4. Check your items_control.txt — you may be picking up too much junk")
        elif is_low_hp:
            heal_ratio = hp / max(hp, 1)
            advice_parts.append(f"  ❌ Critical HP ({hp}/{max_hp}) — you need healing before doing anything else.")
            advice_parts.append("")
            advice_parts.append("  **⚡ Recovery:**")
            advice_parts.append("    1. Sit and regenerate HP before attempting NPC interactions")
            advice_parts.append("    2. Use healing items if available (Red Potions, White Potions)")
            advice_parts.append("    3. Visit a healer NPC (nun/priest/monk) for free healing")
            advice_parts.append("    4. Check your auto-buy config — you may need more potions")
            if zeny < 100:
                advice_parts.append("    ⚠️ You have very little zeny — consider selling items first")
        elif is_loop:
            advice_parts.append(f"  🔁 Loop detected — {attempts} consecutive failed attempts to interact with this NPC.")
            advice_parts.append("")
            advice_parts.append("  **⚡ Recovery:**")
            advice_parts.append("    1. Stop the current action and re-evaluate")
            advice_parts.append("    2. Use NPCDiscoveryEngine to find a different vendor NPC")
            advice_parts.append("    3. Consider changing hunting strategy to avoid needing this NPC")
            advice_parts.append("    4. If this is a buy loop, try buying from a different shop or player vendor")

        # Check for recurring pattern across all bots
        if is_loop and is_wrong_npc:
            advice_parts.append("")
            advice_parts.append("**🔧 Long-term Fix:**")
            advice_parts.append("  This NPC interaction is repeatedly failing. Consider:")
            advice_parts.append("  • Use the NPCDiscoveryEngine to dynamically locate the correct vendor")
            advice_parts.append("  • Update the npc_steps in config.txt to match this server's dialog flow")
            advice_parts.append("  • Use LLM-powered NPC dialog (Pro RO Player + NPCDialogEngine)")
            advice_parts.append("  • As a workaround: teleport to a different town with a working vendor")

        # General health/stuck check
        advice_parts.append("")
        advice_parts.append("**📊 Bot Health Check:**")
        advice_parts.append(f"  • HP: {hp}/{max_hp}" + (" ⚠️ CRITICAL" if is_low_hp else ""))
        advice_parts.append(f"  • Weight: {weight_pct:.0f}%" + (" ⚠️ Overburdened" if is_overweight else ""))
        advice_parts.append(f"  • Zeny: {zeny}z" + (" ⚠️ Very low" if zeny < 500 else ""))
        advice_parts.append(f"  • Map: {npc_map}")
        advice_parts.append(f"  • Failed attempts: {attempts}")

        stuck_kind = "npc_stuck_critical" if (is_low_hp or (is_loop and is_wrong_npc)) else "npc_stuck_minor"
        confidence = 0.9 if (is_low_hp or is_loop) else 0.7

        return {
            "kind": stuck_kind,
            "command": "recover_npc_dialog",
            "confidence": confidence,
            "reason": f"NPC dialog stuck: {npc_name} — {error_message or failure_type} ({attempts} attempts)",
            "advice": "\n".join(advice_parts),
            "npc_name": npc_name,
            "npc_map": npc_map,
            "failure_type": failure_type or error_message or "unknown",
            "overweight": is_overweight,
            "critical_hp": is_low_hp,
            "loop_detected": is_loop,
            "wrong_npc": is_wrong_npc,
            "suggested_actions": [
                "discover_vendor_npc" if is_wrong_npc else "",
                "sit_and_heal" if is_low_hp else "",
                "sell_junk_items" if is_overweight else "",
                "change_hunting_strategy" if is_loop else "",
            ],
        }

    # ── NPC dialog helper methods ─────────────────────────────────────────

    @staticmethod
    def _detect_npc_type(npc_name: str) -> str | None:
        """Detect NPC type from name using pattern matching."""
        if not npc_name:
            # Check if name was passed in signals differently
            return None

        # Try exact match first
        if npc_name in NPC_NAME_TO_TYPE:
            return NPC_NAME_TO_TYPE[npc_name]

        # Try partial match against patterns
        for npc_type, patterns in NPC_TYPE_PATTERNS.items():
            for pattern in patterns:
                if pattern in npc_name:
                    return npc_type

        return None

    @staticmethod
    def _get_dialog_sequence(npc_type: str, npc_name: str, goal: str, is_retry: bool = False) -> dict[str, Any]:
        """Get the appropriate dialog sequence for an NPC."""
        # Check if we have a specific shop template for this NPC name
        for template_key, template in SHOP_COMMAND_TEMPLATES.items():
            if template_key in npc_name or any(k in npc_name for k in template_key.split("_")):
                if goal in template:
                    return {
                        "sequence": template[goal],
                        "alternative": template.get("fallback", template[goal]),
                        "notes": template.get("notes", ""),
                        "confidence": 0.8,
                    }

        # Check NPC_DIALOG_SEQUENCES
        if npc_type == "vendor" and goal == "buy":
            base = NPC_DIALOG_SEQUENCES["vendor_buy"]
            return {
                "sequence": " ".join(base["steps"]),
                "alternative": " ".join(base["alternative_steps"]),
                "notes": base["notes"],
                "confidence": 0.75,
            }
        elif npc_type == "vendor" and goal == "sell":
            base = NPC_DIALOG_SEQUENCES["vendor_sell"]
            return {
                "sequence": " ".join(base["steps"]),
                "alternative": " ".join(base["steps"]),
                "notes": base["notes"],
                "confidence": 0.7,
            }

        # Map NPC dialog sequence names to npc_type
        type_to_seq = {
            "kafra": "kafra_storage",
            "warp": "warp_generic",
            "healer": "healer",
            "skill": "skill_reset",
            "identify": "identify",
        }

        seq_name = type_to_seq.get(npc_type, "generic_npc")
        if seq_name in NPC_DIALOG_SEQUENCES:
            base = NPC_DIALOG_SEQUENCES[seq_name]
            return {
                "sequence": " ".join(base["steps"]),
                "alternative": " ".join(base["steps"]),
                "notes": base.get("notes", ""),
                "confidence": 0.7,
            }

        # Fallback
        return {
            "sequence": "c r1",
            "alternative": "c r1 c r1",
            "notes": "Generic NPC — try talking and selecting first option",
            "confidence": 0.4,
        }

