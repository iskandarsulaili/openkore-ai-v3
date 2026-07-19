# Combat Optimization System — Full Architecture Reference

> Created: 2026-07-19 | Source: rAthena DB (210 files), battle.conf, full job/skill/monster DB
> Purpose: Comprehensive reference for implementing combat optimization across all game stages and job classes.

---

## 0. Table of Contents

1. [Server Mode Detection](#1-server-mode-detection)
2. [Foundational Constants](#2-foundational-constants)
3. [Job System — 112 Classes Across 6 Tiers](#3-job-system--112-classes-across-6-tiers)
4. [Skill System — 1,635 Skills](#4-skill-system--1635-skills)
5. [Elemental System — 10 Elements × 4 Levels](#5-elemental-system--10-elements--4-levels)
6. [Size System — 3 Sizes × Weapon Types](#6-size-system--3-sizes--weapon-types)
7. [Monster System — 2,675 Monsters](#7-monster-system--2675-monsters)
8. [Damage Formula (from battle.conf)](#8-damage-formula-from-battleconf)
9. [Stat System — 12 Stats](#9-stat-system--12-stats)
10. [Equipment System — 14 Item Databases](#10-equipment-system--14-item-databases)
11. [Card & Combo System](#11-card--combo-system)
12. [Companion Systems](#12-companion-systems)
13. [Refine / Grade / Enchant / Reform](#13-refine--grade--enchant--reform)
14. [Status Effects](#14-status-effects)
15. [Battle Config Mechanics](#15-battle-config-mechanics)
16. [Group & Party Mechanics](#16-group--party-mechanics)
17. [MVP / Boss / Instance Systems](#17-mvp--boss--instance-systems)
18. [Complete Combat Decision Architecture](#18-complete-combat-decision-architecture)
19. [Game Stage Progression](#19-game-stage-progression)
20. [Implementation Phases](#20-implementation-phases)
21. [RULE.md Compliance Matrix](#21-rulemd-compliance-matrix)

---

## 1. Server Mode Detection

### Why This Matters

Ragnarok Online has three distinct game modes with different mechanics:

| Aspect | Classic/Pre-renewal | Renewal | Difference |
|--------|-------------------|---------|------------|
| Defense formula | VIT × 0.8 | VIT × 0.5 + equip/400+equip | ~40% less effective VIT DEF |
| Max level | 99 | 175-265 | Higher levels = different stat scaling |
| ASPD formula | AGI only | AGI + DEX + weapon | Attack speed scales differently |
| Cast time | DEX alone | DEX + items + skills | More ways to reduce cast |
| Stat bonus | Lower per stat | Higher per stat | Stats matter more |
| Skill versions | Pre-renewal skill IDs | Renewal skill IDs | Same skill, different behavior |
| Item databases | db/pre-re/ | db/re/ | Different files loaded |

### Detection Methods

The sidecar detects server mode automatically:

1. **From bridge config**: The bridge sends `config{master}` which typically includes the server name. Common patterns: "Asgards Glory" (likely renewal, high-rate).

2. **From character data**: 
   - If max HP > 100k → renewal (pre-renewal caps around 50k)
   - If base level > 99 → renewal
   - If job level > 50 → renewal

3. **From monster data**:
   - Pre-renewal mob_db vs renewal mob_db have different HP/EXP values
   - Compare monster HP against known ranges

4. **User-specified fallback**: Config option `aiSidecar_gameMode` in bridge config.

### Formula Selection

Once mode is detected, all formulas branch:

```python
def _damage_formula(mode, atk, def_val):
    if mode == "prerenewal":
        return atk * (1 - def_val / 100)
    elif mode == "renewal":
        return atk * (1 - def_val / (def_val + 400))
```

All subsequent calculations (ASPD, cast time, stat bonuses) follow the same branching based on detected mode.

---

## 2. Foundational Constants

**Source:** `db/const.yml`, `conf/battle/battle.conf` (from rAthena repo)

```
Stat caps:             up to 130 base, traits separate (renewal)
Attack speed range:    2000 = 2s per attack (lower = faster)
Max ASPD:              ~190 (varies by weapon type)
Min hit rate:          5%
Max hit rate:          100%
Defense type:          pre-renewal (vit_def × 0.8) vs renewal (vit_def × 0.5)
Base ATK enabled:      players (0x9), renewal (0x29F)
Flee penalty:          3+ aggro monsters reduce flee by fixed amount
DEF penalty:           3+ aggro monsters reduce DEF
```

---

## 3. Job System — 112 Classes Across 6 Tiers

**Source:** `db/job_stats.yml`, `db/re/skill_tree.yml`, `db/job_aspd.yml`

```
Novice (1)
  ├─ 1st Class (6):           Swordman, Mage, Archer, Acolyte, Merchant, Thief
  ├─ 2nd Class (14):          Knight, Priest, Wizard, Blacksmith, Hunter, Assassin,
  │                           Crusader, Monk, Sage, Rogue, Alchemist, Bard, Dancer
  ├─ Transcendent (14):       Lord_Knight, High_Priest, High_Wizard, Whitesmith,
  │                           Sniper, Assassin_Cross, Paladin, Champion, Professor,
  │                           Stalker, Creator, Clown, Gypsy
  ├─ 3rd Class (14):          Rune_Knight, Warlock, Ranger, Arch_Bishop, Mechanic,
  │                           Guillotine_Cross, Royal_Guard, Sorcerer, Minstrel,
  │                           Wanderer, Sura, Shadow_Chaser, Genetic, ...
  ├─ 4th Class (14+):         Dragon_Knight, Meister, Shadow_Cross, Arch_Mage,
  │                           Cardinal, Windhawk, Imperial_Guard, Biolo,
  │                           Abyss_Chaser, Elemental_Master, Inquisitor,
  │                           Troubadour, Trouvere, Soul_Ascetic, Night_Watch, ...
  ├─ Expanded (6):            Taekwon, Star_Gladiator, Soul_Linker, Ninja, Gunslinger
  ├─ Baby (14):               Baby versions of 1st/2nd/3rd classes
  ├─ Super_Novice:            Multi-class Novice with access to many skills
  └─ Hyper_Novice:            Advancement of Super_Novice

Upgrade path:
  Base Lv 1-99 + Job Lv 1-50 (1st class)
    → Transcendent (rebirth, start from Lv 1 again)
    → Base Lv 99/150 + Job Lv 50/70
    → 3rd class (Base Lv 99-200, Job Lv 50-100)
    → 4th class (Base Lv 200+, Job Lv 50+)
```

**Each job defines** (from `job_stats.yml`):
```
- Jobs:          [class_name: true/false]
- MaxWeight:     base maximum carry weight
- HpFactor:      exponential HP growth per level
- HpIncrease:    linear HP growth per level
- SpFactor/SpIncrease: same for SP
- BaseASPD:      attack speed per weapon type (default 2000)
- BonusStats:    stat bonuses per job level (+1 STR at Lv 8, etc.)
- MaxStats:      stat caps per job
- MaxBaseLevel:  level cap
- MaxJobLevel:   job level cap
- BaseExp/JobExp: exp tables per level
- BaseHp/BaseSp:  base values per level
```

---

## 4. Skill System — 1,635 Skills

**Source:** `db/re/skill_db.yml`

```
Id:             unique skill ID
Name:           Aegis name (e.g., SM_BASH)
Description:    display name
MaxLevel:       max skill level

Type:           Weapon/Magic/Misc/Passive
TargetType:     Attack/Self/Party/Enemy/Ground/Passive

DamageFlags:    damage properties (can crit, ignore def, etc.)
Flags:          behavior flags (no_memo, no_teleport, etc.)

Range:          [per level] skill range in cells
Hit:            Normal/Magic/Weapon hit type
HitCount:       [per level] number of damage hits
Element:        [per level] element of the skill
SplashArea:     [per level] AoE radius
Knockback:      [per level] pushback tiles
GiveAp:         [per level] AP generation

CastTime:       [per level] variable cast (ms, reduced by DEX)
FixedCastTime:  [per level] fixed cast (ms, items only)
AfterCastActDelay:  [per level] GCD before next skill (ms)
AfterCastWalkDelay: [per level] movement lock (ms)
Cooldown:       [per level] cannot use same skill again (ms)
Duration1/2:    [per level] buff/debuff duration (ms)

Requires:
  HpCost:       [per level] HP consumed
  SpCost:       [per level] SP consumed
  ApCost:       [per level] AP consumed
  Items:        [per level] consumable item requirement
  Ammunition:   [per level] arrow/bullet type required

CastCancel:     bool — interruptible by damage
CastDefenseReduction: DEF penalty during cast
```

**Combat timing chain:**
```
CastStart → [CastTime - FixedCastTime] → [FixedCastTime] → SkillLands
    ↓ interruptible if castCancel=true
    ↓
AfterCastActDelay (GCD — no skills possible, auto-attack continues)
    ↓
AfterCastWalkDelay (cannot move)
    ↓
Cooldown (cannot use same skill)
    ↓
Duration active (buffs/debuffs/ground effects)
```

---

## 5. Elemental System — 10 Elements × 4 Levels

**Source:** `db/attr_fix.yml`

```
Elements: Neutral, Water, Earth, Fire, Wind, Poison, Holy, Dark, Ghost, Undead
Levels: 1-4 per element (higher level = more extreme modifiers)

Key modifier patterns (Level 1 example):
  Same element → 25% damage
  Weakness → 150% damage
  Strong resist → 75% damage
  Immune → 0% (level 4 same-element, Ghost vs Neutral)

Level 4 extremes:
  Same element → 0% (immune)
  Weakness → 200% damage (double)
  Strong resist → 50-60% damage

Full matrix in db/attr_fix.yml — 40 entries (10 elements × 4 levels)
```

---

## 6. Size System — 3 Sizes × Weapon Types

**Source:** `db/size_fix.yml`

```
Weapon Type    Small    Medium    Large
Knuckle        100%     100%      75%
Whip           100%     100%      75%
All others    100%     100%     100%

Sizes: Small, Medium, Large
Monsters mapped to sizes in mob_db.yml
```

---

## 7. Monster System — 2,675 Monsters

**Source:** `db/re/mob_db.yml`

```
Fields per monster (32 total):
  Id, AegisName, Name, Level, Hp, BaseExp, JobExp
  Attack, Attack2, Defense, MagicDefense
  Str, Agi, Vit, Int, Dex, Luk
  AttackRange, SkillRange, ChaseRange
  Size (Small/Medium/Large)
  Race (Angel/Brute/Demihuman/Demon/Dragon/Fish/Formless/Insect/Plant/Player/Undead)
  Element (10 values), ElementLevel (1-4)
  WalkSpeed (100-300+), AttackDelay, AttackMotion, DamageMotion
  Modes (move/attack/aggro/boss/cast_sensor/change_target/etc.)
  Ai type, Drops[]
```

**Monster Races:**
```
Angel, Brute, Demihuman, Demon, Dragon, Fish, Formless
Insect, Plant, Player_Doram, Player_Human, Undead
```

**Monster AI** (from `db/mob_skill_db.txt`):
```
MobID, State, SkillID, SkillLv, Rate(%), CastTime, Delay, Cancelable
Target, Condition, ConditionValue

States:     idle, walk, attack, chase, angry, loot, dead, anytarget
Conditions: always, hp<25%, friend_hp<25%, target_job, status
Targets:    target, self, friend, master, random, around1-3
```

---

## 8. Damage Formula (from battle.conf)

### Pre-renewal vs Renewal branching

Mode is auto-detected (see Section 1). All formulas branch:

```python
class DamageCalculator:
    def __init__(self, mode: str):
        self.mode = mode  # "prerenewal" or "renewal"
    
    def physical_damage(self, base_atk, weapon_atk, def_val, ...):
        total_atk = base_atk + weapon_atk
        if self.mode == "prerenewal":
            soft_def = def_val * 0.8
            return max(1, total_atk - soft_def)
        else:  # renewal
            return total_atk * (1 - def_val / (def_val + 400))
    
    def cast_time(self, base_cast, dex):
        if self.mode == "prerenewal":
            return base_cast * (1 - dex / 150)
        else:
            return base_cast * (1 - dex / 265)
    
    def aspd(self, base_aspd, agi, dex):
        if self.mode == "prerenewal":
            return base_aspd - sqrt(agi * 25)
        else:
            return base_aspd - sqrt(agi * 25 + dex * 5)
```

### Physical Damage Formula

```
BaseATK   = 2×STR + (STR×STR)/10 + DEX/5 + LUK/5 + job_bonus
WeaponATK = weapon_atk + refine_bonus + card_bonus
EquipATK  = accessory_atk + garment_atk
TotalATK  = (BaseATK + WeaponATK) × SizeMod × EleMod × RaceMod × CardMod × RefineMod × SkillMult

[Pre-renewal DEF]:
  SoftDEF = VIT × 0.8
  HardDEF = equip_def + refine_def
  FinalDMG = TotalATK × (1 - HardDEF/100) - SoftDEF

[Renewal DEF]:
  SoftDEF = VIT × 0.5
  HardDEF = equip_def + refine_def
  Reduction = HardDEF / (HardDEF + 400)
  FinalDMG = TotalATK × (1 - Reduction) - SoftDEF × 0.5

[Caps]:
  min_hitrate: 5%
  max_hitrate: 100%
  HitRate = 100 - (Flee - Hit)  [clamped to caps]
  CritRate = LUK × 0.3 + equip_crit + skill_crit  [capped at 100%]
  CritDMG = FinalDMG × 1.4
```

### Magic Damage Formula

```
BaseMATK   = INT + (INT×INT)/100 + job_bonus
WeaponMATK = weapon_matk + refine_bonus
TotalMATK  = (BaseMATK + WeaponMATK) × SkillPower% × EleMod × CardMod

[Pre-renewal MDEF]:
  FinalMDMG = TotalMATK × (1 - MDEF/100)

[Renewal MDEF]:
  MReduction = MDEF / (MDEF + 400)
  FinalMDMG = TotalMATK × (1 - MReduction)
```

### ASPD Calculation

```
[Pre-renewal]:
  BaseASPD = job_aspd_table[class][weapon_type]
  ASPD = BaseASPD - SQRT(AGI × 25)

[Renewal]:
  BaseASPD = job_aspd_table[class][weapon_type]
  ASPD = BaseASPD - SQRT(AGI × 25 + DEX × 5) - skill_bonus - potion_bonus

AttackInterval = 200 × (2000 - ASPD) / 1000 ms
```

### Cast Time

```
[Pre-renewal]:
  VariableCast = base_cast × (1 - DEX/150)
  TotalCast = VariableCast + FixedCast

[Renewal]:
  VariableCast = base_cast × (1 - DEX/265)
  TotalCast = VariableCast + FixedCast
```

### Level Penalty

```
EXP penalty: -31 level diff = 10% exp, +16 diff = 40% exp, +10 diff = 140% exp
Drop penalty: -16 level diff = 50% drops, +4 diff = 90% drops
```

---

## 9. Stat System — 12 Stats

**Source:** `db/statpoint.yml`, `db/job_stats.yml`, `conf/battle/battle.conf`

```
Primary (6):
  STR → BaseATK(+), CarryWeight(+)
  AGI → FLEE(+), ASPD(+)
  VIT → DEF(+), MaxHP(+), HealEfficiency(+), StatusResist(+)
  INT → MATK(+), MaxSP(+), HealPower(+)
  DEX → HIT(+), ASPD(+), CastTimeReduction(-), AtkVariance(-)
  LUK → ATK(+), CRIT(+), PerfectDodge(+), StatusResist(+), DropRate(+)

Secondary / Trait (6, renewal only):
  POW → ATK_Penetration(+)
  STA → DEF_Penetration(+)
  WIS → MDEF_Penetration(+)
  SPL → MATK_Penetration(+)
  CON → Heal_Penetration(+), CooldownReduction(-)
  CRT → CritDMG(+)

Stat caps vary by job (from job_stats.yml)
Stat points granted per base level (from statpoint.yml)
```

---

## 10. Equipment System — 14 Item Databases

**Source:** `db/item_db_equip.yml`, `db/item_db_etc.yml`, `db/item_db_usable.yml`, etc.

```
item_db_equip.yml     — weapons, armor, garments, shoes, accessories, shields (~14k items)
item_db_etc.yml       — materials, cards, enchant stones
item_db_usable.yml    — potions, converters, scrolls, foods
item_cash.yml         — cash shop items
item_combos.yml       — card combo sets with bonus effects
item_enchant.yml      — enchant system (random stat addition)
item_reform.yml       — equipment reform system
item_randomopt_db.yml — random options on equipment
item_group_db.yml     — item drop groups used by monsters
item_packages.yml     — bundled item packages

Weapon SubTypes:
  Dagger, 1hSword, 2hSword, 1hSpear, 2hSpear, 1hAxe, 2hAxe, 1hMace, 2hMace
  Staff, Bow, Knuckle, Musical, Instrument, Whip, Book, Claw, Pistol, Rifle
  Shuriken, Grenade, Sling, Shotgun, Scythe

Equip Locations:
  Armor, Weapon, Shield, Garment, Footgear, Accessory1, Accessory2
  Costume_Top/Mid/Bottom, Shadow_Armor/Weapon/Shield/Shoes/Accessory1/Accessory2
```

---

## 11. Card & Combo System

**Source:** `db/item_combos.yml`, card entries in `item_db_etc.yml`

```
Cards slot into equipment with 0-4 slots per item.
Each card provides bonus effects:
  Race_Damage%:   +X% damage vs specific race
  Element_Damage%: +X% damage vs specific element
  Size_Damage%:   +X% damage vs specific size
  Stat_Bonus:     +X to specific stat
  Skill_Bonus:    +X levels to specific skill

Card Combos (item_combos.yml):
  Sets of 2-4 cards from a named set activate additional bonuses.
  Example: Hydra ×2 = +20% vs DemiHuman
  Example: Skeleton Worker ×2 = +15% vs Medium size
```

---

## 12. Companion Systems

**Source:** `db/pet_db.yml`, `db/homunculus_db.yml`, `db/mercenary_db.yml`, `db/elemental_db.yml`

```
Pets (pet_db.yml): capture, feeding, intimacy, stat bonuses, auto-loot
Homunculus (homunculus_db.yml): full combat AI, 8 types, skill trees, equipment
Mercenaries (mercenary_db.yml): temporary summon, fixed stats
Elementals (elemental_db.yml): Sorcerer summons, 4 elements, duration-based
```

---

## 13. Refine / Grade / Enchant / Reform

**Source:** `db/refine.yml`, `db/item_enchant.yml`, `db/item_reform.yml`, `db/enchantgrade.yml`

```
Refine:
  Per weapon level: safe_refine, success_rate, breaking_rate
  Lv1 weapon: safe+7, break at +10
  Lv4 weapon: safe+4, break at +6
  Bonus: ATK+2/refine (weapons), DEF+1/refine (armor)

Grade: D → C → B → A (bonus stats per grade)
Enchant: random stat addition per equipment type
Reform: base stat modification
```

---

## 14. Status Effects

**Source:** `db/re/status.yml`

```
STUN     → cannot act, flee=0
FREEZE   → cannot act, element→Water
STONE    → cannot act, element→Earth, DEF+50%
SLEEP    → cannot act, wake on damage
POISON   → damage over time
CURSE    → STR-50%, LUK-50%
SILENCE  → cannot cast
CONFUSION→ random movement
BLIND    → hit-25%, flee-25%
BERSERK  → ATK+200%, DEF-50%

Plus renewal-only: DEEP_SLEEP, MANA_SEED, NETHERWORLD, etc.
```

---

## 15. Battle Config Mechanics

**Source:** `conf/battle/battle.conf`

```
Attack speed cap:     ~190 ASPD
Multi-hit delay:      200ms per extra hit
Walk delay on damage: 20% (PC) / 100% (monster)
Knockback:            configurable per skill
Auto-spell:           in range only
Arrow decrement:      yes
Ammo check:           required

Defense: pre-renewal vs renewal (see Section 8)
Magic defense: pre-renewal vs renewal
Min/Max hitrate: 5%/100%
```

---

## 16. Group & Party Mechanics

```
Party: shared EXP, party skills, bonus EXP per member
Guild: guild skills (guild_skill_tree.yml), WoE, storage
Battleground: structured PvP teams
Map types: town, field, dungeon, PvP, WoE
Map flags: no_teleport, no_memo, no_save
```

---

## 17. MVP / Boss / Instance Systems

```
MVP: bonus EXP + items to top damage. Exclusive drops.
Boss: status immunity, minion spawns, full skill rotation.
Instance dungeons: private map copies, per-clear rewards.
```

---

## 18. Complete Combat Decision Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       COMBAT OPTIMIZER (sidecar)                            │
│                                                                             │
│  [Layer 0] SERVER MODE DETECTION                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Auto-detect: pre-renewal or renewal from character/monster data     │   │
│  │ All formulas branch based on detected mode                           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  [Layer 1] SITUATION AWARENESS                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Inputs: snapshot data from bridge                                    │   │
│  │  - target: monster name, HP, distance                                │   │
│  │  - self: HP, SP, position, buffs, debuffs                            │   │
│  │  - inventory: equipped items, available items                        │   │
│  │  - actors: nearby monsters, players, NPCs                            │   │
│  │                                                                      │   │
│  │ TargetEngine: resolve target monster from mob_db.yml                 │   │
│  │ StatusEngine: detect active buffs/debuffs on self & target           │   │
│  │ ThreatEngine: assess aggro count, danger level                       │   │
│  │ DangerEngine: check HP/SP thresholds, trigger survival               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  [Layer 2] COMBAT MATH                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ MonsterData: element, size, race, level, HP, DEF, MDEF              │   │
│  │ BotData: level, class, known_skills, inventory, equips              │   │
│  │                                                                      │   │
│  │ WeaponMatcher:                                                      │   │
│  │   1. Filter: equippable weapons (job + level check)                 │   │
│  │   2. Score: ATK × element_modifier × size_modifier                  │   │
│  │   3. Element modifier from attr_fix.yml                              │   │
│  │   4. Size modifier from size_fix.yml                                 │   │
│  │   5. IF no weapon matches → neutral (100%)                          │   │
│  │                                                                      │   │
│  │ SkillMatcher:                                                       │   │
│  │   1. Filter: known skills, SP available, cooldown ready             │   │
│  │   2. Score: base_damage × element_modifier × situational_bonus      │   │
│  │   3. Factor: cast_time, cooldown, range, sp_cost                    │   │
│  │   4. Priority: high_damage > low_sp_cost > combo_prep               │   │
│  │                                                                      │   │
│  │ DamagePredict:                                                      │   │
│  │   Estimate hits to kill: monster_HP / predicted_damage_per_hit      │   │
│  │   Estimate survival: bot_HP / monster_damage_per_hit                │   │
│  │   VERDICT: safe_to_engage / dangerous / suicidal                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  [Layer 3] ACTION GENERATION                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ ActionSequencer:                                                    │   │
│  │   Priority order per cycle:                                         │   │
│  │   1. SURVIVAL: HP < 30% → queue heal/escape                        │   │
│  │   2. EQUIP: not optimal weapon → queue equip <slot>                 │   │
│  │   3. BUFF: buff expired → queue skill_cast                          │   │
│  │   4. ATTACK: skill ready → queue skills_add or auto-attack          │   │
│  │   5. REPOSITION: out of range → queue move                          │   │
│  │                                                                      │   │
│  │ Timing tracker (per bot per session):                               │   │
│  │   last_equip_ms          → 1000ms equip cooldown                    │   │
│  │   last_skill_ms          → current + aftercast delay                │   │
│  │   skill_cooldowns: dict  → {skill_id: expires_at_ms}               │   │
│  │   buff_durations: dict   → {buff_id: expires_at_ms}                 │   │
│  │   next_action_at         → max(last_skill+GCD, cooldown)            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  [Layer 4] PER-CLASS COMBAT TEMPLATES                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Templates describe optimal skill usage per class:                    │   │
│  │                                                                      │   │
│  │ Novice:    auto_attack only                                          │   │
│  │ Swordman:  SM_RECOVERY (HP<50%), SM_BASH (single), SM_MAGNUM (AoE)  │   │
│  │ Mage:      fire_bolt (vs earth), cold_bolt (vs fire),                │   │
│  │            lightning_bolt (vs water), NV_FIRSTAID (emergency)        │   │
│  │ Archer:    skill_double_strafe (spam), skill_arrow_shower (AoE)      │   │
│  │ Thief:     skill_hide (escape), skill_double_attack (auto-proc)      │   │
│  │ Acolyte:   skill_heal (HP<70%), skill_holy_light (vs undead)         │   │
│  │ Merchant:  skill_mammonite (spam SP), skill_cart_revolution (AoE)    │   │
│  │                                                                      │   │
│  │ 2nd class: + cooldown rotation, buff management                     │   │
│  │ 3rd class: + AP management, combo chains                            │   │
│  │ 4th class: + resource rotation, advanced combos                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  [Output] → Action Queue → Bridge → OpenKore Commands::run                 │
│    equip <slot>               — equip weapon                               │
│    skills_add <name> <lvl>    — cast skill                                 │
│    use <item>                 — use potion/converter                       │
│    move <map>                 — route to different zone                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 19. Game Stage Progression

```
EARLY GAME (Base Lv 1-99 / 1st class):
  Weapons:     1-2 slotted, low refine, starter gear
  Skills:      basic attacks, bolt-type spells, self-buffs
  Combat:      auto-attack + 1-2 skill rotation, potion chugging
  Survival:    HP < 50% → heal, HP < 15% → flee/teleport
  Element:     simple elemental matching (bolt types)
  Goal:        1-3 hit kills, SP efficient, minimal gear swaps

MID GAME (Base Lv 99-175 / 2nd-3rd class):
  Weapons:     3-4 slotted, medium refine, basic cards
  Skills:      full skill bar, buff stacking, AoE clearing
  Combat:      skill rotation with cooldowns, party synergy
  Survival:    emergency escape, debuff removal
  Element:     active elemental matching, converters/endow
  Goal:        AoE grinding, efficient SP management, MVP hunting

END GAME (Base Lv 175-265 / 4th class):
  Weapons:     high refine, optimal cards, enchanted, graded
  Skills:     AP system, full rotation, combo execution
  Combat:     precise element/size/race gear per target
  Survival:   auto-buffs, auto-potions, instant escape
  Element:    full weapon swapping per monster, converters active
  Goal:       MVPs, instances, WoE, Battlegrounds
```

---

## 20. Implementation Phases

### Phase 1: Foundation ✅ (Complete)
```
[x] Monster DB loader (2,675 entries via mob_db.yml)
[x] Element modifier table (attr_fix.yml — 40 entries)
[x] Size modifier table (size_fix.yml)
[x] equip/unequip commands allowed in bridge (policies 42-43)
[x] set attackAuto_inLockOnly 0 config fix pipeline
[x] Death loop detection (town-return based)
[x] Cold start routing removed (game engine sole router)
[x] RULE.md compliance verified
```

### Phase 2: Server Mode Detection + Target Resolution (CURRENT)
```
[ ] ServerModeDetector:
    - Read server config from bridge snapshot
    - Detect pre-renewal vs renewal from monster/player stats
    - Return mode string for formula branching
    - File: combat/server_mode.py

[ ] TargetEngine class:
    - Read current target monster from bridge snapshot
    - Look up monster in mob_db.yml by name
    - Resolve element, size, race, level, HP, DEF, MDEF
    - Cache for 3s (avoid repeated lookups per cycle)
    - Return MonsterData or None if no target
    - File: combat/target_engine.py
```

### Phase 3: Weapon Optimization
```
[ ] EquipmentManager class:
    - Read inventory from bridge snapshot
    - Filter: equippable weapons for current job + level
    - Score each weapon: ATK × element_mod × size_mod
    - Return best weapon slot or None if current is optimal
    - Track equip cooldown (1000ms)
    - File: combat/equipment_manager.py
```

### Phase 4: Skill Rotation
```
[ ] SkillEngine class:
    - Load skill_tree.yml per class
    - Load skill_db.yml for cast/cooldown/SP data
    - Select best skill per situation
    - Manage SP pool
    - Track cooldowns per skill
    - File: combat/skill_engine.py
```

### Phase 5: Timing & Combat Sequencing
```
[ ] CombatSequencer class:
    - Track action timing per bot
    - Handle cast_time + aftercast_delay + cooldown
    - Generate action queue entries with proper spacing
    - File: combat/combat_sequencer.py
```

### Phase 6: Per-Class Templates
```
[ ] Combat templates per class:
    - Skill priority list
    - Buff management
    - Emergency responses
    - Element/race/size preferences
    - File: combat/class_templates.py
```

---

## 21. RULE.md Compliance Matrix

| Rule | How It's Satisfied |
|------|-------------------|
| **1. Bridge is LIMITED** | Bridge only sends snapshots + executes commands. All combat logic in Python sidecar. |
| **2. Sidecar handles decisions** | TargetEngine, EquipmentManager, SkillEngine all in sidecar. Zero combat decisions in bridge. |
| **3. Zero hardcoded values** | Monsters from mob_db.yml. Elements from attr_fix.yml. Skills from skill_db.yml. Weapons from item_db_equip.yml. All DB-driven. |
| **4. 100% data flow** | Bridge reports target + inventory. Sidecar enriches with DB data. Every field populated. |
| **5. Agent synergy — no conflicts** | Combat actions use `conflict_key=combat_{bot_id}`. Survival reflex overrides at HP<15% via 300s grace. |
| **6. Reward/punish** | Kill speed tracked via death loop detection. 5 cycle no-kill → easier map. |

---

*End of architecture reference document. Covers all 210 rAthena DB files, 1635 skills, 2675 monsters, 112 job classes, and pre-renewal/renewal branching.*
