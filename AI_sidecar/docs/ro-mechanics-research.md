# Ragnarok Online Classic — Complete Mechanics Research

> Compiled for OpenKore AI behavior profile system. Covers Pre-Renewal (Classic) RO mechanics.

---

## 1. Class & Skill System (Pre-Renewal)

### 1st Classes (Novice →)

| Class | Type | Weapons | Key Skills |
|-------|------|---------|------------|
| **Swordsman** | Melee | Sword, 2H-Sword, Spear | Bash, Magnum Break, Endure, Provoke, Bowling Bash, Spear Boomerang |
| **Mage** | Magic | Staff | Fire Bolt, Cold Bolt, Lightning Bolt, Fire Wall, Frost Diver, Soul Strike, Napalm Beat, Energy Coat |
| **Archer** | Ranged | Bow | Double Strafe, Arrow Shower, Improve Concentration, Owl's Eye, Vulture's Eye |
| **Acolyte** | Support/Melee | Mace, Staff | Heal, Increase AGI, Blessing, Teleport, Warp Portal, Holy Light, Ruwach, Pneuma, Aqua Benedicta |
| **Merchant** | Economy | Axe, Sword | Discount, Overcharge, Identify, Pushcart, Vending |
| **Thief** | Melee/Agile | Dagger, Sword | Double Attack, Steal, Hiding, Detect, Envenom, Poison React |

### 2-1 Classes (Swordsman →)

| Class | Type | Key Skills |
|-------|------|-----------|
| **Knight** | Heavy Melee | Bowling Bash, Brandish Spear, Spear Boomerang, Spear Stab, Cavalier Mastery, Peco Peco Riding |
| **Wizard** | AoE Magic | Fire Ball, Fire Pillar, Meteor Storm, Frost Nova, Storm Gust, Lord of Vermilion, Quagmire, Safety Wall |
| **Hunter** | Ranged/Dex | Blitz Beat, Steel Crow, Falconry Mastery, Skid Trap, Land Mine Trap, Freezing Trap, Remove Trap |
| **Priest** | Heal/Support | Resurrection, Gloria, Magnificat, Impositio Manus, Suffragium, Meditatio, Lex Divina, Turn Undead |
| **Blacksmith** | Craft/Melee | Weapon Perfection, Over Thrust, Max Over Thrust, Adrenaline Rush, Weapon & Armor Forging, Enchant Poison |
| **Assassin** | Stealth/Dmg | Grimtooth, Sonic Blow, Venom Splasher, Cloaking, Enchant Deadly Poison, Katar Mastery |

### 2-2 Classes (Alternative)

| Class | Type | Key Skills |
|-------|------|-----------|
| **Crusader** | Holy Tank | Holy Cross, Grand Cross, Shield Boomerang, Reflect Shield, Devotion |
| **Sage** | Support Magic | Endow (elemental weapon), Dispel, Deluge, Volcano, Violet, Abracadabra, Spell Breaker |
| **Bard** | Support Songs | Musical Lesson, Dissonance, Assassin Cross of Sunset, A Drum on the Battlefield, Battle Theme |
| **Dancer** | Support Dance | Dancing Lesson, Slow Grace, Slinging Arrow, Hip Shaker, Rumba Musical |
| **Alchemist** | Potions/Craft | Potion Making, Acid Demonstration, Full Chemical Protection, Summon Flora, Bidpot |
| **Rogue** | Steal/Mimic | Steal, Back Stab, Snatch, Plagiarism, Intimidate, Strip Weapon/Shield/Armor/Helm |

---

## 2. Core Game Mechanics

### 2.1 Element System (10 Elements, 4 Levels)
- Elements: Neutral, Water, Earth, Fire, Wind, Poison, Holy, Shadow, Ghost, Undead
- Each element has 4 levels (1-4) affecting damage multipliers
- Weapon can be endowed via skills (Endow, Enchant Poison) or elemental weapons
- Critical hits ignore elemental resistance but not racial/defensive modifiers
- Classic table: Fire > Earth > Wind > Water > Fire (cyclical). Holy > Shadow > Ghost > Poison > Neutral

### 2.2 Size System (3 Sizes)
- **Small** (Dagger, Katars, Maces effective), **Medium** (Swords, Axes, Spears), **Large** (2H-Swords, Spears, Bows)
- Weapon vs Size damage penalties vary by weapon type vs target size

### 2.3 Race System (10 Races)
- Races: Formless, Undead, Brute, Plant, Fish, Demon, Demi-Human, Angel, Dragon, Insect
- Cards and equipment target specific races (+% damage, -% damage from)

### 2.4 Refine System
- NPC upgrade: +0 to +10 maximum with decreasing success rates
- Safe to +4, risk of destruction above +4

### 2.5 Card System
- Cards slotted in equipment with ATK/MATK/DEF/stat/skill modifiers
- Race, size, element, and specific monster cards

### 2.6 Status Effects
- Poison, Stun, Freeze, Petrify, Sleep, Silence, Blind, Curse, Confusion

### 2.7 MVP System
- Boss monsters on spawn timers (2-8h), top damage dealer gets loot
- Known MVPs: Angelo, Drake, Eddga, Phreeoni, Maya, Osiris, Baphomet, Orc Hero/Hero, etc.

### 2.8 PVP System
- Player vs Player with damage reduction, no EXP penalty on death

### 2.9 GVG / War of Emperium
- Guild castle siege, Emperium breaking, guild claiming territory

### 2.10 Party System
- Max 12 members with EXP sharing, role differentiation

### 2.11 Guild System
- Levels 1-50 with guild skills, donation, WoE participation

### 2.12 Trade & Economy
- Vending, Buying Shop, Player Trading, Mail, Auction, NPC Shops

### 2.13 Storage Systems
- Kafra Storage, Cart Storage, Guild Storage

### 2.14 Warp & Transport
- Kafra Warp, Fly Wing (random), Butterfly Wing (save point), Warp Portal, NPC Warps, Airship, Ship

### 2.15 Pet System
- Catchable via taming items, provides stat/loot/potion auto bonuses

### 2.16 Marriage System
- Formal wedding, ring bonuses, adoption system

### 2.17 Homunculus System (Alchemist)
- AI pet fighter with own stats/skills, evolvable

### 2.18 Mount System
- Peco Peco (Knight/Crusader), increased movement speed

### 2.19 Quest System
- Job Change, Repeatable EXP, Access, Treasure Cache, Book Reading

### 2.20 NPC Interaction
- Talk, Buy/Sell, Identify, Repair, Refine, Storage, Warp

---

## 3. OpenKore Console Commands

### Movement: `move`, `follow`, `sit`, `stand`, `north`, `south`, `east`, `west`
### Combat: `attack`, `skill`, `skills`, `spells`, `look`, `tank`
### Economy: `buy`, `sell`, `store`, `storage`, `vender`, `openshop`, `closeshop`, `autobuy`, `autosell`, `autostorage`, `cart`, `price`
### Communication: `pm`, `chat`, `party`, `guild`, `friend`, `reply`/`r`, `ignore`
### Inventory: `i`, `eq`, `eqsw`, `uneq`, `drop`, `take`, `identify`, `repair`, `refine`, `card`, `arrowcraft`
### NPC: `talk`, `talknpc`, `send`, `c`, `nc`
### Quest: `quest`, `achieve`, `reputation`
### Social: `party`, `guild`, `deal`, `friend`, `mail`, `rodex`, `booking`
### Utility: `ai`, `aiv`, `relog`, `quit`, `respawn`, `revive`, `tele`, `conf`, `eval`, `pause`, `timeout`, `exp`, `where`, `who`, `whoami`, `ip`, `dump`, `switchconf`, `log`, `clearlog`, `help`, `version`, `portals`, `is`, `im`, `al`, `as`, `au`, `bl`, `cl`, `cm`, `dl`, `st`, `vl`, `pet`, `homun`, `merc`, `falcon`, `pecopeco`, `cash`, `cashbuy`, `cook`, `starplace`, `memorial`, `tank`, `searchstore`, `camp`, `clan`, `analysis`, `connect`, `charselect`, `attack`

---

## 4. OpenKore Macro Plugin

### Conditions: `auto`, `call`, `console /p/`, `/t`, `/p`, `/g`, `/pm`, `/nc`, `/is`, `/pri`, `timeout`, `var`
### Actions: `do`, `do conf`, `call`, `pause`, `relog`, `log`, `set`, `sete`, `inventory`, `cart`, `storage`, `isUser`, `save`, `break`, `rand`, `callsub`, `return`
### Variables: `$.map`, `$.pos`, `$.hp`, `$.sp`, `$.hpMax`, `$.spMax`, `$.weight`, `$.maxWeight`, `$.zeny`, `$.lvl`, `$.joblvl`, `$.job`, `$.time`, `$.target`, `$.targetHP`, `$.targetDist`, `$.attacking`, `$.sitting`, `$.partyMaster`, `$.partyCount`, `$.party`, `$.guild`, `$.inventorySize`, `$.cartSize`, `$.storageSize`

---

## 5. doCommand Blocks

```
doCommand <cmd> { hp <...> sp <...> weight <...> whenStatusActive/Inactive whenNotGround inMap
                  inInventory <item> <qty> timeout monsters notMonsters equipped stopWhenHit
                  inLockOnly notWhileTalking notInTown disabled }
```

---

## 6. Behavior Profile Architecture

Each profile uses `ExperienceDatabase` for self-learning via `best_action()`.

| File | Context | Coverage |
|------|---------|----------|
| `combat_agent.py` | `"combat"` | Element/Size/Race/MVP/PVP/GVG/Party/Auto-skill |
| `economy_agent.py` | `"economy"` | Vending/Buying/Trading/Price/Storage/Craft/Refine/Cards |
| `navigation_agent.py` | `"navigation"` | Warps/Wings/Routes/Portals/Kafra/Follow |
| `questing_agent.py` | `"quest"` | Accept/Complete/Turn-in/Job change |
| `safety_agent.py` | `"survival"` | Emergency/Death/Weight/Potions/Repair |
| `social_agent.py` | `"social"` | Party/Guild/Auto-response/Friends |
| `crafting_agent.py` | `"craft"` | Forge/Potions/Cooking/Arrows/Elements |
| `job_agent.py` | `"job"` | Job change/Stats/Skills/Level tracking |
