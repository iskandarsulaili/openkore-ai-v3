# ═══════════════════════════════════════════
# BEHAVIOR TREE: Priority-Ordered Reflex System
# ═══════════════════════════════════════════
#
# PRO PLAYER DESIGN:
#   Selector nodes (highest priority first):
#     Tier 1 — IMMEDIATE THREAT (sub-50ms): AoE dodge, interrupt cast
#     Tier 2 — LETHAL THREAT (sub-100ms): zonk, HP <15% + aggro flee
#     Tier 3 — CRITICAL SURVIVAL (sub-200ms): HP <12% tele/sit, HP <50% heal
#     Tier 4 — WARNING (throttled): aggro >5, SP low, weight, broken gear
#     Tier 5 — PROACTIVE (throttled): pre-buff, pre-pot, auto-sit, top-off
#
# Each tier is a Selector. Only ONE action fires per tick.
# Global cooldowns: healing_item, movement, buff, sit — can't fire two
# of the same category in the same tick.
#
# KEY IMPROVEMENT OVER FLAT IF-BLOCKS:
#   1. Pre-dodge (#17) now fires BEFORE heal (#1) — lethal AoE > HP recovery
#   2. Interrupt cast (#9) fires BEFORE heal — caster is the root cause
#   3. Pre-buff (#16) only runs when truly out of combat (no aggro)
#   4. Global cooldowns prevent item/skill spam across reflex categories
#   5. Priority is EXPLICIT (tier numbers), not IMPLICIT (file order)

# ── Global cooldown tracker per tick ──
my %_tick_cooldowns = (
    healing_item => 0,   # Can only use ONE heal item per tick
    movement     => 0,   # Can only move/flee once per tick
    sit          => 0,   # Can only sit/stand once per tick
    buff         => 0,   # Can only buff once per tick
);

# ── Helper: check if a cooldown category is available ──
sub _can_fire_category {
    my ($category, $required_cooldown_ms) = @_;
    my $last = $_tick_cooldowns{$category} || 0;
    my $now_ms = _now_ms();
    return ($now_ms - $last) >= $required_cooldown_ms;
}

# ── Helper: set a cooldown category ──
sub _mark_cooldown {
    my ($category) = @_;
    $_tick_cooldowns{$category} = _now_ms();
}

# ═══════════════════════════════════════════
# TIER 1: IMMEDIATE THREAT — SUB-50ms RESPONSE
# These fire FIRST because surviving AoE > healing.
# A pro player dodges before they even think about HP.
# ═══════════════════════════════════════════

# ── TIER 1A: PRE-DODGE (was Reflex #17) ──
# Priority: HIGHEST. If a monster is casting Storm Gust/Meteor Storm,
# no amount of healing will save you — you MUST move.
my $_tier1_dodge_done = 0;
if (!$in_combat || 1) {  # Always check — even in combat, dodge AoE
    if ($monstersList && !$_tier1_dodge_done) {
        for my $monster (@{$monstersList}) {
            next if !$monster;
            my $casting = $monster->{casting} || undef;
            next if !$casting;
            my $dist = _calc_distance($monster, $char);
            next if !defined $dist || $dist > 12;

            # Known dangerous AoE skills that kill before healing
            my @DODGE_SKILLS = qw(
                WZ_STORMGUST WZ_METEORSTORM WZ_HEAVENDRIVE MG_THUNDERSTORM
                WZ_VERMILION NPC_HELLJUDGEMENT NPC_EARTHQUAKE NPC_DARKBREATH
                NPC_PULSESTRIKE NPC_WIDESTUN NPC_WIDEFREEZE NPC_FIREBREATH
            );
            for my $dodge_skill (@DODGE_SKILLS) {
                if ($casting eq $dodge_skill) {
                    if (_should_fire_reflex($_reflex_last_fired{pre_dodge} || 0, 2000)) {
                        $_reflex_last_fired{pre_dodge} = _now_ms();
                        _mark_cooldown('movement');
                        warning "[aiSidecarBridge] bridge_reflex:pre_dodge (monster casting $casting at dist=$dist)\n";
                        eval { Commands::run("flee"); 1 };
                        _http_post_json('/v2/ingest/event', {
                            kind => 'bridge_reflex',
                            reflex => 'pre_dodge',
                            casting_skill => $casting,
                            distance => $dist,
                            hp_ratio => $hp_ratio,
                            timestamp => $now,
                        });
                        $_tier1_dodge_done = 1;
                        last;
                    }
                    last;
                }
            }
            last if $_tier1_dodge_done;
        }
    }
}

# ── TIER 1B: INTERRUPT CAST (was Reflex #9) ──
# Priority: HIGH. A casting monster within 10 tiles should be interrupted
# BEFORE any other action. This is the root cause of incoming damage.
my $_tier1_interrupt_done = 0;
if (!$in_combat || 1) {
    if ($monstersList && !$_tier1_interrupt_done && !$_tier1_dodge_done) {
        for my $monster (@{$monstersList}) {
            next if !$monster;
            my $casting = $monster->{casting} || undef;
            next if !$casting;
            my $dist = _calc_distance($monster, $char);
            if (defined $dist && $dist <= 10) {
                if (_should_fire_reflex($_reflex_last_fired{interrupt_cast} || 0, 1500)) {
                    $_reflex_last_fired{interrupt_cast} = _now_ms();
                    warning "[aiSidecarBridge] bridge_reflex:interrupt_cast (monster casting within 10 tiles)\n";
                    eval { Commands::run("skill Bash 10"); 1 };
                    $_tier1_interrupt_done = 1;
                    last;
                }
            }
        }
    }
}

# ═══════════════════════════════════════════
# TIER 2: LETHAL THREAT — SUB-100ms RESPONSE
# HP is critically low. No thinking, just survive.
# ═══════════════════════════════════════════

# ── TIER 2A: ZONK / DEAD REFLEX (was Reflex #14) ──
# Priority: IMMEDIATE. If HP <= 5, sit now. No delay.
my $_tier2_zonk_done = 0;
if ($hp <= 0 || ($hp > 0 && $hp <= 5)) {
    if (_should_fire_reflex($_reflex_last_fired{zonk} || 0, 2000)) {
        $_reflex_last_fired{zonk} = _now_ms();
        _mark_cooldown('sit');
        warning "[aiSidecarBridge] bridge_reflex:zonk (HP=$hp/$hp_max, map=$map)\n";
        eval { Commands::run("sit"); 1 };
        _http_post_json('/v2/ingest/event', {
            kind => 'bridge_reflex',
            reflex => 'zonk',
            hp => $hp,
            hp_max => $hp_max,
            map => $map,
            timestamp => $now,
        });
        $_tier2_zonk_done = 1;
    }
}

# ── TIER 2B: EMERGENCY FLEE (was Reflex #2) ──
# Priority: CRITICAL. HP <15% AND aggroed. No delay, flee now.
my $_tier2_flee_done = 0;
if ($hp_ratio < 0.15 && $aggro_count > 0 && !$_tier2_zonk_done) {
    if (_should_fire_reflex($_reflex_last_fired{flee} || 0, 1000)) {
        $_reflex_last_fired{flee} = _now_ms();
        _mark_cooldown('movement');
        warning "[aiSidecarBridge] bridge_reflex:emergency_flee (HP=$hp/$hp_max, aggro=$aggro_count)\n";
        eval { Commands::run("flee"); 1 };
        $_tier2_flee_done = 1;
    }
}

# ═══════════════════════════════════════════
# TIER 3: CRITICAL SURVIVAL — SUB-200ms RESPONSE
# HP is low but not yet lethal. Systematic survival actions.
# ═══════════════════════════════════════════

# ── TIER 3A: HIGH AGGRO SURROUND (was Reflex #13) ──
# Priority: CRITICAL. >10 aggro = being swarmed. Flee + teleport combo.
my $_tier3_surround_done = 0;
if ($aggro_count > 10 && !$_tier2_flee_done) {
    if (_should_fire_reflex($_reflex_last_fired{high_aggro_surround} || 0, 3000)) {
        $_reflex_last_fired{high_aggro_surround} = _now_ms();
        _mark_cooldown('movement');
        warning "[aiSidecarBridge] bridge_reflex:high_aggro_surround (aggro=$aggro_count)\n";
        eval { Commands::run("flee"); 1 };
        if ($hp_ratio < 0.25) {
            eval { Commands::run("tele"); 1 };
        }
        _http_post_json('/v2/ingest/event', {
            kind => 'bridge_reflex',
            reflex => 'high_aggro_surround',
            aggro_count => $aggro_count,
            hp_ratio => $hp_ratio,
            map => $map,
            timestamp => $now,
        });
        $_tier3_surround_done = 1;
    }
}

# ── TIER 3B: EMERGENCY TELEPORT / SIT (was Reflex #3) ──
# Priority: HIGH. HP <12% — need to escape or sit to regen.
my $_tier3_tele_done = 0;
if ($hp_ratio < 0.12 && !$aggro_count && !$_tier3_surround_done) {
    if (_should_fire_reflex($_reflex_last_fired{teleport} || 0, 3000)) {
        $_reflex_last_fired{teleport} = _now_ms();
        if ($aggro_count > 0) {
            _mark_cooldown('movement');
            warning "[aiSidecarBridge] bridge_reflex:emergency_teleport (HP=$hp/$hp_max, aggro=$aggro_count)\n";
            eval { Commands::run("tele"); 1 };
        } else {
            _mark_cooldown('sit');
            warning "[aiSidecarBridge] bridge_reflex:emergency_sit_regen (HP=$hp/$hp_max)\n";
            eval { Commands::run("sit"); 1 };
        }
        $_tier3_tele_done = 1;
    }
}

# ── TIER 3C: EMERGENCY HEAL (was Reflex #1) ──
# Priority: HIGH. HP <50% — use healing items/skills immediately.
# This now comes AFTER dodge and interrupt checks (Tier 1).
my $_tier3_heal_done = 0;
if ($hp_ratio < 0.50 && $hp > 0 && !$_tier3_surround_done && !$_tier3_tele_done) {
    my $heal_triggered = 0;
    _update_heal_cache();

    # Try config-pushed items first (dynamic, class-aware)
    for my $item_name (@_heal_items) {
        $item_name = _trim($item_name);
        next if !$item_name;
        my $item = eval { Actor::Item::get($item_name) };
        if ($item && $item->{amount} && $item->{amount} > 0) {
            warning "[aiSidecarBridge] bridge_reflex:emergency_heal (HP=$hp/$hp_max=$hp_ratio, item=$item_name qty=$item->{amount})\n";
            eval { Commands::run("is $item_name"); 1 };
            $heal_triggered = 1;
            _mark_cooldown('healing_item');
            $_tier3_heal_done = 1;
            last;
        }
    }

    # Try config-pushed skills if no items available
    if (!$heal_triggered) {
        for my $skill_name (@_heal_skills) {
            $skill_name = _trim($skill_name);
            next if !$skill_name;
            my $skill = eval { Skill::get($skill_name) };
            if ($skill && $skill->{level} && $skill->{level} > 0 && $sp > 0) {
                warning "[aiSidecarBridge] bridge_reflex:emergency_heal_skill (HP=$hp/$hp_max, skill=$skill_name lv=$skill->{level}, SP=$sp)\n";
                eval { Commands::run("skill $skill_name 1"); 1 };
                $heal_triggered = 1;
                _mark_cooldown('buff');  # Skills use the buff cooldown category
                $_tier3_heal_done = 1;
                last;
            }
        }
    }

    # HARD CODED FALLBACK: White Potion (absolute safety net)
    if (!$heal_triggered) {
        my $fallback = eval { Actor::Item::get($HARDCODED_FALLBACK_ITEM) };
        if ($fallback && $fallback->{amount} && $fallback->{amount} > 0) {
            warning "[aiSidecarBridge] bridge_reflex:emergency_heal_fallback (HP=$hp/$hp_max, item=$HARDCODED_FALLBACK_ITEM)\n";
            eval { Commands::run("is $HARDCODED_FALLBACK_ITEM"); 1 };
            $heal_triggered = 1;
            _mark_cooldown('healing_item');
            $_tier3_heal_done = 1;
        }
    }

    # If NO healing resources at all: trigger emergency survival
    if (!$heal_triggered && $hp_ratio < 0.50) {
        if (_should_fire_reflex($_reflex_last_fired{no_heal} || 0, 10000)) {
            $_reflex_last_fired{no_heal} = _now_ms();
            warning "[aiSidecarBridge] bridge_reflex:emergency_no_heal (HP=$hp/$hp_max, map=$map, job=$job_name, lvl=$base_level/$job_level)\n";

            _http_post_json('/v2/ingest/event', {
                kind => 'bridge_reflex',
                reflex => 'emergency_no_heal',
                hp_ratio => $hp_ratio,
                hp => $hp,
                max_hp => $hp_max,
                sp_ratio => $sp_ratio,
                sp => $sp,
                aggro_count => $aggro_count,
                weight_ratio => $weight_ratio,
                map => $map,
                job_name => $job_name,
                base_level => $base_level,
                job_level => $job_level,
                heal_items_cached => scalar(@_heal_items),
                heal_skills_cached => scalar(@_heal_skills),
                timestamp => _now_ms(),
            });

            # IMMEDIATE EMERGENCY SURVIVAL: flee if aggro, teleport if no aggro
            if ($aggro_count > 0) {
                _mark_cooldown('movement');
                eval { Commands::run("flee"); 1 };
            } elsif ($hp_ratio < 0.30) {
                eval { Commands::run("tele"); 1 };
            } elsif ($hp_ratio < 0.15) {
                _mark_cooldown('sit');
                eval { Commands::run("sit"); 1 };
            }
        }
    }

    # Bot-to-bot cooperation request if no heal and aggro
    if (!$heal_triggered && $hp_ratio < 0.50 && $aggro_count > 0) {
        if (_should_fire_reflex($_reflex_last_fired{bot_request} || 0, 5000)) {
            $_reflex_last_fired{bot_request} = _now_ms();
            warning "[aiSidecarBridge] bridge_reflex:bot_cooperation_request (HP=$hp/$hp_max, aggro=$aggro_count)\n";
            _http_post_json('/v2/ingest/event', {
                kind => 'bridge_reflex',
                reflex => 'bot_cooperation_request',
                hp_ratio => $hp_ratio,
                hp => $hp,
                max_hp => $hp_max,
                aggro_count => $aggro_count,
                map => $map,
                base_level => $base_level,
                job_name => $job_name,
                timestamp => _now_ms(),
            });
        }
    }
}

# ═══════════════════════════════════════════
# TIER 4: WARNING — THROTTLED NOTIFICATIONS
# These inform the sidecar but don't require instant action.
# ═══════════════════════════════════════════

# ── TIER 4A: AGGRO WARNING (was Reflex #4) ──
# Notify sidecar when heavily aggroed (>5 attackers).
my $_tier4_warn_done = 0;
if ($aggro_count > 5) {
    if (_should_fire_reflex($_reflex_last_fired{aggro_warning} || 0, 5000)) {
        $_reflex_last_fired{aggro_warning} = _now_ms();
        warning "[aiSidecarBridge] bridge_reflex:aggro_warning (aggro=$aggro_count)\n";
        _http_post_json('/v2/ingest/event', {
            kind => 'bridge_reflex',
            reflex => 'aggro_warning',
            aggro_count => $aggro_count,
            hp_ratio => $hp_ratio,
            map => $map,
            timestamp => $now,
        });
        $_tier4_warn_done = 1;
    }
}

# ── TIER 4B: LOW SP (was Reflex #5) ──
# Notify sidecar when SP is critically low.
my $_tier4_sp_done = 0;
if ($sp_ratio < 0.15) {
    if (_should_fire_reflex($_reflex_last_fired{low_sp} || 0, 10000)) {
        $_reflex_last_fired{low_sp} = _now_ms();
        warning "[aiSidecarBridge] bridge_reflex:low_sp (SP=$sp/$sp_max, ratio=$sp_ratio)\n";
        _http_post_json('/v2/ingest/event', {
            kind => 'bridge_reflex',
            reflex => 'low_sp',
            sp_ratio => $sp_ratio,
            sp => $sp,
            max_sp => $sp_max,
            timestamp => $now,
        });
        $_tier4_sp_done = 1;
    }
}

# ── TIER 4C: GM / ADMIN DETECTION (was Reflex #6) ──
# Detect GM/Admin players within 15 tiles, switch to manual.
my $_tier4_gm_done = 0;
if (!$playersList) {
    # Skip if no players — already handled by TIER 1-3 logic
} else {
    my $gm_detected = 0;
    for my $player (@{$playersList}) {
        next if !$player;
        my $pname = $player->{name} || '';
        next if $pname eq '';
        if ($pname =~ /GM|GameMaster|Admin|Support/i) {
            my $dist = _calc_distance($player, $char);
            if (defined $dist && $dist <= 15) {
                $gm_detected = 1;
                last;
            }
        }
    }
    if ($gm_detected) {
        if (_should_fire_reflex($_reflex_last_fired{gm_detected} || 0, 60000)) {
            $_reflex_last_fired{gm_detected} = _now_ms();
            warning "[aiSidecarBridge] bridge_reflex:gm_detected (GM/Admin player within 15 tiles)\n";
            eval { Commands::run("ai manual"); 1 };
            _http_post_json('/v2/ingest/event', {
                kind => 'bridge_reflex',
                reflex => 'gm_detected',
                message => 'GM/Admin player detected within 15 tiles, AI switched to manual',
                timestamp => $now,
            });
            $_tier4_gm_done = 1;
        }
    }
}

# ── TIER 4D: PARTY MEMBER LOW HP (was Reflex #12) ──
# Notify sidecar if any party member has critically low HP.
my $_tier4_party_done = 0;
if ($playersList) {
    for my $player (@{$playersList}) {
        next if !$player;
        my $pname = $player->{name} || '';
        next if $pname eq '';
        next if $char && defined $char->{name} && $pname eq $char->{name};

        my $player_hp = $player->{hp} || 0;
        my $player_hp_max = $player->{hp_max} || 1;
        my $player_hp_ratio = ($player_hp_max > 0) ? $player_hp / $player_hp_max : 1;

        if ($player_hp_ratio < 0.20) {
            my $dist = _calc_distance($player, $char);
            next if !defined $dist || $dist > 20;

            if (_should_fire_reflex($_reflex_last_fired{party_low_hp} || 0, 10000)) {
                $_reflex_last_fired{party_low_hp} = _now_ms();
                warning "[aiSidecarBridge] bridge_reflex:party_low_hp (player=$pname HP=$player_hp/$player_hp_max=$player_hp_ratio, dist=$dist)\n";
                _http_post_json('/v2/ingest/event', {
                    kind => 'bridge_reflex',
                    reflex => 'party_low_hp',
                    player_name => $pname,
                    player_hp => $player_hp,
                    player_hp_max => $player_hp_max,
                    player_hp_ratio => $player_hp_ratio,
                    distance => $dist,
                    timestamp => $now,
                });
                $_tier4_party_done = 1;
                last;  # Only report first low-HP party member per cycle
            }
        }
    }
}

# ── TIER 4E: WEIGHT WARNING (was Reflex #7) ──
# Notify sidecar when weight exceeds 85%.
if ($weight_ratio > 0.85) {
    if (_should_fire_reflex($_reflex_last_fired{weight_warning} || 0, 30000)) {
        $_reflex_last_fired{weight_warning} = _now_ms();
        warning "[aiSidecarBridge] bridge_reflex:weight_warning (weight=$weight/$weight_max, ratio=$weight_ratio)\n";
        _http_post_json('/v2/ingest/event', {
            kind => 'bridge_reflex',
            reflex => 'weight_warning',
            weight_ratio => $weight_ratio,
            weight => $weight,
            max_weight => $weight_max,
            timestamp => $now,
        });
    }
}

# ── TIER 4F: EQUIPMENT BROKEN (was Reflex #8) ──
# Notify sidecar when equipped item is damaged/broken.
if ($char->{equipment}) {
    my $broken_found = 0;
    for my $slot (keys %{$char->{equipment}}) {
        my $item = $char->{equipment}{$slot};
        next if !$item;
        if ($item->{broken} || (defined $item->{damage} && $item->{damage} > 0)) {
            $broken_found = 1;
            last;
        }
    }
    if ($broken_found) {
        if (_should_fire_reflex($_reflex_last_fired{equipment_broken} || 0, 60000)) {
            $_reflex_last_fired{equipment_broken} = _now_ms();
            warning "[aiSidecarBridge] bridge_reflex:equipment_broken (broken equipment detected)\n";
            _http_post_json('/v2/ingest/event', {
                kind => 'bridge_reflex',
                reflex => 'equipment_broken',
                message => 'Broken equipment detected',
                timestamp => $now,
            });
        }
    }
}

# ── TIER 4G: DEATH SPIKE (was Reflex #15) ──
# Track death count and notify sidecar if deaths spike.
if ($death_count > 0 && $death_count % 5 == 0) {
    if (_should_fire_reflex($_reflex_last_fired{death_spike} || 0, 120000)) {
        $_reflex_last_fired{death_spike} = _now_ms();
        warning "[aiSidecarBridge] bridge_reflex:death_spike (deaths=$death_count, map=$map)\n";
        _http_post_json('/v2/ingest/event', {
            kind => 'bridge_reflex',
            reflex => 'death_spike',
            death_count => $death_count,
            map => $map,
            timestamp => $now,
        });
    }
}

# ═══════════════════════════════════════════
# TIER 5: PROACTIVE — THROTTLED OPTIMIZATION
# These improve efficiency but are never urgent.
# ═══════════════════════════════════════════

# ── TIER 5A: PRE-BUFF (was Reflex #16) ──
# Priority: LOW. Pre-buff before engaging only when truly safe.
my $_tier5_prebuff_done = 0;
if (!$in_combat && $hp_ratio > 0.8 && $sp_ratio > 0.3 && !$aggro_count) {
    if (_should_fire_reflex($_reflex_last_fired{pre_buff} || 0, 15000)) {
        $_reflex_last_fired{pre_buff} = _now_ms();
        _mark_cooldown('buff');
        my @buffs = (
            "skill Twohand Quicken 1",    # Knight ASPD buff
            "skill Increase AGI 10",      # Acolyte/Priest
            "skill Blessing 10",          # Acolyte/Priest
            "skill Magnificat 5",         # Priest SP regen
            "skill Kyrie Eleison 10",     # Priest shield
            "skill Improve Concentration 10", # Swordsman
            "skill Enchant Poison 5",     # Assassin
            "skill Owl's Eye 10",         # Archer
            "skill Vulture's Eye 10",     # Archer
            "skill Energy Coat 5",        # Mage
        );
        for my $buff (@buffs) {
            my ($cmd, $skill_name) = $buff =~ /^skill\s+(.+?)\s+\d+$/;
            next if !$skill_name;
            my $skill = eval { Skill::get($skill_name) };
            if ($skill && $skill->{level} && $skill->{level} > 0) {
                my $already_active = 0;
                if ($char->{buffs} && ref $char->{buffs} eq 'HASH') {
                    $already_active = 1 if exists $char->{buffs}{$skill_name};
                }
                if (!$already_active) {
                    _random_action_delay();
                    eval { Commands::run($buff); 1 };
                    $_tier5_prebuff_done = 1;
                    last;  # One buff per tick
                }
            }
        }
    }
}

# ── TIER 5B: PRE-POT (was Reflex #10) ──
# Priority: LOW. Pre-heal before engaging a boss.
my $_tier5_prepot_done = 0;
if (!$in_combat || 1) {
    my @BOSS_IDS = (1038, 1046, 1049, 1059, 1086, 1087, 1088, 1112, 1115, 1147, 1150, 1159, 1205, 1272, 1312, 1313, 1511, 1630, 1639, 1719, 1751, 1871, 1874);
    my %BOSS_LOOKUP = map { $_ => 1 } @BOSS_IDS;
    if ($monstersList) {
        my $boss_nearby = 0;
        for my $monster (@{$monstersList}) {
            next if !$monster;
            my $name_id = $monster->{nameID} || 0;
            next if !$name_id;
            if ($BOSS_LOOKUP{$name_id}) {
                my $dist = _calc_distance($monster, $char);
                if (defined $dist && $dist <= 15) {
                    $boss_nearby = 1;
                    last;
                }
            }
        }
        if ($boss_nearby && $hp_ratio > 0.9) {
            if (_should_fire_reflex($_reflex_last_fired{pre_pot} || 0, 5000)) {
                $_reflex_last_fired{pre_pot} = _now_ms();
                _mark_cooldown('healing_item');
                _update_heal_cache();
                my $healed = 0;
                for my $item_name (@_heal_items) {
                    my $item = eval { Actor::Item::get($item_name) };
                    if ($item) {
                        warning "[aiSidecarBridge] bridge_reflex:pre_pot (boss within 15 tiles, HP=$hp/$hp_max, item=$item_name)\n";
                        eval { Commands::run("is $item_name"); 1 };
                        $healed = 1;
                        last;
                    }
                }
                if (!$healed) {
                    warning "[aiSidecarBridge] bridge_reflex:pre_pot_no_items (boss within 15 tiles, HP=$hp/$hp_max)\n";
                }
                $_tier5_prepot_done = 1;
            }
        }
    }
}

# ── TIER 5C: AUTO-SIT REGEN (was Reflex #18) ──
# Priority: LOW. Auto-sit when out of combat and HP/SP is low.
my $_tier5_autosit_done = 0;
if (!$in_combat && !$aggro_count && $hp_ratio < 0.6 && $hp > 0) {
    if (_should_fire_reflex($_reflex_last_fired{auto_sit} || 0, 5000)) {
        $_reflex_last_fired{auto_sit} = _now_ms();
        _mark_cooldown('sit');
        my $ai_top = @ai_seq ? $ai_seq[0] : '';
        if ($ai_top ne 'sit') {
            _random_action_delay();
            eval { Commands::run("sit"); 1 };
            $_tier5_autosit_done = 1;
        }
    }
}

# ── TIER 5D: POTION TOP-OFF (was Reflex #19) ──
# Priority: LOW. Top off HP when out of combat and HP 30-80%.
my $_tier5_topoff_done = 0;
if (!$in_combat && !$aggro_count && $hp_ratio > 0.3 && $hp_ratio < 0.8 && $hp > 0) {
    if (_should_fire_reflex($_reflex_last_fired{top_off} || 0, 10000)) {
        $_reflex_last_fired{top_off} = _now_ms();
        _mark_cooldown('healing_item');
        _update_heal_cache();
        for my $item_name (@_heal_items) {
            $item_name = _trim($item_name);
            next if !$item_name;
            my $item = eval { Actor::Item::get($item_name) };
            if ($item && $item->{amount} && $item->{amount} > 0) {
                _random_action_delay();
                eval { Commands::run("is $item_name"); 1 };
                $_tier5_topoff_done = 1;
                last;
            }
        }
    }
}

