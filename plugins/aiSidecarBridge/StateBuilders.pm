package aiSidecarBridge::StateBuilders;

use strict;
use warnings;

# Suppress "used only once" warnings for OpenKore globals accessed via %::
no warnings qw(once);

=head1 NAME

aiSidecarBridge::StateBuilders - 17 specialized state builders for OpenKore

=head1 SYNOPSIS

    use aiSidecarBridge::StateBuilders;

    my $builders = aiSidecarBridge::StateBuilders->new();

    my $char_state    = $builders->build_character_state();
    my $inventory     = $builders->build_inventory_state();
    my $map_state     = $builders->build_map_state();
    my $party_state   = $builders->build_party_state();
    # ... etc

    # Build all states into a single combined hashref
    my $all = $builders->build_all_states();

=head1 DESCRIPTION

Provides 17 specialized state builder methods that extract structured
data from OpenKore globals (C<$char>, C<$field>, C<%monsters>, etc.).

Each builder returns a plain hashref suitable for JSON encoding.

These are ADDITIONS to the existing C<_build_snapshot_payload()> method
in C<aiSidecarBridge.pl>. They provide finer-grained, domain-specific
state views for the sidecar.

=cut

our $VERSION = '1.0.0';

sub new {
    my ($class, %args) = @_;
    my $self = {
        _debug_log_cb  => $args{debug_log_cb}   || sub { },
        _max_items     => $args{max_items}      || 200,
        _max_actors    => $args{max_actors}     || 24,
    };
    bless $self, $class;
    return $self;
}

=head2 build_all_states

Returns a single hashref containing ALL 17 states combined, keyed by
state type. Useful for a single snapshot send.

    {
        character      => { ... },
        inventory      => { ... },
        map            => { ... },
        party          => { ... },
        guild          => { ... },
        buff           => { ... },
        pet            => { ... },
        homunculus     => { ... },
        mercenary      => { ... },
        mount          => { ... },
        equipment      => { ... },
        npc_dialogue   => { ... },
        quest          => { ... },
        market         => { ... },
        environment    => { ... },
        ground_items   => { ... },
        instance       => { ... },
    }

=cut

sub build_all_states {
    my ($self) = @_;
    return {
        character    => $self->build_character_state(),
        inventory    => $self->build_inventory_state(),
        map          => $self->build_map_state(),
        party        => $self->build_party_state(),
        guild        => $self->build_guild_state(),
        buff         => $self->build_buff_state(),
        pet          => $self->build_pet_state(),
        homunculus   => $self->build_homunculus_state(),
        mercenary    => $self->build_mercenary_state(),
        mount        => $self->build_mount_state(),
        equipment    => $self->build_equipment_state(),
        npc_dialogue => $self->build_npc_dialogue_state(),
        quest        => $self->build_quest_state(),
        market       => $self->build_market_state(),
        environment  => $self->build_environment_state(),
        ground_items => $self->build_ground_items_state(),
        instance     => $self->build_instance_state(),
    };
}

=head2 build_character_state

Extracts character vitals, stats, position, and progression.

Keys: name, job_id, job_name, base_level, job_level, hp, hp_max,
hp_ratio, sp, sp_max, sp_ratio, weight, weight_max, weight_ratio,
zeny, str, agi, vit, int_, dex, luk, atk_min, atk_max, matk_min,
matk_max, def, mdef, hit, flee, crit, aspd, attack_power, sitting,
dead, map, x, y, sex, hair_style, hair_color, weapon, shield,
head_top, head_mid, head_bottom, robe

=cut

sub build_character_state {
    my ($self) = @_;
    my $char = $::char;
    return {} unless $char;

    my $state = {
        name       => _scalarize($char->{name}),
        job_id     => _scalarize($char->{jobID}),
        job_name   => _scalarize($char->{jobName}),
        base_level => _scalarize($char->{level}),
        job_level  => _scalarize($char->{level_job}),
        hp         => _scalarize($char->{hp}),
        hp_max     => _scalarize($char->{hp_max}),
        hp_ratio   => ($char->{hp_max} && $char->{hp_max} > 0)
            ? sprintf('%.4f', ($char->{hp} || 0) / $char->{hp_max}) : 0,
        sp         => _scalarize($char->{sp}),
        sp_max     => _scalarize($char->{sp_max}),
        sp_ratio   => ($char->{sp_max} && $char->{sp_max} > 0)
            ? sprintf('%.4f', ($char->{sp} || 0) / $char->{sp_max}) : 0,
        weight         => _scalarize($char->{weight}),
        weight_max     => _scalarize($char->{weight_max}),
        weight_ratio   => ($char->{weight_max} && $char->{weight_max} > 0)
            ? sprintf('%.4f', ($char->{weight} || 0) / $char->{weight_max}) : 0,
        zeny           => _scalarize($char->{zeny}),
        str            => _scalarize($char->{str}),
        agi            => _scalarize($char->{agi}),
        vit            => _scalarize($char->{vit}),
        int            => _scalarize($char->{int}),
        dex            => _scalarize($char->{dex}),
        luk            => _scalarize($char->{luk}),
        atk_min        => _scalarize($char->{attack}),
        atk_max        => _scalarize($char->{attack_max} || $char->{attack}),
        matk_min       => _scalarize($char->{matk_min}),
        matk_max       => _scalarize($char->{matk_max}),
        def            => _scalarize($char->{def}),
        mdef           => _scalarize($char->{mdef}),
        hit            => _scalarize($char->{hit}),
        flee           => _scalarize($char->{flee}),
        crit           => _scalarize($char->{crit}),
        aspd           => _scalarize($char->{aspd}),
        attack_power   => _scalarize($char->{attack} || $char->{atk}),
        sitting        => $char->{sitting} ? 1 : 0,
        dead           => ($char->{hp} || 0) <= 0 ? 1 : 0,
        map            => _scalarize($char->{map}),
        x              => _pos_x($char),
        y              => _pos_y($char),
        sex            => _scalarize($char->{sex}),
        hair_style     => _scalarize($char->{hair}),
        hair_color     => _scalarize($char->{hairColor}),
        weapon         => _scalarize($char->{weapon}),
        shield         => _scalarize($char->{shield}),
        head_top       => _scalarize($char->{head}),
        head_mid       => _scalarize($char->{headMid}),
        head_bottom    => _scalarize($char->{headBottom}),
        robe           => _scalarize($char->{robe}),
    };
    return $state;
}

=head2 build_inventory_state

Extracts inventory items with name, amount, type, and slot info.

Keys: item_count, weight, weight_max, weight_ratio, zeny, items (array)

Each item: name_id, name, type, amount, slot, equipped, refine, cards,
identified, broken

=cut

sub build_inventory_state {
    my ($self) = @_;
    my $char = $::char;
    my $state = {
        item_count   => 0,
        zeny         => _scalarize($char->{zeny}),
        weight       => _scalarize($char->{weight}),
        weight_max   => _scalarize($char->{weight_max}),
        weight_ratio => ($char->{weight_max} && $char->{weight_max} > 0)
            ? sprintf('%.4f', ($char->{weight} || 0) / $char->{weight_max}) : 0,
        items        => [],
    };
    return $state unless $char && $char->{inventory} && ref($char->{inventory}) eq 'ARRAY';

    my $max = $self->{_max_items} || 200;
    my @items;
    for my $item (@{$char->{inventory}}) {
        next unless $item;
        last if scalar(@items) >= $max;
        push @items, {
            name_id   => _scalarize($item->{nameID}),
            name      => _scalarize($item->{name}),
            type      => _scalarize($item->{type}),
            type_name => _item_type_name($item->{type}),
            amount    => _scalarize($item->{amount}),
            slot      => _scalarize($item->{slot}),
            equipped  => $item->{equipped} ? 1 : 0,
            refine    => _scalarize($item->{refine}),
            cards     => [
                grep { defined && $_ > 0 } (
                    _scalarize($item->{card1}),
                    _scalarize($item->{card2}),
                    _scalarize($item->{card3}),
                    _scalarize($item->{card4})
                )
            ],
            identified => defined $item->{identified} ? ($item->{identified} ? 1 : 0) : 1,
            broken     => $item->{broken} ? 1 : 0,
        };
    }
    $state->{item_count} = scalar(@items);
    $state->{items} = \@items;
    return $state;
}

=head2 build_map_state

Extracts current map metadata and nearby actors.

Keys: name, base_name, width, height, is_city, instance_id,
monster_count, player_count, npc_count, portal_count, actors

=cut

sub build_map_state {
    my ($self) = @_;
    my $field = $::field;
    my $state = {
        name         => _scalarize($field ? $field->name() : ''),
        base_name    => _scalarize($field ? $field->baseName() : ''),
        width        => _scalarize($field ? $field->width() : 0),
        height       => _scalarize($field ? $field->height() : 0),
        is_city      => ($field && $field->isCity()) ? 1 : 0,
        instance_id  => _scalarize($field ? $field->instanceID() : undef),
        monster_count => scalar(keys %::monsters) + 0,
        player_count  => scalar(keys %::players) + 0,
        npc_count     => scalar(keys %::npcs) + 0,
        portal_count  => scalar(keys %::portals) + 0,
        actors        => [],
    };

    my $max = $self->{_max_actors} || 24;
    my @actors;

    # Add monsters
    for my $id (keys %::monsters) {
        last if scalar(@actors) >= $max;
        my $m = $::monsters{$id};
        next unless $m;
        push @actors, {
            actor_id   => _actor_id($m, $id),
            type       => 'monster',
            name       => _scalarize($m->{name}),
            x          => _pos_x($m),
            y          => _pos_y($m),
            hp         => _scalarize($m->{hp}),
            hp_max     => _scalarize($m->{hp_max}),
            level      => _scalarize($m->{level}),
            relation   => 'hostile',
            name_id    => _scalarize($m->{nameID}),
            distance   => _calc_distance($m, $::char),
        };
    }
    # Add players
    for my $id (keys %::players) {
        last if scalar(@actors) >= $max;
        my $p = $::players{$id};
        next unless $p;
        my $party_member = 0;
        if ($::char && $::char->{party} && ref($::char->{party}{users}) eq 'HASH') {
            for my $uid (keys %{$::char->{party}{users}}) {
                my $u = $::char->{party}{users}{$uid};
                if ($u && (_scalarize($u->{name}) eq _scalarize($p->{name}) || $uid eq $id)) {
                    $party_member = 1;
                    last;
                }
            }
        }
        push @actors, {
            actor_id   => _actor_id($p, $id),
            type       => 'player',
            name       => _scalarize($p->{name}),
            x          => _pos_x($p),
            y          => _pos_y($p),
            level      => _scalarize($p->{level}),
            relation   => $party_member ? 'party' : 'neutral',
            distance   => _calc_distance($p, $::char),
        };
    }
    # Add NPCs
    for my $id (keys %::npcs) {
        last if scalar(@actors) >= $max;
        my $n = $::npcs{$id};
        next unless $n;
        push @actors, {
            actor_id   => _actor_id($n, $id),
            type       => 'npc',
            name       => _scalarize($n->{name}),
            x          => _pos_x($n),
            y          => _pos_y($n),
            relation   => 'neutral',
            distance   => _calc_distance($n, $::char),
        };
    }

    $state->{actors} = \@actors;
    return $state;
}

=head2 build_party_state

Extracts party membership info.

Keys: in_party, party_name, member_count, members (array)

Each member: name, level, hp, hp_max, map, x, y

=cut

sub build_party_state {
    my ($self) = @_;
    my $char  = $::char;
    my $state = {
        in_party     => 0,
        party_name   => undef,
        member_count => 0,
        members      => [],
    };
    return $state unless $char && $char->{party};

    $state->{in_party} = 1;
    $state->{party_name} = _scalarize($char->{party}{name});

    my @members;
    if (ref($char->{party}{users}) eq 'HASH') {
        for my $uid (keys %{$char->{party}{users}}) {
            my $u = $char->{party}{users}{$uid};
            next unless $u;
            push @members, {
                name   => _scalarize($u->{name}),
                level  => _scalarize($u->{level}),
                hp     => _scalarize($u->{hp}),
                hp_max => _scalarize($u->{hp_max}),
                map    => _scalarize($u->{map}),
                x      => _scalarize($u->{x}),
                y      => _scalarize($u->{y}),
                online => defined $u->{hp} ? 1 : 0,
            };
        }
    }
    $state->{member_count} = scalar(@members);
    $state->{members} = \@members;
    return $state;
}

=head2 build_guild_state

Extracts guild information.

Keys: in_guild, guild_id, guild_name, master, member_count, position,
alliance_count, members (array)

=cut

sub build_guild_state {
    my ($self) = @_;
    my %guild = %::guild;
    my $state = {
        in_guild    => scalar(keys %guild) > 0 ? 1 : 0,
        guild_id    => _scalarize($guild{guild_id}),
        guild_name  => _scalarize($guild{guild_name} || $guild{name}),
        master      => _scalarize($guild{master} || $guild{guild_master}),
        member_count => 0,
        position    => _scalarize($guild{position}),
        members     => [],
    };

    my @members;
    if (ref($guild{members}) eq 'HASH') {
        for my $mid (keys %{$guild{members}}) {
            my $m = $guild{members}{$mid};
            next unless $m;
            push @members, {
                name   => _scalarize($m->{name}),
                level  => _scalarize($m->{level}),
                class  => _scalarize($m->{class}),
                online => $m->{online} ? 1 : 0,
                position => _scalarize($m->{position}),
            };
        }
    } elsif (ref($guild{members}) eq 'ARRAY') {
        for my $m (@{$guild{members}}) {
            next unless $m;
            push @members, {
                name   => _scalarize($m->{name}),
                level  => _scalarize($m->{level}),
                class  => _scalarize($m->{class}),
                online => $m->{online} ? 1 : 0,
                position => _scalarize($m->{position}),
            };
        }
    }
    $state->{member_count} = scalar(@members);
    $state->{members} = \@members;
    return $state;
}

=head2 build_buff_state

Extracts active status effects (buffs/debuffs) on the character.

Keys: buff_count, buffs (array)

Each buff: status_name, status_id, remaining_ms, active

=cut

sub build_buff_state {
    my ($self) = @_;
    my $char = $::char;
    my $state = {
        buff_count => 0,
        buffs      => [],
    };
    return $state unless $char;

    my @buffs;
    if ($char->{statuses} && ref($char->{statuses}) eq 'HASH') {
        for my $sid (keys %{$char->{statuses}}) {
            my $s = $char->{statuses}{$sid};
            next unless $s;
            my $remaining = 0;
            if ($s->{time}) {
                $remaining = int(($s->{time} - time) * 1000);
                $remaining = 0 if $remaining < 0;
            }
            push @buffs, {
                status_name  => _scalarize($s->{name} || $::statusName{$sid} || $sid),
                status_id    => _scalarize($sid),
                remaining_ms => $remaining,
                active       => $remaining > 0 ? 1 : 0,
            };
        }
    }

    # Also check from Actor method
    if ($char->can('statusActive')) {
        for my $sname (qw(ASPDPOTION0 ASPDPOTION1 ASPDPOTION2 ASPDPOTION3
                          INCREASEAGI DECREASEAGI BLESSING CURSE
                          POISON HALLUCINATIONWALK CLOAKING HIDING
                          SIGHT ENDURE CONCENTRATION ENCHANTBLADE
                          OVERTHRUST WEAPONPERFECTION REFLECTSHIELD
                          ASSUMPTIO KYRIE MAGNIFICAT ASPERSIO))
        {
            my $active = eval { $char->statusActive($sname) };
            next unless $active;
            push @buffs, {
                status_name => $sname,
                status_id   => undef,
                remaining_ms => undef,
                active      => 1,
            };
        }
    }

    $state->{buff_count} = scalar(@buffs);
    $state->{buffs} = \@buffs;
    return $state;
}

=head2 build_pet_state

Extracts pet information.

Keys: has_pet, pet_name, pet_id, level, hunger, intimacy,
accessory, rename_flag

=cut

sub build_pet_state {
    my ($self) = @_;
    my $char = $::char;
    my %pets = %::pets;
    my $state = {
        has_pet  => scalar(keys %pets) > 0 ? 1 : 0,
    };

    # Pet data from %pets hash
    my $pet_actor;
    for my $pid (keys %pets) {
        $pet_actor = $pets{$pid};
        last;
    }

    if ($pet_actor) {
        $state->{pet_name} = _scalarize($pet_actor->{name});
        $state->{level}    = _scalarize($pet_actor->{level});
        $state->{x}        = _pos_x($pet_actor);
        $state->{y}        = _pos_y($pet_actor);
        $state->{hp}       = _scalarize($pet_actor->{hp});
        $state->{hp_max}   = _scalarize($pet_actor->{hp_max});
    }

    # Also check $char->{pet} for equipped pet info
    if ($char && $char->{pet}) {
        $state->{has_pet} = 1;
        my $cp = $char->{pet};
        $state->{pet_name}    ||= _scalarize($cp->{name});
        $state->{pet_id}      ||= _scalarize($cp->{petID} || $cp->{id});
        $state->{level}       ||= _scalarize($cp->{level});
        $state->{hunger}      ||= _scalarize($cp->{hungry});
        $state->{intimacy}    ||= _scalarize($cp->{intimate});
        $state->{accessory}   ||= _scalarize($cp->{accessory});
        $state->{rename_flag} ||= _scalarize($cp->{rename_flag});
    }

    return $state;
}

=head2 build_homunculus_state

Extracts homunculus information.

Keys: has_homunculus, name, level, hp, hp_max, sp, sp_max, exp,
exp_max, hunger, intimacy, s_int, s_str, s_agi, s_dex, s_vit,
s_luk, skill_points

=cut

sub build_homunculus_state {
    my ($self) = @_;
    my %homunculus = %::homunculus;
    my $state = {
        has_homunculus => scalar(keys %homunculus) > 0 ? 1 : 0,
    };

    my $h;
    for my $hid (keys %homunculus) {
        $h = $homunculus{$hid};
        last;
    }

    return $state unless $h;

    $state->{name}      = _scalarize($h->{name});
    $state->{level}     = _scalarize($h->{level});
    $state->{hp}        = _scalarize($h->{hp});
    $state->{hp_max}    = _scalarize($h->{hp_max});
    $state->{sp}        = _scalarize($h->{sp});
    $state->{sp_max}    = _scalarize($h->{sp_max});
    $state->{exp}       = _scalarize($h->{exp});
    $state->{exp_max}   = _scalarize($h->{exp_max});
    $state->{hunger}    = _scalarize($h->{hungry});
    $state->{intimacy}  = _scalarize($h->{intimate});
    $state->{s_int}     = _scalarize($h->{s_int});
    $state->{s_str}     = _scalarize($h->{s_str});
    $state->{s_agi}     = _scalarize($h->{s_agi});
    $state->{s_dex}     = _scalarize($h->{s_dex});
    $state->{s_vit}     = _scalarize($h->{s_vit});
    $state->{s_luk}     = _scalarize($h->{s_luk});
    $state->{skill_points} = _scalarize($h->{skill_points} || $h->{points_skill});
    $state->{x}         = _pos_x($h);
    $state->{y}         = _pos_y($h);

    return $state;
}

=head2 build_mercenary_state

Extracts mercenary information.

Keys: has_mercenary, name, level, hp, hp_max, sp, sp_max, atk, matk,
hit, crit, def, mdef, flee, aspd

=cut

sub build_mercenary_state {
    my ($self) = @_;
    my $char = $::char;
    my $state = {
        has_mercenary => 0,
    };

    # Mercenary is usually stored differently - check char->{mercenary}
    # or look through Actor::Slave::Mercenary instances
    if ($char && $char->{mercenary}) {
        my $m = $char->{mercenary};
        $state->{has_mercenary} = 1;
        $state->{name}  = _scalarize($m->{name});
        $state->{level} = _scalarize($m->{level});
        $state->{hp}    = _scalarize($m->{hp});
        $state->{hp_max} = _scalarize($m->{hp_max});
        $state->{sp}    = _scalarize($m->{sp});
        $state->{sp_max} = _scalarize($m->{sp_max});
        $state->{atk}   = _scalarize($m->{attack});
        $state->{matk}  = _scalarize($m->{matk} || $m->{matk_min});
        $state->{hit}   = _scalarize($m->{hit});
        $state->{crit}  = _scalarize($m->{crit});
        $state->{def}   = _scalarize($m->{def});
        $state->{mdef}  = _scalarize($m->{mdef});
        $state->{flee}  = _scalarize($m->{flee});
        $state->{aspd}  = _scalarize($m->{aspd});
    }

    return $state;
}

=head2 build_mount_state

Extracts mount information (Peco Peco, Dragon, etc.).

Keys: is_mounted, mount_name, mount_type, mount_level

=cut

sub build_mount_state {
    my ($self) = @_;
    my $char = $::char;
    my $state = {
        is_mounted   => 0,
        mount_name   => undef,
        mount_type   => undef,
        mount_level  => undef,
    };

    return $state unless $char;

    # Check if mounted via status
    my $is_mounted = 0;
    if ($char->{statuses} && ref($char->{statuses}) eq 'HASH') {
        for my $sid (keys %{$char->{statuses}}) {
            my $sname = $::statusName{$sid} || '';
            if ($sname =~ /riding|mount|peco|dragon/i) {
                $is_mounted = 1;
                $state->{mount_name} = _scalarize($sname);
                last;
            }
        }
    }

    # Check for weapons/armor that indicate mount type
    if ($char->{weapon} && $char->{weapon} == 7) {  # Whip = mounted dancer
        $is_mounted = 1;
        $state->{mount_name} ||= 'Peco Peco';
    }

    # Some job IDs are mount-only (Rune Knight with dragon)
    if ($char->{jobID} && $char->{jobID} >= 4080 && $char->{jobID} <= 4095) {
        $is_mounted = 1;
        $state->{mount_type} = 'dragon';
        $state->{mount_name} ||= 'Dragon';
    }

    $state->{is_mounted} = $is_mounted ? 1 : 0;
    return $state;
}

=head2 build_equipment_state

Extracts detailed equipment information.

Keys: equip_count, equipment (hashref slot -> item)

Each slot: slot_name, name_id, name, refine, cards (array),
position, broken

=cut

sub build_equipment_state {
    my ($self) = @_;
    my $char = $::char;
    my $state = {
        equip_count => 0,
        equipment   => {},
    };
    return $state unless $char;

    my $equipment = {};
    if ($char->{equipment} && ref($char->{equipment}) eq 'HASH') {
        for my $slot (keys %{$char->{equipment}}) {
            my $item = $char->{equipment}{$slot};
            next unless $item;
            $equipment->{$slot} = {
                slot_name  => _scalarize($slot),
                name_id    => _scalarize($item->{nameID}),
                name       => _scalarize($item->{name}),
                refine     => _scalarize($item->{refine}),
                cards      => [
                    grep { defined && $_ > 0 } (
                        _scalarize($item->{card1}),
                        _scalarize($item->{card2}),
                        _scalarize($item->{card3}),
                        _scalarize($item->{card4})
                    )
                ],
                position   => _scalarize($item->{position} || $slot),
                broken     => $item->{broken} ? 1 : 0,
            };
        }
    }
    $state->{equip_count} = scalar(keys %{$equipment});
    $state->{equipment} = $equipment;
    return $state;
}

=head2 build_npc_dialogue_state

Extracts current NPC dialogue state (if talking to an NPC).

Keys: in_dialogue, npc_name, npc_id, npc_unique_id, talk_state,
responses (array), current_responses

=cut

sub build_npc_dialogue_state {
    my ($self) = @_;
    my %talk     = %::talk;
    my %responses = %::responses;
    my $char = $::char;
    my $state = {
        in_dialogue   => scalar(keys %talk) > 0 ? 1 : 0,
        npc_name      => undef,
        npc_id        => undef,
        npc_unique_id  => undef,
        talk_state    => undef,
        responses     => [],
    };

    return $state unless $state->{in_dialogue};

    $state->{talk_state} = _scalarize($talk{state} || $talk{'talk'} || 0);
    $state->{npc_name}   = _scalarize($talk{name} || $talk{npcName});
    $state->{npc_id}     = _scalarize($talk{ID} || $talk{npcID});
    $state->{npc_unique_id} = _scalarize($talk{uniqueID} || $talk{ownerID});

    # Collect available responses (from OpenKore's %responses hash)
    my @responses;
    if (scalar(keys %responses) > 0) {
        for my $rkey (sort { ($a =~ /(\d+)/)[0] || 0 <=> ($b =~ /(\d+)/)[0] || 0 } keys %responses) {
            push @responses, {
                key   => _scalarize($rkey),
                value => _scalarize($responses{$rkey}),
            };
        }
    }
    $state->{responses} = \@responses;

    return $state;
}

=head2 build_quest_state

Extracts quest log information.

Keys: quest_count, quests (array)

Each quest: quest_id, title, status, active, objectives (array)

=cut

sub build_quest_state {
    my ($self) = @_;
    my $state = {
        quest_count => 0,
        quests      => [],
    };

    my $quest_list = $::questList;
    if ($quest_list) {
        my @items;
        # Try getItems method (OpenKore's ActorList pattern)
        my $items = eval { $quest_list->getItems() } || [];
        if (ref($items) eq 'ARRAY') {
            for my $q (@{$items}) {
                next unless $q;
                push @items, {
                    quest_id => _scalarize($q->{questID} || $q->{id}),
                    title    => _scalarize($q->{title} || $q->{name}),
                    status   => _scalarize($q->{status}),
                    active   => $q->{active} ? 1 : 0,
                    objectives => _extract_array($q->{objectives}),
                };
            }
        }
        # Try hash access
        if (!@items && ref($quest_list) eq 'HASH') {
            for my $qid (keys %{$quest_list}) {
                my $q = $quest_list->{$qid};
                next unless $q && ref($q) eq 'HASH';
                push @items, {
                    quest_id => _scalarize($qid),
                    title    => _scalarize($q->{title} || $q->{name}),
                    status   => _scalarize($q->{status}),
                    active   => $q->{active} ? 1 : 0,
                };
            }
        }
        $state->{quest_count} = scalar(@items);
        $state->{quests} = \@items;
    }

    return $state;
}

=head2 build_market_state

Extracts market/buying store/vending information.

Keys: is_selling, is_buying, shop_title, shop_items (array), buyer_shop

Each shop_item: name_id, name, amount, price, slot

=cut

sub build_market_state {
    my ($self) = @_;
    my %shop      = %::shop;
    my %buyer_shop = %::buyer_shop;
    my $char = $::char;
    my $state = {
        is_selling  => scalar(keys %shop) > 0 ? 1 : 0,
        is_buying   => scalar(keys %buyer_shop) > 0 ? 1 : 0,
        shop_title  => _scalarize($shop{title} || $shop{shop_title}),
        shop_items  => [],
        buyer_items => [],
    };

    # Shop items (selling)
    my @shop_items;
    if (ref($shop{items}) eq 'ARRAY') {
        for my $item (@{$shop{items}}) {
            next unless $item;
            push @shop_items, {
                name_id => _scalarize($item->{nameID}),
                name    => _scalarize($item->{name}),
                amount  => _scalarize($item->{amount}),
                price   => _scalarize($item->{price}),
            };
        }
    }
    $state->{shop_items} = \@shop_items;

    # Buyer shop items (buying)
    my @buyer_items;
    if (ref($buyer_shop{items}) eq 'ARRAY') {
        for my $item (@{$buyer_shop{items}}) {
            next unless $item;
            push @buyer_items, {
                name_id => _scalarize($item->{nameID}),
                name    => _scalarize($item->{name}),
                amount  => _scalarize($item->{amount}),
                price   => _scalarize($item->{price}),
            };
        }
    }
    $state->{buyer_items} = \@buyer_items;

    return $state;
}

=head2 build_environment_state

Extracts environment state: time, weather, surroundings.

Keys: time_s, daylight, monsters_nearby, players_nearby, npcs_nearby,
in_town, town_name, weight_ratio, ai_sequence, network_state,
is_in_game, is_dead

=cut

sub build_environment_state {
    my ($self) = @_;
    my $char  = $::char;
    my $field = $::field;
    my $net   = $::net;

    my $map_name = $field ? lc($field->baseName() || $field->name() || '') : '';
    my @town_maps = qw(prontera morocc geffen payon aldebaran alberta izlude comodo);
    my $in_town = 0;
    for my $t (@town_maps) {
        if ($map_name =~ /^$t/i) {
            $in_town = 1;
            last;
        }
    }

    my $ai_top = @::ai_seq ? $::ai_seq[0] : '';

    return {
        time_s           => time,
        daylight         => _is_daytime() ? 1 : 0,
        monsters_nearby  => scalar(keys %::monsters) + 0,
        players_nearby   => scalar(keys %::players) + 0,
        npcs_nearby      => scalar(keys %::npcs) + 0,
        in_town          => $in_town,
        town_name        => $in_town ? $map_name : undef,
        weight_ratio     => ($char && $char->{weight_max} && $char->{weight_max} > 0)
            ? sprintf('%.4f', ($char->{weight} || 0) / $char->{weight_max}) : 0,
        ai_sequence      => _scalarize($ai_top),
        network_state    => $net ? _scalarize($net->getState()) : -1,
        is_in_game       => ($net && $net->getState() == 5) ? 1 : 0,  # 5 = IN_GAME
        is_dead          => ($char && ($char->{hp} || 0) <= 0) ? 1 : 0,
    };
}

=head2 build_ground_items_state

Extracts items on the ground near the character.

Keys: item_count, items (array)

Each item: name_id, name, amount, x, y, distance, identified,
dropper (name of who dropped it)

=cut

sub build_ground_items_state {
    my ($self) = @_;
    my $char = $::char;
    my $state = {
        item_count => 0,
        items      => [],
    };

    # Ground items in OpenKore are tracked via $itemsList or %items hash
    my $items_list = $::itemsList;
    my %items_hash = %::items;
    my $max = $self->{_max_items} || 200;
    my @items;

    # Try $itemsList first (ActorList)
    if ($items_list) {
        my $list_items = eval { $items_list->getItems() } || [];
        for my $item (@{$list_items}) {
            next unless $item;
            last if scalar(@items) >= $max;
            push @items, {
                name_id    => _scalarize($item->{nameID}),
                name       => _scalarize($item->{name}),
                amount     => _scalarize($item->{amount}),
                x          => _pos_x($item),
                y          => _pos_y($item),
                distance   => _calc_distance($item, $char),
                identified => defined $item->{identified} ? ($item->{identified} ? 1 : 0) : 1,
            };
        }
    }

    # Fall back to %items hash
    if (!@items && scalar(keys %items_hash) > 0) {
        for my $iid (keys %items_hash) {
            my $item = $items_hash{$iid};
            next unless $item;
            last if scalar(@items) >= $max;
            push @items, {
                name_id    => _scalarize($item->{nameID}),
                name       => _scalarize($item->{name}),
                amount     => _scalarize($item->{amount}),
                x          => _pos_x($item),
                y          => _pos_y($item),
                distance   => _calc_distance($item, $char),
                identified => 1,
            };
        }
    }

    $state->{item_count} = scalar(@items);
    $state->{items} = \@items;
    return $state;
}

=head2 build_instance_state

Extracts instance/dungeon information.

Keys: in_instance, instance_id, instance_name, instance_level,
instance_time_left_s

=cut

sub build_instance_state {
    my ($self) = @_;
    my $field = $::field;
    my $state = {
        in_instance        => 0,
        instance_id        => undef,
        instance_name      => undef,
        instance_level     => undef,
        instance_time_left_s => undef,
    };

    if ($field) {
        my $inst_id = eval { $field->instanceID() };
        if (defined $inst_id && $inst_id ne '' && $inst_id != 0) {
            $state->{in_instance} = 1;
            $state->{instance_id} = _scalarize($inst_id);
            my $base = $field->baseName() || $field->name() || '';
            $state->{instance_name} = _scalarize($base);
        }
    }

    # Check char for instance data
    my $char = $::char;
    if ($char && $char->{instance}) {
        $state->{in_instance} = 1;
        $state->{instance_id}   ||= _scalarize($char->{instance}{id});
        $state->{instance_name} ||= _scalarize($char->{instance}{name});
        $state->{instance_level} = _scalarize($char->{instance}{level});
        if ($char->{instance}{time}) {
            my $remaining = $char->{instance}{time} - time;
            $state->{instance_time_left_s} = $remaining > 0 ? $remaining : 0;
        }
    }

    return $state;
}

# ── Internal helpers ──

sub _scalarize {
    my ($val) = @_;
    return undef unless defined $val;
    return $val if !ref($val);
    return "$val";
}

sub _pos_x {
    my ($actor) = @_;
    return undef unless $actor;
    if (ref($actor->{pos_to}) eq 'HASH') {
        return _scalarize($actor->{pos_to}{x});
    }
    if (ref($actor->{pos}) eq 'HASH') {
        return _scalarize($actor->{pos}{x});
    }
    return _scalarize($actor->{x});
}

sub _pos_y {
    my ($actor) = @_;
    return undef unless $actor;
    if (ref($actor->{pos_to}) eq 'HASH') {
        return _scalarize($actor->{pos_to}{y});
    }
    if (ref($actor->{pos}) eq 'HASH') {
        return _scalarize($actor->{pos}{y});
    }
    return _scalarize($actor->{y});
}

sub _actor_id {
    my ($actor, $fallback) = @_;
    my $id = _scalarize($actor->{ID});
    return $id if defined $id && $id ne '';
    my $bin = _scalarize($actor->{binID});
    return $bin if defined $bin && $bin ne '';
    return _scalarize($fallback) if defined $fallback;
    return '';
}

sub _calc_distance {
    my ($a, $b) = @_;
    return undef unless $a && $b;
    my $ax = _pos_x($a);
    my $ay = _pos_y($a);
    my $bx = _pos_x($b);
    my $by = _pos_y($b);
    return undef if !defined $ax || !defined $ay || !defined $bx || !defined $by;
    return int(sqrt(($ax - $bx)**2 + ($ay - $by)**2));
}

sub _item_type_name {
    my ($type) = @_;
    return undef unless defined $type;
    my %types = (
        0  => 'Harming',
        1  => 'Healing',
        2  => 'Usable',
        3  => 'Etc',
        4  => 'Weapon',
        5  => 'Armor',
        6  => 'Card',
        7  => 'Pet',
        8  => 'Ammo',
        10 => 'Cash',
        11 => 'Costume',
        12 => 'Enchant',
    );
    return $types{$type} || "Type$type";
}

sub _extract_array {
    my ($val) = @_;
    return [] unless defined $val;
    return $val if ref($val) eq 'ARRAY';
    return [$val] if !ref($val);
    return [];
}

sub _is_daytime {
    # Simple heuristic: 6am-6pm is daytime (RO server time ~= local time)
    my $hour = (localtime(time))[2];
    return $hour >= 6 && $hour < 18;
}

1;

__END__

=head1 COPYRIGHT

Copyright (C) 2026 by the OpenKore AI project.
