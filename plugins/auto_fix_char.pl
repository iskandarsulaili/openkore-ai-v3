package auto_fix_char;

use strict;
use Plugins;
use Globals;
use Network;
use Misc qw(configModify relog);
use Log qw(message warning error debug);
use Translation qw(T TF);

Plugins::register('auto_fix_char', 'Auto-fix empty-named characters', \&unload);

my $hook = Plugins::addHook('charSelectScreen', \&on_char_screen);
my $ran = 0;

sub on_char_screen {
    my (undef, $args) = @_;
    
    my $username = $config{username} || '';
    message "[auto_fix] Checking characters for $username...\n", 'system';
    
    # Find which slots have valid characters and which are empty/broken
    my @valid_slots;
    my @empty_slots;
    for (my $i = 0; $i < @chars; $i++) {
        next unless $chars[$i];
        my $name = $chars[$i]->{name} || '';
        if ($name ne '') {
            push @valid_slots, $i;
            message "[auto_fix] Slot $i: '$name'\n", 'system';
        } else {
            push @empty_slots, $i;
            message "[auto_fix] Slot $i: EMPTY (broken)\n", 'warning';
        }
    }
    
    if (@valid_slots) {
        # Respect the configured slot if it holds a valid character;
        # otherwise fall back to the first valid slot.
        my $slot;
        if (defined $config{char} && $config{char} ne '' && $config{char} =~ /^\d+$/
            && $chars[$config{char}] && $chars[$config{char}]->{name} ne '') {
            $slot = $config{char};
            message "[auto_fix] Config slot $slot is valid, keeping it\n", 'system';
        } else {
            $slot = $valid_slots[0];
            message "[auto_fix] Using slot $slot (already has character)\n", 'system';
        }
        configModify("char", $slot);
        
        # CRITICAL: send char_login explicitly. Misc.pm's charSelectScreen
        # only auto-logs-in when called with autoLogin=1 (::charSelectScreen(1)).
        # The char list packet handler chain (Network/Receive.pm) calls it with
        # NO args, so autoLogin is undef and char_login is NEVER sent — the bot
        # loops re-receiving the char list forever and never enters the game.
        # Sending it here guarantees login every time a valid character exists.
        message "[auto_fix] Sending char_login for slot $slot...\n", 'system';
        eval { $messageSender->sendCharLogin($slot); 1; } or do {
            warning "[auto_fix] sendCharLogin failed: $@\n", 'system';
        };
        $timeout{charlogin}{time} = time;
        
        # Do NOT hijack $args->{return} — Misc.pm's charSelectScreen continues
        # afterward; forcing a return value prevented the game from loading.
        # We've already sent char_login above, so the flow proceeds to the map.
        $ran = 1;
        Plugins::delHook('charSelectScreen', $hook) if $hook;
        return;
    }
    
    # No valid characters - create one using slot progression
    # Store attempt counter in custom config key
    my $attempt = $config{auto_fix_slot_attempt} || 0;
    $attempt++;
    # Slot progression: 0, 1, 2, 3, 4, 5, 6, 7, 8 (0-indexed for server)
    my $target_slot = ($attempt - 1) % 9;
    configModify("auto_fix_slot_attempt", $attempt);
    
    message "[auto_fix] Creating character '$username' in slot $target_slot (attempt #$attempt)...\n", 'system';
    # Create Novice (job_id=0), Male (sex=1) -- match account sex (Boy)
    $messageSender->sendCharCreate($target_slot, $username, 1, 9, 0, 1);
    $ran = 1;
    
    # Set char to target slot and schedule relog
    configModify("char", $target_slot);
    
    # Use mainLoop hook to relog after character creation
    my $count = 0;
    my $relog_hook;
    $relog_hook = sub {
        $count++;
        if ($count > 30) {
            message "[auto_fix] Relogging to pick up new character...\n", 'system';
            Plugins::delHook('mainLoop', $relog_hook) if $relog_hook;
            Plugins::delHook('charSelectScreen', $hook) if $hook;
            relog(1);
        }
    };
    Plugins::addHook('mainLoop', $relog_hook);
    
    # Tell charSelectScreen to return and skip the menu
    $args->{return} = \"";
}

sub unload {
    Plugins::delHook('charSelectScreen', $hook) if $hook;
}

1;