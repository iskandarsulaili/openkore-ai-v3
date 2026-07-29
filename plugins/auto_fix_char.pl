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
    return if $ran;
    
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
        # Use the first valid character
        my $slot = $valid_slots[0];
        message "[auto_fix] Using slot $slot (already has character)\n", 'system';
        configModify("char", $slot);
        $ran = 1;
        Plugins::delHook('charSelectScreen', $hook) if $hook;
        $args->{return} = \"";
        return;
    }
    
    # No valid characters - create one in the first empty slot
    my $target_slot = -1;
    if (@empty_slots) {
        # Check if empty slots have real charID (broken ones don't free the slot)
        # Just use slot 1 which is always free on a new account
        $target_slot = 1;
    } else {
        $target_slot = 1;
    }
    
    message "[auto_fix] Creating character '$username' in slot $target_slot...\n", 'system';
    $messageSender->sendCharCreate($target_slot, $username, 1, 9, 1, 1, 9, 1, 0, 0);
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
