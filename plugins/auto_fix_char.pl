package auto_fix_char;

use strict;
use Plugins;
use Globals;
use Network;
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
    
    my @broken;
    for (my $i = 0; $i < @chars; $i++) {
        next unless $chars[$i];
        my $name = $chars[$i]->{name} || '';
        if ($name eq '') {
            push @broken, $i;
            message "[auto_fix] Empty character in slot $i\n", 'warning';
        }
    }
    
    if (@broken) {
        # Set char to empty so OpenKore doesn't auto-select before we fix
        configModify("char","");
        message "[auto_fix] Char auto-select disabled for fixing\n", 'system';
        
        # Send delete2 packet for each broken char (no email needed for newer servers)
        foreach my $slot (@broken) {
            my $charID = $chars[$slot]->{charID};
            message "[auto_fix] Deleting broken char in slot $slot (ID: " . unpack("H*", $charID) . ")\n", 'system';
            $messageSender->sendCharDelete2($charID);
        }
        $ran = 1;
        Plugins::addTimer(\&create_char, 5, 0, 1);
    } elsif (scalar(grep { $_ && $_->{name} } @chars) == 0) {
        # No valid chars - create one
        configModify("char","");
        message "[auto_fix] No characters exist, creating...\n", 'system';
        $ran = 1;
        Plugins::addTimer(\&create_char, 1, 0, 1);
    } else {
        message "[auto_fix] Account has valid characters, no fix needed\n", 'system';
        $ran = 1;
        Plugins::delHook('charSelectScreen', $hook) if $hook;
    }
}

sub create_char {
    my $timer = shift;
    return unless $net && $net->getState() == Network::CONNECTED_TO_LOGIN_SERVER;
    
    my $username = $config{username} || 'bot';
    message "[auto_fix] Creating character '$username'\n", 'system';
    
    $messageSender->sendCharCreate(0, $username, 1, 9, 1, 1, 9, 1, 0, 0);
    $ran = 1;
    
    Plugins::addTimer(sub {
        message "[auto_fix] Character created! Reconnecting...\n", 'system';
        configModify("char","0");
        Plugins::delHook('charSelectScreen', $hook) if $hook;
        relog(1);
    }, 3, 0, 1);
}

sub unload {
    Plugins::delHook('charSelectScreen', $hook) if $hook;
}

1;
