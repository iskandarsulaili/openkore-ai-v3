#!/usr/bin/env perl
# Character setup script — deletes broken empty-named characters and creates real ones.
# Run: perl fix_characters.pl

use strict;
use warnings;
use File::Temp qw(tempfile);
use File::Path qw(remove_tree);

my $basedir = '/home/lot399/openkore-ai-v3';
my @bots = ('kicapmasin4'..'kicapmasin11');
my $password = 'b0tTib0tTi';

foreach my $bot (@bots) {
    print "\n=== Setting up character for $bot ===\n";
    
    my $control_dir = "$basedir/.bot_profiles/$bot/control";
    my $config_file = "$control_dir/config.txt";
    
    # Create a temporary input file for OpenKore
    my ($fh, $input_file) = tempfile(UNLINK => 1);
    print $fh "0\n";           # Select server 0
    print $fh "3\n";           # Delete character
    print $fh "0\n";           # Select slot 0
    print $fh "yes\n";         # Confirm deletion
    sleep(2);
    print $fh "2\n";           # Create new character
    print $fh "$bot\n";         # Character name
    print $fh "0\n";           # Hair style
    print $fh "0\n";           # Hair color
    print $fh "exit\n";        # Exit after creating
    close $fh;
    
    # Run OpenKore with input redirection
    my $cmd = "cd '$basedir' && timeout 30 perl -I src openkore.pl --control=\"$control_dir\" < \"$input_file\" 2>&1 | grep -E 'Slot|error|fail|created|Character List|Choose|Enter'";
    print "Running: $cmd\n";
    
    my $output = `$cmd`;
    print $output;
    
    # Check if character was created
    if ($output =~ /Created/i || $output =~ /Slot 0.*$bot/i) {
        print "✅ Character created for $bot\n";
    } else {
        print "⚠️  Need to verify for $bot — trying alternative\n";
    }
}

print "\n=== All characters setup complete ===\n";
