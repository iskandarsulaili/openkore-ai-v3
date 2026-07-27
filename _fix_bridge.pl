#!/usr/bin/env perl
# Fix aiSidecarBridge.pl: Phase 3 (config override fix) and Phase 6 (anti-detection delays)

use strict;
use warnings;

my $file = 'plugins/aiSidecarBridge/aiSidecarBridge.pl';
open(my $fh, '<', $file) or die "Cannot open $file: $!";
my @lines = <$fh>;
close($fh);

# Phase 6: Fix anti-detection delays (lines 22-23)
for my $i (0..$#lines) {
    if ($lines[$i] =~ /my \$ANTI_DETECTION_MIN_DELAY_MS = 10;/) {
        $lines[$i] = "my \$ANTI_DETECTION_MIN_DELAY_MS = 200;\n";
        print "Fixed ANTI_DETECTION_MIN_DELAY_MS at line " . ($i+1) . "\n";
    }
    if ($lines[$i] =~ /my \$ANTI_DETECTION_MAX_DELAY_MS = 50;/) {
        $lines[$i] = "my \$ANTI_DETECTION_MAX_DELAY_MS = 600;\n";
        print "Fixed ANTI_DETECTION_MAX_DELAY_MS at line " . ($i+1) . "\n";
    }
}

# Phase 3: Add _sidecar_set tracking in the "set" command handler (around line 2847-2856)
for my $i (0..$#lines) {
    if ($lines[$i] =~ /^# Handle set commands: "set <config_key> <value>"/) {
        # Find the line that sets $::config{$orig_key}
        for my $j ($i..$i+15) {
            if ($j <= $#lines && $lines[$j] =~ /\$::config\{\$orig_key\} = \$set_val;/) {
                # Add tracking after this line
                my $indent = $lines[$j] =~ /^(\s+)/ ? $1 : '';
                splice(@lines, $j+1, 0, 
                    $indent . "\t# Track that sidecar set this value so _apply_bot_config doesn't override\n",
                    $indent . "\t\$bridge_cfg{\"_sidecar_set_\$orig_key\"} = 1;\n"
                );
                print "Added _sidecar_set tracking at line " . ($j+2) . "\n";
                last;
            }
        }
        last;
    }
}

# Phase 3: Add "unless $bridge_cfg{...}" guards to _apply_bot_config
for my $i (0..$#lines) {
    if ($lines[$i] =~ /^\tsub _apply_bot_config/) {
        print "Found _apply_bot_config at line " . ($i+1) . "\n";
        # Find the config lines and add unless guards
        for my $j ($i..$i+30) {
            if ($j <= $#lines) {
                # attackAuto
                if ($lines[$j] =~ /^\s*\$::config\{'attackAuto'\} = _cfg\('aiSidecar_attackAuto', '2'\);/) {
                    $lines[$j] =~ s/;$/ unless \$bridge_cfg{'_sidecar_set_attackAuto'};/;
                    print "  Guarded attackAuto at line " . ($j+1) . "\n";
                }
                # attackAuto_inLockOnly
                elsif ($lines[$j] =~ /^\s*\$::config\{'attackAuto_inLockOnly'\} = _cfg\('aiSidecar_attackAutoInLockOnly', '1'\);/) {
                    $lines[$j] =~ s/;$/ unless \$bridge_cfg{'_sidecar_set_attackAuto_inLockOnly'};/;
                    print "  Guarded attackAuto_inLockOnly at line " . ($j+1) . "\n";
                }
                # attackAuto_followTarget
                elsif ($lines[$j] =~ /^\s*\$::config\{'attackAuto_followTarget'\} = _cfg\('aiSidecar_attackAutoFollowTarget', '0'\);/) {
                    $lines[$j] =~ s/;$/ unless \$bridge_cfg{'_sidecar_set_attackAuto_followTarget'};/;
                    print "  Guarded attackAuto_followTarget at line " . ($j+1) . "\n";
                }
                # attackAuto_onlyWhenSafe
                elsif ($lines[$j] =~ /^\s*\$::config\{'attackAuto_onlyWhenSafe'\} = _cfg\('aiSidecar_attackAutoOnlyWhenSafe', '0'\);/) {
                    $lines[$j] =~ s/;$/ unless \$bridge_cfg{'_sidecar_set_attackAuto_onlyWhenSafe'};/;
                    print "  Guarded attackAuto_onlyWhenSafe at line " . ($j+1) . "\n";
                }
                # attackAuto_noMove
                elsif ($lines[$j] =~ /^\s*\$::config\{'attackAuto_noMove'\} = _cfg\('aiSidecar_attackAutoNoMove', '0'\);/) {
                    $lines[$j] =~ s/;$/ unless \$bridge_cfg{'_sidecar_set_attackAuto_noMove'};/;
                    print "  Guarded attackAuto_noMove at line " . ($j+1) . "\n";
                }
                # sitAuto_hp_lower
                elsif ($lines[$j] =~ /^\s*\$::config\{'sitAuto_hp_lower'\} = _cfg\('aiSidecar_sitAutoHpLower', '0'\);/) {
                    $lines[$j] =~ s/;$/ unless \$bridge_cfg{'_sidecar_set_sitAuto_hp_lower'};/;
                    print "  Guarded sitAuto_hp_lower at line " . ($j+1) . "\n";
                }
                # sitAuto_hp_upper
                elsif ($lines[$j] =~ /^\s*\$::config\{'sitAuto_hp_upper'\} = _cfg\('aiSidecar_sitAutoHpUpper', '0'\);/) {
                    $lines[$j] =~ s/;$/ unless \$bridge_cfg{'_sidecar_set_sitAuto_hp_upper'};/;
                    print "  Guarded sitAuto_hp_upper at line " . ($j+1) . "\n";
                }
                # sitAuto_maxDmg
                elsif ($lines[$j] =~ /^\s*\$::config\{'sitAuto_maxDmg'\} = _cfg\('aiSidecar_sitAutoMaxDmg', '99999'\);/) {
                    $lines[$j] =~ s/;$/ unless \$bridge_cfg{'_sidecar_set_sitAuto_maxDmg'};/;
                    print "  Guarded sitAuto_maxDmg at line " . ($j+1) . "\n";
                }
                # itemsTakeAuto
                elsif ($lines[$j] =~ /^\s*\$::config\{'itemsTakeAuto'\} = '2';/) {
                    $lines[$j] =~ s/;$/ unless \$bridge_cfg{'_sidecar_set_itemsTakeAuto'};/;
                    print "  Guarded itemsTakeAuto at line " . ($j+1) . "\n";
                }
                # itemsTakeAuto_party
                elsif ($lines[$j] =~ /^\s*\$::config\{'itemsTakeAuto_party'\} = '1';/) {
                    $lines[$j] =~ s/;$/ unless \$bridge_cfg{'_sidecar_set_itemsTakeAuto_party'};/;
                    print "  Guarded itemsTakeAuto_party at line " . ($j+1) . "\n";
                }
                # itemsGatherAuto
                elsif ($lines[$j] =~ /^\s*\$::config\{'itemsGatherAuto'\} = '2';/) {
                    $lines[$j] =~ s/;$/ unless \$bridge_cfg{'_sidecar_set_itemsGatherAuto'};/;
                    print "  Guarded itemsGatherAuto at line " . ($j+1) . "\n";
                }
                # sellAuto
                elsif ($lines[$j] =~ /^\s*\$::config\{'sellAuto'\} = _cfg\('aiSidecar_sellAuto', '0'\);/) {
                    $lines[$j] =~ s/;$/ unless \$bridge_cfg{'_sidecar_set_sellAuto'};/;
                    print "  Guarded sellAuto at line " . ($j+1) . "\n";
                }
                # sellAuto_distance
                elsif ($lines[$j] =~ /^\s*\$::config\{'sellAuto_distance'\} = '25';/) {
                    $lines[$j] =~ s/;$/ unless \$bridge_cfg{'_sidecar_set_sellAuto_distance'};/;
                    print "  Guarded sellAuto_distance at line " . ($j+1) . "\n";
                }
                # storageAuto
                elsif ($lines[$j] =~ /^\s*\$::config\{'storageAuto'\} = _cfg\('aiSidecar_storageAuto', '0'\);/) {
                    $lines[$j] =~ s/;$/ unless \$bridge_cfg{'_sidecar_set_storageAuto'};/;
                    print "  Guarded storageAuto at line " . ($j+1) . "\n";
                }
            }
        }
        last;
    }
}

open(my $out, '>', $file) or die "Cannot write $file: $!";
print $out @lines;
close($out);

print "\nDone writing $file\n";
