package AiSidecarBridgeTest;

use strict;
use warnings;

use Test::More;
use FindBin qw($RealBin);
use File::Spec;

use Plugins;

sub start {
	subtest 'aiSidecarBridge rewrites legacy savepoint move command to respawn' => sub {
		my $plugin_file = File::Spec->catfile($RealBin, '..', '..', 'plugins', 'aiSidecarBridge', 'aiSidecarBridge.pl');

		eval { Plugins::load($plugin_file) };
		ok(!$@, 'loads aiSidecarBridge plugin') or do {
			diag $@ if $@;
			return;
		};

		ok(Plugins::registered('aiSidecarBridge'), 'registers aiSidecarBridge plugin');

		my ($rewritten, $kind) = aiSidecarBridge::_rewrite_runtime_command('move savepoint', {});
		is($rewritten, 'respawn', 'legacy move savepoint command is rewritten to respawn');
		is($kind, 'move_savepoint_rewritten', 'rewrite kind marks savepoint compatibility rewrite');

		my ($passthrough, $passthrough_kind) = aiSidecarBridge::_rewrite_runtime_command('respawn', {});
		is($passthrough, 'respawn', 'respawn command passes through without rewrite');
		is($passthrough_kind, '', 'respawn passthrough does not set rewrite kind');

		ok(Plugins::unload('aiSidecarBridge'), 'unloads aiSidecarBridge plugin');
		done_testing();
	};
}

1;
