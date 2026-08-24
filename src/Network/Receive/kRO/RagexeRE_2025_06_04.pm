#########################################################################
#  OpenKore - Packet Receiveing
#  This module contains functions for Receiveing packets to the server.
#
#  This software is open source, licensed under the GNU General Public
#  License, version 2.
#  Basically, this means that you're allowed to modify and distribute
#  this software. However, if you distribute modified versions, you MUST
#  also distribute the source code.
#  See http://www.gnu.org/licenses/gpl.html for the full license.
########################################################################
# kRO RagexeRE 2025-06-04 (Speedrun client) — specialized for rathena-ai-world
#
# rAthena PACKETVER 20250604 login server responds with the secure login
# key packet 0x01DC that vanilla 2021_11_03 tables don't register. This
# module registers it so character-server selection completes.
package Network::Receive::kRO::RagexeRE_2025_06_04;
use strict;
use base qw(Network::Receive::kRO::RagexeRE_2021_11_03);

sub new {
	my ($class) = @_;
	my $self = $class->SUPER::new(@_);

	# Secure login key (PACKET_AC_ACCEPT_LOGIN / 0x01DC) — required by
	# rathena-ai-world 20250604 login handshake
	$self->{packet_list}{'01DC'} = ['secure_login_key', 'x2 a*', [qw(secure_key)]];

	# HC_NOTIFY_ACCESSIBLE_MAPNAME (0x0840) — sent by rathena-ai-world
	# 20250604 char-server when char-select is waiting on an available
	# map-server (accessible-maps handshake). Without this handler the bot
	# logged "Unknown switch: 0840" and timed out on char select instead of
	# replying 0x0841 once a map-server becomes ready.
	$self->{packet_list}{'0840'} = ['notify_accessible_mapname', 'v a*', [qw(len mapList)]];

	return $self;
}

1;
