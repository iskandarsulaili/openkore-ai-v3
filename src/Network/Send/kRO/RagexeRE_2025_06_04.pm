#########################################################################
#  OpenKore - Packet sending
#  This module contains functions for sending packets to the server.
#
#  This software is open source, licensed under the GNU General Public
#  License, version 2.
#  Basically, this means that you're allowed to modify and distribute
#  this software. However, if you distribute modified versions, you MUST
#  also distribute the source code.
#  See http://www.gnu.org/licenses/gpl.html for the full license.
########################################################################
#  kRO RagexeRE 2025-06-04 (Speedrun client) — specialized for rathena-ai-world
#
#  rAthena PACKETVER 20250604 uses CH_MAKE_CHAR (0x0a39):
#    int16 packetType; char name[24]; uint8 slot; uint16 hair_color;
#    uint16 hair_style; uint32 job; uint8 sex   (36 bytes with header)
#
#  The vanilla kRO::RagexeRE_2021_11_03 chain inherits the OLD 0x0a39
#  layout ('a24 C v4 C' with an undef 'unknown' field) which produces
#  empty character names on this server. This module overrides 0x0a39
#  with the correct uint32-job layout.
package Network::Send::kRO::RagexeRE_2025_06_04;

use strict;
use base qw(Network::Send::kRO::RagexeRE_2021_11_03);
use Log qw(debug);
use Utils qw(getTickCount);

sub new {
	my ($class) = @_;
	my $self = $class->SUPER::new(@_);

	my %packets = (
		# CH_MAKE_CHAR — PACKETVER >= 20151001 layout: name[24], slot,
		# hair_color u16, hair_style u16, job u32, sex u8 (36B w/ header)
		'0A39' => ['char_create', 'a24 C v2 V C', [qw(name slot hair_color hair_style job_id sex)]],
		# CH_SELECT_ACCESSIBLE_MAPNAME (0x0841) — replies to the char-server's
		# HC_NOTIFY_ACCESSIBLE_MAPNAME (0x0840) "wait for map-server" packet.
		# rathena-ai-world 20250604 char-server uses this Renewal handshake to
		# finish char-select once a map-server is available. Without it the bot
		# timed out on char select (the accessible-map reply never went out).
		'0841' => ['select_accessible_mapname', 'C C', [qw(char_slot map_slot)]],
	);

	$self->{packet_list}{$_} = $packets{$_} for keys %packets;
	$self->{char_create_version} = 0x0A39;

	return $self;
}

# CZ_ENTER2 / 0x0436 — PACKETVER >= 20220330 layout (23 bytes):
#   account_id.L char_id.L auth_code.L client_tick.L x4 sex.B
# The inherited RagexeRE_2021_11_03 map_login struct ('a4 a4 a4 V2 C') is
# 21 bytes with sex at offset 20 — the rAthena 20250604 server reads sex at
# byte 22 (pos 2,6,10,14,22) and rejects the short packet ("wrong length"),
# desyncing the whole stream and disconnecting every bot on map-enter. We
# override sendMapLogin to pack the exact 23-byte form.
sub sendMapLogin {
	my ($self, $accountID, $charID, $sessionID, $sex) = @_;
	$sex = 0 if ($sex > 1 || $sex < 0); # Sex can only be 0 (female) or 1 (male)

	my $msg = pack(
		'v a4 a4 a4 V x4 C',
		0x0436, $accountID, $charID, $sessionID, getTickCount(), $sex
	);

	$self->sendToServer($msg);
	debug "Sent sendMapLogin (0x0436, 23 bytes)\n", "sendPacket", 2;
}

1;