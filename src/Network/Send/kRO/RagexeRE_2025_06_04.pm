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
use Globals qw(%config);
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

# CZ_ENTER2 / 0x0436 — map-login packet length is SERVER-ADAPTED, never hardcoded.
#   The RAW server's map-server binary changed mid-deploy (multi-login era):
#   SOURCE tree = 19 bytes, the RUNNING binary (built 10:24) = 26 bytes.
#   RULE.md: adapt to the LIVE server — the sidecar probes the live server +
#   persists the accepted length in ServerSolutionsStore, then writes
#   `mapLoginLength <N>` into the bot config (the bridge's server-adaptation
#   flow). Supported: 19 (id + accountID + charID + sessionID + tick + sex),
#   23 (id + 4 longs + sex + 4-byte unknown — the LIVE server's actual
#   expectation, PROBED: 23-byte 0x0436 -> 23B reply, the real client sends
#   this), 26 (id + 5 longs + sex + 3 pad). Default 23 (the server's form).
sub sendMapLogin {
	my ($self, $accountID, $charID, $sessionID, $sex) = @_;
	my $msg;
	$sex = 0 if ($sex > 1 || $sex < 0); # Sex can only be 0 (female) or 1 (male)

	# Server-adapted length (the sidecar sets it from a live probe). Default 23.
	my $mlen = $config{mapLoginLength} || 23;
	# BQ (2026-08-31): the capture consumer learns the REAL field offsets and
	# writes them as mapLoginLayout (JSON). If present, emit the packet with the
	# LEARNED offsets — the hardcoded 23-byte form (account@6) was WRONG: the
	# captured real client sends account@2 (id@0 account@2 char@6 login1@10
	# login2@14 tick@18 sex@22). The learned layout is authoritative.
	my $layout = $config{mapLoginLayout};
	# The config stores it as a JSON string (ControlParser writes 'key value').
	# JSON::PP is core Perl (always present); JSON may not be installed.
	$layout = eval { JSON::PP::decode_json($layout) } if ($layout && !ref($layout));
	my $packet;
	if ($layout && ref($layout) eq 'HASH' && $layout->{length}) {
		my $n = $layout->{length};
		my $acc = $layout->{account_offset} // 2;
		my $chr = $layout->{char_offset} // ($acc + 4);
		my $l1  = $layout->{login1_offset} // ($acc + 8);
		my $l2  = $layout->{login2_offset} // ($acc + 12);
		my $tk  = $layout->{tick_offset} // ($acc + 16);
		my $sx  = $layout->{sex_offset} // ($acc + 20);
		# Build a zero-filled buffer of length $n, then place each field.
		$packet = pack('v', 0x0436) . ("\x00" x ($n - 2));
		substr($packet, $acc, 4) = pack('V', $accountID);
		substr($packet, $chr, 4) = pack('V', $charID);
		substr($packet, $l1, 4)  = pack('V', $sessionID);
		substr($packet, $l2, 4)  = pack('V', 0);  # loginID2 (unused by the server)
		substr($packet, $tk, 4)  = pack('V', getTickCount());
		substr($packet, $sx, 1)  = pack('C', $sex);
		$msg = $packet;
	} elsif ($mlen == 26) {
		# 26-byte layout: id + 5 longs (extra 0 tick slot) + sex + 3 pad
		$packet = pack(
			'v V5 C a3',
			0x0436,
			$accountID,
			$charID,
			$sessionID,
			time(),
			0,
			$sex,
			'',
		);
	} elsif ($mlen == 23) {
		# 23-byte form — AGNOSTIC (BQ 2026-08-31): do NOT hardcode a specific
		# server's field offsets. The capture consumer LEARNS the real layout
		# (mapLoginLayout) from the live server's packets and that is the
		# authoritative adaptation. This fallback is only a cold-start probe —
		# it uses the standard rAthena field order (id + account + char +
		# login1 + tick + sex) padded to 23 bytes. The learned layout overrides
		# it once available.
		$packet = pack(
			'v V V V V C a2',
			0x0436,
			$accountID,  # @2-5
			$charID,     # @6-9
			$sessionID,  # @10-13 (loginID1)
			getTickCount(),  # @14-17 (client tick)
			$sex,        # @18
			'',          # pad @19-22
		);
	} else {
		# 19-byte layout: id + 4 longs + sex (the SOURCE's canonical form)
		$packet = pack(
			'v V4 C',
			0x0436,
			$accountID,
			$charID,
			$sessionID,
			time(),
			$sex,
		);
	}
	$msg = $packet;
	$self->sendToServer($msg);
	debug "Sent sendMapLogin (0x0436, " . length($msg) . " bytes, mapLoginLength=$mlen, accountID=$accountID, charID=$charID)\n", 'sendPacket';
}

1;