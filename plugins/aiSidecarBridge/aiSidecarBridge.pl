package aiSidecarBridge;

use strict;
use Data::Dumper;
$Data::Dumper::Terse = 1;
use warnings;
use feature 'state';

use Commands;
use FileParsers qw(parseConfigFile);
use Globals qw(%config $char $field @ai_seq $net %monsters %players %npcs $monstersList $playersList $npcsList %timeout $messageSender);
use IO::Socket::INET;
use Log qw(debug message warning);
use Network;
use Plugins;
use Scalar::Util qw(reftype);
use Settings;
use Time::HiRes qw(alarm time usleep);
use Cwd;
use File::Basename;
use lib dirname(__FILE__);
use CircuitBreaker;
use ConnectionMetrics;
use HTTPClient;
use StateBuilders;

# Anti-detection: random delay to simulate human reaction time
# Each bot gets a unique behavior profile to avoid pattern detection
my $ANTI_DETECTION_ENABLED = 0;
my $ANTI_DETECTION_MIN_DELAY_MS = 200;
my $ANTI_DETECTION_MAX_DELAY_MS = 600;

# Behavior profile per bot — randomized at plugin load to give each bot
# a unique "personality" that looks human. Config-driven via sidecar push.
# Profile affects: command pacing, reaction time, movement patterns, heal timing
our %_behavior_profile;
sub _init_behavior_profile {
	my $bot_id = _bot_id();
	return if $_behavior_profile{$bot_id};
	
	# Seed from bot name for deterministic but unique profile per bot
	my $seed = 0;
	for my $c (split //, $bot_id) { $seed += ord($c); }
	srand($seed);
	
	$_behavior_profile{$bot_id} = {
		# Command pacing: base delay + random jitter (ms)
		# Different per bot: Bot1=fast, Bot2=medium, Bot3=slow
		cmd_min_delay_ms => 150 + int(rand(300)),
		cmd_max_delay_ms => 400 + int(rand(400)),
		
		# Reaction time: how fast bot responds to events (ms)
		# Human reaction: 200-500ms average
		reaction_time_ms => 150 + int(rand(350)),
		
		# Heal timing: slight delay before using potion (ms)
		# Humans don't heal instantly — they notice HP drop, then act
		heal_reaction_ms => 100 + int(rand(300)),
		
		# Movement pattern: how often bot changes direction
		# 0=straight line (bot-like), 1=some variation, 2=erratic (human-like)
		movement_variation => int(rand(3)),
		
		# Sit duration: how long bot sits to regen (seconds)
		# Humans don't sit for exactly the same time every time
		sit_min_seconds => 3 + int(rand(5)),
		sit_max_seconds => 8 + int(rand(10)),
		
		# Attack pattern: slight delay before attacking new target
		attack_delay_ms => 50 + int(rand(200)),
		
		# Profile name for logging
		profile_name => ('aggressive', 'cautious', 'balanced', 'lazy', 'twitchy')[int(rand(5))],
	};
	
	my $p = $_behavior_profile{$bot_id};
	debug "[aiSidecarBridge] behavior_profile: bot=$bot_id profile=$p->{profile_name} cmd_delay=$p->{cmd_min_delay_ms}-$p->{cmd_max_delay_ms}ms reaction=$p->{reaction_time_ms}ms heal_reaction=$p->{heal_reaction_ms}ms movement=$p->{movement_variation} sit=$p->{sit_min_seconds}-$p->{sit_max_seconds}s\n", 'aiSidecarBridge', 2;
}

# Get behavior profile for current bot
sub _profile {
	my $bot_id = _bot_id();
	_init_behavior_profile() if !$_behavior_profile{$bot_id};
	return $_behavior_profile{$bot_id};
}

# ── mon_control persistence (dedup) ──
# Writes a `mon_control <entry>` line to every bot profile's mon_control.txt (and the
# shared control/mon_control.txt) ONLY if the exact line isn't already present. Without
# this, the sidecar re-emitting `mon_control <name> -1 0 0` every map/poll appended the
# SAME line hundreds of times, ballooning the file. OpenKore uses last-match, so the
# duplicates were harmless functionally but were unbounded config churn.
sub _append_mon_control_dedup {
	my ($entry) = @_;
	$entry = '' if !defined $entry;
	$entry =~ s/^\s+|\s+$//g;
	return if $entry eq '';
	my @_mc_files = glob(Cwd::cwd() . '/.bot_profiles/*/control/mon_control.txt');
	push @_mc_files, Cwd::cwd() . '/control/mon_control.txt';
	for my $_mc_file (@_mc_files) {
		my $_already = 0;
		if (open my $_rfh, '<', $_mc_file) {
			while (my $_rl = <$_rfh>) {
				$_rl =~ s/[\r\n]+//g;
				$_rl =~ s/^\s+|\s+$//g;
				if ($_rl eq $entry) { $_already = 1; last; }
			}
			close $_rfh;
		}
		next if $_already;   # already present — skip (dedup)
		if (open my $_wf, '>>', $_mc_file) {
			print $_wf "$entry\n";
			close $_wf;
			debug "[mon_control] appended (dedup) '$entry' to $_mc_file\n", 'aiSidecarBridge', 1;
		} else {
			warning "[mon_control] cannot write to $_mc_file: $!\n", 'aiSidecarBridge', 1;
		}
	}
}

# Human-like command delay with per-bot profile
sub _human_cmd_delay_ms {
	my $p = _profile();
	return $p->{cmd_min_delay_ms} + int(rand($p->{cmd_max_delay_ms} - $p->{cmd_min_delay_ms} + 1));
}

# Human-like reaction delay before heal
sub _human_heal_delay_ms {
	my $p = _profile();
	return $p->{heal_reaction_ms} + int(rand(200));
}

# Human-like reaction delay before attack
sub _human_attack_delay_ms {
	my $p = _profile();
	return $p->{attack_delay_ms} + int(rand(150));
}

# Reflex cooldown tracking - prevent survival reflexes from firing every cycle
our %_last_reflex_fire_ms = ();

# Committed action guard - prevents conflicting commands within 30s window
our %_committed_actions = ();
our %_committed_commands = ();  # normalized command text => last-executed ms
our $_last_move_humanize_ms = 0;
our %_pending_batch_actions = ();
our $COMMITTED_ACTION_COOLDOWN_MS = 30000;

# ── Skill delay tracking ──
# Tracks per-skill cooldowns: skill_name => timestamp_ms when it becomes available again
our %_skill_delays = ();
# Tracks cast time per skill: skill_name => cast_time_ms
our %_cast_times = ();
# Tracks after-cast delay per skill: skill_name => after_cast_delay_ms
our %_after_cast_delays = ();
# Tracks when a skill was last used: skill_name => timestamp_ms
our %_last_skill_use_ms = ();
# NPC dialog state tracking (declared here so the buy-rewrite at ~line 5350
# can reference it; re-initialized later with full defaults).
our %_npc_dialog_state = ();
# Tracks if currently casting (don't send movement during cast)
our $_is_casting = 0;
our $_casting_until_ms = 0;
# Max actions per poll
our $MAX_ACTIONS_PER_POLL = 5;

# ── Party coordination state ──
# Tracks active party buffs: buff_name => { source_bot, expires_at_ms }
our %_party_active_buffs = ();
# Tracks party member positions: member_name => { x, y, map, updated_at_ms }
our %_party_member_positions = ();
# Last time we sent a party buff request
our $_last_party_buff_request_ms = 0;
# Party buff request cooldown (ms)
our $PARTY_BUFF_REQUEST_COOLDOWN_MS = 5000;

# ── MVP hunting state ──
# Tracks MVP spawn timers: mvp_name => { killed_at_ms, respawn_window_start, respawn_window_end, map }
our %_mvp_spawn_timers = ();
# Tracks current MVP target (if any)
our $_current_mvp_target = '';
# Last time we checked MVP status
our $_last_mvp_check_ms = 0;
# MVP check interval (ms)
our $MVP_CHECK_INTERVAL_MS = 10000;

# ── WOE state ──
# Is WOE currently active?
our $_woe_active = 0;
# Last WOE time check
our $_last_woe_check_ms = 0;
# WOE check interval (ms)
our $WOE_CHECK_INTERVAL_MS = 30000;
# Emperium target ID (if any)
our $_emperium_target_id = '';
# Last escape reflex time for WOE
our $_last_woe_escape_ms = 0;
# WOE escape cooldown (ms)
our $WOE_ESCAPE_COOLDOWN_MS = 10000;

# ── Batch completion tracking ──
# Tracks completed batches so we can report them to the sidecar
our %_completed_batches = ();

# ── NPC shop data ──
# Persistent hash of NPC shop data: npc_name => { map, items => [ { name, price, type } ] }
our %_npc_shop_data = ();
# Last time we scanned NPC shops
our $_last_npc_shop_scan_ms = 0;
# NPC shop scan interval (ms)
our $NPC_SHOP_SCAN_INTERVAL_MS = 60000;

# ── Player vendor data ──
# Persistent hash of player vendor data: player_name => { map, x, y, title, items => [ { name, price, amount } ] }
our %_player_vendor_data = ();
# Last time we scanned player vendors
our $_last_player_vendor_scan_ms = 0;
# Player vendor scan interval (ms)
our $PLAYER_VENDOR_SCAN_INTERVAL_MS = 30000;

# ── Game time tracking ──
# Last reported game time
our $_last_reported_game_time = '';
# Last game time check
our $_last_game_time_check_ms = 0;
# Game time check interval (ms)
our $GAME_TIME_CHECK_INTERVAL_MS = 60000;

# ── Server announcement tracking ──
# Last time we flushed announcements
our $_last_announcement_flush_ms = 0;
# Announcement flush interval (ms)
our $ANNOUNCEMENT_FLUSH_INTERVAL_MS = 2000;
# Queue of pending announcements
our @_pending_announcements = ();

# ── Dispel tracking ──
# Tracks dispel events: { detected_at_ms, map, source_id, buffs_lost => [] }
our @_dispel_events = ();
# Last dispel check time
our $_last_dispel_check_ms = 0;
# Dispel check interval (ms)
our $DISPEL_CHECK_INTERVAL_MS = 5000;
# Previous buff count for detecting dispels
our $_prev_buff_count = 0;
# Previous buff list for detecting which buffs were lost
our @_prev_buff_names = ();

# ── Hardcoded safety net: White Potion ──
# This is the ABSOLUTE LAST RESORT item — always available as fallback.
# Used only when dynamic heal cache fails and HP is critically low.
my $HARDCODED_FALLBACK_ITEM = _cfg('aiSidecar_fallbackItem', _cfg('aiSidecar_fallbackHealItem', 'White Potion'));

Plugins::register(
	'aiSidecarBridge',
	'Local HTTP sidecar bridge for OpenKore AI IPC',
	\&on_unload,
	\&on_reload,
);

my $hooks = Plugins::addHooks(
	['start3', \&on_start3, undef],
	['mainLoop_pre', \&on_mainLoop_pre, undef],
	['mainLoop_post', \&on_mainLoop_post, undef],
	['Network::stateChanged', \&on_network_state_changed, undef],
	['add_monster_list', \&on_add_actor_list_probe, 'monster'],
	['add_player_list', \&on_add_actor_list_probe, 'player'],
	['add_npc_list', \&on_add_actor_list_probe, 'npc'],
	['packet_pre/public_chat', \&on_packet_hook, 'packet_pre.public_chat'],
	['packet_pre/private_message', \&on_packet_hook, 'packet_pre.private_message'],
	['packet_pre/party_chat', \&on_packet_hook, 'packet_pre.party_chat'],
	['packet_pre/guild_chat', \&on_packet_hook, 'packet_pre.guild_chat'],
	['packet_pre/system_chat', \&on_packet_hook, 'packet_pre.system_chat'],
	['packet_pre/map_change', \&on_packet_hook, 'packet_pre.map_change'],
	['packet_pre/skill_use', \&on_packet_hook, 'packet_pre.skill_use'],
	['packet_pre/area_spell', \&on_packet_hook, 'packet_pre.area_spell'],
	['packet/public_chat', \&on_packet_hook, 'packet.public_chat'],
	['packet/private_message', \&on_packet_hook, 'packet.private_message'],
	['packet/party_chat', \&on_packet_hook, 'packet.party_chat'],
	['packet/guild_chat', \&on_packet_hook, 'packet.guild_chat'],
	['packet/system_chat', \&on_packet_hook, 'packet.system_chat'],
	['packet/map_change', \&on_packet_hook, 'packet.map_change'],
	['packet/skill_use', \&on_packet_hook, 'packet.skill_use'],
	['packet/skill_use_no_damage', \&on_packet_hook, 'packet.skill_use_no_damage'],
	['packet/area_spell', \&on_packet_hook, 'packet.area_spell'],
	['packet/area_spell_disappears', \&on_packet_hook, 'packet.area_spell_disappears'],
	['packet_privMsg', \&on_chat_message, 'pm'],
	['pre/npc_talk_responses', \&on_npc_menu, undef],
	['packet_pubMsg', \&on_chat_message, 'publicchat'],
	['packet_partyMsg', \&on_chat_message, 'partychat'],
	['packet_guildMsg', \&on_chat_message, 'guildchat'],
	['packet_sysMsg', \&on_chat_message, 'systemchat'],
	['packet_mapChange', \&on_legacy_packet_hook, 'packet_legacy.map_change'],
	['packet_skilluse', \&on_legacy_packet_hook, 'packet_legacy.skill_use'],
	['packet_areaSpell', \&on_legacy_packet_hook, 'packet_legacy.area_spell'],
	['post_configModify', \&on_post_config_modify, undef],
	['post_bulkConfigModify', \&on_post_bulk_config_modify, undef],
	['Commands::run/pre', \&on_command_intercept, undef],
	['Commands::run/post', \&on_command_run_post, undef],
);

my $control_handle;
my $policy_handle;

my %bridge_cfg;
my %bridge_policy;
my @policy_allow;
my @policy_deny;
my %ml_pending_outcome;

my $registered = 0;
my $_registered_char_name = '';
my $_last_reregister = 0;
my $next_snapshot_at_ms = 0;
my $next_poll_at_ms = 0;
my $next_ack_at_ms = 0;
my $next_telemetry_at_ms = 0;
my $next_register_at_ms = 0;
my $next_event_ingest_at_ms = 0;
my $next_chat_ingest_at_ms = 0;
my $next_config_ingest_at_ms = 0;
my $next_keepalive_at_ms = 0;
my $next_party_status_at_ms = 0;
my $next_autoequip_at_ms = 0;

# ── charstatus.json real-time state file (2026-08-27) ──
# Monotonic snapshot sequence per bot (rejects stale/out-of-order reads).
my %_charstatus_seq;
# Last written charstatus path per bot (dedup + atomic-write tracking).
my %_charstatus_last_path;
# charstatus.json output directory (default: data/charstatus/ under repo root).
my $_charstatus_dir = '';

# ── New module instances (upgraded IPC, state builders, circuit breaker) ──
my $_circuit_breaker;
my $_connection_metrics;
my $_http_client;
my $_state_builders;

# Pre-existing state variables (undeclared in original)
my $_last_emergency_move = 0;

my @ack_queue;
my %_action_queue;
my @telemetry_queue;
my @event_queue;
my @chat_queue;
my %pending_config_keys;
my %last_warn_at_ms;
my $event_seq = 0;
my $last_ai_seq_top = '';
my $consecutive_poll_failures = 0;
my $consecutive_v2_event_failures = 0;
my %known_actor_ids;
my $last_net_in_game;
my $last_disconnect_at_ms = 0;
my $last_hp;
my $death_count = 0;
my $respawn_state = 'unknown';
# Survival mode persistent variables (not state — state resets on map change)
our $_survival_mode_until_ms = 0;
our $_last_survival_check_ms = 0;
my $last_map_name = '';
my $last_route_signature = '';
my $_last_ai_toggle_ms = 0;
my $_pro_ro_last_lock_set = '';
my $_pro_ro_respawn_ms = 0;  # Timestamp of last respawn
my $_pro_ro_stay_in_town_ms = 0;  # Stay in town until this timestamp (0 = not active)
my $_pro_ro_last_lock_ms = 0;
my $_last_ai_mode = '';
my $route_churn_count = 0;
my $route_failure_count = 0;
my $last_actor_source_probe_log_ms = 0;
my $last_actor_post_parse_probe_log_ms = 0;
my %actor_add_probe_count;
my %actor_add_probe_last_log_ms;
my $consecutive_empty_actor_snapshots = 0;

my $json_available = eval { require JSON::PP; 1; };

sub on_reload {
	_cleanup_runtime();
	on_start3();
}

my $_unloaded_once = 0;

sub on_unload {
	if ($_unloaded_once) { return; }
	$_unloaded_once = 1;
	_cleanup_runtime();
	# CRITICAL: guard delHooks — during shutdown/global destruction a hook
	# handle's HOOKNAME may already be cleared or its INDEX stale, so
	# Plugins::delHook() throws ArgumentException ("Invalid hook handle")
	# and the Assertion "Can't remove undefined item". That die hits
	# ErrorHandler::showError which does <STDIN> — hanging forever under
	# nohup. Swallow it here; the process is exiting anyway.
	eval { Plugins::delHooks($hooks) if defined $hooks; 1; } or do {
		warning "[aiSidecarBridge] delHooks during unload failed (already torn down): $@\n", 'aiSidecarBridge' if defined $@ && $@;
	};
	undef $hooks;
}

sub _cleanup_runtime {
	if (defined $control_handle) {
		Settings::removeFile($control_handle);
		undef $control_handle;
	}
	if (defined $policy_handle) {
		Settings::removeFile($policy_handle);
		undef $policy_handle;
	}

	$registered = 0;
	$next_snapshot_at_ms = 0;
	$next_poll_at_ms = 0;
	$next_ack_at_ms = 0;
	$next_telemetry_at_ms = 0;
	$next_register_at_ms = 0;
	$next_event_ingest_at_ms = 0;
	$next_chat_ingest_at_ms = 0;
	$next_config_ingest_at_ms = 0;
	@ack_queue = ();
	@telemetry_queue = ();
	@event_queue = ();
	@chat_queue = ();
	%pending_config_keys = ();
	%last_warn_at_ms = ();
	$event_seq = 0;
	$last_ai_seq_top = '';
	$consecutive_poll_failures = 0;
	$consecutive_v2_event_failures = 0;
	%known_actor_ids = ();
	$last_net_in_game = undef;
	$last_disconnect_at_ms = 0;
	$last_hp = undef;
	$death_count = 0;
	$respawn_state = 'unknown';
	$last_map_name = '';
	$last_route_signature = '';
	$route_churn_count = 0;
	$route_failure_count = 0;
	$last_actor_source_probe_log_ms = 0;
	$last_actor_post_parse_probe_log_ms = 0;
	%actor_add_probe_count = ();
	%actor_add_probe_last_log_ms = ();
	$consecutive_empty_actor_snapshots = 0;

	# Cleanup upgraded IPC modules
	if ($_http_client) {
		$_http_client->close();
		undef $_http_client;
	}
	if ($_circuit_breaker) {
		$_circuit_breaker->reset();
		undef $_circuit_breaker;
	}
	if ($_connection_metrics) {
		$_connection_metrics->reset();
		undef $_connection_metrics;
	}
	if ($_state_builders) {
		undef $_state_builders;
	}
}

sub on_network_state_changed {
	my (undef, $args) = @_;
	return unless $args && ref($args) eq 'HASH';
	my $state = $args->{state};
	return unless defined $state;
	warning "[aiSidecarBridge] Network state changed: $state\n", 'aiSidecarBridge', 3;
	# State 0=disconnected, 1=connecting, 2=connected, 3=disconnecting
	if ($state == 2) {
		# Try to register with sidecar immediately
		eval { _attempt_register(); 1; };
		# Send initial discovery tables
		eval { _send_discovery_data(); 1; };
	} elsif ($state == 0) {
		# Clear action queue on disconnect
		my $bot_id = _bot_id();
		delete $_action_queue{$bot_id} if exists $_action_queue{$bot_id};
	}
}

sub on_start3 {
	# NOTE: sitAuto NOT disabled here — let the heuristic control it
	# via the config audit which sets sitAuto_hp_lower=30, sitAuto_hp_upper=60.
	# The old code disabled sitAuto entirely, which prevented the bot from
	# sitting to regen HP (leading to 9% HP death spirals).
	
	# Disable useSelf_item at startup — delete ALL entries
	# OpenKore's useSelf_item system calls $messageSender->sendItemUse() directly,
	# bypassing Actor::Item::use(). The "on cooldown, skipping" message is from the server.
	# Deleting entries at startup prevents the system from ever running.
	{
		my $_usi_idx = 0;
		while (exists $::config{"useSelf_item_$_usi_idx"}) {
			delete $::config{"useSelf_item_$_usi_idx"};
			delete $::config{"useSelf_item_${_usi_idx}_timeout"};
			delete $::config{"useSelf_item_${_usi_idx}_cooldown"};
			$_usi_idx++;
		}
		# Also set the global timeout to prevent the system from running
		$timeout{ai_item_use_auto}{time} = time;
		$timeout{ai_item_use_auto}{timeout} = 300;
	}
	# sitAuto controlled by heuristic - not overridden here
	# sitAuto_hp_upper controlled by heuristic
	# sitAuto_sp controlled by heuristic
	# sitAuto_sp_upper controlled by heuristic
	# sitAuto_idle controlled by heuristic
	# STALE PORTAL FILTER: remove known-stale NPC teleport entries from portalsLOS.txt
	# OpenKore regenerates this file at runtime, so stale entries keep reappearing
	my $_portals_file = Settings::getTableFilename("portalsLOS.txt");
	if (-f $_portals_file) {
		my $_content = '';
		if (open(my $_fh, '<', $_portals_file)) {
			local $/;
			$_content = <$_fh>;
			close $_fh;
		}
		if ($_content ne '') {
			my $_changed = 0;
			# Remove lines containing known-stale NPC teleport coordinates
			for my $_pattern (qw(prontera 156 229 prontera 157 40 prontera 157 38 prontera 157 36)) {
				if ($_content =~ /$_pattern/) {
					$_content =~ s/^.*$_pattern.*\n?//gm;
					$_changed = 1;
				}
			}
			if ($_changed) {
				if (open(my $_fh, '>', $_portals_file)) {
					print $_fh $_content;
					close $_fh;
					debug "[stale_portal] filtered known-stale entries from $_portals_file\n", 'aiSidecarBridge', 1;
				}
			}
		}
	}


	if (!$json_available) {
# 		warning "[aiSidecarBridge] JSON::PP is unavailable, bridge is disabled (fail-open).\n";
		return;
	}

	$control_handle = Settings::addControlFile(
		'ai_sidecar.txt',
		loader => [\&_load_bridge_config, \%bridge_cfg],
		mustExist => 0,
	);
	$policy_handle = Settings::addControlFile(
		'ai_sidecar_policy.txt',
		loader => [\&_load_bridge_policy, \%bridge_policy],
		mustExist => 0,
	);

	Settings::loadByHandle($control_handle);
	Settings::loadByHandle($policy_handle);

	my $now = _now_ms();
	$next_snapshot_at_ms = $now;
	$next_poll_at_ms = $now;
	$next_ack_at_ms = $now;
	$next_telemetry_at_ms = $now;
	$next_register_at_ms = $now;
	$next_event_ingest_at_ms = $now;
	$next_chat_ingest_at_ms = $now;
	$next_config_ingest_at_ms = $now;
	$next_party_status_at_ms = $now;

	_attempt_register('start3');
	my $_reg_char_name = $char ? ($char->{name} || '') : '';
	if ($registered && $_reg_char_name ne '' && $_reg_char_name ne $_registered_char_name) {
		$registered = 0;
		$next_register_at_ms = _now_ms();
	}

	# ── Initialize upgraded IPC modules ──
	$_circuit_breaker = CircuitBreaker->new(
		threshold => 10,
		name      => 'zmq_push',
	);
	$_connection_metrics = ConnectionMetrics->new(
		max_latency_samples => 100,
		window_seconds      => 300,
	);
	$_http_client = HTTPClient->new(
		zmq_enabled       => _cfg_bool('aiSidecar_zmqEnabled', 0),  # HTTP-only sidecar
		zmq_address       => $ENV{SIDECAR_ZMQ_ADDR} || 'tcp://127.0.0.1:5559',
		http_base_url     => _cfg('aiSidecar_baseUrl', 'http://127.0.0.1:18081'),
		zmq_connect_ms    => 500,
		zmq_linger_ms     => 100,
		http_connect_ms   => _cfg_int('aiSidecar_connectTimeoutMs', 2000),
		http_io_ms        => _cfg_int('aiSidecar_ioTimeoutMs', 5000),
		json_encode_cb    => sub { JSON::PP::encode_json($_[0]) },
		debug_log_cb      => sub { debug($_[0], 'HTTPClient', 2) },
		warn_log_cb       => sub { warning($_[0]) },
		circuit_breaker   => $_circuit_breaker,
		metrics           => $_connection_metrics,
	);
	$_state_builders = StateBuilders->new(
		debug_log_cb => sub { debug($_[0], 'StateBuilders', 2) },
		max_items    => _cfg_int('aiSidecar_maxItems', 200),
		max_actors   => _cfg_int('aiSidecar_maxActors', 24),
	);
	debug "[aiSidecarBridge] upgraded IPC modules initialized (ZMQ + HTTP fallback, circuit breaker, metrics, state builders)\n", 'aiSidecarBridge', 1;
}

sub _load_profile_to_char {
	my $resp = _http_get_json('/v1/fleet/bots');
	return if !$resp || !$resp->{json} || !$resp->{json}{bots};
	my %mapping;
	for my $bot (@{$resp->{json}{bots}}) {
		my $bot_id = $bot->{bot_id} || '';
		next if $bot_id eq '';
		my ($prefix, $profile) = split(':', $bot_id, 2);
		next if !$profile;
		my $char_name = $bot->{attributes}{identity_char_name} || $profile;
		$mapping{$profile} = $char_name;
	}
	%::aiSidecar_profile_to_char = %mapping;
	# Hardcoded fallback for known profiles
	if (!keys %mapping) {
		%::aiSidecar_profile_to_char = (
			kicapmasin => 'openkoreai',
			kicapmasin2 => 'openkoreaiobs',
			kicapmasin3 => 'openkoreaihuman',
		);
		debug "bridge_profile_to_char: using hardcoded fallback (3 mappings)\n", 'aiSidecarBridge', 1;
	}
	debug "bridge_profile_to_char: loaded " . scalar(keys %mapping) . " mappings\n", 'aiSidecarBridge', 1;
}
sub _check_reregister {
	my $current_name = $char && $char->{name} ? $char->{name} : ($config{username} || '');
	if ($registered && $current_name ne '' && $current_name ne $_registered_char_name) {
		debug "bridge_reregister: char name changed from '$_registered_char_name' to '$current_name'\n", 'aiSidecarBridge', 1;
		$_registered_char_name = $current_name;
		$registered = 0;
		$next_register_at_ms = _now_ms();
	}
	# Force re-register every 30s to catch stale empty-name registration
	if ($registered && _now_ms() - ($_last_reregister || 0) > 30000) {
		$_last_reregister = _now_ms();
		debug "bridge_reregister: 30s heartbeat re-registration\n", 'aiSidecarBridge', 1;
		$registered = 0;
		$next_register_at_ms = _now_ms();
	}
}
sub on_mainLoop_pre {
    return unless _bridge_enabled();
    debug "[on_mainLoop_pre] bridge enabled check\n", 'aiSidecarBridge', 1;
    # NOTE: attackAuto override removed — heuristic handles this correctly.
    # The old code force-set attackAuto=0 when 0 potions on hunting map,
    # which ran EVERY cycle and overrode the heuristic's attackAuto=3.
    # This created a deadlock: can't attack → can't earn zeny → can't buy potions.
    # The heuristic's cold start pipeline handles potion buying and attack config.
    my $now = _now_ms();
    # ── LEAK DIAGNOSTIC (throttled, always-fires) ──
    # Correlate RSS growth with Field/PathFinding object leaks. Runs on the
    # main-loop tick. ~5MB/Field, ~5.6MB/PathFinding session.
    state $_last_leaklog_ms = 0;
    if ($now - $_last_leaklog_ms > 10000) {
        $_last_leaklog_ms = $now;
        my $_fs = (Field->can('stats') ? Field->stats() : {});
        my $_ps = (PathFinding->can('stats') ? PathFinding->stats() : {});
        my $_rss = 0;
        if (open my $_fh, '<', "/proc/self/status") {
            while (<$_fh>) { $_rss = $1/1024 if /^VmRSS:\s+(\d+)/; }
        }
        warning sprintf(
            "[leakdiag] rss_mb=%.0f fields_live=%d fields_created=%d pf_live=%d pf_created=%d | Perl: ai_seq=%d monsters=%d players=%d items=%d npcs=%d portals=%d chars=%d friends=%d\n",
            $_rss, 0+($_fs->{live}||0), 0+($_fs->{created}||0),
            0+($_ps->{live}||0), 0+($_ps->{created}||0),
            0+scalar(@Globals::ai_seq), 0+scalar(keys %Globals::monsters), 0+scalar(keys %Globals::players),
            0+scalar(@Globals::itemsID), 0+scalar(@Globals::npcsID), 0+scalar(keys %Globals::portals_lut),
            0+scalar(@Globals::chars), 0+scalar(@Globals::friendsID),
        ), 'aiSidecarBridge', 1;
    }

	if (_cfg_bool('aiSidecar_snapshotEnabled', 1) && $now >= $next_snapshot_at_ms) {
	    my $snap_base = _cfg_int('aiSidecar_snapshotIntervalMs', 500);
	    my $snap_jitter = int(rand(1 + $snap_base * 0.2));
	    $snap_jitter = -$snap_jitter if int(rand(2)) == 0;
	    $next_snapshot_at_ms = $now + $snap_base + $snap_jitter;
	    _send_snapshot();
	    _check_bridge_reflexes();
	# ── Re-register if char name became available ──
	_check_reregister();
	}

	_track_lifecycle_transitions();
	_track_ai_sequence_transition();
	
	# ── DISABLE useSelf_item ENTIRELY: safety net in main loop ──
	# Deleted at startup in on_start3, but OpenKore may re-read config.
	# This safety net runs every cycle to catch any re-created entries.
	if ($char) {
		state $_last_useSelf_disable_ms = 0;
		my $_now_ms = _now_ms();
		if ($_now_ms - $_last_useSelf_disable_ms > 60000) {
			$_last_useSelf_disable_ms = $_now_ms;
			my $_usi_idx = 0;
			while (exists $::config{"useSelf_item_$_usi_idx"}) {
				delete $::config{"useSelf_item_$_usi_idx"};
				delete $::config{"useSelf_item_${_usi_idx}_timeout"};
				delete $::config{"useSelf_item_${_usi_idx}_cooldown"};
				$_usi_idx++;
			}
			$timeout{ai_item_use_auto}{time} = time;
			$timeout{ai_item_use_auto}{timeout} = 300;
		}
	}
	
	# ── PREVENT TELEPORT AT LOW HP: disable OpenKore internal teleport ──
	if ($char) {
	    # Disable OpenKore's internal teleport at 30% HP - heuristic handles HP management
	    # teleportAuto controlled by heuristic - not overridden here
	    # $::config{attackAuto_maxDistance} = 3;  # heuristic controls this
	    # $::config{attackAuto_unstuck} = 1;  # heuristic controls this
	    # Override sitAuto config every cycle
	    # sitAuto controlled by heuristic - not overridden here
	    # sitAuto_hp_upper controlled by heuristic
	    # teleportAuto controlled by heuristic - not overridden here
	    # sitAuto_sp controlled by heuristic
	    # sitAuto_sp_upper controlled by heuristic
	    # sitAuto_idle controlled by heuristic
	    # sitAuto_over_50 controlled by heuristic
	    # Force stand if sitting
	    if ($char->{sitting}) {
	        # stand controlled by heuristic
	        # ai auto controlled by heuristic
	    }
	}
}


sub on_mainLoop_post {
	# ── PORTAL EXIT DETECTION DISABLED ──
	# Disabled: moving bot from portal exit creates a warp loop
	# Heuristic handles positioning on hunting maps

        # Override attack distances (also disable auto-detection)
        # attackDistance controlled by heuristic - not overridden here
        # attackMaxDistance controlled by heuristic - not overridden here
        # attackDistanceAuto controlled by heuristic - not overridden here  # Prevent server packet from overriding
        # ── DISABLE SIT IN AI: prevent OpenKore's internal AI from sitting ──
        if ($char) {
            # Override sitAuto config every cycle to prevent AI from re-enabling it
            # sitAuto controlled by heuristic - not overridden here
            # sitAuto_hp_upper controlled by heuristic
            # sitAuto_sp controlled by heuristic
            # sitAuto_sp_upper controlled by heuristic
            # sitAuto_idle controlled by heuristic
            # sitAuto_over_50 controlled by heuristic
            # Also clear sit from AI sequence if present
            # AI sequence sit removal controlled by heuristic
        }
        

        # ── FORCE STAND: if bot is sitting, force stand to prevent sit-spam ──
        # Force stand if HP >= 50%, OR in town (cold start needs to move),
        # OR on hunting map with 0 potions (can't heal anyway).
        if ($char && $char->{sitting}) {
            my $_fs_hp = $char->{hp} || 0;
            my $_fs_hp_max = $char->{hp_max} || 1;
            my $_fs_hp_pct = $_fs_hp_max > 0 ? int($_fs_hp * 100 / $_fs_hp_max) : 0;
            my $_fs_map = lc($field->name() || '');
            my @_fs_town_maps = qw(pronenta morocc geffen payon aldebaran alberta izlude);
            my $_fs_is_town = grep { $_fs_map =~ /\Q$_/ } @_fs_town_maps;
            # Check if bot has 0 potions (cold start or emergency)
            my $_fs_has_potions = 0;
            if (@{_char_inventory($char)}) {
                for my $_fi (@{_char_inventory($char)}) {
                    next unless $_fi;
                    my $_fn = $_fi->{name} || '';
                    if ($_fn =~ /potion|herb|fruit|berry|red|orange|white|yellow|blue|green/i) {
                        $_fs_has_potions = 1;
                        last;
                    }
                }
            }
            # Per RULE.md: reflex uses commands only, config changes through heuristic.
            # Force stand when HP >= 20% (healthy enough to fight) AND (HP >= 50% OR in town).
            if (($_fs_hp_pct >= 50 || $_fs_is_town) && $_fs_hp_pct >= 20) {
                warning "[force_stand] bot sitting (HP=$_fs_hp_pct% town=$_fs_is_town pots=$_fs_has_potions), forcing stand\n", 'aiSidecarBridge', 1;
                # Let bot use OpenKore's internal sit AI for HP regen
                # sitAuto_hp_lower = default (30%) — bot will sit when HP < 30%
                # sitAuto_hp_upper = default (60%) — bot will stand when HP > 60%
                # Don't disable sitAuto — the bot needs to regen HP!
                eval { Commands::run("stand"); 1 };
                eval { Commands::run("ai auto"); 1 };
            }
        }
        
        # ── EMERGENCY REFLEX: HP critically low + overweight + not in town → walk to Kafra ──
        # Deadlock: can't regen (overweight blocks HP regen), can't attack (low HP), can't farm.
        # Walk to Kafra Employee (290, 224) on same map — deposit heavy items for FREE.
        # Kafra is ~20 steps from spawn on prt_fild05, safe and immediate.
        # Per RULE.md: reflex uses commands only, NEVER config overrides.
        # State variable $_last_emergency_move is declared in the FORE STAND section above.
        if ($char && $field) {
            my $_er_hp = 0;
            if ($char->{hp_max} && $char->{hp} > 0) {
                $_er_hp = $char->{hp} / ($char->{hp_max} || 1);
            }
            my $_er_weight = 0;
            if ($char->{weight_max} && $char->{weight} > 0) {
                $_er_weight = $char->{weight} / ($char->{weight_max} || 1);
            }
            my $_er_map = lc($field->baseName() || '');
            $_er_map =~ s/\.gat$//;
            my $_er_is_town = ($_er_map =~ /^prontera|^morocc|^geffen|^payon|^alberta|^aldebaran|^izlude|^comodo$/i);
            # Cooldown: only send move command every 10s to avoid spam
            my $_em_cooldown = 10;
            if (!$_er_is_town && $_er_hp > 0 && $_er_hp < 0.2 && $_er_weight > 0.7 && (time - $_last_emergency_move) >= $_em_cooldown) {
                $_last_emergency_move = time;
                warning "[emergency] HP=$_er_hp% weight=$_er_weight% on $_er_map — walking to Prontera portal (373,205)\n", 'aiSidecarBridge', 1;
                # Dequeue conflicting AI states — OpenKore ignores move commands when AI is
                # in attack/route/follow states. Clear the queue so move takes effect.
                eval { AI::dequeue(); 1 };
                eval { Commands::run("stand"); 1 };
                # Walk to Prontera portal (373, 205) — once in town, OpenKore's built-in
                # sellAuto/storageAuto trigger naturally. The bridge must NOT issue NPC
                # commands (talknpc) — OpenKore's AI ignores them from idle state.
                eval { Commands::run("move 373 205"); 1 };
                eval { Commands::run("ai auto"); 1 };
            }
        }

        # ── DISABLE useSelf_item FOR POTIONS: handled in on_mainLoop_pre ──
        # (Moved to on_mainLoop_pre to run before useSelf_item fires)
        
        # ── PORTAL EXIT POTION CHECK: fires on MAP CHANGE (portal exit detected), not on a timer ──
        # When bot steps through portal from Prontera to a hunting map, check inventory.
        # If 0 potions, turn around and go back to Prontera immediately.
        # Only runs when bot is IN_GAME (not during STARTING phase).
        if ($char && $field && $net && $net->getState() == Network::IN_GAME) {
            state $_last_portal_map = '';
            my $_pm = lc($field->name());
            $_pm =~ s/\.gat$//;
            if ($_pm ne $_last_portal_map) {
                $_last_portal_map = $_pm;
                if ($_pm =~ /_fild|_dun/i) {
                    # Just arrived on a hunting map — check potions.
                    # IMPORTANT: OpenKore re-syncs inventory on map change — for the
                    # first ~1-3s after entering, _char_inventory() can return EMPTY
                    # even though the char owns potions. Turning back on an
                    # empty-but-unsynced inventory traps the bot in a town↔farm
                    # loop (log: "0 potions ... turning back" every map change
                    # while the Stackable list later shows x300). So an EMPTY
                    # inventory is NOT proof of zero potions: only turn back when
                    # we have BOTH a synced inventory AND genuinely no healing item.
                    my $_hp = 0;
                    my $_inv_synced = 0;
                    if (@{_char_inventory($char)}) {
                        $_inv_synced = 1;
                        for my $_hi (@{_char_inventory($char)}) {
                            next unless $_hi;
                            my $_hn = $_hi->{name} || '';
                            if ($_hn =~ /potion|herb|fruit|berry|red|orange|white|yellow|blue|green/i) {
                                $_hp = 1;
                                last;
                            }
                        }
                    }
                    if ($_inv_synced && !$_hp) {
                        # Check if lockMap matches current map (heuristic wants bot here)
                        my $_pe_lockmap = $::config{lockMap} || '';
                        if ($_pe_lockmap ne '' && $_pe_lockmap eq $_pm) {
                            debug "[portal_exit] 0 potions on $_pm but lockMap=$_pe_lockmap - staying (heuristic override)\n", 'aiSidecarBridge', 1;
                        } else {
                            warning "[portal_exit] 0 potions on $_pm, turning back\n", 'aiSidecarBridge', 1;
                            # sitAuto NOT disabled — let heuristic control it per RULE.md
                            eval { Commands::run("stand"); 1 };
                            eval { Commands::run("move prontera"); 1 };
                        }
                    }
                }
            }
        }

        

        



        


		return unless _bridge_enabled();
		my $now = _now_ms();
		_probe_actor_post_parse($now);

		# Keepalive ping to prevent server timeout (every 5s)
		# rAthena drops idle connections after ~30-40s without packet activity
		# 5s gives a safety margin before the 30s idle-drop threshold
		if ($messageSender && $now >= ($next_keepalive_at_ms || 0)) {
		    $next_keepalive_at_ms = $now + 5000;  # Every 5 seconds
		    $messageSender->sendPing();
		}
		# Periodic party status relay (every 30s) to keep sidecar informed
		if ($registered && $now >= $next_party_status_at_ms) {
		    $next_party_status_at_ms = $now + 30000;
		    _send_party_status('periodic');
		}
		if (!$registered && $now >= $next_register_at_ms) {
	    $next_register_at_ms = $now + _cfg_int('aiSidecar_registerRetryMs', 1000);
	    _attempt_register('retry');
	}

	# ── AUTO-EQUIP UNEQUIPPED WEAPON (reflex, commands-only) ──
	# The sidecar's cold-start latches "weapon present" once any weapon is ever
	# observed, so it never re-emits `equip`. But if the weapon is in inventory
	# yet UNEQUIPPED (verified live: Dmg 1-2 = bare fists), the bot can't kill
	# anything. Periodically (10s) auto-equip the best unequipped weapon held.
	# Commands-only (per RULE.md): never touches config.
	if ($char && $net && $net->getState() == Network::IN_GAME && $now >= ($next_autoequip_at_ms || 0)) {
	    $next_autoequip_at_ms = $now + 10000;  # every 10s
	    my $_ae_map = $field ? lc($field->name()) : '';
	    $_ae_map =~ s/\.gat$//;
	    my $_ae_slot = undef;
	    my $_ae_item = undef;
	    for my $_eq (@{_char_inventory($char)}) {
	        next unless ref($_eq);
	        next if $_eq->{equipped};
	        # Identify a weapon by NAME pattern (agnostic). The fork's `type`/
	        # `type_equip` fields are unreliable across versions (charstatus shows
	        # Main-Gauche type=5, armor type=4 — inverted vs stock). Equip by name
	        # only; OpenKore cmdEquip resolves 'equip <name>' via Actor::Item::get.
	        my $_eq_name = $_eq->{name} || '';
	        next unless $_eq_name =~ /(knife|dagger|sword|blade|rapier|mace|rod|staff|bow|gauche|gladius|claw|katar|axe|hammer|spear|pole|lance|halberd|jitte|huuma|whip|gun|katana|talon|kukri|bagger|main.?gauche)/i;
	        $_ae_item = $_eq;
	        last;
	    }
	    if ($_ae_item) {
	        my $_ae_name = $_ae_item->{name} || '';
	        # This fork's command is `eq` (NOT `equip` — Commands.pm registers
	        # ['eq', ..., \&cmdEquip]; 'equip' → "Unknown command"). Verified live:
	        # every `equip <name>` logged Unknown command → weapon never equipped.
	        my $_ae_ok = eval { Commands::run("eq $_ae_name"); 1 };
	        warning "[autoequip] equipping held weapon '$_ae_name' on $_ae_map (${\\($_ae_ok ? 'OK' : 'FAILED')})\n", 'aiSidecarBridge', 1;
	    }
	}

	if (_cfg_bool('aiSidecar_actionPollEnabled', 1) && $now >= $next_poll_at_ms) {
		debug "[on_mainLoop_post] polling (now=$now next=$next_poll_at_ms)\n", 'aiSidecarBridge', 1;
	    my $poll_ok = _poll_next_action();
	    my $base_delay_ms = _cfg_int('aiSidecar_pollIntervalMs', 100);
	    my $jitter_pct = _cfg_int('aiSidecar_pollJitterPct', 30);
	    my $jitter = int(rand($base_delay_ms * $jitter_pct / 100));
	    $jitter = -$jitter if int(rand(2)) == 0;
	    my $next_delay_ms = $poll_ok ? ($base_delay_ms + $jitter) : _poll_failure_delay_ms();
	    $next_poll_at_ms = $now + $next_delay_ms;
	}

	if (_cfg_bool('aiSidecar_ackEnabled', 1) && $now >= $next_ack_at_ms) {
		$next_ack_at_ms = $now + _cfg_int('aiSidecar_ackRetryMs', 500);
		_flush_ack_queue();
	}

	if (_cfg_bool('aiSidecar_telemetryEnabled', 1) && $now >= $next_telemetry_at_ms) {
		$next_telemetry_at_ms = $now + _cfg_int('aiSidecar_telemetryIntervalMs', 1000);
		_flush_telemetry_queue();
	}

	if (_cfg_bool('aiSidecar_v2Enabled', 1) && _cfg_bool('aiSidecar_configIngestEnabled', 1) && $now >= $next_config_ingest_at_ms) {
		$next_config_ingest_at_ms = $now + _cfg_int('aiSidecar_configIngestIntervalMs', 2000);
		_flush_config_updates();
	}

	if (_cfg_bool('aiSidecar_v2Enabled', 1) && _cfg_bool('aiSidecar_chatIngestEnabled', 1) && $now >= $next_chat_ingest_at_ms) {
		$next_chat_ingest_at_ms = $now + _cfg_int('aiSidecar_chatIngestIntervalMs', 700);
		_flush_chat_queue();
	}

	if (_cfg_bool('aiSidecar_v2Enabled', 1) && _cfg_bool('aiSidecar_eventIngestEnabled', 1) && $now >= $next_event_ingest_at_ms) {
		my $event_ok = _flush_event_queue();
		my $next_delay_ms = $event_ok ? _cfg_int('aiSidecar_eventIngestIntervalMs', 500) : _event_ingest_failure_delay_ms();
		$next_event_ingest_at_ms = $now + $next_delay_ms;
	}
		# ── Default survival auto-grind loop (bottom-up fallback) ──
		if (_cfg_bool('aiSidecar_survivalEnabled', 1) && _bridge_enabled() && _bridge_enabled()) {
			# STRIPPED: _apply_bot_config, _discover_shops, _discover_portals, _send_discovery_data, _survival_check removed
			# All config management and survival logic is handled by the sidecar heuristic service.
			# Bridge only does: snapshot forwarding, command execution, emergency sit.

        # Keep lockMap set to hunting map at ALL times (not just when on one)
        # This prevents OpenKore's internal AI from generating "move prontera" endlessly
        if ($char) {
            my $_sm_map = lc($char->{map} || '');
            $_sm_map =~ s/\.gat$//;
            my $_hunt_map = _cfg('aiSidecar_huntingMap', 'prt_fild05') || 'prt_fild05';
            if ($_sm_map =~ /^[a-z]+_fild/) {
                # On hunting map: ensure lockMap matches current map
                # BUT respect heuristic's lockMap if it was explicitly set via sidecar
                # Also respect if the bot is en route to a different map (lockMap != current map)
                my $_heuristic_lockmap = $::config{'_sidecar_set_lockMap'} || $::config{'_sidecar_set_lockmap'} || '';
                if (!defined $::config{lockMap} || ($::config{lockMap} ne $_sm_map && !$_heuristic_lockmap)) {
                    $::config{lockMap} = $_sm_map;
                }
                # PORTAL EXIT REFLEX: if bot is at portal exit, move to center
                # Works for any map (field or dungeon)
                if ($_sm_map eq 'prt_fild05' || $_sm_map eq 'pay_dun00' || $_sm_map eq 'gef_dun00') {
                    state $_portal_exit_last_move_ms = 0;
                    # Get position from any available source
                    my $_px = 0; my $_py = 0;
                    if (defined $char->{pos_to}) { $_px = $char->{pos_to}{x} || 0; $_py = $char->{pos_to}{y} || 0; }
                    if ($_px == 0 && defined $char->{pos}->{x}) { $_px = $char->{pos}{x}; $_py = $char->{pos}{y}; }
                    if ($_px == 0 && defined $char->{x}) { $_px = $char->{x}; $_py = $char->{y}; }
                    if ($_px == 0 && defined $field) { my ($fx, $fy) = $field->base(); $_px = $fx; $_py = $fy; }
                    # If position is near portal exit, move to center
                    if ($_px > 360 || ($_px == 0 && $_py == 0)) {
                        my $_now_ms = _now_ms();
                        # Issue move every 10s until bot leaves portal area
                        if ($_now_ms - $_portal_exit_last_move_ms > 10000) {
                            $_portal_exit_last_move_ms = $_now_ms;
                            warning "[portal_exit] bot near portal exit (pos=$_px,$_py) - moving to center\n", 'aiSidecarBridge', 1;
                            eval { Commands::run("move 200 200"); 1; };
                        }
                    }
                }
            } else {
                # In town: ensure lockMap is set to hunting map
                # RULE.md: the bridge must NOT override the sidecar's routing decisions.
                # When the bot is in the academy room (iz_ac01_a), the sidecar's exit guard
                # is deliberately deferring the hunt until the bot exits the room — forcing
                # lockMap to prt_fild05 here would make the client loop an unroutable route.
                # Respect the sidecar's deferral by leaving lockMap alone in that case.
                if ($_sm_map eq 'iz_ac01_a' || $_sm_map eq 'iz_ac01') {
                    # Sidecar owns this decision (academy room exit / registration).
                } elsif (!defined $::config{lockMap} || $::config{lockMap} eq 'prontera' || $::config{lockMap} eq '') {
                    $::config{lockMap} = $_hunt_map;
                }
                # In town: disable random walk (prevents "move prontera" spam from OpenKore's AI)
                # BUT only if the heuristic hasn't explicitly set it (sidecar_set flag)
                if ($_sm_map eq 'prontera') {
                    if (!defined $::config{'_sidecar_set_route_randomWalk'} && defined $::config{route_randomWalk} && $::config{route_randomWalk} != 0) {
                        $::config{route_randomWalk} = 0;
                        warning "[town] disabling random walk in town to prevent 'move prontera' spam\n", 'aiSidecarBridge', 1;
                    }
                    # PORTAL ROUTE: every 30s, move directly to portal
                    state $_last_portal_move_ms = 0;
                    my $_now_ms = _now_ms();
                    if ($_now_ms - $_last_portal_move_ms > 30000) {
                        $_last_portal_move_ms = $_now_ms;
                        warning "[portal_route] in Prontera, moving to portal\n", 'aiSidecarBridge', 1;
                        eval { Commands::run("move 22 203"); 1; };
                    }
                }
            }
            # Ensure attack config on hunting maps
            if ($_sm_map =~ /^[a-z]+_fild/) {
                # attackAuto_inLockOnly + attackDistance controlled by heuristic
                # Per RULE.md: bridge must NOT override attack configs
            }
        }
        }
}

sub on_add_actor_list_probe {
	my ($hook, $actor, $actor_type) = @_;
	return if !_bridge_enabled();

	my $type = lc(_trim(_scalarize($actor_type), 16));
	return if $type !~ /^(?:monster|player|npc)$/;

	$actor_add_probe_count{$type} = 0 + ($actor_add_probe_count{$type} || 0) + 1;

	my $now_ms = _now_ms();
	my $last_ms = 0 + ($actor_add_probe_last_log_ms{$type} || 0);
	my $count = 0 + ($actor_add_probe_count{$type} || 0);
	my $throttle_ms = 5000;
	my $should_log = ($count == 1 || ($now_ms - $last_ms) >= $throttle_ms) ? 1 : 0;
	return if !$should_log;

	my ($hash_count, $list_count) = _actor_probe_counts_by_type($type);
	my $sample = _actor_probe_sample($actor);

	debug sprintf(
		"[aiSidecarBridge] actor add-hook probe type=%s add_count=%d containers={hash=%d,list=%d} sample=%s\n",
		$type,
		$count,
		0 + $hash_count,
		0 + $list_count,
		$sample,
	), 'aiSidecarBridge', 2;

	$actor_add_probe_last_log_ms{$type} = $now_ms;
}

sub _probe_actor_post_parse {
	my ($now_ms) = @_;
	return if !_bridge_enabled();
	$now_ms = _now_ms() if !defined $now_ms;

	my $throttle_ms = 5000;
	return if ($now_ms - $last_actor_post_parse_probe_log_ms) < $throttle_ms;

	my ($m_hash, $m_list) = _actor_probe_counts_by_type('monster');
	my ($p_hash, $p_list) = _actor_probe_counts_by_type('player');
	my ($n_hash, $n_list) = _actor_probe_counts_by_type('npc');

	my $m_sample = _actor_probe_sample_from_sources($monstersList, \%monsters);
	my $p_sample = _actor_probe_sample_from_sources($playersList, \%players);
	my $n_sample = _actor_probe_sample_from_sources($npcsList, \%npcs);

	debug sprintf(
		"[aiSidecarBridge] actor post-parse probe containers={monster:{hash=%d,list=%d,sample=%s} player:{hash=%d,list=%d,sample=%s} npc:{hash=%d,list=%d,sample=%s}} non_zero={monster=%d player=%d npc=%d}\n",
		0 + $m_hash,
		0 + $m_list,
		$m_sample,
		0 + $p_hash,
		0 + $p_list,
		$p_sample,
		0 + $n_hash,
		0 + $n_list,
		$n_sample,
		($m_hash > 0 || $m_list > 0) ? 1 : 0,
		($p_hash > 0 || $p_list > 0) ? 1 : 0,
		($n_hash > 0 || $n_list > 0) ? 1 : 0,
	), 'aiSidecarBridge', 2;

	$last_actor_post_parse_probe_log_ms = $now_ms;
}

sub _actor_probe_counts_by_type {
	my ($actor_type) = @_;
	my $type = lc(_trim(_scalarize($actor_type), 16));

	if ($type eq 'monster') {
		return (scalar(keys %monsters) + 0, scalar(_actor_list_items($monstersList)) + 0);
	}
	if ($type eq 'player') {
		return (scalar(keys %players) + 0, scalar(_actor_list_items($playersList)) + 0);
	}
	if ($type eq 'npc') {
		return (scalar(keys %npcs) + 0, scalar(_actor_list_items($npcsList)) + 0);
	}

	return (0, 0);
}

sub _actor_probe_sample_from_sources {
	my ($list_obj, $hash_ref) = @_;
	my $actor;

	my @items = _actor_list_items($list_obj);
	$actor = $items[0] if @items;

	if (!_is_hash_like($actor) && ref($hash_ref) eq 'HASH') {
		my @hash_items = values %{$hash_ref};
		$actor = $hash_items[0] if @hash_items;
	}

	return _actor_probe_sample($actor);
}

sub _actor_probe_sample {
	my ($actor) = @_;
	return 'none' if !_is_hash_like($actor);

	my $id = _actor_id_from_any($actor->{ID});
	$id = '?' if !defined $id || $id eq '';

	my $name = _trim(_scalarize($actor->{name}), 40);
	$name = '?' if !defined $name || $name eq '';

	return _trim("$id/$name", 96);
}

sub on_packet_hook {
	my ($hook, $args, $event_type) = @_;
	return if !_bridge_enabled();
	return if !_cfg_bool('aiSidecar_v2Enabled', 1);
	return if !_cfg_bool('aiSidecar_packetEventsEnabled', 1);

	my $normalized_type = _normalize_event_type($event_type || $hook || 'packet.unknown');

	my $payload = _extract_hook_payload($args);
	my $text = _trim("captured $normalized_type", _cfg_int('aiSidecar_maxEventTextChars', 220));

	_enqueue_normalized_event(
		'packet',
		$normalized_type,
		$hook,
		$text,
		$payload,
		{ map => _safe_field_map(), ai_seq_top => _safe_ai_seq_top() },
		{},
		'info',
	);
}

sub on_legacy_packet_hook {
	my ($hook, $args, $event_type) = @_;
	return if !_bridge_enabled();
	return if !_cfg_bool('aiSidecar_v2Enabled', 1);
	return if !_cfg_bool('aiSidecar_packetEventsEnabled', 1);

	my $normalized_type = _normalize_event_type($event_type || $hook || 'packet.legacy');
	my $payload = _extract_hook_payload($args);
	my $text = _trim("captured $normalized_type", _cfg_int('aiSidecar_maxEventTextChars', 220));

	_enqueue_normalized_event(
		'packet',
		$normalized_type,
		$hook,
		$text,
		$payload,
		{ map => _safe_field_map(), ai_seq_top => _safe_ai_seq_top() },
		{},
		'info',
	);
}

sub on_npc_menu {
	my ($hook, $args) = @_;
	return if !_bridge_enabled();
	my $responses = $args->{responses};
	return if !$responses || ref($responses) ne 'ARRAY';
	# Capture the ACTUAL NPC menu options so the conscious-tier LLM dialog
	# responder (_llm_npc_dialog_response) can read them and pick the response
	# agnostically (founder 2026-08-25: no hardcoded talknpc sequences).
	my @opts = map { defined($_) ? _trim(_scalarize($_), 200) : () } @$responses;
	# The last entry is "Cancel Chat" — keep it (it is a real option).
	$_npc_dialog_state{menu_options} = \@opts;
	$_npc_dialog_state{in_dialog} = 1;
	$_npc_dialog_state{last_interaction_ms} = _now_ms();
	# NPC identity: if we're talking to a known NPC, set npc_x/npc_y from the
	# talk target so the LLM can build the talknpc command.
	my $_nid = $args->{ID} || '';
	if ($_nid ne '') {
		my $aid = unpack('V', substr($_nid, 0, 4));
		my $npc = $main::npcs{$aid};
		if ($npc && ref($npc) eq 'HASH' && $npc->{pos}) {
			$_npc_dialog_state{npc_x} = $npc->{pos}{x} || 0;
			$_npc_dialog_state{npc_y} = $npc->{pos}{y} || 0;
			$_npc_dialog_state{npc_name} = $npc->{name} || '';
		}
	}
	debug "[npc_menu] captured " . scalar(@opts) . " options for " . ($_npc_dialog_state{npc_name} || 'unknown') . "\n", 'aiSidecarBridge', 2;
}

sub on_chat_message {
	my ($hook, $args, $channel) = @_;
	return if !_bridge_enabled();
	return if !_cfg_bool('aiSidecar_v2Enabled', 1);
	return if !_cfg_bool('aiSidecar_chatCaptureEnabled', 1);

	my $msg = _pick_first($args, qw(Msg msg message));
	return if !defined $msg || $msg eq '';

	my $sender = _pick_first($args, qw(MsgUser privMsgUser user name));
	my $target = $channel && $channel eq 'pm' ? ($char ? $char->{name} : undef) : undef;
	my $message_text = _trim(_scalarize($msg), _cfg_int('aiSidecar_maxChatChars', 500));

	my $chat_event = {
		channel => _trim($channel || 'unknown', 64),
		sender => defined($sender) && $sender ne '' ? _trim(_scalarize($sender), 128) : undef,
		target => defined($target) && $target ne '' ? _trim(_scalarize($target), 128) : undef,
		message => $message_text,
		map => _safe_field_map() || undef,
		kind => _trim($hook || '', 64),
		raw => _extract_hook_payload($args),
	};

	_enqueue_chat_event($chat_event);

	# ── NPC dialog failure detection from system messages ──
	if ($channel eq 'systemchat' && defined $message_text && $message_text ne '') {
		if ($message_text =~ /wrong\s+npc|npc.*fail|talking\s+to\s+wrong/i) {
			_enqueue_normalized_event(
				'npc',
				'npc.dialogue_failed',
				$hook,
				"NPC dialog failed: $message_text",
				{
					message => $message_text,
					map => _safe_field_map() || undef,
				},
				{},
				{},
				'warning',
			);
		}
	}

	_enqueue_normalized_event(
		'chat',
		_normalize_event_type('chat.' . ($channel || 'unknown')),
		$hook,
		_trim("chat message from " . ($chat_event->{sender} || 'unknown'), _cfg_int('aiSidecar_maxEventTextChars', 220)),
		{
			channel => $chat_event->{channel},
			sender => $chat_event->{sender},
			target => $chat_event->{target},
			message => $chat_event->{message},
			map => $chat_event->{map},
		},
		{ channel => $chat_event->{channel} },
		{},
		'info',
	);
}

sub on_post_config_modify {
	my ($hook, $args) = @_;
	return if !_bridge_enabled();
	return if !_cfg_bool('aiSidecar_v2Enabled', 1);
	return if !_cfg_bool('aiSidecar_configTrackEnabled', 1);
	return if ref($args) ne 'HASH';

	my $key = $args->{key};
	return if !defined $key || $key eq '';

	$pending_config_keys{$key} = 1;
	my $value = defined $config{$key} ? _trim(_scalarize($config{$key}), _cfg_int('aiSidecar_maxConfigValueChars', 220)) : '';

	_enqueue_normalized_event(
		'config',
		'config.key_changed',
		$hook,
		_trim("config key changed: $key", _cfg_int('aiSidecar_maxEventTextChars', 220)),
		{
			key => _trim($key, 128),
			bulk => 0,
			value => $value,
		},
		{ key => _trim($key, 64) },
		{},
		'info',
	);
}

sub on_post_bulk_config_modify {
	my ($hook, $args) = @_;
	return if !_bridge_enabled();
	return if !_cfg_bool('aiSidecar_v2Enabled', 1);
	return if !_cfg_bool('aiSidecar_configTrackEnabled', 1);
	return if ref($args) ne 'HASH';

	my $keys = $args->{keys};
	return if ref($keys) ne 'HASH';

	my @changed = sort keys %{$keys};
	foreach my $key (@changed) {
		$pending_config_keys{$key} = 1;
	}

	my @sample = @changed;
	splice @sample, 12 if @sample > 12;

	_enqueue_normalized_event(
		'config',
		'config.bulk_changed',
		$hook,
		_trim('bulk config keys changed', _cfg_int('aiSidecar_maxEventTextChars', 220)),
		{
			count => scalar(@changed) + 0,
			keys => \@sample,
		},
		{},
		{ changed_count => scalar(@changed) + 0 },
		'info',
	);
}

# Package-level shared state for potion interception
our @_heal_items;
our @_heal_skills;

# ── ACTOR::ITEM::USE OVERRIDE: block potion use when 0 potions on hunting map ──
# OpenKore's built-in useSelf_item system calls Actor::Item::use() directly,
# bypassing the Commands::run/pre hook. This override catches ALL potion use.
# When the bot has 0 potions on a hunting map, block potion use to prevent
# the "[use] item 'red potion' on cooldown, skipping" spam.
my $_orig_item_use = \&Actor::Item::use;
no warnings 'redefine';
*Actor::Item::use = sub {
	my $self = shift;
	my $item_name = $self->{name} || '';
	# Only intercept potion items
	if ($item_name =~ /potion|herb|fruit|berry/i) {
		my $_ic_map = '';
		if ($field) { $_ic_map = lc($field->name()); $_ic_map =~ s/\.gat$//; }
		elsif ($char) { $_ic_map = lc($char->{map} || ''); $_ic_map =~ s/\.gat$//; }
		# Only block on hunting maps (not in town)
		if ($_ic_map =~ /_fild|_dun/i) {
			# Check if we have any potions in inventory
			my $_ic_has_potions = 0;
			for my $item_name2 (@_heal_items) {
				$item_name2 = _trim($item_name2);
				next if !$item_name2;
				my $item = eval { Actor::Item::get($item_name2) };
				if ($item && $item->{amount} && $item->{amount} > 0) {
					$_ic_has_potions = 1;
					last;
				}
			}
			if (!$_ic_has_potions) {
				# No potions — block silently
				return;
			}
		}
	}
	# Call original with same arguments
	unshift @_, $self;
	goto &$_orig_item_use;
};

sub on_command_intercept {
	# Pre-command hook: intercept ALL commands including OpenKore internal AI
	# This is the LAST LINE OF DEFENSE against "move prontera" spam
	# Also blocks potion use when bot has 0 potions on hunting map
	# Also humanizes movement coordinates to avoid bot detection
	# Hook name: Commands::run/pre, params: {switch, args}
	#
	# ⚠ IMPORTANT: this hook LOGS ONLY — it CANNOT block execution. Vanilla
	# Commands.pm dispatches `$handler->($switch, $args)` with the ORIGINAL
	# switch/args captured before the hook (Commands.pm ~line 908), so
	# setting $args->{switch}='' here has no effect on what executes.
	# The party/ai-mode spam fix is STRUCTURAL: sidecar layers emit only
	# kind="log" observability intents for party + ai-mode, so no party
	# command ever reaches Commands::run. Do not rely on this hook to
	# block; fix the emitter instead.
	my (undef, $args) = @_;
	my $switch = $args->{switch} || '';
	my $cmd_args = $args->{args} || '';
	my $full_cmd = $switch . ' ' . $cmd_args;
	$full_cmd =~ s/\s+$//;

	# ── LOGGED-OUT GATE (core-AI path): block ALL commands when not in-game ──
	# Core AI / partyAuto / macros emit commands directly via Commands::run
	# (Plugins::callHook), bypassing _execute_action's hoisted gate. Every
	# Commands::run command is invalid while logged out (login/char-select use
	# the network layer, not Commands::run) — so block the command AND name
	# the emitter. Allowlist: reconnect/lifecycle commands that are valid
	# logged out. This is FLAW 6's structural fix extended to the core path.
	if (!$net || $net->getState() != Network::IN_GAME) {
		my @_ca = caller(1);
		my $_caller = @_ca ? ($_ca[3] || 'unknown') : 'unknown';
		my $_lc_switch = lc($switch || '');
		my %_logged_out_allow = map { $_ => 1 } (
			'relog', 'quit', 'logout', 'exit', 'getmapinfo',
			'getplayerinfo', 'charselect', 'servertype',
		);
		if (!$_logged_out_allow{$_lc_switch}) {
			warning "[cmd_pre_logged_out] BLOCKED cmd=$full_cmd caller=$_caller\n", 'aiSidecarBridge', 1;
			$args->{switch} = '';
			$args->{args} = '';
			return;
		}
		warning "[cmd_pre_logged_out] ALLOWED cmd=$full_cmd caller=$_caller\n", 'aiSidecarBridge', 1;
	}

	# ── SOLO PARTY-COMMAND SUPPRESSION (in-game core-AI noise) ──
	# Core OpenKore partyAuto=1 fires 'party request <name>' / 'party share exp'
	# every cycle even when the bot is SOLO, producing "You're not in a party"
	# log spam (Commands.pm:4355). The sidecar's fleet coordinator is the party
	# source of truth (it creates the party, gates on level/cooldown) — core
	# membership commands while solo are noise. Block membership-requiring
	# subcommands (share/request/leave/kick/leader/exp) when the bot has no
	# party users; allow create/join (party formation is the sidecar's job).
	if ($switch eq 'party') {
		# OpenKore puts the bot ITSELF in $char->{party}{users} even when solo,
		# so a users-count check never fires. The joined flag is the truth.
		my $_pm_joined = ($char && $char->{party}) ? ($char->{party}{joined} || 0) : 0;
		if (!$_pm_joined && $cmd_args =~ /^(share|request|leave|kick|leader|exp|shareitem|shareauto|sharediv)\b/i) {
			warning "[party_solo] BLOCKED cmd=$full_cmd (not joined)\n", 'aiSidecarBridge', 1;
			$args->{switch} = '';
			$args->{args} = '';
			return;
		}
	}

	# ── MOVE HUMANIZATION: intercept move x y and perturb coordinates ──
	if ($switch eq 'move' && $cmd_args =~ /^\s*(\d+)\s+(\d+)\s*$/) {
		my $tx = int($1);
		my $ty = int($2);
		my $cx = ($char && $char->{pos_to}) ? int($char->{pos_to}{x}) : 0;
		my $cy = ($char && $char->{pos_to}) ? int($char->{pos_to}{y}) : 0;
		# Rate limit: only humanize every 1s to avoid blocking main loop
		my $now_ms = int(Time::HiRes::time() * 1000);
		my $last_ms = $_last_move_humanize_ms || 0;
		my $dist = abs($tx - $cx) + abs($ty - $cy);
		if ($dist > 3 && $dist < 100 && ($now_ms - $last_ms) > 1000) {
			$_last_move_humanize_ms = $now_ms;
			my $resp = _http_post_json('/v1/humanize/move', {
				bot_id => _scalarize(_bot_id()),
				current_x => $cx,
				current_y => $cy,
				target_x => $tx,
				target_y => $ty,
			});
			# _http_post_json returns {status, error, json, raw}
			# The actual response fields are in ->{json}
			my $body = $resp ? $resp->{json} : undef;
			if ($body && $body->{humanized}) {
				my $hx = int($body->{humanized_x} + 0.5);
				my $hy = int($body->{humanized_y} + 0.5);
				if ($hx != $tx || $hy != $ty) {
					$args->{args} = "$hx $hy";
					debug "[aiSidecarBridge] move humanized: ($tx,$ty) -> ($hx,$hy) dev=$body->{deviation}\n", 'aiSidecarBridge', 3;
				}
			}
		}
	}

	# ── POTION USE INTERCEPTION: block when 0 potions on hunting map ──
	# OpenKore's built-in useSelf_item system fires independently of the bridge.
	# This hook catches ALL potion use commands, including from OpenKore's internal AI.
	# When the bot has 0 potions on a hunting map, block potion use to prevent spam.
	if ($full_cmd =~ /^use\s+(?:red\s+potion|orange\s+potion|white\s+potion|yellow\s+potion|blue\s+potion|green\s+potion)$/i) {
		my $_ic_map = '';
		if ($field) { $_ic_map = lc($field->name()); $_ic_map =~ s/\.gat$//; }
		elsif ($char) { $_ic_map = lc($char->{map} || ''); $_ic_map =~ s/\.gat$//; }
		# Only block on hunting maps (not in town)
		if ($_ic_map =~ /_fild|_dun/i) {
			# Check if we have any potions in inventory
			my $_ic_has_potions = 0;
			for my $item_name (@_heal_items) {
				$item_name = _trim($item_name);
				next if !$item_name;
				my $item = eval { Actor::Item::get($item_name) };
				if ($item && $item->{amount} && $item->{amount} > 0) {
					$_ic_has_potions = 1;
					last;
				}
			}
			if (!$_ic_has_potions) {
				# No potions — block the command silently
				$args->{switch} = '';
				$args->{args} = '';
				return;
			}
		}
	}

	# ── MOVE PRONTERA INTERCEPTION: existing logic ──
	return if $full_cmd !~ /^move\s+prontera$/i;
	# Get current map from field or character
	my $_ic_map = '';
	if ($field) { $_ic_map = lc($field->name()); $_ic_map =~ s/\.gat$//; }
	elsif ($char) { $_ic_map = lc($char->{map} || ''); $_ic_map =~ s/\.gat$//; }
	return if !$_ic_map;
	if ($_ic_map eq 'prontera') {
	    # In Prontera: redirect "move prontera" to portal coordinates
	    my $_portal_x = _cfg('aiSidecar_portalX', '22') || '22';
	    my $_portal_y = _cfg('aiSidecar_portalY', '203') || '203';
	    my $_lm = $::config{lockMap} || '';
	    if ($_lm =~ /^[a-z]+_fild/ || $_lm =~ /_field/) {
	        warning "[command_intercept] '$full_cmd' in Prontera (lockMap=$_lm) -> portal\n", 'aiSidecarBridge', 1;
	        # Override args to redirect command
	        $args->{switch} = 'move';
	        $args->{args} = "$_portal_x $_portal_y";
	    }
	} elsif ($_ic_map =~ /^[a-z]+_fild/ || $_ic_map =~ /_field/) {
	    # On hunting map: block "move prontera" unless bot has 0 potions
	    # (0 potions = need to return to town to buy)
	    my $_ic_has_potions = 0;
	    if ($char && @{_char_inventory($char)}) {
	        for my $_gi (@{_char_inventory($char)}) {
	            next unless $_gi;
	            my $_gi_name = $_gi->{name} || '';
	            if ($_gi_name =~ /potion|herb|fruit|berry|red|orange|white|yellow|blue|green/i) {
	                $_ic_has_potions = 1;
	                last;
	            }
	        }
	    }
	    if ($_ic_has_potions) {
	        warning "[command_intercept] blocking '$full_cmd' on hunting map $_ic_map\n", 'aiSidecarBridge', 1;
	        $args->{switch} = '';
	        $args->{args} = '';
	    } else {
	        warning "[command_intercept] allowing '$full_cmd' on hunting map $_ic_map (0 potions)\n", 'aiSidecarBridge', 1;
	    }
	} elsif ($_ic_map eq 'prontera' && $full_cmd =~ /^move\s+(?:prontera|22\s+203)$/i) {
	    # In Prontera: if bot has 0 potions, DO NOT redirect to portal
	    # Bot needs to buy potions first. Only redirect if bot has potions.
	    my $_ic_has_potions = 0;
	    if ($char && @{_char_inventory($char)}) {
	        for my $_gi (@{_char_inventory($char)}) {
	            next unless $_gi;
	            my $_gi_name = $_gi->{name} || '';
	            if ($_gi_name =~ /potion|herb|fruit|berry|red|orange|white|yellow|blue|green/i) {
	                $_ic_has_potions = 1;
	                last;
	            }
	        }
	    }
	    if (!$_ic_has_potions) {
	        warning "[command_intercept] blocking 'move prontera/22 203' in town (0 potions, need to buy first)\n", 'aiSidecarBridge', 1;
	        $args->{switch} = '';
	        $args->{args} = '';
	    } else {
	        # Have potions — redirect to portal
	        my $_portal_x = _cfg('aiSidecar_portalX', '22') || '22';
	        my $_portal_y = _cfg('aiSidecar_portalY', '203') || '203';
	        my $_lm = $::config{lockMap} || '';
	        if ($_lm =~ /^[a-z]+_fild/ || $_lm =~ /_field/) {
	            warning "[command_intercept] 'move prontera' in town (has potions, lockMap=$_lm) -> portal\n", 'aiSidecarBridge', 1;
	            $args->{switch} = 'move';
	            $args->{args} = "$_portal_x $_portal_y";
	        }
	    }
	}
		# ── ATTACK BLOCK: intercept attack commands for ignored monsters ──
	# OpenKore's internal mon_control doesn't reliably prevent attacks for
	# monsters with ignore=-1 when attackAuto=3 is active. This bridge-level
	# intercept catches ALL attack commands and blocks them for monsters
	# that the heuristic has marked as ignore via mon_control entries.
	if ($full_cmd =~ /^attack\s+(.+)$/i) {
	    my $_atk_id = $1;
	    # Try direct monster lookup by ID
	    my $_atk_monster = $monsters{$_atk_id};
	    my $_atk_name = '';
	    if ($_atk_monster) {
	        $_atk_name = lc($_atk_monster->{name} || '');
	    } else {
	        # Fallback: search all monsters by nameID or position
	        for my $_atk_mid (keys %monsters) {
	            my $_atk_m = $monsters{$_atk_mid};
	            if ($_atk_m && ($_atk_m->{nameID} eq $_atk_id || $_atk_m->{binID} eq $_atk_id)) {
	                $_atk_monster = $_atk_m;
	                $_atk_name = lc($_atk_m->{name} || '');
	                last;
	            }
	        }
	    }
	    if ($_atk_name) {
	        my $_atk_control = main::mon_control($_atk_name, ($_atk_monster->{nameID} || ''));
	        if ($_atk_control && defined $_atk_control->{attack_auto} && $_atk_control->{attack_auto} <= 0) {
	            debug "[attack_block] blocking attack on $_atk_name (attack_auto=$_atk_control->{attack_auto})\n", 'aiSidecarBridge', 1;
	            $args->{switch} = '';
	            $args->{args} = '';
	            return;
	        }
	    }
	}

	# Handle mon_control command - write to mon_control.txt and reload
	if ($full_cmd =~ /^mon_control\s+(.+)$/i) {
	    my $_mc_entry = $1;
	    _append_mon_control_dedup($_mc_entry);
	    eval { Commands::run("reload mon_control"); 1; };
	    $args->{switch} = '';
	    $args->{args} = '';
	    return;
	}
# PvP/GvG/Turbo maps: allow through
	return;
}

sub on_command_run_post {
	my ($hook, $args) = @_;
	return if !_bridge_enabled();
	return if !_cfg_bool('aiSidecar_v2Enabled', 1);
	return if !_cfg_bool('aiSidecar_macroTraceEnabled', 1);
	return if ref($args) ne 'HASH';

	my $switch = lc(_scalarize($args->{switch}));
	return if $switch eq '';

	my $arg_text = _scalarize($args->{args});
	my $input = $switch;
	$input .= ' ' . $arg_text if defined $arg_text && $arg_text ne '';

	my $is_macro_cmd = 0;
	$is_macro_cmd = 1 if $switch eq 'macro' || $switch eq 'eventmacro';
	$is_macro_cmd = 1 if $input =~ /^\s*plugin\s+reload\s+(?:macro|eventmacro)\b/i;
	$is_macro_cmd = 1 if $input =~ /^\s*conf\s+(?:macro_file|eventmacro_file)\b/i;

	my $trace_all = _cfg_bool('aiSidecar_traceAllCommands', 0);
	return if !$is_macro_cmd && !$trace_all;

	my $family = $is_macro_cmd ? 'macro' : 'action';
	my $event_type = $is_macro_cmd ? 'macro.command' : 'action.command';

	_enqueue_normalized_event(
		$family,
		$event_type,
		$hook,
		_trim("command executed: $input", _cfg_int('aiSidecar_maxEventTextChars', 220)),
		{
			switch => _trim($switch, 64),
			args => _trim($arg_text, 256),
			input => _trim($input, 320),
			is_macro => $is_macro_cmd ? 1 : 0,
		},
		{ switch => _trim($switch, 64), macro => $is_macro_cmd ? '1' : '0' },
		{},
		$is_macro_cmd ? 'info' : 'debug',
	);
}

sub _track_ai_sequence_transition {
	return if !_bridge_enabled();
	return if !_cfg_bool('aiSidecar_v2Enabled', 1);
	return if !_cfg_bool('aiSidecar_macroTraceEnabled', 1);

	my $current = _safe_ai_seq_top();
	return if $current eq $last_ai_seq_top;

	my $previous = $last_ai_seq_top;
	$last_ai_seq_top = $current;

	my $was_macro = $previous =~ /macro/i ? 1 : 0;
	my $is_macro = $current =~ /macro/i ? 1 : 0;
	return if !$was_macro && !$is_macro;

	_enqueue_normalized_event(
		'macro',
		'macro.ai_sequence_transition',
		'mainLoop_pre',
		_trim("AI sequence transition: '$previous' -> '$current'", _cfg_int('aiSidecar_maxEventTextChars', 220)),
		{ from => $previous, to => $current },
		{ from => _trim($previous, 64), to => _trim($current, 64) },
		{ entered_macro => $is_macro ? 1 : 0 },
		'info',
	);
}

sub _track_lifecycle_transitions {
	return if !_bridge_enabled();
	return if !_cfg_bool('aiSidecar_v2Enabled', 1);

	my $in_game = ($net && $net->getState() == Network::IN_GAME) ? 1 : 0;
	my $now_ms = _now_ms();

	if (!defined $last_net_in_game) {
		$last_net_in_game = $in_game;
	} elsif ($last_net_in_game != $in_game) {
		if ($in_game) {
			my $age_s = 0.0;
			if ($last_disconnect_at_ms > 0) {
				$age_s = ($now_ms - $last_disconnect_at_ms) / 1000.0;
				$age_s = 0.0 if $age_s < 0.0;
			}
			_enqueue_normalized_event(
				'lifecycle',
				'lifecycle.reconnected',
				'mainLoop_pre',
				'reconnected to game state',
				{ reconnect_age_s => $age_s + 0.0 },
				{ state => 'in_game' },
				{ reconnect_age_s => $age_s + 0.0 },
				'info',
			);
			_send_party_status('reconnect');
		} else {
			$last_disconnect_at_ms = $now_ms;
			_enqueue_normalized_event(
				'lifecycle',
				'lifecycle.disconnected',
				'mainLoop_pre',
				'disconnected from game state',
				{ net_state => 'disconnected' },
				{ state => 'disconnected' },
				{},
				'warning',
			);
		}
		$last_net_in_game = $in_game;
	}

	my $hp = $char ? $char->{hp} : undef;
	if (defined $hp && $hp =~ /^\d+$/) {
		if (defined $last_hp) {
			if ($last_hp > 0 && $hp <= 0) {
				$death_count += 1;
				$respawn_state = 'dead';
				_enqueue_normalized_event(
					'lifecycle',
					'lifecycle.death',
					'mainLoop_pre',
					'character died',
					{ hp => 0 + $hp, death_count => 0 + $death_count, respawn_state => $respawn_state },
					{},
					{ death_count => 0 + $death_count },
					'warning',
				);
				_send_party_status('death');
			} elsif ($last_hp <= 0 && $hp > 0) {
				$respawn_state = 'respawned';
				_enqueue_normalized_event(
					'lifecycle',
					'lifecycle.respawn',
					'mainLoop_pre',
					'character respawned',
					{ hp => 0 + $hp, death_count => 0 + $death_count, respawn_state => $respawn_state },
					{},
					{ death_count => 0 + $death_count },
					'info',
				);
				_send_party_status('respawn');
			}
		}
		$last_hp = 0 + $hp;
	}

	my $map = _safe_field_map();
	if (defined $map && $map ne '' && defined $last_map_name && $last_map_name ne '' && $map ne $last_map_name) {
		_enqueue_normalized_event(
			'lifecycle',
			'lifecycle.map_transfer',
			'mainLoop_pre',
			"map transfer: $last_map_name -> $map",
			{ from_map => $last_map_name, to_map => $map },
			{ from_map => _trim($last_map_name, 64), to_map => _trim($map, 64) },
			{},
			'info',
		);
		_send_party_status('map_change');
	}
	$last_map_name = $map if defined $map && $map ne '';

	my $ai_top = _safe_ai_seq_top();
	my $x = undef;
	my $y = undef;
	if ($char) {
		my $pos = eval {
			# NOTE: never use bare 'return' inside this eval BLOCK —
			# return in eval returns from the enclosing sub, not the eval.
			my $p;
			if ($char->{pos_to} && ref $char->{pos_to} eq 'HASH') {
				$p = $char->{pos_to};
			} elsif ($char->{pos} && ref $char->{pos} eq 'HASH') {
				$p = $char->{pos};
			}
			$p;
		};
		if ($pos) {
			$x = $pos->{x};
			$y = $pos->{y};
		}
	}
	my $route_signature = join(':', ($map || ''), (defined $x ? $x : ''), (defined $y ? $y : ''), ($ai_top || ''));
	if ($ai_top =~ /^(?:route|move)/i && defined $last_route_signature && $last_route_signature eq $route_signature) {
		$route_churn_count += 1;
		my $threshold = _cfg_int('aiSidecar_routeChurnThreshold', 8);
		$threshold = 1 if $threshold < 1;
		my $emit_every = _cfg_int('aiSidecar_routeFailureEvery', 16);
		$emit_every = $threshold if $emit_every < $threshold;

		if ($route_churn_count % $threshold == 0) {
			_enqueue_normalized_event(
				'lifecycle',
				'lifecycle.route_churn',
				'mainLoop_pre',
				'route churn without position gain detected',
				{
					map => $map,
					x => $x,
					y => $y,
					route_churn_count => 0 + $route_churn_count,
				},
				{ map => _trim($map || '', 64) },
				{ route_churn_count => 0 + $route_churn_count },
				'warning',
			);
		}

		if ($route_churn_count % $emit_every == 0) {
			$route_failure_count += 1;
			_enqueue_normalized_event(
				'lifecycle',
				'lifecycle.route_failure',
				'mainLoop_pre',
				'route failure inferred from repeated churn',
				{
					map => $map,
					x => $x,
					y => $y,
					route_failure_count => 0 + $route_failure_count,
					route_churn_count => 0 + $route_churn_count,
				},
				{ map => _trim($map || '', 64) },
				{ route_failure_count => 0 + $route_failure_count, route_churn_count => 0 + $route_churn_count },
				'warning',
			);
		}
	} else {
		$route_churn_count = 0 if $ai_top !~ /^(?:route|move)/i;
	}
	$last_route_signature = $route_signature;
}

sub _load_bridge_config {
	my ($file, $target) = @_;
	%{$target} = ();
	parseConfigFile($file, $target, 0);

	_load_bridge_config_overrides();

	my %defaults = (
		aiSidecar_enable => 1,
		aiSidecar_baseUrl => 'http://127.0.0.1:18081',
		aiSidecar_contractVersion => 'v1',
		aiSidecar_source => 'openkore-bridge',
		aiSidecar_connectTimeoutMs => 2000,
		aiSidecar_ioTimeoutMs => 30000,
		aiSidecar_snapshotEnabled => 1,
		aiSidecar_snapshotIntervalMs => 1000,
				aiSidecar_pollWhenDisconnected => 1,
		aiSidecar_actionPollEnabled => 1,
		aiSidecar_pollIntervalMs => 500,
		aiSidecar_pollFailureBackoffBaseMs => 600,
		aiSidecar_pollFailureBackoffMaxMs => 6000,
		aiSidecar_pollFailureResetRegistrationAfter => 3,
		aiSidecar_ackEnabled => 1,
		# ── Server-agnostic NPC & map config (override per bot profile in ai_sidecar.txt) ──
		aiSidecar_recoveryCity => '',           # Default city to retreat to (auto-detected from current map)
		aiSidecar_fallbackItem => _cfg('aiSidecar_fallbackHealItem', 'White Potion'), # Fallback healing item when none configured
		# ── Bot behavior overrides (overrides OpenKore config.txt) ──
		aiSidecar_attackAuto => '2',
		aiSidecar_attackAutoInLockOnly => '1',
		aiSidecar_attackAutoFollowTarget => '0',
		aiSidecar_attackAutoOnlyWhenSafe => '0',
		aiSidecar_attackAutoNoMove => '0',
		aiSidecar_sitAutoHpLower => '0',
		aiSidecar_sitAutoHpUpper => '0',
		aiSidecar_sitAutoMaxDmg => '99999',
		# ── Reflex cooldowns (milliseconds per reflex type) ──
		aiSidecar_reflexNoHealCooldownMs => 10000,
		aiSidecar_reflexFleeCooldownMs => 1000,
		aiSidecar_reflexTeleportCooldownMs => 3000,
		aiSidecar_reflexAggroWarningCooldownMs => 5000,
		aiSidecar_reflexLowSpCooldownMs => 10000,
		aiSidecar_reflexGmDetectedCooldownMs => 60000,
		aiSidecar_reflexWeightWarningCooldownMs => 30000,
		aiSidecar_reflexEquipBrokenCooldownMs => 60000,
		aiSidecar_reflexInterruptCastCooldownMs => 1500,
		aiSidecar_reflexPrePotCooldownMs => 5000,
		aiSidecar_reflexBotRequestCooldownMs => 5000,
		aiSidecar_reflexPartyLowHpCooldownMs => 10000,
		aiSidecar_reflexHighAggroCooldownMs => 3000,
		aiSidecar_reflexZonkCooldownMs => 2000,
		aiSidecar_reflexDeathSpikeCooldownMs => 120000,
		aiSidecar_reflexPreBuffCooldownMs => 15000,
		aiSidecar_reflexPreDodgeCooldownMs => 2000,
		aiSidecar_reflexAutoSitCooldownMs => 5000,
		aiSidecar_reflexTopOffCooldownMs => 10000,
		aiSidecar_reflexEmergencyMoveCooldownMs => 60000,
		aiSidecar_reflexEscapeCooldownMs => 5000,
		aiSidecar_reflexEscapeSpikeCooldownMs => 30000,
		aiSidecar_huntingMap => '',              # Default hunting map (auto-detected from knowledge DB)
		aiSidecar_sellNpc => '',                 # Sell NPC "map x y" or empty for auto-detect
		aiSidecar_storageNpc => '',              # Storage NPC "map x y" or empty for auto-detect
		aiSidecar_healNpc => '',                 # Heal/buy NPC "map x y" or empty for auto-detect

		aiSidecar_ackRetryMs => 400,
		aiSidecar_ackMaxAgeMs => 120000,
		aiSidecar_registerRetryMs => 3000,
		aiSidecar_telemetryEnabled => 1,
		aiSidecar_configReloadEnabled => 1,
		aiSidecar_telemetryIntervalMs => 1000,
		aiSidecar_maxRawChars => 256,
		aiSidecar_maxCommandLength => 160,
		aiSidecar_botIdentity => '',
		aiSidecar_botIdOverride => '',
		aiSidecar_macroReloadEnabled => 1,
		aiSidecar_macroFile => 'ai_sidecar_generated_macros.txt',
		aiSidecar_eventMacroFile => 'ai_sidecar_generated_eventmacros.txt',
		aiSidecar_macroPluginName => 'macro',
		aiSidecar_eventMacroPluginName => 'eventMacro',
		aiSidecar_v2Enabled => 1,
		aiSidecar_packetEventsEnabled => 1,
		aiSidecar_chatCaptureEnabled => 1,
		aiSidecar_configTrackEnabled => 1,
		aiSidecar_macroTraceEnabled => 1,
		aiSidecar_eventIngestEnabled => 1,
		aiSidecar_chatIngestEnabled => 1,
		aiSidecar_configIngestEnabled => 1,
		aiSidecar_eventIngestIntervalMs => 700,
		aiSidecar_chatIngestIntervalMs => 900,
		aiSidecar_configIngestIntervalMs => 2000,
		aiSidecar_eventIngestFailureBackoffBaseMs => 1000,
		aiSidecar_eventIngestFailureBackoffMaxMs => 10000,
		aiSidecar_eventBatchSize => 20,
		aiSidecar_chatBatchSize => 20,
		aiSidecar_maxEventQueue => 500,
		aiSidecar_maxChatQueue => 200,
		aiSidecar_maxEventPayloadFields => 16,
		aiSidecar_maxEventTextChars => 220,
		aiSidecar_maxChatChars => 500,
		aiSidecar_maxConfigValueChars => 220,
		aiSidecar_maxConfigKeysPerPush => 64,
		aiSidecar_traceAllCommands => 0,
		aiSidecar_routeChurnThreshold => 8,
		aiSidecar_routeFailureEvery => 16,
		aiSidecar_verbose => 1,

		# ── Upgraded IPC modules ──
		aiSidecar_stateBuildersEnabled => 1,
		aiSidecar_maxItems => 200,
		aiSidecar_zmqAddress => 'tcp://127.0.0.1:5559',
	);

	foreach my $key (keys %defaults) {
		$target->{$key} = $defaults{$key} if !defined $target->{$key} || $target->{$key} eq '';
	}

	debug "[aiSidecarBridge] loaded control file $file\n", 'aiSidecarBridge', 2;
}

sub _load_bridge_config_overrides {
	my $path = eval { Settings::getControlFilename('ai_sidecar.txt') };
	return if !defined $path || $path eq '';
	return if !-e $path;

	my %fresh;
	parseConfigFile($path, \%fresh, 0);
	foreach my $key (keys %fresh) {
		next if !defined $key || $key eq '';
		$bridge_cfg{$key} = $fresh{$key} if defined $fresh{$key};
	}
	debug "[aiSidecarBridge] refreshed control file $path\n", 'aiSidecarBridge', 2;
}

sub _load_bridge_policy {
	my ($file, $target) = @_;
	%{$target} = ();
	parseConfigFile($file, $target, 0);

	my %defaults = (
		aiSidecarPolicy_mode => 'allowlist',
		aiSidecarPolicy_allow_0 => 'ai',
		aiSidecarPolicy_allow_1 => 'move',
		aiSidecarPolicy_allow_2 => 'macro',
		aiSidecarPolicy_allow_3 => 'eventMacro',
		aiSidecarPolicy_allow_4 => 'talknpc',
		aiSidecarPolicy_allow_5 => 'respawn',
		aiSidecarPolicy_allow_6 => 'attack',
		aiSidecarPolicy_allow_7 => 'sit',
		aiSidecarPolicy_allow_8 => 'stand',
		aiSidecarPolicy_allow_9 => 'take',
		aiSidecarPolicy_allow_10 => 'party',
		aiSidecarPolicy_allow_11 => 'guild',
		aiSidecarPolicy_allow_12 => 'skill',
		aiSidecarPolicy_allow_13 => 'use_skill',
		aiSidecarPolicy_allow_14 => 'send',
		aiSidecarPolicy_allow_15 => 'shop',
		aiSidecarPolicy_allow_16 => 'storage',
		aiSidecarPolicy_allow_17 => 'buy',
		aiSidecarPolicy_allow_18 => 'sell',
		aiSidecarPolicy_allow_19 => 'craft',
		aiSidecarPolicy_allow_20 => 'identify',
		aiSidecarPolicy_allow_21 => 'follow',
		aiSidecarPolicy_allow_22 => 'teleport',
		aiSidecarPolicy_allow_23 => 'set',
		aiSidecarPolicy_allow_24 => 'warp',
		aiSidecarPolicy_allow_25 => 'portals',
		aiSidecarPolicy_allow_26 => 'route',
		aiSidecarPolicy_allow_27 => 'stats',
		aiSidecarPolicy_allow_28 => 'stat',
		aiSidecarPolicy_allow_29 => 'skills_add',
		aiSidecarPolicy_allow_30 => 'skills',
		aiSidecarPolicy_allow_31 => 'stat_add',
		aiSidecarPolicy_allow_32 => 'stat_add',  # Note: stats_add is not a real command
		aiSidecarPolicy_allow_33 => 'deal',
		aiSidecarPolicy_allow_34 => 'trade',
		aiSidecarPolicy_allow_35 => 'friend',
		aiSidecarPolicy_allow_36 => 'storage',
		aiSidecarPolicy_allow_37 => 'cart',
		aiSidecarPolicy_allow_38 => 'mail',
		aiSidecarPolicy_allow_39 => 'storageadd',
		aiSidecarPolicy_allow_40 => 'storageget',
		aiSidecarPolicy_allow_41 => 'use',
		aiSidecarPolicy_allow_42 => 'equip',
		aiSidecarPolicy_allow_43 => 'unequip',
		aiSidecarPolicy_allow_44 => 'items',
		aiSidecarPolicy_allow_45 => 'inventory',
		aiSidecarPolicy_allow_46 => 'look',
		aiSidecarPolicy_allow_47 => 'talk',
		aiSidecarPolicy_allow_48 => 'talknpc',
		aiSidecarPolicy_allow_49 => 'response',
		aiSidecarPolicy_allow_50 => 'c',
		aiSidecarPolicy_allow_51 => 'pm',
		aiSidecarPolicy_allow_52 => 'join',
		aiSidecarPolicy_allow_53 => 'leave',
		aiSidecarPolicy_allow_54 => 'exp',
		aiSidecarPolicy_allow_55 => 'storageprice',
		aiSidecarPolicy_allow_56 => 'reparse',
		aiSidecarPolicy_allow_57 => 'ss',
		aiSidecarPolicy_allow_58 => '@go',

		aiSidecarPolicy_deny_0 => 'quit',
		aiSidecarPolicy_deny_1 => 'plugin',
		aiSidecarPolicy_deny_2 => 'reload',
		aiSidecarPolicy_deny_3 => 'eval',
		aiSidecarPolicy_deny_4 => 'conf',
	);

	foreach my $key (keys %defaults) {
		$target->{$key} = $defaults{$key} if !defined $target->{$key} || $target->{$key} eq '';
	}

	_rebuild_policy_lists();
	debug "[aiSidecarBridge] loaded policy file $file\n", 'aiSidecarBridge', 2;
}

sub _rebuild_policy_lists {
	@policy_allow = ();
	@policy_deny = ();
	foreach my $key (sort keys %bridge_policy) {
		if ($key =~ /^aiSidecarPolicy_allow_\d+$/i && defined $bridge_policy{$key}) {
			push @policy_allow, lc($bridge_policy{$key});
		} elsif ($key =~ /^aiSidecarPolicy_deny_\d+$/i && defined $bridge_policy{$key}) {
			push @policy_deny, lc($bridge_policy{$key});
		}
	}
}

sub _bridge_enabled {
	return 0 if !$json_available;
	return _cfg_bool('aiSidecar_enable', 1);
}

sub _attempt_register {
	my ($reason) = @_;
	return if !_bridge_enabled();

	my $payload = {
		meta => _meta(_bot_id()),
		bot_name => $char ? $char->{name} : undef,
		capabilities => [
			'bridge_snapshot_push',
			'bridge_action_poll',
			'bridge_action_ack',
			'bridge_telemetry_push',
			'bridge_macro_reload_orchestration',
			'bridge_config_reload_orchestration',
			'bridge_v2_event_ingest',
			'bridge_v2_chat_ingest',
			'bridge_v2_config_ingest',
			'bridge_packet_hook_capture',
			'bridge_config_change_capture',
			'bridge_macro_execution_trace',
		],
		attributes => {
			reason => $reason,
			master => ($config{master} || ''),
			identity_username => ($config{username} || ''),
			identity_char_name => ($char && $char->{name} ? $char->{name} : ($config{username} || '')),
			identity_override => _cfg('aiSidecar_botIdentity', ''),
			profile => eval { $profiles::profile } || '',
			control_folder => _active_control_folder(),
		},
	};

	# Profile->char mapping is hardcoded in the party request handler
	my $resp = _http_post_json('/v1/ingest/register', $payload);
	if ($resp && $resp->{status} >= 200 && $resp->{status} < 500) {
		$registered = 1;
		_load_profile_to_char();
		debug "[aiSidecarBridge] sidecar registration succeeded\n", 'aiSidecarBridge', 2;
		return;
	}

	$registered = 0;
	_throttled_warning('register_failed', '[aiSidecarBridge] sidecar registration failed, running fail-open.');
	_emit_telemetry('warning', 'bridge', 'register_failed', 'sidecar registration failed');
}

# ── Inventory accessor ─────────────────────────────────────────────────────
# OpenKore stores the character inventory in an InventoryList tied-array at
# $char->inventory() (method), NOT at the hash key $char->{inventory}. All the
# bridge's old reads used the dead hash key, so every snapshot reported an
# EMPTY inventory (item_count=0, inventory_items=[], has_weapon=0) even when
# the character owned items — which made the cold-start academy logic believe
# every bot was "weapon-less" forever. This helper returns the real list.
sub _char_inventory {
    my ($char) = @_;
    return [] if !$char;
    my $inv = eval { $char->inventory() } || undef;
    return [] if !$inv;
    # Use the DOCUMENTED ObjectList iteration API. This fork's `@{}` overload
    # is unreliable (deref failed → raw=err), so prefer getItems() which returns
    # the real Actor::Item array (OL_cItems). Fall back to get($binID) when the
    # deref yields binID integers (older fork semantics).
    my @items;
    my $_gi = eval { $inv->getItems() } || undef;
    if (ref($_gi) eq 'ARRAY' && @{$_gi}) {
        for my $el (@{$_gi}) {
            push @items, $el if ref($el) && eval { $el->isa('Actor::Item') };
        }
    }
    if (!@items) {
        for my $el (eval { @{$inv} } || ()) {
            if (ref($el) && eval { $el->isa('Actor::Item') }) {
                push @items, $el;
            } else {
                my $obj = eval { $inv->get($el) } || undef;
                push @items, $obj if $obj;
            }
        }
    }
    return \@items;
}

# ── Best available healing item (survivability fallback) ──
# When the sidecar/reflex emits "use red_potion" but the bot only owns a
# different heal (e.g. Novice Potion 569 x300 academy kit), pick the strongest
# potion actually in inventory so the bot heals instead of dying. Returns the
# item NAME (string) or '' if no heal item is owned.
sub _best_available_heal_name {
    my ($char) = @_;
    my @inv = @{_char_inventory($char)};
    my @owned_heal;
    for my $_item (@inv) {
        next unless ref($_item);
        my $_n = $_item->{name} || '';
        next unless $_n =~ /potion|herb|fruit|berry|red|orange|white|yellow|blue|green|grape|milk|juice/i;
        push @owned_heal, [ $_item->{amount} || 1, $_n ];
    }
    return '' unless @owned_heal;
    # Prefer the highest heal-potency item present; among equal potency pick
    # by quantity then name. Heal potency order (approx, by max heal): white>
    # orange>yellow>blue>green>red>grape>novice. We sort by a rough potency rank
    # then quantity, so "300x Novice" still yields to a single "Red" if owned.
    my %rank = (
        'white potion' => 9, 'orange potion' => 8, 'yellow potion' => 7,
        'blue potion' => 6, 'green potion' => 5, 'red potion' => 4,
        'grape' => 3, 'novice potion' => 2, 'apples' => 1, 'banana' => 1,
    );
    @owned_heal = sort {
        my $ra = $rank{lc $a->[1]} || 0; my $rb = $rank{lc $b->[1]} || 0;
        ($rb <=> $ra) || ($b->[0] <=> $a->[0]);
    } @owned_heal;
    return $owned_heal[0][1];
}

# ── Inventory slot for an owned item (for 'equip <slot> <name>') ──
# OpenKore cmdEquip accepts 'equip <slot> <item-name>'. Returns the equip slot
# (e.g. 'weapon', 'armor', 'head_top') or undef if the item has no equip slot.
sub _inventory_slot_for_item {
    my ($item) = @_;
    return undef unless ref($item);
    my $_slot = $item->{slot} || $item->{equipSlot} || '';
    return $_slot if $_slot;
    # Fall back: derive from item type (Actor::Item type_equip / type)
    my $_type = $item->{type} || 0;
    return 'weapon' if ($_type == 4 || ($item->{type_equip} && $item->{type_equip} == 4));
    return 'armor'  if ($_type == 5 || ($item->{type_equip} && $item->{type_equip} == 5));
    return undef;  # no slot mapping — caller falls back to 'equip <name>'
}

sub _send_snapshot {
    return if !_bridge_enabled();
    return if !$registered;  # Don't send snapshots until registered
    if (!$net || $net->getState() != Network::IN_GAME) {
        return if !_cfg_bool('aiSidecar_pollWhenDisconnected', 1);
    }

    my $snapshot;
    eval { $snapshot = _build_snapshot_payload(); 1; } or do {
        my $err = $@ || 'snapshot_build_failed';
        _throttled_warning('snapshot_build_failed', "[aiSidecarBridge] _build_snapshot_payload error: $err");
        return;
    };
    # Send via the bridge's own _http_post_json — the same proven path
    # used by telemetry (works reliably inside OpenKore's main loop).
    # NOTE: NOT via $_http_client->send_state() — the HTTPClient alarm()
    # + SIGALRM pattern conflicts with OpenKore's own alarm timers in the
    # live loop and silently returns 0 (verified: 0 snapshot POSTs reach
    # the sidecar via that path, while telemetry via _http_post_json
    # delivers 100%).
    my $resp = _http_post_json('/v1/ingest/snapshot', $snapshot);
    if (!$resp || $resp->{status} < 200 || $resp->{status} >= 300) {
        _throttled_warning('snapshot_failed', '[aiSidecarBridge] snapshot push failed, fail-open retained.');
        _emit_telemetry('warning', 'bridge', 'snapshot_failed', 'snapshot push failed');
    }

    # ── Send 17 specialized state builder snapshots (disabled — sidecar uses ingest/snapshot) ──
    if (0 && $_state_builders && _cfg_bool('aiSidecar_stateBuildersEnabled', 1)) {
        my $states = $_state_builders->build_all_states();
        if ($_http_client) {
            $_http_client->send_json('/v1/state/builders', $states);
        } else {
            _http_post_json('/v1/state/builders', $states);
        }
    }

    if (_cfg_bool('aiSidecar_v2Enabled', 1) && _cfg_bool('aiSidecar_actorsEnabled', 1)) {
        _send_actor_delta_from_snapshot($snapshot);
    }

    # ── charstatus.json real-time state file (2026-08-27) ──
    # Write the complete, enriched char+world state to a per-bot JSON file so
    # the Conscious (LLM), Subconscious (ML) and Reflex brains can read the
    # SAME authoritative snapshot. Atomic write (temp+rename) + monotonic seq
    # so a concurrent reader never sees a torn/out-of-order file.
    _write_charstatus_file($snapshot);
}

# ── charstatus.json atomic writer ──
# Writes the enriched snapshot to data/charstatus/charstatus_<bot>.json.
# Atomic: write to a temp file in the same dir, then rename (atomic on same fs).
# Monotonic seq: rejects stale/out-of-order writes from a concurrent reader.
sub _write_charstatus_file {
    my ($snapshot) = @_;
    return if ref($snapshot) ne 'HASH';
    return if !_cfg_bool('aiSidecar_charstatusEnabled', 1);
    return if !$char;  # no char state yet (char-select / disconnected)

    my $bot_id = _bot_id();
    my $safe_bot = $bot_id;
    $safe_bot =~ s/[^A-Za-z0-9_.-]/_/g;
    $safe_bot = 'unknown' if $safe_bot eq '';

    # Resolve output dir once (default: data/charstatus/ under repo root).
    if ($_charstatus_dir eq '') {
        my $dir = _cfg('aiSidecar_charstatusDir', '');
        if ($dir eq '') {
            $dir = 'data/charstatus';
        }
        $_charstatus_dir = $dir;
    }
    my $dir = $_charstatus_dir;
    if (!-d $dir) {
        eval { require File::Path; File::Path::make_path($dir); 1; } or do {
            _throttled_warning('charstatus_mkdir_failed', "[aiSidecarBridge] cannot create charstatus dir $dir: $@");
            return;
        };
    }

    my $path = "$dir/charstatus_$safe_bot.json";
    my $tmp  = "$path.tmp.$$";

    # Monotonic seq — reject out-of-order writes.
    my $seq = ($_charstatus_seq{$bot_id} || 0) + 1;
    $_charstatus_seq{$bot_id} = $seq;

    # Enrich the snapshot with the full charstatus contract fields.
    my $cs = _build_charstatus_payload($snapshot, $seq);

    # Sanitize ALL strings to valid UTF-8 — game item/mob names carry CP949/
    # CP1252 bytes that JSON::PP would emit as invalid UTF-8 (corrupting the
    # file). Recursively clean every scalar.
    _sanitize_utf8_deep($cs);

    my $json;
    if ($json_available) {
        $json = eval { JSON::PP->new->canonical->encode($cs); 1; } ? JSON::PP->new->canonical->encode($cs) : undef;
    }
    if (!defined $json) {
        _throttled_warning('charstatus_encode_failed', "[aiSidecarBridge] charstatus JSON encode failed");
        return;
    }

    # Atomic write: temp + rename.
    if (open my $fh, '>', $tmp) {
        print $fh $json;
        close $fh;
        if (rename $tmp, $path) {
            $_charstatus_last_path{$bot_id} = $path;
        } else {
            unlink $tmp;
            _throttled_warning('charstatus_rename_failed', "[aiSidecarBridge] charstatus rename failed for $path");
        }
    } else {
        _throttled_warning('charstatus_write_failed', "[aiSidecarBridge] charstatus write failed for $tmp: $!");
    }
}

# ── charstatus.json full contract builder ──
# Builds the COMPLETE char+world state contract (all 11 sections) from the
# bridge snapshot + OpenKore core. This is the authoritative INPUT for all
# three brains. Read-only for brains — they never write to it.
sub _build_charstatus_payload {
    my ($snapshot, $seq) = @_;
    my $bot_id = _bot_id();
    my $now = time();
    my $in_game = ($net && $net->getState() == Network::IN_GAME) ? 1 : 0;

    # ── Status effects (SC) + cooldowns from $char->{statuses} ──
    my %status_effects;
    my %cooldowns;
    if ($char && ref($char->{statuses}) eq 'HASH') {
        for my $handle (keys %{$char->{statuses}}) {
            my $st = $char->{statuses}{$handle};
            next unless ref($st) eq 'HASH';
            my $remaining = 0;
            if ($st->{tick} && $st->{time}) {
                $remaining = int(($st->{time} + ($st->{tick} / 1000)) - $now);
                $remaining = 0 if $remaining < 0;
            }
            if ($handle =~ /_DELAY$/) {
                # Skill cooldown (ZC_SKILL_POSTDELAY stores <skill>_DELAY status)
                my $skill = $handle;
                $skill =~ s/_DELAY$//;
                $cooldowns{$skill} = $remaining;
            } else {
                $status_effects{$handle} = $remaining;
            }
        }
    }

    # ── Stats (str/agi/vit/int/dex/luk) from OpenKore core ──
    my %stats;
    if ($char) {
        $stats{str} = $char->{str} // 0;
        $stats{agi} = $char->{agi} // 0;
        $stats{vit} = $char->{vit} // 0;
        $stats{int} = $char->{int} // 0;
        $stats{dex} = $char->{dex} // 0;
        $stats{luk} = $char->{luk} // 0;
        $stats{str_bonus} = $char->{str_bonus} // 0;
        $stats{agi_bonus} = $char->{agi_bonus} // 0;
        $stats{vit_bonus} = $char->{vit_bonus} // 0;
        $stats{int_bonus} = $char->{int_bonus} // 0;
        $stats{dex_bonus} = $char->{dex_bonus} // 0;
        $stats{luk_bonus} = $char->{luk_bonus} // 0;
    }

    # ── Current attack target from AI task args ──
    my $target_id = '';
    my $target_name = '';
    my $target_hp_pct = 0;
    eval {
        my $args = AI::args(0);
        if ($args && $args->{attackID}) {
            $target_id = $args->{attackID};
            my $t = $monsters{$target_id};
            if ($t) {
                $target_name = $t->{name} || '';
                $target_hp_pct = ($t->{hp_max} && $t->{hp_max} > 0) ? int(($t->{hp} || 0) * 100 / $t->{hp_max}) : 0;
            }
        }
    };

    # ── Environment (server time, map flags) ──
    my $map = $snapshot->{position}{map} || '';
    my $is_town = 0;
    if ($map) {
        my @town_maps = qw(prontera morocc geffen payon aldebaran alberta izlude lutie xmas comodo yuno einbroch rachel veins nameless);
        for my $tm (@town_maps) {
            if ($map =~ /^\Q$tm\E/) { $is_town = 1; last; }
        }
    }

    return {
        schema_version => 1,
        seq            => $seq,
        snapshot_ts    => _iso_now(),
        server_time    => $now,
        freshness      => $in_game ? 'live' : 'stale',
        in_game        => $in_game,
        last_seen_ts   => $now,
        bot_id         => $bot_id,
        # ── 1. Identity ──
        identity => {
            account_id => $char ? _actor_id_from_any($char->{accountID}) : '',
            char_id    => $char ? _actor_id_from_any($char->{ID}) : '',
            name       => $char ? ($char->{name} // '') : '',
            job        => $char ? (defined $char->{jobName} ? $char->{jobName} : (_state_get('assigned_job') || 'novice')) : '',
            job_id     => $char ? ($char->{jobID} // 0) : 0,
            base_level => $char ? ($char->{lv} // 0) : 0,
            job_level  => $char ? ($char->{lv_job} // 0) : 0,
            gender     => $char ? ($char->{sex} // '') : '',
            guild_id   => $char ? _actor_id_from_any($char->{guildID}) : 0,
            party_id   => $char ? _actor_id_from_any($char->{party}{ID}) : 0,
        },
        # ── 2. Vitals ──
        vitals => {
            hp         => $char ? ($char->{hp} // 0) : 0,
            hp_max     => $char ? ($char->{hp_max} // 0) : 0,
            hp_ratio   => ($char && $char->{hp_max} > 0) ? (($char->{hp} || 0) / $char->{hp_max}) : 0,
            sp         => $char ? ($char->{sp} // 0) : 0,
            sp_max     => $char ? ($char->{sp_max} // 0) : 0,
            sp_ratio   => ($char && $char->{sp_max} > 0) ? (($char->{sp} || 0) / $char->{sp_max}) : 0,
            weight     => $char ? ($char->{weight} // 0) : 0,
            weight_max => $char ? ($char->{weight_max} // 0) : 0,
            weight_ratio => ($char && $char->{weight_max} > 0) ? (($char->{weight} || 0) / $char->{weight_max}) : 0,
            dead       => ($char && $char->{dead}) ? 1 : 0,
            sitting    => ($char && $char->{sitting}) ? 1 : 0,
            status_effects => \%status_effects,
        },
        # ── 3. Position & Movement ──
        position => {
            map => $map,
            x   => $snapshot->{position}{x},
            y   => $snapshot->{position}{y},
            direction => $char ? ($char->{direction} // 0) : 0,
            move_dest => ($char && ref($char->{pos_to}) eq 'HASH') ? { x => $char->{pos_to}{x}, y => $char->{pos_to}{y} } : undef,
            route_failure_count => $snapshot->{raw}{route_failure_count} || 0,
            route_churn_count   => $snapshot->{raw}{route_churn_count} || 0,
            stuck_detected      => ($snapshot->{raw}{route_failure_count} || 0) > 3 ? 1 : 0,
        },
        # ── 4. Inventory ──
        inventory => {
            zeny       => $char ? ($char->{zeny} // 0) : 0,
            item_count => $snapshot->{inventory}{item_count} || 0,
            weight     => $char ? ($char->{weight} // 0) : 0,
            weight_max => $char ? ($char->{weight_max} // 0) : 0,
            items      => $snapshot->{inventory_items} || [],
            equipment  => $snapshot->{progression}{equipment} || {},
        },
        # ── 5. Stats & Skills ──
        stats => \%stats,
        skills => {
            list        => $snapshot->{skills} || [],
            cooldowns   => \%cooldowns,
            skill_points => $char ? ($char->{points_skill} // 0) : 0,
            stat_points  => $char ? ($char->{status_points} // $char->{points_free} // 0) : 0,
        },
        # ── 6. Combat State ──
        combat => {
            ai_sequence   => $snapshot->{combat}{ai_sequence} || '',
            is_in_combat  => $snapshot->{combat}{is_in_combat} ? 1 : 0,
            target_id     => $target_id,
            target_name   => $target_name,
            target_hp_pct => $target_hp_pct,
            monster_count => $snapshot->{combat}{monster_count} || 0,
            nearby_monsters => $snapshot->{actors} || [],
        },
        # ── 7. Environment ──
        environment => {
            map_name => $map,
            is_town  => $is_town,
            is_field => ($map =~ /_fild|_field|_dun|_gld/) ? 1 : 0,
            time_of_day => _time_of_day($now),
        },
        # ── 8. Party/Guild ──
        party => {
            in_party      => $snapshot->{in_party} ? 1 : 0,
            members       => $snapshot->{party_members} || [],
            all_bots      => $snapshot->{all_bots} || [],
        },
        # ── 9. Economy ──
        economy => {
            zeny => $char ? ($char->{zeny} // 0) : 0,
        },
        # ── 10. AI/Internal ──
        ai => {
            current_ai_state => $snapshot->{combat}{ai_sequence} || '',
            ai_queue        => $snapshot->{raw}{ai_queue} || '',
            death_count     => $snapshot->{raw}{death_count} || 0,
            respawn_state   => $snapshot->{raw}{respawn_state} || '',
            reconnect_age_s => $snapshot->{raw}{reconnect_age_s} || 0,
            npc_dialog      => {
                npc_name => $snapshot->{raw}{npc_name} || '',
                npc_x    => $snapshot->{raw}{npc_x} || 0,
                npc_y    => $snapshot->{raw}{npc_y} || 0,
                last_text => $snapshot->{raw}{last_npc_text} || '',
                menu_options => $snapshot->{raw}{menu_options} || [],
            },
        },
        # ── 11. Telemetry ──
        telemetry => {
            latency_ms   => 0,
            session_id   => $bot_id,
            server_time  => $now,
        },
    };
}

# ── Time-of-day helper (server time) ──
sub _time_of_day {
    my ($t) = @_;
    my @g = gmtime($t);
    my $hour = $g[2];
    return 'night' if $hour < 6 || $hour >= 18;
    return 'day';
}

# ── Recursive UTF-8 sanitizer ──
# Game item/mob/char names carry CP949/CP1252 bytes (mojibake). JSON::PP emits
# them as invalid UTF-8, corrupting the charstatus file. Recursively clean
# every scalar: decode CP1252→UTF-8, then drop any remaining invalid bytes.
sub _sanitize_utf8_deep {
    my ($ref) = @_;
    if (ref($ref) eq 'HASH') {
        for my $k (keys %$ref) {
            my $v = $ref->{$k};
            if (ref($v)) {
                _sanitize_utf8_deep($v);
            } elsif (defined $v) {
                $ref->{$k} = _sanitize_utf8_scalar($v);
            }
        }
    } elsif (ref($ref) eq 'ARRAY') {
        for my $i (0 .. $#$ref) {
            my $v = $ref->[$i];
            if (ref($v)) {
                _sanitize_utf8_deep($v);
            } elsif (defined $v) {
                $ref->[$i] = _sanitize_utf8_scalar($v);
            }
        }
    }
}

sub _sanitize_utf8_scalar {
    my ($s) = @_;
    return $s unless defined $s;
    return $s if utf8::is_utf8($s) && eval { utf8::valid($s); 1; };
    # Try CP1252→UTF-8 (covers most RO mojibake); fall back to dropping
    # invalid bytes.
    my $out = eval {
        require Encode;
        Encode::decode('cp1252', $s, Encode::FB_DEFAULT());
    };
    if (defined $out && utf8::is_utf8($out) && eval { utf8::valid($out); 1; }) {
        return $out;
    }
    # Last resort: strip non-UTF8 bytes.
    $s =~ s/([^\x00-\x7F])/sprintf('\\x%02X', ord($1))/ge;
    return $s;
}

sub _build_snapshot_payload {
	my $bot_id = _bot_id();
	my $max_raw = _cfg_int('aiSidecar_maxRawChars', 256);
	my $progression = {};  # built at top-level below (2026-08-28)

	# ── PROGRESSION CACHE (2026-08-28) ──
	# Populate the last-known level/exp cache UNCONDITIONALLY on every
	# snapshot. $char (the OpenKore player actor) is only bound while
	# in-game; during death/respawn + reconnect cycles it is undef and the
	# nested progression eval below (inside `if ($_leader_lv >= 40)`) never
	# runs — so the snapshot progression was ALWAYS empty and the sidecar's
	# EXP-delta kill proxy never fired. This cache keeps the last-known
	# values across those cycles. NOTE: this fork stores level as {lv} and
	# job level as {lv_job} (VAR_CLEVEL/VAR_CJOBLEVEL handlers in
	# Network/Receive.pm) — NOT the stock {level}/{level_job} keys.
	if (defined($char) && ref($char)) {
		my $_ck = $bot_id || $::config{username} || 'unknown';
		$::aiSidecar_cached_progression{$_ck}{base_level}   = $char->{lv}       if defined $char->{lv};
		$::aiSidecar_cached_progression{$_ck}{base_exp}     = $char->{exp}      if defined $char->{exp};
		$::aiSidecar_cached_progression{$_ck}{base_exp_max} = $char->{exp_max}  if defined $char->{exp_max};
		$::aiSidecar_cached_progression{$_ck}{job_level}    = $char->{lv_job}   if defined $char->{lv_job};
		$::aiSidecar_cached_progression{$_ck}{job_exp}      = $char->{exp_job}  if defined $char->{exp_job};
		$::aiSidecar_cached_progression{$_ck}{job_exp_max}  = $char->{exp_job_max} if defined $char->{exp_job_max};
		$::aiSidecar_cached_progression{$_ck}{job_id}       = $char->{jobID}    if defined $char->{jobID};
		$::aiSidecar_cached_progression{$_ck}{skill_points} = $char->{points_skill} if defined $char->{points_skill};
		$::aiSidecar_cached_progression{$_ck}{stat_points}  = (defined $char->{status_points}) ? $char->{status_points}
			: (defined $char->{points_free}) ? $char->{points_free}
			: (defined $char->{stat_pts}) ? $char->{stat_pts} : 0;
	}

	# ── PROGRESSION PAYLOAD — built HERE, unconditionally (2026-08-28) ──
	# The old `$progression = eval {...}` below is nested inside
	# `if ($_leader_lv >= 40)` (party-formation) — a sub-40 bot NEVER ran it,
	# so the snapshot progression was ALWAYS empty and the sidecar's EXP-delta
	# kill proxy never fired. Build the payload at sub top-level instead.
	{
		my $_ck = $bot_id || $::config{username} || 'unknown';
		my $_c = $::aiSidecar_cached_progression{$_ck} || {};
		my %_pp = (
			job_id       => $_c->{job_id},
			base_level   => $_c->{base_level},
			job_level    => $_c->{job_level},
			base_exp     => $_c->{base_exp},
			base_exp_max => $_c->{base_exp_max},
			job_exp      => $_c->{job_exp},
			job_exp_max  => $_c->{job_exp_max},
			skill_points => $_c->{skill_points},
			stat_points  => $_c->{stat_points},
		);
		# drop undef keys so pydantic gets real values only
		$progression = { map { defined $_pp{$_} ? ($_ => $_pp{$_}) : () } keys %_pp };
	}

	my ($x, $y);
	my $map = '';
	if ($char) {
		my $pos = eval {
			# NOTE: never use bare 'return' inside this eval BLOCK —
			# return in eval returns from _build_snapshot_payload, not
			# the eval. Assign to $p and use it as last expression.
			my $p;
			if ($char->{pos_to} && ref $char->{pos_to} eq 'HASH') {
				$p = $char->{pos_to};
			} elsif ($char->{pos} && ref $char->{pos} eq 'HASH') {
				$p = $char->{pos};
			}
			$p;
		};
		if ($pos) {
			$x = $pos->{x};
			$y = $pos->{y};
		}
	}
	$map = eval { $field ? $field->baseName() : '' } || '';

	my $ai_top = @ai_seq ? $ai_seq[0] : '';
	my $in_combat = defined $ai_top && $ai_top =~ /^(?:attack|skill_use|route|follow)/ ? 1 : 0;

	my $item_count;
	my @inventory_items_digest;
	my $has_weapon_in_inventory = 0;
	if ($char) {
		my $_inv = _char_inventory($char);
		$item_count = scalar @{$_inv};
		# Build a compact inventory digest (id + name + qty + equipped + type) so
		# the sidecar's cold-start / gear logic sees REAL items (not the dead
		# hash-key reads that always returned empty).
		for my $_inv_item (@{$_inv}) {
			next unless defined $_inv_item;
			my $_inv_type = $_inv_item->{type} || 0;
			my $_inv_name = $_inv_item->{name} || '';
			push @inventory_items_digest, {
				item_id  => (($_inv_item->{nameID} // 0) + 0) . '',  # contract: str
				name     => $_inv_name,
				quantity => ($_inv_item->{amount} // 0) + 0,
				type     => $_inv_type,
				equipped => (($_inv_item->{equipped} || 0) ? 1 : 0),
			} if $_inv_name || ($_inv_item->{nameID} // 0);
			# type 4 = weapon, type 5 = armor, type 6 = card, type 7 = pet, type 8 = accessory
			if ($_inv_type == 4 || $_inv_type == 5 || $_inv_type == 8) {
				$has_weapon_in_inventory = 1;
			}
		}
	}

	my $raw = {
		char_name  => _trim($char ? ($char->{name} || '') : '', $max_raw),
		master     => _trim($config{master} || '', $max_raw),
		map        => $map,
		ai_sequence => _trim($ai_top || '', $max_raw),
		ai_queue   => _trim(join(',', @ai_seq[0 .. ($#ai_seq < 4 ? $#ai_seq : 4)]), $max_raw),
		in_game => ($net && $net->getState() == Network::IN_GAME) ? JSON::PP::true() : JSON::PP::false(),
		net_state => ($net ? ($net->getState() + 0) : -1),
		lockMap => _trim($::config{lockMap} || '', 64),
		reconnect_age_s => ($last_disconnect_at_ms > 0 && $net && $net->getState() == Network::IN_GAME)
			? ((_now_ms() - $last_disconnect_at_ms) / 1000.0)
			: 0.0,
		death_count => 0 + $death_count,
		respawn_state => _trim($respawn_state, 32),
		route_churn_count => 0 + $route_churn_count,
		route_failure_count => 0 + $route_failure_count,
		# ── NPC DIALOG STATE (conscious-tier LLM dialog responder input) ──
		# The LLM (_llm_npc_dialog_response) reads the ACTUAL menu options and
		# NPC identity to pick the response agnostically (founder 2026-08-25:
		# no hardcoded talknpc sequences). Feed the live dialog state through.
		npc_name => _trim($_npc_dialog_state{npc_name} || '', $max_raw),
		npc_x    => 0 + ($_npc_dialog_state{npc_x} || 0),
		npc_y    => 0 + ($_npc_dialog_state{npc_y} || 0),
		last_npc_text => _trim($_npc_dialog_state{last_text} || '', $max_raw),
		menu_options => [ map { _trim($_, 200) } @{$_npc_dialog_state{menu_options} || []} ],
		# ── PATHFINDING LIVE-OBJECT COUNTERS (leak diagnostics 2026-08-25) ──
		# The live bot showed steady RAM growth; expose the XS create/destroy
		# counters so the sidecar can tell an OBJECT leak (live grows) from a
		# core-loop leak (live flat, RSS grows). Guarded — PathFinding->stats()
		# is only present in the instrumented .so.
		pathfinding => eval {
			my $pfs = PathFinding->can('stats') ? PathFinding->stats() : undef;
			$pfs ? {
				created   => 0 + ($pfs->{created} || 0),
				destroyed => 0 + ($pfs->{destroyed} || 0),
				live      => 0 + ($pfs->{live} || 0),
			} : undef;
		},
		# Field objects hold the full map data (~5MB each). If live grows while
		# RSS grows, Fields are leaking (route calc loads a Field per map attempt).
		fields => eval {
			my $fs = Field->can('stats') ? Field->stats() : undef;
			$fs ? {
				created   => 0 + ($fs->{created} || 0),
				destroyed => 0 + ($fs->{destroyed} || 0),
				live      => 0 + ($fs->{live} || 0),
			} : undef;
		},
		# ── charstatus contract enrichment (2026-08-27) ──
		# Status effects (SC) + skill cooldowns from $char->{statuses}.
		status_effects => eval {
			my %se;
			if ($char && ref($char->{statuses}) eq 'HASH') {
				for my $h (keys %{$char->{statuses}}) {
					next if $h =~ /_DELAY$/;
					my $st = $char->{statuses}{$h};
					next unless ref($st) eq 'HASH';
					my $rem = 0;
					if ($st->{tick} && $st->{time}) {
						$rem = int(($st->{time} + ($st->{tick} / 1000)) - time());
						$rem = 0 if $rem < 0;
					}
					$se{$h} = $rem;
				}
			}
			\%se;
		},
		cooldowns => eval {
			my %cd;
			if ($char && ref($char->{statuses}) eq 'HASH') {
				for my $h (keys %{$char->{statuses}}) {
					next unless $h =~ /_DELAY$/;
					my $st = $char->{statuses}{$h};
					next unless ref($st) eq 'HASH';
					my $rem = 0;
					if ($st->{tick} && $st->{time}) {
						$rem = int(($st->{time} + ($st->{tick} / 1000)) - time());
						$rem = 0 if $rem < 0;
					}
					my $skill = $h;
					$skill =~ s/_DELAY$//;
					$cd{$skill} = $rem;
				}
			}
			\%cd;
		},
		# Stats (str/agi/vit/int/dex/luk) from OpenKore core.
		stats => eval {
			my %s;
			if ($char) {
				$s{str} = $char->{str} // 0;
				$s{agi} = $char->{agi} // 0;
				$s{vit} = $char->{vit} // 0;
				$s{int} = $char->{int} // 0;
				$s{dex} = $char->{dex} // 0;
				$s{luk} = $char->{luk} // 0;
			}
			\%s;
		},
		# Current attack target from AI task args.
		target => eval {
			my $args = AI::args(0);
			my %t;
			if ($args && $args->{attackID}) {
				$t{id} = $args->{attackID};
				my $m = $monsters{$args->{attackID}};
				if ($m) {
					$t{name} = $m->{name} || '';
					$t{hp_pct} = ($m->{hp_max} && $m->{hp_max} > 0) ? int(($m->{hp} || 0) * 100 / $m->{hp_max}) : 0;
				}
			}
			\%t;
		},
	};

	# --- Progression digest (job, level, exp) ---
	if ($char) {
		# ── Populate all_bots from .bot_profiles directory ──
		if (!defined $::aiSidecar_all_bots || $::aiSidecar_all_bots eq '') {
			my @_profiles;
			my $_prof_dir = ".bot_profiles";
			if (-d $_prof_dir) {
				opendir my $_dh, $_prof_dir or do { debug "bridge_all_bots: cannot open $_prof_dir\n", 'aiSidecarBridge', 1 };
				if ($_dh) {
					@_profiles = sort grep { !/^\./ && -d "$_prof_dir/$_" } readdir($_dh);
					closedir $_dh;
				}
			}
			$::aiSidecar_all_bots = join(',', @_profiles);
			debug "bridge_all_bots: discovered " . scalar(@_profiles) . " profiles: $::aiSidecar_all_bots\n", 'aiSidecarBridge', 1;
		}
		my $_leader_lv = defined($char) ? ($char->{lv} || $char->{level} || 0) : 0;
		if ($_leader_lv >= 40) {
		# ── Party join auto-accept: non-leader bots accept invites ──
				# Non-leader: set partyAuto=2 to auto-accept invites
				# Leader is determined by all_bots order from sidecar
				if (defined($char) && !defined($char->{party})) {
					# partyAuto controlled by heuristic — bridge must NOT override
				}
		# ── Direct party invite: leader invites missing members ──
		# OBSERVABILITY ONLY — the fleet coordinator (sidecar) is the party
		# actor. These blocks are leftover pre-coordinator logic; emitting
		# party commands from the bridge fights the coordinator's gating and
		# produced the frozen party-request spam (Commands::run dispatches the
		# handler with the ORIGINAL switch, so hook gates cannot stop it).
		if (@::aiSidecar_all_bots_split && ($::config{username} || '') eq $::aiSidecar_all_bots_split[0] && $_leader_lv >= 40) {
			if (!defined($char->{party})) {
				my $_ts = time();
				debug "bridge_party_create: would create party AI$_ts (coordinator owns formation)\n", 'aiSidecarBridge', 1;
			}
		}
	# PARTY HEARTBEAT: If not in party, create one (only at level 40+)
	# OBSERVABILITY ONLY — coordinator owns party formation (see above).
	if (!defined($char->{party}) && @::aiSidecar_all_bots_split && $_leader_lv >= 40) {
		my $_now = time();
		if (($_now - ($::aiSidecar_last_party_create || 0)) > 5) {
			$::aiSidecar_last_party_create = $_now;
			my $_party_name = 'AI' . int($_now);
			debug "bridge_party_create: would create '$_party_name' (coordinator owns formation)\n", 'aiSidecarBridge', 1;
		}
	}
		# Then invite missing members (only at level 40+)
		if (@::aiSidecar_all_bots_split && ($::config{username} || '') eq $::aiSidecar_all_bots_split[0] && defined($char->{party}) && $_leader_lv >= 40) {
			my $_pu = $char->{party}{users} || {};
				my %_mn;
				for my $_uid (keys %$_pu) {
					my $_pm = $_pu->{$_uid};
					my $_pn = eval { $_pm->{name} || $_pm->name() || '' } || '';
					$_mn{$_pn} = 1 if $_pn;
				}
				$_mn{$char->{name}} = 1;
				# Dynamic mapping: use all_bots from sidecar
				for my $_pn (@::aiSidecar_all_bots_split) {
					next if $_pn eq ($::config{username} || '');
					my $_cn = $::aiSidecar_profile_to_char{$_pn} || $_pn;  # Use char name from sidecar
					if (!$_mn{$_cn}) {
						debug "bridge_party_invite: would request $_cn (coordinator owns invites)\n", 'aiSidecarBridge', 1;
					}
				}
				}
				# Leader invites missing members
				# Leader detection: check if this bot is the first in all_bots
				# all_bots comes from sidecar via snapshot cache
				if (defined($char->{party})) {
				# Check if we're the leader by reading all_bots from shared state
				# For now, use a simple heuristic: the bot with the lowest username alphabetically is leader
				my $_all_bots_str = $::aiSidecar_all_bots || '';
				my @_all_bots = split(',', $_all_bots_str);
				my $_is_leader = @_all_bots && ($::config{username} || '') eq $_all_bots[0];
				if ($_is_leader) {
				my $_pu = $char->{party}{users} || {};
				my %_mn;
				for my $_uid (keys %$_pu) {
					my $_pm = $_pu->{$_uid};
					my $_pn = eval { $_pm->{name} || $_pm->name() || '' } || '';
					$_mn{$_pn} = 1 if $_pn;
				$_mn{$char->{name}} = 1;
				# Dynamic mapping: use all_bots from sidecar
				for my $_pn (@_all_bots) {
					next if $_pn eq ($::config{username} || '');
					my $_cn = $::aiSidecar_profile_to_char{$_pn} || $_pn;  # Use char name from sidecar
					if (!$_mn{$_cn}) {
						debug "bridge_party_invite: would request $_cn (coordinator owns invites)\n", 'aiSidecarBridge', 1;
					}
				}
				}
				}
		}
		$progression = eval {
			my %p;
			$p{job_id}       = $char->{jobID}     if defined $char->{jobID};
			$p{base_level}   = $char->{level}      if defined $char->{level};
			$p{job_level}    = $char->{level_job}  if defined $char->{level_job};
			$p{base_exp}     = $char->{exp}        if defined $char->{exp};
			$p{base_exp_max} = $char->{exp_max}    if defined $char->{exp_max};
			$p{job_exp}      = $char->{exp_job}    if defined $char->{exp_job};
			$p{job_exp_max}  = $char->{exp_job_max} if defined $char->{exp_job_max};
			$p{skill_points} = $char->{points_skill} if defined $char->{points_skill};
			if (defined $char->{status_points}) { $p{stat_points} = $char->{status_points}; }
			elsif (defined $char->{points_free}) { $p{stat_points} = $char->{points_free}; }
			elsif (defined $char->{stat_pts}) { $p{stat_points} = $char->{stat_pts}; }
			else { $p{stat_points} = 0; }
			$p{job_name}     = (defined $char->{jobName} ? $char->{jobName} : (_state_get('assigned_job') || 'novice')) if defined $char;
			# Party signals go into raw field (BotStateSnapshot ignores extra top-level fields)
			# Party members are in $char->{party}{users}{$id}{'name'} (keys are numeric IDs, values are HASH refs)
			# Cache party state to survive death/respawn
			# Use bot_id (profile name) as cache key
			my $_cache_key = $p{bot_id} || $::config{username} || $ENV{BOT_NAME} || 'unknown';
			# Check $char exists first - can be undef during death/disconnect
			if (defined($char) && defined($char->{party})) {
				$raw->{in_party} = 1;
				$raw->{party_members} = [];
				if (ref($char->{party}{users}) eq "HASH") {
					for my $_pm_key (keys %{$char->{party}{users}}) {
						my $_pm = $char->{party}{users}{$_pm_key};
						my $_pm_name = '';
						if (UNIVERSAL::can($_pm, 'name')) {
							$_pm_name = eval { $_pm->name() } || '';
						}
						if (!$_pm_name) {
							$_pm_name = eval { $_pm->{name} } || '';
						}
						if ($_pm_name) {
							push @{$raw->{party_members}}, lc($_pm_name);
						}
					}
				}
				if (scalar(@{$raw->{party_members}}) == 0 && $char->{name}) {
					push @{$raw->{party_members}}, lc($char->{name});
				}
				$::aiSidecar_cached_party{$_cache_key} = {
					in_party => $raw->{in_party},
					members => [@{$raw->{party_members}}],
				};
			} elsif (defined($::aiSidecar_cached_party{$_cache_key})) {
				my $cache = $::aiSidecar_cached_party{$_cache_key};
				$raw->{in_party} = $cache->{in_party};
				$raw->{party_members} = [@{$cache->{members}}];
			} else {
				$raw->{in_party} = 0;
				$raw->{party_members} = [];
			}
			# attack_power: total attack power (weapon + stats)
			$p{attack_power} = $char->{attack} || $char->{atk} || 0;
			# equipment digest: all equipped items with slot/card/refine info
			my %_equip;
			if ($char->{equipment}) {
				for my $_eq_slot (keys %{$char->{equipment}}) {
					my $_eq = $char->{equipment}{$_eq_slot};
					next unless defined $_eq;
					$_equip{$_eq_slot} = {
						id => $_eq->{nameID} || 0,
						name => $_eq->{name} || '',
						refine => $_eq->{refine} || 0,
						cards => [grep { defined && $_ > 0 } ($_eq->{card1} || 0, $_eq->{card2} || 0, $_eq->{card3} || 0, $_eq->{card4} || 0)],
					};
				}
			}
			$p{equipment} = \%_equip;
			# inventory weapon check: do we have any weapon in inventory?
			$p{has_weapon_in_inventory} = 0;
			$p{inventory_items} = [];
			if (@{_char_inventory($char)}) {
				for my $_inv_item (@{_char_inventory($char)}) {
					next unless defined $_inv_item;
					my $_inv_type = $_inv_item->{type} || 0;
					my $_inv_name = $_inv_item->{name} || '';
					push @{$p{inventory_items}}, $_inv_name if $_inv_name;
					# type 4 = weapon, type 5 = armor, type 6 = card, type 7 = pet, type 8 = accessory
					if ($_inv_type == 4 || $_inv_type == 5 || $_inv_type == 8) {
						$p{has_weapon_in_inventory} = 1;
						last;
					}
				}
			}
			# total attack with weapon info
			$p{atk_min} = $char->{attack} || 0;
			$p{atk_max} = $char->{attack_max} || $char->{attack} || 0;
			$p{matk_min} = $char->{matk_min} || 0;
			$p{matk_max} = $char->{matk_max} || 0;
			$p{def} = $char->{def} || 0;
			$p{mdef} = $char->{mdef} || 0;
			$p{hit} = $char->{hit} || 0;
			$p{flee} = $char->{flee} || 0;
			$p{crit} = $char->{crit} || 0;
			$p{aspd} = $char->{aspd} || 0;
			# all_bots: list of all known bot names (from env vars BOT_*_PASS)
			my @_bot_names;
			for my $_env_key (keys %ENV) {
				if ($_env_key =~ /^BOT_(.+)_PASS$/i) {
					push @_bot_names, lc($1);
				}
			}
			# Fallback: if no env vars found, use configured bot profiles
			if (!@_bot_names) {
				require Cwd;
				# NOTE: never use bare 'return;' here — this whole block is
				# inside 'eval { ... }', and return inside eval BLOCK returns
				# from the ENCLOSING SUB (_build_snapshot_payload), aborting
				# the snapshot with undef. Use if/else instead.
				if (opendir(my $_dh, Cwd::cwd() . "/.bot_profiles")) {
					while (my $_entry = readdir($_dh)) {
						next if $_entry =~ /^\./;
						next unless -d Cwd::cwd() . "/.bot_profiles/$_entry";
						push @_bot_names, $_entry;
					}
					closedir($_dh);
				} else {
					$p{all_bots} = [];
				}
			}
			debug "[all_bots] found: @_bot_names\n", 'aiSidecarBridge', 1;
			$p{all_bots} = \@_bot_names;
			$raw->{all_bots} = \@_bot_names;
			\%p;
		} || {};
	}
	} # CLOSE if($char) — CRITICAL FIX: without this the skills/actors/characters
	  # digests AND the return hash are swallowed inside if($char), so when
	  # $char is undef (char-select) the snapshot builds as "" and cold_start
	  # never sees existing characters.

	# --- Skills digest (known skills with levels) ---
	my @skills_list;
	if ($char && $char->{skills} && ref $char->{skills} eq 'HASH') {
		@skills_list = map {
			my $skill = $char->{skills}{$_};
			+{
				skill_id   => $_,
				skill_name => (defined $skill->{name} ? $skill->{name} : $_),
				level      => (defined $skill->{lv} ? $skill->{lv} + 0 : 0),
			}
		} sort keys %{$char->{skills}};
	}

	# --- Actors digest (nearby mobs, players, NPCs) ---
	my @actors;
	my $actor_discovery = {
		enabled => _cfg_bool('aiSidecar_actorsEnabled', 1) ? 1 : 0,
		source_counts => {
			monster => { hash => 0, list => 0, merged_candidates => 0 },
			player  => { hash => 0, list => 0, merged_candidates => 0 },
			npc     => { hash => 0, list => 0, merged_candidates => 0 },
		},
		normalize => {
			seen_total => 0,
			kept_total => 0,
			skipped_total => 0,
			seen_by_type => { monster => 0, player => 0, npc => 0 },
			kept_by_type => { monster => 0, player => 0, npc => 0 },
			skipped_by_type => { monster => 0, player => 0, npc => 0 },
			skipped_reasons => {
				non_hash => 0,
				missing_actor_id => 0,
				duplicate_actor_id => 0,
				over_limit => 0,
			},
			id_fallback_from_hash_key => 0,
		},
		payload => {
			snapshot_actor_count => 0,
			max_actors => 0,
			truncated => 0,
		},
	};
	if (_cfg_bool('aiSidecar_actorsEnabled', 1)) {
		my $max_actors = _cfg_int('aiSidecar_maxActors', 24);
		$max_actors = 0 if !defined $max_actors || $max_actors < 0;
		$actor_discovery->{payload}{max_actors} = 0 + $max_actors;

		my %seen_actor_ids;
		my $party_members = ($char && $char->{party} && ref($char->{party}{users}) eq 'HASH')
			? $char->{party}{users}
			: undef;
		my %_party_member_names;
		if ($party_members) {
			for my $_pmk (keys %{$party_members}) {
				my $_pmv = $party_members->{$_pmk};
				# Actor::Party objects are blessed HASH refs - use eval for safe access
				my $_pmv_name = eval { $_pmv->{name} } || '';
				if ($_pmv_name) {
					$_party_member_names{$_pmk} = 1;
				}
			}
		}

		my $append_actor = sub {
			my (%args) = @_;
			my $actor = $args{actor};
			my $actor_type = _trim(_scalarize($args{actor_type}), 32);
			$actor_type = 'unknown' if !defined $actor_type || $actor_type eq '';

			$actor_discovery->{normalize}{seen_total} += 1;
			$actor_discovery->{normalize}{seen_by_type}{$actor_type} =
				0 + ($actor_discovery->{normalize}{seen_by_type}{$actor_type} || 0) + 1;

			if (!_is_hash_like($actor)) {
				$actor_discovery->{normalize}{skipped_total} += 1;
				$actor_discovery->{normalize}{skipped_by_type}{$actor_type} =
					0 + ($actor_discovery->{normalize}{skipped_by_type}{$actor_type} || 0) + 1;
				$actor_discovery->{normalize}{skipped_reasons}{non_hash} += 1;
				return;
			}

			my $actor_id = _actor_id_from_any($actor->{ID});
			my $used_fallback_id = 0;
			if ($actor_id eq '' && defined $args{fallback_actor_id}) {
				my $fallback_actor_id = _actor_id_from_any($args{fallback_actor_id});
				if ($fallback_actor_id ne '') {
					$actor_id = $fallback_actor_id;
					$used_fallback_id = 1;
				}
			}
			if ($actor_id eq '') {
				$actor_discovery->{normalize}{skipped_total} += 1;
				$actor_discovery->{normalize}{skipped_by_type}{$actor_type} =
					0 + ($actor_discovery->{normalize}{skipped_by_type}{$actor_type} || 0) + 1;
				$actor_discovery->{normalize}{skipped_reasons}{missing_actor_id} += 1;
				return;
			}
			if ($used_fallback_id) {
				$actor_discovery->{normalize}{id_fallback_from_hash_key} =
					0 + ($actor_discovery->{normalize}{id_fallback_from_hash_key} || 0) + 1;
			}

			if ($seen_actor_ids{$actor_id}) {
				$actor_discovery->{normalize}{skipped_total} += 1;
				$actor_discovery->{normalize}{skipped_by_type}{$actor_type} =
					0 + ($actor_discovery->{normalize}{skipped_by_type}{$actor_type} || 0) + 1;
				$actor_discovery->{normalize}{skipped_reasons}{duplicate_actor_id} += 1;
				return;
			}
			$seen_actor_ids{$actor_id} = 1;

			if (scalar(@actors) >= $max_actors) {
				$actor_discovery->{normalize}{skipped_total} += 1;
				$actor_discovery->{normalize}{skipped_by_type}{$actor_type} =
					0 + ($actor_discovery->{normalize}{skipped_by_type}{$actor_type} || 0) + 1;
				$actor_discovery->{normalize}{skipped_reasons}{over_limit} += 1;
				return;
			}

			my $relation = _trim(_scalarize($args{relation} || 'neutral'), 32);
			if (ref($args{relation_cb}) eq 'CODE') {
				$relation = _trim(_scalarize($args{relation_cb}->($actor)), 32);
				$relation = 'neutral' if !defined $relation || $relation eq '';
			}

			my $dmg_to     = defined $args{include_damage} && $args{include_damage} ? ($actor->{dmgTo} || 0) : undef;
			my $dmg_from   = defined $args{include_damage} && $args{include_damage} ? ($actor->{dmgFrom} || 0) : undef;
			my $dmg_to_you = defined $args{include_damage} && $args{include_damage} ? ($actor->{dmgFromYou} || 0) : undef;
			my $name_id    = defined $args{include_damage} && $args{include_damage} ? ($actor->{nameID} || undef) : undef;
			my $casting    = defined $args{include_damage} && $args{include_damage} ? ($actor->{casting} || undef) : undef;
			my $missed_you = defined $args{include_damage} && $args{include_damage} ? ($actor->{missedYou} || 0) : undef;
			my $cast_on_you = defined $args{include_damage} && $args{include_damage} ? ($actor->{castOnToYou} || 0) : undef;
			push @actors, {
				actor_id   => $actor_id,
				actor_type => $actor_type,
				name       => _trim($actor->{name} || ($args{default_name} || 'Actor'), 64),
				relation   => $relation,
				x          => defined $actor->{pos_to} && ref $actor->{pos_to} eq 'HASH' ? $actor->{pos_to}{x} : ($actor->{pos} && ref $actor->{pos} eq 'HASH' ? $actor->{pos}{x} : undef),
				y          => defined $actor->{pos_to} && ref $actor->{pos_to} eq 'HASH' ? $actor->{pos_to}{y} : ($actor->{pos} && ref $actor->{pos} eq 'HASH' ? $actor->{pos}{y} : undef),
				distance   => _calc_distance($actor, $char),
				hp         => defined $args{include_hp} && $args{include_hp} ? ($actor->{hp} || undef) : undef,
				hp_max     => defined $args{include_hp} && $args{include_hp} ? ($actor->{hp_max} || undef) : undef,
				level      => $actor->{level} || undef,
				dmg_to_us  => $dmg_from,
				dmg_to_you => $dmg_to_you,
				dmg_from_us => $dmg_to,
				name_id    => $name_id,
				casting    => $casting,
				missed_you => $missed_you,
				cast_on_you => $cast_on_you,
			};

			$actor_discovery->{normalize}{kept_total} += 1;
			$actor_discovery->{normalize}{kept_by_type}{$actor_type} =
				0 + ($actor_discovery->{normalize}{kept_by_type}{$actor_type} || 0) + 1;
		};

		# Nearby monsters (hash + ActorList)
		eval {
			my @mons_hash = map {
				{
					actor => $monsters{$_},
					fallback_actor_id => $_,
				}
			} keys %monsters;
			my @mons_list = _actor_list_items($monstersList);
			$actor_discovery->{source_counts}{monster}{hash} = scalar(@mons_hash) + 0;
			$actor_discovery->{source_counts}{monster}{list} = scalar(@mons_list) + 0;
			my @mons_merged = (
				@mons_hash,
				map {
					{
						actor => $_,
						fallback_actor_id => undef,
					}
				} @mons_list,
			);
			$actor_discovery->{source_counts}{monster}{merged_candidates} = scalar(@mons_merged) + 0;

			for my $entry (@mons_merged) {
				$append_actor->(
					actor => $entry->{actor},
					fallback_actor_id => $entry->{fallback_actor_id},
					actor_type => 'monster',
					default_name => 'Monster',
					relation => 'hostile',
					include_hp => 1,
					include_damage => 1,
				);
			}
		};

		# Nearby players (hash + ActorList)
		eval {
			my @players_hash = map {
				{
					actor => $players{$_},
					fallback_actor_id => $_,
				}
			} keys %players;
			my @players_list = _actor_list_items($playersList);
			$actor_discovery->{source_counts}{player}{hash} = scalar(@players_hash) + 0;
			$actor_discovery->{source_counts}{player}{list} = scalar(@players_list) + 0;
			my @players_merged = (
				@players_hash,
				map {
					{
						actor => $_,
						fallback_actor_id => undef,
					}
				} @players_list,
			);
			$actor_discovery->{source_counts}{player}{merged_candidates} = scalar(@players_merged) + 0;

			for my $entry (@players_merged) {
				$append_actor->(
					actor => $entry->{actor},
					fallback_actor_id => $entry->{fallback_actor_id},
					actor_type => 'player',
					default_name => 'Player',
					relation_cb => sub {
						my ($row) = @_;
						my $relation = 'neutral';
						if ($party_members && _is_hash_like($row) && defined $row->{binID}) {
							$relation = 'party' if exists $_party_member_names{$row->{binID}};
						}
						return $relation;
					},
					include_hp => 0,
				);
			}
		};

		# Nearby NPCs (hash + ActorList)
		eval {
			my @npcs_hash = map {
				{
					actor => $npcs{$_},
					fallback_actor_id => $_,
				}
			} keys %npcs;
			my @npcs_list = _actor_list_items($npcsList);
			$actor_discovery->{source_counts}{npc}{hash} = scalar(@npcs_hash) + 0;
			$actor_discovery->{source_counts}{npc}{list} = scalar(@npcs_list) + 0;
			my @npcs_merged = (
				@npcs_hash,
				map {
					{
						actor => $_,
						fallback_actor_id => undef,
					}
				} @npcs_list,
			);
			$actor_discovery->{source_counts}{npc}{merged_candidates} = scalar(@npcs_merged) + 0;

			for my $entry (@npcs_merged) {
				$append_actor->(
					actor => $entry->{actor},
					fallback_actor_id => $entry->{fallback_actor_id},
					actor_type => 'npc',
					default_name => 'NPC',
					relation => 'neutral',
					include_hp => 0,
				);
			}
		};

		my $now_ms = _now_ms();
		if ($now_ms - $last_actor_source_probe_log_ms >= 5000) {
			debug sprintf(
				"[aiSidecarBridge] actor source probe pre-normalize raw_containers={monster:{hash=%d,list=%d} player:{hash=%d,list=%d} npc:{hash=%d,list=%d}}\n",
				0 + ($actor_discovery->{source_counts}{monster}{hash} || 0),
				0 + ($actor_discovery->{source_counts}{monster}{list} || 0),
				0 + ($actor_discovery->{source_counts}{player}{hash} || 0),
				0 + ($actor_discovery->{source_counts}{player}{list} || 0),
				0 + ($actor_discovery->{source_counts}{npc}{hash} || 0),
				0 + ($actor_discovery->{source_counts}{npc}{list} || 0),
			), 'aiSidecarBridge', 2;
			$last_actor_source_probe_log_ms = $now_ms;
		}

		$actor_discovery->{payload}{truncated} = $actor_discovery->{normalize}{skipped_reasons}{over_limit} > 0 ? 1 : 0;
		debug sprintf(
			"[aiSidecarBridge] actor discovery source_counts={m:h=%d,l=%d p:h=%d,l=%d n:h=%d,l=%d} normalize={seen=%d kept=%d skipped=%d missing_id=%d id_fallback=%d dup_id=%d over_limit=%d} payload={count=%d max=%d truncated=%d}\n",
			0 + ($actor_discovery->{source_counts}{monster}{hash} || 0),
			0 + ($actor_discovery->{source_counts}{monster}{list} || 0),
			0 + ($actor_discovery->{source_counts}{player}{hash} || 0),
			0 + ($actor_discovery->{source_counts}{player}{list} || 0),
			0 + ($actor_discovery->{source_counts}{npc}{hash} || 0),
			0 + ($actor_discovery->{source_counts}{npc}{list} || 0),
			0 + ($actor_discovery->{normalize}{seen_total} || 0),
			0 + ($actor_discovery->{normalize}{kept_total} || 0),
			0 + ($actor_discovery->{normalize}{skipped_total} || 0),
			0 + ($actor_discovery->{normalize}{skipped_reasons}{missing_actor_id} || 0),
			0 + ($actor_discovery->{normalize}{id_fallback_from_hash_key} || 0),
			0 + ($actor_discovery->{normalize}{skipped_reasons}{duplicate_actor_id} || 0),
			0 + ($actor_discovery->{normalize}{skipped_reasons}{over_limit} || 0),
			scalar(@actors) + 0,
			0 + $max_actors,
			0 + ($actor_discovery->{payload}{truncated} || 0),
		), 'aiSidecarBridge', 2;
	}
	$actor_discovery->{payload}{snapshot_actor_count} = scalar(@actors) + 0;
	$raw->{actor_discovery} = $actor_discovery;

	# --- Character list digest (parsed @chars from char server) ---
	# Lets the sidecar cold_start see existing characters so it stops
	# trying to create new ones / relogging at the char select screen.
	my @characters;
	my @_chars_list = @main::chars;
	if (@_chars_list) {
		for my $i (0 .. $#_chars_list) {
			next unless $_chars_list[$i] && ref($_chars_list[$i]) eq 'HASH' && %{$_chars_list[$i]};
			my $c = $_chars_list[$i];
			push @characters, {
				slot       => $i,
				name       => _trim($c->{name} || '', 64),
				job_id     => ($c->{jobID} // $c->{job} // 0) + 0,
				base_level => ($c->{lv} // $c->{level} // 0) + 0,
				job_level  => ($c->{lv_job} // 0) + 0,
				sex        => ($c->{sex} // '') . '',
				last_map   => _trim($c->{last_map} || '', 32),
				zeny       => ($c->{zeny} // 0) + 0,
			};
		}
	}
	$raw->{characters} = \@characters;

	return {
		meta       => _meta($bot_id),
		tick_id    => _trace_id(),
		observed_at => _iso_now(),
		position   => {
			map => $map || undef,
			x   => $x,
			y   => $y,
		},
		vitals => {
			hp         => $char ? $char->{hp}         : undef,
			hp_max     => $char ? $char->{hp_max}     : undef,
			hp_ratio   => ($char && $char->{hp_max} > 0) ? ($char->{hp} || 0) / $char->{hp_max} : 0,
			sp         => $char ? $char->{sp}         : undef,
			sp_max     => $char ? $char->{sp_max}     : undef,
			sp_ratio   => ($char && $char->{sp_max} > 0) ? ($char->{sp} || 0) / $char->{sp_max} : 0,
			weight     => $char ? $char->{weight}     : undef,
			weight_max => $char ? $char->{weight_max} : undef,
			weight_ratio => ($char && $char->{weight_max} > 0) ? ($char->{weight} || 0) / $char->{weight_max} : 0,
			level      => $char ? $char->{level}      : undef,
			base_level => $char ? $char->{level}      : undef,
			job_level  => $char ? $char->{level_job}  : undef,
			zeny       => $char ? $char->{zeny}       : undef,
		},
		combat => {
			ai_sequence  => $ai_top || undef,
			ai_state_str => _safe_ai_seq_top(),
			is_sitting   => ($char && $char->{sitting}) ? 1 : 0,
			target_id    => undef,
			is_in_combat => $in_combat,
			monster_count => scalar(keys %monsters) + 0,
		},
		inventory => {
			zeny       => $char ? $char->{zeny} : undef,
			item_count => $item_count,
			weight     => $char ? $char->{weight} : undef,
			weight_max => $char ? $char->{weight_max} : undef,
			weight_ratio => ($char && $char->{weight_max} > 0) ? ($char->{weight} || 0) / $char->{weight_max} : 0,
			overweight_ratio => ($char && $char->{weight_max} > 0) ? ($char->{weight} || 0) / $char->{weight_max} : 0,
		},
		inventory_items => \@inventory_items_digest,
		has_weapon_in_inventory => $has_weapon_in_inventory,
		progression => $progression,
		skills      => \@skills_list,
		actors      => \@actors,
		characters  => \@characters,
		in_party      => $raw->{in_party} || 0,
		party_members => $raw->{party_members} || [],
		all_bots      => $raw->{all_bots} || [],
		raw         => $raw,
	};
}

sub _send_actor_delta_from_snapshot {
	my ($snapshot) = @_;
	return if ref($snapshot) ne 'HASH';

	my $actors = $snapshot->{actors};
	$actors = [] if ref($actors) ne 'ARRAY';
	my $snapshot_actor_count = scalar(@{$actors}) + 0;
	my $actor_discovery = {};
	if (ref($snapshot->{raw}) eq 'HASH' && ref($snapshot->{raw}{actor_discovery}) eq 'HASH') {
		$actor_discovery = $snapshot->{raw}{actor_discovery};
	} elsif (ref($snapshot->{actor_discovery}) eq 'HASH') {
		$actor_discovery = $snapshot->{actor_discovery};
	}

	my $map_name = '';
	if (ref($snapshot->{position}) eq 'HASH') {
		$map_name = _trim(_scalarize($snapshot->{position}{map}), 64);
	}

	my %observed_ids;
	my @observed;
	for my $actor (@{$actors}) {
		next if ref($actor) ne 'HASH';
		my $actor_id = _trim(_scalarize($actor->{actor_id}), 128);
		next if $actor_id eq '';
		next if $observed_ids{$actor_id};
		$observed_ids{$actor_id} = 1;

		push @observed, {
			actor_id => $actor_id,
			actor_type => _trim(_scalarize($actor->{actor_type} || 'unknown'), 64),
			name => _trim(_scalarize($actor->{name}), 128),
			map => _trim(_scalarize($actor->{map} || $map_name), 64),
			x => $actor->{x},
			y => $actor->{y},
			hp => $actor->{hp},
			hp_max => $actor->{hp_max},
			level => $actor->{level},
			relation => _trim(_scalarize($actor->{relation}), 64),
			dmg_to_us => $actor->{dmg_to_us},
			dmg_to_you => $actor->{dmg_to_you},
			dmg_from_us => $actor->{dmg_from_us},
			name_id => $actor->{name_id},
			casting => $actor->{casting},
			missed_you => $actor->{missed_you},
			cast_on_you => $actor->{cast_on_you},
			raw => {},
		};
	}

	my $known_before_count = scalar(keys %known_actor_ids) + 0;
	my $observed_count = scalar(@observed) + 0;
	if ($observed_count > 0) {
		$consecutive_empty_actor_snapshots = 0;
	} else {
		$consecutive_empty_actor_snapshots += 1;
	}

	if ($observed_count == 0 && $known_before_count > 0) {
		my $source_candidates = _actor_discovery_source_candidates($actor_discovery);
		my $empty_grace_snapshots = _cfg_int('aiSidecar_emptyActorRemovalGraceSnapshots', 2);
		$empty_grace_snapshots = 1 if !defined $empty_grace_snapshots || $empty_grace_snapshots < 1;
		my $within_grace = $consecutive_empty_actor_snapshots < $empty_grace_snapshots ? 1 : 0;
		my $has_source_candidates = $source_candidates > 0 ? 1 : 0;

		if ($within_grace || $has_source_candidates) {
			debug sprintf(
				"[aiSidecarBridge] actor delta empty-snapshot guard active observed=0 known=%d empty_streak=%d grace=%d source_candidates=%d; retaining previous actor-set state this tick\n",
				0 + $known_before_count,
				0 + $consecutive_empty_actor_snapshots,
				0 + $empty_grace_snapshots,
				0 + $source_candidates,
			), 'aiSidecarBridge', 2;

			_enqueue_normalized_event(
				'actor_state',
				'actor_state.bridge_delta_empty_guarded',
				'mainLoop_pre',
				'bridge actor delta skipped by empty-snapshot guard',
				{
					revision => _trim(_scalarize($snapshot->{tick_id} || _trace_id()), 128),
					observed_count => 0 + $observed_count,
					known_before_count => 0 + $known_before_count,
					empty_streak => 0 + $consecutive_empty_actor_snapshots,
					empty_grace_snapshots => 0 + $empty_grace_snapshots,
					source_candidates => 0 + $source_candidates,
					snapshot_actor_count => 0 + $snapshot_actor_count,
					actor_discovery => $actor_discovery,
				},
				{ outcome => 'guarded_empty' },
				{
					observed_count => 0 + $observed_count,
					known_before_count => 0 + $known_before_count,
					empty_streak => 0 + $consecutive_empty_actor_snapshots,
					source_candidates => 0 + $source_candidates,
					snapshot_actor_count => 0 + $snapshot_actor_count,
				},
				'info',
			);
			return;
		}
	}

	my @removed_actor_ids = grep { !$observed_ids{$_} } sort keys %known_actor_ids;
	my %actor_type_counts;
	my $hostile_count = 0;
	for my $row (@observed) {
		next if ref($row) ne 'HASH';
		my $actor_type = _trim(_scalarize($row->{actor_type}), 64);
		$actor_type = 'unknown' if !defined $actor_type || $actor_type eq '';
		$actor_type_counts{$actor_type} = 0 + ($actor_type_counts{$actor_type} || 0) + 1;

		my $relation = lc(_trim(_scalarize($row->{relation}), 64));
		if ($relation eq 'hostile' || $relation eq 'enemy' || $relation eq 'monster' || $actor_type eq 'monster') {
			$hostile_count += 1;
		}
	}
	my $removed_count = scalar(@removed_actor_ids) + 0;
	my $payload_counts = {
		snapshot_actor_count => 0 + $snapshot_actor_count,
		observed_count => 0 + $observed_count,
		removed_count => 0 + $removed_count,
		hostile_count => 0 + $hostile_count,
	};

	my $payload = {
		meta => ref($snapshot->{meta}) eq 'HASH' ? $snapshot->{meta} : _meta(_bot_id()),
		observed_at => _trim(_scalarize($snapshot->{observed_at} || _iso_now()), 64),
		revision => _trim(_scalarize($snapshot->{tick_id} || _trace_id()), 128),
		actors => \@observed,
		removed_actor_ids => \@removed_actor_ids,
	};

	my $resp = _http_post_json('/v2/ingest/actors', $payload);
	if (!$resp || $resp->{status} < 200 || $resp->{status} >= 300) {
		my $status = _http_status_code($resp);
		my $err = _http_error_text($resp);
		_throttled_warning(
			'v2_actors_failed',
			"[aiSidecarBridge] v2 actor push failed status=$status error=$err observed=$observed_count removed=$removed_count hostile=$hostile_count, retaining previous actor-set state.",
		);
		_emit_telemetry(
			'warning',
			'bridge',
			'v2_actor_delta_failed',
			'v2 actor delta push failed',
			{
				status => 0 + $status,
				observed_count => 0 + $observed_count,
				removed_count => 0 + $removed_count,
				hostile_count => 0 + $hostile_count,
				snapshot_actor_count => 0 + $snapshot_actor_count,
			},
			{ endpoint => '/v2/ingest/actors', error => $err },
		);
		_enqueue_normalized_event(
			'actor_state',
			'actor_state.bridge_delta_failed',
			'mainLoop_pre',
			'bridge actor delta push failed',
			{
				revision => $payload->{revision},
				status => 0 + $status,
				error => $err,
				observed_count => 0 + $observed_count,
				removed_count => 0 + $removed_count,
				hostile_count => 0 + $hostile_count,
				snapshot_actor_count => 0 + $snapshot_actor_count,
				actor_type_counts => \%actor_type_counts,
				payload_counts => $payload_counts,
				actor_discovery => $actor_discovery,
			},
			{},
			{
				observed_count => 0 + $observed_count,
				removed_count => 0 + $removed_count,
				hostile_count => 0 + $hostile_count,
				snapshot_actor_count => 0 + $snapshot_actor_count,
				status => 0 + $status,
			},
			'warning',
		);
		return;
	}

	my $response_json = (ref($resp) eq 'HASH' && ref($resp->{json}) eq 'HASH') ? $resp->{json} : {};
	my $accepted = int($response_json->{accepted} || 0);
	my $dropped = int($response_json->{dropped} || 0);
	my $message = _trim(_scalarize($response_json->{message} || ''), 220);
	my $outcome = ($observed_count == 0 && $removed_count == 0) ? 'none_visible' : 'delta_sent';
	_enqueue_normalized_event(
		'actor_state',
		'actor_state.bridge_delta_sent',
		'mainLoop_pre',
		'bridge actor delta sent',
		{
			revision => $payload->{revision},
			outcome => $outcome,
			observed_count => 0 + $observed_count,
			removed_count => 0 + $removed_count,
			hostile_count => 0 + $hostile_count,
			snapshot_actor_count => 0 + $snapshot_actor_count,
			accepted => 0 + $accepted,
			dropped => 0 + $dropped,
			message => $message,
			actor_type_counts => \%actor_type_counts,
			payload_counts => $payload_counts,
			actor_discovery => $actor_discovery,
		},
		{ outcome => $outcome },
		{
			observed_count => 0 + $observed_count,
			removed_count => 0 + $removed_count,
			hostile_count => 0 + $hostile_count,
			snapshot_actor_count => 0 + $snapshot_actor_count,
			accepted => 0 + $accepted,
			dropped => 0 + $dropped,
		},
		'info',
	);

	%known_actor_ids = %observed_ids;
}

sub _is_hash_like {
	my ($value) = @_;
	return 0 if !defined $value;
	my $kind = eval { reftype($value) };
	return defined($kind) && $kind eq 'HASH' ? 1 : 0;
}

sub _actor_id_from_any {
	my ($value) = @_;
	return '' if !defined $value;

	my $ref = ref($value);
	if (!$ref) {
		my $raw = "$value";
		if ($raw =~ /^\d+$/) {
			return _trim($raw, 64);
		}
		if (length($raw) == 4) {
			my $unpacked = unpack('V', $raw);
			return _trim(_scalarize($unpacked), 64);
		}
		return _trim($raw, 64);
	}

	return _trim(_scalarize($value), 64);
}

sub _actor_list_items {
	my ($list_obj) = @_;
	return () if !defined $list_obj;
	return () if !ref($list_obj);

	my $items = eval {
		# NOTE: never use bare 'return' inside eval BLOCK — it returns
		# from _actor_list_items itself, skipping the validation below.
		my $res;
		if ($list_obj->can('getItems')) {
			$res = $list_obj->getItems();
		}
		$res;
	};
	return () if !$items || ref($items) ne 'ARRAY';
	return @{$items};
}

sub _actor_discovery_source_candidates {
	my ($actor_discovery) = @_;
	return 0 if ref($actor_discovery) ne 'HASH';
	return 0 if ref($actor_discovery->{source_counts}) ne 'HASH';

	my $count = 0;
	for my $actor_type (qw(monster player npc)) {
		next if ref($actor_discovery->{source_counts}{$actor_type}) ne 'HASH';
		$count += 0 + ($actor_discovery->{source_counts}{$actor_type}{hash} || 0);
		$count += 0 + ($actor_discovery->{source_counts}{$actor_type}{list} || 0);
	}

	return 0 + $count;
}


sub _poll_next_action {
	return 1 if !_bridge_enabled();
	if (!$net || $net->getState() != Network::IN_GAME) {
		return 1 if !_cfg_bool('aiSidecar_pollWhenDisconnected', 1);
	}

	# Force-set sitAuto_hp_lower=0 every cycle — OpenKore's AI re-enables it
	# sitAuto NOT disabled — heuristic controls it per RULE.md
	# Force-set attackAuto=0 when 0 potions on hunting map
	# Also force-set attackAuto_inLockOnly=0 and attackAuto_routeToLock=0
	# because OpenKore's getAttackAutoModeForContext returns 1 when
	# attackAuto_inLockOnly==1 regardless of attackAuto value.
	if ($char && @{_char_inventory($char)}) {
		my $_pa_has_potions = 0;
		for my $_pai (@{_char_inventory($char)}) {
			next unless $_pai;
			my $_pan = $_pai->{name} || '';
			if ($_pan =~ /potion|herb|fruit|berry|red|orange|white|yellow|blue|green/i) {
				$_pa_has_potions = 1;
				last;
			}
		}
				if (!$_pa_has_potions) {
					my $_pm = $field ? lc($field->name()) : '';
					$_pm =~ s/\.gat$//;
					if ($_pm =~ /_fild|_dun/i) {
						# attackAuto NOT overridden — heuristic controls it per RULE.md
					}
				} elsif ($::config{'route_randomWalk'} == 0) {
					# Has potions or not on hunting map — restore route_randomWalk
					$::config{'route_randomWalk'} = 1;
				}
			}

	my $poll_id = _trace_id();

	# ── FOLLOWUP EXECUTION (self-aware multi-step actions) ──
	# Some actions carry a followup in metadata (e.g. job_change = move THEN
	# talknpc). The bridge processes ONE action per poll, so a followup is
	# stored here and executed on the NEXT poll before fetching a new action.
	my $_followup = $::aiSidecar_followup{_bot_id()};
	if ($_followup) {
		delete $::aiSidecar_followup{_bot_id()};
		my ($_fup_cmd, $_fup_meta, $_fup_action_id) = @{$_followup};
		debug "[aiSidecarBridge] executing followup command: $_fup_cmd (action $_fup_action_id)\n", 'aiSidecarBridge', 1;
		my ($_fs, $_fr, $_fm) = _execute_action($_fup_action_id, {
			action_id => $_fup_action_id,
			kind => 'command',
			command => $_fup_cmd,
			metadata => $_fup_meta,
		});
		push @ack_queue, {
			queued_at => _now_ms(),
			action_id => $_fup_action_id,
			poll_id => $poll_id,
			success => $_fs,
			result_code => $_fr,
			message => $_fm,
			observed_latency_ms => 0,
			kind => 'command',
		};
		return;
	}

	my $resp = _http_post_json('/v1/actions/next', {
		meta => _meta(_bot_id()),
		poll_id => $poll_id,
		max_actions => $MAX_ACTIONS_PER_POLL,
	});

	my $status = _http_status_code($resp);
	if ($status < 200 || $status >= 300) {
		$consecutive_poll_failures += 1;
		my $backoff_ms = _poll_failure_delay_ms();
		my $reset_after = _cfg_int('aiSidecar_pollFailureResetRegistrationAfter', 3);
		$registered = 0 if $reset_after > 0 && $consecutive_poll_failures >= $reset_after;

		my $err = _http_error_text($resp);
		_throttled_warning(
			'poll_failed',
			"[aiSidecarBridge] action poll failed status=$status error=$err failures=$consecutive_poll_failures next_retry_ms=$backoff_ms (fail-open retained).",
		);
		_emit_telemetry(
			'warning',
			'bridge',
			'poll_failed',
			'action poll failed',
			{
				status => 0 + $status,
				consecutive_failures => 0 + $consecutive_poll_failures,
				next_retry_ms => 0 + $backoff_ms,
			},
			{ endpoint => '/v1/actions/next', error => $err },
		);
		return 0;
	}

	if ($consecutive_poll_failures > 0) {
		debug "[aiSidecarBridge] poll recovered after $consecutive_poll_failures consecutive failures\n", 'aiSidecarBridge', 2;
	}
	$consecutive_poll_failures = 0;

	my $json = $resp->{json};
	return 1 if ref($json) ne 'HASH';
	return 1 if !$json->{has_action};

	# ── Multi-action per poll ──
	# Accept either a single action (backward compat) or an array of actions
	my @actions = ();
	if (ref($json->{actions}) eq 'ARRAY') {
		@actions = @{$json->{actions}};
	} elsif (ref($json->{action}) eq 'HASH') {
		@actions = ($json->{action});
	}

	# Limit to max actions per poll
	if (@actions > $MAX_ACTIONS_PER_POLL) {
		splice @actions, $MAX_ACTIONS_PER_POLL;
	}

				# ── Execute ALL actions in sequence with proper delays ──
				# FIX: Collect results and send ACKs AFTER all actions are done,
				# so the sidecar doesn't interpret individual ACKs as batch completion.
				my $executed_count = 0;
				my @action_results = ();
				for my $action (@actions) {
					last if ref($action) ne 'HASH';
					my $result = _execute_action($poll_id, $action);
					push @action_results, $result if $result;
					$executed_count++;

					# Check if we need to wait for cast/animation before next action
					if ($_is_casting && $_casting_until_ms > _now_ms()) {
						my $wait_ms = $_casting_until_ms - _now_ms();
						$wait_ms = 0 if $wait_ms < 0;
						$wait_ms = 5000 if $wait_ms > 5000;  # Safety cap
						usleep($wait_ms * 1000) if $wait_ms > 0;
					} elsif ($executed_count < @actions) {
						# Small inter-action delay to let server process
						usleep(50000);  # 50ms between actions
					}
				}

	# ── Send ACKs for ALL executed actions in batch ──
	# This prevents the sidecar from treating individual ACKs as batch completion.
	# The sidecar expects ACKs after the batch is fully processed.
	for my $_ar (@action_results) {
		next if !$_ar;
		my $_ar_id = $_ar->{action_id} || '';
		next if $_ar_id eq '' || $_ar_id eq 'unknown_action';
		_http_post_json('/v1/acknowledgements/action', {
			meta => _meta(_bot_id()),
			action_id => $_ar_id,
			poll_id => $poll_id,
			success => $_ar->{success} ? JSON::PP::true() : JSON::PP::false(),
			result_code => $_ar->{result_code} || 'unknown',
			message => $_ar->{message} || '',
		});
	}

	# ── Report completed batches to sidecar ──
	_flush_completed_batches();

	# ── Check WOE status periodically ──
	_check_woe_status();

	# ── Check MVP status periodically ──
	_check_mvp_status();

	# ── Check party buff expiry ──
	_check_party_buff_expiry();

	# ── Scan NPC shops periodically ──
	_scan_npc_shops();

	# ── Scan player vendors periodically ──
	_scan_player_vendors();

	# ── Report party member positions periodically ──
	_report_party_positions();

	# ── Report game time periodically ──
	_report_game_time();

	# ── Flush server announcements periodically ──
	_flush_announcements();

	# ── Detect dispel effects periodically ──
	_detect_dispel();

	# ── Discover NPC shops periodically ──
	_discover_shops();

	# ── Discover portals periodically ──
	_discover_portals();

	return 1;
}

sub _execute_action {
	my ($poll_id, $action) = @_;

	my $action_id = $action->{action_id} || 'unknown_action';
	my $kind = lc($action->{kind} || 'command');
	my $command = defined $action->{command} ? $action->{command} : '';
	my $metadata = ref($action->{metadata}) eq 'HASH' ? $action->{metadata} : {};
	my $started = _now_ms();
	# Anti-detection: random delay before executing action (per-bot profile)
	if ($ANTI_DETECTION_ENABLED) {
		my $delay_ms = _human_cmd_delay_ms();
		usleep($delay_ms * 1000) if $delay_ms > 0;
	}
	
	my ($effective_command, $rewrite_kind) = _rewrite_runtime_command($command, $metadata, $action_id);

	# ── LOGGED-OUT GATE (hoisted to top of dispatch) ──
	# Previously an elsif at the END of the dispatch chain, so early
	# execution branches (auto-stand ~3245, party-leave ~3278, escape
	# ~3322) ran Commands::run BEFORE the gate and fired in-game commands
	# against a logged-out session — "You must be logged in" spam with no
	# [ai_action] trace. Hoisting here short-circuits any effective command
	# that isn't in the char_select/char_create/conf reconnect family
	# before ANY internal branch runs. The elsif at the chain end remains
	# as defense-in-depth.
	if (_logged_out_execution_block($effective_command)) {
		$rewrite_kind = 'logged_out_gate';
		$effective_command = '';
		debug "[logged_out_gate_hoisted] blocked '$command' while bot not in game\n", 'aiSidecarBridge', 1;
	}

	# ── AI action log — proves decisions reach OpenKore
	if ($effective_command ne '' && $rewrite_kind ne 'committed_action_blocked' && $rewrite_kind ne 'empty_command') {
		warning "[ai_action] id=$action_id kind=$kind cmd=$command effective=$effective_command rewrite=$rewrite_kind\n", 'aiSidecarBridge', 1;
	}

	my $tick_latency_ms = 200;
	# LATENCY-ADAPTIVE TIMING: ConnectionMetrics details unavailable in this build
	$tick_latency_ms = ($tick_latency_ms < 50) ? 50 : ($tick_latency_ms > 1000) ? 1000 : $tick_latency_ms;
	my $tick_buffer = int($tick_latency_ms / 150) + 1;


	# ── ACTION BATCH COORDINATION ──
	# The sidecar can send up to 5 actions per poll. We track which
	# batch this belongs to and log batch completion.
	if ($action_id ne 'unknown_action' && $action_id =~ /^(\d+)_/) {
		my $batch_id = $1;
		$_pending_batch_actions{$batch_id} ||= [];
		push @{$_pending_batch_actions{$batch_id}}, $effective_command;
		# Log batch progress
		warning "[ai_batch] batch=$batch_id action=$action_id cmd=$effective_command\n", 'aiSidecarBridge', 2;
	}

	# ── COMMITTED ACTION TRACKING ──
	# Wire up _committed_actions: track committed actions to prevent conflicts
	# within the COMMITTED_ACTION_COOLDOWN_MS window.
	# _committed_actions is declared at line 110 but was never written to.
	# Now we track every executed action so _rewrite_runtime_command can check it.
	if ($action_id ne 'unknown_action' && $effective_command ne '') {
		my $now = _now_ms();
		# Clean stale entries
		for my $_ca_key (keys %_committed_actions) {
			if ($now - $_committed_actions{$_ca_key} > $COMMITTED_ACTION_COOLDOWN_MS) {
				delete $_committed_actions{$_ca_key};
			}
		}
		for my $_cc_key (keys %_committed_commands) {
			if ($now - $_committed_commands{$_cc_key} > $COMMITTED_ACTION_COOLDOWN_MS) {
				delete $_committed_commands{$_cc_key};
			}
		}
		# Track this action (by action_id AND normalized command text so the
		# rewrite can suppress exact duplicates of one-shot commands)
		$_committed_actions{$action_id} = $now;
		my $_cmd_key = lc($effective_command);
		$_cmd_key =~ s/\s+/ /g;
		$_cmd_key =~ s/^\s+|\s+$//g;
		$_committed_commands{$_cmd_key} = $now if $_cmd_key ne '';
		debug "[committed_action] tracked action_id=$action_id cmd=$effective_command\n", 'aiSidecarBridge', 2;
	}

	# ── SKILL DELAY CHECK ──
	# Check if the command involves a skill and if that skill is on cooldown.
	my ($success, $result_code, $msg) = (0, 'invalid_action', 'invalid action payload');
	my $_skill_name = '';
	if ($effective_command =~ /^(?:use_skill|skill)\s+(.+)$/i) {
		$_skill_name = lc($1);
		$_skill_name =~ s/\s+$//;
	} elsif ($effective_command =~ /^attack\s+skill\s+(.+)$/i) {
		$_skill_name = lc($1);
		$_skill_name =~ s/\s+$//;
	}
	if ($_skill_name ne '') {
		my $now = _now_ms();
		if (defined $_skill_delays{$_skill_name} && $now < $_skill_delays{$_skill_name}) {
			my $remaining_ms = $_skill_delays{$_skill_name} - $now;
			($success, $result_code, $msg) = (1, 'skill_on_cooldown', "skill '$_skill_name' on cooldown for ${remaining_ms}ms");
			$rewrite_kind = 'skill_on_cooldown';
			$effective_command = '';
			warning "[skill_delay] skill='$_skill_name' on cooldown, remaining=${remaining_ms}ms, skipping\n", 'aiSidecarBridge', 1;
		} else {
			# Skill is available — track its delay after execution
			# The delay values come from the action metadata or defaults
			my $skill_delay_ms = defined $metadata->{skill_delay_ms} ? $metadata->{skill_delay_ms} : 0;
			my $cast_time_ms = defined $metadata->{cast_time_ms} ? $metadata->{cast_time_ms} : 0;
			my $after_cast_delay_ms = defined $metadata->{after_cast_delay_ms} ? $metadata->{after_cast_delay_ms} : 0;

			# Update skill delay tracking
			if ($skill_delay_ms > 0) {
				$_skill_delays{$_skill_name} = $now + $skill_delay_ms;
				$_last_skill_use_ms{$_skill_name} = $now;
				debug "[skill_delay] set delay for '$_skill_name': ${skill_delay_ms}ms (until $_skill_delays{$_skill_name})\n", 'aiSidecarBridge', 2;
			}
			if ($cast_time_ms > 0) {
				$_cast_times{$_skill_name} = $cast_time_ms;
				$_is_casting = 1;
				$_casting_until_ms = $now + $cast_time_ms;
				debug "[skill_delay] set cast time for '$_skill_name': ${cast_time_ms}ms (until $_casting_until_ms)\n", 'aiSidecarBridge', 2;
			}
			if ($after_cast_delay_ms > 0) {
				$_after_cast_delays{$_skill_name} = $after_cast_delay_ms;
			}
		}
	}

	# ── CAST TIME AWARENESS: don't send movement commands during cast ──
	if ($_is_casting && $_casting_until_ms > _now_ms() && $effective_command =~ /^move\s+/i) {
		$rewrite_kind = 'movement_blocked_during_cast';
		$effective_command = '';
		debug "[cast_aware] blocking move command during cast (until $_casting_until_ms)\n", 'aiSidecarBridge', 2;
	}

	# ── PARTY LEAVE SUPPRESSION: one-time leave, then suppress forever ──
	# The sidecar may queue 'stand' and 'move prontera' as separate actions.
	# The bridge only processes ONE action per poll, so 'move prontera' arrives
	# while the bot is still sitting. Auto-stand ensures the bot can actually move.
	# Also disable sitAuto temporarily to prevent immediate re-sit (especially at low HP).
	if ($char && $char->{sitting} && $effective_command =~ /^move\s+/i) {
		# sitAuto NOT disabled — heuristic controls it per RULE.md
		Commands::run("stand");
		debug "[auto_stand] bot was sitting, auto-stand before move command\n", 'aiSidecarBridge', 1;
	}

	# ── PARTY LEAVE SUPPRESSION: one-time leave, then suppress forever ──
	# Once a bot has left a party, it should never try again unless it somehow rejoins.
	# The bridge tracks this independently of the sidecar's cooldown.
	# Uses a persistent file-based state to survive restarts.
	# NOTE: the latch is bypassed while the bot is ACTUALLY in a party, so
	# stale-party cleanup after a rejoin still works (one-way latches that
	# never reset would otherwise suppress every future 'party leave').
	if (lc($command || '') eq 'party leave') {
		# Read persistent state from file
		my $_pl_file = _party_leave_state_file();
		my $_pl_actually_in_party = (defined($char) && defined($char->{party})) ? 1 : 0;
		my $_has_left_party = 0;
		if (-e $_pl_file) {
			open my $_pl_fh, '<', $_pl_file or do { debug "[party_leave] cannot read $_pl_file: $!\n", 'aiSidecarBridge', 1 };
			if ($_pl_fh) {
				my $_pl_content = <$_pl_fh>;
				chomp $_pl_content;
				$_has_left_party = ($_pl_content eq '1') ? 1 : 0;
				close $_pl_fh;
			}
		}
		if ($_has_left_party && !$_pl_actually_in_party) {
			($success, $result_code, $msg) = (1, 'ok', 'party_leave_already_left');
			$rewrite_kind = 'party_leave_already_left';
			$effective_command = '';
		} else {
			# Execute the party leave command
			my $ok = eval { Commands::run('party leave'); 1; };
			if ($ok) {
				$_has_left_party = 1;
				($success, $result_code, $msg) = (1, 'ok', 'party_leave_executed');
			} else {
				# Command failed (not in party) — still mark as left to prevent retry
				$_has_left_party = 1;
				($success, $result_code, $msg) = (1, 'ok', 'party_leave_not_in_party');
			}
			$rewrite_kind = 'party_leave';
			$effective_command = '';
			# Write persistent state to file
			open my $_pl_fh, '>', $_pl_file or do { debug "[party_leave] cannot write $_pl_file: $!\n", 'aiSidecarBridge', 1 };
			if ($_pl_fh) {
				print $_pl_fh "1\n";
				close $_pl_fh;
			}
		}
	}

	# ── AI AUTO SUPPRESSION: if already in auto mode, skip 'ai auto' commands ──
	# Prevents "AI is already set to auto mode" spam from sidecar pushing ai auto every cycle
	if (($rewrite_kind eq 'ai_auto_already_auto' || lc($command || '') eq 'ai auto') && defined $AI::AI && $AI::AI == 2) {
		($success, $result_code, $msg) = (1, 'ok', 'ai_auto_already_auto');
		$rewrite_kind = 'ai_auto_already_auto';
		$effective_command = '';
	}

	# ── ITEM 602 SUPPRESSION: block 'use 602' / 'use Butterfly Wing' at execution level ──
	# The items_control.txt fix only works at startup. This catches it at runtime.
	# Item 602 (Butterfly Wing) is managed by the AI system, not auto-use.
	# Return success=1 so the sidecar doesn't retry the command every cycle.
	if (lc($command || '') =~ /^(?:use\s+)?602$/ || lc($command || '') =~ /^use\s+butterfly\s+wing/i) {
		($success, $result_code, $msg) = (1, 'ok', 'item_602_suppressed');
		$rewrite_kind = 'item_602_suppressed';
		$effective_command = '';
	}

	# ── ESCAPE COMMAND: sidecar pushes "escape" → fire escape reflex ──
	# When the sidecar tells the bot to escape (teleport or flee),
	# immediately execute and fire the escape reflex event.
	if (lc($command || '') =~ /^(?:escape|teleport|flee)\s*$/i) {
		my $_escape_now = _now_ms();
		# Execute the escape command
		my $_escape_ok = eval { Commands::run($effective_command); 1; };
		($success, $result_code, $msg) = (1, 'ok', 'escape_command_executed');

		# Post escape event to sidecar
		_post_event({
			kind => 'bridge_reflex',
			reflex => 'escape',
			severity => 'warning',
			text => "escape reflex triggered by sidecar command: $command",
			command => $command,
			run_ok => $_escape_ok ? '1' : '0',
			hp => ($char ? $char->{hp} : 0) || 0,
			hp_max => ($char ? $char->{hp_max} : 1) || 1,
			map => _safe_field_map() || '',
		});
		debug "[aiSidecarBridge] escape_command: executing '$command' ok=$_escape_ok\n", 'aiSidecarBridge', 1;

		# Force AI to manual to prevent re-engagement
		if (defined $AI::AI && $AI::AI == 2) {
			eval { require AI; AI::state(1); 1; };
		}

		# Set survival mode cooldown
		$_survival_mode_until_ms = $_escape_now + 60000 if $_survival_mode_until_ms < $_escape_now;

		$rewrite_kind = 'escape_command';
		$effective_command = '';
	}

	# ── Apply ML overrides (source="ml" actions carry learned recommendations) ──
	my $action_source = lc($action->{source} || '');
	if ($action_source eq 'ml' && defined $metadata->{ml_override}) {
		_apply_ml_override($metadata->{ml_override});
		($success, $result_code, $msg) = (1, 'ok', 'ml override applied');
	} elsif ($kind eq 'macro_reload') {
		($success, $result_code, $msg) = _execute_macro_reload_action($metadata);
	} elsif ($kind eq 'config_reload') {
		($success, $result_code, $msg) = _execute_config_reload_action($metadata);
	} elsif ($kind ne 'command') {
		($success, $result_code, $msg) = (0, 'unsupported_kind', "unsupported action kind '$kind'");
	} elsif ($rewrite_kind eq 'bare_take_delegated') {
		($success, $result_code, $msg) = (1, 'ok', 'loot pickup delegated to OpenKore auto-loot configuration');
	} elsif ($rewrite_kind eq 'random_walk_seek_already_auto' || $rewrite_kind eq 'bare_move_already_auto' || $rewrite_kind eq 'map_move_already_auto' || $rewrite_kind eq 'teleport_already_auto') {
		($success, $result_code, $msg) = (1, 'ok', 'movement runtime command is already satisfied (AI already in auto mode)');
	} elsif ($rewrite_kind eq 'map_move_toggle_manual') {
		# Don't toggle to manual if critically low HP — let auto-AI handle survival
		my $_ai_hp = $main::char ? $main::char->{hp} : 9999;
		my $_ai_hp_max = $main::char ? $main::char->{hp_max} : 1;
		if ($_ai_hp_max > 0 && ($_ai_hp / $_ai_hp_max) < 0.50) {
			($success, $result_code, $msg) = (1, 'ok', 'ai_manual_suppressed_low_hp');
		} else {
			my $ok = eval { _toggle_ai_mode('manual'); 1; };
			($success, $result_code, $msg) = $ok ? (1, 'ok', 'ai toggled to manual to force route recalculation') : (0, 'dispatch_error', $@);
		}
	} elsif ($rewrite_kind eq 'coordinate_move_raw') {
		# ── pos_to DESYNC RESYNC (agnostic, data-driven, 2026-08-25) ──
		# Task::Route short-circuits to "reached the destination" with ZERO walk
		# packets when $char->{pos_to} already equals the dest (Route.pm:286). If
		# a prior partial walk set pos_to = target but the server never moved us
		# (e.g. the old dist<5 route-loop suppression armed pos_to without a send),
		# every subsequent move to that target "arrives" instantly without walking
		# -> the bot never steps onto the warp tile. When the char's real pos
		# differs from pos_to, reset pos_to = pos so the route recomputes from the
		# ACTUAL position and actually sends the walk. Uses the char's own
		# position FACT, no hardcoded coordinate.
		if ($char && $char->{pos} && $char->{pos_to}
			&& ref $char->{pos} eq 'HASH' && ref $char->{pos_to} eq 'HASH'
			&& (($char->{pos}{x} || 0) != ($char->{pos_to}{x} || 0)
				|| ($char->{pos}{y} || 0) != ($char->{pos_to}{y} || 0))) {
			my $_old = "(" . ($char->{pos_to}{x} || 0) . "," . ($char->{pos_to}{y} || 0) . ")";
			%{$char->{pos_to}} = %{$char->{pos}};
			$char->{solution} = [];
			debug "[move_resync] pos_to desync ($_old -> ($char->{pos}{x},$char->{pos}{y})) reset before coordinate move\n", 'aiSidecarBridge', 1;
		}
		my $ok = eval { Commands::run($effective_command); 1; };
		($success, $result_code, $msg) = $ok ? (1, 'ok', 'coordinate move executed') : (0, 'dispatch_error', $@);
	} elsif ($rewrite_kind eq 'chat_sent') {
		($success, $result_code, $msg) = (1, 'ok', 'chat message sent');
	} elsif ($rewrite_kind eq 'go_command_sent') {
		($success, $result_code, $msg) = (1, 'ok', '@go command sent as chat');
	} elsif ($rewrite_kind =~ /^use_item_/) {
		($success, $result_code, $msg) = (1, 'ok', "item use handled: $rewrite_kind");
	} elsif ($rewrite_kind eq 'skills_add_rewritten') {
		($success, $result_code, $msg) = (1, 'ok', 'skills_add rewritten to skills add');
	} elsif ($rewrite_kind eq 'attack_skill_basic_attack_ignored') {
		($success, $result_code, $msg) = (1, 'ok', 'basic attack ignored (auto-attack handles this)');
	} elsif ($rewrite_kind eq 'attack_skill_delegated') {
		($success, $result_code, $msg) = (1, 'ok', 'attack skill delegated to auto-AI');
	} elsif ($rewrite_kind eq 'ai_manual_already_manual' || $rewrite_kind eq 'ai_auto_already_auto') {
		($success, $result_code, $msg) = (1, 'ok', "AI mode already satisfied: $rewrite_kind");
	} elsif ($rewrite_kind eq 'map_move_already_set') {
		($success, $result_code, $msg) = (1, 'ok', 'lockMap already set to target');
	} elsif ($rewrite_kind eq 'stale_npc_blocked') {
		($success, $result_code, $msg) = (1, 'ok', 'blocked: stale NPC teleport');
	} elsif ($rewrite_kind eq 'ai_manual_to_sit') {
		my $ok = eval { Commands::run('sit'); 1; };
		($success, $result_code, $msg) = $ok ? (1, 'ok', 'ai manual rewritten to sit') : (0, 'dispatch_error', $@);
	} elsif ($rewrite_kind eq 'ai_manual_suppressed') {
		($success, $result_code, $msg) = (1, 'ok', 'ai_manual_suppressed');
	} elsif ($rewrite_kind eq 'sit_blocked_on_hunting_map') {
		($success, $result_code, $msg) = (1, 'ok', 'sit_blocked_on_hunting_map');
	} elsif ($rewrite_kind eq 'ai_manual_allowed') {
		    my $ok = eval { Commands::run($effective_command); 1; };
		    ($success, $result_code, $msg) = $ok ? (1, 'ok', 'ai_manual') : (0, 'dispatch_error', $@);
		} elsif ($rewrite_kind eq 'ai_manual_throttled') {
		($success, $result_code, $msg) = (1, 'ok', 'ai manual throttled (30s cooldown)');
	} elsif ($rewrite_kind eq 'macro_potion_cooldown') {
		($success, $result_code, $msg) = (1, 'ok', 'blocked: potion cooldown');
	} elsif ($rewrite_kind eq 'committed_action_blocked') {
		($success, $result_code, $msg) = (1, 'ok', 'blocked: conflicting action within cooldown');
	} elsif ($rewrite_kind eq 'kafra_teleport_auto') {
		($success, $result_code, $msg) = (1, 'ok', 'kafra teleport sequence auto-completed');
	} elsif ($rewrite_kind eq 'tool_dealer_auto') {
		($success, $result_code, $msg) = (1, 'ok', 'tool dealer sequence auto-completed');
	} elsif ($rewrite_kind eq 'dialog_guard_blocked') {
		($success, $result_code, $msg) = (1, 'ok', 'blocked: bot is in NPC dialog');
	} elsif ($rewrite_kind eq 'combat_guard_blocked') {
		($success, $result_code, $msg) = (1, 'ok', 'blocked: bot is in combat');
	} elsif ($rewrite_kind eq 'sit_blocked_town') {
		($success, $result_code, $msg) = (1, 'ok', 'sit blocked in town');
	} elsif ($rewrite_kind eq 'sit_blocked_hunting') {
		($success, $result_code, $msg) = (1, 'ok', 'sit blocked: on hunting map');
	} elsif ($rewrite_kind eq 'map_move_low_hp_no_toggle') {
		($success, $result_code, $msg) = (1, 'ok', 'ai_manual suppressed: critically low HP');
	} elsif ($rewrite_kind eq 'config_set_ok') {
		($success, $result_code, $msg) = (1, 'ok', 'config_set_ok');
	} elsif ($rewrite_kind eq 'party_leave' || $rewrite_kind eq 'party_leave_already_left' || $rewrite_kind eq 'party_leave_not_in_party' || $rewrite_kind eq 'party_leave_executed') {
		($success, $result_code, $msg) = (1, 'ok', $rewrite_kind);
	} elsif ($rewrite_kind =~ /^use_item_not_found_/) {
		($success, $result_code, $msg) = (1, 'ok', "item not found in inventory: $rewrite_kind");
	} elsif ($rewrite_kind eq 'char_create') {
		# Parse: char_create <slot> "<name>" [<str> <agi> <vit> <int> <dex> <luk>]
		# Only slot and name are required — OpenKore's createCharacter handles the rest
		# IMPORTANT: Only execute when at character select screen (state 3)
		my $_cc_state = $net ? $net->getState() : 0;
		if ($_cc_state == 3) {
			my ($_cc_slot, $_cc_name) =
				($effective_command =~ /^char_create\s+(\d+)\s+"([^"]+)"/i);
			if (defined $_cc_slot && defined $_cc_name && $_cc_name ne '') {
				debug "[char_create] Creating character '$_cc_name' in slot $_cc_slot ...\n", 'aiSidecarBridge', 1;
				my $_cc_ok = 0;
				my $_cc_err = '';
				eval {
					$_cc_ok = Misc::createCharacter($_cc_slot, $_cc_name);
					1;
				} or $_cc_err = $@ || 'eval failed';
				if ($_cc_ok) {
					($success, $result_code, $msg) = (1, 'ok', "character '$_cc_name' created in slot $_cc_slot");
					warning "[char_create] SUCCESS: '$_cc_name' in slot $_cc_slot\n", 'aiSidecarBridge', 1;
					# Auto-enter the game: set config and send char login
					# char_select command only works IN-GAME, so we must do this here
					eval { Commands::run("conf char $_cc_slot"); 1; };
					eval { $messageSender->sendCharLogin($_cc_slot); 1; };
					warning "[char_create] Auto-entering game with slot $_cc_slot\n", 'aiSidecarBridge', 1;
				} else {
					my $_cc_err = $@ || 'unknown error';
					($success, $result_code, $msg) = (0, 'char_create_failed', $_cc_err);
					warning "[char_create] FAILED for '$_cc_name' slot $_cc_slot: $_cc_err\n", 'aiSidecarBridge', 1;
				}
			} else {
				($success, $result_code, $msg) = (0, 'char_create_invalid', "invalid char_create format: $effective_command");
			}
		} elsif ($_cc_state >= 5) {
			# Already in-game — char_create would fail, skip silently
			($success, $result_code, $msg) = (1, 'ok', 'char_create skipped: already in-game');
		} else {
			# Not at character select yet — defer: don't execute, return as-is so bridge retries
			debug "[char_create] Deferred: bot state=$_cc_state (need 3 for char select), will retry on next poll\n", 'aiSidecarBridge', 1;
			# Return success without executing — action stays in queue for retry
			($success, $result_code, $msg) = (1, 'ok', "char_create deferred: state=$_cc_state");
		}
	} elsif ($rewrite_kind eq 'char_select_handled') {
		($success, $result_code, $msg) = (1, 'ok', 'char_select handled by char_create auto-enter');
	} elsif ($effective_command eq '') {
		($success, $result_code, $msg) = (0, 'empty_command', 'empty command');
	} elsif (length($effective_command) > _cfg_int('aiSidecar_maxCommandLength', 160)) {
		($success, $result_code, $msg) = (0, 'command_too_long', 'command length exceeds policy');
	} elsif (!_command_allowed($effective_command)) {
		($success, $result_code, $msg) = (0, 'policy_rejected', 'command rejected by bridge policy');
	} elsif (_logged_out_execution_block($effective_command)) {
		# ── LOGGED-OUT GATE: never execute in-game commands while the bot is
		# disconnected / at char-select. Executing them produces
		# "You must be logged in the game to use this command" error spam.
		# char_select/char_create/reconnect actions are the ONLY ones allowed
		# through while logged out — they are what bring the bot back in-game.
		($success, $result_code, $msg) = (1, 'ok', 'skipped: bot not in game (logged-out gate)');
	} else {
		my $ok = eval { Commands::run($effective_command); 1; };
		if ($ok) {
			if ($rewrite_kind ne '') {
				($success, $result_code, $msg) = (1, 'ok', "command rewritten to '$effective_command' for runtime compatibility");
			} else {
				($success, $result_code, $msg) = (1, 'ok', 'command dispatched through OpenKore console pathway');
			}
		} else {
			my $err = $@ || 'command execution failure';
			($success, $result_code, $msg) = (0, 'dispatch_error', _trim($err, 220));
		}
	}

	my $latency_ms = _now_ms() - $started;
	my $event_name = $kind eq 'macro_reload' ? 'macro_reload_executed' : $kind eq 'config_reload' ? 'config_reload_executed' : 'action_executed';
	my $category = $kind eq 'macro_reload' ? 'macro' : $kind eq 'config_reload' ? 'config' : 'action';

	# ── STORE FOLLOWUP for the next poll (multi-step self-aware actions) ──
	# e.g. job_change: metadata.followup_command = "talknpc <x> <y>" — executed
	# after the move lands. Stored only on SUCCESS (a failed move means the bot
	# is not at the target; retry the primary command instead).
	if ($success && ref($metadata) eq 'HASH' && $metadata->{followup_command}) {
		my $_fup_cmd = $metadata->{followup_command};
		debug "[aiSidecarBridge] storing followup for action $action_id: $_fup_cmd\n", 'aiSidecarBridge', 1;
		$::aiSidecar_followup{_bot_id()} = [$_fup_cmd, $metadata, $action_id];
	}

	push @ack_queue, {
		queued_at => _now_ms(),
		action_id => $action_id,
		poll_id => $poll_id,
		success => $success,
		result_code => $result_code,
		message => $msg,
		observed_latency_ms => $latency_ms,
		kind => $kind,
	};

		_emit_telemetry(
			$success ? 'info' : 'warning',
			$category,
			$event_name,
			$msg,
			{ observed_latency_ms => $latency_ms + 0 },
			{ result_code => $result_code, kind => $kind },
		);
	
		# ── ACKNOWLEDGE ACTION: tell sidecar this action is done ──
		# Must be called after execution so the action queue removes this item.
		# Without ack, the queue holds it in 'dispatched' state and blocks new actions.
		if ($action_id && $action_id ne 'unknown_action') {
		    _http_post_json('/v1/acknowledgements/action', {
		        meta => _meta(_bot_id()),
		        action_id => $action_id,
		        poll_id => $poll_id,
		        success => $success ? JSON::PP::true() : JSON::PP::false(),
		        result_code => $result_code,
		        message => $msg,
		    });
		}

	# ── BATCH COMPLETION: wire up _pending_batch_actions ──
	# _pending_batch_actions was write-only dead code — actions were pushed
	# into it but never consumed. Now we track completed batches and report
	# them to the sidecar via _flush_completed_batches.
	if ($action_id ne 'unknown_action' && $action_id =~ /^(\d+)_/) {
		my $batch_id = $1;
		if (defined $_pending_batch_actions{$batch_id}) {
			my $batch_size = scalar(@{$_pending_batch_actions{$batch_id}});
			# Check if this is the last action in the batch by looking at
			# the action_id suffix (e.g., "123_5" for 5th action in batch 123)
			my ($batch_seq) = ($action_id =~ /^\d+_(\d+)$/);
			if (defined $batch_seq && $batch_seq >= $batch_size) {
				# Batch is complete — mark for reporting
				$_completed_batches{$batch_id} = {
					batch_id => $batch_id,
					action_count => $batch_size,
					completed_at => _now_ms(),
					actions => $_pending_batch_actions{$batch_id},
				};
				delete $_pending_batch_actions{$batch_id};
				warning "[ai_batch] batch=$batch_id complete ($batch_size actions)\n", 'aiSidecarBridge', 1;
			}
		}
	}
}

# ── Flush completed batches to sidecar ──
# Reports completed batch actions so the sidecar knows the batch is done.
# This wires up _pending_batch_actions (previously write-only dead code).
sub _flush_completed_batches {
	return if !_bridge_enabled();
	return if !%_completed_batches;

	for my $_cb_id (keys %_completed_batches) {
		my $_cb = $_completed_batches{$_cb_id};
		my $_resp = _http_post_json('/v1/acknowledgements/batch', {
			meta => _meta(_bot_id()),
			batch_id => $_cb->{batch_id},
			action_count => $_cb->{action_count},
			completed_at => $_cb->{completed_at},
			actions => $_cb->{actions},
		});
		if ($_resp && $_resp->{status} >= 200 && $_resp->{status} < 300) {
			delete $_completed_batches{$_cb_id};
			debug "[ai_batch] reported batch=$_cb_id complete to sidecar\n", 'aiSidecarBridge', 2;
		} else {
			debug "[ai_batch] failed to report batch=$_cb_id, will retry\n", 'aiSidecarBridge', 1;
		}
	}
}

# ── Party coordination: send buff request to other bots ──
# Sends a buff request to the sidecar, which forwards it to the appropriate bot.
# The sidecar handles routing — the bridge just posts the request.
sub _send_party_buff_request {
	my ($buff_name, $target_bot) = @_;
	return if !_bridge_enabled();
	return if !$registered;
	return if !$buff_name;

	my $now = _now_ms();
	return if $now - $_last_party_buff_request_ms < $PARTY_BUFF_REQUEST_COOLDOWN_MS;
	$_last_party_buff_request_ms = $now;

	my $_resp = _http_post_json('/v2/party/buff_request', {
		meta => _meta(_bot_id()),
		buff_name => $buff_name,
		target_bot => $target_bot || '',
		requested_at => $now,
	});
	if ($_resp && $_resp->{status} >= 200 && $_resp->{status} < 300) {
		debug "[party_buff] buff request sent: $buff_name -> $target_bot\n", 'aiSidecarBridge', 2;
	} else {
		debug "[party_buff] failed to send buff request: $buff_name\n", 'aiSidecarBridge', 1;
	}
}

# ── Party coordination: share target info with party members ──
# Sends current target information to the sidecar for party target coordination.
sub _send_target_coordination {
	my ($target_id, $target_name, $target_hp_pct) = @_;
	return if !_bridge_enabled();
	return if !$registered;

	my $_resp = _http_post_json('/v2/party/target', {
		meta => _meta(_bot_id()),
		target_id => $target_id || '',
		target_name => $target_name || '',
		target_hp_pct => $target_hp_pct || 0,
		map => _safe_field_map() || '',
		timestamp => _now_ms(),
	});
	if ($_resp && $_resp->{status} >= 200 && $_resp->{status} < 300) {
		debug "[party_target] shared target: $target_name (ID=$target_id, HP=$target_hp_pct%)\n", 'aiSidecarBridge', 2;
	}
}

# ── Party coordination: send position update for formation keeping ──
# Shares the bot's current position with party members via the sidecar.
sub _send_position_update {
	return if !_bridge_enabled();
	return if !$registered;
	return if !$char;

	my ($x, $y);
	if ($char->{pos_to} && ref $char->{pos_to} eq 'HASH') {
		$x = $char->{pos_to}{x};
		$y = $char->{pos_to}{y};
	} elsif ($char->{pos} && ref $char->{pos} eq 'HASH') {
		$x = $char->{pos}{x};
		$y = $char->{pos}{y};
	}
	return if !defined $x || !defined $y;

	my $_resp = _http_post_json('/v2/party/position', {
		meta => _meta(_bot_id()),
		x => $x + 0,
		y => $y + 0,
		map => _safe_field_map() || '',
		timestamp => _now_ms(),
	});
	if ($_resp && $_resp->{status} >= 200 && $_resp->{status} < 300) {
		debug "[party_position] sent position: ($x, $y) on $_resp->{map}\n", 'aiSidecarBridge', 3;
	}
}

# ── Party coordination: check party buff expiry ──
# Checks if any tracked party buffs have expired and removes them.
sub _check_party_buff_expiry {
	my $now = _now_ms();
	for my $_pb_name (keys %_party_active_buffs) {
		my $_pb = $_party_active_buffs{$_pb_name};
		if ($now >= $_pb->{expires_at_ms}) {
			delete $_party_active_buffs{$_pb_name};
			debug "[party_buff] buff '$_pb_name' expired (was from $_pb->{source_bot})\n", 'aiSidecarBridge', 2;
		}
	}
}

# ── Party coordination: track a party buff as active ──
# Called when a party member uses a buff skill. Tracks the buff so we
# don't cast the same buff if another bot already has it active.
sub _track_party_buff {
	my ($buff_name, $source_bot, $duration_ms) = @_;
	return if !$buff_name;
	$duration_ms ||= 240000;  # Default 4 minutes for most buffs

	$_party_active_buffs{$buff_name} = {
		source_bot => $source_bot || '',
		expires_at_ms => _now_ms() + $duration_ms,
	};
	debug "[party_buff] tracking buff '$buff_name' from $source_bot for ${duration_ms}ms\n", 'aiSidecarBridge', 2;
}

# ── WOE support: check if WOE is active ──
# Checks the current time against WOE schedule. WOE is typically active
# on Wed/Sat/Sun evenings. Also checks for guild castle map indicators.
sub _check_woe_status {
	my $now = _now_ms();
	return if $now - $_last_woe_check_ms < $WOE_CHECK_INTERVAL_MS;
	$_last_woe_check_ms = $now;

	# Check if we're on a WOE map (guild castle)
	my $map = _safe_field_map() || '';
	my $_on_woe_map = ($map =~ /^guild|^schguild|^turbo_room|^pvp_/i) ? 1 : 0;

	# Check time-based WOE schedule
	# WOE is typically active on Wed/Sat/Sun from 20:00-22:00 server time
	my ($sec, $min, $hour, $mday, $mon, $year, $wday) = localtime(time);
	my $_is_woe_day = ($wday == 3 || $wday == 6 || $wday == 0) ? 1 : 0;  # Wed, Sat, Sun
	my $_is_woe_hour = ($hour >= 20 && $hour < 22) ? 1 : 0;

	my $_was_woe_active = $_woe_active;
	$_woe_active = ($_on_woe_map || ($_is_woe_day && $_is_woe_hour)) ? 1 : 0;

	if ($_woe_active && !$_was_woe_active) {
		warning "[woe] WOE detected as active (map=$map, woe_day=$_is_woe_day, woe_hour=$_is_woe_hour)\n", 'aiSidecarBridge', 1;
		# Post WOE active event
		_post_event({
			kind => 'bridge_reflex',
			reflex => 'woe_active',
			severity => 'info',
			text => "WOE detected active on $map",
			map => $map,
		});
	} elsif (!$_woe_active && $_was_woe_active) {
		warning "[woe] WOE no longer active\n", 'aiSidecarBridge', 1;
	}

	# ── Wire sub-functions into WOE check ──
	if ($_woe_active) {
		_check_emperium_target();
		_woe_defensive_position();
		_woe_escape_reflex();
		# Check dispel risk — if high, avoid casting buffs
		if (_is_woe_dispel_risk()) {
			debug "[woe] dispel risk detected — avoiding buff casts\n", 'aiSidecarBridge', 2;
		}
	}
}

# ── WOE support: emperium targeting ──
# Checks if there's an emperium (guild castle crystal) in the monster list
# and prioritizes it as a target.
sub _check_emperium_target {
	return if !$_woe_active;

	# Search for emperium in the monster list
	if ($monstersList && ref($monstersList) eq 'HASH') {
		for my $_em_id (keys %{$monstersList}) {
			my $_em = $monstersList->{$_em_id};
			next if !ref($_em) eq 'HASH';
			my $_em_name = lc($_em->{name} || '');
			# Emperium names vary by server: "Emperium", "Guild Castle Crystal", etc.
			if ($_em_name =~ /emperium|crystal|guild.*castle/i) {
				$_emperium_target_id = $_em_id;
				debug "[woe] emperium detected: ID=$_em_id name=$_em->{name}\n", 'aiSidecarBridge', 2;
				return 1;
			}
		}
	}
	$_emperium_target_id = '';
	return 0;
}

# ── WOE support: dispel chain awareness ──
# In WOE, certain skills (Dispel, Lex Divina) remove buffs. This checks
# if we should avoid casting buffs that will likely be dispelled.
sub _is_woe_dispel_risk {
	return 0 if !$_woe_active;

	# If we're near enemy players (within 10 cells), buffs are at risk of being dispelled
	if ($playersList && ref($playersList) eq 'HASH') {
		my ($my_x, $my_y);
		if ($char && $char->{pos_to} && ref $char->{pos_to} eq 'HASH') {
			$my_x = $char->{pos_to}{x};
			$my_y = $char->{pos_to}{y};
		}
		return 0 if !defined $my_x || !defined $my_y;

		for my $_pl_id (keys %{$playersList}) {
			my $_pl = $playersList->{$_pl_id};
			next if !ref($_pl) eq 'HASH';
			next if $_pl->{name} && lc($_pl->{name}) eq lc($char->{name} || '');
			# Check if player is within 10 cells
			my $_dx = abs(($_pl->{x} || 0) - $my_x);
			my $_dy = abs(($_pl->{y} || 0) - $my_y);
			if ($_dx <= 10 && $_dy <= 10) {
				return 1;  # Enemy player nearby — dispel risk
			}
		}
	}
	return 0;
}

# ── WOE support: defensive positioning ──
# In WOE, stay near guild members and don't overextend.
sub _woe_defensive_position {
	return if !$_woe_active;
	return if !$char;

	# Check if we're too far from guild members
	if ($playersList && ref($playersList) eq 'HASH') {
		my ($my_x, $my_y);
		if ($char->{pos_to} && ref $char->{pos_to} eq 'HASH') {
			$my_x = $char->{pos_to}{x};
			$my_y = $char->{pos_to}{y};
		}
		return if !defined $my_x || !defined $my_y;

		my $_nearest_guildie_dist = 999;
		for my $_pl_id (keys %{$playersList}) {
			my $_pl = $playersList->{$_pl_id};
			next if !ref($_pl) eq 'HASH';
			# Check if this player is a guild member (same guild)
			next if !$_pl->{guild} || !$_pl->{name};
			my $_dx = abs(($_pl->{x} || 0) - $my_x);
			my $_dy = abs(($_pl->{y} || 0) - $my_y);
			my $_dist = $_dx + $_dy;
			$_nearest_guildie_dist = $_dist if $_dist < $_nearest_guildie_dist;
		}

		# If nearest guild member is > 20 cells away, move toward center
		if ($_nearest_guildie_dist > 20 && $_nearest_guildie_dist < 999) {
			debug "[woe_defense] nearest guild member is ${_nearest_guildie_dist} cells away, staying defensive\n", 'aiSidecarBridge', 2;
		}
	}
}

# ── WOE support: escape reflex for WOE ──
# In WOE, teleport when HP is low instead of sitting.
sub _woe_escape_reflex {
	return if !$_woe_active;
	return if !$char;

	my $now = _now_ms();
	return if $now - $_last_woe_escape_ms < $WOE_ESCAPE_COOLDOWN_MS;

	my $hp = $char->{hp} || 0;
	my $hp_max = $char->{hp_max} || 1;
	my $hp_pct = $hp_max > 0 ? ($hp * 100 / $hp_max) : 100;

	# In WOE, escape at 40% HP instead of sitting
	if ($hp_pct < 40 && $hp_pct > 0) {
		$_last_woe_escape_ms = $now;
		warning "[woe_escape] HP=$hp_pct% on WOE map, teleporting instead of sitting\n", 'aiSidecarBridge', 1;
		eval { Commands::run("use_skill teleport"); 1; };
		_post_event({
			kind => 'bridge_reflex',
			reflex => 'woe_escape',
			severity => 'warning',
			text => "WOE escape reflex: HP=$hp_pct%",
			hp => $hp,
			hp_max => $hp_max,
			map => _safe_field_map() || '',
		});
		return 1;
	}
	return 0;
}

# ── MVP hunting: track MVP spawn timers ──
# Called when an MVP is killed. Records the kill time and calculates
# the respawn window based on the MVP's known respawn time.
sub _track_mvp_kill {
	my ($mvp_name, $mvp_map) = @_;
	return if !$mvp_name;

	my $now = _now_ms();
	# Default respawn window: 60-120 minutes for most MVPs
	# Some MVPs have shorter (30min) or longer (4-8hr) windows
	my $respawn_min_ms = 60 * 60 * 1000;    # 60 minutes
	my $respawn_max_ms = 120 * 60 * 1000;   # 120 minutes

	# Known MVP respawn times (in minutes)
	my %_mvp_respawn_times = (
		'maya' => 60,
		'phreeoni' => 60,
		'moonlight' => 60,
		'edga' => 60,
		'doppelganger' => 60,
		'baphomet' => 120,
		'osiris' => 120,
		'drake' => 60,
		'pharaoh' => 60,
		'mistress' => 60,
		'orc_hero' => 60,
		'orc_lord' => 60,
		'golem' => 30,
		'golden_bug' => 60,
		'kobold_leader' => 60,
		'kobold_king' => 60,
		'stormy_knight' => 60,
		'knight_of_abyss' => 60,
		'hatii' => 60,
		'leib_olmai' => 60,
		'kraken' => 60,
		'turtle_general' => 60,
		'kiel' => 60,
		'kiel_d' => 60,
		'thanatos' => 60,
		'gloom_under_night' => 60,
		'ifrit' => 120,
		'beelzebub' => 120,
		'valkyrie' => 60,
		'randel' => 60,
		'flamel' => 60,
		'skogul' => 60,
		'skeggiold' => 60,
	);

	my $mvp_lc = lc($mvp_name);
	$mvp_lc =~ s/[^a-z0-9_]//g;
	if (defined $_mvp_respawn_times{$mvp_lc}) {
		my $respawn_min = $_mvp_respawn_times{$mvp_lc};
		$respawn_min_ms = $respawn_min * 60 * 1000;
		$respawn_max_ms = $respawn_min * 2 * 60 * 1000;  # 2x for window
	}

	$_mvp_spawn_timers{$mvp_name} = {
		killed_at_ms => $now,
		respawn_window_start => $now + $respawn_min_ms,
		respawn_window_end => $now + $respawn_max_ms,
		map => $mvp_map || _safe_field_map() || '',
	};

	warning "[mvp_tracker] tracked MVP kill: $mvp_name (respawn window: " . int($respawn_min_ms/60000) . "-" . int($respawn_max_ms/60000) . " min on $_mvp_spawn_timers{$mvp_name}->{map})\n", 'aiSidecarBridge', 1;

	# Post MVP kill event to sidecar
	_post_event({
		kind => 'bridge_event',
		event_type => 'mvp.killed',
		severity => 'info',
		text => "MVP killed: $mvp_name",
		mvp_name => $mvp_name,
		map => $_mvp_spawn_timers{$mvp_name}->{map},
		respawn_window_start => $_mvp_spawn_timers{$mvp_name}->{respawn_window_start},
		respawn_window_end => $_mvp_spawn_timers{$mvp_name}->{respawn_window_end},
	});
}

# ── MVP hunting: check MVP status ──
# Checks if any tracked MVPs are in their respawn window and alerts the party.
sub _check_mvp_status {
	my $now = _now_ms();
	return if $now - $_last_mvp_check_ms < $MVP_CHECK_INTERVAL_MS;
	$_last_mvp_check_ms = $now;

	# Check if any tracked MVPs are in their respawn window
	for my $_mvp_name (keys %_mvp_spawn_timers) {
		my $_mvp = $_mvp_spawn_timers{$_mvp_name};
		if ($now >= $_mvp->{respawn_window_start} && $now <= $_mvp->{respawn_window_end}) {
			# MVP is in respawn window — check if we're on the right map
			my $current_map = _safe_field_map() || '';
			if ($current_map eq $_mvp->{map}) {
				warning "[mvp_tracker] MVP '$_mvp_name' is in respawn window on $current_map\n", 'aiSidecarBridge', 1;
				# Alert the sidecar
				_post_event({
					kind => 'bridge_event',
					event_type => 'mvp.respawn_window',
					severity => 'info',
					text => "MVP respawn window: $_mvp_name on $_mvp->{map}",
					mvp_name => $_mvp_name,
					map => $_mvp->{map},
				});
			}
		} elsif ($now > $_mvp->{respawn_window_end}) {
			# Respawn window has passed — clean up
			delete $_mvp_spawn_timers{$_mvp_name};
			debug "[mvp_tracker] MVP '$_mvp_name' respawn window passed, cleaning up\n", 'aiSidecarBridge', 2;
		}
	}

	# Check current monster list for MVP monsters
	if ($monstersList && ref($monstersList) eq 'HASH') {
		for my $_mm_id (keys %{$monstersList}) {
			my $_mm = $monstersList->{$_mm_id};
			next if !ref($_mm) eq 'HASH';
			my $_mm_name = $_mm->{name} || '';
			# MVPs typically have special flags or names
			if ($_mm->{monsterType} && $_mm->{monsterType} eq 'MVP') {
				$_current_mvp_target = $_mm_id;
				debug "[mvp_tracker] MVP detected on map: $_mm_name (ID=$_mm_id)\n", 'aiSidecarBridge', 1;
				# Alert party about MVP spawn
				_post_event({
					kind => 'bridge_event',
					event_type => 'mvp.spawned',
					severity => 'info',
					text => "MVP spawned: $_mm_name",
					mvp_name => $_mm_name,
					mvp_id => $_mm_id,
					map => _safe_field_map() || '',
					x => $_mm->{x} || 0,
					y => $_mm->{y} || 0,
				});
			}
		}
	}

	# ── Wire sub-functions into MVP check ──
	# Check if any tracked MVPs need strategy updates
	for my $_mvp_name (keys %_mvp_spawn_timers) {
		my $_mvp = $_mvp_spawn_timers{$_mvp_name};
		if ($now >= $_mvp->{respawn_window_start} && $now <= $_mvp->{respawn_window_end}) {
			my $strategy = _get_mvp_strategy($_mvp_name);
			if ($strategy && $strategy->{role}) {
				debug "[mvp_tracker] MVP '$_mvp_name' strategy: role=$strategy->{role} min_party=$strategy->{min_party}\n", 'aiSidecarBridge', 2;
			}
		}
	}
}

# ── MVP hunting: get MVP combat strategy ──
# Returns the recommended combat strategy for a given MVP.
# Used by the sidecar to determine tank/DPS/support roles.
sub _get_mvp_strategy {
	my ($mvp_name) = @_;
	return {} if !$mvp_name;

	my $mvp_lc = lc($mvp_name);
	$mvp_lc =~ s/[^a-z0-9_]//g;

	# MVP combat strategies: tank (high HP/def), DPS (glass cannon), support (heal/buff)
	my %_mvp_strategies = (
		'baphomet' => { role => 'tank', min_party => 3, notes => 'high damage, need tank and healer' },
		'osiris' => { role => 'tank', min_party => 3, notes => 'undead, use holy water' },
		'drake' => { role => 'dps', min_party => 2, notes => 'water element, use wind' },
		'pharaoh' => { role => 'dps', min_party => 2, notes => 'neutral element' },
		'maya' => { role => 'tank', min_party => 2, notes => 'reflects physical damage' },
		'phreeoni' => { role => 'dps', min_party => 2, notes => 'ranged attacker' },
		'moonlight' => { role => 'dps', min_party => 2, notes => 'fast movement' },
		'doppelganger' => { role => 'tank', min_party => 3, notes => 'high damage, need tank' },
		'mistress' => { role => 'dps', min_party => 2, notes => 'wind element' },
		'orc_hero' => { role => 'tank', min_party => 2, notes => 'brute force' },
		'orc_lord' => { role => 'tank', min_party => 2, notes => 'brute force' },
		'stormy_knight' => { role => 'dps', min_party => 2, notes => 'wind element' },
		'golden_bug' => { role => 'dps', min_party => 2, notes => 'immune to physical' },
		'ifrit' => { role => 'tank', min_party => 5, notes => 'fire element, need high fire resist' },
		'beelzebub' => { role => 'tank', min_party => 5, notes => 'dark element, need high dark resist' },
		'thanatos' => { role => 'tank', min_party => 4, notes => 'ghost element' },
		'turtle_general' => { role => 'tank', min_party => 3, notes => 'water element' },
		'gloom_under_night' => { role => 'dps', min_party => 3, notes => 'shadow element' },
	);

	return $_mvp_strategies{$mvp_lc} || { role => 'dps', min_party => 1, notes => 'unknown MVP, use standard tactics' };
}

sub _party_leave_state_file {
	my $_pl_dir = $::config{control} || '.';
	$_pl_dir =~ s/\/control$//;
	# Normalize per-profile control dirs (.bot_profiles/<name>/control) to a
	# single shared location, so the leave-suppression latch is actually
	# honored by all 8 bots (previously it resolved to a per-profile path
	# that was never created, so 'party leave' fired and spammed every cycle).
	$_pl_dir =~ s{\.bot_profiles/[^/]+$}{.bot_profiles};
	return "$_pl_dir/ai_sidecar_party_leave_state.txt";
}

# ── Party status relay ──
# Sends party status to the sidecar when char info changes (HP, level, map).
# The sidecar uses this data for party coordination (leader assignment,
# member tracking, party reform after death/respawn).
# Called from _track_lifecycle_transitions on state changes.

sub _send_party_status {
	my ($reason) = @_;
	$reason ||= 'periodic';
	return if !_bridge_enabled();
	return if !$registered;
	return if !$char;

	my $_party_status = {
		bot_id => _bot_id(),
		char_name => $char->{name} || '',
		base_level => $char->{lv} || $char->{level} || 0,
		job_level => $char->{level_job} || 0,
		hp => $char->{hp} || 0,
		hp_max => $char->{hp_max} || 1,
		map => _safe_field_map() || '',
		in_party => 0,
		party_members => [],
		reason => $reason,
		timestamp => _now_ms(),
	};

	if (defined $char->{party}) {
		$_party_status->{in_party} = 1;
		my $pu = $char->{party}{users} || {};
		my @members;
		for my $_pk (keys %$pu) {
			my $_pm = $pu->{$_pk};
			my $_pn = '';
			if (UNIVERSAL::can($_pm, 'name')) {
				$_pn = eval { $_pm->name() } || '';
			}
			if (!$_pn) {
				$_pn = eval { $_pm->{name} } || '';
			}
			push @members, lc($_pn) if $_pn;
		}
		$_party_status->{party_members} = \@members;

		# Cache party state to survive death/disconnect
		my $_cache_key = _bot_id();
		$::aiSidecar_cached_party{$_cache_key} = {
			in_party => 1,
			members => [@members],
		};

		# Party leader detection (first bot in all_bots)
		if ($::aiSidecar_all_bots && $::config{username}) {
			my @_all = split(',', $::aiSidecar_all_bots);
			$_party_status->{is_party_leader} = (@_all && $::config{username} eq $_all[0]) ? 1 : 0;
		}
	} else {
		# Not in party — check cache for stale party state
		my $_cache_key = _bot_id();
		if (defined $::aiSidecar_cached_party{$_cache_key}) {
			my $cache = $::aiSidecar_cached_party{$_cache_key};
			$_party_status->{cached_party} = {
				in_party => $cache->{in_party},
				members => [@{$cache->{members}}],
			};
		}
	}

	# POST to sidecar
	my $resp = _http_post_json('/v2/party/status', $_party_status);
	if (!$resp || $resp->{status} < 200 || $resp->{status} >= 300) {
		debug "[party_status] failed to send party status (reason=$reason)\n", 'aiSidecarBridge', 1;
	} else {
		debug "[party_status] sent party status (reason=$reason, in_party=$_party_status->{in_party})\n", 'aiSidecarBridge', 2;
	}

	# If leader and in party, also check for missing members
	if ($_party_status->{is_party_leader} && $_party_status->{in_party}) {
		my %_mn;
		for my $_m (@{$_party_status->{party_members}}) {
			$_mn{$_} = 1;
		}
		$_mn{lc($char->{name} || '')} = 1;

		my @_all_bots = split(',', $::aiSidecar_all_bots || '');
		for my $_pn (@_all_bots) {
			next if $_pn eq ($::config{username} || '');
			my $_cn = $::aiSidecar_profile_to_char{$_pn} || $_pn;
			if (!$_mn{lc($_cn)}) {
				debug "[party_status] leader would invite missing member: $_cn (coordinator owns invites)\n", 'aiSidecarBridge', 1;
			}
		}
	}
}

# ── State persistence relay ──
# Relays key-value state between the sidecar's PersistenceManager and
# the local filesystem. The sidecar pushes state updates via "set" commands
# and reads state via HTTP GET. This bridge layer persists to a JSON file
# so state survives process restarts.
# API:
#   _state_set(key, value) -> persist a key-value pair
#   _state_get(key) -> retrieve a value (undef if not found)
#   _state_clear() -> remove all persisted state

sub _state_file {
	my $_sf_dir = $::config{control} || '.';
	$_sf_dir =~ s/\/control$//;
	return "$_sf_dir/ai_sidecar_bridge_state.json";
}

sub _state_set {
	my ($key, $value) = @_;
	return if !defined $key || $key eq '';

	my $_sf = _state_file();
	my %state = ();

	# Read existing state
	if (-e $_sf) {
		if (open my $_fh, '<', $_sf) {
			local $/;
			my $_raw = <$_fh>;
			close $_fh;
			if (defined $_raw && $_raw ne '') {
				eval { %state = %{ JSON::PP::decode_json($_raw) }; 1; };
			}
		}
	}

	# Set the new value
	$state{$key} = defined $value ? $value : '';

	# Write back atomically
	if (open my $_fh, '>', $_sf) {
		print $_fh JSON::PP::encode_json(\%state);
		close $_fh;
		debug "[state_persistence] set key='$key' file=$_sf\n", 'aiSidecarBridge', 2;
	} else {
		debug "[state_persistence] cannot write $_sf: $!\n", 'aiSidecarBridge', 1;
	}
}

sub _state_get {
	my ($key) = @_;
	return undef if !defined $key || $key eq '';

	my $_sf = _state_file();
	return undef if !-e $_sf;

	my %state = ();
	if (open my $_fh, '<', $_sf) {
		local $/;
		my $_raw = <$_fh>;
		close $_fh;
		if (defined $_raw && $_raw ne '') {
			eval { %state = %{ JSON::PP::decode_json($_raw) }; 1; };
		}
	}

	my $value = $state{$key};
	return $value if defined $value && $value ne '';
	return undef;
}

sub _state_clear {
	my $_sf = _state_file();
	if (-e $_sf) {
		unlink $_sf or debug "[state_persistence] cannot unlink $_sf: $!\n", 'aiSidecarBridge', 1;
	}
	debug "[state_persistence] cleared all state from $_sf\n", 'aiSidecarBridge', 2;
}

sub _execute_config_reload_action {
	my ($metadata) = @_;
	if (!_cfg_bool('aiSidecar_configReloadEnabled', 1)) {
		return (0, 'config_reload_disabled', 'config reload orchestration disabled by bridge config');
	}
	my $target = _safe_control_filename($metadata->{target} || 'config.txt', 'config.txt');
	my $command = "reload $target";
	my ($ok, $err) = _run_safe_openkore_command($command);
	if (!$ok) {
		return (0, 'config_reload_failed', "config reload failed for '$command': $err");
	}
	return (1, 'ok', "config reload completed through OpenKore command pathway for '$target'");
}

sub _execute_macro_reload_action {
	my ($metadata) = @_;

	if (!_cfg_bool('aiSidecar_macroReloadEnabled', 1)) {
		return (0, 'macro_reload_disabled', 'macro reload orchestration disabled by bridge config');
	}

	my $macro_file = _safe_control_filename(
		$metadata->{macro_file} || _cfg('aiSidecar_macroFile', 'ai_sidecar_generated_macros.txt'),
		'ai_sidecar_generated_macros.txt',
	);
	my $event_macro_file = _safe_control_filename(
		$metadata->{event_macro_file} || _cfg('aiSidecar_eventMacroFile', 'ai_sidecar_generated_eventmacros.txt'),
		'ai_sidecar_generated_eventmacros.txt',
	);
	my $macro_plugin = _safe_plugin_name(
		$metadata->{macro_plugin} || _cfg('aiSidecar_macroPluginName', 'macro'),
		'macro',
	);
	my $event_macro_plugin = _safe_plugin_name(
		$metadata->{event_macro_plugin} || _cfg('aiSidecar_eventMacroPluginName', 'eventMacro'),
		'eventMacro',
	);

	if (!exists $config{macro_file}) {
		debug "[aiSidecarBridge] macro_file is missing; forcing creation before macro plugin reload\n", 'aiSidecarBridge', 2;
	}

	my @commands = (
		"conf -f macro_file $macro_file",
		"plugin reload $macro_plugin",
		"conf eventMacro_file $event_macro_file",
		"plugin reload $event_macro_plugin",
	);

	foreach my $safe_command (@commands) {
		my ($ok, $err) = _run_safe_openkore_command($safe_command);
		if (!$ok) {
			return (0, 'macro_reload_failed', "macro reload step failed for '$safe_command': $err");
		}

		if ($safe_command =~ /^conf\s+-f\s+macro_file\b/) {
			if (!exists $config{macro_file} || !defined $config{macro_file} || $config{macro_file} ne $macro_file) {
				my $actual = exists $config{macro_file} ? _trim(_scalarize($config{macro_file}), 120) : 'undef';
				return (0, 'macro_reload_failed', "macro reload step failed for '$safe_command': macro_file did not persist (actual='$actual')");
			}
		}
	}

	my $publication_id = defined $metadata->{publication_id} ? _trim($metadata->{publication_id}, 64) : '';
	my $version = defined $metadata->{version} ? _trim($metadata->{version}, 64) : '';
	my $suffix = '';
	$suffix .= " publication_id=$publication_id" if $publication_id ne '';
	$suffix .= " version=$version" if $version ne '';

	return (1, 'ok', "macro and eventMacro hot reload completed through existing OpenKore command pathways$suffix");
}

sub _run_safe_openkore_command {
	my ($command) = @_;
	# LOGGED-OUT GATE: this helper is used by the macro/config hot-reload
	# paths, which run OUTSIDE _execute_action and therefore bypass the
	# hoisted logged-out gate. Block in-game commands here too — reloading
	# macros while at char-select is fine (config-only), but executing
	# in-game commands is not.
	my $_rs_root = lc(($command =~ /^\s*(\S+)/)[0] || '');
	if (!($net && $net->getState() == Network::IN_GAME)
		&& $_rs_root ne 'char_select' && $_rs_root ne 'char_create' && $_rs_root ne 'conf') {
		debug "[safe_cmd_logged_out] blocked '$command' while bot not in game\n", 'aiSidecarBridge', 2;
		return (1, 'skipped: bot not in game');
	}
	my $ok = eval { Commands::run($command); 1; };
	if ($ok) {
		debug "[aiSidecarBridge] executed safe command '$command'\n", 'aiSidecarBridge', 2;
		return (1, '');
	}

	my $err = $@ || 'command execution failure';
	return (0, _trim($err, 220));
}

sub _safe_control_filename {
	my ($candidate, $default) = @_;
	$candidate = $default if !defined $candidate || $candidate eq '';
	$candidate =~ s/^\s+//;
	$candidate =~ s/\s+$//;

	if ($candidate =~ m{[\\/]} || $candidate !~ /^[A-Za-z0-9_.-]+$/) {
		return $default;
	}

	return $candidate;
}

sub _safe_plugin_name {
	my ($candidate, $default) = @_;
	$candidate = $default if !defined $candidate || $candidate eq '';
	$candidate =~ s/^\s+//;
	$candidate =~ s/\s+$//;

	if ($candidate =~ m{[\\/]} || $candidate !~ /^[A-Za-z0-9_.:-]+$/) {
		return $default;
	}

	return $candidate;
}

sub _flush_ack_queue {
	return if !_bridge_enabled();
	return if !@ack_queue;

	my $now = _now_ms();
	my $max_age_ms = _cfg_int('aiSidecar_ackMaxAgeMs', 5000);
	while (@ack_queue && $now - $ack_queue[0]{queued_at} > $max_age_ms) {
		my $dropped = shift @ack_queue;
		_throttled_warning('ack_dropped', "[aiSidecarBridge] dropped stale ack '$dropped->{action_id}'.");
	}
	return if !@ack_queue;

	my $ack = $ack_queue[0];
	my $payload = {
		meta => _meta(_bot_id()),
		action_id => $ack->{action_id},
		poll_id => $ack->{poll_id},
		success => $ack->{success} ? 1 : 0,
		result_code => $ack->{result_code},
		message => $ack->{message},
		observed_latency_ms => int($ack->{observed_latency_ms} || 0),
	};

	my $resp = _http_post_json('/v1/acknowledgements/action', $payload);
	if ($resp && $resp->{status} >= 200 && $resp->{status} < 300) {
		shift @ack_queue;
		return;
	}

	$registered = 0;
	_throttled_warning('ack_failed', '[aiSidecarBridge] action ack failed, will retry while within ack age budget.');
}


# ── Wrapper: POST event to sidecar in proper batch format ──
sub _post_event {
	my ($event) = @_;
	return if !_bridge_enabled();
	# Normalize flat bridge event to NormalizedEvent schema
	my $event_family = $event->{kind} || 'bridge_reflex';
	my $event_type = $event->{reflex} || $event->{event_type} || 'unknown';
	my $severity = $event->{severity} || 'info';
	# Build tags (string values) and numeric (float values) from event params
	my %tags = ();
	my %numeric = ();
	my $text = $event->{text} || '';
	my %payload = ();
	while (my ($k, $v) = each %$event) {
		next if $k eq 'kind' || $k eq 'reflex' || $k eq 'severity' || $k eq 'text' || $k eq 'event_type' || $k eq 'timestamp';
		if (!defined $v) { next }
		if ($v =~ /^-?\d+\.?\d*$/) {
			$numeric{$k} = $v + 0.0;
		} else {
			$tags{$k} = substr($v, 0, 256);
		}
	}
	$text = substr($event_type . ' ' . join(' ', values %tags), 0, 1024) if $text eq '';
	my $normalized = {
		meta => _meta(_bot_id()),
		event_family => $event_family,
		event_type => $event_type,
		severity => $severity,
		text => $text,
		tags => \%tags,
		numeric => \%numeric,
		payload => \%payload,
	};
	my $payload = {
		meta => _meta(_bot_id()),
		events => [$normalized],
	};
	my $resp = _http_post_json('/v2/ingest/event', $payload);
	return $resp;
}

sub _emit_telemetry {
	my ($level, $category, $event, $message_text, $metrics, $tags) = @_;
	return if !_cfg_bool('aiSidecar_telemetryEnabled', 1);

	$metrics ||= {};
	$tags ||= {};
	push @telemetry_queue, {
		timestamp => _iso_now(),
		level => $level,
		category => $category,
		event => $event,
		message => _trim($message_text || '', 500),
		metrics => $metrics,
		tags => $tags,
	};

	if (@telemetry_queue > 200) {
		splice @telemetry_queue, 0, @telemetry_queue - 200;
	}
}

sub _flush_telemetry_queue {
	return if !_bridge_enabled();
	return if !@telemetry_queue;

	my $batch_size = @telemetry_queue > 20 ? 20 : scalar @telemetry_queue;
	my @batch = splice @telemetry_queue, 0, $batch_size;

	my $payload = {
		meta => _meta(_bot_id()),
		events => \@batch,
	};

	my $resp = _http_post_json('/v1/telemetry/ingest', $payload);
	if (!$resp || $resp->{status} < 200 || $resp->{status} >= 300) {
		unshift @telemetry_queue, @batch;
		splice @telemetry_queue, 0, @telemetry_queue - 200 if @telemetry_queue > 200;
		_throttled_warning('telemetry_failed', '[aiSidecarBridge] telemetry push failed, fail-open retained.');
	}
}

sub _flush_event_queue {
	return 1 if !_bridge_enabled();
	return 1 if !@event_queue;

	my $batch_size = _cfg_int('aiSidecar_eventBatchSize', 20);
	$batch_size = 1 if $batch_size < 1;
	$batch_size = 100 if $batch_size > 100;
	$batch_size = scalar(@event_queue) if $batch_size > scalar(@event_queue);

	my @batch = splice @event_queue, 0, $batch_size;
	# Normalize each event in batch
	my @normalized = map {
		my $event = $_;
		my $event_family = $event->{kind} || 'bridge_event';
		# Normalize event_family to valid values for the sidecar API
		my %family_map = (
			'reflex' => 'bridge_reflex',
			'bridge_reflex' => 'bridge_reflex',
			'bridge_event' => 'bridge_event',
			'bridge_telemetry' => 'bridge_telemetry',
			'discovery_shops' => 'discovery_shops',
			'discovery' => 'discovery',
			'snapshot' => 'bridge_event',
			'config' => 'bridge_event',
		);
		$event_family = $family_map{$event_family} if exists $family_map{$event_family};
		my $event_type = $event->{reflex} || $event->{event_type} || 'bridge.unknown';
		# Normalize event_type to only contain safe characters
		$event_type =~ s/[^A-Za-z0-9_.:\/-]/_/g;
		my $severity = $event->{severity} || 'info';
		my %tags = (); my %numeric = (); my $text = $event->{text} || '';
		my %payload = ();
		while (my ($k, $v) = each %$event) {
			next if $k eq 'kind' || $k eq 'reflex' || $k eq 'severity' || $k eq 'text' || $k eq 'event_type' || $k eq 'timestamp';
			if (!defined $v) { next }
			# Handle array/ref values: flatten to comma-separated string
			if (ref($v) eq 'ARRAY') {
				$tags{$k} = join(',', map { defined $_ ? _scalarize($_) : '' } @$v);
				next;
			}
			if (ref($v)) {
				$tags{$k} = _scalarize($v);
				next;
			}
			if ($v =~ /^-?\d+\.?\d*$/) { $numeric{$k} = $v + 0.0; }
			else { $tags{$k} = substr($v, 0, 256); }
		}
		$text = substr($event_type . ' ' . join(' ', values %tags), 0, 1024) if $text eq '';
		+{
			meta => _meta(_bot_id()),
			event_family => $event_family,
			event_type => $event_type,
			severity => $severity,
			text => $text,
			tags => \%tags,
			numeric => \%numeric,
			payload => \%payload,
		};
	} @batch;
	my $payload = {
		meta => _meta(_bot_id()),
		events => \@normalized,
	};

	my $resp = _http_post_json('/v2/ingest/event', $payload);
	my $status = _http_status_code($resp);
	if ($status < 200 || $status >= 300) {
		if ($status >= 400 && $status < 500) {
			# Client error (422 etc) — discard batch, won't fix on retry
			_throttled_warning('v2_event_discarded', "[aiSidecarBridge] v2 event discarded status=$status (client error, discarding batch).");
			$consecutive_v2_event_failures += 1;
			return 0;
		}
		unshift @event_queue, @batch;
		my $max_queue = _cfg_int('aiSidecar_maxEventQueue', 300);
		$max_queue = 50 if $max_queue < 50;
		splice @event_queue, 0, @event_queue - $max_queue if @event_queue > $max_queue;

		$consecutive_v2_event_failures += 1;
		my $backoff_ms = _event_ingest_failure_delay_ms();
		my $err = _http_error_text($resp);
		my $depth = scalar(@event_queue);
		_throttled_warning(
			'v2_event_failed',
			"[aiSidecarBridge] v2 event push failed status=$status error=$err failures=$consecutive_v2_event_failures queue_depth=$depth next_retry_ms=$backoff_ms (bounded queue retained).",
		);
		_emit_telemetry(
			'warning',
			'bridge',
			'v2_event_failed',
			'v2 event push failed',
			{
				status => 0 + $status,
				consecutive_failures => 0 + $consecutive_v2_event_failures,
				queue_depth => 0 + $depth,
				next_retry_ms => 0 + $backoff_ms,
			},
			{ endpoint => '/v2/ingest/event', error => $err },
		);
		return 0;
	}

	if ($consecutive_v2_event_failures > 0) {
		debug "[aiSidecarBridge] v2 event ingest recovered after $consecutive_v2_event_failures consecutive failures\n", 'aiSidecarBridge', 2;
	}
	$consecutive_v2_event_failures = 0;
	return 1;
}

sub _flush_chat_queue {
	return if !_bridge_enabled();
	return if !@chat_queue;

	my $batch_size = _cfg_int('aiSidecar_chatBatchSize', 20);
	$batch_size = 1 if $batch_size < 1;
	$batch_size = 100 if $batch_size > 100;
	$batch_size = scalar(@chat_queue) if $batch_size > scalar(@chat_queue);

	my @batch = splice @chat_queue, 0, $batch_size;
	my %channels;
	foreach my $event (@batch) {
		next if ref($event) ne 'HASH';
		my $channel = _trim(_scalarize($event->{channel}), 64);
		$channels{$channel} = 1 if $channel ne '';
	}

	my $payload = {
		meta => _meta(_bot_id()),
		observed_at => _iso_now(),
		events => \@batch,
		interaction_intent => {
			source => 'bridge',
			channels => [sort keys %channels],
		},
	};

	my $resp = _http_post_json('/v2/ingest/chat', $payload);
	if (!$resp || $resp->{status} < 200 || $resp->{status} >= 300) {
		unshift @chat_queue, @batch;
		my $max_queue = _cfg_int('aiSidecar_maxChatQueue', 200);
		$max_queue = 40 if $max_queue < 40;
		splice @chat_queue, 0, @chat_queue - $max_queue if @chat_queue > $max_queue;
		_throttled_warning('v2_chat_failed', '[aiSidecarBridge] v2 chat push failed, retaining bounded queue.');
		return;
	}
}

sub _flush_config_updates {
	return if !_bridge_enabled();
	return if !%pending_config_keys;

	my @all_keys = sort keys %pending_config_keys;
	my $max_keys = _cfg_int('aiSidecar_maxConfigKeysPerPush', 64);
	$max_keys = 1 if $max_keys < 1;
	$max_keys = scalar(@all_keys) if $max_keys > scalar(@all_keys);

	my @keys = @all_keys[0 .. ($max_keys - 1)];
	my %values;
	foreach my $key (@keys) {
		my $value = defined $config{$key} ? $config{$key} : '';
		$values{$key} = _trim(_scalarize($value), _cfg_int('aiSidecar_maxConfigValueChars', 220));
	}

	my $payload = {
		meta => _meta(_bot_id()),
		observed_at => _iso_now(),
		fingerprint => _stable_config_fingerprint(\@keys, \%values),
		doctrine_version => _cfg('aiSidecar_contractVersion', 'v1'),
		changed_keys => \@keys,
		values => \%values,
		source_files => ['config.txt', 'ai_sidecar.txt', 'ai_sidecar_policy.txt', _active_control_folder()],
	};

	my $resp = _http_post_json('/v2/ingest/config', $payload);
	if (!$resp || $resp->{status} < 200 || $resp->{status} >= 300) {
		if ($resp && $resp->{status} >= 400 && $resp->{status} < 500) {
			# Client error — discard pending keys, won't fix on retry
			foreach my $key (@keys) { delete $pending_config_keys{$key}; }
			_throttled_warning('v2_config_discarded', "[aiSidecarBridge] v2 config discarded status=$resp->{status} (client error, cleared pending).");
			return;
		}
		_throttled_warning('v2_config_failed', '[aiSidecarBridge] v2 config push failed, pending keys retained.');
		return;
	}

	foreach my $key (@keys) {
		delete $pending_config_keys{$key};
	}
}

sub _enqueue_chat_event {
	my ($event) = @_;
	return if ref($event) ne 'HASH';

	my $chat = {
		channel => _trim(_scalarize($event->{channel}), 64),
		sender => _trim(_scalarize($event->{sender}), 128),
		target => _trim(_scalarize($event->{target}), 128),
		message => _trim(_scalarize($event->{message}), _cfg_int('aiSidecar_maxChatChars', 500)),
		map => _trim(_scalarize($event->{map}), 64),
		kind => _trim(_scalarize($event->{kind}), 64),
		raw => ref($event->{raw}) eq 'HASH' ? $event->{raw} : {},
	};

	return if $chat->{channel} eq '' || $chat->{message} eq '';

	$chat->{sender} = undef if $chat->{sender} eq '';
	$chat->{target} = undef if $chat->{target} eq '';
	$chat->{map} = undef if $chat->{map} eq '';
	$chat->{kind} = undef if $chat->{kind} eq '';

	push @chat_queue, $chat;
	my $max_queue = _cfg_int('aiSidecar_maxChatQueue', 200);
	$max_queue = 40 if $max_queue < 40;
	splice @chat_queue, 0, @chat_queue - $max_queue if @chat_queue > $max_queue;
}

sub _enqueue_normalized_event {
	my ($family, $event_type, $source_hook, $text, $payload, $tags, $numeric, $severity, $correlation_id) = @_;
	$payload ||= {};
	$tags ||= {};
	$numeric ||= {};

	my %allowed_family = map { $_ => 1 } qw(snapshot hook packet config actor_state chat quest telemetry macro action lifecycle system);
	my %allowed_severity = map { $_ => 1 } qw(debug info warning error critical);

	$family = _trim(_scalarize($family), 32);
	$family = 'system' if $family eq '' || !$allowed_family{$family};

	$severity = _trim(lc(_scalarize($severity)), 16);
	$severity = 'info' if $severity eq '' || !$allowed_severity{$severity};

	my %safe_tags;
	if (ref($tags) eq 'HASH') {
		foreach my $key (sort keys %{$tags}) {
			next if !defined $key || $key eq '';
			my $tag_key = _trim(_scalarize($key), 64);
			next if $tag_key eq '';
			my $tag_val = _trim(_scalarize($tags->{$key}), 128);
			next if $tag_val eq '';
			$safe_tags{$tag_key} = $tag_val;
		}
	}

	my %safe_numeric;
	if (ref($numeric) eq 'HASH') {
		foreach my $key (sort keys %{$numeric}) {
			next if !defined $key || $key eq '';
			my $num_key = _trim(_scalarize($key), 64);
			next if $num_key eq '';
			my $val = $numeric->{$key};
			next if !defined $val;
			my $str = _scalarize($val);
			next if $str !~ /^-?(?:\d+|\d*\.\d+)$/;
			$safe_numeric{$num_key} = 0 + $str;
		}
	}

	my $event = {
		meta => _meta(_bot_id()),
		event_id => 'evt-' . _trace_id(),
		event_family => $family,
		event_type => _normalize_event_type($event_type),
		observed_at => _iso_now(),
		sequence => _next_event_seq(),
		source_hook => _trim(_scalarize($source_hook), 256),
		correlation_id => _trim(_scalarize($correlation_id), 128),
		severity => $severity,
		text => _trim(_scalarize($text), 1024),
		tags => \%safe_tags,
		numeric => \%safe_numeric,
		payload => ref($payload) eq 'HASH' ? $payload : {},
	};

	$event->{source_hook} = undef if $event->{source_hook} eq '';
	$event->{correlation_id} = undef if $event->{correlation_id} eq '';

	push @event_queue, $event;
	my $max_queue = _cfg_int('aiSidecar_maxEventQueue', 300);
	$max_queue = 50 if $max_queue < 50;
	splice @event_queue, 0, @event_queue - $max_queue if @event_queue > $max_queue;
}

sub _next_event_seq {
	$event_seq += 1;
	$event_seq = 1 if $event_seq < 1;
	return $event_seq;
}

sub _extract_hook_payload {
	my ($args) = @_;
	return {} if ref($args) ne 'HASH';

	my $max_fields = _cfg_int('aiSidecar_maxEventPayloadFields', 16);
	$max_fields = 1 if $max_fields < 1;

	my %out;
	my $count = 0;
	foreach my $key (sort keys %{$args}) {
		last if $count >= $max_fields;
		next if !defined $key || $key eq '';

		my $value = $args->{$key};
		my $ref = ref($value);
		if (!$ref) {
			$out{$key} = _trim(_scalarize($value), 240);
		} elsif ($ref eq 'SCALAR' || $ref eq 'REF') {
			my $deref = eval { defined $$value ? $$value : '' };
			$out{$key} = _trim(_scalarize($deref), 240);
		} elsif ($ref eq 'ARRAY') {
			my @vals;
			my $i = 0;
			foreach my $item (@{$value}) {
				last if $i >= 6;
				push @vals, _trim(_scalarize($item), 140);
				$i++;
			}
			$out{$key} = \@vals;
		} elsif ($ref eq 'HASH') {
			my %sub;
			my $i = 0;
			foreach my $sub_key (sort keys %{$value}) {
				last if $i >= 6;
				next if !defined $sub_key || $sub_key eq '';
				$sub{$sub_key} = _trim(_scalarize($value->{$sub_key}), 140);
				$i++;
			}
			$out{$key} = \%sub;
		} else {
			$out{$key} = _trim("[$ref]", 64);
		}

		$count++;
	}

	return \%out;
}

sub _pick_first {
	my ($hash, @keys) = @_;
	return undef if ref($hash) ne 'HASH';
	foreach my $key (@keys) {
		if (exists $hash->{$key} && defined $hash->{$key}) {
			return $hash->{$key};
		}
	}
	return undef;
}

sub _scalarize {
	my ($value) = @_;
	return '' if !defined $value;

	my $ref = ref($value);
	return "$value" if !$ref;

	if ($ref eq 'SCALAR' || $ref eq 'REF') {
		my $deref = eval { $$value };
		return defined $deref ? "$deref" : '';
	}

	if ($ref eq 'ARRAY') {
		my @parts;
		my $i = 0;
		foreach my $item (@{$value}) {
			last if $i >= 8;
			push @parts, _trim(_scalarize($item), 80);
			$i++;
		}
		return join(',', @parts);
	}

	if ($ref eq 'HASH') {
		my @parts;
		my $i = 0;
		foreach my $key (sort keys %{$value}) {
			last if $i >= 8;
			push @parts, $key . '=' . _trim(_scalarize($value->{$key}), 60);
			$i++;
		}
		return join(',', @parts);
	}

	my $string = eval { "$value" };
	return defined $string ? $string : $ref;
}

sub _normalize_event_type {
	my ($value) = @_;
	$value = lc(_scalarize($value));
	$value =~ s/\s+/_/g;
	$value =~ s/[^a-z0-9_.:\/\-]+/_/g;
	$value =~ s/_+/_/g;
	$value =~ s/^_+//;
	$value =~ s/_+$//;
	$value = 'system.unknown' if $value eq '';
	return _trim($value, 120);
}

sub _safe_field_map {
	my $map = eval { $field ? $field->baseName() : '' };
	return _trim(_scalarize($map), 64);
}

sub _safe_ai_seq_top {
	my $top = @ai_seq ? $ai_seq[0] : '';
	return _trim(_scalarize($top), 64);
}

sub _stable_config_fingerprint {
	my ($keys, $values) = @_;
	my $timestamp = int(time() * 1000);

	if (ref($keys) ne 'ARRAY' || !@{$keys}) {
		return sprintf('cfg-%x-%x-%d', $timestamp, int(rand(0xFFFFFF)), 0);
	}

	my $hash = 5381;
	foreach my $key (@{$keys}) {
		my $val = (ref($values) eq 'HASH' && exists $values->{$key}) ? $values->{$key} : '';
		my $pair = $key . '=' . $val;
		foreach my $ch (split //, $pair) {
			$hash = (($hash * 33) + ord($ch)) & 0x7FFFFFFF;
		}
	}

	return sprintf('cfg-%x-%x-%d', $timestamp, $hash, scalar(@{$keys}));
}


sub _http_get_json {
	my ($path) = @_;
	return undef if !$json_available;
	_load_bridge_config_overrides();
	my $base_url = _cfg('aiSidecar_baseUrl', 'http://127.0.0.1:18081');
	$base_url =~ s{/+$}{};
	my ($scheme, $host, $port) = $base_url =~ m{^(https?)://([^/:]+):?(\d*)}i;
	return { status => 0, error => 'invalid_base_url', json => undef, raw => '' } if !$scheme || lc($scheme) ne 'http' || !$host;
	$port ||= 80;
	require IO::Socket::INET;
	my $sock = IO::Socket::INET->new(PeerHost=>$host, PeerPort=>$port, Proto=>'tcp', Timeout=>2)
		or return { status=>0, error=>'connect_failed', json=>undef, raw=>'' };
	my $req = "GET $path HTTP/1.1\r\nHost: $host:$port\r\nAccept: application/json\r\nConnection: close\r\n\r\n";
	$sock->send($req);
	# Read with alarm timeout: if the server doesn't close the connection
	# (e.g. keep-alive race), the while loop blocks forever.
	my $io_timeout = _cfg_int('aiSidecar_ioTimeoutMs', 5000) / 1000;
	$io_timeout = 0.001 if $io_timeout <= 0;
	my $resp = '';
	eval {
		local $SIG{ALRM} = sub { die "bridge_http_get_timeout\n"; };
		alarm($io_timeout);
		while (<$sock>) { $resp .= $_; }
		alarm(0);
		1;
	};
	alarm(0);
	close($sock);
	my ($header, $body) = split /\r\n\r\n/, $resp, 2;
	my $status = ($header =~ /HTTP\/\d\.\d\s+(\d+)/) ? $1 : 0;
	my $json = undef;
	eval { $json = JSON::PP::decode_json($body) } if defined $body && $body ne '';
	return { status=>$status, error=>'', json=>$json, raw=>$body };
}
sub _http_post_json {
	my ($path, $payload) = @_;
	return undef if !$json_available;
	_load_bridge_config_overrides();

	my $base_url = _cfg('aiSidecar_baseUrl', 'http://127.0.0.1:18081');
	$base_url =~ s{/+$}{};
	my ($scheme, $host, $port, $base_path) = $base_url =~ m{^(https?)://([^/:]+)(?::(\d+))?(/.*)?$}i;
	if (!$scheme || lc($scheme) ne 'http' || !$host) {
		_throttled_warning('invalid_base_url', "[aiSidecarBridge] invalid aiSidecar_baseUrl '$base_url'; expected http://host:port");
		return {
			status => 0,
			error => 'invalid_base_url',
			json => undef,
			raw => '',
		};
	}

	$port ||= 80;
	$base_path ||= '';
	my $request_path = "$base_path$path";
	$request_path =~ s{//+}{/}g;
	$request_path = "/$request_path" if $request_path !~ m{^/};

	my $body = eval { JSON::PP::encode_json($payload) };
	if (!$body || $@) {
		_throttled_warning('json_encode_failed', '[aiSidecarBridge] JSON encoding failed; request skipped.');
		return {
			status => 0,
			error => 'json_encode_failed',
			json => undef,
			raw => '',
		};
	}

	my $connect_timeout = _cfg_int('aiSidecar_connectTimeoutMs', 2000) / 1000;
	my $io_timeout = _cfg_int('aiSidecar_ioTimeoutMs', 5000) / 1000;
	$connect_timeout = 0.001 if $connect_timeout <= 0;
	$io_timeout = 0.001 if $io_timeout <= 0;

	# ── Connection ──
	my $sock = IO::Socket::INET->new(
	    PeerHost => $host,
	    PeerPort => $port,
	    Proto => 'tcp',
	    Timeout => $connect_timeout,
	);
	if (!$sock) {
	    return {
	        status => 0,
	        error => _trim('connect_failed:' . ($! || 'socket_open_failed'), 220),
	        json => undef,
	        raw => '',
	    };
	}
	$sock->autoflush(1);

	my $request = join(
	    "\r\n",
	    "POST $request_path HTTP/1.1",
	    "Host: $host:$port",
	    "Content-Type: application/json",
	    "Accept: application/json",
	    "Connection: close",
	    "Content-Length: " . length($body),
	    '',
	    $body,
	);

		my $raw_response = '';
		my $io_error = '';
		my $ok = eval {
		    local $SIG{ALRM} = sub { die "bridge_http_timeout\n"; };
		    alarm($io_timeout);
		    print {$sock} $request;
		    # Read headers first (stop at double CRLF)
		    my $header_buf = '';
		    while (1) {
		        my $chunk = '';
		        my $read = sysread($sock, $chunk, 4096);
		        last if !defined $read || $read <= 0;
		        $header_buf .= $chunk;
		        last if $header_buf =~ /\r?\n\r?\n/;
		    }
		    # Parse Content-Length from headers
		    my $content_length = 0;
		    if ($header_buf =~ /Content-Length:\s*(\d+)/i) {
		        $content_length = $1;
		    }
		    $raw_response = $header_buf;
		    # Read remaining body if Content-Length says there's more
		    if ($content_length > 0) {
		        my ($headers, $body_so_far) = split(/\r?\n\r?\n/, $header_buf, 2);
		        my $have = defined $body_so_far ? length($body_so_far) : 0;
		        while ($have < $content_length) {
		            my $chunk = '';
		            my $read = sysread($sock, $chunk, 4096);
		            last if !defined $read || $read <= 0;
		            $raw_response .= $chunk;
		            $have += $read;
		        }
		    }
		    alarm(0);  # Clear alarm BEFORE the 1 so it runs even if the eval
		               # body completes normally (not via die).
		    1;
		};
	$io_error = $@ if !$ok;
	alarm(0);  # Safety net: clear alarm AFTER the eval too, in case the
	           # eval died from a timeout (alarm(0) inside eval was never
	           # reached). Without this, the SIGALRM continues ticking and
	           # fires in unrelated code 5 seconds later, corrupting state.
	# Close socket on both success and failure — the keep-alive header is
	# sent but we never reuse the socket (it goes out of scope at sub end).
	# Leaking ~16 sockets/s × 8 bots × hours exhausts file descriptors.
	close $sock;
	if (!$ok) {

	    $io_error = _trim($io_error || 'io_failure', 220);
	    $io_error =~ s/\s+$//;
	    return {
	        status => 0,
	        error => $io_error,
	        json => undef,
	        raw => '',
	    };
	}

	my ($headers, $response_body) = split(/\r?\n\r?\n/, $raw_response, 2);
	$headers ||= '';
	my ($status) = $headers =~ m{^HTTP/\d+\.\d+\s+(\d+)};
	$status ||= 0;

	my $json;
	if (defined $response_body && $response_body ne '') {
		eval { $json = JSON::PP::decode_json($response_body); 1; };
	}

	return {
		status => $status,
		error => '',
		json => $json,
		raw => $response_body,
	};
}

sub _http_status_code {
	my ($resp) = @_;
	return 0 if ref($resp) ne 'HASH';
	return int($resp->{status} || 0);
}

sub _http_error_text {
	my ($resp) = @_;
	return 'none' if ref($resp) ne 'HASH';
	my $err = _trim(_scalarize($resp->{error}), 220);
	return $err ne '' ? $err : 'none';
}

# ── LOGGED-OUT GATE: block in-game commands while the bot is not in-game ──
# Executing normal commands while disconnected / at char-select produces
# "You must be logged in the game to use this command" error spam.
# char_select / char_create / conf (char slot select) are the ONLY commands
# allowed through while logged out — they are what bring the bot back in-game.
sub _logged_out_execution_block {
	my ($command) = @_;
	# Use the NETWORK state as the in-game signal, NOT $char presence.
	# OpenKore keeps $char populated (stale) through char-select and
	# relogin, so gating on $char alone lets logged-out commands through.
	my $_in_game = ($net && $net->getState() == Network::IN_GAME) ? 1 : 0;
	return 0 if $_in_game;
	# If the network is not connected at all, there is nothing to execute
	# against — but still let the reconnect-family commands through.
	my $root = lc(($command =~ /^\s*(\S+)/)[0] || '');
	return 0 if $root eq '';
	# Always allow the reconnect / char-management family.
	for my $_ok_root ('char_select', 'char_create', 'conf') {
		return 0 if $root eq $_ok_root;
	}
	# Bare 'party'/'guild' info commands are harmless but noisy — block them
	# too so the bot stays quiet while logged out.
	return 1;
}

sub _command_allowed {
	my ($command) = @_;
	my ($root) = $command =~ /^\s*(\S+)/;
	$root = lc($root || '');
	return 0 if $root eq '';

	foreach my $deny (@policy_deny) {
		return 0 if defined $deny && $deny ne '' && $root eq $deny;
	}

	my $mode = lc(_policy('aiSidecarPolicy_mode', 'allowlist'));
	if ($mode eq 'allowlist') {
		foreach my $allow (@policy_allow) {
			return 1 if defined $allow && $allow ne '' && $root eq $allow;
		}
		return 0;
	}

	return 1;
}

my $_last_pro_ro_lockmap_ms = 0;

sub _rewrite_runtime_command {
	my ($command, $metadata, $action_id) = @_;

	# ── COMMITTED-ACTION CONSULTATION ──
	# _committed_actions/_committed_commands are written in _execute_action.
	# Suppress exact duplicates of ONE-SHOT commands (party ops, ai mode
	# toggles, stand/sit/respawn) re-issued within the 30s cooldown window —
	# the sidecar queue may re-emit the same proposal (e.g. after a poll
	# retry), and repeating these is always wrong. Repeating commands like
	# 'attack' are deliberately NOT in this set: they are legitimately
	# re-issued every cycle.
	if (defined $action_id && $action_id ne 'unknown_action') {
		my %_one_shot = map { $_ => 1 } (
			'party create', 'party join', 'party leave',
			'ai auto', 'ai manual', 'stand', 'sit', 'respawn',
		);
		my $_cmd_norm = lc($command || '');
		$_cmd_norm =~ s/\s+/ /g;
		$_cmd_norm =~ s/^\s+|\s+$//g;
		if ($_cmd_norm ne '' && $_one_shot{$_cmd_norm}) {
			my $_now_ms = _now_ms();
			if (exists $_committed_commands{$_cmd_norm} && $_now_ms - $_committed_commands{$_cmd_norm} <= $COMMITTED_ACTION_COOLDOWN_MS) {
				debug "[committed_suppress] $command already executed within cooldown; dropping\n", 'aiSidecarBridge', 2;
				return ('', 'committed_action_duplicate');
			}
		}
	}

	# ── ITEM 602 SUPPRESSION: block 'use 602' / 'use Butterfly Wing' at rewrite level ──
	# Must be FIRST before any cooldown/rewrite logic to prevent "on cooldown" log spam.
	# Item 602 (Butterfly Wing) is managed by the AI system, not auto-use.
	if (lc($command || '') =~ /^(?:use\s+)?602$/ || lc($command || '') =~ /^use\s+butterfly\s+wing/i) {
		return ('', 'item_602_suppressed');
	}

	# ── MOVE REWRITE: context-aware map-name to coordinate conversion ──
	if ($command =~ /^move\s+(\S+)$/i) {
		my $_target = lc($1);
		my $_cur_map = $field ? lc($field->name()) : '';
		$_cur_map =~ s/\.gat$//;
		# ── ISLAND DEADLOCK GATE (fixes EVERY bot, present + future) ──
		# A bot stranded on the Secluded Island (int_land) has NO route to
		# prt_fild05 / any Prontera field — OpenKore emits "Cannot calculate a
		# route from int_land to prt_fild05" and spins forever. The ONLY way out
		# is the (49,57) sailor warp. Any OTHER `move <map>` from the island is
		# a losing directive (from hunting-map defaults, cold-start econ step,
		# survival fallback, etc.) and must be suppressed so it cannot fight the
		# escape. This covers every bot: `move 49 57` is the ONLY move allowed
		# while on int_land (matches the block above). This gate is map-based,
		# so it works for all present AND future bots regardless of which code
		# emitted the bad move.
		if ($_cur_map =~ /^int_land/ && $_target !~ /^49$/) {
			debug "[island_gate] on $_cur_map, suppressing 'move $_target' (only (49,57) escape is routable) -> let island escape proceed\n", 'aiSidecarBridge', 2;
			return ('', 'island_move_suppressed');
		}
		# Secluded Island sailor escape: dedupe so OpenKore routes to the
		# (49,57) OnTouch warp ONCE and actually walks there. Without this,
		# PDCA re-issues `move 49 57` from every horizon (immediate/short/
		# medium) each cycle, cancelling the in-progress route each time —
		# the bot never reaches the warp ("Calculating route... : 49, 57"
		# spam forever). Keyed on the command itself (the escape only ever
		# targets (49,57) from the island), so it works regardless of $field.
		if ($_target =~ /^49$/) {
			my $_now_ms = _now_ms();
			my $_key = 'move_int_land_sailor';
			if (exists $_committed_commands{$_key} && $_now_ms - $_committed_commands{$_key} <= $COMMITTED_ACTION_COOLDOWN_MS) {
				debug "[move_dedupe] Secluded Island sailor move already issued; letting route complete\n", 'aiSidecarBridge', 2;
				return ('', 'move_int_land_sailor_deduped');
			}
			$_committed_commands{$_key} = $_now_ms;
			debug "[move_rewrite] Secluded Island escape move 49 57 (deduped, cooldown=${COMMITTED_ACTION_COOLDOWN_MS}ms)\n", 'aiSidecarBridge', 2;
		}
		# ── ISLAND PORTAL STEP-ON ──
		# The (49,57) #intro_to_izlude is a WARPNPC/portal (int_landX -> izlude).
		# Once the MapRoute fix (3a2db163d) lets the bot walk to it, OpenKore's
		# portal-avoidance keeps it ADJACENT ("Avoiding out of sight actor Portal")
		# instead of stepping ONTO the tile, so the OnTouch warp never fires. When
		# the bot is on an island and close to the warp tile, deliberately step onto
		# the detected portal via `move <portal#>` (cmdMove's portal branch routes the
		# bot ONTO the portal to warp through).
		if ($_cur_map =~ /^int_land/ && $_target =~ /^49$/) {
			my $_px = ($char && $char->{pos}{x}) ? $char->{pos}{x} : 0;
			my $_py = ($char && $char->{pos}{y}) ? $char->{pos}{y} : 0;
			# find a portal at/near (49,57)
			my $_pidx;
			for my $_pi (0..$#main::portalsID) {
				next unless $main::portalsID[$_pi];
				my $_p = $main::portals{$main::portalsID[$_pi]};
				next unless $_p;
				if ($_p->{pos}{x} == 49 && $_p->{pos}{y} == 57) {
					$_pidx = $_pi;
					last;
				}
			}
			if (defined $_pidx && abs($_px-49) <= 4 && abs($_py-57) <= 4) {
				debug "[island_portal] bot at ($_px,$_py) near warp portal idx=$_pidx -> stepping ONTO it to trigger OnTouch\n", 'aiSidecarBridge', 1;
				return ("move $_pidx", 'island_portal_step');
			}
		}
		# Direct portal coordinate - always pass through
		if ($_target eq '22 203') {
			debug "[move_rewrite] portal coordinate 22 203 - passing through\n", 'aiSidecarBridge', 2;
			# Set a 5-second lock to prevent PDCA from interrupting the portal walk
			# This blocks ALL subsequent commands until the bot has time to walk through portal
			$_last_reflex_fire_ms{'portal_walk_lock'} = _now_ms() + 5000;
			return ($command, 'coordinate_move_raw');
		}
		# ── ACADEMY-DOOR WALK LOCK (map-agnostic, data-driven) ──
		# A fresh level-1 weapon-less novice walking to the academy door (the warp
		# into iz_ac01 from ANY town) must not have its route cancelled by the
		# goal-decomposer's zone moves (`move prt_fild08`) or any other emitter.
		# Resolve the academy warp for the CURRENT map from the portal table —
		# exactly what the sidecar's _cold_start_academy_door does — and apply the
		# same portal_walk_lock when the target IS that warp, so conflicting moves
		# are suppressed for the walk window.
		{
			my $_tables_root = '';
			# Resolve repo root by walking up from this file's dir (plugins/aiSidecarBridge/)
			my $_here = __FILE__;
			for (my $i = 0; $i < 6; $i++) {
				$_here =~ s#/[^/]+$##;
				last if -f "$_here/tables/portals.txt";
			}
			$_tables_root = "$_here/tables" if -f "$_here/tables/portals.txt";
			if ($_tables_root ne '' && $_cur_map ne '') {
				my $_academy_door_coords;
				open my $_pfh, '<', "$_tables_root/portals.txt" or undef;
				if ($_pfh) {
					while (my $_pline = <$_pfh>) {
						$_pline =~ s/^\s+|\s+$//g;
						next if $_pline eq '' || $_pline =~ /^#/;
						my @_f = split /\s+/, $_pline;
						next if @_f < 5;
						# portal: from_map x y to_map tx ty
						if (lc($_f[0]) eq $_cur_map && lc($_f[3]) eq 'iz_ac01') {
							$_academy_door_coords = "$_f[1] $_f[2]";
							last;
						}
					}
					close $_pfh;
				}
				if (defined $_academy_door_coords && $_target eq $_academy_door_coords) {
					debug "[move_rewrite] academy-door warp $_academy_door_coords ($_cur_map -> iz_ac01) - locking portal walk\n", 'aiSidecarBridge', 1;
					$_last_reflex_fire_ms{'portal_walk_lock'} = _now_ms() + 5000;
					return ($command, 'coordinate_move_raw');
				}
			}
		}

		# If already on target map, ignore the move (already there)
		if ($_cur_map eq $_target && $_target =~ /^[a-z]+_fild/) {
			debug "[move_rewrite] already on $_target, ignoring\n", 'aiSidecarBridge', 2;
			return ('', 'already_on_target_map');
		}
		# If in Prontera and target is a hunting map, walk to the Prontera-side portal
		my %_portal_coords = (
			'prt_fild05' => 'move 22 203',  # Portal in Prontera to prt_fild05
			# prt_fild08 (the academy farm) — prontera 156,26 -> prt_fild08 170,378.
			'prt_fild08' => 'move 156 26',
		);
		# izlude_c connects ONLY to prt_fild08c (izlude_c 20,98 <-> prt_fild08c 367,212).
		# A bot on izlude_c locked to a prt_fild08 farm variant must cross into
		# prt_fild08c (the actual academy farm) via the izlude_c->prt_fild08c portal,
		# not get sent to prontera's prt_fild08 portal (unroutable from izlude_c).
		if ($_cur_map eq 'izlude_c' && $_target =~ /^prt_fild08/) {
			debug "[move_rewrite] izlude_c -> prt_fild08c portal (20,98)\n", 'aiSidecarBridge', 2;
			return ('move 20 98', 'coordinate_move_raw');
		}
		if ($_cur_map eq 'prontera' && exists $_portal_coords{$_target}) {
			my $_new_cmd = $_portal_coords{$_target};
			debug "[move_rewrite] from Prontera: $command -> $_new_cmd\n", 'aiSidecarBridge', 2;
			$command = $_new_cmd;
			return ($command, 'coordinate_move_raw');
		}
		# Always allow return to town - the heuristic service handles economy timing
		if ($_cur_map =~ /^[a-z]+_fild/ && $_target eq 'prontera') {
		    debug "[move_rewrite] on $_cur_map, allowing return to town\n", 'aiSidecarBridge', 2;
		    return ($command, 'coordinate_move_raw');
		}
	}
	# ── PARTY REWRITE: fix party create/join syntax ──
	if ($command =~ /^party\s+create\s+(.+)$/i) {
		my $_party_name = $1;
		debug "[party_rewrite] $command -> party create $_party_name\n", 'aiSidecarBridge', 2;
		$command = "party create $_party_name";
		return ($command, 'party_command');
	}
	if ($command =~ /^party\s+join\s+(.+)$/i) {
		my $_join_target = $1;
		# If it's a name (not digits), look up the player's account ID
		if ($_join_target !~ /^\d+$/) {
			my $_found_id = 0;
			if ($playersList) {
				for my $_pl (@{$playersList->getItems()}) {
					if (defined $_pl && lc($_pl->{name}) eq lc($_join_target)) {
						$_found_id = $_pl->{ID};
						last;
					}
				}
			}
			if ($_found_id) {
				debug "[party_rewrite] $command -> party join $_found_id (resolved name)\n", 'aiSidecarBridge', 2;
				$command = "party join $_found_id";
			} else {
				debug "[party_rewrite] $command - player '$_join_target' not found nearby, using name\n", 'aiSidecarBridge', 2;
				$command = "party join $_join_target";
			}
		} else {
			debug "[party_rewrite] $command -> party join $_join_target\n", 'aiSidecarBridge', 2;
			$command = "party join $_join_target";
		}
		return ($command, 'party_command');
	}
	# ── TALK REWRITE: fix talknpc dialog commands ──
	if ($command =~ /^talknpc\s+(\d+)\s+(\d+)$/i) {
		debug "[talk_rewrite] $command\n", 'aiSidecarBridge', 2;
		return ($command, 'talknpc_command');
	}
	if ($command =~ /^talk\s+(resp|continue|any|next|close)/i) {
		debug "[talk_rewrite] $command\n", 'aiSidecarBridge', 2;
		return ($command, 'talk_command');
	}
	# ── SELL/STORE REWRITE: fix sell/store commands ──
	if ($command =~ /^sell$/i) {
		debug "[sell_rewrite] $command -> sell auto\n", 'aiSidecarBridge', 2;
		$command = "sell auto";
		return ($command, 'sell_command');
	}
	if ($command =~ /^store$/i) {
		debug "[store_rewrite] $command -> store\n", 'aiSidecarBridge', 2;
		return ($command, 'store_command');
	}
	if ($command =~ /^buy\s+(\d+)\s+(\d+)$/i) {
		my $_buy_id = $1;
		my $_buy_qty = $2;
		# Bare `buy <item> <qty>` requires an OPEN store window; OpenKore
		# errors ("Store Item N does not exist") if the bot isn't in a shop
		# dialog. Potions/weapons are actually purchased by OpenKore's own
		# buyAuto (configured in the profile with npc + npc_steps), which
		# walks to the vendor and opens the shop itself. So a bare `buy`
		# with no dialog open is redundant AND spammy — suppress it and let
		# buyAuto do the real purchase (heuristic_service was emitting this
		# every cycle for Novices, causing endless buy-error spam).
		if (!$_npc_dialog_state{in_dialog}) {
			debug "[buy_suppress] store not open for bare 'buy $_buy_id $_buy_qty' — buyAuto handles it\n", 'aiSidecarBridge', 1;
			return ('', 'buy_suppressed_no_store');
		}
		debug "[buy_rewrite] $command\n", 'aiSidecarBridge', 2;
		return ($command, 'buy_command');
	}
		# Known hunting maps from Prontera
	my $trimmed = _trim(_scalarize($command), 256);
	my $normalized = lc($trimmed || '');
	# PORTAL WALK LOCK: if bot is walking to portal, only block move commands
	$_last_reflex_fire_ms{'portal_walk_lock'} = 0 unless exists $_last_reflex_fire_ms{'portal_walk_lock'};
	my $_pl_check = $_last_reflex_fire_ms{'portal_walk_lock'} || 0;
	if ($_pl_check > 0 && _now_ms() < $_pl_check) {
		# Block move commands AND the stand/ai-auto resets that cancel a
		# coordinate walk mid-route. The hunting domain emits `stand` every
		# cycle while the bot is in TOWN_HUNT; firing it right after a
		# portal/coordinate move (academy-door walk) restarts OpenKore's AI
		# which then random-walks instead of completing the route (2.27
		# defect — the move was dispatched but never walked). Suppress both
		# during the 5s walk lock.
		if ($normalized =~ /^move\s+/ || $normalized eq 'stand' || $normalized eq 'ai auto') {
			debug "[portal_lock] blocking $command - portal walk in progress\n", 'aiSidecarBridge', 2;
			return ('', 'portal_walk_lock');
		}
	}

	$metadata = {} if ref($metadata) ne 'HASH';

	# VENDOR BLOCK: only block talknpc when bot is on a hunting map (not in town)
	if ($normalized =~ /^talknpc\s+/ && $_last_reflex_fire_ms{'block_vendor_until'} && _now_ms() < $_last_reflex_fire_ms{'block_vendor_until'}) {
	    my $_vb_map = lc($char->{map} || '');
	    $_vb_map =~ s/\.gat$//;
	    my @_vb_towns = qw(prontera izlude morocc payon geffen aldebaran comodo);
	    my $_vb_in_town = grep { $_vb_map eq $_ } @_vb_towns;
	    if (!$_vb_in_town) {
	        debug "[vendor_block] blocking talknpc - not in town\n", 'aiSidecarBridge', 1;
	        return ('', 'vendor_blocked_hunting_forced');
	    }
	}
	
	# SIT BLOCK: allow sit when HP is low (let bots regen)
	if ($normalized eq 'sit') {
	    my $_sit_hp = _safe_hp_ratio();
	    if ($_sit_hp >= 0.5) {
	        debug "[sit_block] blocking sit - HP=$_sit_hp is fine\n", 'aiSidecarBridge', 1;
	        return ('', 'sit_blocked_high_hp');
	    }
	    debug "[sit_block] allowing sit - HP=$_sit_hp is low\n", 'aiSidecarBridge', 1;
	    return ($trimmed, 'sit_allowed_low_hp');
	}
	
	# MACRO GUARD: block broken macros that cause syntax errors
	if ($normalized =~ /^macro\s+(reflex_mob_swarm|reflex_pvp_escape|reflex_relog|reflex_survival_escape)/) {
	    debug "[macro_guard] blocking broken macro: $1\n", 'aiSidecarBridge', 1;
	    return ('', 'macro_blocked_broken');
	}
	
	# PARTY JOIN: allow party join from sidecar (config auto-join may not work)
	# Allow party join 1 (accept invite) - this is needed when config auto-join fails
	if ($normalized =~ /^party\s+join\s+1$/) {
	    debug "[party_join] allowing party join 1 (accept invite)\n", 'aiSidecarBridge', 1;
	    return ($trimmed, 'party_join_allowed');
	}
	# Block other party join commands (syntax errors)
	if ($normalized =~ /^party\s+join\s+(.+)$/) {
	    debug "[party_guard] blocking party join - use 'party join 1' to accept invites\n", 'aiSidecarBridge', 1;
	    return ('', 'party_join_blocked_syntax');
	}




	debug "[aiSidecarBridge_DEBUG] rewrite_runtime_command: raw='$command' normalized='$normalized'\n", 'aiSidecarBridge', 0;
	# ── SIT GUARD: block sit commands (bot should never sit) ──
	if ($normalized eq q{sit}) {
	    my $_sit_hp = _safe_hp_ratio();
	    my $_sit_map = lc($char->{map} || '');
	    $_sit_map =~ s/\.gat$//;
	    # Block sit in town unconditionally
	    my @_sit_blocked_maps = qw(prontera izlude morocc payon geffen aldebaran comodo);
	    if (grep { $_sit_map eq $_ } @_sit_blocked_maps) {
	        warning "[sit_guard] blocking sit in town '$_sit_map'\n", 'aiSidecarBridge', 1;
	        return ('', 'sit_blocked_town');
	    }
	    # Block sit on hunting maps unless HP is low (< 50%)
	    if ($_sit_map =~ /^[a-z]+_fild/ && $_sit_hp >= 0.5) {
	        warning "[sit_guard] blocking sit on hunting map '$_sit_map' (HP=$_sit_hp)\n", 'aiSidecarBridge', 1;
	        return ('', 'sit_blocked_hunting');
	    }
		return ($trimmed, 'sit_allowed');
	}
	# NPC DIALOG AUTO-COMPLETION

	# MACRO POTION SPAM FIX: add 5-minute cooldown to emergency potion macros
	if ($normalized =~ /^macro\s+reflex_survival_(red|orange|white)(?:_potion)?$/) {
		my $_potion_type = $1;
		my $_now = _now_ms();
		my $_last_macro = $_last_reflex_fire_ms{"macro_$_potion_type"} || 0;
		if ($_last_macro > 0 && ($_now - $_last_macro) < 300000) {
			warning "[macro] emergency_${_potion_type}_potion on 5min cooldown, skipping\n", 'aiSidecarBridge', 1;
			return ('', 'macro_potion_cooldown');
		}
		$_last_reflex_fire_ms{"macro_$_potion_type"} = $_now;
	}

	# STALE NPC TELEPORT BLOCK: prevent attempts to teleport via non-existent NPCs
	# OpenKore regenerates portalsLOS.txt at runtime, so stale entries keep reappearing
	if ($normalized =~ /^talknpc\s+(\d+)\s+(\d+)/) {
		my $_npc_x = $1;
		my $_npc_y = $2;
		# Known stale NPC coordinates in Prontera (no NPC exists at these locations)
		if (($_npc_x == 156 && $_npc_y == 229) ||
		    ($_npc_x == 157 && ($_npc_y == 40 || $_npc_y == 38 || $_npc_y == 36))) {
			debug "[stale_npc] blocking talknpc to ($_npc_x,$_npc_y) - known stale portal\n", 'aiSidecarBridge', 1;
			return ('', 'stale_npc_blocked');
		}
	}

	# NPC DIALOG AUTO-COMPLETION: rewrite talknpc to include full interaction sequence
	if ($normalized eq 'talknpc 29 207') {
		debug "[talknpc] Kafra Employee at (29,207): auto-completing teleport sequence\n", 'aiSidecarBridge', 1;
		return ("talknpc 29 207 c r2 c r0 c r0 n", 'kafra_teleport_auto');
	}
	if ($normalized eq 'talknpc 156 212') {
		debug "[talknpc] Tool Dealer at (156,212): auto-completing buy sequence\n", 'aiSidecarBridge', 1;
		return ("talknpc 156 212 c r0 n", 'tool_dealer_auto');
	}
	# Secluded Island sailor (#intro_to_izlude WARPNPC): the bot must step
	# onto (49,57) to fire OnTouch, then advance the dialog. For a fresh
	# academy-bound character quest 21008 is NOT active, so the script runs
	# `mes "[Sailor] Let's head towards Izlude!"; close2;` which AUTO-WARPS
	# to izlude once the dialog is advanced — a single `c` closes it and
	# triggers the close2 warp. (If the quest ever is active, the dialog has
	# a select; `talk resp 1` from the cold-start handles that branch.)
	if ($normalized =~ /^talknpc\s+49\s+57/) {
		debug "[talknpc] Secluded Island sailor (49,57): advancing dialog -> close2 warps to Izlude\n", 'aiSidecarBridge', 1;
		return ("talknpc 49 57 c", 'izlude_sailor_auto');
	}

	if ($normalized =~ /^move\s+savepoint$/) {
		return ('respawn', 'move_savepoint_rewritten');
	}

	if ($normalized eq 'move random_walk_seek') {
		if (_ai_already_auto_mode()) {
			return ('', 'random_walk_seek_already_auto');
		}
		return ('ai auto', 'random_walk_seek_rewritten');
	}

	if ($normalized eq 'move') {
		if (_ai_already_auto_mode()) {
			return ('', 'bare_move_already_auto');
		}
		return ('ai auto', 'bare_move_rewritten');
	}

		# Handle 'job_change <profile> <class>' - store assigned job for this bot
	# Leader emits this for all bots; each bot's bridge stores its own assignment.
	my $_jc_trimmed = _trim(_scalarize($command), 256);
	my $_jc_lc = lc($_jc_trimmed || '');
	if ($_jc_lc =~ /^job_change\s+(\S+)\s+(\S+)$/) {
		my $_jc_profile = $1;
		my $_jc_job = ucfirst(lc($2));
		_state_set('assigned_job', $_jc_job);
		warning "[job_change] profile=$_jc_profile job=$_jc_job stored\n", 'aiSidecarBridge', 1;
		return ('', 'job_change_stored');
	}

	# Handle 'mon_control <entry>' - write to mon_control.txt and reload
		# Writes to ALL bot profiles so the setting takes effect for all bots
		if ($normalized =~ /^mon_control\s+(.+)$/) {
			my $_mc_entry = $1;
			_append_mon_control_dedup($_mc_entry);
			# Force reload of mon_control via internal parser (not Commands::run which is unreliable)
		eval {
		    require FileParsers;
		    my $_mc_file = Settings::getMonControlFilename();
		    FileParsers::parseMonControl($_mc_file, \%::mon_control);
		    1;
		};
		warning "[mon_control] applied " . scalar(keys %::mon_control) . " entries (last='$_mc_entry')\n", 'aiSidecarBridge', 1;
		# After applying mon_control, force AI target reselection
		# OpenKore only checks mon_control BEFORE starting an attack, not during.
		# If the bot is currently attacking a now-ignored monster, it must be
		# stopped so the AI re-evaluates targets.
		# Restart AI state machine to force target re-evaluation
		# AI::state(2) restarts the auto-attack system without clearing
		# other AI queues (movement, loot, etc.)
		eval { AI::state(2); 1; };
		warning "[mon_control] AI restarted (state 2) for target reselection\n", 'aiSidecarBridge', 1;
		return ('', 'mon_control_applied');
		}

	# Handle 'set lockMap' - set lockMap to hunting map
	if ($normalized =~ /^set\s+lockMap\s+(.+)$/i) {
	    my $_lm_target = $1;
	    # ── ISLAND LOCKMAP GATE ──
	    # Setting lockMap (e.g. prt_fild05) while stranded on int_land makes
	    # OpenKore spin on "Cannot calculate a route from int_land to prt_fild05"
	    # forever, fighting the seabreaker escape. Suppress ANY lockMap write on
	    # the island so the bot only does the escape. Map-based => all bots.
	    my $_lc_cur = $field ? lc($field->name()) : '';
	    if ($_lc_cur =~ /^int_land/) {
		warning "[island_gate] on int_land, suppressing 'set lockMap $_lm_target' (no route off the island except the (49,57) escape)\n", 'aiSidecarBridge', 1;
		return ('', 'island_lockmap_suppressed');
	    }
	    warning "[set_lockMap] setting lockMap to $_lm_target\n", 'aiSidecarBridge', 1;
	    $::config{'lockMap'} = $_lm_target;
	    $::config{'_sidecar_set_lockMap'} = 1;
	    return ('', 'lockmap_set');
	}

	# ── ISLAND STUCK-LOCKMAP CLEAR (last line of defense) ──
	# A bot stranded on int_land can carry a STALE farm lockMap (e.g. prt_fild05
	# set by an earlier cold-start econ step before it was on the island). That
	# stale value makes OpenKore's hunting brain spin "Cannot calculate a route
	# from int_land to prt_fild05" instead of doing the (49,57) escape. Once we
	# are inside this per-command rewrite (any command while the bot is on
	# int_land), actively WIPE an unreachable lockMap so the only directive left
	# is the sailor escape. Runs for every command issued while on the island, so
	# it self-heals the stale value even if the sidecar never re-sets it.
	my $_lc_field_cur = $field ? lc($field->name()) : '';
	if ($_lc_field_cur =~ /^int_land/ && ($::config{lockMap} || '') ne '' && lc($::config{lockMap}) ne $_lc_field_cur) {
		warning "[island_gate] wiping stale lockMap '$$::config{lockMap}' while on $_lc_field_cur (unreachable; escape-only)\n", 'aiSidecarBridge', 1;
		delete $::config{lockMap};
		delete $::config{'_sidecar_set_lockMap'};
	}
	# Handle set commands: "set <config_key> <value>" -> modify config directly
	# Use $trimmed (original case) to preserve config key case
	# Value can be empty (e.g. "set avoidList " to disable)
	if ($trimmed =~ /^set\s+(\S+)\s*(.*)$/i) {
		my $set_key = $1;
		my $set_val = $2 || '';
		# Find the original case from existing config keys, or use the original case from the command
		my $orig_key = (grep { lc($_) eq lc($set_key) } keys %::config)[0];
		$orig_key = $set_key unless defined $orig_key;

		# STAY IN TOWN DISABLED: heuristic handles all routing decisions
		# HUNTING MAP STICKINESS: heuristic handles all routing decisions
		if (lc($orig_key) eq 'lockmap') {
			my $_target_is_town = $set_val =~ /^(prontera|izlude|morocc|payon|geffen|aldebaran|comodo|umbala|niflheim|rachel|veins|einbroch|lighthalzen|juno|hugel|yuno|amatsu|gonryun|louyang|ayothaya)$/i;
			if ($_target_is_town) {
				my $_actual_map = $field ? $field->name() : '';
				$_actual_map = lc($_actual_map || '');
				$_actual_map =~ s/\.gat$//;
				if ($_actual_map =~ /^[a-z]+_fild/) {
					# Allow town move - heuristic handles economy timing
					$::config{lockMap} = $set_val;
					$_pro_ro_last_lock_set = $set_val;
					$_pro_ro_respawn_ms = _now_ms();
					# AI sequence clearing controlled by heuristic
					return ('', 'config_set_ok');
				}
			}
		}

		my $old_val = $::config{$orig_key};
		# Suppress no-op config_set log when value hasn't changed
		# Use orig_key (lowercase) — OpenKore stores config keys lowercase internally
		if (defined $old_val && $old_val eq $set_val) {
			return ('', 'config_set_ok');
		}
		$::config{$orig_key} = $set_val;
		$::config{"_sidecar_set_$orig_key"} = 1;
		warning "[aiSidecarBridge] config_set: $orig_key = '$set_val' (was " . (defined $old_val ? "'$old_val'" : 'undef') . ")\n", 'aiSidecarBridge', 1;
		return ('', 'config_set_ok');
	}

	# Handle map-name moves: "move <map>" -> set lockMap + ai auto
	if ($normalized =~ /^move\s+(.+)$/) {
		my $target = $1;
		# ── INVALID-DESTINATION GUARD (2026-08-25, spin/leak root cause) ──
		# 'move 0 0' is an invalid destination (map origin is usually a wall).
		# It makes OpenKore A* fail -> route-fail spin -> ~45MB/s RAM leak.
		# Suppress it at the chokepoint so NO emitter (sidecar survival, macro,
		# stuck-detector) can ever send the bot to (0,0).
		if ($target eq '0 0') {
			debug "[invalid_move] suppressing move 0 0 (invalid destination)\n", 'aiSidecarBridge', 1;
			return ('', 'invalid_move_suppressed');
		}
		# Direct portal coordinates - pass through immediately
		if ($target eq '22 203') {
		    return ($trimmed, 'coordinate_move_raw');
		}
		# Coordinate moves (e.g. "move 150 150") - pass through with route loop detection
		if ($target =~ /^\d+\s+\d+$/) {
			# ROUTE LOOP DETECTION: if bot is already at or near the target coordinates,
			# suppress the move to prevent infinite route recalculation.
			# Also suppress if bot is already on a hunting map and target is a portal
			# (the bot is already on the hunting map, no need to go to portal)
			my ($tx, $ty) = split(/\s+/, $target);
						# WALKABILITY SNAP: OpenKore cannot route to an unwalkable tile (a
						# wall), but rAthena warps are 2x2 AREA triggers — the anchor tile the
						# sidecar resolved from portals.txt (e.g. the Academy door at izlude
						# 125,257) may itself be a wall in the client GAT, with the walkable
						# trigger tile on an ADJACENT cell (126,257). Snap the target to the
						# nearest walkable 4-neighbor so the bot actually walks INTO the warp
						# trigger zone. Uses the client Field walkability FACT — no hardcoded
						# coordinate (founder rule).
						if ($field && !$field->isWalkable($tx, $ty)) {
							my ($snap_x, $snap_y, $snap_d);
							for my $_d ([0,1],[0,-1],[1,0],[-1,0],[1,1],[-1,-1],[1,-1],[-1,1]) {
								my ($nx, $ny) = ($tx + $_d->[0], $ty + $_d->[1]);
								next if $nx < 0 || $ny < 0 || $nx >= $field->{width} || $ny >= $field->{height};
								next unless $field->isWalkable($nx, $ny);
								my $d = abs($nx - $tx) + abs($ny - $ty);
								if (!defined $snap_d || $d < $snap_d) { ($snap_x, $snap_y, $snap_d) = ($nx, $ny, $d); }
							}
							if (defined $snap_x) {
								debug "[walkability] target ($tx,$ty) unwalkable -> snap to ($snap_x,$snap_y)\n", 'aiSidecarBridge', 1;
								($tx, $ty) = ($snap_x, $snap_y);
								$target = "$tx $ty";
							}
						}
						my $cx = 0; my $cy = 0;
			# Use the bot's ACTUAL CURRENT position ($char->{pos}), not its pending
			# destination ($char->{pos_to}). If pos_to is the first-choice the dist
			# check below compares the new move target against the bot's CURRENT
			# destination — so re-issuing the same move (e.g. a cold-start walk to a
			# warp tile like the academy door at izlude 125,257) reads dist=0 and gets
			# suppressed as a false "already there", stranding the bot before it ever
			# walks the final tile to trigger the warp. pos_to is the route the bot is
			# ALREADY executing, not where it stands; only pos reflects reality.
			if ($char) {
				if ($char->{pos} && ref $char->{pos} eq 'HASH') { $cx = $char->{pos}{x} || 0; $cy = $char->{pos}{y} || 0; }
				elsif (defined $char->{x}) { $cx = $char->{x}; $cy = $char->{y}; }
				elsif ($char->{pos_to} && ref $char->{pos_to} eq 'HASH') { $cx = $char->{pos_to}{x} || 0; $cy = $char->{pos_to}{y} || 0; }
			}
			# Map-level check: if bot is on a hunting map and target is a portal
			# coordinate, suppress the move — bot is already on the hunting map.
			# Covers BOTH portal exit (367,205 on prt_fild05) AND portal entry (22,203 on prontera).
			# Bot2 got "move 22 203" while on prt_fild05 — those coords don't exist on hunting maps.
			my $_cm = $field ? lc($field->name()) : '';
			$_cm =~ s/\.gat$//;
			if ($_cm =~ /_fild|_dun/i && (($tx == 373 && $ty == 205) || ($tx == 22 && $ty == 203))) {
			        # Check if bot has 0 potions — if so, allow portal move to reach town
			        my $_rp_has_potions = 0;
			        if ($char && @{_char_inventory($char)}) {
			            for my $_rpi (@{_char_inventory($char)}) {
			                next unless $_rpi;
			                my $_rpn = $_rpi->{name} || '';
			                if ($_rpn =~ /potion|herb|fruit|berry|red|orange|white|yellow|blue|green/i) {
			                    $_rp_has_potions = 1;
			                    last;
			                }
			            }
			        }
			        if (!$_rp_has_potions) {
			            # 0 potions on hunting map — allow portal exit to reach town
			            # Set portal walk lock to block other commands while bot walks
			            $_last_reflex_fire_ms{'portal_walk_lock'} = _now_ms() + 5000;
			            debug "[route_loop] 0 potions on $_cm, allowing portal exit ($tx,$ty), lock=5s\n", 'aiSidecarBridge', 1;
			            return ($trimmed, 'coordinate_move_raw');
			        }
			        debug "[route_loop] already on hunting map $_cm, has potions, suppressing portal move to ($tx,$ty)\n", 'aiSidecarBridge', 1;
			        return ('', 'route_loop_suppressed');
			}
			my $dist = sqrt(($cx - $tx)**2 + ($cy - $ty)**2);
			# Suppress ONLY when the bot is essentially ON the target tile (dist < 2,
			# i.e. within the 4-neighbor cell) — a warp trigger (academy door, portal
			# exit) needs the bot to actually STEP on/adjacent to the tile, so a
			# coarse dist < 5 would treat a 4-tile-away warp as "already there" and
			# strand the cold-start bot 4 tiles from the door forever (2.27 defect).
			# Dist 2 still covers the standard "standing on the tile" case.
			if ($dist < 2) {
				debug "[route_loop] already at target ($tx,$ty), current=($cx,$cy) dist=$dist, suppressing\n", 'aiSidecarBridge', 1;
				return ('', 'route_loop_suppressed');
			}
			# Arm the portal-walk lock so a same-batch `stand`/`ai auto` (or the
			# immediate-hunting reset) does NOT cancel this coordinate route before
			# the bot walks. Without this a cold-start academy-door move is
			# dispatched then instantly overwritten by the next action in the batch
			# -> the bot never actually walks onto the warp tile (2.27 defect).
			$_last_reflex_fire_ms{'portal_walk_lock'} = _now_ms() + 5000;
			debug "[route_loop] arming portal_walk_lock (5s) for coordinate move to ($tx,$ty)\n", 'aiSidecarBridge', 1;
			# Return the (possibly snapped) target so the walk actually executes.
			return ("move $target", 'coordinate_move_raw');
		}
		# If already on target map, "move <map>" is a no-op random walk
		# BUT: if bot is in Prontera and target is Prontera, rewrite to portal coords
		# This prevents OpenKore's internal AI from spamming "move prontera" endlessly
		my $_current_map = $field ? lc($field->name()) : '';
		$_current_map =~ s/\.gat$//;
		if ($_current_map eq lc($target)) {
		    # In Prontera, "move prontera" should go to portal (22 203) to reach lockMap
		    # BUT: if bot has 0 potions, block the redirect — bot needs to buy potions first
		    if ($_current_map eq 'prontera') {
		        my $_pr_has_potions = 0;
		        if ($char && @{_char_inventory($char)}) {
		            for my $_pr_gi (@{_char_inventory($char)}) {
		                next unless $_pr_gi;
		                my $_pr_name = $_pr_gi->{name} || '';
		                if ($_pr_name =~ /potion|herb|fruit|berry|red|orange|white|yellow|blue|green/i) {
		                    $_pr_has_potions = 1;
		                    last;
		                }
		            }
		        }
		        if ($_pr_has_potions) {
		            warning "[portal_rewrite] bot in Prontera, rewriting 'move prontera' to 'move 22 203'\n", 'aiSidecarBridge', 1;
		            return ('move 22 203', 'portal_rewrite');
		        } else {
		            warning "[portal_rewrite] bot in Prontera with 0 potions, blocking portal redirect\n", 'aiSidecarBridge', 1;
		            return ('', 'portal_rewrite_blocked_no_potions');
		        }
		    }
		    return ($trimmed, 'coordinate_move_raw');
		}
		# HUNTING MAP GUARD: if bot is on a hunting map, block "move prontera"
		# Heuristic handles all return-to-town logic - other modules should not override
		# EXCEPTION: if bot has 0 potions, allow return to town to buy potions
		my $_guard_has_potions = 0;
		if ($_current_map =~ /^[a-z]+_fild/ && lc($target) eq 'prontera') {
		        # Check if bot has any potions
		        $_guard_has_potions = 0;
			if ($char && @{_char_inventory($char)}) {
				for my $_gi (@{_char_inventory($char)}) {
					next unless $_gi;
					my $_gi_name = $_gi->{name} || '';
					if ($_gi_name =~ /potion|herb|fruit|berry|red|orange|white|yellow|blue|green/i) {
						$_guard_has_potions = 1;
						last;
					}
				}
			}
			if ($_guard_has_potions) {
				warning "[hunting_guard] blocking 'move prontera' - bot is on $_current_map, heuristic handles routing\n", 'aiSidecarBridge', 1;
				return ('ai auto', 'hunting_guard_blocked');
			}
			# No potions - allow return to town AND set lockMap so AI routes to town
			$::config{'lockMap'} = 'prontera';
			warning "[hunting_guard] allowing 'move prontera' - bot has 0 potions on $_current_map\\n", 'aiSidecarBridge', 1;
			return ($trimmed, 'coordinate_move_raw');
			}
			# Set lockMap to target only for hunting maps (not towns)
		# Heuristic handles town routing - bridge should not lock to town
		my $_move_target_is_town = $target =~ /^(prontera|izlude|morocc|payon|geffen|aldebaran|comodo|umbala|niflheim|rachel|veins|einbroch|lighthalzen|juno|hugel|yuno|amatsu|gonryun|louyang|ayothaya)$/i;
		if (!$_move_target_is_town) {
		    $::config{'lockMap'} = $target;
		}
		if (!_ai_already_auto_mode()) {
			return ('ai auto', 'move_rewritten');
		}
				# Already in auto mode: only suppress if target has potions (AI handles routing)
				# If bot has 0 potions on hunting map, allow the move through to override AI route.
				# If target is direct coordinates (contains space), allow through for portal walk.
				my $_move_is_coords = $target =~ /^\d+\s+\d+$/;
				if (!$_move_is_coords && (($_current_map !~ /^[a-z]+_fild/ && $_current_map !~ /_dun/i) || $_guard_has_potions)) {
				    return ('', 'move_already_auto');
				}
		# Bot has 0 potions on hunting map — allow move to town despite auto mode
		return ($trimmed, 'coordinate_move_raw');
	}

	# Handle 'use <item>' -> 'is <item>' with 30s cooldown
	# Extended to 5-minute cooldown when bot has 0 potions total
	if ($normalized =~ /^use\s+(.+)$/) {
		my $item_name = $1;
		my $now_ms = _now_ms();
		my $cooldown_key = "use_$item_name";
		my $last_attempt = $_last_reflex_fire_ms{$cooldown_key} || 0;
		# Check if bot has ANY potions in inventory
		my $total_potions = 0;
		if ($char && @{_char_inventory($char)}) {
			for my $_pi (@{_char_inventory($char)}) {
				if ($_pi && $_pi->{name} =~ /potion|herb|fruit|berry/i) {
					$total_potions += $_pi->{amount} || 1;
				}
			}
		}
		my $cooldown_ms = ($total_potions == 0) ? 300000 : 30000;
		if ($last_attempt > 0 && ($now_ms - $last_attempt) < $cooldown_ms) {
			warning "[use] item '$item_name' on cooldown, skipping\n", 'aiSidecarBridge', 1;
			return ('', "use_item_cooldown_$item_name");
		}
		my $found = 0;
		if ($char && @{_char_inventory($char)}) {
			for my $item (@{_char_inventory($char)}) {
				if ($item && lc($item->{name}) eq lc($item_name)) {
					$found = 1;
					last;
				}
			}
		}
		if (!$found) {
			# ── POTION FALLBACK (survivability-critical) ──
			# The sidecar/reflex may emit "use red_potion" (its preferred heal) while
			# the bot only owns a different heal (e.g. Novice Potion 569 x300 — the
			# academy starter kit). Without this fallback the bot DIED with potions
			# unused (verified live: 34 deaths, 0 heals executed). If the requested
			# item is a heal and any potion exists, substitute the best available one.
			my $is_heal_req = ($item_name =~ /potion|herb|fruit|berry|red|orange|white|yellow|blue|green|grape/i);
			if ($is_heal_req) {
				my ($fallback_name) = _best_available_heal_name($char);
				if ($fallback_name) {
					$_last_reflex_fire_ms{$cooldown_key} = $now_ms;
					my $ok2 = eval { Commands::run("is $fallback_name"); 1 };
					debug "[use] '$item_name' not owned -> falling back to '$fallback_name' (${\\($ok2 ? 'OK' : 'FAILED')})\n", 'aiSidecarBridge', 1;
					return ('', $ok2 ? "use_item_fallback_$fallback_name" : "use_item_fallback_failed_$fallback_name");
				}
			}
			$_last_reflex_fire_ms{$cooldown_key} = $now_ms;
			warning "[use] item '$item_name' not in inventory, skipping (cooldown ${\\(int($cooldown_ms/1000))}s)\n", 'aiSidecarBridge', 1;
			return ('', "use_item_not_found_$item_name");
		}
		my $ok = eval { Commands::run("is $item_name"); 1 };
		return ('', $ok ? "use_item_$item_name" : "use_item_failed_$item_name");
	}

	# Handle 'equip <nameID>' -> 'eq <item name>' (nameID -> inventory name)
	# OpenKore's cmdEquip treats a NUMERIC arg as an inventory BINID index, not the
	# item's nameID (item_db id). The sidecar emits `equip 1243` (item_db id) which
	# cmdEquip fails to find -> the weapon never equips -> the bot fights with bare
	# fists (Dmg 1-2, verified live). Resolve the nameID to the owned item's NAME.
	# ALSO: this fork registers the command as `eq`, NOT `equip` — any raw
	# `equip ...` is "Unknown command" (verified live). Rewrite both forms.
	if ($normalized =~ /^equip\s+(\d+)$/ && $char) {
		my $_want_id = $1;
		my $_item = undef;
		for my $_eq (@{_char_inventory($char)}) {
			next unless ref($_eq);
			if ((($_eq->{nameID} // 0) + 0) == ($_want_id + 0)) {
				$_item = $_eq;
				last;
			}
		}
		if ($_item) {
			my $_eq_name = $_item->{name} || '';
			# This fork's command is `eq` (NOT `equip` — "Unknown command").
			# Equip by NAME only: cmdEquip resolves names via Actor::Item::get,
			# and the fork's type/type_equip mapping is inverted vs stock so
			# slot-qualified forms misfire (verified live).
			debug "[equip] nameID $_want_id -> '$_eq_name'\n", 'aiSidecarBridge', 1;
			my $_eq_ok = eval { Commands::run("eq $_eq_name"); 1 };
			return ('', $_eq_ok ? "equip_ok_$_eq_name" : "equip_failed_$_eq_name");
		}
		warning "[equip] item nameID $_want_id not in inventory, skipping\n", 'aiSidecarBridge', 1;
		return ('', "equip_item_not_found_$_want_id");
	}

	# Handle 'equip <name>' (sidecar name-based emits) -> 'eq <name>'
	# This fork registers `eq` only; 'equip' → "Unknown command" (verified live).
	if ($normalized =~ /^equip\s+(.+)$/ && $char) {
		my $_eq_arg = $1;
		debug "[equip] rewrite 'equip $_eq_arg' -> 'eq $_eq_arg'\n", 'aiSidecarBridge', 1;
		my $_eq_ok = eval { Commands::run("eq $_eq_arg"); 1 };
		return ('', $_eq_ok ? "equip_ok_$_eq_arg" : "equip_failed_$_eq_arg");
	}

	# Handle sit/stand
	if ($normalized eq 'sit') {
		my $_actual_map = $field ? $field->name() : '';
		$_actual_map = lc($_actual_map || '');
		$_actual_map =~ s/\.gat$//;
		if ($_actual_map =~ /^[a-z]+_fild/) {
			my $_hp_ratio = _safe_hp_ratio();
			if ($_hp_ratio >= 0.01) {
				debug "[sit_guard] on hunting map '$_actual_map', blocking sit (HP=$_hp_ratio >= 0.15)\n", 'aiSidecarBridge', 1;
				return ('', 'sit_blocked_hunting');
			}
		}
		return ($trimmed, 'sit_allowed');
	}

	my $rewrite_kind = '';

	# Handle 'attack_skill' -> rewrite to use_skill or ignore basic_attack
	if ($normalized =~ /^attack_skill\s+(.+)$/) {
		my $_skill_name = $1;
		if ($_skill_name eq 'basic_attack') {
			# Basic attack is handled by auto-attack, no need to send command
			return ('', 'attack_skill_basic_attack_ignored');
		}
		# Rewrite attack_skill <name> to use_skill <name>
		$normalized = "use_skill $_skill_name";
		$rewrite_kind = 'attack_skill_delegated';
		return ($normalized, $rewrite_kind);
	}

	# Handle raw 'skill <name>' -> rewrite to 'use_skill <name>' (strip level)
	if ($normalized =~ /^skill\s+(.+)$/) {
		my $_skill_name = $1;
		# Strip trailing level number if present (e.g. "Bash 10" -> "Bash")
		$_skill_name =~ s/\s+\d+$//;
		$normalized = "use_skill $_skill_name";
		$rewrite_kind = 'skill_rewritten';
		return ($normalized, $rewrite_kind);
	}

	# Handle 'party join <name>' -> rewrite to 'party join 1' (accept invite)
	if ($normalized =~ /^party\s+join\s+(.+)$/) {
		$normalized = 'party join 1';
		$rewrite_kind = 'party_join_rewritten';
		return ($normalized, $rewrite_kind);
	}

	# Handle 'party request <name>' -> send name directly (works across maps)
	if ($normalized =~ /^party\s+request\s+(.+)$/i) {
		my $_req_name = $1;
		# Resolve profile name to character name using known mapping
		my $_char_name = $_req_name;
		my %_profile_to_char = (
			# Dynamic mapping built from all_bots and config files
		);
		if (exists $_profile_to_char{lc($_req_name)}) {
			$_char_name = $_profile_to_char{lc($_req_name)};
		}
		warning "[party_request] requesting '$_char_name' (profile=$_req_name)\n", 'aiSidecarBridge', 1;
		$command = "party request $_char_name";
		$rewrite_kind = 'party_request_rewritten';
		return ($command, $rewrite_kind);
	}
	# Handle 'party join 1' (accept invite) - already handled above
	# Handle 'party create' with name
	if ($normalized =~ /^party\s+create\s+(.+)$/) {
		my $_party_name = $1;
		warning "[party_create] creating party $_party_name\n", 'aiSidecarBridge', 1;
		$command = "party create $_party_name";
		$rewrite_kind = 'party_create_rewritten';
		return ($command, $rewrite_kind);
	}

	# Handle 'party create' -> rewrite
	if ($normalized eq 'party create') {
		$rewrite_kind = 'party_create_rewritten';
		return ($normalized, $rewrite_kind);
	}

	# Handle 'party share exp' -> rewrite
	if ($normalized eq 'party share exp' || $normalized eq 'party share') {
		$normalized = 'party share exp';
		$rewrite_kind = 'party_share_rewritten';
		return ($normalized, $rewrite_kind);
	}

	# Handle 'stand' -> always allow (prevents sitting)
	# Track death - allow return to town after death
	if ($normalized eq 'stand') {
		$rewrite_kind = 'stand_allowed';
		return ($normalized, $rewrite_kind);
	}
	# Track player death for return-to-town allowance
	if ($normalized eq 'died' || $normalized eq 'death') {
	    $_last_reflex_fire_ms{'player_died'} = _now_ms();
	    return ($normalized, 'death_tracked');
	}

	# Handle 'ai auto' -> rewrite
	if ($normalized eq 'ai auto') {
		$rewrite_kind = 'ai_auto_rewritten';
		return ($normalized, $rewrite_kind);
	}

	# Handle 'ai manual' -> rewrite
	if ($normalized eq 'ai manual') {
		$rewrite_kind = 'ai_manual_rewritten';
		return ($normalized, $rewrite_kind);
	}

	# Handle 'talknpc' commands (including atomic with embedded dialog)
	if ($normalized =~ /^talknpc\s+(.+)$/) {
		$rewrite_kind = 'talknpc_rewritten';
		return ($trimmed, $rewrite_kind);
	}

	# Handle 'talk' commands
	if ($normalized =~ /^talk\s+(.+)$/) {
		$rewrite_kind = 'talk_rewritten';
		return ($normalized, $rewrite_kind);
	}

	# Handle 'buy' commands
	if ($normalized =~ /^buy\s+(.+)$/) {
		$rewrite_kind = 'buy_rewritten';
		return ($normalized, $rewrite_kind);
	}

	# Handle 'sell' commands
	if ($normalized =~ /^sell\s+(.+)$/) {
		$rewrite_kind = 'sell_rewritten';
		return ($normalized, $rewrite_kind);
	}

	# Handle 'use_item' commands
	if ($normalized =~ /^use_item\s+(.+)$/) {
		$rewrite_kind = 'use_item_rewritten';
		return ($normalized, $rewrite_kind);
	}

	# Handle 'use_skill' commands
	if ($normalized =~ /^use_skill\s+(.+)$/) {
		$rewrite_kind = 'use_skill_rewritten';
		return ($normalized, $rewrite_kind);
	}

	# Handle 'skills add' commands
	if ($normalized =~ /^skills?\s+add\s+(\d+)$/) {
		$rewrite_kind = 'skills_add_rewritten';
		return ($normalized, $rewrite_kind);
	}

	# Handle 'stat_add' commands - rewrite to 'st add' (correct OpenKore command)
	if ($normalized =~ /^stat_add\s+(.+)$/) {
		my $_stat_name = $1;
		debug "[stat_add_rewrite] $normalized -> st add $_stat_name\n", 'aiSidecarBridge', 2;
		$command = "st add $_stat_name";
		$rewrite_kind = 'stat_add_rewritten';
		return ($command, $rewrite_kind);
	}

	# Handle 'move' commands (already handled above in guard section)
	# This is a fallthrough for commands that don't need special handling

	# Handle 'teleport auto' -> rewrite to skill or ai auto
	if ($normalized eq 'teleport' || $normalized eq 'teleport auto') {
		my $has_teleport = 0;
		if ($char && $char->{skills}) {
			if (ref($char->{skills}) ne 'ARRAY') { return ('', 'skills_add_no_skills'); }
			for my $skill (@{$char->{skills}}) {
				if ($skill && lc($skill->{name}) eq 'al_teleport') {
					$has_teleport = 1;
					last;
				}
			}
		}
		if ($has_teleport) {
			return ('ss AL_TELEPORT', 'teleport_rewritten');
		}
		return ('ai auto', 'teleport_fallback_auto');
	}

	# Handle 'skills add' / 'skills_add'
	# OpenKore expects: skills add <skill_id> (numeric)
	# If we get skills_add <name> <level>, try to extract the numeric ID
	if ($normalized =~ /^skills?\s*add\s+(\d+)$/) {
		my $skill_points = $1;
		# Check if bot has any skill points available
	} elsif ($normalized =~ /^skills?\s*add\s+(\w+)\s+(\d+)$/) {
		# skills_add <name> <level> -> try to use the level as skill ID (fallback)
		my $_skill_id = $2;
		warning "[skills_add] rewriting skills_add $1 $2 -> skills add $_skill_id\n", 'aiSidecarBridge', 1;
		$normalized = "skills add $_skill_id";
		# $char->{skills} may be a hash or array depending on OpenKore version
		my $has_skill_points = 0;
		if ($char && $char->{skills} && ref($char->{skills}) eq 'ARRAY') {
			for my $skill (@{$char->{skills}}) {
				if ($skill && ref($skill) eq 'HASH' && $skill->{level} < $skill->{max} && $2 > 0) {
					$has_skill_points = 1;
					last;
				}
			}
		}
		if (!$has_skill_points) {
			debug "[skills_add] no skill points available, skipping\n", 'aiSidecarBridge', 1;
			return ('', 'skills_add_no_points');
		}
		my $now_ms = _now_ms();
		my $last_skills_add = $_last_reflex_fire_ms{'skills_add'} || 0;
		if ($last_skills_add > 0 && ($now_ms - $last_skills_add) < 30000) {
			debug "[skills_add] on cooldown, skipping\n", 'aiSidecarBridge', 1;
			return ('', 'skills_add_cooldown');
		}
		$_last_reflex_fire_ms{'skills_add'} = $now_ms;
		return ($trimmed, 'skills_add_allowed');
	}

	# Handle 'ai auto'
	if ($normalized eq 'ai auto') {
		# Always pass through - re-triggers AI state machine for continuous combat
		return ($trimmed, 'ai_auto_ok');
	}

	# Handle 'stand'
	if ($normalized eq 'stand') {
		return ($trimmed, 'stand_ok');
	}

	# Handle 'talknpc' (without auto-completion)
	if ($normalized =~ /^talknpc\s+(\d+)\s+(\d+)/) {
		return ($trimmed, 'talknpc_ok');
	}

	# Handle 'talk' commands
	if ($normalized =~ /^talk\s+(cont|resp|no|\d+)/) {
		return ($trimmed, 'talk_ok');
	}

	# ── USE_ITEM REWRITE: OpenKore uses 'is <item>' for item-on-self ──
	# The macro system may emit 'use_fly_wing', 'use_butterfly_wing', or 'use <item_id>'
	if ($normalized =~ /^use_fly_wing\s*$/i) {
		debug "[use_item_rewrite] $trimmed -> is 602\n", 'aiSidecarBridge', 2;
		return ('is 602', 'use_item_fly_wing');
	}
	if ($normalized =~ /^use_butterfly_wing\s*$/i) {
		debug "[use_item_rewrite] $trimmed -> is 602\n", 'aiSidecarBridge', 2;
		return ('is 602', 'use_item_butterfly_wing');
	}
	if ($normalized =~ /^use\s+(\d+)\s*$/i) {
		my $_item_id = $1;
		debug "[use_item_rewrite] $trimmed -> is $_item_id\n", 'aiSidecarBridge', 2;
		return ("is $_item_id", 'use_item_rewritten');
	}

	# ── CHAR_CREATE: OpenKore's Commands::run doesn't support character creation ──
	# We intercept 'char_create <slot> "name" ...'
	# and call Misc::createCharacter() directly.
	if ($normalized =~ /^char_create\s+/) {
		return ($trimmed, 'char_create');
	}

	# ── CHAR_SELECT: OpenKore's cmdCharSelect requires IN_GAME, not char select screen ──
	# After char_create, the bridge auto-enters the game. This command is redundant.
	if ($normalized =~ /^char_select\s+/i) {
		return ($trimmed, 'char_select_handled');
	}

	# Default: pass through
	return ($trimmed, 'passthrough');

}
sub _ai_already_auto_mode {
	my $state = eval { AI::state() };
	return 0 if $@;
	my $auto = eval { AI::AUTO() };
	return 0 if $@;
	return ($state == $auto) ? 1 : 0;
}

sub _meta {
	my ($bot_id) = @_;
	return {
		contract_version => _cfg('aiSidecar_contractVersion', 'v1'),
		emitted_at => _iso_now(),
		trace_id => _trace_id(),
		source => _cfg('aiSidecar_source', 'openkore-bridge'),
		bot_id => $bot_id,
	};
}

sub _active_control_folder {
	my $folder = '';
	eval {
		if (ref(\@Settings::controlFolders) eq 'ARRAY' && @Settings::controlFolders) {
			$folder = $Settings::controlFolders[0] || '';
		}
	};
	$folder = _trim(_scalarize($folder), 220);
	return $folder ne '' ? $folder : 'control';
}

sub _bot_id {
	my $override_bot_id = _trim(_scalarize(_cfg('aiSidecar_botIdOverride', '')), 128);
	if ($override_bot_id =~ /^([^:]+):(.+)$/) {
		my $override_master = _normalize_identity_part($1, '');
		my $override_identity = _normalize_identity_part($2, '');
		if ($override_master ne '' && $override_identity ne '') {
			return "$override_master:$override_identity";
		}
	}

	my $cfg_master = _cfg('aiSidecar_master', '');
	# Force override: if aiSidecar_master is set, use it instead of runtime master
	my $master;
	if ($cfg_master ne '') {
		$master = _normalize_identity_part($cfg_master, 'unknown_master');
	} else {
		$master = _normalize_identity_part($config{master}, 'unknown_master');
	}
	# ALWAYS use username (profile name) as identity - matches heuristic bot_id format
	# Using char_name causes bot_id mismatch: bridge polls with master:char_name
	# but heuristic enqueues actions for master:profile
	my $username = _normalize_identity_part($config{username}, 'unknown_user');
	return "$master:$username";
}

sub _normalize_identity_part {
	my ($value, $default) = @_;
	$value = _trim(_scalarize($value), 64);
	$value =~ s/^\s+//;
	$value =~ s/\s+$//;
	$value =~ s/\s+/ /g;
	return $default if !defined $value || $value eq '';
	return $value;
}

sub _exp_backoff_ms {
	my ($failures, $base_ms, $max_ms) = @_;
	$failures = 0 + ($failures || 0);
	$base_ms = 0 + ($base_ms || 1);
	$max_ms = 0 + ($max_ms || $base_ms);
	$base_ms = 1 if $base_ms < 1;
	$max_ms = $base_ms if $max_ms < $base_ms;

	return $base_ms if $failures <= 0;
	my $power = $failures - 1;
	$power = 8 if $power > 8;
	my $factor = 2 ** $power;
	my $delay = int($base_ms * $factor);
	$delay = $max_ms if $delay > $max_ms;
	return $delay;
}

sub _poll_failure_delay_ms {
	return _exp_backoff_ms(
		$consecutive_poll_failures,
		_cfg_int('aiSidecar_pollFailureBackoffBaseMs', 600),
		_cfg_int('aiSidecar_pollFailureBackoffMaxMs', 6000),
	);
}

sub _event_ingest_failure_delay_ms {
	return _exp_backoff_ms(
		$consecutive_v2_event_failures,
		_cfg_int('aiSidecar_eventIngestFailureBackoffBaseMs', 1000),
		_cfg_int('aiSidecar_eventIngestFailureBackoffMaxMs', 10000),
	);
}

sub _cfg {
	my ($key, $default) = @_;
	return $default if !exists $bridge_cfg{$key};
	return $default if !defined $bridge_cfg{$key};
	return $default if $bridge_cfg{$key} eq '';
	return $bridge_cfg{$key};
}

sub _policy {
	my ($key, $default) = @_;
	return $default if !exists $bridge_policy{$key};
	return $default if !defined $bridge_policy{$key};
	return $default if $bridge_policy{$key} eq '';
	return $bridge_policy{$key};
}

sub _cfg_int {
	my ($key, $default) = @_;
	my $value = _cfg($key, $default);
	return $default if !defined $value || $value !~ /^-?\d+$/;
	return int($value);
}

sub _cfg_bool {
	my ($key, $default) = @_;
	my $value = _cfg($key, $default ? 1 : 0);
	return ($value && $value =~ /^(?:1|true|yes|on)$/i) ? 1 : 0;
}


# -- AI mode debounce -- prevent rapid auto/manual oscillation --
sub _toggle_ai_mode {
	my ($mode) = @_;
	return if !$mode;
	return if $mode eq $_last_ai_mode;
	my $now = _now_ms();
	return if ($now - $_last_ai_toggle_ms) < 10000;
	$_last_ai_toggle_ms = $now;
	$_last_ai_mode = $mode;
	eval { require AI; if ($mode eq 'auto') { AI::state(2) } elsif ($mode eq 'manual') { AI::state(1) } else { AI::state(0) }; 1 };
}

sub _trim {
	my ($value, $max_len) = @_;
	$value = '' if !defined $value;
	$max_len = 0 + ($max_len || 0);
	return $value if $max_len <= 0 || length($value) <= $max_len;
	return substr($value, 0, $max_len);
}

sub _trace_id {
	my $r = int(rand(0xFFFFFF));
	return sprintf('%x-%x', int(time() * 1000), $r);
}

sub _iso_now {
	my $t = time();
	my @g = gmtime($t);
	my $frac_raw = $t - int($t);
	$frac_raw = 0.0 if $frac_raw < 0.0;
	$frac_raw = 0.999 if $frac_raw >= 1.0;
	my $frac = sprintf('%.3f', $frac_raw);
	$frac = '.999' if $frac eq '1.000';
	$frac =~ s/^0//;
	return sprintf(
		'%04d-%02d-%02dT%02d:%02d:%02d%sZ',
		$g[5] + 1900,
		$g[4] + 1,
		$g[3],
		$g[2],
		$g[1],
		$g[0],
		$frac,
	);
}

sub _now_ms {
	return int(time() * 1000);
}

sub _throttled_warning {
	return;
	# [SIDECAR] suppressed - noise reduction
	my ($key, $msg) = @_;
	my $now = _now_ms();
	my $interval = 10_000;
	my $last = $last_warn_at_ms{$key} || 0;
	if ($now - $last >= $interval) {
		warning "$msg\n";
		$last_warn_at_ms{$key} = $now;
	}
}

sub _calc_distance {
	my ($actor, $char) = @_;
	return undef unless $actor && $char;
	my $ax = undef;
	my $ay = undef;
	if (defined $actor->{pos_to} && ref $actor->{pos_to} eq 'HASH') {
		$ax = $actor->{pos_to}{x};
		$ay = $actor->{pos_to}{y};
	} elsif (defined $actor->{pos} && ref $actor->{pos} eq 'HASH') {
		$ax = $actor->{pos}{x};
		$ay = $actor->{pos}{y};
	}
	return undef unless defined $ax && defined $ay;
	my $cx = $char->{pos_to}{x} || $char->{pos}{x} || 0;
	my $cy = $char->{pos_to}{y} || $char->{pos}{y} || 0;
	return int(sqrt(($ax - $cx)**2 + ($ay - $cy)**2) + 0.5);
}

# ── Bridge-level emergency reflexes (sub-5ms, pure Perl, no LLM/network) ──
# Each reflex has its own cooldown tracked in %_reflex_last_fired.
# In-game actions use Commands::run(). Sidecar alerts use _http_post_json().
# All reflexes include human-like randomization to avoid detection patterns.
# NOTE: All shared state variables are package-level (not lexical inside bare block)
# to avoid Perl "will not stay shared" closure warnings. The bare block { } below
# is for organizational scoping only — state vars live at package level.
our %_reflex_last_fired;
# @_heal_items / @_heal_skills are declared ONCE at package level (line ~1226,
# on_post_bulk_config_modify) — do NOT redeclare here (Perl warns
# "our variable ... redeclared"). These are the same package arrays.
our $_heal_cache_last_update_ms = 0;
our $_last_prontera_recovery_ms = 0;

{
	# Human-like randomization helpers
	sub _human_delay_ms {
		# Simulate human reaction time: 150-400ms base + random variance
		return int(rand(250)) + 150;
	}

	sub _jitter_cooldown {
		my ($base_ms, $jitter_pct) = @_;
		$jitter_pct ||= 0.3;
		# Add ±30% random jitter to cooldown
		return int($base_ms * (1.0 + (rand() * 2 - 1) * $jitter_pct));
	}

	sub _should_fire_reflex {
		my ($last_fired, $cooldown_ms) = @_;
		my $now = _now_ms();
		my $jittered = _jitter_cooldown($cooldown_ms);
		return ($now - $last_fired) >= $jittered;
	}

	sub _random_action_delay {
		# Add per-bot profile-based random delay before action to look human
		my $delay_ms = _human_heal_delay_ms();
		select(undef, undef, undef, $delay_ms / 1000.0) if $delay_ms > 0;
	}

	# ── Cached healing resources (from sidecar config push) ──
	sub _update_heal_cache {
		my $now = _now_ms();
		# Guard: initialize if undef (first call or after reload)
		$_heal_cache_last_update_ms = 0 if !defined $_heal_cache_last_update_ms;
		return if $now - $_heal_cache_last_update_ms < 250;
		$_heal_cache_last_update_ms = $now;

		# Read from sidecar-pushed config (comma-separated lists)
		# Format: "aegis:qty:heal_hp:heal_sp:weight,aegis2:qty:..."
		# OR plain item names: "Red Potion,Orange Potion,White Potion"
		my $items_str = _cfg('aiSidecar_healItems', 'Red Potion,Orange Potion,White Potion');
		@_heal_items = split /,/, $items_str;
		@_heal_items = grep { $_ ne '' } @_heal_items;
		
		# Parse aegis:qty:heal_hp:heal_sp:weight format → extract aegis name
		# Also strip quantity suffix from plain names (e.g. "Red Potion:10" → "Red Potion")
		@_heal_items = map {
			my $entry = $_;
			# If entry contains colons, it's aegis:qty:heal_hp:heal_sp:weight format
			if ($entry =~ /:/) {
				my @parts = split /:/, $entry;
				# First part is aegis name (underscore-separated) — convert to display name
				my $aegis = $parts[0];
				$aegis =~ s/_/ /g;
				$aegis;
			} else {
				$entry;
			}
		} @_heal_items;

		my $skills_str = _cfg('aiSidecar_healSkills', '');
		@_heal_skills = split /,/, $skills_str;
		@_heal_skills = grep { $_ ne '' } @_heal_skills;
	}

sub _check_bridge_reflexes {
	# STRIPPED BRIDGE: only emergency sit at <10% HP
	# All other reflexes (heal, flee, teleport, auto-sit, etc.) are handled
	# by the sidecar's heuristic service and PDCA loop.
	# The bridge only does: snapshot forwarding, command execution, emergency sit.
	my $now = _now_ms();
	return if !$char;

	# ── CONNECTION STATE GUARD ──
	if (!$net || $net->getState() != Network::IN_GAME) {
		return;
	}

	# ── CAN'T-SIT DETECTION ──
	state $_can_sit_checked = 0;
	state $_can_sit = 1;
	state $_last_can_sit_recheck_ms = 0;
	if (!$_can_sit_checked || $now - $_last_can_sit_recheck_ms > 60000) {
		$_can_sit_checked = 1;
		$_last_can_sit_recheck_ms = $now;
		my $_basic_skill_lv = 0;
		if ($char && $char->{skills} && ref $char->{skills} eq 'ARRAY') {
			for my $_sk (@{$char->{skills}}) {
				next if !$_sk;
				my $_sk_name = $_sk->{name} || $_sk->{skillName} || '';
				if ($_sk_name =~ /^basic_skill$/i || $_sk_name =~ /^basic skill$/i || $_sk_name eq 'NV_BASIC') {
					$_basic_skill_lv = $_sk->{lv} || $_sk->{level} || 0;
					last;
				}
			}
		}
		if ($_basic_skill_lv > 0 && $_basic_skill_lv < 3) {
			$_can_sit = 0;
		} elsif ($_basic_skill_lv >= 3) {
			$_can_sit = 1;
		}
	}

	# ── RECONNECT COOLDOWN ──
	if ($last_disconnect_at_ms > 0) {
		my $_time_since_disconnect = $now - $last_disconnect_at_ms;
		if ($_time_since_disconnect < 10000) {
			my $_rc_hp = $char->{hp} || 0;
			my $_rc_hp_max = $char->{hp_max} || 1;
			my $_rc_hp_ratio = ($_rc_hp_max > 0) ? $_rc_hp / $_rc_hp_max : 1;
			if ($_rc_hp_ratio < 0.10 && $_can_sit) {
				if ($AI::AI != 2) {
					eval { Commands::run("sit"); 1 };
				}
			}
			return;
		}
	}

	# ── TRAVEL MODE ──
	my $_ai_seq_top = @ai_seq ? $ai_seq[0] : '';
	my $_is_moving = $_ai_seq_top =~ /^(?:route|move|follow)/ ? 1 : 0;
	state $_travel_mode_until_ms = 0;
	if ($_is_moving) {
		$_travel_mode_until_ms = $now + 60000 if $_travel_mode_until_ms < $now;
		if ($now < $_travel_mode_until_ms) {
			my $_travel_hp = $char->{hp} || 0;
			my $_travel_hp_max = $char->{hp_max} || 1;
			my $_travel_hp_ratio = ($_travel_hp_max > 0) ? $_travel_hp / $_travel_hp_max : 1;
			if ($_travel_hp_ratio < 0.10) {
				if ($AI::AI != 2) {
					eval { Commands::run("sit"); 1 };
				}
			}
			return;
		}
	} else {
		$_travel_mode_until_ms = 0;
	}

	# ── EMERGENCY SIT at <10% HP ──
	my $hp = $char->{hp} || 0;
	my $hp_max = $char->{hp_max} || 1;
	my $hp_ratio = ($hp_max > 0) ? $hp / $hp_max : 1;
	if ($hp_ratio < 0.10 && $hp > 0 && $_can_sit) {
		my $_ai_top = @ai_seq ? $ai_seq[0] : '';
		if ($_ai_top ne 'sit') {
			debug "[aiSidecarBridge] emergency_sit: HP=$hp/$hp_max < 10%, sitting\n", 'aiSidecarBridge', 1;
			eval { Commands::run("sit"); 1 };
		}
	}

	# ── ESCAPE REFLEX: detect escape conditions ──
	# Detects teleport/escape from AI sequence transitions, HP spike recovery,
	# or sidecar "escape" commands. Posts an event and sets survival mode.
	my $_ai_top_reflex = @ai_seq ? $ai_seq[0] : '';
	my $_is_escape_ai = $_ai_top_reflex =~ /^(?:teleport|escape|skill_use\s+teleport)/i ? 1 : 0;

	# Track previous AI top to detect transition into escape
	state $_last_escape_ai_top = '';
	state $_last_escape_fire_ms = 0;

	if ($_is_escape_ai && $_last_escape_ai_top ne $_ai_top_reflex) {
		# Entering escape state — fire once per transition
		my $_escape_cooldown_ms = _cfg_int('aiSidecar_reflexEscapeCooldownMs', 5000);
		if ($now - $_last_escape_fire_ms > $_escape_cooldown_ms) {
			$_last_escape_fire_ms = $now;
			$_last_escape_ai_top = $_ai_top_reflex;

			# Post escape event to sidecar
			_post_event({
				kind => 'bridge_reflex',
				reflex => 'escape',
				severity => 'warning',
				text => "escape reflex triggered: AI sequence $_ai_top_reflex",
				ai_seq => $_ai_top_reflex,
				hp => $hp,
				hp_max => $hp_max,
				map => _safe_field_map() || '',
			});
			debug "[aiSidecarBridge] escape_reflex: detected escape via AI seq=$_ai_top_reflex\n", 'aiSidecarBridge', 1;

			# Force AI to manual to prevent re-engagement during escape
			if (defined $AI::AI && $AI::AI == 2) {
				eval { require AI; AI::state(1); 1; };
			}

			# Set survival mode cooldown (60s) to prevent immediate re-hunt
			$_survival_mode_until_ms = $now + 60000 if $_survival_mode_until_ms < $now;
		}
	} elsif (!$_is_escape_ai) {
		# Reset tracking when no longer in escape state
		$_last_escape_ai_top = '';
	}

	# ── HP SPIKE ESCAPE DETECTION ──
	# Detect when HP drops from >30% to <10% then recovers to >30% within 5s.
	# This pattern strongly suggests the bot used an escape teleport.
	state $_escape_spike_hp = -1;
	state $_escape_spike_t_ms = 0;
	my $_hp_pct = $hp_max > 0 ? int($hp * 100 / $hp_max) : 100;

	if ($_hp_pct < 10 && $hp > 0) {
		# Entered danger zone
		if ($_escape_spike_hp < 0) {
			$_escape_spike_hp = $_hp_pct;
			$_escape_spike_t_ms = $now;
		}
	} elsif ($_hp_pct >= 30 && $_escape_spike_hp >= 0 && $_escape_spike_hp < 10) {
		# Recovered from danger zone — likely escape teleport
		my $_spike_age = $now - $_escape_spike_t_ms;
		if ($_spike_age > 0 && $_spike_age < 15000) {
			my $_spike_cooldown_ms = _cfg_int('aiSidecar_reflexEscapeSpikeCooldownMs', 30000);
			if ($now - $_last_escape_fire_ms > $_spike_cooldown_ms) {
				$_last_escape_fire_ms = $now;
				_post_event({
					kind => 'bridge_reflex',
					reflex => 'escape_hp_spike',
					severity => 'warning',
					text => "escape reflex triggered: HP spike recovery ($_escape_spike_hp% -> $_hp_pct% in ${_spike_age}ms)",
					hp => $hp,
					hp_max => $hp_max,
					spike_age_ms => $_spike_age,
					map => _safe_field_map() || '',
				});
				debug "[aiSidecarBridge] escape_reflex: HP spike escape detected ($_escape_spike_hp% -> $_hp_pct% in ${_spike_age}ms)\n", 'aiSidecarBridge', 1;

				# Set survival mode to prevent re-engagement
				$_survival_mode_until_ms = $now + 60000 if $_survival_mode_until_ms < $now;
			}
		}
		$_escape_spike_hp = -1;
		$_escape_spike_t_ms = 0;
	} elsif ($_hp_pct >= 30 && $_escape_spike_hp >= 0) {
		# Gradual recovery (not a spike) — reset
		$_escape_spike_hp = -1;
		$_escape_spike_t_ms = 0;
	}

	# ── Sidecar-managed config sync ──
	# (Was orphaned fall-through tail of _build_snapshot_payload; moved here so
	#  it runs on every main-loop tick via _check_bridge_reflexes.)
	my $_sell_npc = _cfg('aiSidecar_sellNpc', '');
	my $_stor_npc = _cfg('aiSidecar_storageNpc', '');
	# COMBAT CONFIG: All controlled by heuristic - bridge does NOT override
	# attackAuto, attackAuto_maxDistance, attackAuto_inLockOnly, attackAuto_followTarget,
	# attackAuto_noMove, attackAuto_onlyWhenSafe, attackAuto_unstuck, attackAuto_minDistance,
	# attackMaxDistance, attackDistance, route_randomWalk, route_randomWalk_inLockOnly
	# are ALL controlled by the heuristic service. Bridge does not touch them.
	state $_last_attackAuto = '';
	state $_last_teleportAuto = '';
	state $_last_teleportAutoHp = '';
	state $_last_teleportAutoMinAgg = '';
	state $_last_itemsTakeAuto = '';
	state $_last_itemsGatherAuto = '';
	state $_last_itemsMaxWeight = '';
	state $_last_sitAuto_hp = '';
	state $_last_sitAuto_hp_max = '';
	state $_last_sitAuto_sp = '';
	state $_last_sitAuto_sp_max = '';
	state $_last_sitAuto_idle = '';
	state $_last_sitAuto_look = '';
	state $_last_followAuto = '';
	state $_last_partyAuto = '';
	state $_last_partyAutoShare = '';
	state $_last_sellAuto = '';
	state $_last_storageAuto = '';

	my $new_attackAuto = _cfg('aiSidecar_attackAuto', '3');
	if ($new_attackAuto ne $_last_attackAuto) { $::config{'attackAuto'} = $new_attackAuto unless $::config{'_sidecar_set_attackAuto'}; $_last_attackAuto = $new_attackAuto; }
	my $new_teleportAuto = _cfg('aiSidecar_teleportAuto', '1');
	if ($new_teleportAuto ne $_last_teleportAuto) { $::config{'teleportAuto'} = $new_teleportAuto unless $::config{'_sidecar_set_teleportAuto'}; $_last_teleportAuto = $new_teleportAuto; }
	my $new_teleportAutoHp = _cfg('aiSidecar_teleportAutoHp', '10');
	if ($new_teleportAutoHp ne $_last_teleportAutoHp) { $::config{'teleportAuto_hp'} = $new_teleportAutoHp unless $::config{'_sidecar_set_teleportAuto_hp'}; $_last_teleportAutoHp = $new_teleportAutoHp; }
	my $new_teleportAutoMinAgg = _cfg('aiSidecar_teleportAutoMinAggressivesInLock', '5');
	if ($new_teleportAutoMinAgg ne $_last_teleportAutoMinAgg) { $::config{'teleportAuto_minAggressivesInLock'} = $new_teleportAutoMinAgg unless $::config{'_sidecar_set_teleportAuto_minAggressivesInLock'}; $_last_teleportAutoMinAgg = $new_teleportAutoMinAgg; }
	my $new_itemsTakeAuto = _cfg('aiSidecar_itemsTakeAuto', '2');
	if ($new_itemsTakeAuto ne $_last_itemsTakeAuto) { $::config{'itemsTakeAuto'} = $new_itemsTakeAuto unless $::config{'_sidecar_set_itemsTakeAuto'}; $_last_itemsTakeAuto = $new_itemsTakeAuto; }
	my $new_itemsGatherAuto = _cfg('aiSidecar_itemsGatherAuto', '2');
	if ($new_itemsGatherAuto ne $_last_itemsGatherAuto) { $::config{'itemsGatherAuto'} = $new_itemsGatherAuto unless $::config{'_sidecar_set_itemsGatherAuto'}; $_last_itemsGatherAuto = $new_itemsGatherAuto; }
	my $new_itemsMaxWeight = _cfg('aiSidecar_itemsMaxWeight', '89');
	if ($new_itemsMaxWeight ne $_last_itemsMaxWeight) { $::config{'itemsMaxWeight'} = $new_itemsMaxWeight unless $::config{'_sidecar_set_itemsMaxWeight'}; $_last_itemsMaxWeight = $new_itemsMaxWeight; }
	my $new_sitAuto_hp = _cfg('aiSidecar_sitAutoHp', '20');
	if ($new_sitAuto_hp ne $_last_sitAuto_hp) { $::config{'sitAuto_hp_lower'} = $new_sitAuto_hp unless $::config{'_sidecar_set_sitAuto_hp'}; $_last_sitAuto_hp = $new_sitAuto_hp; }
	# sitAuto controlled by heuristic through config audit
	# Bridge fallback uses aiSidecar_sitAutoHp default=20
	my $new_sitAuto_hp_max = _cfg('aiSidecar_sitAutoHpMax', '0');
	if ($new_sitAuto_hp_max ne $_last_sitAuto_hp_max) { $::config{'sitAuto_hp_upper'} = $new_sitAuto_hp_max unless $::config{'_sidecar_set_sitAuto_hp_max'}; $_last_sitAuto_hp_max = $new_sitAuto_hp_max; }
	# Force-set sitAuto_hp_upper=0 every cycle when disabled
	if ($new_sitAuto_hp_max eq '0' && !$::config{'_sidecar_set_sitAuto_hp_max'}) {
		$::config{'sitAuto_hp_upper'} = '0';
	}
	my $new_sitAuto_sp = _cfg('aiSidecar_sitAutoSp', '0');
	if ($new_sitAuto_sp ne $_last_sitAuto_sp) { $::config{'sitAuto_sp'} = $new_sitAuto_sp unless $::config{'_sidecar_set_sitAuto_sp'}; $_last_sitAuto_sp = $new_sitAuto_sp; }
	my $new_sitAuto_sp_max = _cfg('aiSidecar_sitAutoSpMax', '0');
	if ($new_sitAuto_sp_max ne $_last_sitAuto_sp_max) { $::config{'sitAuto_sp_max'} = $new_sitAuto_sp_max unless $::config{'_sidecar_set_sitAuto_sp_max'}; $_last_sitAuto_sp_max = $new_sitAuto_sp_max; }
	my $new_sitAuto_idle = _cfg('aiSidecar_sitAutoIdle', '0');
	if ($new_sitAuto_idle ne $_last_sitAuto_idle) { $::config{'sitAuto_idle'} = $new_sitAuto_idle unless $::config{'_sidecar_set_sitAuto_idle'}; $_last_sitAuto_idle = $new_sitAuto_idle; }
	my $new_sitAuto_look = _cfg('aiSidecar_sitAutoLook', '0');
	if ($new_sitAuto_look ne $_last_sitAuto_look) { $::config{'sitAuto_look'} = $new_sitAuto_look unless $::config{'_sidecar_set_sitAuto_look'}; $_last_sitAuto_look = $new_sitAuto_look; }
	my $new_followAuto = _cfg('aiSidecar_followAuto', '0');
	if ($new_followAuto ne $_last_followAuto) { $::config{'followAuto'} = $new_followAuto unless $::config{'_sidecar_set_followAuto'}; $_last_followAuto = $new_followAuto; }
	my $new_partyAuto = _cfg('aiSidecar_partyAuto', '1');
	if ($new_partyAuto ne $_last_partyAuto) { $::config{'partyAuto'} = $new_partyAuto unless $::config{'_sidecar_set_partyAuto'}; $_last_partyAuto = $new_partyAuto; }
	my $new_partyAutoShare = _cfg('aiSidecar_partyAutoShare', '1');
	if ($new_partyAutoShare ne $_last_partyAutoShare) { $::config{'partyAutoShare'} = $new_partyAutoShare unless $::config{'_sidecar_set_partyAutoShare'}; $_last_partyAutoShare = $new_partyAutoShare; }
	my $new_sellAuto = _cfg('aiSidecar_sellAuto', '0');
	if ($new_sellAuto ne $_last_sellAuto) { $::config{'sellAuto'} = $new_sellAuto unless $::config{'_sidecar_set_sellAuto'}; $_last_sellAuto = $new_sellAuto; }
	$::config{'sellAuto_npc'} = $_sell_npc if $_sell_npc;
		my $new_storageAuto = _cfg('aiSidecar_storageAuto', '0');
	if ($new_storageAuto ne $_last_storageAuto) { $::config{'storageAuto'} = $new_storageAuto unless $::config{'_sidecar_set_storageAuto'}; $_last_storageAuto = $new_storageAuto; }
	$::config{'storageAuto_npc'} = $_stor_npc if $_stor_npc;
													    # Pro RO: disable auto-sit, let sidecar handle healing
    # sitAuto_hp_lower controlled by heuristic — bridge must NOT override
    # sitAuto_hp_upper controlled by heuristic
    # $::config{attackAuto} = 3;
    # attackAuto_inLockOnly controlled by heuristic
    # $::config{attackDistance} = 7;  # heuristic controls this
}
} # CLOSE the organizational bare block opened at line ~5292 (was closed by the
  # removed orphan brace; the config-sync tail moved inside _check_bridge_reflexes)

# ── Safe character accessor ──
sub _safe_char {
	return $main::char || $char;
}

sub _safe_hp_ratio {
	my $cr = _safe_char();
	return 1.0 if !$cr;
	my $hp = $cr->{hp} || 0;
	my $hp_max = $cr->{hp_max} || 1;
	return $hp / $hp_max;
}

# ── Survival / Progression monitor (last-resort fallback) ──
# Most execution is handled by OpenKore's built-in systems (attackAuto,
# buyAuto, sellAuto, items_control). This function only fires when the
# Pro RO LLM's goals detect no progression for >5 minutes, or when
# HP is critically low (< 20%) as an emergency override.
# ── _survival_check: REMOVED — stripped from main loop, all config/survival
# handled by sidecar heuristic service. The function was dead code that
# called undefined _apply_bot_config. _sell_junk_items also removed as
# it was only called from within _survival_check.

1;
sub _discover_shops {
	my $now = _now_ms();
	state $_last_ss = 0;
	return if $now - $_last_ss < 3600000;
	$_last_ss = $now;
	# Scan NPCs from npcsList for shop-type NPCs
	my $shops = _discover_shops_sync();
	_post_event({ kind => 'discovery_shops', shops => $shops });
}

sub _discover_portals {
	my $now = _now_ms();
	state $_last_ps = 0;
	return if $now - $_last_ps < 3600000;
	$_last_ps = $now;
	my %conn;
	my $pf = ($::Settings->{tablesPath} || 'tables') . '/portals.txt';
	if (open(my $fh, '<', $pf)) {
	    while (my $ln = <$fh>) {
	        chomp $ln; next if $ln =~ /^#/ || $ln =~ /^\s*$/;
	        my @p = split(/\s+/, $ln); next if @p < 6;
	        $conn{$p[0]}{$p[3]} = 1; $conn{$p[3]}{$p[0]} = 1;
	    }
	    close $fh;
	}
	if (keys %conn > 0) {
	    _post_event({ kind => 'discovery_portals', connections => \%conn });
	}
}

# ── Sync variants (call within same request cycle, no rate limit) ──
sub _discover_shops_sync {
	my @shops;
	if (defined $main::npcsList && ref($main::npcsList) eq 'HASH') {
	    foreach my $_n (values %{$main::npcsList}) {
	        next unless ref($_n) eq 'HASH' && $_n->{name} && $_n->{shop};
	        push @shops, {
	            name => $_n->{name}, map => $_n->{map} || '',
	            x => $_n->{x} || 0, y => $_n->{y} || 0,
	            shop => $_n->{shop},
	        };
	    }
	}
	}


sub _discover_portals_sync {
	my %conn;
	my $pf = ($::Settings->{tablesPath} || 'tables') . '/portals.txt';
	if (open(my $fh, '<', $pf)) {
	    while (my $ln = <$fh>) {
	        chomp $ln; next if $ln =~ /^#/ || $ln =~ /^\s*$/;
	        my @p = split(/\s+/, $ln); next if @p < 6;
	        $conn{$p[0]}{$p[3]} = 1; $conn{$p[3]}{$p[0]} = 1;
	    }
	    close $fh;
	}
	return \%conn;
}

# ── Safe inventory list ──
sub _safe_inventory_list {
	my $cr = _safe_char();
	return [] if !$cr;
	my @items;
	foreach my $_i (@{_char_inventory($cr)}) {
	    next unless ref($_i) eq 'HASH';
	    push @items, { name => $_i->{name} || '', amount => $_i->{amount} || 0 };
	}
	return \@items;
}



# ── Table file reader (reads OpenKore tables as data source) ──
sub _read_table_file {
	my ($pattern) = @_;
	my $tdir = $::Settings->{tablesPath} || './tables';
	my $file = "$tdir/$pattern";
	return [] unless -f $file;
	open(my $fh, '<', $file) or return [];
	my @lines;
	while (my $line = <$fh>) {
	    chomp $line;
	    next if $line =~ /^#/ || $line =~ /^\s*$/;
	    push @lines, $line;
	}
	close $fh;
	return \@lines;
}

# ── Send ALL table data to sidecar (source of truth) ──
sub _send_discovery_data {
	my $data = {
	    npcs => _read_table_file('npcs.txt'),
	    npc_shops => _read_table_file('npc_shops.txt'),
	    portals => _read_table_file('portals.txt'),
	    portal_commands => _read_table_file('portals_commands.txt'),
	    portal_los => _read_table_file('portalsLOS.txt'),
	    portal_spawns => _read_table_file('portals_spawns.txt'),
	    cities => _read_table_file('cities.txt'),
	    monsters => _read_table_file('monsters.txt'),
	    monsters_table => _read_table_file('monsters_table.txt'),
	    item_weights => _read_table_file('item_weights.txt'),
	    job_change => _read_table_file('job_change_locations.txt'),
	    skill_handle => _read_table_file('SKILL_id_handle.txt'),
	    item_hand_types => _read_table_file('item_hand_type.txt'),
	    no_teleport => _read_table_file('no_teleport_maps.txt'),
	    elements => _read_table_file('elements.txt'),
	};
	_http_post_json('/discover/tables/ingest', {
	    kind => 'discovery_all_tables',
	    tables => $data,
	    timestamp => _now_ms(),
	});
}

# ── Apply ML overrides from source="ml" actions ──
# ── Check pending ML execution outcomes (survival check) ──
sub _check_ml_outcome {
	my $bot_id = _bot_id();
	return unless defined $ml_pending_outcome{$bot_id};

	my $pending = $ml_pending_outcome{$bot_id};
	my $current_hp = $char ? $char->{hp} : 0;
	my $hp_max = $pending->{hp_max} || 1;
	my $hp_ratio = $current_hp / $hp_max;
	my $success = 1;

	if ($current_hp <= 0) {
		$success = 0;  # bot died
	} elsif ($hp_ratio < 0.3) {
		$success = 0;  # critically low HP
	}

	my $resp = _http_post_json('/v2/ml/outcome', {
		bot_id => $bot_id,
		family => $pending->{family},
		success => ($success ? "yes" : "no"),
	});
	if ($resp && $resp->{status} >= 200 && $resp->{status} < 300) {
		delete $ml_pending_outcome{$bot_id};
		warning("[aiSidecarBridge] ml_outcome reported: family=$pending->{family} success=$success");
	} else {
		# Retry on next cycle — don't delete pending outcome
		warning("[aiSidecarBridge] ml_outcome retry pending: family=$pending->{family}");
	}
}

sub _apply_ml_override {
	my ($override) = @_;
	return unless defined $override && ref($override) eq 'HASH';

	my $family = $override->{family} || '';
	my $rec = $override->{recommendation} || {};

	if ($family eq 'encounter_classifier' && defined $rec->{encounter_profile}) {
		my $profile = lc($rec->{encounter_profile});
		if ($profile eq 'aggressive') {
			$::config{attackAuto} = 3;
			$::config{autoMove} = 2;
			_apply_ml_config_guard('attackAuto', 3);
			_apply_ml_config_guard('autoMove', 2);
			warning("ml_override applied: encounter_classifier=aggressive (attackAuto = 3, autoMove=2)");
		} elsif ($profile eq 'safe') {
			$::config{attackAuto} = 1;
			$::config{autoMove} = 0;
			_apply_ml_config_guard('attackAuto', 1);
			_apply_ml_config_guard('autoMove', 0);
			warning("ml_override applied: encounter_classifier=safe (attackAuto = 1, autoMove=0)");
		} else {
			$::config{attackAuto} = 3;
			$::config{autoMove} = 1;
			_apply_ml_config_guard('attackAuto', 3);
			_apply_ml_config_guard('autoMove', 1);
			warning("ml_override applied: encounter_classifier=balanced (attackAuto = 3, autoMove=1)");
		}
	} elsif ($family eq 'route_recovery_classifier' && defined $rec->{stuck_strategy}) {
		my $strategy = lc($rec->{stuck_strategy});
		if ($strategy eq 'repath' || $strategy eq 'recalc') {
			my $ok = eval { _toggle_ai_mode('manual'); 1; };
			warning("ml_override applied: route_recovery=$strategy (ai toggled to manual for recalc)");
		} elsif ($strategy eq 'teleport') {
			my $skill = eval { $char->skills->get('AL_TELEPORT') } || eval { $char->skills->get('TF_TELEPORT') };
			if ($skill) { eval { Commands::run("use_skill teleport"); 1; }; }
			warning("ml_override applied: route_recovery=teleport");
		} else {
			warning("ml_override applied: route_recovery=$strategy (no specific handler)");
		}
	} elsif ($family eq 'loot_ranker' && defined $rec->{loot_item}) {
		my $item = $rec->{loot_item};
		# Enable auto-loot if it's off so the ranked item is actually picked up,
		# and record the priority intent so the pickup is not skipped.
		$::config{itemsTakeAuto} = 2 unless $::config{'_sidecar_set_itemsTakeAuto'};
		_apply_ml_config_guard('itemsTakeAuto', 2);
		$::config{_sidecar_loot_priority} = $item;
		warning("ml_override applied: loot_ranker=$item (loot pickup enabled, priority recorded)");
	} elsif ($family eq 'npc_dialogue_predictor' && defined $rec->{dialogue_branch}) {
		my $branch = $rec->{dialogue_branch};
		# Apply the predicted dialogue branch (a talk response index) so the
		# bot follows the recommended NPC dialog path.
		if ($branch =~ /^(\d+)$/) {
			eval { Commands::run("talk resp $1"); 1; };
			warning("ml_override applied: npc_dialogue branch=$branch (emitted talk resp $1)");
		} else {
			warning("ml_override applied: npc_dialogue=$branch (non-numeric branch, logged only)");
		}
	} elsif ($family eq 'risk_anomaly_detector' && defined $rec->{risk_label}) {
		my $score = $rec->{risk_label};
		warning("ml_override applied: risk_anomaly score=$score");
	} elsif ($family eq 'memory_retrieval_ranker' && defined $rec->{memory_id}) {
		my $mem_id = $rec->{memory_id};
		warning("ml_override applied: memory_retrieval=$mem_id (logging only)");
	} else {
	        warning("ml_override received but not applied: unknown family=$family or missing recommendation keys");
	    }
}

# ── Apply an ML-recommended config delta ──
# Sets $::config{key} to the given value so the override actually takes
# effect immediately. Respects the _sidecar_set_<key> shield (set by an
# explicit sidecar `set <key> <value>` command) so ML overrides never
# fight an explicit user/bot setting.
sub _apply_ml_config_guard {
	my ($key, $val) = @_;
	if (!defined $key || $key eq '') {
		return undef;
	}
	if (!$::config{"_sidecar_set_$key"}) {
		$::config{$key} = $val;
	}
	return $::config{$key};
}

	# ═══════════════════════════════════════════════════════════════════════════
	# ── Periodic snapshot-only collection tasks (called from _poll_next_action) ──
	# ═══════════════════════════════════════════════════════════════════════════

	# ── Scan NPC shops from snapshot data (no dialog interaction) ──
	# Tracks NPCs that have shop data visible in the snapshot.
	# Called every poll cycle from _poll_next_action.
	sub _scan_npc_shops {
	    my $now = _now_ms();
	    return if $now - $_last_npc_shop_scan_ms < $NPC_SHOP_SCAN_INTERVAL_MS;
	    $_last_npc_shop_scan_ms = $now;

	    # Collect NPC shop data from the npcs hash
	    my @shops;
	    if (defined $main::npcsList && ref($main::npcsList) eq 'HASH') {
	        foreach my $_nid (keys %{$main::npcsList}) {
	            my $_n = $main::npcsList->{$_nid};
	            next unless ref($_n) eq 'HASH';
	            my $_nname = $_n->{name} || '';
	            next if $_nname eq '';
	            # Check if this NPC has a shop flag or is a known shop NPC
	            if ($_n->{shop} || $_nname =~ /tool\s*dealer|kafra|weapon|armor|potion|accessory|item\s*shop|general\s*store|inn/i) {
	                push @shops, {
	                    name => $_nname,
	                    map => $_n->{map} || _safe_field_map() || '',
	                    x => $_n->{x} || 0,
	                    y => $_n->{y} || 0,
	                    shop => $_n->{shop} || 0,
	                };
	            }
	        }
	    }

	    # Report to sidecar if we found shops
	    if (@shops) {
	        _post_event({
	            kind => 'discovery_shops',
	            event_type => 'npc.shops_scanned',
	            severity => 'info',
	            text => 'NPC shops scanned from snapshot',
	            shops => \@shops,
	            count => scalar(@shops),
	        });
	        debug "[npc_shops] scanned " . scalar(@shops) . " NPC shops from snapshot\n", 'aiSidecarBridge', 2;
	    }
	}

	# ── Scan player vendors from snapshot data (no dialog interaction) ──
	# Tracks players who are vending (visible in the player list).
	# Called every poll cycle from _poll_next_action.
	sub _scan_player_vendors {
	    my $now = _now_ms();
	    return if $now - $_last_player_vendor_scan_ms < $PLAYER_VENDOR_SCAN_INTERVAL_MS;
	    $_last_player_vendor_scan_ms = $now;

	    my @vendors;
	    if (defined $main::playersList && ref($main::playersList) eq 'HASH') {
	        foreach my $_pid (keys %{$main::playersList}) {
	            my $_p = $main::playersList->{$_pid};
	            next unless ref($_p) eq 'HASH';
	            my $_pname = $_p->{name} || '';
	            next if $_pname eq '';
	            # Check if player is vending (has shop_open flag or vendor title)
	            if ($_p->{shop_open} || $_p->{vendor_title} || $_p->{vending}) {
	                push @vendors, {
	                    name => $_pname,
	                    map => $_p->{map} || _safe_field_map() || '',
	                    x => $_p->{x} || 0,
	                    y => $_p->{y} || 0,
	                    title => $_p->{vendor_title} || $_p->{shop_title} || '',
	                };
	            }
	        }
	    }

	    if (@vendors) {
	        _post_event({
	            kind => 'discovery_vendors',
	            event_type => 'player.vendors_scanned',
	            severity => 'info',
	            text => 'Player vendors scanned from snapshot',
	            vendors => \@vendors,
	            count => scalar(@vendors),
	        });
	        debug "[player_vendors] scanned " . scalar(@vendors) . " player vendors from snapshot\n", 'aiSidecarBridge', 2;
	    }
	}

	# ── Report party member positions to sidecar ──
	sub _report_party_positions {
	    return if !_bridge_enabled();
	    return if !$registered;
	    return if !$char;
	    return if !$char->{party};

	    my $now = _now_ms();
	    state $_last_rpp_ms = 0;
	    return if $now - $_last_rpp_ms < 30000;
	    $_last_rpp_ms = $now;

	    my $map = _safe_field_map() || '';
	    my ($x, $y);
	    if ($char->{pos_to} && ref $char->{pos_to} eq 'HASH') {
	        $x = $char->{pos_to}{x}; $y = $char->{pos_to}{y};
	    } elsif ($char->{pos} && ref $char->{pos} eq 'HASH') {
	        $x = $char->{pos}{x}; $y = $char->{pos}{y};
	    }
	    return if !defined $x || !defined $y;

	    my $resp = _http_post_json('/v2/party/position', {
	        meta => _meta(_bot_id()),
	        x => $x + 0,
	        y => $y + 0,
	        map => $map,
	        timestamp => $now,
	    });
	    if ($resp && $resp->{status} >= 200 && $resp->{status} < 300) {
	        debug "[party_position] reported position ($x,$y) on $map\n", 'aiSidecarBridge', 3;
	    }
	}

	# ── Report game time to sidecar ──
	sub _report_game_time {
	    my $now = _now_ms();
	    return if $now - $_last_game_time_check_ms < $GAME_TIME_CHECK_INTERVAL_MS;
	    $_last_game_time_check_ms = $now;

	    # Get server time from OpenKore's internal clock
	    my $game_time = '';
	    if (defined $main::timeInfo && ref($main::timeInfo) eq 'HASH') {
	        $game_time = $main::timeInfo->{serverTime} || '';
	    }
	    if ($game_time eq '' && defined $main::field) {
	        # Fallback: use local time as approximation
	        my ($sec, $min, $hour) = (localtime(time))[0,1,2];
	        $game_time = sprintf('%02d:%02d:%02d', $hour, $min, $sec);
	    }

	    if ($game_time ne '' && $game_time ne $_last_reported_game_time) {
	        $_last_reported_game_time = $game_time;
	        _post_event({
	            kind => 'bridge_event',
	            event_type => 'game.time',
	            severity => 'info',
	            text => "Game time: $game_time",
	            game_time => $game_time,
	        });
	        debug "[game_time] reported: $game_time\n", 'aiSidecarBridge', 2;
	    }
	}

	# ── Flush pending server announcements to sidecar ──
	sub _flush_announcements {
	    my $now = _now_ms();
	    return if $now - $_last_announcement_flush_ms < $ANNOUNCEMENT_FLUSH_INTERVAL_MS;
	    $_last_announcement_flush_ms = $now;

	    return if !@_pending_announcements;

	    my @batch = splice @_pending_announcements, 0, 10;
	    for my $_ann (@batch) {
	        _post_event({
	            kind => 'bridge_event',
	            event_type => 'server.announcement',
	            severity => 'info',
	            text => $_ann,
	        });
	    }
	    debug "[announcements] flushed " . scalar(@batch) . " announcements\n", 'aiSidecarBridge', 2;
	}

	# ── Detect dispel effects by monitoring buff count changes ──
	sub _detect_dispel {
	    my $now = _now_ms();
	    return if $now - $_last_dispel_check_ms < $DISPEL_CHECK_INTERVAL_MS;
	    $_last_dispel_check_ms = $now;

	    return if !$char;

	    my @current_buffs;
	    if ($char->{buffs} && ref($char->{buffs}) eq 'ARRAY') {
	        @current_buffs = map { $_->{name} || '' } @{$char->{buffs}};
	    } elsif ($char->{buffs} && ref($char->{buffs}) eq 'HASH') {
	        @current_buffs = keys %{$char->{buffs}};
	    }
	    my $current_count = scalar(@current_buffs);

	    if ($_prev_buff_count > 0 && $current_count < $_prev_buff_count) {
	        my @lost_buffs;
	        if (@_prev_buff_names) {
	            my %current_map = map { $_ => 1 } @current_buffs;
	            for my $_pb (@_prev_buff_names) {
	                push @lost_buffs, $_pb if !$current_map{$_pb};
	            }
	        }
	        my $lost_count = scalar(@lost_buffs);
	        push @_dispel_events, {
	            detected_at_ms => $now,
	            map => _safe_field_map() || '',
	            buffs_lost => \@lost_buffs,
	            prev_count => $_prev_buff_count,
	            current_count => $current_count,
	        };
	        # Keep only last 10 dispel events
	        splice @_dispel_events, 0, scalar(@_dispel_events) - 10 if @_dispel_events > 10;

	        _post_event({
	            kind => 'bridge_reflex',
	            reflex => 'dispel_detected',
	            severity => 'warning',
	            text => "Dispel detected: lost $lost_count buffs",
	            lost_buffs => join(',', @lost_buffs),
	            prev_count => $_prev_buff_count,
	            current_count => $current_count,
	            map => _safe_field_map() || '',
	        });
	        debug "[dispel] detected: lost $lost_count buffs (was $_prev_buff_count, now $current_count)\n", 'aiSidecarBridge', 1;
	    }

	    $_prev_buff_count = $current_count;
	    @_prev_buff_names = @current_buffs;
	}

	# ═══════════════════════════════════════════════════════════════════════════
	# ── NPC Shop Dialog Interaction ──
	# ═══════════════════════════════════════════════════════════════════════════

	# NPC dialog state tracking (initialized; `our` at top of file)
	%_npc_dialog_state = (
	    in_dialog => 0,          # Are we currently in an NPC dialog?
	    npc_name => '',          # Name of the NPC we're talking to
	    npc_x => 0,              # NPC X position
	    npc_y => 0,              # NPC Y position
	    dialog_stage => '',      # Current dialog stage (menu, shop, text, etc.)
	    menu_options => [],      # Available menu options
	    shop_items => [],        # Shop items if in a shop dialog
	    last_interaction_ms => 0, # When we last interacted
	    dialog_timeout_ms => 30000, # Max time to stay in dialog
	);

	# ── Open NPC shop and buy items ──
	# Called when sidecar sends "npc_buy <npc_name> <item_name> <quantity>"
	# Walks to NPC, talks to them, navigates dialog to buy, reports result.
	sub _open_npc_shop {
	    my ($npc_name, $item_name, $quantity) = @_;
	    return (0, 'missing_params', 'npc_name, item_name, and quantity required')
	        if !$npc_name || !$item_name || !$quantity;

	    # Find the NPC in the npcs hash
	    my ($npc_id, $npc_x, $npc_y);
	    if (defined $main::npcsList && ref($main::npcsList) eq 'HASH') {
	        foreach my $_nid (keys %{$main::npcsList}) {
	            my $_n = $main::npcsList->{$_nid};
	            next unless ref($_n) eq 'HASH';
	            my $_nname = lc($_n->{name} || '');
	            next if $_nname ne lc($npc_name);
	            $npc_id = $_nid;
	            $npc_x = $_n->{x} || 0;
	            $npc_y = $_n->{y} || 0;
	            last;
	        }
	    }

	    if (!$npc_id) {
	        return (0, 'npc_not_found', "NPC '$npc_name' not found nearby");
	    }

	    # Walk to NPC
	    my $walk_ok = eval { Commands::run("move $npc_x $npc_y"); 1; };
	    if (!$walk_ok) {
	        return (0, 'walk_failed', "Failed to walk to NPC '$npc_name' at ($npc_x,$npc_y)");
	    }

	    # Small delay to let movement start
	    usleep(500000);

	    # Talk to NPC
	    my $talk_ok = eval { Commands::run("talknpc $npc_x $npc_y"); 1; };
	    if (!$talk_ok) {
	        return (0, 'talk_failed', "Failed to talk to NPC '$npc_name'");
	    }

	    # Update dialog state
	    $_npc_dialog_state{in_dialog} = 1;
	    $_npc_dialog_state{npc_name} = $npc_name;
	    $_npc_dialog_state{npc_x} = $npc_x;
	    $_npc_dialog_state{npc_y} = $npc_y;
	    $_npc_dialog_state{dialog_stage} = 'started';
	    $_npc_dialog_state{last_interaction_ms} = _now_ms();

	    # Wait for dialog to open
	    usleep(300000);

	    # Try to navigate: send 'c' (continue) to get past intro text
	    eval { Commands::run("talk c"); 1; };
	    usleep(200000);

	    # Try to find the buy option in the menu
	    # Common shop menu patterns: "Buy", "Purchase", "Shop", "Trade"
	    eval { Commands::run("talk resp 0"); 1; };  # First option is often "Buy"
	    usleep(200000);

	    # Now we should be in the shop window — try to buy the item
	    # OpenKore's shop interface: items are listed with indices
	    # We need to find the item index in the shop
	    my $item_idx = _find_shop_item_index($item_name);
	    if (!defined $item_idx) {
	        # Close dialog and report failure
	        eval { Commands::run("talk close"); 1; };
	        $_npc_dialog_state{in_dialog} = 0;
	        return (0, 'item_not_in_shop', "Item '$item_name' not found in NPC '$npc_name' shop");
	    }

	    # Buy the item
	    my $buy_ok = eval { Commands::run("buy $item_idx $quantity"); 1; };
	    if (!$buy_ok) {
	        eval { Commands::run("talk close"); 1; };
	        $_npc_dialog_state{in_dialog} = 0;
	        return (0, 'buy_failed', "Failed to buy $quantity x $item_name from NPC '$npc_name'");
	    }

	    usleep(300000);

	    # Close dialog
	    eval { Commands::run("talk close"); 1; };
	    $_npc_dialog_state{in_dialog} = 0;

	    # Report success
	    _post_event({
	        kind => 'bridge_event',
	        event_type => 'npc.shop_bought',
	        severity => 'info',
	        text => "Bought $quantity x $item_name from NPC $npc_name",
	        npc_name => $npc_name,
	        item_name => $item_name,
	        quantity => $quantity,
	    });

	    return (1, 'ok', "Bought $quantity x $item_name from NPC '$npc_name'");
	}

	# ── Find item index in NPC shop ──
	# Searches the current shop window for an item by name.
	# Returns the item index (0-based) or undef if not found.
	sub _find_shop_item_index {
	    my ($item_name) = @_;
	    return undef if !$item_name;

	    my $item_lc = lc($item_name);

	    # Check if we have shop data from the NPC dialog
	    if ($_npc_dialog_state{shop_items} && @{$_npc_dialog_state{shop_items}}) {
	        for my $i (0 .. $#{$_npc_dialog_state{shop_items}}) {
	            my $_si = $_npc_dialog_state{shop_items}[$i];
	            next unless ref($_si) eq 'HASH';
	            my $_sin = lc($_si->{name} || '');
	            return $i if $_sin eq $item_lc;
	        }
	    }

	    # Fallback: search inventory for recently bought items to infer index
	    # This is a best-effort approach when shop data isn't available
	    return undef;
	}

	# ── Parse shop items from NPC dialog response ──
	# Called when we receive a shop listing from an NPC dialog.
	# Parses the item list and stores it in _npc_dialog_state.
	sub _parse_npc_shop_items {
	    my ($dialog_text) = @_;
	    return if !$dialog_text;

	    my @items;
	    my @lines = split(/\n/, $dialog_text);
	    for my $_line (@lines) {
	        chomp $_line;
	        next if $_line =~ /^\s*$/;
	        # Shop item format: "<index> - <name> : <price>z"
	        if ($_line =~ /^(\d+)\s*[-:]\s*(.+?)\s*:\s*(\d+)\s*z/i) {
	            push @items, {
	                index => $1,
	                name => $2,
	                price => $3,
	            };
	        }
	        # Alternative format: "<name> (<price>z)"
	        elsif ($_line =~ /^(\d+)\s*[-:]\s*(.+?)\s*\((\d+)z\)/i) {
	            push @items, {
	                index => $1,
	                name => $2,
	                price => $3,
	            };
	        }
	    }

	    if (@items) {
	        $_npc_dialog_state{shop_items} = \@items;
	        debug "[npc_shop] parsed " . scalar(@items) . " shop items from dialog\n", 'aiSidecarBridge', 2;
	    }
	}

	# ═══════════════════════════════════════════════════════════════════════════
	# ── Player Vendor Interaction ──
	# ═══════════════════════════════════════════════════════════════════════════

	# ── Open player vendor and buy items ──
	# Called when sidecar sends "vendor_buy <player_name> <item_name> <quantity>"
	# Walks to player, clicks vendor, buys items, reports result.
	sub _open_player_vendor {
	    my ($player_name, $item_name, $quantity) = @_;
	    return (0, 'missing_params', 'player_name, item_name, and quantity required')
	        if !$player_name || !$item_name || !$quantity;

	    # Find the player in the players hash
	    my ($player_id, $player_x, $player_y);
	    if (defined $main::playersList && ref($main::playersList) eq 'HASH') {
	        foreach my $_pid (keys %{$main::playersList}) {
	            my $_p = $main::playersList->{$_pid};
	            next unless ref($_p) eq 'HASH';
	            my $_pname = lc($_p->{name} || '');
	            next if $_pname ne lc($player_name);
	            $player_id = $_pid;
	            $player_x = $_p->{x} || 0;
	            $player_y = $_p->{y} || 0;
	            last;
	        }
	    }

	    if (!$player_id) {
	        return (0, 'player_not_found', "Player '$player_name' not found nearby");
	    }

	    # Walk to player
	    my $walk_ok = eval { Commands::run("move $player_x $player_y"); 1; };
	    if (!$walk_ok) {
	        return (0, 'walk_failed', "Failed to walk to player '$player_name' at ($player_x,$player_y)");
	    }

	    usleep(500000);

	    # Open vendor by talking to the player
	    # OpenKore uses talknpc with player ID for vendors
	    my $talk_ok = eval { Commands::run("talknpc $player_id"); 1; };
	    if (!$talk_ok) {
	        return (0, 'vendor_open_failed', "Failed to open vendor for player '$player_name'");
	    }

	    usleep(300000);

	    # Try to find the item in the vendor window
	    # Vendor items are listed with indices — we need to find the right one
	    my $item_idx = _find_vendor_item_index($item_name, $player_name);
	    if (!defined $item_idx) {
	        eval { Commands::run("talk close"); 1; };
	        return (0, 'item_not_in_vendor', "Item '$item_name' not found in player '$player_name' vendor");
	    }

	    # Buy the item from vendor
	    my $buy_ok = eval { Commands::run("buy $item_idx $quantity"); 1; };
	    if (!$buy_ok) {
	        eval { Commands::run("talk close"); 1; };
	        return (0, 'buy_failed', "Failed to buy $quantity x $item_name from player '$player_name' vendor");
	    }

	    usleep(300000);

	    # Close vendor window
	    eval { Commands::run("talk close"); 1; };

	    _post_event({
	        kind => 'bridge_event',
	        event_type => 'player.vendor_bought',
	        severity => 'info',
	        text => "Bought $quantity x $item_name from player $player_name vendor",
	        player_name => $player_name,
	        item_name => $item_name,
	        quantity => $quantity,
	    });

	    return (1, 'ok', "Bought $quantity x $item_name from player '$player_name' vendor");
	}

	# ── Find item index in player vendor ──
	# Searches the vendor window for an item by name.
	# Returns the item index (0-based) or undef if not found.
	sub _find_vendor_item_index {
	    my ($item_name, $player_name) = @_;
	    return undef if !$item_name;

	    my $item_lc = lc($item_name);

	    # Check cached vendor data
	    if ($player_name && $_player_vendor_data{$player_name}) {
	        my $vendor = $_player_vendor_data{$player_name};
	        if ($vendor->{items} && ref($vendor->{items}) eq 'ARRAY') {
	            for my $i (0 .. $#{$vendor->{items}}) {
	                my $_vi = $vendor->{items}[$i];
	                next unless ref($_vi) eq 'HASH';
	                my $_vin = lc($_vi->{name} || '');
	                return $i if $_vin eq $item_lc;
	            }
	        }
	    }

	    return undef;
	}

	# ═══════════════════════════════════════════════════════════════════════════
	# ── Vending Setup ──
	# ═══════════════════════════════════════════════════════════════════════════

	# ── Set up vending shop ──
	# Called when sidecar sends "start_vending <title> <items>"
	# Opens vending shop with given title and item list.
	# Items format: "item_name:price,item_name2:price2"
	sub _setup_vending {
	    my ($title, $items_str) = @_;
	    return (0, 'missing_params', 'title and items required')
	        if !$title || !$items_str;

	    # Parse items: "item_name:price,item_name2:price2"
	    my @items;
	    my @pairs = split(/,/, $items_str);
	    for my $_pair (@pairs) {
	        chomp $_pair;
	        $_pair =~ s/^\s+//;
	        $_pair =~ s/\s+$//;
	        next if $_pair eq '';
	        my ($item_name, $price) = split(/:/, $_pair, 2);
	        next if !$item_name || !$price;
	        $item_name =~ s/^\s+//;
	        $item_name =~ s/\s+$//;
	        $price =~ s/^\s+//;
	        $price =~ s/\s+$//;
	        push @items, { name => $item_name, price => int($price) };
	    }

	    if (!@items) {
	        return (0, 'no_items', 'No valid items specified for vending');
	    }

	    # OpenKore vending command: "vending <title>"
	    # Then add items one by one
	    my $vending_ok = eval { Commands::run("vending $title"); 1; };
	    if (!$vending_ok) {
	        return (0, 'vending_open_failed', "Failed to open vending shop with title '$title'");
	    }

	    usleep(300000);

	    # Add each item to the vending shop
	    for my $_item (@items) {
	        # Find item in inventory by name
	        my $_item_idx = 0;
	        my $_found = 0;
	        if ($char && @{_char_inventory($char)}) {
	            for my $_inv_item (@{_char_inventory($char)}) {
	                next unless ref($_inv_item) eq 'HASH';
	                my $_inv_name = $_inv_item->{name} || '';
	                if (lc($_inv_name) eq lc($_item->{name})) {
	                    $_found = 1;
	                    last;
	                }
	                $_item_idx++;
	            }
	        }
	        if ($_found) {
	            # OpenKore: "vending_add <inventory_index> <quantity> <price>"
	            my $add_ok = eval { Commands::run("vending_add $_item_idx 1 $_item->{price}"); 1; };
	            if ($add_ok) {
	                debug "[vending] added item '$_item->{name}' at $_item->{price}z (inv_idx=$_item_idx)\n", 'aiSidecarBridge', 2;
	            } else {
	                warning "[vending] failed to add item '$_item->{name}': $@\n", 'aiSidecarBridge', 1;
	            }
	            usleep(100000);
	        } else {
	            warning "[vending] item '$_item->{name}' not found in inventory, skipping\n", 'aiSidecarBridge', 1;
	        }
	    }

	    # Open the vending shop
	    my $open_ok = eval { Commands::run("vending_open"); 1; };
	    if (!$open_ok) {
	        return (0, 'vending_open_failed', "Failed to open vending shop");
	    }

	    _post_event({
	        kind => 'bridge_event',
	        event_type => 'player.vending_started',
	        severity => 'info',
	        text => "Vending started: $title with " . scalar(@items) . " items",
	        title => $title,
	        item_count => scalar(@items),
	    });

	    return (1, 'ok', "Vending started: '$title' with " . scalar(@items) . " items");
	}

	# ═══════════════════════════════════════════════════════════════════════════
	# ── Trade Request Handling ──
	# ═══════════════════════════════════════════════════════════════════════════

	# Trade state tracking
	our %_trade_state = (
	    active => 0,              # Is a trade currently active?
	    partner => '',            # Who we're trading with
	    stage => '',              # Current trade stage (requested, accepted, adding, confirmed)
	    items_to_add => [],       # Items we need to add to the trade window
	    started_at_ms => 0,       # When the trade started
	    timeout_ms => 60000,      # Trade timeout
	);

	# ── Handle trade request ──
	# Called when sidecar sends "trade_request <player_name> [items]"
	# Items format: "item_name:quantity,item_name2:quantity2"
	sub _handle_trade {
	    my ($player_name, $items_str) = @_;
	    return (0, 'missing_params', 'player_name required')
	        if !$player_name;

	    # Find the player
	    my ($player_id, $player_x, $player_y);
	    if (defined $main::playersList && ref($main::playersList) eq 'HASH') {
	        foreach my $_pid (keys %{$main::playersList}) {
	            my $_p = $main::playersList->{$_pid};
	            next unless ref($_p) eq 'HASH';
	            my $_pname = lc($_p->{name} || '');
	            next if $_pname ne lc($player_name);
	            $player_id = $_pid;
	            $player_x = $_p->{x} || 0;
	            $player_y = $_p->{y} || 0;
	            last;
	        }
	    }

	    if (!$player_id) {
	        return (0, 'player_not_found', "Player '$player_name' not found nearby");
	    }

	    # Walk to player
	    my $walk_ok = eval { Commands::run("move $player_x $player_y"); 1; };
	    if (!$walk_ok) {
	        return (0, 'walk_failed', "Failed to walk to player '$player_name'");
	    }

	    usleep(500000);

	    # Send trade request
	    # OpenKore: "deal <player_name>" or "trade <player_name>"
	    my $deal_ok = eval { Commands::run("deal $player_name"); 1; };
	    if (!$deal_ok) {
	        return (0, 'trade_request_failed', "Failed to send trade request to '$player_name'");
	    }

	    # Update trade state
	    $_trade_state{active} = 1;
	    $_trade_state{partner} = $player_name;
	    $_trade_state{stage} = 'requested';
	    $_trade_state{started_at_ms} = _now_ms();

	    # Parse items to add
	    if ($items_str) {
	        my @items;
	        my @pairs = split(/,/, $items_str);
	        for my $_pair (@pairs) {
	            chomp $_pair;
	            $_pair =~ s/^\s+//;
	            $_pair =~ s/\s+$//;
	            next if $_pair eq '';
	            my ($item_name, $qty) = split(/:/, $_pair, 2);
	            next if !$item_name;
	            $item_name =~ s/^\s+//;
	            $item_name =~ s/\s+$//;
	            $qty ||= 1;
	            $qty =~ s/^\s+//;
	            $qty =~ s/\s+$//;
	            push @items, { name => $item_name, quantity => int($qty) };
	        }
	        $_trade_state{items_to_add} = \@items;
	    }

	    usleep(500000);

	    # Wait for trade to be accepted (we'll check on next poll)
	    # For now, assume accepted and add items
	    if ($_trade_state{items_to_add} && @{$_trade_state{items_to_add}}) {
	        for my $_item (@{$_trade_state{items_to_add}}) {
	            # Find item in inventory
	            my $_inv_idx = 0;
	            my $_found = 0;
	            if ($char && @{_char_inventory($char)}) {
	                for my $_inv_item (@{_char_inventory($char)}) {
	                    next unless ref($_inv_item) eq 'HASH';
	                    my $_inv_name = $_inv_item->{name} || '';
	                    if (lc($_inv_name) eq lc($_item->{name})) {
	                        $_found = 1;
	                        last;
	                    }
	                    $_inv_idx++;
	                }
	            }
	            if ($_found) {
	                # OpenKore: "deal_add <inventory_index> <quantity>"
	                my $add_ok = eval { Commands::run("deal_add $_inv_idx $_item->{quantity}"); 1; };
	                if ($add_ok) {
	                    debug "[trade] added item '$_item->{name}' x$_item->{quantity} to trade (inv_idx=$_inv_idx)\n", 'aiSidecarBridge', 2;
	                }
	                usleep(100000);
	            } else {
	                warning "[trade] item '$_item->{name}' not found in inventory, skipping\n", 'aiSidecarBridge', 1;
	            }
	        }
	    }

	    usleep(300000);

	    # Confirm the trade
	    my $confirm_ok = eval { Commands::run("deal_ok"); 1; };
	    if (!$confirm_ok) {
	        $_trade_state{active} = 0;
	        return (0, 'trade_confirm_failed', "Failed to confirm trade with '$player_name'");
	    }

	    $_trade_state{stage} = 'confirmed';
	    $_trade_state{active} = 0;

	    _post_event({
	        kind => 'bridge_event',
	        event_type => 'player.trade_completed',
	        severity => 'info',
	        text => "Trade completed with $player_name",
	        player_name => $player_name,
	    });

	    return (1, 'ok', "Trade completed with '$player_name'");
	}

	# ═══════════════════════════════════════════════════════════════════════════
	# ── NPC Shop Data Collection ──
	# ═══════════════════════════════════════════════════════════════════════════

	# ── Collect NPC shop data ──
	# Periodically visits known NPC shops to collect price data.
	# Records item names and prices, sends to sidecar for market intelligence.
	sub _collect_npc_shop_data {
	    my $now = _now_ms();
	    state $_last_npc_collect_ms = 0;
	    return if $now - $_last_npc_collect_ms < 300000;  # Every 5 minutes
	    $_last_npc_collect_ms = $now;

	    return if !_bridge_enabled();
	    return if !$registered;
	    return if !$char;

	    # Collect shop data from known NPCs on the current map
	    my $map = _safe_field_map() || '';
	    my @shop_data;

	    if (defined $main::npcsList && ref($main::npcsList) eq 'HASH') {
	        foreach my $_nid (keys %{$main::npcsList}) {
	            my $_n = $main::npcsList->{$_nid};
	            next unless ref($_n) eq 'HASH';
	            my $_nname = $_n->{name} || '';
	            next if $_nname eq '';

	            # Check if this NPC has shop data
	            if ($_n->{shop} && ref($_n->{shop}) eq 'ARRAY') {
	                my @items;
	                for my $_si (@{$_n->{shop}}) {
	                    next unless ref($_si) eq 'HASH';
	                    push @items, {
	                        name => $_si->{name} || '',
	                        price => $_si->{price} || 0,
	                        type => $_si->{type} || '',
	                    };
	                }
	                if (@items) {
	                    push @shop_data, {
	                        npc_name => $_nname,
	                        map => $map,
	                        x => $_n->{x} || 0,
	                        y => $_n->{y} || 0,
	                        items => \@items,
	                    };
	                    # Cache locally
	                    $_npc_shop_data{$_nname} = {
	                        map => $map,
	                        items => \@items,
	                    };
	                }
	            }
	        }
	    }

	    if (@shop_data) {
	        # Send to sidecar
	        my $resp = _http_post_json('/v2/market/npc_shops', {
	            meta => _meta(_bot_id()),
	            map => $map,
	            shops => \@shop_data,
	            collected_at => $now,
	        });
	        if ($resp && $resp->{status} >= 200 && $resp->{status} < 300) {
	            debug "[npc_shop_data] collected data for " . scalar(@shop_data) . " NPC shops on $map\n", 'aiSidecarBridge', 2;
	        }
	    }
	}

	# ═══════════════════════════════════════════════════════════════════════════
	# ── Player Vendor Data Collection ──
	# ═══════════════════════════════════════════════════════════════════════════

	# ── Collect player vendor data ──
	# Periodically scans town maps for player vendors.
	# Clicks on vendors to read their item lists.
	# Records item names, prices, quantities, sends to sidecar.
	sub _collect_vendor_data {
	    my $now = _now_ms();
	    state $_last_vendor_collect_ms = 0;
	    return if $now - $_last_vendor_collect_ms < 120000;  # Every 2 minutes
	    $_last_vendor_collect_ms = $now;

	    return if !_bridge_enabled();
	    return if !$registered;
	    return if !$char;

	    my $map = _safe_field_map() || '';
	    my @vendor_data;

	    if (defined $main::playersList && ref($main::playersList) eq 'HASH') {
	        foreach my $_pid (keys %{$main::playersList}) {
	            my $_p = $main::playersList->{$_pid};
	            next unless ref($_p) eq 'HASH';
	            my $_pname = $_p->{name} || '';
	            next if $_pname eq '';
	            next if $_pname eq ($char->{name} || '');  # Skip self

	            # Check if player is vending
	            if ($_p->{shop_open} || $_p->{vendor_title} || $_p->{vending}) {
	                my $title = $_p->{vendor_title} || $_p->{shop_title} || '';
	                my @items;

	                # If vendor items are visible in the player data, collect them
	                if ($_p->{vendor_items} && ref($_p->{vendor_items}) eq 'ARRAY') {
	                    for my $_vi (@{$_p->{vendor_items}}) {
	                        next unless ref($_vi) eq 'HASH';
	                        push @items, {
	                            name => $_vi->{name} || '',
	                            price => $_vi->{price} || 0,
	                            amount => $_vi->{amount} || 0,
	                        };
	                    }
	                }

	                push @vendor_data, {
	                    player_name => $_pname,
	                    map => $map,
	                    x => $_p->{x} || 0,
	                    y => $_p->{y} || 0,
	                    title => $title,
	                    items => \@items,
	                };

	                # Cache locally
	                $_player_vendor_data{$_pname} = {
	                    map => $map,
	                    x => $_p->{x} || 0,
	                    y => $_p->{y} || 0,
	                    title => $title,
	                    items => \@items,
	                };
	            }
	        }
	    }

	    if (@vendor_data) {
	        my $resp = _http_post_json('/v2/market/player_vendors', {
	            meta => _meta(_bot_id()),
	            map => $map,
	            vendors => \@vendor_data,
	            collected_at => $now,
	        });
	        if ($resp && $resp->{status} >= 200 && $resp->{status} < 300) {
	            debug "[vendor_data] collected data for " . scalar(@vendor_data) . " player vendors on $map\n", 'aiSidecarBridge', 2;
	        }
	    }
	}

	# ═══════════════════════════════════════════════════════════════════════════
	# ── Market Order Execution ──
	# ═══════════════════════════════════════════════════════════════════════════

	# ── Execute market order ──
	# When sidecar sends "market_buy <item> <max_price> <quantity>":
	#   Find cheapest vendor/NPC and buy
	# When sidecar sends "market_sell <item> <min_price> <quantity>":
	#   Set up vending or sell to NPC
	sub _execute_market_order {
	    my ($order_type, $item_name, $price, $quantity) = @_;
	    return (0, 'missing_params', 'order_type, item_name, price, and quantity required')
	        if !$order_type || !$item_name || !defined $price || !$quantity;

	    my $order_lc = lc($order_type);

	    if ($order_lc eq 'buy') {
	        return _execute_market_buy($item_name, $price, $quantity);
	    } elsif ($order_lc eq 'sell') {
	        return _execute_market_sell($item_name, $price, $quantity);
	    } else {
	        return (0, 'unknown_order_type', "Unknown market order type '$order_type' (use 'buy' or 'sell')");
	    }
	}

	# ── Execute market buy order ──
	# Finds the cheapest source (NPC shop or player vendor) and buys the item.
	sub _execute_market_buy {
	    my ($item_name, $max_price, $quantity) = @_;
	    my $item_lc = lc($item_name);

	    # Strategy 1: Check NPC shops first (usually cheapest)
	    for my $_npc_name (keys %_npc_shop_data) {
	        my $_npc = $_npc_shop_data{$_npc_name};
	        next unless $_npc && $_npc->{items} && ref($_npc->{items}) eq 'ARRAY';
	        for my $_item (@{$_npc->{items}}) {
	            next unless ref($_item) eq 'HASH';
	            my $_iname = lc($_item->{name} || '');
	            next if $_iname ne $item_lc;
	            my $_iprice = $_item->{price} || 0;
	            next if $max_price > 0 && $_iprice > $max_price;

	            # Found a matching NPC shop — buy from it
	            my ($ok, $code, $msg) = _open_npc_shop($_npc_name, $item_name, $quantity);
	            if ($ok) {
	                _post_event({
	                    kind => 'bridge_event',
	                    event_type => 'market.buy_executed',
	                    severity => 'info',
	                    text => "Market buy: $quantity x $item_name from NPC $_npc_name at $_iprice z",
	                    source => 'npc',
	                    source_name => $_npc_name,
	                    item_name => $item_name,
	                    quantity => $quantity,
	                    price => $_iprice,
	                });
	                return ($ok, $code, $msg);
	            }
	        }
	    }

	    # Strategy 2: Check player vendors
	    for my $_pv_name (keys %_player_vendor_data) {
	        my $_pv = $_player_vendor_data{$_pv_name};
	        next unless $_pv && $_pv->{items} && ref($_pv->{items}) eq 'ARRAY';
	        for my $_item (@{$_pv->{items}}) {
	            next unless ref($_item) eq 'HASH';
	            my $_iname = lc($_item->{name} || '');
	            next if $_iname ne $item_lc;
	            my $_iprice = $_item->{price} || 0;
	            next if $max_price > 0 && $_iprice > $max_price;

	            # Found a matching player vendor — buy from it
	            my ($ok, $code, $msg) = _open_player_vendor($_pv_name, $item_name, $quantity);
	            if ($ok) {
	                _post_event({
	                    kind => 'bridge_event',
	                    event_type => 'market.buy_executed',
	                    severity => 'info',
	                    text => "Market buy: $quantity x $item_name from player $_pv_name at $_iprice z",
	                    source => 'vendor',
	                    source_name => $_pv_name,
	                    item_name => $item_name,
	                    quantity => $quantity,
	                    price => $_iprice,
	                });
	                return ($ok, $code, $msg);
	            }
	        }
	    }

	    return (0, 'no_source_found', "No source found for '$item_name' at or under $max_price z");
	}

	# ── Execute market sell order ──
	# Sells item to NPC shop or sets up vending.
	sub _execute_market_sell {
	    my ($item_name, $min_price, $quantity) = @_;

	    # Strategy 1: Sell to NPC shop (instant, but lower price)
	    # Find an NPC that buys this item
	    for my $_npc_name (keys %_npc_shop_data) {
	        my $_npc = $_npc_shop_data{$_npc_name};
	        next unless $_npc && $_npc->{items} && ref($_npc->{items}) eq 'ARRAY';
	        for my $_item (@{$_npc->{items}}) {
	            next unless ref($_item) eq 'HASH';
	            my $_iname = lc($_item->{name} || '');
	            next if $_iname ne lc($item_name);
	            my $_iprice = $_item->{price} || 0;

	            # NPC buy price is typically half the sell price
	            my $buy_price = int($_iprice / 2);
	            next if $min_price > 0 && $buy_price < $min_price;

	            # Find item in inventory
	            my $_inv_idx = 0;
	            my $_found = 0;
	            if ($char && @{_char_inventory($char)}) {
	                for my $_inv_item (@{_char_inventory($char)}) {
	                    next unless ref($_inv_item) eq 'HASH';
	                    my $_inv_name = $_inv_item->{name} || '';
	                    if (lc($_inv_name) eq lc($item_name)) {
	                        $_found = 1;
	                        last;
	                    }
	                    $_inv_idx++;
	                }
	            }
	            if (!$_found) {
	                return (0, 'item_not_in_inventory', "Item '$item_name' not found in inventory");
	            }

	            # Walk to NPC and sell
	            my $npc_x = $_npc->{x} || 0;
	            my $npc_y = $_npc->{y} || 0;
	            eval { Commands::run("move $npc_x $npc_y"); 1; };
	            usleep(500000);
	            eval { Commands::run("talknpc $npc_x $npc_y"); 1; };
	            usleep(300000);
	            eval { Commands::run("talk c"); 1; };
	            usleep(200000);
	            eval { Commands::run("talk resp 1"); 1; };  # "Sell" option
	            usleep(200000);
	            eval { Commands::run("sell $_inv_idx $quantity"); 1; };
	            usleep(200000);
	            eval { Commands::run("talk close"); 1; };

	            _post_event({
	                kind => 'bridge_event',
	                event_type => 'market.sell_executed',
	                severity => 'info',
	                text => "Market sell: $quantity x $item_name to NPC $_npc_name at $buy_price z",
	                source => 'npc',
	                source_name => $_npc_name,
	                item_name => $item_name,
	                quantity => $quantity,
	                price => $buy_price,
	            });

	            return (1, 'ok', "Sold $quantity x $item_name to NPC '$_npc_name' at $buy_price z");
	        }
	    }

	    # Strategy 2: Set up vending (better price, but takes time)
	    my $vending_title = "Selling $item_name";
	    my $vending_items = "$item_name:$min_price";
	    my ($ok, $code, $msg) = _setup_vending($vending_title, $vending_items);
	    if ($ok) {
	        _post_event({
	            kind => 'bridge_event',
	            event_type => 'market.sell_vending',
	            severity => 'info',
	            text => "Market sell: vending $quantity x $item_name at $min_price z",
	            source => 'vending',
	            item_name => $item_name,
	            quantity => $quantity,
	            price => $min_price,
	        });
	        return ($ok, $code, $msg);
	    }

	    return (0, 'no_buyer_found', "No buyer found for '$item_name' at or above $min_price z");
	}

	# ═══════════════════════════════════════════════════════════════════════════
	# ── Trade Negotiation ──
	# ═══════════════════════════════════════════════════════════════════════════

	# ── Negotiate trade with another player ──
	# When sidecar sends "negotiate <player_name> <item_want> <item_give>":
	# Initiates trade negotiation, sends trade request, waits for response,
	# if accepted, completes trade, reports result.
	sub _negotiate_trade {
	    my ($player_name, $item_want, $item_give) = @_;
	    return (0, 'missing_params', 'player_name, item_want, and item_give required')
	        if !$player_name || !$item_want || !$item_give;

	    # Find the player
	    my ($player_id, $player_x, $player_y);
	    if (defined $main::playersList && ref($main::playersList) eq 'HASH') {
	        foreach my $_pid (keys %{$main::playersList}) {
	            my $_p = $main::playersList->{$_pid};
	            next unless ref($_p) eq 'HASH';
	            my $_pname = lc($_p->{name} || '');
	            next if $_pname ne lc($player_name);
	            $player_id = $_pid;
	            $player_x = $_p->{x} || 0;
	            $player_y = $_p->{y} || 0;
	            last;
	        }
	    }

	    if (!$player_id) {
	        return (0, 'player_not_found', "Player '$player_name' not found nearby");
	    }

	    # Walk to player
	    my $walk_ok = eval { Commands::run("move $player_x $player_y"); 1; };
	    if (!$walk_ok) {
	        return (0, 'walk_failed', "Failed to walk to player '$player_name'");
	    }

	    usleep(500000);

	    # Send a chat message proposing the trade
	    my $chat_msg = "Hi $player_name, I'd like to trade my $item_give for your $item_want. Interested?";
	    my $chat_ok = eval { Commands::run("pm $player_name $chat_msg"); 1; };
	    if (!$chat_ok) {
	        # Fallback: try public chat
	        eval { Commands::run("c $chat_msg"); 1; };
	    }

	    usleep(1000000);  # Wait 1 second for response

	    # Send trade request
	    my $deal_ok = eval { Commands::run("deal $player_name"); 1; };
	    if (!$deal_ok) {
	        return (0, 'trade_request_failed', "Failed to send trade request to '$player_name'");
	    }

	    # Update trade state
	    $_trade_state{active} = 1;
	    $_trade_state{partner} = $player_name;
	    $_trade_state{stage} = 'negotiating';
	    $_trade_state{started_at_ms} = _now_ms();

	    usleep(500000);

	    # Add our item to the trade window
	    my $_inv_idx = 0;
	    my $_found = 0;
	    if ($char && @{_char_inventory($char)}) {
	        for my $_inv_item (@{_char_inventory($char)}) {
	            next unless ref($_inv_item) eq 'HASH';
	            my $_inv_name = $_inv_item->{name} || '';
	            if (lc($_inv_name) eq lc($item_give)) {
	                $_found = 1;
	                last;
	            }
	            $_inv_idx++;
	        }
	    }

	    if ($_found) {
	        my $add_ok = eval { Commands::run("deal_add $_inv_idx 1"); 1; };
	        if ($add_ok) {
	            debug "[negotiate] added item '$item_give' to trade (inv_idx=$_inv_idx)\n", 'aiSidecarBridge', 2;
	        }
	    } else {
	        $_trade_state{active} = 0;
	        return (0, 'item_not_found', "Item '$item_give' not found in inventory");
	    }

	    usleep(500000);

	    # Confirm the trade
	    my $confirm_ok = eval { Commands::run("deal_ok"); 1; };
	    if (!$confirm_ok) {
	        $_trade_state{active} = 0;
	        return (0, 'trade_confirm_failed', "Failed to confirm trade with '$player_name'");
	    }

	    $_trade_state{stage} = 'confirmed';
	    $_trade_state{active} = 0;

	    _post_event({
	        kind => 'bridge_event',
	        event_type => 'player.trade_negotiated',
	        severity => 'info',
	        text => "Trade negotiated: $item_give for $item_want with $player_name",
	        player_name => $player_name,
	        item_give => $item_give,
	        item_want => $item_want,
	    });

	    return (1, 'ok', "Trade negotiated: gave $item_give for $item_want with '$player_name'");
	}

	# ═══════════════════════════════════════════════════════════════════════════
	# ── Periodic commerce data collection (called from _poll_next_action) ──
	# ═══════════════════════════════════════════════════════════════════════════

	# ── Periodic NPC shop data collection ──
	# Called from _poll_next_action to collect NPC shop price data.
	sub _periodic_npc_shop_collect {
	    _collect_npc_shop_data();
	}

	# ── Periodic player vendor data collection ──
	# Called from _poll_next_action to collect player vendor data.
	sub _periodic_vendor_collect {
	    _collect_vendor_data();
	}

	1;
