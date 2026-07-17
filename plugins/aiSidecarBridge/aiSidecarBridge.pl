package aiSidecarBridge;

use strict;
use warnings;
use feature 'state';

use Commands;
use FileParsers qw(parseConfigFile);
use Globals qw(%config $char $field @ai_seq $net %monsters %players %npcs $monstersList $playersList $npcsList);
use IO::Socket::INET;
use Globals qw($char);
use Log qw(debug message warning);
use Network;
use Plugins;
use Scalar::Util qw(reftype);
use Settings;
use Time::HiRes qw(alarm time usleep);

# Anti-detection: random delay to simulate human reaction time
# Reduced for pro-level reaction speed: 10-50ms instead of 100-500ms
my $ANTI_DETECTION_ENABLED = 1;
my $ANTI_DETECTION_MIN_DELAY_MS = 10;
my $ANTI_DETECTION_MAX_DELAY_MS = 50;

# ── Hardcoded safety net: White Potion ──
# This is the ABSOLUTE LAST RESORT item — always available as fallback.
# Used only when dynamic heal cache fails and HP is critically low.
my $HARDCODED_FALLBACK_ITEM = 'White Potion';

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
	['packet_pubMsg', \&on_chat_message, 'publicchat'],
	['packet_partyMsg', \&on_chat_message, 'partychat'],
	['packet_guildMsg', \&on_chat_message, 'guildchat'],
	['packet_sysMsg', \&on_chat_message, 'systemchat'],
	['packet_mapChange', \&on_legacy_packet_hook, 'packet_legacy.map_change'],
	['packet_skilluse', \&on_legacy_packet_hook, 'packet_legacy.skill_use'],
	['packet_areaSpell', \&on_legacy_packet_hook, 'packet_legacy.area_spell'],
	['post_configModify', \&on_post_config_modify, undef],
	['post_bulkConfigModify', \&on_post_bulk_config_modify, undef],
	['Commands::run/post', \&on_command_run_post, undef],
);

my $control_handle;
my $policy_handle;

my %bridge_cfg;
my %bridge_policy;
my @policy_allow;
my @policy_deny;

my $registered = 0;
my $next_snapshot_at_ms = 0;
my $next_poll_at_ms = 0;
my $next_ack_at_ms = 0;
my $next_telemetry_at_ms = 0;
my $next_register_at_ms = 0;
my $next_event_ingest_at_ms = 0;
my $next_chat_ingest_at_ms = 0;
my $next_config_ingest_at_ms = 0;

my @ack_queue;
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
my $last_map_name = '';
my $last_route_signature = '';
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

sub on_unload {
	_cleanup_runtime();
	Plugins::delHooks($hooks);
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
}

sub on_start3 {
	if (!$json_available) {
		warning "[aiSidecarBridge] JSON::PP is unavailable, bridge is disabled (fail-open).\n";
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

	_attempt_register('start3');
}

sub on_mainLoop_pre {
	return unless _bridge_enabled();
	my $now = _now_ms();

	if (_cfg_bool('aiSidecar_snapshotEnabled', 1) && $now >= $next_snapshot_at_ms) {
		$next_snapshot_at_ms = $now + _cfg_int('aiSidecar_snapshotIntervalMs', 500);
		_send_snapshot();
		_check_bridge_reflexes();
	}

	_track_lifecycle_transitions();
	_track_ai_sequence_transition();
}

sub on_mainLoop_post {
	return unless _bridge_enabled();
	my $now = _now_ms();
	_probe_actor_post_parse($now);

	if (!$registered && $now >= $next_register_at_ms) {
	    $next_register_at_ms = $now + _cfg_int('aiSidecar_registerRetryMs', 1000);
	    _attempt_register('retry');
	}

	if (_cfg_bool('aiSidecar_actionPollEnabled', 1) && $now >= $next_poll_at_ms) {
	    my $poll_ok = _poll_next_action();
	    my $next_delay_ms = $poll_ok ? _cfg_int('aiSidecar_pollIntervalMs', 100) : _poll_failure_delay_ms();
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
		_apply_bot_config();
		_discover_shops();
		_discover_portals();
	_send_discovery_data();
		_survival_check();
		_teamplay_check();
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
	}
	$last_map_name = $map if defined $map && $map ne '';

	my $ai_top = _safe_ai_seq_top();
	my $x = undef;
	my $y = undef;
	if ($char) {
		my $pos = eval {
			if ($char->{pos_to} && ref $char->{pos_to} eq 'HASH') {
				return $char->{pos_to};
			}
			if ($char->{pos} && ref $char->{pos} eq 'HASH') {
				return $char->{pos};
			}
			return undef;
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
		aiSidecar_pollWhenDisconnected => 0,
		aiSidecar_actionPollEnabled => 1,
		aiSidecar_pollIntervalMs => 500,
		aiSidecar_pollFailureBackoffBaseMs => 600,
		aiSidecar_pollFailureBackoffMaxMs => 6000,
		aiSidecar_pollFailureResetRegistrationAfter => 3,
		aiSidecar_ackEnabled => 1,
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
		aiSidecarPolicy_allow_23 => 'warp',
		aiSidecarPolicy_allow_24 => 'portals',
		aiSidecarPolicy_allow_25 => 'route',
		aiSidecarPolicy_allow_26 => 'stats',
		aiSidecarPolicy_allow_27 => 'stat',
		aiSidecarPolicy_allow_28 => 'skills_add',
		aiSidecarPolicy_allow_29 => 'skills',
		aiSidecarPolicy_allow_30 => 'stat_add',
		aiSidecarPolicy_allow_31 => 'stats_add',
		aiSidecarPolicy_allow_32 => 'deal',
		aiSidecarPolicy_allow_33 => 'trade',
		aiSidecarPolicy_allow_34 => 'friend',
		aiSidecarPolicy_allow_35 => 'storage',
		aiSidecarPolicy_allow_36 => 'cart',
		aiSidecarPolicy_allow_37 => 'mail',
		aiSidecarPolicy_allow_38 => 'storageadd',
		aiSidecarPolicy_allow_39 => 'storageget',
		aiSidecarPolicy_allow_40 => 'use',
		aiSidecarPolicy_allow_41 => 'equip',
		aiSidecarPolicy_allow_42 => 'unequip',
		aiSidecarPolicy_allow_43 => 'items',
		aiSidecarPolicy_allow_44 => 'inventory',
		aiSidecarPolicy_allow_45 => 'look',
		aiSidecarPolicy_allow_46 => 'talk',
		aiSidecarPolicy_allow_47 => 'talknpc',
		aiSidecarPolicy_allow_48 => 'response',
		aiSidecarPolicy_allow_49 => 'c',
		aiSidecarPolicy_allow_50 => 'pm',
		aiSidecarPolicy_allow_51 => 'join',
		aiSidecarPolicy_allow_52 => 'leave',
		aiSidecarPolicy_allow_53 => 'exp',
		aiSidecarPolicy_allow_54 => 'storageprice',
		aiSidecarPolicy_allow_55 => 'reparse',
		aiSidecarPolicy_allow_56 => 'ss',
		aiSidecarPolicy_allow_57 => '@go',

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
			identity_char_name => ($char ? ($char->{name} || '') : ''),
			identity_override => _cfg('aiSidecar_botIdentity', ''),
			profile => eval { $profiles::profile } || '',
			control_folder => _active_control_folder(),
		},
	};

	my $resp = _http_post_json('/v1/ingest/register', $payload);
	if ($resp && $resp->{status} >= 200 && $resp->{status} < 300) {
		$registered = 1;
		debug "[aiSidecarBridge] sidecar registration succeeded\n", 'aiSidecarBridge', 2;
		return;
	}

	$registered = 0;
	_throttled_warning('register_failed', '[aiSidecarBridge] sidecar registration failed, running fail-open.');
	_emit_telemetry('warning', 'bridge', 'register_failed', 'sidecar registration failed');
}

sub _send_snapshot {
    return if !_bridge_enabled();
    return if !$registered;  # Don't send snapshots until registered
    if (!$net || $net->getState() != Network::IN_GAME) {
        return if !_cfg_bool('aiSidecar_pollWhenDisconnected', 0);
    }

    my $snapshot = _build_snapshot_payload();
    my $resp = _http_post_json('/v1/ingest/snapshot', $snapshot);
    if (!$resp || $resp->{status} < 200 || $resp->{status} >= 300) {
        _throttled_warning('snapshot_failed', '[aiSidecarBridge] snapshot push failed, fail-open retained.');
        _emit_telemetry('warning', 'bridge', 'snapshot_failed', 'snapshot push failed');
    }

    if (_cfg_bool('aiSidecar_v2Enabled', 1) && _cfg_bool('aiSidecar_actorsEnabled', 1)) {
        _send_actor_delta_from_snapshot($snapshot);
    }
}

sub _build_snapshot_payload {
	my $bot_id = _bot_id();
	my $max_raw = _cfg_int('aiSidecar_maxRawChars', 256);

	my ($x, $y);
	my $map = '';
	if ($char) {
		my $pos = eval {
			if ($char->{pos_to} && ref $char->{pos_to} eq 'HASH') {
				return $char->{pos_to};
			}
			if ($char->{pos} && ref $char->{pos} eq 'HASH') {
				return $char->{pos};
			}
			return undef;
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
	if ($char && $char->{inventory} && ref $char->{inventory} eq 'ARRAY') {
		$item_count = scalar @{$char->{inventory}};
	}

	my $raw = {
		char_name  => _trim($char ? ($char->{name} || '') : '', $max_raw),
		master     => _trim($config{master} || '', $max_raw),
		ai_sequence => _trim($ai_top || '', $max_raw),
		ai_queue   => _trim(join(',', @ai_seq[0 .. ($#ai_seq < 4 ? $#ai_seq : 4)]), $max_raw),
		in_game => ($net && $net->getState() == Network::IN_GAME) ? JSON::PP::true() : JSON::PP::false(),
		net_state => ($net ? ($net->getState() + 0) : -1),
		reconnect_age_s => ($last_disconnect_at_ms > 0 && $net && $net->getState() == Network::IN_GAME)
			? ((_now_ms() - $last_disconnect_at_ms) / 1000.0)
			: 0.0,
		death_count => 0 + $death_count,
		respawn_state => _trim($respawn_state, 32),
		route_churn_count => 0 + $route_churn_count,
		route_failure_count => 0 + $route_failure_count,
	};

	# --- Progression digest (job, level, exp) ---
	my $progression = {};
	if ($char) {
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
			$p{stat_points}  = $char->{points_free}  if defined $char->{points_free};
			$p{job_name}     = $char->{jobName}      if defined $char->{jobName};
			\%p;
		} || {};
	}

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
		my $party_members = ($char && $char->{party} && ref($char->{party}{party}{member}) eq 'HASH')
			? $char->{party}{party}{member}
			: undef;

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
							$relation = 'party' if exists $party_members->{$row->{binID}};
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
			sp         => $char ? $char->{sp}         : undef,
			sp_max     => $char ? $char->{sp_max}     : undef,
			weight     => $char ? $char->{weight}     : undef,
			weight_max => $char ? $char->{weight_max} : undef,
		},
		combat => {
			ai_sequence  => $ai_top || undef,
			target_id    => undef,
			is_in_combat => $in_combat,
		},
		inventory => {
			zeny       => $char ? $char->{zeny} : undef,
			item_count => $item_count,
		},
		progression => $progression,
		skills      => \@skills_list,
		actors      => \@actors,
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
		return () if !$list_obj->can('getItems');
		return $list_obj->getItems();
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
		return 1 if !_cfg_bool('aiSidecar_pollWhenDisconnected', 0);
	}

	my $poll_id = _trace_id();
	my $resp = _http_post_json('/v1/actions/next', {
		meta => _meta(_bot_id()),
		poll_id => $poll_id,
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
	return 1 if ref($json->{action}) ne 'HASH';

	_execute_action($poll_id, $json->{action});
	return 1;
}

sub _execute_action {
	my ($poll_id, $action) = @_;

	my $action_id = $action->{action_id} || 'unknown_action';
	my $kind = lc($action->{kind} || 'command');
	my $command = defined $action->{command} ? $action->{command} : '';
	my $metadata = ref($action->{metadata}) eq 'HASH' ? $action->{metadata} : {};
	my $started = _now_ms();
	
	# Anti-detection: random delay before executing action
	if ($ANTI_DETECTION_ENABLED) {
		my $delay_ms = $ANTI_DETECTION_MIN_DELAY_MS + int(rand($ANTI_DETECTION_MAX_DELAY_MS - $ANTI_DETECTION_MIN_DELAY_MS + 1));
		usleep($delay_ms * 1000) if $delay_ms > 0;
	}
	
	my ($effective_command, $rewrite_kind) = _rewrite_runtime_command($command, $metadata);

	my ($success, $result_code, $msg) = (0, 'invalid_action', 'invalid action payload');

	if ($kind eq 'macro_reload') {
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
		my $ok = eval { Commands::run($effective_command); 1; };
		($success, $result_code, $msg) = $ok ? (1, 'ok', 'coordinate move executed') : (0, 'dispatch_error', $@);
	} elsif ($rewrite_kind eq 'chat_sent') {
		($success, $result_code, $msg) = (1, 'ok', 'chat message sent');
	} elsif ($rewrite_kind eq 'go_command_sent') {
		($success, $result_code, $msg) = (1, 'ok', '@go command sent as chat');
	} elsif ($rewrite_kind =~ /^use_item_/) {
		($success, $result_code, $msg) = (1, 'ok', "item use handled: $rewrite_kind");
	} elsif ($rewrite_kind eq 'attack_skill_delegated') {
		($success, $result_code, $msg) = (1, 'ok', 'attack skill delegated to auto-AI');
	} elsif ($rewrite_kind eq 'ai_manual_already_manual' || $rewrite_kind eq 'ai_auto_already_auto') {
		($success, $result_code, $msg) = (1, 'ok', "AI mode already satisfied: $rewrite_kind");
	} elsif ($rewrite_kind eq 'map_move_already_set') {
		($success, $result_code, $msg) = (1, 'ok', 'lockMap already set to target');
	} elsif ($rewrite_kind eq 'map_move_low_hp_no_toggle') {
		($success, $result_code, $msg) = (1, 'ok', 'ai_manual suppressed: critically low HP');
	} elsif ($rewrite_kind =~ /^use_item_not_found_/) {
		($success, $result_code, $msg) = (1, 'ok', "item not found in inventory: $rewrite_kind");
	} elsif ($effective_command eq '') {
		($success, $result_code, $msg) = (0, 'empty_command', 'empty command');
	} elsif (length($effective_command) > _cfg_int('aiSidecar_maxCommandLength', 160)) {
		($success, $result_code, $msg) = (0, 'command_too_long', 'command length exceeds policy');
	} elsif (!_command_allowed($effective_command)) {
		($success, $result_code, $msg) = (0, 'policy_rejected', 'command rejected by bridge policy');
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
		my $event_type = $event->{reflex} || $event->{event_type} || 'unknown';
		my $severity = $event->{severity} || 'info';
		my %tags = (); my %numeric = (); my $text = $event->{text} || '';
		my %payload = ();
		while (my ($k, $v) = each %$event) {
			next if $k eq 'kind' || $k eq 'reflex' || $k eq 'severity' || $k eq 'text' || $k eq 'event_type' || $k eq 'timestamp';
			if (!defined $v) { next }
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
	    "Connection: keep-alive",
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
		    1;
		};
	$io_error = $@ if !$ok;
	alarm(0);
	# Don't close the socket on success — keep-alive reuses it
	# Only close on error to reset the connection
	if (!$ok) {
	    close $sock;

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

sub _rewrite_runtime_command {
	my ($command, $metadata) = @_;
	my $trimmed = _trim(_scalarize($command), 256);
	my $normalized = lc($trimmed || '');
	$metadata = {} if ref($metadata) ne 'HASH';

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

	# Handle map-name moves: "move prt_fild08" → set lockMap + ai auto
	# OpenKore's auto AI uses lockMap to navigate to and stay on a map.
	# Setting lockMap first, then enabling auto AI, gives the bot a
	# meaningful destination instead of aimless wandering.
	# Also handle coordinate moves: "move 181 186" → direct movement
	if ($normalized =~ /^move\s+(.+)$/) {
		my $target = $1;
		# Check if target is numeric coordinates (x y) — not a map name
		if ($target =~ /^\d+\s+\d+$/) {
			# Coordinate-based move — pass through directly, do NOT set lockMap
			# OpenKore's move command accepts x y coordinates for pixel movement
			return ($trimmed, 'coordinate_move_raw');
		}
		# Only set lockMap for valid map names (not special commands)
		if ($target !~ /^(savepoint|random_walk_seek)$/) {
			my $current_lockMap = defined $::config{"lockMap"} ? $::config{"lockMap"} : '';
			$::config{"lockMap"} = $target;
			$::config{"lockMap_x"} = "";
			$::config{"lockMap_y"} = "";
			$::config{"lockMap_randX"} = 0;
			$::config{"lockMap_randY"} = 0;
			# Only toggle AI if lockMap actually changed — prevents rapid cycling
			if ($current_lockMap ne '' && $current_lockMap eq $target) {
				# lockMap already set to this target, no need to toggle
				if (_ai_already_auto_mode()) {
					return ('', 'map_move_already_set');
				}
				return ('ai auto', 'map_move_rewritten');
			}
		}
		if (_ai_already_auto_mode()) {
			# Don't toggle manual if critically low HP — survival first
			my $_rw_hp = $main::char ? $main::char->{hp} : 9999;
			my $_rw_hp_max = $main::char ? $main::char->{hp_max} : 1;
			if ($_rw_hp_max > 0 && ($_rw_hp / $_rw_hp_max) < 0.50) {
				return ('', 'map_move_low_hp_no_toggle');
			}
			# Toggle AI mode to force route recalculation with new lockMap
			return ('ai manual', 'map_move_toggle_manual');
		}
		return ('ai auto', 'map_move_rewritten');
	}

	if ($normalized eq 'take') {
		return ('', 'bare_take_delegated');
	}

	# Handle chat — send a public chat message
	if ($normalized =~ /^chat\s+(.+)$/i) {
	    my $msg = $1;
	    Commands::run("c $msg");
	    return ('', 'chat_sent');
	}
    
	# Handle teleport — OpenKore doesn't have a teleport command.
	# Setting AI to auto mode lets OpenKore's auto-logic handle movement.
	# If the bot has the Teleport skill, use it directly.
	if ($normalized eq 'teleport') {
	    if (_ai_already_auto_mode()) {
	        return ('', 'teleport_already_auto');
	    }
	    # Try using the teleport skill first
	    my $teleport_skill = eval { $char->skills->get('AL_TELEPORT') } || eval { $char->skills->get('TF_TELEPORT') };
	    if ($teleport_skill) {
	        return ('use_skill teleport', 'teleport_skill_used');
	    }
	    return ('ai auto', 'teleport_rewritten');
	}

		# Handle skills_add — allocate skill points (e.g. skills_add NV_BASIC 3)
	# This enables the LLM / Pro RO player to learn skills like Basic Skill 3
	if ($normalized =~ /^skills_add\s+(\w+)\s*(\d*)$/) {
	    my $skill_id = uc($1);
	    my $level = $2 || 1;
	    if (defined $char && defined $char->{skills}) {
	        my $current = eval { $char->{skills}->{$skill_id}{lv} } || 0;
	        my $needed = $level - $current;
	        if ($needed > 0) {
	            my $ok = eval { $char->{skills}->addPoint($skill_id, $needed); 1 };
	            return ('', $ok ? "skill_points_added_$skill_id" : "skill_add_failed_$skill_id");
	        }
	        return ('', "skill_already_learned_$skill_id");
	    }
	    return ('', 'skill_add_no_skills_obj');
	}

	# Handle stats_add — allocate stat points (e.g. stats_add STR 1)
	if ($normalized =~ /^stats_add\s+(\w+)\s*(\d*)$/) {
	    my $stat = uc($1);
	    my $amount = $2 || 1;
	    my $ok = eval { Commands::run("stat_add $stat $amount"); 1 };
	    return ('', $ok ? "stat_points_added_$stat" : "stat_add_failed_$stat");
	}

	# Handle attack_skill — already handled by auto-AI, no-op
	if ($normalized =~ /^attack_skill\b/) {
	    return ('', 'attack_skill_delegated');
	}

	# Handle use <item> — find item in inventory and use it
	# We must look up the item by name in the character's inventory
	# and call $item->use() directly.
	if ($normalized =~ /^use\s+(.+)$/) {
	    my $item_name = $1;
	    # Normalize: replace underscores with spaces for matching
	    $item_name =~ s/_/ /g;
	    # Guard: $char must be defined and have inventory
	    if (defined $char && defined $char->{inventory}) {
	        # Try exact match first
	        my $item = eval { $char->inventory->getByName($item_name) };
	        if ($item) {
	            $item->use();
	            return ('', "use_item_$item_name");
	        }
	        # Try case-insensitive partial match
	        my @all_items = eval { @{$char->{inventory}} };
	        if (@all_items) {
	            my $lc_name = lc($item_name);
	            my @matches = grep { defined $_ && defined $_->{name} && lc($_->{name}) eq $lc_name } @all_items;
	            if (!@matches) {
	                @matches = grep { defined $_ && defined $_->{name} && lc($_->{name}) =~ /\Q$lc_name\E/ } @all_items;
	            }
	            if (@matches) {
	                $matches[0]->use();
	                return ('', "use_item_$item_name");
	            }
	        }
	    }
	    # Item not found or char not ready — return empty to suppress error
	    return ('', "use_item_not_found_$item_name");
	}

	# Handle ai manual — no-op if already in manual mode
	if ($normalized eq 'ai manual') {
	    if (!_ai_already_auto_mode()) {
	        return ('', 'ai_manual_already_manual');
	    }
	    return ($trimmed, '');
	}

	# Handle ai auto — no-op if already in auto mode
	if ($normalized eq 'ai auto') {
	    if (_ai_already_auto_mode()) {
	        return ('', 'ai_auto_already_auto');
	    }
	    return ($trimmed, '');
	}

	# Handle @go command — must be sent as chat, not console command
	# OpenKore's Commands::run doesn't handle @go natively
	if ($normalized =~ /^\@go\s+(.+)$/i) {
	    my $go_target = $1;
	    Commands::run("c \@go $go_target");
	    return ('', 'go_command_sent');
	}

	return ($trimmed, '');
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

	my $master = _normalize_identity_part($config{master}, 'unknown_master');
	my $identity_override = _normalize_identity_part(_cfg('aiSidecar_botIdentity', ''), '');
	my $char_name = _normalize_identity_part($char && $char->{name} ? $char->{name} : '', '');
	my $username = _normalize_identity_part($config{username}, 'unknown_user');
	my $identity = $identity_override ne '' ? $identity_override : ($char_name ne '' ? $char_name : $username);
	return "$master:$identity";
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
my $_last_ai_toggle_ms = 0;
my $_last_ai_mode = '';
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
{
	my %_reflex_last_fired;

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
		# Add 50-200ms random delay before action to look human
		my $delay_ms = int(rand(150)) + 50;
		select(undef, undef, undef, $delay_ms / 1000.0) if $delay_ms > 0;
	}

	# ── Cached healing resources (from sidecar config push) ──
	my @_heal_items;
	my @_heal_skills;
	my $_heal_cache_last_update_ms = 0;

	sub _update_heal_cache {
		my $now = _now_ms();
		return if $now - $_heal_cache_last_update_ms < 250;
		$_heal_cache_last_update_ms = $now;

		# Read from sidecar-pushed config (comma-separated lists)
		my $items_str = _cfg('aiSidecar_healItems', 'Red Potion,Orange Potion,White Potion');
		@_heal_items = split /,/, $items_str;
		@_heal_items = grep { $_ ne '' } @_heal_items;

		my $skills_str = _cfg('aiSidecar_healSkills', '');
		@_heal_skills = split /,/, $skills_str;
		@_heal_skills = grep { $_ ne '' } @_heal_skills;
	}

	# Last time we initiated town recovery (to prevent re-trigger spam)
my $_last_prontera_recovery_ms = 0;

sub _check_bridge_reflexes {
		my $now = _now_ms();
		return if !$char;

		# ── Compute current vitals ──
		my $hp = $char->{hp} || 0;
		my $hp_max = $char->{hp_max} || 1;
		my $sp = $char->{sp} || 0;
		my $sp_max = $char->{sp_max} || 1;
		my $hp_ratio = ($hp_max > 0) ? $hp / $hp_max : 1;
		my $sp_ratio = ($sp_max > 0) ? $sp / $sp_max : 1;
		my $weight = $char->{weight} || 0;
		my $weight_max = $char->{weight_max} || 1;
		my $weight_ratio = ($weight_max > 0) ? $weight / $weight_max : 0;
		my $map = $char->{map} || '';
		my $job_name = ($char && $char->{job_name}) ? _trim(_scalarize($char->{job_name}), 64) : '';
		my $base_level = $char->{base_level} || 1;
		my $job_level = $char->{job_level} || 1;
		my $in_combat = @ai_seq && $ai_seq[0] =~ /^(?:attack|skill_use|route|follow)/ ? 1 : 0;

		# Count aggro: monsters that have dealt damage to us
		my $aggro_count = 0;
		if ($monstersList) {
			for my $monster (@{$monstersList}) {
				next if !$monster;
				$aggro_count++ if $monster->{dmg_to_us} || $monster->{dmg_to_you};
			}
		}

		# ═══════════════════════════════════════════
		# REFLEX #1 — EMERGENCY HEAL REFLEX (FIRST PRIORITY)
		# ═══════════════════════════════════════════
		# Threshold: 50% HP (preemptive) → bridge reflex fires at <50%
		# Action: Use best available heal item/skill INSTANTLY
		# Rules:
		#   - No random delay on emergency heal (survival critical)
		#   - Dynamic selection from sidecar-pushed config
		#   - Hardcoded White Potion fallback (absolute safety net)
		#   - If nothing available: immediately trigger emergency survival
		my $heal_triggered = 0;

		if ($hp_ratio < 0.50 && $hp > 0) {
			_update_heal_cache();

			# Try config-pushed items first (dynamic, class-aware)
			for my $item_name (@_heal_items) {
				$item_name = _trim($item_name);
				next if !$item_name;
				my $item = eval { Actor::Item::get($item_name) };
				if ($item && $item->{amount} && $item->{amount} > 0) {
					warning "[aiSidecarBridge] bridge_reflex:emergency_heal (HP=$hp/$hp_max=$hp_ratio, item=$item_name qty=$item->{amount})\n";
					eval { $item->use(); 1 };
					$heal_triggered = 1;
					last;
				}
			}

			# Try config-pushed skills if no items available
			if (!$heal_triggered) {
				for my $skill_name (@_heal_skills) {
					$skill_name = _trim($skill_name);
					next if !$skill_name;
					my $skill = eval { Skill::get($skill_name) };
					if ($skill && $skill->{level} && $skill->{level} > 0 && $sp > 0) {
						warning "[aiSidecarBridge] bridge_reflex:emergency_heal_skill (HP=$hp/$hp_max, skill=$skill_name lv=$skill->{level}, SP=$sp)\n";
						eval { Commands::run("skill $skill_name 1"); 1 };
						$heal_triggered = 1;
						last;
					}
				}
			}

			# If in town with low HP and no items found, trigger auto-buy
			if (!$heal_triggered && $map =~ /^prontera/i && $hp_ratio < 0.50) {
				warning "[aiSidecarBridge] bridge_reflex:emergency_autobuy (HP=$hp/$hp_max, map=$map)\n";
				eval { Commands::run("autobuy"); 1 };
			}

			# HARD CODED FALLBACK: White Potion (absolute safety net)
			if (!$heal_triggered) {
				my $fallback = eval { Actor::Item::get($HARDCODED_FALLBACK_ITEM) };
				if ($fallback && $fallback->{amount} && $fallback->{amount} > 0) {
					warning "[aiSidecarBridge] bridge_reflex:emergency_fallback_heal (HP=$hp/$hp_max, item=$HARDCODED_FALLBACK_ITEM)\n";
					eval { $fallback->use(); 1 };
					$heal_triggered = 1;
				}
			}

			# If NO healing resources at all: trigger emergency survival immediately
			# Let conscious engine handle the rest (buy potions, retreat to town, request help)
			if (!$heal_triggered && $hp_ratio < 0.50) {
			# Skip if HP has not dropped (reflex noise filter)
			state $_last_hp = 0;
			if ($hp >= $_last_hp && $_last_hp > 0) { next; }
			$_last_hp = $hp;
				if (_should_fire_reflex($_reflex_last_fired{no_heal} || 0, 10000)) {
					$_reflex_last_fired{no_heal} = _now_ms();
					warning "[aiSidecarBridge] bridge_reflex:emergency_no_heal (HP=$hp/$hp_max, map=$map, job=$job_name, lvl=$base_level/$job_level)\n";

					# POST EVENT TO SIDECAR (let conscious handle rest)
					_post_event({
						kind => 'bridge_reflex',
						reflex => 'emergency_no_heal',
						hp_ratio => $hp_ratio,
						hp => $hp,
						max_hp => $hp_max,
						sp_ratio => $sp_ratio,
						sp => $sp,
						aggro_count => $aggro_count,
						weight_ratio => $weight_ratio,
						map => $map,
						job_name => $job_name,
						base_level => $base_level,
						job_level => $job_level,
						heal_items_cached => scalar(@_heal_items),
						heal_skills_cached => scalar(@_heal_skills),
						timestamp => _now_ms(),
					});

					# SIT FALLBACK: if no healing items in inventory and low zeny, sit to regen HP naturally
					my $_has_heal_item = 0;
					for my $_hitem (@_heal_items) {
						my $_inv = eval { Actor::Item::get($_hitem) };
						if ($_inv && ref($_inv) eq 'HASH' && ($_inv->{amount} || 0) > 0) {
							$_has_heal_item = 1;
							last;
						}
					}
					my $_zeny = $char->{zeny} || 0;
					my $ai_state = $AI::AI || '';
					# Debug: log sit condition values
					warning "[aiSidecarBridge] emergency_sit_debug: has_heal=$_has_heal_item zeny=$_zeny hp=$hp max=$hp_max ai=$ai_state\n";
					if (!$_has_heal_item) {
						if ($ai_state ne 'sit' && $hp < $hp_max * 0.5) {
							## SIT REMOVED — reflex should never sit. Let survival_check handle it.
							# warning "[aiSidecarBridge] emergency_sit_regen (HP=$hp/$hp_max, zeny=$char->{zeny})\n";
						}
					} elsif ($hp > $hp_max * 0.8 && ($AI::AI == 2)) {
						# Stand up if HP is healthy and we were sitting
						Commands::run('stand');
						warning "[aiSidecarBridge] emergency_stand (HP=$hp/$hp_max)\n";
					}

					# IMMEDIATE EMERGENCY SURVIVAL: move to town when critically low
					my $_now_ms = _now_ms();
					my $_reflex_map = $char->{map} || '';
					# 60s cooldown on town recovery to let auto-buy/sell complete
					if ($_now_ms - $_last_prontera_recovery_ms > 60000) {
						if ($aggro_count > 0) {
							$_last_prontera_recovery_ms = $_now_ms;
							eval { my $_emap = $char->{map}||""; if ($_emap !~ m{^prontera}i) { $::config{"lockMap"}="prontera"; _toggle_ai_mode('auto'); } 1 };
						} elsif ($hp_ratio < 0.30 && $_reflex_map !~ /^prontera/i) {
							$_last_prontera_recovery_ms = $_now_ms;
							eval { my $_emap = $char->{map}||""; if ($_emap !~ m{^prontera}i) { $::config{"lockMap"}="prontera"; _toggle_ai_mode('auto'); } 1 };
						} elsif ($hp_ratio < 0.15 && $_reflex_map !~ /^prontera/i) {
							$_last_prontera_recovery_ms = $_now_ms;
							eval { my $_emap = $char->{map}||""; if ($_emap !~ m{^prontera}i) { $::config{"lockMap"}="prontera"; _toggle_ai_mode('auto'); } 1 };
						}
					}
				}
			}

			# Cooldown for heal reflex
			if ($heal_triggered) {
				$_reflex_last_fired{heal} = _now_ms();
			}
		}

		# ═══════════════════════════════════════════
		# REFLEX #2 — EMERGENCY FLEE REFLEX
		# ═══════════════════════════════════════════
		# Instant flee when HP <15% and aggroed (no anti-detection delay)
		if ($hp_ratio < 0.15 && $aggro_count > 0) {
			if (_should_fire_reflex($_reflex_last_fired{flee} || 0, 1000)) {
				$_reflex_last_fired{flee} = _now_ms();
				warning "[aiSidecarBridge] bridge_reflex:emergency_flee (HP=$hp/$hp_max, aggro=$aggro_count)\n";
				my $_reflex_map2 = $char->{map} || '';
				if ($_reflex_map2 !~ /^prontera/i) {
					eval { my $_emap = $char->{map}||""; if ($_emap !~ m{^prontera}i) { $::config{"lockMap"}="prontera"; _toggle_ai_mode('auto'); } 1 };
				}
			}
		}

		# ═══════════════════════════════════════════
		# REFLEX #3 — EMERGENCY TELEPORT / SIT REFLEX
		# ═══════════════════════════════════════════
		# Teleport if <12% HP and aggroed, sit+regen otherwise
		if ($hp_ratio < 0.12) {
			if (_should_fire_reflex($_reflex_last_fired{teleport} || 0, 3000)) {
				$_reflex_last_fired{teleport} = _now_ms();
				if ($aggro_count > 0) {
					warning "[aiSidecarBridge] bridge_reflex:emergency_teleport (HP=$hp/$hp_max, aggro=$aggro_count)\n";
					eval { my $_emap = $char->{map}||""; if ($_emap !~ m{^prontera}i) { $::config{"lockMap"}="prontera"; _toggle_ai_mode('auto'); } 1 };
				} else {
					my $_reflex_map3 = $char->{map} || '';
					if ($_reflex_map3 !~ /^prontera/i) {
						warning "[aiSidecarBridge] bridge_reflex:emergency_move_prontera (HP=$hp/$hp_max)\n";
						eval { my $_emap = $char->{map}||""; if ($_emap !~ m{^prontera}i) { $::config{"lockMap"}="prontera"; _toggle_ai_mode('auto'); } 1 };
					}
				}
			}
		}

		# ═══════════════════════════════════════════
		# REFLEX #4 — AGGRO WARNING REFLEX
		# ═══════════════════════════════════════════
		# Notify sidecar when heavily aggroed (>5 attackers)
		if ($aggro_count > 5) {
			if (_should_fire_reflex($_reflex_last_fired{aggro_warning} || 0, 5000)) {
				$_reflex_last_fired{aggro_warning} = _now_ms();
				warning "[aiSidecarBridge] bridge_reflex:aggro_warning (aggro=$aggro_count)\n";
				_post_event({
					kind => 'bridge_reflex',
					reflex => 'aggro_warning',
					aggro_count => $aggro_count,
					hp_ratio => $hp_ratio,
					map => $map,
					timestamp => $now,
				});
			}
		}

		# ═══════════════════════════════════════════
		# REFLEX #5 — LOW SP REFLEX
		# ═══════════════════════════════════════════
		# Notify sidecar when SP is critically low
		if ($sp_ratio < 0.15) {
			if (_should_fire_reflex($_reflex_last_fired{low_sp} || 0, 10000)) {
				$_reflex_last_fired{low_sp} = _now_ms();
				warning "[aiSidecarBridge] bridge_reflex:low_sp (SP=$sp/$sp_max, ratio=$sp_ratio)\n";
				_post_event({
					kind => 'bridge_reflex',
					reflex => 'low_sp',
					sp_ratio => $sp_ratio,
					sp => $sp,
					max_sp => $sp_max,
					timestamp => $now,
				});
			}
		}

		# ═══════════════════════════════════════════
		# REFLEX #6 — GM / ADMIN DETECTION REFLEX
		# ═══════════════════════════════════════════
		# Detect GM/Admin players within 15 tiles, switch to manual
		if ($playersList) {
			my $gm_detected = 0;
			for my $player (@{$playersList}) {
				next if !$player;
				my $pname = $player->{name} || '';
				next if $pname eq '';
				if ($pname =~ /GM|GameMaster|Admin|Support/i) {
					my $dist = _calc_distance($player, $char);
					if (defined $dist && $dist <= 15) {
						$gm_detected = 1;
						last;
					}
				}
			}
			if ($gm_detected) {
				if (_should_fire_reflex($_reflex_last_fired{gm_detected} || 0, 60000)) {
					$_reflex_last_fired{gm_detected} = _now_ms();
					warning "[aiSidecarBridge] bridge_reflex:gm_detected (GM/Admin player within 15 tiles)\n";
					eval { _toggle_ai_mode('manual'); 1 };
					_post_event({
						kind => 'bridge_reflex',
						reflex => 'gm_detected',
						message => 'GM/Admin player detected within 15 tiles, AI switched to manual',
						timestamp => $now,
					});
				}
			}
		}

		# ═══════════════════════════════════════════
		# REFLEX #7 — WEIGHT WARNING REFLEX
		# ═══════════════════════════════════════════
		# Notify sidecar when weight exceeds 85%
		if ($weight_ratio > 0.85) {
			if (_should_fire_reflex($_reflex_last_fired{weight_warning} || 0, 30000)) {
				$_reflex_last_fired{weight_warning} = _now_ms();
				warning "[aiSidecarBridge] bridge_reflex:weight_warning (weight=$weight/$weight_max, ratio=$weight_ratio)\n";
				_post_event({
					kind => 'bridge_reflex',
					reflex => 'weight_warning',
					weight_ratio => $weight_ratio,
					weight => $weight,
					max_weight => $weight_max,
					timestamp => $now,
				});
			}
		}

		# ═══════════════════════════════════════════
		# REFLEX #8 — EQUIPMENT BROKEN REFLEX
		# ═══════════════════════════════════════════
		# Notify sidecar when equipped item is damaged/broken
		if ($char->{equipment}) {
			my $broken_found = 0;
			for my $slot (keys %{$char->{equipment}}) {
				my $item = $char->{equipment}{$slot};
				next if !$item;
				if ($item->{broken} || (defined $item->{damage} && $item->{damage} > 0)) {
					$broken_found = 1;
					last;
				}
			}
			if ($broken_found) {
				if (_should_fire_reflex($_reflex_last_fired{equipment_broken} || 0, 60000)) {
					$_reflex_last_fired{equipment_broken} = _now_ms();
					warning "[aiSidecarBridge] bridge_reflex:equipment_broken (broken equipment detected)\n";
					_post_event({
						kind => 'bridge_reflex',
						reflex => 'equipment_broken',
						message => 'Broken equipment detected',
						timestamp => $now,
					});
				}
			}
		}

		# ═══════════════════════════════════════════
		# REFLEX #9 — INTERRUPT CAST REFLEX
		# ═══════════════════════════════════════════
		# Interrupt enemy casters within 10 tiles using Bash or similar
		if ($monstersList) {
			my $interrupted = 0;
			for my $monster (@{$monstersList}) {
				next if !$monster;
				my $casting = $monster->{casting} || undef;
				next if !$casting;
				my $dist = _calc_distance($monster, $char);
				if (defined $dist && $dist <= 10) {
					$interrupted = 1;
					last;
				}
			}
			if ($interrupted) {
				if (_should_fire_reflex($_reflex_last_fired{interrupt_cast} || 0, 1500)) {
					$_reflex_last_fired{interrupt_cast} = _now_ms();
					warning "[aiSidecarBridge] bridge_reflex:interrupt_cast (monster casting within 10 tiles)\n";
					eval { Commands::run("skill Bash 10"); 1 };
				}
			}
		}

		# ═══════════════════════════════════════════
		# REFLEX #10 — PRE-POT REFLEX (BOSS WITHIN 15 TILES)
		# ═══════════════════════════════════════════
		# Preemptively heal before engaging a boss
		my @BOSS_IDS = (1038, 1046, 1049, 1059, 1086, 1087, 1088, 1112, 1115, 1147, 1150, 1159, 1205, 1272, 1312, 1313, 1511, 1630, 1639, 1719, 1751, 1871, 1874);
		my %BOSS_LOOKUP = map { $_ => 1 } @BOSS_IDS;
		if ($monstersList) {
			my $boss_nearby = 0;
			for my $monster (@{$monstersList}) {
				next if !$monster;
				my $name_id = $monster->{nameID} || 0;
				next if !$name_id;
				if ($BOSS_LOOKUP{$name_id}) {
					my $dist = _calc_distance($monster, $char);
					if (defined $dist && $dist <= 15) {
						$boss_nearby = 1;
						last;
					}
				}
			}
			if ($boss_nearby && $hp_ratio > 0.9) {
				if (_should_fire_reflex($_reflex_last_fired{pre_pot} || 0, 5000)) {
					$_reflex_last_fired{pre_pot} = _now_ms();
					_update_heal_cache();
					my $healed = 0;
					for my $item_name (@_heal_items) {
						my $item = eval { Actor::Item::get($item_name) };
						if ($item) {
							warning "[aiSidecarBridge] bridge_reflex:pre_pot (boss within 15 tiles, HP=$hp/$hp_max, item=$item_name)\n";
							eval { Commands::run("is $item_name"); 1 };
							$healed = 1;
							last;
						}
					}
					if (!$healed) {
						warning "[aiSidecarBridge] bridge_reflex:pre_pot_no_items (boss within 15 tiles, HP=$hp/$hp_max)\n";
					}
				}
			}
		}

		# ═══════════════════════════════════════════
		# REFLEX #11 — BOT-TO-BOT COOPERATION REQUEST
		# ═══════════════════════════════════════════
		# If HP critically low with no heal resources and aggro, request help from other bots
		if (!$heal_triggered && $hp_ratio < 0.50 && $aggro_count > 0) {
			if (_should_fire_reflex($_reflex_last_fired{bot_request} || 0, 5000)) {
				$_reflex_last_fired{bot_request} = _now_ms();
				warning "[aiSidecarBridge] bridge_reflex:bot_cooperation_request (HP=$hp/$hp_max, aggro=$aggro_count)\n";
				_post_event({
					kind => 'bridge_reflex',
					reflex => 'bot_cooperation_request',
					hp_ratio => $hp_ratio,
					hp => $hp,
					max_hp => $hp_max,
					aggro_count => $aggro_count,
					map => $map,
					base_level => $base_level,
					job_name => $job_name,
					timestamp => _now_ms(),
				});
			}
		}

		# ═══════════════════════════════════════════
		# REFLEX #12 — PARTY MEMBER LOW HP ALERT
		# ═══════════════════════════════════════════
		# Notify sidecar if any party member has critically low HP
		if ($playersList) {
			for my $player (@{$playersList}) {
				next if !$player;
				my $pname = $player->{name} || '';
				next if $pname eq '';
				# Skip self
				next if $char && defined $char->{name} && $pname eq $char->{name};

				my $player_hp = $player->{hp} || 0;
				my $player_hp_max = $player->{hp_max} || 1;
    # Skip ghost entries: real players never have max_hp < 100
    next if $player_hp_max < 100;
				my $player_hp_ratio = ($player_hp_max > 0) ? $player_hp / $player_hp_max : 1;

				if ($player_hp_ratio < 0.20) {
					my $dist = _calc_distance($player, $char);
					next if !defined $dist || $dist > 20;

					if (_should_fire_reflex($_reflex_last_fired{party_low_hp} || 0, 10000)) {
						$_reflex_last_fired{party_low_hp} = _now_ms();
						warning "[aiSidecarBridge] bridge_reflex:party_low_hp (player=$pname HP=$player_hp/$player_hp_max=$player_hp_ratio, dist=$dist)\n";
						_post_event({
							kind => 'bridge_reflex',
							reflex => 'party_low_hp',
							player_name => $pname,
							player_hp => $player_hp,
							player_hp_max => $player_hp_max,
							player_hp_ratio => $player_hp_ratio,
							distance => $dist,
							timestamp => $now,
						});
					}
					last;  # Only report first low-HP party member per cycle
				}
			}
		}

		# ═══════════════════════════════════════════
		# REFLEX #13 — HIGH AGGRO SURROUND REFLEX
		# ═══════════════════════════════════════════
		# Immediate emergency when surrounded (>10 aggro)
		if ($aggro_count > 10) {
			if (_should_fire_reflex($_reflex_last_fired{high_aggro_surround} || 0, 3000)) {
				$_reflex_last_fired{high_aggro_surround} = _now_ms();
				warning "[aiSidecarBridge] bridge_reflex:high_aggro_surround (aggro=$aggro_count)\n";

				# Immediate flee + teleport combo
				eval { my $_emap = $char->{map}||""; if ($_emap !~ m{^prontera}i) { $::config{"lockMap"}="prontera"; _toggle_ai_mode('auto'); } 1 };
				if ($hp_ratio < 0.25) {
					eval { Commands::run("tele"); 1 };
				}

				_post_event({
					kind => 'bridge_reflex',
					reflex => 'high_aggro_surround',
					aggro_count => $aggro_count,
					hp_ratio => $hp_ratio,
					map => $map,
					timestamp => $now,
				});
			}
		}

		# ═══════════════════════════════════════════
		# REFLEX #14 — ZONK / DEAD REFLEX (INSTANT)
		# ═══════════════════════════════════════════
		# Immediate sit if HP is zero or critically zero (zombie state)
		if ($hp <= 0 || ($hp > 0 && $hp <= 5)) {
			if (_should_fire_reflex($_reflex_last_fired{zonk} || 0, 2000)) {
				$_reflex_last_fired{zonk} = _now_ms();
				warning "[aiSidecarBridge] bridge_reflex:zonk (HP=$hp/$hp_max, map=$map)\n";
				eval { my $_emap = $char->{map}||""; if ($_emap !~ m{^prontera}i) { $::config{"lockMap"}="prontera"; _toggle_ai_mode('auto'); } 1 };
				_post_event({
					kind => 'bridge_reflex',
					reflex => 'zonk',
					hp => $hp,
					hp_max => $hp_max,
					map => $map,
					timestamp => $now,
				});
			}
		}

		# ═══════════════════════════════════════════
		# REFLEX #15 — DEATH COUNTER (INFORM SIDECAR)
		# ═══════════════════════════════════════════
		# Track death count and notify sidecar if deaths spike
		if ($death_count > 0 && $death_count % 5 == 0) {
			if (_should_fire_reflex($_reflex_last_fired{death_spike} || 0, 120000)) {
				$_reflex_last_fired{death_spike} = _now_ms();
				warning "[aiSidecarBridge] bridge_reflex:death_spike (deaths=$death_count, map=$map)\n";
				_post_event({
					kind => 'bridge_reflex',
					reflex => 'death_spike',
					death_count => $death_count,
					map => $map,
					timestamp => $now,
				});
			}
		}

		# ═══════════════════════════════════════════
		# REFLEX #16 — PROACTIVE PRE-BUFF REFLEX
		# ═══════════════════════════════════════════
		# Pre-buff before engaging: cast self-buffs when out of combat
		# Pro players always buff before engaging, never during combat
		if (!$in_combat && $hp_ratio > 0.8 && $sp_ratio > 0.3) {
			if (_should_fire_reflex($_reflex_last_fired{pre_buff} || 0, 15000)) {
				$_reflex_last_fired{pre_buff} = _now_ms();
				# Try common self-buffs based on class
				my @buffs = (
					"skill Twohand Quicken 1",   # Knight ASPD buff
					"skill Increase AGI 10",     # Acolyte/Priest
					"skill Blessing 10",          # Acolyte/Priest
					"skill Magnificat 5",         # Priest SP regen
					"skill Kyrie Eleison 10",     # Priest shield
					"skill Improve Concentration 10", # Swordsman
					"skill Enchant Poison 5",     # Assassin
					"skill Owl's Eye 10",         # Archer
					"skill Vulture's Eye 10",     # Archer
					"skill Energy Coat 5",        # Mage
				);
				for my $buff (@buffs) {
					my ($cmd, $skill_name) = $buff =~ /^skill\s+(.+?)\s+\d+$/;
					next if !$skill_name;
					my $skill = eval { Skill::get($skill_name) };
					if ($skill && $skill->{level} && $skill->{level} > 0) {
						# Check if buff is already active
						my $already_active = 0;
						if ($char->{buffs} && ref $char->{buffs} eq 'HASH') {
							$already_active = 1 if exists $char->{buffs}{$skill_name};
						}
						if (!$already_active) {
							_random_action_delay();
							eval { Commands::run($buff); 1 };
							last;  # One buff per tick
						}
					}
				}
			}
		}

		# ═══════════════════════════════════════════
		# REFLEX #17 — PROACTIVE PRE-DODGE REFLEX
		# ═══════════════════════════════════════════
		# Pre-dodge when a monster starts casting a dangerous AoE
		# Pro players don't wait for the hit — they dodge the cast bar
		if ($monstersList) {
			for my $monster (@{$monstersList}) {
				next if !$monster;
				my $casting = $monster->{casting} || undef;
				next if !$casting;
				my $dist = _calc_distance($monster, $char);
				next if !defined $dist || $dist > 12;

				# Known dangerous AoE skills to dodge
				my @DODGE_SKILLS = qw(
					WZ_STORMGUST WZ_METEORSTORM WZ_HEAVENDRIVE MG_THUNDERSTORM
					WZ_VERMILION NPC_HELLJUDGEMENT NPC_EARTHQUAKE NPC_DARKBREATH
					NPC_PULSESTRIKE NPC_WIDESTUN NPC_WIDEFREEZE NPC_FIREBREATH
				);
				my $should_dodge = 0;
				for my $dodge_skill (@DODGE_SKILLS) {
					if ($casting eq $dodge_skill) {
						$should_dodge = 1;
						last;
					}
				}

				if ($should_dodge) {
					if (_should_fire_reflex($_reflex_last_fired{pre_dodge} || 0, 2000)) {
						$_reflex_last_fired{pre_dodge} = _now_ms();
						warning "[aiSidecarBridge] bridge_reflex:pre_dodge (monster casting $casting at dist=$dist)\n";
						# Move away immediately — no delay
						eval { my $_emap = $char->{map}||""; if ($_emap !~ m{^prontera}i) { $::config{"lockMap"}="prontera"; _toggle_ai_mode('auto'); } 1 };
						_post_event({
							kind => 'bridge_reflex',
							reflex => 'pre_dodge',
							casting_skill => $casting,
							distance => $dist,
							hp_ratio => $hp_ratio,
							timestamp => $now,
						});
					}
					last;
				}
			}
		}

		# ═══════════════════════════════════════════
		# REFLEX #18 — AUTO-SIT REGEN REFLEX
		# ═══════════════════════════════════════════
		# Auto-sit when out of combat and HP/SP is low
		# Pro players sit to regen between pulls, never fight at low HP
		if (!$in_combat && $hp_ratio < 0.6 && $hp > 0) {
			if (_should_fire_reflex($_reflex_last_fired{auto_sit} || 0, 5000)) {
				$_reflex_last_fired{auto_sit} = _now_ms();
				my $ai_top = @ai_seq ? $ai_seq[0] : '';
				if ($ai_top ne 'sit') {
					_random_action_delay();
					eval { my $_emap = $char->{map}||""; if ($_emap !~ m{^prontera}i) { $::config{"lockMap"}="prontera"; _toggle_ai_mode('auto'); } 1 };
				}
			}
		}

		# ═══════════════════════════════════════════
		# REFLEX #19 — PROACTIVE POTION TOP-OFF REFLEX
		# ═══════════════════════════════════════════
		# Top off HP when out of combat and HP < 80%
		# Pro players keep HP topped off between pulls, not just in emergencies
		if (!$in_combat && $hp_ratio > 0.3 && $hp_ratio < 0.8 && $hp > 0) {
			if (_should_fire_reflex($_reflex_last_fired{top_off} || 0, 10000)) {
				$_reflex_last_fired{top_off} = _now_ms();
				_update_heal_cache();
				for my $item_name (@_heal_items) {
					$item_name = _trim($item_name);
					next if !$item_name;
					my $item = eval { Actor::Item::get($item_name) };
					if ($item && $item->{amount} && $item->{amount} > 0) {
						_random_action_delay();
						eval { Commands::run("is $item_name"); 1 };
						last;
					}
				}
			}
		}
	}
}
sub _apply_bot_config {
	# Apply optimal OpenKore configs for autonomous operation
	# (Overrides user config with AI-optimized values)
	
	# Enable continuous auto-attack on monsters
	if (defined $::config) {
	    $::config{'attackAuto'} = '2';
	    $::config{'attackAuto_inLockOnly'} = '1';
	    $::config{'attackAuto_followTarget'} = '0';
	    $::config{'attackAuto_onlyWhenSafe'} = '0';
	    $::config{'attackAuto_noMove'} = '0';
	    $::config{'sitAuto_hp_lower'} = '0';  # Disable sit auto (bridge handles it)
	    $::config{'sitAuto_hp_upper'} = '0';
	    $::config{'sitAuto_maxDmg'} = '99999';
	    
	    # Auto-pickup everything (we sell junk)
	    $::config{'itemsTakeAuto'} = '2';
	    $::config{'itemsTakeAuto_party'} = '1';
	    $::config{'itemsGatherAuto'} = '2';
	    
	    # Auto-sell junk items (Jellopy, Stems, etc.)
	    $::config{'sellAuto'} = '1';
	    $::config{'sellAuto_npc'} = 'prontera 179 187';  # General Store
	    $::config{'sellAuto_distance'} = '25';
	    
	    # Storage (deposit junk, withdraw potions)
	    $::config{'storageAuto'} = '1';
	    $::config{'storageAuto_npc'} = 'prontera 151 29';  # Kafra
	    $::config{'storageAuto_distance'} = '5';
	    $::config{'storageAuto_npc_type'} = '1';
	    $::config{'storageAuto_npc_steps'} = 'c r1 c';
	    $::config{'relogAfterStorage'} = '0';
	    $::config{'minStorageZeny'} = '0';
	    
	    # Party auto-join
	    $::config{'partyAuto'} = '1';
	    $::config{'partyAutoShare'} = '1';
	}
}

# ── Safe character accessor ──
sub _safe_char {
	return $main::char || $char;
}

# ── Survival / Progression monitor (last-resort fallback) ──
# Most execution is handled by OpenKore's built-in systems (attackAuto,
# buyAuto, sellAuto, items_control). This function only fires when the
# Pro RO LLM's goals detect no progression for >5 minutes, or when
# HP is critically low (< 20%) as an emergency override.
sub _survival_check {
	my $now = _now_ms();
	state $_last_sc_ms = 0;
	return if $now - $_last_sc_ms < 60000;  # Check every 1 MINUTE (was 2s)
	$_last_sc_ms = $now;

	my $cr = _safe_char();
	return if !$cr;
	my $hp = $cr->{hp} || 0;
	my $hp_max = $cr->{hp_max} || 1;
	my $zeny = $cr->{zeny} || 0;
	my $base_lv = $cr->{lv} || $cr->{level} || 1;
	my $hp_pct = $hp_max > 0 ? int($hp * 100 / $hp_max) : 0;
	my $map = $cr->{map} || '';

# ── Emergency: HP critically low — call Pro RO LLM for healing strategy ──
	if ($hp_pct < 20 && $hp_pct > 0) {
	    my $_strat = _http_post_json('/v1/discover/heal', {
	        bot_id => _bot_id(),
	        hp => $hp,
	        hp_max => $hp_max,
	        zeny => $zeny,
	        map => $map,
	        inventory => _safe_inventory_list(),
	        known_shops => _discover_shops_sync(),
	        known_portals => _discover_portals_sync(),
	    });
	    
	    if ($_strat && $_strat->{status} == 200 && $_strat->{json}{command}) {
	        my $_cmd = $_strat->{json}{command};
	        eval { Commands::run($_cmd); 1 };
	        my $_tgt_map = $_strat->{json}{target_map} || '';
	        if ($_tgt_map ne '' && $_tgt_map ne $map) {
	            $::config{'lockMap'} = $_tgt_map;
	            if ($AI::AI != 2) { eval { require AI; AI::state(2); 1 }; }
	        }
	    } else {
	        # Fallback: stand up, try buying, sit as last resort
	        if (0) { eval { Commands::run('stand'); 1 }; }
	        $::config{'lockMap'} = 'prt_in';
	        if ($AI::AI != 2) { eval { require AI; AI::state(2); 1 }; }
	        eval { Commands::run('buy 501 30'); 1 };
	        eval { Commands::run('use Red Potion'); 1 };
	        if ($hp_pct < 15) { eval { Commands::run('sit'); 1 }; }
	    }
	    return;
	}
	# ── Economy: Not enough zeny and not in combat → go grind ──
	if ($zeny < 100 && $base_lv < 99) {
	    my $clm = defined $::config{'lockMap'} ? $::config{'lockMap'} : '';
	    if ($clm eq '' || $clm eq 'prt_in' || $clm eq 'prontera') {
	        $::config{'lockMap'} = 'prt_fild08';
	    }
	    if ($AI::AI != 2) {
	        eval { require AI; AI::state(2); 1 };
	    }
	    return;
	}

	# ── Progression: Apply optimal configs if not already set ──
	_apply_bot_config();

	# ── Auto-mode: Enable if disabled and conditions safe ──
	if ($hp_pct > 50 && $AI::AI != 2) {
	    eval { require AI; AI::state(2); 1 };
	}
}

# ── Economy / Sell loop (auto-sell junk items for zeny) ──
# This is called by the survival_check when zeny is needed.
sub _sell_junk_items {
	my $cr = _safe_char();
	return if !$cr || !$cr->{inventory};
	
	# Items worth less than 100z that should be sold automatically
	my %sell_items = (
	    'Jellopy' => 1, 'Stem' => 1, 'Empty Bottle' => 1,
	    'Red Herb' => 1, 'Yellow Herb' => 1, 'White Herb' => 1,
	    'Feather' => 1, 'Cactus Needle' => 1, 'Shell' => 1,
	);
	
	foreach my $item (@{$cr->{inventory}}) {
	    next unless ref($item) eq 'HASH';
	    my $iname = $item->{name} || '';
	    my $iamount = $item->{amount} || 0;
	    next if $iname eq '' || $iamount < 1;
	    next unless $sell_items{$iname};
	    eval { Commands::run("sell $iname"); 1 };
	}
}

sub _survival_check {
	my $now = _now_ms();
	state $_last_sc_ms = 0;
	return if $now - $_last_sc_ms < 60000;
	$_last_sc_ms = $now;

	my $cr = _safe_char();
	return if !$cr;
	my $hp = $cr->{hp} || 0;
	my $hp_max = $cr->{hp_max} || 1;
	my $zeny = $cr->{zeny} || 0;
	my $base_lv = $cr->{lv} || $cr->{level} || 1;
	my $ai_mode = $AI::AI || '';
	my $hp_pct = $hp_max > 0 ? int($hp * 100 / $hp_max) : 0;
	my $map = $cr->{map} || '';

	_apply_bot_config();

	# ── Emergency (HP < 20%): Stand up, navigate, buy, use — NEVER sit ──
	if ($hp_pct < 20 && $hp_pct > 0) {
	    my $_strat = _http_post_json('/v1/discover/heal', {
	        bot_id => _bot_id(),
	        hp => $hp, hp_max => $hp_max, zeny => $zeny, map => $map,
	        inventory => _safe_inventory_list(),
	        known_shops => _discover_shops_sync(),
	        known_portals => _discover_portals_sync(),
	    });
	    if ($_strat && $_strat->{status} == 200 && $_strat->{json}{command}) {
	        eval { Commands::run($_strat->{json}{command}); 1 };
	        my $_tgt = $_strat->{json}{target_map} || '';
	        if ($_tgt ne '' && $_tgt ne $map) { $::config{'lockMap'} = $_tgt; }
	    }
	    # Fallback: stand, navigate to healer/prt_in, buy, use
	    if ($ai_mode eq 'sit') { eval { Commands::run('stand'); 1 }; }
	    $::config{'lockMap'} = 'prt_in';
	    if ($ai_mode !~ /auto/i) { eval { require AI; AI::state(2); 1 }; }
	    eval { Commands::run('buy 501 30'); 1 };
	    eval { Commands::run('use Red Potion'); 1 };
	    # Fall through to economy check and auto mode
	}

	# ── Economy: If zeny low, go grind ──
	if ($zeny < 300 && $base_lv < 99) {
	    $::config{'lockMap'} = 'prt_fild08';
	    if ($ai_mode !~ /auto/i) { eval { require AI; AI::state(2); 1 }; }
	}

	# ── Stay in auto mode when HP is safe ──
	if ($hp_pct > 40 && $ai_mode !~ /auto/i) {
	    eval { require AI; AI::state(2); 1 };
	}
}

# ── Team Play: Auto-party and coordinate sibling bots ──

# ── Team Play: Auto-party with sibling bots ──
# Sub_teamplay_check runs every 30s. Creates/joins party, invites siblings,
# and sends non-sibling candidates to Pro RO LLM for evaluation.
sub _teamplay_check {
	my $now = _now_ms();
	state $_last_tp_ms = 0;
	return if $now - $_last_tp_ms < 30000;
	$_last_tp_ms = $now;

	my $cr = _safe_char();
	return if !$cr;
	my $name = $cr->{name} || '';
	return if $name eq '';

	my $party = $main::partiesList;
	my $in_party = (defined $party && ref($party) eq 'HASH' && scalar(keys %$party) > 0) ? 1 : 0;

	if (!$in_party) {
	    eval { Commands::run('party create "AI Team"'); 1 };
	    eval { Commands::run('party accept'); 1 };
	}

	if ($in_party) {
	    # Auto-invite known sibling bots
	    my @_siblings = ('openkoreai', 'openkoreaiobs', 'openkoreaihuman');
	    foreach my $_sib (@_siblings) {
	        next if $_sib eq $name;
	        my $_already = 0;
	        if (ref($party) eq 'HASH') {
	            foreach my $_key (keys %$party) {
	                my $_pm = ref($party->{$_key}) eq 'HASH' ? ($party->{$_key}{name}||$party->{$_key}{actor}||$party->{$_key}) : $party->{$_key};
	                $_pm = ref($_pm) eq 'HASH' ? ($_pm->{name}||'') : $_pm;
	                if ($_pm eq $_sib) { $_already = 1; last; }
	            }
	        }
	        if (!$_already) {
	            eval { Commands::run("party invite $_sib"); 1 };
	        }
	    }

	    # Pro RO LLM: evaluate non-sibling nearby players for party
	    if ($main::playersList) {
	        foreach my $_pl (@{$main::playersList}) {
	            next unless ref($_pl) eq 'HASH';
	            my $_pn = $_pl->{name} || '';
	            next if $_pn eq '' || $_pn eq $name;
	            my $_is_sib = 0;
	            foreach ('openkoreai', 'openkoreaiobs', 'openkoreaihuman') { if ($_pn eq $_) { $_is_sib = 1; last; } }
	            next if $_is_sib;
	            my $_already = 0;
	            if (ref($party) eq 'HASH') {
	                foreach my $_k (keys %$party) {
	                    my $_pm = ref($party->{$_k}) eq 'HASH' ? ($party->{$_k}{name}||'') : $party->{$_k};
	                    if (ref($_pm) eq 'HASH') { $_pm = $_pm->{name}||''; }
	                    if ($_pm eq $_pn) { $_already = 1; last; }
	                }
	            }
	            next if $_already;
	            _post_event({
	                kind => 'teamplay_candidate',
	                player => $_pn,
	                job => ($_pl->{job_name} || $_pl->{job} || ''),
	                level => ($_pl->{lv} || $_pl->{level} || 0),
	            });
	            last;
	        }
	    }
	}
}



# ── Hook: Auto-accept party invites from console messages ──
sub _check_party_invite {
	my $msg = shift || '';
	if ($msg =~ /invite.*to.*party/i || $msg =~ /party.*invitation/i || $msg =~ /joined.*party/i) {
	    eval { Commands::run('party accept'); 1 };
	    eval { Commands::run('party join 1'); 1 };
	}
}

# ── Wire team play into the main loop ──

# ── Server Discovery: Scan NPC shops for healing/progression items ──
sub _discover_shops {
	my $now = _now_ms();
	state $_last_ss = 0;
	return if $now - $_last_ss < 3600000;
	$_last_ss = $now;
	# Stub — will be populated with actual NPC data from observed shops
	_post_event({ kind => 'discovery_shops', shops => [] });
}

sub _discover_portals {
	my $now = _now_ms();
	state $_last_ps = 0;
	return if $now - $_last_ps < 3600000;
	$_last_ps = $now;
	my %conn;
	my $pf = $::Settings->{tablesPath} . '/portals.txt';
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
	return \@shops;
}

sub _discover_portals_sync {
	my %conn;
	my $pf = $::Settings->{tablesPath} . '/portals.txt';
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
	return [] if !$cr || !$cr->{inventory};
	my @items;
	foreach my $_i (@{$cr->{inventory}}) {
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
	_http_post_json('/v1/discover/tables/ingest', {
	    kind => 'discovery_all_tables',
	    tables => $data,
	    timestamp => _now_ms(),
	});
}

1;
