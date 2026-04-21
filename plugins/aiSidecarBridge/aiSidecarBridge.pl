package aiSidecarBridge;

use strict;
use warnings;

use Commands;
use FileParsers qw(parseConfigFile);
use Globals qw(%config $char $field @ai_seq $net);
use IO::Socket::INET;
use Log qw(debug message warning);
use Network;
use Plugins;
use Settings;
use Time::HiRes qw(alarm time);

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
	}

	_track_ai_sequence_transition();
}

sub on_mainLoop_post {
	return unless _bridge_enabled();
	my $now = _now_ms();

	if (!$registered && $now >= $next_register_at_ms) {
		$next_register_at_ms = $now + _cfg_int('aiSidecar_registerRetryMs', 5000);
		_attempt_register('retry');
	}

	if (_cfg_bool('aiSidecar_actionPollEnabled', 1) && $now >= $next_poll_at_ms) {
		$next_poll_at_ms = $now + _cfg_int('aiSidecar_pollIntervalMs', 250);
		_poll_next_action();
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
		$next_event_ingest_at_ms = $now + _cfg_int('aiSidecar_eventIngestIntervalMs', 500);
		_flush_event_queue();
	}
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

sub _load_bridge_config {
	my ($file, $target) = @_;
	%{$target} = ();
	parseConfigFile($file, $target, 0);

	my %defaults = (
		aiSidecar_enable => 1,
		aiSidecar_baseUrl => 'http://127.0.0.1:18081',
		aiSidecar_contractVersion => 'v1',
		aiSidecar_source => 'openkore-bridge',
		aiSidecar_connectTimeoutMs => 40,
		aiSidecar_ioTimeoutMs => 90,
		aiSidecar_snapshotEnabled => 1,
		aiSidecar_snapshotIntervalMs => 500,
		aiSidecar_pollWhenDisconnected => 0,
		aiSidecar_actionPollEnabled => 1,
		aiSidecar_pollIntervalMs => 250,
		aiSidecar_ackEnabled => 1,
		aiSidecar_ackRetryMs => 500,
		aiSidecar_ackMaxAgeMs => 5000,
		aiSidecar_registerRetryMs => 5000,
		aiSidecar_telemetryEnabled => 1,
		aiSidecar_telemetryIntervalMs => 1000,
		aiSidecar_maxRawChars => 256,
		aiSidecar_maxCommandLength => 160,
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
		aiSidecar_eventIngestIntervalMs => 500,
		aiSidecar_chatIngestIntervalMs => 700,
		aiSidecar_configIngestIntervalMs => 2000,
		aiSidecar_eventBatchSize => 20,
		aiSidecar_chatBatchSize => 20,
		aiSidecar_maxEventQueue => 300,
		aiSidecar_maxChatQueue => 200,
		aiSidecar_maxEventPayloadFields => 16,
		aiSidecar_maxEventTextChars => 220,
		aiSidecar_maxChatChars => 500,
		aiSidecar_maxConfigValueChars => 220,
		aiSidecar_maxConfigKeysPerPush => 64,
		aiSidecar_traceAllCommands => 0,
		aiSidecar_verbose => 1,
	);

	foreach my $key (keys %defaults) {
		$target->{$key} = $defaults{$key} if !defined $target->{$key} || $target->{$key} eq '';
	}

	debug "[aiSidecarBridge] loaded control file $file\n", 'aiSidecarBridge', 2;
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
	if (!$net || $net->getState() != Network::IN_GAME) {
		return if !_cfg_bool('aiSidecar_pollWhenDisconnected', 0);
	}

	my $snapshot = _build_snapshot_payload();
	my $resp = _http_post_json('/v1/ingest/snapshot', $snapshot);
	if (!$resp || $resp->{status} < 200 || $resp->{status} >= 300) {
		_throttled_warning('snapshot_failed', '[aiSidecarBridge] snapshot push failed, fail-open retained.');
		_emit_telemetry('warning', 'bridge', 'snapshot_failed', 'snapshot push failed');
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
		char_name => _trim($char ? ($char->{name} || '') : '', $max_raw),
		master => _trim($config{master} || '', $max_raw),
		ai_sequence => _trim($ai_top || '', $max_raw),
		ai_queue => _trim(join(',', @ai_seq[0 .. ($#ai_seq < 4 ? $#ai_seq : 4)]), $max_raw),
	};

	return {
		meta => _meta($bot_id),
		tick_id => _trace_id(),
		observed_at => _iso_now(),
		position => {
			map => $map || undef,
			x => $x,
			y => $y,
		},
		vitals => {
			hp => $char ? $char->{hp} : undef,
			hp_max => $char ? $char->{hp_max} : undef,
			sp => $char ? $char->{sp} : undef,
			sp_max => $char ? $char->{sp_max} : undef,
			weight => $char ? $char->{weight} : undef,
			weight_max => $char ? $char->{weight_max} : undef,
		},
		combat => {
			ai_sequence => $ai_top || undef,
			target_id => undef,
			is_in_combat => $in_combat,
		},
		inventory => {
			zeny => $char ? $char->{zeny} : undef,
			item_count => $item_count,
		},
		raw => $raw,
	};
}

sub _poll_next_action {
	return if !_bridge_enabled();
	if (!$net || $net->getState() != Network::IN_GAME) {
		return if !_cfg_bool('aiSidecar_pollWhenDisconnected', 0);
	}

	my $poll_id = _trace_id();
	my $resp = _http_post_json('/v1/actions/next', {
		meta => _meta(_bot_id()),
		poll_id => $poll_id,
	});

	if (!$resp || $resp->{status} < 200 || $resp->{status} >= 300) {
		$registered = 0;
		_throttled_warning('poll_failed', '[aiSidecarBridge] action poll failed, fail-open retained.');
		_emit_telemetry('warning', 'bridge', 'poll_failed', 'action poll failed');
		return;
	}

	my $json = $resp->{json};
	return if ref($json) ne 'HASH';
	return if !$json->{has_action};
	return if ref($json->{action}) ne 'HASH';

	_execute_action($poll_id, $json->{action});
}

sub _execute_action {
	my ($poll_id, $action) = @_;

	my $action_id = $action->{action_id} || 'unknown_action';
	my $kind = lc($action->{kind} || 'command');
	my $command = defined $action->{command} ? $action->{command} : '';
	my $metadata = ref($action->{metadata}) eq 'HASH' ? $action->{metadata} : {};
	my $started = _now_ms();

	my ($success, $result_code, $msg) = (0, 'invalid_action', 'invalid action payload');

	if ($kind eq 'macro_reload') {
		($success, $result_code, $msg) = _execute_macro_reload_action($metadata);
	} elsif ($kind ne 'command') {
		($success, $result_code, $msg) = (0, 'unsupported_kind', "unsupported action kind '$kind'");
	} elsif ($command eq '') {
		($success, $result_code, $msg) = (0, 'empty_command', 'empty command');
	} elsif (length($command) > _cfg_int('aiSidecar_maxCommandLength', 160)) {
		($success, $result_code, $msg) = (0, 'command_too_long', 'command length exceeds policy');
	} elsif (!_command_allowed($command)) {
		($success, $result_code, $msg) = (0, 'policy_rejected', 'command rejected by bridge policy');
	} else {
		my $ok = eval { Commands::run($command); 1; };
		if ($ok) {
			($success, $result_code, $msg) = (1, 'ok', 'command dispatched through OpenKore console pathway');
		} else {
			my $err = $@ || 'command execution failure';
			($success, $result_code, $msg) = (0, 'dispatch_error', _trim($err, 220));
		}
	}

	my $latency_ms = _now_ms() - $started;
	my $event_name = $kind eq 'macro_reload' ? 'macro_reload_executed' : 'action_executed';
	my $category = $kind eq 'macro_reload' ? 'macro' : 'action';
	push @ack_queue, {
		queued_at => _now_ms(),
		action_id => $action_id,
		poll_id => $poll_id,
		success => $success,
		result_code => $result_code,
		message => $msg,
		observed_latency_ms => $latency_ms,
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

	my @commands = (
		"conf macro_file $macro_file",
		"plugin reload $macro_plugin",
		"conf eventMacro_file $event_macro_file",
		"plugin reload $event_macro_plugin",
	);

	foreach my $safe_command (@commands) {
		my ($ok, $err) = _run_safe_openkore_command($safe_command);
		if (!$ok) {
			return (0, 'macro_reload_failed', "macro reload step failed for '$safe_command': $err");
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
	return if !_bridge_enabled();
	return if !@event_queue;

	my $batch_size = _cfg_int('aiSidecar_eventBatchSize', 20);
	$batch_size = 1 if $batch_size < 1;
	$batch_size = 100 if $batch_size > 100;
	$batch_size = scalar(@event_queue) if $batch_size > scalar(@event_queue);

	my @batch = splice @event_queue, 0, $batch_size;
	my $payload = {
		meta => _meta(_bot_id()),
		events => \@batch,
	};

	my $resp = _http_post_json('/v2/ingest/event', $payload);
	if (!$resp || $resp->{status} < 200 || $resp->{status} >= 300) {
		unshift @event_queue, @batch;
		my $max_queue = _cfg_int('aiSidecar_maxEventQueue', 300);
		$max_queue = 50 if $max_queue < 50;
		splice @event_queue, 0, @event_queue - $max_queue if @event_queue > $max_queue;
		_throttled_warning('v2_event_failed', '[aiSidecarBridge] v2 event push failed, retaining bounded queue.');
		return;
	}
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
		source_files => ['config.txt', 'ai_sidecar.txt', 'ai_sidecar_policy.txt'],
	};

	my $resp = _http_post_json('/v2/ingest/config', $payload);
	if (!$resp || $resp->{status} < 200 || $resp->{status} >= 300) {
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

	my $base_url = _cfg('aiSidecar_baseUrl', 'http://127.0.0.1:18081');
	$base_url =~ s{/+$}{};
	my ($scheme, $host, $port, $base_path) = $base_url =~ m{^(https?)://([^/:]+)(?::(\d+))?(/.*)?$}i;
	if (!$scheme || lc($scheme) ne 'http' || !$host) {
		_throttled_warning('invalid_base_url', "[aiSidecarBridge] invalid aiSidecar_baseUrl '$base_url'; expected http://host:port");
		return undef;
	}

	$port ||= 80;
	$base_path ||= '';
	my $request_path = "$base_path$path";
	$request_path =~ s{//+}{/}g;
	$request_path = "/$request_path" if $request_path !~ m{^/};

	my $body = eval { JSON::PP::encode_json($payload) };
	if (!$body || $@) {
		_throttled_warning('json_encode_failed', '[aiSidecarBridge] JSON encoding failed; request skipped.');
		return undef;
	}

	my $connect_timeout = _cfg_int('aiSidecar_connectTimeoutMs', 40) / 1000;
	my $io_timeout = _cfg_int('aiSidecar_ioTimeoutMs', 90) / 1000;
	$connect_timeout = 0.001 if $connect_timeout <= 0;
	$io_timeout = 0.001 if $io_timeout <= 0;

	my $sock = IO::Socket::INET->new(
		PeerHost => $host,
		PeerPort => $port,
		Proto => 'tcp',
		Timeout => $connect_timeout,
	);
	if (!$sock) {
		return undef;
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
	my $ok = eval {
		local $SIG{ALRM} = sub { die "bridge_http_timeout\n"; };
		alarm($io_timeout);
		print {$sock} $request;
		while (1) {
			my $chunk = '';
			my $read = sysread($sock, $chunk, 4096);
			last if !defined $read || $read <= 0;
			$raw_response .= $chunk;
			last if length($raw_response) > 1_000_000;
		}
		1;
	};
	alarm(0);
	close $sock;
	if (!$ok) {
		return undef;
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
		json => $json,
		raw => $response_body,
	};
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

sub _bot_id {
	my $master = $config{master} || 'unknown_master';
	my $identity = $char && $char->{name} ? $char->{name} : ($config{username} || 'unknown_user');
	return "$master:$identity";
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

sub _trim {
	my ($value, $max_len) = @_;
	$value = '' if !defined $value;
	$max_len = 0 + $max_len;
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
	my $frac = sprintf('%.3f', $t - int($t));
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

1;
