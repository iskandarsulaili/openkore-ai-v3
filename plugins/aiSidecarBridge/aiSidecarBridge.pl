package aiSidecarBridge;

use strict;
use warnings;

use Commands;
use FileParsers qw(parseConfigFile);
use Globals qw(%config $char $field @ai_seq $net);
use IO::Socket::INET;
use Log qw(debug message warning);
use Misc qw(calcPosition);
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

my @ack_queue;
my @telemetry_queue;
my %last_warn_at_ms;

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
	@ack_queue = ();
	@telemetry_queue = ();
	%last_warn_at_ms = ();
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

	_attempt_register('start3');
}

sub on_mainLoop_pre {
	return unless _bridge_enabled();
	my $now = _now_ms();

	if (_cfg_bool('aiSidecar_snapshotEnabled', 1) && $now >= $next_snapshot_at_ms) {
		$next_snapshot_at_ms = $now + _cfg_int('aiSidecar_snapshotIntervalMs', 500);
		_send_snapshot();
	}
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
		my $pos = eval { calcPosition($char) };
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
	my $started = _now_ms();

	my ($success, $result_code, $msg) = (0, 'invalid_action', 'invalid action payload');

	if ($kind ne 'command') {
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
		'action',
		'action_executed',
		$msg,
		{ observed_latency_ms => $latency_ms + 0 },
		{ result_code => $result_code },
	);
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
