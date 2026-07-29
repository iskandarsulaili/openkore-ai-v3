package HTTPClient;

use strict;
use warnings;

=head1 NAME

aiSidecarBridge::HTTPClient - ZMQ push socket with HTTP/1.1 fallback IPC

=head1 SYNOPSIS

    use HTTPClient;
    use CircuitBreaker;
    use ConnectionMetrics;

    my $cb = CircuitBreaker->new(threshold => 10);
    my $metrics = ConnectionMetrics->new();

    my $client = HTTPClient->new(
        zmq_address       => 'tcp://127.0.0.1:5559',
        http_base_url     => 'http://127.0.0.1:8000',
        json_encode_cb    => sub { JSON::PP::encode_json($_[0]) },
        circuit_breaker   => $cb,
        metrics           => $metrics,
        debug_log_cb      => sub { Log::debug($_[0], 'HTTPClient', 2) },
        warn_log_cb       => sub { Log::warning($_[0]) },
    );

    # Send a hashref — tries ZMQ first, falls back to HTTP
    my $ok = $client->send_json('/v1/state', { hello => 'world' });

    # Check which transport is active
    my $transport = $client->active_transport();  # 'zmq', 'http', or 'none'

=head1 DESCRIPTION

Dual-transport IPC client. On every C<send_json()> call:

 1. Checks the circuit breaker — if tripped, goes straight to HTTP.
 2. Tries ZMQ push socket (ZMQ::FFI).
 3. On ZMQ failure (socket unavailable, send error), increments the
    circuit breaker, records the failure in metrics, and falls back to
    an HTTP POST to C<http_base_url + path>.
 4. On ZMQ success, resets the circuit breaker and records success.

All metrics (latency, success/failure counts, etc.) are forwarded to
the C<ConnectionMetrics> object.

=cut

our $VERSION = '1.0.0';

sub new {
    my ($class, %args) = @_;
    my $self = {
        # ZMQ configuration
        zmq_address       => $args{zmq_address}    || 'tcp://127.0.0.1:5559',
        zmq_connect_ms    => $args{zmq_connect_ms} || 500,
        zmq_linger_ms     => $args{zmq_linger_ms}  || 100,

        # HTTP fallback configuration
        http_base_url     => $args{http_base_url}  || 'http://127.0.0.1:8000',
        http_connect_ms   => $args{http_connect_ms} || 2000,
        http_io_ms        => $args{http_io_ms}      || 5000,

        # Callbacks
        json_encode_cb    => $args{json_encode_cb},
        debug_log_cb      => $args{debug_log_cb}   || sub { },
        warn_log_cb       => $args{warn_log_cb}    || sub { },

        # Circuit breaker and metrics
        circuit_breaker   => $args{circuit_breaker},
        metrics           => $args{metrics},

        # Internal state
        _zmq_context     => undef,
        _zmq_push_socket => undef,
        _zmq_available   => 0,
        _zmq_initialized => 0,
        _last_transport  => 'none',  # 'zmq', 'http', or 'none'
    };

    bless $self, $class;
    return $self;
}

=head2 send_json

Send a structured payload (hashref) to the sidecar.

Parameters:
    path    - URL path for HTTP fallback (e.g. '/v1/state')
    payload - hashref to encode as JSON

Returns 1 on success (ZMQ or HTTP), 0 on failure.

=cut

sub send_json {
    my ($self, $path, $payload) = @_;
    return 0 unless defined $payload && ref($payload) eq 'HASH';
    $path ||= '/v1/state';

    $self->{metrics}->record_message() if $self->{metrics};

    my $start_ms = _now_ms();
    my $latency_ms;

    # ── Try ZMQ first (if circuit breaker allows) ──
    my $cb = $self->{circuit_breaker};
    if (!$cb || $cb->check()) {
        my $zmq_ok = $self->_send_via_zmq($payload);
        if ($zmq_ok) {
            $latency_ms = _now_ms() - $start_ms;
            $self->{_last_transport} = 'zmq';
            $self->{metrics}->record_success($latency_ms) if $self->{metrics};
            $cb->record_success() if $cb;
            $self->{debug_log_cb}->("[HTTPClient] sent via ZMQ to $self->{zmq_address} ($latency_ms ms)");
            return 1;
        }

        # ZMQ failed — record failure in circuit breaker
        $cb->record_failure() if $cb;
        my $fail_count = $cb ? $cb->consecutive_failures() : 0;
        if ($fail_count >= 10) {
            $self->{warn_log_cb}->("[HTTPClient] ZMQ circuit breaker tripped after $fail_count failures, falling back to HTTP");
        } else {
            $self->{debug_log_cb}->("[HTTPClient] ZMQ send failed, attempt $fail_count/10, falling back to HTTP");
        }
    } elsif ($cb && $cb->is_tripped()) {
        $self->{debug_log_cb}->("[HTTPClient] ZMQ circuit breaker is OPEN — using HTTP fallback");
    }

    # ── HTTP fallback ──
    my $http_ok = $self->_send_via_http($path, $payload);
    $latency_ms = _now_ms() - $start_ms - ($self->{zmq_connect_ms} || 0);

    if ($http_ok) {
        $self->{_last_transport} = 'http';
        $self->{metrics}->record_success($latency_ms) if $self->{metrics};
        $self->{debug_log_cb}->("[HTTPClient] sent via HTTP to $self->{http_base_url}$path ($latency_ms ms)");
        return 1;
    }

    # Both failed
    $self->{_last_transport} = 'none';
    $self->{metrics}->record_failure('all_transports_failed') if $self->{metrics};
    $self->{warn_log_cb}->("[HTTPClient] BOTH ZMQ and HTTP failed for $path");
    return 0;
}

=head2 send_state (convenience)

Shortcut for C<send_json('/v1/state', $state)>.

=cut

sub send_state {
    my ($self, $state) = @_;
    return $self->send_json('/v1/state', $state);
}

=head2 active_transport

Returns the transport used by the last successful C<send_json()> call:
'zmq', 'http', or 'none'.

=cut

sub active_transport {
    my ($self) = @_;
    return $self->{_last_transport};
}

=head2 close

Tear down the ZMQ socket and context. Call on plugin unload/reload.

=cut

sub close {
    my ($self) = @_;
    if ($self->{_zmq_push_socket}) {
        eval { $self->{_zmq_push_socket}->disconnect($self->{zmq_address}) };
        eval { $self->{_zmq_push_socket}->close() };
        undef $self->{_zmq_push_socket};
    }
    if ($self->{_zmq_context}) {
        eval { $self->{_zmq_context}->term() };
        undef $self->{_zmq_context};
    }
    $self->{_zmq_initialized} = 0;
    $self->{_zmq_available} = 0;
    $self->{_last_transport} = 'none';
}

=head2 reset_circuit_breaker

Convenience: reset the circuit breaker attached to this client.

=cut

sub reset_circuit_breaker {
    my ($self) = @_;
    $self->{circuit_breaker}->reset() if $self->{circuit_breaker};
}

=head2 metrics_summary

Return the metrics object's summary hashref.

=cut

sub metrics_summary {
    my ($self) = @_;
    return $self->{metrics} ? $self->{metrics}->summary() : {};
}

# ── internal: ZMQ send via ZMQ::FFI push socket ──
sub _send_via_zmq {
    my ($self, $payload) = @_;

    # Lazy-init ZMQ
    $self->_zmq_init() unless $self->{_zmq_initialized};
    return 0 unless $self->{_zmq_available};

    my $socket = $self->{_zmq_push_socket};
    return 0 unless $socket;

    # Encode payload to JSON bytes
    my $json;
    if ($self->{json_encode_cb}) {
        $json = eval { $self->{json_encode_cb}->($payload) };
        return 0 if !defined $json || $@;
    } else {
        eval { $json = JSON::PP::encode_json($payload) };
        return 0 if !defined $json || $@;
    }

    # Send via ZMQ push socket
    my $ok = eval {
        $socket->send($json);
        1;
    };
    if (!$ok) {
        my $err = $@ || 'zmq_send_failed';
        $self->{debug_log_cb}->("[HTTPClient] ZMQ send error: $err");
        return 0;
    }

    return 1;
}

sub _zmq_init {
    my ($self) = @_;
    $self->{_zmq_initialized} = 1;
    $self->{_zmq_available} = 0;

    # Check that ZMQ::FFI is available
    my $zmq_ffi_ok = eval { require ZMQ::FFI; 1; };
    if (!$zmq_ffi_ok) {
        $self->{debug_log_cb}->("[HTTPClient] ZMQ::FFI not available, ZMQ transport disabled");
        return;
    }

    # Check that ZMQ::FFI::Socket is available
    my $socket_class_ok = eval { require ZMQ::FFI::Socket; 1; };
    if (!$socket_class_ok) {
        $self->{debug_log_cb}->("[HTTPClient] ZMQ::FFI::Socket not available, ZMQ transport disabled");
        return;
    }

    # Create ZMQ context
    my $context;
    eval {
        $context = ZMQ::FFI->new();
    };
    if (!$context || $@) {
        $self->{debug_log_cb}->("[HTTPClient] Failed to create ZMQ context: $@");
        return;
    }
    $self->{_zmq_context} = $context;

    # Create PUSH socket
    my $socket;
    eval {
        $socket = $context->socket('ZMQ_PUSH');
    };
    if (!$socket || $@) {
        $self->{debug_log_cb}->("[HTTPClient] Failed to create ZMQ_PUSH socket: $@");
        return;
    }
    $self->{_zmq_push_socket} = $socket;

    # Set linger
    eval {
        $socket->setsockopt(ZMQ::FFI::Constants->constant('ZMQ_LINGER'),
                            $self->{zmq_linger_ms});
    };

    # Connect
    eval {
        $socket->connect($self->{zmq_address});
    };
    if ($@) {
        $self->{debug_log_cb}->("[HTTPClient] ZMQ connect to $self->{zmq_address} failed: $@");
        return;
    }

    $self->{_zmq_available} = 1;
    $self->{debug_log_cb}->("[HTTPClient] ZMQ push socket connected to $self->{zmq_address}");
}

# ── internal: HTTP fallback via raw TCP socket (same pattern as bridge _http_post_json) ──
sub _send_via_http {
    my ($self, $path, $payload) = @_;

    # Encode payload
    my $body;
    if ($self->{json_encode_cb}) {
        $body = eval { $self->{json_encode_cb}->($payload) };
        return 0 if !defined $body || $@;
    } else {
        eval { $body = JSON::PP::encode_json($payload) };
        return 0 if !defined $body || $@;
    }

    # Parse base URL
    my $base_url = $self->{http_base_url} || 'http://127.0.0.1:8000';
    $base_url =~ s{/+$}{};
    my ($scheme, $host, $port, $base_path) =
        $base_url =~ m{^(https?)://([^/:]+)(?::(\d+))?(/.*)?$}i;

    return 0 if !$scheme || lc($scheme) ne 'http' || !$host;
    $port ||= 80;
    $base_path ||= '';

    my $request_path = "$base_path$path";
    $request_path =~ s{//+}{/}g;
    $request_path = "/$request_path" if $request_path !~ m{^/};

    # Socket connect with timeout
    require IO::Socket::INET;
    my $timeout = ($self->{http_connect_ms} || 2000) / 1000;
    $timeout = 0.001 if $timeout <= 0;

    my $sock = IO::Socket::INET->new(
        PeerHost => $host,
        PeerPort => $port,
        Proto    => 'tcp',
        Timeout  => $timeout,
    );
    return 0 if !$sock;
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

    # Send request and read response with I/O timeout
    my $io_timeout = ($self->{http_io_ms} || 5000) / 1000;
    $io_timeout = 0.001 if $io_timeout <= 0;

    my $ok = eval {
        local $SIG{ALRM} = sub { die "httpclient_timeout\n"; };
        alarm($io_timeout);

        print {$sock} $request;

        my $raw_response = '';
        while (1) {
            my $chunk = '';
            my $read = sysread($sock, $chunk, 4096);
            last if !defined $read || $read <= 0;
            $raw_response .= $chunk;
            last if $raw_response =~ /\r?\n\r?\n.*$/s && length($raw_response) > 0;
        }

        alarm(0);
        $raw_response =~ /^HTTP\/\d+\.\d+\s+(\d+)/;
        my $status = $1 || 0;
        return $status >= 200 && $status < 300 ? 1 : 0;
    };
    alarm(0);

    if (!$ok) {
        my $err = $@ || 'http_io_failure';
        $self->{debug_log_cb}->("[HTTPClient] HTTP POST $request_path failed: $err");
    }

    CORE::close($sock);
    return $ok ? 1 : 0;
}

sub _now_ms {
    my $epoch_secs = time;
    my $ms = 0;
    eval {
        require Time::HiRes;
        my ($sec, $usec) = Time::HiRes::gettimeofday();
        $epoch_secs = $sec;
        $ms = int($usec / 1000);
    };
    return ($epoch_secs * 1000) + $ms;
}

1;

__END__

=head1 COPYRIGHT

Copyright (C) 2026 by the OpenKore AI project.
