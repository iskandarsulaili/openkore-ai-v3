package aiSidecarBridge::ConnectionMetrics;

use strict;
use warnings;

=head1 NAME

aiSidecarBridge::ConnectionMetrics - Track ZMQ/HTTP connection quality

=head1 SYNOPSIS

    use aiSidecarBridge::ConnectionMetrics;

    my $m = aiSidecarBridge::ConnectionMetrics->new(
        max_latency_samples => 100,
    );

    $m->record_success(12.5);   # latency in ms
    $m->record_failure('timeout');

    my $stats = $m->summary();
    print "avg_latency: $stats->{avg_latency_ms}\n";

=head1 DESCRIPTION

Tracks connection quality metrics for the ZMQ/HTTP IPC layer:
total messages, successes, failures, average latency, consecutive
failures, and latency distribution.

=cut

our $VERSION = '1.0.0';

sub new {
    my ($class, %args) = @_;
    my $self = {
        max_latency_samples => $args{max_latency_samples} || 100,
        # Counters
        total_messages       => 0,
        total_successes      => 0,
        total_failures       => 0,
        consecutive_failures => 0,
        # Latency tracking
        latency_samples      => [],
        latency_sum_ms       => 0,
        min_latency_ms       => undef,
        max_latency_ms       => undef,
        # Failure breakdown
        failure_reasons      => {},
        # Last-event timestamps
        last_success_at_ms   => 0,
        last_failure_at_ms   => 0,
        last_message_at_ms   => 0,
        # Window tracking (rolling window of last N seconds)
        window_seconds       => $args{window_seconds} || 300,  # 5 min default
        window_messages      => 0,
        window_failures      => 0,
        window_start_ms      => _now_ms(),
    };
    bless $self, $class;
    return $self;
}

=head2 record_message

Increment the total message counter. Call this for every send attempt
regardless of success/failure.

=cut

sub record_message {
    my ($self) = @_;
    $self->{total_messages} += 1;
    $self->{last_message_at_ms} = _now_ms();
    $self->_prune_window();
    $self->{window_messages} += 1;
}

=head2 record_success

Record a successful message send.

Parameters:
    latency_ms - round-trip latency in milliseconds (optional, defaults to 0)

=cut

sub record_success {
    my ($self, $latency_ms) = @_;
    $latency_ms = 0 unless defined $latency_ms && $latency_ms >= 0;

    $self->{total_successes} += 1;
    $self->{consecutive_failures} = 0;
    $self->{last_success_at_ms} = _now_ms();
    $self->{last_message_at_ms} = _now_ms();

    # Track latency
    push @{$self->{latency_samples}}, $latency_ms;
    if (scalar(@{$self->{latency_samples}}) > $self->{max_latency_samples}) {
        shift @{$self->{latency_samples}};
    } else {
        $self->{latency_sum_ms} += $latency_ms;
    }

    if (!defined $self->{min_latency_ms} || $latency_ms < $self->{min_latency_ms}) {
        $self->{min_latency_ms} = $latency_ms;
    }
    if (!defined $self->{max_latency_ms} || $latency_ms > $self->{max_latency_ms}) {
        $self->{max_latency_ms} = $latency_ms;
    }

    $self->_prune_window();
    $self->{window_messages} += 1;
}

=head2 record_failure

Record a failed message send.

Parameters:
    reason - string describing the failure (optional, defaults to 'unknown')

=cut

sub record_failure {
    my ($self, $reason) = @_;
    $reason = 'unknown' unless defined $reason && $reason ne '';

    $self->{total_failures} += 1;
    $self->{consecutive_failures} += 1;
    $self->{last_failure_at_ms} = _now_ms();
    $self->{last_message_at_ms} = _now_ms();

    $self->{failure_reasons}{$reason} = 0
        unless exists $self->{failure_reasons}{$reason};
    $self->{failure_reasons}{$reason} += 1;

    $self->_prune_window();
    $self->{window_messages} += 1;
    $self->{window_failures} += 1;
}

=head2 reset

Reset all counters and samples to their initial state.

=cut

sub reset {
    my ($self) = @_;
    $self->{total_messages}       = 0;
    $self->{total_successes}      = 0;
    $self->{total_failures}       = 0;
    $self->{consecutive_failures} = 0;
    $self->{latency_samples}      = [];
    $self->{latency_sum_ms}       = 0;
    $self->{min_latency_ms}       = undef;
    $self->{max_latency_ms}       = undef;
    $self->{failure_reasons}      = {};
    $self->{last_success_at_ms}   = 0;
    $self->{last_failure_at_ms}   = 0;
    $self->{last_message_at_ms}   = 0;
    $self->{window_messages}      = 0;
    $self->{window_failures}      = 0;
    $self->{window_start_ms}      = _now_ms();
}

=head2 summary

Returns a hashref with computed metrics suitable for logging or telemetry.

Keys:
    total_messages, total_successes, total_failures,
    consecutive_failures, success_rate (0-1),
    avg_latency_ms, min_latency_ms, max_latency_ms,
    last_success_at_ms, last_failure_at_ms,
    window_error_rate, failure_reasons

=cut

sub summary {
    my ($self) = @_;
    my $total = $self->{total_messages} || 1;
    my $win_total = $self->{window_messages} || 1;
    my $avg_latency = 0;
    if (scalar(@{$self->{latency_samples}}) > 0) {
        $avg_latency = $self->{latency_sum_ms} / scalar(@{$self->{latency_samples}});
    }
    return {
        total_messages       => $self->{total_messages},
        total_successes      => $self->{total_successes},
        total_failures       => $self->{total_failures},
        consecutive_failures => $self->{consecutive_failures},
        success_rate         => sprintf('%.4f', $self->{total_successes} / $total),
        avg_latency_ms       => sprintf('%.2f', $avg_latency),
        min_latency_ms       => defined $self->{min_latency_ms} ? sprintf('%.2f', $self->{min_latency_ms}) : undef,
        max_latency_ms       => defined $self->{max_latency_ms} ? sprintf('%.2f', $self->{max_latency_ms}) : undef,
        last_success_at_ms   => $self->{last_success_at_ms},
        last_failure_at_ms   => $self->{last_failure_at_ms},
        window_error_rate    => sprintf('%.4f', $self->{window_failures} / $win_total),
        window_messages      => $self->{window_messages},
        window_failures      => $self->{window_failures},
        failure_reasons      => {%{$self->{failure_reasons}}},
    };
}

=head2 to_json_hash

Returns a plain hashref suitable for JSON encoding (all values are
simple scalars, no blessed references).

=cut

sub to_json_hash {
    my ($self) = @_;
    my $total = $self->{total_messages} || 1;
    my $win_total = $self->{window_messages} || 1;
    my $avg_latency = 0;
    if (scalar(@{$self->{latency_samples}}) > 0) {
        $avg_latency = $self->{latency_sum_ms} / scalar(@{$self->{latency_samples}});
    }
    return {
        total_messages       => $self->{total_messages},
        total_successes      => $self->{total_successes},
        total_failures       => $self->{total_failures},
        consecutive_failures => $self->{consecutive_failures},
        success_rate         => sprintf('%.4f', $self->{total_successes} / $total),
        avg_latency_ms       => sprintf('%.2f', $avg_latency),
        min_latency_ms       => defined $self->{min_latency_ms} ? sprintf('%.2f', $self->{min_latency_ms}) : 0,
        max_latency_ms       => defined $self->{max_latency_ms} ? sprintf('%.2f', $self->{max_latency_ms}) : 0,
        last_success_at_ms   => $self->{last_success_at_ms},
        last_failure_at_ms   => $self->{last_failure_at_ms},
        window_error_rate    => sprintf('%.4f', $self->{window_failures} / $win_total),
        window_messages      => $self->{window_messages},
        window_failures      => $self->{window_failures},
    };
}

# Prune the rolling window: if window_start is older than window_seconds,
# shift the window forward and reset window counters.
sub _prune_window {
    my ($self) = @_;
    my $now_ms = _now_ms();
    my $window_ms = ($self->{window_seconds} || 300) * 1000;
    if ($now_ms - $self->{window_start_ms} > $window_ms) {
        $self->{window_messages} = 0;
        $self->{window_failures} = 0;
        $self->{window_start_ms} = $now_ms;
    }
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
