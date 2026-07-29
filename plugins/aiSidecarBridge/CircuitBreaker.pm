package aiSidecarBridge::CircuitBreaker;

use strict;
use warnings;

=head1 NAME

aiSidecarBridge::CircuitBreaker - Circuit breaker for ZMQ IPC connections

=head1 SYNOPSIS

    use aiSidecarBridge::CircuitBreaker;

    my $cb = aiSidecarBridge::CircuitBreaker->new(
        threshold => 10,
        name      => 'zmq_main',
    );

    if ($cb->check()) {
        # attempt connection
        if ($ok) { $cb->record_success(); }
        else     { $cb->record_failure(); }
    }

    # Manual reset
    $cb->reset();

=head1 DESCRIPTION

Tracks consecutive ZMQ connection failures. After C<threshold> consecutive
failures the circuit opens (trips) and refuses further connection attempts
until a manual C<reset()> call (typically on plugin reload).

=head1 ATTRIBUTES

=over

=item * threshold - number of consecutive failures before tripping (default 10)

=item * name - human-readable label for debug logging (default 'zmq')

=item * _consecutive_failures - internal counter

=item * _tripped - boolean, true when circuit is open

=item * _tripped_at_ms - timestamp when circuit was tripped

=item * _trip_count - total times the circuit has ever tripped

=back

=cut

our $VERSION = '1.0.0';

sub new {
    my ($class, %args) = @_;
    my $self = {
        threshold           => $args{threshold} || 10,
        name                => $args{name}      || 'zmq',
        _consecutive_failures => 0,
        _tripped            => 0,
        _tripped_at_ms      => 0,
        _trip_count         => 0,
    };
    bless $self, $class;
    return $self;
}

=head2 check

Returns 1 if the circuit is closed (allows requests).
Returns 0 if the circuit is open (tripped, blocks requests).

=cut

sub check {
    my ($self) = @_;
    if ($self->{_tripped}) {
        return 0;
    }
    return 1;
}

=head2 record_success

Resets the consecutive-failure counter. If the circuit was half-open
this transitions it to fully closed. No-op when already closed.

=cut

sub record_success {
    my ($self) = @_;
    $self->{_consecutive_failures} = 0;
    if ($self->{_tripped}) {
        $self->{_tripped} = 0;
        $self->{_tripped_at_ms} = 0;
    }
}

=head2 record_failure

Increments the consecutive-failure counter. If it reaches C<threshold>
the circuit opens (trips) and refuses further requests until C<reset()>.

Returns the new consecutive-failure count.

=cut

sub record_failure {
    my ($self) = @_;
    $self->{_consecutive_failures} += 1;

    if ($self->{_consecutive_failures} >= $self->{threshold}) {
        $self->{_tripped} = 1;
        $self->{_tripped_at_ms} = _now_ms();
        $self->{_trip_count} += 1;
    }

    return $self->{_consecutive_failures};
}

=head2 reset

Manually reset the circuit breaker. Clears the consecutive-failure
counter and closes the circuit. This is the ONLY way to re-open a
tripped circuit (called on plugin reload).

=cut

sub reset {
    my ($self) = @_;
    $self->{_consecutive_failures} = 0;
    $self->{_tripped} = 0;
    $self->{_tripped_at_ms} = 0;
}

=head2 is_tripped

Returns 1 if the circuit is currently open (tripped), 0 otherwise.

=cut

sub is_tripped {
    my ($self) = @_;
    return $self->{_tripped} ? 1 : 0;
}

=head2 consecutive_failures

Returns the current consecutive-failure count.

=cut

sub consecutive_failures {
    my ($self) = @_;
    return $self->{_consecutive_failures};
}

=head2 summary

Returns a hashref with the current circuit breaker state for diagnostics.

=cut

sub summary {
    my ($self) = @_;
    return {
        name                => $self->{name},
        tripped             => $self->{_tripped} ? 1 : 0,
        consecutive_failures => $self->{_consecutive_failures},
        threshold           => $self->{threshold},
        tripped_at_ms       => $self->{_tripped_at_ms},
        trip_count          => $self->{_trip_count},
    };
}

=head2 to_json_hash

Returns a plain hashref suitable for JSON encoding (all values are
simple scalars).

=cut

sub to_json_hash {
    my ($self) = @_;
    return {
        name                => $self->{name},
        tripped             => $self->{_tripped} ? 1 : 0,
        consecutive_failures => $self->{_consecutive_failures},
        threshold           => $self->{threshold},
        trip_count          => $self->{_trip_count},
    };
}

# ── Millisecond timestamp helper (same as bridge) ──
sub _now_ms {
    my $epoch_secs = time;
    my $ms = 0;
    # Try Time::HiRes if available
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
