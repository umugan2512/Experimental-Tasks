"""
Which Bpod input the lick sensor is wired to, per behavior box -- and the one name its events get.

Most boxes use the behavior port's own lick sensor (Port1: events Port1In/Port1Out). training01's
OPB815WZ IR beam break isn't read by the lick port, so an Arduino relays it into Bpod BNC In 2
(Calibration/arduino_lick_relay/): BNC2High = lick in, BNC2Low = lick out.

Same names on every box (since 2026-11-12): TrialRunner calls name_lick_events(), which relabels
this box's physical lick events in the Bpod's event-name list as LOGGED_LICK_EVENTS
(Port1In/Port1Out). Names are positional, so only the label changes: state machines listen for
Port1In, and every CSV EVENT row says Port1In, on every box, whatever the wiring. On a box where the
lick isn't on Port1, the physical Port1 events become Port1In_unwired/Port1Out_unwired.

Task scripts take their lick events from lick_events() (no argument = the logged names) and call
register_lick_input(): LICK_INPUT is the channel the CSV's lick events are named after (always
'Port1' now), LICK_INPUT_PHYSICAL the real wiring. The session CSV parser reads LICK_INPUT, so older
training01 sessions (LICK_INPUT=BNC2, BNC2High/BNC2Low rows) still parse. A session without it
(older ones) is Port1.
"""
import os

DEFAULT_LICK_INPUT = 'Port1'

# The channel whose event names lick events are logged under, on every box.
LOGGED_LICK_INPUT = 'Port1'
LOGGED_LICK_EVENTS = ('Port1In', 'Port1Out')
UNWIRED_SUFFIX = '_unwired'

# COMPUTERNAME (upper case) -> lick input channel, for boxes that don't use DEFAULT_LICK_INPUT.
LICK_INPUT_BY_COMPUTER = {
    'TRAINING01': 'BNC2',
}


def lick_input(computer_name=None):
    """ This box's physical lick input channel, e.g. 'Port1' or 'BNC2'. """
    if computer_name is None:
        computer_name = os.environ.get('COMPUTERNAME', '')
    return LICK_INPUT_BY_COMPUTER.get(computer_name.upper(), DEFAULT_LICK_INPUT)


def physical_lick_events(channel=None):
    """ The hardware's own (lick-in, lick-out) event names for `channel` (default: this box's
    lick_input()). A behavior port's In is a lick in; on a BNC/Wire input HIGH is a lick in (the
    Arduino relay drives HIGH while the beam is broken). """
    if not channel:
        channel = lick_input()
    if channel.startswith('Port'):
        return channel + 'In', channel + 'Out'
    if channel.startswith(('BNC', 'Wire')):
        return channel + 'High', channel + 'Low'
    raise ValueError('Unknown lick input channel: {0!r}'.format(channel))


def lick_events(channel=None):
    """ With no argument: the (lick-in, lick-out) names state machines and the CSV use, the same
    on every box (LOGGED_LICK_EVENTS; requires name_lick_events() on the Bpod, which TrialRunner
    does). With a channel: that channel's physical event names (rig_check, tests). """
    if not channel:
        return LOGGED_LICK_EVENTS
    return physical_lick_events(channel)


def name_lick_events(bpod, channel=None):
    """ Relabels this box's physical lick events in bpod.hardware.channels.event_names as
    LOGGED_LICK_EVENTS, and the physical Port1In/Port1Out as *_unwired so no name is duplicated.
    Idempotent; no-op on a Port1 box or without a hardware description (a test double). Returns
    True if anything was renamed. Must run before any StateMachine is built. """
    names = getattr(getattr(getattr(bpod, 'hardware', None), 'channels', None), 'event_names', None)
    if not names:
        return False
    physical = physical_lick_events(channel)
    if physical == LOGGED_LICK_EVENTS:
        return False
    if not all(p in names for p in physical):
        return False   # already renamed, or this Bpod has no such input
    phys_idx = [names.index(p) for p in physical]
    logged_idx = [names.index(l) if l in names else None for l in LOGGED_LICK_EVENTS]
    for idx, logged in zip(logged_idx, LOGGED_LICK_EVENTS):
        if idx is not None:
            names[idx] = logged + UNWIRED_SUFFIX
    for idx, logged in zip(phys_idx, LOGGED_LICK_EVENTS):
        names[idx] = logged
    return True


def register_lick_input(bpod):
    """ Session-level VALs: LICK_INPUT = the channel the CSV's lick events are named after,
    LICK_INPUT_PHYSICAL = where the sensor is really wired. """
    bpod.register_value('LICK_INPUT', LOGGED_LICK_INPUT)
    bpod.register_value('LICK_INPUT_PHYSICAL', lick_input())
