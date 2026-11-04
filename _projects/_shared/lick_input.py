"""
Which Bpod input the lick sensor is wired to, per behavior box.

Most boxes use the behavior port's own lick sensor (Port1: events Port1In/Port1Out). training01's
OPB815WZ IR beam break isn't read by the lick port, so an Arduino relays it into Bpod BNC In 2
(Calibration/arduino_lick_relay/): BNC2High = lick in, BNC2Low = lick out.

Task scripts take their lick events from lick_events() and register the channel as the
session-level LICK_INPUT value. The session CSV parser reads LICK_INPUT back, so a session is
parsed correctly on any box. A session without it (older ones) is Port1.
"""
import os

DEFAULT_LICK_INPUT = 'Port1'

# COMPUTERNAME (upper case) -> lick input channel, for boxes that don't use DEFAULT_LICK_INPUT.
LICK_INPUT_BY_COMPUTER = {
    'TRAINING01': 'BNC2',
}


def lick_input(computer_name=None):
    """ This box's lick input channel, e.g. 'Port1' or 'BNC2'. """
    if computer_name is None:
        computer_name = os.environ.get('COMPUTERNAME', '')
    return LICK_INPUT_BY_COMPUTER.get(computer_name.upper(), DEFAULT_LICK_INPUT)


def lick_events(channel=None):
    """ (lick-in event, lick-out event) for `channel` (default: this box's lick_input()).
    A behavior port's In is a lick in; on a BNC/Wire input HIGH is a lick in (the Arduino relay
    drives HIGH while the beam is broken). """
    if not channel:
        channel = lick_input()
    if channel.startswith('Port'):
        return channel + 'In', channel + 'Out'
    if channel.startswith(('BNC', 'Wire')):
        return channel + 'High', channel + 'Low'
    raise ValueError('Unknown lick input channel: {0!r}'.format(channel))
