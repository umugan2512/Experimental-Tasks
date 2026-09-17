# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Valve flush utility -- holds a reward valve open indefinitely (not a timed pulse) so clean water or
ethanol can be run through the tubing for cleaning/priming, independent of calibration.

Uses `Bpod.manual_override(...)` rather than a StateMachine pulse (what `calibrate_liquid.py`'s
`PulseRunner` uses): a StateMachine's `state_timer` always has a fixed duration, but
`manual_override` can set an output and leave it there indefinitely across separate calls -- open on
one call, close on a later one. This is only safe here because this is a standalone script with no
state machine ever running concurrently (see CLAUDE.md's own documented hazard: manual_override/
trigger_output/trigger_softcode share the same physical Bpod USB connection a running
`run_state_machine()` call's own blocking loop reads from, so calling any of them from a second,
concurrent thread/process is genuinely unsynchronized access -- not a concern for this script, which
never has a state machine running at all).

Run directly: `python flush_valve.py` (needs the Bpod board and valve physically connected). Same
"connects directly, skips all PyBpod project/task scaffolding" convention as `calibrate_liquid.py`
-- see that file's own module docstring for why `Bpod(serial_port=VAR_BPOD_SERIAL_PORT)` is required
here too, not a bare `Bpod()`.

Interactive terminal loop -- no GUI needed for a two-state (open/closed) maintenance tool:
    Enter    -> toggle the valve open/closed
    q, Enter -> close the valve (if open) and quit

The valve is always closed on exit (normal quit, Ctrl+C, or any error) -- never left open
unattended.
"""
from pybpodapi.protocol import Bpod

VAR_VALVE_IDS = [1]   # this rig's single valve (see CLAUDE.md: "wheel-turn choice + single valve")
VAR_BPOD_SERIAL_PORT = 'COM10'   # MACHINE-SPECIFIC -- this rig's own confirmed Bpod COM port, same
                                  # constant/convention as calibrate_liquid.py's own
                                  # VAR_BPOD_SERIAL_PORT. Update per box.


def _set_valve(my_bpod, valve_id, is_open):
    my_bpod.manual_override(Bpod.ChannelTypes.OUTPUT, Bpod.ChannelNames.VALVE,
                             channel_number=valve_id, value=1 if is_open else 0)


def main():
    valve_id = VAR_VALVE_IDS[0]
    if len(VAR_VALVE_IDS) > 1:
        raw = input('Valve id {0}: '.format(VAR_VALVE_IDS)).strip()
        if raw:
            valve_id = int(raw)

    my_bpod = Bpod(serial_port=VAR_BPOD_SERIAL_PORT)
    print("Connected to Bpod on {0}".format(my_bpod.serial_port), flush=True)

    is_open = False
    try:
        print("Valve {0} flush -- press Enter to toggle open/closed, 'q' + Enter to quit.".format(
            valve_id))
        while True:
            cmd = input('[{0}] > '.format('OPEN' if is_open else 'closed')).strip().lower()
            if cmd == 'q':
                break
            is_open = not is_open
            _set_valve(my_bpod, valve_id, is_open)
            print('Valve {0} is now {1}.'.format(valve_id, 'OPEN' if is_open else 'CLOSED'))
    except KeyboardInterrupt:
        print()
    finally:
        if is_open:
            _set_valve(my_bpod, valve_id, False)
            print('Valve {0} closed on exit.'.format(valve_id))
        my_bpod.close()


if __name__ == '__main__':
    main()
