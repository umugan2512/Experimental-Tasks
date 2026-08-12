# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Lick sensor / beam-break bring-up test for Port1: each trial waits for the beam to break
(Port1In), then opens Valve1 for VAR_REWARD_DURATION as a reward. No timer pacing -- unlike
lick_timer.py's global-timer-driven design, a trial here only advances once the beam is actually
broken, so it can be exercised by hand at whatever pace the tester likes.

A Port1In has to hold (stay broken) for VAR_MIN_LICK_DURATION_S before it counts as a real lick --
filters brief ambient-light-triggered flicker on the IR sensor without needing to touch the port's
own physical sensitivity trimpot. Kept deliberately tiny: a genuine tongue lick's own beam-break
dwell time is itself short, so this only needs to be nominal, not a real debounce window.
"""
from pybpodapi.protocol import Bpod, StateMachine

VAR_N_TRIALS = 30
VAR_REWARD_DURATION = 0.1        # seconds Valve1 stays open per lick
VAR_MIN_LICK_DURATION_S = 0.01   # beam must stay broken this long to count -- see module docstring

my_bpod = Bpod()
print("Connected to Bpod on {0}".format(my_bpod.serial_port), flush=True)

print("Starting {0} trials -- break the Port1 beam to trigger a reward".format(VAR_N_TRIALS), flush=True)

for trial in range(VAR_N_TRIALS):
    sma = StateMachine(my_bpod)

    sma.add_state(
        state_name='WaitForLick',
        state_timer=0,
        state_change_conditions={Bpod.Events.Port1In: 'ConfirmLick'},
        output_actions=[])

    sma.add_state(
        state_name='ConfirmLick',
        state_timer=VAR_MIN_LICK_DURATION_S,
        state_change_conditions={
            Bpod.Events.Port1Out: 'WaitForLick',  # beam un-broke before the minimum -- noise, retry
            Bpod.Events.Tup: 'Reward',            # stayed broken long enough -- genuine lick
        },
        output_actions=[])

    sma.add_state(
        state_name='Reward',
        state_timer=VAR_REWARD_DURATION,
        state_change_conditions={Bpod.Events.Tup: 'exit'},
        output_actions=[(Bpod.OutputChannels.Valve, 1)])

    my_bpod.send_state_machine(sma)
    my_bpod.run_state_machine(sma)

    print("Trial {0}/{1}: beam broken, reward delivered".format(trial + 1, VAR_N_TRIALS), flush=True)

my_bpod.close()

print("Done: {0} trials completed".format(VAR_N_TRIALS), flush=True)
