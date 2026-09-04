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

**Refractory lockout after Reward, confirmed necessary on hardware**: crossing the beam toward the
spout AND crossing back away from it both appeared to trigger a reward. The actual cause isn't
Port1Out being scored anywhere (ConfirmLick already routes it to a no-reward retry, never to
Reward) -- it's that each trial builds and runs a brand-new StateMachine, and Bpod's digital-input
detection re-arms fresh every time a new state machine starts listening on a port. If the beam is
STILL broken when the next trial's WaitForLick begins (any lick/contact that outlasts the ~0.11s
ConfirmLick+Reward cycle), the still-held beam gets immediately misdetected as a brand-new Port1In,
re-firing the whole cycle -- repeatedly, for as long as contact is held. The last such repeat just
happens to land near the moment of actually retracting, which is why retracting looks like it
triggers a reward. Fixed with a fixed-duration Refractory state after Reward, not an explicit
"wait for Port1Out" state -- Bpod events are edge-triggered with no way to query a digital input's
CURRENT level, so a state trying to wait for a release that may have already happened (the common
case for any lick shorter than the reward cycle) would hang forever. A timer-based lockout needs no
level query, and doubles as the requested minimum time between separately-counted licks.
"""
from pybpodapi.protocol import Bpod, StateMachine

VAR_N_TRIALS = 30
VAR_REWARD_DURATION = 0.1        # seconds Valve1 stays open per lick
VAR_MIN_LICK_DURATION_S = 0.01   # beam must stay broken this long to count -- see module docstring
VAR_LICK_REFRACTORY_S = 0.3      # minimum gap between separately-counted/rewarded licks -- see
                                  # module docstring's "Refractory lockout" note. Separate from
                                  # VAR_MIN_LICK_DURATION_S (that one filters sub-10ms noise WITHIN
                                  # confirming one lick; this one is the minimum gap BETWEEN them).

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
        state_change_conditions={Bpod.Events.Tup: 'Refractory'},
        output_actions=[(Bpod.OutputChannels.Valve, 1)])

    # No Port1In/Port1Out conditions here at all -- see module docstring's "Refractory lockout"
    # note. Any beam activity during this window (the same lick lingering, or its eventual
    # release) is still logged in the trial's raw event list, but cannot re-arm anything until
    # this fixed window elapses and a fresh WaitForLick begins.
    sma.add_state(
        state_name='Refractory',
        state_timer=VAR_LICK_REFRACTORY_S,
        state_change_conditions={Bpod.Events.Tup: 'exit'},
        output_actions=[])

    my_bpod.send_state_machine(sma)
    my_bpod.run_state_machine(sma)

    print("Trial {0}/{1}: beam broken, reward delivered".format(trial + 1, VAR_N_TRIALS), flush=True)

my_bpod.close()

print("Done: {0} trials completed".format(VAR_N_TRIALS), flush=True)
