# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Placeholder protocol for the lick sensor wired to Port1.

Real lick-triggered task logic hasn't been decided yet, so for now trials
are paced by a free-running global timer (2s period) instead of by licks.
Every Port1In (lick) opens Valve1 for VAR_REWARD_DURATION seconds as a reward.
"""
from pybpodapi.protocol import Bpod, StateMachine

VAR_N_TRIALS = 30            # number of 2s ticks to run before ending the session
VAR_TICK_DURATION = 2        # global timer period, in seconds
VAR_REWARD_DURATION = .1    # seconds Valve1 stays open per lick

my_bpod = Bpod()

for trial in range(VAR_N_TRIALS):
    sma = StateMachine(my_bpod)

    sma.set_global_timer(timer_id=1, timer_duration=VAR_TICK_DURATION)

    sma.add_state(
        state_name='TriggerTimer',
        state_timer=0,
        state_change_conditions={Bpod.Events.Tup: 'WaitForTick'},
        output_actions=[(Bpod.OutputChannels.GlobalTimerTrig, 1)])

    sma.add_state(
        state_name='WaitForTick',
        state_timer=0,
        state_change_conditions={
            Bpod.Events.GlobalTimer1_End: 'Reward'
        },
        output_actions=[])

    sma.add_state(
        state_name='Reward',
        state_timer=VAR_REWARD_DURATION,
        state_change_conditions={Bpod.Events.Tup: 'exit'},
        output_actions=[(Bpod.OutputChannels.Valve, 1)])

    my_bpod.send_state_machine(sma)
    my_bpod.run_state_machine(sma)

    licks = sum(1 for e in my_bpod.session.current_trial.events_occurrences if e.event_name == 'Port1In')
    print("LickTimer trigger {0}/{1} ({2}s elapsed, licks this tick: {3})".format(
        trial + 1, VAR_N_TRIALS, VAR_TICK_DURATION, licks))

my_bpod.close()
