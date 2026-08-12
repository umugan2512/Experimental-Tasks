# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Quick hardware check for the go-cue LED (PWM1, Port 1's built-in LED -- same channel used as the
go-cue in hifi_singleside_test.py/hifi_alternating_easy_test.py): on for 5s, off for 2s, repeated.
"""
import time

from pybpodapi.protocol import Bpod, StateMachine

VAR_N_CYCLES = 10
VAR_LED_CHANNEL = 'PWM1'
VAR_ON_DURATION = 5
VAR_OFF_DURATION = 2

my_bpod = Bpod()
print("Connected to Bpod on {0}".format(my_bpod.serial_port), flush=True)

for cycle in range(VAR_N_CYCLES):
    sma = StateMachine(my_bpod)

    sma.add_state(
        state_name='LedOn',
        state_timer=VAR_ON_DURATION,
        state_change_conditions={Bpod.Events.Tup: 'LedOff'},
        output_actions=[(VAR_LED_CHANNEL, 255)])

    sma.add_state(
        state_name='LedOff',
        state_timer=VAR_OFF_DURATION,
        state_change_conditions={Bpod.Events.Tup: 'exit'},
        output_actions=[(VAR_LED_CHANNEL, 0)])

    my_bpod.send_state_machine(sma)
    if not my_bpod.run_state_machine(sma):
        print("Bpod stopped running trials (Stop/Kill) -- ending early after cycle "
              "{0}/{1}.".format(cycle + 1, VAR_N_CYCLES), flush=True)
        break

    print("Cycle {0}/{1}: LED on {2}s, off {3}s".format(
        cycle + 1, VAR_N_CYCLES, VAR_ON_DURATION, VAR_OFF_DURATION), flush=True)
else:
    print("Done: {0} cycles completed".format(VAR_N_CYCLES), flush=True)

my_bpod.close()
