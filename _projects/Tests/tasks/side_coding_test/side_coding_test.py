# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Left/right speaker check for the Sanworks Bpod HiFi module (self-announces as HiFi1).

One stereo waveform, loaded once: a low tone (VAR_LEFT_FREQ_HZ) on the left channel starting at
t=0, then a high tone (VAR_RIGHT_FREQ_HZ, two octaves above) on the right channel starting
VAR_LEFT_RIGHT_GAP_S later -- sequenced, not simultaneous, so a listener can independently confirm
each speaker fires at the expected moment. Both tones are baked into one waveform (silence padding
the channel that isn't playing yet) rather than two separate PLAY triggers, so this still only ever
fires PLAY once per trial -- see below for why that matters.

Only ever triggers PLAY once per trial (one output_actions entry, one state). A custom Bpod-module
trigger fired more than once within the same trial has reliably hung run_state_machine() in earlier
testing with this exact module (root cause never found) -- see the repo's own CLAUDE.md. Nothing
here retriggers within a trial.
"""
import time

import numpy as np

from pybpodapi.protocol import Bpod, StateMachine
from pybpod_hifi_module.module import HiFi, HiFiCommandType
from pybpod_hifi_module.module_api import HiFiModule
from pybpod_hifi_module.utils.generate_sound import pure_tone

VAR_N_TRIALS = 20
VAR_LEFT_FREQ_HZ = 1000        # left channel tone, starts at t=0
VAR_RIGHT_FREQ_HZ = 4000       # right channel tone -- exactly 2 octaves above left (1000 * 2**2)
VAR_TONE_DURATION = 0.5        # seconds -- long enough to clearly hear/differentiate both pitches
VAR_LEFT_RIGHT_GAP_S = 1.0     # right tone starts this long after the left tone starts
VAR_RAMP_MS = 10               # cosine onset/offset ramp, avoids a click at tone start/end
VAR_ITI = 2                    # seconds between trials

# --- connect to Bpod, resolve the module by name ---------------------------------------------

my_bpod = Bpod()
print("Connected to Bpod on {0}".format(my_bpod.serial_port), flush=True)

hifi_bpod_module = next(m for m in my_bpod.modules if m.name and m.name.startswith('HiFi'))
print("Resolved module '{0}' on Bpod relay Serial{1}".format(
    hifi_bpod_module.name, hifi_bpod_module.serial_port), flush=True)

# --- connect the HiFi module's own USB link, load the stereo test tone -----------------------

hifi = HiFiModule.discover(exclude_ports=[my_bpod.serial_port])
print("Connected to HiFi module on {0} ({1}Hz sampling)".format(
    hifi.arcom.serial_object.port, hifi.sampling_rate), flush=True)

def _delayed_tone(freq, delay_s):
    """ VAR_TONE_DURATION-long tone on one channel, preceded by delay_s of silence -- used to
    sequence the two channels within a single waveform instead of two separate PLAY triggers. """
    tone = pure_tone(VAR_TONE_DURATION, freq, hifi.sampling_rate, ramp_ms=VAR_RAMP_MS)
    silence = np.zeros(int(round(delay_s * hifi.sampling_rate)))
    return np.concatenate([silence, tone])


left = _delayed_tone(VAR_LEFT_FREQ_HZ, delay_s=0.0)
right = _delayed_tone(VAR_RIGHT_FREQ_HZ, delay_s=VAR_LEFT_RIGHT_GAP_S)

# right is longer (it starts later) -- pad left with trailing silence so both channels match length
n_samples = max(len(left), len(right))
left = np.pad(left, (0, n_samples - len(left)))
right = np.pad(right, (0, n_samples - len(right)))

hifi.load(0, np.array([left, right]))
hifi.push()
VAR_WAVEFORM_DURATION = VAR_LEFT_RIGHT_GAP_S + VAR_TONE_DURATION
print("Loaded sequenced tone: left={0}Hz at t=0, right={1}Hz at t={2}s (each {3}s)".format(
    VAR_LEFT_FREQ_HZ, VAR_RIGHT_FREQ_HZ, VAR_LEFT_RIGHT_GAP_S, VAR_TONE_DURATION), flush=True)

# --- register the PLAY trigger once; only ever fired once per trial (see module docstring) ----

play_msg_id = hifi_bpod_module.load_message(HiFi.get_command(HiFiCommandType.PLAY, sound_index=0))
play_channel = 'Serial{0}'.format(hifi_bpod_module.serial_port)

# --- trial loop ---------------------------------------------------------------------------------

print("Starting {0} trials -- listen for {1}Hz on the left, then {2}Hz on the right {3}s later"
      .format(VAR_N_TRIALS, VAR_LEFT_FREQ_HZ, VAR_RIGHT_FREQ_HZ, VAR_LEFT_RIGHT_GAP_S), flush=True)

for trial in range(VAR_N_TRIALS):
    sma = StateMachine(my_bpod)

    sma.add_state(
        state_name='PlayTone',
        state_timer=VAR_WAVEFORM_DURATION + 0.1,
        state_change_conditions={Bpod.Events.Tup: 'exit'},
        output_actions=[(play_channel, play_msg_id)])

    my_bpod.send_state_machine(sma)
    my_bpod.run_state_machine(sma)

    print("Trial {0}/{1}: played stereo tone".format(trial + 1, VAR_N_TRIALS), flush=True)

    time.sleep(VAR_ITI)

hifi.close()
my_bpod.close()

print("Done: {0} trials completed".format(VAR_N_TRIALS), flush=True)
