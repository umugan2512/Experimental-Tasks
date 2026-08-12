# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Same combined Poisson-clicks + wheel-coupled Gabor-patch harness as
hifi_singleside_gabor_test.py, now cycling difficulty trial-by-trial through a fixed round-robin:
AOS -> G65 -> G38 -> repeat -- the same "AOS + two easiest graded levels" convention
hifi_alternating_easy_test.py used (there: AOS -> E15 -> E11, the two highest-gamma non-AOS levels
in the old grid; here: G65/G38, the two highest-gamma non-AOS levels in click_train_v2.py's new
grid, since higher gamma = larger rate contrast = easier under the new linear-contrast formula).

Forked from hifi_alternating_easy_test.py (kept byte-identical/untouched) the same way
hifi_singleside_gabor_test.py was forked from hifi_singleside_test.py -- see that file's own
docstring for the full rationale behind every change from the original (IBL thresholds, shortened
cue/delay timing, lick-gating removal, the Gabor choice-phase machine, jitter semantics, and
Bpod-aligned VAL-row registration). Nothing here differs from that file's design beyond
per-trial difficulty cycling (`VAR_DIFFICULTY_CYCLE`, `rewarded_event_for_side()`'s generalization)
-- exactly mirroring how the two original (non-Gabor) scripts already differed from each other.

Run this like any other PyBpod task, via the GUI's Run button -- requires the Bpod board, rotary
encoder module, HiFi module, and a second monitor (for the Gabor display; falls back to screen 0
with a printed warning if only one is detected) all connected first.
"""
import os
import random
import sys
import threading
import time
import traceback

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

_TASK_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_TASK_DIR, '..', 'poisson_clicks_test'))
sys.path.insert(0, os.path.join(_TASK_DIR, '..', '..', '..', '_shared'))
import click_train_v2 as click_train
from live_plots import LiveBenchPlots
from bpod_trial_helpers import TrialRunner, was_visited
import rotary_setup
import hifi_setup
from gabor_display import GaborDisplay

from pybpodapi.protocol import Bpod, StateMachine

VAR_N_TRIALS = 30
VAR_HOLD_MIN_S = 0.2
VAR_HOLD_MAX_S = 0.6
VAR_STEADY_THRESHOLD_DEG = 4        # IBL convention
VAR_LEFT_THRESHOLD_DEG = -35        # IBL convention
VAR_RIGHT_THRESHOLD_DEG = 35        # IBL convention
VAR_CUE_ABORT_THRESHOLD_DEG = 2.5 * VAR_STEADY_THRESHOLD_DEG   # 10deg -- looser than quiescence,
                                                                 # still much tighter than a choice turn
VAR_RESPONSE_TIMEOUT = 5
VAR_REWARD_DURATION = 0.1
VAR_CONSUMPTION_WINDOW_S = 3.0
VAR_ITI = 2
VAR_ABORT_ITI_S = 5.0               # cue-period wheel-abort timeout (unchanged from before)
VAR_INCORRECT_ITI_S = 4.0           # longer than VAR_ITI -- error timeout for a wrong turn
VAR_STILL_POLL_HZ = 50
VAR_POLL_HZ = 10
VAR_ROTARY_USB_PORT = None
VAR_DIFFICULTY_CYCLE = ['AOS', 'G65', 'G38']   # fixed round-robin, not random -- see docstring

VAR_GABOR_SCREEN_INDEX = 1          # second monitor; falls back to 0 with a warning if not found
VAR_GABOR_SIZE_PX = 400
VAR_GABOR_SPATIAL_FREQ_CPP = 6
VAR_GABOR_SIGMA_PX = 80
VAR_GABOR_CONTRAST = 1.0
VAR_GABOR_BACKGROUND_GRAY = 128
VAR_DEG_TO_PX_GAIN = 4.0            # uncalibrated placeholder -- see gabor_display.py docstring
VAR_RENDER_HZ = 60

VAR_GABOR_ONSET_JITTER_MIN_S = 0.05   # patch onset, positive-only, after LED/WheelGaborPeriod start
VAR_GABOR_ONSET_JITTER_MAX_S = 0.35
VAR_GABOR_DISAPPEAR_MIN_S = 0.4       # threshold crossing -> patch clears, positive-only
VAR_GABOR_DISAPPEAR_MAX_S = 0.9

# 6 thresholds -- the documented max for RotaryEncoderModule.set_thresholds() -- index order
# determines event numbering (RotaryEncoder1_1..._6); rotary_setup.set_and_enable_thresholds()
# returns the corresponding event names below instead of this file re-deriving them by hand.
ALL_THRESHOLDS_DEG = [-VAR_STEADY_THRESHOLD_DEG, VAR_STEADY_THRESHOLD_DEG,
                       VAR_LEFT_THRESHOLD_DEG, VAR_RIGHT_THRESHOLD_DEG,
                       -VAR_CUE_ABORT_THRESHOLD_DEG, VAR_CUE_ABORT_THRESHOLD_DEG]

# --- connect to Bpod, resolve modules -------------------------------------------------------------

my_bpod = Bpod()
print("Connected to Bpod on {0}".format(my_bpod.serial_port), flush=True)

rotary, rotary_bpod_module = rotary_setup.connect_rotary(my_bpod, usb_port=VAR_ROTARY_USB_PORT)
reset_positions_trigger_id, rotary_channel = rotary_setup.build_reset_trigger(rotary_bpod_module)

event_names = rotary_setup.set_and_enable_thresholds(rotary, ALL_THRESHOLDS_DEG)
rotary.enable_evt_transmission()
left_event, right_event = event_names[2], event_names[3]
wheel_abort_event_neg, wheel_abort_event_pos = event_names[4], event_names[5]

VAR_GO_CUE_LED_CHANNEL = 'PWM1'   # Port 1's built-in LED, confirmed as the go-cue LED

hifi = hifi_setup.connect_hifi(my_bpod)
hifi_stop_msg_id, hifi_channel = hifi_setup.build_stop_trigger(my_bpod)

my_bpod.register_value('LEFT_THRESHOLD_DEG', VAR_LEFT_THRESHOLD_DEG)
my_bpod.register_value('RIGHT_THRESHOLD_DEG', VAR_RIGHT_THRESHOLD_DEG)

log_python_t0 = time.time()
runner = TrialRunner(my_bpod, rotary, log_python_t0, still_poll_hz=VAR_STILL_POLL_HZ,
                      poll_hz=VAR_POLL_HZ)

gabor = GaborDisplay(screen_index=VAR_GABOR_SCREEN_INDEX, size_px=VAR_GABOR_SIZE_PX,
                      spatial_freq_cpp=VAR_GABOR_SPATIAL_FREQ_CPP, sigma_px=VAR_GABOR_SIGMA_PX,
                      contrast=VAR_GABOR_CONTRAST, background_gray=VAR_GABOR_BACKGROUND_GRAY,
                      deg_to_px_gain=VAR_DEG_TO_PX_GAIN)
gabor.show()
gabor.clear()

render_interval = 1.0 / VAR_RENDER_HZ


def rewarded_event_for_side(side):
    """ generate_trial_clicks()'s `side` argument is already "the higher-evidence side,"
    regardless of difficulty level, so this mapping needs no per-level branching even though
    G65/G38 have both sides active (unlike AOS). """
    return right_event if side == 'R' else left_event


# --- live plots ------------------------------------------------------------------------------------

bench_plots = LiveBenchPlots(
    waveform_duration=click_train.TOTAL_WAVEFORM_DURATION_S,
    click_start_offset=click_train.CLICK_START_OFFSET_S,
    stim_end=click_train.CLICK_START_OFFSET_S + click_train.VAR_STIM_DURATION_S,
    onset_pulse_duration=click_train.VAR_ONSET_PULSE_DURATION_S)

CUE_MONITOR_DURATION_S = click_train.TOTAL_WAVEFORM_DURATION_S + 0.1

# --- trial loop -----------------------------------------------------------------------------------

print("Starting {0} trials, cycling difficulty {1}".format(VAR_N_TRIALS, VAR_DIFFICULTY_CYCLE),
      flush=True)

for trial in range(VAR_N_TRIALS):
    try:
        difficulty = VAR_DIFFICULTY_CYCLE[trial % len(VAR_DIFFICULTY_CYCLE)]
        required_hold = random.uniform(VAR_HOLD_MIN_S, VAR_HOLD_MAX_S)

        rotary.disable_evt_transmission()
        n_breaks = runner.wait_for_held_steady(required_hold, VAR_STEADY_THRESHOLD_DEG,
                                                require_no_lick=False)
        if n_breaks is None:
            print("Bpod stopped running trials (Stop/Kill) -- ending session early after trial "
                  "{0}/{1}.".format(trial + 1, VAR_N_TRIALS), flush=True)
            break
        trial_start_t = time.time() - log_python_t0

        with runner.rotary_lock:
            rotary.set_zero_position()
            rotary_setup.set_and_enable_thresholds(rotary, ALL_THRESHOLDS_DEG)
        rotary.enable_evt_transmission()

        runner.register('TRIAL_START', trial_start_t)
        gabor.clear()
        gabor.pump()

        side = click_train.draw_side()
        trial_clicks = click_train.generate_trial_clicks(difficulty, side)
        left_wave, right_wave = click_train.build_waveform(trial_clicks, hifi.sampling_rate)
        hifi.load(0, np.array([left_wave, right_wave]))
        hifi.push()

        runner.register('DIFFICULTY_LEVEL', difficulty)
        runner.register('TRIAL_SIDE', side)
        runner.register('REALIZED_DELTA', trial_clicks['realized_delta'])

        print("Trial {0}: held steady -- playing {1} stimulus (side={2}, n_L={3}, n_R={4})".format(
            trial + 1, difficulty, side, trial_clicks['n_left'], trial_clicks['n_right']),
            flush=True)

        hifi.play(0)

        cue_send_t = time.time() - log_python_t0

        cue_sma = StateMachine(my_bpod)

        cue_sma.add_state(
            state_name='CuePeriod',
            state_timer=CUE_MONITOR_DURATION_S,
            state_change_conditions={
                wheel_abort_event_neg: 'WheelAbort',
                wheel_abort_event_pos: 'WheelAbort',
                Bpod.Events.Tup: 'CueComplete',
            },
            output_actions=[])

        cue_sma.add_state(
            state_name='WheelAbort',
            state_timer=0,
            state_change_conditions={Bpod.Events.Tup: 'AbortITI'},
            output_actions=[(hifi_channel, hifi_stop_msg_id)])

        cue_sma.add_state(
            state_name='AbortITI',
            state_timer=VAR_ABORT_ITI_S,
            state_change_conditions={Bpod.Events.Tup: 'exit'},
            output_actions=[])

        cue_sma.add_state(
            state_name='CueComplete',
            state_timer=0,
            state_change_conditions={Bpod.Events.Tup: 'exit'},
            output_actions=[(hifi_channel, hifi_stop_msg_id)])

        if not runner.run_trial_state_machine(cue_sma):
            print("Bpod stopped running trials (Stop/Kill) -- ending session early after trial "
                  "{0}/{1}.".format(trial + 1, VAR_N_TRIALS), flush=True)
            break

        cue_visited = my_bpod.session.current_trial.states_durations
        aborted = was_visited(cue_visited, 'WheelAbort')

        cue_events = my_bpod.session.current_trial.get_all_timestamps_by_event()
        cue_lick_times_abs = [cue_send_t + t for t in cue_events.get('Port1In', [])]

        if aborted:
            abort_time_abs = cue_send_t + cue_visited['WheelAbort'][-1][0]
            runner.register('ABORT', abort_time_abs)
            print("Trial {0}/{1}: difficulty={2}, held={3:.2f}s (broke {4}x), side={5}, "
                  "Abort (WheelMoved)".format(
                      trial + 1, VAR_N_TRIALS, difficulty, required_hold, n_breaks, side),
                  flush=True)

            bench_plots.add_trial_licks(trial + 1, 'Abort', cue_lick_times_abs)

            plot_t0 = time.time()
            bench_plots.add_trial(difficulty, side, click_train.nominal_delta(difficulty),
                                   trial_clicks, outcome='Abort')
            plot_render_s = time.time() - plot_t0
            if plot_render_s > 0.5:
                print("NOTE: live-plot redraw took {0:.2f}s this trial".format(plot_render_s),
                      flush=True)

            continue

        rewarded_event = rewarded_event_for_side(side)
        unrewarded_event = left_event if rewarded_event == right_event else right_event

        # Both jitters drawn once, up front -- never decided reactively mid-trial.
        gabor_onset_delay = random.uniform(VAR_GABOR_ONSET_JITTER_MIN_S, VAR_GABOR_ONSET_JITTER_MAX_S)
        disappear_delay_s = random.uniform(VAR_GABOR_DISAPPEAR_MIN_S, VAR_GABOR_DISAPPEAR_MAX_S)

        send_epoch = time.time()
        send_t = send_epoch - log_python_t0

        sma = StateMachine(my_bpod)

        sma.add_state(
            state_name='WheelGaborPeriod',
            state_timer=VAR_RESPONSE_TIMEOUT,
            state_change_conditions={
                rewarded_event: 'Reward',
                unrewarded_event: 'IncorrectITI',
                Bpod.Events.Tup: 'NoResponse',
            },
            output_actions=[(rotary_channel, reset_positions_trigger_id),
                             (VAR_GO_CUE_LED_CHANNEL, 255)])

        # Reward fires immediately on threshold crossing -- no jitter, same as every other reward
        # state in this project.
        sma.add_state(
            state_name='Reward',
            state_timer=VAR_REWARD_DURATION,
            state_change_conditions={Bpod.Events.Tup: 'Consumption'},
            output_actions=[(Bpod.OutputChannels.Valve, 1), (VAR_GO_CUE_LED_CHANNEL, 0)])

        sma.add_state(
            state_name='Consumption',
            state_timer=VAR_CONSUMPTION_WINDOW_S,
            state_change_conditions={Bpod.Events.Tup: 'ITI'},
            output_actions=[])

        sma.add_state(
            state_name='IncorrectITI',
            state_timer=VAR_INCORRECT_ITI_S,
            state_change_conditions={Bpod.Events.Tup: 'exit'},
            output_actions=[(VAR_GO_CUE_LED_CHANNEL, 0)])

        sma.add_state(
            state_name='NoResponse',
            state_timer=0,
            state_change_conditions={Bpod.Events.Tup: 'ITI'},
            output_actions=[(VAR_GO_CUE_LED_CHANNEL, 0)])

        sma.add_state(
            state_name='ITI',
            state_timer=VAR_ITI,
            state_change_conditions={Bpod.Events.Tup: 'exit'},
            output_actions=[])

        # Thread-inverted, same pattern as gabor_wheel_test.py: the Bpod call runs in a background
        # thread; this (main) thread renders the Gabor patch, since Qt's event loop must run on the
        # thread that owns the QApplication.
        decision_result = {}
        decision_done = threading.Event()

        def _run_decision():
            try:
                decision_result['ran'] = runner.run_trial_state_machine(sma)
            except SystemExit:
                # Kill's own handling calls exit(0) from inside run_state_machine() -- raised on
                # this background thread, it would otherwise only kill this thread silently,
                # leaving the render loop below spinning forever. Flag it so the main thread can
                # re-raise SystemExit itself once it notices (see after the render loop).
                decision_result['killed'] = True
            finally:
                decision_done.set()

        decision_thread = threading.Thread(target=_run_decision, daemon=True)
        decision_thread.start()

        gabor_visible = False
        frozen = False
        frozen_at = None

        while not decision_done.is_set():
            now = time.time()
            if not gabor_visible and (now - send_epoch) >= gabor_onset_delay:
                gabor_visible = True

            with runner.rotary_lock:
                pos = rotary.current_position()

            if gabor_visible and not frozen:
                if pos <= VAR_LEFT_THRESHOLD_DEG or pos >= VAR_RIGHT_THRESHOLD_DEG:
                    frozen = True
                    frozen_at = now
                else:
                    gabor.set_position_deg(pos)   # live tracking, pre-threshold

            if frozen and (now - frozen_at) >= disappear_delay_s:
                gabor.clear()

            gabor.pump()
            time.sleep(render_interval)

        decision_thread.join(timeout=3.0)
        if decision_thread.is_alive():
            print("WARNING: decision-period Bpod thread did not stop within 3s -- continuing "
                  "anyway.", flush=True)

        gabor.clear()
        gabor.pump()

        if decision_result.get('killed'):
            print("Bpod Kill received -- ending session.", flush=True)
            gabor.close()
            hifi.close()
            rotary.close()
            my_bpod.close()
            sys.exit(0)

        if not decision_result.get('ran', False):
            print("Bpod stopped running trials (Stop/Kill) -- ending session early after trial "
                  "{0}/{1}.".format(trial + 1, VAR_N_TRIALS), flush=True)
            break

        visited = my_bpod.session.current_trial.states_durations
        if was_visited(visited, 'Reward'):
            outcome_state, outcome, rewarded = 'Reward', 'Reward', True
        elif was_visited(visited, 'IncorrectITI'):
            outcome_state, outcome, rewarded = 'IncorrectITI', 'NoReward', False
        elif was_visited(visited, 'NoResponse'):
            outcome_state, outcome, rewarded = 'NoResponse', 'NoResponse', False
        else:
            outcome_state, outcome, rewarded = 'Unknown', 'Unknown', False

        # --- Bpod-aligned VAL-row registration -- see hifi_singleside_gabor_test.py's docstring ---
        led_on_t = send_t   # WheelGaborPeriod start == LED-on == wheel-turn-becomes-allowed
        gabor_onset_t = led_on_t + gabor_onset_delay
        runner.register('LED_ON_TIME', led_on_t)
        runner.register('GABOR_ONSET_JITTER_S', gabor_onset_delay)
        runner.register('GABOR_ONSET_TIME', gabor_onset_t)

        if outcome_state in ('Reward', 'IncorrectITI'):
            threshold_crossing_t = send_t + visited[outcome_state][-1][0]
            gabor_disappear_t = threshold_crossing_t + disappear_delay_s
            runner.register('THRESHOLD_CROSSING_TIME', threshold_crossing_t)
            runner.register('GABOR_DISAPPEAR_JITTER_S', disappear_delay_s)
            runner.register('GABOR_DISAPPEAR_TIME', gabor_disappear_t)
            runner.register('CORRECT', rewarded)
        elif outcome_state == 'NoResponse':
            runner.register('TIMED_OUT', send_t + visited['NoResponse'][-1][0])

        print("Trial {0}/{1}: difficulty={2}, held={3:.2f}s (broke {4}x), side={5}, {6}{7}".format(
            trial + 1, VAR_N_TRIALS, difficulty, required_hold, n_breaks, side, outcome,
            ' -> reward' if rewarded else ''), flush=True)

        choice_events = my_bpod.session.current_trial.get_all_timestamps_by_event()
        choice_lick_times_abs = [send_t + t for t in choice_events.get('Port1In', [])]
        lick_times_abs = sorted(cue_lick_times_abs + choice_lick_times_abs)

        reward_time_abs = None
        if rewarded:
            reward_time_abs = send_t + visited['Reward'][-1][0]

        bench_plots.add_trial_licks(trial + 1, outcome, lick_times_abs,
                                     reward_time_abs=reward_time_abs)

        plot_t0 = time.time()
        bench_plots.add_trial(difficulty, side, click_train.nominal_delta(difficulty),
                               trial_clicks, outcome=outcome)
        plot_render_s = time.time() - plot_t0
        if plot_render_s > 0.5:
            print("NOTE: live-plot redraw took {0:.2f}s this trial".format(plot_render_s), flush=True)

    except Exception as err:
        print("Trial {0}/{1} FAILED: {2}".format(trial + 1, VAR_N_TRIALS, err), flush=True)
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        raise
else:
    print("Done: {0} trials completed".format(VAR_N_TRIALS), flush=True)

gabor.close()
hifi.close()
rotary.close()
my_bpod.close()

print("Close the plot windows to exit.", flush=True)
plt.ioff()
plt.show()
