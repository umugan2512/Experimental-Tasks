# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Same combined Poisson-clicks + wheel-coupled visual-stimulus harness this session has been
building on, now using training_protocol.md's dot stimulus (SS1.2: "one parameter, no internal
structure") in place of the earlier Gabor patch, plus trial_scheduler.py's three offline-validated
lookback mechanisms (choice side-bias handling, performance handling/remedial-easy, disengagement
handling/circuit-breaker) in place of a fixed difficulty round-robin and flat 50/50 side draw.

**Edited in place, not forked** -- unlike every other script this session, which stayed untouched
once hardware-tested. This file specifically had NOT been hardware-validated yet (that's exactly
what HARDWARE_TEST_EXAMPLES.md's Stage A dry run is for), so swapping the visual stimulus here
carries none of the regression risk that ruled out editing e.g. hifi_alternating_easy_gabor_test.py
in place. `gabor_display.py`/`gabor_wheel_test.py`/both `hifi_*_gabor_test/` tasks are untouched by
this change.

**Only the visual stimulus, its two jitter ranges, and the post-choice Consumption/ITI flow have
changed since the original wiring pass** -- cue timing, thresholds, and all four trial-scheduling
integration points below are otherwise exactly as before:

1. **Difficulty draw**: `PerformanceMonitor.next_difficulty()`, gated on whether a remedial-easy
   block is currently active -- `click_train_v2.draw_difficulty()` (the full weighted 6-level
   grid) is still what actually draws a level whenever trial_type=='main'; the scheduler forces
   `AOS` (fully one-sided click train) unconditionally during `remedial_easy`.
2. **Side draw**: the recency-weighted, error-driven debiasing logic (`side_error_fraction`/
   `compute_p_right`/`recency_weighted_right_fraction`/`draw_side_debiased`), all reading from one
   shared `TrialHistory` -- drawn fresh every trial, remedial_easy included; only the stimulus
   difficulty is restricted during remedial_easy, not the side.
3. **Session loop**: a `while` loop gated on `trial_scheduler.should_stop_session(history)`, so the
   circuit-breaker (SS7) can actually end a session early, with `VAR_MAX_TRIALS` as a hard-ceiling
   safety backstop.
4. **VAL registrations + live plots**: `TRIAL_TYPE`/`INCLUDED`/`P_RIGHT_TARGET`/
   `RECENT_RIGHT_FRACTION` registered every trial via `runner.register(...)`, and
   `live_plots_lookback.LookbackBenchPlots` adds three live panels for the three mechanisms.

**What changed in this pass**: `GaborDisplay` -> `DotDisplay` (`_shared/dot_display.py`); the fixed
`VAR_DEG_TO_PX_GAIN` placeholder -> the same geometry-aware gain calibration built for
`dot_wheel_test.py` (hitting `VAR_RIGHT_THRESHOLD_DEG` on the wheel now moves the dot to
`VAR_DOT_EDGE_FRACTION` of the actual screen's half-width, not a guessed px/deg constant); the
onset-jitter range narrowed from the Gabor tests' 0.05-0.35s to training_protocol.md SS1.5's own J1
range (0.1-0.2s); the disappear-jitter range (J2, 0.4-0.9s) is unchanged -- it already matched
SS1.5 exactly. `VAR_DOT_DIAMETER_PX` is an unconfirmed placeholder against SS1.2's 3-4 visual-deg
spec, same flag as `dot_wheel_test.py` (needs monitor size + viewing distance, not measured yet).

**Consumption/ITI restructuring (latest pass)**: `Consumption` (post-reward) and the new
`ErrorConsumption` (post-error, mirrors `Consumption`) are lick-detection windows lumped together
with whichever ITI follows -- a `Port1In` lick ends consumption immediately and moves straight into
that outcome's `ITI`/`IncorrectITI`; with no lick, the same transition still happens once the full
`VAR_CONSUMPTION_WINDOW_S` elapses. Whether the animal licked at all during that window
(`consumption_licked`, registered as `CONSUMPTION_LICKED`) now also feeds
`trial_scheduler.py`'s disengagement/circuit-breaker check, alongside the existing
no-response-to-cue signal.

**Implementation note carried over from the original wiring pass**: `click_train_v2.
draw_difficulty()` calls `rng.choice(levels, p=weights)` -- `numpy.random`-only, so `np.random`
(not the stdlib `random` module) is what gets passed into `PerformanceMonitor.next_difficulty()`/
`draw_side_debiased()` below; `required_hold`'s own `random.uniform(...)` call is unrelated and
still uses the stdlib module.

Run this like any other PyBpod task, via the GUI's Run button -- requires the Bpod board, rotary
encoder module, HiFi module, and a second monitor (for the dot display; falls back to screen 0
with a printed warning if only one is detected) all connected first. See
HARDWARE_TEST_EXAMPLES.md in this folder for staged example sessions (including edge cases) to
actually exercise all three mechanisms on the rig.
"""
import os
import random
import sys
import threading
import time
import traceback

import numpy as np
import matplotlib
# Qt5Agg, not TkAgg: DotDisplay already owns a PyQt5 QApplication/event loop for the second-monitor
# window -- mixing that with a SEPARATE Tk/Tcl event loop (TkAgg's live-plot window) on the same
# thread is what produced a `Fatal Python error: PyEval_RestoreThread: NULL tstate` crash twice,
# reproducibly while dragging the live-plot window (see CLAUDE.md). Qt5Agg makes the live-plot
# window a Qt window too, so matplotlib's first figure creation below reuses DotDisplay's existing
# QApplication instead of spinning up a second, competing native toolkit -- one event loop for the
# whole process.
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

_TASK_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_TASK_DIR, '..', 'poisson_clicks_test'))
sys.path.insert(0, os.path.join(_TASK_DIR, '..', '..', '..', '_shared'))
import click_train_v2 as click_train
import trial_scheduler as ts
from live_plots_lookback import LookbackBenchPlots
from bpod_trial_helpers import TrialRunner, was_visited
import rotary_setup
import hifi_setup
from dot_display import DotDisplay

from pybpodapi.protocol import Bpod, StateMachine

VAR_MAX_TRIALS = 200                # hard-ceiling safety backstop -- the circuit-breaker is what's
                                     # meant to actually end most sessions early. Flagged, easy to
                                     # retune.
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
VAR_ITI = 0.1 #2s
VAR_ABORT_ITI_S = 5.0               # cue-period wheel-abort timeout (unchanged from before)
VAR_INCORRECT_ITI_S = 0.1 #5s           # longer than VAR_ITI -- error timeout for a wrong turn
VAR_STILL_POLL_HZ = 50
VAR_POLL_HZ = 10
VAR_ROTARY_USB_PORT = None

VAR_DOT_SCREEN_INDEX = 1             # second monitor; falls back to 0 with a warning if not found
VAR_DOT_DIAMETER_PX = 60             # UNCONFIRMED against training_protocol.md SS1.2's 3-4 visual-
                                     # deg spec -- guessed pixel value, not derived from it (needs
                                     # monitor size + viewing distance -- see dot_display.py's
                                     # visual_deg_to_px() for the one-line fix once those exist).
VAR_DOT_BACKGROUND_GRAY = 128
VAR_DOT_GRAY = 0                    # full black, per training_protocol.md SS1.2's default (doc also
                                     # floats a sub-maximal-contrast option -- flagged, not built here)
VAR_DOT_EDGE_FRACTION = 0.9          # training_protocol.md SS1.3: place the threshold at ~90% of
                                     # edge azimuth -- gain is derived below from the ACTUAL
                                     # resolved screen width, not a fixed guessed px/deg constant.
VAR_RENDER_HZ = 30                   # lowered from 60 -- halves main-thread Qt-pump frequency
                                     # during the decision period, reducing concurrent GIL pressure
                                     # alongside decision_thread + the nested WHEEL_POS poll thread
                                     # (see CLAUDE.md's PyEval_RestoreThread crash note). A
                                     # probabilistic mitigation, not a guaranteed fix -- 30Hz dot
                                     # repositioning is still visually smooth.

VAR_DOT_ONSET_JITTER_MIN_S = 0.1      # J1 (training_protocol.md SS1.5) -- narrower than the earlier
VAR_DOT_ONSET_JITTER_MAX_S = 0.2      # Gabor tests' 0.05-0.35s, an intentional difference
VAR_DOT_DISAPPEAR_MIN_S = 0.4         # J2 -- same range already used for the Gabor tests, unchanged
VAR_DOT_DISAPPEAR_MAX_S = 0.9

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

dot = DotDisplay(screen_index=VAR_DOT_SCREEN_INDEX, diameter_px=VAR_DOT_DIAMETER_PX,
                  background_gray=VAR_DOT_BACKGROUND_GRAY, dot_gray=VAR_DOT_GRAY)
dot.show()
dot.clear()

# Geometry-aware gain: hitting VAR_RIGHT_THRESHOLD_DEG on the wheel should move the dot to
# VAR_DOT_EDGE_FRACTION of the actual screen's half-width, not a fixed guessed px/wheel-deg
# constant -- same fix applied to dot_wheel_test.py after the earlier placeholder (4.0) turned out
# to barely move the dot at all.
screen_width_px = dot.get_screen_width_px()
dot_gain = (VAR_DOT_EDGE_FRACTION * (screen_width_px / 2.0)) / VAR_RIGHT_THRESHOLD_DEG
dot.set_deg_to_px_gain(dot_gain)
print("Dot gain calibrated to {0:.2f} px/wheel-deg (screen width {1}px, edge fraction {2})".format(
    dot_gain, screen_width_px, VAR_DOT_EDGE_FRACTION), flush=True)

render_interval = 1.0 / VAR_RENDER_HZ

# --- trial-lookback scheduler: one shared TrialHistory for all three mechanisms --------------------

history = ts.TrialHistory()
monitor = ts.PerformanceMonitor(history)


def rewarded_event_for_side(side):
    return right_event if side == 'R' else left_event


# --- live plots ------------------------------------------------------------------------------------

bench_plots = LookbackBenchPlots(
    waveform_duration=click_train.TOTAL_WAVEFORM_DURATION_S,
    click_start_offset=click_train.CLICK_START_OFFSET_S,
    stim_end=click_train.CLICK_START_OFFSET_S + click_train.VAR_STIM_DURATION_S,
    onset_pulse_duration=click_train.VAR_ONSET_PULSE_DURATION_S)

CUE_MONITOR_DURATION_S = click_train.TOTAL_WAVEFORM_DURATION_S + 0.1

# --- trial loop -----------------------------------------------------------------------------------

print("Starting up to {0} trials, stopping early on the disengagement circuit-breaker".format(
    VAR_MAX_TRIALS), flush=True)

trial = 0
while trial < VAR_MAX_TRIALS:
    if ts.should_stop_session(history):
        print("Circuit-breaker: ending session after trial {0} (disengagement detected).".format(
            trial), flush=True)
        break
    trial += 1
    try:
        trial_type = monitor.next_trial_type()
        difficulty = monitor.next_difficulty(np.random, trial_type, click_train.draw_difficulty)
        required_hold = random.uniform(VAR_HOLD_MIN_S, VAR_HOLD_MAX_S)

        rotary.disable_evt_transmission()
        n_breaks = runner.wait_for_held_steady(required_hold, VAR_STEADY_THRESHOLD_DEG,
                                                require_no_lick=False)
        if n_breaks is None:
            print("Bpod stopped running trials (Stop/Kill) -- ending session early after trial "
                  "{0}.".format(trial), flush=True)
            break
        trial_start_t = time.time() - log_python_t0

        with runner.rotary_lock:
            rotary.set_zero_position()
            rotary_setup.set_and_enable_thresholds(rotary, ALL_THRESHOLDS_DEG)
        rotary.enable_evt_transmission()

        runner.register('TRIAL_START', trial_start_t)
        dot.clear()
        dot.pump()

        # e_right/e_left/p_right_target/recent_right_frac directly drive the side draw EVERY trial,
        # remedial_easy included -- the side is never locked; only DIFFICULTY_LEVEL is restricted
        # (to AOS) during remedial_easy, via monitor.next_difficulty() above. draw_side_debiased_capped()
        # additionally forces a break whenever honoring the probabilistic draw would extend a
        # same-side or strict-alternation run past MAX_SAME_SIDE_RUN/MAX_ALTERNATION_RUN.
        e_right = ts.side_error_fraction(history, 'R')
        e_left = ts.side_error_fraction(history, 'L')
        p_right_target = ts.compute_p_right(e_right, e_left)
        recent_right_frac = ts.recency_weighted_right_fraction(history)
        side = ts.draw_side_debiased_capped(np.random, p_right_target, recent_right_frac, history)

        trial_clicks = click_train.generate_trial_clicks(difficulty, side)
        left_wave, right_wave = click_train.build_waveform(trial_clicks, hifi.sampling_rate)
        hifi.load(0, np.array([left_wave, right_wave]))
        hifi.push()

        runner.register('DIFFICULTY_LEVEL', difficulty)
        runner.register('TRIAL_SIDE', side)
        runner.register('REALIZED_DELTA', trial_clicks['realized_delta'])
        runner.register('TRIAL_TYPE', trial_type)
        runner.register('P_RIGHT_TARGET', p_right_target)
        runner.register('RECENT_RIGHT_FRACTION', recent_right_frac)

        print("Trial {0}: held steady -- playing {1} stimulus (side={2}, n_L={3}, n_R={4}, "
              "trial_type={5}, p_right_target={6:.3f})".format(
                  trial, difficulty, side, trial_clicks['n_left'], trial_clicks['n_right'],
                  trial_type, p_right_target), flush=True)

        def _finish_trial(outcome, lick_times_abs, response, reward_time_abs=None):
            """ Shared by the abort branch and the end-of-trial branch below -- both update the
            live plots the same way once a trial's final outcome/lick times are known. """
            bench_plots.add_trial_lookback(p_right_target, recent_right_frac, trial_type, history,
                                            response=response)
            bench_plots.add_trial_licks(trial, outcome, lick_times_abs,
                                         reward_time_abs=reward_time_abs)
            plot_t0 = time.time()
            bench_plots.add_trial(difficulty, side, click_train.nominal_delta(difficulty),
                                   trial_clicks, outcome=outcome)
            plot_render_s = time.time() - plot_t0
            if plot_render_s > 0.5:
                print("NOTE: live-plot redraw took {0:.2f}s this trial".format(plot_render_s),
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
                  "{0}.".format(trial), flush=True)
            break

        cue_visited = my_bpod.session.current_trial.states_durations
        aborted = was_visited(cue_visited, 'WheelAbort')

        cue_events = my_bpod.session.current_trial.get_all_timestamps_by_event()
        cue_lick_times_abs = [cue_send_t + t for t in cue_events.get('Port1In', [])]

        if aborted:
            abort_time_abs = cue_send_t + cue_visited['WheelAbort'][-1][0]
            runner.register('ABORT', abort_time_abs)

            monitor.record_outcome(difficulty, side, trial_type, abort=True, response=None,
                                    correct=None, rt=None)
            runner.register('INCLUDED', False)

            print("Trial {0}: difficulty={1}, held={2:.2f}s (broke {3}x), side={4}, "
                  "Abort (WheelMoved)".format(trial, difficulty, required_hold, n_breaks, side),
                  flush=True)

            _finish_trial('Abort', cue_lick_times_abs, response=None)

            continue

        rewarded_event = rewarded_event_for_side(side)
        unrewarded_event = left_event if rewarded_event == right_event else right_event

        # Both jitters drawn once, up front -- never decided reactively mid-trial.
        dot_onset_delay = random.uniform(VAR_DOT_ONSET_JITTER_MIN_S, VAR_DOT_ONSET_JITTER_MAX_S)
        disappear_delay_s = random.uniform(VAR_DOT_DISAPPEAR_MIN_S, VAR_DOT_DISAPPEAR_MAX_S)

        send_epoch = time.time()
        send_t = send_epoch - log_python_t0

        sma = StateMachine(my_bpod)

        sma.add_state(
            state_name='WheelDotPeriod',
            state_timer=VAR_RESPONSE_TIMEOUT,
            state_change_conditions={
                rewarded_event: 'Reward',
                unrewarded_event: 'ErrorConsumption',
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

        # Consumption is a lick-detection window lumped together with ITI, not a separate fixed
        # duration stacked before it: the first lick ends consumption immediately and moves
        # straight into ITI; with no lick, it still falls into ITI once the full window elapses.
        sma.add_state(
            state_name='Consumption',
            state_timer=VAR_CONSUMPTION_WINDOW_S,
            state_change_conditions={'Port1In': 'ITI', Bpod.Events.Tup: 'ITI'},
            output_actions=[])

        # Same lick-lumping behavior as Consumption, but on the error path, leading into
        # IncorrectITI instead of ITI -- a lick just makes IncorrectITI start earlier; with no
        # lick, IncorrectITI still starts once the full window elapses. Owns the go-cue LED-off
        # action since it's now the first state entered on an incorrect turn.
        sma.add_state(
            state_name='ErrorConsumption',
            state_timer=0.5,
            state_change_conditions={'Port1In': 'IncorrectITI', Bpod.Events.Tup: 'IncorrectITI'},
            output_actions=[(VAR_GO_CUE_LED_CHANNEL, 0)])

        sma.add_state(
            state_name='IncorrectITI',
            state_timer=VAR_INCORRECT_ITI_S,
            state_change_conditions={Bpod.Events.Tup: 'exit'},
            output_actions=[])

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

        # Thread-inverted, same pattern as gabor_wheel_test.py/dot_wheel_test.py: the Bpod call runs
        # in a background thread; this (main) thread renders the dot, since Qt's event loop must
        # run on the thread that owns the QApplication.
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

        dot_visible = False
        frozen = False
        frozen_at = None

        while not decision_done.is_set():
            now = time.time()
            if not dot_visible and (now - send_epoch) >= dot_onset_delay:
                dot_visible = True

            with runner.rotary_lock:
                pos = rotary.current_position()

            if dot_visible and not frozen:
                if pos <= VAR_LEFT_THRESHOLD_DEG or pos >= VAR_RIGHT_THRESHOLD_DEG:
                    frozen = True
                    frozen_at = now
                else:
                    dot.set_position_deg(pos)   # live tracking, pre-threshold

            if frozen and (now - frozen_at) >= disappear_delay_s:
                dot.clear()

            dot.pump()
            time.sleep(render_interval)

        decision_thread.join(timeout=3.0)
        if decision_thread.is_alive():
            print("WARNING: decision-period Bpod thread did not stop within 3s -- continuing "
                  "anyway.", flush=True)

        dot.clear()
        dot.pump()

        if decision_result.get('killed'):
            print("Bpod Kill received -- ending session.", flush=True)
            dot.close()
            hifi.close()
            rotary.close()
            my_bpod.close()
            sys.exit(0)

        if not decision_result.get('ran', False):
            print("Bpod stopped running trials (Stop/Kill) -- ending session early after trial "
                  "{0}.".format(trial), flush=True)
            break

        visited = my_bpod.session.current_trial.states_durations
        choice_events = my_bpod.session.current_trial.get_all_timestamps_by_event()

        if was_visited(visited, 'Reward'):
            outcome_state, outcome, rewarded = 'Reward', 'Reward', True
        elif was_visited(visited, 'ErrorConsumption'):
            outcome_state, outcome, rewarded = 'ErrorConsumption', 'NoReward', False
        elif was_visited(visited, 'NoResponse'):
            outcome_state, outcome, rewarded = 'NoResponse', 'NoResponse', False
        else:
            outcome_state, outcome, rewarded = 'Unknown', 'Unknown', False

        # --- trial-lookback scheduler update ---
        if outcome_state == 'Reward':
            response = side
        elif outcome_state == 'ErrorConsumption':
            response = 'L' if side == 'R' else 'R'
        else:
            response = None
        correct = rewarded if outcome_state in ('Reward', 'ErrorConsumption') else None
        rt = visited[outcome_state][-1][0] if outcome_state in ('Reward', 'ErrorConsumption') else None

        # Whether the animal licked at all during its consumption window (Consumption on a reward,
        # ErrorConsumption on an error) -- None if the trial never reached one (NoResponse/Abort).
        # Fed into the disengagement/circuit-breaker signal in trial_scheduler.py.
        consumption_state_name = {'Reward': 'Consumption',
                                   'ErrorConsumption': 'ErrorConsumption'}.get(outcome_state)
        consumption_licked = None
        if consumption_state_name is not None:
            c_start, c_end = visited[consumption_state_name][-1]
            consumption_licked = any(c_start <= t <= c_end for t in choice_events.get('Port1In', []))

        monitor.record_outcome(difficulty, side, trial_type, abort=False, response=response,
                                correct=correct, rt=rt, consumption_licked=consumption_licked)
        runner.register('INCLUDED', response is not None)
        runner.register('CONSUMPTION_LICKED', consumption_licked)

        # --- Bpod-aligned VAL-row registration -- see hifi_singleside_gabor_test.py's docstring ---
        led_on_t = send_t   # WheelDotPeriod start == LED-on == wheel-turn-becomes-allowed
        dot_onset_t = led_on_t + dot_onset_delay
        runner.register('LED_ON_TIME', led_on_t)
        runner.register('DOT_ONSET_JITTER_S', dot_onset_delay)
        runner.register('DOT_ONSET_TIME', dot_onset_t)

        if outcome_state in ('Reward', 'ErrorConsumption'):
            threshold_crossing_t = send_t + visited[outcome_state][-1][0]
            dot_disappear_t = threshold_crossing_t + disappear_delay_s
            runner.register('THRESHOLD_CROSSING_TIME', threshold_crossing_t)
            runner.register('DOT_DISAPPEAR_JITTER_S', disappear_delay_s)
            runner.register('DOT_DISAPPEAR_TIME', dot_disappear_t)
            runner.register('CORRECT', rewarded)
        elif outcome_state == 'NoResponse':
            runner.register('TIMED_OUT', send_t + visited['NoResponse'][-1][0])

        print("Trial {0}: difficulty={1}, held={2:.2f}s (broke {3}x), side={4}, {5}{6}".format(
            trial, difficulty, required_hold, n_breaks, side, outcome,
            ' -> reward' if rewarded else ''), flush=True)

        choice_lick_times_abs = [send_t + t for t in choice_events.get('Port1In', [])]
        lick_times_abs = sorted(cue_lick_times_abs + choice_lick_times_abs)

        reward_time_abs = None
        if rewarded:
            reward_time_abs = send_t + visited['Reward'][-1][0]

        _finish_trial(outcome, lick_times_abs, response=response, reward_time_abs=reward_time_abs)

    except Exception as err:
        print("Trial {0} FAILED: {1}".format(trial, err), flush=True)
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        raise
else:
    print("Done: reached VAR_MAX_TRIALS ({0}) without the circuit-breaker firing".format(
        VAR_MAX_TRIALS), flush=True)

dot.close()
hifi.close()
rotary.close()
my_bpod.close()

print("Close the plot windows to exit.", flush=True)
plt.ioff()
plt.show()
