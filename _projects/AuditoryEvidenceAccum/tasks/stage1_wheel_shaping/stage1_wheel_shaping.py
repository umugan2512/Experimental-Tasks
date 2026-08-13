# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""
training_protocol.md Part 4, Stage 1 -- "Wheel unlocked, spout close, small movements rewarded".

Aim: teach that the wheel moves the dot and that moving it earns reward. No auditory stimulus, no
correct side, no error condition. Quiescence 100ms fixed; LED on, dot appears centred after a J1
jitter, coupled to the wheel at ~2x final gain; ANY movement past the current reward threshold in
EITHER direction rewards; dot freezes through the J2 jitter, then off; short (~0.5s) ITI. Gain
drops, and the reward threshold GROWS, once per qualifying (>200 trial) session (rates not
specified by the doc -- flagged/tunable, see _wheel_shaping_shared/staircase.py's
decay_gain()/grow_stage1_threshold()) -- both fixed steps ACROSS sessions, not a within-session,
per-trial staircase (that's Stage 2's mechanism, `ThresholdStaircase`, deliberately not reused
here since Stage 1 isn't tracking moment-to-moment performance the way Stage 2 is). The threshold
starts small (~5% of final, ~1.75deg) and grows toward Stage 2's own 20% starting point -- both
stages share the same persisted `threshold_fraction`, just updated via different mechanisms.
Advance at >200 trials/session on 2 CONSECUTIVE sessions -- auto-tracked here via a persisted state
file (see session_state.py), printed as an explicit ADVANCE-READY line at session end.

Trial-loop/module-connection plumbing reused as-is from _shared/ (bpod_trial_helpers.py's
TrialRunner, dot_display.py's DotDisplay, rotary_setup.py) -- the same proven, hardware-tested
pieces every other task in this codebase already uses. No HiFi/audio setup at all (Stage 1 has no
sound). The dot IS coupled to the wheel from this stage onward (per the doc's own J1/J2 jitter
language, "coupled to the wheel at high training gain") -- there is no wheel-only, dot-less phase
anywhere in the curriculum; visual-motor coupling starts on day one.

Bpod-state/render-loop split mirrors full_protocol_lookback_test.py's proven WheelDotPeriod design:
Bpod only needs to know WHETHER/WHEN a threshold was crossed (WheelPeriod -> Reward/NoMovement);
the dot's own appear-after-J1 / freeze-on-crossing / off-after-J2 behavior lives entirely in the
Python render loop (wall-clock + wheel-position driven, independent of which Bpod state is active),
run on the main thread (owns the Qt QApplication) while the Bpod call itself runs on a background
thread -- same thread-inversion pattern, same Kill/SystemExit handling.

Hardware-timestamp gap (documented, not solved here): training_protocol.md SS1.7 wants
photodiode-based dot-onset timestamps (commanded time is quantised to ~17ms at 60Hz) -- no
photodiode is confirmed present anywhere in this project's hardware notes, so this task uses
software/commanded timestamps like every other task in this codebase. Not a blocker at Stage 1,
which doesn't need frame-accurate timing the way the later psychometric-fitting stages will.
"""
import os
import sys
import threading
import time
import traceback

import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')  # DotDisplay owns the QApplication -- see CLAUDE.md's PyEval_RestoreThread
                           # crash note for why this must not be TkAgg.
import matplotlib.pyplot as plt

_TASK_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_TASK_DIR, '..', '_wheel_shaping_shared'))
sys.path.insert(0, os.path.join(_TASK_DIR, '..', '..', '..', '_shared'))
import staircase
import session_state
import session_struct_export
from wheel_shaping_plots import WheelShapingPlots
from bpod_trial_helpers import TrialRunner, was_visited
import rotary_setup
from dot_display import DotDisplay

from pybpodapi.protocol import Bpod, StateMachine


def _export_session_struct(csv_path):
    """ Every VAR_* constant this run used, harvested automatically -- stays complete as new
    parameters get added later, no hand-maintained list to fall out of sync. Wrapped in try/except
    so an export hiccup (e.g. a scipy/disk issue) never blocks session teardown -- the animal's run
    is already fully logged in the CSV regardless of whether this convenience export succeeds. """
    task_params = {k: v for k, v in globals().items() if k.startswith('VAR_')}
    try:
        mat_path, json_path = session_struct_export.export_session_struct(csv_path, task_params)
        print("Session struct exported: {0} / {1}".format(mat_path, json_path), flush=True)
    except Exception as err:
        print("WARNING: session struct export failed: {0}".format(err), flush=True)

# --- who this session is for -----------------------------------------------------------------------

VAR_SUBJECT_ID = 'REPLACE_ME'   # edit before every session -- see session_state.py's module
                                 # docstring for why this is an explicit constant, not auto-derived
VAR_PROJECT_DIR = os.path.abspath(os.path.join(_TASK_DIR, '..', '..'))

# --- stage parameters (training_protocol.md Part 4, Stage 1's table) --------------------------------

VAR_MAX_TRIALS = 400                # hard-ceiling safety backstop, well above the 200-trial goal
VAR_QUIESCENCE_S = 0.1              # fixed 100ms -- doc: "trivially short, but present from the
                                     # very first operant trial"
VAR_STEADY_THRESHOLD_DEG = 4        # IBL convention, same as every other task -- the HOLD-STILL
                                     # deadzone, distinct from the (much larger) reward threshold
VAR_RESPONSE_TIMEOUT_S = 10.0       # not specified by the doc for Stage 1 (no error concept) --
                                     # flagged/tunable; generous so it essentially never binds
                                     # under normal exploration, but still bounds an unresponsive
                                     # trial so a Bpod machine can't wait forever
VAR_ITI_S = 0.5                     # doc: "short (~0.5s)"

VAR_THRESHOLD_FINAL_DEG = 35        # matches this project's established final-task convention
                                     # (VAR_RIGHT_THRESHOLD_DEG elsewhere)
VAR_THRESHOLD_STARTING_FRACTION = 0.05   # only used the very first time this subject runs Stage 1
                                     # (StageState default) -- doc: "~15-25% of final" was the OLD
                                     # fixed-value design; per later instruction, the threshold now
                                     # GROWS across sessions instead (see staircase.
                                     # grow_stage1_threshold()), starting small (~1.75deg) and
                                     # reaching Stage 2's own 20% starting point by the time Stage 1
                                     # ends. Fixed for the whole SESSION (grows only between
                                     # sessions, not per-trial) -- read from persisted state below.

VAR_TRIAL_COUNT_ADVANCE = 200       # doc: ">200 completed trials/session on two consecutive
                                     # sessions"

VAR_REWARD_DURATION = 0.1
VAR_REWARD_UL = 4.0                 # uncalibrated placeholder -- no valve uL calibration exists
                                     # anywhere in this codebase yet; update once real calibration
                                     # data ties valve-open time to delivered volume (same
                                     # "uncalibrated placeholder" convention as VAR_DEG_TO_PX_GAIN
                                     # elsewhere)

VAR_DOT_ONSET_JITTER_MIN_S = 0.1    # J1 (training_protocol.md SS1.5) -- runs at its FINAL value
VAR_DOT_ONSET_JITTER_MAX_S = 0.2    # from Stage 1 onward and never changes (doc: "the LED->dot and
VAR_DOT_DISAPPEAR_MIN_S = 0.4       # crossing->offset relationships should never change during
VAR_DOT_DISAPPEAR_MAX_S = 0.9       # training")

VAR_DOT_SCREEN_INDEX = 1            # second monitor; falls back to 0 with a printed warning
VAR_DOT_DIAMETER_PX = 60            # UNCONFIRMED against training_protocol.md SS1.2's 3-4 visual-
                                     # deg spec -- same flag as every other dot-stimulus task in
                                     # this codebase (needs monitor size + viewing distance)
VAR_DOT_BACKGROUND_GRAY = 128
VAR_DOT_GRAY = 0
VAR_DOT_EDGE_FRACTION = 0.9         # SS1.3: place the threshold at ~90% of edge azimuth
VAR_GAIN_INITIAL_MULT = 2.0         # doc: "~2x final" -- decays via staircase.decay_gain()
VAR_RENDER_HZ = 30

VAR_ROTARY_USB_PORT = None
VAR_STILL_POLL_HZ = 50
VAR_POLL_HZ = 10

VAR_GO_CUE_LED_CHANNEL = 'PWM1'     # Port 1's built-in LED, same convention as every other task

# --- persisted cross-session state -------------------------------------------------------------------

state = session_state.StageState(VAR_PROJECT_DIR, VAR_SUBJECT_ID, defaults={
    'gain_mult': VAR_GAIN_INITIAL_MULT,
    'threshold_fraction': VAR_THRESHOLD_STARTING_FRACTION,   # shared with Stage 2 -- see
                                                              # staircase.grow_stage1_threshold()
    'sessions_trial_count_history': [],   # list of bools: did each past session clear the
                                           # advance-trial-count gate? (used for the
                                           # two-CONSECUTIVE-sessions criterion)
})
cur_gain_mult = state.get('gain_mult')
cur_threshold_fraction = state.get('threshold_fraction')
cur_threshold_deg = cur_threshold_fraction * VAR_THRESHOLD_FINAL_DEG

# --- connect to Bpod, resolve modules ----------------------------------------------------------------

my_bpod = Bpod()
print("Connected to Bpod on {0}".format(my_bpod.serial_port), flush=True)

rotary, rotary_bpod_module = rotary_setup.connect_rotary(my_bpod, usb_port=VAR_ROTARY_USB_PORT)
reset_positions_trigger_id, rotary_channel = rotary_setup.build_reset_trigger(rotary_bpod_module)

event_names = rotary_setup.set_and_enable_thresholds(
    rotary, [-cur_threshold_deg, cur_threshold_deg])
rotary.enable_evt_transmission()
neg_event, pos_event = event_names[0], event_names[1]

my_bpod.register_value('THRESHOLD_DEG', cur_threshold_deg)
my_bpod.register_value('REWARD_UL', VAR_REWARD_UL)

log_python_t0 = time.time()
runner = TrialRunner(my_bpod, rotary, log_python_t0, still_poll_hz=VAR_STILL_POLL_HZ,
                      poll_hz=VAR_POLL_HZ)

dot = DotDisplay(screen_index=VAR_DOT_SCREEN_INDEX, diameter_px=VAR_DOT_DIAMETER_PX,
                  background_gray=VAR_DOT_BACKGROUND_GRAY, dot_gray=VAR_DOT_GRAY)
dot.show()
dot.clear()

# Geometry-aware FINAL gain (same derivation as full_protocol_lookback_test.py) -- Stage 1's actual
# gain is this, multiplied by the current (persisted, decaying) gain multiplier.
screen_width_px = dot.get_screen_width_px()
final_dot_gain = (VAR_DOT_EDGE_FRACTION * (screen_width_px / 2.0)) / VAR_THRESHOLD_FINAL_DEG
dot.set_deg_to_px_gain(final_dot_gain * cur_gain_mult)
print("Dot gain calibrated to {0:.2f} px/wheel-deg (final={1:.2f}, mult={2:.2f}x, screen width "
      "{3}px)".format(final_dot_gain * cur_gain_mult, final_dot_gain, cur_gain_mult,
                       screen_width_px), flush=True)

render_interval = 1.0 / VAR_RENDER_HZ

bench_plots = WheelShapingPlots(
    stage=1, threshold_final_deg=VAR_THRESHOLD_FINAL_DEG,
    prev_session_values={'threshold_deg': cur_threshold_deg, 'gain_mult': cur_gain_mult})

# --- trial loop -----------------------------------------------------------------------------------

print("Starting up to {0} trials, targeting {1} for advancement".format(
    VAR_MAX_TRIALS, VAR_TRIAL_COUNT_ADVANCE), flush=True)

session_trial_count = 0

for trial in range(1, VAR_MAX_TRIALS + 1):
    try:
        rotary.disable_evt_transmission()
        n_breaks = runner.wait_for_held_steady(VAR_QUIESCENCE_S, VAR_STEADY_THRESHOLD_DEG,
                                                require_no_lick=False)
        if n_breaks is None:
            print("Bpod stopped running trials (Stop/Kill) -- ending session early after trial "
                  "{0}.".format(trial), flush=True)
            break
        trial_start_t = time.time() - log_python_t0

        with runner.rotary_lock:
            rotary.set_zero_position()
            rotary_setup.set_and_enable_thresholds(rotary, [-cur_threshold_deg, cur_threshold_deg])
        rotary.enable_evt_transmission()

        runner.register('TRIAL_START', trial_start_t)
        runner.register('QUIESCENCE_BREAKS', n_breaks)
        dot.clear()
        dot.pump()

        dot_onset_delay = np.random.uniform(VAR_DOT_ONSET_JITTER_MIN_S, VAR_DOT_ONSET_JITTER_MAX_S)
        disappear_delay_s = np.random.uniform(VAR_DOT_DISAPPEAR_MIN_S, VAR_DOT_DISAPPEAR_MAX_S)

        send_epoch = time.time()
        send_t = send_epoch - log_python_t0

        sma = StateMachine(my_bpod)

        sma.add_state(
            state_name='WheelPeriod',
            state_timer=VAR_RESPONSE_TIMEOUT_S,
            state_change_conditions={
                neg_event: 'Reward',
                pos_event: 'Reward',
                Bpod.Events.Tup: 'NoMovement',
            },
            output_actions=[(rotary_channel, reset_positions_trigger_id),
                             (VAR_GO_CUE_LED_CHANNEL, 255)])

        sma.add_state(
            state_name='Reward',
            state_timer=VAR_REWARD_DURATION,
            state_change_conditions={Bpod.Events.Tup: 'ITI'},
            output_actions=[(Bpod.OutputChannels.Valve, 1), (VAR_GO_CUE_LED_CHANNEL, 0)])

        sma.add_state(
            state_name='NoMovement',
            state_timer=0,
            state_change_conditions={Bpod.Events.Tup: 'ITI'},
            output_actions=[(VAR_GO_CUE_LED_CHANNEL, 0)])

        sma.add_state(
            state_name='ITI',
            state_timer=VAR_ITI_S,
            state_change_conditions={Bpod.Events.Tup: 'exit'},
            output_actions=[])

        # Thread-inverted, same pattern as full_protocol_lookback_test.py: the Bpod call runs in a
        # background thread; this (main) thread renders the dot, since Qt's event loop must run on
        # the thread that owns the QApplication.
        decision_result = {}
        decision_done = threading.Event()

        def _run_decision():
            try:
                decision_result['ran'] = runner.run_trial_state_machine(sma)
            except SystemExit:
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
                if pos <= -cur_threshold_deg or pos >= cur_threshold_deg:
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
            rotary.close()
            csv_path = my_bpod.session._path   # grab before close() -- close() deletes the
                                                # Session object that holds it
            my_bpod.close()
            _export_session_struct(csv_path)
            sys.exit(0)

        if not decision_result.get('ran', False):
            print("Bpod stopped running trials (Stop/Kill) -- ending session early after trial "
                  "{0}.".format(trial), flush=True)
            break

        visited = my_bpod.session.current_trial.states_durations
        rewarded = was_visited(visited, 'Reward')
        outcome_state = 'Reward' if rewarded else 'NoMovement'
        outcome = 'Rewarded' if rewarded else 'NoMovement'

        session_trial_count += 1

        choice_events = my_bpod.session.current_trial.get_all_timestamps_by_event()
        neg_crossings = choice_events.get(neg_event, [])
        pos_crossings = choice_events.get(pos_event, [])
        if neg_crossings and pos_crossings:
            side = 'L' if neg_crossings[-1] < pos_crossings[-1] else 'R'
        elif neg_crossings:
            side = 'L'
        elif pos_crossings:
            side = 'R'
        else:
            side = 'R'   # NoMovement trial -- arbitrary, only affects the raster's left/right color
        magnitude_deg = cur_threshold_deg if rewarded else 0.0

        runner.register('TRIAL_TYPE', 'stage1')
        runner.register('MOVEMENT_DIRECTION', side)
        runner.register('OUTCOME', outcome)
        runner.register('GAIN_MULT', cur_gain_mult)
        if rewarded:
            runner.register('THRESHOLD_CROSSING_TIME', send_t + visited[outcome_state][-1][0])

        print("Trial {0}: held {1:.2f}s (broke {2}x), {3}, gain={4:.2f}x".format(
            trial, VAR_QUIESCENCE_S, n_breaks, outcome, cur_gain_mult), flush=True)

        # Port1In licks are already natively logged as raw EVENT rows in the session CSV (see
        # CLAUDE.md) -- extracted here only for the LIVE plot's own in-memory use, same technique
        # full_protocol_lookback_test.py already uses, not a new VAL registration.
        lick_times_abs = [send_t + t for t in choice_events.get('Port1In', [])]
        reward_time_abs = send_t + visited['Reward'][-1][0] if rewarded else None

        bench_plots.add_trial(side, magnitude_deg, cur_threshold_deg, outcome,
                               gain_mult=cur_gain_mult, lick_times_abs=lick_times_abs,
                               reward_time_abs=reward_time_abs)

    except Exception as err:
        print("Trial {0} FAILED: {1}".format(trial, err), flush=True)
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        raise
else:
    runner.register('SESSION_END_REASON', 'completed')
    print("Done: reached VAR_MAX_TRIALS ({0})".format(VAR_MAX_TRIALS), flush=True)

# --- session-end bookkeeping: advancement gate + gain decay + threshold growth ----------------------

cleared_this_session = session_trial_count > VAR_TRIAL_COUNT_ADVANCE
history = state.get('sessions_trial_count_history')
history.append(cleared_this_session)
state.set('sessions_trial_count_history', history[-2:])   # only need the last 2 to judge
                                                            # "two consecutive"

advance_ready = len(history) >= 2 and history[-1] and history[-2]

if cleared_this_session:
    state.set('gain_mult', staircase.decay_gain(cur_gain_mult))
    state.set('threshold_fraction', staircase.grow_stage1_threshold(cur_threshold_fraction))

state.save()

print("Session trial count: {0} ({1} the {2}-trial gate)".format(
    session_trial_count, "cleared" if cleared_this_session else "did NOT clear",
    VAR_TRIAL_COUNT_ADVANCE), flush=True)
print("ADVANCE-READY: {0}".format("yes" if advance_ready else "no"), flush=True)
if not advance_ready:
    print("  (needs >{0} trials on 2 CONSECUTIVE sessions; history so far this run: {1})".format(
        VAR_TRIAL_COUNT_ADVANCE, history), flush=True)
print("Threshold now {0:.2f}deg ({1:.0%} of final){2}".format(
    state.get('threshold_fraction') * VAR_THRESHOLD_FINAL_DEG, state.get('threshold_fraction'),
    " -- grew this session" if cleared_this_session else " -- unchanged (session didn't qualify)"),
    flush=True)

dot.close()
rotary.close()
csv_path = my_bpod.session._path   # grab before close() -- close() deletes the Session object
                                    # that holds it
my_bpod.close()
_export_session_struct(csv_path)

print("Close the plot window to exit.", flush=True)
plt.ioff()
plt.show()
