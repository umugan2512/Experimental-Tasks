# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""
training_protocol.md Part 4, Stage 2 -- "Threshold staircase, ITI growth, spout retraction".

Aim: grow the movement threshold, ITI, and spout position to their final values BEFORE Stage 3
introduces errors/clicks, so nothing else is still changing when clicks are introduced. Four
concurrent processes: (a) threshold staircase, +10% per 20 consecutive successes / -10% per 5
consecutive failures, ceiling at the final threshold; (b) ITI +0.1s/qualifying-session to a 1.5s
ceiling; (c) spout retraction ~2-3mm/qualifying-session (physical, manual -- this script can only
remind/log the schedule, there's no spout actuator to command); (d) side-bias management -- track
the direction ratio over the last ~40 trials, withhold reward on the over-used side if it leaves
the 30-70% band.

Advancement is primarily a STATISTICAL test (bimodal movement distribution + rewarded-turn
velocity clearly separated from drift) that the doc describes conceptually but doesn't fully
specify an algorithm for. Deliberately deferred: this script auto-checks the simple, doc-specified
numeric gates (trial count, ITI at ceiling, direction ratio in-band, via
staircase.stage2_simple_gates_met()) and prints an explicit reminder that the statistical test is
NOT implemented -- advancement should still be a human judgment call (reviewing the movement raster
in the live plot / logged data) until a concrete, validated implementation exists to check against
real Stage 2 movement data.

Shares its persisted state file with stage1_wheel_shaping.py (same subjects/<id>/
wheel_shaping_state.json, via session_state.StageState) -- the whole wheel-shaping curriculum is
one continuous per-subject record, not per-stage; Stage 2's own new keys (threshold_fraction etc.)
default to Stage 1's known ending values (see VAR_* defaults below) the first time this script runs
for a given subject.

Design change from the two-Bpod-machine pattern used elsewhere in this codebase for a similar
"detect outcome, then decide reward" need (full_protocol_lookback_test.py's cue_sma-then-sma):
NOT needed here, because which side (if either) is currently withheld is knowable from
DirectionRatioTracker BEFORE the trial starts (it depends only on the last ~40 trials' history,
not on anything that happens during this trial) -- so a single state machine can be built upfront
with the reward-or-withhold branching already baked into its state_change_conditions, no second
Bpod round-trip required.

Trial-loop/module-connection/dot-render-loop plumbing otherwise identical to
stage1_wheel_shaping.py -- see that file's docstring for the render-loop/thread-inversion/
hardware-timestamp-gap details, all unchanged here.
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
from direction_tracker import DirectionRatioTracker
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

VAR_SUBJECT_ID = 'REPLACE_ME'   # edit before every session -- must match the ID used for this
                                 # animal's Stage 1 sessions, so the staircase continues from where
                                 # Stage 1 left off rather than starting a fresh state file
VAR_PROJECT_DIR = os.path.abspath(os.path.join(_TASK_DIR, '..', '..'))

# --- stage parameters (training_protocol.md Part 4, Stage 2) -----------------------------------------

VAR_MAX_TRIALS = 400
VAR_QUIESCENCE_S = 0.1              # unchanged from Stage 1 -- doc: quiescence "resumes staircasing
                                     # only at Stage 4", frozen at its Stage-1 value through Stage 3
VAR_STEADY_THRESHOLD_DEG = 4
VAR_RESPONSE_TIMEOUT_S = 10.0       # same flagged/tunable placeholder as Stage 1

VAR_THRESHOLD_FINAL_DEG = 35        # matches Stage 1 / this project's established final-task value
VAR_TRIAL_COUNT_ADVANCE = 200

VAR_REWARD_DURATION = 0.1

VAR_DOT_ONSET_JITTER_MIN_S = 0.1    # J1/J2 unchanged from Stage 1 -- never change during training
VAR_DOT_ONSET_JITTER_MAX_S = 0.2
VAR_DOT_DISAPPEAR_MIN_S = 0.4
VAR_DOT_DISAPPEAR_MAX_S = 0.9

VAR_DOT_SCREEN_INDEX = 1
VAR_DOT_DIAMETER_PX = 60            # same unconfirmed-placeholder flag as every dot-stimulus task
VAR_DOT_BACKGROUND_GRAY = 128
VAR_DOT_GRAY = 0
VAR_DOT_EDGE_FRACTION = 0.9
VAR_RENDER_HZ = 30

VAR_SPOUT_STEP_MM = 2.5             # doc: "~2-3mm per session" -- midpoint, flagged/tunable;
                                     # printed as a manual-action reminder only, no actuator exists

VAR_ROTARY_USB_PORT = None
VAR_STILL_POLL_HZ = 50
VAR_POLL_HZ = 10

VAR_GO_CUE_LED_CHANNEL = 'PWM1'

# --- persisted cross-session state (shared file with Stage 1) ---------------------------------------

state = session_state.StageState(VAR_PROJECT_DIR, VAR_SUBJECT_ID, defaults={
    'threshold_fraction': 0.20,      # Stage 1's own fixed value -- the natural starting point
    'consecutive_successes': 0,
    'consecutive_failures': 0,
    'iti_s': 0.5,                    # Stage 1's own fixed ITI -- the natural starting point
    'spout_position_mm': 0.0,        # cumulative retraction so far, relative to Stage 1's
                                      # close-to-mouth position
    'sessions_trial_count_history': [],
})

staircase_obj = staircase.ThresholdStaircase(
    current_fraction=state.get('threshold_fraction'),
    consecutive_successes=state.get('consecutive_successes'),
    consecutive_failures=state.get('consecutive_failures'))
cur_iti_s = state.get('iti_s')
cur_spout_position_mm = state.get('spout_position_mm')

direction_tracker = DirectionRatioTracker()

# --- connect to Bpod, resolve modules ----------------------------------------------------------------

my_bpod = Bpod()
print("Connected to Bpod on {0}".format(my_bpod.serial_port), flush=True)

rotary, rotary_bpod_module = rotary_setup.connect_rotary(my_bpod, usb_port=VAR_ROTARY_USB_PORT)
reset_positions_trigger_id, rotary_channel = rotary_setup.build_reset_trigger(rotary_bpod_module)

log_python_t0 = time.time()
runner = TrialRunner(my_bpod, rotary, log_python_t0, still_poll_hz=VAR_STILL_POLL_HZ,
                      poll_hz=VAR_POLL_HZ)

dot = DotDisplay(screen_index=VAR_DOT_SCREEN_INDEX, diameter_px=VAR_DOT_DIAMETER_PX,
                  background_gray=VAR_DOT_BACKGROUND_GRAY, dot_gray=VAR_DOT_GRAY)
dot.show()
dot.clear()

screen_width_px = dot.get_screen_width_px()
final_dot_gain = (VAR_DOT_EDGE_FRACTION * (screen_width_px / 2.0)) / VAR_THRESHOLD_FINAL_DEG
dot.set_deg_to_px_gain(final_dot_gain)   # Stage 2 has no gain multiplier -- that was Stage 1's
                                          # concept only, gain is already at its final value here
print("Dot gain calibrated to {0:.2f} px/wheel-deg (screen width {1}px)".format(
    final_dot_gain, screen_width_px), flush=True)

render_interval = 1.0 / VAR_RENDER_HZ

print("Spout should be at approximately {0:.1f}mm retracted from Stage 1's position (per the "
      "~{1:.1f}mm/qualifying-session schedule) -- confirm this matches the physical setup before "
      "starting.".format(cur_spout_position_mm, VAR_SPOUT_STEP_MM), flush=True)

bench_plots = WheelShapingPlots(
    stage=2, threshold_final_deg=VAR_THRESHOLD_FINAL_DEG,
    prev_session_values={'threshold_deg': staircase_obj.current_fraction * VAR_THRESHOLD_FINAL_DEG,
                          'iti_s': cur_iti_s})

# --- trial loop -----------------------------------------------------------------------------------

print("Starting up to {0} trials, targeting {1} for advancement (current threshold "
      "{2:.1f}deg = {3:.0%} of final, ITI {4:.2f}s)".format(
          VAR_MAX_TRIALS, VAR_TRIAL_COUNT_ADVANCE,
          staircase_obj.current_fraction * VAR_THRESHOLD_FINAL_DEG, staircase_obj.current_fraction,
          cur_iti_s), flush=True)

session_trial_count = 0

for trial in range(1, VAR_MAX_TRIALS + 1):
    try:
        cur_threshold_deg = staircase_obj.current_fraction * VAR_THRESHOLD_FINAL_DEG

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
            event_names = rotary_setup.set_and_enable_thresholds(
                rotary, [-cur_threshold_deg, cur_threshold_deg])
        rotary.enable_evt_transmission()
        neg_event, pos_event = event_names[0], event_names[1]

        runner.register('TRIAL_START', trial_start_t)
        runner.register('THRESHOLD_DEG', cur_threshold_deg)
        dot.clear()
        dot.pump()

        # Known BEFORE the trial starts -- depends only on the last ~40 trials' recorded
        # directions, not on anything that happens this trial (see module docstring for why this
        # means a single state machine, not the two-machine pattern used elsewhere).
        withhold_left = direction_tracker.should_withhold('L')
        withhold_right = direction_tracker.should_withhold('R')

        dot_onset_delay = np.random.uniform(VAR_DOT_ONSET_JITTER_MIN_S, VAR_DOT_ONSET_JITTER_MAX_S)
        disappear_delay_s = np.random.uniform(VAR_DOT_DISAPPEAR_MIN_S, VAR_DOT_DISAPPEAR_MAX_S)

        send_epoch = time.time()
        send_t = send_epoch - log_python_t0

        sma = StateMachine(my_bpod)

        sma.add_state(
            state_name='WheelPeriod',
            state_timer=VAR_RESPONSE_TIMEOUT_S,
            state_change_conditions={
                neg_event: 'NoRewardL' if withhold_left else 'RewardL',
                pos_event: 'NoRewardR' if withhold_right else 'RewardR',
                Bpod.Events.Tup: 'NoMovement',
            },
            output_actions=[(rotary_channel, reset_positions_trigger_id),
                             (VAR_GO_CUE_LED_CHANNEL, 255)])

        sma.add_state(
            state_name='RewardL',
            state_timer=VAR_REWARD_DURATION,
            state_change_conditions={Bpod.Events.Tup: 'ITI'},
            output_actions=[(Bpod.OutputChannels.Valve, 1), (VAR_GO_CUE_LED_CHANNEL, 0)])

        sma.add_state(
            state_name='RewardR',
            state_timer=VAR_REWARD_DURATION,
            state_change_conditions={Bpod.Events.Tup: 'ITI'},
            output_actions=[(Bpod.OutputChannels.Valve, 1), (VAR_GO_CUE_LED_CHANNEL, 0)])

        sma.add_state(
            state_name='NoRewardL',
            state_timer=0,
            state_change_conditions={Bpod.Events.Tup: 'ITI'},
            output_actions=[(VAR_GO_CUE_LED_CHANNEL, 0)])

        sma.add_state(
            state_name='NoRewardR',
            state_timer=0,
            state_change_conditions={Bpod.Events.Tup: 'ITI'},
            output_actions=[(VAR_GO_CUE_LED_CHANNEL, 0)])

        sma.add_state(
            state_name='NoMovement',
            state_timer=0,
            state_change_conditions={Bpod.Events.Tup: 'ITI'},
            output_actions=[(VAR_GO_CUE_LED_CHANNEL, 0)])

        sma.add_state(
            state_name='ITI',
            state_timer=cur_iti_s,
            state_change_conditions={Bpod.Events.Tup: 'exit'},
            output_actions=[])

        # Thread-inverted, same pattern as stage1_wheel_shaping.py/full_protocol_lookback_test.py.
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
                    dot.set_position_deg(pos)

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
        choice_events = my_bpod.session.current_trial.get_all_timestamps_by_event()
        outcome_state = next(
            (name for name in ('RewardL', 'RewardR', 'NoRewardL', 'NoRewardR', 'NoMovement')
             if was_visited(visited, name)), 'Unknown')

        side = outcome_state[-1] if outcome_state in ('RewardL', 'RewardR', 'NoRewardL',
                                                        'NoRewardR') else None
        rewarded = outcome_state in ('RewardL', 'RewardR')
        staircase_success = outcome_state != 'NoMovement'   # cleared the threshold at all,
                                                              # independent of reward withholding
        outcome = 'Rewarded' if rewarded else ('Withheld' if side is not None else 'NoMovement')

        session_trial_count += 1
        staircase_obj.record_outcome(success=staircase_success)
        if side is not None:
            direction_tracker.record(side)

        runner.register('TRIAL_TYPE', 'stage2')
        runner.register('MOVEMENT_DIRECTION', side if side is not None else 'None')
        runner.register('OUTCOME', outcome)
        runner.register('DIRECTION_RATIO', direction_tracker.right_fraction())
        if outcome_state in ('RewardL', 'RewardR', 'NoRewardL', 'NoRewardR'):
            runner.register('THRESHOLD_CROSSING_TIME', send_t + visited[outcome_state][-1][0])

        print("Trial {0}: threshold={1:.1f}deg, {2}, direction_ratio={3:.2f}".format(
            trial, cur_threshold_deg, outcome, direction_tracker.right_fraction()), flush=True)

        # Port1In licks are already natively logged as raw EVENT rows in the session CSV (see
        # CLAUDE.md) -- extracted here only for the LIVE plot's own in-memory use, same technique
        # full_protocol_lookback_test.py already uses, not a new VAL registration.
        lick_times_abs = [send_t + t for t in choice_events.get('Port1In', [])]
        reward_time_abs = send_t + visited[outcome_state][-1][0] if rewarded else None

        magnitude_deg = cur_threshold_deg if staircase_success else 0.0
        bench_plots.add_trial(side if side is not None else 'R', magnitude_deg, cur_threshold_deg,
                               outcome, iti_s=cur_iti_s,
                               direction_ratio=direction_tracker.right_fraction(),
                               lick_times_abs=lick_times_abs, reward_time_abs=reward_time_abs)

    except Exception as err:
        print("Trial {0} FAILED: {1}".format(trial, err), flush=True)
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        raise
else:
    print("Done: reached VAR_MAX_TRIALS ({0})".format(VAR_MAX_TRIALS), flush=True)

# --- session-end bookkeeping: staircase/ITI/spout persistence + advancement gates -------------------

cleared_this_session = session_trial_count > VAR_TRIAL_COUNT_ADVANCE

if cleared_this_session:
    cur_iti_s = staircase.grow_iti(cur_iti_s)
    cur_spout_position_mm += VAR_SPOUT_STEP_MM

state.set('threshold_fraction', staircase_obj.current_fraction)
state.set('consecutive_successes', staircase_obj.consecutive_successes)
state.set('consecutive_failures', staircase_obj.consecutive_failures)
state.set('iti_s', cur_iti_s)
state.set('spout_position_mm', cur_spout_position_mm)

history = state.get('sessions_trial_count_history')
history.append(cleared_this_session)
state.set('sessions_trial_count_history', history[-2:])

state.save()

simple_gates_met = staircase.stage2_simple_gates_met(
    trial_count=session_trial_count, iti_s=cur_iti_s,
    direction_ratio_in_band=direction_tracker.in_band(), trial_count_gate=VAR_TRIAL_COUNT_ADVANCE)

print("Session trial count: {0}".format(session_trial_count), flush=True)
print("Threshold now {0:.1f}deg ({1:.0%} of final), ITI now {2:.2f}s, spout position ~{3:.1f}mm "
      "(schedule only -- confirm physically)".format(
          staircase_obj.current_fraction * VAR_THRESHOLD_FINAL_DEG, staircase_obj.current_fraction,
          cur_iti_s, cur_spout_position_mm), flush=True)
print("Simple advancement gates met (trial count / ITI at ceiling / direction ratio in-band): "
      "{0}".format(simple_gates_met), flush=True)
print("NOTE: the doc's real advancement criterion is a statistical test (bimodal movement "
      "distribution + velocity separation) that is NOT YET IMPLEMENTED -- treat advancement as a "
      "human judgment call (review the movement raster above) even when the simple gates are met.",
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
