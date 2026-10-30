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

**Retrofit pass (propagating what Tests/ proved out, same as stage1_wheel_shaping.py)**:
`CameraRecorder` now records every session here too; the dot-gain computation now goes through
`rotary_setup.screen_direction_gain()` (previously missing here, unlike every other dot-coupled
script); `_cleanup_and_export()` consolidates the three separate hand-copied cleanup blocks into
one and fixes a real gap -- an unhandled per-trial exception used to `raise` straight out of the
loop, skipping teardown (and the staircase/ITI/spout persistence bookkeeping) entirely.
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
sys.path.insert(0, os.path.join(_TASK_DIR, '..', '..', '..', '..', 'Calibration'))
from water_tracker import ConsumedWaterTracker
import staircase
import session_state
import session_struct_export
import session_csv_parser
from direction_tracker import DirectionRatioTracker
from wheel_shaping_plots import WheelShapingPlots
from bpod_trial_helpers import TrialRunner, was_visited, PC_EVENT_QUIESCENCE_MET, PC_EVENT_DOT_SHOWN
from trial_render import run_trial_render_loop, register_trial_timing
import host_clock
import rotary_setup
import dot_display
from camera_recorder import CameraRecorder
from liquid_calibration import get_reward_duration_s

from confapp import conf as settings
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


def _cleanup_and_export():
    """ Shared teardown, called from every exit path (normal end, Stop-triggered break, Kill, and
    an unhandled per-trial exception) so hardware connections and the struct export are never
    skipped regardless of how the session ended -- fixes a real gap: an unhandled exception used
    to `raise` straight out of the trial loop, skipping this teardown (and the Kill/final-cleanup
    code below used to duplicate this by hand in three places instead of sharing it). """
    runner.stop()   # writes the last buffered wheel samples, then stops the rotary's USB stream --
                     # before rotary.close() and my_bpod.close(), which it needs.
    camera.close()
    realized_fps = camera.compute_realized_fps()
    if realized_fps is not None:
        my_bpod.register_value('CAMERA_FPS_REALIZED', realized_fps)
    dot.close()
    rotary.close()
    csv_path = my_bpod.session._path   # grab before close() -- close() deletes the Session object
                                        # that holds it
    my_bpod.close()
    _export_session_struct(csv_path)

# --- who this session is for -----------------------------------------------------------------------

_raw_subjects = getattr(settings, 'PYBPOD_SUBJECTS', None)
if not _raw_subjects:
    raise RuntimeError(
        "No subject selected in the GUI for this session (PYBPOD_SUBJECTS is empty) -- select a "
        "subject before running this task.")
VAR_SUBJECT_ID = session_csv_parser.parse_subject_name(_raw_subjects[0])   # derived from the GUI's
                                 # own selected subject -- matches whatever Stage 1 used for this
                                 # same animal automatically, no manual ID to keep in sync by hand
VAR_PROJECT_DIR = os.path.abspath(os.path.join(_TASK_DIR, '..', '..'))

# --- stage parameters (training_protocol.md Part 4, Stage 2) -----------------------------------------

VAR_MAX_TRIALS = 400                # hard-ceiling SAFETY backstop only, above both real caps
                                     # below (VAR_TRIAL_COUNT_ADVANCE trials / VAR_SESSION_WATER_
                                     # CAP_UL) -- should essentially never be the thing that
                                     # actually ends a session.
# If the LAST session for this subject+protocol ended via Stop/Kill/crash (no SESSION_END_REASON
# registered) and this one is starting soon enough to count as its own continuation (same
# merge-eligibility window build_training_log.py's own session-merging uses), reduce today's own
# trial quota by however many trials that interrupted bout already ran -- so a restart doesn't
# hand the animal a full fresh 400 trials on top of what it already did. Resets back to the full
# value automatically once a session reaches its own deliberate end (see
# session_csv_parser.compute_remaining_quota()'s own docstring).
VAR_MAX_TRIALS, _ = session_csv_parser.compute_remaining_quota(
    os.path.dirname(os.path.abspath(os.getcwd())), os.path.basename(os.path.abspath(os.getcwd())),
    os.path.splitext(os.path.basename(__file__))[0], VAR_SUBJECT_ID, full_max_trials=VAR_MAX_TRIALS)
VAR_QUIESCENCE_S = 0.1              # unchanged from Stage 1 -- used as a FIXED value only until
                                     # the quiescence staircase unlocks (see 'quiescence_unlocked'
                                     # below: the first time the movement-threshold staircase steps
                                     # up at least once within a session, staircasing begins the
                                     # NEXT session). Before that unlock, every trial holds here.
VAR_STEADY_THRESHOLD_DEG = 4
VAR_RESPONSE_TIMEOUT_S = 10.0       # same flagged/tunable placeholder as Stage 1

VAR_THRESHOLD_FINAL_DEG = 35        # matches Stage 1 / this project's established final-task value
VAR_TRIAL_COUNT_ADVANCE = 100       # The QUALIFYING threshold only (lowered from the doc's
                                     # original 200, per explicit instruction) -- how many
                                     # trials with a genuine movement (rewarded or withheld; see
                                     # staircase_success below) this session needs for
                                     # advancement purposes. Deliberately NOT the same thing as
                                     # the session's own hard trial cap any more (see
                                     # VAR_TRIAL_CAP below) -- those two numbers used to share
                                     # one variable/value, but per explicit instruction they're
                                     # now independent.
VAR_TRIAL_CAP = 250                 # The session's own hard trial-count cap (one of three --
                                     # see below), checked against every ATTEMPTED trial, not
                                     # the qualifying count above.
VAR_DAILY_WATER_CAP_UL = 1500.0     # CONSUMED per animal per calendar DAY, across all of its
                                     # sessions (session_csv_parser.water_consumed_today()).
                                     # Per explicit instruction: three caps on the session --
                                     # VAR_TRIAL_CAP trials, 1.5mL consumed, or 30 minutes
                                     # elapsed -- whichever comes first.
VAR_SESSION_DURATION_CAP_S = 30 * 60

VAR_REWARD_UL = float(os.environ.get('RUN_SESSION_REWARD_UL', 4.0))   # target reward volume --
                                     # valve open time is derived from this via
                                     # Calibration/liquid_calibration.py's own fitted curve. The
                                     # env var lets run_session.py's own launch-time reward field
                                     # override this default; absent (real PyBpod-GUI launch) ->
                                     # falls back to the hardcoded default, unchanged behavior.
                                     # (falls back to a hardcoded 0.1s if no calibration data/fit
                                     # exists yet on this machine -- see get_reward_duration_s()).
VAR_REWARD_DURATION = get_reward_duration_s(VAR_REWARD_UL)

VAR_DOT_ONSET_JITTER_MIN_S = 0.1    # J1/J2 unchanged from Stage 1 -- never change during training
VAR_DOT_ONSET_JITTER_MAX_S = 0.2
VAR_DOT_DISAPPEAR_MIN_S = 0.4
VAR_DOT_DISAPPEAR_MAX_S = 0.9

VAR_DOT_EDGE_FRACTION = 0.9
VAR_RENDER_HZ = 30

VAR_SPOUT_STEP_MM = 2.5             # doc: "~2-3mm per session" -- midpoint, flagged/tunable;
                                     # printed as a manual-action reminder only, no actuator exists

VAR_ROTARY_USB_PORT = None
VAR_STILL_POLL_HZ = 100              # how often the hold-to-init check re-evaluates the streamed
                                      # samples. The wheel's own sample rate is set by the rotary
                                      # firmware's USB stream, not by this (see rotary_stream.py).
VAR_POLL_HZ = 100                    # unused in stream mode (polling-mode TrialRunner only)

SOFTCODE_DOT_ON = 1                  # sent by DotOn's entry: draw the dot (and answer once painted)
SOFTCODE_CROSSED = 2                 # sent by RewardL/R and NoRewardL/R's entry: freeze the dot
VAR_DOT_HANDSHAKE_TIMEOUT_S = 1.0    # DotOn -> DotTimeout if the PC hasn't confirmed the dot by then
VAR_WAIT_QUIESCENCE_MAX_S = 3600.0   # WaitQuiescence's own timer ("as long as it takes")

VAR_CAMERA_INDEX = None                      # None = auto-discover, see
                                              # camera_recorder.discover_camera()
VAR_CAMERA_OUTPUT_PATH = 'session_video.avi' # relative to cwd -- lands in the real session
                                              # folder when run for real via the GUI's Run button.
VAR_CAMERA_FPS = 30.0
VAR_CAMERA_PREVIEW = os.environ.get('RUN_SESSION_DASHBOARD') != '1'   # preview shown only in short
                                       # snippets (see camera.show_snippet() calls below), never
                                       # continuously -- same low-overhead choice
                                       # full_protocol_lookback_test.py already made (see
                                       # camera_recorder.py's own docstring). False when launched
                                       # from run_session.py -- see stage1_wheel_shaping.py's own
                                       # identical comment.
VAR_CAMERA_SNIPPET_S = 1.0            # snippet duration for both preview windows below.

# --- persisted cross-session state (shared file with Stage 1) ---------------------------------------

state = session_state.StageState(VAR_PROJECT_DIR, VAR_SUBJECT_ID, defaults={
    'threshold_fraction': 0.20,      # Stage 1's own fixed value -- the natural starting point
    'consecutive_successes': 0,
    'consecutive_failures': 0,
    'iti_s': 0.5,                    # Stage 1's own fixed ITI -- the natural starting point
    'spout_position_mm': 0.0,        # cumulative retraction so far, relative to Stage 1's
                                      # close-to-mouth position
    'sessions_trial_count_history': [],
    'quiescence_unlocked': False,    # one-way latch: flips True the first session in which the
                                      # movement-threshold staircase below steps UP at least once.
                                      # Staircasing of the hold-to-initiate duration then begins
                                      # the NEXT session (not this one) -- see the trial loop and
                                      # session-end bookkeeping below. Carried forward unchanged by
                                      # Stage 3/4 once set; Stage 5 never re-touches it.
    'quiescence_s': staircase.QUIESCENCE_FLOOR_S,
    'quiescence_consecutive_successes': 0,
    'quiescence_consecutive_failures': 0,
})

staircase_obj = staircase.ThresholdStaircase(
    current_fraction=state.get('threshold_fraction'),
    consecutive_successes=state.get('consecutive_successes'),
    consecutive_failures=state.get('consecutive_failures'))
cur_iti_s = state.get('iti_s')
cur_spout_position_mm = state.get('spout_position_mm')

quiescence_unlocked = state.get('quiescence_unlocked')
quiescence_unlocked_at_session_start = quiescence_unlocked   # snapshot -- session-end bookkeeping
                                                              # below uses this to know whether
                                                              # quiescence_obj was actually built
if quiescence_unlocked:
    quiescence_obj = staircase.QuiescenceStaircase(
        current_s=state.get('quiescence_s'),
        consecutive_successes=state.get('quiescence_consecutive_successes'),
        consecutive_failures=state.get('quiescence_consecutive_failures'))
session_saw_threshold_step_up = False   # set True the first time this session's threshold
                                         # staircase steps up at least once (see record_outcome call
                                         # below) -- drives the one-way unlock above.

direction_tracker = DirectionRatioTracker()

# --- connect to Bpod, resolve modules ----------------------------------------------------------------

my_bpod = Bpod()
print("Connected to Bpod on {0}".format(my_bpod.serial_port), flush=True)

rotary, rotary_bpod_module = rotary_setup.connect_rotary(my_bpod, usb_port=VAR_ROTARY_USB_PORT)
reset_positions_trigger_id, rotary_channel = rotary_setup.build_reset_trigger(
    rotary_bpod_module, marker_code=rotary_setup.DOT_ONSET_MARKER_CODE)

# Thresholds are armed once here, before the stream starts; the trial loop only re-sets them (inside
# stream.paused()) when the staircase actually moves. Event names never change (two thresholds).
cur_threshold_deg_at_start = staircase_obj.current_fraction * VAR_THRESHOLD_FINAL_DEG
neg_event, pos_event = rotary_setup.set_and_enable_thresholds(
    rotary, [-cur_threshold_deg_at_start, cur_threshold_deg_at_start])
rotary.enable_evt_transmission()   # once per session, never toggled (MATLAB's sendThresholdEvents)

my_bpod.register_value('REWARD_UL', VAR_REWARD_UL)
my_bpod.register_value('SOFTCODE_MAP', '1=DotOnRequest,2=OutcomeFreeze')   # Bpod->PC SoftCode numbers (SOFTCODE rows)
# What compute_remaining_quota() found at startup (an interrupted-and-restarted bout shortens this
# session's quota) -- registered so a surprisingly short session is traceable to its cause.
my_bpod.register_value('QUOTA_REDUCED_BY', session_csv_parser.LAST_QUOTA_INFO.get('used_trials', 0))
my_bpod.register_value('QUOTA_CHAIN_SESSIONS',
                       ','.join(session_csv_parser.LAST_QUOTA_INFO.get('chain_sessions', [])))

log_python_t0 = host_clock.now()
runner = TrialRunner(my_bpod, rotary, log_python_t0, still_poll_hz=VAR_STILL_POLL_HZ,
                      poll_hz=VAR_POLL_HZ, use_stream=True,
                      marker_code=rotary_setup.DOT_ONSET_MARKER_CODE)
runner.start()   # starts the rotary's own session-long USB stream (see rotary_stream.py) -- from
                  # here on, never call an ack-reading rotary command directly.

dot = dot_display.create_dot_display()
dot.show()
dot.clear()

camera = CameraRecorder(log_python_t0, camera_index=VAR_CAMERA_INDEX,
                         output_path=VAR_CAMERA_OUTPUT_PATH, fps=VAR_CAMERA_FPS,
                         preview=VAR_CAMERA_PREVIEW)
camera.start()

my_bpod.register_value('CAMERA_START_TIME', camera.start_time_s)
my_bpod.register_value('CAMERA_OUTPUT_PATH', camera.output_path)
my_bpod.register_value('CAMERA_FPS', camera.fps)

# rotary_setup.screen_direction_gain() applies this rig's confirmed wheel->screen sign correction
# (see rotary_setup.py's WHEEL_TO_SCREEN_SIGN) -- previously missing here, unlike every other
# dot-coupled script, so the dot most likely moved opposite the wheel-turn direction on this rig.
screen_width_px = dot.get_screen_width_px()
final_dot_gain = rotary_setup.screen_direction_gain(
    (VAR_DOT_EDGE_FRACTION * (screen_width_px / 2.0)) / VAR_THRESHOLD_FINAL_DEG)
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
                          'iti_s': cur_iti_s},
    reward_ul=VAR_REWARD_UL,
    # See stage1_wheel_shaping.py's own identical comment -- suppresses this script's own popup
    # window only when launched from run_session.py, which shows the same data in its own window.
    embed=(os.environ.get('RUN_SESSION_DASHBOARD') == '1'),
    headless=(os.environ.get('RUN_SESSION_DASHBOARD') == '1'))

# --- trial loop -----------------------------------------------------------------------------------

# The water cap is per DAY: what this animal already consumed in today's earlier sessions counts.
water_earlier_today_ul, _water_earlier_sessions = session_csv_parser.water_consumed_today(
    VAR_PROJECT_DIR, VAR_SUBJECT_ID, os.path.basename(os.path.abspath(os.getcwd()))[:8],
    exclude_session=os.path.basename(os.path.abspath(os.getcwd())))
runner.register('WATER_CONSUMED_EARLIER_TODAY_UL', water_earlier_today_ul)
print("Water consumed earlier today: {0:.0f}uL in {1} session(s){2}".format(
    water_earlier_today_ul, len(_water_earlier_sessions),
    ' ({0})'.format(', '.join(_water_earlier_sessions)) if _water_earlier_sessions else ''),
    flush=True)
print("Starting up to {0} trials, capped at {1} attempted trials, {2:.1f}mL consumed today, or "
      "{3:.0f}min elapsed -- whichever first ({4} genuine-movement trials targeted for "
      "advancement; current threshold {5:.1f}deg = {6:.0%} of final, ITI {7:.2f}s)".format(
          VAR_MAX_TRIALS, VAR_TRIAL_CAP, VAR_DAILY_WATER_CAP_UL / 1000.0,
          VAR_SESSION_DURATION_CAP_S / 60.0, VAR_TRIAL_COUNT_ADVANCE,
          staircase_obj.current_fraction * VAR_THRESHOLD_FINAL_DEG, staircase_obj.current_fraction,
          cur_iti_s), flush=True)

session_trial_count = 0
session_reward_ul_total = 0.0
water = ConsumedWaterTracker(VAR_REWARD_UL)   # the water cap counts CONSUMED water
session_start_time = host_clock.now()
session_end_reason = None
skipped_for_daily_cap = False   # ended before trial 1: no session happened

applied_threshold_deg = cur_threshold_deg_at_start   # what the rotary is currently armed with

for trial in range(1, VAR_MAX_TRIALS + 1):
    if trial == 1 and water_earlier_today_ul >= VAR_DAILY_WATER_CAP_UL:
        session_end_reason = 'water_cap'
        skipped_for_daily_cap = True
        print("Daily water cap already reached ({0:.0f}uL of {1:.0f}uL consumed today) -- ending "
              "before the first trial.".format(water_earlier_today_ul, VAR_DAILY_WATER_CAP_UL),
              flush=True)
        break
    try:
        cur_threshold_deg = staircase_obj.current_fraction * VAR_THRESHOLD_FINAL_DEG
        if cur_threshold_deg != applied_threshold_deg:
            # set_thresholds() reads an ack byte, so it runs inside stream.paused() -- between
            # trials, before this trial's hold-to-init wait, never while a machine is running.
            rotary_setup.set_and_enable_thresholds(
                rotary, [-cur_threshold_deg, cur_threshold_deg], stream=runner.stream)
            applied_threshold_deg = cur_threshold_deg

        if quiescence_unlocked:
            quiescence_duration_this_trial = float(np.clip(
                np.random.exponential(quiescence_obj.current_s),
                staircase.QUIESCENCE_FLOOR_S, staircase.QUIESCENCE_CEILING_S))
        else:
            quiescence_duration_this_trial = VAR_QUIESCENCE_S
        # J1 lives INSIDE the quiescence requirement (see stage1_wheel_shaping.py): the dot may only
        # appear after quiescence, and the whole trial is one Bpod machine (_shared/trial_render.py).
        dot_onset_jitter_s = np.random.uniform(VAR_DOT_ONSET_JITTER_MIN_S,
                                               VAR_DOT_ONSET_JITTER_MAX_S)
        quiescence_hold_s = quiescence_duration_this_trial + dot_onset_jitter_s
        disappear_delay_s = np.random.uniform(VAR_DOT_DISAPPEAR_MIN_S, VAR_DOT_DISAPPEAR_MAX_S)

        # Known BEFORE the trial starts -- depends only on the last ~40 trials' recorded
        # directions, not on anything that happens this trial (see module docstring for why this
        # means a single state machine, not the two-machine pattern used elsewhere).
        withhold_left = direction_tracker.should_withhold('L')
        withhold_right = direction_tracker.should_withhold('R')

        sma = StateMachine(my_bpod)

        sma.add_state(
            state_name='WaitQuiescence',
            state_timer=VAR_WAIT_QUIESCENCE_MAX_S,
            state_change_conditions={PC_EVENT_QUIESCENCE_MET: 'DotOn', Bpod.Events.Tup: 'exit'},
            output_actions=[])

        # Same handshake as stage1_wheel_shaping.py's DotOn: the response window opens only once the
        # PC confirms the dot is painted; a turn before that ends the trial as MovedBeforeDot.
        sma.add_state(
            state_name='DotOn',
            state_timer=VAR_DOT_HANDSHAKE_TIMEOUT_S,
            state_change_conditions={
                PC_EVENT_DOT_SHOWN: 'WheelPeriod',
                neg_event: 'MovedBeforeDot',
                pos_event: 'MovedBeforeDot',
                Bpod.Events.Tup: 'DotTimeout',
            },
            output_actions=[('SoftCode', SOFTCODE_DOT_ON)])

        sma.add_state(
            state_name='WheelPeriod',
            state_timer=VAR_RESPONSE_TIMEOUT_S,
            state_change_conditions={
                neg_event: 'NoRewardL' if withhold_left else 'RewardL',
                pos_event: 'NoRewardR' if withhold_right else 'RewardR',
                Bpod.Events.Tup: 'NoMovement',
            },
            output_actions=[(rotary_channel, reset_positions_trigger_id)])

        sma.add_state(
            state_name='RewardL',
            state_timer=VAR_REWARD_DURATION,
            state_change_conditions={Bpod.Events.Tup: 'ITI'},
            output_actions=[(Bpod.OutputChannels.Valve, 1), ('SoftCode', SOFTCODE_CROSSED)])

        sma.add_state(
            state_name='RewardR',
            state_timer=VAR_REWARD_DURATION,
            state_change_conditions={Bpod.Events.Tup: 'ITI'},
            output_actions=[(Bpod.OutputChannels.Valve, 1), ('SoftCode', SOFTCODE_CROSSED)])

        sma.add_state(
            state_name='NoRewardL',
            state_timer=0,
            state_change_conditions={Bpod.Events.Tup: 'ITI'},
            output_actions=[('SoftCode', SOFTCODE_CROSSED)])

        sma.add_state(
            state_name='NoRewardR',
            state_timer=0,
            state_change_conditions={Bpod.Events.Tup: 'ITI'},
            output_actions=[('SoftCode', SOFTCODE_CROSSED)])

        sma.add_state(
            state_name='NoMovement',
            state_timer=0,
            state_change_conditions={Bpod.Events.Tup: 'ITI'},
            output_actions=[])

        sma.add_state(
            state_name='MovedBeforeDot',
            state_timer=0,
            state_change_conditions={Bpod.Events.Tup: 'ITI'},
            output_actions=[])

        sma.add_state(
            state_name='DotTimeout',
            state_timer=0,
            state_change_conditions={Bpod.Events.Tup: 'ITI'},
            output_actions=[])

        sma.add_state(
            state_name='ITI',
            state_timer=cur_iti_s,
            state_change_conditions={Bpod.Events.Tup: 'exit'},
            output_actions=[])

        runner.upload(sma)

        # Thread-inverted, same pattern as stage1_wheel_shaping.py/full_protocol_lookback_test.py.
        decision_result = {}
        decision_done = threading.Event()

        def _run_decision():
            try:
                decision_result['ran'] = runner.run_trial_state_machine(
                    sma, marker_state='WheelPeriod')
            except SystemExit:
                decision_result['killed'] = True
            except Exception:
                decision_result['error'] = traceback.format_exc()
            finally:
                decision_done.set()

        decision_thread = threading.Thread(target=_run_decision, daemon=True)
        prev_return_t = runner.last_return_t
        runner.begin_trial()   # main thread, BEFORE the Bpod thread -- see its docstring
        runner.start_quiescence(quiescence_hold_s, VAR_STEADY_THRESHOLD_DEG)
        decision_thread.start()

        render = run_trial_render_loop(
            runner, dot, camera, decision_done, cur_threshold_deg, disappear_delay_s,
            render_interval, SOFTCODE_DOT_ON, (SOFTCODE_CROSSED,),
            quiescence=(quiescence_hold_s, VAR_STEADY_THRESHOLD_DEG))

        decision_thread.join(timeout=3.0)
        if decision_thread.is_alive():
            print("WARNING: decision-period Bpod thread did not stop within 3s -- continuing "
                  "anyway.", flush=True)

        dot.clear()
        dot.pump()
        camera.pump()

        if decision_result.get('killed'):
            print("Bpod Kill received -- ending session.", flush=True)
            _cleanup_and_export()
            sys.exit(0)

        if decision_result.get('error'):
            # Reported as what it is -- it used to fall through to the "Stop/Kill" message below.
            print("Trial {0} CRASHED in the Bpod thread -- ending session:\n{1}".format(
                trial, decision_result['error']), flush=True)
            runner.register('SESSION_ERROR', decision_result['error'].strip().splitlines()[-1])
            session_end_reason = 'error'
            break

        if not decision_result.get('ran', False):
            print("Bpod stopped running trials (Stop/Kill) -- ending session early after trial "
                  "{0}.".format(trial), flush=True)
            break

        # Host anchor for every Bpod-relative time below: captured right before RUN (see
        # TrialRunner.run_trial_state_machine()).
        send_t = runner.last_run_t
        n_breaks = runner.quiescence_breaks
        visited = my_bpod.session.current_trial.states_durations
        choice_events = my_bpod.session.current_trial.get_all_timestamps_by_event()
        runner.register('SM_RUN_TIME', send_t)
        runner.register('QUIESCENCE_DUR_S', quiescence_duration_this_trial)
        runner.register('THRESHOLD_DEG', cur_threshold_deg)
        runner.register('DOT_ONSET_JITTER_S', dot_onset_jitter_s)
        register_trial_timing(runner, render, quiescence_hold_s, n_breaks, prev_return_t,
                              visited, send_t)

        early = next((s for s in ('MovedBeforeDot', 'DotTimeout') if was_visited(visited, s)), None)
        outcome_state = next(
            (name for name in ('RewardL', 'RewardR', 'NoRewardL', 'NoRewardR', 'NoMovement',
                               'MovedBeforeDot', 'DotTimeout')
             if was_visited(visited, name)), 'Unknown')

        side = outcome_state[-1] if outcome_state in ('RewardL', 'RewardR', 'NoRewardL',
                                                        'NoRewardR') else None
        rewarded = outcome_state in ('RewardL', 'RewardR')
        staircase_success = side is not None   # cleared the threshold at all, independent of
                                                 # reward withholding
        outcome = ('Rewarded' if rewarded else 'Withheld' if side is not None
                   else early if early else 'NoMovement')

        # Per explicit instruction: a genuine movement counts toward the qualifying/session-cap
        # threshold even if reward was withheld (debiasing) -- only a true NoMovement trial (no
        # response at all) is excluded, same "disengagement vs. genuine attempt" line Stage 3/4/5
        # already draw via their own trial_type=='main' and correct is not None gate. A trial that
        # ended before the dot (MovedBeforeDot/DotTimeout) never offered a response window, so it
        # is no movement-threshold staircase outcome either way -- but it IS a failed hold.
        if staircase_success:
            session_trial_count += 1
        if rewarded:
            session_reward_ul_total += VAR_REWARD_UL
        if not early:
            fraction_before_step = staircase_obj.current_fraction
            staircase_obj.record_outcome(success=staircase_success)
            if staircase_obj.current_fraction > fraction_before_step:
                session_saw_threshold_step_up = True
        if quiescence_unlocked:
            quiescence_obj.record_outcome(success=(n_breaks == 0 and early != 'MovedBeforeDot'))
        if side is not None:
            direction_tracker.record(side)

        runner.register('TRIAL_TYPE', 'stage2')
        runner.register('MOVEMENT_DIRECTION', side if side is not None else 'None')
        runner.register('OUTCOME', outcome)
        runner.register('DIRECTION_RATIO', direction_tracker.right_fraction())
        if side is not None:
            runner.register('THRESHOLD_CROSSING_TIME', send_t + visited[outcome_state][-1][0])

        print("Trial {0}: threshold={1:.1f}deg, {2}, direction_ratio={3:.2f}".format(
            trial, cur_threshold_deg, outcome, direction_tracker.right_fraction()), flush=True)

        # Port1In/Port1Out licks are already natively logged as raw EVENT rows in the session CSV
        # (see CLAUDE.md) -- register_port_events() also explicitly registers both edges as their
        # own VALs (per explicit instruction: always report port in AND out times, not just in).
        lick_times_abs, _lick_times_out = runner.register_port_events('LICK', choice_events, send_t)
        water.record_trial(choice_events.get('Port1In', []), visited[outcome_state][-1][0] if rewarded else None)
        runner.register('SESSION_WATER_CONSUMED_UL', water.consumed_ul)
        runner.register('DAY_WATER_CONSUMED_UL', water_earlier_today_ul + water.consumed_ul)
        reward_time_abs = send_t + visited[outcome_state][-1][0] if rewarded else None

        magnitude_deg = cur_threshold_deg if staircase_success else 0.0
        bench_plots.add_trial(side if side is not None else 'R', magnitude_deg, cur_threshold_deg,
                               'Abort' if early else outcome, iti_s=cur_iti_s,
                               direction_ratio=direction_tracker.right_fraction(),
                               lick_times_abs=lick_times_abs, reward_time_abs=reward_time_abs)

        # Deliberate session cap, per explicit instruction: stop at VAR_TRIAL_CAP ATTEMPTED
        # trials (not the separate, smaller VAR_TRIAL_COUNT_ADVANCE qualifying-count threshold),
        # 1.5mL CONSUMED today (licked after delivery, all of the day's sessions), or 30 minutes elapsed -- whichever comes first. VAR_MAX_TRIALS (400)
        # above is only a further-out safety backstop, not the real target. Checked at trial end
        # (not trial start) so the trial that actually crosses a cap still completes and gets
        # logged.
        if trial >= VAR_TRIAL_CAP:
            session_end_reason = 'trial_count_cap'
        elif water_earlier_today_ul + water.consumed_ul >= VAR_DAILY_WATER_CAP_UL:
            session_end_reason = 'water_cap'
        elif (host_clock.now() - session_start_time) >= VAR_SESSION_DURATION_CAP_S:
            session_end_reason = 'duration_cap_30min'
        if session_end_reason is not None:
            print("Session cap reached ({0} trials attempted, {1} qualifying, {2:.1f}uL consumed of "
                  "{3:.1f}uL delivered, {4:.1f}min elapsed) -- ending session.".format(
                      trial, session_trial_count, water.consumed_ul, session_reward_ul_total,
                      (host_clock.now() - session_start_time) / 60.0), flush=True)
            break

    except Exception as err:
        print("Trial {0} FAILED: {1}".format(trial, err), flush=True)
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        # Fix (was a real gap, not intentional): re-raising here used to skip everything below,
        # including session-end bookkeeping and teardown -- camera/dot/rotary/my_bpod were all
        # left open and no struct was exported on any trial-loop exception.
        _cleanup_and_export()
        raise
else:
    runner.register('SESSION_END_REASON', 'completed')
    print("Done: reached VAR_MAX_TRIALS ({0})".format(VAR_MAX_TRIALS), flush=True)

if session_end_reason is not None:
    runner.register('SESSION_END_REASON', session_end_reason)

# --- session-end bookkeeping: staircase/ITI/spout persistence + advancement gates -------------------

# >= , not > -- the session's own hard cap is now VAR_TRIAL_CAP (a separate, larger number), not
# this qualifying threshold, so session_trial_count can legitimately exceed VAR_TRIAL_COUNT_ADVANCE
# by the time the session actually ends; >= still correctly recognizes "met or exceeded."
cleared_this_session = session_trial_count >= VAR_TRIAL_COUNT_ADVANCE

if cleared_this_session:
    cur_iti_s = staircase.grow_iti(cur_iti_s)
    cur_spout_position_mm += VAR_SPOUT_STEP_MM

state.set('threshold_fraction', staircase_obj.current_fraction)
state.set('consecutive_successes', staircase_obj.consecutive_successes)
state.set('consecutive_failures', staircase_obj.consecutive_failures)
state.set('iti_s', cur_iti_s)
state.set('spout_position_mm', cur_spout_position_mm)

if not quiescence_unlocked and session_saw_threshold_step_up:
    quiescence_unlocked = True
    print("Quiescence hold-to-initiate UNLOCKED this session (threshold staircase stepped up) -- "
          "staircasing begins NEXT session.", flush=True)
state.set('quiescence_unlocked', quiescence_unlocked)
if quiescence_unlocked_at_session_start:   # i.e. quiescence_obj was actually built/used this
                                            # session -- a same-session unlock (above) only takes
                                            # effect starting NEXT session, so there's nothing of
                                            # quiescence_obj's to persist yet in that case.
    state.set('quiescence_s', quiescence_obj.current_s)
    state.set('quiescence_consecutive_successes', quiescence_obj.consecutive_successes)
    state.set('quiescence_consecutive_failures', quiescence_obj.consecutive_failures)

history = state.get('sessions_trial_count_history')
if not skipped_for_daily_cap:   # a session that never ran is not a non-qualifying session
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

_cleanup_and_export()

print("Close the plot window to exit.", flush=True)
plt.ioff()
plt.show()
