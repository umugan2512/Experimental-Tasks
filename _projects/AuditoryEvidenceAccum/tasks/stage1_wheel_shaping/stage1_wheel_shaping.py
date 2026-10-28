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

Timing (2026-10-23): each trial is ONE Bpod state machine, WaitQuiescence -> DotOn -> WheelPeriod ->
Reward/NoMovement -> ITI, driven from the PC side by _shared/trial_render.py:
- WaitQuiescence: the PC watches the rotary's hardware-timestamped USB stream; once the wheel has
  been still for VAR_QUIESCENCE_S + J1 (J1 lives inside quiescence), it re-arms the rotary and
  injects SoftCode1 -> DotOn. Licks and Stop/Kill are handled throughout, since a machine is running.
- DotOn fires SOFTCODE_DOT_ON; the PC draws the dot centred, pumps Qt and answers -> WheelPeriod,
  whose entry re-zeroes the rotary and sends the '#' marker. So the dot's onset, the wheel's zero and
  the opening of the reward window are the same moment, and no reward can precede the dot. A turn
  before the answer -> MovedBeforeDot.
- Reward fires SOFTCODE_REWARD, which freezes the dot; it clears J2 later.
- The render loop runs on the main thread (owns the Qt QApplication) while the Bpod call runs on a
  background thread -- same thread-inversion pattern, same Kill/SystemExit handling as before.

**Retrofit pass (propagating what Tests/ proved out)**: `CameraRecorder` now records every
session here too (same construction/`.start()`/`.pump()`/snippet-preview/`.close()` wiring as
`full_protocol_lookback_test.py` -- see `camera_recorder.py`'s own docstring for why preview is
snippet-only, never continuous). The dot-gain computation now goes through
`rotary_setup.screen_direction_gain()` -- previously missing here (unlike every other dot-coupled
script in this codebase), meaning the dot most likely moved opposite the wheel-turn direction on
this rig until now. `_cleanup_and_export()` consolidates what used to be three separate hand-copied
cleanup blocks (Kill, exception, normal end) into one, and fixes a real gap in the exception path:
an unhandled per-trial exception used to `raise` straight out of the loop, skipping teardown
entirely (`camera`/`dot`/`rotary`/`my_bpod` left open, no struct exported).
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
from wheel_shaping_plots import WheelShapingPlots
from bpod_trial_helpers import TrialRunner, was_visited, triggering_event, PC_EVENT_QUIESCENCE_MET, PC_EVENT_DOT_SHOWN
from trial_render import run_trial_render_loop, register_trial_timing
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
                                 # own selected subject -- same settings-resolution mechanism
                                 # pybpodapi itself uses for SUBJECT-NAME, not a hand-edited constant
VAR_PROJECT_DIR = os.path.abspath(os.path.join(_TASK_DIR, '..', '..'))

# --- stage parameters (training_protocol.md Part 4, Stage 1's table) --------------------------------

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
VAR_THRESHOLD_STARTING_FRACTION = 0.10   # only used the very first time this subject runs Stage 1
                                     # (StageState default) -- doc: "~15-25% of final" was the OLD
                                     # fixed-value design; per later instruction, the threshold now
                                     # GROWS across sessions instead (see staircase.
                                     # grow_stage1_threshold()), starting at 10% (~3.5deg, raised
                                     # from an original 5%/1.75deg after real bench testing showed
                                     # 1.75deg was too easily crossed by incidental drift) and
                                     # reaching Stage 2's own 20% starting point by the time Stage 1
                                     # ends. Fixed for the whole SESSION (grows only between
                                     # sessions, not per-trial) -- read from persisted state below.

VAR_TRIAL_COUNT_ADVANCE = 100       # The QUALIFYING threshold only (lowered from the doc's
                                     # original 200, per explicit instruction) -- how many
                                     # REWARDED trials this session needs for advancement
                                     # purposes. Deliberately NOT the same thing as the session's
                                     # own hard trial cap any more (see VAR_TRIAL_CAP below) --
                                     # those two numbers used to share one variable/value, but per
                                     # explicit instruction they're now independent: a session can
                                     # run well past 100 rewarded trials before actually ending.
VAR_TRIAL_CAP = 200                 # The session's own hard trial-count cap (one of three --
                                     # see below), checked against every ATTEMPTED trial
                                     # (rewarded or not), not the qualifying count above.
VAR_SESSION_WATER_CAP_UL = 1500.0   # per explicit instruction: three caps on the session --
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

VAR_DOT_ONSET_JITTER_MIN_S = 0.1    # J1 (training_protocol.md SS1.5) -- runs at its FINAL value
VAR_DOT_ONSET_JITTER_MAX_S = 0.2    # from Stage 1 onward and never changes (doc: "the LED->dot and
VAR_DOT_DISAPPEAR_MIN_S = 0.4       # crossing->offset relationships should never change during
VAR_DOT_DISAPPEAR_MAX_S = 0.9       # training")

VAR_DOT_EDGE_FRACTION = 0.9         # SS1.3: place the threshold at ~90% of edge azimuth
VAR_GAIN_INITIAL_MULT = 3.0         # decays via staircase.decay_gain() (-0.1x/qualifying session,
                                     # floor 2.0x -- see staircase.py)
VAR_RENDER_HZ = 30

VAR_ROTARY_USB_PORT = None
VAR_STILL_POLL_HZ = 100              # how often the hold-to-init check re-evaluates the streamed
                                      # samples. The wheel's own sample rate is set by the rotary
                                      # firmware's USB stream, not by this (see rotary_stream.py).
VAR_POLL_HZ = 100                    # unused in stream mode (polling-mode TrialRunner only)

SOFTCODE_DOT_ON = 1                  # sent by DotOn's entry: draw the dot (and answer once painted)
SOFTCODE_REWARD = 2                  # sent by Reward's entry: freeze the dot
VAR_DOT_HANDSHAKE_TIMEOUT_S = 1.0    # DotOn -> DotTimeout if the PC hasn't confirmed the dot by then
VAR_WAIT_QUIESCENCE_MAX_S = 3600.0   # WaitQuiescence's own timer -- effectively "wait as long as it
                                      # takes"; a Stop/Kill ends it at any time

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
                                       # from run_session.py -- that dashboard's own embedded camera
                                       # panel already shows this same feed (see its own
                                       # RUN_SESSION_DASHBOARD env var docstring), so this script's
                                       # own popup preview window would just be a redundant second
                                       # one. True (unchanged) for a real PyBpod-GUI-launched session.
VAR_CAMERA_SNIPPET_S = 1.0            # snippet duration for both preview windows below.

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
reset_positions_trigger_id, rotary_channel = rotary_setup.build_reset_trigger(
    rotary_bpod_module, marker_code=rotary_setup.DOT_ONSET_MARKER_CODE)

event_names = rotary_setup.set_and_enable_thresholds(
    rotary, [-cur_threshold_deg, cur_threshold_deg])
rotary.enable_evt_transmission()
neg_event, pos_event = event_names[0], event_names[1]

my_bpod.register_value('THRESHOLD_DEG', cur_threshold_deg)
my_bpod.register_value('REWARD_UL', VAR_REWARD_UL)
my_bpod.register_value('SOFTCODE_MAP', '1=DotOnRequest,2=OutcomeFreeze')   # Bpod->PC SoftCode numbers (SOFTCODE rows)
# What compute_remaining_quota() found at startup (an interrupted-and-restarted bout shortens this
# session's quota) -- registered so a surprisingly short session is traceable to its cause.
my_bpod.register_value('QUOTA_REDUCED_BY', session_csv_parser.LAST_QUOTA_INFO.get('used_trials', 0))
my_bpod.register_value('QUOTA_CHAIN_SESSIONS',
                       ','.join(session_csv_parser.LAST_QUOTA_INFO.get('chain_sessions', [])))

log_python_t0 = time.time()
runner = TrialRunner(my_bpod, rotary, log_python_t0, still_poll_hz=VAR_STILL_POLL_HZ,
                      poll_hz=VAR_POLL_HZ, use_stream=True,
                      marker_code=rotary_setup.DOT_ONSET_MARKER_CODE)
runner.start()   # starts the rotary's own session-long USB stream (see rotary_stream.py) -- from
                  # here on, never call an ack-reading rotary command (current_position(),
                  # set_thresholds(), enable/disable_evt_transmission(), ...) directly.

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

# Geometry-aware FINAL gain (same derivation as full_protocol_lookback_test.py) -- Stage 1's actual
# gain is this, multiplied by the current (persisted, decaying) gain multiplier.
# rotary_setup.screen_direction_gain() applies this rig's confirmed wheel->screen sign correction
# (see rotary_setup.py's WHEEL_TO_SCREEN_SIGN) -- previously missing here, unlike every other
# dot-coupled script, so the dot most likely moved opposite the wheel-turn direction on this rig.
screen_width_px = dot.get_screen_width_px()
final_dot_gain = rotary_setup.screen_direction_gain(
    (VAR_DOT_EDGE_FRACTION * (screen_width_px / 2.0)) / VAR_THRESHOLD_FINAL_DEG)
dot.set_deg_to_px_gain(final_dot_gain * cur_gain_mult)
print("Dot gain calibrated to {0:.2f} px/wheel-deg (final={1:.2f}, mult={2:.2f}x, screen width "
      "{3}px)".format(final_dot_gain * cur_gain_mult, final_dot_gain, cur_gain_mult,
                       screen_width_px), flush=True)

render_interval = 1.0 / VAR_RENDER_HZ

bench_plots = WheelShapingPlots(
    stage=1, threshold_final_deg=VAR_THRESHOLD_FINAL_DEG,
    prev_session_values={'threshold_deg': cur_threshold_deg, 'gain_mult': cur_gain_mult},
    reward_ul=VAR_REWARD_UL,
    # When launched from run_session.py (its own dashboard already shows this same data,
    # reconstructed from the session CSV, embedded in its own window), embed=True builds the
    # figure without ever opening a separate top-level window -- see wheel_shaping_plots.py's own
    # docstring. Absent (a real PyBpod-GUI-launched session), this is False, identical to before.
    embed=(os.environ.get('RUN_SESSION_DASHBOARD') == '1'),
    headless=(os.environ.get('RUN_SESSION_DASHBOARD') == '1'))

# --- trial loop -----------------------------------------------------------------------------------

print("Starting up to {0} trials, capped at {1} attempted trials, {2:.1f}mL consumed, or "
      "{3:.0f}min elapsed -- whichever first ({4} REWARDED trials targeted for "
      "advancement)".format(
          VAR_MAX_TRIALS, VAR_TRIAL_CAP, VAR_SESSION_WATER_CAP_UL / 1000.0,
          VAR_SESSION_DURATION_CAP_S / 60.0, VAR_TRIAL_COUNT_ADVANCE), flush=True)

session_trial_count = 0
session_reward_ul_total = 0.0
water = ConsumedWaterTracker(VAR_REWARD_UL)   # the water cap counts CONSUMED water
session_start_time = time.time()
session_end_reason = None

for trial in range(1, VAR_MAX_TRIALS + 1):
    try:
        # J1 lives INSIDE the quiescence requirement (per explicit instruction: the dot may only
        # appear after quiescence, and the wheel is zeroed the moment it does): the wheel must stay
        # still for VAR_QUIESCENCE_S + J1. The whole trial -- hold, dot, response, ITI -- is ONE
        # Bpod machine (see _shared/trial_render.py for the PC side of the handshake), so licks
        # are logged and Stop/Kill is handled during the hold too, and the reward window can only
        # open once the dot is actually on screen.
        quiescence_hold_s = VAR_QUIESCENCE_S + np.random.uniform(VAR_DOT_ONSET_JITTER_MIN_S,
                                                                  VAR_DOT_ONSET_JITTER_MAX_S)
        disappear_delay_s = np.random.uniform(VAR_DOT_DISAPPEAR_MIN_S, VAR_DOT_DISAPPEAR_MAX_S)

        sma = StateMachine(my_bpod)

        sma.add_state(
            state_name='WaitQuiescence',
            state_timer=VAR_WAIT_QUIESCENCE_MAX_S,
            state_change_conditions={PC_EVENT_QUIESCENCE_MET: 'DotOn', Bpod.Events.Tup: 'exit'},
            output_actions=[])

        # The PC draws the dot on SOFTCODE_DOT_ON and answers once it is painted. Thresholds were
        # re-armed (relative to the still wheel) when quiescence was met, so a turn here can't be
        # swallowed -- it ends the trial as MovedBeforeDot.
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

        # Entered the moment the dot is confirmed on screen: one relay message re-zeroes the rotary
        # and sends it the '#' marker (this instant on the rotary's own clock), so the dot's onset,
        # the wheel's zero and the start of response collection are the same instant.
        sma.add_state(
            state_name='WheelPeriod',
            state_timer=VAR_RESPONSE_TIMEOUT_S,
            state_change_conditions={
                neg_event: 'Reward',
                pos_event: 'Reward',
                Bpod.Events.Tup: 'NoMovement',
            },
            output_actions=[(rotary_channel, reset_positions_trigger_id)])

        sma.add_state(
            state_name='Reward',
            state_timer=VAR_REWARD_DURATION,
            state_change_conditions={Bpod.Events.Tup: 'ITI'},
            output_actions=[(Bpod.OutputChannels.Valve, 1), ('SoftCode', SOFTCODE_REWARD)])

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
            state_timer=VAR_ITI_S,
            state_change_conditions={Bpod.Events.Tup: 'exit'},
            output_actions=[])

        runner.upload(sma)

        # Thread-inverted, same pattern as full_protocol_lookback_test.py: the Bpod call runs in a
        # background thread; this (main) thread renders the dot, since Qt's event loop must run on
        # the thread that owns the QApplication.
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
            render_interval, SOFTCODE_DOT_ON, (SOFTCODE_REWARD,),
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
        # TrialRunner.run_trial_state_machine()). Offline, session_csv_parser.fit_host_to_bpod()
        # maps the whole host clock onto Bpod's own (TrialStartTimestamp) clock from these.
        send_t = runner.last_run_t
        n_breaks = runner.quiescence_breaks
        runner.register('SM_RUN_TIME', send_t)
        register_trial_timing(runner, render, quiescence_hold_s, n_breaks, prev_return_t,
                              my_bpod.session.current_trial.states_durations, send_t)

        visited = my_bpod.session.current_trial.states_durations
        rewarded = was_visited(visited, 'Reward')
        early = next((s for s in ('MovedBeforeDot', 'DotTimeout') if was_visited(visited, s)), None)
        outcome_state = 'Reward' if rewarded else (early or 'NoMovement')
        outcome = 'Rewarded' if rewarded else (early or 'NoMovement')

        # Per explicit instruction: only genuinely REWARDED trials count toward the qualifying/
        # session-cap threshold -- a NoMovement trial doesn't demonstrate the behavior this
        # threshold exists to confirm, so it shouldn't count toward reaching it.
        if rewarded:
            session_trial_count += 1
            session_reward_ul_total += VAR_REWARD_UL

        # Side = the crossing that actually entered Reward (bpod_trial_helpers.triggering_event()),
        # not the first/last crossing in the trial -- the old rule mislabeled every trial where an
        # ignored crossing came before the rewarded one.
        choice_events = my_bpod.session.current_trial.get_all_timestamps_by_event()
        side = None
        if rewarded:
            trigger = triggering_event(choice_events, visited['Reward'][-1][0],
                                       [neg_event, pos_event])
            side = {neg_event: 'L', pos_event: 'R'}.get(trigger)
        magnitude_deg = cur_threshold_deg if rewarded else 0.0

        runner.register('TRIAL_TYPE', 'stage1')
        runner.register('MOVEMENT_DIRECTION', side if side is not None else 'none')
        runner.register('OUTCOME', outcome)
        runner.register('GAIN_MULT', cur_gain_mult)
        if rewarded:
            runner.register('THRESHOLD_CROSSING_TIME', send_t + visited[outcome_state][-1][0])

        print("Trial {0}: held {1:.2f}s (broke {2}x), {3}{4}, gain={5:.2f}x".format(
            trial, quiescence_hold_s, n_breaks, outcome,
            " ({0})".format(side) if side else "", cur_gain_mult), flush=True)

        # Port1In/Port1Out licks are already natively logged as raw EVENT rows in the session CSV
        # (see CLAUDE.md) -- register_port_events() also explicitly registers both edges as their
        # own VALs (per explicit instruction: always report port in AND out times, not just in --
        # useful on its own, and makes per-contact duration directly auditable without hand-
        # parsing raw EVENT rows, relevant to a suspected sensor debounce/threshold issue).
        lick_times_abs, _lick_times_out = runner.register_port_events('LICK', choice_events, send_t)
        water.record_trial(choice_events.get('Port1In', []), visited['Reward'][-1][0] if rewarded else None)
        runner.register('SESSION_WATER_CONSUMED_UL', water.consumed_ul)
        reward_time_abs = send_t + visited['Reward'][-1][0] if rewarded else None

        # NoMovement has no side; 'R' is arbitrary there and only colors the raster row.
        bench_plots.add_trial(side or 'R', magnitude_deg, cur_threshold_deg,
                               'Abort' if early else outcome,
                               gain_mult=cur_gain_mult, lick_times_abs=lick_times_abs,
                               reward_time_abs=reward_time_abs)

        # Deliberate session cap, per explicit instruction: stop at VAR_TRIAL_CAP ATTEMPTED
        # trials (not the separate, smaller VAR_TRIAL_COUNT_ADVANCE qualifying-count threshold),
        # 1.5mL CONSUMED (licked after delivery), or 30 minutes elapsed -- whichever comes first. VAR_MAX_TRIALS (400)
        # above is only a further-out safety backstop, not the real target. Checked at trial end
        # (not trial start) so the trial that actually crosses a cap still completes and gets
        # logged.
        if trial >= VAR_TRIAL_CAP:
            session_end_reason = 'trial_count_cap'
        elif water.consumed_ul >= VAR_SESSION_WATER_CAP_UL:
            session_end_reason = 'water_cap'
        elif (time.time() - session_start_time) >= VAR_SESSION_DURATION_CAP_S:
            session_end_reason = 'duration_cap_30min'
        if session_end_reason is not None:
            print("Session cap reached ({0} trials attempted, {1} rewarded, {2:.1f}uL consumed of {3:.1f}uL "
                  "delivered, {4:.1f}min elapsed) -- ending session.".format(
                      trial, session_trial_count, water.consumed_ul, session_reward_ul_total,
                      (time.time() - session_start_time) / 60.0), flush=True)
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

# --- session-end bookkeeping: advancement gate + gain decay + threshold growth ----------------------

# >= , not > -- the session's own hard cap is now VAR_TRIAL_CAP (a separate, larger number), not
# this qualifying threshold, so session_trial_count can legitimately exceed VAR_TRIAL_COUNT_ADVANCE
# by the time the session actually ends; >= still correctly recognizes "met or exceeded."
cleared_this_session = session_trial_count >= VAR_TRIAL_COUNT_ADVANCE
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

_cleanup_and_export()

print("Close the plot window to exit.", flush=True)
plt.ioff()
plt.show()
