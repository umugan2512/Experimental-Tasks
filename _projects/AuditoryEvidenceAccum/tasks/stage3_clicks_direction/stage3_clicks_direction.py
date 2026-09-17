# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""
training_protocol.md Revision 2, Stage 3 -- "clicks and direction".

Aim: the click-side association -- the first moment clicks and direction both appear together.
0.25s Poisson clicks at gamma=+-1.0 (all clicks on the rewarded side, i.e. click_train_v2's 'AOS'
grid entry) -> a short fixed 100-200ms cue-to-go delay (Tag A) -> LED + dot (centred, appears after
a further J1 jitter) -> turn toward the click side for reward (ITI 1.5s), away for no reward (ITI
4.0s). Quiescence (100ms) and response threshold (frozen at whatever fraction Stage 2 ended on) do
NOT restaircase here -- both resume at Stage 4. An in-trial wheel-excursion threshold is enforced
for the FIRST time, monitored through CuePeriod+DelayPeriod (mirrors
Tests/tasks/full_protocol_lookback_test/full_protocol_lookback_test.py's existing wheel-abort
pattern exactly, including its two-state-machine/thread-inversion structure) -- crossing it aborts
the trial, which is then automatically RE-OFFERED (same side, TRIAL_TYPE='repeat', after a 1-2s
timeout) rather than simply dropped, per SS3.3.

Click level starts ~6dB below final for this subject's first VAR_CLICK_ATTENUATION_SESSIONS
sessions (persisted counter, decremented once per session run regardless of trial count), then
switches to full level -- an additive amplitude_scale on click_train_v2.build_waveform(), not a
change to its ISI/generation math. The in-trial threshold itself starts at a PLACEHOLDER value
(see VAR_CUE_ABORT_THRESHOLD_DEG_PLACEHOLDER below) since the doc's own empirical-derivation
procedure (derive_intrial_threshold.py) only runs at the END of Stage 4 -- replace the persisted
'in_trial_threshold_deg' once that has run at least once for this subject.

Every session opens with a warm-up block (20-40 trials, drawn fresh each session, TRIAL_TYPE=
'warmup') excluded from the accuracy/abort-rate stats that feed advancement. After an error, the
SAME side is re-presented with high probability (debiasing.next_side_after_error(), a small
literal implementation of the doc's own rule -- deliberately NOT trial_scheduler.py's more
elaborate lookback machinery, which implements a different policy). A session ends at the first
of: the 90-minute cap, 3 continuous minutes with no trial initiated, 20 consecutive aborts, or
sub-chance accuracy across 3 consecutive 20-trial blocks (SS3.2) -- SESSION_END_REASON records
which. Advance after three consecutive sessions meeting staircase.stage3_gates_met() (>70% correct
at gamma=+-1.0, abort rate <20%, >200 main trials) -- unlike Stage 2's gate, this covers the WHOLE
doc-specified criterion, no deferred statistical test.

Tag H (day-5 triage) and Tag J (non-learner checklist: per-rig speaker SPL calibration, ear
inspection, supra-threshold click-train response check) are human judgment calls printed as
reminders at session end, same as Stage 2's own deferred-statistical-test reminder -- see
STAGE3_TEST_PLAN.md for the full manual checklist. Stage 3.5 (asymmetric response thresholds,
"held in reserve, deployed only by Stage-3 triage") is NOT built here -- see
VAR_ASYMMETRIC_THRESHOLD_ENABLED below, an unused hook for a future toggle.

Hardware/module connection plumbing, thread-inversion render loop, and camera integration are all
reused as-is from _shared/ and mirror full_protocol_lookback_test.py's already-hardware-validated
patterns exactly (including that file's own additions.txt instrumentation pass) -- see this
project's stage1_wheel_shaping.py/stage2_threshold_staircase.py for the same conventions applied
to this project's own persisted-state/subject-derivation machinery.

Deliberately NOT touched/reimplemented here: click_train_v2.py's ISI floor, rate calibration, or
Poisson generation math (only its new, additive stim_duration_s/amplitude_scale parameters are
used); trial_scheduler.py's lookback machinery. additions.txt's T9 monotonic-clock item is
deliberately NOT applied -- every timestamp in this file uses the same time.time() - log_python_t0
convention as the rest of the codebase, for consistency; its CSV-flush-per-trial item needs no
action here -- pybpodapi's own Session.write() already flushes on every row (confirmed by reading
pybpod/base/pybpod-api/pybpodapi/session.py), so no submodule change was needed.
"""
import os
import random
import subprocess
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
import staircase
import session_state
import session_struct_export
import session_csv_parser
import debiasing
import click_train_v2 as click_train
from wheel_shaping_plots import WheelShapingPlots
from bpod_trial_helpers import TrialRunner, was_visited
import rotary_setup
import hifi_setup
from dot_display import DotDisplay
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
    skipped regardless of how the session ended -- see stage1_wheel_shaping.py's own identical
    helper for the fixed exception-path gap this pattern avoids. """
    try:
        runner.register('SESSION_WATER_UL', session_reward_count * VAR_REWARD_UL)
    except Exception as err:
        print("WARNING: could not register SESSION_WATER_UL: {0}".format(err), flush=True)
    camera.close()
    dot.close()
    hifi.close()
    rotary.close()
    csv_path = my_bpod.session._path   # grab before close() -- close() deletes the Session object
                                        # that holds it
    my_bpod.close()
    _export_session_struct(csv_path)


def _git_commit():
    """ additions.txt T9: best-effort git commit hash (+dirty flag) for provenance, registered as
    a session-level VAL. Never blocks the task -- returns 'unknown' if git isn't available or this
    checkout isn't a git repo (e.g. a zipped deployment). """
    try:
        commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=_TASK_DIR,
            stderr=subprocess.DEVNULL).decode('utf-8').strip()
        dirty = subprocess.call(
            ['git', 'diff', '--quiet'], cwd=_TASK_DIR,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) != 0
        return '{0}{1}'.format(commit, '-dirty' if dirty else '')
    except Exception:
        return 'unknown'


# --- who this session is for -----------------------------------------------------------------------

_raw_subjects = getattr(settings, 'PYBPOD_SUBJECTS', None)
if not _raw_subjects:
    raise RuntimeError(
        "No subject selected in the GUI for this session (PYBPOD_SUBJECTS is empty) -- select a "
        "subject before running this task.")
VAR_SUBJECT_ID = session_csv_parser.parse_subject_name(_raw_subjects[0])
VAR_PROJECT_DIR = os.path.abspath(os.path.join(_TASK_DIR, '..', '..'))

# --- stage parameters (training_protocol.md Revision 2, Stage 3) -------------------------------------

VAR_MAX_TRIALS = 1000                # hard-ceiling safety backstop, well above what a 90-min
                                      # session at this stage's pace should ever reach
VAR_QUIESCENCE_S = 0.1               # frozen through Stage 3 (doc: resumes staircasing at Stage 4)
VAR_STEADY_THRESHOLD_DEG = 4         # IBL convention, same as every other task -- hold-still deadzone
VAR_RESPONSE_TIMEOUT_S = 5.0         # doc's own epoch table: response window <=5s

VAR_THRESHOLD_FINAL_DEG = 35         # matches this project's established final-task convention
VAR_TRIAL_COUNT_ADVANCE = 200        # doc: ">200 trials/session" (MAIN trials only -- warmup/
                                      # repeat excluded, see session_trial_count below)

VAR_STIM_DURATION_S = 0.25           # doc: "0.25s Poisson clicks" -- overrides click_train_v2's
                                      # own 2.0s module default via its additive stim_duration_s
                                      # parameter (generation/ISI math itself untouched).
VAR_DELAY_DURATION_MIN_S = 0.1       # Tag A: cue-to-go gap, 100-200ms FIXED range at Stage 3
VAR_DELAY_DURATION_MAX_S = 0.2       # (staircased 0-750ms only from Stage 6 onward, not built here)
VAR_ABORT_TIMEOUT_MIN_S = 1.0        # SS3.3: abort re-offer timeout before the same trial repeats
VAR_ABORT_TIMEOUT_MAX_S = 2.0

VAR_CUE_ABORT_THRESHOLD_DEG_PLACEHOLDER = 2.5 * VAR_STEADY_THRESHOLD_DEG   # 10deg -- PLACEHOLDER
                                      # only. Reused from full_protocol_lookback_test.py's own
                                      # hardware-precedented value since the doc's real derivation
                                      # procedure (derive_intrial_threshold.py) only runs at the
                                      # END of Stage 4. Replace the persisted 'in_trial_threshold_deg'
                                      # once that has run at least once for this subject.

VAR_ITI_S = 1.5                      # correct-trial ITI
VAR_INCORRECT_ITI_S = 4.0            # error-trial ITI -- longer, per the doc's epoch table
VAR_CONSUMPTION_WINDOW_S = 3.0       # not doc-specified for this stage -- same flagged/tunable
                                      # placeholder convention as every other unspecified duration
                                      # in this codebase; matches full_protocol_lookback_test.py's
                                      # own value.

VAR_REWARD_UL = 4.0                  # target reward volume -- valve open time is derived from this
                                      # via Calibration/liquid_calibration.py's own fitted curve
VAR_REWARD_DURATION = get_reward_duration_s(VAR_REWARD_UL)

VAR_DOT_ONSET_JITTER_MIN_S = 0.1     # J1 -- unchanged from every earlier stage
VAR_DOT_ONSET_JITTER_MAX_S = 0.2
VAR_DOT_DISAPPEAR_MIN_S = 0.4        # J2 -- unchanged
VAR_DOT_DISAPPEAR_MAX_S = 0.9

VAR_DOT_SCREEN_INDEX = 1
VAR_DOT_DIAMETER_PX = 60             # same unconfirmed-placeholder flag as every dot-stimulus task
VAR_DOT_BACKGROUND_GRAY = 128
VAR_DOT_GRAY = 0
VAR_DOT_EDGE_FRACTION = 0.9
VAR_RENDER_HZ = 30

VAR_CLICK_ATTENUATION_DB = 6.0       # doc: "start ~6dB below final"
VAR_CLICK_ATTENUATION_SESSIONS = 2   # doc: "for 2-3 sessions" -- picked 2, tunable; decremented
                                      # once per session RUN (not per qualifying session)

VAR_WARMUP_TRIAL_MIN = 20            # doc SS3.1 (Tag D): 20-40 trials at the easiest level,
VAR_WARMUP_TRIAL_MAX = 40            # drawn fresh each session, excluded from advancement stats

VAR_SESSION_CAP_S = 90 * 60          # doc SS3.2 (Tag E): 90-minute cap
VAR_NO_INIT_TIMEOUT_S = 3 * 60       # ...or 3 continuous minutes with no trial initiated
VAR_CONSECUTIVE_ABORT_LIMIT = 20     # ...or 20 consecutive aborts
VAR_SUBCHANCE_BLOCK_SIZE = 20        # ...or sub-chance accuracy across 3 consecutive 20-trial
VAR_SUBCHANCE_BLOCKS = 3             # blocks at the easiest level (every Stage-3 trial, here)

VAR_ASYMMETRIC_THRESHOLD_ENABLED = False   # Stage 3.5 scaffold hook ("held in reserve, deployed
                                      # only by Stage-3 triage") -- NOT built, per the doc's own
                                      # framing; this flag exists only so a future implementation
                                      # has an obvious place to hook in, not as a working feature.

VAR_ROTARY_USB_PORT = None
VAR_STILL_POLL_HZ = 100              # additions.txt T4: NEEDS ON-RIG CONFIRMATION that the poll
VAR_POLL_HZ = 100                    # thread keeps up without delaying state-machine handling --
                                      # if not, fall back to 50/50 and note which rate was used.

VAR_GO_CUE_LED_CHANNEL = 'PWM1'

VAR_CAMERA_INDEX = None
VAR_CAMERA_OUTPUT_PATH = 'session_video.avi'
VAR_CAMERA_FPS = 30.0
VAR_CAMERA_PREVIEW = True             # preview shown only in short snippets (see
                                       # camera.show_snippet() calls below), never continuously.
VAR_CAMERA_SNIPPET_S = 1.0

# --- persisted cross-session state (same file as Stage 1/2) ------------------------------------------

state = session_state.StageState(VAR_PROJECT_DIR, VAR_SUBJECT_ID, defaults={
    'threshold_fraction': 1.0,        # Stage 2's own ceiling, in case this subject somehow never
                                       # ran Stage 2 -- normally already set to Stage 2's real
                                       # ending value by the time Stage 3 first runs.
    'in_trial_threshold_deg': VAR_CUE_ABORT_THRESHOLD_DEG_PLACEHOLDER,
    'click_attenuation_sessions_remaining': VAR_CLICK_ATTENUATION_SESSIONS,
    'sessions_trial_count_history': [],   # shared list name with Stage 1/2 -- harmless, unused by
                                           # this stage's own advancement gate (stage3_sessions_history)
    'stage3_sessions_history': [],        # bools: did each past Stage-3 session meet
                                           # staircase.stage3_gates_met()? (3-consecutive check)
})

VAR_RESPONSE_THRESHOLD_DEG = state.get('threshold_fraction') * VAR_THRESHOLD_FINAL_DEG
VAR_CUE_ABORT_THRESHOLD_DEG = state.get('in_trial_threshold_deg')
_click_attenuation_remaining = state.get('click_attenuation_sessions_remaining')

# --- connect to Bpod, resolve modules ----------------------------------------------------------------

my_bpod = Bpod()
print("Connected to Bpod on {0}".format(my_bpod.serial_port), flush=True)

rotary, rotary_bpod_module = rotary_setup.connect_rotary(my_bpod, usb_port=VAR_ROTARY_USB_PORT)
reset_positions_trigger_id, rotary_channel = rotary_setup.build_reset_trigger(rotary_bpod_module)

# 6 thresholds -- same ordering convention as full_protocol_lookback_test.py: steady (hold-still),
# response (choice), cue-abort (in-trial) -- index order determines event numbering.
ALL_THRESHOLDS_DEG = [-VAR_STEADY_THRESHOLD_DEG, VAR_STEADY_THRESHOLD_DEG,
                       -VAR_RESPONSE_THRESHOLD_DEG, VAR_RESPONSE_THRESHOLD_DEG,
                       -VAR_CUE_ABORT_THRESHOLD_DEG, VAR_CUE_ABORT_THRESHOLD_DEG]
event_names = rotary_setup.set_and_enable_thresholds(rotary, ALL_THRESHOLDS_DEG)
rotary.enable_evt_transmission()
left_event, right_event = event_names[2], event_names[3]
wheel_abort_event_neg, wheel_abort_event_pos = event_names[4], event_names[5]

hifi = hifi_setup.connect_hifi(my_bpod)
hifi_stop_msg_id, hifi_channel = hifi_setup.build_stop_trigger(my_bpod)

my_bpod.register_value('LEFT_THRESHOLD_DEG', -VAR_RESPONSE_THRESHOLD_DEG)
my_bpod.register_value('RIGHT_THRESHOLD_DEG', VAR_RESPONSE_THRESHOLD_DEG)
my_bpod.register_value('GIT_COMMIT', _git_commit())

log_python_t0 = time.time()
runner = TrialRunner(my_bpod, rotary, log_python_t0, still_poll_hz=VAR_STILL_POLL_HZ,
                      poll_hz=VAR_POLL_HZ)

dot = DotDisplay(screen_index=VAR_DOT_SCREEN_INDEX, diameter_px=VAR_DOT_DIAMETER_PX,
                  background_gray=VAR_DOT_BACKGROUND_GRAY, dot_gray=VAR_DOT_GRAY)
dot.show()
dot.clear()

camera = CameraRecorder(log_python_t0, camera_index=VAR_CAMERA_INDEX,
                         output_path=VAR_CAMERA_OUTPUT_PATH, fps=VAR_CAMERA_FPS,
                         preview=VAR_CAMERA_PREVIEW)
camera.start()

my_bpod.register_value('CAMERA_START_TIME', camera.start_time_s)
my_bpod.register_value('CAMERA_OUTPUT_PATH', camera.output_path)
my_bpod.register_value('CAMERA_FPS', camera.fps)

# Geometry-aware gain (same derivation as every other dot-coupled script), signed via
# rotary_setup.screen_direction_gain() for this rig's confirmed wheel->screen direction.
screen_width_px = dot.get_screen_width_px()
dot_gain = rotary_setup.screen_direction_gain(
    (VAR_DOT_EDGE_FRACTION * (screen_width_px / 2.0)) / VAR_RESPONSE_THRESHOLD_DEG)
dot.set_deg_to_px_gain(dot_gain)
print("Dot gain calibrated to {0:.2f} px/wheel-deg (screen width {1}px, response threshold "
      "{2:.1f}deg)".format(dot_gain, screen_width_px, VAR_RESPONSE_THRESHOLD_DEG), flush=True)

render_interval = 1.0 / VAR_RENDER_HZ

bench_plots = WheelShapingPlots(
    stage=3, threshold_final_deg=VAR_THRESHOLD_FINAL_DEG,
    prev_session_values={'threshold_deg': VAR_RESPONSE_THRESHOLD_DEG},
    session_status={'click_attenuated': _click_attenuation_remaining > 0,
                     'in_trial_threshold_deg': VAR_CUE_ABORT_THRESHOLD_DEG})

# --- trial loop -----------------------------------------------------------------------------------

print("Starting Stage 3 -- response threshold frozen at {0:.1f}deg, in-trial threshold {1:.1f}deg"
      "{2}, click level {3}".format(
          VAR_RESPONSE_THRESHOLD_DEG, VAR_CUE_ABORT_THRESHOLD_DEG,
          ' (PLACEHOLDER -- not yet empirically derived)'
          if VAR_CUE_ABORT_THRESHOLD_DEG == VAR_CUE_ABORT_THRESHOLD_DEG_PLACEHOLDER else '',
          'ATTENUATED (-{0:.0f}dB, {1} session(s) remaining)'.format(
              VAR_CLICK_ATTENUATION_DB, _click_attenuation_remaining)
          if _click_attenuation_remaining > 0 else 'FULL'), flush=True)

session_start_time = time.time()
last_initiation_time = session_start_time
session_reward_count = 0
session_trial_count = 0        # MAIN trials only -- feeds VAR_TRIAL_COUNT_ADVANCE
total_attempts = 0              # every trial attempt (warmup+main+repeat), feeds abort_rate
total_aborts = 0
consecutive_aborts = 0
main_trial_correct = []         # bools, MAIN trials only -- feeds accuracy_aos + sub-chance check
warmup_target = random.randint(VAR_WARMUP_TRIAL_MIN, VAR_WARMUP_TRIAL_MAX)
fresh_trial_count = 0           # non-repeat trial draws so far this session

prev_outcome = None             # 'correct'/'incorrect'/None -- drives debiasing
prev_side = None
pending_repeat = False
repeat_side = None

session_end_reason = None
trial = 0

while True:
    now = time.time()
    if now - session_start_time >= VAR_SESSION_CAP_S:
        session_end_reason = 'cap_90min'
        break
    if now - last_initiation_time >= VAR_NO_INIT_TIMEOUT_S:
        session_end_reason = 'no_initiation_3min'
        break
    if consecutive_aborts >= VAR_CONSECUTIVE_ABORT_LIMIT:
        session_end_reason = 'consecutive_aborts'
        break
    if len(main_trial_correct) >= VAR_SUBCHANCE_BLOCK_SIZE * VAR_SUBCHANCE_BLOCKS:
        recent = main_trial_correct[-(VAR_SUBCHANCE_BLOCK_SIZE * VAR_SUBCHANCE_BLOCKS):]
        blocks = [recent[i:i + VAR_SUBCHANCE_BLOCK_SIZE]
                  for i in range(0, len(recent), VAR_SUBCHANCE_BLOCK_SIZE)]
        if all((sum(b) / float(len(b))) < 0.5 for b in blocks):
            session_end_reason = 'subchance_accuracy'
            break
    if trial >= VAR_MAX_TRIALS:
        session_end_reason = 'max_trials'
        break

    trial += 1
    try:
        if pending_repeat:
            trial_type = 'repeat'
            side = repeat_side
            pending_repeat = False
        else:
            fresh_trial_count += 1
            trial_type = 'warmup' if fresh_trial_count <= warmup_target else 'main'
            if prev_outcome == 'incorrect':
                side = debiasing.next_side_after_error(prev_side)
            else:
                side = np.random.choice(['L', 'R'])

        rotary.disable_evt_transmission()
        n_breaks = runner.wait_for_held_steady(VAR_QUIESCENCE_S, VAR_STEADY_THRESHOLD_DEG,
                                                require_no_lick=False)
        if n_breaks is None:
            print("Bpod stopped running trials (Stop/Kill) -- ending session early after trial "
                  "{0}.".format(trial), flush=True)
            break
        trial_start_t = time.time() - log_python_t0
        last_initiation_time = time.time()

        with runner.rotary_lock:
            rotary.set_zero_position()
            rotary_setup.set_and_enable_thresholds(rotary, ALL_THRESHOLDS_DEG)
        rotary.enable_evt_transmission()

        runner.register('TRIAL_START', trial_start_t)
        runner.register('QUIESCENCE_BREAKS', n_breaks)
        runner.register('TRIAL_TYPE', trial_type)
        dot.clear()
        dot.pump()
        camera.pump()

        # additions.txt T1: fresh per-trial seed, not a globally reseeded generator.
        stim_seed = int(np.random.randint(0, 2 ** 31 - 1))
        trial_clicks = click_train.generate_trial_clicks(
            'AOS', side, rng=np.random.RandomState(stim_seed), stim_duration_s=VAR_STIM_DURATION_S)
        amplitude_scale = (10 ** (-VAR_CLICK_ATTENUATION_DB / 20.0)
                            if _click_attenuation_remaining > 0 else 1.0)
        left_wave, right_wave = click_train.build_waveform(
            trial_clicks, hifi.sampling_rate, amplitude_scale=amplitude_scale,
            stim_duration_s=VAR_STIM_DURATION_S)
        hifi.load(0, np.array([left_wave, right_wave]))
        hifi.push()

        runner.register('TRIAL_SIDE', side)
        runner.register('CLICK_TIMES_L', ','.join(
            '{0:.4f}'.format(t) for t in trial_clicks['left_times']))
        runner.register('CLICK_TIMES_R', ','.join(
            '{0:.4f}'.format(t) for t in trial_clicks['right_times']))
        runner.register('N_CLICKS_L', trial_clicks['n_left'])
        runner.register('N_CLICKS_R', trial_clicks['n_right'])
        runner.register('STIM_SEED', stim_seed)
        runner.register('CUE_ABORT_THRESHOLD_DEG', VAR_CUE_ABORT_THRESHOLD_DEG)
        runner.register('RESPONSE_THRESHOLD_DEG', VAR_RESPONSE_THRESHOLD_DEG)
        runner.register('QUIESCENCE_DUR_S', VAR_QUIESCENCE_S)
        runner.register('ITI_S', VAR_ITI_S)
        runner.register('CLICK_LEVEL_ATTENUATED', _click_attenuation_remaining > 0)

        print("Trial {0} ({1}): side={2}, n_L={3}, n_R={4}".format(
            trial, trial_type, side, trial_clicks['n_left'], trial_clicks['n_right']), flush=True)

        hifi.play(0)

        cue_send_t = time.time() - log_python_t0
        runner.register('SM_SEND_TIME', cue_send_t)
        runner.register('CUE_ONSET_TIME', cue_send_t + click_train.CLICK_START_OFFSET_S)

        delay_duration_this_trial = random.uniform(VAR_DELAY_DURATION_MIN_S,
                                                     VAR_DELAY_DURATION_MAX_S)
        abort_timeout_this_trial = random.uniform(VAR_ABORT_TIMEOUT_MIN_S, VAR_ABORT_TIMEOUT_MAX_S)

        cue_sma = StateMachine(my_bpod)

        cue_sma.add_state(
            state_name='CuePeriod',
            state_timer=click_train.CLICK_START_OFFSET_S + VAR_STIM_DURATION_S,
            state_change_conditions={
                wheel_abort_event_neg: 'WheelAbort',
                wheel_abort_event_pos: 'WheelAbort',
                Bpod.Events.Tup: 'DelayPeriod',
            },
            output_actions=[])

        cue_sma.add_state(
            state_name='DelayPeriod',
            state_timer=delay_duration_this_trial,
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
            state_timer=abort_timeout_this_trial,
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

        total_attempts += 1

        if aborted:
            total_aborts += 1
            consecutive_aborts += 1
            abort_time_abs = cue_send_t + cue_visited['WheelAbort'][-1][0]
            abort_epoch = 'delay' if was_visited(cue_visited, 'DelayPeriod') else 'cue'
            runner.register('ABORT', abort_time_abs)
            runner.register('ABORT_EPOCH', abort_epoch)
            # 'INCLUDED' means: did this trial produce a genuine L/R response (Reward or
            # ErrorConsumption)? False here (and for NoResponse) -- analysis applies its own
            # filters downstream, this is not a second, redundant filter.
            runner.register('INCLUDED', False)

            pending_repeat = True
            repeat_side = side

            print("Trial {0}: Abort ({1} epoch) -- re-offering side {2} next trial.".format(
                trial, abort_epoch, side), flush=True)

            bench_plots.add_trial(side, 0.0, VAR_RESPONSE_THRESHOLD_DEG, 'Abort',
                                   lick_times_abs=cue_lick_times_abs, trial_type=trial_type,
                                   abort_epoch=abort_epoch,
                                   click_times_l=trial_clicks['left_times'],
                                   click_times_r=trial_clicks['right_times'])
            continue

        consecutive_aborts = 0

        rewarded_event = right_event if side == 'R' else left_event
        unrewarded_event = left_event if rewarded_event == right_event else right_event

        dot_onset_delay = random.uniform(VAR_DOT_ONSET_JITTER_MIN_S, VAR_DOT_ONSET_JITTER_MAX_S)
        disappear_delay_s = random.uniform(VAR_DOT_DISAPPEAR_MIN_S, VAR_DOT_DISAPPEAR_MAX_S)

        send_epoch = time.time()
        send_t = send_epoch - log_python_t0
        runner.register('SM_SEND_TIME', send_t)

        camera.show_snippet(VAR_CAMERA_SNIPPET_S)

        sma = StateMachine(my_bpod)

        sma.add_state(
            state_name='WheelDotPeriod',
            state_timer=VAR_RESPONSE_TIMEOUT_S,
            state_change_conditions={
                rewarded_event: 'Reward',
                unrewarded_event: 'ErrorConsumption',
                Bpod.Events.Tup: 'NoResponse',
            },
            output_actions=[(rotary_channel, reset_positions_trigger_id),
                             (VAR_GO_CUE_LED_CHANNEL, 255)])

        sma.add_state(
            state_name='Reward',
            state_timer=VAR_REWARD_DURATION,
            state_change_conditions={Bpod.Events.Tup: 'Consumption'},
            output_actions=[(Bpod.OutputChannels.Valve, 1), (VAR_GO_CUE_LED_CHANNEL, 0)])

        sma.add_state(
            state_name='Consumption',
            state_timer=VAR_CONSUMPTION_WINDOW_S,
            state_change_conditions={'Port1In': 'ITI', Bpod.Events.Tup: 'ITI'},
            output_actions=[])

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
            state_timer=VAR_ITI_S,
            state_change_conditions={Bpod.Events.Tup: 'exit'},
            output_actions=[])

        # Thread-inverted, same pattern as full_protocol_lookback_test.py.
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
                if pos <= -VAR_RESPONSE_THRESHOLD_DEG or pos >= VAR_RESPONSE_THRESHOLD_DEG:
                    frozen = True
                    frozen_at = now
                else:
                    dot.set_position_deg(pos)

            if frozen and (now - frozen_at) >= disappear_delay_s:
                dot.clear()

            dot.pump()
            camera.pump()
            time.sleep(render_interval)

        decision_thread.join(timeout=3.0)
        if decision_thread.is_alive():
            print("WARNING: decision-period Bpod thread did not stop within 3s -- continuing "
                  "anyway.", flush=True)

        dot.clear()
        dot.pump()

        camera.show_snippet(VAR_CAMERA_SNIPPET_S)
        camera.pump()

        if decision_result.get('killed'):
            print("Bpod Kill received -- ending session.", flush=True)
            _cleanup_and_export()
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

        if outcome_state == 'Reward':
            response = side
            session_reward_count += 1
        elif outcome_state == 'ErrorConsumption':
            response = 'L' if side == 'R' else 'R'
        else:
            response = None
        correct = rewarded if outcome_state in ('Reward', 'ErrorConsumption') else None

        prev_outcome = 'correct' if correct else ('incorrect' if correct is False else None)
        prev_side = side

        # accuracy_aos (session end) is conditional on a genuine L/R response -- a NoResponse
        # trial contributes to neither the numerator nor denominator here (it's already a
        # separately-tracked engagement signal via VAR_NO_INIT_TIMEOUT_S/consecutive_aborts, not
        # folded into "accuracy," matching the same "correct is not None" gate INCLUDED uses).
        if trial_type == 'main' and correct is not None:
            session_trial_count += 1
            main_trial_correct.append(bool(correct))

        consumption_state_name = {'Reward': 'Consumption',
                                   'ErrorConsumption': 'ErrorConsumption'}.get(outcome_state)
        consumption_licked = None
        if consumption_state_name is not None:
            c_start, c_end = visited[consumption_state_name][-1]
            consumption_licked = any(c_start <= t <= c_end for t in choice_events.get('Port1In', []))

        # 'INCLUDED' means: did this trial produce a genuine L/R response (Reward or
        # ErrorConsumption)? False for NoResponse/Abort -- analysis applies its own filters
        # downstream, this is not a second, redundant filter.
        runner.register('INCLUDED', response is not None)
        runner.register('CONSUMPTION_LICKED', consumption_licked)

        led_on_t = send_t
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

        print("Trial {0} ({1}): side={2}, {3}{4}".format(
            trial, trial_type, side, outcome, ' -> reward' if rewarded else ''), flush=True)

        choice_lick_times_abs = [send_t + t for t in choice_events.get('Port1In', [])]
        lick_times_abs = sorted(cue_lick_times_abs + choice_lick_times_abs)

        reward_time_abs = None
        if rewarded:
            reward_time_abs = send_t + visited['Reward'][-1][0]

        # WheelDotPeriod is state index 0 of `sma`, so its own trial-relative outcome timestamp IS
        # already "time since go-cue" -- no extra send_t arithmetic needed.
        response_time_s = (visited[outcome_state][-1][0]
                            if outcome_state in ('Reward', 'ErrorConsumption') else None)

        bench_plots.add_trial(side, VAR_RESPONSE_THRESHOLD_DEG if response is not None else 0.0,
                               VAR_RESPONSE_THRESHOLD_DEG, outcome,
                               lick_times_abs=lick_times_abs, reward_time_abs=reward_time_abs,
                               trial_type=trial_type, response_time_s=response_time_s,
                               click_times_l=trial_clicks['left_times'],
                               click_times_r=trial_clicks['right_times'])

    except Exception as err:
        print("Trial {0} FAILED: {1}".format(trial, err), flush=True)
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        _cleanup_and_export()
        raise

# SESSION_END_REASON is only registered for the four deliberate, explicitly-detected end
# conditions above (cap/no-init/consecutive-aborts/subchance/max-trials) -- a Stop/Kill/crash
# mid-trial break leaves session_end_reason as None and registers nothing, same "absence signals
# an abnormal/early end" convention already established by full_protocol_lookback_test.py/Stage 1/2.
if session_end_reason is not None:
    runner.register('SESSION_END_REASON', session_end_reason)
print("Session ending: {0}".format(session_end_reason or 'Stop/Kill/crash (see log above)'),
      flush=True)

# --- session-end bookkeeping: advancement gate + click-attenuation countdown -------------------------

accuracy_aos = (sum(main_trial_correct) / float(len(main_trial_correct))
                if main_trial_correct else 0.0)
abort_rate = (total_aborts / float(total_attempts)) if total_attempts else 0.0

this_session_gates_met = staircase.stage3_gates_met(
    accuracy_aos, abort_rate, session_trial_count, trial_count_gate=VAR_TRIAL_COUNT_ADVANCE)
history = state.get('stage3_sessions_history')
history.append(this_session_gates_met)
state.set('stage3_sessions_history', history[-3:])
advance_ready = len(history) >= 3 and all(history[-3:])

if _click_attenuation_remaining > 0:
    state.set('click_attenuation_sessions_remaining', _click_attenuation_remaining - 1)

state.save()

print("Session: {0} main trials, accuracy(AOS)={1:.1%}, abort_rate={2:.1%}, gates_met={3}".format(
    session_trial_count, accuracy_aos, abort_rate, this_session_gates_met), flush=True)
print("ADVANCE-READY (3 consecutive qualifying sessions): {0}".format(
    "yes" if advance_ready else "no"), flush=True)
if not advance_ready:
    print("  (history so far: {0})".format(history), flush=True)
if accuracy_aos <= 0.5 and len(main_trial_correct) >= VAR_SUBCHANCE_BLOCK_SIZE:
    print("NOTE: accuracy is at or below chance -- per the doc's Tag J, before declaring a "
          "non-learner: (1) verify per-rig speaker SPL calibration measured at the head position, "
          "(2) inspect ears for infection, (3) confirm response to a brief supra-threshold "
          "unilateral click-train block. See STAGE3_TEST_PLAN.md for the full checklist.",
          flush=True)

_cleanup_and_export()

print("Close the plot window to exit.", flush=True)
plt.ioff()
plt.show()
