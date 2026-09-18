# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""
training_protocol.md Revision 2, Stage 4 -- "resume threshold and quiescence staircases".

Structural twin of stage3_clicks_direction.py (same 0.25s cue, same 100-200ms delay range --
Stage 4 doesn't extend either, that's Stages 5/6; same warm-up block, debiasing rule, abort
re-offer, and engagement-based session-end triggers) with the two staircases that were frozen
since Stage 2/3 now resumed: response threshold (staircase.ThresholdStaircase, the SAME class
Stage 2 uses, continuing from wherever Stage 2/3 left its persisted counters) toward
STAGE4_RESPONSE_THRESHOLD_FINAL_FRACTION (90% of edge azimuth, not ThresholdStaircase's own 1.0
ceiling) -- +10%/20 successes, -10%/5 failures; quiescence (staircase.QuiescenceStaircase, new)
toward a 0.1-0.5s exponential -- +25ms/20 successful initiations, -25ms/5 resets.

Tag B (training_protocol.md Revision 2 Appendix A): at most ONE of the two staircases may tighten
in any given SESSION -- selected once, at session start, via
staircase.select_stage4_staircase_to_tighten() (policy: prioritize whichever is furthest from its
own final value, tie-break by alternating from the previous session's choice). Implementation
choice made here: the INACTIVE staircase is fully frozen for the whole session -- its
record_outcome() is simply never called, so neither its current value NOR its internal
consecutive-success/failure counters move until a future session selects it. (A more granular
"track counters but withhold the step" mode was considered and rejected: ThresholdStaircase/
QuiescenceStaircase's own record_outcome() has no way to separate "count this" from "apply the
resulting step" without changing those classes, and full-freeze is what "the other holds" most
unambiguously means.) Which staircase was active is registered every trial (STAIRCASE_ACTIVE) and
persisted as 'stage4_last_staircase_advanced' for next session's tie-break.

No click-level attenuation here -- Stage 3's own countdown (click_attenuation_sessions_remaining)
should already have reached 0 by the time a subject advances to Stage 4; this script always plays
at full click level. Still single-evidence-level (gamma=+-1.0/AOS) -- Stage 4's own advancement
criterion is stated in terms of "accuracy at gamma=+-1.0", same as Stage 3, so the evidence grid
does not expand until a later stage.

At the end of the FINAL qualifying session, run `_wheel_shaping_shared/derive_intrial_threshold.py`
manually (not from this script -- Stage 5 doesn't exist yet in this codebase to consume the
result automatically) against that session's own CSV to derive a new in-trial threshold for
whatever comes next; hand-enter the result into the persisted 'in_trial_threshold_deg' (or a
future Stage 5 script's own starting config) once available.

See stage3_clicks_direction.py's own docstring for everything else this file shares with it
unchanged (hardware/module connection plumbing, additions.txt instrumentation conventions, the
T9 monotonic-clock/CSV-flush scope notes).
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
    """ Shared teardown, called from every exit path -- see stage3_clicks_direction.py's own
    identical helper for the fixed exception-path gap this pattern avoids. """
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
    """ additions.txt T9: best-effort git commit hash (+dirty flag) for provenance. Runs the git
    calls on a background daemon thread and bounds the MAIN thread with thread.join(timeout=...)
    instead of relying on subprocess's own timeout= -- confirmed on real hardware (Stage 3, same
    _git_commit() copy, log-stage3.txt) that subprocess.check_output(..., timeout=3) alone is NOT
    reliably enforced on Windows here: it still stalled ~84s despite the timeout, a known CPython
    Windows quirk where Popen.communicate(timeout=...), on timing out, kills the immediate child
    then does a SECOND unbounded communicate() to drain output -- if a descendant process holds the
    stdout/stderr pipe open, that second call blocks until the real operation finishes on its own.
    Bounding the join instead means the main thread is never blocked past the join timeout
    regardless of what the underlying subprocess does; the daemon thread is simply abandoned.

    Timeout is 15s, not the original 5s -- confirmed on real hardware (Stage 3) that even 'git
    rev-parse HEAD' alone was still regularly taking longer than 5s on this rig even after this
    threading fix went in, so 5s was cutting it off before it could ever actually succeed. The
    underlying git slowness on this rig is unexplained and NOT fixed by this -- 15s just gives it
    more room to finish, while staying far below the original 60-90s stall. """
    result = {}

    def _worker():
        try:
            commit = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], cwd=_TASK_DIR,
                stderr=subprocess.DEVNULL, timeout=15).decode('utf-8').strip()
        except Exception:
            return
        try:
            dirty = subprocess.call(
                ['git', 'diff', '--quiet'], cwd=_TASK_DIR,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15) != 0
            dirty_suffix = '-dirty' if dirty else ''
        except Exception:
            dirty_suffix = '-unknown'
        result['commit'] = '{0}{1}'.format(commit, dirty_suffix)

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()
    worker.join(timeout=15.0)
    return result.get('commit', 'unknown')


# --- who this session is for -----------------------------------------------------------------------

_raw_subjects = getattr(settings, 'PYBPOD_SUBJECTS', None)
if not _raw_subjects:
    raise RuntimeError(
        "No subject selected in the GUI for this session (PYBPOD_SUBJECTS is empty) -- select a "
        "subject before running this task.")
VAR_SUBJECT_ID = session_csv_parser.parse_subject_name(_raw_subjects[0])
VAR_PROJECT_DIR = os.path.abspath(os.path.join(_TASK_DIR, '..', '..'))

# --- stage parameters (training_protocol.md Revision 2, Stage 4) -------------------------------------

VAR_MAX_TRIALS = 1000
VAR_STEADY_THRESHOLD_DEG = 4
VAR_RESPONSE_TIMEOUT_S = 5.0

VAR_THRESHOLD_FINAL_DEG = 35
VAR_TRIAL_COUNT_ADVANCE = 200        # doc: ">=200 trials/session" (MAIN trials only)

VAR_STIM_DURATION_S = 0.25           # unchanged from Stage 3 -- extended only from Stage 5 onward
VAR_DELAY_DURATION_MIN_S = 0.1       # unchanged from Stage 3 -- extended only from Stage 6 onward
VAR_DELAY_DURATION_MAX_S = 0.2
VAR_ABORT_TIMEOUT_MIN_S = 1.0
VAR_ABORT_TIMEOUT_MAX_S = 2.0

VAR_TARGET_SPL_DB = 70.0             # not specified by the doc -- flagged/tunable, same default
                                      # as Stage 3. Waveform amplitude is derived from this via
                                      # Calibration/sound_calibration.py's fitted curve (see
                                      # VAR_LEFT_AMPLITUDE_SCALE/VAR_RIGHT_AMPLITUDE_SCALE below) --
                                      # Stage 4 has no warmup-attenuation mechanism, so this applies
                                      # directly/unconditionally, unlike Stage 3.

VAR_CUE_ABORT_THRESHOLD_DEG_PLACEHOLDER = 2.5 * VAR_STEADY_THRESHOLD_DEG   # see
                                      # stage3_clicks_direction.py's own identical constant --
                                      # still a placeholder through Stage 4's own trials; only
                                      # RE-DERIVED (via derive_intrial_threshold.py, manually, by
                                      # the experimenter) at the end of Stage 4, for a future stage.

VAR_ITI_S = 1.5
VAR_INCORRECT_ITI_S = 4.0
VAR_CONSUMPTION_WINDOW_S = 3.0

VAR_REWARD_UL = 4.0
VAR_REWARD_DURATION = get_reward_duration_s(VAR_REWARD_UL)

VAR_DOT_ONSET_JITTER_MIN_S = 0.1
VAR_DOT_ONSET_JITTER_MAX_S = 0.2
VAR_DOT_DISAPPEAR_MIN_S = 0.4
VAR_DOT_DISAPPEAR_MAX_S = 0.9

VAR_DOT_BACKGROUND_GRAY = 128
VAR_DOT_GRAY = 0
VAR_DOT_EDGE_FRACTION = 0.9
VAR_RENDER_HZ = 30

VAR_WARMUP_TRIAL_MIN = 20
VAR_WARMUP_TRIAL_MAX = 40

VAR_SESSION_CAP_S = 90 * 60
VAR_NO_INIT_TIMEOUT_S = 3 * 60
VAR_CONSECUTIVE_ABORT_LIMIT = 20
VAR_SUBCHANCE_BLOCK_SIZE = 20
VAR_SUBCHANCE_BLOCKS = 3

VAR_ROTARY_USB_PORT = None
VAR_STILL_POLL_HZ = 100              # additions.txt T4: NEEDS ON-RIG CONFIRMATION that the poll
VAR_POLL_HZ = 100                    # thread keeps up without delaying state-machine handling --
                                      # if not, fall back to 50/50 and note which rate was used.

VAR_GO_CUE_LED_CHANNEL = 'PWM1'

VAR_CAMERA_INDEX = None
VAR_CAMERA_OUTPUT_PATH = 'session_video.avi'
VAR_CAMERA_FPS = 30.0
VAR_CAMERA_PREVIEW = True
VAR_CAMERA_SNIPPET_S = 1.0

# --- persisted cross-session state (same file as Stage 1/2/3) ----------------------------------------

state = session_state.StageState(VAR_PROJECT_DIR, VAR_SUBJECT_ID, defaults={
    'threshold_fraction': staircase.STAGE4_RESPONSE_THRESHOLD_FINAL_FRACTION * 0.5,   # sensible
                                       # fallback only -- normally already set by Stage 2/3.
    'consecutive_successes': 0,
    'consecutive_failures': 0,
    'in_trial_threshold_deg': VAR_CUE_ABORT_THRESHOLD_DEG_PLACEHOLDER,
    'quiescence_s': staircase.QUIESCENCE_FLOOR_S,
    'quiescence_consecutive_successes': 0,
    'quiescence_consecutive_failures': 0,
    'stage4_last_staircase_advanced': None,   # 'response' / 'quiescence' / None -- Tag B tie-break
    'stage4_sessions_history': [],            # bools: did each past session meet
                                               # staircase.stage4_gates_met()? (2-consecutive check)
})

threshold_obj = staircase.ThresholdStaircase(
    current_fraction=state.get('threshold_fraction'),
    consecutive_successes=state.get('consecutive_successes'),
    consecutive_failures=state.get('consecutive_failures'))
quiescence_obj = staircase.QuiescenceStaircase(
    current_s=state.get('quiescence_s'),
    consecutive_successes=state.get('quiescence_consecutive_successes'),
    consecutive_failures=state.get('quiescence_consecutive_failures'))

VAR_CUE_ABORT_THRESHOLD_DEG = state.get('in_trial_threshold_deg')

# Tag B: decided ONCE, for the whole session -- only this staircase's record_outcome() is ever
# called below; the other is fully frozen this session (see module docstring).
active_staircase = staircase.select_stage4_staircase_to_tighten(
    threshold_obj.current_fraction, quiescence_obj.current_s,
    state.get('stage4_last_staircase_advanced'))

# --- connect to Bpod, resolve modules ----------------------------------------------------------------

my_bpod = Bpod()
print("Connected to Bpod on {0}".format(my_bpod.serial_port), flush=True)

rotary, rotary_bpod_module = rotary_setup.connect_rotary(my_bpod, usb_port=VAR_ROTARY_USB_PORT)
reset_positions_trigger_id, rotary_channel = rotary_setup.build_reset_trigger(rotary_bpod_module)

hifi = hifi_setup.connect_hifi(my_bpod)
hifi_stop_msg_id, hifi_channel = hifi_setup.build_stop_trigger(my_bpod)

VAR_LEFT_AMPLITUDE_SCALE, VAR_RIGHT_AMPLITUDE_SCALE = hifi_setup.compute_calibrated_amplitudes(
    VAR_TARGET_SPL_DB, click_train.VAR_LEFT_FREQ_HZ, click_train.VAR_RIGHT_FREQ_HZ)

my_bpod.register_value('GIT_COMMIT', _git_commit())
my_bpod.register_value('STAIRCASE_ACTIVE_THIS_SESSION', active_staircase)
my_bpod.register_value('REWARD_UL', VAR_REWARD_UL)
my_bpod.register_value('TARGET_SPL_DB', VAR_TARGET_SPL_DB)
my_bpod.register_value('LEFT_FREQ_HZ', click_train.VAR_LEFT_FREQ_HZ)
my_bpod.register_value('RIGHT_FREQ_HZ', click_train.VAR_RIGHT_FREQ_HZ)

log_python_t0 = time.time()
runner = TrialRunner(my_bpod, rotary, log_python_t0, still_poll_hz=VAR_STILL_POLL_HZ,
                      poll_hz=VAR_POLL_HZ)

dot = dot_display.create_dot_display(background_gray=VAR_DOT_BACKGROUND_GRAY, dot_gray=VAR_DOT_GRAY)
dot.show()
dot.clear()

camera = CameraRecorder(log_python_t0, camera_index=VAR_CAMERA_INDEX,
                         output_path=VAR_CAMERA_OUTPUT_PATH, fps=VAR_CAMERA_FPS,
                         preview=VAR_CAMERA_PREVIEW)
camera.start()

my_bpod.register_value('CAMERA_START_TIME', camera.start_time_s)
my_bpod.register_value('CAMERA_OUTPUT_PATH', camera.output_path)
my_bpod.register_value('CAMERA_FPS', camera.fps)

render_interval = 1.0 / VAR_RENDER_HZ

bench_plots = WheelShapingPlots(
    stage=4, threshold_final_deg=VAR_THRESHOLD_FINAL_DEG,
    prev_session_values={'threshold_deg': threshold_obj.current_fraction * VAR_THRESHOLD_FINAL_DEG},
    session_status={'staircase_active': active_staircase, 'quiescence_s': quiescence_obj.current_s,
                     'in_trial_threshold_deg': VAR_CUE_ABORT_THRESHOLD_DEG},
    reward_ul=VAR_REWARD_UL)

print("Starting Stage 4 -- staircase active this session: {0} (response threshold {1:.1f}deg = "
      "{2:.0%} of final, quiescence scale {3:.3f}s)".format(
          active_staircase, threshold_obj.current_fraction * VAR_THRESHOLD_FINAL_DEG,
          threshold_obj.current_fraction, quiescence_obj.current_s), flush=True)

# --- trial loop -----------------------------------------------------------------------------------

session_start_time = time.time()
last_initiation_time = session_start_time
session_reward_count = 0
session_trial_count = 0
total_attempts = 0
total_aborts = 0
consecutive_aborts = 0
main_trial_correct = []
warmup_target = random.randint(VAR_WARMUP_TRIAL_MIN, VAR_WARMUP_TRIAL_MAX)
fresh_trial_count = 0

prev_outcome = None
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
        # Response threshold IS this trial's own value if the response staircase is active (it may
        # have moved on a previous trial this session); otherwise it's frozen at session start.
        cur_response_threshold_deg = threshold_obj.current_fraction * VAR_THRESHOLD_FINAL_DEG
        cur_quiescence_s = quiescence_obj.current_s

        if pending_repeat:
            trial_type = 'repeat'
            side = repeat_side
            # A post-abort repeat re-offers the same side deterministically, not a probabilistic
            # draw -- p_right_target is 1.0/0.0 accordingly (feeds the sidebias plot's target line).
            p_right_target = 1.0 if side == 'R' else 0.0
            pending_repeat = False
        else:
            fresh_trial_count += 1
            trial_type = 'warmup' if fresh_trial_count <= warmup_target else 'main'
            if prev_outcome == 'incorrect':
                p_right_target = (debiasing.VAR_DEBIAS_REPEAT_PROB if prev_side == 'R'
                                   else 1.0 - debiasing.VAR_DEBIAS_REPEAT_PROB)
                side = debiasing.next_side_after_error(prev_side)
            else:
                p_right_target = 0.5
                side = np.random.choice(['L', 'R'])

        quiescence_duration_this_trial = float(
            np.clip(np.random.exponential(cur_quiescence_s),
                    staircase.QUIESCENCE_FLOOR_S, staircase.QUIESCENCE_CEILING_S))

        # Thresholds are armed AFTER the hold-to-init succeeds, not before -- quiescence itself is
        # governed by wait_for_held_steady()'s own rolling-window check on raw current_position(),
        # never by these Bpod-native threshold-crossing events (same order as every other stage's
        # trial loop; arming them earlier would let a stray crossing during the hold pre-empt the
        # in-trial-abort/response logic before the trial has even properly started).
        rotary.disable_evt_transmission()
        n_breaks = runner.wait_for_held_steady(quiescence_duration_this_trial,
                                                VAR_STEADY_THRESHOLD_DEG, require_no_lick=False)
        if n_breaks is None:
            print("Bpod stopped running trials (Stop/Kill) -- ending session early after trial "
                  "{0}.".format(trial), flush=True)
            break
        trial_start_t = time.time() - log_python_t0
        last_initiation_time = time.time()

        if active_staircase == 'quiescence':
            quiescence_obj.record_outcome(success=(n_breaks == 0))

        with runner.rotary_lock:
            rotary.set_zero_position()
            event_names = rotary_setup.set_and_enable_thresholds(
                rotary, [-VAR_STEADY_THRESHOLD_DEG, VAR_STEADY_THRESHOLD_DEG,
                         -cur_response_threshold_deg, cur_response_threshold_deg,
                         -VAR_CUE_ABORT_THRESHOLD_DEG, VAR_CUE_ABORT_THRESHOLD_DEG])
        rotary.enable_evt_transmission()
        left_event, right_event = event_names[2], event_names[3]
        wheel_abort_event_neg, wheel_abort_event_pos = event_names[4], event_names[5]

        runner.register('TRIAL_START', trial_start_t)
        runner.register('QUIESCENCE_BREAKS', n_breaks)
        runner.register('TRIAL_TYPE', trial_type)
        runner.register('STAIRCASE_ACTIVE', active_staircase)
        dot.clear()
        dot.pump()
        camera.pump()

        stim_seed = int(np.random.randint(0, 2 ** 31 - 1))
        trial_clicks = click_train.generate_trial_clicks(
            'AOS', side, rng=np.random.RandomState(stim_seed), stim_duration_s=VAR_STIM_DURATION_S)
        left_wave, right_wave = click_train.build_waveform(
            trial_clicks, hifi.sampling_rate,
            amplitude_scale_left=VAR_LEFT_AMPLITUDE_SCALE,
            amplitude_scale_right=VAR_RIGHT_AMPLITUDE_SCALE,
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
        runner.register('RESPONSE_THRESHOLD_DEG', cur_response_threshold_deg)
        runner.register('QUIESCENCE_DUR_S', quiescence_duration_this_trial)
        runner.register('ITI_S', VAR_ITI_S)

        print("Trial {0} ({1}, staircase={2}): side={3}, n_L={4}, n_R={5}".format(
            trial, trial_type, active_staircase, side, trial_clicks['n_left'],
            trial_clicks['n_right']), flush=True)

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
            runner.register('INCLUDED', False)

            pending_repeat = True
            repeat_side = side

            print("Trial {0}: Abort ({1} epoch) -- re-offering side {2} next trial.".format(
                trial, abort_epoch, side), flush=True)

            bench_plots.add_trial(side, 0.0, cur_response_threshold_deg, 'Abort',
                                   lick_times_abs=cue_lick_times_abs, trial_type=trial_type,
                                   p_right_target=p_right_target)
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

        # Geometry-aware gain -- recomputed each trial since cur_response_threshold_deg can move
        # (the response staircase, when active, updates between trials).
        screen_width_px = dot.get_screen_width_px()
        dot_gain = rotary_setup.screen_direction_gain(
            (VAR_DOT_EDGE_FRACTION * (screen_width_px / 2.0)) / cur_response_threshold_deg)
        dot.set_deg_to_px_gain(dot_gain)

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
                if pos <= -cur_response_threshold_deg or pos >= cur_response_threshold_deg:
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

        # Response staircase, when active: "success" = cleared the threshold at all (a genuine
        # response, correct or incorrect) -- same convention as Stage 2's own ThresholdStaircase
        # usage, not gated on correctness.
        if active_staircase == 'response':
            threshold_obj.record_outcome(success=outcome_state in ('Reward', 'ErrorConsumption'))

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

        # accuracy_aos (session end) is conditional on a genuine L/R response -- see
        # stage3_clicks_direction.py's own identical comment for why NoResponse is excluded.
        if trial_type == 'main' and correct is not None:
            session_trial_count += 1
            main_trial_correct.append(bool(correct))

        consumption_state_name = {'Reward': 'Consumption',
                                   'ErrorConsumption': 'ErrorConsumption'}.get(outcome_state)
        consumption_licked = None
        if consumption_state_name is not None:
            c_start, c_end = visited[consumption_state_name][-1]
            consumption_licked = any(c_start <= t <= c_end for t in choice_events.get('Port1In', []))

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
        # already "time since go-cue" -- no extra send_t arithmetic needed. click_diff (signed
        # click-count evidence) feeds the psychometric curve; both are None for a NoResponse trial.
        response_time_s = (visited[outcome_state][-1][0]
                            if outcome_state in ('Reward', 'ErrorConsumption') else None)
        click_diff = trial_clicks['realized_delta'] if response_time_s is not None else None

        bench_plots.add_trial(side, cur_response_threshold_deg if response is not None else 0.0,
                               cur_response_threshold_deg, outcome,
                               lick_times_abs=lick_times_abs, reward_time_abs=reward_time_abs,
                               trial_type=trial_type, p_right_target=p_right_target,
                               response_time_s=response_time_s, click_diff=click_diff,
                               in_trial_threshold_deg=VAR_CUE_ABORT_THRESHOLD_DEG)

    except Exception as err:
        print("Trial {0} FAILED: {1}".format(trial, err), flush=True)
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        _cleanup_and_export()
        raise

if session_end_reason is not None:
    runner.register('SESSION_END_REASON', session_end_reason)
print("Session ending: {0}".format(session_end_reason or 'Stop/Kill/crash (see log above)'),
      flush=True)

# --- session-end bookkeeping: persist staircases + advancement gate ----------------------------------

state.set('threshold_fraction', threshold_obj.current_fraction)
state.set('consecutive_successes', threshold_obj.consecutive_successes)
state.set('consecutive_failures', threshold_obj.consecutive_failures)
state.set('quiescence_s', quiescence_obj.current_s)
state.set('quiescence_consecutive_successes', quiescence_obj.consecutive_successes)
state.set('quiescence_consecutive_failures', quiescence_obj.consecutive_failures)
state.set('stage4_last_staircase_advanced', active_staircase)

accuracy_aos = (sum(main_trial_correct) / float(len(main_trial_correct))
                if main_trial_correct else 0.0)
abort_rate = (total_aborts / float(total_attempts)) if total_attempts else 0.0

this_session_gates_met = staircase.stage4_gates_met(
    threshold_obj.current_fraction, quiescence_obj.current_s, accuracy_aos, abort_rate,
    session_trial_count, trial_count_gate=VAR_TRIAL_COUNT_ADVANCE)
history = state.get('stage4_sessions_history')
history.append(this_session_gates_met)
state.set('stage4_sessions_history', history[-2:])
advance_ready = len(history) >= 2 and all(history[-2:])

state.save()

print("Session: {0} main trials, accuracy(AOS)={1:.1%}, abort_rate={2:.1%}, gates_met={3}".format(
    session_trial_count, accuracy_aos, abort_rate, this_session_gates_met), flush=True)
print("Response threshold now {0:.1f}deg ({1:.0%} of final, target {2:.0%}), quiescence scale "
      "now {3:.3f}s (target {4:.1f}s)".format(
          threshold_obj.current_fraction * VAR_THRESHOLD_FINAL_DEG, threshold_obj.current_fraction,
          staircase.STAGE4_RESPONSE_THRESHOLD_FINAL_FRACTION, quiescence_obj.current_s,
          staircase.QUIESCENCE_CEILING_S), flush=True)
print("ADVANCE-READY (2 consecutive qualifying sessions): {0}".format(
    "yes" if advance_ready else "no"), flush=True)
if not advance_ready:
    print("  (history so far: {0})".format(history), flush=True)
if advance_ready:
    print("NOTE: run derive_intrial_threshold.py against this (or the final qualifying) "
          "session's own CSV to re-derive the in-trial threshold for whatever stage comes next, "
          "per training_protocol.md SS1.5 -- not done automatically.", flush=True)

_cleanup_and_export()

print("Close the plot window to exit.", flush=True)
plt.ioff()
plt.show()
