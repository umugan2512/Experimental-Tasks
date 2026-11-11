# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""
training_protocol.md Revision 2, Stage 3 -- "tone-side association" (new stage, inserted ahead of
what used to be Stage 3 -- see stage4_clicks_direction.py, pushed up to Stage 4 to make room, and
stage5_resume_staircases.py, pushed up to Stage 5).

Aim: the click-side association, in its easiest possible form -- the first moment clicks and
direction appear together. Every trial plays clicks ONLY for its side (click_train_v2's 'AOS' grid
entry guarantees the other channel gets exactly zero clicks), and the side comes in long BLOCKS
that alternate L/R within a session (staircase.Stage3Blocks, per explicit instruction on
2026-11-09; until then one side was held for the whole day, which let the animal learn "today is
left" from reward history without listening). A block lasts a random number of responded trials
in [stage3_block_min, stage3_block_max] (30-60 at first, shrinking one step per qualifying session
toward 10-20), so the animal has to follow the clicks across a switch. Stage 4 is where the side
varies per trial. The click -> side mapping itself never changes.

Every other mechanic is a direct, unmodified carryover from Stage 4 (formerly Stage 3) -- the
quiescence hold, the in-trial wheel-excursion abort threshold (monitored through
CuePeriod+DelayPeriod, aborting into a short timeout), the delay period (Tag A, 100-200ms fixed),
LED+dot choice period, reward/error/no-response handling, the click-level attenuation ramp (~6dB
below final for the first VAR_CLICK_ATTENUATION_SESSIONS session runs), HiFi output calibrated via
Calibration/sound_calibration.py, camera recording, GIT_COMMIT/REWARD_UL provenance registration,
and session-struct export -- none of that depends on whether the side is drawn per-trial or held
fixed for the session, so it's reused here exactly as-is.

What's genuinely different from Stage 4:
  - VAR_FIRST_BLOCK_SIDE: the side of the session's FIRST block. Per explicit instruction this is
    NOT a fresh random draw every session. The very FIRST time this subject ever runs this stage,
    it's a coin flip (so different subjects don't all start on the same side). After that, it
    deterministically ALTERNATES ('R' <-> 'L') once a new CALENDAR DAY starts, but stays
    UNCHANGED across a same-day Stop/Kill/crash restart. Persisted as
    'stage3_last_side'/'stage3_last_side_date' and saved IMMEDIATELY (not just at session end)
    so a near-instant crash still leaves the decision durable for a same-day restart. A restart
    begins a fresh block sequence from that side. Registered as FIRST_BLOCK_SIDE (SESSION_SIDE
    before 2026-11-09), with BLOCK_MIN/BLOCK_MAX; every trial registers BLOCK_INDEX, BLOCK_SIDE,
    BLOCK_LENGTH and BLOCK_TRIAL (responded trials earlier in that block). Only a genuine L/R
    response advances a block, so aborts and timeouts don't use it up.
  - The sub-chance stop ignores the first STAGE3_SUBCHANCE_SKIP_AFTER_SWITCH (10) responded
    trials after each switch, where a mouse is expected to keep turning the old way at first.
    accuracy_aos, the advancement gate and the stim/ITI staircases count every main trial. The
    session end prints accuracy on the first 5 responded trials after a switch.
  - No debiasing at all (debiasing.next_side_after_error() is Stage 4's own mechanism for choosing
    the NEXT trial's side after an error; here the block decides the side).
  - No post-abort "repeat" re-offer either -- the next trial is in the same block anyway.
    trial_type is therefore only 'warmup'/'main', never 'repeat'.
  - p_right_target is 1.0/0.0 for the current block's side.
  - Advancement gate is a new staircase.stage3_gates_met() (the doc doesn't specify this new
    stage's own numeric criterion yet -- reusing Stage 4's own values as a placeholder pending real
    empirical tuning, same "flagged placeholder" convention already used elsewhere in this
    codebase): >70% correct at gamma=+-1.0 (AOS), abort rate <20%, >200 main trials/session, three
    consecutive qualifying sessions.

The cue is FIXED at VAR_STIM_DURATION_S = 0.35s through Stages 3-5 (2026-11-11 spec A6/B3).
The former within-session cue-duration staircase is gone: it added a second change while
direction was being learned, and with one-sided clicks it only added same-side clicks. A saved
'stim_duration_s' other than 0.35s (from that staircase) is registered as
STIM_DURATION_RESET_FROM_S; the key is no longer written.

The ITIs are FIXED from Stage 3 on (2026-11-11 spec A2): correct VAR_ITI_S = 1.5s (where Stage 2
leaves 'iti_s'), error VAR_INCORRECT_ITI_S = 4.0s. A correct-ITI staircase raised the ITI on
correct streaks, which cut the reward rate for good performance and shrank the error/correct
contrast from 2.7x toward 1.33x while direction was being learned. A saved 'iti_s' above 1.5s
(from the old ramp) is registered as ITI_RESET_FROM_S; Stages 3-5 no longer write 'iti_s'.

Hardware/module connection plumbing, thread-inversion render loop, and camera integration are all
reused as-is from _shared/ and mirror full_protocol_lookback_test.py's already-hardware-validated
patterns exactly -- see this project's stage1_wheel_shaping.py/stage2_threshold_staircase.py for
the same conventions applied to this project's own persisted-state/subject-derivation machinery.
"""
import datetime
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
import click_train_v2 as click_train
from wheel_shaping_plots import WheelShapingPlots
from bpod_trial_helpers import TrialRunner, was_visited, PC_EVENT_QUIESCENCE_MET, PC_EVENT_DOT_SHOWN
from trial_render import run_trial_render_loop, register_trial_timing
import host_clock
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
    """ Shared teardown, called from every exit path (normal end, Stop-triggered break, Kill, and
    an unhandled per-trial exception) so hardware connections and the struct export are never
    skipped regardless of how the session ended -- see stage1_wheel_shaping.py's own identical
    helper for the fixed exception-path gap this pattern avoids. """
    try:
        runner.register('SESSION_WATER_UL', session_reward_count * VAR_REWARD_UL)
    except Exception as err:
        print("WARNING: could not register SESSION_WATER_UL: {0}".format(err), flush=True)
    runner.stop()   # writes the last buffered wheel samples, then stops the rotary's USB stream --
                     # before rotary.close() and my_bpod.close(), which it needs.
    camera.close()
    realized_fps = camera.compute_realized_fps()
    if realized_fps is not None:
        my_bpod.register_value('CAMERA_FPS_REALIZED', realized_fps)
    # longest frame read / video write -- tells a camera frame gap's cause (see CameraRecorder)
    my_bpod.register_value('CAMERA_MAX_READ_MS', camera.max_read_ms)
    my_bpod.register_value('CAMERA_MAX_WRITE_MS', camera.max_write_ms)
    dot.close()
    hifi.close()
    rotary.close()
    csv_path = my_bpod.session._path   # grab before close() -- close() deletes the Session object
                                        # that holds it
    my_bpod.close()
    _export_session_struct(csv_path)


def _git_commit():
    """ additions.txt T9: best-effort git commit hash (+dirty flag) for provenance, registered as
    a session-level VAL. Never blocks the task -- returns 'unknown' if git isn't available, this
    checkout isn't a git repo, or the git calls don't finish promptly.

    Runs the actual git calls on a background daemon thread and bounds the MAIN thread with
    thread.join(timeout=...) instead of relying on subprocess's own timeout= to bound the wait --
    confirmed on real hardware (log-stage3.txt) that subprocess.check_output(..., timeout=3) alone
    is NOT reliably enforced here: it still stalled ~84s with 'git rev-parse HEAD' despite the
    timeout. Root cause: on Windows, when Popen.communicate(timeout=...) hits its timeout it calls
    kill() then does a SECOND, unbounded communicate() to drain any remaining output before
    raising TimeoutExpired -- if git.exe (or something wrapping it) leaves a descendant process
    holding the stdout/stderr pipe open, kill() only terminates the immediate child, and that
    second unbounded drain blocks until the real operation finishes on its own. Bounding the JOIN
    instead sidesteps this entirely: even if the underlying subprocess call never returns, the main
    thread is never blocked past the join timeout -- the daemon thread is simply abandoned (same
    "stale thread" tolerance already used elsewhere in this codebase, e.g.
    TrialRunner.run_trial_state_machine()'s _stale_poll_threads).

    Timeout is 15s, not the original 5s -- confirmed on real hardware that even 'git rev-parse
    HEAD' alone (normally near-instant) was still regularly taking longer than 5s on this rig even
    after this threading fix went in (registering GIT_COMMIT='unknown' every real session since),
    so 5s was cutting it off before it could ever actually succeed. The underlying git slowness on
    this rig is unexplained and NOT fixed by this -- 15s just gives it more room to finish before
    giving up, while staying far below the original 60-90s stall this whole mechanism exists to
    bound. """
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

# --- stage parameters (training_protocol.md Revision 2, Stage 3) -------------------------------------

VAR_MAX_TRIALS = 1000                # hard-ceiling safety backstop, well above what a 90-min
                                      # session at this stage's pace should ever reach
VAR_QUIESCENCE_S = 0.1               # FIXED fallback only -- used unless 'quiescence_unlocked' is
                                      # already True (the normal case by the time a subject reaches
                                      # Stage 3; see Stage 2's own module for the unlock condition),
                                      # in which case the persisted per-trial staircase continues
                                      # uninterrupted instead (see the trial loop below).
VAR_STEADY_THRESHOLD_DEG = 4         # IBL convention, same as every other task -- hold-still deadzone
VAR_RESPONSE_TIMEOUT_S = 5.0         # doc's own epoch table: response window <=5s

VAR_THRESHOLD_FINAL_DEG = 35         # matches this project's established final-task convention
VAR_TRIAL_COUNT_ADVANCE = 100        # lowered from the doc's original 200, per explicit
                                      # instruction. >100 trials/session (MAIN trials only --
                                      # warmup excluded,
                                      # see session_trial_count below), same convention as Stage 4

VAR_STIM_DURATION_S = staircase.CUE_DURATION_S   # fixed 0.35s cue through Stages 3-5 (spec A6/B3)
VAR_DELAY_DURATION_MIN_S = 0.1       # Tag A: cue-to-go gap, 100-200ms FIXED range at this stage
VAR_DELAY_DURATION_MAX_S = 0.2       # (staircased 0-750ms only from Stage 7 onward, not built here)

VAR_CUE_ABORT_THRESHOLD_DEG_PLACEHOLDER = 2.5 * VAR_STEADY_THRESHOLD_DEG   # 10deg -- PLACEHOLDER
                                      # only. Reused from full_protocol_lookback_test.py's own
                                      # hardware-precedented value since the doc's real derivation
                                      # procedure (derive_intrial_threshold.py) only runs at the
                                      # END of Stage 5. Replace the persisted 'in_trial_threshold_deg'
                                      # once that has run at least once for this subject.

VAR_ITI_S = 1.5                      # fixed correct-trial ITI from Stage 3 on (spec A2):
                                      # Stage 2's final 'iti_s'. No staircase after Stage 2.
# An abort (wheel moved during cue, delay or J1) costs the same time as an error (spec A4):
# AbortITI lasts VAR_INCORRECT_ITI_S. With a cheaper abort, breaking fixation would be the cheaper
# exit from a trial the animal expects to get wrong -- concentrated at low |gamma| once the cue has
# wrong-side clicks, which biases the psychometric and accumulator fits.
VAR_INCORRECT_ITI_S = 4.0            # fixed error-trial ITI, per the doc's epoch table
VAR_CONSUMPTION_WINDOW_S = 3.0       # not doc-specified for this stage -- same flagged/tunable
                                      # placeholder convention as every other unspecified duration
                                      # in this codebase; matches full_protocol_lookback_test.py's
                                      # own value.

VAR_REWARD_UL = float(os.environ.get('RUN_SESSION_REWARD_UL', 4.0))   # target reward volume --
                                      # valve open time is derived from this via
                                      # Calibration/liquid_calibration.py's own fitted curve. The
                                      # env var lets run_session.py's own launch-time reward field
                                      # override this default; absent (real PyBpod-GUI launch) ->
                                      # falls back to the hardcoded default, unchanged behavior.
VAR_REWARD_DURATION = get_reward_duration_s(VAR_REWARD_UL)

VAR_DOT_ONSET_JITTER_MIN_S = 0.1     # J1 -- unchanged from every earlier stage
VAR_DOT_ONSET_JITTER_MAX_S = 0.2     # (now a native, abort-guarded DotOnsetDelay state at the
                                     # end of cue_sma -- see the trial loop)
VAR_DOT_DISAPPEAR_MIN_S = staircase.DOT_DISAPPEAR_MIN_S   # J2, shared by all stages (spec A5)
VAR_DOT_DISAPPEAR_MAX_S = staircase.DOT_DISAPPEAR_MAX_S

VAR_DOT_EDGE_FRACTION = 0.9
VAR_RENDER_HZ = 30

VAR_CLICK_ATTENUATION_DB = 6.0       # doc: "start ~6dB below final"
VAR_CLICK_ATTENUATION_SESSIONS = 2   # doc: "for 2-3 sessions" -- picked 2, tunable; decremented
                                      # once per session RUN (not per qualifying session)

VAR_TARGET_SPL_DB = 70.0             # not specified by the doc -- flagged/tunable, same "pick a
                                      # reasonable default" convention as other unset parameters
                                      # here. Waveform amplitude is derived from this via
                                      # Calibration/sound_calibration.py's fitted curve (see
                                      # VAR_LEFT_AMPLITUDE_SCALE/VAR_RIGHT_AMPLITUDE_SCALE below) --
                                      # VAR_CLICK_ATTENUATION_DB above is then an additional
                                      # multiplier ON TOP of this calibrated base amplitude during
                                      # the warmup-attenuation sessions, not a substitute for it.

VAR_WARMUP_TRIAL_MIN = 20            # doc SS3.1 (Tag D): 20-40 trials at the easiest level,
VAR_WARMUP_TRIAL_MAX = 40            # drawn fresh each session, excluded from advancement stats

VAR_SESSION_CAP_S = 45 * 60          # 45-minute cap, per explicit instruction (was the doc SS3.2/
                                      # Tag E 90-minute value)
# If the LAST session for this subject+protocol ended via Stop/Kill/crash (no SESSION_END_REASON
# registered) and this one is starting soon enough to count as its own continuation (same
# merge-eligibility window build_training_log.py's own session-merging uses), reduce today's own
# time cap by however long that interrupted bout already ran -- so a restart doesn't hand the
# animal a full fresh 90 minutes on top of what it already did. Resets back to the full value
# automatically once a session reaches its own deliberate end (see
# session_csv_parser.compute_remaining_quota()'s own docstring).
_, VAR_SESSION_CAP_S = session_csv_parser.compute_remaining_quota(
    os.path.dirname(os.path.abspath(os.getcwd())), os.path.basename(os.path.abspath(os.getcwd())),
    os.path.splitext(os.path.basename(__file__))[0], VAR_SUBJECT_ID, full_cap_s=VAR_SESSION_CAP_S)
VAR_NO_INIT_TIMEOUT_S = 3 * 60       # ...or 3 continuous minutes with no trial initiated
VAR_CONSECUTIVE_ABORT_LIMIT = 20     # ...or 20 consecutive aborts
VAR_SUBCHANCE_BLOCK_SIZE = 20        # ...or sub-chance accuracy across 3 consecutive 20-trial
VAR_SUBCHANCE_BLOCKS = 3             # blocks at the easiest level (every trial, here)

VAR_ROTARY_USB_PORT = None
VAR_STILL_POLL_HZ = 100              # how often the hold-to-init check re-evaluates the streamed
                                      # samples. The wheel's own sample rate is set by the rotary
                                      # firmware's USB stream, not by this (see rotary_stream.py).
VAR_POLL_HZ = 100                    # unused in stream mode (polling-mode TrialRunner only)

SOFTCODE_DOT_ON = 1                  # sent by DotOn's entry: draw the dot (and answer once painted)
VAR_DOT_HANDSHAKE_TIMEOUT_S = 1.0    # DotOn -> DotTimeout if the PC hasn't confirmed the dot by then
VAR_WAIT_QUIESCENCE_MAX_S = 3600.0   # WaitQuiescence's own timer ("as long as it takes")
SOFTCODE_CROSSED = 2                 # sent by Reward/ErrorConsumption's entry: freeze the dot
SOFTCODE_PUNISH = 3                  # sent by AbortPunish/ErrorPunish's entry: white screen flash,
                                     # in lockstep with the native LED pulse

VAR_GO_CUE_LED_CHANNEL = 'PWM1'      # repurposed as a punishment signal only (abort / incorrect
                                      # choice) -- the dot alone is the go cue, see WheelDotPeriod.
# ErrorConsumption -> ErrorPunish: the error flash comes 0.7-1.0s after the error crossing, drawn per
# trial independently of J2 and never ended by a lick (spec B8). It used to come at the first lick
# or 0.5s, so it was locked to lick onset, overlapped most of the J2 window, and could hide the dot
# offset on error trials only. 0.7s is 100ms after the latest possible dot offset (0.6s).
VAR_ERROR_FLASH_DELAY_MIN_S = 0.7
VAR_ERROR_FLASH_DELAY_MAX_S = 1.0
VAR_ERROR_FLASH_S = 0.35             # punishment flash duration (LED + full-white screen), on
                                      # WheelAbort and the incorrect-choice path only.

VAR_CAMERA_INDEX = None
VAR_CAMERA_OUTPUT_PATH = 'session_video.avi'
VAR_CAMERA_FPS = 30.0
VAR_CAMERA_PREVIEW = os.environ.get('RUN_SESSION_DASHBOARD') != '1'   # preview shown only in short
                                       # snippets (see camera.show_snippet() calls below), never
                                       # continuously. False when launched from run_session.py --
                                       # see stage1_wheel_shaping.py's own identical comment.
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
    'stage3_last_side': None,             # None only before this subject's very first Stage 3
                                           # session ever -- see the side-decision block below.
    'stage3_last_side_date': None,
    'stage3_block_min': staircase.STAGE3_BLOCK_START[0],   # side-block length range (responded
    'stage3_block_max': staircase.STAGE3_BLOCK_START[1],   # trials), shrinks per qualifying session
    'stim_duration_s': VAR_STIM_DURATION_S,   # read only, to report a value left by the old cue
                                               # staircase; never written since 2026-11-11
    'quiescence_unlocked': False,    # set by Stage 2 only -- fallback for a subject that somehow
                                      # skipped it (see VAR_QUIESCENCE_S above).
    'quiescence_s': staircase.QUIESCENCE_FLOOR_S,
    'quiescence_consecutive_successes': 0,
    'quiescence_consecutive_failures': 0,
})

VAR_RESPONSE_THRESHOLD_DEG = state.get('threshold_fraction') * VAR_THRESHOLD_FINAL_DEG
VAR_CUE_ABORT_THRESHOLD_DEG = state.get('in_trial_threshold_deg')
_click_attenuation_remaining = state.get('click_attenuation_sessions_remaining')

STIM_DURATION_RESET_FROM_S = (state.get('stim_duration_s')
                              if abs((state.get('stim_duration_s') or 0) - VAR_STIM_DURATION_S) > 1e-9
                              else None)   # left by the old cue staircase
# Read-only: Stage 1/2 own 'iti_s'. A value above VAR_ITI_S came from the old Stage 3/4 ramp.
ITI_RESET_FROM_S = (state.get('iti_s') if (state.get('iti_s') or 0) > VAR_ITI_S + 1e-9 else None)

quiescence_unlocked = state.get('quiescence_unlocked')
if quiescence_unlocked:
    quiescence_obj = staircase.QuiescenceStaircase(
        current_s=state.get('quiescence_s'),
        consecutive_successes=state.get('quiescence_consecutive_successes'),
        consecutive_failures=state.get('quiescence_consecutive_failures'),
        ceiling_s=staircase.STAGE234_QUIESCENCE_CEILING_S)

# The side of the session's FIRST block (blocks then alternate, see staircase.Stage3Blocks). 'AOS'
# click generation guarantees zero clicks on the other channel, so every trial's clicks belong to
# its block's side only. Per explicit instruction: a coin flip only the very FIRST time this
# subject ever runs this stage (so different subjects don't all start on the same side); after
# that, UNCHANGED across a same-day restart, deterministically ALTERNATING once a new calendar day
# starts. (Until 2026-11-09 this was the side of the whole session.)
_today_str = datetime.date.today().isoformat()
_last_side = state.get('stage3_last_side')
_last_side_date = state.get('stage3_last_side_date')
if _last_side is None:
    VAR_FIRST_BLOCK_SIDE = str(np.random.choice(['L', 'R']))
else:
    VAR_FIRST_BLOCK_SIDE = staircase.next_alternating_value(
        _last_side, _last_side_date, _today_str, ('L', 'R'))
state.set('stage3_last_side', VAR_FIRST_BLOCK_SIDE)
state.set('stage3_last_side_date', _today_str)
# Saved IMMEDIATELY, not just at session end like every other state.set() in this file -- the
# side decision above must survive even a near-instant Kill/crash, since a same-day restart needs
# to see it to avoid re-randomizing/re-alternating. A deliberate, narrow exception to StageState's
# own "save once at the end" convention (see its own docstring for why that's normally right): safe
# here because this is one single, atomic, already-fully-decided value, not a partially-computed
# one -- every OTHER key in state.data at this point is still exactly what was already on disk, so
# this early save can't write out any half-finished session update.
state.save()
VAR_BLOCK_MIN = state.get('stage3_block_min')
VAR_BLOCK_MAX = state.get('stage3_block_max')
blocks = staircase.Stage3Blocks(VAR_FIRST_BLOCK_SIDE, VAR_BLOCK_MIN, VAR_BLOCK_MAX, random)

# --- connect to Bpod, resolve modules ----------------------------------------------------------------

my_bpod = Bpod()
print("Connected to Bpod on {0}".format(my_bpod.serial_port), flush=True)

rotary, rotary_bpod_module = rotary_setup.connect_rotary(my_bpod, usb_port=VAR_ROTARY_USB_PORT)
reset_positions_trigger_id, rotary_channel = rotary_setup.build_reset_trigger(
    rotary_bpod_module, marker_code=rotary_setup.DOT_ONSET_MARKER_CODE)

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

VAR_LEFT_AMPLITUDE_SCALE, VAR_RIGHT_AMPLITUDE_SCALE = hifi_setup.compute_calibrated_amplitudes(
    VAR_TARGET_SPL_DB, click_train.VAR_LEFT_FREQ_HZ, click_train.VAR_RIGHT_FREQ_HZ)

my_bpod.register_value('LEFT_THRESHOLD_DEG', -VAR_RESPONSE_THRESHOLD_DEG)
my_bpod.register_value('RIGHT_THRESHOLD_DEG', VAR_RESPONSE_THRESHOLD_DEG)
my_bpod.register_value('GIT_COMMIT', _git_commit())
my_bpod.register_value('REWARD_UL', VAR_REWARD_UL)
my_bpod.register_value('SOFTCODE_MAP', '1=DotOnRequest,2=OutcomeFreeze,3=PunishFlash')   # Bpod->PC SoftCode numbers (SOFTCODE rows)
# What compute_remaining_quota() found at startup (an interrupted-and-restarted bout shortens this
# session's quota) -- registered so a surprisingly short session is traceable to its cause.
my_bpod.register_value('QUOTA_REDUCED_BY', session_csv_parser.LAST_QUOTA_INFO.get('used_trials', 0))
my_bpod.register_value('QUOTA_CHAIN_SESSIONS',
                       ','.join(session_csv_parser.LAST_QUOTA_INFO.get('chain_sessions', [])))
my_bpod.register_value('TARGET_SPL_DB', VAR_TARGET_SPL_DB)
if STIM_DURATION_RESET_FROM_S is not None:
    my_bpod.register_value('STIM_DURATION_RESET_FROM_S', STIM_DURATION_RESET_FROM_S)
if ITI_RESET_FROM_S is not None:
    my_bpod.register_value('ITI_RESET_FROM_S', ITI_RESET_FROM_S)
    print("ITI fixed at {0:.2f}s from Stage 3 on (saved value was {1:.2f}s)".format(
        VAR_ITI_S, ITI_RESET_FROM_S), flush=True)
my_bpod.register_value('LEFT_FREQ_HZ', click_train.VAR_LEFT_FREQ_HZ)
my_bpod.register_value('RIGHT_FREQ_HZ', click_train.VAR_RIGHT_FREQ_HZ)
my_bpod.register_value('FIRST_BLOCK_SIDE', VAR_FIRST_BLOCK_SIDE)   # SESSION_SIDE before 2026-11-09
my_bpod.register_value('BLOCK_MIN', VAR_BLOCK_MIN)
my_bpod.register_value('BLOCK_MAX', VAR_BLOCK_MAX)

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
my_bpod.register_value('CAMERA_BACKEND', camera.backend)

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
                     'in_trial_threshold_deg': VAR_CUE_ABORT_THRESHOLD_DEG,
                     'block_range': (VAR_BLOCK_MIN, VAR_BLOCK_MAX)},
    reward_ul=VAR_REWARD_UL,
    # See stage1_wheel_shaping.py's own identical comment -- suppresses this script's own popup
    # window only when launched from run_session.py, which shows the same data in its own window.
    embed=(os.environ.get('RUN_SESSION_DASHBOARD') == '1'),
    headless=(os.environ.get('RUN_SESSION_DASHBOARD') == '1'))

# --- trial loop -----------------------------------------------------------------------------------

print("Starting Stage 3 -- side blocks of {5}-{6} responded trials, first block {0}, response "
      "threshold frozen at {1:.1f}deg, in-trial threshold {2:.1f}deg{3}, click level {4}".format(
          VAR_FIRST_BLOCK_SIDE, VAR_RESPONSE_THRESHOLD_DEG, VAR_CUE_ABORT_THRESHOLD_DEG,
          ' (PLACEHOLDER -- not yet empirically derived)'
          if VAR_CUE_ABORT_THRESHOLD_DEG == VAR_CUE_ABORT_THRESHOLD_DEG_PLACEHOLDER else '',
          'ATTENUATED (-{0:.0f}dB, {1} session(s) remaining)'.format(
              VAR_CLICK_ATTENUATION_DB, _click_attenuation_remaining)
          if _click_attenuation_remaining > 0 else 'FULL', VAR_BLOCK_MIN, VAR_BLOCK_MAX), flush=True)

session_start_time = host_clock.now()
last_initiation_time = session_start_time
session_reward_count = 0
session_trial_count = 0        # MAIN trials only -- feeds VAR_TRIAL_COUNT_ADVANCE
total_attempts = 0              # every trial attempt (warmup+main), feeds abort_rate
total_aborts = 0
consecutive_aborts = 0
main_trial_correct = []         # bools, MAIN trials only -- feeds accuracy_aos
subchance_correct = []          # the same, minus the first STAGE3_SUBCHANCE_SKIP_AFTER_SWITCH
                                 # responded trials after a block switch -- feeds the sub-chance stop
after_switch_correct = []       # main trials within 5 responded trials of a switch (session readout)
warmup_target = random.randint(VAR_WARMUP_TRIAL_MIN, VAR_WARMUP_TRIAL_MAX)
fresh_trial_count = 0           # trial draws so far this session (every attempt counts -- no
                                 # separate 'repeat' category here, see module docstring)

session_end_reason = None
trial = 0

while True:
    now = host_clock.now()
    if now - session_start_time >= VAR_SESSION_CAP_S:
        session_end_reason = 'cap_90min'
        break
    if now - last_initiation_time >= VAR_NO_INIT_TIMEOUT_S:
        session_end_reason = 'no_initiation_3min'
        break
    if consecutive_aborts >= VAR_CONSECUTIVE_ABORT_LIMIT:
        session_end_reason = 'consecutive_aborts'
        break
    if len(subchance_correct) >= VAR_SUBCHANCE_BLOCK_SIZE * VAR_SUBCHANCE_BLOCKS:
        recent = subchance_correct[-(VAR_SUBCHANCE_BLOCK_SIZE * VAR_SUBCHANCE_BLOCKS):]
        chunks = [recent[i:i + VAR_SUBCHANCE_BLOCK_SIZE]
                  for i in range(0, len(recent), VAR_SUBCHANCE_BLOCK_SIZE)]
        if all((sum(b) / float(len(b))) < 0.5 for b in chunks):
            session_end_reason = 'subchance_accuracy'
            break
    if trial >= VAR_MAX_TRIALS:
        session_end_reason = 'max_trials'
        break

    trial += 1
    try:
        fresh_trial_count += 1
        trial_type = 'warmup' if fresh_trial_count <= warmup_target else 'main'
        side = blocks.side
        p_right_target = 1.0 if side == 'R' else 0.0
        block_index_this_trial = blocks.index
        block_trial_this_trial = blocks.trial_in_block   # responded trials earlier in this block
        just_switched_this_trial = blocks.just_switched()

        if quiescence_unlocked:
            quiescence_scale_this_trial = quiescence_obj.current_s
            quiescence_duration_this_trial = staircase.draw_quiescence_hold(
                quiescence_scale_this_trial, np.random)
        else:
            quiescence_scale_this_trial = None
            quiescence_duration_this_trial = VAR_QUIESCENCE_S

        # additions.txt T1: fresh per-trial seed, not a globally reseeded generator.
        stim_duration_this_trial = VAR_STIM_DURATION_S
        iti_this_trial = VAR_ITI_S
        trial_clicks, stim_seed, stim_resamples = click_train.draw_trial_clicks_with_cue(
            'AOS', side, stim_duration_this_trial)
        # Warmup attenuation is a multiplier ON TOP of the calibrated base amplitude (dB below the
        # calibrated target), not a substitute for it.
        attenuation_mult = (10 ** (-VAR_CLICK_ATTENUATION_DB / 20.0)
                             if _click_attenuation_remaining > 0 else 1.0)
        left_wave, right_wave = click_train.build_waveform(
            trial_clicks, hifi.sampling_rate,
            amplitude_scale_left=VAR_LEFT_AMPLITUDE_SCALE * attenuation_mult,
            amplitude_scale_right=VAR_RIGHT_AMPLITUDE_SCALE * attenuation_mult,
            stim_duration_s=stim_duration_this_trial)
        hifi.load(0, np.array([left_wave, right_wave]))
        hifi.push()

        runner.register('TRIAL_TYPE', trial_type)
        runner.register('TRIAL_SIDE', side)
        runner.register('BLOCK_INDEX', block_index_this_trial)
        runner.register('BLOCK_SIDE', side)
        runner.register('BLOCK_LENGTH', blocks.length)
        runner.register('BLOCK_TRIAL', block_trial_this_trial)
        runner.register('CLICK_TIMES_L', ','.join(
            '{0:.4f}'.format(t) for t in trial_clicks['left_times']))
        runner.register('CLICK_TIMES_R', ','.join(
            '{0:.4f}'.format(t) for t in trial_clicks['right_times']))
        runner.register('N_CLICKS_L', trial_clicks['n_left'])
        runner.register('N_CLICKS_R', trial_clicks['n_right'])
        runner.register('STIM_SEED', stim_seed)
        if stim_resamples:
            runner.register('STIM_RESAMPLES', stim_resamples)
        runner.register('CUE_ABORT_THRESHOLD_DEG', VAR_CUE_ABORT_THRESHOLD_DEG)
        runner.register('RESPONSE_THRESHOLD_DEG', VAR_RESPONSE_THRESHOLD_DEG)
        runner.register('QUIESCENCE_DUR_S', quiescence_duration_this_trial)
        if quiescence_scale_this_trial is not None:
            runner.register('QUIESCENCE_SCALE_S', quiescence_scale_this_trial)
        runner.register('STIM_DURATION_S', stim_duration_this_trial)
        runner.register('ITI_S', iti_this_trial)
        runner.register('CLICK_LEVEL_ATTENUATED', _click_attenuation_remaining > 0)

        print("Trial {0} ({1}): side={2}, n_L={3}, n_R={4}".format(
            trial, trial_type, side, trial_clicks['n_left'], trial_clicks['n_right']), flush=True)

        delay_duration_this_trial = random.uniform(VAR_DELAY_DURATION_MIN_S,
                                                     VAR_DELAY_DURATION_MAX_S)
        abort_timeout_this_trial = VAR_INCORRECT_ITI_S
        # J1 is a native, abort-guarded state at the END of cue_sma (DotOnsetDelay), so the wheel
        # must stay within the cue-abort tolerance right up to the moment the dot appears -- the
        # dot itself appears the instant `sma`'s first state (WheelDotPeriod) is entered, signalled
        # by SoftCode, not by a host timer. (It used to appear at a host-timed `send_epoch + J1`,
        # while WheelDotPeriod -- already listening for the choice -- started whenever the upload
        # and RUN happened to finish.)
        dot_onset_jitter_s = random.uniform(VAR_DOT_ONSET_JITTER_MIN_S, VAR_DOT_ONSET_JITTER_MAX_S)
        disappear_delay_s = random.uniform(VAR_DOT_DISAPPEAR_MIN_S, VAR_DOT_DISAPPEAR_MAX_S)
        error_flash_delay_s = random.uniform(VAR_ERROR_FLASH_DELAY_MIN_S, VAR_ERROR_FLASH_DELAY_MAX_S)

        cue_sma = StateMachine(my_bpod)

        # The hold-to-init wait runs INSIDE this machine (licks logged, Stop/Kill handled): the PC
        # injects SoftCode1 once the wheel has been still for the hold window (and starts the
        # audio right then) -- see _shared/trial_render.py.
        cue_sma.add_state(
            state_name='WaitQuiescence',
            state_timer=VAR_WAIT_QUIESCENCE_MAX_S,
            state_change_conditions={PC_EVENT_QUIESCENCE_MET: 'CuePeriod', Bpod.Events.Tup: 'exit'},
            output_actions=[])

        cue_sma.add_state(
            state_name='CuePeriod',
            state_timer=click_train.CLICK_START_OFFSET_S + stim_duration_this_trial,
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
                Bpod.Events.Tup: 'DotOnsetDelay',
            },
            output_actions=[])

        cue_sma.add_state(
            state_name='DotOnsetDelay',
            state_timer=dot_onset_jitter_s,
            state_change_conditions={
                wheel_abort_event_neg: 'WheelAbort',
                wheel_abort_event_pos: 'WheelAbort',
                Bpod.Events.Tup: 'CueComplete',
            },
            output_actions=[])

        cue_sma.add_state(
            state_name='WheelAbort',
            state_timer=0,
            state_change_conditions={Bpod.Events.Tup: 'AbortPunish'},
            output_actions=[(hifi_channel, hifi_stop_msg_id)])

        # SoftCode on entry so the screen flash lands in near-lockstep with the native LED pulse --
        # a hardware signal, instead of the render loop's own threshold comparison.
        cue_sma.add_state(
            state_name='AbortPunish',
            state_timer=VAR_ERROR_FLASH_S,
            state_change_conditions={Bpod.Events.Tup: 'AbortITI'},
            output_actions=[(VAR_GO_CUE_LED_CHANNEL, 255), ('SoftCode', SOFTCODE_PUNISH)])

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

        # Confirmed on real hardware: the correct response is turning the wheel OPPOSITE to the
        # click side, not toward it -- left_event/right_event's own left/right naming is purely a
        # raw-rotary-sign convention (see ALL_THRESHOLDS_DEG above), not guaranteed to match
        # physical turn direction one-to-one, and on this rig it's inverted relative to the
        # click side. If this rig's wiring/mounting ever changes, re-confirm this mapping on
        # hardware again rather than assuming it still holds.
        rewarded_event = left_event if side == 'R' else right_event
        unrewarded_event = right_event if rewarded_event == left_event else left_event

        # Built now (everything it needs is already known) so it can be uploaded and run the moment
        # cue_sma ends, from the same thread -- see _run_cue_then_choice() below.
        sma = StateMachine(my_bpod)

        # The choice window opens only once the PC confirms the dot is painted (same handshake as
        # Stage 1/2); a turn before that ends the trial as MovedBeforeDot.
        sma.add_state(
            state_name='DotOn',
            state_timer=VAR_DOT_HANDSHAKE_TIMEOUT_S,
            state_change_conditions={
                PC_EVENT_DOT_SHOWN: 'WheelDotPeriod',
                rewarded_event: 'MovedBeforeDot',
                unrewarded_event: 'MovedBeforeDot',
                Bpod.Events.Tup: 'DotTimeout',
            },
            output_actions=[('SoftCode', SOFTCODE_DOT_ON)])

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
            state_name='WheelDotPeriod',
            state_timer=VAR_RESPONSE_TIMEOUT_S,
            state_change_conditions={
                rewarded_event: 'Reward',
                unrewarded_event: 'ErrorConsumption',
                Bpod.Events.Tup: 'NoResponse',
            },
            output_actions=[(rotary_channel, reset_positions_trigger_id)])

        sma.add_state(
            state_name='Reward',
            state_timer=VAR_REWARD_DURATION,
            state_change_conditions={Bpod.Events.Tup: 'Consumption'},
            output_actions=[(Bpod.OutputChannels.Valve, 1), ('SoftCode', SOFTCODE_CROSSED)])

        sma.add_state(
            state_name='Consumption',
            state_timer=VAR_CONSUMPTION_WINDOW_S,
            state_change_conditions={'Port1In': 'ITI', Bpod.Events.Tup: 'ITI'},
            output_actions=[])

        sma.add_state(
            state_name='ErrorConsumption',
            state_timer=error_flash_delay_s,
            state_change_conditions={Bpod.Events.Tup: 'ErrorPunish'},
            output_actions=[('SoftCode', SOFTCODE_CROSSED)])

        sma.add_state(
            state_name='ErrorPunish',
            state_timer=VAR_ERROR_FLASH_S,
            state_change_conditions={Bpod.Events.Tup: 'IncorrectITI'},
            output_actions=[(VAR_GO_CUE_LED_CHANNEL, 255), ('SoftCode', SOFTCODE_PUNISH)])

        sma.add_state(
            state_name='IncorrectITI',
            state_timer=VAR_INCORRECT_ITI_S,
            state_change_conditions={Bpod.Events.Tup: 'exit'},
            output_actions=[])

        sma.add_state(
            state_name='NoResponse',
            state_timer=0,
            state_change_conditions={Bpod.Events.Tup: 'ITI'},
            output_actions=[])

        sma.add_state(
            state_name='ITI',
            state_timer=iti_this_trial,
            state_change_conditions={Bpod.Events.Tup: 'exit'},
            output_actions=[])

        runner.upload(cue_sma)
        camera.show_snippet(VAR_CAMERA_SNIPPET_S)
        camera.pump()

        # cue_sma and then (unless it aborted) `sma` run back-to-back on ONE background thread, so
        # there's no unwatched gap between the cue ending and the choice period starting. cue_sma's
        # own rows are registered from that thread, right after it returns and before `sma` runs,
        # so they land in the CSV between the two machines' rows, as always.
        cue_result = {}
        decision_result = {}
        cue_done = threading.Event()
        choice_started = threading.Event()
        decision_done = threading.Event()
        cue_audio = {}

        def _start_cue_audio():
            # Called by the render loop the moment quiescence is met and the cue machine has been
            # told to enter CuePeriod -- the audio starts within milliseconds of CuePeriod.
            hifi.play(0)
            cue_audio['t'] = host_clock.now() - log_python_t0

        def _run_cue_then_choice():
            try:
                cue_result['ran'] = runner.run_trial_state_machine(cue_sma, expect_marker=False)
                cue_result['anchor_t'] = runner.last_run_t
                cue_result['return_t'] = runner.last_return_t
                proceed = False
                if cue_result['ran']:
                    trial_obj = my_bpod.session.current_trial
                    cue_result['visited'] = trial_obj.states_durations
                    cue_result['events'] = trial_obj.get_all_timestamps_by_event()
                    runner.register('SM_SEND_TIME', cue_result['anchor_t'])
                    if 't' in cue_audio:
                        runner.register('CUE_ONSET_TIME',
                                        cue_audio['t'] + click_train.CLICK_START_OFFSET_S)
                    cue_result['lick_times_abs'], _cue_lick_times_out = \
                        runner.register_port_events('CUE_LICK', cue_result['events'],
                                                    cue_result['anchor_t'])
                    proceed = was_visited(cue_result['visited'], 'CueComplete')
                if proceed:
                    runner.begin_trial()   # fresh signal window for the choice machine
                    choice_started.set()
                cue_done.set()
                if proceed:
                    decision_result['ran'] = runner.run_trial_state_machine(
                        sma, arm_rotary=True, marker_state='WheelDotPeriod')
            except SystemExit:
                decision_result['killed'] = True
            except Exception:
                decision_result['error'] = traceback.format_exc()
            finally:
                cue_done.set()
                decision_done.set()

        decision_thread = threading.Thread(target=_run_cue_then_choice, daemon=True)
        prev_return_t = runner.last_return_t
        runner.begin_trial()   # main thread, BEFORE the Bpod thread -- see its docstring
        runner.start_quiescence(quiescence_duration_this_trial, VAR_STEADY_THRESHOLD_DEG)
        decision_thread.start()

        render = run_trial_render_loop(
            runner, dot, camera, decision_done, VAR_RESPONSE_THRESHOLD_DEG, disappear_delay_s,
            render_interval, SOFTCODE_DOT_ON, (SOFTCODE_CROSSED,),
            quiescence=(quiescence_duration_this_trial, VAR_STEADY_THRESHOLD_DEG),
            punish_softcode=SOFTCODE_PUNISH, punish_flash_s=VAR_ERROR_FLASH_S,
            on_quiescence_met=_start_cue_audio, dot_phase_event=choice_started)
        n_breaks = runner.quiescence_breaks
        last_initiation_time = host_clock.now()
        if quiescence_unlocked:
            quiescence_obj.record_outcome(success=(n_breaks == 0))

        decision_thread.join(timeout=3.0)
        if decision_thread.is_alive():
            print("WARNING: trial Bpod thread did not stop within 3s -- continuing anyway.",
                  flush=True)

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

        if not cue_result.get('ran', False) or 'visited' not in cue_result:
            print("Bpod stopped running trials (Stop/Kill) -- ending session early after trial "
                  "{0}.".format(trial), flush=True)
            break
        if prev_return_t is not None:
            runner.register('INTER_TRIAL_GAP_MS', (cue_result['anchor_t'] - prev_return_t) * 1e3)

        cue_send_t = cue_result['anchor_t']
        cue_visited = cue_result['visited']
        aborted = was_visited(cue_visited, 'WheelAbort')
        cue_lick_times_abs = cue_result['lick_times_abs']

        total_attempts += 1

        if aborted:
            total_aborts += 1
            consecutive_aborts += 1
            abort_time_abs = cue_send_t + cue_visited['WheelAbort'][-1][0]
            abort_epoch = 'delay' if was_visited(cue_visited, 'DelayPeriod') else 'cue'
            runner.register('ABORT', abort_time_abs)
            runner.register('ABORT_EPOCH', abort_epoch)
            runner.register('ABORT_ITI_S', abort_timeout_this_trial)
            # 'INCLUDED' means: did this trial produce a genuine L/R response (Reward or
            # ErrorConsumption)? False here (and for NoResponse) -- analysis applies its own
            # filters downstream, this is not a second, redundant filter.
            runner.register('INCLUDED', False)
            register_trial_timing(runner, render, quiescence_duration_this_trial, n_breaks, None,
                                  cue_visited, cue_send_t, response_state='WheelDotPeriod')

            print("Trial {0}: Abort ({1} epoch).".format(trial, abort_epoch), flush=True)

            bench_plots.add_trial(side, 0.0, VAR_RESPONSE_THRESHOLD_DEG, 'Abort',
                                   lick_times_abs=cue_lick_times_abs, trial_type=trial_type,
                                   p_right_target=p_right_target, iti_s=iti_this_trial,
                                   stim_duration_s=stim_duration_this_trial)
            continue

        consecutive_aborts = 0

        if not decision_result.get('ran', False):
            print("Bpod stopped running trials (Stop/Kill) -- ending session early after trial "
                  "{0}.".format(trial), flush=True)
            break

        # Host anchor for every choice-period Bpod-relative time below: captured right before RUN
        # (see TrialRunner.run_trial_state_machine()).
        send_t = runner.last_run_t
        runner.register('SM_RUN_TIME', send_t)
        runner.register('DOT_ONSET_JITTER_S', dot_onset_jitter_s)
        runner.register('CUE_TO_CHOICE_GAP_MS', (send_t - cue_result['return_t']) * 1e3)
        register_trial_timing(runner, render, quiescence_duration_this_trial, n_breaks, None,
                              my_bpod.session.current_trial.states_durations, send_t,
                              response_state='WheelDotPeriod')

        visited = my_bpod.session.current_trial.states_durations
        choice_events = my_bpod.session.current_trial.get_all_timestamps_by_event()

        if was_visited(visited, 'Reward'):
            outcome_state, outcome, rewarded = 'Reward', 'Reward', True
        elif was_visited(visited, 'ErrorConsumption'):
            outcome_state, outcome, rewarded = 'ErrorConsumption', 'NoReward', False
        elif was_visited(visited, 'NoResponse'):
            outcome_state, outcome, rewarded = 'NoResponse', 'NoResponse', False
        elif was_visited(visited, 'MovedBeforeDot'):
            outcome_state, outcome, rewarded = 'MovedBeforeDot', 'Abort', False
        elif was_visited(visited, 'DotTimeout'):
            outcome_state, outcome, rewarded = 'DotTimeout', 'Abort', False
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

        # accuracy_aos (session end) is conditional on a genuine L/R response -- a NoResponse
        # trial contributes to neither the numerator nor denominator here (it's already a
        # separately-tracked engagement signal via VAR_NO_INIT_TIMEOUT_S/consecutive_aborts, not
        # folded into "accuracy," matching the same "correct is not None" gate INCLUDED uses).
        if trial_type == 'main' and correct is not None:
            session_trial_count += 1
            main_trial_correct.append(bool(correct))
            if not just_switched_this_trial:
                subchance_correct.append(bool(correct))
            if block_index_this_trial > 0 and block_trial_this_trial < 5:
                after_switch_correct.append(bool(correct))
        # Only a genuine L/R response (warmup or main) advances the side block.
        if correct is not None and blocks.record_response():
            print("Block {0} done -- switching to {1} for {2} responded trials".format(
                blocks.index, blocks.side, blocks.length), flush=True)

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

        # DOT_ONSET_TIME (WheelDotPeriod's start = the dot confirmed on screen) is registered by
        # register_trial_timing() above. LED_ON_TIME is no longer registered: the LED stopped being
        # a go cue (punishment signal only).

        if outcome_state in ('Reward', 'ErrorConsumption'):
            threshold_crossing_t = send_t + visited[outcome_state][-1][0]
            dot_disappear_t = threshold_crossing_t + disappear_delay_s
            runner.register('THRESHOLD_CROSSING_TIME', threshold_crossing_t)
            runner.register('DOT_DISAPPEAR_JITTER_S', disappear_delay_s)
            if outcome_state == 'ErrorConsumption':
                runner.register('ERROR_FLASH_DELAY_S', error_flash_delay_s)
            runner.register('DOT_DISAPPEAR_TIME', dot_disappear_t)
            runner.register('CORRECT', rewarded)
        elif outcome_state == 'NoResponse':
            runner.register('TIMED_OUT', send_t + visited['NoResponse'][-1][0])

        print("Trial {0} ({1}): side={2}, {3}{4}".format(
            trial, trial_type, side, outcome, ' -> reward' if rewarded else ''), flush=True)

        choice_lick_times_abs, _choice_lick_times_out = runner.register_port_events(
            'CHOICE_LICK', choice_events, send_t)
        lick_times_abs = sorted(cue_lick_times_abs + choice_lick_times_abs)

        reward_time_abs = None
        if rewarded:
            reward_time_abs = send_t + visited['Reward'][-1][0]

        # WheelDotPeriod is state index 0 of `sma`, so its own trial-relative outcome timestamp IS
        # already "time since go-cue" -- no extra send_t arithmetic needed. click_diff (signed
        # click-count evidence) feeds the psychometric curve; both are None (and simply not fed to
        # the panel) for a NoResponse trial, matching add_trial()'s own gating.
        response_time_s = (visited[outcome_state][-1][0]
                            if outcome_state in ('Reward', 'ErrorConsumption') else None)
        click_diff = trial_clicks['realized_delta'] if response_time_s is not None else None

        bench_plots.add_trial(side, VAR_RESPONSE_THRESHOLD_DEG if response is not None else 0.0,
                               VAR_RESPONSE_THRESHOLD_DEG, outcome,
                               lick_times_abs=lick_times_abs, reward_time_abs=reward_time_abs,
                               trial_type=trial_type, p_right_target=p_right_target,
                               response_time_s=response_time_s, click_diff=click_diff,
                               iti_s=iti_this_trial, stim_duration_s=stim_duration_this_trial)

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

# A qualifying session shortens the side blocks one step, toward Stage 4's per-trial sides.
if this_session_gates_met:
    _new_lo, _new_hi = staircase.shrink_stage3_block_range(VAR_BLOCK_MIN, VAR_BLOCK_MAX)
    state.set('stage3_block_min', _new_lo)
    state.set('stage3_block_max', _new_hi)


if quiescence_unlocked:
    state.set('quiescence_s', quiescence_obj.current_s)
    state.set('quiescence_consecutive_successes', quiescence_obj.consecutive_successes)
    state.set('quiescence_consecutive_failures', quiescence_obj.consecutive_failures)

state.save()

print("Session: {0} side block(s) from {1}, {2} main trials, accuracy(AOS)={3:.1%}, "
      "abort_rate={4:.1%}, gates_met={5}".format(
          blocks.index + 1, VAR_FIRST_BLOCK_SIDE, session_trial_count, accuracy_aos, abort_rate,
          this_session_gates_met), flush=True)
# Is it listening? A mouse using the clicks switches at once; one going by reward history keeps
# turning the old way for a few trials after each switch.
_n_after = len(after_switch_correct)
print("Accuracy on the first 5 responded trials after a switch: {0} (n={1}); all main trials: "
      "{2:.1%}".format('{0:.1%}'.format(sum(after_switch_correct) / float(_n_after))
                       if _n_after else 'n/a', _n_after, accuracy_aos), flush=True)
print("Block range next session: {0}-{1}".format(
    state.get('stage3_block_min'), state.get('stage3_block_max')), flush=True)
print("ADVANCE-READY (3 consecutive qualifying sessions): {0}".format(
    "yes" if advance_ready else "no"), flush=True)
if not advance_ready:
    print("  (history so far: {0})".format(history), flush=True)
if accuracy_aos <= 0.5 and len(main_trial_correct) >= VAR_SUBCHANCE_BLOCK_SIZE:
    print("NOTE: accuracy is at or below chance -- per the doc's Tag J, before declaring a "
          "non-learner: (1) verify per-rig speaker SPL calibration measured at the head position, "
          "(2) inspect ears for infection, (3) confirm response to a brief supra-threshold "
          "unilateral click-train block.", flush=True)

_cleanup_and_export()

print("Close the plot window to exit.", flush=True)
plt.ioff()
plt.show()
