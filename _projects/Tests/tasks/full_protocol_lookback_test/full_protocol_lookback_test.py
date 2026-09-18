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

**Camera recording, added in this pass**: `_shared/camera_recorder.py`'s `CameraRecorder` records
the whole session on its own background thread (fire-and-forget, time-synced to this script's own
`log_python_t0`, same clock every VAL registration already uses) regardless of anything below. The
live preview is deliberately **snippet-only, never continuous** -- `camera.show_snippet(...)` is
called at exactly two moments per trial (decision-period start, and right after the choice
resolves) rather than repainting on every render-loop iteration. This script already runs a
`WHEEL_POS` poll thread plus a `decision_thread` and has a documented real interpreter-crash
history from GIL/native-toolkit pressure between competing threads (see CLAUDE.md) -- a third
background thread was unavoidable for recording itself, but confining the (comparatively
expensive) per-frame preview conversion+repaint to two ~1s windows instead of the whole ~5s
decision period was a confirmed, worthwhile reduction in that same risk, not just a cosmetic
choice.

**additions.txt instrumentation pass, added in this pass**: per-click times (`CLICK_TIMES_L/R`,
`N_CLICKS_L/R`, `STIM_SEED` -- a fresh per-trial seed, not a globally reseeded generator, so a
trial's exact click train is reproducible offline from the logged seed alone); Bpod-send anchors
(`SM_SEND_TIME` once per state-machine run, `CUE_ONSET_TIME` for the cue machine only) so a
trial-relative Bpod timestamp can be mapped onto the session clock without wrongly assuming the
send instant equals `TRIAL_START` (false here -- the hold-to-init wait runs first, a variable-
length gap); the cue state machine's single `CuePeriod` split into `CuePeriod` + a new
`DelayPeriod` (identical wheel-abort monitoring, same total enforced hold to the millisecond --
verify via realized `states_durations` on hardware) so `ABORT_EPOCH` can distinguish a cue-phase
abort from a delay-phase one; per-trial staircase/timing state (`CUE_ABORT_THRESHOLD_DEG`,
`RESPONSE_THRESHOLD_DEG`, `QUIESCENCE_DUR_S`, `ITI_S` -- frozen values here, but registered every
trial for convention-consistency with Stage 3/4, where they genuinely vary); `QUIESCENCE_BREAKS`
(Stage 1/2 already had it, this script didn't); `SESSION_END_REASON='completed'` on the natural-
end path (deliberately still absent on every early-end path -- same "absence signals abnormal
end" convention Stage 1/2 already use); `SESSION_WATER_UL` at session end; and a genuinely new
`_export_session_struct()`/`_cleanup_and_export()` pair (this file never called
`session_struct_export.py` at all before) wired into every exit path, including a real,
previously-unfixed gap: an unhandled per-trial exception used to `raise` straight out of the
whole trial loop, skipping teardown (`camera`/`dot`/`hifi`/`rotary`/`my_bpod` all left unclosed,
no struct exported) entirely -- `_cleanup_and_export()` now runs first on that path too.
Deliberately NOT touched: `click_train_v2.py`'s ISI floor, rate calibration, or Poisson generation
math; the difficulty grid; any staircase/threshold/timing parameter's own value; or trial
scheduling logic (`trial_scheduler.py`) -- per additions.txt's own explicit scope.
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
sys.path.insert(0, os.path.join(_TASK_DIR, '..', '..', '..', '..', 'Calibration'))
# AuditoryEvidenceAccum's own _wheel_shaping_shared/ -- only for session_struct_export.py (+ its
# session_csv_parser.py dependency), which are protocol-agnostic parsing/export utilities despite
# their current location (see additions.txt T8: this script never had a struct-export call at all).
# Not importing anything paradigm-specific from there (no staircase/session_state/direction_tracker).
sys.path.insert(0, os.path.join(_TASK_DIR, '..', '..', '..', 'AuditoryEvidenceAccum', 'tasks',
                                 '_wheel_shaping_shared'))
import click_train_v2 as click_train
import trial_scheduler as ts
from live_plots_lookback import LookbackBenchPlots
from bpod_trial_helpers import TrialRunner, was_visited
import rotary_setup
import hifi_setup
import dot_display
from camera_recorder import CameraRecorder
from liquid_calibration import get_reward_duration_s
import session_struct_export

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

VAR_TARGET_SPL_DB = 70.0            # not specified anywhere -- flagged/tunable. Waveform amplitude
                                     # is derived from this via Calibration/sound_calibration.py's
                                     # fitted curve (see VAR_LEFT_AMPLITUDE_SCALE/
                                     # VAR_RIGHT_AMPLITUDE_SCALE below).

VAR_REWARD_UL = 4.0                 # target reward volume -- valve open time is derived from this
                                     # via Calibration/liquid_calibration.py's own fitted curve
                                     # (falls back to a hardcoded 0.1s if no calibration data/fit
                                     # exists yet on this machine -- see get_reward_duration_s()).
VAR_REWARD_DURATION = get_reward_duration_s(VAR_REWARD_UL)
VAR_CONSUMPTION_WINDOW_S = 3.0
VAR_ITI = 0.1 #2s
VAR_ABORT_ITI_S = 5.0               # cue-period wheel-abort timeout (unchanged from before)
VAR_INCORRECT_ITI_S = 0.1 #5s           # longer than VAR_ITI -- error timeout for a wrong turn
VAR_STILL_POLL_HZ = 100              # additions.txt T4: raised 50->100. NEEDS ON-RIG CONFIRMATION
VAR_POLL_HZ = 100                    # that the poll thread keeps up without delaying state-machine
                                      # handling -- if not, fall back to 50/50 (still a real
                                      # improvement over the old 50/10 split) and note which rate
                                      # was actually used when reporting results from a real session.
VAR_ROTARY_USB_PORT = None

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

VAR_CAMERA_INDEX = None                      # None = auto-discover, see
                                              # camera_recorder.discover_camera()
VAR_CAMERA_OUTPUT_PATH = 'session_video.avi' # relative to cwd -- lands in the real session
                                              # folder when run for real via the GUI's Run button.
VAR_CAMERA_FPS = 30.0
VAR_CAMERA_PREVIEW = True             # preview shown only in short snippets (see
                                       # camera.show_snippet() calls below), never continuously --
                                       # this script already runs a WHEEL_POS poll thread + a
                                       # decision_thread and has a documented GIL-pressure crash
                                       # history (see CLAUDE.md); a continuous preview repaint on
                                       # every render-loop iteration would add to that risk, while
                                       # snippets confirmed as a meaningful lag reduction instead.
VAR_CAMERA_SNIPPET_S = 1.0            # snippet duration for both "decision period start" and
                                       # "after choice" preview windows below.

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

def _export_session_struct(csv_path):
    """ Every VAR_* constant this run used, harvested automatically -- stays complete as new
    parameters get added later, no hand-maintained list to fall out of sync. Wrapped in try/except
    so an export hiccup (e.g. a scipy/disk issue) never blocks session teardown -- the animal's run
    is already fully logged in the CSV regardless of whether this convenience export succeeds. Same
    helper shape as stage1_wheel_shaping.py/stage2_threshold_staircase.py's own. """
    task_params = {k: v for k, v in globals().items() if k.startswith('VAR_')}
    try:
        mat_path, json_path = session_struct_export.export_session_struct(csv_path, task_params)
        print("Session struct exported: {0} / {1}".format(mat_path, json_path), flush=True)
    except Exception as err:
        print("WARNING: session struct export failed: {0}".format(err), flush=True)


def _cleanup_and_export():
    """ Shared teardown, called from every exit path (normal end, Stop-triggered break, Kill,
    and an unhandled per-trial exception -- see each call site below) so hardware connections and
    the struct export are never skipped regardless of how the session ended. Fixes a real gap in
    the previous version of this file: an unhandled exception used to `raise` straight out of the
    trial loop, skipping this teardown entirely (confirmed: `camera`/`dot`/`hifi`/`rotary`/
    `my_bpod` were all left unclosed, and no struct was ever exported, on that path). Only ever
    called once per run (each call site's own control flow -- sys.exit()/re-raise/falling out of
    the loop -- ensures no path reaches more than one of them). """
    try:
        runner.register('SESSION_WATER_UL', session_reward_count * VAR_REWARD_UL)
    except Exception as err:
        print("WARNING: could not register SESSION_WATER_UL: {0}".format(err), flush=True)
    camera.close()
    dot.close()
    hifi.close()
    rotary.close()
    csv_path = my_bpod.session._path   # grab before close() -- close() deletes the Session object
                                        # that holds it (see CLAUDE.md)
    my_bpod.close()
    _export_session_struct(csv_path)


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

VAR_LEFT_AMPLITUDE_SCALE, VAR_RIGHT_AMPLITUDE_SCALE = hifi_setup.compute_calibrated_amplitudes(
    VAR_TARGET_SPL_DB, click_train.VAR_LEFT_FREQ_HZ, click_train.VAR_RIGHT_FREQ_HZ)

my_bpod.register_value('LEFT_THRESHOLD_DEG', VAR_LEFT_THRESHOLD_DEG)
my_bpod.register_value('RIGHT_THRESHOLD_DEG', VAR_RIGHT_THRESHOLD_DEG)
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

# Geometry-aware gain: hitting VAR_RIGHT_THRESHOLD_DEG on the wheel should move the dot to
# VAR_DOT_EDGE_FRACTION of the actual screen's half-width, not a fixed guessed px/wheel-deg
# constant -- same fix applied to dot_wheel_test.py after the earlier placeholder (4.0) turned out
# to barely move the dot at all. rotary_setup.screen_direction_gain() applies this rotary's
# confirmed wheel->screen sign correction (see rotary_setup.py) -- same centralized fix as every
# other dot-coupled script, instead of a per-script sign flip.
screen_width_px = dot.get_screen_width_px()
dot_gain = rotary_setup.screen_direction_gain(
    (VAR_DOT_EDGE_FRACTION * (screen_width_px / 2.0)) / VAR_RIGHT_THRESHOLD_DEG)
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

# --- trial loop -----------------------------------------------------------------------------------

print("Starting up to {0} trials, stopping early on the disengagement circuit-breaker".format(
    VAR_MAX_TRIALS), flush=True)

trial = 0
session_reward_count = 0   # feeds SESSION_WATER_UL at session end (additions.txt T8)
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
        runner.register('QUIESCENCE_BREAKS', n_breaks)
        dot.clear()
        dot.pump()
        camera.pump()

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

        # additions.txt T1: a fresh per-trial seed, not a globally-reseeded generator -- the exact
        # click train is then reproducible offline from STIM_SEED alone (regenerate_trial_clicks
        # with the same difficulty/side/rng seed reproduces left_times/right_times exactly).
        stim_seed = int(np.random.randint(0, 2 ** 31 - 1))
        trial_clicks = click_train.generate_trial_clicks(
            difficulty, side, rng=np.random.RandomState(stim_seed))
        left_wave, right_wave = click_train.build_waveform(
            trial_clicks, hifi.sampling_rate,
            amplitude_scale_left=VAR_LEFT_AMPLITUDE_SCALE,
            amplitude_scale_right=VAR_RIGHT_AMPLITUDE_SCALE)
        hifi.load(0, np.array([left_wave, right_wave]))
        hifi.push()

        runner.register('DIFFICULTY_LEVEL', difficulty)
        runner.register('TRIAL_SIDE', side)
        runner.register('REALIZED_DELTA', trial_clicks['realized_delta'])
        runner.register('TRIAL_TYPE', trial_type)
        runner.register('P_RIGHT_TARGET', p_right_target)
        runner.register('RECENT_RIGHT_FRACTION', recent_right_frac)

        # additions.txt T1: per-click times, generator-relative (i.e. relative to stimulus onset,
        # NOT offset by CLICK_START_OFFSET_S -- CUE_ONSET_TIME below is what analysis adds back to
        # reconstruct an absolute click time), comma-joined at 4dp. Empty side -> empty string.
        runner.register('CLICK_TIMES_L', ','.join(
            '{0:.4f}'.format(t) for t in trial_clicks['left_times']))
        runner.register('CLICK_TIMES_R', ','.join(
            '{0:.4f}'.format(t) for t in trial_clicks['right_times']))
        runner.register('N_CLICKS_L', trial_clicks['n_left'])
        runner.register('N_CLICKS_R', trial_clicks['n_right'])
        runner.register('STIM_SEED', stim_seed)

        # additions.txt T6: the staircase/timing state actually in force THIS trial -- Bpod-side
        # values here are frozen (this protocol has no active staircase), but registering them per
        # trial (not just once at session start) keeps the convention identical to Stage 3/4, where
        # they genuinely do vary trial-to-trial.
        runner.register('CUE_ABORT_THRESHOLD_DEG', VAR_CUE_ABORT_THRESHOLD_DEG)
        runner.register('RESPONSE_THRESHOLD_DEG', VAR_RIGHT_THRESHOLD_DEG)
        runner.register('QUIESCENCE_DUR_S', required_hold)
        runner.register('ITI_S', VAR_ITI)

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
        # additions.txt T2: the Bpod-send anchor for this cue state machine, plus the moment the
        # click waveform's own onset (past the bilateral marker pulse + gap) actually plays --
        # without SM_SEND_TIME, a state's trial-relative Bpod timestamp can only be mapped to the
        # session clock by wrongly assuming the state machine's send instant equals TRIAL_START,
        # which is false here (the hold-to-init wait runs first, a variable-length gap).
        runner.register('SM_SEND_TIME', cue_send_t)
        runner.register('CUE_ONSET_TIME', cue_send_t + click_train.CLICK_START_OFFSET_S)

        cue_sma = StateMachine(my_bpod)

        # additions.txt T5: CuePeriod (the click train itself) and DelayPeriod (the enforced silent
        # gap after it) are now separate states with identical wheel-abort monitoring, so an abort
        # can be attributed to the right epoch -- previously one combined CuePeriod state spanned
        # both, so every abort during the whole waveform was indistinguishable. The 0.1s safety
        # buffer moves from CuePeriod's own timer to DelayPeriod's, so the TOTAL enforced hold
        # (CuePeriod + DelayPeriod) is unchanged to the millisecond -- verify via realized
        # states_durations on hardware, not by reading this code.
        cue_sma.add_state(
            state_name='CuePeriod',
            state_timer=click_train.CLICK_START_OFFSET_S + click_train.VAR_STIM_DURATION_S,
            state_change_conditions={
                wheel_abort_event_neg: 'WheelAbort',
                wheel_abort_event_pos: 'WheelAbort',
                Bpod.Events.Tup: 'DelayPeriod',
            },
            output_actions=[])

        cue_sma.add_state(
            state_name='DelayPeriod',
            state_timer=click_train.VAR_DELAY_DURATION_S + 0.1,
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
            # additions.txt T5: DelayPeriod is only genuinely entered (a real, non-nan start time)
            # if CuePeriod completed without aborting -- so "DelayPeriod visited" cleanly
            # distinguishes which epoch this abort actually happened in.
            abort_epoch = 'delay' if was_visited(cue_visited, 'DelayPeriod') else 'cue'
            runner.register('ABORT_EPOCH', abort_epoch)

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
        # additions.txt T2: the decision machine's own send anchor (a trial with a cue machine +
        # decision machine gets two SM_SEND_TIME rows -- this is the second).
        runner.register('SM_SEND_TIME', send_t)

        # "First second after task start" -- one of the two brief preview windows requested (see
        # VAR_CAMERA_PREVIEW's own comment for why this is snippet-only, not continuous).
        camera.show_snippet(VAR_CAMERA_SNIPPET_S)

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
            camera.pump()
            time.sleep(render_interval)

        decision_thread.join(timeout=3.0)
        if decision_thread.is_alive():
            print("WARNING: decision-period Bpod thread did not stop within 3s -- continuing "
                  "anyway.", flush=True)

        dot.clear()
        dot.pump()

        # "After choice" -- the second brief preview window; outcome is now known (decision_done
        # is set), so this shows what the animal/rig looked like right as the choice resolved.
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

        # --- trial-lookback scheduler update ---
        if outcome_state == 'Reward':
            session_reward_count += 1   # feeds SESSION_WATER_UL at session end
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
        # Fix (was a real gap, not intentional): re-raising here used to skip teardown entirely --
        # nothing after this while/else block ever ran, so camera/dot/hifi/rotary/my_bpod were all
        # left open and no struct was exported on any trial-loop exception. Clean up FIRST, then
        # still re-raise so the failure is exactly as loud/visible as before.
        _cleanup_and_export()
        raise
else:
    runner.register('SESSION_END_REASON', 'completed')
    print("Done: reached VAR_MAX_TRIALS ({0}) without the circuit-breaker firing".format(
        VAR_MAX_TRIALS), flush=True)

# Reached on natural completion (the else: clause above already ran) OR any Stop-triggered break
# (should_stop_session, a mid-trial Stop signal) -- SESSION_END_REASON is deliberately left
# unregistered on the break paths (absence itself signals an abnormal/early end, same convention
# documented in CLAUDE.md for Stage 1/2). The Kill and exception paths never reach here at all --
# each already called _cleanup_and_export() itself above, before exiting/re-raising.
_cleanup_and_export()

print("Close the plot windows to exit.", flush=True)
plt.ioff()
plt.show()
