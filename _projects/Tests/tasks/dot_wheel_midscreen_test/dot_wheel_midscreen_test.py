# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Bench test: a wheel turn during the decision period drags a filled dot across a second monitor,
coupled 1:1 to wheel position -- training_protocol.md's dot stimulus (SS1.2: "one parameter, no
internal structure, no phase or orientation"), replacing the Gabor patch used in this session's
earlier bench tests/combined tasks. Forked from gabor_wheel_test.py (kept byte-identical/untouched
-- still its own valid, separately-tested bench test) rather than edited in place, using
`_shared/dot_display.py`'s `DotDisplay` in place of `GaborDisplay`.

**Scope, per current instruction: just the dot movement across the screen coupled to wheel
movement.** No reward/ITI distinction between correct/incorrect yet -- mirrors
gabor_wheel_test.py's own original first-pass scope (pure coupling/rendering) before reward scoring
was layered on in that script's later revision. Turning the wheel either direction past threshold
simply ends the decision period and moves to a plain ITI; everything about difficulty grids,
click trains, reward, and the full staircase/training-protocol machinery in training_protocol.md is
explicitly out of scope here, to be built later.

**Onset/disappear jitter use training_protocol.md's own J1/J2 ranges (SS1.5), not the wider ranges
already hardware-tested for the Gabor combined tasks:**
- J1 (LED -> dot visible): 0.1-0.2s, positive-only, from decision-period onset.
- J2 (threshold crossing -> dot off): 0.4-0.9s, positive-only, from crossing -- same range already
  used and hardware-tested for the Gabor disappear jitter, so no new risk there.
Both drawn once, up front, per trial -- never decided reactively mid-trial, same convention already
established for the Gabor jitters.

Hold-to-init, module connection/setup, and the background WHEEL_POS-publishing/bounded-join
plumbing reuse `_projects/_shared/` (`bpod_trial_helpers.py`'s `TrialRunner`/`was_visited`,
`rotary_setup.py`) exactly like `gabor_wheel_test.py`. `VAR_REQUIRE_NO_LICK` defaults to False here
for the same reason it did there -- no lick sensor assumed relevant to this particular bench rig.

**Thread roles are inverted, same as gabor_wheel_test.py and every combined Gabor+clicks task**:
the decision-period Bpod call runs in a *background* thread while the *foreground* (main) thread
runs the dot render loop, because Qt's event loop/repaints have to happen on the main thread.
`SystemExit` from a Kill received on that background thread is caught and re-raised on the main
thread once noticed -- identical handling to every other script in this project that uses this
pattern.

Run this like any other PyBpod task, via the GUI's Run button -- requires the Bpod board and
rotary encoder module physically connected via USB, and a second monitor connected for the dot
display (falls back to screen 0 with a printed warning if only one screen is detected).

**Forked from dot_wheel_test.py (kept byte-identical/untouched -- still its own valid, separately-
tested bench test), not a parametrized variant of it, per current instruction.** The only
difference from dot_wheel_test.py: this script uses `_shared/dot_display.py`'s new
`MiddleScreenDotDisplay` instead of `DotDisplay`. Reason: this rig's three physical dot-stimulus
monitors do NOT show up as three separate Qt screens -- a direct, read-only `QApplication().
screens()` query on this machine found only 2 screens total, `DISPLAY1` (a separate 1920x1080
monitor) and `DISPLAY2` at 6144x1536 (6144 / 3 = 2048px, a standard panel width -- almost certainly
the three physical rig panels bonded by the GPU into one combined Qt screen, not three independent
ones). Because of that, `DotDisplay` alone draws the dot relative to the FULL 6144px-wide combined
window, so its full range of motion visibly crosses from the middle physical monitor onto the two
outer ones as the wheel turns -- `MiddleScreenDotDisplay` fixes this by always painting the two
outer thirds of that same combined window solid black and confining the dot's rendering AND its
full range of motion (via `get_screen_width_px()` returning only the active column's width, which
this script's existing VAR_DOT_EDGE_FRACTION-based gain calibration already uses) to one
VAR_ACTIVE_MONITOR_INDEX-th of VAR_N_PHYSICAL_MONITORS_IN_SPAN equal-width columns. The equal-
thirds split (2048px each) should be visually double-checked against the real monitor bezels the
first time this runs on hardware.
"""
import os
import random
import sys
import threading
import time
import traceback

_TASK_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_TASK_DIR, '..', '..', '..', '_shared'))
from bpod_trial_helpers import TrialRunner, was_visited
import rotary_setup
from dot_display import MiddleScreenDotDisplay

from pybpodapi.protocol import Bpod, StateMachine

VAR_N_TRIALS = 20
VAR_HOLD_MIN_S = 0.1
VAR_HOLD_MAX_S = 0.5
VAR_STEADY_THRESHOLD_DEG = 5       # same as gabor_wheel_test.py -- not being retuned here
VAR_LEFT_THRESHOLD_DEG = -30
VAR_RIGHT_THRESHOLD_DEG = 30
VAR_RESPONSE_TIMEOUT = 5
VAR_ITI = 2
VAR_STILL_POLL_HZ = 50
VAR_POLL_HZ = 10
VAR_ROTARY_USB_PORT = None
VAR_REQUIRE_NO_LICK = False        # no lick sensor assumed relevant to this bench rig

VAR_GO_CUE_LED_CHANNEL = 'PWM1'   # Port 1's built-in LED, same convention as every other script

VAR_DOT_SCREEN_INDEX = 1          # the combined 3-panel Qt screen; falls back to 0 with a warning
                                   # if not found -- see module docstring for why this is one
                                   # combined screen, not the middle monitor directly.
VAR_N_PHYSICAL_MONITORS_IN_SPAN = 3   # confirmed via screens(): DISPLAY2 is 6144px wide, 6144/3 =
                                       # 2048px per physical panel -- see module docstring.
VAR_ACTIVE_MONITOR_INDEX = 1          # 0=left, 1=middle, 2=right -- middle panel only.
VAR_DOT_DIAMETER_PX = 60          # UNCONFIRMED against training_protocol.md SS1.2's 3-4 visual-deg
                                   # spec -- this is a guessed pixel value, not derived from it.
                                   # Converting visual degrees -> px needs the monitor's physical
                                   # width and the animal's viewing distance, neither measured yet
                                   # (SS Part 6, item 1). Once both exist, replace this line with:
                                   #   from dot_display import visual_deg_to_px
                                   #   VAR_DOT_DIAMETER_PX = visual_deg_to_px(
                                   #       3.5, dot.get_screen_width_px(),
                                   #       screen_width_mm, viewing_distance_mm)
                                   # (3.5 = midpoint of the doc's 3-4deg range; needs dot_display's
                                   # get_screen_width_px(), so this can only run after DotDisplay is
                                   # constructed, same ordering already used for the gain below.)
VAR_DOT_BACKGROUND_GRAY = 128
VAR_DOT_GRAY = 0                  # full black, per training_protocol.md SS1.2's default (doc also
                                   # floats a sub-maximal-contrast option -- flagged, not built here)
VAR_DOT_EDGE_FRACTION = 0.9        # training_protocol.md SS1.3: place the threshold at ~90% of edge
                                    # azimuth, so the dot freezes still visible ("push it off the
                                    # edge") -- gain is derived below from the ACTUAL resolved
                                    # screen width so hitting VAR_RIGHT_THRESHOLD_DEG lands the dot
                                    # at this fraction of the half-screen-width, rather than a fixed
                                    # guessed px/deg constant (the earlier flat 4.0 barely moved the
                                    # dot at all -- most of the screen went unused).
VAR_RENDER_HZ = 60

VAR_DOT_ONSET_JITTER_MIN_S = 0.1   # J1 (training_protocol.md SS1.5) -- narrower than the Gabor
VAR_DOT_ONSET_JITTER_MAX_S = 0.2   # tests' 0.05-0.35s, an intentional difference
VAR_DOT_DISAPPEAR_MIN_S = 0.4      # J2 -- same range already hardware-tested for Gabor
VAR_DOT_DISAPPEAR_MAX_S = 0.9

ALL_THRESHOLDS_DEG = [-VAR_STEADY_THRESHOLD_DEG, VAR_STEADY_THRESHOLD_DEG,
                       VAR_LEFT_THRESHOLD_DEG, VAR_RIGHT_THRESHOLD_DEG]

# --- connect to Bpod, resolve modules -------------------------------------------------------------

my_bpod = Bpod()
print("Connected to Bpod on {0}".format(my_bpod.serial_port), flush=True)

rotary, rotary_bpod_module = rotary_setup.connect_rotary(my_bpod, usb_port=VAR_ROTARY_USB_PORT)
reset_positions_trigger_id, rotary_channel = rotary_setup.build_reset_trigger(rotary_bpod_module)

event_names = rotary_setup.set_and_enable_thresholds(rotary, ALL_THRESHOLDS_DEG)
rotary.enable_evt_transmission()
left_event, right_event = event_names[2], event_names[3]

my_bpod.register_value('LEFT_THRESHOLD_DEG', VAR_LEFT_THRESHOLD_DEG)
my_bpod.register_value('RIGHT_THRESHOLD_DEG', VAR_RIGHT_THRESHOLD_DEG)

log_python_t0 = time.time()
runner = TrialRunner(my_bpod, rotary, log_python_t0, still_poll_hz=VAR_STILL_POLL_HZ,
                      poll_hz=VAR_POLL_HZ)

dot = MiddleScreenDotDisplay(screen_index=VAR_DOT_SCREEN_INDEX,
                              n_segments=VAR_N_PHYSICAL_MONITORS_IN_SPAN,
                              active_segment_index=VAR_ACTIVE_MONITOR_INDEX,
                              diameter_px=VAR_DOT_DIAMETER_PX,
                              background_gray=VAR_DOT_BACKGROUND_GRAY, dot_gray=VAR_DOT_GRAY)
dot.show()
dot.clear()

# Geometry-aware gain: hitting VAR_RIGHT_THRESHOLD_DEG on the wheel should move the dot to
# VAR_DOT_EDGE_FRACTION of the actual screen's half-width, not a fixed guessed px/wheel-deg
# constant -- see VAR_DOT_EDGE_FRACTION's own comment above. get_screen_width_px() returns just the
# active monitor's column width (not the full 3-panel-wide window), so this keeps the dot's full
# range of motion confined to that one physical monitor automatically.
screen_width_px = dot.get_screen_width_px()
# rotary_setup.screen_direction_gain() applies this rotary's confirmed wheel->screen sign
# correction -- see that function's own docstring/WHEEL_TO_SCREEN_SIGN comment in rotary_setup.py.
# Centralized there instead of a per-script sign flip, so any future recalibration only needs to
# change one constant, not every display-coupling task script.
dot_gain = rotary_setup.screen_direction_gain(
    (VAR_DOT_EDGE_FRACTION * (screen_width_px / 2.0)) / VAR_RIGHT_THRESHOLD_DEG)
dot.set_deg_to_px_gain(dot_gain)
print("Dot gain calibrated to {0:.2f} px/wheel-deg (active screen width {1}px, edge fraction "
      "{2})".format(dot_gain, screen_width_px, VAR_DOT_EDGE_FRACTION), flush=True)

render_interval = 1.0 / VAR_RENDER_HZ

# --- trial loop -----------------------------------------------------------------------------------

print("Starting {0} trials".format(VAR_N_TRIALS), flush=True)

for trial in range(VAR_N_TRIALS):
    try:
        required_hold = random.uniform(VAR_HOLD_MIN_S, VAR_HOLD_MAX_S)

        rotary.disable_evt_transmission()
        n_breaks = runner.wait_for_held_steady(required_hold, VAR_STEADY_THRESHOLD_DEG,
                                                require_no_lick=VAR_REQUIRE_NO_LICK)
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
        dot.clear()
        dot.pump()

        print("Trial {0}: held steady -- decision period open".format(trial + 1), flush=True)

        # Both jitters drawn once, up front -- never decided reactively mid-trial.
        dot_onset_delay = random.uniform(VAR_DOT_ONSET_JITTER_MIN_S, VAR_DOT_ONSET_JITTER_MAX_S)
        disappear_delay_s = random.uniform(VAR_DOT_DISAPPEAR_MIN_S, VAR_DOT_DISAPPEAR_MAX_S)

        send_epoch = time.time()
        send_t = send_epoch - log_python_t0

        decision_sma = StateMachine(my_bpod)

        decision_sma.add_state(
            state_name='WheelDotPeriod',
            state_timer=VAR_RESPONSE_TIMEOUT,
            state_change_conditions={
                left_event: 'ITI',
                right_event: 'ITI',
                Bpod.Events.Tup: 'ITI',
            },
            output_actions=[(rotary_channel, reset_positions_trigger_id),
                             (VAR_GO_CUE_LED_CHANNEL, 255)])

        decision_sma.add_state(
            state_name='ITI',
            state_timer=VAR_ITI,
            state_change_conditions={Bpod.Events.Tup: 'exit'},
            output_actions=[(VAR_GO_CUE_LED_CHANNEL, 0)])

        # Inverted from the usual pattern -- see module docstring. The Bpod call runs in a
        # background thread; this (main) thread renders the dot instead of blocking here.
        decision_result = {}
        decision_done = threading.Event()

        def _run_decision():
            try:
                decision_result['ran'] = runner.run_trial_state_machine(decision_sma)
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
            rotary.close()
            my_bpod.close()
            sys.exit(0)

        if not decision_result.get('ran', False):
            print("Bpod stopped running trials (Stop/Kill) -- ending session early after trial "
                  "{0}/{1}.".format(trial + 1, VAR_N_TRIALS), flush=True)
            break

        # ITI is visited on every path (crossing OR timeout both transition into it), so it can't
        # distinguish the two -- checking whether left_event/right_event actually fired can.
        # WheelDotPeriod's own end -- send_t + visited['WheelDotPeriod'][-1][0] -- is the crossing
        # instant either way (a timeout ends that same state at VAR_RESPONSE_TIMEOUT instead).
        visited = my_bpod.session.current_trial.states_durations
        choice_events = my_bpod.session.current_trial.get_all_timestamps_by_event()
        crossed = bool(choice_events.get(left_event)) or bool(choice_events.get(right_event))
        crossing_t = send_t + visited['WheelDotPeriod'][-1][0] if crossed else None

        runner.register('DOT_ONSET_JITTER_S', dot_onset_delay)
        runner.register('DOT_ONSET_TIME', send_t + dot_onset_delay)
        if crossing_t is not None:
            runner.register('THRESHOLD_CROSSING_TIME', crossing_t)
            runner.register('DOT_DISAPPEAR_JITTER_S', disappear_delay_s)
            runner.register('DOT_DISAPPEAR_TIME', crossing_t + disappear_delay_s)

        print("Trial {0}/{1}: held={2:.2f}s (broke {3}x), decision period ended".format(
            trial + 1, VAR_N_TRIALS, required_hold, n_breaks), flush=True)

    except Exception as err:
        print("Trial {0}/{1} FAILED: {2}".format(trial + 1, VAR_N_TRIALS, err), flush=True)
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        raise
else:
    print("Done: {0} trials completed".format(VAR_N_TRIALS), flush=True)

dot.close()
rotary.close()
my_bpod.close()
