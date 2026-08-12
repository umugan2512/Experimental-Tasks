# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Bench test: a wheel turn during the decision period drags a Gabor patch across a second monitor,
coupled 1:1 to wheel position -- a standalone test of the visual-motor coupling piece itself,
before it's later embedded into the Poisson-clicks protocol's own decision period (replacing/
augmenting the go-cue LED with this visual cue). First pass (pure coupling/rendering, no reward)
verified clean on hardware; this pass adds reward scoring -- a left turn is rewarded
(`VAR_REWARDED_DIRECTION`-style hardcoded convention, matching wheel_turn_reward.py; not yet a
toggle), a right turn is not, followed by a real Bpod `ITI` state (not a Python `time.sleep()`,
matching the HiFi scripts' own convention) -- hold-to-init -> decision period with Gabor-position
coupling -> Reward/NoReward/NoResponse -> ITI. No Consumption state (no lick sensor relevant to
this bench rig) -- `Reward` routes straight to `ITI`.

Coupling convention (see gabor_display.py's docstring for the research behind this): wheel
position maps to the *stimulus's on-screen position* (a signed pixel offset from center), matching
the IBL/Burgess-Carandini-Harris wheel-based visual discrimination task's own data schema
(confirmed directly from ibllib's own extractors, which are installed in this env as a
pybpod-gui-plugin-alyx dependency) -- turning the wheel drags the patch like pulling it on a
string. VAR_DEG_TO_PX_GAIN is a deliberately uncalibrated placeholder (real degrees-of-visual-angle
calibration depends on this rig's screen size/viewing distance, not established yet).

Hold-to-init, module connection/setup, and the background WHEEL_POS-publishing/bounded-join
plumbing reuse `_projects/_shared/` (`bpod_trial_helpers.py`'s `TrialRunner`/`was_visited`,
`rotary_setup.py`) exactly like the HiFi scripts. `VAR_REQUIRE_NO_LICK` defaults to False here --
no lick sensor is assumed relevant to this particular bench rig; flip it on if this rig does have
Port1 wired to something meaningful.

**Thread roles are inverted from every other script in this project.** Everywhere else,
`TrialRunner.run_trial_state_machine()` runs the Bpod call in the *foreground* (blocking) while a
background thread publishes WHEEL_POS. Here, the decision-period Bpod call runs in a *background*
thread instead, while the *foreground* (main) thread runs the Gabor render loop -- because Qt's
event loop/repaints have to happen on the main thread (GaborDisplay.pump() calls
QApplication.processEvents(), which only works correctly from the thread that owns the
QApplication). Nothing in TrialRunner needed to change for this; this script just wraps its own
call to `runner.run_trial_state_machine(decision_sma)` in a `threading.Thread`, the same
bounded-join shape TrialRunner already uses internally, just with the roles swapped. Both the
foreground render loop and TrialRunner's own internal WHEEL_POS thread call
`rotary.current_position()` concurrently during the decision period, so both go through the same
`runner.rotary_lock` to serialize access to the rotary's own USB connection.

**Kill needs special handling because of this inversion.** `run_state_machine()`'s kill handling
(`bpod_base.py`) calls `exit(0)` *from inside* the function once it reads a kill command off
stdin -- every other script in this project relies on that call naturally being on the main thread,
where `SystemExit` propagating out uncaught actually terminates the process. Raised on a
*background* thread instead (as it now is here), `SystemExit` only kills that thread silently --
the main thread's render loop would otherwise spin forever waiting on a `decision_done` event that
never gets set. `_run_decision()` below catches `SystemExit` specifically, records that a Kill
happened, and always sets `decision_done` in a `finally` so the render loop can't hang; the main
thread then re-raises `SystemExit` itself once it notices, so the process actually terminates the
way every other script's Kill handling already does.

Run this like any other PyBpod task, via the GUI's Run button -- requires the Bpod board and
rotary encoder module physically connected via USB, and a second monitor connected for the Gabor
display (falls back to screen 0 with a printed warning if only one screen is detected, so this
still runs -- just without a second-monitor split -- when tested without one attached).
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
from gabor_display import GaborDisplay

from pybpodapi.protocol import Bpod, StateMachine

VAR_N_TRIALS = 20
VAR_HOLD_MIN_S = 0.1
VAR_HOLD_MAX_S = 0.5
VAR_STEADY_THRESHOLD_DEG = 5       # same as wheel_turn_reward.py -- not being retuned here
VAR_LEFT_THRESHOLD_DEG = -30
VAR_RIGHT_THRESHOLD_DEG = 30
VAR_RESPONSE_TIMEOUT = 5
VAR_REWARD_DURATION = 0.1
VAR_ITI = 2
VAR_STILL_POLL_HZ = 50
VAR_POLL_HZ = 10
VAR_ROTARY_USB_PORT = None
VAR_REQUIRE_NO_LICK = False        # no lick sensor assumed relevant to this bench test -- flip on
                                   # if this rig has Port1 wired to something meaningful

VAR_GO_CUE_LED_CHANNEL = 'PWM1'   # Port 1's built-in LED, same convention as the HiFi scripts

VAR_GABOR_SCREEN_INDEX = 1        # second monitor; falls back to 0 with a warning if not found
VAR_GABOR_SIZE_PX = 400
VAR_GABOR_SPATIAL_FREQ_CPP = 6
VAR_GABOR_SIGMA_PX = 80
VAR_GABOR_CONTRAST = 1.0
VAR_GABOR_BACKGROUND_GRAY = 128
VAR_DEG_TO_PX_GAIN = 4.0          # uncalibrated placeholder -- see module docstring
VAR_RENDER_HZ = 60

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

gabor = GaborDisplay(screen_index=VAR_GABOR_SCREEN_INDEX, size_px=VAR_GABOR_SIZE_PX,
                      spatial_freq_cpp=VAR_GABOR_SPATIAL_FREQ_CPP, sigma_px=VAR_GABOR_SIGMA_PX,
                      contrast=VAR_GABOR_CONTRAST, background_gray=VAR_GABOR_BACKGROUND_GRAY,
                      deg_to_px_gain=VAR_DEG_TO_PX_GAIN)
gabor.show()
gabor.clear()

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
        gabor.clear()
        gabor.pump()

        print("Trial {0}: held steady -- decision period open".format(trial + 1), flush=True)

        send_t = time.time() - log_python_t0

        decision_sma = StateMachine(my_bpod)

        decision_sma.add_state(
            state_name='WheelGaborPeriod',
            state_timer=VAR_RESPONSE_TIMEOUT,
            state_change_conditions={
                left_event: 'Reward',
                right_event: 'NoReward',
                Bpod.Events.Tup: 'NoResponse',
            },
            output_actions=[(rotary_channel, reset_positions_trigger_id),
                             (VAR_GO_CUE_LED_CHANNEL, 255)])

        decision_sma.add_state(
            state_name='Reward',
            state_timer=VAR_REWARD_DURATION,
            state_change_conditions={Bpod.Events.Tup: 'ITI'},
            output_actions=[(Bpod.OutputChannels.Valve, 1), (VAR_GO_CUE_LED_CHANNEL, 0)])

        decision_sma.add_state(
            state_name='NoReward',
            state_timer=0,
            state_change_conditions={Bpod.Events.Tup: 'ITI'},
            output_actions=[(VAR_GO_CUE_LED_CHANNEL, 0)])

        decision_sma.add_state(
            state_name='NoResponse',
            state_timer=0,
            state_change_conditions={Bpod.Events.Tup: 'ITI'},
            output_actions=[(VAR_GO_CUE_LED_CHANNEL, 0)])

        decision_sma.add_state(
            state_name='ITI',
            state_timer=VAR_ITI,
            state_change_conditions={Bpod.Events.Tup: 'exit'},
            output_actions=[])

        # Inverted from the usual pattern -- see module docstring. The Bpod call runs in a
        # background thread; this (main) thread renders the Gabor patch instead of blocking here.
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

        while not decision_done.is_set():
            with runner.rotary_lock:
                pos = rotary.current_position()
            gabor.set_position_deg(pos)
            gabor.pump()
            time.sleep(render_interval)

        decision_thread.join(timeout=3.0)
        if decision_thread.is_alive():
            print("WARNING: decision-period Bpod thread did not stop within 3s -- continuing "
                  "anyway.", flush=True)

        gabor.clear()
        gabor.pump()

        if decision_result.get('killed'):
            print("Bpod Kill received -- ending session.", flush=True)
            gabor.close()
            rotary.close()
            my_bpod.close()
            sys.exit(0)

        if not decision_result.get('ran', False):
            print("Bpod stopped running trials (Stop/Kill) -- ending session early after trial "
                  "{0}/{1}.".format(trial + 1, VAR_N_TRIALS), flush=True)
            break

        visited = my_bpod.session.current_trial.states_durations
        outcome = next((name for name in ('Reward', 'NoReward', 'NoResponse')
                         if was_visited(visited, name)), 'Unknown')
        rewarded = outcome == 'Reward'

        if outcome in ('Reward', 'NoReward'):
            runner.register('CHOICE_MADE', send_t + visited[outcome][-1][0])
            runner.register('CORRECT', rewarded)
        elif outcome == 'NoResponse':
            runner.register('TIMED_OUT', send_t + visited['NoResponse'][-1][0])

        print("Trial {0}/{1}: held={2:.2f}s (broke {3}x), {4}{5}".format(
            trial + 1, VAR_N_TRIALS, required_hold, n_breaks, outcome,
            ' -> reward' if rewarded else ''), flush=True)

    except Exception as err:
        print("Trial {0}/{1} FAILED: {2}".format(trial + 1, VAR_N_TRIALS, err), flush=True)
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        raise
else:
    print("Done: {0} trials completed".format(VAR_N_TRIALS), flush=True)

gabor.close()
rotary.close()
my_bpod.close()
