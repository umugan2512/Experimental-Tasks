# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Plan step 2: HiFi per-trial click-train playback + wheel-turn choice + single-valve reward,
stimulus content restricted to AOS (one side has clicks, the other is silent every trial) so the
rewarded direction is unambiguous.

Hold-to-init, module connection/setup, and the background WHEEL_POS-publishing/bounded-join
plumbing now live in `_projects/_shared/` (`bpod_trial_helpers.py`'s `TrialRunner`/`was_visited`,
`rotary_setup.py`, `hifi_setup.py`) rather than being copy-pasted in this file -- that duplication
used to exist across this file, hifi_alternating_easy_test.py, and wheel_turn_reward.py, with the
latter's copy already quietly drifted out of sync (missing the try/except Kill-race fix). See
those modules' own docstrings for the generic rationale; this docstring only covers what's
specific to this protocol.

Choice thresholds (steady +/-5deg, choice +/-30deg) are reused from wheel_turn_reward.py's own
values -- that gap is deliberate: the choice threshold only needs to be safely far from the
quiescence tolerance so a hold-breaking jitter can never be misread as a deliberate turn, not a
precision-calibrated number.

Hold-to-init now also gates on licking, not just wheel steadiness (`TrialRunner.
wait_for_held_steady(..., require_no_lick=True)`, the default) -- a trial can't start while Port1
has been recently active, for the same reason the cue period needs licking to matter: an animal
still consuming/licking shouldn't have a new trial sneak up on it.

Cue period (step 4): the trial can now abort for two independent reasons during the cue/delay
window (the entire baked stimulus: onset pulse + gap + 3s stimulus + 1s silent delay) --
1. the wheel deviates past VAR_CUE_ABORT_THRESHOLD_DEG, or
2. an early lick on Port1 (the reward-port lick sensor, same input lick_reward.py uses).
Both must prevent the reward valve from ever firing, which is why the cue period is still its own,
separate Bpod state machine from the choice phase (Reward is never in the same machine as
anything cue-related). It needs to be a *running* Bpod state machine (not a pure-Python check),
because Bpod has no way to poll Port1's *current* state outside of an active state machine
(checked bpod_base.py/bpod_com_protocol.py -- manual_override/trigger_input only *set* outputs).
Port1In is only observable while something is actively listening for it via
state_change_conditions -- unlike the rotary encoder, which has its own USB connection
independently pollable via current_position() at any time regardless of whether Bpod is running
anything.

**Both abort reasons are fully in-band.** CuePeriod's state_change_conditions listen for Port1In
(-> EarlyLick) *and* two rotary threshold-crossing events at +/-VAR_CUE_ABORT_THRESHOLD_DEG
(-> WheelAbort), so a wheel-deviation abort gets exactly the same instant reaction EarlyLick
already had -- no background-thread polling involved in the abort decision at all. A background
thread polling current_position() can flag a deviation, but it can't stop or signal into an
already-running run_state_machine() call (that would mean two threads writing the same physical
Bpod USB connection concurrently -- genuinely unsynchronized access, confirmed by reading
bpod_com_protocol.py: manual_override/trigger_output/trigger_softcode all use that same
connection, and the only thread-safe channel into a running trial, self.stdin, is wired to the
GUI's own subprocess management for Stop/Kill, not something a same-process background thread can
write into). Going native instead means enable_evt_transmission() has to happen *before* cue_sma
is sent (not after cue-phase returns) and stays on through both cue and choice phases -- accepting
some reintroduced risk of the board buffering/flushing stray crossings at a state-machine
boundary, which keeping transmission off during the cue period was previously guarding against
(the "stale rest position" half of that risk is still covered by WaitForChoice's own
SETZEROPOS+ENABLE_ALLTHRESHOLDS reset regardless).

It also reintroduces some reliance on the rotary's own firmware threshold comparator, previously
avoided (see CLAUDE.md) because its crossing *events* were found to drift from current_position()'s
own (reliable) readings -- that finding was about the tighter +/-30deg *scoring* thresholds though,
a stricter precision need than this much looser abort gate, so the same drift likely matters less
here. Because Reward still isn't part of this machine, both abort reasons still fully prevent the
valve from firing -- cue-phase and choice-phase remain two separate Bpod-level trials for any
non-aborted user-trial, a necessary consequence of needing Bpod to actually be running to see
Port1In at all, not a regression.

Consumption (step 4): a new state after Reward (state_timer=VAR_CONSUMPTION_WINDOW_S, no
output_actions) so Port1In keeps getting logged for a few seconds after the valve closes, long
enough to see licking, without extending the valve-open duration itself (Reward's own state_timer
stays VAR_REWARD_DURATION). Feeds the live lick plots (live_plots.py) -- see there for the
"first lick" highlighting and the session-wide/full-session lick panels.

Go-cue + choice-window reset are both fired as native Bpod output_actions on WaitForChoice's own
entry, not from a Python-side timer -- an earlier version tried announcing "choice window open"
and re-enabling thresholds from a background thread once elapsed time crossed the Stimulus state's
known duration, but two real hardware runs showed that has ~0.1s of slop and, worse, if the wheel
drifted past a threshold during Stimulus (ignored, since Stimulus didn't listen for wheel events)
and was still resting there when thresholds got re-enabled at the WaitForChoice boundary, the
board would sometimes immediately re-signal that stale crossing -- scoring a trial from wherever
cue-period drift happened to leave the wheel, not a genuine post-cue choice. Firing the LED
(PWM1, Port 1's built-in LED) and the rotary module's own SETZEROPOS+ENABLE_ALLTHRESHOLDS reset
message together as WaitForChoice's own output_actions fixes both problems at once: it fires in
perfect sync with the actual state transition, and re-zeroing means left/right are always measured
fresh from wherever the wheel is resting the instant the window opens. The LED turns back off in
Reward/NoReward/NoResponse's own output_actions.

HiFi setup (one bake-and-trigger-once PLAY per trial via hifi.play(0), directly over the module's
own USB connection) follows side_coding_test.py's pattern, but the waveform is freshly generated
every trial (click_train.py) instead of built once outside the loop. PLAY stays on that direct-USB
channel deliberately, even though CuePeriod is a real Bpod state now (which would make a
Bpod-relayed PLAY trigger possible again): EarlyLick/WheelAbort/CueComplete all fire a Bpod-relayed
STOP_ALL, and firing both PLAY and STOP_ALL as output_actions within the same Bpod-trial would
trigger the same custom module twice in one trial -- the exact, already confirmed
run_state_machine()-hanging hazard. Keeping PLAY on direct-USB means STOP_ALL is the only
Bpod-relayed HiFi output per trial, so it never collides with itself. STOP_ALL stops the cue in
exact lockstep with Bpod's own abort detection (immediate for EarlyLick/WheelAbort; a harmless
no-op safety net for CueComplete, since the baked audio has normally already finished playing
naturally by then).

The abort ITI (VAR_ABORT_ITI_S) is a real Bpod state (AbortITI), not a Python time.sleep() --
folded directly into cue_sma itself: EarlyLick and WheelAbort both transition into it, possible
only because both abort reasons are in-band. Because Bpod keeps logging any Port1In it sees for as
long as some state in the machine is running, whether or not anything transitions on it, a single
get_all_timestamps_by_event() call after cue_sma returns picks up every lick across the whole
trial -- cue, abort, and its ITI -- in one shot. Same idea for the normal (non-aborted) path's ITI
-- also a real state now (Consumption/NoReward/NoResponse all route to it instead of exiting
directly), so its own Port1In events fall out of the existing single query for free.

Live plots (live_plots.py): one window (LiveBenchPlots) -- a click raster (with a blended-color
marker for the genuinely-concurrent bilateral onset pulse), a psychometric/lick-raster/outcome
row, and a full-session lick timeline. This is separate from, and doesn't touch, the existing
pybpod-gui-plugin-rotaryencoder "Wheel Position" GUI plugin window.

click_train.py/live_plots.py live in the sibling ../poisson_clicks_test/ folder (shared with
hifi_alternating_easy_test.py) rather than being duplicated into this task folder -- paradigm-
specific, so they stay there rather than moving into the protocol-agnostic _shared/ folder above.
Both sys.path inserts below rely on the same mechanism, verified directly against the actual GUI
subprocess launch mechanism (board_com.py's run_task(): `subprocess.Popen(['python',
os.path.abspath(task.filepath)], cwd=self._running_session.path, ...)`) -- __file__ resolves to
this script's own real location regardless of that subprocess's cwd (the session folder, not this
task folder), so both inserts work identically whether launched via the GUI's Run button or by
hand, and are location-agnostic (confirmed against run_task()'s actual code, not assumed) -- there
is nothing tying this trick to a sibling folder specifically.

Run this like any other PyBpod task, via the GUI's Run button -- requires the Bpod board, rotary
encoder module, and HiFi module all physically connected via USB first.
"""
import os
import random
import sys
import time
import traceback

import numpy as np
import matplotlib
matplotlib.use('TkAgg')   # explicit, not auto-selected -- see module docstring re: the live-plot
                          # freeze investigation (auto-selection in a GUI-launched subprocess is a
                          # plausible reason plt.pause()-driven updates could misbehave)
import matplotlib.pyplot as plt

_TASK_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_TASK_DIR, '..', 'poisson_clicks_test'))
sys.path.insert(0, os.path.join(_TASK_DIR, '..', '..', '..', '_shared'))
import click_train
from live_plots import LiveBenchPlots
from bpod_trial_helpers import TrialRunner, was_visited
import rotary_setup
import hifi_setup

from pybpodapi.protocol import Bpod, StateMachine

VAR_N_TRIALS = 20
VAR_HOLD_MIN_S = 0.1
VAR_HOLD_MAX_S = 0.5
VAR_STEADY_THRESHOLD_DEG = 5       # same as wheel_turn_reward.py -- not being retuned here
VAR_LEFT_THRESHOLD_DEG = -30
VAR_RIGHT_THRESHOLD_DEG = 30
VAR_CUE_ABORT_THRESHOLD_DEG = 2.5 * VAR_STEADY_THRESHOLD_DEG   # 12.5deg -- looser than quiescence,
                                                                 # still much tighter than a choice turn
VAR_RESPONSE_TIMEOUT = 5
VAR_REWARD_DURATION = 0.1
VAR_CONSUMPTION_WINDOW_S = 3.0    # default -- how long after Reward to keep watching Port1 for
                                  # licks; not a calibrated value, just enough to see a few licks
VAR_ITI = 2
VAR_ABORT_ITI_S = 5.0             # longer than the normal ITI -- a mild timeout after an abort
                                  # (particularly EarlyLick), distinct from a completed trial's ITI
VAR_STILL_POLL_HZ = 50
VAR_POLL_HZ = 10
VAR_ROTARY_USB_PORT = None
VAR_DIFFICULTY = 'AOS'            # fixed for this step -- step 3 cycles through several

# 6 thresholds -- the documented max for RotaryEncoderModule.set_thresholds() -- index order
# determines event numbering (RotaryEncoder1_1..._6); rotary_setup.set_and_enable_thresholds()
# returns the corresponding event names below instead of this file re-deriving them by hand.
ALL_THRESHOLDS_DEG = [-VAR_STEADY_THRESHOLD_DEG, VAR_STEADY_THRESHOLD_DEG,
                       VAR_LEFT_THRESHOLD_DEG, VAR_RIGHT_THRESHOLD_DEG,
                       -VAR_CUE_ABORT_THRESHOLD_DEG, VAR_CUE_ABORT_THRESHOLD_DEG]

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

my_bpod.register_value('LEFT_THRESHOLD_DEG', VAR_LEFT_THRESHOLD_DEG)
my_bpod.register_value('RIGHT_THRESHOLD_DEG', VAR_RIGHT_THRESHOLD_DEG)

log_python_t0 = time.time()
runner = TrialRunner(my_bpod, rotary, log_python_t0, still_poll_hz=VAR_STILL_POLL_HZ,
                      poll_hz=VAR_POLL_HZ)


def rewarded_event_for_side(side):
    return right_event if side == 'R' else left_event


# --- live plots ------------------------------------------------------------------------------------

bench_plots = LiveBenchPlots(
    waveform_duration=click_train.TOTAL_WAVEFORM_DURATION_S,
    click_start_offset=click_train.CLICK_START_OFFSET_S,
    stim_end=click_train.CLICK_START_OFFSET_S + click_train.VAR_STIM_DURATION_S,
    onset_pulse_duration=click_train.VAR_ONSET_PULSE_DURATION_S)

CUE_MONITOR_DURATION_S = click_train.TOTAL_WAVEFORM_DURATION_S + 0.1

# --- trial loop -----------------------------------------------------------------------------------

print("Starting {0} {1}-only trials".format(VAR_N_TRIALS, VAR_DIFFICULTY), flush=True)

for trial in range(VAR_N_TRIALS):
    try:
        required_hold = random.uniform(VAR_HOLD_MIN_S, VAR_HOLD_MAX_S)

        rotary.disable_evt_transmission()
        n_breaks = runner.wait_for_held_steady(required_hold, VAR_STEADY_THRESHOLD_DEG)
        if n_breaks is None:
            print("Bpod stopped running trials (Stop/Kill) -- ending session early after trial "
                  "{0}/{1}.".format(trial + 1, VAR_N_TRIALS), flush=True)
            break
        trial_start_t = time.time() - log_python_t0

        with runner.rotary_lock:
            rotary.set_zero_position()
            rotary_setup.set_and_enable_thresholds(rotary, ALL_THRESHOLDS_DEG)

        # Enabled here, before cue_sma is even built, and left on through both cue and choice
        # phases -- CuePeriod's own state_change_conditions now listen for the wheel-abort
        # threshold events directly (RotaryEncoder1_5/_6), so transmission has to be on for Bpod
        # to see them at all. Accepted trade-off: this reintroduces some risk of the board
        # buffering/flushing a backlog of stray crossings at a state-machine boundary, which
        # keeping transmission off during the cue period was previously guarding against -- watch
        # for a sub-0.1s "reaction time" with WHEEL_POS unchanged as the tell if that recurs.
        rotary.enable_evt_transmission()

        runner.register('TRIAL_START', trial_start_t)

        side = click_train.draw_side()
        trial_clicks = click_train.generate_trial_clicks(VAR_DIFFICULTY, side)
        left_wave, right_wave = click_train.build_waveform(trial_clicks, hifi.sampling_rate)
        hifi.load(0, np.array([left_wave, right_wave]))
        hifi.push()

        runner.register('DIFFICULTY_LEVEL', VAR_DIFFICULTY)
        runner.register('TRIAL_SIDE', side)
        runner.register('REALIZED_DELTA', trial_clicks['realized_delta'])

        print("Trial {0}: held steady -- playing {1} stimulus (side={2}, n_L={3}, n_R={4})".format(
            trial + 1, VAR_DIFFICULTY, side, trial_clicks['n_left'], trial_clicks['n_right']),
            flush=True)

        hifi.play(0)

        cue_send_t = time.time() - log_python_t0

        cue_sma = StateMachine(my_bpod)

        cue_sma.add_state(
            state_name='CuePeriod',
            state_timer=CUE_MONITOR_DURATION_S,
            state_change_conditions={
                'Port1In': 'EarlyLick',
                wheel_abort_event_neg: 'WheelAbort',
                wheel_abort_event_pos: 'WheelAbort',
                Bpod.Events.Tup: 'CueComplete',
            },
            output_actions=[])

        cue_sma.add_state(
            state_name='EarlyLick',
            state_timer=0,
            state_change_conditions={Bpod.Events.Tup: 'AbortITI'},
            output_actions=[(hifi_channel, hifi_stop_msg_id)])

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
                  "{0}/{1}.".format(trial + 1, VAR_N_TRIALS), flush=True)
            break

        cue_visited = my_bpod.session.current_trial.states_durations
        early_lick = was_visited(cue_visited, 'EarlyLick')
        wheel_moved = was_visited(cue_visited, 'WheelAbort')
        aborted = early_lick or wheel_moved

        cue_events = my_bpod.session.current_trial.get_all_timestamps_by_event()
        cue_lick_times_abs = [cue_send_t + t for t in cue_events.get('Port1In', [])]

        if aborted:
            abort_state = 'EarlyLick' if early_lick else 'WheelAbort'
            abort_reason = 'EarlyLick' if early_lick else 'WheelMoved'
            abort_time_abs = cue_send_t + cue_visited[abort_state][-1][0]
            runner.register('ABORT', abort_time_abs)
            print("Trial {0}/{1}: held={2:.2f}s (broke {3}x), side={4}, Abort ({5})".format(
                trial + 1, VAR_N_TRIALS, required_hold, n_breaks, side, abort_reason), flush=True)

            # cue_lick_times_abs already covers the whole trial (CuePeriod + EarlyLick/WheelAbort +
            # AbortITI, all one machine now), so no further query/concatenation is needed here.
            bench_plots.add_trial_licks(trial + 1, 'Abort', cue_lick_times_abs)

            plot_t0 = time.time()
            bench_plots.add_trial(VAR_DIFFICULTY, side, click_train.nominal_delta(VAR_DIFFICULTY),
                                   trial_clicks, outcome='Abort')
            plot_render_s = time.time() - plot_t0
            if plot_render_s > 0.5:
                print("NOTE: live-plot redraw took {0:.2f}s this trial".format(plot_render_s),
                      flush=True)

            continue

        rewarded_event = rewarded_event_for_side(side)
        unrewarded_event = left_event if rewarded_event == right_event else right_event

        send_t = time.time() - log_python_t0

        sma = StateMachine(my_bpod)

        sma.add_state(
            state_name='WaitForChoice',
            state_timer=VAR_RESPONSE_TIMEOUT,
            state_change_conditions={
                rewarded_event: 'Reward',
                unrewarded_event: 'NoReward',
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
            state_change_conditions={Bpod.Events.Tup: 'ITI'},
            output_actions=[])

        sma.add_state(
            state_name='NoReward',
            state_timer=0,
            state_change_conditions={Bpod.Events.Tup: 'ITI'},
            output_actions=[(VAR_GO_CUE_LED_CHANNEL, 0)])

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

        if not runner.run_trial_state_machine(sma):
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

        print("Trial {0}/{1}: held={2:.2f}s (broke {3}x), side={4}, {5}{6}".format(
            trial + 1, VAR_N_TRIALS, required_hold, n_breaks, side, outcome,
            ' -> reward' if rewarded else ''), flush=True)

        choice_events = my_bpod.session.current_trial.get_all_timestamps_by_event()
        choice_lick_times_abs = [send_t + t for t in choice_events.get('Port1In', [])]
        lick_times_abs = sorted(cue_lick_times_abs + choice_lick_times_abs)

        reward_time_abs = None
        if rewarded:
            # Reward's own start in the choice-phase machine's own clock (== WaitForChoice's
            # duration, i.e. the reaction time), converted to absolute session-elapsed time.
            choice_time = visited['Reward'][-1][0]
            reward_time_abs = send_t + choice_time

        bench_plots.add_trial_licks(trial + 1, outcome, lick_times_abs,
                                     reward_time_abs=reward_time_abs)

        plot_t0 = time.time()
        bench_plots.add_trial(VAR_DIFFICULTY, side, click_train.nominal_delta(VAR_DIFFICULTY),
                               trial_clicks, outcome=outcome)
        plot_render_s = time.time() - plot_t0
        if plot_render_s > 0.5:
            print("NOTE: live-plot redraw took {0:.2f}s this trial".format(plot_render_s), flush=True)

    except Exception as err:
        print("Trial {0}/{1} FAILED: {2}".format(trial + 1, VAR_N_TRIALS, err), flush=True)
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        raise
else:
    print("Done: {0} trials completed".format(VAR_N_TRIALS), flush=True)

hifi.close()
rotary.close()
my_bpod.close()

print("Close the plot windows to exit.", flush=True)
plt.ioff()
plt.show()
