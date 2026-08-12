# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Plan step 3: same harness as hifi_singleside_test.py (step 2), now cycling difficulty
trial-by-trial through a fixed round-robin: AOS -> E15 -> E11 -> repeat ("two levels of easy"
taken as E15/E11, the two easiest graded levels below AOS). E15/E11 trials have both sides
active, so this is where dynamic per-trial reward-side mapping first becomes necessary --
rewarded_event_for_side() below resolves it from whichever side generate_trial_clicks() drew as
the higher-evidence side each trial (the same one-liner works unchanged from AOS, since
generate_trial_clicks()'s `side` argument is already defined as "the higher-evidence side"
regardless of difficulty level).

Everything else (hold-to-init incl. lick-gating, choice thresholds, HiFi direct-USB PLAY +
Bpod-relayed STOP_ALL, the cue-period abort detection, the go-cue LED + rotary position-reset
fired as WaitForChoice's own output_actions, Consumption, the AbortITI/ITI Bpod states, live
plots) is identical to hifi_singleside_test.py -- see that file's docstring for the full
rationale. Module connection/setup and the trial-loop plumbing (hold-to-init, background
WHEEL_POS thread, bounded-join wrapper, states_durations nan-check) now live in
`_projects/_shared/` (`bpod_trial_helpers.py`'s `TrialRunner`/`was_visited`, `rotary_setup.py`,
`hifi_setup.py`) rather than being duplicated in this file -- see those modules for the generic
rationale.

click_train.py/live_plots.py live in the sibling ../poisson_clicks_test/ folder (shared with
hifi_singleside_test.py) rather than being duplicated into this task folder -- paradigm-specific,
so they stay there rather than moving into the protocol-agnostic _shared/ folder. See
hifi_singleside_test.py's docstring for why both sys.path inserts resolve correctly under the
GUI's actual subprocess launch mechanism, regardless of location.

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
matplotlib.use('TkAgg')   # explicit, not auto-selected -- see hifi_singleside_test.py's docstring
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

VAR_N_TRIALS = 30
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
VAR_DIFFICULTY_CYCLE = ['AOS', 'E15', 'E11']   # fixed round-robin, not random -- see docstring

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
    """ Generalizes unchanged from step 2's AOS-only special case: generate_trial_clicks()'s
    `side` argument is already "the higher-evidence side," regardless of difficulty level, so
    this mapping needs no per-level branching even though E15/E11 now have both sides active. """
    return right_event if side == 'R' else left_event


# --- live plots ------------------------------------------------------------------------------------

bench_plots = LiveBenchPlots(
    waveform_duration=click_train.TOTAL_WAVEFORM_DURATION_S,
    click_start_offset=click_train.CLICK_START_OFFSET_S,
    stim_end=click_train.CLICK_START_OFFSET_S + click_train.VAR_STIM_DURATION_S,
    onset_pulse_duration=click_train.VAR_ONSET_PULSE_DURATION_S)

CUE_MONITOR_DURATION_S = click_train.TOTAL_WAVEFORM_DURATION_S + 0.1

# --- trial loop -----------------------------------------------------------------------------------

print("Starting {0} trials, cycling difficulty {1}".format(VAR_N_TRIALS, VAR_DIFFICULTY_CYCLE),
      flush=True)

for trial in range(VAR_N_TRIALS):
    try:
        difficulty = VAR_DIFFICULTY_CYCLE[trial % len(VAR_DIFFICULTY_CYCLE)]
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
        trial_clicks = click_train.generate_trial_clicks(difficulty, side)
        left_wave, right_wave = click_train.build_waveform(trial_clicks, hifi.sampling_rate)
        hifi.load(0, np.array([left_wave, right_wave]))
        hifi.push()

        runner.register('DIFFICULTY_LEVEL', difficulty)
        runner.register('TRIAL_SIDE', side)
        runner.register('REALIZED_DELTA', trial_clicks['realized_delta'])

        print("Trial {0}: held steady -- playing {1} stimulus (side={2}, n_L={3}, n_R={4})".format(
            trial + 1, difficulty, side, trial_clicks['n_left'], trial_clicks['n_right']),
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
            print("Trial {0}/{1}: difficulty={2}, held={3:.2f}s (broke {4}x), side={5}, "
                  "Abort ({6})".format(
                      trial + 1, VAR_N_TRIALS, difficulty, required_hold, n_breaks, side,
                      abort_reason),
                  flush=True)

            # cue_lick_times_abs already covers the whole trial (CuePeriod + EarlyLick/WheelAbort +
            # AbortITI, all one machine now), so no further query/concatenation is needed here.
            bench_plots.add_trial_licks(trial + 1, 'Abort', cue_lick_times_abs)

            plot_t0 = time.time()
            bench_plots.add_trial(difficulty, side, click_train.nominal_delta(difficulty),
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

        print("Trial {0}/{1}: difficulty={2}, held={3:.2f}s (broke {4}x), side={5}, {6}{7}".format(
            trial + 1, VAR_N_TRIALS, difficulty, required_hold, n_breaks, side, outcome,
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
        bench_plots.add_trial(difficulty, side, click_train.nominal_delta(difficulty),
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
