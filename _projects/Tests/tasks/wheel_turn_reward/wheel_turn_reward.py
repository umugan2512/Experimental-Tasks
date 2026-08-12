# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Rotary encoder self-initiated trial: hold-steady gating (in Python) + left/right choice (in Bpod).

Every previous version of this task tried to detect "held steady" using the rotary board's own
firmware threshold comparator (armed/disarmed via Bpod SoftCode, or reconfigured via
set_thresholds()) -- across several hardware sessions, that comparator repeatedly drifted away from
what current_position() reports, causing huge break counts and Hold completing while the wheel
visibly sat outside the configured deadzone. current_position() itself has been reliable throughout
every one of those sessions -- only the board's own threshold-crossing events ever looked wrong. So
hold-detection here is now plain Python position polling (wait_for_held_steady()), and Bpod is only
handed a state machine once that's already confirmed -- Bpod covers just the part it's actually been
trustworthy for: the choice phase (WaitForChoice -> Reward/NoReward/NoResponse), where thresholds are
a generous +/-30deg and a few degrees of drift has never mattered.

Since wait_for_held_steady() doesn't care about absolute position -- only whether the last
required_hold-seconds of samples all agree -- "held steady" can happen wherever the wheel settles
after the previous trial's choice turn, in whatever direction, after however long the subject kept
moving it first. Zeroing happens exactly once per trial, right after that settle is confirmed, right
before the choice thresholds are configured -- not before the hold-wait begins.

Trade-off, accepted deliberately: Bpod's states_durations no longer has a Hold segment (there's no
Hold state left in the state machine), so hold timing isn't native Trial Timeline data anymore --
it's still fully captured, just as a VAL row (TRIAL_START, same mechanism as always) instead of a
state segment. WaitForChoice/Reward/NoReward/NoResponse are still real Bpod states, still show up in
Trial Timeline, and Kill/Stop still work natively during that phase.

Wheel position is published live via a background thread (trial_poll_loop) during the Bpod-run
choice phase, and directly from wait_for_held_steady()'s own polling loop before that -- both write
through the same register_lock-protected register_value() calls, so the live "Wheel Position" plot
sees one continuous trace across the whole trial regardless of which phase is running.
"""
import math
import random
import sys
import threading
import time
import traceback
from collections import deque

from pybpodapi.protocol import Bpod, StateMachine
from pybpod_rotaryencoder_module.module_api import RotaryEncoderModule

VAR_N_TRIALS = 30
VAR_HOLD_MIN_S = 1                # randomized hold-steady duration range
VAR_HOLD_MAX_S = 1.5
VAR_STEADY_THRESHOLD_DEG = 5      # symmetric deadzone; a hold attempt restarts if this is exceeded
VAR_RESPONSE_TIMEOUT = 5          # seconds to wait for a choice before counting as no-response
VAR_POLL_HZ = 10                  # wheel-position live-plot publish rate during the choice phase
VAR_ITI = 2                       # seconds between trials
VAR_LEFT_THRESHOLD_DEG = -30      # confirmed via the Tools > Rotary encoder live-plot panel
VAR_RIGHT_THRESHOLD_DEG = 30
VAR_REWARDED_DIRECTION = 'LEFT'   # 'LEFT' or 'RIGHT' -- flip to test each side independently
VAR_REWARD_DURATION = 0.1         # seconds Valve1 stays open
VAR_ROTARY_USB_PORT = None        # optional override; None = auto-discover
VAR_STILL_POLL_HZ = 50            # wait_for_held_steady()'s own position-sampling rate

poll_interval = 1.0 / VAR_POLL_HZ

# [steady-neg, steady-pos, left-choice, right-choice] -- the steady slots (0, 1) are no longer
# listened to (hold-steady detection is pure Python now, see wait_for_held_steady()), but the array
# stays 4 elements to avoid disturbing the index-order -> event-name mapping the choice events
# (RotaryEncoder1_3/_4) rely on.
STEADY_AND_CHOICE_THRESHOLDS_DEG = [-VAR_STEADY_THRESHOLD_DEG, VAR_STEADY_THRESHOLD_DEG,
                                     VAR_LEFT_THRESHOLD_DEG, VAR_RIGHT_THRESHOLD_DEG]


# --- connect to Bpod, resolve the module by name -------------------------------------------------

my_bpod = Bpod()
print("Connected to Bpod on {0}".format(my_bpod.serial_port), flush=True)

rotary_bpod_module = next(m for m in my_bpod.modules if m.name and m.name.startswith('RotaryEncoder'))
print("Resolved module '{0}' on Bpod port {1}, event_names={2}".format(
    rotary_bpod_module.name, rotary_bpod_module.serial_port, rotary_bpod_module.event_names), flush=True)

# --- connect the rotary encoder's own USB link, configure thresholds ----------------------------

if VAR_ROTARY_USB_PORT:
    rotary = RotaryEncoderModule(VAR_ROTARY_USB_PORT)
else:
    rotary = RotaryEncoderModule.discover(exclude_ports=[my_bpod.serial_port])
print("Connected to rotary encoder on {0}".format(rotary.arcom.serial_object.port), flush=True)

# Index order determines event index (RotaryEncoder1_1.._4), confirmed pattern from earlier
# research. Verify against the printed event_names above before trusting this mapping.
rotary.set_thresholds(STEADY_AND_CHOICE_THRESHOLDS_DEG)
rotary.enable_all_thresholds()
rotary.enable_evt_transmission()

left_event = 'RotaryEncoder1_3'
right_event = 'RotaryEncoder1_4'
rewarded_event = left_event if VAR_REWARDED_DIRECTION == 'LEFT' else right_event
unrewarded_event = right_event if rewarded_event == left_event else left_event

print("Thresholds: steady=+/-{0}deg, left={1}deg ({2}), right={3}deg ({4}); rewarded={5}".format(
    VAR_STEADY_THRESHOLD_DEG, VAR_LEFT_THRESHOLD_DEG, left_event,
    VAR_RIGHT_THRESHOLD_DEG, right_event, VAR_REWARDED_DIRECTION), flush=True)

# read live/after the fact via right-click "Wheel Position" on this session in the project tree
my_bpod.register_value('LEFT_THRESHOLD_DEG', VAR_LEFT_THRESHOLD_DEG)
my_bpod.register_value('RIGHT_THRESHOLD_DEG', VAR_RIGHT_THRESHOLD_DEG)

log_python_t0 = time.time()

# --- locks: rotary_lock serializes access to the rotary's own serial connection across threads ---
# (wait_for_held_steady()'s polling loop, trial_poll_loop()'s polling loop, and the per-trial
# zero/threshold baseline block all touch it); register_lock serializes register_value() calls.

rotary_lock = threading.Lock()
register_lock = threading.Lock()
poll_thread_stop = threading.Event()


def trial_poll_loop():
    """ Publishes WHEEL_POS continuously while the choice-phase state machine is running. """
    while not poll_thread_stop.is_set():
        with rotary_lock:
            p = rotary.current_position()
        now = time.time()
        with register_lock:
            my_bpod.register_value('WHEEL_POS', '{0:.3f},{1:.2f}'.format(now - log_python_t0, p))
        time.sleep(poll_interval)


def run_trial_state_machine(sma):
    """
    Run sma with the live wheel-position thread active for exactly its duration.

    Returns run_state_machine()'s own result: False means Bpod has internally stopped running
    trials at all (its _skip_all_trials flag, set once a Stop/Kill command has been processed) --
    the caller should stop looping rather than keep sending state machines Bpod will just no-op on.
    """
    poll_thread_stop.clear()
    t = threading.Thread(target=trial_poll_loop, daemon=True)
    t.start()
    try:
        my_bpod.send_state_machine(sma)
        ran = my_bpod.run_state_machine(sma)
    finally:
        poll_thread_stop.set()
        # Bounded, not join() -- run_state_machine() calls exit(0) straight from inside its own
        # kill handling, which unwinds through this finally; an unbounded join here could block
        # Kill indefinitely if this thread is ever mid-stall on a contended serial read. It's a
        # daemon thread, so an orphaned one doesn't block process exit either way.
        t.join(timeout=3.0)
        if t.is_alive():
            print("WARNING: wheel-position poll thread did not stop within 3s -- continuing "
                  "anyway (possible serial contention with the rotary encoder board).", flush=True)
    return ran


def wait_for_held_steady(required_hold):
    """
    Blocks until the wheel has stayed within VAR_STEADY_THRESHOLD_DEG for a continuous
    required_hold-length window -- this is the entire hold-steady check now, done in plain Python
    instead of a Bpod Hold state reacting to the rotary board's own threshold-crossing events (see
    module docstring for why). Doesn't care about absolute position -- only whether the last
    required_hold seconds of samples all agree -- so "held steady" can happen wherever the wheel
    ends up resting.

    The window is trimmed by actual elapsed wall-clock time, not a sample count sized against
    VAR_STILL_POLL_HZ -- the poll loop's real rate runs below that nominal figure (serial
    round-trip + register_value() overhead each iteration; confirmed ~32Hz actual vs 50Hz nominal
    on real hardware), so a count-based window would silently enforce a substantially longer hold
    than required_hold actually calls for.

    Returns the number of large (> VAR_STEADY_THRESHOLD_DEG) consecutive-sample jumps seen before
    settling, for the console print only -- not registered as individual VAL rows (a jittery wait
    could jump many times before settling, and logging each one was the dominant driver of
    session.data growth in an earlier pass).
    """
    window = deque()  # (timestamp, position)
    n_breaks = 0
    wait_start = time.time()
    warned = False
    while True:
        time.sleep(1.0 / VAR_STILL_POLL_HZ)
        with rotary_lock:
            pos = rotary.current_position()
        now = time.time()
        if window and abs(pos - window[-1][1]) > VAR_STEADY_THRESHOLD_DEG:
            n_breaks += 1
        window.append((now, pos))
        # Keep one entry at-or-before the window start rather than trimming down to nothing --
        # standard sliding-time-window trim.
        while len(window) > 1 and window[1][0] <= now - required_hold:
            window.popleft()
        with register_lock:
            my_bpod.register_value('WHEEL_POS', '{0:.3f},{1:.2f}'.format(now - log_python_t0, pos))
        if not warned and (now - wait_start) > 3.0 + required_hold:
            warned = True
            print("Waiting for the wheel to hold steady (current position: {0:.1f}deg)...".format(
                pos), flush=True)
        positions = [p for _, p in window]
        has_full_span = window[0][0] <= now - required_hold
        if has_full_span and (max(positions) - min(positions)) <= VAR_STEADY_THRESHOLD_DEG:
            return n_breaks


# --- trial loop -----------------------------------------------------------------------------------

print("Starting {0} trials".format(VAR_N_TRIALS), flush=True)

for trial in range(VAR_N_TRIALS):
    try:
        required_hold = random.uniform(VAR_HOLD_MIN_S, VAR_HOLD_MAX_S)

        # Disable event transmission while waiting for stillness so stray settling movement isn't
        # relayed to Bpod at all -- otherwise the firmware could buffer those events and dump them
        # into the state machine the instant it's sent.
        rotary.disable_evt_transmission()
        n_breaks = wait_for_held_steady(required_hold)
        trial_start_t = time.time() - log_python_t0

        with rotary_lock:
            rotary.set_zero_position()
            rotary.set_thresholds(STEADY_AND_CHOICE_THRESHOLDS_DEG)
            rotary.enable_all_thresholds()
            rotary.enable_evt_transmission()

        with register_lock:
            my_bpod.register_value('TRIAL_START', trial_start_t)

        print("Trial {0}: held steady -- choice window open".format(trial + 1), flush=True)

        send_t = time.time() - log_python_t0

        # Extending this machine with more states: Bpod's per-state hardware timer does not
        # restart on a self-referential transition (state X -> state X, confirmed on real
        # hardware earlier this session) -- any new state that needs to loop back to itself must
        # route through a distinct intermediate state instead, or its timer will never actually
        # restart on repeated re-entry.
        sma = StateMachine(my_bpod)

        sma.add_state(
            state_name='WaitForChoice',
            state_timer=VAR_RESPONSE_TIMEOUT,
            state_change_conditions={
                rewarded_event: 'Reward',
                unrewarded_event: 'NoReward',
                Bpod.Events.Tup: 'NoResponse',
            },
            output_actions=[])

        sma.add_state(
            state_name='Reward',
            state_timer=VAR_REWARD_DURATION,
            state_change_conditions={Bpod.Events.Tup: 'exit'},
            output_actions=[(Bpod.OutputChannels.Valve, 1)])

        sma.add_state(
            state_name='NoReward',
            state_timer=0,
            state_change_conditions={Bpod.Events.Tup: 'exit'},
            output_actions=[])

        sma.add_state(
            state_name='NoResponse',
            state_timer=0,
            state_change_conditions={Bpod.Events.Tup: 'exit'},
            output_actions=[])

        if not run_trial_state_machine(sma):
            print("Bpod stopped running trials (Stop/Kill) -- ending session early after trial "
                  "{0}/{1}.".format(trial + 1, VAR_N_TRIALS), flush=True)
            break

        # Ground truth for diagnosing choice-detection issues: which rotary events Bpod actually
        # recorded for this trial, regardless of outcome. events_occurrences is a list of
        # EventOccurrence objects (not a dict) -- get_all_timestamps_by_event() is Trial's own
        # helper for turning that into {event_name: [timestamps]}.
        all_events = my_bpod.session.current_trial.get_all_timestamps_by_event()
        rotary_events = {k: v for k, v in all_events.items() if k.startswith('RotaryEncoder1_')}
        print("Trial {0} raw events: {1}".format(trial + 1, rotary_events), flush=True)

        visited = my_bpod.session.current_trial.states_durations

        def was_visited(name):
            # Bpod logs a synthetic (nan, nan) entry for every state DEFINED in the machine, not
            # just ones actually entered -- so `name in visited` is always true regardless of
            # outcome. A state was genuinely visited only if it has a real (non-nan) start time.
            durations = visited.get(name)
            return bool(durations) and not math.isnan(durations[-1][0])

        outcome = next((name for name in ('Reward', 'NoReward', 'NoResponse') if was_visited(name)),
                        'Unknown')
        rewarded = outcome == 'Reward'

        with register_lock:
            if outcome in ('Reward', 'NoReward'):
                my_bpod.register_value('CHOICE_MADE', send_t + visited[outcome][-1][0])
            elif outcome == 'NoResponse':
                my_bpod.register_value('TIMED_OUT', send_t + visited['NoResponse'][-1][0])

        print("Trial {0}/{1}: held={2:.2f}s (broke {3}x), {4}{5}".format(
            trial + 1, VAR_N_TRIALS, required_hold, n_breaks, outcome,
            ' -> reward' if rewarded else ''), flush=True)

        time.sleep(VAR_ITI)

    except Exception as err:
        print("Trial {0}/{1} FAILED: {2}".format(trial + 1, VAR_N_TRIALS, err), flush=True)
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        raise
else:
    # Only reached if the loop finished all VAR_N_TRIALS iterations without an early `break`
    # (Kill/Stop) -- that path already prints its own "ending session early" message and
    # shouldn't be followed by a claim that every trial completed.
    print("Done: {0} trials completed".format(VAR_N_TRIALS), flush=True)

rotary.close()
my_bpod.close()
