# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Bench validation for the rotary-USB-stream / '#'-marker / SoftCode design every
AuditoryEvidenceAccum stage script now depends on (see _shared/rotary_stream.py and
bpod_trial_helpers.py's TrialRunner(use_stream=True)). Run this on the rig, with no animal, BEFORE
the first real session on the new code -- it checks the firmware behaviors that design assumes but
that couldn't be verified without hardware:

  1. The stream arrives, parses cleanly (no resyncs), and its tick unit resolves (us vs ms).
  2. stream.paused() + set_thresholds() works mid-session (the ack is read correctly and the
     stream resumes afterwards).
  3. A state whose output_actions carry BOTH a rotary relay message ([Z, '#', code]) and a
     SoftCode runs without hanging run_state_machine() (each fires exactly once per trial -- see
     CLAUDE.md's "module output twice in one trial" hazard), on every one of VAR_N_TRIALS trials.
  4. Both SoftCodes arrive, with the same values the state machine sent.
  5. The '#' marker shows up in the stream on every trial, and the first sample after it is ~0deg
     (i.e. 'Z' was applied before '#').
  6. Marker times (module clock) line up with the Bpod state's start (Bpod clock) after removing a
     linear drift: residual SD < 1ms.
  7. Threshold events left enabled between trials are NOT delivered as stale events at the start of
     the next trial (counts events at t < 5ms). Needs the wheel to be turned between trials.
  8. The PC -> Bpod handshake every stage now relies on (TrialRunner.inject_softcode(), via
     pybpod's stdin command channel): an injected 'SoftCode1' reaches the running state machine as
     input event bpod_trial_helpers.PC_SOFTCODE_EVENT, moves it to the next state, and does so
     within a few ms (PC time of the inject vs the Bpod-clock event time, mapped through the
     per-trial host anchor).

**Turn the wheel back and forth, gently but continuously, for the whole run** -- the firmware only
streams a sample when the position changes, and checks 1/5/7 need movement.

Prints PASS/FAIL per check and an OVERALL line. Nothing here touches the valve.
"""
import math
import os
import sys
import threading
import time

_TASK_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_TASK_DIR, '..', '..', '..', '_shared'))
from bpod_trial_helpers import TrialRunner, PC_SOFTCODE_EVENT
import rotary_setup

from pybpodapi.protocol import Bpod, StateMachine

VAR_N_TRIALS = 40
VAR_MARKER_STATE_S = 0.5
VAR_SECOND_STATE_S = 0.2
VAR_INTER_TRIAL_S = 0.5
VAR_THRESHOLD_DEG = 10
VAR_RUN_TIMEOUT_S = 5.0     # a trial taking this much longer than its own states = a hang
VAR_ROTARY_USB_PORT = None

SOFTCODE_A = 1
SOFTCODE_B = 2

results = []


def check(name, ok, detail=''):
    results.append(ok)
    print("[{0}] {1}{2}".format('PASS' if ok else 'FAIL', name,
                                ' -- ' + detail if detail else ''), flush=True)


my_bpod = Bpod()
print("Connected to Bpod on {0}".format(my_bpod.serial_port), flush=True)
rotary, rotary_bpod_module = rotary_setup.connect_rotary(my_bpod, usb_port=VAR_ROTARY_USB_PORT)
trigger_id, rotary_channel = rotary_setup.build_reset_trigger(
    rotary_bpod_module, marker_code=rotary_setup.DOT_ONSET_MARKER_CODE)
neg_event, pos_event = rotary_setup.set_and_enable_thresholds(
    rotary, [-VAR_THRESHOLD_DEG, VAR_THRESHOLD_DEG])
rotary.enable_evt_transmission()

log_python_t0 = time.time()
runner = TrialRunner(my_bpod, rotary, log_python_t0, use_stream=True,
                     marker_code=rotary_setup.DOT_ONSET_MARKER_CODE)
runner.start()

print("Turn the wheel back and forth now, and keep doing so until the test ends.", flush=True)
time.sleep(3.0)

# --- 1. stream health ------------------------------------------------------------------------------
stream = runner.stream
n_samples = len(stream.samples_since_host(0.0))
check("1a stream delivers samples", n_samples > 0, "{0} samples in 3s".format(n_samples))
check("1b tick unit resolved", stream._scale is not None,
      "{0} s/tick".format(stream._scale))
check("1c no byte resyncs", stream.n_resyncs == 0, "{0} resyncs".format(stream.n_resyncs))
if n_samples > 1:
    samples = stream.samples_since_host(0.0)
    span = samples[-1][0] - samples[0][0]
    print("    stream rate while moving: ~{0:.0f} samples/s".format(
        (len(samples) - 1) / span if span > 0 else float('nan')), flush=True)

# --- 2. paused() + ack-reading command mid-stream ------------------------------------------------
try:
    rotary_setup.set_and_enable_thresholds(rotary, [-VAR_THRESHOLD_DEG, VAR_THRESHOLD_DEG],
                                           stream=stream)
    t_resume = time.time() - log_python_t0
    time.sleep(1.0)
    resumed = len(stream.samples_since_host(t_resume)) > 0
    check("2 set_thresholds() inside paused() is acknowledged; stream resumes", resumed,
          "" if resumed else "no samples after resuming (was the wheel moving?)")
except Exception as err:
    check("2 set_thresholds() inside paused() is acknowledged; stream resumes", False, str(err))

# --- 3-7. trials ---------------------------------------------------------------------------------
hangs = 0
softcode_ok = 0
markers = []           # (module_t, bpod_t of the marker state's start)
first_after_marker = []
stale_events = 0
ran_trials = 0

for trial in range(1, VAR_N_TRIALS + 1):
    sma = StateMachine(my_bpod)
    sma.add_state(
        state_name='MarkerState',
        state_timer=VAR_MARKER_STATE_S,
        state_change_conditions={Bpod.Events.Tup: 'SecondState'},
        output_actions=[(rotary_channel, trigger_id), ('SoftCode', SOFTCODE_A)])
    sma.add_state(
        state_name='SecondState',
        state_timer=VAR_SECOND_STATE_S,
        state_change_conditions={Bpod.Events.Tup: 'exit'},
        output_actions=[('SoftCode', SOFTCODE_B)])
    runner.upload(sma)
    time.sleep(VAR_INTER_TRIAL_S)

    result = {}

    def _run():
        try:
            result['ran'] = runner.run_trial_state_machine(sma, arm_rotary=True)
        except SystemExit:
            result['killed'] = True

    t = threading.Thread(target=_run, daemon=True)
    runner.begin_trial()   # fresh SoftCode/marker window for this trial
    t.start()
    t.join(timeout=VAR_MARKER_STATE_S + VAR_SECOND_STATE_S + VAR_RUN_TIMEOUT_S)
    if t.is_alive():
        hangs += 1
        print("Trial {0}: run_state_machine() did not return -- HANG. Stopping.".format(trial),
              flush=True)
        break
    if result.get('killed') or not result.get('ran'):
        print("Stopped/killed at trial {0}.".format(trial), flush=True)
        break
    ran_trials += 1

    received = dict(runner._softcodes)
    if set(received) == {SOFTCODE_A, SOFTCODE_B}:
        softcode_ok += 1
    else:
        print("Trial {0}: softcodes received {1} (expected {2} and {3})".format(
            trial, sorted(received), SOFTCODE_A, SOFTCODE_B), flush=True)

    trial_obj = my_bpod.session.current_trial
    states = trial_obj.states_durations
    bpod_marker_t = trial_obj.trial_start_timestamp + states['MarkerState'][-1][0]
    if runner.last_marker is not None:
        markers.append((runner.last_marker[0], bpod_marker_t))
        after = [p for m, _, p in stream.samples_since_host(runner.last_run_t - 1.0)
                 if m >= runner.last_marker[0]]
        if after:
            first_after_marker.append(after[0])
    else:
        print("Trial {0}: no '#' marker seen in the stream".format(trial), flush=True)

    events = trial_obj.get_all_timestamps_by_event()
    for name in (neg_event, pos_event):
        stale_events += sum(1 for ts in events.get(name, []) if ts < 0.005)

check("3 no run_state_machine() hang with relay + SoftCode in one state", hangs == 0,
      "{0}/{1} trials ran".format(ran_trials, VAR_N_TRIALS))
check("4 both SoftCodes arrive with the values sent", softcode_ok == ran_trials and ran_trials > 0,
      "{0}/{1} trials".format(softcode_ok, ran_trials))
check("5a '#' marker seen in the stream every trial", len(markers) == ran_trials and ran_trials > 0,
      "{0}/{1}".format(len(markers), ran_trials))
if first_after_marker:
    worst = max(abs(p) for p in first_after_marker)
    check("5b first sample after the marker is ~0deg (Z applied before '#')", worst <= 2.0,
          "worst |pos| {0:.1f}deg over {1} trials".format(worst, len(first_after_marker)))
else:
    check("5b first sample after the marker is ~0deg", False, "no samples after any marker "
          "(was the wheel moving?)")

if len(markers) >= 3:
    xs = [m for m, _ in markers]
    ys = [b for _, b in markers]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    res = [y - (my + slope * (x - mx)) for x, y in zip(xs, ys)]
    sd_ms = math.sqrt(sum(r * r for r in res) / n) * 1e3
    check("6 module-clock marker vs Bpod state start: residual SD < 1ms", sd_ms < 1.0,
          "SD {0:.3f}ms, max {1:.3f}ms, drift {2:.1f}ppm".format(
              sd_ms, max(abs(r) for r in res) * 1e3, (slope - 1) * 1e6))
else:
    check("6 module-clock marker vs Bpod state start", False, "fewer than 3 markers")

check("7 no stale threshold events at trial start", stale_events == 0,
      "{0} events at t<5ms (only meaningful if the wheel was turned between trials)".format(
          stale_events))

# --- 8. injected SoftCode round trip ----------------------------------------------------------------
latencies, transitions = [], 0
for trial in range(1, 21):
    sma = StateMachine(my_bpod)
    sma.add_state(state_name='WaitPC', state_timer=2.0,
                  state_change_conditions={PC_SOFTCODE_EVENT: 'GotPC', Bpod.Events.Tup: 'exit'},
                  output_actions=[])
    sma.add_state(state_name='GotPC', state_timer=0.05,
                  state_change_conditions={Bpod.Events.Tup: 'exit'}, output_actions=[])
    runner.upload(sma)
    result = {}
    runner.begin_trial()
    t = threading.Thread(target=lambda: result.update(ran=runner.run_trial_state_machine(sma)),
                         daemon=True)
    t.start()
    time.sleep(0.3)
    runner.inject_softcode()
    inject_host = runner.inject_log[-1][1]
    t.join(timeout=5.0)
    if t.is_alive() or not result.get('ran'):
        print("Trial {0}: round-trip machine did not finish".format(trial), flush=True)
        break
    states = my_bpod.session.current_trial.states_durations
    if states['GotPC'][-1][0] == states['GotPC'][-1][0]:   # not nan
        transitions += 1
        latencies.append((runner.last_run_t + states['GotPC'][-1][0] - inject_host) * 1e3)
check("8a injected SoftCode moves the running machine ({0})".format(PC_SOFTCODE_EVENT),
      transitions == 20, "{0}/20 trials".format(transitions))
if latencies:
    srt = sorted(latencies)
    check("8b inject -> Bpod event latency < 10ms (p95)", srt[int(0.95 * (len(srt) - 1))] < 10.0,
          "median {0:.1f}ms, max {1:.1f}ms (includes the ~ms host-anchor uncertainty)".format(
              srt[len(srt) // 2], srt[-1]))

print("OVERALL: {0}".format("ALL PASS" if results and all(results) else "AT LEAST ONE FAILURE"),
      flush=True)

runner.stop()
rotary.close()
my_bpod.close()
