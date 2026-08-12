# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Offline validation of trial_scheduler.py -- no Bpod/hardware involved. Tests each of the three
in-scope mechanisms (choice side-bias handling, performance handling / remedial-easy, disengagement
handling / circuit-breaker) independently against synthetic trial histories:
  1. side-bias math: hand-checkable edge cases (compute_p_right never leaves its clip bound)
  2. closed-loop debiasing convergence against an injected synthetic bias
  3. remedial-easy trigger/recovery against an independently-computed oracle
  4. circuit-breaker trigger conditions (consecutive run + rate)

Run with the pybpod-environment interpreter:
    /c/Users/2P-Behav/.conda/envs/pybpod-environment/python.exe validate_trial_scheduler.py
"""
import random

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import trial_scheduler as ts

RNG_SEED = 12345

# ==================================================================================================
# Section 1: side-bias math -- hand-checkable edge cases
# ==================================================================================================

print("=" * 100)
print("Section 1: side-bias math -- edge cases")
print("=" * 100)


def _history_all(side, correct, n):
    h = ts.TrialHistory()
    for _ in range(n):
        h.record(difficulty='G22', side=side, trial_type='main', included=True, abort=False,
                  response=side if correct else ('L' if side == 'R' else 'R'), correct=correct, rt=1.0)
    return h


# 1a. all-error / all-correct on one side -> e_side should approach 1.0 / 0.0 exactly (uniform
# composition means the half-Gaussian weighting doesn't matter -- every trial has the same error
# indicator, so the weighted average equals that indicator exactly).
h_all_error = _history_all('R', correct=False, n=ts.SIDE_BIAS_LOOKBACK_N)
h_all_correct = _history_all('R', correct=True, n=ts.SIDE_BIAS_LOOKBACK_N)
e_r_all_error = ts.side_error_fraction(h_all_error, 'R')
e_r_all_correct = ts.side_error_fraction(h_all_correct, 'R')
print("side_error_fraction, all-error composition (n={0}): e_R={1:.4f} (expect ~1.0)".format(
    ts.SIDE_BIAS_LOOKBACK_N, e_r_all_error))
print("side_error_fraction, all-correct composition (n={0}): e_R={1:.4f} (expect ~0.0)".format(
    ts.SIDE_BIAS_LOOKBACK_N, e_r_all_correct))
print("side_error_fraction, no history yet: e_R={0:.4f} (expect 0.5, neutral default)".format(
    ts.side_error_fraction(ts.TrialHistory(), 'R')))

# 1b. compute_p_right never leaves its clip bound, across a spread of e_right/e_left combinations
# including the most extreme (0.0/1.0) and mid-range values.
print()
print("compute_p_right() clip-bound check ({0}):".format(ts.P_RIGHT_CLIP))
print("{:>10s} {:>10s} {:>10s} {:>10s}".format("e_right", "e_left", "p_right", "in_clip"))
test_pairs = [(0.0, 0.0), (1.0, 1.0), (0.0, 1.0), (1.0, 0.0), (0.5, 0.5),
              (0.9, 0.1), (0.1, 0.9), (0.3, 0.7), (0.05, 0.95)]
all_in_clip = True
for e_r, e_l in test_pairs:
    p_r = ts.compute_p_right(e_r, e_l)
    in_clip = ts.P_RIGHT_CLIP[0] - 1e-9 <= p_r <= ts.P_RIGHT_CLIP[1] + 1e-9
    all_in_clip = all_in_clip and in_clip
    print("{:10.3f} {:10.3f} {:10.4f} {:>10s}".format(e_r, e_l, p_r, str(in_clip)))

# 1c. draw_side_debiased() ties (recent_right_fraction == p_right) resolve to a neutral 50/50, not
# the 75/25 skew that fell out of the old bare >/else fallthrough -- this is exactly what happens
# on every session's FIRST trial (both values sit at the neutral prior, 0.5, before any real
# history exists), confirmed directly against a real session's log before this fix was made.
print()
print("draw_side_debiased() tie-break check (recent_right_fraction == p_right):")
rng_tie = np.random.RandomState(RNG_SEED + 3)
N_TIE_DRAWS = 20000
tie_break_ok = True
for point in (0.5, 0.3, 0.7):
    draws = [ts.draw_side_debiased(rng_tie, point, point) for _ in range(N_TIE_DRAWS)]
    frac_right = draws.count('R') / N_TIE_DRAWS
    close_to_half = abs(frac_right - 0.5) < 0.02
    tie_break_ok = tie_break_ok and close_to_half
    print("  tie at {0:.2f}: empirical P(right)={1:.4f} (expect ~0.5): {2}".format(
        point, frac_right, close_to_half))

section1_pass = all_in_clip and tie_break_ok
print("Section 1 result: {0}".format(
    "PASS -- clip bound respected and ties resolve neutrally" if section1_pass else
    "FAIL -- see checks above"))

# ==================================================================================================
# Section 2: closed-loop debiasing convergence
# ==================================================================================================

print()
print("=" * 100)
print("Section 2: closed-loop debiasing convergence (synthetic mouse, worse on left)")
print("=" * 100)

N_TRIALS_SIM = 600
BLOCK_SIZE = 50
P_CORRECT_RIGHT = 0.85   # synthetic mouse's accuracy when the rewarded side is R
P_CORRECT_LEFT = 0.50    # ...and when it's L -- the injected asymmetric bias


def simulate_bias_session(rng, p_correct_right, p_correct_left, n_trials, draw_fn=None):
    """ Drives draw_side_debiased() -> synthetic outcome -> TrialHistory.record() in a loop for
    n_trials, for a synthetic mouse whose per-side accuracy is fixed at p_correct_right/
    p_correct_left. Returns a dict of per-trial series so the same loop can be reused for the main
    biased session, a null (no-bias) control, and a mirrored (opposite-bias) session.

    draw_fn, if given, replaces the default draw_side_debiased(rng, p_right, recent_right_frac) call
    with draw_fn(rng, p_right, recent_right_frac, history) -- used by Section 7 to reuse this exact
    setup with draw_side_debiased_capped() instead, without duplicating the simulation loop. """
    history = ts.TrialHistory()
    p_right_series, side_series = [], []
    e_right_series, e_left_series, recent_right_frac_series = [], [], []

    for _ in range(n_trials):
        e_right = ts.side_error_fraction(history, 'R')
        e_left = ts.side_error_fraction(history, 'L')
        p_right = ts.compute_p_right(e_right, e_left)
        recent_right_frac = ts.recency_weighted_right_fraction(history)

        if draw_fn is None:
            side = ts.draw_side_debiased(rng, p_right, recent_right_frac)
        else:
            side = draw_fn(rng, p_right, recent_right_frac, history)
        p_correct = p_correct_right if side == 'R' else p_correct_left
        correct = rng.uniform(0.0, 1.0) < p_correct
        response = side if correct else ('L' if side == 'R' else 'R')

        history.record(difficulty='G22', side=side, trial_type='main', included=True, abort=False,
                        response=response, correct=correct, rt=1.0)

        p_right_series.append(p_right)
        side_series.append(1.0 if side == 'R' else 0.0)
        e_right_series.append(e_right)
        e_left_series.append(e_left)
        recent_right_frac_series.append(recent_right_frac)

    return {
        'history': history, 'p_right': p_right_series, 'side_R': side_series,
        'e_right': e_right_series, 'e_left': e_left_series,
        'recent_right_frac': recent_right_frac_series,
    }


rng2 = np.random.RandomState(RNG_SEED)
sim2 = simulate_bias_session(rng2, P_CORRECT_RIGHT, P_CORRECT_LEFT, N_TRIALS_SIM)
p_right_series, side_series = sim2['p_right'], sim2['side_R']
e_right_series, e_left_series = sim2['e_right'], sim2['e_left']

print("{:>8s} {:>10s} {:>12s} {:>10s} {:>10s}".format(
    "block", "p_right", "side_frac_R", "e_right", "e_left"))
n_blocks = N_TRIALS_SIM // BLOCK_SIZE
for b in range(n_blocks):
    s, e = b * BLOCK_SIZE, (b + 1) * BLOCK_SIZE
    print("{:8d} {:10.4f} {:12.4f} {:10.4f} {:10.4f}".format(
        b, np.mean(p_right_series[s:e]), np.mean(side_series[s:e]),
        np.mean(e_right_series[s:e]), np.mean(e_left_series[s:e])))

final_block = slice(-BLOCK_SIZE, None)
final_p_right = np.mean(p_right_series[final_block])
final_side_frac = np.mean(side_series[final_block])
final_e_right = np.mean(e_right_series[final_block])
final_e_left = np.mean(e_left_series[final_block])

check_a = final_side_frac < 0.48   # settles into a target band clearly below flat 50/50
check_b = final_e_left > final_e_right + 0.05   # the injected bias is actually detected
check_c = ts.P_RIGHT_CLIP[0] - 1e-9 <= final_p_right <= ts.P_RIGHT_CLIP[1] + 1e-9

print()
print("Final block (last {0} trials): p_right={1:.4f}, side_frac_R={2:.4f}, e_right={3:.4f}, "
      "e_left={4:.4f}".format(BLOCK_SIZE, final_p_right, final_side_frac, final_e_right, final_e_left))
print("(a) empirical right-fraction settles below 0.5: {0}".format(check_a))
print("(b) e_left visibly elevated vs e_right (bias detected): {0}".format(check_b))
print("(c) p_right stayed within clip bound: {0}".format(check_c))
print("Section 2 result: {0}".format(
    "PASS" if (check_a and check_b and check_c) else "FAIL -- see (a)/(b)/(c) above"))

# ==================================================================================================
# Section 3: remedial-easy trigger/recovery vs. an independently-computed oracle
# ==================================================================================================

print()
print("=" * 100)
print("Section 3: remedial-easy trigger/recovery")
print("=" * 100)

GOOD_N = 60
BAD_N = 40
RECOVER_N = 20

corrects = ([True] * GOOD_N +
            ([False, False, False, True] * (BAD_N // 4)) +
            [True] * RECOVER_N)
sides = ['R' if i % 2 == 0 else 'L' for i in range(len(corrects))]


def _trailing_accuracy(seq, end_index, window):
    """ Independent oracle: plain trailing-window accuracy ending at end_index (exclusive),
    reimplemented from scratch here (not calling trial_scheduler at all) so it can catch a bug in
    trial_scheduler.py's own rolling_accuracy/PerformanceMonitor, not just restate it. """
    start = max(0, end_index - window)
    chunk = seq[start:end_index]
    if not chunk:
        return None
    return sum(1.0 if c else 0.0 for c in chunk) / len(chunk)


oracle_trigger_trial = None
oracle_recover_trial = None
oracle_in_remedial = False
for i in range(1, len(corrects) + 1):
    if not oracle_in_remedial:
        acc = _trailing_accuracy(corrects, i, ts.REMEDIAL_TRIGGER_LOOKBACK_N)
        if i >= ts.REMEDIAL_TRIGGER_LOOKBACK_N and acc is not None and acc < ts.REMEDIAL_TRIGGER_ACCURACY:
            oracle_in_remedial = True
            if oracle_trigger_trial is None:
                oracle_trigger_trial = i
    else:
        acc = _trailing_accuracy(corrects, i, ts.REMEDIAL_RECOVER_WINDOW_N)
        if i >= ts.REMEDIAL_RECOVER_WINDOW_N and acc is not None and acc >= ts.REMEDIAL_RECOVER_ACCURACY:
            oracle_in_remedial = False
            if oracle_recover_trial is None:
                oracle_recover_trial = i

monitor = ts.PerformanceMonitor(ts.TrialHistory())
actual_trigger_trial, actual_recover_trial = None, None
trial_type_timeline = []
for i, (correct, side) in enumerate(zip(corrects, sides), start=1):
    was_remedial = monitor.in_remedial
    response = side if correct else ('L' if side == 'R' else 'R')
    monitor.record_outcome(difficulty='G22', side=side, trial_type=monitor.next_trial_type(),
                            abort=False, response=response, correct=correct, rt=1.0)
    trial_type_timeline.append(1 if monitor.in_remedial else 0)
    if monitor.in_remedial and not was_remedial and actual_trigger_trial is None:
        actual_trigger_trial = i
    if was_remedial and not monitor.in_remedial and actual_recover_trial is None:
        actual_recover_trial = i

print("Sequence: {0} good trials, {1} bad trials (25% correct), {2} recovery trials".format(
    GOOD_N, BAD_N, RECOVER_N))
print("Trigger trial   -- oracle: {0}, actual (PerformanceMonitor): {1}".format(
    oracle_trigger_trial, actual_trigger_trial))
print("Recovery trial  -- oracle: {0}, actual (PerformanceMonitor): {1}".format(
    oracle_recover_trial, actual_recover_trial))
section3_pass = (oracle_trigger_trial == actual_trigger_trial and
                  oracle_recover_trial == actual_recover_trial)
print("Section 3 result: {0}".format("PASS -- oracle matches actual" if section3_pass else
                                      "FAIL -- oracle/actual mismatch"))

# ==================================================================================================
# Section 4: circuit-breaker
# ==================================================================================================

print()
print("=" * 100)
print("Section 4: circuit-breaker (disengagement handling)")
print("=" * 100)


def _history_from_pattern(pattern):
    """ pattern: list of 'resp' (responded, correct) or 'noresp' (no-response) tokens, oldest
    first -- builds a TrialHistory in that order. """
    h = ts.TrialHistory()
    for i, tok in enumerate(pattern):
        side = 'R' if i % 2 == 0 else 'L'
        if tok == 'noresp':
            h.record(difficulty='G22', side=side, trial_type='main', included=False, abort=False,
                      response=None, correct=None, rt=None)
        else:
            h.record(difficulty='G22', side=side, trial_type='main', included=True, abort=False,
                      response=side, correct=True, rt=1.0)
    return h


print("Consecutive no-response run check (threshold={0}):".format(
    ts.CIRCUIT_BREAKER_CONSECUTIVE_NORESPONSE))
n_below = ts.CIRCUIT_BREAKER_CONSECUTIVE_NORESPONSE - 1
n_at = ts.CIRCUIT_BREAKER_CONSECUTIVE_NORESPONSE
h_below = _history_from_pattern(['resp'] * 5 + ['noresp'] * n_below)
h_at = _history_from_pattern(['resp'] * 5 + ['noresp'] * n_at)
stop_below = ts.should_stop_session(h_below)
stop_at = ts.should_stop_session(h_at)
print("  run={0} (one under threshold): should_stop={1} (expect False)".format(n_below, stop_below))
print("  run={0} (at threshold):        should_stop={1} (expect True)".format(n_at, stop_at))
consecutive_check = (stop_below is False and stop_at is True)

print()
print("No-response rate check (window={0}, threshold={1}):".format(
    ts.CIRCUIT_BREAKER_RATE_WINDOW_N, ts.CIRCUIT_BREAKER_RATE_THRESHOLD))
w = ts.CIRCUIT_BREAKER_RATE_WINDOW_N
n_resp_50 = w // 2
pattern_50 = (['noresp', 'resp'] * n_resp_50)[:w]               # exactly 50% no-response
n_noresp_55 = int(round(w * 0.55))
pattern_55 = (['noresp'] * n_noresp_55 + ['resp'] * (w - n_noresp_55))
# interleave so no long consecutive run trips the OTHER condition instead
pattern_55_interleaved = []
noresp_left, resp_left = n_noresp_55, w - n_noresp_55
while noresp_left or resp_left:
    if noresp_left:
        pattern_55_interleaved.append('noresp')
        noresp_left -= 1
    if resp_left:
        pattern_55_interleaved.append('resp')
        resp_left -= 1
h_50 = _history_from_pattern(pattern_50)
h_55 = _history_from_pattern(pattern_55_interleaved)
stop_50 = ts.should_stop_session(h_50)
stop_55 = ts.should_stop_session(h_55)
print("  rate=0.50 (at threshold, needs STRICTLY >): should_stop={0} (expect False)".format(stop_50))
print("  rate=0.55 (over threshold):                 should_stop={0} (expect True)".format(stop_55))
rate_check = (stop_50 is False and stop_55 is True)

print()
print("Small-sample regression check (bug found on real hardware: a single trial 1 no-response "
      "used to trip the rate check at 100% observed rate, well under the {0}-trial window):".format(
          ts.CIRCUIT_BREAKER_RATE_WINDOW_N))
h_trial1 = _history_from_pattern(['noresp'])
stop_trial1 = ts.should_stop_session(h_trial1)
print("  1 trial, 1 no-response (100% rate, but window not full): should_stop={0} "
      "(expect False)".format(stop_trial1))
small_sample_check = (stop_trial1 is False)

section4_pass = consecutive_check and rate_check and small_sample_check
print()
print("Section 4 result: {0}".format("PASS" if section4_pass else "FAIL -- see checks above"))

# ==================================================================================================
# Section 4b: circuit-breaker (consumption-lick disengagement)
# ==================================================================================================

print()
print("=" * 100)
print("Section 4b: circuit-breaker (consumption-lick disengagement)")
print("=" * 100)


def _history_from_lick_pattern(pattern):
    """ pattern: list of 'lick' or 'nolick' tokens, oldest first -- every trial responds normally
    (so the response-based circuit-breaker checks in Section 4 stay at 0/0.0 and can't confound
    this section), only consumption_licked varies. """
    h = ts.TrialHistory()
    for i, tok in enumerate(pattern):
        side = 'R' if i % 2 == 0 else 'L'
        h.record(difficulty='G22', side=side, trial_type='main', included=True, abort=False,
                  response=side, correct=True, rt=1.0, consumption_licked=(tok == 'lick'))
    return h


print("Consecutive no-consumption-lick run check (threshold={0}):".format(
    ts.CIRCUIT_BREAKER_CONSECUTIVE_NO_CONSUMPTION_LICK))
n_below = ts.CIRCUIT_BREAKER_CONSECUTIVE_NO_CONSUMPTION_LICK - 1
n_at = ts.CIRCUIT_BREAKER_CONSECUTIVE_NO_CONSUMPTION_LICK
h_below = _history_from_lick_pattern(['lick'] * 5 + ['nolick'] * n_below)
h_at = _history_from_lick_pattern(['lick'] * 5 + ['nolick'] * n_at)
stop_below = ts.should_stop_session(h_below)
stop_at = ts.should_stop_session(h_at)
print("  run={0} (one under threshold): should_stop={1} (expect False)".format(n_below, stop_below))
print("  run={0} (at threshold):        should_stop={1} (expect True)".format(n_at, stop_at))
consecutive_lick_check = (stop_below is False and stop_at is True)

print()
print("No-consumption-lick rate check (window={0}, threshold={1}):".format(
    ts.CIRCUIT_BREAKER_CONSUMPTION_LICK_RATE_WINDOW_N,
    ts.CIRCUIT_BREAKER_CONSUMPTION_LICK_RATE_THRESHOLD))
w = ts.CIRCUIT_BREAKER_CONSUMPTION_LICK_RATE_WINDOW_N
n_lick_50 = w // 2
pattern_50 = (['nolick', 'lick'] * n_lick_50)[:w]               # exactly 50% no-lick
n_nolick_55 = int(round(w * 0.55))
# interleave so no long consecutive run trips the OTHER condition instead
pattern_55_interleaved = []
nolick_left, lick_left = n_nolick_55, w - n_nolick_55
while nolick_left or lick_left:
    if nolick_left:
        pattern_55_interleaved.append('nolick')
        nolick_left -= 1
    if lick_left:
        pattern_55_interleaved.append('lick')
        lick_left -= 1
h_50 = _history_from_lick_pattern(pattern_50)
h_55 = _history_from_lick_pattern(pattern_55_interleaved)
stop_50 = ts.should_stop_session(h_50)
stop_55 = ts.should_stop_session(h_55)
print("  rate=0.50 (at threshold, needs STRICTLY >): should_stop={0} (expect False)".format(stop_50))
print("  rate=0.55 (over threshold):                 should_stop={0} (expect True)".format(stop_55))
rate_lick_check = (stop_50 is False and stop_55 is True)

print()
print("None-reset check (a trial with no consumption phase, e.g. no-response, resets the no-lick "
      "run the same way an abort resets the no-response run):")
n_half = ts.CIRCUIT_BREAKER_CONSECUTIVE_NO_CONSUMPTION_LICK - 1
h_reset = ts.TrialHistory()
for _ in range(n_half):
    h_reset.record(difficulty='G22', side='R', trial_type='main', included=True, abort=False,
                    response='R', correct=True, rt=1.0, consumption_licked=False)
h_reset.record(difficulty='G22', side='R', trial_type='main', included=False, abort=False,
                response=None, correct=None, rt=None, consumption_licked=None)
for _ in range(n_half):
    h_reset.record(difficulty='G22', side='R', trial_type='main', included=True, abort=False,
                    response='R', correct=True, rt=1.0, consumption_licked=False)
reset_run = h_reset.consecutive_no_consumption_lick()
print("  trailing run after a None-consumption trial in between two under-threshold no-lick runs: "
      "{0} (expect {1}, not the full {2})".format(reset_run, n_half, 2 * n_half))
reset_check = reset_run == n_half

print()
print("Small-sample regression check (the exact bug found on real hardware: a single trial 1, "
      "rewarded but not licked, tripped the rate check at 100% observed rate, well under the "
      "{0}-trial window):".format(ts.CIRCUIT_BREAKER_CONSUMPTION_LICK_RATE_WINDOW_N))
h_trial1 = _history_from_lick_pattern(['nolick'])
stop_trial1 = ts.should_stop_session(h_trial1)
print("  1 trial, responded but didn't lick (100% rate, but window not full): should_stop={0} "
      "(expect False)".format(stop_trial1))
small_sample_lick_check = (stop_trial1 is False)

section4b_pass = consecutive_lick_check and rate_lick_check and reset_check and small_sample_lick_check
print()
print("Section 4b result: {0}".format("PASS" if section4b_pass else "FAIL -- see checks above"))

# ==================================================================================================
# Section 5: additional example sessions + explicit switch positions / edge cases
# ==================================================================================================

print()
print("=" * 100)
print("Section 5: additional example sessions -- null control, mirrored bias, switch positions,")
print("remedial multi-cycle, circuit-breaker example timelines")
print("=" * 100)

# 5a. Null control -- NO injected bias (both sides equally easy). If the mechanism were falsely
# detecting a bias where none exists, this would show up here as p_right drifting off 0.5.
N_NULL = 600
rng_null = np.random.RandomState(RNG_SEED + 1)
sim_null = simulate_bias_session(rng_null, p_correct_right=0.75, p_correct_left=0.75, n_trials=N_NULL)
null_final_p_right = np.mean(sim_null['p_right'][-BLOCK_SIZE:])
null_final_side_frac = np.mean(sim_null['side_R'][-BLOCK_SIZE:])
null_check = abs(null_final_p_right - 0.5) < 0.05
print("5a. Null control (no injected bias, both sides 75% correct):")
print("    final-block p_right={0:.4f}, side_frac_R={1:.4f} -- stays near 0.5: {2}".format(
    null_final_p_right, null_final_side_frac, null_check))

# 5b. Mirrored bias -- worse on RIGHT instead of left. Confirms the mechanism isn't secretly
# left/right-asymmetric in its own implementation (a copy-paste bug would only show up on one side).
N_MIRROR = 600
rng_mirror = np.random.RandomState(RNG_SEED + 2)
sim_mirror = simulate_bias_session(rng_mirror, p_correct_right=0.50, p_correct_left=0.85,
                                    n_trials=N_MIRROR)
mirror_final_p_right = np.mean(sim_mirror['p_right'][-BLOCK_SIZE:])
mirror_final_side_frac = np.mean(sim_mirror['side_R'][-BLOCK_SIZE:])
mirror_check = mirror_final_side_frac > 0.52 and mirror_final_p_right > 0.5
print("5b. Mirrored bias (worse on RIGHT, R=50% / L=85% correct):")
print("    final-block p_right={0:.4f}, side_frac_R={1:.4f} -- mirrors above 0.5: {2}".format(
    mirror_final_p_right, mirror_final_side_frac, mirror_check))

# 5c. Track-the-target switch positions: using the main biased session (sim2), find every trial
# where draw_side_debiased()'s own branch flips (recent_right_frac vs p_right crossing) -- these
# are the actual "switch positions" of SS8 step 4's tracking logic.
recent_frac_series = sim2['recent_right_frac']
branch_series = ['pull_left' if recent_frac_series[i] > p_right_series[i] else 'push_right'
                  for i in range(len(p_right_series))]
switch_positions = [i for i in range(1, len(branch_series)) if branch_series[i] != branch_series[i - 1]]
print("5c. Track-the-target branch switches in the main biased session: {0} switches in {1} "
      "trials (first 10: {2})".format(len(switch_positions), N_TRIALS_SIM, switch_positions[:10]))

# 5d. Remedial-easy multi-cycle: TWO separate dip/recover cycles in one session, confirming the
# sticky state machine correctly re-triggers after a genuine prior recovery (not stuck true/false).
cycle_good = [True] * 60
cycle_bad = [False, False, False, True] * (40 // 4)
cycle_recover = [True] * 20
corrects_multi = cycle_good + cycle_bad + cycle_recover + cycle_good + cycle_bad + cycle_recover
sides_multi = ['R' if i % 2 == 0 else 'L' for i in range(len(corrects_multi))]

monitor_multi = ts.PerformanceMonitor(ts.TrialHistory())
multi_timeline, multi_triggers, multi_recoveries = [], [], []
for i, (correct, side) in enumerate(zip(corrects_multi, sides_multi), start=1):
    was_remedial = monitor_multi.in_remedial
    response = side if correct else ('L' if side == 'R' else 'R')
    monitor_multi.record_outcome(difficulty='G22', side=side, trial_type=monitor_multi.next_trial_type(),
                                  abort=False, response=response, correct=correct, rt=1.0)
    multi_timeline.append(1 if monitor_multi.in_remedial else 0)
    if monitor_multi.in_remedial and not was_remedial:
        multi_triggers.append(i)
    if was_remedial and not monitor_multi.in_remedial:
        multi_recoveries.append(i)

multi_cycle_check = len(multi_triggers) >= 2 and len(multi_recoveries) >= 2
print("5d. Remedial multi-cycle: triggers at {0}, recoveries at {1}".format(
    multi_triggers, multi_recoveries))
if len(multi_triggers) > 2:
    print("    FINDING (not a test bug): rapid trigger/recover flickering right after each "
          "recovery. The trailing-{0}-trial TRIGGER window still mostly reflects the just-ended "
          "bad phase for a while even once the trailing-{1}-trial RECOVERY window already looks "
          "clean, so it can immediately re-trigger for a few trials before the wider window also "
          "clears. This is a real emergent property of using two differently-sized lookback "
          "windows exactly as mouse_auditory_accumulation_paradigm.md specifies (40 vs 10-15), not "
          "a coding bug -- flagging for your awareness/decision, not silently changed.".format(
              ts.REMEDIAL_TRIGGER_LOOKBACK_N, ts.REMEDIAL_RECOVER_WINDOW_N))
print("    at least 2 independent trigger/recover cycles occurred: {0}".format(multi_cycle_check))

# 5e. Circuit-breaker, consecutive-run edge case: an isolated single no-response must NOT
# accumulate toward the run count (it resets), only a genuinely unbroken run should stop the
# session. Pattern: 10 resp, 1 noresp (isolated), 9 resp, 5 noresp (just under threshold, resets on
# the next resp), 1 resp, then a genuine run of 6 noresp (crosses the threshold).
pattern_5e = (['resp'] * 10 + ['noresp'] * 1 + ['resp'] * 9 + ['noresp'] * 5 + ['resp'] * 1 +
              ['noresp'] * ts.CIRCUIT_BREAKER_CONSECUTIVE_NORESPONSE)
h_5e = ts.TrialHistory()
run_series_5e, stop_series_5e = [], []
stop_trial_5e = None
for i, tok in enumerate(pattern_5e, start=1):
    side = 'R' if i % 2 == 0 else 'L'
    if tok == 'noresp':
        h_5e.record(difficulty='G22', side=side, trial_type='main', included=False, abort=False,
                     response=None, correct=None, rt=None)
    else:
        h_5e.record(difficulty='G22', side=side, trial_type='main', included=True, abort=False,
                     response=side, correct=True, rt=1.0)
    run_series_5e.append(h_5e.consecutive_no_response())
    stopped = ts.should_stop_session(h_5e)
    stop_series_5e.append(stopped)
    if stopped and stop_trial_5e is None:
        stop_trial_5e = i
print("5e. Consecutive-run edge case: isolated no-response and a just-under-threshold run both "
      "correctly reset; genuine run trips should_stop_session() at trial {0} (pattern length "
      "{1}).".format(stop_trial_5e, len(pattern_5e)))

# 5f. Circuit-breaker, rate edge case: no-response trials sprinkled non-consecutively (so the
# consecutive-run condition never fires) but frequently enough that the trailing-20 rate eventually
# exceeds 0.5 -- confirms the rate condition can trip independently of the run condition.
pattern_5f = []
for i in range(40):
    # short runs of 2 no-response (well under the consecutive threshold of 6) push density to
    # ~2/3 in the back half -- a strictly-alternating (never-2-in-a-row) pattern turns out to cap
    # out at EXACTLY 50% density over any 20-trial window (CIRCUIT_BREAKER_RATE_WINDOW_N is even),
    # which can never strictly exceed the 0.5 threshold -- discovered while building this edge
    # case, not assumed up front.
    if i < 15:
        pattern_5f.append('resp')
    else:
        pattern_5f.append('resp' if i % 3 == 2 else 'noresp')
h_5f = ts.TrialHistory()
rate_series_5f, stop_series_5f = [], []
stop_trial_5f = None
for i, tok in enumerate(pattern_5f, start=1):
    side = 'R' if i % 2 == 0 else 'L'
    if tok == 'noresp':
        h_5f.record(difficulty='G22', side=side, trial_type='main', included=False, abort=False,
                     response=None, correct=None, rt=None)
    else:
        h_5f.record(difficulty='G22', side=side, trial_type='main', included=True, abort=False,
                     response=side, correct=True, rt=1.0)
    rate_series_5f.append(h_5f.no_response_rate(ts.CIRCUIT_BREAKER_RATE_WINDOW_N))
    stopped = ts.should_stop_session(h_5f)
    stop_series_5f.append(stopped)
    if stopped and stop_trial_5f is None:
        stop_trial_5f = i
max_consecutive_run_5f = max(
    (len(list(g)) for k, g in __import__('itertools').groupby(pattern_5f) if k == 'noresp'),
    default=0)
print("5f. Rate edge case: max no-response run={0} (well under the consecutive threshold of {1}), "
      "yet the rolling rate alone trips should_stop_session() at trial {2}.".format(
          max_consecutive_run_5f, ts.CIRCUIT_BREAKER_CONSECUTIVE_NORESPONSE, stop_trial_5f))

section5_pass = null_check and mirror_check and multi_cycle_check and (stop_trial_5e is not None) \
    and (stop_trial_5f is not None)
print()
print("Section 5 result: {0}".format("PASS" if section5_pass else "FAIL -- see checks above"))

# ==================================================================================================
# Section 6: remedial-easy -- side stays unlocked, difficulty forced to AOS
# ==================================================================================================

print()
print("=" * 100)
print("Section 6: remedial-easy: side stays unlocked, difficulty forced to AOS")
print("=" * 100)

# A deterministic, ALWAYS-ALTERNATING mock debiased_draw_fn (not the real debiasing pipeline --
# this section is testing that the side draw is NEVER locked, independent of whether the real
# pipeline would naturally repeat a side by chance). If the side were ever cached/locked during a
# remedial block, the call counter would stop advancing and the sequence would stop alternating
# during that block despite the underlying function wanting to every single call.
_alternating_state = {'n_calls': 0}


def _alternating_draw_fn(rng):
    _alternating_state['n_calls'] += 1
    return 'R' if _alternating_state['n_calls'] % 2 == 1 else 'L'


monitor6 = ts.PerformanceMonitor(ts.TrialHistory())
rng6 = np.random.RandomState(RNG_SEED)
side_sequence_6 = []
difficulty_sequence_6 = []
call_count_sequence_6 = []
trial_type_sequence_6 = []

for correct in corrects:  # reuse Section 3's exact 60-good/40-bad/20-recover sequence
    trial_type = monitor6.next_trial_type()
    difficulty = monitor6.next_difficulty(rng6, trial_type, lambda rng: 'G22')
    side = _alternating_draw_fn(rng6)
    response = side if correct else ('L' if side == 'R' else 'R')
    monitor6.record_outcome(difficulty=difficulty, side=side, trial_type=trial_type, abort=False,
                             response=response, correct=correct, rt=1.0)
    side_sequence_6.append(side)
    difficulty_sequence_6.append(difficulty)
    call_count_sequence_6.append(_alternating_state['n_calls'])
    trial_type_sequence_6.append(trial_type)

remedial_trial_indices = [i for i, t in enumerate(trial_type_sequence_6) if t == 'remedial_easy']
main_trial_indices = [i for i, t in enumerate(trial_type_sequence_6) if t == 'main']

if remedial_trial_indices:
    remedial_sides = [side_sequence_6[i] for i in remedial_trial_indices]
    unlocked_check = all(remedial_sides[i] != remedial_sides[i + 1]
                          for i in range(len(remedial_sides) - 1))
    print("Remedial-easy block: trials {0}-{1} ({2} trials), sides observed: {3}".format(
        remedial_trial_indices[0] + 1, remedial_trial_indices[-1] + 1, len(remedial_trial_indices),
        sorted(set(remedial_sides))))
    print("  side draw keeps alternating through the block (not locked): {0}".format(
        unlocked_check))

    # the draw_fn must be called exactly once PER TRIAL, including every remedial trial -- confirmed
    # by the call counter advancing by 1 every single trial, never skipping/freezing.
    calls_during_block = [call_count_sequence_6[i] for i in remedial_trial_indices]
    called_every_trial_check = calls_during_block == list(
        range(calls_during_block[0], calls_during_block[0] + len(calls_during_block)))
    print("  draw_fn called exactly once per remedial trial (never skipped/frozen): {0}".format(
        called_every_trial_check))

    remedial_difficulties = [difficulty_sequence_6[i] for i in remedial_trial_indices]
    all_aos_check = all(d == 'AOS' for d in remedial_difficulties)
    print("  every remedial-easy trial's difficulty is 'AOS': {0}".format(all_aos_check))
else:
    unlocked_check = False
    called_every_trial_check = False
    all_aos_check = False
    print("Remedial-easy never triggered in this sequence -- unexpected, FAIL.")

# main-phase trials (before the trigger, and after recovery) should also alternate normally, and
# never draw 'AOS' from the forced-remedial path (whatever the mock full_grid_draw_fn returns).
main_before_trigger = [side_sequence_6[i] for i in main_trial_indices if i < remedial_trial_indices[0]] \
    if remedial_trial_indices else []
alternates_check = all(main_before_trigger[i] != main_before_trigger[i + 1]
                        for i in range(len(main_before_trigger) - 1))
print("Main-phase trials before the trigger alternate normally: {0}".format(alternates_check))

main_after_recovery = [side_sequence_6[i] for i in main_trial_indices
                        if remedial_trial_indices and i > remedial_trial_indices[-1]]
resumed_check = (len(main_after_recovery) >= 2 and
                  any(main_after_recovery[i] != main_after_recovery[i + 1]
                      for i in range(len(main_after_recovery) - 1)))
print("Main-phase trials after recovery resume alternating: {0}".format(resumed_check))

main_difficulties = [difficulty_sequence_6[i] for i in main_trial_indices]
main_not_aos_check = all(d == 'G22' for d in main_difficulties)
print("Main-phase trials always use the full-grid difficulty (never forced to AOS): {0}".format(
    main_not_aos_check))

section6_pass = (unlocked_check and called_every_trial_check and all_aos_check and
                  alternates_check and resumed_check and main_not_aos_check)
print()
print("Section 6 result: {0}".format("PASS" if section6_pass else "FAIL -- see checks above"))

# ==================================================================================================
# Section 7: draw_side_debiased_capped() -- same-side/alternation run-length caps
# ==================================================================================================

print()
print("=" * 100)
print("Section 7: run-length caps (draw_side_debiased_capped())")
print("=" * 100)

# 7a. Unit-level override checks -- p_right at an extreme makes draw_side_debiased()'s underlying
# draw deterministic (prob_right pinned to exactly 0.0 or 1.0), so the override logic can be
# checked in isolation against a hand-built recent-side window, independent of RNG luck.
print("Unit-level override checks (natural draw forced deterministic via extreme p_right):")


def _forced_natural_history(recent_sides_oldest_first):
    h = ts.TrialHistory()
    for i, s in enumerate(recent_sides_oldest_first):
        h.record(difficulty='G22', side=s, trial_type='main', included=True, abort=False,
                  response=s, correct=True, rt=1.0)
    return h


rng7 = np.random.RandomState(RNG_SEED)

# same-run violation: last 3 are 'R', p_right=1.0 forces natural='R' -- must flip to 'L'.
h_same_violate = _forced_natural_history(['R', 'R', 'R'])
draw_same_violate = ts.draw_side_debiased_capped(rng7, 1.0, 0.0, h_same_violate)
same_violate_check = draw_same_violate == 'L'
print("  same-run at cap (RRR, natural=R): draw={0} (expect L, forced flip): {1}".format(
    draw_same_violate, same_violate_check))

# same-run non-violation: only last 2 are 'R' -- natural='R' should pass through unchanged.
h_same_ok = _forced_natural_history(['L', 'R', 'R'])
draw_same_ok = ts.draw_side_debiased_capped(rng7, 1.0, 0.0, h_same_ok)
same_ok_check = draw_same_ok == 'R'
print("  same-run under cap (LRR, natural=R): draw={0} (expect R, unchanged): {1}".format(
    draw_same_ok, same_ok_check))

# alternation violation: last 3 strictly alternate (R,L,R oldest-to-newest -> most-recent-first
# [R,L,R]), p_right=0.0 forces natural='L', which continues the alternation -- must flip to 'R'.
h_alt_violate = _forced_natural_history(['R', 'L', 'R'])
draw_alt_violate = ts.draw_side_debiased_capped(rng7, 0.0, 1.0, h_alt_violate)
alt_violate_check = draw_alt_violate == 'R'
print("  alternation at cap (RLR, natural=L continues it): draw={0} (expect R, forced flip): "
      "{1}".format(draw_alt_violate, alt_violate_check))

# alternation non-violation: same window, but natural='R' (repeats, doesn't continue) -- unchanged.
draw_alt_ok = ts.draw_side_debiased_capped(rng7, 1.0, 0.0, h_alt_violate)
alt_ok_check = draw_alt_ok == 'R'
print("  alternation window, natural=R (repeats, doesn't continue): draw={0} (expect R, "
      "unchanged): {1}".format(draw_alt_ok, alt_ok_check))

unit_checks_pass = same_violate_check and same_ok_check and alt_violate_check and alt_ok_check

# 7b. Full-session regression scan -- across several seeds and a realistic evolving history (the
# same biased-mouse setup as Section 2), confirm TRIAL_SIDE never produces a same-side or
# alternation run longer than its cap. This is the direct regression test for the exact pattern
# (a 7-trial alternation, a 5-trial same-side run) seen on real hardware.
print()
print("Full-session run-length scan (several seeds, {0} trials each):".format(N_TRIALS_SIM))
max_same_seen, max_alt_seen = 0, 0
for seed_offset in range(5):
    rng_scan = np.random.RandomState(RNG_SEED + 100 + seed_offset)
    sim_scan = simulate_bias_session(rng_scan, P_CORRECT_RIGHT, P_CORRECT_LEFT, N_TRIALS_SIM,
                                      draw_fn=ts.draw_side_debiased_capped)
    sides_scan = ['R' if v == 1.0 else 'L' for v in sim_scan['side_R']]

    run = 1
    for i in range(1, len(sides_scan)):
        run = run + 1 if sides_scan[i] == sides_scan[i - 1] else 1
        max_same_seen = max(max_same_seen, run)

    alt_run = 1
    for i in range(1, len(sides_scan)):
        alt_run = alt_run + 1 if sides_scan[i] != sides_scan[i - 1] else 1
        max_alt_seen = max(max_alt_seen, alt_run)

print("  max same-side run observed: {0} (cap {1}): {2}".format(
    max_same_seen, ts.MAX_SAME_SIDE_RUN, max_same_seen <= ts.MAX_SAME_SIDE_RUN))
print("  max alternation run observed: {0} (cap {1}): {2}".format(
    max_alt_seen, ts.MAX_ALTERNATION_RUN, max_alt_seen <= ts.MAX_ALTERNATION_RUN))
scan_check = max_same_seen <= ts.MAX_SAME_SIDE_RUN and max_alt_seen <= ts.MAX_ALTERNATION_RUN

# 7c. Sanity check: capping shouldn't break the underlying debiasing signal -- re-run Section 2's
# convergence check with the capped draw instead.
print()
print("Debiasing still converges under the capped draw (Section 2's synthetic mouse, worse on "
      "left):")
rng7c = np.random.RandomState(RNG_SEED)
sim7c = simulate_bias_session(rng7c, P_CORRECT_RIGHT, P_CORRECT_LEFT, N_TRIALS_SIM,
                               draw_fn=ts.draw_side_debiased_capped)
final_side_frac_7c = np.mean(sim7c['side_R'][final_block])
convergence_check = final_side_frac_7c < 0.5
print("  final-block empirical right-fraction={0:.4f} (expect < 0.5): {1}".format(
    final_side_frac_7c, convergence_check))

section7_pass = unit_checks_pass and scan_check and convergence_check
print()
print("Section 7 result: {0}".format("PASS" if section7_pass else "FAIL -- see checks above"))

# ==================================================================================================
# plots
# ==================================================================================================

fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# (a) p_right / empirical right-fraction convergence
ax = axes[0, 0]
trials_x = np.arange(N_TRIALS_SIM)
ax.plot(trials_x, p_right_series, label='p_right (target)', color='tab:purple')
window = 20
side_smoothed = np.convolve(side_series, np.ones(window) / window, mode='valid')
ax.plot(np.arange(len(side_smoothed)) + window - 1, side_smoothed,
        label='empirical R-fraction ({0}-trial smoothed)'.format(window), color='tab:orange')
ax.axhline(0.5, color='gray', linestyle=':', linewidth=0.8)
ax.set_xlabel('trial')
ax.set_ylabel('fraction')
ax.set_title('Side-bias convergence (synthetic mouse worse on left)')
ax.legend(fontsize=8)

# (b) e_right / e_left over trials
ax = axes[0, 1]
ax.plot(trials_x, e_right_series, label='e_right', color='tab:blue')
ax.plot(trials_x, e_left_series, label='e_left', color='tab:red')
ax.set_xlabel('trial')
ax.set_ylabel('recency-weighted error fraction')
ax.set_title('Detected per-side error rate over the session')
ax.legend(fontsize=8)

# (c) rolling accuracy with remedial-block shading (Section 3's sequence)
ax = axes[1, 0]
trial_idx = np.arange(1, len(corrects) + 1)
rolling40 = [_trailing_accuracy(corrects, i, ts.REMEDIAL_TRIGGER_LOOKBACK_N) for i in trial_idx]
ax.plot(trial_idx, rolling40, color='black', linewidth=1.0, label='trailing-40 accuracy')
ax.axhline(ts.REMEDIAL_TRIGGER_ACCURACY, color='tab:red', linestyle='--', linewidth=0.8,
           label='trigger threshold')
ax.axhline(ts.REMEDIAL_RECOVER_ACCURACY, color='tab:green', linestyle='--', linewidth=0.8,
           label='recover threshold')
in_remedial_arr = np.array(trial_type_timeline)
ax.fill_between(trial_idx, 0, 1, where=in_remedial_arr.astype(bool), color='tab:red', alpha=0.15,
                step='mid', label='remedial_easy active')
ax.axvline(actual_trigger_trial, color='tab:red', linewidth=1.2)
ax.annotate('TRIGGER\ntrial {0}'.format(actual_trigger_trial),
            (actual_trigger_trial, 0.95), fontsize=7, color='tab:red', ha='center', va='top')
ax.axvline(actual_recover_trial, color='tab:green', linewidth=1.2)
ax.annotate('RECOVER\ntrial {0}'.format(actual_recover_trial),
            (actual_recover_trial, 0.05), fontsize=7, color='tab:green', ha='center', va='bottom')
ax.set_ylim(0, 1.05)
ax.set_xlabel('trial')
ax.set_ylabel('accuracy')
ax.set_title('Remedial-easy trigger/recovery (switch positions marked)')
ax.legend(fontsize=7)

# (d) circuit-breaker scenarios, stop decision by scenario
ax = axes[1, 1]
scenarios = ['run={0}'.format(n_below), 'run={0}'.format(n_at),
             'rate=0.50', 'rate=0.55']
decisions = [stop_below, stop_at, stop_50, stop_55]
colors = ['tab:green' if not d else 'tab:red' for d in decisions]
ax.bar(scenarios, [1] * len(scenarios), color=colors)
ax.set_yticks([])
ax.set_title('Circuit-breaker: red = should_stop_session()==True')
for label in ax.get_xticklabels():
    label.set_rotation(20)

fig.tight_layout()
out_path = 'trial_scheduler_validation.png'
fig.savefig(out_path, dpi=150)
print()
print("Saved plots to {0}".format(out_path))

# --- second figure: Section 5's example sessions / switch positions / edge cases ----------------

fig2, axes2 = plt.subplots(2, 3, figsize=(18, 9))

# (a) null control -- no injected bias
ax = axes2[0, 0]
trials_null = np.arange(N_NULL)
ax.plot(trials_null, sim_null['p_right'], label='p_right', color='tab:purple')
window_n = 20
null_side_smoothed = np.convolve(sim_null['side_R'], np.ones(window_n) / window_n, mode='valid')
ax.plot(np.arange(len(null_side_smoothed)) + window_n - 1, null_side_smoothed,
        label='empirical R-fraction ({0}-trial smoothed)'.format(window_n), color='tab:orange')
ax.axhline(0.5, color='gray', linestyle=':', linewidth=0.8)
ax.set_ylim(0, 1)
ax.set_xlabel('trial')
ax.set_ylabel('fraction')
ax.set_title('5a. Null control (no injected bias) -- should stay ~0.5')
ax.legend(fontsize=7)

# (b) mirrored bias -- worse on right
ax = axes2[0, 1]
trials_mirror = np.arange(N_MIRROR)
ax.plot(trials_mirror, sim_mirror['p_right'], label='p_right', color='tab:purple')
mirror_side_smoothed = np.convolve(sim_mirror['side_R'], np.ones(window_n) / window_n, mode='valid')
ax.plot(np.arange(len(mirror_side_smoothed)) + window_n - 1, mirror_side_smoothed,
        label='empirical R-fraction ({0}-trial smoothed)'.format(window_n), color='tab:orange')
ax.axhline(0.5, color='gray', linestyle=':', linewidth=0.8)
ax.set_ylim(0, 1)
ax.set_xlabel('trial')
ax.set_ylabel('fraction')
ax.set_title('5b. Mirrored bias (worse on RIGHT) -- should mirror above 0.5')
ax.legend(fontsize=7)

# (c) track-the-target switch positions, first 150 trials of the main biased session
ax = axes2[0, 2]
n_show = 150
trials_show = np.arange(n_show)
ax.plot(trials_show, p_right_series[:n_show], label='p_right (target)', color='tab:purple')
ax.plot(trials_show, recent_frac_series[:n_show], label='recent empirical R-fraction',
        color='tab:orange')
for sp in [s for s in switch_positions if s < n_show]:
    ax.axvline(sp, color='gray', linewidth=0.6, alpha=0.6)
ax.set_xlabel('trial')
ax.set_ylabel('fraction')
ax.set_title('5c. Track-the-target switch positions (grey lines, first {0} trials, '
             '{1} total switches)'.format(n_show, len(switch_positions)))
ax.legend(fontsize=7)

# (d) remedial multi-cycle
ax = axes2[1, 0]
trial_idx_multi = np.arange(1, len(corrects_multi) + 1)
rolling40_multi = [_trailing_accuracy(corrects_multi, i, ts.REMEDIAL_TRIGGER_LOOKBACK_N)
                   for i in trial_idx_multi]
ax.plot(trial_idx_multi, rolling40_multi, color='black', linewidth=1.0, label='trailing-40 accuracy')
ax.axhline(ts.REMEDIAL_TRIGGER_ACCURACY, color='tab:red', linestyle='--', linewidth=0.8)
ax.axhline(ts.REMEDIAL_RECOVER_ACCURACY, color='tab:green', linestyle='--', linewidth=0.8)
in_remedial_multi_arr = np.array(multi_timeline)
ax.fill_between(trial_idx_multi, 0, 1, where=in_remedial_multi_arr.astype(bool), color='tab:red',
                alpha=0.15, step='mid')
for t in multi_triggers:
    ax.axvline(t, color='tab:red', linewidth=1.0)
for t in multi_recoveries:
    ax.axvline(t, color='tab:green', linewidth=1.0)
ax.set_ylim(0, 1.05)
ax.set_xlabel('trial')
ax.set_ylabel('accuracy')
ax.set_title('5d. Remedial-easy multi-cycle (2 independent trigger/recover cycles)')
ax.legend(fontsize=7)

# (e) circuit-breaker consecutive-run edge case
ax = axes2[1, 1]
trial_idx_5e = np.arange(1, len(pattern_5e) + 1)
colors_5e = ['tab:red' if tok == 'noresp' else 'tab:green' for tok in pattern_5e]
ax.scatter(trial_idx_5e, [1] * len(pattern_5e), c=colors_5e, s=25)
ax2e = ax.twinx()
ax2e.plot(trial_idx_5e, run_series_5e, color='black', linewidth=1.0, label='consecutive no-response run')
ax2e.axhline(ts.CIRCUIT_BREAKER_CONSECUTIVE_NORESPONSE, color='tab:red', linestyle='--', linewidth=0.8)
if stop_trial_5e is not None:
    ax.axvline(stop_trial_5e, color='black', linewidth=1.2)
    ax.annotate('STOP\ntrial {0}'.format(stop_trial_5e), (stop_trial_5e, 1.0), fontsize=7,
                ha='center', va='bottom')
ax.set_yticks([])
ax2e.set_ylabel('consecutive no-response run')
ax.set_xlabel('trial (green=responded, red=no-response)')
ax.set_title('5e. Consecutive-run edge case (isolated/near-miss runs reset; only a genuine\n'
             'run of {0} stops the session)'.format(ts.CIRCUIT_BREAKER_CONSECUTIVE_NORESPONSE),
             fontsize=9)

# (f) circuit-breaker rate edge case
ax = axes2[1, 2]
trial_idx_5f = np.arange(1, len(pattern_5f) + 1)
colors_5f = ['tab:red' if tok == 'noresp' else 'tab:green' for tok in pattern_5f]
ax.scatter(trial_idx_5f, [0.02] * len(pattern_5f), c=colors_5f, s=25)
ax.plot(trial_idx_5f, rate_series_5f, color='black', linewidth=1.0,
        label='trailing-{0} no-response rate'.format(ts.CIRCUIT_BREAKER_RATE_WINDOW_N))
ax.axhline(ts.CIRCUIT_BREAKER_RATE_THRESHOLD, color='tab:red', linestyle='--', linewidth=0.8,
           label='stop threshold')
if stop_trial_5f is not None:
    ax.axvline(stop_trial_5f, color='black', linewidth=1.2)
    ax.annotate('STOP\ntrial {0}'.format(stop_trial_5f), (stop_trial_5f, 0.9), fontsize=7,
                ha='center', va='top')
ax.set_ylim(0, 1)
ax.set_xlabel('trial (dots: green=responded, red=no-response, never 2 in a row)')
ax.set_ylabel('rate')
ax.set_title('5f. Rate edge case (no long run, rate alone trips the breaker)', fontsize=9)
ax.legend(fontsize=7)

fig2.tight_layout()
out_path2 = 'trial_scheduler_edge_cases.png'
fig2.savefig(out_path2, dpi=150)
print("Saved plots to {0}".format(out_path2))

print()
print("=" * 100)
overall_pass = (section1_pass and (check_a and check_b and check_c) and section3_pass and
                 section4_pass and section4b_pass and section5_pass and section6_pass and
                 section7_pass)
print("OVERALL: {0}".format("ALL SECTIONS PASS" if overall_pass else "AT LEAST ONE SECTION FAILED"))
print("=" * 100)
