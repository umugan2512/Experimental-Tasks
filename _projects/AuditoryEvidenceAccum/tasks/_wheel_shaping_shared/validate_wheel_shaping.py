# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Offline validation of _wheel_shaping_shared's pure-Python pieces -- no Bpod/hardware involved, same
discipline as poisson_clicks_test/validate_trial_scheduler.py. Tests:
  1. ThresholdStaircase: step-up/step-down thresholds, ceiling/floor, counter-reset-on-opposite-outcome
  2. grow_iti()/decay_gain(): rates and ceilings/floors
  3. DirectionRatioTracker: rolling window, band detection, withhold logic
  4. StageState: save/load round-tripping, defaults on first run, merge-over-defaults
  5. stage2_simple_gates_met(): the three simple Stage 2 advancement gates
  6. Session merging: classify_trial()'s consumed-reward classification, group_sessions()'s
     merge-eligible-ending/same-protocol/gap-threshold rule, combine_group()'s field combination,
     and the stage-ordering helpers
  7. QuiescenceStaircase (unlocked as early as Stage 2): step-up/step-down, floor/ceiling, same
     shape as Section 1
  8. stage4_gates_met()/stage5_gates_met()/select_stage5_staircase_to_tighten() (Tag B policy)
  9. debiasing.next_side_after_error(): statistical repeat-rate check over many draws
  10. session_csv_parser.STREAM_KEYS fix: repeated VAL rows (e.g. WHEEL_POS) are retained as a
      list, not collapsed to the last value -- the additions.txt T3 regression this guards against
  11. derive_intrial_threshold.py: sliding-window excursion percentile against a synthetic trace
      with a known injected value
  12. session_csv_parser._chain_already_used(): the live-session quota-reduction chaining logic
      (same merge-eligibility shape as group_sessions() above, but keyed on actual absence of a
      SESSION_END_REASON rather than the literal string 'completed')
  13. BoundedStaircase (Stage 3/4's shared stimulus-duration/ITI ramps) + next_alternating_value()
      (Stage 3's own same-day-unchanged/across-day-alternating tone-side rule)
  14. classify_trial()'s side = the crossing that TRIGGERED the outcome state (not the first one
      in the trial), plus TRIAL-BPOD-TIME parsing, fit_host_to_bpod(), wheel_samples_bpod_time()
      and trial_host_anchor() against a synthetic clock pair with known drift/offset
  15. TrialRunner/RotaryStream (no hardware): injected-stdin ordering, the non-blocking quiescence
      tracker, WHEEL_POS keepalive filling (no gaps across still periods / late flushes), the
      module->PC clock map's lower envelope, and exact Bpod-clock placement of 3-field WHEEL_POS rows
  16. pybpodapi's real Session under a concurrent VAL writer with install_session_lock() (the race
      that crashed session 20261023-120425 at trial 5), and struct export of 3-field WHEEL_POS rows
  17. IncrementalSessionParser (the live dashboard's parser) == parse_session_csv() on a file
      written in odd-sized chunks that split lines and a UTF-8 character across reads

Run with the pybpod-environment interpreter:
    /c/Users/2P-Behav/.conda/envs/pybpod-environment/python.exe validate_wheel_shaping.py
"""
import csv
import datetime
import os
import shutil
import sys
import tempfile

import numpy as np

import staircase
import session_state
import session_csv_parser
import debiasing
import derive_intrial_threshold
from direction_tracker import DirectionRatioTracker

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'records'))
import build_training_log   # noqa: E402 -- same-project import (_wheel_shaping_shared/ -> records/)

# ==================================================================================================
# Section 1: ThresholdStaircase
# ==================================================================================================

print("=" * 100)
print("Section 1: ThresholdStaircase")
print("=" * 100)

sc = staircase.ThresholdStaircase(current_fraction=0.20)
for _ in range(staircase.STAIRCASE_SUCCESSES_TO_STEP_UP - 1):
    sc.record_outcome(success=True)
below_step = abs(sc.current_fraction - 0.20) < 1e-9
sc.record_outcome(success=True)
stepped_up = abs(sc.current_fraction - 0.30) < 1e-9
print("  {0} consecutive successes: fraction unchanged (0.20): {1}".format(
    staircase.STAIRCASE_SUCCESSES_TO_STEP_UP - 1, below_step))
print("  {0}th consecutive success: fraction stepped up to 0.30: {1}".format(
    staircase.STAIRCASE_SUCCESSES_TO_STEP_UP, stepped_up))
reset_after_step = sc.consecutive_successes == 0
print("  consecutive_successes reset to 0 after stepping: {0}".format(reset_after_step))

sc2 = staircase.ThresholdStaircase(current_fraction=0.30)
for _ in range(staircase.STAIRCASE_FAILURES_TO_STEP_DOWN - 1):
    sc2.record_outcome(success=False)
below_step_down = abs(sc2.current_fraction - 0.30) < 1e-9
sc2.record_outcome(success=False)
stepped_down = abs(sc2.current_fraction - 0.20) < 1e-9
print("  {0} consecutive failures: fraction unchanged (0.30): {1}".format(
    staircase.STAIRCASE_FAILURES_TO_STEP_DOWN - 1, below_step_down))
print("  {0}th consecutive failure: fraction stepped down to 0.20: {1}".format(
    staircase.STAIRCASE_FAILURES_TO_STEP_DOWN, stepped_down))

sc3 = staircase.ThresholdStaircase(current_fraction=0.95)
for _ in range(staircase.STAIRCASE_SUCCESSES_TO_STEP_UP * 3):
    sc3.record_outcome(success=True)
ceiling_check = sc3.current_fraction <= 1.0 + 1e-9
print("  ceiling never exceeds 1.0 after many successes (final={0:.2f}): {1}".format(
    sc3.current_fraction, ceiling_check))

sc4 = staircase.ThresholdStaircase(current_fraction=0.5, consecutive_successes=10)
sc4.record_outcome(success=False)
reset_on_opposite = sc4.consecutive_successes == 0 and sc4.consecutive_failures == 1
print("  a single failure resets the success streak to 0 (not just decrements): {0}".format(
    reset_on_opposite))

# floor (Stage 2 passes 0.20 since withheld turns count as failures): step-downs stop there, and
# a value already below it is not raised
sc_floor = staircase.ThresholdStaircase(current_fraction=0.30, floor=0.20)
for _ in range(15):
    sc_floor.record_outcome(success=False)
sc_low = staircase.ThresholdStaircase(current_fraction=0.10, floor=0.20)
for _ in range(5):
    sc_low.record_outcome(success=False)
floor_ok = abs(sc_floor.current_fraction - 0.20) < 1e-9 and abs(sc_low.current_fraction - 0.10) < 1e-9
print("  floor=0.20: 15 failures from 0.30 stop at 0.20; a value below the floor stays: {0}".format(floor_ok))
# record_neutral() (Stage 2's NoMovement since 2026-11-05): blocks a step-up, never steps down,
# and leaves a running failure streak alone
sc_n1 = staircase.ThresholdStaircase(current_fraction=0.20)
for _ in range(staircase.STAIRCASE_SUCCESSES_TO_STEP_UP - 1):
    sc_n1.record_outcome(success=True)
sc_n1.record_neutral()
sc_n1.record_outcome(success=True)
sc_n2 = staircase.ThresholdStaircase(current_fraction=0.30)
for _ in range(staircase.STAIRCASE_FAILURES_TO_STEP_DOWN - 1):
    sc_n2.record_outcome(success=False)
sc_n2.record_neutral()
sc_n2.record_outcome(success=False)
sc_n3 = staircase.ThresholdStaircase(current_fraction=0.30, floor=0.20)
for _ in range(10):
    sc_n3.record_neutral()
neutral_ok = (sc_n1.current_fraction == 0.20 and sc_n1.consecutive_successes == 1
              and sc_n2.current_fraction == 0.20 and sc_n3.current_fraction == 0.30
              and sc_n3.consecutive_failures == 0)
print("  record_neutral: no step-up across it, failure streak survives it, x10 changes nothing: {0}".format(
    neutral_ok))
# steps are rounded, so the fraction (and the logged degrees) carry no float noise
sc_r = staircase.ThresholdStaircase(current_fraction=0.20)
for _ in range(staircase.STAIRCASE_SUCCESSES_TO_STEP_UP):
    sc_r.record_outcome(success=True)
for _ in range(staircase.STAIRCASE_FAILURES_TO_STEP_DOWN):
    sc_r.record_outcome(success=False)
rounded_ok = sc_r.current_fraction == 0.2 and repr(round(sc_r.current_fraction * 35, 3)) == '7.0'
print("  0.20 -> 0.30 -> 0.20 lands on exactly 0.2 (7.0deg logged): {0} ({1!r})".format(
    rounded_ok, sc_r.current_fraction))
section1_pass = (floor_ok and below_step and stepped_up and reset_after_step and below_step_down and
                  stepped_down and ceiling_check and reset_on_opposite and neutral_ok and rounded_ok)
print("Section 1 result: {0}".format("PASS" if section1_pass else "FAIL -- see checks above"))

# ==================================================================================================
# Section 2: grow_iti() / decay_gain() / grow_stage1_threshold()
# ==================================================================================================

print()
print("=" * 100)
print("Section 2: grow_iti() / decay_gain() / grow_stage1_threshold()")
print("=" * 100)

iti_after_9 = 0.5
for _ in range(9):
    iti_after_9 = staircase.grow_iti(iti_after_9)
iti_before_ceiling = abs(iti_after_9 - 1.4) < 1e-9
iti_after_10 = staircase.grow_iti(iti_after_9)
iti_at_ceiling = abs(iti_after_10 - 1.5) < 1e-9
iti_past_ceiling = staircase.grow_iti(iti_after_10)
iti_stays_at_ceiling = abs(iti_past_ceiling - 1.5) < 1e-9
print("  ITI after 9 sessions from 0.5s (expect 1.4s): {0:.2f} -- {1}".format(
    iti_after_9, iti_before_ceiling))
print("  ITI after 10th session (expect ceiling 1.5s): {0:.2f} -- {1}".format(
    iti_after_10, iti_at_ceiling))
print("  ITI never exceeds the 1.5s ceiling on further growth: {0}".format(
    iti_stays_at_ceiling))

gain_after_9 = 3.0
for _ in range(9):
    gain_after_9 = staircase.decay_gain(gain_after_9)
gain_before_floor = abs(gain_after_9 - 2.1) < 1e-9
gain_after_10 = staircase.decay_gain(gain_after_9)
gain_at_floor = abs(gain_after_10 - staircase.GAIN_FLOOR_MULT) < 1e-9
gain_past_floor = staircase.decay_gain(gain_after_10)
gain_stays_at_floor = abs(gain_past_floor - staircase.GAIN_FLOOR_MULT) < 1e-9
print("  Gain after 9 sessions from 3.0x (expect 2.1x): {0:.2f} -- {1}".format(
    gain_after_9, gain_before_floor))
print("  Gain after 10th session (expect floor {0:.1f}x): {1:.2f} -- {2}".format(
    staircase.GAIN_FLOOR_MULT, gain_after_10, gain_at_floor))
print("  Gain never decays below the {0:.1f}x floor on further sessions: {1}".format(
    staircase.GAIN_FLOOR_MULT, gain_stays_at_floor))

# Stage 1's threshold growth -- session-level steps (NOT ThresholdStaircase's within-session,
# per-trial mechanism), same shape as ITI growth above.
threshold_after_7 = 0.05
for _ in range(7):
    threshold_after_7 = staircase.grow_stage1_threshold(threshold_after_7)
threshold_before_ceiling = abs(threshold_after_7 - 0.19) < 1e-9
threshold_after_8 = staircase.grow_stage1_threshold(threshold_after_7)
threshold_at_ceiling = abs(threshold_after_8 - 0.20) < 1e-9
threshold_past_ceiling = staircase.grow_stage1_threshold(threshold_after_8)
threshold_stays_at_ceiling = abs(threshold_past_ceiling - 0.20) < 1e-9
print("  Stage 1 threshold after 7 sessions from 0.05 (expect 0.19): {0:.2f} -- {1}".format(
    threshold_after_7, threshold_before_ceiling))
print("  Stage 1 threshold after 8th session (expect ceiling 0.20, Stage 2's own start): "
      "{0:.2f} -- {1}".format(threshold_after_8, threshold_at_ceiling))
print("  Stage 1 threshold never exceeds the 0.20 ceiling on further growth: {0}".format(
    threshold_stays_at_ceiling))

# session-level steps are rounded: no 0.9999999999999999 / 2.3999999999999995 in the state file
steps_rounded = (staircase.grow_iti(0.9) == 1.0 and staircase.decay_gain(2.5) == 2.4
                 and staircase.grow_stage1_threshold(0.16) == 0.18)
print("  grow_iti(0.9) == 1.0, decay_gain(2.5) == 2.4, grow_stage1_threshold(0.16) == 0.18 exactly: {0}".format(
    steps_rounded))
section2_pass = steps_rounded and (iti_before_ceiling and iti_at_ceiling and iti_stays_at_ceiling and
                  gain_before_floor and gain_at_floor and gain_stays_at_floor and
                  threshold_before_ceiling and threshold_at_ceiling and threshold_stays_at_ceiling)
print("Section 2 result: {0}".format("PASS" if section2_pass else "FAIL -- see checks above"))

# ==================================================================================================
# Section 3: DirectionRatioTracker
# ==================================================================================================

print()
print("=" * 100)
print("Section 3: DirectionRatioTracker")
print("=" * 100)

dt = DirectionRatioTracker()
neutral_start = dt.in_band() and abs(dt.right_fraction() - 0.5) < 1e-9
print("  neutral before any trials (0.5, in-band): {0}".format(neutral_start))

for _ in range(35):
    dt.record('R')
for _ in range(5):
    dt.record('L')
# weighted: the 5 most recent L turns outweigh the 35 older R turns, so R is no longer withheld
# (the flat ratio, 0.88, would still withhold R)
recent_l_counts = (dt.in_band() and not dt.should_withhold('R')
                   and dt.right_fraction() < 0.6 and dt.plain_right_fraction() > 0.85)
print("  35R then 5L: weighted ratio {0:.2f} follows the recent L turns, flat ratio {1:.2f}, R not "
      "withheld: {2}".format(dt.right_fraction(), dt.plain_right_fraction(), recent_l_counts))
n_more_l = 0
while not dt.should_withhold('L') and n_more_l < 40:
    dt.record('L'); n_more_l += 1
over_used_left = n_more_l <= 6 and dt.over_used_side() == 'L' and not dt.should_withhold('R')
print("  ...L over-used after {0} more L turns (<= 6), withhold L only: {1}".format(
    n_more_l, over_used_left))

dt2 = DirectionRatioTracker()
for i in range(40):
    dt2.record('R' if i % 2 == 0 else 'L')
balanced_in_band = dt2.in_band() and dt2.over_used_side() is None
print("  perfectly alternating 40 trials: in_band=True, over_used_side=None: {0}".format(
    balanced_in_band))

dt3 = DirectionRatioTracker(window=40)
for _ in range(50):
    dt3.record('R')
window_check = len(dt3._sides) == 40 and dt3.should_withhold('R')
print("  rolling window caps at 40 entries after 50 records: {0}".format(window_check))

# A run from a balanced history is withheld within a few turns, and a few opposite turns end it
# (20261102-161142: the flat 40-trial ratio let runs of ~16 through, and needed ~11 to recover).
dt_run = DirectionRatioTracker(history=['R', 'L'] * 20)
n_run = 0
while not dt_run.should_withhold('R'):
    dt_run.record('R'); n_run += 1
n_back = 0
while dt_run.should_withhold('R'):
    dt_run.record('L'); n_back += 1
run_ok = n_run <= 6 and n_back <= 3
print("  balanced history: R withheld after {0} R turns, back in band after {1} L turns "
      "(<= 6 / <= 3): {2}".format(n_run, n_back, run_ok))

# Before DIRECTION_RATIO_MIN_TRIALS directions are known, even an extreme ratio must NOT trigger
# withholding -- OUTCOME=Withheld fired as early as trial 2 when there was no gate at all.
dt4 = DirectionRatioTracker()
dt4.record('L')
one_trial_not_enough = not dt4.has_enough_trials() and not dt4.has_full_window()
one_trial_in_band = dt4.in_band() and not dt4.should_withhold('L') and not dt4.should_withhold('R')
print("  1 trial recorded: has_enough_trials()=False: {0}".format(one_trial_not_enough))
print("  1 trial, all-'L': in_band=True, no withholding on either side: {0}".format(one_trial_in_band))

# Fresh window (no earlier sessions): withholding starts after DIRECTION_RATIO_MIN_TRIALS (10).
dt5 = DirectionRatioTracker()
for _ in range(9):
    dt5.record('L')
still_no_withhold_at_9 = not dt5.has_enough_trials() and not dt5.should_withhold('L')
print("  fresh window, 9 trials all-'L': still no withholding: {0}".format(still_no_withhold_at_9))
dt5.record('L')
withholds_at_10 = dt5.has_enough_trials() and dt5.should_withhold('L')
print("  10th trial: has_enough_trials()=True, NOW withholds 'L': {0}".format(withholds_at_10))

# Seeded from earlier sessions: withholding is live from this session's first trial.
seed = ['R'] * 5 + ['L'] * 40          # 45 entries, oldest first; only the last 40 are kept
dt6 = DirectionRatioTracker(history=seed)
seeded_truncated = dt6.seeded_trials == 40 and dt6.history() == ['L'] * 40
seeded_withholds_first_trial = dt6.should_withhold('L') and not dt6.should_withhold('R')
print("  seeded with 45 earlier directions: keeps the last 40: {0}".format(seeded_truncated))
print("  seeded all-'L' window: withholds 'L' on trial 1: {0}".format(seeded_withholds_first_trial))
dt8 = DirectionRatioTracker(history=['L', 'None', None, 'R', 'x'])
seed_filters_junk = dt8.history() == ['L', 'R'] and DirectionRatioTracker(history=None).seeded_trials == 0
print("  seeding ignores anything but 'L'/'R' (and None history): {0}".format(seed_filters_junk))
import json as _json
dt9 = DirectionRatioTracker(history=_json.loads(_json.dumps(dt_run.history())))
seed_round_trip = dt9.history() == dt_run.history() and dt9.right_fraction() == dt_run.right_fraction()
print("  history() survives a JSON round trip: {0}".format(seed_round_trip))

section3_pass = (neutral_start and recent_l_counts and over_used_left and balanced_in_band and
                  window_check and run_ok and one_trial_not_enough and one_trial_in_band and
                  still_no_withhold_at_9 and withholds_at_10 and seeded_truncated and
                  seeded_withholds_first_trial and seed_filters_junk and seed_round_trip)
print("Section 3 result: {0}".format("PASS" if section3_pass else "FAIL -- see checks above"))

# ==================================================================================================
# Section 4: StageState
# ==================================================================================================

print()
print("=" * 100)
print("Section 4: StageState (save/load round-trip)")
print("=" * 100)

tmp_dir = tempfile.mkdtemp(prefix='wheel_shaping_state_test_')
try:
    defaults = {'gain_mult': 2.0, 'sessions_trial_count_history': []}

    s1 = session_state.StageState(tmp_dir, 'test_subject', defaults)
    first_run_defaults = s1.get('gain_mult') == 2.0 and s1.get('sessions_trial_count_history') == []
    print("  first-ever run gets defaults (no file on disk yet): {0}".format(first_run_defaults))

    s1.set('gain_mult', 1.62)
    s1.set('sessions_trial_count_history', [True, True])
    s1.save()
    file_created = os.path.exists(session_state._state_path(tmp_dir, 'test_subject'))
    print("  save() creates the state file: {0}".format(file_created))

    s2 = session_state.StageState(tmp_dir, 'test_subject', defaults)
    round_trip_ok = (s2.get('gain_mult') == 1.62 and
                      s2.get('sessions_trial_count_history') == [True, True])
    print("  a fresh StageState loads the saved values back: {0}".format(round_trip_ok))

    # A future code change adding a new default key shouldn't break loading an older, smaller file.
    defaults_with_new_key = dict(defaults)
    defaults_with_new_key['spout_position_mm'] = 0.0
    s3 = session_state.StageState(tmp_dir, 'test_subject', defaults_with_new_key)
    new_key_defaulted = s3.get('spout_position_mm') == 0.0
    old_keys_preserved = s3.get('gain_mult') == 1.62
    print("  loading an older file with a new default key added: new key defaults ({0}), old "
          "keys preserved ({1})".format(new_key_defaulted, old_keys_preserved))

    section4_pass = (first_run_defaults and file_created and round_trip_ok and
                      new_key_defaulted and old_keys_preserved)
finally:
    shutil.rmtree(tmp_dir, ignore_errors=True)

print("Section 4 result: {0}".format("PASS" if section4_pass else "FAIL -- see checks above"))

# ==================================================================================================
# Section 5: stage2_simple_gates_met()
# ==================================================================================================

print()
print("=" * 100)
print("Section 5: stage2_simple_gates_met()")
print("=" * 100)

all_met = staircase.stage2_simple_gates_met(trial_count=201, iti_s=1.5, direction_ratio_in_band=True)
print("  201 trials, ITI=1.5s, in-band: expect True: {0}".format(all_met))

exactly_at_cap = staircase.stage2_simple_gates_met(
    trial_count=200, iti_s=1.5, direction_ratio_in_band=True)
print("  exactly 200 trials (needs >=, since a session is now capped AT 200): expect True: "
      "{0}".format(exactly_at_cap))

trial_count_fails = staircase.stage2_simple_gates_met(
    trial_count=199, iti_s=1.5, direction_ratio_in_band=True)
print("  199 trials: expect False: {0}".format(trial_count_fails is False))

iti_fails = staircase.stage2_simple_gates_met(
    trial_count=250, iti_s=1.4, direction_ratio_in_band=True)
print("  ITI not yet at ceiling (1.4s): expect False: {0}".format(iti_fails is False))

ratio_fails = staircase.stage2_simple_gates_met(
    trial_count=250, iti_s=1.5, direction_ratio_in_band=False)
print("  direction ratio out of band: expect False: {0}".format(ratio_fails is False))

section5_pass = (all_met and exactly_at_cap and trial_count_fails is False and
                  iti_fails is False and ratio_fails is False)
print("Section 5 result: {0}".format("PASS" if section5_pass else "FAIL -- see checks above"))

# ==================================================================================================
# Section 6: Session merging -- classify_trial() consumed, group_sessions(), combine_group(),
# stage-ordering helpers
# ==================================================================================================

print()
print("=" * 100)
print("Section 6: Session merging")
print("=" * 100)


def _trial(states, events, vals=None):
    return {'states': states, 'events': events, 'vals': vals or {}}


config1 = session_csv_parser.PROTOCOL_CONFIG['stage1_wheel_shaping']

consumed_trial = _trial({'Reward': (1.0, 1.1, 0.1)}, {'Port1In': [1.05]})
outcome, _side, _dur, consumed = session_csv_parser.classify_trial(consumed_trial, config1)
consumed_true = outcome == 'rewarded' and consumed is True
print("  lick AFTER reward delivery -> consumed=True: {0}".format(consumed_true))

not_consumed_trial = _trial({'Reward': (1.0, 1.1, 0.1)}, {'Port1In': [0.5]})
_o, _s, _d, consumed2 = session_csv_parser.classify_trial(not_consumed_trial, config1)
consumed_false = consumed2 is False
print("  lick only BEFORE reward delivery -> consumed=False: {0}".format(consumed_false))

no_lick_trial = _trial({'Reward': (1.0, 1.1, 0.1)}, {})
_o, _s, _d, consumed3 = session_csv_parser.classify_trial(no_lick_trial, config1)
consumed_no_lick = consumed3 is False
print("  no lick at all -> consumed=False: {0}".format(consumed_no_lick))

not_rewarded_trial = _trial({'NoMovement': (1.0, 1.1, 0.1)}, {'Port1In': [1.05]})
_o, _s, _d, consumed4 = session_csv_parser.classify_trial(not_rewarded_trial, config1)
consumed_none_when_not_rewarded = consumed4 is None
print("  non-rewarded trial -> consumed=None: {0}".format(consumed_none_when_not_rewarded))

section6a_pass = (consumed_true and consumed_false and consumed_no_lick and
                   consumed_none_when_not_rewarded)


def _fake_session(started, protocol, session_end_reason, trial_count=10, duration_s=10.0,
                   reward_ul=4.0, consumed_volume_ul=8.0, csv_path=None):
    started_dt = datetime.datetime.strptime(started, '%Y-%m-%d %H:%M:%S')
    ended_dt = started_dt + datetime.timedelta(seconds=duration_s)
    return {
        'session_started': started, 'started_dt': started_dt, 'ended_dt': ended_dt,
        'session_end_reason': session_end_reason, 'subject': 'FixtureMouse', 'date': started[:10],
        'time_of_day': started_dt.strftime('%H:%M'),
        'protocol': protocol, 'trial_count': trial_count, 'session_duration_s': duration_s,
        'reward_count': 2, 'consumed_reward_count': 2, 'withheld_count': 0,
        'no_movement_count': 0, 'incorrect_count': 0, 'wheel_abort_count': 0,
        'l_count': 3, 'r_count': 4, 'lick_count': 5, 'aborts': 1,
        'reward_ul': reward_ul, 'consumed_volume_ul': consumed_volume_ul,
        'threshold_start_deg': 7.0, 'threshold_end_deg': 7.0, 'threshold_final_deg': 35.0,
        'iti_end_s': 0.5, 'gain_mult_end': 2.0, 'direction_ratio_end': None,
        'simple_gates_met': None,
        # Stage 4/5 fields -- None here since this fixture models a Stage 1/2 session (matches
        # what a real summarize_session() call returns for those protocols: the keys always
        # exist, populated only when the protocol's own PROTOCOL_CONFIG/VAL registrations apply).
        'response_threshold_start_deg': None, 'response_threshold_end_deg': None,
        'quiescence_start_s': None, 'quiescence_end_s': None, 'cue_abort_threshold_deg': None,
        'click_level_attenuated': None, 'accuracy_aos': None, 'abort_rate': None,
        'aborts_quiescence': 0, 'aborts_cue': 0, 'aborts_delay': 0, 'aborts_response': 0,
        'warmup_trial_count': 0, 'repeat_trial_count': 0, 'session_water_ul': None,
        'staircase_advanced_this_session': None, 'git_commit': None,
        'target_spl_db': None, 'left_freq_hz': None, 'right_freq_hz': None,
        'session_csv_path': csv_path or (started.replace(' ', '_').replace(':', '') + '.csv'),
        'session_struct_path': None,
    }


# (a) two same-protocol sessions 5 minutes apart, first NOT completed -> merge
sA1 = _fake_session('2026-01-01 10:00:00', 'stage1_wheel_shaping', None, duration_s=300)
sA2 = _fake_session('2026-01-01 10:05:00', 'stage1_wheel_shaping', None, duration_s=300)
groups_a = build_training_log.group_sessions([sA1, sA2])
merge_when_not_completed = len(groups_a) == 1 and len(groups_a[0]) == 2
print("  (a) 5min gap, first NOT completed -> merges into 1 group: {0}".format(
    merge_when_not_completed))

# (b) same gap, but first session completed fully -> must NOT merge
sB1 = _fake_session('2026-01-01 10:00:00', 'stage1_wheel_shaping', 'completed', duration_s=300)
sB2 = _fake_session('2026-01-01 10:05:00', 'stage1_wheel_shaping', None, duration_s=300)
groups_b = build_training_log.group_sessions([sB1, sB2])
no_merge_when_completed = len(groups_b) == 2
print("  (b) 5min gap, first completed fully -> stays 2 separate groups: {0}".format(
    no_merge_when_completed))

# (c) 90 minutes apart, first not completed -> gap too large, must NOT merge
sC1 = _fake_session('2026-01-01 10:00:00', 'stage1_wheel_shaping', None, duration_s=300)
sC2 = _fake_session('2026-01-01 11:35:00', 'stage1_wheel_shaping', None, duration_s=300)
groups_c = build_training_log.group_sessions([sC1, sC2])
no_merge_when_gap_too_large = len(groups_c) == 2
print("  (c) 90min gap, first not completed -> gap too large, stays 2 separate groups: {0}".format(
    no_merge_when_gap_too_large))

# different protocol, otherwise mergeable -> must NOT merge
sD1 = _fake_session('2026-01-01 10:00:00', 'stage1_wheel_shaping', None, duration_s=300)
sD2 = _fake_session('2026-01-01 10:05:00', 'stage2_threshold_staircase', None, duration_s=300)
groups_d = build_training_log.group_sessions([sD1, sD2])
no_merge_across_protocols = len(groups_d) == 2
print("  different protocol, 5min gap, first not completed -> stays 2 separate groups: {0}".format(
    no_merge_across_protocols))

# merging applies uniformly regardless of subject name -- confirmed via _fake_session's own
# 'FixtureMouse' subject above; there's no more per-subject exclusion to special-case (removed
# per explicit instruction: bench-test sessions merge exactly like a real animal's would now).
section6b_pass = (merge_when_not_completed and no_merge_when_completed and
                   no_merge_when_gap_too_large and no_merge_across_protocols)

# combine_group() field combination on the (a) merge
combined = build_training_log.combine_group(groups_a[0])
combined_trials = combined['trial_count'] == sA1['trial_count'] + sA2['trial_count']
combined_duration = abs(combined['session_duration_s'] - (sA1['session_duration_s'] +
                                                            sA2['session_duration_s'])) < 1e-9
combined_volume = abs(combined['consumed_volume_ul'] - (sA1['consumed_volume_ul'] +
                                                          sA2['consumed_volume_ul'])) < 1e-9
combined_num_sessions = combined['num_sessions'] == 2
combined_key = combined['session_started'] == '{0}; {1}'.format(sA1['session_started'],
                                                                  sA2['session_started'])
print("  combine_group(): trial_count summed: {0}".format(combined_trials))
print("  combine_group(): session_duration_s summed (NOT wall-clock span, no gap double-counted): "
      "{0}".format(combined_duration))
print("  combine_group(): consumed_volume_ul summed: {0}".format(combined_volume))
print("  combine_group(): num_sessions=2: {0}".format(combined_num_sessions))
print("  combine_group(): session_started is '; '-joined member keys: {0}".format(combined_key))

# a singleton group's Duration must be UNCHANGED from that one session's own value (regression
# check -- combine_group() must not silently switch to a wall-clock-span calculation that would
# include Bpod connection/setup overhead for the common unmerged case)
singleton = build_training_log.combine_group([sA1])
singleton_duration_unchanged = abs(singleton['session_duration_s'] -
                                    sA1['session_duration_s']) < 1e-9
print("  combine_group() on a singleton group: Duration unchanged from that session's own value: "
      "{0}".format(singleton_duration_unchanged))

section6c_pass = (combined_trials and combined_duration and combined_volume and
                   combined_num_sessions and combined_key and singleton_duration_unchanged)

# stage-ordering helpers
stage_order = sorted(['stage2_threshold_staircase', 'stage1_wheel_shaping', 'stage10_future'],
                      key=build_training_log._stage_sort_key)
stage_order_correct = stage_order == ['stage1_wheel_shaping', 'stage2_threshold_staircase',
                                       'stage10_future']
stage_labels_correct = (build_training_log._stage_label('stage2_threshold_staircase') == 'Stage 2'
                         and build_training_log._stage_label('some_other_protocol') ==
                         'some_other_protocol')
print("  stage ordering is numeric (1, 2, 10), not alphabetical (1, 10, 2): {0}".format(
    stage_order_correct))
print("  stage labels: 'Stage N' for numbered protocols, passthrough otherwise: {0}".format(
    stage_labels_correct))

section6d_pass = stage_order_correct and stage_labels_correct

# merge_session_structs(): every combined trial names its member session; members stay untouched
import json as _json6
import session_struct_export
tmp_merge_dir = tempfile.mkdtemp(prefix='merge_struct_test_')
try:
    merge_csvs = []
    for name, n in (('20261104-162439', 3), ('20261104-165242', 2)):
        folder = os.path.join(tmp_merge_dir, name)
        os.makedirs(folder)
        with open(os.path.join(folder, name + '_struct.json'), 'w') as f:
            _json6.dump({'info': {}, 'task_params': {},
                         'trials': [{'bpod_start': float(k)} for k in range(n)]}, f)
        merge_csvs.append(os.path.join(folder, name + '.csv'))
    _mat, merge_json = session_struct_export.merge_session_structs(
        merge_csvs, os.path.join(tmp_merge_dir, 'combined'))
    with open(merge_json) as f:
        merged = _json6.load(f)
finally:
    shutil.rmtree(tmp_merge_dir, ignore_errors=True)
merge_labels_ok = ([(t['session_member'], t['session_name']) for t in merged['trials']] ==
                   [(1, '20261104-162439')] * 3 + [(2, '20261104-165242')] * 2
                   and all('session_member' not in t for m in merged['members'] for t in m['trials']))
print("  merged struct: each trial labelled with its member (1-based) + session, members unlabelled: "
      "{0}".format(merge_labels_ok))

section6_pass = (section6a_pass and section6b_pass and section6c_pass and section6d_pass
                 and merge_labels_ok)
print("Section 6 result: {0}".format("PASS" if section6_pass else "FAIL -- see checks above"))

# ==================================================================================================
# Section 7: QuiescenceStaircase (unlocked as early as Stage 2, continuous through Stage 5) --
# same step-up/step-down/floor/ceiling shape as Section 1's ThresholdStaircase
# ==================================================================================================

print()
print("=" * 100)
print("Section 7: QuiescenceStaircase")
print("=" * 100)

qs = staircase.QuiescenceStaircase(current_s=0.1)
for _ in range(staircase.QUIESCENCE_SUCCESSES_TO_STEP_UP - 1):
    qs.record_outcome(success=True)
qs_below_step = abs(qs.current_s - 0.1) < 1e-9
qs.record_outcome(success=True)
qs_stepped_up = abs(qs.current_s - (0.1 + staircase.QUIESCENCE_STEP_UP_S)) < 1e-9
print("  {0} consecutive successful initiations: scale unchanged (0.1s): {1}".format(
    staircase.QUIESCENCE_SUCCESSES_TO_STEP_UP - 1, qs_below_step))
print("  {0}th consecutive success: scale stepped up by {1}s: {2}".format(
    staircase.QUIESCENCE_SUCCESSES_TO_STEP_UP, staircase.QUIESCENCE_STEP_UP_S, qs_stepped_up))

qs2 = staircase.QuiescenceStaircase(current_s=staircase.QUIESCENCE_CEILING_S)
for _ in range(staircase.QUIESCENCE_SUCCESSES_TO_STEP_UP * 3):
    qs2.record_outcome(success=True)
qs_ceiling_check = qs2.current_s <= staircase.QUIESCENCE_CEILING_S + 1e-9
print("  scale never exceeds the {0}s ceiling after many successes (final={1:.3f}s): {2}".format(
    staircase.QUIESCENCE_CEILING_S, qs2.current_s, qs_ceiling_check))

qs3 = staircase.QuiescenceStaircase(current_s=staircase.QUIESCENCE_FLOOR_S)
for _ in range(staircase.QUIESCENCE_FAILURES_TO_STEP_DOWN * 3):
    qs3.record_outcome(success=False)
qs_floor_check = qs3.current_s >= staircase.QUIESCENCE_FLOOR_S - 1e-9
print("  scale never drops below the {0}s floor after many resets (final={1:.3f}s): {2}".format(
    staircase.QUIESCENCE_FLOOR_S, qs3.current_s, qs_floor_check))

section7_pass = qs_below_step and qs_stepped_up and qs_ceiling_check and qs_floor_check
print("Section 7 result: {0}".format("PASS" if section7_pass else "FAIL -- see checks above"))

# ==================================================================================================
# Section 8: stage4_gates_met() / stage5_gates_met() / select_stage5_staircase_to_tighten()
# (formerly stage3_gates_met()/stage4_gates_met()/select_stage4_staircase_to_tighten(), before a
# new Stage 3 was inserted)
# ==================================================================================================

print()
print("=" * 100)
print("Section 8: Stage 4/5 gates + Tag B staircase selection")
print("=" * 100)

s3_pass = staircase.stage4_gates_met(accuracy_aos=0.75, abort_rate=0.10, trial_count=250)
s3_fail_acc = staircase.stage4_gates_met(accuracy_aos=0.60, abort_rate=0.10, trial_count=250)
s3_fail_abort = staircase.stage4_gates_met(accuracy_aos=0.75, abort_rate=0.25, trial_count=250)
s3_fail_trials = staircase.stage4_gates_met(accuracy_aos=0.75, abort_rate=0.10, trial_count=150)
print("  Stage 4 gates: all met (75% acc, 10% abort, 250 trials) -> True: {0}".format(s3_pass))
print("  Stage 4 gates: accuracy below 70% -> False: {0}".format(s3_fail_acc is False))
print("  Stage 4 gates: abort rate above 20% -> False: {0}".format(s3_fail_abort is False))
print("  Stage 4 gates: trial count below 200 -> False: {0}".format(s3_fail_trials is False))
section8a_pass = s3_pass and s3_fail_acc is False and s3_fail_abort is False and s3_fail_trials is False

s4_pass = staircase.stage5_gates_met(
    response_threshold_fraction=0.95, quiescence_s=staircase.QUIESCENCE_CEILING_S,
    accuracy_aos=0.75, abort_rate=0.10, trial_count=250)
s4_fail_resp = staircase.stage5_gates_met(
    response_threshold_fraction=0.5, quiescence_s=staircase.QUIESCENCE_CEILING_S,
    accuracy_aos=0.75, abort_rate=0.10, trial_count=250)
s4_fail_quiescence = staircase.stage5_gates_met(
    response_threshold_fraction=0.95, quiescence_s=0.2,
    accuracy_aos=0.75, abort_rate=0.10, trial_count=250)
print("  Stage 5 gates: both staircases at target -> True: {0}".format(s4_pass))
print("  Stage 5 gates: response threshold not at target -> False: {0}".format(
    s4_fail_resp is False))
print("  Stage 5 gates: quiescence not at ceiling -> False: {0}".format(
    s4_fail_quiescence is False))
section8b_pass = s4_pass and s4_fail_resp is False and s4_fail_quiescence is False

# Tag B: whichever parameter is furthest from its own final value wins; ties alternate.
pick_response = staircase.select_stage5_staircase_to_tighten(
    response_threshold_fraction=0.3, quiescence_s=0.45, last_advanced=None)
pick_quiescence = staircase.select_stage5_staircase_to_tighten(
    response_threshold_fraction=0.85, quiescence_s=0.15, last_advanced=None)
print("  Tag B: response furthest from target -> picks 'response': {0}".format(
    pick_response == 'response'))
print("  Tag B: quiescence furthest from target -> picks 'quiescence': {0}".format(
    pick_quiescence == 'quiescence'))

# A genuine tie (both staircases equally far from final, on the SAME normalized 0-1 scale this
# function itself uses) -- tie-break alternates away from last_advanced.
tie_resp_frac = 0.5 * staircase.STAGE5_RESPONSE_THRESHOLD_FINAL_FRACTION
tie_quiescence_s = staircase.QUIESCENCE_CEILING_S - 0.5 * (
    staircase.QUIESCENCE_CEILING_S - staircase.QUIESCENCE_FLOOR_S)
tie_break_from_response = staircase.select_stage5_staircase_to_tighten(
    tie_resp_frac, tie_quiescence_s, last_advanced='response')
tie_break_from_quiescence = staircase.select_stage5_staircase_to_tighten(
    tie_resp_frac, tie_quiescence_s, last_advanced='quiescence')
tie_break_correct = (tie_break_from_response == 'quiescence' and
                      tie_break_from_quiescence == 'response')
print("  Tag B: exact tie alternates away from last_advanced: {0}".format(tie_break_correct))

section8c_pass = (pick_response == 'response' and pick_quiescence == 'quiescence' and
                   tie_break_correct)
section8_pass = section8a_pass and section8b_pass and section8c_pass
print("Section 8 result: {0}".format("PASS" if section8_pass else "FAIL -- see checks above"))

# ==================================================================================================
# Section 9: debiasing.next_side_after_error() -- statistical repeat-rate check
# ==================================================================================================

print()
print("=" * 100)
print("Section 9: debiasing.next_side_after_error()")
print("=" * 100)

rng = np.random.RandomState(12345)
n_draws = 20000
repeats = sum(1 for _ in range(n_draws)
              if debiasing.next_side_after_error('L', rng=rng) == 'L')
realized_rate = repeats / float(n_draws)
rate_within_tol = abs(realized_rate - debiasing.VAR_DEBIAS_REPEAT_PROB) < 0.02
print("  Realized repeat rate over {0} draws (target {1:.2f}): {2:.4f} -- within 2%: {3}".format(
    n_draws, debiasing.VAR_DEBIAS_REPEAT_PROB, realized_rate, rate_within_tol))

flip_only = all(debiasing.next_side_after_error('R', rng=rng, repeat_prob=0.0) == 'L'
                 for _ in range(50))
repeat_only = all(debiasing.next_side_after_error('R', rng=rng, repeat_prob=1.0) == 'R'
                   for _ in range(50))
print("  repeat_prob=0.0 always flips: {0}".format(flip_only))
print("  repeat_prob=1.0 always repeats: {0}".format(repeat_only))

section9_pass = rate_within_tol and flip_only and repeat_only
print("Section 9 result: {0}".format("PASS" if section9_pass else "FAIL -- see checks above"))

# ==================================================================================================
# Section 10: session_csv_parser.STREAM_KEYS fix -- repeated VAL rows retained as a list
# ==================================================================================================

print()
print("=" * 100)
print("Section 10: session_csv_parser STREAM_KEYS fix")
print("=" * 100)

tmp_csv_dir = tempfile.mkdtemp(prefix='stream_keys_test_')
try:
    csv_path = os.path.join(tmp_csv_dir, 'synth_session.csv')
    rows = [
        ['INFO', '2026-01-01 00:00:00.000000', '', '', 'SUBJECT-NAME', "['t', 'x']"],
        ['TRIAL', '', '', '', ''],
        ['STATE', '', '0.0', '0.5', 'WheelPeriod', '0.5'],
        ['VAL', '', '', '', 'WHEEL_POS', '0.100,1.20'],
        ['VAL', '', '', '', 'WHEEL_POS', '0.200,1.40'],
        ['VAL', '', '', '', 'WHEEL_POS', '0.300,1.60'],
        ['VAL', '', '', '', 'QUIESCENCE_RESET_TIME', '0.150'],
        ['VAL', '', '', '', 'QUIESCENCE_RESET_TIME', '0.280'],
        ['VAL', '', '', '', 'TRIAL_START', '0.05'],   # NOT a stream key -- scalar, last-value-wins
        ['VAL', '', '', '', 'TRIAL_START', '0.06'],   # (shouldn't normally repeat, but confirms
                                                        # non-stream keys still collapse correctly)
    ]
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f, delimiter=';').writerows(rows)

    _info, _session_vals, trials = session_csv_parser.parse_session_csv(csv_path)
    wheel_pos_all_retained = len(trials[0]['vals']['WHEEL_POS']) == 3
    quiescence_all_retained = len(trials[0]['vals']['QUIESCENCE_RESET_TIME']) == 2
    scalar_still_collapses = trials[0]['vals']['TRIAL_START'] == '0.06'
    print("  3x repeated WHEEL_POS rows all retained (not collapsed to the last): {0}".format(
        wheel_pos_all_retained))
    print("  2x repeated QUIESCENCE_RESET_TIME rows all retained: {0}".format(
        quiescence_all_retained))
    print("  a NON-stream-key repeated VAL still collapses to its last value: {0}".format(
        scalar_still_collapses))

    section10_pass = (wheel_pos_all_retained and quiescence_all_retained and
                       scalar_still_collapses)
finally:
    shutil.rmtree(tmp_csv_dir, ignore_errors=True)

print("Section 10 result: {0}".format("PASS" if section10_pass else "FAIL -- see checks above"))

# ==================================================================================================
# Section 11: derive_intrial_threshold.py -- sliding-window excursion percentile against a
# synthetic trace with a KNOWN injected excursion
# ==================================================================================================

print()
print("=" * 100)
print("Section 11: derive_intrial_threshold.py")
print("=" * 100)

tmp_thresh_dir = tempfile.mkdtemp(prefix='intrial_threshold_test_')
try:
    csv_path = os.path.join(tmp_thresh_dir, 'synth_stage4_session.csv')
    rows = [
        ['INFO', '2026-01-01 00:00:00.000000', '', '', 'SUBJECT-NAME', "['t', 'x']"],
        ['TRIAL', '', '', '', ''],
        ['STATE', '', '0.0', '60.0', 'WheelPeriod', '60.0'],
    ]
    # A sine wave, amplitude 5deg (peak-to-peak 10deg), sampled at 100Hz for 60s -- every FULL
    # 2.75s window should see ~10deg excursion, so the 80th percentile should land close to 10.0.
    amp, period, dt_step = 5.0, 0.5, 0.01
    t = 0.0
    while t < 60.0:
        pos = amp * np.sin(2 * np.pi * t / period)
        rows.append(['VAL', '', '', '', 'WHEEL_POS', '{0:.3f},{1:.3f}'.format(t, pos)])
        t += dt_step
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f, delimiter=';').writerows(rows)

    threshold_deg, n_samples, n_windows = derive_intrial_threshold.derive_intrial_threshold(
        csv_path, window_s=2.75, percentile=80)
    threshold_near_expected = abs(threshold_deg - 2 * amp) < 0.5
    samples_correct = n_samples == n_windows and n_samples > 5000
    print("  derived threshold from a known 10deg-peak-to-peak trace (expect ~10.0): {0:.2f}deg "
          "-- within tolerance: {1}".format(threshold_deg, threshold_near_expected))
    print("  sample/window counts sane ({0} samples): {1}".format(n_samples, samples_correct))

    section11_pass = threshold_near_expected and samples_correct
finally:
    shutil.rmtree(tmp_thresh_dir, ignore_errors=True)

print("Section 11 result: {0}".format("PASS" if section11_pass else "FAIL -- see checks above"))

# ==================================================================================================
# Section 12: session_csv_parser._chain_already_used() -- live-session quota-reduction chaining
# ==================================================================================================

print()
print("=" * 100)
print("Section 12: _chain_already_used() quota-reduction chaining")
print("=" * 100)


def _lite(started, ended, trial_count, duration_s, session_end_reason):
    return {'started_dt': started, 'ended_dt': ended, 'trial_count': trial_count,
            'duration_s': duration_s, 'session_end_reason': session_end_reason}


_t0 = datetime.datetime(2026, 1, 1, 12, 0, 0)

# No prior sessions at all -> nothing to resume.
used12a = session_csv_parser._chain_already_used([], _t0, 3600.0)
print("  no prior sessions -> (0, 0.0): {0}".format(used12a == (0, 0.0)))

# One prior session that ended DELIBERATELY (a real SESSION_END_REASON) -> never resumed,
# regardless of how recently it ended.
deliberate = [_lite(_t0 - datetime.timedelta(minutes=30), _t0 - datetime.timedelta(minutes=20),
                     150, 1800.0, 'cap_90min')]
used12b = session_csv_parser._chain_already_used(deliberate, _t0, 3600.0)
print("  one deliberately-ended prior session -> (0, 0.0): {0}".format(used12b == (0, 0.0)))

# One prior session with NO reason (Stop/Kill/crash), ended 10 minutes ago, well within the
# 1-hour merge gap -> its own trial_count/duration_s are "already used".
killed = [_lite(_t0 - datetime.timedelta(minutes=40), _t0 - datetime.timedelta(minutes=10),
                80, 1200.0, None)]
used12c = session_csv_parser._chain_already_used(killed, _t0, 3600.0)
print("  one killed prior session, 10min ago (within gap) -> (80, 1200.0): {0}".format(
    used12c == (80, 1200.0)))

# Same killed session, but it ended 2 HOURS ago -- gap expired, treat as a fresh session.
killed_stale = [_lite(_t0 - datetime.timedelta(hours=3), _t0 - datetime.timedelta(hours=2),
                       80, 1200.0, None)]
used12d = session_csv_parser._chain_already_used(killed_stale, _t0, 3600.0)
print("  one killed prior session, 2h ago (gap expired) -> (0, 0.0): {0}".format(
    used12d == (0, 0.0)))

# A deliberate session followed (soon after) by a killed one -- only the killed one's own amount
# counts; the deliberate one's ending closes its own bout even though the killed one starts
# promptly after it.
mixed = [
    _lite(_t0 - datetime.timedelta(hours=2), _t0 - datetime.timedelta(minutes=70), 200, 2400.0,
          'max_trials'),
    _lite(_t0 - datetime.timedelta(minutes=65), _t0 - datetime.timedelta(minutes=20), 50, 600.0,
          None),
]
used12e = session_csv_parser._chain_already_used(mixed, _t0, 3600.0)
print("  deliberate session then a killed restart -> only the restart's own (50, 600.0): {0}".format(
    used12e == (50, 600.0)))

# Two chained killed sessions (each restarted soon after the previous one's own end) -> sums both.
chained = [
    _lite(_t0 - datetime.timedelta(minutes=90), _t0 - datetime.timedelta(minutes=50), 60, 900.0,
          None),
    _lite(_t0 - datetime.timedelta(minutes=45), _t0 - datetime.timedelta(minutes=10), 40, 600.0,
          None),
]
used12f = session_csv_parser._chain_already_used(chained, _t0, 3600.0)
print("  two chained killed restarts -> summed (100, 1500.0): {0}".format(
    used12f == (100, 1500.0)))

# compute_remaining_quota()'s own floor -- a long-interrupted bout never leaves literally nothing
# to run.
remaining_trials = max(1, 200 - 199)
remaining_cap = max(60.0, 5400.0 - 5399.0)
print("  floor: 1 trial min / 60s min: {0}".format(remaining_trials == 1 and remaining_cap == 60.0))

section12_pass = (used12a == (0, 0.0) and used12b == (0, 0.0) and used12c == (80, 1200.0) and
                   used12d == (0, 0.0) and used12e == (50, 600.0) and used12f == (100, 1500.0))
print("Section 12 result: {0}".format("PASS" if section12_pass else "FAIL -- see checks above"))

# ==================================================================================================
# Section 13: BoundedStaircase (Stage 3/4's shared stim-duration/ITI ramps) +
# next_alternating_value() (Stage 3's own tone-side rule)
# ==================================================================================================

print()
print("=" * 100)
print("Section 13: BoundedStaircase + next_alternating_value()")
print("=" * 100)

sd = staircase.new_stim_duration_staircase(staircase.STAGE34_STIM_DURATION_STARTING_S)
for _ in range(staircase.STAGE34_STIM_DURATION_SUCCESSES_TO_STEP_UP - 1):
    sd.record_outcome(True)
print("  19 consecutive correct: duration unchanged (0.25s): {0}".format(
    sd.current_value == staircase.STAGE34_STIM_DURATION_STARTING_S))
sd.record_outcome(True)
print("  20th consecutive correct: duration steps UP by {0}s: {1}".format(
    staircase.STAGE34_STIM_DURATION_STEP_S,
    sd.current_value == (staircase.STAGE34_STIM_DURATION_STARTING_S
                          + staircase.STAGE34_STIM_DURATION_STEP_S)))

sd2 = staircase.new_stim_duration_staircase(staircase.STAGE34_STIM_DURATION_FINAL_S)
for _ in range(500):
    sd2.record_outcome(True)
print("  duration never exceeds the final ceiling after many correct: {0}".format(
    sd2.current_value == staircase.STAGE34_STIM_DURATION_FINAL_S))

sd3 = staircase.new_stim_duration_staircase(staircase.STAGE34_STIM_DURATION_STARTING_S)
for _ in range(500):
    sd3.record_outcome(False)
print("  duration never drops below the starting floor after many incorrect: {0}".format(
    sd3.current_value == staircase.STAGE34_STIM_DURATION_STARTING_S))

sd4 = staircase.new_stim_duration_staircase(
    staircase.STAGE34_STIM_DURATION_STARTING_S + staircase.STAGE34_STIM_DURATION_STEP_S)
for _ in range(staircase.STAGE34_STIM_DURATION_FAILURES_TO_STEP_DOWN - 1):
    sd4.record_outcome(False)
print("  4 consecutive incorrect: duration unchanged: {0}".format(
    sd4.current_value == (staircase.STAGE34_STIM_DURATION_STARTING_S
                           + staircase.STAGE34_STIM_DURATION_STEP_S)))
sd4.record_outcome(False)
print("  5th consecutive incorrect: duration steps back DOWN: {0}".format(
    sd4.current_value == staircase.STAGE34_STIM_DURATION_STARTING_S))

# A single incorrect after a long correct streak resets the success counter (no step), same
# "streak resets on the opposite outcome" convention as ThresholdStaircase.
sd5 = staircase.new_stim_duration_staircase(staircase.STAGE34_STIM_DURATION_STARTING_S)
for _ in range(19):
    sd5.record_outcome(True)
sd5.record_outcome(False)
print("  a single incorrect after 19 correct resets the streak (no step, consecutive_successes="
      "0): {0}".format(sd5.current_value == staircase.STAGE34_STIM_DURATION_STARTING_S
                        and sd5.consecutive_successes == 0))

# next_alternating_value(): same-day unchanged, new-day alternates.
same_day = staircase.next_alternating_value('R', '2026-01-01', '2026-01-01', ('L', 'R'))
new_day_from_r = staircase.next_alternating_value('R', '2026-01-01', '2026-01-02', ('L', 'R'))
new_day_from_l = staircase.next_alternating_value('L', '2026-01-01', '2026-01-02', ('L', 'R'))
print("  same-day restart keeps the side unchanged ('R' -> 'R'): {0}".format(same_day == 'R'))
print("  new day alternates 'R' -> 'L': {0}".format(new_day_from_r == 'L'))
print("  new day alternates 'L' -> 'R': {0}".format(new_day_from_l == 'R'))
# A multi-day GAP (subject skipped days) still just alternates from whatever was last used, not
# based on elapsed day count.
gap_day = staircase.next_alternating_value('R', '2026-01-01', '2026-01-10', ('L', 'R'))
print("  a multi-day gap still just alternates once, from the last-used side: {0}".format(
    gap_day == 'L'))

section13_pass = (
    sd.current_value == (staircase.STAGE34_STIM_DURATION_STARTING_S
                          + staircase.STAGE34_STIM_DURATION_STEP_S)
    and sd2.current_value == staircase.STAGE34_STIM_DURATION_FINAL_S
    and sd3.current_value == staircase.STAGE34_STIM_DURATION_STARTING_S
    and sd4.current_value == staircase.STAGE34_STIM_DURATION_STARTING_S
    and sd5.current_value == staircase.STAGE34_STIM_DURATION_STARTING_S
    and sd5.consecutive_successes == 0
    and same_day == 'R' and new_day_from_r == 'L' and new_day_from_l == 'R' and gap_day == 'L')
print("Section 13 result: {0}".format("PASS" if section13_pass else "FAIL -- see checks above"))

# ==================================================================================================
# Section 14: side = the TRIGGERING crossing; Bpod-clock parsing/alignment
# ==================================================================================================

print()
print("=" * 100)
print("Section 14: triggering-crossing side + TrialStartTimestamp / host->Bpod clock alignment")
print("=" * 100)

# The real 2026-10-22 bug shape: R threshold crossed (ignored) at 0.056s, the rewarded crossing is
# the L one at 2.7435s -- both the old first-crossing and last-vs-first rules got this wrong.
swallowed = _trial({'WheelPeriod': (0.1814, 2.7435, 2.5621), 'Reward': (2.7435, 2.8245, 0.081)},
                   {'RotaryEncoder1_2': [0.056], 'RotaryEncoder1_1': [2.7435]})
_o, side_swallowed, _d, _c = session_csv_parser.classify_trial(swallowed, config1)
side_ok = side_swallowed == 'L'
print("  ignored R crossing then rewarded L crossing -> side 'L': {0} (got {1})".format(
    side_ok, side_swallowed))

no_move = _trial({'NoMovement': (10.0, 10.0, 0.0)}, {'RotaryEncoder1_2': [0.05]})
_o, side_nm, _d, _c = session_csv_parser.classify_trial(no_move, config1)
nm_ok = side_nm is None
print("  NoMovement trial with an ignored crossing has no side: {0} (got {1})".format(nm_ok, side_nm))

tmp_clock_dir = tempfile.mkdtemp(prefix='clock_test_')
try:
    csv_path = os.path.join(tmp_clock_dir, 'synth_session.csv')
    # bpod_t = host_t * (1 - 10e-6) + 4.43 (a 10ppm, 4.43s-offset clock pair), 5 trials
    rows = []
    for i in range(5):
        host_run = 10.0 + 5.0 * i
        bpod_start = host_run * (1 - 10e-6) + 4.43
        rows += [
            ['TRIAL', '', '', '', 'New trial'],
            ['EVENT', '', '0.5', '', '1', 'RotaryEncoder1_2'],
            ['INFO', '', '{0:.6f}'.format(bpod_start), '{0:.6f}'.format(bpod_start + 1.0),
             'TRIAL-BPOD-TIME', '1.0'],
            ['STATE', '', '0', '0.5', 'WheelPeriod', '0.5'],
            ['STATE', '', '0.5', '0.6', 'Reward', '0.1'],
            ['VAL', '', '', '', 'SM_RUN_TIME', '{0:.6f}'.format(host_run)],
            # stream-mode wheel rows: marker at module 100.0 + i*5 == WheelPeriod start
            ['VAL', '', '', '', 'ROTARY_MARKER', '{0:.6f},{1:.4f}'.format(100.0 + 5 * i, host_run)],
            ['VAL', '', '', '', 'WHEEL_POS', '{0:.4f},1.00'.format(host_run + 0.25)],
            ['VAL', '', '', '', 'WHEEL_CLOCK_OFFSET', '{0:.6f}'.format(host_run - (100.0 + 5 * i))],
        ]
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f, delimiter=';').writerows(rows)
    _info, _sv, trials = session_csv_parser.parse_session_csv(csv_path)
    starts_ok = all(abs(t['bpod_start'] - (10.0 + 5.0 * i) * (1 - 10e-6) - 4.43) < 1e-6
                    for i, t in enumerate(trials))
    print("  TRIAL-BPOD-TIME parsed into each trial's bpod_start: {0}".format(starts_ok))
    fit = session_csv_parser.fit_host_to_bpod(trials)
    fit_ok = (fit is not None and abs(fit['slope'] - (1 - 10e-6)) < 1e-9
              and abs(fit['intercept'] - 4.43) < 1e-6 and fit['max_abs_residual_s'] < 1e-6)
    print("  fit_host_to_bpod() recovers the 10ppm/4.43s clock pair: {0}".format(fit_ok))
    # one trial whose RUN left 8ms after its anchor: dropped as an outlier, fit unaffected
    _out_trials = [dict(t, vals=dict(t['vals'])) for t in trials]
    _out_trials[2]['vals']['SM_RUN_TIME'] = '{0:.6f}'.format(
        float(_out_trials[2]['vals']['SM_RUN_TIME']) - 0.008)
    fit_o = session_csv_parser.fit_host_to_bpod(_out_trials)
    robust_ok = (fit_o is not None and fit_o['n_outliers'] == 1 and fit_o['outliers'][0][0] == 2
                 and abs(fit_o['slope'] - (1 - 10e-6)) < 1e-9 and fit_o['residual_sd_s'] < 1e-6)
    print("  an 8ms delayed-RUN anchor is flagged as the one outlier and doesn't bend the fit: "
          "{0}".format(robust_ok))
    fit_ok = fit_ok and robust_ok
    exact = session_csv_parser.wheel_samples_bpod_time(trials[2], 'WheelPeriod')
    # sample 0.25s after the marker -> bpod_start + 0 + 0.25
    exact_ok = exact is not None and abs(exact[0][0] - (trials[2]['bpod_start'] + 0.25)) < 1e-4
    print("  wheel_samples_bpod_time() places a sample 0.25s after the marker at "
          "bpod_start+0.25: {0}".format(exact_ok))
    anchor_ok = session_csv_parser.trial_host_anchor(trials[3]) == 25.0
    print("  trial_host_anchor() reads the trial's own SM_RUN_TIME: {0}".format(anchor_ok))
    # legacy session: no SM_RUN_TIME -- anchor recovered from THRESHOLD_CROSSING_TIME
    legacy = _trial({'Reward': (2.7435, 2.8245, 0.081)}, {},
                    {'THRESHOLD_CROSSING_TIME': '10.4986'})
    legacy_ok = abs(session_csv_parser.trial_host_anchor(legacy) - (10.4986 - 2.7435)) < 1e-9
    print("  legacy anchor recovered from THRESHOLD_CROSSING_TIME - Reward start: {0}".format(
        legacy_ok))
finally:
    shutil.rmtree(tmp_clock_dir, ignore_errors=True)

section14_pass = (side_ok and nm_ok and starts_ok and fit_ok and exact_ok and anchor_ok
                  and legacy_ok)
print("Section 14 result: {0}".format("PASS" if section14_pass else "FAIL -- see checks above"))

# ==================================================================================================
# Section 15: trial handshake + continuous wheel logging (TrialRunner/RotaryStream, no hardware)
# ==================================================================================================

print()
print("=" * 100)
print("Section 15: injected SoftCodes, quiescence tracker, wheel-log keepalive, clock map, 3-field rows")
print("=" * 100)

import threading as _threading
import time as _time
import subprocess as _subprocess
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '_shared'))
import bpod_trial_helpers as _bth
import rotary_stream as _rs


class _Inner(object):
    def __init__(self):
        self.lines = ['kill']
    def readline(self, timeout=None):
        return self.lines.pop(0) if self.lines else None
    def close(self):
        pass


inj = _bth._InjectableStdin(_Inner())
inj.put('SoftCode1')
order = [inj.readline(), inj.readline(), inj.readline()]
inject_ok = order == ['SoftCode1', 'kill', None]
print("  injected line is read before the real stdin (GUI Stop/Kill still delivered): {0} {1}".format(
    inject_ok, order))


class _FakeBpod(object):
    def __init__(self):
        self.rows = []
        self.stdin = None
    def register_value(self, k, v, flush=True):
        self.rows.append((k, v))


class _FakeStream(object):
    """ Just enough of RotaryStream for TrialRunner's logging/quiescence paths. """
    def __init__(self):
        self.samples = []      # (module_t, host_t, pos)
        self.queue = []
        self.off = 0.25
    def module_to_host(self, m):
        return m + self.off
    def samples_since_host(self, t):
        return [s for s in self.samples if s[1] >= t]
    def drain_for_logging(self):
        q, self.queue = self.queue, []
        return q
    def latest(self):
        return self.samples[-1] if self.samples else None
    def add(self, host_t, pos):
        s_ = (host_t - self.off, host_t, pos)
        self.samples.append(s_)
        self.queue.append(s_)


fb = _FakeBpod()
runner = _bth.TrialRunner(fb, None, _time.time(), use_stream=False)
runner.stream = _FakeStream()
t0 = runner._now()
runner.stream.add(t0, 3.0)
runner.flush_wheel_log()
# 0.6s of stillness, then a late flush, then movement resumes
fake_now = [t0 + 0.6]
runner._now = lambda: fake_now[0]
runner.flush_wheel_log()
runner.stream.add(t0 + 0.65, 3.5)
fake_now[0] = t0 + 0.9
runner.stream.add(t0 + 0.9, 9.0)
runner.flush_wheel_log()
rows = [tuple(map(float, v.split(','))) for k, v in fb.rows if k == 'WHEEL_POS']
ts = [r[0] for r in rows]
max_gap = max(b - a for a, b in zip(ts, ts[1:]))
keepalive_ok = max_gap <= _bth.WHEEL_LOG_MIN_INTERVAL_S + 1e-6 and ts == sorted(ts)
three_field_ok = all(len(r) == 3 and abs(r[0] - r[2] - 0.25) < 1e-3 for r in rows)
held_ok = all(r[1] == 3.0 for r in rows if t0 < r[0] < t0 + 0.65 - 1e-3)   # rows are 4-decimal
print("  still wheel + late flush -> no WHEEL_POS gap > {0:.0f}ms, rows in order: {1} (max gap "
      "{2:.0f}ms, {3} rows)".format(_bth.WHEEL_LOG_MIN_INTERVAL_S * 1e3, keepalive_ok,
                                     max_gap * 1e3, len(rows)))
print("  filled rows hold the last real position: {0}".format(held_ok))
print("  every row carries t,pos,t_module with t_module = t - offset: {0}".format(three_field_ok))

# Keepalive vs. a reader that is behind (20261102-161142, t=763.5s): rows written at "now" landed
# ahead of samples still in flight, and the never-backward clamp pinned 6 of them to one time.
# Keepalive rows must stop at the reader's caught-up time.
class _LaggingStream(_FakeStream):
    def __init__(self):
        _FakeStream.__init__(self)
        self.caught = None
    def caught_up_t(self):
        return self.caught
fb3 = _FakeBpod()
runner3 = _bth.TrialRunner(fb3, None, _time.time(), use_stream=False)
runner3.stream = _LaggingStream()
lag_now = [10.0]
runner3._now = lambda: lag_now[0]
runner3.stream.add(10.0, 1.0)
runner3.stream.caught = 10.0
runner3.flush_wheel_log()
lag_now[0] = 10.6                  # 0.6s later, but the reader last saw an empty port at 10.30
runner3.stream.caught = 10.30
runner3.flush_wheel_log()
ahead_rows = [float(v.split(',')[0]) for k, v in fb3.rows if k == 'WHEEL_POS']
no_row_ahead = max(ahead_rows) <= 10.30
for k in range(20):                # the buffered movement arrives: 10.31 .. 10.50, 10ms apart
    runner3.stream.add(10.31 + 0.01 * k, 1.0 + 0.4 * (k + 1))
runner3.stream.caught = 10.6
runner3.flush_wheel_log()
lag_rows = [tuple(map(float, v.split(','))) for k, v in fb3.rows if k == 'WHEEL_POS']
lag_mods = [r[2] for r in lag_rows]
lag_run = max(len(list(g)) for _, g in __import__('itertools').groupby(lag_mods))
lag_real = [r for r in lag_rows if r[1] != 1.0]
lag_ok = (no_row_ahead and lag_run < 3 and lag_mods == sorted(lag_mods)
          and all(abs(r[0] - (10.31 + 0.01 * i)) < 1e-3 for i, r in enumerate(lag_real)))
print("  keepalive stops at the reader's caught-up time; late samples keep their own times "
      "(last keepalive {0:.3f}s, longest same-time run {1}): {2}".format(max(ahead_rows), lag_run, lag_ok))
keepalive_ok = keepalive_ok and lag_ok

# quiescence tracker: moving for 0.3s, then still -> met only after a full still hold window
runner2 = _bth.TrialRunner(_FakeBpod(), None, _time.time(), use_stream=False)
runner2.stream = _FakeStream()
clock = [0.0]
runner2._now = lambda: clock[0]
runner2.start_quiescence(0.25, 4.0)
met_at = None
for k in range(120):
    clock[0] = k * 0.01
    if clock[0] < 0.3 and k % 3 == 0:
        runner2.stream.add(clock[0], 10.0 * ((k // 3) % 2))
    if runner2.poll_quiescence() and met_at is None:
        met_at = clock[0]
quiescence_ok = met_at is not None and 0.5 <= met_at <= 0.6 and runner2.quiescence_breaks >= 1
print("  quiescence met only after 0.25s of stillness following movement (met at {0}, breaks {1}): "
      "{2}".format(met_at, runner2.quiescence_breaks, quiescence_ok))

# RotaryStream clock map: lower envelope, no reset during a long data-free stretch, bounded rise
st = _rs.RotaryStream.__new__(_rs.RotaryStream)
st._lock = _threading.Lock(); st._carry = b''; st._raw_last = 0; st._rollover_ticks = 0
st._scale = 1e-6; st._scale_probe = []; st._samples = _rs.deque(); st._events = _rs.deque()
st._log_queue = []; st._offset_est = None; st._offset_t = None; st._latest = None
st.n_resyncs = 0; st.max_buffer_s = 60.0; st._max_read_gap_s = 0.0
st._epoch_base = 0.0; st._new_epoch = False; st._last_module_t = 0.0
def _msg(module_s, ticks=0):
    return b'P' + int(ticks).to_bytes(2, 'little', signed=True) + int(module_s * 1e6).to_bytes(4, 'little')
st._ingest(_msg(1.0), 11.010)      # latency 10ms
st._ingest(_msg(1.1), 11.102)      # latency 2ms -> adopted
est_after_low = st._offset_est
st._ingest(_msg(31.1), 41.130)     # after 30s of silence, latency 30ms -> may rise only ~1.5ms
est_after_gap = st._offset_est
offset_ok = (abs(est_after_low - 10.002) < 1e-6
             and est_after_gap - est_after_low <= (41.130 - 11.102) * 50e-6 + 1e-9)
print("  clock map keeps its lower envelope across a 30s still period (rise {0:.2f}ms, limit "
      "{1:.2f}ms): {2}".format((est_after_gap - est_after_low) * 1e3, 30 * 50e-3, offset_ok))
# A stream pause restarts the firmware clock at 0 (20261030-142131: every Stage 2 threshold step).
# The new epoch must continue on the old offset: module times keep increasing, host mapping right.
st._begin_epoch()
st._log_queue = []
st._ingest(_msg(0.010, 5) + _msg(0.020, 9), 41.333)    # restarted clock; newest latency 2ms
st._ingest(_msg(0.030, 12), 41.341)                    # latency 1ms -> offset lowered at once
drained = st.drain_for_logging()
mod_ts = [m for m, _, _ in drained]
host_ts = [h for _, h, _ in drained]
epoch_ok = (mod_ts == sorted(set(mod_ts)) and mod_ts[0] > 31.1
            and all(abs(h - want) < 0.0015 for h, want in zip(host_ts, (41.320, 41.330, 41.340))))
print("  stream restart (module clock back to 0) continues the module time and host map "
      "(module {0}, host {1}): {2}".format(['%.4f' % m for m in mod_ts],
                                           ['%.4f' % h for h in host_ts], epoch_ok))
offset_ok = offset_ok and epoch_ok

# host_clock: wall-anchored perf_counter, unaffected by a wall-clock step
import host_clock as _hc
_real_time = _rs.time.time
_t1 = _hc.now()
_hc.time.time = lambda: _real_time() + 3600.0     # the wall clock jumps an hour
try:
    _t2 = _hc.now()
finally:
    _hc.time.time = _real_time
hc_ok = 0.0 <= _t2 - _t1 < 0.5
print("  host_clock.now() ignores a wall-clock step: {0}".format(hc_ok))
offset_ok = offset_ok and hc_ok
tail_ok = [s[1] for s in _rs._tail_since(_rs.deque([(0, 1.0), (0, 2.0), (0, 3.0)]), 2.0)] == [2.0, 3.0]
print("  tail scan returns exactly the items at/after the cutoff: {0}".format(tail_ok))

# 3-field WHEEL_POS rows place samples on the Bpod clock exactly (per-row module time)
tr = {'vals': {'ROTARY_MARKER': '100.000000,10.0000', 'WHEEL_CLOCK_OFFSET': ['-90.0', '-89.98'],
               'WHEEL_POS': ['10.2500,1.00,100.250000', '10.5000,2.00,100.500000']},
      'states': {'WheelPeriod': (0.5, 1.0, 0.5)}, 'bpod_start': 20.0}
exact = session_csv_parser.wheel_samples_bpod_time(tr, 'WheelPeriod')
exact3_ok = exact is not None and abs(exact[0][0] - 20.75) < 1e-6 and abs(exact[1][0] - 21.0) < 1e-6
print("  3-field rows map to the Bpod clock via their own module time: {0}".format(exact3_ok))

section15_pass = (inject_ok and keepalive_ok and held_ok and three_field_ok and quiescence_ok
                  and offset_ok and tail_ok and exact3_ok)
print("Section 15 result: {0}".format("PASS" if section15_pass else "FAIL -- see checks above"))

# ==================================================================================================
# Section 16: pybpod Session thread-safety (the trial-5 crash) + struct export of stream-mode rows
# ==================================================================================================

print()
print("=" * 100)
print("Section 16: real pybpodapi Session under concurrent writers; struct export of 3-field rows")
print("=" * 100)

_saved_out, _saved_err = sys.stdout, sys.stderr
try:
    from pybpodapi import session as _pyb_session
    from pybpodapi.com.messaging.trial import Trial as _Trial
    from pybpodapi.com.messaging.event_occurrence import EventOccurrence as _Ev
    from pybpodapi.com.messaging.value import ValueMessage as _Val
    _have_pybpod = True
except ImportError:
    _have_pybpod = False


def _race(n_trials):
    """ Bpod-thread pattern (new Trial + events) vs main-thread pattern (VALs), concurrently.
    Returns how many trials ended up with events missing / replaced in session.trials. """
    sess = _pyb_session.Session()
    sys.stdout, sys.stderr = _saved_out, _saved_err   # Session() redirects them
    stop, bad = _threading.Event(), [0]

    def bpod_side():
        for _ in range(n_trials):
            t = _Trial(None)
            sess.add_message(t)
            for k in range(3):
                sess.add_message(_Ev(1, 'Port1In', 0.1 * k))
            if len(t.events_occurrences) != 3 or sess.trials[-1] is not t:
                bad[0] += 1
        stop.set()

    def main_side():
        while not stop.is_set():
            sess.add_message(_Val('X', 1))

    a, b = _threading.Thread(target=bpod_side), _threading.Thread(target=main_side)
    b.start(); a.start(); a.join(); b.join()
    _race.keep = getattr(_race, 'keep', []) + [sess]   # don't let Session.__del__ close stdout
    return bad[0]


if _have_pybpod:
    _old_switch = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    _bth.install_session_lock()
    bad_locked = _race(3000)
    sys.setswitchinterval(_old_switch)
    race_ok = bad_locked == 0
    print("  with install_session_lock(): 0 of 3000 trials lose/misattribute events under a concurrent "
          "VAL writer: {0} ({1} bad)".format(race_ok, bad_locked))
    # A slow disk must not block the writing threads (session 20261102-122138: one 240 ms flush
    # inside the render loop); rows still land in call order once drained.
    class _SlowStream(object):
        def write(self, data):
            pass
        def flush(self):
            _time.sleep(0.2)
        def close(self):
            pass
    tmp_slow = tempfile.mkdtemp(prefix='slowcsv_')
    try:
        slow_path = os.path.join(tmp_slow, 's.csv')
        sess = _pyb_session.Session(slow_path)
        sys.stdout, sys.stderr = _saved_out, _saved_err
        sess.csvstream.streams.append(_SlowStream())
        worst = [0.0]

        def _writer(tag):
            for k in range(40):
                t_a = _time.perf_counter()
                sess.add_message(_Val(tag, k))
                worst[0] = max(worst[0], _time.perf_counter() - t_a)
                _time.sleep(0.003)
        ths = [_threading.Thread(target=_writer, args=(tag,)) for tag in ('A', 'B')]
        for th in ths:
            th.start()
        for th in ths:
            th.join()
        sess._lab_drain()
        with open(slow_path) as f:
            got = [l.split(';')[4:6] for l in f.read().splitlines() if l.startswith('VAL;')]
        order_ok = ([v for k, v in got if k == 'A'] == [str(i) for i in range(40)] and
                    [v for k, v in got if k == 'B'] == [str(i) for i in range(40)])
        slow_ok = worst[0] < 0.005 and order_ok and len(got) == 80
        print("  200 ms disk flushes: add_message() never blocks ({0:.2f} ms worst), all 80 rows "
              "written in order after drain: {1}".format(worst[0] * 1e3, slow_ok))
        _race.keep.append(sess)

        # Kill path: exit(0) with rows still queued -> the atexit drain completes the file
        exit_path = os.path.join(tmp_slow, 'exit.csv')
        script = ("import sys, time; sys.path.insert(0, {0!r}); import bpod_trial_helpers as b; "
                  "b.install_session_lock(); from pybpodapi import session as s; "
                  "from pybpodapi.com.messaging.value import ValueMessage as V; "
                  "x = s.Session({1!r}); "
                  "[x.add_message(V('K', i)) for i in range(500)]; sys.exit(0)").format(
                      os.path.dirname(_bth.__file__), exit_path)
        _subprocess.run([sys.executable, '-c', script], timeout=60,
                        stdout=_subprocess.DEVNULL, stderr=_subprocess.DEVNULL)
        with open(exit_path) as f:
            n_exit = sum(1 for l in f if l.startswith('VAL;'))
        exit_ok = n_exit == 500
        print("  exit(0) with rows queued: all 500 rows reach the file ({0}): {1}".format(
            n_exit, exit_ok))
        race_ok = race_ok and slow_ok and exit_ok
    finally:
        sys.stdout, sys.stderr = _saved_out, _saved_err
        shutil.rmtree(tmp_slow, ignore_errors=True)
else:
    race_ok = True
    print("  pybpodapi not importable in this interpreter -- race check skipped (run with the "
          "pybpod-environment interpreter)")

# struct export of a stream-mode (3-field WHEEL_POS) CSV
tmp16 = tempfile.mkdtemp(prefix='struct3_')
try:
    path16 = os.path.join(tmp16, 'synth.csv')
    with open(path16, 'w', newline='') as f:
        w = csv.writer(f, delimiter=';')
        w.writerows([
            ['TRIAL', '', '', '', 'New trial'],
            ['INFO', '', '10.000000', '11.000000', 'TRIAL-BPOD-TIME', '1.0'],
            ['STATE', '', '0', '0.5', 'WheelPeriod', '0.5'],
            ['VAL', '', '', '', 'SM_RUN_TIME', '5.0'],
            ['VAL', '', '', '', 'WHEEL_POS', '5.1000,1.00,105.100000'],
            ['VAL', '', '', '', 'WHEEL_POS', '5.2000,2.00,105.200000'],
        ])
    import session_struct_export as _sse
    _m, _j = _sse.export_session_struct(path16, {'VAR_X': 1})
    import json as _json
    _d = _json.load(open(_j))
    export_ok = _d['trials'][0]['vals']['WHEEL_POS']['t_module'] == [105.1, 105.2]
except Exception as _err:
    export_ok = False
    print("  struct export raised: {0}".format(_err))
finally:
    shutil.rmtree(tmp16, ignore_errors=True)
print("  struct export handles 3-field WHEEL_POS rows (keeps t_module): {0}".format(export_ok))

# Camera: a slow video write must not delay frame capture (20261102-161142: one 229ms frame gap
# while the wheel and CSV logged normally -- read and write shared one thread).
import camera_recorder as _cr
import queue as _queue
# A real session runs at a 1ms Windows timer (TrialRunner.start()); at the default 15.6ms the stub
# camera's own sleeps overshoot and the 60ms bound below failed by chance on training05.
_cam_timer = _bth._set_windows_timer_resolution(True)
class _StubCap(object):
    def __init__(self): self.next_t = _time.perf_counter()
    def read(self):                    # a 30 fps camera: blocks until its next frame is due
        self.next_t += 1 / 30.0
        d = self.next_t - _time.perf_counter()
        if d > 0: _time.sleep(d)
        return True, np.zeros((4, 4, 3), dtype=np.uint8)
    def release(self): pass
class _SlowOnceWriter(object):
    def __init__(self): self.n = 0
    def write(self, frame):
        self.n += 1
        if self.n == 5: _time.sleep(0.2)
    def release(self): pass
_cam_dir = tempfile.mkdtemp(prefix='camtest_')
cam = _cr.CameraRecorder.__new__(_cr.CameraRecorder)
cam._log_python_t0 = _time.time(); cam.fps = 30.0; cam._cap = _StubCap(); cam._writer = _SlowOnceWriter()
cam.output_path = os.path.join(_cam_dir, 'v.avi'); cam._live_frame_path = os.path.join(_cam_dir, 'l.jpg')
cam._live_frame_write_interval = 10 ** 6; cam._cap_lock = _threading.Lock(); cam._latest_frame = None
cam._timestamps = []; cam._stop_event = None; cam._thread = None; cam._writer_thread = None
cam._write_queue = _queue.Queue(maxsize=60); cam.max_read_ms = 0.0; cam.max_write_ms = 0.0
cam._preview_widget = None; cam._app = None
cam.start()
_time.sleep(1.0)
cam.close()
cam_ts = [t for _, t in cam._timestamps]
cam_gap = max(b2 - a2 for a2, b2 in zip(cam_ts, cam_ts[1:]))
camera_ok = (cam_gap < 0.06 and cam._writer.n == len(cam_ts) and cam.max_write_ms >= 190
             and os.path.exists(os.path.join(_cam_dir, 'v_timestamps.csv')))
print("  camera: a 200ms video write leaves no frame gap > 60ms (max {0:.0f}ms), every frame "
      "written ({1}/{2}), CAMERA_MAX_WRITE_MS {3:.0f}: {4}".format(
          cam_gap * 1e3, cam._writer.n, len(cam_ts), cam.max_write_ms, camera_ok))
shutil.rmtree(_cam_dir, ignore_errors=True)

# Camera: a backlog of frames already in the driver must drain (2026-11-03). With the old
# whole-frame-interval throttle the loop read at most one frame per 1/fps, so a backlog never
# drained: every later frame was read ~5 frames late (and real sessions dropped to 28.5fps).
class _BackloggedCap(object):
    def __init__(self, backlog=5, fps=30.0):
        now = _time.perf_counter()
        self.iv = 1 / fps
        self.k = 0
        self.t_first = now - backlog * self.iv   # frames 0..backlog-1 are already waiting
        self.latency = []                        # read return - frame available, per frame
    def read(self):
        due = self.t_first + self.k * self.iv
        d = due - _time.perf_counter()
        if d > 0: _time.sleep(d)
        self.latency.append(_time.perf_counter() - due)
        self.k += 1
        return True, np.zeros((4, 4, 3), dtype=np.uint8)
    def release(self): pass
class _NullWriter(object):
    def write(self, frame): pass
    def release(self): pass
_cam_dir = tempfile.mkdtemp(prefix='camtest_')
cam = _cr.CameraRecorder.__new__(_cr.CameraRecorder)
cam._log_python_t0 = _time.time(); cam.fps = 30.0; cam._cap = _BackloggedCap(); cam._writer = _NullWriter()
cam.output_path = os.path.join(_cam_dir, 'v.avi'); cam._live_frame_path = os.path.join(_cam_dir, 'l.jpg')
cam._live_frame_write_interval = 10 ** 6; cam._cap_lock = _threading.Lock(); cam._latest_frame = None
cam._timestamps = []; cam._stop_event = None; cam._thread = None; cam._writer_thread = None
cam._write_queue = _queue.Queue(maxsize=60); cam.max_read_ms = 0.0; cam.max_write_ms = 0.0
cam._preview_widget = None; cam._app = None
cam.start()
_time.sleep(1.0)
cam.close()
_late = cam._cap.latency[len(cam._cap.latency) // 2:]   # second half of the run
backlog_ok = len(_late) > 0 and max(_late) < 0.02 and cam.compute_realized_fps() > 29.0
print("  camera: a 5-frame driver backlog drains -- 2nd-half read latency max {0:.1f}ms (< 20ms), "
      "{1:.1f}fps realized (> 29): {2}".format(max(_late) * 1e3 if _late else float('nan'),
                                               cam.compute_realized_fps() or 0.0, backlog_ok))
shutil.rmtree(_cam_dir, ignore_errors=True)
if _cam_timer:
    _bth._set_windows_timer_resolution(False)

# Full garbage collections are held off for a session and every collection is timed (2026-11-03:
# full collections stalled every thread for 46-87ms, growing with session length).
import gc as _gc
_gc_before = _gc.get_threshold()
_bth._hold_full_gc(True)
_gc_held = _gc.get_threshold()
_gc.collect()
_gc_ms = _bth.pop_gc_max_ms()
_gc_popped_again = _bth.pop_gc_max_ms()
_bth._hold_full_gc(False)
gc_ok = (_gc_held == (_gc_before[0], _gc_before[1], _bth.GC_FULL_THRESHOLD_HOLD)
         and _gc_ms > 0 and _gc_popped_again == 0.0
         and _gc.get_threshold() == _gc_before and _bth._gc_timer not in _gc.callbacks)
print("  gc: full collections held during a session (thresholds {0} -> {1}), a collection timed "
      "({2:.2f}ms), restored afterwards: {3}".format(_gc_before, _gc_held, _gc_ms, gc_ok))

section16_pass = race_ok and export_ok and camera_ok and backlog_ok and gc_ok
print("Section 16 result: {0}".format("PASS" if section16_pass else "FAIL -- see checks above"))

# ==================================================================================================
# Section 17: IncrementalSessionParser == parse_session_csv() on a file written in pieces
# ==================================================================================================

print()
print("=" * 100)
print("Section 17: incremental CSV parsing (live dashboard) vs full parse")
print("=" * 100)

import json as _json17


def _norm17(parsed):
    return _json17.dumps(parsed, sort_keys=True, default=str).replace('NaN', 'null')


tmp17 = tempfile.mkdtemp(prefix='incparse_')
try:
    path17 = os.path.join(tmp17, 'growing.csv')
    rows17 = [['INFO', 'pc', '', '', 'PROTOCOL-NAME', 'stage1_wheel_shaping'],
              ['VAL', 'pc', '', '', 'THRESHOLD_DEG', '3.5']]
    for i in range(30):
        rows17 += [['TRIAL', 'pc', '', '', 'New trial', ''],
                   ['EVENT', 'pc', '0.0{0}'.format(i % 10), '', '75', 'PC_QuiescenceMet'],
                   ['VAL', 'pc', '', '', 'WHEEL_POS', '{0}.1000,1.50,{0}.050000'.format(i)],
                   ['VAL', 'pc', '', '', 'WHEEL_POS', '{0}.1500,-2.00,{0}.100000'.format(i)],
                   ['END-TRIAL', 'pc', '', '', 'The trial ended', ''],
                   ['INFO', 'pc', '{0}.0'.format(i), '{0}.9'.format(i), 'TRIAL-BPOD-TIME', '0.9'],
                   ['STATE', 'pc', '0', '0.5', 'WheelPeriod', '0.5'],
                   ['STATE', 'pc', 'nan', 'nan', 'NoMovement', 'nan'],
                   ['VAL', 'pc', '', '', 'OUTCOME', 'Rewarded °'],   # non-ASCII on purpose
                   ['VAL', 'pc', '', '', 'LICK_TIMES_OUT', '']]
    import io as _io17
    buf = _io17.StringIO()
    csv.writer(buf, delimiter=';', lineterminator='\n').writerows(rows17)
    data17 = buf.getvalue().encode('utf-8')
    inc17 = session_csv_parser.IncrementalSessionParser(path17)
    open(path17, 'wb').close()
    pos, step, k = 0, 7, 0
    mid_states_ok = True
    with open(path17, 'ab') as f:
        while pos < len(data17):
            chunk = data17[pos:pos + step]          # chunks cut lines (and the UTF-8 char) apart
            f.write(chunk); f.flush()
            pos += len(chunk)
            step = 7 + (k * 13) % 61; k += 1
            _info, _sv, _tr = inc17.update()
            # a partially written trial must never look complete (the dashboard relies on this)
            for i_, tr_ in enumerate(_tr[:-1]):
                if not tr_['states']:
                    mid_states_ok = False
    same17 = _norm17(inc17.update()) == _norm17(session_csv_parser.parse_session_csv(path17))
    n_tr = len(inc17.result()[2])
finally:
    shutil.rmtree(tmp17, ignore_errors=True)
print("  file written in {0} odd-sized chunks (lines and a UTF-8 char split across reads) parses "
      "identically to a full parse: {1} ({2} trials)".format(k, same17, n_tr))
print("  every trial before the last is complete at every poll: {0}".format(mid_states_ok))

section17_pass = same17 and mid_states_ok and n_tr == 30
print("Section 17 result: {0}".format("PASS" if section17_pass else "FAIL -- see checks above"))

# ==================================================================================================
# Section 18: consumed-water accounting (Stage 1/2 logging + training log)
# ==================================================================================================

print()
print("=" * 100)
print("Section 18: ConsumedWaterTracker -- a drop counts once licked after delivery")
print("=" * 100)

from water_tracker import ConsumedWaterTracker

w18 = ConsumedWaterTracker(4.0)
w18.record_trial([], 1.0)                      # delivered, not licked yet
c1 = (w18.delivered_ul, w18.consumed_ul, w18.pending_ul) == (4.0, 0.0, 4.0)
w18.record_trial([0.2], None)                  # licked in the next (unrewarded) trial
c2 = (w18.consumed_ul, w18.pending_ul) == (4.0, 0.0)
w18.record_trial([0.5], 1.0)                   # lick only BEFORE this reward -> not consumed
c3 = (w18.consumed_ul, w18.pending_ul) == (4.0, 4.0)
w18.record_trial([], 1.0)
w18.record_trial([2.0], 1.0)                   # one lick consumes all three pooled drops
c4 = (w18.delivered_ul, w18.consumed_ul, w18.pending_ul) == (16.0, 16.0, 0.0)
for label, ok in (("unlicked drop stays pending", c1), ("a later trial's lick consumes it", c2),
                  ("a lick before delivery doesn't count", c3), ("pooled drops consumed at once", c4)):
    print("  {0}: {1}".format(label, ok))
section18_pass = c1 and c2 and c3 and c4
print("Section 18 result: {0}".format("PASS" if section18_pass else "FAIL -- see checks above"))

# ==================================================================================================
# Section 19: QC follow-ups -- monotonic keepalive module times, pre-trial VALs, lick chatter
# ==================================================================================================

print()
print("=" * 100)
print("Section 19: keepalive module times, TRIAL_INIT_START / pre-trial wheel rows, lick chatter")
print("=" * 100)

# (a) a fill row's module time is an estimate (host - offset); with the offset estimate moving
# DOWN between flushes it used to land after the next real sample's module time
fb19 = _FakeBpod()
r19 = _bth.TrialRunner(fb19, None, _time.time(), use_stream=False)
r19.stream = _FakeStream()
t19 = r19._now()
r19.stream.add(t19, 1.0)
r19.flush_wheel_log()
r19.stream.off = 0.2475                       # map tightened by 2.5 ms
r19.stream.add(t19 + 0.251, 2.0)              # real sample 1 ms after a fill row
r19.stream.samples[-1] = r19.stream.queue[-1] = (t19 + 0.251 - 0.2500, t19 + 0.251, 2.0)
r19.flush_wheel_log()
mods19 = [float(v.split(',')[2]) for k, v in fb19.rows if k == 'WHEEL_POS']
mono19 = all(b >= a for a, b in zip(mods19, mods19[1:]))
print("  module times never step backward across keepalive fill rows: {0} ({1} rows)".format(
    mono19, len(mods19)))

# (b) parser: TRIAL_INIT_START registered between END-TRIAL and the next TRIAL belongs to the next
# trial; WHEEL_POS rows before trial 1 are kept (filed under trial 1)
tmp19 = tempfile.mkdtemp(prefix='pretrial_')
try:
    p19 = os.path.join(tmp19, 's.csv')
    rows19 = [['INFO', 'pc', '', '', 'PROTOCOL-NAME', 'stage1_wheel_shaping'],
              ['VAL', 'pc', '', '', 'WHEEL_POS', '0.1,0.0,0.05'],
              ['VAL', 'pc', '', '', 'WHEEL_POS', '0.15,0.0,0.10'],
              ['VAL', 'pc', '', '', 'TRIAL_INIT_START', '1.0'],
              ['TRIAL', 'pc', '', '', 'New trial', ''],
              ['VAL', 'pc', '', '', 'WHEEL_POS', '1.1,0.5,1.05'],
              ['END-TRIAL', 'pc', '', '', 'The trial ended', ''],
              ['VAL', 'pc', '', '', 'OUTCOME', 'Rewarded'],
              ['VAL', 'pc', '', '', 'TRIAL_INIT_START', '2.0'],
              ['TRIAL', 'pc', '', '', 'New trial', ''],
              ['VAL', 'pc', '', '', 'TRIAL_INIT_START', '9.0'],    # older sessions: inside the trial
              ['END-TRIAL', 'pc', '', '', 'The trial ended', '']]
    with open(p19, 'w', newline='') as f:
        csv.writer(f, delimiter=';').writerows(rows19)
    _i19, sv19, tr19 = session_csv_parser.parse_session_csv(p19)
finally:
    shutil.rmtree(tmp19, ignore_errors=True)
pre_ok = (tr19[0]['vals'].get('TRIAL_INIT_START') == '1.0'
          and tr19[1]['vals'].get('TRIAL_INIT_START') == '9.0'
          and tr19[0]['vals'].get('OUTCOME') == 'Rewarded'
          and len(tr19[0]['vals'].get('WHEEL_POS', [])) == 3 and 'WHEEL_POS' not in sv19
          and 'TRIAL_INIT_START' not in sv19)
print("  TRIAL_INIT_START filed under its own trial, pre-trial-1 wheel rows kept: {0}".format(pre_ok))

# (c) lick chatter: an In < 40 ms after the previous Out is the same contact
merged19 = session_csv_parser.merge_lick_chatter([0.00, 0.05, 0.30, 0.45], [0.03, 0.08, 0.33, 0.48])
chatter_ok = merged19 == [0.00, 0.30, 0.45]
print("  In 20 ms after an Out merged, 120 ms after kept: {0} {1}".format(chatter_ok, merged19))

section19_pass = mono19 and pre_ok and chatter_ok
print("Section 19 result: {0}".format("PASS" if section19_pass else "FAIL -- see checks above"))

# ==================================================================================================
# Section 20: daily water total + session duration on the Bpod clock
# ==================================================================================================

print()
print("=" * 100)
print("Section 20: water_consumed_today() across sessions; training-log duration")
print("=" * 100)


def _write_session20(project, name, subject, n_rewarded, lick=True, reward_ul='7.0'):
    folder = os.path.join(project, 'experiments', 'Stage1', 'setups', 'Stage1', 'sessions', name)
    os.makedirs(folder)
    rows = [['INFO', 'pc', '', '', 'PROTOCOL-NAME', 'stage1_wheel_shaping'],
            ['INFO', 'pc', '', '', 'SUBJECT-NAME', "['{0}', 'x']".format(subject)],
            ['VAL', 'pc', '', '', 'REWARD_UL', reward_ul]]
    for i in range(n_rewarded):
        rows += [['TRIAL', 'pc', '', '', 'New trial', ''],
                 ['END-TRIAL', 'pc', '', '', 'The trial ended', ''],
                 ['INFO', 'pc', str(10.0 * i), str(10.0 * i + 1.5), 'TRIAL-BPOD-TIME', '1.5'],
                 ['STATE', 'pc', '0.5', '0.56', 'Reward', '0.06']]
        if lick:
            rows += [['EVENT', 'pc', '0.7', '', '68', 'Port1In'],
                     ['EVENT', 'pc', '0.75', '', '69', 'Port1Out']]
    with open(os.path.join(folder, name + '.csv'), 'w', newline='') as f:
        csv.writer(f, delimiter=';').writerows(rows)
    return os.path.join(folder, name + '.csv')


proj20 = tempfile.mkdtemp(prefix='daywater_')
try:
    _write_session20(proj20, '20261028-090000', 'M1', 10)            # 70 uL
    _write_session20(proj20, '20261028-100000', 'M1', 5)             # 35 uL
    _write_session20(proj20, '20261028-110000', 'M1', 5, lick=False) # delivered, never licked
    _write_session20(proj20, '20261028-120000', 'M2', 50)            # other animal
    _write_session20(proj20, '20261029-090000', 'M1', 50)            # next day
    now_csv = _write_session20(proj20, '20261028-130000', 'M1', 3)   # the session now starting
    tot20, names20 = session_csv_parser.water_consumed_today(proj20, 'M1', '20261028',
                                                             exclude_session='20261028-130000')
    dur20 = build_training_log.summarize_session(now_csv)['session_duration_s']
finally:
    shutil.rmtree(proj20, ignore_errors=True)
day_ok = abs(tot20 - 105.0) < 1e-9 and names20 == ['20261028-090000', '20261028-100000']
print("  same animal + same day only, consumed (licked) water only, current session excluded: "
      "{0} ({1:.0f} uL from {2})".format(day_ok, tot20, names20))
dur_ok = dur20 is not None and abs(dur20 - 21.5) < 1e-9
print("  session duration = last trial end - first trial start on the Bpod clock (21.5 s): "
      "{0} ({1})".format(dur_ok, dur20))

section20_pass = day_ok and dur_ok
print("Section 20 result: {0}".format("PASS" if section20_pass else "FAIL -- see checks above"))

# ==================================================================================================

print()
print("=" * 100)
print("Section 21: side-biased day halves the next day's Stage 2 starting threshold")
print("=" * 100)


def _write_session21(project, name, subject, outcomes, protocol='stage2_threshold_staircase'):
    folder = os.path.join(project, 'experiments', 'Stage2', 'setups', 'Stage2', 'sessions', name)
    os.makedirs(folder)
    rows = [['INFO', 'pc', '', '', 'PROTOCOL-NAME', protocol],
            ['INFO', 'pc', '', '', 'SUBJECT-NAME', "['{0}', 'x']".format(subject)]]
    for o in outcomes:
        rows += [['TRIAL', 'pc', '', '', 'New trial', ''],
                 ['VAL', 'pc', '', '', 'OUTCOME', o],
                 ['END-TRIAL', 'pc', '', '', 'The trial ended', '']]
    with open(os.path.join(folder, name + '.csv'), 'w', newline='') as f:
        csv.writer(f, delimiter=';').writerows(rows)


proj21 = tempfile.mkdtemp(prefix='biasday_')
try:
    none21 = session_csv_parser.previous_day_withheld(proj21, 'M1', '20261103', 'stage2_threshold_staircase')
    _write_session21(proj21, '20261030-090000', 'M1', ['Withheld'] * 50)          # older day
    _write_session21(proj21, '20261102-090000', 'M1', ['Rewarded'] * 6 + ['Withheld'] * 2 + ['NoMovement'] * 5)
    _write_session21(proj21, '20261102-100000', 'M1', ['Rewarded'] * 2)
    _write_session21(proj21, '20261102-110000', 'M2', ['Withheld'] * 50)          # other animal
    _write_session21(proj21, '20261102-120000', 'M1', ['Rewarded'] * 50, protocol='stage1_wheel_shaping')
    _write_session21(proj21, '20261103-090000', 'M1', ['Withheld'] * 50)          # today
    prev21 = session_csv_parser.previous_day_withheld(proj21, 'M1', '20261103', 'stage2_threshold_staircase')
    skip21 = session_csv_parser.previous_day_withheld(proj21, 'M1', '20261101', 'stage2_threshold_staircase')
    days21 = session_csv_parser.protocol_days_before(proj21, 'M1', '20261103', 'stage2_threshold_staircase')
    days21_early = session_csv_parser.protocol_days_before(proj21, 'M1', '20261030', 'stage2_threshold_staircase')
finally:
    shutil.rmtree(proj21, ignore_errors=True)
scan_ok = (none21 is None and prev21 == ('20261102', 2, 10, ['20261102-090000', '20261102-100000'])
           and skip21 == ('20261030', 50, 50, ['20261030-090000']))
print("  newest earlier day, this animal + Stage 2 only, sessions combined, NoMovement not a turn, "
      "today ignored, None without history: {0} ({1})".format(scan_ok, prev21))
_m = 12.5 / 35
_b = staircase.bias_day_start_fraction
halve_cases = [(_b(1.0, 0.26, 0.20, _m) * 35, 17.5), (_b(15.0 / 35, 0.20, 0.20, _m) * 35, 12.5),
               (_b(7.0 / 35, 0.21, 0.20, _m) * 35, 12.5), (_b(7.0 / 35, 0.10, 0.20, _m) * 35, 12.5),
               (_b(1.0, 0.19, 0.20, _m) * 35, 35.0), (_b(20.0 / 35, None, 0.20, _m) * 35, 20.0)]
halve_ok = all(abs(a - b) < 1e-9 for a, b in halve_cases)
print("  35->17.5, 15->12.5 (min), 7->12.5 biased or not, 19% and no history unchanged: {0} ({1})".format(
    halve_ok, [round(a, 2) for a, _ in halve_cases]))

days_ok = days21 == ['20261030', '20261102'] and days21_early == []
print("  Stage 2 day count (day-start rule only on days <= 3): this animal + Stage 2 only, one entry "
      "per day, today excluded: {0} ({1})".format(days_ok, days21))

_l = staircase.later_day_start_fraction
later_cases = [(_l(19.5 / 35, 30.0 / 35) * 35, 33.5), (_l(30.0 / 35, 20.0 / 35) * 35, 33.5),
               (_l(33.0 / 35, 34.0 / 35) * 35, 35.0), (_l(20.0 / 35, None) * 35, 23.5)]
later_ok = all(abs(a - b) < 1e-3 for a, b in later_cases)
print("  day >= 4 start: max(saved, yesterday's peak) + 3.5deg, cap 35, no peak -> saved: {0} ({1})".format(
    later_ok, [round(a, 2) for a, _ in later_cases]))
sigma_cases = [staircase.grow_direction_sigma(8.0), staircase.grow_direction_sigma(19.0),
               staircase.grow_direction_sigma(20.0)]
sigma_ok = sigma_cases == [10.0, 20.0, 20.0]
print("  direction sigma +2/day, ceiling 20: {0} ({1})".format(sigma_ok, sigma_cases))


def _recovery_turns(sigma, run=30):
    t = DirectionRatioTracker(history=['L', 'R'] * 20 + ['L'] * run, sigma=sigma)
    n = 0
    while t.over_used_side() == 'L' and n < 80:
        t.record('R')
        n += 1
    return n


recover8, recover20 = _recovery_turns(8.0), _recovery_turns(20.0)
recover_ok = recover20 > recover8
print("  a wider sigma needs more opposite turns to recover from a 30-turn run: {0} "
      "(sigma 8: {1}, sigma 20: {2})".format(recover_ok, recover8, recover20))

proj21b = tempfile.mkdtemp(prefix='daymax_')
try:
    for _name, _deg in [('20261109-090000', ['23.0', '26.5', '30.000000000000004', '19.5']),
                        ('20261109-100000', ['19.5', '23.0'])]:
        _folder = os.path.join(proj21b, 'experiments', 'Stage2', 'setups', 'Stage2', 'sessions', _name)
        os.makedirs(_folder)
        _rows = [['INFO', 'pc', '', '', 'PROTOCOL-NAME', 'stage2_threshold_staircase'],
                 ['INFO', 'pc', '', '', 'SUBJECT-NAME', "['M1', 'x']"]]
        for _d in _deg:
            _rows += [['TRIAL', 'pc', '', '', 'New trial', ''],
                      ['VAL', 'pc', '', '', 'THRESHOLD_DEG', _d],
                      ['END-TRIAL', 'pc', '', '', 'The trial ended', '']]
        with open(os.path.join(_folder, _name + '.csv'), 'w', newline='') as f:
            csv.writer(f, delimiter=';').writerows(_rows)
    max21 = session_csv_parser.day_max_threshold_deg(proj21b, 'M1', '20261109', 'stage2_threshold_staircase')
    max21_none = session_csv_parser.day_max_threshold_deg(proj21b, 'M1', '20261108', 'stage2_threshold_staircase')
finally:
    shutil.rmtree(proj21b, ignore_errors=True)
max_ok = max21 == 30.0 and max21_none is None
print("  yesterday's peak threshold over all that day's sessions, rounded: {0} ({1})".format(
    max_ok, max21))

section21_pass = scan_ok and halve_ok and days_ok and later_ok and sigma_ok and recover_ok and max_ok
print("Section 21 result: {0}".format("PASS" if section21_pass else "FAIL -- see checks above"))

# ==================================================================================================

print()
print("=" * 100)
print("Section 22: Stage 3 side blocks")
print("=" * 100)

import random as _random22
_ranges = [staircase.STAGE3_BLOCK_START]
for _ in range(6):
    _ranges.append(staircase.shrink_stage3_block_range(*_ranges[-1]))
shrink_ok = _ranges == [(30, 60), (25, 50), (20, 40), (15, 30), (10, 20), (10, 20), (10, 20)]
print("  block range shrinks 30-60 -> 25-50 -> 20-40 -> 15-30 -> 10-20, then stays: {0} ({1})".format(
    shrink_ok, _ranges[:5]))

_b = staircase.Stage3Blocks('L', 30, 60, _random22.Random(7))
_sides, _lengths, _run, _skipped = [], [], 0, 0
for _ in range(2000):
    _sides.append(_b.side)
    if _b.just_switched():
        _skipped += 1
    _len_before = _b.length
    _run += 1
    if _b.record_response():
        _lengths.append(_run)
        assert _run == _len_before
        _run = 0
_flips = [a != b for a, b in zip(_sides, _sides[1:])]
blocks_ok = (_sides[0] == 'L' and all(30 <= n <= 60 for n in _lengths) and len(_lengths) >= 30
             and sum(_flips) == len(_lengths) and len(set(_lengths)) > 5)
print("  first block = the day's side, every block 30-60 responded trials and random, side flips at "
      "every block end: {0} ({1} blocks, lengths {2}-{3})".format(
          blocks_ok, len(_lengths), min(_lengths), max(_lengths)))
skip_ok = _skipped == 10 * len(_lengths)
print("  sub-chance skip covers the first 10 responded trials of every block after the first: "
      "{0} ({1} skipped)".format(skip_ok, _skipped))

section22_pass = shrink_ok and blocks_ok and skip_ok
print("Section 22 result: {0}".format("PASS" if section22_pass else "FAIL -- see checks above"))

# ==================================================================================================

print()
print("=" * 100)
print("Section 23: A1: quiescence hold = truncated exponential scaled by q (IBL at q = 0.55)")
print("=" * 100)

import scipy.stats as _st23
_rng23 = np.random.RandomState(23)
q_checks = []
for _q in (0.1, 0.25, 0.55):
    _h = np.array([staircase.draw_quiescence_hold(_q, _rng23) for _ in range(20000)])
    _lo, _hi = staircase.QUIESCENCE_WINDOW_LO_FACTOR * _q, staircase.QUIESCENCE_WINDOW_HI_FACTOR * _q
    _in = bool(_h.min() >= _lo - 1e-12 and _h.max() <= _hi + 1e-12)
    _mean_ok = abs(_h.mean() / _q - 0.962) < 0.01
    # no mass piled on a bound: the share within 0.5% of the window width of each bound is ~0.5%
    _eps = 0.005 * (_hi - _lo)
    _at_bounds = float(np.mean((_h < _lo + _eps) | (_h > _hi - _eps)))
    _bounds_ok = _at_bounds < 0.02
    q_checks.append(_in and _mean_ok and _bounds_ok)
    print("  q={0:.2f}: window {1:.3f}-{2:.3f}s, observed {3:.3f}-{4:.3f}s, mean {5:.3f}s ({6:.3f}*q), "
          "{7:.1%} within 0.5% of a bound: {8}".format(_q, _lo, _hi, _h.min(), _h.max(), _h.mean(),
                                                       _h.mean() / _q, _at_bounds, q_checks[-1]))


def _ibl_texp(rng):   # iblrig base_choice_world.py: 0.2 + truncated_exponential(0.35, 0.2, 0.5)
    while True:
        x = rng.exponential(0.35)
        if 0.2 <= x <= 0.5:
            return 0.2 + x


_ibl = np.array([_ibl_texp(_rng23) for _ in range(20000)])
_ours = np.array([staircase.draw_quiescence_hold(0.55, _rng23) for _ in range(20000)])
_ks = _st23.ks_2samp(_ibl, _ours)
ibl_ok = _ks.pvalue > 0.01
print("  q=0.55 vs iblrig's own sampler: KS D={0:.4f}, p={1:.3f} (means {2:.3f} vs {3:.3f}s): {4}".format(
    _ks.statistic, _ks.pvalue, _ibl.mean(), _ours.mean(), ibl_ok))
ceiling_ok = staircase.QUIESCENCE_CEILING_S == 0.55 and staircase.QUIESCENCE_FLOOR_S == 0.1
print("  staircase floor/ceiling 0.10/0.55s: {0}".format(ceiling_ok))
section23_pass = all(q_checks) and ibl_ok and ceiling_ok

print("Section 23 result: {0}".format("PASS" if section23_pass else "FAIL -- see checks above"))

# ==================================================================================================

print()
print("=" * 100)
print("Section 24: B10: Stages 2-4 cap the quiescence scale at 0.25s, Stage 5 at 0.55s")
print("=" * 100)

_q24 = staircase.QuiescenceStaircase(current_s=0.2, ceiling_s=staircase.STAGE234_QUIESCENCE_CEILING_S)
for _ in range(400):
    _q24.record_outcome(True)
cap234_ok = abs(_q24.current_s - 0.25) < 1e-9
_q24b = staircase.QuiescenceStaircase(current_s=0.2)
for _ in range(400):
    _q24b.record_outcome(True)
cap5_ok = abs(_q24b.current_s - 0.55) < 1e-9
_q24c = staircase.QuiescenceStaircase(current_s=0.4, ceiling_s=staircase.STAGE234_QUIESCENCE_CEILING_S)
clamp_ok = abs(_q24c.current_s - 0.25) < 1e-9
_src24 = {}
for _stage_dir in ('stage2_threshold_staircase', 'stage3_tone_side_association', 'stage4_clicks_direction',
                   'stage5_resume_staircases'):
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', _stage_dir, _stage_dir + '.py'),
              encoding='utf-8') as _f:
        _src24[_stage_dir] = 'STAGE234_QUIESCENCE_CEILING_S' in _f.read()
wiring_ok = (_src24['stage2_threshold_staircase'] and _src24['stage3_tone_side_association']
             and _src24['stage4_clicks_direction'] and not _src24['stage5_resume_staircases'])
print("  Stage 2-4 staircase stops at 0.25s: {0}; default (Stage 5) stops at 0.55s: {1}; a saved 0.40s "
      "is clamped to 0.25s in Stages 2-4: {2}; Stages 2-4 pass the cap, Stage 5 doesn't: {3} {4}".format(
          cap234_ok, cap5_ok, clamp_ok, wiring_ok, _src24))
section24_pass = cap234_ok and cap5_ok and clamp_ok and wiring_ok

print("Section 24 result: {0}".format("PASS" if section24_pass else "FAIL -- see checks above"))

# ==================================================================================================

print()
print("=" * 100)
print("Section 25: A2: fixed ITIs in Stages 3-5 (correct 1.5s, error 4.0s), no ITI staircase")
print("=" * 100)

import re as _re25
_TASKS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
STAGE_DIRS = {1: 'stage1_wheel_shaping', 2: 'stage2_threshold_staircase', 3: 'stage3_tone_side_association',
              4: 'stage4_clicks_direction', 5: 'stage5_resume_staircases'}


def stage_source(stage):
    """ A stage script's source text (used by the source-level checks from here on). """
    with open(os.path.join(_TASKS_DIR, STAGE_DIRS[stage], STAGE_DIRS[stage] + '.py'), encoding='utf-8') as f:
        return f.read()


def stage_var(stage, name):
    """ The literal value of a module-level `NAME = <number>` in a stage script, or None. """
    m = _re25.search(r'^' + name + r'\s*=\s*([0-9.]+)', stage_source(stage), _re25.M)
    return float(m.group(1)) if m else None


iti_checks = {}
for _st in (3, 4, 5):
    _src = stage_source(_st)
    iti_checks[_st] = (stage_var(_st, 'VAR_ITI_S') == 1.5 and stage_var(_st, 'VAR_INCORRECT_ITI_S') == 4.0
                       and 'iti_staircase' not in _src and "state.set('iti_s'" not in _src
                       and "register_value('ITI_RESET_FROM_S'" in _src
                       and ('state_timer=iti_this_trial' in _src or 'state_timer=VAR_ITI_S' in _src))
print("  Stages 3-5: VAR_ITI_S 1.5, VAR_INCORRECT_ITI_S 4.0, no ITI staircase, never write 'iti_s', "
      "register ITI_RESET_FROM_S: {0}".format(iti_checks))
_ns25 = {'VAR_ITI_S': 1.5}
for _saved, _want in ((2.25, 2.25), (1.5, None), (1.3, None), (None, None)):
    _ns25['state'] = type('S', (), {'get': staticmethod(lambda k, v=_saved: v)})
    exec(_re25.search(r'^ITI_RESET_FROM_S = .*$', stage_source(3), _re25.M).group(0), _ns25)
    iti_checks['reset %s' % _saved] = _ns25['ITI_RESET_FROM_S'] == _want
print("  a saved 'iti_s' above 1.5s is reported, at or below it isn't: {0}".format(
    {k: v for k, v in iti_checks.items() if str(k).startswith('reset')}))
no_ramp_ok = not hasattr(staircase, 'new_iti_staircase') and not hasattr(staircase, 'STAGE34_ITI_STARTING_S')
print("  staircase.py has no ITI ramp left: {0}".format(no_ramp_ok))
section25_pass = all(iti_checks.values()) and no_ramp_ok

print("Section 25 result: {0}".format("PASS" if section25_pass else "FAIL -- see checks above"))

# ==================================================================================================

print()
print("=" * 100)
print("Section 26: A4: an abort costs the error ITI (AbortITI = VAR_INCORRECT_ITI_S)")
print("=" * 100)

abort_checks = {}
for _st in (3, 4, 5):
    _src = stage_source(_st)
    _abort_iti = _re25.search(r"state_name='AbortITI',\s*state_timer=(\w+)", _src)
    abort_checks[_st] = ('VAR_ABORT_TIMEOUT' not in _src
                         and 'abort_timeout_this_trial = VAR_INCORRECT_ITI_S' in _src
                         and _abort_iti is not None and _abort_iti.group(1) == 'abort_timeout_this_trial'
                         and "runner.register('ABORT_ITI_S', abort_timeout_this_trial)" in _src)
print("  Stages 3-5: AbortITI lasts VAR_INCORRECT_ITI_S, no VAR_ABORT_TIMEOUT_* left, ABORT_ITI_S "
      "registered: {0}".format(abort_checks))
section26_pass = all(abort_checks.values())

print("Section 26 result: {0}".format("PASS" if section26_pass else "FAIL -- see checks above"))

# ==================================================================================================

print()
print("=" * 100)
print("Section 27: A5: J2 (crossing -> dot offset) is 0.3-0.6s in every stage, from one constant")
print("=" * 100)

j2_checks = {}
for _st in (1, 2, 3, 4, 5):
    _src = stage_source(_st)
    j2_checks[_st] = ('VAR_DOT_DISAPPEAR_MIN_S = staircase.DOT_DISAPPEAR_MIN_S' in _src
                      and 'VAR_DOT_DISAPPEAR_MAX_S = staircase.DOT_DISAPPEAR_MAX_S' in _src
                      and 'uniform(VAR_DOT_DISAPPEAR_MIN_S, VAR_DOT_DISAPPEAR_MAX_S)' in _src)
window_ok = (staircase.DOT_DISAPPEAR_MIN_S, staircase.DOT_DISAPPEAR_MAX_S) == (0.3, 0.6)
print("  shared window 0.3-0.6s: {0}; every stage draws J2 from it (one window for both outcomes): "
      "{1}".format(window_ok, j2_checks))
section27_pass = window_ok and all(j2_checks.values())

print("Section 27 result: {0}".format("PASS" if section27_pass else "FAIL -- see checks above"))

# ==================================================================================================

print()
print("=" * 100)
overall_pass = (section1_pass and section2_pass and section3_pass and section4_pass and
                 section5_pass and section6_pass and section7_pass and section8_pass and
                 section9_pass and section10_pass and section11_pass and section12_pass and
                 section13_pass and section14_pass and section15_pass and section16_pass and
                 section17_pass and section18_pass and
                 section19_pass and section20_pass and section21_pass and section22_pass and
                 section23_pass and
                 section24_pass and
                 section25_pass and
                 section26_pass and
                 section27_pass)
print("OVERALL: {0}".format("ALL SECTIONS PASS" if overall_pass else "AT LEAST ONE SECTION FAILED"))
print("=" * 100)
