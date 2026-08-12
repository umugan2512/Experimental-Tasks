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

Run with the pybpod-environment interpreter:
    /c/Users/2P-Behav/.conda/envs/pybpod-environment/python.exe validate_wheel_shaping.py
"""
import os
import shutil
import tempfile

import staircase
import session_state
from direction_tracker import DirectionRatioTracker

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

section1_pass = (below_step and stepped_up and reset_after_step and below_step_down and
                  stepped_down and ceiling_check and reset_on_opposite)
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

gain = 2.0
for _ in range(50):
    gain = staircase.decay_gain(gain)
gain_floor_check = abs(gain - staircase.GAIN_FLOOR_MULT) < 1e-6
print("  gain decays to its floor ({0:.1f}x) after many qualifying sessions: {1:.4f} -- {2}".format(
    staircase.GAIN_FLOOR_MULT, gain, gain_floor_check))

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

section2_pass = (iti_before_ceiling and iti_at_ceiling and iti_stays_at_ceiling and
                  gain_floor_check and threshold_before_ceiling and threshold_at_ceiling and
                  threshold_stays_at_ceiling)
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
over_used_right = dt.over_used_side() == 'R' and not dt.in_band()
withhold_right = dt.should_withhold('R') and not dt.should_withhold('L')
print("  35R/5L over 40 trials (87.5% R): over_used_side='R', out of band: {0}".format(
    over_used_right))
print("  should_withhold('R')=True, should_withhold('L')=False: {0}".format(withhold_right))

dt2 = DirectionRatioTracker()
for i in range(40):
    dt2.record('R' if i % 2 == 0 else 'L')
balanced_in_band = dt2.in_band() and dt2.over_used_side() is None
print("  perfectly alternating 40 trials (50/50): in_band=True, over_used_side=None: {0}".format(
    balanced_in_band))

dt3 = DirectionRatioTracker(window=40)
for _ in range(50):
    dt3.record('R')
window_check = len(dt3._sides) == 40   # rolling window caps at 40, doesn't grow unbounded
print("  rolling window caps at 40 entries after 50 records: {0}".format(window_check))

section3_pass = neutral_start and over_used_right and withhold_right and balanced_in_band and \
    window_check
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

trial_count_fails = staircase.stage2_simple_gates_met(
    trial_count=200, iti_s=1.5, direction_ratio_in_band=True)
print("  exactly 200 trials (needs STRICTLY >): expect False: {0}".format(
    trial_count_fails is False))

iti_fails = staircase.stage2_simple_gates_met(
    trial_count=250, iti_s=1.4, direction_ratio_in_band=True)
print("  ITI not yet at ceiling (1.4s): expect False: {0}".format(iti_fails is False))

ratio_fails = staircase.stage2_simple_gates_met(
    trial_count=250, iti_s=1.5, direction_ratio_in_band=False)
print("  direction ratio out of band: expect False: {0}".format(ratio_fails is False))

section5_pass = all_met and trial_count_fails is False and iti_fails is False and \
    ratio_fails is False
print("Section 5 result: {0}".format("PASS" if section5_pass else "FAIL -- see checks above"))

# ==================================================================================================

print()
print("=" * 100)
overall_pass = section1_pass and section2_pass and section3_pass and section4_pass and section5_pass
print("OVERALL: {0}".format("ALL SECTIONS PASS" if overall_pass else "AT LEAST ONE SECTION FAILED"))
print("=" * 100)
