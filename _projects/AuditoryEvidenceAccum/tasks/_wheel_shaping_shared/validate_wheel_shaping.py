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

Run with the pybpod-environment interpreter:
    /c/Users/2P-Behav/.conda/envs/pybpod-environment/python.exe validate_wheel_shaping.py
"""
import datetime
import os
import shutil
import sys
import tempfile

import staircase
import session_state
import session_csv_parser
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

section2_pass = (iti_before_ceiling and iti_at_ceiling and iti_stays_at_ceiling and
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

# Regression check: before the window is full, even an extreme (all-one-side) ratio must NOT
# trigger withholding -- confirmed on hardware that OUTCOME=Withheld fired as early as trial 2
# before this gate existed (right_fraction() over 1-2 trials is pure noise, not a real bias).
dt4 = DirectionRatioTracker()
dt4.record('L')
one_trial_not_full = not dt4.has_full_window()
one_trial_in_band = dt4.in_band() and not dt4.should_withhold('L') and not dt4.should_withhold('R')
print("  1 trial recorded: has_full_window()=False: {0}".format(one_trial_not_full))
print("  1 trial, all-'L' (extreme ratio): in_band=True, no withholding on either side "
      "(not enough data yet to trust it): {0}".format(one_trial_in_band))

dt5 = DirectionRatioTracker()
for _ in range(39):
    dt5.record('L')
still_not_full_at_39 = not dt5.has_full_window()
still_no_withhold_at_39 = dt5.in_band() and not dt5.should_withhold('L')
print("  39/40 trials, all-'L': still has_full_window()=False, still no withholding: {0}".format(
    still_not_full_at_39 and still_no_withhold_at_39))
dt5.record('L')
full_at_40_withholds = dt5.has_full_window() and dt5.should_withhold('L')
print("  40th trial completes the window: has_full_window()=True, NOW withholds 'L': {0}".format(
    full_at_40_withholds))

section3_pass = (neutral_start and over_used_right and withhold_right and balanced_in_band and
                  window_check and one_trial_not_full and one_trial_in_band and
                  still_not_full_at_39 and still_no_withhold_at_39 and full_at_40_withholds)
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
        'no_movement_count': 0, 'l_count': 3, 'r_count': 4, 'lick_count': 5, 'aborts': 1,
        'reward_ul': reward_ul, 'consumed_volume_ul': consumed_volume_ul,
        'threshold_start_deg': 7.0, 'threshold_end_deg': 7.0, 'threshold_final_deg': 35.0,
        'iti_end_s': 0.5, 'gain_mult_end': 2.0, 'direction_ratio_end': None,
        'simple_gates_met': None,
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

section6_pass = section6a_pass and section6b_pass and section6c_pass and section6d_pass
print("Section 6 result: {0}".format("PASS" if section6_pass else "FAIL -- see checks above"))

# ==================================================================================================

print()
print("=" * 100)
overall_pass = (section1_pass and section2_pass and section3_pass and section4_pass and
                 section5_pass and section6_pass)
print("OVERALL: {0}".format("ALL SECTIONS PASS" if overall_pass else "AT LEAST ONE SECTION FAILED"))
print("=" * 100)
