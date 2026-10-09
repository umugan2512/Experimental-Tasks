# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Native Bpod session-CSV parsing, shared by `build_training_log.py` and `session_struct_export.py`
-- reconstructs the per-trial STATE/EVENT/VAL structure directly from the raw CSV rows (the same
per-trial shape `states_durations`/`get_all_timestamps_by_event()` give a *live* task script -- see
CLAUDE.md's "native-STATE/EVENT-parsing" note for why this is preferred over custom VAL
registrations wherever the native rows already have the answer).

Extracted from `build_training_log.py` (originally the only consumer) so `session_struct_export.py`
doesn't need its own copy.
"""
import ast
import csv
import datetime
import glob
import math
import os

# VAL keys that get registered repeatedly WITHIN a single trial (e.g. once per wheel-position
# sample, once per quiescence reset) rather than once -- the generic VAL branch below used to keep
# only the LAST such registration per trial (current['vals'][key] = value), silently discarding
# every earlier sample. WHEEL_POS's own full per-sample stream never round-tripped into
# session_struct_export.py's exported struct for exactly this reason. Any key in this tuple is
# instead accumulated into a list, in registration order, preserving every sample.
STREAM_KEYS = ('WHEEL_POS', 'QUIESCENCE_RESET_TIME')

# --- per-protocol state/event meaning -- extend as new protocols (Stage 3+) are built ---------------

PROTOCOL_CONFIG = {
    'stage1_wheel_shaping': {
        'rewarded_states': ['Reward'],
        'withheld_states': [],
        'no_movement_states': ['NoMovement'],
        # Stage 1 doesn't split its Reward state by side (either direction routes to the same
        # state) -- side comes from the raw threshold-crossing EVENT names instead.
        'side_from': 'event',
        'left_event': 'RotaryEncoder1_1',
        'right_event': 'RotaryEncoder1_2',
    },
    'stage2_threshold_staircase': {
        'rewarded_states': ['RewardL', 'RewardR'],
        'withheld_states': ['NoRewardL', 'NoRewardR'],
        'no_movement_states': ['NoMovement'],
        'side_from': 'state',   # the state name's own trailing L/R suffix
    },
    # A genuine 2AFC choice protocol (_projects/Tests/tasks/full_protocol_lookback_test/), a
    # structurally different trial-outcome shape than Stage 1/2's shaping-only one -- 'incorrect'
    # (a real wrong-choice outcome, not "withheld") and 'aborted' (a mid-CUE wheel movement, from
    # the separate cue_sma that runs BEFORE the choice state machine even exists for that trial --
    # never co-occurs with rewarded/incorrect/no_movement, since a trial that aborts never reaches
    # the second state machine at all) are real, distinct outcomes here, not present in Stage 1/2 at
    # all -- see classify_trial()'s own docstring for why they're kept as their own outcome values
    # rather than approximated onto 'withheld'/'aborts'. 'no_movement_states' here really means "no
    # response within the timeout" (NoResponse), not "never moved the wheel" the way it does in
    # Stage 1/2 -- close enough in spirit (no choice made) to reuse the same bucket/column.
    'full_protocol_lookback_test': {
        'rewarded_states': ['Reward'],
        'incorrect_states': ['ErrorConsumption'],
        'no_movement_states': ['NoResponse'],
        'aborted_states': ['WheelAbort'],
        'withheld_states': [],
        'side_from': 'event',
        'left_event': 'RotaryEncoder1_3',    # ALL_THRESHOLDS_DEG's 3rd entry (LEFT_THRESHOLD_DEG)
        'right_event': 'RotaryEncoder1_4',   # ALL_THRESHOLDS_DEG's 4th entry (RIGHT_THRESHOLD_DEG)
    },
    # Stage 3 ("tone-side association"), Stage 4 ("clicks and direction") and Stage 5 ("resume
    # threshold and quiescence staircases") -- training_protocol.md Revision 2 (Stage 3 is a new
    # stage inserted ahead of what used to be Stage 3/Stage 4, pushing those up to 4/5). Same
    # real-2AFC-choice shape as full_protocol_lookback_test above (a genuine incorrect/aborted
    # outcome exists, unlike Stage 1/2's shaping-only outcomes), and deliberately the SAME state
    # names (CuePeriod/DelayPeriod/WheelAbort/WheelDotPeriod/Reward/ErrorConsumption/NoResponse)
    # and event-index convention (ALL_THRESHOLDS_DEG's 3rd/4th entries for left/right) as that
    # already-validated script, since stage3_tone_side_association.py/stage4_clicks_direction.py/
    # stage5_resume_staircases.py's own state machines were built as direct structural copies of it.
    'stage3_tone_side_association': {
        'rewarded_states': ['Reward'],
        'incorrect_states': ['ErrorConsumption'],
        'no_movement_states': ['NoResponse'],
        'aborted_states': ['WheelAbort'],
        'withheld_states': [],
        'side_from': 'event',
        'left_event': 'RotaryEncoder1_3',
        'right_event': 'RotaryEncoder1_4',
    },
    'stage4_clicks_direction': {
        'rewarded_states': ['Reward'],
        'incorrect_states': ['ErrorConsumption'],
        'no_movement_states': ['NoResponse'],
        'aborted_states': ['WheelAbort'],
        'withheld_states': [],
        'side_from': 'event',
        'left_event': 'RotaryEncoder1_3',
        'right_event': 'RotaryEncoder1_4',
    },
    'stage5_resume_staircases': {
        'rewarded_states': ['Reward'],
        'incorrect_states': ['ErrorConsumption'],
        'no_movement_states': ['NoResponse'],
        'aborted_states': ['WheelAbort'],
        'withheld_states': [],
        'side_from': 'event',
        'left_event': 'RotaryEncoder1_3',
        'right_event': 'RotaryEncoder1_4',
    },
}


def _isnan(x):
    return x is None or (isinstance(x, float) and math.isnan(x))


def _parse_float(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def parse_session_csv(path):
    """ Returns (info, session_vals, trials).
    info: dict of INFO-row key/value pairs (SUBJECT-NAME, PROTOCOL-NAME, SESSION-STARTED, ...).
    session_vals: VAL rows registered BEFORE the first TRIAL marker (e.g. a once-at-startup
        registration like stage1_wheel_shaping.py's THRESHOLD_DEG, which isn't per-trial there).
    trials: list of {'states': {name: (start, end, duration)}, 'events': {name: [t, ...]},
        'vals': {key: value_str}} in trial order. """
    info = {}
    session_vals = {}
    trials = []
    current = None

    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f, delimiter=';')
        for row in reader:
            if len(row) < 5:
                continue
            row_type = row[0]
            if row_type == 'INFO':
                info[row[4]] = row[5] if len(row) > 5 else ''
            elif row_type == 'TRIAL':
                current = {'states': {}, 'events': {}, 'vals': {}}
                trials.append(current)
            elif row_type == 'STATE' and len(row) > 5:
                name = row[4]
                start, end, duration = _parse_float(row[2]), _parse_float(row[3]), _parse_float(row[5])
                if current is not None:
                    current['states'][name] = (start, end, duration)
            elif row_type == 'EVENT' and len(row) > 5:
                t = _parse_float(row[2])
                name = row[5]
                if t is not None and current is not None:
                    current['events'].setdefault(name, []).append(t)
            elif row_type == 'VAL' and len(row) > 5:
                key, value = row[4], row[5]
                if current is not None:
                    if key in STREAM_KEYS:
                        current['vals'].setdefault(key, []).append(value)
                    else:
                        current['vals'][key] = value
                else:
                    session_vals[key] = value

    return info, session_vals, trials


def parse_subject_name(raw):
    """ INFO's SUBJECT-NAME value looks like "['test', '7f6bd106-...']" -- a Python-list repr, not
    JSON (single quotes). ast.literal_eval handles it safely without eval(). """
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, (list, tuple)) and parsed:
            return str(parsed[0])
    except (ValueError, SyntaxError):
        pass
    return raw or 'UNKNOWN'


# --- per-trial classification ------------------------------------------------------------------

def classify_trial(trial, config):
    """ Returns (outcome, side, reward_duration_s, consumed). outcome in {'rewarded', 'withheld',
    'no_movement', 'incorrect', 'aborted', 'unknown'} -- 'incorrect'/'aborted' only occur for
    protocols whose PROTOCOL_CONFIG sets 'incorrect_states'/'aborted_states' (Stage 1/2 never
    produce them, since those keys are simply absent from their config dicts). side in
    {'L', 'R', None}; reward_duration_s is that trial's own logged Reward-state duration if
    rewarded, else None. consumed is True/False for a rewarded trial (whether a Port1In lick was
    detected at/after that trial's own reward-delivery start time -- no dedicated
    Consumption-window state exists in Stage 1/2, Port1In is a raw event regardless of which state
    is active), None for a non-rewarded trial. """
    states = trial['states']
    events = trial['events']

    def visited(name):
        return name in states and not _isnan(states[name][0])

    rewarded_name = next((n for n in config.get('rewarded_states', []) if visited(n)), None)
    withheld_name = next((n for n in config.get('withheld_states', []) if visited(n)), None)
    incorrect_name = next((n for n in config.get('incorrect_states', []) if visited(n)), None)

    if rewarded_name:
        outcome = 'rewarded'
    elif withheld_name:
        outcome = 'withheld'
    elif incorrect_name:
        outcome = 'incorrect'
    elif any(visited(n) for n in config.get('aborted_states', [])):
        outcome = 'aborted'
    elif any(visited(n) for n in config.get('no_movement_states', [])):
        outcome = 'no_movement'
    else:
        outcome = 'unknown'

    side = None
    if config.get('side_from') == 'state':
        state_name = rewarded_name or withheld_name
        if state_name and state_name.endswith('L'):
            side = 'L'
        elif state_name and state_name.endswith('R'):
            side = 'R'
    elif config.get('side_from') == 'event':
        left_ev = events.get(config.get('left_event'), [])
        right_ev = events.get(config.get('right_event'), [])
        if left_ev and right_ev:
            side = 'L' if left_ev[0] < right_ev[0] else 'R'
        elif left_ev:
            side = 'L'
        elif right_ev:
            side = 'R'

    reward_duration = states[rewarded_name][2] if rewarded_name else None

    consumed = None
    if rewarded_name:
        reward_start = states[rewarded_name][0]
        lick_times = events.get('Port1In', [])
        consumed = any(t >= reward_start for t in lick_times) if not _isnan(reward_start) else False

    return outcome, side, reward_duration, consumed


def _val_float(vals_dicts, key):
    """ Checks each dict in vals_dicts in order (e.g. trial-level first, then session-level
    fallback for a once-at-startup registration like Stage 1's THRESHOLD_DEG) and returns the
    first float it finds. """
    for vals in vals_dicts:
        if key in vals:
            parsed = _parse_float(vals[key])
            if parsed is not None:
                return parsed
    return None


def find_val_backward(trials, session_vals, key):
    """ Searches trials from the LAST back to the first for a VAL row matching key, falling back
    to session_vals (a once-at-startup registration, e.g. Stage 1's THRESHOLD_DEG) if none of the
    trials have it. Robust against a trailing INCOMPLETE trial (session stopped/killed mid-hold,
    before that trial's Bpod machine ever ran -- confirmed on a real bench-test log: its own
    'vals' has only WHEEL_POS/TRIAL_START, none of the outcome-time registrations), which
    trials[-1] alone is not. """
    for trial in reversed(trials):
        value = _val_float([trial['vals']], key)
        if value is not None:
            return value
    return _val_float([session_vals], key)


def find_val_forward(trials, session_vals, key):
    """ Same as find_val_backward but searches from the FIRST trial forward -- used for a
    session-*start* value. """
    for trial in trials:
        value = _val_float([trial['vals']], key)
        if value is not None:
            return value
    return _val_float([session_vals], key)


def find_val_str_backward(trials, session_vals, key):
    """ Same backward search as find_val_backward, but returns the raw string (no float parsing)
    -- for a VAL whose value is genuinely text, e.g. SESSION_END_REASON. """
    for trial in reversed(trials):
        if key in trial['vals']:
            return trial['vals'][key]
    return session_vals.get(key)


def sum_val(trials, key):
    """ Sums a per-trial numeric VAL across every trial in the given (already real_trials()
    -filtered) list -- e.g. QUIESCENCE_BREAKS -> total quiescence-hold breaks for the session.
    Trials missing the VAL contribute 0, not None -- a session where nothing was ever registered
    for this key sums to a plain 0, matching every other count in this module. """
    total = 0
    for trial in trials:
        value = _val_float([trial['vals']], key)
        if value is not None:
            total += value
    return total


def parse_datetime(s):
    """ Bpod's own INFO-row timestamps (SESSION-STARTED, SESSION-ENDED) are Python
    datetime.now()'s default str() format -- microseconds are present unless they happen to be
    exactly zero, in which case str() omits them entirely, so both forms need trying. """
    if not s:
        return None
    for fmt in ('%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S'):
        try:
            return datetime.datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def session_end_datetime(info, started_dt, duration_s):
    """ Absolute session-end time: prefers the native SESSION-ENDED INFO row (written by
    Bpod.close() -- present for every cleanly-terminated session, i.e. completed, Stopped, or
    Killed). Falls back to started_dt + duration_s (the crash case -- nothing wrote a clean end
    record, so this is only an estimate based on the last logged event time). Returns None if
    neither is available (no started_dt and no SESSION-ENDED). """
    ended = parse_datetime(info.get('SESSION-ENDED'))
    if ended is not None:
        return ended
    if started_dt is not None and duration_s is not None:
        return started_dt + datetime.timedelta(seconds=duration_s)
    return None


def real_trials(trials):
    """ A trailing trial with no STATE rows at all means the session was stopped/killed mid-hold,
    before that trial's Bpod machine ever ran (confirmed on a real bench-test log) -- not a
    genuine attempted trial. """
    return [t for t in trials if t['states']]


# --- live-session quota reduction after an interrupted (Stop/Kill/crash) restart -----------------
# A stage script calls compute_remaining_quota() at its own startup, right after its full
# VAR_MAX_TRIALS/VAR_SESSION_CAP_S are defined, to shrink whichever of those this session is
# really a CONTINUATION of an interrupted bout for (same protocol+subject, restarted soon after an
# abnormal end) -- so the animal isn't given a full fresh quota on top of what it already did in
# the part that got cut short. A session that reaches its own deliberate end (any registered
# SESSION_END_REASON at all -- "absence signals an abnormal/early end", same convention as
# elsewhere in this codebase) always resets the next session back to the FULL quota, matching
# build_training_log.py's own group_sessions() merge-eligibility rule but keyed on actual absence
# of a reason rather than the literal string 'completed' (stage3/4/5 scripts never register that
# exact value for their own deliberate endings, unlike stage1/2's for...else branch).

def session_lite_summary(path, protocol_name_filter=None, subject_id_filter=None):
    """ Minimal per-session summary for compute_remaining_quota() below -- deliberately NOT
    build_training_log.py's own summarize_session() (no openpyxl dependency, so a live task
    script can import this at launch without pulling that in). Returns None if the file isn't
    parseable, has no real trials, or doesn't match protocol_name_filter/subject_id_filter (either
    may be left None to skip that check). """
    try:
        info, session_vals, trials = parse_session_csv(path)
    except (IOError, OSError):
        return None
    if protocol_name_filter is not None and info.get('PROTOCOL-NAME') != protocol_name_filter:
        return None
    if subject_id_filter is not None:
        if parse_subject_name(info.get('SUBJECT-NAME')) != subject_id_filter:
            return None
    trials = real_trials(trials)
    if not trials:
        return None
    all_event_times = [t for trial in trials for times in trial['events'].values() for t in times]
    duration_s = (max(all_event_times) - min(all_event_times)) if all_event_times else None
    started_dt = parse_datetime(info.get('SESSION-STARTED'))
    return {
        'trial_count': len(trials),
        'duration_s': duration_s,
        'started_dt': started_dt,
        'ended_dt': session_end_datetime(info, started_dt, duration_s),
        'session_end_reason': find_val_str_backward(trials, session_vals, 'SESSION_END_REASON'),
    }


def _chain_already_used(sorted_summaries, now_dt, merge_gap_s):
    """ Pure chaining logic (unit-testable without real files -- see validate_wheel_shaping.py) --
    given lite summaries for ONE protocol+subject, chronologically sorted, returns (trial_count,
    duration_s) already consumed by the still-open bout (if any) that a session starting at
    now_dt would be continuing. A bout stays "open" across successive summaries only while each
    one's own session_end_reason is absent (Stop/Kill/crash) and it started within merge_gap_s of
    the previous one's own end; any registered reason closes the bout right there, even if the
    NEXT summary in the list starts promptly after it (that one starts a fresh bout of its own
    instead). Returns (0, 0.0) if the list is empty, the most recent summary's bout isn't open, or
    the gap from its end to now_dt has already expired. """
    chain_trials = 0
    chain_duration_s = 0.0
    open_since_dt = None   # None means "no open bout right now"
    for summary in sorted_summaries:
        started_dt = summary['started_dt']
        continues_open_bout = (open_since_dt is not None and started_dt is not None
                                and (started_dt - open_since_dt).total_seconds() <= merge_gap_s)
        if continues_open_bout:
            chain_trials += summary['trial_count']
            chain_duration_s += (summary['duration_s'] or 0.0)
        else:
            chain_trials = summary['trial_count']
            chain_duration_s = (summary['duration_s'] or 0.0)
        open_since_dt = None if summary['session_end_reason'] is not None else summary['ended_dt']

    if open_since_dt is None:
        return 0, 0.0
    if (now_dt - open_since_dt).total_seconds() > merge_gap_s:
        return 0, 0.0
    return chain_trials, chain_duration_s


def compute_remaining_quota(sessions_dir, this_session_name, protocol_name, subject_id,
                             full_max_trials=None, full_cap_s=None, merge_gap_s=3600.0):
    """ Scans sibling session folders under sessions_dir (same setup, i.e. same task) older than
    this_session_name, and reduces whichever of full_max_trials/full_cap_s the caller passed by
    however much of it the still-open interrupted bout (see _chain_already_used() above) already
    used -- floored at 1 trial / 60s so a long-interrupted bout never leaves literally nothing to
    run. Returns (remaining_max_trials, remaining_cap_s); either is None if the caller didn't pass
    that quota at all (e.g. stage1/2 only ever pass full_max_trials, stage3/4/5 only full_cap_s).
    Never raises on a bad/unreadable sibling file -- this is a best-effort welfare nicety, not
    something that should ever block a session from starting. """
    summaries = []
    for folder in sorted(glob.glob(os.path.join(sessions_dir, '*'))):
        name = os.path.basename(folder)
        if name >= this_session_name:
            continue
        csv_paths = glob.glob(os.path.join(folder, '*.csv'))
        if not csv_paths:
            continue
        try:
            summary = session_lite_summary(csv_paths[0], protocol_name_filter=protocol_name,
                                            subject_id_filter=subject_id)
        except Exception:
            continue
        if summary is not None:
            summaries.append(summary)

    used_trials, used_duration_s = _chain_already_used(
        summaries, datetime.datetime.now(), merge_gap_s)

    remaining_max_trials = (max(1, full_max_trials - used_trials)
                             if full_max_trials is not None else None)
    remaining_cap_s = (max(60.0, full_cap_s - used_duration_s)
                        if full_cap_s is not None else None)
    return remaining_max_trials, remaining_cap_s
