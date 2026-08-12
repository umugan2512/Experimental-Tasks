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
import math

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
    """ Returns (outcome, side, reward_duration_s). outcome in {'rewarded', 'withheld',
    'no_movement', 'unknown'}; side in {'L', 'R', None}; reward_duration_s is that trial's own
    logged Reward-state duration if rewarded, else None. """
    states = trial['states']
    events = trial['events']

    def visited(name):
        return name in states and not _isnan(states[name][0])

    rewarded_name = next((n for n in config.get('rewarded_states', []) if visited(n)), None)
    withheld_name = next((n for n in config.get('withheld_states', []) if visited(n)), None)

    if rewarded_name:
        outcome = 'rewarded'
    elif withheld_name:
        outcome = 'withheld'
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
    return outcome, side, reward_duration


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


def real_trials(trials):
    """ A trailing trial with no STATE rows at all means the session was stopped/killed mid-hold,
    before that trial's Bpod machine ever ran (confirmed on a real bench-test log) -- not a
    genuine attempted trial. """
    return [t for t in trials if t['states']]
