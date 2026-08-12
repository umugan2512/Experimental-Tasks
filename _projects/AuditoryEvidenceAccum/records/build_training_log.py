# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Scans every session CSV under experiments/*/setups/*/sessions/, parses the native Bpod
STATE/EVENT/TRIAL structure directly out of the raw CSV rows (the same per-trial shape
`states_durations`/`get_all_timestamps_by_event()` give a *live* task script -- see CLAUDE.md's
"native-STATE/EVENT-parsing" note for why this is preferred over custom VAL registrations wherever
the native rows already have the answer), merges in the two hand-maintained files in this same
folder (animals_metadata.json, session_manual_entries.csv), and **rebuilds training_log.xlsx from
scratch** every run -- one sheet per animal, one row per session below a DOB/sex/strain/baseline-
weight header block.

Rebuilding from scratch (not incrementally appending) is deliberate: training_log.xlsx itself is a
disposable, always-safe-to-delete generated artifact. Never hand-edit it -- edit
animals_metadata.json/session_manual_entries.csv instead, then re-run this script.

Run with the pybpod-environment interpreter:
    /c/Users/2P-Behav/.conda/envs/pybpod-environment/python.exe build_training_log.py
"""
import ast
import csv
import glob
import json
import math
import os
import sys

import openpyxl
from openpyxl.styles import Alignment, Font
from openpyxl.utils import get_column_letter

_RECORDS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.abspath(os.path.join(_RECORDS_DIR, '..'))
_ANIMALS_METADATA_PATH = os.path.join(_RECORDS_DIR, 'animals_metadata.json')
_SESSION_MANUAL_ENTRIES_PATH = os.path.join(_RECORDS_DIR, 'session_manual_entries.csv')
_OUTPUT_PATH = os.path.join(_RECORDS_DIR, 'training_log.xlsx')

sys.path.insert(0, os.path.join(_PROJECT_DIR, 'tasks', '_wheel_shaping_shared'))
import staircase   # noqa: E402 -- same-project import (records/ -> tasks/_wheel_shaping_shared/),
                    # not the cross-project coupling this module's docstring warns against elsewhere

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

_SIMPLE_GATES_PROTOCOLS = ('stage2_threshold_staircase',)   # which protocols have a
                                                              # stage2_simple_gates_met()-style
                                                              # advancement check worth computing


def _isnan(x):
    return x is None or (isinstance(x, float) and math.isnan(x))


# --- raw CSV parsing -- reconstructs the native per-trial STATE/EVENT/VAL structure -----------------

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


# --- per-trial / per-session classification ----------------------------------------------------

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


def _find_val_backward(trials, session_vals, key):
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


def _find_val_forward(trials, session_vals, key):
    """ Same as _find_val_backward but searches from the FIRST trial forward -- used for a
    session-*start* value. """
    for trial in trials:
        value = _val_float([trial['vals']], key)
        if value is not None:
            return value
    return _val_float([session_vals], key)


def summarize_session(path):
    """ Parses one session CSV and returns a flat dict of everything the training log wants for
    one row, or None if the file has no recognizable protocol name. """
    info, session_vals, trials = parse_session_csv(path)
    protocol_name = info.get('PROTOCOL-NAME')
    if not protocol_name:
        return None
    config = PROTOCOL_CONFIG.get(protocol_name, {})

    subject = parse_subject_name(info.get('SUBJECT-NAME', ''))
    session_started = info.get('SESSION-STARTED', '')
    date = session_started.split(' ')[0] if session_started else ''

    # A trailing trial with no STATE rows at all means the session was stopped/killed mid-hold,
    # before that trial's Bpod machine ever ran (confirmed on a real bench-test log) -- not a
    # genuine attempted trial, excluded from trial_count and all the tallies below.
    real_trials = [t for t in trials if t['states']]

    reward_count = withheld_count = no_movement_count = 0
    l_count = r_count = lick_count = 0
    reward_durations = []

    for trial in real_trials:
        outcome, side, reward_duration = classify_trial(trial, config)
        if outcome == 'rewarded':
            reward_count += 1
            if reward_duration is not None:
                reward_durations.append(reward_duration)
        elif outcome == 'withheld':
            withheld_count += 1
        elif outcome == 'no_movement':
            no_movement_count += 1
        if side == 'L':
            l_count += 1
        elif side == 'R':
            r_count += 1
        lick_count += len(trial['events'].get('Port1In', []))

    all_event_times = [t for trial in real_trials for times in trial['events'].values() for t in times]
    session_duration_s = (max(all_event_times) - min(all_event_times)) if all_event_times else None

    threshold_start_deg = _find_val_forward(real_trials, session_vals, 'THRESHOLD_DEG')
    threshold_end_deg = _find_val_backward(real_trials, session_vals, 'THRESHOLD_DEG')
    threshold_final_deg = 35.0   # matches every stage script's VAR_THRESHOLD_FINAL_DEG
    gain_mult_end = _find_val_backward(real_trials, session_vals, 'GAIN_MULT')
    direction_ratio_end = _find_val_backward(real_trials, session_vals, 'DIRECTION_RATIO')

    # ITI is a native STATE (the 'ITI' state's own logged duration), not a custom VAL row --
    # constant within a session in both stages, so any visited trial's value works; take the last.
    iti_end_s = None
    for trial in reversed(real_trials):
        if 'ITI' in trial['states'] and not _isnan(trial['states']['ITI'][0]):
            iti_end_s = trial['states']['ITI'][2]
            break

    simple_gates_met = None
    if protocol_name in _SIMPLE_GATES_PROTOCOLS and iti_end_s is not None:
        simple_gates_met = staircase.stage2_simple_gates_met(
            trial_count=len(real_trials), iti_s=iti_end_s,
            direction_ratio_in_band=(direction_ratio_end is not None and
                                      0.30 <= direction_ratio_end <= 0.70))

    return {
        'subject': subject,
        'date': date,
        'session_started': session_started,
        'protocol': protocol_name,
        'trial_count': len(real_trials),
        'session_duration_s': session_duration_s,
        'reward_count': reward_count,
        'withheld_count': withheld_count,
        'no_movement_count': no_movement_count,
        'l_count': l_count,
        'r_count': r_count,
        'lick_count': lick_count,
        'reward_duration_s': (sum(reward_durations) / len(reward_durations)
                               if reward_durations else None),
        'threshold_start_deg': threshold_start_deg,
        'threshold_end_deg': threshold_end_deg,
        'threshold_final_deg': threshold_final_deg,
        'iti_end_s': iti_end_s,
        'gain_mult_end': gain_mult_end,
        'direction_ratio_end': direction_ratio_end,
        'simple_gates_met': simple_gates_met,
    }


# --- hand-maintained source files -----------------------------------------------------------------

def load_animals_metadata():
    if not os.path.exists(_ANIMALS_METADATA_PATH):
        return {}
    with open(_ANIMALS_METADATA_PATH, 'r') as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if not k.startswith('_')}


def load_session_manual_entries():
    """ Returns {(subject, date): weight_g}. """
    entries = {}
    if not os.path.exists(_SESSION_MANUAL_ENTRIES_PATH):
        return entries
    with open(_SESSION_MANUAL_ENTRIES_PATH, 'r', newline='') as f:
        for row in csv.DictReader(f):
            weight = _parse_float(row.get('weight_g'))
            if weight is not None:
                entries[(row.get('subject'), row.get('date'))] = weight
    return entries


# --- scan every session ------------------------------------------------------------------------

def scan_all_sessions():
    """ Returns {subject: [session_summary, ...]}, each subject's list sorted by session_started. """
    pattern = os.path.join(_PROJECT_DIR, 'experiments', '*', 'setups', '*', 'sessions', '*', '*.csv')
    by_subject = {}
    for path in sorted(glob.glob(pattern)):
        try:
            summary = summarize_session(path)
        except Exception as err:
            print("WARNING: failed to parse {0}: {1}".format(path, err), flush=True)
            continue
        if summary is None:
            continue
        by_subject.setdefault(summary['subject'], []).append(summary)
    for sessions in by_subject.values():
        sessions.sort(key=lambda s: s['session_started'])
    return by_subject


# --- workbook building --------------------------------------------------------------------------

COLUMNS = [
    ('date', 'Date'),
    ('protocol', 'Protocol / Stage'),
    ('trial_count', 'Trials'),
    ('session_duration_s', 'Duration (s)'),
    ('reward_count', 'Rewards'),
    ('withheld_count', 'Withheld'),
    ('no_movement_count', 'No-movement'),
    ('l_count', 'L turns'),
    ('r_count', 'R turns'),
    ('lick_count', 'Licks'),
    ('reward_duration_s', 'Reward duration (s)'),
    ('threshold_start_deg', 'Threshold start (deg)'),
    ('threshold_end_deg', 'Threshold end (deg)'),
    ('threshold_pct_final', 'Threshold end (% final)'),
    ('iti_end_s', 'ITI (s)'),
    ('gain_mult_end', 'Gain mult (Stage 1)'),
    ('direction_ratio_end', 'Direction ratio (Stage 2)'),
    ('simple_gates_met', 'Simple gates met (Stage 2)'),
    ('weight_g', 'Weight (g)'),
    ('weight_pct', 'Weight %'),
    ('notes', 'Notes'),
]

_HEADER_FONT = Font(bold=True)
_TITLE_FONT = Font(bold=True, size=12)


def _write_animal_sheet(wb, subject, sessions, metadata, manual_entries):
    # Excel sheet names can't exceed 31 chars or contain []:*?/\\
    sheet_name = ''.join(c for c in subject if c not in r'[]:*?/\\')[:31] or 'sheet'
    ws = wb.create_sheet(sheet_name)

    meta = metadata.get(subject, {})
    ws['A1'] = 'Animal: {0}'.format(subject)
    ws['A1'].font = _TITLE_FONT
    header_fields = [
        ('DOB', meta.get('dob', '')),
        ('Sex', meta.get('sex', '')),
        ('Strain', meta.get('strain', '')),
        ('Baseline weight (g)', meta.get('baseline_weight_g', '')),
    ]
    baseline_weight_row = None
    for i, (label, value) in enumerate(header_fields):
        row = 2 + i
        ws.cell(row=row, column=1, value=label).font = _HEADER_FONT
        ws.cell(row=row, column=2, value=value)
        if label == 'Baseline weight (g)':
            baseline_weight_row = row

    table_start_row = 2 + len(header_fields) + 1
    for col_idx, (_, label) in enumerate(COLUMNS, start=1):
        cell = ws.cell(row=table_start_row, column=col_idx, value=label)
        cell.font = _HEADER_FONT
        cell.alignment = Alignment(wrap_text=True, vertical='bottom')

    baseline_weight = meta.get('baseline_weight_g')
    weight_col = next(c for c, (k, _) in enumerate(COLUMNS, start=1) if k == 'weight_g')
    baseline_col_letter = get_column_letter(2)   # column B of the header block above

    for row_offset, session in enumerate(sessions, start=1):
        row = table_start_row + row_offset
        weight = manual_entries.get((subject, session['date']))
        row_data = dict(session)
        row_data['weight_g'] = weight
        row_data['notes'] = ''
        if session['threshold_end_deg'] is not None and session['threshold_final_deg']:
            row_data['threshold_pct_final'] = session['threshold_end_deg'] / session['threshold_final_deg']
        else:
            row_data['threshold_pct_final'] = None

        for col_idx, (key, _) in enumerate(COLUMNS, start=1):
            if key == 'weight_pct':
                if weight is not None and baseline_weight and baseline_weight_row:
                    cell = ws.cell(row=row, column=col_idx)
                    cell.value = '={0}{1}/${2}${3}'.format(
                        get_column_letter(weight_col), row, baseline_col_letter, baseline_weight_row)
                    cell.number_format = '0.0%'
                continue
            value = row_data.get(key)
            if key == 'threshold_pct_final' and value is not None:
                cell = ws.cell(row=row, column=col_idx, value=value)
                cell.number_format = '0%'
            else:
                ws.cell(row=row, column=col_idx, value=value)

    for col_idx in range(1, len(COLUMNS) + 1):
        ws.column_dimensions[get_column_letter(col_idx)].width = 14
    ws.freeze_panes = ws.cell(row=table_start_row + 1, column=1).coordinate


def build_workbook():
    by_subject = scan_all_sessions()
    metadata = load_animals_metadata()
    manual_entries = load_session_manual_entries()

    wb = openpyxl.Workbook()
    wb.remove(wb.active)   # drop the default blank sheet -- one real sheet per subject instead

    if not by_subject:
        ws = wb.create_sheet('No sessions found')
        ws['A1'] = 'No session CSVs found under experiments/*/setups/*/sessions/'
    else:
        for subject in sorted(by_subject):
            _write_animal_sheet(wb, subject, by_subject[subject], metadata, manual_entries)

    wb.save(_OUTPUT_PATH)
    total_sessions = sum(len(v) for v in by_subject.values())
    print("Wrote {0} -- {1} animal(s), {2} session(s) total.".format(
        _OUTPUT_PATH, len(by_subject), total_sessions), flush=True)


if __name__ == '__main__':
    build_workbook()
