# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Scans every session CSV under experiments/*/setups/*/sessions/ **on this machine** and
update-in-place merges the results into training_log.xlsx -- one sheet per animal, one row per
session below a hand-editable DOB/Sex/Strain/Baseline-weight header block.

**Update-in-place, not rebuild-from-scratch** -- this file is meant to be shared across multiple
behavior-box computers via git (each box sees only its own local session data for its own
animal(s)). A run on this machine:
  - loads the existing training_log.xlsx if present (e.g. just `git pull`-ed),
  - touches ONLY the sheet(s) for subjects it found sessions for locally,
  - leaves every other subject's sheet in the workbook byte-for-byte untouched (safe for Box A to
    run this without ever seeing, let alone clobbering, Box B's/C's animals),
  - matches sessions to existing rows by their own SESSION-STARTED timestamp (a hidden key column)
    and updates the auto-computed columns in place if a row already exists, appends a new row
    otherwise,
  - NEVER touches the `Weight (g)`/`Notes` columns of an existing row, or the DOB/Sex/Strain/
    Baseline-weight header cells once they've been hand-filled -- those are the genuinely manual,
    hand-typed-in-Excel fields (there are no longer separate animals_metadata.json/
    session_manual_entries.csv files -- fold their contents directly into the xlsx cells instead,
    since the xlsx is the one non-code file this repo commits).

Parses the native Bpod STATE/EVENT/VAL structure directly out of the raw CSV rows (see
CLAUDE.md's "native-STATE/EVENT-parsing" note) via the shared `session_csv_parser` module (also
used by `session_struct_export.py`).

Workflow per box, per day: `git pull` -> run this script -> hand-fill any new blank `Weight (g)`
cells for today's sessions -> `git add training_log.xlsx && git commit && git push`.

Run with the pybpod-environment interpreter:
    /c/Users/2P-Behav/.conda/envs/pybpod-environment/python.exe build_training_log.py
"""
import glob
import math
import os
import sys

import openpyxl
from openpyxl.styles import Alignment, Font
from openpyxl.utils import get_column_letter

_RECORDS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.abspath(os.path.join(_RECORDS_DIR, '..'))
_OUTPUT_PATH = os.path.join(_RECORDS_DIR, 'training_log.xlsx')

sys.path.insert(0, os.path.join(_PROJECT_DIR, 'tasks', '_wheel_shaping_shared'))
import session_csv_parser   # noqa: E402 -- same-project import (records/ -> tasks/_wheel_shaping_shared/)
import staircase             # noqa: E402

_SIMPLE_GATES_PROTOCOLS = ('stage2_threshold_staircase',)   # which protocols have a
                                                              # stage2_simple_gates_met()-style
                                                              # advancement check worth computing


def summarize_session(path):
    """ Parses one session CSV and returns a flat dict of everything the training log wants for
    one row, or None if the file has no recognizable protocol name. """
    info, session_vals, trials = session_csv_parser.parse_session_csv(path)
    protocol_name = info.get('PROTOCOL-NAME')
    if not protocol_name:
        return None
    config = session_csv_parser.PROTOCOL_CONFIG.get(protocol_name, {})

    subject = session_csv_parser.parse_subject_name(info.get('SUBJECT-NAME', ''))
    session_started = info.get('SESSION-STARTED', '')
    date = session_started.split(' ')[0] if session_started else ''

    trials = session_csv_parser.real_trials(trials)

    reward_count = withheld_count = no_movement_count = 0
    l_count = r_count = lick_count = 0
    reward_durations = []

    for trial in trials:
        outcome, side, reward_duration = session_csv_parser.classify_trial(trial, config)
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

    all_event_times = [t for trial in trials for times in trial['events'].values() for t in times]
    session_duration_s = (max(all_event_times) - min(all_event_times)) if all_event_times else None

    threshold_start_deg = session_csv_parser.find_val_forward(trials, session_vals, 'THRESHOLD_DEG')
    threshold_end_deg = session_csv_parser.find_val_backward(trials, session_vals, 'THRESHOLD_DEG')
    threshold_final_deg = 35.0   # matches every stage script's VAR_THRESHOLD_FINAL_DEG
    gain_mult_end = session_csv_parser.find_val_backward(trials, session_vals, 'GAIN_MULT')
    direction_ratio_end = session_csv_parser.find_val_backward(trials, session_vals, 'DIRECTION_RATIO')

    # ITI is a native STATE (the 'ITI' state's own logged duration), not a custom VAL row --
    # constant within a session in both stages, so any visited trial's value works; take the last.
    iti_end_s = None
    for trial in reversed(trials):
        if 'ITI' in trial['states']:
            start = trial['states']['ITI'][0]
            if not (start is None or math.isnan(start)):
                iti_end_s = trial['states']['ITI'][2]
                break

    simple_gates_met = None
    if protocol_name in _SIMPLE_GATES_PROTOCOLS and iti_end_s is not None:
        simple_gates_met = staircase.stage2_simple_gates_met(
            trial_count=len(trials), iti_s=iti_end_s,
            direction_ratio_in_band=(direction_ratio_end is not None and
                                      0.30 <= direction_ratio_end <= 0.70))

    struct_mat_path = os.path.splitext(path)[0] + '_struct.mat'
    session_struct_path = struct_mat_path if os.path.exists(struct_mat_path) else None

    return {
        'session_started': session_started,
        'subject': subject,
        'date': date,
        'protocol': protocol_name,
        'trial_count': len(trials),
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
        'session_csv_path': path,
        'session_struct_path': session_struct_path,
    }


# --- scan every session on THIS machine ----------------------------------------------------------

def scan_all_sessions():
    """ Returns {subject: [session_summary, ...]}, each subject's list sorted by session_started.
    Only scans this machine's own local experiments/ folder -- inherently per-box. """
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


# --- workbook update-in-place --------------------------------------------------------------------

# ('internal key', 'column header label'). Order only matters for brand-new sheets/columns --
# existing sheets are matched by label text (see _get_column_map), so adding a new entry here
# later just appends a new column to every sheet on its next update, no migration needed.
COLUMNS = [
    ('session_started', 'Session started'),   # hidden key column -- how existing rows are matched
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
    ('session_csv_path', 'Session data (local path)'),
    ('session_struct_path', 'Session struct (local path)'),
    ('weight_g', 'Weight (g)'),      # MANUAL -- the script never writes to an existing row's cell
    ('weight_pct', 'Weight %'),      # auto formula, refreshed every run
    ('notes', 'Notes'),              # MANUAL -- the script never writes to an existing row's cell
]
_MANUAL_COLUMNS = {'weight_g', 'notes'}

_HEADER_FIELD_LABELS = ['DOB', 'Sex', 'Strain', 'Baseline weight (g)']
_BASELINE_WEIGHT_ROW = 2 + _HEADER_FIELD_LABELS.index('Baseline weight (g)')   # = 5
_BASELINE_WEIGHT_COL_LETTER = 'B'

_HEADER_FONT = Font(bold=True)
_TITLE_FONT = Font(bold=True, size=12)


def _sheet_name_for(subject):
    return ''.join(c for c in subject if c not in r'[]:*?/\\')[:31] or 'sheet'


def _get_or_create_subject_sheet(wb, subject):
    """ Returns (ws, table_start_row). If the sheet already exists, its DOB/Sex/Strain/Baseline
    weight header cells are left completely untouched (may already be hand-filled). """
    sheet_name = _sheet_name_for(subject)
    table_start_row = 2 + len(_HEADER_FIELD_LABELS) + 1
    if sheet_name in wb.sheetnames:
        return wb[sheet_name], table_start_row

    ws = wb.create_sheet(sheet_name)
    ws['A1'] = 'Animal: {0}'.format(subject)
    ws['A1'].font = _TITLE_FONT
    for i, label in enumerate(_HEADER_FIELD_LABELS):
        ws.cell(row=2 + i, column=1, value=label).font = _HEADER_FONT
        # value cell (column B) deliberately left blank for hand-entry
    return ws, table_start_row


def _get_column_map(ws, table_start_row):
    """ Scans the existing header row for known column labels; any COLUMNS entry not already
    present gets appended as a new column at the end. This is what lets the schema grow (e.g.
    this run's new session_started/session_csv_path/session_struct_path columns) without
    disturbing an existing sheet's already-populated columns/rows at all. """
    label_to_key = {label: key for key, label in COLUMNS}
    col_map = {}
    max_used_col = 0
    for col_idx in range(1, ws.max_column + 1):
        value = ws.cell(row=table_start_row, column=col_idx).value
        if value:
            max_used_col = col_idx
        if value in label_to_key:
            col_map[label_to_key[value]] = col_idx

    next_col = max_used_col + 1
    for key, label in COLUMNS:
        if key not in col_map:
            cell = ws.cell(row=table_start_row, column=next_col, value=label)
            cell.font = _HEADER_FONT
            cell.alignment = Alignment(wrap_text=True, vertical='bottom')
            col_map[key] = next_col
            next_col += 1
    return col_map


def _existing_rows_by_key(ws, table_start_row, key_col):
    mapping = {}
    for row in range(table_start_row + 1, ws.max_row + 1):
        value = ws.cell(row=row, column=key_col).value
        if value:
            mapping[value] = row
    return mapping


def _write_row(ws, row, col_map, session):
    row_data = dict(session)
    if session.get('threshold_end_deg') is not None and session.get('threshold_final_deg'):
        row_data['threshold_pct_final'] = session['threshold_end_deg'] / session['threshold_final_deg']
    else:
        row_data['threshold_pct_final'] = None

    for key, _ in COLUMNS:
        if key in _MANUAL_COLUMNS or key == 'weight_pct':
            continue   # manual columns: never written here. weight_pct: formula, handled below.
        col_idx = col_map[key]
        value = row_data.get(key)
        cell = ws.cell(row=row, column=col_idx, value=value)
        if key == 'threshold_pct_final' and value is not None:
            cell.number_format = '0%'

    # weight_pct is a formula (not a hand-typed value), safe to refresh every run regardless of
    # whether the row is new or existing -- it only ever reads weight_g/baseline, never writes them.
    weight_col_letter = get_column_letter(col_map['weight_g'])
    pct_cell = ws.cell(row=row, column=col_map['weight_pct'])
    pct_cell.value = '=IFERROR({0}{1}/${2}${3},"")'.format(
        weight_col_letter, row, _BASELINE_WEIGHT_COL_LETTER, _BASELINE_WEIGHT_ROW)
    pct_cell.number_format = '0.0%'


def _upsert_subject(wb, subject, sessions):
    ws, table_start_row = _get_or_create_subject_sheet(wb, subject)
    col_map = _get_column_map(ws, table_start_row)
    existing_by_key = _existing_rows_by_key(ws, table_start_row, col_map['session_started'])
    next_append_row = (ws.max_row + 1) if ws.max_row >= table_start_row else (table_start_row + 1)

    new_count = updated_count = 0
    for session in sessions:
        key = session['session_started']
        if not key:
            continue   # can't track without a key -- shouldn't happen for a real session
        if key in existing_by_key:
            row = existing_by_key[key]
            updated_count += 1
        else:
            row = next_append_row
            next_append_row += 1
            new_count += 1
        _write_row(ws, row, col_map, session)

    for col_idx in col_map.values():
        ws.column_dimensions[get_column_letter(col_idx)].width = 14
    ws.column_dimensions[get_column_letter(col_map['session_started'])].hidden = True
    ws.freeze_panes = ws.cell(row=table_start_row + 1, column=1).coordinate
    return new_count, updated_count


def build_workbook():
    by_subject = scan_all_sessions()
    if not by_subject:
        print("No session CSVs found under experiments/*/setups/*/sessions/ on this machine -- "
              "nothing to update.", flush=True)
        return

    if os.path.exists(_OUTPUT_PATH):
        wb = openpyxl.load_workbook(_OUTPUT_PATH)
    else:
        wb = openpyxl.Workbook()
        wb.remove(wb.active)   # drop the default blank sheet -- one real sheet per subject instead

    total_new = total_updated = 0
    for subject in sorted(by_subject):
        new_count, updated_count = _upsert_subject(wb, subject, by_subject[subject])
        total_new += new_count
        total_updated += updated_count
        print("  {0}: {1} new row(s), {2} updated row(s)".format(subject, new_count, updated_count),
              flush=True)

    wb.save(_OUTPUT_PATH)
    print("Wrote {0} -- {1} new row(s), {2} updated row(s) across {3} subject(s) found on this "
          "machine. Every other subject's sheet in the workbook was left untouched.".format(
              _OUTPUT_PATH, total_new, total_updated, len(by_subject)), flush=True)


if __name__ == '__main__':
    build_workbook()
