# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Scans every session CSV under experiments/*/setups/*/sessions/ **on this machine** and
update-in-place merges the results into training_log.xlsx -- one sheet per animal, sessions
grouped into contiguous per-stage blocks (Stage 1's rows together, then Stage 2's, ...) below a
hand-editable DOB/Sex/Strain/Baseline-weight header block.

**Session merging**: a session that got Stopped, Killed, or crashed partway through and was
restarted within an hour (same protocol) is combined into ONE row + one combined data-output file
-- see `group_sessions()`. A session that reached its own natural end (`VAR_MAX_TRIALS`) is never
merged with what follows. Applies to every subject uniformly, including bench-test ones.

**Update-in-place, not rebuild-from-scratch** -- this file is meant to be shared across multiple
behavior-box computers via git (each box sees only its own local session data for its own
animal(s)). A run on this machine:
  - loads the existing training_log.xlsx if present (e.g. just `git pull`-ed),
  - touches ONLY the sheet(s) for subjects it found sessions for locally,
  - leaves every other subject's sheet in the workbook byte-for-byte untouched (safe for Box A to
    run this without ever seeing, let alone clobbering, Box B's/C's animals),
  - regenerates its own subject's data rows fresh every run from (current local scan + whatever
    Weight (g)/Notes values were already hand-typed into the sheet) -- necessary because merging
    can change which rows exist between runs (a restart appearing later can absorb a session that
    used to stand alone), and because rows are grouped into per-stage sections rather than a flat
    append-only list, so a newly-appearing stage's rows may need to land earlier in the sheet than
    rows already there. See `_rewrite_subject_sheet()`.
  - NEVER touches the `Weight (g)`/`Notes` values of a row that already had them, or the
    DOB/Sex/Strain/Baseline-weight header cells once they've been hand-filled -- those are the
    genuinely manual, hand-typed-in-Excel fields.

Parses the native Bpod STATE/EVENT/VAL structure directly out of the raw CSV rows (see
CLAUDE.md's "native-STATE/EVENT-parsing" note) via the shared `session_csv_parser` module (also
used by `session_struct_export.py`).

Workflow per box, per day: `git pull` -> run this script -> hand-fill any new blank `Weight (g)`
cells for today's sessions -> `git add training_log.xlsx && git commit && git push`.

Run with the pybpod-environment interpreter:
    /c/Users/2P-Behav/.conda/envs/pybpod-environment/python.exe build_training_log.py
"""
import glob
import json
import math
import os
import re
import sys
import threading

import openpyxl
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

_RECORDS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.abspath(os.path.join(_RECORDS_DIR, '..'))
_OUTPUT_PATH = os.path.join(_RECORDS_DIR, 'training_log.xlsx')

# {raw PyBpod subject name: canonical animal ID} -- lets several raw subject identities display/
# group as ONE sheet without ever touching the underlying session CSVs' own historical SUBJECT-NAME
# record (editing already-recorded raw data would be data falsification, not a fix). Deliberately
# COMMITTED (unlike most of records/, which is machine-specific/gitignored) -- training_log.xlsx
# itself is the one shared/synced file across boxes, so how its rows get grouped needs to be
# consistent everywhere it's regenerated, not a local per-machine choice. Missing file/key is a
# plain no-op (every raw subject name maps to itself, today's behavior for everyone until an alias
# is actually added).
_SUBJECT_ALIASES_PATH = os.path.join(_RECORDS_DIR, 'subject_aliases.json')


def _load_subject_aliases():
    try:
        with open(_SUBJECT_ALIASES_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (IOError, OSError, ValueError):
        return {}

sys.path.insert(0, os.path.join(_PROJECT_DIR, 'tasks', '_wheel_shaping_shared'))
import session_csv_parser     # noqa: E402 -- same-project import (records/ -> tasks/_wheel_shaping_shared/)
from water_tracker import ConsumedWaterTracker   # noqa: E402 -- same folder as session_csv_parser
import session_struct_export  # noqa: E402
import staircase               # noqa: E402

_SIMPLE_GATES_PROTOCOLS = ('stage2_threshold_staircase', 'stage3_tone_side_association',
                            'stage4_clicks_direction', 'stage5_resume_staircases')
                           # which protocols have an auto-checkable advancement-gate function
                           # worth computing -- see _compute_simple_gates_met()'s own per-protocol
                           # dispatch (each of these uses a DIFFERENT staircase.py
                           # function/signature).

_MERGE_GAP = 3600.0   # seconds -- restarts within this long of a non-'completed' session's own end
                       # are combined into one row/struct with it (same protocol only)


def _compute_simple_gates_met(protocol_name, trial_count, iti_end_s, direction_ratio_end,
                               accuracy_aos, abort_rate, response_threshold_end_deg,
                               threshold_final_deg, quiescence_end_s):
    """ Dispatches to whichever staircase.py gate-check function matches this protocol -- each of
    _SIMPLE_GATES_PROTOCOLS uses a genuinely different function/signature (Stage 2's is
    deliberately partial, deferring to a human for its real statistical criterion; Stage 3/4/5's
    each cover their WHOLE doc-specified criterion). Returns None if the protocol isn't one of
    these four, or if a required input is missing (e.g. no ITI-bearing trial found yet). """
    if protocol_name == 'stage2_threshold_staircase':
        if iti_end_s is None:
            return None
        return staircase.stage2_simple_gates_met(
            trial_count=trial_count, iti_s=iti_end_s,
            direction_ratio_in_band=(direction_ratio_end is not None
                                      and 0.30 <= direction_ratio_end <= 0.70))
    if protocol_name == 'stage3_tone_side_association':
        return staircase.stage3_gates_met(accuracy_aos or 0.0, abort_rate or 0.0, trial_count)
    if protocol_name == 'stage4_clicks_direction':
        return staircase.stage4_gates_met(accuracy_aos or 0.0, abort_rate or 0.0, trial_count)
    if protocol_name == 'stage5_resume_staircases':
        if response_threshold_end_deg is None or quiescence_end_s is None:
            return None
        response_frac = response_threshold_end_deg / threshold_final_deg
        return staircase.stage5_gates_met(response_frac, quiescence_end_s, accuracy_aos or 0.0,
                                           abort_rate or 0.0, trial_count)
    return None


def _minutes(seconds):
    return round(seconds / 60.0, 1) if seconds is not None else None


def summarize_session(path, parsed=None):
    """ Parses one session CSV and returns a flat dict of everything the training log wants for
    one row, or None if the file has no recognizable protocol name OR the protocol has no
    PROTOCOL_CONFIG entry. The latter matters now that scan_all_sessions() also scans the Tests
    project (see its own docstring) -- that folder has many bench-test protocols (dot_wheel_test,
    camera_test, lick_reward, ...) with no meaningful outcome taxonomy to report; only protocols
    explicitly configured (currently: stage1-5 and full_protocol_lookback_test) generate a row, so
    the log doesn't fill up with all-'unknown'-outcome noise from every bench script ever run.

    `parsed` = an already-parsed (info, session_vals, trials) for this file (e.g. from the live
    dashboard's IncrementalSessionParser), to avoid reading the whole CSV again. """
    info, session_vals, trials = (parsed if parsed is not None
                                  else session_csv_parser.parse_session_csv(path))
    protocol_name = info.get('PROTOCOL-NAME')
    if not protocol_name or protocol_name not in session_csv_parser.PROTOCOL_CONFIG:
        return None
    config = session_csv_parser.PROTOCOL_CONFIG[protocol_name]

    subject = session_csv_parser.parse_subject_name(info.get('SUBJECT-NAME', ''))
    subject = _load_subject_aliases().get(subject, subject)   # see _SUBJECT_ALIASES_PATH's own
                                                                 # docstring above
    session_started = info.get('SESSION-STARTED', '')
    date = session_started.split(' ')[0] if session_started else ''

    trials = session_csv_parser.real_trials(trials)

    reward_count = withheld_count = no_movement_count = 0
    incorrect_count = wheel_abort_count = 0
    l_count = r_count = lick_count = consumed_reward_count = 0

    # Stage 3/4/5 only -- their own PROTOCOL_CONFIG entries produce 'incorrect'/'aborted' outcomes
    # (see session_csv_parser.PROTOCOL_CONFIG), so these stay 0 for Stage 1/2's rows, same as
    # incorrect_count/wheel_abort_count above already do.
    warmup_trial_count = repeat_trial_count = 0
    main_trial_total = main_trial_correct_count = 0
    aborts_quiescence = aborts_cue = aborts_delay = aborts_response = 0

    # Consumed water uses the same rule as the Stage 1/2 water cap (water_tracker.py): a drop counts
    # once the animal licks after it was delivered, even if that lick lands in a later trial.
    water = ConsumedWaterTracker(1.0)   # counts drops; scaled by reward_ul below
    for trial in trials:
        outcome, side, _reward_duration, consumed = session_csv_parser.classify_trial(trial, config)
        reward_start = None
        if outcome == 'rewarded':
            for st in config.get('rewarded_states', ()):
                occ = trial['states'].get(st)
                occ = occ[-1] if occ and isinstance(occ[0], (list, tuple)) else occ
                if occ and not session_csv_parser._isnan(occ[0]):
                    reward_start = occ[0]
                    break
        water.record_trial(trial['events'].get('Port1In', []), reward_start)
        trial_type = trial['vals'].get('TRIAL_TYPE')   # Stage 3/4/5's own custom VAL -- absent (None)
                                                          # for Stage 1/2 and full_protocol_lookback_test
        if outcome == 'rewarded':
            reward_count += 1
            if consumed:
                consumed_reward_count += 1
            if trial_type == 'main':
                main_trial_total += 1
                main_trial_correct_count += 1
        elif outcome == 'withheld':
            withheld_count += 1
        elif outcome == 'no_movement':
            no_movement_count += 1
        elif outcome == 'incorrect':
            incorrect_count += 1
            if trial_type == 'main':
                main_trial_total += 1
        elif outcome == 'aborted':
            wheel_abort_count += 1
            epoch = trial['vals'].get('ABORT_EPOCH')
            if epoch == 'quiescence':
                aborts_quiescence += 1
            elif epoch == 'cue':
                aborts_cue += 1
            elif epoch == 'delay':
                aborts_delay += 1
            elif epoch == 'response':
                aborts_response += 1
        if side == 'L':
            l_count += 1
        elif side == 'R':
            r_count += 1
        lick_count += len(session_csv_parser.trial_licks(trial))   # chatter merged

        if trial_type == 'warmup':
            warmup_trial_count += 1
        elif trial_type == 'repeat':
            repeat_trial_count += 1

    accuracy_aos = (main_trial_correct_count / float(main_trial_total)
                    if main_trial_total else None)
    abort_rate = (wheel_abort_count / float(len(trials))) if trials else None

    # Session length on the Bpod clock: first trial's start -> last trial's end (TRIAL-BPOD-TIME).
    # It used to be max - min over every trial's EVENT times, but those are TRIAL-relative, so it
    # came out as roughly the longest single trial (7.5 s for a 5-minute session).
    session_duration_s = None
    starts = [t['bpod_start'] for t in trials if t.get('bpod_start') is not None]
    ends = [t['bpod_end'] for t in trials if t.get('bpod_end') is not None]
    if starts and ends and max(ends) > min(starts):
        session_duration_s = max(ends) - min(starts)
    else:   # older sessions without TRIAL-BPOD-TIME rows: the session's own start/end stamps
        _t0 = session_csv_parser.parse_datetime(info.get('SESSION-STARTED', ''))
        _t1 = session_csv_parser.parse_datetime(info.get('SESSION-ENDED', ''))
        if _t0 is not None and _t1 is not None and _t1 > _t0:
            session_duration_s = (_t1 - _t0).total_seconds()

    threshold_start_deg = session_csv_parser.find_val_forward(trials, session_vals, 'THRESHOLD_DEG')
    threshold_end_deg = session_csv_parser.find_val_backward(trials, session_vals, 'THRESHOLD_DEG')
    threshold_final_deg = 35.0   # matches every stage script's VAR_THRESHOLD_FINAL_DEG
    gain_mult_end = session_csv_parser.find_val_backward(trials, session_vals, 'GAIN_MULT')
    direction_ratio_end = session_csv_parser.find_val_backward(trials, session_vals, 'DIRECTION_RATIO')
    reward_ul = session_csv_parser.find_val_backward(trials, session_vals, 'REWARD_UL')
    aborts = session_csv_parser.sum_val(trials, 'QUIESCENCE_BREAKS')
    session_end_reason = session_csv_parser.find_val_str_backward(trials, session_vals,
                                                                    'SESSION_END_REASON')

    # Stage 3/4/5 only -- all from custom VALs those protocols' own task scripts register;
    # simply absent (None) for every other protocol.
    response_threshold_start_deg = session_csv_parser.find_val_forward(
        trials, session_vals, 'RESPONSE_THRESHOLD_DEG')
    response_threshold_end_deg = session_csv_parser.find_val_backward(
        trials, session_vals, 'RESPONSE_THRESHOLD_DEG')
    quiescence_start_s = session_csv_parser.find_val_forward(trials, session_vals, 'QUIESCENCE_DUR_S')
    quiescence_end_s = session_csv_parser.find_val_backward(trials, session_vals, 'QUIESCENCE_DUR_S')
    cue_abort_threshold_deg = session_csv_parser.find_val_backward(
        trials, session_vals, 'CUE_ABORT_THRESHOLD_DEG')
    _click_level_attenuated_raw = session_csv_parser.find_val_str_backward(
        trials, session_vals, 'CLICK_LEVEL_ATTENUATED')
    click_level_attenuated = (_click_level_attenuated_raw == 'True'
                               if _click_level_attenuated_raw is not None else None)
    session_water_ul = session_csv_parser.find_val_backward(trials, session_vals, 'SESSION_WATER_UL')
    staircase_advanced_this_session = session_csv_parser.find_val_str_backward(
        trials, session_vals, 'STAIRCASE_ACTIVE_THIS_SESSION')
    git_commit = session_csv_parser.find_val_str_backward(trials, session_vals, 'GIT_COMMIT')
    target_spl_db = session_csv_parser.find_val_backward(trials, session_vals, 'TARGET_SPL_DB')
    left_freq_hz = session_csv_parser.find_val_backward(trials, session_vals, 'LEFT_FREQ_HZ')
    right_freq_hz = session_csv_parser.find_val_backward(trials, session_vals, 'RIGHT_FREQ_HZ')

    # ITI is a native STATE (the 'ITI' state's own logged duration), not a custom VAL row --
    # constant within a session in both stages, so any visited trial's value works; take the last.
    iti_end_s = None
    for trial in reversed(trials):
        if 'ITI' in trial['states']:
            start = trial['states']['ITI'][0]
            if not (start is None or math.isnan(start)):
                iti_end_s = trial['states']['ITI'][2]
                break

    simple_gates_met = _compute_simple_gates_met(
        protocol_name, len(trials), iti_end_s, direction_ratio_end, accuracy_aos, abort_rate,
        response_threshold_end_deg, threshold_final_deg, quiescence_end_s)

    started_dt = session_csv_parser.parse_datetime(session_started)
    ended_dt = session_csv_parser.session_end_datetime(info, started_dt, session_duration_s)
    time_of_day = started_dt.strftime('%H:%M') if started_dt is not None else ''

    consumed_volume_ul = (reward_ul * water.consumed_ul) if reward_ul is not None else None
    delivered_volume_ul = (reward_ul * water.delivered_ul) if reward_ul is not None else None

    struct_mat_path = os.path.splitext(path)[0] + '_struct.mat'
    session_struct_path = struct_mat_path if os.path.exists(struct_mat_path) else None

    return {
        'session_started': session_started,
        'started_dt': started_dt,
        'ended_dt': ended_dt,
        'session_end_reason': session_end_reason,
        'subject': subject,
        'date': date,
        'time_of_day': time_of_day,
        'protocol': protocol_name,
        'trial_count': len(trials),
        'session_duration_s': session_duration_s,
        'session_duration_min': _minutes(session_duration_s),
        'reward_count': reward_count,
        'consumed_reward_count': consumed_reward_count,
        'withheld_count': withheld_count,
        'no_movement_count': no_movement_count,
        'incorrect_count': incorrect_count,
        'wheel_abort_count': wheel_abort_count,
        'l_count': l_count,
        'r_count': r_count,
        'lick_count': lick_count,
        'aborts': aborts,
        'reward_ul': reward_ul,
        'consumed_volume_ul': consumed_volume_ul,
        'delivered_volume_ul': delivered_volume_ul,
        'threshold_start_deg': threshold_start_deg,
        'threshold_end_deg': threshold_end_deg,
        'threshold_final_deg': threshold_final_deg,
        'iti_end_s': iti_end_s,
        'gain_mult_end': gain_mult_end,
        'direction_ratio_end': direction_ratio_end,
        'simple_gates_met': simple_gates_met,
        'response_threshold_start_deg': response_threshold_start_deg,
        'response_threshold_end_deg': response_threshold_end_deg,
        'quiescence_start_s': quiescence_start_s,
        'quiescence_end_s': quiescence_end_s,
        'cue_abort_threshold_deg': cue_abort_threshold_deg,
        'click_level_attenuated': click_level_attenuated,
        'accuracy_aos': accuracy_aos,
        'abort_rate': abort_rate,
        'aborts_quiescence': aborts_quiescence,
        'aborts_cue': aborts_cue,
        'aborts_delay': aborts_delay,
        'aborts_response': aborts_response,
        'warmup_trial_count': warmup_trial_count,
        'repeat_trial_count': repeat_trial_count,
        'session_water_ul': session_water_ul,
        'staircase_advanced_this_session': staircase_advanced_this_session,
        'git_commit': git_commit,
        'target_spl_db': target_spl_db,
        'left_freq_hz': left_freq_hz,
        'right_freq_hz': right_freq_hz,
        'session_csv_path': path,
        'session_struct_path': session_struct_path,
    }


# --- scan every session on THIS machine ----------------------------------------------------------

_SCAN_PROJECT_DIRS = [
    _PROJECT_DIR,                                          # AuditoryEvidenceAccum -- real training
    os.path.join(_PROJECT_DIR, '..', 'Tests'),              # bench-test sessions -- see
                                                             # summarize_session()'s own docstring
                                                             # for why only EXPLICITLY configured
                                                             # protocols from here generate a row
]


def _session_csv_paths():
    for project_dir in _SCAN_PROJECT_DIRS:
        pattern = os.path.join(project_dir, 'experiments', '*', 'setups', '*', 'sessions', '*',
                                '*.csv')
        for path in sorted(glob.glob(pattern)):
            yield path


def _ran_here(csv_path):
    """ True if this session was run on THIS computer. Every launcher (PyBpod GUI's board_com.py,
    run_session.py) writes a session-local user_settings.py into the session folder, and that file
    is gitignored -- so it only ever exists on the computer that actually ran the session, never on
    one that just pulled the session's data from git. """
    return os.path.exists(os.path.join(os.path.dirname(csv_path), 'user_settings.py'))


def _quick_session_header(csv_path):
    """ (subject, session_started) from just the INFO header rows at the top of a session CSV --
    cheap enough to run over every computer's shared sessions on every rebuild/Save. """
    subject = started = None
    with open(csv_path, 'r', encoding='utf-8', errors='replace') as f:
        for i, line in enumerate(f):
            if i > 60 or (subject and started):
                break
            parts = line.rstrip('\r\n').split(';', 5)
            if len(parts) == 6 and parts[0] == 'INFO':
                if parts[4] == 'SUBJECT-NAME':
                    subject = session_csv_parser.parse_subject_name(parts[5])
                elif parts[4] == 'SESSION-STARTED':
                    started = parts[5]
    if subject is not None:
        subject = _load_subject_aliases().get(subject, subject)
    return subject, started or ''


def is_real_animal_subject(name):
    """ True for a real training-animal subject name -- this lab's convention: starts with 'UM'
    (e.g. 'UM001') AND doesn't look like a bench-test/dev subject (contains 'test', e.g. 'test',
    'test1', 'testTask', or a hypothetical 'UMtest'). Per explicit instruction: bench-test
    subjects should not get their own sheet in training_log.xlsx, and their session data should
    never be committed to GitHub via the "Save to Training Log" button (see sync_training_log.py's
    own use of this). """
    return bool(name) and name.upper().startswith('UM') and 'test' not in name.lower()


def session_subject(csv_path):
    """ Public wrapper around _quick_session_header() -- just the subject name (or None),
    for callers (sync_training_log.py) that only need subject identity, not session_started. """
    return _quick_session_header(csv_path)[0]


def owned_subjects():
    """ Subjects this computer is responsible for in training_log.xlsx: those whose MOST RECENT
    session (across every computer's sessions shared via git) ran here, AND which are real
    training animals (is_real_animal_subject()) -- bench-test subjects never get a sheet here,
    per explicit instruction. One mouse is trained on one computer, so each subject's sheet has
    exactly one writer -- no computer ever rewrites another computer's mouse (or its hand-typed
    weights/notes, or its session folders' combined structs). If a mouse moves to another
    computer, ownership follows it as soon as it runs a session there. """
    latest = {}   # subject -> (session_started, ran_here)
    for path in _session_csv_paths():
        try:
            subject, started = _quick_session_header(path)
        except Exception as err:
            print("WARNING: failed to read {0}: {1}".format(path, err), flush=True)
            continue
        if subject is None or not is_real_animal_subject(subject):
            continue
        if subject not in latest or started > latest[subject][0]:
            latest[subject] = (started, _ran_here(path))
    return {subject for subject, (_started, ran_here) in latest.items() if ran_here}


def scan_all_sessions(subjects=None):
    """ Returns {subject: [session_summary, ...]}, each subject's list sorted by session_started.
    Scans the local experiments/ folders (AuditoryEvidenceAccum + Tests, see _SCAN_PROJECT_DIRS),
    which now also hold other computers' sessions pulled via git. `subjects`, if given, limits the
    (expensive) full parse to just those subjects. """
    by_subject = {}
    for path in _session_csv_paths():
        if subjects is not None:
            try:
                if _quick_session_header(path)[0] not in subjects:
                    continue
            except Exception:
                pass   # let summarize_session() below report it
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


# --- session merging (restart-combining) ---------------------------------------------------------

def group_sessions(sessions):
    """ Returns a list of groups (each a chronological list of >=1 session summaries). A session
    whose `session_end_reason` is 'completed' (reached VAR_MAX_TRIALS naturally) always ends a
    group -- it's never combined with whatever runs next. Any other ending (Stop, Kill, or a
    crash -- `session_end_reason` absent) is eligible to merge with the next same-protocol session
    if it starts within `_MERGE_GAP` seconds of this one's own end. Applies to every subject
    uniformly, including bench-test ones -- merging reflects how the sessions actually happened,
    regardless of which GUI subject they were run under. """
    groups = []
    for session in sessions:
        if groups:
            last = groups[-1][-1]
            mergeable = (
                last['protocol'] == session['protocol']
                and last['session_end_reason'] != 'completed'
                and last['ended_dt'] is not None and session['started_dt'] is not None
                and (session['started_dt'] - last['ended_dt']).total_seconds() <= _MERGE_GAP
            )
            if mergeable:
                groups[-1].append(session)
                continue
        groups.append([session])
    return groups


def combine_group(group):
    """ Combines one group (>=1 raw session summaries) into a single row dict. A singleton group
    just carries its one session's values through unchanged; a real merge sums the per-trial
    tallies, including each member's own event-span `session_duration_s` (NOT the wall-clock gap
    between them -- summing active durations means a restart doesn't silently double-count idle
    time, and keeps a singleton group's Duration identical to that one session's own value, no
    behavior change for the common unmerged case), and takes threshold_start from the first
    member / everything else "as of end of session" from the last member (same semantics as a
    single session's own "_end" fields, just now "end of bout"). `session_struct_path` is left
    unset here -- the caller fills it in (single session: that session's own struct path;
    multi-session: the path `session_struct_export.merge_session_structs()` returns, since
    building that file needs filesystem access this pure function deliberately doesn't do). """
    first, last = group[0], group[-1]

    durations = [s['session_duration_s'] for s in group if s['session_duration_s'] is not None]
    duration_s = sum(durations) if durations else None

    consumed_vals = [s['consumed_volume_ul'] for s in group if s['consumed_volume_ul'] is not None]
    delivered_vals = [s.get('delivered_volume_ul') for s in group
                      if s.get('delivered_volume_ul') is not None]
    water_vals = [s['session_water_ul'] for s in group if s['session_water_ul'] is not None]

    # accuracy_aos/abort_rate are session-level RATES, not counts -- can't just sum them across a
    # merged group. Re-derive each from the group's own summed numerator/denominator instead of
    # averaging the per-member rates (correct under unequal trial counts; averaging isn't).
    total_trials = sum(s['trial_count'] for s in group)
    accuracy_nums = [(s['accuracy_aos'], s['trial_count']) for s in group
                      if s['accuracy_aos'] is not None]
    # accuracy_aos's own denominator (MAIN trials with a genuine response) isn't separately
    # tracked per-member here -- approximating it via trial_count is a known imprecision for a
    # merged (restart) group specifically; an un-merged single-session row (the common case) is
    # exact, since there's nothing to weight against.
    accuracy_aos = (sum(a * n for a, n in accuracy_nums) / sum(n for _a, n in accuracy_nums)
                    if accuracy_nums else None)
    abort_rate = ((sum(s['wheel_abort_count'] for s in group) / float(total_trials))
                  if total_trials else None)

    return {
        'session_started': '; '.join(s['session_started'] for s in group),
        'protocol': first['protocol'],
        'date': first['date'],
        'time_of_day': first['time_of_day'],
        # blank (not 1) for an ordinary, un-merged row -- "how many sessions were combined" only
        # means something once an actual merge happened; a bare "1" there would read as if
        # something got combined when nothing did.
        'num_sessions': len(group) if len(group) > 1 else None,
        'trial_count': sum(s['trial_count'] for s in group),
        'session_duration_s': duration_s,
        'session_duration_min': _minutes(duration_s),
        'reward_ul': last['reward_ul'],
        'consumed_volume_ul': sum(consumed_vals) if consumed_vals else None,
        'delivered_volume_ul': sum(delivered_vals) if delivered_vals else None,
        'aborts': sum(s['aborts'] for s in group),
        'withheld_count': sum(s['withheld_count'] for s in group),
        'no_movement_count': sum(s['no_movement_count'] for s in group),
        'incorrect_count': sum(s['incorrect_count'] for s in group),
        'wheel_abort_count': sum(s['wheel_abort_count'] for s in group),
        'l_count': sum(s['l_count'] for s in group),
        'r_count': sum(s['r_count'] for s in group),
        'lick_count': sum(s['lick_count'] for s in group),
        'threshold_start_deg': first['threshold_start_deg'],
        'threshold_end_deg': last['threshold_end_deg'],
        'threshold_final_deg': last['threshold_final_deg'],
        'iti_end_s': last['iti_end_s'],
        'gain_mult_end': last['gain_mult_end'],
        'direction_ratio_end': last['direction_ratio_end'],
        'simple_gates_met': last['simple_gates_met'],
        'response_threshold_start_deg': first['response_threshold_start_deg'],
        'response_threshold_end_deg': last['response_threshold_end_deg'],
        'quiescence_start_s': first['quiescence_start_s'],
        'quiescence_end_s': last['quiescence_end_s'],
        'cue_abort_threshold_deg': last['cue_abort_threshold_deg'],
        'click_level_attenuated': last['click_level_attenuated'],
        'accuracy_aos': accuracy_aos,
        'abort_rate': abort_rate,
        'aborts_quiescence': sum(s['aborts_quiescence'] for s in group),
        'aborts_cue': sum(s['aborts_cue'] for s in group),
        'aborts_delay': sum(s['aborts_delay'] for s in group),
        'aborts_response': sum(s['aborts_response'] for s in group),
        'warmup_trial_count': sum(s['warmup_trial_count'] for s in group),
        'repeat_trial_count': sum(s['repeat_trial_count'] for s in group),
        'session_water_ul': sum(water_vals) if water_vals else None,
        'staircase_advanced_this_session': last['staircase_advanced_this_session'],
        'git_commit': last['git_commit'],
        'target_spl_db': last['target_spl_db'],
        'left_freq_hz': last['left_freq_hz'],
        'right_freq_hz': last['right_freq_hz'],
        'session_csv_path': '; '.join(s['session_csv_path'] for s in group),
        'session_struct_path': None,   # filled in by the caller
    }


def _stage_sort_key(protocol_name):
    """ Ascending stage-number order (stage2_... -> 2); anything not matching the stageN_
    convention sorts alphabetically after every numbered stage. """
    match = re.match(r'stage(\d+)', protocol_name)
    if match:
        return (0, int(match.group(1)), protocol_name)
    return (1, 0, protocol_name)


_TEST_PROTOCOL_LABELS = {
    'full_protocol_lookback_test': 'Test',   # the Tests-project protocol currently scanned -- see
                                              # summarize_session()'s docstring. Add further
                                              # entries here if more Tests-project protocols ever
                                              # get a real PROTOCOL_CONFIG entry.
}


def _stage_label(protocol_name):
    match = re.match(r'stage(\d+)', protocol_name)
    if match:
        return 'Stage {0}'.format(match.group(1))
    return _TEST_PROTOCOL_LABELS.get(protocol_name, protocol_name)


# --- workbook update-in-place --------------------------------------------------------------------

# ('internal key', 'column header label'). This is the CANONICAL order -- every run rewrites the
# entire table (header row included) to match it exactly, so a reorder here actually takes effect
# on an existing sheet too, not just a brand-new one. No 'protocol' column: which stage a row
# belongs to is already unambiguous from the section header it sits under (see
# _write_section_header()), so a per-row repeat of the same text would just be redundant.
COLUMNS = [
    ('date', 'Date'),
    ('time_of_day', 'Time (HH:MM)'),
    ('weight_g', 'Weight (g)'),                    # MANUAL -- never written for an existing row
    ('weight_pct', 'Weight %'),                    # formula, refreshed every run
    ('weight_after_task_g', 'Weight after task (g)'),   # MANUAL, same treatment as weight_g
    ('weight_after_task_pct', 'Weight after task %'),   # formula, same treatment as weight_pct --
                                                         # references weight_after_task_g instead
    ('hand_watered', 'Water given by hand after task'), # MANUAL, TRUE/FALSE
    ('trial_count', 'Trials'),
    ('session_duration_min', 'Duration (min)'),   # first trial start -> last trial end
    ('reward_ul', 'Reward Amount (uL)'),            # auto, from the REWARD_UL VAL
    ('consumed_volume_ul', 'Water Consumed (uL)'),  # auto, licked after delivery (same rule as the
                                                     # Stage 1/2 water cap, water_tracker.py)
    ('delivered_volume_ul', 'Water Delivered (uL)'),  # auto, every reward the valve gave
    ('aborts', 'Aborts'),
    ('withheld_count', 'Withheld'),
    ('no_movement_count', 'No-movement'),
    ('incorrect_count', 'Incorrect'),           # full_protocol_lookback_test only -- a genuine
                                                 # wrong-choice outcome, distinct from Withheld
    ('wheel_abort_count', 'Wheel Aborts'),      # full_protocol_lookback_test only -- mid-cue wheel
                                                 # movement aborts, distinct from Aborts (which is
                                                 # Stage 1/2's QUIESCENCE_BREAKS count)
    ('l_count', 'L turns'),
    ('r_count', 'R turns'),
    ('lick_count', 'Licks'),
    ('threshold_start_deg', 'Threshold start (deg)'),
    ('threshold_end_deg', 'Threshold end (deg)'),
    ('threshold_pct_final', 'Threshold end (% final)'),
    ('iti_end_s', 'ITI (s)'),
    ('gain_mult_end', 'Gain mult (Stage 1)'),
    ('direction_ratio_end', 'Direction ratio (Stage 2)'),
    ('simple_gates_met', 'Simple gates met'),   # covers Stage 2 (partial) and Stage 3/4/5 (whole
                                                 # criterion) -- see _compute_simple_gates_met()
    # Stage 3/4/5 only, below -- every cross-session-varying value those protocols' own task
    # scripts track, so the training log has a column for each without opening a session struct.
    ('response_threshold_start_deg', 'Response Thresh Start (deg)'),
    ('response_threshold_end_deg', 'Response Thresh End (deg)'),
    ('quiescence_start_s', 'Quiescence Start (s)'),
    ('quiescence_end_s', 'Quiescence End (s)'),
    ('cue_abort_threshold_deg', 'In-trial Threshold (deg)'),
    ('click_level_attenuated', 'Click Attenuated (Stage 4)'),
    ('accuracy_aos', 'Accuracy (gamma=1.0)'),
    ('abort_rate', 'Abort Rate'),
    ('aborts_quiescence', 'Aborts: Quiescence'),
    ('aborts_cue', 'Aborts: Cue'),
    ('aborts_delay', 'Aborts: Delay'),
    ('aborts_response', 'Aborts: Response'),
    ('warmup_trial_count', 'Warmup Trials'),
    ('repeat_trial_count', 'Repeat Trials'),
    ('session_water_ul', 'Session Water (uL)'),
    ('staircase_advanced_this_session', 'Staircase Advanced (Stage 5)'),
    ('git_commit', 'Git Commit'),
    ('target_spl_db', 'Target SPL (dB)'),
    ('left_freq_hz', 'Left Freq (Hz)'),
    ('right_freq_hz', 'Right Freq (Hz)'),
    ('num_sessions', 'Num. Sessions'),
    ('notes', 'Notes'),                       # MANUAL -- never written for an existing row
    ('session_started', 'Session started'),   # hidden key -- `; `-joined member timestamps
    ('session_csv_path', 'Session data (local path)'),
    ('session_struct_path', 'Session struct (local path)'),
]
_MANUAL_COLUMNS = {'weight_g', 'notes', 'weight_after_task_g', 'hand_watered'}

_HEADER_FIELD_LABELS = ['DOB', 'Sex', 'Strain', 'Baseline weight (g)']
_BASELINE_WEIGHT_ROW = 2 + _HEADER_FIELD_LABELS.index('Baseline weight (g)')   # = 5
_BASELINE_WEIGHT_COL_LETTER = 'B'

_HEADER_FONT = Font(bold=True)
_TITLE_FONT = Font(bold=True, size=12)
_SECTION_FONT = Font(bold=True, size=11)

# One distinct fill per stage number (1-indexed into this list, wrapping if there are ever more
# stages than colors) -- blue/green/yellow/orange/purple, all light enough for black text to stay
# readable. Anything that doesn't match the stageN_ naming convention falls back to plain gray.
_STAGE_FILL_COLORS = ['DDEBF7', 'E2F0D9', 'FFF2CC', 'FCE4D6', 'E4DFEC']
_DEFAULT_FILL_COLOR = 'F2F2F2'


def _stage_fill_color(protocol_name):
    match = re.match(r'stage(\d+)', protocol_name)
    if not match:
        return _DEFAULT_FILL_COLOR
    stage_num = int(match.group(1))
    return _STAGE_FILL_COLORS[(stage_num - 1) % len(_STAGE_FILL_COLORS)]


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


def _canonical_col_map():
    """ Fixed column positions matching COLUMNS' own order, 1-indexed. Used for ALL writing --
    the table gets fully rewritten in this order every run (see _rewrite_subject_sheet()). """
    return {key: idx for idx, (key, _label) in enumerate(COLUMNS, start=1)}


def _scan_existing_column_labels(ws, table_start_row):
    """ Returns {key: col_idx} for whatever COLUMNS-known labels are CURRENTLY in the header row,
    at whatever position they happen to be -- used ONLY to locate old data (hand-typed manual
    cells, the hidden session-key column) before the table gets cleared and rewritten in canonical
    order. Any label not in COLUMNS (an old, now-retired column, e.g. a since-removed 'Protocol /
    Stage' or legacy 'Rewards' column from an earlier schema) is simply not reported -- its data
    doesn't survive the rewrite, which is intentional: those columns were already fully
    superseded, nothing of value is lost. """
    label_to_key = {label: key for key, label in COLUMNS}
    col_map = {}
    for col_idx in range(1, ws.max_column + 1):
        value = ws.cell(row=table_start_row, column=col_idx).value
        if value in label_to_key:
            col_map[label_to_key[value]] = col_idx
    return col_map


def _read_existing_manual_cells(ws, table_start_row, col_map):
    """ Returns {frozenset(member session_started keys): {'weight_g':..., 'notes':...}} for every
    existing DATA row (section-header rows have no session_started value, so they're skipped
    automatically). Read BEFORE the data area is cleared, so hand-typed Weight (g)/Notes values
    can be carried forward onto whichever row now represents that same underlying session(s). """
    existing = {}
    key_col = col_map.get('session_started')
    if key_col is None:
        return existing
    weight_col = col_map.get('weight_g')
    notes_col = col_map.get('notes')
    weight_after_col = col_map.get('weight_after_task_g')
    hand_watered_col = col_map.get('hand_watered')
    for row in range(table_start_row + 1, ws.max_row + 1):
        key_val = ws.cell(row=row, column=key_col).value
        if not key_val:
            continue
        member_keys = frozenset(k.strip() for k in str(key_val).split(';') if k.strip())
        if not member_keys:
            continue
        existing[member_keys] = {
            'weight_g': ws.cell(row=row, column=weight_col).value if weight_col else None,
            'notes': ws.cell(row=row, column=notes_col).value if notes_col else None,
            'weight_after_task_g': (ws.cell(row=row, column=weight_after_col).value
                                     if weight_after_col else None),
            'hand_watered': (ws.cell(row=row, column=hand_watered_col).value
                              if hand_watered_col else None),
        }
    return existing


def _carry_forward_manual_cells(member_key_set, existing):
    """ Exact match (this exact group already had a row) carries its manual cells straight over.
    Otherwise, if some existing row's member-set is a strict SUBSET of this (bigger) group's
    member-set, that row represents a session that used to stand alone (or was a smaller group)
    and has now been absorbed into a bigger one after a restart appeared -- its manual cells still
    carry forward, since it's the same underlying training bout, just now more complete. Returns
    (manual_cells_dict, matched_existing_row: bool). """
    if member_key_set in existing:
        return existing[member_key_set], True
    for old_keys, old_manual in existing.items():
        if old_keys < member_key_set:   # strict subset
            return old_manual, True
    return {'weight_g': None, 'notes': None, 'weight_after_task_g': None,
            'hand_watered': None}, False


def _write_row(ws, row, col_map, row_data):
    row_data = dict(row_data)
    if row_data.get('threshold_end_deg') is not None and row_data.get('threshold_final_deg'):
        row_data['threshold_pct_final'] = (row_data['threshold_end_deg']
                                            / row_data['threshold_final_deg'])
    else:
        row_data['threshold_pct_final'] = None

    for key, _ in COLUMNS:
        if key in _MANUAL_COLUMNS or key in ('weight_pct', 'weight_after_task_pct'):
            continue   # manual columns: written from whatever was carried forward, below.
        col_idx = col_map[key]
        value = row_data.get(key)
        cell = ws.cell(row=row, column=col_idx, value=value)
        if key == 'threshold_pct_final' and value is not None:
            cell.number_format = '0%'
        elif key in ('accuracy_aos', 'abort_rate') and value is not None:
            cell.number_format = '0%'

    for key in _MANUAL_COLUMNS:
        ws.cell(row=row, column=col_map[key], value=row_data.get(key))

    # weight_pct/weight_after_task_pct are formulas (not hand-typed values), safe to refresh every
    # run regardless of whether the row is new or existing -- they only ever READ weight_g/
    # weight_after_task_g + baseline, never write them.
    for weight_key, pct_key in (('weight_g', 'weight_pct'),
                                 ('weight_after_task_g', 'weight_after_task_pct')):
        weight_col_letter = get_column_letter(col_map[weight_key])
        pct_cell = ws.cell(row=row, column=col_map[pct_key])
        pct_cell.value = '=IFERROR({0}{1}/${2}${3},"")'.format(
            weight_col_letter, row, _BASELINE_WEIGHT_COL_LETTER, _BASELINE_WEIGHT_ROW)
        pct_cell.number_format = '0.0%'


def _write_section_header(ws, row, title, num_cols, fill_color):
    ws.cell(row=row, column=1, value=title).font = _SECTION_FONT
    fill = PatternFill('solid', fgColor=fill_color)
    for col in range(1, num_cols + 1):
        ws.cell(row=row, column=col).fill = fill


def _rewrite_subject_sheet(wb, subject, sessions):
    """ Regenerates this subject's entire data area (below the header block) from a fresh scan
    every run -- see the module docstring for why a simple incremental append isn't enough once
    merging + per-stage sectioning are both in play. Every OTHER subject's sheet in the workbook
    is never touched. """
    ws, table_start_row = _get_or_create_subject_sheet(wb, subject)
    old_col_map = _scan_existing_column_labels(ws, table_start_row)
    existing = _read_existing_manual_cells(ws, table_start_row, old_col_map)

    groups = group_sessions(sessions)

    row_dicts = []
    new_count = updated_count = 0
    for group in groups:
        row = combine_group(group)
        member_key_set = frozenset(row['session_started'].split('; '))
        manual, matched = _carry_forward_manual_cells(member_key_set, existing)
        row['weight_g'] = manual.get('weight_g')
        row['notes'] = manual.get('notes')
        row['weight_after_task_g'] = manual.get('weight_after_task_g')
        row['hand_watered'] = manual.get('hand_watered')
        if matched:
            updated_count += 1
        else:
            new_count += 1

        if len(group) > 1:
            member_csv_paths = [s['session_csv_path'] for s in group]
            out_base = os.path.splitext(group[0]['session_csv_path'])[0]
            mat_path, _json_path = session_struct_export.merge_session_structs(
                member_csv_paths, out_base)
            row['session_struct_path'] = mat_path
        else:
            row['session_struct_path'] = group[0]['session_struct_path']

        row_dicts.append(row)

    buckets = {}
    for row in row_dicts:
        buckets.setdefault(row['protocol'], []).append(row)
    for rows in buckets.values():
        rows.sort(key=lambda r: r['date'])
    ordered_protocols = sorted(buckets.keys(), key=_stage_sort_key)

    # Clear the ENTIRE table -- header row included, not just the data rows below it -- and
    # rewrite it fresh in canonical order. This is what actually lets a schema/column-order change
    # (like dropping 'Protocol / Stage' and moving Weight (g)/Weight % to C/D) take effect on an
    # existing sheet, not just a brand-new one; the old scan-and-append design preserved whatever
    # historical order a sheet already had forever, which is exactly what was wrong.
    if ws.max_row >= table_start_row:
        ws.delete_rows(table_start_row, ws.max_row - table_start_row + 1)

    col_map = _canonical_col_map()
    for key, label in COLUMNS:
        cell = ws.cell(row=table_start_row, column=col_map[key], value=label)
        cell.font = _HEADER_FONT
        cell.alignment = Alignment(wrap_text=True, vertical='bottom')

    row_num = table_start_row + 1
    num_cols = len(COLUMNS)
    for protocol in ordered_protocols:
        _write_section_header(ws, row_num, _stage_label(protocol), num_cols,
                               _stage_fill_color(protocol))
        row_num += 1
        for row in buckets[protocol]:
            _write_row(ws, row_num, col_map, row)
            row_num += 1

    # ws.column_dimensions is keyed by LETTER and persists independently of cell content/row
    # deletes -- if 'session_started' sat at a different letter in an earlier schema iteration,
    # that letter's hidden=True flag lingers forever unless explicitly cleared here. Reset every
    # column's hidden flag before re-hiding only the one that's actually 'session_started' now.
    for dim in ws.column_dimensions.values():
        dim.hidden = False
    for col_idx in col_map.values():
        ws.column_dimensions[get_column_letter(col_idx)].width = 14
    ws.column_dimensions[get_column_letter(col_map['session_started'])].hidden = True
    ws.freeze_panes = ws.cell(row=table_start_row + 1, column=1).coordinate
    return new_count, updated_count


def _start_from_main(timeout_s=30.0):
    """ Replaces training_log.xlsx on disk with the newest copy on main (keeping unsaved edits to
    this computer's own mice) before it gets rebuilt -- a computer-specific branch's own copy is
    usually older than main's. See sync_training_log.xlsx_from_main(). Best-effort: git runs on a
    daemon thread bounded by `timeout_s` (git has stalled for a minute+ on some rigs, see
    CLAUDE.md), and on any failure the local copy is used as before, with a warning. """
    result = {}

    def work():
        try:
            import sync_training_log
            result['bytes'] = sync_training_log.xlsx_from_main()
        except Exception as err:
            result['err'] = err

    thread = threading.Thread(target=work, daemon=True)
    thread.start()
    thread.join(timeout_s)
    if thread.is_alive() or 'err' in result:
        print("WARNING: could not load training_log.xlsx from main ({0}) -- updating the local "
              "copy instead.".format(result.get('err', 'git timed out')), flush=True)
        return
    if result.get('bytes') is not None:
        with open(_OUTPUT_PATH, 'wb') as f:
            f.write(result['bytes'])
        print("Started from the newest training_log.xlsx on main.", flush=True)


def build_workbook():
    _start_from_main()
    # Only this computer's own mice (see owned_subjects()) -- every other sheet, and every other
    # computer's session folders, are left exactly as they came from git.
    by_subject = scan_all_sessions(subjects=owned_subjects())
    if not by_subject:
        print("No sessions run on this computer found under experiments/*/setups/*/sessions/ -- "
              "nothing to update.", flush=True)
        return

    if os.path.exists(_OUTPUT_PATH):
        wb = openpyxl.load_workbook(_OUTPUT_PATH)
    else:
        wb = openpyxl.Workbook()
        wb.remove(wb.active)   # drop the default blank sheet -- one real sheet per subject instead

    total_new = total_updated = 0
    for subject in sorted(by_subject):
        new_count, updated_count = _rewrite_subject_sheet(wb, subject, by_subject[subject])
        total_new += new_count
        total_updated += updated_count
        print("  {0}: {1} new row(s), {2} updated row(s)".format(subject, new_count, updated_count),
              flush=True)

    _save_if_changed(wb, _OUTPUT_PATH)
    print("Wrote {0} -- {1} new row(s), {2} updated row(s) across {3} subject(s) run on this "
          "computer. Every other subject's sheet in the workbook was left untouched.".format(
              _OUTPUT_PATH, total_new, total_updated, len(by_subject)), flush=True)



def _xlsx_parts(data):
    """ {member name: bytes} of an .xlsx, without docProps/core.xml (it only holds the save time). """
    import io as _io
    import zipfile
    with zipfile.ZipFile(_io.BytesIO(data)) as z:
        return {n: z.read(n) for n in z.namelist() if n != 'docProps/core.xml'}


def _save_if_changed(wb, path):
    """ Saves wb to path unless the existing file already has exactly the same content. openpyxl
    stamps the save time into the file, so an unchanged workbook used to be rewritten -- and show
    up as changed in git -- on every rebuild. """
    import io as _io
    buf = _io.BytesIO()
    wb.save(buf)
    new = buf.getvalue()
    if os.path.exists(path):
        try:
            with open(path, 'rb') as f:
                if _xlsx_parts(f.read()) == _xlsx_parts(new):
                    return False
        except Exception:
            pass   # unreadable/corrupt existing file: just overwrite it
    with open(path, 'wb') as f:
        f.write(new)
    return True

if __name__ == '__main__':
    build_workbook()
