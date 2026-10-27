# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Exports a "recreate this session" data structure next to a session's own CSV -- everything needed
to reconstruct what the task was configured to do (every VAR_* parameter, harvested automatically
via globals() at the call site) plus everything that actually happened (the full native per-trial
STATE/EVENT structure, via session_csv_parser). Written locally only, at every termination path
(normal finish, Stop, Kill) -- never committed to git (see the repo's .gitignore).

Two output files, same base name as the session CSV:
  <session>_struct.mat   -- scipy.io.savemat, for opening directly in MATLAB. `trials` comes out
                            as a 1xN cell array of structs, same shape as Bpod's own native
                            SessionData.RawEvents.Trial -- familiar to anyone who's used Bpod's
                            own MATLAB-side data format.
  <session>_struct.json  -- for Python/Claude to read back directly.
"""
import json
import os
import re

import scipy.io

import session_csv_parser

_INVALID_FIELD_CHARS = re.compile(r'[^A-Za-z0-9_]')
_MATLAB_MAX_FIELD_LEN = 63


def _sanitize_key(key):
    """ MATLAB struct field names must be valid identifiers (letters/digits/underscore only, max
    63 chars, can't start with a digit) -- most keys in this module's structures (state/event
    names, VAR_* params, INFO-row keys like SUBJECT-NAME/SESSION-STARTED) just need the hyphen
    replaced, but one native Bpod INFO row's own "key" is actually a full descriptive sentence
    ("This is a PYBPOD file. Find more info at ...") -- confirmed present in every real session
    CSV's preamble, not a parsing bug -- so this also has to strip arbitrary punctuation and
    truncate to fit. """
    key = _INVALID_FIELD_CHARS.sub('_', str(key))
    if key and key[0].isdigit():
        key = '_' + key
    key = key[:_MATLAB_MAX_FIELD_LEN]
    return key or '_'


def _matlab_safe(obj):
    """ Recursively converts None -> NaN (MATLAB has no None) and sanitizes dict keys, leaving
    everything else (numbers, strings, nested lists/dicts) as-is for scipy.io.savemat to encode
    (dict -> struct, list -> cell array). """
    if obj is None:
        return float('nan')
    if isinstance(obj, dict):
        return {_sanitize_key(k): _matlab_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_matlab_safe(v) for v in obj]
    return obj


def _split_stream_values(trials):
    """ Post-processes each trial's STREAM_KEYS values (lists of raw VAL strings -- see
    session_csv_parser.STREAM_KEYS) into numeric arrays, so the exported struct is directly usable
    without a second parse pass in MATLAB/Python. 'WHEEL_POS' rows are "t,pos" pairs -> {'t':
    [...], 'pos': [...]}; 'QUIESCENCE_RESET_TIME' rows are bare floats -> a flat list of floats.
    Mutates each trial's 'vals' dict in place and returns trials, so it can be chained inline at
    the call site. A trial with no rows for a given stream key simply has no key there (nothing to
    split), same as before this existed. """
    for trial in trials:
        vals = trial['vals']
        if 'WHEEL_POS' in vals:
            times, positions, module_times = [], [], []
            for raw in vals['WHEEL_POS']:
                parts = raw.split(',')   # 't,pos' or, since 2026-10-23, 't,pos,t_module'
                times.append(float(parts[0]))
                positions.append(float(parts[1]))
                module_times.append(float(parts[2]) if len(parts) > 2 else float('nan'))
            vals['WHEEL_POS'] = {'t': times, 'pos': positions, 't_module': module_times}
        if 'QUIESCENCE_RESET_TIME' in vals:
            vals['QUIESCENCE_RESET_TIME'] = [float(raw) for raw in vals['QUIESCENCE_RESET_TIME']]
        if 'WHEEL_CLOCK_OFFSET' in vals:
            vals['WHEEL_CLOCK_OFFSET'] = [float(raw) for raw in vals['WHEEL_CLOCK_OFFSET']]
    return trials


# The state each stage's trial starts in -- the one that sends the rotary its '#' marker (Stage 1/2:
# WheelPeriod, Stage 3/4/5: WheelDotPeriod). See session_csv_parser.wheel_samples_bpod_time().
_MARKER_STATES = ('WheelPeriod', 'WheelDotPeriod')


def _add_bpod_clock(trials, csv_path):
    """ Puts host-clock data on the Bpod state machine's clock, MATLAB Bpod's convention (every
    native STATE/EVENT time is already Bpod-trial-relative; bpod_start is the trial's
    TrialStartTimestamp):
    - 'TrialStartTimestamp'/'TrialEndTimestamp' per trial (copies of bpod_start/bpod_end, under
      MATLAB's own names).
    - 'WHEEL_POS_bpod_t' per trial: exact via the trial's ROTARY_MARKER (stream-mode sessions),
      else via the session-wide host->Bpod fit.
    - session-level 'camera' {'frame', 'host_t', 'bpod_t'} from <session folder>/
      session_video_timestamps.csv, via that same fit.
    Returns the fit dict (or None). Must run before _split_stream_values(), which replaces the raw
    WHEEL_POS strings. """
    fit = session_csv_parser.fit_host_to_bpod(trials)
    for trial in trials:
        trial['TrialStartTimestamp'] = trial.get('bpod_start')
        trial['TrialEndTimestamp'] = trial.get('bpod_end')
        exact = None
        for state in _MARKER_STATES:
            exact = session_csv_parser.wheel_samples_bpod_time(trial, state)
            if exact is not None:
                break
        if exact is not None:
            trial['WHEEL_POS_bpod_t'] = [t for t, _ in exact]
        elif fit is not None and trial['vals'].get('WHEEL_POS'):
            trial['WHEEL_POS_bpod_t'] = [
                session_csv_parser.host_to_bpod(fit, float(raw.split(',')[0]))
                for raw in trial['vals']['WHEEL_POS']]
    return fit


def _camera_times(csv_path, fit):
    path = os.path.join(os.path.dirname(csv_path), 'session_video_timestamps.csv')
    if not os.path.exists(path):
        return None
    frames, host_t = [], []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            try:
                frames.append(int(parts[0]))
                host_t.append(float(parts[1]))
            except (ValueError, IndexError):
                continue   # header row
    bpod_t = [session_csv_parser.host_to_bpod(fit, t) for t in host_t] if fit else None
    return {'frame': frames, 'host_t': host_t, 'bpod_t': bpod_t}


def export_session_struct(csv_path, task_params):
    """ task_params: dict of every VAR_* constant the calling task script used this run -- pass
    {k: v for k, v in globals().items() if k.startswith('VAR_')} from the call site. Automatic,
    stays complete as new parameters get added, no hand-maintained list to fall out of sync.
    Returns (mat_path, json_path). """
    info, session_vals, trials = session_csv_parser.parse_session_csv(csv_path)
    fit = _add_bpod_clock(trials, csv_path)
    trials = _split_stream_values(trials)
    struct = {'info': info, 'task_params': task_params, 'session_vals': session_vals,
              'trials': trials, 'clock': {'host_to_bpod': fit},
              'camera': _camera_times(csv_path, fit)}

    base = os.path.splitext(csv_path)[0]
    mat_path = base + '_struct.mat'
    json_path = base + '_struct.json'

    scipy.io.savemat(mat_path, _matlab_safe(struct), long_field_names=True)
    with open(json_path, 'w') as f:
        json.dump(struct, f, indent=2, default=str)   # default=str handles any stray
                                                        # non-JSON-native values; NaN is handled
                                                        # fine by json's own allow_nan=True default

    return mat_path, json_path


def _load_member(csv_path):
    """ Prefers that session's own already-exported <csv>_struct.json (has task_params -- only
    ever captured live from the running task's own globals(), not recoverable by re-parsing the
    CSV). Falls back to parsing the raw CSV directly if no struct was ever exported for it (the
    crash case) or the JSON is unreadable for any reason. """
    json_path = os.path.splitext(csv_path)[0] + '_struct.json'
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            return {'info': data.get('info', {}), 'task_params': data.get('task_params', {}),
                    'trials': data.get('trials', [])}
        except (ValueError, OSError):
            pass   # fall through to the raw-CSV fallback below
    info, _session_vals, trials = session_csv_parser.parse_session_csv(csv_path)
    _add_bpod_clock(trials, csv_path)
    trials = _split_stream_values(trials)
    return {'info': info, 'task_params': None, 'trials': trials}


def merge_session_structs(member_csv_paths, out_base):
    """ Combines multiple raw sessions (a Stop/Kill/crash-interrupted bout and its restart(s), per
    build_training_log.py's grouping) into one struct: 'trials' is every member's trials
    concatenated in chronological order (the "just give me one continuous trial list" view);
    'members' keeps each sub-session's own info/task_params traceable rather than silently
    collapsing them (they may differ slightly between a Stop and its restart). Writes
    <out_base>_combined_struct.mat/.json. Returns (mat_path, json_path). """
    members = [_load_member(p) for p in member_csv_paths]
    all_trials = [trial for member in members for trial in member['trials']]
    struct = {'members': members, 'trials': all_trials}

    mat_path = out_base + '_combined_struct.mat'
    json_path = out_base + '_combined_struct.json'

    scipy.io.savemat(mat_path, _matlab_safe(struct), long_field_names=True)
    with open(json_path, 'w') as f:
        json.dump(struct, f, indent=2, default=str)

    return mat_path, json_path
