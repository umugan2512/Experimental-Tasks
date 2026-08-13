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


def export_session_struct(csv_path, task_params):
    """ task_params: dict of every VAR_* constant the calling task script used this run -- pass
    {k: v for k, v in globals().items() if k.startswith('VAR_')} from the call site. Automatic,
    stays complete as new parameters get added, no hand-maintained list to fall out of sync.
    Returns (mat_path, json_path). """
    info, session_vals, trials = session_csv_parser.parse_session_csv(csv_path)
    struct = {'info': info, 'task_params': task_params, 'session_vals': session_vals,
              'trials': trials}

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
