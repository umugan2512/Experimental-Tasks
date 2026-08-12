# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Self-persisting cross-session state for the wheel-shaping stages (training_protocol.md Stages 1-2)
-- a single session's Bpod process has no memory of yesterday's staircase/ITI/gain values on its
own, so this reads/writes a small JSON file in the subject's own project folder, updated once per
session by the task script itself (no manual data entry between sessions).

VAR_SUBJECT_ID (an explicit constant at the top of each stage task script, edited by hand when
starting/switching animals) is what selects which file -- no existing task in this codebase
auto-derives "which subject is this session for" from PyBpod's own session/experiment machinery
(no task anywhere references subject identity at all), so an explicit constant is the deliberate,
lower-risk choice over guessing at that association.
"""
import json
import os

_STATE_FILENAME = 'wheel_shaping_state.json'


def _state_path(project_dir, subject_id):
    return os.path.join(project_dir, 'subjects', subject_id, _STATE_FILENAME)


class StageState(object):
    """ Thin wrapper around a JSON dict persisted to disk. load() happens once, in __init__;
    save() is explicit (called once at session end by the task script, after every update this
    session has made) -- not autosaved per .set() call, so a session that crashes mid-run doesn't
    leave a half-updated state file from a partially-completed session.

    `defaults` seeds every key this stage cares about so a first-ever run (no file on disk yet)
    doesn't need special-casing by the caller -- get()/set() always operate on a fully-populated
    dict, either freshly defaulted or loaded-and-merged-over-defaults (so adding a new key to
    `defaults` in a future code change doesn't break loading an older, smaller saved file). """

    def __init__(self, project_dir, subject_id, defaults):
        self._path = _state_path(project_dir, subject_id)
        self._defaults = dict(defaults)
        self.data = self._load()

    def _load(self):
        if os.path.exists(self._path):
            with open(self._path, 'r') as f:
                loaded = json.load(f)
            merged = dict(self._defaults)
            merged.update(loaded)
            return merged
        return dict(self._defaults)

    def save(self):
        dir_path = os.path.dirname(self._path)
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)
        with open(self._path, 'w') as f:
            json.dump(self.data, f, indent=2, sort_keys=True)

    def get(self, key):
        return self.data[key]

    def set(self, key, value):
        self.data[key] = value
