# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Standalone session runner/dashboard -- one widget to launch a wheel-shaping stage task, watch its
live progress (everything PyBpod itself currently shows for a running session -- the "Wheel
Position" plot, every `WheelShapingPlots` outcome panel, a camera snippet), and record
weight/baseline/notes directly into `training_log.xlsx`, without having the PyBpod GUI open at all.
Lives at this project's own top level (a sibling of `tasks/`/`records/`/`experiments/`), not inside
`records/`, since it's the thing you open to RUN a session, not a record-keeping script.

Run directly: `python run_session.py` (or double-click `Run Session.bat` -- see that file).
Deliberately standalone, same "bench tool, not a PyBpod GUI task" reasoning as
`Calibration/rig_check.py`/`calibrate_liquid.py`: it never opens a Bpod/rotary/HiFi connection of
its own, and launches a real stage task the exact same way `board_com.py:run_task()` does (create
the session folder, write a matching `user_settings.py`, `subprocess.Popen` the task script with
that folder as `cwd`) -- so from the task script's own point of view, nothing is different from
being launched by the real PyBpod GUI.

**Isolation**: this is the ONE new file. It only *reads* via `session_csv_parser.py`/
`build_training_log.py`/`wheel_shaping_plots.py` (imports them, never edits them) and only
*writes* to `training_log.xlsx`'s existing manual cells (the same cells a human already edits by
hand) and to a brand-new timestamped session folder it creates for a task it launches. No existing
task script or shared module is touched.

**Live view, everything reconstructed from the session's own CSV file (no live Bpod connection
needed), all embedded in ONE window (no separate popups)**:
- A small wheel-position-over-time plot, mirroring `pybpod-gui-plugin-rotaryencoder`'s own "Wheel
  Position" live plot -- built from `WHEEL_POS` VAL rows, which already encode both elapsed time
  and position together (`"<elapsed_s>,<position_deg>"`). Also overlays each trial's own threshold
  (a step line -- Stage 2's own staircase changes it trial to trial) and a light vertical marker at
  every trial's own start time.
- The FULL `WheelShapingPlots` panel set (raster, progress, accuracy, engagement, sidebias,
  percent_outcome, psychometric, reaction_time, reward_licks, lick_timeline) -- reused directly via
  its own `embed=True` mode (see `wheel_shaping_plots.py`), which builds a bare Figure/
  FigureCanvasQTAgg instead of letting `plt.subplot_mosaic()` open its own top-level window, so its
  canvas can be embedded directly into this window's own layout (inside a scroll area, since the
  full panel set is taller than most screens). Fed incrementally: each new trial found in the
  parsed CSV since the last poll has its `add_trial()` arguments reconstructed from the raw STATE/
  EVENT/VAL rows (`classify_trial()` for outcome/side, direct VAL lookups for threshold/gain/ITI/
  trial_type/response_time/click counts). `p_right_target` (Stage 3/4's debiasing target) is the
  one value never logged -- it's fully recoverable anyway, since `debiasing.next_side_after_error()`'s
  target is a deterministic function of only the PREVIOUS trial's own logged outcome/side, replayed
  here trial-by-trial rather than guessed.
- A camera snippet, refreshed from `CameraRecorder`'s own new live-frame JPEG export
  (`<video_stem>_latest.jpg`, see `_shared/camera_recorder.py`) rather than the growing
  `session_video.avi` file itself -- confirmed directly that `cv2.VideoWriter` (XVID/.avi) writes
  NOTHING to disk at all until `release()` is called (the file stays 0 bytes for the whole
  session), so reading the growing video file was never actually going to work; the small JPEG
  companion file is a plain, complete, always-readable image with no video-container concerns.

**Known, accepted approximations** (documented rather than silently wrong):
- `magnitude_deg` fed to the raster is reconstructed as "that trial's own threshold if the wheel
  crossed it, else 0.0" -- confirmed by reading all four real stage scripts directly, this is
  EXACTLY what they themselves already pass to `add_trial()` (none of them log the true continuous
  peak deflection), so this is not an approximation at all, just matching existing behavior.
- `lick_times_abs`/`reward_time_abs` are anchored to each trial's own logged absolute-time VAL
  (`TRIAL_START` for Stage 1/2, `SM_SEND_TIME` for Stage 3/4) plus that trial's raw event/state
  offsets. For Stage 3/4, a lick occurring during the earlier CUE period (before the choice-period
  state machine sends) is not separately recoverable this way and is simply not included in the
  lick timeline -- a minor, rare gap (most licks happen during/after the choice period).
"""
import contextlib
import datetime
import glob
import io
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid

import cv2
import numpy as np

import matplotlib
matplotlib.use('Qt5Agg')   # same fix already applied to full_protocol_lookback_test.py -- one Qt
                            # event loop for the whole process, see CLAUDE.md's PyEval_RestoreThread
                            # crash note.

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QComboBox, QPushButton, QLabel,
    QGroupBox, QMessageBox, QLineEdit, QTextEdit, QDoubleSpinBox, QCheckBox, QScrollArea,
    QDialog, QDialogButtonBox, QListWidget, QListWidgetItem)

_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
_RECORDS_DIR = os.path.join(_PROJECT_DIR, 'records')
_TASKS_DIR = os.path.join(_PROJECT_DIR, 'tasks')
_SHARED_DIR = os.path.join(_TASKS_DIR, '_wheel_shaping_shared')
_OUTPUT_XLSX = os.path.join(_RECORDS_DIR, 'training_log.xlsx')
# Calibration/ is a sibling of _projects/ (see CLAUDE.md), not nested under this project --
# _PROJECT_DIR is _projects/AuditoryEvidenceAccum/, so two levels up reaches the repo root.
_CALIBRATION_DIR = os.path.abspath(os.path.join(_PROJECT_DIR, '..', '..', 'Calibration'))
VAR_RIG_CHECK_SCRIPT = os.path.join(_CALIBRATION_DIR, 'rig_check.py')
VAR_FLUSH_VALVE_SCRIPT = os.path.join(_CALIBRATION_DIR, 'flush_valve.py')

sys.path.insert(0, _SHARED_DIR)
sys.path.insert(0, _RECORDS_DIR)   # build_training_log.py lives in records/, a sibling directory
                                    # now that this script lives at the project's own top level
                                    # rather than inside records/ itself -- an explicit path entry
                                    # instead of relying on the script's own directory implicitly
                                    # being on sys.path (which only happens for the __main__ entry
                                    # script, not a module imported from elsewhere).
import session_csv_parser     # noqa: E402
import build_training_log     # noqa: E402
import sync_training_log      # noqa: E402
import wheel_shaping_plots    # noqa: E402
import debiasing               # noqa: E402

import openpyxl  # noqa: E402

VAR_BPOD_FIRMWARE_VERSION = '22'   # matches every real session-local user_settings.py's own
                                    # TARGET_BPOD_FIRMWARE_VERSION on this rig -- same "machine-
                                    # specific constant" convention as VAR_BPOD_SERIAL_PORT
                                    # elsewhere in this codebase, see pybpodapi/settings.py's own
                                    # module-level default for confirmation.
VAR_PROJECT_NAME = 'AuditoryEvidenceAccum'

# The interpreter used to launch a TASK subprocess, deliberately separate from whatever interpreter
# is running this dashboard's own GUI (sys.executable) -- run_session.py's own deps (PyQt5/cv2/
# matplotlib/openpyxl) and a task script's own deps (pybpodapi + hardware modules) are genuinely
# different, so forcing them to share one interpreter only worked by coincidence when both happened
# to be the same env. Confirmed as a real, already-hit failure: a task launched with the wrong
# interpreter dies instantly with "ModuleNotFoundError: No module named 'pybpodapi'", and previously
# that was only ever visible in a log file inside the new session folder, not in the UI at all (see
# SessionRunnerWindow's startup self-check / post-launch quick-death detection below).
# Machine-specific -- edit/extend per box, same convention as VAR_BPOD_SERIAL_PORT elsewhere in
# this codebase. Tries each known real environment in order (the real rig's own conda env is named
# "pybpod-environment"; this dev machine's is "UM-pybpod", confirmed directly:
# `C:\Users\Gil\anaconda3\envs\UM-pybpod\python.exe -c "import pybpodapi"` succeeds, resolving to
# this repo's own editable pybpod-api install), falling back to sys.executable only if none exist.
_TASK_PYTHON_CANDIDATES = [
    r'C:\Users\2P-Behav\.conda\envs\pybpod-environment\python.exe',
    r'C:\Users\Gil\anaconda3\envs\UM-pybpod\python.exe',
]
VAR_TASK_PYTHON_EXE = next((p for p in _TASK_PYTHON_CANDIDATES if os.path.exists(p)), sys.executable)
VAR_THRESHOLD_FINAL_DEG = 35.0     # matches every stage script's own VAR_THRESHOLD_FINAL_DEG /
                                    # build_training_log.py's own hardcoded threshold_final_deg.

VAR_DEFAULT_REWARD_UL = 7.0    # default shown in the Launch Task reward field -- overridable per
                                # launch; passed to the task subprocess via RUN_SESSION_REWARD_UL

VAR_SESSION_POLL_MS = 1000    # how often to look for a newer session folder on disk
VAR_PLOTS_POLL_MS = 500       # how often to read the watched CSV's new rows and feed new trials
                              # (incremental -- see session_csv_parser.IncrementalSessionParser)
VAR_CAMERA_POLL_MS = 1000     # how often to read the latest camera JPEG (see grab_latest_frame())
VAR_STOP_GRACE_S = 8.0        # how long to wait for a graceful 'close' before escalating to kill
VAR_QUICK_DEATH_CHECK_MS = 2000   # how long after launch to check whether the task subprocess has
                                   # already died (e.g. missing dependency, bad COM port) -- long
                                   # enough for a real Bpod handshake/import to get underway, short
                                   # enough that a genuine crash is still reported promptly.

_VALID_NAME_RE = re.compile(r'^[A-Za-z0-9_-]+$')


def task_python_can_import_pybpodapi():
    """ Startup self-check: does VAR_TASK_PYTHON_EXE actually have pybpodapi importable? Run once
    at window construction so a misconfigured/missing task environment is reported immediately as
    a visible warning banner, instead of only surfacing after a session folder has already been
    created and the task subprocess has already died silently (the exact failure mode that
    prompted this check -- see VAR_TASK_PYTHON_EXE's own docstring above). """
    try:
        # stdout=/stderr=PIPE, not capture_output=True -- this whole file needs to run under
        # Python 3.6 (e.g. launched via the "UM-pybpod" Anaconda env, which already bundles every
        # GUI dependency this file needs alongside pybpodapi itself), and capture_output is a
        # Python 3.7+-only subprocess.run() kwarg. Confirmed as a real, already-hit bug: under
        # Python 3.6 this raised TypeError on every call, silently swallowed by the broad except
        # below, making the self-check permanently (and wrongly) report "not importable".
        result = subprocess.run([VAR_TASK_PYTHON_EXE, '-c', 'import pybpodapi'],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15)
        return result.returncode == 0
    except Exception:
        return False


_DASHBOARD_PROTOCOL_INFO = {
    'stage1_wheel_shaping': {
        'stage': 1,
        'anchor_key': 'TRIAL_START',
        'threshold_key': 'THRESHOLD_DEG',
        'outcome_map': {'rewarded': 'Rewarded', 'no_movement': 'NoMovement', 'aborted': 'Abort'},
        'crossed_outcomes': {'rewarded'},
    },
    'stage2_threshold_staircase': {
        'stage': 2,
        'anchor_key': 'TRIAL_START',
        'threshold_key': 'THRESHOLD_DEG',
        'outcome_map': {'rewarded': 'Rewarded', 'withheld': 'Withheld', 'no_movement': 'NoMovement',
                         'aborted': 'Abort'},
        'crossed_outcomes': {'rewarded', 'withheld'},
    },
    'stage3_tone_side_association': {
        'stage': 3,
        'anchor_key': 'SM_SEND_TIME',
        'threshold_key': 'RESPONSE_THRESHOLD_DEG',
        'outcome_map': {'rewarded': 'Reward', 'incorrect': 'NoReward', 'no_movement': 'NoResponse',
                         'aborted': 'Abort'},
        'crossed_outcomes': {'rewarded', 'incorrect'},
    },
    'stage4_clicks_direction': {
        'stage': 4,
        'anchor_key': 'SM_SEND_TIME',
        'threshold_key': 'RESPONSE_THRESHOLD_DEG',
        'outcome_map': {'rewarded': 'Reward', 'incorrect': 'NoReward', 'no_movement': 'NoResponse',
                         'aborted': 'Abort'},
        'crossed_outcomes': {'rewarded', 'incorrect'},
    },
    'stage5_resume_staircases': {
        'stage': 5,
        'anchor_key': 'SM_SEND_TIME',
        'threshold_key': 'RESPONSE_THRESHOLD_DEG',
        'outcome_map': {'rewarded': 'Reward', 'incorrect': 'NoReward', 'no_movement': 'NoResponse',
                         'aborted': 'Abort'},
        'crossed_outcomes': {'rewarded', 'incorrect'},
    },
}

_OUTCOME_STATE_NAME = {
    'stage1_wheel_shaping': {'rewarded': 'Reward'},
    'stage2_threshold_staircase': {'rewarded': ('RewardL', 'RewardR'),
                                    'withheld': ('NoRewardL', 'NoRewardR')},
    'stage3_tone_side_association': {'rewarded': 'Reward', 'incorrect': 'ErrorConsumption'},
    'stage4_clicks_direction': {'rewarded': 'Reward', 'incorrect': 'ErrorConsumption'},
    'stage5_resume_staircases': {'rewarded': 'Reward', 'incorrect': 'ErrorConsumption'},
}

_USER_SETTINGS_TEMPLATE = """SETTINGS_PRIORITY = 0

PYBPOD_SERIAL_PORT       = '{serial_port}'
PYBPOD_NET_PORT          = {net_port}

{bnc_ports_line}
{wired_ports_line}
{behavior_ports_line}

PYBPOD_API_LOG_LEVEL = None
PYBPOD_API_LOG_FILE  = None

PYBPOD_API_STREAM2STDOUT = True
PYBPOD_API_ACCEPT_STDIN  = True

PYBPOD_PROTOCOL \t= '{protocol}'
PYBPOD_CREATOR \t\t= '{creator}'
PYBPOD_PROJECT \t\t= '{project}'
PYBPOD_EXPERIMENT \t= '{experiment}'
PYBPOD_BOARD \t\t= '{board}'
PYBPOD_SETUP \t\t= '{setup}'
PYBPOD_SESSION \t\t= '{session}'
PYBPOD_SESSION_PATH = '{session_path}'
PYBPOD_SUBJECTS \t= [{subjects}]
PYBPOD_SUBJECT_EXTRA = ''
PYBPOD_USER_EXTRA = ''


TARGET_BPOD_FIRMWARE_VERSION = '{firmware}'

#import logging
#PYBPOD_API_LOG_LEVEL = logging.DEBUG
#PYBPOD_API_LOG_FILE  = 'pybpod-api.log'

PYBPOD_VARSNAMES = []
"""


# --- project/board/subject/user discovery (mirrors what the PyBpod GUI itself already has on
# disk -- never guessed/hardcoded) ------------------------------------------------------------

def _subject_json_path(name):
    return os.path.join(_PROJECT_DIR, 'subjects', name, name + '.json')


def _subject_exists(name):
    return os.path.exists(_subject_json_path(name))


def _known_uuids():
    """ Every __UUID4__ of a subject/board/task/user descriptor on disk (the files a setup's
    __EXTERNAL-REF__ can point at) -- same "basename matches its parent folder" filter as
    discover_subjects(). """
    uuids = set()
    for kind in ('subjects', 'boards', 'users'):
        for path in glob.glob(os.path.join(_PROJECT_DIR, kind, '*', '*.json')):
            if os.path.splitext(os.path.basename(path))[0] != os.path.basename(os.path.dirname(path)):
                continue
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    uuids.add(json.load(f).get('__UUID4__'))
            except Exception:
                continue
    for path in glob.glob(os.path.join(_TASKS_DIR, '*', '*.json')):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                uuids.add(json.load(f).get('__UUID4__'))
        except Exception:
            continue
    uuids.discard(None)
    return uuids


def _prune_setup_data(setup_data, known_uuids):
    """ Removes, in place, subject names with no subject JSON on disk and __EXTERNAL-REF__
    UUIDs matching nothing on disk. Returns (removed_names, removed_uuids). """
    names = setup_data.get('subjects') or []
    removed_names = [n for n in names if not _subject_exists(n)]
    setup_data['subjects'] = [n for n in names if _subject_exists(n)]
    refs = setup_data.get('__EXTERNAL-REF__') or []
    removed_uuids = [u for u in refs if u not in known_uuids]
    if '__EXTERNAL-REF__' in setup_data:
        setup_data['__EXTERNAL-REF__'] = [u for u in refs if u in known_uuids]
    return removed_names, removed_uuids


def prune_dangling_setup_refs():
    """ Run once at dashboard startup: strips every setup JSON of subjects that were deleted
    without being detached first (their names and dangling UUIDs). Setup JSONs are machine-local
    and gitignored; recorded session data is never touched. Returns {setup_path: (names, uuids)}
    for what was removed. """
    known = _known_uuids()
    removed = {}
    pattern = os.path.join(_PROJECT_DIR, 'experiments', '*', 'setups', '*', '*.json')
    for path in sorted(glob.glob(pattern)):
        if os.path.splitext(os.path.basename(path))[0] != os.path.basename(os.path.dirname(path)):
            continue
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            continue
        names, uuids = _prune_setup_data(data, known)
        if names or uuids:
            data['__UPDATED-ON__'] = datetime.datetime.now().isoformat(' ')
            _write_pybpod_json(path, data)
            removed[path] = (names, uuids)
            print("Removed deleted subject(s) {0} ({1} dangling ref(s)) from setup {2}".format(
                names, len(uuids), os.path.relpath(path, _PROJECT_DIR)), flush=True)
    return removed


def discover_stage_configs():
    """ Returns a list of {experiment, setup, board, task_name, task_script, subjects}, one per
    already-GUI-configured experiment/setup this project has on disk (scanned, not a hardcoded
    table, so a not-yet-created stage -- e.g. Stage 4 before its first GUI setup -- simply doesn't
    appear rather than needing separate scaffolding-creation logic this tool doesn't own). Only
    setups whose task actually has a real script under tasks/ are included. """
    results = []
    pattern = os.path.join(_PROJECT_DIR, 'experiments', '*', 'setups', '*', '*.json')
    for path in sorted(glob.glob(pattern)):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            continue
        setup_dir = os.path.dirname(path)
        experiment_dir = os.path.dirname(os.path.dirname(setup_dir))
        experiment = os.path.basename(experiment_dir)
        setup = os.path.basename(setup_dir)
        task_name = data.get('task')
        task_script = os.path.join(_TASKS_DIR, task_name or '', (task_name or '') + '.py')
        if not task_name or not os.path.exists(task_script):
            continue
        results.append({
            'experiment': experiment, 'setup': setup, 'board': data.get('board'),
            'task_name': task_name, 'task_script': task_script,
            # Only subjects that still exist on disk -- a subject deleted outside this tool (or by
            # an older version of it) stays listed in the setup JSON, and used to show up in every
            # Stage dropdown as a ghost that couldn't be launched or detached.
            'subjects': [n for n in (data.get('subjects') or []) if _subject_exists(n)],
        })
    results.sort(key=lambda c: build_training_log._stage_sort_key(c['task_name']))
    return results


def discover_users():
    results = []
    for path in sorted(glob.glob(os.path.join(_PROJECT_DIR, 'users', '*', '*.json'))):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            continue
        name = os.path.splitext(os.path.basename(path))[0]
        uuid_ = data.get('__UUID4__')
        if uuid_:
            results.append({'name': name, 'uuid': uuid_})
    return results


def load_subject_uuid(subject_name):
    path = os.path.join(_PROJECT_DIR, 'subjects', subject_name, subject_name + '.json')
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)['__UUID4__']


def load_board_config(board_name):
    path = os.path.join(_PROJECT_DIR, 'boards', board_name, board_name + '.json')
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def discover_subjects():
    """ Returns [{'name', 'uuid'}] for every real subject on disk, regardless of whether it's
    attached to a Setup yet -- used by the New Setup dialog's subject checklist (discover_stage_
    configs()'s own per-stage subject lists only include already-attached subjects). Only the json
    file whose own basename matches its PARENT directory name counts as the real subject
    descriptor (same convention discover_all_tasks() already uses) -- a subject's own folder also
    holds OTHER json files that happen to match a naive `subjects/*/*.json` glob (confirmed
    directly: session_state.py's own wheel_shaping_state.json, the per-subject cross-session
    training-state file, sits right next to e.g. test_stage1.json inside subjects/test_stage1/ --
    without this filter it showed up as a bogus "wheel_shaping_state" subject, once per subject
    folder that has one). """
    results = []
    for path in sorted(glob.glob(os.path.join(_PROJECT_DIR, 'subjects', '*', '*.json'))):
        name = os.path.splitext(os.path.basename(path))[0]
        if name != os.path.basename(os.path.dirname(path)):
            continue
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            continue
        results.append({'name': name, 'uuid': data.get('__UUID4__')})
    return results


def discover_boards():
    """ Same "basename must match parent directory name" filter as discover_subjects()/
    discover_all_tasks() -- no board folder happens to have a stray extra json file today, but
    this is the identical bug class pre-empted the same way rather than waiting for it to bite. """
    results = []
    for path in sorted(glob.glob(os.path.join(_PROJECT_DIR, 'boards', '*', '*.json'))):
        name = os.path.splitext(os.path.basename(path))[0]
        if name != os.path.basename(os.path.dirname(path)):
            continue
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            continue
        results.append({'name': name, 'uuid': data.get('__UUID4__')})
    return results


def discover_all_tasks():
    """ Every task script this project has under tasks/<name>/<name>.py -- for the New Setup
    dialog's Task dropdown (discover_stage_configs() only lists tasks an existing Setup already
    uses, not the raw set of available scripts, since a new Setup needs to be able to pick one that
    isn't in use anywhere yet, e.g. stage5_resume_staircases before its first Setup exists). """
    results = []
    for path in sorted(glob.glob(os.path.join(_TASKS_DIR, '*', '*.py'))):
        name = os.path.splitext(os.path.basename(path))[0]
        if name == os.path.basename(os.path.dirname(path)) and not name.startswith('_'):
            results.append(name)
    return results


def _validate_new_name(name):
    if not name or not _VALID_NAME_RE.match(name):
        raise ValueError("Name must be non-empty and contain only letters, numbers, '_' or '-'.")


def _scadict_boilerplate(def_text):
    """ Matches every real project JSON's own boilerplate shape exactly -- confirmed against
    `sca/formats/json.py`'s `scadict`/`dump()` (isoformat(' ') timestamps, this fixed key set) and
    `sca/format_flags.py`'s own flag names, by reading PyBpod's actual source. """
    now = datetime.datetime.now().isoformat(' ')
    return {
        '__UUID4__': str(uuid.uuid4()),
        '__CREATED-ON__': now,
        '__UPDATED-ON__': now,
        '__SOFTWARE__': 'PyBpod GUI API v1.8.2',
        '__DEF-URL__': 'http://pybpod.readthedocs.org',
        '__DEF-TEXT__': def_text,
    }


def _write_pybpod_json(path, data):
    # indent=4, sort_keys=True -- matches sca/formats/json.py's own dump(), confirmed against
    # every real example file already in this project (alphabetically-sorted keys).
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, sort_keys=True)


def create_subject(name):
    """ Writes subjects/<name>/<name>.json, matching pybpod-gui-api's own SubjectIO.save() shape
    exactly -- unassigned to any Setup yet. Note "setup": "None" is literally the STRING "None",
    not JSON null -- confirmed from the real source (`str(self.setup.uuid4 if self.setup else
    None)`), reproduced exactly rather than "fixed" to null, so a real PyBpod GUI opening this
    project later sees byte-for-byte what it would have written itself. """
    _validate_new_name(name)
    subject_dir = os.path.join(_PROJECT_DIR, 'subjects', name)
    if os.path.exists(subject_dir):
        raise ValueError('Subject "{0}" already exists.'.format(name))
    os.makedirs(subject_dir)
    data = _scadict_boilerplate('This file contains information about a subject used on PyBpod GUI.')
    data['setup'] = 'None'
    _write_pybpod_json(os.path.join(subject_dir, name + '.json'), data)
    return data['__UUID4__']


def create_setup(name, board_name, task_name, subject_names):
    """ Creates experiments/<name>/setups/<name>/ and writes <name>.json matching pybpod-gui-api's
    own SetupBaseIO.save() shape exactly: board/task as plain name strings (not UUIDs), subjects as
    a name list, __EXTERNAL-REF__ containing the board's UUID plus every assigned subject's UUID
    (add_external_ref()). `name` is used for BOTH the experiment and setup directory -- matches
    this project's own established 1:1 convention (every existing Stage1/2/3 example does this;
    confirmed no experiment-level JSON is ever written at all -- an "experiment" is purely a
    directory). Also updates each assigned subject's own JSON "setup" field to this new setup's
    UUID -- the two-way link the real PyBpod GUI itself maintains. """
    _validate_new_name(name)
    setup_dir = os.path.join(_PROJECT_DIR, 'experiments', name, 'setups', name)
    setup_json_path = os.path.join(setup_dir, name + '.json')
    if os.path.exists(setup_json_path):
        raise ValueError('An experiment/setup named "{0}" already exists.'.format(name))
    # setup_dir itself (NOT its own .json) can legitimately already exist here -- delete_setup()
    # deliberately only removes the .json ("unregister only," per explicit instruction), leaving
    # the directory (and any real recorded session data in its own sessions/ subfolder) in place.
    # Re-registering the same name afterward must still succeed -- that's the whole point of
    # "unregister" rather than "retire this name forever."

    board_cfg_path = os.path.join(_PROJECT_DIR, 'boards', board_name, board_name + '.json')
    with open(board_cfg_path, 'r', encoding='utf-8') as f:
        board_uuid = json.load(f)['__UUID4__']

    subject_paths = {n: os.path.join(_PROJECT_DIR, 'subjects', n, n + '.json') for n in subject_names}
    subject_uuids = {}
    for subject_name, subject_path in subject_paths.items():
        with open(subject_path, 'r', encoding='utf-8') as f:
            subject_uuids[subject_name] = json.load(f)['__UUID4__']

    os.makedirs(setup_dir, exist_ok=True)   # exist_ok -- see the comment above
    data = _scadict_boilerplate(
        'This file contains information about a PyBpod experiment setup.')
    data['board'] = board_name
    data['task'] = task_name
    data['subjects'] = list(subject_names)
    data['detached'] = False
    data['variables'] = []
    data['update-variables'] = False
    data['__EXTERNAL-REF__'] = [board_uuid] + [subject_uuids[n] for n in subject_names]
    _write_pybpod_json(setup_json_path, data)

    setup_uuid = data['__UUID4__']
    for subject_name, subject_path in subject_paths.items():
        with open(subject_path, 'r', encoding='utf-8') as f:
            subject_data = json.load(f)
        subject_data['setup'] = setup_uuid
        subject_data['__UPDATED-ON__'] = datetime.datetime.now().isoformat(' ')
        _write_pybpod_json(subject_path, subject_data)

    return setup_uuid


def attach_subject_to_setup(experiment, setup, subject_name):
    """ Attaches an EXISTING subject to an EXISTING setup -- the operation create_setup() has no
    equivalent for once a setup already exists (it only builds brand-new ones). Confirmed as a
    real, already-hit gap: a subject created via "New Subject..." starts with no setup at all
    (create_subject()'s own "setup": "None"), and a stage's own Subject dropdown in this dashboard
    is populated entirely from its Setup JSON's own `subjects` list -- so a never-attached subject
    is simply never selectable for any stage until this is called. Same two-way-link convention as
    create_setup(): updates the setup's own `subjects`/`__EXTERNAL-REF__` (both no-ops if the
    subject is already listed) and the subject's own `setup` field (a subject belongs to only one
    setup at a time, matching the real PyBpod GUI's own model -- re-attaching an already-attached
    subject to a different setup simply moves it, same as create_setup() already does
    unconditionally). """
    setup_path = os.path.join(_PROJECT_DIR, 'experiments', experiment, 'setups', setup,
                               setup + '.json')
    if not os.path.exists(setup_path):
        raise ValueError('Setup "{0}/{1}" does not exist.'.format(experiment, setup))
    subject_path = os.path.join(_PROJECT_DIR, 'subjects', subject_name, subject_name + '.json')
    if not os.path.exists(subject_path):
        raise ValueError('Subject "{0}" does not exist.'.format(subject_name))

    with open(subject_path, 'r', encoding='utf-8') as f:
        subject_data = json.load(f)
    subject_uuid = subject_data['__UUID4__']

    with open(setup_path, 'r', encoding='utf-8') as f:
        setup_data = json.load(f)
    if subject_name not in setup_data.get('subjects', []):
        setup_data.setdefault('subjects', []).append(subject_name)
    if subject_uuid not in setup_data.get('__EXTERNAL-REF__', []):
        setup_data.setdefault('__EXTERNAL-REF__', []).append(subject_uuid)
    setup_data['__UPDATED-ON__'] = datetime.datetime.now().isoformat(' ')
    _write_pybpod_json(setup_path, setup_data)

    setup_uuid = setup_data['__UUID4__']
    subject_data['setup'] = setup_uuid
    subject_data['__UPDATED-ON__'] = datetime.datetime.now().isoformat(' ')
    _write_pybpod_json(subject_path, subject_data)
    return setup_uuid


def detach_subject_from_setup(experiment, setup, subject_name):
    """ The reverse of attach_subject_to_setup() -- removes subject_name from the setup's own
    `subjects` list (and its UUID from `__EXTERNAL-REF__`), and resets the subject's own `setup`
    field back to "None" (ONLY if it currently points at THIS setup -- if the subject was already
    reattached elsewhere since, leave that alone rather than clobbering a different, newer
    assignment). This only changes the Setup's CURRENT subject roster -- which subject(s) are
    offered to launch a NEW session -- it never touches any already-recorded session's own
    historical SUBJECT-NAME (immutable, baked in at record time) or any other file. Used to fix a
    Setup that was given a badly-named placeholder subject (e.g. "test_stage1", stage information
    baked into what should be a plain mouse ID -- see subject_aliases.json's own docstring in
    build_training_log.py) once the right one has been created/attached instead. """
    setup_path = os.path.join(_PROJECT_DIR, 'experiments', experiment, 'setups', setup,
                               setup + '.json')
    if not os.path.exists(setup_path):
        raise ValueError('Setup "{0}/{1}" does not exist.'.format(experiment, setup))
    subject_path = os.path.join(_PROJECT_DIR, 'subjects', subject_name, subject_name + '.json')
    with open(setup_path, 'r', encoding='utf-8') as f:
        setup_data = json.load(f)
    if not os.path.exists(subject_path):
        # Already deleted (outside this tool): drop the name and any UUID that matches nothing on
        # disk any more, instead of refusing and leaving a permanent ghost entry.
        if subject_name in setup_data.get('subjects', []):
            setup_data['subjects'].remove(subject_name)
        _prune_setup_data(setup_data, _known_uuids())
        setup_data['__UPDATED-ON__'] = datetime.datetime.now().isoformat(' ')
        _write_pybpod_json(setup_path, setup_data)
        return
    with open(subject_path, 'r', encoding='utf-8') as f:
        subject_data = json.load(f)
    subject_uuid = subject_data['__UUID4__']

    if subject_name in setup_data.get('subjects', []):
        setup_data['subjects'].remove(subject_name)
    if subject_uuid in setup_data.get('__EXTERNAL-REF__', []):
        setup_data['__EXTERNAL-REF__'].remove(subject_uuid)
    setup_data['__UPDATED-ON__'] = datetime.datetime.now().isoformat(' ')
    _write_pybpod_json(setup_path, setup_data)

    if subject_data.get('setup') == setup_data['__UUID4__']:
        subject_data['setup'] = 'None'
        subject_data['__UPDATED-ON__'] = datetime.datetime.now().isoformat(' ')
        _write_pybpod_json(subject_path, subject_data)


def delete_setup(experiment, setup):
    """ Removes ONLY the Setup's own JSON descriptor (experiments/<experiment>/setups/<setup>/
    <setup>.json) -- per explicit instruction, this "unregisters" the stage from this dashboard's
    own Stage dropdown (discover_stage_configs() scans for *.json files under setups/*, so
    removing this file alone is enough to make it disappear) WITHOUT touching any of its real
    recorded session data (experiments/<experiment>/setups/<setup>/sessions/ and everything under
    it), which is left on disk untouched in case it's needed later -- per explicit instruction,
    NOT the same scope as deleting an animal's own training-log row or session files.
    Detaches every subject still listed in it first (same reasoning as delete_subject() below --
    a subject's own `setup` field left pointing at a now-unregistered setup would be a dangling
    reference otherwise). Irreversible for the registration itself (recreating a Setup of the same
    name via "New Setup..." gets a fresh UUID, not the original); the caller (GUI) is responsible
    for confirming with the user first. """
    setup_dir = os.path.join(_PROJECT_DIR, 'experiments', experiment, 'setups', setup)
    setup_path = os.path.join(setup_dir, setup + '.json')
    if not os.path.exists(setup_path):
        raise ValueError('Setup "{0}/{1}" does not exist.'.format(experiment, setup))
    with open(setup_path, 'r', encoding='utf-8') as f:
        setup_data = json.load(f)
    for subject_name in list(setup_data.get('subjects', [])):
        detach_subject_from_setup(experiment, setup, subject_name)
    os.remove(setup_path)


def delete_subject(name):
    """ Deletes subjects/<name>/ entirely (its own identity json AND any other per-subject file
    living next to it, e.g. session_state.py's own persisted wheel_shaping_state.json) -- real
    recorded session data is untouched (it lives under experiments/.../sessions/, keyed by Setup,
    not by subject folder), so this only removes the subject's own identity/cross-session-state
    record, never any animal's already-recorded history.

    First auto-detaches it from EVERY setup that still lists it -- a subject can appear in more
    than one setup's own `subjects` list even though its own `setup` field points at only the most
    recently-attached one (attach_subject_to_setup() adds to a new setup's list without removing
    it from any previous one), so checking only the "current" setup would miss older, stale
    listings and leave them as dangling name/UUID references to something that no longer exists.
    Uses the exact same detach_subject_from_setup() the "Detach Subject from Setup..." button
    already calls.

    Irreversible (a real shutil.rmtree()); the caller (GUI) is responsible for confirming with the
    user first. """
    subject_dir = os.path.join(_PROJECT_DIR, 'subjects', name)
    if not os.path.exists(subject_dir):
        raise ValueError('Subject "{0}" does not exist.'.format(name))
    for cfg in discover_stage_configs():
        if name in cfg['subjects']:
            detach_subject_from_setup(cfg['experiment'], cfg['setup'], name)
    shutil.rmtree(subject_dir)


# --- session-folder creation + subprocess launch, mirroring board_com.py:run_task() exactly ----

def build_user_settings_content(board_name, board_cfg, experiment, setup, session_name,
                                 session_path, task_name, subject_name, subject_uuid, user_name,
                                 user_uuid):
    bnc = board_cfg.get('enabled-bncports')
    wired = board_cfg.get('enabled-wiredports')
    behavior = board_cfg.get('enabled-behaviorports')
    # PYBPOD_SUBJECTS uses plain Python str() (single-quoted); PYBPOD_CREATOR uses json.dumps()
    # (double-quoted) -- confirmed against board_com.py's own run_task(), which builds them via
    # two genuinely different serializations (str([s.name, str(s.uuid4)]) vs.
    # json.dumps([user.name, str(user.uuid4), user.connection])), not a copy-paste inconsistency.
    subjects_field = '"{0}"'.format(str([subject_name, subject_uuid]))
    creator_field = json.dumps([user_name, user_uuid, 'local'])
    session_path_field = os.path.abspath(session_path).encode('unicode_escape').decode()
    return _USER_SETTINGS_TEMPLATE.format(
        serial_port=board_cfg.get('serial-port'), net_port=board_cfg.get('net-port'),
        bnc_ports_line=('BPOD_BNC_PORTS_ENABLED = {0}'.format(bnc) if bnc else ''),
        wired_ports_line=('BPOD_WIRED_PORTS_ENABLED = {0}'.format(wired) if wired else ''),
        behavior_ports_line=('BPOD_BEHAVIOR_PORTS_ENABLED = {0}'.format(behavior)
                              if behavior else ''),
        protocol=task_name, creator=creator_field, project=VAR_PROJECT_NAME, experiment=experiment,
        board=board_name, setup=setup, session=session_name, session_path=session_path_field,
        subjects=subjects_field, firmware=VAR_BPOD_FIRMWARE_VERSION)


class TaskLauncher(object):
    """ Creates a brand-new session folder + user_settings.py and launches the task subprocess,
    the exact same way board_com.py:run_task() does -- so the launched task script sees nothing
    different from being started by the real PyBpod GUI. Owns the Popen handle for Stop/Kill. """

    def __init__(self):
        self.proc = None
        self.session_path = None

    @property
    def is_running(self):
        return self.proc is not None and self.proc.poll() is None

    def start(self, stage_config, subject_name, user, reward_ul=VAR_DEFAULT_REWARD_UL):
        if self.is_running:
            raise RuntimeError('A task launched from this dashboard is already running.')

        board_cfg = load_board_config(stage_config['board'])
        subject_uuid = load_subject_uuid(subject_name)

        session_name = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        session_path = os.path.join(_PROJECT_DIR, 'experiments', stage_config['experiment'],
                                     'setups', stage_config['setup'], 'sessions', session_name)
        os.makedirs(session_path, exist_ok=True)

        with open(os.path.join(session_path, '__init__.py'), 'w') as f:
            pass

        settings_content = build_user_settings_content(
            stage_config['board'], board_cfg, stage_config['experiment'], stage_config['setup'],
            session_name, session_path, stage_config['task_name'], subject_name, subject_uuid,
            user['name'], user['uuid'])
        with open(os.path.join(session_path, 'user_settings.py'), 'w') as f:
            f.write(settings_content)

        env = os.environ.copy()
        # Only the session folder itself -- NOT this (parent) process's own sys.path.
        # board_com.py's real run_task() does append its own sys.path here too, but that's only
        # safe there because the launcher and the launched task always share the SAME interpreter.
        # This dashboard deliberately launches the task under a DIFFERENT interpreter
        # (VAR_TASK_PYTHON_EXE) than whatever runs the GUI itself -- forwarding THIS process's own
        # sys.path (e.g. a Python 3.9 install's own stdlib paths) onto a Python 3.6 child's
        # PYTHONPATH corrupts its stdlib resolution entirely. Confirmed directly: doing so produced
        # "Fatal Python error: Py_Initialize: can't initialize sys standard streams" /
        # "No module named '_abc'" from the child trying to load Python 3.9's io.py/abc.py under
        # Python 3.6. The session folder alone is all `import user_settings` inside pybpodapi
        # itself actually needs.
        env['PYTHONPATH'] = os.path.abspath(session_path)
        # Tells the task script (stage1-4, see their own identical comment at their own
        # WheelShapingPlots(...) construction) to suppress its own separate live-plot popup window
        # -- this dashboard's own embedded view already shows the same data, reconstructed from the
        # session CSV. Absent entirely for a real PyBpod-GUI-launched session (board_com.py never
        # sets this), so that path is completely unaffected.
        env['RUN_SESSION_DASHBOARD'] = '1'
        # Overrides each stage script's own hardcoded VAR_REWARD_UL default -- see this dashboard's
        # own reward_ul_input field. Absent entirely for a real PyBpod-GUI-launched session (same
        # reasoning as RUN_SESSION_DASHBOARD above), so that path keeps each script's own default.
        env['RUN_SESSION_REWARD_UL'] = str(reward_ul)

        stdout_log = open(os.path.join(session_path, 'dashboard_launch_stdout.log'), 'w')
        stderr_log = open(os.path.join(session_path, 'dashboard_launch_stderr.log'), 'w')
        self.proc = subprocess.Popen(
            [VAR_TASK_PYTHON_EXE, os.path.abspath(stage_config['task_script'])],
            stdin=subprocess.PIPE, stdout=stdout_log, stderr=stderr_log, cwd=session_path, env=env)
        self.session_path = session_path
        return session_path

    def stop(self, graceful=True):
        """ Graceful: writes 'close\\r\\n' to stdin -- the exact same command board_com.py's own
        stop_task() sends, causing the task's current Bpod call to unwind and its normal
        post-loop cleanup (struct export, Bpod.close() writing SESSION-ENDED) to still run. Caller
        is responsible for escalating to terminate()/kill() if the process hasn't exited after
        VAR_STOP_GRACE_S (see SessionRunnerWindow._on_stop_check). """
        if self.proc is None or self.proc.poll() is not None:
            return
        try:
            if graceful and self.proc.stdin is not None:
                self.proc.stdin.write('close\r\n'.encode())
                self.proc.stdin.flush()
            else:
                self.proc.terminate()
        except Exception:
            print(traceback.format_exc(), flush=True)

    def force_kill(self):
        if self.proc is not None and self.proc.poll() is None:
            try:
                self.proc.kill()
            except Exception:
                print(traceback.format_exc(), flush=True)


# --- CSV -> WheelShapingPlots.add_trial() reconstruction ---------------------------------------

def _val_float(trial, key, session_vals=None):
    """ Checks this trial's own vals first, falling back to session_vals -- some registrations
    (e.g. Stage 1's own THRESHOLD_DEG, which only changes across SESSIONS not within one) happen
    once before the first TRIAL marker rather than every trial, landing in session_vals instead of
    any individual trial's own vals (see session_csv_parser.find_val_backward()'s own docstring for
    the same distinction). """
    v = trial['vals'].get(key)
    if v is None and session_vals is not None:
        v = session_vals.get(key)
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _trial_anchor(trial, info, session_vals):
    """ Host time this trial's Bpod machine ran (see session_csv_parser.trial_host_anchor()) --
    what its Port1In/state times are added to. Falls back to the protocol's legacy anchor_key only
    when the trial has nothing better. For Stage 1/2 that legacy key (TRIAL_START) was wrong in two
    ways: it is registered BEFORE the trial's own TRIAL row, so the parser files it under the
    PREVIOUS trial (every lick/reward time was offset by roughly one whole trial), and it is a
    pre-upload host time, 0.2-1.4s before the hardware trial actually started. """
    anchor = session_csv_parser.trial_host_anchor(trial)
    if anchor is not None:
        return anchor
    return _val_float(trial, info['anchor_key'], session_vals)


def _val_str(trial, key, session_vals=None):
    v = trial['vals'].get(key)
    if v is None and session_vals is not None:
        v = session_vals.get(key)
    return v


class PlotFeeder(object):
    """ Owns one long-lived WheelShapingPlots instance for the currently-watched session and feeds
    it incrementally -- each poll() call only replays trials beyond the count already fed, mirroring
    how a real stage script itself calls add_trial() once per trial rather than replaying the whole
    session from scratch every tick. """

    def __init__(self, protocol_name, first_trial_vals, session_vals):
        info = _DASHBOARD_PROTOCOL_INFO[protocol_name]
        self.protocol_name = protocol_name
        self.info = info
        self.session_vals = session_vals
        prev_session_values = {}
        starting_threshold = (first_trial_vals.get(info['threshold_key'])
                               or session_vals.get(info['threshold_key']))
        try:
            starting_threshold = float(starting_threshold) if starting_threshold is not None else None
        except (TypeError, ValueError):
            starting_threshold = None
        if starting_threshold is not None:
            prev_session_values['threshold_deg'] = starting_threshold
        session_status = {}
        cue_abort = (first_trial_vals.get('CUE_ABORT_THRESHOLD_DEG')
                     or session_vals.get('CUE_ABORT_THRESHOLD_DEG'))
        if cue_abort is not None:
            try:
                session_status['in_trial_threshold_deg'] = float(cue_abort)
            except (TypeError, ValueError):
                pass
        # Stage 3 only -- SESSION_SIDE is a session-level VAL (registered once, before the first
        # TRIAL marker, so it only ever lands in session_vals, never first_trial_vals -- but
        # checking both mirrors every other lookup here and costs nothing). Missing this was
        # exactly why the dashboard's own embedded progress panel showed "today's side: ?" even
        # though the real task script's own separate live-plot window (which builds this same
        # dict directly, not through a CSV round-trip) always showed the real side correctly.
        session_side = first_trial_vals.get('SESSION_SIDE') or session_vals.get('SESSION_SIDE')
        if session_side is not None:
            session_status['session_side'] = session_side
        # Stage 4 only -- CLICK_LEVEL_ATTENUATED is a per-trial VAL, logged as the string
        # 'True'/'False' (see session_csv_parser's own str-vs-float VAL convention).
        click_attenuated = first_trial_vals.get('CLICK_LEVEL_ATTENUATED')
        if click_attenuated is not None:
            session_status['click_attenuated'] = (click_attenuated == 'True')
        # Stage 5 only -- STAIRCASE_ACTIVE is per-trial, STAIRCASE_ACTIVE_THIS_SESSION is the
        # session-level fallback (see stage5_resume_staircases.py's own two registrations).
        staircase_active = (first_trial_vals.get('STAIRCASE_ACTIVE')
                             or session_vals.get('STAIRCASE_ACTIVE_THIS_SESSION'))
        if staircase_active is not None:
            session_status['staircase_active'] = staircase_active
        self.plots = wheel_shaping_plots.WheelShapingPlots(
            info['stage'], VAR_THRESHOLD_FINAL_DEG, prev_session_values=prev_session_values,
            session_status=session_status, embed=True)
        self._n_fed = 0
        self._replay_prev_outcome = None
        self._replay_prev_side = None

    def feed_new_trials(self, trials, session_ended=False):
        """ Feeds every trial whose rows are complete, in order, exactly once. A poll can land while
        the task is still writing the most recent trial's STATE/EVENT/VAL burst (pybpod flushes row
        by row), and `self._n_fed` is a COUNT, so a half-written trial must never be fed: the last
        trial is fed only once _trial_rows_complete() sees its final post-trial row; every earlier
        trial is complete by construction. This replaced holding back the last entry
        unconditionally, which -- on top of real_trials() already dropping the running trial --
        kept the plots ~2 trials behind.

        `session_ended=True` (SESSION-ENDED is present, i.e. Bpod.close() already ran) feeds
        everything that's left. """
        n_ready = 0
        for i, trial in enumerate(trials):
            if session_ended or i < len(trials) - 1 or _trial_rows_complete(trial):
                n_ready = i + 1
            else:
                break
        for trial in trials[self._n_fed:n_ready]:
            kwargs = self._reconstruct_kwargs(trial)
            if kwargs is not None:
                self.plots.add_trial(**kwargs)
        self._n_fed = max(self._n_fed, n_ready)

    def _reconstruct_kwargs(self, trial):
        info = self.info
        config = session_csv_parser.PROTOCOL_CONFIG[self.protocol_name]
        outcome_raw, event_side, _reward_duration, _consumed = session_csv_parser.classify_trial(
            trial, config)
        side = _val_str(trial, 'TRIAL_SIDE', self.session_vals) or event_side or 'R'
        outcome = info['outcome_map'].get(outcome_raw)
        if outcome is None:
            return None   # 'unknown' -- no recognizable outcome states visited, nothing to plot

        threshold_deg = _val_float(trial, info['threshold_key'], self.session_vals)
        magnitude_deg = (threshold_deg or 0.0) if outcome_raw in info['crossed_outcomes'] else 0.0
        anchor = _trial_anchor(trial, info, self.session_vals)

        kwargs = dict(side=side, magnitude_deg=magnitude_deg, threshold_deg=threshold_deg or 0.0,
                      outcome=outcome)

        lick_times_abs = []
        if anchor is not None:
            lick_times_abs = [anchor + t for t in trial['events'].get('Port1In', [])]
        kwargs['lick_times_abs'] = lick_times_abs

        reward_state_name = _OUTCOME_STATE_NAME.get(self.protocol_name, {}).get('rewarded')
        if outcome_raw == 'rewarded' and anchor is not None:
            name = reward_state_name
            if isinstance(name, tuple):
                name = 'RewardL' if side == 'L' else 'RewardR'
            state = trial['states'].get(name)
            if state is not None and state[0] is not None:
                kwargs['reward_time_abs'] = anchor + state[0]

        if info['stage'] == 1:
            kwargs['gain_mult'] = _val_float(trial, 'GAIN_MULT', self.session_vals)
        elif info['stage'] == 2:
            kwargs['direction_ratio'] = _val_float(trial, 'DIRECTION_RATIO', self.session_vals)
            iti_state = trial['states'].get('ITI')
            if iti_state is not None and iti_state[2] is not None:
                kwargs['iti_s'] = iti_state[2]
        else:   # stage 3/4
            trial_type = _val_str(trial, 'TRIAL_TYPE', self.session_vals)
            kwargs['trial_type'] = trial_type
            in_trial_threshold = _val_float(trial, 'CUE_ABORT_THRESHOLD_DEG', self.session_vals)
            if in_trial_threshold is not None:
                kwargs['in_trial_threshold_deg'] = in_trial_threshold

            if trial_type == 'repeat':
                p_right_target = 1.0 if side == 'R' else 0.0
            elif self._replay_prev_outcome == 'incorrect':
                p_right_target = (debiasing.VAR_DEBIAS_REPEAT_PROB if self._replay_prev_side == 'R'
                                   else 1.0 - debiasing.VAR_DEBIAS_REPEAT_PROB)
            else:
                p_right_target = 0.5
            kwargs['p_right_target'] = p_right_target

            if outcome_raw in ('rewarded', 'incorrect'):
                outcome_state_name = _OUTCOME_STATE_NAME[self.protocol_name][outcome_raw]
                state = trial['states'].get(outcome_state_name)
                if state is not None and state[0] is not None:
                    kwargs['response_time_s'] = state[0]
                n_l = _val_float(trial, 'N_CLICKS_L', self.session_vals)
                n_r = _val_float(trial, 'N_CLICKS_R', self.session_vals)
                if n_l is not None and n_r is not None:
                    kwargs['click_diff'] = n_r - n_l
                self._replay_prev_outcome = 'correct' if outcome_raw == 'rewarded' else 'incorrect'
                self._replay_prev_side = side
            elif outcome_raw == 'no_movement':
                self._replay_prev_outcome = None
                self._replay_prev_side = side
            # outcome_raw == 'aborted': prev_outcome/prev_side deliberately left unchanged --
            # matches the real task scripts' own `continue` skipping that update for an Abort.

        return kwargs


def _trial_rows_complete(trial):
    """ True once a finished trial's post-trial rows are all in the CSV: register_port_events()
    (a `*LICK_TIMES_OUT` VAL) is the last thing every stage registers per Bpod machine. Lets the
    feeder show a trial the moment it is complete instead of holding it back until the NEXT trial
    exists -- that hold-back, on top of real_trials() already excluding the running trial, kept the
    live plots ~2 trials behind (session 20261023-122212). """
    return any(key.endswith('LICK_TIMES_OUT') for key in trial['vals'])


class WheelSeriesCache(object):
    """ parse_wheel_position_series() for a growing session, without redoing finished trials:
    a trial's WHEEL_POS rows are complete once a later trial exists, so those are decimated once
    and kept; only the last trial is recomputed each poll. """

    def __init__(self, decimate_s=0.5):
        self.decimate_s = decimate_s
        self._n_done = 0
        self._xs, self._ys = [], []

    def series(self, trials):
        done = trials[:-1]
        if len(done) > self._n_done:
            xs, ys = parse_wheel_position_series(done[self._n_done:], self.decimate_s)
            last_t = self._xs[-1] if self._xs else None
            for x, y in zip(xs, ys):
                if last_t is None or x - last_t >= self.decimate_s:
                    self._xs.append(x)
                    self._ys.append(y)
                    last_t = x
            self._n_done = len(done)
        tail_x, tail_y = parse_wheel_position_series(trials[-1:], self.decimate_s) if trials else ([], [])
        start = self._xs[-1] if self._xs else None
        keep = [(x, y) for x, y in zip(tail_x, tail_y) if start is None or x > start]
        return self._xs + [x for x, _ in keep], self._ys + [y for _, y in keep]


def parse_wheel_position_series(trials, decimate_s=0.5):
    """ Every WHEEL_POS VAL across all trials, as (elapsed_s, position_deg) pairs -- WHEEL_POS is
    never collapsed to the last sample per trial (see session_csv_parser.STREAM_KEYS), so the full
    stream survives. Decimated to at most one sample per `decimate_s` seconds, same principle as
    the real "Wheel Position" plot's own DISPLAY_SAMPLE_INTERVAL_S, so a long session doesn't force
    an ever-growing full-resolution redraw every poll.

    Plots the RAW, +/-180deg-wrapped position exactly as the rotary reports it -- deliberately NOT
    per-trial-unwrapped the way derive_intrial_threshold.py's own trace is. A previous version of
    this function did unwrap (added at the same time as that one, on the same reasoning); measured
    against a real session it drove this panel's y-range to -1635..+895deg, because every genuine
    wrap crossing of a briskly-turned wheel accumulates into true total rotation (one trial there
    crossed the boundary 5 times in 5.5s, ~4.5 rotations). That cumulative total is the right
    quantity for a sliding-window excursion percentile; it is the wrong one for a monitoring panel
    whose threshold reference lines sit at single-digit degrees. The rotary itself really does wrap
    at +/-180 (verified across every session recorded on this rig: zero samples beyond it), so raw
    values are bounded by construction and a wrap reads as an obvious full-height vertical jump.
    """
    samples = []
    for trial in trials:
        raw = trial['vals'].get('WHEEL_POS')
        if not raw:
            continue
        for entry in raw:
            try:
                t_str, pos_str = entry.split(',')[:2]   # stream rows add a 3rd field (module time)
                samples.append((float(t_str), float(pos_str)))
            except (ValueError, AttributeError):
                continue
    samples.sort()

    xs, ys = [], []
    last_t = None
    for t, pos in samples:
        if last_t is not None and (t - last_t) < decimate_s:
            continue
        xs.append(t)
        ys.append(pos)
        last_t = t
    return xs, ys


def build_trial_markers(trials, protocol_name, session_vals):
    """ For the wheel-position plot's overlay: returns (trial_start_times, threshold_segments).
    trial_start_times is a sorted list of each trial's own anchor time (a light vertical line per
    trial). threshold_segments is a list of (x_start, x_end, threshold_deg) -- one horizontal
    segment per trial, spanning from that trial's own start to the next trial's start (or a short
    pad past the last one), so a per-trial threshold CHANGE (e.g. Stage 2's own staircase) shows as
    a literal step rather than one flat line silently averaging over the whole session. Threshold
    falls back to the session-level VAL the same way _val_float() already does (e.g. Stage 1's own
    THRESHOLD_DEG, constant for the whole session, registered once before any trial). """
    if protocol_name not in _DASHBOARD_PROTOCOL_INFO:
        return [], []
    info = _DASHBOARD_PROTOCOL_INFO[protocol_name]
    points = []
    for trial in trials:
        anchor = _trial_anchor(trial, info, session_vals)
        if anchor is None:
            continue
        threshold = _val_float(trial, info['threshold_key'], session_vals)
        points.append((anchor, threshold))
    points.sort(key=lambda p: p[0])
    starts = [x for x, _threshold in points]
    segments = []
    for i, (x, threshold) in enumerate(points):
        if threshold is None:
            continue
        x_end = points[i + 1][0] if i + 1 < len(points) else x + 5.0
        segments.append((x, x_end, threshold))
    return starts, segments


# --- weight / notes / baseline write path, reusing build_training_log.py's own column layout ---

def find_row_for_session(ws, table_start_row, col_map, session_started):
    key_col = col_map.get('session_started')
    if key_col is None:
        return None
    for row in range(table_start_row + 1, ws.max_row + 1):
        raw = ws.cell(row=row, column=key_col).value
        if not raw:
            continue
        members = {k.strip() for k in str(raw).split(';') if k.strip()}
        if session_started in members:
            return row
    return None


def _write_header_fields(ws, header_fields):
    """ DOB/Sex/Strain/Baseline weight, in that fixed row order (rows 2-5, column B -- matches
    build_training_log.py's own _HEADER_FIELD_LABELS layout exactly). Only writes a field if
    given (non-empty) -- never overwrites an existing value with blank, so a field pre-filled
    from load_header_fields() and left untouched by the caller round-trips as a no-op. """
    for i, value in enumerate(header_fields):
        if value:
            ws.cell(row=2 + i, column=2, value=value)


def save_subject_header_fields(subject, header_fields):
    """ Writes DOB/Sex/Strain/Baseline weight into training_log.xlsx's own per-subject sheet
    header block, creating the sheet (and the workbook itself) if either doesn't exist yet --
    used right at subject-creation time (see NewSubjectDialog) so these rarely-if-ever-changing
    fields are asked ONCE, not re-prompted on every session's own weight/notes save. Deliberately
    does NOT call build_training_log.build_workbook() first (unlike save_manual_fields()) -- that
    only creates a sheet for a subject it already found a real session CSV for, which a
    brand-new subject never has yet; this opens/creates the workbook directly instead, same
    "load if present else start fresh, drop the default blank sheet" shape build_workbook()
    itself uses. """
    if os.path.exists(_OUTPUT_XLSX):
        wb = openpyxl.load_workbook(_OUTPUT_XLSX)
    else:
        wb = openpyxl.Workbook()
        wb.remove(wb.active)
    ws, _table_start_row = build_training_log._get_or_create_subject_sheet(wb, subject)
    _write_header_fields(ws, header_fields)
    wb.save(_OUTPUT_XLSX)


def save_manual_fields(subject, session_started, weight_g, weight_after_g, hand_watered, notes,
                        header_fields):
    """ Regenerates training_log.xlsx first (so this session's own row definitely exists / is
    current -- same call "Update Training Log" already makes), then writes directly into the
    manual cells for that row plus the subject's header block (DOB/Sex/Strain/Baseline weight --
    see _write_header_fields()'s own docstring for why a blank field is always a safe no-op).
    Returns True if the session's own row was found and written. """
    build_training_log.build_workbook()
    wb = openpyxl.load_workbook(_OUTPUT_XLSX)
    ws, table_start_row = build_training_log._get_or_create_subject_sheet(wb, subject)
    col_map = build_training_log._scan_existing_column_labels(ws, table_start_row)

    _write_header_fields(ws, header_fields)

    row = find_row_for_session(ws, table_start_row, col_map, session_started)
    found = row is not None
    if found:
        if weight_g is not None:
            ws.cell(row=row, column=col_map['weight_g'], value=weight_g)
        if weight_after_g is not None:
            ws.cell(row=row, column=col_map['weight_after_task_g'], value=weight_after_g)
        ws.cell(row=row, column=col_map['hand_watered'], value=hand_watered)
        if notes:
            ws.cell(row=row, column=col_map['notes'], value=notes)

    wb.save(_OUTPUT_XLSX)
    return found


def load_header_fields(subject):
    if not os.path.exists(_OUTPUT_XLSX):
        return ['', '', '', '']
    wb = openpyxl.load_workbook(_OUTPUT_XLSX)
    sheet_name = build_training_log._sheet_name_for(subject)
    if sheet_name not in wb.sheetnames:
        return ['', '', '', '']
    ws = wb[sheet_name]
    return [ws.cell(row=2 + i, column=2).value or '' for i in range(4)]


# --- camera snippet: reads CameraRecorder's own live-frame JPEG export, never the growing video
# file itself (confirmed cv2.VideoWriter writes nothing to disk at all until release() -- see
# camera_recorder.py's own "Live-frame JPEG export" docstring section) and never opens the camera
# device itself (that's exclusively owned by the running task's own CameraRecorder) ---------------

def grab_latest_frame(session_dir):
    """ Reads <video_stem>_latest.jpg -- a plain, complete, always-fully-written image file
    (overwritten in place by the running task's own CameraRecorder, ~5Hz) -- so this is a normal,
    reliable file read, not a video-container race. Returns None if the session hasn't started
    writing camera frames yet (no camera on this task, or too early in the session). """
    jpg_path = os.path.join(session_dir, 'session_video_latest.jpg')
    if not os.path.exists(jpg_path):
        return None
    return cv2.imread(jpg_path)


# --- newest-session discovery (same glob scan_all_sessions() already uses) ---------------------

def find_newest_session_csv():
    pattern = os.path.join(_PROJECT_DIR, 'experiments', '*', 'setups', '*', 'sessions', '*', '*.csv')
    paths = glob.glob(pattern)
    if not paths:
        return None
    return max(paths, key=os.path.getmtime)


# --- New Subject / New Setup dialogs, mirroring what the PyBpod GUI's own equivalent actions
# would ask for -------------------------------------------------------------------------------

class NewSubjectDialog(QDialog):
    """ DOB/Sex/Strain/Baseline weight are asked HERE, once, at creation time -- not on every
    session's own Weight/Notes save (see SessionRunnerWindow._on_save_weight()'s own docstring
    for why re-asking rarely-if-ever-changing fields every session was the wrong design). All are
    optional -- a mouse can be catalogued now and these filled in later directly in
    training_log.xlsx or by re-running "New Subject..." is not possible (name must be unique), so
    leaving them blank here just means the header cells stay blank until hand-edited. """

    def __init__(self, parent=None):
        super(NewSubjectDialog, self).__init__(parent)
        self.setWindowTitle('New Subject')
        layout = QFormLayout()
        self.name_input = QLineEdit()
        layout.addRow('Subject name:', self.name_input)
        self.dob_input = QLineEdit()
        layout.addRow('DOB (optional):', self.dob_input)
        self.sex_input = QLineEdit()
        layout.addRow('Sex (optional):', self.sex_input)
        self.strain_input = QLineEdit()
        layout.addRow('Strain (optional):', self.strain_input)
        self.baseline_weight_input = QLineEdit()
        layout.addRow('Baseline weight (g, optional):', self.baseline_weight_input)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)
        self.setLayout(layout)

    def subject_name(self):
        return self.name_input.text().strip()

    def header_fields(self):
        return [self.dob_input.text().strip(), self.sex_input.text().strip(),
                self.strain_input.text().strip(), self.baseline_weight_input.text().strip()]


class NewSetupDialog(QDialog):
    def __init__(self, boards, tasks, subjects, parent=None):
        super(NewSetupDialog, self).__init__(parent)
        self.setWindowTitle('New Setup')
        layout = QFormLayout()
        self.name_input = QLineEdit()
        layout.addRow('Name (experiment + setup):', self.name_input)
        self.board_combo = QComboBox()
        for board in boards:
            self.board_combo.addItem(board['name'])
        layout.addRow('Board:', self.board_combo)
        self.task_combo = QComboBox()
        for task_name in tasks:
            self.task_combo.addItem(task_name)
        layout.addRow('Task:', self.task_combo)
        self.subject_list = QListWidget()
        for subject in subjects:
            item = QListWidgetItem(subject['name'])
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.subject_list.addItem(item)
        layout.addRow('Subjects:', self.subject_list)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)
        self.setLayout(layout)

    def setup_name(self):
        return self.name_input.text().strip()

    def board_name(self):
        return self.board_combo.currentText()

    def task_name(self):
        return self.task_combo.currentText()

    def selected_subjects(self):
        return [self.subject_list.item(i).text() for i in range(self.subject_list.count())
                if self.subject_list.item(i).checkState() == Qt.Checked]


class AttachSubjectDialog(QDialog):
    """ Attaches an EXISTING subject to an EXISTING setup -- see attach_subject_to_setup()'s own
    docstring for why this is a genuinely separate action from "New Setup..." (which only ever
    builds a brand-new setup, never adds a subject to one that already exists). """

    def __init__(self, subjects, stage_configs, parent=None):
        super(AttachSubjectDialog, self).__init__(parent)
        self.setWindowTitle('Attach Subject to Setup')
        layout = QFormLayout()
        self.subject_combo = QComboBox()
        for subject in subjects:
            self.subject_combo.addItem(subject['name'])
        layout.addRow('Subject:', self.subject_combo)
        self.setup_combo = QComboBox()
        for cfg in stage_configs:
            label = '{0} ({1})'.format(cfg['experiment'], cfg['task_name'])
            self.setup_combo.addItem(label, cfg)
        layout.addRow('Setup:', self.setup_combo)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)
        self.setLayout(layout)

    def subject_name(self):
        return self.subject_combo.currentText()

    def stage_config(self):
        return self.setup_combo.currentData()


class DetachSubjectDialog(QDialog):
    """ The reverse of AttachSubjectDialog -- see detach_subject_from_setup()'s own docstring.
    The Subject dropdown is scoped to whichever Setup is currently selected (only subjects
    actually attached to it are valid things to detach), refreshed live as the Setup changes. """

    def __init__(self, stage_configs, parent=None):
        super(DetachSubjectDialog, self).__init__(parent)
        self.setWindowTitle('Detach Subject from Setup')
        self._stage_configs = stage_configs
        layout = QFormLayout()
        self.setup_combo = QComboBox()
        for cfg in stage_configs:
            label = '{0} ({1})'.format(cfg['experiment'], cfg['task_name'])
            self.setup_combo.addItem(label, cfg)
        self.setup_combo.currentIndexChanged.connect(self._on_setup_changed)
        layout.addRow('Setup:', self.setup_combo)
        self.subject_combo = QComboBox()
        layout.addRow('Subject:', self.subject_combo)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)
        self.setLayout(layout)
        self._on_setup_changed()

    def _on_setup_changed(self):
        self.subject_combo.clear()
        cfg = self.setup_combo.currentData()
        if cfg is None:
            return
        for subject_name in cfg['subjects']:
            self.subject_combo.addItem(subject_name)

    def subject_name(self):
        return self.subject_combo.currentText()

    def stage_config(self):
        return self.setup_combo.currentData()


class DeleteSetupDialog(QDialog):
    """ Picks an existing Setup to unregister -- see delete_setup()'s own docstring for exactly
    what this does and doesn't remove (the Setup's own JSON only, never its recorded session
    data). """

    def __init__(self, stage_configs, parent=None):
        super(DeleteSetupDialog, self).__init__(parent)
        self.setWindowTitle('Delete Stage')
        layout = QFormLayout()
        layout.addRow(QLabel(
            'Only unregisters the stage from the dropdown above -- its recorded session data\n'
            'stays on disk untouched. This cannot be undone (a new "New Setup..." of the same\n'
            'name gets a fresh identity, not this one back).'))
        self.setup_combo = QComboBox()
        for cfg in stage_configs:
            label = '{0} ({1})'.format(cfg['experiment'], cfg['task_name'])
            self.setup_combo.addItem(label, cfg)
        layout.addRow('Setup:', self.setup_combo)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)
        self.setLayout(layout)

    def stage_config(self):
        return self.setup_combo.currentData()


class DeleteSubjectDialog(QDialog):
    """ Picks an existing Subject to delete -- a real, irreversible shutil.rmtree() (see
    delete_subject()'s own docstring), so this requires typing the subject's own name to confirm,
    not just a combo-box pick -- the same "extra friction for a true directory removal" principle
    CLAUDE.md's own git-safety guidance already applies to `rm -rf`. """

    def __init__(self, subjects, parent=None):
        super(DeleteSubjectDialog, self).__init__(parent)
        self.setWindowTitle('Delete Subject')
        layout = QFormLayout()
        layout.addRow(QLabel(
            'Permanently deletes this subject\'s own identity/cross-session training state.\n'
            'Already-recorded session data is untouched (it lives under experiments/, keyed by\n'
            'Setup, not by subject). This cannot be undone.'))
        self.subject_combo = QComboBox()
        for subject in subjects:
            self.subject_combo.addItem(subject['name'])
        self.subject_combo.currentIndexChanged.connect(self._on_subject_changed)
        layout.addRow('Subject:', self.subject_combo)
        self.confirm_input = QLineEdit()
        layout.addRow('Type name to confirm:', self.confirm_input)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)
        self.setLayout(layout)

    def _on_subject_changed(self):
        self.confirm_input.clear()

    def subject_name(self):
        return self.subject_combo.currentText()

    def confirmed(self):
        return (self.confirm_input.text().strip() == self.subject_name()
                and bool(self.subject_name()))


# --- main window ---------------------------------------------------------------------------------

class SessionRunnerWindow(QWidget):
    def __init__(self):
        super(SessionRunnerWindow, self).__init__()
        self.setWindowTitle('Run Session -- AuditoryEvidenceAccum')

        self.launcher = TaskLauncher()
        self._stage_configs = discover_stage_configs()
        self._users = discover_users()
        self._watched_csv_path = None
        self._feeder = None
        self._stop_deadline = None
        self._flush_valve_proc = None
        self._rig_check_proc = None

        self._build_ui()
        self._populate_stage_combo()
        self._run_task_python_check()

        self._session_timer = QTimer(self)
        self._session_timer.timeout.connect(self._on_session_poll)
        self._session_timer.start(VAR_SESSION_POLL_MS)

        self._plots_timer = QTimer(self)
        self._plots_timer.timeout.connect(self._on_plots_poll)
        self._plots_timer.start(VAR_PLOTS_POLL_MS)

        self._camera_timer = QTimer(self)
        self._camera_timer.timeout.connect(self._on_camera_poll)
        self._camera_timer.start(VAR_CAMERA_POLL_MS)

        self._stop_timer = QTimer(self)
        self._stop_timer.timeout.connect(self._on_stop_check)

    # --- UI ------------------------------------------------------------------------------------

    def _build_ui(self):
        outer = QVBoxLayout()

        # 1/3 of the actual screen's own width, queried live (same "don't hardcode a guess, ask
        # the real screen" convention wheel_shaping_plots.py's own _capped_figsize() already uses)
        # rather than a fixed pixel guess that would look right on one monitor and cramped/
        # oversized on another. Drives both the sidebar's own width and the camera preview's, so
        # the preview actually uses the space rather than sitting small in a now-wide sidebar.
        screen_width_px = QApplication.desktop().screenGeometry().width()
        sidebar_width_px = screen_width_px // 3
        self._camera_preview_width_px = (sidebar_width_px - 60) // 2

        self.task_python_warning_label = QLabel('')
        self.task_python_warning_label.setWordWrap(True)
        self.task_python_warning_label.setStyleSheet(
            'background-color: #ffcccc; color: #660000; padding: 6px; font-weight: bold;')
        self.task_python_warning_label.hide()
        outer.addWidget(self.task_python_warning_label)

        # Slim top strip -- reference info glanced at occasionally, not acted on repeatedly, so it
        # doesn't need a full group box competing for space with the sidebar below.
        top_widget = QWidget()
        top_layout = QVBoxLayout()
        top_layout.setContentsMargins(6, 4, 6, 4)
        top_layout.setSpacing(2)
        data_row = QHBoxLayout()
        data_row.addWidget(QLabel('<b>Session data:</b>'))
        self.session_data_label = QLabel('(none watched -- Start Task or Watch Latest Session)')
        self.session_data_label.setWordWrap(True)
        data_row.addWidget(self.session_data_label, stretch=1)
        top_layout.addLayout(data_row)
        log_row = QHBoxLayout()
        log_row.addWidget(QLabel('<b>Training log:</b>'))
        self.training_log_label = QLabel(_OUTPUT_XLSX)
        self.training_log_label.setWordWrap(True)
        log_row.addWidget(self.training_log_label, stretch=1)
        self.flush_valve_button = QPushButton('Flush Valve')
        self.flush_valve_button.setToolTip(
            'Launches Calibration/flush_valve.py in its own window (Open/Close Valve buttons, '
            'holds the valve open indefinitely for cleaning/priming, independent of calibration) '
            '-- connects to the Bpod board directly, same as a launched task, so don\'t run this '
            'while a task is running here or on another box sharing this rig.')
        self.flush_valve_button.clicked.connect(self._on_flush_valve)
        log_row.addWidget(self.flush_valve_button)
        self.rig_check_button = QPushButton('Rig Check')
        self.rig_check_button.setToolTip(
            'Launches Calibration/rig_check.py in its own window for a quick hardware sanity '
            'check -- connects to the Bpod board directly, same as a launched task, so don\'t run '
            'this while a task is running here or on another box sharing this rig.')
        self.rig_check_button.clicked.connect(self._on_rig_check)
        log_row.addWidget(self.rig_check_button)
        self.update_log_button = QPushButton('Update Training Log')
        self.update_log_button.clicked.connect(self._on_update_training_log)
        log_row.addWidget(self.update_log_button)
        top_layout.addLayout(log_row)
        self.update_log_status_label = QLabel('')
        self.update_log_status_label.setWordWrap(True)
        top_layout.addWidget(self.update_log_status_label)
        top_widget.setLayout(top_layout)
        outer.addWidget(top_widget)

        main_row = QHBoxLayout()

        # --- left sidebar: fixed, comfortable width -- a form-heavy control column doesn't need
        # to grow with the window, and looks messier if it does. Scrollable so nothing clips
        # regardless of screen height. Ordered top-to-bottom to match how these are actually used:
        # launch a task first, monitor it while it runs, fill in weight/notes last. -------------
        left_content = QWidget()
        left_layout = QVBoxLayout()

        launch_group = QGroupBox('Launch task')
        launch_form = QFormLayout()
        stage_row = QHBoxLayout()
        self.stage_combo = QComboBox()
        self.stage_combo.currentIndexChanged.connect(self._on_stage_changed)
        stage_row.addWidget(self.stage_combo, stretch=1)
        self.new_setup_button = QPushButton('New Setup...')
        self.new_setup_button.clicked.connect(self._on_new_setup)
        stage_row.addWidget(self.new_setup_button)
        self.delete_setup_button = QPushButton('Delete Stage...')
        self.delete_setup_button.setToolTip(
            'Unregisters a stage from the dropdown above -- its recorded session data stays on '
            'disk untouched. See the confirmation dialog for exactly what this does and doesn\'t '
            'remove.')
        self.delete_setup_button.clicked.connect(self._on_delete_setup)
        stage_row.addWidget(self.delete_setup_button)
        launch_form.addRow('Stage:', stage_row)
        subject_row = QHBoxLayout()
        self.subject_combo = QComboBox()
        self.subject_combo.currentIndexChanged.connect(self._on_subject_changed)
        subject_row.addWidget(self.subject_combo, stretch=1)
        self.new_subject_button = QPushButton('New Subject...')
        self.new_subject_button.clicked.connect(self._on_new_subject)
        subject_row.addWidget(self.new_subject_button)
        self.delete_subject_button = QPushButton('Delete Subject...')
        self.delete_subject_button.setToolTip(
            'Permanently deletes a subject\'s own identity/cross-session training state (auto-'
            'detaching it from any stage first). Already-recorded session data is untouched.')
        self.delete_subject_button.clicked.connect(self._on_delete_subject)
        subject_row.addWidget(self.delete_subject_button)
        launch_form.addRow('Subject:', subject_row)
        attach_detach_row = QHBoxLayout()
        self.attach_subject_button = QPushButton('Attach Subject to Setup...')
        self.attach_subject_button.setToolTip(
            'Attach an already-existing subject (e.g. one created via "New Subject...") to an '
            'already-existing setup -- lets one real animal run every stage under the same '
            'subject identity instead of being stuck unattached/unselectable.')
        self.attach_subject_button.clicked.connect(self._on_attach_subject)
        attach_detach_row.addWidget(self.attach_subject_button)
        self.detach_subject_button = QPushButton('Detach Subject from Setup...')
        self.detach_subject_button.setToolTip(
            'Remove a subject from a setup\'s own roster -- e.g. to retire a badly-named '
            'placeholder subject (stage info baked into the name) once the right one has been '
            'attached instead. Only changes which subjects are offered for NEW sessions; never '
            'touches any already-recorded session\'s own data.')
        self.detach_subject_button.clicked.connect(self._on_detach_subject)
        attach_detach_row.addWidget(self.detach_subject_button)
        launch_form.addRow(attach_detach_row)
        self.user_combo = QComboBox()
        for user in self._users:
            self.user_combo.addItem(user['name'], user)
        launch_form.addRow('User:', self.user_combo)
        self.reward_ul_input = QDoubleSpinBox()
        self.reward_ul_input.setRange(0.1, 50.0)
        self.reward_ul_input.setSingleStep(0.5)
        self.reward_ul_input.setDecimals(1)
        self.reward_ul_input.setValue(VAR_DEFAULT_REWARD_UL)
        self.reward_ul_input.setToolTip(
            'Target reward volume for this launch -- overrides the task script\'s own hardcoded '
            'default via RUN_SESSION_REWARD_UL. Valve open time is still derived from this via '
            'Calibration/liquid_calibration.py\'s own fitted curve, same as always.')
        launch_form.addRow('Reward (uL):', self.reward_ul_input)
        buttons_row = QHBoxLayout()
        self.start_button = QPushButton('Start Task')
        self.start_button.clicked.connect(self._on_start_task)
        buttons_row.addWidget(self.start_button)
        self.stop_button = QPushButton('Stop Task')
        self.stop_button.clicked.connect(self._on_stop_task)
        self.stop_button.setEnabled(False)
        buttons_row.addWidget(self.stop_button)
        launch_form.addRow(buttons_row)
        self.watch_latest_button = QPushButton('Watch Latest Session')
        self.watch_latest_button.setToolTip(
            'Attach to whichever session is currently newest on disk -- e.g. one started from the '
            'real PyBpod GUI instead of this dashboard. The window opens blank; nothing is '
            'auto-watched.')
        self.watch_latest_button.clicked.connect(self._on_watch_latest)
        launch_form.addRow(self.watch_latest_button)
        self.launch_status_label = QLabel('Not running.')
        self.launch_status_label.setWordWrap(True)
        launch_form.addRow(self.launch_status_label)
        launch_group.setLayout(launch_form)
        left_layout.addWidget(launch_group)

        self.summary_label = QLabel('No session watched yet.')
        self.summary_label.setWordWrap(True)
        left_layout.addWidget(self.summary_label)

        camera_group = QGroupBox('Camera')
        camera_layout = QVBoxLayout()
        self.camera_label = QLabel('No session watched yet.')
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumHeight(self._camera_preview_width_px * 3 // 4)
        self.camera_label.setStyleSheet('background-color: #222; color: #aaa;')
        camera_layout.addWidget(self.camera_label)
        camera_group.setLayout(camera_layout)
        left_layout.addWidget(camera_group)

        wheel_group = QGroupBox('Wheel position (session time; dotted = threshold, '
                                 'light vertical = trial start)')
        wheel_layout = QVBoxLayout()
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
        self._wheel_fig = Figure(figsize=(6, 3.6), constrained_layout=True)
        self._wheel_ax = self._wheel_fig.add_subplot(111)
        self._wheel_ax.set_xlabel('session time (s)', fontsize=8)
        self._wheel_ax.set_ylabel('wheel pos (deg)', fontsize=8)
        self._wheel_ax.tick_params(labelsize=7)
        self._wheel_canvas = FigureCanvasQTAgg(self._wheel_fig)
        self._wheel_canvas.setMinimumHeight(260)
        self._wheel_canvas.setMaximumHeight(340)
        wheel_layout.addWidget(self._wheel_canvas)
        wheel_group.setLayout(wheel_layout)
        left_layout.addWidget(wheel_group)

        weight_group = QGroupBox('Weight / notes (writes to training_log.xlsx)')
        weight_form = QFormLayout()
        # DOB/Sex/Strain are deliberately NOT here -- asked once, at subject-creation time (see
        # NewSubjectDialog), not re-prompted every session. Baseline weight stays here since it
        # can occasionally need updating, but it's pre-filled with whatever's already on record
        # (see _prefill_baseline_weight()) -- leaving it untouched round-trips as a no-op, only
        # typing a new value actually changes it.
        self.baseline_weight_input = QLineEdit()
        weight_form.addRow('Baseline weight (g):', self.baseline_weight_input)
        self.weight_before_input = QDoubleSpinBox()
        self.weight_before_input.setRange(0, 100)
        self.weight_before_input.setDecimals(1)
        weight_form.addRow('Weight before (g):', self.weight_before_input)
        self.weight_after_input = QDoubleSpinBox()
        self.weight_after_input.setRange(0, 100)
        self.weight_after_input.setDecimals(1)
        weight_form.addRow('Weight after (g):', self.weight_after_input)
        self.hand_watered_check = QCheckBox('Water given by hand after task')
        weight_form.addRow(self.hand_watered_check)
        self.notes_input = QTextEdit()
        self.notes_input.setMaximumHeight(80)
        weight_form.addRow('Notes:', self.notes_input)
        self.save_weight_button = QPushButton('Save to Training Log')
        self.save_weight_button.clicked.connect(self._on_save_weight)
        weight_form.addRow(self.save_weight_button)
        self.save_status_label = QLabel('')
        weight_form.addRow(self.save_status_label)
        weight_group.setLayout(weight_form)
        left_layout.addWidget(weight_group)

        left_layout.addStretch(1)
        left_content.setLayout(left_layout)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setWidget(left_content)
        left_scroll.setFixedWidth(sidebar_width_px)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        main_row.addWidget(left_scroll, stretch=0)

        # --- main area (right, fills all remaining space): the full WheelShapingPlots outcome-
        # panel set, embedded directly (see PlotFeeder's own embed=True) instead of letting it
        # open as a separate popup window. Taller than most screens (esp. Stage 3/4's 6-row
        # mosaic), so it sits inside a scroll area. A fixed-width sidebar + an expanding main area
        # gives this the dominant share of the window on any real screen, without the awkward
        # competition a 50/50 split creates against a form column that doesn't need that much
        # width. -----------------------------------------------------------------------------
        self.plots_scroll = QScrollArea()
        self.plots_scroll.setWidgetResizable(True)
        self.plots_container = QWidget()
        self.plots_container_layout = QVBoxLayout()
        self.plots_placeholder_label = QLabel(
            'No session watched yet.\n\nClick "Start Task" to launch and watch a new session, or '
            '"Watch Latest Session" to attach to one already running (e.g. started from the real '
            'PyBpod GUI).\n\nOnce trials start arriving, the full outcome-panel set (raster, '
            'accuracy, sidebias, psychometric, lick panels, ...) will appear here.')
        self.plots_placeholder_label.setAlignment(Qt.AlignCenter)
        self.plots_placeholder_label.setWordWrap(True)
        self.plots_placeholder_label.setStyleSheet('color: #666; font-style: italic; padding: 40px;')
        self.plots_container_layout.addWidget(self.plots_placeholder_label)
        self.plots_container.setLayout(self.plots_container_layout)
        self.plots_scroll.setWidget(self.plots_container)
        main_row.addWidget(self.plots_scroll, stretch=1)

        outer.addLayout(main_row, stretch=1)
        self.setLayout(outer)

    def _populate_stage_combo(self):
        self.stage_combo.clear()
        for cfg in self._stage_configs:
            label = '{0} ({1})'.format(cfg['experiment'], cfg['task_name'])
            self.stage_combo.addItem(label, cfg)
        self._on_stage_changed()

    def _on_stage_changed(self):
        self.subject_combo.clear()
        cfg = self.stage_combo.currentData()
        if cfg is None:
            return
        for subject in cfg['subjects']:
            self.subject_combo.addItem(subject)

    def _on_subject_changed(self):
        subject = self.subject_combo.currentText()
        if subject:
            self._prefill_baseline_weight(subject)

    def _prefill_baseline_weight(self, subject):
        """ Pre-fills Baseline weight from whatever's already on record for this subject (DOB/
        Sex/Strain are no longer part of this form at all -- see weight_group's own comment) --
        called on a genuine subject CHANGE only (the Subject combo, or a newly-watched session
        revealing its own subject), never on every ~2s plots-poll tick, so it can't clobber
        whatever the user might currently be typing into the field. """
        self.baseline_weight_input.setText(load_header_fields(subject)[3])

    # --- task-python self-check / training log / new subject+setup -----------------------------

    def _run_task_python_check(self):
        if task_python_can_import_pybpodapi():
            self.task_python_warning_label.hide()
            return
        self.task_python_warning_label.setText(
            'WARNING: task launches will fail -- "pybpodapi" is not importable via "{0}". Edit '
            'VAR_TASK_PYTHON_EXE at the top of run_session.py to point at your pybpod-environment\'s'
            ' own python.exe.'.format(VAR_TASK_PYTHON_EXE))
        self.task_python_warning_label.show()

    def _confirm_bpod_contention(self, tool_label):
        """ Both flush_valve.py and rig_check.py connect to the Bpod board directly (their own
        separate process, separate from this dashboard's own launched-task subprocess) -- running
        either one concurrently with an actively-running task is genuine unsynchronized access to
        the same physical COM port (see CLAUDE.md's documented Bpod-hardware-module hazards).
        Returns True if it's safe to proceed (no task running here), or if the user explicitly
        confirms anyway. """
        if not self.launcher.is_running:
            return True
        return QMessageBox.question(
            self, 'Task is running',
            'A task launched from this dashboard is currently running, using the same Bpod COM '
            'port {0} would need. Launch anyway?'.format(tool_label),
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No) == QMessageBox.Yes

    def _on_flush_valve(self):
        if self._flush_valve_proc is not None and self._flush_valve_proc.poll() is None:
            QMessageBox.information(self, 'Already running',
                                     'Flush Valve is already open in its own console window.')
            return
        if not os.path.exists(VAR_FLUSH_VALVE_SCRIPT):
            QMessageBox.critical(self, 'Not found',
                                  'Could not find {0}.'.format(VAR_FLUSH_VALVE_SCRIPT))
            return
        if not self._confirm_bpod_contention('Flush Valve'):
            return
        try:
            # flush_valve.py builds its own QApplication/window (Open/Close Valve buttons) --
            # launched as a fully separate process (not imported in-process) so its own Qt event
            # loop and Bpod connection never collide with this dashboard's own, same convention as
            # _on_rig_check() below.
            self._flush_valve_proc = subprocess.Popen(
                [VAR_TASK_PYTHON_EXE, VAR_FLUSH_VALVE_SCRIPT])
        except Exception as err:
            QMessageBox.critical(self, 'Launch failed', str(err))

    def _on_rig_check(self):
        if self._rig_check_proc is not None and self._rig_check_proc.poll() is None:
            QMessageBox.information(self, 'Already running',
                                     'Rig Check is already open in its own window.')
            return
        if not os.path.exists(VAR_RIG_CHECK_SCRIPT):
            QMessageBox.critical(self, 'Not found',
                                  'Could not find {0}.'.format(VAR_RIG_CHECK_SCRIPT))
            return
        if not self._confirm_bpod_contention('Rig Check'):
            return
        try:
            # rig_check.py builds its own QApplication/window -- launched as a fully separate
            # process (not imported in-process) so its own Qt event loop and Bpod connection never
            # collide with this dashboard's own.
            self._rig_check_proc = subprocess.Popen([VAR_TASK_PYTHON_EXE, VAR_RIG_CHECK_SCRIPT])
        except Exception as err:
            QMessageBox.critical(self, 'Launch failed', str(err))

    def _on_update_training_log(self):
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                build_training_log.build_workbook()
        except Exception as err:
            print(traceback.format_exc(), flush=True)
            QMessageBox.critical(self, 'Update failed', str(err))
            return
        output = buf.getvalue().strip()
        if output:
            print(output, flush=True)
        last_line = output.split('\n')[-1] if output else 'Done.'
        self.update_log_status_label.setText(last_line)

    def _on_new_subject(self):
        dialog = NewSubjectDialog(self)
        if dialog.exec_() != QDialog.Accepted:
            return
        name = dialog.subject_name()
        try:
            create_subject(name)
            save_subject_header_fields(name, dialog.header_fields())
        except Exception as err:
            QMessageBox.critical(self, 'Could not create subject', str(err))
            return
        QMessageBox.information(
            self, 'Subject created',
            'Created subject "{0}" -- not attached to any Setup yet, so it won\'t appear in the '
            'Subject dropdown until you attach it. Use "Attach Subject to Setup..." to add it to '
            'an existing one (e.g. Stage1), or "New Setup..." to build a brand-new one with it '
            'already included.'.format(name))

    def _on_new_setup(self):
        boards = discover_boards()
        if not boards:
            QMessageBox.warning(self, 'No boards',
                                 'No boards found under boards/ -- create one via the PyBpod GUI '
                                 'first.')
            return
        dialog = NewSetupDialog(boards, discover_all_tasks(), discover_subjects(), self)
        if dialog.exec_() != QDialog.Accepted:
            return
        name = dialog.setup_name()
        try:
            create_setup(name, dialog.board_name(), dialog.task_name(), dialog.selected_subjects())
        except Exception as err:
            QMessageBox.critical(self, 'Could not create setup', str(err))
            return
        self._stage_configs = discover_stage_configs()
        self._populate_stage_combo()
        QMessageBox.information(self, 'Setup created',
                                 'Created experiment/setup "{0}".'.format(name))

    def _on_attach_subject(self):
        subjects = discover_subjects()
        if not subjects:
            QMessageBox.warning(self, 'No subjects',
                                 'No subjects found under subjects/ -- create one via '
                                 '"New Subject..." first.')
            return
        if not self._stage_configs:
            QMessageBox.warning(self, 'No setups',
                                 'No setups found under experiments/ -- create one via '
                                 '"New Setup..." first.')
            return
        dialog = AttachSubjectDialog(subjects, self._stage_configs, self)
        if dialog.exec_() != QDialog.Accepted:
            return
        subject_name = dialog.subject_name()
        cfg = dialog.stage_config()
        try:
            attach_subject_to_setup(cfg['experiment'], cfg['setup'], subject_name)
        except Exception as err:
            QMessageBox.critical(self, 'Could not attach subject', str(err))
            return
        self._stage_configs = discover_stage_configs()
        self._populate_stage_combo()
        QMessageBox.information(
            self, 'Subject attached',
            'Attached "{0}" to {1} ({2}) -- it will now appear in the Subject dropdown for that '
            'stage.'.format(subject_name, cfg['experiment'], cfg['task_name']))

    def _on_detach_subject(self):
        if not self._stage_configs:
            QMessageBox.warning(self, 'No setups',
                                 'No setups found under experiments/ -- nothing to detach from.')
            return
        if not any(cfg['subjects'] for cfg in self._stage_configs):
            QMessageBox.information(self, 'Nothing to detach',
                                     'No setup currently has any subject attached.')
            return
        dialog = DetachSubjectDialog(self._stage_configs, self)
        if dialog.exec_() != QDialog.Accepted:
            return
        subject_name = dialog.subject_name()
        cfg = dialog.stage_config()
        if not subject_name:
            return
        try:
            detach_subject_from_setup(cfg['experiment'], cfg['setup'], subject_name)
        except Exception as err:
            QMessageBox.critical(self, 'Could not detach subject', str(err))
            return
        self._stage_configs = discover_stage_configs()
        self._populate_stage_combo()
        QMessageBox.information(
            self, 'Subject detached',
            'Detached "{0}" from {1} ({2}).'.format(subject_name, cfg['experiment'],
                                                       cfg['task_name']))

    def _on_delete_setup(self):
        if not self._stage_configs:
            QMessageBox.warning(self, 'No setups',
                                 'No setups found under experiments/ -- nothing to delete.')
            return
        dialog = DeleteSetupDialog(self._stage_configs, self)
        if dialog.exec_() != QDialog.Accepted:
            return
        cfg = dialog.stage_config()
        if cfg is None:
            return
        if QMessageBox.question(
                self, 'Delete stage?',
                'Unregister {0} ({1}) from the Stage dropdown? Its recorded session data stays '
                'on disk untouched. This cannot be undone.'.format(
                    cfg['experiment'], cfg['task_name']),
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        try:
            delete_setup(cfg['experiment'], cfg['setup'])
        except Exception as err:
            QMessageBox.critical(self, 'Could not delete setup', str(err))
            return
        self._stage_configs = discover_stage_configs()
        self._populate_stage_combo()
        QMessageBox.information(self, 'Setup deleted',
                                 'Unregistered {0} ({1}).'.format(cfg['experiment'],
                                                                   cfg['task_name']))

    def _on_delete_subject(self):
        subjects = discover_subjects()
        if not subjects:
            QMessageBox.warning(self, 'No subjects',
                                 'No subjects found under subjects/ -- nothing to delete.')
            return
        dialog = DeleteSubjectDialog(subjects, self)
        if dialog.exec_() != QDialog.Accepted:
            return
        subject_name = dialog.subject_name()
        if not subject_name:
            return
        if not dialog.confirmed():
            QMessageBox.warning(self, 'Not confirmed',
                                 'Typed name didn\'t match "{0}" -- nothing deleted.'.format(
                                     subject_name))
            return
        try:
            delete_subject(subject_name)
        except Exception as err:
            QMessageBox.critical(self, 'Could not delete subject', str(err))
            return
        self._stage_configs = discover_stage_configs()
        self._populate_stage_combo()
        QMessageBox.information(self, 'Subject deleted',
                                 'Deleted subject "{0}".'.format(subject_name))

    # --- launch / stop ---------------------------------------------------------------------------

    def _on_start_task(self):
        cfg = self.stage_combo.currentData()
        subject = self.subject_combo.currentText()
        user = self.user_combo.currentData()
        if cfg is None or not subject or user is None:
            QMessageBox.warning(self, 'Missing selection',
                                 'Choose a stage, subject, and user first.')
            return
        try:
            session_path = self.launcher.start(cfg, subject, user,
                                                reward_ul=self.reward_ul_input.value())
        except Exception as err:
            print(traceback.format_exc(), flush=True)
            QMessageBox.critical(self, 'Launch failed', str(err))
            return
        self.launch_status_label.setText('Running: {0}'.format(session_path))
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self._watch_session(os.path.join(
            session_path, os.path.basename(session_path) + '.csv'))
        QTimer.singleShot(VAR_QUICK_DEATH_CHECK_MS, self._on_check_quick_death)

    def _on_check_quick_death(self):
        """ If the task process already died within VAR_QUICK_DEATH_CHECK_MS of launching, surface
        its captured stderr directly in the UI instead of leaving it silently sitting in a log file
        inside the new session folder -- see VAR_TASK_PYTHON_EXE's own docstring for the failure
        mode this exists to catch. """
        proc = self.launcher.proc
        if proc is None or proc.poll() is None:
            return   # still running (or a newer launch has already replaced this one)
        session_path = self.launcher.session_path
        tail = ''
        if session_path:
            stderr_path = os.path.join(session_path, 'dashboard_launch_stderr.log')
            try:
                with open(stderr_path, 'r', encoding='utf-8', errors='replace') as f:
                    tail = f.read().strip()[-4000:]
            except Exception:
                pass
        QMessageBox.critical(
            self, 'Task exited immediately',
            'The task process exited right after launching (exit code {0}).\n\n{1}'.format(
                proc.returncode, tail or '(no stderr captured -- check {0})'.format(session_path)))
        self._on_task_finished()

    def _on_stop_task(self):
        if not self.launcher.is_running:
            self._on_task_finished()   # already exited on its own: just reset the UI
            return
        self.launcher.stop(graceful=True)
        self.stop_button.setEnabled(False)
        self.launch_status_label.setText('Stopping (waiting up to {0:.0f}s for a clean '
                                          'finish)...'.format(VAR_STOP_GRACE_S))
        self._stop_deadline = time.time() + VAR_STOP_GRACE_S
        self._stop_timer.start(500)

    def _on_stop_check(self):
        if not self.launcher.is_running:
            self._stop_timer.stop()
            self._on_task_finished()
            return
        if time.time() >= self._stop_deadline:
            self.launcher.force_kill()

    def _on_task_finished(self):
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.launch_status_label.setText('Finished.')
        self._audit_finished_session(self.launcher.session_path)

    def _audit_finished_session(self, session_path):
        """ Runs records/audit_session.py on the session that just ended (once per session):
        writes its report to <session>/audit_report.txt and shows any FAIL lines -- so a timing or
        logging regression (dot before the trial, reward before the dot, logging gaps, stalls) is
        visible right after the run instead of only on a later hand analysis. Never blocks or
        raises: an audit problem must not get in the way of the session itself. """
        if not session_path or getattr(self, '_audited_path', None) == session_path:
            return
        self._audited_path = session_path
        try:
            import audit_session
            name, results = audit_session.audit(session_path)
        except Exception as err:
            self.launch_status_label.setText('Finished (audit could not run: {0}).'.format(err))
            return
        lines = ['[{0:4}] {1}  --  {2}'.format(st, check, detail) for check, st, detail in results]
        try:
            with open(os.path.join(session_path, 'audit_report.txt'), 'w', encoding='utf-8') as f:
                f.write('AUDIT {0}\n'.format(name) + '\n'.join(lines) + '\n')
        except Exception:
            pass
        fails = [l for l in lines if l.startswith('[FAIL]')]
        self.launch_status_label.setText('Finished -- timing audit: {0}.'.format(
            'all checks pass' if not fails else '{0} FAIL (see audit_report.txt)'.format(len(fails))))
        if fails:
            QMessageBox.warning(self, 'Session timing audit',
                                'This session failed {0} timing/logging check(s):\n\n{1}'.format(
                                    len(fails), '\n'.join(fails)))

    # --- session watching --------------------------------------------------------------------

    def _watch_session(self, csv_path):
        self._watched_csv_path = csv_path
        self._parser = session_csv_parser.IncrementalSessionParser(csv_path)
        self._wheel_cache = WheelSeriesCache()
        self._feeder = None
        self.session_data_label.setText(csv_path)
        self.summary_label.setText('Watching -- waiting for trial data...')

        # Reset the camera/wheel/outcome panels immediately -- otherwise the PREVIOUS session's
        # own stale frame/plot would stay visible until the newly watched session's own first
        # frame/trial arrives.
        self.camera_label.setPixmap(QPixmap())
        self.camera_label.setText('No camera frame yet.')
        self._wheel_ax.cla()
        self._wheel_ax.set_xlabel('session time (s)', fontsize=8)
        self._wheel_ax.set_ylabel('wheel pos (deg)', fontsize=8)
        self._wheel_ax.tick_params(labelsize=7)
        self._wheel_canvas.draw_idle()
        self.plots_placeholder_label.setText(
            'Watching this session -- the full outcome-panel set will appear here once it has at '
            'least one trial.')
        self._embed_plots_canvas(self.plots_placeholder_label)

    def _on_session_poll(self):
        # Deliberately does NOT auto-watch whatever session is newest on disk -- the window opens
        # to a blank slate; only Start Task or "Watch Latest Session" ever picks a session to
        # watch. This only tracks whether a task THIS dashboard itself launched has finished.
        # A task that ends on its own (trial/water/time cap, crash) must reset the UI too. The old
        # test here (`is_running and proc.poll() is not None`) could never be true, since
        # is_running itself requires poll() is None -- so the dashboard stayed "running" forever.
        if self.launcher.proc is not None and not self.launcher.is_running                 and self.stop_button.isEnabled():
            self._on_task_finished()

    def _on_watch_latest(self):
        newest = find_newest_session_csv()
        if newest is None:
            QMessageBox.information(self, 'No sessions found',
                                     'No session CSVs found on disk yet.')
            return
        self._watch_session(newest)

    # --- live plots ----------------------------------------------------------------------------

    def _on_plots_poll(self):
        if self._watched_csv_path is None or not os.path.exists(self._watched_csv_path):
            return
        try:
            info, session_vals, trials = self._parser.update()
        except Exception:
            print(traceback.format_exc(), flush=True)
            return
        protocol_name = info.get('PROTOCOL-NAME')
        real = session_csv_parser.real_trials(trials)

        summary = None
        if protocol_name in session_csv_parser.PROTOCOL_CONFIG:
            try:
                summary = build_training_log.summarize_session(
                    self._watched_csv_path, parsed=(info, session_vals, trials))
            except Exception:
                print(traceback.format_exc(), flush=True)
        if summary is not None:
            self.summary_label.setText(
                'Subject: {subject}  |  Protocol: {protocol}  |  Trials: {trial_count}  |  '
                'Rewarded: {reward_count}  |  Licks: {lick_count}'.format(**summary))
            if summary['subject'] != getattr(self, '_current_subject', None):
                # Only on a genuine CHANGE -- this poll fires every ~2s, and re-prefilling on
                # every tick would clobber whatever the user might be actively typing.
                self._prefill_baseline_weight(summary['subject'])
            self._current_subject = summary['subject']
            self._current_session_started = summary['session_started']

        self._redraw_wheel_plot(real, protocol_name, session_vals)

        if protocol_name in _DASHBOARD_PROTOCOL_INFO and real:
            if self._feeder is None or self._feeder.protocol_name != protocol_name:
                self._feeder = PlotFeeder(protocol_name, real[0]['vals'], session_vals)
                self._embed_plots_canvas(self._feeder.plots.canvas)
            try:
                self._feeder.feed_new_trials(real, session_ended=('SESSION-ENDED' in info))
            except Exception:
                print(traceback.format_exc(), flush=True)

    def _redraw_wheel_plot(self, real, protocol_name, session_vals):
        """ Small cla()+rebuild every poll -- cheap at this cadence/session length (same "small
        panel, cheap redraw" convention already used elsewhere for low-cardinality panels; only
        the FULL WheelShapingPlots raster needs persistent artists, since that one redraws every
        single trial rather than every ~2s poll). """
        starts, segments = build_trial_markers(real, protocol_name, session_vals)
        xs, ys = self._wheel_cache.series(real)
        self._wheel_ax.cla()
        for x in starts:
            self._wheel_ax.axvline(x, color='#dddddd', linewidth=0.6, zorder=0)
        for x_start, x_end, threshold in segments:
            self._wheel_ax.plot([x_start, x_end], [threshold, threshold], color='black',
                                 linestyle=':', linewidth=0.9, zorder=1)
            self._wheel_ax.plot([x_start, x_end], [-threshold, -threshold], color='black',
                                 linestyle=':', linewidth=0.9, zorder=1)
        self._wheel_ax.plot(xs, ys, linewidth=0.8, color='tab:blue', zorder=2)
        self._wheel_ax.axhline(0, color='#888888', linewidth=0.5, zorder=0)
        self._wheel_ax.set_xlabel('session time (s)', fontsize=8)
        self._wheel_ax.set_ylabel('wheel pos (deg)', fontsize=8)
        self._wheel_ax.tick_params(labelsize=7)
        if xs or segments:
            all_x = xs + [s[0] for s in segments] + [s[1] for s in segments]
            all_y = ys + [s[2] for s in segments] + [-s[2] for s in segments]
            self._wheel_ax.set_xlim(min(all_x), max(all_x) + 1)
            y_span = max([abs(y) for y in all_y] + [5.0]) * 1.1
            self._wheel_ax.set_ylim(-y_span, y_span)
        self._wheel_canvas.draw_idle()

    def _embed_plots_canvas(self, canvas):
        """ Swaps the currently-shown WheelShapingPlots canvas (or the placeholder label, the
        first time) for a new one -- happens whenever a new PlotFeeder is created, i.e. a
        different session or a different protocol/stage is now being watched. """
        old = self.plots_container_layout.takeAt(0)
        if old is not None:
            widget = old.widget()
            if widget is not None:
                widget.setParent(None)
        self.plots_container_layout.addWidget(canvas)

    # --- camera --------------------------------------------------------------------------------

    def _on_camera_poll(self):
        if self._watched_csv_path is None:
            return
        session_dir = os.path.dirname(self._watched_csv_path)
        try:
            frame = grab_latest_frame(session_dir)
        except Exception:
            print(traceback.format_exc(), flush=True)
            return
        if frame is None:
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        h, w, _ch = rgb.shape
        qimage = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage).scaledToWidth(
            self._camera_preview_width_px, Qt.SmoothTransformation)
        self.camera_label.setPixmap(pixmap)

    # --- weight / notes save -------------------------------------------------------------------

    def _on_save_weight(self):
        subject = self.subject_combo.currentText() or getattr(self, '_current_subject', None)
        session_started = getattr(self, '_current_session_started', None)
        if not subject:
            QMessageBox.warning(self, 'No subject', 'Choose a subject (or watch a session) first.')
            return
        if not session_started:
            QMessageBox.warning(self, 'No session', 'No watched session to attach this entry to '
                                                      'yet -- start or wait for a session first.')
            return
        # Per explicit instruction: bench-test subjects (anything not named like a real training
        # animal, build_training_log.is_real_animal_subject() -- this lab's convention is a 'UM'
        # prefix, e.g. 'UM001') never get a training_log.xlsx sheet and are never committed to
        # GitHub via this button. Checked here, before either the xlsx write or the commit, so a
        # test subject's data never even lands in the shared file to begin with.
        if not build_training_log.is_real_animal_subject(subject):
            self.save_status_label.setText(
                "Not saved -- '{0}' isn't a real-animal subject name (bench-test subjects "
                "aren't added to the shared training log).".format(subject))
            return
        # DOB/Sex/Strain are blank here on purpose -- see weight_group's own comment in
        # _build_ui(); _write_header_fields() never overwrites an existing value with blank, so
        # this never touches those cells. Baseline weight is the only one this form can change.
        header_fields = ['', '', '', self.baseline_weight_input.text().strip()]
        try:
            found = save_manual_fields(
                subject, session_started,
                weight_g=(self.weight_before_input.value() or None),
                weight_after_g=(self.weight_after_input.value() or None),
                hand_watered=self.hand_watered_check.isChecked(),
                notes=self.notes_input.toPlainText().strip(),
                header_fields=header_fields)
        except Exception as err:
            print(traceback.format_exc(), flush=True)
            QMessageBox.critical(self, 'Save failed', str(err))
            return
        if found:
            saved_text = 'Saved.'
        else:
            saved_text = ('Header fields saved -- session row not found yet '
                          '(try again once the session has at least one trial).')
        self.save_status_label.setText(saved_text + ' Committing to main...')
        self._commit_training_log(saved_text)

    def _commit_training_log(self, saved_text):
        """ Commits the just-saved training_log.xlsx and any new session data onto the local
        `main` branch (never pushes -- see sync_training_log.commit_data_locally()). Runs on a
        background thread since git can be slow on some rigs (CLAUDE.md's Stage 3 startup-freeze
        note), polled via QTimer so the label is only ever touched from the GUI thread. The Save
        button stays disabled meanwhile, so a second save can't rewrite the file while git is
        reading it. """
        result = {}

        def work():
            try:
                result['msg'] = sync_training_log.commit_data_locally()
            except Exception as err:
                print(traceback.format_exc(), flush=True)
                result['msg'] = 'Commit to main FAILED: {0}'.format(err)

        thread = threading.Thread(target=work, daemon=True)
        self.save_weight_button.setEnabled(False)
        thread.start()

        def poll():
            if thread.is_alive():
                QTimer.singleShot(500, poll)
                return
            self.save_weight_button.setEnabled(True)
            print(result['msg'], flush=True)
            self.save_status_label.setText(saved_text + ' ' + result['msg'])

        QTimer.singleShot(500, poll)

    # --- shutdown --------------------------------------------------------------------------------

    def closeEvent(self, event):
        if self.launcher.is_running:
            reply = QMessageBox.question(
                self, 'Task still running',
                'A task launched from this dashboard is still running. Stop it before closing?',
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
            if reply == QMessageBox.Cancel:
                event.ignore()
                return
            if reply == QMessageBox.Yes:
                self.launcher.stop(graceful=True)
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    prune_dangling_setup_refs()
    window = SessionRunnerWindow()
    window.showMaximized()
    sys.exit(app.exec_())
