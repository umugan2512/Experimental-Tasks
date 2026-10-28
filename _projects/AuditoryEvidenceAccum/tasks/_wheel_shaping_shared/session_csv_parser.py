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
STREAM_KEYS = ('WHEEL_POS', 'QUIESCENCE_RESET_TIME', 'WHEEL_CLOCK_OFFSET')

# VALs the task registers for a trial BEFORE that trial's own TRIAL row is written (TRIAL_INIT_START
# is registered before the Bpod thread starts, on purpose -- see install_session_lock()). Seen while
# no trial is open (after END-TRIAL, before the next TRIAL), they are held and filed under the next
# trial instead of the previous one. Registered inside an open trial (older sessions), unchanged.
PRE_TRIAL_KEYS = ('TRIAL_INIT_START',)

# --- per-protocol state/event meaning -- extend as new protocols (Stage 3+) are built ---------------

PROTOCOL_CONFIG = {
    'stage1_wheel_shaping': {
        'rewarded_states': ['Reward'],
        'withheld_states': [],
        'no_movement_states': ['NoMovement'],
        # Trial ended before the response window opened (one-machine-per-trial handshake,
        # 2026-10-23): the wheel moved before the dot was confirmed on screen, or the PC never
        # confirmed it.
        'aborted_states': ['MovedBeforeDot', 'DotTimeout'],
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
        'aborted_states': ['MovedBeforeDot', 'DotTimeout'],
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
        'aborted_states': ['WheelAbort', 'MovedBeforeDot', 'DotTimeout'],
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
        'aborted_states': ['WheelAbort', 'MovedBeforeDot', 'DotTimeout'],
        'withheld_states': [],
        'side_from': 'event',
        'left_event': 'RotaryEncoder1_3',
        'right_event': 'RotaryEncoder1_4',
    },
    'stage4_clicks_direction': {
        'rewarded_states': ['Reward'],
        'incorrect_states': ['ErrorConsumption'],
        'no_movement_states': ['NoResponse'],
        'aborted_states': ['WheelAbort', 'MovedBeforeDot', 'DotTimeout'],
        'withheld_states': [],
        'side_from': 'event',
        'left_event': 'RotaryEncoder1_3',
        'right_event': 'RotaryEncoder1_4',
    },
    'stage5_resume_staircases': {
        'rewarded_states': ['Reward'],
        'incorrect_states': ['ErrorConsumption'],
        'no_movement_states': ['NoResponse'],
        'aborted_states': ['WheelAbort', 'MovedBeforeDot', 'DotTimeout'],
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
        'vals': {key: value_str}, 'bpod_start': float, 'bpod_end': float} in trial order
        (bpod_start/bpod_end absent for a trial whose machine never finished). """
    state = _new_parse_state()
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for row in csv.reader(f, delimiter=';'):
            _consume_row(state, row)
    return state['info'], state['session_vals'], state['trials']


def _new_parse_state():
    return {'info': {}, 'session_vals': {}, 'trials': [], 'current': None, 'open': False,
            'pending': []}


def _consume_row(state, row):
    """ Applies one CSV row to a parse state -- the single implementation behind both
    parse_session_csv() and IncrementalSessionParser, so the two can never disagree. """
    if len(row) < 5:
        return
    current = state['current']
    row_type = row[0]
    if row_type == 'INFO':
        state['info'][row[4]] = row[5] if len(row) > 5 else ''
        if row[4] == 'TRIAL-BPOD-TIME' and current is not None:
            # BPOD-INITIAL-TIME/BPOD-FINAL-TIME columns: the trial's start/end on the Bpod state
            # machine's own clock (MATLAB's TrialStartTimestamp/TrialEndTimestamp) -- every
            # STATE/EVENT time in this trial is relative to bpod_start.
            current['bpod_start'] = _parse_float(row[2])
            current['bpod_end'] = _parse_float(row[3])
    elif row_type == 'TRIAL':
        state['current'] = {'states': {}, 'events': {}, 'vals': {}}
        state['trials'].append(state['current'])
        state['open'] = True
        for key, value in state['pending']:
            _add_val(state['current'], key, value)
        state['pending'] = []
    elif row_type == 'END-TRIAL':
        state['open'] = False
    elif row_type == 'STATE' and len(row) > 5:
        if current is not None:
            current['states'][row[4]] = (_parse_float(row[2]), _parse_float(row[3]),
                                         _parse_float(row[5]))
    elif row_type == 'EVENT' and len(row) > 5:
        t = _parse_float(row[2])
        if t is not None and current is not None:
            current['events'].setdefault(row[5], []).append(t)
    elif row_type == 'VAL' and len(row) > 5:
        key, value = row[4], row[5]
        if key in PRE_TRIAL_KEYS and not state['open']:
            state['pending'].append((key, value))
        elif current is not None:
            _add_val(current, key, value)
        elif key in STREAM_KEYS:
            # wheel samples logged before trial 1 belong to the wheel trace (filed under trial 1,
            # where wheel_samples_bpod_time() places them by module time) -- not a session scalar
            state['pending'].append((key, value))
        else:
            state['session_vals'][key] = value


# A Port1In that follows the previous Port1Out by less than this is the lick sensor breaking one
# tongue contact in two (QC of 20261027-102843: 17% of inter-lick intervals sat in a separate
# 25-30 ms mode, Out->In gap median 15 ms, against a 137 ms lick mode). Analysis side only: the raw
# EVENT and LICK_TIMES_* rows always keep every edge as recorded.
LICK_CHATTER_GAP_S = 0.040


def merge_lick_chatter(lick_ins, lick_outs, min_gap_s=LICK_CHATTER_GAP_S):
    """ Port1In times with every In that follows the preceding Port1Out by < min_gap_s dropped --
    one count per tongue contact. Times may be on any clock, as long as ins and outs share it. """
    ins, outs = sorted(lick_ins or []), sorted(lick_outs or [])
    kept, j, last_out = [], 0, None
    for t in ins:
        while j < len(outs) and outs[j] <= t:
            last_out = outs[j]
            j += 1
        if kept and last_out is not None and last_out >= kept[-1] and t - last_out < min_gap_s:
            continue
        kept.append(t)
    return kept


def trial_licks(trial):
    """ This trial's Port1In times (trial-relative), sensor chatter merged. """
    return merge_lick_chatter(trial['events'].get('Port1In', []), trial['events'].get('Port1Out', []))


def _add_val(trial, key, value):
    if key in STREAM_KEYS:
        trial['vals'].setdefault(key, []).append(value)
    else:
        trial['vals'][key] = value


class IncrementalSessionParser(object):
    """ parse_session_csv() for a file that is still being written: each update() reads only the
    bytes appended since the previous call (complete lines only -- a partially written last line
    waits for the next call) and applies them to the same parse state. The live dashboard used to
    re-parse the whole CSV (twice) every poll, a cost that grows with the session: ~114ms per poll
    3 minutes in, projected ~1.2s per 2s poll at 30 minutes. If the file shrinks (rewritten), it
    starts over. """

    def __init__(self, path):
        self.path = path
        self._reset()

    def _reset(self):
        self._pos = 0
        self._carry = b''
        self._state = _new_parse_state()

    def update(self):
        """ Returns (info, session_vals, trials) -- the same objects as parse_session_csv() would
        return for the file as it stands now (lists/dicts are updated in place between calls). """
        try:
            size = os.path.getsize(self.path)
        except OSError:
            return self.result()
        if size < self._pos:
            self._reset()
        if size > self._pos:
            with open(self.path, 'rb') as f:
                f.seek(self._pos)
                data = f.read(size - self._pos)
            self._pos += len(data)
            data = self._carry + data
            cut = data.rfind(b'\n')
            if cut < 0:
                self._carry = data
            else:
                self._carry = data[cut + 1:]
                text = data[:cut + 1].decode('utf-8', errors='replace')
                for row in csv.reader(text.splitlines(), delimiter=';'):
                    _consume_row(self._state, row)
        return self.result()

    def result(self):
        return self._state['info'], self._state['session_vals'], self._state['trials']


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
        # The side is whichever threshold crossing caused the transition INTO the outcome state --
        # not simply the first (or last) crossing in the trial. A crossing during a state that
        # wasn't listening for it is still logged as a raw event and disarms that threshold, so the
        # crossing Bpod actually acted on can be a later one in the OTHER direction (confirmed on
        # real Stage 1 sessions: every trial with a pre-dot crossing was mislabeled by the old
        # first-crossing rule). A trial with no choice outcome (no-movement/abort) has no side.
        outcome_name = rewarded_name or withheld_name or incorrect_name
        if outcome_name:
            side = side_of_outcome(events, states[outcome_name][0], config)

    reward_duration = states[rewarded_name][2] if rewarded_name else None

    consumed = None
    if rewarded_name:
        reward_start = states[rewarded_name][0]
        lick_times = events.get('Port1In', [])
        consumed = any(t >= reward_start for t in lick_times) if not _isnan(reward_start) else False

    return outcome, side, reward_duration, consumed


def side_of_outcome(events, outcome_start, config):
    """ 'L'/'R' for the threshold crossing that entered an outcome state starting at
    `outcome_start` (trial-relative seconds): the left/right event whose timestamp equals that
    state's start (same rule as bpod_trial_helpers.triggering_event(), duplicated here so this
    parser stays importable without pybpodapi). Falls back to the latest crossing at or before the
    state's start if no timestamp matches exactly; None if there is none. """
    if _isnan(outcome_start):
        return None
    left_ev = events.get(config.get('left_event'), [])
    right_ev = events.get(config.get('right_event'), [])
    for t in left_ev:
        if abs(t - outcome_start) < 1e-4:
            return 'L'
    for t in right_ev:
        if abs(t - outcome_start) < 1e-4:
            return 'R'
    prior = [(t, 'L') for t in left_ev if t <= outcome_start] +             [(t, 'R') for t in right_ev if t <= outcome_start]
    return max(prior)[1] if prior else None


# --- clock alignment (MATLAB Bpod's convention: everything on the Bpod state machine's clock) ----

def trial_host_anchor(trial):
    """ The host time (time.time() - log_python_t0) at which THIS trial's Bpod machine was run --
    the anchor every one of its trial-relative STATE/EVENT times must be added to for a host-clock
    (session-elapsed) time. Only ever read from the trial's OWN vals (never session_vals): an
    anchor registered before a trial's TRIAL row lands in the PREVIOUS trial's vals, which is
    exactly why TRIAL_START can't serve as one. None if nothing usable was logged. """
    for key in ('SM_RUN_TIME', 'SM_SEND_TIME'):
        v = _parse_float(trial['vals'].get(key))
        if v is not None:
            return v
    return derived_host_anchor(trial)


def fit_host_to_bpod(trials, anchor_keys=('SM_RUN_TIME', 'SM_SEND_TIME')):
    """ Least-squares linear map from the host clock every VAL row uses (time.time() -
    log_python_t0) to the Bpod state machine's own clock (seconds, the clock
    `bpod_start` -- MATLAB's TrialStartTimestamp -- is on), fitted across every trial that has
    both a host anchor (the first of `anchor_keys` present) and a Bpod start. The host anchor is
    taken just before RUN, so the intercept absorbs the (roughly constant) RUN latency; measured
    on real Stage 1 sessions: 6-16ppm drift, ~4ms residual SD.

    Outlier-resistant: fit, drop anchors whose residual exceeds max(1ms, 4 x MAD of the
    residuals), refit once. A single delayed RUN (e.g. 8ms in session 20261027-095504, from before
    the anchor was taken at the RUN write itself) otherwise dominates both the residual SD and the
    drift estimate of a short session.

    Returns {'slope', 'intercept', 'n', 'n_outliers', 'outliers' ([(index, residual_s)]),
    'residual_sd_s', 'max_abs_residual_s', 'span_s'} with bpod_t = slope * host_t + intercept (the
    residual stats are over the kept points), or None with fewer than 3 usable trials. `outliers`
    indexes refer to the trials list passed in. """
    xs, ys, idx = [], [], []
    for i, trial in enumerate(trials):
        start = trial.get('bpod_start')
        if start is None:
            continue
        host = None
        for key in anchor_keys:
            host = _parse_float(trial['vals'].get(key))
            if host is not None:
                break
        if host is None:
            host = derived_host_anchor(trial)
        if host is None:
            continue
        xs.append(host)
        ys.append(start)
        idx.append(i)
    if len(xs) < 3:
        return None
    line = _theil_sen(xs, ys)   # robust start: one bad anchor can't pull it toward itself
    if line is None:
        return None
    residuals = [y - (line[0] * x + line[1]) for x, y in zip(xs, ys)]
    mad = sorted(abs(r) for r in residuals)[len(residuals) // 2]
    limit = max(0.001, 4.0 * mad)
    keep = [k for k, r in enumerate(residuals) if abs(r) <= limit]
    outliers = [(idx[k], residuals[k]) for k, r in enumerate(residuals) if abs(r) > limit]
    refit = _linear_fit([xs[k] for k in keep], [ys[k] for k in keep]) if len(keep) >= 3 else None
    if refit is not None:
        line = refit
    kept_res = [ys[k] - (line[0] * xs[k] + line[1]) for k in keep]
    return {
        'slope': line[0],
        'intercept': line[1],
        'n': len(keep),
        'n_outliers': len(outliers),
        'outliers': outliers,
        'residual_sd_s': math.sqrt(sum(r * r for r in kept_res) / len(kept_res)),
        'max_abs_residual_s': max(abs(r) for r in kept_res),
        'span_s': max(xs) - min(xs),
    }


def _theil_sen(xs, ys, max_points=200):
    """ (slope, intercept): median of pairwise slopes, median intercept. Robust to up to ~29%
    outlying points. Long sessions are thinned to max_points (evenly spaced) to bound the O(n^2)
    pair count. None if xs has no spread. """
    n = len(xs)
    if n > max_points:
        step = n / float(max_points)
        pick = [int(k * step) for k in range(max_points)]
        xs, ys = [xs[k] for k in pick], [ys[k] for k in pick]
        n = len(xs)
    slopes = sorted((ys[j] - ys[i]) / (xs[j] - xs[i])
                    for i in range(n) for j in range(i + 1, n) if xs[j] != xs[i])
    if not slopes:
        return None
    slope = slopes[len(slopes) // 2]
    intercepts = sorted(y - slope * x for x, y in zip(xs, ys))
    return slope, intercepts[len(intercepts) // 2]


def _linear_fit(xs, ys):
    """ (slope, intercept) least squares, or None if xs has no spread. """
    n = len(xs)
    mx, my = sum(xs) / float(n), sum(ys) / float(n)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return None
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return slope, my - slope * mx


_CROSSING_STATES = ('Reward', 'RewardL', 'RewardR', 'NoRewardL', 'NoRewardR', 'ErrorConsumption')


def derived_host_anchor(trial):
    """ For sessions recorded before SM_RUN_TIME existed: recovers the host anchor a task script
    used from rows it registered as `anchor + <Bpod-relative time>` -- THRESHOLD_CROSSING_TIME
    (anchor + the crossing state's start) or LICK_TIMES_IN (anchor + each Port1In time). None if
    neither is present. """
    crossing = _parse_float(trial['vals'].get('THRESHOLD_CROSSING_TIME'))
    if crossing is not None:
        for name in _CROSSING_STATES:
            state = trial['states'].get(name)
            if state is not None and not _isnan(state[0]):
                return crossing - state[0]
    licks_in = trial['vals'].get('LICK_TIMES_IN')
    port_in = trial['events'].get('Port1In', [])
    if licks_in and port_in:
        first = _parse_float(licks_in.split(',')[0])
        if first is not None:
            return first - port_in[0]
    return None


def host_to_bpod(fit, host_t):
    return None if fit is None or host_t is None else fit['slope'] * host_t + fit['intercept']


def wheel_samples_bpod_time(trial, first_state):
    """ This trial's WHEEL_POS samples (logged during/after it) placed on the Bpod clock exactly,
    via its ROTARY_MARKER row (the '#' marker the trial's first state sent the rotary, timestamped
    on the module's clock) and the WHEEL_CLOCK_OFFSET rows (host minus module seconds) logged with
    the samples: module_t = host_t - offset, and the marker's module_t corresponds to Bpod time
    bpod_start + <first_state's start>. Returns [(bpod_t, pos), ...], or None if the trial lacks
    any of those (polling-mode sessions never have them). Uses the trial's LAST logged offset for
    every sample -- the offset changes by at most a few ms over a trial. """
    marker = trial['vals'].get('ROTARY_MARKER')
    offsets = trial['vals'].get('WHEEL_CLOCK_OFFSET')
    samples = trial['vals'].get('WHEEL_POS')
    start = trial.get('bpod_start')
    state = trial['states'].get(first_state)
    if not (marker and offsets and samples) or start is None or state is None or _isnan(state[0]):
        return None
    offsets = offsets if isinstance(offsets, list) else [offsets]
    marker_module_t = _parse_float(marker.split(',')[0])
    offset = _parse_float(offsets[-1])
    if marker_module_t is None or offset is None:
        return None
    marker_bpod_t = start + state[0]
    out = []
    for raw in samples:
        parts = raw.split(',')
        host_t, pos = _parse_float(parts[0]), _parse_float(parts[1]) if len(parts) > 1 else None
        if host_t is None or pos is None:
            continue
        # A 3rd field is the sample's own module time (rows written since 2026-10-23) -- exact.
        # Older rows only allow module_t = host_t - (the trial's last logged offset).
        module_t = _parse_float(parts[2]) if len(parts) > 2 else host_t - offset
        out.append((marker_bpod_t + (module_t - marker_module_t), pos))
    return out


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


# Filled by compute_remaining_quota() -- what it found, so the stage script can register it
# (QUOTA_REDUCED_BY / QUOTA_CHAIN_SESSIONS) once the Bpod session exists.
LAST_QUOTA_INFO = {}


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
            summary['name'] = name
            summaries.append(summary)

    now = datetime.datetime.now()
    used_trials, used_duration_s = _chain_already_used(summaries, now, merge_gap_s)
    # Which sessions that reduction came from -- the open bout is the trailing run of summaries
    # whose _chain_already_used() total reproduces used_trials.
    chain = []
    if used_trials:
        for i in range(len(summaries)):
            if _chain_already_used(summaries[i:], now, merge_gap_s)[0] == used_trials and                     sum(x['trial_count'] for x in summaries[i:]) == used_trials:
                chain = [x['name'] for x in summaries[i:]]
                break
    LAST_QUOTA_INFO.clear()
    LAST_QUOTA_INFO.update({'used_trials': used_trials, 'used_duration_s': used_duration_s,
                            'chain_sessions': chain})
    if used_trials:
        print("Quota reduced by {0} trials / {1:.0f}s: continuing the interrupted bout from "
              "session(s) {2}".format(used_trials, used_duration_s, ', '.join(chain) or '?'),
              flush=True)

    remaining_max_trials = (max(1, full_max_trials - used_trials)
                             if full_max_trials is not None else None)
    remaining_cap_s = (max(60.0, full_cap_s - used_duration_s)
                        if full_cap_s is not None else None)
    return remaining_max_trials, remaining_cap_s
