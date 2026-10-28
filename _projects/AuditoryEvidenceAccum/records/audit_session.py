# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Timing / continuity audit of one recorded session -- PASS/FAIL per check, so "is anything not
continuously logged, is there a freeze, a lock or a pause" has a one-command answer instead of a
hand analysis. Built from the 2026-10-22/23 investigation (see CLAUDE.md, "Clocks" and the
dot-handshake notes) and run by run_session.py at the end of every dashboard-launched session.

    python audit_session.py <session_folder> [<session_folder> ...]

Checks marked N/A mean the session predates the VAL that check needs -- not a pass.
"""
import csv
import datetime
import math
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'tasks', '_wheel_shaping_shared'))
import session_csv_parser as _scp     # noqa: E402 -- the same robust clock fit the struct export uses

# --- budgets (see the docstring of each check) ------------------------------------------------------
BUDGET = {
    'sm_coverage_min': 0.97,          # fraction of session time inside a running Bpod machine
    'wheel_gap_max_s': 0.10,          # longest WHEEL_POS gap (keepalive writes every 50ms)
    'csv_silence_max_s': 0.50,        # longest stretch with no CSV row at all
    'clock_residual_sd_max_ms': 2.0,  # host->Bpod linear fit
    'offset_jitter_max_ms': 5.0,      # WHEEL_CLOCK_OFFSET p95-p5 after the first 5s
    'camera_gap_max_ms': 150.0,
    'onset_latency_p95_max_ms': 40.0,     # SOFTCODE_DOT_ON arrival -> dot painted (one 60Hz frame
                                          # is 16.7ms; the handshake means this only delays the
                                          # trial's start, it can't let a reward precede the dot)
    'freeze_latency_p95_max_ms': 30.0,    # freeze SoftCode arrival -> dot frozen
    'render_frame_max_ms': 100.0,         # longest single render-loop frame (work, not wait)
    'inter_trial_gap_max_ms': 150.0,      # previous Bpod call returned -> next RUN
    'stream_read_gap_max_ms': 50.0,       # rotary reader-thread loop
}

_RESPONSE_STATES = ('WheelPeriod', 'WheelDotPeriod')
_OUTCOME_STATES = ('Reward', 'RewardL', 'RewardR', 'NoRewardL', 'NoRewardR', 'ErrorConsumption')


def _pc_time(row):
    try:
        return datetime.datetime.strptime(row[1], '%Y-%m-%d %H:%M:%S.%f')
    except (ValueError, IndexError):
        return None


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def parse(folder):
    name = os.path.basename(os.path.normpath(folder))
    path = os.path.join(folder, name + '.csv')
    with open(path, encoding='utf-8', errors='replace') as f:
        rows = [r for r in csv.reader(f, delimiter=';') if len(r) >= 5]
    trials, cur = [], None
    for r in rows:
        kind = r[0]
        if kind == 'TRIAL':
            cur = {'states': {}, 'events': {}, 'vals': {}}
            trials.append(cur)
        elif cur is None:
            continue
        elif kind == 'STATE' and len(r) > 5:
            cur['states'][r[4]] = _f(r[2])
        elif kind == 'EVENT' and len(r) > 5:
            cur['events'].setdefault(r[5], []).append(_f(r[2]))
        elif kind == 'INFO' and len(r) > 5 and r[4] == 'TRIAL-BPOD-TIME':
            cur['bpod'] = (_f(r[2]), _f(r[3]))
        elif kind == 'VAL' and len(r) > 5:
            cur['vals'].setdefault(r[4], []).append(r[5])
    return name, rows, trials


def _anchor(trial):
    """ Host time a Bpod machine was RUN: SM_RUN_TIME (choice/single machines) or SM_SEND_TIME
    (Stage 3-5 cue machines). """
    for key in ('SM_RUN_TIME', 'SM_SEND_TIME'):
        v = _val(trial, key)
        if v is not None:
            return v
    return None


def _val(trial, key):
    v = trial['vals'].get(key)
    return _f(v[-1]) if v else None


def _pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else None


def audit(folder):
    """ Returns (session_name, [(check, status, detail)]) with status in PASS/FAIL/N/A. """
    name, rows, trials = parse(folder)
    out = []

    def add(check, ok, detail):
        out.append((check, 'N/A' if ok is None else ('PASS' if ok else 'FAIL'), detail))

    ran = [t for t in trials if _anchor(t) is not None and 'bpod' in t]

    # 1. the dot is up before the trial can be rewarded, and never frozen before its trial
    onset_known = [t for t in ran if _val(t, 'DOT_ONSET_TIME') is not None]
    early_reward, early_freeze, invisible = [], [], []
    for i, t in enumerate(trials, 1):
        paint, cross = _val(t, 'DOT_ONSET_PAINT_TIME'), _val(t, 'THRESHOLD_CROSSING_TIME')
        onset, freeze = _val(t, 'DOT_ONSET_TIME'), _val(t, 'DOT_FREEZE_TIME')
        run = _val(t, 'SM_RUN_TIME')
        if cross is not None and paint is not None and cross < paint:
            early_reward.append(i)
        if freeze is not None and ((onset is not None and freeze < onset - 0.001)
                                   or (run is not None and freeze <= run + 0.001)):
            early_freeze.append(i)
        vis = _val(t, 'DOT_VISIBLE_S')
        if paint is not None and vis is not None and vis <= 0:
            invisible.append(i)
    add('reward only after the dot is painted', not early_reward,
        'trials {0}'.format(early_reward) if early_reward else '{0} trials ok'.format(len(trials)))
    add('dot never frozen before its trial', not early_freeze,
        'trials {0}'.format(early_freeze) if early_freeze else 'ok')
    add('dot visible on every painted trial', (not invisible) if onset_known else None,
        'trials {0}'.format(invisible) if invisible else ('ok' if onset_known else 'no DOT_VISIBLE_S'))

    # 2. Bpod machine coverage (licks/inputs are only logged inside a running machine)
    spans = sorted((_anchor(t), _anchor(t) + (t['bpod'][1] - t['bpod'][0]))
                   for t in ran if t['bpod'][0] is not None)
    if len(spans) > 1:
        total = spans[-1][1] - spans[0][0]
        covered = sum(b - a for a, b in spans)
        gaps = [c - b for (_, b), (c, _) in zip(spans, spans[1:])]
        add('Bpod machine running >= {0:.0%} of the session'.format(BUDGET['sm_coverage_min']),
            covered / total >= BUDGET['sm_coverage_min'],
            '{0:.1%} covered; between-machine gaps median {1:.0f}ms, max {2:.0f}ms'.format(
                covered / total, statistics.median(gaps) * 1e3, max(gaps) * 1e3))
    else:
        add('Bpod machine coverage', None, 'fewer than 2 trials with SM_RUN_TIME')

    # 3. wheel log continuity + order
    wheel = []
    for r in rows:
        if r[0] == 'VAL' and len(r) > 5 and r[4] == 'WHEEL_POS':
            parts = r[5].split(',')
            wheel.append((_f(parts[0]), _f(parts[1])))
    wheel = [w for w in wheel if w[0] is not None]
    if len(wheel) > 1:
        order_breaks = sum(1 for a, b in zip(wheel, wheel[1:]) if b[0] < a[0] - 0.002)
        ts = sorted(t for t, _ in wheel)
        gaps = [(b - a, a) for a, b in zip(ts, ts[1:])]
        worst = max(gaps)
        n_big = sum(1 for g, _ in gaps if g > BUDGET['wheel_gap_max_s'])
        add('WHEEL_POS has no gap > {0:.0f}ms'.format(BUDGET['wheel_gap_max_s'] * 1e3), n_big == 0,
            '{0} gaps; longest {1:.2f}s at t={2:.1f}s; {3:.0f} rows/s'.format(
                n_big, worst[0], worst[1], len(wheel) / (ts[-1] - ts[0])))
        add('WHEEL_POS rows in time order', order_breaks == 0, '{0} out of order'.format(order_breaks))
    else:
        add('WHEEL_POS continuity', None, 'no WHEEL_POS rows')

    offsets = []
    last_t = None
    for r in rows:
        if r[0] == 'VAL' and len(r) > 5:
            if r[4] == 'WHEEL_POS':
                last_t = _f(r[5].split(',')[0])
            elif r[4] == 'WHEEL_CLOCK_OFFSET' and last_t is not None and last_t > 5.0:
                offsets.append((last_t, _f(r[5])))
    if len(offsets) > 10:
        # Detrended: the module and PC crystals differ by a steady few tens of ppm, so the offset
        # ramps linearly over a session (82 -> 95 ms over 10 min in 20261027-102843, 21.5 ppm).
        # That ramp is harmless -- samples are placed by module time relative to each trial's own
        # marker -- and undetrended it failed every session longer than ~4 min. Jitter around
        # the line is what would misplace samples.
        n = len(offsets)
        mx = sum(t for t, _ in offsets) / n
        my = sum(o for _, o in offsets) / n
        sxx = sum((t - mx) ** 2 for t, _ in offsets)
        slope = sum((t - mx) * (o - my) for t, o in offsets) / sxx if sxx > 0 else 0.0
        resid = [o - (my + slope * (t - mx)) for t, o in offsets]
        jitter = (_pct(resid, .95) - _pct(resid, .05)) * 1e3
        add('wheel module->PC clock map stable (detrended p95-p5 < {0:.0f}ms)'.format(
            BUDGET['offset_jitter_max_ms']), jitter < BUDGET['offset_jitter_max_ms'],
            '{0:.2f}ms around a {1:+.1f}ppm drift'.format(jitter, slope * 1e6))
    else:
        add('wheel clock map stability', None, 'too few WHEEL_CLOCK_OFFSET rows')

    # 4. whole-process logging silence
    # from the first trial on: connecting/handshaking the modules (e.g. the rotary's 2s boot delay)
    # legitimately writes nothing for a while before trial 1
    # ...and up to the last trial's END-TRIAL: teardown after it (camera close, SESSION-ENDED,
    # struct export) is not session time either
    first_trial = next((i for i, r in enumerate(rows) if r[0] == 'TRIAL'), len(rows))
    last_end = max([i for i, r in enumerate(rows) if r[0] == 'END-TRIAL'] or [len(rows) - 1])
    pcs = [p for p in (_pc_time(r) for r in rows[first_trial:last_end + 1]) if p is not None]
    if len(pcs) > 1:
        sil = max(((b - a).total_seconds(), a) for a, b in zip(pcs, pcs[1:]))
        add('no CSV-wide silence > {0:.1f}s'.format(BUDGET['csv_silence_max_s']),
            sil[0] <= BUDGET['csv_silence_max_s'], 'longest {0:.2f}s after {1}'.format(sil[0], sil[1].time()))

    # 5. clock alignment -- same outlier-resistant fit as session_csv_parser.fit_host_to_bpod()
    pairs = [(_anchor(t), t['bpod'][0], i) for i, t in enumerate(trials, 1)
             if t in ran and t['bpod'][0] is not None]
    if len(pairs) > 2:
        def lin(ps):
            mx = statistics.mean(p[0] for p in ps)
            my = statistics.mean(p[1] for p in ps)
            sxx = sum((p[0] - mx) ** 2 for p in ps)
            sl = sum((p[0] - mx) * (p[1] - my) for p in ps) / sxx if sxx else 1.0
            return sl, my - sl * mx
        sl, ic = _scp._theil_sen([p[0] for p in pairs], [p[1] for p in pairs]) or lin(pairs)
        res = [(p[1] - (sl * p[0] + ic), p[2]) for p in pairs]
        mad = sorted(abs(r) for r, _ in res)[len(res) // 2]
        limit = max(0.001, 4.0 * mad)
        outl = [(i, r) for r, i in res if abs(r) > limit]
        kept = [p for p, (r, _) in zip(pairs, res) if abs(r) <= limit]
        if len(kept) > 2:
            sl, ic = lin(kept)
        kres = [p[1] - (sl * p[0] + ic) for p in kept]
        sd = math.sqrt(sum(r * r for r in kres) / len(kres)) * 1e3
        span = pairs[-1][0] - pairs[0][0]
        max_out = max(1, int(0.05 * len(pairs)))
        add('host->Bpod clock fit residual SD < {0:.0f}ms (outliers <= 5%)'.format(
            BUDGET['clock_residual_sd_max_ms']),
            sd < BUDGET['clock_residual_sd_max_ms'] and len(outl) <= max_out,
            '{0:.2f}ms (max {1:.2f}ms) over {2} trials{3}{4}'.format(
                sd, max(map(abs, kres)) * 1e3, len(kept),
                ', drift {0:.1f}ppm'.format((sl - 1) * 1e6) if span >= 120 else
                ' (session < 120s: drift not estimated)',
                '; outliers: ' + ', '.join('trial {0} {1:+.1f}ms'.format(i, r * 1e3) for i, r in outl)
                if outl else ''))

    # 6. camera
    cam_path = os.path.join(folder, 'session_video_timestamps.csv')
    if os.path.exists(cam_path):
        cam = []
        with open(cam_path) as f:
            for r in csv.reader(f):
                v = _f(r[1]) if len(r) > 1 else None
                if v is not None:
                    cam.append(v)
        if len(cam) > 1:
            g = max(b - a for a, b in zip(cam, cam[1:])) * 1e3
            add('camera frame gap < {0:.0f}ms'.format(BUDGET['camera_gap_max_ms']),
                g < BUDGET['camera_gap_max_ms'],
                'max {0:.0f}ms, {1:.1f}fps realized'.format(g, (len(cam) - 1) / (cam[-1] - cam[0])))

    # 7. latency / freeze instrumentation (registered per trial since 2026-10-23)
    def budget_check(key, label, limit, stat, skip_first=False):
        vals = [(i, x) for i, x in enumerate((_val(t, key) for t in trials), 1) if x is not None]
        note = ''
        if skip_first and vals and vals[0][0] == 1:
            vals = vals[1:]
            note = ', trial 1 excluded'
        if not vals:
            add(label, None, 'no {0} rows'.format(key))
            return
        xs = [x for _, x in vals]
        v = _pct(xs, 0.95) if stat == 'p95' else max(xs)
        over = [i for i, x in vals if x >= limit]
        add(label, v < limit, '{0} {1:.1f}ms (median {2:.1f}ms, n={3}{4}){5}'.format(
            stat, v, statistics.median(xs), len(xs), note,
            '; over budget on trial(s) {0}'.format(', '.join(str(i) for i in over[:10]))
            if stat == 'max' and over else ''))
    budget_check('RENDER_ONSET_LATENCY_MS', 'dot painted within {0:.0f}ms of DotOn (p95)'.format(
        BUDGET['onset_latency_p95_max_ms']), BUDGET['onset_latency_p95_max_ms'], 'p95')
    budget_check('DOT_FREEZE_LATENCY_MS', 'dot frozen within {0:.0f}ms of the outcome (p95)'.format(
        BUDGET['freeze_latency_p95_max_ms']), BUDGET['freeze_latency_p95_max_ms'], 'p95')
    budget_check('RENDER_MAX_FRAME_MS', 'no render-loop frame > {0:.0f}ms'.format(
        BUDGET['render_frame_max_ms']), BUDGET['render_frame_max_ms'], 'max')
    budget_check('INTER_TRIAL_GAP_MS', 'between-machine gap < {0:.0f}ms'.format(
        BUDGET['inter_trial_gap_max_ms']), BUDGET['inter_trial_gap_max_ms'], 'max')
    budget_check('CUE_TO_CHOICE_GAP_MS', 'cue->choice machine gap < {0:.0f}ms (Stage 3-5)'.format(
        BUDGET['inter_trial_gap_max_ms']), BUDGET['inter_trial_gap_max_ms'], 'max')
    budget_check('STREAM_MAX_READ_GAP_MS', 'rotary reader never stalls > {0:.0f}ms'.format(
        BUDGET['stream_read_gap_max_ms']), BUDGET['stream_read_gap_max_ms'], 'max',
        skip_first=True)   # trial 1 includes the stream's start-up

    # 8. markers + warnings
    with_resp = [t for t in ran if any(t['states'].get(s) is not None and not math.isnan(t['states'][s])
                                       for s in _RESPONSE_STATES)]
    if with_resp:
        missing = sum(1 for t in with_resp if not t['vals'].get('ROTARY_MARKER'))
        add("rotary '#' marker logged on every trial that reached the response state", missing == 0,
            '{0}/{1} missing'.format(missing, len(with_resp)))
    crash = [r[5] for r in rows if r[0] == 'VAL' and len(r) > 5 and r[4] == 'SESSION_ERROR']
    # pybpod writes a traceback as multi-line stderr rows with too few fields to survive parse() --
    # so count it in the raw file text
    name_ = os.path.basename(os.path.normpath(folder))
    with open(os.path.join(folder, name_ + '.csv'), encoding='utf-8', errors='replace') as f:
        raw_text = f.read()
    tracebacks = raw_text.count('Traceback (most recent call last)')
    add('no crash in the task (SESSION_ERROR / stderr traceback)', not crash and not tracebacks,
        crash[0] if crash else ('{0} traceback(s) on stderr'.format(tracebacks) if tracebacks
                                else 'none'))
    softmap = [r[5] for r in rows if r[0] == 'VAL' and len(r) > 5 and r[4] == 'SOFTCODE_MAP']
    if softmap:
        out.append(('Bpod->PC SoftCode numbers', 'INFO', softmap[0]))
    warns = [';'.join(r)[:140] for r in rows
             if any(k in ';'.join(r) for k in ('WARNING', 'FAILED', 'Traceback', 'missed hardware'))]
    add('no warnings/errors logged', not warns, warns[0] if warns else 'none')
    return name, out


def print_report(folder):
    name, results = audit(folder)
    print('=' * 100)
    print('AUDIT {0}'.format(name))
    for check, status, detail in results:
        print('  [{0:4}] {1:58} {2}'.format(status, check, detail))
    fails = sum(1 for _, s_, _ in results if s_ == 'FAIL')
    print('  -> {0}'.format('ALL PASS' if fails == 0 else '{0} FAIL'.format(fails)))
    return fails


if __name__ == '__main__':
    total = sum(print_report(p) for p in sys.argv[1:])
    sys.exit(1 if total else 0)
