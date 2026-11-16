# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Click-weight criterion for Stages 3 and 4 (2026-11-11 spec A7): is the animal's choice driven by the
clicks, or by its own trial history? Overall accuracy can't tell -- a win-stay/lose-shift animal
scores well above chance with Stage 3's correction trials, and within side blocks it scored 93-98%
without listening. So both stages gate on a logistic regression of the choice:

    logit P(choose R)_t = b0 + b_click * S_t + b_prev * C_{t-1} + b_win * C_{t-1} * Rw_{t-1}
    S_t     = +1 if the clicks were on the right, -1 if left           (current trial)
    C_{t-1} = +1 / -1 for the previous responded trial's choice R / L   (0 before the first one)
    Rw      = 1 if that previous trial was rewarded, else 0

Fit on the session's main, non-correction, non-warm-up responded trials; the history terms come
from ALL responded trials (correction and warm-up ones included).
    criterion met  <=>  b_click > 0  AND  z(b_click) >= 3  AND  b_click > max(|b_prev|, |b_win|)

Plain maximum likelihood gives an animal that never errs an infinite b_click with an infinite
standard error, i.e. z -> 0, so the best possible animal would FAIL. The fit is therefore penalised:
a ridge (a N(0, 1/RIDGE) prior) on the three slope terms and a near-zero one on the intercept, and
z uses the penalised Hessian. Plain numpy IRLS, no new dependency. Pure Python, no Bpod -- see
validate_wheel_shaping.py.
"""
import numpy as np

CHOICE_REGRESSION_RIDGE = 1.0
CHOICE_REGRESSION_INTERCEPT_RIDGE = 0.01   # tiny: keeps an animal that only ever turns one way
                                           # (all-R choices) finite instead of a singular fit
CHOICE_REGRESSION_MIN_TRIALS = 20     # fewer fitted trials -> no estimate (None)
CLICK_WEIGHT_Z_CRITERION = 3.0


def build_design(trials):
    """ trials: the session's trials in order, each a dict with
        'choice'   'L'/'R' if the animal responded, else None (abort, no response)
        'stim'     'L'/'R', the side the clicks were on
        'rewarded' bool
        'fit'      bool, whether this trial is a current trial of the fit (main, non-correction,
                   non-warm-up, responded)
    Returns (X, y) with X columns [1, S, C_prev, C_prev * Rw_prev]. """
    rows, ys = [], []
    prev_c, prev_rw = 0.0, 0.0
    for t in trials:
        if t['choice'] not in ('L', 'R'):
            continue
        if t.get('fit'):
            s = 1.0 if t['stim'] == 'R' else -1.0
            rows.append([1.0, s, prev_c, prev_c * prev_rw])
            ys.append(1.0 if t['choice'] == 'R' else 0.0)
        prev_c = 1.0 if t['choice'] == 'R' else -1.0
        prev_rw = 1.0 if t['rewarded'] else 0.0
    return np.array(rows, dtype=float).reshape(-1, 4), np.array(ys, dtype=float)


def fit_choice_regression(trials, ridge=CHOICE_REGRESSION_RIDGE, min_trials=CHOICE_REGRESSION_MIN_TRIALS):
    """ Returns a dict (b0, b_click, b_prev, b_win, z_click, n, criterion_met), or None if fewer
    than min_trials current trials. """
    X, y = build_design(trials)
    n = len(y)
    if n < min_trials:
        return None
    penalty = np.diag([CHOICE_REGRESSION_INTERCEPT_RIDGE, ridge, ridge, ridge])
    beta = np.zeros(4)
    for _ in range(100):
        p = 1.0 / (1.0 + np.exp(-X.dot(beta)))
        w = p * (1.0 - p)
        hessian = (X * w[:, None]).T.dot(X) + penalty
        grad = X.T.dot(y - p) - penalty.dot(beta)
        step = np.linalg.solve(hessian, grad)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-9:
            break
    p = 1.0 / (1.0 + np.exp(-X.dot(beta)))
    hessian = (X * (p * (1.0 - p))[:, None]).T.dot(X) + penalty
    se = np.sqrt(np.diag(np.linalg.inv(hessian)))
    b0, b_click, b_prev, b_win = (float(b) for b in beta)
    z_click = b_click / float(se[1])
    met = bool(b_click > 0 and z_click >= CLICK_WEIGHT_Z_CRITERION
               and b_click > max(abs(b_prev), abs(b_win)))
    return {'b0': b0, 'b_click': b_click, 'b_prev': b_prev, 'b_win': b_win, 'z_click': z_click,
            'n': n, 'criterion_met': met}
