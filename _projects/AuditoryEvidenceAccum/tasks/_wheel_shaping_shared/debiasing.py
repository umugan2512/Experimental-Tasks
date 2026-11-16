# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Side debiasing for Stages 3-5: Hu et al. (2009)'s rule (2026-11-11 spec B2), replacing the old
"after an error, re-present the same side with probability 0.80" (next_side_after_error()).

Why the old rule went: it does penalise a fixed side bias (an always-right agent earned 29%), but an
animal that simply switches its choice after an error (lose-shift) earned 80% on post-error trials
and 62% overall without listening. That teaches a history strategy and builds a history dependence
into the stimulus sequence, which contaminates the trial-history terms of any choice model.

The rule (same formula as Tests/tasks/poisson_clicks_test/trial_scheduler.py's lookback code, which
this module deliberately doesn't import -- the real project doesn't depend on Tests/):
    e_R, e_L = half-Gaussian recency-weighted error rates (sigma 20 trials) over the last 40 right /
               left trials, blended with PRIOR_WEIGHT neutral (0.5) pseudo-trials
    p_R      = clip(sqrt(e_R)) / (clip(sqrt(e_R)) + clip(sqrt(e_L))),  clip to [0.15, 0.85]
    f_R      = empirical right-trial fraction, half-Gaussian sigma 60 over the last 60 trials
    draw right with probability 0.5 * p_R if f_R > p_R, else 0.5 * (1 + p_R)
An exact tie f_R == p_R draws 50/50 (trial_scheduler's fix: both are 0.5 before any history exists,
which would otherwise push every session's first trial to the right with probability 0.75).

Only trials with a genuine L/R response are recorded. The abort re-offer (same side again after an
abort) is not debiasing and stays in the stage scripts; Stage 3's temporary correction trials are a
separate mechanism layered on top (stage3_scaffold.py) and are not recorded here as current trials.
The history persists across sessions (state key 'hu_debias_history'), same pattern as
direction_tracker's 'recent_directions'. Stage 2's DirectionRatioTracker is a different mechanism.
"""
import math

HU_ERROR_LOOKBACK_N = 40      # last N trials of each side for e_R / e_L
HU_ERROR_SIGMA = 20.0
HU_SIDE_FRACTION_LOOKBACK_N = 60
HU_SIDE_FRACTION_SIGMA = 60.0
HU_PRIOR_WEIGHT = 0.0
HU_SQRT_CLIP = (0.15, 0.85)
HU_HISTORY_KEEP = 200         # trials kept (and persisted): enough for 40 of each side unless the
                              # sides ran >80% one way for 200 trials


def _weighted_fraction(values, sigma, prior_weight):
    """ values oldest first; newest weight 1, half-Gaussian decay with age, plus prior_weight
    neutral (0.5) pseudo-trials. """
    n = len(values)
    if n == 0 and prior_weight == 0:
        return 0.5
    weights = [math.exp(-0.5 * ((n - 1 - i) / sigma) ** 2) for i in range(n)]
    return ((sum(w * v for w, v in zip(weights, values)) + prior_weight * 0.5)
            / (sum(weights) + prior_weight))


class HuDebiaser(object):
    """ next_side(rng) draws the next trial's side and keeps what it used in `last` (p_right, e_right,
    e_left, f_right, prob_right) for logging; record(side, correct) after every responded trial. """

    def __init__(self, history=None):
        """ :param history: [[side, correct], ...] oldest first, as history() returns it. """
        self._trials = [(s, bool(c)) for s, c in (history or []) if s in ('L', 'R')][-HU_HISTORY_KEEP:]
        self.seeded_trials = len(self._trials)
        self.last = None

    def record(self, side, correct):
        self._trials.append((side, bool(correct)))
        del self._trials[:-HU_HISTORY_KEEP]

    def history(self):
        return [[s, c] for s, c in self._trials]

    def error_rate(self, side):
        errors = [0.0 if c else 1.0 for s, c in self._trials if s == side][-HU_ERROR_LOOKBACK_N:]
        return _weighted_fraction(errors, HU_ERROR_SIGMA, HU_PRIOR_WEIGHT)

    def right_fraction(self):
        rights = [1.0 if s == 'R' else 0.0 for s, _ in self._trials][-HU_SIDE_FRACTION_LOOKBACK_N:]
        return _weighted_fraction(rights, HU_SIDE_FRACTION_SIGMA, HU_PRIOR_WEIGHT)

    def p_right(self):
        lo, hi = HU_SQRT_CLIP
        sqrt_r = min(max(math.sqrt(self.error_rate('R')), lo), hi)
        sqrt_l = min(max(math.sqrt(self.error_rate('L')), lo), hi)
        return sqrt_r / (sqrt_r + sqrt_l)

    def next_side(self, rng):
        """ rng: anything with random_sample() (np.random or a RandomState). """
        p_r, f_r = self.p_right(), self.right_fraction()
        if f_r == p_r:
            prob_right = 0.5
        elif f_r > p_r:
            prob_right = 0.5 * p_r
        else:
            prob_right = 0.5 * (1.0 + p_r)
        self.last = {'p_right': p_r, 'e_right': self.error_rate('R'), 'e_left': self.error_rate('L'),
                     'f_right': f_r, 'prob_right': prob_right}
        return 'R' if float(rng.random_sample()) < prob_right else 'L'
