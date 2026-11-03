# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Rolling last-N-trial direction ratio + in-band check, for Stage 2(d)'s side-bias management --
training_protocol.md: "Track the running direction ratio over the last ~40 trials; if it leaves the
30-70% band, withhold reward on the over-used side until it returns to band." Pure Python, no
Bpod/hardware dependency -- independently unit-testable, see validate_wheel_shaping.py.

Stage 1 does NOT use this at all -- the doc places side-bias management under Stage 2 only (Stage 1
has no reward-withholding of any kind, either direction always rewards).
"""
import math
from collections import deque

DIRECTION_RATIO_WINDOW = 40
DIRECTION_RATIO_BAND = (0.30, 0.70)
DIRECTION_RATIO_MIN_TRIALS = 10   # withholding starts once this many directions are known (seeded
                                  # from earlier sessions + this one) -- see has_enough_trials()
# The ratio is recency-weighted (2026-11-02): half-Gaussian weights over the window, sigma in
# trials (~10 effective trials at 8), blended with PRIOR_WEIGHT neutral (0.5) pseudo-trials -- the
# same formula as Tests/.../trial_scheduler.py's _half_gaussian_weighted_fraction(). The flat
# 40-trial ratio lagged so far behind that a perseverating animal oscillated between long runs:
# UM001 20261102-161142 turned R x31, L x29, ... (22% switches), each run rewarded for ~16 trials
# before the ratio left the band, then withheld until it flipped. Weighted, a run from a balanced
# history is withheld after 6 same-side turns and 3 opposite turns bring it back into band.
DIRECTION_RATIO_SIGMA = 8.0
DIRECTION_RATIO_PRIOR_WEIGHT = 2.0


class DirectionRatioTracker(object):
    """ Usage per trial: check should_withhold(side) / over_used_side() FIRST (using the ratio
    from trials up to but not including this one), THEN call record(side) after the direction is
    known -- so a trial's own withhold decision is judged against the ratio that existed BEFORE
    it, matching how trial_scheduler.py's side-bias functions compute e_right/e_left before
    drawing the current trial's side, then record the outcome after.

    record(side) is called for every trial's actual movement direction regardless of whether
    reward was delivered or withheld -- the ratio tracks what the animal DID, not what was
    rewarded. """

    def __init__(self, window=DIRECTION_RATIO_WINDOW, band=DIRECTION_RATIO_BAND,
                 min_trials=DIRECTION_RATIO_MIN_TRIALS, history=None,
                 sigma=DIRECTION_RATIO_SIGMA, prior_weight=DIRECTION_RATIO_PRIOR_WEIGHT):
        """ :param history: the subject's most recent directions from earlier sessions (oldest
        first; Stage 2 persists history() as `recent_directions`). Only the last `window` are kept.
        Without it the window started empty every session, so a side-biased animal got ~40 free
        rewards on its favoured side before withholding could start (UM001, 20261102-160353:
        22/22 left turns, all rewarded). """
        self._band = band
        self._sigma = sigma
        self._prior_weight = prior_weight
        self._min_trials = min(min_trials, window)
        self._sides = deque((s for s in (history or []) if s in ('L', 'R')), maxlen=window)
        self.seeded_trials = len(self._sides)

    def record(self, side):
        self._sides.append(side)

    def right_fraction(self):
        """ Recency-weighted right-fraction over the window (newest trial weight 1, decaying as a
        half-Gaussian of age), blended with prior_weight neutral pseudo-trials -- 0.5 if no trials
        recorded yet. This is what drives withholding. """
        n = len(self._sides)
        weights = [math.exp(-0.5 * ((n - 1 - i) / self._sigma) ** 2) for i in range(n)]
        num = sum(w for w, s in zip(weights, self._sides) if s == 'R') + self._prior_weight * 0.5
        return num / (sum(weights) + self._prior_weight)

    def plain_right_fraction(self):
        """ The unweighted right-fraction over the window (the pre-2026-11-02 ratio), for logging
        only -- 0.5 if no trials recorded yet. """
        if not self._sides:
            return 0.5
        return sum(1 for s in self._sides if s == 'R') / len(self._sides)

    def history(self):
        """ The directions in the window, oldest first -- what Stage 2 persists between sessions. """
        return list(self._sides)

    def has_full_window(self):
        return len(self._sides) >= self._sides.maxlen

    def has_enough_trials(self):
        """ False until `min_trials` directions are known (earlier sessions' included) --
        right_fraction() over a handful of trials is pure noise (a single early trial already
        swings it to 0.0/1.0), the same class of bug already fixed elsewhere in this codebase for
        trial_scheduler.should_stop_session()'s rate checks (see CLAUDE.md). Confirmed on hardware:
        OUTCOME=Withheld fired as early as trial 2 when there was no gate at all. Until 2026-11-02
        the gate was the full 40-trial window of the current session only. """
        return len(self._sides) >= self._min_trials

    def in_band(self):
        """ Always True (no withholding) until has_enough_trials() -- not enough data yet to trust
        the ratio. """
        if not self.has_enough_trials():
            return True
        lo, hi = self._band
        return lo <= self.right_fraction() <= hi

    def over_used_side(self):
        """ 'R'/'L' if the weighted ratio has left the band on that side, else None. """
        if self.in_band():
            return None
        return 'R' if self.right_fraction() > self._band[1] else 'L'

    def should_withhold(self, side):
        """ True if `side` is currently the over-used side -- i.e. this trial's movement
        direction should not be rewarded even though it cleared the movement threshold. """
        return self.over_used_side() == side
