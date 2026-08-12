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
from collections import deque

DIRECTION_RATIO_WINDOW = 40
DIRECTION_RATIO_BAND = (0.30, 0.70)


class DirectionRatioTracker(object):
    """ Usage per trial: check should_withhold(side) / over_used_side() FIRST (using the ratio
    from trials up to but not including this one), THEN call record(side) after the direction is
    known -- so a trial's own withhold decision is judged against the ratio that existed BEFORE
    it, matching how trial_scheduler.py's side-bias functions compute e_right/e_left before
    drawing the current trial's side, then record the outcome after.

    record(side) is called for every trial's actual movement direction regardless of whether
    reward was delivered or withheld -- the ratio tracks what the animal DID, not what was
    rewarded. """

    def __init__(self, window=DIRECTION_RATIO_WINDOW, band=DIRECTION_RATIO_BAND):
        self._band = band
        self._sides = deque(maxlen=window)

    def record(self, side):
        self._sides.append(side)

    def right_fraction(self):
        """ Rolling right-fraction over however many trials exist so far (up to the window size) --
        0.5 (neutral) if no trials recorded yet. """
        if not self._sides:
            return 0.5
        return sum(1 for s in self._sides if s == 'R') / len(self._sides)

    def in_band(self):
        lo, hi = self._band
        return lo <= self.right_fraction() <= hi

    def over_used_side(self):
        """ 'R'/'L' if the rolling ratio has left the band on that side, else None. """
        if self.in_band():
            return None
        return 'R' if self.right_fraction() > self._band[1] else 'L'

    def should_withhold(self, side):
        """ True if `side` is currently the over-used side -- i.e. this trial's movement
        direction should not be rewarded even though it cleared the movement threshold. """
        return self.over_used_side() == side
