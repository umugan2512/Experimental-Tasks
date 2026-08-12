# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Pure-Python (no Bpod/hardware) rules for Stage 1's gain decay and Stage 2's movement-threshold
staircase and ITI growth -- training_protocol.md Part 4, Stage 1's parameter table and Stage 2(a)/(b).
Independently unit-testable offline, see validate_wheel_shaping.py.
"""

STAIRCASE_STEP_UP = 0.10          # +10% (linear, of the FINAL threshold) after
STAIRCASE_SUCCESSES_TO_STEP_UP = 20  # this many consecutive successes
STAIRCASE_STEP_DOWN = 0.10        # -10% after
STAIRCASE_FAILURES_TO_STEP_DOWN = 5  # this many consecutive failures
# Doc: "Ceiling at the final threshold -- never exceed it." current_fraction is expressed as a
# fraction of the final threshold (0.0-1.0), so the ceiling is simply 1.0.

ITI_GROWTH_PER_SESSION_S = 0.1    # doc-specified: "~0.1s per session"
ITI_CEILING_S = 1.5               # doc-specified: Stage 2's final ITI

# Stage 1's threshold grows ACROSS sessions (one fixed step per qualifying session), NOT within a
# session like ThresholdStaircase above -- Stage 1 isn't tracking moment-to-moment performance the
# way Stage 2 is, it just needs to reach Stage 2's own starting point by the time Stage 1 ends.
# Same fixed-per-qualifying-session-step shape as ITI growth above, not the consecutive-count
# staircase. Rate not specified by the doc (which predates this progression existing at all) --
# flagged/tunable, picked so 0.05->0.20 takes roughly the same number of qualifying sessions as
# training_protocol.md's original 3-7 day Stage 1 duration estimate.
THRESHOLD_GROWTH_PER_SESSION_FRACTION = 0.02
THRESHOLD_GROWTH_CEILING_FRACTION = 0.20   # Stage 2's own starting point

# The doc states Stage 1's wheel gain drops "after a session >200 trials" but doesn't give an
# exact rate (unlike ITI's explicit "0.1s per session") -- flagged/tunable, picked to match the
# same fixed-per-qualifying-session-step convention as ITI growth.
GAIN_DECAY_PER_SESSION = 0.90     # multiplicative, per qualifying (>200 trial) session
GAIN_FLOOR_MULT = 1.0             # never decay below the final (1x) gain


class ThresholdStaircase(object):
    """ Tracks a movement threshold (as a fraction of the FINAL threshold, 0.0-1.0) plus
    consecutive success/failure counters, per training_protocol.md Stage 2(a): +10% threshold
    after 20 consecutive successes, -10% after 5 consecutive failures, ceiling at 1.0 (never
    exceed the final threshold; no floor is specified, but 0.0 is the natural lower bound). Each
    consecutive counter resets to 0 on the OPPOSITE outcome (a single failure resets the success
    streak and vice versa) -- same "streak resets on the opposite outcome" convention as
    trial_scheduler.py's own consecutive-run counters elsewhere in this project.

    Constructed fresh from persisted StageState each session (current_fraction/
    consecutive_successes/consecutive_failures all round-trip through it), not reset to an initial
    value -- the whole point of the staircase is that it survives across days. """

    def __init__(self, current_fraction, consecutive_successes=0, consecutive_failures=0):
        self.current_fraction = current_fraction
        self.consecutive_successes = consecutive_successes
        self.consecutive_failures = consecutive_failures

    def record_outcome(self, success):
        """ success=True if this trial's movement cleared the CURRENT threshold (a genuine turn),
        False if it was a sub-threshold/aborted attempt. """
        if success:
            self.consecutive_successes += 1
            self.consecutive_failures = 0
            if self.consecutive_successes >= STAIRCASE_SUCCESSES_TO_STEP_UP:
                self.current_fraction = min(1.0, self.current_fraction + STAIRCASE_STEP_UP)
                self.consecutive_successes = 0
        else:
            self.consecutive_failures += 1
            self.consecutive_successes = 0
            if self.consecutive_failures >= STAIRCASE_FAILURES_TO_STEP_DOWN:
                self.current_fraction = max(0.0, self.current_fraction - STAIRCASE_STEP_DOWN)
                self.consecutive_failures = 0


def grow_iti(prev_iti_s):
    """ +ITI_GROWTH_PER_SESSION_S per qualifying session, ceiling at ITI_CEILING_S -- call once per
    session (not per trial), per training_protocol.md Stage 2(b). """
    return min(ITI_CEILING_S, prev_iti_s + ITI_GROWTH_PER_SESSION_S)


def grow_stage1_threshold(prev_fraction):
    """ +THRESHOLD_GROWTH_PER_SESSION_FRACTION per qualifying session, ceiling at
    THRESHOLD_GROWTH_CEILING_FRACTION -- call once per qualifying session (not per trial), same
    call-site shape as grow_iti(). Not ThresholdStaircase: Stage 1 doesn't get a per-trial,
    consecutive-count-driven staircase, just this fixed session-level step. """
    return min(THRESHOLD_GROWTH_CEILING_FRACTION, prev_fraction + THRESHOLD_GROWTH_PER_SESSION_FRACTION)


def decay_gain(prev_gain_mult):
    """ Multiplicative decay toward GAIN_FLOOR_MULT per qualifying (>200 trial) session, per
    Stage 1's "gain ~2x final, dropping after a session >200 trials". Call once per qualifying
    session (not per trial). """
    return max(GAIN_FLOOR_MULT, prev_gain_mult * GAIN_DECAY_PER_SESSION)


def stage2_simple_gates_met(trial_count, iti_s, direction_ratio_in_band, trial_count_gate=200):
    """ The three simple, doc-specified numeric gates for Stage 2 advancement (trial count, ITI at
    its ceiling, direction ratio in-band) -- NOT the full advancement decision, which per
    training_protocol.md is primarily a statistical test (bimodal movement distribution + a
    velocity-separation criterion) deliberately deferred (no real Stage 2 movement data exists yet
    to validate a concrete algorithm against -- see stage2_threshold_staircase.py's module
    docstring). This only reports whether the easy, unambiguous gates are met; the statistical
    test remains a human judgment call until it's built. """
    return trial_count > trial_count_gate and iti_s >= ITI_CEILING_S and direction_ratio_in_band
