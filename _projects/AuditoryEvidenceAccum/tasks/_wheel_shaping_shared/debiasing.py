# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Stage 3/4 side-debiasing rule (training_protocol.md Revision 2): "after an error, re-present the
same side with high probability." Deliberately a small, standalone function, NOT a port of
Tests/tasks/full_protocol_lookback_test/trial_scheduler.py's more elaborate recency-weighted
PerformanceMonitor/TrialHistory lookback machinery -- that system implements a different, more
complex debiasing policy than what the Stage 3/4 protocol document actually specifies, so building
a small function that matches the written spec directly (and stays easy to verify against it) was
chosen over reusing/adapting a system that would diverge from the doc. See
validate_wheel_shaping.py for an offline statistical check of the realized repeat-rate against
VAR_DEBIAS_REPEAT_PROB.
"""
import numpy as np

# The doc says "high probability" without giving an exact number -- documented, tunable default.
# Not the same knob as anything in trial_scheduler.py (that system has no equivalent constant).
VAR_DEBIAS_REPEAT_PROB = 0.80


def next_side_after_error(prev_side, rng=None, repeat_prob=VAR_DEBIAS_REPEAT_PROB):
    """ Call only when the PREVIOUS trial was an error (outcome == 'incorrect') -- callers should
    draw a fresh 50/50 side via rng.choice(['L', 'R']) on the first trial of a session and after
    any CORRECT trial; this function is the error-branch only.

    :param str prev_side: the side that was presented (and turned away from) on the error trial.
    :param rng: numpy Generator/RandomState, defaults to np.random.
    :param float repeat_prob: probability of re-presenting prev_side again; defaults to
        VAR_DEBIAS_REPEAT_PROB, but a caller (or an offline validation script) can supply an
        explicit value to test a different setting.
    :return: 'L' or 'R'.
    """
    rng = rng if rng is not None else np.random
    if rng.random() < repeat_prob:
        return prev_side
    return 'L' if prev_side == 'R' else 'R'
