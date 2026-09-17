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
# same fixed-per-qualifying-session-step convention as ITI growth. Fixed additive step (not
# multiplicative), same shape as grow_iti()/grow_stage1_threshold() -- confirmed on hardware that
# an early multiplicative version decayed too fast/too far (floor 1.0x made early-session
# movements barely discernible); the floor is now 2.0x, never below "clearly visually easy."
GAIN_DECAY_STEP_PER_SESSION = 0.1   # per qualifying (>200 trial) session
GAIN_FLOOR_MULT = 2.0               # never decay below 2x -- floor picked so movements stay
                                     # clearly discernible even late in Stage 1


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
    """ -GAIN_DECAY_STEP_PER_SESSION per qualifying (>200 trial) session, floor at GAIN_FLOOR_MULT
    -- call once per qualifying session (not per trial), same call-site shape as grow_iti(). """
    return max(GAIN_FLOOR_MULT, prev_gain_mult - GAIN_DECAY_STEP_PER_SESSION)


def stage2_simple_gates_met(trial_count, iti_s, direction_ratio_in_band, trial_count_gate=200):
    """ The three simple, doc-specified numeric gates for Stage 2 advancement (trial count, ITI at
    its ceiling, direction ratio in-band) -- NOT the full advancement decision, which per
    training_protocol.md is primarily a statistical test (bimodal movement distribution + a
    velocity-separation criterion) deliberately deferred (no real Stage 2 movement data exists yet
    to validate a concrete algorithm against -- see stage2_threshold_staircase.py's module
    docstring). This only reports whether the easy, unambiguous gates are met; the statistical
    test remains a human judgment call until it's built. """
    return trial_count > trial_count_gate and iti_s >= ITI_CEILING_S and direction_ratio_in_band


# --- Stage 3/4 (training_protocol.md Revision 2) -----------------------------------------------

QUIESCENCE_STEP_UP_S = 0.025        # doc: "+25ms per 20 consecutive successful initiations"
QUIESCENCE_SUCCESSES_TO_STEP_UP = 20
QUIESCENCE_STEP_DOWN_S = 0.025      # doc: "-25ms per 5 resets"
QUIESCENCE_FAILURES_TO_STEP_DOWN = 5
QUIESCENCE_FLOOR_S = 0.1            # Stage 1-3's own frozen value -- the natural starting point
QUIESCENCE_CEILING_S = 0.5          # doc: "0.1-0.5s exponential"

STAGE3_ACCURACY_GATE = 0.70         # doc: ">70% correct at gamma=+-1.0"
STAGE3_ABORT_RATE_GATE = 0.20       # doc: "abort rate <20%"

STAGE4_ACCURACY_GATE = 0.70         # doc: ">70% correct at gamma=+-1.0", same as Stage 3
STAGE4_ABORT_RATE_GATE = 0.25       # doc: "abort rate <25%" -- looser than Stage 3's, since Stage
                                     # 4's staircases are actively tightening again
STAGE4_RESPONSE_THRESHOLD_FINAL_FRACTION = 0.90   # doc: "current -> final (90% of edge azimuth)"
                                                    # -- Stage 4's own target, NOT ThresholdStaircase's
                                                    # own 1.0 ceiling (which stays unchanged; Stage 4
                                                    # just treats 0.90 as "done" for advancement).


class QuiescenceStaircase(object):
    """ Tracks the quiescence-duration EXPONENTIAL DRAW's scale parameter (seconds), resumed at
    Stage 4 per training_protocol.md: frozen at 100ms through Stages 1-3, then staircased toward a
    0.1-0.5s exponential -- +25ms per 20 consecutive successful initiations (no reset during that
    trial's quiescence hold), -25ms per 5 consecutive resets, floor QUIESCENCE_FLOOR_S, ceiling
    QUIESCENCE_CEILING_S. Same consecutive-counter/streak-reset shape as ThresholdStaircase above.
    A trial's actual quiescence duration is drawn fresh each trial as
    np.random.exponential(current_s) clipped to [QUIESCENCE_FLOOR_S, QUIESCENCE_CEILING_S] --
    current_s here is the distribution's scale parameter, not a fixed per-trial duration.

    Constructed fresh from persisted StageState each session, same round-trip convention as
    ThresholdStaircase. """

    def __init__(self, current_s, consecutive_successes=0, consecutive_failures=0):
        self.current_s = current_s
        self.consecutive_successes = consecutive_successes
        self.consecutive_failures = consecutive_failures

    def record_outcome(self, success):
        """ success=True if this trial's quiescence hold completed without a reset (a genuine
        clean initiation), False if at least one reset occurred during it. """
        if success:
            self.consecutive_successes += 1
            self.consecutive_failures = 0
            if self.consecutive_successes >= QUIESCENCE_SUCCESSES_TO_STEP_UP:
                self.current_s = min(QUIESCENCE_CEILING_S, self.current_s + QUIESCENCE_STEP_UP_S)
                self.consecutive_successes = 0
        else:
            self.consecutive_failures += 1
            self.consecutive_successes = 0
            if self.consecutive_failures >= QUIESCENCE_FAILURES_TO_STEP_DOWN:
                self.current_s = max(QUIESCENCE_FLOOR_S, self.current_s - QUIESCENCE_STEP_DOWN_S)
                self.consecutive_failures = 0


def stage3_gates_met(accuracy_aos, abort_rate, trial_count, trial_count_gate=200):
    """ training_protocol.md Stage 3 advancement: >70% correct at gamma=+-1.0 (AOS), abort rate
    <20%, >200 trials/session. Unlike Stage 2's gate (stage2_simple_gates_met -- deliberately
    partial, since Stage 2's real criterion is a deferred statistical test), this covers Stage 3's
    ENTIRE doc-specified advancement criterion; no separate human-judgment component. Caller
    tracks the "three consecutive sessions" requirement itself (same StageState-backed history
    list pattern as sessions_trial_count_history elsewhere), this only checks one session's own
    gates. accuracy_aos/abort_rate must already exclude warmup/repeat trials (see
    stage3_clicks_direction.py's own trial classification) -- this function trusts its inputs,
    it doesn't re-derive them from raw trial data. """
    return (accuracy_aos > STAGE3_ACCURACY_GATE and abort_rate < STAGE3_ABORT_RATE_GATE
            and trial_count > trial_count_gate)


def stage4_gates_met(response_threshold_fraction, quiescence_s, accuracy_aos, abort_rate,
                      trial_count, trial_count_gate=200):
    """ training_protocol.md Stage 4 advancement: both staircases at final values (response
    threshold >= STAGE4_RESPONSE_THRESHOLD_FINAL_FRACTION of edge azimuth, quiescence at
    QUIESCENCE_CEILING_S), >70% correct at gamma=+-1.0, abort rate <25%, >=200 trials/session.
    Caller tracks the "two consecutive sessions" requirement itself. """
    response_at_final = response_threshold_fraction >= STAGE4_RESPONSE_THRESHOLD_FINAL_FRACTION
    quiescence_at_final = quiescence_s >= QUIESCENCE_CEILING_S
    return (response_at_final and quiescence_at_final and accuracy_aos > STAGE4_ACCURACY_GATE
            and abort_rate < STAGE4_ABORT_RATE_GATE and trial_count >= trial_count_gate)


def select_stage4_staircase_to_tighten(response_threshold_fraction, quiescence_s, last_advanced):
    """ Tag B (training_protocol.md Revision 2 Appendix A): at most ONE of Stage 4's two
    staircases (response threshold, quiescence) may tighten in any given SESSION -- running both
    concurrently can compound-step the animal on a dimension it didn't earn, producing a
    performance "collapse" that reads as the animal losing the task and makes a later drop
    decision unattributable to either staircase specifically. The doc leaves the selection policy
    itself open ("alternate across sessions, or prioritise whichever is further from final").

    Policy chosen here: prioritize whichever parameter is furthest from ITS OWN final value,
    normalized to a comparable 0.0-1.0 "fraction remaining" for each (response threshold's own
    units are already a 0-1 fraction; quiescence's seconds are normalized against its own
    floor-ceiling span) since the two staircases use different units and aren't otherwise
    comparable; tie-break by alternating away from last_advanced (the previous session's own
    choice -- 'response'/'quiescence'/None, read from persisted state).

    Returns 'response' or 'quiescence' -- the caller applies record_outcome() calls to ONLY the
    returned staircase's own object this session; the other staircase's counters may still be
    tracked/logged, but its current_fraction/current_s must not change until a future session
    selects it. """
    response_remaining = (max(0.0, STAGE4_RESPONSE_THRESHOLD_FINAL_FRACTION
                               - response_threshold_fraction) / STAGE4_RESPONSE_THRESHOLD_FINAL_FRACTION)
    quiescence_span = QUIESCENCE_CEILING_S - QUIESCENCE_FLOOR_S
    quiescence_remaining = max(0.0, QUIESCENCE_CEILING_S - quiescence_s) / quiescence_span

    if response_remaining > quiescence_remaining:
        return 'response'
    if quiescence_remaining > response_remaining:
        return 'quiescence'
    return 'quiescence' if last_advanced == 'response' else 'response'
