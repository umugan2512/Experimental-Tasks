# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Pure-Python (no Bpod/hardware) rules for Stage 1's gain decay and Stage 2's movement-threshold
staircase and ITI growth -- training_protocol.md Part 4, Stage 1's parameter table and Stage 2(a)/(b).
Independently unit-testable offline, see validate_wheel_shaping.py.
"""
import math

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


# --- Stage 3/4/5 (training_protocol.md Revision 2) -----------------------------------------------

QUIESCENCE_STEP_UP_S = 0.025        # doc: "+25ms per 20 consecutive successful initiations"
QUIESCENCE_SUCCESSES_TO_STEP_UP = 20
QUIESCENCE_STEP_DOWN_S = 0.025      # doc: "-25ms per 5 resets"
QUIESCENCE_FAILURES_TO_STEP_DOWN = 5
QUIESCENCE_FLOOR_S = 0.1            # Stage 1's own frozen value -- the natural starting point
QUIESCENCE_CEILING_S = 0.25         # per explicit instruction: hold-to-initiate should land in the
                                     # 100-250ms range, not the doc's original 0.1-0.5s -- lowered
                                     # from an earlier 0.5s ceiling.

STAGE3_ACCURACY_GATE = 0.70         # doc doesn't specify this new stage's own numeric criterion
                                     # yet -- reusing Stage 4's values as a flagged placeholder
                                     # pending real empirical tuning (same convention as
                                     # VAR_CUE_ABORT_THRESHOLD_DEG_PLACEHOLDER elsewhere)
STAGE3_ABORT_RATE_GATE = 0.20

STAGE4_ACCURACY_GATE = 0.70         # doc: ">70% correct at gamma=+-1.0"
STAGE4_ABORT_RATE_GATE = 0.20       # doc: "abort rate <20%"

STAGE5_ACCURACY_GATE = 0.70         # doc: ">70% correct at gamma=+-1.0", same as Stage 3/4
STAGE5_ABORT_RATE_GATE = 0.25       # doc: "abort rate <25%" -- looser than Stage 3/4's, since
                                     # Stage 5's staircases are actively tightening again
STAGE5_RESPONSE_THRESHOLD_FINAL_FRACTION = 0.90   # doc: "current -> final (90% of edge azimuth)"
                                                    # -- Stage 5's own target, NOT ThresholdStaircase's
                                                    # own 1.0 ceiling (which stays unchanged; Stage 5
                                                    # just treats 0.90 as "done" for advancement).

# Stage 3/4's stimulus-duration and ITI ramps, per explicit instruction ("similar to degree or ISI
# ramp" -- i.e. the same step-up-on-consecutive-correct/step-down-on-consecutive-incorrect shape
# ThresholdStaircase already established for the movement threshold). Doc doesn't specify either
# ramp at all (0.25s stimulus duration and 1.5s ITI were previously FIXED Stage 3/4 constants) --
# every value below is a flagged/tunable placeholder, same convention as
# VAR_CUE_ABORT_THRESHOLD_DEG_PLACEHOLDER elsewhere. Both start at the doc's own original fixed
# value and grow toward a longer (easier -- more time/evidence per trial, or more consumption/
# rest time) final value as performance improves, shrinking back toward that starting value on a
# consecutive-incorrect streak -- never below it, since that was the doc's own doc-specified
# floor to begin with. Shared across Stage 3 and Stage 4 (one continuous persisted value each,
# not reset when a subject advances from Stage 3 to Stage 4), same cross-stage persistence
# convention as the response-threshold `threshold_fraction`.
STAGE34_STIM_DURATION_STARTING_S = 0.25   # doc's own original fixed Stage 3/4 value, now reused
                                           # as this ramp's own starting point AND floor
STAGE34_STIM_DURATION_FINAL_S = 2.0       # click_train_v2's own module-level default duration --
                                           # a natural, already-present "full" reference
STAGE34_STIM_DURATION_STEP_S = 0.25
STAGE34_STIM_DURATION_SUCCESSES_TO_STEP_UP = 20   # same consecutive-counts as ThresholdStaircase
STAGE34_STIM_DURATION_FAILURES_TO_STEP_DOWN = 5

STAGE34_ITI_STARTING_S = 1.5               # doc's own original fixed Stage 3/4 correct-trial ITI
                                            # (also Stage 2's own final ITI_CEILING_S) -- this
                                            # ramp's own starting point AND floor. The separate
                                            # error-trial VAR_INCORRECT_ITI_S (4.0s) is NOT ramped
                                            # -- only the one continuous ITI value Stage 2's own
                                            # grow_iti() already established gets this treatment.
STAGE34_ITI_FINAL_S = 3.0
STAGE34_ITI_STEP_S = 0.25
STAGE34_ITI_SUCCESSES_TO_STEP_UP = 20
STAGE34_ITI_FAILURES_TO_STEP_DOWN = 5


class BoundedStaircase(object):
    """ Generic step-up-after-N-consecutive-successes / step-down-after-N-consecutive-failures
    staircase, bounded by [floor_value, ceiling_value] -- the same mechanic as ThresholdStaircase/
    QuiescenceStaircase above, generalized so Stage 3/4's new stimulus-duration and ITI ramps
    don't need their own near-duplicate classes (they're the identical shape, just different
    units/step-sizes/bounds). Each consecutive counter resets to 0 on the OPPOSITE outcome, same
    "streak resets on the opposite outcome" convention used throughout this file.

    Constructed fresh from persisted StageState each session (current_value/consecutive_successes/
    consecutive_failures all round-trip through it), not reset to the starting value -- the whole
    point is that it survives across days (and, for Stage 3/4's shared ramps, across the stage
    boundary too). """

    def __init__(self, current_value, step_up, step_down, successes_to_step_up,
                 failures_to_step_down, floor_value, ceiling_value,
                 consecutive_successes=0, consecutive_failures=0):
        self.current_value = current_value
        self._step_up = step_up
        self._step_down = step_down
        self._successes_to_step_up = successes_to_step_up
        self._failures_to_step_down = failures_to_step_down
        self._floor_value = floor_value
        self._ceiling_value = ceiling_value
        self.consecutive_successes = consecutive_successes
        self.consecutive_failures = consecutive_failures

    def record_outcome(self, success):
        if success:
            self.consecutive_successes += 1
            self.consecutive_failures = 0
            if self.consecutive_successes >= self._successes_to_step_up:
                self.current_value = min(self._ceiling_value, self.current_value + self._step_up)
                self.consecutive_successes = 0
        else:
            self.consecutive_failures += 1
            self.consecutive_successes = 0
            if self.consecutive_failures >= self._failures_to_step_down:
                self.current_value = max(self._floor_value, self.current_value - self._step_down)
                self.consecutive_failures = 0


def new_stim_duration_staircase(current_s, consecutive_successes=0, consecutive_failures=0):
    """ Factory for Stage 3/4's shared stimulus-duration BoundedStaircase -- see
    STAGE34_STIM_DURATION_* constants above for the actual numbers. """
    return BoundedStaircase(
        current_s, STAGE34_STIM_DURATION_STEP_S, STAGE34_STIM_DURATION_STEP_S,
        STAGE34_STIM_DURATION_SUCCESSES_TO_STEP_UP, STAGE34_STIM_DURATION_FAILURES_TO_STEP_DOWN,
        STAGE34_STIM_DURATION_STARTING_S, STAGE34_STIM_DURATION_FINAL_S,
        consecutive_successes, consecutive_failures)


def new_iti_staircase(current_s, consecutive_successes=0, consecutive_failures=0):
    """ Factory for Stage 3/4's shared ITI BoundedStaircase -- see STAGE34_ITI_* constants above
    for the actual numbers. """
    return BoundedStaircase(
        current_s, STAGE34_ITI_STEP_S, STAGE34_ITI_STEP_S,
        STAGE34_ITI_SUCCESSES_TO_STEP_UP, STAGE34_ITI_FAILURES_TO_STEP_DOWN,
        STAGE34_ITI_STARTING_S, STAGE34_ITI_FINAL_S,
        consecutive_successes, consecutive_failures)


def next_alternating_value(last_value, last_date_str, today_str, choices):
    """ Deterministic part of Stage 3's own "same value within a calendar day, alternates across
    days" persisted side choice -- callers handle the separate "very first time ever" random-pick
    branch themselves (last_value is None), since there's nothing to unit-test about picking
    randomly from `choices`. Returns last_value unchanged if last_date_str == today_str (a
    same-day restart); otherwise returns the OTHER entry in `choices` (deterministically
    alternates, NOT a fresh random draw). `choices` must have exactly 2 distinct entries for "the
    other one" to be well-defined -- e.g. ('L', 'R'). """
    if last_date_str == today_str:
        return last_value
    return next(c for c in choices if c != last_value)


class QuiescenceStaircase(object):
    """ Tracks the quiescence-duration EXPONENTIAL DRAW's scale parameter (seconds). Frozen at
    100ms in Stage 1, and in Stage 2 until that stage's own movement-threshold staircase steps up
    at least once within a session (a one-way 'quiescence_unlocked' latch, persisted in
    StageState) -- staircasing then begins the NEXT session and continues uninterrupted through
    Stage 2/3/4/5: +25ms per 20 consecutive successful initiations (no reset during that trial's
    quiescence hold), -25ms per 5 consecutive resets, floor QUIESCENCE_FLOOR_S, ceiling
    QUIESCENCE_CEILING_S (100-250ms). Same consecutive-counter/streak-reset shape as
    ThresholdStaircase above.
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
    """ training_protocol.md Stage 3 ("tone-side association", new stage) advancement -- the doc
    doesn't specify this new stage's own numeric criterion yet, so this reuses Stage 4's values
    (STAGE3_ACCURACY_GATE/STAGE3_ABORT_RATE_GATE, currently identical to STAGE4_*) as a flagged
    placeholder pending real empirical tuning. Same shape/caller contract as stage4_gates_met()
    below: caller tracks the "three consecutive sessions" requirement itself, accuracy_aos/
    abort_rate must already exclude warmup trials (see stage3_tone_side_association.py's own
    trial classification). """
    return (accuracy_aos > STAGE3_ACCURACY_GATE and abort_rate < STAGE3_ABORT_RATE_GATE
            and trial_count > trial_count_gate)


def stage4_gates_met(accuracy_aos, abort_rate, trial_count, trial_count_gate=200):
    """ training_protocol.md Stage 4 (formerly "Stage 3" before a new Stage 3 -- tone-side
    association -- was inserted) advancement: >70% correct at gamma=+-1.0 (AOS), abort rate
    <20%, >200 trials/session. Unlike Stage 2's gate (stage2_simple_gates_met -- deliberately
    partial, since Stage 2's real criterion is a deferred statistical test), this covers Stage 4's
    ENTIRE doc-specified advancement criterion; no separate human-judgment component. Caller
    tracks the "three consecutive sessions" requirement itself (same StageState-backed history
    list pattern as sessions_trial_count_history elsewhere), this only checks one session's own
    gates. accuracy_aos/abort_rate must already exclude warmup/repeat trials (see
    stage4_clicks_direction.py's own trial classification) -- this function trusts its inputs,
    it doesn't re-derive them from raw trial data. """
    return (accuracy_aos > STAGE4_ACCURACY_GATE and abort_rate < STAGE4_ABORT_RATE_GATE
            and trial_count > trial_count_gate)


def stage5_gates_met(response_threshold_fraction, quiescence_s, accuracy_aos, abort_rate,
                      trial_count, trial_count_gate=200):
    """ training_protocol.md Stage 5 (formerly "Stage 4" before a new Stage 3 was inserted)
    advancement: both staircases at final values (response threshold >=
    STAGE5_RESPONSE_THRESHOLD_FINAL_FRACTION of edge azimuth, quiescence at QUIESCENCE_CEILING_S),
    >70% correct at gamma=+-1.0, abort rate <25%, >=200 trials/session. Caller tracks the "two
    consecutive sessions" requirement itself.

    NOTE: since quiescence now starts staircasing as early as Stage 2 (see QuiescenceStaircase's
    own docstring) rather than being introduced fresh at Stage 5, `quiescence_at_final` may already
    be trivially True on Stage 5's very first session if Stage 2/3/4 already pushed it to
    QUIESCENCE_CEILING_S. This is intended, not a bug -- Stage 5's own job in that case is just the
    response-threshold staircase. """
    response_at_final = response_threshold_fraction >= STAGE5_RESPONSE_THRESHOLD_FINAL_FRACTION
    quiescence_at_final = quiescence_s >= QUIESCENCE_CEILING_S
    return (response_at_final and quiescence_at_final and accuracy_aos > STAGE5_ACCURACY_GATE
            and abort_rate < STAGE5_ABORT_RATE_GATE and trial_count >= trial_count_gate)


def select_stage5_staircase_to_tighten(response_threshold_fraction, quiescence_s, last_advanced):
    """ Tag B (training_protocol.md Revision 2 Appendix A): at most ONE of Stage 5's two
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
    response_remaining = (max(0.0, STAGE5_RESPONSE_THRESHOLD_FINAL_FRACTION
                               - response_threshold_fraction) / STAGE5_RESPONSE_THRESHOLD_FINAL_FRACTION)
    quiescence_span = QUIESCENCE_CEILING_S - QUIESCENCE_FLOOR_S
    quiescence_remaining = max(0.0, QUIESCENCE_CEILING_S - quiescence_s) / quiescence_span

    # A genuine tie between the two normalized "remaining" fractions should fall through to the
    # alternation tie-break below, but comparing floats for EXACT equality is fragile -- it only
    # happened to work for the floor/ceiling pair this was originally tuned against (0.1/0.5,
    # which cancels out to bit-exact IEEE754 values here) and silently breaks for other pairs (e.g.
    # 0.1/0.25) that don't cancel exactly. isclose() treats a near-enough tie as a real tie
    # regardless of which floor/ceiling values are in play.
    if math.isclose(response_remaining, quiescence_remaining, rel_tol=1e-9, abs_tol=1e-9):
        return 'quiescence' if last_advanced == 'response' else 'response'
    if response_remaining > quiescence_remaining:
        return 'response'
    return 'quiescence'
