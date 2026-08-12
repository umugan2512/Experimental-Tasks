# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Trial-lookback corrections from mouse_auditory_accumulation_paradigm.md's SS7/SS8 -- choice
side-bias handling, performance handling (remedial-easy block), and disengagement handling
(circuit-breaker). Pure Python, no Bpod/hardware dependency (same convention as click_train.py/
click_train_v2.py) so every mechanism here is independently unit-testable offline -- see
validate_trial_scheduler.py.

Scoped to exactly these three mechanisms. Deliberately does NOT implement the warm-up block (a
fixed first-N-trials prelude, not lookback/performance-driven) and is NOT wired into any task
script -- click_train_v2.py's draw_side()/draw_difficulty() are untouched and still what every
existing (already-tested) task uses. This module is additive, ready to be wired into a future task
without touching anything that already works.

Side-bias handling (SS8 steps 1-4): the rewarded side for the next trial is drawn from a
probability p_right computed from recency-weighted left/right error rates (half-Gaussian kernel,
most recent trial weighted highest), so the side the animal is worse at gets over-presented.
compute_p_right() clips sqrt(e_right)/sqrt(e_left) to P_RIGHT_CLIP *before* combining them -- this
is what bounds p_right itself into the same range (clipping both terms of a ratio-sum-normalized
combination to [a,b] bounds the combination to [a,b] too), not a separate clip on the output.

Performance handling (SS8's remedial-easy block): a rolling-accuracy detector, implemented as a
small *sticky* state machine (PerformanceMonitor) -- once a remedial block triggers it stays active
until a genuine recovery window is met, not re-evaluated fresh every trial (a single good trial
right after triggering must not immediately clear it, and a single bad trial deep in a recovery
window must not immediately re-trigger it).

Disengagement handling (SS7's circuit-breaker): should_stop_session() is a pure query over
TrialHistory -- true on a long-enough consecutive no-response run, or a high no-response rate over
a recent window. "No-response" here means SS6's specific sense (quiescence held, no lick/turn within
the response window) -- distinct from an abort (quiescence broken mid-trial), which does not count
toward this.
"""
import math

# --- trial history -----------------------------------------------------------------------------


class TrialHistory(object):
    """ Append-only record of every trial, queried by every mechanism below. Trials are stored in
    the order recorded; `included_trials()`/`recent()` return them most-recent-first, since every
    lookback calculation here is naturally expressed as "the last N trials of interest". """

    def __init__(self):
        self._trials = []

    def record(self, difficulty, side, trial_type, included, abort, response, correct, rt,
               consumption_licked=None):
        """
        :param str difficulty: this trial's difficulty level
        :param str side: 'R' or 'L' -- the rewarded (higher-evidence) side this trial
        :param str trial_type: 'main' or 'remedial_easy' (warm-up is out of scope here)
        :param bool included: a genuinely valid trial per SS7 ("initiated + responded") --
            `not abort and response is not None`. Independent of trial_type; callers combine both
            at analysis time, this module doesn't.
        :param bool abort: quiescence broken mid-trial (SS6's abort, not a response)
        :param response: 'R'/'L' if a choice was made, else None (no-response/timeout)
        :param correct: bool if a choice was made, else None
        :param rt: reaction time in seconds if a choice was made, else None
        :param consumption_licked: True/False if the trial reached a consumption window (either
            outcome) and licked or didn't during it, else None if it never reached one
            (no-response/abort)
        """
        self._trials.append({
            'difficulty': difficulty, 'side': side, 'trial_type': trial_type,
            'included': included, 'abort': abort, 'response': response,
            'correct': correct, 'rt': rt, 'consumption_licked': consumption_licked,
        })

    def __len__(self):
        return len(self._trials)

    def recent(self, n_lookback=None):
        """ All trials, most-recent-first, optionally truncated to the last n_lookback. """
        trials = list(reversed(self._trials))
        return trials if n_lookback is None else trials[:n_lookback]

    def included_trials(self, n_lookback=None, side=None):
        """ Included trials only, most-recent-first, optionally truncated to the last n_lookback
        of that filtered set (NOT the last n_lookback trials overall -- e.g. side='R' with
        n_lookback=40 means the last 40 included right-side trials, however far back that reaches,
        matching SS8's "last 40 right / left trials" phrasing). """
        trials = [t for t in reversed(self._trials) if t['included']]
        if side is not None:
            trials = [t for t in trials if t['side'] == side]
        return trials if n_lookback is None else trials[:n_lookback]

    def _consecutive_trailing_count(self, predicate):
        """ Current run length of trailing trials matching predicate(trial) -- stops counting at
        the first trial going backwards where predicate is False. Shared by
        consecutive_no_response()/consecutive_no_consumption_lick(), which differ only in
        predicate. """
        count = 0
        for t in reversed(self._trials):
            if predicate(t):
                count += 1
            else:
                break
        return count

    def _rate(self, n_lookback, predicate):
        """ Fraction of the last n_lookback trials (overall, not included-only) matching
        predicate(trial). Shared by no_response_rate()/no_consumption_lick_rate(), which differ
        only in predicate. """
        trials = self.recent(n_lookback)
        if not trials:
            return 0.0
        return sum(1 for t in trials if predicate(t)) / len(trials)

    def consecutive_no_response(self):
        """ Current run length of trailing no-response trials (response is None, not an abort) --
        stops counting at the first abort or responded trial encountered going backwards. """
        return self._consecutive_trailing_count(lambda t: t['response'] is None and not t['abort'])

    def no_response_rate(self, n_lookback):
        """ Fraction of the last n_lookback trials (overall, not included-only) that were
        no-response (response is None, not an abort). """
        return self._rate(n_lookback, lambda t: t['response'] is None and not t['abort'])

    def consecutive_no_consumption_lick(self):
        """ Current run length of trailing trials where the animal reached a consumption window
        but never licked during it (consumption_licked is False, not None) -- stops counting at
        the first trial where it licked (True) or never reached a consumption window at all (None,
        e.g. no-response/abort), same exclusion pattern as consecutive_no_response()'s own
        abort-stops-the-run behavior. """
        return self._consecutive_trailing_count(lambda t: t['consumption_licked'] is False)

    def no_consumption_lick_rate(self, n_lookback):
        """ Fraction of the last n_lookback trials (overall) where the animal reached a
        consumption window but never licked during it (consumption_licked is False). """
        return self._rate(n_lookback, lambda t: t['consumption_licked'] is False)


def _half_gaussian_weights(n, sigma):
    """ weight(i) = exp(-i^2 / (2*sigma^2)) for i=0 (most recent) .. n-1, unnormalized. """
    return [math.exp(-(i ** 2) / (2.0 * sigma ** 2)) for i in range(n)]


def rolling_accuracy(history, n_lookback):
    """ Flat (non-Gaussian) accuracy over the last n_lookback INCLUDED trials -- distinct from the
    side-bias calc's explicit half-Gaussian weighting; the doc just says "rolling N-trial accuracy"
    for the remedial-easy trigger/recovery. Returns None if there are no included trials yet
    (callers should treat that as "not enough data to evaluate", not as 0% accuracy). """
    trials = history.included_trials(n_lookback=n_lookback)
    if not trials:
        return None
    return sum(1.0 if t['correct'] else 0.0 for t in trials) / len(trials)


# --- 1a. choice side-bias handling (SS8 steps 1-4) -----------------------------------------------

SIDE_BIAS_LOOKBACK_N = 40        # last N included trials of that side considered
SIDE_BIAS_SIGMA = 20             # half-Gaussian recency weight, in trials-ago
P_RIGHT_CLIP = (0.15, 0.85)      # clip bound, applied to sqrt(e_R)/sqrt(e_L) directly (see below)
TARGET_TRACK_LOOKBACK_N = 60
TARGET_TRACK_SIGMA = 60
SIDE_BIAS_PRIOR_WEIGHT = 2.0     # ~2 virtual neutral (0.5) pseudo-trials blended into
TARGET_TRACK_PRIOR_WEIGHT = 2.0 # side_error_fraction()/recency_weighted_right_fraction(), so both
                                 # settle in gradually instead of pinning to an extreme (0 or 1) on
                                 # the first trial or two of real history -- with zero real trials
                                 # the result is still exactly 0.5 (prior_weight*0.5 / prior_weight),
                                 # matching the old hard-cutover behavior; the prior's influence
                                 # fades as real weighted history accumulates and dominates it.
                                 # Flagged/tunable.


def _half_gaussian_weighted_fraction(trials, sigma, prior_weight, value_fn):
    """ Half-Gaussian-recency-weighted mean of value_fn(trial) over trials, blended with
    prior_weight virtual neutral (0.5) pseudo-trials -- shared by side_error_fraction() and
    recency_weighted_right_fraction(), which differ only in trial selection and value_fn. """
    weights = _half_gaussian_weights(len(trials), sigma)
    values = [value_fn(t) for t in trials]
    weighted_sum = sum(w * v for w, v in zip(weights, values)) + prior_weight * 0.5
    weight_total = sum(weights) + prior_weight
    return weighted_sum / weight_total


def side_error_fraction(history, side, n_lookback=SIDE_BIAS_LOOKBACK_N, sigma=SIDE_BIAS_SIGMA,
                         prior_weight=SIDE_BIAS_PRIOR_WEIGHT):
    """ e_R or e_L: among the last n_lookback included trials with this rewarded side, the
    half-Gaussian-recency-weighted error fraction, blended with prior_weight virtual neutral (0.5)
    pseudo-trials so it settles in gradually rather than pinning to 0/1 on sparse early history --
    see SIDE_BIAS_PRIOR_WEIGHT. """
    trials = history.included_trials(n_lookback=n_lookback, side=side)
    return _half_gaussian_weighted_fraction(trials, sigma, prior_weight,
                                             lambda t: 0.0 if t['correct'] else 1.0)


def compute_p_right(e_right, e_left, clip=P_RIGHT_CLIP):
    """ sqrt(e_right)/sqrt(e_left), each clipped to `clip` BEFORE combining -- this is what bounds
    p_right itself into `clip` too (both terms of a ratio-sum-normalized combination bounded to
    [a,b] bounds the combination to [a,b] as well -- re-checked numerically in
    validate_trial_scheduler.py's Section 1, not just asserted here). """
    lo, hi = clip
    sqrt_r = min(max(math.sqrt(e_right), lo), hi)
    sqrt_l = min(max(math.sqrt(e_left), lo), hi)
    return sqrt_r / (sqrt_r + sqrt_l)


def recency_weighted_right_fraction(history, n_lookback=TARGET_TRACK_LOOKBACK_N,
                                     sigma=TARGET_TRACK_SIGMA, prior_weight=TARGET_TRACK_PRIOR_WEIGHT):
    """ The 'recent empirical right-fraction' SS8 step 4 tracks p_right against -- same
    half-Gaussian recency-weighting convention as side_error_fraction, over included trials
    regardless of side/outcome, blended with prior_weight virtual neutral (0.5) pseudo-trials so it
    settles in gradually rather than pinning to 0/1 on sparse early history -- see
    TARGET_TRACK_PRIOR_WEIGHT. """
    trials = history.included_trials(n_lookback=n_lookback)
    return _half_gaussian_weighted_fraction(trials, sigma, prior_weight,
                                             lambda t: 1.0 if t['side'] == 'R' else 0.0)


def draw_side_debiased(rng, p_right, recent_right_fraction):
    """ SS8 step 4's track-the-target logic: if the recent empirical right-fraction has already
    overshot the p_right target, pull back toward left by drawing right with only half of
    p_right's probability; if it's undershot, push toward right by drawing with half of
    (1+p_right). Returns 'R' or 'L'.

    Exact ties (recent_right_fraction == p_right) are special-cased to a neutral prob_right=0.5,
    not left to fall through to the >/else branches -- without this, an exact tie always landed in
    the "push right" else branch (prob_right=0.75, not 0.5), which happens on EVERY session's
    first trial (both values are 0.5, the neutral prior, before any real history exists),
    systematically skewing trial 1 toward 'R' rather than a fair coin flip. Confirmed against a
    real session's log before fixing -- not a hypothetical. Ties essentially never recur once real
    trial history exists (both quantities are continuously-varying independent computations by
    then), so this only changes session-start behavior. """
    if recent_right_fraction == p_right:
        prob_right = 0.5
    elif recent_right_fraction > p_right:
        prob_right = 0.5 * p_right
    else:
        prob_right = 0.5 * (1.0 + p_right)
    return 'R' if rng.uniform(0.0, 1.0) < prob_right else 'L'


MAX_SAME_SIDE_RUN = 3      # after this many consecutive same-side draws, the next is forced to
                            # switch
MAX_ALTERNATION_RUN = 3    # after this many consecutive strict alternations, the next is forced
                            # to repeat -- both catch a mouse learning to automate off the raw
                            # TRIAL_SIDE sequence itself (constant-repeat OR strict-alternation are
                            # equally predictable), not just the probabilistic debiasing target.
                            # Flagged/tunable -- no doc guidance for this specific mechanism, picked
                            # to match common anti-perseveration convention in 2AFC rodent tasks.


def _trailing_run_length(recent_sides, candidate):
    """ How many leading entries of recent_sides (most-recent-first) equal candidate. """
    count = 0
    for s in recent_sides:
        if s == candidate:
            count += 1
        else:
            break
    return count


def _extends_alternation(recent_sides, candidate, max_run):
    """ True if the leading max_run entries of recent_sides (most-recent-first) already strictly
    alternate AND candidate would continue that alternation one trial further. """
    if len(recent_sides) < max_run:
        return False
    window = recent_sides[:max_run]
    alternates = all(window[i] != window[i + 1] for i in range(len(window) - 1))
    return alternates and candidate != window[0]


def draw_side_debiased_capped(rng, p_right, recent_right_fraction, history,
                               max_same_run=MAX_SAME_SIDE_RUN, max_alt_run=MAX_ALTERNATION_RUN):
    """ draw_side_debiased(), then override with the flip if honoring it would extend either a
    same-side run or a strict-alternation run past its cap. The two violation checks are mutually
    exclusive by construction (one requires a constant trailing window, the other a strictly-
    alternating one), and the correction for either is always the same action -- the side that a
    same-run violation is repeating is exactly the side an alternation-run violation would
    otherwise skip -- so a single check-and-flip is sufficient; no need to re-check after flipping
    (confirmed over many random sessions in validate_trial_scheduler.py, not just asserted here). """
    natural = draw_side_debiased(rng, p_right, recent_right_fraction)
    recent_sides = [t['side'] for t in history.recent(max(max_same_run, max_alt_run))]
    if (_trailing_run_length(recent_sides, natural) >= max_same_run or
            _extends_alternation(recent_sides, natural, max_alt_run)):
        return 'L' if natural == 'R' else 'R'
    return natural


# --- 1b. performance handling -- remedial-easy block (SS8) ---------------------------------------

REMEDIAL_TRIGGER_ACCURACY = 0.60
REMEDIAL_TRIGGER_LOOKBACK_N = 40        # flat rolling window -- doc just says "rolling 40-trial
                                         # accuracy", unlike the side-bias calc's explicit
                                         # "half-Gaussian" language
REMEDIAL_RECOVER_ACCURACY = 0.75
REMEDIAL_RECOVER_WINDOW_N = 12          # doc says "10-15" -- midpoint, flagged, easy to retune
REMEDIAL_DEBOUNCE_N_TRIALS = 15         # trials after a recovery during which the trigger check is
                                         # suppressed -- doc's own "10-15" upper bound, reused here
                                         # since it isn't specified separately. Without this, the
                                         # TRIGGER window (40 trials) and RECOVER window (12 trials)
                                         # clear at very different rates: right after a recovery,
                                         # the 12-window looks clean but the wider 40-window still
                                         # mostly reflects the just-ended bad patch, so it can
                                         # immediately re-trigger for a few trials before the wider
                                         # window also clears -- observed directly in
                                         # validate_trial_scheduler.py's Section 5d before this was
                                         # added (rapid trigger/recover flicker right after each
                                         # recovery). Flagged, easy to retune.


class PerformanceMonitor(object):
    """ Wraps an EXTERNALLY-OWNED TrialHistory (shared with the side-bias functions and
    should_stop_session() in a real task, rather than each mechanism keeping its own independent
    log) and tracks whether a remedial-easy block is currently active. Sticky: once triggered, stays
    in remedial_easy until REMEDIAL_RECOVER_WINDOW_N included trials genuinely average >=
    REMEDIAL_RECOVER_ACCURACY -- the trigger check is only evaluated while NOT already in remedial,
    and the recovery check only while already in remedial, so a single good/bad trial can't flip the
    state on its own mid-block. After a recovery, the trigger check is further suppressed for
    REMEDIAL_DEBOUNCE_N_TRIALS trials (see that constant's docstring) so the wider, slower-clearing
    trigger window can't immediately re-fire on stale data from the block that was just exited. """

    def __init__(self, history):
        self.history = history
        self.in_remedial = False
        self._debounce_remaining = 0

    def next_trial_type(self):
        return 'remedial_easy' if self.in_remedial else 'main'

    def next_difficulty(self, rng, trial_type, full_grid_draw_fn):
        """ trial_type=='remedial_easy' -> always 'AOS' (fully one-sided click train -- the
        maximally easy, unambiguous stimulus, not a graded level); trial_type=='main' ->
        full_grid_draw_fn(rng) (caller passes e.g. click_train_v2.draw_difficulty, so this module
        never needs to import a specific grid itself). """
        if trial_type == 'remedial_easy':
            return 'AOS'
        return full_grid_draw_fn(rng)

    def record_outcome(self, difficulty, side, trial_type, abort, response, correct, rt,
                        consumption_licked=None):
        included = (not abort) and (response is not None)
        self.history.record(difficulty, side, trial_type, included, abort, response, correct, rt,
                             consumption_licked=consumption_licked)

        if not self.in_remedial:
            if self._debounce_remaining > 0:
                self._debounce_remaining -= 1
                return
            acc = rolling_accuracy(self.history, REMEDIAL_TRIGGER_LOOKBACK_N)
            has_full_window = len(self.history.included_trials()) >= REMEDIAL_TRIGGER_LOOKBACK_N
            if has_full_window and acc is not None and acc < REMEDIAL_TRIGGER_ACCURACY:
                self.in_remedial = True
        else:
            acc = rolling_accuracy(self.history, REMEDIAL_RECOVER_WINDOW_N)
            has_full_window = len(self.history.included_trials()) >= REMEDIAL_RECOVER_WINDOW_N
            if has_full_window and acc is not None and acc >= REMEDIAL_RECOVER_ACCURACY:
                self.in_remedial = False
                self._debounce_remaining = REMEDIAL_DEBOUNCE_N_TRIALS


# --- 1c. disengagement handling -- circuit-breaker (SS7) -----------------------------------------

CIRCUIT_BREAKER_CONSECUTIVE_NORESPONSE = 6   # doc says "5-8" -- midpoint, flagged
CIRCUIT_BREAKER_RATE_THRESHOLD = 0.5
CIRCUIT_BREAKER_RATE_WINDOW_N = 20

# Second, independent disengagement signal: the animal keeps responding to trials but stops
# licking/checking during its consumption window (reward or error). Same two-shape check
# (consecutive-run OR rate) as the response-based signal above, reusing its thresholds as a
# starting point -- flagged/tunable, same convention as every other threshold in this module.
CIRCUIT_BREAKER_CONSECUTIVE_NO_CONSUMPTION_LICK = 6
CIRCUIT_BREAKER_CONSUMPTION_LICK_RATE_THRESHOLD = 0.5
CIRCUIT_BREAKER_CONSUMPTION_LICK_RATE_WINDOW_N = 20


def should_stop_session(history):
    """ True if the current trailing no-response run is >= CIRCUIT_BREAKER_CONSECUTIVE_NORESPONSE,
    or the no-response rate over the last CIRCUIT_BREAKER_RATE_WINDOW_N trials exceeds
    CIRCUIT_BREAKER_RATE_THRESHOLD -- or the equivalent pair of conditions on
    "reached consumption but didn't lick" (see CIRCUIT_BREAKER_CONSECUTIVE_NO_CONSUMPTION_LICK/
    CIRCUIT_BREAKER_CONSUMPTION_LICK_RATE_THRESHOLD), an independent disengagement signal for a
    mouse that keeps choosing but stops checking for/consuming the outcome.

    Each rate check is gated on having seen at least a full window's worth of trials
    (len(history) >= its own *_WINDOW_N) before it's trusted -- without this, a rate computed over
    however few trials exist so far (e.g. 1/1 = 100%) can trip the breaker on pure early-session
    noise. The consecutive-run checks don't need this: a run of N consecutive events can't exist
    with fewer than N trials in the first place. """
    if history.consecutive_no_response() >= CIRCUIT_BREAKER_CONSECUTIVE_NORESPONSE:
        return True
    if (len(history) >= CIRCUIT_BREAKER_RATE_WINDOW_N and
            history.no_response_rate(CIRCUIT_BREAKER_RATE_WINDOW_N) > CIRCUIT_BREAKER_RATE_THRESHOLD):
        return True
    if (history.consecutive_no_consumption_lick() >=
            CIRCUIT_BREAKER_CONSECUTIVE_NO_CONSUMPTION_LICK):
        return True
    if (len(history) >= CIRCUIT_BREAKER_CONSUMPTION_LICK_RATE_WINDOW_N and
            history.no_consumption_lick_rate(CIRCUIT_BREAKER_CONSUMPTION_LICK_RATE_WINDOW_N) >
            CIRCUIT_BREAKER_CONSUMPTION_LICK_RATE_THRESHOLD):
        return True
    return False
