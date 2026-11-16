# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Stage 3's fading scaffolds (2026-11-11 spec A7), as pure Python so the whole schedule can be
simulated offline (validate_wheel_shaping.py). Stage 3 builds the click -> side association from
trial 1 with random sides -- no side blocks: within a block the last rewarded side predicts the
next one as well as the clicks do, and that simpler predictor wins (a win-stay/lose-shift agent
scores 98% with 30-60-trial blocks, 93% with 10-20, without listening), leaving little prediction
error for the clicks to learn from.

Two temporary aids fade on a fixed schedule, one step per QUALIFYING session (>= 100 responded main
trials, abort rate < 30%, click-weight criterion met -- choice_regression.py):
    step   p_corr   click tail          lose-shift payoff without listening
     0      0.8     until crossing      ~64%
     1      0.6     until crossing      ~62%
     2      0.4     until crossing      ~58%
     3      0.2     until crossing      ~55%
     4      0       until crossing      50%
     5      0       until DOT ONSET     50%     (silent during the response, as in Stage 4)
- Correction trials: after an error, the next trial repeats the same side with probability p_corr
  (fresh clicks). Otherwise -- and after every correct trial -- the side comes from the Hu debiaser.
- Click tail: the train keeps playing after the 0.35s cue window, through delay and J1 and on
  through the response window until the crossing / response timeout / abort, so the sound is
  present at the moment of the turn and at reward.
Two consecutive sessions with click-weight z < 1 step back one step. Advance to Stage 4 after 2
consecutive qualifying sessions at step 5.

Reserve scaffold, an asymmetric threshold: if the criterion has never been met by the end of the
4th session at step 0, the correct side needs 40% of the nominal turn and the wrong side 130%. It
fades to symmetric in 4 steps on the same qualifying-session rule, before step 1 is allowed. Still
never met by session 8: flag the animal (check speaker lateralisation and level at head position).
"""

FADE_P_CORR = (0.8, 0.6, 0.4, 0.2, 0.0, 0.0)
FADE_TAIL = ('crossing', 'crossing', 'crossing', 'crossing', 'crossing', 'dot_onset')
FADE_LAST_STEP = len(FADE_P_CORR) - 1

QUALIFY_MIN_MAIN_TRIALS = 100
QUALIFY_MAX_ABORT_RATE = 0.30
LOW_Z = 1.0
LOW_Z_SESSIONS_TO_STEP_BACK = 2
ADVANCE_SESSIONS_AT_LAST_STEP = 2

RESERVE_TRIGGER_SESSIONS_AT_STEP0 = 4
FLAG_SESSIONS = 8
# Asymmetric-threshold steps 0-3 (4 = symmetric, i.e. off): fractions of the nominal response
# threshold for the correct and the wrong turn.
ASYM_CORRECT_FRACTIONS = (0.40, 0.55, 0.70, 0.85, 1.0)
ASYM_WRONG_FRACTIONS = (1.30, 1.225, 1.15, 1.075, 1.0)
ASYM_LAST_STEP = len(ASYM_CORRECT_FRACTIONS) - 1

DEFAULT_PROGRESS = {
    'fade_step': 0,
    'low_z_streak': 0,
    'sessions_at_step0': 0,      # sessions run at step 0 before the criterion was ever met
    'criterion_ever_met': False,
    'asym_step': None,           # None = off; 0..ASYM_LAST_STEP-1 while the reserve scaffold runs
    'asym_used': False,          # the reserve scaffold triggers at most once
    'step5_qualifying': 0,       # consecutive qualifying sessions at the last step
    'sessions_run': 0,
}


def p_corr(fade_step, asym_step=None):
    """ Correction-trial probability this session. While the asymmetric threshold runs, the fade
    holds at step 0. """
    return FADE_P_CORR[0 if asym_step is not None else fade_step]


def tail_mode(fade_step):
    return FADE_TAIL[fade_step]


def asym_fractions(asym_step):
    """ (correct, wrong) threshold fractions; (1.0, 1.0) when off. """
    if asym_step is None:
        return 1.0, 1.0
    return ASYM_CORRECT_FRACTIONS[asym_step], ASYM_WRONG_FRACTIONS[asym_step]


class Stage3Sides(object):
    """ Per-session side drawing: next_side() after each trial's outcome is known. Correction
    trials are NOT recorded in the debiaser as current trials (spec A7); everything else is. """

    def __init__(self, debiaser, p_corr_value, rng):
        self.debiaser = debiaser
        self.p_corr = p_corr_value
        self.rng = rng

    def next_side(self, prev_error_side):
        """ prev_error_side: the side of the previous responded trial if it was an error, else
        None. Returns (side, is_correction). """
        if prev_error_side is not None and float(self.rng.random_sample()) < self.p_corr:
            return prev_error_side, True
        return self.debiaser.next_side(self.rng), False


def session_qualifies(n_main_responded, abort_rate, criterion_met):
    return (n_main_responded >= QUALIFY_MIN_MAIN_TRIALS and abort_rate < QUALIFY_MAX_ABORT_RATE
            and bool(criterion_met))


def end_session(progress, qualifies, z_click, criterion_met):
    """ Applies one finished session to the persisted progress dict. Returns (new_progress,
    events, advance_ready, flag) -- events lists what changed, for the session-end print.
    z_click None (too few trials to fit) leaves the low-z streak unchanged. """
    p = dict(DEFAULT_PROGRESS)
    p.update(progress or {})
    events = []
    at_step0_before = p['fade_step'] == 0 and p['asym_step'] is None and not p['criterion_ever_met']
    p['sessions_run'] += 1
    if criterion_met:
        p['criterion_ever_met'] = True
    if z_click is not None:
        p['low_z_streak'] = p['low_z_streak'] + 1 if z_click < LOW_Z else 0

    if qualifies:
        if p['asym_step'] is not None:
            p['asym_step'] += 1
            if p['asym_step'] >= ASYM_LAST_STEP:
                p['asym_step'] = None
                events.append('asymmetric threshold faded out (symmetric again)')
            else:
                events.append('asymmetric threshold step -> {0}'.format(p['asym_step']))
        elif p['fade_step'] < FADE_LAST_STEP:
            p['fade_step'] += 1
            events.append('fade step -> {0}'.format(p['fade_step']))
        else:
            p['step5_qualifying'] += 1
    elif p['fade_step'] == FADE_LAST_STEP:
        p['step5_qualifying'] = 0

    if p['low_z_streak'] >= LOW_Z_SESSIONS_TO_STEP_BACK:
        if p['fade_step'] > 0:
            p['fade_step'] -= 1
            events.append('fade step back -> {0} ({1} sessions with click z < {2:g})'.format(
                p['fade_step'], LOW_Z_SESSIONS_TO_STEP_BACK, LOW_Z))
        p['low_z_streak'] = 0
        p['step5_qualifying'] = 0

    if at_step0_before and not p['criterion_ever_met']:
        p['sessions_at_step0'] += 1
        if (p['sessions_at_step0'] >= RESERVE_TRIGGER_SESSIONS_AT_STEP0 and not p['asym_used']
                and p['asym_step'] is None):
            p['asym_step'] = 0
            p['asym_used'] = True
            events.append('asymmetric threshold ON from next session (criterion not met in {0} '
                          'sessions at step 0)'.format(p['sessions_at_step0']))
    advance_ready = p['step5_qualifying'] >= ADVANCE_SESSIONS_AT_LAST_STEP
    flag = p['sessions_run'] >= FLAG_SESSIONS and not p['criterion_ever_met']
    return p, events, advance_ready, flag
