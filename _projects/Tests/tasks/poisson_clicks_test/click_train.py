# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Poisson click-train stimulus generator for the wheel-turn evidence-accumulation task.

Implements the difficulty grid and per-side floored-Poisson click generation described in
_projects/mouse_auditory_accumulation_paradigm.md, rescaled from the doc's total rate
lambda=8 Hz to lambda=20 Hz: gamma (log rate ratio) is held fixed per difficulty level, so
r_high/r_low/E|Delta| all scale up by the same 2.5x factor as the rate.

The ISI floor is NOT scaled proportionally from the doc's 100ms -- it's set from the actual
physical/perceptual requirement instead (click_duration + a small guard gap, so consecutive
clicks stay non-overlapping *and* audibly separated). A first pass that scaled the floor
proportionally (40ms) turned out to bite hard at lambda=20 (realized rates undershot nominal by
up to 19%, Fano factor down to 0.46), and closing that gap at a 40ms floor would have required
attempt rates far above target, pushing the process toward near-deterministic spacing. With the
smaller, physically-grounded 15ms floor used here, closing the gap needs only a small correction
(single digits of percent) -- see the rate calibration below.

"Floored" means each drawn exponential inter-click interval is *clipped* up to the floor when it
falls short (`max(isi, floor)`), not shifted by adding the floor to every interval -- clipping
only perturbs the fraction of draws that would have violated the floor. Because clipping always
lengthens (never shortens) intervals, feeding the nominal target rate directly into the
exponential sampler systematically undershoots the realized rate. This module instead solves,
per target rate, for the "attempt rate" that -- after flooring -- realizes the nominal target
rate: realized_rate(r) = 1 / (floor + (1/r)*exp(-r*floor)), a closed-form, strictly monotonic
function of r (derived from E[max(X,floor)] for X~Exponential(r), using the memoryless
property), so the inverse is a one-time, unambiguous root-find per rate. See
validate_click_train.py for the before/after comparison.

Carrier frequencies are NOT the doc's mouse-range 8/16 kHz -- they're shifted to an audible,
octave-spaced pair (1 kHz / 4 kHz, two octaves apart), matching the convention already
established in side_coding_test.py, so click rate/side can be judged by ear during development.
"""
import numpy as np
from scipy.optimize import brentq

# --- timing parameters (lambda=20 Hz grid; see module docstring for the 2.5x rescaling) --------

VAR_TOTAL_RATE_HZ = 20.0            # lambda: total click rate, both sides combined
VAR_STIM_DURATION_S = 3.0           # T: stimulus/cue duration
VAR_DELAY_DURATION_S = 1.0          # enforced silent delay after the cue (cue:delay = 3:1)

VAR_CLICK_DURATION_S = 0.008        # each click: cosine-gated tone pip -- ~8 cycles at 1kHz /
                                     # ~32 cycles at 4kHz, long enough for a discernible pitch
VAR_CLICK_RAMP_MS = 2.0             # cosine ramp at onset/offset of each click
VAR_ISI_FLOOR_S = 0.015             # click_duration (8ms) + ~7ms guard gap: the true physical
                                     # minimum for non-overlapping, audibly-separated clicks --
                                     # NOT a proportional scaling of the doc's 100ms (see docstring)
VAR_ONSET_GAP_S = VAR_ISI_FLOOR_S   # gap after the bilateral onset pulse, = ISI floor
VAR_ONSET_PULSE_DURATION_S = 0.006  # bilateral onset marker pulse

VAR_LEFT_FREQ_HZ = 1000             # left-side clicks
VAR_RIGHT_FREQ_HZ = 4000            # right-side clicks -- 2 octaves above left (audible-range
                                     # convention from side_coding_test.py, not the doc's 8/16kHz)
VAR_ONSET_RAMP_MS = 1.0             # cosine ramp for the (very short) bilateral onset pulse

# Time within the baked waveform at which the click-train proper begins (clicks' own times, as
# returned by generate_trial_clicks(), are relative to THIS point, i.e. relative to stimulus
# onset -- not to the start of the waveform buffer, which also has the onset pulse + gap first).
CLICK_START_OFFSET_S = VAR_ONSET_PULSE_DURATION_S + VAR_ONSET_GAP_S
TOTAL_WAVEFORM_DURATION_S = CLICK_START_OFFSET_S + VAR_STIM_DURATION_S + VAR_DELAY_DURATION_S

# --- difficulty grid -----------------------------------------------------------------------------
# gamma = ln(r_high/r_low), reverse-engineered from the doc's lambda=8 grid and held fixed here;
# r_high/r_low recomputed at lambda=20 (scale factor k=20/8=2.5, so E|Delta|=(r_high-r_low)*T
# also scales by k relative to the doc's original 3/7/11/15). These are the NOMINAL rates the
# task/analysis code should use for labeling and logging -- see calibrated_rate() below for the
# (higher) rate actually fed into the generator to hit these nominal rates after flooring.

_GAMMA_BY_LEVEL = {
    'E3': 0.2513,
    'E7': 0.6009,
    'E11': 0.9903,
    'E15': 1.4663,
}


def _rates_for_gamma(gamma, total_rate):
    r_high = total_rate * np.exp(gamma) / (1 + np.exp(gamma))
    r_low = total_rate - r_high
    return r_high, r_low


DIFFICULTY_GRID = {}
for _level, _gamma in _GAMMA_BY_LEVEL.items():
    _r_high, _r_low = _rates_for_gamma(_gamma, VAR_TOTAL_RATE_HZ)
    DIFFICULTY_GRID[_level] = {'r_high': _r_high, 'r_low': _r_low, 'gamma': _gamma}
DIFFICULTY_GRID['AOS'] = {'r_high': VAR_TOTAL_RATE_HZ, 'r_low': 0.0, 'gamma': float('inf')}

DIFFICULTY_ORDER = ['E3', 'E7', 'E11', 'E15', 'AOS']
DIFFICULTY_WEIGHTS = {'E3': 0.15, 'E7': 0.20, 'E11': 0.25, 'E15': 0.25, 'AOS': 0.15}


def nominal_delta(difficulty):
    grid = DIFFICULTY_GRID[difficulty]
    return (grid['r_high'] - grid['r_low']) * VAR_STIM_DURATION_S


# --- rate calibration ------------------------------------------------------------------------------
# realized_rate(r) = 1 / (floor + (1/r)*exp(-r*floor)) is strictly increasing in r (both terms of
# the denominator shrink as r grows), so for any achievable target (target_rate < 1/floor) there's
# exactly one attempt rate r solving realized_rate(r) = target_rate. Solved once per distinct
# nominal rate and cached.

_calibration_cache = {}


def _attempt_rate_for_target(target_rate, floor=VAR_ISI_FLOOR_S):
    if target_rate <= 0:
        return 0.0

    def realized_rate(r):
        return 1.0 / (floor + (1.0 / r) * np.exp(-r * floor))

    lo = target_rate                              # realized_rate(target_rate) < target_rate always
    hi = min(target_rate * 50, 0.999 / floor)     # stay below the 1/floor asymptote
    return brentq(lambda r: realized_rate(r) - target_rate, lo, hi)


def calibrated_rate(nominal_rate):
    """ The attempt rate to feed into generate_floored_poisson_train so that, after flooring, the
    realized rate matches nominal_rate. """
    if nominal_rate <= 0:
        return 0.0
    key = round(nominal_rate, 6)
    if key not in _calibration_cache:
        _calibration_cache[key] = _attempt_rate_for_target(nominal_rate)
    return _calibration_cache[key]


def draw_difficulty(rng=None):
    rng = rng or np.random
    levels = list(DIFFICULTY_WEIGHTS.keys())
    weights = list(DIFFICULTY_WEIGHTS.values())
    return rng.choice(levels, p=weights)


def draw_side(rng=None):
    """ Which side is the higher-evidence (rewarded) side this trial -- 50/50, no side-bias
    correction (that's a later, separate addition -- see the plan). """
    rng = rng or np.random
    return rng.choice(['L', 'R'])


def generate_floored_poisson_train(rate_hz, duration_s, isi_floor_s, rng=None):
    """
    One side's click times: exponential inter-click intervals at rate_hz, clipped to a minimum
    of isi_floor_s. NOTE: rate_hz here is the *attempt* rate (see calibrated_rate()), not
    necessarily the nominal target rate -- callers wanting the nominal target realized should
    pass calibrated_rate(nominal_rate).

    :return: sorted 1-D numpy array of click times (seconds from stimulus onset), all < duration_s
    """
    rng = rng or np.random
    if rate_hz <= 0:
        return np.array([])

    times = []
    t = 0.0
    while True:
        isi = max(rng.exponential(1.0 / rate_hz), isi_floor_s)
        t += isi
        if t >= duration_s:
            break
        times.append(t)
    return np.array(times)


def generate_trial_clicks(difficulty, side, rng=None):
    """
    :param str difficulty: one of DIFFICULTY_GRID's keys
    :param str side: 'L' or 'R' -- which side is the higher-evidence side this trial
    :return: dict with left_times/right_times (seconds from stimulus onset), n_left/n_right,
        realized_delta (n_right - n_left, signed), and the nominal per-side rates used.
    """
    rng = rng or np.random
    grid = DIFFICULTY_GRID[difficulty]
    r_high, r_low = grid['r_high'], grid['r_low']
    r_left, r_right = (r_low, r_high) if side == 'R' else (r_high, r_low)

    left_times = generate_floored_poisson_train(
        calibrated_rate(r_left), VAR_STIM_DURATION_S, VAR_ISI_FLOOR_S, rng)
    right_times = generate_floored_poisson_train(
        calibrated_rate(r_right), VAR_STIM_DURATION_S, VAR_ISI_FLOOR_S, rng)

    return {
        'left_times': left_times,
        'right_times': right_times,
        'n_left': len(left_times),
        'n_right': len(right_times),
        'realized_delta': len(right_times) - len(left_times),
        'nominal_r_left': r_left,
        'nominal_r_right': r_right,
    }


def build_waveform(trial_clicks, sampling_rate):
    """
    Assemble the actual stereo audio buffer for one trial from a generate_trial_clicks() result:
    bilateral onset pulse (both bands, both channels -- leaks no side) + onset gap (silence) +
    per-click cosine-gated tone pips placed at their generated times + silent padding for the
    enforced delay. One buffer, one PLAY trigger per trial (see module/repo CLAUDE.md notes on
    why a custom Bpod-module output must not be retriggered within a trial).

    :param dict trial_clicks: a generate_trial_clicks() result
    :param int sampling_rate: HiFi module sampling rate (Hz)
    :return: (left, right) -- equal-length 1-D numpy arrays, each TOTAL_WAVEFORM_DURATION_S long
    """
    # Imported lazily so click_train.py stays importable (e.g. for validate_click_train.py) on a
    # machine without the HiFi plugin package installed.
    from pybpod_hifi_module.utils.generate_sound import pure_tone

    n_total = int(round(TOTAL_WAVEFORM_DURATION_S * sampling_rate))
    left = np.zeros(n_total)
    right = np.zeros(n_total)

    def _place(buf, t_offset, tone):
        start = int(round(t_offset * sampling_rate))
        end = min(start + len(tone), len(buf))
        buf[start:end] += tone[:end - start]

    # bilateral onset pulse: both bands summed onto both channels, at t=0 -- side-neutral marker.
    # Each tone is individually peak-normalized to 1 by pure_tone(), so halve both before summing
    # -- otherwise the sum can peak past +/-1 (clipping/distortion) wherever the two sines
    # constructively align.
    onset_low = pure_tone(VAR_ONSET_PULSE_DURATION_S, VAR_LEFT_FREQ_HZ, sampling_rate,
                           ramp_ms=VAR_ONSET_RAMP_MS)
    onset_high = pure_tone(VAR_ONSET_PULSE_DURATION_S, VAR_RIGHT_FREQ_HZ, sampling_rate,
                            ramp_ms=VAR_ONSET_RAMP_MS)
    onset_pulse = 0.5 * (onset_low + onset_high)
    _place(left, 0.0, onset_pulse)
    _place(right, 0.0, onset_pulse)

    click_tone_left = pure_tone(VAR_CLICK_DURATION_S, VAR_LEFT_FREQ_HZ, sampling_rate,
                                 ramp_ms=VAR_CLICK_RAMP_MS)
    click_tone_right = pure_tone(VAR_CLICK_DURATION_S, VAR_RIGHT_FREQ_HZ, sampling_rate,
                                  ramp_ms=VAR_CLICK_RAMP_MS)

    for t in trial_clicks['left_times']:
        _place(left, CLICK_START_OFFSET_S + t, click_tone_left)
    for t in trial_clicks['right_times']:
        _place(right, CLICK_START_OFFSET_S + t, click_tone_right)

    return left, right
