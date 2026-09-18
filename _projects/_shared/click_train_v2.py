# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Poisson click-train stimulus generator for the wheel-turn evidence-accumulation task -- v2 grid.

Forked from click_train.py (kept byte-identical/untouched -- see repo CLAUDE.md's "Poisson-clicks
/ wheel-turn evidence-accumulation task" section) rather than edited in place, so the original
audio-only tests (hifi_singleside_test.py/hifi_alternating_easy_test.py) keep producing exactly
what they always have. This version:
  - Shortens VAR_STIM_DURATION_S (3.0 -> 2.0s) and VAR_DELAY_DURATION_S (1.0 -> 0.75s).
  - Replaces the old gamma = ln(r_high/r_low) (log rate ratio) grid with a linear-contrast
    gamma = (r_high - r_low) / lambda formulation: r_high = lambda*(1+gamma)/2,
    r_low = lambda*(1-gamma)/2. gamma=1.0 recovers the fully one-sided AOS case
    (r_high=lambda, r_low=0), same as before -- just via a different formula.
  - New 6-level grid: G05/G12/G22/G38/G65 (gamma = 0.05/0.12/0.22/0.38/0.65) plus AOS
    (gamma=1.0) as the 6th, most-extreme level.

Everything else (floored-Poisson generation, rate calibration, build_waveform) is unchanged from
click_train.py -- see that file's docstring for the rationale behind the ISI floor, rate
calibration, and audible-range carrier frequencies, all of which still apply here unmodified.

Lives in `_projects/_shared/` (moved here from `Tests/tasks/poisson_clicks_test/`) so the real
`AuditoryEvidenceAccum` project can use it without depending on `Tests/`, the bench-test sandbox --
same "protocol-agnostic, usable by any task in any GUI project" placement already used for
`rotary_setup.py`/`dot_display.py`/`hifi_setup.py`/`camera_recorder.py`. `Tests/tasks/
poisson_clicks_test/` keeps `click_train.py` (v1, byte-identical, untouched) and every other
paradigm-specific file (`live_plots.py`, `trial_scheduler.py`, etc.) -- only this generator and its
offline validator moved.

Two additive parameters were added on top of the original (ISI floor/rate calibration/Poisson
generation math untouched, per this repo's own "don't touch stimulus generation" convention):
  - `build_waveform(..., amplitude_scale=1.0)`: an outer multiply on the returned stereo buffers,
    for a click-level attenuation ramp (e.g. AuditoryEvidenceAccum's Stage 3, which starts new
    subjects a few dB below final level for their first few sessions). Never touches `pure_tone`/
    `_place` internals.
  - `generate_trial_clicks(..., stim_duration_s=None)` / `build_waveform(..., stim_duration_s=None)`:
    both default to the module's own `VAR_STIM_DURATION_S` (2.0s) so every existing caller is
    byte-for-byte unaffected. AuditoryEvidenceAccum's Stage 3/4 pass `stim_duration_s=0.25` (per
    the training protocol's "0.25s Poisson clicks" spec) -- a real duration mismatch against this
    module's original 2.0s grid, not something the original 6-level DIFFICULTY_GRID/timing design
    anticipated, so it's a parameter rather than a second hardcoded constant.
"""
import numpy as np
from scipy.optimize import brentq

# --- timing parameters (lambda=20 Hz grid; shortened cue/delay vs. click_train.py) --------------

VAR_TOTAL_RATE_HZ = 20.0            # lambda: total click rate, both sides combined
VAR_STIM_DURATION_S = 2.0           # T: stimulus/cue duration
VAR_DELAY_DURATION_S = 0.75         # enforced silent delay after the cue

VAR_CLICK_DURATION_S = 0.008        # each click: cosine-gated tone pip -- ~8 cycles at 1kHz /
                                     # ~32 cycles at 4kHz, long enough for a discernible pitch
VAR_CLICK_RAMP_MS = 2.0             # cosine ramp at onset/offset of each click
VAR_ISI_FLOOR_S = 0.015             # click_duration (8ms) + ~7ms guard gap: the true physical
                                     # minimum for non-overlapping, audibly-separated clicks
VAR_ONSET_GAP_S = VAR_ISI_FLOOR_S   # gap after the bilateral onset pulse, = ISI floor
VAR_ONSET_PULSE_DURATION_S = 0.006  # bilateral onset marker pulse

VAR_LEFT_FREQ_HZ = 4000              # left-side clicks -- matches sound_calibration.py's own
                                      # CHANNEL_FREQUENCIES_HZ['L'] (the lower of its two calibrated
                                      # left frequencies), so a real dB-SPL calibration fit actually
                                      # exists for the frequency being played (was 1000Hz, which
                                      # sound_calibration.py never calibrated at all).
VAR_RIGHT_FREQ_HZ = 10000            # right-side clicks -- matches CHANNEL_FREQUENCIES_HZ['R']'s
                                      # lower calibrated right frequency (was 4000Hz).
VAR_ONSET_RAMP_MS = 1.0             # cosine ramp for the (very short) bilateral onset pulse

# Time within the baked waveform at which the click-train proper begins (clicks' own times, as
# returned by generate_trial_clicks(), are relative to THIS point, i.e. relative to stimulus
# onset -- not to the start of the waveform buffer, which also has the onset pulse + gap first).
CLICK_START_OFFSET_S = VAR_ONSET_PULSE_DURATION_S + VAR_ONSET_GAP_S
TOTAL_WAVEFORM_DURATION_S = CLICK_START_OFFSET_S + VAR_STIM_DURATION_S + VAR_DELAY_DURATION_S


def total_waveform_duration_s(stim_duration_s=None):
    """ TOTAL_WAVEFORM_DURATION_S generalized to a caller-supplied stim_duration_s (see module
    docstring) -- callers needing the module's own default duration can keep using the
    TOTAL_WAVEFORM_DURATION_S constant directly. """
    stim_duration_s = VAR_STIM_DURATION_S if stim_duration_s is None else stim_duration_s
    return CLICK_START_OFFSET_S + stim_duration_s + VAR_DELAY_DURATION_S


# --- difficulty grid -----------------------------------------------------------------------------
# gamma = (r_high - r_low) / lambda (linear contrast, NOT the old log rate ratio) --
# r_high = lambda*(1+gamma)/2, r_low = lambda*(1-gamma)/2. These are the NOMINAL rates the
# task/analysis code should use for labeling and logging -- see calibrated_rate() below for the
# (higher) rate actually fed into the generator to hit these nominal rates after flooring.

_GAMMA_BY_LEVEL = {
    'G05': 0.05,
    'G12': 0.12,
    'G22': 0.22,
    'G38': 0.38,
    'G65': 0.65,
}


def _rates_for_gamma(gamma, total_rate):
    r_high = total_rate * (1 + gamma) / 2.0
    r_low = total_rate * (1 - gamma) / 2.0
    return r_high, r_low


DIFFICULTY_GRID = {}
for _level, _gamma in _GAMMA_BY_LEVEL.items():
    _r_high, _r_low = _rates_for_gamma(_gamma, VAR_TOTAL_RATE_HZ)
    DIFFICULTY_GRID[_level] = {'r_high': _r_high, 'r_low': _r_low, 'gamma': _gamma}
DIFFICULTY_GRID['AOS'] = {'r_high': VAR_TOTAL_RATE_HZ, 'r_low': 0.0, 'gamma': 1.0}

DIFFICULTY_ORDER = ['G05', 'G12', 'G22', 'G38', 'G65', 'AOS']
# No weighting specified for the new grid -- defaults to equal (1/6 each). Easy to retune.
DIFFICULTY_WEIGHTS = {level: 1.0 / len(DIFFICULTY_ORDER) for level in DIFFICULTY_ORDER}


def nominal_delta(difficulty, stim_duration_s=None):
    grid = DIFFICULTY_GRID[difficulty]
    stim_duration_s = VAR_STIM_DURATION_S if stim_duration_s is None else stim_duration_s
    return (grid['r_high'] - grid['r_low']) * stim_duration_s


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


def generate_trial_clicks(difficulty, side, rng=None, stim_duration_s=None):
    """
    :param str difficulty: one of DIFFICULTY_GRID's keys
    :param str side: 'L' or 'R' -- which side is the higher-evidence side this trial
    :param float stim_duration_s: cue duration in seconds -- defaults to the module's own
        VAR_STIM_DURATION_S (2.0s). Pass an explicit value (e.g. 0.25 for AuditoryEvidenceAccum's
        Stage 3/4) to generate a shorter/longer click train without touching the module default or
        any other caller. See module docstring.
    :return: dict with left_times/right_times (seconds from stimulus onset), n_left/n_right,
        realized_delta (n_right - n_left, signed), and the nominal per-side rates used.
    """
    rng = rng or np.random
    stim_duration_s = VAR_STIM_DURATION_S if stim_duration_s is None else stim_duration_s
    grid = DIFFICULTY_GRID[difficulty]
    r_high, r_low = grid['r_high'], grid['r_low']
    r_left, r_right = (r_low, r_high) if side == 'R' else (r_high, r_low)

    left_times = generate_floored_poisson_train(
        calibrated_rate(r_left), stim_duration_s, VAR_ISI_FLOOR_S, rng)
    right_times = generate_floored_poisson_train(
        calibrated_rate(r_right), stim_duration_s, VAR_ISI_FLOOR_S, rng)

    return {
        'left_times': left_times,
        'right_times': right_times,
        'n_left': len(left_times),
        'n_right': len(right_times),
        'realized_delta': len(right_times) - len(left_times),
        'nominal_r_left': r_left,
        'nominal_r_right': r_right,
    }


def build_waveform(trial_clicks, sampling_rate, amplitude_scale_left=1.0,
                    amplitude_scale_right=1.0, stim_duration_s=None):
    """
    Assemble the actual stereo audio buffer for one trial from a generate_trial_clicks() result:
    bilateral onset pulse (both bands, both channels -- leaks no side) + onset gap (silence) +
    per-click cosine-gated tone pips placed at their generated times + silent padding for the
    enforced delay. One buffer, one PLAY trigger per trial (see module/repo CLAUDE.md notes on
    why a custom Bpod-module output must not be retriggered within a trial).

    :param dict trial_clicks: a generate_trial_clicks() result
    :param int sampling_rate: HiFi module sampling rate (Hz)
    :param float amplitude_scale_left: outer multiplier applied to the returned left buffer
        (default 1.0, i.e. unchanged) -- e.g. a sound_calibration.get_calibrated_amplitude() result
        for a target dB SPL, or that times 10**(-6/20) for an additional click-level attenuation
        ramp on top. Applied after all tone placement, never inside pure_tone()/_place(). Separate
        from amplitude_scale_right because the two channels' calibration curves generally differ
        (was a single amplitude_scale applied to both -- split for per-channel SPL calibration).
    :param float amplitude_scale_right: same as amplitude_scale_left, for the right buffer.
    :param float stim_duration_s: must match whatever stim_duration_s was passed to
        generate_trial_clicks() for this same trial_clicks (defaults to VAR_STIM_DURATION_S,
        matching generate_trial_clicks()'s own default) -- determines the buffer's total length.
    :return: (left, right) -- equal-length 1-D numpy arrays, each
        total_waveform_duration_s(stim_duration_s) long
    """
    # Imported lazily so click_train_v2.py stays importable (e.g. for validate_click_train_v2.py)
    # on a machine without the HiFi plugin package installed.
    from pybpod_hifi_module.utils.generate_sound import pure_tone

    n_total = int(round(total_waveform_duration_s(stim_duration_s) * sampling_rate))
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

    if amplitude_scale_left != 1.0:
        left = left * amplitude_scale_left
    if amplitude_scale_right != 1.0:
        right = right * amplitude_scale_right

    return left, right
