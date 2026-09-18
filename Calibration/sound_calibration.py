# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""
HiFi module output-level calibration -- ties a tone's waveform amplitude to the sound pressure
level it actually produces at the animal's head position (measured with an external SPL meter), so
a task script can ask for a target dB SPL instead of guessing an amplitude. Lives alongside
`liquid_calibration.py` at the repo's own top-level `Calibration/` folder -- same "shared,
hardware-specific infrastructure, not data belonging to any single project" placement rationale.

Same underlying approach as `liquid_calibration.py`: per calibration axis, collect a handful of
(amplitude, measured_spl_db) points and fit a 2nd-order polynomial, then evaluate that polynomial
at a target SPL to get the amplitude to actually use. The HiFi driver
(`pybpod/plugins/pybpod-gui-plugin-hifi/`, package `pybpod_hifi_module`) has no native volume/gain/
attenuation command at all -- `HiFiModule`'s own API is only `load()`/`push()`/`play()`/`stop()`,
and the Bpod-relayed `HiFiCommandType` only defines `PLAY`/`STOP_ALL` -- so "gain" here can only
mean the waveform's own peak-amplitude scale (a `[0, 1]` multiplier applied to
`pure_tone()`'s already-peak-normalized output before `load()`), the same `amplitude_scale`
convention `click_train_v2.build_waveform()` already uses for its own click-level attenuation ramp.

**Calibration axis is (channel, frequency_hz), not just frequency** -- and channel and frequency
are NOT independently choosable: left is always the low-frequency channel, right always the
high-frequency channel, matching the convention `click_train_v2.py` established
(`VAR_LEFT_FREQ_HZ = 4000` &lt; `VAR_RIGHT_FREQ_HZ = 10000`, chosen to match this module's own
`CHANNEL_FREQUENCIES_HZ` below so a real calibration fit exists for the frequencies actually
played). `CHANNEL_FREQUENCIES_HZ` below is the one place this pairing is defined; every other
consumer (this module's own validation, `calibrate_sound.py`'s GUI) reads it from here rather than
each keeping its own copy.

An external amplifier/speaker volume control may also exist in the signal path -- if so, it is
assumed fixed (set once, left alone) for the whole calibration; this module has no way to control
it and doesn't attempt to log or account for it.

Storage is a plain JSON file (not MATLAB's `.mat`) -- git-diffable, same precedent as
`liquid_calibration.json`. Both the JSON and the auto-generated text report are gitignored: this
rig's own physical calibration (speakers, amplifier, mounting, room acoustics) is specific to THIS
rig's own hardware, same reasoning already applied to `liquid_calibration.json`.
"""
import json
import os
import time

import numpy as np

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sound_calibration.json')
DEFAULT_REPORT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'sound_calibration_report.txt')

MIN_POINTS_TO_FIT = 3   # same minimum liquid_calibration.py itself enforces

# The hard low=left/high=right pairing (see module docstring) -- left channel only ever tests the
# two lower frequencies, right only the two upper ones. One source of truth: both this module's
# own validation and calibrate_sound.py's GUI read from here, never a separately maintained copy.
CHANNEL_FREQUENCIES_HZ = {
    'L': [4000, 5000],
    'R': [10000, 12000],
}

# Reference SPL targets write_report() previews an amplitude for, purely for a quick human-readable
# sanity check in the report -- not used anywhere else.
_REPORT_REFERENCE_SPLS_DB = (50.0, 60.0, 70.0, 80.0)


class SoundCalibration(object):
    """
    Holds one calibration table per (channel, frequency_hz) pair (a list of
    [amplitude, mean_spl_db, n_readings, std_db] measurement rows), plus that pair's fitted
    polynomial coefficients once enough points exist. Persists to a single JSON file (see
    DEFAULT_PATH) -- every mutating method saves immediately, so a calibration session can be
    killed/restarted at any point without losing prior measurements. save() also regenerates a
    human-readable text report (see write_report()) so it's never a separate, forgettable step.
    """

    def __init__(self, path=DEFAULT_PATH, report_path=DEFAULT_REPORT_PATH):
        self.path = path
        self.report_path = report_path
        self._data = {}   # channel (str) -> {frequency_hz (int): {'table': [[amplitude,
                           #   mean_spl_db, n_readings, std_db], ...], 'coeffs': [a,b,c] or None,
                           #   'last_modified': str or None}}
        self.load()

    def load(self):
        """ Loads calibration data from self.path if it exists; otherwise starts empty (a brand
        new rig has no calibration yet -- this is the normal first-run state, not an error). """
        if not os.path.isfile(self.path):
            self._data = {}
            return
        with open(self.path, 'r') as f:
            raw = json.load(f)
        # JSON object keys are always strings -- frequencies are ints everywhere else in this
        # module (channels are already strings, 'L'/'R', so no conversion needed at that level).
        self._data = {
            channel: {int(freq_hz): entry for freq_hz, entry in freqs.items()}
            for channel, freqs in raw.items()
        }

    def save(self):
        with open(self.path, 'w') as f:
            json.dump(
                {channel: {str(freq_hz): entry for freq_hz, entry in freqs.items()}
                 for channel, freqs in self._data.items()},
                f, indent=2)
        self.write_report()

    def _validate_pair(self, channel, frequency_hz):
        valid_freqs = CHANNEL_FREQUENCIES_HZ.get(channel)
        if valid_freqs is None:
            raise ValueError("Unknown channel {0!r} -- must be one of {1}.".format(
                channel, list(CHANNEL_FREQUENCIES_HZ.keys())))
        if frequency_hz not in valid_freqs:
            raise ValueError(
                "{0}Hz is not a valid frequency for channel {1!r} -- channel {1!r} only tests "
                "{2} (left=low, right=high is a fixed pairing, see module docstring).".format(
                    frequency_hz, channel, valid_freqs))

    def _entry(self, channel, frequency_hz):
        self._validate_pair(channel, frequency_hz)
        return self._data.setdefault(channel, {}).setdefault(
            frequency_hz, {'table': [], 'coeffs': None, 'last_modified': None})

    def add_measurement(self, channel, frequency_hz, amplitude, readings_db):
        """
        Records one calibration point for (channel, frequency_hz): amplitude is the waveform peak-
        amplitude scale [0, 1] the tone was played at (already run by the caller -- this method
        only records the result), and readings_db is a list of individual dB SPL values read off
        the external meter while that tone played. Averages them here (not the caller's job, so
        every consumer of this data sees the same averaging logic) -- mean_spl_db = mean(readings_db),
        std_db = population std (0.0 if only one reading) -- appends
        [amplitude, mean_spl_db, len(readings_db), std_db] to that pair's table, and saves
        immediately. Raises ValueError (via _entry()) if (channel, frequency_hz) isn't a valid
        pairing, and separately if readings_db is empty (at least one reading is required).
        """
        if not readings_db:
            raise ValueError("At least one meter reading is required to add a point.")
        entry = self._entry(channel, frequency_hz)
        mean_db = float(np.mean(readings_db))
        std_db = float(np.std(readings_db)) if len(readings_db) > 1 else 0.0
        entry['table'].append([amplitude, mean_db, len(readings_db), std_db])
        self.save()
        return mean_db, std_db

    def remove_point(self, channel, frequency_hz, index):
        """
        Removes the measurement at index (0-based, same order as table()) from
        (channel, frequency_hz)'s table -- for discarding a bad/outlier point without needing to
        hand-edit the JSON file. Also clears any existing fit for this pair, same "don't show stale
        data" principle as liquid_calibration.py's own remove_point() -- fit() must be called again
        explicitly after removing a point. Saves immediately.
        """
        entry = self._entry(channel, frequency_hz)
        del entry['table'][index]
        entry['coeffs'] = None
        entry['last_modified'] = None
        self.save()

    def fit(self, channel, frequency_hz):
        """
        Fits (channel, frequency_hz)'s table as a 2nd-order polynomial amplitude = f(spl_db) --
        the direction useful at runtime (a task knows the target dB SPL it wants and needs the
        amplitude to produce it), the same "fit in the lookup direction, not the direction data was
        physically collected in" convention liquid_calibration.py's own fit() uses
        (duration_ms = f(volume_uL) there, not the reverse). Requires at least MIN_POINTS_TO_FIT
        points -- raises ValueError rather than silently fitting a degenerate/meaningless curve.
        """
        entry = self._entry(channel, frequency_hz)
        table = entry['table']
        if len(table) < MIN_POINTS_TO_FIT:
            raise ValueError(
                "{0} {1}Hz has only {2} measurement(s) -- need at least {3} before "
                "fitting.".format(channel, frequency_hz, len(table), MIN_POINTS_TO_FIT))

        amplitudes = np.array([row[0] for row in table], dtype=float)
        spls_db = np.array([row[1] for row in table], dtype=float)
        coeffs = np.polyfit(spls_db, amplitudes, 2)

        entry['coeffs'] = coeffs.tolist()
        entry['last_modified'] = time.strftime('%Y-%m-%d %H:%M:%S')
        self.save()
        return coeffs

    def get_amplitude_for_spl(self, channel, frequency_hz, target_spl_db):
        """
        Returns the waveform peak-amplitude scale [0, 1] that this calibration predicts will
        produce target_spl_db from (channel, frequency_hz) -- numpy.polyval(coeffs, target_spl_db).
        Raises ValueError if this pair has never been fit yet. NOT clamped to [0, 1] -- a result
        outside that range (or a target_spl_db well outside the calibrated table's own range) means
        the request is extrapolating past real data; treat that as a warning sign in the caller,
        not something to silently clip. Task scripts should generally call the module-level
        get_calibrated_amplitude() instead of this directly -- it adds a graceful hardcoded
        fallback for a box with no calibration data yet.
        """
        entry = self._entry(channel, frequency_hz)
        if entry.get('coeffs') is None:
            raise ValueError(
                "{0} {1}Hz has no calibration fit yet -- run calibrate_sound.py and fit it "
                "before requesting an amplitude.".format(channel, frequency_hz))
        return float(np.polyval(entry['coeffs'], target_spl_db))

    def table(self, channel, frequency_hz):
        """ Read-only view of (channel, frequency_hz)'s raw measurement rows, oldest first -- e.g.
        for displaying in calibrate_sound.py's own table widget. """
        return list(self._entry(channel, frequency_hz).get('table', []))

    def coeffs(self, channel, frequency_hz):
        """ (channel, frequency_hz)'s fitted polynomial coefficients, or None if not fit yet. """
        return self._entry(channel, frequency_hz).get('coeffs')

    def all_entries(self):
        """ Yields (channel, frequency_hz, entry) for every (channel, frequency_hz) pair that has
        at least one recorded point, in a fixed order ('L' then 'R', ascending frequency within
        each channel) -- the one iteration helper both write_report() and calibrate_sound.py's
        all-curves plot use, so what counts as "every calibration value" can never disagree between
        them. Pairs with an empty table are skipped (nothing to report/plot yet). """
        for channel in ('L', 'R'):
            for frequency_hz in sorted(CHANNEL_FREQUENCIES_HZ.get(channel, [])):
                entry = self._data.get(channel, {}).get(frequency_hz)
                if entry and entry.get('table'):
                    yield channel, frequency_hz, entry

    def write_report(self, path=None):
        """
        Writes a plain-text summary of every calibration point and fit across every (channel,
        frequency_hz) pair to path (defaults to self.report_path) -- called automatically at the
        end of save(), so this is always current and never a separate manual export step to
        forget. Lists each pair's raw points (amplitude, mean dB, n readings, std dB) and, if
        fitted, the polynomial coefficients + fit timestamp + the amplitude this fit predicts for
        a handful of reference SPL targets (flagged "(extrapolated)" if outside the measured SPL
        range) -- a quick human-readable sanity check without having to open the GUI or the JSON.
        """
        path = path or self.report_path
        lines = [
            "Sound calibration report -- generated {0}".format(
                time.strftime('%Y-%m-%d %H:%M:%S')),
            "=" * 78,
        ]
        any_entries = False
        for channel, frequency_hz, entry in self.all_entries():
            any_entries = True
            table = entry['table']
            lines.append("")
            lines.append("Channel {0}, {1} Hz -- {2} point(s)".format(
                channel, frequency_hz, len(table)))
            lines.append("-" * 78)
            lines.append("{0:>10s}  {1:>10s}  {2:>10s}  {3:>10s}".format(
                "amplitude", "mean dB", "n", "std dB"))
            spls_db = [row[1] for row in table]
            for amplitude, mean_db, n_readings, std_db in table:
                lines.append("{0:10.4f}  {1:10.2f}  {2:10d}  {3:10.2f}".format(
                    amplitude, mean_db, n_readings, std_db))
            if entry.get('coeffs') is not None:
                lines.append("Fit coefficients (amplitude = f(spl_db)): {0}".format(
                    ['{0:.6g}'.format(c) for c in entry['coeffs']]))
                lines.append("Last fit: {0}".format(entry['last_modified']))
                spl_min, spl_max = min(spls_db), max(spls_db)
                lines.append("Reference lookups:")
                for target_db in _REPORT_REFERENCE_SPLS_DB:
                    amp = float(np.polyval(entry['coeffs'], target_db))
                    extrapolated = target_db < spl_min or target_db > spl_max
                    lines.append("  {0:5.1f} dB SPL -> amplitude {1:.4f}{2}".format(
                        target_db, amp, " (extrapolated)" if extrapolated else ""))
            else:
                lines.append("Not fit yet (need >= {0} points).".format(MIN_POINTS_TO_FIT))
        if not any_entries:
            lines.append("")
            lines.append("No calibration points recorded yet.")

        with open(path, 'w') as f:
            f.write('\n'.join(lines) + '\n')


def get_calibrated_amplitude(target_spl_db, frequency_hz, channel='L', fallback_amplitude=1.0):
    """
    The function a task script should actually call to turn a target SPL into a waveform
    amplitude scale -- wraps SoundCalibration.get_amplitude_for_spl() with a graceful fallback: if
    no calibration data/fit exists yet for (channel, frequency_hz) on this machine, returns
    fallback_amplitude (with a printed warning) instead of raising -- so a missing/incomplete
    calibration doesn't crash a task script at startup. Loads Calibration/sound_calibration.json
    fresh on every call rather than caching/sharing one instance, same reasoning as
    get_reward_duration_s(): calibration data changes rarely, and callers only call this once, at
    startup, so the extra file read is negligible.

    Called from every HiFi-using task script's startup (via _shared/hifi_setup.py's
    compute_calibrated_amplitudes()) to turn each script's VAR_TARGET_SPL_DB into the actual
    per-channel waveform amplitude passed to build_waveform().

    :param float target_spl_db: desired sound pressure level, in dB SPL
    :param int frequency_hz: tone frequency, in Hz -- must be valid for `channel` per
        CHANNEL_FREQUENCIES_HZ
    :param str channel: 'L' or 'R'
    :param float fallback_amplitude: waveform amplitude scale [0, 1] to fall back to if no
        calibration exists
    :return: waveform peak-amplitude scale [0, 1] -- clamped here (unlike get_amplitude_for_spl(),
        which stays unclamped for display/preview purposes), since this is the value that goes
        straight into a real waveform and an out-of-range value would otherwise clip/distort it.
    """
    try:
        cal = SoundCalibration()
        amplitude = cal.get_amplitude_for_spl(channel, frequency_hz, target_spl_db)
        if amplitude > 1.0:
            print("WARNING: calibrated amplitude for {0} {1}Hz @ {2:.1f}dB SPL is {3:.4f} (>1.0) "
                  "-- clipping to 1.0 to avoid waveform clipping/distortion; the requested SPL may "
                  "not actually be reachable at this rig's current volume/gain settings.".format(
                      channel, frequency_hz, target_spl_db, amplitude), flush=True)
            amplitude = 1.0
        elif amplitude < 0.0:
            print("WARNING: calibrated amplitude for {0} {1}Hz @ {2:.1f}dB SPL is negative "
                  "({3:.4f}) -- clipping to 0.0.".format(
                      channel, frequency_hz, target_spl_db, amplitude), flush=True)
            amplitude = 0.0
        print("Amplitude from calibration: {0:.4f} for {1:.1f}dB SPL ({2} {3}Hz).".format(
            amplitude, target_spl_db, channel, frequency_hz), flush=True)
        return amplitude
    except ValueError as err:
        print("WARNING: sound calibration unavailable ({0}) -- using hardcoded fallback "
              "amplitude={1}.".format(err, fallback_amplitude), flush=True)
        return fallback_amplitude
