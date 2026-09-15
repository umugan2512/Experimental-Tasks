# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Liquid (water) reward valve calibration -- ties a valve's open duration to the volume it actually
delivers, so a task script can ask for a target microliter volume instead of guessing a duration.
Lives at the repo's own top level (`Calibration/`, a sibling of `_projects/` and `pybpod/`), not
nested inside any one `_projects/<ProjectName>/` -- calibration is shared, hardware-specific
infrastructure (this rig's own valve, tubing, reservoir height), not data belonging to any single
project, the same "sibling, not nested" placement rationale `_projects/_shared/` already uses one
level down.

Ports the same underlying approach Sanworks' original MATLAB Bpod software uses
(`Bpod_Gen2/Functions/Calibration/Liquid Reward/LegacyBpodLiquidCalibration.m`/
`LegacyGetValveTimes.m`, researched directly from that repo): per valve, collect a handful of
`(duration_ms, volume_uL)` measurement pairs by firing N pulses at a given duration and weighing
the total dispensed liquid, fit a 2nd-order polynomial as `duration_ms = f(volume_uL)` (not the
reverse, and not piecewise interpolation, despite the raw points being kept around), then evaluate
that polynomial at a target volume to get the duration to actually use. `pybpod-api` itself has no
equivalent -- calibration is left entirely to the experimenter upstream in real Bpod, and this
module is this lab's own port of that same idea rather than something reusable from the Python
Bpod stack.

Storage is a plain JSON file (not MATLAB's `.mat`) -- git-diffable, and consistent with this
codebase's own precedent (`wheel_shaping_state.json`) for small persisted-state files elsewhere.
The file itself is gitignored (see repo root `.gitignore`): a valve's physical calibration is
specific to THIS rig's own hardware, the same reasoning already applied to `boards/`/`subjects/`
configs being machine-specific and gitignored.
"""
import json
import os
import time

import numpy as np

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'liquid_calibration.json')

MIN_POINTS_TO_FIT = 3   # same minimum LegacyGetValveTimes.m itself enforces


class LiquidCalibration(object):
    """
    Holds one calibration table per valve ID (a list of [duration_ms, volume_uL] measurement
    pairs), plus that valve's fitted polynomial coefficients once enough points exist. Persists
    to a single JSON file (see DEFAULT_PATH) -- every mutating method saves immediately, so a
    calibration session can be killed/restarted at any point without losing prior measurements.
    """

    def __init__(self, path=DEFAULT_PATH):
        self.path = path
        self._valves = {}   # valve_id (int) -> {'table': [[duration_ms, volume_uL], ...],
                             #                    'coeffs': [a, b, c] or None, 'last_modified': str}
        self.load()

    def load(self):
        """ Loads calibration data from self.path if it exists; otherwise starts empty (a brand
        new rig/valve has no calibration yet -- this is the normal first-run state, not an error).
        """
        if not os.path.isfile(self.path):
            self._valves = {}
            return
        with open(self.path, 'r') as f:
            raw = json.load(f)
        # JSON object keys are always strings -- valve IDs are ints everywhere else in this module.
        self._valves = {int(valve_id): entry for valve_id, entry in raw.items()}

    def save(self):
        with open(self.path, 'w') as f:
            json.dump({str(k): v for k, v in self._valves.items()}, f, indent=2)

    def _entry(self, valve_id):
        return self._valves.setdefault(valve_id, {'table': [], 'coeffs': None, 'last_modified': None})

    def add_measurement(self, valve_id, duration_ms, mass_g, n_pulses):
        """
        Records one calibration point for valve_id: fires n_pulses at duration_ms were already
        run (by the caller -- this method only records the result), and mass_g is the TOTAL
        weighed liquid across all n_pulses combined. Converts to a per-pulse volume the same way
        LegacyBpodLiquidCalibration.m does -- water density assumed 1 g/mL, so
        volume_uL_per_pulse = mass_g * 1000 / n_pulses -- appends (duration_ms, volume_uL) to that
        valve's table, and saves immediately.
        """
        volume_ul = mass_g * 1000.0 / n_pulses
        entry = self._entry(valve_id)
        entry['table'].append([duration_ms, volume_ul])
        self.save()
        return volume_ul

    def remove_point(self, valve_id, index):
        """
        Removes the measurement at index (0-based, same order as table()) from valve_id's table --
        for discarding a bad/outlier point (e.g. a mistyped mass or pulse count) without needing to
        hand-edit the JSON file. Also clears any existing fit for this valve: a fit computed from
        the table BEFORE this removal no longer reflects the current data, and silently leaving a
        stale fit/coeffs in place would be misleading -- fit() must be called again explicitly
        after removing a point, same "don't show stale data" principle as everywhere else in this
        project. Saves immediately, same as add_measurement().
        """
        entry = self._entry(valve_id)
        del entry['table'][index]
        entry['coeffs'] = None
        entry['last_modified'] = None
        self.save()

    def fit(self, valve_id):
        """
        Fits valve_id's table as a 2nd-order polynomial duration_ms = f(volume_uL) -- same
        direction and order LegacyBpodLiquidCalibration.m itself uses (fit against volume, not
        against duration), via numpy.polyfit(volumes, durations_ms, 2). Requires at least
        MIN_POINTS_TO_FIT points, matching LegacyGetValveTimes.m's own minimum -- raises ValueError
        rather than silently fitting a degenerate/meaningless curve from too few points.
        """
        entry = self._entry(valve_id)
        table = entry['table']
        if len(table) < MIN_POINTS_TO_FIT:
            raise ValueError(
                "Valve {0} has only {1} measurement(s) -- need at least {2} before fitting.".format(
                    valve_id, len(table), MIN_POINTS_TO_FIT))

        durations_ms = np.array([row[0] for row in table], dtype=float)
        volumes_ul = np.array([row[1] for row in table], dtype=float)
        coeffs = np.polyfit(volumes_ul, durations_ms, 2)

        entry['coeffs'] = coeffs.tolist()
        entry['last_modified'] = time.strftime('%Y-%m-%d %H:%M:%S')
        self.save()
        return coeffs

    def get_valve_time_s(self, volume_ul, valve_id=1):
        """
        Returns the valve-open duration, in SECONDS (ready to assign straight to a state_timer),
        that this calibration predicts will deliver volume_ul from valve_id --
        numpy.polyval(coeffs, volume_ul) / 1000.0, mirroring LegacyGetValveTimes.m's own ms->s
        conversion. Raises ValueError if valve_id has never been fit (fit() must be called, and
        succeed, at least once first) -- never silently returns a guessed/placeholder duration.
        Task scripts should generally call the module-level get_reward_duration_s() instead of
        this directly -- it adds a graceful hardcoded-fallback for a box with no calibration data
        yet, so a missing/incomplete calibration doesn't crash every reward-delivering task script.
        """
        entry = self._valves.get(valve_id)
        if entry is None or entry.get('coeffs') is None:
            raise ValueError(
                "Valve {0} has no calibration fit yet -- run calibrate_liquid.py and fit it "
                "before requesting a valve time.".format(valve_id))

        duration_ms = np.polyval(entry['coeffs'], volume_ul)
        return duration_ms / 1000.0

    def table(self, valve_id):
        """ Read-only view of valve_id's raw (duration_ms, volume_uL) measurement pairs, oldest
        first -- e.g. for displaying in calibrate_liquid.py's own table widget. """
        return list(self._entry(valve_id).get('table', []))

    def coeffs(self, valve_id):
        """ valve_id's fitted polynomial coefficients, or None if it hasn't been fit yet. """
        return self._entry(valve_id).get('coeffs')


def get_reward_duration_s(target_volume_ul, valve_id=1, fallback_s=0.1):
    """
    The function a task script should actually call to turn a target reward volume into a valve
    open duration -- wraps LiquidCalibration.get_valve_time_s() with a graceful fallback: if no
    calibration data/fit exists yet for valve_id on this machine (a fresh box, or one that just
    hasn't been calibrated yet), returns fallback_s (with a printed warning) instead of raising --
    so a missing/incomplete calibration doesn't crash every reward-delivering task script at
    startup. Loads Calibration/liquid_calibration.json fresh on every call rather than trying to
    cache/share one LiquidCalibration instance across callers -- calibration data changes rarely
    (only during an actual bench-calibration session), and each task script only calls this once,
    at startup, so the extra file read is negligible.

    :param float target_volume_ul: desired reward volume, in microliters
    :param int valve_id: which valve (this rig has one, see CLAUDE.md -- "wheel-turn choice +
        single valve")
    :param float fallback_s: hardcoded duration to fall back to (seconds) if no calibration exists
    :return: valve-open duration in seconds, ready to assign straight to a state_timer
    """
    try:
        cal = LiquidCalibration()
        duration_s = cal.get_valve_time_s(target_volume_ul, valve_id)
        print("Reward duration from calibration: {0:.1f}ms for {1}uL (valve {2}).".format(
            duration_s * 1000.0, target_volume_ul, valve_id), flush=True)
        return duration_s
    except ValueError as err:
        print("WARNING: liquid calibration unavailable ({0}) -- using hardcoded fallback "
              "VAR_REWARD_DURATION={1}s.".format(err, fallback_s), flush=True)
        return fallback_s
