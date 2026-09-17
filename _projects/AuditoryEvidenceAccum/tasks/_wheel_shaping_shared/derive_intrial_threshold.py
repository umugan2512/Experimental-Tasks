# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Offline analysis tool (no hardware, no live Bpod connection) that derives the in-trial abort
threshold for Stage 5+ from a completed Stage 4 session, per training_protocol.md Revision 2
SS1.5: "The in-trial threshold is a procedure, not a number. Derive at Stage 4, re-derive at the
start of Stage 5, and re-derive whenever the abort rate exceeds the stage ceiling." -- fixed wheel
excursions over 2.75s windows, taken at the 80th percentile.

Run manually by the experimenter after a session (this is NOT called automatically from any task
script -- Stage 5 itself doesn't exist yet in this codebase, so there is nowhere for a live task to
consume the result; the derived value is meant to be hand-entered into a future Stage 5 script's
own VAR_CUE_ABORT_THRESHOLD_DEG once that exists):

    python derive_intrial_threshold.py <session.csv> [--window-s 2.75] [--percentile 80]

Depends on session_csv_parser.py's STREAM_KEYS fix (WHEEL_POS retained as a full per-sample list,
not just the last sample per trial) -- run against a session recorded AFTER that fix landed; an
older session CSV predating it will only have one WHEEL_POS sample per trial and produce a
meaninglessly small/noisy excursion distribution.
"""
import argparse
import sys

import numpy as np

import session_csv_parser


def _session_wheel_trace(trials):
    """ Concatenates every trial's own WHEEL_POS stream (session-elapsed-time, degrees) into one
    time-sorted (t, pos) pair of arrays spanning the whole session -- excursion/drift is a
    continuous phenomenon, not scoped to trial boundaries, so this deliberately does not respect
    trial edges. Trials with no WHEEL_POS rows at all (state-machine-only, e.g. a trailing
    incomplete trial -- see session_csv_parser.real_trials()) simply contribute nothing. """
    times, positions = [], []
    for trial in trials:
        for raw in trial['vals'].get('WHEEL_POS', []):
            t_str, pos_str = raw.split(',')
            times.append(float(t_str))
            positions.append(float(pos_str))
    order = np.argsort(times)
    return np.asarray(times)[order], np.asarray(positions)[order]


def compute_window_excursions(times, positions, window_s):
    """ For each sample as a window START, the max-min wheel position (degrees) among every sample
    falling within [t, t + window_s) -- a sliding-window excursion series, one value per input
    sample (the last few samples near the end of the trace produce a short/partial window and are
    still included, same as every other window; callers wanting only full windows can filter by
    checking the window's own sample count if needed, not done here since the doc's own procedure
    doesn't distinguish partial windows).

    :return: 1-D numpy array of excursions, same length as times/positions.
    """
    n = len(times)
    excursions = np.empty(n)
    right = 0
    # Two-pointer sliding window -- times is sorted, so the window's right edge only ever advances
    # as the left edge (the loop variable) advances, making this O(n) instead of O(n^2).
    for left in range(n):
        if right < left:
            right = left
        while right < n and times[right] < times[left] + window_s:
            right += 1
        window = positions[left:right]
        excursions[left] = window.max() - window.min() if len(window) else 0.0
    return excursions


def derive_intrial_threshold(session_csv_path, window_s=2.75, percentile=80):
    """ Reads session_csv_path, returns (threshold_deg, n_samples, n_windows) -- threshold_deg is
    the `percentile`-th percentile of the sliding-window excursion distribution, the actual value
    to use as a future Stage 5's starting in-trial abort threshold. Raises ValueError if the
    session has no WHEEL_POS data at all (e.g. recorded before the STREAM_KEYS fix, or a
    zero-trial session). """
    _info, _session_vals, trials = session_csv_parser.parse_session_csv(session_csv_path)
    times, positions = _session_wheel_trace(trials)
    if len(times) < 2:
        raise ValueError(
            "No (or only one) WHEEL_POS sample found in {0} -- either an empty session, or "
            "recorded before session_csv_parser.py's STREAM_KEYS fix (which is required for this "
            "tool to see more than the last sample per trial).".format(session_csv_path))

    excursions = compute_window_excursions(times, positions, window_s)
    threshold_deg = float(np.percentile(excursions, percentile))
    return threshold_deg, len(times), len(excursions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('session_csv', help="Path to a Stage 4 session's own CSV.")
    parser.add_argument('--window-s', type=float, default=2.75,
                         help="Sliding-window length in seconds (default: 2.75, per "
                              "training_protocol.md SS1.5).")
    parser.add_argument('--percentile', type=float, default=80,
                         help="Percentile of the excursion distribution to use (default: 80).")
    args = parser.parse_args()

    threshold_deg, n_samples, n_windows = derive_intrial_threshold(
        args.session_csv, window_s=args.window_s, percentile=args.percentile)

    print("Session: {0}".format(args.session_csv))
    print("WHEEL_POS samples: {0}, sliding windows ({1}s each): {2}".format(
        n_samples, args.window_s, n_windows))
    print("Derived in-trial threshold ({0:.0f}th percentile of {1}s-window excursion): "
          "{2:.2f}deg".format(args.percentile, args.window_s, threshold_deg))
    print()
    print("Hand-enter this value as the starting VAR_CUE_ABORT_THRESHOLD_DEG for this subject's "
          "next stage.")


if __name__ == '__main__':
    sys.exit(main())
