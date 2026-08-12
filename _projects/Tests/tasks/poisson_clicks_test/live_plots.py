# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Standalone, live-updating matplotlib plot for the hifi_singleside_test.py /
hifi_alternating_easy_test.py bench-test scripts (plan steps 2-4). One window (a click raster, a
psychometric/lick-raster/outcome row, and a full-session lick timeline), not several -- separate
from, and not touching, the existing pybpod-gui-plugin-rotaryencoder "Wheel Position" GUI plugin
window, which is its own thing opened via right-click on a session in the project tree.

Not the final GUI Session Monitor plugin -- this redraws with a full clear+replot every trial,
which is fine at bench-test scale (tens of trials) but is exactly the O(session-length) pattern
plugins/pybpod-gui-plugin-rotaryencoder/wheel_position_plot.py's docstring warns against for a
long, unattended real session. The raster/distribution design here is the prototype for that
plugin's live click-time panel; the plugin itself will need the incremental-artist treatment
(persistent Line2D/eventplot collections mutated in place, windowed to the last ~50 trials)
instead of this script's cla()-per-trial approach.
"""
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

OUTCOME_COLORS = {
    'Reward': 'tab:green',
    'NoReward': 'tab:red',
    'NoResponse': 'tab:gray',
    'Abort': 'black',
}

LEFT_COLOR = 'tab:blue'
RIGHT_COLOR = 'tab:orange'
# The bilateral onset pulse is genuinely concurrent (both frequency bands summed onto both
# channels at t=0 in click_train.py's build_waveform()) -- averaged left/right RGB rather than a
# hardcoded name so it stays correct if those base colors ever change.
ONSET_COLOR = tuple(np.mean([mcolors.to_rgb(LEFT_COLOR), mcolors.to_rgb(RIGHT_COLOR)], axis=0))


def _style_axes(ax):
    """ Despine (drop top/right borders) + soften remaining spines + light horizontal-only
    gridlines -- called after every ax.cla(), which resets spine visibility each time. """
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#444444')
    ax.spines['bottom'].set_color('#444444')
    ax.grid(axis='y', alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)


def _capped_figsize(desired_w_in, desired_h_in, margin_px=80):
    """ Scales (desired_w_in, desired_h_in) down, preserving aspect ratio, so the rendered figure
    fits within the actual screen's available area (minus a small margin for window chrome/
    taskbar) -- queried via a throwaway Tk root, which works regardless of which matplotlib
    backend (Tk or Qt) actually drives the persistent window, no new dependency. Without this, a
    hardcoded inches-based figsize can render larger than the monitor's usable area (confirmed on
    this machine), forcing a manual resize every session. Falls back to the desired size unscaled
    if screen geometry can't be determined (e.g. headless). """
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        screen_w_px, screen_h_px = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
    except Exception:
        return desired_w_in, desired_h_in
    dpi = plt.rcParams['figure.dpi']
    max_w_in = max((screen_w_px - margin_px) / dpi, 1.0)
    max_h_in = max((screen_h_px - margin_px) / dpi, 1.0)
    scale = min(1.0, max_w_in / desired_w_in, max_h_in / desired_h_in)
    return desired_w_in * scale, desired_h_in * scale


class LiveBenchPlots(object):
    """
    One figure, three rows (`Figure.subplot_mosaic`): a click raster, a
    psychometric/lick-raster/outcome row, and a full-session lick timeline.

    - Raster: one row per trial, x-axis = time within the baked waveform (onset pulse -> gap ->
      3s stimulus -> 1s delay). Left/right clicks share a row, colored differently; the onset
      pulse gets its own marker in a blended left/right color, since it plays on both channels at
      once. A colored dot past the right edge marks the trial's outcome once known.
    - Psychometric: P(chose right) vs. realized count difference (N_right - N_left), aggregated
      per (difficulty, side) bucket -- per mouse_auditory_accumulation_paradigm.md's own analysis
      plan ("P(right) vs realized (N_R-N_L)"). Marker size grows with n trials in that bucket.
      NoResponse/Abort trials have no choice and are excluded, same as the paradigm doc's "only
      valid trials enter analysis."
    - Lick raster (session-wide, rewarded trials only): one row per rewarded trial, growing for
      the whole session like the click raster does, x-axis = time relative to that trial's own
      reward delivery (t=0, the one fixed reference every rewarded trial shares). First lick per
      row gets a larger, distinctly colored marker, mirroring the click raster's own per-row dot.
    - Outcome tally: running correct/error/no-response/abort counts.
    - Lick timeline (full session, every outcome): every Port1In event logged so far, plotted at
      its own absolute session-elapsed time (same basis as WHEEL_POS: seconds since session
      start) -- a single continuously-growing panel, not reset per trial, mirroring the Wheel
      Position plugin's own continuous-elapsed-time convention. First lick of each trial gets a
      larger, differently-colored marker. Title shows a running total lick count (not a
      fabricated water-volume estimate -- there's no valve µL calibration anywhere in this
      codebase; pass a real one in if you have it and this can switch to a volume total instead).
    """

    def __init__(self, waveform_duration, click_start_offset, stim_end, onset_pulse_duration):
        self._waveform_duration = waveform_duration
        self._click_start_offset = click_start_offset
        self._stim_end = stim_end
        self._onset_marker_t = onset_pulse_duration / 2.0

        self._left_events = []
        self._right_events = []
        self._outcomes = []
        self._psych_records = []   # list of (bucket_key, realized_delta, chose_right_or_None)
        self._outcome_counts = {}

        self._reward_count = 0
        self._total_lick_count = 0
        self._reward_lick_rows = []      # list of lick-time lists, one per rewarded trial so far
        self._all_lick_times = []        # every Port1In event, absolute session-elapsed seconds
        self._all_first_lick_times = []  # one entry per trial that had >=1 lick

        plt.ion()
        self._fig, self._axes = plt.subplot_mosaic(
            [['raster', 'raster', 'raster'],
             ['psych', 'lickraster', 'outcome'],
             ['session_lick', 'session_lick', 'session_lick']],
            figsize=_capped_figsize(13, 11))
        try:
            self._fig.canvas.manager.set_window_title('Live bench plots')
        except Exception:
            pass
        self._redraw()

    def add_trial(self, difficulty, side, nominal_delta, trial_clicks, outcome=None):
        self._left_events.append(np.asarray(trial_clicks['left_times']) + self._click_start_offset)
        self._right_events.append(np.asarray(trial_clicks['right_times']) + self._click_start_offset)
        self._outcomes.append(outcome)

        if outcome == 'Reward':
            chose_right = (side == 'R')
        elif outcome == 'NoReward':
            chose_right = (side == 'L')
        else:
            chose_right = None   # NoResponse/Abort -- no choice was made, excluded from psychometric
        self._psych_records.append(((difficulty, side), trial_clicks['realized_delta'], chose_right))

        if outcome is not None:
            self._outcome_counts[outcome] = self._outcome_counts.get(outcome, 0) + 1

        self._redraw()

    def add_trial_licks(self, trial_number, outcome, lick_times_abs, reward_time_abs=None):
        """
        Call once per trial, on every outcome (Reward/NoReward/NoResponse/Abort).

        lick_times_abs: this trial's own Port1In event times, in absolute session-elapsed seconds
        (same basis as WHEEL_POS logging: seconds since session start / log_python_t0) --
        combines both the cue-phase and, if it ran, choice-phase Bpod machines. Feeds the
        full-session lick timeline (bottom panel), which keeps growing across the whole session
        rather than resetting per trial.

        reward_time_abs: this trial's own absolute reward-delivery time, same basis -- required
        only when outcome == 'Reward'. Feeds the session-wide rewarded-trial lick raster, which
        is still anchored per-row to that trial's own reward delivery (unlike the full-session
        panel above).

        No redraw here -- the caller's very next add_trial() call redraws every panel once.
        """
        self._total_lick_count += len(lick_times_abs)
        self._all_lick_times.extend(lick_times_abs)
        if lick_times_abs:
            self._all_first_lick_times.append(min(lick_times_abs))

        if outcome == 'Reward':
            self._reward_count += 1
            self._reward_lick_rows.append([t - reward_time_abs for t in lick_times_abs])

    def _redraw(self):
        self._redraw_raster(self._axes['raster'])
        self._redraw_psychometric(self._axes['psych'])
        self._redraw_reward_lick_raster(self._axes['lickraster'])
        self._redraw_outcome(self._axes['outcome'])
        self._redraw_session_lick(self._axes['session_lick'])
        self._fig.tight_layout()
        plt.pause(0.001)

    def _redraw_raster(self, ax):
        ax.cla()
        n = len(self._left_events)
        if n:
            offsets = list(range(n))
            ax.eventplot([[self._onset_marker_t]] * n, lineoffsets=offsets, colors=[ONSET_COLOR],
                         linelengths=0.8)
            ax.eventplot(self._left_events, lineoffsets=offsets, colors=LEFT_COLOR, linelengths=0.8)
            ax.eventplot(self._right_events, lineoffsets=offsets, colors=RIGHT_COLOR, linelengths=0.8)
            for i, outcome in enumerate(self._outcomes):
                if outcome is None:
                    continue
                ax.scatter([self._waveform_duration * 1.02], [i],
                           color=OUTCOME_COLORS.get(outcome, 'white'), s=18, clip_on=False, zorder=3)
        ax.axvline(self._click_start_offset, color='k', linestyle=':', linewidth=0.8)
        ax.axvline(self._stim_end, color='k', linestyle=':', linewidth=0.8)
        ax.set_xlim(0, self._waveform_duration * 1.08)
        ax.set_ylim(-1, max(n, 1))
        ax.invert_yaxis()
        ax.set_xlabel('time within trial (s) -- dotted lines: stimulus onset / stimulus end')
        ax.set_ylabel('trial (top = first)')
        ax.set_title('Live click raster (blue=left, orange=right, blend=onset pulse, dot=outcome)')
        _style_axes(ax)

    def _redraw_psychometric(self, ax):
        ax.cla()
        buckets = {}
        for bucket_key, delta, chose_right in self._psych_records:
            b = buckets.setdefault(bucket_key, {'deltas': [], 'chose_right': []})
            b['deltas'].append(delta)
            if chose_right is not None:
                b['chose_right'].append(chose_right)

        xs, ys, sizes = [], [], []
        for b in buckets.values():
            n_valid = len(b['chose_right'])
            if n_valid == 0:
                continue
            xs.append(np.mean(b['deltas']))
            ys.append(np.mean(b['chose_right']))
            sizes.append(20 + 8 * n_valid)

        if xs:
            ax.scatter(xs, ys, s=sizes, color='tab:purple', alpha=0.85, zorder=3)
        ax.axhline(0.5, color='gray', linestyle=':', linewidth=0.8)
        ax.axvline(0.0, color='gray', linestyle=':', linewidth=0.8)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel('realized count difference (N_right - N_left)')
        ax.set_ylabel('P(chose right)')
        n_valid_total = sum(1 for _, _, c in self._psych_records if c is not None)
        ax.set_title('Psychometric (N={0} valid trials)'.format(n_valid_total))
        _style_axes(ax)

    def _redraw_outcome(self, ax):
        ax.cla()
        labels = list(self._outcome_counts.keys())
        counts = [self._outcome_counts[l] for l in labels]
        colors = [OUTCOME_COLORS.get(l, 'gray') for l in labels]
        if labels:
            ax.bar(labels, counts, color=colors)
        ax.set_ylabel('trial count')
        ax.set_title('Outcome tally')
        _style_axes(ax)

    def _redraw_reward_lick_raster(self, ax):
        ax.cla()
        n = len(self._reward_lick_rows)
        if n:
            offsets = list(range(n))
            ax.eventplot(self._reward_lick_rows, lineoffsets=offsets, colors=LEFT_COLOR,
                         linelengths=0.8)
            first_lick_rows = [i for i, row in enumerate(self._reward_lick_rows) if row]
            first_licks = [self._reward_lick_rows[i][0] for i in first_lick_rows]
            if first_licks:
                ax.scatter(first_licks, first_lick_rows, color=RIGHT_COLOR, s=60, zorder=3,
                           edgecolors='black', linewidths=0.8, alpha=0.85)
        ax.axvline(0.0, color='black', linestyle='-', linewidth=0.8)
        ax.set_ylim(-1, max(n, 1))
        ax.invert_yaxis()
        ax.set_xlabel('time relative to reward delivery (s)')
        ax.set_ylabel('rewarded trial (top = first)')
        ax.set_title('Lick raster -- rewarded trials (n={0})'.format(n))
        _style_axes(ax)

    def _redraw_session_lick(self, ax):
        ax.cla()
        if self._all_lick_times:
            ax.scatter(self._all_lick_times, [0] * len(self._all_lick_times), color=LEFT_COLOR,
                       s=20, zorder=2, alpha=0.85, label='licks')
        if self._all_first_lick_times:
            ax.scatter(self._all_first_lick_times, [0] * len(self._all_first_lick_times),
                       color=RIGHT_COLOR, s=90, zorder=3, edgecolors='black', linewidths=0.8,
                       alpha=0.85, label='first lick of trial')
        ax.set_yticks([])
        ax.set_xlabel('session time (s)')
        ax.set_title('Lick events -- full session ({0} total licks logged)'.format(
            self._total_lick_count))
        ax.legend(fontsize=7, loc='upper right', frameon=False)
        _style_axes(ax)
