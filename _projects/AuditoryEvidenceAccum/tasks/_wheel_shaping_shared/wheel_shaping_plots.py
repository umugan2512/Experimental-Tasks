# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Live-updating matplotlib window for the wheel-shaping stages (training_protocol.md Stages 1-2) --
scoped to what these two stages actually need, unlike poisson_clicks_test/live_plots_lookback.py's
side-bias/remedial/circuit-breaker panels, all irrelevant here (no auditory clicks, no difficulty
grid, no correct-side concept until Stage 3). Deliberately not placed under poisson_clicks_test/ or
_shared/ -- this is shared *within the wheel-shaping stages only*, same "shared within this paradigm
only, not generic enough for _shared/" placement convention CLAUDE.md documents for
poisson_clicks_test/ itself.

4-row mosaic: a movement raster (the doc's own advancement criterion is about the SHAPE of this
distribution -- bimodal, velocity-separated -- so this is the panel actually worth eyeballing), a
progress/staircase-state panel (persistent bars, same pattern as live_plots_lookback.py's
disengagement panel, now also annotated with each value's PREVIOUS session's ending value so
day-over-day movement is visible without cross-referencing), an outcome tally, (Stage 2 only) the
rolling direction ratio with its 30-70% withhold band, a reward-aligned lick raster, and a
session-wide lick timeline -- the last two mirror poisson_clicks_test/live_plots_lookback.py's own
two lick panels exactly (adapted, not imported -- see below). Stage 1 has no direction-ratio panel
(no reward-withholding exists at Stage 1 at all) -- that slot instead holds the outcome tally.

Movement raster and both lick panels use persistent artists (set_offsets()/incremental
eventplot+scatter), not clear-and-replot -- same "only touch what changed" principle CLAUDE.md
flags for any panel that grows for a whole session (confirmed elsewhere in this project: cla()+
replot every trial grows from ~0.8s to ~2.3s per redraw over 150 trials); the lick-panel persistent-
artist pattern is copied directly from live_plots_lookback.py's own already-optimized
`_redraw_reward_lick_raster()`/`_redraw_session_lick()`. The outcome tally and progress bars stay
cheap cla()+redraw, matching how even the fully-optimized live_plots_lookback.py still does this for
its own low-cardinality tally/bar panels -- not a bottleneck at that scale.

`_style_axes()`/`_capped_figsize()` below are deliberately local copies of the identically-named
helpers in Tests/tasks/poisson_clicks_test/live_plots.py, not a cross-project import -- this is the
real, ongoing project and shouldn't depend on the bench-test one (the "moving to real protocols"
point of this whole module), and both helpers are small/stable enough that the duplication cost is
lower than that coupling.
"""
import matplotlib.pyplot as plt


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
    taskbar) -- queried via a throwaway Tk root, works regardless of which matplotlib backend (Tk
    or Qt) actually drives the persistent window. Falls back to the desired size unscaled if
    screen geometry can't be determined (e.g. headless). """
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


OUTCOME_COLORS = {
    'Rewarded': 'tab:green',
    'Withheld': 'tab:orange',
    'NoMovement': 'tab:gray',
}
SIDE_COLORS = {'L': 'tab:blue', 'R': 'tab:red'}
LICK_COLOR = 'tab:cyan'
FIRST_LICK_COLOR = 'tab:pink'

# Display-only reference for the Stage 1 gain-progress bar (see _redraw_progress()) -- matches
# stage1_wheel_shaping.py's own VAR_GAIN_INITIAL_MULT (training_protocol.md's stated "~2x final").
# Not used by the actual decay logic (staircase.decay_gain()), which needs no such reference.
GAIN_INITIAL_MULT_DISPLAY_REF = 2.0


class WheelShapingPlots(object):

    def __init__(self, stage, threshold_final_deg, prev_session_values=None):
        """
        :param int stage: 1 or 2 -- controls which panels are meaningful (Stage 1 has no
            direction-ratio withholding and a fixed threshold/decaying gain instead of a staircase).
        :param float threshold_final_deg: the FINAL (Stage 3+) movement threshold, used to draw the
            raster's y-axis in real degrees and to express the progress panel's threshold bar as a
            fraction of this value.
        :param dict prev_session_values: this session's own STARTING values (i.e. where the LAST
            session left off), e.g. {'threshold_deg': 7.0, 'iti_s': 0.5, 'gain_mult': 1.8} -- the
            caller already has these loaded from StageState before this session's own trials can
            change them. Shown as an annotation on the progress panel so day-over-day movement is
            visible without cross-referencing the state file. Keys not present are simply not
            annotated (e.g. Stage 1 has no 'iti_s').
        """
        self._stage = stage
        self._threshold_final_deg = threshold_final_deg
        self._prev_session_values = prev_session_values or {}

        self._trial_idx = []
        self._magnitude_deg = []
        self._side_colors = []
        self._outcome_counts = {}
        self._direction_ratio_series = []

        self._cur_threshold_deg = None
        self._cur_gain_mult = None
        self._cur_iti_s = None
        self._session_trial_count = 0

        self._total_lick_count = 0
        self._all_lick_times = []
        self._all_first_lick_times = []
        self._reward_lick_rows = []      # list of lick-time lists (relative to reward), one per
                                          # rewarded trial so far -- mirrors live_plots.py's own
        self._lickraster_rendered_n = 0

        if stage == 2:
            # Each label's cells must form one contiguous rectangle (subplot_mosaic's own
            # requirement).
            mosaic = [['raster', 'raster'],
                      ['progress', 'direction_ratio'],
                      ['reward_licks', 'outcome'],
                      ['lick_timeline', 'lick_timeline']]
        else:
            mosaic = [['raster', 'raster'],
                      ['progress', 'outcome'],
                      ['reward_licks', 'reward_licks'],
                      ['lick_timeline', 'lick_timeline']]

        plt.ion()
        self._fig, self._axes = plt.subplot_mosaic(mosaic, figsize=_capped_figsize(11, 12))
        try:
            self._fig.canvas.manager.set_window_title(
                'Wheel shaping -- Stage {0} live plots'.format(stage))
        except Exception:
            pass

        self._did_initial_layout = False
        self._setup_raster_axes()
        self._setup_progress_axes()
        self._setup_outcome_axes()
        self._setup_reward_licks_axes()
        self._setup_lick_timeline_axes()
        if stage == 2:
            self._setup_direction_ratio_axes()
        self._redraw()

    # --- data intake -------------------------------------------------------------------------------

    def add_trial(self, side, magnitude_deg, threshold_deg, outcome, gain_mult=None, iti_s=None,
                   direction_ratio=None, lick_times_abs=None, reward_time_abs=None):
        """ Call once per trial. side='L'/'R' (whichever direction the movement was, even if
        below threshold -- use the larger-magnitude direction for a NoMovement trial); magnitude_deg
        is the signed peak displacement that trial; outcome is 'Rewarded'/'Withheld'/'NoMovement';
        threshold_deg is that trial's own (possibly staircase-updated) threshold. gain_mult (Stage 1)
        / direction_ratio (Stage 2) are stage-specific, pass whichever applies.

        lick_times_abs: this trial's own Port1In event times, in absolute session-elapsed seconds
        (same basis as WHEEL_POS logging) -- feeds the full-session lick timeline, same convention
        as live_plots.py's add_trial_licks(). reward_time_abs: this trial's own absolute
        reward-delivery time, required only when outcome == 'Rewarded' -- feeds the reward-aligned
        lick raster. """
        self._session_trial_count += 1
        self._trial_idx.append(self._session_trial_count)
        signed_mag = magnitude_deg if side == 'R' else -magnitude_deg
        self._magnitude_deg.append(signed_mag)
        self._side_colors.append(OUTCOME_COLORS.get(outcome, 'gray'))
        self._outcome_counts[outcome] = self._outcome_counts.get(outcome, 0) + 1

        self._cur_threshold_deg = threshold_deg
        self._cur_gain_mult = gain_mult
        self._cur_iti_s = iti_s
        if direction_ratio is not None:
            self._direction_ratio_series.append(direction_ratio)

        lick_times_abs = lick_times_abs or []
        self._total_lick_count += len(lick_times_abs)
        self._all_lick_times.extend(lick_times_abs)
        if lick_times_abs:
            self._all_first_lick_times.append(min(lick_times_abs))
        if outcome == 'Rewarded':
            self._reward_lick_rows.append([t - reward_time_abs for t in lick_times_abs])

        self._redraw()

    # --- raster (persistent scatter) ----------------------------------------------------------------

    def _setup_raster_axes(self):
        ax = self._axes['raster']
        self._raster_scatter = ax.scatter([], [], s=14, c=[])
        self._raster_threshold_pos = ax.axhline(0, color='k', linestyle=':', linewidth=0.8)
        self._raster_threshold_neg = ax.axhline(0, color='k', linestyle=':', linewidth=0.8)
        ax.axhline(0, color='#888888', linewidth=0.6)
        ax.set_xlabel('trial')
        ax.set_ylabel('signed peak movement (deg)')
        ax.set_title('Movement raster (watch for bimodality)')
        _style_axes(ax)

    def _redraw_raster(self, ax):
        if not self._trial_idx:
            return
        offsets = list(zip(self._trial_idx, self._magnitude_deg))
        self._raster_scatter.set_offsets(offsets)
        self._raster_scatter.set_color(self._side_colors)
        if self._cur_threshold_deg is not None:
            self._raster_threshold_pos.set_ydata([self._cur_threshold_deg] * 2)
            self._raster_threshold_neg.set_ydata([-self._cur_threshold_deg] * 2)
        ax.set_xlim(0, max(10, self._trial_idx[-1] + 1))
        y_span = max(self._threshold_final_deg, max(abs(m) for m in self._magnitude_deg)) * 1.1
        ax.set_ylim(-y_span, y_span)

    # --- progress / staircase state (persistent bars, mirrors live_plots_lookback.py's style) ------

    def _setup_progress_axes(self):
        ax = self._axes['progress']
        if self._stage == 1:
            labels = ['trials/200', 'threshold/final', 'gain (final=1.0x)']
        else:
            labels = ['trials/200', 'threshold/final', 'ITI/1.5s']
        self._progress_bars = ax.barh(labels, [0] * len(labels), color='tab:blue')
        self._progress_texts = [ax.text(0.02, i, '', va='center', fontsize=8)
                                 for i in range(len(labels))]
        ax.axvline(1.0, color='black', linestyle='--', linewidth=0.8)
        ax.set_xlim(0, 1.3)
        ax.set_title('Session progress (vs. last session, in parens)')
        _style_axes(ax)

    def _prev_session_suffix(self, key, fmt):
        """ '' if prev_session_values has no entry for key, else ' (was <fmt(value)>)' -- used to
        annotate each progress-bar's text label with where it stood at the START of this session
        (i.e. the END of the last one), so day-over-day movement is visible without opening the
        state file. """
        if key not in self._prev_session_values:
            return ''
        return ' (was {0})'.format(fmt.format(self._prev_session_values[key]))

    def _redraw_progress(self, ax):
        trial_frac = min(self._session_trial_count / 200.0, 1.3)
        threshold_frac = min((self._cur_threshold_deg or 0) / self._threshold_final_deg, 1.3)
        threshold_text = '{0:.1f}/{1:.0f}deg{2}'.format(
            self._cur_threshold_deg or 0, self._threshold_final_deg,
            self._prev_session_suffix('threshold_deg', '{0:.1f}deg'))
        if self._stage == 1:
            # Gain DECAYS toward 1.0x (unlike every other bar here, which grows toward its
            # target) -- framed as "fraction of the way decayed from the initial 2.0x down to
            # 1.0x" so the bar still fills up (and turns green) as training progresses, same
            # visual language as the other bars, instead of a raw gain_mult/1.0 ratio that would
            # show FULL/green at the *start* of the stage (furthest from converged) and empty once
            # actually converged -- backwards.
            gain_mult = self._cur_gain_mult or GAIN_INITIAL_MULT_DISPLAY_REF
            decay_progress = ((GAIN_INITIAL_MULT_DISPLAY_REF - gain_mult) /
                               (GAIN_INITIAL_MULT_DISPLAY_REF - 1.0))
            gain_frac = min(max(decay_progress, 0.0), 1.3)
            fracs = [trial_frac, threshold_frac, gain_frac]
            texts = ['{0}/200'.format(self._session_trial_count),
                     threshold_text,
                     '{0:.2f}x (target 1.0x){1}'.format(
                         gain_mult, self._prev_session_suffix('gain_mult', '{0:.2f}x'))]
        else:
            iti_frac = min((self._cur_iti_s or 0) / 1.5, 1.3)
            fracs = [trial_frac, threshold_frac, iti_frac]
            texts = ['{0}/200'.format(self._session_trial_count),
                     threshold_text,
                     '{0:.2f}/1.50s{1}'.format(
                         self._cur_iti_s or 0, self._prev_session_suffix('iti_s', '{0:.2f}s'))]
        for i, (bar, text, frac, label) in enumerate(
                zip(self._progress_bars, self._progress_texts, fracs, texts)):
            bar.set_width(frac)
            bar.set_color('tab:green' if frac >= 1.0 else 'tab:blue')
            text.set_position((frac + 0.02, i))
            text.set_text(label)

    # --- outcome tally (cheap cla()+bar, matches existing convention for low-cardinality tallies) --

    def _setup_outcome_axes(self):
        _style_axes(self._axes['outcome'])

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

    # --- reward-aligned lick raster (persistent: incremental eventplot+scatter, copied from
    # live_plots_lookback.py's own already-optimized _redraw_reward_lick_raster()) ------------------

    def _setup_reward_licks_axes(self):
        ax = self._axes['reward_licks']
        ax.axvline(0.0, color='black', linestyle='-', linewidth=0.8)
        ax.set_ylim(1, -1)
        ax.set_xlabel('time relative to reward delivery (s)')
        ax.set_ylabel('rewarded trial (top = first)')
        _style_axes(ax)

    def _redraw_reward_licks(self, ax):
        n = len(self._reward_lick_rows)
        for i in range(self._lickraster_rendered_n, n):
            row = self._reward_lick_rows[i]
            if row:
                ax.eventplot([row], lineoffsets=[i], colors=LICK_COLOR, linelengths=0.8)
                ax.scatter([row[0]], [i], color=FIRST_LICK_COLOR, s=60, zorder=3,
                           edgecolors='black', linewidths=0.8, alpha=0.85)
        self._lickraster_rendered_n = n
        if n:
            ax.set_ylim(max(n, 1), -1)
        ax.set_title('Lick raster -- rewarded trials (n={0})'.format(n))

    # --- session-wide lick timeline (persistent: two scatter artists, offsets replaced each call,
    # copied from live_plots_lookback.py's own already-optimized _redraw_session_lick()) ------------

    def _setup_lick_timeline_axes(self):
        ax = self._axes['lick_timeline']
        self._lick_timeline_scatter = ax.scatter([], [], color=LICK_COLOR, s=20, zorder=2,
                                                  alpha=0.85, label='licks')
        self._lick_timeline_first_scatter = ax.scatter(
            [], [], color=FIRST_LICK_COLOR, s=90, zorder=3, edgecolors='black', linewidths=0.8,
            alpha=0.85, label='first lick of trial')
        ax.set_yticks([])
        ax.set_xlabel('session time (s)')
        ax.legend(fontsize=7, loc='upper right', frameon=False)
        _style_axes(ax)

    def _redraw_lick_timeline(self, ax):
        if self._all_lick_times:
            self._lick_timeline_scatter.set_offsets(
                list(zip(self._all_lick_times, [0] * len(self._all_lick_times))))
        if self._all_first_lick_times:
            self._lick_timeline_first_scatter.set_offsets(
                list(zip(self._all_first_lick_times, [0] * len(self._all_first_lick_times))))
        all_x = list(self._all_lick_times) + list(self._all_first_lick_times)
        if all_x:
            ax.set_xlim(min(all_x) - 1, max(all_x) + 1)
        ax.set_title('Lick events -- full session ({0} total licks logged)'.format(
            self._total_lick_count))

    # --- direction ratio (Stage 2 only, persistent line) --------------------------------------------

    def _setup_direction_ratio_axes(self):
        ax = self._axes['direction_ratio']
        self._direction_ratio_line, = ax.plot([], [], color='tab:purple')
        ax.axhline(0.30, color='gray', linestyle=':', linewidth=0.8)
        ax.axhline(0.70, color='gray', linestyle=':', linewidth=0.8)
        ax.axhline(0.5, color='#cccccc', linewidth=0.6)
        ax.set_ylim(0, 1)
        ax.set_xlabel('trial')
        ax.set_ylabel('rolling R-fraction')
        ax.set_title('Direction ratio (withhold band 0.30-0.70)')
        _style_axes(ax)

    def _redraw_direction_ratio(self, ax):
        if not self._direction_ratio_series:
            return
        x = self._trial_idx[-len(self._direction_ratio_series):]
        self._direction_ratio_line.set_data(x, self._direction_ratio_series)
        ax.set_xlim(0, max(10, x[-1] + 1))

    # --- unified redraw ------------------------------------------------------------------------------

    def _redraw(self):
        self._redraw_raster(self._axes['raster'])
        self._redraw_progress(self._axes['progress'])
        self._redraw_outcome(self._axes['outcome'])
        self._redraw_reward_licks(self._axes['reward_licks'])
        self._redraw_lick_timeline(self._axes['lick_timeline'])
        if self._stage == 2:
            self._redraw_direction_ratio(self._axes['direction_ratio'])
        if not self._did_initial_layout:
            self._fig.tight_layout()
            self._did_initial_layout = True
        plt.pause(0.001)
