# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Live indicators for trial_scheduler.py's three lookback mechanisms (choice side-bias handling,
performance handling/remedial-easy, disengagement handling/circuit-breaker), for use by
full_protocol_lookback_test.py. live_plots.py itself stays completely untouched -- both existing
gabor tests (hifi_singleside_gabor_test.py/hifi_alternating_easy_gabor_test.py) keep using it
exactly as before, with zero risk of regression.

LookbackBenchPlots subclasses LiveBenchPlots, adding one extra mosaic row (side-bias, performance,
disengagement panels). Call add_trial_lookback() once per trial, alongside the inherited
add_trial() call -- same "no redraw here, the next add_trial() call redraws everything once"
convention add_trial_licks() already uses, so all panels stay in sync on one redraw per trial.

**Persistent-artist rendering, not clear-and-replot.** The inherited LiveBenchPlots redraws every
panel via ax.cla() + a full re-plot from the *entire* accumulated session history, every single
trial -- exactly the O(session-length)-per-tick pattern CLAUDE.md already flags as a hazard for the
Wheel Position GUI plugin. Measured directly on this module before this fix: per-trial redraw time
grew from 0.81s to 2.28s (2.82x) over 150 synthetic trials. Every _redraw_* method here (both the
5 inherited-name ones and the 3 lookback-specific ones) sets up static chrome -- spines, gridlines,
axhline/axvline markers, labels, title skeleton -- exactly ONCE (in __init__), then on every
subsequent call only touches NEW data via persistent artist handles (Line2D.set_data(),
PathCollection.set_offsets(), Rectangle.set_height()/set_width(), or -- for the two per-trial
eventplot rasters, which don't support incremental appends -- adding one small new artist per NEW
trial only, never re-touching old trials' artists). ax.cla() is never called again after initial
setup anywhere in this file. This class fully owns _redraw() (does NOT delegate to
super()._redraw()) for two reasons: it needs to call all 8 panels' own persistent-artist versions
directly (Python's normal method-dispatch would technically already route calls made from inside
super()._redraw() to this class's overrides, but relying on that indirection was found to have
caused fig.tight_layout()+plt.pause() to run TWICE per trial -- once inside the inherited
_redraw(), once again in this class's own -- an avoidable inefficiency independent of the
cla()-vs-persistent-artist issue, only caught by owning the method directly instead).
fig.tight_layout() itself is also now called only ONCE (after the very first draw), not every
trial -- it's a real, non-trivial cost (a full renderer pass to measure every text element) and
per-trial title-text-length changes don't need a full re-layout.
"""
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from live_plots import LiveBenchPlots, OUTCOME_COLORS, LEFT_COLOR, RIGHT_COLOR, ONSET_COLOR, \
    _style_axes, _capped_figsize
import trial_scheduler as ts


class LookbackBenchPlots(LiveBenchPlots):

    CHOICE_ROLLING_WINDOW = 20   # trials -- matches CIRCUIT_BREAKER_RATE_WINDOW_N's existing
                                 # convention elsewhere in this module, not independently tuned

    def __init__(self, waveform_duration, click_start_offset, stim_end, onset_pulse_duration):
        self._p_right_series = []
        self._recent_right_frac_series = []
        self._choice_series = []   # 1.0 if response=='R', 0.0 if 'L', None if no response/abort
        self._trial_type_series = []
        self._rolling_acc_series = []
        self._consecutive_noresp = 0
        self._noresp_rate = 0.0
        self._consecutive_no_lick = 0
        self._no_lick_rate = 0.0

        # Re-implemented (not super().__init__()) so the mosaic can include the extra row from the
        # start -- LiveBenchPlots.__init__ builds its own fixed 3-row mosaic and would need to be
        # duplicated here anyway to insert a 4th row before the first _redraw() call.
        self._waveform_duration = waveform_duration
        self._click_start_offset = click_start_offset
        self._stim_end = stim_end
        self._onset_marker_t = onset_pulse_duration / 2.0

        self._left_events = []
        self._right_events = []
        self._outcomes = []
        self._psych_records = []
        self._outcome_counts = {}

        self._reward_count = 0
        self._total_lick_count = 0
        self._reward_lick_rows = []
        self._all_lick_times = []
        self._all_first_lick_times = []

        plt.ion()
        self._fig, self._axes = plt.subplot_mosaic(
            [['raster', 'raster', 'raster'],
             ['psych', 'lickraster', 'outcome'],
             ['session_lick', 'session_lick', 'session_lick'],
             ['sidebias', 'performance', 'disengagement']],
            figsize=_capped_figsize(13, 14))
        try:
            self._fig.canvas.manager.set_window_title('Live bench plots (+ lookback mechanisms)')
        except Exception:
            pass

        self._did_initial_layout = False
        self._setup_raster_axes()
        self._setup_psych_axes()
        self._setup_lickraster_axes()
        self._setup_outcome_axes()
        self._setup_session_lick_axes()
        self._setup_sidebias_axes()
        self._setup_performance_axes()
        self._setup_disengagement_axes()

        self._redraw()

    def add_trial_lookback(self, p_right_target, recent_right_frac, trial_type, history, response=None):
        """
        Call once per trial, alongside the inherited add_trial(). p_right_target/recent_right_frac
        are the two values that determined THIS trial's side draw (see
        trial_scheduler.compute_p_right()/recency_weighted_right_fraction()); trial_type is
        'main'/'remedial_easy' for this trial; history is the shared TrialHistory the whole
        scheduler reads/writes, used here to pull live disengagement-indicator readouts. response
        is the side the animal actually turned to ('R'/'L'), or None for an abort/no-response --
        distinct from p_right_target/recent_right_frac, which describe what was PRESENTED/targeted,
        not what the animal actually CHOSE. No redraw here -- the caller's very next add_trial()
        call redraws every panel once.
        """
        self._p_right_series.append(p_right_target)
        self._recent_right_frac_series.append(recent_right_frac)
        self._choice_series.append({'R': 1.0, 'L': 0.0}.get(response))
        self._trial_type_series.append(trial_type)
        self._rolling_acc_series.append(ts.rolling_accuracy(history, ts.REMEDIAL_TRIGGER_LOOKBACK_N))
        self._consecutive_noresp = history.consecutive_no_response()
        self._noresp_rate = history.no_response_rate(ts.CIRCUIT_BREAKER_RATE_WINDOW_N)
        self._consecutive_no_lick = history.consecutive_no_consumption_lick()
        self._no_lick_rate = history.no_consumption_lick_rate(
            ts.CIRCUIT_BREAKER_CONSUMPTION_LICK_RATE_WINDOW_N)

    def _rolling_choice_right_fraction(self):
        """ Trailing CHOICE_ROLLING_WINDOW-trial P(chose right), among trials that had a response
        (no-response/abort trials are skipped, not counted as either side) -- the animal's own
        actual choice behavior, distinct from the presentation-side series above. """
        window = self.CHOICE_ROLLING_WINDOW
        result = []
        for i in range(len(self._choice_series)):
            chunk = [c for c in self._choice_series[max(0, i - window + 1):i + 1] if c is not None]
            result.append(sum(chunk) / len(chunk) if chunk else float('nan'))
        return result

    # --- unified redraw: owns the full sequence, does not delegate to super()._redraw() ----------

    def _redraw(self):
        self._redraw_raster(self._axes['raster'])
        self._redraw_psychometric(self._axes['psych'])
        self._redraw_reward_lick_raster(self._axes['lickraster'])
        self._redraw_outcome(self._axes['outcome'])
        self._redraw_session_lick(self._axes['session_lick'])
        self._redraw_sidebias(self._axes['sidebias'])
        self._redraw_performance(self._axes['performance'])
        self._redraw_disengagement(self._axes['disengagement'])
        if not self._did_initial_layout:
            self._fig.tight_layout()
            self._did_initial_layout = True
        plt.pause(0.001)

    # --- raster (persistent: one small eventplot/scatter added per NEW trial only) ----------------

    def _setup_raster_axes(self):
        ax = self._axes['raster']
        ax.axvline(self._click_start_offset, color='k', linestyle=':', linewidth=0.8)
        ax.axvline(self._stim_end, color='k', linestyle=':', linewidth=0.8)
        ax.set_xlim(0, self._waveform_duration * 1.08)
        ax.set_ylim(1, -1)   # inverted (top = trial 0), updated live as trials accumulate
        ax.set_xlabel('time within trial (s) -- dotted lines: stimulus onset / stimulus end')
        ax.set_ylabel('trial (top = first)')
        ax.set_title('Live click raster (blue=left, orange=right, blend=onset pulse, dot=outcome)')
        _style_axes(ax)
        self._raster_rendered_n = 0

    def _redraw_raster(self, ax):
        n = len(self._left_events)
        for i in range(self._raster_rendered_n, n):
            ax.eventplot([[self._onset_marker_t]], lineoffsets=[i], colors=[ONSET_COLOR],
                         linelengths=0.8)
            if len(self._left_events[i]):
                ax.eventplot([self._left_events[i]], lineoffsets=[i], colors=LEFT_COLOR,
                             linelengths=0.8)
            if len(self._right_events[i]):
                ax.eventplot([self._right_events[i]], lineoffsets=[i], colors=RIGHT_COLOR,
                             linelengths=0.8)
            outcome = self._outcomes[i]
            if outcome is not None:
                ax.scatter([self._waveform_duration * 1.02], [i],
                           color=OUTCOME_COLORS.get(outcome, 'white'), s=18, clip_on=False, zorder=3)
        self._raster_rendered_n = n
        if n:
            ax.set_ylim(max(n, 1), -1)

    # --- psychometric (persistent: running per-bucket sums/counts, one scatter artist) ------------

    def _setup_psych_axes(self):
        ax = self._axes['psych']
        ax.axhline(0.5, color='gray', linestyle=':', linewidth=0.8)
        ax.axvline(0.0, color='gray', linestyle=':', linewidth=0.8)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel('realized count difference (N_right - N_left)')
        ax.set_ylabel('P(chose right)')
        _style_axes(ax)
        self._psych_scatter = ax.scatter([], [], color='tab:purple', alpha=0.85, zorder=3)
        self._psych_bucket_stats = {}
        self._psych_rendered_n = 0
        self._psych_n_valid = 0

    def _redraw_psychometric(self, ax):
        for i in range(self._psych_rendered_n, len(self._psych_records)):
            bucket_key, delta, chose_right = self._psych_records[i]
            b = self._psych_bucket_stats.setdefault(
                bucket_key, {'delta_sum': 0.0, 'delta_n': 0, 'right_sum': 0.0, 'right_n': 0})
            b['delta_sum'] += delta
            b['delta_n'] += 1
            if chose_right is not None:
                b['right_sum'] += (1.0 if chose_right else 0.0)
                b['right_n'] += 1
                self._psych_n_valid += 1
        self._psych_rendered_n = len(self._psych_records)

        xs, ys, sizes = [], [], []
        for b in self._psych_bucket_stats.values():
            if b['right_n'] == 0:
                continue
            xs.append(b['delta_sum'] / b['delta_n'])
            ys.append(b['right_sum'] / b['right_n'])
            sizes.append(20 + 8 * b['right_n'])

        if xs:
            self._psych_scatter.set_offsets(np.column_stack([xs, ys]))
            self._psych_scatter.set_sizes(sizes)
            # ax.relim() does NOT support collections (matplotlib docs: "Collections are not
            # supported") -- set_offsets() alone never updates the axes' autoscale, so without this
            # the scatter silently stays offscreen at whatever tiny default xlim an empty scatter()
            # started with. Caught by the before/after visual spot-check (the panel rendered
            # completely empty despite the title's trial count being correct).
            x_min, x_max = min(xs), max(xs)
            pad = max((x_max - x_min) * 0.1, 1.0)
            ax.set_xlim(x_min - pad, x_max + pad)
        ax.set_title('Psychometric (N={0} valid trials)'.format(self._psych_n_valid))

    # --- reward lick raster (persistent: one small eventplot/scatter added per NEW trial only) ----

    def _setup_lickraster_axes(self):
        ax = self._axes['lickraster']
        ax.axvline(0.0, color='black', linestyle='-', linewidth=0.8)
        ax.set_ylim(1, -1)
        ax.set_xlabel('time relative to reward delivery (s)')
        ax.set_ylabel('rewarded trial (top = first)')
        _style_axes(ax)
        self._lickraster_rendered_n = 0

    def _redraw_reward_lick_raster(self, ax):
        n = len(self._reward_lick_rows)
        for i in range(self._lickraster_rendered_n, n):
            row = self._reward_lick_rows[i]
            if row:
                ax.eventplot([row], lineoffsets=[i], colors=LEFT_COLOR, linelengths=0.8)
                ax.scatter([row[0]], [i], color=RIGHT_COLOR, s=60, zorder=3, edgecolors='black',
                           linewidths=0.8, alpha=0.85)
        self._lickraster_rendered_n = n
        if n:
            ax.set_ylim(max(n, 1), -1)
        ax.set_title('Lick raster -- rewarded trials (n={0})'.format(n))

    # --- outcome tally (persistent: 4 pre-created bars, height updated in place) -------------------

    def _setup_outcome_axes(self):
        ax = self._axes['outcome']
        labels = list(OUTCOME_COLORS.keys())
        colors = [OUTCOME_COLORS[l] for l in labels]
        self._outcome_bars = ax.bar(labels, [0] * len(labels), color=colors)
        self._outcome_bar_index = {label: i for i, label in enumerate(labels)}
        ax.set_ylabel('trial count')
        ax.set_title('Outcome tally')
        _style_axes(ax)

    def _redraw_outcome(self, ax):
        max_count = 1
        for label, i in self._outcome_bar_index.items():
            count = self._outcome_counts.get(label, 0)
            self._outcome_bars[i].set_height(count)
            max_count = max(max_count, count)
        ax.set_ylim(0, max_count)

    # --- session lick (persistent: two scatter artists, offsets replaced each call) -----------------

    def _setup_session_lick_axes(self):
        ax = self._axes['session_lick']
        self._session_lick_scatter = ax.scatter([], [], color=LEFT_COLOR, s=20, zorder=2,
                                                 alpha=0.85, label='licks')
        self._session_first_lick_scatter = ax.scatter(
            [], [], color=RIGHT_COLOR, s=90, zorder=3, edgecolors='black', linewidths=0.8,
            alpha=0.85, label='first lick of trial')
        ax.set_yticks([])
        ax.set_xlabel('session time (s)')
        ax.legend(fontsize=7, loc='upper right', frameon=False)
        _style_axes(ax)

    def _redraw_session_lick(self, ax):
        if self._all_lick_times:
            self._session_lick_scatter.set_offsets(
                np.column_stack([self._all_lick_times, [0] * len(self._all_lick_times)]))
        if self._all_first_lick_times:
            self._session_first_lick_scatter.set_offsets(np.column_stack(
                [self._all_first_lick_times, [0] * len(self._all_first_lick_times)]))
        all_x = list(self._all_lick_times) + list(self._all_first_lick_times)
        if all_x:
            ax.set_xlim(min(all_x) - 1, max(all_x) + 1)
        ax.set_title('Lick events -- full session ({0} total licks logged)'.format(
            self._total_lick_count))

    # --- side-bias handling (persistent: 3 Line2D artists) -------------------------------------------

    def _setup_sidebias_axes(self):
        ax = self._axes['sidebias']
        self._sidebias_line_target, = ax.plot([], [], label='p_right (target)', color='tab:purple')
        self._sidebias_line_recent, = ax.plot([], [], label='recent empirical R-fraction',
                                               color='tab:orange')
        self._sidebias_line_choice, = ax.plot(
            [], [], label='P(chose right), {0}-trial rolling'.format(self.CHOICE_ROLLING_WINDOW),
            color='tab:green')
        ax.axhline(0.5, color='gray', linestyle=':', linewidth=0.8)
        ax.set_ylim(0, 1)
        ax.set_xlabel('trial')
        ax.set_ylabel('fraction')
        ax.set_title('Side-bias handling')
        ax.legend(fontsize=7)
        _style_axes(ax)

    def _redraw_sidebias(self, ax):
        n = len(self._p_right_series)
        if n:
            x = list(range(1, n + 1))
            self._sidebias_line_target.set_data(x, self._p_right_series)
            self._sidebias_line_recent.set_data(x, self._recent_right_frac_series)
            self._sidebias_line_choice.set_data(x, self._rolling_choice_right_fraction())
            ax.set_xlim(1, max(n, 2))

    # --- performance handling (persistent: 1 Line2D + a recreated-each-call shaded region) -----------

    def _setup_performance_axes(self):
        ax = self._axes['performance']
        self._performance_line, = ax.plot([], [], color='black', linewidth=1.0,
                                           label='trailing accuracy')
        self._performance_shade = None
        ax.axhline(ts.REMEDIAL_TRIGGER_ACCURACY, color='tab:red', linestyle='--', linewidth=0.8)
        ax.axhline(ts.REMEDIAL_RECOVER_ACCURACY, color='tab:green', linestyle='--', linewidth=0.8)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel('trial')
        ax.set_ylabel('accuracy')
        ax.set_title('Performance handling (remedial-easy)')
        # A legend proxy for the shaded region -- built once, decoupled from the actual
        # fill_between artist (which gets removed/recreated every call below, so it can't be
        # relied on to still exist at legend-build time the way ax.legend() normally expects).
        remedial_patch = mpatches.Patch(color='tab:red', alpha=0.15, label='remedial_easy active')
        ax.legend(handles=[self._performance_line, remedial_patch], fontsize=7)
        _style_axes(ax)

    def _redraw_performance(self, ax):
        n = len(self._rolling_acc_series)
        if n:
            x = list(range(1, n + 1))
            accs = [a if a is not None else float('nan') for a in self._rolling_acc_series]
            self._performance_line.set_data(x, accs)
            # fill_between has no simple set_data() equivalent -- removed/recreated each call, but
            # that's still far cheaper than the axis-wide cla() this replaces (one collection, not
            # every artist/spine/legend on the panel).
            if self._performance_shade is not None:
                self._performance_shade.remove()
            in_remedial = [t == 'remedial_easy' for t in self._trial_type_series]
            self._performance_shade = ax.fill_between(
                x, 0, 1, where=in_remedial, color='tab:red', alpha=0.15, step='mid')
            ax.set_xlim(1, max(n, 2))

    # --- disengagement handling (persistent: 4 bar patches + 4 text labels) ---------------------------

    def _setup_disengagement_axes(self):
        ax = self._axes['disengagement']
        self._disengagement_bars = ax.barh(
            ['no-resp run', 'no-resp rate', 'no-lick run', 'no-lick rate'], [0, 0, 0, 0],
            color=['tab:green'] * 4)
        self._disengagement_run_text = ax.text(0.02, 0, '', va='center', fontsize=8)
        self._disengagement_rate_text = ax.text(0.02, 1, '', va='center', fontsize=8)
        self._disengagement_lick_run_text = ax.text(0.02, 2, '', va='center', fontsize=8)
        self._disengagement_lick_rate_text = ax.text(0.02, 3, '', va='center', fontsize=8)
        ax.axvline(1.0, color='black', linestyle='--', linewidth=0.8, label='stop threshold')
        ax.set_xlim(0, 1.5)
        ax.set_title('Disengagement handling (live)')
        ax.legend(fontsize=7, loc='lower right')
        _style_axes(ax)

    def _redraw_disengagement(self, ax):
        run_frac = min(self._consecutive_noresp / float(ts.CIRCUIT_BREAKER_CONSECUTIVE_NORESPONSE), 1.5)
        rate_frac = min(self._noresp_rate / ts.CIRCUIT_BREAKER_RATE_THRESHOLD, 1.5) \
            if ts.CIRCUIT_BREAKER_RATE_THRESHOLD else 0
        lick_run_frac = min(
            self._consecutive_no_lick / float(ts.CIRCUIT_BREAKER_CONSECUTIVE_NO_CONSUMPTION_LICK), 1.5)
        lick_rate_frac = min(self._no_lick_rate / ts.CIRCUIT_BREAKER_CONSUMPTION_LICK_RATE_THRESHOLD, 1.5) \
            if ts.CIRCUIT_BREAKER_CONSUMPTION_LICK_RATE_THRESHOLD else 0
        run_color = 'tab:red' if self._consecutive_noresp >= ts.CIRCUIT_BREAKER_CONSECUTIVE_NORESPONSE \
            else 'tab:green'
        rate_color = 'tab:red' if self._noresp_rate > ts.CIRCUIT_BREAKER_RATE_THRESHOLD else 'tab:green'
        lick_run_color = 'tab:red' \
            if self._consecutive_no_lick >= ts.CIRCUIT_BREAKER_CONSECUTIVE_NO_CONSUMPTION_LICK \
            else 'tab:green'
        lick_rate_color = 'tab:red' \
            if self._no_lick_rate > ts.CIRCUIT_BREAKER_CONSUMPTION_LICK_RATE_THRESHOLD else 'tab:green'
        self._disengagement_bars[0].set_width(run_frac)
        self._disengagement_bars[0].set_color(run_color)
        self._disengagement_bars[1].set_width(rate_frac)
        self._disengagement_bars[1].set_color(rate_color)
        self._disengagement_bars[2].set_width(lick_run_frac)
        self._disengagement_bars[2].set_color(lick_run_color)
        self._disengagement_bars[3].set_width(lick_rate_frac)
        self._disengagement_bars[3].set_color(lick_rate_color)
        self._disengagement_run_text.set_position((run_frac + 0.02, 0))
        self._disengagement_run_text.set_text('{0}/{1}'.format(
            self._consecutive_noresp, ts.CIRCUIT_BREAKER_CONSECUTIVE_NORESPONSE))
        self._disengagement_rate_text.set_position((rate_frac + 0.02, 1))
        self._disengagement_rate_text.set_text('{0:.0%}/{1:.0%}'.format(
            self._noresp_rate, ts.CIRCUIT_BREAKER_RATE_THRESHOLD))
        self._disengagement_lick_run_text.set_position((lick_run_frac + 0.02, 2))
        self._disengagement_lick_run_text.set_text('{0}/{1}'.format(
            self._consecutive_no_lick, ts.CIRCUIT_BREAKER_CONSECUTIVE_NO_CONSUMPTION_LICK))
        self._disengagement_lick_rate_text.set_position((lick_rate_frac + 0.02, 3))
        self._disengagement_lick_rate_text.set_text('{0:.0%}/{1:.0%}'.format(
            self._no_lick_rate, ts.CIRCUIT_BREAKER_CONSUMPTION_LICK_RATE_THRESHOLD))
