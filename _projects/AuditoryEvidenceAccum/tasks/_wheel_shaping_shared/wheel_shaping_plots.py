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

Extended for Stage 3/4 (training_protocol.md Revision 2) with panels those stages actually need
that Stage 1/2 have no use for -- a click raster and abort-by-epoch tally (both new outcome/
stimulus dimensions Stage 1/2 don't have at all), a response-time histogram split by correct/
incorrect (no correct/incorrect concept exists before Stage 3), a rolling-accuracy line with
warmup/repeat trials visually distinguished from the advancement-relevant 'main' subset, a
session-engagement indicator (against the same consecutive-abort/no-initiation thresholds the
task script itself uses to end a session), and -- Stage 4 only -- a dual-staircase progress panel
highlighting which of response-threshold/quiescence is this session's Tag-B-selected active one.
The click-raster/abort-tally panels are small local ports of Tests/tasks/poisson_clicks_test/
live_plots.py's own drawing logic (duplicated, not imported -- same "shared within this paradigm
only" placement convention already used for _style_axes()/_capped_figsize() above).
"""
import time

import matplotlib.pyplot as plt

import staircase   # for Stage 4's own target constants (STAGE4_RESPONSE_THRESHOLD_FINAL_FRACTION/
                    # QUIESCENCE_CEILING_S/QUIESCENCE_FLOOR_S), so the progress panel's fractions
                    # match the task script's own advancement gate exactly rather than a second,
                    # separately-maintained copy of the same numbers.


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
    # Stage 3/4's own outcome vocabulary (a real 2AFC choice, unlike Stage 1/2's shaping-only set)
    'Reward': 'tab:green',
    'NoReward': 'tab:red',
    'NoResponse': 'tab:gray',
    'Abort': 'tab:purple',
}
SIDE_COLORS = {'L': 'tab:blue', 'R': 'tab:red'}
LICK_COLOR = 'tab:cyan'
FIRST_LICK_COLOR = 'tab:pink'
ABORT_EPOCH_COLORS = {'quiescence': 'tab:gray', 'cue': 'tab:purple', 'delay': 'tab:brown'}
TRIAL_TYPE_MARKERS = {'main': 'o', 'warmup': '^', 'repeat': 's'}

# Display-only reference matching stage3/4's own VAR_CONSECUTIVE_ABORT_LIMIT -- not imported from
# either task script (this module has no dependency on either), so kept as its own constant; if
# that VAR_ is ever retuned, update this one too.
_ENGAGEMENT_ABORT_LIMIT_DISPLAY = 20
_CLICK_RASTER_MAX_TRIALS = 30   # trailing-window cap, same "bounded redraw cost" principle as
                                 # wheel_position_plot.py's own MAX_PLOTTED_POINTS

# Display-only reference for the Stage 1 gain-progress bar (see _redraw_progress()) -- matches
# stage1_wheel_shaping.py's own VAR_GAIN_INITIAL_MULT (training_protocol.md's stated "~2x final").
# Not used by the actual decay logic (staircase.decay_gain()), which needs no such reference.
GAIN_INITIAL_MULT_DISPLAY_REF = 2.0


class WheelShapingPlots(object):

    def __init__(self, stage, threshold_final_deg, prev_session_values=None, session_status=None):
        """
        :param int stage: 1, 2, 3, or 4 -- controls which panels are meaningful. Stage 1 has no
            direction-ratio withholding and a fixed threshold/decaying gain instead of a staircase.
            Stage 3/4 are a real 2AFC choice (correct/incorrect/abort exist) with clicks -- a
            structurally different panel set from Stage 1/2's shaping-only one, see the new
            mosaic below.
        :param float threshold_final_deg: the FINAL (Stage 3+) movement threshold, used to draw the
            raster's y-axis in real degrees and to express the progress panel's threshold bar as a
            fraction of this value.
        :param dict prev_session_values: this session's own STARTING values (i.e. where the LAST
            session left off), e.g. {'threshold_deg': 7.0, 'iti_s': 0.5, 'gain_mult': 1.8} -- the
            caller already has these loaded from StageState before this session's own trials can
            change them. Shown as an annotation on the progress panel so day-over-day movement is
            visible without cross-referencing the state file. Keys not present are simply not
            annotated (e.g. Stage 1 has no 'iti_s').
        :param dict session_status: Stage 3/4 only -- static-for-the-whole-session display info the
            progress panel can't derive from prev_session_values/add_trial() alone:
            {'in_trial_threshold_deg': 10.0} (Stage 3/4, the in-trial abort threshold in force),
            {'click_attenuated': True} (Stage 3 only, whether the click-level ramp is still active),
            {'staircase_active': 'response'} (Stage 4 only, this session's Tag-B selection).
        """
        self._stage = stage
        self._threshold_final_deg = threshold_final_deg
        self._prev_session_values = prev_session_values or {}
        self._session_status = session_status or {}

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

        # --- Stage 3/4 only, from here down -----------------------------------------------------
        self._click_trial_idx = []       # one row index per trial that had any clicks
        self._click_times = []           # flat list of (trial_row, t, side) for the raster
        self._click_rendered_n = 0
        self._abort_epoch_counts = {}
        self._response_times_correct = []
        self._response_times_incorrect = []
        self._accuracy_trial_idx = []
        self._accuracy_values = []       # 1.0/0.0 per MAIN trial, for the rolling-mean line
        self._accuracy_rolling = []
        self._accuracy_marker_idx = {'main': [], 'warmup': [], 'repeat': []}
        self._accuracy_marker_val = {'main': [], 'warmup': [], 'repeat': []}
        self._consecutive_aborts = 0
        self._last_trial_time_s = None
        self._time_since_last_trial_s = None
        self._cur_in_trial_threshold_deg = self._session_status.get('in_trial_threshold_deg')

        if stage in (3, 4):
            mosaic = [['raster', 'raster', 'click_raster'],
                      ['progress', 'abort_epoch', 'response_time'],
                      ['accuracy', 'accuracy', 'engagement'],
                      ['reward_licks', 'reward_licks', 'outcome'],
                      ['lick_timeline', 'lick_timeline', 'lick_timeline']]
        elif stage == 2:
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
        figsize = (13, 13) if stage in (3, 4) else (11, 12)
        self._fig, self._axes = plt.subplot_mosaic(mosaic, figsize=_capped_figsize(*figsize))
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
        if stage in (3, 4):
            self._setup_click_raster_axes()
            self._setup_abort_epoch_axes()
            self._setup_response_time_axes()
            self._setup_accuracy_axes()
            self._setup_engagement_axes()
        self._redraw()

    # --- data intake -------------------------------------------------------------------------------

    def add_trial(self, side, magnitude_deg, threshold_deg, outcome, gain_mult=None, iti_s=None,
                   direction_ratio=None, lick_times_abs=None, reward_time_abs=None,
                   trial_type=None, abort_epoch=None, response_time_s=None,
                   click_times_l=None, click_times_r=None, in_trial_threshold_deg=None):
        """ Call once per trial. side='L'/'R' (whichever direction the movement was, even if
        below threshold -- use the larger-magnitude direction for a NoMovement trial); magnitude_deg
        is the signed peak displacement that trial; outcome is 'Rewarded'/'Withheld'/'NoMovement'
        (Stage 1/2) or 'Reward'/'NoReward'/'NoResponse'/'Abort' (Stage 3/4); threshold_deg is that
        trial's own (possibly staircase-updated) threshold. gain_mult (Stage 1) / direction_ratio
        (Stage 2) are stage-specific, pass whichever applies.

        lick_times_abs: this trial's own Port1In event times, in absolute session-elapsed seconds
        (same basis as WHEEL_POS logging) -- feeds the full-session lick timeline, same convention
        as live_plots.py's add_trial_licks(). reward_time_abs: this trial's own absolute
        reward-delivery time, required only when outcome is a reward outcome -- feeds the
        reward-aligned lick raster.

        Stage 3/4 only, all optional: trial_type ('main'/'warmup'/'repeat') -- feeds the accuracy
        line's marker style and excludes non-'main' trials from the rolling-accuracy/advancement
        computation shown here (the task script's own advancement gate is the authority, this is
        display only). abort_epoch ('cue'/'delay', only when outcome=='Abort') -- feeds the
        abort-by-epoch tally. response_time_s (go-cue to threshold-crossing, only for a genuine
        response) -- feeds the correct/incorrect response-time histogram. click_times_l/
        click_times_r (this trial's own generator-relative click times, i.e. straight from
        CLICK_TIMES_L/R) -- feeds the click raster. in_trial_threshold_deg -- if given (and
        different from the previous call), redraws the raster's extra abort-threshold reference
        lines. """
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
        if in_trial_threshold_deg is not None:
            self._cur_in_trial_threshold_deg = in_trial_threshold_deg

        lick_times_abs = lick_times_abs or []
        self._total_lick_count += len(lick_times_abs)
        self._all_lick_times.extend(lick_times_abs)
        if lick_times_abs:
            self._all_first_lick_times.append(min(lick_times_abs))
        if outcome in ('Rewarded', 'Reward') and reward_time_abs is not None:
            self._reward_lick_rows.append([t - reward_time_abs for t in lick_times_abs])

        if self._stage in (3, 4):
            row = self._session_trial_count
            # NOT `click_times_l or []` -- click_times_l/r are commonly numpy arrays straight from
            # generate_trial_clicks() (see stage3_clicks_direction.py), and `array or []` raises
            # ValueError ("truth value of an array with more than one element is ambiguous") for
            # any array with 2+ elements. `is None` is the only safe emptiness check here.
            for t in ([] if click_times_l is None else click_times_l):
                self._click_times.append((row, t, 'L'))
            for t in ([] if click_times_r is None else click_times_r):
                self._click_times.append((row, t, 'R'))

            if outcome == 'Abort':
                key = abort_epoch or 'unknown'
                self._abort_epoch_counts[key] = self._abort_epoch_counts.get(key, 0) + 1
                self._consecutive_aborts += 1
            else:
                self._consecutive_aborts = 0

            if response_time_s is not None:
                if outcome in ('Reward',):
                    self._response_times_correct.append(response_time_s)
                elif outcome in ('NoReward',):
                    self._response_times_incorrect.append(response_time_s)

            correct = 1.0 if outcome == 'Reward' else (0.0 if outcome == 'NoReward' else None)
            ttype = trial_type or 'main'
            if correct is not None and ttype == 'main':
                self._accuracy_trial_idx.append(row)
                self._accuracy_values.append(correct)
                window = self._accuracy_values[-20:]
                self._accuracy_rolling.append(sum(window) / float(len(window)))
            if correct is not None:
                self._accuracy_marker_idx[ttype if ttype in TRIAL_TYPE_MARKERS else 'main'].append(row)
                self._accuracy_marker_val[ttype if ttype in TRIAL_TYPE_MARKERS else 'main'].append(
                    correct)

            now = time.time()
            if self._last_trial_time_s is not None:
                self._time_since_last_trial_s = now - self._last_trial_time_s
            self._last_trial_time_s = now

        self._redraw()

    # --- raster (persistent scatter) ----------------------------------------------------------------

    def _setup_raster_axes(self):
        ax = self._axes['raster']
        self._raster_scatter = ax.scatter([], [], s=14, c=[])
        self._raster_threshold_pos = ax.axhline(0, color='k', linestyle=':', linewidth=0.8)
        self._raster_threshold_neg = ax.axhline(0, color='k', linestyle=':', linewidth=0.8)
        ax.axhline(0, color='#888888', linewidth=0.6)
        if self._stage in (3, 4):
            # Stage 3/4 only: a second, TIGHTER pair of reference lines for the in-trial
            # (cue-abort) threshold, distinct from the (looser) response/choice threshold the
            # first pair above already draws -- so it's visible at a glance how close excursions
            # are running to the abort boundary vs. the actual choice boundary.
            self._raster_intrial_pos = ax.axhline(
                0, color='tab:purple', linestyle='--', linewidth=0.8, alpha=0.7)
            self._raster_intrial_neg = ax.axhline(
                0, color='tab:purple', linestyle='--', linewidth=0.8, alpha=0.7)
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
        if self._stage in (3, 4) and self._cur_in_trial_threshold_deg is not None:
            self._raster_intrial_pos.set_ydata([self._cur_in_trial_threshold_deg] * 2)
            self._raster_intrial_neg.set_ydata([-self._cur_in_trial_threshold_deg] * 2)
        ax.set_xlim(0, max(10, self._trial_idx[-1] + 1))
        y_span = max(self._threshold_final_deg, max(abs(m) for m in self._magnitude_deg)) * 1.1
        ax.set_ylim(-y_span, y_span)

    # --- progress / staircase state (persistent bars, mirrors live_plots_lookback.py's style) ------

    def _setup_progress_axes(self):
        ax = self._axes['progress']
        if self._stage == 1:
            labels = ['trials/200', 'threshold/final', 'gain (final=1.0x)']
        elif self._stage == 2:
            labels = ['trials/200', 'threshold/final', 'ITI/1.5s']
        elif self._stage == 3:
            labels = ['main trials', 'response thresh (frozen)', 'click level']
        else:   # stage 4
            labels = ['main trials', 'response thresh/target', 'quiescence/target']
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
        colors = None
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
        elif self._stage == 2:
            iti_frac = min((self._cur_iti_s or 0) / 1.5, 1.3)
            fracs = [trial_frac, threshold_frac, iti_frac]
            texts = ['{0}/200'.format(self._session_trial_count),
                     threshold_text,
                     '{0:.2f}/1.50s{1}'.format(
                         self._cur_iti_s or 0, self._prev_session_suffix('iti_s', '{0:.2f}s'))]
        elif self._stage == 3:
            attenuated = bool(self._session_status.get('click_attenuated'))
            click_frac = 0.5 if attenuated else 1.0
            fracs = [trial_frac, threshold_frac, click_frac]
            texts = ['{0} (this session)'.format(self._session_trial_count),
                     '{0:.1f}deg (frozen from Stage 2)'.format(self._cur_threshold_deg or 0),
                     'ATTENUATED (-6dB)' if attenuated else 'FULL LEVEL']
            colors = ['tab:blue', 'tab:blue', 'tab:orange' if attenuated else 'tab:green']
        else:   # stage 4
            active = self._session_status.get('staircase_active')
            resp_target = staircase.STAGE4_RESPONSE_THRESHOLD_FINAL_FRACTION
            resp_frac = min((self._cur_threshold_deg or 0)
                             / (resp_target * self._threshold_final_deg), 1.3)
            quiescence_s = self._session_status.get('quiescence_s')
            q_span = staircase.QUIESCENCE_CEILING_S - staircase.QUIESCENCE_FLOOR_S
            q_frac = (min(max((quiescence_s - staircase.QUIESCENCE_FLOOR_S) / q_span, 0.0), 1.3)
                      if quiescence_s is not None else 0.0)
            fracs = [trial_frac, resp_frac, q_frac]
            texts = ['{0} (this session)'.format(self._session_trial_count),
                     '{0:.1f}deg (target {1:.0%} of final){2}'.format(
                         self._cur_threshold_deg or 0, resp_target,
                         '  <- ACTIVE' if active == 'response' else ''),
                     '{0}s (target {1:.1f}s){2}'.format(
                         '{0:.3f}'.format(quiescence_s) if quiescence_s is not None else '?',
                         staircase.QUIESCENCE_CEILING_S,
                         '  <- ACTIVE' if active == 'quiescence' else '')]
            colors = ['tab:blue',
                      'tab:orange' if active == 'response' else 'tab:blue',
                      'tab:orange' if active == 'quiescence' else 'tab:blue']
        for i, (bar, text, frac, label) in enumerate(
                zip(self._progress_bars, self._progress_texts, fracs, texts)):
            bar.set_width(frac)
            if colors is not None:
                bar.set_color('tab:green' if frac >= 1.0 and colors[i] == 'tab:blue' else colors[i])
            else:
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

    # --- click raster (Stage 3/4 only, persistent scatter, small local port of
    # poisson_clicks_test/live_plots.py's own raster drawing -- duplicated, not imported, see
    # module docstring) --------------------------------------------------------------------------

    def _setup_click_raster_axes(self):
        ax = self._axes['click_raster']
        self._click_scatter_l = ax.scatter([], [], s=4, color='tab:blue', label='L', alpha=0.7)
        self._click_scatter_r = ax.scatter([], [], s=4, color='tab:red', label='R', alpha=0.7)
        ax.axvline(0.0, color='black', linewidth=0.6)
        ax.set_xlabel('time from stimulus onset (s)')
        ax.set_ylabel('trial')
        ax.set_title('Click raster (last {0} trials)'.format(_CLICK_RASTER_MAX_TRIALS))
        ax.legend(fontsize=7, loc='upper right', frameon=False)
        _style_axes(ax)

    def _redraw_click_raster(self, ax):
        n = len(self._click_times)
        if n == self._click_rendered_n:
            return
        self._click_rendered_n = n
        if not self._click_times:
            return
        # Only the trailing window's own trials are kept on screen -- same "bounded redraw cost
        # regardless of session length" principle as MAX_PLOTTED_POINTS elsewhere in this codebase.
        min_row = max(1, self._session_trial_count - _CLICK_RASTER_MAX_TRIALS + 1)
        visible = [(row, t, side) for row, t, side in self._click_times if row >= min_row]
        left = [(t, row) for row, t, side in visible if side == 'L']
        right = [(t, row) for row, t, side in visible if side == 'R']
        if left:
            self._click_scatter_l.set_offsets(left)
        if right:
            self._click_scatter_r.set_offsets(right)
        ax.set_ylim(self._session_trial_count + 1, max(min_row - 1, 0))
        ax.set_xlim(-0.05, 0.3)

    # --- abort-by-epoch tally (Stage 3/4 only, cheap cla()+bar, same low-cardinality-tally
    # convention as the outcome panel) -------------------------------------------------------------

    def _setup_abort_epoch_axes(self):
        _style_axes(self._axes['abort_epoch'])

    def _redraw_abort_epoch(self, ax):
        ax.cla()
        labels = list(self._abort_epoch_counts.keys())
        counts = [self._abort_epoch_counts[l] for l in labels]
        colors = [ABORT_EPOCH_COLORS.get(l, 'gray') for l in labels]
        if labels:
            ax.bar(labels, counts, color=colors)
        ax.set_ylabel('abort count')
        ax.set_title('Aborts by epoch (consecutive: {0})'.format(self._consecutive_aborts))
        _style_axes(ax)

    # --- response-time histogram (Stage 3/4 only, cheap cla()+hist -- session-scale trial counts
    # make a full rebuild cheap, same reasoning already used for the outcome tally) -----------------

    def _setup_response_time_axes(self):
        _style_axes(self._axes['response_time'])

    def _redraw_response_time(self, ax):
        ax.cla()
        all_times = self._response_times_correct + self._response_times_incorrect
        if all_times:
            bins = min(20, max(5, len(all_times) // 2))
            if self._response_times_correct:
                ax.hist(self._response_times_correct, bins=bins, alpha=0.6, color='tab:green',
                        label='correct')
            if self._response_times_incorrect:
                ax.hist(self._response_times_incorrect, bins=bins, alpha=0.6, color='tab:red',
                        label='incorrect')
            ax.legend(fontsize=7, loc='upper right', frameon=False)
        ax.set_xlabel('go-cue to threshold-crossing (s)')
        ax.set_ylabel('trial count')
        ax.set_title('Response time by outcome')
        _style_axes(ax)

    # --- rolling accuracy (Stage 3/4 only, persistent line + per-trial-type marker scatter) --------

    def _setup_accuracy_axes(self):
        ax = self._axes['accuracy']
        self._accuracy_line, = ax.plot([], [], color='black', linewidth=1.2,
                                        label='rolling accuracy (main, window=20)')
        self._accuracy_scatters = {}
        for ttype, marker in TRIAL_TYPE_MARKERS.items():
            self._accuracy_scatters[ttype] = ax.scatter(
                [], [], s=14, marker=marker, alpha=0.5,
                color='tab:blue' if ttype == 'main' else 'tab:gray', label=ttype)
        ax.axhline(0.70, color='gray', linestyle=':', linewidth=0.8, label='70% advancement gate')
        ax.axhline(0.5, color='#cccccc', linewidth=0.6)
        ax.set_ylim(0, 1)
        ax.set_xlabel('trial')
        ax.set_ylabel('accuracy (gamma=+-1.0)')
        ax.set_title('Accuracy -- warmup/repeat shown but excluded from the rolling line')
        ax.legend(fontsize=6, loc='lower right', frameon=False, ncol=2)
        _style_axes(ax)

    def _redraw_accuracy(self, ax):
        if self._accuracy_trial_idx:
            self._accuracy_line.set_data(self._accuracy_trial_idx, self._accuracy_rolling)
        for ttype, scatter in self._accuracy_scatters.items():
            idx = self._accuracy_marker_idx[ttype]
            val = self._accuracy_marker_val[ttype]
            if idx:
                scatter.set_offsets(list(zip(idx, val)))
        if self._trial_idx:
            ax.set_xlim(0, max(10, self._trial_idx[-1] + 1))

    # --- session-engagement indicator (Stage 3/4 only, cheap cla()+text against the same triggers
    # the task script itself uses to end a session) --------------------------------------------------

    def _setup_engagement_axes(self):
        ax = self._axes['engagement']
        ax.axis('off')

    def _redraw_engagement(self, ax):
        ax.cla()
        ax.axis('off')
        gap_s = self._time_since_last_trial_s
        gap_text = '{0:.0f}s since last trial'.format(gap_s) if gap_s is not None else 'n/a'
        lines = [
            'Consecutive aborts: {0}/{1}'.format(self._consecutive_aborts,
                                                   _ENGAGEMENT_ABORT_LIMIT_DISPLAY),
            gap_text,
        ]
        color = ('tab:red' if self._consecutive_aborts >= _ENGAGEMENT_ABORT_LIMIT_DISPLAY - 5
                 else 'black')
        ax.text(0.05, 0.6, '\n'.join(lines), fontsize=10, va='center', color=color,
                transform=ax.transAxes)
        ax.set_title('Session engagement')

    # --- unified redraw ------------------------------------------------------------------------------

    def _redraw(self):
        self._redraw_raster(self._axes['raster'])
        self._redraw_progress(self._axes['progress'])
        self._redraw_outcome(self._axes['outcome'])
        self._redraw_reward_licks(self._axes['reward_licks'])
        self._redraw_lick_timeline(self._axes['lick_timeline'])
        if self._stage == 2:
            self._redraw_direction_ratio(self._axes['direction_ratio'])
        if self._stage in (3, 4):
            self._redraw_click_raster(self._axes['click_raster'])
            self._redraw_abort_epoch(self._axes['abort_epoch'])
            self._redraw_response_time(self._axes['response_time'])
            self._redraw_accuracy(self._axes['accuracy'])
            self._redraw_engagement(self._axes['engagement'])
        if not self._did_initial_layout:
            self._fig.tight_layout()
            self._did_initial_layout = True
        plt.pause(0.001)
