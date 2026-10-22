# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Protocol-agnostic Bpod trial-loop helpers, factored out of wheel_turn_reward.py/
hifi_singleside_test.py/hifi_alternating_easy_test.py (Tests/tasks/) where this logic used to be
copy-pasted near-verbatim. Lives at _projects/_shared/ (a sibling of every GUI project, not nested
inside one) so any task in any project can reach it via the same __file__-relative sys.path.insert
trick already used for tasks/poisson_clicks_test/ -- confirmed against board_com.py's run_task():
the task script's own absolute path is what's launched, so __file__ always resolves correctly
regardless of the subprocess's cwd (the session folder) or how many directories separate the task
from this one.

State-machine *construction* for a specific protocol (which states, which transitions, which
output_actions) is deliberately NOT here -- that's protocol-specific and belongs in the task
script itself. This module only owns the generic plumbing every wheel/hold-based protocol needs:
the SESSION-LONG WHEEL_POS-publishing thread, the thin wrapper around run_state_machine(), the
states_durations nan-check, the hold-to-init wait (wheel steadiness, optionally also gated on no
recent licking), and port-event (lick) reporting.

**WHEEL_POS polling is session-long, not per-Bpod-call** -- this was a real, confirmed defect in
an earlier version of this module: a background poll thread used to be spawned fresh and torn
down completely on every single run_trial_state_machine() call (and wait_for_held_steady() had its
own SECOND, independently-scoped poll thread), so there was a real window with ZERO WHEEL_POS
logging between one trial's state machine finishing and the next trial's starting -- covering the
rotary zero/threshold-arming serial round-trips, several CSV-flushed register() calls, dot/camera
pump calls, and StateMachine construction. Call start() once (after connecting to the rotary
encoder) and stop() once (at true session end, before rotary.close()) -- see those methods' own
docstrings. The remaining, NOT-eliminated residual: the poll thread still genuinely blocks on
rotary_lock for the real duration of set_zero_position()/set_and_enable_thresholds() (two threads
cannot talk to the same physical serial connection at once) -- that's a hardware constraint, not a
software-scoping artifact, and is not fixed by this change.
"""
import ctypes
import math
import sys
import threading
import time
from collections import deque

from pybpodapi.protocol import Bpod, StateMachine


def was_visited(visited, name):
    """ Bpod logs a synthetic (nan, nan) entry for every state DEFINED in a state machine, not
    just ones actually entered -- so `name in visited` is always true regardless of outcome. A
    state was genuinely visited only if it has a real (non-nan) start time. """
    durations = visited.get(name)
    return bool(durations) and not math.isnan(durations[-1][0])


class _HoldWatch(object):
    """ Per-wait_for_held_steady()-call state for the rolling wheel-steadiness window --
    previously the closure state of a private per-call thread (wheel_watch()), now just a plain
    object the shared session-long poll loop reads/updates via TrialRunner._hold_watch. """

    def __init__(self, required_hold, steady_threshold_deg):
        self.required_hold = required_hold
        self.steady_threshold_deg = steady_threshold_deg
        self.window = deque()
        self.wheel_ready = threading.Event()
        self.n_breaks_holder = [0]
        self.warned = [False]
        self.wait_start = time.time()


class TrialRunner(object):
    """
    Wraps a connected Bpod + rotary encoder pair with the trial-loop plumbing every wheel-based
    protocol in this project has needed so far. One instance per task script, constructed once
    connections are up: `runner = TrialRunner(my_bpod, rotary, log_python_t0)`. Call
    `runner.start()` once right after construction and `runner.stop()` once at session end.
    """

    def __init__(self, bpod, rotary, log_python_t0, still_poll_hz=100, poll_hz=100):
        self.bpod = bpod
        self.rotary = rotary
        self.log_python_t0 = log_python_t0
        self.still_poll_hz = still_poll_hz
        self.poll_interval = 1.0 / poll_hz

        self.rotary_lock = threading.Lock()
        self.register_lock = threading.Lock()
        self._hold_watch = None       # set/cleared by wait_for_held_steady() around its own wait
        self.last_send_t = None       # see run_trial_state_machine()'s own docstring
        self._poll_thread = None
        self._poll_stop_event = None
        self._timer_res_active = False

    def start(self):
        """ Starts the ONE session-long WHEEL_POS poll thread (see module docstring for why this
        replaced the old per-Bpod-call thread lifecycle). Also requests 1ms Windows timer
        resolution for the duration of the session -- time.sleep() below ~16ms is otherwise
        quantized to Windows' default ~15.6ms scheduler tick, confirmed as a real contributor to
        this poll loop's own realized-rate shortfall. Call once, right after connecting to the
        rotary encoder. """
        if sys.platform.startswith('win'):
            try:
                ctypes.windll.kernel32.timeBeginPeriod(1)
                self._timer_res_active = True
            except Exception:
                pass
        self._poll_stop_event = threading.Event()
        self._poll_thread = threading.Thread(target=self._session_poll_loop,
                                              args=(self._poll_stop_event,), daemon=True)
        self._poll_thread.start()

    def stop(self):
        """ Stops the session-long poll thread (bounded join, same 3s convention every other
        thread join in this module already uses) and releases the Windows timer-resolution
        request. Call once, at true session end -- before rotary.close(), since the poll thread
        still touches self.rotary right up until it's told to stop. """
        if self._poll_stop_event is not None:
            self._poll_stop_event.set()
        if self._poll_thread is not None:
            self._poll_thread.join(timeout=3.0)
            if self._poll_thread.is_alive():
                print("WARNING: session-long wheel-poll thread did not stop within 3s -- "
                      "continuing anyway.", flush=True)
        if self._timer_res_active:
            try:
                ctypes.windll.kernel32.timeEndPeriod(1)
            except Exception:
                pass
            self._timer_res_active = False

    def register(self, key, value, flush=True):
        """ flush=False is for TrialRunner's own high-frequency internal use only (WHEEL_POS) --
        every task-script call site keeps the default (always-flush) durability. See
        _publish_wheel_pos()/pybpod-api's own register_value(flush=...) for what this actually
        changes. """
        with self.register_lock:
            self.bpod.register_value(key, value, flush=flush)

    def register_port_events(self, prefix, events_dict, anchor_t, port_in_key='Port1In',
                              port_out_key='Port1Out'):
        """ Extracts and registers BOTH edges of a digital input (Port1In/Port1Out by default) as
        absolute-time VALs, comma-joined strings matching the existing CLICK_TIMES_L/CLICK_TIMES_R
        convention -- per explicit instruction, every lick-time extraction in this codebase should
        report both the contact start AND end, not just the start: it's useful data in its own
        right, and pairing them makes per-contact duration directly auditable (relevant to
        diagnosing a suspected sensor debounce/threshold issue -- implausibly few, implausibly
        short Port1In events) without hand-parsing raw EVENT rows. Returns (in_times_abs,
        out_times_abs) so callers can use the first element exactly where they already use their
        own hand-built lick_times_abs/cue_lick_times_abs/choice_lick_times_abs. """
        in_times = [anchor_t + t for t in events_dict.get(port_in_key, [])]
        out_times = [anchor_t + t for t in events_dict.get(port_out_key, [])]
        self.register('{0}_TIMES_IN'.format(prefix),
                       ','.join('{0:.4f}'.format(t) for t in in_times))
        self.register('{0}_TIMES_OUT'.format(prefix),
                       ','.join('{0:.4f}'.format(t) for t in out_times))
        return in_times, out_times

    def _now(self):
        return time.time() - self.log_python_t0

    def _publish_wheel_pos(self, pos, t=None):
        t = self._now() if t is None else t
        # flush=False: WHEEL_POS is by far the highest-frequency VAL in the whole system -- an
        # unconditional flush() on every single sample (the default for every other, low-frequency
        # VAL) was itself a confirmed major contributor to this loop's realized-rate shortfall.
        # pybpod-api's own register_value()/Session machinery still forces a flush at a bounded
        # interval regardless (see its own docstring) -- this only relaxes "every sample" to
        # "at least every ~50ms", not "never".
        self.register('WHEEL_POS', '{0:.3f},{1:.2f}'.format(t, pos), flush=False)

    def _update_hold_watch(self, hw, pos, now_abs, t_rel):
        """ Exact logic the old per-call wheel_watch() thread used, now operating on a shared
        _HoldWatch object the session-long poll loop updates in place. """
        if hw.window and abs(pos - hw.window[-1][1]) > hw.steady_threshold_deg:
            hw.n_breaks_holder[0] += 1
            self.register('QUIESCENCE_RESET_TIME', t_rel)
        hw.window.append((now_abs, pos))
        while len(hw.window) > 1 and hw.window[1][0] <= now_abs - hw.required_hold:
            hw.window.popleft()
        if not hw.warned[0] and (now_abs - hw.wait_start) > 3.0 + hw.required_hold:
            hw.warned[0] = True
            print("Waiting for the wheel to hold steady (current position: "
                  "{0:.1f}deg)...".format(pos), flush=True)
        positions = [p for _, p in hw.window]
        has_full_span = hw.window[0][0] <= now_abs - hw.required_hold
        if has_full_span and (max(positions) - min(positions)) <= hw.steady_threshold_deg:
            hw.wheel_ready.set()
        else:
            hw.wheel_ready.clear()

    def _session_poll_loop(self, stop_event):
        """ Publishes WHEEL_POS continuously for the WHOLE SESSION -- not scoped to whichever
        Bpod call happens to be running (see module docstring). Also drives the rolling
        wheel-steadiness window for wait_for_held_steady(), via self._hold_watch, whenever one is
        active. Wrapped in a broad try/except: a Kill/Stop can tear down the Bpod session out
        from under this thread mid-register_value() (observed on hardware: AttributeError on
        bpod._session after a mid-trial Stop), which is harmless in effect -- the process is
        ending either way -- but was spraying an alarming traceback; fail silently instead.

        Uses "measure elapsed, sleep the remainder" (matching camera_recorder.py's own capture
        loop) instead of an unconditional full-interval sleep after the iteration's own work --
        the previous unconditional-sleep version was confirmed to make each period
        work_time + poll_interval instead of max(work_time, poll_interval), needlessly lowering
        the realized rate further on top of the real serial-I/O + register() cost already
        dominating it. """
        while not stop_event.is_set():
            iter_start = time.time()
            try:
                with self.rotary_lock:
                    p = self.rotary.current_position()
                now_abs = time.time()
                t_rel = now_abs - self.log_python_t0
                self._publish_wheel_pos(p, t_rel)
                hw = self._hold_watch
                if hw is not None:
                    self._update_hold_watch(hw, p, now_abs, t_rel)
            except Exception:
                break
            sleep_time = self.poll_interval - (time.time() - iter_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def run_trial_state_machine(self, sma):
        """ Thin wrapper around send/run_state_machine() -- no thread lifecycle to manage here
        any more (see module docstring); WHEEL_POS polling is session-long now, started/stopped
        once via start()/stop(), not per call.

        Also captures `self.last_send_t` right after the upload (send_state_machine()) completes
        and right before the actual RUN command -- the closest anchor to the real hardware
        trial-start instant reachable without instrumenting pybpod-api itself. Callers should
        re-anchor any send_t/cue_send_t-based VAL registration (THRESHOLD_CROSSING_TIME,
        SM_SEND_TIME, reward/abort/lick times, etc.) to `runner.last_send_t` once this call
        returns, instead of whatever earlier time.time() they captured before building the
        StateMachine -- that earlier value doesn't account for SM-construction + upload +
        RUN-handshake latency, which was a confirmed source of 0.22-0.48s of drift between
        derived event timestamps and the hardware-relative offsets they were added to. """
        self.bpod.send_state_machine(sma)
        self.last_send_t = self._now()
        return self.bpod.run_state_machine(sma)

    def wait_for_held_steady(self, required_hold, steady_threshold_deg, require_no_lick=True):
        """
        Blocks until the wheel has stayed within steady_threshold_deg (peak-to-peak) for a
        continuous trailing window of required_hold seconds -- a rolling-window check, not a
        threshold-crossing one, since a Bpod self-looping state can't reproduce "hasn't moved more
        than X degrees at any point in the last N seconds" (only "did a crossing happen"). The
        rolling window itself is maintained by the shared session-long poll loop (via
        self._hold_watch, set here and cleared in this method's own finally), not a private
        per-call thread.

        When require_no_lick is True, a trial also can't start while Port1 has been recently
        active: since Bpod has no way to observe a digital input outside a running state machine,
        the foreground repeatedly sends a tiny HoldCheck/LickDuringHold machine (state_timer=
        required_hold) and only returns once a full required_hold-length chunk passes with no lick
        *and* the wheel-steadiness window is independently satisfied. The two checks aren't on the
        same clock (the lick check is forward-looking from when each chunk starts; the wheel check
        is the trailing window, as before) -- looping until both hold at once converges to
        "neither happened in at least the last required_hold seconds" without a more complex
        synchronized dual-window implementation. When require_no_lick is False, no Bpod machine is
        sent at all during this wait -- the two branches differ ONLY in whether a Bpod machine
        also runs concurrently; the wheel-steadiness window/return contract is identical either
        way.

        Returns the wheel break count on success, or None if a Stop/Kill interrupted the lick-check
        loop (require_no_lick=True only) -- callers should check for None and stop their own trial
        loop the same way they already do when run_trial_state_machine() returns False.

        Registers 'TRIAL_INIT_START' once, right at entry (the moment this trial's quiescence-hold
        attempt begins), and a 'QUIESCENCE_RESET_TIME' row each time the trailing window is broken
        by movement -- both on the same time.time() - log_python_t0 clock as every other VAL row,
        so a trial's own initiation cost (and how many resets it took) is directly recoverable from
        the session CSV without having to reconstruct it from HoldCheck/LickDuringHold state times.
        """
        self.register('TRIAL_INIT_START', self._now())

        hw = _HoldWatch(required_hold, steady_threshold_deg)
        self._hold_watch = hw
        try:
            if not require_no_lick:
                while not hw.wheel_ready.is_set():
                    time.sleep(1.0 / self.still_poll_hz)
                return hw.n_breaks_holder[0]

            while True:
                hold_sma = StateMachine(self.bpod)
                hold_sma.add_state(
                    state_name='HoldCheck',
                    state_timer=required_hold,
                    state_change_conditions={'Port1In': 'LickDuringHold', Bpod.Events.Tup: 'exit'},
                    output_actions=[])
                hold_sma.add_state(
                    state_name='LickDuringHold',
                    state_timer=0,
                    state_change_conditions={Bpod.Events.Tup: 'exit'},
                    output_actions=[])

                if not self.run_trial_state_machine(hold_sma):
                    return None

                visited = self.bpod.session.current_trial.states_durations
                licked = was_visited(visited, 'LickDuringHold')
                if not licked and hw.wheel_ready.is_set():
                    return hw.n_breaks_holder[0]
        finally:
            self._hold_watch = None
