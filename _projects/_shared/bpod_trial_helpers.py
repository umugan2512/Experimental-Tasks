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
the background WHEEL_POS-publishing thread, the bounded-join wrapper around run_state_machine(),
the states_durations nan-check, the hold-to-init wait (wheel steadiness, optionally also gated on
no recent licking), and port-event (lick) reporting.

**REVERTED (see git history): a session-long, continuously-running poll thread design was tried
here briefly, to close a confirmed wheel-logging blind window at trial onset, but caused the task
to fail to start reliably on real hardware (no repro possible without the rig, root cause not
pinned down -- most likely a continuously-running background thread hammering the rotary's serial
connection for the ENTIRE session interfering with other rotary commands issued from the main
thread, where previously there were natural gaps between trials with no polling at all). Reverted
back to the original, previously-proven-working per-call-scoped thread lifecycle below. If this
is revisited, test the threading change in isolation on real hardware BEFORE combining it with
any other change (e.g. require_no_lick=True, also reverted here for the same reason -- it was
genuinely dead code before this session, never exercised on this hardware, and should not be
re-enabled bundled with other unproven changes). start()/stop() are kept as harmless no-ops so
callers that already added them don't need to be re-edited.
"""
import math
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


class TrialRunner(object):
    """
    Wraps a connected Bpod + rotary encoder pair with the trial-loop plumbing every wheel-based
    protocol in this project has needed so far. One instance per task script, constructed once
    connections are up: `runner = TrialRunner(my_bpod, rotary, log_python_t0)`.
    """

    def __init__(self, bpod, rotary, log_python_t0, still_poll_hz=100, poll_hz=100):
        self.bpod = bpod
        self.rotary = rotary
        self.log_python_t0 = log_python_t0
        self.still_poll_hz = still_poll_hz
        self.poll_interval = 1.0 / poll_hz

        self.rotary_lock = threading.Lock()
        self.register_lock = threading.Lock()
        self._stale_poll_threads = []
        self.last_send_t = None   # see run_trial_state_machine()'s own docstring

    def start(self):
        """ No-op (see module docstring -- the session-long poll thread this used to start was
        reverted). Kept so existing call sites don't need to be removed. """
        pass

    def stop(self):
        """ No-op (see module docstring). Kept so existing call sites don't need to be removed. """
        pass

    def register(self, key, value, flush=True):
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
        self.register('WHEEL_POS', '{0:.3f},{1:.2f}'.format(t, pos))

    def _trial_poll_loop(self, stop_event):
        """ Publishes WHEEL_POS continuously for whichever Bpod call is currently running.
        Wrapped in a broad try/except: a Kill/Stop can tear down the Bpod session out from under
        this thread mid-register_value() (observed on hardware: AttributeError on bpod._session
        after a mid-trial Stop), which is harmless in effect -- the process is ending either way
        -- but was spraying an alarming traceback; fail silently instead.

        stop_event is a fresh threading.Event() created per call by run_trial_state_machine(), not
        a shared instance attribute -- see that method's docstring for why a shared one is unsafe. """
        while not stop_event.is_set():
            try:
                with self.rotary_lock:
                    p = self.rotary.current_position()
                self._publish_wheel_pos(p)
            except Exception:
                break
            time.sleep(self.poll_interval)

    def run_trial_state_machine(self, sma):
        """ Bounded-join wrapper around send/run_state_machine(): Kill's own handling
        (run_state_machine() calls exit(0) from inside once it reads a kill command off stdin)
        unwinds through any try/finally wrapped around the call -- an UNBOUNDED join() here on the
        background poll thread would risk Kill hanging indefinitely if that thread is ever
        mid-stall talking to hardware, so give it a timeout instead.

        Each call gets its OWN fresh stop_event (a local variable), not a shared
        self.poll_thread_stop reused across calls -- a shared Event was a latent bug: if a
        previous trial's poll thread ever failed to join within the 3s timeout below (left
        dangling, still alive), the NEXT trial's clear() on that same shared Event would silently
        un-signal the stale thread too, letting it keep running indefinitely alongside the new
        one. Scoping the Event per call means a stale thread's own stop signal is never touched by
        a later call -- it either exits on its own once its still-referenced Event is set below, or
        stays tracked in self._stale_poll_threads for visibility (see CLAUDE.md's note on the
        PyEval_RestoreThread crash this was investigated for).

        Also captures `self.last_send_t` right after the upload (send_state_machine()) completes
        and right before the actual RUN command -- the closest anchor to the real hardware
        trial-start instant reachable without instrumenting pybpod-api itself. Callers should
        re-anchor any send_t/cue_send_t-based VAL registration to `runner.last_send_t` once this
        call returns, instead of whatever earlier time.time() they captured before building the
        StateMachine -- that earlier value doesn't account for SM-construction + upload +
        RUN-handshake latency, a confirmed source of drift between derived event timestamps and
        the hardware-relative offsets they were added to. """
        stop_event = threading.Event()
        t = threading.Thread(target=self._trial_poll_loop, args=(stop_event,), daemon=True)
        t.start()
        try:
            self.bpod.send_state_machine(sma)
            self.last_send_t = self._now()
            ran = self.bpod.run_state_machine(sma)
        finally:
            stop_event.set()
            t.join(timeout=3.0)
            if t.is_alive():
                self._stale_poll_threads.append(t)
                print("WARNING: wheel-position poll thread did not stop within 3s -- continuing "
                      "anyway ({0} stale poll thread(s) accumulated this session).".format(
                          len(self._stale_poll_threads)), flush=True)
        return ran

    def wait_for_held_steady(self, required_hold, steady_threshold_deg, require_no_lick=True):
        """
        Blocks until the wheel has stayed within steady_threshold_deg (peak-to-peak) for a
        continuous trailing window of required_hold seconds -- a rolling-window check, not a
        threshold-crossing one, since a Bpod self-looping state can't reproduce "hasn't moved more
        than X degrees at any point in the last N seconds" (only "did a crossing happen"). This
        runs in a background thread exactly as it always has, polling rotary.current_position()
        at still_poll_hz.

        When require_no_lick is True, a trial also can't start while Port1 has been recently
        active: since Bpod has no way to observe a digital input outside a running state machine,
        the foreground repeatedly sends a tiny HoldCheck/LickDuringHold machine (state_timer=
        required_hold) and only returns once a full required_hold-length chunk passes with no lick
        *and* the wheel thread's own window is independently satisfied. The two checks aren't on
        the same clock (the lick check is forward-looking from when each chunk starts; the wheel
        check is the trailing window, as before) -- looping until both hold at once converges to
        "neither happened in at least the last required_hold seconds" without a more complex
        synchronized dual-window implementation.

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

        wheel_ready = threading.Event()
        stop_wheel_watch = threading.Event()
        n_breaks_holder = [0]

        def wheel_watch():
            window = deque()
            wait_start = time.time()
            warned = [False]
            while not stop_wheel_watch.is_set():
                time.sleep(1.0 / self.still_poll_hz)
                with self.rotary_lock:
                    pos = self.rotary.current_position()
                now = time.time()
                if window and abs(pos - window[-1][1]) > steady_threshold_deg:
                    n_breaks_holder[0] += 1
                    self.register('QUIESCENCE_RESET_TIME', now - self.log_python_t0)
                window.append((now, pos))
                while len(window) > 1 and window[1][0] <= now - required_hold:
                    window.popleft()
                self._publish_wheel_pos(pos, now - self.log_python_t0)
                if not warned[0] and (now - wait_start) > 3.0 + required_hold:
                    warned[0] = True
                    print("Waiting for the wheel to hold steady (current position: "
                          "{0:.1f}deg)...".format(pos), flush=True)
                positions = [p for _, p in window]
                has_full_span = window[0][0] <= now - required_hold
                if has_full_span and (max(positions) - min(positions)) <= steady_threshold_deg:
                    wheel_ready.set()
                else:
                    wheel_ready.clear()

        t = threading.Thread(target=wheel_watch, daemon=True)
        t.start()
        try:
            if not require_no_lick:
                while not wheel_ready.is_set():
                    time.sleep(1.0 / self.still_poll_hz)
                return n_breaks_holder[0]

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
                if not licked and wheel_ready.is_set():
                    return n_breaks_holder[0]
        finally:
            stop_wheel_watch.set()
            t.join(timeout=3.0)
