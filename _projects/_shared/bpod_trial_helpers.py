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

**Two wheel-position modes** (chosen per script via `TrialRunner(..., use_stream=...)`):

- `use_stream=True` (the AuditoryEvidenceAccum stage scripts): the rotary's own hardware-timestamped
  USB stream (`rotary_stream.RotaryStream`, MATLAB Bpod_Gen2's approach) runs for the whole session.
  Its reader thread only ever READS the port, so it can't collide with anything -- unlike the
  session-long 'Q'-polling thread that was tried and reverted, which WROTE on the same connection
  as the main thread and crashed real sessions with SerialTimeoutException. The wheel is logged
  continuously: no blind window at trial onset, and sample times come from the module's clock, not
  host scheduling. In this mode, NEVER call `rotary.current_position()` or any other ack-reading
  rotary command directly: use `runner.current_position()`, and wrap ack-reading commands in
  `with runner.stream.paused():`.
- `use_stream=False` (default; every Tests/ script, unchanged): the original per-Bpod-call 'Q'
  polling thread below, exactly as before.

`upload()` + `run_trial_state_machine()` separate the state machine upload from RUN (MATLAB's
SendStateMachine-then-RunStateMachine): a stage script uploads the trial's machine BEFORE its
hold-to-init wait, so RUN follows the end of quiescence within milliseconds instead of after a
0.2-1.4s construction+upload gap (measured on real Stage 1 sessions) in which nothing checked the
wheel. `softcode_seen()`/`wait_softcode()` expose Bpod SoftCode outputs (`('SoftCode', n)` in a
state's output_actions -- delivered to `softcode_handler_function` while `run_state_machine()` is
reading the Bpod connection), which is how the render loop learns the exact moment the dot-onset
and reward states are entered instead of guessing from a host timer.

require_no_lick=True remains dead code on purpose (every caller passes False) -- see CLAUDE.md
before re-enabling it.
"""
import math
import sys
import threading
import time
from collections import deque

from pybpodapi.protocol import Bpod, StateMachine

import rotary_setup
from rotary_stream import RotaryStream

_WIN_TICK_S = 0.0156 if sys.platform.startswith('win') else 0.0   # Event.wait() timeout granularity
BPOD_LOOP_SLEEP_S = 0.005        # per-iteration sleep in pybpod's run loop (stream mode) -- see
                                  # TrialRunner.start() for the rig measurement behind this value
WHEEL_LOG_MIN_INTERVAL_S = 0.05   # stream mode: a WHEEL_POS row is written whenever the position
                                  # changes, and at least this often while it holds still
WHEEL_LOG_PUMP_INTERVAL_S = 0.025 # how often pump() writes the buffered stream samples to the CSV
                                  # (also the keepalive cadence while the wheel is still)


# PC -> Bpod events (TrialRunner.inject_softcode()). MATLAB's SendBpodSoftCode(N) and pybpod's
# handle_inline('SoftCodeN') both send byte N-1, which the state machine reports as input event
# 'SoftCodeN' (confirmed against Bpod_Gen2's Functions/SendBpodSoftCode.m). TrialRunner renames
# those entries of the Bpod's event-name list to what they mean (names are positional, so only
# the label changes -- in state machines and in every EVENT row of the CSV).
PC_EVENT_CODES = {
    'PC_QuiescenceMet': 1,   # the wheel was still for the hold window
    'PC_DotShown': 2,        # the dot has been painted on screen
}
PC_EVENT_QUIESCENCE_MET = 'PC_QuiescenceMet'
PC_EVENT_DOT_SHOWN = 'PC_DotShown'
PC_SOFTCODE = PC_EVENT_CODES[PC_EVENT_QUIESCENCE_MET]   # kept for older callers
PC_SOFTCODE_EVENT = PC_EVENT_QUIESCENCE_MET


def _name_pc_events(bpod):
    """ Renames the Bpod's 'SoftCodeN' input events listed in PC_EVENT_CODES to their meaning.
    No-op if the hardware description isn't available (e.g. a test double). """
    names = getattr(getattr(getattr(bpod, 'hardware', None), 'channels', None), 'event_names', None)
    if not names:
        return
    for name, code in PC_EVENT_CODES.items():
        generic = 'SoftCode{0}'.format(code)
        if generic in names:
            names[names.index(generic)] = name


def install_session_lock():
    """ Makes pybpodapi's Session.add_message() thread-safe (patched once per process; idempotent).

    add_message() does `self.current_trial += msg`; through the `current_trial` property that is
    get trials[-1] -> __add__ -> setter `trials[-1] = value`, not atomic. When the Bpod thread
    appends a new Trial between another thread's get and set, the setter writes the PREVIOUS trial
    object back over the new one. The new trial's events then land in the old trial's list, and
    pybpod's own __update_timestamps() indexes past its timestamp list -> IndexError. Confirmed:
    session 20261023-120425 crashed this way at trial 5, with the render loop registering
    TRIAL_INIT_START as the trial started. One lock around add_message() serializes every session
    write -- the Bpod thread's (trials, events, states), the main thread's register_value()s, and
    print() via pybpod's StdoutBuffer -- so the get/set pair can no longer interleave with a new
    Trial. Behaviour is otherwise unchanged. """
    try:
        from pybpodapi import session as _session
    except ImportError:
        return   # a test double standing in for pybpodapi (no real Session to protect)
    if getattr(_session.Session.add_message, '_lab_locked', False):
        return
    original = _session.Session.add_message
    lock = threading.RLock()   # reentrant: a print() from inside a write must not deadlock

    def add_message(self, msg, flush=True):
        with lock:
            return original(self, msg, flush=flush)

    add_message._lab_locked = True
    _session.Session.add_message = add_message


def _set_windows_timer_resolution(enable):
    """ timeBeginPeriod(1)/timeEndPeriod(1) on Windows; no-op (False) elsewhere or on failure. """
    if not sys.platform.startswith('win'):
        return False
    try:
        import ctypes
        winmm = ctypes.WinDLL('winmm')
        (winmm.timeBeginPeriod if enable else winmm.timeEndPeriod)(1)
        return True
    except Exception:
        return False


def _set_windows_process_priority(high):
    """ HIGH_PRIORITY_CLASS (or back to NORMAL) for this process on Windows; False elsewhere or on
    failure. Deliberately not REALTIME_PRIORITY_CLASS, which can starve the OS's own input/USB
    handling. """
    if not sys.platform.startswith('win'):
        return False
    try:
        import ctypes
        k32 = ctypes.windll.kernel32
        return bool(k32.SetPriorityClass(k32.GetCurrentProcess(), 0x80 if high else 0x20))
    except Exception:
        return False


class _InjectableStdin(object):
    """ Stands in for `bpod.stdin` (pybpod's NonBlockingStreamReader over sys.stdin, or None).
    run_state_machine() polls `self.stdin.readline()` on every loop iteration and acts on the line
    itself (bpod_base.py's handle_inline()) -- including 'SoftCodeN', which makes THAT loop send
    soft code N to the state machine. So a line put here is the one thread-safe way for another
    thread to signal into a running trial: the write to the Bpod connection still happens on the
    thread that owns it. Injected lines are returned first; the real stdin (the GUI's Stop/Kill)
    is still read every iteration after that. """

    def __init__(self, inner):
        self._inner = inner
        self._lines = deque()
        self._lock = threading.Lock()

    def put(self, line):
        with self._lock:
            self._lines.append(line)

    def readline(self, timeout=None):
        with self._lock:
            if self._lines:
                return self._lines.popleft()
        return self._inner.readline(timeout) if self._inner is not None else None

    def close(self):
        if self._inner is not None:
            self._inner.close()

    def is_alive(self):
        return self._inner.is_alive() if self._inner is not None else True


def triggering_event(events_dict, state_start, candidates):
    """ Which of `candidates` (event names) caused the transition into a state that started at
    `state_start` (trial-relative seconds) -- the one whose timestamp equals that state's start.
    Returns None if none matches (e.g. a Tup-entered state).

    Needed because the threshold that ENDS a wheel period isn't necessarily the first one that
    fired in the trial: a crossing in a state that wasn't listening for it is still logged, and
    the firmware disarms that threshold, so a later crossing of the OTHER threshold may be the one
    Bpod acted on. Confirmed on real Stage 1 sessions: picking the first (or last) crossing by
    timestamp mislabeled the side on every such trial. """
    if state_start is None or (isinstance(state_start, float) and math.isnan(state_start)):
        return None
    best, best_dt = None, None
    for name in candidates:
        for t in events_dict.get(name, []):
            dt = abs(t - state_start)
            if dt < 1e-4 and (best_dt is None or dt < best_dt):
                best, best_dt = name, dt
    return best


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

    def __init__(self, bpod, rotary, log_python_t0, still_poll_hz=100, poll_hz=100,
                 use_stream=False, marker_code=None):
        self.bpod = bpod
        self.rotary = rotary
        self.log_python_t0 = log_python_t0
        self.still_poll_hz = still_poll_hz
        self.poll_interval = 1.0 / poll_hz

        self.rotary_lock = threading.Lock()
        self.register_lock = threading.Lock()
        self._stale_poll_threads = []
        self.last_send_t = None   # see run_trial_state_machine()'s own docstring
        self.last_run_t = None
        self.last_return_t = None        # when the previous Bpod call returned (inter-trial gap)

        self.use_stream = use_stream
        self.stream = RotaryStream(rotary, log_python_t0) if use_stream else None
        self.marker_code = marker_code   # '#' code the trial's first state sends (stream mode)
        self.last_marker = None          # (module_t, host_t) of that marker, this trial
        self._uploaded_sma = None
        self._wheel_log_last = None      # (host_t, pos) of the last WHEEL_POS row written
        self._wheel_log_pumped_at = 0.0
        self._wheel_log_lock = threading.Lock()

        self._softcode_lock = threading.Lock()
        self._softcodes = {}             # code -> host time received, this trial
        self._softcode_event = threading.Event()
        self._loop_wake = threading.Event()
        self.trial_t0 = 0.0
        bpod.softcode_handler_function = self._on_softcode
        install_session_lock()
        _name_pc_events(bpod)
        self._hw_run_t = None            # host time the RUN byte was written (see _hook_run_write)
        self.run_write_ms = None
        self._hook_run_write()
        self._stdin = _InjectableStdin(getattr(bpod, 'stdin', None))
        bpod.stdin = self._stdin
        self._quiescence = None
        self.inject_log = []             # (code, host_t) injected this trial

    def start(self):
        """ Stream mode: starts the session-long rotary USB stream (see module docstring). No-op
        otherwise, kept so every call site can stay as-is. """
        if self.stream is not None:
            self.stream.start()
            # pybpod's run_state_machine() loop otherwise polls the Bpod's USB serial port
            # (in_waiting) continuously. Measured on the rig (scratchpad stall_diag.py, 120s of
            # Bpod + dot render + camera): with sleep(0) here, EVERY thread in the process --
            # including one doing nothing but sleep(1ms) -- regularly woke 25-93ms late (74 stalls
            # >25ms in 120s; the audit's "rotary reader stalls" of up to 123ms). With a real 2ms
            # sleep: 3 stalls; with 5ms: none (max gap 19ms). Event TIMING is unaffected (Bpod
            # timestamps events in hardware); only the PC reacts up to BPOD_LOOP_SLEEP_S later.
            self.bpod.loop_handler = self._loop_sleep
            sys.setswitchinterval(0.0005)
            # Windows' default timer resolution is 15.6ms: every time.sleep() in the reader, render
            # and hold loops was really ~15.6ms, quantizing reaction times to 16/32/48ms. 1ms for
            # the session (what PsychoPy does), undone in stop(). (It does NOT fix Event.wait()
            # timeouts, which keep a ~15.6ms floor -- so the timing-critical waits here use 1ms
            # time.sleep() slices instead.)
            self._win_timer = _set_windows_timer_resolution(True)
            # The residual stalls left after the loop sleep hit every thread at once (OS
            # scheduling). High priority class for the task process (not REALTIME), the reader
            # thread itself goes ABOVE_NORMAL (rotary_stream): measured back to back on the rig,
            # 120s each -- normal: 4 stalls/max 60ms and 2/27ms (a 1ms sleeper woke 58-77ms late);
            # high: 2/33.5ms and 0/25ms (sleeper <= 26ms).
            self._win_priority = _set_windows_process_priority(True)

    def stop(self):
        """ Stream mode: writes any not-yet-logged wheel samples, then stops the stream -- call
        before rotary.close() and before my_bpod.close(). No-op otherwise. """
        if self.stream is not None:
            try:
                self.flush_wheel_log()
            except Exception:
                pass
            self.stream.stop()
            if getattr(self, '_win_timer', False):
                _set_windows_timer_resolution(False)
            if getattr(self, '_win_priority', False):
                _set_windows_process_priority(False)

    def register(self, key, value, flush=True):
        with self.register_lock:
            self.bpod.register_value(key, value, flush=flush)

    # --- wheel position ----------------------------------------------------------------------------

    def current_position(self):
        """ Latest wheel position in degrees. Stream mode: the most recent streamed sample (0.0
        before any has arrived -- the firmware only sends a sample when the position changes, so a
        wheel that hasn't moved since the last zero hasn't sent anything). Otherwise: a 'Q' request
        under rotary_lock, exactly what every caller used to do by hand. """
        if self.stream is None:
            with self.rotary_lock:
                return self.rotary.current_position()
        latest = self.stream.latest()
        return 0.0 if latest is None else latest[2]

    def current_position_since_host(self, host_t):
        """ Stream mode: latest position among samples received at/after host_t (None if there
        are none yet) -- lets a render loop ignore samples from before the trial's hardware
        re-zero, so the dot never flashes at a stale pre-zero offset. """
        samples = self.stream.samples_since_host(host_t)
        return samples[-1][2] if samples else None

    def last_marker_seen(self):
        """ Stream mode: has this trial's '#' marker arrived yet (i.e. is the response-window zero
        in effect)? """
        return (self.stream is not None and self.marker_code is not None and
                self.stream.find_event(self.marker_code, after_host_t=self.trial_t0) is not None)

    def position_after_marker(self):
        """ Stream mode: the latest wheel position acquired (module clock) at or after this
        trial's '#' marker -- i.e. strictly in the frame the trial's first state just zeroed.
        None until the marker has arrived (callers should treat that as 0: the zero is happening
        right then) or if there's no sample since it (the wheel hasn't moved off the zero). """
        if self.stream is None or self.marker_code is None:
            return None
        ev = self.stream.find_event(self.marker_code, after_host_t=self.trial_t0)
        if ev is None:
            return None
        samples = self.stream.samples_since_host(ev[1] - 0.5)
        after = [p for m, _, p in samples if m >= ev[0]]
        return after[-1] if after else None

    def flush_wheel_log(self):
        """ Stream mode: writes every buffered sample as a `WHEEL_POS` row, `t,pos,t_module` -- t is
        the sample's module acquisition time mapped onto the host clock (the first two fields are
        what every consumer reads; t_module lets session_csv_parser.wheel_samples_bpod_time() put
        each sample on the Bpod clock exactly, via the trial's ROTARY_MARKER). Decimated to
        "position changed, or WHEEL_LOG_MIN_INTERVAL_S elapsed".

        Keepalive: the firmware only streams when the position CHANGES, so a still wheel used to
        leave multi-second holes (up to 10.8s, session 20261022-162023) indistinguishable from a
        stalled logger. If nothing new arrived and the last row is older than
        WHEEL_LOG_MIN_INTERVAL_S, the current (unchanged) position is written at "now" -- a gap in
        WHEEL_POS therefore always means a real logging dropout.

        Rows are kept in nondecreasing time order. One `WHEEL_CLOCK_OFFSET` row (host minus module
        seconds) per flush records the mapping used. No-op in polling mode. """
        if self.stream is None:
            return
        with self._wheel_log_lock:
            t_lock = time.time()
            samples = self.stream.drain_for_logging()
            offset = self.stream.module_to_host(0.0)
            last = self._wheel_log_last
            if not samples:
                now = self._now()
                if offset is None or (last is not None and now - last[0] < WHEEL_LOG_MIN_INTERVAL_S):
                    return
                # No new sample means the position has not changed since the last one (the
                # firmware streams every change) -- fill the elapsed time at the keepalive cadence,
                # so the log has no gap even if this flush ran late.
                pos = self.current_position()
                if last is None:
                    samples = [(now - offset, now, pos)]
                else:
                    n = int((now - last[0]) / WHEEL_LOG_MIN_INTERVAL_S)
                    samples = [(last[0] + k * WHEEL_LOG_MIN_INTERVAL_S - offset,
                                last[0] + k * WHEEL_LOG_MIN_INTERVAL_S, pos) for k in range(1, n + 1)]
            for module_t, host_t, pos in samples:
                if last is not None:
                    # The firmware streams every position change, so between the previous row and
                    # this sample the wheel sat at last[1]: fill that stretch at the keepalive
                    # cadence (also covers a flush that ran late, or a busy main thread).
                    k = 1
                    while host_t - (last[0] + k * WHEEL_LOG_MIN_INTERVAL_S) > 1e-6:
                        fill_t = last[0] + k * WHEEL_LOG_MIN_INTERVAL_S
                        self.register('WHEEL_POS', '{0:.4f},{1:.2f},{2:.6f}'.format(
                            fill_t, last[1], fill_t - (offset or 0.0)), flush=False)
                        k += 1
                    if k > 1:
                        last = (last[0] + (k - 1) * WHEEL_LOG_MIN_INTERVAL_S, last[1])
                    if pos == last[1] and host_t - last[0] < WHEEL_LOG_MIN_INTERVAL_S - 1e-6:
                        continue
                    host_t = max(host_t, last[0])
                self.register('WHEEL_POS', '{0:.4f},{1:.2f},{2:.6f}'.format(host_t, pos, module_t),
                              flush=False)
                last = (host_t, pos)
            self._wheel_log_last = last
            if offset is not None:
                self.register('WHEEL_CLOCK_OFFSET', '{0:.6f}'.format(offset))
            self._wheel_log_pumped_at = time.time()
            self.wheel_log_lock_max_ms = max(getattr(self, 'wheel_log_lock_max_ms', 0.0),
                                             (time.time() - t_lock) * 1e3)

    def pump(self):
        """ Call from any render/wait loop: writes buffered wheel samples at most every
        WHEEL_LOG_PUMP_INTERVAL_S, so the live dashboard's wheel plot keeps updating mid-trial.
        (Registering from the main thread while the decision thread runs the Bpod call is the same
        concurrency the old per-call poll thread always had.) """
        if self.stream is not None and time.time() - self._wheel_log_pumped_at >= \
                WHEEL_LOG_PUMP_INTERVAL_S:
            self.flush_wheel_log()

    # --- softcodes ---------------------------------------------------------------------------------

    def _on_softcode(self, code):
        """ Runs on whichever thread is inside run_state_machine() (it's called from that loop when
        the SoftCode opcode arrives) -- only records and signals, never touches hardware. """
        with self._softcode_lock:
            self._softcodes.setdefault(code, self._now())
        self._softcode_event.set()

    def _hook_run_write(self):
        """ Timestamps the RUN command where pybpod actually writes it: wraps this Bpod instance's
        _bpodcom_run_state_machine() (one write_char('R')). The anchor used to be taken before
        calling run_state_machine(), which first adds the Trial to the session log -- a step that
        can wait on the session lock while the main thread writes wheel rows, so RUN sometimes
        left several ms after the anchor (an 8ms clock-fit outlier in session 20261027-095504).
        No-op if the Bpod object has no such method (test doubles). """
        original = getattr(self.bpod, '_bpodcom_run_state_machine', None)
        if original is None:
            return

        def timed_run():
            t = self._now()
            original()
            self.run_write_ms = (self._now() - t) * 1e3
            self._hw_run_t = t

        self.bpod._bpodcom_run_state_machine = timed_run

    def _loop_sleep(self):
        """ pybpod's run-loop hook (see start()): one plain BPOD_LOOP_SLEEP_S time.sleep().
        Measured on the rig (scratchpad stall_diag.py / handshake_rig.py, 120s of Bpod + dot render
        + camera): this gives zero reader stalls > 25ms with a 21ms DotOn -> response handshake.
        Waking early on an injected SoftCode by sleeping in 1ms slices shaved ~3ms off the
        handshake but let 2 stalls of up to 48ms back in; Event.wait() can't be used for the
        early wake because its timeout has a ~15.6ms floor on Windows/Python 3.6. """
        time.sleep(BPOD_LOOP_SLEEP_S)

    def inject_softcode(self, code=PC_SOFTCODE):
        """ Signals the RUNNING state machine: it receives input event 'SoftCode<code>' -- named
        by PC_EVENT_CODES, e.g. inject_softcode(PC_EVENT_DOT_SHOWN) (see _InjectableStdin). No
        trailing newline -- handle_inline() parses the last character. """
        if not isinstance(code, int):
            code = PC_EVENT_CODES[code]
        self._stdin.put('SoftCode{0}'.format(int(code)))
        self._loop_wake.set()
        self.inject_log.append((code, self._now()))

    def wait_signal(self, timeout):
        """ Sleeps up to `timeout`, waking immediately if a SoftCode arrives from the state
        machine -- render loops use this instead of a fixed sleep so the dot reacts within ~1ms. """
        # Event.wait() wakes the instant a SoftCode arrives. Its TIMEOUT, though, rounds up to the
        # ~15.6ms Windows tick (see _loop_sleep()): a 33ms wait really took ~47ms (21 frames/s), so
        # one tick is taken off -- 33ms -> ~31ms (32 frames/s). Not 1ms time.sleep() slices here:
        # measured on the rig, slicing both this and the pybpod loop brought back whole-process
        # stalls (7 > 25ms in 120s, max 53ms) that slicing the pybpod loop alone does not (2, max
        # 39ms).
        if self._softcode_event.wait(max(0.001, timeout - _WIN_TICK_S)):
            self._softcode_event.clear()

    def softcode_seen(self, code):
        """ Host time SoftCode `code` arrived during the current trial, or None. """
        with self._softcode_lock:
            return self._softcodes.get(code)

    def begin_trial(self):
        """ Clears last trial's SoftCodes/marker and starts this trial's signal window. Call on the
        MAIN thread BEFORE starting the thread that runs the Bpod machine: if the reset happened
        inside that thread (as it used to), the render loop's first frames still saw the previous
        trial's SoftCodes and marker and froze the dot at a stale position before the trial began
        (confirmed on session 20261022-150327: 10 of 13 trials). """
        with self._softcode_lock:
            self._softcodes = {}
        self._softcode_event.clear()
        self.last_marker = None
        self.trial_t0 = self._now()
        self.inject_log = []

    def dot_onset_signal(self, softcode):
        """ Host time the trial's dot-onset state was entered, from whichever hardware signal
        arrives first: SoftCode `softcode` (via the Bpod connection), or (stream mode) the '#'
        marker that same state sends to the rotary. Both are fired by the same state entry, so
        either is correct; accepting both means a mismatch in how the firmware reports one of
        them can delay, but never suppress, the dot. None until either has arrived. """
        t = self.softcode_seen(softcode)
        if t is not None:
            return t
        if self.stream is not None and self.marker_code is not None:
            ev = self.stream.find_event(self.marker_code, after_host_t=self.trial_t0)
            if ev is not None:
                return ev[1]
        return None

    # --- state machines ----------------------------------------------------------------------------

    def upload(self, sma):
        """ Sends `sma` to the Bpod now (MATLAB's SendStateMachine) so the next
        run_trial_state_machine(sma) call only has to issue RUN. """
        self.bpod.send_state_machine(sma)
        self._uploaded_sma = sma

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

    def run_trial_state_machine(self, sma, arm_rotary=False, expect_marker=True, marker_state=None):
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
        the hardware-relative offsets they were added to.

        If `sma` was already sent with upload(), it isn't re-sent: `last_send_t`/`last_run_t` are
        then captured right before RUN instead (the closest host anchor to the hardware trial start
        reachable without instrumenting pybpod-api -- measured 4ms residual SD against Bpod's own
        TrialStartTimestamp on real sessions). Stream mode runs no poll thread at all (the stream
        logs the wheel continuously) and, afterwards, registers `ROTARY_MARKER` (`module_t,host_t`
        of this trial's '#' marker) when `marker_code` was given.

        `arm_rotary=True` (stream mode) re-zeroes and re-arms every rotary threshold over USB
        (rotary_setup.arm_for_trial(), no reply bytes, so stream-safe) immediately before RUN.
        `expect_marker=False` skips the ROTARY_MARKER lookup for a machine that sends no marker
        (e.g. Stage 3-5's cue_sma) -- otherwise its absence would cost the 0.3s lookup timeout.
        `marker_state` (the state whose entry sends the marker): the lookup only waits if that state
        was actually entered -- a trial that ended early (MovedBeforeDot/DotTimeout) never sent one,
        and waiting for it cost 0.3s of unmonitored time between machines (soak test). """
        if self.stream is not None:
            try:
                if self._uploaded_sma is not sma:
                    self.bpod.send_state_machine(sma)
                self._uploaded_sma = None
                if arm_rotary:
                    rotary_setup.arm_for_trial(self.rotary)
                self.flush_wheel_log()   # keeps WHEEL_POS gap-free across the between-machine gap
                self._hw_run_t = None
                self.last_send_t = self.last_run_t = self._now()   # fallback anchor
                ran = self.bpod.run_state_machine(sma)
            finally:
                self._uploaded_sma = None
                self.last_return_t = self._now()
                if self._hw_run_t is not None:
                    self.last_send_t = self.last_run_t = self._hw_run_t   # exact: RUN's write
            if marker_state is not None:
                try:
                    expect_marker = was_visited(self.bpod.session.current_trial.states_durations,
                                                marker_state)
                except Exception:
                    pass
            if self.marker_code is not None and expect_marker:
                self.last_marker = self._wait_marker(timeout=0.3)
                if self.last_marker is not None:
                    self.register('ROTARY_MARKER', '{0:.6f},{1:.4f}'.format(*self.last_marker))
            self.flush_wheel_log()
            return ran

        stop_event = threading.Event()
        t = threading.Thread(target=self._trial_poll_loop, args=(stop_event,), daemon=True)
        t.start()
        try:
            if self._uploaded_sma is not sma:
                self.bpod.send_state_machine(sma)
            self._uploaded_sma = None
            self.last_send_t = self.last_run_t = self._now()
            ran = self.bpod.run_state_machine(sma)
        finally:
            self._uploaded_sma = None
            stop_event.set()
            t.join(timeout=3.0)
            if t.is_alive():
                self._stale_poll_threads.append(t)
                print("WARNING: wheel-position poll thread did not stop within 3s -- continuing "
                      "anyway ({0} stale poll thread(s) accumulated this session).".format(
                          len(self._stale_poll_threads)), flush=True)
        return ran

    def _wait_marker(self, timeout):
        """ (module_t, host_t_mapped) of this trial's '#' marker, waiting briefly for USB delivery.
        None if it never showed up (a trial that ended before its first state's output fired, or a
        firmware that doesn't report markers -- see Tests/tasks/rotary_stream_test/). """
        deadline = time.time() + timeout
        while True:
            ev = self.stream.find_event(self.marker_code, after_host_t=self.trial_t0)
            if ev is not None:
                host_t = self.stream.module_to_host(ev[0])
                return ev[0], (ev[1] if host_t is None else host_t)
            if time.time() >= deadline:
                return None
            time.sleep(0.005)

    def start_quiescence(self, required_hold, steady_threshold_deg):
        """ Stream mode, non-blocking: begins this trial's hold-to-init check (registers
        TRIAL_INIT_START). Call poll_quiescence() every render-loop frame WHILE the trial's state
        machine sits in WaitQuiescence -- so the hold is inside a running Bpod machine (licks and
        Stop/Kill are processed throughout) instead of in a gap between machines. Same criterion as
        wait_for_held_steady(). """
        now = self._now()
        self.register('TRIAL_INIT_START', now)
        self._quiescence = {'hold': required_hold, 'thr': steady_threshold_deg, 'start': now,
                            'breaks': 0, 'moving': False, 'met_t': None}

    def poll_quiescence(self):
        """ True once the wheel's peak-to-peak range over the trailing hold window is within the
        threshold (and at least one full hold has elapsed since start_quiescence()). Registers a
        QUIESCENCE_RESET_TIME row each time movement restarts the window. """
        q = self._quiescence
        if q is None:
            return False
        if q['met_t'] is not None:
            return True
        now = self._now()
        if self._is_moving(now, q['hold'], q['thr']):
            if not q['moving']:
                q['breaks'] += 1
                self.register('QUIESCENCE_RESET_TIME', now)
            q['moving'] = True
            return False
        q['moving'] = False
        if now - q['start'] >= q['hold']:
            q['met_t'] = now
            return True
        return False

    @property
    def quiescence_breaks(self):
        return self._quiescence['breaks'] if self._quiescence else 0

    @property
    def quiescence_met_t(self):
        return self._quiescence['met_t'] if self._quiescence else None

    def _is_moving(self, now, required_hold, steady_threshold_deg):
        """ Peak-to-peak wheel range over [now - required_hold, now] > threshold, on streamed
        samples placed at their module-clock times (the position at the window's start is the last
        sample before it -- the firmware only sends on change). """
        offset = self.stream.module_to_host(0.0)
        samples = self.stream.samples_since_host(now - required_hold - 5.0)
        if offset is None:
            points = [(h, p) for _, h, p in samples]
        else:
            points = [(m + offset, p) for m, _, p in samples]
        window_start = now - required_hold
        inside = [p for t, p in points if t >= window_start]
        before = [p for t, p in points if t < window_start]
        if before:
            inside.append(before[-1])
        elif not inside:
            inside = [self.current_position()]
        return (max(inside) - min(inside)) > steady_threshold_deg

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

        Stream mode: see _wait_held_steady_stream() (same criterion, evaluated on the streamed,
        module-timestamped samples on the calling thread -- no extra thread, no rotary writes).
        """
        if self.stream is not None:
            if require_no_lick:
                raise ValueError("require_no_lick=True is not supported in stream mode")
            return self._wait_held_steady_stream(required_hold, steady_threshold_deg)

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

    def _wait_held_steady_stream(self, required_hold, steady_threshold_deg):
        """ Returns once the wheel's peak-to-peak range over the trailing `required_hold` seconds
        is <= steady_threshold_deg, judged on streamed samples placed at their module-clock
        acquisition times (mapped to host time). The firmware only sends a sample when the position
        changes, so the position at the start of the window is the last sample BEFORE it (held
        constant until the next one) -- a wheel that sent nothing during the window counts as
        still. Requires at least `required_hold` of waiting so a hold can't be satisfied by samples
        from before this call. Returns the number of times movement reset the hold window
        (registered as QUIESCENCE_RESET_TIME rows, same as polling mode). """
        wait_start = self._now()
        self.register('TRIAL_INIT_START', wait_start)
        n_breaks = 0
        was_moving = False
        warned = False
        interval = 1.0 / self.still_poll_hz
        lookback_s = required_hold + 5.0
        while True:
            self.pump()
            now = self._now()
            offset = self.stream.module_to_host(0.0)
            samples = self.stream.samples_since_host(now - lookback_s)
            if offset is None:
                points = [(h, p) for _, h, p in samples]
            else:
                points = [(m + offset, p) for m, _, p in samples]
            window_start = now - required_hold
            inside = [p for t, p in points if t >= window_start]
            before = [p for t, p in points if t < window_start]
            if before:
                inside.append(before[-1])
            elif not inside:
                inside = [self.current_position()]
            moving = (max(inside) - min(inside)) > steady_threshold_deg
            if moving and not was_moving:
                n_breaks += 1
                self.register('QUIESCENCE_RESET_TIME', now)
            was_moving = moving
            if not moving and (now - wait_start) >= required_hold:
                return n_breaks
            if not warned and (now - wait_start) > 3.0 + required_hold:
                warned = True
                print("Waiting for the wheel to hold steady (current position: "
                      "{0:.1f}deg)...".format(self.current_position()), flush=True)
            time.sleep(interval)
