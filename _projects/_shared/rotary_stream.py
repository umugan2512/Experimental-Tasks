# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Continuous, hardware-timestamped wheel position from the rotary encoder module's own USB stream --
the same mechanism MATLAB Bpod_Gen2 uses (`RotaryEncoderModule.m`'s startUSBStream()/
getUSBStream()/readUSBStream(eventCode)), replacing the old COM_GETCURRENTPOS ('Q') request/
response polling this codebase used everywhere before.

Why (confirmed in real Stage 1 session CSVs, 2026-10-21/22):
- Polling topped out at ~20-33Hz and left a 0.2-1.5s logging gap at every trial onset, because the
  main thread's own per-trial rotary writes and the state-machine upload happened while nothing was
  polling. A session-long poll thread to close that gap was tried and reverted: it crashed real
  sessions with SerialTimeoutException, because a polling thread WRITES ('Q') on the same
  connection the main thread writes on.
- In stream mode the firmware pushes every sample on its own; this reader thread only ever READS
  the port, so it can't collide with anything. Each 'P' sample and each 'E' event carries the
  module's own clock timestamp, so sample timing no longer depends on host scheduling at all.
- A state machine output action can send '#'+code to the module over the Bpod relay (see
  rotary_setup.build_reset_trigger()). The module logs it as an 'E' event on its own clock, so a
  Bpod state transition (known exactly on Bpod's clock) and the wheel stream (module clock) share
  a common, hardware-timed reference point every trial -- MATLAB's readUSBStream(eventCode) split
  uses exactly this.

Stream message layout (7 bytes each, identical in module_api.read_stream() and MATLAB
getUSBStream()): 'P' + int16 position ticks + uint32 time, or 'E' + uint8 type + uint8 code + uint32
time. The time unit differs by firmware (pybpod's read_stream() divides by 1e3, MATLAB's fw v6 path
by 1e6) -- auto-detected here by comparing module-clock elapsed time against host elapsed time once
enough data exists, rather than trusting either constant blindly. Verify on the rig with
Tests/tasks/rotary_stream_test/.

**While streaming, no other command that reads an acknowledgment byte may be sent** (the ack would
be swallowed by this reader, or a stream byte would be read as the ack) -- MATLAB's own
assertNotUSBStreaming() enforces the same rule. Use `with stream.paused(): rotary.set_thresholds(...)`
for those. Commands with no reply (relay-sent 'Z'/'E'/'#') are fine at any time.
"""
import contextlib
import sys
import threading
import time
from collections import deque

_MSG_LEN = 7
_READER_TIMEOUT_S = 0.005        # reader's blocking-read timeout (ack reads get the original back)
_SCALE_CANDIDATES = (1e-6, 1e-3)   # seconds per module-clock tick: fw v6 (us) vs older (ms)
_DEFAULT_SCALE = 1e-6
_MAX_DRIFT = 50e-6                # module->host offset may rise at most this fast (s per s); a
                                  # LOWER observed latency is adopted immediately (see _ingest)


class RotaryStream(object):
    """
    One instance per task script, constructed once the rotary is connected:
    `stream = RotaryStream(rotary, log_python_t0)`, `stream.start()`, ..., `stream.stop()`.

    All "host" times returned here are on the same `time.time() - log_python_t0` clock every VAL
    registration in this codebase already uses; "module" times are seconds on the module's own clock.
    """

    def __init__(self, rotary, log_python_t0, max_buffer_s=60.0):
        self.rotary = rotary
        self.log_python_t0 = log_python_t0
        self.max_buffer_s = max_buffer_s

        self._lock = threading.Lock()
        self._thread = None
        self._stop_event = None
        self._reader_idle = threading.Event()
        self._pause_request = threading.Event()

        self._carry = b''
        self._raw_last = 0
        self._rollover_ticks = 0
        self._scale = None
        self._scale_probe = []        # (raw_ticks_unwrapped, host_t) until the unit is resolved

        self._samples = deque()       # (module_t, host_rx_t, pos_deg)
        self._events = deque()        # (module_t, host_rx_t, event_type, event_code)
        self._log_queue = []          # samples not yet written to the session CSV
        self._offset_est = None       # host minus module seconds (see _ingest)
        self._offset_t = None         # host time of the last estimate update
        self._latest = None           # (module_t, host_rx_t, pos_deg)
        self.n_resyncs = 0
        self._max_read_gap_s = 0.0    # longest reader-loop iteration since last pop_max_read_gap_ms()

    # --- lifecycle ---------------------------------------------------------------------------------

    def start(self):
        """ Starts the firmware stream and the read-only reader thread. Idempotent. """
        if self._thread is not None and self._thread.is_alive():
            return
        serial_object = self.rotary.arcom.serial_object
        self.rotary.disable_stream()
        time.sleep(0.05)
        serial_object.reset_input_buffer()
        self._carry = b''
        self._ack_timeout = getattr(serial_object, 'timeout', None)
        _set_timeout(serial_object, _READER_TIMEOUT_S)
        self.rotary.enable_stream()

        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._read_loop, args=(self._stop_event,),
                                         daemon=True)
        self._thread.start()

    def stop(self):
        """ Stops the reader thread and the firmware stream (bounded join -- see CLAUDE.md's Kill
        note). Safe to call more than once. """
        if self._stop_event is not None:
            self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        try:
            self.rotary.disable_stream()
            time.sleep(0.05)
            self.rotary.arcom.serial_object.reset_input_buffer()
            _set_timeout(self.rotary.arcom.serial_object, getattr(self, '_ack_timeout', None))
        except Exception:
            pass

    @contextlib.contextmanager
    def paused(self):
        """ Temporarily stops the firmware stream so ONE ack-reading command (set_thresholds(),
        enable_evt_transmission(), ...) can run on this connection. Data already in flight is still
        parsed and kept. Never use this while a trial's state machine is running -- the wheel is
        unlogged for the (short, ~0.1s) duration. """
        self._reader_idle.clear()
        self._pause_request.set()
        self._reader_idle.wait(timeout=1.0)
        try:
            self.rotary.disable_stream()
            time.sleep(0.05)
            serial_object = self.rotary.arcom.serial_object
            n = serial_object.in_waiting
            if n:
                self._ingest(serial_object.read(n), self._now())
            self._carry = b''
            serial_object.reset_input_buffer()
            _set_timeout(serial_object, self._ack_timeout)   # ack reads need their normal timeout
            yield
        finally:
            _set_timeout(self.rotary.arcom.serial_object, _READER_TIMEOUT_S)
            self.rotary.arcom.serial_object.reset_input_buffer()
            self.rotary.enable_stream()
            self._pause_request.clear()

    # --- reader thread -----------------------------------------------------------------------------

    def _now(self):
        return time.time() - self.log_python_t0

    def _read_loop(self, stop_event):
        """ READ-ONLY. Wrapped in a broad try/except: a Kill/Stop can close the port out from under
        this thread, which is harmless (the process is ending) -- same tolerance as TrialRunner's
        old poll loop. """
        serial_object = self.rotary.arcom.serial_object
        _raise_this_thread_priority()
        last = time.time()
        while not stop_event.is_set():
            now = time.time()
            self._max_read_gap_s = max(self._max_read_gap_s, now - last)
            last = now
            if self._pause_request.is_set():
                self._reader_idle.set()
                time.sleep(0.005)
                continue
            try:
                # Block in the driver until at least one byte arrives (bounded by the short reader
                # timeout set in start()) instead of sleep-polling -- wakes the moment data lands,
                # with the GIL released while waiting.
                data = serial_object.read(max(1, serial_object.in_waiting))
                if data:
                    n = serial_object.in_waiting
                    if n:
                        data += serial_object.read(n)
                    self._ingest(data, self._now())
                else:
                    time.sleep(0.001)
            except Exception:
                break

    def _ingest(self, data, host_t):
        buf = self._carry + data
        i = 0
        new_samples = []
        new_events = []
        while len(buf) - i >= _MSG_LEN:
            op = buf[i:i + 1]
            if op == b'P':
                ticks = int.from_bytes(buf[i + 1:i + 3], byteorder='little', signed=True)
                raw = int.from_bytes(buf[i + 3:i + 7], byteorder='little', signed=False)
                new_samples.append((self._unwrap(raw), ticks))
                i += _MSG_LEN
            elif op == b'E':
                ev_type, ev_code = buf[i + 1], buf[i + 2]
                raw = int.from_bytes(buf[i + 3:i + 7], byteorder='little', signed=False)
                new_events.append((self._unwrap(raw), ev_type, ev_code))
                i += _MSG_LEN
            else:
                i += 1   # lost alignment (should only ever happen right after start/pause)
                self.n_resyncs += 1
        self._carry = buf[i:]

        if not new_samples and not new_events:
            return
        with self._lock:
            if self._scale is None:
                self._resolve_scale(new_samples, new_events, host_t)
            scale = self._scale if self._scale is not None else _DEFAULT_SCALE
            for raw, ticks in new_samples:
                module_t = raw * scale
                sample = (module_t, host_t, _ticks_to_deg(ticks))
                self._samples.append(sample)
                self._log_queue.append(sample)
                self._latest = sample
            for raw, ev_type, ev_code in new_events:
                self._events.append((raw * scale, host_t, ev_type, ev_code))
            # The newest message in this read has the smallest transport latency, so host_t minus
            # its module time is an UPPER bound on the true clock offset. Track the lower envelope:
            # a lower bound-candidate is adopted at once; a higher one only at the plausible clock
            # drift rate. The estimate is never discarded during still periods (no data) -- the
            # old 5s-window minimum emptied then and restarted high, jittering the mapped
            # WHEEL_POS times by up to 30ms (session 20261022-162023).
            newest_raw = max([s[0] for s in new_samples] + [e[0] for e in new_events])
            candidate = host_t - newest_raw * scale
            if self._offset_est is None or candidate < self._offset_est:
                self._offset_est = candidate
            else:
                allowed = _MAX_DRIFT * max(0.0, host_t - (self._offset_t or host_t))
                self._offset_est += min(candidate - self._offset_est, allowed)
            self._offset_t = host_t
            while self._samples and self._samples[0][1] < host_t - self.max_buffer_s:
                self._samples.popleft()
            while self._events and self._events[0][1] < host_t - self.max_buffer_s:
                self._events.popleft()

    def _unwrap(self, raw):
        if raw < self._raw_last and (self._raw_last - raw) > (1 << 31):
            self._rollover_ticks += (1 << 32)
        self._raw_last = raw
        return raw + self._rollover_ticks

    def _resolve_scale(self, new_samples, new_events, host_t):
        """ Picks whichever candidate tick length makes module-clock elapsed time match host elapsed
        time, once >=1s of host time has passed. Until then _DEFAULT_SCALE is used. """
        raws = [s[0] for s in new_samples] + [e[0] for e in new_events]
        self._scale_probe.append((max(raws), host_t))
        first_raw, first_host = self._scale_probe[0]
        last_raw, last_host = self._scale_probe[-1]
        host_span = last_host - first_host
        if host_span < 1.0 or last_raw <= first_raw:
            return
        ratio = {c: abs((last_raw - first_raw) * c / host_span - 1.0) for c in _SCALE_CANDIDATES}
        best = min(ratio, key=ratio.get)
        if ratio[best] > 0.05:
            print("WARNING: RotaryStream couldn't match the module clock to host time (module/host "
                  "elapsed ratio off by {0:.0%}) -- assuming {1}s/tick".format(
                      ratio[best], _DEFAULT_SCALE), flush=True)
            best = _DEFAULT_SCALE
        if best != _DEFAULT_SCALE:
            # rescale anything already stored under the provisional default
            factor = best / _DEFAULT_SCALE
            self._samples = deque((m * factor, h, p) for m, h, p in self._samples)
            self._log_queue = [(m * factor, h, p) for m, h, p in self._log_queue]
            self._events = deque((m * factor, h, a, b) for m, h, a, b in self._events)
            self._offset_est = None
            if self._latest is not None:
                m, h, p = self._latest
                self._latest = (m * factor, h, p)
        self._scale = best
        print("RotaryStream: module clock tick = {0}s".format(best), flush=True)

    # --- queries -----------------------------------------------------------------------------------

    def pop_max_read_gap_ms(self):
        """ Longest gap between reader-loop iterations since the last call (ms) -- a stalled
        reader would show here, separately from a still wheel. """
        gap, self._max_read_gap_s = self._max_read_gap_s, 0.0
        return gap * 1e3

    def _offset(self):
        return self._offset_est

    def module_to_host(self, module_t):
        """ Module-clock seconds -> host (log_python_t0-relative) seconds, via the minimum observed
        transport latency over the last few seconds. None until any data has arrived. """
        with self._lock:
            off = self._offset()
        return None if off is None else module_t + off

    def latest(self):
        """ (module_t, host_rx_t, pos_deg) of the most recent sample, or None. """
        with self._lock:
            return self._latest

    def samples_since_host(self, host_t):
        """ Every buffered sample received at or after host time host_t. """
        with self._lock:
            return _tail_since(self._samples, host_t)

    def find_event(self, code, after_host_t=None, event_type=None):
        """ First buffered event with this code (and type, if given) received at/after
        after_host_t -- e.g. the '#' marker the trial's first state sends. None if not (yet) seen. """
        with self._lock:
            for module_t, host_t, ev_type, ev_code in self._events:
                if ev_code != code:
                    continue
                if event_type is not None and ev_type != event_type:
                    continue
                if after_host_t is not None and host_t < after_host_t:
                    continue
                return (module_t, host_t, ev_type, ev_code)
        return None

    def events_since_host(self, host_t):
        with self._lock:
            return _tail_since(self._events, host_t)

    def drain_for_logging(self):
        """ Returns, and forgets, every sample not yet handed out for CSV logging, each as
        (module_t, host_t_mapped, pos_deg) -- host_t_mapped puts the sample at its true
        module-clock acquisition time on the host clock, not at whenever the USB read happened to
        deliver it. """
        with self._lock:
            queue, self._log_queue = self._log_queue, []
            off = self._offset()
        if off is None:
            return [(m, h, p) for m, h, p in queue]
        return [(m, m + off, p) for m, h, p in queue]


def _raise_this_thread_priority():
    """ THREAD_PRIORITY_ABOVE_NORMAL for the calling thread on Windows (no-op elsewhere): the
    reader is the one thread whose lateness delays the live dot and the hold-still check. """
    if not sys.platform.startswith('win'):
        return
    try:
        import ctypes
        k32 = ctypes.windll.kernel32
        k32.SetThreadPriority(k32.GetCurrentThread(), 1)
    except Exception:
        pass


def _set_timeout(serial_object, timeout):
    """ pyserial timeout setter that tolerates objects without one (and None = leave as is). """
    if timeout is not None and hasattr(serial_object, 'timeout'):
        try:
            serial_object.timeout = timeout
        except Exception:
            pass


def _tail_since(buffer, host_t):
    """ Items of `buffer` (appended in host-receive order, host time at index 1) received at or
    after host_t -- scanned from the newest end, so the cost is the size of the answer, not of the
    whole buffer (the old full scan ran under the stream lock on every render frame and every
    hold-check tick). """
    out = []
    for item in reversed(buffer):
        if item[1] < host_t:
            break
        out.append(item)
    out.reverse()
    return out


def _ticks_to_deg(ticks):
    """ Same conversion (and 0.1deg rounding) as RotaryEncoderModule.__pos_2_degrees(). """
    return round(((float(ticks) / 512.0) * 180.0) * 10.0) / 10.0
