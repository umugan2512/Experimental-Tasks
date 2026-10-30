# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""
The host clock every task-side timestamp uses (`host_clock.now() - log_python_t0`).

Why not time.time(): it is the Windows wall clock, which the Windows Time service steps when it
syncs. Session 20261030-142131 got a +1.158s step mid-session (System log, Kernel-General event 1,
14:25:55): every host time after it moved 1.16s against the Bpod clock, and the camera, the CSV and
the rotary reader all showed a 1.2s "gap" that never happened.

now() is the wall time once at import plus time.perf_counter() (QueryPerformanceCounter on
Windows) elapsed since then: same epoch-like values as time.time(), sub-microsecond resolution,
and never stepped. (time.monotonic() is also step-free but ticks at 15.6ms on Windows.)
"""
import time

_WALL_T0 = time.time()
_PERF_T0 = time.perf_counter()


def now():
    return _WALL_T0 + (time.perf_counter() - _PERF_T0)
