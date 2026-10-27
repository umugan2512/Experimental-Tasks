# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""
The one render loop every AuditoryEvidenceAccum stage runs on the main thread while its trial's
Bpod machine(s) run on a background thread -- shared so the five stages can't drift apart (the
same reasoning as dot_display.create_dot_display()).

It drives the PC side of the per-trial handshake (2026-10-23), MATLAB Bpod's model: everything
happens inside a running state machine, and the PC reacts to and answers it with SoftCodes.

    WaitQuiescence   PC watches the wheel stream; once it has been still for the hold window, the
                     PC re-arms the rotary (Z+E over USB) and injects SoftCode1 -> DotOn
                     (on_quiescence_met runs right then -- Stage 3-5 start the audio there).
    DotOn            entry sends SOFTCODE_DOT_ON; the PC draws the dot centred, pumps Qt so it is
                     on screen, re-arms the rotary again and injects SoftCode1 -> the response
                     state (WheelPeriod/WheelDotPeriod), whose entry re-zeroes the rotary and sends
                     the '#' marker. A crossing in DotOn -> MovedBeforeDot; no answer -> DotTimeout.
    response state   the dot follows the wheel (post-marker samples only) and freezes on the
                     freeze SoftCode (Reward etc.) or the same-zero position crossing; it is
                     cleared J2 after freezing.

So the reward window can only open once the dot has been drawn -- before this, 4 of 36 trials in
session 20261022-162023 were rewarded before the dot was painted.

Every wait uses runner.wait_signal() (wakes on any SoftCode) instead of a fixed sleep.
"""
import time

import rotary_setup
from bpod_trial_helpers import PC_EVENT_DOT_SHOWN, PC_EVENT_QUIESCENCE_MET


def run_trial_render_loop(runner, dot, camera, done_event, threshold_deg, disappear_delay_s,
                          render_interval, softcode_dot_on, freeze_softcodes,
                          quiescence=None, punish_softcode=None, on_quiescence_met=None,
                          dot_phase_event=None, punish_flash_s=0.35):
    """
    Runs until `done_event` is set (the Bpod thread finished). Returns a dict of what happened and
    when (host seconds, log_python_t0-relative), for the caller to register:
      quiescence_met_t, dot_paint_t, freeze_t, freeze_source, clear_t,
      render_max_frame_ms, onset_latency_ms (SOFTCODE_DOT_ON arrival -> dot painted),
      freeze_latency_ms (freeze SoftCode arrival -> dot frozen).

    :param quiescence: (required_hold_s, steady_threshold_deg) when the trial's machine starts in
        WaitQuiescence (Stage 1/2 trials, Stage 3-5 cue machines); None if it doesn't. The caller
        must have called runner.start_quiescence(*quiescence) BEFORE starting the Bpod thread (it
        registers TRIAL_INIT_START; registering it after RUN put a session write right on top of
        the new Trial's creation -- see bpod_trial_helpers.install_session_lock()).
    :param freeze_softcodes: SoftCode values whose arrival freezes the dot (an outcome state was
        entered).
    :param punish_softcode: SoftCode that triggers dot.flash_white() (Stage 3-5's AbortPunish/
        ErrorPunish), or None.
    :param on_quiescence_met: called (no args) right after the quiescence-met SoftCode is injected.
    :param dot_phase_event: if given, the dot handshake is only serviced once this Event is set
        (Stage 3-5: the choice machine has started -- set by the Bpod thread right after its
        runner.begin_trial()).
    """
    t0 = runner.log_python_t0
    now_s = lambda: time.time() - t0
    out = {'quiescence_met_t': None, 'dot_paint_t': None, 'freeze_t': None, 'freeze_source': None,
           'clear_t': None, 'render_max_frame_ms': 0.0, 'onset_latency_ms': None,
           'freeze_latency_ms': None}
    phase = 'quiescence' if quiescence is not None else 'await_dot'
    punished = False
    last_frame = time.time()

    while not done_event.is_set():
        if phase == 'quiescence' and runner.poll_quiescence():
            rotary_setup.arm_for_trial(runner.rotary)
            runner.inject_softcode(PC_EVENT_QUIESCENCE_MET)
            out['quiescence_met_t'] = now_s()
            if on_quiescence_met is not None:
                on_quiescence_met()
            phase = 'await_dot'

        if (phase == 'await_dot' and (dot_phase_event is None or dot_phase_event.is_set())
                and runner.softcode_seen(softcode_dot_on) is not None):
            dot.set_position_deg(0.0)
            dot.pump()
            out['dot_paint_t'] = now_s()
            out['onset_latency_ms'] = (out['dot_paint_t'] - runner.softcode_seen(softcode_dot_on)) * 1e3
            rotary_setup.arm_for_trial(runner.rotary)
            runner.inject_softcode(PC_EVENT_DOT_SHOWN)
            phase = 'tracking'

        if phase == 'tracking':
            frozen_by = next((c for c in freeze_softcodes if runner.softcode_seen(c) is not None), None)
            pos = runner.position_after_marker()
            pos = 0.0 if pos is None else pos
            dot.set_position_deg(pos)   # render first, so the frame that crosses still shows travel
            if frozen_by is not None:
                out['freeze_source'] = 'state'
                out['freeze_latency_ms'] = (now_s() - runner.softcode_seen(frozen_by)) * 1e3
            elif runner.last_marker_seen() and (pos <= -threshold_deg or pos >= threshold_deg):
                out['freeze_source'] = 'position'
            if out['freeze_source'] is not None:
                out['freeze_t'] = now_s()
                phase = 'frozen'

        if phase == 'frozen' and out['clear_t'] is None and now_s() - out['freeze_t'] >= disappear_delay_s:
            dot.clear()
            out['clear_t'] = now_s()

        if punish_softcode is not None and not punished and runner.softcode_seen(punish_softcode) is not None:
            dot.flash_white(punish_flash_s)
            punished = True

        dot.pump()
        camera.pump()
        runner.pump()
        t = time.time()
        out['render_max_frame_ms'] = max(out['render_max_frame_ms'], (t - last_frame) * 1e3)
        runner.wait_signal(render_interval)
        last_frame = time.time()

    return out


def register_trial_timing(runner, render, quiescence_hold_s, n_breaks, prev_return_t, visited,
                          send_t, response_state='WheelPeriod'):
    """ Registers the per-trial timing/continuity VALs every stage logs (same names everywhere,
    so records/audit_session.py can check any stage):
    - QUIESCENCE_HOLD_S / QUIESCENCE_BREAKS / QUIESCENCE_MET_TIME (also as TRIAL_START, which has
      always meant "the moment the hold was satisfied")
    - DOT_ONSET_TIME: the response state's start -- the hardware time the PC's "dot is painted"
      answer arrived -- on the host clock (send_t + Bpod-relative start)
    - DOT_ONSET_PAINT_TIME, DOT_FREEZE_TIME/_SOURCE, DOT_VISIBLE_S (paint -> clear)
    - RENDER_MAX_FRAME_MS, RENDER_ONSET_LATENCY_MS, DOT_FREEZE_LATENCY_MS
    - INTER_TRIAL_GAP_MS: previous Bpod call returning -> this RUN (the only time no state
      machine is running), STREAM_MAX_READ_GAP_MS, WHEEL_LOG_LOCK_MAX_MS """
    reg = runner.register
    if quiescence_hold_s is not None:
        reg('QUIESCENCE_HOLD_S', quiescence_hold_s)
        reg('QUIESCENCE_BREAKS', n_breaks)
    if render['quiescence_met_t'] is not None:
        reg('QUIESCENCE_MET_TIME', render['quiescence_met_t'])
        reg('TRIAL_START', render['quiescence_met_t'])
    state = visited.get(response_state)
    if state and state[-1][0] == state[-1][0]:   # not nan
        reg('DOT_ONSET_TIME', send_t + state[-1][0])
    if render['dot_paint_t'] is not None:
        reg('DOT_ONSET_PAINT_TIME', render['dot_paint_t'])
    if render['freeze_t'] is not None:
        reg('DOT_FREEZE_TIME', render['freeze_t'])
        reg('DOT_FREEZE_SOURCE', render['freeze_source'])
    if render['dot_paint_t'] is not None:
        end = render['clear_t'] if render['clear_t'] is not None else runner._now()
        reg('DOT_VISIBLE_S', end - render['dot_paint_t'])
    reg('RENDER_MAX_FRAME_MS', render['render_max_frame_ms'])
    if render['onset_latency_ms'] is not None:
        reg('RENDER_ONSET_LATENCY_MS', render['onset_latency_ms'])
    if render['freeze_latency_ms'] is not None:
        reg('DOT_FREEZE_LATENCY_MS', render['freeze_latency_ms'])
    if prev_return_t is not None and send_t is not None:
        reg('INTER_TRIAL_GAP_MS', (send_t - prev_return_t) * 1e3)
    if runner.stream is not None:
        reg('STREAM_MAX_READ_GAP_MS', runner.stream.pop_max_read_gap_ms())
    reg('WHEEL_LOG_LOCK_MAX_MS', getattr(runner, 'wheel_log_lock_max_ms', 0.0))
    if getattr(runner, 'run_write_ms', None) is not None:
        reg('SM_RUN_WRITE_MS', runner.run_write_ms)
    runner.wheel_log_lock_max_ms = 0.0
