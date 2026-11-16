# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Stage 3-5: upload the next trial's click train during this trial's ITI (NextTrialAudio).
No hardware imports, so the soak harness runs the real class against its fake HiFi.
"""
import threading
import time
import traceback


class NextTrialAudio(object):
    """ Builds and uploads the NEXT trial's click train during THIS trial's ITI, like MATLAB
    Bpod_Gen2's HiFiSound2AFC_TrialManager example (which loads the next sounds after the
    outcome, during the running trial, and makes them live at the next trial's start). The next
    trial's side isn't known until the outcome is processed, so both sides are prepared, in
    SLOTS; take(side) pushes and returns the matching slot.

    The HiFi is only touched after `outcome_seen()` turns True (the HiFi is silent from then on)
    and before `give_up` is set (the trial's machine ended): if the machine ends first, nothing is
    prepared and the next trial falls back to uploading during its own WaitQuiescence. Everything
    else on this HiFi (play, the fallback upload) runs strictly before or after this thread. """

    SLOTS = {'L': 1, 'R': 2}   # slot 0 is the fallback upload's

    def __init__(self, hifi):
        self.hifi = hifi
        self._thread = None
        self._result = None

    def prepare_after_outcome(self, make_audio, outcome_seen, give_up, poll_s=0.01):
        """ make_audio(side) -> (meta, waveform). Starts the background thread; returns at once. """
        self._result = None

        def _run():
            while not outcome_seen():
                if give_up.is_set():
                    return
                time.sleep(poll_s)
            t0 = time.perf_counter()
            try:
                metas = {}
                for side, slot in sorted(self.SLOTS.items()):
                    metas[side], wave = make_audio(side)
                    self.hifi.load(slot, wave)
                self._result = {'meta': metas, 'prep_ms': (time.perf_counter() - t0) * 1e3}
            except Exception:
                self._result = {'error': traceback.format_exc()}

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def take(self, side, timeout=5.0):
        """ (meta, slot, prep_ms, wait_ms) for `side` once prepared, after pushing the new slots
        live; None if nothing was prepared. Raises if the upload failed: a partly sent waveform
        leaves the HiFi waiting for the rest, so the session can't go on. """
        if self._thread is None:
            return None
        t0 = time.perf_counter()
        self._thread.join(timeout)
        wait_ms = (time.perf_counter() - t0) * 1e3
        if self._thread.is_alive():
            raise RuntimeError('next-trial HiFi upload still running after {0}s'.format(timeout))
        self._thread, result, self._result = None, self._result, None
        if result is None:
            return None
        if 'error' in result:
            raise RuntimeError('next-trial HiFi upload failed:\n' + result['error'])
        self.hifi.push()
        return result['meta'][side], self.SLOTS[side], result['prep_ms'], wait_ms
