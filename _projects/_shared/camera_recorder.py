# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""
USB camera video recording, time-synced to the same Bpod-session clock every other timing
registration in this codebase already uses. Lives at _projects/_shared/ (see
bpod_trial_helpers.py's module docstring for why) since recording video alongside a behavioral
session isn't specific to any one protocol.

Scope, per current instruction: record + show a live preview + save -- nothing reads or acts on
the video DURING a task. discover_camera()/CameraRecorder mirror rotary_setup.py/hifi_setup.py's
own house style (auto-discover-with-override connection helper, richly-commented docstrings), and
CameraRecorder's own pump()-driven design (no app.exec_()) mirrors dot_display.py/gabor_display.py.

**Recording runs on a background thread, fully decoupled from the caller's own render-loop
cadence** -- recording must proceed at the camera's own native rate regardless of how often (or
whether) anything else calls pump(), unlike dot_display.py/gabor_display.py where pump() itself
drives the one thing that matters (screen repaint). Locking follows bpod_trial_helpers.py's
TrialRunner convention exactly: one lock held only around the actual hardware read, a fresh
threading.Event() created per start() call (never a shared instance attribute reused across
restarts -- a previously-fixed real bug in TrialRunner for exactly that reason), a daemon thread,
a BOUNDED join() on stop, and a broad except Exception: break in the poll loop for graceful
teardown if the connection is torn down mid-thread (Stop/Kill). The capture loop is also
rate-limited to ~fps (measure elapsed, sleep the remainder, same shape as
TrialRunner._trial_poll_loop's own time.sleep(self.poll_interval)) -- confirmed on hardware as
necessary: cv2.VideoCapture.read() doesn't reliably block on every camera/backend combination, and
without an explicit throttle the loop can spin unthrottled and visibly lag the whole system by
contending for the GIL against the main thread's own Bpod serial I/O and Qt event processing.

**Live preview deliberately reuses QApplication.instance() (same single-Qt-toolkit convention
dot_display.py/gabor_display.py already establish) instead of cv2's own imshow()/HighGUI window.**
Mixing a second native GUI toolkit into a process that may already be running a Qt event loop for
a dot/gabor display is the same class of native-toolkit conflict that produced a confirmed real
interpreter crash elsewhere in this codebase (TkAgg + PyQt5's competing event loops, see CLAUDE.md)
-- one event loop for the whole process, always.

**Time sync**: every timing value registered anywhere else in this codebase (TRIAL_START,
DOT_ONSET_TIME, THRESHOLD_CROSSING_TIME, ...) is `time.time() - log_python_t0`, where
log_python_t0 is a single epoch captured once near the top of a task script and threaded through
everything else (TrialRunner itself takes it as a constructor argument for the same reason).
CameraRecorder requires the SAME log_python_t0 -- every captured frame is timestamped
`time.time() - log_python_t0` (immediately after cap.read() returns) and appended, in memory, to
a list; close() writes the full list to a companion CSV once the capture thread is confirmed
stopped. Per-frame timestamping (not an assumed fixed fps) is deliberate -- no live camera can
guarantee an exact delivered frame rate, so recording the real observed time for every frame is
what actually lets a downstream analysis map any Bpod event time to its nearest video frame.
CameraRecorder itself never calls runner.register(...) -- matching rotary_setup.py/hifi_setup.py's
own separation (hardware-connection modules don't touch Bpod's session/VAL API; that stays the
task script's own job). A caller should register camera.start_time_s itself, the same way
LEFT_THRESHOLD_DEG/etc. are registered directly in every task script rather than inside a _shared/
helper.

**Not yet confirmed on hardware**: the timestamp captured is when cap.read() returns to Python,
not a true hardware capture timestamp from the camera itself -- the best achievable synchronization
without deeper camera-hardware timestamp support. Actual jitter/latency against real Bpod events
hasn't been measured yet (see docs_local/CAMERA_TEST_PLAN.md's Stage 4). OpenCV on Windows
sometimes benefits from an explicit cv2.CAP_DSHOW backend hint for reliable/fast camera open --
flagged, not applied here, same "UNCONFIRMED" convention used elsewhere in this codebase for
anything not yet hardware-validated.
"""
import os
import sys
import threading
import time

import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt


def discover_camera(max_index=5):
    """
    Probes camera indices 0..max_index-1 via cv2.VideoCapture, opening each and confirming a real
    frame comes back, returning the first working index -- same handshake-probe discover()
    convention as RotaryEncoderModule.discover()/HiFiModule.discover(). OpenCV camera indices are
    enumeration order among CURRENTLY ACTIVE capture devices, not the same ordering as Windows'
    own PnP device list, and can shift if other capture devices are plugged/unplugged -- so this
    is re-derived at connection time rather than hardcoded, mirroring why the rotary/HiFi modules
    auto-discover their own USB port instead of assuming a fixed one.

    :param int max_index: highest index to probe (exclusive)
    :raises RuntimeError: if no index in range opens and yields a real frame
    """
    for index in range(max_index):
        cap = cv2.VideoCapture(index)
        try:
            if cap.isOpened():
                ok, frame = cap.read()
                if ok and frame is not None:
                    return index
        finally:
            cap.release()
    raise RuntimeError("No working camera found in indices 0..{0}".format(max_index - 1))


class _PreviewWidget(QWidget):
    def __init__(self):
        super(_PreviewWidget, self).__init__()
        self.setWindowTitle('Camera Preview')
        self.setWindowFlags(Qt.WindowStaysOnTopHint)
        layout = QVBoxLayout()
        self._label = QLabel()
        layout.addWidget(self._label)
        self.setLayout(layout)

    def set_frame(self, frame_bgr):
        """ frame_bgr is a raw cv2 frame (BGR, uint8, HxWx3) -- converted to RGB for QImage
        (Qt has no native BGR888 format), and np.ascontiguousarray keeps the buffer alive/
        contiguous for as long as the QImage needs it (same reasoning gabor_display.py's own
        _GaborWidget already documents for its own QImage-from-numpy-array construction). """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        h, w, _ = rgb.shape
        qimage = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        self._label.setPixmap(QPixmap.fromImage(qimage))
        self.resize(w, h)


class CameraRecorder(object):
    """
    Owns a background thread continuously grabbing frames from a USB camera and writing them to a
    video file, plus an optional live-preview window refreshed from the main thread's own pump()
    calls. See module docstring for the full threading/time-sync design.
    """

    def __init__(self, log_python_t0, camera_index=None, output_path='session_video.avi',
                 fps=30.0, preview=True):
        self._log_python_t0 = log_python_t0
        self.camera_index = camera_index if camera_index is not None else discover_camera()
        self.output_path = output_path
        self.fps = fps

        self._cap = cv2.VideoCapture(self.camera_index)
        if not self._cap.isOpened():
            raise RuntimeError("Could not open camera index {0}".format(self.camera_index))

        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # .avi/XVID -- no extra codec install needed on Windows with the plain opencv-python wheel,
        # unlike H264/mp4 which often needs one.
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self._writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (width, height))

        self._cap_lock = threading.Lock()
        self._latest_frame = None
        self._timestamps = []   # list of (frame_index, timestamp_s) -- see module docstring
        self._stop_event = None
        self._thread = None

        self.start_time_s = None

        self._preview_enabled = preview
        self._continuous_preview = False
        self._snippet_until = None
        self._app = None
        self._preview_widget = None
        if self._preview_enabled:
            self._app = QApplication.instance()
            if self._app is None:
                self._app = QApplication(sys.argv)
            self._preview_widget = _PreviewWidget()

        print("CameraRecorder: index {0}, {1}x{2} @ {3}fps -> {4}".format(
            self.camera_index, width, height, self.fps, self.output_path), flush=True)

    def show(self):
        """ Shows the preview widget continuously (repainted on every pump() call), if enabled --
        no-op otherwise. For tasks running alongside other background threads/GIL-sensitive code,
        prefer show_snippet() instead -- see its own docstring. """
        if self._preview_widget is not None:
            self._continuous_preview = True
            self._preview_widget.show()

    def show_snippet(self, duration_s=1.0):
        """
        Shows the preview widget for a brief window (duration_s seconds) instead of continuously --
        call at specific, meaningful moments (e.g. decision-period start, choice made) rather than
        leaving the preview open/repainting for the whole task. Confirmed on hardware as a real,
        worthwhile lag reduction: the per-frame cv2.cvtColor/QImage/QPixmap conversion + repaint is
        the dominant remaining main-thread cost once the capture thread itself is rate-limited (see
        _capture_loop's own comment) -- doing that work only within short snippet windows instead of
        on every render-loop iteration meaningfully reduces GIL contention against the main thread's
        own Bpod serial I/O and Qt event processing, in scripts (like full_protocol_lookback_test.py)
        that already run other background threads and have a documented GIL-pressure crash history.
        Recording itself is completely unaffected either way -- this only controls the PREVIEW
        WIDGET's own visibility/update window, never the background capture+write thread.
        No-op if preview wasn't enabled at construction.
        """
        if self._preview_widget is not None:
            self._snippet_until = time.time() + duration_s
            self._preview_widget.show()

    def start(self):
        """ Spawns the background capture+write thread and records start_time_s (see module
        docstring's "Time sync" section) -- call once, before the first pump(). """
        self.start_time_s = time.time() - self._log_python_t0
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._capture_loop, args=(self._stop_event,),
                                         daemon=True)
        self._thread.start()

    def _capture_loop(self, stop_event):
        # Rate-limited to ~self.fps, same "measure elapsed, sleep the remainder" shape as this
        # codebase's other timing-sensitive poll loops (e.g. TrialRunner._trial_poll_loop's own
        # time.sleep(self.poll_interval)) -- cv2.VideoCapture.read() is USUALLY naturally paced by
        # the camera's own frame delivery, but some Windows camera/backend combinations return
        # immediately with a cached frame instead of blocking. Without this, the loop would spin
        # as fast as the CPU allows, hammering the GIL against the main thread's own Bpod serial
        # I/O and Qt event processing -- confirmed on hardware as the cause of a real "very laggy"
        # symptom the first time this ran without this throttle.
        frame_interval = (1.0 / self.fps) if self.fps > 0 else 0
        frame_index = 0
        while not stop_event.is_set():
            loop_start = time.time()
            try:
                with self._cap_lock:
                    ok, frame = self._cap.read()
                t = time.time() - self._log_python_t0
                if not ok or frame is None:
                    continue
                self._writer.write(frame)
                self._timestamps.append((frame_index, t))
                frame_index += 1
                self._latest_frame = frame
            except Exception:
                break
            sleep_time = frame_interval - (time.time() - loop_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def pump(self):
        """ Refreshes the live preview (if enabled) from the most recently captured frame, and
        processes pending Qt events -- call repeatedly from the caller's own render/trial loop,
        same convention as dot_display.py/gabor_display.py's own pump(). Recording itself proceeds
        independently on the background thread regardless of how often (or whether) this is
        called.

        Preview repaint only happens in continuous mode (after show()) or while inside an active
        show_snippet() window -- see those methods' own docstrings. Outside of both, this is just
        a cheap processEvents() call, no frame-conversion/repaint work at all. """
        if self._preview_widget is not None and self._latest_frame is not None:
            if self._continuous_preview:
                self._preview_widget.set_frame(self._latest_frame)
            elif self._snippet_until is not None:
                if time.time() < self._snippet_until:
                    self._preview_widget.set_frame(self._latest_frame)
                else:
                    self._preview_widget.hide()
                    self._snippet_until = None
        if self._app is not None:
            self._app.processEvents()

    def close(self):
        """ Stops the background thread (bounded join, same convention as TrialRunner), releases
        the camera and finalizes the video file, writes the frame-timestamps CSV, and closes the
        preview window. """
        if self._stop_event is not None:
            self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)

        self._cap.release()
        self._writer.release()
        self._save_timestamps()

        if self._preview_widget is not None:
            self._preview_widget.close()

    def _save_timestamps(self):
        stem, _ext = os.path.splitext(self.output_path)
        ts_path = stem + '_timestamps.csv'
        with open(ts_path, 'w') as f:
            f.write('frame_index,timestamp_s\n')
            for frame_index, t in self._timestamps:
                f.write('{0},{1:.6f}\n'.format(frame_index, t))
        print("CameraRecorder: wrote {0} frame timestamps to {1}".format(
            len(self._timestamps), ts_path), flush=True)
