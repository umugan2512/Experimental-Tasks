# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Pre-session hardware/rig check -- one interactive tool for the things the static
`session_startup_checklist.xlsx`/`.md` currently only asks a human to tick off by hand
("Water line primed and checked for drips" etc.) with no tool actually behind it: prime the water
lines, confirm the lick sensor detects real licks (with real timestamps, not just yes/no), play a
calibrated-volume test tone, blink the go-cue LED, and report live Bpod/rotary/HiFi connection
status -- all in one place before starting a real session.

Run directly: `python rig_check.py` (needs the Bpod board physically connected; rotary encoder and
HiFi module are each optional -- a failed connection to either is reported, not fatal, so the rest
of the tool still works). Deliberately **standalone, not a PyBpod GUI task** -- same "bench
maintenance procedure, not project/session-tracked experiment data" reasoning as
`calibrate_liquid.py`/`calibrate_sound.py`/`flush_valve.py`: it connects directly to
`Bpod(serial_port=VAR_BPOD_SERIAL_PORT)` (see those files' own docstrings for why the explicit
`serial_port=` is required here, unlike a real task script) and skips all PyBpod project/session
scaffolding entirely -- there is no session data to suppress, because none is ever generated in the
first place.

**Water line priming**: `flush_valve.py`'s exact toggle mechanism, reused directly --
`manual_override(Bpod.ChannelTypes.OUTPUT, Bpod.ChannelNames.VALVE, channel_number=1, value=1/0)`,
open on one call, closed on a later one (no Bpod-side auto-close/timeout exists for
`manual_override`, confirmed in `flush_valve.py`'s own docstring). Enter/Return is bound
**window-wide** to this toggle (not scoped to one widget's focus, same `QShortcut` convention
`calibrate_sound.py` already uses for its own Enter-to-play binding) -- this is the ONE thing Enter
does in this tool, so unlike `calibrate_sound.py` the tone-check "Play Tone" button deliberately
has no keyboard shortcut of its own, to avoid two very different physical actions (opening a valve,
playing a tone) fighting over the same key. `closeEvent()` force-closes the valve (and turns off
the LED) if either was left on when the window closes, mirroring `flush_valve.py`'s own
`finally`-guaranteed close-on-exit.

**Lick sensor / TTL monitor** (training01): this box's lick sensor is an OPB815WZ IR beam break
read by an Arduino (`arduino_lick_relay/`), which copies the beam state to Bpod BNC In 2
(HIGH = beam broken: BNC2High = lick in). Digital inputs have no polling/`manual_override`-read API -- they're only
timestamped while a `StateMachine` runs (see this repo's own `CLAUDE.md`, Hardware module notes).
`TtlMonitorThread` repeatedly sends a short `Listen` state (`Tup`: `exit`, nothing else) back to
back on a background `QThread` (same "background thread for anything that blocks on real Bpod
time" pattern as `calibrate_liquid.py`'s own `PulseRunner`). Bpod logs every enabled input's edges
whether or not a state listens for them, so every Port/BNC/Wire edge is reported with its Bpod
timestamp (elapsed since the monitor started). The per-channel edge counts show which input is
actually wired; the "Lick input" channel (default: this box's own, `_shared/lick_input.py`; or, on
Auto, the first input that fires) is read as the
lick port: `PortNIn`/`BNCNHigh`/`WireNHigh` = lick in. Edges during the ~50 ms handover between
1 s windows are not logged.

**Tone check**: same channel/frequency selector as `calibrate_sound.py`, but the amplitude is
looked up live via `sound_calibration.get_calibrated_amplitude()` at a fixed `VAR_CHECK_SPL_DB`
reference (default matches every real task script's own `VAR_TARGET_SPL_DB`) -- so this checks
"does the real in-task volume actually sound right," not just "is a tone audible at all." Same
repeated-click-train playback shape `calibrate_sound.py` uses (not one continuous tone -- an SPL
meter/the ear can't time-integrate a single 8ms click pip the way it does a sustained tone; see that
file's own docstring), built from `click_train_v2.py`'s own click shape so it matches what a real
task actually plays.

**Camera check**: deliberately does NOT use `_shared/camera_recorder.py`'s `CameraRecorder` --
that class always opens a `cv2.VideoWriter` and writes a real video file (plus a timestamps CSV on
close) the moment its capture thread starts, with no preview-only mode; using it here would violate
this whole tool's own "no data is ever saved" design just to look at a live feed. `CameraCheckThread`
below is a small, local, preview-only capture loop (reuses `camera_recorder.discover_camera()` for
the same handshake-probe camera discovery, but talks to `cv2.VideoCapture` directly and never opens
a `VideoWriter` at all) -- opened lazily only while the preview is actually running, not held open
for the tool's whole lifetime the way Bpod/rotary/HiFi are, since nothing else in this tool needs
the camera connection to stay alive.
"""
import os
import re
import sys
import time
import traceback

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QKeySequence, QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QComboBox, QDoubleSpinBox,
    QPushButton, QLabel, QGroupBox, QMessageBox, QTextEdit, QShortcut)

import sound_calibration

_CALIBRATION_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_CALIBRATION_DIR, '..', '_projects', '_shared'))
import rotary_setup
import hifi_setup
import lick_input
from bpod_trial_helpers import was_visited
from camera_recorder import discover_camera, open_capture
from click_train_v2 import VAR_CLICK_DURATION_S, VAR_CLICK_RAMP_MS

from pybpod_hifi_module.module_api import HiFiModule
from pybpod_hifi_module.utils.generate_sound import pure_tone
from bpod_port import resolve_bpod_serial_port
from pybpodapi.protocol import Bpod, StateMachine

VAR_BPOD_SERIAL_PORT = None   # None = auto-detect from this box's own board config (bpod_port.py);
                              # set e.g. 'COM20' only to force a specific port.
VAR_VALVE_ID = 1                 # this rig's single valve (see CLAUDE.md).
VAR_LED_CHANNEL_NUMBER = 1       # PWM1, Port 1's built-in LED -- same convention every task script
                                  # already uses for a go-cue LED.
VAR_LICK_INPUT_CHOICES = ['BNC1', 'BNC2', 'Wire1', 'Wire2', 'Port1', 'Port2', 'Port3', 'Port4']
VAR_LICK_LISTEN_WINDOW_S = 1.0   # per-window duration for the repeated lick-monitor state machine
                                  # -- how often the monitor briefly hands control back to Python
                                  # between listens; not a limit on total monitoring time.
VAR_CHECK_SPL_DB = 70.0          # reference target for the tone check -- matches VAR_TARGET_SPL_DB
                                  # used across every real HiFi-using task script, so this checks
                                  # the real in-task volume, not an arbitrary one.
VAR_DEFAULT_TONE_DURATION_S = 2.0
VAR_TONE_ISI_S = 0.1             # fixed 10Hz click-train repeat rate, same as calibrate_sound.py's
                                  # own VAR_CALIBRATION_CLICK_ISI_S -- steady enough to judge by ear.
VAR_CAMERA_FPS = 30.0            # preview-loop rate limit -- same "measure elapsed, sleep the
                                  # remainder" throttle camera_recorder.py's own capture loop uses,
                                  # for the same reason (cv2.VideoCapture.read() doesn't reliably
                                  # block on every camera/backend combination).
VAR_CAMERA_PREVIEW_WIDTH_PX = 400   # displayed preview width (aspect-preserved) -- caps the
                                     # window's own size regardless of the camera's native
                                     # resolution.


class CameraCheckThread(QThread):
    """ Continuously grabs frames from a USB camera and emits each one -- NO video file or
    timestamps CSV is ever written (unlike `_shared/camera_recorder.py`'s `CameraRecorder`, which
    always writes both; see module docstring for why that class isn't used here). Opens/releases
    its own `cv2.VideoCapture` for the lifetime of one run() call only. """
    frame_ready = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, camera_index, fps=VAR_CAMERA_FPS):
        super(CameraCheckThread, self).__init__()
        self.camera_index = camera_index
        self.fps = fps
        self._stop_requested = False

    def stop(self):
        self._stop_requested = True

    def run(self):
        # same backend (and so the same device order) as discover_camera() -- see camera_recorder.py
        cap = open_capture(self.camera_index, self.fps)
        if cap is None:
            self.error.emit('Could not open camera index {0}'.format(self.camera_index))
            return
        try:
            frame_interval = (1.0 / self.fps) if self.fps > 0 else 0
            while not self._stop_requested:
                loop_start = time.time()
                ok, frame = cap.read()
                if ok and frame is not None:
                    self.frame_ready.emit(frame)
                sleep_time = frame_interval - (time.time() - loop_start)
                if sleep_time > 0:
                    time.sleep(sleep_time)
        except Exception as err:
            print(traceback.format_exc(), flush=True)
            self.error.emit(str(err))
        finally:
            cap.release()


INPUT_EDGE_RE = re.compile(r'^(Port\d+(In|Out)|BNC\d+(High|Low)|Wire\d+(High|Low))$')
_EDGE_SUFFIX_RE = re.compile(r'(In|Out|High|Low)$')


def ttl_edges(events):
    """ [(t, event_name), ...] sorted by time, for the digital-input edges in a trial's
    {event_name: [timestamps]} dict (get_all_timestamps_by_event()) -- Tup, SoftCodes and module
    events (RotaryEncoder1_N ...) are dropped. """
    return sorted((t, name) for name, times in events.items() if INPUT_EDGE_RE.match(name)
                  for t in times)


def edge_channel(name):
    """ 'BNC1High' -> 'BNC1', 'Port1In' -> 'Port1'. """
    return _EDGE_SUFFIX_RE.sub('', name)


def lick_edge_kind(name, channel):
    """ 'in'/'out' if `name` is an edge of lick input `channel`, else None. A port's In is a lick
    in; on a BNC/Wire input HIGH is a lick in (arduino_lick_relay drives HIGH while the beam is
    broken). """
    if edge_channel(name) != channel:
        return None
    return 'in' if name.endswith(('In', 'High')) else 'out'


class TtlMonitorThread(QThread):
    """ Repeatedly sends a short listening StateMachine while running, emitting ttl_edge
    (event_name, elapsed_s_since_monitor_started) for every digital-input edge Bpod logged -- see
    module docstring for why this repeated-short-listen approach is the only way to get continuous
    input timestamps at all outside a real Bpod session. stop() requests the loop end after its
    CURRENT window finishes (up to VAR_LICK_LISTEN_WINDOW_S later) -- it can't interrupt an
    in-flight run_state_machine() call, same "a background thread can never signal into an
    already-running call" constraint documented in CLAUDE.md. """
    ttl_edge = pyqtSignal(str, float)
    error = pyqtSignal(str)

    def __init__(self, my_bpod, listen_window_s):
        super(TtlMonitorThread, self).__init__()
        self.my_bpod = my_bpod
        self.listen_window_s = listen_window_s
        self._stop_requested = False

    def stop(self):
        self._stop_requested = True

    def run(self):
        first_bpod_start = None
        try:
            while not self._stop_requested:
                sma = StateMachine(self.my_bpod)
                # no input transitions: Bpod still logs every enabled input's edges, so the
                # window always runs its full length and catches every channel
                sma.add_state(
                    state_name='Listen',
                    state_timer=self.listen_window_s,
                    state_change_conditions={Bpod.Events.Tup: 'exit'},
                    output_actions=[])
                self.my_bpod.send_state_machine(sma)
                self.my_bpod.run_state_machine(sma)
                trial = self.my_bpod.session.current_trial
                # Bpod clock throughout: each window's start (TrialStartTimestamp) relative to
                # the first window's, plus the window-relative event time
                if first_bpod_start is None:
                    first_bpod_start = trial.trial_start_timestamp
                window_start_elapsed = trial.trial_start_timestamp - first_bpod_start
                for t, name in ttl_edges(trial.get_all_timestamps_by_event()):
                    self.ttl_edge.emit(name, window_start_elapsed + t)
        except Exception as err:
            print(traceback.format_exc(), flush=True)
            self.error.emit(str(err))


class RigCheckWindow(QWidget):
    def __init__(self):
        super(RigCheckWindow, self).__init__()
        self.setWindowTitle('Rig Check')

        self.my_bpod = None
        self.rotary = None
        self.hifi = None
        self._valve_open = False
        self._led_on = False
        self._lick_thread = None
        self._camera_thread = None

        self._connect_hardware()
        self._build_ui()
        self._populate_frequency_combo()

        self._position_timer = QTimer(self)
        self._position_timer.timeout.connect(self._update_position_label)
        self._position_timer.start(100)

        # Enter/Return toggles the valve, window-wide -- see module docstring for why this is the
        # ONE thing Enter does here (deliberately not also bound to Play Tone, unlike
        # calibrate_sound.py, to avoid two very different actions fighting over the same key).
        for key in (Qt.Key_Return, Qt.Key_Enter):
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.activated.connect(self._on_valve_toggle)

    # --- hardware connection (each wrapped independently -- one failure doesn't block the rest) --

    def _connect_hardware(self):
        try:
            self.my_bpod = Bpod(serial_port=VAR_BPOD_SERIAL_PORT or resolve_bpod_serial_port())
            self.bpod_status = 'Connected on {0}'.format(self.my_bpod.serial_port)
            print("Connected to Bpod on {0}".format(self.my_bpod.serial_port), flush=True)
        except Exception as err:
            print(traceback.format_exc(), flush=True)
            self.my_bpod = None
            self.bpod_status = 'FAILED: {0}'.format(err)

        if self.my_bpod is not None:
            try:
                self.rotary, _rotary_bpod_module = rotary_setup.connect_rotary(self.my_bpod)
                self.rotary_status = 'Connected on {0}'.format(
                    self.rotary.arcom.serial_object.port)
            except Exception as err:
                print(traceback.format_exc(), flush=True)
                self.rotary = None
                self.rotary_status = 'FAILED: {0}'.format(err)

            try:
                self.hifi = hifi_setup.connect_hifi(self.my_bpod)
                self.hifi_status = 'Connected on {0} ({1}Hz)'.format(
                    self.hifi.arcom.serial_object.port, self.hifi.sampling_rate)
            except Exception as err:
                print(traceback.format_exc(), flush=True)
                self.hifi = None
                self.hifi_status = 'FAILED: {0}'.format(err)
        else:
            self.rotary_status = 'Skipped (no Bpod connection)'
            # HiFi has its own direct-USB connection, independent of Bpod (see hifi_setup.py's own
            # docstring) -- still worth trying even if Bpod itself failed, same reasoning
            # calibrate_sound.py's own no-Bpod-at-all design already established.
            try:
                self.hifi = HiFiModule.discover()
                self.hifi_status = 'Connected on {0} ({1}Hz)'.format(
                    self.hifi.arcom.serial_object.port, self.hifi.sampling_rate)
            except Exception as err:
                print(traceback.format_exc(), flush=True)
                self.hifi = None
                self.hifi_status = 'FAILED: {0}'.format(err)

    # --- UI ----------------------------------------------------------------------------------------

    def _build_ui(self):
        layout = QVBoxLayout()

        status_group = QGroupBox('Hardware status')
        status_form = QFormLayout()
        self.bpod_status_label = QLabel(self.bpod_status)
        status_form.addRow('Bpod:', self.bpod_status_label)
        self.rotary_status_label = QLabel(self.rotary_status)
        status_form.addRow('Rotary encoder:', self.rotary_status_label)
        self.position_label = QLabel('-- deg')
        status_form.addRow('Wheel position (live):', self.position_label)
        self.hifi_status_label = QLabel(self.hifi_status)
        status_form.addRow('HiFi module:', self.hifi_status_label)
        status_group.setLayout(status_form)
        layout.addWidget(status_group)

        valve_group = QGroupBox('1. Water line priming -- Enter toggles open/closed')
        valve_form = QFormLayout()
        self.valve_status_label = QLabel('Valve: CLOSED')
        valve_form.addRow(self.valve_status_label)
        self.valve_button = QPushButton('Toggle Valve (Enter)')
        self.valve_button.clicked.connect(self._on_valve_toggle)
        valve_form.addRow(self.valve_button)
        valve_group.setLayout(valve_form)
        layout.addWidget(valve_group)

        led_group = QGroupBox('2. LED check')
        led_form = QFormLayout()
        self.led_status_label = QLabel('LED: OFF')
        led_form.addRow(self.led_status_label)
        self.led_button = QPushButton('Toggle LED')
        self.led_button.clicked.connect(self._on_led_toggle)
        led_form.addRow(self.led_button)
        led_group.setLayout(led_form)
        layout.addWidget(led_group)

        lick_group = QGroupBox('3. Lick sensor / TTL monitor (every Bpod input)')
        lick_form = QFormLayout()
        self.lick_input_combo = QComboBox()
        self.lick_input_combo.addItem('Auto (first input that fires)', None)
        for channel in VAR_LICK_INPUT_CHOICES:
            self.lick_input_combo.addItem(channel, channel)
        # start on this box's own lick input (_shared/lick_input.py: BNC2 on training01)
        box_index = self.lick_input_combo.findData(lick_input.lick_input())
        if box_index >= 0:
            self.lick_input_combo.setCurrentIndex(box_index)
        self.lick_input_combo.currentIndexChanged.connect(self._on_lick_input_changed)
        lick_form.addRow('Lick input:', self.lick_input_combo)
        self.lick_button = QPushButton('Start Lick Monitor')
        self.lick_button.clicked.connect(self._on_lick_monitor_toggle)
        lick_form.addRow(self.lick_button)
        self.lick_status_label = QLabel('Not running.')
        lick_form.addRow(self.lick_status_label)
        self.ttl_counts_label = QLabel('--')
        lick_form.addRow('Edges per input:', self.ttl_counts_label)
        self.lick_log = QTextEdit()
        self.lick_log.setReadOnly(True)
        self.lick_log.setMaximumHeight(160)
        lick_form.addRow(self.lick_log)
        lick_group.setLayout(lick_form)
        layout.addWidget(lick_group)

        tone_group = QGroupBox('4. Tone check (calibrated volume)')
        tone_form = QFormLayout()
        selector_row = QHBoxLayout()
        selector_row.addWidget(QLabel('Channel:'))
        self.channel_combo = QComboBox()
        self.channel_combo.addItem('Left (low freq)', 'L')
        self.channel_combo.addItem('Right (high freq)', 'R')
        self.channel_combo.currentIndexChanged.connect(self._on_channel_changed)
        selector_row.addWidget(self.channel_combo)
        selector_row.addWidget(QLabel('Frequency:'))
        self.frequency_combo = QComboBox()
        self.frequency_combo.currentIndexChanged.connect(self._update_amplitude_preview)
        selector_row.addWidget(self.frequency_combo)
        selector_row.addStretch(1)
        tone_form.addRow(selector_row)

        self.amplitude_label = QLabel('--')
        tone_form.addRow('Calibrated amplitude:', self.amplitude_label)

        self.tone_duration_input = QDoubleSpinBox()
        self.tone_duration_input.setDecimals(1)
        self.tone_duration_input.setRange(0.5, 30.0)
        self.tone_duration_input.setSuffix(' s')
        self.tone_duration_input.setValue(VAR_DEFAULT_TONE_DURATION_S)
        tone_form.addRow('Duration:', self.tone_duration_input)

        tone_buttons_row = QHBoxLayout()
        self.play_button = QPushButton('Play Tone')
        self.play_button.clicked.connect(self._on_play_tone)
        tone_buttons_row.addWidget(self.play_button)
        self.stop_tone_button = QPushButton('Stop Tone')
        self.stop_tone_button.clicked.connect(self._on_stop_tone)
        tone_buttons_row.addWidget(self.stop_tone_button)
        tone_form.addRow(tone_buttons_row)

        self.tone_status_label = QLabel('Ready.')
        tone_form.addRow(self.tone_status_label)
        tone_group.setLayout(tone_form)
        layout.addWidget(tone_group)

        camera_group = QGroupBox('5. Camera check')
        camera_form = QFormLayout()
        self.camera_button = QPushButton('Start Camera Preview')
        self.camera_button.clicked.connect(self._on_camera_toggle)
        camera_form.addRow(self.camera_button)
        self.camera_status_label = QLabel('Not running.')
        camera_form.addRow(self.camera_status_label)
        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumHeight(VAR_CAMERA_PREVIEW_WIDTH_PX * 3 // 4)
        self.camera_label.setStyleSheet('background-color: #222;')
        camera_form.addRow(self.camera_label)
        camera_group.setLayout(camera_form)
        layout.addWidget(camera_group)

        self.setLayout(layout)

    # --- hardware status / live position ------------------------------------------------------------

    def _update_position_label(self):
        if self.rotary is None:
            return
        try:
            pos = self.rotary.current_position()
            self.position_label.setText('{0:.2f} deg'.format(pos))
        except Exception:
            pass

    # --- valve / LED ---------------------------------------------------------------------------------

    def _on_valve_toggle(self):
        if self.my_bpod is None:
            QMessageBox.warning(self, 'No Bpod connection', 'Bpod is not connected.')
            return
        self._valve_open = not self._valve_open
        self.my_bpod.manual_override(Bpod.ChannelTypes.OUTPUT, Bpod.ChannelNames.VALVE,
                                      channel_number=VAR_VALVE_ID,
                                      value=1 if self._valve_open else 0)
        self.valve_status_label.setText('Valve: {0}'.format('OPEN' if self._valve_open else
                                                              'CLOSED'))

    def _on_led_toggle(self):
        if self.my_bpod is None:
            QMessageBox.warning(self, 'No Bpod connection', 'Bpod is not connected.')
            return
        self._led_on = not self._led_on
        self.my_bpod.manual_override(Bpod.ChannelTypes.OUTPUT, Bpod.ChannelNames.PWM,
                                      channel_number=VAR_LED_CHANNEL_NUMBER,
                                      value=255 if self._led_on else 0)
        self.led_status_label.setText('LED: {0}'.format('ON' if self._led_on else 'OFF'))

    # --- lick monitor ----------------------------------------------------------------------------

    def _on_lick_monitor_toggle(self):
        if self._lick_thread is not None and self._lick_thread.isRunning():
            self._lick_thread.stop()
            self.lick_button.setEnabled(False)
            self.lick_status_label.setText(
                'Stopping (finishing current {0:.1f}s window)...'.format(
                    VAR_LICK_LISTEN_WINDOW_S))
            return

        if self.my_bpod is None:
            QMessageBox.warning(self, 'No Bpod connection', 'Bpod is not connected.')
            return

        self.lick_log.clear()
        self._ttl_counts = {}
        self._lick_in_t = None
        self._n_licks = 0
        self._lick_channel = self.lick_input_combo.currentData()
        self.ttl_counts_label.setText('none yet')
        self._lick_thread = TtlMonitorThread(self.my_bpod, VAR_LICK_LISTEN_WINDOW_S)
        self._lick_thread.ttl_edge.connect(self._on_ttl_edge)
        self._lick_thread.error.connect(self._on_lick_monitor_error)
        self._lick_thread.finished.connect(self._on_lick_monitor_finished)
        self._lick_thread.start()
        self.lick_button.setText('Stop Lick Monitor')
        self._update_lick_status()

    def _on_lick_input_changed(self):
        self._lick_channel = self.lick_input_combo.currentData()
        self._lick_in_t = None
        self._update_lick_status()

    def _update_lick_status(self):
        if self._lick_thread is None or not self._lick_thread.isRunning():
            return
        if self._lick_channel is None:
            self.lick_status_label.setText('Listening on every input -- waiting for one to fire.')
        else:
            self.lick_status_label.setText('Listening -- lick input {0}, {1} lick(s).'.format(
                self._lick_channel, self._n_licks))

    def _on_ttl_edge(self, name, elapsed_s):
        channel = edge_channel(name)
        self._ttl_counts[channel] = self._ttl_counts.get(channel, 0) + 1
        self.ttl_counts_label.setText(', '.join(
            '{0}: {1}'.format(ch, n) for ch, n in sorted(self._ttl_counts.items())))
        if self._lick_channel is None:   # Auto: the first input that fires is the lick input
            self._lick_channel = channel
        kind = lick_edge_kind(name, self._lick_channel)
        if kind == 'in':
            self._lick_in_t = elapsed_s
            self._n_licks += 1
            self.lick_log.append('t={0:.4f}s  {1}  LICK IN'.format(elapsed_s, name))
        elif kind == 'out':
            contact = ('' if self._lick_in_t is None else
                       ' (contact {0:.0f} ms)'.format((elapsed_s - self._lick_in_t) * 1000))
            self._lick_in_t = None
            self.lick_log.append('t={0:.4f}s  {1}  LICK OUT{2}'.format(elapsed_s, name, contact))
        else:
            self.lick_log.append('t={0:.4f}s  {1}'.format(elapsed_s, name))
        self._update_lick_status()

    def _on_lick_monitor_error(self, message):
        QMessageBox.critical(self, 'Lick monitor failed', message)

    def _on_lick_monitor_finished(self):
        self.lick_button.setEnabled(True)
        self.lick_button.setText('Start Lick Monitor')
        self.lick_status_label.setText('Stopped.')

    # --- tone check ------------------------------------------------------------------------------

    def _current_channel(self):
        return self.channel_combo.currentData()

    def _current_frequency(self):
        return self.frequency_combo.currentData()

    def _populate_frequency_combo(self):
        self.frequency_combo.blockSignals(True)
        self.frequency_combo.clear()
        for freq in sound_calibration.CHANNEL_FREQUENCIES_HZ[self._current_channel()]:
            self.frequency_combo.addItem('{0} Hz'.format(freq), freq)
        self.frequency_combo.blockSignals(False)
        self._update_amplitude_preview()

    def _on_channel_changed(self):
        self._populate_frequency_combo()

    def _update_amplitude_preview(self):
        frequency_hz = self._current_frequency()
        if frequency_hz is None:
            return
        amplitude = sound_calibration.get_calibrated_amplitude(
            VAR_CHECK_SPL_DB, frequency_hz, channel=self._current_channel())
        self.amplitude_label.setText('{0:.4f} (target {1:.0f}dB SPL)'.format(
            amplitude, VAR_CHECK_SPL_DB))

    def _on_play_tone(self):
        if self.hifi is None:
            QMessageBox.warning(self, 'No HiFi connection', 'HiFi module is not connected.')
            return
        channel = self._current_channel()
        frequency_hz = self._current_frequency()
        duration_s = self.tone_duration_input.value()
        amplitude = sound_calibration.get_calibrated_amplitude(
            VAR_CHECK_SPL_DB, frequency_hz, channel=channel)

        try:
            # Repeated click train, not one long tone -- see module docstring for why. Same click
            # shape click_train_v2.py's own build_waveform() uses, imported directly.
            click = pure_tone(VAR_CLICK_DURATION_S, frequency_hz, self.hifi.sampling_rate,
                               ramp_ms=VAR_CLICK_RAMP_MS) * amplitude
            sampling_rate = self.hifi.sampling_rate
            n_total = int(round(duration_s * sampling_rate))
            isi_samples = int(round(VAR_TONE_ISI_S * sampling_rate))
            track = np.zeros(n_total)
            for start in range(0, n_total - len(click) + 1, isi_samples):
                track[start:start + len(click)] += click
            silence = np.zeros_like(track)
            waveform = np.array([track, silence]) if channel == 'L' else np.array([silence, track])
            self.hifi.load(0, waveform)
            self.hifi.push()
            self.hifi.play(0)
        except Exception as err:
            print(traceback.format_exc(), flush=True)
            QMessageBox.critical(self, 'Playback failed', "{0}\n\n{1}".format(
                err, traceback.format_exc()))
            return

        self.tone_status_label.setText(
            'Playing {0}Hz click train on {1} at amplitude {2:.4f} (target {3:.0f}dB SPL) for '
            '{4:.1f}s.'.format(frequency_hz, channel, amplitude, VAR_CHECK_SPL_DB, duration_s))

    def _on_stop_tone(self):
        if self.hifi is None:
            return
        try:
            self.hifi.stop()
        except Exception as err:
            print(traceback.format_exc(), flush=True)
            QMessageBox.critical(self, 'Stop failed', str(err))
            return
        self.tone_status_label.setText('Stopped.')

    # --- camera check ----------------------------------------------------------------------------

    def _on_camera_toggle(self):
        if self._camera_thread is not None and self._camera_thread.isRunning():
            self._camera_thread.stop()
            self.camera_button.setEnabled(False)
            self.camera_status_label.setText('Stopping...')
            return

        try:
            camera_index = discover_camera()
        except Exception as err:
            print(traceback.format_exc(), flush=True)
            QMessageBox.critical(self, 'Camera not found', str(err))
            return

        self._camera_thread = CameraCheckThread(camera_index)
        self._camera_thread.frame_ready.connect(self._on_camera_frame)
        self._camera_thread.error.connect(self._on_camera_error)
        self._camera_thread.finished.connect(self._on_camera_finished)
        self._camera_thread.start()
        self.camera_button.setText('Stop Camera Preview')
        self.camera_status_label.setText('Camera index {0} -- previewing...'.format(camera_index))

    def _on_camera_frame(self, frame_bgr):
        """ frame_bgr is a raw cv2 frame (BGR, uint8, HxWx3) -- same conversion
        camera_recorder.py's own _PreviewWidget.set_frame() uses (Qt has no native BGR888 format;
        np.ascontiguousarray keeps the buffer alive/contiguous for as long as the QImage needs
        it). Scaled to VAR_CAMERA_PREVIEW_WIDTH_PX so the window's own size doesn't depend on the
        camera's native resolution. """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        h, w, _ch = rgb.shape
        qimage = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage).scaledToWidth(
            VAR_CAMERA_PREVIEW_WIDTH_PX, Qt.SmoothTransformation)
        self.camera_label.setPixmap(pixmap)

    def _on_camera_error(self, message):
        QMessageBox.critical(self, 'Camera error', message)

    def _on_camera_finished(self):
        self.camera_button.setEnabled(True)
        self.camera_button.setText('Start Camera Preview')
        self.camera_status_label.setText('Stopped.')
        self.camera_label.clear()

    # --- shutdown --------------------------------------------------------------------------------

    def closeEvent(self, event):
        if self._camera_thread is not None and self._camera_thread.isRunning():
            self._camera_thread.stop()
            self._camera_thread.wait(2000)
        if self._lick_thread is not None and self._lick_thread.isRunning():
            self._lick_thread.stop()
            self._lick_thread.wait(int((VAR_LICK_LISTEN_WINDOW_S + 1.0) * 1000))
        if self.my_bpod is not None:
            if self._valve_open:
                self.my_bpod.manual_override(Bpod.ChannelTypes.OUTPUT, Bpod.ChannelNames.VALVE,
                                              channel_number=VAR_VALVE_ID, value=0)
            if self._led_on:
                self.my_bpod.manual_override(Bpod.ChannelTypes.OUTPUT, Bpod.ChannelNames.PWM,
                                              channel_number=VAR_LED_CHANNEL_NUMBER, value=0)
            self.my_bpod.close()
        if self.hifi is not None:
            self.hifi.close()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = RigCheckWindow()
    window.resize(520, 820)
    window.show()
    sys.exit(app.exec_())
