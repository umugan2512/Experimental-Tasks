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

**Lick sensor**: `Port1In` has no polling/`manual_override`-read API at all -- it's only observable
while a `StateMachine` is actively listening via `state_change_conditions` (see this repo's own
`CLAUDE.md`, Hardware module notes). `LickMonitorThread` gets real, continuous lick timestamps
anyway by repeatedly sending a short listening state machine
(`Listen` -> `Port1In`: `Licked`, `Tup`: `exit`) back-to-back on a background `QThread` (same
"background thread for anything that blocks on real Bpod time" pattern as `calibrate_liquid.py`'s
own `PulseRunner`) for as long as the monitor is running, logging each lick's own real elapsed time
since the monitor started (not just "a lick happened at some point").

**Tone check**: same channel/frequency selector as `calibrate_sound.py`, but the amplitude is
looked up live via `sound_calibration.get_calibrated_amplitude()` at a fixed `VAR_CHECK_SPL_DB`
reference (default matches every real task script's own `VAR_TARGET_SPL_DB`) -- so this checks
"does the real in-task volume actually sound right," not just "is a tone audible at all." Same
repeated-click-train playback shape `calibrate_sound.py` uses (not one continuous tone -- an SPL
meter/the ear can't time-integrate a single 8ms click pip the way it does a sustained tone; see that
file's own docstring), built from `click_train_v2.py`'s own click shape so it matches what a real
task actually plays.
"""
import os
import sys
import time
import traceback

import numpy as np
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QComboBox, QDoubleSpinBox,
    QPushButton, QLabel, QGroupBox, QMessageBox, QTextEdit, QShortcut)

import sound_calibration

_CALIBRATION_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_CALIBRATION_DIR, '..', '_projects', '_shared'))
import rotary_setup
import hifi_setup
from bpod_trial_helpers import was_visited
from click_train_v2 import VAR_CLICK_DURATION_S, VAR_CLICK_RAMP_MS

from pybpod_hifi_module.module_api import HiFiModule
from pybpod_hifi_module.utils.generate_sound import pure_tone
from pybpodapi.protocol import Bpod, StateMachine

VAR_BPOD_SERIAL_PORT = 'COM10'   # MACHINE-SPECIFIC -- same "edit per box" convention as
                                  # flush_valve.py/calibrate_liquid.py's own VAR_BPOD_SERIAL_PORT.
VAR_VALVE_ID = 1                 # this rig's single valve (see CLAUDE.md).
VAR_LED_CHANNEL_NUMBER = 1       # PWM1, Port 1's built-in LED -- same convention every task script
                                  # already uses for a go-cue LED.
VAR_LICK_LISTEN_WINDOW_S = 1.0   # per-window duration for the repeated lick-monitor state machine
                                  # -- how often the monitor briefly hands control back to Python
                                  # between listens; not a limit on total monitoring time.
VAR_CHECK_SPL_DB = 70.0          # reference target for the tone check -- matches VAR_TARGET_SPL_DB
                                  # used across every real HiFi-using task script, so this checks
                                  # the real in-task volume, not an arbitrary one.
VAR_DEFAULT_TONE_DURATION_S = 2.0
VAR_TONE_ISI_S = 0.1             # fixed 10Hz click-train repeat rate, same as calibrate_sound.py's
                                  # own VAR_CALIBRATION_CLICK_ISI_S -- steady enough to judge by ear.


class LickMonitorThread(QThread):
    """ Repeatedly sends a short listening StateMachine while running, emitting lick_detected
    (elapsed_s_since_monitor_started) for every real Port1In event observed -- see module docstring
    for why this repeated-short-listen approach is the only way to get continuous lick timestamps
    at all outside a real Bpod session. stop() requests the loop end after its CURRENT window
    finishes (up to VAR_LICK_LISTEN_WINDOW_S later) -- it can't interrupt an in-flight
    run_state_machine() call, same "a background thread can never signal into an already-running
    call" constraint documented in CLAUDE.md. """
    lick_detected = pyqtSignal(float)
    error = pyqtSignal(str)

    def __init__(self, my_bpod, listen_window_s):
        super(LickMonitorThread, self).__init__()
        self.my_bpod = my_bpod
        self.listen_window_s = listen_window_s
        self._stop_requested = False

    def stop(self):
        self._stop_requested = True

    def run(self):
        start_time = time.time()
        try:
            while not self._stop_requested:
                window_start_elapsed = time.time() - start_time
                sma = StateMachine(self.my_bpod)
                sma.add_state(
                    state_name='Listen',
                    state_timer=self.listen_window_s,
                    state_change_conditions={'Port1In': 'Licked', Bpod.Events.Tup: 'exit'},
                    output_actions=[])
                sma.add_state(
                    state_name='Licked',
                    state_timer=0,
                    state_change_conditions={Bpod.Events.Tup: 'exit'},
                    output_actions=[])
                self.my_bpod.send_state_machine(sma)
                self.my_bpod.run_state_machine(sma)
                if self._stop_requested:
                    break
                visited = self.my_bpod.session.current_trial.states_durations
                if was_visited(visited, 'Licked'):
                    events = self.my_bpod.session.current_trial.get_all_timestamps_by_event()
                    for t in events.get('Port1In', []):
                        self.lick_detected.emit(window_start_elapsed + t)
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
            self.my_bpod = Bpod(serial_port=VAR_BPOD_SERIAL_PORT)
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

        lick_group = QGroupBox('3. Lick sensor monitor')
        lick_form = QFormLayout()
        self.lick_button = QPushButton('Start Lick Monitor')
        self.lick_button.clicked.connect(self._on_lick_monitor_toggle)
        lick_form.addRow(self.lick_button)
        self.lick_status_label = QLabel('Not running.')
        lick_form.addRow(self.lick_status_label)
        self.lick_log = QTextEdit()
        self.lick_log.setReadOnly(True)
        self.lick_log.setMaximumHeight(120)
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
        self._lick_thread = LickMonitorThread(self.my_bpod, VAR_LICK_LISTEN_WINDOW_S)
        self._lick_thread.lick_detected.connect(self._on_lick_detected)
        self._lick_thread.error.connect(self._on_lick_monitor_error)
        self._lick_thread.finished.connect(self._on_lick_monitor_finished)
        self._lick_thread.start()
        self.lick_button.setText('Stop Lick Monitor')
        self.lick_status_label.setText('Listening...')

    def _on_lick_detected(self, elapsed_s):
        self.lick_log.append('Lick at t={0:.3f}s'.format(elapsed_s))

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

    # --- shutdown --------------------------------------------------------------------------------

    def closeEvent(self, event):
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
