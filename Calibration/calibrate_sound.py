# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Interactive HiFi output-level (SPL) calibration tool -- PyQt5 GUI equivalent of
`calibrate_liquid.py`, for sound instead of valve timing. See `sound_calibration.py`'s own module
docstring for the calibration approach itself (per-(channel, frequency_hz) (amplitude, dB SPL)
points, 2nd-order polynomial fit, MATLAB-derived precedent via `liquid_calibration.py`) -- this
file is purely the interactive data-collection front end.

Run directly: `python calibrate_sound.py` (needs the HiFi module physically connected, plus an
external sound-level meter positioned at the animal's head position). Deliberately **not** a
PyBpod GUI task/plugin, same reasoning as `calibrate_liquid.py` -- calibration is a bench
maintenance procedure, not project/session-tracked experiment data, so this connects directly to
the HiFi module's own direct-USB connection and skips all PyBpod project/task scaffolding. No
`Bpod()` object is needed at all here (unlike `calibrate_liquid.py`) -- `HiFiModule.discover()`'s
own `exclude_ports` parameter defaults to `()`, so there's no Bpod serial port that needs excluding
when this tool has no Bpod connection in the first place.

**No `QThread` here, unlike `calibrate_liquid.py`'s `PulseRunner`** -- a deliberate simplification.
Firing N Bpod-timed valve pulses sequentially genuinely blocks the Qt event loop for real
wall-clock time (each pulse's own `state_timer` must elapse before the next can fire), which is
why that tool needs a background thread. `hifi.load()`/`push()`/`play()` just send USB commands and
return immediately -- the tone then plays out autonomously on the HiFi module's own hardware, with
no Python-side waiting required. A single, several-seconds-long one-shot tone buffer (the exact
same `load(0, waveform); push(); play(0)` call shape every task script in this codebase already
uses -- no untested `loop_mode=1` path) is enough for a steady meter reading; "Stop Tone" just
calls `hifi.stop()` directly and synchronously from the main thread for early cutoff.

**Channel/frequency pairing is fixed, not freely choosable**: left is always the low-frequency
channel, right always the high-frequency channel (`sound_calibration.CHANNEL_FREQUENCIES_HZ`),
matching the convention `click_train_v2.py` already established. The frequency selector is
repopulated from that same dict whenever the channel selector changes, rather than offering a
free-standing list of all four frequencies regardless of channel.

**Calibration stimulus is a repeated click train, not one long continuous tone** -- a real,
confirmed-on-hardware mismatch: an SPL meter's time-integration (even on "fast" response, ~125ms)
cannot fully catch up to a single `VAR_CLICK_DURATION_S`-long (8ms) pip, so a click played at the
SAME peak amplitude as a long calibration tone reads (and sounds) noticeably quieter than the tone
did -- the peak-amplitude-to-dB mapping from a sustained tone simply doesn't transfer to something
that brief. Fixed by building the actual calibration playback out of `click_train_v2.py`'s own
`VAR_CLICK_DURATION_S`/`VAR_CLICK_RAMP_MS` click shape (imported directly, not duplicated, so this
can never silently drift from what a real task actually plays), repeated at a fixed
`VAR_CALIBRATION_CLICK_ISI_S` interval for the requested play duration -- steady enough for the
meter to read, while being duty-cycle-representative of the real stimulus instead of a continuous
tone.
"""
import os
import sys
import traceback

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QComboBox, QDoubleSpinBox,
    QPushButton, QLabel, QTableWidget, QTableWidgetItem, QGroupBox, QMessageBox,
    QAbstractItemView, QShortcut)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

import sound_calibration

_CALIBRATION_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_CALIBRATION_DIR, '..', '_projects', '_shared'))
from click_train_v2 import VAR_CLICK_DURATION_S, VAR_CLICK_RAMP_MS   # the real task's own click
                                                                      # shape -- see module docstring

from pybpod_hifi_module.module_api import HiFiModule
from pybpod_hifi_module.utils.generate_sound import pure_tone

VAR_HIFI_USB_PORT = None       # None = auto-discover via HiFiModule.discover(); MACHINE-SPECIFIC
                                # override (e.g. 'COM7') if auto-discovery ever picks the wrong
                                # port on a box with multiple similar USB devices connected.
VAR_DEFAULT_AMPLITUDE = 0.1    # start quiet, not at max -- a deliberate safety-conscious default,
                                # increase gradually while watching the meter.
VAR_DEFAULT_DURATION_S = 5.0   # total playback duration (a click train, not one tone -- see module
                                # docstring); long enough for a steady meter reading, adjustable.
VAR_CALIBRATION_CLICK_ISI_S = 0.1   # fixed 10Hz repeat rate for the calibration click train --
                                     # independent of any real task's own variable/Poisson click
                                     # rate, this only needs to be fast enough for a steady meter
                                     # reading; the click SHAPE (duration/ramp) is what must match
                                     # the real task, not the repeat rate.

# Fixed per-series plot colors, keyed by (channel, frequency_hz) -- so a given channel/frequency's
# color stays the same between the interactive plot and the all-curves summary plot.
_SERIES_COLORS = {
    ('L', 4000): 'tab:blue', ('L', 5000): 'tab:cyan',
    ('R', 10000): 'tab:red', ('R', 12000): 'tab:orange',
}


class CalibrationWindow(QWidget):
    def __init__(self):
        super(CalibrationWindow, self).__init__()
        self.setWindowTitle('Sound Calibration')

        self.calibration = sound_calibration.SoundCalibration()

        if VAR_HIFI_USB_PORT:
            self.hifi = HiFiModule(VAR_HIFI_USB_PORT)
        else:
            self.hifi = HiFiModule.discover()
        print("Connected to HiFi module on {0} ({1}Hz sampling)".format(
            self.hifi.arcom.serial_object.port, self.hifi.sampling_rate), flush=True)

        self._pending_readings = []   # in-progress meter readings for the point being built

        self._build_ui()
        self._populate_frequency_combo()
        self._refresh_table()

        # Enter/Return re-plays the tone without reaching for the mouse -- added per explicit
        # request: with one hand holding the sound-level meter at the animal's head position,
        # clicking "Play Tone" between readings is hard to time cleanly by hand. Both Key_Return
        # (main keyboard) and Key_Enter (numpad) are bound, application-wide (not tied to a single
        # widget's focus), so it works regardless of which field currently has focus.
        for key in (Qt.Key_Return, Qt.Key_Enter):
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.activated.connect(self._on_play_tone)

    def _build_ui(self):
        layout = QVBoxLayout()

        selector_row = QHBoxLayout()
        selector_row.addWidget(QLabel('Channel:'))
        self.channel_combo = QComboBox()
        self.channel_combo.addItem('Left (low freq)', 'L')
        self.channel_combo.addItem('Right (high freq)', 'R')
        self.channel_combo.currentIndexChanged.connect(self._on_channel_changed)
        selector_row.addWidget(self.channel_combo)

        selector_row.addWidget(QLabel('Frequency:'))
        self.frequency_combo = QComboBox()   # populated by _populate_frequency_combo() below,
                                              # filtered by whichever channel is selected
        self.frequency_combo.currentIndexChanged.connect(self._on_frequency_changed)
        selector_row.addWidget(self.frequency_combo)
        selector_row.addStretch(1)
        layout.addLayout(selector_row)

        play_group = QGroupBox('1. Play tone')
        play_form = QFormLayout()
        self.amplitude_input = QDoubleSpinBox()
        self.amplitude_input.setDecimals(3)
        self.amplitude_input.setRange(0.0, 1.0)
        self.amplitude_input.setSingleStep(0.01)
        self.amplitude_input.setValue(VAR_DEFAULT_AMPLITUDE)
        play_form.addRow('Amplitude (0-1):', self.amplitude_input)

        self.duration_input = QDoubleSpinBox()
        self.duration_input.setDecimals(1)
        self.duration_input.setRange(0.5, 60.0)
        self.duration_input.setSuffix(' s')
        self.duration_input.setValue(VAR_DEFAULT_DURATION_S)
        play_form.addRow('Tone duration:', self.duration_input)

        play_buttons_row = QHBoxLayout()
        self.play_button = QPushButton('Play Tone (Enter)')
        self.play_button.clicked.connect(self._on_play_tone)
        play_buttons_row.addWidget(self.play_button)
        self.stop_button = QPushButton('Stop Tone')
        self.stop_button.clicked.connect(self._on_stop_tone)
        play_buttons_row.addWidget(self.stop_button)
        play_form.addRow(play_buttons_row)

        self.status_label = QLabel('Ready.')
        play_form.addRow(self.status_label)
        play_group.setLayout(play_form)
        layout.addWidget(play_group)

        readings_group = QGroupBox('2. Enter meter readings, add point')
        readings_form = QFormLayout()
        self.reading_input = QDoubleSpinBox()
        self.reading_input.setDecimals(1)
        self.reading_input.setRange(0.0, 140.0)
        self.reading_input.setSuffix(' dB SPL')
        readings_form.addRow('Meter reading:', self.reading_input)

        reading_buttons_row = QHBoxLayout()
        self.add_reading_button = QPushButton('Add Reading')
        self.add_reading_button.clicked.connect(self._on_add_reading)
        reading_buttons_row.addWidget(self.add_reading_button)
        self.clear_readings_button = QPushButton('Clear Readings')
        self.clear_readings_button.clicked.connect(self._on_clear_readings)
        reading_buttons_row.addWidget(self.clear_readings_button)
        readings_form.addRow(reading_buttons_row)

        self.pending_label = QLabel('No readings entered yet.')
        self.pending_label.setWordWrap(True)
        readings_form.addRow(self.pending_label)

        self.commit_button = QPushButton('Commit Point (average of readings above)')
        self.commit_button.clicked.connect(self._on_commit_point)
        readings_form.addRow(self.commit_button)
        readings_group.setLayout(readings_form)
        layout.addWidget(readings_group)

        self.points_table = QTableWidget(0, 4)
        self.points_table.setHorizontalHeaderLabels(
            ['Amplitude', 'Mean SPL (dB)', 'N readings', 'Std (dB)'])
        self.points_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.points_table.setSelectionMode(QAbstractItemView.SingleSelection)
        layout.addWidget(self.points_table)

        self.delete_point_button = QPushButton('Delete Selected Point')
        self.delete_point_button.clicked.connect(self._on_delete_point)
        layout.addWidget(self.delete_point_button)

        fit_group = QGroupBox('3. Fit and look up')
        fit_form = QFormLayout()
        self.fit_button = QPushButton('Fit This Channel/Frequency')
        self.fit_button.clicked.connect(self._on_fit)
        fit_form.addRow(self.fit_button)

        self.coeffs_label = QLabel('(not fit yet)')
        fit_form.addRow('Fit coefficients:', self.coeffs_label)

        self.lookup_input = QDoubleSpinBox()
        self.lookup_input.setDecimals(1)
        self.lookup_input.setRange(0.0, 140.0)
        self.lookup_input.setSuffix(' dB SPL')
        self.lookup_input.setValue(60.0)
        self.lookup_input.valueChanged.connect(self._update_lookup_preview)
        fit_form.addRow('Preview target SPL:', self.lookup_input)

        self.lookup_result_label = QLabel('(no fit yet)')
        fit_form.addRow('-> amplitude:', self.lookup_result_label)
        fit_group.setLayout(fit_form)
        layout.addWidget(fit_group)

        self.figure = Figure(figsize=(4, 3))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax = self.figure.add_subplot(111)
        layout.addWidget(self.canvas)

        all_curves_group = QGroupBox('All calibration curves')
        all_curves_layout = QVBoxLayout()
        self.all_figure = Figure(figsize=(4, 3))
        self.all_canvas = FigureCanvasQTAgg(self.all_figure)
        self.all_ax = self.all_figure.add_subplot(111)
        all_curves_layout.addWidget(self.all_canvas)
        all_curves_group.setLayout(all_curves_layout)
        layout.addWidget(all_curves_group)

        self.setLayout(layout)

    # --- selectors -----------------------------------------------------------------------------

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

    def _on_channel_changed(self):
        self._populate_frequency_combo()
        self._refresh_table()

    def _on_frequency_changed(self):
        self._refresh_table()

    # --- tone playback ---------------------------------------------------------------------------

    def _on_play_tone(self):
        channel = self._current_channel()
        frequency_hz = self._current_frequency()
        amplitude = self.amplitude_input.value()
        duration_s = self.duration_input.value()

        try:
            # Repeated click train, not one long tone -- see module docstring for why (a meter
            # can't time-integrate a single 8ms pip the way it does a sustained tone, so
            # calibrating with a long tone doesn't transfer to what a real click sounds/measures
            # like). Same click shape click_train_v2.py's own build_waveform() uses, imported
            # directly so it can't drift from the real task's stimulus.
            click = pure_tone(VAR_CLICK_DURATION_S, frequency_hz, self.hifi.sampling_rate,
                               ramp_ms=VAR_CLICK_RAMP_MS) * amplitude
            sampling_rate = self.hifi.sampling_rate
            n_total = int(round(duration_s * sampling_rate))
            isi_samples = int(round(VAR_CALIBRATION_CLICK_ISI_S * sampling_rate))
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

        self.status_label.setText(
            'Playing {0}Hz click train on {1} at amplitude {2:.3f} for {3:.1f}s ({4:.0f}Hz repeat '
            'rate) -- read the meter, then "Add Reading" below (repeat while it plays for multiple '
            'readings).'.format(frequency_hz, channel, amplitude, duration_s,
                                 1.0 / VAR_CALIBRATION_CLICK_ISI_S))

    def _on_stop_tone(self):
        try:
            self.hifi.stop()
        except Exception as err:
            print(traceback.format_exc(), flush=True)
            QMessageBox.critical(self, 'Stop failed', str(err))
            return
        self.status_label.setText('Stopped.')

    # --- meter readings / point commit --------------------------------------------------------

    def _on_add_reading(self):
        self._pending_readings.append(self.reading_input.value())
        self._refresh_pending_label()

    def _on_clear_readings(self):
        self._pending_readings = []
        self._refresh_pending_label()

    def _refresh_pending_label(self):
        if not self._pending_readings:
            self.pending_label.setText('No readings entered yet.')
            return
        mean_db = sum(self._pending_readings) / len(self._pending_readings)
        text = 'n={0}: {1}  ->  mean={2:.2f} dB'.format(
            len(self._pending_readings),
            ', '.join('{0:.1f}'.format(v) for v in self._pending_readings), mean_db)
        if len(self._pending_readings) > 1:
            std_db = (sum((v - mean_db) ** 2 for v in self._pending_readings)
                      / len(self._pending_readings)) ** 0.5
            text += ', std={0:.2f} dB'.format(std_db)
        self.pending_label.setText(text)

    def _on_commit_point(self):
        if not self._pending_readings:
            QMessageBox.warning(self, 'No readings',
                                 'Add at least one meter reading first.')
            return
        channel = self._current_channel()
        frequency_hz = self._current_frequency()
        amplitude = self.amplitude_input.value()
        mean_db, std_db = self.calibration.add_measurement(
            channel, frequency_hz, amplitude, list(self._pending_readings))
        self.status_label.setText(
            'Added point: amplitude={0:.3f} -> {1:.2f}dB (n={2}, std={3:.2f}).'.format(
                amplitude, mean_db, len(self._pending_readings), std_db))
        self._pending_readings = []
        self._refresh_pending_label()
        self._refresh_table()

    def _on_delete_point(self):
        channel = self._current_channel()
        frequency_hz = self._current_frequency()
        selected_rows = self.points_table.selectionModel().selectedRows()
        if not selected_rows:
            QMessageBox.warning(self, 'No point selected',
                                 'Click a row in the points table first, then Delete Selected '
                                 'Point.')
            return

        row = selected_rows[0].row()
        amplitude, mean_db, n_readings, _std_db = self.calibration.table(
            channel, frequency_hz)[row]
        reply = QMessageBox.question(
            self, 'Delete point',
            'Delete this point?\n\namplitude={0:.4f} -> {1:.2f}dB (n={2})\n\n'
            'Any existing fit for this channel/frequency will be cleared -- fit again '
            'afterward.'.format(amplitude, mean_db, n_readings))
        if reply != QMessageBox.Yes:
            return

        self.calibration.remove_point(channel, frequency_hz, row)
        self.status_label.setText('Deleted point: amplitude={0:.4f} -> {1:.2f}dB.'.format(
            amplitude, mean_db))
        self._refresh_table()

    # --- fit / lookup ----------------------------------------------------------------------------

    def _on_fit(self):
        channel = self._current_channel()
        frequency_hz = self._current_frequency()
        try:
            coeffs = self.calibration.fit(channel, frequency_hz)
            self.coeffs_label.setText('[{0:.6g}, {1:.6g}, {2:.6g}]'.format(*coeffs))
        except ValueError as err:
            QMessageBox.warning(self, 'Cannot fit', str(err))
            return
        self._update_lookup_preview()
        self._refresh_plot()
        self._refresh_all_curves_plot()

    def _update_lookup_preview(self):
        channel = self._current_channel()
        frequency_hz = self._current_frequency()
        target_db = self.lookup_input.value()
        try:
            amplitude = self.calibration.get_amplitude_for_spl(channel, frequency_hz, target_db)
            self.lookup_result_label.setText('{0:.4f}'.format(amplitude))
        except ValueError:
            self.lookup_result_label.setText('(no fit yet)')

    # --- table / plots -----------------------------------------------------------------------------

    def _refresh_table(self):
        channel = self._current_channel()
        frequency_hz = self._current_frequency()
        table = self.calibration.table(channel, frequency_hz)
        self.points_table.setRowCount(len(table))
        for row, (amplitude, mean_db, n_readings, std_db) in enumerate(table):
            self.points_table.setItem(row, 0, QTableWidgetItem('{0:.4f}'.format(amplitude)))
            self.points_table.setItem(row, 1, QTableWidgetItem('{0:.2f}'.format(mean_db)))
            self.points_table.setItem(row, 2, QTableWidgetItem(str(n_readings)))
            self.points_table.setItem(row, 3, QTableWidgetItem('{0:.2f}'.format(std_db)))

        coeffs = self.calibration.coeffs(channel, frequency_hz)
        if coeffs:
            self.coeffs_label.setText('[{0:.6g}, {1:.6g}, {2:.6g}]'.format(*coeffs))
        else:
            self.coeffs_label.setText('(not fit yet)')

        self._update_lookup_preview()
        self._refresh_plot()
        self._refresh_all_curves_plot()

    def _refresh_plot(self):
        """ Scatters the selected (channel, frequency)'s raw (amplitude, mean dB SPL) points
        (error bars from std dB), and -- if a fit exists -- overlays the fitted curve. x=amplitude,
        y=SPL is the natural "what SPL does this amplitude produce" view, even though the stored
        fit runs the opposite direction (amplitude = f(spl), see sound_calibration.py's own
        docstring for why) -- the fit line is drawn by sampling an SPL range and evaluating
        amplitude = polyval(coeffs, spl), which traces the same curve regardless of which
        direction the polynomial itself was fit in. """
        channel = self._current_channel()
        frequency_hz = self._current_frequency()
        table = self.calibration.table(channel, frequency_hz)
        coeffs = self.calibration.coeffs(channel, frequency_hz)

        self.ax.clear()
        if table:
            amplitudes = [row[0] for row in table]
            spls = [row[1] for row in table]
            stds = [row[3] for row in table]
            self.ax.errorbar(amplitudes, spls, yerr=stds, fmt='o', color='tab:blue',
                              label='Measured points', capsize=3)

            if coeffs:
                spl_min, spl_max = min(spls), max(spls)
                spl_span = max(spl_max - spl_min, 1e-6)
                spl_range = np.linspace(spl_min - 0.1 * spl_span, spl_max + 0.1 * spl_span, 100)
                amp_fit = np.polyval(coeffs, spl_range)
                self.ax.plot(amp_fit, spl_range, color='tab:orange', label='Fit')

            self.ax.legend(loc='best')

        self.ax.set_xlabel('Amplitude (waveform scale)')
        self.ax.set_ylabel('SPL (dB)')
        self.ax.set_title('{0} {1}Hz calibration'.format(channel, frequency_hz))
        self.figure.tight_layout()
        self.canvas.draw()

    def _refresh_all_curves_plot(self):
        """ Every (channel, frequency) pair with at least one point, all on one plot -- the
        "all the different calibration values...noted in a graph" view. Always current, redrawn
        on every table refresh (i.e. every add/delete/fit), not just when this specific pair
        changes. """
        self.all_ax.clear()
        any_data = False
        for channel, frequency_hz, entry in self.calibration.all_entries():
            any_data = True
            table = entry['table']
            amplitudes = [row[0] for row in table]
            spls = [row[1] for row in table]
            color = _SERIES_COLORS.get((channel, frequency_hz))
            label = '{0} {1:g}kHz'.format(channel, frequency_hz / 1000.0)
            self.all_ax.scatter(amplitudes, spls, color=color, label=label)
            coeffs = entry.get('coeffs')
            if coeffs:
                spl_min, spl_max = min(spls), max(spls)
                spl_span = max(spl_max - spl_min, 1e-6)
                spl_range = np.linspace(spl_min - 0.1 * spl_span, spl_max + 0.1 * spl_span, 50)
                amp_fit = np.polyval(coeffs, spl_range)
                self.all_ax.plot(amp_fit, spl_range, color=color)

        if any_data:
            self.all_ax.legend(loc='best', fontsize=8)
        self.all_ax.set_xlabel('Amplitude (waveform scale)')
        self.all_ax.set_ylabel('SPL (dB)')
        self.all_ax.set_title('All calibration curves')
        self.all_figure.tight_layout()
        self.all_canvas.draw()

    def closeEvent(self, event):
        self.hifi.close()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CalibrationWindow()
    window.resize(560, 980)
    window.show()
    sys.exit(app.exec_())
