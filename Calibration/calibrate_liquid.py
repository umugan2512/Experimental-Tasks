# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Interactive liquid valve calibration tool -- Python/PyQt5 equivalent of Sanworks MATLAB Bpod's own
`BpodLiquidCalibration('Calibrate')` GUI (dropdown valve selection, editable duration/pulse-count/
mass fields), not a command-line prompt sequence. See `liquid_calibration.py`'s own module
docstring for the calibration approach itself (per-valve (duration_ms, volume_uL) points, 2nd-order
polynomial fit, MATLAB-derived) -- this file is purely the interactive data-collection front end.

Run directly: `python calibrate_liquid.py` (needs the Bpod board and valve physically connected,
plus a scale to weigh dispensed liquid). Deliberately NOT a PyBpod GUI task/plugin -- calibration is
a bench maintenance procedure, not project/session-tracked experiment data, so this connects to the
Bpod board directly (same `Bpod()` convention every task script uses) and skips all PyBpod project/
task scaffolding entirely.

PyQt5 is already a dependency of this whole stack (pulled in by pyforms-gui), so no new dependency
was needed -- imported directly, same convention `_shared/dot_display.py`/`gabor_display.py` use
(not AnyQt, which is pyforms_generic_editor's own internal choice).

Firing the requested pulses runs on a background QThread (Bpod calls, one real StateMachine per
pulse -- hardware-timed, not a Python time.sleep() loop, same "prefer native Bpod timing" principle
used throughout this codebase) while the main thread keeps the Qt UI responsive; the pulse-related
buttons are disabled for the run's duration so only one thread ever touches the Bpod connection at
a time (per CLAUDE.md's own documented hazard: concurrent access to the same Bpod USB connection
from two threads is genuinely unsynchronized, not just a style choice).
"""
import sys
import traceback

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QComboBox, QDoubleSpinBox,
    QSpinBox, QPushButton, QLabel, QTableWidget, QTableWidgetItem, QGroupBox, QMessageBox)
from PyQt5.QtCore import QThread, pyqtSignal

from liquid_calibration import LiquidCalibration

from pybpodapi.protocol import Bpod, StateMachine

VAR_VALVE_IDS = [1]   # this rig's single valve (see CLAUDE.md: "wheel-turn choice + single valve")
                       # -- extend this list for a future second valve, nothing else needs to change.
VAR_DEFAULT_DURATION_MS = 50.0
VAR_DEFAULT_N_PULSES = 200   # within MATLAB's own suggested 100-500 range


class PulseRunner(QThread):
    """ Fires n_pulses at duration_ms on valve_id, each pulse a real one-state Bpod StateMachine
    (hardware-timed valve-open duration), sequentially, on this background thread. Emits
    finished_ok when all pulses complete, or failed(message) if anything raises -- either way the
    main thread re-enables the UI from a slot connected to one of these signals, never by polling
    from the pulsing thread itself. """
    finished_ok = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(self, my_bpod, valve_id, duration_ms, n_pulses):
        super(PulseRunner, self).__init__()
        self.my_bpod = my_bpod
        self.valve_id = valve_id
        self.duration_ms = duration_ms
        self.n_pulses = n_pulses

    def run(self):
        try:
            duration_s = self.duration_ms / 1000.0
            for _ in range(self.n_pulses):
                sma = StateMachine(self.my_bpod)
                sma.add_state(
                    state_name='Pulse',
                    state_timer=duration_s,
                    state_change_conditions={Bpod.Events.Tup: 'exit'},
                    output_actions=[(Bpod.OutputChannels.Valve, self.valve_id)])
                self.my_bpod.send_state_machine(sma)
                self.my_bpod.run_state_machine(sma)
            self.finished_ok.emit()
        except Exception as err:
            print(traceback.format_exc(), flush=True)
            self.failed.emit("{0}\n\n{1}".format(err, traceback.format_exc()))


class CalibrationWindow(QWidget):
    def __init__(self):
        super(CalibrationWindow, self).__init__()
        self.setWindowTitle('Liquid Reward Calibration')

        self.calibration = LiquidCalibration()

        self.my_bpod = Bpod()
        print("Connected to Bpod on {0}".format(self.my_bpod.serial_port), flush=True)

        self.runner = None   # holds the current PulseRunner while a pulse batch is in flight

        self._build_ui()
        self._refresh_table()
        self._update_lookup_preview()

    def _build_ui(self):
        layout = QVBoxLayout()

        valve_row = QHBoxLayout()
        valve_row.addWidget(QLabel('Valve:'))
        self.valve_combo = QComboBox()
        for valve_id in VAR_VALVE_IDS:
            self.valve_combo.addItem('Valve {0}'.format(valve_id), valve_id)
        self.valve_combo.currentIndexChanged.connect(self._on_valve_changed)
        valve_row.addWidget(self.valve_combo)
        valve_row.addStretch(1)
        layout.addLayout(valve_row)

        pulse_group = QGroupBox('1. Run pulses')
        pulse_form = QFormLayout()
        self.duration_input = QDoubleSpinBox()
        self.duration_input.setRange(1.0, 2000.0)
        self.duration_input.setSuffix(' ms')
        self.duration_input.setValue(VAR_DEFAULT_DURATION_MS)
        pulse_form.addRow('Pulse duration:', self.duration_input)

        self.n_pulses_input = QSpinBox()
        self.n_pulses_input.setRange(1, 2000)
        self.n_pulses_input.setValue(VAR_DEFAULT_N_PULSES)
        pulse_form.addRow('Pulse count:', self.n_pulses_input)

        self.run_button = QPushButton('Run Pulses')
        self.run_button.clicked.connect(self._on_run_pulses)
        pulse_form.addRow(self.run_button)

        self.status_label = QLabel('Ready.')
        pulse_form.addRow(self.status_label)
        pulse_group.setLayout(pulse_form)
        layout.addWidget(pulse_group)

        measure_group = QGroupBox('2. Enter weighed liquid, add point')
        measure_form = QFormLayout()
        self.mass_input = QDoubleSpinBox()
        self.mass_input.setDecimals(4)
        self.mass_input.setRange(0.0, 1000.0)
        self.mass_input.setSuffix(' g')
        measure_form.addRow('Weighed mass (total, all pulses):', self.mass_input)

        self.add_point_button = QPushButton('Add Point')
        self.add_point_button.clicked.connect(self._on_add_point)
        measure_form.addRow(self.add_point_button)
        measure_group.setLayout(measure_form)
        layout.addWidget(measure_group)

        self.points_table = QTableWidget(0, 2)
        self.points_table.setHorizontalHeaderLabels(['Duration (ms)', 'Volume (uL/pulse)'])
        layout.addWidget(self.points_table)

        fit_group = QGroupBox('3. Fit and look up')
        fit_form = QFormLayout()
        self.fit_button = QPushButton('Fit This Valve')
        self.fit_button.clicked.connect(self._on_fit)
        fit_form.addRow(self.fit_button)

        self.coeffs_label = QLabel('(not fit yet)')
        fit_form.addRow('Fit coefficients:', self.coeffs_label)

        self.lookup_input = QDoubleSpinBox()
        self.lookup_input.setDecimals(3)
        self.lookup_input.setRange(0.0, 1000.0)
        self.lookup_input.setSuffix(' uL')
        self.lookup_input.setValue(4.0)
        self.lookup_input.valueChanged.connect(self._update_lookup_preview)
        fit_form.addRow('Preview volume:', self.lookup_input)

        self.lookup_result_label = QLabel('(no fit yet)')
        fit_form.addRow('-> valve open time:', self.lookup_result_label)
        fit_group.setLayout(fit_form)
        layout.addWidget(fit_group)

        self.setLayout(layout)

    def _current_valve_id(self):
        return self.valve_combo.currentData()

    def _on_valve_changed(self):
        self._refresh_table()
        self._update_lookup_preview()

    def _on_run_pulses(self):
        valve_id = self._current_valve_id()
        duration_ms = self.duration_input.value()
        n_pulses = self.n_pulses_input.value()

        self.run_button.setEnabled(False)
        self.add_point_button.setEnabled(False)
        self.status_label.setText('Running {0} pulses at {1:.1f}ms...'.format(n_pulses, duration_ms))

        self.runner = PulseRunner(self.my_bpod, valve_id, duration_ms, n_pulses)
        self.runner.finished_ok.connect(self._on_pulses_done)
        self.runner.failed.connect(self._on_pulses_failed)
        self.runner.start()

    def _on_pulses_done(self):
        self.run_button.setEnabled(True)
        self.add_point_button.setEnabled(True)
        self.status_label.setText('Done -- weigh the dispensed liquid, enter the mass, click '
                                   'Add Point.')

    def _on_pulses_failed(self, message):
        self.run_button.setEnabled(True)
        self.add_point_button.setEnabled(True)
        self.status_label.setText('FAILED: {0}'.format(message))
        QMessageBox.critical(self, 'Pulse run failed', message)

    def _on_add_point(self):
        valve_id = self._current_valve_id()
        duration_ms = self.duration_input.value()
        n_pulses = self.n_pulses_input.value()
        mass_g = self.mass_input.value()

        if mass_g <= 0:
            QMessageBox.warning(self, 'Invalid mass', 'Enter the weighed mass in grams first.')
            return

        volume_ul = self.calibration.add_measurement(valve_id, duration_ms, mass_g, n_pulses)
        self.status_label.setText('Added point: {0:.1f}ms -> {1:.4f} uL/pulse.'.format(
            duration_ms, volume_ul))
        self.mass_input.setValue(0.0)
        self._refresh_table()

    def _on_fit(self):
        valve_id = self._current_valve_id()
        try:
            coeffs = self.calibration.fit(valve_id)
            self.coeffs_label.setText('[{0:.6g}, {1:.6g}, {2:.6g}]'.format(*coeffs))
        except ValueError as err:
            QMessageBox.warning(self, 'Cannot fit', str(err))
            return
        self._update_lookup_preview()

    def _update_lookup_preview(self):
        valve_id = self._current_valve_id()
        volume_ul = self.lookup_input.value()
        try:
            duration_s = self.calibration.get_valve_time_s(volume_ul, valve_id)
            self.lookup_result_label.setText('{0:.1f} ms ({1:.4f} s)'.format(
                duration_s * 1000.0, duration_s))
        except ValueError:
            self.lookup_result_label.setText('(no fit yet)')

    def _refresh_table(self):
        valve_id = self._current_valve_id()
        table = self.calibration.table(valve_id)
        self.points_table.setRowCount(len(table))
        for row, (duration_ms, volume_ul) in enumerate(table):
            self.points_table.setItem(row, 0, QTableWidgetItem('{0:.1f}'.format(duration_ms)))
            self.points_table.setItem(row, 1, QTableWidgetItem('{0:.4f}'.format(volume_ul)))

        coeffs = self.calibration.coeffs(valve_id)
        if coeffs:
            self.coeffs_label.setText('[{0:.6g}, {1:.6g}, {2:.6g}]'.format(*coeffs))
        else:
            self.coeffs_label.setText('(not fit yet)')

    def closeEvent(self, event):
        self.my_bpod.close()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CalibrationWindow()
    window.resize(480, 560)
    window.show()
    sys.exit(app.exec_())
