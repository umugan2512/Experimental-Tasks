# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Valve flush utility -- holds a reward valve open indefinitely (not a timed pulse) so clean water or
ethanol can be run through the tubing for cleaning/priming, independent of calibration.

Uses `Bpod.manual_override(...)` rather than a StateMachine pulse (what `calibrate_liquid.py`'s
`PulseRunner` uses): a StateMachine's `state_timer` always has a fixed duration, but
`manual_override` can set an output and leave it there indefinitely across separate calls -- open on
one call, close on a later one. This is only safe here because this is a standalone script with no
state machine ever running concurrently (see CLAUDE.md's own documented hazard: manual_override/
trigger_output/trigger_softcode share the same physical Bpod USB connection a running
`run_state_machine()` call's own blocking loop reads from, so calling any of them from a second,
concurrent thread/process is genuinely unsynchronized access -- not a concern for this script, which
never has a state machine running at all).

Run directly: `python flush_valve.py` (needs the Bpod board and valve physically connected). Same
"connects directly, skips all PyBpod project/task scaffolding" convention as `calibrate_liquid.py`/
`rig_check.py` -- see those files' own module docstrings for why `Bpod(serial_port=
VAR_BPOD_SERIAL_PORT)` is required here too, not a bare `Bpod()`.

**Small standalone GUI** (per explicit request -- launched from a button in `run_session.py`'s own
dashboard, so a separate "type into a console" interactive loop was a worse fit than a couple of
buttons): one status label plus explicit, SEPARATE "Open Valve" / "Close Valve" buttons -- not one
toggle button (the Enter key does toggle, though: one press opens, the next closes), so there's never any ambiguity about which state a click will produce, and the
layout stays uncrowded for a tool this small. `closeEvent()` force-closes the valve (and the Bpod
connection) if the window is closed while the valve is still open, same guaranteed-close-on-exit
safety net the original terminal version's own `finally:` block provided -- the valve, and the
whole tool's hardware connection, is always returned to its original (closed/disconnected) state
no matter how the window is dismissed.
"""
import sys

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QShortcut)

from bpod_port import resolve_bpod_serial_port
from pybpodapi.protocol import Bpod

VAR_VALVE_ID = 1   # this rig's single valve (see CLAUDE.md: "wheel-turn choice + single valve")
VAR_BPOD_SERIAL_PORT = None   # None = auto-detect from this box's own board config (bpod_port.py);
                              # set e.g. 'COM20' only to force a specific port.


class FlushValveWindow(QWidget):
    def __init__(self):
        super(FlushValveWindow, self).__init__()
        self.setWindowTitle('Flush Valve')

        self.my_bpod = None
        self._valve_open = False

        self._connect_hardware()
        self._build_ui()

    def _connect_hardware(self):
        try:
            self.my_bpod = Bpod(serial_port=VAR_BPOD_SERIAL_PORT or resolve_bpod_serial_port())
            self.bpod_status = 'Connected on {0}'.format(self.my_bpod.serial_port)
            print("Connected to Bpod on {0}".format(self.my_bpod.serial_port), flush=True)
        except Exception as err:
            self.my_bpod = None
            self.bpod_status = 'FAILED: {0}'.format(err)

    def _build_ui(self):
        layout = QVBoxLayout()

        layout.addWidget(QLabel('Bpod: {0}'.format(self.bpod_status)))

        self.status_label = QLabel('Valve {0}: CLOSED'.format(VAR_VALVE_ID))
        self.status_label.setStyleSheet('font-weight: bold; padding: 6px;')
        layout.addWidget(self.status_label)

        buttons_row = QHBoxLayout()
        self.open_button = QPushButton('Open Valve')
        self.open_button.clicked.connect(self._on_open)
        buttons_row.addWidget(self.open_button)
        self.close_button = QPushButton('Close Valve')
        self.close_button.clicked.connect(self._on_close)
        self.close_button.setEnabled(False)
        buttons_row.addWidget(self.close_button)
        layout.addLayout(buttons_row)

        # Enter toggles: one press opens, the next closes. Both the main Return key and the
        # keypad Enter key are bound. The buttons take no keyboard focus, so Enter/Space can't
        # also trigger whichever button happens to be focused.
        for button in (self.open_button, self.close_button):
            button.setFocusPolicy(Qt.NoFocus)
        for key in (Qt.Key_Return, Qt.Key_Enter):
            QShortcut(QKeySequence(key), self, activated=self._on_toggle)
        layout.addWidget(QLabel('Press Enter to open / close the valve.'))

        self.setLayout(layout)
        self._update_buttons()

    def _update_buttons(self):
        self.open_button.setEnabled(self.my_bpod is not None and not self._valve_open)
        self.close_button.setEnabled(self.my_bpod is not None and self._valve_open)
        self.status_label.setText('Valve {0}: {1}'.format(
            VAR_VALVE_ID, 'OPEN -- dispensing' if self._valve_open else 'CLOSED'))
        self.status_label.setStyleSheet(
            'font-weight: bold; padding: 6px; background-color: {0};'.format(
                '#ffdddd' if self._valve_open else '#ddffdd'))

    def _set_valve(self, is_open):
        self.my_bpod.manual_override(Bpod.ChannelTypes.OUTPUT, Bpod.ChannelNames.VALVE,
                                      channel_number=VAR_VALVE_ID, value=1 if is_open else 0)
        self._valve_open = is_open
        self._update_buttons()

    def _on_open(self):
        self._set_valve(True)

    def _on_close(self):
        self._set_valve(False)

    def _on_toggle(self):
        if self.my_bpod is not None:
            self._set_valve(not self._valve_open)

    def closeEvent(self, event):
        """ Guaranteed close-on-exit, regardless of how the window is dismissed -- the valve (and
        the Bpod connection itself) is always returned to its original state, same safety net the
        original terminal version's own `finally:` block provided. """
        if self.my_bpod is not None:
            if self._valve_open:
                self._set_valve(False)
            self.my_bpod.close()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = FlushValveWindow()
    window.resize(320, 140)
    window.show()
    sys.exit(app.exec_())
