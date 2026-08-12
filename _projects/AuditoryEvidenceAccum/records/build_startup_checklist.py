# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Generates session_startup_checklist.xlsx -- a single-page, printable/duplicable form (not a data
table) for physical record-keeping: whoever runs a session fills in animal ID, which protocol was
run, handwritten weight/weight%/reward-consumed, ticks off an equipment-on/off checklist and a set
of pre-session checks, then fills in an outcome summary at the end. Meant to be printed (or
duplicated as a new sheet/copy per session) the same way a physical lab notebook page would be --
purely a blank template, never auto-populated (there's no digital source for any of this; contrast
with build_training_log.py, which reads real session logs).

Run once (or re-run any time you want to reset it back to blank):
    /c/Users/2P-Behav/.conda/envs/pybpod-environment/python.exe build_startup_checklist.py
"""
import os

import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

_RECORDS_DIR = os.path.dirname(os.path.abspath(__file__))
_OUTPUT_PATH = os.path.join(_RECORDS_DIR, 'session_startup_checklist.xlsx')

_TITLE_FONT = Font(bold=True, size=14)
_SECTION_FONT = Font(bold=True, size=11)
_LABEL_FONT = Font(bold=True)
_THIN = Side(style='thin', color='999999')
_BOX_BORDER = Border(left=_THIN, right=_THIN, top=_THIN, bottom=_THIN)
_SECTION_FILL = PatternFill('solid', fgColor='DDEBF7')
CHECKBOX = '☐'   # ☐ -- hand-marked with an X or a checkmark when printed/filled in


def _section(ws, row, title):
    ws.cell(row=row, column=1, value=title).font = _SECTION_FONT
    for col in range(1, 7):
        ws.cell(row=row, column=col).fill = _SECTION_FILL
    return row + 1


def _label_blank(ws, row, label, col_label=1, col_blank=2, blank_width=3):
    ws.cell(row=row, column=col_label, value=label).font = _LABEL_FONT
    for c in range(col_blank, col_blank + blank_width):
        ws.cell(row=row, column=c).border = _BOX_BORDER
    return row + 1


def _checkbox_line(ws, row, text):
    ws.cell(row=row, column=1, value=CHECKBOX)
    ws.cell(row=row, column=2, value=text)
    return row + 1


def build():
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = 'Startup checklist'

    row = 1
    ws.cell(row=row, column=1, value='Session Startup Checklist -- AuditoryEvidenceAccum').font = _TITLE_FONT
    row += 2

    row = _section(ws, row, 'Session identification')
    row = _label_blank(ws, row, 'Date:')
    row = _label_blank(ws, row, 'Animal ID:')
    row = _label_blank(ws, row, 'Experimenter:')
    row = _label_blank(ws, row, 'Protocol run (e.g. "Stage 2 threshold staircase"):', blank_width=4)
    row = _label_blank(ws, row, 'Time started:')
    row = _label_blank(ws, row, 'Time ended:')
    row += 1

    row = _section(ws, row, 'Weight and reward (handwritten)')
    row = _label_blank(ws, row, 'Weight (g):')
    row = _label_blank(ws, row, 'Weight % of baseline:')
    row = _label_blank(ws, row, 'Reward consumed (count or est. volume):')
    row += 1

    row = _section(ws, row, 'Equipment on (before starting)')
    for text in ['Bpod board powered on and connected',
                 'Rotary encoder module connected',
                 'HiFi module connected (if this protocol uses sound)',
                 'Monitor / dot display connected and positioned',
                 'Water line primed and checked for drips',
                 'Spout position confirmed against the training-stage schedule']:
        row = _checkbox_line(ws, row, text)
    row += 1

    row = _section(ws, row, 'Pre-session checks')
    for text in ['Animal health check normal (grooming, posture, behavior)',
                 'Water restriction on schedule (checked against the training-log workbook)',
                 'Wheel spins freely, no obstruction',
                 'Live plot window opened and confirmed showing data']:
        row = _checkbox_line(ws, row, text)
    row += 1

    row = _section(ws, row, 'Equipment off (after finishing)')
    for text in ['Bpod session stopped cleanly (not Killed)',
                 'HiFi module powered off (if used)',
                 'Monitor / dot display powered off',
                 'Water line checked, no residual dripping',
                 'Animal returned to home cage, weighed if required']:
        row = _checkbox_line(ws, row, text)
    row += 1

    row = _section(ws, row, 'Outcome summary (fill in after the session)')
    row = _label_blank(ws, row, 'Trial count:')
    row = _label_blank(ws, row, 'Advance-ready (Y/N, from console output):')
    row = _label_blank(ws, row, 'Any errors / crashes / hardware issues:', blank_width=4)
    row = _label_blank(ws, row, 'Notes:', blank_width=4)
    notes_row = row
    for extra in range(3):
        for c in range(2, 6):
            ws.cell(row=notes_row + extra, column=c).border = _BOX_BORDER
    row = notes_row + 3 + 1

    row = _label_blank(ws, row, 'Signature / initials:')

    ws.column_dimensions['A'].width = 42
    for col in 'BCDEF':
        ws.column_dimensions[col].width = 16
    ws.sheet_view.showGridLines = False
    ws.print_area = 'A1:F{0}'.format(row + 2)

    wb.save(_OUTPUT_PATH)
    print("Wrote {0}".format(_OUTPUT_PATH), flush=True)


if __name__ == '__main__':
    build()
