# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Minimal filled-dot-on-a-second-monitor display, coupled to wheel position. Lives at
_projects/_shared/ (see bpod_trial_helpers.py's module docstring for why) since a wheel-coupled
visual stimulus isn't specific to any one protocol -- same placement rationale as gabor_display.py.

Forked from gabor_display.py's own architecture (QWidget + QPainter, screen selection with
fallback, show()/clear()/set_position_deg()/pump() driven by an external polling loop rather than
app.exec_()) but for training_protocol.md's dot stimulus instead of a Gabor patch: "one parameter,
no internal structure, no phase or orientation" (SS1.2) -- a plain filled circle via
QPainter.drawEllipse(), no numpy grating/envelope image buffer needed at all. Rigid wheel->azimuth
coupling starts from the same fixed-multiplier convention gabor_display.py's own
VAR_DEG_TO_PX_GAIN uses, but adds get_screen_width_px()/set_deg_to_px_gain() so a caller can instead
derive a geometry-aware gain (e.g. "reach 90% of the half-screen-width at the wheel threshold," per
SS1.3's placement principle) from the screen actually resolved at runtime -- real
degrees-of-visual-angle calibration (monitor size/viewing distance) still isn't established, but
this at least scales with the actual screen instead of an arbitrary guessed px/deg constant.

This module only implements the stimulus + rigid coupling itself -- the contrast-onset ramp
(SS1.2: "contrast ramped over ~50-100ms, not stepped") and everything else in
training_protocol.md (staircases, debiasing, the deliberateness gate, etc.) are out of scope here,
per current instruction; those are separate, later work.

Driven by an external polling loop (see dot_wheel_test.py), not app.exec_() -- same reasoning as
gabor_display.py: QApplication.exec_() would block whichever thread calls it, which doesn't fit a
task script that also needs to run its own trial loop. Call pump() repeatedly from that external
loop instead to actually process Qt's pending events/repaints.
"""
import math
import sys

from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtGui import QPainter, QColor
from PyQt5.QtCore import Qt


def visual_deg_to_px(visual_deg, screen_width_px, screen_width_mm, viewing_distance_mm):
    """
    Converts a target size in VISUAL degrees (e.g. training_protocol.md SS1.2's 3-4deg dot
    diameter) to pixels, given the screen's actual physical width and the animal's viewing
    distance -- both still unmeasured on this rig (SS Part 6, item 1: "Measure the monitor's
    subtended azimuth"). NOT used anywhere yet -- VAR_DOT_DIAMETER_PX in dot_wheel_test.py is
    still a guessed pixel value, unconfirmed against the 3-4deg spec, until those two
    measurements exist. Once they do, this is a one-line fix:

        diameter_px = visual_deg_to_px(3.5, screen_width_px, screen_width_mm, viewing_distance_mm)

    :param float visual_deg: target size in visual degrees
    :param int screen_width_px: screen width in pixels (DotDisplay.get_screen_width_px())
    :param float screen_width_mm: screen's physical width in mm
    :param float viewing_distance_mm: distance from the animal's eye to the screen, in mm
    """
    total_visual_deg_width = 2.0 * math.degrees(math.atan((screen_width_mm / 2.0) / viewing_distance_mm))
    px_per_deg = screen_width_px / total_visual_deg_width
    return visual_deg * px_per_deg


class _DotWidget(QWidget):
    def __init__(self, diameter_px, background_gray, dot_gray):
        super(_DotWidget, self).__init__()
        self._diameter_px = diameter_px
        self._background_gray = background_gray
        self._dot_gray = dot_gray
        self._x_offset_px = 0
        self._visible_dot = False

    def set_x_offset(self, x_offset_px):
        self._x_offset_px = x_offset_px
        self.update()

    def set_dot_visible(self, visible):
        self._visible_dot = visible
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        bg = self._background_gray
        painter.fillRect(self.rect(), QColor(bg, bg, bg))
        if self._visible_dot:
            cx = self.width() // 2 + int(round(self._x_offset_px))
            cy = self.height() // 2
            r = self._diameter_px // 2
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(self._dot_gray, self._dot_gray, self._dot_gray))
            painter.drawEllipse(cx - r, cy - r, self._diameter_px, self._diameter_px)


class DotDisplay(object):
    """
    Owns a QApplication (reuses QApplication.instance() if one already exists, else creates one)
    and a frameless window positioned on a specific screen. Meant to be constructed once per
    session, driven by an external render loop calling set_position_deg()/pump() repeatedly during
    a trial's decision period -- same usage pattern as gabor_display.py's GaborDisplay.
    """

    def __init__(self, screen_index=1, diameter_px=60, background_gray=128, dot_gray=0,
                 deg_to_px_gain=4.0):
        self._deg_to_px_gain = deg_to_px_gain

        self._app = QApplication.instance()
        if self._app is None:
            self._app = QApplication(sys.argv)

        screens = self._app.screens()
        if screen_index >= len(screens):
            print("WARNING: DotDisplay requested screen index {0} but only {1} screen(s) "
                  "detected -- falling back to screen 0.".format(screen_index, len(screens)),
                  flush=True)
            screen_index = 0
        screen_geometry = screens[screen_index].geometry()

        self._widget = _DotWidget(diameter_px, background_gray, dot_gray)
        self._widget.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self._widget.setGeometry(screen_geometry)

    def show(self):
        self._widget.show()

    def close(self):
        self._widget.close()

    def get_screen_width_px(self):
        """ The actual resolved screen's width in px -- lets a caller compute a geometry-aware
        deg_to_px_gain (e.g. so a given wheel threshold reaches some target fraction of the
        half-screen-width) instead of guessing a fixed gain, then apply it via
        set_deg_to_px_gain() before the render loop starts. """
        return self._widget.geometry().width()

    def set_deg_to_px_gain(self, deg_to_px_gain):
        self._deg_to_px_gain = deg_to_px_gain

    def clear(self):
        """ Blank/neutral screen -- call between trials, before a decision period starts. """
        self._widget.set_dot_visible(False)

    def set_position_deg(self, wheel_position_deg):
        """ Repositions the (already-visible) dot from the current wheel position, using
        deg_to_px_gain (an uncalibrated placeholder -- see module docstring) to convert wheel
        degrees to an on-screen pixel offset from center. """
        self._widget.set_dot_visible(True)
        self._widget.set_x_offset(wheel_position_deg * self._deg_to_px_gain)

    def pump(self):
        """ Processes pending Qt events/repaints -- call repeatedly from an external polling loop
        instead of app.exec_() (which would block the calling thread). """
        self._app.processEvents()


class _MiddleOnlyDotWidget(QWidget):
    """
    Same rendering idea as _DotWidget, but for a rig where several physical monitors are bonded by
    the GPU into a single Qt/Windows screen -- confirmed on this rig via a direct screens() query:
    three physical panels report as ONE wide QScreen (6144x1536, i.e. three 2048px-wide panels),
    not three separate QScreen entries, so there is no separate screen to point a second window at
    for the other two panels. Instead, this widget still spans the whole combined window, but only
    ever draws (background + dot) inside one active_segment_index-th of n_segments equal-width
    columns; everything outside that column is always painted solid black.
    """
    def __init__(self, diameter_px, background_gray, dot_gray, n_segments, active_segment_index):
        super(_MiddleOnlyDotWidget, self).__init__()
        self._diameter_px = diameter_px
        self._background_gray = background_gray
        self._dot_gray = dot_gray
        self._n_segments = n_segments
        self._active_segment_index = active_segment_index
        self._x_offset_px = 0
        self._visible_dot = False

    def set_x_offset(self, x_offset_px):
        self._x_offset_px = x_offset_px
        self.update()

    def set_dot_visible(self, visible):
        self._visible_dot = visible
        self.update()

    def _active_rect(self):
        """ (x0, width) in widget-local px of the active_segment_index-th of n_segments equal
        columns -- the last column absorbs any leftover px from integer division. """
        seg_w = self.width() // self._n_segments
        x0 = seg_w * self._active_segment_index
        if self._active_segment_index == self._n_segments - 1:
            return x0, self.width() - x0
        return x0, seg_w

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 0, 0))   # whole spanned window black by default
        x0, w = self._active_rect()
        bg = self._background_gray
        painter.fillRect(x0, 0, w, self.height(), QColor(bg, bg, bg))
        if self._visible_dot:
            cx = x0 + w // 2 + int(round(self._x_offset_px))
            cy = self.height() // 2
            r = self._diameter_px // 2
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(self._dot_gray, self._dot_gray, self._dot_gray))
            painter.drawEllipse(cx - r, cy - r, self._diameter_px, self._diameter_px)


class MiddleScreenDotDisplay(object):
    """
    Purely additive alternative to DotDisplay -- does not read, call, or modify DotDisplay/
    _DotWidget in any way. For a rig where n_segments physical monitors are bonded into a single
    Qt screen_index (see _MiddleOnlyDotWidget's own docstring for how that was confirmed on this
    rig), this keeps the dot's rendering AND its full range of motion confined to one
    active_segment_index-th column of that combined window, with the rest of the window always
    solid black -- so the two other physical monitors sharing that same combined screen appear
    blank even though they're not separate windows.

    Same constructor shape and exact same public method surface as DotDisplay (show()/close()/
    get_screen_width_px()/set_deg_to_px_gain()/clear()/set_position_deg()/pump()) -- a drop-in
    replacement for DotDisplay in a task script. get_screen_width_px() here deliberately returns
    the ACTIVE COLUMN's width, not the full spanned window's width, so a caller's existing
    geometry-aware gain calibration (e.g. dot_wheel_test.py's own VAR_DOT_EDGE_FRACTION-based gain)
    automatically keeps the dot's full range of motion within one physical monitor with no other
    code changes needed.
    """

    def __init__(self, screen_index=1, n_segments=3, active_segment_index=1, diameter_px=60,
                 background_gray=128, dot_gray=0, deg_to_px_gain=4.0):
        self._deg_to_px_gain = deg_to_px_gain

        self._app = QApplication.instance()
        if self._app is None:
            self._app = QApplication(sys.argv)

        screens = self._app.screens()
        if screen_index >= len(screens):
            print("WARNING: MiddleScreenDotDisplay requested screen index {0} but only {1} "
                  "screen(s) detected -- falling back to screen 0.".format(
                      screen_index, len(screens)), flush=True)
            screen_index = 0
        screen_geometry = screens[screen_index].geometry()

        self._widget = _MiddleOnlyDotWidget(diameter_px, background_gray, dot_gray,
                                             n_segments, active_segment_index)
        self._widget.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self._widget.setGeometry(screen_geometry)

    def show(self):
        self._widget.show()

    def close(self):
        self._widget.close()

    def get_screen_width_px(self):
        """ The ACTIVE COLUMN's width in px (not the full spanned window's width) -- see class
        docstring for why. """
        _x0, w = self._widget._active_rect()
        return w

    def set_deg_to_px_gain(self, deg_to_px_gain):
        self._deg_to_px_gain = deg_to_px_gain

    def clear(self):
        """ Blank/neutral active column (rest of the window stays black regardless) -- call
        between trials, before a decision period starts. """
        self._widget.set_dot_visible(False)

    def set_position_deg(self, wheel_position_deg):
        """ Repositions the (already-visible) dot from the current wheel position, using
        deg_to_px_gain to convert wheel degrees to a pixel offset from the active column's own
        center. """
        self._widget.set_dot_visible(True)
        self._widget.set_x_offset(wheel_position_deg * self._deg_to_px_gain)

    def pump(self):
        """ Processes pending Qt events/repaints -- call repeatedly from an external polling loop
        instead of app.exec_() (which would block the calling thread). """
        self._app.processEvents()
