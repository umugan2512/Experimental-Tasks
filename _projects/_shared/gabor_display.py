# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Minimal Gabor-patch-on-a-second-monitor display, coupled to wheel position. Lives at
_projects/_shared/ (see bpod_trial_helpers.py's module docstring for why) since a wheel-coupled
visual stimulus isn't specific to any one protocol -- this was built as a standalone bench test
(gabor_wheel_test/) specifically so it can later be embedded into the Poisson-clicks protocol's own
decision period.

No psychopy/pygame/pyglet in this project's conda env -- built on PyQt5 instead (already a hard
dependency via pyforms-gui), a plain QWidget + QPainter blitting a precomputed grayscale Gabor
image, repositioned per frame. Checked ibllib/brainbox (both installed here, pulled in by
pybpod-gui-plugin-alyx) for reusable code first -- neither has any Gabor-rendering/stimulus-
presentation code (that lives in IBL's separate iblrig repo, not installed here); ibllib's own
training data extractors do confirm the coupling convention this module follows: wheel position is
coupled to the *stimulus's on-screen position* (a signed offset from center), not an internal
drift phase -- turning the wheel drags an otherwise-static patch across the screen. VAR_DEG_TO_PX_GAIN
below is a deliberately uncalibrated placeholder, not a real degrees-of-visual-angle conversion --
that depends on this rig's actual screen size/viewing distance, not established yet.

This module is driven by an external polling loop (see gabor_wheel_test.py), not app.exec_() --
QApplication.exec_() would block whichever thread calls it and take over the whole script, which
doesn't fit a task script that also needs to run its own trial loop. Call pump() repeatedly from
that external loop instead to actually process Qt's pending events/repaints.
"""
import sys

import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtGui import QImage, QPainter, QColor
from PyQt5.QtCore import Qt


def make_gabor_image(size_px, spatial_freq_cpp, sigma_px, contrast=1.0):
    """
    A cosine grating masked by a Gaussian envelope, normalized to 0-255 grayscale.

    :param int size_px: patch width/height in pixels (square patch)
    :param float spatial_freq_cpp: grating spatial frequency, in cycles per patch width
    :param float sigma_px: Gaussian envelope standard deviation, in pixels
    :param float contrast: 0-1, grating amplitude before the envelope is applied
    :return: size_px x size_px uint8 numpy array
    """
    x = np.linspace(-size_px / 2.0, size_px / 2.0, size_px)
    xx, yy = np.meshgrid(x, x)
    grating = np.cos(2 * np.pi * spatial_freq_cpp * xx / size_px)
    envelope = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma_px ** 2))
    patch = 0.5 + 0.5 * contrast * grating * envelope
    return (np.clip(patch, 0.0, 1.0) * 255).astype(np.uint8)


class _GaborWidget(QWidget):
    def __init__(self, image_array, background_gray):
        super(_GaborWidget, self).__init__()
        self._background_gray = background_gray
        self._x_offset_px = 0
        self._visible_patch = False

        h, w = image_array.shape
        # QImage keeps a reference to the underlying buffer; np.ascontiguousarray keeps that
        # buffer alive and contiguous for as long as this QImage is (Qt's own docs warn against
        # passing a non-contiguous/transient array directly).
        self._image_data = np.ascontiguousarray(image_array)
        self._qimage = QImage(self._image_data.data, w, h, w, QImage.Format_Grayscale8)

    def set_x_offset(self, x_offset_px):
        self._x_offset_px = x_offset_px
        self.update()

    def set_patch_visible(self, visible):
        self._visible_patch = visible
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        gray = self._background_gray
        painter.fillRect(self.rect(), QColor(gray, gray, gray))
        if self._visible_patch:
            cx = self.width() // 2 + int(round(self._x_offset_px))
            cy = self.height() // 2
            painter.drawImage(cx - self._qimage.width() // 2, cy - self._qimage.height() // 2,
                               self._qimage)


class GaborDisplay(object):
    """
    Owns a QApplication (reuses QApplication.instance() if one already exists, else creates one)
    and a frameless window positioned on a specific screen. Meant to be constructed once per
    session, driven by an external render loop calling set_position_deg()/pump() repeatedly during
    a trial's decision period.
    """

    def __init__(self, screen_index=1, size_px=400, spatial_freq_cpp=6, sigma_px=80,
                 contrast=1.0, background_gray=128, deg_to_px_gain=4.0):
        self._deg_to_px_gain = deg_to_px_gain

        self._app = QApplication.instance()
        if self._app is None:
            self._app = QApplication(sys.argv)

        screens = self._app.screens()
        if screen_index >= len(screens):
            print("WARNING: GaborDisplay requested screen index {0} but only {1} screen(s) "
                  "detected -- falling back to screen 0.".format(screen_index, len(screens)),
                  flush=True)
            screen_index = 0
        screen_geometry = screens[screen_index].geometry()

        image = make_gabor_image(size_px, spatial_freq_cpp, sigma_px, contrast)
        self._widget = _GaborWidget(image, background_gray)
        self._widget.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self._widget.setGeometry(screen_geometry)

    def show(self):
        self._widget.show()

    def close(self):
        self._widget.close()

    def clear(self):
        """ Blank/neutral screen -- call between trials, before a decision period starts. """
        self._widget.set_patch_visible(False)

    def set_position_deg(self, wheel_position_deg):
        """ Repositions the (already-visible) patch from the current wheel position, using
        deg_to_px_gain (an uncalibrated placeholder -- see module docstring) to convert wheel
        degrees to an on-screen pixel offset from center. """
        self._widget.set_patch_visible(True)
        self._widget.set_x_offset(wheel_position_deg * self._deg_to_px_gain)

    def pump(self):
        """ Processes pending Qt events/repaints -- call repeatedly from an external polling loop
        instead of app.exec_() (which would block the calling thread). """
        self._app.processEvents()
