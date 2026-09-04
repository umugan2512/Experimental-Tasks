# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Camera-recording bench test: exercises _shared/camera_recorder.py's CameraRecorder alongside a
minimal real Bpod session (rotary hold-to-init -> reward loop, no visual stimulus) -- demonstrates
recording, live preview, and Bpod-clock time-sync end to end. No dot/Gabor display here; the point
of this bench test is exercising the camera, not wheel/visual-stimulus logic. See
docs_local/CAMERA_TEST_PLAN.md for the staged hardware test procedure.

Trial structure is deliberately the simplest "real" Bpod session in this codebase: hold the wheel
steady (TrialRunner.wait_for_held_steady(), same as every other wheel-coupled task) then a fixed
reward -- no thresholds, no choice, no timeout branching. Camera recording runs continuously on its
own background thread for the WHOLE SESSION (started once, before the trial loop, not per-trial),
independent of the trial loop's own pacing -- see camera_recorder.py's own docstring for why
(nothing needs to be done with the video during the task).

**Kill handling deviates from this project's simpler bench tests (e.g. lick_reward.py/
wheel_turn_reward.py), which don't wrap their own plain per-trial state-machine calls in a
try/except at all** -- here it matters more: run_state_machine()'s Kill handling calls exit(0)
(raising SystemExit) from inside the function on a kill command, and an unfinalized
cv2.VideoWriter (no close()/release() call) can leave a genuinely corrupt, unplayable video file,
unlike a merely truncated-but-still-parseable Bpod session CSV. So this script explicitly catches
SystemExit around the state-machine call to guarantee camera.close() still runs.

Run this like any other PyBpod task, via the GUI's Run button -- requires the Bpod board, rotary
encoder module, and the USB camera all connected first. Falls back across camera indices 0-4 via
discover_camera() if VAR_CAMERA_INDEX is left at its default (None).
"""
import os
import random
import sys
import time

_TASK_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_TASK_DIR, '..', '..', '..', '_shared'))
from bpod_trial_helpers import TrialRunner
import rotary_setup
from camera_recorder import CameraRecorder

from pybpodapi.protocol import Bpod, StateMachine

VAR_N_TRIALS = 20
VAR_HOLD_MIN_S = 0.1
VAR_HOLD_MAX_S = 0.5
VAR_STEADY_THRESHOLD_DEG = 5
VAR_REWARD_DURATION = 0.1
VAR_ITI = 2
VAR_STILL_POLL_HZ = 50
VAR_POLL_HZ = 10
VAR_ROTARY_USB_PORT = None

VAR_CAMERA_INDEX = None                      # None = auto-discover, see
                                              # camera_recorder.discover_camera()
VAR_CAMERA_OUTPUT_PATH = 'session_video.avi' # relative to cwd -- lands in the real session
                                              # folder when run for real via the GUI's Run button
                                              # (board_com.py's run_task() sets cwd to the
                                              # session path; confirmed unchanged this session).
VAR_CAMERA_FPS = 30.0
VAR_CAMERA_PREVIEW = True

# --- connect to Bpod, resolve modules -------------------------------------------------------------

my_bpod = Bpod()
print("Connected to Bpod on {0}".format(my_bpod.serial_port), flush=True)

rotary, rotary_bpod_module = rotary_setup.connect_rotary(my_bpod, usb_port=VAR_ROTARY_USB_PORT)

log_python_t0 = time.time()
runner = TrialRunner(my_bpod, rotary, log_python_t0, still_poll_hz=VAR_STILL_POLL_HZ,
                      poll_hz=VAR_POLL_HZ)

camera = CameraRecorder(log_python_t0, camera_index=VAR_CAMERA_INDEX,
                         output_path=VAR_CAMERA_OUTPUT_PATH, fps=VAR_CAMERA_FPS,
                         preview=VAR_CAMERA_PREVIEW)
camera.show()
camera.start()

my_bpod.register_value('CAMERA_START_TIME', camera.start_time_s)
my_bpod.register_value('CAMERA_OUTPUT_PATH', camera.output_path)
my_bpod.register_value('CAMERA_FPS', camera.fps)

# --- trial loop -----------------------------------------------------------------------------------

print("Starting {0} trials -- hold the wheel steady to trigger a reward".format(VAR_N_TRIALS),
      flush=True)

for trial in range(VAR_N_TRIALS):
    required_hold = random.uniform(VAR_HOLD_MIN_S, VAR_HOLD_MAX_S)

    n_breaks = runner.wait_for_held_steady(required_hold, VAR_STEADY_THRESHOLD_DEG,
                                            require_no_lick=False)
    if n_breaks is None:
        print("Bpod stopped running trials (Stop/Kill) -- ending session early after trial "
              "{0}/{1}.".format(trial + 1, VAR_N_TRIALS), flush=True)
        break
    trial_start_t = time.time() - log_python_t0
    runner.register('TRIAL_START', trial_start_t)

    camera.pump()

    sma = StateMachine(my_bpod)

    sma.add_state(
        state_name='Reward',
        state_timer=VAR_REWARD_DURATION,
        state_change_conditions={Bpod.Events.Tup: 'ITI'},
        output_actions=[(Bpod.OutputChannels.Valve, 1)])

    sma.add_state(
        state_name='ITI',
        state_timer=VAR_ITI,
        state_change_conditions={Bpod.Events.Tup: 'exit'},
        output_actions=[])

    try:
        ran = runner.run_trial_state_machine(sma)
    except SystemExit:
        # See module docstring's "Kill handling" note -- guarantee the video file gets finalized.
        print("Bpod Kill received -- ending session.", flush=True)
        camera.close()
        rotary.close()
        my_bpod.close()
        sys.exit(0)

    if not ran:
        print("Bpod stopped running trials (Stop/Kill) -- ending session early after trial "
              "{0}/{1}.".format(trial + 1, VAR_N_TRIALS), flush=True)
        break

    camera.pump()

    print("Trial {0}/{1}: held steady ({2:.2f}s, broke {3}x) -- reward delivered".format(
        trial + 1, VAR_N_TRIALS, required_hold, n_breaks), flush=True)

else:
    print("Done: {0} trials completed".format(VAR_N_TRIALS), flush=True)

camera.close()
rotary.close()
my_bpod.close()
