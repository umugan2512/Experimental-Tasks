"""
Resolves this box's Bpod state-machine COM port for the standalone Calibration/ tools
(calibrate_liquid.py, flush_valve.py, rig_check.py).

Those tools skip the PyBpod GUI's per-session `user_settings.py`, the mechanism that normally
supplies `PYBPOD_SERIAL_PORT`, so they need an explicit port. A hardcoded per-box constant
(`'COM10'`) shipped on `main` and broke every other box ("COM10 not found"). The port is instead
read from this box's own GUI board config(s), `_projects/*/boards/*/*.json` ("serial-port"),
which are gitignored and therefore always describe the box the tool is running on.

Resolution order:
1. `BPOD_SERIAL_PORT` environment variable, if set (explicit override).
2. The board JSONs' "serial-port" values, keeping only ports that are actually present right
   now. Used if exactly one distinct port remains.
3. Otherwise raise `RuntimeError`, listing what was found, rather than guessing.
"""
import glob
import json
import os

_REPO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')


def _present_ports():
    try:
        from serial.tools import list_ports
        return {p.device.upper() for p in list_ports.comports()}
    except Exception:
        return None   # can't enumerate -- don't filter


def _board_ports():
    ports = {}
    for path in glob.glob(os.path.join(_REPO_DIR, '_projects', '*', 'boards', '*', '*.json')):
        try:
            with open(path, 'r') as f:
                port = json.load(f).get('serial-port')
        except (OSError, ValueError):
            continue
        if port:
            ports.setdefault(port.upper(), []).append(os.path.relpath(path, _REPO_DIR))
    return ports


def resolve_bpod_serial_port():
    override = os.environ.get('BPOD_SERIAL_PORT')
    if override:
        return override

    board_ports = _board_ports()
    present = _present_ports()
    candidates = sorted(p for p in board_ports if present is None or p in present)
    if len(candidates) == 1:
        return candidates[0]

    raise RuntimeError(
        "Could not determine the Bpod serial port.\n"
        "  Ports in board configs: {0}\n"
        "  Ports currently present: {1}\n"
        "Set the BPOD_SERIAL_PORT environment variable (e.g. `set BPOD_SERIAL_PORT=COM20`) "
        "or fix the board's serial port in the PyBpod GUI.".format(
            {p: srcs for p, srcs in board_ports.items()} or 'none',
            sorted(present) if present is not None else 'unknown'))
