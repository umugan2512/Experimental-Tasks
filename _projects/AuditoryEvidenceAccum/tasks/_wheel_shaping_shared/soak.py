"""Soak test: runs a REAL stage script for N trials against a fake Bpod that INTERPRETS the state
machine (timers, injected stdin SoftCodes, rotary threshold events from a scripted mouse, relay
outputs, SoftCode outputs), with a fake rotary firmware stream, and writes a pybpod-format CSV
that records/audit_session.py then audits.

Usage: python soak.py <stage_script> <n_trials> [seed]

Mouse profile (env vars, all optional):
  SOAK_P_CORRECT  P(turn to the rewarded side) -- a mouse that listens to the cue
  SOAK_P_LEFT     P(turn left) -- a side-biased mouse
  SOAK_P_REPEAT   P(repeat its last turn) -- a perseverating mouse
  SOAK_P_NOMOVE   P(no turn in the choice window), default 0.15
  SOAK_P_EARLY    P(turn before the dot), default 0.05
  SOAK_P_CUE_ABORT P(turn during the cue/delay, Stages 3-5 abort), default 0.03
State (stand-in for subjects/<id>/wheel_shaping_state.json):
  SOAK_STATE_JSON  JSON dict merged over the stage's defaults
  SOAK_STATE_FILE  path: loaded at start, written by the stage's save(), so runs chain like days
"""
import csv, datetime, math, os, random, runpy, sys, tempfile, threading, time, types

SCRIPT = os.path.abspath(sys.argv[1]); N_TRIALS = int(sys.argv[2])
random.seed(int(sys.argv[3]) if len(sys.argv) > 3 else 7)
TICKS_PER_DEG = 512.0 / 180.0

# ---- fake rotary firmware ----------------------------------------------------------------------
class FakeSerial(object):
    def __init__(self):
        self.buf = bytearray(); self.lock = threading.Lock(); self.streaming = False
        self.pos_ticks = 0; self.t0 = time.time(); self.last = None
        threading.Thread(target=self._fw, daemon=True).start()
    def ticks(self): return int((time.time() - self.t0) / 1e-6) & 0xFFFFFFFF
    def emit(self, op, a, b):
        with self.lock: self.buf += op + a + b + self.ticks().to_bytes(4, 'little')
    def _fw(self):
        while True:
            time.sleep(0.001)
            if self.streaming and self.pos_ticks != self.last:   # real firmware: on change only
                pb = int(self.pos_ticks).to_bytes(2, 'little', signed=True)
                self.emit(b'P', pb[:1], pb[1:]); self.last = self.pos_ticks
    @property
    def in_waiting(self):
        with self.lock: return len(self.buf)
    def read(self, n):
        with self.lock:
            out = bytes(self.buf[:n]); del self.buf[:n]; return out
    def reset_input_buffer(self):
        with self.lock: self.buf.clear()
    port = 'FAKE'

class FakeArcom(object):
    def __init__(self): self.serial_object = FakeSerial()
    def write_array(self, arr):
        arr = list(arr)
        if arr[:1] == [ord('S')]:
            if arr[1] and not self.serial_object.streaming:
                self.serial_object.t0 = time.time()   # real fw: clock restarts on stream start
            self.serial_object.streaming = bool(arr[1])
        if ord('Z') in arr: self.serial_object.pos_ticks = 0

class FakeRotary(object):
    def __init__(self): self.arcom = FakeArcom()
    def enable_stream(self): self.arcom.write_array([ord('S'), 1])
    def disable_stream(self): self.arcom.write_array([ord('S'), 0])
    thresholds = []
    def set_thresholds(self, t):
        assert not self.arcom.serial_object.streaming, 'ack command while streaming'
        self.thresholds = list(t)   # event RotaryEncoder1_<i+1> fires on crossing t[i]
    def enable_all_thresholds(self): pass
    def enable_evt_transmission(self):
        assert not self.arcom.serial_object.streaming; return True
    def disable_evt_transmission(self): raise AssertionError('evt transmission toggled')
    def current_position(self): raise AssertionError('Q polled while streaming')
    def set_zero_position(self): raise AssertionError('ack-verified zero while streaming')
    def close(self): pass

ROT = FakeRotary(); SER = ROT.arcom.serial_object

# ---- fake pybpodapi with a state-machine interpreter ---------------------------------------------
class Events: Tup = 'Tup'
class OutputChannels: Valve = 'Valve'
ROWS = []
def pc(): return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
def row(*cols): ROWS.append([cols[0], pc()] + list(cols[1:]))

class FakeTrial(object):
    def __init__(self, states, events, start):
        self.states_durations = states; self._events = events; self.trial_start_timestamp = start
    def get_all_timestamps_by_event(self): return {k: list(v) for k, v in self._events.items()}

class StateMachine(object):
    def __init__(self, bpod): self.states = []
    def add_state(self, state_name, state_timer=0, state_change_conditions=None, output_actions=()):
        self.states.append((state_name, float(state_timer), dict(state_change_conditions or {}),
                            list(output_actions)))

HW0 = time.time()
STATS = {'outcomes': {}}

class Bpod(object):
    Events = Events; OutputChannels = OutputChannels
    def __init__(self, *a, **k):
        self.serial_port = 'FAKE'; self.modules = []; self.stdin = None
        self.session = types.SimpleNamespace(current_trial=None, _path=os.path.join(os.getcwd(), os.path.basename(os.getcwd()) + '.csv'))
        BPOD[0] = self; self._uploaded = None
        started = pc()   # the preamble real pybpod writes, so build_training_log can read the CSV
        for k, v in (('PROTOCOL-NAME', os.path.splitext(os.path.basename(SCRIPT))[0]),
                     ('SUBJECT-NAME', "['SOAK', 'x']"), ('SESSION-NAME', 'SOAK'),
                     ('SESSION-STARTED', started)):
            row('INFO', '', '', k, v)
    def register_value(self, k, v, flush=True): row('VAL', '', '', k, v)
    def send_state_machine(self, sma): self._uploaded = sma; time.sleep(0.004)
    def softcode_handler_function(self, code): pass
    def loop_handler(self): pass
    def close(self):
        row('INFO', '', '', 'SESSION-ENDED', pc())
        with open(self.session._path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f, delimiter=';')
            w.writerow(['TYPE', 'PC-TIME', 'BPOD-INITIAL-TIME', 'BPOD-FINAL-TIME', 'MSG', '+INFO'])
            for r in ROWS: w.writerow(r)

    def run_state_machine(self, sma):
        assert self._uploaded is sma, 'RUN without upload'
        row('TRIAL', '', '', 'New trial', '')
        start_hw = time.time() - HW0
        t0 = time.time(); T = lambda: time.time() - t0
        names = [s[0] for s in sma.states]; by = {s[0]: s for s in sma.states}
        visits = {}; events = {}
        cur = names[0]
        while cur != 'exit':
            name, timer, conds, outs = by[cur]
            s_start = T(); visits[name] = [s_start, None]
            for ch, val in outs:
                if str(ch).startswith('Serial'):      # rotary relay: Z, '#', 101
                    SER.pos_ticks = 0; SER.emit(b'E', bytes([1]), bytes([101]))
                elif ch == 'SoftCode':
                    row('SOFTCODE', '', '', str(val), ''); self.softcode_handler_function(val)
            nxt = None
            plan = mouse_plan(name, conds)
            while nxt is None:
                self.loop_handler()
                line = self.stdin.readline() if self.stdin is not None else None
                if line and line.startswith('SoftCode'):
                    ev = PC_NAMES.get(int(line[-1]), 'SoftCode' + line[-1])
                    events.setdefault(ev, []).append(T()); row('EVENT', '%.4f' % T(), '', '0', ev)
                    if ev in conds: nxt = conds[ev]; break
                now = T() - s_start
                ev = plan(now)
                if ev:
                    events.setdefault(ev, []).append(T()); row('EVENT', '%.4f' % T(), '', '1', ev)
                    if ev in conds: nxt = conds[ev]; break
                if now >= timer:
                    events.setdefault('Tup', []).append(T()); nxt = conds.get('Tup', 'exit'); break
                time.sleep(0.0005)
            visits[name][1] = T(); cur = nxt
        end_hw = start_hw + T()
        row('END-TRIAL', '', '', 'The trial ended', '')
        row('INFO', '%.6f' % start_hw, '%.6f' % end_hw, 'TRIAL-BPOD-TIME', '%.6f' % (end_hw - start_hw))
        states = {}
        for n in names:
            if n in visits:
                states[n] = [(visits[n][0], visits[n][1])]
                row('STATE', '%.4f' % visits[n][0], '%.4f' % visits[n][1], n, '%.4f' % (visits[n][1] - visits[n][0]))
            else:
                states[n] = [(float('nan'), float('nan'))]
                row('STATE', 'nan', 'nan', n, 'nan')
        last = [n for n in names if n in visits][-2:] if len(visits) > 1 else list(visits)
        STATS['outcomes'][tuple(last)] = STATS['outcomes'].get(tuple(last), 0) + 1
        self.session.current_trial = FakeTrial(states, events, start_hw)
        return True
BPOD = [None]
PC_NAMES = {}   # filled once bpod_trial_helpers is importable (code -> renamed event name)

def _turn(sign, deg, dur):
    """mouse turns the wheel by deg over dur seconds (changes the streamed position)."""
    def go():
        n = 20
        for k in range(1, n + 1):
            SER.pos_ticks = int(sign * deg * TICKS_PER_DEG * k / n); time.sleep(dur / n)
    threading.Thread(target=go, daemon=True).start()

def _ev_sign(ev):
    """ -1 / +1: the sign of the threshold whose crossing raises rotary event `ev`. """
    try:
        return -1 if ROT.thresholds[int(str(ev).rsplit('_', 1)[1]) - 1] < 0 else 1
    except (IndexError, ValueError):
        return 1

def mouse_plan(state, conds):
    """Returns f(t_in_state) -> event name or None, and moves the wheel accordingly."""
    rot = [e for e in conds if str(e).startswith('RotaryEncoder')]
    if state == 'WaitQuiescence':
        jig = random.uniform(0.0, 0.5)
        def f(t):
            if t < jig and int(t * 50) % 2 == 0: SER.pos_ticks = random.choice((-16, 16))
            elif t < jig: SER.pos_ticks = 0
            return None
        return f
    if state == 'DotOn' and rot and random.random() < float(os.environ.get('SOAK_P_EARLY', '0.05')):
        _turn(random.choice((-1, 1)), 8, 0.01); return lambda t: rot[0] if t > 0.002 else None
    if state in ('WheelPeriod', 'WheelDotPeriod') and rot:
        # [left, right] by the sign of each event's threshold: conds order puts the REWARDED one
        # first in Stages 3-5, and the asymmetric threshold uses different events per side
        lr = sorted(rot[:2], key=_ev_sign)
        if random.random() < float(os.environ.get('SOAK_P_NOMOVE', '0.15')): return lambda t: None
        delay = random.uniform(0.02, 0.8)
        _pl = os.environ.get('SOAK_P_LEFT')   # biased mouse: P(turn toward the negative/left threshold)
        _pr = os.environ.get('SOAK_P_REPEAT') # perseverating mouse: P(same side as its last turn)
        _pc = os.environ.get('SOAK_P_CORRECT') # listening mouse: P(turn to the side that leads to Reward)
        _good = [e for e in rot[:2] if conds.get(e) == 'Reward']
        if _pc and len(_good) == 1:
            ev = _good[0] if random.random() < float(_pc) else [e for e in rot[:2] if e != _good[0]][0]
        elif _pr and LAST_SIDE[0] is not None and random.random() < float(_pr):
            ev = lr[0] if LAST_SIDE[0] < 0 else lr[1]   # same SIDE as last time, whatever the event
        else:
            ev = (lr[0] if random.random() < float(_pl) else lr[1]) if _pl else random.choice(lr)
        sign = -1 if ev == lr[0] else 1
        LAST_SIDE[0] = sign
        fired = [False]
        def f(t):
            if t >= delay and not fired[0]:
                fired[0] = True; _turn(sign, 12, 0.05)
            return ev if fired[0] and t >= delay + 0.05 else None
        return f
    if state in ('CuePeriod', 'DelayPeriod', 'DotOnsetDelay') and rot and random.random() < float(os.environ.get('SOAK_P_CUE_ABORT', '0.03')):
        return lambda t: rot[-1] if t > 0.05 else None
    if state in ('Reward', 'Consumption') :
        return lambda t: 'Port1In' if random.random() < 0.002 else None
    return lambda t: None

proto = types.ModuleType('pybpodapi.protocol'); proto.Bpod = Bpod; proto.StateMachine = StateMachine
sys.modules['pybpodapi'] = types.ModuleType('pybpodapi'); sys.modules['pybpodapi.protocol'] = proto
class _Conf(object):
    PYBPOD_SUBJECTS = ["['SOAK', 'x']"]
    def __iadd__(self, o): return self
    def __getattr__(self, n): return None
cm = types.ModuleType('confapp'); cm.conf = _Conf(); sys.modules['confapp'] = cm

task_dir = os.path.dirname(SCRIPT)
_HERE = os.path.dirname(os.path.abspath(__file__))
_R = os.path.abspath(os.path.join(_HERE, '..', '..', '..'))   # <repo>/_projects
for extra in (os.path.join(_R, '_shared'), _HERE, os.path.join(_R, '..', 'Calibration')):
    sys.path.insert(0, extra)
for rel in ('..', os.path.join('..', '_wheel_shaping_shared'), os.path.join('..', '..', '..', '_shared'),
            os.path.join('..', '..', '..', '..', 'Calibration')):
    sys.path.insert(0, os.path.abspath(os.path.join(task_dir, rel)))
for n in ('pybpod_rotaryencoder_module', 'pybpod_rotaryencoder_module.module_api', 'pybpod_rotaryencoder_module.module'):
    sys.modules[n] = types.ModuleType(n)
class RotaryEncoder: COM_SETZEROPOS = ord('Z'); COM_ENABLE_ALLTHRESHOLDS = ord('E')
sys.modules['pybpod_rotaryencoder_module.module'].RotaryEncoder = RotaryEncoder
sys.modules['pybpod_rotaryencoder_module.module_api'].RotaryEncoderModule = object

import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.show = lambda *a, **k: None; matplotlib.use = lambda *a, **k: None
import rotary_setup
import bpod_trial_helpers as _bth
import session_csv_parser as _scp
if os.environ.get('SOAK_DAY_WATER_UL'):   # pretend this much was consumed in earlier sessions today
    _ul = float(os.environ['SOAK_DAY_WATER_UL'])
    _scp.water_consumed_today = lambda *a, **k: (_ul, ['SOAK-EARLIER'])
if os.environ.get('SOAK_PREV_DAY'):   # 'YYYYMMDD,withheld,turns': pretend this was the previous Stage 2 day
    _d, _w, _t = os.environ['SOAK_PREV_DAY'].split(',')
    _scp.previous_day_withheld = lambda *a, **k: (_d, int(_w), int(_t), ['SOAK-PREV'])
PC_NAMES.update({v: k for k, v in _bth.PC_EVENT_CODES.items()})
class _Mod(object):
    serial_port = 1; name = 'RotaryEncoder1'
    def load_message(self, msg, msg_id=None):
        assert len(msg) <= 3, msg; return msg_id or 1
rotary_setup.connect_rotary = lambda bpod, usb_port=None: (ROT, _Mod())

DOT = {'visible': False, 'log': []}
class FakeDot(object):
    def show(self): pass
    def clear(self): DOT['visible'] = False
    def pump(self): time.sleep(0.004)                 # ~ a real repaint
    def get_screen_width_px(self): return 2048
    def set_deg_to_px_gain(self, g): pass
    def set_position_deg(self, p): DOT['visible'] = True
    def flash_white(self, s): pass
    def close(self): pass
import dot_display
if os.environ.get('SOAK_REAL_DOT') != '1':
    dot_display.create_dot_display = lambda **k: FakeDot()
else:
    print('SOAK: using the REAL dot display (rig stimulus screen)', flush=True)
import camera_recorder
LAST_SIDE = [None]
class FakeCam(object):
    max_read_ms = 0.0; max_write_ms = 0.0; backend = 'FAKE'
    start_time_s = 0.0; output_path = 'session_video.avi'; fps = 30.0
    def __init__(self, t0, **k): self.t0 = t0; self.ts = []; self.stop = False
    def start(self):
        def loop():
            while not self.stop: self.ts.append(time.time() - self.t0); time.sleep(1 / 21.0)
        threading.Thread(target=loop, daemon=True).start()
    def pump(self): pass
    def show_snippet(self, s): pass
    def close(self):
        self.stop = True
        with open('session_video_timestamps.csv', 'w') as f:
            f.write('frame_index,timestamp_s\n')
            for i, t in enumerate(self.ts): f.write('%d,%.6f\n' % (i, t))
    def compute_realized_fps(self): return None
camera_recorder.CameraRecorder = FakeCam
import wheel_shaping_plots
class FakePlots(object):
    def __init__(self, *a, **k): pass
    def add_trial(self, *a, **k): pass
    def __getattr__(self, n): return lambda *a, **k: None
wheel_shaping_plots.WheelShapingPlots = FakePlots
import session_state
class FakeState(object):
    def __init__(self, p, s, defaults):
        self.d = dict(defaults)
        self.path = os.environ.get('SOAK_STATE_FILE')
        if self.path and os.path.exists(self.path):
            with open(self.path) as f: self.d.update(json.load(f))
        if os.environ.get('SOAK_STATE_JSON'): self.d.update(json.loads(os.environ['SOAK_STATE_JSON']))
        FAKE_STATE.append(self)
    data = property(lambda self: self.d)
    def get(self, k): return self.d[k]   # KeyError on a missing key, like the real StageState
    def set(self, k, v): self.d[k] = v
    def save(self):
        if self.path:
            with open(self.path, 'w') as f: json.dump(self.d, f, indent=1, sort_keys=True)
session_state.StageState = FakeState
FAKE_STATE = []
import json
import session_csv_parser
session_csv_parser.compute_remaining_quota = lambda *a, **k: (
    N_TRIALS if k.get('full_max_trials') is not None else None,
    (N_TRIALS * 3.0 if k.get('full_cap_s') is not None else None))
# the REAL struct export runs (a stub here is how its 3-field WHEEL_POS crash went unnoticed)
try:
    import liquid_calibration; liquid_calibration.get_reward_duration_s = lambda ul, **k: 0.05
except ImportError: pass
hs = types.ModuleType('hifi_setup')
class FakeHifi(object):
    """ Slots like the real module: load() stages a waveform, push() makes staged ones live,
    play(slot) plays the live one. Raises on two threads at once, on playing a slot nothing was
    pushed to, and on playing one waveform twice (a stale replay of an earlier trial's sound). """
    UPLOAD_S = float(os.environ.get('SOAK_HIFI_UPLOAD_S', '0.12'))   # ~a 5.75s 44.1 kHz stereo wave
    def __init__(self, sampling_rate=192000):
        self.sampling_rate = sampling_rate; self.busy = False; self.gen = 0
        self.staged = {}; self.live = {}; self.played = set(); self.uploads = 0
    def _use(self, what):
        assert not self.busy, 'HiFi used by two threads at once (%s)' % what
        self.busy = True
    def load(self, slot, *a):
        self._use('load'); time.sleep(self.UPLOAD_S)
        self.gen += 1; self.staged[slot] = self.gen; self.uploads += 1; self.busy = False
    def push(self):
        self._use('push'); self.live.update(self.staged); self.staged = {}; self.busy = False
    def play(self, slot):
        self._use('play')
        g = self.live.get(slot)
        assert g is not None, 'play(%d): nothing pushed to that slot' % slot
        assert g not in self.played, 'play(%d): replaying an already-played waveform' % slot
        self.played.add(g); self.busy = False
    def close(self): pass
hs.connect_hifi = lambda b, usb_port=None, sampling_rate=None: FakeHifi(sampling_rate or 192000)
hs.build_stop_trigger = lambda b: (2, 'Serial2')
hs.build_play_trigger = lambda b, sound_index=0: (1, 'Serial2')
hs.compute_calibrated_amplitudes = lambda *a: (1.0, 1.0); sys.modules['hifi_setup'] = hs
import numpy as _np
gs = types.ModuleType('pybpod_hifi_module.utils.generate_sound')
gs.pure_tone = lambda dur, freq, sr, *a, **k: _np.zeros(max(1, int(round(dur * sr))))
for n in ('pybpod_hifi_module', 'pybpod_hifi_module.utils'): sys.modules[n] = types.ModuleType(n)
sys.modules['pybpod_hifi_module.utils.generate_sound'] = gs

TMP = tempfile.mkdtemp(prefix='soak_'); SESS = os.path.join(TMP, 'sessions', 'SOAK')
os.makedirs(SESS); os.chdir(SESS)
import gc
GC = {'n': 0, 'max_ms': 0.0, 'gen2': [], '_t': None}
def _gc_cb(phase, info):
    if phase == 'start': GC['_t'] = time.perf_counter()
    else:
        d = (time.perf_counter() - GC['_t']) * 1e3; GC['n'] += 1; GC['max_ms'] = max(GC['max_ms'], d)
        if info.get('generation') == 2: GC['gen2'].append(round(d, 1))
gc.callbacks.append(_gc_cb)
t_start = time.time()
try:
    runpy.run_path(SCRIPT, run_name='__main__')
except SystemExit:
    pass
print('GC: %d collections, max %.1f ms, gen2 pauses (ms): %s' % (GC['n'], GC['max_ms'], GC['gen2'][:20]))
print('ran %.1fs; final state pairs: %s' % (time.time() - t_start, dict(sorted(STATS['outcomes'].items(), key=lambda x: -x[1])[:6])))
sys.path.insert(0, os.path.join(_R, 'AuditoryEvidenceAccum', 'records'))
import audit_session
fails = audit_session.print_report(SESS)
print('struct exported:', os.path.exists(os.path.join(SESS, 'SOAK_struct.json')))
evs = set(r[5] for r in ROWS if r[0] == 'EVENT' and len(r) > 5)
print('PC event names in CSV:', sorted(e for e in evs if e.startswith('PC_') or e.startswith('SoftCode')))
print('SOAK_FOLDER', SESS)
if FAKE_STATE: print('SAVED recent_directions:', ''.join(FAKE_STATE[-1].d.get('recent_directions') or []))
