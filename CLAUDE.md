# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

This is the **pybpod** super-repo: a thin umbrella project that pulls together ~17 independent git
submodules (each its own PyPI package, with its own `setup.py`, tests, and GitHub repo) into one
installable application. PyBpod itself is a GUI application (built on PyForms/Qt) for controlling
Bpod behavioral-experiment hardware from Sanworks, maintained by the Champalimaud Foundation.

The top-level repo contains almost no application code of its own — just glue: `utils/install.py`,
`utils/deploy-pypi.py`, `user_settings.py`, and the submodule pointers in `.gitmodules` /
`base/repositories.yml`.

### Submodules are not checked out by default

`base/`, `libraries/`, and `plugins/` are git submodules. If they appear as **empty directories**,
they have not been initialized — this is the normal state right after a plain `git clone`. Before
expecting to find real source code under those paths, run:

```bash
git submodule update --init --recursive
```

Do not assume a submodule is "missing" or "broken" just because its folder is empty — check
submodule init status first (`git submodule status`, entries prefixed with `-` are uninitialized).

## Repository layout

- `base/` — the three core packages that make up PyBpod itself:
  - `pybpod` (this repo's own package, defined by `base/pybpod/setup.py`) — the top-level PyPI
    package and `start-pybpod` console-script entry point (`pybpodgui_plugin.__main__:start`).
  - `pybpod-api` — low-level Python API / driver for talking to the Bpod device over serial.
  - `pybpod-gui-api` — GUI-facing API layer between `pybpod-api` and the GUI plugin.
  - `pybpod-gui-plugin` — the main PyForms-based GUI application (Setups, Subjects, Boards, running
    protocols).
- `libraries/` — shared/forked dependencies vendored as submodules rather than plain pip deps:
  `pyforms-gui`, `pyforms-generic-editor`, `logging-bootstrap`, `safe-collaborative-architecture`.
- `plugins/` — optional GUI plugins that extend `pybpod-gui-plugin` via the generic-editor plugin
  system (each is independently versioned and pip-installable): session history, timeline,
  trial-timeline, stmdiagram, emulator, waveplayer, soundcard, rotaryencoder, alyx, and a terminal
  plugin (`pge-plugin-terminal`).
- `docs/` — Sphinx documentation source, published to Read the Docs
  (https://pybpod.readthedocs.io/).
- `utils/` — dev/release tooling (see Commands below).

Because every submodule is its own package with its own version pin, cross-cutting changes (e.g. a
new feature touching both `pybpod-api` and `pybpod-gui-plugin`) require editing multiple submodules
and bumping their versions independently — check `CHANGELOG.rst` for the shared-versioning history
(PyBpod's own version has been kept in sync with `pybpod-api`, `pybpod-gui-api`, and
`pybpod-gui-plugin` since v1.8.0).

## Local project data (`_projects/`)

Actual PyBpod projects (real experiment data, not code) for this machine live outside the repo, as a
sibling directory: `C:\Users\2P-Behav\Documents\_projects\`. Each subfolder is one PyBpod GUI project
— e.g. `_projects/AuditoryEvidenceAccum/` — containing a `<ProjectName>.json` descriptor plus
`boards/`, `experiments/`, `subjects/`, `tasks/`, and `users/` folders that the GUI reads/writes when
a project is open. This directory is not part of the `pybpod` git repo and should not be assumed to
exist on other machines/checkouts; treat it as runtime data, not source, when reasoning about the
codebase.

## Commands

Development environment (Python 3.6, conda-based):

```bash
conda create -n pybpod-environment python=3.6
activate pybpod-environment   # Windows; use `conda activate` on macOS/Linux
git submodule update --init --recursive
python utils/install.py       # pip install -e each submodule, then generate user_settings.py
start-pybpod                  # run the application
```

Confirmed working on this machine's `pybpod-environment` conda env (Python 3.6.13) after
`git submodule update --init --recursive` + `python utils/install.py`: `start-pybpod` launches the
Qt GUI and stays resident in its event loop. A one-line `WARNING ... Plugins path was not defined by
user` on startup is expected/benign, not a failure.

`utils/install.py` does two things:
1. `pip install -e` every folder listed in `SUBMODULES_FOLDERS` (all of `base/`, `libraries/`,
   `plugins/`) so the whole stack is editable in-place.
2. Writes `user_settings.py` (if it doesn't already exist) with `GENERIC_EDITOR_PLUGINS_LIST` set to
   the default plugin set (`pybpodgui_plugin`, `pybpodgui_plugin_timeline`,
   `pybpodgui_plugin_session_history`). Delete/edit `user_settings.py` to change which plugins load.

There are OS-specific conda environment files under `utils/` (`environment-windows-10.yml`,
`environment-ubuntu-17.10.yml`, `environment-macOSx.yml`) with pinned native deps (Qt, PyQt, HDF5,
etc.) — use the one matching the dev machine's OS instead of a fresh `conda create` if native builds
are an issue.

There is no build/lint/test tooling at the top-level repo (no top-level `pytest.ini`, `tox.ini`, or
CI test workflow — `.github/` only has issue templates). Tests, if any, live inside individual
submodules once checked out; run them from within that submodule's own directory using whatever
framework it defines.

### Versioning / release

- `base/pybpod/.bumpversion.cfg` drives version bumps for the top-level `pybpod` package via
  `bump2version`/`bumpversion`, run from `base/pybpod/`. It keeps `setup.py`,
  `../../docs/source/conf.py`, and `../../README.md` in sync.
- `utils/deploy-pypi.py` walks `libraries/`, `base/`, and `plugins/`, compares each submodule's local
  `setup.py --version` against the latest release on PyPI, and runs `sdist bdist_wheel` +
  `twine upload` for any that are ahead. This is a real publish action — do not run it without the
  user's explicit intent to release.

## Hardware module notes

Custom Bpod hardware modules (e.g. `pybpod-gui-plugin-soundcard`, `pybpod-gui-plugin-hifi`,
`pybpod-gui-plugin-rotaryencoder`) follow a consistent two-file pattern: `module_api.py` (direct
USB connection to the module for hardware-facing calls) + `module.py` (a `BpodModule` subclass —
the state-machine-relay side that self-announces to Bpod, e.g. `HiFi1`, `RotaryEncoder1`).
`plugins/pybpod-gui-plugin-hifi/` (package `pybpod_hifi_module`) is a from-scratch driver added
this way for the Sanworks HiFi HD audio module (no prior Python support anywhere); it's currently
a local uncommitted package, not yet its own git submodule/remote like the other `plugins/`
entries — a deliberate open decision, not an oversight.

Durable facts/gotchas about this module layer and Bpod hardware in general:

- A module's own USB connection is a **separate physical connection** from the Bpod state
  machine's own COM port (Serial1-5 relay) — nothing about a module's own USB port is ever
  reported over that relay, so it's always either hardcoded or auto-discovered by handshake
  probing (e.g. `RotaryEncoderModule.discover()`, `HiFiModule.discover()`), never read off the
  connected `Bpod` object.
- Two separate, easy-to-conflate registration mechanisms in `user_settings.py`:
  `PYBPOD_API_MODULES` (module *classification* — must include a module's package name for
  `BpodModules.create_module()` to instantiate its custom class instead of a generic `BpodModule`)
  vs. `GENERIC_EDITOR_PLUGINS_LIST` (GUI *menu*/plugin registration, handled by
  `pyforms-generic-editor`). A module package usually needs to be in both to work end-to-end in
  the GUI.
- **A task's own `PYBPOD_API_MODULES` may silently not apply to it.** Each session folder under
  `_projects/.../experiments/.../setups/.../sessions/<timestamp>/` gets its own auto-generated
  `user_settings.py` (board/serial/session identity only — no `PYBPOD_API_MODULES` key at all),
  and `board_com.py`'s `run_task()` launches the task as `subprocess.Popen(['python',
  task.filepath], cwd=self._running_session.path, ...)` — i.e. with that session folder as `cwd`,
  not the project root where the real `user_settings.py` (with `PYBPOD_API_MODULES` set) lives.
  `BpodModules.create_module()` only populates its custom-class registry once, the first time
  it's called, from whatever `conf.PYBPOD_API_MODULES` resolves to at that moment — if that's the
  session-local file (empty list), every custom module for the rest of that process resolves to a
  plain `BpodModule`, not its intended subclass, even though the *project's* `user_settings.py`
  looks correctly configured. Confirmed on hardware: `rotary_bpod_module.create_resetpositions_trigger()`
  raised `AttributeError: 'BpodModule' object has no attribute ...` for exactly this reason, while
  `hifi_bpod_module.load_message(...)` right next to it worked fine — because `load_message()` is
  defined on the `BpodModule` *base* class itself, so it works regardless of classification, but
  any subclass-only convenience method (like `RotaryEncoder.create_resetpositions_trigger()`)
  won't exist on the resolved instance. **Don't assume a module resolved via
  `next(m for m in my_bpod.modules if ...)` is actually its custom subclass inside a task script** —
  prefer building any subclass-specific serial message manually via the always-available
  `load_message()` plus the subclass's own command-byte class constants (importable directly from
  the class, no instance/classification needed), e.g.
  `rotary_bpod_module.load_message([RotaryEncoder.COM_SETZEROPOS, RotaryEncoder.COM_ENABLE_ALLTHRESHOLDS])`
  instead of `rotary_bpod_module.create_resetpositions_trigger()`.
- The rotary encoder board resets when a new serial connection opens and needs ~2s before it
  reliably responds to a handshake (`pybpod-gui-plugin-rotaryencoder/module_api.py`'s
  `BOOT_DELAY_S`/`HANDSHAKE_RETRIES`).
- Rotary encoder firmware disables a threshold the instant it fires, and crossing any *one*
  threshold disarms the *whole* configured set (not just that one) until
  `enable_all_thresholds()` is called again — see `module.py`'s
  `RotaryEncoder.create_resetpositions_trigger()`, which bundles `SETZEROPOS` +
  `ENABLE_ALLTHRESHOLDS` for exactly this reason.
- The rotary encoder's own firmware threshold-crossing *events* have shown an unexplained drift
  from what `current_position()` itself reports (which has been reliable and internally
  consistent) — don't depend on the firmware comparator for anything precision-sensitive; poll
  `current_position()` from Python instead.
- **`states_durations` contains an entry for every state *defined* in a state machine, not just
  ones actually visited** — Bpod logs a synthetic `(nan, nan)` duration for states never entered.
  `'SomeState' in trial.states_durations` is therefore **always true** regardless of whether that
  state actually ran; check the entry's start timestamp is a real float instead
  (`not math.isnan(durations[-1][0])`), or use `Trial.get_all_timestamps_by_event()`.
- **`Trial.events_occurrences` is a `list` of `EventOccurrence` objects, not a dict** —
  `.events_occurrences.items()` raises `AttributeError`. Use
  `Trial.get_all_timestamps_by_event()` (in `pybpod-api/pybpodapi/com/messaging/trial.py`) to get
  a `{event_name: [timestamps]}` dict instead.
- **Unbounded `thread.join()` around a `run_state_machine()` call risks hanging Kill.**
  `run_state_machine()`'s kill handling (`bpod_base.py`) calls `exit(0)` **from inside** the
  function once it reads a `kill` command off stdin, unwinding through any `try/finally` wrapped
  around the call. If that `finally` does an unbounded `join()` on a background thread that also
  talks to hardware and is ever mid-stall, Kill will appear to hang indefinitely even though the
  kill command was received and processed. Give such a join a bounded `timeout=`.
- **Board console output** (`print()` inside a running protocol) only appears if the relevant
  Board's **Console** window has been explicitly opened (and stays open) in the GUI —
  `pybpodgui_api/models/board/board_com.py`'s `log2board()` silently drops output otherwise.
- **Known hazard, still unexplained**: triggering a custom Bpod-module output (a message
  registered via `Bpod.load_serial_message`, fired through `output_actions` on a `SerialN`
  channel) a *second* time within the same trial has reliably hung `run_state_machine()`
  indefinitely in testing; root cause unknown. Work around it (e.g. bake multiple events into one
  waveform triggered once) rather than firing a module output more than once per trial. This is
  specifically about relay-triggered *outputs* to a module — distinct from a module's own *input*
  events (e.g. threshold-crossing) driving `state_change_conditions`, which has not shown this
  hang.
- Redrawing/requerying a UI element off a growing `session.data` table on every timer tick is
  O(session-length) per tick and will make long sessions unresponsive (see
  `pybpod-gui-plugin-rotaryencoder`'s `wheel_position_plot.py`, added as a "Wheel Position"
  right-click option on session tree nodes, mirroring `pybpodgui_plugin_trial_timeline`'s
  session-treenode files). Slice only new rows (`iloc[rows_read:]`) instead of re-querying
  everything, and use persistent, updatable plot artists instead of clear-and-replot every tick —
  same "only touch what changed" principle on both the data and rendering side.

## Poisson-clicks / wheel-turn evidence-accumulation task

Building toward a wheel-turn analog of the Brunton-style Poisson-clicks 2AFC task — see
`_projects/mouse_auditory_accumulation_paradigm.md` for the original design and its `## 13.
Implementation amendments` section for how the built version deliberately diverges from it
(wheel-turn choice + single valve, not lick spouts on retractable arms).

**Two code locations, split by how reusable each piece is:**
- `_projects/_shared/` — protocol-*agnostic* Bpod trial-loop plumbing, usable by any task in any
  GUI project (a sibling of every project, not nested inside one — deliberately, since "used
  across protocols during training" was the stated goal, not just within `Tests/`):
  `bpod_trial_helpers.py` (`TrialRunner` — background WHEEL_POS-publishing thread, bounded-join
  `run_trial_state_machine()`, hold-to-init wait — plus standalone `was_visited()`),
  `rotary_setup.py` / `hifi_setup.py` (module connection boilerplate, the rotary reset-trigger and
  HiFi STOP_ALL relay-message builders, `set_and_enable_thresholds()`). Extracted after
  `wait_for_held_steady()`/`trial_poll_loop()`/`run_trial_state_machine()`/`was_visited()` were
  found duplicated near-verbatim across `wheel_turn_reward.py` and the two `hifi_*` scripts below
  (with `wheel_turn_reward.py`'s copy already quietly drifted out of sync, missing a later Kill-race
  fix) — **check here before writing a new copy of trial-loop/module-connection boilerplate for
  any future protocol.** State-machine *construction* (which states, which transitions) is
  deliberately not here — that stays protocol-specific, in the task script itself.
- `_projects/Tests/tasks/` — the protocols themselves:
  - `poisson_clicks_test/` — shared *within this paradigm only* (not generic enough for
    `_shared/`): `click_train.py` (difficulty grid, per-side floored-Poisson click generation,
    waveform assembly for the HiFi module) and `live_plots.py` (`LiveBenchPlots`, one combined
    matplotlib window: click raster; psychometric/rewarded-trial-lick-raster/outcome-tally row;
    full-session lick timeline), plus `validate_click_train.py` (offline stats validation, no
    hardware needed).
  - `hifi_singleside_test/`, `hifi_alternating_easy_test/` — actual GUI-runnable protocols (proper
    task folders: matching `.py`/`.json`/`__init__.py`) that import both the above and
    `_shared/` from their respective sibling/cross-project locations rather than duplicating them.
  - `wheel_turn_reward.py` still has its own local, now out-of-sync copies of the trial-loop
    helpers — not yet migrated to `_shared/` (flagged as a clear next candidate, not an oversight).

Durable facts/gotchas from building this:

- **A GUI task subprocess can cleanly import sibling modules from *another* task folder** via
  `sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '<folder>'))`
  at the top of the script — confirmed against `board_com.py`'s actual `run_task()`, which launches
  `subprocess.Popen(['python', os.path.abspath(task.filepath)], cwd=self._running_session.path,
  ...)`. Because that's a genuine subprocess invocation of the script's own absolute path,
  `__file__` always resolves to the task's real location regardless of the subprocess's `cwd`
  (which is the *session* folder, not the task folder) — this is how the shared
  `click_train.py`/`live_plots.py` above stay a single copy instead of being duplicated into every
  task folder that uses them.
- **Floored-Poisson click-train rate correction**: generating inter-click intervals as
  `max(Exponential(1/r), floor)` (clip up to a refractory floor) always undershoots the *realized*
  rate relative to the nominal `r` fed in, because clipping only ever lengthens intervals. Closed
  form: `realized_rate(r) = 1 / (floor + (1/r)*e^(-r*floor))`, strictly monotonic in `r` (derived
  from `E[max(X,floor)]` for `X~Exponential(r)` via the memoryless property) — so solving it
  backwards (a one-time root-find per target rate, e.g. `scipy.optimize.brentq`) gives the
  "attempt rate" that actually realizes a given nominal rate after flooring. Worth remembering for
  any future refractory-limited point-process generator, not just this task.
- **A Bpod state with no `state_change_conditions` tied to a given event silently ignores that
  event even if the hardware fires it** — Bpod still logs it in the trial's raw event list, it
  just never triggers a transition. Compare event timestamps against the state's own known
  `state_timer` window to catch input that arrived while nothing was listening for it, rather than
  trusting the choice-mapping logic alone.
- **Don't gate a reward/scoring decision on a background-thread timer re-arming hardware state**
  (e.g. re-enabling rotary thresholds once elapsed time crosses a boundary) — a background thread
  can't preempt a blocking `run_state_machine()` call, so any such timer has real slop (confirmed
  ~0.1s), enough for stale pre-choice wheel drift to immediately re-trigger the instant it's
  re-armed and score a trial off drift rather than a genuine choice. A background poll thread is
  fine for something purely informational to a human (e.g. a print cue); anything that gates a
  reward should instead be a native Bpod `output_actions` entry fired on the real state transition
  — exact lockstep with hardware, no Python timing involved.
- **A background thread can never signal into an already-running `run_state_machine()` call** —
  confirmed via `bpod_com_protocol.py`: `manual_override`/`trigger_output`/`trigger_softcode` all
  write the *same* physical Bpod USB connection that call's own blocking loop is reading, so a
  second thread calling any of them concurrently is genuinely unsynchronized access, not a style
  choice. The only thread-safe channel into a running trial (`self.stdin`, polled once per loop
  iteration inside `run_state_machine()` itself) is wired to the GUI's own subprocess management
  for Stop/Kill, not usable by a same-process background thread. Consequence: anything a
  background thread alone can detect (a Python-computed condition with no native Bpod event behind
  it) can only be *acted on* once the current call returns — never mid-call.
- **`Port1In` (a lick sensor) has no "read current state" API outside a running state machine** —
  checked `bpod_base.py`/`bpod_com_protocol.py`: `manual_override`/`trigger_input` only *set*
  outputs; a digital input is only observable while some state machine is actively listening via
  `state_change_conditions`. Rotary encoder *threshold-crossing events* are the same way (need a
  running machine *and* `enable_evt_transmission()`), but the rotary's own `current_position()` is
  independently pollable from Python at any time via its separate USB connection, regardless of
  Bpod. This is why this task's cue-period abort detection could go **fully native** once a second
  pair of rotary thresholds (at the abort tolerance) was added: `CuePeriod`'s own
  `state_change_conditions` listen for both `Port1In` (`-> EarlyLick`) and the new threshold
  crossings (`-> WheelAbort`) directly, giving the wheel-abort check the same instant, in-band
  reaction as the lick check — no background thread involved in that decision at all anymore (an
  earlier version used one, flagging a Python-only wheel-abort_event for the main thread to check
  after the call returned, which meant a `WheelAbort` couldn't stop cue playback until the whole
  cue duration had already elapsed).
- **Hold-to-init still needs the hybrid (Bpod-chunk + background-thread) pattern, unlike
  wheel-abort, because the wheel-steadiness check itself isn't expressible as a native Bpod event**
  — it's "hasn't moved more than X° at any point in the trailing N-second window," a continuous
  rolling-window computation Bpod's own threshold-crossing events can't reproduce (only "did a
  crossing happen," not "stayed within bounds the whole time"). `TrialRunner.wait_for_held_steady()`
  (`_shared/bpod_trial_helpers.py`) runs that unchanged Python algorithm in a background thread,
  concurrently with the foreground repeatedly sending a short `HoldCheck`/`LickDuringHold` Bpod
  machine to gate on licking too — looping until both conditions hold at once, since (per the
  bullet above) neither side can cancel or signal the other mid-call.
- **`Consumption`/`ErrorConsumption` are lick-detection windows lumped together with whichever ITI
  follows, not a separate fixed duration stacked before it.** After `Reward`, `Consumption` listens
  for `Port1In` and exits into `ITI` the instant a lick occurs, or once the full
  `VAR_CONSUMPTION_WINDOW_S` elapses if it never does; the mirrored `ErrorConsumption` (entered
  instead of going straight from `WheelDotPeriod` to `IncorrectITI`) does the same, leading into
  `IncorrectITI` either way. Total post-outcome time is therefore variable trial-to-trial, not
  fixed. Whether the animal licked at all during that window (`consumption_licked`, logged as
  `CONSUMPTION_LICKED`) is a second, independent disengagement signal in
  `trial_scheduler.should_stop_session()` alongside the original no-response-to-cue one — a mouse
  that keeps turning the wheel but stops checking for/consuming the outcome can still trip the
  circuit-breaker.
- **A rate computed as "fraction of the last N trials" silently degrades to pure noise once fewer
  than N trials exist** — `history.recent(n_lookback)` returns however many trials are actually
  available if there are fewer than `n_lookback`, so a rate check with no minimum-sample gate can
  fire off a single early data point (e.g. trial 1's lone no-response/no-lick trial = "100% rate").
  Confirmed on hardware: a session's circuit-breaker fired after trial 1 for exactly this reason.
  `trial_scheduler.should_stop_session()`'s rate checks (both the no-response one and the newer
  no-consumption-lick one) now require `len(history) >= window_size` before trusting the rate at
  all — mirrors the `has_full_window` gate `PerformanceMonitor`'s remedial-easy trigger already
  used. Worth remembering for any future "rate over a trailing window" check, not just this one.
- **`draw_side_debiased()`'s raw output can still produce long, structurally predictable
  `TRIAL_SIDE` runs purely by chance** — the `P_RIGHT_CLIP=(0.15,0.85)` bound only keeps the
  *target probability* from running away, it does nothing about the *realized sequence*
  occasionally drawing a long same-side run OR a long strict-alternation run (`R L R L R L R...`),
  either of which a mouse could learn to automate off instead of attending to the stimulus.
  Confirmed on hardware: a real session showed a 7-trial alternation and a 5-trial same-side run.
  `draw_side_debiased_capped()` wraps `draw_side_debiased()` and forces a flip whenever honoring
  the natural draw would extend either pattern past `MAX_SAME_SIDE_RUN`/`MAX_ALTERNATION_RUN`
  (3 trials each) — this is what the task actually calls now, not `draw_side_debiased()` directly.
- **A live per-trial plot and a live full-session plot need different time bases** — a panel that
  resets every trial (e.g. lick timing relative to that trial's own reward) can't just be
  concatenated across trials to get a full-session view, since each trial's own zero point is a
  different absolute time. A continuously-growing panel needs absolute session-elapsed time
  (matching how `WHEEL_POS` is logged, as `now - log_python_t0`) — add back whatever wall-clock
  offset was subtracted out.
- **Fatal interpreter crash** (`Fatal Python error: PyEval_RestoreThread: NULL tstate`) hit twice in
  `full_protocol_lookback_test.py`, mid-trial, inside the `WHEEL_POS`-publishing poll thread's
  `time.sleep()` (`bpod_trial_helpers.py`'s `_trial_poll_loop`) — a CPython-internal GIL/thread-state
  fault, not a catchable Python exception. **Root cause identified on the second occurrence**: the
  user was actively dragging/moving the live-plot window when it happened both times. The script was
  mixing **two independent native GUI toolkits on the same main thread** — `matplotlib.use('TkAgg')`
  (Tcl/Tk event loop for the live-plot window) alongside `DotDisplay`/`GaborDisplay`'s own PyQt5
  event loop (`dot.pump()`) — on top of a background `decision_thread` blocked in
  `run_state_machine()` with a nested `WHEEL_POS` poll thread. Dragging a Tk window floods Tcl/Tk's
  event loop with window-manager messages right as Qt's event loop and the background threads are
  also active on/around the same thread — two competing native toolkits' C-level event pumps
  interleaving under concurrent GIL pressure is a known-fragile combination. **Fix**: switched
  `full_protocol_lookback_test.py` to `matplotlib.use('Qt5Agg')` instead of `TkAgg` — the live-plot
  window becomes a Qt window too, and matplotlib's first figure creation reuses the *same*
  `QApplication` `DotDisplay` already constructed (confirmed via smoke test: `id()`-identical
  instance) instead of spinning up a second, competing toolkit. One event loop for the whole process
  going forward. **Still needs hardware confirmation** that dragging the window no longer crashes it
  — not yet re-tested at the time of writing. **Known, not-yet-fixed residual risk**:
  `hifi_singleside_gabor_test.py`/`hifi_alternating_easy_gabor_test.py` have the identical
  `TkAgg` + `GaborDisplay` (PyQt5) mix and are exposed to the same crash class if their live-plot
  window is ever dragged during a run — left untouched per the don't-touch-already-tested-scripts
  discipline; backport the same one-line `Qt5Agg` fix to them once confirmed working here. Also
  found and fixed, while investigating, one real independent latent bug (kept regardless of the
  above): `TrialRunner.run_trial_state_machine()`'s poll-thread stop signal used to be a single
  instance-level `threading.Event()` reused across every trial's call — if a previous trial's poll
  thread ever failed to join within its 3s timeout (only warned about, not tracked), the next
  trial's `.clear()` would silently un-signal that stale thread too, letting it keep running
  indefinitely alongside the new one. Each call now gets its own fresh `Event`, scoped to that call
  only, and a stale thread (join timeout) is tracked in `self._stale_poll_threads` for visibility.
  `VAR_RENDER_HZ` was also lowered (60→30) in `full_protocol_lookback_test.py` — harmless, but a
  minor mitigation, not the fix.
- **`remedial_easy` (the performance-triggered easy block in `trial_scheduler.py`'s
  `PerformanceMonitor`) does NOT lock `TRIAL_SIDE`** — an earlier version of this session's work
  did add a side lock (`PerformanceMonitor.next_side()`, drawing the side once via the debiasing
  correction when the block started and holding it fixed for the whole block), but it was
  deliberately reverted: `TRIAL_SIDE` keeps drawing fresh every trial from the normal debiasing
  pipeline (`side_error_fraction`/`compute_p_right`/`recency_weighted_right_fraction`/
  `draw_side_debiased`, including its existing 85%/15% floor), remedial_easy included. What
  actually makes remedial trials easy is `next_difficulty()` forcing `DIFFICULTY_LEVEL` to `'AOS'`
  (fully one-sided click train, gamma=1.0) unconditionally for the whole block — not the
  `'AOS'`/`'G65'` 50/50 draw an even earlier version used, since `'G65'` (gamma=0.65) is still
  graded/ambiguous, not "all on one side." Traced from a real session's `log.txt`: a stretch where
  the logged `P_RIGHT_TARGET`/`RECENT_RIGHT_FRACTION` implied ~33% chance of 'R' yet 'R' kept
  landing anyway turned out to be the (now-reverted) side lock — `TRIAL_TYPE == 'remedial_easy'`
  for that whole stretch, with the live debiasing values still being computed/logged for display
  even though they weren't driving the (locked) side decision at the time. Don't reintroduce a
  side lock without re-checking this reasoning.

## Wheel-shaping training stages (Stage 1/2, `AuditoryEvidenceAccum`)

`training_protocol.md` (in `_projects/`, alongside `mouse_auditory_accumulation_paradigm.md`) is the
full staged training curriculum (Part 4: Stage 1 through Stage 8) for the wheel-turn/dot variant of
the evidence-accumulation task. Stage 1 ("Wheel unlocked, spout close, small movements rewarded")
and Stage 2 ("Threshold staircase, ITI growth, spout retraction") are built in
`_projects/AuditoryEvidenceAccum/tasks/` — the **real** project, not `Tests/`. `Tests/` stays what it
always was (bench-test/dry-run scripts, `full_protocol_lookback_test.py` etc.) and is untouched by
this work; `AuditoryEvidenceAccum` is where actual animal training happens going forward. Neither
stage has any auditory stimulus, correct-side concept, or error condition yet — those start at
Stage 3 — but the dot **is** coupled to the wheel from Stage 1 onward (no wheel-only, dot-less phase
anywhere in the curriculum).

- `AuditoryEvidenceAccum/tasks/_wheel_shaping_shared/` is shared *within these two stages only* —
  same "shared within this paradigm, not generic enough for `_shared/`" placement convention as
  `Tests/tasks/poisson_clicks_test/`. It deliberately does NOT import from `Tests/` (the real project
  shouldn't depend on the bench-test one) — `_style_axes()`/`_capped_figsize()` in
  `wheel_shaping_plots.py` are small, intentional duplicates of the identically-named helpers in
  `poisson_clicks_test/live_plots.py`, not a cross-project import.
- **Cross-session state (movement threshold staircase, ITI, wheel gain, spout-position reminder,
  the two-consecutive-qualifying-sessions advancement check) persists via a self-persisting JSON
  file** (`session_state.py`'s `StageState`) at
  `AuditoryEvidenceAccum/subjects/<VAR_SUBJECT_ID>/wheel_shaping_state.json` — read/written
  automatically by the task script itself, no manual data entry between sessions.
  `VAR_SUBJECT_ID` is an explicit constant at the top of each stage script (edit it before every
  session) — no task anywhere in this codebase auto-derives "which subject is this session for"
  from PyBpod's own session/experiment machinery, so this is the deliberate, lower-risk choice over
  guessing at that association. **Stage 1 and Stage 2 share one continuous state file per subject**
  (same filename, same path derivation) — Stage 2's own new keys (`threshold_fraction` etc.) default
  to Stage 1's known ending values the first time it runs for a given subject, rather than each
  stage starting its own separate record.
- `training_protocol.md`'s Stage 2 advancement criterion is primarily a **statistical** test
  (bimodal movement distribution + rewarded-turn velocity clearly separated from drift) that the doc
  describes conceptually but doesn't fully specify an algorithm for. Deliberately deferred (per
  explicit decision, not an oversight): `stage2_threshold_staircase.py` auto-checks only the simple,
  unambiguous numeric gates (trial count, ITI at its ceiling, direction ratio in-band, via
  `staircase.stage2_simple_gates_met()`) and prints an explicit reminder that the statistical test
  isn't implemented — advancement should stay a human judgment call (reviewing the movement raster)
  until there's real Stage 2 movement data to validate a concrete implementation against, rather than
  guessing at unspecified details blind.
- **A "detect outcome, then decide reward" need doesn't always require the two-Bpod-machine pattern**
  used elsewhere in this codebase for a superficially similar problem
  (`full_protocol_lookback_test.py`'s `cue_sma`-then-`sma`, needed because which side is rewarded
  there depends on that trial's own freshly-drawn `TRIAL_SIDE`, unknowable before the state machine
  is built). Stage 2's reward-withholding decision is different: which side (if either) is currently
  over-used is knowable from `DirectionRatioTracker` *before* the trial starts (it only depends on
  the last ~40 trials' history, not on anything this trial does), so a single state machine can bake
  the reward-or-withhold branching directly into its `state_change_conditions` up front — no second
  Bpod round-trip needed. Worth checking which situation actually applies before reaching for the
  two-machine pattern by default.
- **Stage 1's movement threshold grows ACROSS sessions (one fixed step per qualifying session), NOT
  Stage 2's within-session, per-trial `ThresholdStaircase`** — per explicit correction: Stage 1 isn't
  tracking moment-to-moment performance the way Stage 2 is, so `staircase.grow_stage1_threshold()`
  (same shape as `grow_iti()`) is used instead, starting ~5% of final and capped at Stage 2's own 20%
  starting point. Both stages still update the *same* persisted `threshold_fraction` — just via
  different mechanisms depending on which stage is currently active. Don't reach for
  `ThresholdStaircase` by default for a "should this grow over time" need — check whether the growth
  is meant to be *responsive* (per-trial, performance-driven, Stage 2's case) or just a *scheduled*
  step (per-session, Stage 1's case, same as ITI/gain/spout-position) first.
- **Prefer parsing the native Bpod `STATE`/`EVENT`/`TRIAL` rows directly out of a session CSV over
  adding new custom `VAL` registrations, wherever the native rows already have the answer** — per
  explicit instruction, for consistency across training and future experiment code. Confirmed
  directly against a real session CSV: every `STATE` row (`state_name;start;end;duration`, `nan` for
  a state never visited — this **is** `states_durations`/MATLAB Bpod's
  `SessionData.RawEvents.Trial{n}.States`, serialized as plain rows instead of an in-memory dict) and
  every `EVENT` row (raw hardware/software events including `Port1In` licks — the
  `get_all_timestamps_by_event()` equivalent) is already in the file; `VAL` rows are only the
  custom, script-level extras layered on top. `AuditoryEvidenceAccum/records/build_training_log.py`
  reconstructs this native per-trial structure straight from the CSV text (no live Bpod connection
  needed) and reads reward count/duration, L/R turn counts, and lick count from it directly — a small
  per-protocol config (`PROTOCOL_CONFIG` in that file) maps state names to meaning (which states
  count as rewarded/withheld/no-movement, and whether side comes from state-name suffixes or raw
  event names), so the *parsing technique* itself stays identical as new protocols with different
  state names get added; only that config needs to grow. Custom `VAL` registrations
  (`THRESHOLD_DEG`/`GAIN_MULT`/`DIRECTION_RATIO`/`TRIAL_TYPE`) are still the right call for anything
  genuinely absent from any state machine — Python-level scheduling decisions, not hardware states.
  **A trailing trial with no `STATE` rows at all means the session was stopped/killed mid-hold**,
  before that trial's Bpod machine ever ran — confirmed on a real bench-test log (its own `vals` had
  only `WHEEL_POS`/`TRIAL_START`, none of the outcome-time registrations). Don't assume the *last*
  trial in a parsed session is a genuine one; search backward for the last trial that actually has
  `STATE` data instead (`build_training_log.py`'s `_find_val_backward()`).
- **`AuditoryEvidenceAccum/records/`** holds the lab's record-keeping: `training_log.xlsx`
  (auto-generated by `build_training_log.py`, one sheet per animal — **never hand-edit this file**,
  it's fully regenerated from scratch every run; edit `animals_metadata.json` (DOB/sex/strain/
  baseline weight, one entry per animal) or `session_manual_entries.csv` (per-session hand-recorded
  weight, transcribed from the paper startup checklist) instead, then re-run the script) and
  `session_startup_checklist.xlsx` (a static, printable form template generated once by
  `build_startup_checklist.py` — animal ID, protocol run, equipment on/off checklist, pre-session
  checks, outcome summary; purely physical record-keeping, never auto-populated, since none of it has
  a digital source).

## Architecture notes

- The GUI plugin system (`pybpod-gui-plugin` + everything in `plugins/`) is built on the
  `pyforms-generic-editor` framework (in `libraries/`), which loads plugins by Python package name
  from `GENERIC_EDITOR_PLUGINS_LIST` in `user_settings.py`. Adding a new plugin submodule to the repo
  is not enough by itself — it must also be added to that list (and installed) to actually load.
  `SETTINGS_PRIORITY` in `user_settings.py` controls settings-file precedence when multiple
  `user_settings.py` files are discoverable (see `logging-bootstrap`/`pyforms` settings resolution).
- `pybpod-api` talks to the physical Bpod device over serial; `pybpod-gui-api` and
  `pybpod-gui-plugin` are the layers that expose that over the desktop GUI (built on `pyforms-gui`,
  a Qt-based forms framework). Hardware-facing changes typically belong in `pybpod-api`; UI/workflow
  changes belong in `pybpod-gui-plugin` or the relevant plugin under `plugins/`.
- Because packages are versioned and pinned against each other (see `base/pybpod/setup.py`'s
  `install_requires`), a change in one submodule generally isn't picked up by a pip-installed
  (non-editable) instance of another until versions are bumped and republished — this only doesn't
  matter in the `-e` editable-install dev setup produced by `utils/install.py`.
